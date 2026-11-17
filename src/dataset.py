import os
import ast
import warnings
from torch.utils.data import Dataset


def read_mabsa_file(file_path):
    """
    Đọc file định dạng M-ABSA. Mỗi dòng: sentence####[[aspect, category, polarity], ...]
    Trả về (sentences, labels). Dùng ast.literal_eval (an toàn, không thực thi code).
    Dòng lỗi định dạng sẽ được đếm và cảnh báo (không bỏ qua im lặng).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    sentences, labels = [], []
    n_bad = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("####")
            if len(parts) != 2:
                n_bad += 1
                continue
            sentence, raw_labels = parts[0].strip(), parts[1].strip()
            try:
                triplets = ast.literal_eval(raw_labels)
            except (ValueError, SyntaxError):
                n_bad += 1
                continue
            sentences.append(sentence)
            labels.append(triplets)

    if n_bad:
        warnings.warn(f"{file_path}: bỏ qua {n_bad} dòng lỗi định dạng")
    return sentences, labels


def read_domain_tags(file_path, expected_len):
    """Đọc file {split}_domains.txt nằm cạnh {split}.txt (nếu có). Không có -> None."""
    base = os.path.splitext(file_path)[0]
    tag_path = base + "_domains.txt"
    if not os.path.exists(tag_path):
        return None
    with open(tag_path, "r", encoding="utf-8") as f:
        tags = [ln.strip() for ln in f if ln.strip()]
    if len(tags) != expected_len:
        warnings.warn(f"{tag_path}: {len(tags)} tag nhưng có {expected_len} câu -> bỏ qua tag miền")
        return None
    return tags


def format_target(triplets, task="uabsa"):
    """
    Chuyển nhãn thành chuỗi đích cho mô hình seq2seq.
    - uabsa: (aspect, polarity); ...   (các cặp trùng trong cùng câu được gộp, giữ thứ tự xuất hiện)
    - tasd : (aspect, category, polarity); ...
    """
    if not triplets:
        return "None"

    if task == "uabsa":
        seen, pairs = set(), []
        for tri in triplets:
            if len(tri) < 3:
                continue
            key = (tri[0], tri[2])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(f"({tri[0]}, {tri[2]})")
        return "; ".join(pairs) if pairs else "None"

    elif task == "tasd":
        tri_str = [f"({tri[0]}, {tri[1]}, {tri[2]})" for tri in triplets if len(tri) >= 3]
        return "; ".join(tri_str) if tri_str else "None"

    raise ValueError(f"Unsupported task: {task}. Choose 'uabsa' or 'tasd'.")


def resolve_data_path(data_dir, split="train", lang="en", base_dir="M-ABSA/data"):
    """
    Tìm file dữ liệu linh hoạt:
      1. data_dir/{split}.txt                      (thư mục trực tiếp, vd data/multi_domain)
      2. data/{data_dir}/{split}.txt
      3. M-ABSA/data/{data_dir}/{split}.txt
      4. M-ABSA/data/{data_dir}/{lang}/{split}.txt (vd restaurant + en)
    """
    candidates = [
        os.path.join(data_dir, f"{split}.txt"),
        os.path.join("data", data_dir, f"{split}.txt"),
        os.path.join(base_dir, data_dir, f"{split}.txt"),
        os.path.join(base_dir, data_dir, lang, f"{split}.txt"),
        os.path.join("..", base_dir, data_dir, lang, f"{split}.txt"),
        os.path.join("data", data_dir, lang, f"{split}.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"Could not locate {split}.txt for data_dir='{data_dir}', lang='{lang}'. Checked paths:\n"
        + "\n".join(candidates)
    )


class ABSADataset(Dataset):
    """
    Dataset PyTorch cho ABSA dạng sinh chuỗi (mT5 / T5).
    max_len: độ dài tối đa phía nguồn; max_target_len: phía đích (mặc định = max_len).
    """

    def __init__(self, tokenizer, data_dir, split="train", task="uabsa", lang="en",
                 max_len=128, max_target_len=None):
        self.tokenizer = tokenizer
        self.task = task
        self.max_len = max_len
        self.max_target_len = max_target_len or max_len
        self.file_path = resolve_data_path(data_dir=data_dir, split=split, lang=lang)
        self.sentences, self.raw_labels = read_mabsa_file(self.file_path)
        self.domains = read_domain_tags(self.file_path, len(self.sentences))

        self.inputs = []
        self.targets = []
        self.n_truncated_source = 0
        self.n_truncated_target = 0
        self._build_examples()

    def _build_examples(self):
        for sent, triplets in zip(self.sentences, self.raw_labels):
            target_str = format_target(triplets, task=self.task)

            full_src = self.tokenizer(sent, add_special_tokens=True)["input_ids"]
            full_tgt = self.tokenizer(target_str, add_special_tokens=True)["input_ids"]
            self.n_truncated_source += int(len(full_src) > self.max_len)
            self.n_truncated_target += int(len(full_tgt) > self.max_target_len)

            self.inputs.append(self.tokenizer(
                sent, max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"))
            self.targets.append(self.tokenizer(
                target_str, max_length=self.max_target_len, padding="max_length",
                truncation=True, return_tensors="pt"))

        if self.n_truncated_source or self.n_truncated_target:
            warnings.warn(
                f"[{os.path.basename(self.file_path)}] bị cắt cụt: nguồn={self.n_truncated_source}, "
                f"đích={self.n_truncated_target} (max_len={self.max_len}, max_target_len={self.max_target_len})")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze(0)
        source_mask = self.inputs[index]["attention_mask"].squeeze(0)
        target_ids = self.targets[index]["input_ids"].squeeze(0)
        target_mask = self.targets[index]["attention_mask"].squeeze(0)

        # Pad trong labels phải là -100 để bị bỏ qua khi tính cross-entropy
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "labels": labels,
            "raw_sentence": self.sentences[index],
            "raw_target": format_target(self.raw_labels[index], task=self.task),
            "domain": self.domains[index] if self.domains else "all",
        }
