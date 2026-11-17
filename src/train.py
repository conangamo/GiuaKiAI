import os
import sys
import json
import time
import random
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ABSADataset  # noqa: E402
from metrics import compute_metrics, POLARITIES  # noqa: E402
from engine import trim_batch, compute_val_loss, generate_predictions  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune mT5 cho Aspect-Based Sentiment Analysis")
    p.add_argument("--model_name_or_path", type=str, default="google/mt5-small")
    p.add_argument("--data_dir", type=str, default="data/multi_domain")
    p.add_argument("--task", type=str, default="uabsa", choices=["uabsa", "tasd"])
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_seq_len", type=int, default=128, help="độ dài tối đa phía nguồn (câu dài nhất ~136 token)")
    p.add_argument("--max_target_len", type=int, default=64, help="độ dài tối đa phía đích (tối đa thực tế 62)")
    p.add_argument("--max_eval_samples", type=int, default=0,
                   help="0 = đánh giá TOÀN BỘ dev mỗi epoch (mặc định). >0 = chỉ N câu đầu (nhanh nhưng kém tin cậy)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="checkpoints/run")
    p.add_argument("--select_metric", type=str, default="f1", choices=["f1", "macro_f1"],
                   help="chỉ số trên dev dùng để chọn checkpoint tốt nhất")
    p.add_argument("--fp16", action="store_true", default=False,
                   help="mT5 dễ NaN khi fp16 (đã gặp trên GTX 1650) nên mặc định tắt")
    # --- xử lý mất cân bằng nhãn (Bước 2) ---
    p.add_argument("--imbalance", type=str, default="none", choices=["none", "oversample", "weighted_loss", "both"],
                   help="none | oversample (nhân bản câu chứa nhãn hiếm) | weighted_loss (tăng trọng số token cảm xúc hiếm) | both")
    p.add_argument("--power", type=float, default=0.5,
                   help="độ mạnh cân bằng: trọng số lớp c = (n_lớp_đông_nhất / n_c) ** power. 0 = không cân bằng, 1 = cân bằng hoàn toàn")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------
# Mất cân bằng nhãn
# ----------------------------------------------------------------------------------------------
def count_train_polarity(dataset):
    """Đếm số cặp (aspect, polarity) duy nhất mỗi câu trong tập train, theo từng lớp."""
    cnt = Counter()
    for labs in dataset.raw_labels:
        for p in {(t[0].lower(), t[2].lower()) for t in labs if len(t) >= 3}:
            cnt[p[1]] += 1
    return cnt


def class_weights(counts, power):
    n_max = max(counts[c] for c in POLARITIES)
    return {c: (n_max / max(1, counts[c])) ** power for c in POLARITIES}


def build_oversampled_indices(dataset, weights, seed):
    """
    Mỗi câu được lặp lại round(w) lần với w = trọng số lớp hiếm nhất mà câu chứa (câu không có nhãn: w=1).
    Giữ nguyên toàn bộ câu gốc (không bỏ câu nào), chỉ nhân bản thêm.
    """
    indices = []
    for i, labs in enumerate(dataset.raw_labels):
        pols = {t[2].lower() for t in labs if len(t) >= 3}
        w = max([weights.get(p, 1.0) for p in pols], default=1.0)
        indices.extend([i] * max(1, int(round(w))))
    random.Random(seed).shuffle(indices)
    return indices


class IndexListSampler(Sampler):
    """Sampler lặp qua danh sách chỉ số (có lặp) và xáo trộn lại mỗi epoch theo seed."""
    def __init__(self, indices, seed):
        self.base, self.seed, self.epoch = list(indices), seed, 0

    def __iter__(self):
        idx = self.base[:]
        random.Random(self.seed + self.epoch).shuffle(idx)
        self.epoch += 1
        return iter(idx)

    def __len__(self):
        return len(self.base)


def polarity_token_weight_vector(tokenizer, weights, vocab_size, device):
    """
    Vector trọng số theo token cho cross-entropy. Chỉ token ĐẶC TRƯNG của từ cảm xúc được nhân trọng số:
      ' positive' (1 token), ' neutral' (1 token), và 'negative' (đứng sau token ' ' dùng chung nên chỉ gắn vào 'negative').
    """
    vec = torch.ones(vocab_size, device=device)
    marked = {}
    for pol in POLARITIES:
        toks = tokenizer.tokenize("(x, " + pol + ")")  # [' (', 'x', ',', ' positive', ')']
        distinctive = toks[3:-1][-1]                    # token cuối của từ cảm xúc
        tid = tokenizer.convert_tokens_to_ids(distinctive)
        vec[tid] = weights[pol]
        # Bổ sung token không có leading space nếu có trong vocab
        bare_id = tokenizer.convert_tokens_to_ids(pol)
        if bare_id is not None and bare_id != tokenizer.unk_token_id:
            vec[bare_id] = weights[pol]
        marked[pol] = (distinctive, tid, round(weights[pol], 3))
    return vec, marked


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ABSA mT5 | {args.model_name_or_path} | device={device} ===")
    print(f"data={args.data_dir} task={args.task} imbalance={args.imbalance} power={args.power} seed={args.seed}")
    print(f"batch={args.batch_size} x accum={args.gradient_accumulation_steps} "
          f"(effective {args.batch_size * args.gradient_accumulation_steps}) | epochs={args.epochs} | lr={args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

    kw = dict(task=args.task, lang=args.lang, max_len=args.max_seq_len, max_target_len=args.max_target_len)
    train_ds = ABSADataset(tokenizer, args.data_dir, "train", **kw)
    dev_ds = ABSADataset(tokenizer, args.data_dir, "dev", **kw)
    print(f"Train={len(train_ds)} | Dev={len(dev_ds)}")

    # ---- cân bằng nhãn ----
    train_counts = count_train_polarity(train_ds)
    weights = class_weights(train_counts, args.power) if args.imbalance != "none" else None
    print(f"Phân bố cặp trong train: {dict(train_counts)}"
          + (f" -> trọng số lớp {({k: round(v, 2) for k, v in weights.items()})}" if weights else ""))

    sampler = None
    if args.imbalance in ("oversample", "both"):
        idx = build_oversampled_indices(train_ds, weights, args.seed)
        sampler = IndexListSampler(idx, args.seed)
        print(f"Oversample: {len(train_ds)} -> {len(idx)} mẫu/epoch (x{len(idx) / len(train_ds):.2f})")

    tok_weight_vec = None
    if args.imbalance in ("weighted_loss", "both"):
        tok_weight_vec, marked = polarity_token_weight_vector(tokenizer, weights, model.config.vocab_size, device)
        print(f"Weighted loss: token đặc trưng -> {marked}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False)

    no_decay = ["bias", "layer_norm", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=args.lr)
    total_steps = max(1, (len(train_loader) // args.gradient_accumulation_steps) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "train_polarity_counts": dict(train_counts),
                   "class_weights": weights, "train_samples_per_epoch": len(train_loader) * args.batch_size},
                  f, indent=2, ensure_ascii=False)

    history, best_score, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n_steps = 0.0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            b = trim_batch(batch)
            ids, mask, labels = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if tok_weight_vec is None:
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                else:
                    dec_in = model.prepare_decoder_input_ids_from_labels(labels=labels)
                    logits = model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec_in).logits
                    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1),
                                           weight=tok_weight_vec, ignore_index=-100)
                loss_scaled = loss / args.gradient_accumulation_steps

            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            run_loss += loss.item()
            n_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix(train_loss=f"{run_loss / n_steps:.4f}")
            if args.dry_run and step >= 3:
                print(f"[dry-run] train_loss={run_loss / n_steps:.4f} - pipeline OK")
                return

        train_loss = run_loss / max(1, n_steps)
        val_loss = compute_val_loss(model, dev_loader, device)
        res = generate_predictions(model, dev_loader, tokenizer, device, desc="Dev eval",
                                   max_samples=args.max_eval_samples or None)
        rep = compute_metrics(res["preds"], res["golds"], task=args.task,
                              domains=res["domains"] if dev_ds.domains else None)

        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
               "dev_precision": rep["precision"], "dev_recall": rep["recall"], "dev_f1": rep["f1"],
               "dev_macro_f1": rep["macro_f1"], "dev_n_sentences": len(res["preds"]),
               "dev_f1_per_class": {c: rep["per_class"][c]["f1"] for c in POLARITIES},
               "elapsed_min": round((time.time() - t0) / 60, 1)}
        history.append(row)
        print(f"\n--- Epoch {epoch} --- train_loss={train_loss:.4f} val_loss={val_loss:.4f} | dev(n={row['dev_n_sentences']}) "
              f"P={rep['precision']} R={rep['recall']} F1={rep['f1']} macroF1={rep['macro_f1']} "
              f"| F1/lớp={row['dev_f1_per_class']}")

        score = rep["f1"] if args.select_metric == "f1" else rep["macro_f1"]
        if score > best_score:
            best_score = score
            best_dir = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            row["is_best"] = True
            print(f"--> lưu best_model ({args.select_metric}={score})")

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\nHoàn tất. Best dev {args.select_metric} = {best_score} | thời gian = {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
