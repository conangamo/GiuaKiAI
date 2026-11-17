"""Các hàm dùng chung cho train.py và evaluate_model.py."""
import torch
from tqdm import tqdm


def trim_batch(batch):
    """
    Cắt bớt phần pad dư thừa trong batch (padding bên phải).
    Kết quả toán học giữ nguyên (pad đã bị mask / ignore_index=-100) nhưng nhanh hơn nhiều
    vì độ dài đích trung bình chỉ ~10 token trong khi dataset pad cứng tới max_len.
    """
    src_len = max(1, int(batch["source_mask"].sum(dim=1).max()))
    tgt_len = max(1, int((batch["labels"] != -100).sum(dim=1).max()))
    return {
        "input_ids": batch["source_ids"][:, :src_len],
        "attention_mask": batch["source_mask"][:, :src_len],
        "labels": batch["labels"][:, :tgt_len],
    }


@torch.no_grad()
def compute_val_loss(model, dataloader, device):
    """Loss teacher-forcing trên dev, trung bình theo số token đích."""
    model.eval()
    total, n_tok = 0.0, 0
    for batch in dataloader:
        b = trim_batch(batch)
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    labels=b["labels"].to(device))
        n = int((b["labels"] != -100).sum())
        total += out.loss.item() * n
        n_tok += n
    return total / max(1, n_tok)


@torch.no_grad()
def generate_predictions(model, dataloader, tokenizer, device, max_new_length=64, num_beams=1,
                         desc="Evaluating", max_samples=None):
    """Sinh dự đoán cho toàn bộ dataloader. Trả về dict các list song song."""
    model.eval()
    res = {"sentences": [], "preds": [], "golds": [], "domains": []}
    for batch in tqdm(dataloader, desc=desc, leave=False):
        b = trim_batch(batch)
        gen_kwargs = dict(input_ids=b["input_ids"].to(device),
                          attention_mask=b["attention_mask"].to(device),
                          max_length=max_new_length, num_beams=num_beams)
        if num_beams > 1:
            gen_kwargs["early_stopping"] = True
        out = model.generate(**gen_kwargs)
        res["preds"].extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        res["golds"].extend(batch["raw_target"])
        res["sentences"].extend(batch["raw_sentence"])
        res["domains"].extend(batch["domain"])
        if max_samples and len(res["preds"]) >= max_samples:
            break
    return res

