"""
Đánh giá một checkpoint trên TOÀN BỘ tập dev/test, báo cáo theo lớp cảm xúc và theo miền.

Ví dụ:
  python src/evaluate_model.py --model_path checkpoints/best_model --data_dir data/multi_domain --split test --run_name legacy_baseline
Kết quả lưu ở results/<run_name>_<split>.json và results/<run_name>_<split>_predictions.tsv
"""
import os
import sys
import json
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ABSADataset  # noqa: E402
from metrics import compute_metrics, format_report, parse_uabsa_string  # noqa: E402
from engine import generate_predictions  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Đánh giá mô hình ABSA trên dev/test")
    p.add_argument("--model_path", type=str, default="checkpoints/best_model")
    p.add_argument("--data_dir", type=str, default="data/multi_domain")
    p.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    p.add_argument("--task", type=str, default="uabsa", choices=["uabsa"])
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=128, help="độ dài tối đa phía nguồn")
    p.add_argument("--max_target_len", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--run_name", type=str, default=None, help="tên dùng để đặt file kết quả (mặc định: tên thư mục model)")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--num_examples_to_show", type=int, default=5)
    return p.parse_args()


def evaluate_checkpoint(model_path, data_dir, split, task="uabsa", lang="en", batch_size=16,
                        max_len=128, max_target_len=64, num_beams=1, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    ds = ABSADataset(tokenizer, data_dir=data_dir, split=split, task=task, lang=lang,
                     max_len=max_len, max_target_len=max_target_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    res = generate_predictions(model, loader, tokenizer, device, num_beams=num_beams,
                               desc=f"Evaluating {split}")
    domains = res["domains"] if ds.domains else None
    report = compute_metrics(res["preds"], res["golds"], task=task, domains=domains)
    return report, res


def save_outputs(report, res, results_dir, run_name, split, meta=None):
    os.makedirs(results_dir, exist_ok=True)
    out = dict(report)
    out["meta"] = meta or {}
    json_path = os.path.join(results_dir, f"{run_name}_{split}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    tsv_path = os.path.join(results_dir, f"{run_name}_{split}_predictions.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("domain\tsentence\tgold\tprediction\texact_match\n")
        for d, s, g, p in zip(res["domains"], res["sentences"], res["golds"], res["preds"]):
            ok = int(parse_uabsa_string(g) == parse_uabsa_string(p))
            f.write(f"{d}\t{s}\t{g}\t{p}\t{ok}\n")
    return json_path, tsv_path


def main():
    args = parse_args()
    run_name = args.run_name or os.path.basename(os.path.normpath(args.model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model_path} | Data: {args.data_dir} ({args.split}) | Device: {device}")

    report, res = evaluate_checkpoint(args.model_path, args.data_dir, args.split, args.task, args.lang,
                                      args.batch_size, args.max_len, args.max_target_len, args.num_beams, device)
    print(format_report(report, title=f"KẾT QUẢ {args.split.upper()} - {run_name}"))

    meta = {"model_path": args.model_path, "data_dir": args.data_dir, "split": args.split,
            "num_beams": args.num_beams}
    jp, tp = save_outputs(report, res, args.results_dir, run_name, args.split, meta)
    print(f"\nĐã lưu: {jp}\n        {tp}")

    print(f"\n--- {args.num_examples_to_show} mẫu dự đoán đầu tiên ---")
    for i in range(min(args.num_examples_to_show, len(res["preds"]))):
        print(f"[{i+1}] ({res['domains'][i]}) {res['sentences'][i]}\n    Gold: {res['golds'][i]}\n    Pred: {res['preds'][i]}")


if __name__ == "__main__":
    main()
