"""
Chạy lần lượt các cấu hình thí nghiệm xử lý mất cân bằng nhãn (cùng seed, cùng siêu tham số),
sau đó đánh giá checkpoint tốt nhất trên TOÀN BỘ dev và test.

Ví dụ:
  python src/run_ablation.py --epochs 8 --seed 42
  python src/run_ablation.py --configs baseline wl_p03 --epochs 8 --seed 42
Có thể chạy lại: cấu hình nào đã có results/<tên>_test.json sẽ được bỏ qua (dùng --force để chạy lại).
"""
import os
import sys
import argparse
import subprocess

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    "baseline": ["--imbalance", "none", "--select_metric", "f1"],
    "wl_p03": ["--imbalance", "weighted_loss", "--power", "0.3", "--select_metric", "macro_f1"],
    "oversample": ["--imbalance", "oversample", "--power", "0.5", "--select_metric", "macro_f1"],
    "weighted_loss": ["--imbalance", "weighted_loss", "--power", "0.5", "--select_metric", "macro_f1"],
}


def run(cmd, log_path):
    print(">>", " ".join(cmd))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Lỗi khi chạy (xem {log_path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["baseline", "wl_p03", "oversample"], choices=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--data_dir", default="data/multi_domain")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for name in args.configs:
        run_name = f"{name}_e{args.epochs}_s{args.seed}"
        out_dir = os.path.join("checkpoints", run_name)
        if not args.force and os.path.exists(os.path.join("results", f"{run_name}_test.json")):
            print(f"[bỏ qua] {run_name} đã có kết quả")
            continue

        train_cmd = [
            PY, os.path.join(HERE, "train.py"),
            "--data_dir", args.data_dir,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--early_stopping_patience", str(args.patience),
            "--output_dir", out_dir,
        ] + CONFIGS[name]
        run(train_cmd, os.path.join("results", "logs", f"{run_name}_train.log"))

        for split in ("dev", "test"):
            eval_cmd = [
                PY, os.path.join(HERE, "evaluate_model.py"),
                "--model_path", os.path.join(out_dir, "best_model"),
                "--data_dir", args.data_dir,
                "--split", split,
                "--run_name", run_name,
                "--num_examples_to_show", "0",
            ]
            run(eval_cmd, os.path.join("results", "logs", f"{run_name}_eval_{split}.log"))
        print(f"[xong] {run_name}")


if __name__ == "__main__":
    main()
