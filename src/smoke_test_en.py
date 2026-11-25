"""
Kiểm thử nhanh chất lượng dự đoán trên 20 câu đánh giá tiếng Anh tự nhiên
(đa miền Hotel, Laptop, Restaurant: tích cực, tiêu cực, trung tính, trái chiều, khía cạnh ẩn NULL).

Ví dụ:
  python src/smoke_test_en.py --model_path checkpoints/best_model
  python src/smoke_test_en.py --model_path checkpoints/wl_p03_e8_s42/best_model --num_beams 2
"""
import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import normalize_english_spacing  # noqa: E402
from metrics import parse_uabsa_string  # noqa: E402

EN_TEST_CASES = [
    # --- Restaurant ---
    ("restaurant", "The food was absolutely amazing, but the service was terrible and slow."),
    ("restaurant", "Atmosphere was nice and the waiter was very friendly."),
    ("restaurant", "Not the biggest portions, but adequate for the price."),
    ("restaurant", "The pizza crust is thin and the sauce is just okay."),
    ("restaurant", "If you're in the area, you shouldn't be disappointed."),
    ("restaurant", "Overpriced drinks and rude staff, never coming back."),
    ("restaurant", "Their appetizers are all delicious!"),
    # --- Laptop ---
    ("laptop", "Great battery life and clear screen, though the keyboard feels a bit cheap."),
    ("laptop", "No backlit keyboard."),
    ("laptop", "Screen is average by 2017 standards."),
    ("laptop", "It works well as a tablet."),
    ("laptop", "Used as bought, it is an ok laptop, but not great."),
    ("laptop", "Fast SSD boot time and quiet fan, very happy with this purchase."),
    ("laptop", "The touchpad stopped working after two weeks and support was useless."),
    # --- Hotel ---
    ("hotel", "The hotel location is perfect, near the beach, but the room was quite small."),
    ("hotel", "Rooms are on the basic side, but they have everything that you'd need."),
    ("hotel", "I'd definitely stay there again."),
    ("hotel", "Breakfast buffet was fresh, however the Wi-Fi was painfully slow."),
    ("hotel", "Clean bathroom and comfortable bed, great value for money."),
    ("hotel", "Check-in took forever and the front desk staff seemed annoyed."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="checkpoints/best_model")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--no_norm", action="store_true", help="Không chuẩn hóa dấu câu kiểu SemEval")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== English Smoke Test (20 natural reviews) ===")
    print(f"Model: {args.model_path} | Device: {device} | Beams: {args.num_beams} | NormSpacing: {not args.no_norm}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()

    pol_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for i, (domain, sent) in enumerate(EN_TEST_CASES, 1):
        inp_text = sent if args.no_norm else normalize_english_spacing(sent)
        enc = tokenizer(inp_text, max_length=128, truncation=True, return_tensors="pt").to(device)
        gen_kwargs = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                          max_length=64, num_beams=args.num_beams)
        if args.num_beams > 1:
            gen_kwargs["early_stopping"] = True
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        raw = tokenizer.decode(out[0], skip_special_tokens=True)
        pairs = sorted(parse_uabsa_string(raw))
        for _, p in pairs:
            if p in pol_counts:
                pol_counts[p] += 1
        print(f"[{i:02d}] ({domain:<10}) {sent}")
        print(f"     -> Raw : {raw}")
        print(f"     -> Pairs: {pairs}")

    print("\n" + "=" * 60)
    print(f"Tổng phân bố nhãn dự đoán trên 20 câu: {pol_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()

