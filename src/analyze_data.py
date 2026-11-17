"""
Phân tích thống kê THẬT của tập dữ liệu đa miền (không gõ tay con số nào).

Chạy:  python src/analyze_data.py
Kết quả: in ra màn hình + lưu results/data_analysis.json
"""
import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import read_mabsa_file  # noqa: E402

DOMAINS = ["hotel", "laptop", "restaurant"]
SPLITS = ["train", "dev", "test"]
BASE = os.path.join("M-ABSA", "data")
MULTI = os.path.join("data", "multi_domain")
POLS = ["positive", "negative", "neutral"]


def load(domain, split, lang="en"):
    return read_mabsa_file(os.path.join(BASE, domain, lang, f"{split}.txt"))


def analyze(sents, labels):
    triplet_pol = Counter()
    pair_pol = Counter()  # (aspect, polarity) sau khi gộp trùng trong từng câu - đúng như cách metric đếm
    aspects = set()
    null_triplets = 0
    n_triplets = 0
    special_char_aspects = 0
    for s, labs in zip(sents, labels):
        pairs = set()
        for t in labs:
            if len(t) < 3:
                continue
            a, _, p = t[0], t[1], t[2].lower()
            n_triplets += 1
            triplet_pol[p] += 1
            pairs.add((a.lower(), p))
            aspects.add(a.lower())
            if a == "NULL":
                null_triplets += 1
            if "," in a or ";" in a:
                special_char_aspects += 1
        for _, p in pairs:
            pair_pol[p] += 1
    return {
        "sentences": len(sents),
        "sentences_without_aspect": sum(1 for l in labels if not l),
        "triplets": n_triplets,
        "unique_pairs_after_dedup": sum(pair_pol.values()),
        "triplet_polarity": dict(triplet_pol),
        "pair_polarity": dict(pair_pol),
        "unique_aspect_terms": len(aspects),
        "null_aspect_triplets": null_triplets,
        "aspects_with_comma_or_semicolon": special_char_aspects,
    }


def main():
    out = {"per_domain": {}, "combined": {}}
    print("=" * 78)
    print("THỐNG KÊ THẬT TỪ FILE DỮ LIỆU GỐC (M-ABSA/data/<domain>/en)")
    print("=" * 78)
    combined = {s: ([], []) for s in SPLITS}
    for d in DOMAINS:
        out["per_domain"][d] = {}
        for s in SPLITS:
            sents, labels = load(d, s)
            combined[s][0].extend(sents)
            combined[s][1].extend(labels)
            st = analyze(sents, labels)
            out["per_domain"][d][s] = st
            tp = st["triplet_polarity"]
            print(f"[{d:<10}|{s:<5}] câu={st['sentences']:>5} triplet={st['triplets']:>5} "
                  f"pos/neg/neu={tp.get('positive',0)}/{tp.get('negative',0)}/{tp.get('neutral',0)} "
                  f"NULL={st['null_aspect_triplets']} uniq_aspect={st['unique_aspect_terms']}")

    print("-" * 78)
    for s in SPLITS:
        st = analyze(*combined[s])
        out["combined"][s] = st
        tp, pp = st["triplet_polarity"], st["pair_polarity"]
        print(f"[ALL |{s:<5}] câu={st['sentences']:>5} triplet={st['triplets']:>5} "
              f"cặp(đã gộp trùng)={st['unique_pairs_after_dedup']:>5}")
        print(f"             triplet pos/neg/neu = {tp.get('positive',0)}/{tp.get('negative',0)}/{tp.get('neutral',0)}"
              f" | cặp pos/neg/neu = {pp.get('positive',0)}/{pp.get('negative',0)}/{pp.get('neutral',0)}")
        print(f"             khía cạnh duy nhất={st['unique_aspect_terms']} | NULL={st['null_aspect_triplets']} "
              f"| khía cạnh chứa ',' hoặc ';'={st['aspects_with_comma_or_semicolon']}"
              f" | câu không có khía cạnh={st['sentences_without_aspect']}")

    all_aspects = set()
    for s in SPLITS:
        for labs in combined[s][1]:
            for t in labs:
                if len(t) >= 3:
                    all_aspects.add(t[0].lower())
    out["unique_aspect_terms_all_splits"] = len(all_aspects)
    print(f"\nTổng số khía cạnh duy nhất (cả 3 tập, 3 miền) = {len(all_aspects)}")

    # Tỷ lệ neutral theo miền (train)
    print("\nTỷ lệ cảm xúc theo miền (tập train, mức triplet):")
    for d in DOMAINS:
        tp = out["per_domain"][d]["train"]["triplet_polarity"]
        tot = sum(tp.values())
        print(f"  {d:<11}: " + ", ".join(f"{p}={tp.get(p,0)} ({100*tp.get(p,0)/tot:.1f}%)" for p in POLS))

    # Kiểm tra rò rỉ dữ liệu: cùng 1 câu xuất hiện ở train và test/dev
    train_set = set(combined["train"][0])
    leak = {}
    for s in ("dev", "test"):
        n = sum(1 for x in combined[s][0] if x in train_set)
        leak[s] = n
        print(f"\nCâu của tập {s} xuất hiện y hệt trong train: {n}/{len(combined[s][0])}")
    out["sentence_overlap_with_train"] = leak

    # Đối chiếu với file đã trộn
    if os.path.exists(os.path.join(MULTI, "train.txt")):
        ms, ml = read_mabsa_file(os.path.join(MULTI, "train.txt"))
        print(f"\nfile đa miền train.txt: {len(ms)} câu "
              f"(tổng 3 miền gốc = {len(combined['train'][0])}) -> "
              f"{'KHỚP' if len(ms) == len(combined['train'][0]) else 'LỆCH'}")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "data_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nĐã lưu results/data_analysis.json")


if __name__ == "__main__":
    main()

