"""
Metric cho UABSA (cặp khía cạnh - cảm xúc), tính trên TẬP CẶP (set) của từng câu.

Quy ước (ghi rõ để báo cáo trung thực):
  * Một cặp = (aspect.lower(), polarity). Cặp trùng trong cùng câu được gộp (set).
    => số "cặp" nhỏ hơn số "triplet" gốc (vd test: 2,531 triplet -> 2,185 cặp).
  * Khớp tuyệt đối (exact match) về chuỗi khía cạnh + nhãn cảm xúc.
  * 'NULL' là khía cạnh ẩn (không xuất hiện trong câu); được tính như một khía cạnh bình thường
    trong số liệu 'overall', đồng thời báo cáo tách riêng 'explicit_only' và 'null_only'.
  * Kết quả theo cảm xúc: micro-F1 (overall) và macro-F1 (trung bình F1 của 3 lớp).
"""
import re
import editdistance

POLARITIES = ["positive", "negative", "neutral"]
# Một cặp "(aspect, polarity)" kết thúc ở ')' đứng trước '; (' hoặc cuối chuỗi
_ITEM_RE = re.compile(r"\((.*?)\)(?=\s*;\s*\(|\s*$)", re.S)


def fuzzy_match_polarity(candidate, max_dist=2):
    best, best_d = None, 10 ** 6
    for val in POLARITIES:
        d = editdistance.eval(candidate.lower(), val)
        if d < best_d and d <= max_dist:
            best, best_d = val, d
    return best


def parse_uabsa_string(text):
    """'(food, positive); (service, negative)' -> {('food','positive'), ('service','negative')}"""
    if text is None:
        return set()
    text = text.strip()
    if not text or text.lower() == "none":
        return set()

    pairs = set()
    for item in _ITEM_RE.findall(text):
        if "," not in item:
            continue
        aspect, polarity = item.rsplit(",", 1)  # rsplit: khía cạnh có dấu phẩy vẫn đúng
        aspect = aspect.strip().lower()
        polarity = polarity.strip().lower()
        if not aspect:
            continue
        if polarity not in POLARITIES:
            polarity = fuzzy_match_polarity(polarity)
        if polarity:
            pairs.add((aspect, polarity))
    return pairs


def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2), "f1": round(100 * f, 2),
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def _score(pred_sets, gold_sets, keep=lambda pair: True, ignore_polarity=False):
    tp = n_pred = n_gold = 0
    for pred, gold in zip(pred_sets, gold_sets):
        if ignore_polarity:
            pred = {(a, "_") for a, _ in pred if keep((a, _))}
            gold = {(a, "_") for a, _ in gold if keep((a, _))}
        else:
            pred = {x for x in pred if keep(x)}
            gold = {x for x in gold if keep(x)}
        tp += len(pred & gold)
        n_pred += len(pred)
        n_gold += len(gold)
    return _prf(tp, n_pred, n_gold)


def _full_report(pred_sets, gold_sets):
    overall = _score(pred_sets, gold_sets)

    per_class = {}
    for c in POLARITIES:
        per_class[c] = _score(pred_sets, gold_sets, keep=lambda x, c=c: x[1] == c)
    macro_f1 = round(sum(per_class[c]["f1"] for c in POLARITIES) / len(POLARITIES), 2)

    # Tách bài toán: (1) trích xuất khía cạnh (bỏ qua cảm xúc)  (2) phân loại cảm xúc khi đã tìm đúng khía cạnh
    aspect_only = _score(pred_sets, gold_sets, ignore_polarity=True)

    confusion = {g: {p: 0 for p in POLARITIES} for g in POLARITIES}
    matched = correct = 0
    for pred, gold in zip(pred_sets, gold_sets):
        pred_map, gold_map = {}, {}
        for a, p in pred:
            pred_map.setdefault(a, set()).add(p)
        for a, p in gold:
            gold_map.setdefault(a, set()).add(p)
        for a, gpols in gold_map.items():
            if a not in pred_map:
                continue
            for g in gpols:
                # nếu mô hình có đoán đúng nhãn g cho khía cạnh này thì tính đúng, ngược lại lấy nhãn bất kỳ đã đoán
                pp = g if g in pred_map[a] else sorted(pred_map[a])[0]
                confusion[g][pp] += 1
                matched += 1
                correct += int(pp == g)
    pol_acc = round(100 * correct / matched, 2) if matched else 0.0

    return {
        # các khóa cũ (giữ tương thích với train.py / app.py)
        "precision": overall["precision"], "recall": overall["recall"], "f1": overall["f1"],
        "total_gold": overall["n_gold"], "total_pred": overall["n_pred"], "true_positives": overall["tp"],
        # các khóa mới
        "macro_f1": macro_f1,
        "per_class": per_class,
        "aspect_only": aspect_only,
        "polarity_acc_on_matched_aspects": pol_acc,
        "n_matched_aspects": matched,
        "polarity_confusion(gold->pred)": confusion,
        "explicit_only": _score(pred_sets, gold_sets, keep=lambda x: x[0] != "null"),
        "null_only": _score(pred_sets, gold_sets, keep=lambda x: x[0] == "null"),
    }


def compute_metrics(predictions, targets, task="uabsa", domains=None):
    """
    predictions, targets: list[str]; domains: list[str] cùng độ dài (tùy chọn) để báo cáo theo miền.
    """
    pred_sets = [parse_uabsa_string(p) for p in predictions]
    gold_sets = [parse_uabsa_string(g) for g in targets]
    report = _full_report(pred_sets, gold_sets)
    report["n_sentences"] = len(predictions)

    if domains is not None:
        report["per_domain"] = {}
        for d in sorted(set(domains)):
            idx = [i for i, x in enumerate(domains) if x == d]
            sub = _full_report([pred_sets[i] for i in idx], [gold_sets[i] for i in idx])
            sub["n_sentences"] = len(idx)
            report["per_domain"][d] = sub
    return report


def format_report(rep, title=""):
    """In báo cáo dạng bảng văn bản."""
    L = []
    if title:
        L.append("=" * 72)
        L.append(title)
    L.append("=" * 72)
    L.append(f"Số câu: {rep.get('n_sentences', '?')} | cặp gold: {rep['total_gold']} | cặp dự đoán: {rep['total_pred']} | đúng: {rep['true_positives']}")
    L.append(f"OVERALL (micro)   P={rep['precision']:.2f}  R={rep['recall']:.2f}  F1={rep['f1']:.2f}   | MACRO-F1 (3 lớp cảm xúc) = {rep['macro_f1']:.2f}")
    ao = rep["aspect_only"]
    L.append(f"Chỉ trích xuất khía cạnh (bỏ qua cảm xúc): P={ao['precision']:.2f} R={ao['recall']:.2f} F1={ao['f1']:.2f}")
    L.append(f"Độ chính xác cảm xúc khi đã tìm đúng khía cạnh: {rep['polarity_acc_on_matched_aspects']:.2f}% (trên {rep['n_matched_aspects']} khía cạnh)")
    ex, nl = rep["explicit_only"], rep["null_only"]
    L.append(f"Khía cạnh hiện (không NULL): P={ex['precision']:.2f} R={ex['recall']:.2f} F1={ex['f1']:.2f} (gold={ex['n_gold']})")
    L.append(f"Khía cạnh ẩn NULL        : P={nl['precision']:.2f} R={nl['recall']:.2f} F1={nl['f1']:.2f} (gold={nl['n_gold']})")
    L.append("-" * 72)
    L.append(f"{'Lớp cảm xúc':<12}{'Precision':>10}{'Recall':>10}{'F1':>8}{'Gold':>8}{'Pred':>8}")
    for c in POLARITIES:
        s = rep["per_class"][c]
        L.append(f"{c:<12}{s['precision']:>10.2f}{s['recall']:>10.2f}{s['f1']:>8.2f}{s['n_gold']:>8}{s['n_pred']:>8}")
    L.append("Ma trận nhầm lẫn cảm xúc (hàng = gold, cột = dự đoán; chỉ các khía cạnh đã tìm đúng):")
    L.append(f"{'':<12}" + "".join(f"{c:>10}" for c in POLARITIES))
    for g in POLARITIES:
        L.append(f"{g:<12}" + "".join(f"{rep['polarity_confusion(gold->pred)'][g][p]:>10}" for p in POLARITIES))
    if "per_domain" in rep:
        L.append("-" * 72)
        L.append(f"{'Miền':<12}{'Câu':>6}{'P':>8}{'R':>8}{'F1':>8}{'MacroF1':>9}{'F1 neu':>8}")
        for d, s in rep["per_domain"].items():
            L.append(f"{d:<12}{s['n_sentences']:>6}{s['precision']:>8.2f}{s['recall']:>8.2f}{s['f1']:>8.2f}"
                     f"{s['macro_f1']:>9.2f}{s['per_class']['neutral']['f1']:>8.2f}")
    L.append("=" * 72)
    return "\n".join(L)
