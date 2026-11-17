"""Xóa các triplet có polarity ``conflict`` khỏi dữ liệu M-ABSA.

Không xóa câu chứa nhãn conflict: các triplet positive/negative/neutral còn lại
trong câu vẫn được giữ. Chạy từ thư mục gốc dự án:

    python src/remove_conflict_labels.py
"""
import ast
from pathlib import Path


DATA_ROOTS = (Path("M-ABSA/data"), Path("data/multi_domain"))


def clean_file(path: Path) -> tuple[int, int]:
    """Trả về (số dòng đã thay đổi, số triplet conflict đã xóa)."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    cleaned_lines, changed_lines, removed = [], 0, 0

    for line in lines:
        body = line.rstrip("\r\n")
        newline = line[len(body):]
        if "####" not in body:
            cleaned_lines.append(line)
            continue

        sentence, raw_labels = body.split("####", 1)
        try:
            labels = ast.literal_eval(raw_labels)
        except (SyntaxError, ValueError):
            cleaned_lines.append(line)
            continue

        kept = [triplet for triplet in labels if len(triplet) < 3 or triplet[2].lower() != "conflict"]
        n_removed = len(labels) - len(kept)
        if n_removed:
            cleaned_lines.append(f"{sentence}####{repr(kept)}{newline}")
            changed_lines += 1
            removed += n_removed
        else:
            cleaned_lines.append(line)

    if removed:
        path.write_text("".join(cleaned_lines), encoding="utf-8")
    return changed_lines, removed


def main():
    changed_files = changed_lines = removed = 0
    for root in DATA_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            n_lines, n_removed = clean_file(path)
            if n_removed:
                print(f"{path}: removed {n_removed} conflict triplets from {n_lines} lines")
                changed_files += 1
                changed_lines += n_lines
                removed += n_removed
    print(f"Done: removed {removed} conflict triplets from {changed_lines} lines in {changed_files} files.")


if __name__ == "__main__":
    main()
