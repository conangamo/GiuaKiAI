import os
import ast
import random
import json
from collections import Counter

def parse_mabsa_file(file_path):
    examples = []
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return examples

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("####")
            if len(parts) != 2:
                continue
            sentence, raw_labels = parts[0].strip(), parts[1].strip()
            try:
                triplets = ast.literal_eval(raw_labels)
            except (ValueError, SyntaxError):
                print(f"Warning: bỏ qua dòng lỗi định dạng trong {file_path}")
                continue
            examples.append({
                "line": line,
                "sentence": sentence,
                "triplets": triplets
            })
    return examples


def analyze_split(examples):
    total_sentences = len(examples)
    total_triplets = sum(len(ex["triplets"]) for ex in examples)
    polarity_counts = Counter()
    aspect_terms = set()

    for ex in examples:
        for tri in ex["triplets"]:
            if len(tri) >= 3:
                aspect_terms.add(tri[0].lower())
                polarity_counts[tri[2].lower()] += 1

    return {
        "num_sentences": total_sentences,
        "num_triplets": total_triplets,
        "num_unique_aspects": len(aspect_terms),
        "polarity_distribution": dict(polarity_counts)
    }


def merge_domains(domains=["hotel", "laptop", "restaurant"], lang="en",
                  base_data_dir="M-ABSA/data", output_dir="data/multi_domain", seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Creating Multi-Domain ABSA Dataset ===")
    print(f"Domains to merge: {domains}")
    print(f"Language: {lang}")
    print(f"Output directory: {output_dir}\n")

    overall_stats = {}

    for split in ["train", "dev", "test"]:
        combined_examples = []
        domain_split_stats = {}

        for domain in domains:
            file_path = os.path.join(base_data_dir, domain, lang, f"{split}.txt")
            if not os.path.exists(file_path):
                # Try relative to script
                file_path = os.path.join("..", base_data_dir, domain, lang, f"{split}.txt")
            
            examples = parse_mabsa_file(file_path)
            for ex in examples:
                ex["domain"] = domain
            stats = analyze_split(examples)
            domain_split_stats[domain] = stats
            combined_examples.extend(examples)
            print(f"[{split.upper()}] Domain '{domain}': {len(examples)} sentences, {stats['num_triplets']} aspect triplets")

        # Shuffle combined examples
        random.shuffle(combined_examples)

        # Write merged file
        out_file = os.path.join(output_dir, f"{split}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            for ex in combined_examples:
                f.write(ex["line"] + "\n")

        # Tag miền cho từng dòng (cùng thứ tự với file trên) -> dùng để đánh giá theo từng miền
        with open(os.path.join(output_dir, f"{split}_domains.txt"), "w", encoding="utf-8") as f:
            for ex in combined_examples:
                f.write(ex["domain"] + "\n")

        combined_stats = analyze_split(combined_examples)
        overall_stats[split] = {
            "per_domain": domain_split_stats,
            "combined": combined_stats
        }
        print(f"--> Saved merged {split.upper()} set: {len(combined_examples)} sentences -> {out_file}\n")

    # Save stats summary
    stats_file = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)

    # Print summary table for Slide
    print("=" * 60)
    print("MULTI-DOMAIN DATASET SUMMARY (FOR SLIDE):")
    print(f"{'Split':<10} | {'Total Sentences':<16} | {'Aspect Triplets':<16} | {'Polarity (Pos / Neg / Neu)'}")
    print("-" * 60)
    for split in ["train", "dev", "test"]:
        s = overall_stats[split]["combined"]
        pos = s["polarity_distribution"].get("positive", 0)
        neg = s["polarity_distribution"].get("negative", 0)
        neu = s["polarity_distribution"].get("neutral", 0)
        print(f"{split.upper():<10} | {s['num_sentences']:<16} | {s['num_triplets']:<16} | {pos} / {neg} / {neu}")
    print("=" * 60)
    print(f"Detailed JSON stats saved to: {stats_file}")


if __name__ == "__main__":
    merge_domains()

