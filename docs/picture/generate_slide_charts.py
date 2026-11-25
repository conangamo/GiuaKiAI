import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Thiết lập font chữ hỗ trợ đầy đủ tiếng Việt có dấu trên Windows
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(OUTPUT_DIR, "..", ".."))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_json(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------
# CHART 1: Slide 1 - Concept: Sentence-Level vs ABSA
# -------------------------------------------------------------
def create_slide1_concept():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, "SO SÁNH: PHÂN LOẠI CẢM XÚC CẤP CÂU vs BÀI TOÁN ABSA ĐA TÁC VỤ",
            fontsize=14.5, fontweight='bold', ha='center', color='#1A237E')

    # Example sentence
    sentence = "“The food was absolutely amazing, but the service was terrible and slow.”"
    ax.text(0.5, 0.82, sentence, fontsize=12, fontstyle='italic', ha='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F7FA', edgecolor='#B0BEC5', lw=1.2))

    # Box 1: Sentence-level
    ax.text(0.25, 0.64, "Phân Loại Cấp Câu (Sentence-Level)", fontsize=12, fontweight='bold', ha='center', color='#C62828')
    rect1 = FancyBboxPatch((0.05, 0.28), 0.40, 0.32, facecolor='#FFEBEE', edgecolor='#EF5350', lw=1.5,
                           boxstyle="round,pad=0.03")
    ax.add_patch(rect1)
    ax.text(0.25, 0.44, "Gán 1 nhãn chung cho cả câu:\n[-] 'Negative' hoặc 'Mixed'\n(Làm mất thông tin khen/chê cụ thể)",
            fontsize=11, ha='center', va='center', color='#B71C1C')

    # Box 2: Aspect-Based (ABSA)
    ax.text(0.75, 0.64, "Phân Tích Theo Khía Cạnh (ABSA)", fontsize=12, fontweight='bold', ha='center', color='#2E7D32')
    rect2 = FancyBboxPatch((0.55, 0.28), 0.40, 0.32, facecolor='#E8F5E9', edgecolor='#66BB6A', lw=1.5,
                           boxstyle="round,pad=0.03")
    ax.add_patch(rect2)
    ax.text(0.75, 0.44, "Thực hiện đồng thời 2 tác vụ:\n[+] (food, positive) -> Khen món ăn\n[-] (service, negative) -> Chê phục vụ\n(Bóc tách chính xác từng thực thể & cực tính)",
            fontsize=10.5, ha='center', va='center', color='#1B5E20')

    # Footer note (dùng ký tự chuẩn Unicode trong Segoe UI)
    ax.text(0.5, 0.12, "=> ABSA giải quyết đồng thời Trích xuất khía cạnh (AE) + Phân loại cảm xúc (ASC) ở mức chi tiết.",
            fontsize=11, fontweight='bold', ha='center', color='#283593')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide1_absa_concept.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


# -------------------------------------------------------------
# CHART 2: Slide 2 - Multi-Domain Distribution (1:1:1 Balance)
# -------------------------------------------------------------
def create_slide2_domain_distribution():
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

    domains = ['Hotel (Khách sạn)\n[Tổng: 2,147 câu]', 'Laptop (Máy tính)\n[Tổng: 2,122 câu]', 'Restaurant (Nhà hàng)\n[Tổng: 2,124 câu]']
    train_counts = [1255, 1264, 1264]
    dev_counts = [308, 326, 316]
    test_counts = [584, 532, 544]

    x = np.arange(len(domains))
    width = 0.24

    rects1 = ax.bar(x - width, train_counts, width, label='Train (3,783 câu)', color='#1976D2', edgecolor='white')
    rects2 = ax.bar(x, dev_counts, width, label='Dev (950 câu)', color='#FB8C00', edgecolor='white')
    rects3 = ax.bar(x + width, test_counts, width, label='Test (1,660 câu)', color='#43A047', edgecolor='white')

    ax.set_ylabel('Số lượng câu (Sentences)', fontsize=11.5, fontweight='bold')
    ax.set_title('PHÂN BỐ TẬP DỮ LIỆU ĐA MIỀN (TỶ LỆ CÂN BẰNG 1 : 1 : 1)', fontsize=13.5, fontweight='bold', pad=15, color='#1A237E')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10.5, fontweight='bold')
    ax.set_ylim(0, 1450)
    ax.legend(frameon=True, facecolor='#FAFAFA', fontsize=10.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.text(0.5, -0.18, "Tổng: 6,393 câu | 1,767 khía cạnh duy nhất | Mỗi miền đóng góp ~33.3% (Triệt tiêu Domain Bias)",
            transform=ax.transAxes, ha='center', fontsize=10.5, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#90CAF9'))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide2_domain_distribution.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


# -------------------------------------------------------------
# CHART 3: Slide 2 - Sentiment Polarity Donut Chart
# -------------------------------------------------------------
def create_slide2_sentiment_donut():
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=300)

    labels = ['Tích cực (Positive)\n6,580 mẫu', 'Tiêu cực (Negative)\n2,250 mẫu', 'Trung tính (Neutral)\n379 mẫu']
    sizes = [6580, 2250, 379]
    colors = ['#2E7D32', '#C62828', '#FFA000']
    explode = (0.03, 0.03, 0.06)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors, explode=explode,
        pctdistance=0.74, wedgeprops=dict(width=0.45, edgecolor='white', lw=2)
    )

    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(10.5)
        at.set_weight('bold')

    for t in texts:
        t.set_fontsize(10.5)
        t.set_weight('bold')

    ax.set_title('PHÂN BỐ CẢM XÚC TRÊN 9,209 NHÃN BỘ BA (MẤT CÂN BẰNG NẶNG)', fontsize=12.5, fontweight='bold', color='#1A237E', pad=12)
    ax.text(0, 0, "9,209\nTriplets\n(7,882 cặp\nduy nhất)", ha='center', va='center', fontsize=11, fontweight='bold', color='#37474F')

    ax.text(0.5, -0.06, "Lưu ý: Sau khi khử trùng lặp cặp (aspect, polarity) theo câu cho tác vụ UABSA:\nTrain = 4,582 cặp | Dev = 1,115 cặp | Test = 2,185 cặp (Lớp Neutral chỉ chiếm ~4.1%)",
            transform=ax.transAxes, ha='center', fontsize=9.5, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF8E1', edgecolor='#FFB300'))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide2_sentiment_donut.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


# -------------------------------------------------------------
# CHART 3B: Slide 2 - Data Mixing Pipeline (Cách trộn dữ liệu đa miền)
# -------------------------------------------------------------
def create_slide2_data_mixing_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    ax.axis('off')
    ax.text(0.5, 0.94, "QUY TRÌNH VÀ CÁCH TRỘN TẬP DỮ LIỆU ĐA MIỀN (MULTI-DOMAIN DATA MIXING)", ha='center',
            fontsize=13.5, fontweight='bold', color='#1A237E')

    stages = [
        (0.02, 0.22, '#ECEFF1', '#455A64', '1. Ba Miền Nguồn',
         '• Hotel: 2,147 câu\n• Laptop: 2,122 câu\n• Restaurant: 2,124 câu\n(Nguồn: M-ABSA SemEval)\nTổng: 6,393 câu / 9,210 triplets'),
        (0.28, 0.22, '#E3F2FD', '#1976D2', '2. Thuật Toán Trộn',
         '• Ghép tương ứng từng split:\n  Train + Dev + Test\n• Shuffle ngẫu nhiên (seed=42)\n• Sinh file {split}_domains.txt\n  để theo dõi nguồn gốc miền'),
        (0.54, 0.22, '#FFF3E0', '#EF6C00', '3. Tập Đa Miền Hoàn Chỉnh',
         '• Train: 3,783 câu\n• Dev: 950 câu\n• Test: 1,660 câu\n• Tỷ lệ cân bằng 1 : 1 : 1\n  (Triệt tiêu Domain Bias)'),
        (0.80, 0.18, '#E8F5E9', '#2E7D32', '4. Chuẩn Hóa UABSA',
         '• Khử trùng lặp cặp\n  (aspect, polarity)\n  trong cùng 1 câu\n• Test: 2,185 cặp chuẩn\n• Train: 4,583 | Dev: 1,115'),
    ]

    for x, w, face, edge, title, body in stages:
        ax.add_patch(FancyBboxPatch((x, 0.32), w, 0.48, facecolor=face, edgecolor=edge, lw=2,
                                    boxstyle='round,pad=0.02'))
        ax.text(x + w / 2, 0.72, title, ha='center', va='center', fontsize=11, fontweight='bold', color=edge)
        ax.text(x + w / 2, 0.50, body, ha='center', va='center', fontsize=9.2)

    for start in (0.24, 0.50, 0.76):
        ax.annotate('', xy=(start + 0.04, 0.56), xytext=(start, 0.56),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='#455A64'))

    ax.text(0.5, 0.14, "Mã nguồn thực hiện: src/build_multi_domain.py - Đảm bảo cân bằng 1:1:1 giữa 3 miền và toàn vẹn dữ liệu (kiểm định mã băm MD5)",
            ha='center', fontsize=10.2, fontweight='bold', color='#1A237E',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#BDBDBD'))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide2_data_mixing_pipeline.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")



# -------------------------------------------------------------
# CHART 4: Slide 3 & 4 - Model Architecture (Seq2Seq Text-to-Text)
# -------------------------------------------------------------
def create_slide3_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=300)
    ax.axis('off')

    ax.text(0.5, 0.94, "KIẾN TRÚC MÔ HÌNH mT5 CHO BÀI TOÁN ABSA ĐA TÁC VỤ (TEXT-TO-TEXT)",
            fontsize=13.5, fontweight='bold', ha='center', color='#1A237E')

    # Box 1: Input
    rect_in = FancyBboxPatch((0.02, 0.36), 0.21, 0.42, facecolor='#E3F2FD', edgecolor='#1976D2', lw=2, boxstyle="round,pad=0.02")
    ax.add_patch(rect_in)
    ax.text(0.125, 0.69, "Đầu vào (Source)", fontsize=11, fontweight='bold', ha='center', color='#0D47A1')
    ax.text(0.125, 0.50, "“The food was\namazing, but the\nservice was slow.”", fontsize=9.5, ha='center', fontstyle='italic')

    # Arrow 1
    ax.annotate('', xy=(0.29, 0.57), xytext=(0.24, 0.57),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='#455A64'))

    # Box 2: mT5 Shared Encoder
    rect_enc = FancyBboxPatch((0.30, 0.31), 0.21, 0.52, facecolor='#EDE7F6', edgecolor='#512DA8', lw=2, boxstyle="round,pad=0.02")
    ax.add_patch(rect_enc)
    ax.text(0.405, 0.71, "mT5 Shared\nEncoder", fontsize=11.5, fontweight='bold', ha='center', color='#311B92')
    ax.text(0.405, 0.47, "• Self-Attention\n• Biểu diễn ngữ cảnh\n  đa miền (Hotel,\n  Laptop, Restaurant)", fontsize=9, ha='center')

    # Arrow 2
    ax.annotate('', xy=(0.57, 0.57), xytext=(0.52, 0.57),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='#455A64'))

    # Box 3: mT5 Autoregressive Decoder
    rect_dec = FancyBboxPatch((0.58, 0.31), 0.21, 0.52, facecolor='#FFF3E0', edgecolor='#E65100', lw=2, boxstyle="round,pad=0.02")
    ax.add_patch(rect_dec)
    ax.text(0.685, 0.71, "mT5 Shared\nDecoder", fontsize=11.5, fontweight='bold', ha='center', color='#BF360C')
    ax.text(0.685, 0.47, "• Cross-Attention\n• Sinh tự hồi quy\n• Sinh chuỗi UABSA\n  có cấu trúc", fontsize=9, ha='center')

    # Arrow 3
    ax.annotate('', xy=(0.85, 0.57), xytext=(0.80, 0.57),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='#455A64'))

    # Box 4: Output
    rect_out = FancyBboxPatch((0.86, 0.36), 0.12, 0.42, facecolor='#E8F5E9', edgecolor='#2E7D32', lw=2, boxstyle="round,pad=0.02")
    ax.add_patch(rect_out)
    ax.text(0.92, 0.69, "Chuỗi Đích", fontsize=11, fontweight='bold', ha='center', color='#1B5E20')
    ax.text(0.92, 0.50, "(food, positive);\n(service, negative)", fontsize=8.8, ha='center', fontweight='bold', color='#1B5E20')

    # Bottom caption (tiếng Việt có dấu đầy đủ)
    ax.text(0.5, 0.12, "Một mT5 Encoder-Decoder sinh trực tiếp chuỗi UABSA có cấu trúc.\nCơ chế hai tác vụ được giải thích riêng ở Slide 4.",
            fontsize=10.2, fontweight='bold', ha='center', color='#1A237E',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#F5F5F5', edgecolor='#BDBDBD'))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide3_model_architecture.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


# -------------------------------------------------------------
# CHART 5: Slide 6 - Training Loss & F1 Curve (8 Epochs) + Test Metrics
# -------------------------------------------------------------
def create_slide6_training_metrics():
    hist_wl = _load_json("checkpoints/wl_p03_e8_s42/history.json")
    test_legacy = _load_json("results/legacy_baseline_test.json")
    test_base8 = _load_json("results/baseline_e8_s42_test.json")
    test_wl8 = _load_json("results/wl_p03_e8_s42_test.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.9), dpi=300)

    # Subplot 1: 8-Epoch Convergence (True Train Loss, Val Loss & Dev Micro/Macro F1)
    epochs = [r["epoch"] for r in hist_wl]
    train_loss = [r["train_loss"] for r in hist_wl]
    val_loss = [r["val_loss"] for r in hist_wl]
    dev_micro = [r["dev_f1"] for r in hist_wl]
    dev_macro = [r["dev_macro_f1"] for r in hist_wl]

    color1 = '#C62828'
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cross-Entropy Loss', color=color1, fontsize=11, fontweight='bold')
    l1 = ax1.plot(epochs, train_loss, color='#E53935', marker='o', linestyle='--', lw=1.8, label='Train Loss')
    l2 = ax1.plot(epochs, val_loss, color='#8E24AA', marker='d', lw=2.3, label='Val Loss (Teacher-Forcing)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(epochs)
    ax1.set_ylim(0, max(val_loss) * 1.6)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax1_twin = ax1.twinx()
    color2 = '#1565C0'
    ax1_twin.set_ylabel('Điểm F1 trên toàn bộ Dev 950 câu (%)', color=color2, fontsize=10.5, fontweight='bold')
    l3 = ax1_twin.plot(epochs, dev_micro, color='#1E88E5', marker='s', lw=2.2, label='Dev Micro-F1 (%)')
    l4 = ax1_twin.plot(epochs, dev_macro, color='#2E7D32', marker='^', lw=2.5, label='Dev Macro-F1 (%)')
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1_twin.set_ylim(0, 75)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=8.5, frameon=True, facecolor='white')
    ax1.set_title('Tiến Trình Hội Tụ Qua 8 Epoch (Weighted Loss p=0.3)', fontsize=11.5, fontweight='bold', pad=12, color='#1A237E')

    # Subplot 2: Test Set Metrics Breakdown (Best Model: wl_p03_e8_s42 on 1,660 sentences)
    metrics = [
        'Trích xuất\nAspect (F1)',
        'Độ chính xác\nCảm xúc*',
        'Micro-F1\n(WL 8ep)',
        'Macro-F1\n(Base 3ep)',
        'Macro-F1\n(Base 8ep)',
        'Macro-F1\n(WL 8ep)',
    ]
    values = [
        test_wl8["aspect_only"]["f1"],
        test_wl8["polarity_acc_on_matched_aspects"],
        test_wl8["f1"],
        test_legacy["macro_f1"],
        test_base8["macro_f1"],
        test_wl8["macro_f1"],
    ]
    bar_colors = ['#0288D1', '#00897B', '#1976D2', '#E53935', '#FB8C00', '#2E7D32']

    bars = ax2.bar(metrics, values, color=bar_colors, width=0.58, edgecolor='black', lw=0.9)
    ax2.set_ylabel('Phần trăm (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 98)
    ax2.set_title('ĐÁNH GIÁ TOÀN DIỆN TẬP TEST ĐA MIỀN\n(1,660 câu - 2,185 cặp khía cạnh chuẩn)', fontsize=11.5, fontweight='bold', pad=12, color='#1A237E')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    ax2.tick_params(axis='x', labelsize=8.5)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide6_training_metrics.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


# -------------------------------------------------------------
# CHART 6: Slide 6 - Qualitative Error Analysis
# -------------------------------------------------------------
def create_slide6_error_analysis():
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)
    ax.axis('off')

    ax.text(0.5, 0.95, "PHÂN TÍCH ĐỊNH TÍNH TRÊN TẬP TEST: ĐÚNG TUYỆT ĐỐI, KHÔI PHỤC NEUTRAL & LỆCH RANH GIỚI",
            fontsize=12, fontweight='bold', ha='center', color='#1A237E')

    # Case 1: Exact Correct
    rect_c1 = FancyBboxPatch((0.04, 0.65), 0.92, 0.24, facecolor='#E8F5E9', edgecolor='#43A047', lw=1.5,
                             boxstyle="round,pad=0.02")
    ax.add_patch(rect_c1)
    ax.text(0.06, 0.83, "[1] TRƯỜNG HỢP KHỚP CHÍNH XÁC TUYỆT ĐỐI (EXACT MATCH)", fontsize=10.5, fontweight='bold', color='#1B5E20')
    ax.text(0.06, 0.75, "Câu Test (#1 - Laptop): “it works well as a tablet .”", fontsize=9.8, fontstyle='italic')
    ax.text(0.06, 0.68, "Nhãn chuẩn (Gold): (tablet, positive)   |   Mô hình dự đoán: (tablet, positive)  ->  Đúng 100%", fontsize=9.8, fontweight='bold', color='#2E7D32')

    # Case 2: Neutral Recovery
    rect_c2 = FancyBboxPatch((0.04, 0.36), 0.92, 0.24, facecolor='#E3F2FD', edgecolor='#1E88E5', lw=1.5,
                             boxstyle="round,pad=0.02")
    ax.add_patch(rect_c2)
    ax.text(0.06, 0.54, "[2] KHÔI PHỤC THÀNH CÔNG LỚP TRUNG TÍNH NHỜ WEIGHTED LOSS (BƯỚC 2)", fontsize=10.5, fontweight='bold', color='#0D47A1')
    ax.text(0.06, 0.46, "Câu Test (#388 - Hotel): “Rooms are on the basic side , but they have everything that you ' d need .”", fontsize=9.8, fontstyle='italic')
    ax.text(0.06, 0.39, "Nhãn chuẩn: (Rooms, neutral)  |  Baseline đoán: (Rooms, positive) [Sai]  ->  Weighted Loss: (Rooms, neutral) [Đúng!]",
            fontsize=9.5, fontweight='bold', color='#1565C0')

    # Case 3: Boundary mismatch
    rect_c3 = FancyBboxPatch((0.04, 0.07), 0.92, 0.24, facecolor='#FFF8E1', edgecolor='#FFB300', lw=1.5,
                             boxstyle="round,pad=0.02")
    ax.add_patch(rect_c3)
    ax.text(0.06, 0.25, "[3] HIỆN TƯỢNG LỆCH RANH GIỚI CỤM TỪ (SPAN BOUNDARY MISMATCH)", fontsize=10.5, fontweight='bold', color='#E65100')
    ax.text(0.06, 0.17, "Câu Test (#4 - Laptop): “no backlit keyboard .”", fontsize=9.8, fontstyle='italic')
    ax.text(0.06, 0.10, "Nhãn chuẩn (Gold): (keyboard, negative)   !=   Dự đoán: (backlit keyboard, negative)", fontsize=9.8, fontweight='bold', color='#D84315')

    ax.text(0.5, 0.01, "Nhận xét: Tiêu chí Exact Match trừ điểm câu #4, nhưng dự đoán 'backlit keyboard' thực tế còn cụ thể hơn nhãn gốc.",
            fontsize=9.5, fontweight='bold', ha='center', color='#37474F')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide6_error_analysis.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


# -------------------------------------------------------------
# CHART 7: Slide 6 - Imbalance Mitigation Comparison (from real JSON)
# -------------------------------------------------------------
def create_slide6_imbalance_comparison():
    test_legacy = _load_json("results/legacy_baseline_test.json")
    test_base8 = _load_json("results/baseline_e8_s42_test.json")
    test_wl8 = _load_json("results/wl_p03_e8_s42_test.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=300)

    # Subplot 1: Test Micro-F1, Macro-F1, Positive, Negative, Neutral F1
    categories = ['Micro-F1', 'Macro-F1', 'Positive\nF1', 'Negative\nF1', 'Neutral\nF1']
    s_leg = [test_legacy["f1"], test_legacy["macro_f1"],
             test_legacy["per_class"]["positive"]["f1"],
             test_legacy["per_class"]["negative"]["f1"],
             test_legacy["per_class"]["neutral"]["f1"]]
    s_b8 = [test_base8["f1"], test_base8["macro_f1"],
            test_base8["per_class"]["positive"]["f1"],
            test_base8["per_class"]["negative"]["f1"],
            test_base8["per_class"]["neutral"]["f1"]]
    s_wl8 = [test_wl8["f1"], test_wl8["macro_f1"],
             test_wl8["per_class"]["positive"]["f1"],
             test_wl8["per_class"]["negative"]["f1"],
             test_wl8["per_class"]["neutral"]["f1"]]

    x = np.arange(len(categories))
    width = 0.26

    r1 = ax1.bar(x - width, s_leg, width, label='Baseline 3ep (CE chuẩn)', color='#BDBDBD', edgecolor='black', lw=0.7)
    r2 = ax1.bar(x, s_b8, width, label='Baseline 8ep (Hội tụ)', color='#1976D2', edgecolor='black', lw=0.7)
    r3 = ax1.bar(x + width, s_wl8, width, label='Weighted Loss p=0.3, 8ep (Tốt nhất)', color='#2E7D32', edgecolor='black', lw=0.7)

    ax1.set_ylabel('Điểm F1 trên tập Test (%)', fontsize=11, fontweight='bold')
    ax1.set_title('ĐỐI SÁNH HIỆU NĂNG TẬP TEST (1,660 CÂU)', fontsize=12, fontweight='bold', pad=12, color='#1A237E')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9.5, fontweight='bold')
    ax1.set_ylim(0, 90)
    ax1.legend(frameon=True, fontsize=8.8, loc='upper right')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    for rects, col in [(r1, '#616161'), (r2, '#0D47A1'), (r3, '#1B5E20')]:
        for r in rects:
            h = r.get_height()
            ax1.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=col)

    # Subplot 2: Neutral F1 per Domain on Test Set
    dom_keys = ['hotel', 'laptop', 'restaurant']
    domains = ['Hotel\n(584 câu)', 'Laptop\n(532 câu)', 'Restaurant\n(544 câu)']
    leg_neu = [test_legacy["per_domain"][d]["per_class"]["neutral"]["f1"] for d in dom_keys]
    b8_neu = [test_base8["per_domain"][d]["per_class"]["neutral"]["f1"] for d in dom_keys]
    wl8_neu = [test_wl8["per_domain"][d]["per_class"]["neutral"]["f1"] for d in dom_keys]

    xd = np.arange(len(domains))
    rd1 = ax2.bar(xd - width, leg_neu, width, label='Baseline 3ep (0.0%)', color='#E57373', edgecolor='black', lw=0.7)
    rd2 = ax2.bar(xd, b8_neu, width, label='Baseline 8ep', color='#64B5F6', edgecolor='black', lw=0.7)
    rd3 = ax2.bar(xd + width, wl8_neu, width, label='Weighted Loss p=0.3, 8ep', color='#2E7D32', edgecolor='black', lw=0.7)

    ax2.set_ylabel('Neutral F1-Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('F1 LỚP NEUTRAL THEO TỪNG MIỀN (TEST)', fontsize=12, fontweight='bold', pad=12, color='#1A237E')
    ax2.set_xticks(xd)
    ax2.set_xticklabels(domains, fontsize=10.5, fontweight='bold')
    ax2.set_ylim(0, 42)
    ax2.legend(frameon=True, fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    for rects, col in [(rd1, '#B71C1C'), (rd2, '#1565C0'), (rd3, '#1B5E20')]:
        for r in rects:
            h = r.get_height()
            ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=col)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide6_imbalance_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def create_slide4_multitask_uabsa():
    """Sơ đồ chính xác cho Unified Generative UABSA, không diễn giải thành hai head."""
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=300)
    ax.axis('off')
    ax.text(0.5, 0.94, "CƠ CHẾ MULTI-TASK: UNIFIED GENERATIVE UABSA", ha='center',
            fontsize=14, fontweight='bold', color='#1A237E')

    boxes = [
        (0.03, 0.36, 0.20, 0.36, '#E3F2FD', '#1976D2', 'Input review',
         '“The food was amazing,\nbut the service was slow.”'),
        (0.29, 0.30, 0.22, 0.48, '#EDE7F6', '#512DA8', 'mT5 dùng chung tham số',
         'Shared Encoder + Decoder\nMột forward pass\nTeacher forcing khi train'),
        (0.57, 0.36, 0.18, 0.36, '#E8F5E9', '#2E7D32', 'Task 1: AE',
         'Sinh token khía cạnh\nfood ; service'),
        (0.79, 0.36, 0.18, 0.36, '#FFF3E0', '#E65100', 'Task 2: ASC',
         'Sinh polarity đúng cặp\npositive ; negative'),
    ]
    for x, y, w, h, face, edge, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge, lw=2,
                                    boxstyle='round,pad=0.02'))
        ax.text(x + w / 2, y + h * .72, title, ha='center', va='center', fontsize=10.5,
                fontweight='bold', color=edge)
        ax.text(x + w / 2, y + h * .38, body, ha='center', va='center', fontsize=9.3)
    for start, end in [(.24, .285), (.52, .565), (.76, .785)]:
        ax.annotate('', xy=(end, .54), xytext=(start, .54), arrowprops=dict(arrowstyle='->', lw=2.5, color='#455A64'))
    ax.text(.77, .20, 'Output duy nhất: (food, positive); (service, negative)', ha='center', fontsize=11,
            fontweight='bold', color='#1B5E20', bbox=dict(boxstyle='round,pad=.45', facecolor='#F1F8E9', edgecolor='#66BB6A'))
    ax.text(.5, .07, 'Loss tối ưu tất cả token đích: token aspect + token polarity. Không dùng pipeline hai mô hình hay hai classifier head.',
            ha='center', fontsize=10, color='#37474F')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'slide4_multitask_uabsa.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'Generated: {out_path}')


def create_slide5_transfer_learning():
    """Sơ đồ transfer learning từ checkpoint mT5 sang UABSA đa miền."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    ax.axis('off')
    ax.text(.5, .94, 'TRANSFER LEARNING: PRE-TRAINED mT5 → FINE-TUNING ABSA ĐA MIỀN', ha='center',
            fontsize=13.5, fontweight='bold', color='#1A237E')
    stages = [
        (.03, '#ECEFF1', '#455A64', '1. Pre-training', 'mT5-small\nTri thức ngôn ngữ\nđa ngôn ngữ'),
        (.29, '#E3F2FD', '#1976D2', '2. Fine-tuning', 'Train: 3,783 câu\nHotel + Laptop\n+ Restaurant'),
        (.55, '#FFF3E0', '#EF6C00', '3. Tối ưu ổn định', 'AdamW · Warmup\nBatch hiệu dụng = 16\nGrad clip = 1.0'),
        (.81, '#E8F5E9', '#2E7D32', '4. Chọn checkpoint', 'Dev: 950 câu\nLưu best_model\nđể đánh giá Test'),
    ]
    for x, face, edge, title, body in stages:
        ax.add_patch(FancyBboxPatch((x, .38), .16, .34, facecolor=face, edgecolor=edge, lw=2,
                                    boxstyle='round,pad=.02'))
        ax.text(x + .08, .63, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=edge)
        ax.text(x + .08, .49, body, ha='center', va='center', fontsize=9.4)
    for start in (.20, .46, .72):
        ax.annotate('', xy=(start + .07, .55), xytext=(start, .55), arrowprops=dict(arrowstyle='->', lw=2.5, color='#455A64'))
    ax.text(.5, .18, 'Toàn bộ tham số được fine-tune end-to-end: mô hình thích ứng định dạng UABSA và ngữ cảnh review của ba miền.',
            ha='center', fontsize=10.5, fontweight='bold', color='#1A237E',
            bbox=dict(boxstyle='round,pad=.45', facecolor='#F5F5F5', edgecolor='#BDBDBD'))
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'slide5_transfer_learning.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'Generated: {out_path}')


def main():
    print("=== Generating High-Quality Slide Images ===")
    create_slide1_concept()
    create_slide2_domain_distribution()
    create_slide2_sentiment_donut()
    create_slide2_data_mixing_pipeline()
    create_slide3_architecture()
    create_slide4_multitask_uabsa()
    create_slide5_transfer_learning()
    create_slide6_training_metrics()
    create_slide6_error_analysis()
    create_slide6_imbalance_comparison()
    print("\n[OK] All 10 slide images successfully created in docs/picture/")


if __name__ == "__main__":
    main()
