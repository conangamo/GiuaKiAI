import os
import re
import html
import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.dataset import resolve_data_path, read_mabsa_file, format_target, normalize_english_spacing
from src.metrics import parse_uabsa_string, compute_metrics

# Page configuration
st.set_page_config(
    page_title="Hệ Thống Phân Tích Quan Điểm Theo Khía Cạnh (ABSA)",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Hệ Thống Phân Tích Quan Điểm Theo Khía Cạnh (ABSA)")
st.markdown("Đồ án: **Aspect-Based Sentiment Analysis với Multi-Task & Multi-Domain Learning**")

# Sidebar
st.sidebar.header("⚙️ Cấu Hình Hệ Thống")

device_str = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Thiết bị sử dụng: **{device_str.upper()}** ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

st.sidebar.info("Tác vụ hiện tại: **UABSA** — Trích xuất đồng thời cặp `(Khía cạnh, Cảm xúc)`")

def has_weights(path):
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "model.safetensors")) or
        os.path.exists(os.path.join(path, "pytorch_model.bin"))
    )

def discover_models():
    models = {}
    ckpt_root = "checkpoints"
    if os.path.isdir(ckpt_root):
        # Ưu tiên các thư mục con mới hơn (vd: baseline_e8_s42/best_model, wl_p03_e8_s42/best_model, ...)
        for entry in sorted(os.listdir(ckpt_root), reverse=True):
            sub_best = os.path.join(ckpt_root, entry, "best_model")
            if has_weights(sub_best):
                models[f"{entry}/best_model"] = sub_best
    if has_weights("checkpoints/best_model"):
        models["best_model (Baseline 3-epoch gốc)"] = "checkpoints/best_model"
    models["google/mt5-small (Pre-trained chưa fine-tune)"] = "google/mt5-small"
    return models

model_dict = discover_models()
selected_label = st.sidebar.selectbox("Mô hình suy luận:", list(model_dict.keys()))
model_name = model_dict[selected_label]

num_beams = st.sidebar.selectbox("Chế độ giải mã (Beam Search):", [1, 2, 4], index=0,
                                 help="1 = Greedy (giống evaluate_model.py), 2-4 = Beam Search")
use_norm_spacing = st.sidebar.checkbox("Chuẩn hóa dấu câu tiếng Anh (khớp SemEval)", value=True,
                                       help="Thêm khoảng trắng quanh dấu câu như dữ liệu train SemEval/M-ABSA")

@st.cache_resource
def load_absa_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model, device

def highlight_aspects_html(sentence, pairs):
    """Tô màu khía cạnh không phân biệt hoa/thường và an toàn HTML (không phá vỡ thẻ span)."""
    explicit = {}
    for aspect, polarity in pairs:
        if aspect.lower() != "null" and aspect.strip():
            explicit[aspect.lower()] = polarity
    if not explicit:
        return html.escape(sentence)

    # Sắp xếp khía cạnh dài trước để ưu tiên khớp cụm từ dài (vd: 'battery life' trước 'battery')
    sorted_terms = sorted(explicit.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?i)(" + "|".join(re.escape(t) for t in sorted_terms) + r")"
    )
    parts = []
    last = 0
    for m in pattern.finditer(sentence):
        parts.append(html.escape(sentence[last:m.start()]))
        matched_text = m.group(0)
        pol = explicit[matched_text.lower()]
        color = "#28a745" if pol == "positive" else ("#dc3545" if pol == "negative" else "#e0a800")
        parts.append(
            f"<span style='background-color: {color}; color: white; padding: 2px 6px; "
            f"border-radius: 4px; font-weight: bold;'>{html.escape(matched_text)} ({pol})</span>"
        )
        last = m.end()
    parts.append(html.escape(sentence[last:]))
    return "".join(parts)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Trực Quan Hóa Dự Đoán (Live Demo)", "📊 Khám Phá Dữ Liệu Đa Miền", "📈 Bảng Đánh Giá Hiệu Năng"])

# TAB 1: LIVE INFERENCE
with tab1:
    st.subheader("Trích xuất Khía cạnh & Phân loại Cảm xúc theo thời gian thực")

    sample_sentences = [
        "The food was absolutely amazing, but the service was terrible and slow.",
        "Great battery life and clear screen, though the keyboard feels a bit cheap.",
        "The hotel location is perfect, near the beach, but the room was quite small.",
        "Rooms are on the basic side, but they have everything that you'd need.",
        "Screen is average by 2017 standards.",
        "If you're in the area you shouldn't be disappointed.",
        "No backlit keyboard.",
    ]

    selected_sample = st.selectbox("Chọn câu ví dụ:", ["(Tự nhập câu đánh giá... )"] + sample_sentences)

    if selected_sample == "(Tự nhập câu đánh giá... )":
        user_input = st.text_area("Nhập văn bản đánh giá (Review):", "The coffee was delicious and fresh, but the price is too high.")
    else:
        user_input = st.text_area("Nhập văn bản đánh giá (Review):", selected_sample)

    if st.button("🚀 Phân Tích Quan Điểm", type="primary"):
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản cần phân tích!")
        else:
            with st.spinner("Đang nạp mô hình và phân tích câu..."):
                tokenizer, model, device = load_absa_model(model_name)
                model_input_text = normalize_english_spacing(user_input) if use_norm_spacing else user_input.strip()
                inputs = tokenizer(model_input_text, max_length=128, truncation=True, return_tensors="pt").to(device)
                gen_kwargs = dict(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=64,
                    num_beams=num_beams,
                )
                if num_beams > 1:
                    gen_kwargs["early_stopping"] = True
                with torch.no_grad():
                    outputs = model.generate(**gen_kwargs)
                raw_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.markdown("### Kết Quả Trích Xuất:")
            st.info(f"**Chuỗi dự đoán thô (Seq2Seq Output):** `{raw_prediction}`")

            pairs = sorted(parse_uabsa_string(raw_prediction))
            if not pairs:
                st.write("Không phát hiện khía cạnh nào mang cảm xúc trong câu.")
            else:
                cols = st.columns(min(len(pairs), 4))
                for idx, (aspect, polarity) in enumerate(pairs):
                    label_aspect = "KHÍA CẠNH ẨN (NULL)" if aspect == "null" else aspect.upper()
                    with cols[idx % len(cols)]:
                        if polarity == "positive":
                            st.success(f"**Khía cạnh:** {label_aspect}\n\n**Cảm xúc:** Tích cực 👍")
                        elif polarity == "negative":
                            st.error(f"**Khía cạnh:** {label_aspect}\n\n**Cảm xúc:** Tiêu cực 👎")
                        else:
                            st.warning(f"**Khía cạnh:** {label_aspect}\n\n**Cảm xúc:** Trung tính 😐")

                highlighted_html = highlight_aspects_html(user_input, pairs)
                st.markdown("### Trực quan hóa câu văn:")
                st.markdown(f"<div style='font-size: 18px; padding: 15px; border-left: 5px solid #007bff; background: #f8f9fa; color: #212529;'>{highlighted_html}</div>", unsafe_allow_html=True)


# TAB 2: DATASET EXPLORER
with tab2:
    st.subheader("Khám Phá Dữ Liệu Các Miền (M-ABSA Dataset)")
    
    col_domain, col_lang, col_split = st.columns(3)
    with col_domain:
        domain_raw = st.selectbox("Chọn miền (Domain):", ["multi_domain (Trộn Hotel + Laptop + Restaurant)", "restaurant", "hotel", "laptop", "phone", "food", "coursera", "sight"])
        domain = "multi_domain" if "multi_domain" in domain_raw else domain_raw
    with col_lang:
        lang = st.selectbox("Chọn ngôn ngữ (Language):", ["en", "vi", "zh", "fr", "de", "es", "ja"])
    with col_split:
        split = st.selectbox("Chọn tập (Split):", ["train", "dev", "test"])

    try:
        file_path = resolve_data_path(domain, split=split, lang=lang)
        sents, labels = read_mabsa_file(file_path)
        
        st.write(f"📁 Tệp dữ liệu: `{file_path}`")
        
        # Calculate stats
        total_sents = len(sents)
        total_triplets = sum(len(l) for l in labels)
        
        pol_counts = {"positive": 0, "negative": 0, "neutral": 0}
        aspect_list = []
        for l in labels:
            for tri in l:
                if len(tri) >= 3:
                    aspect_list.append(tri[0])
                    p = tri[2].lower()
                    if p in pol_counts:
                        pol_counts[p] += 1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng số câu", total_sents)
        c2.metric("Tổng số khía cạnh", total_triplets)
        c3.metric("Số khía cạnh duy nhất", len(set([a.lower() for a in aspect_list])))
        c4.metric("Tỷ lệ Tích cực / Tiêu cực", f"{pol_counts['positive']}/{pol_counts['negative']}")

        # Plot charts
        df_pol = pd.DataFrame(list(pol_counts.items()), columns=["Cảm xúc", "Số lượng"])
        fig_pie = px.pie(df_pol, values="Số lượng", names="Cảm xúc", color="Cảm xúc",
                         color_discrete_map={"positive": "#28a745", "negative": "#dc3545", "neutral": "#ffc107"},
                         title="Biểu đồ phân bố cảm xúc của miền")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Show samples table
        st.markdown("#### Xem 10 mẫu câu đầu tiên trong tập dữ liệu:")
        samples_df = []
        for i in range(min(10, len(sents))):
            samples_df.append({
                "STT": i + 1,
                "Câu đánh giá (Sentence)": sents[i],
                "Khía cạnh & Cảm xúc (Triplets)": str(labels[i])
            })
        st.dataframe(pd.DataFrame(samples_df), use_container_width=True)

    except Exception as e:
        st.error(f"Không thể nạp dữ liệu cho tùy chọn này: {e}")


# TAB 3: EVALUATION METRICS (TRUNG THỰC & ĐỘNG TỪ RESULTS/)
with tab3:
    st.subheader("📈 Bảng Đánh Giá Hiệu Năng Mô Hình (Đo lường trung thực)")
    
    # Quét tất cả file kết quả trong thư mục results/
    res_dir = "results"
    avail_runs = []
    if os.path.isdir(res_dir):
        for fname in sorted(os.listdir(res_dir)):
            if fname.endswith(".json") and fname != "data_analysis.json":
                avail_runs.append(fname)
                
    if not avail_runs:
        st.warning("Chưa tìm thấy tệp kết quả nào trong thư mục `results/`. Vui lòng chạy `src/evaluate_model.py` để xuất kết quả!")
    else:
        col_sel, col_info = st.columns([2, 3])
        with col_sel:
            selected_file = st.selectbox("Chọn kết quả kiểm thử cần xem:", avail_runs,
                                         index=avail_runs.index("legacy_baseline_test.json") if "legacy_baseline_test.json" in avail_runs else 0)
        with col_info:
            st.info(f"Đang hiển thị dữ liệu đo lường thực tế từ tệp: `{os.path.join(res_dir, selected_file)}`")

        import json
        with open(os.path.join(res_dir, selected_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Hàng thẻ chỉ số tổng quát
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Micro F1-Score", f"{data.get('f1', 0):.2f}%")
        c2.metric("Macro F1-Score", f"{data.get('macro_f1', 0):.2f}%",
                  help="Trung bình cộng F1 của cả 3 lớp: positive, negative, neutral")
        c3.metric("Precision", f"{data.get('precision', 0):.2f}%")
        c4.metric("Recall", f"{data.get('recall', 0):.2f}%")
        c5.metric("Polarity Acc (Aspect đúng)", f"{data.get('polarity_acc_on_matched_aspects', 0):.2f}%")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        # Bảng chi tiết theo từng lớp cảm xúc
        with col_left:
            st.markdown("#### 🎯 Hiệu năng theo từng lớp cảm xúc (Per-Class Metrics)")
            per_class = data.get("per_class", {})
            class_rows = []
            for cls_name in ["positive", "negative", "neutral"]:
                c_stats = per_class.get(cls_name, {})
                class_rows.append({
                    "Lớp cảm xúc": cls_name.upper(),
                    "Precision (%)": f"{c_stats.get('precision', 0):.2f}",
                    "Recall (%)": f"{c_stats.get('recall', 0):.2f}",
                    "F1-Score (%)": f"{c_stats.get('f1', 0):.2f}",
                    "Số mẫu thực (Gold)": c_stats.get("n_gold", 0),
                    "Số mô hình đoán (Pred)": c_stats.get("n_pred", 0),
                    "Đoán đúng (TP)": c_stats.get("tp", 0)
                })
            df_class = pd.DataFrame(class_rows)
            st.dataframe(df_class, use_container_width=True)

            if "neutral" in per_class and per_class["neutral"].get("f1", 0) == 0:
                st.warning("⚠️ **Phát hiện hiện tượng sụp đổ lớp trung tính (Neutral Collapse):** Mô hình baseline hoàn toàn không dự đoán được lớp Neutral (Recall = 0.0%), khiến Macro-F1 bị kéo xuống đáng kể!")

        # Bảng chi tiết theo miền dữ liệu
        with col_right:
            st.markdown("#### 🏢 Hiệu năng phân rã theo từng miền (Domain Breakdown)")
            per_domain = data.get("per_domain", {})
            dom_rows = []
            for d_name, d_stats in per_domain.items():
                neu_f1 = d_stats.get("per_class", {}).get("neutral", {}).get("f1", d_stats.get("neutral_f1", 0))
                dom_rows.append({
                    "Miền (Domain)": d_name.capitalize(),
                    "Số câu": d_stats.get("n_sentences", 0),
                    "Precision (%)": f"{d_stats.get('precision', 0):.2f}",
                    "Recall (%)": f"{d_stats.get('recall', 0):.2f}",
                    "Micro F1 (%)": f"{d_stats.get('f1', 0):.2f}",
                    "Macro F1 (%)": f"{d_stats.get('macro_f1', 0):.2f}",
                    "Neutral F1 (%)": f"{neu_f1:.2f}"
                })
            if dom_rows:
                df_dom = pd.DataFrame(dom_rows)
                st.dataframe(df_dom, use_container_width=True)
            else:
                st.write("Không có dữ liệu phân rã theo miền.")

        # Ma trận nhầm lẫn
        conf_mat = data.get("polarity_confusion(gold->pred)", {})
        if conf_mat:
            st.markdown("#### 🔀 Ma trận nhầm lẫn cảm xúc (Polarity Confusion Matrix: Gold ➔ Pred)")
            st.caption("Chỉ xét trên các khía cạnh mà mô hình đã trích xuất đúng vị trí:")
            conf_df = pd.DataFrame(conf_mat).T
            conf_df.columns = [f"Pred: {c}" for c in conf_df.columns]
            conf_df.index = [f"Gold: {idx}" for idx in conf_df.index]
            st.dataframe(conf_df, use_container_width=True)

        # Xem dự đoán chi tiết từ TSV nếu có
        tsv_name = selected_file.replace(".json", "_predictions.tsv")
        tsv_path = os.path.join(res_dir, tsv_name)
        if os.path.exists(tsv_path):
            st.markdown("---")
            st.markdown("#### 🔬 Phân tích mẫu dự đoán thực tế (Error & Case Analysis)")
            df_preds = pd.read_csv(tsv_path, sep="\t")
            pred_col = "prediction" if "prediction" in df_preds.columns else "pred"
            
            filter_opt = st.radio(
                "Bộ lọc mẫu:",
                [
                    "Tất cả mẫu",
                    "Chỉ các mẫu sai (Gold != Pred)",
                    "Chỉ các mẫu chứa nhãn Neutral trong Gold",
                    "Các mẫu Neutral được mô hình dự đoán đúng"
                ],
                horizontal=True
            )
            if filter_opt == "Chỉ các mẫu sai (Gold != Pred)":
                df_preds = df_preds[df_preds["gold"] != df_preds[pred_col]]
            elif filter_opt == "Chỉ các mẫu chứa nhãn Neutral trong Gold":
                df_preds = df_preds[df_preds["gold"].str.contains("neutral", na=False)]
            elif filter_opt == "Các mẫu Neutral được mô hình dự đoán đúng":
                df_preds = df_preds[
                    df_preds["gold"].str.contains("neutral", na=False)
                    & df_preds[pred_col].str.contains("neutral", na=False)
                ]
                
            st.caption(f"Hiển thị {min(50, len(df_preds))} / {len(df_preds)} mẫu phù hợp:")
            st.dataframe(df_preds.head(50), use_container_width=True)


