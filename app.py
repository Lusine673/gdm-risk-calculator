import streamlit as st
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="Калькулятор риска ГСД", page_icon="🧪", layout="centered")

# === Коэффициенты LASSO‑модели (ваши) ===
BETA0 = 0.0
COEFFS = {
    "Tyrosine": 0.1279647860222218,
    "MH3": 0.8913890705856571,             # 3‑метилгистидин
    "Phosphoethanolamine": -0.8390359958954429,
    "Phosphoserine": -1.1553144548078098
}
THRESH = 0.382  # порог Юдена по LOO
FEATURES = list(COEFFS.keys())

# === Сырые TRAIN‑данные (ммоль/моль креатинина) для вычисления параметров предобработки ===
TRAIN_RAW = {
    "Tyrosine": [45.31, 23.43, 15.03, 20.80, 19.32, 9.58, 12.33, 10.53, 10.17, 14.52],
    "MH3": [46.08, 35.35, 39.78, 15.54, 36.53, 11.93, 10.46, 15.63, 16.70, 11.75],
    "Phosphoethanolamine": [1.08, 0.55, 1.26, 1.09, 0.69, 2.93, 1.91, 1.91, 1.88, 2.20],
    "Phosphoserine": [1.68, 2.12, 0.67, 0.53, 0.35, 8.00, 3.78, 3.59, 1.80, 2.52]
}

# === Параметры предобработки: log10 → Pareto ((x − mean_log10)/sqrt(sd_log10)) ===
MEAN_LOG, SD_LOG = {}, {}
for k, arr in TRAIN_RAW.items():
    x = np.array(arr, dtype=float)
    xlog = np.log10(x)
    MEAN_LOG[k] = float(np.mean(xlog))
    SD_LOG[k] = float(np.std(xlog, ddof=1))  # sample SD (ddof=1)

def normalize_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Сырые значения (ммоль/моль) → log10 → Pareto с параметрами train."""
    z = pd.DataFrame(index=df_raw.index)
    for k in FEATURES:
        if k not in df_raw.columns:
            raise ValueError(f"Нет столбца: {k}")
        v = df_raw[k].astype(float)
        if (v <= 0).any():
            raise ValueError(f"{k}: все значения должны быть > 0 (для log10)")
        xlog = np.log10(v)
        denom = np.sqrt(SD_LOG[k]) if SD_LOG[k] > 0 else 1.0
        z[k] = (xlog - MEAN_LOG[k]) / denom
    return z

def predict_proba_from_norm(df_norm: pd.DataFrame) -> np.ndarray:
    """Считает вероятность по нормализованным признакам."""
    missing = [c for c in FEATURES if c not in df_norm.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {missing}")
    X = df_norm[FEATURES].astype(float).values
    beta = np.array([COEFFS[f] for f in FEATURES])
    logit = BETA0 + X @ beta
    return 1 / (1 + np.exp(-logit))

st.title("Калькулятор риска ГСД (пилотная модель)")
st.caption("LASSO‑логистическая регрессия. Поддерживается ввод сырых значений (ммоль/моль креатинина) и нормализованных (log10 + Pareto).")

tabs = st.tabs([
    "Один пациент (RAW)",
    "Пакет (RAW CSV)",
    "Нормализованные",
    "Шаблоны CSV",
    "QR‑код",
    "Параметры предобработки"
])

# --- Один пациент: RAW ---
with tabs[0]:
    st.subheader("Один пациент — ввод сырых значений (ммоль/моль креатинина)")
    c1, c2 = st.columns(2)
    with c1:
        tyrosine = st.number_input("Tyrosine (ммоль/моль)", min_value=0.0001, value=10.0, step=0.01, format="%.4f")
        pe = st.number_input("Phosphoethanolamine (ммоль/моль)", min_value=0.0001, value=1.00, step=0.01, format="%.4f")
    with c2:
        mh3 = st.number_input("MH3 / 3‑метилгистидин (ммоль/моль)", min_value=0.0001, value=12.0, step=0.01, format="%.4f")
        ps = st.number_input("Phosphoserine (ммоль/моль)", min_value=0.0001, value=2.00, step=0.01, format="%.4f")

    if st.button("Рассчитать (RAW)"):
        df_raw = pd.DataFrame([{
            "Tyrosine": tyrosine,
            "MH3": mh3,
            "Phosphoethanolamine": pe,
            "Phosphoserine": ps
        }])
        try:
            df_norm = normalize_raw_df(df_raw)
            p = float(predict_proba_from_norm(df_norm)[0])
            label = "ГСД" if p >= THRESH else "Контроль"
            st.metric("Вероятность ГСД", f"{p:.3f}")
            st.write(f"Классификация при пороге {THRESH:.3f}: **{label}**")
            with st.expander("Промежуточные нормализованные значения (log10 + Pareto)"):
                st.dataframe(df_norm)
        except Exception as e:
            st.error(str(e))

# --- Пакет: RAW CSV ---
def read_flexible_csv(uploaded):
    """Пытается прочитать CSV с разными разделителями/десятичными."""
    try:
        uploaded.seek(0)
        return pd.read_csv(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=";")
        except Exception:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=";", decimal=",")

with tabs[1]:
    st.subheader("Пакетная оценка — CSV c сырыми значениями (ммоль/моль креатинина)")
    st.write("Требуемые столбцы: Tyrosine, MH3, Phosphoethanolamine, Phosphoserine")
    file = st.file_uploader("Загрузите CSV", type=["csv"])
    if file is not None:
        try:
            df_raw = read_flexible_csv(file)
            df_norm = normalize_raw_df(df_raw)
            probs = predict_proba_from_norm(df_norm)
            out = df_raw.copy()
            out["P_GDM"] = np.round(probs, 4)
            out["Class_at_threshold"] = np.where(out["P_GDM"] >= THRESH, "GDM", "Control")
            st.dataframe(out, use_container_width=True)
            st.download_button("Скачать результаты (CSV)",
                               out.to_csv(index=False).encode("utf-8"),
                               "gdm_predictions.csv", "text/csv")
        except Exception as e:
            st.error(str(e))

# --- Нормализованные ---
with tabs[2]:
    st.subheader("Нормализованные значения (log10 + Pareto)")
    c1, c2 = st.columns(2)
    with c1:
        tyrosine_n = st.number_input("Tyrosine (норм.)", value=0.0, step=0.01, format="%.3f")
        pe_n = st.number_input("Phosphoethanolamine (норм.)", value=0.0, step=0.01, format="%.3f")
    with c2:
        mh3_n = st.number_input("MH3 (норм.)", value=0.0, step=0.01, format="%.3f")
        ps_n = st.number_input("Phosphoserine (норм.)", value=0.0, step=0.01, format="%.3f")
    if st.button("Рассчитать (норм.)"):
        df_norm = pd.DataFrame([{
            "Tyrosine": tyrosine_n,
            "MH3": mh3_n,
            "Phosphoethanolamine": pe_n,
            "Phosphoserine": ps_n
        }])
        p = float(predict_proba_from_norm(df_norm)[0])
        label = "ГСД" if p >= THRESH else "Контроль"
        st.metric("Вероятность ГСД", f"{p:.3f}")
        st.write(f"Классификация при пороге {THRESH:.3f}: **{label}**")

# --- Шаблоны CSV ---
with tabs[3]:
    st.subheader("Шаблоны CSV")
    raw_tmpl = pd.DataFrame(columns=FEATURES)
    st.download_button("Скачать шаблон RAW (ммоль/моль)",
                       raw_tmpl.to_csv(index=False).encode("utf-8"),
                       "template_raw.csv", "text/csv")
    norm_tmpl = pd.DataFrame(columns=FEATURES)
    st.download_button("Скачать шаблон нормализованных значений",
                       norm_tmpl.to_csv(index=False).encode("utf-8"),
                       "template_normalized.csv", "text/csv")
    st.caption("Если CSV из Excel с запятой как десятичным разделителем — сохраните как CSV (разделители - запятая) или загрузите как есть; парсер попытается догадаться.")

# --- QR‑код ---
with tabs[4]:
    st.subheader("QR‑код на ссылку приложения")
    url = st.text_input("Вставьте URL приложения после деплоя (https://…streamlit.app)")
    if url:
        try:
            import qrcode
            img = qrcode.make(url)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption="QR‑код")
            st.download_button("Скачать QR (PNG)", buf.getvalue(), "qr_gdm_calculator.png", "image/png")
        except Exception:
            st.info("Если QR не сгенерировался, проверьте requirements.txt (qrcode[pil]).")

# --- Параметры предобработки ---
with tabs[5]:
    st.subheader("Параметры предобработки (на обучающих 10 образцах)")
    rows = []
    for k in FEATURES:
        rows.append([k, MEAN_LOG[k], SD_LOG[k]])
    st.dataframe(pd.DataFrame(rows, columns=["Маркер", "mean_log10", "sd_log10"]), use_container_width=True)

st.markdown("---")
with st.expander("Дисклеймер"):
    st.markdown("""
- Исследовательский прототип. Не является медицинским изделием.
- RAW‑ввод: применяется предобработка, совпадающая с обучением (логарифмирование log10, Pareto‑масштабирование).
- Модель обучена на n=10 (5/5); метрики — пилотные. Требуется внешняя валидация перед клиническим использованием.
    """)
