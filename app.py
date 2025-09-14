import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Модель раннего прогноза ГСД", page_icon="🧪", layout="centered")

# ====== Коэффициенты LASSO‑модели ======
BETA0 = 0.0
COEFFS = {
    "Tyrosine": 0.1279647860222218,
    "MH3": 0.8913890705856571,             # 3‑метилгистидин
    "Phosphoethanolamine": -0.8390359958954429,
    "Phosphoserine": -1.1553144548078098
}
THRESH = 0.382  # порог Юдена (для решения «высокий/низкий риск»)
FEATURES = list(COEFFS.keys())

# ====== TRAIN‑данные (ммоль/моль) для расчёта параметров предобработки ======
TRAIN_RAW = {
    "Tyrosine": [45.31, 23.43, 15.03, 20.80, 19.32, 9.58, 12.33, 10.53, 10.17, 14.52],
    "MH3": [46.08, 35.35, 39.78, 15.54, 36.53, 11.93, 10.46, 15.63, 16.70, 11.75],
    "Phosphoethanolamine": [1.08, 0.55, 1.26, 1.09, 0.69, 2.93, 1.91, 1.91, 1.88, 2.20],
    "Phosphoserine": [1.68, 2.12, 0.67, 0.53, 0.35, 8.00, 3.78, 3.59, 1.80, 2.52]
}

# ====== Параметры предобработки: log10 → Pareto ((x − mean_log10)/sqrt(sd_log10)) ======
MEAN_LOG, SD_LOG = {}, {}
for k, arr in TRAIN_RAW.items():
    x = np.array(arr, dtype=float)
    xlog = np.log10(x)
    MEAN_LOG[k] = float(np.mean(xlog))
    SD_LOG[k] = float(np.std(xlog, ddof=1))  # sample SD

def normalize_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Сырые значения (ммоль/моль) → log10 → Pareto с параметрами train."""
    z = pd.DataFrame(index=df_raw.index)
    for k in FEATURES:
        if k not in df_raw.columns:
            raise ValueError(f"Нет столбца: {k}")
        v = df_raw[k].astype(float)
        if (v <= 0).any():
            raise ValueError(f"{k}: значения должны быть > 0 (для log10)")
        xlog = np.log10(v)
        denom = np.sqrt(SD_LOG[k]) if SD_LOG[k] > 0 else 1.0
        z[k] = (xlog - MEAN_LOG[k]) / denom
    return z

def predict_proba_from_norm(df_norm: pd.DataFrame) -> np.ndarray:
    X = df_norm[FEATURES].astype(float).values
    beta = np.array([COEFFS[f] for f in FEATURES])
    logit = BETA0 + X @ beta
    return 1 / (1 + np.exp(-logit))

def parse_num(s: str) -> float | None:
    """Принимает числа с точкой или запятой, возвращает float или None."""
    if s is None: 
        return None
    try:
        return float(str(s).replace(",", "."))
    except:
        return None

# ====== Заголовок (по центру) ======
st.markdown(
    "<h3 style='text-align:center'>Модель раннего прогноза (первый триместр беременности) гестационного сахарного диабета</h3>",
    unsafe_allow_html=True
)

st.write("")  # небольшой отступ

# ====== Ввод значений (сырые, ммоль/моль) ======
st.markdown("Введите значения (ммоль/моль креатинина):")

col1, col2 = st.columns(2)
with col1:
    tyrosine_str = st.text_input("Тирозин", value="")
    pe_str = st.text_input("Фосфоэтаноламин", value="")
with col2:
    mh3_str = 
