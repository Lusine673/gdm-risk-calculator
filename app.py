import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Модель раннего прогноза ГСД", page_icon="🧪", layout="centered")

# ===== Коэффициенты LASSO‑модели =====
BETA0 = 0.0
COEFFS = {
    "Tyrosine": 0.1279647860222218,
    "MH3": 0.8913890705856571,             # 3-метилгистидин
    "Phosphoethanolamine": -0.8390359958954429,
    "Phosphoserine": -1.1553144548078098
}
THRESH = 0.382  # порог Юдена
FEATURES = list(COEFFS.keys())

# ===== TRAIN-данные (ммоль/моль) для расчета параметров предобработки =====
TRAIN_RAW = {
    "Tyrosine": [45.31, 23.43, 15.03, 20.80, 19.32, 9.58, 12.33, 10.53, 10.17, 14.52],
    "MH3": [46.08, 35.35, 39.78, 15.54, 36.53, 11.93, 10.46, 15.63, 16.70, 11.75],
    "Phosphoethanolamine": [1.08, 0.55, 1.26, 1.09, 0.69, 2.93, 1.91, 1.91, 1.88, 2.20],
    "Phosphoserine": [1.68, 2.12, 0.67, 0.53, 0.35, 8.00, 3.78, 3.59, 1.80, 2.52]
}

# ===== Параметры предобработки: log10 -> Pareto ((x - mean_log10)/sqrt(sd_log10)) =====
MEAN_LOG, SD_LOG = {}, {}
for k, arr in TRAIN_RAW.items():
    x = np.array(arr, dtype=float)
    xlog = np.log10(x)
    MEAN_LOG[k] = float(np.mean(xlog))
    SD_LOG[k] = float(np.std(xlog, ddof=1))  # sample SD

def normalize_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Сырые значения (ммоль/моль) -> log10 -> Pareto с параметрами train."""
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

def parse_num(s):
    """Принимает числа с точкой или запятой, возвращает float или None."""
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return None

# ===== Заголовок по центру =====
st.markdown(
    "<h3 style='text-align:center'>Модель раннего прогноза (первый триместр беременности) гестационного сахарного диабета</h3>",
    unsafe_allow_html=True
)
st.write("")

# ===== Ввод сырых значений (ммоль/моль креатинина) =====
st.markdown("Введите значения (ммоль/моль креатинина):")

col1, col2 = st.columns(2)
with col1:
    tyrosine_str = st.text_input("Тирозин (ммоль/моль)", value="")
    pe_str = st.text_input("Фосфоэтаноламин (ммоль/моль)", value="")
with col2:
    mh3_str = st.text_input("3-метилгистидин (MH3) (ммоль/моль)", value="")
    ps_str = st.text_input("Фосфосерин (ммоль/моль)", value="")

calc = st.button("Рассчитать риск")

if calc:
    tyrosine = parse_num(tyrosine_str)
    mh3 = parse_num(mh3_str)
    pe = parse_num(pe_str)
    ps = parse_num(ps_str)

    # Валидация
    errors = []
    for name, val in [("Тирозин", tyrosine), ("3-метилгистидин (MH3)", mh3),
                      ("Фосфоэтаноламин", pe), ("Фосфосерин", ps)]:
        if val is None:
            errors.append(f"{name}: введите число (допускается запятая или точка).")
        elif val <= 0:
            errors.append(f"{name}: значение должно быть > 0.")
    if errors:
        for e in errors:
            st.error(e)
    else:
        df_raw = pd.DataFrame([{
            "Tyrosine": tyrosine,
            "MH3": mh3,
            "Phosphoethanolamine": pe,
            "Phosphoserine": ps
        }])
        try:
            df_norm = normalize_raw_df(df_raw)
            p = float(predict_proba_from_norm(df_norm)[0])
            high_risk = p >= THRESH

            # Цветной индикатор без чисел
            if high_risk:
                st.markdown(
                    "<div style='background:#fdecea;color:#611a15;padding:14px;border-radius:10px;"
                    "text-align:center;font-size:20px;'><b>Высокий риск</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='background:#e8f5e9;color:#1b5e20;padding:14px;border-radius:10px;"
                    "text-align:center;font-size:20px;'><b>Низкий риск</b></div>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(str(e))

st.markdown("---")
with st.expander("Дисклеймер"):
    st.markdown("""
- Исследовательский прототип. Не является медицинским изделием.
- Модель: регуляризованная логистическая регрессия (LASSO), обучена на пилотной выборке n=10 (5 случаев ГСД, 5 контролей).
- Ввод: сырые концентрации в моче (ммоль/моль креатинина); внутри автоматически применяется предобработка, совпадающая с обучением (log10 → Pareto).
- Порог решения фиксирован (индекс Юдена по LOO‑валидации). Требуется внешняя валидация на независимых данных.
    """)
