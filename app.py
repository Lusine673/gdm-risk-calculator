import streamlit as st
import numpy as np
import pandas as pd

# ---------------------- СТИЛИ ----------------------
st.set_page_config(page_title="Модель прогноза осложнений при ГСД", page_icon="🧪", layout="centered")

st.markdown("""
<style>
:root{
  --ok:#1b5e20;
  --okbg:#e8f5e9;
  --mid:#ff6f00;
  --midbg:#fff8e1;
  --warn:#b71c1c;
  --warnbg:#ffebee;
}
.card{
  background:#ffffff;
  border:1px solid #e9eef2;
  border-radius:14px;
  padding:1.2rem 1rem;
  box-shadow:0 2px 10px rgba(0,0,0,.04);
}
.risk-low{
  background:var(--okbg); color:var(--ok);
  font-size:18px; font-weight:700;
  text-align:center; padding:1rem; border-radius:10px;
}
.risk-mid{
  background:var(--midbg); color:var(--mid);
  font-size:18px; font-weight:700;
  text-align:center; padding:1rem; border-radius:10px;
}
.risk-high{
  background:var(--warnbg); color:var(--warn);
  font-size:18px; font-weight:700;
  text-align:center; padding:1rem; border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- УТИЛИТЫ ----------------------

def extended_category(p, low, high):
    if p < low:
        return "Низкий"
    elif p < high:
        return "Промежуточный"
    else:
        return "Высокий"

def color_class(category):
    return {
        "Низкий": "risk-low",
        "Промежуточный": "risk-mid",
        "Высокий": "risk-high"
    }.get(category, "risk-mid")

def logistic(z): return 1 / (1 + np.exp(-z))
def parse_num(s):
    try: return float(str(s).replace(",", "."))
    except: return None

# ---------------------- 1. КЛИНИЧЕСКАЯ МОДЕЛЬ ----------------------
BASE_COEF = {"beta0": -2.8830, "bmi": 0.1043, "fam_dm": 0.8860}
def base_risk(bmi, fam_dm): return logistic(BASE_COEF["beta0"] + BASE_COEF["bmi"] * bmi + BASE_COEF["fam_dm"] * fam_dm)

# ---------------------- 2. ЛИПИДНАЯ МОДЕЛЬ ----------------------
def lipid_risk(tg, hdl): return logistic(-2.837 + 2.431 * tg - 1.323 * hdl)

# ---------------------- 3. МЕТАБОЛОМНАЯ МОДЕЛЬ ----------------------
COEFFS = {
    "Tyrosine": 2.33, "AlphaAminoadipicAcid": 0.96, "MH3": 1.13,
    "Phosphoethanolamine": -2.89, "Phosphoserine": -2.48
}
BETA0 = 0.0
FEATURES = list(COEFFS.keys())
TRAIN_RAW = {
    "Tyrosine": [45.31, 23.43, 15.03, 20.80, 19.32, 9.58, 12.33, 10.53, 10.17, 14.52],
    "AlphaAminoadipicAcid": [0.95, 1.12, 1.02, 1.08, 1.00, 2.40, 2.20, 2.50, 2.10, 2.35],
    "MH3": [46.08, 35.35, 39.78, 15.54, 36.53, 11.93, 10.46, 15.63, 16.70, 11.75],
    "Phosphoethanolamine": [1.08, 0.55, 1.26, 1.09, 0.69, 2.93, 1.91, 1.91, 1.88, 2.20],
    "Phosphoserine": [1.68, 2.12, 0.67, 0.53, 0.35, 8.00, 3.78, 3.59, 1.80, 2.52]
}
MEAN_LOG, SD_LOG = {}, {}
for k, arr in TRAIN_RAW.items():
    xlog = np.log10(np.array(arr))
    MEAN_LOG[k] = float(np.mean(xlog))
    SD_LOG[k]  = float(np.std(xlog, ddof=1))

def normalize_raw(df_raw):
    z = pd.DataFrame()
    for k in FEATURES:
        xlog = np.log10(df_raw[k])
        z[k] = (xlog - MEAN_LOG[k]) / np.sqrt(SD_LOG[k])
    return z

def meta_predict(df_norm):
    beta = np.array([COEFFS[k] for k in FEATURES])
    X = df_norm[FEATURES].values
    z = BETA0 + X @ beta
    return logistic(z)

# ---------------------- РАЗДЕЛЫ ----------------------

tab1, tab2, tab3 = st.tabs(["1. Клинический риск", "2. Биохимический риск", "3. Метаболомный риск"])

# ---------------------- TAB 1 ----------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Клинико‑анамнестические данные")
    c1, c2 = st.columns(2)
    with c1:
        bmi = st.number_input("ИМТ (кг/м²)", 14.0, 60.0, 27.0)
    with c2:
        fam_dm = st.radio("СД у родственников 1-й линии", ["Нет", "Да"]) == "Да"

    if st.button("Рассчитать", key="btn_base"):
        p = base_risk(bmi, int(fam_dm))
        cat = extended_category(p, low=0.388, high=0.607)
        st.session_state["base_p"], st.session_state["base_cat"] = p, cat
        st.markdown(f"<div class='{color_class(cat)}'>{cat} риск ({p*100:.1f}%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- TAB 2 ----------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Липидный профиль")
    c1, c2 = st.columns(2)
    with c1:
        tg = st.number_input("Триглицериды, ммоль/л", 0.1, 20.0, 2.0)
    with c2:
        hdl = st.number_input("ЛПВП, ммоль/л", 0.1, 5.0, 1.2)

    if st.button("Рассчитать", key="btn_lipid"):
        p = lipid_risk(tg, hdl)
        cat = extended_category(p, low=0.35, high=0.689)
        st.session_state["lipid_p"], st.session_state["lipid_cat"] = p, cat
        st.markdown(f"<div class='{color_class(cat)}'>{cat} риск осложнений ({p*100:.1f}%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- TAB 3 ----------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Уровни аминокислот")
    inputs = {}
    for k in FEATURES:
        inputs[k] = parse_num(st.text_input(k, value=""))

    if st.button("Рассчитать", key="btn_meta"):
        if any(v is None or v <= 0 for v in inputs.values()):
            st.error("Все значения должны быть корректно введены (> 0).")
        else:
            try:
                df_raw = pd.DataFrame([inputs])
                df_norm = normalize_raw(df_raw)
                p = float(meta_predict(df_norm))
                cat = extended_category(p, low=0.05, high=0.1)
                st.session_state["meta_p"], st.session_state["meta_cat"] = p, cat
                st.markdown(f"<div class='{color_class(cat)}'>{cat} риск по метаболомике ({p*100:.1f}%)</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Ошибка расчёта: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
