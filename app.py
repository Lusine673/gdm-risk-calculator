import streamlit as st
import numpy as np
import pandas as pd

# --- CONFIG ---
st.set_page_config(page_title="Модель ГСД", page_icon="🧪", layout="centered")

# --- СТИЛЬ ---
st.markdown("""
<style>
h3 {
  text-align: center;
  font-size: 26px;
  font-weight: 700;
  margin-bottom: 1.5rem;
}
div.stButton > button {
  background: linear-gradient(90deg, #109C90, #14b8a6);
  color: white;
  border: 0;
  border-radius: 10px;
  padding: 0.65rem 1.05rem;
  font-size: 16px;
  font-weight: 600;
}
div.stButton > button:hover {
  filter: brightness(1.05);
}
.risk-low {
  background: #e8f5e9;
  color: #1b5e20;
  padding: 14px;
  font-size: 20px;
  font-weight: 800;
  border-radius: 12px;
  text-align: center;
}
.risk-mid {
  background: #fff8e1;
  color: #ff6f00;
  padding: 14px;
  font-size: 20px;
  font-weight: 800;
  border-radius: 12px;
  text-align: center;
}
.risk-high {
  background: #ffebee;
  color: #b71c1c;
  padding: 14px;
  font-size: 20px;
  font-weight: 800;
  border-radius: 12px;
  text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- ХЕЛПЕРЫ ---
def logistic(z): return 1 / (1 + np.exp(-z))
def extended_category(p, low, high):
    if p < low: return "Низкий"
    elif p < high: return "Промежуточный"
    else: return "Высокий"
def color_class(cat):
    return {"Низкий": "risk-low", "Промежуточный": "risk-mid", "Высокий": "risk-high"}.get(cat, "risk-mid")
def combine_categories(*cats):
    order = {"Низкий": 0, "Промежуточный": 1, "Высокий": 2}
    return max(filter(None, cats), key=lambda x: order.get(x, 0))
def parse_num(s):
    try: return float(str(s).replace(",", ".").strip())
    except: return None

# --- МОДЕЛИ ---
BASE_COEF = {"beta0": -2.8830, "bmi": 0.1043, "fam_dm": 0.8860}

def base_risk(bmi, fam_dm):
    return logistic(BASE_COEF["beta0"] + BASE_COEF["bmi"] * bmi + BASE_COEF["fam_dm"] * fam_dm)

def lipid_risk(tg, hdl):
    return logistic(-2.837 + 2.431 * tg - 1.323 * hdl)

COEFFS = {
    "Tyrosine": 2.33, "AlphaAminoadipicAcid": 0.96, "MH3": 1.13,
    "Phosphoethanolamine": -2.89, "Phosphoserine": -2.48
}
FEATURES = list(COEFFS.keys())
TRAIN_RAW = {
    "Tyrosine": [45.31, 23.43, 15.03, 20.80, 19.32, 9.58, 12.33, 10.53, 10.17, 14.52],
    "AlphaAminoadipicAcid": [0.95, 1.12, 1.02, 1.08, 1.00, 2.40, 2.20, 2.50, 2.10, 2.35],
    "MH3": [46.08, 35.35, 39.78, 15.54, 36.53, 11.93, 10.46, 15.63, 16.70, 11.75],
    "Phosphoethanolamine": [1.08, 0.55, 1.26, 1.09, 0.69, 2.93, 1.91, 1.91, 1.88, 2.20],
    "Phosphoserine": [1.68, 2.12, 0.67, 0.53, 0.35, 8.00, 3.78, 3.59, 1.80, 2.52]
}
MEAN_LOG, SD_LOG = {}, {}
for k, v in TRAIN_RAW.items():
    xlog = np.log10(v)
    MEAN_LOG[k] = np.mean(xlog)
    SD_LOG[k] = np.std(xlog, ddof=1)

def normalize_raw_df(df_raw):
    z = pd.DataFrame()
    for k in FEATURES:
        z[k] = (np.log10(df_raw[k]) - MEAN_LOG[k]) / np.sqrt(SD_LOG[k])
    return z

def meta_predict(df_norm):
    X = df_norm[FEATURES].values[0]
    beta = np.array([COEFFS[k] for k in FEATURES])
    return logistic(np.dot(X, beta))

# --- Заголовки ---
st.markdown("<h3>Комплексная модель прогнозирования осложнений<br/>при наличии гестационного сахарного диабета</h3>", unsafe_allow_html=True)

# --- ВКЛАДКИ ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Раздел 1. Стратификация по риску ГСД",
    "Раздел 2. Прогноз осложнений при ГСД",
    "Раздел 3. Метаболомный профиль",
    "Итог"
])

# --- TAB 1 ---
with tab1:
    st.markdown("Введите клинико‑анамнестические данные:")
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input("ИМТ (кг/м²)", min_value=14.0, max_value=60.0, value=27.0)
    with col2:
        fam_dm_label = st.radio("СД у родственников первой линии", ["Нет", "Да"], horizontal=True)
        fam_dm = 1 if fam_dm_label == "Да" else 0

    if st.button("Рассчитать базовый риск"):
        p = base_risk(bmi, fam_dm)
        cat = extended_category(p, low=0.388, high=0.607)
        st.session_state["base_p"] = p
        st.session_state["base_cat"] = cat
        st.markdown(f"<div class='{color_class(cat)}'>{cat} базовый риск ({p*100:.1f}%)</div>", unsafe_allow_html=True)

# --- TAB 2 ---
with tab2:
    st.markdown("Введите показатели липидного профиля:")
    col1, col2 = st.columns(2)
    with col1:
        tg = st.number_input("Триглицериды, ммоль/л", 0.1, 20.0, 2.0)
    with col2:
        hdl = st.number_input("ЛПВП, ммоль/л", 0.1, 5.0, 1.2)

    if st.button("Рассчитать риск осложнений"):
        p_lip = lipid_risk(tg, hdl)
        cat = extended_category(p_lip, 0.35, 0.689)
        st.session_state["lipid_p"] = p_lip
        st.session_state["lipid_cat"] = cat
        st.markdown(f"<div class='{color_class(cat)}'>{cat} риск осложнений ({p_lip*100:.1f}%)</div>", unsafe_allow_html=True)

# --- TAB 3 ---
with tab3:
    st.markdown("Введите уровни аминокислот (ммоль/моль креатинина):")
    col1, col2 = st.columns(2)
    with col1:
        tyrosine = parse_num(st.text_input("Тирозин"))
        alphaaaa = parse_num(st.text_input("α‑Аминоадипиновая кислота"))
        pe = parse_num(st.text_input("Фосфоэтаноламин"))
    with col2:
        mh3 = parse_num(st.text_input("3‑Метилгистидин (MH3)"))
        ps = parse_num(st.text_input("Фосфосерин"))
    
    if st.button("Рассчитать метаболомный риск"):
        vals = {"Tyrosine": tyrosine, "AlphaAminoadipicAcid": alphaaaa, "MH3": mh3,
                "Phosphoethanolamine": pe, "Phosphoserine": ps}
        if any(v is None or v <= 0 for v in vals.values()):
            st.error("Введите корректные числовые значения > 0.")
        else:
            df = pd.DataFrame([vals])
            df_norm = normalize_raw_df(df)
            p_meta = meta_predict(df_norm)
            cat = extended_category(p_meta, 0.05, 0.1)
            st.session_state["meta_p"] = p_meta
            st.session_state["meta_cat"] = cat
            st.markdown(f"<div class='{color_class(cat)}'>{cat} риск по метаболомике ({p_meta*100:.1f}%)</div>", unsafe_allow_html=True)

# --- TAB 4 ---
with tab4:
    st.subheader("Итоговая категория")

    base_cat = st.session_state.get("base_cat")
    lipid_cat = st.session_state.get("lipid_cat")
    meta_cat = st.session_state.get("meta_cat")

    collected = [base_cat, lipid_cat, meta_cat]
    collected = [c for c in collected if c is not None]

    if collected:
        final_cat = combine_categories(*collected)
        html_class = color_class(final_cat)
        st.markdown(f"<div class='{html_class}'>Итоговая категория риска: {final_cat}</div>", unsafe_allow_html=True)
    else:
        st.info("Пожалуйста, рассчитайте хотя бы один компонент риска.")
