import streamlit as st
import numpy as np
import pandas as pd

# Настройка страницы
st.set_page_config(page_title="Модель раннего прогноза ГСД", page_icon="🧪", layout="centered")

# ----------- CSS стили -----------
st.markdown("""
<style>
:root{
  --ok:#1b5e20;
  --okbg:#e8f5e9;
  --warn:#b71c1c;
  --warnbg:#ffebee;
  --mid:#ff6f00;
  --midbg:#fff8e1;
}
.block-container {padding-top:2rem;}
.risk-high {
  background:var(--warnbg); color:var(--warn); font-size:20px; font-weight:800;
  text-align:center; padding:14px; border-radius:12px;
}
.risk-low {
  background:var(--okbg); color:var(--ok); font-size:20px; font-weight:800;
  text-align:center; padding:14px; border-radius:12px;
}
.risk-mid {
  background:var(--midbg); color:var(--mid); font-size:20px; font-weight:800;
  text-align:center; padding:14px; border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# ----------- Утилиты -----------
def logistic(z): return 1 / (1 + np.exp(-z))

def extended_category(p: float, low: float, high: float) -> str:
    if p < low:
        return "Низкий"
    elif p < high:
        return "Промежуточный"
    else:
        return "Высокий"

def color_class(cat: str) -> str:
    if cat == "Высокий":
        return "risk-high"
    elif cat == "Промежуточный":
        return "risk-mid"
    else:
        return "risk-low"

def combine_categories(*cats):
    order = {"Низкий": 0, "Промежуточный": 1, "Высокий": 2}
    return max(cats, key=lambda x: order.get(x, 0))

def parse_num(s):  # для ручного ввода пользовательских чисел
    try:
        return float(str(s).replace(",", "."))
    except:
        return None

# ----------- Блоки моделей -----------
BASE_COEF = {"beta0": -2.8830, "bmi": 0.1043, "fam_dm": 0.8860}
META_THRESH = 0.1
COEFFS = {
    "Tyrosine": 2.33,
    "AlphaAminoadipicAcid": 0.96,
    "MH3": 1.13,
    "Phosphoethanolamine": -2.89,
    "Phosphoserine": -2.48
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
for k, values in TRAIN_RAW.items():
    xlog = np.log10(values)
    MEAN_LOG[k] = np.mean(xlog)
    SD_LOG[k] = np.std(xlog, ddof=1)

def base_risk(bmi, fam_dm): return logistic(BASE_COEF["beta0"] + BASE_COEF["bmi"] * bmi + BASE_COEF["fam_dm"] * fam_dm)

def base_category(p): return extended_category(p, low=0.388, high=0.607)

def lipid_risk(tg, hdl): return logistic(-2.837 + 2.431 * tg - 1.323 * hdl)

def normalize_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=df_raw.index)
    for k in FEATURES:
        xlog = np.log10(df_raw[k])
        z[k] = (xlog - MEAN_LOG[k]) / np.sqrt(SD_LOG[k])
    return z

def meta_predict(df_norm: pd.DataFrame) -> float:
    X = df_norm[FEATURES].values
    beta = np.array([COEFFS[k] for k in FEATURES])
    logit = np.dot(X, beta)
    return logistic(logit[0])

# Заголовок
st.markdown(
    "<h3 style='text-align:center'>"
    "Модель раннего прогноза<br/>"
    "(первый триместр беременности)<br/>"
    "гестационного сахарного диабета"
    "</h3>", unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align:center'>"
    "Комплексная модель прогнозирования осложнений<br/>"
    "при наличии гестационного сахарного диабета"
    "</h3>", unsafe_allow_html=True
)

# Вкладки
tab1, tab2, tab3, tab4 = st.tabs([
    "Раздел 1. Стратификация по риску ГСД",
    "Раздел 2. Прогноз осложнений при ГСД",
    "Раздел 3. Метаболомный профиль",
    "Итог"
])

# -------- TAB 1 --------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите клинико‑анамнестические данные:")

    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input("ИМТ (кг/м²)", min_value=14.0, max_value=60.0, value=27.0, step=0.1, format="%.1f")
    with col2:
        fam_dm_label = st.radio("СД у родственников первой линии", ["Нет", "Да"], horizontal=True)
        fam_dm = 1 if fam_dm_label == "Да" else 0

    if st.button("Рассчитать базовый риск", key="btn_base"):
        p_base = base_risk(bmi, fam_dm)
        cat_base = base_category(p_base)
        st.session_state["base_p"] = p_base
        st.session_state["base_cat"] = cat_base
        st.markdown(f"<div class='{color_class(cat_base)}'>{cat_base} риск ({p_base*100:.1f}%)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------- TAB 2 --------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите показатели липидного профиля:")

    col1, col2 = st.columns(2)
    with col1:
        tg = st.number_input("Триглицериды, ммоль/л", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    with col2:
        hdl = st.number_input("ЛПВП, ммоль/л", min_value=0.1, max_value=5.0, value=1.2, step=0.1)

    if st.button("Рассчитать риск осложнений", key="btn_lipid"):
        p_lipid = lipid_risk(tg, hdl)
        cat_lipid = extended_category(p_lipid, low=0.35, high=0.689)
        st.session_state["lipid_p"] = p_lipid
        st.session_state["lipid_cat"] = cat_lipid
        st.markdown(f"<div class='{color_class(cat_lipid)}'>{cat_lipid} риск осложнений ({p_lipid*100:.1f}%)</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------- TAB 3 --------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите уровни аминокислот (ммоль/моль креатинина):")

    col1, col2 = st.columns(2)
    with col1:
        tyrosine_str = st.text_input("Тирозин")
        alphaaaa_str = st.text_input("α‑Аминоадипиновая кислота")
        pe_str = st.text_input("Фосфоэтаноламин")
    with col2:
        mh3_str = st.text_input("3‑Метилгистидин (MH3)")
        ps_str = st.text_input("Фосфосерин")

    if st.button("Рассчитать метаболомный риск", key="btn_meta"):
        vals = {
            "Tyrosine": parse_num(tyrosine_str),
            "AlphaAminoadipicAcid": parse_num(alphaaaa_str),
            "MH3": parse_num(mh3_str),
            "Phosphoethanolamine": parse_num(pe_str),
            "Phosphoserine": parse_num(ps_str)
        }

        if any(v is None or v <= 0 for v in vals.values()):
            st.error("Проверьте все поля — введены ли положительные значения чисел.")
        else:
            df_raw = pd.DataFrame([vals])
            df_norm = normalize_raw_df(df_raw)
            p_meta = meta_predict(df_norm)
            st.session_state["meta_p"] = p_meta
            cat_meta = extended_category(p_meta, low=0.05, high=0.1)
            st.session_state["meta_cat"] = cat_meta

            st.markdown(f"<div class='{color_class(cat_meta)}'>{cat_meta} риск по метаболомике ({p_meta*100:.1f}%)</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------- TAB 4 --------
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Итоговая категория")

    base_cat = st.session_state.get("base_cat")
    lipid_cat = st.session_state.get("lipid_cat")
    meta_cat = st.session_state.get("meta_cat")

    results = []
    if base_cat:
        st.write(f"Базовый риск → {base_cat}")
        results.append(base_cat)
    if lipid_cat:
        p_lipid = st.session_state.get("lipid_p")
        st.write(f"Липидный риск → {lipid_cat} ({p_lipid*100:.1f}%)")
        results.append(lipid_cat)
    if meta_cat:
        p_meta = st.session_state.get("meta_p")
        st.write(f"Метаболомный риск → {meta_cat} ({p_meta*100:.1f}%)")
        results.append(meta_cat)

    if results:
        final_risk = combine_categories(*results)
        st.markdown(f"<div class='{color_class(final_risk)}'>Итоговая категория риска: {final_risk}</div>", unsafe_allow_html=True)
    else:
        st.caption("Пожалуйста, рассчитайте хотя бы один риск.")
    st.markdown("</div>", unsafe_allow_html=True)
