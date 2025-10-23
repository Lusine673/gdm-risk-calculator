import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Модель раннего прогноза ГСД", page_icon="🧪", layout="centered")

# ---------- Стили ----------
st.markdown("""
<style>
:root{
  --primary:#0ea5a2;
  --ok:#1b5e20;
  --okbg:#e8f5e9;
  --warn:#b71c1c;
  --warnbg:#ffebee;
  --card:#ffffff;
  --border:#e9eef2;
}
.block-container{padding-top:2rem;padding-bottom:2rem;}
h3{font-weight:800;letter-spacing:.2px;margin-bottom:.8rem;}

.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:14px;
  padding:16px 18px;
  box-shadow:0 2px 10px rgba(0,0,0,.04);
}

div.stButton > button{
  background:linear-gradient(90deg,var(--primary),#14b8a6);
  color:#fff;border:0;border-radius:10px;padding:.65rem 1.05rem;font-weight:600;
}
div.stButton > button:hover{filter:brightness(1.05);}

.risk-high{
  background:var(--warnbg);
  color:var(--warn);
  padding:14px;border-radius:12px;
  text-align:center;font-size:20px;font-weight:800;
}
.risk-low{
  background:var(--okbg);
  color:var(--ok);
  padding:14px;border-radius:12px;
  text-align:center;font-size:20px;font-weight:800;
}
.hr{height:1px;background:#edf2f7;margin:1.25rem 0;}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Раздел 1 — Клинико‑анамнестическая модель риска ГСД
# =====================================================
BASE_COEF = {
    "beta0": -2.8830,
    "bmi": 0.1043,
    "fam_dm": 0.8860
}
TLOW = 0.388
THIGH = 0.607

def logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def base_risk(bmi: float, fam_dm_01: int) -> float:
    b = BASE_COEF
    z = b["beta0"] + b["bmi"] * bmi + b["fam_dm"] * fam_dm_01
    return logistic(z)

def base_category(p: float, tlow=TLOW, thigh=THIGH) -> str:
    if p < tlow:
        return "Низкий"
    elif p < thigh:
        return "Промежуточный"
    else:
        return "Высокий"

def combine_categories(cat1: str, cat2: str) -> str:
    order = {"Низкий": 0, "Промежуточный": 1, "Высокий": 2}
    return max(cat1, cat2, key=lambda x: order.get(x, 1))

# =====================================================
# Раздел 3 — Метаболомная модель
# =====================================================
BETA0 = 0.0
COEFFS = {
    "Tyrosine": 2.33,
    "AlphaAminoadipicAcid": 0.96,
    "MH3": 1.13,
    "Phosphoethanolamine": -2.89,
    "Phosphoserine": -2.48
}
META_THRESH = 0.1
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
    x = np.log10(np.array(arr))
    MEAN_LOG[k] = np.mean(x)
    SD_LOG[k] = np.std(x, ddof=1)

def normalize_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=df_raw.index)
    for k in FEATURES:
        v = df_raw[k].astype(float)
        if (v <= 0).any():
            raise ValueError(f"{k}: значения должны быть > 0")
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
    try:
        return float(str(s).replace(",", "."))
    except:
        return None

# =====================================================
# Заголовок
# =====================================================
st.markdown(
    "<h3 style='text-align:center'>"
    "Комплексная модель прогнозирования осложнений<br/>"
    "при наличии гестационного сахарного диабета"
    "</h3>", unsafe_allow_html=True)

# =====================================================
# Вкладки
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Раздел 1. Стратификация по риску ГСД",
    "Раздел 2. Прогноз осложнений при ГСД",
    "Раздел 3. Метаболомный профиль",
    "Итог"
])

# -----------------------------------------------------
# TAB 1 — Клинико‑анамнестическая модель
# -----------------------------------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите клинико‑анамнестические данные:")

    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input("ИМТ (кг/м²)", min_value=14.0, max_value=60.0, value=27.0, step=0.1, format="%.1f")
    with col2:
        fam_dm_label = st.radio("СД у родственников первой линии", ["Нет", "Да"], horizontal=True)
        fam_dm_01 = 1 if fam_dm_label == "Да" else 0

    if st.button("Рассчитать базовый риск", key="btn_base"):
        p_base = float(base_risk(bmi, fam_dm_01))
        cat_base = base_category(p_base)
        st.session_state["base_p"] = p_base
        st.session_state["base_cat"] = cat_base
        html_class = "risk-high" if cat_base == "Высокий" else "risk-low"
        st.markdown(f"<div class='{html_class}'>{cat_base} риск ({p_base*100:.1f}%)</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# TAB 2 — Прогноз осложнений
# -----------------------------------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите показатели липидного профиля:")

    col1, col2 = st.columns(2)
    with col1:
        tg = st.number_input("Триглицериды, ммоль/л", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    with col2:
        hdl = st.number_input("ЛПВП, ммоль/л", min_value=0.1, max_value=5.0, value=1.2, step=0.1)

    if st.button("Рассчитать риск осложнений", key="btn_complications"):
        z = -2.837 + 2.431 * tg - 1.323 * hdl
        p = 1 / (1 + np.exp(-z)) * 100
        html_class = "risk-high" if p >= 50 else "risk-low"
        st.markdown(f"<div class='{html_class}'>Риск осложнений: {p:.1f}%</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# TAB 3 — Метаболомное профилирование
# -----------------------------------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите уровни аминокислот (ммоль/моль креатинина):")

    col1, col2 = st.columns(2)
    with col1:
        tyrosine_str = st.text_input("Тирозин", value="")
        alphaaaa_str = st.text_input("α‑Аминоадипиновая кислота", value="")
        pe_str = st.text_input("Фосфоэтаноламин", value="")
    with col2:
        mh3_str = st.text_input("3-метилгистидин (MH3)", value="")
        ps_str = st.text_input("Фосфосерин", value="")

    if st.button("Рассчитать метаболомный риск", key="btn_meta"):
        vals = {
            "Tyrosine": parse_num(tyrosine_str),
            "AlphaAminoadipicAcid": parse_num(alphaaaa_str),
            "MH3": parse_num(mh3_str),
            "Phosphoethanolamine": parse_num(pe_str),
            "Phosphoserine": parse_num(ps_str)
        }

        errors = []
        for name, val in vals.items():
            if val is None:
                errors.append(f"{name.replace('_',' ')}: введите число.")
            elif val <= 0:
                errors.append(f"{name.replace('_',' ')}: значение должно быть > 0.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            df_raw = pd.DataFrame([vals])
            try:
                df_norm = normalize_raw_df(df_raw)
                p_meta = float(predict_proba_from_norm(df_norm)[0])
                st.session_state["meta_p"] = p_meta
                cat_meta = "Высокий" if p_meta >= META_THRESH else "Низкий"
                st.session_state["meta_cat"] = cat_meta
                html_class = "risk-high" if cat_meta == "Высокий" else "risk-low"
                st.markdown(f"<div class='{html_class}'>{cat_meta} риск ({p_meta*100:.1f}%)</div>", unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"Ошибка предобработки: {ex}")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# TAB 4 — Итоговая оценка
# -----------------------------------------------------
with tab4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Итоговая категория")

    p_base = st.session_state.get("base_p")
    cat_base = st.session_state.get("base_cat")
    p_meta = st.session_state.get("meta_p")
    cat_meta = st.session_state.get("meta_cat")

    if (p_base is None) and (p_meta is None):
        st.caption("Сначала рассчитайте исходные показатели.")
    elif (p_base is not None) and (p_meta is None):
        st.write(f"Базовый риск: {p_base*100:.1f}% → {cat_base}")
    elif (p_base is None) and (p_meta is not None):
        st.write(f"Метаболомный риск: {p_meta*100:.1f}% → {cat_meta}")
    else:
        final_cat = combine_categories(cat_base, cat_meta)
        st.write(f"Базовый: {p_base*100:.1f}% → {cat_base}")
        st.write(f"Метаболомика: {p_meta*100:.1f}% → {cat_meta}")
        st.success(f"Итоговая категория (консервативно): {final_cat}")

    st.markdown("</div>", unsafe_allow_html=True)

# Разделительная линия внизу
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
