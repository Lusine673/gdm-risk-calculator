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
# NEW: Раздел 1 — клинико‑анамнестическая модель (BMI + fam_dm)
# =====================================================
BASE_COEF = {
    "beta0": -2.8830,
    "bmi":   0.1043,
    "fam_dm": 0.8860  # 0/1: нет/да семейной наследственности СД у родственников первой линии
}
# Пороговая логика (двухпороговая, из вашей валидации)
TLOW  = 0.388  # rule-out (Se≈0.90)
THIGH = 0.607  # rule-in  (Sp≈0.80)

def logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def base_risk(bmi: float, fam_dm_01: int) -> float:
    b = BASE_COEF
    z = b["beta0"] + b["bmi"]*bmi + b["fam_dm"]*fam_dm_01
    return logistic(z)

def base_category(p: float, tlow=TLOW, thigh=THIGH) -> str:
    if p < tlow:
        return "Низкий"
    elif p < thigh:
        return "Промежуточный"
    else:
        return "Высокий"

def combine_categories(cat_base: str, cat_meta: str) -> str:
    # Консервативная эскалация: «хуже из двух»
    order = {"Низкий": 0, "Промежуточный": 1, "Высокий": 2}
    # Если метаболомика бинарная («Низкий/Высокий»), всё равно работает
    return max(cat_base, cat_meta, key=lambda x: order.get(x, 1))

# =====================================================
# Раздел 2 — метаболомная модель (ваш существующий код)
# =====================================================

# ---------- Модель (метаболомика) ----------
BETA0 = 0.0
COEFFS = {
    "Tyrosine": 2.33,
    "AlphaAminoadipicAcid": 0.96,
    "MH3": 1.13,
    "Phosphoethanolamine": -2.89,
    "Phosphoserine": -2.48
}
META_THRESH = 0.1   # Порог метаболомной модели (чувствительность 100%, специфичность 80%)
FEATURES = list(COEFFS.keys())

# ---------- Тренировочные данные (пример, для нормализации)
TRAIN_RAW = {
    "Tyrosine": [
        45.31, 23.43, 15.03, 20.80, 19.32,
        9.58, 12.33, 10.53, 10.17, 14.52
    ],
    "AlphaAminoadipicAcid": [
        0.95, 1.12, 1.02, 1.08, 1.00,
        2.40, 2.20, 2.50, 2.10, 2.35
    ],
    "MH3": [
        46.08, 35.35, 39.78, 15.54, 36.53,
        11.93, 10.46, 15.63, 16.70, 11.75
    ],
    "Phosphoethanolamine": [
        1.08, 0.55, 1.26, 1.09, 0.69,
        2.93, 1.91, 1.91, 1.88, 2.20
    ],
    "Phosphoserine": [
        1.68, 2.12, 0.67, 0.53, 0.35,
        8.00, 3.78, 3.59, 1.80, 2.52
    ]
}

# ---------- Предобработка (Log10 + Pareto)
MEAN_LOG, SD_LOG = {}, {}
for k, arr in TRAIN_RAW.items():
    x = np.array(arr, dtype=float)
    xlog = np.log10(x)
    MEAN_LOG[k] = float(np.mean(xlog))
    SD_LOG[k]  = float(np.std(xlog, ddof=1))  # sample SD

def normalize_raw_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=df_raw.index)
    for k in FEATURES:
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
    try:
        return float(str(s).replace(",", "."))
    except:
        return None

# ---------- Заголовок ----------
st.markdown(
    "<h3 style='text-align:center'>"
    "Модель раннего прогноза<br/>"
    "(первый триместр беременности)<br/>"
    "гестационного сахарного диабета"
    "</h3>",
    unsafe_allow_html=True
)

# =====================================================
# NEW: вкладки — Раздел 1, Раздел 2, Итог
# =====================================================
tab1, tab2, tab3 = st.tabs(["Раздел 1. Клинико‑анамнестический", "Раздел 2. Метаболомный", "Итог"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите клинико‑анамнестические данные:")

    c1, c2 = st.columns(2)
    with c1:
        bmi = st.number_input("ИМТ (кг/м²)", min_value=14.0, max_value=60.0, value=27.0, step=0.1, format="%.1f")
    with c2:
        fam_dm_label = st.radio("Сахарный диабет у родственников первой линии", ["Нет", "Да"], horizontal=True)
        fam_dm_01 = 1 if fam_dm_label == "Да" else 0

    if st.button("Рассчитать базовый риск", key="btn_base"):
        p_base = float(base_risk(bmi, fam_dm_01))
        cat_base = base_category(p_base)
        st.session_state["base_p"] = p_base
        st.session_state["base_cat"] = cat_base

        if cat_base == "Высокий":
            st.markdown(f"<div class='risk-high'>Высокий базовый риск ({p_base*100:.1f}%)</div>", unsafe_allow_html=True)
        elif cat_base == "Низкий":
            st.markdown(f"<div class='risk-low'>Низкий базовый риск ({p_base*100:.1f}%)</div>", unsafe_allow_html=True)
        else:
            st.info(f"Промежуточный базовый риск ({p_base*100:.1f}%). Рекомендуется исследование мочи на метаболомный профиль аминокислот для уточнения риска.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("Введите значения аминокислот (ммоль/моль креатинина):")

    col1, col2 = st.columns(2)
    with col1:
        tyrosine_str  = st.text_input("Тирозин", value="")
        alphaaaa_str  = st.text_input("α‑Аминоадипиновая кислота", value="")
        pe_str        = st.text_input("Фосфоэтаноламин", value="")
    with col2:
        mh3_str       = st.text_input("3‑Метилгистидин (MH3)", value="")
        ps_str        = st.text_input("Фосфосерин", value="")

    if st.button("Рассчитать метаболомный риск", key="btn_meta"):
        tyrosine = parse_num(tyrosine_str)
        alphaaaa = parse_num(alphaaaa_str)
        mh3      = parse_num(mh3_str)
        pe       = parse_num(pe_str)
        ps       = parse_num(ps_str)

        errors = []
        for name, val in [("Тирозин", tyrosine),
                          ("α‑Аминоадипиновая кислота", alphaaaa),
                          ("3‑метилгистидин (MH3)", mh3),
                          ("Фосфоэтаноламин", pe),
                          ("Фосфосерин", ps)]:
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
                "AlphaAminoadipicAcid": alphaaaa,
                "MH3": mh3,
                "Phosphoethanolamine": pe,
                "Phosphoserine": ps
            }])
            try:
                df_norm = normalize_raw_df(df_raw)
                p_meta = float(predict_proba_from_norm(df_norm)[0])
                st.session_state["meta_p"] = p_meta
                # Бинарная категория по вашему порогу 0.1
                cat_meta = "Высокий" if p_meta >= META_THRESH else "Низкий"
                st.session_state["meta_cat"] = cat_meta

                if cat_meta == "Высокий":
                    st.markdown(f"<div class='risk-high'>Высокий риск по метаболомике ({p_meta*100:.1f}%)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-low'>Низкий риск по метаболомике ({p_meta*100:.1f}%)</div>", unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"Ошибка предобработки: {ex}")

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Итоговая категория")

    p_base   = st.session_state.get("base_p")
    cat_base = st.session_state.get("base_cat")
    p_meta   = st.session_state.get("meta_p")
    cat_meta = st.session_state.get("meta_cat")

    if (p_base is None) and (p_meta is None):
        st.caption("Сначала рассчитайте базовый и/или метаболомный риск.")
    elif (p_base is not None) and (p_meta is None):
        st.write(f"Базовый риск: {p_base*100:.1f}% → {cat_base}")
        st.info("Для промежуточного/высокого базового риска предложите метаболомный шаг.")
    elif (p_base is None) and (p_meta is not None):
        st.write(f"Метаболомный риск: {p_meta*100:.1f}% → {cat_meta}")
    else:
        final_cat = combine_categories(cat_base, cat_meta)
        st.write(f"Базовый: {p_base*100:.1f}% → {cat_base}")
        st.write(f"Метаболомика: {p_meta*100:.1f}% → {cat_meta}")
        st.success(f"Итоговая категория (консервативная эскалация): {final_cat}")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Дисклеймер ----------
with st.expander("Дисклеймер"):
    st.markdown("""
Исследовательский прототип. Не является медицинским изделием.

Базовая модель (ступень 1): логистическая регрессия по ИМТ и семейной наследственности СД (β₀ = −2.8830; βИМТ = 0.1043; βСД = 0.8860).
Пороговая логика: rule‑out Tlow = 0.388 (Se≈0.90), rule‑in Thigh = 0.607 (Sp≈0.80).

Метаболомная модель (ступень 2): LASSO‑логистическая регрессия по 5 аминокислотам (C=10), предобработка log10 → Pareto; бинарный порог 0.10
(чувствительность 100%, специфичность 80% в LOO‑валидации пилотной выборки n=10). Нужна внешняя валидация.
    """)
