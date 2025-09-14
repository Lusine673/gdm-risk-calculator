import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Модель раннего прогноза ГСД", page_icon="🧪", layout="centered")

# ---------- Стили (спокойная медицинская палитра) ----------
st.markdown("""
<style>
:root{
  --primary:#0ea5a2;           /* бирюзовый акцент */
  --ok:#1b5e20;                /* зелёный текст */
  --okbg:#e8f5e9;              /* зелёный фон */
  --warn:#b71c1c;              /* красный текст */
  --warnbg:#ffebee;            /* красный фон */
  --card:#ffffff;              /* фон карточек */
  --border:#e9eef2;            /* светлая рамка */
}
.block-container{padding-top:2rem;padding-bottom:2rem;}
h3{font-weight:800;letter-spacing:.2px;margin-bottom:.8rem;}

/* карточка ввода */
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:14px;
  padding:16px 18px;
  box-shadow:0 2px 10px rgba(0,0,0,.04);
}

/* кнопка */
div.stButton > button{
  background:linear-gradient(90deg,var(--primary),#14b8a6);
  color:#fff;border:0;border-radius:10px;padding:.65rem 1.05rem;font-weight:600;
}
div.stButton > button:hover{filter:brightness(1.05);}

/* индикаторы */
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

/* 🔽 скрытие лишнего интерфейса Streamlit */
#MainMenu {visibility: hidden;}      /* меню (гамбургер) */
footer {visibility: hidden;}         /* нижний футер "Made with Streamlit" */
header {visibility: hidden;}         /* стандартный заголовок */
[data-testid="stToolbar"] {visibility: hidden !important;}  /* панель Fork/GitHub */
</style>
""", unsafe_allow_html=True)

# ---------- Модель ----------
BETA0 = 0.0
COEFFS = {
    "Tyrosine": 0.1279647860222218,
    "MH3": 0.8913890705856571,             # 3-метилгистидин
    "Phosphoethanolamine": -0.8390359958954429,
    "Phosphoserine": -1.1553144548078098
}
THRESH = 0.382
FEATURES = list(COEFFS.keys())

TRAIN_RAW = {
    "Tyrosine": [45.31, 23.43, 15.03, 20.80, 19.32, 9.58, 12.33, 10.53, 10.17, 14.52],
    "MH3": [46.08, 35.35, 39.78, 15.54, 36.53, 11.93, 10.46, 15.63, 16.70, 11.75],
    "Phosphoethanolamine": [1.08, 0.55, 1.26, 1.09, 
