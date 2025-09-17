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
THIGH = 
