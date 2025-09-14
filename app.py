import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Модель раннего прогноза ГСД", page_icon="🧪", layout="centered")

# ---------- Стили ----------
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
  padding:16px 
