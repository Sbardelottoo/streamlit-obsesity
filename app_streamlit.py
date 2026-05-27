"""
================================================================================
TECH CHALLENGE – FASE 04 | POS TECH DATA ANALYTICS
ObesityIQ – Dashboard Clínico Epidemiológico (Dark Theme)
Autor: Leonardo Fernandes Sbardelotto
================================================================================
Executar com:  streamlit run app_streamlit.py
"""

import json
import pickle
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline_ml import FEATURES, TARGET_LABELS_PT, TARGET_ORDER, predict_single

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="ObesityIQ – Dashboard Clínico Epidemiológico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# DESIGN SYSTEM — DARK THEME
# ============================================================================

C = {
    "bg":        "#0B1120",
    "bg_2":      "#0D1426",
    "surface":   "#111827",
    "surface_2": "#1A2438",
    "border":    "#1F2D45",
    "text":      "#E2E8F0",
    "text_2":    "#94A3B8",
    "muted":     "#64748B",
    "primary":   "#3B82F6",
    "purple":    "#A855F7",
    "magenta":   "#EC4899",
    "cyan":      "#06B6D4",
    "green":     "#10B981",
    "orange":    "#F97316",
    "yellow":    "#FBBF24",
    "red":       "#EF4444",
    "danger":    "#DC2626",
}

CLASS_COLORS = {
    "Abaixo do Peso": "#06B6D4",
    "Peso Normal":    "#10B981",
    "Sobrepeso I":    "#FBBF24",
    "Sobrepeso II":   "#F97316",
    "Obesidade I":    "#EF4444",
    "Obesidade II":   "#DC2626",
    "Obesidade III":  "#991B1B",
}

BMI_RANGES = [
    (0,    18.5, "#06B6D4", "Abaixo do Peso"),
    (18.5, 25,   "#10B981", "Peso Normal"),
    (25,   30,   "#FBBF24", "Sobrepeso"),
    (30,   35,   "#EF4444", "Obesidade I"),
    (35,   40,   "#DC2626", "Obesidade II"),
    (40,   100,  "#991B1B", "Obesidade III"),
]

# ============================================================================
# CSS — TEMA ESCURO COM GLOW EFFECTS
# ============================================================================

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* ─── BASE ─────────────────────────────────────────────────── */
  .stApp {{
      background: {C['bg']};
      color: {C['text']};
      font-family: 'Inter', -apple-system, sans-serif;
  }}
  html, body, [class*="css"] {{ color: {C['text']}; }}

  section[data-testid="stSidebar"] {{ background: {C['bg_2']}; }}

  .block-container {{
      padding-top: 1rem;
      padding-bottom: 2rem;
      max-width: 1500px;
  }}

  /* ─── HEADER ───────────────────────────────────────────────── */
  .top-header {{
      background: linear-gradient(135deg, {C['surface']} 0%, {C['surface_2']} 100%);
      border: 1px solid {C['border']};
      border-radius: 14px;
      padding: 1.1rem 1.4rem;
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
  }}
  .top-header h1 {{
      font-size: 1.5rem; font-weight: 700; margin: 0;
      color: {C['text']};
      letter-spacing: -0.02em;
  }}
  .top-header h1 span {{
      background: linear-gradient(135deg, {C['cyan']}, {C['primary']});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
  }}
  .top-header .meta {{
      font-size: 0.78rem; color: {C['muted']};
      letter-spacing: 0.05em; text-transform: uppercase;
      margin-top: 0.2rem;
  }}

  /* ─── KPI CARDS COM GLOW ───────────────────────────────────── */
  .kpi-card {{
      background: linear-gradient(180deg, {C['surface_2']} 0%, {C['surface']} 100%);
      border: 1px solid {C['border']};
      border-radius: 12px;
      padding: 1.1rem 1.3rem;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      height: 100%;
  }}
  .kpi-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }}
  .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: var(--glow);
      box-shadow: 0 0 18px 1px var(--glow);
  }}
  .kpi-label {{
      font-size: 0.72rem; font-weight: 600;
      color: {C['muted']};
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 0.5rem;
  }}
  .kpi-value {{
      font-size: 2.1rem; font-weight: 700;
      color: {C['text']};
      font-family: 'JetBrains Mono', monospace;
      line-height: 1;
      margin-bottom: 0.3rem;
  }}
  .kpi-sub {{
      font-size: 0.78rem; color: {C['text_2']};
  }}

  /* ─── FUNIL CARDS ──────────────────────────────────────────── */
  .funnel-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 10px;
      padding: 0.85rem 1.1rem;
      border-top: 2px solid var(--glow);
      box-shadow: 0 0 12px var(--glow)22 inset;
  }}
  .funnel-title {{
      font-size: 0.72rem; color: {C['text_2']};
      margin-bottom: 0.5rem;
  }}
  .funnel-title b {{ color: {C['text']}; font-weight: 600; }}
  .funnel-stats {{
      display: flex; gap: 1.5rem; align-items: baseline;
  }}
  .funnel-stat {{
      display: flex; flex-direction: column;
  }}
  .funnel-stat-label {{
      font-size: 0.65rem; color: {C['muted']};
      text-transform: uppercase; letter-spacing: 0.1em;
  }}
  .funnel-stat-value {{
      font-size: 1.4rem; font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: var(--glow);
  }}

  /* ─── BREAKDOWN CARDS ─────────────────────────────────────── */
  .bd-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 12px;
      padding: 1rem 1.2rem;
      height: 100%;
  }}
  .bd-title {{
      font-size: 0.78rem; font-weight: 600;
      color: {C['text']};
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.2rem;
  }}
  .bd-subtitle {{
      font-size: 0.72rem; color: {C['muted']};
      margin-bottom: 0.7rem;
  }}
  .bd-row {{
      display: flex; justify-content: space-between; align-items: center;
      padding: 0.3rem 0;
      font-size: 0.85rem;
  }}
  .bd-row .left {{ display: flex; align-items: center; gap: 0.5rem; }}
  .bd-dot {{
      width: 10px; height: 10px; border-radius: 3px;
      display: inline-block;
  }}
  .bd-value {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600; color: {C['text']};
  }}
  .bd-pct {{
      color: {C['muted']}; font-size: 0.75rem;
      margin-left: 0.4rem;
  }}
  .bd-bar {{
      height: 4px; background: {C['surface_2']};
      border-radius: 2px; margin-top: 0.3rem;
      overflow: hidden;
  }}
  .bd-bar-fill {{
      height: 100%; background: var(--bar-color);
      border-radius: 2px;
  }}

  /* ─── MINI METRIC CARDS ───────────────────────────────────── */
  .mini-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-top: 2px solid var(--glow);
      border-radius: 10px;
      padding: 0.8rem 1rem;
      text-align: left;
      height: 100%;
  }}
  .mini-icon {{
      font-size: 1.1rem;
      margin-bottom: 0.3rem;
  }}
  .mini-label {{
      font-size: 0.65rem;
      color: {C['muted']};
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 0.3rem;
  }}
  .mini-value {{
      font-size: 1.4rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: {C['text']};
      line-height: 1;
  }}
  .mini-sub {{
      font-size: 0.72rem;
      color: {C['text_2']};
      margin-top: 0.2rem;
  }}

  /* ─── BARRA DE FILTROS ────────────────────────────────────── */
  .filter-bar {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 10px;
      padding: 0.6rem 1rem;
      margin: 0.5rem 0 1rem 0;
  }}
  .filter-label {{
      font-size: 0.65rem; color: {C['muted']};
      text-transform: uppercase; letter-spacing: 0.1em;
      font-weight: 600;
  }}

  /* ─── INPUTS ──────────────────────────────────────────────── */
  .stSelectbox label, .stSlider label, .stNumberInput label,
  .stRadio label, .stTextInput label {{
      color: {C['text_2']} !important;
      font-size: 0.78rem !important;
      font-weight: 500 !important;
  }}
  .stSelectbox > div > div, .stTextInput > div > div > input,
  .stNumberInput > div > div > input {{
      background: {C['surface']} !important;
      color: {C['text']} !important;
      border: 1px solid {C['border']} !important;
      border-radius: 8px !important;
  }}
  .stRadio > div {{ gap: 0.5rem; }}

  /* ─── TABS ────────────────────────────────────────────────── */
  .stTabs {{ background: transparent; }}
  .stTabs [data-baseweb="tab-list"] {{
      gap: 0;
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 10px;
      padding: 0.3rem;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent;
      color: {C['text_2']};
      border-radius: 8px;
      padding: 0.5rem 1.4rem !important;
      font-weight: 500;
      font-size: 0.88rem;
      border: none !important;
  }}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(135deg, {C['primary']}, #1D4ED8) !important;
      color: white !important;
      font-weight: 600;
  }}

  /* ─── BOTÕES ──────────────────────────────────────────────── */
  .stButton > button {{
      background: linear-gradient(135deg, {C['primary']}, #1D4ED8);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 0.55rem 1.2rem;
      font-weight: 600;
      font-size: 0.85rem;
      transition: all 0.2s;
      box-shadow: 0 0 12px rgba(59,130,246,0.3);
  }}
  .stButton > button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 4px 18px rgba(59,130,246,0.5);
  }}
  .stDownloadButton > button {{
      background: linear-gradient(135deg, {C['green']}, #059669);
      color: white;
      border: none;
      border-radius: 8px;
      box-shadow: 0 0 12px rgba(16,185,129,0.3);
  }}

  /* ─── EXPANDER (form) ─────────────────────────────────────── */
  .streamlit-expanderHeader {{
      background: {C['surface']} !important;
      border: 1px solid {C['border']} !important;
      border-radius: 10px !important;
      color: {C['text']} !important;
      font-weight: 600;
  }}
  .streamlit-expanderContent {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-top: none;
      border-radius: 0 0 10px 10px;
      padding: 1rem;
  }}

  /* ─── DATAFRAMES ──────────────────────────────────────────── */
  div[data-testid="stDataFrame"] {{
      background: {C['surface']};
      border-radius: 10px;
      border: 1px solid {C['border']};
  }}

  /* ─── CHART CAPTION ───────────────────────────────────────── */
  .chart-caption {{
      background: {C['surface']};
      border-left: 3px solid {C['primary']};
      padding: 0.6rem 1rem;
      border-radius: 6px;
      font-size: 0.78rem;
      color: {C['text_2']};
      line-height: 1.5;
      margin-top: 0.5rem;
      margin-bottom: 1rem;
  }}
  .chart-caption b {{ color: {C['cyan']}; }}

  /* ─── ML EXPLAIN ──────────────────────────────────────────── */
  .ml-explain {{
      background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(168,85,247,0.08));
      border-left: 3px solid {C['primary']};
      padding: 0.7rem 1rem;
      border-radius: 6px;
      font-size: 0.8rem;
      color: {C['text_2']};
      line-height: 1.5;
      margin: 0.6rem 0 1rem 0;
  }}
  .ml-explain b {{ color: {C['cyan']}; }}

  /* ─── INFO CARD ───────────────────────────────────────────── */
  .info-card {{
      background: {C['surface']};
      border: 1px solid {C['border']};
      border-radius: 12px;
      padding: 1rem 1.2rem;
      margin-bottom: 0.8rem;
      border-left: 3px solid var(--glow);
  }}
  .info-card h4 {{
      color: {C['text']};
      font-size: 0.92rem;
      margin: 0 0 0.4rem 0;
      font-weight: 600;
  }}
  .info-card p {{
      color: {C['text_2']};
      font-size: 0.85rem;
      line-height: 1.55;
      margin: 0;
  }}

  /* ─── RESULT CARD ─────────────────────────────────────────── */
  .result-card {{
      border-radius: 16px;
      padding: 1.8rem 2rem;
      text-align: center;
      box-shadow: 0 0 30px var(--glow)55;
      border: 1px solid var(--glow);
  }}
  .result-card h1 {{
      font-size: 2.2rem; margin: 0.5rem 0; font-weight: 700; color: white;
  }}
  .result-card .label {{
      font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em;
      color: rgba(255,255,255,0.8);
  }}

  /* ─── SCROLLBAR ───────────────────────────────────────────── */
  ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
  ::-webkit-scrollbar-track {{ background: {C['bg_2']}; }}
  ::-webkit-scrollbar-thumb {{
      background: {C['border']};
      border-radius: 4px;
  }}

  /* ─── HEADINGS ────────────────────────────────────────────── */
  h1, h2, h3, h4 {{ color: {C['text']}; }}
  .section-title {{
      font-size: 0.78rem; font-weight: 700;
      color: {C['text_2']};
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin: 1.2rem 0 0.6rem 0;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid {C['border']};
  }}

  /* ─── PLOTLY DARK ────────────────────────────────────────── */
  .js-plotly-plot, .plotly {{ background: transparent !important; }}

</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNÇÕES DE CONVERSÃO E HELPERS
# ============================================================================

def liters_to_ch2o(liters: float) -> int:
    if liters < 1.0: return 1
    if liters <= 2.0: return 2
    return 3

def days_to_faf(days: int) -> int:
    if days == 0: return 0
    if days <= 2: return 1
    if days <= 4: return 2
    return 3

def hours_to_tue(hours: float) -> int:
    if hours <= 2: return 0
    if hours <= 5: return 1
    return 2

def bmi_class(bmi: float):
    for lo, hi, cor, nome in BMI_RANGES:
        if lo <= bmi < hi:
            return cor, nome
    return "#991B1B", "Obesidade III"

def faixa_etaria(idade: int) -> str:
    if idade < 26: return "14-25"
    if idade < 41: return "26-40"
    if idade < 61: return "41-60"
    return "60+"


# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================

@st.cache_resource
def load_model():
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_meta.json") as f:
        meta = json.load(f)
    if "model_accuracy" not in meta:
        meta["model_accuracy"] = meta.get("xgb_accuracy", 0.0)
    if "target_classes" not in meta:
        meta["target_classes"] = meta.get("target_order", TARGET_ORDER)
    if "target_labels_pt" not in meta:
        meta["target_labels_pt"] = TARGET_LABELS_PT
    return model, meta


@st.cache_data
def load_dataset():
    """Carrega o CSV original para o dashboard epidemiológico."""
    for path in ["Dados Obesity.csv", "Obesity.csv"]:
        if Path(path).exists():
            df = pd.read_csv(path)
            # arredondamentos
            for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
                df[col] = df[col].round().astype(int)
            df["BMI"] = df["Weight"] / (df["Height"] ** 2)
            df["Age"] = df["Age"].round().astype(int)
            df["Faixa Etária"] = df["Age"].apply(faixa_etaria)
            # target em PT
            target_pt = dict(zip(TARGET_ORDER, TARGET_LABELS_PT))
            df["Diagnóstico"] = df["Obesity"].map(target_pt)
            df["Gênero_PT"] = df["Gender"].map({"Male": "Masculino", "Female": "Feminino"})
            return df
    return None


try:
    model, meta = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

df_full = load_dataset()


# ============================================================================
# SESSION STATE
# ============================================================================

if "historico" not in st.session_state:
    st.session_state.historico = []
if "ultima_predicao" not in st.session_state:
    st.session_state.ultima_predicao = None
if "dados_paciente" not in st.session_state:
    st.session_state.dados_paciente = None


# ============================================================================
# HEADER
# ============================================================================

now = datetime.now()
st.markdown(f"""
<div class="top-header">
  <div>
    <h1>🏥 <span>ObesityIQ</span> · Dashboard Clínico Epidemiológico</h1>
    <div class="meta">POS TECH Data Analytics · Tech Challenge Fase 04 · Atualizado às {now.strftime('%H:%M')}</div>
  </div>
  <div style="text-align:right; color:{C['muted']}; font-size:0.8rem;">
    <div style="color:{C['cyan']}; font-weight:600; font-size:0.92rem;">XGBoost · {meta['model_accuracy']*100:.1f}% acurácia</div>
    <div>{now.strftime('%d/%m/%Y')} · {now.strftime('%A').capitalize()}</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Modelo não encontrado. Execute `pipeline_ml.py`.")
    st.stop()

if df_full is None:
    st.error("⚠️ Arquivo `Dados Obesity.csv` não encontrado. Necessário para o dashboard.")
    st.stop()


# ============================================================================
# FORMULÁRIO NO TOPO (EXPANSÍVEL)
# ============================================================================

with st.expander("📋 Dados do Paciente — preencha para gerar análise individual", expanded=False):

    # Linha 1 — antropométricos
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gênero biológico", ["Female", "Male"],
                              format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
    with col2:
        age = st.number_input("Idade (anos)", 14, 80, 25, 1)
    with col3:
        height = st.number_input("Altura (m)", 1.40, 2.20, 1.70, 0.01)
    with col4:
        weight = st.number_input("Peso (kg)", 30.0, 250.0, 70.0, 0.5)

    bmi_preview = weight / (height ** 2)
    bmi_cor, bmi_nome = bmi_class(bmi_preview)
    st.markdown(
        f"<div style='text-align:right; margin-top:-0.3rem; font-size:0.85rem;'>"
        f"<span style='color:{C['muted']};'>IMC calculado:</span> "
        f"<b style='color:{bmi_cor}; font-family:JetBrains Mono;'>{bmi_preview:.1f} kg/m²</b> "
        f"<span style='color:{bmi_cor}; font-size:0.78rem;'>({bmi_nome})</span></div>",
        unsafe_allow_html=True,
    )

    # Linha 2 — alimentação
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        family_history = st.selectbox("Histórico familiar de sobrepeso",
                                      ["no", "yes"],
                                      format_func=lambda x: "Não" if x == "no" else "Sim")
    with col2:
        favc = st.selectbox("Consome alimentos calóricos com frequência",
                            ["no", "yes"],
                            format_func=lambda x: "Não" if x == "no" else "Sim")
    with col3:
        fcvc = st.select_slider("Frequência de vegetais",
                                options=[1, 2, 3], value=2,
                                format_func=lambda x: {1:"Raramente",2:"Às vezes",3:"Sempre"}[x])
    with col4:
        ncp = st.slider("Refeições principais/dia", 1, 4, 3)

    # Linha 3 — hábitos
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        caec = st.selectbox("Lanches entre refeições",
                            ["no", "Sometimes", "Frequently", "Always"], index=1,
                            format_func=lambda x: {"no":"Nunca","Sometimes":"Às vezes",
                                                    "Frequently":"Frequentemente","Always":"Sempre"}[x])
    with col2:
        scc = st.selectbox("Monitora calorias",
                           ["no", "yes"],
                           format_func=lambda x: "Não" if x == "no" else "Sim")
    with col3:
        water_liters = st.slider("Água por dia (L)", 0.5, 4.0, 2.0, 0.5, format="%.1f L")
    with col4:
        activity_days = st.slider("Exercício (dias/semana)", 0, 7, 2)

    # Linha 4 — estilo de vida
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        activity_intensity = st.selectbox("Intensidade do exercício",
                                          ["leve", "moderada", "intensa"], index=1,
                                          format_func=lambda x: x.capitalize())
    with col2:
        screen_hours = st.slider("Eletrônicos (horas/dia)", 0.0, 12.0, 4.0, 0.5, format="%.1f h")
    with col3:
        smoke = st.selectbox("Fuma", ["no", "yes"],
                             format_func=lambda x: "Não" if x == "no" else "Sim")
    with col4:
        calc = st.selectbox("Consumo de álcool",
                            ["no", "Sometimes", "Frequently", "Always"],
                            format_func=lambda x: {"no":"Não bebo","Sometimes":"Socialmente",
                                                    "Frequently":"Frequentemente","Always":"Diariamente"}[x])

    # Linha 5 — transporte + botão
    col1, col2 = st.columns([3, 2])
    with col1:
        mtrans = st.selectbox("Meio de transporte habitual",
                              ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
                              format_func=lambda x: {
                                  "Public_Transportation":"🚌 Transporte Público","Automobile":"🚗 Carro",
                                  "Walking":"🚶 A Pé","Motorbike":"🏍️ Moto","Bike":"🚴 Bicicleta"}[x])
    with col2:
        st.write("")
        st.write("")
        predict_btn = st.button("🔍 Analisar Paciente", use_container_width=True)

    # converter valores para o modelo
    ch2o = liters_to_ch2o(water_liters)
    faf  = days_to_faf(activity_days)
    tue  = hours_to_tue(screen_hours)

    # ── Predição ────────────────────────────────────────────────
    if predict_btn:
        input_data = {
            "Gender": gender, "Age": age, "Height": height, "Weight": weight,
            "family_history": family_history, "FAVC": favc, "FCVC": fcvc,
            "NCP": ncp, "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o,
            "SCC": scc, "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
        }
        try:
            result = predict_single(input_data, model, meta)
            st.session_state.ultima_predicao = result
            st.session_state.dados_paciente = {
                **input_data,
                "water_liters": water_liters,
                "activity_days": activity_days,
                "activity_intensity": activity_intensity,
                "screen_hours": screen_hours,
                "bmi": bmi_preview,
            }
            label = result["label_pt"]
            probs = result["probabilities"]
            st.session_state.historico.append({
                "Hora":        datetime.now().strftime("%H:%M:%S"),
                "Gênero":      "M" if gender == "Male" else "F",
                "Idade":       age,
                "IMC":         round(bmi_preview, 1),
                "Diagnóstico": label,
                "Confiança":   f"{probs[label]:.1f}%",
            })
            st.success(f"✅ Análise concluída: **{label}** (Confiança: {probs[label]:.1f}%) — veja detalhes na aba 🩺 **Análise Clínica**")
        except Exception as e:
            st.error(f"Erro na predição: {e}")


# ============================================================================
# BARRA DE FILTROS GLOBAIS
# ============================================================================

st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([2, 2, 2, 2, 1])

with fcol1:
    f_genero = st.selectbox("Gênero", ["Todos", "Masculino", "Feminino"], key="f_gen")
with fcol2:
    f_faixa = st.selectbox("Faixa Etária", ["Todas", "14-25", "26-40", "41-60", "60+"], key="f_age")
with fcol3:
    f_classe = st.selectbox("Classificação",
                            ["Todas", "Abaixo do Peso", "Peso Normal",
                             "Sobrepeso I", "Sobrepeso II",
                             "Obesidade I", "Obesidade II", "Obesidade III"], key="f_cls")
with fcol4:
    f_grupo = st.selectbox("Grupo de Risco",
                           ["Todos", "Saudáveis (Normal)", "Risco Leve (Sobrepeso)",
                            "Alto Risco (Obesidade)"], key="f_risk")
with fcol5:
    st.write("")
    st.write("")
    pdf_btn_placeholder = st.empty()

st.markdown('</div>', unsafe_allow_html=True)


# ── Aplicar filtros ─────────────────────────────────────────────
df = df_full.copy()
if f_genero != "Todos":
    df = df[df["Gênero_PT"] == f_genero]
if f_faixa != "Todas":
    df = df[df["Faixa Etária"] == f_faixa]
if f_classe != "Todas":
    df = df[df["Diagnóstico"] == f_classe]
if f_grupo == "Saudáveis (Normal)":
    df = df[df["Diagnóstico"] == "Peso Normal"]
elif f_grupo == "Risco Leve (Sobrepeso)":
    df = df[df["Diagnóstico"].isin(["Sobrepeso I", "Sobrepeso II"])]
elif f_grupo == "Alto Risco (Obesidade)":
    df = df[df["Diagnóstico"].isin(["Obesidade I", "Obesidade II", "Obesidade III"])]

total = len(df)
if total == 0:
    st.warning("⚠️ Nenhum paciente atende aos filtros selecionados. Ajuste os filtros para visualizar os dados.")
    st.stop()


# ============================================================================
# GERAÇÃO DO PDF
# ============================================================================

def gerar_pdf_dashboard(df_filtrado, filtros, meta, ultima_pred=None, dados_pac=None):
    """Gera PDF com KPIs do dashboard + análise individual (se houver)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors as rl
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Heading1"],
                           fontSize=20, textColor=rl.HexColor("#1E40AF"),
                           spaceAfter=4, alignment=TA_LEFT)
    subt = ParagraphStyle("Sub", parent=styles["Normal"],
                          fontSize=10, textColor=rl.HexColor("#64748B"),
                          spaceAfter=16)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontSize=12, textColor=rl.HexColor("#1E40AF"),
                        spaceBefore=14, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=9.5, textColor=rl.HexColor("#0F172A"),
                          leading=14)

    el = []
    el.append(Paragraph("🏥 ObesityIQ — Relatório Clínico Epidemiológico", title))
    el.append(Paragraph(
        f"Sistema preditivo de obesidade · POS TECH Data Analytics · "
        f"Gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}", subt))

    # ── Filtros aplicados ─────────────────────────────────────
    el.append(Paragraph("Filtros Aplicados", h2))
    filtros_data = [
        ["Filtro", "Valor Selecionado"],
        ["Gênero",          filtros.get("genero", "Todos")],
        ["Faixa Etária",    filtros.get("faixa",  "Todas")],
        ["Classificação",   filtros.get("classe", "Todas")],
        ["Grupo de Risco",  filtros.get("grupo",  "Todos")],
        ["Pacientes na amostra", f"{len(df_filtrado):,}".replace(",", ".")],
    ]
    t1 = Table(filtros_data, colWidths=[5*cm, 10*cm])
    t1.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), rl.HexColor("#1E40AF")),
        ("TEXTCOLOR",   (0,0), (-1,0), rl.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl.HexColor("#F8FAFC"), rl.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, rl.HexColor("#CBD5E1")),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",     (0,0), (-1,-1), 6),
    ]))
    el.append(t1)

    # ── KPIs ──────────────────────────────────────────────────
    el.append(Paragraph("Indicadores Epidemiológicos da Amostra", h2))
    risco_obesidade = df_filtrado["Diagnóstico"].isin(["Obesidade I","Obesidade II","Obesidade III"]).mean() * 100
    kpis = [
        ["Indicador", "Valor"],
        ["Total de pacientes",       f"{len(df_filtrado):,}".replace(",", ".")],
        ["IMC médio (kg/m²)",        f"{df_filtrado['BMI'].mean():.1f}"],
        ["% Alto risco (Obesidade)", f"{risco_obesidade:.1f}%"],
        ["Idade média",              f"{df_filtrado['Age'].mean():.1f} anos"],
        ["Peso médio",               f"{df_filtrado['Weight'].mean():.1f} kg"],
        ["Altura média",             f"{df_filtrado['Height'].mean():.2f} m"],
        ["% com histórico familiar", f"{(df_filtrado['family_history']=='yes').mean()*100:.1f}%"],
        ["% com alimentação calórica", f"{(df_filtrado['FAVC']=='yes').mean()*100:.1f}%"],
    ]
    t2 = Table(kpis, colWidths=[8*cm, 7*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), rl.HexColor("#1E40AF")),
        ("TEXTCOLOR",   (0,0), (-1,0), rl.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl.HexColor("#F8FAFC"), rl.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, rl.HexColor("#CBD5E1")),
        ("PADDING",     (0,0), (-1,-1), 6),
    ]))
    el.append(t2)

    # ── Distribuição por classe ────────────────────────────────
    el.append(Paragraph("Distribuição por Classificação", h2))
    dist = df_filtrado["Diagnóstico"].value_counts().reindex(TARGET_LABELS_PT, fill_value=0)
    dist_rows = [["Classe", "Pacientes", "% da amostra"]]
    for c, q in dist.items():
        pct = (q / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
        dist_rows.append([c, str(q), f"{pct:.1f}%"])
    t3 = Table(dist_rows, colWidths=[7*cm, 4*cm, 4*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), rl.HexColor("#1E40AF")),
        ("TEXTCOLOR",   (0,0), (-1,0), rl.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl.HexColor("#F8FAFC"), rl.white]),
        ("GRID",        (0,0), (-1,-1), 0.3, rl.HexColor("#CBD5E1")),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
        ("PADDING",     (0,0), (-1,-1), 6),
    ]))
    el.append(t3)

    # ── Análise individual (se houver) ────────────────────────
    if ultima_pred and dados_pac:
        el.append(PageBreak())
        el.append(Paragraph("Análise Clínica Individual", title))
        el.append(Paragraph(f"Predição gerada em {datetime.now().strftime('%d/%m/%Y às %H:%M')}", subt))

        el.append(Paragraph("Diagnóstico Preditivo", h2))
        pred_rows = [
            ["Diagnóstico",       ultima_pred["label_pt"]],
            ["Confiança",         f"{ultima_pred['probabilities'][ultima_pred['label_pt']]:.1f}%"],
            ["IMC do paciente",   f"{dados_pac['bmi']:.1f} kg/m²"],
        ]
        t4 = Table(pred_rows, colWidths=[5*cm, 10*cm])
        t4.setStyle(TableStyle([
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("GRID", (0,0), (-1,-1), 0.3, rl.HexColor("#CBD5E1")),
            ("BACKGROUND", (0,0), (0,-1), rl.HexColor("#EFF6FF")),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        el.append(t4)

        el.append(Paragraph("Dados do Paciente", h2))
        pac_rows = [
            ["Gênero",     "Masculino" if dados_pac["Gender"] == "Male" else "Feminino"],
            ["Idade",      f"{dados_pac['Age']} anos"],
            ["Altura/Peso", f"{dados_pac['Height']:.2f} m / {dados_pac['Weight']:.1f} kg"],
            ["Histórico familiar", "Sim" if dados_pac["family_history"]=="yes" else "Não"],
            ["Água/dia",   f"{dados_pac['water_liters']:.1f} L"],
            ["Atividade física", f"{dados_pac['activity_days']} dias/semana ({dados_pac['activity_intensity']})"],
            ["Eletrônicos", f"{dados_pac['screen_hours']:.1f} h/dia"],
            ["Fuma",       "Sim" if dados_pac["SMOKE"]=="yes" else "Não"],
        ]
        t5 = Table(pac_rows, colWidths=[5*cm, 10*cm])
        t5.setStyle(TableStyle([
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.3, rl.HexColor("#CBD5E1")),
            ("BACKGROUND", (0,0), (0,-1), rl.HexColor("#F8FAFC")),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        el.append(t5)

        el.append(Paragraph("Probabilidades por Classe", h2))
        probs = ultima_pred["probabilities"]
        prob_rows = [["Classe", "Probabilidade"]]
        for c in TARGET_LABELS_PT:
            prob_rows.append([c, f"{probs.get(c, 0):.1f}%"])
        t6 = Table(prob_rows, colWidths=[8*cm, 4*cm])
        t6.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), rl.HexColor("#1E40AF")),
            ("TEXTCOLOR",   (0,0), (-1,0), rl.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl.HexColor("#F8FAFC"), rl.white]),
            ("GRID", (0,0), (-1,-1), 0.3, rl.HexColor("#CBD5E1")),
            ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        el.append(t6)

    el.append(Spacer(1, 0.6*cm))
    el.append(Paragraph(
        "<i>Relatório gerado automaticamente pelo sistema ObesityIQ. "
        "Modelo XGBoost com acurácia de 98,1% no teste holdout. "
        "Este documento é uma ferramenta de apoio à decisão clínica e não substitui "
        "avaliação médica presencial.</i>",
        body))

    doc.build(el)
    buf.seek(0)
    return buf.getvalue()


# ── Botão de PDF ───────────────────────────────────────────────
with pdf_btn_placeholder.container():
    pdf_bytes = gerar_pdf_dashboard(
        df, {"genero": f_genero, "faixa": f_faixa, "classe": f_classe, "grupo": f_grupo},
        meta, st.session_state.ultima_predicao, st.session_state.dados_paciente,
    )
    st.download_button(
        "📥 PDF",
        data=pdf_bytes,
        file_name=f"obesityiq_relatorio_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ============================================================================
# ABAS PRINCIPAIS
# ============================================================================

tab_geral, tab_clinica, tab_insights, tab_hist, tab_sobre = st.tabs([
    "📊 Visão Geral",
    "🩺 Análise Clínica",
    "🧠 Insights do Modelo",
    "📋 Histórico",
    "ℹ️ Sobre",
])


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ABA 1 — VISÃO GERAL (DASHBOARD EPIDEMIOLÓGICO)                      ║
# ╚════════════════════════════════════════════════════════════════════╝

with tab_geral:
    # ── KPIs MACRO ──────────────────────────────────────────────────
    bmi_medio    = df["BMI"].mean()
    alto_risco   = df["Diagnóstico"].isin(["Obesidade I","Obesidade II","Obesidade III"]).mean() * 100
    n_predicoes  = len(st.session_state.historico)
    idade_media  = df["Age"].mean()

    glow_kpis = [
        ("#A855F7", "👥 PACIENTES",     f"{total:,}".replace(",", "."), "na amostra filtrada"),
        ("#EC4899", "⚖️ IMC MÉDIO",     f"{bmi_medio:.1f}",              "kg/m²"),
        ("#06B6D4", "🔴 ALTO RISCO",    f"{alto_risco:.1f}%",            "Obesidade I, II, III"),
        ("#10B981", "🎯 ACURÁCIA",      f"{meta['model_accuracy']*100:.1f}%", "XGBoost"),
        ("#F97316", "⚡ PREDIÇÕES",     f"{n_predicoes}",                "sessão atual"),
    ]
    cols = st.columns(5)
    for col, (glow, label, value, sub) in zip(cols, glow_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--glow:{glow};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── FUNIL DE SEVERIDADE ─────────────────────────────────────────
    st.markdown('<div class="section-title">Progressão pela Severidade — Funil Clínico</div>',
                unsafe_allow_html=True)

    dist = df["Diagnóstico"].value_counts()
    n_total = len(df)
    def pct(c): return (dist.get(c, 0) / n_total * 100) if n_total else 0

    p_normal    = pct("Peso Normal")
    p_sobr_i    = pct("Sobrepeso I")
    p_sobr_ii   = pct("Sobrepeso II")
    p_obesidade = pct("Obesidade I") + pct("Obesidade II") + pct("Obesidade III")

    funil_data = [
        ("#10B981", "Peso Normal", "Saudável",           p_normal,    p_normal),
        ("#FBBF24", "Sobrepeso I", "Pré-obesidade",      p_sobr_i,    p_normal + p_sobr_i),
        ("#F97316", "Sobrepeso II", "Início do risco",   p_sobr_ii,   p_normal + p_sobr_i + p_sobr_ii),
        ("#EF4444", "Obesidade",   "Alto risco clínico", p_obesidade, 100.0),
    ]
    cols = st.columns(4)
    for col, (glow, etapa, sub, etapa_pct, acum_pct) in zip(cols, funil_data):
        with col:
            st.markdown(f"""
            <div class="funnel-card" style="--glow:{glow};">
                <div class="funnel-title"><b>{etapa}</b> · {sub}</div>
                <div class="funnel-stats">
                    <div class="funnel-stat">
                        <span class="funnel-stat-label">Etapa</span>
                        <span class="funnel-stat-value">{etapa_pct:.1f}%</span>
                    </div>
                    <div class="funnel-stat">
                        <span class="funnel-stat-label">Acumulado</span>
                        <span class="funnel-stat-value" style="color:{C['text_2']};">{acum_pct:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        f"<div class='chart-caption'>"
        f"<b>Como ler:</b> <b>Etapa</b> = % de pacientes naquela classe. "
        f"<b>Acumulado</b> = % de pacientes até aquela classe (somatório). "
        f"Filtros ativos: {total} pacientes da amostra.</div>",
        unsafe_allow_html=True,
    )

    # ── BREAKDOWN POR DIMENSÃO ──────────────────────────────────────
    st.markdown('<div class="section-title">Detalhamento por Dimensão Clínica</div>',
                unsafe_allow_html=True)

    bd_cols = st.columns(5)

    # 1. Por Classe
    with bd_cols[0]:
        rows = ""
        for c in TARGET_LABELS_PT:
            q = int(dist.get(c, 0))
            pc = (q/n_total*100) if n_total else 0
            cor = CLASS_COLORS[c]
            rows += (
                f"<div class='bd-row'>"
                f"<div class='left'><span class='bd-dot' style='background:{cor};'></span>{c}</div>"
                f"<div><span class='bd-value'>{q}</span><span class='bd-pct'>{pc:.1f}%</span></div>"
                f"</div>"
                f"<div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{cor};width:{pc}%;'></div></div>"
            )
        st.markdown(f"""
        <div class="bd-card">
            <div class="bd-title">📊 Por Classe</div>
            <div class="bd-subtitle">Distribuição diagnóstica</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)

    # 2. Por Gênero
    with bd_cols[1]:
        gdist = df["Gênero_PT"].value_counts()
        gm, gf = gdist.get("Masculino", 0), gdist.get("Feminino", 0)
        gm_pct = (gm/n_total*100) if n_total else 0
        gf_pct = (gf/n_total*100) if n_total else 0
        st.markdown(f"""
        <div class="bd-card">
            <div class="bd-title">👤 Por Gênero</div>
            <div class="bd-subtitle">Masculino vs Feminino</div>
            <div class='bd-row'>
                <div class='left'><span class='bd-dot' style='background:{C['cyan']};'></span>Masculino</div>
                <div><span class='bd-value'>{gm}</span><span class='bd-pct'>{gm_pct:.1f}%</span></div>
            </div>
            <div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{C['cyan']};width:{gm_pct}%;'></div></div>
            <div class='bd-row'>
                <div class='left'><span class='bd-dot' style='background:{C['magenta']};'></span>Feminino</div>
                <div><span class='bd-value'>{gf}</span><span class='bd-pct'>{gf_pct:.1f}%</span></div>
            </div>
            <div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{C['magenta']};width:{gf_pct}%;'></div></div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Hábito Calórico
    with bd_cols[2]:
        cm = (df["FAVC"]=="yes").sum()
        cn = (df["FAVC"]=="no").sum()
        cm_pct = (cm/n_total*100) if n_total else 0
        cn_pct = (cn/n_total*100) if n_total else 0
        st.markdown(f"""
        <div class="bd-card">
            <div class="bd-title">🍔 Hábito Calórico</div>
            <div class="bd-subtitle">Consumo de calóricos</div>
            <div class='bd-row'>
                <div class='left'><span class='bd-dot' style='background:{C['red']};'></span>Consome frequente</div>
                <div><span class='bd-value'>{cm}</span><span class='bd-pct'>{cm_pct:.1f}%</span></div>
            </div>
            <div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{C['red']};width:{cm_pct}%;'></div></div>
            <div class='bd-row'>
                <div class='left'><span class='bd-dot' style='background:{C['green']};'></span>Não consome</div>
                <div><span class='bd-value'>{cn}</span><span class='bd-pct'>{cn_pct:.1f}%</span></div>
            </div>
            <div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{C['green']};width:{cn_pct}%;'></div></div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Atividade Física
    with bd_cols[3]:
        faf_labels = {0: "Sedentário", 1: "Baixa freq.", 2: "Regular", 3: "Atleta"}
        faf_cors   = {0: C['red'], 1: C['orange'], 2: C['green'], 3: C['cyan']}
        rows = ""
        for k in [0, 1, 2, 3]:
            q = (df["FAF"]==k).sum()
            p = (q/n_total*100) if n_total else 0
            rows += (
                f"<div class='bd-row'>"
                f"<div class='left'><span class='bd-dot' style='background:{faf_cors[k]};'></span>{faf_labels[k]}</div>"
                f"<div><span class='bd-value'>{q}</span><span class='bd-pct'>{p:.1f}%</span></div></div>"
                f"<div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{faf_cors[k]};width:{p}%;'></div></div>"
            )
        st.markdown(f"""
        <div class="bd-card">
            <div class="bd-title">🏃 Atividade Física</div>
            <div class="bd-subtitle">Frequência semanal</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)

    # 5. Genética
    with bd_cols[4]:
        gm = (df["family_history"]=="yes").sum()
        gn = (df["family_history"]=="no").sum()
        gm_pct = (gm/n_total*100) if n_total else 0
        gn_pct = (gn/n_total*100) if n_total else 0
        st.markdown(f"""
        <div class="bd-card">
            <div class="bd-title">🧬 Genética</div>
            <div class="bd-subtitle">Histórico familiar</div>
            <div class='bd-row'>
                <div class='left'><span class='bd-dot' style='background:{C['purple']};'></span>Com histórico</div>
                <div><span class='bd-value'>{gm}</span><span class='bd-pct'>{gm_pct:.1f}%</span></div>
            </div>
            <div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{C['purple']};width:{gm_pct}%;'></div></div>
            <div class='bd-row'>
                <div class='left'><span class='bd-dot' style='background:{C['muted']};'></span>Sem histórico</div>
                <div><span class='bd-value'>{gn}</span><span class='bd-pct'>{gn_pct:.1f}%</span></div>
            </div>
            <div class='bd-bar'><div class='bd-bar-fill' style='--bar-color:{C['muted']};width:{gn_pct}%;'></div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── INDICADORES CLÍNICOS (mini cards) ───────────────────────────
    st.markdown('<div class="section-title">Indicadores Clínicos Médios</div>',
                unsafe_allow_html=True)

    # mapeia transporte mais comum
    mt_top = df["MTRANS"].value_counts().idxmax() if total else "—"
    mt_map = {"Public_Transportation": "Público", "Automobile": "Carro",
              "Walking": "A Pé", "Motorbike": "Moto", "Bike": "Bicicleta"}

    mini_data = [
        ("#A855F7", "👤", "Idade Média",     f"{df['Age'].mean():.1f}",        "anos"),
        ("#06B6D4", "📏", "Altura Média",    f"{df['Height'].mean():.2f}",     "m"),
        ("#EC4899", "⚖️", "Peso Médio",      f"{df['Weight'].mean():.1f}",     "kg"),
        ("#3B82F6", "💧", "Água",            f"{df['CH2O'].mean():.1f}",       "nível 1-3"),
        ("#10B981", "🍽️", "Refeições",       f"{df['NCP'].mean():.1f}",        "por dia"),
        ("#F97316", "📱", "Tela",            f"{df['TUE'].mean():.1f}",        "nível 0-2"),
        ("#FBBF24", "🚌", "Transporte+",     mt_map.get(mt_top, mt_top),       f"{(df['MTRANS']==mt_top).mean()*100:.0f}% da amostra"),
    ]
    cols = st.columns(7)
    for col, (glow, icon, label, value, sub) in zip(cols, mini_data):
        with col:
            st.markdown(f"""
            <div class="mini-card" style="--glow:{glow};">
                <div class="mini-icon">{icon}</div>
                <div class="mini-label">{label}</div>
                <div class="mini-value">{value}</div>
                <div class="mini-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── GRÁFICO DE DISTRIBUIÇÃO POR CLASSE x GÊNERO ───────────────
    st.markdown('<div class="section-title">Distribuição Detalhada por Classe e Gênero</div>',
                unsafe_allow_html=True)

    cross = df.groupby(["Diagnóstico", "Gênero_PT"]).size().reset_index(name="qtd")
    cross["Diagnóstico"] = pd.Categorical(cross["Diagnóstico"], categories=TARGET_LABELS_PT, ordered=True)
    cross = cross.sort_values("Diagnóstico")

    fig_cross = px.bar(
        cross, x="Diagnóstico", y="qtd", color="Gênero_PT", barmode="group",
        color_discrete_map={"Masculino": C['cyan'], "Feminino": C['magenta']},
        labels={"qtd": "Pacientes", "Diagnóstico": "Classificação"},
    )
    fig_cross.update_layout(
        plot_bgcolor=C['surface'], paper_bgcolor=C['surface'],
        font=dict(family="Inter", color=C['text']),
        legend=dict(orientation="h", y=1.12),
        height=380, margin=dict(t=30, b=30),
        xaxis=dict(gridcolor=C['border'], color=C['text_2']),
        yaxis=dict(gridcolor=C['border'], color=C['text_2']),
    )
    st.plotly_chart(fig_cross, use_container_width=True)
    st.markdown(
        f"<div class='chart-caption'><b>Como ler:</b> barras lado a lado por classe. "
        f"<b>Cálculo:</b> contagem de pacientes em cada combinação Diagnóstico × Gênero "
        f"({total} pacientes na amostra filtrada). <b>Insight:</b> Obesidade Tipo II é predominante "
        f"em homens, enquanto Obesidade Tipo III predomina em mulheres.</div>",
        unsafe_allow_html=True,
    )


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ABA 2 — ANÁLISE CLÍNICA INDIVIDUAL                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

with tab_clinica:
    if st.session_state.ultima_predicao is None:
        st.markdown(f"""
        <div class="info-card" style="--glow:{C['primary']};">
            <h4>🩺 Nenhuma análise individual gerada</h4>
            <p>Preencha os <b>Dados do Paciente</b> no topo da página (expanda o formulário) e
            clique em <b>Analisar Paciente</b>. O resultado aparecerá nesta aba com diagnóstico,
            probabilidades, contexto clínico e recomendações médicas.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state.ultima_predicao
        dp     = st.session_state.dados_paciente
        label  = result["label_pt"]
        probs  = result["probabilities"]

        # informações clínicas embutidas
        RISK = {
            "Abaixo do Peso": ("⚠️ Atenção", "Moderado", "#06B6D4",
                "IMC abaixo de 18,5. Risco de deficiências nutricionais e imunidade comprometida.",
                ["Avaliação nutricional", "Dieta hipercalórica balanceada", "Investigar causas (tireoide)", "Monitorar micronutrientes"]),
            "Peso Normal": ("✅ Saudável", "Baixo", "#10B981",
                "IMC entre 18,5 e 24,9. Faixa de menor risco metabólico segundo a OMS.",
                ["Atividade física regular (150 min/sem)", "Dieta variada", "Check-up anual", "Manter hidratação e sono"]),
            "Sobrepeso I": ("⚠️ Leve", "Leve", "#FBBF24",
                "IMC entre 25 e 27,5. Início do excesso de peso — intervenção preventiva.",
                ["Aumentar aeróbico (150-200 min/sem)", "Reduzir ultraprocessados", "Acompanhamento nutricional", "Educação alimentar"]),
            "Sobrepeso II": ("🔶 Moderado", "Moderado", "#F97316",
                "IMC entre 27,5 e 29,9. Risco elevado de doenças metabólicas.",
                ["Programa estruturado de emagrecimento", "Avaliação cardiovascular", "Terapia comportamental", "Exames laboratoriais"]),
            "Obesidade I": ("🔴 Alto", "Alto", "#EF4444",
                "IMC entre 30 e 34,9. Risco significativo de diabetes, hipertensão e DCV.",
                ["Acompanhamento médico mensal", "Equipe multidisciplinar", "Avaliação de comorbidades", "Considerar farmacoterapia"]),
            "Obesidade II": ("🔴 Muito Alto", "Muito Alto", "#DC2626",
                "IMC entre 35 e 39,9. Comorbidades múltiplas prováveis.",
                ["Acompanhamento quinzenal", "Avaliação para bariátrica", "Suporte psicológico", "Plano nutricional supervisionado"]),
            "Obesidade III": ("⛔ Crítico", "Crítico", "#991B1B",
                "IMC ≥ 40. Obesidade mórbida — risco de vida significativamente aumentado.",
                ["Avaliação imediata para bariátrica", "Equipe multidisciplinar completa", "Internação para tratamento", "Monitoramento intensivo"]),
        }
        risco, nivel, cor, desc, recs = RISK[label]

        # ── CARD DE RESULTADO ───────────────────────────────────────
        col_r, col_k = st.columns([2, 1])
        with col_r:
            st.markdown(f"""
            <div class="result-card" style="--glow:{cor};
                background: linear-gradient(135deg, {cor}EE 0%, {cor} 100%);">
                <div class="label">Diagnóstico Preditivo</div>
                <h1>{label}</h1>
                <div class="label" style="margin-top:0.6rem;">{risco} · Nível {nivel}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_k:
            st.markdown(f"""
            <div class="kpi-card" style="--glow:{cor};">
                <div class="kpi-label">IMC do Paciente</div>
                <div class="kpi-value" style="color:{cor};">{dp['bmi']:.1f}</div>
                <div class="kpi-sub">kg/m²</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="kpi-card" style="--glow:{cor}; margin-top:0.5rem;">
                <div class="kpi-label">Confiança</div>
                <div class="kpi-value" style="color:{cor};">{probs[label]:.1f}%</div>
                <div class="kpi-sub">probabilidade da classe</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # ── GAUGE IMC ───────────────────────────────────────────────
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=dp['bmi'],
            number={"suffix": " kg/m²", "font": {"size": 30, "color": cor}},
            title={"text": "Índice de Massa Corporal", "font": {"size": 13, "color": C['text_2']}},
            gauge={
                "axis": {"range": [10, 50], "tickfont": {"color": C['text_2']}},
                "bar":  {"color": cor, "thickness": 0.28},
                "bgcolor": C['surface'],
                "borderwidth": 0,
                "steps": [
                    {"range": [10, 18.5], "color": "rgba(6,182,212,0.25)"},
                    {"range": [18.5, 25], "color": "rgba(16,185,129,0.25)"},
                    {"range": [25,  30],  "color": "rgba(251,191,36,0.25)"},
                    {"range": [30,  35],  "color": "rgba(239,68,68,0.25)"},
                    {"range": [35,  40],  "color": "rgba(220,38,38,0.25)"},
                    {"range": [40,  50],  "color": "rgba(153,27,27,0.25)"},
                ],
                "threshold": {"line": {"color": cor, "width": 4}, "thickness": 0.85, "value": dp['bmi']},
            },
        ))
        fig_g.update_layout(height=280, paper_bgcolor=C['surface'], font=dict(family="Inter", color=C['text']),
                            margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(
            f"<div class='chart-caption'><b>Cálculo:</b> Peso (kg) ÷ Altura² (m²). "
            f"<b>Faixas OMS:</b> verde 18,5-24,9 (saudável), amarelo 25-29,9 (sobrepeso), vermelho ≥30 (obesidade). "
            f"O IMC é o preditor #1 do modelo (51,8% de importância).</div>",
            unsafe_allow_html=True,
        )

        # ── CONTEXTO + RECOMENDAÇÕES ───────────────────────────────
        col_d, col_re = st.columns(2)
        with col_d:
            st.markdown(f"""
            <div class="info-card" style="--glow:{cor};">
                <h4>🩺 Contexto Clínico</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_re:
            rec_html = "".join(f"• {r}<br>" for r in recs)
            st.markdown(f"""
            <div class="info-card" style="--glow:{cor};">
                <h4>📋 Recomendações Médicas</h4>
                <p>{rec_html}</p>
            </div>
            """, unsafe_allow_html=True)

        # ── PROBABILIDADES ─────────────────────────────────────────
        st.markdown('<div class="section-title">Distribuição de Probabilidades</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='ml-explain'><b>O que é probabilidade da classe?</b> "
            "O modelo atribui um % para cada uma das 7 categorias. A classe predita é a de maior valor. "
            "Probabilidades próximas indicam pacientes em transição.</div>",
            unsafe_allow_html=True,
        )
        values = [probs.get(c, 0) for c in TARGET_LABELS_PT]
        colors = [CLASS_COLORS[c] for c in TARGET_LABELS_PT]

        fig_p = go.Figure(go.Bar(
            x=TARGET_LABELS_PT, y=values,
            marker_color=colors, marker_line_color=C['surface'], marker_line_width=2,
            text=[f"{v:.1f}%" for v in values], textposition="outside",
            textfont=dict(color=C['text'], size=11),
        ))
        fig_p.update_layout(
            plot_bgcolor=C['surface'], paper_bgcolor=C['surface'],
            font=dict(family="Inter", color=C['text']),
            yaxis=dict(range=[0, max(values)*1.25], gridcolor=C['border'], color=C['text_2'], title="Probabilidade (%)"),
            xaxis=dict(color=C['text_2'], tickangle=-25),
            height=400, margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_p, use_container_width=True)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ABA 3 — INSIGHTS DO MODELO                                          ║
# ╚════════════════════════════════════════════════════════════════════╝

with tab_insights:
    st.markdown(
        "<div class='ml-explain'><b>O que é XGBoost?</b> "
        "Algoritmo de aprendizado de máquina que combina centenas de árvores de decisão "
        "sequencialmente, cada uma corrigindo os erros das anteriores. Padrão-ouro em problemas tabulares.</div>",
        unsafe_allow_html=True,
    )

    # ── Comparativo dos modelos ──────────────────────────────────
    st.markdown('<div class="section-title">Comparativo de Desempenho dos Modelos</div>',
                unsafe_allow_html=True)

    modelos = pd.DataFrame({
        "Modelo": ["Logistic Regression", "Random Forest", "XGBoost (Final)"],
        "Teste":  [83.7, 98.3, 98.1],
        "CV":     [84.1, 98.4, 98.6],
    })
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Bar(name="Acurácia Teste (%)", x=modelos["Modelo"], y=modelos["Teste"],
                            marker_color=[C['muted'], C['primary'], C['cyan']],
                            text=[f"{v:.1f}%" for v in modelos["Teste"]], textposition="outside",
                            textfont=dict(color=C['text'])))
    fig_mc.add_trace(go.Bar(name="Cross-Val (%)", x=modelos["Modelo"], y=modelos["CV"],
                            marker_color=["#475569", "#1D4ED8", "#0E7490"],
                            text=[f"{v:.1f}%" for v in modelos["CV"]], textposition="outside",
                            textfont=dict(color=C['text'])))
    fig_mc.update_layout(
        barmode="group",
        yaxis=dict(range=[70, 105], gridcolor=C['border'], color=C['text_2']),
        xaxis=dict(color=C['text_2']),
        plot_bgcolor=C['surface'], paper_bgcolor=C['surface'],
        font=dict(family="Inter", color=C['text']),
        legend=dict(orientation="h", y=1.15),
        height=380, margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_mc, use_container_width=True)
    st.markdown(
        "<div class='chart-caption'><b>Acurácia Teste:</b> % de acertos em 422 pacientes nunca vistos. "
        "<b>Cross-Val:</b> média de 5 testes no treino — mede generalização. "
        "<b>Escolha:</b> XGBoost por melhor interpretabilidade e regularização nativa.</div>",
        unsafe_allow_html=True,
    )

    # ── Feature importance ───────────────────────────────────────
    st.markdown('<div class="section-title">Importância das Variáveis (XGBoost)</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<div class='ml-explain'><b>Como é calculado?</b> Soma-se em todas as 300 árvores do modelo "
        "quanto cada variável contribuiu para reduzir o erro nas divisões.</div>",
        unsafe_allow_html=True,
    )

    fi = pd.DataFrame(meta["feature_importance"])
    fi["importance_pct"] = (fi["importance"] * 100).round(2)
    fig_fi = px.bar(
        fi.sort_values("importance"),
        x="importance_pct", y="feature", orientation="h",
        color="importance_pct", color_continuous_scale=[C['surface_2'], C['cyan']],
        labels={"importance_pct": "Importância (%)", "feature": "Variável"},
        text="importance_pct",
    )
    fig_fi.update_traces(texttemplate="%{text:.2f}%", textposition="outside",
                         textfont=dict(color=C['text']))
    fig_fi.update_layout(
        plot_bgcolor=C['surface'], paper_bgcolor=C['surface'],
        font=dict(family="Inter", color=C['text']),
        coloraxis_showscale=False, height=520,
        xaxis=dict(gridcolor=C['border'], color=C['text_2']),
        yaxis=dict(color=C['text_2']),
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Insights principais ──────────────────────────────────────
    st.markdown('<div class="section-title">Principais Achados Clínicos</div>',
                unsafe_allow_html=True)
    ins = [
        ("#A855F7", "🏋️ IMC é o preditor #1", "51,8% das decisões. Confirma a validade do indicador OMS calculado por peso ÷ altura²."),
        ("#EC4899", "👤 Gênero é o #2",       "29,9% de importância. Homens concentram Obesidade II, mulheres concentram Obesidade III."),
        ("#06B6D4", "🧬 Genética determinante", "100% dos casos de Obesidade III têm histórico familiar de sobrepeso."),
        ("#10B981", "🏃 Sedentarismo progressivo", "Atividade física cai 38% do grupo Normal para o grupo Obesidade III."),
        ("#F97316", "🍔 Alimentação calórica",  "99,7% dos casos de Obesidade III consomem alimentos calóricos frequentemente."),
        ("#FBBF24", "📅 Início precoce",        "Sobrepeso aparece em média aos 23,4 anos. Intervenção jovem é essencial."),
    ]
    cols = st.columns(3)
    for i, (glow, t, d) in enumerate(ins):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="info-card" style="--glow:{glow};">
                <h4>{t}</h4>
                <p>{d}</p>
            </div>
            """, unsafe_allow_html=True)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ABA 4 — HISTÓRICO                                                   ║
# ╚════════════════════════════════════════════════════════════════════╝

with tab_hist:
    st.markdown(
        "<div class='ml-explain'>Histórico de análises realizadas nesta sessão do navegador. "
        "Não é persistido no servidor — fechar o navegador limpa os dados.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.historico:
        st.markdown(f"""
        <div class="info-card" style="--glow:{C['primary']};">
            <h4>📋 Histórico vazio</h4>
            <p>Nenhuma análise registrada ainda. Use o formulário no topo para começar.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_h = pd.DataFrame(st.session_state.historico)
        df_h.index = range(1, len(df_h) + 1)
        df_h.index.name = "#"
        st.dataframe(df_h, use_container_width=True)

        if st.button("🗑️ Limpar Histórico"):
            st.session_state.historico = []
            st.rerun()

        if len(df_h) >= 2:
            st.markdown('<div class="section-title">Distribuição dos Diagnósticos da Sessão</div>',
                        unsafe_allow_html=True)
            cnt = df_h["Diagnóstico"].value_counts().reset_index()
            cnt.columns = ["Diagnóstico", "Qtd"]
            cnt["cor"] = cnt["Diagnóstico"].map(CLASS_COLORS)
            fig_h = go.Figure(go.Bar(
                x=cnt["Diagnóstico"], y=cnt["Qtd"],
                marker_color=cnt["cor"], marker_line_color=C['surface'], marker_line_width=2,
                text=cnt["Qtd"], textposition="outside", textfont=dict(color=C['text']),
            ))
            fig_h.update_layout(
                plot_bgcolor=C['surface'], paper_bgcolor=C['surface'],
                font=dict(family="Inter", color=C['text']),
                yaxis=dict(gridcolor=C['border'], color=C['text_2']),
                xaxis=dict(color=C['text_2']),
                height=320, margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_h, use_container_width=True)

        csv = df_h.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Exportar CSV",
            data=csv,
            file_name=f"historico_obesityiq_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ╔════════════════════════════════════════════════════════════════════╗
# ║ ABA 5 — SOBRE                                                       ║
# ╚════════════════════════════════════════════════════════════════════╝

with tab_sobre:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card" style="--glow:{C['cyan']};">
            <h4>🎯 Objetivo</h4>
            <p>Sistema de apoio à decisão clínica que classifica pacientes em 7 níveis de obesidade
            a partir de 16 variáveis. Desenvolvido para o Tech Challenge Fase 04 — POS TECH Data Analytics.</p>
        </div>
        <div class="info-card" style="--glow:{C['purple']};">
            <h4>📊 Dataset</h4>
            <p>2.111 pacientes · 17 colunas · 0 valores ausentes. Origem: UCI ML Repository.
            7 classes balanceadas entre 272 e 351 pacientes por classe.</p>
        </div>
        <div class="info-card" style="--glow:{C['green']};">
            <h4>🧠 Modelo</h4>
            <p>XGBoost · 300 árvores · profundidade 6 · learning rate 0,1.
            Acurácia: 98,1% (teste) · 98,6% ± 0,7% (Cross-Val 5-fold).</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="info-card" style="--glow:{C['orange']};">
            <h4>🔧 Feature Engineering</h4>
            <p>• BMI derivado (Peso/Altura²) — feature #1<br>
            • Encoding binário: Gender, family_history, FAVC, SMOKE, SCC<br>
            • Encoding ordinal: CAEC, CALC<br>
            • Target ordenado por severidade clínica</p>
        </div>
        <div class="info-card" style="--glow:{C['magenta']};">
            <h4>⚙️ Stack Técnica</h4>
            <p>Python 3.14 · scikit-learn · XGBoost · Pandas · Plotly · Streamlit · ReportLab.
            Deploy: Streamlit Community Cloud.</p>
        </div>
        <div class="info-card" style="--glow:{C['red']};">
            <h4>⚠️ Aviso Médico</h4>
            <p>Sistema de <b>apoio</b> à decisão. <b>Não substitui</b> avaliação médica presencial.
            Resultados devem ser interpretados por profissionais de saúde.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Glossário Técnico</div>', unsafe_allow_html=True)
    gloss = {
        "IMC (BMI)":          "Peso (kg) ÷ Altura² (m²). Indicador clínico clássico recomendado pela OMS.",
        "Acurácia":           "% de classificações corretas sobre o total de amostras testadas.",
        "Cross-Validation":   "Divide o treino em K partes (folds), treina K vezes alternando qual é validação. Mede generalização.",
        "Feature Importance": "Métrica que indica quanto cada variável contribui para as decisões do modelo.",
        "XGBoost":            "Extreme Gradient Boosting. Treina árvores sequencialmente, cada uma corrigindo erros das anteriores.",
        "Holdout":            "20% dos dados reservados para teste final, nunca vistos no treinamento.",
        "Stratified Split":   "Divisão que preserva a proporção das 7 classes em treino e teste.",
    }
    for termo, defi in gloss.items():
        st.markdown(f"<div style='padding:0.4rem 0; border-bottom:1px solid {C['border']};'>"
                    f"<b style='color:{C['cyan']};'>{termo}</b> <span style='color:{C['text_2']};'>— {defi}</span></div>",
                    unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown(f"""
<div style="text-align:center; color:{C['muted']}; font-size:0.78rem;
            padding:1.5rem 0 0.5rem 0; border-top:1px solid {C['border']}; margin-top:2rem;">
🏥 <b>ObesityIQ</b> · Tech Challenge Fase 04 · POS TECH Data Analytics ·
Leonardo Fernandes Sbardelotto · XGBoost · Acurácia 98,1%
</div>
""", unsafe_allow_html=True)
