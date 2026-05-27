"""
================================================================================
TECH CHALLENGE – FASE 04 | POS TECH DATA ANALYTICS
ObesityIQ – Aplicação Preditiva de Obesidade (Streamlit)
Autor: Leonardo Fernandes Sbardelotto
================================================================================
Executar com:  streamlit run app_streamlit.py
"""

import json
import pickle
from datetime import datetime

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
    page_title="ObesityIQ – Preditor Clínico de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DESIGN SYSTEM — paleta, tipografia, componentes
# ============================================================================

COLORS = {
    "primary":    "#1E40AF",   # azul institucional
    "primary_2":  "#3B82F6",   # azul vibrante
    "primary_3":  "#60A5FA",   # azul claro
    "bg":         "#F8FAFC",   # fundo geral
    "surface":    "#FFFFFF",   # cards
    "border":     "#E2E8F0",   # bordas suaves
    "text":       "#0F172A",   # texto principal
    "text_2":     "#334155",   # texto secundário
    "muted":      "#64748B",   # texto auxiliar
    "success":    "#10B981",
    "warning":    "#F59E0B",
    "danger":     "#EF4444",
    "info":       "#0EA5E9",
}

CLASS_COLORS = {
    "Abaixo do Peso": "#06B6D4",
    "Peso Normal":    "#10B981",
    "Sobrepeso I":    "#F59E0B",
    "Sobrepeso II":   "#F97316",
    "Obesidade I":    "#EF4444",
    "Obesidade II":   "#DC2626",
    "Obesidade III":  "#991B1B",
}

# Faixas oficiais da OMS para classificação por IMC
BMI_RANGES = [
    (0,    18.5, "#06B6D4", "Abaixo do Peso"),
    (18.5, 25,   "#10B981", "Peso Normal"),
    (25,   30,   "#F59E0B", "Sobrepeso"),
    (30,   35,   "#EF4444", "Obesidade I"),
    (35,   40,   "#DC2626", "Obesidade II"),
    (40,   100,  "#991B1B", "Obesidade III"),
]

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      color: {COLORS['text']};
  }}
  .main {{ background-color: {COLORS['bg']}; }}

  /* — botão principal — */
  .stButton > button {{
      background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_2']} 100%);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 0.7rem 1.5rem;
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: 0.01em;
      width: 100%;
      transition: all 0.2s ease;
      box-shadow: 0 1px 3px rgba(30,64,175,0.15);
  }}
  .stButton > button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(30,64,175,0.35);
  }}

  /* — cards genéricos — */
  .info-card {{
      background: {COLORS['surface']};
      border-radius: 14px;
      padding: 1rem 1.25rem;
      box-shadow: 0 1px 3px rgba(15,23,42,0.06);
      border: 1px solid {COLORS['border']};
      margin-bottom: 1rem;
  }}
  .info-card h4 {{
      color: {COLORS['text']};
      font-size: 0.95rem;
      margin: 0 0 0.4rem 0;
      font-weight: 600;
  }}
  .info-card p {{
      color: {COLORS['text_2']};
      font-size: 0.85rem;
      line-height: 1.5;
      margin: 0;
  }}

  /* — métricas KPI — */
  .metric-card {{
      background: {COLORS['surface']};
      border-radius: 14px;
      padding: 1.1rem 1.3rem;
      box-shadow: 0 1px 3px rgba(15,23,42,0.06);
      border-left: 4px solid {COLORS['primary_2']};
  }}

  /* — card de resultado — */
  .result-card {{
      background: linear-gradient(135deg, {COLORS['primary']} 0%, #1D4ED8 100%);
      color: white;
      border-radius: 18px;
      padding: 1.8rem 2rem;
      text-align: center;
      box-shadow: 0 8px 24px rgba(30,64,175,0.25);
  }}
  .result-card h1 {{ font-size: 2.2rem; margin: 0.4rem 0; font-weight: 700; }}
  .result-card p  {{ opacity: 0.92; margin: 0; }}

  /* — caption explicativa dos gráficos — */
  .chart-caption {{
      background: #F1F5F9;
      border-left: 3px solid {COLORS['primary_2']};
      padding: 0.7rem 1rem;
      border-radius: 6px;
      font-size: 0.82rem;
      color: {COLORS['text_2']};
      line-height: 1.55;
      margin-top: 0.5rem;
      margin-bottom: 1.5rem;
  }}
  .chart-caption b {{ color: {COLORS['primary']}; }}

  /* — explicação técnica curta (2 linhas) — */
  .ml-explain {{
      background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
      border-left: 4px solid {COLORS['primary']};
      padding: 0.8rem 1rem;
      border-radius: 8px;
      font-size: 0.83rem;
      color: {COLORS['text_2']};
      line-height: 1.5;
      margin: 0.6rem 0 1.2rem 0;
  }}
  .ml-explain b {{ color: {COLORS['primary']}; }}

  /* — divisores de seção na sidebar — */
  .section-header {{
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: {COLORS['primary']};
      margin: 1.1rem 0 0.3rem 0;
      padding-bottom: 0.3rem;
      border-bottom: 1px solid {COLORS['border']};
  }}

  h1, h2, h3, h4 {{ color: {COLORS['text']}; font-weight: 600; }}
  h3 {{ font-size: 1.2rem; margin-top: 0.5rem; }}

  /* — tab styling — */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent;
      border-radius: 8px 8px 0 0;
      padding: 0.6rem 1.2rem;
      font-weight: 500;
  }}
  .stTabs [aria-selected="true"] {{
      background: {COLORS['surface']};
      color: {COLORS['primary']};
      font-weight: 600;
  }}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNÇÕES DE CONVERSÃO — UX ↔ Modelo
# ============================================================================
# O modelo foi treinado com valores categóricos (1, 2, 3 para água;
# 0-3 para atividade; 0-2 para eletrônicos). A UX agora coleta valores
# humanos reais (litros, dias, horas) e converte para o que o modelo espera.

def liters_to_ch2o(liters: float) -> int:
    """Litros de água/dia → CH2O do modelo (1=<1L, 2=1-2L, 3=>2L)."""
    if liters < 1.0:
        return 1
    elif liters <= 2.0:
        return 2
    return 3


def days_to_faf(days: int) -> int:
    """Dias/semana de exercício → FAF (0=nenhum, 1=1-2, 2=3-4, 3=5+)."""
    if days == 0:
        return 0
    elif days <= 2:
        return 1
    elif days <= 4:
        return 2
    return 3


def hours_to_tue(hours: float) -> int:
    """Horas/dia com eletrônicos → TUE (0=0-2h, 1=3-5h, 2=>5h)."""
    if hours <= 2:
        return 0
    elif hours <= 5:
        return 1
    return 2


def bmi_class_color(bmi: float) -> tuple:
    """Retorna (cor, classificação) baseado no IMC."""
    for lo, hi, cor, nome in BMI_RANGES:
        if lo <= bmi < hi:
            return cor, nome
    return "#991B1B", "Obesidade III"


# ============================================================================
# CARREGAMENTO DO MODELO
# ============================================================================

@st.cache_resource
def load_model():
    """Carrega modelo XGBoost + metadados. Compatível com JSONs antigos."""
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_meta.json") as f:
        meta = json.load(f)
    # backward-compat: aceita meta de qualquer versão do pipeline
    if "model_accuracy" not in meta:
        meta["model_accuracy"] = meta.get("xgb_accuracy", 0.0)
    if "target_classes" not in meta:
        meta["target_classes"] = meta.get("target_order", TARGET_ORDER)
    if "target_labels_pt" not in meta:
        meta["target_labels_pt"] = TARGET_LABELS_PT
    return model, meta


try:
    model, meta = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ============================================================================
# INFORMAÇÕES CLÍNICAS POR CLASSE
# ============================================================================

RISK_INFO = {
    "Abaixo do Peso": {
        "risco": "⚠️ Atenção", "nivel": "Moderado", "cor": "#06B6D4",
        "desc": "IMC abaixo de 18,5. Risco de deficiências nutricionais, queda de imunidade e perda de massa muscular.",
        "rec": [
            "Avaliação nutricional especializada",
            "Plano alimentar hipercalórico balanceado",
            "Investigar causas (tireoide, absorção intestinal)",
            "Monitorar micronutrientes (ferro, B12, vitamina D)",
        ],
    },
    "Peso Normal": {
        "risco": "✅ Saudável", "nivel": "Baixo", "cor": "#10B981",
        "desc": "IMC entre 18,5 e 24,9. Faixa de menor risco metabólico segundo a OMS. Mantenha os hábitos atuais.",
        "rec": [
            "Atividade física regular (150 min/semana)",
            "Dieta variada com vegetais e proteínas magras",
            "Check-up clínico anual",
            "Manter qualidade do sono e hidratação",
        ],
    },
    "Sobrepeso I": {
        "risco": "⚠️ Leve", "nivel": "Leve", "cor": "#F59E0B",
        "desc": "IMC entre 25 e 27,5. Início do excesso de peso. Intervenção preventiva pode evitar a progressão para obesidade.",
        "rec": [
            "Aumentar atividade aeróbica para 150-200 min/sem",
            "Reduzir ultraprocessados e açúcares",
            "Acompanhamento nutricional",
            "Educação alimentar e mudança de hábitos",
        ],
    },
    "Sobrepeso II": {
        "risco": "🔶 Moderado", "nivel": "Moderado", "cor": "#F97316",
        "desc": "IMC entre 27,5 e 29,9. Risco elevado de doenças metabólicas (pré-diabetes, dislipidemia, hipertensão).",
        "rec": [
            "Programa estruturado de emagrecimento",
            "Avaliação cardiovascular completa",
            "Terapia comportamental",
            "Exames laboratoriais (glicemia, lipidograma)",
        ],
    },
    "Obesidade I": {
        "risco": "🔴 Alto", "nivel": "Alto", "cor": "#EF4444",
        "desc": "IMC entre 30 e 34,9. Risco significativo de diabetes tipo 2, hipertensão e doenças cardiovasculares.",
        "rec": [
            "Acompanhamento médico mensal",
            "Equipe multidisciplinar (nutricionista, educador físico, psicólogo)",
            "Avaliação de comorbidades",
            "Considerar farmacoterapia se indicado",
        ],
    },
    "Obesidade II": {
        "risco": "🔴 Muito Alto", "nivel": "Muito Alto", "cor": "#DC2626",
        "desc": "IMC entre 35 e 39,9. Comorbidades múltiplas prováveis. Aumento da mortalidade cardiovascular.",
        "rec": [
            "Acompanhamento clínico quinzenal",
            "Avaliação para cirurgia bariátrica (se IMC ≥ 35 com comorbidades)",
            "Suporte psicológico contínuo",
            "Plano nutricional rigoroso supervisionado",
        ],
    },
    "Obesidade III": {
        "risco": "⛔ Crítico", "nivel": "Crítico", "cor": "#991B1B",
        "desc": "IMC ≥ 40. Obesidade mórbida. Risco de vida significativamente aumentado. Requer intervenção urgente.",
        "rec": [
            "Avaliação imediata para cirurgia bariátrica",
            "Equipe multidisciplinar completa",
            "Internação para condução do tratamento",
            "Monitoramento intensivo de comorbidades",
        ],
    },
}


# ============================================================================
# HISTÓRICO DE SESSÃO
# ============================================================================

if "historico" not in st.session_state:
    st.session_state.historico = []


# ============================================================================
# SIDEBAR – COLETA DE DADOS DO PACIENTE
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=56)
    st.markdown(
        f"<h2 style='margin:0; color:{COLORS['primary']};'>ObesityIQ</h2>"
        f"<p style='color:{COLORS['muted']}; font-size:0.78rem; margin:0;'>"
        "Sistema Preditivo Clínico · POS TECH F04</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── 1. IDENTIFICAÇÃO ────────────────────────────────────────────────
    st.markdown('<div class="section-header">1. Identificação</div>', unsafe_allow_html=True)
    col_g, col_i = st.columns(2)
    with col_g:
        gender = st.selectbox(
            "Gênero biológico",
            ["Female", "Male"],
            format_func=lambda x: "Feminino" if x == "Female" else "Masculino",
            help="Gênero biológico do paciente. Usado pelo modelo (29,9% de importância).",
        )
    with col_i:
        age = st.number_input("Idade (anos)", 14, 80, 25, 1,
                              help="Idade em anos completos.")

    # ── 2. MEDIDAS ANTROPOMÉTRICAS ─────────────────────────────────────
    st.markdown('<div class="section-header">2. Medidas Antropométricas</div>',
                unsafe_allow_html=True)
    col_h, col_w = st.columns(2)
    with col_h:
        height = st.number_input(
            "Altura (m)", 1.40, 2.20, 1.70, 0.01,
            help="Estatura em metros. Ex.: 1,75",
        )
    with col_w:
        weight = st.number_input(
            "Peso (kg)", 30.0, 250.0, 70.0, 0.5,
            help="Peso atual em quilogramas.",
        )

    # IMC calculado dinamicamente
    bmi_preview = weight / (height ** 2)
    bmi_cor, bmi_classe = bmi_class_color(bmi_preview)
    st.markdown(
        f"<div style='background:{bmi_cor}15; border-left:4px solid {bmi_cor}; "
        f"padding:0.6rem 0.9rem; border-radius:8px; margin-top:0.6rem;'>"
        f"<div style='font-size:0.75rem; color:{COLORS['muted']}; "
        f"text-transform:uppercase; letter-spacing:0.08em;'>IMC calculado</div>"
        f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
        f"<span style='font-size:1.4rem; font-weight:700; color:{bmi_cor};'>"
        f"{bmi_preview:.1f}</span>"
        f"<span style='font-size:0.8rem; color:{bmi_cor}; font-weight:600;'>"
        f"{bmi_classe}</span></div></div>",
        unsafe_allow_html=True,
    )

    # ── 3. HISTÓRICO & GENÉTICA ────────────────────────────────────────
    st.markdown('<div class="section-header">3. Histórico Familiar</div>',
                unsafe_allow_html=True)
    family_history = st.radio(
        "Algum familiar próximo tem ou teve sobrepeso/obesidade?",
        ["no", "yes"], horizontal=True,
        format_func=lambda x: "Não" if x == "no" else "Sim",
        help="Pais, irmãos ou avós. Predisposição genética é forte preditor.",
    )

    # ── 4. ALIMENTAÇÃO ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">4. Hábitos Alimentares</div>',
                unsafe_allow_html=True)
    favc = st.radio(
        "Consome alimentos altamente calóricos com frequência?",
        ["no", "yes"], horizontal=True,
        format_func=lambda x: "Não" if x == "no" else "Sim",
        help="Frituras, fast-food, doces, refrigerantes e ultraprocessados.",
    )
    fcvc = st.select_slider(
        "Frequência de consumo de vegetais",
        options=[1, 2, 3],
        value=2,
        format_func=lambda x: {1: "Raramente", 2: "Às vezes", 3: "Sempre"}[x],
        help="Com que frequência você come vegetais nas refeições principais.",
    )
    ncp = st.slider(
        "Refeições principais por dia",
        1, 4, 3,
        help="Café da manhã, almoço, jantar e ceia. Conta apenas refeições completas.",
    )
    caec = st.selectbox(
        "Lanches entre as refeições",
        ["no", "Sometimes", "Frequently", "Always"],
        index=1,
        format_func=lambda x: {"no": "Nunca", "Sometimes": "Às vezes",
                                "Frequently": "Frequentemente", "Always": "Sempre"}[x],
        help="Beliscar entre as refeições principais.",
    )
    scc = st.radio(
        "Você monitora as calorias que consome?",
        ["no", "yes"], horizontal=True,
        format_func=lambda x: "Não" if x == "no" else "Sim",
        help="Uso de apps, anotações ou contagem deliberada de calorias.",
    )

    # ── 5. HIDRATAÇÃO ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">5. Hidratação</div>',
                unsafe_allow_html=True)
    water_liters = st.slider(
        "Quantos litros de água você bebe por dia?",
        min_value=0.5, max_value=4.0, value=2.0, step=0.5,
        format="%.1f L",
        help="Apenas água. Não conta café, refrigerantes ou sucos.",
    )
    ch2o = liters_to_ch2o(water_liters)
    # feedback visual com a categoria mapeada
    cat_agua = {1: "Baixa (<1L)", 2: "Adequada (1–2L)", 3: "Alta (>2L)"}[ch2o]
    cor_agua = {1: COLORS["danger"], 2: COLORS["success"], 3: COLORS["info"]}[ch2o]
    st.markdown(
        f"<p style='font-size:0.78rem; color:{cor_agua}; margin:-0.4rem 0 0.5rem 0;'>"
        f"💧 Categoria: <b>{cat_agua}</b></p>",
        unsafe_allow_html=True,
    )

    # ── 6. ATIVIDADE FÍSICA ────────────────────────────────────────────
    st.markdown('<div class="section-header">6. Atividade Física</div>',
                unsafe_allow_html=True)
    activity_days = st.slider(
        "Quantos dias por semana você se exercita?",
        min_value=0, max_value=7, value=2, step=1,
        format="%d dias",
        help="Conte qualquer atividade física estruturada (caminhada, academia, esporte).",
    )
    activity_intensity = st.selectbox(
        "Intensidade predominante das atividades",
        ["leve", "moderada", "intensa"],
        index=1,
        format_func=lambda x: {
            "leve":      "Leve (caminhada, alongamento)",
            "moderada":  "Moderada (corrida leve, musculação)",
            "intensa":   "Intensa (HIIT, esportes competitivos)",
        }[x],
        help="Esta informação complementa o quadro clínico (não entra no modelo).",
    )
    faf = days_to_faf(activity_days)
    cat_faf = {0: "Sedentário", 1: "Baixa frequência", 2: "Regular", 3: "Atleta"}[faf]
    cor_faf = {0: COLORS["danger"], 1: COLORS["warning"],
               2: COLORS["success"], 3: COLORS["info"]}[faf]
    st.markdown(
        f"<p style='font-size:0.78rem; color:{cor_faf}; margin:-0.4rem 0 0.5rem 0;'>"
        f"🏃 Perfil: <b>{cat_faf}</b></p>",
        unsafe_allow_html=True,
    )

    # ── 7. ESTILO DE VIDA ──────────────────────────────────────────────
    st.markdown('<div class="section-header">7. Estilo de Vida</div>',
                unsafe_allow_html=True)
    screen_hours = st.slider(
        "Horas/dia em dispositivos eletrônicos (lazer)",
        min_value=0.0, max_value=12.0, value=4.0, step=0.5,
        format="%.1f h",
        help="TV, celular, videogame, redes sociais. Não conte o trabalho.",
    )
    tue = hours_to_tue(screen_hours)

    smoke = st.radio("Você fuma?", ["no", "yes"], horizontal=True,
                     format_func=lambda x: "Não" if x == "no" else "Sim")
    calc = st.selectbox(
        "Frequência de consumo de álcool",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {"no": "Não bebo", "Sometimes": "Socialmente",
                                "Frequently": "Frequentemente", "Always": "Diariamente"}[x],
    )
    mtrans = st.selectbox(
        "Meio de transporte mais usado",
        ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
        format_func=lambda x: {
            "Public_Transportation": "🚌 Transporte Público",
            "Automobile":            "🚗 Carro",
            "Walking":               "🚶 A Pé",
            "Motorbike":             "🏍️ Moto",
            "Bike":                  "🚴 Bicicleta",
        }[x],
        help="Modo predominante de deslocamento diário.",
    )

    st.divider()
    predict_btn = st.button("🔍 Analisar Paciente", use_container_width=True)
    if st.session_state.historico:
        if st.button("🗑️ Limpar Histórico", use_container_width=True):
            st.session_state.historico = []
            st.rerun()


# ============================================================================
# HEADER E KPIs DO MODELO
# ============================================================================

st.markdown(
    f"<h1 style='margin-bottom:0;'>🏥 ObesityIQ</h1>"
    f"<p style='color:{COLORS['muted']}; margin-top:0; font-size:0.95rem;'>"
    "Sistema preditivo clínico de obesidade · "
    "POS TECH Data Analytics · Tech Challenge Fase 04</p>",
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error("⚠️ Modelo `model_xgb.pkl` não encontrado. Execute `pipeline_ml.py` para gerá-lo.")
    st.stop()

# KPIs com explicação técnica curta
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🎯 Acurácia no Teste", f"{meta['model_accuracy']*100:.1f}%",
              help="% de acertos no conjunto de teste (20% dos dados nunca vistos no treino).")
with c2:
    st.metric("📊 Cross-Validation",
              f"{meta['cv_mean']*100:.1f}% ± {meta['cv_std']*100:.1f}%",
              help="Média ± desvio em 5 dobras estratificadas. Mede generalização.")
with c3:
    st.metric("🧠 Algoritmo", "XGBoost",
              help="Extreme Gradient Boosting — ensemble de árvores de decisão.")
with c4:
    st.metric("📁 Features", str(len(meta["features"])),
              help="Quantidade de variáveis usadas para a predição.")

# Explicação do modelo em 2 linhas
st.markdown(
    "<div class='ml-explain'>"
    "<b>O que é XGBoost?</b> Algoritmo de aprendizado de máquina que combina centenas "
    "de árvores de decisão sequencialmente, corrigindo os erros das anteriores. "
    "É padrão-ouro em problemas tabulares clínicos pela alta acurácia e interpretabilidade."
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================================
# ABAS PRINCIPAIS
# ============================================================================

tab_pred, tab_insights, tab_hist, tab_sobre = st.tabs([
    "🔍 Predição Clínica",
    "📊 Insights do Modelo",
    "📋 Histórico da Sessão",
    "ℹ️ Sobre o Sistema",
])

# ╔════════════════════════════════════════════════════════════════════════╗
# ║ ABA 1 — PREDIÇÃO CLÍNICA                                                ║
# ╚════════════════════════════════════════════════════════════════════════╝

with tab_pred:
    if predict_btn:
        # Monta o input com os valores MAPEADOS para o modelo
        input_data = {
            "Gender":         gender,
            "Age":            age,
            "Height":         height,
            "Weight":         weight,
            "family_history": family_history,
            "FAVC":           favc,
            "FCVC":           fcvc,
            "NCP":            ncp,
            "CAEC":           caec,
            "SMOKE":          smoke,
            "CH2O":           ch2o,     # convertido de litros
            "SCC":            scc,
            "FAF":            faf,      # convertido de dias/semana
            "TUE":            tue,      # convertido de horas/dia
            "CALC":           calc,
            "MTRANS":         mtrans,
        }

        with st.spinner("Analisando dados do paciente..."):
            try:
                result = predict_single(input_data, model, meta)
            except Exception as e:
                st.error(f"Erro na predição: {e}")
                st.stop()

        label = result["label_pt"]
        probs = result["probabilities"]
        info  = RISK_INFO[label]
        cor   = info["cor"]

        # adiciona ao histórico
        st.session_state.historico.append({
            "Data/Hora":  datetime.now().strftime("%H:%M:%S"),
            "Gênero":     "M" if gender == "Male" else "F",
            "Idade":      age,
            "IMC":        round(bmi_preview, 1),
            "Diagnóstico": label,
            "Confiança":  f"{probs[label]:.1f}%",
            "Risco":      info["nivel"],
        })

        # ── Card de resultado principal ──────────────────────────────────
        col_res, col_kpi = st.columns([2, 1])
        with col_res:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {cor}DD, {cor});">
                <p style="font-size:0.9rem; opacity:0.9; text-transform:uppercase;
                          letter-spacing:0.1em;">Diagnóstico Preditivo</p>
                <h1>{label}</h1>
                <p style="font-size:1rem; margin-top:0.6rem;">{info['risco']} · Nível: <b>{info['nivel']}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{cor};">
                <p style="color:{COLORS['muted']}; font-size:0.78rem;
                          text-transform:uppercase; letter-spacing:0.08em; margin:0;">
                IMC do Paciente</p>
                <h2 style="margin:0.2rem 0; font-size:2.2rem; color:{cor};">{bmi_preview:.1f}</h2>
                <p style="color:{COLORS['muted']}; font-size:0.78rem; margin:0;">kg/m²</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card" style="margin-top:0.8rem; border-left-color:{cor};">
                <p style="color:{COLORS['muted']}; font-size:0.78rem;
                          text-transform:uppercase; letter-spacing:0.08em; margin:0;">
                Confiança da Predição</p>
                <h2 style="margin:0.2rem 0; font-size:2rem; color:{cor};">{probs[label]:.1f}%</h2>
                <p style="color:{COLORS['muted']}; font-size:0.78rem; margin:0;">
                Probabilidade da classe predita</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Gauge de IMC ──────────────────────────────────────────────────
        st.markdown("### Indicador Visual de IMC")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=bmi_preview,
            number={"suffix": " kg/m²", "font": {"size": 30, "color": cor}},
            title={"text": "Índice de Massa Corporal", "font": {"size": 14}},
            gauge={
                "axis": {"range": [10, 50], "tickwidth": 1, "tickfont": {"size": 11}},
                "bar":  {"color": cor, "thickness": 0.28},
                "bgcolor": "white",
                "steps": [
                    {"range": [10,  18.5], "color": "#CFFAFE"},
                    {"range": [18.5, 25],  "color": "#D1FAE5"},
                    {"range": [25,  30],   "color": "#FEF3C7"},
                    {"range": [30,  35],   "color": "#FECACA"},
                    {"range": [35,  40],   "color": "#FCA5A5"},
                    {"range": [40,  50],   "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": cor, "width": 4},
                    "thickness": 0.85,
                    "value": bmi_preview,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=40, b=10, l=30, r=30),
            paper_bgcolor="white", font=dict(family="Inter"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(
            "<div class='chart-caption'>"
            "<b>Como ler:</b> O ponteiro indica o IMC calculado por "
            "<b>Peso (kg) ÷ Altura² (m²)</b>. As faixas coloridas seguem a classificação "
            "oficial da OMS: verde (saudável, 18,5–24,9), amarelo (sobrepeso, 25–29,9), "
            "vermelho (obesidade, ≥ 30). O IMC é o preditor mais importante do modelo (51,8%)."
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Contexto clínico + recomendações ─────────────────────────────
        col_desc, col_rec = st.columns(2)
        with col_desc:
            st.markdown(
                f"<div class='info-card' style='border-left:4px solid {cor};'>"
                f"<h4>🩺 Contexto Clínico</h4>"
                f"<p>{info['desc']}</p></div>",
                unsafe_allow_html=True,
            )
        with col_rec:
            st.markdown(
                f"<div class='info-card' style='border-left:4px solid {cor};'>"
                f"<h4>📋 Recomendações Médicas</h4>"
                f"<p>{''.join(f'• {r}<br>' for r in info['rec'])}</p></div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Distribuição de probabilidades ───────────────────────────────
        col_prob, col_resumo = st.columns([3, 2])

        with col_prob:
            st.markdown("### Distribuição de Probabilidades por Classe")
            st.markdown(
                "<div class='ml-explain'>"
                "<b>O que é probabilidade da classe?</b> O modelo XGBoost atribui um "
                "% para cada um dos 7 níveis de obesidade. A classe predita é a de maior "
                "probabilidade. Probabilidades próximas indicam casos de fronteira."
                "</div>",
                unsafe_allow_html=True,
            )
            labels_sorted = TARGET_LABELS_PT
            values_sorted = [probs.get(l, 0) for l in labels_sorted]
            colors_sorted = [CLASS_COLORS[l] for l in labels_sorted]

            fig_bar = go.Figure(go.Bar(
                x=labels_sorted, y=values_sorted,
                marker_color=colors_sorted,
                marker_line_color="white", marker_line_width=1,
                text=[f"{v:.1f}%" for v in values_sorted],
                textposition="outside",
                textfont={"size": 11},
            ))
            fig_bar.update_layout(
                yaxis_title="Probabilidade (%)",
                yaxis=dict(range=[0, max(values_sorted) * 1.2]),
                xaxis_tickangle=-25,
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=20, b=10),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'>"
                "<b>Cálculo:</b> O XGBoost gera 7 valores de probabilidade que somam 100%, "
                "um para cada classe. A barra mais alta é o diagnóstico final. "
                "<b>Interpretação clínica:</b> se duas barras estão próximas (ex.: Sobrepeso II "
                "com 45% e Obesidade I com 40%), o paciente está em zona de transição e merece "
                "acompanhamento próximo."
                "</div>",
                unsafe_allow_html=True,
            )

        with col_resumo:
            st.markdown("### Resumo do Paciente")
            st.markdown(f"""
            <div class='info-card'>
              <p style='line-height:1.9; margin:0;'>
              <b>Gênero:</b> {'Masculino' if gender == 'Male' else 'Feminino'}<br>
              <b>Idade:</b> {age} anos<br>
              <b>Altura/Peso:</b> {height:.2f} m / {weight:.1f} kg<br>
              <b>IMC:</b> {bmi_preview:.1f} kg/m²<br>
              <b>Histórico familiar:</b> {'Sim' if family_history == 'yes' else 'Não'}<br>
              <b>Alim. calórica frequente:</b> {'Sim' if favc == 'yes' else 'Não'}<br>
              <b>Água/dia:</b> {water_liters:.1f} L<br>
              <b>Exercício:</b> {activity_days} dias/sem ({activity_intensity})<br>
              <b>Eletrônicos:</b> {screen_hours:.1f} h/dia<br>
              <b>Fuma:</b> {'Sim' if smoke == 'yes' else 'Não'}
              </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Estado inicial: sem predição ─────────────────────────────────
        st.markdown(
            "<div class='info-card' style='border-left:4px solid " + COLORS["primary_2"] + ";'>"
            "<h4>👈 Como usar</h4>"
            "<p>1. Preencha os dados do paciente na barra lateral.<br>"
            "2. Clique em <b>Analisar Paciente</b>.<br>"
            "3. Veja o diagnóstico, probabilidades e recomendações clínicas.</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Variáveis Mais Relevantes do Modelo")
        st.markdown(
            "<div class='ml-explain'>"
            "<b>O que é Feature Importance?</b> Mede o quanto cada variável contribuiu "
            "para as decisões do modelo, somando o ganho de informação em todas as "
            "árvores. Valores mais altos = variável mais determinante na predição."
            "</div>",
            unsafe_allow_html=True,
        )
        fi_data = pd.DataFrame(meta["feature_importance"]).head(10)
        fi_data["importance_pct"] = (fi_data["importance"] * 100).round(1)

        fig_fi = px.bar(
            fi_data.sort_values("importance"),
            x="importance_pct", y="feature", orientation="h",
            color="importance_pct",
            color_continuous_scale=["#DBEAFE", COLORS["primary"]],
            labels={"importance_pct": "Importância (%)", "feature": "Variável"},
            text="importance_pct",
        )
        fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                             textfont={"size": 11})
        fig_fi.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=420, coloraxis_showscale=False,
            margin=dict(l=10, r=80, t=20, b=10),
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown(
            "<div class='chart-caption'>"
            "<b>Interpretação:</b> O <b>IMC</b> (51,8%) domina as decisões — confirma a "
            "validade do indicador clínico clássico. <b>Gênero</b> (29,9%) reflete diferenças "
            "biológicas na distribuição das classes (homens em Obesidade II, mulheres em "
            "Obesidade III). As demais variáveis comportamentais somam ~18% e refinam o "
            "diagnóstico em casos limítrofes."
            "</div>",
            unsafe_allow_html=True,
        )

# ╔════════════════════════════════════════════════════════════════════════╗
# ║ ABA 2 — INSIGHTS DO MODELO                                              ║
# ╚════════════════════════════════════════════════════════════════════════╝

with tab_insights:
    st.markdown("### Comparativo de Desempenho dos Modelos Testados")
    st.markdown(
        "<div class='ml-explain'>"
        "<b>Por que comparar 3 modelos?</b> Treinar múltiplos algoritmos e escolher o "
        "melhor é a prática padrão em ML clínico. Aqui comparamos um modelo simples "
        "(Logistic Regression) com dois ensembles de árvores (Random Forest e XGBoost)."
        "</div>",
        unsafe_allow_html=True,
    )
    modelos_df = pd.DataFrame({
        "Modelo":         ["Logistic Regression", "Random Forest", "XGBoost (Final)"],
        "Acurácia Teste": [83.7, 98.3, 98.1],
        "CV Mean":        [84.1, 98.4, 98.6],
    })
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Acurácia no Teste (%)",
        x=modelos_df["Modelo"], y=modelos_df["Acurácia Teste"],
        marker_color=["#CBD5E1", "#60A5FA", COLORS["primary"]],
        text=[f"{v:.1f}%" for v in modelos_df["Acurácia Teste"]],
        textposition="outside",
    ))
    fig_comp.add_trace(go.Bar(
        name="Cross-Validation 5-fold (%)",
        x=modelos_df["Modelo"], y=modelos_df["CV Mean"],
        marker_color=["#94A3B8", "#3B82F6", "#1D4ED8"],
        text=[f"{v:.1f}%" for v in modelos_df["CV Mean"]],
        textposition="outside",
    ))
    fig_comp.update_layout(
        barmode="group",
        yaxis=dict(range=[70, 102], title="Acurácia (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=380, margin=dict(t=20, b=10),
        font=dict(family="Inter"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown(
        "<div class='chart-caption'>"
        "<b>Acurácia no Teste:</b> % de acertos em 422 pacientes nunca vistos no treino. "
        "<b>Cross-Validation:</b> média de 5 testes independentes no conjunto de treino — "
        "mede se o modelo generaliza bem. <b>Conclusão:</b> Random Forest e XGBoost empatam "
        "em performance, mas XGBoost foi escolhido por melhor interpretabilidade e "
        "regularização nativa, evitando overfitting."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Feature importance completa ──────────────────────────────────────
    st.markdown("### Importância de Todas as Variáveis (XGBoost)")
    st.markdown(
        "<div class='ml-explain'>"
        "<b>Como é calculado?</b> Somando, em todas as 300 árvores do modelo, "
        "quanto cada variável contribuiu para reduzir o erro nas divisões. "
        "Variáveis no topo são as mais decisivas."
        "</div>",
        unsafe_allow_html=True,
    )
    fi_all = pd.DataFrame(meta["feature_importance"])
    fi_all["importance_pct"] = (fi_all["importance"] * 100).round(2)

    fig_fi_all = px.bar(
        fi_all.sort_values("importance"),
        x="importance_pct", y="feature", orientation="h",
        color="importance_pct",
        color_continuous_scale=["#DBEAFE", COLORS["primary"]],
        labels={"importance_pct": "Importância (%)", "feature": "Variável"},
        text="importance_pct",
    )
    fig_fi_all.update_traces(texttemplate="%{text:.2f}%", textposition="outside",
                             textfont={"size": 10})
    fig_fi_all.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=520, coloraxis_showscale=False,
        margin=dict(l=10, r=80, t=10, b=10),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_fi_all, use_container_width=True)
    st.markdown(
        "<div class='chart-caption'>"
        "<b>Significado das variáveis:</b><br>"
        "• <b>BMI</b>: Índice de Massa Corporal (Peso/Altura²) — preditor clínico clássico.<br>"
        "• <b>Gender</b>: gênero biológico (homens=1, mulheres=0).<br>"
        "• <b>NCP</b>: número de refeições principais por dia.<br>"
        "• <b>MTRANS</b>: meio de transporte (proxy de sedentarismo).<br>"
        "• <b>FAVC</b>: consumo frequente de alimentos calóricos.<br>"
        "• <b>family_history</b>: histórico familiar de sobrepeso (genética).<br>"
        "• <b>FAF</b>: frequência de atividade física semanal.<br>"
        "• <b>CAEC</b>: lanches entre refeições.<br>"
        "• <b>CH2O</b>: consumo de água."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Insights principais ──────────────────────────────────────────────
    st.markdown("### 💡 Principais Achados do Dataset (2.111 pacientes)")
    insights = [
        ("🏋️ IMC é o preditor #1",
         "Responsável por 51,8% das decisões do modelo. Confirma a validade do indicador clássico da OMS, calculado por peso ÷ altura²."),
        ("👤 Gênero é o #2",
         "29,9% de importância. Homens concentram Obesidade Tipo II; mulheres, Obesidade Tipo III — diferenças hormonais e de composição corporal."),
        ("🧬 Genética determinante",
         "100% dos casos de Obesidade III têm histórico familiar de sobrepeso. Predisposição genética é forte fator de risco."),
        ("🏃 Sedentarismo progressivo",
         "Frequência de atividade física cai 38% do grupo Peso Normal até o grupo Obesidade III. Padrão estatisticamente significativo (p<0,001)."),
        ("🍔 Alimentação calórica",
         "99,7% dos pacientes com Obesidade III consomem alimentos altamente calóricos (ultraprocessados, fast-food) com frequência."),
        ("📅 Início precoce",
         "Sobrepeso aparece em média aos 23,4 anos no dataset. Intervenções na adolescência e início da vida adulta são fundamentais."),
    ]
    col_a, col_b = st.columns(2)
    for i, (titulo, texto) in enumerate(insights):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""
            <div class='info-card' style='border-left:4px solid {COLORS['primary_2']};'>
                <h4>{titulo}</h4>
                <p>{texto}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Tabela OMS de referência ─────────────────────────────────────────
    st.markdown("### 📏 Tabela de Referência IMC (OMS)")
    st.markdown(
        "<div class='ml-explain'>"
        "Classificação oficial da Organização Mundial da Saúde, baseada em estudos "
        "epidemiológicos que correlacionam IMC com mortalidade e morbidade cardiovascular."
        "</div>",
        unsafe_allow_html=True,
    )
    imc_ref = pd.DataFrame({
        "Classificação": ["Abaixo do Peso", "Peso Normal", "Sobrepeso",
                          "Obesidade I", "Obesidade II", "Obesidade III"],
        "Faixa IMC (kg/m²)": ["< 18,5", "18,5 – 24,9", "25,0 – 29,9",
                              "30,0 – 34,9", "35,0 – 39,9", "≥ 40,0"],
        "Risco de Saúde": ["Moderado", "Baixo", "Aumentado",
                           "Alto", "Muito Alto", "Crítico"],
        "Ação Recomendada": [
            "Avaliação nutricional",
            "Manter hábitos",
            "Prevenção / mudança de hábitos",
            "Tratamento clínico",
            "Tratamento clínico intensivo",
            "Avaliação cirúrgica",
        ],
    })
    st.dataframe(imc_ref, use_container_width=True, hide_index=True)


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ ABA 3 — HISTÓRICO                                                       ║
# ╚════════════════════════════════════════════════════════════════════════╝

with tab_hist:
    st.markdown("### 📋 Histórico de Análises desta Sessão")
    st.markdown(
        "<div class='ml-explain'>"
        "Esta tabela acumula todas as análises feitas durante a sessão atual no navegador. "
        "Não é persistida no servidor — fechar o navegador limpa o histórico."
        "</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.historico:
        st.info("Nenhuma análise registrada ainda. Use a aba **Predição Clínica** para começar.")
    else:
        df_hist = pd.DataFrame(st.session_state.historico)
        df_hist.index = range(1, len(df_hist) + 1)
        df_hist.index.name = "#"
        st.dataframe(df_hist, use_container_width=True)

        if len(df_hist) >= 2:
            st.markdown("#### Distribuição dos Diagnósticos na Sessão")
            contagem = df_hist["Diagnóstico"].value_counts().reset_index()
            contagem.columns = ["Diagnóstico", "Qtd"]
            contagem["cor"] = contagem["Diagnóstico"].map(CLASS_COLORS)

            fig_hist = go.Figure(go.Bar(
                x=contagem["Diagnóstico"], y=contagem["Qtd"],
                marker_color=contagem["cor"],
                marker_line_color="white", marker_line_width=1,
                text=contagem["Qtd"], textposition="outside",
            ))
            fig_hist.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=320, margin=dict(t=20, b=10),
                yaxis_title="Nº de Pacientes",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown(
                "<div class='chart-caption'>"
                "<b>Como ler:</b> Cada barra representa quantos pacientes da sessão foram "
                "classificados naquela categoria. Útil para identificar padrões no fluxo "
                "ambulatorial (ex.: predominância de Sobrepeso indica necessidade de "
                "programas preventivos)."
                "</div>",
                unsafe_allow_html=True,
            )

        csv = df_hist.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Exportar histórico (CSV)",
            data=csv,
            file_name=f"historico_obesityiq_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ╔════════════════════════════════════════════════════════════════════════╗
# ║ ABA 4 — SOBRE                                                           ║
# ╚════════════════════════════════════════════════════════════════════════╝

with tab_sobre:
    st.markdown("### Sobre o ObesityIQ")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class='info-card'>
        <h4>🎯 Objetivo</h4>
        <p>Sistema de apoio à decisão clínica que classifica pacientes em 7 níveis de obesidade
        a partir de 16 variáveis (antropométricas, alimentares e comportamentais).
        Desenvolvido para o Tech Challenge Fase 04 — POS TECH Data Analytics.</p>
        </div>

        <div class='info-card'>
        <h4>📊 Dataset</h4>
        <p>2.111 pacientes · 17 colunas · 0 valores ausentes.<br>
        Origem: UCI Machine Learning Repository.<br>
        7 classes balanceadas (entre 272 e 351 pacientes por classe).</p>
        </div>

        <div class='info-card'>
        <h4>🧠 Modelo</h4>
        <p><b>XGBoost</b> com 300 árvores, profundidade máxima 6, learning rate 0,1.
        Treinado em 80% dos dados, validado em 20% holdout + Cross-Validation 5-fold.
        Acurácia: 98,1% (teste) e 98,6% ± 0,7% (CV).</p>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
        <div class='info-card'>
        <h4>🔧 Feature Engineering</h4>
        <p>• <b>BMI</b> derivado (Peso/Altura²) — feature mais importante.<br>
        • Encoding binário: Gender, family_history, FAVC, SMOKE, SCC.<br>
        • Encoding ordinal: CAEC, CALC (4 níveis de frequência).<br>
        • Target ordenado por severidade clínica (0 a 6).</p>
        </div>

        <div class='info-card'>
        <h4>⚙️ Stack Técnica</h4>
        <p><b>Python 3.14</b> · <b>scikit-learn</b> · <b>XGBoost</b> · <b>Pandas</b> ·
        <b>Plotly</b> · <b>Streamlit</b>. Deploy em Streamlit Cloud.</p>
        </div>

        <div class='info-card'>
        <h4>⚠️ Aviso Médico</h4>
        <p>Este sistema é uma ferramenta de <b>apoio</b> à decisão clínica e <b>não substitui</b>
        avaliação médica presencial. Os resultados devem ser interpretados por profissionais
        de saúde no contexto completo do paciente.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Glossário Técnico")
    glossario = {
        "IMC (BMI)":         "Índice de Massa Corporal = Peso (kg) / Altura² (m²). Indicador clínico clássico.",
        "Acurácia":          "Percentual de classificações corretas sobre o total de amostras.",
        "Cross-Validation":  "Técnica que divide o treino em K partes (folds) e treina K vezes, alternando qual parte é usada como validação. Mede generalização.",
        "Feature Importance": "Métrica que indica quanto cada variável contribui para as decisões do modelo. Calculada pelo ganho médio de cada split nas árvores.",
        "XGBoost":           "Extreme Gradient Boosting. Algoritmo que treina árvores sequencialmente, cada uma corrigindo os erros das anteriores.",
        "Holdout":            "Parte dos dados (20% aqui) reservada para teste final, nunca vista durante o treinamento.",
        "Stratified Split":  "Divisão dos dados que preserva a proporção das classes em treino e teste.",
    }
    for termo, definicao in glossario.items():
        st.markdown(f"**{termo}** — {definicao}")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:{COLORS['muted']}; font-size:0.8rem;'>"
    "🏥 <b>ObesityIQ</b> · Tech Challenge Fase 04 · POS TECH Data Analytics · "
    "Leonardo Fernandes Sbardelotto · Modelo: XGBoost · Acurácia 98,1%"
    "</p>",
    unsafe_allow_html=True,
)
