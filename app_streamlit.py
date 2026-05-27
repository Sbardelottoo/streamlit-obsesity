"""
================================================================================
TECH CHALLENGE – FASE 04 | POS TECH DATA ANALYTICS
Aplicação Preditiva de Obesidade – Streamlit App
Autor: Leonardo Fernandes Sbardelotto
================================================================================
Execute com:  streamlit run app_streamlit.py
"""

import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Importa função de predição do pipeline ─────────────────────────────────
from pipeline_ml import predict_single, FEATURES, TARGET_LABELS_PT, TARGET_ORDER

# ===========================================================================
# CONFIGURAÇÃO DA PÁGINA
# ===========================================================================

st.set_page_config(
    page_title="ObesityIQ – Preditor de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=DM+Mono&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background-color: #F8FAFC; }
  .stButton > button {
      background: linear-gradient(135deg, #1E40AF, #3B82F6);
      color: white; border: none; border-radius: 12px;
      padding: 0.75rem 2rem; font-size: 1rem; font-weight: 600;
      width: 100%; transition: all 0.2s;
  }
  .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(59,130,246,0.35); }
  .metric-card {
      background: white; border-radius: 16px;
      padding: 1.2rem 1.5rem; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
      border-left: 4px solid #3B82F6;
  }
  .result-card {
      background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 100%);
      color: white; border-radius: 20px; padding: 2rem;
      text-align: center; box-shadow: 0 10px 30px rgba(30,64,175,0.3);
  }
  .result-card h1 { font-size: 2.5rem; margin: 0.5rem 0; }
  .result-card p { opacity: 0.85; margin: 0; }
  .bmi-badge {
      display: inline-block; padding: 0.3rem 1rem;
      border-radius: 999px; font-weight: 700; font-size: 0.9rem;
  }
  h2, h3 { color: #1E3A5F; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# CARREGAMENTO DO MODELO
# ===========================================================================

@st.cache_resource
def load_model():
    with open("model_xgb.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


try:
    model, meta = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ===========================================================================
# PALETA DE CORES POR CLASSE
# ===========================================================================

CLASS_COLORS = {
    "Abaixo do Peso": "#06B6D4",
    "Peso Normal":    "#10B981",
    "Sobrepeso I":    "#F59E0B",
    "Sobrepeso II":   "#F97316",
    "Obesidade I":    "#EF4444",
    "Obesidade II":   "#DC2626",
    "Obesidade III":  "#991B1B",
}

RISK_INFO = {
    "Abaixo do Peso": {
        "risco": "⚠️ Moderado", "cor": "#06B6D4",
        "desc": "IMC abaixo da faixa saudável. Risco de deficiências nutricionais e imunidade comprometida.",
        "rec": ["Avaliação nutricional", "Dieta hipercalórica balanceada", "Monitoramento de micronutrientes"],
    },
    "Peso Normal": {
        "risco": "✅ Baixo", "cor": "#10B981",
        "desc": "IMC dentro da faixa saudável. Mantenha os hábitos atuais.",
        "rec": ["Manter atividade física regular", "Dieta equilibrada", "Check-up anual"],
    },
    "Sobrepeso I": {
        "risco": "⚠️ Leve", "cor": "#F59E0B",
        "desc": "Início de excesso de peso. Intervenção preventiva pode evitar progressão.",
        "rec": ["150 min/sem de atividade aeróbica", "Reduzir alimentos ultraprocessados", "Acompanhamento nutricional"],
    },
    "Sobrepeso II": {
        "risco": "🔶 Moderado", "cor": "#F97316",
        "desc": "Risco elevado de doenças metabólicas. Intervenção necessária.",
        "rec": ["Programa de emagrecimento estruturado", "Avaliação cardiovascular", "Terapia comportamental"],
    },
    "Obesidade I": {
        "risco": "🔴 Alto", "cor": "#EF4444",
        "desc": "Risco significativo de diabetes tipo 2, hipertensão e doenças cardiovasculares.",
        "rec": ["Acompanhamento médico regular", "Programa multidisciplinar", "Avaliação de comorbidades"],
    },
    "Obesidade II": {
        "risco": "🔴 Muito Alto", "cor": "#DC2626",
        "desc": "Risco muito alto. Comorbidades múltiplas prováveis.",
        "rec": ["Acompanhamento médico semanal", "Possível indicação cirúrgica", "Suporte psicológico intensivo"],
    },
    "Obesidade III": {
        "risco": "⛔ Crítico", "cor": "#991B1B",
        "desc": "Obesidade mórbida. Risco de vida aumentado. Intervenção urgente necessária.",
        "rec": ["Avaliação cirúrgica imediata", "Internação para tratamento", "Equipe multidisciplinar completa"],
    },
}


# ===========================================================================
# SIDEBAR – INPUTS DO PACIENTE
# ===========================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.title("🏥 ObesityIQ")
    st.caption("Sistema Preditivo de Obesidade | POS TECH Fase 04")
    st.divider()

    st.subheader("📋 Dados do Paciente")

    # Dados antropométricos
    col1, col2 = st.columns(2)
    with col1:
        gender    = st.selectbox("Gênero", ["Female", "Male"],
                                  format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
        height    = st.number_input("Altura (m)", 1.40, 2.00, 1.70, 0.01)
    with col2:
        age       = st.number_input("Idade", 14, 80, 25, 1)
        weight    = st.number_input("Peso (kg)", 30.0, 200.0, 70.0, 0.5)

    bmi_preview = weight / (height ** 2)
    st.info(f"**IMC calculado:** {bmi_preview:.1f} kg/m²")

    st.divider()
    st.subheader("🍎 Hábitos Alimentares")

    family_history = st.radio("Histórico familiar de sobrepeso?",
                               ["no", "yes"], horizontal=True,
                               format_func=lambda x: "Não" if x == "no" else "Sim")
    favc = st.radio("Consome alimentos calóricos frequentemente?",
                    ["no", "yes"], horizontal=True,
                    format_func=lambda x: "Não" if x == "no" else "Sim")
    fcvc = st.slider("Frequência de consumo de vegetais (1=raramente | 3=sempre)", 1, 3, 2)
    ncp  = st.slider("Refeições principais por dia", 1, 4, 3)
    caec = st.selectbox("Come entre as refeições?",
                         ["no", "Sometimes", "Frequently", "Always"],
                         index=1,
                         format_func=lambda x: {"no":"Não", "Sometimes":"Às vezes",
                                                 "Frequently":"Frequentemente", "Always":"Sempre"}[x])

    st.divider()
    st.subheader("💧 Hidratação & Hábitos")

    ch2o = st.slider("Água consumida por dia (L)", 1, 3, 2,
                      help="1=<1L | 2=1-2L | 3=>2L")
    smoke = st.radio("Fuma?", ["no", "yes"], horizontal=True,
                     format_func=lambda x: "Não" if x == "no" else "Sim")
    scc   = st.radio("Monitora calorias ingeridas?", ["no", "yes"], horizontal=True,
                     format_func=lambda x: "Não" if x == "no" else "Sim")

    st.divider()
    st.subheader("🏃 Atividade & Estilo de Vida")

    faf    = st.slider("Frequência de atividade física (dias/sem)", 0, 3, 1,
                       help="0=Nenhuma | 1=1-2x | 2=3-4x | 3=5x+")
    tue    = st.slider("Tempo com dispositivos eletrônicos (h/dia)", 0, 2, 1,
                       help="0=0-2h | 1=3-5h | 2=>5h")
    calc   = st.selectbox("Frequência de consumo de álcool?",
                           ["no", "Sometimes", "Frequently", "Always"],
                           format_func=lambda x: {"no":"Não", "Sometimes":"Às vezes",
                                                   "Frequently":"Frequentemente", "Always":"Sempre"}[x])
    mtrans = st.selectbox("Transporte habitual?",
                           ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"],
                           format_func=lambda x: {
                               "Public_Transportation":"Transporte Público",
                               "Automobile":"Carro", "Walking":"A Pé",
                               "Motorbike":"Moto", "Bike":"Bicicleta"
                           }[x])

    st.divider()
    predict_btn = st.button("🔍 Analisar Paciente", use_container_width=True)


# ===========================================================================
# ÁREA PRINCIPAL
# ===========================================================================

st.title("🏥 ObesityIQ – Sistema Preditivo de Obesidade")
st.caption("POS TECH Data Analytics · Tech Challenge Fase 04 · Leonardo Fernandes Sbardelotto")

if not model_loaded:
    st.error("⚠️ Modelo não encontrado. Execute primeiro o `pipeline_ml.py` para gerar `model_xgb.pkl`.")
    st.stop()

# ── Métricas do Modelo (header) ────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🎯 Acurácia do Modelo", f"{meta['model_accuracy']*100:.1f}%")
with c2:
    st.metric("📊 Cross-Val (5-fold)", f"{meta['cv_mean']*100:.1f}% ± {meta['cv_std']*100:.1f}%")
with c3:
    st.metric("🧠 Algoritmo", "XGBoost")
with c4:
    st.metric("📁 Features", str(len(meta['features'])))

st.divider()

# ===========================================================================
# RESULTADO DA PREDIÇÃO
# ===========================================================================

if predict_btn:
    input_data = {
        "Gender": gender, "Age": age, "Height": height, "Weight": weight,
        "family_history": family_history, "FAVC": favc, "FCVC": fcvc,
        "NCP": ncp, "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o,
        "SCC": scc, "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
    }

    with st.spinner("Analisando dados do paciente..."):
        result = predict_single(input_data, model, meta)

    label  = result["label_pt"]
    probs  = result["probabilities"]
    info   = RISK_INFO[label]
    cor    = info["cor"]

    # ── Card de resultado ──────────────────────────────────────────────────
    col_res, col_bmi = st.columns([2, 1])

    with col_res:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {cor}CC, {cor});">
            <p style="font-size:1rem; opacity:0.9; margin-bottom:0.5rem;">Diagnóstico Preditivo</p>
            <h1>{label}</h1>
            <p style="font-size:1.1rem; margin-top:0.5rem;">{info['risco']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_bmi:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:0; border-left-color:{cor};">
            <p style="color:#64748B; font-size:0.85rem; margin:0;">IMC do Paciente</p>
            <h2 style="margin:0.25rem 0; font-size:2.5rem; color:{cor};">{bmi_preview:.1f}</h2>
            <p style="color:#64748B; font-size:0.8rem; margin:0;">kg/m²</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem; border-left-color:{cor};">
            <p style="color:#64748B; font-size:0.85rem; margin:0;">Confiança</p>
            <h2 style="margin:0.25rem 0; font-size:2rem; color:{cor};">{probs[label]:.1f}%</h2>
            <p style="color:#64748B; font-size:0.8rem; margin:0;">probabilidade</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Descrição clínica + recomendações ─────────────────────────────────
    col_desc, col_rec = st.columns(2)
    with col_desc:
        st.info(f"**Contexto Clínico**\n\n{info['desc']}")
    with col_rec:
        with st.expander("📋 Recomendações Médicas", expanded=True):
            for r in info["rec"]:
                st.write(f"• {r}")

    st.divider()

    # ── Gráfico de probabilidades ──────────────────────────────────────────
    col_prob, col_input = st.columns([3, 2])

    with col_prob:
        st.subheader("📊 Distribuição de Probabilidades")
        labels_sorted = TARGET_LABELS_PT
        values_sorted = [probs.get(l, 0) for l in labels_sorted]
        colors_sorted = [CLASS_COLORS[l] for l in labels_sorted]

        fig = go.Figure(go.Bar(
            x=labels_sorted, y=values_sorted,
            marker_color=colors_sorted,
            text=[f"{v:.1f}%" for v in values_sorted],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Probabilidade (%)",
            xaxis_tickangle=-30,
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(t=20, b=10),
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_input:
        st.subheader("🧾 Resumo do Paciente")
        dados_resumo = {
            "Gênero":          "Masculino" if gender == "Male" else "Feminino",
            "Idade":           f"{age} anos",
            "Altura / Peso":   f"{height:.2f}m / {weight:.1f}kg",
            "IMC":             f"{bmi_preview:.1f} kg/m²",
            "Histórico familiar": "Sim" if family_history == "yes" else "Não",
            "Alim. calórica":  "Sim" if favc == "yes" else "Não",
            "Atividade física": f"Nível {faf}/3",
            "Água/dia":        f"Nível {ch2o}/3",
            "Fumo":            "Sim" if smoke == "yes" else "Não",
        }
        for k, v in dados_resumo.items():
            st.markdown(f"**{k}:** {v}")

else:
    # ── Estado inicial ─────────────────────────────────────────────────────
    st.info("👈 Preencha os dados do paciente na barra lateral e clique em **Analisar Paciente**.")

    # Exibe feature importance como card inicial
    st.subheader("📌 Variáveis mais importantes para o modelo")
    fi_data = pd.DataFrame(meta["feature_importance"]).head(10)
    fi_data["importance_pct"] = (fi_data["importance"] * 100).round(1)

    fig = px.bar(
        fi_data.sort_values("importance"),
        x="importance_pct", y="feature",
        orientation="h",
        color="importance_pct",
        color_continuous_scale=["#DBEAFE", "#1E40AF"],
        labels={"importance_pct": "Importância (%)", "feature": "Feature"},
        text="importance_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, coloraxis_showscale=False,
        margin=dict(l=10, r=60, t=20, b=10),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🏥 ObesityIQ · Desenvolvido para o Tech Challenge Fase 04 · POS TECH Data Analytics · "
    "Leonardo Fernandes Sbardelotto · Modelo: XGBoost (Acc. 98.1%)"
)
