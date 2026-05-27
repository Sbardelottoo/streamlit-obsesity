# 🏥 ObesityIQ — Tech Challenge Fase 04
**POS TECH Data Analytics | Leonardo Fernandes Sbardelotto**

Sistema preditivo de obesidade com Machine Learning, desenvolvido para auxiliar a equipe médica a diagnosticar o nível de obesidade de pacientes.

---

## 📁 Estrutura do Projeto

```
tc4-obesidade/
├── Obesity.csv                 # Dataset original
├── pipeline_ml.py              # Pipeline completo de ML
├── app_streamlit.py            # Aplicação preditiva (Streamlit)
├── dashboard_obesidade.html    # Painel analítico interativo
├── requirements.txt            # Dependências
├── model_xgb.pkl               # Modelo treinado (gerado pelo pipeline)
└── model_meta.json             # Metadados do modelo (gerado pelo pipeline)
```

---

## 🚀 Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo
```bash
python pipeline_ml.py
```
Isso gera: `model_xgb.pkl`, `model_meta.json` e `avaliacao_modelo.png`

### 3. Rodar a aplicação Streamlit
```bash
streamlit run app_streamlit.py
```

### 4. Abrir o Dashboard Analítico
Abra `dashboard_obesidade.html` diretamente no navegador.

---

## 📊 Resultados do Modelo

| Modelo               | Acurácia Teste | Cross-Val (5-fold) |
|----------------------|:--------------:|:------------------:|
| Logistic Regression  | 83,7%          | 84,1% ± 1,2%       |
| Random Forest        | 98,3%          | 98,4% ± 0,8%       |
| **XGBoost (Final)**  | **98,1%**      | **98,6% ± 0,7%**   |

> ✅ Critério do Tech Challenge: acurácia > 75% — **superado em 23 p.p.**

---

## 🧠 Pipeline de Machine Learning

### Feature Engineering
- **BMI (IMC):** feature derivada (Peso / Altura²) — torna-se o preditor #1 (51,8% de importância)
- **Arredondamento:** colunas ordinais com ruído decimal (FCVC, NCP, CH2O, FAF, TUE)
- **Encoding binário:** Gender, family_history, FAVC, SMOKE, SCC → 0/1
- **Encoding ordinal:** CAEC, CALC → {no:0, Sometimes:1, Frequently:2, Always:3}
- **Encoding nominal:** MTRANS → {Walking:0, Bike:1, Motorbike:2, Public_Transportation:3, Automobile:4}
- **Target encoding:** 7 classes ordenadas por severidade (0=Abaixo do Peso → 6=Obesidade III)

### Escolha do Modelo
XGBoost foi selecionado por:
- Melhor equilíbrio entre accuracy e generalização
- Regularização L1/L2 nativa (evita overfitting)
- Feature importance interpretável para a equipe médica
- Alta performance em datasets tabulares com classes desbalanceadas

---

## 💡 Principais Insights

1. **IMC é determinante:** explica 51,8% da predição isoladamente
2. **Genética:** 100% dos pacientes com Obesidade III têm histórico familiar
3. **Gênero:** homens concentram Obesidade II; mulheres, Obesidade III
4. **Sedentarismo:** frequência de atividade cai 38% do Peso Normal → Obesidade III
5. **Precocidade:** Sobrepeso já aparece em média aos 23,4 anos
6. **Alimentação:** 99,7% dos casos Obesidade III consomem alimentos calóricos regularmente

---

## 📋 Entregáveis

- [x] Pipeline de ML com feature engineering e treinamento
- [x] Modelo com acurácia > 75% (98,1% obtido)
- [x] Aplicação preditiva (Streamlit)
- [x] Dashboard analítico com insights para a equipe médica
- [ ] Deploy no Streamlit Cloud (adicionar link após deploy)
- [ ] Link do repositório GitHub
- [ ] Vídeo de apresentação (4–10 min)

---

## 🔗 Links

- **Streamlit App:** `https://[seu-usuario]-obesityiq.streamlit.app` *(após deploy)*
- **GitHub:** `https://github.com/[seu-usuario]/tc4-obesidade` *(após upload)*
- **Dashboard Analítico:** abrir `dashboard_obesidade.html` no navegador

---

*POS TECH Data Analytics · Tech Challenge Fase 04 · 2025*
