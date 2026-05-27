"""
================================================================================
TECH CHALLENGE – FASE 04 | POS TECH DATA ANALYTICS
Modelo Preditivo de Obesidade
Autor: Leonardo Fernandes Sbardelotto
================================================================================
Pipeline completo: Feature Engineering → Treinamento → Avaliação → Export
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 0. CONFIGURAÇÕES GLOBAIS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = "Obesity.csv"          # ajuste se necessário
MODEL_OUTPUT = "model_xgb.pkl"
META_OUTPUT = "model_meta.json"

TARGET_ORDER = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]

TARGET_LABELS_PT = [
    "Abaixo do Peso", "Peso Normal",
    "Sobrepeso I", "Sobrepeso II",
    "Obesidade I", "Obesidade II", "Obesidade III",
]

FEATURES = [
    "Gender", "Age", "Height", "Weight",
    "family_history", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC",
    "FAF", "TUE", "CALC", "MTRANS", "BMI",
]


# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    """Carrega o CSV e faz validações básicas."""
    df = pd.read_csv(path)
    print(f"[1/5] Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"      Valores ausentes: {df.isnull().sum().sum()}")
    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todo o pré-processamento e engenharia de features.

    Transformações:
      - Criação de BMI (feature derivada altamente preditiva)
      - Arredondamento de colunas ordinais com ruído decimal
      - Encoding binário (yes/no, Male/Female)
      - Encoding ordinal (CAEC, CALC)
      - Encoding nominal (MTRANS)
      - Encoding do target (ordinal pela severidade)
    """
    df = df.copy()

    # ── Feature derivada: IMC ──────────────────────────────────
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # ── Arredondar colunas ordinais com ruído decimal ──────────
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        df[col] = df[col].round().astype(int)

    # ── Encoding binário ───────────────────────────────────────
    binary_cols = {
        "Gender":         lambda x: (x == "Male").astype(int),
        "family_history": lambda x: (x == "yes").astype(int),
        "FAVC":           lambda x: (x == "yes").astype(int),
        "SMOKE":          lambda x: (x == "yes").astype(int),
        "SCC":            lambda x: (x == "yes").astype(int),
    }
    for col, fn in binary_cols.items():
        df[col] = fn(df[col])

    # ── Encoding ordinal ───────────────────────────────────────
    ordinal_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df["CAEC"] = df["CAEC"].map(ordinal_map)
    df["CALC"] = df["CALC"].map(ordinal_map)

    # ── Encoding nominal: MTRANS (intensidade física implícita) ─
    mtrans_map = {
        "Walking": 0, "Bike": 1, "Motorbike": 2,
        "Public_Transportation": 3, "Automobile": 4,
    }
    df["MTRANS"] = df["MTRANS"].map(mtrans_map)

    # ── Encoding do target ─────────────────────────────────────
    target_map = {label: i for i, label in enumerate(TARGET_ORDER)}
    df["target"] = df["Obesity"].map(target_map)

    print(f"[2/5] Feature engineering concluído. Total de features: {len(FEATURES)}")
    return df


# ============================================================
# 3. SPLIT TREINO / TESTE
# ============================================================

def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[3/5] Split: treino={len(X_train)} | teste={len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================
# 4. TREINAMENTO E AVALIAÇÃO DE MODELOS
# ============================================================

def train_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Treina e compara três modelos:
      - Logistic Regression (baseline)
      - Random Forest
      - XGBoost (modelo final)
    """
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":             XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0,
        ),
    }

    print("\n[4/5] Treinamento e Avaliação:")
    print(f"{'Modelo':<25} {'Acc. Teste':>10} {'CV Mean':>10} {'CV Std':>10}")
    print("─" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc_test = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="accuracy")
        results[name] = {
            "model": model,
            "accuracy": acc_test,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "predictions": preds,
        }
        print(f"{name:<25} {acc_test:>10.4f} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f}")

    # Relatório detalhado do melhor modelo (XGBoost)
    print("\n── Classification Report (XGBoost) ──────────────────────")
    print(classification_report(
        y_test,
        results["XGBoost"]["predictions"],
        target_names=TARGET_LABELS_PT,
    ))

    # Feature Importance
    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": results["XGBoost"]["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n── Feature Importance (XGBoost) ─────────────────────────")
    print(fi.to_string(index=False))

    return results


# ============================================================
# 5. GERAÇÃO DE GRÁFICOS
# ============================================================

def generate_plots(results, X_test, y_test):
    """Salva gráficos de avaliação."""
    xgb_model = results["XGBoost"]["model"]
    preds = results["XGBoost"]["predictions"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Avaliação do Modelo XGBoost – Obesidade", fontsize=14)

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[0],
                xticklabels=TARGET_LABELS_PT, yticklabels=TARGET_LABELS_PT,
                cmap="Blues")
    axes[0].set_title("Matriz de Confusão")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # Feature Importance
    fi = pd.DataFrame({"feature": FEATURES,
                        "importance": xgb_model.feature_importances_}
                      ).sort_values("importance")
    axes[1].barh(fi["feature"], fi["importance"], color="#2563EB")
    axes[1].set_title("Feature Importance (XGBoost)")
    axes[1].set_xlabel("Importância")

    plt.tight_layout()
    plt.savefig("avaliacao_modelo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo em: avaliacao_modelo.png")


# ============================================================
# 6. EXPORT DO MODELO E METADADOS
# ============================================================

def export_model(results):
    """Exporta o modelo XGBoost e os metadados em JSON."""
    model = results["XGBoost"]["model"]

    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(model, f)

    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    meta = {
        "features":           FEATURES,
        "target_classes":     TARGET_ORDER,
        "target_labels_pt":   TARGET_LABELS_PT,
        "target_map":         {v: i for i, v in enumerate(TARGET_ORDER)},
        "model_accuracy":     round(results["XGBoost"]["accuracy"], 4),
        "cv_mean":            round(results["XGBoost"]["cv_mean"], 4),
        "cv_std":             round(results["XGBoost"]["cv_std"], 4),
        "feature_importance": fi.to_dict("records"),
    }

    with open(META_OUTPUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[5/5] Modelo exportado → {MODEL_OUTPUT}")
    print(f"      Metadados exportados → {META_OUTPUT}")
    print(f"\n✅  Acurácia final: {results['XGBoost']['accuracy']*100:.2f}%")
    return model, meta


# ============================================================
# FUNÇÃO DE PREDIÇÃO (usada pelo Streamlit)
# ============================================================

def predict_single(input_dict: dict, model=None, meta=None):
    """
    Realiza predição para um único paciente.

    Args:
        input_dict: dict com as features no formato ORIGINAL (antes do encoding).
                    Exemplo: {"Gender": "Female", "Age": 25, ...}
        model: modelo XGBoost (se None, carrega do disco)
        meta:  metadados (se None, carrega do disco)

    Returns:
        dict com "class", "label_pt", "probabilities"
    """
    if model is None:
        with open(MODEL_OUTPUT, "rb") as f:
            model = pickle.load(f)
    if meta is None:
        with open(META_OUTPUT) as f:
            meta = json.load(f)

    df = pd.DataFrame([input_dict])

    # Aplicar as mesmas transformações do pipeline
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE"]:
        df[col] = df[col].round().astype(int)
    df["Gender"]         = (df["Gender"] == "Male").astype(int)
    df["family_history"] = (df["family_history"] == "yes").astype(int)
    df["FAVC"]           = (df["FAVC"] == "yes").astype(int)
    df["SMOKE"]          = (df["SMOKE"] == "yes").astype(int)
    df["SCC"]            = (df["SCC"] == "yes").astype(int)
    ordinal_map  = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    df["CAEC"]   = df["CAEC"].map(ordinal_map)
    df["CALC"]   = df["CALC"].map(ordinal_map)
    mtrans_map   = {"Walking": 0, "Bike": 1, "Motorbike": 2,
                    "Public_Transportation": 3, "Automobile": 4}
    df["MTRANS"] = df["MTRANS"].map(mtrans_map)

    X = df[FEATURES]
    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()

    return {
        "class":         pred_class,
        "label_en":      meta["target_classes"][pred_class],
        "label_pt":      meta["target_labels_pt"][pred_class],
        "probabilities": {
            meta["target_labels_pt"][i]: round(p * 100, 1)
            for i, p in enumerate(proba)
        },
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TECH CHALLENGE FASE 04 – Pipeline de Machine Learning")
    print("=" * 60)

    df = load_data(DATA_PATH)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)
    results = train_models(X_train, X_test, y_train, y_test)
    generate_plots(results, X_test, y_test)
    model, meta = export_model(results)

    # Exemplo de predição
    exemplo = {
        "Gender": "Male", "Age": 30, "Height": 1.75, "Weight": 95,
        "family_history": "yes", "FAVC": "yes", "FCVC": 2, "NCP": 3,
        "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2, "SCC": "no",
        "FAF": 1, "TUE": 1, "CALC": "Sometimes", "MTRANS": "Public_Transportation",
    }
    pred = predict_single(exemplo, model, meta)
    print(f"\nExemplo de predição: {pred['label_pt']} ({pred['class']})")
    print("Probabilidades:", pred["probabilities"])
