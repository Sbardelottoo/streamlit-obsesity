"""
================================================================================
ObesityIQ — Helpers Clínicos e de Persistência
================================================================================
Módulo de apoio com:
  - Validação e formatação de CPF
  - Cálculos clínicos (TMB Mifflin-St Jeor, TDEE, plano dieta/exercício)
  - Score heurístico de progressão de grupos
  - Persistência JSON por CPF (banco simples)
================================================================================
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PACIENTES_DB = Path("pacientes.json")


# ============================================================================
# CPF — validação e formatação
# ============================================================================

def limpar_cpf(cpf: str) -> str:
    """Remove tudo que não for dígito."""
    return re.sub(r"\D", "", cpf or "")


def formatar_cpf_visual(cpf: str) -> str:
    """Formata como 000.000.000-00. Aceita parcial."""
    d = limpar_cpf(cpf)[:11]
    if len(d) <= 3:
        return d
    if len(d) <= 6:
        return f"{d[:3]}.{d[3:]}"
    if len(d) <= 9:
        return f"{d[:3]}.{d[3:6]}.{d[6:]}"
    return f"{d[:3]}.{d[3:6]}.{d[6:9]}-{d[9:]}"


def validar_cpf(cpf: str) -> bool:
    """Valida CPF pelos dígitos verificadores."""
    d = limpar_cpf(cpf)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for i in (9, 10):
        s = sum(int(d[j]) * ((i + 1) - j) for j in range(i))
        v = (s * 10) % 11
        if v == 10:
            v = 0
        if v != int(d[i]):
            return False
    return True


# ============================================================================
# Cálculos clínicos — TMB, TDEE, plano
# ============================================================================

FATOR_ATIVIDADE = {
    0: 1.20,  # sedentário (FAF=0)
    1: 1.375, # leve (1-2 dias)
    2: 1.55,  # moderado (3-4 dias)
    3: 1.725, # intenso (5+ dias)
}

KCAL_POR_MIN = {
    "leve":      3.5,   # caminhada leve
    "moderada":  6.0,   # caminhada rápida, bike leve
    "intensa":  10.0,   # corrida, spinning
}

IMC_SAUDAVEL_TETO = 24.9


def calcular_tmb(peso_kg: float, altura_m: float, idade: int, gender: str) -> float:
    """Taxa Metabólica Basal — Mifflin-St Jeor."""
    altura_cm = altura_m * 100
    base = 10 * peso_kg + 6.25 * altura_cm - 5 * idade
    return base + (5 if gender == "Male" else -161)


def calcular_tdee(tmb: float, faf: int) -> float:
    """Total Daily Energy Expenditure — TMB × fator atividade."""
    return tmb * FATOR_ATIVIDADE.get(faf, 1.2)


def peso_meta_saudavel(altura_m: float) -> float:
    """Peso correspondente ao teto do IMC saudável (24,9)."""
    return round(IMC_SAUDAVEL_TETO * altura_m ** 2, 1)


def gerar_plano_clinico(
    peso_atual: float,
    peso_meta: float,
    altura_m: float,
    idade: int,
    gender: str,
    faf: int,
    intensidade: str,
    prazo_semanas: int,
) -> dict:
    """
    Gera plano de dieta e exercício para atingir o peso-meta no prazo.

    Returns dict com:
      - tmb, tdee, deficit_dia, kcal_alvo, perda_semanal
      - dieta_kcal, exercicio_kcal_dia, exercicio_min_dia, exercicio_min_semana
      - dias_exercicio, alertas (lista)
      - viavel (bool)
    """
    alertas: list[str] = []

    tmb = calcular_tmb(peso_atual, altura_m, idade, gender)
    tdee = calcular_tdee(tmb, faf)

    perda_total = peso_atual - peso_meta
    if perda_total <= 0:
        return {
            "manutencao": True,
            "tmb": round(tmb),
            "tdee": round(tdee),
            "kcal_alvo": round(tdee),
            "mensagem": "Peso atual já está abaixo ou igual à meta — foco em manutenção.",
            "alertas": [],
            "viavel": True,
        }

    deficit_dia = (perda_total * 7700) / (prazo_semanas * 7)
    perda_semanal = perda_total / prazo_semanas

    if deficit_dia > 1000:
        alertas.append(
            f"⚠️ Déficit calórico de {deficit_dia:.0f} kcal/dia é agressivo. "
            f"Recomendado: até 1.000 kcal/dia. Considere aumentar o prazo."
        )
    if perda_semanal > 1.0:
        alertas.append(
            f"⚠️ Perda semanal projetada de {perda_semanal:.1f} kg excede a faixa segura "
            f"(0,5–1,0 kg/semana). Risco de perda de massa muscular."
        )

    kcal_alvo = tdee - deficit_dia
    kcal_min = 1500 if gender == "Male" else 1200
    if kcal_alvo < kcal_min:
        alertas.append(
            f"⚠️ Meta calórica ({kcal_alvo:.0f} kcal) abaixo do mínimo seguro "
            f"({kcal_min} kcal). Aumente o prazo ou ajuste a meta de peso."
        )
        kcal_alvo = kcal_min

    # Split 70% dieta, 30% exercício
    deficit_exercicio = deficit_dia * 0.30
    deficit_dieta = deficit_dia * 0.70
    dieta_kcal = tdee - deficit_dieta

    kcal_min_ex = KCAL_POR_MIN.get(intensidade, 6.0)
    exercicio_min_dia = deficit_exercicio / kcal_min_ex
    dias_exercicio = 5  # padrão clínico
    exercicio_min_semana = exercicio_min_dia * 7
    exercicio_min_por_sessao = exercicio_min_semana / dias_exercicio

    if exercicio_min_por_sessao < 20:
        dias_exercicio = 3
        exercicio_min_por_sessao = exercicio_min_semana / dias_exercicio

    return {
        "manutencao": False,
        "tmb": round(tmb),
        "tdee": round(tdee),
        "deficit_dia": round(deficit_dia),
        "kcal_alvo": round(kcal_alvo),
        "dieta_kcal": round(dieta_kcal),
        "exercicio_kcal_dia": round(deficit_exercicio),
        "exercicio_min_dia": round(exercicio_min_dia),
        "exercicio_min_semana": round(exercicio_min_semana),
        "exercicio_min_por_sessao": round(exercicio_min_por_sessao),
        "dias_exercicio_semana": dias_exercicio,
        "intensidade": intensidade,
        "perda_semanal": round(perda_semanal, 2),
        "perda_total": round(perda_total, 1),
        "prazo_semanas": prazo_semanas,
        "alertas": alertas,
        "viavel": len(alertas) < 2,
    }


# ============================================================================
# Score heurístico de progressão (Visão Geral)
# ============================================================================

def score_progressao(df) -> dict:
    """
    Calcula score de progressão heurístico do grupo filtrado.

    Multiplicadores de risco baseados em literatura:
      - histórico familiar      × 2,0
      - sedentário (FAF=0)      × 1,5
      - alimentação calórica    × 1,4
      - lanches frequentes      × 1,3
      - alto tempo de tela      × 1,2

    Taxa-base de progressão: 3% ao ano sem intervenção.
    Projeta-se 5 anos.
    """
    n = len(df)
    if n == 0:
        return {"score": 0, "prog_5a": 0, "fatores": [], "n": 0}

    fam_pct  = (df["family_history"] == "yes").mean()
    sed_pct  = (df["FAF"] == 0).mean()
    favc_pct = (df["FAVC"] == "yes").mean()
    caec_pct = df["CAEC"].isin(["Frequently", "Always"]).mean()
    tela_pct = (df["TUE"] == 2).mean()

    fatores = [
        ("Histórico familiar de sobrepeso", fam_pct,  2.0),
        ("Sedentarismo (0 dias de exercício)", sed_pct, 1.5),
        ("Consumo frequente de calóricos",   favc_pct, 1.4),
        ("Lanches frequentes entre refeições", caec_pct, 1.3),
        ("Alto tempo de tela (>5h/dia)",     tela_pct, 1.2),
    ]

    # multiplicador ponderado: cada fator contribui proporcional à sua prevalência no grupo
    mult = 1.0
    for _, pct, peso in fatores:
        mult *= (1 + (peso - 1) * pct)

    taxa_anual = 0.03  # 3% ao ano - taxa-base
    prog_5a = (1 - (1 - taxa_anual * mult) ** 5) * 100
    prog_5a = min(prog_5a, 95.0)  # cap em 95%

    # determinar a próxima classe esperada
    dist = df["Diagnóstico"].value_counts(normalize=True)
    classe_dominante = dist.idxmax() if len(dist) else "—"
    NEXT = {
        "Abaixo do Peso":  "Peso Normal",
        "Peso Normal":     "Sobrepeso I",
        "Sobrepeso I":     "Sobrepeso II",
        "Sobrepeso II":    "Obesidade I",
        "Obesidade I":     "Obesidade II",
        "Obesidade II":    "Obesidade III",
        "Obesidade III":   "Obesidade III",
    }
    proxima_classe = NEXT.get(classe_dominante, "—")

    # nivel de risco
    if prog_5a < 15:
        nivel, cor_risco = "Baixo", "#10B981"
    elif prog_5a < 35:
        nivel, cor_risco = "Moderado", "#FBBF24"
    elif prog_5a < 60:
        nivel, cor_risco = "Alto", "#F97316"
    else:
        nivel, cor_risco = "Muito Alto", "#EF4444"

    # top 3 fatores mais prevalentes
    top3 = sorted(fatores, key=lambda x: x[1], reverse=True)[:3]

    return {
        "score": round(mult, 2),
        "prog_5a": round(prog_5a, 1),
        "fatores": fatores,
        "top3": top3,
        "nivel": nivel,
        "cor_risco": cor_risco,
        "classe_dominante": classe_dominante,
        "proxima_classe": proxima_classe,
        "n": n,
    }


# ============================================================================
# Persistência por CPF — pacientes.json
# ============================================================================

def carregar_pacientes() -> dict:
    """Carrega o banco JSON. Retorna {} se não existir."""
    if not PACIENTES_DB.exists():
        return {}
    try:
        with open(PACIENTES_DB, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def salvar_paciente(cpf: str, entrada: dict[str, Any]) -> None:
    """
    Persiste uma nova entrada de avaliação para o CPF.
    Cada CPF guarda uma lista de avaliações ao longo do tempo.
    """
    cpf_clean = limpar_cpf(cpf)
    if not cpf_clean:
        return

    db = carregar_pacientes()
    if cpf_clean not in db:
        db[cpf_clean] = []
    entrada = {**entrada, "timestamp": datetime.now().isoformat(timespec="seconds")}
    db[cpf_clean].append(entrada)

    with open(PACIENTES_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False, default=str)


def historico_paciente(cpf: str) -> list[dict]:
    """Retorna lista de avaliações do CPF (ordenada cronologicamente)."""
    cpf_clean = limpar_cpf(cpf)
    db = carregar_pacientes()
    return db.get(cpf_clean, [])


def listar_cpfs_cadastrados() -> list[str]:
    """Retorna CPFs cadastrados (já formatados)."""
    return [formatar_cpf_visual(c) for c in carregar_pacientes().keys()]
