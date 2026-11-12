"""Funções utilitárias para previsão de tempo restante."""
from pathlib import Path
import joblib
import pandas as pd

FEATURES = ["distancia_percorrida_unidades", "dano_recebido", "itens_coletados", "salas_exploradas", "tempo_decorrido_min"]
CAMINHO_MODELO = Path(__file__).resolve().parent / "modelos" / "melhor_modelo.joblib"

def carregar_modelo():
    pacote = joblib.load(CAMINHO_MODELO)
    return pacote["modelo"] if isinstance(pacote, dict) else pacote

def prever_tempo_restante(valores: dict) -> float:
    modelo = carregar_modelo()
    entrada = pd.DataFrame([{chave: valores[chave] for chave in FEATURES}], columns=FEATURES)
    return max(0.0, float(modelo.predict(entrada)[0]))
