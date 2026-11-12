"""Treina e compara regressão linear e Random Forest com dados sintéticos."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "dados" / "partidas_simuladas.csv"
MODELOS = ROOT / "modelos"; MODELOS.mkdir(parents=True, exist_ok=True)
RESULTADOS = ROOT.parent / "resultados"; RESULTADOS.mkdir(parents=True, exist_ok=True)
FEATURES = ["distancia_percorrida_unidades", "dano_recebido", "itens_coletados", "salas_exploradas", "tempo_decorrido_min"]
TARGET = "tempo_restante_min"
df = pd.read_csv(CSV)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelos = {
    "Regressao Linear": Pipeline([("escala", StandardScaler()), ("modelo", LinearRegression())]),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
}
linhas=[]; treinados={}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred=modelo.predict(X_test)
    mse=mean_squared_error(y_test,pred)
    metricas={"modelo":nome,"MAE":mean_absolute_error(y_test,pred),"MSE":mse,"RMSE":mse**0.5,"R2":r2_score(y_test,pred)}
    linhas.append(metricas); treinados[nome]=modelo
    print(f"{nome}: MAE={metricas['MAE']:.4f} | RMSE={metricas['RMSE']:.4f} | R²={metricas['R2']:.4f}")
comparacao=pd.DataFrame(linhas).sort_values("RMSE")
comparacao.to_csv(RESULTADOS/"comparacao_modelos.csv",index=False,encoding="utf-8-sig")
melhor_nome=comparacao.iloc[0]["modelo"]
melhor=treinados[melhor_nome]
joblib.dump({"modelo":melhor,"features":FEATURES,"target":TARGET,"nome_modelo":melhor_nome,"metricas":comparacao.iloc[0].to_dict()},MODELOS/"melhor_modelo.joblib")
print(f"\nMelhor modelo: {melhor_nome}")
print(f"Modelo salvo em: {MODELOS/'melhor_modelo.joblib'}")
