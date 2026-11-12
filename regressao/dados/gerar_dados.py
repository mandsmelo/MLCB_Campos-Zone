
"""Gera 600 snapshots sintéticos de missões com duração média de 30 minutos."""

import csv
import random
from pathlib import Path

SEED = 42
QUANTIDADE = 600
DURACAO_MEDIA = 30

PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_DADOS.mkdir(parents=True, exist_ok=True)

ARQUIVO_CSV = PASTA_DADOS / "partidas_simuladas.csv"

random.seed(SEED)
registros = []

for i in range(1, QUANTIDADE + 1):

    # Duração total estimada da missão, com variações naturais.
    tempo_total = random.gauss(DURACAO_MEDIA, 3.5)
    tempo_total = max(20, min(40, tempo_total))

    # Estado parcial observado durante a missão.
    tempo_decorrido = round(
        random.uniform(1.0, min(24.0, tempo_total - 1)),
        2
    )

    distancia = round(random.uniform(40, 1800), 2)
    dano = random.randint(0, 3)
    itens = random.randint(0, 12)
    salas = random.randint(0, 10)

    restante = max(0, tempo_total - tempo_decorrido)

    registros.append({
        "id_snapshot": i,
        "distancia_percorrida_unidades": distancia,
        "dano_recebido": dano,
        "itens_coletados": itens,
        "salas_exploradas": salas,
        "tempo_decorrido_min": tempo_decorrido,
        "tempo_restante_min": round(restante, 2),
        "origem_dados": "simulados"
    })

with ARQUIVO_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=registros[0].keys())
    writer.writeheader()
    writer.writerows(registros)

media_duracao = sum(
    registro["tempo_restante_min"] + registro["tempo_decorrido_min"]
    for registro in registros
) / QUANTIDADE

print(f"{QUANTIDADE} snapshots simulados gerados.")
print(f"Duração média simulada: {media_duracao:.2f} minutos.")
print(f"Arquivo salvo em: {ARQUIVO_CSV}")
