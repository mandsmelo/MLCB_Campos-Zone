
import csv
import random
from pathlib import Path

# Reprodutibilidade: gera os mesmos dados em cada execução
random.seed(42)

# Localização do arquivo CSV
PASTA_PROJETO = Path(__file__).resolve().parent.parent
PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_DADOS.mkdir(parents=True, exist_ok=True)

ARQUIVO_CSV = PASTA_DADOS / "partidas_simuladas.csv"

# Dificuldades e seus tempos-base simulados
dificuldades = {
    "facil": 12,
    "medio": 18,
    "dificil": 25
}

registros = []

# Geração de 600 partidas simuladas
for numero in range(1, 601):

    zumbis_derrotados = random.randint(0, 35)
    distancia_percorrida_m = random.randint(300, 2500)
    dano_recebido = random.randint(0, 100)
    itens_coletados = random.randint(0, 15)
    salas_exploradas = random.randint(2, 12)

    dificuldade = random.choice(list(dificuldades.keys()))
    tempo_base = dificuldades[dificuldade]

    # Fórmula simulada para representar o tempo de conclusão
    variacao = random.gauss(0, 2.5)

    tempo_conclusao_min = (
        tempo_base
        + zumbis_derrotados * 0.18
        + distancia_percorrida_m * 0.006
        + dano_recebido * 0.035
        + itens_coletados * 0.45
        + salas_exploradas * 1.2
        + variacao
    )

    # Evita tempos negativos ou impossíveis
    tempo_conclusao_min = max(5, tempo_conclusao_min)

    registro = {
        "id_partida": numero,
        "zumbis_derrotados": zumbis_derrotados,
        "distancia_percorrida_m": distancia_percorrida_m,
        "dano_recebido": dano_recebido,
        "itens_coletados": itens_coletados,
        "salas_exploradas": salas_exploradas,
        "dificuldade": dificuldade,
        "tempo_conclusao_min": round(tempo_conclusao_min, 2),
        "origem_dados": "simulados"
    }

    registros.append(registro)

# Salva os dados em CSV
with open(ARQUIVO_CSV, "w", newline="", encoding="utf-8-sig") as arquivo:
    colunas = registros[0].keys()

    escritor = csv.DictWriter(
        arquivo,
        fieldnames=colunas
    )

    escritor.writeheader()
    escritor.writerows(registros)

print("=" * 50)
print("GERACAO DE DADOS - CAMPOS ZONE")
print("=" * 50)
print(f"Registros gerados: {len(registros)}")
print(f"Variaveis explicativas: 6")
print(f"Arquivo salvo em: {ARQUIVO_CSV}")
print("Origem: dados totalmente simulados")
print("=" * 50)
