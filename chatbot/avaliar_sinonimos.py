
from pathlib import Path
from time import perf_counter
import csv

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURACOES
# ==========================================

PASTA_CHATBOT = Path(__file__).resolve().parent

CAMINHO_BANCO = (
    PASTA_CHATBOT
    / "modelos"
    / "banco_vetorial"
)

PASTA_RESULTADOS = (
    PASTA_CHATBOT.parent
    / "resultados"
)

ARQUIVO_RESULTADOS = (
    PASTA_RESULTADOS
    / "avaliacao_sinonimos.csv"
)

NOME_COLECAO = "campos_zone_faq"

MODELO_EMBEDDING = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

LIMIAR_SIMILARIDADE = 0.35

QUANTIDADE_CANDIDATOS = 3

REPETICOES_BUSCA = 3


# ==========================================
# 2. PERGUNTAS DE AVALIACAO
# ==========================================

# As perguntas são reformulações das FAQs reais.
# Cada pergunta deve preservar o significado da FAQ esperada.

perguntas_teste = [
    {
        "pergunta": "Quais teclas posso usar para andar pelo cenário?",
        "id_esperado": "faq01"
    },
    {
        "pergunta": "Como ativo o modo de corrida do personagem?",
        "id_esperado": "faq02"
    },
    {
        "pergunta": "Até quantos pontos de energia o personagem pode chegar?",
        "id_esperado": "faq03"
    },
    {
        "pergunta": "De que maneira a stamina se regenera depois de ser utilizada?",
        "id_esperado": "faq04"
    },
    {
        "pergunta": "Depois que a energia chega a zero, quando posso voltar a correr?",
        "id_esperado": "faq05"
    },
    {
        "pergunta": "Qual é a duração do salto do personagem?",
        "id_esperado": "faq08"
    },
    {
        "pergunta": "O que ocorre com o jogador quando ele recebe um ataque de zumbi?",
        "id_esperado": "faq10"
    },
    {
        "pergunta": "Qual sistema o zumbi utiliza para encontrar o caminho até o jogador?",
        "id_esperado": "faq16"
    },
    {
        "pergunta": "Em qual sequência de direções os zumbis patrulham o cenário?",
        "id_esperado": "faq20"
    },
    {
        "pergunta": "Qual comando do controle permite acionar a lanterna?",
        "id_esperado": "faq24"
    },
    {
        "pergunta": "Como faço para repor a carga da lanterna utilizando uma pilha?",
        "id_esperado": "faq27"
    },
    {
        "pergunta": "Qual botão do controle devo utilizar para interagir com portas e objetos?",
        "id_esperado": "faq30"
    }
]


# ==========================================
# 3. CARREGAR MODELO E BANCO
# ==========================================

def carregar_recursos():

    print("Carregando modelo de embeddings...")

    modelo = SentenceTransformer(
        MODELO_EMBEDDING
    )

    print("Conectando ao banco vetorial...")

    if not CAMINHO_BANCO.exists():
        raise FileNotFoundError(
            "Banco vetorial não encontrado. "
            "Execute chatbot/indexacao.py primeiro."
        )

    cliente = chromadb.PersistentClient(
        path=str(CAMINHO_BANCO)
    )

    colecao = cliente.get_collection(
        name=NOME_COLECAO
    )

    if colecao.count() == 0:
        raise RuntimeError(
            "A coleção vetorial está vazia."
        )

    return modelo, colecao


# ==========================================
# 4. REALIZAR BUSCA VETORIAL
# ==========================================

def realizar_busca(colecao, embedding):

    inicio = perf_counter()

    resultado = colecao.query(
        query_embeddings=embedding.tolist(),
        n_results=min(
            QUANTIDADE_CANDIDATOS,
            colecao.count()
        ),
        include=[
            "metadatas",
            "distances"
        ]
    )

    fim = perf_counter()

    tempo_ms = (fim - inicio) * 1000

    ids = resultado["ids"][0]

    distancias = resultado["distances"][0]

    similaridades = [
        max(0.0, min(1.0, 1.0 - float(distancia)))
        for distancia in distancias
    ]

    return ids, similaridades, tempo_ms


# ==========================================
# 5. AQUECIMENTO DA BUSCA
# ==========================================

def aquecer_busca(modelo, colecao):

    print("\nPreparando o banco vetorial...")

    embedding = modelo.encode(
        ["Como faço para movimentar o personagem?"],
        normalize_embeddings=True
    )

    # A primeira consulta não entra na medição.
    realizar_busca(
        colecao,
        embedding
    )

    print("Aquecimento concluído!")


# ==========================================
# 6. AVALIAR PERGUNTAS
# ==========================================

def avaliar_sinonimos():

    modelo, colecao = carregar_recursos()

    print(f"\nDocumentos na base: {colecao.count()}")
    print(f"Perguntas avaliadas: {len(perguntas_teste)}")
    print(f"Candidatos por pergunta: {QUANTIDADE_CANDIDATOS}")
    print(f"Repetições por busca: {REPETICOES_BUSCA}")

    aquecer_busca(
        modelo,
        colecao
    )

    acertos_primeiro = 0
    acertos_top3 = 0

    similaridades = []
    tempos_busca = []

    resultados_csv = []

    print("\n" + "=" * 80)
    print("AVALIACAO DE RECUPERACAO SEMANTICA")
    print("=" * 80)

    for numero, teste in enumerate(
        perguntas_teste,
        start=1
    ):

        pergunta = teste["pergunta"]

        id_esperado = teste["id_esperado"]

        # Gera o embedding uma única vez.
        embedding = modelo.encode(
            [pergunta],
            normalize_embeddings=True
        )

        tempos_repeticoes = []

        ids_encontrados = []
        similaridades_candidatos = []

        # Repete somente a busca vetorial.
        for _ in range(REPETICOES_BUSCA):

            ids, sims, tempo = realizar_busca(
                colecao,
                embedding
            )

            tempos_repeticoes.append(tempo)

            ids_encontrados = ids
            similaridades_candidatos = sims

        tempo_medio = (
            sum(tempos_repeticoes)
            / len(tempos_repeticoes)
        )

        tempos_busca.append(tempo_medio)

        similaridade_primeiro = (
            similaridades_candidatos[0]
        )

        similaridades.append(
            similaridade_primeiro
        )

        # Verifica o primeiro resultado.
        acertou_primeiro = (
            ids_encontrados[0] == id_esperado
            and similaridade_primeiro >= LIMIAR_SIMILARIDADE
        )

        # Verifica se a FAQ esperada está no Top 3.
        acertou_top3 = any(
            id_encontrado == id_esperado
            and similaridade >= LIMIAR_SIMILARIDADE

            for id_encontrado, similaridade
            in zip(
                ids_encontrados,
                similaridades_candidatos
            )
        )

        if acertou_primeiro:
            acertos_primeiro += 1

        if acertou_top3:
            acertos_top3 += 1

        status = (
            "ACERTO"
            if acertou_primeiro
            else "ERRO"
        )

        print(f"\nTeste {numero}: {status}")

        print(f"Pergunta: {pergunta}")

        print(f"FAQ esperada: {id_esperado}")

        print(
            f"Primeiro resultado: {ids_encontrados[0]}"
        )

        print(
            f"Similaridade: "
            f"{similaridade_primeiro * 100:.2f}%"
        )

        print("Candidatos:")

        for posicao, (
            id_encontrado,
            similaridade
        ) in enumerate(
            zip(
                ids_encontrados,
                similaridades_candidatos
            ),
            start=1
        ):

            print(
                f"  {posicao}. {id_encontrado} "
                f"({similaridade * 100:.2f}%)"
            )

        print(
            "FAQ esperada no Top 3:",
            "SIM" if acertou_top3 else "NAO"
        )

        print(
            f"Tempo medio da busca: {tempo_medio:.2f} ms"
        )

        # Guarda os dados para o CSV.
        resultados_csv.append({
            "pergunta": pergunta,
            "faq_esperada": id_esperado,
            "primeiro_resultado": ids_encontrados[0],
            "similaridade_primeiro": round(
                similaridade_primeiro * 100, 2
            ),
            "acerto_primeiro": acertou_primeiro,
            "acerto_top3": acertou_top3,
            "tempo_medio_ms": round(tempo_medio, 2)
        })

    # ==========================================
    # 7. RESULTADOS GERAIS
    # ==========================================

    total = len(perguntas_teste)

    taxa_primeiro = (
        acertos_primeiro / total
    ) * 100

    taxa_top3 = (
        acertos_top3 / total
    ) * 100

    media_similaridade = (
        sum(similaridades) / total
    ) * 100

    media_tempo = (
        sum(tempos_busca) / total
    )

    maior_tempo = max(tempos_busca)

    print("\n" + "=" * 80)
    print("RESULTADO GERAL DA AVALIACAO")
    print("=" * 80)

    print(f"Total de perguntas: {total}")

    print(
        f"Acertos em primeiro lugar: "
        f"{acertos_primeiro}"
    )

    print(
        f"Precisao em primeiro lugar: "
        f"{taxa_primeiro:.2f}%"
    )

    print(
        f"Perguntas recuperadas no Top 3: "
        f"{acertos_top3}"
    )

    print(
        f"Recall@3: {taxa_top3:.2f}%"
    )

    print(
        f"Similaridade media do primeiro resultado: "
        f"{media_similaridade:.2f}%"
    )

    print(
        f"Tempo medio da busca vetorial: "
        f"{media_tempo:.2f} ms"
    )

    print(
        f"Maior tempo medio por pergunta: "
        f"{maior_tempo:.2f} ms"
    )

    if maior_tempo < 200:

        print(
            "\nTodas as perguntas ficaram abaixo de 200 ms."
        )

    else:

        print(
            "\nPelo menos uma pergunta apresentou "
            "tempo medio acima de 200 ms."
        )

    # ==========================================
    # 8. SALVAR RELATORIO CSV
    # ==========================================

    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        ARQUIVO_RESULTADOS,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo:

        campos = [
            "pergunta",
            "faq_esperada",
            "primeiro_resultado",
            "similaridade_primeiro",
            "acerto_primeiro",
            "acerto_top3",
            "tempo_medio_ms"
        ]

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        escritor.writeheader()

        escritor.writerows(
            resultados_csv
        )

    print(
        f"\nRelatorio CSV salvo em:\n{ARQUIVO_RESULTADOS}"
    )


# ==========================================
# 9. EXECUCAO
# ==========================================

if __name__ == "__main__":

    try:
        avaliar_sinonimos()

    except Exception as erro:

        print("\nERRO DURANTE A AVALIACAO:")
        print(erro)
