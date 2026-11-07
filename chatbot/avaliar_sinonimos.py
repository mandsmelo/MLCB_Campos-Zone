
from pathlib import Path
from time import perf_counter

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

NOME_COLECAO = "campos_zone_faq"

MODELO_EMBEDDING = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

LIMIAR_SIMILARIDADE = 0.35

QUANTIDADE_CANDIDATOS = 3


# ==========================================
# 2. PERGUNTAS DE AVALIACAO
# ==========================================

perguntas_teste = [
    {
        "pergunta": "Quais comandos permitem deslocar o personagem pelo cenário?",
        "id_esperado": "faq01"
    },
    {
        "pergunta": "Como faço o personagem se deslocar mais rapidamente?",
        "id_esperado": "faq02"
    },
    {
        "pergunta": "Qual é o limite de energia que o personagem consegue armazenar?",
        "id_esperado": "faq03"
    },
    {
        "pergunta": "O que devo fazer para a energia do personagem voltar depois de gastar?",
        "id_esperado": "faq04"
    },
    {
        "pergunta": "Se a energia acabar, o personagem ainda consegue disparar uma corrida?",
        "id_esperado": "faq05"
    },
    {
        "pergunta": "Quanto tempo o personagem permanece no ar ao saltar?",
        "id_esperado": "faq08"
    },
    {
        "pergunta": "Qual penalidade o jogador sofre ao ser atingido por um morto-vivo?",
        "id_esperado": "faq10"
    },
    {
        "pergunta": "Como os inimigos calculam a rota para alcançar o sobrevivente?",
        "id_esperado": "faq16"
    },
    {
        "pergunta": "Qual caminho os mortos-vivos percorrem quando estão fazendo ronda?",
        "id_esperado": "faq20"
    },
    {
        "pergunta": "Qual equipamento permite enxergar melhor no escuro e utiliza carga?",
        "id_esperado": "faq24"
    },
    {
        "pergunta": "Onde encontro recursos para abastecer novamente a iluminação portátil?",
        "id_esperado": "faq27"
    },
    {
        "pergunta": "Como atravesso uma entrada para acessar outro cenário do jogo?",
        "id_esperado": "faq30"
    }
]


# ==========================================
# 3. AVALIAR RECUPERACAO
# ==========================================

def avaliar_sinonimos():

    print("Carregando modelo de embeddings...")

    modelo = SentenceTransformer(
        MODELO_EMBEDDING
    )

    print("Conectando ao banco vetorial...")

    cliente = chromadb.PersistentClient(
        path=str(CAMINHO_BANCO)
    )

    colecao = cliente.get_collection(
        name=NOME_COLECAO
    )

    print(f"Documentos na base: {colecao.count()}")
    print(f"Perguntas para avaliacao: {len(perguntas_teste)}")
    print(f"Candidatos por pergunta: {QUANTIDADE_CANDIDATOS}")

    print("-" * 90)

    acertos_primeiro = 0
    acertos_top3 = 0

    similaridades = []
    tempos_busca = []

    for numero, teste in enumerate(
        perguntas_teste,
        start=1
    ):

        pergunta = teste["pergunta"]
        id_esperado = teste["id_esperado"]

        # Gera o embedding da pergunta.

        embedding = modelo.encode(
            [pergunta],
            normalize_embeddings=True
        )

        # Mede somente a busca vetorial.

        inicio_busca = perf_counter()

        resultado = colecao.query(
            query_embeddings=embedding.tolist(),
            n_results=QUANTIDADE_CANDIDATOS,
            include=[
                "metadatas",
                "distances"
            ]
        )

        fim_busca = perf_counter()

        tempo_busca_ms = (
            fim_busca - inicio_busca
        ) * 1000

        tempos_busca.append(
            tempo_busca_ms
        )

        ids_encontrados = resultado["ids"][0]

        distancias = resultado["distances"][0]

        similaridades_candidatos = [
            1.0 - distancia
            for distancia in distancias
        ]

        similaridade_primeiro = (
            similaridades_candidatos[0]
        )

        similaridades.append(
            similaridade_primeiro
        )

        # Verifica se a FAQ correta esta em primeiro.

        acertou_primeiro = (
            ids_encontrados[0] == id_esperado
            and similaridade_primeiro >= LIMIAR_SIMILARIDADE
        )

        # Verifica se a FAQ correta aparece entre os tres.

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
            f"Similaridade: {similaridade_primeiro * 100:.2f}%"
        )

        print("Tres candidatos:")

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
            "FAQ esperada encontrada no Top 3:",
            "SIM" if acertou_top3 else "NAO"
        )

        print(
            f"Tempo da busca: {tempo_busca_ms:.2f} ms"
        )

    # ==========================================
    # 4. RESULTADOS GERAIS
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

    print("\n" + "=" * 90)

    print("RESULTADO GERAL DA AVALIACAO")

    print("=" * 90)

    print(f"Total de perguntas: {total}")

    print(
        f"Acertos em primeiro lugar: {acertos_primeiro}"
    )

    print(
        f"Precisao do primeiro resultado: {taxa_primeiro:.2f}%"
    )

    print(
        f"Perguntas recuperadas no Top 3: {acertos_top3}"
    )

    print(
        f"Recall@3: {taxa_top3:.2f}%"
    )

    print(
        f"Similaridade media do primeiro resultado: "
        f"{media_similaridade:.2f}%"
    )

    print(
        f"Tempo medio de busca vetorial: "
        f"{media_tempo:.2f} ms"
    )

    print(
        f"Maior tempo de busca vetorial: "
        f"{maior_tempo:.2f} ms"
    )

    if maior_tempo < 200:

        print(
            "\nTodas as buscas ficaram abaixo de 200 ms."
        )

    else:

        print(
            "\nPelo menos uma busca ultrapassou 200 ms."
        )


# ==========================================
# 5. EXECUCAO
# ==========================================

if __name__ == "__main__":

    avaliar_sinonimos()
