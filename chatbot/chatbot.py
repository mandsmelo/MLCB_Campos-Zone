
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURACOES
# ==========================================

PASTA_CHATBOT = Path(__file__).resolve().parent

PASTA_BANCO = (
    PASTA_CHATBOT
    / "modelos"
    / "banco_vetorial"
)

NOME_COLECAO = "campos_zone_faq"

MODELO_EMBEDDING = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Limiar inicial, que deverá ser ajustado
# depois dos testes com perguntas e sinônimos.
LIMIAR_SIMILARIDADE = 0.35


# ==========================================
# 2. CARREGAR MODELO
# ==========================================

def carregar_modelo():

    print("Carregando modelo de embeddings...")

    modelo = SentenceTransformer(
        MODELO_EMBEDDING
    )

    print("Modelo carregado!\n")

    return modelo


# ==========================================
# 3. CONECTAR AO CHROMADB
# ==========================================

def conectar_banco():

    if not PASTA_BANCO.exists():
        raise FileNotFoundError(
            "Banco vetorial não encontrado. "
            "Execute primeiro o indexacao.py."
        )

    cliente = chromadb.PersistentClient(
        path=str(PASTA_BANCO)
    )

    try:
        colecao = cliente.get_collection(
            name=NOME_COLECAO
        )

    except Exception as erro:
        raise RuntimeError(
            "Coleção não encontrada. "
            "Execute primeiro o indexacao.py."
        ) from erro

    if colecao.count() == 0:
        raise RuntimeError(
            "A coleção está vazia. "
            "Execute novamente a indexação."
        )

    print("Conexão com ChromaDB realizada!")
    print(f"Documentos disponíveis: {colecao.count()}\n")

    return colecao


# ==========================================
# 4. BUSCAR RESPOSTA
# ==========================================

def buscar_resposta(pergunta, modelo, colecao):

    if not pergunta.strip():
        return None

    # Gera o embedding da pergunta do jogador.
    embedding = modelo.encode(
        [pergunta],
        normalize_embeddings=True
    )

    # Pesquisa o documento mais semelhante.
    resultado = colecao.query(
        query_embeddings=embedding.tolist(),
        n_results=1,
        include=[
            "metadatas",
            "documents",
            "distances"
        ]
    )

    if not resultado["ids"][0]:
        return None

    distancia = resultado["distances"][0][0]

    # Para a distância de cosseno do ChromaDB:
    # similaridade = 1 - distância.
    similaridade = 1.0 - distancia

    metadados = resultado["metadatas"][0][0]

    resposta = {
        "pergunta_encontrada": metadados["pergunta"],
        "resposta": metadados["resposta"],
        "fonte": metadados.get(
            "fonte",
            "base_conhecimento.json"
        ),
        "similaridade": similaridade
    }

    return resposta


# ==========================================
# 5. EXIBIR RESPOSTA
# ==========================================

def exibir_resposta(resultado):

    if resultado is None:
        print("\nNão foi possível realizar a busca.")
        return

    similaridade = resultado["similaridade"]

    print("\n----------------------------------------")

    print("RESULTADO DA BUSCA")

    print("----------------------------------------")

    print(
        "\nPergunta encontrada:",
        resultado["pergunta_encontrada"]
    )

    print(
        "\nSimilaridade:",
        f"{similaridade * 100:.2f}%"
    )

    if similaridade < LIMIAR_SIMILARIDADE:

        print(
            "\nNão encontrei uma resposta "
            "com relevância suficiente."
        )

    else:

        print("\nResposta:")

        print(resultado["resposta"])

        print("\nFonte:")

        print(resultado["fonte"])

    print("----------------------------------------\n")


# ==========================================
# 6. CHAT PELO TERMINAL
# ==========================================

def iniciar_chat():

    print("\n====================================")
    print("       CAMPOS ZONE - CHATBOT")
    print("====================================")

    modelo = carregar_modelo()

    colecao = conectar_banco()

    print("Chatbot pronto para responder!")
    print("Digite 'sair' para encerrar.\n")

    while True:

        pergunta = input("Você: ").strip()

        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("\nChatbot encerrado.")
            break

        if not pergunta:
            print("Digite uma pergunta.\n")
            continue

        resultado = buscar_resposta(
            pergunta,
            modelo,
            colecao
        )

        exibir_resposta(resultado)


# ==========================================
# 7. EXECUCAO PRINCIPAL
# ==========================================

if __name__ == "__main__":

    try:
        iniciar_chat()

    except Exception as erro:

        print("\nERRO AO INICIAR O CHATBOT:")
        print(erro)
