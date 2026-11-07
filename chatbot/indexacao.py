
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURACOES
# ==========================================

PASTA_CHATBOT = Path(__file__).resolve().parent

ARQUIVO_BASE = (
    PASTA_CHATBOT
    / "documentos"
    / "base_conhecimento.json"
)

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


# ==========================================
# 2. CARREGAR BASE DE CONHECIMENTO
# ==========================================

def carregar_documentos():

    if not ARQUIVO_BASE.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {ARQUIVO_BASE}"
        )

    with open(
        ARQUIVO_BASE,
        "r",
        encoding="utf-8"
    ) as arquivo:

        dados = json.load(arquivo)

    # Aceita uma lista ou um objeto
    # contendo a chave "perguntas".

    if isinstance(dados, dict):

        documentos = dados.get("perguntas", [])

    elif isinstance(dados, list):

        documentos = dados

    else:

        raise ValueError(
            "Formato do JSON invalido."
        )

    if not documentos:

        raise ValueError(
            "A base de conhecimento esta vazia."
        )

    ids = []

    for doc in documentos:

        campos_obrigatorios = [
            "id",
            "pergunta",
            "resposta"
        ]

        for campo in campos_obrigatorios:

            if campo not in doc:

                raise ValueError(
                    f"Documento sem o campo '{campo}': {doc}"
                )

        if not isinstance(doc["id"], str):
            raise ValueError("O ID deve ser um texto.")

        if not isinstance(doc["pergunta"], str):
            raise ValueError("A pergunta deve ser um texto.")

        if not isinstance(doc["resposta"], str):
            raise ValueError("A resposta deve ser um texto.")

        if not doc["id"].strip():
            raise ValueError("Existe um ID vazio.")

        if not doc["pergunta"].strip():
            raise ValueError("Existe uma pergunta vazia.")

        if not doc["resposta"].strip():
            raise ValueError("Existe uma resposta vazia.")

        ids.append(doc["id"])

        # A fonte e opcional.
        # Caso nao exista, utiliza o nome do arquivo.

        doc["fonte"] = doc.get(
            "fonte",
            "base_conhecimento.json"
        )

    # Verifica se existem IDs repetidos.

    if len(ids) != len(set(ids)):

        raise ValueError(
            "Existem IDs duplicados na base."
        )

    print(
        f"{len(documentos)} documentos carregados."
    )

    return documentos

# ==========================================
# 3. PREPARAR TEXTOS COM TERMOS ALTERNATIVOS
# ==========================================

# Termos adicionais para melhorar a recuperação semântica.
# Eles são utilizados somente na geração dos embeddings.
# As perguntas e respostas originais permanecem inalteradas.

TERMOS_BUSCA = {

    "faq01": [
        "andar pelo cenário",
        "movimentação básica do personagem",
        "movimentar para frente, trás e lados",
        "comandos direcionais",
        "teclas para caminhar",
        "deslocamento normal sem correr"
    ],

    "faq02": [
        "correr",
        "corrida",
        "sprint",
        "acelerar o personagem",
        "aumentar velocidade de movimento",
        "movimentação rápida",
        "correr pelo cenário",
        "deslocamento em alta velocidade"
    ],

    "faq03": [
        "capacidade máxima de energia",
        "limite de stamina",
        "energia máxima",
        "quantidade máxima de resistência",
        "barra de energia"
    ],

     "faq04": [
        "recuperar energia gasta",
        "regenerar stamina",
        "restaurar resistência",
        "recuperar fôlego",
        "energia se recuperar com o tempo",
        "descansar para recuperar energia",
        "repor a barra de energia",
        "voltar a ter energia"
    ],

    "faq05": [
        "energia esgotada",
        "ficar sem stamina",
        "não conseguir mais correr",
        "limite de corrida",
        "cansaço do personagem",
        "energia insuficiente para correr",
        "consequências de gastar toda a energia"
    ],
    
    "faq06": [
        "rastejar",
        "engatinhar",
        "movimentação agachada",
        "andar abaixado",
        "deslocamento furtivo",
        "movimento silencioso"
    ],

    "faq08": [
        "pular",
        "salto",
        "duração do pulo",
        "tempo no ar",
        "altura do salto",
        "movimento de salto"
    ],

    "faq10": [
        "receber dano",
        "ser atingido",
        "perder vida",
        "sofrer ataque",
        "penalidade ao receber dano",
        "dano causado por inimigos"
    ],

    "faq16": [
        "perseguição dos zumbis",
        "navegação dos inimigos",
        "calcular rota",
        "encontrar caminho até o jogador",
        "seguir o sobrevivente",
        "trajeto dos inimigos",
        "NavigationAgent2D"
    ],

   
    "faq20": [
        "patrulha dos zumbis",
        "ronda dos mortos-vivos",
        "caminho percorrido durante a ronda",
        "rota de patrulhamento",
        "circular pelo cenário sem perseguir",
        "movimentação dos inimigos em patrulha",
        "andar de um lado para outro",
        "percurso dos zumbis quando não detectam o jogador"
    ],


      "faq23": [
        "morte dos zumbis",
        "inimigos derrotados",
        "destruição dos mortos-vivos",
        "quando um zumbi morre",
        "consequências de eliminar um inimigo",
        "estado do inimigo após ser derrotado"
    ],

    "faq24": [
        "lanterna",
        "acender a lanterna",
        "equipamento para iluminar",
        "iluminação portátil",
        "enxergar no escuro",
        "fonte de luz do personagem",
        "utilizar luz em ambientes escuros",
        "equipamento que consome carga"
    ],

      "faq27": [
        "pilhas",
        "baterias para lanterna",
        "encontrar pilhas no cenário",
        "coletar recursos de iluminação",
        "recarregar a lanterna",
        "repor a carga da lanterna",
        "abastecer equipamento de luz",
        "recuperar a bateria",
        "itens usados para recarregar a iluminação"
    ],

    "faq30": [
        "abrir uma porta",
        "atravessar uma entrada",
        "mudar de cenário",
        "trocar de área",
        "transição entre cenas",
        "acessar outra sala"
    ]
}


def preparar_textos(documentos):

    textos = []

    for doc in documentos:

        termos = TERMOS_BUSCA.get(
            doc["id"],
            []
        )

        texto = (
            f"Pergunta: {doc['pergunta']} "
            f"Termos relacionados: {', '.join(termos)} "
            f"Resposta: {doc['resposta']}"
        )

        textos.append(texto)

    return textos


# ==========================================
# 4. CARREGAR MODELO DE EMBEDDINGS
# ==========================================

def carregar_modelo():

    print("\nCarregando modelo de embeddings...")

    modelo = SentenceTransformer(
        MODELO_EMBEDDING
    )

    print("Modelo carregado com sucesso!")

    return modelo


# ==========================================
# 5. GERAR EMBEDDINGS
# ==========================================

def gerar_embeddings(textos, modelo):

    print("\nGerando embeddings...")

    embeddings = modelo.encode(
        textos,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embeddings gerados com sucesso!")

    return embeddings


# ==========================================
# 6. CRIAR COLECAO COM COSINE
# ==========================================

def criar_colecao(cliente):

    # Verifica se a colecao ja existe.
    # Se existir, remove para garantir
    # a configuracao de distancia de cosseno.

    colecoes = cliente.list_collections()

    nomes_colecoes = [
        colecao.name
        if hasattr(colecao, "name")
        else colecao
        for colecao in colecoes
    ]

    if NOME_COLECAO in nomes_colecoes:

        print("\nRemovendo colecao anterior...")

        cliente.delete_collection(
            name=NOME_COLECAO
        )

    # Cria uma nova colecao utilizando
    # distancia de cosseno.

    colecao = cliente.create_collection(
        name=NOME_COLECAO,

        metadata={
            "descricao": "Base de conhecimento do Campos Zone",
            "hnsw:space": "cosine"
        }
    )

    print(
        "Colecao criada com distancia de cosseno."
    )

    return colecao


# ==========================================
# 7. ARMAZENAR NO CHROMADB
# ==========================================

def armazenar_no_banco(
    documentos,
    textos,
    embeddings
):

    PASTA_BANCO.mkdir(
        parents=True,
        exist_ok=True
    )

    cliente = chromadb.PersistentClient(
        path=str(PASTA_BANCO)
    )

    colecao = criar_colecao(cliente)

    # Armazena os IDs, textos, embeddings
    # e metadados de cada documento.

    colecao.add(

        ids=[
            doc["id"]
            for doc in documentos
        ],

        documents=textos,

        embeddings=embeddings.tolist(),

        metadatas=[
            {
                "pergunta": doc["pergunta"],
                "resposta": doc["resposta"],
                "fonte": doc["fonte"]
            }
            for doc in documentos
        ]
    )

    print("\nDocumentos armazenados no ChromaDB!")

    print(
        f"Total de registros: {colecao.count()}"
    )

    return colecao


# ==========================================
# 8. EXECUCAO PRINCIPAL
# ==========================================

def executar_indexacao():

    print("\n====================================")
    print("     CAMPOS ZONE - INDEXACAO RAG")
    print("====================================\n")

    documentos = carregar_documentos()

    textos = preparar_textos(documentos)

    modelo = carregar_modelo()

    embeddings = gerar_embeddings(
        textos,
        modelo
    )

    armazenar_no_banco(
        documentos,
        textos,
        embeddings
    )

    print("\n====================================")
    print("       INDEXACAO FINALIZADA!")
    print("====================================\n")


# ==========================================
# 9. INICIO DO PROGRAMA
# ==========================================

if __name__ == "__main__":

    try:

        executar_indexacao()

    except Exception as erro:

        print("\nERRO DURANTE A INDEXACAO:")
        print(erro)
