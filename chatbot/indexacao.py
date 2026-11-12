
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

        doc["fonte"] = doc.get(
            "fonte",
            "base_conhecimento.json"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Existem IDs duplicados na base."
        )

    print(
        f"{len(documentos)} documentos carregados."
    )

    return documentos


# ==========================================
# 3. TERMOS ALTERNATIVOS DE BUSCA
# ==========================================

# Estes termos ajudam o chatbot a encontrar
# respostas mesmo quando o jogador utiliza
# palavras diferentes das perguntas cadastradas.
#
# Eles sao usados somente na geracao dos embeddings.
# As respostas exibidas continuam sendo as do JSON.

TERMOS_BUSCA = {

    # Movimentacao
    "faq01": [
        "teclas WASD",
        "W A S D",
        "andar para cima baixo esquerda direita",
        "movimentar com teclado",
        "controles de movimentacao"
    ],

    # Correr com CTRL
    "faq02": [
        "ctrl",
        "control",
        "tecla ctrl",
        "segurar ctrl para correr",
        "correr com control",
        "como correr",
        "botao de corrida",
        "corrida do personagem",
        "sprint",
        "gastar energia correndo",
        "barra de energia durante a corrida"
    ],

    # Barra de energia
    "faq03": [
        "capacidade maxima de energia",
        "energia maxima",
        "quantidade maxima de resistencia",
        "barra de energia",
        "limite de stamina",
        "quanto de energia o personagem possui",
        "energia total do jogador"
    ],

    # Recuperacao de energia
    "faq04": [
        "recuperar energia gasta",
        "regenerar stamina",
        "restaurar resistencia",
        "recuperar folego",
        "energia se recuperar com o tempo",
        "descansar para recuperar energia",
        "repor a barra de energia",
        "voltar a ter energia",
        "como recarregar energia"
    ],

    # Energia esgotada
    "faq05": [
        "energia esgotada",
        "ficar sem stamina",
        "nao conseguir mais correr",
        "limite de corrida",
        "cansaco do personagem",
        "energia insuficiente para correr",
        "consequencias de gastar toda a energia",
        "barra de energia vazia"
    ],

    # Rastejar com SHIFT
    "faq06": [
        "shift",
        "tecla shift",
        "segurar shift para rastejar",
        "como rastejar",
        "botao para rastejar",
        "agachar",
        "andar agachado",
        "movimento furtivo",
        "se esconder dos zumbis",
        "passar despercebido"
    ],

    # Velocidades
    "faq07": [
        "velocidade andando",
        "velocidade correndo",
        "velocidade rastejando",
        "rapidez do personagem",
        "velocidade de movimentacao",
        "diferenca entre andar correr e rastejar"
    ],

    # Pulo
    "faq08": [
        "barra de espaco",
        "space",
        "tecla espaco",
        "salto",
        "pular"
    ],

    # Dano e vidas
    "faq10": [
        "receber dano",
        "ser atingido",
        "perder vida",
        "sofrer ataque",
        "penalidade ao receber dano",
        "dano causado por inimigos"
    ],

    # Perseguicao dos zumbis
    "faq12": [
        "como os zumbis enxergam o jogador",
        "deteccao do jogador",
        "percepcao dos inimigos",
        "distancia para detectar o jogador"
    ],

    "faq13": [
        "correr chama a atencao dos zumbis",
        "zumbis detectam quem corre",
        "barulho da corrida",
        "ser detectado correndo"
    ],

    "faq14": [
        "rastejar evita zumbis",
        "esconder dos zumbis",
        "evitar deteccao",
        "movimento furtivo",
        "passar despercebido pelos inimigos"
    ],

    "faq15": [
        "alcance de deteccao",
        "distancia de percepcao dos zumbis",
        "distancia para ser encontrado",
        "raio de deteccao dos inimigos"
    ],

    "faq16": [
        "perseguicao dos zumbis",
        "navegacao dos inimigos",
        "calcular rota",
        "encontrar caminho ate o jogador",
        "seguir o sobrevivente",
        "trajeto dos inimigos",
        "NavigationAgent2D"
    ],

    "faq17": [
        "ataque dos zumbis",
        "distancia de ataque",
        "quando o inimigo ataca",
        "zumbi perto do jogador"
    ],

    "faq18": [
        "intervalo entre ataques",
        "frequencia de ataque",
        "tempo entre golpes",
        "quantas vezes o zumbi ataca"
    ],

    "faq19": [
        "patrulha dos zumbis",
        "inimigos andando pelo cenario",
        "zumbis sem detectar o jogador"
    ],

    "faq20": [
        "patrulha dos zumbis",
        "ronda dos mortos-vivos",
        "caminho percorrido durante a ronda",
        "rota de patrulhamento",
        "circular pelo cenario sem perseguir",
        "movimentacao dos inimigos em patrulha",
        "andar de um lado para outro"
    ],

    "faq21": [
        "zumbi desiste da perseguicao",
        "perder zumbi de vista",
        "distancia para o inimigo parar de perseguir",
        "interromper perseguicao"
    ],

    "faq22": [
        "zumbi preso",
        "inimigo preso em obstaculo",
        "destravar zumbi",
        "zumbi nao consegue andar"
    ],

    "faq23": [
        "matar zumbi",
        "derrotar zumbis",
        "atacar inimigos",
        "combater zumbi",
        "nao consigo matar zumbi"
    ],

    # Lanterna
    "faq24": [
        "ligar lanterna no controle",
        "eixo 4",
        "botao do controle",
        "tecla da lanterna"
    ],

    "faq25": [
        "bateria inicial da lanterna",
        "carga inicial",
        "capacidade da bateria",
        "bateria maxima"
    ],

    "faq26": [
        "duracao da bateria",
        "quanto tempo a lanterna funciona",
        "consumo de bateria",
        "bateria acabando"
    ],

    "faq27": [
        "pilhas",
        "baterias para lanterna",
        "encontrar pilhas no cenario",
        "coletar recursos de iluminacao",
        "recarregar a lanterna",
        "repor a carga da lanterna",
        "abastecer equipamento de luz",
        "recuperar a bateria"
    ],

    "faq28": [
        "lanterna sem bateria",
        "bateria zerada",
        "lanterna piscando",
        "clarões da lanterna",
        "o que acontece quando acaba a carga"
    ],

    "faq29": [
        "nao consigo pegar pilha",
        "coletar bateria",
        "pilha nao desaparece",
        "bateria cheia",
        "problema ao coletar pilha"
    ],

    # Interacao
    "faq30": [
        "interagir com objetos",
        "botao 1 do controle",
        "controle gamepad",
        "tecla de interacao",
        "abrir porta",
        "interagir com portas",
        "como usar objetos"
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
