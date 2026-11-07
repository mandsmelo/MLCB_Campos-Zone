import json
import urllib.request
import urllib.error

from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import chromadb
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURACOES
# ==========================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

# Chatbot
PASTA_CHATBOT = PASTA_PROJETO / "chatbot"
PASTA_BANCO = (
    PASTA_CHATBOT / "modelos" / "banco_vetorial"
)

NOME_COLECAO = "campos_zone_faq"

MODELO_EMBEDDING = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

LIMIAR_SIMILARIDADE = 0.35

# Modelo de linguagem - Ollama
URL_OLLAMA = "http://localhost:11434/api/chat"
MODELO_OLLAMA = "qwen2.5:1.5b"

# Regressao
CAMINHO_MODELO_REGRESSAO = (
    PASTA_PROJETO
    / "notebooks"
    / "regressao"
    / "modelos"
    / "melhor_modelo.joblib"
)

ERRO_MEDIO_MAE = 1.8632


# ==========================================
# 2. VARIAVEIS DO SERVICO
# ==========================================

modelo = None
colecao = None
modelo_regressao = None


# ==========================================
# 3. INICIALIZACAO DA API
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global modelo, colecao, modelo_regressao

    print("\n====================================")
    print("       CAMPOS ZONE - API ML")
    print("====================================")

    # ----------------------------------
    # CARREGAR CHATBOT
    # ----------------------------------

    if not PASTA_BANCO.exists():
        raise RuntimeError(
            "Banco vetorial não encontrado. "
            "Execute python chatbot/indexacao.py"
        )

    print("\nCarregando modelo de embeddings...")

    modelo = SentenceTransformer(MODELO_EMBEDDING)

    print("Modelo de embeddings carregado!")

    print("\nConectando ao ChromaDB...")

    cliente = chromadb.PersistentClient(
        path=str(PASTA_BANCO)
    )

    try:
        colecao = cliente.get_collection(
            name=NOME_COLECAO
        )
    except Exception as erro:
        raise RuntimeError(
            "Coleção do chatbot não encontrada."
        ) from erro

    quantidade = colecao.count()

    if quantidade == 0:
        raise RuntimeError(
            "A coleção do chatbot está vazia."
        )

    print(f"Documentos disponíveis: {quantidade}")

    # ----------------------------------
    # CARREGAR MODELO DE REGRESSAO
    # ----------------------------------

    print("\nCarregando modelo de regressão...")

    if CAMINHO_MODELO_REGRESSAO.exists():

        try:
            modelo_regressao = joblib.load(
                CAMINHO_MODELO_REGRESSAO
            )

            print("Modelo de regressão carregado!")

        except Exception as erro:
            modelo_regressao = None

            print(
                "AVISO: Não foi possível carregar "
                f"o modelo de regressão: {erro}"
            )

    else:
        modelo_regressao = None

        print(
            "AVISO: Modelo de regressão não encontrado. "
            "A rota de previsão ficará indisponível."
        )

    print("\nAPI pronta!")
    print("====================================\n")

    yield

    modelo = None
    colecao = None
    modelo_regressao = None

    print("\nAPI encerrada.")


# ==========================================
# 4. CRIAR APLICACAO FASTAPI
# ==========================================

app = FastAPI(
    title="Campos Zone - API de Machine Learning",
    description=(
        "API integrada com chatbot semântico "
        "e previsão de tempo de conclusão de missões."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ==========================================
# 5. MODELOS DAS REQUISICOES
# ==========================================

class PerguntaRequest(BaseModel):

    pergunta: str = Field(
        min_length=1,
        description="Pergunta enviada pelo jogador."
    )


class PrevisaoRequest(BaseModel):

    zumbis_derrotados: int = Field(
        ge=0,
        description="Quantidade de zumbis derrotados."
    )

    distancia_percorrida_m: float = Field(
        ge=0,
        description="Distância percorrida em metros."
    )

    dano_recebido: float = Field(
        ge=0,
        le=100,
        description="Quantidade de dano recebido."
    )

    itens_coletados: int = Field(
        ge=0,
        description="Quantidade de itens coletados."
    )

    salas_exploradas: int = Field(
        ge=0,
        description="Quantidade de salas exploradas."
    )

    dificuldade: str = Field(
        description="Dificuldade: facil, medio ou dificil."
    )


# ==========================================
# 6. ROTA INICIAL
# ==========================================

@app.get("/")
def inicio():

    return {
        "projeto": "Campos Zone",
        "servico": "API de Machine Learning",
        "status": "online",
        "servicos": {
            "chatbot": "/faq/search",
            "regressao": "/predict/value"
        },
        "documentacao": "/docs"
    }


# ==========================================
# 7. VERIFICAR STATUS
# ==========================================

@app.get("/health")
def verificar_status():

    if modelo is None or colecao is None:

        raise HTTPException(
            status_code=503,
            detail="Serviço do chatbot indisponível."
        )

    return {
        "status": "online",
        "chatbot": {
            "status": "online",
            "documentos_indexados": colecao.count()
        },
        "regressao": {
            "status": (
                "online"
                if modelo_regressao is not None
                else "indisponivel"
            )
        }
    }

# ==========================================
# GERACAO DE RESPOSTAS COM O OLLAMA
# ==========================================

def gerar_resposta_ollama(pergunta, contexto):

    prompt_sistema = """
    Você é o assistente oficial do jogo Campos Zone.

    Sua função é ajudar os jogadores com dúvidas
    sobre missões, comandos, tarefas e funcionalidades.

    Regras:
    1. Responda sempre em português brasileiro.
    2. Utilize somente as informações fornecidas no contexto.
    3. Não invente comandos, itens, missões ou mecânicas.
    4. Se o contexto não tiver informação suficiente,
       informe que não encontrou essa informação na
       base de conhecimento.
    5. Seja claro, objetivo e amigável.
    """

    dados = {
        "model": MODELO_OLLAMA,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": prompt_sistema
            },
            {
                "role": "user",
                "content": (
                    f"Contexto recuperado da base de conhecimento:\n"
                    f"{contexto}\n\n"
                    f"Pergunta do jogador:\n{pergunta}\n\n"
                    "Responda utilizando o contexto fornecido."
                )
            }
        ],
        "options": {
            "temperature": 0.2
        }
    }

    requisicao = urllib.request.Request(
        URL_OLLAMA,
        data=json.dumps(dados).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            requisicao,
            timeout=240
        ) as resposta:

            resultado = json.loads(
                resposta.read().decode("utf-8")
            )

        return resultado["message"]["content"].strip()

    except urllib.error.URLError as erro:

        raise HTTPException(
            status_code=503,
            detail=(
                "Não foi possível conectar ao Ollama. "
                "Verifique se o serviço está em execução."
            )
        ) from erro

    except (KeyError, ValueError) as erro:

        raise HTTPException(
            status_code=500,
            detail="Resposta inválida recebida do Ollama."
        ) from erro
        
# ==========================================
# 8. ENDPOINT DE BUSCA SEMANTICA - PROPOSTA 2
# ==========================================

# ==========================================
# 8. ENDPOINT RAG - PROPOSTA 2
# ==========================================

@app.post("/faq/search")
def buscar_faq(dados: PerguntaRequest):

    if modelo is None or colecao is None:

        raise HTTPException(
            status_code=503,
            detail="Serviço do chatbot indisponível."
        )

    pergunta = dados.pergunta.strip()

    if not pergunta:

        raise HTTPException(
            status_code=422,
            detail="A pergunta não pode estar vazia."
        )

    inicio_total = perf_counter()

    try:

        # 1. GERAR EMBEDDING DA PERGUNTA

        inicio_embedding = perf_counter()

        embedding = modelo.encode(
            [pergunta],
            normalize_embeddings=True
        )

        fim_embedding = perf_counter()

        tempo_embedding_ms = (
            fim_embedding - inicio_embedding
        ) * 1000

        # 2. BUSCAR DOCUMENTOS NO CHROMADB

        inicio_busca = perf_counter()

        quantidade_busca = min(3, colecao.count())

        resultado = colecao.query(
            query_embeddings=embedding.tolist(),
            n_results=quantidade_busca,
            include=["metadatas", "distances"]
        )

        fim_busca = perf_counter()

        tempo_busca_ms = (
            fim_busca - inicio_busca
        ) * 1000

        documentos = []

        for metadados, distancia in zip(
            resultado["metadatas"][0],
            resultado["distances"][0]
        ):

            similaridade = 1.0 - distancia

            if similaridade >= LIMIAR_SIMILARIDADE:

                documentos.append({
                    "pergunta": metadados["pergunta"],
                    "resposta": metadados["resposta"],
                    "fonte": metadados.get(
                        "fonte",
                        "base_conhecimento.json"
                    ),
                    "similaridade": similaridade
                })

        # 3. VERIFICAR SE EXISTE CONTEXTO RELEVANTE

        fim_recuperacao = perf_counter()

        if not documentos:

            return {
                "sucesso": True,
                "encontrou_resposta": False,
                "resposta": None,
                "mensagem": (
                    "Não encontrei informações suficientes "
                    "na base de conhecimento para responder."
                ),
                "tempo_embedding_ms": round(
                    tempo_embedding_ms, 2
                ),
                "tempo_busca_ms": round(
                    tempo_busca_ms, 2
                ),
                "tempo_total_ms": round(
                    (fim_recuperacao - inicio_total) * 1000,
                    2
                )
            }

        # 4. PREPARAR CONTEXTO PARA O MODELO DE LINGUAGEM

        contexto = "\n\n".join(
            [
                (
                    f"Pergunta cadastrada: {doc['pergunta']}\n"
                    f"Resposta cadastrada: {doc['resposta']}"
                )
                for doc in documentos
            ]
        )

        # 5. GERAR RESPOSTA COM O OLLAMA

        inicio_geracao = perf_counter()

        resposta_gerada = gerar_resposta_ollama(
            pergunta,
            contexto
        )

        fim_geracao = perf_counter()

        tempo_geracao_ms = (
            fim_geracao - inicio_geracao
        ) * 1000

        # 6. ORGANIZAR RESULTADO

        melhor_documento = documentos[0]

        fim_total = perf_counter()

        return {
            "sucesso": True,
            "encontrou_resposta": True,

            "pergunta_recebida": pergunta,

            "resposta": resposta_gerada,

            "modelo_geracao": MODELO_OLLAMA,

            "pergunta_encontrada": (
                melhor_documento["pergunta"]
            ),

            "fontes": list(set(
                doc["fonte"]
                for doc in documentos
            )),

            "similaridade": round(
                melhor_documento["similaridade"], 4
            ),

            "similaridade_percentual": round(
                melhor_documento["similaridade"] * 100, 2
            ),

            "documentos_recuperados": len(documentos),

            "tempo_embedding_ms": round(
                tempo_embedding_ms, 2
            ),

            "tempo_busca_ms": round(
                tempo_busca_ms, 2
            ),

            "tempo_geracao_ms": round(
                tempo_geracao_ms, 2
            ),

            "tempo_total_ms": round(
                (fim_total - inicio_total) * 1000, 2
            )
        }

    except HTTPException:
        raise

    except Exception as erro:

        raise HTTPException(
            status_code=500,
            detail=f"Erro durante a busca RAG: {str(erro)}"
        ) from erro

# ==========================================
# 9. ENDPOINT DE PREVISAO - PROPOSTA 4
# ==========================================

@app.post("/predict/value")
def prever_tempo(dados: PrevisaoRequest):

    if modelo_regressao is None:

        raise HTTPException(
            status_code=503,
            detail="Modelo de regressão indisponível."
        )

    # Validar dificuldade
    dificuldade = dados.dificuldade.strip().lower()

    if dificuldade not in ["facil", "medio", "dificil"]:

        raise HTTPException(
            status_code=422,
            detail=(
                "Dificuldade inválida. "
                "Utilize: facil, medio ou dificil."
            )
        )

    inicio_previsao = perf_counter()

    try:

        # Organizar os dados no mesmo formato do treinamento
        entrada = pd.DataFrame([{
            "zumbis_derrotados": dados.zumbis_derrotados,
            "distancia_percorrida_m": dados.distancia_percorrida_m,
            "dano_recebido": dados.dano_recebido,
            "itens_coletados": dados.itens_coletados,
            "salas_exploradas": dados.salas_exploradas,
            "dificuldade": dificuldade
        }])

        # Executar previsão
        previsao = modelo_regressao.predict(entrada)[0]

        fim_previsao = perf_counter()

        tempo_processamento_ms = (
            fim_previsao - inicio_previsao
        ) * 1000

        tempo_estimado = max(0, float(previsao))

        return {
            "sucesso": True,
            "projeto": "Campos Zone",
            "modelo_utilizado": "Regressão Linear",
            "previsao": {
                "tempo_estimado_minutos": round(
                    tempo_estimado, 2
                ),
                "erro_medio_referencia_minutos": ERRO_MEDIO_MAE,
                "tempo_processamento_ms": round(
                    tempo_processamento_ms, 2
                )
            },
            "dados_informados": entrada.iloc[0].to_dict(),
            "origem_dados_treinamento": "simulados",
            "observacao": (
                "Previsão baseada em dados simulados. "
                "O erro médio de referência é o MAE "
                "obtido no conjunto de teste e não "
                "representa uma garantia para cada previsão."
            )
        }

    except Exception as erro:

        raise HTTPException(
            status_code=500,
            detail=f"Erro durante a previsão: {str(erro)}"
        ) from erro
