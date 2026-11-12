"""API Campos Zone: recuperação semântica e regressão de tempo restante."""
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
import joblib
import pandas as pd
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PASTA_BANCO = ROOT / "chatbot" / "modelos" / "banco_vetorial"
COLECAO = "campos_zone_faq"
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIMIAR_SIMILARIDADE = 0.35
CAMINHO_MODELO = ROOT / "regressao" / "modelos" / "melhor_modelo.joblib"
FEATURES = ["distancia_percorrida_unidades", "dano_recebido", "itens_coletados", "salas_exploradas", "tempo_decorrido_min"]
modelo_embedding = None
colecao = None
modelo_regressao = None
nome_modelo_regressao = None
mae_modelo = None
chatbot_erro = None
regressao_erro = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global modelo_embedding, colecao, modelo_regressao, nome_modelo_regressao, mae_modelo, chatbot_erro, regressao_erro
    print("\n=== CAMPOS ZONE - API ML ===")
    # Falha do chatbot não impede a API de regressão de iniciar.
    try:
        if not PASTA_BANCO.exists():
            raise FileNotFoundError("Banco vetorial ausente; execute python chatbot/indexacao.py")
        modelo_embedding = SentenceTransformer(MODELO_EMBEDDING)
        cliente = chromadb.PersistentClient(path=str(PASTA_BANCO))
        colecao = cliente.get_collection(name=COLECAO)
        if colecao.count() == 0:
            raise RuntimeError("A coleção de perguntas está vazia.")
        chatbot_erro = None
        print(f"Chatbot carregado: {colecao.count()} documentos")
    except Exception as exc:
        modelo_embedding = None; colecao = None; chatbot_erro = str(exc)
        print(f"AVISO chatbot indisponível: {exc}")
    # Falha da regressão também não impede o chatbot de iniciar.
    try:
        if not CAMINHO_MODELO.exists():
            raise FileNotFoundError("Modelo ausente; execute python regressao/dados/gerar_dados.py e python regressao/treinamento.py")
        pacote = joblib.load(CAMINHO_MODELO)
        if isinstance(pacote, dict):
            modelo_regressao = pacote["modelo"]
            nome_modelo_regressao = pacote.get("nome_modelo", "Modelo treinado")
            mae_modelo = pacote.get("metricas", {}).get("MAE")
        else:
            modelo_regressao = pacote; nome_modelo_regressao = type(pacote).__name__
        regressao_erro = None
        print(f"Regressão carregada: {nome_modelo_regressao}")
    except Exception as exc:
        modelo_regressao = None; regressao_erro = str(exc)
        print(f"AVISO regressão indisponível: {exc}")
    print("API inicializada.\n")
    yield
    modelo_embedding = colecao = modelo_regressao = None

app = FastAPI(title="Campos Zone - API de Machine Learning", description="Chatbot por recuperação semântica e previsão do tempo restante da missão.", version="3.0.0", lifespan=lifespan)

class PerguntaRequest(BaseModel):
    pergunta: str = Field(min_length=1, description="Pergunta do jogador")

class PrevisaoRequest(BaseModel):
    distancia_percorrida_unidades: float = Field(ge=0, description="Distância em unidades do mundo Godot; não equivale necessariamente a metros")
    dano_recebido: int = Field(ge=0, description="Quantidade acumulada de impactos/vidas perdidas")
    itens_coletados: int = Field(ge=0)
    salas_exploradas: int = Field(ge=0)
    tempo_decorrido_min: float = Field(ge=0, description="Tempo de missão já decorrido, em minutos")

@app.get("/")
def inicio():
    return {"projeto":"Campos Zone","status":"online","servicos":{"chatbot":"/faq/search","regressao":"/predict/value"},"documentacao":"/docs"}

@app.get("/health")
def health():
    return {"status":"online","chatbot":{"status":"online" if colecao is not None and modelo_embedding is not None else "indisponivel","documentos_indexados":colecao.count() if colecao is not None else 0,"erro":chatbot_erro},"regressao":{"status":"online" if modelo_regressao is not None else "indisponivel","modelo":nome_modelo_regressao,"erro":regressao_erro}}

@app.post("/faq/search")
def buscar_faq(dados: PerguntaRequest):
    if modelo_embedding is None or colecao is None:
        raise HTTPException(status_code=503, detail=f"Chatbot indisponível: {chatbot_erro or 'serviço não carregado'}")
    pergunta=dados.pergunta.strip()
    if not pergunta:
        raise HTTPException(status_code=422, detail="A pergunta não pode estar vazia.")
    inicio=perf_counter()
    try:
        t=perf_counter(); embedding=modelo_embedding.encode([pergunta], normalize_embeddings=True); tempo_embedding=(perf_counter()-t)*1000
        t=perf_counter(); n=min(3,colecao.count()); resultado=colecao.query(query_embeddings=embedding.tolist(),n_results=n,include=["metadatas","distances"]); tempo_busca=(perf_counter()-t)*1000
        candidatos=[]
        for meta,dist in zip(resultado["metadatas"][0],resultado["distances"][0]):
            sim=max(0.0,min(1.0,1.0-float(dist)))
            candidatos.append({"pergunta":meta.get("pergunta",""),"resposta":meta.get("resposta",""),"fonte":meta.get("fonte","base_conhecimento.json"),"similaridade":sim})
        melhor=max(candidatos,key=lambda x:x["similaridade"]) if candidatos else None
        relevante=melhor is not None and melhor["similaridade"] >= LIMIAR_SIMILARIDADE
        total=(perf_counter()-inicio)*1000
        if not relevante:
            return {"sucesso":True,"encontrou_resposta":False,"pergunta_recebida":pergunta,"resposta":"Desculpe, não encontrei uma resposta para essa pergunta. Posso ajudar com informações sobre os comandos, movimentação, zumbis, lanterna e interações do Campos Zone!","similaridade":round(melhor["similaridade"],4) if melhor else 0.0,"similaridade_percentual":round(melhor["similaridade"]*100,2) if melhor else 0.0,"fontes":[],"tempo_embedding_ms":round(tempo_embedding,2),"tempo_busca_ms":round(tempo_busca,2),"tempo_total_ms":round(total,2),"modo_resposta":"fallback_semantico"}
        return {"sucesso":True,"encontrou_resposta":True,"pergunta_recebida":pergunta,"resposta":melhor["resposta"],"pergunta_encontrada":melhor["pergunta"],"fontes":[melhor["fonte"]],"similaridade":round(melhor["similaridade"],4),"similaridade_percentual":round(melhor["similaridade"]*100,2),"documentos_recuperados":len(candidatos),"tempo_embedding_ms":round(tempo_embedding,2),"tempo_busca_ms":round(tempo_busca,2),"tempo_total_ms":round(total,2),"modo_resposta":"recuperacao_semantica"}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(status_code=500,detail=f"Erro na busca semântica: {exc}") from exc

@app.post("/predict/value")
def prever_tempo(dados: PrevisaoRequest):
    if modelo_regressao is None:
        raise HTTPException(status_code=503,detail=f"Modelo de regressão indisponível: {regressao_erro or 'modelo não carregado'}")
    inicio=perf_counter()
    try:
        entrada=pd.DataFrame([{chave:getattr(dados,chave) for chave in FEATURES}],columns=FEATURES)
        previsao=max(0.0,float(modelo_regressao.predict(entrada)[0]))
        tempo=(perf_counter()-inicio)*1000
        return {"sucesso":True,"projeto":"Campos Zone","modelo_utilizado":nome_modelo_regressao,"previsao":{"tempo_restante_minutos":round(previsao,2),"mae_referencia_minutos":round(float(mae_modelo),4) if mae_modelo is not None else None,"tempo_processamento_ms":round(tempo,2)},"dados_informados":entrada.iloc[0].to_dict(),"origem_dados_treinamento":"simulados","observacao":"Estimativa experimental treinada com snapshots sintéticos. A distância está em unidades do mundo Godot, e os resultados não representam dados reais de partidas."}
    except Exception as exc: raise HTTPException(status_code=500,detail=f"Erro durante a previsão: {exc}") from exc
