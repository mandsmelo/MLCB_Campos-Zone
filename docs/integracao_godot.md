# Integração do Godot com a API — Campos Zone

## Visão geral

O projeto Campos Zone utiliza uma API desenvolvida em Python com FastAPI para disponibilizar duas funcionalidades de Machine Learning:

- **Chatbot com RAG:** consulta a base de conhecimento do jogo e utiliza o modelo Qwen 2.5 1.5B, executado localmente pelo Ollama, para gerar respostas.
- **Regressão:** estima o tempo de conclusão de uma partida com base nas características informadas.

## Endpoints utilizados

A API deve estar em execução no computador que executa o jogo.

Endereço local:

`http://127.0.0.1:8000`

| Funcionalidade | Método | Endpoint |
|---|---|---|
| Chatbot | POST | `/faq/search` |
| Previsão de tempo | POST | `/predict/value` |

## Integração no Godot

Os scripts de integração utilizados são:

- `chatbot_api.gd`: envia perguntas ao endpoint `/faq/search`.
- `regressao_api.gd`: envia os dados da partida ao endpoint `/predict/value`.

Os scripts utilizam requisições HTTP para se comunicar com a API.

## Como testar

1. Inicie a API no computador.
2. Abra a documentação interativa em `http://127.0.0.1:8000/docs`.
3. Execute os testes dos endpoints `/faq/search` e `/predict/value`.
4. Abra o projeto no Godot e teste as funcionalidades integradas.

## Observações

- O endereço `127.0.0.1` funciona quando o jogo e a API estão no mesmo computador.
- Para executar o jogo em outro computador, será necessário configurar um endereço de API acessível pela rede.
- O chatbot depende do Ollama e do modelo `qwen2.5:1.5b`.
- Os dados utilizados no treinamento da regressão são simulados e identificados como tal no projeto.
