
# Campos Zone — Machine Learning

## 1. Sobre o projeto

O Campos Zone é um jogo cooperativo de sobrevivência contra zumbis, desenvolvido em contexto universitário.

Este repositório apresenta duas aplicações de Machine Learning:

- **Proposta 2:** Chatbot inteligente para responder dúvidas sobre o jogo.
- **Proposta 4:** Modelo de regressão para estimar o tempo de conclusão das missões.

## 2. Proposta 2 — Chatbot com RAG

Desenvolvimento de um chatbot que utiliza busca semântica para recuperar informações de uma base de conhecimento e um modelo de linguagem para gerar respostas.

**Tecnologias:** Python, FastAPI, Sentence Transformers, ChromaDB, Ollama e Qwen 2.5 1.5B.

**Resultados:**
- 30 perguntas e respostas indexadas.
- Busca por similaridade de cosseno.
- Integração com modelo generativo.
- Endpoint `POST /faq/search`.
- Acurácia Top-1: 66,67%.
- Recall@3: 91,67%.
- Tempo médio de busca: 4,87 ms.

## 3. Proposta 4 — Regressão

Desenvolvimento de um modelo para estimar o tempo de conclusão de uma missão com base nas características da partida.

**Variáveis:** zumbis derrotados, distância percorrida, dano recebido, itens coletados, salas exploradas e dificuldade.

Foram utilizados **600 registros simulados** para treinamento e avaliação de dois modelos.

| Métrica | Regressão Linear | Random Forest |
|---|---:|---:|
| MAE | 1,8632 | 2,6522 |
| RMSE | 2,2830 | 3,3003 |
| R² | 0,9276 | 0,8487 |

A Regressão Linear apresentou o melhor desempenho e foi escolhida para a API.

**Endpoint:** `POST /predict/value`.

*Observação: os dados utilizados são simulados, não provenientes de partidas reais.*

## 4. API

O projeto utiliza FastAPI para disponibilizar os serviços:

| Endpoint | Método | Funcionalidade |
|---|---|---|
| `/` | GET | Informações da API |
| `/health` | GET | Verificação de disponibilidade |
| `/faq/search` | POST | Chatbot |
| `/predict/value` | POST | Previsão de tempo |

A documentação interativa está disponível em `/docs`.

## 5. Tecnologias

Python, FastAPI, Sentence Transformers, ChromaDB, Ollama, Qwen, Scikit-learn, Pandas e Jupyter Notebook.

## 6. Considerações finais

O projeto implementa duas soluções de Machine Learning para o Campos Zone: um chatbot inteligente para auxiliar os jogadores e um modelo de regressão para estimar o tempo de conclusão das missões.

Os resultados demonstram o funcionamento das soluções e sua disponibilização por meio de uma API REST. E a integração com o jogo no Godot.


