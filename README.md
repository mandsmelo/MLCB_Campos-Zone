# Campos Zone — Machine Learning

## 1. Sobre o projeto

O Campos Zone é um jogo cooperativo de sobrevivência contra zumbis, desenvolvido em contexto universitário.

Este repositório apresenta duas aplicações de Machine Learning integradas ao projeto:

- **Proposta 2:** Chatbot inteligente para responder dúvidas sobre o jogo.
- **Proposta 4:** Modelo de regressão para estimar o tempo de conclusão das missões.

## 2. Proposta 2 — Chatbot com RAG

Desenvolvimento de um chatbot que utiliza busca semântica para recuperar informações de uma base de conhecimento e um modelo de linguagem para gerar respostas contextualizadas.

**Tecnologias:** Python, FastAPI, Sentence Transformers, ChromaDB, Ollama e Qwen 2.5 1.5B.

**Resultados:**
- 30 perguntas e respostas indexadas.
- Busca por similaridade de cosseno.
- Integração com modelo generativo.
- Endpoint `POST /faq/search`.
- Acurácia Top-1: 66,67%.
- Recall@3: 91,67%.
- Tempo médio de busca vetorial: 4,87 ms.

*Observação: o tempo médio informado corresponde à etapa de busca vetorial, não ao tempo total de geração da resposta.*

## 3. Proposta 4 — Regressão

Desenvolvimento de um modelo para estimar o tempo de conclusão de uma missão com base nas características da partida.

**Variáveis:** zumbis derrotados, distância percorrida, dano recebido, itens coletados, salas exploradas e dificuldade.

Foram utilizados **600 registros simulados** para treinamento e avaliação de dois modelos.

| Métrica | Regressão Linear | Random Forest |
|---|---:|---:|
| MAE | 1,8632 | 2,6522 |
| RMSE | 2,2830 | 3,3003 |
| R² | 0,9276 | 0,8487 |

A Regressão Linear apresentou o melhor desempenho e foi escolhida para disponibilização na API.

**Endpoint:** `POST /predict/value`.

*Observação: os dados utilizados são simulados, não provenientes de partidas reais.*

## 4. API

O projeto utiliza FastAPI para disponibilizar os serviços por meio de uma API REST.

| Endpoint | Método | Funcionalidade |
|---|---|---|
| `/` | GET | Informações da API |
| `/health` | GET | Verificação de disponibilidade |
| `/faq/search` | POST | Chatbot |
| `/predict/value` | POST | Previsão de tempo |

A documentação interativa dos endpoints está disponível em `/docs`.

### Execução local

Com as dependências Python instaladas, a API pode ser iniciada a partir da pasta raiz do projeto com:

```bash
uvicorn api.servidor:app --reload
```

Para utilizar o chatbot, também é necessário ter o Ollama instalado e o modelo `qwen2.5:1.5b` disponível localmente.

A base vetorial pode ser criada ou reconstruída executando:

```bash
python chatbot/indexacao.py
```

Após iniciar a API, a documentação interativa pode ser acessada em:

`http://127.0.0.1:8000/docs`

## 5. Tecnologias

Python, FastAPI, Sentence Transformers, ChromaDB, Ollama, Qwen, Scikit-learn, Pandas e Jupyter Notebook.

## 6. Integração com Godot

As soluções foram integradas ao jogo Campos Zone, desenvolvido na Godot, por meio de requisições HTTP à API.

O chatbot permite consultar informações sobre as mecânicas do jogo, enquanto o modelo de regressão disponibiliza estimativas de tempo de conclusão das missões.

## 7. Considerações finais

O projeto implementa duas soluções de Machine Learning para o Campos Zone: um chatbot inteligente para auxiliar os jogadores e um modelo de regressão para estimar o tempo de conclusão das missões.

Os resultados demonstram o funcionamento das soluções, sua disponibilização por meio de uma API REST e sua integração com o jogo desenvolvido na Godot.
