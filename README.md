# Campos Zone — Machine Learning

## 1. Sobre o projeto

O Campos Zone é um jogo cooperativo de sobrevivência contra zumbis, desenvolvido em contexto universitário.

Este repositório apresenta duas aplicações de Machine Learning integradas ao projeto:

- **Proposta 2:** Chatbot para responder dúvidas sobre o jogo por meio de busca semântica.
- **Proposta 4:** Modelo de regressão para estimar o tempo restante das missões.

## 2. Proposta 2 — Chatbot com busca semântica

Desenvolvimento de um chatbot que utiliza embeddings e similaridade semântica para recuperar respostas de uma base de conhecimento sobre as mecânicas do jogo.

**Tecnologias:** Python, FastAPI, Sentence Transformers e ChromaDB.

**Resultados:**

- 30 perguntas e respostas indexadas na base de conhecimento.
- Busca por similaridade de cosseno.
- Armazenamento e recuperação dos vetores utilizando ChromaDB.
- Endpoint `POST /faq/search`.
- Precisão Top-1: **75,00%**.
- Recall@3: **91,67%**.
- Similaridade média do primeiro resultado: **63,71%**.
- Tempo médio de busca vetorial: **4,21 ms**.
- Maior tempo médio por pergunta: **5,55 ms**.

*Observação: os tempos informados correspondem exclusivamente à etapa de busca vetorial, não incluindo a geração dos embeddings ou o processamento total da API.*

## 3. Proposta 4 — Regressão

Desenvolvimento de um modelo para estimar o tempo restante de uma missão com base nas características da partida.

**Variáveis utilizadas:** distância percorrida, dano recebido, itens coletados, salas exploradas e tempo decorrido.

Foram utilizados **600 registros simulados** para treinamento e avaliação de dois modelos.

| **Métrica** | **Regressão Linear** | **Random Forest** |
|---|---:|---:|
| MAE | 1,8632 | 2,6522 |
| RMSE | 2,2830 | 3,3003 |
| R² | 0,9276 | 0,8487 |

A Regressão Linear apresentou o melhor desempenho e foi escolhida para disponibilização na API.

**Endpoint:** `POST /predict/value`.

*Observação: os dados utilizados são simulados, não provenientes de partidas reais.*

## 4. API

O projeto utiliza FastAPI para disponibilizar os serviços por meio de uma API REST.

| **Endpoint** | **Método** | **Funcionalidade** |
|---|---|---|
| `/` | GET | Informações da API |
| `/health` | GET | Verificação de disponibilidade |
| `/faq/search` | POST | Busca semântica no chatbot |
| `/predict/value` | POST | Previsão do tempo restante da missão |

A documentação interativa dos endpoints está disponível em `/docs`.

### Execução local

Com as dependências Python instaladas, a API pode ser iniciada a partir da pasta raiz do projeto com:

```bash
uvicorn api.servidor:app --reload
```

A base vetorial pode ser criada ou reconstruída executando:

```bash
python chatbot/indexacao.py
```

Para avaliar a recuperação semântica, execute:

```bash
python chatbot/avaliar_sinonimos.py
```

O relatório detalhado da avaliação é salvo em `resultados/avaliacao_sinonimos.csv`.

Após iniciar a API, a documentação interativa pode ser acessada em:

`http://127.0.0.1:8000/docs`

## 5. Tecnologias

Python, FastAPI, Sentence Transformers, ChromaDB, Scikit-learn, Pandas e Jupyter Notebook.

## 6. Integração com Godot

As soluções foram integradas ao jogo Campos Zone, desenvolvido na Godot, por meio de requisições HTTP à API.

O chatbot permite consultar informações sobre as mecânicas do jogo, recuperando respostas cadastradas na base de conhecimento. O modelo de regressão disponibiliza estimativas do tempo restante das missões.

## 7. Considerações finais

O projeto implementa duas soluções de Machine Learning para o Campos Zone: um chatbot com recuperação semântica de informações para auxiliar os jogadores e um modelo de regressão para estimar o tempo restante das missões.

Os resultados demonstram o funcionamento das soluções, sua disponibilização por meio de uma API REST e sua integração com o jogo desenvolvido na Godot.

A avaliação do chatbot apresentou 75,00% de precisão Top-1 e 91,67% de Recall@3, com tempo médio de busca vetorial de 4,21 ms. Já a Regressão Linear apresentou o melhor desempenho entre os modelos avaliados para a estimativa do tempo de missão.