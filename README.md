# pedro_de_mello_ramos — DR3 TP2 (FastAPI e LangChain com LLMs)

Teste de Performance com cinco aplicações simples usando FastAPI e LangChain.

| Arquivo   | Questão            | Descrição                                         | Porta |
|-----------|--------------------|---------------------------------------------------|-------|
| `p1q1.py` | Parte 1 – Q1       | Geração de texto com GPT-2 (HuggingFace)          | 8001  |
| `p1q2.py` | Parte 1 – Q2       | Tradução EN→FR com `Helsinki-NLP/opus-mt-en-fr`   | 8002  |
| `p2q1.py` | Parte 2 – Q1       | Chatbot com LangChain + Fake LLM                  | 8003  |
| `p2q2.py` | Parte 2 – Q2       | Tradução EN→FR com LangChain + Groq               | 8004  |
| `p2q3.py` | Parte 2 – Q3       | Tradução EN→DE com LangChain + HuggingFace        | 8005  |

As análises, os diagramas e a tabela comparativa estão no PDF da entrega.

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Para `p2q2.py`, copie `.env.example` para `.env` e preencha `GROQ_API_KEY`
(chave gratuita em https://console.groq.com/keys).

## Execução

```bash
uvicorn p1q1:app --port 8001      # ou: python p1q1.py
uvicorn p1q2:app --port 8002
uvicorn p2q1:app --port 8003
uvicorn p2q2:app --port 8004
uvicorn p2q3:app --port 8005
```

A documentação interativa (Swagger) de cada API fica em `http://127.0.0.1:<porta>/docs`.
Na primeira execução, os modelos da HuggingFace são baixados automaticamente.

## Exemplos de requisição

```bash
curl -X POST http://127.0.0.1:8001/generate -H "Content-Type: application/json" \
     -d '{"text": "Once upon a time", "max_new_tokens": 50}'

curl -X POST http://127.0.0.1:8002/translate -H "Content-Type: application/json" \
     -d '{"text": "The weather is beautiful today."}'

curl -X POST http://127.0.0.1:8003/chat -H "Content-Type: application/json" \
     -d '{"message": "Qual é o horário de atendimento?"}'

curl -X POST http://127.0.0.1:8004/translate -H "Content-Type: application/json" \
     -d '{"text": "Good morning! How are you?"}'

curl -X POST http://127.0.0.1:8005/translate -H "Content-Type: application/json" \
     -d '{"text": "The weather is beautiful today."}'
```
