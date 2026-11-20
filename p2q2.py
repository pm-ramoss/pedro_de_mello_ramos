"""
Parte 2 - Questão 2
API FastAPI de tradução EN -> FR usando LangChain + Groq (API compatível com a OpenAI).

Pré-requisitos:
    1) Crie uma chave em https://console.groq.com/keys
    2) Copie .env.example para .env e preencha GROQ_API_KEY (nunca suba o .env ao GitHub)

Como executar:
    uvicorn p2q2:app --reload --port 8004

Exemplo de requisição:
    curl -X POST http://127.0.0.1:8004/translate \
         -H "Content-Type: application/json" \
         -d '{"text": "Good morning! How are you?"}'
"""

import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

# O modelo é configurável por variável de ambiente, pois a Groq troca/aposenta
# modelos com frequência. Confira a lista atual em https://console.groq.com/docs/models
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

app = FastAPI(
    title="API de Tradução EN -> FR (LangChain + Groq)",
    description="Traduz inglês para francês chamando um LLM da Groq via LangChain.",
    version="1.0.0",
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um tradutor profissional. Traduza o texto do usuário do inglês "
            "para o francês. Responda SOMENTE com a tradução, sem comentários, "
            "sem aspas e sem explicações.",
        ),
        ("human", "{text}"),
    ]
)


@lru_cache(maxsize=1)
def build_chain():
    """Monta (uma única vez) a cadeia prompt -> LLM -> parser.

    O resultado fica em cache para reaproveitar o cliente HTTP entre requisições.
    Falha com erro claro se faltar a chave (a exceção não é guardada em cache).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY não definida. Preencha o arquivo .env.")
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=2, timeout=30, api_key=api_key)
    return prompt | llm | StrOutputParser()


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto em inglês a ser traduzido.")


class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    model: str
    latency_seconds: float


@app.get("/")
def health() -> dict:
    return {"status": "ok", "model": GROQ_MODEL, "api_key_configurada": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    try:
        chain = build_chain()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    start = time.perf_counter()
    try:
        translated = await chain.ainvoke({"text": req.text})
    except Exception as exc:  # limite de taxa, chave inválida, modelo inexistente etc.
        raise HTTPException(status_code=502, detail=f"Falha ao chamar a API da Groq: {exc}") from exc
    latency = time.perf_counter() - start

    return TranslateResponse(
        original_text=req.text,
        translated_text=translated.strip(),
        model=GROQ_MODEL,
        latency_seconds=round(latency, 3),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("p2q2:app", host="127.0.0.1", port=8004, reload=False)
