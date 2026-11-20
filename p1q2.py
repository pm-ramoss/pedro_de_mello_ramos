"""
Parte 1 - Questão 2
API FastAPI de tradução inglês -> francês com Helsinki-NLP/opus-mt-en-fr.

O modelo (MarianMT) é carregado diretamente com AutoTokenizer + AutoModelForSeq2SeqLM
e a tradução é feita com model.generate(). Essa forma funciona em qualquer versão do
transformers (a tarefa "translation" do pipeline foi removida no transformers 5).

Como executar:
    uvicorn p1q2:app --reload --port 8002

Exemplo de requisição:
    curl -X POST http://127.0.0.1:8002/translate \
         -H "Content-Type: application/json" \
         -d '{"text": "The weather is beautiful today."}'

Textos longos: o modelo opus-mt aceita até ~512 tokens por vez. Por isso o texto é
dividido em sentenças, agrupado em blocos curtos e traduzido em lote; ao final os
blocos são unidos novamente.
"""

import re
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
MAX_CHARS_PER_CHUNK = 400  # blocos pequenos ficam bem abaixo do limite de 512 tokens
MAX_TOKENS = 512
BATCH_SIZE = 8

resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega tokenizer e modelo uma única vez, na inicialização."""
    resources["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    resources["model"] = model
    yield
    resources.clear()


app = FastAPI(
    title="API de Tradução EN -> FR (HuggingFace)",
    description="Traduz textos do inglês para o francês com o modelo Helsinki-NLP/opus-mt-en-fr.",
    version="1.0.0",
    lifespan=lifespan,
)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto em inglês a ser traduzido.")


class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    model: str
    chunks: int


def split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Divide o texto em sentenças e agrupa até max_chars por bloco."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def translate_texts(texts: list[str]) -> list[str]:
    """Traduz uma lista de textos em lotes usando tokenizer + model.generate()."""
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    results: list[str] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, num_beams=4, max_new_tokens=MAX_TOKENS)
        results.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return results


@app.get("/")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    if "model" not in resources:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")

    chunks = split_into_chunks(req.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="O texto não pode ser vazio.")

    try:
        translated_chunks = translate_texts(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na tradução: {exc}") from exc

    return TranslateResponse(
        original_text=req.text,
        translated_text=" ".join(t.strip() for t in translated_chunks),
        model=MODEL_NAME,
        chunks=len(chunks),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("p1q2:app", host="127.0.0.1", port=8002, reload=False)
