"""
Parte 1 - Questão 1
API FastAPI para geração de texto com GPT-2 (HuggingFace Transformers).

Como executar:
    uvicorn p1q1:app --reload --port 8001

Exemplo de requisição:
    curl -X POST http://127.0.0.1:8001/generate \
         -H "Content-Type: application/json" \
         -d '{"text": "Once upon a time", "max_new_tokens": 50}'
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, set_seed

MODEL_NAME = "gpt2"

# Dicionário que guarda o modelo carregado (evita variável global solta).
resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma única vez, na inicialização do servidor."""
    resources["generator"] = pipeline("text-generation", model=MODEL_NAME)
    yield
    resources.clear()


app = FastAPI(
    title="API de Geração de Texto - GPT-2",
    description="Gera a continuação de uma frase usando o GPT-2 da HuggingFace.",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Frase de entrada (prompt).")
    max_new_tokens: int = Field(50, ge=1, le=200, description="Tamanho máximo da continuação.")
    temperature: float = Field(0.8, gt=0, le=2.0, description="Aleatoriedade da geração.")
    seed: int | None = Field(None, description="Semente opcional para reproduzir o resultado.")


class GenerateResponse(BaseModel):
    input_text: str
    generated_text: str
    model: str


@app.get("/")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    generator = resources.get("generator")
    if generator is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")

    if req.seed is not None:
        set_seed(req.seed)

    try:
        output = generator(
            req.text,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            num_return_sequences=1,
            pad_token_id=50256,  # token <|endoftext|> do GPT-2 (evita warning)
        )
    except Exception as exc:  # erro inesperado do modelo
        raise HTTPException(status_code=500, detail=f"Erro na geração: {exc}") from exc

    return GenerateResponse(
        input_text=req.text,
        generated_text=output[0]["generated_text"],
        model=MODEL_NAME,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("p1q1:app", host="127.0.0.1", port=8001, reload=False)
