"""
Parte 2 - Questão 3
API FastAPI de tradução EN -> DE usando LangChain + HuggingFace (Helsinki-NLP/opus-mt-en-de).

O LangChain gerencia a chamada ao modelo local por meio de um LLM personalizado
(MarianTranslationLLM), que herda de langchain_core.language_models.llms.LLM e usa
AutoTokenizer + AutoModelForSeq2SeqLM da HuggingFace.

Por que um LLM personalizado? O wrapper pronto HuggingFacePipeline depende da tarefa
"translation" do pipeline do transformers, que foi removida no transformers 5. Com este
wrapper próprio o LangChain continua gerenciando a chamada (invoke/batch, cadeias e
parsers) e o código funciona em qualquer versão do transformers.

Diferente de um LLM de chat, o opus-mt é um modelo de tradução dedicado: ele recebe
APENAS o texto a ser traduzido (sem prompt de instrução).

Como executar:
    uvicorn p2q3:app --reload --port 8005

Exemplo de requisição:
    curl -X POST http://127.0.0.1:8005/translate \
         -H "Content-Type: application/json" \
         -d '{"text": "The weather is beautiful today."}'
"""

import re
import time
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
MAX_CHARS_PER_CHUNK = 400
MAX_TOKENS = 512


class MarianTranslationLLM(LLM):
    """LLM do LangChain que traduz com um modelo MarianMT (opus-mt) da HuggingFace."""

    hf_model_id: str
    hf_tokenizer: Any = None
    hf_model: Any = None
    max_tokens: int = MAX_TOKENS
    num_beams: int = 4

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Any) -> "MarianTranslationLLM":
        """Baixa/carrega tokenizer e modelo uma única vez."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model.eval()
        return cls(hf_model_id=model_id, hf_tokenizer=tokenizer, hf_model=model, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "huggingface-marian-translation"

    def _translate(self, texts: List[str]) -> List[str]:
        inputs = self.hf_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens
        )
        with torch.inference_mode():
            output_ids = self.hf_model.generate(
                **inputs, num_beams=self.num_beams, max_new_tokens=self.max_tokens
            )
        return self.hf_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._translate([prompt])[0]

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # Traduz todos os blocos de uma vez (batching real, mais rápido que um a um).
        texts = self._translate(prompts)
        return LLMResult(generations=[[Generation(text=t)] for t in texts])


resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cria o LLM do LangChain uma única vez, na inicialização."""
    llm = MarianTranslationLLM.from_model_id(MODEL_NAME)
    # Cadeia: LLM (HuggingFace/Marian) -> parser de string
    resources["chain"] = llm | StrOutputParser()
    yield
    resources.clear()


app = FastAPI(
    title="API de Tradução EN -> DE (LangChain + HuggingFace)",
    description="Traduz inglês para alemão com Helsinki-NLP/opus-mt-en-de gerenciado pelo LangChain.",
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
    latency_seconds: float


def split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Divide o texto em sentenças e agrupa até max_chars por bloco (textos longos)."""
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


@app.get("/")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    chain = resources.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")

    chunks = split_into_chunks(req.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="O texto não pode ser vazio.")

    start = time.perf_counter()
    try:
        results = chain.batch(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na tradução: {exc}") from exc
    latency = time.perf_counter() - start

    return TranslateResponse(
        original_text=req.text,
        translated_text=" ".join(r.strip() for r in results),
        model=MODEL_NAME,
        chunks=len(chunks),
        latency_seconds=round(latency, 3),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("p2q3:app", host="127.0.0.1", port=8005, reload=False)
