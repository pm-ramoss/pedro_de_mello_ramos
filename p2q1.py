"""
Parte 2 - Questão 1
Protótipo de chatbot com LangChain + Fake LLM, exposto via FastAPI.

O FakeListLLM não chama nenhum modelo real: ele devolve respostas pré-definidas.
Isso permite testar a API, a cadeia (prompt -> LLM -> parser) e o fluxo de dados
sem custo, sem internet, sem chave de API e com latência praticamente zero.

Como executar:
    uvicorn p2q1:app --reload --port 8003

Exemplo de requisição:
    curl -X POST http://127.0.0.1:8003/chat \
         -H "Content-Type: application/json" \
         -d '{"message": "Qual é o horário de atendimento?"}'
"""

import re
import unicodedata

from fastapi import FastAPI
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Base de perguntas pré-definidas: palavra-chave -> resposta simulada
# --------------------------------------------------------------------------------------
FAQ: dict[str, str] = {
    "horario": "Nosso atendimento funciona de segunda a sexta, das 8h às 18h.",
    "preco": "Os planos começam em R$ 49,90 por mês. (resposta simulada)",
    "contato": "Você pode falar conosco pelo e-mail contato@exemplo.com.",
    "ola": "Olá! Eu sou um chatbot de teste. Como posso ajudar?",
    "obrigado": "De nada! Posso ajudar em mais alguma coisa?",
}
DEFAULT_ANSWER = "Desculpe, não entendi a pergunta. Tente: horário, preço ou contato."


def normalize(text: str) -> str:
    """Minúsculas e sem acentos, para facilitar a busca por palavra-chave."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def pick_answer(question: str) -> str:
    q = normalize(question)
    for keyword, answer in FAQ.items():
        if re.search(rf"\b{keyword}", q):  # palavra-chave no início de uma palavra
            return answer
    return DEFAULT_ANSWER


# --------------------------------------------------------------------------------------
# Cadeia LangChain: PromptTemplate -> Fake LLM -> StrOutputParser
# --------------------------------------------------------------------------------------
prompt = PromptTemplate.from_template(
    "Você é um assistente de atendimento. Responda à pergunta do cliente.\n"
    "Pergunta: {question}\nResposta:"
)


def fake_llm_step(prompt_value) -> str:
    """Escolhe a resposta pré-definida e a devolve pelo FakeListLLM.

    O FakeListLLM devolve as respostas da lista em ordem; por isso criamos, a cada
    pergunta, um FakeListLLM com a resposta correspondente. A chamada .invoke()
    segue exatamente o mesmo contrato de um LLM real.
    """
    question = prompt_value.to_string().split("Pergunta:")[-1].split("Resposta:")[0]
    llm = FakeListLLM(responses=[pick_answer(question)])
    return llm.invoke(prompt_value)


chain = prompt | RunnableLambda(fake_llm_step) | StrOutputParser()

# --------------------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------------------
app = FastAPI(
    title="Chatbot de Teste - LangChain + Fake LLM",
    description="Protótipo de chatbot que responde perguntas pré-definidas.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Mensagem do usuário.")


class ChatResponse(BaseModel):
    message: str
    answer: str
    llm: str = "FakeListLLM"


@app.get("/")
def health() -> dict:
    return {"status": "ok", "llm": "FakeListLLM", "perguntas_conhecidas": list(FAQ)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    answer = chain.invoke({"question": req.message})
    return ChatResponse(message=req.message, answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("p2q1:app", host="127.0.0.1", port=8003, reload=False)
