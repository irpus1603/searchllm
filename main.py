from fastapi import FastAPI
from agents.gemini_llm import router as gemini_router  # Relative import
#from agents.qwen_llm import router as qwen_router    # Relative import
from agents.sahabat_llm import router as sahabat_router
from agents.mistral_llm import router as mistral_router
from agents.rag_chat import router as rag_chat_router
from agents.embedding_doc import router as embedding_doc_router
from agents.telegram import router as telegram_router

app = FastAPI()

app.include_router(gemini_router, prefix="/api")
#app.include_router(qwen_router, prefix="/api")
app.include_router(sahabat_router, prefix="/api")
app.include_router(mistral_router, prefix="/api")
app.include_router(rag_chat_router, prefix="/v2")
app.include_router(embedding_doc_router, prefix="/v2")
app.include_router(telegram_router, prefix="/engine")


@app.get("/")
async def root():
    return {"message": "LLM API is running!"}