from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ask_vector_data import OLLAMA_MODEL, OLLAMA_URL, TOP_K, VECTOR_DB_HOST, VECTOR_DB_PORT, answer_question

app = FastAPI(title="ai-rag-text-vector ask service")
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


class AskRequest(BaseModel):
    query: str


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_db_host": VECTOR_DB_HOST,
        "vector_db_port": VECTOR_DB_PORT,
        "top_k": TOP_K,
        "ollama_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
    }


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer_question(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
