"""FastAPI entry point for the LLM Chat Assistant."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()

from chat_engine import ChatEngine
from rag_retriever import RAGRetriever
from schemas import ChatRequest, ChatResponse

chat_engine = ChatEngine(window_size=10)
retriever = RAGRetriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever.load_index()  # Load FAISS index on startup if available
    yield


app = FastAPI(
    title="LLM Chat Assistant",
    description="GPT-4o powered conversational AI with optional RAG",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint with optional RAG context injection."""
    message = request.message

    if request.use_rag:
        contexts = retriever.retrieve(message)
        if contexts:
            context_block = "\n---\n".join(contexts)
            message = f"[Context]\n{context_block}\n\n[Question]\n{message}"

    try:
        reply = chat_engine.chat(request.session_id, message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(session_id=request.session_id, reply=reply)


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    chat_engine.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
