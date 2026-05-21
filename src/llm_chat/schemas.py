"""Pydantic schemas for the LLM Chat Assistant API."""
from pydantic import BaseModel, Field
from typing import Optional, List


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation session ID")
    message: str = Field(..., min_length=1, max_length=4096)
    use_rag: bool = Field(default=False, description="Enable RAG retrieval")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: Optional[List[str]] = None
    tokens_used: Optional[int] = None
