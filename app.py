"""
FastAPI server exposing the support chatbot over HTTP.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /chat   {"session_id": "abc", "message": "..."} -> reply + metadata
    POST /reset  {"session_id": "abc"}                    -> clears that session
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bot import SupportBot

app = FastAPI(title="Customer Support Chatbot")

# In-memory session store: {session_id: SupportBot}.
# Swap this for Redis / a database in production so sessions survive restarts
# and work across multiple server processes.
sessions: dict[str, SupportBot] = {}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    escalate: bool
    sources: list[str]


class ResetRequest(BaseModel):
    session_id: str


def _get_or_create_bot(session_id: str | None) -> tuple[str, SupportBot]:
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = SupportBot()
    return new_id, sessions[new_id]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session_id, bot = _get_or_create_bot(req.session_id)
    result = bot.send(req.message)

    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        escalate=result["escalate"],
        sources=result["sources"],
    )


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    if req.session_id in sessions:
        sessions[req.session_id].reset()
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
