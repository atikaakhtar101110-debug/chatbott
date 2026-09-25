# Customer Support Chatbot

A small, self-contained AI customer support chatbot built with the Anthropic
Python SDK, scikit-learn (for knowledge-base retrieval), and FastAPI (for a
web API). Grounds answers in your own FAQ/knowledge-base content and flags
conversations that should go to a human.

## How it works

1. **Retrieval** (`bot.py`): each incoming message is compared against your
   knowledge base (`knowledge_base.json`) using TF-IDF + cosine similarity,
   and the top matching articles are pulled in.
2. **Generation**: those articles are injected into the system prompt as
   grounding context, so Claude answers from your actual policies instead of
   guessing, and says so when nothing relevant is found.
3. **Escalation**: messages containing frustration cues or explicit requests
   for a human are flagged (`escalate: true`) so your app can route them to a
   live agent queue.
4. **Memory**: each `SupportBot` instance keeps its own conversation history,
   so replies stay coherent across turns.

## Setup

```bash
pip install -r requirements.txt
```

By default the bot uses the Anthropic API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

To use OpenAI instead, set `LLM_PROVIDER=openai` and an OpenAI key:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-proj-...
export OPENAI_MODEL=gpt-5-mini   # optional, this is the default
```

Everything else (retrieval, escalation, the FastAPI/CLI interfaces) works
identically regardless of provider — only `bot.py`'s `_call_anthropic` /
`_call_openai` methods differ. You can also pass `provider="openai"` directly
when constructing `SupportBot(...)` instead of using the env var.

## Try it in the terminal

```bash
python cli.py
```

## Run it as a web API

```bash
uvicorn app:app --reload --port 8000
```

Then:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?"}'
```

Response:

```json
{
  "session_id": "generated-uuid",
  "reply": "To reset your password...",
  "escalate": false,
  "sources": ["How to reset your password"]
}
```

Pass the returned `session_id` on your next request to continue the same
conversation. `POST /reset` with a `session_id` clears that session's memory.

## Customizing

- **Knowledge base**: edit `knowledge_base.json` — add one object per article
  with `id`, `title`, and `content`. No retraining needed; retrieval is
  computed fresh each run.
- **Tone/rules**: edit `SYSTEM_TEMPLATE` in `bot.py`.
- **Escalation triggers**: edit `ESCALATION_PHRASES` in `bot.py`, or replace
  the keyword check with a small classifier if you need something smarter.
- **Sessions in production**: swap the in-memory `sessions` dict in `app.py`
  for Redis or a database so history survives restarts and works across
  multiple server processes.

## Notes

- The API is stateless per call — `SupportBot` keeps history in memory and
  resends it each turn, which is how the underlying Claude API works.
- For larger knowledge bases (hundreds+ of articles), consider swapping the
  TF-IDF retrieval for a vector database (e.g. Chroma, Pinecone, pgvector)
  with embedding-based search instead.
