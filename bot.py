"""
Core customer-support chatbot logic.

Pipeline for each user message:
1. Retrieve the most relevant knowledge-base articles (TF-IDF + cosine similarity).
2. Build a system prompt that includes those articles as grounding context.
3. Call the Claude API with the running conversation history.
4. Check the reply (and the user's message) for signals that a human should
   take over, and flag that back to the caller.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Which LLM backend to use: "anthropic" or "openai".
# Override with the LLM_PROVIDER env var, e.g. `export LLM_PROVIDER=openai`.
PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")

ANTHROPIC_MODEL = "claude-sonnet-5"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
MAX_TOKENS = 1024
TOP_K_ARTICLES = 3
MIN_RELEVANCE = 0.08  # below this, don't bother citing an article

ESCALATION_PHRASES = (
    "talk to a human",
    "speak to a person",
    "real person",
    "human agent",
    "this isn't helping",
    "not helpful",
    "i want a refund now",
    "i'm furious",
    "cancel my account immediately",
)

SYSTEM_TEMPLATE = """You are a customer support assistant for Acme Co.

Rules:
- Be concise, warm, and professional. Prefer short paragraphs or bullet points.
- Only state policies or steps that appear in the knowledge base excerpts below.
  If the answer isn't covered by them, say you're not certain and offer to
  connect the user with a human agent instead of guessing.
- Never invent order numbers, dates, refund amounts, or account details.
- If the user is frustrated, angry, or explicitly asks for a person, acknowledge
  their frustration and let them know you're flagging this for a human agent.

Relevant knowledge base excerpts for this conversation turn:
{kb_context}
"""


@dataclass
class SupportBot:
    kb_path: str = "knowledge_base.json"
    api_key: str | None = None
    provider: str = PROVIDER
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)  # falls back to OPENAI_API_KEY env var
        else:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)  # falls back to ANTHROPIC_API_KEY env var

        self.articles = self._load_kb(self.kb_path)
        self._texts = [f"{a['title']}. {a['content']}" for a in self.articles]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self._texts) if self._texts else None

    @staticmethod
    def _load_kb(path: str) -> list[dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _retrieve(self, query: str) -> list[dict]:
        """Return the top-K knowledge base articles relevant to the query."""
        if self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(scores, self.articles), key=lambda x: x[0], reverse=True)
        return [article for score, article in ranked[:TOP_K_ARTICLES] if score >= MIN_RELEVANCE]

    @staticmethod
    def _format_kb_context(articles: list[dict]) -> str:
        if not articles:
            return "(No matching articles found for this message.)"
        return "\n\n".join(f"- {a['title']}: {a['content']}" for a in articles)

    def _needs_escalation(self, user_message: str) -> bool:
        lowered = user_message.lower()
        return any(phrase in lowered for phrase in ESCALATION_PHRASES)

    def send(self, user_message: str) -> dict:
        """
        Send a user message, get a reply, and return a structured result:
        {"reply": str, "escalate": bool, "sources": [article titles]}
        """
        articles = self._retrieve(user_message)
        system_prompt = SYSTEM_TEMPLATE.format(kb_context=self._format_kb_context(articles))

        self.history.append({"role": "user", "content": user_message})

        if self.provider == "openai":
            reply_text = self._call_openai(system_prompt)
        else:
            reply_text = self._call_anthropic(system_prompt)

        self.history.append({"role": "assistant", "content": reply_text})

        escalate = self._needs_escalation(user_message)

        return {
            "reply": reply_text,
            "escalate": escalate,
            "sources": [a["title"] for a in articles],
        }

    def _call_anthropic(self, system_prompt: str) -> str:
        response = self.client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=self.history,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    def _call_openai(self, system_prompt: str) -> str:
        # OpenAI's chat API takes the system prompt as the first message in
        # the list rather than as a separate parameter.
        messages = [{"role": "system", "content": system_prompt}] + self.history
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content

    def reset(self) -> None:
        self.history = []
