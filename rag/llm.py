"""
STEP 7 — LLM client.

OpenAI-compatible API use karte hain, to SIRF .env badal ke provider badal sakte ho:
  Groq   : LLM_BASE_URL=https://api.groq.com/openai/v1   LLM_MODEL=llama-3.3-70b-versatile
  OpenAI : LLM_BASE_URL=https://api.openai.com/v1        LLM_MODEL=gpt-4o-mini
  Ollama : LLM_BASE_URL=http://localhost:11434/v1        LLM_MODEL=llama3.1   (local, free, key = "ollama")
"""
from __future__ import annotations

from typing import Iterator, List


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 1024):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "LLM API key nahi mili. `.env` file mein LLM_API_KEY (ya GROQ_API_KEY) set karo. "
                    "Free Groq key: https://console.groq.com/keys"
                )
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=60, max_retries=2)
        return self._client

    def chat(self, messages: List[dict], temperature: float | None = None, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: List[dict]) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for event in resp:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                yield event.choices[0].delta.content
