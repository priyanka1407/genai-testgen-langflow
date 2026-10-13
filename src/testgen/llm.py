"""
LLM backends.

ClaudeLLM   - Anthropic SDK with structured outputs
              (client.messages.parse -> validated Pydantic object).
GeminiLLM   - Google Gen AI SDK with a JSON response schema
              (models.generate_content -> validated Pydantic object).
OfflineLLM  - deterministic, rule-based stand-in (see offline.py) so the whole
              pipeline, the Langflow flow and CI can run without an API key.

All expose: generate(system, prompt, schema, context) -> schema instance.
The provider is chosen from the model name: "gemini-*" -> Gemini, anything else -> Claude.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel

log = logging.getLogger("testgen.llm")
T = TypeVar("T", bound=BaseModel)


def default_model() -> str:
    """Read at call time (not import time) so a .env loaded at startup is honoured."""
    return os.getenv("TESTGEN_MODEL", "claude-opus-5-5")


def default_judge_model() -> str:
    """The judge can be pointed at a different model; defaults to the generator model."""
    return os.getenv("TESTGEN_JUDGE_MODEL") or default_model()


class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str, schema: type[T], context: dict) -> T: ...


class LLMRefusal(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """Quota that will not reset within a sensible wait (e.g. a daily free-tier limit). Never retried."""


class ClaudeLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 16000, api_key: str | None = None):
        import anthropic  # imported lazily so offline mode needs no SDK

        # api_key=None -> ANTHROPIC_API_KEY env var or an `ant auth login` profile
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model or default_model()
        self.max_tokens = max_tokens
        self.name = self.model

    def generate(self, system: str, prompt: str, schema: type[T], context: dict) -> T:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        log.info(
            "llm call model=%s schema=%s in=%s out=%s cache_read=%s request_id=%s",
            self.model, schema.__name__, response.usage.input_tokens, response.usage.output_tokens,
            response.usage.cache_read_input_tokens, response._request_id,
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal(f"Model declined: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Output truncated - split the input or raise max_tokens")
        return response.parsed_output


class GeminiLLM:
    RETRY_CODES = {429, 500, 502, 503, 504}   # rate limit (common on the free tier) + transient errors
    BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

    MAX_WAIT = 90   # seconds; a 429 asking us to wait longer (daily quota) fails immediately

    def __init__(self, model: str, max_tokens: int = 16000, api_key: str | None = None, max_retries: int = 5):
        from google import genai  # imported lazily so offline / Claude mode needs no Google SDK
        from google.genai import errors, types

        self._errors, self._types = errors, types
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("No Gemini API key: set GEMINI_API_KEY, or paste the key into the node's API Key field")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.name = model

    def generate(self, system: str, prompt: str, schema: type[T], context: dict) -> T:
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,          # SDK converts the Pydantic model to Gemini's schema format
            max_output_tokens=self.max_tokens,
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                break
            except self._errors.APIError as e:
                wait, quota = self._quota_info(e) if e.code == 429 else (None, "")
                if e.code == 429 and ((wait is not None and wait > self.MAX_WAIT) or "PerDay" in quota):
                    raise QuotaExhausted(        # a daily quota will not reset in time - fail fast
                        f"Gemini quota exhausted for {self.model} ({quota or 'limit reached'}); "
                        f"resets in about {round((wait or 0) / 3600, 1)} h. Use another model, "
                        "enable billing in Google AI Studio, or switch Offline on.") from e
                if e.code not in self.RETRY_CODES or attempt == self.max_retries:
                    raise
                delay = min(60, wait if wait is not None else 5 * 2 ** attempt)   # Google's wait, else backoff
                log.warning("gemini %s (%s) - retry %d in %ss", e.code, e.status, attempt + 1, delay)
                time.sleep(delay)

        usage = response.usage_metadata
        log.info("llm call model=%s schema=%s in=%s out=%s thinking=%s", self.model, schema.__name__,
                 getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None),
                 getattr(usage, "thoughts_token_count", None))
        feedback = response.prompt_feedback
        if feedback and feedback.block_reason:
            raise LLMRefusal(f"Gemini blocked the prompt: {feedback.block_reason}")
        finish = response.candidates[0].finish_reason if response.candidates else None
        finish_name = getattr(finish, "name", str(finish))
        if finish_name == "MAX_TOKENS":
            raise RuntimeError("Output truncated - split the input or raise max_tokens")
        if finish_name in self.BLOCKED:
            raise LLMRefusal(f"Gemini stopped the response: {finish_name}")
        if isinstance(response.parsed, schema):
            return response.parsed
        return schema.model_validate_json(response.text)   # validate ourselves if the SDK could not parse

    @staticmethod
    def _quota_info(e) -> tuple[float | None, str]:
        """(seconds Google asks us to wait, quota id) from a 429 error body."""
        wait, quota = None, ""
        body = e.details if isinstance(getattr(e, "details", None), dict) else {}
        for d in body.get("error", {}).get("details", []) or []:
            if "RetryInfo" in d.get("@type", "") and str(d.get("retryDelay", "")).endswith("s"):
                try:
                    wait = float(d["retryDelay"][:-1])
                except ValueError:
                    pass
            for v in d.get("violations", []) or []:
                quota = v.get("quotaId", quota)
        return wait, quota


def provider_for(model: str) -> str:
    return "gemini" if model.lower().startswith("gemini") else "anthropic"


def get_llm(offline: bool | None = None, api_key: str | None = None, model: str | None = None) -> LLM:
    if offline is None:
        offline = os.getenv("TESTGEN_OFFLINE", "").lower() in {"1", "true", "yes"}
    if offline:
        from .offline import OfflineLLM
        return OfflineLLM()
    model = model or default_model()
    if provider_for(model) == "gemini":
        return GeminiLLM(model=model, api_key=api_key)
    return ClaudeLLM(model=model, api_key=api_key)
