from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List
from typing import Optional, Type, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class TokenStreamHandler(BaseCallbackHandler):
    def __init__(self, label: str = "LLM") -> None:
        self.label = label

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if not token:
            return
        print(f"[{self.label}] {token}", end="", flush=True)


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: Optional[int] = 30,
    streaming: bool = False,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    max_retries: int = 2,
) -> ChatGroq:
    if os.getenv("MACRS_USE_LLM", "1").lower() in {"0", "false", "no"}:
        raise RuntimeError("LLM usage disabled via MACRS_USE_LLM")
    model_name = model or os.getenv("MACRS_LLM_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        streaming=streaming,
        callbacks=callbacks,
    )


def generate_structured_output(prompt: str, schema: Type[T], model: Optional[str] = None) -> Optional[T]:
    try:
        stream_tokens = os.getenv("MACRS_STREAM_TOKENS", "0").lower() in {"1", "true", "yes"}
        callbacks = [TokenStreamHandler()] if stream_tokens else None
        llm = get_llm(model=model, streaming=stream_tokens, callbacks=callbacks)
    except RuntimeError as exc:
        logging.info("%s", exc)
        return None
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    system = SystemMessage(
        content=(
            "You are a strict JSON generator. "
            "Return only a JSON object that matches the provided JSON schema. "
            "Do not include code fences, comments, or extra text."
        )
    )
    user = HumanMessage(content=f"JSON Schema:\n{schema_json}\n\nTask:\n{prompt}\n\nReturn JSON only.")
    try:
        start = time.perf_counter()
        response = llm.invoke([system, user])
        elapsed = time.perf_counter() - start
        logging.info("LLM call completed in %.2fs", elapsed)
    except Exception as exc:
        logging.warning("LLM call failed: %s", exc)
        return None

    raw = response.content if hasattr(response, "content") else str(response)
    payload = _extract_json(raw)
    if payload is None:
        logging.warning("Unable to parse JSON from LLM response")
        return None
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        logging.warning("LLM JSON failed schema validation: %s", exc)
        return None


def _extract_json(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None
