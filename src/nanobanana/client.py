from __future__ import annotations

import base64
import random
import time
from typing import Any, Callable

try:
    from google import genai
except ImportError:
    genai = None

from nanobanana.constants import ExitCode
from nanobanana.types import ImageRequest, ModelDecision
from nanobanana.utils import _get, vlog

# =============================================================================
# Request Construction
# =============================================================================


def build_interaction_input(
    request: ImageRequest,
    normalized_prompt: str,
    decision: ModelDecision,
) -> dict:
    """Build kwargs for client.interactions.create."""
    input_data: dict[str, Any] = {
        "model": decision.resolved,
        "contents": [{"role": "user", "parts": [{"text": normalized_prompt}]}],
    }

    if request.references:
        parts: list[dict[str, Any]] = [{"text": normalized_prompt}]
        for ref in request.references:
            image_bytes = ref.path.read_bytes()
            b64 = base64.b64encode(image_bytes).decode("ascii")
            parts.append({"inline_data": {"mime_type": ref.mime_type, "data": b64}})
        input_data["contents"] = [{"role": "user", "parts": parts}]

    if request.text_output:
        input_data["response_format"] = [
            {"type": "text"},
            {"type": request.mime_type},
        ]
    else:
        input_data["response_format"] = {"type": request.mime_type}

    if request.grounding:
        input_data["tools"] = [{"type": "google_search"}]

    generation_config: dict[str, Any] = {}
    if request.thinking_level and request.thinking_level != "auto":
        generation_config["thinking_level"] = request.thinking_level
    if request.seed:
        try:
            generation_config["seed"] = int(request.seed)
        except ValueError:
            generation_config["seed"] = request.seed
    if generation_config:
        input_data["generation_config"] = generation_config

    return input_data


# =============================================================================
# Retry Handler
# =============================================================================


def with_retry(
    fn: Callable[[], Any],
    max_attempts: int = 3,
    max_delay: float = 30.0,
) -> Any:
    """Run fn with exponential backoff and jitter, retrying transient errors."""
    delay = 1.0
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exception = exc
            if not is_retryable(exc):
                raise
            if attempt == max_attempts:
                break
            vlog(
                attempt=attempt,
                max_attempts=max_attempts,
                retry_delay=round(delay, 2),
                error=type(exc).__name__,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay) * random.uniform(0.8, 1.2)
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retry exhausted without exception")


def is_retryable(exc: Exception) -> bool:
    """Return True for retryable network/server errors; False for client errors."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is not None:
        code_str = str(code)
        return code_str == "429" or code_str.startswith("5")
    message = str(exc).lower()
    return (
        "429" in message
        or "rate limit" in message
        or "timeout" in message
        or "connection" in message
    )


def classify_api_exception(exc: Exception) -> int:
    """Map an API exception to the appropriate exit code."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is not None:
        code_str = str(code)
        if code_str == "429":
            return ExitCode.QUOTA_EXCEEDED
        if code_str in {"401", "403"}:
            return ExitCode.AUTH_FAILURE
        if code_str.startswith("4"):
            return ExitCode.INVALID_ARGS
    message = str(exc).lower()
    if "quota" in message or "rate limit" in message or "429" in message:
        return ExitCode.QUOTA_EXCEEDED
    if "api key" in message or "unauthorized" in message or "forbidden" in message:
        return ExitCode.AUTH_FAILURE
    if "safety" in message or "blocked" in message:
        return ExitCode.SAFETY_REFUSAL
    return ExitCode.INTERNAL


def is_safety_refusal(response: object) -> bool:
    """Check finish_reason and prompt feedback for safety blocks."""
    if response is None:
        return False
    candidates = _get(response, "candidates", []) or []
    for candidate in candidates:
        finish = _get(candidate, "finish_reason")
        if finish is not None and str(finish).upper() in {"SAFETY", "BLOCKED"}:
            return True
    prompt_feedback = _get(response, "prompt_feedback")
    if prompt_feedback is not None:
        block = _get(prompt_feedback, "block_reason", "")
        if block and "safety" in str(block).lower():
            return True
    return False


# =============================================================================
# Execution
# =============================================================================


def execute_request(client, input_data: dict) -> object:
    """Call client.interactions.create with retry."""
    return with_retry(lambda: client.interactions.create(**input_data))
