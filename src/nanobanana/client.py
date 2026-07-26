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
    """Build kwargs for client.interactions.create (google-genai >=2.0.0 API)."""
    # Build input parts — text first, then reference images
    parts: list[dict[str, Any]] = [{"type": "text", "text": normalized_prompt}]

    if request.references:
        for ref in request.references:
            image_bytes = ref.path.read_bytes()
            b64 = base64.b64encode(image_bytes).decode("ascii")
            parts.append({
                "type": "image",
                "data": b64,
                "mime_type": ref.mime_type,
            })

    input_data: dict[str, Any] = {
        "model": decision.resolved,
        "input": parts if len(parts) > 1 else parts[0]["text"],
    }

    # Build new-style response_format
    if request.text_output:
        input_data["response_format"] = [
            {"type": "text"},
            {"type": "image", "mime_type": request.mime_type},
        ]
    else:
        rf: dict[str, Any] = {"type": "image", "mime_type": request.mime_type}
        if request.aspect_ratio:
            rf["aspect_ratio"] = request.aspect_ratio
        if request.image_size:
            rf["image_size"] = request.image_size
        input_data["response_format"] = rf

    if request.grounding:
        input_data["tools"] = [{"google_search": {}}]

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
    """Check steps and status for safety blocks in v2 API."""
    if response is None:
        return False
    # Check interaction status
    status = _get(response, "status")
    if status is not None and str(status).upper() in {"SAFETY", "BLOCKED"}:
        return True
    # Check steps for safety-related finish reasons
    steps = _get(response, "steps", []) or []
    for step in steps:
        finish = _get(step, "finish_reason")
        if finish is not None and str(finish).upper() in {"SAFETY", "BLOCKED"}:
            return True
    return False


# =============================================================================
# Execution
# =============================================================================


def execute_request(client, input_data: dict) -> object:
    """Call client.interactions.create with retry."""
    return with_retry(lambda: client.interactions.create(**input_data))
