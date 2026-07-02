from __future__ import annotations

import base64
from typing import Any

from nanobanana.client import is_safety_refusal
from nanobanana.constants import ExitCode
from nanobanana.utils import _get

# =============================================================================
# Response Extraction
# =============================================================================


def extract_images(response: object) -> list[bytes]:
    """Decode base64 inline_data from response candidates."""
    images: list[bytes] = []
    candidates = _get(response, "candidates", []) or []
    for candidate in candidates:
        content = _get(candidate, "content")
        if content is None:
            continue
        parts = _get(content, "parts", []) or []
        for part in parts:
            if isinstance(part, dict):
                inline_data = part.get("inline_data")
                if isinstance(inline_data, dict) and inline_data.get("data"):
                    images.append(base64.b64decode(inline_data["data"]))
            else:
                inline_data = _get(part, "inline_data")
                if inline_data is not None:
                    data = _get(inline_data, "data")
                    if data:
                        images.append(base64.b64decode(data))
    return images


def extract_grounding_metadata(response: object) -> dict | None:
    """Extract grounding sources, citations, and suggestions."""
    metadata: dict[str, list[Any]] = {
        "sources": [],
        "citations": [],
        "search_suggestions": [],
    }
    candidates = _get(response, "candidates", []) or []
    for candidate in candidates:
        grounding = _get(candidate, "grounding_metadata")
        if grounding is None:
            continue
        sources = _get(grounding, "sources", _get(grounding, "grounding_chunks", []))
        citations = _get(grounding, "citations", _get(grounding, "grounding_supports", []))
        suggestions = _get(grounding, "suggestions", _get(grounding, "search_entry_point", []))
        metadata["sources"].extend(list(sources))
        metadata["citations"].extend(list(citations))
        metadata["search_suggestions"].extend(list(suggestions))
    if any(metadata.values()):
        return metadata
    return None


def classify_response_error(response: object) -> int | None:
    """Classify a response as a safety refusal, empty response, or valid."""
    if response is None:
        return ExitCode.EMPTY_RESPONSE
    if is_safety_refusal(response):
        return ExitCode.SAFETY_REFUSAL
    images = extract_images(response)
    if images and any(images):
        return None
    return ExitCode.EMPTY_RESPONSE
