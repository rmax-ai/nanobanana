from __future__ import annotations

import base64
from typing import Any

from nanobanana.constants import ExitCode
from nanobanana.utils import _get


# =============================================================================
# Response Extraction (google-genai >=2.0.0)
# =============================================================================


def extract_images(response: object) -> list[bytes]:
    """Decode images from interaction response (v2 API).

    Uses convenience property output_image for simple cases, falls back to
    iterating steps for multi-image or interleaved responses.
    """
    images: list[bytes] = []

    # Try convenience property first
    output_image = _get(response, "output_image")
    if output_image is not None:
        data = _get(output_image, "data")
        if data:
            images.append(base64.b64decode(data))
            return images

    # Fall back to iterating steps
    steps = _get(response, "steps", []) or []
    for step in steps:
        step_type = _get(step, "type", "")
        if step_type != "model_output":
            continue
        content = _get(step, "content", []) or []
        for item in content:
            item_type = _get(item, "type", "")
            if item_type == "image":
                data = _get(item, "data")
                if data:
                    images.append(base64.b64decode(data))

    return images


def extract_grounding_metadata(response: object) -> dict | None:
    """Extract grounding sources, citations, and suggestions from v2 response."""
    metadata: dict[str, list[Any]] = {
        "sources": [],
        "citations": [],
        "search_suggestions": [],
    }

    steps = _get(response, "steps", []) or []
    for step in steps:
        grounding = _get(step, "grounding_metadata")
        if grounding is None:
            continue
        sources = _get(grounding, "sources", _get(grounding, "grounding_chunks", []))
        citations = _get(
            grounding, "citations", _get(grounding, "grounding_supports", [])
        )
        suggestions = _get(
            grounding, "suggestions", _get(grounding, "search_entry_point", [])
        )
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

    # Check for safety refusals via client module
    from nanobanana.client import is_safety_refusal

    if is_safety_refusal(response):
        return ExitCode.SAFETY_REFUSAL

    images = extract_images(response)
    if images and any(images):
        return None
    return ExitCode.EMPTY_RESPONSE
