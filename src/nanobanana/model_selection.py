from __future__ import annotations

import re
from typing import Any

from nanobanana.constants import (
    ALIAS_TO_MODEL,
    CAPABILITIES,
    COST_TABLE,
    FLASH_ONLY_ASPECT_RATIOS,
    SUPPORTED_ASPECT_RATIOS,
    ExitCode,
)
from nanobanana.types import ImageRequest, ModelDecision
from nanobanana.utils import fail


def resolve_model_alias(alias: str) -> str:
    """Map CLI alias to full API model name."""
    if alias == "auto":
        return "auto"
    if alias in CAPABILITIES:
        return alias
    if alias in ALIAS_TO_MODEL:
        return ALIAS_TO_MODEL[alias]
    fail(
        ExitCode.INVALID_ARGS,
        f"Unknown model '{alias}'. Use one of: auto, {', '.join(sorted(ALIAS_TO_MODEL))}.",
    )


def get_capability(model: str, key: str, default: Any = None) -> Any:
    """Safely read a model capability."""
    resolved = model if model in CAPABILITIES else ALIAS_TO_MODEL.get(model, "")
    return CAPABILITIES.get(resolved, {}).get(key, default)


def supports(model: str, feature: str) -> bool:
    """Return True when a capability is explicitly enabled."""
    return bool(get_capability(model, feature, False))


def _contains_explicit_text_request(prompt: str) -> bool:
    """Heuristic for prompts asking for exact rendered text."""
    return bool(
        re.search(
            r"\b(text|label|labels|caption|headline|title|word|words|spelled|exact)\b",
            prompt.lower(),
        ),
    )


def _should_use_lite(request: ImageRequest) -> bool:
    """Return True when the request matches the low-cost Lite envelope."""
    if request.image_size != "1K":
        return False
    if request.grounding:
        return False
    if len(request.references) > 1:
        return False

    command = request.command.lower()
    if command == "generate":
        return request.quality == "draft" and not request.text_output

    lite_commands = {
        "thumbnail",
        "variation",
        "variations",
        "background_adjustment",
        "sticker",
        "draft",
        "batch_preview",
    }
    return command in lite_commands


def _should_use_pro(request: ImageRequest) -> bool:
    """Return True when quality or complexity warrants Pro."""
    if request.quality == "professional":
        return True
    if request.command.lower() in {"diagram", "infographic", "product", "layout"}:
        return True
    return len(request.prompt) > 100 and _contains_explicit_text_request(request.prompt)


def select_model(request: ImageRequest) -> ModelDecision:
    """Resolve auto policy to a concrete API model."""
    if request.model != "auto":
        resolved = resolve_model_alias(request.model)
        return ModelDecision(
            requested=request.model,
            resolved=resolved,
            selection_reason="explicitly requested model",
        )

    if _should_use_lite(request):
        return ModelDecision(
            requested="auto",
            resolved="gemini-3.1-flash-lite-image",
            selection_reason="matched low-cost Lite policy envelope",
        )

    if _should_use_pro(request):
        return ModelDecision(
            requested="auto",
            resolved="gemini-3-pro-image",
            selection_reason="matched high-fidelity Pro policy envelope",
        )

    return ModelDecision(
        requested="auto",
        resolved="gemini-3.1-flash-image",
        selection_reason="default general-purpose model",
    )


def validate_request(request: ImageRequest, allow_degraded: bool = False) -> list[str]:
    """Validate request against model capabilities before API calls.

    Returns a list of capability warnings. When allow_degraded is False, issues
    raise an exception instead of being returned.
    """
    if request.model == "auto":
        resolved_model = select_model(request).resolved
    else:
        resolved_model = resolve_model_alias(request.model)
    caps = CAPABILITIES.get(resolved_model)
    if not caps:
        fail(
            ExitCode.CAPABILITY_MISMATCH,
            f"Model '{resolved_model}' is not registered in capabilities.",
        )
    assert caps is not None

    issues: list[str] = []

    if request.image_size:
        supported_sizes = caps.get("supported_sizes", ())
        if request.image_size not in supported_sizes:
            issues.append(
                f"size '{request.image_size}' is unsupported for model '{caps['alias']}' "
                f"(supported: {', '.join(supported_sizes)})",
            )

    if request.aspect_ratio:
        all_ratios = set(SUPPORTED_ASPECT_RATIOS) | set(FLASH_ONLY_ASPECT_RATIOS)
        if request.aspect_ratio not in all_ratios:
            issues.append(
                f"aspect ratio '{request.aspect_ratio}' is invalid "
                f"(supported: {', '.join(sorted(all_ratios))})",
            )
        elif (
            request.aspect_ratio in FLASH_ONLY_ASPECT_RATIOS
            and resolved_model != ALIAS_TO_MODEL["flash"]
        ):
            issues.append(
                f"aspect ratio '{request.aspect_ratio}' is Flash-only, "
                "use model 'flash' or change aspect ratio",
            )

    if request.grounding and not supports(resolved_model, "grounding"):
        issues.append(f"grounding is unsupported for model '{caps['alias']}'")

    if request.thinking_level and request.thinking_level != "auto":
        allowed_thinking = tuple(caps.get("thinking_levels", ()))
        if not allowed_thinking:
            issues.append(
                f"thinking controls are unavailable for model '{caps['alias']}', remove --thinking",
            )
        elif request.thinking_level not in allowed_thinking:
            issues.append(
                f"thinking level '{request.thinking_level}' is unsupported "
                f"for model '{caps['alias']}' "
                f"(supported: {', '.join(allowed_thinking)})",
            )

    max_refs = int(caps.get("max_references", 0))
    if len(request.references) > max_refs:
        issues.append(
            f"{len(request.references)} references exceed model "
            f"'{caps['alias']}' limit of {max_refs}",
        )

    if issues and not allow_degraded:
        fail(
            ExitCode.CAPABILITY_MISMATCH,
            "Capability mismatch: " + "; ".join(issues),
        )
    return issues


def estimate_cost(model: str, size: str, count: int = 1) -> float | None:
    """Estimate total request cost in USD from the static table."""
    resolved_model = resolve_model_alias(model)
    if resolved_model == "auto":
        return None
    unit_cost = COST_TABLE.get(resolved_model, {}).get(size)
    if unit_cost is None:
        return None
    return unit_cost * max(count, 0)
