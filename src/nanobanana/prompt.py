from __future__ import annotations

import re

from nanobanana.types import ImageRequest, ReferenceImage

# =============================================================================
# Prompt Assembly
# =============================================================================


def build_normalized_prompt(
    request: ImageRequest,
    negative: str | None = None,
    preset: dict | None = None,
) -> str:
    """Build a structured, auditable prompt from the request and constraints."""
    sections: list[tuple[str, str]] = []
    prefix = (preset or {}).get("prompt_prefix", "").strip()
    user_prompt = request.prompt.strip()

    if prefix:
        sections.append(("TASK", prefix))
        sections.append(("CONTEXT", user_prompt))
    else:
        sections.append(("TASK", user_prompt))

    requirements: list[str] = []
    if request.aspect_ratio:
        requirements.append(f"Aspect ratio: {request.aspect_ratio}")
    if request.image_size:
        requirements.append(f"Image size: {request.image_size}")
    if request.quality:
        requirements.append(f"Quality: {request.quality}")
    if request.thinking_level and request.thinking_level != "auto":
        requirements.append(f"Thinking level: {request.thinking_level}")
    if request.grounding:
        requirements.append("Grounded with web search")
    if request.text_output:
        requirements.append("Include text output / reasoning")
    if request.count > 1:
        requirements.append(f"Generate {request.count} images")
    if request.mime_type:
        requirements.append(f"Output format: {request.mime_type}")
    if requirements:
        sections.append(("REQUIREMENTS", "\n".join(f"- {item}" for item in requirements)))

    if request.references:
        sections.append(("REFERENCE ROLES", annotate_references(request.references)))

    if request.command in {"edit", "iterate", "compose"}:
        preserve_msg = (
            "Preserve original subject identity, structure, and any explicitly "
            "requested elements unless the task requires changing them."
        )
        sections.append(("PRESERVE", preserve_msg))

    avoid_parts: list[str] = []
    negative_bullets = expand_negative(negative) if negative else ""
    if negative_bullets:
        avoid_parts.append(negative_bullets)
    avoid_parts.extend(
        [
            "- Illegible labels or text",
            "- Generic or off-brand visual clichés",
        ]
    )
    sections.append(("AVOID", "\n".join(avoid_parts)))

    output_intent = (
        "Produce a single, self-contained image that fulfills the task and requirements."
    )
    if request.text_output:
        output_intent += " Include a brief text explanation alongside the image."
    sections.append(("OUTPUT INTENT", output_intent))

    return "\n\n".join(f"{label}:\n{content}" for label, content in sections)


def expand_negative(negative: str) -> str:
    """Split negative prompt on commas/semicolons and format as bullet points."""
    if not negative:
        return ""
    parts = re.split(r"[,;]+", negative)
    items = [p.strip() for p in parts if p.strip()]
    return "\n".join(f"- {item}" for item in items)


def annotate_references(references: tuple[ReferenceImage, ...]) -> str:
    """Return per-role instructions for each reference image."""
    if not references:
        return ""
    lines = []
    for idx, ref in enumerate(references, 1):
        lines.append(f"Reference {idx} ({ref.role}): use this image as the {ref.role} reference.")
    return "\n".join(lines)


def build_edit_instruction(
    instruction: str,
    preserve: str | None = None,
    change: str | None = None,
    mask: str | None = None,
    strict_preservation: bool = False,
) -> str:
    """Build a full edit instruction with preservation and change sections."""
    sections: list[str] = [instruction.strip()]
    if preserve:
        sections.append(f"PRESERVE:\n- {preserve.strip()}")
    if change:
        sections.append(f"CHANGE:\n- {change.strip()}")
    if mask:
        mask_lower = mask.lower().strip()
        if mask_lower == "semantic":
            sections.append(
                "MASK: apply changes only to the semantically described regions; "
                "preserve everything else.",
            )
        elif mask_lower != "none":
            sections.append(
                "MASK: use the provided mask image to guide the edit. "
                "Pixel-accurate masking is not guaranteed in this version.",
            )
    if strict_preservation:
        sections.append(
            "STRICT PRESERVATION: preserve all original identity, structure, "
            "proportions, and details unless explicitly listed in CHANGE.",
        )
    return "\n\n".join(sections)
