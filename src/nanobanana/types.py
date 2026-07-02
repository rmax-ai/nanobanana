from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ImageRequest:
    command: str
    prompt: str
    model: str
    references: tuple[ReferenceImage, ...] = ()
    aspect_ratio: str | None = None
    image_size: str | None = None
    mime_type: str = "image/png"
    thinking_level: str | None = None
    grounding: str | None = None
    seed: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    quality: str | None = None
    count: int = 1
    text_output: bool = False
    preset_name: str | None = None


@dataclass(frozen=True)
class ReferenceImage:
    path: Path
    role: str
    mime_type: str
    sha256: str


@dataclass(frozen=True)
class GeneratedAsset:
    path: Path
    mime_type: str
    sha256: str


@dataclass(frozen=True)
class ModelDecision:
    requested: str
    resolved: str
    selection_reason: str
