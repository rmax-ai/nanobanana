from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, cast

import typer

from nanobanana.constants import PRESETS, ExitCode
from nanobanana.types import ReferenceImage


def vlog(**kwargs: object) -> None:
    """Emit a structured verbose log line to stderr."""
    entry: dict[str, str] = {"timestamp": datetime.now(timezone.utc).isoformat()}
    for k, v in kwargs.items():
        entry[k] = str(v)
    print(json.dumps(entry, default=str), file=sys.stderr, flush=True)


def fail(code: int, message: str) -> NoReturn:
    """Print error to stderr and raise typed exit."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=code)


def resolve_api_key(api_key: str | None, config_path: Path | None = None) -> str:
    """Resolve API key: --api-key → GEMINI_API_KEY → config → fail."""
    if api_key:
        return api_key
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    if config_path and config_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        config = tomllib.loads(config_path.read_text())
        cfg_key = config.get("api_key")
        if cfg_key:
            return cfg_key
    fail(ExitCode.AUTH_FAILURE, "No API key found. Set GEMINI_API_KEY or pass --api-key.")


def load_config(config_path: Path | None = None) -> dict:
    """Load optional config from ~/.config/nanobanana/config.toml."""
    if config_path is None:
        config_path = Path.home() / ".config" / "nanobanana" / "config.toml"
    if not config_path.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore
    return tomllib.loads(config_path.read_text())


def detect_mime_type(path: Path) -> str:
    """Detect image MIME type from magic bytes."""
    if not path.exists():
        fail(ExitCode.INPUT_FAILURE, f"Input file not found: {path}")
    header = path.read_bytes()[:8]
    if header.startswith(b"\x89PNG"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    fail(
        ExitCode.INPUT_FAILURE,
        f"Unsupported image format for '{path}'. Expected PNG or JPEG magic bytes.",
    )


def slugify(text: str, max_len: int = 64) -> str:
    """Create a short, safe filename slug from roughly the first three words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(words[:3]) if words else "image"
    slug = re.sub(r"-+", "-", slug).strip("-")
    slug = slug[:max_len].strip("-")
    return slug or "image"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def expand_preset(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a preset and merge non-None override values."""
    preset = PRESETS.get(name)
    if preset is None:
        available = ", ".join(sorted(PRESETS))
        fail(
            ExitCode.INVALID_ARGS,
            f"Unknown preset '{name}'. Available presets: {available}",
        )

    expanded = dict(preset)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                expanded[key] = value
    return expanded


def load_reference(path: Path, role: str) -> ReferenceImage:
    """Load one local reference image and compute reproducibility metadata."""
    if not path.exists():
        fail(ExitCode.INPUT_FAILURE, f"Reference file not found: {path}")
    if not path.is_file():
        fail(ExitCode.INPUT_FAILURE, f"Reference path is not a file: {path}")
    return ReferenceImage(
        path=path,
        role=role,
        mime_type=detect_mime_type(path),
        sha256=sha256_file(path),
    )


def load_references(**kwargs: Any) -> tuple[ReferenceImage, ...]:
    """Convert compose command reference flags into typed reference images."""
    references: list[ReferenceImage] = []

    ordered_roles = (
        ("subject", "subject"),
        ("object_ref", "object"),
        ("object", "object"),
        ("character", "character"),
        ("style", "style"),
        ("background", "background"),
    )
    for key, role in ordered_roles:
        value = kwargs.get(key)
        if isinstance(value, Path):
            references.append(load_reference(value, role=role))

    generic = kwargs.get("reference")
    if isinstance(generic, list):
        for item in generic:
            if isinstance(item, Path):
                references.append(load_reference(item, role="reference"))

    return tuple(references)


def _get(obj: object, key: str, default: Any = None) -> Any:
    """Get a value from a dict or object by attribute/key name."""
    if isinstance(obj, dict):
        mapping = cast(dict[str, Any], obj)
        return mapping.get(key, default)
    return getattr(obj, key, default)
