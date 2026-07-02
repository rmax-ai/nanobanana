from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from nanobanana.model_selection import estimate_cost
from nanobanana.types import GeneratedAsset, ImageRequest, ModelDecision
from nanobanana.utils import slugify

console = Console(stderr=True)


# =============================================================================
# Output Layer
# =============================================================================


def atomic_write(path: Path, data: bytes, overwrite: bool = False) -> Path:
    """Write data to path atomically via a tempfile and rename."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        shutil.move(tmp, path)
        return path
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise


def generate_filename(
    slug: str, index: int = 1, extension: str = "png", max_slug_len: int = 64
) -> str:
    """Build a safe, timestamped filename for a generated asset."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^a-z0-9_-]", "-", slug.lower())
    safe = re.sub(r"-+", "-", safe).strip("-")
    safe = safe[:max_slug_len].strip("-")
    if not safe:
        safe = "image"
    return f"{timestamp}-{safe}-{index:02d}.{extension}"


def write_manifest(
    request: ImageRequest,
    decision: ModelDecision,
    normalized_prompt: str,
    assets: list[GeneratedAsset],
    output_dir: Path | None,
    **kwargs: Any,
) -> Path:
    """Serialize a v1.0 manifest next to the first output asset."""
    if assets:
        manifest_path = assets[0].path.parent / (assets[0].path.name + ".manifest.json")
    elif output_dir:
        manifest_path = output_dir / f"{request.request_id}.manifest.json"
    else:
        manifest_path = Path.cwd() / f"{request.request_id}.manifest.json"

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": request.request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": request.command,
        "model": {
            "requested": decision.requested,
            "resolved": decision.resolved,
            "selection_reason": decision.selection_reason,
        },
        "input": {
            "prompt": request.prompt,
            "references": [
                {
                    "path": str(ref.path),
                    "role": ref.role,
                    "mime_type": ref.mime_type,
                    "sha256": ref.sha256,
                }
                for ref in request.references
            ],
        },
        "response_format": {
            "mime_type": request.mime_type,
            "text_output": request.text_output,
        },
        "generation": {
            "normalized_prompt": normalized_prompt,
            "aspect_ratio": request.aspect_ratio,
            "image_size": request.image_size,
            "quality": request.quality,
            "thinking_level": request.thinking_level,
            "grounding": request.grounding,
            "seed": request.seed,
            "count": request.count,
            "preset": request.preset_name,
        },
        "outputs": [
            {
                "path": str(asset.path),
                "mime_type": asset.mime_type,
                "sha256": asset.sha256,
            }
            for asset in assets
        ],
    }

    cost = estimate_cost(
        decision.resolved,
        request.image_size or "1K",
        request.count,
    )
    if cost is not None:
        manifest["usage"] = {
            "estimated_cost_usd": cost,
            "image_count": request.count,
        }

    grounding = kwargs.get("grounding_metadata")
    if grounding:
        manifest["sources"] = grounding

    warnings = kwargs.get("warnings")
    if warnings:
        manifest["warnings"] = list(warnings)

    atomic_write(manifest_path, json.dumps(manifest, indent=2).encode("utf-8"))
    return manifest_path


def resolve_output_path(
    request: ImageRequest,
    specified: Path | None,
    output_dir: Path | None,
    fmt: str,
) -> Path:
    """Choose an output path, using a generated filename when none is given."""
    if specified:
        return specified
    filename = generate_filename(slugify(request.prompt), extension=fmt)
    if output_dir:
        return output_dir / filename
    return Path.cwd() / filename


def emit_json(
    status: str,
    run_id: str,
    outputs: list[dict[str, Any]] | None = None,
    manifest: Path | None = None,
    error: str | None = None,
    exit_code: int | None = None,
) -> None:
    """Print a single JSON object to stdout for machine-readable output."""
    payload: dict[str, Any] = {"status": status, "run_id": run_id}
    if outputs is not None:
        payload["outputs"] = outputs
    if manifest is not None:
        payload["manifest"] = str(manifest)
    if error is not None:
        payload["error"] = error
    if exit_code is not None:
        payload["exit_code"] = exit_code
    print(json.dumps(payload, indent=2, default=str), file=sys.stdout, flush=True)


def print_dry_run(
    request: ImageRequest,
    decision: ModelDecision,
    normalized_prompt: str,
    output_path: Path,
    is_json: bool = False,
) -> None:
    """Print a dry-run summary as a rich table or JSON object."""
    if is_json:
        emit_json(
            status="dry-run",
            run_id=request.request_id,
            outputs=[
                {
                    "path": str(output_path),
                    "count": request.count,
                    "format": request.mime_type,
                }
            ],
        )
        return

    cost = estimate_cost(
        decision.resolved,
        request.image_size or "1K",
        request.count,
    )
    table = Table(title="Dry Run")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", decision.resolved)
    table.add_row("Reason", decision.selection_reason)
    table.add_row("Prompt", normalized_prompt)
    table.add_row("Files", f"{output_path} (count: {request.count})")
    table.add_row("Format", request.mime_type)
    table.add_row("Cost", f"${cost:.4f}" if cost is not None else "N/A")
    console.print(table)


# =============================================================================
# Generate Command Helpers
# =============================================================================


def _resolve_output_path_for_image(
    request: ImageRequest,
    specified: Path | None,
    output_dir: Path | None,
    fmt: str,
    index: int,
    count: int,
) -> Path:
    """Choose an output path for a single generated image, supporting multi-output runs."""
    if specified:
        if count == 1:
            return specified
        return specified.parent / f"{specified.stem}-{index:02d}{specified.suffix}"
    return resolve_output_path(request, None, output_dir, fmt)
