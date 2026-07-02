from __future__ import annotations

import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

try:
    from google import genai
except ImportError:
    genai = None

from nanobanana.client import (
    build_interaction_input,
    classify_api_exception,
    execute_request,
    genai,
)
from nanobanana.constants import MIME_MAP, ExitCode
from nanobanana.model_selection import estimate_cost, select_model, validate_request
from nanobanana.output import (
    _resolve_output_path_for_image,
    atomic_write,
    console,
    emit_json,
    print_dry_run,
    write_manifest,
)
from nanobanana.prompt import build_normalized_prompt
from nanobanana.response import (
    classify_response_error,
    extract_grounding_metadata,
    extract_images,
)
from nanobanana.types import (
    GeneratedAsset,
    ImageRequest,
    ModelDecision,
    ReferenceImage,
)
from nanobanana.utils import expand_preset, fail, resolve_api_key, sha256_file, vlog

# =============================================================================
# Pipeline
# =============================================================================


def run_generate_pipeline(
    ctx: typer.Context,
    prompt: str | None,
    *,
    prompt_file: Path | None = None,
    negative: str | None = None,
    aspect: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    thinking: str | None = None,
    count: int = 1,
    output: Path | None = None,
    text_output: bool = False,
    ground: bool = False,
    retry_with_pro: bool = False,
    preset_name: str | None = None,
    max_estimated_cost: float | None = None,
    show_estimate: bool = False,
    references: tuple[ReferenceImage, ...] = (),
    allow_degraded: bool = False,
    warnings: list[str] | None = None,
    command: str = "generate",
) -> None:
    """Execute the full generate pipeline from prompt to manifest."""
    # 1. Read prompt from file if provided, else use the positional argument.
    if prompt_file is not None:
        try:
            prompt_text = prompt_file.read_text()
        except Exception as exc:
            fail(ExitCode.INPUT_FAILURE, f"Failed to read prompt file: {exc}")
    elif prompt:
        prompt_text = prompt
    else:
        fail(
            ExitCode.INVALID_ARGS,
            "Provide a prompt argument or --prompt-file.",
        )

    # 2. Load preset if requested; CLI flags override preset values.
    preset: dict[str, Any] | None = None
    model_alias = ctx.obj.get("model", "auto")
    if preset_name is not None:
        overrides = {
            "model": model_alias if model_alias != "auto" else None,
            "aspect": aspect,
            "size": size,
            "quality": quality,
            "thinking": thinking,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        preset = expand_preset(preset_name, overrides)
        if model_alias == "auto" and "model" in preset:
            model_alias = preset["model"]
        if aspect is None and "aspect" in preset:
            aspect = preset["aspect"]
        if size is None and "size" in preset:
            size = preset["size"]
        if quality is None and "quality" in preset:
            quality = preset["quality"]
        if thinking is None and "thinking" in preset:
            thinking = preset["thinking"]

    # Apply defaults for any values still unset.
    if aspect is None:
        aspect = "1:1"
    if size is None:
        size = "1K"
    if quality is None:
        quality = "balanced"
    if thinking is None:
        thinking = "auto"

    fmt = ctx.obj.get("format", "png")
    request_id = ctx.obj.get("request_id") or str(uuid.uuid4())

    # 3. Build the typed request object.
    request = ImageRequest(
        command=command,
        prompt=prompt_text,
        model=model_alias,
        references=references,
        aspect_ratio=aspect,
        image_size=size,
        mime_type=MIME_MAP.get(fmt, "image/png"),
        thinking_level=thinking,
        grounding="web" if ground else None,
        seed=ctx.obj.get("seed"),
        request_id=request_id,
        quality=quality,
        count=count,
        text_output=text_output,
        preset_name=preset_name,
    )

    vlog(
        event="request_built",
        run_id=request.request_id,
        command=request.command,
        model_alias=request.model,
        aspect_ratio=request.aspect_ratio,
        image_size=request.image_size,
        count=request.count,
    )

    # 4. Select the concrete model and 5. validate against capabilities.
    decision = select_model(request)
    vlog(
        event="model_selected",
        run_id=request.request_id,
        model=decision.resolved,
        reason=decision.selection_reason,
    )
    capability_warnings = validate_request(request, allow_degraded=allow_degraded)
    if warnings is None:
        warnings = []
    warnings.extend(capability_warnings)

    # 6. Cost ceiling / estimate display.
    cost = estimate_cost(decision.resolved, request.image_size or "1K", request.count)
    if show_estimate or max_estimated_cost is not None:
        if cost is None:
            fail(
                ExitCode.INVALID_ARGS,
                "Cannot estimate cost for the selected model/size/count.",
            )
        if show_estimate:
            if ctx.obj.get("json"):
                emit_json(
                    status="estimate",
                    run_id=request.request_id,
                    outputs=[{"estimated_cost_usd": cost}],
                )
            else:
                typer.echo(f"Estimated cost: ${cost:.4f}")
            return
        if max_estimated_cost is not None and cost > max_estimated_cost:
            fail(
                ExitCode.INVALID_ARGS,
                f"Estimated cost ${cost:.4f} exceeds maximum ${max_estimated_cost:.4f}.",
            )

    # 7. Assemble the structured, auditable prompt.
    normalized_prompt = build_normalized_prompt(request, negative=negative, preset=preset)

    # 8. Dry-run stops before any API call or key resolution.
    if ctx.obj.get("dry_run"):
        output_path = _resolve_output_path_for_image(
            request, output, ctx.obj.get("output_dir"), fmt, 1, count
        )
        if ctx.obj.get("json") or not ctx.obj.get("quiet"):
            print_dry_run(
                request,
                decision,
                normalized_prompt,
                output_path,
                is_json=ctx.obj.get("json", False),
            )
        return

    # 9. Resolve API key and create the SDK client.
    if genai is None:
        fail(ExitCode.INTERNAL, "google-genai SDK is not installed.")
    api_key = resolve_api_key(ctx.obj.get("api_key"), ctx.obj.get("config"))
    client = genai.Client(api_key=api_key)

    # 10. Build interaction input and execute with retries.
    def _execute(req: ImageRequest, dec: ModelDecision) -> object:
        input_data = build_interaction_input(req, normalized_prompt, dec)
        vlog(
            event="api_call_start",
            run_id=req.request_id,
            model=dec.resolved,
        )
        start = time.time()
        try:
            response = execute_request(client, input_data)
        except Exception as exc:
            vlog(
                event="api_call_error",
                run_id=req.request_id,
                model=dec.resolved,
                elapsed_ms=round((time.time() - start) * 1000),
                error=type(exc).__name__,
            )
            raise
        vlog(
            event="api_call_end",
            run_id=req.request_id,
            model=dec.resolved,
            elapsed_ms=round((time.time() - start) * 1000),
        )
        return response

    response: object | None = None
    api_error: Exception | None = None
    try:
        response = _execute(request, decision)
    except Exception as exc:
        api_error = exc

    response_error = classify_response_error(response) if response is not None else None

    # 16. Optional escalation to Pro when Flash failed.
    if (
        (api_error is not None or response_error is not None)
        and retry_with_pro
        and decision.resolved == "gemini-3.1-flash-image"
    ):
        vlog(event="flash_failed_escalating_to_pro", run_id=request.request_id)
        pro_decision = ModelDecision(
            requested="pro",
            resolved="gemini-3-pro-image",
            selection_reason="retry_with_pro escalation after Flash failure",
        )
        pro_request = replace(request, model="pro", request_id=str(uuid.uuid4()))
        try:
            validate_request(pro_request, allow_degraded=allow_degraded)
            response = _execute(pro_request, pro_decision)
            response_error = classify_response_error(response)
            if response_error is None:
                decision = pro_decision
                request = pro_request
                api_error = None
        except Exception as pro_exc:
            fail(ExitCode.INTERNAL, f"Pro retry failed: {pro_exc}")

    # 11. Surface any remaining API or response error with the right exit code.
    if api_error is not None:
        fail(classify_api_exception(api_error), str(api_error))
    if response_error is not None:
        fail(response_error, "API returned an error response")

    # 12. Extract images and handle an empty response.
    images = extract_images(response)
    if not images:
        fail(ExitCode.EMPTY_RESPONSE, "No images returned in the API response.")

    # 13. Write each image atomically with SHA-256.
    assets: list[GeneratedAsset] = []
    output_dir = ctx.obj.get("output_dir")
    overwrite = ctx.obj.get("overwrite", False)
    for i, image_bytes in enumerate(images, 1):
        output_path = _resolve_output_path_for_image(
            request, output, output_dir, fmt, i, len(images)
        )
        try:
            atomic_write(output_path, image_bytes, overwrite=overwrite)
        except Exception as exc:
            fail(ExitCode.OUTPUT_FAILURE, f"Failed to write output: {exc}")
        sha = sha256_file(output_path)
        assets.append(GeneratedAsset(path=output_path, mime_type=request.mime_type, sha256=sha))
        vlog(
            event="output_written",
            run_id=request.request_id,
            path=str(output_path),
            sha256=sha,
        )

    # 14. Persist the manifest.
    grounding_metadata = extract_grounding_metadata(response)
    manifest_path = write_manifest(
        request,
        decision,
        normalized_prompt,
        assets,
        output_dir,
        grounding_metadata=grounding_metadata,
        warnings=warnings,
    )

    # 15. Emit machine-readable or human-readable status.
    if ctx.obj.get("json"):
        emit_json(
            status="success",
            run_id=request.request_id,
            outputs=[
                {
                    "path": str(asset.path),
                    "sha256": asset.sha256,
                    "mime_type": asset.mime_type,
                }
                for asset in assets
            ],
            manifest=manifest_path,
        )
    elif not ctx.obj.get("quiet"):
        table = Table(title="Generated")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Model", decision.resolved)
        table.add_row("Reason", decision.selection_reason)
        table.add_row("Files", ", ".join(str(asset.path) for asset in assets))
        table.add_row("Manifest", str(manifest_path))
        console.print(table)
