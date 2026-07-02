from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from nanobanana.constants import CAPABILITIES, ExitCode
from nanobanana.pipeline import run_generate_pipeline
from nanobanana.prompt import build_edit_instruction
from nanobanana.types import ReferenceImage
from nanobanana.utils import fail, load_config, load_reference, load_references

# =============================================================================
# CLI Setup
# =============================================================================


console = Console(stderr=True)
app = typer.Typer(no_args_is_help=True, name="nanobanana")


# =============================================================================
# Global Options Callback
# =============================================================================


@app.callback()
def main(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model alias: auto, lite, flash, pro, legacy",
        ),
    ] = "auto",
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-d",
            help="Output directory for generated files",
            exists=False,
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Gemini API key (overrides GEMINI_API_KEY)"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to config file (default: ~/.config/nanobanana/config.toml)",
            exists=True,
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output image format",
        ),
    ] = "png",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output JSON to stdout, logs to stderr"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-error output"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Emit structured verbose logs to stderr"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print what would happen without making API calls"),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite existing output files"),
    ] = False,
    seed: Annotated[
        str | None,
        typer.Option("--seed", help="Local run identifier (not deterministic seeding)"),
    ] = None,
    request_id: Annotated[
        str | None,
        typer.Option("--request-id", help="Override the run UUID"),
    ] = None,
) -> None:
    """nanobanana — Single-File Gemini Image CLI

    Task-oriented image generation using Google's Gemini Interactions API.
    Automatic model selection between Lite, Flash, and Pro variants.
    """
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["output_dir"] = output_dir
    ctx.obj["api_key"] = api_key
    ctx.obj["config"] = config
    ctx.obj["format"] = fmt
    ctx.obj["json"] = json_output
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run
    ctx.obj["overwrite"] = overwrite
    ctx.obj["seed"] = seed
    ctx.obj["request_id"] = request_id


# =============================================================================
# Commands
# =============================================================================


@app.command()
def generate(
    ctx: typer.Context,
    prompt: Annotated[str | None, typer.Argument(help="Text prompt for image generation")] = None,
    prompt_file: Annotated[
        Path | None, typer.Option("--prompt-file", help="Read prompt from file")
    ] = None,
    negative: Annotated[
        str | None, typer.Option("--negative", help="Negative prompt constraints")
    ] = None,
    aspect: Annotated[str | None, typer.Option("--aspect", "-a", help="Aspect ratio")] = None,
    size: Annotated[
        str | None,
        typer.Option("--size", "-s", help="Output resolution: 0.5K, 1K, 2K, 4K"),
    ] = None,
    quality: Annotated[
        str | None,
        typer.Option("--quality", help="Quality tier: draft, balanced, professional"),
    ] = None,
    thinking: Annotated[
        str | None,
        typer.Option("--thinking", help="Thinking level: minimal, high, auto"),
    ] = None,
    count: Annotated[int, typer.Option("--count", "-n", help="Number of images to generate")] = 1,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file path")] = None,
    text_output: Annotated[
        bool, typer.Option("--text-output", help="Include text in response")
    ] = False,
    ground: Annotated[bool, typer.Option("--ground", help="Enable web search grounding")] = False,
    retry_with_pro: Annotated[
        bool, typer.Option("--retry-with-pro", help="Escalate to Pro on Flash failure")
    ] = False,
    preset: Annotated[str | None, typer.Option("--preset", "-p", help="Apply named preset")] = None,
    max_estimated_cost: Annotated[
        float | None, typer.Option("--max-estimated-cost", help="Cost ceiling in USD")
    ] = None,
    show_estimate: Annotated[
        bool, typer.Option("--show-estimate", help="Print cost estimate and exit")
    ] = False,
) -> None:
    """Generate an image from a text prompt."""
    run_generate_pipeline(
        ctx,
        prompt,
        prompt_file=prompt_file,
        negative=negative,
        aspect=aspect,
        size=size,
        quality=quality,
        thinking=thinking,
        count=count,
        output=output,
        text_output=text_output,
        ground=ground,
        retry_with_pro=retry_with_pro,
        preset_name=preset,
        max_estimated_cost=max_estimated_cost,
        show_estimate=show_estimate,
    )


@app.command()
def edit(
    ctx: typer.Context,
    input_image: Annotated[Path, typer.Argument(help="Input image to edit")],
    instruction: Annotated[str | None, typer.Argument(help="Editing instruction")] = None,
    instruction_file: Annotated[
        Path | None,
        typer.Option("--instruction-file", help="Read instruction from file"),
    ] = None,
    preserve: Annotated[str | None, typer.Option("--preserve", help="Elements to preserve")] = None,
    change: Annotated[str | None, typer.Option("--change", help="Elements to change")] = None,
    mask: Annotated[
        str | None,
        typer.Option("--mask", help="Mask type: semantic, none, or path to mask image"),
    ] = None,
    aspect: Annotated[str, typer.Option("--aspect", "-a", help="Aspect ratio")] = "1:1",
    size: Annotated[str, typer.Option("--size", "-s", help="Output resolution")] = "1K",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file path")] = None,
    strict_preservation: Annotated[
        bool,
        typer.Option("--strict-preservation", help="Stronger preservation constraints"),
    ] = False,
) -> None:
    """Modify an existing image while preserving selected elements."""
    if instruction_file is not None:
        try:
            instruction_text = instruction_file.read_text()
        except Exception as exc:
            fail(ExitCode.INPUT_FAILURE, f"Failed to read instruction file: {exc}")
    elif instruction:
        instruction_text = instruction
    else:
        fail(
            ExitCode.INVALID_ARGS,
            "Provide an instruction argument or --instruction-file.",
        )

    input_ref = load_reference(input_image, role="input")
    references: list[ReferenceImage] = [input_ref]
    if mask:
        mask_lower = mask.lower().strip()
        if mask_lower not in {"semantic", "none"}:
            mask_path = Path(mask)
            references.append(load_reference(mask_path, role="mask"))

    full_instruction = build_edit_instruction(
        instruction_text,
        preserve=preserve,
        change=change,
        mask=mask,
        strict_preservation=strict_preservation,
    )

    warnings: list[str] = []
    if strict_preservation:
        warnings.append(
            "Strict preservation enabled; the model may still alter fine details.",
        )

    run_generate_pipeline(
        ctx,
        full_instruction,
        aspect=aspect,
        size=size,
        output=output,
        references=tuple(references),
        warnings=warnings,
        command="edit",
    )


@app.command()
def compose(
    ctx: typer.Context,
    instruction: Annotated[str, typer.Argument(help="Composition instruction")],
    subject: Annotated[Path | None, typer.Option("--subject", help="Primary subject image")] = None,
    object_ref: Annotated[
        Path | None, typer.Option("--object", help="Object reference image")
    ] = None,
    character: Annotated[
        Path | None, typer.Option("--character", help="Character reference image")
    ] = None,
    style: Annotated[Path | None, typer.Option("--style", help="Style reference image")] = None,
    background: Annotated[
        Path | None, typer.Option("--background", help="Background reference image")
    ] = None,
    reference: Annotated[
        list[Path] | None,
        typer.Option("--reference", help="Generic reference image (repeatable)"),
    ] = None,
    aspect: Annotated[str, typer.Option("--aspect", "-a", help="Aspect ratio")] = "1:1",
    size: Annotated[str, typer.Option("--size", "-s", help="Output resolution")] = "1K",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output file path")] = None,
    allow_degraded: Annotated[
        bool,
        typer.Option(
            "--allow-degraded",
            help="Allow degraded quality for unsupported combinations",
        ),
    ] = False,
) -> None:
    """Combine multiple reference images into a coherent output."""
    references = load_references(
        subject=subject,
        object_ref=object_ref,
        character=character,
        style=style,
        background=background,
        reference=reference,
    )

    if not references:
        fail(
            ExitCode.INVALID_ARGS,
            "Provide at least one reference image (--subject, --object, --character, "
            "--style, --background, or --reference).",
        )

    run_generate_pipeline(
        ctx,
        instruction,
        aspect=aspect,
        size=size,
        output=output,
        references=references,
        allow_degraded=allow_degraded,
        command="compose",
    )


@app.command()
def diagram(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="Diagram description")],
    type_: Annotated[
        str,
        typer.Option("--type", help="Diagram type: architecture, sequence, flowchart, etc."),
    ] = "architecture",
    labels_file: Annotated[
        Path | None, typer.Option("--labels-file", help="File with terminology")
    ] = None,
    size: Annotated[str, typer.Option("--size", "-s", help="Output resolution")] = "2K",
    text_plan: Annotated[
        str, typer.Option("--text-plan", help="Two-phase text planning: auto, on, off")
    ] = "auto",
    validate_text: Annotated[
        bool, typer.Option("--validate-text", help="OCR-free text validation")
    ] = False,
) -> None:
    """Produce diagrams, infographics, or explanatory visuals."""
    fail(ExitCode.INTERNAL, "diagram: not yet implemented")


@app.command()
def product(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="Product shot description")],
    subject: Annotated[Path | None, typer.Option("--subject", help="Subject image")] = None,
    logo: Annotated[Path | None, typer.Option("--logo", help="Logo image")] = None,
    packaging: Annotated[
        Path | None, typer.Option("--packaging", help="Packaging reference")
    ] = None,
    bg: Annotated[Path | None, typer.Option("--background", help="Background image")] = None,
    view: Annotated[str | None, typer.Option("--view", help="Camera view angle")] = None,
    lighting: Annotated[str | None, typer.Option("--lighting", help="Lighting style")] = None,
    surface: Annotated[str | None, typer.Option("--surface", help="Surface material")] = None,
    brand_colors: Annotated[
        str | None, typer.Option("--brand-colors", help="Brand color palette")
    ] = None,
    copy_text: Annotated[str | None, typer.Option("--copy", help="Copy text overlay")] = None,
    preserve_geometry: Annotated[
        bool,
        typer.Option("--preserve-geometry", help="Preserve exact proportions and controls"),
    ] = False,
) -> None:
    """Create product shots and mockups."""
    fail(ExitCode.INTERNAL, "product: not yet implemented")


@app.command()
def grounded(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="Grounded generation prompt")],
    aspect: Annotated[str, typer.Option("--aspect", "-a", help="Aspect ratio")] = "16:9",
    search: Annotated[
        str, typer.Option("--search", help="Search type: web, web-and-images")
    ] = "web",
    citations: Annotated[
        Path | None, typer.Option("--citations", help="Path for citation output")
    ] = None,
    show_sources: Annotated[
        bool, typer.Option("--show-sources", help="Display source attribution")
    ] = False,
    freshness: Annotated[
        str | None, typer.Option("--freshness", help="Search freshness constraint")
    ] = None,
) -> None:
    """Generate a visual using current web information."""
    fail(ExitCode.INTERNAL, "grounded: not yet implemented")


@app.command()
def variations(
    ctx: typer.Context,
    source: Annotated[Path | None, typer.Argument(help="Source image for variations")] = None,
    prompt: Annotated[
        str | None, typer.Argument(help="Variation prompt (if no source image)")
    ] = None,
    instruction: Annotated[
        str | None, typer.Option("--instruction", help="Variation direction")
    ] = None,
    count: Annotated[int, typer.Option("--count", "-n", help="Number of variations")] = 4,
    parallel: Annotated[int, typer.Option("--parallel", help="Parallel generations")] = 1,
    naming_template: Annotated[
        str | None, typer.Option("--naming-template", help="Filename template")
    ] = None,
    contact_sheet: Annotated[
        bool, typer.Option("--contact-sheet", help="Generate contact sheet")
    ] = False,
    select_best: Annotated[
        bool,
        typer.Option("--select-best", help="Select best via multimodal evaluation"),
    ] = False,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Model for best-selection")
    ] = None,
) -> None:
    """Produce prompt or image variations."""
    fail(ExitCode.INTERNAL, "variations: not yet implemented")


@app.command()
def batch(
    ctx: typer.Context,
    jobs_file: Annotated[Path, typer.Argument(help="JSONL file with batch jobs")],
    parallel: Annotated[int, typer.Option("--parallel", help="Max parallel jobs")] = 1,
    continue_on_error: Annotated[
        bool,
        typer.Option("--continue-on-error", help="Continue after individual job failures"),
    ] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast", help="Stop on first failure")] = False,
    resume: Annotated[bool, typer.Option("--resume", help="Resume incomplete batch")] = False,
    retry: Annotated[int, typer.Option("--retry", help="Max retries per job")] = 3,
    retry_backoff: Annotated[
        float, typer.Option("--retry-backoff", help="Backoff factor in seconds")
    ] = 2.0,
    model_budget: Annotated[
        str | None, typer.Option("--model-budget", help="Model budget constraints")
    ] = None,
    max_estimated_cost: Annotated[
        float | None, typer.Option("--max-estimated-cost", help="Max total cost in USD")
    ] = None,
) -> None:
    """Execute jobs from JSONL."""
    fail(ExitCode.INTERNAL, "batch: not yet implemented")


@app.command()
def inspect(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="Image or manifest file to inspect")],
) -> None:
    """Inspect local image properties and run metadata."""
    fail(ExitCode.INTERNAL, "inspect: not yet implemented")


@app.command()
def models(ctx: typer.Context) -> None:
    """Display supported model capabilities."""
    table = Table(title="Supported Models")
    table.add_column("Alias", style="cyan")
    table.add_column("API Model", style="green")
    table.add_column("Sizes")
    table.add_column("Max Refs")
    table.add_column("Grounding")
    table.add_column("Thinking")

    for api_model, caps in CAPABILITIES.items():
        table.add_row(
            caps["alias"],
            api_model,
            ", ".join(caps["supported_sizes"]),
            str(caps["max_references"]),
            "✓" if caps["grounding"] else "—",
            ", ".join(caps["thinking_levels"]) if caps["thinking_levels"] else "—",
        )

    console.print(table)


@app.command()
def config_cmd(
    ctx: typer.Context,
    show: Annotated[bool, typer.Option("--show", help="Show current configuration")] = False,
) -> None:
    """Read or update non-secret defaults."""
    if show:
        cfg = load_config(ctx.obj.get("config"))
        if not cfg:
            typer.echo("No config file found. Create ~/.config/nanobanana/config.toml")
        else:
            typer.echo(json.dumps(cfg, indent=2))
    else:
        typer.echo("Usage: nanobanana config --show")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    app()
