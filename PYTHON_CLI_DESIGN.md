# Python CLI Design Guidelines — nanobanana

CLI architecture standards for Typer-based single-file tools.

---

## Command Structure

```
nanobanana [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

**Principles:**
- Commands encode tasks, not API parameters — `generate`, not `call-api`
- Subcommands group related workflows — `diagram` is not a flag on `generate`
- Global options affect all commands — `--model`, `--verbose`, `--json`
- Command options are scoped to that workflow — `--ground` on generate, `--mask` on edit

---

## Global Options Implementation

```python
@app.callback()
def main(
    ctx: typer.Context,
    model: Annotated[str, typer.Option("--model", "-m", help="Model alias")] = "auto",
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-d")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key")] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    format: Annotated[str, typer.Option("--format", "-f")] = "png",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    seed: Annotated[str | None, typer.Option("--seed")] = None,
    request_id: Annotated[str | None, typer.Option("--request-id")] = None,
) -> None:
    """nanobanana — Single-File Gemini Image CLI"""
    ctx.ensure_object(dict)
    ctx.obj["model"] = model
    ctx.obj["output_dir"] = output_dir
    ctx.obj["api_key"] = api_key
    ctx.obj["config"] = config
    ctx.obj["format"] = format
    ctx.obj["json"] = json_output
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose
    ctx.obj["dry_run"] = dry_run
    ctx.obj["overwrite"] = overwrite
    ctx.obj["seed"] = seed
    ctx.obj["request_id"] = request_id
```

**Rules:**
- `ctx.ensure_object(dict)` to safely store context
- Global options are resolved once in `main()` callback
- Commands access via `ctx.obj["key"]`
- `--json` mode: print JSON to stdout, logs to stderr
- `--quiet`: suppress all non-error output
- `--verbose`: structured JSON log lines to stderr

---

## Command Signatures

Each command follows this pattern:

```python
@app.command()
def generate(
    ctx: typer.Context,
    prompt: Annotated[str, typer.Argument(help="Text prompt for image generation")],
    prompt_file: Annotated[Path | None, typer.Option("--prompt-file")] = None,
    negative: Annotated[str | None, typer.Option("--negative")] = None,
    aspect: Annotated[str, typer.Option("--aspect", "-a")] = "1:1",
    size: Annotated[str, typer.Option("--size", "-s")] = "1K",
    quality: Annotated[str, typer.Option("--quality", "-q")] = "balanced",
    thinking: Annotated[str, typer.Option("--thinking")] = "auto",
    count: Annotated[int, typer.Option("--count", "-n")] = 1,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    text_output: Annotated[bool, typer.Option("--text-output")] = False,
    ground: Annotated[bool, typer.Option("--ground")] = False,
    retry_with_pro: Annotated[bool, typer.Option("--retry-with-pro")] = False,
    preset: Annotated[str | None, typer.Option("--preset", "-p")] = None,
    max_estimated_cost: Annotated[float | None, typer.Option("--max-estimated-cost")] = None,
    show_estimate: Annotated[bool, typer.Option("--show-estimate")] = False,
) -> None:
    """Generate an image from a text prompt."""
    ...
```

**Rules:**
- `ctx: typer.Context` as first parameter to access global options
- `Annotated[Type, typer.Argument/Option(...)]` for all parameters
- Help text on every option — it becomes the auto-generated help output
- Sensible defaults encoded in the command signature (not scattered in logic)

---

## Output Contract

### Normal mode (human)

Output goes to the specified path or auto-generated filename. Manifest written alongside.

```
$ nanobanana generate "A test" -o output.png
✓ Generated output.png
  Model: gemini-3.1-flash-image (auto)
  Size: 1K, Aspect: 1:1
  Manifest: output.manifest.json
```

### JSON mode (pipeline)

```json
{"status": "success", "run_id": "01J...", "outputs": ["output.png"], "manifest": "output.manifest.json"}
```

### Dry-run mode

```
$ nanobanana generate "A test" --dry-run
Selected model: gemini-3.1-flash-image
Selection reason: default general-purpose model for text-to-image
Normalized prompt: Create an image of a test.
  Requirements: ...
  Avoid: ...
Input files: none
Response format: image/png, 1:1, 1K
Estimated output count: 1
No API call made (--dry-run)
```

---

## Error Output

All errors go to stderr. Format depends on mode:

```
# Normal mode
Error: model 'lite' does not support size '4K'.
Use '--model flash', '--model pro', or change '--size' to '1K'.

# JSON mode (--json)
{"status": "error", "exit_code": 4, "error": "model 'lite' does not support size '4K'"}
```

---

## Presets

Presets encode defaults per use case. CLI flags always override preset values:

```python
PRESETS: dict[str, dict] = {
    "architecture-diagram": {
        "model": "pro",
        "aspect": "16:9",
        "size": "2K",
        "thinking": "high",
        "prompt_prefix": (
            "Create a technically precise software architecture diagram. "
            "Use a restrained visual language, explicit directional arrows, "
            "legible labels, and clear trust boundaries."
        ),
    },
    "icon": {
        "model": "lite",
        "aspect": "1:1",
        "size": "1K",
        "prompt_prefix": "Create a simple, clean icon. Minimal detail. Single subject centered.",
    },
    # ... 14 more
}
```

Resolution order: CLI flag → preset value → command default.

---

## Pipeline Integration

For shell scripting and agent workflows:

```bash
# Generate and capture output path from JSON
result=$(nanobanana generate "prompt" --json --output-dir ./gen)
output=$(echo "$result" | jq -r '.outputs[0]')
manifest=$(echo "$result" | jq -r '.manifest')

# Conditional on status
if echo "$result" | jq -e '.status == "success"' > /dev/null; then
    echo "Generated $output"
else
    echo "Failed: $(echo "$result" | jq -r '.error')" >&2
    exit 1
fi
```

---

## Progressive Disclosure

- `--help` shows command summary + most common options
- `--help --verbose` (or `nanobanana generate --help` with `rich_help_panel`) groups advanced options
- `nanobanana models` shows capabilities without requiring a command
- `nanobanana config` shows current defaults
- `nanobanana inspect image.png` shows metadata without generating

---

## Design Anti-Patterns

- ❌ Mirroring `interactions.create()` parameters as CLI flags — encode tasks instead
- ❌ `--extra-args` JSON blob — defeats discoverability, breaks `--help`
- ❌ Positional arguments that change meaning by position — use named options for clarity
- ❌ Silent model upgrades — always log selection reason
- ❌ Writing manifest to stdout — JSON mode handles pipelines; normal mode writes files
