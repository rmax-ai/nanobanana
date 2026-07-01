# Python Architecture Guidelines — nanobanana

Architecture patterns for the nanobanana single-file CLI.

---

## Single-File Architecture

Until the file exceeds ~1,500-2,000 lines, all code lives in one file with explicit sections:

```
1. Shebang + PEP 723 metadata
2. Imports (stdlib → third-party → google-genai)
3. Constants (EXIT_CODES, ASPECT_RATIOS, SUPPORTED_SIZES, MIME_TYPES)
4. Type definitions (@dataclass frozen=True)
5. Capability registry (MODEL_CAPABILITIES dict)
6. Preset definitions (PRESETS dict)
7. Cost estimates (COST_TABLE dict with retrieval_date)
8. Config loading (load_config, resolve_api_key)
9. Model selection engine (select_model, CAPABILITIES registry queries)
10. Capability validation (validate_request against registry)
11. Prompt assembly (build_normalized_prompt, expand_negative)
12. Request construction (build_interaction_input)
13. Retry handler (with_retry, is_retryable, is_safety_refusal)
14. Response extraction (extract_images, extract_grounding)
15. Image writing (atomic_write, generate_filename, sha256_file)
16. Manifest (write_manifest, ManifestSchema)
17. CLI commands (Typer app, callback, command functions)
18. main() entry point
```

**Rules:**
- Section order matches the data flow (top → bottom = input → output)
- No section imports from a later section (except type definitions at the top)
- Functions at module level; no nested function definitions beyond trivial lambdas
- Section boundaries marked with `# === Section Name ===` comments
- No circular dependencies between sections (the layered structure prevents this)

---

## Layered Data Flow

```
CLI parsing (Typer)
    ↓ ImageRequest
Command normalization (presets, defaults)
    ↓ ImageRequest (normalized)
Capability validation (registry lookup)
    ↓ ImageRequest (validated)
Model selection (policy engine)
    ↓ (model_alias, selection_reason)
Prompt assembly (structured sections)
    ↓ Content parts list
Request construction (SDK input)
    ↓ API call
Response extraction (base64 → bytes)
    ↓ list[bytes]
Image writing (atomic, SHA-256)
    ↓ GeneratedAsset
Manifest creation (JSON)
    ↓ manifest.json
```

Each layer is a pure function (or close to it). State flows forward; no back-references.

---

## Capability Registry Pattern

Centralize all model-specific knowledge in one data structure:

```python
CAPABILITIES: dict[str, dict] = {
    "gemini-3.1-flash-lite-image": {
        "alias": "lite",
        "supported_sizes": ["1K"],
        "max_references": 14,
        "character_consistency": False,
        "style_references": False,
        "grounding": False,
        "thinking_levels": [],
        "image_search": False,
    },
    "gemini-3.1-flash-image": {
        "alias": "flash",
        "supported_sizes": ["0.5K", "1K", "2K", "4K"],
        "max_references": 10,
        "character_consistency": True,
        "style_references": True,
        "grounding": True,
        "thinking_levels": ["minimal", "high"],
        "image_search": True,
    },
    "gemini-3-pro-image": {
        "alias": "pro",
        "supported_sizes": ["1K", "2K", "4K"],
        "max_references": 6,
        "character_consistency": True,
        "style_references": True,
        "grounding": True,
        "thinking_levels": [],  # model defaults
        "image_search": False,
    },
    "gemini-2.5-flash-image": {
        "alias": "legacy",
        "supported_sizes": ["1K"],
        "max_references": 4,
        "character_consistency": False,
        "style_references": False,
        "grounding": False,
        "thinking_levels": [],
        "image_search": False,
    },
}
```

**Rules:**
- Capability queries go through accessor functions, never raw dict key access
- `get_capability(model, key) -> Any` with fallback
- `supports(model, feature) -> bool` for boolean checks
- Adding a new model variant = one row in the registry + capability tests

Validation functions reference the registry:

```python
def validate_request(request: ImageRequest) -> None:
    caps = CAPABILITIES[request.model]
    if request.image_size not in caps["supported_sizes"]:
        fail(ExitCode.CAPABILITY_MISMATCH,
             f"model '{request.model}' does not support size '{request.image_size}'")
    if request.grounding and not caps["grounding"]:
        fail(ExitCode.CAPABILITY_MISMATCH, ...)
```

---

## Model Selection Policy

```python
def select_model(request: ImageRequest) -> ModelDecision:
    if request.model != "auto":
        return ModelDecision(request.model, request.model, "explicitly requested")

    # Lite rules
    if _should_use_lite(request):
        return ModelDecision("auto", "gemini-3.1-flash-lite-image", "...")

    # Pro rules
    if _should_use_pro(request):
        return ModelDecision("auto", "gemini-3-pro-image", "...")

    # Default: Flash
    return ModelDecision("auto", "gemini-3.1-flash-image", "default general-purpose model")
```

**Rules:**
- `_should_use_lite()` checks: size=1K AND no grounding AND no character consistency AND no style references AND ≤1 reference image AND task in {thumbnail, variation, background_adjustment, sticker, draft, batch_preview}
- `_should_use_pro()` checks: quality=professional OR task in {diagram, infographic, product, layout} OR extensive exact text OR style references present OR complex spatial instructions OR `--retry-with-pro` after Flash failure
- Always return a `ModelDecision` with `selection_reason` — visible in `--dry-run` and manifest

---

## Prompt Assembly

Structured, auditable prompt construction:

```python
def build_normalized_prompt(
    original_prompt: str,
    negative: str | None = None,
    references: tuple[ReferenceImage, ...] = (),
    preset_prompt_prefix: str | None = None,
    aspect_ratio: str | None = None,
) -> str:
    sections = []
    if preset_prompt_prefix:
        sections.append(("TASK", preset_prompt_prefix))
    sections.append(("TASK" if not preset_prompt_prefix else "CONTEXT", original_prompt))

    requirements = []
    if aspect_ratio:
        requirements.append(f"- {aspect_ratio} composition")
    # ... reference roles, preserve constraints

    if requirements:
        sections.append(("REQUIREMENTS", "\n".join(requirements)))

    avoids = []
    if negative:
        avoids.append(f"- {negative}")
    avoids.append("- Generic humanoid robots")
    avoids.append("- Decorative pseudo-code")
    avoids.append("- Illegible labels")
    if avoids:
        sections.append(("AVOID", "\n".join(avoids)))

    return "\n\n".join(f"{label}:\n{content}" for label, content in sections)
```

**Rules:**
- Always produce structured sections (TASK, REQUIREMENTS, AVOID, REFERENCE ROLES, PRESERVE, OUTPUT INTENT)
- `--negative` is converted to AVOID constraints, not a separate API parameter
- Reference roles are annotated with explicit instructions ("Reference 1 is the primary character...")
- The normalized prompt is stored in the manifest for auditability

---

## Design Non-Negotiables

1. **No model-specific conditionals in command handlers.** Route through capability registry and model selector.
2. **Every output has a manifest.** `output.png` → `output.manifest.json`. No manifest = bug.
3. **No API key in any stored artifact.** Not in manifests, filenames, verbose logs, or exception traces.
4. **Atomic writes only.** Write to temp file, rename on success. Never write directly to output path.
5. **Auto model must explain itself.** `selection_reason` in manifest and `--dry-run` output.
6. **Structured prompt assembly always.** Never concatenate flags into prose.
7. **Validation before API call.** Local capability checks catch invalid combinations without spending quota.
8. **Exit codes are stable.** Shell scripts and agents depend on specific codes.

---

## File Size Trigger for Package Conversion

When the single file exceeds ~1,500-2,000 lines, extract into a conventional package:

```
nanobanana/
├── pyproject.toml
├── src/nanobanana/
│   ├── __init__.py
│   ├── cli.py          # Typer app + commands
│   ├── types.py        # ImageRequest, ReferenceImage, etc.
│   ├── capabilities.py # Registry + validation
│   ├── models.py       # Model selection engine
│   ├── prompts.py      # Prompt assembly
│   ├── api.py          # SDK interaction, retry, response extraction
│   ├── outputs.py      # Image writing, manifests, filename generation
│   ├── config.py       # Config loading, auth
│   └── presets.py      # Preset definitions
├── tests/
│   └── ...
└── nanobanana           # Entry-point shell script (uv run wrapper)
```
