# nanobanana — Single-File Gemini Image CLI

## 1. Executive synthesis

Build nanobanana as a self-contained Python CLI distributed as one executable uv script. It exposes task-oriented commands—generate, edit, compose, iterate, diagram, product mockup, grounded visual, and batch—rather than merely mirroring Gemini API parameters. Model selection defaults to gemini-3.1-flash-image, automatically downgrades to Flash Lite for cheap high-volume work, and escalates to Pro for complex composition, precise text, or professional assets. Uses Google's Interactions API through the google-genai SDK.

## 2. Problem framing

The CLI solves four related problems:
- Generate images from natural-language prompts
- Transform existing images while preserving selected subjects or structure
- Combine multiple reference images into a coherent output
- Make image generation reproducible for scripting, automation, and agent workflows

Dominant constraints:
- Single-file distribution with no manually managed virtual environment
- Predictable output paths and machine-readable metadata
- Explicit cost-quality-latency trade-offs
- Safe handling of API keys and local image files
- Support for both human interactive use and non-interactive automation
- Avoid exposing every SDK option as an unstructured flag set

Out of scope for v1: GUI, video generation, training, local image models, full DAM, pixel-accurate reproduction, long-running batch orchestration.

## 3. Distribution model

Single executable Python file with PEP 723 inline metadata:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "google-genai>=1,<2",
#   "pillow>=11,<12",
#   "typer>=0.16,<1",
#   "rich>=14,<15",
# ]
# ///
```

Installation: `chmod +x nanobanana && cp nanobanana ~/.local/bin/`
Execution: `uv run --script ./nanobanana generate "prompt"`

## 4. Authentication

Resolution order: `--api-key` → `GEMINI_API_KEY` → config file → fail (exit 2).
Optional config: `~/.config/nanobanana/config.toml`
Secrets must never appear in manifests, logs, filenames, prompt files, or exception traces.

## 5. Model policy

| CLI alias | API model | Intended use |
|---|---|---|
| lite | gemini-3.1-flash-lite-image | Cheapest, lowest latency |
| flash | gemini-3.1-flash-image | Default general-purpose |
| pro | gemini-3-pro-image | Complex, professional, high-fidelity |
| legacy | gemini-2.5-flash-image | Compatibility only |
| auto | Selected by policy | Default CLI behavior |

Automatic selection rules documented in ARCHITECTURE.md.

## 6. Command surface

```
nanobanana [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]

Commands:
  generate    Generate an image from text
  edit        Modify one existing image
  compose     Combine multiple reference images
  iterate     Continue an image-editing session
  diagram     Produce diagrams, infographics, or explanatory visuals
  product     Create product shots and mockups
  grounded    Generate a visual using current web information
  variations  Produce prompt or image variations
  batch       Execute jobs from JSONL
  inspect     Inspect local image properties and run metadata
  models      Display supported model capabilities
  config      Read or update non-secret defaults
```

## 7. Global options

`--model`, `--output-dir`, `--api-key`, `--config`, `--format`, `--json`, `--quiet`, `--verbose`, `--dry-run`, `--overwrite`, `--seed`, `--request-id`

## 8. Commands

Full command details with options, examples, and defaults documented in individual command sections of the spec.

## 9. Presets

16 presets encoding prompt scaffolding and API defaults: photo, portrait, product, editorial, icon, sticker, logo-concept, social-card, slide-hero, architecture-diagram, infographic, character-sheet, storyboard, texture, background, thumbnail, wireframe.

CLI flags override preset values.

## 10. Resolution and aspect-ratio validation

Supported ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
Flash additional: 1:4, 4:1, 1:8, 8:1

Invalid combinations fail locally with actionable error messages.

## 11. Thinking controls

`--thinking auto|minimal|high`
Flash: minimal default, high for complex layouts. Pro: model defaults. Lite/Legacy: not exposed.

## 12. Output contract

Every successful run writes `output.png` + `output.manifest.json` with schema version, run_id, model selection rationale, input/output hashes, usage, sources, and warnings.

`--json` prints one JSON object to stdout, logs to stderr.

## 13. Filename policy

`{timestamp}-{slug}-{index}.{extension}`
Sanitize unsafe chars, max slug 64, atomic writes, never overwrite without `--overwrite`.

## 14. Internal architecture

Layered pipeline: CLI parsing → command normalization → preset expansion → capability validation → model selection → prompt assembly → Gemini request construction → retry/rate-limit handling → response extraction → image validation/writing → manifest creation.

Core dataclasses: `ImageRequest`, `ReferenceImage`, `GeneratedAsset`, `ModelDecision`.

## 15. API mapping

Uses `google.genai.Client().interactions.create()` with explicit `response_format` and `generation_config`. Grounded calls use `tools=[{"type": "google_search"}]`.

## 16. Prompt assembly

Structured prompt with sections: TASK, PRIMARY REQUIREMENTS, REFERENCE ROLES, PRESERVE, AVOID, OUTPUT INTENT.

## 17. Error model

Exit codes 0-10 covering success, internal failure, invalid args, input failure, capability mismatch, auth failure, quota, safety refusal, empty response, output failure, partial batch failure.

Retryable: 429, transient 5xx, network timeout, connection reset, empty output (once).
Default: 3 attempts, exponential backoff with jitter, max 30s delay.

## 18. Safety and provenance

SynthID preserved. C2PA where supported. Manifest records generated/edited status, input hashes, model, grounding, attribution, timestamp, normalized prompt.

`--confirm-rights` for real-person editing.

## 19. Cost controls

`--max-estimated-cost`, `--show-estimate`, `--prefer-cheapest`, `--prefer-fastest`, `--prefer-quality`.
Versioned estimate table with retrieval date. Never silently upgrade to more expensive model.

## 20. Observability

`--verbose` logs run_id, command, model resolution, reference count, aspect ratio, size, attempt, elapsed_ms, output_sha256. Never logs keys, base64 data, binary payloads, signed URLs, or sensitive content with `--redact-prompts`.

## 21. Testing strategy

- Unit tests: model selection, capability validation, prompt normalization, filename generation, MIME detection, manifest serialization, reference limits, retry classification, cost estimation, secret redaction
- Contract tests: mocked SDK responses for all output types and failure modes
- Live smoke tests: opt-in with `NANOBANANA_LIVE_TESTS=1`

## 22. Typical failure modes

Documented: prompt overloading, false preservation confidence, text errors, model escalation inflation, reference-role ambiguity, grounding misconceptions, batch duplication.

## 23. Evolution path

- v0.1: Core CLI (generate, edit, compose, model selection, manifests)
- v0.2: Specialized workflows (diagram, product, presets, two-phase text, variations)
- v0.3: Automation (batch, concurrency, resume, cost ceilings, grounded)
- v0.4: Reproducible sessions (lineage, versioning, comparison, validation)

Package conversion trigger: ~1,500-2,000 lines, plugin presets needed, multiple maintainers, persistent databases, library APIs required.

## 24. Acceptance criteria

Single-file executable, uv + API key only, generate/edit/compose work without setup, model capability validation before API calls, auto model with visible explanation, reproducibility manifests, stable `--json`, secret safety, atomic writes, bounded retries, actionable errors, Interactions API usage, comprehensive tests.
