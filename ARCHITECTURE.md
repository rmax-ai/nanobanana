# nanobanana Architecture

## Problem statement

Provide a single-file, zero-setup Python CLI for Gemini image generation that encodes task-oriented commands, automatic model selection, and reproducible output — rather than exposing raw API parameters.

## Design goals

1. **Zero-install distribution** — one executable file with PEP 723 inline dependencies
2. **Task-oriented commands** — generate, edit, compose, diagram, product, grounded, variations, batch
3. **Automatic model selection** — policy-driven routing between lite/flash/pro/legacy
4. **Reproducibility** — every run produces a machine-readable manifest
5. **Safety** — secrets never logged, atomic writes, bounded retries
6. **Agent-friendly** — `--json` for pipeable structured output

## Component diagram

```
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│   CLI    │───▶│  Normalizer  │───▶│  Preset Engine  │
│ (Typer)  │    │  + Validator │    │                 │
└──────────┘    └──────┬───────┘    └────────┬────────┘
                       │                     │
                       ▼                     ▼
              ┌────────────────┐   ┌─────────────────┐
              │   Capability   │◀──│  Model Selector │
              │    Registry    │   │                 │
              └───────┬────────┘   └────────┬────────┘
                      │                    │
                      ▼                    ▼
              ┌────────────────┐   ┌─────────────────┐
              │ Request Builder│──▶│  Gemini Client  │
              │ (Prompt Asm)   │   │ (google-genai)  │
              └───────┬────────┘   └────────┬────────┘
                      │                    │
                      ▼                    ▼
              ┌────────────────┐   ┌─────────────────┐
              │ Retry + Rate   │   │ Response Extr.  │
              │ Limit Handler  │   │ (base64 decode) │
              └───────┬────────┘   └────────┬────────┘
                      │                    │
                      ▼                    ▼
              ┌────────────────┐   ┌─────────────────┐
              │ Image Writer   │   │   Manifest      │
              │ (atomic, SHA)  │   │   Generator     │
              └────────────────┘   └─────────────────┘
```

## Data flow

1. CLI parses user input into typed `ImageRequest`
2. Normalizer expands presets, validates capability constraints
3. Model selector applies policy rules (auto → lite/flash/pro)
4. Prompt assembler builds structured prompt with task/requirements/avoid sections
5. Gemini client calls `interactions.create()` with `response_format` + `generation_config`
6. Response extractor decodes base64 image data
7. Image writer saves with atomic rename + SHA-256
8. Manifest writer produces `output.manifest.json`

## Module layout

Single-file structure with explicit layers:

```
nanobanana
├── CLI parsing (Typer app + commands)
├── Normalization (presets, aspect/size validation)
├── Capability registry (model capabilities, limits)
├── Model selection (auto policy rules)
├── Prompt assembly (structured prompt builder)
├── Request construction (SDK call preparation)
├── Retry/rate-limit (exponential backoff + jitter)
├── Response extraction (base64 → bytes, grounding metadata)
├── Image writing (atomic writes, SHA-256, filename policy)
├── Manifest (JSON serialization, schema v1.0)
└── Types (ImageRequest, ReferenceImage, GeneratedAsset, ModelDecision)
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Single-file uv script | Eliminates venv management; PEP 723 handles deps |
| Task-oriented commands over API-parameter mirroring | Encodes correct defaults; reduces caller cognitive load |
| Auto model selection with policy rules | Users shouldn't need to know which model for which task |
| Structured prompt assembly | Auditable, reduces ambiguity vs concatenated flags |
| Manifest per run | Enables reproducibility, lineage tracking, agent integration |
| Capability registry over scattered conditionals | Single source of truth for model limits |
| Atomic writes with temp file | Prevents corrupted output from interrupted writes |
| Built-in retry classification | Retryable (429, 5xx) vs non-retryable (4xx, safety refusals) |

## Trade-offs

| Trade-off | Benefit | Cost |
|---|---|---|
| Single file vs package | Zero setup, trivial distribution | Harder to test modules independently, ~2K line limit |
| Typer over argparse | Automatic help, shell completion, type coercion | One more dependency |
| google-genai over raw REST | Official SDK, maintained auth, streaming | Version coupling, SDK overhead |
| Auto model selection vs explicit | Better defaults, fewer flags | Users may not understand why a model was chosen |
| Manifest per run vs optional | Always-auditable output | Extra I/O, disk usage for high-volume batch |
| Capability registry vs inline checks | Centralized validation, easy to update | Upfront mapping work for all model variants |
