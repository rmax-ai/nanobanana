# nanobanana Roadmap

## v0.1 — Core CLI

- [x] Project scaffold (SPEC.md, ARCHITECTURE.md, AGENTS.md)
- [ ] PEP 723 uv script header with dependencies
- [ ] CLI framework (Typer app, global options, `--json`, `--verbose`, `--dry-run`)
- [ ] Authentication (env var, `--api-key`, config file, resolution order)
- [ ] Config file support (`~/.config/nanobanana/config.toml`)
- [ ] `generate` command with aspect/size/quality/thinking/presets
- [ ] `edit` command with input image, instruction, preserve/change directives
- [ ] `compose` command with typed reference roles (subject, object, character, style, background)
- [ ] Model selection engine (auto policy with lite/flash/pro/legacy routing)
- [ ] Capability registry with per-model validation
- [ ] Structured prompt assembly (TASK/REQUIREMENTS/AVOID sections)
- [ ] Gemini Interactions API integration (`interactions.create()`)
- [ ] Response extraction (base64 decode, multi-image support)
- [ ] Manifest output (schema v1.0, run_id, model decision, SHA-256)
- [ ] Filename policy (timestamp-slug-index.ext, atomic writes)
- [ ] Error model with exit codes 0-10
- [ ] Retry handler (3 attempts, exponential backoff + jitter)
- [ ] Output: PNG/JPEG, `--json` mode for pipelines
- [ ] Unit tests (model selection, validation, serialization, retry classification)
- [ ] Contract tests (mocked SDK responses)
- [ ] Live smoke test suite (`NANOBANANA_LIVE_TESTS=1 self-test`)

## v0.2 — Specialized Workflows

- [ ] `diagram` command with types (architecture, sequence, flowchart, infographic, etc.)
- [ ] Two-phase text planning for text-heavy generation
- [ ] OCR-free text validation with retry
- [ ] `product` command (subject, logo, packaging, lighting, surface, brand-colors)
- [ ] `--preserve-geometry` for product shots
- [ ] 16 presets with prompt scaffolding and defaults
- [ ] `variations` command with parallel generation
- [ ] Contact sheet generation
- [ ] `--select-best` with multimodal evaluation
- [ ] `inspect` command (image properties, manifest metadata)

## v0.3 — Automation

- [ ] `batch` command with JSONL input format
- [ ] Bounded concurrency (`--parallel`)
- [ ] Resume support for interrupted batch runs
- [ ] Cost controls (`--max-estimated-cost`, `--show-estimate`)
- [ ] Cost preference flags (`--prefer-cheapest`, `--prefer-fastest`, `--prefer-quality`)
- [ ] `grounded` command with web search and image-search grounding
- [ ] Source attribution and citation output
- [ ] Model budget tracking during batch runs
- [ ] `--retry-with-pro` escalation for Flash validation failures

## v0.4 — Reproducible Creative Sessions

- [ ] `iterate` command with manifest-based continuation
- [ ] Output lineage (run-001/ with numbered outputs + manifest)
- [ ] Branch support for divergent edit paths
- [ ] Prompt versioning
- [ ] Automatic visual comparison between iterations
- [ ] Model-based validation of edits
- [ ] Directory-level asset projects
- [ ] `--confirm-rights` for real-person editing

## Future (post single-file)

Trigger: file exceeds ~1,500-2,000 lines → convert to conventional package
- [ ] Plugin-defined presets
- [ ] Persistent databases or remote job workers
- [ ] Reusable Python library APIs
- [ ] Multiple maintainer support
- [ ] Package-based distribution (pip install)
