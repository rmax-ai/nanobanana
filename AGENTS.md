# AGENTS.md — Guidelines for nanobanana

This document captures the conventions that all contributors and AI coding agents must follow when working on nanobanana.

---

## 1. Project DNA

- **Package distribution via pyproject.toml + uv.** Thin PEP 723 wrapper for single-file compat.
- **pyproject.toml with uv.** Thin PEP 723 wrapper available.
- **uv-managed virtualenv via pyproject.toml.**
- **Task-oriented CLI.** Commands encode use cases (generate, edit, compose), not API parameters.
- **Model selection is policy-driven.** Never scatter model-specific conditionals across command handlers.
- **Every run is auditable.** Output includes a manifest with SHA-256, model decision rationale, and normalized prompt.

## 2. Code Organisation

```
src/nanobanana/
├── __init__.py          — Package marker
├── types.py             — Frozen dataclasses (ImageRequest, ReferenceImage, GeneratedAsset, ModelDecision)
├── constants.py         — Exit codes, capabilities, presets, cost table, aspect ratios
├── utils.py             — Filesystem, hashing, config, MIME detection, reference loading
├── model_selection.py   — Model resolution, auto selection policy, capability validation
├── prompt.py            — Structured prompt assembly, edit instruction builder
├── client.py            — Gemini API request construction, retry, error classification
├── response.py          — Image extraction, grounding metadata, response error detection
├── output.py            — Atomic file I/O, manifest serialization, dry-run display
├── pipeline.py          — run_generate_pipeline orchestration (prompt → API → output)
└── cli.py               — Typer app, global options, 11 commands
```

- Import order: stdlib → third-party → google-genai
- Use `@dataclass(frozen=True)` for value types
- Functions do one thing; no function over 50 lines without justification
- No module-level state except immutable constants (CAPABILITIES, PRESETS, EXIT_CODES)

## 3. Error Handling

- Exit codes 0-10 per spec section 17
- Use `typer.Exit(code=N)` for user-facing errors
- Use `typer.echo(message, err=True)` for error messages
- Classification functions: `is_retryable(exception) -> bool`, `is_safety_refusal(response) -> bool`
- Never silently swallow API errors — log with `--verbose`, surface to user

## 4. Python Conventions

- Python 3.12+ only
- Type hints on all public functions
- Use `Path` for all file paths, never raw strings
- Use `|` union syntax (`str | None`) not `Optional[str]`
- Use `frozen=True` dataclasses for request/response types
- Use `match/case` where it improves readability over if/elif chains
- Use `subprocess.run` only if unavoidable; prefer `pathlib`, `shutil`, `hashlib`

## 5. Testing

- Unit tests in `tests/unit/` — no API calls, no I/O
- Contract tests in `tests/contract/` — mock `google.genai.Client`
- Live smoke tests in `tests/live/` — gated behind `NANOBANANA_LIVE_TESTS=1`
- Test file: `tests/test_nanobanana.py` (unit + contract in one file until line count warrants split)
- Test framework: pytest
- Use `monkeypatch` for environment and filesystem isolation
- Mock SDK at the `client.interactions.create()` level for contract tests
- Every model selection branch must have a unit test
- Every exit code must have a contract test

## 6. Documentation

- `SPEC.md` is authoritative — update it when scope changes
- `ARCHITECTURE.md` describes design decisions and trade-offs
- `ROADMAP.md` tracks version progress with checkboxes
- README.md: quickstart, example commands, config reference
- Inline comments explain *why*, not *what*

## 7. Dependencies

- `google-genai>=1,<2` — Gemini Interactions API
- `pillow>=11,<12` — Image property inspection, format conversion
- `typer>=0.16,<1` — CLI framework with type coercion
- `rich>=14,<15` — Rich terminal output (tables, progress, markup)
- No transitive dependencies beyond what these pull in
- No new dependencies without explicit justification

## 8. Formatting and Linting

- Ruff for both format and lint
- Line length: 100
- Single quotes for strings unless string contains single quotes
- Trailing commas in multi-line collections
- No unused imports, no dead code

## 9. CI/CD

- GitHub Actions: ruff format --check, ruff check, ty check, pytest
- No deployment — single-file script, users copy it
- Release tags follow ROADMAP.md versions

## 10. Architecture Non-Negotiables

- **No API key in any stored artifact.** Not in manifests, not in filenames, not in verbose logs, not in exception traces.
- **No model-specific conditionals in command handlers.** Route through capability registry and model selector.
- **Every output has a manifest.** No manifest = bug.
- **Atomic writes only.** Write to temp, rename on success.
- **Auto model must explain itself.** Selection reason in manifest and visible in `--dry-run`.
- **Structured prompt assembly always.** Never concatenate flags into prose.
