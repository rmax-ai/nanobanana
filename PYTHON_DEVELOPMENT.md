# Python Development Guidelines — nanobanana

Day-to-day engineering standards for the nanobanana single-file CLI.

---

## Language Version

- **Python 3.12+** required (PEP 723 minimum)
- Use `|` union syntax (`str | None`) not `Optional[str]`
- Use `match/case` where it improves readability over if/elif chains
- Use `datetime.now(UTC)` — never `datetime.utcnow()` (deprecated in 3.12)

---

## CLI Framework: Typer

```python
import typer
from pathlib import Path
from typing import Annotated

app = typer.Typer(no_args_is_help=True)

@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="Text prompt for image generation")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model alias")] = "auto",
    aspect: Annotated[str, typer.Option("--aspect", "-a", help="Aspect ratio")] = "1:1",
    size: Annotated[str, typer.Option("--size", "-s", help="Output resolution")] = "1K",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output path")] = None,
) -> None:
    """Generate an image from a text prompt."""
    ...
```

**Rules:**
- Use `typer.Typer(no_args_is_help=True)` to show help when no subcommand given
- Add callback for global options (`--api-key`, `--verbose`, `--json`, etc.)
- Annotated types for all parameters — Typer uses them for coercion and help text
- Rich help text via `rich_help_panel` for grouping related options
- Exit codes: `raise typer.Exit(code=N)` for user-facing errors
- Error messages: `typer.echo(message, err=True)` to stderr

---

## Data Modeling: Dataclasses

Prefer `@dataclass(frozen=True)` over Pydantic for internal types (no JSON deserialization needed):

```python
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, UTC
import uuid

@dataclass(frozen=True)
class ImageRequest:
    command: str
    prompt: str
    model: str
    references: tuple["ReferenceImage", ...] = ()
    aspect_ratio: str | None = None
    image_size: str | None = None
    mime_type: str = "image/png"
    thinking_level: str | None = None
    grounding: str | None = None
    seed: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass(frozen=True)
class ReferenceImage:
    path: Path
    role: str
    mime_type: str
    sha256: str

@dataclass(frozen=True)
class GeneratedAsset:
    path: Path
    mime_type: str
    sha256: str

@dataclass(frozen=True)
class ModelDecision:
    requested: str
    resolved: str
    selection_reason: str
```

**Rules:**
- `frozen=True` for immutability
- Use `field(default_factory=...)` for mutable defaults (never bare `[]` or `{}`)
- `tuple[...]` for immutable sequences, `list[...]` only when mutation is needed
- `uuid.uuid4()` for run IDs via `default_factory`

---

## Error Handling

Exit codes per spec section 17:

```python
class ExitCode:
    SUCCESS = 0
    INTERNAL = 1
    INVALID_ARGS = 2
    INPUT_FAILURE = 3
    CAPABILITY_MISMATCH = 4
    AUTH_FAILURE = 5
    QUOTA_EXCEEDED = 6
    SAFETY_REFUSAL = 7
    EMPTY_RESPONSE = 8
    OUTPUT_FAILURE = 9
    PARTIAL_BATCH = 10

def fail(code: int, message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=code)
```

Classification functions:
```python
def is_retryable(exc: Exception) -> bool:
    """Classify whether an exception warrants a retry."""
    # 429, transient 5xx, network timeout, connection reset
    ...

def is_safety_refusal(response: object) -> bool:
    """Check if the API response is a safety refusal."""
    ...
```

---

## Retry Pattern

```python
import time
import random
from typing import TypeVar, Callable

T = TypeVar("T")

def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 3,
    max_delay: float = 30.0,
) -> T:
    """Execute fn with exponential backoff + jitter for retryable errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if not is_retryable(e) or attempt == max_attempts:
                raise
            delay = min(2 ** (attempt - 1) + random.uniform(0, 1), max_delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")
```

---

## File I/O: Atomic Writes

```python
import tempfile
import shutil
from pathlib import Path

def atomic_write(path: Path, data: bytes, overwrite: bool = False) -> Path:
    """Write data atomically via temp file + rename."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    try:
        Path(tmp).write_bytes(data)
        shutil.move(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return path
```

---

## Hashing

```python
import hashlib

def sha256_file(path: Path) -> str:
    """SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sha256_bytes(data: bytes) -> str:
    """SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()
```

---

## Logging (Structured JSON for --verbose)

```python
import json
import sys
from datetime import datetime, UTC, timezone

def vlog(**kwargs) -> None:
    """Emit a structured verbose log line to stderr."""
    entry = {"timestamp": datetime.now(UTC).isoformat(), **kwargs}
    print(json.dumps(entry, default=str), file=sys.stderr, flush=True)

# Usage:
# vlog(run_id=request.request_id, command="generate", resolved_model=decision.resolved, ...)
```

**Never log:** API keys, base64 image data, binary payloads, signed URLs, sensitive prompt content when `--redact-prompts` is active.

---

## Filename Generation

```python
import re
from datetime import datetime, UTC

def generate_filename(
    slug: str,
    index: int = 1,
    extension: str = "png",
    max_slug_len: int = 64,
) -> str:
    """Generate timestamp-slug-index.ext filename."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^a-z0-9_-]", "-", slug.lower())[:max_slug_len]
    safe = re.sub(r"-{2,}", "-", safe).strip("-")
    return f"{timestamp}-{safe}-{index:02d}.{extension}"
```

---

## Testing

```python
import pytest
from unittest.mock import MagicMock, patch

# Unit test — no I/O, no API
def test_model_selection_auto_routes_2k_to_flash():
    request = ImageRequest(command="generate", prompt="test", model="auto", image_size="2K")
    decision = select_model(request)
    assert decision.resolved == "gemini-3.1-flash-image"
    assert "2K" in decision.selection_reason

# Contract test — mock SDK
def test_generate_handles_safety_refusal(monkeypatch):
    mock_client = MagicMock()
    mock_client.interactions.create.return_value = safety_refusal_response()
    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(typer.Exit) as exc:
            run_generate(ImageRequest(...))
        assert exc.value.exit_code == ExitCode.SAFETY_REFUSAL
```

**Rules:**
- Unit tests in `tests/test_nanobanana.py` (single file until line count warrants split)
- Mock at `client.interactions.create()` level — never mock internal functions
- Use `monkeypatch.setenv("GEMINI_API_KEY", "test-key")` for auth setup
- Every model selection branch → unit test
- Every exit code → contract test
- Live tests: `@pytest.mark.skipif(not os.getenv("NANOBANANA_LIVE_TESTS"), reason="opt-in")`

---

## Tooling

```bash
# Format + lint
ruff format nanobanana tests/
ruff check --fix nanobanana tests/

# Type check
ty check nanobanana

# Tests
uv run --script nanobanana --help  # smoke test
pytest tests/ -v
```

---

## Key Gotchas

- **No `pyproject.toml`.** Dependencies live in PEP 723 inline metadata.
- **No `src/` layout.** It's a single file. Tests import it via `sys.path` manipulation or `importlib`.
- **No async.** CLI is synchronous. SDK calls block; no need for asyncio.
- **No structlog.** Use the `vlog()` pattern for structured verbose output. No heavy logging framework for a single file.
- **Trailing comma in tuples.** `("str",)` is a tuple, not a string. Watch for this in parenthesized expressions.
- **`datetime.utcnow()` deprecated.** Use `datetime.now(UTC)`.
- **PEP 723 `requires-python`.** Must be `">=3.12"` not `">=3.12,<4"` — uv handles the upper bound.
