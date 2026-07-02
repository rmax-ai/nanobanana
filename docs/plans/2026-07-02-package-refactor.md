# Refactor nanobanana to Proper Python Package

> **For Hermes:** Use Droid to implement this plan phase-by-phase. Sequential phases on `main` (single file, no worktrees).

**Goal:** Split the 1,965-line single-file script into a `src/nanobanana/` package with `pyproject.toml` and uv-based dependency management, while keeping a thin PEP 723 backward-compat wrapper.

**Architecture:** `src/` layout with modules following existing logical layers. Tests switch from `exec()` import hack to normal package imports. Entry point via `[project.scripts]`.

**Tech Stack:** Python 3.12+, uv, typer, rich, google-genai, pillow, pytest, ruff

---

## Acceptance Criteria

- [ ] `uv run nanobanana generate "test" --dry-run` works identically to before
- [ ] `uv run pytest` passes all 60+ tests with zero changes to test logic
- [ ] `uv run ruff check src/ tests/` is clean
- [ ] `pyproject.toml` has all 4 dependencies with correct version ranges
- [ ] Old `./nanobanana` PEP 723 script still works as a thin wrapper
- [ ] `src/nanobanana/` has 10 modules matching logical layers
- [ ] No imports use the `exec()` hack — tests import from `nanobanana.*` normally
- [ ] AGENTS.md updated to reflect package structure

---

## Implementation Tasks

### Phase 1: Scaffold project structure

#### Task 1.1: Create pyproject.toml

**Objective:** Replace PEP 723 inline metadata with a standard pyproject.toml.

**Files:**
- Create: `pyproject.toml`

```toml
[project]
name = "nanobanana"
version = "0.1.0"
description = "Single-file Gemini Image CLI"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "google-genai>=1,<2",
    "pillow>=11,<12",
    "typer>=0.16,<1",
    "rich>=14,<15",
]

[project.scripts]
nanobanana = "nanobanana.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8,<9",
    "ruff>=0.11,<1",
]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "RUF", "SIM"]
```

**Step 1: Write the file.**

**Step 2: Verify:** `uv sync` installs all deps, no errors.

#### Task 1.2: Create src/nanobanana/__init__.py

**Objective:** Package init — expose key public API.

**Files:**
- Create: `src/nanobanana/__init__.py`

```python
"""nanobanana — Single-File Gemini Image CLI."""
```

#### Task 1.3: Create directory structure

**Objective:** Empty files for all 10 modules.

```bash
mkdir -p src/nanobanana
touch src/nanobanana/types.py
touch src/nanobanana/constants.py
touch src/nanobanana/utils.py
touch src/nanobanana/model_selection.py
touch src/nanobanana/prompt.py
touch src/nanobanana/client.py
touch src/nanobanana/response.py
touch src/nanobanana/output.py
touch src/nanobanana/pipeline.py
touch src/nanobanana/cli.py
```

**Verify:** `python -c "import nanobanana"` works.

---

### Phase 2: Split constants and types

#### Task 2.1: Create constants.py

**Objective:** Move all immutable constants to their own module.

**Files:**
- Create: `src/nanobanana/constants.py`

**Content to move** (lines from original `nanobanana`):
- `ExitCode` class (lines 49-60)
- `SUPPORTED_ASPECT_RATIOS` (line 63-73)
- `FLASH_ONLY_ASPECT_RATIOS` (line 75)
- `SUPPORTED_SIZES` (line 76)
- `SUPPORTED_FORMATS` (line 77)
- `MIME_MAP` (line 78)
- `CAPABILITIES` dict (lines 284-325)
- `ALIAS_TO_MODEL` (line 327)
- `PRESETS` dict (lines 334-363)
- `COST_TABLE` (lines 369-374)
- `COST_RETRIEVAL_DATE` (line 375)

**Step 1: Copy these to constants.py.**
**Step 2: Verify:** `python -c "from nanobanana.constants import CAPABILITIES, ExitCode; print(len(CAPABILITIES))"` prints `4`.

#### Task 2.2: Create types.py

**Objective:** Move all dataclasses.

**Files:**
- Create: `src/nanobanana/types.py`

**Content to move:**
- `ImageRequest` (lines 86-103)
- `ReferenceImage` (lines 105-110)
- `GeneratedAsset` (lines 113-118)
- `ModelDecision` (lines 120-125)

Imports needed: `from __future__ import annotations`, `from dataclasses import dataclass, field`, `import uuid`, `from pathlib import Path`

**Step 1: Create types.py with full content.**
**Step 2: Verify:** `python -c "from nanobanana.types import ImageRequest; r = ImageRequest(command='test', prompt='x', model='auto'); print(r.command)"` prints `test`.

---

### Phase 3: Split utility functions

#### Task 3.1: Create utils.py

**Objective:** Move all utility/helper functions.

**Files:**
- Create: `src/nanobanana/utils.py`

**Content to move:**
- `vlog()` (lines 135-140)
- `fail()` (lines 143-146)
- `resolve_api_key()` (lines 149-167)
- `load_config()` (lines 170-180)
- `slugify()` (lines 583-589)
- `sha256_file()` (lines 592-598)
- `sha256_bytes()` (lines 601-603)
- `detect_mime_type()` (lines 568-580)
- `expand_preset()` (lines 606-621)
- `load_reference()` (lines 624-635)
- `load_references()` (lines 638-661)
- `_get()` (lines 664-669)

Imports from types: `ReferenceImage`
Imports from constants: `ExitCode`, `PRESETS`

**Step 1: Write utils.py with all functions.**
**Step 2: Verify:** `python -c "from nanobanana.utils import slugify; print(slugify('Hello World'))"` prints `hello-world`.

---

### Phase 4: Split model selection

#### Task 4.1: Create model_selection.py

**Objective:** Move model resolution, selection, validation, cost estimation.

**Files:**
- Create: `src/nanobanana/model_selection.py`

**Content to move:**
- `resolve_model_alias()` (lines 382-393)
- `get_capability()` (lines 396-402)
- `supports()` (lines 405-407)
- `_contains_explicit_text_request()` (lines 410-417)
- `_should_use_lite()` (lines 420-442)
- `_should_use_pro()` (lines 445-451)
- `select_model()` (lines 454-482)
- `validate_request()` (lines 485-554)
- `estimate_cost()` (lines 557-565)

Imports from constants: `CAPABILITIES`, `ALIAS_TO_MODEL`, `SUPPORTED_ASPECT_RATIOS`, `FLASH_ONLY_ASPECT_RATIOS`, `ExitCode`
Imports from types: `ImageRequest`, `ModelDecision`
Imports from utils: `fail`

**Step 1: Write model_selection.py.**
**Step 2: Verify:** `python -c "from nanobanana.model_selection import select_model; from nanobanana.types import ImageRequest; r = ImageRequest(command='generate', prompt='x', model='auto'); d = select_model(r); print(d.resolved)"` prints `gemini-3.1-flash-image`.

---

### Phase 5: Split prompt assembly

#### Task 5.1: Create prompt.py

**Objective:** Move prompt construction functions.

**Files:**
- Create: `src/nanobanana/prompt.py`

**Content to move:**
- `build_normalized_prompt()` (lines 677-742)
- `expand_negative()` (lines 745-751)
- `annotate_references()` (lines 754-763)
- `build_edit_instruction()` (lines 1197-1227)

Imports from types: `ImageRequest`, `ReferenceImage`

**Step 1: Write prompt.py.**
**Step 2: Verify:** `python -c "from nanobanana.prompt import build_normalized_prompt; from nanobanana.types import ImageRequest; r = ImageRequest(command='generate', prompt='x', model='auto'); p = build_normalized_prompt(r); print('TASK' in p)"` prints `True`.

---

### Phase 6: Split client and response

#### Task 6.1: Create client.py

**Objective:** Move API request construction, retry handler, error classification.

**Files:**
- Create: `src/nanobanana/client.py`

**Content to move:**
- `build_interaction_input()` (lines 771-812)
- `with_retry()` (lines 820-847)
- `is_retryable()` (lines 850-864)
- `classify_api_exception()` (lines 867-885)
- `is_safety_refusal()` (lines 888-902)
- `execute_request()` (lines 910-912)

Imports from types: `ImageRequest`, `ModelDecision`
Imports from utils: `vlog`, `_get`
Imports from constants: `ExitCode`

**Step 1: Write client.py.**
**Step 2: Verify:** `python -c "from nanobanana.client import is_retryable; class E(Exception): code='429'; print(is_retryable(E()))"` prints `True`.

#### Task 6.2: Create response.py

**Objective:** Move response extraction functions.

**Files:**
- Create: `src/nanobanana/response.py`

**Content to move:**
- `extract_images()` (lines 915-935)
- `extract_grounding_metadata()` (lines 938-962)
- `classify_response_error()` (lines 965-974)

Imports from utils: `_get`
Imports from client: `is_safety_refusal`
Imports from constants: `ExitCode`

**Step 1: Write response.py.**
**Step 2: Verify:** `python -c "from nanobanana.response import classify_response_error; from nanobanana.constants import ExitCode; print(classify_response_error(None) == ExitCode.EMPTY_RESPONSE)"` prints `True`.

---

### Phase 7: Split output layer

#### Task 7.1: Create output.py

**Objective:** Move file writing, manifest, emit_json, dry-run display.

**Files:**
- Create: `src/nanobanana/output.py`

**Content to move:**
- `atomic_write()` (lines 982-999)
- `generate_filename()` (lines 1002-1012)
- `write_manifest()` (lines 1015-1098)
- `resolve_output_path()` (lines 1101-1113)
- `emit_json()` (lines 1116-1134)
- `print_dry_run()` (lines 1137-1173)
- `_resolve_output_path_for_image()` (lines 1181-1194)

Imports from types: `ImageRequest`, `ModelDecision`, `GeneratedAsset`
Imports from utils: `atomic_write` actually — wait, it's in this file. Let me make sure nothing circular.
Imports from model_selection: `estimate_cost`
Imports from constants: `COST_RETRIEVAL_DATE`

**Step 1: Write output.py.**
**Step 2: Verify:** `python -c "from nanobanana.output import slugify_import_test; print('ok')"` — just make sure it imports cleanly.

---

### Phase 8: Split pipeline and CLI

#### Task 8.1: Create pipeline.py

**Objective:** Move `run_generate_pipeline()` — the main orchestration function.

**Files:**
- Create: `src/nanobanana/pipeline.py`

**Content to move:** `run_generate_pipeline()` (lines 1230-1527)

This function imports from nearly every module. It needs:
- `types`: `ImageRequest`, `ModelDecision`, `ReferenceImage`, `GeneratedAsset`
- `constants`: `ExitCode`, `MIME_MAP`
- `utils`: `vlog`, `fail`, `resolve_api_key`, `expand_preset`
- `model_selection`: `select_model`, `validate_request`, `estimate_cost`
- `prompt`: `build_normalized_prompt`
- `client`: `build_interaction_input`, `execute_request`, `classify_api_exception`
- `response`: `extract_images`, `extract_grounding_metadata`, `classify_response_error`
- `output`: `_resolve_output_path_for_image`, `write_manifest`, `emit_json`, `print_dry_run`, `atomic_write`, `sha256_file`

**Step 1: Write pipeline.py.**
**Step 2: Verify:** `python -c "from nanobanana.pipeline import run_generate_pipeline; print('ok')"` — imports clean.

#### Task 8.2: Create cli.py

**Objective:** Move the Typer app, global callback, and all command definitions.

**Files:**
- Create: `src/nanobanana/cli.py`

**Content to move:**
- `console` instantiation (line 131)
- `app` instantiation (line 132)
- `main()` callback (lines 188-277)
- `generate` command (lines 1535-1605)
- `edit` command (lines 1608-1684)
- `compose` command (lines 1687-1749)
- `diagram` command (lines 1752-1774)
- `product` command (lines 1777-1814)
- `grounded` command (lines 1817-1838)
- `variations` command (lines 1841-1874)
- `batch` command (lines 1877-1906)
- `inspect` command (lines 1909-1915)
- `models` command (lines 1918-1939)
- `config_cmd` command (lines 1942-1957)
- `if __name__ == "__main__": app()` (lines 1964-1965)

Imports from pipeline: `run_generate_pipeline`
Imports from utils: `fail`, `load_config`, `load_reference`, `load_references`
Imports from prompt: `build_edit_instruction`
Imports from constants: `CAPABILITIES`, `ExitCode`
Imports from types: `ReferenceImage`, `ImageRequest`

**Step 1: Write cli.py.**
**Step 2: Verify:** `uv run nanobanana --help` shows the help text.

---

### Phase 9: Fix tests

#### Task 9.1: Rewrite conftest.py

**Objective:** Replace `exec()` hack with normal package imports.

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_nanobanana.py`

**conftest.py new content:**

```python
from pathlib import Path

import pytest

# Import everything from the package — no exec() hack
from nanobanana.constants import (  # noqa: E402, F401
    CAPABILITIES,
    COST_TABLE,
    PRESETS,
    ExitCode,
)
from nanobanana.types import (  # noqa: E402, F401
    GeneratedAsset,
    ImageRequest,
    ModelDecision,
    ReferenceImage,
)
from nanobanana.model_selection import (  # noqa: F401
    _should_use_lite,
    _should_use_pro,
    estimate_cost,
    get_capability,
    resolve_model_alias,
    select_model,
    supports,
    validate_request,
)
from nanobanana.utils import (  # noqa: F401
    detect_mime_type,
    load_reference,
    sha256_bytes,
    sha256_file,
    slugify,
)
from nanobanana.prompt import (  # noqa: F401
    annotate_references,
    build_edit_instruction,
    build_normalized_prompt,
    expand_negative,
)
from nanobanana.client import (  # noqa: F401
    build_interaction_input,
    classify_api_exception,
    execute_request,
    is_retryable,
    is_safety_refusal,
    with_retry,
)
from nanobanana.response import (  # noqa: F401
    extract_grounding_metadata,
)
from nanobanana.output import (  # noqa: F401
    atomic_write,
    generate_filename,
    resolve_output_path,
    write_manifest,
)
from nanobanana.pipeline import run_generate_pipeline  # noqa: F401
from nanobanana.cli import app  # noqa: F401

# Make old conftest-style module import work for test files
import sys
sys.modules.setdefault("conftest", sys.modules[__name__])


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_request():
    return ImageRequest(
        command="generate",
        prompt="a red banana",
        model="auto",
    )


@pytest.fixture
def sample_reference(temp_dir):
    path = temp_dir / "ref.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return load_reference(path, role="style")


class MockGenai:
    def __init__(self):
        self.Client = MagicMock


@pytest.fixture
def mock_client(monkeypatch):
    from unittest.mock import MagicMock
    mock_genai = MockGenai()
    # Patch the google.genai module in the client module
    monkeypatch.setattr("nanobanana.client.genai", mock_genai.__class__())
    # Also the pipeline imports it
    monkeypatch.setattr("nanobanana.pipeline.genai", mock_genai.__class__())
    return mock_genai
```

**Step 1: Write new conftest.py.**
**Step 2: Verify:** `python -c "import conftest"` works from tests/ directory.

#### Task 9.2: Update test imports

**Objective:** Fix the test file to work with new package structure.

**Files:**
- Modify: `tests/test_nanobanana.py`

The test file currently does:
```python
import conftest
from conftest import (ExitCode, ImageRequest, ...)
```

This should still work since conftest.py re-exports everything. But `genai` patching needs to target `nanobanana.client.genai` and `nanobanana.pipeline.genai` instead of `sys.modules["nanobanana"].genai`.

**Step 1: Update mock_client fixture in conftest.py to patch the right module paths.**
**Step 2: Run:** `uv run pytest tests/ -v` — all tests should pass.

#### Task 9.3: Run full test suite

**Objective:** Verify all 60+ tests pass.

```bash
uv run pytest tests/ -v
```

**Verify:** All tests pass. Fix any import errors or patching issues.

---

### Phase 10: Backward-compat wrapper + cleanup

#### Task 10.1: Strip original nanobanana to thin wrapper

**Objective:** Replace the 1,965-line file with a thin PEP 723 shell.

**Files:**
- Modify: `nanobanana` (the original script)

Replace entire content with:

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
"""nanobanana — backwards-compat PEP 723 wrapper. Delegates to the package CLI."""
from nanobanana.cli import app

if __name__ == "__main__":
    app()
```

**Step 1: Write the thin wrapper.**
**Step 2: Verify:** `uv run --script ./nanobanana --help` shows help text (this won't find the package unless installed, but as a script it will since uv run picks up the project).

#### Task 10.2: Update AGENTS.md

**Objective:** Reflect the new package structure.

**Files:**
- Modify: `AGENTS.md`

Update Section 1 (Project DNA) from:
- "Single-file distribution" → "Package distribution with thin PEP 723 wrapper"
- "uv script. PEP 723 inline metadata" → "pyproject.toml with uv; thin PEP 723 script for compat"
- "All code lives in nanobanana until ~1,500-2,000 lines" → "Code lives in src/nanobanana/ package"

Update Section 2 (Code Organisation) to list the module layout:
```
src/nanobanana/
├── __init__.py
├── types.py          # Dataclasses
├── constants.py      # Exit codes, capabilities, presets, cost table
├── utils.py          # Filesystem, hashing, config, helpers
├── model_selection.py # Model resolution, auto selection, validation
├── prompt.py         # Structured prompt assembly
├── client.py         # API request construction, retry, error classification
├── response.py       # Image extraction, grounding metadata
├── output.py         # File I/O, manifest, emit_json, dry-run display
├── pipeline.py       # run_generate_pipeline orchestration
└── cli.py            # Typer app, global options, all commands
```

#### Task 10.3: Update README.md

**Objective:** Reflect uv-based package install option.

**Files:**
- Modify: `README.md`

Add a new install method:
```markdown
## Install as package

```bash
uv tool install git+https://github.com/rmax-ai/nanobanana.git
```

Or for development:
```bash
git clone https://github.com/rmax-ai/nanobanana.git
cd nanobanana
uv sync
uv run nanobanana generate "test"
```
```

Replace "No virtualenv. No pip install. No pyproject.toml. Just one file." with "Package-managed with uv. Single-file wrapper also available."

#### Task 10.4: Run final verification

```bash
# Install deps
uv sync

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Tests
uv run pytest tests/ -v

# Smoke test (dry run)
uv run nanobanana generate "test" --dry-run

# Backward compat
uv run --script ./nanobanana generate "test" --dry-run
```

**Verify:** All commands pass with exit 0.

---

### Phase 11: Commit and push

```bash
git add -A
git commit -m "refactor: split into src/nanobanana/ package with pyproject.toml

- Split 1,965-line single file into 10 modules following existing layers
- pyproject.toml replaces PEP 723 inline metadata for primary workflow
- Thin PEP 723 wrapper (nanobanana) preserved for backward compat
- Tests use normal package imports, no exec() hack
- All 60+ tests pass identically
- AGENTS.md + README.md updated"
git push origin main
```
