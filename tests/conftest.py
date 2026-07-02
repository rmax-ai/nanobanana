# Make 'import conftest' work in test modules.
import sys
from unittest.mock import MagicMock

import pytest

from nanobanana.cli import app  # noqa: F401
from nanobanana.client import (  # noqa: F401
    build_interaction_input,
    classify_api_exception,
    execute_request,
    is_retryable,
    is_safety_refusal,
    with_retry,
)

# Re-export everything the test file imports.
from nanobanana.constants import (  # noqa: F401
    CAPABILITIES,
    COST_TABLE,
    PRESETS,
    ExitCode,
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
from nanobanana.output import (  # noqa: F401
    atomic_write,
    generate_filename,
    resolve_output_path,
    write_manifest,
)
from nanobanana.pipeline import run_generate_pipeline  # noqa: F401
from nanobanana.prompt import (  # noqa: F401
    annotate_references,
    build_edit_instruction,
    build_normalized_prompt,
    expand_negative,
)
from nanobanana.response import (  # noqa: F401
    extract_grounding_metadata,
)
from nanobanana.types import (  # noqa: F401
    GeneratedAsset,
    ImageRequest,
    ModelDecision,
    ReferenceImage,
)
from nanobanana.utils import (  # noqa: F401
    detect_mime_type,
    load_reference,
    sha256_bytes,
    sha256_file,
    slugify,
)

sys.modules.setdefault("conftest", sys.modules[__name__])


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_request():
    return ImageRequest(command="generate", prompt="a red banana", model="auto")


@pytest.fixture
def sample_reference(temp_dir):
    path = temp_dir / "ref.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return load_reference(path, role="style")


@pytest.fixture
def mock_client(monkeypatch):
    """Mock google.genai.Client for contract tests."""
    mock_client_instance = MagicMock()
    mock_genai_module = MagicMock()
    mock_genai_module.Client = MagicMock(return_value=mock_client_instance)
    # Patch genai in both modules that import it.
    monkeypatch.setattr("nanobanana.client.genai", mock_genai_module)
    monkeypatch.setattr("nanobanana.pipeline.genai", mock_genai_module)
    return mock_genai_module
