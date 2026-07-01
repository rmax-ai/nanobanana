import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

nanobanana_mod = types.ModuleType("nanobanana")
# Register the module before exec so dataclass string-annotation evaluation can
# resolve forward references against the module namespace.
sys.modules["nanobanana"] = nanobanana_mod
source = Path(PROJECT_ROOT, "nanobanana").read_text()
exec(source.split("if __name__")[0], nanobanana_mod.__dict__)

# Expose all public names and single-underscore helpers used by tests at module level.
for name in dir(nanobanana_mod):
    if not name.startswith("__"):
        globals()[name] = getattr(nanobanana_mod, name)

# Bind names used by fixtures so static linters can resolve them.
ImageRequest = nanobanana_mod.ImageRequest
load_reference = nanobanana_mod.load_reference

# Make the conftest module importable as `conftest` from test modules.
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
        self.Client = MagicMock()


@pytest.fixture
def mock_client(monkeypatch):
    mock_genai = MockGenai()
    # Functions in the execed nanobanana module look up `genai` in their own
    # module namespace, so patch that rather than the conftest namespace.
    monkeypatch.setattr(sys.modules["nanobanana"], "genai", mock_genai)
    return mock_genai
