import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

# Load the single-file nanobanana script as an importable module for tests.
NANOBANANA_PATH = Path(__file__).parent.parent / "nanobanana"
spec = importlib.util.spec_from_file_location(
    "nanobanana",
    NANOBANANA_PATH,
    loader=SourceFileLoader("nanobanana", str(NANOBANANA_PATH)),
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load nanobanana from {NANOBANANA_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules["nanobanana"] = module
spec.loader.exec_module(module)
