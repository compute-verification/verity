"""``verity_flock.record_text`` for the agreement scripts, which run by path where the package may not be installed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from verity_flock.record_text import session_text  # noqa: F401
