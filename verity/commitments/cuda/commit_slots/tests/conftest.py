import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: the engine's build cache for the tests, the one the benchmarks' settings default to
BUILD = Path.home() / ".cache" / "verity" / "sha512_rows"
