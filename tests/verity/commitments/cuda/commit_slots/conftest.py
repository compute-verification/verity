from pathlib import Path

#: the engine's build cache for the tests, the one the benchmarks' settings default to
BUILD = Path.home() / ".cache" / "verity" / "sha512_rows"
