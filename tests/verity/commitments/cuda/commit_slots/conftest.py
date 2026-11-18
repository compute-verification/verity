from pathlib import Path

import numpy as np

#: the engine's build cache for the tests, the one the benchmarks' settings default to
BUILD = Path.home() / ".cache" / "verity" / "sha512_rows"


def salts_of(layout: object, slots: int, seed: int = 0) -> np.ndarray:
    """A step's salts for a test, ``(slots, rows, 192)`` uint8 as ``slots.fresh_salts`` gives a prover's: public test bytes
    from a seeded generator, so a test repeats; a prover's are the operating system's, drawn once."""
    rng = np.random.default_rng([seed, slots, layout.row_count])
    return rng.integers(0, 256, (slots, layout.row_count, 192), dtype=np.uint8)
