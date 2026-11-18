from __future__ import annotations

from pathlib import Path

import numpy as np

from verity.device import compute_capability, gpu_card_reporting

#: the engine's build cache for the tests, the one the benchmarks' settings default to
BUILD = Path.home() / ".cache" / "verity" / "sha512_rows"

#: the GPU card files (`verity.device`), whose ``arch`` is the target the device library is built for
CARD_FILES = Path(__file__).resolve().parents[5] / "calibration/hardware/cards"


def arch(name: str) -> str:
    """The target a library is built for on the device reporting ``name`` (torch's ``get_device_name``): its card file's
    ``arch`` (`verity.device.gpu_card_reporting`) without ``sm_``, such as ``90a``; a device no card file names is refused."""
    card = gpu_card_reporting(name, directory=CARD_FILES)
    compute_capability(card)
    return card["arch"].removeprefix("sm_")


def target(torch: object, device: str) -> str | None:
    """The target a plan on ``device`` is built for: none on the host, whose twin is no card's and asks torch for no
    device's name; on a CUDA device, the card file's of the name the device reports (`arch`)."""
    return None if device == "host" else arch(torch.cuda.get_device_name(device))


def salts_of(layout: object, slots: int, seed: int = 0) -> np.ndarray:
    """A step's salts for a test, ``(slots, rows, 192)`` uint8 as ``slots.fresh_salts`` gives a prover's: public test bytes
    from a seeded generator, so a test repeats; a prover's are the operating system's, drawn once."""
    rng = np.random.default_rng([seed, slots, layout.row_count])
    return rng.integers(0, 256, (slots, layout.row_count, 192), dtype=np.uint8)
