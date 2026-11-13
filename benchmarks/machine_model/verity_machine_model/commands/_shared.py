"""What several commands share: the `--device` option's help, a percentage, and a design's registry entry."""
from __future__ import annotations

from verity_machine_model import eps as E
from verity_machine_model.model import devices


def device_help() -> str:
    return f"the device: {', '.join(devices())} (default sm_120, data/ itself; the others data/devices/sm_NN/)"


def pct(x: float | None) -> str:
    return "unmeasured" if x is None else f"{100 * x:.3f}%"


def registry(d: E.Design) -> dict:
    return {"name": d.name, "title": d.title, "aliases": list(d.aliases), "base": d.base or None,
            "summary": d.summary, "noise": d.noise, "sums": d.sums, "fold": {"name": d.fold_name, **d.fold},
            "filler": [{"per_word": f.per_word, "mix": f.mix, "route": f.route} for f in d.filler], "L": d.L,
            "unit": d.unit, "coins": {"bits": d.coin_bits, "source": d.coins_source}, "ppl": d.ppl or None}
