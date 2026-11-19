"""Refuses unless every GPU named (nvidia-smi's names) has the card file whose capability and SM count are the target a
served row of record registers (`adapters/vllm/workloads/generation.toml`): `python3 row_card.py ROW NAME...`, with the
tree's root on PYTHONPATH."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from verity.device import compute_capability, gpu_card_reporting, number

ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "calibration/hardware/cards"
ROWS = ROOT / "adapters/vllm/workloads/generation.toml"


def main(row: str, names: list[str]) -> int | str:
    rows = tomllib.loads(ROWS.read_text())["row"]
    if row not in rows or "target" not in rows[row]:
        return f"row_card.py: {row!r} has no target in {ROWS}"
    want = rows[row]["target"]
    if not names:
        return f"row_card.py: no GPU named: {row} serves on compute capability {want['compute_capability']} with {want['num_sms']} SMs"
    for name in names:
        try:
            card = gpu_card_reporting(name, directory=CARDS)
        except KeyError as e:
            return f"row_card.py: {e.args[0]}"
        have = [list(compute_capability(card)), int(number(card["sm"]["count"]))]
        if have != [list(want["compute_capability"]), want["num_sms"]]:
            return (f"row_card.py: {name} is {card['sku']} (compute capability {have[0]}, {have[1]} SMs), and {row} registers "
                    f"compute capability {want['compute_capability']} with {want['num_sms']} SMs ({ROWS}): run it on that card")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
