"""The reading of run.sh's runs: per operand class, the maximum sustained int8 rate over every attempt (kernel, encoding,
round) and card, alone and beside SHA-512 at each ratio, against θ = 4096 MACs a SM-clock · 132 SMs · 1830 MHz.

    python3 summarize.py OUT_DIR... [--json FILE]

An arm beside the hash counts at its ratio only where the hash's window lay inside the tensor's (`covered`) and the
ratio it achieved reached the target (to 1%): an arm whose hash fell short is listed with the ratio it reached, not
counted. `per_clock` is the rate a SM-clock times θ's clock (wgmma only, whose kernel reads its own clock)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SMS, MACS_PER_SMCLK, CLOCK_MHZ = 132, 4096, 1830
THETA = MACS_PER_SMCLK * SMS * CLOCK_MHZ * 1e6 / 1e12
CLASSES = ("s8", "s2", "u2", "u4", "zero")


def rows(out: Path) -> tuple[dict, list[dict]]:
    card = json.loads((out / "card.json").read_text())
    arms = [json.loads(line) for line in (out / "rates.jsonl").read_text().splitlines() if line.startswith("{")]
    return card, [dict(a, card=card["card"]) for a in arms]


def per_clock(arm: dict) -> float:
    return arm["macs_per_smclk"] * SMS * CLOCK_MHZ * 1e6 / 1e12


def counts(arm: dict) -> bool:
    if arm["ratio"] == 0:
        return arm.get("checked", True)
    return arm["covered"] and arm["ratio_achieved"] >= 0.99 * arm["ratio"]


def reading(arms: list[dict]) -> dict:
    """{class: {ratio: {"best": arm or None, "short": [arms whose hash fell short]}}}, each best the fastest arm."""
    out: dict = {}
    for c in CLASSES:
        mine = [a for a in arms if a["class"] == c]
        out[c] = {}
        for r in sorted({a["ratio"] for a in mine}):
            at = [a for a in mine if a["ratio"] == r]
            good = [a for a in at if counts(a)]
            out[c][r] = {"best": max(good, key=lambda a: a["tmacs_per_s"]) if good else None,
                         "short": [a for a in at if not counts(a)]}
    return out


def _arm(a: dict) -> str:
    clock = (f", {a['mhz']:.0f} MHz, {a['macs_per_smclk']:.0f} MACs a SM-clock = {per_clock(a):.1f} T MAC/s at "
             f"{CLOCK_MHZ} MHz" if "mhz" in a else "")
    hash_ = f", hash {a['ratio_achieved']:.3f} M/T MAC" if a["ratio"] else ""
    if a.get("tensor_paced", 1) < 1:
        hash_ += f", tensor paced to {a['tensor_paced']:.3f} of alone"
    return f"{a['kernel']} {a['enc']} r{a['round']} on {a['card']}{clock}{hash_}"


def render(card_rows: list[tuple[dict, list[dict]]]) -> str:
    lines = [f"theta = {THETA:.1f} T MAC/s ({MACS_PER_SMCLK} MACs a SM-clock, {SMS} SMs, {CLOCK_MHZ} MHz)"]
    for card, _ in card_rows:
        lines.append(f"card {card['card']} {card.get('uuid')}: clock {card['clock_mhz']} MHz, locked {card['clock_locked']}")
    scopes = [(card["card"], arms) for card, arms in card_rows]
    if len(scopes) > 1:
        scopes.append(("all cards", [a for _, arms in card_rows for a in arms]))
    for scope, arms in scopes:
        lines.append(f"\n{scope}:")
        for c, by_ratio in reading(arms).items():
            for r, got in by_ratio.items():
                b = got["best"]
                head = f"  {c:>4} at {r:g} M/T MAC: "
                if b is None:
                    lines.append(head + "no arm counts")
                else:
                    lines.append(head + f"{b['tmacs_per_s']:.1f} T MAC/s = {b['tmacs_per_s'] / THETA:.4f} theta ({_arm(b)})")
                for a in got["short"]:
                    lines.append(f"        not counted: {a['tmacs_per_s']:.1f} T MAC/s, {_arm(a)}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("outs", nargs="+", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    card_rows = [rows(o) for o in a.outs]
    print(render(card_rows))
    if a.json:
        everything = [x for _, arms in card_rows for x in arms]
        a.json.write_text(json.dumps({"theta": THETA, "cards": [c for c, _ in card_rows],
                                      "reading": {c: {str(r): v for r, v in by.items()}
                                                  for c, by in reading(everything).items()}}, indent=1))


if __name__ == "__main__":
    main()
