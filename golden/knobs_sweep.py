"""GD-KNOBS: one-at-a-time sensitivity of J to each knob, on GD-MODEL's golden/model.py (an estimate, not a measurement).

Run from the root of a checkout that has GD-MODEL's golden/ (cursor/compute-research-gd-model-6eb4, merged at 1fa3c67b20):
python3 -m golden.knobs_sweep [--json OUT] [--sha512 ROWS]

--sha512 ROWS prices a SHA-512 compression at ROWS in circuit (and hm96's hiding part at twice that) instead of the
model's INPUTS, e.g. 81402, C-Flock's SHA-512 circuit (the coordinator, 20:00Z).
"""
from __future__ import annotations

import copy
import json
import sys

from golden import model as M

PRESETS = ("demo", "throughput")

#: (knob, section, key, values); None is the model's default
KNOBS = [
    ("partition tile r x c (a linear's unit: r tokens x c columns; 1 x 1 is Q_word X = 16)", "partition", "tile", [[1, 1], [1, 4], [4, 4], [8, 8], [16, 16], [32, 32]]),
    ("partition K chunks (a unit below one coordinate)", "partition", "kchunk", [1, 2, 4, 8, 16]),
    ("Commit row bytes", "commit", "row_bytes", [512, 1024, 2048, 4096, 8192, 16384]),
    ("Commit open", "commit", "open", ["whole", "exact"]),
    ("Commit values (fusion)", "commit", "values", ["every_op", "dedup", "fused_mlp"]),
    ("Commit bits a bf16 word", "commit", "bits_per_word", [8, 12, 16]),
    ("Dither L", "pouw", "L", [256, 512, 1024, 1536]),
    ("Dither L' (noise period)", "pouw", "L_prime", [32, 64, 128, 256]),
    ("Dither value row bytes", "pouw", "value_row_bytes", [1024, 2048, 4096, 8192, 16384, 32768]),
    ("Dither word bits", "pouw", "word_bits", [8, 12, 16, 24, 32]),
    ("Dither noise", "pouw", "noise", ["periodic", "fresh", "permuted"]),
    ("sampling law", "sampling", "law", ["achievability", "uniform"]),
    ("C-Flock m_max", "prover", "m_max", [33, 35, 37, 40]),
    ("statements", "prover", "stmt", ["template", "universal"]),
    ("steps a window", "point", "steps_per_window", [1, 4, 16, 256]),
    ("k_log cap (prover.k_max)", "prover", "k_max", [27, 28, 30]),
    ("C-Flock range padding (prover.k_pad, bits past ceil(log2 ANDs))", "prover", "k_pad", [0, 1, 2]),
]

#: INPUTS overrides: (knob, input, values)
INPUT_KNOBS = [
    ("hiding part in circuit (hm96_ands; 0 ~ a PRF leaf whose salt fits the row's blocks)", "hm96_ands", [0, 58_120, 116_240]),
    ("Dither noise amplitude (noise_amp)", "noise_amp", [16, 40, 80, 127]),
]

#: a J10-R proof block's (R, C): 4 x 8 (the scheme's least, 4 | r, 8 | c) ... 32 x 32 (H1's warp)
H1_TILES = [(4, 8), (8, 8), (8, 16), (16, 16), (16, 32), (32, 32)]


def h1_tile(preset: str, form: str, tile: tuple[int, int]) -> dict:
    r = M.J(M.design(preset, pouw={"scheme": "j10r", "tile": tile, "form": form}))
    return {"value": [form, *tile], "J": r["J"], "serve": r["serve_term"], "prove": r["prove_term"],
            "infeasible": r.get("infeasible") or None}


def run(preset: str, sec: str | None, key: str, value, inp: bool = False) -> dict:
    saved = copy.deepcopy(M.INPUTS)
    try:
        if inp:
            v = M.INPUTS[key]
            M.INPUTS[key] = (value, *v[1:]) if isinstance(v, tuple) else value
            d = M.design(preset)
        else:
            d = M.design(preset, **{sec: {key: value}})
        r = M.J(d)
    except Exception as e:  # a design the model refuses
        return {"value": value, "error": f"{type(e).__name__}: {e}"[:200]}
    finally:
        M.INPUTS.clear()
        M.INPUTS.update(saved)
    return {"value": value, "J": r["J"], "serve": r["serve_term"], "prove": r["prove_term"],
            "infeasible": r.get("infeasible") or None}


def main() -> None:
    if "--sha512" in sys.argv:
        rows = int(sys.argv[sys.argv.index("--sha512") + 1])
        for key, n in (("sha512_ands", rows), ("hm96_ands", 2 * rows)):
            M.INPUTS[key] = (n, "E", f"--sha512 {rows}: C-Flock's SHA-512 circuit, rows a compression")
    out = {}
    for preset in PRESETS:
        base = M.J(M.design(preset))
        out[preset] = {"base": {"J": base["J"], "serve": base["serve_term"], "prove": base["prove_term"]}, "knobs": {}}
        for name, sec, key, values in KNOBS:
            out[preset]["knobs"][name] = [run(preset, sec, key, v) for v in values]
        for name, key, values in INPUT_KNOBS:
            out[preset]["knobs"][name] = [run(preset, None, key, v, inp=True) for v in values]
        out[preset]["knobs"]["H1 proof block cells under J10-R, sampled (tile | lanes)"] = [
            h1_tile(preset, form, c) for form in ("tile", "lanes") for c in H1_TILES]
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f, indent=1, default=str)
    for preset, o in out.items():
        b = o["base"]
        print(f"== {preset}: base J {b['J']:.4g} (serve {b['serve']:.4g}, prove {b['prove']:.4g})")
        for name, rows in o["knobs"].items():
            cells = []
            for r in rows:
                if "error" in r:
                    cells.append(f"{r['value']}: refused ({r['error'][:60]})")
                else:
                    flag = " INFEASIBLE" if r["infeasible"] else ""
                    cells.append(f"{r['value']}: J {r['J']:.4g} (s {r['serve']:.3g}, p {r['prove']:.4g}){flag}")
            print(f"  {name}")
            for c in cells:
                print(f"    {c}")
    print("\n== summary: each knob's best feasible value, and J's change against the preset's default (* infeasible)")
    for preset, o in out.items():
        b = o["base"]["J"]
        print(f"-- {preset} (J {b:.4g})")
        for name, rows in o["knobs"].items():
            ok = [r for r in rows if "J" in r]
            feas = [r for r in ok if not r["infeasible"]]
            best = min(feas, key=lambda r: r["J"]) if feas else None
            spread = " ".join(f"{r['value']}:{100 * (r['J'] / b - 1):+.0f}%{'*' if r['infeasible'] else ''}" for r in ok)
            head = f"best {best['value']} ({100 * (best['J'] / b - 1):+.1f}%)" if best else "none feasible"
            print(f"  {name[:60]:60s} {head:28s} {spread}")


if __name__ == "__main__":
    main()
