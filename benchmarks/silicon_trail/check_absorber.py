"""Check that the `plane` step of keccak.py is the synthesized absorber's.

    python3 check_absorber.py --rtl DIR [--vectors N]

DIR holds the absorber's Verilog as its synthesis run stored it: `gov_ft.v` (module ft_plane4: four rounds a clock,
256 input bits each) and `synth/keccak1600_rnd.v` (the round).  Both are flat continuous assignments; this evaluates
them directly (no simulator) on random vectors and compares with `keccak.chain(state, 4 inputs, 64, "plane", rc)`.
It also reports what the Verilog fixes: which state bits the input enters, and that the round constant is an input
port, which the absorber leaves to its surroundings.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import keccak as K


def parse_module(text: str, name: str) -> dict:
    m = re.search(rf"module\s+{name}\s*\((.*?)\);(.*?)endmodule", text, re.S)
    assert m, f"module {name} not found"
    ports = {}
    for kind, width, pname in re.findall(r"(input|output)\s*(?:\[(\d+):0\])?\s*(\w+)", m.group(1)):
        ports[pname] = (kind, int(width) + 1 if width else 1)
    return {"name": name, "ports": ports, "body": m.group(2)}


def to_python(expr: str, widths: dict[str, int]) -> str:
    """A Verilog expression of ^ & ~ ( ) part-selects and concatenations, as a Python expression over ints."""
    def part(p: str) -> tuple[str, int]:
        p = p.strip()
        if m := re.fullmatch(r"(\d+)'b0", p):
            return "0", int(m.group(1))
        if m := re.fullmatch(r"(\w+)\[(\d+):(\d+)\]", p):
            hi, lo = int(m.group(2)), int(m.group(3))
            return f"(({m.group(1)} >> {lo}) & {(1 << (hi - lo + 1)) - 1})", hi - lo + 1
        assert re.fullmatch(r"\w+", p), p
        return p, widths[p]

    def concat(m: re.Match) -> str:
        out = "0"
        for p in m.group(1).split(","):
            v, n = part(p)
            out = f"(({out} << {n}) | {v})"
        return out

    expr = re.sub(r"\{([^{}]*)\}", concat, expr)
    expr = re.sub(r"(?<![\w)])(\w+)\[(\d+):(\d+)\]", lambda m: part(m.group(0))[0], expr)
    expr = re.sub(r"~(\w+)", lambda m: f"({m.group(1)} ^ {(1 << widths[m.group(1)]) - 1})", expr)
    return expr


def eval_flat(mod: dict, inputs: dict[str, int]) -> dict[str, int]:
    env = dict(inputs)
    widths = {p: n for p, (_, n) in mod["ports"].items()}
    for stmt in mod["body"].split(";"):
        stmt = stmt.strip()
        if m := re.fullmatch(r"wire\s*\[(\d+):0\]\s*(\w+)\s*=\s*(.*)", stmt, re.S):
            widths[m.group(2)] = int(m.group(1)) + 1
            env[m.group(2)] = eval(to_python(m.group(3), widths), {}, env) & ((1 << widths[m.group(2)]) - 1)
        elif m := re.fullmatch(r"assign\s+(\w+)\s*=\s*(.*)", stmt, re.S):
            env[m.group(1)] = eval(to_python(m.group(2), widths), {}, env)
        else:
            assert not stmt, f"unhandled statement: {stmt[:80]}"
    return env


def eval_top(top: dict, rnd: dict, inputs: dict[str, int]) -> int:
    env = dict(inputs)
    widths = {p: n for p, (_, n) in top["ports"].items()}
    for stmt in top["body"].split(";"):
        stmt = stmt.strip()
        if m := re.fullmatch(r"wire\s*\[(\d+):0\]\s*([\w,\s]+)", stmt):
            for nm in m.group(2).split(","):
                widths[nm.strip()] = int(m.group(1)) + 1
        elif m := re.fullmatch(r"keccak1600_rnd\s+\w+\s*\((.*)\)", stmt, re.S):
            conn = dict(re.findall(r"\.(\w+)\(((?:[^()]|\([^()]*\))*)\)", m.group(1)))
            args = {p: eval(to_python(e, widths), {}, env) for p, e in conn.items() if rnd["ports"][p][0] == "input"}
            out = eval_flat(rnd, args)
            env[conn["y"].strip()] = out["y"]
        else:
            assert not stmt, f"unhandled statement: {stmt[:80]}"
    return env["y"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rtl", required=True, type=Path)
    ap.add_argument("--vectors", type=int, default=200)
    a = ap.parse_args()
    rnd = parse_module((a.rtl / "synth" / "keccak1600_rnd.v").read_text(), "keccak1600_rnd")
    top = parse_module((a.rtl / "gov_ft.v").read_text(), "ft_plane4")
    print("round ports:", {p: v for p, v in rnd["ports"].items()})
    print("absorber ports:", {p: v for p, v in top["ports"].items()})

    rng = random.Random(7)
    m256 = (1 << 256) - 1
    for _ in range(a.vectors):
        st, m, rc = rng.getrandbits(1600), rng.getrandbits(1024), rng.getrandbits(256)
        got = eval_top(top, rnd, {"st": st, "m": m, "rc": rc})
        s = K.from_int(st, 64)
        for j in range(4):
            s = K.step(s, (m >> (256 * j)) & m256, 64, "plane", (rc >> (64 * j)) & ((1 << 64) - 1))
        assert got == K.to_int(s, 64), "keccak.step(plane) differs from the absorber's Verilog"
    print(f"ft_plane4 == four `plane` steps of keccak.py on {a.vectors} random (state, input, constants) vectors")

    # which state bits the input enters: flip each input bit of the first block, with the rounds removed
    flat = eval_flat(rnd, {"st": 0, "m": m256, "rc": 0})
    entered = flat["s"]
    lanes = [i for i in range(25) if (entered >> (64 * i)) & ((1 << 64) - 1)]
    assert entered == m256 and lanes == [0, 1, 2, 3]
    print("the 256 input bits enter state bits 0..255: lanes (x, y) = (0,0), (1,0), (2,0), (3,0), all 64 bits of each;"
          " lane (4,0) takes none")
    print("the round constant is the input port rc (64 bits a round): the absorber's Verilog does not fix its value")


if __name__ == "__main__":
    main()
