"""Adversarial VU batches for the A-GKR full-VU prover (``verity-gkr run-vu``): chain-linkage forgeries.

Every case is a complete ``run-vu`` directory (``circuit.txt``, ``epilogue.txt``, ``units.bin``, ``epilogue.bin``,
``public.bin``) built from real K=1536 VUs (the bases of ``vu-k1536-neg`` with their *correct* words, so no fixture
larger than the committed negatives is needed) with one adversarial edit.  Unlike the ``vu-k1536-neg`` negatives
(a wrong *claimed* word with honest hints), these are adversarial *witnesses*: every unit row satisfies the unit
circuit on its own (``verify_unit`` = None) and the forger claims the word the edited chain leads to.  What must
catch each case is therefore exactly the chain / endpoint machinery of PROTOCOL.md 12.2 (the linear constraints
batched into the Ligero opening) or the epilogue circuit's assertion -- nothing else.

Cases (``expected`` is always ``reject``; ``mechanism`` names the constraint that must fire):

* ``a_splice``          unit k of VU 1 takes ``x.c`` from VU 0's unit k-1 output (a valid unit; the rest of VU 1 is
                        re-chained honestly from the spliced word); public word = the spliced chain's cast.
                        Must fail the linkage constraint ``X[(1,k),c] - X[(1,k-1),y] = 0``.
* ``a_epilogue_y32``    VU 0's epilogue row is generated from a different FP32 word whose RN-even cast is
                        ``true_word + 1``; public word = ``true_word + 1``.  Every row is self-consistent; only
                        ``X_epi[0,y32] - X[(0,95),y] = 0`` fails.
* ``a_epilogue_claim``  honest epilogue hints for the true word but ``y16`` column = ``true_word + 1`` and the public
                        word = ``true_word + 1``.  Only the epilogue circuit's assertion ``y16 = hi + carry`` fails.
* ``a_public_word``     everything honest, public word of VU 0 = ``true_word ^ 1`` (the endpoint constraint).
* ``a_swap_words``      honest witnesses, public words of VU 0 and VU 1 exchanged (each a valid word for a real VU).
* ``a_c0_nonzero``      VU 1's unit 0 regenerated from ``c = VU 0's last word`` (a valid unit); public word = that
                        chain's cast.  Must fail ``X[(1,0),c] = 0``.
* ``a_honest``          control: the untouched batch (expected ``accept``).

The ``+p`` accumulator shadow of the BabyBear battery (``chain`` class) has no Goldilocks counterpart: ``x.c`` and ``y``
are single columns holding a 32-bit word, ``p > 2^32``, and ``c' = y + p`` is the *same* field element as ``y``.  It is
recorded as an argument, not a directory (``manifest.json: not_representable``).

    python -m verity_numerical.redteam.vu_forge_a --out DIR [--vus 2] [--root fixtures/bench-instances/v1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from verity_numerical.checker import REAL, Params, Reject, gen_step
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.step import step_output
from verity_numerical.gkr_export.circuit import (
    GOLDILOCKS,
    build_unit_circuit,
    export_circuit,
    unit_witness,
    verify_unit,
)
from verity_numerical.gkr_export.vu import (
    STEP,
    build_epilogue_circuit,
    epilogue_witness,
    write_rows,
)

__all__ = ["build_cases", "main"]

VERSION = "v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _read_u16(path: Path) -> list[int]:
    b = path.read_bytes()
    return list(struct.unpack(f"<{len(b) // 2}H", b))


def load_real_vus(root: Path, n: int) -> list[tuple[list[int], list[int], int]]:
    """The first ``n`` bases of ``vu-k1536-neg`` with their correct words."""
    x = _read_u16(root / "vu-k1536-neg.x.u16")
    w = _read_u16(root / "vu-k1536-neg.w.u16")
    idx = json.loads((root / "vu-k1536-neg.index.json").read_text())
    cols = idx["columns"]
    k = 16 * 96
    out = []
    for i in range(n):
        row = dict(zip(cols, idx["rows"][i]))
        if row["correct_y"] is None:
            continue
        out.append((x[i * k : (i + 1) * k], w[i * k : (i + 1) * k], int(row["correct_y"])))
    return out


class Chain:
    """One VU as unit rows + epilogue row, re-chainable from any step with any incoming word."""

    def __init__(self, unit, epi, a: list[int], b: list[int], p: Params = REAL):
        self.unit, self.epi, self.a, self.b, self.p = unit, epi, a, b, p
        self.steps = len(a) // STEP
        self.rows: list[list[int]] = []
        self.words: list[int] = []  # y of every unit
        self.rechain(0, 0)

    def unit_row(self, i: int, c: int) -> tuple[list[int], int]:
        ai, bi = self.a[i * STEP : (i + 1) * STEP], self.b[i * STEP : (i + 1) * STEP]
        h = gen_step(self.p, c, ai, bi)
        y = step_output(self.p, c, ai, bi)
        row = unit_witness(self.unit[0], self.unit[1], c, ai, bi, y, h)
        assert verify_unit(self.unit[0], row) is None, "a forged unit must still satisfy the unit circuit"
        return row, y

    def rechain(self, start: int, c: int) -> None:
        """Regenerate units ``start..`` honestly from incoming word ``c`` (the forger's splice point)."""
        rows, words = self.rows[:start], self.words[:start]
        for i in range(start, self.steps):
            row, y = self.unit_row(i, c)
            rows.append(row)
            words.append(y)
            c = y
        self.rows, self.words = rows, words

    @property
    def y32(self) -> int:
        return self.words[-1]

    def cast(self, y32: int | None = None) -> int:
        y32 = self.y32 if y32 is None else y32
        return int(Epilogue.check(IntCtx(), self.p, y32, Epilogue.gen(self.p, y32)))

    def epilogue_row(self, y32: int | None = None, y16: int | None = None) -> list[int]:
        y32 = self.y32 if y32 is None else y32
        y16 = self.cast(y32) if y16 is None else y16
        return epilogue_witness(self.epi[0], self.epi[1], y32, y16, Epilogue.gen(self.p, y32))


def _fp32_with_cast(target16: int) -> int:
    """An FP32 word whose RN-even BF16 cast is ``target16``: the exact word ``target16 << 16``."""
    return target16 << 16


def _write_case(out: Path, unit, epi, chains: list[Chain], epi_rows: list[list[int]], public: list[int]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    export_circuit(unit[0], out / "circuit.txt")
    export_circuit(epi[0], out / "epilogue.txt")
    write_rows([r for ch in chains for r in ch.rows], out / "units.bin")
    write_rows(epi_rows, out / "epilogue.bin")
    write_rows([[y] for y in public], out / "public.bin")
    return {f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in ("units.bin", "epilogue.bin", "public.bin")}


def build_cases(out: Path, root: Path, n_vus: int = 2, p: Params = REAL) -> dict:
    unit = build_unit_circuit(p)
    epi = build_epilogue_circuit(p)
    vus = load_real_vus(root, n_vus)
    if len(vus) < 2:
        raise Reject("need at least two real VUs")
    honest = [Chain(unit, epi, a, b, p) for a, b, _ in vus]
    for ch, (_, _, y16) in zip(honest, vus):
        assert ch.cast() == y16, "the checker's chain must reproduce the recorded correct word"
    words = [ch.cast() for ch in honest]
    cases: list[dict] = []

    def emit(cid: str, chains: list[Chain], epi_rows: list[list[int]], public: list[int], expected: str, mechanism: str, desc: str) -> None:
        d = out / cid
        shas = _write_case(d, unit, epi, chains, epi_rows, public)
        cases.append({"id": cid, "dir": cid, "expected": expected, "mechanism": mechanism, "mutation": desc,
                      "vus": len(chains), "steps": chains[0].steps, "public_y16": public, "honest_y16": words[: len(chains)], "sha256": shas})

    # control
    emit("a_honest", honest, [ch.epilogue_row() for ch in honest], words, "accept", "-", "untouched batch (control)")

    # splice: VU 1 unit k takes VU 0's word k-1
    k = 40
    sp = Chain(unit, epi, vus[1][0], vus[1][1], p)
    sp.rechain(k, honest[0].words[k - 1])
    assert sp.words != honest[1].words
    chains = [honest[0], sp] + honest[2:]
    emit("a_splice", chains, [ch.epilogue_row() for ch in chains], [words[0], sp.cast()] + words[2:], "reject",
         "linkage X[(1,k),c] - X[(1,k-1),y] = 0 (rho_geo-batched linear constraint in the Ligero opening)",
         f"VU 1 unit {k}: x.c := VU 0's y_{k - 1} = {honest[0].words[k - 1]:#010x} (was {honest[1].words[k - 1]:#010x}); units {k}.. re-chained honestly; claimed word {sp.cast():#06x} (true {words[1]:#06x})")

    # epilogue fed a different y32 whose cast is word+1
    w1 = (words[0] + 1) & 0xFFFF
    y32_alt = _fp32_with_cast(w1)
    assert honest[0].cast(y32_alt) == w1
    emit("a_epilogue_y32", honest, [honest[0].epilogue_row(y32_alt, w1)] + [ch.epilogue_row() for ch in honest[1:]], [w1] + words[1:], "reject",
         "linkage X_epi[0,y32] - X[(0,95),y] = 0",
         f"VU 0 epilogue row generated from y32 = {y32_alt:#010x} (true {honest[0].y32:#010x}); claimed word {w1:#06x}")

    # epilogue claim column differs from the hints' cast
    emit("a_epilogue_claim", honest, [honest[0].epilogue_row(None, w1)] + [ch.epilogue_row() for ch in honest[1:]], [w1] + words[1:], "reject",
         "epilogue circuit assertion y16 = hi + carry (GKR), not the linkage",
         f"VU 0 epilogue hints honest for {honest[0].y32:#010x}, y16 column and public word = {w1:#06x}")

    # public word only
    emit("a_public_word", honest, [ch.epilogue_row() for ch in honest], [words[0] ^ 1] + words[1:], "reject",
         "endpoint X_epi[0,y16] = word_0", f"public word of VU 0 = {words[0] ^ 1:#06x}, witness honest for {words[0]:#06x}")

    # swapped words
    emit("a_swap_words", honest, [ch.epilogue_row() for ch in honest], [words[1], words[0]] + words[2:], "reject",
         "endpoint constraints of VU 0 and VU 1", "public words of VU 0 and VU 1 exchanged, witnesses honest")

    # c_0 != 0 for VU 1 (a valid unit chained from VU 0's final word)
    c0 = Chain(unit, epi, vus[1][0], vus[1][1], p)
    c0.rechain(0, honest[0].y32)
    chains = [honest[0], c0] + honest[2:]
    emit("a_c0_nonzero", chains, [ch.epilogue_row() for ch in chains], [words[0], c0.cast()] + words[2:], "reject",
         "start constraint X[(1,0),c] = 0", f"VU 1 unit 0: x.c := VU 0's y32 = {honest[0].y32:#010x}; claimed word {c0.cast():#06x} (true {words[1]:#06x})")

    manifest = {
        "version": VERSION, "target": "verity-gkr run-vu (Candidate A full VU, PROTOCOL.md 12)",
        "source": "vu-k1536-neg bases 0..n with their correct words (fixtures/bench-instances/v1)",
        "field": {"name": "Goldilocks", "p": GOLDILOCKS}, "steps": 96, "vus": len(honest), "cases": cases,
        "not_representable": {
            "a_plus_p_shadow": "x.c and y are single Goldilocks columns holding a 32-bit word; c' = y + p is the same field element as y, "
                               "so the BabyBear battery's `chain` class (accumulator handed over as c + j p) has no distinct witness. "
                               "A non-word x.c (>= 2^32) is rejected by Fp32Decode's range checks (acc.t < 255 etc.) at witness generation.",
        },
        "run": "for c in cases: verity-gkr run-vu --dir DIR/<id> [--clear]; expected verdicts in `expected`",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=_repo_root() / "fixtures/bench-instances/v1")
    ap.add_argument("--vus", type=int, default=2)
    args = ap.parse_args(argv)
    m = build_cases(args.out, args.root, args.vus)
    print(json.dumps({"cases": [(c["id"], c["expected"]) for c in m["cases"]], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
