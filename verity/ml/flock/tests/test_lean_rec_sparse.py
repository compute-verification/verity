"""`Flock.RecSparse`, the Lean verifier's own `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` unit: at the staged shapes its
`flock-ir-unit/v2` text (`flock-rec-sparse net`) is `rec_sparse.lowering`'s (`ir_lower.netlist`) byte for byte, and
`RecSparse.check` accepts the composed circuit file and refuses it with a row, the name or the shape changed."""
from __future__ import annotations

import hashlib
import json
import resource
import subprocess
from pathlib import Path

import pytest
from verity_flock import circuit as FC
from verity_flock import ir_lower as IL
from verity_flock import rec_sparse as RS

BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-rec-sparse"

#: the staged shapes: the binding's, one template and one slot, two slots of one template, four of two templates
SHAPES = [(2, 6, 1, 1), (2, 6, 0, 0), (1, 6, 2, 0), (2, 6, 0, 1)]


def _needs_lean():
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")


@pytest.mark.slow
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "n{}-e{}-s{}-t{}".format(*s))
def test_the_verifiers_net_is_the_lowerings_byte_for_byte(shape):
    """Lean builds the unit from its name alone (the shape, the ports, the products' S leaf-summed, the claim chains and Φ's
    tree with their operands committed, the τ demultiplexer, the layout) and writes the same text as the Python lowering."""
    _needs_lean()
    sh = RS.Shape(*shape)
    low = RS.lowering(sh)
    want = IL.netlist(low.layout, low.name)
    p = subprocess.run([str(BIN), "net", RS.unit_name(sh)], capture_output=True, check=False)
    assert p.returncode == 0, p.stderr.decode()
    assert hashlib.sha256(p.stdout).hexdigest() == hashlib.sha256(want.encode()).hexdigest(), _first_diff(p.stdout.decode(), want)


def _first_diff(got: str, want: str) -> str:
    g, w = got.splitlines(), want.splitlines()
    for i, (a, b) in enumerate(zip(g, w)):
        if a != b:
            return f"line {i}: lean {a[:200]!r} python {b[:200]!r}"
    return f"lean {len(g)} lines, python {len(w)}"


def _check(path: Path, *names: str) -> list[dict]:
    """`flock-rec-sparse check` on the file's unit and on it renamed to each name, in an address space of 8 GiB."""
    p = subprocess.run([str(BIN), "check", str(path), *names], capture_output=True, text=True, check=False,
                       preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30)))
    got = [json.loads(ln) for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert len(got) == 1 + len(names), p.stdout + p.stderr
    return got


@pytest.mark.slow
def test_check_accepts_the_composed_unit_and_refuses_a_changed_one(tmp_path):
    """`RecSparse.check` on the unit's circuit file at (2,6,0,1), as `rec_sparse.statement` composes it: the unit is accepted;
    with one row's operands swapped it is refused at that row; renamed to another shape it is refused for its bits before
    that shape's net is built, to `rec-sparse/n1000000000-…` for its shape before any of the shape's sizes is computed, out
    of the name's form, and to an invalid shape, each by name."""
    _needs_lean()
    sh = RS.Shape(2, 6, 0, 1)
    lines = FC.compose(RS.lowering(sh), RS.subcircuit(sh), partition="ab" * 64, program="cd" * 64).split("\n")
    (tmp_path / "honest.txt").write_text("\n".join(lines))
    big, other = RS.Shape(10**9, 6, 0, 0), RS.Shape(2, 6, 1, 1)
    names = [RS.unit_name(other), RS.unit_name(big), "rec-sparse/n2-e6-s0", "rec-sparse/n2-e5-s0-t1"]
    ins, outs = 16 * sum(n for _, n in RS.ports(sh)), 16 * RS.out_words(sh)
    why = [None,
           f"unit {RS.unit_name(other)} has {ins} input and {outs} output bits, not rec-sparse's "
           f"{16 * sum(n for _, n in RS.ports(other))} and {16 * RS.out_words(other)}",
           f"unit {RS.unit_name(big)}'s shape exceeds its {ins} input bits",
           "unit rec-sparse/n2-e6-s0 is not rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>",
           "rec-sparse's NU is at least 1, ELL at least 6 and T_LOG at most 16"]
    for g, w in zip(_check(tmp_path / "honest.txt", *names), why, strict=True):
        assert g["ok"] == (w is None) and (w is None or w in g["error"]), (g, w)

    head = next(i for i, ln in enumerate(lines) if ln.startswith("CIRCUIT unit ")) + 1
    rows = lines[head + 1:]
    at = next(i for i in range(50_000, len(rows)) if (f := rows[i].split(" "))[0] == "1" and len(f) == 4 and f[2] == "1"
              and f[1] != f[3])
    a, b = rows[at].split(" ")[1::2]
    lines[head + 1 + at] = f"1 {b} 1 {a}"
    (tmp_path / "forged.txt").write_text("\n".join(lines))
    (g,) = _check(tmp_path / "forged.txt")
    assert not g["ok"] and f"not the verifier's {RS.unit_name(sh)} (row {at} differs)" in g["error"], g
