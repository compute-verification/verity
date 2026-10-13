"""The verifier's registered-opening units (`Flock.RecOpening`, through `flock-rec-opening`) against `rec_residuals`' builders:
its S of the opening (`rec_algebra.structure` at an opening-only shape), its row ports and its parts are `rec_residuals.spec`'s
and `parts`', and its unit for each part, ``rec-opening/m<m>-c<claims>-p<j>``, is the one `stage_opening` writes, row for row.
A unit that is not is refused by what it differs in: its claims, its part, its name's form, its m, one row."""
from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_algebra as A
from verity_flock import rec_residuals as RR

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-rec-opening"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@functools.cache
def _build() -> None:
    subprocess.run(["lake", "build", "flock-rec-opening"], cwd=LEAN, check=True, capture_output=True)


def _run(*args: str) -> tuple[int, list[dict]]:
    _build()
    p = subprocess.run([str(BIN), *args], capture_output=True, text=True, check=False)
    return p.returncode, [json.loads(ln) for ln in p.stdout.splitlines()] or [{"stderr": p.stderr}]


def _shape(m: int, c: int) -> A.Shape:
    return A.Shape(m, 6, c, opening_only=True)


@pytest.mark.parametrize("m,c", [(m, 2) for m in range(22, 36)] + [(22, 1), (27, 8)])
def test_the_verifiers_opening_algebra_is_rec_residuals_spec_and_parts(m, c):
    S = RR.spec(_shape(m, c))
    rc, [line] = _run("structure", str(m), str(c))
    assert rc == 0, line
    assert line["gf"] == S["gf"]
    assert line["ports"] == [n for _, n in S["ports"]]
    assert [{"ports": [n for _, n in p.S["ports"]], "nv": p.S["gf"]["nv"], "ops": p.S["gf"]["ops"], "res": p.S["gf"]["res"]}
            for p in RR.parts(S)] == line["parts"]


def test_an_opening_whose_residual_fits_no_part_is_refused_as_parts_refuses_it():
    """At 21 claims a ring switch's residual alone reads more sha512x3 slots than even a lone part holds (`MAX_LONE_SLOTS`), in
    Python and in Lean."""
    with pytest.raises(ValueError) as e:
        RR.parts(RR.spec(_shape(30, 21)))
    rc, [line] = _run("structure", "30", "21")
    assert rc == 1 and str(e.value) in line["stderr"], line


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """`stage_opening`'s circuit file of each part at claims 2 (`rec_residuals.statement` over one instance), by m, each
    staged once."""
    @functools.cache
    def stage(m: int) -> list[Path]:
        d = tmp_path_factory.mktemp(f"rec-opening-m{m}")
        out = []
        for j, p in enumerate(RR.parts(RR.spec(_shape(m, 2)))):
            f = d / f"p{j}.txt"
            f.write_text(RR.statement(p.S, f"rec-opening/m{m}-c2-p{j}", 1).text)
            out.append(f)
        return out
    return stage


@pytest.mark.parametrize("m,rows", [(22, 1_839_105), (23, None)])
def test_the_verifiers_opening_unit_is_the_one_stage_opening_writes(m, rows, staged):
    """The staged shapes: claims 2 at m = 22 (1,839,105 unit rows) and m = 23, every part. About 1.5 minutes per m, most of it
    Python's lowering: the only test that compares the verifier's net with the unit a prover is handed."""
    files = staged(m)
    if rows is not None:
        assert len(files) == 1 and f"\nflock-ir-unit/v2 rec-opening/m{m}-c2-p0 {rows} ".encode() in files[0].read_bytes()
    rc, lines = _run("check", *map(str, files))
    assert rc == 0 and all(ln["ok"] for ln in lines) and len(lines) == len(files), lines


def test_the_staged_opening_unit_reads_each_element_whole(staged):
    """`RecResiduals.partlyRead` and `RecOpening.rsUnread` in Python, on m = 22's file: its computed rows read each row-port
    element in all 128 bits or in none, and each element of a claim's ring switch (ports ``m0``, ``m1``) in all 128. The
    check refuses the verifier's own net, so this is where a builder change that reads an element in part shows by the
    element."""
    [f] = staged(22)
    lines = f.read_bytes().split(b"\n")
    head = lines[3].split()
    const, groups, at = int(head[3]), int(head[4]), 5
    words = 0
    for _ in range(groups):
        words += int(head[at + 1])
        at += 3 + int(head[at + 2])
    n_in = min(16 * words, const)
    read = set()
    for r in range(n_in, const):
        row = list(map(int, lines[4 + r].split()))
        read.update(c for c in row[1:1 + row[0]] + row[2 + row[0]:] if c < n_in)
    [p] = RR.parts(RR.spec(_shape(22, 2)))
    col, counts = 0, {}
    for name, n in p.S["ports"]:
        for e in range(n):
            counts[name, e] = sum(col + 128 * e + b in read for b in range(128))
        col += 16 * RR.port_words(n)
    assert {k for k in counts.values()} <= {0, 128}, {e: k for e, k in counts.items() if k not in (0, 128)}
    assert all(counts[f"m{k}", e] == 128 for k in range(2) for e in range(128))


def test_a_unit_off_the_verifiers_opening_is_refused_by_what_it_differs_in(staged, tmp_path):
    """m = 22's unit renamed to another claim count, part, form and m, and with one row's operand moved: each is refused by
    the value it differs in. About 1.5 minutes, the verifier building m = 22's net for each: only a unit as large as the
    staged one shows that a net one row off is refused, not passed as the verifier's."""
    [f] = staged(22)
    lines = f.read_bytes().split(b"\n")
    name = b"rec-opening/m22-c2-p0"
    assert lines[3].startswith(b"flock-ir-unit/v2 " + name + b" ")
    at = 999_999
    assert lines[at] == b"2 105427 105431 2 53459 53463", lines[at]
    cases = {"rec-opening/m22-c3-p0": "unit rec-opening/m22-c3-p0 has 154624 input and 2048 output bits, not its part's 204800 and 2048",
             "rec-opening/m22-c2-p1": "unit rec-opening/m22-c2-p1: part 1 of 1",
             "rec-opening/m22-c2": "unit rec-opening/m22-c2 is not rec-opening/m<m>-c<claims>-p<j>",
             "rec-opening/m21-c2-p0": "rec-opening's m has a fast100 schedule, claims at least 1"}
    files = []
    for k, new in enumerate(cases):
        g = tmp_path / f"name{k}.txt"
        g.write_bytes(b"\n".join(lines[:3] + [lines[3].replace(name, new.encode())] + lines[4:]))
        files.append(g)
    g = tmp_path / "row.txt"
    g.write_bytes(b"\n".join(lines[:at] + [b"2 105427 105431 2 53459 53464"] + lines[at + 1:]))
    files.append(g)
    rc, got = _run("check", *map(str, files))
    want = [f"circuit {name.decode()}: {why}" for why in cases.values()]
    want.append(f"circuit {name.decode()}: not the verifier's {name.decode()} (row 999995 differs)")
    assert rc == 1 and [ln.get("error") for ln in got] == want, got
