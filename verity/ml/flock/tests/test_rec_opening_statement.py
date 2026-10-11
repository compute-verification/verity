"""`HmRow.parse` calls `RecOpening.check`: a statement's unit named `rec-opening/m<m>-c<claims>-p<j>` must be the net the
verifier builds for that part of the opening's algebra (PROTOCOL.md §16.10). `flock-verify statement` accepts the files
`rec_residuals.stage_opening` writes, and refuses them with one row of the unit changed and with the unit renamed out of
the name's form."""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest
from verity.core.protocols.sampled_proofs.one_stage import registered as RG
from verity_flock import rec_algebra as A
from verity_flock import rec_residuals as RR
from verity_flock.rec_outer import CLAIM_VALUE

BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"


def _statement(d: Path, circuit: Path):
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(circuit),
                           "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"),
                           "--program", str(d / "program.json"), "--registered", str(d / "registered.json")],
                          capture_output=True, text=True, check=False)


def _swapped(row: str) -> str | None:
    """An AND row ``|A| A |B| B`` with its two forms swapped, the same gate; None if it is not one or its forms are equal."""
    f = row.split(" ")
    if not f[0].isdigit() or not (n := int(f[0])) or len(f) < n + 3 or not f[n + 1].isdigit():
        return None
    a, b = f[1:n + 1], f[n + 2:]
    if len(b) != int(f[n + 1]) or a == b:
        return None
    return " ".join([str(len(b)), *b, str(n), *a])


@pytest.mark.slow
def test_a_rec_opening_unit_other_than_the_verifiers_is_refused(tmp_path):
    """The opening at m = 22 and 2 claims, one part, on random rows (`statement` binds the files and evaluates nothing).
    The staged files are accepted. With one AND row's two forms swapped, the unit is refused at that row; renamed
    `rec-opening/m22-c2`, it is refused for its name before any net is built. About 3 minutes: staging the unit (1,839,105
    rows) and the verifier building its net twice. It is the only test that `flock-verify statement` refuses a rec-opening
    unit off the verifier's net, and m = 22 is the opening's smallest shape."""
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    sh = A.Shape(22, 6, 2, opening_only=True)
    S = RR.spec(sh)
    g = random.Random(7)
    rows = RR.opening_rows(sh, [g.getrandbits(128) for _ in range(S["gf"]["nw"])])
    v = [g.getrandbits(128) for _ in range(S["gf"]["nv"])]
    program = "ab" * 64
    regs = {f"m{i}": RG.Registered.commit(program, f"rec-opening-c{i}", [rows[f"m{i}"]]) for i in range(len(A.rounds(sh)))}
    regs["claim"] = RG.Registered.commit(program, CLAIM_VALUE, [rows["claim0"], rows["claim1"]])
    acc = RG.Registered.commit(program, "rec-acc", [rows["acc0"], rows["acc1"]])
    [_] = RR.stage_opening(sh, regs, (acc, 0, 1), v, tmp_path, evaluate=False)
    d = tmp_path / "p0"
    p = _statement(d, d / "circuit.txt")
    assert p.returncode == 0, p.stdout + p.stderr

    name = "rec-opening/m22-c2-p0"
    lines = (d / "circuit.txt").read_text().split("\n")
    head = next(i for i, ln in enumerate(lines) if ln.startswith(f"flock-ir-unit/v2 {name} "))
    at, row = next((i, s) for i in range(head + 1, len(lines)) if (s := _swapped(lines[i])))
    (d / "forged.txt").write_text("\n".join([*lines[:at], row, *lines[at + 1:]]))
    p = _statement(d, d / "forged.txt")
    why = f"not the verifier's {name} (row {at - head - 1} differs)"
    assert p.returncode == 1 and why in p.stdout, p.stdout + p.stderr

    renamed = [*lines[:head], lines[head].replace(name, "rec-opening/m22-c2", 1), *lines[head + 1:]]
    (d / "renamed.txt").write_text("\n".join(renamed))
    p = _statement(d, d / "renamed.txt")
    why = "unit rec-opening/m22-c2 is not rec-opening/m<m>-c<claims>-p<j>"
    assert p.returncode == 1 and why in p.stdout, p.stdout + p.stderr
