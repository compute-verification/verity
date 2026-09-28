"""`HmRow.parse` calls `RecSparse.check`: a statement's unit named `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` must be the
net the verifier builds at that shape (PROTOCOL.md §16.10). `flock-verify statement` accepts the files `rec_sparse.stage`
writes, and refuses them with one row of the unit changed and with the unit renamed out of the name's form."""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest
from verity_flock import rec_sparse as RS

BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"


def _statement(d: Path, circuit: Path):
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(circuit),
                           "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"),
                           "--program", str(d / "program.json"), "--registered", str(d / "registered.json")],
                          capture_output=True, text=True, check=False)


@pytest.mark.slow
def test_a_rec_sparse_unit_other_than_the_verifiers_is_refused(tmp_path):
    """Two instances at (2,6,0,1) on random rows (`statement` binds the files and evaluates nothing). The staged files are
    accepted. With one AND row's operands swapped, the same gate, the unit is refused at that row; renamed
    `rec-sparse/n2-e6-s0`, it is refused for its name before any net is built."""
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    sh = RS.Shape(2, 6, 0, 1)
    g = random.Random(7)
    inst = [{nm: [g.getrandbits(16) for _ in range(n)] for nm, n in RS.ports(sh)} for _ in range(2)]
    inst[1]["tau"] = inst[0]["tau"]
    RS.stage(sh, inst, tmp_path, "ab" * 64, evaluate=False)
    p = _statement(tmp_path, tmp_path / "circuit.txt")
    assert p.returncode == 0, p.stdout + p.stderr

    lines = (tmp_path / "circuit.txt").read_text().split("\n")
    head = next(i for i, ln in enumerate(lines) if ln.startswith("CIRCUIT unit ")) + 1
    fields = lines[head].split(" ")
    assert fields[:2] == ["flock-ir-unit/v2", RS.unit_name(sh)], fields[:2]
    at = next(i for i in range(head + 1, len(lines))
              if (f := lines[i].split(" "))[0] == "1" and len(f) == 4 and f[2] == "1" and f[1] != f[3])
    a, b = lines[at].split(" ")[1::2]
    forged = [*lines[:at], f"1 {b} 1 {a}", *lines[at + 1:]]
    (tmp_path / "forged.txt").write_text("\n".join(forged))
    p = _statement(tmp_path, tmp_path / "forged.txt")
    why = f"not the verifier's {RS.unit_name(sh)} (row {at - head - 1} differs)"
    assert p.returncode == 1 and why in p.stdout, p.stdout + p.stderr

    renamed = [*lines[:head], " ".join([fields[0], "rec-sparse/n2-e6-s0", *fields[2:]]), *lines[head + 1:]]
    (tmp_path / "renamed.txt").write_text("\n".join(renamed))
    p = _statement(tmp_path, tmp_path / "renamed.txt")
    why = "unit rec-sparse/n2-e6-s0 is not rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>"
    assert p.returncode == 1 and why in p.stdout, p.stdout + p.stderr
