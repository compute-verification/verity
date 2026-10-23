"""`PouwWindow.RegFile`, a drawn H1 leaf's ``--registered`` files from the accepted registration and produced record, held
to the Python on the shared harness (`verity_difftest`): `leaf_lanes.registered_file` at `leaf_lanes.reads` and at
`leaf_lanes.verifier_entries`' entries and roots, the files h1_lanes' statements read (`benchmarks/pouw/h1_lanes/run.py`),
and `pouw-leaf-own`'s check (`leafOwn?`), which accepts each such file and refuses one whose data port is not the
layout's read at the records' root.

The drawn inputs are registrations naming the leaf's four registered values (one sometimes missing, one sometimes named
twice, the partials sometimes among them), produced records with and without the partials, leaf tiles of 2 to 64 lanes,
every lane, the combine and a lane past the last, and known ports that do and don't collide with a read.
"""
from __future__ import annotations

import copy
import random
import shutil
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.sampled_proofs.one_stage import registered as G

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXE = "pouw-window-difftest"
TILES = ((32, 32), (8, 16), (16, 8), (64, 32), (4, 16))
VALUES = (LL.A, LL.B, LL.ZY, LL.RECEIPT)


def _hex(rng: random.Random) -> str:
    return rng.randbytes(64).hex()


def _entry(rng: random.Random, value: str) -> dict:
    e = {"value": value, "root": _hex(rng), "leaves": rng.randrange(1, 65), "words": rng.randrange(1, 300),
         "domain": _hex(rng)}
    if rng.random() < 0.2:
        e["word_bits"] = 32
    return e


def _known(rng: random.Random, name: str) -> dict:
    return {"value": name, "root": _hex(rng), "leaves": 1, "words": 1, "word_bits": 16, "domain": _hex(rng),
            G.SCHEMA_KEY: G.SCHEMA_V2, "reads": [0]}


def records(rng: random.Random, lanes: int, *, whole: bool = False) -> tuple[dict, dict]:
    """A registration record of the leaf's values and a produced record of its partials (``whole``: each value once)."""
    reg = [_entry(rng, v) for v in VALUES if whole or rng.random() > 0.05]
    if not whole and rng.random() < 0.2:
        reg.append(_entry(rng, rng.choice((*VALUES, LL.PARTIALS))))
    rng.shuffle(reg)
    prod = [] if not whole and rng.random() < 0.05 else [
        {"value": LL.PARTIALS, "root": _hex(rng), "leaves": lanes, "words": 8, "domain": _hex(rng)}]
    if rng.random() < 0.2:
        prod.insert(0, {"value": "other", "root": _hex(rng), "leaves": 1, "words": 1, "domain": _hex(rng)})
    return ({"format": "verity/registration/v2", "program": _hex(rng), "registered": reg},
            {"format": "verity/registration/produced/v1", "produced": prod})


def file_in(rng: random.Random) -> dict:
    r, c = rng.choice(TILES)
    n = LL.lanes(r, c)
    registration, produced = records(rng, n)
    names = rng.sample((LL.PUB, LL.ADMIT, LL.OK), rng.randrange(4))
    if rng.random() < 0.1:
        names.append(rng.choice(("a0", "p1", LL.PARTIALS)))
    return {"registration": registration, "produced": produced, "r": r, "c": c, "k": rng.choice((64, 448, 4096)),
            "lane": rng.choice((None, None, None, 0, n - 1, rng.randrange(n), rng.randrange(n), n)),
            "known": {nm: _known(rng, nm) for nm in names}, "complete": rng.random() < 0.8}


def registered_file(x: dict) -> dict:
    """The statement's file as h1_lanes writes it: `registered_file` at `verifier_entries`, every value's rows ``row/v2``."""
    ln = x["lane"]
    kind = LL.COMBINE if ln is None else LL.LANE
    try:
        reads = LL.reads(x["r"], x["c"], kind, ln)
        ent, roots = LL.verifier_entries(x["registration"], x["produced"], LL.lanes(x["r"], x["c"]), 2 * LL.words(x["k"]))
        return LL.registered_file(reads, ent, roots, {v: G.SCHEMA_V2 for v in ent}, x["known"], x["complete"])
    except (ValueError, KeyError) as e:
        raise D.Reject(str(e)) from e


FILE = D.Port(EXE, "registered_file", reference=registered_file, inputs=st.randoms(use_true_random=False).map(file_in),
              package=LEAN, of_record="lean")
#: `pouw-leaf-own`'s check alone: a file's verdict at the records (no Python counterpart)
OWN = D.Port(EXE, "leaf_own", reference=lambda x: None, inputs=st.nothing(), package=LEAN)

server = D.server_fixture(FILE)


def test_the_python_is_the_lean(server) -> None:
    D.run(FILE, examples=300, server=server)


def _served(seed: int, ln: int | None) -> tuple[dict, dict]:
    """A served statement's input (each value registered once, the known ports h1_lanes registers) and its Python file."""
    rng = random.Random(seed)
    registration, produced = records(rng, LL.lanes(32, 32), whole=True)
    known = (LL.PUB, LL.ADMIT, LL.OK) if ln is not None else (LL.OK,)
    x = {"registration": registration, "produced": produced, "r": 32, "c": 32, "k": 448, "lane": ln,
         "known": {nm: _known(rng, nm) for nm in known}, "complete": True}
    return x, registered_file(x)


def _own(server, x: dict, file: dict) -> tuple[str, str]:
    return OWN.lean(server, {k: x[k] for k in ("registration", "produced", "r", "c", "k", "lane")} | {"file": file})


@pytest.mark.parametrize("ln", [0, 5, 31, None])
def test_the_served_file_is_the_lean_and_passes_its_check(ln: int | None, server) -> None:
    """At a served statement the Python's file is the Lean's, and `pouw-leaf-own` accepts it."""
    x, file = _served(ln or 0, ln)
    D.replay(FILE, x, server)
    assert _own(server, x, file) == ("output", "true")


@pytest.mark.parametrize("ln", [3, None])
@pytest.mark.parametrize("field", ["root", "domain", "leaves", "reads"])
def test_the_check_refuses_a_data_port_off_the_records(ln: int | None, field: str, server) -> None:
    """A file one of whose data ports is not the layout's read at the records' root (another root, domain, leaf count or
    row) is refused by name; a file built from another registration is refused too."""
    x, file = _served(10 + (ln or 0), ln)
    port = next(iter(LL.reads(32, 32, LL.COMBINE if ln is None else LL.LANE, ln)))
    bad = copy.deepcopy(file)
    e = bad["ports"][port]
    e[field] = {"root": "00" * 64, "domain": "11" * 64, "leaves": e["leaves"] + 1, "reads": [e["reads"][0] + 1]}[field]
    kind, why = _own(server, x, bad)
    assert kind == "reject" and f"port {port} is not the layout's read" in why
    other, _ = _served(99, ln)
    kind, why = _own(server, other, file)
    assert kind == "reject" and "is not the layout's read" in why
