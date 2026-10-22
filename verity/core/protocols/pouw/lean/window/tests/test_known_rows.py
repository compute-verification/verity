"""`PouwWindow.KnownRows`, a drawn H1 leaf's known rows (lane l's ``pub`` and ``admit``, every statement's ``o0``), held
to the Python that writes them on the shared harness (`verity_difftest`): `benchmarks/pouw/h1_lanes/run.py`'s
`known_rows` (`pub_words`, `bits_words` and `h1unit.OK`), the served path's stand-in, until the served path's producer
of known content is on the tree.

The drawn inputs are a 64-byte salt and roots, positions in and out of a u64, admit rows of 32 bits and of other
lengths, and every lane and the combine. The salts the Python derives (`h1unit.pub_salt`, `ok_salt`) are no part of a
row's content, which is what the known-content check binds, so the port compares the rows' words.
"""
from __future__ import annotations

import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import verity_difftest as D
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[7]
EXE = "pouw-window-difftest"


def _h1_lanes():
    spec = importlib.util.spec_from_file_location("h1_lanes_run", ROOT / "benchmarks/pouw/h1_lanes/run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


H1L = _h1_lanes()


def known_rows(x: dict) -> dict:
    """Each known port's rows, as h1_lanes writes the statement's (``lane`` its index, or None for the combine)."""
    ln = x["lane"]
    kind = H1L.LL.COMBINE if ln is None else H1L.LL.LANE
    leaf = SimpleNamespace(salt=bytes.fromhex(x["salt"]), position=x["position"], admit=x["admit"])
    w = SimpleNamespace(leaf=leaf, roots=tuple(bytes.fromhex(r) for r in x["roots"]))
    ctx = SimpleNamespace(name="h1-lanes", position=x["position"])
    try:
        got = H1L.known_rows(ctx, w, kind, ln, H1L.RC[0] if ln is None else ln)
    except (ValueError, OverflowError) as e:
        raise D.Reject(str(e)) from e
    return {port: [row] for port, (_value, row, _salt) in got.items()}


def known_in(rng: random.Random) -> dict:
    pos = rng.choice((0, 1, (1 << 32) - 1, 1 << 32, (1 << 64) - 1, rng.getrandbits(32), rng.getrandbits(64),
                      rng.getrandbits(64), 1 << 64, rng.getrandbits(80), -1))
    n = rng.choice((32, 32, 32, 0, 1, 15, 16, 17, rng.randrange(80)))
    return {"salt": rng.randbytes(64).hex(), "position": pos, "roots": [rng.randbytes(64).hex() for _ in range(2)],
            "admit": [rng.getrandbits(1) for _ in range(n)], "lane": rng.choice((None, 0, 31, rng.randrange(32)))}


KNOWN = D.Port(EXE, "known_rows", reference=known_rows, inputs=st.randoms(use_true_random=False).map(known_in),
               package=LEAN, of_record="lean")
#: Lean's lane ports alone (`test_lanes.py` holds them to `h1zy`)
LANE_PORTS = D.Port(EXE, "lane_ports", reference=lambda x: None, inputs=st.nothing(), package=LEAN)

server = D.server_fixture(KNOWN)


def test_the_python_is_the_lean(server) -> None:
    D.run(KNOWN, examples=300, server=server)


@pytest.mark.parametrize("ln", [0, 7, 31, None])
def test_the_served_leaf(ln: int | None, server) -> None:
    """At the served leaf's public inputs (`run.py`'s salt, its every-row admit bits) the rows are Python's."""
    rng = random.Random(ln)
    x = {"salt": rng.randbytes(64).hex(), "position": rng.getrandbits(64),
         "roots": [rng.randbytes(64).hex() for _ in range(2)], "admit": [1] * H1L.RC[0], "lane": ln}
    D.replay(KNOWN, x, server)


@pytest.mark.parametrize("k", [448, 4096])
def test_the_rows_fill_their_ports(k: int, server) -> None:
    """Each known row of a served lane is its port's words (`lanePorts`: ``pub`` 101, ``admit`` 2, ``o0`` 1)."""
    rng = random.Random(k)
    x = {"salt": rng.randbytes(64).hex(), "position": rng.getrandbits(64),
         "roots": [rng.randbytes(64).hex() for _ in range(2)], "admit": [1] * H1L.RC[0], "lane": 3}
    kind, rows = D.replay(KNOWN, x, server)
    assert kind == "output"
    kind, ports = LANE_PORTS.lean(server, {"r": 32, "c": 32, "k": k, "lane": 3})
    assert kind == "output"
    words = {p["name"]: p["words"] for p in json.loads(ports)}
    assert {p: len(r[0]) for p, r in json.loads(rows).items()} == {p: words[p] for p in ("admit", "pub", "o0")}
