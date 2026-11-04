"""`circuit.write`'s ``committed`` (through `class_statement.UnitStager.stage`): a statement staged from the value service's
``b || c`` instead of salts of its own. Answered with the ``b || c`` the stager's own salts make, the public file is byte for
byte the salted staging's, and the prover's file is too but for its salts, which are zero (the prover takes the ZK server's on a
sample, `flock-circuit --salt-refs`); the service is asked once, for each port's table under the header's domain id and
schema, the rows' unsalted digests in the order the public file holds their ``b || c``, per instance or a shared-row file's
tables. Salts or a salt key beside it, and an answer that is not one ``b || c`` per row, are refused."""
from __future__ import annotations

import json

import numpy as np
import pytest
from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS

#: a Boolean unit over a 32-bit row and a 2048-bit one (a composed circuit needs a row that spans SHA-512 blocks)
UNIT = bind(definition("CommittedStagingUnit", 1, lambda C, a, b, w: [C.AND(a[k], b[k]) ^ C.AND(w[k], w[2047 - k]) for k in range(16)],
                       (("a", Array(16, BIT)), ("b", Array(16, BIT)), ("w", Array(2048, BIT))), Array(16, BIT), register=False))
ROWS = [("ab", V2, 32, ["a", "b"]), ("w", V2, 2048, ["w"])]
N = 5
INST = hm96.SHA512


def _round(shared: bool) -> tuple[dict, dict]:
    """The rows, and the stage's keywords: per instance, or a shared-row file whose table holds three rows the instances read."""
    rng = np.random.default_rng(7)
    w = rng.integers(0, 1 << 16, (N, 128), dtype=np.uint16)
    if not shared:
        return {"ab": rng.integers(0, 1 << 16, (N, 2), dtype=np.uint16), "w": w}, {}
    table, refs = rng.integers(0, 1 << 16, (3, 2), dtype=np.uint16), np.array([2, 0, 1, 1, 2])
    return {"ab": table[refs], "w": w}, {"tables": {"ab": table, "w": w}, "refs": {"ab": refs, "w": np.arange(N)}}


def _answer(st: CS.UnitStager, salts: dict, asked: list):
    """A value service that commits each table under the salted staging's salts, recording what it was asked."""
    def committed(tables):
        asked.append(tables)
        return [[INST.commit_string(INST.default_key, xs[64 * i:64 * (i + 1)], salts[p.name][i]) for i in range(len(xs) // 64)]
                for p, (_d, _sc, xs) in zip(st.ports, tables)]
    return committed


@pytest.mark.slow
@pytest.mark.parametrize("shared", [False, True], ids=["per-instance", "shared-rows"])
def test_a_committed_staging_is_the_salted_one_with_its_salts_zero(shared, tmp_path):
    st = CS.UnitStager(UNIT, "unit/committed-staging", ROWS, [(V2, (0,))])
    rows, kw = _round(shared)
    salted = st.stage(rows, tmp_path / "salted", b"k" * 32, **kw)
    asked: list = []
    got = st.stage(rows, tmp_path / "committed", None, committed=_answer(st, salted["salts"], asked), **kw)
    assert got["salts"] is None
    assert {k: v for k, v in got.items() if k not in ("salts", "outs")} == {k: v for k, v in salted.items() if k not in ("salts", "outs")}
    assert all(np.array_equal(got["outs"][o], salted["outs"][o]) for o in salted["outs"])
    assert (tmp_path / "committed" / "pub.bin").read_bytes() == (tmp_path / "salted" / "pub.bin").read_bytes()
    order = ([s for p in st.ports for s in salted["salts"][p.name]] if shared else
             [salted["salts"][p.name][i] for i in range(N) for p in st.ports])
    region, inst = b"".join(order), (tmp_path / "salted" / "inst.bin").read_bytes()
    at = inst.index(region)
    assert (tmp_path / "committed" / "inst.bin").read_bytes() == inst[:at] + bytes(len(region)) + inst[at + len(region):]
    head = json.loads((tmp_path / "salted" / "pub.bin").read_bytes().split(b"\n", 1)[0])["frame_v3"]
    (tables,) = asked
    assert [(d.hex(), sc) for d, sc, _xs in tables] == [(head["domain_ids"][p.name], head["schemas"][p.name]) for p in st.ports]
    counts = [3, N, N] if shared else [N, N, N]
    assert [len(xs) for _d, _sc, xs in tables] == [64 * c for c in counts]


@pytest.mark.slow
def test_salts_beside_the_service_and_a_short_answer_are_refused(tmp_path):
    st = CS.UnitStager(UNIT, "unit/committed-staging", ROWS, [(V2, (0,))])
    rows, _kw = _round(False)
    salted = st.stage(rows, tmp_path / "salted", b"k" * 32)
    answer = _answer(st, salted["salts"], [])
    with pytest.raises(ValueError, match="no salts or salt key beside it"):
        st.stage(rows, tmp_path / "key", b"k" * 32, committed=answer)
    with pytest.raises(ValueError, match="no salts or salt key beside it"):
        st.stage(rows, tmp_path / "salts", None, salts=salted["salts"], committed=answer)
    with pytest.raises(ValueError, match="no salts, salt key or registered reads"):
        st.stage(rows, tmp_path / "registered", None, registered={}, committed=answer)
    with pytest.raises(ValueError, match="one 128-byte b [|][|] c per row of each port's table"):
        st.stage(rows, tmp_path / "short", None, committed=lambda t: [bcs[:-1] for bcs in answer(t)])
    with pytest.raises(ValueError, match="one 128-byte b [|][|] c per row of each port's table"):
        st.stage(rows, tmp_path / "one-port", None, committed=lambda t: answer(t)[:1])
