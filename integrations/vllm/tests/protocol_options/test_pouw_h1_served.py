"""`protocol_options.pouw_h1_served`: an H1 window's A, y and z tables, committed as registered values of the verifier's
window Program, register as served trees (`one_stage.served.register_served`) whose record R7 accepts and whose
``--registered`` file is written in the window's directory; a table off its round root, a value name reused by a later
window under another root and a table off its fixed slots are refused.

`flock-verify` is not run here: these tests stop at the registration and the ``--registered`` file's bytes."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw.audit import hm96_salt  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G, registration as R  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage.served import register_served  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.circuits.refs import coll_concat  # noqa: E402
from verity.ml.circuits.types import Array, Value  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_requests as RQ  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_served as SV  # noqa: E402

K, N, T, STEPS = 60, 32, 32, 2
SHAPES = [(N, K), (N, K)]
CALLS = STEPS * len(SHAPES)
SALT = hashlib.sha512(b"salt").digest()
W1, W2 = hashlib.sha512(b"window 1").digest(), hashlib.sha512(b"window 2").digest()
V16 = Value(16)
XOR16 = PrimitiveDefinition("H1ServedTestXor16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: a ^ b, register=False)
LAW = {"law": "subset", "k": 2}
SCHEME = "frame-v3-sha512/hm96-sha512"
V2 = "hm96-sha512/row/v2"


def _slot(c: int):
    """Call c's slot unit: it reads one row of each of the call's tables (ports a<c>, y<c>, z<c>)."""
    def sig(S):
        return tuple((f"{t}{c}", Array(SV.width(t, K, N), V16)) for t in SV.TABLES), V16

    return bind(CompositeDefinition(f"H1ServedTestSlot{c}", 0, (), sig,
                                    lambda B, S, a, y, z: B.call(XOR16, B.call(XOR16, a[0], y[0]), z[0]), register=False))


SLOTS = [_slot(c) for c in range(CALLS)]
Q = P.template_instances_query(SLOTS)


def _program(key: bytes) -> Program:
    """The test's window Program for `key`: each call's tables as T-row parameters named `SV.value`, every slot one unit."""
    params = tuple((SV.value(key, c, t), Array(T, Array(SV.width(t, K, N), V16))) for c in range(CALLS) for t in SV.TABLES)

    def sig(S):
        return params, Array(CALLS * T, V16)

    def body(B, S, *tables):
        return coll_concat(sig(S)[1], [B.call(SLOTS[c], *(tables[3 * c + j][s] for j in range(3)))
                                       for c in range(CALLS) for s in range(T)])

    return Program(bind(CompositeDefinition("H1ServedTestWindow", 0, (), sig, body, register=False)))


def _tap(rng, m):
    a = torch.from_numpy(rng.integers(-127, 128, (m, K), dtype=np.int8))
    w = torch.from_numpy(rng.integers(-127, 128, (N, K), dtype=np.int8))
    z = torch.from_numpy(rng.integers(-(1 << 20), 1 << 20, (m, N), dtype=np.int32))
    return a, torch.rand(m) + 0.5, w, torch.ones(N, 1), z, torch.randn(m, N).bfloat16()


def _window(key: bytes, seed: int = 0, served_key: bytes | None = None, tmp=None):
    """A served window: step 0 serves 5 and 3 rows, step 1 is empty; its tables committed as `key`'s values."""
    rng = np.random.default_rng(seed)
    run = PH.H1Run("t", SALT, {0: "q", 1: "o"}, SHAPES, T, STEPS, key=bytes([seed]) * 32,
                   served=SV.Served(_program, window=served_key or key), transcript_dir=tmp)
    taps = [_tap(rng, 5), _tap(rng, 3)]
    for w, (a, s_a, wt, ws, z, y) in enumerate(taps):
        run.record(w, a, s_a, wt, ws, z=z, y=y)
    run.close()
    return run, taps


def _ports(prog: Program) -> dict:
    part = P.partition(prog, Q)
    n = P.units(prog, Q)
    return {"out": {"leaves": n, "domain": R.served_domain(part["program"], P.digest(part), "out", "u16", n, "r-h1")}}


def _record(prog: Program, reg, row: int = 1, log=()):
    ports = _ports(prog)
    roots = [{"port": "out", "root": "ab" * 64, **ports["out"]}]
    previous = R.digest(log[-1]) if log else None
    return R.record(prog, Q, LAW, SCHEME, roots, {"kind": "served", "row": row, "run": "r-h1"}, previous=previous,
                    registered=reg.registered), ports


def test_the_honest_windows_tables_register_and_r7_accepts_them(tmp_path):
    """Every call's A, y and z table is one served tree under the root its round's root commits, of fixed T slots, the
    served rows first and every empty slot zero words; R7 accepts the record they register in, and the --registered file
    is written in the window's directory, each slot unit reading its own row of its call's tables."""
    run, taps = _window(W1, tmp=str(tmp_path))
    trees = SV.trees(run)
    assert [t.value for t in trees] == [SV.value(W1, c, t) for c in range(CALLS) for t in SV.TABLES]
    assert {(t.leaves, t.row_schema, t.word_bits) for t in trees} == {(T, G.SCHEMA, 16)}
    assert [t.root for t in trees] == [run.tables[c][t] for c in range(CALLS) for t in SV.TABLES]
    prog = _program(W1)
    digest = P.program_digest(prog)
    (a, s_a, _w, _ws, z, y) = taps[1]
    want = {"a": PH.table_leaves(T, K + 4, 3, a.numpy(), s_a.numpy().astype("<f4")),
            "y": PH.table_leaves(T, 2 * N, 3, y.view(torch.int16).numpy()),
            "z": PH.table_leaves(T, 4 * N, 3, z.numpy().astype("<i4"))}
    for j, t in enumerate(SV.TABLES):
        rows = SV.words(want[t], SV.width(t, K, N))
        assert not any(any(r) for r in rows[3:])
        salts = [hm96_salt(run._key, RQ.TABLES_NONCE + 3 + j, i) for i in range(T)]
        assert G.Registered.commit(digest, SV.value(W1, 1, t), rows, salts).root.hex() == run.tables[1][t]
    reg = SV.register(run, Q)
    rec, ports = _record(prog, reg)
    assert R.check(rec, prog, Q, LAW, SCHEME, ports, registered=reg.values, outputs=reg.outputs).ok
    assert (tmp_path / SV.FILE).read_bytes() == reg.file_bytes() and json.loads(reg.file_bytes()) == reg.file
    a1 = reg.ports["a1"]
    assert (a1["value"], a1["root"], a1["leaves"]) == (SV.value(W1, 1, "a"), run.tables[1]["a"], T)
    assert a1["reads"] == [None] * T + list(range(T)) + [None] * (CALLS - 2) * T
    assert reg.partition == P.partition(prog, Q)


def test_the_window_key_is_its_window_open_root():
    commits = {"weights_root": "11" * 64, "requests_root": "22" * 64, "session": {"session": "s-1"}}
    key = SV.Served(_program).key(commits)
    assert key == PS.window_open_root(bytes.fromhex("11" * 64), bytes.fromhex("22" * 64), "s-1")
    with pytest.raises(ValueError, match="the window has no key"):
        SV.Served(_program).key({"weights_root": "11" * 64})


def test_a_table_whose_root_isnt_the_one_under_its_round_root_is_refused():
    run, _ = _window(W1)
    e = run.served.entries[(0, "y")]
    run.served.entries[(0, "y")] = {**e, "root": "cd" * 64}
    with pytest.raises(ValueError, match="the one round 0's root commits"):
        SV.trees(run)
    run.served.entries[(0, "y")] = e
    run.tables[0]["y"] = "cd" * 64
    with pytest.raises(ValueError, match="round 0's root is over calls"):
        SV.trees(run)


def test_a_second_window_reusing_the_first_windows_value_name_with_another_root_is_refused():
    """R7-held: a later window keyed as the first registers its tables under the first's names, so each name meets
    another root, and is refused; keyed as its own window it registers after the first."""
    first, _ = _window(W1)
    rec, _ = _record(_program(W1), SV.register(first, Q))
    again, _ = _window(W2, seed=1, served_key=W1)
    with pytest.raises(ValueError, match="R7-held"):
        SV.register(again, Q, log=[rec])
    own, _ = _window(W2, seed=1)
    reg = SV.register(own, Q, log=[rec])
    assert not set(reg.values) & {e["value"] for e in rec["registered"]}
    later, ports = _record(_program(W2), reg, row=2, log=[rec])
    assert R.check(later, _program(W2), Q, LAW, SCHEME, ports, log=[rec], registered=reg.values).ok


@pytest.mark.parametrize("rows, extra", [(T + 1, 0), (T, 1)], ids=["past-the-slots", "past-the-width"])
def test_a_table_outside_its_fixed_slots_is_refused(rows, extra):
    run, _ = _window(W1)
    key, _p, digest = run.served.held(run.commits)
    n = SV.width("a", K, N) + extra
    reg = G.Registered.commit(digest, SV.value(key, 1, "a"), [[0] * n for _ in range(rows)],
                              [hm96_salt(run._key, 9, i) for i in range(rows)])
    run.served.entries[(1, "a")] = reg.entry()
    with pytest.raises(ValueError, match="off its fixed slots"):
        SV.trees(run)


def test_a_window_whose_tables_arent_registered_values_is_refused():
    run = PH.H1Run("t", SALT, {0: "q", 1: "o"}, SHAPES, T, STEPS)
    run.close()
    with pytest.raises(ValueError, match="aren't registered values"):
        SV.trees(run)


@pytest.mark.xfail(reason="REGV2: register_served refuses hm96-sha512/row/v2 trees until proofs' REGV2 lands; H1's tile "
                          "statement reads its A rows as row/v2 bit rows", raises=ValueError, strict=True)
def test_a_row_v2_window_registers():
    import dataclasses

    run, _ = _window(W1)
    register_served(_program(W1), Q, [dataclasses.replace(t, row_schema=V2) for t in SV.trees(run)])
