"""a5's R7 copies of a window's own values (`a5.window_copies`: each value's words, its row format and 16-bit leaves, and the
window's open root), the values ``records_audit.window_values`` derives from the circuit's registered parameters no pin
holds. A window that registers its scan state afresh in a v2 registration, under the domain R7 derives from the
population Program, the value, its shape and the window's open root, passes R1-R7 by both verifiers with them. Refused
by code: each alteration `a5.window_negatives` makes, the state at another row width or under another window, and the v2
record checked without the copies (as a5 checked it before it handed R7 any). R7 holds no per-window root: the state's
root is held where its reads open, by the Lean verifier's registered reads, ``records_audit.registration_roots`` and the
network certifier's period."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.core.protocols.certifier.ingress_scan import SCAN_STATE
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG, registration as R, verifier as V
from verity.ml.circuits.codec import descriptor_id, program_digest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.windows import interface as IF, rows as CS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5 as A5  # noqa: E402

K, N, RUN = 64, 8, "ab" * 64
COORD = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)
#: the scan state as `records_audit.window_values` derives it from `_window`'s circuit: rows of its read port's K/2 words
VALUES = {SCAN_STATE: {"words": K // 2, "leaves": 4 * K // 2, "schema": REG.SCHEMA_V2}}


def _window(window: str = RUN) -> SimpleNamespace:
    """`N` coordinates each reading its ``wrow`` from the window's scan state (4 rows of K/2 words, coordinate u row u % 4),
    registered in the window's v2 registration under ``window``'s open root; the window's run root is `RUN`."""
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 1 << 16, (4, K // 2), dtype=np.uint16)
    read = IF.Read(descriptor_id(COORD), "wrow", SCAN_STATE, 0, (N // 4, 4), 0, (0, K // 2))
    circuit = IF.Circuit("test", ((COORD, N),), registered=IF.Registered({SCAN_STATE: 4 * K // 2}, reads=(read,)))
    value = REG.Registered.commit(program_digest(circuit.program()[0]), SCAN_STATE, rows.tolist(), schema=REG.SCHEMA_V2,
                                  window=window)
    win = IF.Window(circuit, RUN, D.parse_law("stratified:2"), 4, registered={SCAN_STATE: value})
    x = rng.integers(-128, 128, (N, K)).astype(np.int8)
    sx, sw = ((np.abs(rng.standard_normal(N)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), rows[np.arange(N) % 4].view(np.uint8), sw.view(np.uint32)]
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)])
    c = win.close()
    registrar = SimpleNamespace(KINDS=R.KINDS, record=lambda *a, **kw: R.record(*a, **kw, registered=[value.entry()]))
    mine = c.window
    publics = {m.key: c.public_body(k) for k, m in enumerate(mine.members)}
    return SimpleNamespace(rec=CS.registration(c, {"label": "test"}, registrar=registrar), mine=mine,
                           want=CS.expected_ports(mine), layer=CS.leaf_layer(publics), value=value)


def _check(w: SimpleNamespace, rec: dict, **copies):
    return V.check(rec, w.mine.program, w.mine.query, w.mine.law, CS.SCHEME, w.want, leaf_layer=w.layer,
                   work_table=w.mine.work_table, **copies)


def test_a_window_registering_its_scan_state_passes_r7_with_its_copies_and_each_alteration_is_refused():
    w = _window()
    a = SimpleNamespace(window_values=VALUES, window_root=RUN)
    copies = A5.window_copies(a, root=True)
    assert copies == {"registered": {SCAN_STATE: K // 2}, "schemas": {SCAN_STATE: REG.SCHEMA_V2},
                      "circuit_params": {SCAN_STATE: 4 * K // 2}, "window_root": RUN}
    assert "window_root" not in A5.window_copies(a) and A5.window_copies(SimpleNamespace(window_root=RUN), root=True) == {}
    assert w.rec["format"] == R.FORMAT_V2 and w.rec["registered"] == [w.value.entry()]
    v = _check(w, w.rec, **copies)
    assert v.ok, v.codes
    neg = A5.window_negatives(a, w.rec, w.mine, w.want, w.layer)
    assert {k: x["codes"] for k, x in neg.items() if x["refused"]} == {
        "window-value-left-out (R7)": ["R7-values"], "window-value-at-another-row-count (R7)": ["R7-leaves"],
        "window-value-under-another-window (R7)": ["R7-domain"], "window-values-unregistered (v1)": ["keys"]}
    wide = {SCAN_STATE: {**VALUES[SCAN_STATE], "words": K // 4}}
    for at, c, codes in ((w, {}, ["keys"]),
                         (w, A5.window_copies(SimpleNamespace(window_values=wide, window_root=RUN), root=True),
                          ["R7-domain", "R7-leaves", "R7-words"]),
                         (_window("cd" * 64), copies, ["R7-domain"])):
        v = _check(at, at.rec, **c)
        assert not v.ok and sorted(v.codes) == codes, v.codes


def test_a5s_stream_declares_the_windows_own_values_and_registers_their_v2_record():
    """`a5.stream_of`'s Spec declares each of the window's own values (registered, with its shape public) beside its
    ports, so its stream receives the v2 registration with R7's copies, as a5's draw registers it (`coins_draw`); a
    stream of a window with none declared refuses the record's registered entry (``roots``), as a5's did before it
    took them."""
    w = _window()
    members = [{"defn": COORD, "m": m} for m in w.mine.members]
    base = {"window_root": RUN, "rule": "stratified:2", "session_timeout": 1, "delta": 2.0 ** -40}
    for values, refused in ((VALUES, None), (None, "roots")):
        a = SimpleNamespace(**base, window_values=values)
        stream, counts = A5.stream_of(a, w.mine, members, w.want, w.layer, lambda n: b"\x07" * n)
        assert counts is None and {v.name for v in stream.spec.values if v.registered} == set(values or ())

        def register():
            return stream.register(w.rec, w.want, leaf_layer=w.layer, window_root=RUN, work_table=w.mine.work_table,
                                   **A5.window_copies(a))
        if refused is None:
            assert register().order == 0
        else:
            with pytest.raises(S.Refused) as e:
                register()
            assert refused in e.value.verdict.codes, e.value.verdict.codes
