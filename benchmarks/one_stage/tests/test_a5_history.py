"""a5's registration check on a window with a history table (`a5.registration_check`): the verifier of record, the Lean
and its Python mirror, computes the table's root from its writer's public bodies, which the leaf layer binds
(`ml.windows.history.params`), and accepts the root the window registers; that root with a bit flipped (a5's negative)
and the root over the same rows in another order are refused by both, a5's code ``history``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from verity.core.protocols.spot.one_stage import registration as R, verifier as V
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.commitments import history as HT
from verity.ml.windows import history as HW, interface as I, rows as CS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5 as A5

G8 = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
G8ID = descriptor_id(G8)
LAW = {"law": "stratified", "k": 2}
RUN = "ab" * 64


def _window(d: Path, n: int = 4):
    """``n`` units of ``Gemm_v2{K=16, N=8}``, two a member, their outputs history table ``latent``'s rows, the window
    written to ``d``: -> the verifier's window, the circuit record, the registration, the members' public bodies and the
    table."""
    rng = np.random.default_rng(0)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    x, w = bf16(rng.standard_normal((n, 16))), bf16(rng.standard_normal((n, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(G8, x, w, kernel="reference"), dtype=np.uint16).reshape(n, 8)
    table = HT.HistoryTable("program", "latent", 8, 16)
    table.append(y)
    runs = ({"start": 0, "count": n, "writer": G8ID, "out": "o0", "radix": [n], "base": 0, "step": [1]},)
    circuit = I.Circuit("history", ((G8, n),),
                        registered=I.Registered({}, histories=(I.History("latent", "program", G8ID, "o0", 8, runs, 16),)))
    win = I.Window(circuit, RUN, LAW, 2, registered={"latent": table})
    win.record(G8, [x, w], [y])
    c = win.close(salt_key=bytes(32))
    win.write(c, d, {"label": "test"}, registrar=R)
    doc, rec = (json.loads((d / f).read_text()) for f in ("circuit.json", "registration.json"))
    return win.window, doc, rec, {m.key: c.public_body(k) for k, m in enumerate(win.window.members)}, table


def test_the_verifier_of_record_holds_the_registered_history_root_to_the_writers_bodies(tmp_path):
    mine, doc, rec, publics, table = _window(tmp_path / "w")
    roots = I.registered_roots(rec["window"])
    assert roots == {"latent": table.root.hex()}
    leaf, want = CS.leaf_layer(publics), CS.expected_ports(mine)
    hist = HW.params(doc, mine, publics, roots)

    def check(h: dict | None) -> dict:
        return A5.registration_check(argparse.Namespace(), rec, mine.program, mine.query, mine.law, want, leaf,
                                     mine.work_table, h)

    got = check(hist)
    assert got["ok"] and got["verifiers"]["lean"] == got["verifiers"]["python"], got
    assert check(None)["ok"]
    r = roots["latent"]
    other = HT.HistoryTable("program", "latent", 8, 16)
    other.append(np.stack(table.rows[::-1]), table.salts[::-1])
    for wrong in (r[:-1] + format(int(r[-1], 16) ^ 1, "x"), other.root.hex()):
        bad = {**hist, "registered": {"latent": wrong}}
        got = check(bad)
        assert (got["ok"], got["codes"]) == (False, ["history"]) and "not the table's root" in got["detail"]["history"]
        params = V._params(mine.program, mine.query, mine.law, CS.SCHEME, want, leaf, (), (), None, mine.work_table, None,
                           history=bad)
        assert "refused" in V.lean("registration", {"params": params, "record": rec})
