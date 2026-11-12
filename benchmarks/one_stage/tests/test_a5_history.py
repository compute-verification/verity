"""a5 on a window with a history table: the verifier of record, the Lean and its Python mirror, computes the table's
root from its writer's public bodies, which the leaf layer binds (`ml.windows.history.params`), and accepts the root the
window registers, in a5's check before the draw (`verifier.check`) and in its stream's registration, the run of record's
(`a5.registering`); that root with a bit flipped (a5's negative) and the root over the same rows in another order are
refused by both, `registration.HISTORY`. On the same window, a5's negatives that the verifier decides are refused while
the Lean answers and none is under a Lean that can't run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as D, registration as R, verifier as V
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.definitions.ops import gemm as G
from verity.commitments import history as HT
from verity.commitments.windows import history as HW, interface as I, rows as CS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "verity" / "protocols" / "spot" / "verifier" / "tests"))
import a0  # noqa: E402
import a5 as A5  # noqa: E402
import fake_challenger as FC  # noqa: E402

G8 = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
G8ID = descriptor_id(G8)
LAW = "stratified:2"
#: the stream's rate, whose law over the window's four units is `LAW` (`service.params_law`)
RULE = "stratified-rate:1/2"
RUN = "ab" * 64
SESSION = "5e" * 64
#: the verdict of record on the window's one session, as an audit record takes it
VERDICTS = {"lean": {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}}


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
    win = I.Window(circuit, RUN, D.parse_design(LAW), 2, registered={"latent": table})
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
    hist = HW.params(doc, mine, publics, HW.pins(doc, roots))
    a = argparse.Namespace(rule=RULE, session_timeout=1, window_root=mine.run, delta=0.5)
    assert A5.rated(mine, RULE) == mine

    def check(h: dict | None) -> V.Verdict:
        return V.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, want, leaf_layer=leaf,
                       work_table=mine.work_table, history=h)

    def register(h: dict | None) -> S.WindowCommitmentRecord:
        stream, counts = A5.stream_of(a, mine, [{"defn": G8, "m": m} for m in mine.members], want, leaf,
                                      FC.Challenger("a5-history"))
        return stream.register(rec, want, leaf_layer=leaf, window_root=a.window_root, counts=counts,
                               work_table=mine.work_table, **A5.registering(a, h))

    got = check(hist)
    assert got.ok and got.verifiers["lean"] == got.verifiers["python"], got
    assert check(None).ok and register(hist).digest == R.digest(rec)
    r = roots["latent"]
    other = HT.HistoryTable("program", "latent", 8, 16)
    other.append(np.stack(table.rows[::-1]), table.salts[::-1])
    for wrong in (r[:-1] + format(int(r[-1], 16) ^ 1, "x"), other.root.hex()):
        bad = {**hist, "registered": {"latent": wrong}}
        got = check(bad)
        assert got.codes == (R.HISTORY,) and "not the table's root" in got.detail[R.HISTORY]["python"], got
        with pytest.raises(S.Refused) as e:
            register(bad)
        assert e.value.verdict.codes == (R.HISTORY,)


def _negatives(mine, doc: dict, rec: dict, publics: dict) -> tuple[bool, dict[str, dict]]:
    """a5's negatives that the verifier decides, of each of their five kinds, on the window with one input altered as
    a5 alters it: the registration held to ports bound to another run (R3) and to a leaf layer over a changed public
    body (R4), the history root with a bit flipped, a sample naming a unit outside the partition, a proved set short of
    the sample, and the audit record's own (`a0.audit_negatives`), on the record of two units sampled and proved.
    -> (whether that record accepts, each negative's row)."""
    leaf, want = CS.leaf_layer(publics), CS.expected_ports(mine)
    hist = HW.params(doc, mine, publics, HW.pins(doc, I.registered_roots(rec["window"])))

    def check(ports: dict = want, leaf_layer: str = leaf, history: dict | None = None) -> V.Verdict:
        return V.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, ports, leaf_layer=leaf_layer,
                       work_table=mine.work_table, history=history)

    first = mine.members[0].key
    flipped = dict(publics, **{first: publics[first][:-1] + bytes([publics[first][-1] ^ 1])})
    root = hist["registered"]["latent"]
    wrong = {**hist, "registered": {"latent": root[:-1] + format(int(root[-1], 16) ^ 1, "x")}}
    n, design = rec["population"], rec["law"]
    sample = {**design, "population": n, "units": [0, 1]}
    registered = {r["port"]: r["root"] for r in rec["roots"]}
    record = V.audit_record(rec, R.receipt(rec, "verifier-test", received="0"), sample, design, sample["units"], 2,
                            {"statement": "stub"}, VERDICTS, 0.5, sessions=[SESSION], session_roots=registered)
    return record["accepted"], {
        "R3": a0.negative(check(ports=CS.expected_ports(replace(mine, run="00" * 64)))),
        "R4": a0.negative(check(leaf_layer=CS.leaf_layer(flipped))),
        "history-root": a0.negative(check(history=wrong)),
        "sample": a0.negative(V.check_draw({**sample, "units": [0, n]}, rec, design)),
        "proved": a0.negative(V.check_proved(sample, sample["units"][1:], 1)),
        **a0.audit_negatives(record, design, 0.5, registered)}


def test_a_negative_is_refused_only_where_the_lean_answered(tmp_path, monkeypatch):
    """While the Lean answers, the window's record accepts and each negative is refused: the registration's, the
    sample's and the proved set's by the code both verifiers give, the history root's as `registration.HISTORY` with the
    Lean's reason the Python's, the record's with the two agreeing. Under a Lean that can't run the verifier fails the
    same checks, the honest record's too, each as a disagreement, the history root's among them; no negative is
    refused, a verdict's row saying what stood in the Lean's place, so a5's audit, which passes only with every
    negative refused, fails."""
    mine, doc, rec, publics, _ = _window(tmp_path / "w")
    accepted, rows = _negatives(mine, doc, rec, publics)
    assert accepted and len(rows) == 8 and all(r["refused"] for r in rows.values()), rows
    assert (rows["R3"]["codes"], rows["R4"]["codes"]) == (["R3-domain"], ["R4-leaf-layer"]), rows
    assert rows["history-root"] == {"refused": True, "codes": [R.HISTORY]}
    assert all(V.DISAGREE not in rows[k]["codes"] for k in ("sample", "proved")), rows
    assert all(r["agree"] for r in rows.values() if "agree" in r), rows

    def cannot_run(fn, value):
        raise V.LeanError("the Lean cannot run")
    monkeypatch.setattr(V._server, "call", cannot_run)
    said = {"refused": "the Lean cannot run"}
    accepted, dead = _negatives(mine, doc, rec, publics)
    assert not accepted and dead.keys() == rows.keys() and not any(r["refused"] for r in dead.values()), dead
    for k in ("R3", "R4", "history-root", "sample", "proved"):
        assert dead[k] == {"refused": False, "codes": [V.DISAGREE], "lean": said}, dead[k]
    assert not any(r["agree"] for r in dead.values() if "agree" in r), dead
