"""A stop class at small statics as a serve leaves it, and the wrong responses a stop population's audit refuses, for
`stop_audit.py`'s rehearsal (``stop_audit_small.sh``) and tests: no engine runs.

    python stop_audit_small.py serve --out DIR [--outputs 32] [--spans 6] [--serve NONCE]
    python stop_audit_small.py unlogged HANDOFF.json     # case 2's hand-off, to stdout
    python stop_audit_small.py unaccepted HANDOFF.json   # case 3's boundary outcome, to stdout
    python stop_audit_small.py incomplete HANDOFF.json   # case 4's boundary outcome, to stdout
    python stop_audit_small.py expect DIR                # the five runs' summaries under DIR

``serve`` writes two classes of ``spans`` requests under the stop rule `stop_rule`, each request ended at the stop id
or at its ``max_tokens``, the check `TokenSelect_v4{V}`'s:

* ``DIR/w0/p<P>-o<O>/``, the class as the engine core commits it (`boundary_rows.write`), in the window of the serve
  ``NONCE``;
* ``DIR/wrong/w0/p<P>-o<O>/``, the same requests in the window of the serve ``NONCE-wrong`` (a challenger opens a
  run's root once), and beside it ``DIR/wrong/registered/p<P>-o<O>/commit.json``: that class's commitment with
  `b.stop`'s root the tree's whose every stop word has bit 0 flipped (`stop_audit.opened`), a length that is not the
  rule's for every request. A first stage registers the second; `stop_audit.py --forge` proves it from the first's
  files.

``unlogged``: a hand-off whose stop record is built on a boundary record the class's boundary stream never logged
(the class's own, chained after a record that is not there). ``unaccepted``: an outcome record on the class's boundary
registration that did not accept, as a boundary stream that refused its sample would log it. ``incomplete``: one
that accepted with an entry that did not run, so its audit record is not complete. Today's two-stage streams reach no
outcome, so both records are written here and no stream's.

``expect`` holds the rehearsal's five runs to what each must be, each with the case Lean's verdict on the read across
the class's two streams gives it (`stop_audit.across`): ``accept`` accepted by Lean's verdict of record with every
entry that ran accepted, not complete, and no length held, case 3 since no boundary outcome is handed; ``length``
refused by Lean's verdict on the session alone; ``unlogged`` refused as case 2 before any session, Lean's verdict
that the boundary stream's log holds no such record; ``unaccepted`` refused by the boundary-audit entry alone, case 3;
``incomplete`` accepted as ``accept`` is, the boundary-audit entry without a verdict and with its reason, not
complete, and no length held, case 4."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stop_audit as SA  # noqa: E402
import two_stage as TS  # noqa: E402
from verity.protocols.spot.verifier import service as S  # noqa: E402
from verity.protocols.spot.verifier.one_stage import audit as A, registration as R  # noqa: E402
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS  # noqa: E402

SERVE = "stop-audit-small"
PROMPT = (1, 2, 3)
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
            "ignore_eos": False, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
STOP_ID, MAX_MODEL_LEN = 3, 64
#: a request ended at the stop id, and one ended at its max_tokens
ENDS = (((7, 8, STOP_ID), "stop"), ((7, 8, 9, 9), "length"))
RUNS = ("accept", "length", "unlogged", "unaccepted", "incomplete")


def template(v: int) -> str:
    return f"TokenSelect_v4{{V={v}}}"


def stop_rule(outputs: int = 32) -> BS.Stop:
    return BS.Stop((STOP_ID,), MAX_MODEL_LEN, outputs)


def committed(v: int = 128, outputs: int = 32, spans: int = 6, nonce: str = SERVE, check: Any = None) -> BR.Committed:
    """The class's commitment: ``spans`` requests under `stop_rule`, alternately ended at the stop id and at
    ``max_tokens``, in the window of the serve ``nonce``. ``check``: another check template than `TokenSelect_v4{V}`."""
    check = BR.check_template(template(v)) if check is None else check
    L = BR.Layout(check, 0, 1, BR.cap(len(PROMPT)), outputs, spans, BR.run_root(nonce, 0), stop=stop_rule(outputs), held=True)
    return BR.commit([BR.Span(f"r{i}", PROMPT, SAMPLING, *ENDS[i % len(ENDS)]) for i in range(spans)], L, 0)


def wrong_length(c: BR.Committed) -> dict[str, Any]:
    """``c``'s ``commit.json`` as a prover registers it that committed, for every request, a stop word whose bit 0 is
    flipped: `b.stop`'s root is that tree's (`stop_audit.opened`), every other value the class's own."""
    pub = json.loads(json.dumps(c.public()))
    pub["values"][BR.STOP]["root"] = SA.opened(c, range(c.layout.spans))["roots"][BR.STOP]
    return pub


def serve(out: Path, v: int = 128, outputs: int = 32, spans: int = 6, nonce: str = SERVE) -> dict[str, Any]:
    """The two classes under ``out`` (module docstring). -> the check template's id, the rule as ``--stop`` takes it,
    the class's directory, and the wrong class's two: the one a first stage registers and the one its prover reads."""
    c, w = committed(v, outputs, spans, nonce), committed(v, outputs, spans, f"{nonce}-wrong")
    name = f"p{c.layout.prompt}-o{c.layout.outputs}"
    BR.write(c, out / "w0" / name)
    BR.write(w, out / "wrong" / "w0" / name)
    registered = out / "wrong" / "registered" / name
    registered.mkdir(parents=True, exist_ok=True)
    (registered / BR.COMMIT).write_text(json.dumps(wrong_length(w), indent=1) + "\n")
    return {"check": template(v), "stop": json.dumps(stop_rule(outputs).to_json()), "class": out / "w0" / name,
            "wrong": registered, "wrong_files": out / "wrong" / "w0" / name}


def unlogged(handoff: Mapping[str, Any], check: Any = None) -> dict[str, Any]:
    """``handoff`` with its stop record built on a boundary record the class's boundary stream never logged: the
    class's own record chained after one that is not there, its values and roots unchanged."""
    pub, s = handoff["commit"], handoff["stop"]
    check = TS.template_fn(pub["check"]) if check is None else check
    other = {**handoff["record"], "previous": "5a" * 64}
    record = BS.stop_record(pub, check, BS.Stop.of(pub["stop"]), s["record"]["law"], other)
    assert record["previous"] == R.digest(other) != handoff["registration"]
    return {**handoff, "stop": {**s, "record": record}}


def unaccepted(handoff: Mapping[str, Any]) -> dict[str, Any]:
    """An outcome record on the class's boundary registration that did not accept (module docstring): its audit
    record names that registration and holds no verdict."""
    audit = {"format": A.FORMAT, "registration_digest": handoff["registration"], "accepted": False, "complete": True,
             "checks": {}, "verdicts": {}}
    return {"format": S.OUTCOME, "stream": handoff["spec"], "order": 0, "registration": handoff["registration"],
            "receipt": SA.sha512(dict(handoff["receipt"])), "draw": SA.sha512(dict(handoff["draw"])), "sessions": [],
            "symbol": S.REJECT, "public": {}, "audit": audit}


def incomplete(handoff: Mapping[str, Any]) -> dict[str, Any]:
    """An outcome record on the class's boundary registration that accepted with an entry that did not run (module
    docstring): its audit record names that registration, accepted on its verdict of record, and is not complete."""
    got = unaccepted(handoff)
    verdicts = {"lean-verify/boundary": {"accepted": True, "of_record": True}, "not-run": {"accepted": None, "of_record": False}}
    return {**got, "symbol": S.ACCEPT, "audit": {**got["audit"], "accepted": True, "complete": False, "verdicts": verdicts}}


def expect(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """The five runs' summaries (`stop_audit.py`'s ``summary.json`` by run, `RUNS`) against what each must be."""
    a, w, u, b, i = (summaries.get(k) or {} for k in RUNS)
    ran = {SA.LEAN, SA.M0, SA.READS, SA.ORDER}
    null = {SA.BOUNDARY_AUDIT, SA.REQUEST_ROW, SA.STOP_RULE}
    va, vw, vb, vi = (x.get("verdicts") or {} for x in (a, w, b, i))
    return {
        "accept": a.get("accepted") is True and a.get("symbol") == S.ACCEPT and a.get("failed") == []
        and all(va.get(k) is True for k in ran) and all(k in va and va[k] is None for k in null),
        "accept_is_not_complete": a.get("complete") is False and a.get("length_held") is False
        and a.get("not_run") == sorted(null) and a.get("case") == 3
        and {SA.BOUNDARY_PREMISE, SA.STOP_RULE_PREMISE, SA.REQUEST_PREMISE} == set(a.get("premises") or ()),
        "length": w.get("accepted") is False and vw.get(SA.LEAN) is False and vw.get(SA.ORDER) is True
        and w.get("length_held") is False,
        "unlogged": u.get("accepted") is False and u.get("refused") == [SA.ORDER] and u.get("case") == 2
        and ((u.get("across") or {}).get("logged") or {}).get("accepted") is False and "verdicts" not in u,
        "unaccepted": b.get("accepted") is False and vb.get(SA.BOUNDARY_AUDIT) is False and b.get("case") == 3
        and all(vb.get(k) is True for k in ran) and b.get("failed") == [] and b.get("length_held") is False,
        "incomplete": i.get("accepted") is True and i.get("symbol") == S.ACCEPT and i.get("failed") == []
        and all(vi.get(k) is True for k in ran) and all(k in vi and vi[k] is None for k in null) and i.get("case") == 4
        and i.get("complete") is False and i.get("length_held") is False and i.get("not_run") == sorted(null)
        and (i.get("why") or {}).get(SA.BOUNDARY_AUDIT) == SA.NOT_COMPLETE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--v", type=int, default=128)
    s.add_argument("--outputs", type=int, default=32)
    s.add_argument("--spans", type=int, default=6)
    s.add_argument("--serve", default=SERVE, help="the serve's nonce, which the window's run is of")
    for name in ("unlogged", "unaccepted", "incomplete"):
        sub.add_parser(name).add_argument("handoff", type=Path)
    sub.add_parser("expect").add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        print(json.dumps({k: str(x) for k, x in serve(a.out, a.v, a.outputs, a.spans, a.serve).items()}))
        return 0
    if a.cmd in ("unlogged", "unaccepted", "incomplete"):
        made = {"unlogged": unlogged, "unaccepted": unaccepted, "incomplete": incomplete}[a.cmd]
        print(json.dumps(made(json.loads(a.handoff.read_text())), indent=1))
        return 0
    summaries = {k: json.loads(f.read_text()) for k in RUNS if (f := a.dir / k / "summary.json").exists()}
    checks = expect(summaries)
    print("SMALL " + json.dumps(checks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

