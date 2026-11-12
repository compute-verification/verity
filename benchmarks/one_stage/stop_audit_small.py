"""A stop class at small statics as a serve leaves it, and the wrong responses a stop population's audit refuses, for
`stop_audit.py`'s rehearsal (``stop_audit_small.sh``) and tests: no engine runs.

    python stop_audit_small.py serve --out DIR [--outputs 32] [--spans 6] [--serve NONCE]
    python stop_audit_small.py unlogged STAGE1.json --words F --certifier F   # case 2's hand-off, to stdout
    python stop_audit_small.py unaccepted STAGE1.json    # case 3's boundary outcome, to stdout
    python stop_audit_small.py incomplete STAGE1.json    # case 4's boundary outcome, to stdout
    python stop_audit_small.py expect DIR                # the seven runs' summaries under DIR

``serve`` writes three classes of ``spans`` requests under the stop rule `stop_rule`, each request ended at the stop
id or at its ``max_tokens``, the check `TokenSelect_v4{V}`'s, and beside each class the files of a stand-in network
certifier that released its requests (`stop_words.sign`, under ``certifier/``: the signed head of the class's words
table, the verifier's copy of the certifier, and the table's opening for the prover), one key signing every head:

* ``DIR/w0/p<P>-o<O>/``, the class as the engine core commits it (`boundary_rows.write`), in the window of the serve
  ``NONCE``, with the certifier's table of the requests the engine served (``DIR/certifier/``); and
  ``DIR/changed/``, a second table of that class and window in which every request asked for one token fewer than
  the engine served it under (`changed`): the head a certifier signs that released another ``max_tokens``.
* ``DIR/wrong/w0/p<P>-o<O>/``, the same requests in the window of the serve ``NONCE-wrong`` (a challenger opens a
  run's root once), and beside it ``DIR/wrong/registered/p<P>-o<O>/commit.json``: that class's commitment with
  `b.stop`'s root the tree's whose every stop word has bit 0 flipped (`stop_audit.opened`), a length that is not the
  rule's for every request. A first stage registers the second; `stop_audit.py --forge` proves it from the first's
  files.
* ``DIR/words/w0/p<P>-o<O>/``, the same requests in the window of the serve ``NONCE-words``, with a certifier's
  table changed as ``DIR/changed/``'s is. A first stage registers the class under that table;
  `stop_audit.py --engine-words` proves its stop blocks at the engine's own ``max_tokens``.

``unlogged``: a hand-off whose stop record is built on a boundary record the class's boundary stream never logged
(the class's own, chained after a record that is not there). ``unaccepted``: an outcome record on the class's boundary
registration that did not accept, as a boundary stream that refused its sample would log it. ``incomplete``: one
that accepted with an entry that did not run, so its audit record is not complete. Today's two-stage streams reach no
outcome, so both records are written here and no stream's.

``expect`` holds the rehearsal's seven runs to what each must be, each with the case Lean's verdict on the read across
the class's two streams gives it (`stop_audit.across`): ``accept`` accepted by Lean's verdict of record with every
entry that ran accepted, the certified words' among them, not complete, and no length held, case 3 since no boundary
outcome is handed; ``length``
refused by Lean's verdict on the session alone; ``unlogged`` refused as case 2 before any session, Lean's verdict
that the boundary stream's log holds no such record; ``unaccepted`` refused by the boundary-audit entry alone, case 3;
``incomplete`` accepted as ``accept`` is, the boundary-audit entry without a verdict and with its reason, not
complete, and no length held, case 4; ``table``, the class's own hand-off audited under the changed table, refused
before any session by the stop stream's registration (``R7-certified``: the stop record registers the table the
blocks were proved at, and the verifier's read gives another root); and ``words``, the class registered under the
changed table and proved at the engine's ``max_tokens``, refused by Lean's verdict on the session and by the
certified-words entry (``stop-read-root``), whose reason names the first row that does not open the certified root."""
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
import stop_words as SW  # noqa: E402
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
RUNS = ("accept", "length", "unlogged", "unaccepted", "incomplete", "table", "words")


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


def changed(c: BR.Committed) -> dict[int, int]:
    """The changed certifier's parse of ``c``'s requests (`stop_words.parsed`'s ``max_tokens``): every request asked
    for one token fewer than the engine served it under, so whichever stop blocks a sample names, each is proved at
    another ``max_tokens`` than the certifier's."""
    return {i: s.sampling["max_tokens"] - 1 for i, s in enumerate(c.spans)}


def serve(out: Path, v: int = 128, outputs: int = 32, spans: int = 6, nonce: str = SERVE) -> dict[str, Any]:
    """The three classes and their certifier's files under ``out`` (module docstring). -> the check template's id,
    the rule as ``--stop`` takes it, the verifier's copy of the certifier, the class's directory, its certifier's
    files' and the changed table's, the wrong class's two (the one a first stage registers and the one its prover
    reads) and its certifier's, and the class a changed certifier released with that certifier's."""
    c, w, x = (committed(v, outputs, spans, n) for n in (nonce, f"{nonce}-wrong", f"{nonce}-words"))
    name = f"p{c.layout.prompt}-o{c.layout.outputs}"
    BR.write(c, out / "w0" / name)
    BR.write(w, out / "wrong" / "w0" / name)
    BR.write(x, out / "words" / "w0" / name)
    registered = out / "wrong" / "registered" / name
    registered.mkdir(parents=True, exist_ok=True)
    (registered / BR.COMMIT).write_text(json.dumps(wrong_length(w), indent=1) + "\n")
    issuer = SW.certifier(out / "signer-state.json")
    signed = {
        "certified": SW.sign(issuer, c.spans, 0, out / "certifier", link=nonce),
        "changed": SW.sign(issuer, c.spans, 0, out / "changed", link=nonce, max_tokens=changed(c)),
        "wrong_certified": SW.sign(issuer, w.spans, 0, out / "wrong" / "certifier", link=f"{nonce}-wrong"),
        "words_certified": SW.sign(issuer, x.spans, 0, out / "words" / "certifier", link=f"{nonce}-words",
                                   max_tokens=changed(x)),
    }
    return {"check": template(v), "stop": json.dumps(stop_rule(outputs).to_json()), "class": out / "w0" / name,
            "wrong": registered, "wrong_files": out / "wrong" / "w0" / name, "words": out / "words" / "w0" / name,
            "certifier": signed["certified"]["certifier"], **{k: f["words"].parent for k, f in signed.items()}}


def unlogged(stage1: Mapping[str, Any], words: Mapping[str, Any], check: Any = None) -> dict[str, Any]:
    """``stage1`` with its stop record built on a boundary record the class's boundary stream never logged: the
    class's own record chained after one that is not there, its values and roots unchanged. ``words``: the verifier's
    port of the class's certified words table, which the record registers as the class's own does."""
    pub, s = stage1["commit"], stage1["stop"]
    check = TS.template_fn(pub["check"]) if check is None else check
    other = {**stage1["record"], "previous": "5a" * 64}
    record = BS.stop_record(pub, check, BS.Stop.of(pub["stop"]), s["record"]["law"], other, words)
    assert record["previous"] == R.digest(other) != stage1["registration"]
    return {**stage1, "stop": {**s, "record": record}}


def unaccepted(stage1: Mapping[str, Any]) -> dict[str, Any]:
    """An outcome record on the class's boundary registration that did not accept (module docstring): its audit
    record names that registration and holds no verdict."""
    audit = {"format": A.FORMAT, "registration_digest": stage1["registration"], "accepted": False, "complete": True,
             "checks": {}, "verdicts": {}}
    return {"format": S.OUTCOME, "stream": stage1["spec"], "order": 0, "registration": stage1["registration"],
            "receipt": SA.sha512(dict(stage1["receipt"])), "draw": SA.sha512(dict(stage1["draw"])), "sessions": [],
            "symbol": S.REJECT, "public": {}, "audit": audit}


def incomplete(stage1: Mapping[str, Any]) -> dict[str, Any]:
    """An outcome record on the class's boundary registration that accepted with an entry that did not run (module
    docstring): its audit record names that registration, accepted on its verdict of record, and is not complete."""
    got = unaccepted(stage1)
    verdicts = {"lean-verify/boundary": {"accepted": True, "of_record": True}, "not-run": {"accepted": None, "of_record": False}}
    return {**got, "symbol": S.ACCEPT, "audit": {**got["audit"], "accepted": True, "complete": False, "verdicts": verdicts}}


def expect(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    """The seven runs' summaries (`stop_audit.py`'s ``summary.json`` by run, `RUNS`) against what each must be."""
    a, w, u, b, i, t, x = (summaries.get(k) or {} for k in RUNS)
    ran = {SA.LEAN, SA.M0, SA.READS, SA.ORDER, SA.WORDS}
    null = {SA.BOUNDARY_AUDIT, SA.STOP_RULE}
    va, vw, vb, vi, vx = (y.get("verdicts") or {} for y in (a, w, b, i, x))
    return {
        "accept": a.get("accepted") is True and a.get("symbol") == S.ACCEPT and a.get("failed") == []
        and all(va.get(k) is True for k in ran) and all(k in va and va[k] is None for k in null),
        "accept_is_not_complete": a.get("complete") is False and a.get("length_held") is False
        and a.get("not_run") == sorted(null) and a.get("case") == 3
        and {SA.BOUNDARY_PREMISE, SA.STOP_RULE_PREMISE, SA.WORDS_PREMISE} == set(a.get("premises") or ()),
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
        "table": t.get("accepted") is False and t.get("refused") == ["R7-certified"] and "verdicts" not in t
        and t.get("length_held") is False,
        "words": x.get("accepted") is False and vx.get(SA.LEAN) is False and vx.get(SA.WORDS) is False
        and vx.get(SA.ORDER) is True and x.get("length_held") is False
        and ((x.get("why") or {}).get(SA.WORDS) or "").startswith(f"{BS.READ_ROOT}: "),
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
        sub.add_parser(name).add_argument("stage1", type=Path)
    sub.choices["unlogged"].add_argument("--words", required=True, help=SW.WORDS_HELP)
    sub.choices["unlogged"].add_argument("--certifier", required=True, help=SW.CERTIFIER_HELP)
    sub.add_parser("expect").add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        print(json.dumps({k: str(x) for k, x in serve(a.out, a.v, a.outputs, a.spans, a.serve).items()}))
        return 0
    if a.cmd == "unlogged":
        print(json.dumps(unlogged(json.loads(a.stage1.read_text()), SW.port_of(a.words, a.certifier)), indent=1))
        return 0
    if a.cmd in ("unaccepted", "incomplete"):
        made = {"unaccepted": unaccepted, "incomplete": incomplete}[a.cmd]
        print(json.dumps(made(json.loads(a.stage1.read_text())), indent=1))
        return 0
    summaries = {k: json.loads(f.read_text()) for k in RUNS if (f := a.dir / k / "summary.json").exists()}
    checks = expect(summaries)
    print("SMALL " + json.dumps(checks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

