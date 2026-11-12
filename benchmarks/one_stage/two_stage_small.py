"""A token select class at small statics as a serve and its replay leave it, for `two_stage.py`'s rehearsal
(``two_stage_small.sh``) and tests: no engine runs. ``serve`` writes one size class's files as the engine core commits
them (`boundary_rows.write`: ``w0/p<P>-o<O>/``, its replay block the check's instances, one a served token, its
record committing the ids the block returns as a port, `boundary_rows.Layout.ids`) and a capture of every request's
sampler inputs as a replay writes it (`engine.sampler_capture.write`): each token's logits row, random bf16 bits, and
the id `TokenSelect_v4{V}` selects from it, which is the id the class's egress row and its return's row commit.
``--units K`` makes the check the greedy select in K units, `GreedySelect_v1{V, N=V / K}`, which selects the same id.
``expect`` reads a ``two_stage.py second`` summary and says whether the run held, check by check: round 2 accepting
the statements, the return's read at round 1's rows among them, and the stream's outcome accepted or refused by the
two-stage form alone. ``--inline`` writes the class as a served vocabulary's is, its replay block the check's body
inlined. `TokenSelect_v4`'s body is chunks and a combine a token, which round 2 derives no wires from and whose record
commits no return: ``second --unlinked`` runs it, and ``expect --unlinked`` holds that run to round 2's refusal of the
wires. `GreedySelect_v1`'s body is one unit of each of its K templates a token, each wired row one chunk's output row,
which round 2 derives its wires and its return link from: its record commits the ids, and ``second`` runs it linked.
``--stop`` serves the class under a stop rule (`boundary_stop.Stop`, T the outputs): its one stop id is the id the
check selects at the first request's second token, every request ends at its first stop id or at its ``max_tokens``,
as an engine ends it, and all of them are the one stop class's, (P, T); ``first`` and ``second`` take the rule as
``--stop``.
The window's run is the serve nonce's (``--serve``, `SERVE` unless given): a challenger host opens a run's root once,
ever, so a first stage on one takes a nonce no run has used.

    python two_stage_small.py serve --out DIR [--v 128] [--outputs 4] [--spans 6] [--inline] [--units K] [--stop] [--serve NONCE]
    python two_stage_small.py expect SUMMARY.json [--unlinked]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.circuits.evaluation.reference import evaluate
from verity_vllm.engine import sampler_capture as SC
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS

SERVE = "two-stage-small"
PROMPT = (1, 2, 3)
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
          "ignore_eos": True, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
#: the model length a stop rule registers here
MAX_MODEL_LEN = 4096
#: the stream's code for the two-stage form, which it refuses every run by, last, until the form's proof lands
FORM = "two-stage-form"
#: round 2's code for a replay block it derives no wires from: an inlined class's
UNWIRED = "second-wires"


def template(v: int, units: int = 0) -> str:
    """The check: `TokenSelect_v4{V}`, or with ``units`` = K the greedy select in K units, `GreedySelect_v1{V, N=V / K}`."""
    if units and v % units:
        raise ValueError(f"a vocabulary of {v} is not {units} whole chunks")
    return f"GreedySelect_v1{{V={v},N={v // units}}}" if units else f"TokenSelect_v4{{V={v}}}"


def token(out: Sequence[int]) -> int:
    """The id a check's output is: its one 32-bit word, or its (low, high) 16-bit words."""
    return int(out[0]) if len(out) == 1 else int(out[0]) | int(out[1]) << 16


def layout(v: int, outputs: int, spans: int, inline: bool = False, nonce: str = SERVE, units: int = 0,
           stop: BS.Stop | None = None) -> BR.Layout:
    """The class's layout, a stop class of ``stop`` where given. Its record commits the ids its replay block returns
    wherever round 2 derives that return's link: a class of whole instances, and an inlined class of the greedy select
    in units (``units``)."""
    return BR.Layout(BR.check_template(template(v, units)), 0, 1, BR.cap(len(PROMPT)), outputs, spans, BR.run_root(nonce, 0),
                     inline, ids=bool(units) or not inline, stop=stop)


def serve(out: Path, v: int = 128, outputs: int = 4, spans: int = 6, seed: int = 1108, inline: bool = False,
          nonce: str = SERVE, units: int = 0, stop: bool = False) -> dict[str, Any]:
    """The class's directory and the capture under ``out``: ``spans`` requests ``r<i>`` of ``outputs`` tokens over a
    vocabulary of ``v``, each captured under a launch of its own, in the window of the serve ``nonce``. ``units``: K,
    the check the greedy select in K units (`template`). ``stop``: under a stop rule whose one id is the first request's
    second id, each request ended at its first stop id. -> the check template's id, the two directories, the rule (or
    None), and each request's rows and ids."""
    check = BR.check_template(template(v, units))
    rng = np.random.default_rng(seed)
    rows = {f"r{i}": rng.integers(0, 1 << 16, (outputs, v), dtype=np.uint16) for i in range(spans)}
    toks = {r: [token(evaluate(check, x.tolist())) for x in a] for r, a in rows.items()}
    rule, sampling, finish = None, {**GREEDY, "max_tokens": outputs}, dict.fromkeys(rows, "length")
    if stop:
        rule, sampling = BS.Stop((toks["r0"][min(1, outputs - 1)],), MAX_MODEL_LEN, outputs), {**sampling, "ignore_eos": False}
        for r, t in toks.items():
            if rule.ids[0] in t:
                toks[r], finish[r] = t[:t.index(rule.ids[0]) + 1], "stop"
                rows[r] = rows[r][:len(toks[r])]
    L = layout(v, outputs, spans, inline, nonce, units, rule)
    directory = out / "w0" / f"p{L.prompt}-o{L.outputs}"
    spans_ = [BR.Span(r, PROMPT, sampling, tuple(toks[r]), finish[r]) for r in rows]
    if rule is not None and any(rule.word(s) is None for s in spans_):
        raise ValueError("a request the stop rule doesn't end as it was served: it would be filed without a stop block")
    BR.write(BR.commit(spans_, L, 0), directory)
    SC.write(out / "capture", template(v, units), {r: {"logits": rows[r], "tokens": toks[r], "steps": list(range(len(toks[r]))),
                                                 "program": f"launch:{r}"} for r in rows})
    return {"check": template(v, units), "class": directory, "capture": out / "capture", "stop": rule, "rows": rows,
            "tokens": toks}


def expect(summary: Mapping[str, Any], code: str | None = None) -> dict[str, bool]:
    """A ``two_stage.py second`` summary, check by check: every check before the sample passed, each session's circuit
    file is its template's params', every Lean verdict accepted, every negative refused, and round 2 and the stream's
    outcome are the run's. For a record of round 1 that commits the block's return (``code`` None): round 2 accepts
    the statements with no code, and the outcome is accepted, or refused by its interior record's statements check
    with the two-stage form's code alone (`FORM`), and the one-stage audit's own record of the interior, which holds no
    check of round 2's, fails none of its own (``record``). For an unlinked run (``code`` `UNWIRED`): round 2 refuses
    the statements by that code alone, and so does the outcome. Either way its replay sample's record fails nothing."""
    audit, o, r2 = summary.get("audit") or {}, summary.get("outcome") or {}, summary.get("round_two") or {}
    inner = audit.get("audit") or {}
    negatives = [v if isinstance(v, bool) else bool(v.get("refused")) for v in (audit.get("negatives") or {}).values()]
    refused = o.get("accepted") is False and o.get("interior") == ["statements"]
    if code is None:
        r2_ok = r2.get("ok") is True and r2.get("codes") == []
        o_ok = (o.get("accepted") is True and o.get("statements") == []) or (refused and o.get("statements") == [FORM])
        record_ok = (summary.get("record") or {}).get("failed") == []
    else:
        r2_ok, o_ok, record_ok = r2.get("codes") == [code], refused and o.get("statements") == [code], True
    return {"before_draw": bool(audit.get("checks_before_draw")) and all(audit["checks_before_draw"].values()),
            "pins": bool(summary.get("pins")) and all(summary["pins"].values()),
            "lean_verdicts": bool(inner.get("verdicts")) and all(v is True for v in inner["verdicts"].values()),
            "negatives": bool(negatives) and all(negatives), "record": record_ok,
            "round_two": r2_ok, "outcome": o_ok and o.get("replay_sample") == []}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--v", type=int, default=128)
    s.add_argument("--outputs", type=int, default=4)
    s.add_argument("--spans", type=int, default=6)
    s.add_argument("--inline", action="store_true")
    s.add_argument("--units", type=int, default=0, help="K: the check is GreedySelect_v1{V, N=V / K}, the greedy select in K units")
    s.add_argument("--stop", action="store_true", help="under a stop rule (T the outputs), printed as stop for first and second's --stop")
    s.add_argument("--serve", default=SERVE, help="the serve's nonce, which the window's run is of")
    e = sub.add_parser("expect")
    e.add_argument("summary", type=Path)
    e.add_argument("--unlinked", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        got = serve(a.out, a.v, a.outputs, a.spans, inline=a.inline, nonce=a.serve, units=a.units, stop=a.stop)
        print(json.dumps({**{k: str(got[k]) for k in ("check", "class", "capture")},
                          **({"stop": json.dumps(got["stop"].to_json())} if got["stop"] else {})}))
        return 0
    checks = expect(json.loads(a.summary.read_text()), UNWIRED if a.unlinked else None)
    print("SMALL " + json.dumps(checks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
