"""A token select class at small statics as a serve and its replay leave it, for `two_stage.py`'s rehearsal
(``two_stage_small.sh``) and tests: no engine runs. ``serve`` writes one size class's files as the engine core commits
them (`boundary_rows.write`: ``w0/p<P>-o<O>/``, its replay block the check's instances, one a served token, its
record committing the ids the block returns as a port, `boundary_rows.Layout.ids`) and a capture of every request's
sampler inputs as a replay writes it (`engine.sampler_capture.write`): each token's logits row, random bf16 bits, and
the id `TokenSelect_v4{V}` selects from it, which is the id the class's egress row and its return's row commit.
``expect`` reads a ``two_stage.py second`` summary and says whether the run held, check by check: round 2 accepting
the statements, the return's read at round 1's rows among them, and the stream's outcome accepted or refused by the
two-stage form alone. ``--inline`` writes the class as a served vocabulary's is, its replay block the check's body
inlined (chunks and a combine a token), which round 2 derives no wires from and whose record commits no return:
``second --unlinked`` runs it, and ``expect --unlinked`` holds that run to round 2's refusal of the wires.
The window's run is the serve nonce's (``--serve``, `SERVE` unless given): a challenger host opens a run's root once,
ever, so a first stage on one takes a nonce no run has used.

    python two_stage_small.py serve --out DIR [--v 128] [--outputs 4] [--spans 6] [--inline] [--units K] [--serve NONCE]
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
from verity_vllm.pipeline import boundary_rows as BR

SERVE = "two-stage-small"
PROMPT = (1, 2, 3)
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
          "ignore_eos": True, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
#: the stream's code for the two-stage form, which it refuses every run by, last, until the form's proof lands
FORM = "two-stage-form"
#: round 2's code for a replay block it derives no wires from: an inlined class's
UNWIRED = "second-wires"


def template(v: int, units: int = 0) -> str:
    """The check: `TokenSelect_v4{V}`, or with ``units`` = K the token select by units, `TokenSelectUnits_v1{V, K}`."""
    return f"TokenSelectUnits_v1{{V={v},K={units}}}" if units else f"TokenSelect_v4{{V={v}}}"


def layout(v: int, outputs: int, spans: int, inline: bool = False, nonce: str = SERVE, units: int = 0) -> BR.Layout:
    return BR.Layout(BR.check_template(template(v, units)), 0, 1, BR.cap(len(PROMPT)), outputs, spans, BR.run_root(nonce, 0),
                     inline, ids=not inline)


def serve(out: Path, v: int = 128, outputs: int = 4, spans: int = 6, seed: int = 1108, inline: bool = False,
          nonce: str = SERVE, units: int = 0) -> dict[str, Any]:
    """The class's directory and the capture under ``out``: ``spans`` requests ``r<i>`` of ``outputs`` tokens over a
    vocabulary of ``v``, each captured under a launch of its own, in the window of the serve ``nonce``. ``units``: K,
    the check the token select by units (`template`). -> the check template's id, the two directories, and each
    request's rows and ids."""
    L = layout(v, outputs, spans, inline, nonce, units)
    rng = np.random.default_rng(seed)
    rows = {f"r{i}": rng.integers(0, 1 << 16, (outputs, v), dtype=np.uint16) for i in range(spans)}
    toks = {r: [int(evaluate(L.check, x.tolist())[0]) for x in a] for r, a in rows.items()}
    directory = out / "w0" / f"p{L.prompt}-o{L.outputs}"
    BR.write(BR.commit([BR.Span(r, PROMPT, {**GREEDY, "max_tokens": outputs}, tuple(toks[r]), "length") for r in rows], L, 0),
             directory)
    SC.write(out / "capture", template(v, units), {r: {"logits": rows[r], "tokens": toks[r], "steps": list(range(outputs)),
                                                 "program": f"launch:{r}"} for r in rows})
    return {"check": template(v, units), "class": directory, "capture": out / "capture", "rows": rows, "tokens": toks}


def expect(summary: Mapping[str, Any], code: str | None = None) -> dict[str, bool]:
    """A ``two_stage.py second`` summary, check by check: every check before the sample passed, each session's circuit
    file is its template's params', every Lean verdict accepted, every negative refused, and round 2 and the stream's
    outcome are the run's. For a record of round 1 that commits the block's return (``code`` None): round 2 accepts
    the statements with no code, and the outcome is accepted, or refused by its interior record's statements check
    with the two-stage form's code alone (`FORM`). For an unlinked run (``code`` `UNWIRED`): round 2 refuses the
    statements by that code alone, and so does the outcome. Either way its replay sample's record fails nothing."""
    audit, o, r2 = summary.get("audit") or {}, summary.get("outcome") or {}, summary.get("round_two") or {}
    inner = audit.get("audit") or {}
    negatives = [v if isinstance(v, bool) else bool(v.get("refused")) for v in (audit.get("negatives") or {}).values()]
    refused = o.get("accepted") is False and o.get("interior") == ["statements"]
    if code is None:
        r2_ok = r2.get("ok") is True and r2.get("codes") == []
        o_ok = (o.get("accepted") is True and o.get("statements") == []) or (refused and o.get("statements") == [FORM])
    else:
        r2_ok, o_ok = r2.get("codes") == [code], refused and o.get("statements") == [code]
    return {"before_draw": bool(audit.get("checks_before_draw")) and all(audit["checks_before_draw"].values()),
            "pins": bool(summary.get("pins")) and all(summary["pins"].values()),
            "lean_verdicts": bool(inner.get("verdicts")) and all(v is True for v in inner["verdicts"].values()),
            "negatives": bool(negatives) and all(negatives),
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
    s.add_argument("--units", type=int, default=0, help="K: the check is TokenSelectUnits_v1{V, K}, the token select by units")
    s.add_argument("--serve", default=SERVE, help="the serve's nonce, which the window's run is of")
    e = sub.add_parser("expect")
    e.add_argument("summary", type=Path)
    e.add_argument("--unlinked", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        got = serve(a.out, a.v, a.outputs, a.spans, inline=a.inline, nonce=a.serve, units=a.units)
        print(json.dumps({k: str(got[k]) for k in ("check", "class", "capture")}))
        return 0
    checks = expect(json.loads(a.summary.read_text()), UNWIRED if a.unlinked else None)
    print("SMALL " + json.dumps(checks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
