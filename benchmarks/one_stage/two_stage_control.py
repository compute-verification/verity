"""Negative controls on the small rehearsal's path (``two_stage_small.sh``): a class whose serve committed something
other than what its check selects or what it was asked, and the prover that goes on with it. No engine runs. Each
control is the honest rehearsal's class (`two_stage_small.serve`) with one thing changed, so a run of it says which
check, if any, tells the two apart.

``serve --control id``: every id the class commits (its egress row and its return's row) is another id than the one
the check selects from the token's logits (`changed`). The capture is an honest replay's: the logits, and the ids they
select. `two_stage.py second` stops on the replay side before anything is registered (`interior_rows.captured`: the
replay didn't reproduce the span).

``serve --control id --claimed``: the same class, and a capture that claims the committed ids over the same logits.
``second`` here is `two_stage.py second` run by a prover that claims them too: the greedy select's twin
(`greedy_stages.twin`) gives the committed id as the combine's output, so the replay side's own checks pass
(`interior_rows._greedy`, the return's rows) and the interior registers the combine's units as returning round 1's
rows from logits that select other ids. What refuses it is then the sessions' and the verifier's to say.

``serve --control max-tokens``: every request is committed with a ``max_tokens`` one more than the ids it was served
and finished by length. The class's replay block reads no request field, and no stop rule is registered, so the
honest driver runs it as it runs the honest class.

``first`` is `two_stage.py first`, unchanged.

    python two_stage_control.py serve --control id|max-tokens [--claimed] --out DIR [two_stage_small.py serve's options]
    python two_stage_control.py first|second ...        # two_stage.py's own arguments
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_stage as TS  # noqa: E402
import two_stage_small as SM  # noqa: E402
from verity.circuits.evaluation.reference import evaluate  # noqa: E402
from verity.definitions.registry import greedy_stages as GS  # noqa: E402
from verity_vllm.engine import sampler_capture as SC  # noqa: E402
from verity_vllm.pipeline import boundary_rows as BR  # noqa: E402

CONTROLS = ("id", "max-tokens")


def changed(token: int, v: int) -> int:
    """Another id than ``token`` of a vocabulary of ``v``: the next one."""
    return (int(token) + 1) % v


def serve(out: Path, control: str, v: int = 128, outputs: int = 4, spans: int = 6, seed: int = 1108, inline: bool = False,
          nonce: str = SM.SERVE, units: int = 0, claimed: bool = False) -> dict[str, Any]:
    """`two_stage_small.serve`'s class and capture under ``out`` with ``control`` applied (the module docstring).
    ``claimed``: the capture's tokens are the ids the class commits, else the ids its logits select. -> as
    `two_stage_small.serve`, with ``committed``, each request's committed ids."""
    if control not in CONTROLS:
        raise ValueError(f"control {control!r} is none of {CONTROLS}")
    L = SM.layout(v, outputs, spans, inline, nonce, units)
    rng = np.random.default_rng(seed)
    rows = {f"r{i}": rng.integers(0, 1 << 16, (outputs, v), dtype=np.uint16) for i in range(spans)}
    toks = {r: [SM.token(evaluate(L.check, x.tolist())) for x in a] for r, a in rows.items()}
    committed = {r: [changed(t, v) for t in ts] for r, ts in toks.items()} if control == "id" else toks
    sampling = {**SM.GREEDY, "max_tokens": outputs + (control == "max-tokens")}
    directory = out / "w0" / f"p{L.prompt}-o{L.outputs}"
    BR.write(BR.commit([BR.Span(r, SM.PROMPT, sampling, tuple(committed[r]), "length") for r in rows], L, 0), directory)
    SC.write(out / "capture", SM.template(v, units),
             {r: {"logits": rows[r], "tokens": (committed if claimed else toks)[r], "steps": list(range(outputs)),
                  "program": f"launch:{r}"} for r in rows})
    return {"check": SM.template(v, units), "class": directory, "capture": out / "capture", "rows": rows, "tokens": toks,
            "committed": committed}


def claim() -> None:
    """Make this process's replay side claim the changed ids: `greedy_stages.twin` gives `changed` of the id its
    logits select as the combine's output, every chunk's output as it is."""
    honest = GS.twin

    def twin(V: int, logits: Any, K: int | None = None) -> tuple[int, dict]:
        token, words = honest(V, logits, K)
        token = changed(token, V)
        return token, {**words, ("combine",): [np.array([token % GS.SPAN, token // GS.SPAN], np.uint16)]}

    GS.twin = twin


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["first"]:
        return TS.main(argv)
    if argv[:1] == ["second"]:
        claim()
        return TS.main(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--control", required=True, choices=CONTROLS)
    s.add_argument("--claimed", action="store_true", help="the capture's tokens are the committed ids")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--v", type=int, default=128)
    s.add_argument("--outputs", type=int, default=4)
    s.add_argument("--spans", type=int, default=6)
    s.add_argument("--inline", action="store_true")
    s.add_argument("--units", type=int, default=0)
    s.add_argument("--serve", default=SM.SERVE)
    a = ap.parse_args(argv)
    got = serve(a.out, a.control, a.v, a.outputs, a.spans, inline=a.inline, nonce=a.serve, units=a.units, claimed=a.claimed)
    print(json.dumps({k: str(got[k]) for k in ("check", "class", "capture")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
