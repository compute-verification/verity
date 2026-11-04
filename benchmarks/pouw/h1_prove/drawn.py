"""drawn.py: the CLI of a served H1 window's drawn units staged as C-Flock statements and checked as credit does
(`verity_vllm.protocols.pouw.h1.drawn`, whose docstring is the method), with ``--checkpoint``'s safetensors read by the
live challenger's model registry (`live_challenger.models`) and ``check`` credit's own verification (`credit._verified`
with the Lean ``flock-verify`` at ``--lean``, then `credit._proved`), its Lean runs ``--jobs`` at a time
(`verity_vllm.protocols.pouw.h1.answers`), one ``LEAN`` line a verdict.

    python drawn.py stage SERVE_DIR WINDOW --circuits DIR --work W [--prover P] [--k K] [--recompose]
                          [--weights-key FILE | --gate CMD --model ROLE] [--checkpoint DIR | --weights NPZ]
    python drawn.py statements --work W --out FILE
    python drawn.py check SERVE_DIR WINDOW --circuits DIR --statements FILE --lean FLOCK_VERIFY [--jobs J]
                          [--headroom-gb GB]
    python drawn.py fill SERVE_DIR WINDOW --circuits DIR --statements FILE --answers DIR [--lean FLOCK_VERIFY] [--jobs J]
                         [--headroom-gb GB]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity_vllm.protocols.pouw.h1 import answers as A
from verity_vllm.protocols.pouw.h1 import drawn as D

MODELS = Path(__file__).resolve().parents[2] / "live_challenger" / "models.py"


def _models():
    spec = importlib.util.spec_from_file_location("live_challenger_models", MODELS)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def checkpoint_weights(checkpoint: Path, linears: Sequence[tuple[str, int, int]], k: int
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each of K's linears' int8 codes (n, K) and f32 s_w (n,), in the window's order, from the checkpoint's safetensors
    (a fused linear its parts' rows in turn, `live_challenger.models.parts`)."""
    M = _models()
    ckpt = M.Checkpoint(checkpoint, sorted(p.name for p in checkpoint.iterdir() if p.name.endswith(".safetensors")))
    out = []
    for name, kk, n in linears:
        if kk != k:
            continue
        codes = [np.asarray(ckpt.tensor(f"{p}.weight")) for p in M.parts(name)]
        scales = [np.asarray(ckpt.tensor(f"{p}.weight_scale"), "<f4").reshape(-1) for p in M.parts(name)]
        w, s = np.concatenate(codes), np.concatenate(scales)
        if w.dtype != np.int8 or w.shape != (n, k) or s.shape != (n,):
            raise D.Refused(f"{name}: the checkpoint's weight {w.shape} {w.dtype} and scale {s.shape} are not its {n} × {k} "
                            "int8 codes and n scales")
        out.append((w, s))
    return out


def verified(statements: list | dict, files: dict, win: dict, units: list[int], lean: Path, jobs: int | None = None,
             headroom: float | None = None) -> tuple[list | None, tuple | None]:
    """Credit's verification of a window's drawn statements (`credit._verified` under the window's draw, then
    `credit._proved`), its Lean verifier the ``flock-verify`` at `lean`: every run `_verified` makes answered first,
    `jobs` at a time (`answers.fill`, J by `answers.jobs` of `headroom` when None), then `_verified` judging each answer
    in its order."""
    PF.executable = lambda: lean
    answers = A.Answers(by="the fill")
    runs, _refused = A.window_runs(statements, files, win["raw"], win["layout"], win["entries"], win["root"], units,
                                   win["steps"], win["salt"], win["draw"])
    A.fill(answers, runs, jobs, headroom=headroom)
    proofs, why = CR._verified(statements, files, win["raw"], win["layout"], win["entries"], win["root"], units,
                               win["steps"], answers, win["salt"], win["draw"])
    return proofs, why if why is not None else CR._proved(units, proofs)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("serve_dir")
    s.add_argument("window", type=int)
    s.add_argument("--circuits", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--prover")
    s.add_argument("--k", type=int)
    s.add_argument("--weights-key")
    s.add_argument("--gate")
    s.add_argument("--model")
    s.add_argument("--checkpoint")
    s.add_argument("--weights")
    s.add_argument("--recompose", action="store_true")
    t = sub.add_parser("statements")
    t.add_argument("--work", required=True)
    t.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("serve_dir")
    c.add_argument("window", type=int)
    c.add_argument("--circuits", required=True)
    c.add_argument("--statements", required=True)
    c.add_argument("--lean", required=True)
    c.add_argument("--k", type=int)
    f = sub.add_parser("fill")
    f.add_argument("serve_dir")
    f.add_argument("window", type=int)
    f.add_argument("--circuits", required=True)
    f.add_argument("--statements", required=True)
    f.add_argument("--answers", required=True, help="the memo: a directory in the verifier's own scratch")
    f.add_argument("--lean")
    f.add_argument("--k", type=int)
    for p in (c, f):
        p.add_argument("--jobs", type=int, help="Lean runs at once (default floor((available memory - the headroom) / "
                                                f"{A.RUN_GB:g} GiB))")
        p.add_argument("--headroom-gb", type=float, help="the available memory left beside the Lean runs when --jobs is "
                                                         f"not given (default {A.HEADROOM_GB:g} GiB)")
    a = ap.parse_args(argv)
    a.checkpoint_weights = checkpoint_weights
    a.verified = lambda *x: verified(*x, jobs=getattr(a, "jobs", None), headroom=getattr(a, "headroom_gb", None))
    if a.cmd == "fill" and a.lean:
        PF.executable = lambda: Path(a.lean)
    return D.run(a.cmd, a)


if __name__ == "__main__":
    raise SystemExit(main())
