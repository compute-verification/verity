"""drawn.py: the CLI of a served H1 window's drawn J10-R units staged as C-Flock statements and checked as credit does
(`verity_vllm.protocols.pouw.h1.drawn`, whose docstring is the method), with ``--checkpoint``'s safetensors read by the
live challenger's model registry (`live_challenger.models`) and ``check`` credit's own verification (`credit._verified`
with the Lean ``flock-verify`` at ``--lean``, then `credit._proved`).

    python drawn.py stage SERVE_DIR WINDOW --circuits DIR --work W [--prover P] [--k K] [--recompose]
                          [--weights-key FILE | --gate CMD --model ROLE] [--checkpoint DIR | --weights NPZ]
    python drawn.py statements --work W --out FILE
    python drawn.py check SERVE_DIR WINDOW --circuits DIR --statements FILE --lean FLOCK_VERIFY
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np

from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
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


def verified(statements: list, files: dict, win: dict, units: list[int], lean: Path) -> tuple[list | None, tuple | None]:
    """Credit's verification of a window's drawn-call statements (`credit._verified`, then `credit._proved`), its Lean
    verifier the ``flock-verify`` at `lean`."""
    PF.executable = lambda: lean
    proofs, why = CR._verified(statements, files, win["raw"], win["layout"], win["entries"], win["root"], units,
                               win["steps"], PF.verify, win["salt"])
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
    a = ap.parse_args(argv)
    a.checkpoint_weights, a.verified = checkpoint_weights, verified
    return D.run(a.cmd, a)


if __name__ == "__main__":
    raise SystemExit(main())
