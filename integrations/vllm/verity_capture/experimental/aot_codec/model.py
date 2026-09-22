"""[R8 C-codec] The A3 probability model: fitted on *training* runs, frozen to JSON, loaded unchanged by encoder and decoder.

Symbols of the A2 grammar and their models
  flags  (Bernoulli, Laplace-smoothed counts [n_true+1, n_false+1]):  clean, T_eq, all_steps_match, ngen_eq_max, fr_rule, nc_eq, logits_eq, is_final
  hist   (categorical with escape, Laplace-smoothed):                arrive_delta (zig-zag arrival deltas, bins 0..K-1, escape -> gamma),
                                                                     escape_gap (predicted steps before the next escaped step, bins 0..K-1, escape -> gamma)

A model fitted to the run being evaluated is extra information and is not free: `fit()` takes only the training hints; the frozen file
records the training run ids and the file's own sha256 is written into results.jsonl for every evaluated point.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .codecs import BitCoder, grammar_a2

FLAG_KEYS = ["clean", "T_eq", "all_steps_match", "ngen_eq_max", "fr_rule", "nc_eq", "logits_eq", "is_final", "T_le_E", "nc_rule", "logits_rule"]
HIST_KEYS = {"arrive_delta": 16, "escape_gap": 8}  # K bins + escape


class CountingCoder(BitCoder):
    def __init__(self) -> None:
        super().__init__(True)
        self.flag_counts = {k: [0, 0] for k in FLAG_KEYS}
        self.sym_counts = {k: {} for k in HIST_KEYS}

    def flag(self, cat, key, value=None):
        self.flag_counts.setdefault(key, [0, 0])[0 if value else 1] += 1
        return super().flag(cat, key, value)

    def sym(self, cat, key, value=None):
        d = self.sym_counts.setdefault(key, {})
        d[value] = d.get(value, 0) + 1
        return super().sym(cat, key, value)


def fit(training: list[tuple[dict, dict]], run_ids: list[str]) -> dict:
    """training: [(phi, h)] pairs -- all phi levels of the training runs may be included (they exercise different symbols)."""
    cc = CountingCoder()
    for phi, h in training:
        grammar_a2(cc, phi, h)
    flags = {k: [v[0] + 1, v[1] + 1] for k, v in cc.flag_counts.items()}
    hist = {}
    for key, K in HIST_KEYS.items():
        counts = [1] * (K + 1)  # + escape
        for v, n in cc.sym_counts.get(key, {}).items():
            counts[min(v, K)] += n
        cum = [0]
        for n in counts:
            cum.append(cum[-1] + n)
        hist[key] = {"cum": cum, "total": cum[-1], "K": K}
    return {"schema": "aot-codec/a3-model/v1", "fitted_on": sorted(set(run_ids)), "flags": flags, "hist": hist,
            "raw": {"flags": cc.flag_counts, "syms": {k: {str(a): b for a, b in v.items()} for k, v in cc.sym_counts.items()}}}


def freeze(model: dict, path: Path) -> str:
    s = json.dumps(model, sort_keys=True, indent=1)
    Path(path).write_text(s)
    return hashlib.sha256(s.encode()).hexdigest()


def load(path: Path) -> tuple[dict, str]:
    s = Path(path).read_text()
    return json.loads(s), hashlib.sha256(s.encode()).hexdigest()
