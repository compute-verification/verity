"""γ at the built 1,536-bit point with the partial-root rebuild added to ra-codec-theory's LP.

`gamma.Point.terms` prices rebuilding a word whose payload is free (E with no assumption on W; every KV word) at a
whole encode, saving the whole word. The partial-root attack (partial_root.py) keeps the top two-thirds of a k = 1
word and completes the bottom third from them and the known payload at `frac` of an encode. Per word the adversary's
options are then nothing, the third at `frac`, or the whole at 1: the LP gets the hull, the third at `frac` saving
s/3 and the rest at 1 − frac saving 2s/3 (taken in that order since frac < 1/3). `frac` is measured on the host CPU
(GMP root against FLINT completion); the device's ratio is not measured.

`python partial_gamma.py theory_model [FRAC ...]`: `theory_model/model/` is ra-codec-theory's `gamma.py` and
`frontier.py` vendored verbatim from 218754c30 (cursor/ra-codec-theory-225f); each row is first checked equal to
`frontier.built_options` before it is re-priced.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace


def split(share: float, cost: float, s: float, frac: float) -> list[tuple[float, float, float]]:
    return [(share, frac * cost, s / 3), (share, (1 - frac) * cost, 2 * s / 3)]


def terms(g, p, frac: float | None) -> dict[str, float]:
    """`p.terms()`'s γ_bw and γ_decode_free, with every one-round free-payload option split when `frac` is set."""
    c = p.codec
    e = g.e_bytes(p.shape, c.payload_bits, c.slot_bits)
    kv = p.batch * p.context * p.shape.kv_bytes_per_token()
    act = p.act_crossing * p.shape.activation_bytes(p.batch)
    cert = e + kv if p.kv_certified else e
    a, b = e / cert, (kv / cert if p.kv_certified else 0.0)
    kc = p.kv_codec or c
    enc_kv = p.kv_rounds * kc.r * kc.eps
    new = p.batch * p.shape.kv_bytes_per_token() if p.kv_certified else 0.0
    y_kv = c.y + p.kv_rounds * kc.eps - c.eps + p.kv_extra if p.kv_certified else 1.0
    d = (c.y * e + y_kv * kv + enc_kv * new + p.idle * e) / cert
    budget = d * p.kappa * (1 + p.theta) - c.tau
    chip = g.mu_chip(cert)
    slot = a * g.mu_slot(c.p_bits, c.slot_bits) + b * g.mu_slot(kc.p_bits, kc.slot_bits)
    ca = p.adversary or c
    ka = p.kv_adversary or (kc if p.kv_codec else ca)
    enc_kv_a = p.kv_rounds * ka.r * ka.eps
    kv_one_round = p.kv_rounds == 1

    def free(share, cost, s, one_round):
        return split(share, cost, s, frac) if frac is not None and one_round else [(share, cost, s)]

    kv_opt = free(b, enc_kv_a, g.s_word(kc, 1.0), kv_one_round)
    rebuilt = min(1.0 - chip, g.rebuild(budget, ca.eps * a,
                                        [(a, (ca.r - 1) * ca.eps, g.s_word(c, p.s_c))] + kv_opt))
    rebuilt_free = min(1.0 - chip, g.rebuild(budget, 0.0, free(a, ca.r * ca.eps, g.s_word(c, 1.0), True) + kv_opt))
    q_honest = e + kv + act

    def gamma(lost: float) -> float:
        return 1 - cert * (1 - min(1.0, lost)) / q_honest

    return {"mu_rebuild": rebuilt, "mu_rebuild_free": rebuilt_free,
            "gamma_bw": gamma(chip + rebuilt + slot + p.other),
            "gamma_decode_free": gamma(chip + rebuilt_free + slot + p.other)}


def built_points(g, fr, batch: int) -> dict[str, object]:
    """frontier.built_options's Points (E on 2^1536 − 3453 as built, each KV option), rebuilt to be re-priced."""
    shape = g.LLAMA_70B
    i = int(batch >= 32)
    off = (fr.FUSED_1536_Y[0] - fr.FUSED_768_Y + g.WIDE["p767"].eps) / fr.sq(1536)
    c = replace(fr.e_codec(1536, batch, off), y=fr.FUSED_1536_Y[i])
    adv = g.at_floor(c, fr.sq(1536, None))
    c = replace(c, y=c.y + fr.SKETCH_1536_Y[i] - fr.FUSED_1536_Y[i])
    common = dict(idle=fr.IDLE[shape.name], theta=g.WIDE_THETA["p767"], out_tokens=fr.OUT_TOKENS, adversary=adv,
                  kappa=fr.KAPPA)
    out = {f"k{kr}": g.Point(shape, batch, 4096, True, c, kv_rounds=kr, kv_extra=fr.kv_attn(kr), **common)
           for kr in (1, 2, 4)}
    for n in (1536, 2048):
        kc = fr.kv_codec(n, batch, off)
        out[f"w{n}"] = g.Point(shape, batch, 4096, True, c, kv_codec=kc, kv_adversary=g.at_floor(kc, fr.sq(n, None)),
                               kv_extra=fr.kv_attn(1), **common)
    return out


def main(argv: list[str]) -> None:
    sys.path.insert(0, argv[0])
    from model import frontier as fr
    from model import gamma as g

    fracs = [float(f) for f in argv[1:]] or [0.097]
    for batch in (1, 32):
        built = fr.built_options(g.LLAMA_70B, batch)
        for name, p in built_points(g, fr, batch).items():
            ref, base = built[name], terms(g, p, None)
            assert abs(ref["gamma_bw"] - base["gamma_bw"]) < 1e-12, (name, ref["gamma_bw"], base["gamma_bw"])
            assert abs(ref["gamma_decode_free"] - base["gamma_decode_free"]) < 1e-12
            row = {"batch": batch, "kv": name, "slowdown": round(ref["slowdown"], 3),
                   "gamma_bw": round(100 * base["gamma_bw"], 3),
                   "gamma_decode_free": round(100 * base["gamma_decode_free"], 3)}
            for f in fracs:
                t = terms(g, p, f)
                row[f"gamma_bw@{f}"] = round(100 * t["gamma_bw"], 3)
                row[f"gamma_decode_free@{f}"] = round(100 * t["gamma_decode_free"], 3)
            print(json.dumps(row))


if __name__ == "__main__":
    main(sys.argv[1:])
