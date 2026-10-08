"""The stratified sample of VUs and its seed: seeded round-robin across strata (`sample`), the seed under a verifier
challenge (`challenge_seed`, derived in `verity_vllm.commit.challenge`), and the honest-prover catch probability."""
from __future__ import annotations

from typing import Any

from verity.ml.randomness import Key

from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.commit import challenge as CH


def sample(vus: list[dict], per_stratum: int, seed: int, *, targets: list[str] | None = None,
           key: Key | None = None) -> tuple[list[dict], dict[str, dict[str, int]], dict]:
    """Stratified sample (`challenge.stratum_picks`: legacy from `seed`, or from `key`), emitted ROUND-ROBIN across strata (every stratum's
    first pick before any stratum's second) so that a caller's cap keeps >= 1 VU per stratum.  `targets`: `<op_path>[@<step>]` -- every VU
    of that module (at that request-local step, if given) is added to the sample and recorded as targeted (the negatives of record)."""
    by: dict[str, list[dict]] = {}
    for v in vus:
        by.setdefault(v["stratum"], []).append(v)
    rounds: list[list[dict]] = []
    strata = {}
    names = sorted(by)
    picks = CH.stratum_picks([by[k] for k in names], per_stratum, None if key is not None else seed, key)
    for k, chosen in zip(names, picks):
        for j, c in enumerate(chosen):
            while len(rounds) <= j:
                rounds.append([])
            rounds[j].append(c)
        strata[k] = {"population": len(by[k]), "sampled": len(chosen)}
    picked = [c for rnd in rounds for c in rnd]
    n_strat = len(picked)
    targeted: dict[str, int] = {}
    if targets:
        seen = {id(v) for v in picked}
        for t in targets:
            op, _, stp = str(t).partition("@")
            hit = [v for v in vus if (v["op_path"] == op or v["op_path"].endswith("." + op) or v["op_path"].endswith(op)) and (not stp or int(v["step"]) == int(stp))]
            targeted[str(t)] = len(hit)
            for v in hit:
                if id(v) not in seen:
                    seen.add(id(v))
                    picked.append(v)
                    strata[v["stratum"]]["sampled"] = strata[v["stratum"]].get("sampled", 0) + 1
    return picked, strata, {"hits": targeted, "added": len(picked) - n_strat, "stratified": n_strat}


def challenge_seed(challenge: str, run_root: bytes) -> tuple[int, str]:
    """[rev-cb M-0572, v2] the sample seed under a VERIFIER-ISSUED post-commitment challenge: sha256(challenge || run_root)[:8].  Binding the
    challenge to the root means a challenge issued before the root was fixed (or for another run) selects a different sample, and the
    prover cannot grind the sample by recommitting (the root changes -> the verifier issues a fresh challenge).  Returns (seed, source).
    Legacy (`challenge.LEGACY`); its successor is `challenge.replay_key`, an auditor-keyed draw passed as `challenge_key`."""
    return CH.legacy_replay_seed(challenge, run_root), RC.challenge_seed_source(challenge)


def _catch_probability(strata: dict[str, dict]) -> dict[str, Any]:
    """[rev-cb M-0572] the HONEST-prover claim of a stratified sample, by stratum kind: the probability that ONE faulted VU is among the
    evaluated ones = evaluated / population of its stratum (a fault in every VU of a stratum -- a relabel, a wrong module, a late read of a
    whole tensor -- is caught with probability 1 as soon as the stratum has one evaluation).  Reported per phase (min / median / mean over
    that phase's strata) and overall; the seed's freshness is `sample.seed_source` -- against an ADAPTIVE prover only a post-commitment
    verifier challenge makes these numbers the catch rate (v2)."""
    import statistics
    by_phase: dict[str, list[float]] = {}
    for k, v in strata.items():
        pop, ev = int(v.get("population", 0)), int(v.get("evaluated", 0))
        if pop <= 0:
            continue
        parts = k.split("|")
        phase = parts[3] if len(parts) > 3 else "?"
        by_phase.setdefault(phase, []).append(min(1.0, ev / pop))
    def summ(ps: list[float]) -> dict[str, float]:
        return {"min": round(min(ps), 4), "median": round(statistics.median(ps), 4), "mean": round(sum(ps) / len(ps), 4),
                "strata": len(ps), "strata_fully_evaluated": sum(1 for p in ps if p >= 1.0)}
    allp = [p for ps in by_phase.values() for p in ps]
    return {"single_vu_fault": {"overall": summ(allp) if allp else None, "by_phase": {ph: summ(ps) for ph, ps in sorted(by_phase.items())}},
            "whole_stratum_fault": "1.0 for every stratum with evaluated >= 1 (see strata_unevaluated for the exceptions)",
            "note": "honest-prover measurement: probability that one faulted VU is evaluated = evaluated/population of its stratum; "
                    "adversarially meaningful only under a post-commitment challenge seed (sample.seed_source)"}
