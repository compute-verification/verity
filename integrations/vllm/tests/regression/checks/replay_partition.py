"""(e') [T1, STUB] The sampled-replay population reproduces: `commit/sampled_replay_p<k>.json` -- the population / strata partition,
its `not_executed_by_rule` split, the draw seed and the per-request `facts_of_record` -- recomputed by `verity_capture.commit.sampled_replay`
over the manifest of record + the binding map (`commit/binding_map_p*.json`, 0.5-1.5 GB each, in the vault core tars).  STRONG.

Not implemented tonight; the expected side is lifted (population totals, seed, per-request n_exec / arrive facts) so the contract is frozen.
"""
from __future__ import annotations

from .. import resolver

NAME = "replay_partition"
TIER = "T1"
STRENGTH = "strong"
INPUTS = ("commit/sampled_replay_p0.json", "manifest.json", "commit/binding_map_p0.json")


def applies(row: resolver.Row) -> str | None:
    return None if row.has("commit/sampled_replay_p0.json") else "no sampled_replay record on this row"


def _walk_scalars(d: dict, keys: tuple[str, ...]) -> dict:
    return {k: d.get(k) for k in keys if k in d}


def expected(row: resolver.Row) -> dict:
    sr = row.json("commit/sampled_replay_p0.json")
    body = sr.get("sampled_replay") if isinstance(sr.get("sampled_replay"), dict) else sr
    pop = body.get("population") if isinstance(body.get("population"), dict) else {}
    epr = body.get("executed_prefix_of_record") if isinstance(body.get("executed_prefix_of_record"), dict) else {}
    facts = epr.get("facts_of_record") if isinstance(epr.get("facts_of_record"), dict) else {}
    return {"result": body.get("result", body.get("ok")), "seed": body.get("seed", pop.get("seed")),
            "population": _walk_scalars(pop, ("N", "n", "identities", "strata", "N_excluded", "applied", "manifest_digest", "program_digest")),
            "not_executed_by_rule": _walk_scalars(pop.get("not_executed_by_rule") or {}, ("applied", "N_excluded", "n")),
            "facts_of_record": {rid: _walk_scalars(f, ("a_r", "cap", "arrive", "lag", "linked", "lag_attested", "rows_past_stop", "eos_declared")) for rid, f in facts.items() if isinstance(f, dict)},
            "boundary_linkage_result": (sr.get("boundary_linkage") or {}).get("result") if isinstance(sr.get("boundary_linkage"), dict) else None}


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("commit/sampled_replay_p0.json")
    if cand is not None:
        import json
        from types import SimpleNamespace
        d = json.load(open(cand))
        fake = SimpleNamespace(json=lambda rel: d, has=lambda rel: True, key=row.key)
        return dict(expected(fake), source="candidate")  # type: ignore[arg-type]
    raise NotImplementedError("replay_partition: recomputation over the binding map is T1 (pod) and not implemented yet")
