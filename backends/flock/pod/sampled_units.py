"""The sampled units of a passing deployment, as the Call Definitions class_statement proves (`--definitions`).

    sampled_units.py <deployment evidence dir> <out dir>

The Commit's sampled replay (`commit/sampled_replay_p0.json`) records its draw's seed, k and the Programs it read, not the units
it picked. They are drawn again here exactly as `driver.sampled_replay` draws them (the Build's Program, `population`, the
not-executed split, `family_strata` and `sample` under the record's seed), and the draw must reproduce the record: the picks per
(request, request-relative step) and per family. Run it on the deployment's own executing tree (its `tree.json`), whose draw it
is. A deployment whose draw doesn't reproduce is refused by name, never proved.
Writes `<out>/defs.json` ({Call spec: picked Calls}) and `<out>/picks.json` (each pick's request, step, op_path, spec, family).
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from collections import Counter
from pathlib import Path

from verity_vllm.check.replay import driver as SR
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.population import phase_anchor, population, split_not_executed
from verity_vllm.check.replay.sample import sample
from verity_vllm.query.manifest import format as MF


def draw(ev: Path) -> tuple[list[dict], dict]:
    doc = json.loads((ev / "commit" / "sampled_replay_p0.json").read_text())
    sr = doc["sampled_replay"]
    smp = sr["sample"]
    if smp.get("seed_form") != "run-root" or smp.get("targeted_n"):
        raise SystemExit(f"refused: seed form {smp.get('seed_form')!r}, {smp.get('targeted_n')} targeted (only an untargeted run-root draw is drawn again here)")
    progs_rec = doc.get("programs") or sr.get("programs") or {}
    if not progs_rec:
        raise SystemExit("refused: the record names no Programs")
    manifest = json.loads((ev / "manifest.json").read_text()) if (ev / "manifest.json").exists() else None
    epr = doc.get("executed_prefix_of_record") or sr.get("executed_prefix_of_record") or {}
    facts = epr.get("facts_of_record") or {}
    eos = any(bool((v or {}).get("eos_stop")) for k, v in epr.items() if k.startswith("r") and isinstance(v, dict))
    k = int(smp["k"])
    executed = {} if (k and not eos) else {r: SR.executed_prefix_of_record(r, **f) for r, f in facts.items()}
    arrivals = {r: int(a) for r, a in (smp.get("picks_by_request_arrival") or {}).items()}
    ranks = sr.get("strata_keys", {}).get("ranks") or [0]
    rank = int((re.search(r"\d+", str(ranks[0])) or [0])[0]) if len(ranks) == 1 else 0
    all_vus, indices = [], {}
    for rid, p in progs_rec.items():
        loaded = MF.load_program(p["dir"])
        dig = str((loaded.get("result") or {}).get("program_digest") or (loaded.get("instances") or {}).get("digest") or "")
        if dig != p["program_digest"]:
            raise SystemExit(f"refused: {rid}'s Program at {p['dir']} is {dig[:16]}, the record's is {p['program_digest'][:16]}")
        pi = indices.get(dig) or ProgramIndex(loaded["instances"], program_dir=p["dir"], query=SR.manifest_query(manifest))
        indices[dig] = pi
        anchor = phase_anchor(pi.n_steps, executed.get(str(rid)))
        vus, _ne, _nef = population(pi, rid, rank=rank, arrive=arrivals.get(rid, 0), addresses={}, last_step=int(anchor["last_step"]), codes={})
        all_vus.extend(vus)
    pool, _not_exec = split_not_executed(all_vus, executed)
    kw = {"draw": smp["strata_by"]} if "draw" in inspect.signature(SR.family_strata).parameters else {}
    if smp.get("strata_by") not in ("family", "unit") and not kw:
        raise SystemExit(f"refused: the record's draw is {smp.get('strata_by')!r}, which this tree's family_strata can't draw")
    picked_all, strata, tgt = sample(*SR.family_strata(pool, k, **kw), int(smp["seed"]))
    cap = max(int(k or smp["max_evaluate"]), len(strata))
    picked = picked_all[:min(cap, int(tgt["stratified"]))]
    got_steps: dict = {}
    for v in picked:
        c = got_steps.setdefault(v["request_id"], Counter())
        c[str(v["step"] - arrivals.get(v["request_id"], 0))] += 1
    want_steps = {r: {s: int(x[0]) for s, x in per.items()} for r, per in smp["picks_by_request_step"].items()}
    got = {r: dict(c) for r, c in got_steps.items()}
    fam = dict(Counter(v["family"] for v in picked))
    check = {"picked": len(picked), "record_picked": int(smp["picked"]), "steps_equal": got == want_steps,
             "families_equal": fam == sr["timing"]["per_family_n"], "families": fam}
    if not (check["steps_equal"] and check["families_equal"] and len(picked) == int(smp["picked"])):
        raise SystemExit(f"refused: the draw does not reproduce the record: {json.dumps(check)[:600]}")
    return picked, check


def main() -> int:
    ev, out = Path(sys.argv[1]), Path(sys.argv[2])
    picked, check = draw(ev)
    out.mkdir(parents=True, exist_ok=True)
    defs = Counter(v["spec"] for v in picked)
    (out / "defs.json").write_text(json.dumps(dict(sorted(defs.items())), indent=1))
    (out / "picks.json").write_text(json.dumps({"check": check, "picks": [{x: v[x] for x in ("request_id", "step", "op_path", "spec", "family", "i")}
                                                                       for v in picked]}, indent=1))
    print(f"{len(picked)} units reproduce the record; {len(defs)} Call Definitions: {dict(Counter(v['family'] for v in picked))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
