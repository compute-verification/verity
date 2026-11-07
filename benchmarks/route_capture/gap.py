"""The gap between a route capture's served steps and class G: every (route, statics) pair the captured steps need that class G
doesn't cover, with its proof blocks a served token, whether a Definition exists at those or other statics, and its owner.

    python gap.py CAPTURE_DIR --check CIRCUIT_CHECK.json [--check ...] [--rank R] [--json OUT.json] [--md OUT.md]

Each CIRCUIT_CHECK.json is a `circuit-check` report over Definitions (its `targets`: each id and its failures): an id in one
is registered, and it passes when it has no failures, a later report's verdict replacing an earlier one's.

Each step is evaluated at one tensor-parallel rank as the class G decoder writes its launch
(`verity_vllm.program.predict.step_needs`): every Definition it calls with its calls, and each refusal counted in place of
what it refuses. A Definition call is a proof block here. The step's engine facts class G doesn't run (graphs, a padded
launch, a prefix hit, a row past INDEX_TOPK keys served without the canonical top-k, and each limit past both the class's
registered limit and the limit it serves at) are counted by tokens beside them. vLLM's kernel warm-up, which runs before
serving through the same `execute_model`, is left out and counted (`warmup`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from routes import ranks, steps
from verity_vllm.program.predict.step_needs import Need, StepNeeds, engine_class, workload_phi

#: what closes each refusal: a Definition (circuits), a kernel capture or a deterministic kernel (kernels), serving (vllm)
OWNERS = {"no capture": "kernels", "unregistered": "circuits", "fails circuit-check": "circuits", "no kind": "circuits",
          "kernel top-k order": "vllm", "dense-MHA prefill": "circuits + vllm", "sparse-MQA prefill": "kernels + vllm",
          "unseeded sampling": "vllm", "top-k sampling": "circuits", "min-p sampling": "circuits", "penalties": "circuits",
          "FlashInfer sampler": "vllm"}
#: each engine limit a step can pass: its name in `engine_class` and the fact a step past it is
LIMITS = {"max_num_seqs": "more sequences than", "max_num_batched_tokens": "more tokens than",
          "max_model_len": "a sequence past", "long_prefill_token_threshold": "a prefill chunk past"}
#: class G's workload row (GLM-5.3 at TP8, eager) and its pinned config's case
CLASS_G_ROW, CLASS_G_CASE = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager", "GLM53"
#: the sampling the capture records that `SamplingParams.for_sampler_warmup` turns on together, for every warm-up request
WARMUP_SAMPLING = ("logprobs", "min_p", "penalties", "random", "top_k", "top_p")


def class_g_phi():
    """Class G's φ (`step_needs.workload_phi` of its row)."""
    return workload_phi(CLASS_G_ROW, CLASS_G_CASE)


def warmup(row: dict[str, Any]) -> bool:
    """Whether a step is like vLLM's kernel warm-up (`warmup_kernels`): every request sampled with every feature the warm-up's
    `SamplingParams.for_sampler_warmup` sets. Only the capture's leading such steps are the warm-up."""
    s, n = row.get("sampling") or {}, int(row["reqs"])
    return n > 0 and all(s.get(k, 0) == n for k in WARMUP_SAMPLING)


def decodes(row: dict[str, Any]) -> list[int]:
    """Each decoding request's key count (a capture without them: every decode at the step's longest)."""
    ctx = row.get("decode_ctx")
    return [int(c) for c in ctx] if ctx is not None else [int(row["decode_ctx_max"])] * int(row["decode"])


def needs(S: StepNeeds, row: dict[str, Any], flashinfer_sampler: bool) -> Counter[Need]:
    """A captured step's needs at its rank (`StepNeeds.step`), its launch the graph dispatch's padded tokens."""
    R = int(row["tokens"])
    return S.step(R, int((row.get("cg") or {}).get("num_tokens") or R), [(q, done) for q, done, _ in row["prefill"]], decodes(row),
                  int(row["reqs"]), row.get("sampling") or {}, flashinfer_sampler)


def engine_facts(row: dict[str, Any], index_topk: int = 2048, canonical_topk: bool = False) -> list[str]:
    """What of a captured step's engine class G doesn't run: graphs, a padded launch, a prefix hit, a row past `index_topk` keys
    served without the canonical top-k, and each limit past the class's registered limit and past the one it serves at."""
    R, cg = int(row["tokens"]), row.get("cg") or {}
    seqs = [done + q for q, done, _ in row["prefill"]] + decodes(row)
    size = {"max_num_seqs": int(row["reqs"]), "max_num_batched_tokens": R, "max_model_len": max(seqs, default=0),
            "long_prefill_token_threshold": max((q for q, _, _ in row["prefill"]), default=0)}
    facts = [("CUDA graphs (eager in class G)", cg.get("cg_mode") not in (None, "NONE")),
             ("launch padded past its rows", int(cg.get("num_tokens") or R) != R),
             ("prefix-cache hit (off in class G)", bool(row.get("prefix_hit_tokens"))),
             ("a row past INDEX_TOPK keys served without the canonical top-k", not canonical_topk and size["max_model_len"] > index_topk)]
    G = engine_class("G")
    for k, fact in LIMITS.items():
        lim, srv = G["limits"][k], G["served"][k]
        at = {lim: "registered", srv: "served"} if lim != srv else {lim: "registered and served"}
        facts += [(f"{fact} class G's {k} ({w}: {v:,})", size[k] > v) for v, w in at.items()]
    return [f for f, on in facts if on]


def statuses(checks: list[dict[str, Any]]) -> dict[str, bool]:
    """circuit-check's reports -> each registered id and whether it passes, by the last report that lists it."""
    return {t["target"]: not t.get("failures") for check in checks for t in check["targets"]}


def canonical_topk_on(capture: Path) -> bool:
    """Whether the served engine's environment, as run.sh recorded it, is the one class G serves a row past INDEX_TOPK keys under."""
    env = capture.parent / "engine_env.txt"
    text = env.read_text() if env.is_file() else ""
    return all(re.search(rf"(?<![\w.]){re.escape(k)}={re.escape(v)}(?!\S)", text) for k, v in engine_class("G")["long_rows_env"].items())


def flashinfer_sampler_on(capture: Path) -> bool:
    """Whether the served engine ran FlashInfer's sampler (vLLM's default unless run.sh's fallback set the switch to 0)."""
    env = capture.parent / "engine_env.txt"
    return not re.search(r"VLLM_USE_FLASHINFER_SAMPLER=0\b", env.read_text() if env.is_file() else "")


def engine_had_flashinfer(capture: Path) -> bool | None:
    """`has_flashinfer()` in the served engine, as run.sh recorded it (None: not recorded)."""
    env = capture.parent / "engine_env.txt"
    m = re.search(r"has_flashinfer=(True|False)\b", env.read_text() if env.is_file() else "")
    return None if m is None else m.group(1) == "True"


def _statics(ids: list[str]) -> str:
    """A compact range of the ids' statics: each static that varies as min..max (and how many values), the rest once; an
    all-reduce's TREES left out."""
    vals: dict[str, set] = defaultdict(set)
    for i in ids:
        m = re.search(r"\{(.*)\}$", i)
        body = re.sub(r"TREES=.*?(?=,FTZ=|$)", "", m.group(1)) if m else ""
        for part in filter(None, body.split(",")):
            k, _, v = part.partition("=")
            vals[k].add(v)
    out = []
    for k, vs in vals.items():
        if len(vs) == 1:
            out.append(f"{k}={next(iter(vs))}")
        else:
            nums = sorted(int(v) for v in vs if re.fullmatch(r"-?\d+", v))
            out.append(f"{k}={nums[0]}..{nums[-1]} ({len(vs)} values)" if len(nums) == len(vs) else f"{k}∈{{{','.join(sorted(vs))}}}")
    return ", ".join(out)


def gap(capture: Path, check: dict[str, Any] | list[dict[str, Any]], rank: int | None = None, phi=None) -> dict[str, Any]:
    """The capture's needs at one recording rank (the lowest unless named) against the registered, passing Definitions of
    circuit-check's report or reports."""
    rank = min(ranks(capture)) if rank is None else rank
    fi, canonical = flashinfer_sampler_on(capture), canonical_topk_on(capture)
    phi = phi if phi is not None else class_g_phi()
    if canonical:
        phi = type(phi).from_json({**phi.to_json(), "engine": {**phi.engine, "env": {**(phi.engine.get("env") or {}),
                                                                                   **engine_class("G")["long_rows_env"]}}})
    S = StepNeeds(phi)
    reg = statuses(check if isinstance(check, list) else [check])
    need: Counter[Need] = Counter()
    facts: Counter[str] = Counter()
    tokens = n_steps = 0
    warm: list[dict[str, Any]] = []
    for row in steps(capture, rank):
        if not n_steps and warmup(row):
            warm.append(row)
            continue
        need.update(needs(S, row, fi))
        facts.update({f: int(row["tokens"]) for f in engine_facts(row, S.index_topk, canonical)})
        tokens += int(row["tokens"])
        n_steps += 1
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    blocks = sum(need.values())
    for nd, c in need.items():
        if nd.gap is None:
            status = "covered" if reg.get(nd.id) else ("fails circuit-check" if nd.id in reg else "unregistered")
        else:
            status = nd.gap.split(":")[0]
        key = (nd.route, re.sub(r"\{.*", "", nd.id), status)
        r = rows.setdefault(key, {"route": key[0], "definition": key[1], "status": status, "owner": OWNERS.get(status, ""),
                                  "exists": nd.exists, "why": sorted({nd.gap})[0] if nd.gap else None, "blocks": 0, "ids": []})
        r["blocks"] += c
        r["ids"].append(nd.id)
    out = []
    for r in sorted(rows.values(), key=lambda r: (r["status"] == "covered", -r["blocks"])):
        out.append({**r, "statics": _statics(r["ids"]), "n_statics": len(r["ids"]), "blocks_per_token": r["blocks"] / max(tokens, 1),
                    "share": r["blocks"] / max(blocks, 1), "ids": sorted(r["ids"])[:12]})
    return {"rank": rank, "steps": n_steps, "tokens": tokens, "blocks": blocks, "blocks_per_token": blocks / max(tokens, 1),
            "warmup": {"steps": len(warm), "tokens": sum(int(r["tokens"]) for r in warm)}, "flashinfer_sampler": fi, "engine_had_flashinfer": engine_had_flashinfer(capture),
            "canonical_topk": canonical, "class_g": engine_class("G"),
            "class_g_registered": len(reg), "class_g_passing": sum(reg.values()),
            "gaps": [r for r in out if r["status"] != "covered"], "covered": [r for r in out if r["status"] == "covered"],
            "needs": [{"route": nd.route, "id": nd.id, "gap": nd.gap, "exists": nd.exists, "blocks": c,
                       "registered": nd.id in reg, "passes": bool(reg.get(nd.id))} for nd, c in need.most_common()],
            "engine": {k: {"tokens": v, "share": v / max(tokens, 1)} for k, v in facts.most_common()}}


def markdown(g: dict[str, Any]) -> str:
    lines = [f"Rank {g['rank']}: {g['steps']} steps, {g['tokens']} tokens, {g['blocks_per_token']:.4g} proof blocks a token "
             f"(vLLM's warm-up, {g['warmup']['steps']} steps of {g['warmup']['tokens']} tokens, left out).", "",
             "| route | Definition or refusal | statics | Definition exists | owner | blocks a token | share of blocks |",
             "|---|---|---|---|---|---|---|"]
    for r in g["gaps"]:
        lines.append(f"| {r['route']} | {r['definition']} ({r['status']}) | {r['statics']} | {r['exists']} | "
                     f"{r['owner']} | {r['blocks_per_token']:.4g} | {r['share']:.3%} |")
    lines += ["", "| engine fact | share of tokens |", "|---|---|"]
    lines += [f"| {k} | {v['share']:.1%} |" for k, v in g["engine"].items()]
    if g.get("engine_had_flashinfer") is False:
        lines += ["", "The captured engine had no FlashInfer, which production images ship: its block-FP8 GEMMs below 32 rows "
                      "and its sampler ran other kernels than the rows, which are the decoder's for an engine with FlashInfer."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--check", type=Path, action="append", required=True,
                    help="a circuit-check JSON report over Definitions; repeat it, a later report's verdict replacing an earlier one's")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    g = gap(a.capture, [json.loads(p.read_text()) for p in a.check], a.rank)
    if a.json:
        a.json.write_text(json.dumps(g, indent=1) + "\n")
    md = markdown(g)
    if a.md:
        a.md.write_text(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
