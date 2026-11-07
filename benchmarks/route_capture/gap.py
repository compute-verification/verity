"""The gap between a route capture's served steps and class G: every (route, statics) pair the captured steps need that class G
doesn't cover, with its proof blocks a served token, whether a Definition exists at those or other statics, and its owner.

    python gap.py CAPTURE_DIR --check CIRCUIT_CHECK.json [--rank R] [--json OUT.json] [--md OUT.md]

CIRCUIT_CHECK.json is `circuit-check`'s report over class G's Definitions (its `targets`: each id and its failures): an id
there is registered, and it passes when it has no failures.

Each step is evaluated at one tensor-parallel rank as the class G decoder writes its launch
(`verity_vllm.program.predict.step_needs`): every Definition it calls with its calls, and each refusal counted in place of
what it refuses. A Definition call is a proof block here. The step's engine facts class G doesn't run (graphs, a padded
launch, a prefix hit, its limits) are counted by tokens beside them.
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
from verity_vllm.program.predict.step_needs import Need, StepNeeds, phi_of_row

#: what closes each refusal: a Definition (circuits), a kernel capture or a deterministic kernel (kernels), serving (vllm)
OWNERS = {"no capture": "kernels", "unregistered": "circuits", "fails circuit-check": "circuits", "no kind": "circuits",
          "radix top-k": "circuits + kernels", "dense-MHA prefill": "circuits + vllm", "sparse-MQA prefill": "kernels + vllm",
          "unseeded sampling": "vllm", "top-k sampling": "circuits", "min-p sampling": "circuits", "penalties": "circuits",
          "FlashInfer sampler": "vllm"}
#: class G's engine limits (`CLASS_G_LIMITS`), with the prefix cache off and eager execution
CLASS_G = {"max_num_seqs": 8, "max_num_batched_tokens": 2048, "max_model_len": 2048, "long_prefill_token_threshold": 256}


def decodes(row: dict[str, Any]) -> list[int]:
    """Each decoding request's key count (a capture without them: every decode at the step's longest)."""
    ctx = row.get("decode_ctx")
    return [int(c) for c in ctx] if ctx is not None else [int(row["decode_ctx_max"])] * int(row["decode"])


def needs(S: StepNeeds, row: dict[str, Any], flashinfer_sampler: bool) -> Counter[Need]:
    """A captured step's needs at its rank (`StepNeeds.step`), its launch the graph dispatch's padded tokens."""
    R = int(row["tokens"])
    return S.step(R, int((row.get("cg") or {}).get("num_tokens") or R), [(q, done) for q, done, _ in row["prefill"]], decodes(row),
                  int(row["reqs"]), row.get("sampling") or {}, flashinfer_sampler)


def engine_facts(row: dict[str, Any]) -> list[str]:
    """What of a captured step's engine class G doesn't run: graphs, a padded launch, a prefix hit, past its limits."""
    R, cg = int(row["tokens"]), row.get("cg") or {}
    seqs = [done + q for q, done, _ in row["prefill"]] + decodes(row)
    facts = [("CUDA graphs (eager in class G)", cg.get("cg_mode") not in (None, "NONE")),
             ("launch padded past its rows", int(cg.get("num_tokens") or R) != R),
             ("prefix-cache hit (off in class G)", bool(row.get("prefix_hit_tokens"))),
             ("more sequences than class G's max_num_seqs", int(row["reqs"]) > CLASS_G["max_num_seqs"]),
             ("more tokens than class G's max_num_batched_tokens", R > CLASS_G["max_num_batched_tokens"]),
             ("a sequence past class G's max_model_len", max(seqs, default=0) > CLASS_G["max_model_len"]),
             ("a prefill chunk past class G's long-prefill threshold",
              any(q > CLASS_G["long_prefill_token_threshold"] for q, _, _ in row["prefill"]))]
    return [f for f, on in facts if on]


def statuses(check: dict[str, Any]) -> dict[str, bool]:
    """circuit-check's report -> each registered id and whether it passes."""
    return {t["target"]: not t.get("failures") for t in check["targets"]}


def flashinfer_sampler_on(capture: Path) -> bool:
    """Whether the served engine ran FlashInfer's sampler (vLLM's default unless run.sh's fallback set the switch to 0)."""
    env = capture.parent / "engine_env.txt"
    return not re.search(r"VLLM_USE_FLASHINFER_SAMPLER=0\b", env.read_text() if env.is_file() else "")


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


def gap(capture: Path, check: dict[str, Any], rank: int | None = None, phi=None) -> dict[str, Any]:
    """The capture's needs at one recording rank (the lowest unless named) against class G's registered, passing Definitions."""
    rank = min(ranks(capture)) if rank is None else rank
    S = StepNeeds(phi if phi is not None else phi_of_row())
    reg = statuses(check)
    fi = flashinfer_sampler_on(capture)
    need: Counter[Need] = Counter()
    facts: Counter[str] = Counter()
    tokens = n_steps = 0
    for row in steps(capture, rank):
        need.update(needs(S, row, fi))
        facts.update({f: int(row["tokens"]) for f in engine_facts(row)})
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
            "flashinfer_sampler": fi, "class_g_registered": len(reg), "class_g_passing": sum(reg.values()),
            "gaps": [r for r in out if r["status"] != "covered"], "covered": [r for r in out if r["status"] == "covered"],
            "engine": {k: {"tokens": v, "share": v / max(tokens, 1)} for k, v in facts.most_common()}}


def markdown(g: dict[str, Any]) -> str:
    lines = [f"Rank {g['rank']}: {g['steps']} steps, {g['tokens']} tokens, {g['blocks_per_token']:.4g} proof blocks a token.", "",
             "| route | Definition or refusal | statics | Definition exists | owner | blocks a token | share of blocks |",
             "|---|---|---|---|---|---|---|"]
    for r in g["gaps"]:
        lines.append(f"| {r['route']} | {r['definition']} ({r['status']}) | {r['statics']} | {r['exists']} | "
                     f"{r['owner']} | {r['blocks_per_token']:.4g} | {r['share']:.3%} |")
    lines += ["", "| engine fact | share of tokens |", "|---|---|"]
    lines += [f"| {k} | {v['share']:.1%} |" for k, v in g["engine"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--check", type=Path, required=True, help="circuit-check's JSON report over class G's Definitions")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    g = gap(a.capture, json.loads(a.check.read_text()), a.rank)
    if a.json:
        a.json.write_text(json.dumps(g, indent=1) + "\n")
    md = markdown(g)
    if a.md:
        a.md.write_text(md)
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
