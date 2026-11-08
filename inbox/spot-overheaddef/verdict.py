"""Captured-step verdicts from a capture directory: each step's check on every rank, and the Python-launched and FA3 events."""
import collections
import json
import sys
from pathlib import Path

root, steps = Path(sys.argv[1]), [int(s) for s in sys.argv[2].split(",")]
WATCH = ("fused_norm_rope", "fused_q", "ll_bf16_gemm", "_vllm_fa3_C::fwd")


def shas(v):
    if isinstance(v, dict):
        if isinstance(v.get("sha512"), str):
            yield v["sha512"]
        for x in v.values():
            yield from shas(x)
    elif isinstance(v, list):
        for x in v:
            yield from shas(x)


for s in steps:
    roots, rank_roots = set(), {}
    for r in range(8):
        d = root / f"rank{r}" / f"step{s}"
        if not (d / "meta.json").exists():
            print(f"step {s} rank {r}: no meta.json")
            continue
        meta = json.loads((d / "meta.json").read_text())
        rank_roots[r] = meta["rank_root"]
        commit = json.loads((d / "commit.json").read_text()) if (d / "commit.json").exists() else None
        if commit:
            roots.add(commit["root"])
        events = [json.loads(x) for x in (d / "ops.jsonl").read_text().splitlines()]
        count = collections.Counter(e["op"] for e in events if e["op"] in WATCH)
        norm_out = sorted({k for e in events if e["op"] == "fused_norm_rope" for k in e["out"]})
        fa3 = [e for e in events if e["op"] == "_vllm_fa3_C::fwd"]
        fa3_none = sum(1 for e in fa3 if e["out"].get("ret2") is None and e["out"].get("ret3") is None)
        consumers = collections.Counter()
        gemms_a_layer = collections.Counter()
        for e in events:
            if e["op"] != "fused_norm_rope" or not isinstance(e["out"].get("r0"), dict):
                continue
            q_c = e["out"]["r0"].get("sha512")
            n = 0
            for f in events[e["seq"] + 1:]:
                if f["layer"] == e["layer"] and q_c in set(shas(f["in"])):
                    consumers[f["op"]] += 1
                    n += f["op"].endswith("gemm")
            gemms_a_layer[n] += 1
        c = meta["check"] or {}
        print(f"step {s} rank {r}: tokens {c.get('tokens')} mode {c.get('mode')} recorded_forward {c.get('recorded_forward')} "
              f"equal_tokens {c.get('equal_tokens')} equal_padded {c.get('equal_padded')} differ_words {c.get('differ_words')} "
              f"errors {len(meta['errors'])} events {meta['events']} commit {'yes' if commit else 'NO'} | {dict(count)} | "
              f"fused_norm_rope out {norm_out} | fa3 fwd {len(fa3)}, ret2/ret3 None in {fa3_none} | r0 read by {dict(consumers)} | "
              f"layers by GEMMs reading r0 {dict(sorted(gemms_a_layer.items()))}")
        for err in meta["errors"][:3]:
            print(f"  error: {str(err)[:300]}")
    print(f"step {s}: {len(roots)} distinct step root(s) over the ranks with a commitment; rank roots "
          + " ".join(f"{r}:{h[:16]}" for r, h in sorted(rank_roots.items())))
