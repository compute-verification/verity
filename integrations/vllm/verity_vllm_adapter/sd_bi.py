"""Batch invariance of the speculative configuration (lane SD): the same request alone vs co-batched.

Two commit stores of the SD capture (`speculative.py replay --store-values`, the same production tape) are
compared for one request, step by step: the acceptance record (k, candidates, verdicts, accepted, committed
tokens) must agree, and every committed leaf with the same canonical id is compared *bitwise* — the target's
logits at all k+1 verification slots (rejected slots included), the draft's logits at its k proposal rows,
and every layer interface of both models.  A difference is a finding with its localised id, never a relaxation.

    python -m verity_vllm_adapter.sd_bi --a <store alone> --b <store batched> --request R --out bi.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_T_RE = re.compile(r"/t=(\d+)/")
_P_RE = re.compile(r"^p=[^/]+/")


def _key(cid: str) -> str:
    """The identity without the profile id: the alone run and the co-batched run are different workloads (different
    `p=`), the same request's values are matched on everything after it."""
    return _P_RE.sub("", cid)


def _load(root: Path, request: str) -> dict[int, tuple[dict, np.ndarray]]:
    out = {}
    for d in sorted(p for p in (root / "ru" / request).iterdir() if p.is_dir()):
        idx = json.loads((d / "index.json").read_text())
        out[int(d.name)] = (idx, np.memmap(d / "values.bin", dtype=np.uint8, mode="r"))
    return out


def _words(values, e):
    raw = values[e["offset"]: e["offset"] + e["nbytes"]]
    dt = {"bfloat16": np.uint16, "int32": np.uint32, "float32": np.uint32}[e["dtype"]]
    return np.frombuffer(raw.tobytes(), dtype=dt)


def _class(cid: str, sp: dict) -> str:
    t = int(_T_RE.search(cid).group(1))
    draft = "op=draft." in cid
    dead = set(sp.get("dead") or [])
    slots = sp.get("slots") or []
    if "l=lm_head" in cid or "l=sampler" in cid:
        if draft:
            return "draft.lm_head/sampler @proposal" if t in set(sp.get("draft", {}).get("proposal_positions") or []) else "draft.lm_head/sampler @catch-up"
        if t in slots:
            return "target.lm_head/sampler @rejected slot" if t in dead else "target.lm_head/sampler @accepted slot"
        return "target.lm_head/sampler @other"
    side = "draft" if draft else "target"
    where = "@rejected slot" if (t in dead and not draft) else ("@proposal" if (draft and t in set(sp.get("draft", {}).get("proposal_positions") or [])) else "@committed")
    return f"{side}.layers {where}"


def compare(a: Path, b: Path, request: str) -> dict:
    A, B = _load(a, request), _load(b, request)
    ba = json.loads((a / "ru" / request / "boundary.json").read_text())
    bb = json.loads((b / "ru" / request / "boundary.json").read_text())
    rec: dict = {"request": request, "a": str(a), "b": str(b), "profile_a": next(iter(A.values()))[0]["entries"][0]["canonical_id"].split("/")[0],
                 "profile_b": next(iter(B.values()))[0]["entries"][0]["canonical_id"].split("/")[0],
                 "tape_equal": ba["prompt_token_ids"] == bb["prompt_token_ids"] and ba["output_token_ids"] == bb["output_token_ids"],
                 "steps_a": len(A), "steps_b": len(B), "steps": [], "classes": {}, "first_differences": []}
    agg: dict[str, dict] = defaultdict(lambda: {"leaves": 0, "leaves_equal": 0, "leaves_different": 0, "words": 0, "words_different": 0, "missing_in_b": 0})
    for step, (ia, va) in A.items():
        sa = ia.get("speculative") or {}
        if step not in B:
            rec["steps"].append({"step": step, "status": "missing in b"})
            continue
        ib, vb = B[step]
        sb = ib.get("speculative") or {}
        keys = ("k", "accepted", "candidates", "verdicts", "committed_tokens", "slots", "dead")
        acc_equal = all(sa.get(k) == sb.get(k) for k in keys)
        srec = {"step": step, "positions_a": ia.get("positions"), "positions_b": ib.get("positions"), "acceptance_equal": acc_equal,
                "k": sa.get("k"), "accepted": sa.get("accepted"), "leaves": 0, "leaves_different": 0}
        if not acc_equal:
            srec["acceptance_a"] = {k: sa.get(k) for k in keys}
            srec["acceptance_b"] = {k: sb.get(k) for k in keys}
        bidx = {_key(e["canonical_id"]): e for e in ib["entries"]}
        for e in ia["entries"]:
            cid = e["canonical_id"]
            cls = _class(cid, sa)
            g = agg[cls]
            g["leaves"] += 1
            srec["leaves"] += 1
            eb = bidx.get(_key(cid))
            if eb is None:
                g["missing_in_b"] += 1
                continue
            wa, wb = _words(va, e), _words(vb, eb)
            g["words"] += int(wa.size)
            if wa.size != wb.size:
                nd = max(wa.size, wb.size)
            else:
                nd = int((wa != wb).sum())
            if nd:
                g["leaves_different"] += 1
                g["words_different"] += nd
                srec["leaves_different"] += 1
                if len(rec["first_differences"]) < 40:
                    i = int(np.argmax(wa != wb)) if wa.size == wb.size else -1
                    rec["first_differences"].append({"canonical_id": cid, "class": cls, "words": int(wa.size), "different": nd, "first_word": i,
                                                     "a": int(wa[i]) if i >= 0 else None, "b": int(wb[i]) if i >= 0 else None})
            else:
                g["leaves_equal"] += 1
        rec["steps"].append(srec)
    rec["classes"] = dict(sorted(agg.items()))
    tot = {"leaves": sum(g["leaves"] for g in agg.values()), "leaves_different": sum(g["leaves_different"] for g in agg.values()),
           "words": sum(g["words"] for g in agg.values()), "words_different": sum(g["words_different"] for g in agg.values()),
           "missing_in_b": sum(g["missing_in_b"] for g in agg.values()),
           "acceptance_equal_all_steps": all(s.get("acceptance_equal", False) for s in rec["steps"])}
    rec["total"] = tot
    rec["result"] = "batch-invariant (bitwise)" if tot["leaves_different"] == 0 and tot["missing_in_b"] == 0 and tot["acceptance_equal_all_steps"] and rec["tape_equal"] \
        else "DIFFERENCE"
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rec = compare(args.a, args.b, args.request)
    args.out.write_text(json.dumps(rec, indent=1))
    t = rec["total"]
    print(f"[sd-bi] {args.request}: {rec['result']} — leaves {t['leaves']} ({t['leaves_different']} different, {t['missing_in_b']} missing), "
          f"words {t['words']:,} ({t['words_different']:,} different), tape_equal={rec['tape_equal']}, acceptance_equal={t['acceptance_equal_all_steps']}")
    for cls, g in rec["classes"].items():
        print(f"   {cls:42s} leaves={g['leaves']:4d} equal={g['leaves_equal']:4d} diff={g['leaves_different']:3d} words={g['words']:,} diff_words={g['words_different']:,}")
    return 0 if rec["result"].startswith("batch-invariant") else 1


if __name__ == "__main__":
    raise SystemExit(main())
