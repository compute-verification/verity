"""A served decode step's kernels from an sbench `prof` capture (nsys SQLite export): per arm, each kernel's launches and GPU
time a step, the GPU's busy time a step and the step's wall time, as the difference of the arm's two NVTX ranges
(PROF_RANGE/<arm>/n<N1> and n<N2>, the same 32 requests generating N1 and then N2 tokens) over N2 - N1, so prefill and each
generate call's own overhead cancel.  Only kernels of the process that pushed the range, starting and ending inside it,
count.  Each kernel gets a category from its name (CATEGORIES, first match), so two arms' steps compare by category.

    python steps.py gate.sqlite > steps.json
    python steps.py --compare stock.json exact.json [more.json ...]    # categories and kernels side by side, the first the base
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys

PROF_RANGE = "verity/exactlin/prof/"
CATEGORIES = [
    ("fused quant (exactlin)", re.compile(r"exactlin_quant_rows")),
    ("fused GEMM (exactlin)", re.compile(r"exactlin_scaled_mm")),
    ("stock quant (scaled_int8_quant)", re.compile(r"vllm::dynamic_scaled_int8_quant")),
    ("stock GEMM (triton_scaled_mm)", re.compile(r"^scaled_mm_kernel")),
    ("Inductor/Triton fused (norms, rope, act)", re.compile(r"^triton_(poi|red|per)_")),
    ("attention", re.compile(r"flash|attn|attention|paged|reshape_and_cache", re.IGNORECASE)),
    ("int8 GEMM (torch._int_mm, cuBLASLt)", re.compile(r"(gemm|xmma|nvjet|cutlass|cublas).*(i8|s8|int8|imma)|"
                                                      r"(i8|s8|int8|imma).*(gemm|xmma|nvjet)", re.IGNORECASE)),
    ("other GEMM (lm_head)", re.compile(r"gemm|xmma|nvjet|cutlass|cublas|splitK", re.IGNORECASE)),
    ("torch cat/copy (padding, casts)", re.compile(r"CatArrayBatchedCopy|direct_copy_kernel|copy_kernel")),
    ("torch reduce (amax)", re.compile(r"at::native::.*reduce_kernel")),
    ("torch elementwise", re.compile(r"at::native::")),
]


def category(name: str) -> str:
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def _pid(global_id: int) -> int:
    return (int(global_id) >> 24) & 0xFFFFFF


def _busy(iv: list[tuple[int, int]]) -> int:
    tot, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def short(name: str) -> str:
    n = re.sub(r"\s+", " ", name)
    return n if len(n) <= 140 else n[:137] + "..."


def analyse(path: str) -> dict:
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges: dict[str, dict[int, tuple[int, int, int]]] = {}
    for start, end, text, tid, text_id in db.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text if text is not None else strings.get(text_id)
        if name and name.startswith(PROF_RANGE):
            arm, n = name.removeprefix(PROF_RANGE).rsplit("/n", 1)
            ranges.setdefault(arm, {})[int(n)] = (int(start), int(end), _pid(tid))
    kern = list(db.execute("SELECT demangledName, globalPid, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"))
    try:
        api = list(db.execute("SELECT nameId, globalTid, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    except sqlite3.Error:
        api = []
    doc: dict = {"arms": {}}
    for arm, by_n in ranges.items():
        if len(by_n) != 2:
            doc["arms"][arm] = {"error": f"ranges {sorted(by_n)}"}
            continue
        (n1, r1), (n2, r2) = sorted(by_n.items())
        per: dict[int, dict] = {}
        for n, (start, end, pid) in ((n1, r1), (n2, r2)):
            names: dict[str, list[int]] = {}
            iv = []
            for name_id, gpid, s, e in kern:
                if _pid(gpid) == pid and start <= s and e <= end:
                    k = names.setdefault(strings.get(name_id, str(name_id)), [0, 0])
                    k[0] += 1
                    k[1] += int(e) - int(s)
                    iv.append((int(s), int(e)))
            calls: dict[str, int] = {}
            for name_id, tid, s, e in api:
                if _pid(tid) == pid and start <= s and e <= end:
                    nm = strings.get(name_id, str(name_id))
                    calls[nm] = calls.get(nm, 0) + 1
            per[n] = {"names": names, "busy": _busy(iv), "wall": end - start, "api": calls}
        steps = n2 - n1
        rows = []
        for name in set(per[n1]["names"]) | set(per[n2]["names"]):
            c1, t1 = per[n1]["names"].get(name, [0, 0])
            c2, t2 = per[n2]["names"].get(name, [0, 0])
            dc, dt = (c2 - c1) / steps, (t2 - t1) / steps
            if dc or dt:
                rows.append({"name": short(name), "category": category(name), "launches_per_step": round(dc, 3),
                             "us_per_step": round(dt / 1e3, 2), "us_per_launch": round(dt / 1e3 / dc, 3) if dc else None})
        rows.sort(key=lambda r: -r["us_per_step"])
        cats: dict[str, dict] = {}
        for r in rows:
            c = cats.setdefault(r["category"], {"launches_per_step": 0.0, "us_per_step": 0.0})
            c["launches_per_step"] = round(c["launches_per_step"] + r["launches_per_step"], 3)
            c["us_per_step"] = round(c["us_per_step"] + r["us_per_step"], 2)
        api_rows = {nm: round((per[n2]["api"].get(nm, 0) - per[n1]["api"].get(nm, 0)) / steps, 3)
                    for nm in set(per[n1]["api"]) | set(per[n2]["api"])}
        doc["arms"][arm] = {
            "n": [n1, n2], "decode_steps": steps,
            "wall_ms_per_step": round((per[n2]["wall"] - per[n1]["wall"]) / steps / 1e6, 4),
            "gpu_busy_ms_per_step": round((per[n2]["busy"] - per[n1]["busy"]) / steps / 1e6, 4),
            "kernel_ms_per_step": round(sum(r["us_per_step"] for r in rows) / 1e3, 4),
            "launches_per_step": round(sum(r["launches_per_step"] for r in rows), 3),
            "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1]["us_per_step"])),
            "api_per_step": dict(sorted(((k, v) for k, v in api_rows.items() if v), key=lambda kv: -kv[1])[:12]),
            "kernels": rows}
    return doc


def compare(paths: list[str]) -> dict:
    """Categories are recomputed from the kernels' names, so a capture read under an older CATEGORIES compares too."""
    arms = {}
    for p in paths:
        with open(p) as f:
            arms.update(json.load(f)["arms"])
    for a in arms.values():
        a["categories"] = {}
        for r in a["kernels"]:
            r["category"] = category(r["name"])
            c = a["categories"].setdefault(r["category"], {"launches_per_step": 0.0, "us_per_step": 0.0})
            c["launches_per_step"] = round(c["launches_per_step"] + r["launches_per_step"], 3)
            c["us_per_step"] = round(c["us_per_step"] + r["us_per_step"], 2)
    names = list(arms)
    base = names[0]
    cats = sorted({c for a in arms.values() for c in a.get("categories", {})})
    table = {c: {a: arms[a]["categories"].get(c, {"launches_per_step": 0, "us_per_step": 0}) for a in names} for c in cats}
    head = {a: {k: arms[a][k] for k in ("wall_ms_per_step", "gpu_busy_ms_per_step", "kernel_ms_per_step", "launches_per_step")}
            for a in names}
    delta = {a: {k: round(head[a][k] - head[base][k], 4) for k in head[a]} for a in names[1:]}
    only = {a: [r for r in arms[a]["kernels"] if r["name"] not in {q["name"] for q in arms[base]["kernels"]}][:20]
            for a in names[1:]}
    return {"base": base, "per_step": head, "delta_vs_base": delta, "categories": table, "kernels_not_in_base": only}


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        print(json.dumps(compare(sys.argv[2:]), indent=1))
    else:
        print(json.dumps(analyse(sys.argv[1]), indent=1))
