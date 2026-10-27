"""The cost of `int8_fused.scaled_mm_zy`'s record epilogue (`verity_int8::scaled_mm_zy`, the per-unit z/y records)
at kernel level, after the vLLM integration's tests of it and a gate in the same run (a failed gate times nothing, exit 4):

    tests   pytest on `tests/engine/test_int8_zy.py` and `test_int8_ops.py` (their GPU tests run here), junit in OUT
    gate    per shape, into 0xA5-poisoned y and records at the picked config: y bit for bit `scaled_mm`'s, the records the
            op's torch body's (`int8_exact.scaled_mm_zy`: ``torch._int_mm`` and `dequantize`), the kernel's PTX
            `ptx_check`ed
    timed   the arms of one product (the quantizer is the same before each, so it is left out), kbench's way (`kbench.timed`:
            a CUDA graph per arm over enough weight copies to stream from DRAM, interleaved items, SM clocks):
              scaled_mm     the served product, `int8_fused.scaled_mm` at `pick`'s config
              scaled_mm_zy  the same with the records over T = max(64, M rounded up to 32) slots, each copy into its own
                            records buffer
              int_mm        today's H1 z source beside the served product: `scaled_mm`, then ``torch._int_mm`` of the
                            call's rows (`int8_exact.int_mm`)

Reported per shape: median microseconds a call per arm, zy over scaled_mm, the epilogue's added microseconds and the record
bytes a call; and Llama-3.1-8B's decode step at M = 32 (32 layers of qkv, o, gate_up, down) per arm.

On an RTX PRO 6000 under a GPU lease:

    RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO KSCRIPT=zybench.py bash benchmarks/hash/exactlin/kbench.sh [--reps 15]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kbench
import kgate

from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_fused as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

ARMS = ("scaled_mm", "scaled_mm_zy", "int_mm")
TESTS = ("tests/engine/test_int8_zy.py", "tests/engine/test_int8_ops.py")
WORK = F.Workspace("cuda", words=256 * 28672) if torch.cuda.is_available() else None


def rows_for(m: int) -> int:
    return max(64, -(-m // E.UNIT[0]) * E.UNIT[0])


def tests(out: Path) -> dict:
    cmd = [sys.executable, "-m", "pytest", *TESTS, "-rs", f"--junitxml={out / 'pytest.xml'}"]
    p = subprocess.run(cmd, cwd=ROOT / "integrations/vllm", capture_output=True, text=True)
    (out / "pytest.log").write_text(p.stdout + p.stderr)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed)", tail)}
    return {"cmd": cmd[2:], "rc": p.returncode, "summary": tail, "counts": counts}


def gate_row(n: int, k: int, m: int) -> dict:
    rows = rows_for(m)
    x = kgate.activations(m, k, 7 + m)
    w, sw = kgate.weights(n, k, 100)
    sw = sw.reshape(-1)
    xq, s = F.row_quant(x)
    cfg = F.pick(m, n, k)
    served = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    F.scaled_mm(xq, s, w, sw, served, cfg, WORK)
    y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    zy = torch.empty((E.zy_units(m, n, rows), E.ZY_BYTES), dtype=torch.uint8, device="cuda")
    kgate.poison(y, zy)
    kernel = F.scaled_mm_zy(xq, s, w, sw, y, zy, rows, cfg, WORK)
    ref = E.scaled_mm_zy(xq, s, w, sw, rows)
    ptx = F.ptx_check(kernel)
    row = {"m": m, "n": n, "k": k, "rows": rows, "cfg": str(cfg), "y_equal": torch.equal(y.view(torch.int16),
           served.view(torch.int16)), "y_is_body": torch.equal(y.view(torch.int16), ref[0].view(torch.int16)),
           "records_equal": torch.equal(zy, ref[1]), "ptx": ptx}
    row["ok"] = row["y_equal"] and row["y_is_body"] and row["records_equal"] and ptx["ok"]
    return row


def arm_fn(arm: str, xq, s, ws, y, zys, rows):
    m, k = xq.shape
    cfg = F.pick(m, ws[0][0].shape[0], k)
    if arm == "scaled_mm":
        def f():
            for w, sw in ws:
                F.scaled_mm(xq, s, w, sw, y, cfg, WORK)
    elif arm == "scaled_mm_zy":
        def f():
            for (w, sw), zy in zip(ws, zys):
                F.scaled_mm_zy(xq, s, w, sw, y, zy, rows, cfg, WORK)
    else:
        def f():
            for w, sw in ws:
                F.scaled_mm(xq, s, w, sw, y, cfg, WORK)
                E.int_mm(xq, w)
    return f


def shape_row(n: int, k: int, m: int, reps: int, clock, names: list[str]) -> dict:
    rows = rows_for(m)
    copies = kbench.copies_for(n, k)
    xq, s = F.row_quant(kgate.activations(m, k, 7 + m, special=False))
    ws = [(w, sw.reshape(-1)) for w, sw in (kgate.weights(n, k, 100 + i) for i in range(copies))]
    y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    zys = [torch.empty((E.zy_units(m, n, rows), E.ZY_BYTES), dtype=torch.uint8, device="cuda") for _ in range(copies)]
    med, us, meta = kbench.timed({a: arm_fn(a, xq, s, ws, y, zys, rows) for a in ARMS}, reps, clock, copies)
    record_bytes = zys[0].numel()
    row = {"layer": ",".join(names), "m": m, "n": n, "k": k, "rows": rows, "cfg": str(F.pick(m, n, k)), "copies": copies,
           "record_bytes_per_call": record_bytes, "us_per_call": med, "zy_over_scaled_mm": med["scaled_mm_zy"] / med["scaled_mm"],
           "epilogue_us": med["scaled_mm_zy"] - med["scaled_mm"], "int_mm_over_zy": med["int_mm"] / med["scaled_mm_zy"],
           "us_all": {a: [round(v, 3) for v in us[a]] for a in ARMS}, "sm_clock_mhz": meta["clocks"],
           "replays_per_item": meta["replays_per_item"]}
    del xq, s, ws, y, zys
    torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--ms", default="1,32,256")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ms = tuple(int(v) for v in a.ms.split(","))
    shapes = kgate.shapes_of(("llama-3.1-8b",), ms)
    doc: dict = {"device": torch.cuda.get_device_name(), "arch": F.arch(), "reps": a.reps, "arms": ARMS}
    if not a.skip_tests:
        doc["tests"] = tests(out)
        print(json.dumps({"tests": doc["tests"]}), flush=True)
    gate = [gate_row(n, k, m) for n, k, m, _ in kgate.shapes_of(("llama-3.1-8b",), (1, 17, 32, 64, 256))]
    doc["gate_ok"] = all(r["ok"] for r in gate)
    (out / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    print(json.dumps({"gate_ok": doc["gate_ok"]}), flush=True)
    if not doc["gate_ok"]:
        (out / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
        return 4
    clock = kbench.nvml()
    rows = []
    for n, k, m, names in shapes:
        row = shape_row(n, k, m, a.reps, clock, names)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("us_all", "sm_clock_mhz")}), flush=True)
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    spec = kgate.MODELS["llama-3.1-8b"]
    at32 = {(r["n"], r["k"]): r for r in rows if r["m"] == 32}
    if len(at32) == len(spec["linears"]):
        doc["decode_step_m32"] = {arm: spec["layers"] * sum(at32[nk]["us_per_call"][arm] for nk in spec["linears"].values())
                                  / 1e3 for arm in ARMS}
        doc["decode_step_m32"]["unit"] = "ms of linear products per step"
    doc["rows"] = [{k: v for k, v in r.items() if k not in ("us_all", "sm_clock_mhz")} for r in rows]
    (out / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"decode_step_m32": doc.get("decode_step_m32")}), flush=True)
    return 0 if a.skip_tests or doc["tests"]["rc"] == 0 else 5


if __name__ == "__main__":
    sys.exit(main())
