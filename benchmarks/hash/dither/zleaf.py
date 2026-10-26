"""The z commitment's hashing for one step on Llama-3.1-8B's 128 linears, gated against `served_h1/commit.py` and timed,
in two forms: `row`, dither_one's z table (each slot's N int32 a hiding leaf, domain 'z-table'), and `zy`, CA's z/y
leaves as dither_a commits them (one hiding leaf a `dither_a.UNIT` unit, its record its z int32 then its y bf16, domain
`dither_a.ZY_KIND`).  The y-only arm builds neither; what the leaves cost is this.

    gate     T slots (--t): each table's bytes random (the hash reads bytes, whatever they hold); every table's levels
             and roots poisoned with 0xA5 first; one h1c_build of the step's 128 tables; every call's root = commit.py's
             over the same bytes (a pool of CPU workers); no poison left in a root; a control: one byte of call 5 flipped
             moves that call's root and no other
    time     each form's 128 tables in one h1c_build (`batch`: a between-steps build) and in 128 h1c_builds of one table
             each (`percall`: what the side stream issues, one a call), each captured as a CUDA graph and replayed
             between CUDA events; with the bytes each reads and writes

    python zleaf.py --lib libh1c.so --out OUT [--t 64,256]
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h1c as H1CM  # noqa: E402
from dither_a import UNIT, ZY_KIND, zy_bytes, zy_fields_of  # noqa: E402
from h1c import CM, H1C, Dom, Tab, unwords  # noqa: E402
from h1cgate import POISON, linears  # noqa: E402

FORMS = ("row", "zy")


def fields(form: str, window: str, s: int, w: int, module: str, t: int, n: int) -> dict:
    if form == "zy":
        return zy_fields_of(window, s, w, module, t, n, UNIT)
    return {"window": window, "step": s, "linear": w, "module": module, "slots": t, "n": n, "schema": CM.SCHEMA}


def shape(form: str, t: int, n: int) -> tuple[int, int, str]:
    """(leaves, leaf bytes, domain kind) of one call's table."""
    if form == "zy":
        return (t // UNIT[0]) * (n // UNIT[1]), zy_bytes(UNIT), ZY_KIND
    return t, 4 * n, "z-table"


class Form:
    def __init__(self, lib: H1C, form: str, window: str, t: int, lin, dev, seed: int):
        self.form, self.lib, self.t, self.lin = form, lib, t, lin
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.data, tabs = [], []
        for w, (name, n, _) in enumerate(lin):
            cnt, size, kind = shape(form, t, n)
            self.data.append(torch.randint(0, 256, (cnt * size,), dtype=torch.uint8, generator=g).to(dev))
            tabs.append(Tab(Dom(kind, fields(form, window, 0, w, name, t, n), cnt, {"step": (1, 0)}), cnt,
                            data=self.data[w], stride=size, body=size, length=size,
                            lab=("zy/" if form == "zy" else "z/", f"/{w}")))
        self.set = lib.set(tabs)
        self.step = torch.zeros(2, dtype=torch.int32, device=dev)
        self.tb = C.sizeof(H1CM._Tab)

    def batch(self) -> None:
        self.lib.build(self.set, self.step)

    def percall(self) -> None:
        st, cs = self.set, torch.cuda.current_stream().cuda_stream
        for w in range(len(self.lin)):
            rc = self.lib.lib.h1c_build(C.addressof(st.host) + self.tb * w, st.dev.data_ptr() + self.tb * w, 1,
                                        self.step.data_ptr(), self.lib.btab.data_ptr(), 1, cs)
            if rc:
                raise RuntimeError(f"h1c_build({w}) returned {rc}")

    def roots(self) -> list[bytes]:
        return [unwords(self.set.root(w)) for w in range(len(self.lin))]

    def cpu_roots(self, pool, window: str, key: bytes, s: int) -> list[bytes]:
        out = []
        for w, (name, n, _) in enumerate(self.lin):
            tab = self.set.tabs[w]
            dom = CM.domain(fields(self.form, window, s, w, name, self.t, n), tab.count, tab.dom.kind)
            rows = self.data[w].cpu().numpy().reshape(tab.count, tab.length)
            lab = f"{tab.lab[0]}{s}{tab.lab[1]}"
            step = max(1, tab.count // 16)
            jobs = [(dom.binding, tab.count, key, lab, i, [bytes(r) for r in rows[i:i + step]])
                    for i in range(0, tab.count, step)]
            out.append(CM.levels(dom, CM.split(b"".join(pool.map(CM.leaves, jobs))))[-1][0])
        return out

    def bytes_moved(self) -> dict:
        tabs = self.set.tabs
        data = sum(x.count * x.length for x in tabs)
        segs = sum(x.count * ((x.length + 1023) // 1024) for x in tabs)
        leaves = sum(x.count for x in tabs)
        return {"leaves": leaves, "data_read": data, "segment_digests": 64 * segs, "leaves_and_levels": 64 * 2 * leaves}


def gate(f: Form, pool, window: str, key: bytes, s: int) -> dict:
    f.step.fill_(s)
    f.set.level.fill_(POISON)
    f.set.roots.fill_(POISON)
    f.batch()
    torch.cuda.synchronize()
    got = f.roots()
    poisoned = [w for w, r in enumerate(got) if b"\xa5" * 8 in r]
    ref = f.cpu_roots(pool, window, key, s)
    bad = [w for w in range(len(got)) if got[w] != ref[w]]
    f.set.roots.fill_(POISON)
    f.percall()
    torch.cuda.synchronize()
    percall_bad = [w for w, r in enumerate(f.roots()) if r != got[w]]
    w0 = 5
    f.data[w0][100] ^= 1
    f.batch()
    torch.cuda.synchronize()
    after = f.roots()
    f.data[w0][100] ^= 1
    control = {"tampered_call_moved": after[w0] != got[w0],
               "others_kept": all(after[w] == got[w] for w in range(len(got)) if w != w0)}
    ok = not bad and not poisoned and not percall_bad and all(control.values())
    return {"form": f.form, "s": s, "calls": len(got), "roots_vs_commit_bad": bad, "percall_vs_batch_bad": percall_bad,
            "roots_poisoned": poisoned, "control": control, "ok": ok}


def timed(g: torch.cuda.CUDAGraph, reps: int) -> list[float]:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        out.append(e0.elapsed_time(e1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.environ.get("DITHER_H1C"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default=None, help="unused (sbench.sh's SBENCH_SCRIPT passes it)")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--t", default="64,256")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    key = os.urandom(64)
    window = os.environ.get("RESEARCH_RUN_ID") or "gate"
    lib = H1C(a.lib, key, dev)
    lin = linears()
    doc: dict = {"gpu": torch.cuda.get_device_name(), "unit": list(UNIT), "by_t": {}}
    with ProcessPoolExecutor(a.workers) as pool:
        for t in (int(x) for x in a.t.split(",")):
            d = doc["by_t"][t] = {"t": t}
            forms = {x: Form(lib, x, window, t, lin, dev, t) for x in FORMS}
            for x, f in forms.items():
                d[x] = {"gate": gate(f, pool, window, key, 7), "bytes": f.bytes_moved()}
                print(json.dumps({"t": t, **d[x]["gate"], "bytes": d[x]["bytes"]}), flush=True)
            d["gate_ok"] = all(d[x]["gate"]["ok"] for x in FORMS)
            (out / "zleaf.json").write_text(json.dumps(doc, indent=1) + "\n")
            if not d["gate_ok"]:
                print("zleaf: GATE FAILED", flush=True)
                return 2
            for x, f in forms.items():
                for how in ("batch", "percall"):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        getattr(f, how)()
                    ms = timed(g, a.reps)
                    d[x][f"{how}_ms"] = round(statistics.median(ms), 4)
                    d[x][f"{how}_ms_all"] = [round(v, 4) for v in ms]
                d[x]["percall_us"] = round(1e3 * d[x]["percall_ms"] / len(lin), 2)
                print(json.dumps({"t": t, "form": x, **{k: d[x][k] for k in ("batch_ms", "percall_ms", "percall_us")}}),
                      flush=True)
            d["zy_over_row_batch"] = round(d["zy"]["batch_ms"] / d["row"]["batch_ms"], 3)
            (out / "zleaf.json").write_text(json.dumps(doc, indent=1) + "\n")
            del forms
            torch.cuda.empty_cache()
    doc["gate_ok"] = True
    (out / "zleaf.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
