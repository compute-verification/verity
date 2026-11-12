"""A rehearsal of sampled_accept_capture_gpu.py with no card (scratch; not for the tree).

1. launches: the capture's own `operands`, `wrapper` and `kernels` on CPU tensors, every Triton launch turned into a
   compile for sm_90 (the binder takes the real arguments, nothing runs): do the capture's direct launches bind, and do
   they and the wrapper's compile the PTX instructions the Definition was read from (`PTX_BODY`)?  Alone and as a batch.
2. ptx: the capture's `keep_ptx` on Triton's cache after those compiles.
3. record: the capture's `judge`, `save`, `summarize` and `report` on what an interpreter of the kept PTX (`ptx_check.launch`)
   stores for a few hand vectors, written as the capture writes it; then the capture's `--replay` on that directory with
   `--definition N`.

Run from the tree's root.  Exit 0 when every step did what the capture needs of it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin-src", required=True)
    ap.add_argument("--ptx", required=True, help="the kept PTX (manifest.json and a directory a kernel)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--definition", type=int, default=1)
    ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pin_src = Path(a.pin_src).resolve()

    import torch
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction

    target = GPUTarget("cuda", 90, 32)

    class NoCard:
        def get_current_target(self):
            return target

        def get_current_device(self):
            return 0

        def get_current_stream(self, device):
            return 0

    driver.set_active(NoCard())
    vllm = types.ModuleType("vllm")
    vllm.__path__ = [str(pin_src / "vllm")]
    tu = types.ModuleType("vllm.triton_utils")
    tu.HAS_TRITON, tu.tl, tu.tldevice, tu.triton = True, tl, tldevice, triton
    sys.modules["vllm"], sys.modules["vllm.triton_utils"] = vllm, tu
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as RU

    spec = importlib.util.spec_from_file_location("capture", "adapters/vllm/tests/properties/sampled_accept_capture_gpu.py")
    C = importlib.util.module_from_spec(spec)
    sys.modules["capture"] = C
    spec.loader.exec_module(C)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ptx_check

    compiled: list[tuple[str, str, int]] = []
    real_run = JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        kernel = real_run(self, *args, grid=grid, warmup=True, **kwargs)
        compiled.append((self.fn.__name__, hashlib.sha256(C.ptx_body(kernel.asm["ptx"]).encode()).hexdigest(), id(kernel)))
        return kernel

    JITFunction.run = run
    torch.cuda.synchronize = lambda *x, **k: None

    result: dict = {"triton": triton.__version__, "torch": torch.__version__, "numpy": np.__version__}
    ok = True
    dev = torch.device("cpu")
    hand = C.SV.vectors(C.K, C.V)
    todo = hand + C.SV.random_rows(C.K, C.V, max(0, a.batch - len(hand)), 20261112)

    def launches(what: str, fn) -> list:
        nonlocal ok
        compiled.clear()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            ok = False
            result[what] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{what}: FAILED {type(e).__name__}: {e}")
            return []
        got = list(compiled)
        same = {n: h == C.PTX_BODY[n] for n, h, _ in got}
        result[what] = {"kernels": [n for n, _, _ in got], "read_instructions": same}
        good = [n for n, _, _ in got] == list(C.KERNELS) and all(same.values())
        ok = ok and good
        print(f"{what}: {len(got)} launches {[n for n, _, _ in got]}; the read instructions: {same}")
        return got

    op = C.operands(torch, [todo[0]], dev)
    w = launches("wrapper, one request", lambda: C.wrapper(torch, RU, op))
    k = launches("kernels, one request", lambda: C.kernels(torch, triton, RU, op))
    result["direct_launch_is_the_wrappers_kernel"] = [x[2] == y[2] for x, y in zip(w, k)]
    print("the direct launches reach the wrapper's compiled kernels:", result["direct_launch_is_the_wrappers_kernel"])
    ok = ok and bool(w) and all(result["direct_launch_is_the_wrappers_kernel"])
    order = [int(x) for x in np.random.default_rng(20261112).permutation(C.MAX_REQS)[:len(todo)]]
    bop = C.operands(torch, todo, dev, order)
    launches(f"wrapper, a batch of {len(todo)}", lambda: C.wrapper(torch, RU, bop))

    cache = Path(os.environ["TRITON_CACHE_DIR"])
    kept = C.keep_ptx(cache, out / "ptx")
    result["keep_ptx"] = {n: {"files": len(p["files"]), "ok": p["ok"], "code_ok": p["code_ok"],
                              "cubin_text_sha256": [f["cubin_text_sha256"] for f in p["files"]]} for n, p in kept.items()}
    print("keep_ptx:", result["keep_ptx"])
    print("cache files:", sorted({f.name for f in cache.rglob("*") if f.is_file()}))
    ok = ok and all(p["ok"] and p["code_ok"] for p in kept.values())

    manifest = json.loads((Path(a.ptx) / "manifest.json").read_text())
    (out / "cases").mkdir(exist_ok=True)
    entries = []
    for c in [hand[0], hand[1], hand[-1]]:
        sim = ptx_check.launch(Path(a.ptx), manifest, c["logits"], c["draft"], c["seed"], c["pos"], c["temp"])
        got = {name: np.asarray(sim[name]) for name in C.STORED if name != "num_sampled_after_rejection"}
        got["num_sampled_after_rejection"] = np.array([sim["num_sampled_after_rejection"]], np.int32)
        shapes = {n: (v.shape, str(v.dtype)) for n, v in got.items()}
        first = (got["sampled"].copy(), got["num_sampled"].copy())
        C.save(out / "cases", c, got, first, first)
        e = C.judge(c["name"], c, got)
        e["launches_agree"] = True
        entries.append(e)
        print(f"{c['name']}: {e['words_compared']} words, {e['words_differ']} differ; in domain {e['in_domain']}; {e['notes']}; "
              f"unwritten {e['unwritten']}")
    result["shapes"] = {n: [list(s), d] for n, (s, d) in shapes.items()}
    record = {"schema": C.SCHEMA, "statics": {"K": C.K, "V": C.V, "TILE": C.A.STATS_TILE}, "seed": 20261112, "random": 0,
              "gate": {"ok": False, "rehearsal": "the words are an interpreter's, not a card's"}, "ptx": kept,
              "summary": C.summarize(entries), "launches_disagree": [], "cases": entries,
              "batch": {"requests": 0, "differ_from_own_launch": []}}
    record["ok"] = bool(not record["summary"]["words_differ"] and record["summary"]["cases_in_domain"])
    (out / C.RECORD).write_text(json.dumps(record, indent=1) + "\n")
    print(C.report(record))
    ok = ok and record["ok"]

    rc = C.main(["--replay", str(out), "--definition", str(a.definition), "--out", str(out / "replay")])
    result["replay_rc"] = rc
    ok = ok and rc == 0
    result["ok"] = ok
    (out / "rehearsal.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"REHEARSAL {'OK' if ok else 'NOT OK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
