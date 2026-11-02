"""The class circuit's row census: per template, its units' input rows by what fills them (``wired``, ``registered``
(the weights'), ``value`` (a per-window registered value's, such as the scan state rows), ``table_root``, ``opening``, else
``free``) and their bytes, its output rows and their bytes, and the SHA-512
compressions a by-reference commit spends on them: every output row hashed fresh, every input row only re-leafed in its
port's tree, since a wired row's b||c is its writer's and a registered row's the registration's.

    python class_rows.py --tiny [--slots]                                 # TINY, locally
    python class_rows.py --llama1b [--int8] --probe 1,2,2                 # one Program at a probe class, ~2 s
    python class_rows.py --llama1b [--class 8,1024,64,4,16] --out FILE    # the class of record (~22 min, ~42 GB)
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def kinds_of(circuit, IF):
    """(tid, port) -> the kinds its rows are filled by."""
    k = defaultdict(set)
    for w in circuit.wires:
        k[(w.reader, w.port)].add("wired")
    reg = circuit.registered
    if reg is not None:
        for r in reg.reads:
            k[(r.reader, r.port)].add("registered" if r.param.split("/", 1)[0] == "weights" else "value")
        for r in reg.roots:
            k[(r.reader, r.port)].add("table_root")
        for o in reg.openings:
            for p in o.ports:
                k[(o.reader, p)].add("opening")
    return k


def blocks(nbytes: int) -> int:
    """SHA-512 compressions of a message of ``nbytes`` (its padding: 17 bytes)."""
    return -(-(nbytes + 17) // 128)


#: `rows.commit`'s per-row cost in compressions beyond the row's own digest: hm96's c over the 192-byte salt (2) and h (2),
#: the frame-v3 row leaf (1) and about one tree node a leaf (1); a row whose b||c is another row's (a wire's writer, a
#: registered row) costs h, its leaf and its node only
FRESH_EXTRA, REUSED = 6, 4


def census(circuit, IF) -> dict:
    kinds = kinds_of(circuit, IF)
    out, tot = {}, Counter()
    for fn, n in circuit.templates:
        tid = IF.descriptor_id(fn)
        ins, outs = IF.unit_rows(fn, circuit.cuts.get(tid))
        lay = IF.layout(fn)
        row = Counter()
        rb = Counter()
        for name, _schema, bits, _parts in ins:
            port = name.split("#")[0]
            ks = kinds.get((tid, port), set()) or {"free"}
            kd = "+".join(sorted(ks))
            row[kd] += 1
            rb[kd] += bits // 8
        out_bytes = [16 * w // 8 for _, w in lay.outs]
        row["out"] += len(outs)
        rb["out"] += sum(out_bytes)
        fresh = sum(1 + blocks(128 + b) + FRESH_EXTRA - 1 for b in out_bytes)  # prefix block folded in blocks(128 + b)
        reused = REUSED * sum(v for k, v in row.items() if k != "out")
        comp = n * (fresh + reused)
        lab = fn.id.split("{")[0] + "{" + ",".join(f"{k}={v}" for k, v in fn.bindings.items() if type(v) in (int, bool)) + "}"
        while lab in out:
            lab += "'"
        out[lab] = {"units": n, "rows": dict(row), "bytes": dict(rb), "compressions_a_unit": fresh + reused}
        tot["units"] += n
        for k, v in row.items():
            tot[f"rows.{k}"] += n * v
        for k, v in rb.items():
            tot[f"bytes.{k}"] += n * v
        tot["compressions"] += comp
    return {"templates": out, "total": dict(tot)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tiny", action="store_true")
    g.add_argument("--llama1b", action="store_true")
    ap.add_argument("--slots", action="store_true", help="TINY at the class of record's slots (8, 4 a boundary)")
    ap.add_argument("--class", dest="cls", default=None)
    ap.add_argument("--probe", default=None, help="lower one Program at this class only")
    ap.add_argument("--int8", action="store_true", help="the exact-int8 checkpoint on the H200 (int8_exact linears)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path.cwd() / "integrations/vllm"))
    from verity_vllm.program.predict import window_circuit as WC
    from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, Grain, WindowClass, class_circuit
    from verity.ml.windows import interface as IF
    from verity_vllm.program.predict.steps import Registered
    from verity_vllm.program.predict.phi import Phi

    t0 = time.perf_counter()
    if a.tiny:
        from tests.program.predict import test_steps as S
        reg = S._registered()
        cls = WindowClass(8, 4, 2, lanes=4, every=16) if a.slots else WindowClass(2, 4, 2)
        circuit = class_circuit(reg, cls, None, Grain(gemm_words=512, rows=9, select=64), provable=False)
    else:
        cfg = json.loads((Path.cwd() / "integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
        target = {"compute_capability": [12, 0], "num_sms": 188}
        if a.int8:
            from verity_vllm.program.model import INT8_EXACT_QUANT
            cfg = {**cfg, "quantization_config": dict(INT8_EXACT_QUANT)}
            target = {"compute_capability": [9, 0], "num_sms": 132, "int8_exact": "linears"}
        phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=target)
        reg = Registered.of(phi, default_seed=7)
        cls = WindowClass(*map(int, a.cls.split(","))) if a.cls else CLASS_OF_RECORD
        if a.probe:
            cls = WindowClass(*map(int, a.probe.split(",")))
            circuit = class_circuit(reg, cls, 1, provable=False)
        else:
            built, program = WC.class_program(reg, cls), WC.class_program
            WC.class_program = lambda r, c: built if c == cls else program(r, c)
            circuit = class_circuit(reg, cls)
    t1 = time.perf_counter()
    doc = census(circuit, IF)
    doc["class"] = {"slots": cls.slots, "prompt": cls.prompt, "outputs": cls.outputs, "positions": cls.positions}
    doc["seconds"] = round(t1 - t0, 1)
    doc["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
    s = json.dumps(doc, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(s + "\n")
    print(json.dumps(doc["total"], sort_keys=True))
    print(json.dumps({k: doc[k] for k in ("class", "seconds", "max_rss_gb")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
