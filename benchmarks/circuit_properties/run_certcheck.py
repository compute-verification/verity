"""The upstream certificate on a training step: every consumer's region, required gates and verdict, the regions' overlap
weights, where the step's counted products are, and the explicit cross-checks (`circuit_properties.certcheck`).

    uv run benchmarks/circuit_properties/run_certcheck.py [--shape scaled|real] [--step L,B,S] [--tokens all|T,..]
        [--layers all|L,..] [--designs layer,gemm] [--outputs default|none|W:ROW:COORD,..] [--bits all|B,..]
        [--explicit default|none|D:K:L:T,..] [--out DIR]

``scaled`` is `TrainStep_v1{L=2,B=1,S=16}` at `training.SCALED` (the refusals `training.scaled` lifts), every consumer of
it by default; ``real`` is `TrainStep_v1{L=30,B=8,S=512}` at smollm2-135m's shape as the registry builds it, on the tokens
and layers given. `--step` takes another L, B, S at the same shape: the registry builds smollm2-135m's at B·S = 256 or
4096, with S = 256 or 512. An explicit cross-check lists the region gate by gate, reruns `upstream.required` on that wiring and
checks `upstream.holds` (the Lean `Upstream`, mirrored) with the G found. Writes DIR/certcheck.json (DIR defaults to $OUT),
rewritten after every consumer, and prints one line each.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from circuit_properties import certcheck as CC
from circuit_properties import training as TR
from circuit_properties import upstream as U

REAL = {"L": 30, "B": 8, "S": 512}
#: the explicit cross-checks: a certified and a failing use of the GEMM design, a layer use, and an output (scaled only: a
#: real-shape region is 10^8 gates and more, too many to list one by one)
EXPLICIT = ("gemm:y2:1:5", "gemm:dup:0:5", "layer:fwd:1:5")


def _ints(text: str, everything: range) -> list[int]:
    return list(everything) if text == "all" else [int(x) for x in text.split(",") if x]


def _rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def explicit(S: CC.Step, design: str, kind: CC.Kind, i: int | None, t: int) -> dict:
    """The use's G from its explicit wiring, and whether `Upstream` holds with it."""
    acts = kind.region(S, i, t)
    A = frozenset(g for s in acts for g in range(s.offset, s.offset + s.spec.gates))
    reads = {g: tuple(S.C.gate(g).operands) for g in A}
    classes: Counter = Counter()
    for name in kind.consumers(S, i):
        first, _last, N = S.classes(S.row(name, t))
        for c in first:
            classes[c & A] += N
    G = U.required(reads, A, classes)
    K = U.Consumers(A=(A,), G=(G,), theta=frozenset(), fin_into=(classes,), names=(f"{design}:{kind.name}:{i}:{t}",))
    return {"design": design, "kind": kind.name, "layer": i, "token": t, "region_gates": len(A), "required": len(G),
            "holds": U.holds(reads, K)}


def explicit_output(S: CC.Step, weight: str, r: int, k: int, bit: int) -> dict:
    row = S.row(weight, r)
    acts = S.coordinate(row, k)
    A = frozenset(g for s in acts for g in range(s.offset, s.offset + s.spec.gates))
    reads = {g: tuple(S.C.gate(g).operands) for g in A}
    out = CC.RG.returns(row)[16 * k + bit]
    G = U.required(reads, A, [U.into(reads, frozenset({out}))])
    K = U.Consumers(A=(A,), G=(G,), theta=frozenset(), fins=((frozenset({out}),),))
    return {"weight": weight, "row": r, "coordinate": k, "bit": bit, "region_gates": len(A), "required": len(G),
            "holds": U.holds(reads, K)}


def summary(verdicts: list[dict]) -> dict:
    out = {}
    for key in sorted({(v["design"], v["kind"]) for v in verdicts}):
        vs = [v for v in verdicts if (v["design"], v["kind"]) == key]
        ok = [v["required"] for v in vs if v["required"]]
        out[":".join(key)] = {
            "uses": len(vs), "certified": len(ok),
            "required": {"min": min(ok), "median": statistics.median(ok), "max": max(ok)} if ok else None,
            "region_gates": {"min": min(v["region_gates"] for v in vs), "max": max(v["region_gates"] for v in vs)},
            "finishing_sets": sum(v["finishing_sets"] for v in vs),
            "certified_finishing_sets": sum(v["finishing_sets"] for v in vs if v["required"]),
            "why": dict(Counter(v["why"] for v in vs if v["why"]))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=("scaled", "real"), default="scaled")
    ap.add_argument("--step", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--designs", default="layer,gemm")
    ap.add_argument("--outputs", default="default")
    ap.add_argument("--bits", default="all")
    ap.add_argument("--explicit", default="default")
    ap.add_argument("--out", default=os.environ.get("OUT"))
    a = ap.parse_args(argv)
    if not a.out:
        ap.error("--out DIR, or $OUT")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shape = dict(TR.STEP) if a.shape == "scaled" else dict(REAL)
    if a.step:
        shape = dict(zip(("L", "B", "S"), (int(x) for x in a.step.split(","))))
    doc: dict = {"shape": a.shape, "step": shape, "model": TR.SCALED if a.shape == "scaled" else "smollm2-135m",
                 "argv": sys.argv[1:], "verdicts": [], "outputs": [], "explicit": []}

    def save() -> None:
        doc["seconds"], doc["peak_rss_gb"] = round(time.time() - t0, 1), round(_rss(), 2)
        (out / "certcheck.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")

    with (TR.scaled() if a.shape == "scaled" else nullcontext()):
        program = TR.step(**shape)
        S = CC.Step(program, shape["L"], shape["B"], shape["S"])
        doc["gates"] = program.root.gates
        print(f"built {program.root.id}: {program.root.gates:,} gates, {time.time() - t0:.0f}s, {_rss():.1f} GB",
              flush=True)
        doc["products"] = CC.coverage(S)
        print("products", doc["products"], flush=True)
        tokens = _ints(a.tokens or ("all" if a.shape == "scaled" else f"1,{S.M - 1}"), range(S.M))
        ends = ",".join(str(i) for i in sorted({0, 1, S.L - 2, S.L - 1} & set(range(S.L))))
        layers = _ints(a.layers or ("all" if a.shape == "scaled" else ends), range(S.L))
        doc["tokens"], doc["layers"] = tokens, layers
        designs = [d for d in a.designs.split(",") if d]
        weights = {}
        for d in designs:
            w = CC.overlap_weights(CC.spans(S, d, range(S.M)))
            weights[d] = {str(k): n for k, n in Counter(w.values()).items()}
        doc["overlap_weights"] = weights
        print("overlap weights over every use", weights, flush=True)
        save()
        for d in designs:
            for kind in CC.DESIGNS[d]:
                for i in (layers if kind.per_layer else [None]):
                    for t in tokens:
                        s = time.time()
                        v = CC.as_dict(S.use(d, kind, i, t))
                        v["seconds"] = round(time.time() - s, 1)
                        doc["verdicts"].append(v)
                        print(json.dumps(v), flush=True)
                        save()
        doc["summary"] = summary(doc["verdicts"])
        if a.outputs == "default":
            first = [f"l0.{w}.dw" for w in CC.LINEARS] + [f"l{S.L - 1}.{w}.dw" for w in CC.LINEARS] + ["lm_head.dw"]
            specs = [(w, r, k) for w in first for r, k in ((0, 0), (S.rows(w) - 1, 1))]
        elif a.outputs == "none":
            specs = []
        else:
            specs = [(w, int(r), int(k)) for w, r, k in (x.rsplit(":", 2) for x in a.outputs.split(","))]
        bits = _ints(a.bits, range(16))
        for w, r, k in specs:
            for b in bits:
                s = time.time()
                o = CC.as_dict(S.output(w, r, k, b))
                o["seconds"] = round(time.time() - s, 1)
                doc["outputs"].append(o)
                print(json.dumps(o), flush=True)
            save()
        if a.explicit == "default" and a.shape == "real":
            a.explicit = "none"
        ex = EXPLICIT if a.explicit == "default" else () if a.explicit == "none" else a.explicit.split(",")
        for spec in ex:
            d, name, i, t = spec.split(":")
            kind = next(k for k in CC.DESIGNS[d] if k.name == name)
            s = time.time()
            e = explicit(S, d, kind, None if i == "None" else int(i), int(t))
            e["seconds"] = round(time.time() - s, 1)
            doc["explicit"].append(e)
            print(json.dumps(e), flush=True)
            save()
        if specs and a.explicit != "none":
            w, r, k = specs[0]
            e = explicit_output(S, w, r, k, 15)
            doc["explicit"].append(e)
            print(json.dumps(e), flush=True)
        save()
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
