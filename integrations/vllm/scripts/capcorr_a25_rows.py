"""[CAPCORR-2] Option-B evidence (GUIDANCE §50): locate + value-check the certified record's committed leaves against
DERIVE's all-rows program `a25fc156…` (`GreedyServe(vllm_rows=True)`, flat identity `Derived[Serve[vllm-rows]]_v1`).

`a25fc156…` is flat (14,226 leaf calls named by module fqn); CAPCORR's L2/L3 address calls through the Serve_v1
hierarchy (`t=<t>/layers.<L>/<role>`). This script bridges the two with DERIVE's VERIFIED bijection
(`serve_v1-map.json`: every Serve_v1 leaf <-> one derived node, returns agree) plus the 31 extra nodes
(`lm.model.norm/fused_add_rms_norm_59[row]`, row = prefill position 0..30), so that:

  * L3-by-structure: every capture identity (incl. `norm.rmsnorm` at prefill t<31) resolves to a call of `a25fc156…`
    with the shape / position / dependency check (the extra node at row t must consume t's layer-29 outputs);
  * L2: `verity_vllm.correspondence.core.run` evaluates the resolved calls of `a25fc156…` on the committed operands and
    compares with the committed leaves — restricted here to positions 0..30 and layer 29 (+ embed/norm/final leaves),
    i.e. exactly the leaves the pinned program `d4f4b1f1…` reports `not_in_program` (62 per request) and their operands.

Usage (pod, CPU):  PYTHONPATH=. python scripts/capcorr_a25_rows.py --map /workspace/capcorr/a25/serve_v1-map.json \
    --bundle <bundle> --checkpoint <model.safetensors> --cos-sin-npy <npy> --out <dir> [--requests r1 r2]
Read-only on the record; nothing here re-pins or re-issues anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(REPO), str(REPO / "src"), str(HERE)]

from verity_vllm.correspondence.core import CaptureIdentity, CircuitMap, CommitStore, NotInProgram, _returns_component, run as corr_run  # noqa: E402

SERVE_V1_PREFIX = "Serve_v1{C={6f196795e564},LP=32,STEPS=15}"


def derive_a25(LP: int, STEPS: int):
    import torch
    from verity_vllm.frontend.derive import derive_report
    from verity_vllm.frontend.derive_examples import GreedyServe
    from verity_vllm.registry.b1 import B0_CONFIG

    torch.manual_seed(0)
    with torch.device("meta"):
        module = GreedyServe(B0_CONFIG, steps=STEPS, vllm_rows=True)
    prompt = torch.zeros(LP, dtype=torch.int32, device="meta")
    rep = derive_report(module, (prompt,), mode="profile", profile="b1-eager", config=B0_CONFIG, name="Serve[vllm-rows]")  # derive_cli's name for --rows vllm: the digest is a25fc156…
    if not rep.ok:
        raise RuntimeError(f"derivation refused: {rep.unsupported.to_json() if rep.unsupported else 'unknown'}")
    return rep


def serve_v1(LP: int, STEPS: int):
    from verity.ir.program import Program
    from verity_vllm.registry import b1
    from verity.ir.defs import bind

    return Program(bind(b1.Serve, C=b1.B0_CONFIG, LP=LP, STEPS=STEPS))


def v1_path(call, roots) -> str:
    """Serve_v1 leaf path in `serve_v1-map.json`'s convention: `<prefix>/<root call name>[/<child index>]`
    (child index = position among the named root call's children, found by gate range)."""
    if call.node.name is not None:
        return f"{SERVE_V1_PREFIX}/{call.node.name}"
    key = (call.lo, call.hi)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]
    root = next(r for (lo, hi, r) in roots if lo <= call.lo < hi)
    kids = _KIDS_CACHE.get(root.node.name)
    if kids is None:
        kids = _KIDS_CACHE[root.node.name] = [(c.lo, c.hi) for c in root.children()]
    k = next(i for i, (lo, hi) in enumerate(kids) if lo == call.lo and hi == call.hi)
    _PATH_CACHE[key] = f"{SERVE_V1_PREFIX}/{root.node.name}/{k}"
    return _PATH_CACHE[key]


_PATH_CACHE: dict = {}
_KIDS_CACHE: dict = {}


def self_idx(cmap, call) -> int:
    return cmap.idx_of.get(id(call), -1)


class _WeightsView:
    """`Serve_v1`'s weights record view (`W["embed"]`, `W["layers"][L]["qkv_w"]`, `W["norm"]`, `W["cos_sin"]`) over the flat
    program's weight Record (fields named by module fqn), through the map's `weights_binding` (Serve_v1 field -> derived name)."""

    def __init__(self, W, wb: dict, prefix: str = ""):
        self.W, self.wb, self.prefix = W, wb, prefix

    def __getitem__(self, k):
        key = f"{self.prefix}{k}" if isinstance(k, str) else f"{self.prefix}[{k}]"
        if key in self.wb:
            return self.W[self.wb[key]]
        sep = "." if isinstance(k, int) else ""
        return _WeightsView(self.W, self.wb, key + sep)


def with_serve_v1_weights(program, mp: dict):
    """Make `program.input_coll(1)` answer Serve_v1's field names (what `correspondence.bind_weights` asks for)."""
    orig = program.input_coll
    wb = mp["weights_binding"]
    program.input_coll = lambda pi: _WeightsView(orig(1), wb) if pi == 1 else orig(pi)  # type: ignore[method-assign]
    return program


class MappedCircuitMap:
    """Capture identity -> (call of `a25fc156…`, root-scoped gates), via Serve_v1's locate + DERIVE's bijection;
    `norm.rmsnorm` at prefill t<31 -> the extra node `fused_add_rms_norm_59[t]` (checked: it consumes t's layer-29 outputs)."""

    def __init__(self, a25, v1, mp: dict):
        self.program = a25
        self.circuit = a25.circuit
        self.calls = list(a25.circuit.root_calls())
        self.idx_of = {id(c): i for i, c in enumerate(self.calls)}
        self.node_names = mp["derived_node_names"]
        self.names: dict = {}  # `run()` reads a port name table here; none for the flat program
        if len(self.calls) != len(self.node_names):
            raise RuntimeError(f"a25 root calls {len(self.calls)} != map nodes {len(self.node_names)}")
        bad = [i for i, c in enumerate(self.calls) if c.node.name != self.node_names[i]]
        if bad:
            raise RuntimeError(f"root-call order != DERIVE node order at {bad[:5]} (e.g. {self.calls[bad[0]].node.name!r} vs {self.node_names[bad[0]]!r})")
        self.path_to_idx = {path: idx for idx, path in mp["node_map"]}
        self._inv = {i: p for p, i in self.path_to_idx.items()}
        self.v1map = CircuitMap(v1)
        self.named = self.v1map.named  # for `run()`'s log line
        self.v1roots = [(r.lo, r.hi, r) for r in v1.circuit.root_calls()]  # ranges cached once (lo/hi are not free)
        self.extras = {}
        for ex in mp["extra_nodes"]:
            row = int(ex["name"].rsplit("[", 1)[1].rstrip("]"))
            # dependency check by structure: arg 0 = row's layers.29 down_proj output, arg 1 = row's layers.29 post-norm residual
            deps = [self.path_to_idx_inv(a[0][1]) for a in ex["args"][:2] if a[0][0] == "n"]
            want = (f"{SERVE_V1_PREFIX}/t={row}/layers.29/post/", )
            if len(deps) != 2 or not all(d.startswith(want) for d in deps):
                raise RuntimeError(f"extra node {ex['name']}: operands {deps} are not t={row}'s layers.29 outputs")
            if not ex["id"].startswith("RMSNormFusedCuda_v2{N=576"):
                raise RuntimeError(f"extra node {ex['name']}: unexpected function {ex['id']}")
            self.extras[row] = (ex["index"], deps, ex["id"])
        self.stats = {"via_bijection": 0, "via_extra": 0}

    def path_to_idx_inv(self, idx: int) -> str:
        return self._inv[idx]

    def locate(self, ident: CaptureIdentity):
        if ident.layer == "norm" and f"t={ident.t}/final" not in self.v1map.named:
            if ident.phase != "prefill" or ident.t not in self.extras:
                raise NotInProgram(f"t={ident.t}: no extra final-norm node for this position")
            c = self.calls[self.extras[ident.t][0]]
            self.stats["via_extra"] += 1
            return c, _returns_component(c, ident.output)
        c1, g1 = self.v1map.locate(ident)
        idx = self.path_to_idx[v1_path(c1, self.v1roots)]
        c = self.calls[idx]
        r1 = tuple(c1.returns)  # `returns` is recomputed per access: read once
        pos = {g: i for i, g in enumerate(r1)}
        r2 = tuple(c.returns)
        self.stats["via_bijection"] += 1
        return c, tuple(r2[pos[g]] for g in g1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--cos-sin-npy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--requests", nargs="+", default=["5f2dc8bd-373d-53ac-a6c3-1870b743959e", "c43a1086-c67b-558e-b04b-208567e5718e"])
    ap.add_argument("--positions", default="0-30")
    ap.add_argument("--layers", nargs="+", default=["29"])
    ap.add_argument("--challenge", default=None, help="verify/challenge.json: also locate every sampled VU (L3-by-structure)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from verity_vllm.registry.b1 import B0_CONFIG

    t0 = time.time()
    rep = derive_a25(32, 15)
    a25 = rep.program
    print(f"[a25] derived digest={rep.digest} gates={rep.gates:,} ({time.time() - t0:.1f}s)", flush=True)
    v1 = serve_v1(32, 15)
    mp = json.loads(Path(a.map).read_text())
    if mp["derived_digest"] != rep.digest:
        raise SystemExit(f"map is for {mp['derived_digest']}, derived {rep.digest}")
    cmap = MappedCircuitMap(a25, v1, mp)
    with_serve_v1_weights(a25, mp)
    print(f"[a25] {len(cmap.extras)} extra nodes: RMSNormFusedCuda_v2{{N=576}} at rows {min(cmap.extras)}..{max(cmap.extras)}, each consuming its own row's layers.29 outputs (structural check OK)", flush=True)

    result = {"program": {"digest": rep.digest, "function": rep.root if isinstance(rep.root, str) else str(rep.root), "gates": rep.gates},
              "map": {"family": mp["family"], "node_bijection": mp["node_bijection"], "extra_nodes": len(mp["extra_nodes"])},
              "extra_nodes": {str(r): {"index": i, "function": f, "operands": d} for r, (i, d, f) in sorted(cmap.extras.items())}}

    if a.challenge:
        ch = json.loads(Path(a.challenge).read_text())
        vus = sorted(set(ch["sample"]) | set(ch.get("adjacent_vus", [])))
        located, unlocated = [], []
        for n, v in enumerate(vus):
            if n % 200 == 0:
                print(f"[L3/a25] {n}/{len(vus)} ({time.time() - t0:.0f}s)", flush=True)
            try:
                c, g = cmap.locate(CaptureIdentity.parse(v))
                located.append({"vu": v, "call": c.node.name, "function": c.id, "gates": len(g),
                                "via_extra": any(c is self.calls[i] for i, _, _ in cmap.extras.values()) if False else self_idx(cmap, c) in {i for i, _, _ in cmap.extras.values()}})
            except (NotInProgram, KeyError) as e:
                unlocated.append({"vu": v, "reason": str(e)})
        result["l3_sampled_vus"] = {"total": len(vus), "located": len(located), "unlocated": len(unlocated), "unlocated_list": unlocated,
                                    "via_extra": [x for x in located if x["via_extra"]]}
        print(f"[L3/a25] sampled VUs: {len(vus)} located {len(located)} unlocated {len(unlocated)}; via extra nodes: {len(result['l3_sampled_vus']['via_extra'])}", flush=True)

    lo, hi = a.positions.split("-")
    positions = list(range(int(lo), int(hi) + 1))
    layers = [int(x) for x in a.layers]
    store = CommitStore(Path(a.bundle))
    C = dict(B0_CONFIG)
    C.setdefault("profile", "b1-eager")
    l2 = {}
    for req in a.requests:
        cmap.stats = {"via_bijection": 0, "via_extra": 0}
        t1 = time.time()
        rp = corr_run(store, [Path(p) for p in a.checkpoint], req, positions, layers, 4, set(), out / f"corr-a25-{req[:8]}.json", 0,
                      LP=32, STEPS=15, cos_sin_npy=Path(a.cos_sin_npy), config="B0", program_factory=lambda LP, STEPS: a25,
                      circuit_map=lambda p: cmap, static_config=C)
        doc = json.loads((out / f"corr-a25-{req[:8]}.json").read_text())
        norm = [r for r in doc.get("records", []) if "/l=norm/op=norm.rmsnorm/" in r["capture_id"] and "/ph=prefill/" in r["capture_id"]]
        l2[req] = {"summary": doc.get("summary"), "not_in_program": len(doc.get("not_in_program", [])),
                   "prefill_norm_leaves": {"checked": len(norm), "by_status": {s: sum(1 for r in norm if r.get("status") == s) for s in {r.get("status") for r in norm}}},
                   "locate_stats": dict(cmap.stats), "seconds": round(time.time() - t1, 1)}
        print(f"[L2/a25] {req[:8]}: {l2[req]}", flush=True)
    result["l2"] = l2
    (out / "capcorr-a25-rows.json").write_text(json.dumps(result, indent=1))
    print(f"[done] {out / 'capcorr-a25-rows.json'} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
