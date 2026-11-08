"""Two Commit windows served on CPU, the requests running at the first's close carried into the second (`pipeline.edge_rows`):
the sampled replay's toy Program (`tests.check.test_sampled_replay` at D = V = 64, `toy`, so each key row and logits row is
whole SHA-512 blocks) executed honestly, each window a deferred Commit sealed by `served_rope.seal` whose serving rows the
in-Commit hook commits (`served_rope.hook`, `serving_rows.after_commit`), over the verifier's partition of its key rows and
samples (`TEMPLATES`).

Window 0 serves r0 and r1 from their prompts (2 prefill rows, 2 decodes) and leaves both running: each one's `edge.kv` field
is its key rows at positions 0..3 (a ``RoPE_v1{NHEADS=1}`` row's ``key`` each) and its `edge.last` field its last sample
(its last ``TokenSelect``'s). Window 1 resumes each after that history (a Program taking ``last_token``, which its first
embedding reads, and ``prefix``), runs 2 more rows from position 4 and leaves both again: its `edge.kv` field is the
history's 4 rows, then its own 2. `verify` is the verifier's side of a window: its own window from the Programs, R1-R7
with the edge's ports and the history it takes, and the edge checks.

The GLM shape (`GLM`) gives each token row the sparse-attention decoder's latent row after its key row, as GLM-5.3's layer
computes it: the kv_a projection, then its first 512 words normed (``DsaRMSNorm_v1{N=512}``, `edge.kv_c`) and its last 64
roped (``RoPEHeadGptJ_v2{D=64}``, `edge.kv_pe`), named as `_fused_norm_rope_kernel`'s outputs ``kv_c`` and ``k_pe``; each
port takes its own writer's rows, and window 1 takes them as ``prefix_kv_c`` and ``prefix_kv_pe``.

    python -m tests.pipeline.served_edge OUT [--glm]        (from integrations/vllm)

writes ``OUT/w0`` and ``OUT/w1``, each a Commit with its hook's window, and prints each window's verdicts as one JSON line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import edge as E, edge_rows as ER, serving_rows as SRW

from tests.check.test_sampled_replay import EPS
from tests.pipeline import served_rope as SR

KW = V = 64                            # a key row's words (one head of D = 64) and the vocabulary: rows of whole SHA-512 blocks
KEY, SELECT = f"RoPE_v1{{NHEADS=1,D={KW}}}", f"TokenSelect_v1{{V={V}}}"
TEMPLATES, LAW = [KEY, SELECT], "subset:1"
WORDS = {"edge.kv": KW, "edge.last": 1}
KV_C, KV_PE = 512, 64                  # GLM-5.3's latent row: KV_LORA words of normed kv_c, then QK_ROPE of roped k_pe
NORM, PE = f"DsaRMSNorm_v1{{N={KV_C},EPS={EPS}}}", f"RoPEHeadGptJ_v2{{D={KV_PE}}}"
OPEN = ("a1" * 64, "b2" * 64)          # each window's open root
PROMPTS = {"r0": [1, 2], "r1": [5, 3]}
HISTORY = 4                            # the rows window 0 leaves each request at: 2 prompt rows and 2 decodes


class Shape(NamedTuple):
    """A served toy's edge: the partition's templates, and each latent port's writer template and row words, in port order."""

    templates: list[str]
    writers: dict[str, str]
    words: dict[str, int]


TOY = Shape(TEMPLATES, {"edge.kv": KEY}, WORDS)
GLM = Shape([NORM, PE, SELECT], {"edge.kv_c": NORM, "edge.kv_pe": PE}, {"edge.kv_c": KV_C, "edge.kv_pe": KV_PE, "edge.last": 1})
_KVA, _KVN = "model.model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "model.model.layers.0.self_attn.kv_a_layernorm.weight"


@contextmanager
def toy(**at: int) -> Iterator[Any]:
    """The sampled replay's toy module at D = V = 64 and the module constants ``at``, restored after."""
    import tests.check.test_sampled_replay as T

    at = {"DH": KW, "V": V, **at}
    was = {k: getattr(T, k) for k in at}
    for k, v in at.items():
        setattr(T, k, v)
    try:
        yield T
    finally:
        for k, v in was.items():
            setattr(T, k, v)


def program(C: int, shape: Shape = TOY) -> tuple[dict, dict]:
    """The toy request Program leaving its window with its edge fields (`edge_rows.HEADER`): from its prompt for ``C`` = 0, else
    resumed after a history of ``C`` rows, one decode a step from position C (the toy's own at a 1-row prompt and 1 decode,
    its prompt the ``last_token`` it reads, each latent port's history a parameter and its cos/sin rows C positions on)."""
    with toy(**({"LP": 1, "T": 1} if C else {})) as T:
        inst, woff = T._program(rope=True)
        H = T.H
    if shape is GLM:
        inst, woff = _glm(inst, woff, H)
    ports = list(shape.writers)
    if C:
        wp = 1 + len(ports)
        for r in inst["rows"]:
            shift = C * KW if r["spec"].startswith("RoPE") else 0
            r["args"] = [[[k, wp, lo + shift * (g == 1), hi + shift * (g == 1)] if k == "p" and a == 1 else [k, a, lo, hi]
                          for k, a, lo, hi in grp] for g, grp in enumerate(r["args"])]
        hist = {v.name: v.history for v in E.VALUES}
        inst["params"] = [["last_token", "Array<1,Value<32>>"], *([hist[p], f"Array<{C * shape.words[p]},Value<16>>"] for p in ports),
                          inst["params"][1]]
    last = [r["i"] for r in inst["rows"] if r["spec"] == SELECT][-1]
    inst[ER.HEADER] = {**{p: {"bits": 16, "runs": [["p", 1 + k, 0, C * shape.words[p]]] * bool(C)
                              + [["n", r["i"], 0, shape.words[p]] for r in inst["rows"] if r["spec"] == shape.writers[p]]}
                          for k, p in enumerate(ports)},
                       "edge.last": {"bits": 32, "runs": [["n", last, 0, 1]]}}
    return inst, woff


def _glm(inst: dict, woff: dict, H: int) -> tuple[dict, dict]:
    """``inst`` with each key row followed by its token's latent row: the kv_a projection of the row its qkv projection reads
    (hidden size ``H``), its first `KV_C` words normed (`NORM`) and its last `KV_PE` roped at the key row's cos/sin row (`PE`);
    the rows renumbered and the two weights appended."""
    base = max(hi for _lo, hi in woff.values())
    mid = base + (KV_C + KV_PE) * H
    woff = {**woff, _KVA: (base, mid), _KVN: (mid, mid + KV_C)}
    ws = f"{_KVA}: Array<{KV_C + KV_PE},Array<{H},Value<16>>>, {_KVN}: Array<{KV_C},Value<16>>"
    rows, out, at = inst["rows"], [], {}

    def add(spec: str, name: str, args: list) -> int:
        out.append({"i": len(out), "spec": spec, "form": "call", "name": name, "args": args})
        return len(out) - 1

    for r in rows:
        at[r["i"]] = add(r["spec"], r["name"], [[[k, at[a] if k == "n" else a, lo, hi] for k, a, lo, hi in g] for g in r["args"]])
        if r["spec"] == KEY:
            s, k = re.search(r"rotary_embedding_(\d+)/key\[(\d+)\]", r["name"]).groups()
            ln = rows[r["args"][0][0][1]]["args"][0][0][1]
            kva = add(f"Gemm_v1{{K={H},N={KV_C + KV_PE}}}", f"model.model.layers.0.self_attn.kv_a_proj_with_mqa/triton_launch_{s}k[{k}]",
                      [[["n", at[ln], 0, H]], [["p", 1, *woff[_KVA]]]])
            add(NORM, f"model.model.layers.0.self_attn.mla_attn/fused_norm_rope_{s}/kv_c[{k}]", [[["n", kva, 0, KV_C]], [["p", 1, *woff[_KVN]]]])
            add(PE, f"model.model.layers.0.self_attn.mla_attn/fused_norm_rope_{s}/k_pe[{k}]", [[["n", kva, KV_C, KV_C + KV_PE]], r["args"][1]])
    weights = inst["params"][-1]
    params = [*inst["params"][:-1], [weights[0], weights[1][:-1] + ", " + ws + "}"]]
    return {**inst, "params": params, "rows": out}, woff


def window(root: Path, k: int, prompts: dict[str, list[int]], last: dict | None, seed: int = 17,
           shape: Shape = TOY) -> dict[str, Any]:
    """Window ``k`` under ``root``: its Program served honestly from ``prompts`` (after ``last``'s history, when given), sealed
    as a Commit, and its serving rows committed by the in-Commit hook with the edge it leaves."""
    inst, woff = program(0 if last is None else HISTORY, shape)
    with toy() as T:
        world = T.Honest(inst, woff, SR.weights(woff, seed), prompts)
    c = SR.seal(root / f"w{k}", world, shape.templates, LAW, f"edge-w{k}-cpu")
    c["edge"] = {"window": OPEN[k], "words": shape.words, "order": list(prompts), "last": None if last is None else str(last["hook"])}
    c["hook"] = SR.hook(c, root / f"w{k}" / "hook", run=f"served-edge-w{k}", edge=c["edge"])
    return c


def run(root: Path, shape: Shape = TOY) -> tuple[dict[str, Any], dict[str, Any]]:
    """Window 0 from the prompts, then window 1 from each request's state at its close: its history and last sample."""
    w0 = window(root, 0, PROMPTS, None, shape=shape)
    sel = [r["i"] for r in w0["inst"]["rows"] if r["spec"] == SELECT][-1]
    return w0, window(root, 1, {rid: [int(w0["world"].values[(rid, sel)][0])] for rid in PROMPTS}, w0, shape=shape)


def verify(c: dict[str, Any], last: dict[str, Any] | None = None, held: dict[str, list[bytes]] | None = None,
           units: Any = None, record: dict | None = None, doc: dict | None = None) -> dict[str, Any]:
    """The verifier's side of window ``c`` after ``last`` (its last window, None for the first): its own window over the
    Programs taking the history ``last`` left, R1-R7 with the edge's ports beside the window's (`edge_rows.ports`) and that
    history as ports of ``last``'s record (`edge.history`), and the edge checks, the drawn one over ``units`` (every unit by
    default), ``held`` the strings of ``last``'s check.  ``record`` and ``doc``: a prover's registration and edge file in
    place of the window's own."""
    wdir, spec = Path(c["hook"]), ER.of_request(c["edge"])
    rec = record or json.loads((wdir / "registration.json").read_text())
    before = None if last is None else json.loads((Path(last["hook"]) / ER.FILE).read_text())
    programs = {rid: c["inst"] for rid in c["world"].prompts}
    mine, rows = SRW.population(programs, json.loads(Path(c["partition"]).read_text()), c["run_root"], values=ER.values(before))
    edge, prog = ER.wiring(mine, rows, programs, spec, before), mine.partition["program"]
    publics = {m.key: (wdir / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    log = [] if last is None else [json.loads((Path(last["hook"]) / "registration.json").read_text())]
    hist = {} if before is None else E.history(list(before["ports"]), {p: e["words"] for p, e in before["ports"].items()})
    v = R.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, CS.expected_ports(mine) | ER.ports(prog, edge, spec.window),
                leaf_layer=CS.leaf_layer(publics), log=log, window_root=spec.window, **hist)
    roots, doc = {r["port"]: r["root"] for r in rec["roots"]}, doc or json.loads((wdir / ER.FILE).read_text())
    bad, strings = ER.check(mine.members, publics, roots, edge, prog, spec.window, held)
    drawn = ER.check_drawn(mine.members, publics, roots, edge, prog, spec.window, doc,
                           range(mine.population) if units is None else units)
    return {"registration": v, "edge": bad, "edge_drawn": drawn, "strings": strings, "window": mine, "ports": edge,
            "record": rec, "publics": publics, "doc": doc}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--glm", action="store_true", help="GLM-5.3's latent row as its two writers' ports (`GLM`)")
    a = ap.parse_args(argv)
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    w0, w1 = run(root, GLM if a.glm else TOY)
    v0 = verify(w0)
    v1 = verify(w1, w0, v0["strings"])
    for k, (c, v) in enumerate(((w0, v0), (w1, v1))):
        print(json.dumps({"window": k, "dir": str(c["hook"]), "registration": v["registration"].to_json(), "edge": v["edge"],
                          "edge_drawn": v["edge_drawn"], "format": v["record"]["format"], "previous": v["record"]["previous"],
                          "edge_roots": {p.name: {"rows": len(p.sources), "members": [list(x) for x in p.members],
                                                  "held": sum(u is None for u, _s in p.sources)} for p in v["ports"]},
                          "registered": [e["value"] for e in v["record"].get("registered") or ()]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
