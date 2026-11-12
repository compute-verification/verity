"""A block-FP8 routed MoE layer served on CPU, for the per-expert fused cut's tests: one layer in GLM-5.3's shape at E = 8
experts, TOPK = 2, two tokens of one request, H = 128, I = 64, G = KB = 32.  Its Program is written here in the order the Build
derives the layer (the input's group quant a token, the router, the w13 pair rows, the SiLU, the activation's group quant and
the w2 pair rows a pair, the sum a token), executed honestly by the row kernels over a synthetic checkpoint, and committed as
the MoE block's planes (``moe/L3/<plane>``, pair-major) by the production CPU committer.  The one run of committed steps is

* a deferred Commit's (``OUT/commit``: its record, binding map and sealed replay bundle, as `tests.pipeline.served_rope`
  seals one), whose in-Commit hook's window is the row driver's (`hook`: ``served``), and
* a serve-openai window's, sealed as the serve loop seals a closed window (`serve`, ``OUT/served``), which names no Programs.

Neither bundle carries a registered weight: a window made from one composes the weights its rows read from the checkpoint.

    python -m tests.pipeline.served_experts OUT        (from adapters/vllm)

writes both with ``OUT/row`` (the Program of record), ``OUT/ckpt`` and ``OUT/checkpoints.json`` (the checkpoint and its shard's
pin), ``OUT/tables-rank0.json`` and ``OUT/tables.key`` (the per-expert tables' registration and the committer's key),
``OUT/partition.json`` (each projection's scale-row reads and fused blocks) and ``OUT/hook``, and prints their paths as one
JSON line.  No Build made the Program: its digest is the SHA-512 of its parameters and rows as written here.

    python -m tests.pipeline.served_experts --same A B

exits 0 where the windows in directories ``A`` and ``B`` commit one thing (`same_commitment`), and else says what differs: how
a suite outside the integration holds its window of the layer to the hook's."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from verity.commitments.windows import rows as CS
from verity_vllm.commit import step_rows
from verity_vllm.pipeline import cpu_replay as CPU
from verity_vllm.pipeline import replay_bundle as RB

E, TOPK, M, H, I, G, KB = 8, 2, 2, 128, 64, 32, 32
NV = M * TOPK
LAYER = 3
WRAP = "model."
MLP = f"{WRAP}model.layers.{LAYER}.mlp"
X = f"{MLP}.experts"
OP = f"moe/L{LAYER}"
#: each projection's fused block, in rows
BLOCKS = {"w13": 4, "w2": 8}
#: the window's units by projection: its scale-row reads (a row a G-block of each pair) and its fused blocks
UNITS = {"w13": (NV * 2 * I // G, NV * 2 * I // BLOCKS["w13"]), "w2": (NV * H // G, NV * H // BLOCKS["w2"])}
#: the most units of a template a member holds here: a small number set by hand so a template's units are several members, not
#: the units one flock-circuit statement holds (`serving_rows.chunk_caps`, which stages the template with the verifier's tools)
CAP = 48
REPO, REV = "test/fp8-moe-served", "abc"
KEY = bytes(range(32))
REQUEST = "r0"
LAW = "stratified:1"
LABEL = "fp8-moe-cpu"
EPS = '{"f64":"0x1.0c6f7a0b5ed8dp-20"}'
#: a registered tensor's dtype as the weights tree names it, and a committed plane's bytes, by element width
DTYPES = {8: "float8_e4m3fn", 16: "bfloat16", 32: "float32"}
WORDS = {8: np.uint8, 16: np.uint16, 32: np.uint32}


def fields() -> list[tuple[str, tuple[str, list[int], int]]]:
    """The Program's weight fields in its ``weights`` struct's order, each by a short key: (name under the export wrapper, shape,
    bits)."""
    return [("bias", (f"{MLP}.gate.e_score_correction_bias", [E], 32)),
            ("w13", (f"{X}.w13_weight", [E, 2 * I, H], 8)), ("w13_s", (f"{X}.w13_weight_scale_inv", [E, 2 * I // G, H // G], 32)),
            ("w2", (f"{X}.w2_weight", [E, H, I], 8)), ("w2_s", (f"{X}.w2_weight_scale_inv", [E, H // G, I // G], 32)),
            ("norm", (f"{WRAP}model.layers.{LAYER + 1}.input_layernorm.weight", [H], 16)), ("head", (f"{WRAP}lm_head.weight", [8, H], 16))]


def _type(shape: list[int], bits: int) -> str:
    t = f"Value<{bits}>"
    for n in reversed(shape):
        t = f"Array<{n},{t}>"
    return t


def specs() -> SimpleNamespace:
    """The layer's row Definitions as vLLM's defaults on an H200 serve them: DeepGEMM's ue8m0 group quant, the sigmoid router,
    w13 unweighted on MMA v2, w2 weighted and swapped on QGMMA."""
    from verity.circuits.codec import descriptor_id
    from verity.circuits.defs import bind
    from verity.definitions.ops.prims import HopperE4m3MmaV2Dot32, HopperE4m3QgmmaDot32
    from verity.definitions.registry import act, deepgemm, fp8_moe_experts, moe

    return SimpleNamespace(
        in_quant=descriptor_id(bind(deepgemm.Fp8GroupQuantUe8m0, K=H, G=G)),
        router=descriptor_id(bind(moe.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block")),
        up=descriptor_id(bind(fp8_moe_experts.MoeExpertFp8Gemm, E=E, K=H, N=2 * I, G=G, KB=KB, SWAP=False, DOT=HopperE4m3MmaV2Dot32)),
        act=descriptor_id(bind(act.SiluMul, I=I)),
        act_quant=descriptor_id(bind(deepgemm.Fp8GroupQuantUe8m0, K=I, G=G)),
        down=descriptor_id(bind(fp8_moe_experts.MoeExpertFp8GemmW, E=E, K=I, N=H, G=G, KB=KB, SWAP=True, DOT=HopperE4m3QgmmaDot32)),
        sum=descriptor_id(bind(moe.MoeSum, TOPK=TOPK, H=H)))


def program() -> dict[str, Any]:
    """The layer's Program, one engine step: its rows under the MoE runner's module, each named ``<module>/<kind>[<ordinal>]``
    (token t's slot k is pair t TOPK + k), then the next layer's norm, the logits and the token select that close the step.
    Parameter 0 is the router's logits, 1 the block's input, 2 the weights."""
    S = specs()
    w, off = {}, 0
    for key, (_name, shape, _bits) in fields():
        w[key] = [["p", 2, off, off + math.prod(shape)]]
        off += math.prod(shape)
    rows: list[dict[str, Any]] = []

    def add(spec: str, name: str, args: list) -> int:
        rows.append({"i": len(rows), "spec": spec, "form": "call", "name": name, "args": args})
        return len(rows) - 1

    pairs = [(t, k) for t in range(M) for k in range(TOPK)]
    iq = [add(S.in_quant, f"{X}/in_quant[{t}]", [[["p", 1, t * H, (t + 1) * H]]]) for t in range(M)]
    rt = [add(S.router, f"{X}/router[{t}]", [[["p", 0, t * E, (t + 1) * E]], w["bias"]]) for t in range(M)]
    up = [add(S.up, f"{X}/up[{p}]", [[["n", iq[t], 0, H]], [["n", iq[t], H, H + H // G]], [["n", rt[t], TOPK + k, TOPK + k + 1]],
                                    w["w13"], w["w13_s"]]) for p, (t, k) in enumerate(pairs)]
    ac = [add(S.act, f"{X}/act[{p}]", [[["n", up[p], 0, 2 * I]]]) for p in range(NV)]
    aq = [add(S.act_quant, f"{X}/act_quant[{p}]", [[["n", ac[p], 0, I]]]) for p in range(NV)]
    dn = [add(S.down, f"{X}/down[{p}]", [[["n", aq[p], 0, I]], [["n", aq[p], I, I + I // G]], [["n", rt[t], TOPK + k, TOPK + k + 1]],
                                        [["n", rt[t], k, k + 1]], w["w2"], w["w2_s"]]) for p, (t, k) in enumerate(pairs)]
    sm = [add(S.sum, f"{X}/moe_sum[{t}]", [[["n", dn[t * TOPK + k], 0, H] for k in range(TOPK)]]) for t in range(M)]
    nm = [add(f"RMSNormFusedCuda_v2{{N={H},EPS={EPS}}}", f"{WRAP}model.layers.{LAYER + 1}.input_layernorm/fused_add_rms_norm[{t}]",
              [[["n", s, 0, H]], [["p", 1, t * H, (t + 1) * H]], w["norm"]]) for t, s in enumerate(sm)]
    lg = add(f"Gemm_v1{{K={H},N=8}}", f"{WRAP}lm_head/linear[0]", [[["n", nm[-1], 0, H]], w["head"]])
    add("TokenSelect_v1{V=8}", "argmax[0]", [[["n", lg, 0, 8]]])
    params = [["logits", _type([M, E], 32)], ["x", _type([M, H], 16)],
              ["weights", "{" + ", ".join(f"{name}: {_type(shape, bits)}" for _key, (name, shape, bits) in fields()) + "}"]]
    digest = hashlib.sha512(json.dumps([params, rows], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "cb-a/instance-sequence/v1", "digest": digest, "params": params, "rows": rows}


def _run(parts: list[np.ndarray], lo: int, hi: int) -> np.ndarray:
    """Elements [lo, hi) of ``parts`` laid end to end, inside one of them."""
    for a in parts:
        if lo < a.size:
            if hi > a.size:
                break
            return a[lo:hi]
        lo, hi = lo - a.size, hi - a.size
    raise LookupError(f"a run [{lo}, {hi}) is not inside one member or field")


def execute(inst: dict[str, Any], logits: np.ndarray, x: np.ndarray, weights: dict[str, np.ndarray]) -> dict[int, list[np.ndarray]]:
    """Each MoE row's output members' words by row, in the Program's order: its row kernel (`verity.definitions.numerics.rows`) on
    the words its argument runs read, a node run from its producer's members in return order and a parameter run from the
    logits, the input or the weights' fields (``weights``, by the Program's field name).  The rows after the block are not served."""
    from verity.definitions.numerics import rows as K
    from verity_vllm.query.manifest import format as RM

    params = [[np.asarray(logits).reshape(-1)], [np.asarray(x).reshape(-1)], [weights[name] for _key, (name, _s, _b) in fields()]]
    out: dict[int, list[np.ndarray]] = {}
    for r in inst["rows"]:
        if RM.split_name(r["name"])[0] != X:
            continue
        k = K.ROWS[RM.spec_family(r["spec"])]
        ops = [np.concatenate([_run(out[int(a)] if kind == "n" else params[int(a)], int(lo), int(hi)) for kind, a, lo, hi in group])
               for group in r["args"]]
        got = k.fn(RM.spec_statics(r["spec"]), ops)
        out[int(r["i"])] = [np.asarray(got[m]).reshape(-1) for m in k.returns]
    return out


def planes(inst: dict[str, Any], out: dict[int, list[np.ndarray]]) -> list[tuple[str, bytes, None, str, tuple[int, ...]]]:
    """The step's committed tensors: the block's planes as the MoE block source stages them, each kind's rows in order (so
    pair-major), under their names, torch dtypes and shapes."""
    def of(kind: str, member: int = 0) -> np.ndarray:
        return np.concatenate([out[int(r["i"])][member] for r in inst["rows"] if r["name"].startswith(f"{X}/{kind}[")])

    made = [("router_w", of("router", 0), "float32", (M, TOPK)), ("router_ids", of("router", 1), "int32", (M, TOPK)),
            ("in_q", of("in_quant", 0), "float8_e4m3fn", (M, H)), ("in_scale", of("in_quant", 1), "float32", (M, H // G)),
            ("up", of("up"), "bfloat16", (M, TOPK, 2 * I)), ("act", of("act"), "bfloat16", (NV, I)),
            ("act_q", of("act_quant", 0), "float8_e4m3fn", (NV, I)), ("act_scale", of("act_quant", 1), "float32", (NV, I // G)),
            ("down", of("down"), "bfloat16", (M, TOPK, H)), ("sum", of("moe_sum"), "bfloat16", (M, H))]
    size = {"float32": "<u4", "int32": "<u4", "float8_e4m3fn": "u1", "bfloat16": "<u2"}
    for name, a, _dtype, shape in made:
        if a.size != math.prod(shape):
            raise ValueError(f"plane {name}: {a.size} words, its shape {shape} holds {math.prod(shape)}")
    return [(f"{OP}/{name}", np.ascontiguousarray(a.astype(size[dtype])).tobytes(), None, dtype, shape) for name, a, dtype, shape in made]


def _bf16(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def checkpoint(root: Path, rng: np.random.Generator) -> tuple[Path, Path]:
    """A block-FP8 MoE checkpoint in the HF layout under ``root/ckpt`` (per expert its e4m3 projections, no NaN byte, and their
    f32 block scales; the router's bias; the next norm's weight and the head's) and ``root/checkpoints.json`` pinning its shard."""
    from tests.check.test_weights_of_record import write_safetensors
    from verity_vllm.check import weights_of_record as WR

    c = f"model.layers.{LAYER}.mlp"
    t: dict[str, tuple[str, np.ndarray]] = {}
    for e in range(E):
        for part, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            w = rng.integers(0, 256, (n, k), dtype=np.uint8)
            w[(w & 0x7F) == 0x7F] ^= 1
            t[f"{c}.experts.{e}.{part}.weight"] = ("F8_E4M3", w)
            t[f"{c}.experts.{e}.{part}.weight_scale_inv"] = ("F32", np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), (n // G, k // G))).astype(np.float32))
    t[f"{c}.gate.e_score_correction_bias"] = ("F32", (rng.standard_normal(E) * 1e-2).astype(np.float32))
    t[f"model.layers.{LAYER + 1}.input_layernorm.weight"] = ("BF16", _bf16(rng, H))
    t["lm_head.weight"] = ("BF16", _bf16(rng, 8, H))
    d = root / "ckpt"
    d.mkdir(parents=True)
    write_safetensors(str(d / "model.safetensors"), t)
    (d / "config.json").write_text(json.dumps({
        "hidden_size": H, "moe_intermediate_size": I, "num_experts": E, "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [G, G]}}))
    man = root / "checkpoints.json"
    man.write_text(json.dumps({"checkpoints": [{"repo": REPO, "revision": REV, "local_path": str(d), "files": {
        "model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}))
    return d, man


def served_weights(ck: Any) -> dict[str, np.ndarray]:
    """The registered tensors as served, by registered name (the Program's field less the wrapper prefix): the checkpoint
    composed for the one rank (`weights_of_record.compose`), in the widths the Program reads."""
    from verity_vllm.check import weights_of_record as WR

    out = {}
    for _key, (name, shape, bits) in fields():
        words, _note = WR.compose(name.removeprefix(WRAP), list(shape), bits, ck)
        out[name.removeprefix(WRAP)] = np.ascontiguousarray(words).reshape(-1).view(WORDS[bits])
    return out


def weights_tree(served: dict[str, np.ndarray]) -> dict[str, Any]:
    """The weights registration's tree over the served tensors, as a committer registers them under rows-sha512."""
    geo, wk = bytes(range(32)), bytes(range(64))
    names, roots = [], []
    for i, (_key, (name, shape, bits)) in enumerate(fields()):
        raw = served[name.removeprefix(WRAP)].view(np.uint8)
        names.append([name.removeprefix(WRAP), DTYPES[bits], list(shape), int(raw.size)])
        roots.append(step_rows.tensor_root(geo, i, names[-1], raw, wk))
    return step_rows.weights_tree(geo, names, step_rows.names_digest(names), roots, wk)


def partition() -> dict[str, Any]:
    """The fused cut's partition of the layer: each projection's scale-row read, then its fused block of ``BLOCKS`` rows (of the
    pair row's unit templates, `serving_rows.unit_templates`), each template's members at most ``CAP`` units.  The router's
    slots stay out of the window."""
    from verity.circuits.codec import descriptor_id
    from verity.commitments.gates.embedding import read
    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.pipeline.serving_templates import FUSED
    from verity_vllm.query.program_view import spec_base, spec_statics

    S, tmpls = specs(), []
    for proj, spec, K in (("w13", S.up, H), ("w2", S.down, I)):
        fused = [t for t, _k in SRW.unit_templates(spec) or ()
                 if spec_base(t) == FUSED[spec_base(spec)] and int(spec_statics(t)["N"]) == BLOCKS[proj]]
        if len(fused) != 1:
            raise ValueError(f"{spec}: {len(fused)} fused blocks of {BLOCKS[proj]} rows bind, not one")
        tmpls += [descriptor_id(read(E, 2 * K // G)), fused[0]]
    return {"templates": tmpls, "law": LAW, "scope": {"op_path_prefix": OP}, "chunk_cap": dict.fromkeys(tmpls, CAP)}


def commit(root: Path, *, seed: int = 23) -> dict[str, Any]:
    """Under ``root``: the checkpoint and its pins, the Program's row dir, the tables' registration and key, the partition file
    and the deferred Commit's output dir over the layer's committed step; and the committer, its run commitment and the served
    weights, for `hook` and `serve`.  ``seed`` samples the checkpoint, the logits and the input."""
    import torch

    from tests.check.opened import commit_steps
    from verity_vllm.check import weights_of_record as WR
    from verity_vllm.commit import binding as B
    from verity_vllm.commit import store_dump as SD
    from verity_vllm.pipeline import expert_tables as ET

    rng = np.random.default_rng(seed)
    inst = program()
    ckpt, man = checkpoint(root, rng)
    served = served_weights(WR.CheckpointIndex(str(ckpt), checkpoints_manifest=str(man)))
    logits = (rng.standard_normal((M, E)) * 2).astype(np.float32).view(np.uint32)
    values = execute(inst, logits, _bf16(rng, M, H), {WRAP + name: a for name, a in served.items()})
    com, rc = commit_steps({0: planes(inst, values)}, run_id="served-experts")
    tree = weights_tree(served)
    rc.weight_registration = {"root": tree["root"]}
    com.weights_tree, com.model = tree, torch.nn.Module()

    row_dir = root / "row"
    (row_dir / "build_request").mkdir(parents=True)
    with gzip.open(row_dir / "build_request" / "instances.json.gz", "wt") as f:
        json.dump(inst, f)
    (row_dir / "build_request" / "result.json").write_text(json.dumps({"program_digest": inst["digest"]}))
    (root / "tables-rank0.json").write_text(json.dumps(
        ET.register(str(row_dir / "build_request"), str(ckpt), KEY, BLOCKS, checkpoints_manifest=str(man)), indent=1) + "\n")
    (root / "tables.key").write_bytes(KEY)
    tables = {"key": str(root / "tables.key"), "registration": str(root / "tables-rank{rank}.json")}
    pf = root / "partition.json"
    pf.write_text(json.dumps(partition()) + "\n")

    out = root / "commit"
    p = RB.partial(out, 1)
    SD.dump(com, rc, p / "store")
    (p / "weights" / "extra").mkdir(parents=True)
    (p / "weights" / "tree.json").write_text(json.dumps(tree | {"extra": {}}))
    bmap = B.build_binding_map(com._layouts, rc.steps, rc.run_root, inst["digest"], request_id=REQUEST, rank=0, chunk=com.chunk,
                               weight_registration=rc.weight_registration, request_rows={0: [(REQUEST, 0, M)]})
    (out / "binding_map_p1.json").write_text(json.dumps(bmap, default=str))
    sr, lk = RB.deferred_records(RB.path(out, 1))
    row = {"pair": 1, "arm": "instrumented", "commit": {"run_root": rc.run_root.hex(), "commit_pass": None, "sampled_replay": sr,
                                                        "boundary_linkage": lk, "weight_registration": {"root": tree["root"]}}}
    reqs = [{"request_id": REQUEST, "index": 0, "prompt_token_ids": list(range(M)), "max_tokens": 1, "arrive_step": 0}]
    ctx = {"pair": 1, "run_root": rc.run_root.hex(), "requests": reqs, "sampling": {"max_tokens": 1}, "code": "test",
           "args": {"program_dir": "/served/build_request", "programs_root": None, "program_digest": inst["digest"],
                    "required_manifest": None, "label": LABEL}}
    m = RB.seal(out, 1, ctx, out / "binding_map_p1.json")
    (out / "runs.jsonl").write_text(json.dumps(row) + "\n")
    (out / "verdict.json").write_text(json.dumps({"replay_deferred": {"bundles": [{"pair": 1, "manifest_sha256": m["sha256"]}]}}))
    return {"out": out, "row_dir": row_dir, "partition": pf, "run_root": rc.run_root.hex(), "inst": inst, "checkpoint": ckpt,
            "checkpoints_manifest": man, "tables": tables, "weights_root": tree["root"], "com": com, "rc": rc, "served": served,
            "values": values, "row": row, "reqs": reqs}


def hook(c: dict[str, Any], out_dir: Path, run: str = "served-experts") -> Path:
    """The row driver's window in ``out_dir``: the in-Commit hook at the Commit's C2 site (`c2_replay.serving_rows`, its request
    as the row stage writes it, the tables' registration and key in it) over the committer and the served weights, once the run
    root is final: ``served``."""
    from verity_vllm.pipeline import c2_replay as C2

    req = out_dir.with_name(out_dir.name + ".request.json")
    req.write_text(json.dumps({"dir": str(out_dir), "partition": str(c["partition"]), "run": run,
                               "committer": {"source": "served_experts"}, "tables": c["tables"]}))
    args = SimpleNamespace(program_dir=str(c["row_dir"] / "build_request"), programs_root=None, program_digest=c["inst"]["digest"],
                           label=LABEL, serving_rows=str(req))
    C2.serving_rows(SimpleNamespace(args=args, out=c["out"], pair=1, com=c["com"], rc=c["rc"], reqs=c["reqs"], sampling={"max_tokens": 1},
                                    row=c["row"], manifest=None, weights_of=lambda prefix: CPU.provider(c["served"], prefix, set())))
    if (out_dir / "error.txt").exists():
        raise RuntimeError(f"the in-Commit hook failed: {(out_dir / 'error.txt').read_text()[-2000:]}")
    return out_dir


def serve(c: dict[str, Any], d: Path) -> dict[str, Any]:
    """The same committed step sealed into the row dir ``d`` as the serve loop seals a closed window
    (`serve_commit.CommitWindows`: the window's requests and request rows from the rows its committer kept, the registered
    tensors' bytes its bundle carries, then `serve_seal.seal_window`): its one request holds the step's M token rows, under
    the name its body's ``user`` gives it. -> `seal_window`'s record."""
    from verity_vllm.pipeline import serve_commit as sc
    from verity_vllm.pipeline import serve_seal as ss

    leds = {REQUEST: sc.LiveRequest("cmpl-1", "e1", list(range(M)), {"params": {"max_tokens": 1}}, REQUEST, streamed=[0])}
    reqs, rows = ss.window_requests({0: [("runner-a", 0, M, 0, M)]}, {"runner-a": REQUEST}.get, leds.get)
    com, rc = c["com"], c["rc"]
    return ss.seal_window(com, rc, d, rec={"window": 0, "run_id": "served-experts/w0", "steps": len(rc.steps), "passed": True},
                          reqs=reqs, request_rows=rows, tokens={k: v.streamed for k, v in leds.items()}, sampling={}, label=LABEL,
                          code="test", engine_of_record={"vllm": "test"}, extras=ss.extras_of(com, rc))


#: what the registrations of two windows that commit one thing agree on, beside their ports' domains (`same_commitment`)
SAME = ("format", "program", "query", "law", "partition", "population", "scheme")


def same_commitment(a: Path, b: Path) -> None:
    """Windows ``a`` and ``b`` commit one thing, or an AssertionError naming what differs: one population program, partition,
    law and port domains (which bind the run root), one circuit record and index, the same registered roots, and in every
    member the same committed rows on the ports its own file holds, among them each unit's opening of its table (the table's
    row, its salt and its path, the committer's under its key).  Each window salts its own rows, so the roots of its members'
    trees are its own."""
    a, b = Path(a), Path(b)
    (ra, wa), (rb, wb) = (tuple(json.loads((d / f).read_text()) for f in ("registration.json", "window.json")) for d in (a, b))
    for k in SAME:
        assert ra[k] == rb[k], f"the registrations' {k}: {ra[k]!r} and {rb[k]!r}"
    assert [(x["port"], x["domain"]) for x in ra["roots"]] == [(x["port"], x["domain"]) for x in rb["roots"]], "the ports' domains"
    for k in ("registered_roots", "circuit"):
        assert ra["window"][k] == rb["window"][k], f"the windows' {k}"
    for name in ("partition.json", "circuit.json", "index.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name

    def shape(w: dict) -> list:
        return [(m["name"], m["template"], m["n"], m["ports"], m.get("value_rows"), m.get("registered")) for m in w["members"]]

    assert shape(wa) == shape(wb), "the members"
    ba, bb = CS.Bodies(a, wa), CS.Bodies(b, wb)
    for k, m in enumerate(wa["members"]):
        oa, ob = ba.own(k), bb.own(k)
        assert oa and oa.keys() == ob.keys(), f"member {m['name']}: the ports its own file holds"
        for q in oa:
            assert np.array_equal(oa[q][0], ob[q][0]), f"member {m['name']}: the rows of port {q}"


def flipped(window: Path, out: Path) -> Path:
    """A copy of the window in ``window`` at ``out`` whose first member's first committed row differs in one bit: what
    `same_commitment` must refuse."""
    shutil.copytree(window, out)
    body = out / "m0-private.bin"
    raw = bytearray(body.read_bytes())
    raw[0] ^= 1
    body.write_bytes(bytes(raw))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--same", nargs=2, metavar=("A", "B"), help="hold the windows in A and B to one commitment, and write nothing")
    a = ap.parse_args(argv)
    if a.same:
        same_commitment(*a.same)
        return 0
    if not a.out:
        ap.error("give OUT, or --same A B")
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    c = commit(root)
    s = serve(c, root / "served")
    print(json.dumps({"commit": str(c["out"]), "bundle": str(RB.path(c["out"], 1)), "row_dir": str(c["row_dir"]),
                      "program_dir": str(c["row_dir"] / "build_request"), "program_digest": c["inst"]["digest"],
                      "partition": str(c["partition"]), "run_root": c["run_root"], "weights_root": c["weights_root"],
                      "checkpoint": str(c["checkpoint"]), "checkpoints_manifest": str(c["checkpoints_manifest"]), "tables": c["tables"],
                      "hook": str(hook(c, root / "hook")), "request": REQUEST, "units": UNITS,
                      "served": {k: s[k] for k in ("row_dir", "bundle", "manifest_sha256")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
