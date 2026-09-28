"""A-GKR on the GPU as a Table 2 cell: the frozen instances' operands -> the device witness generator (gpu/v2/witness.py)
-> the prover (gpu/prover.py) -> proof.bin; the Python verifier and the independent Rust verifier (archive/gkr/verifier)
on every recorded proof -> $RESEARCH_RUN_DIR/result.json (research/result/v0.1, bench-result/v1, the campaign contract
verity_numerical.bench.contract).

    research run --on vy-agkr-a100 --project verity --source . --cwd source/backends/gkr --tool a_gpu_prove --scratch triton \
        -- /workspace/venv312/bin/python bench_result.py /workspace/agkr-table/bb/stmt --instances /workspace/bench-instances/v1 \
        --vus 4096 --reps 3 --warmup 1 --verifier /workspace/bin/verity-gkr-verify --threads 15

DIR holds the statement's Params-only files of a checker-v2 export (circuit.txt, epilogue.txt, chain.txt, manifest.json;
`python -m verity_numerical.gkr_export.vu export --field babybear --variant v2`); the witness is never read from it.  The
operands and public words come from --instances after every file of the tier is checked against the committed frozen
manifest (--manifest, default fixtures/bench-instances/v1/manifest.json of the source tree: its sha256 is the
`instances.manifest_sha256` Table 2 requires).  The run directory gets `statement/` (DIR's four files + public.bin written
from the frozen words) and `proofs/rep<i>.bin`: `verity-gkr-verify verify --dir statement --proof proofs/rep0.bin --vus 4096
--relation bf16-ampere`.  The verifier refuses circuit files that are not a pinned circuit set of the relation the result
claims (`archive/gkr/verifier/pins.txt`), so a rep counts as verified only with `circuit_pinned` in its verdict.

Per timed rep (wall clock; the device is synchronised at every boundary; the buckets sum to t.total up to the Python
between the timers, which tables.phases shows as Other):

    t.witness              H2D of the operands + the witness generator (chain, unit rows, limb or packed-word epilogue) + wire evaluation
    t.encoding_commitment  x assembly + RS encode (SIMT NTT) + SHA-512 Merkle tree
    t.lookup               LogUp multiplicities + transcript start + every table's LogUp GKR
    t.arithmetic           the circuit GKR layers + the Ligero functional (input / lookup / chain claims) + w, q (INT8 GEMMs)
    t.serialization        the opened columns + Merkle paths + proof.to_bytes()
    t.zk_additional        0 (non-ZK)

B-Ligero books its Ligero tests as t.arithmetic and its column openings as t.serialization (archive/direct/ligero/vu.py);
this is the same split.  The buckets reported are the median rep's (by t.total), so they sum to its t.total.  Soundness is
computed live on the instance proved (archive/gkr/soundness.py -> security.accounting.account_candidate_a).  `--path`
selects the kernel path by the prover's own environment switches: graphed (default) | packed | torch.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

FIELD = "babybear6"          # F_p[X]/(X^6 - 22), p = 2^31 - 2^27 + 1 (gpu/field.py)
TIER = "vu-k1536"
AMPERE_MODEL = "ampere_bf16_m16n8k16"   # the frozen vu-k1536 tier's model (Params REAL)
AMPERE_RELATION = "bf16-ampere"         # its relation: the name its circuit files are pinned under in the verifier
K_VU = 1536
BACKEND = "verity-gkr gpu (archive/gkr/gpu: checker v2 + LogUp + Ligero over BabyBear^6, SHA-512 Merkle; torch/Triton)"
STATEMENT_FILES = ("circuit.txt", "epilogue.txt", "chain.txt", "manifest.json")
BIND_FILES = ("bind.txt", "instances.txt", "x.bin", "w.bin")


def _m(name: str, value, unit: str) -> dict:
    return {"name": name, "value": value, "unit": unit}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commit() -> str | None:
    """git when there is one; else `research run --on --source` ships `git archive HEAD` to `<root>/src/<commit sha>/`,
    so an ancestor directory's name is the commit; else `RESEARCH_SOURCE_COMMIT`."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        if sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    env = os.environ.get("RESEARCH_SOURCE_COMMIT")
    if env:
        return env
    for p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        if len(p.name) == 40 and all(c in "0123456789abcdef" for c in p.name):
            return p.name
    return None


def load_frozen(root: Path, manifest: Path, tier: str, lo: int, n: int):
    """``(x, w, y, check)``: the tier's operands ``(n, K)`` uint16 and words ``(n,)`` uint16 for ids ``[lo, lo + n)``, after
    every file of the tier under ``root`` matched the sha256 the frozen ``manifest`` records for it."""
    import numpy as np
    from verity_numerical.bench import instances as I

    man = json.loads(manifest.read_text())
    files = man["tiers"][tier]["files"]
    check = {}
    for name, ent in files.items():
        got = _sha256(root / name)
        if got != ent["sha256"]:
            raise SystemExit(f"bench_result.py: {root / name} sha256 {got} != the frozen manifest's {ent['sha256']}")
        check[name] = got
    td = I.load(tier, root)
    if lo + n > td.n:
        raise SystemExit(f"bench_result.py: {tier} has {td.n} ids, [{lo}, {lo + n}) requested")
    x = np.frombuffer(td.x, dtype=np.uint16).reshape(td.n, td.K)[lo:lo + n]
    w = np.frombuffer(td.w, dtype=np.uint16).reshape(td.n, td.K)[lo:lo + n]
    y = np.frombuffer(td.y, dtype=np.uint16)[lo:lo + n]
    return x, w, y, check


def load_relation(name: str, src_root: Path, lo: int, n: int, procs: int):
    """``(x, w, y, rel)`` for a relation whose frozen set is a recipe (``archive/direct/ligero/relations.py``, e.g.
    ``bf16-hopper``, ``fp8-ada``, ``fp8-hopper``): VUs ``[lo, lo + n)`` of ``relchain.instances`` (the relation's finite
    operand words from ``(seed, index)``, the accumulators recorded by its model), ``y`` the chain-end public word (the
    model's final FP32 accumulator cast by the relation's epilogue: BF16 uint16, or the FP8 relations' 22-bit packing
    as uint32)."""
    import numpy as np

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from archive.direct.ligero import relchain
    from archive.direct.ligero.relations import relation
    from gpu.v2.fp8 import relation_params

    rel = relation(name)
    p, epilogue = relation_params(rel)
    y_bits = 22 if epilogue == "fp8" else 16
    if rel.frozen_tier or rel.y_bits != y_bits or rel.k != sum(p.groups) or rel.steps * rel.k != K_VU:
        raise SystemExit(f"bench_result.py: {name} is not a recipe BF16- or FP8-output relation over K = {K_VU}")
    inst = relchain.instances(rel, lo + n, procs=procs)[lo:]
    x = np.array([v[0] for v in inst], dtype=np.uint16)
    w = np.array([v[1] for v in inst], dtype=np.uint16)
    y = np.array([rel.y_public(int(v[3])) for v in inst], dtype=np.uint16 if y_bits == 16 else np.uint32)
    return x, w, y, rel


def load_nvf4(lo: int, n: int, procs: int):
    """``(A, B, Y, inst_ref, inst_check)`` for ``fp4-nvf4``: VUs ``[lo, lo + n)`` of the frozen NVFP4 set
    (``archive/direct/ligero/fp4/chain.instances_fp4``: seed, edge families, model-recorded accumulators), ``A`` / ``B``
    (n, 24, 68) uint8 operand words, ``Y`` (n,) the final FP32 words."""
    from verity_numerical.bench import contract

    from archive.direct.ligero.fp4.chain import INSTANCE_SEED, VU_FAMILIES, instances_digest
    from gpu.nvf4.witness import load_instances

    A, B, Y = load_instances(lo, n, procs)
    digest = instances_digest(lo + n)
    ref = contract.instances_ref(contract.NVFP4_INSTANCES_TIER, lo, lo + n, digest, dataset=contract.NVFP4_INSTANCES_DATASET)
    check = {"recipe": f"fp4-nvf4 synthetic (edge families {', '.join(VU_FAMILIES)} cycling per VU; model-recorded accumulators)",
             "generator": "archive/direct/ligero/fp4/chain.py instances_fp4()", "seed": INSTANCE_SEED, "digest": digest}
    return A, B, Y, ref, check


def write_statement(d: Path, out: Path, y) -> None:
    """``out/``: DIR's Params-only files and ``public.bin`` holding the frozen words (``bb_export.write_rows`` layout);
    ``y`` (N,) one word per VU, or (N, m) for a ``chain.txt`` with a ``public`` line of m columns."""
    from gpu.bb_export import write_rows

    out.mkdir(parents=True, exist_ok=True)
    for name in STATEMENT_FILES:
        shutil.copyfile(d / name, out / name)
    write_rows([[int(c) for c in v] if hasattr(v, "__len__") else [int(v)] for v in y], out / "public.bin")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python archive/gkr/bench_result.py")
    ap.add_argument("dir", help="a checker-v2 export directory (only its Params-only statement files are read)")
    ap.add_argument("--instances", default=None, help="bench-instances/v1 data directory (the tier's files)")
    ap.add_argument("--relation", default=None,
                    help="a recipe relation of archive/direct/ligero/relations.py (bf16-hopper, fp8-ada, fp8-hopper) instead "
                         "of the frozen --tier")
    ap.add_argument("--input-set", default=None,
                    help="VUs [0, --vus) of an A100 BF16 input set (verity_numerical.bench.input_sets) at its K: DIR must be the "
                         "export at that K (gpu.v2.export circuits --k K); the relation is bf16-ampere-k<K> (verifier/pins.txt)")
    ap.add_argument("--input-set-art", default=None, help="the input set's art: id, recorded in the fingerprint")
    ap.add_argument("--manifest", default=None, help="the frozen manifest.json (default: <source>/fixtures/bench-instances/v1)")
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--vus", type=int, default=4096)
    ap.add_argument("--checker", choices=("v2",), default="v2")
    ap.add_argument("--path", choices=("graphed", "packed", "torch"), default="graphed")
    ap.add_argument("--field", default=FIELD)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--verifier", default=None, help="verity-gkr-verify binary (independent Rust verifier)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--target", type=int, default=-128)
    ap.add_argument("--bind", action="store_true",
                    help="the operands-bound statement (gpu/bind.py): bind.txt + the frozen x.bin / w.bin, checked by both verifiers")
    ap.add_argument("--commit", choices=("sha256", "blake3", "vllm-v1"), default=None,
                    help="the operands-committed statement (gpu/commit.py): per-VU row digests as public words, checked against "
                         "the frame-v3 (sha256, blake3) or vllm-v1 trees by the Rust verifier")
    args = ap.parse_args(argv)
    if args.bind and args.commit:
        raise SystemExit("bench_result.py: --bind and --commit are two statements; pick one")

    if args.path == "torch":
        os.environ["VERITY_GPU_TORCH_ONLY"] = "1"
    elif args.path == "packed":
        os.environ["VERITY_GPU_NO_GRAPHS"] = "1"
    if args.field != FIELD:
        print(f"bench_result.py: this prover's field is {FIELD}", file=sys.stderr)
        return 2
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or "backends/gkr/gpu/results/bench_result")
    run_id = os.environ.get("RESEARCH_RUN_ID") or f"local-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    proofs_dir = run_dir / "proofs"
    proofs_dir.mkdir(exist_ok=True)
    d = Path(args.dir)
    man = json.loads((d / "manifest.json").read_text())
    if man.get("variant") != "v2":
        print(f"bench_result.py: {d}/manifest.json is not a checker-v2 export", file=sys.stderr)
        return 2
    src_root = Path(__file__).resolve().parents[2]
    frozen = Path(args.manifest) if args.manifest else src_root / "fixtures" / "bench-instances" / "v1" / "manifest.json"

    import numpy as np
    import torch
    from verity_numerical.bench import contract

    from gpu import ligero, prover
    from gpu.circuit import layers, load_circuit
    from gpu.run import _dev, read_chain
    from gpu.v2.fp8 import is_fp8, relation_params
    from gpu.v2.witness import Generator, GraphedGenerator, Ops
    import soundness

    # ---- preprocessing (Params-only; not in the buckets) -------------------------------------------------------------
    tp0 = time.perf_counter()
    dev = _dev(args.device)
    nvf4 = args.relation == "fp4-nvf4"
    if nvf4:
        from verity.ml.circuits.targets.target import NVFP4_SM120

        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        from gpu.nvf4.circuit import MODEL as NVF4_MODEL
        A8, B8, Y32, inst_ref, inst_check = load_nvf4(args.lo, args.vus, args.threads)
        if man.get("model") != NVF4_MODEL or man.get("relation") != "fp4-nvf4":
            raise SystemExit(f"bench_result.py: {d} is not the fp4-nvf4 export (python -m gpu.nvf4.circuit export)")
        params, profile, model_name = None, NVFP4_SM120.name, NVF4_MODEL
    elif args.input_set:
        from gpu import commit as CM
        from verity_numerical.bench.input_sets import InputSet
        from verity_numerical.checker import REAL

        s = InputSet.open(args.input_set)
        if s.subcircuit.params.get("semantics") != "sm80.mma.m16n8k16.bf16" or args.lo:
            raise SystemExit(f"bench_result.py: --input-set takes ids [0, --vus) of an sm80 BF16 set, not {s.subcircuit.id} from {args.lo}")
        if man.get("model", AMPERE_MODEL) != AMPERE_MODEL:
            raise SystemExit(f"bench_result.py: {d} holds the circuits of {man['model']}, the set is {AMPERE_MODEL}'s")
        x, w, y, _iset, inst_ref = CM.input_set(s.path, args.vus)
        y = y.astype(np.uint16)
        inst_ref = dict(inst_ref, art=args.input_set_art)
        params, profile, model_name = REAL, contract.PROFILE, AMPERE_MODEL
        inst_check = {"input_set": s.name, "content_digest": s.content_digest, "source": s.source, "files_ok": not s.check_files()}
    elif args.relation:
        x, w, y, rel = load_relation(args.relation, src_root, args.lo, args.vus, args.threads)
        from archive.direct.ligero import relchain
        digest = relchain.instances_digest(rel, args.lo + args.vus)
        if man.get("model") != rel.model.name:
            raise SystemExit(f"bench_result.py: {d} holds the circuits of {man.get('model')}, {rel.name} needs {rel.model.name}")
        params, profile, model_name = relation_params(rel)[0], rel.target.name, rel.model.name
        inst_ref = contract.instances_ref(rel.instances_tier, args.lo, args.lo + args.vus, digest, dataset=rel.instances_dataset)
        inst_check = {"recipe": rel.instances_recipe, "generator": "archive/direct/ligero/relchain.py instances()",
                      "seed": rel.instance_seed, "digest": digest}
    else:
        if not args.instances:
            raise SystemExit("bench_result.py: --instances (the frozen tier's data directory) or --relation")
        if man.get("model", AMPERE_MODEL) != AMPERE_MODEL:
            raise SystemExit(f"bench_result.py: {d} holds the circuits of {man['model']}, the frozen {args.tier} is {AMPERE_MODEL}")
        x, w, y, files_check = load_frozen(Path(args.instances), frozen, args.tier, args.lo, args.vus)
        from verity_numerical.checker import REAL
        params, profile, model_name = REAL, contract.PROFILE, AMPERE_MODEL
        inst_ref = contract.instances_ref(args.tier, args.lo, args.lo + args.vus, _sha256(frozen))
        inst_check = {"manifest": str(frozen), "files_sha256_match": files_check, "data_root": str(Path(args.instances).resolve())}
    t_load = time.perf_counter() - tp0
    ucirc, ecirc = load_circuit(d / "circuit.txt"), load_circuit(d / "epilogue.txt")
    ulayers, elayers = layers(ucirc), layers(ecirc)
    steps = man["steps"]
    graphed = dev.type == "cuda" and not os.environ.get("VERITY_GPU_EAGER_WITNESS")
    ggen: list = []
    if nvf4:
        from gpu.nvf4.circuit import STEPS as NVF4_STEPS
        from gpu.nvf4.circuit import statement as nvf4_statement
        from gpu.nvf4.witness import GraphedNVF4Generator, NVF4Generator, public_words

        if steps != NVF4_STEPS or A8.shape[1] != steps:
            raise SystemExit(f"bench_result.py: DIR has {steps} steps, the NVFP4 VU has {NVF4_STEPS}")
        epi_name = "(s, t, f) epilogue: the public FP32 word as three components"
        params_src = "gpu/nvf4/circuit: B-Ligero's compile_fp4_unit on GkrCtx"
        y = public_words(torch.from_numpy(Y32)).numpy()                  # (N, 3): sign, exponent field, fraction
        public = [int(v) for v in y.reshape(-1)]
        ngen = NVF4Generator(nvf4_statement(), dev)
        if ngen.st.ue.text != (d / "circuit.txt").read_text() or ngen.st.ee.text != (d / "epilogue.txt").read_text():
            raise SystemExit(f"bench_result.py: {d}'s circuits are not this tree's gpu.nvf4.circuit export")
        n_vus = A8.shape[0]

        def make_graphed():
            return GraphedNVF4Generator(ngen, n_vus)

        def gen_rows(g):
            if g is not None:
                return g.run(A8, B8, Y32)
            return ngen.run(torch.from_numpy(A8).to(dev).long(), torch.from_numpy(B8).to(dev).long(), torch.from_numpy(Y32).to(dev))
    else:
        unit_k = sum(params.groups)
        if steps * unit_k != x.shape[1]:
            raise SystemExit(f"bench_result.py: DIR has {steps} steps of {unit_k}, the tier has K = {x.shape[1]}")
        fp8 = is_fp8(params)
        epi_name = "packed-word epilogue" if fp8 else "limb epilogue"
        params_src = "gpu/v2/fp8.adder_params: the adder-side view" if fp8 else "checker Params.from_model"
        public = [int(v) for v in y]
        ops = Ops(str(dev))
        gen = Generator(ops, params)
        # the operands as host int16 arrays of their uint16 bits, the FP8 packed words as int32 (the device copy happens
        # inside every timed rep)
        x16, w16 = (np.ascontiguousarray(a).view(np.int16).copy() for a in (x, w))
        y16 = np.ascontiguousarray(y).view(np.int32 if y.dtype == np.uint32 else np.int16).copy()
        n_vus = x16.shape[0]

        def make_graphed():
            return GraphedGenerator(gen, x16.shape[0], x16.shape[1])

        def gen_rows(g):
            if g is not None:
                return g.run(x16, w16, y16)
            return gen.run(ops.asarray(x), ops.asarray(w), ops.asarray(y))
    write_statement(d, run_dir / "statement", y)
    bind = bind_info = None
    commitment = commit_info = None
    chain_path, ecirc_s, elayers_s, public_s, epi_extra = d / man.get("chain_file", "chain.txt"), ecirc, elayers, public, None
    K = K_VU if nvf4 else int(x.shape[1])
    if args.input_set and (args.bind or args.commit):
        raise SystemExit("bench_result.py: --input-set runs the bare statement (route (a), tools/cell.py, commits an input set's operands)")
    base = args.relation or (AMPERE_RELATION if K == K_VU else f"{AMPERE_RELATION}-k{K}")
    relation = base + ("+bound" if args.bind else "")   # gpu.bind.BOUND
    if args.commit:
        from gpu import commit as CM
        from gpu.bb_export import write_rows
        from gpu.circuit import parse_circuit

        sd = run_dir / "statement"
        base = args.relation or AMPERE_RELATION
        ox, ow = (A8.reshape(n_vus, -1), B8.reshape(n_vus, -1)) if nvf4 else (x, w)
        yw = 4 if nvf4 or y.dtype == np.uint32 else 2
        tc0 = time.perf_counter()
        iset = CM.instance_set(base, args.lo, n_vus, src_root, frozen, nvf4_k=ox.shape[1] if nvf4 else None)
        commitment = CM.commit(args.commit, base, iset, args.lo, ox, ow, np.asarray(public, dtype=np.int64).reshape(n_vus, -1), yw)
        t_commit_serving = time.perf_counter() - tc0
        relation = commitment.relation
        (sd / "commitment.txt").write_text(commitment.text())
        etext = CM.extend_epilogue((d / "epilogue.txt").read_text())
        (sd / "epilogue.txt").write_text(etext)
        (sd / "chain.txt").write_text(CM.extend_chain(chain_path.read_text()))
        pub2 = commitment.public(np.asarray(public, dtype=np.int64))
        write_rows(pub2.tolist(), sd / "public.bin")
        chain_path, public_s = sd / "chain.txt", [int(v) for v in pub2.reshape(-1)]
        ecirc_s = parse_circuit(etext)
        elayers_s = layers(ecirc_s)
        epi_extra = torch.from_numpy(commitment.limb_columns()).to(dev)
        commit_info = {**commitment.info(), "serving_commit_seconds": t_commit_serving, "digest_columns": len(CM.DIGEST_COLS),
                       "circuit_binds_digests": False,
                       "spec": "gpu/commit.py: per VU the x row / W column row digests as 16-bit limbs after the y words; the Rust "
                               f"verifier rebuilds the {commitment.scheme} trees and compares the roots (commitments.rs, "
                               "vllm_v1.rs).  SCAFFOLD: no hash layer constrains the digest columns yet"}
    if args.bind:
        from gpu import bind as BD
        from gpu.field import P as FP

        sd = run_dir / "statement"
        (sd / "bind.txt").write_text(BD.spec_text(BD.derive(ucirc, man)))
        spec = BD.read_spec(sd / "bind.txt", ucirc)
        if nvf4:
            ox, ow, xdt, ydt = A8.reshape(n_vus, -1), B8.reshape(n_vus, -1), "u8", "u32"
        else:
            ox, ow, xdt, ydt = x, w, "u16", ("u32" if y.dtype == np.uint32 else "u16")
        bind_info = BD.write_instances(sd, relation, args.lo, ox, ow, np.asarray(public, dtype=np.int64), xdt, ydt)
        bind = prover.Bind(spec.cols, BD.digests(sd), torch.from_numpy(BD.values(spec, ox, ow, steps, FP)).to(dev))
        bind_info.update({"bind_sha256": hashlib.sha256(spec.text.encode()).hexdigest(), "bound_columns": len(spec.cols),
                          "constraints": n_vus * steps * len(spec.cols),
                          "spec": "gpu/bind.py: X[(v,i), col] = sum coef * ARR[v, stride*i + off] per bind.txt line; "
                                  "the verifier reads x.bin / w.bin and checks them and the public words against its pinned table"})
    torch.cuda.synchronize(dev)
    t_prep = time.perf_counter() - tp0
    encoder = "SIMT (archive/direct/encode)" if ligero.simt_encoder(ligero.DEFAULT.k, ligero.DEFAULT.n) is not None else "torch NTT"

    def one_rep():
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        if graphed and not ggen:                                 # captured in the (untimed) warm-up rep
            ggen.append(make_graphed())
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        rows = gen_rows(ggen[0] if graphed else None)
        epi_cols = rows.epilogue if epi_extra is None else torch.cat([rows.epilogue, epi_extra], 1)
        bad = int(rows.bad.sum())
        t1 = time.perf_counter()
        if bad:
            raise SystemExit(f"bench_result.py: {bad} VUs rejected at witness generation (the frozen positives are all valid)")
        gen_peak = torch.cuda.max_memory_allocated(dev)
        chain = read_chain(chain_path, ucirc, ecirc_s, steps, public_s)
        chain.bind = bind
        chain.commit_digest = None if commitment is None else hashlib.sha256(commitment.text().encode()).digest()
        inst = prover.Instance([prover.Segment("unit", ucirc, ulayers, rows.units, ucirc.hash),
                                prover.Segment("epilogue", ecirc_s, elayers_s, epi_cols, ecirc_s.hash)], chain)
        proof, st = prover.prove(inst, True)
        torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        data = proof.to_bytes()
        t3 = time.perf_counter()
        return inst, proof, st, data, {"witness_gen": t1 - t0, "prove": t2 - t1, "bytes": t3 - t2, "total": t3 - t0,
                                       "gen_chain": rows.seconds.get("chain"), "peak": max(gen_peak, st.peak_mem)}

    def verify_py(inst, proof):
        t = time.perf_counter()
        try:
            prover.verify(inst, proof, True)
            err = None
        except prover.VerifyError as e:
            err = str(e)
        torch.cuda.synchronize(dev)
        return err, time.perf_counter() - t

    warm = []
    for _ in range(args.warmup):
        tw = time.perf_counter()
        inst, proof, _st, _data, tm = one_rep()
        err, _ = verify_py(inst, proof)
        warm.append(time.perf_counter() - tw)
        print(json.dumps({"warmup": True, "seconds": warm[-1], "total": tm["total"], "verified": err is None, "error": err}), flush=True)
        del inst, proof, _st, _data
    # The preprocessing state (parsed circuits, tables, captured graphs: ~3.4e5 objects) leaves the collector's
    # generations, so a full collection cannot land inside a timed rep (one did: +0.10 s in the opening).
    gc.collect()
    gc.freeze()
    rows = []
    budget = None
    for rep in range(args.reps):
        inst, proof, st, data, tm = one_rep()
        err, tv = verify_py(inst, proof)
        if budget is None:
            budget = soundness.summary(soundness.budget(inst, st.ligero_rows, st.linkage_constraints, K=K,
                                                        bind_constraints=st.bind_constraints))
        pf = proofs_dir / f"rep{rep}.bin"
        pf.write_bytes(data)
        row = {"rep": rep, "vus": inst.vus(), "units": inst.segs[0].units, "t_total": tm["total"], "witness_gen": tm["witness_gen"],
               "gen_chain": tm["gen_chain"], "prove": tm["prove"], "to_bytes": tm["bytes"],
               "t_witness_wires": st.t_witness_wires, "t_mults": st.t_mults, "t_commit": st.t_commit, "t_lookup": st.t_lookup,
               "t_arith": st.t_arith, "t_open": st.t_open, "t_open_acc": st.t_open_acc, "t_open_wq": st.t_open_wq,
               "t_open_cols": st.t_open_cols, "python_verifier_s": tv, "python_verified": err is None, "error": err,
               "proof_bytes": len(data), "proof_sha256": hashlib.sha256(data).hexdigest(), "proof_file": f"proofs/rep{rep}.bin",
               "slots": st.slots, "sequential_depth": st.sequential_depth(True), "depth_circuit": st.depth_circuit,
               "depth_lookup_max": st.depth_lookup_max, "committed_elements": st.committed_elements, "ligero_rows": st.ligero_rows,
               "witness_elements": sum(int(s.cols.numel()) for s in inst.segs), "peak_mem_bytes": tm["peak"]}
        row["buckets"] = {
            "t.witness": tm["witness_gen"] + st.t_witness_wires,
            "t.encoding_commitment": st.t_commit,
            "t.arithmetic": st.t_arith + st.t_open_acc + st.t_open_wq,
            "t.lookup": st.t_mults + st.t_lookup,
            "t.zk_additional": 0.0,
            "t.serialization": st.t_open_cols + tm["bytes"],
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "buckets"}), flush=True)
        del inst, proof, st, data

    # ---- the independent verifier on every recorded proof ------------------------------------------------------------
    vbin = Path(args.verifier) if args.verifier else None
    verdicts = []
    for row in rows:
        if vbin is None or not vbin.is_file():
            print(f"bench_result.py: no independent verifier at {args.verifier!r}", file=sys.stderr)
            break
        out_json = run_dir / f"verify_rep{row['rep']}.json"
        cmd = [str(vbin), "verify", "--dir", str(run_dir / "statement"), "--proof", str(run_dir / row["proof_file"]),
               "--vus", str(row["vus"]), "--threads", str(args.threads), "--json", str(out_json)]
        # no circuit set is pinned for R+sha256 / R+blake3 until hash layers bind the digest columns: the scaffold's
        # verdict is the tree check under --allow-any-circuit, never an accepted (pinned) result
        cmd += (["--allow-any-circuit", "--require-commitment", "--allow-unpinned-commitment"] if commitment is not None
                else ["--relation", relation])
        t1 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t1
        try:
            doc = json.loads(out_json.read_text()) if out_json.is_file() else None
        except json.JSONDecodeError:
            doc = None
        # a binary built before the pins ignores --relation and verifies whatever circuit files it is given
        pinned = (doc or {}).get("circuit_pinned") is True
        accepted = (bool(doc and doc.get("accepted")) and pinned and r.returncode == 0
                    and (bind is None or (doc or {}).get("operands_bound") is True))
        error = (doc or {}).get("error") or (None if doc is None or pinned else
                                             f"scaffold: no circuit set is pinned for {relation} until hash layers bind the "
                                             f"digest columns" if commitment is not None else
                                             f"{vbin} does not pin circuit files (built before archive/gkr/verifier/pins.txt): rebuild it")
        cdoc = (doc or {}).get("commitment") or {}
        verdicts.append({"rep": row["rep"], "proof_sha256": row["proof_sha256"], "accepted": accepted, "rc": r.returncode,
                         "commitment_checked": bool(doc and doc.get("accepted") and doc.get("operands_committed")) if commitment else None,
                         "commitment_pinned": cdoc.get("pinned") if commitment else None,
                         "commitment_roots_match": (all(cdoc.get(f"root_{n}") == commitment.roots[n].hex() for n in "aby")
                                                    if commitment else None),
                         "error": error, "verify_seconds": (doc or {}).get("verify_seconds"), "process_wall": wall,
                         "relation": (doc or {}).get("relation"), "circuit_pinned": pinned,
                         "threads": (doc or {}).get("threads"), "cpu": (doc or {}).get("cpu"), "json": out_json.name,
                         "command": " ".join(cmd), "stderr_tail": r.stderr[-300:]})
        print(f"verify rep{row['rep']}: {'accept' if accepted else 'REJECT'} {wall:.3f}s ({error})"
              + (f"; commitment scaffold: checked {verdicts[-1]['commitment_checked']}, roots match "
                 f"{verdicts[-1]['commitment_roots_match']}, pinned {verdicts[-1]['commitment_pinned']}" if commitment else ""), flush=True)
    vsha = _sha256(vbin) if vbin and vbin.is_file() else None
    (run_dir / "verify_independent.json").write_text(json.dumps({"verifier": str(vbin), "verifier_sha256": vsha, "threads": args.threads,
                                                                 "items": verdicts}, indent=1))
    (run_dir / "prove_reps.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(dev), "path": args.path, "checker": "v2",
                                                         "encoder": encoder, "runs": rows, "warmup_seconds": warm}, indent=1))

    # ---- the envelope ---------------------------------------------------------------------------------------------------
    n_acc = sum(1 for v in verdicts if v["accepted"])
    rust_ok = bool(verdicts) and n_acc == len(rows)
    python_ok = all(r["python_verified"] for r in rows)
    same_proof = len({r["proof_sha256"] for r in rows}) == 1
    med = sorted(rows, key=lambda r: r["t_total"])[(len(rows) - 1) // 2]
    t_total = med["t_total"]
    Bn = med["vus"]
    rust_reps = [v["verify_seconds"] if v["verify_seconds"] is not None else v["process_wall"] for v in verdicts]
    rust_med = statistics.median(rust_reps) if rust_reps else None
    ach = budget["achieved_log2"]
    hw = contract.probe_hardware()
    import triton

    try:
        import cupy
        cupy_v = cupy.__version__
    except ImportError:
        cupy_v = None
    sw = {"backend": {"name": BACKEND, "commit": source_commit(),
                      "version": f"checker v2 ({epi_name}), proof.bin v2" + ("; operands bound (bind.txt, x.bin, w.bin)" if bind else ""),
                      "model": f"{model_name} ({params_src})", "relation": base if args.input_set else args.relation or args.tier,
                      "path": args.path, "encoder": encoder, "opening": ligero.opening_path(dev),
                      "witness": ("gpu/nvf4/witness.py (device generator: B-Ligero's fp4/hints_device chain + the unit's program)"
                                  if nvf4 else "gpu/v2/witness.py (device generator; byte-exact with the CPU export)")
                                 + ("; replayed as one CUDA graph captured in the warm-up" if graphed else "")},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "triton": triton.__version__, "cupy": cupy_v, "numpy": np.__version__}}
    security = {"target": args.target, "achieved_log2": ach, "achieved_bits": -ach, "field": "BabyBear^6 = F_p[X]/(X^6 - 22)",
                "trace_field": "BabyBear", "challenge_coords": 6,
                "ligero": {"n": ligero.DEFAULT.n, "k": ligero.DEFAULT.k, "t": ligero.DEFAULT.t, "rows": med["ligero_rows"], "zk": False,
                           "row_combination": "powers of one challenge (BCIKS20 curve bound)"},
                "hash": ligero.MERKLE_HASH, "mode": "interactive", "fiat_shamir_queries_log2": None,
                "bound_model": budget["model"], "hash_assumption": "collision resistance of SHA-512 (Merkle, q^2/2^512 at q = 2^64)",
                "zk_statement": "none (no masking)", "by_component_log2": budget["by_component_log2"],
                "dominant_term": budget["dominant_term"], "source": budget["source"], "union_over": 1}
    fp = contract.fingerprint(
        profile=profile, domain=contract.DOMAIN, K=K, B=Bn, N_subbatches=1,
        proof_class="NON_ZK_PROOF_DIAGNOSTIC", authentication="excluded", zk_mode=contract.ZK_NONE,
        security=security, hardware=hw, software=sw,
        preprocessing=(f"circuit compilation (unit + {epi_name} circuits, GKR layers, chain spec: functions of Params only), "
                       f"the witness generator's dense tables on the device, "
                       + (f"generating the {args.relation} recipe instances ({t_load:.2f}s, relchain.instances on "
                          f"{args.threads} processes)" if args.relation else
                          f"loading the frozen operands from disk ({t_load:.2f}s incl. the sha256 check of every tier file)")
                       + ("; the operands-bound statement files (bind.txt, instances.txt, x.bin, w.bin, their digests) and "
                          "the verifier's bound values" if bind else "")
                       + f"; {t_prep:.2f}s in all.  Triton / CUDA-graph / "
                       f"NVRTC compilation happens in {args.warmup} untimed warm-up rep(s) ({', '.join('%.1fs' % v for v in warm)}), "
                       f"then gc.freeze() of the preprocessing objects"),
        retention="retained",
        instances=inst_ref,
        scope="vu", candidate="A-GKR", checker="v2", field=FIELD,
        instances_check=inst_check,
        witness_generation=f"device generator ({'gpu/nvf4/witness.py' if nvf4 else 'gpu/v2/witness.py'}) inside t.witness, "
                           f"from the frozen operands (H2D included)",
        omitted="ZK masking",
        steps=steps, units=med["units"], reps=args.reps, warmup=args.warmup,
        verifier={"name": "verity-gkr-verify (archive/gkr/verifier, Rust, shares no code with the prover)", "path": str(vbin),
                  "sha256": vsha, "threads": args.threads, "cpu": next((v["cpu"] for v in verdicts if v.get("cpu")), None),
                  "relation": relation, "circuit_pinned": bool(verdicts) and all(v["circuit_pinned"] for v in verdicts)},
        python_verifier="gpu.prover.verify (shares field / transcript / LogUp helpers with the prover)",
        note=f"one proof of all {Bn} VUs ({steps} chained units each + the {epi_name})"
             + (f"; operands bound: every unit's operand columns equal the frozen x, W the verifier reads "
                f"({bind_info['constraints']} constraints)" if bind else "")
             + (f"; operands committed ({commit_info['relation']}, SCAFFOLD): each VU's x / W row digests are public words the "
                f"verifier checks against the {commit_info['scheme']} roots, but no circuit layer constrains them yet (serving-side commit "
                f"{commit_info['serving_commit_seconds']:.2f}s, outside t.total)" if commitment else "")
             + f"; buckets of the median rep (rep {med['rep']}) by t.total",
        **({"variant": "A-GKR, operands bound", "operand_binding": bind_info} if bind else {}),
        **({"variant": "A-GKR, operands committed (scaffold: digests not bound in circuit)", "operand_commitment": commit_info}
           if commitment else {}),
    )
    b = med["buckets"]
    meas = [contract.measurement(name, b[name]) for name in contract.BUCKET_NAMES]
    meas.append(contract.measurement("t.total", t_total))
    if rust_med is not None:
        meas.append(contract.measurement("verifier.seconds", rust_med))
    meas += [
        contract.measurement("transcript.bytes", med["proof_bytes"]),
        contract.measurement("rounds.protocol", med["sequential_depth"]),
        contract.measurement("rounds.sequential_depth", med["sequential_depth"]),
        contract.measurement("mem.peak_device_bytes", max(r["peak_mem_bytes"] for r in rows)),
        contract.measurement("census.witness_elements", med["witness_elements"]),
        contract.measurement("census.committed_elements", med["committed_elements"]),
        contract.measurement("census.bytes_encoded", med["ligero_rows"] * ligero.DEFAULT.n * 4),
        contract.measurement("fixed.setup_seconds", t_prep),
        contract.measurement("vu.count", Bn),
        contract.measurement("vu.seconds_per_vu", t_total / Bn),
        contract.measurement("vu.per_second", Bn / t_total),
        *contract.rate_measurements(t_total, Bn, K, profile),
        _m("t_total_min", min(r["t_total"] for r in rows), "s"),
        _m("t_total_max", max(r["t_total"] for r in rows), "s"),
        _m("verify_python_seconds", statistics.median(r["python_verifier_s"] for r in rows), "s"),
        _m("proofs_verified_independent", n_acc, "count"),
        _m("proofs_total", len(rows), "count"),
        _m("soundness_bits", -ach, "bits"),
        _m("warmup_seconds", warm[0] if warm else 0.0, "s"),
    ]
    if commitment is not None:
        meas.append(_m("commit.serving_seconds", commit_info["serving_commit_seconds"], "s"))
    for key in ("witness_gen", "gen_chain", "t_witness_wires", "t_mults", "t_commit", "t_lookup", "t_arith", "t_open_acc", "t_open_wq",
                "t_open_cols", "to_bytes"):
        if med[key] is not None:
            meas.append(_m(f"split.{key}_seconds", med[key], "s"))
    ok = python_ok and rust_ok and same_proof
    detail = (f"python: {sum(r['python_verified'] for r in rows)}/{len(rows)}; rust: {n_acc}/{len(verdicts)} accepted "
              f"(--relation {relation})"
              + (f"; commitment scaffold (--allow-any-circuit): {sum(bool(v['commitment_checked']) for v in verdicts)}/{len(verdicts)} "
                 f"tree checks passed, roots match {all(v['commitment_roots_match'] for v in verdicts)}" if commitment else "")
              + f"{'' if same_proof else '; the reps produced DIFFERENT proofs'}; reps t.total "
              f"{['%.3f' % r['t_total'] for r in rows]} s, rust verify {['%.3f' % v for v in rust_reps]} s")
    result = {"schema": contract.RESULT_SCHEMA, "run_id": run_id, "kind": "bench-result/v1", "workload_fingerprint": fp,
              "validation": {"status": "passed" if ok else "failed",
                             "checker": "archive/gkr/verifier `verify --relation` (independent Rust; circuit files pinned) on every "
                                        "recorded rep's proof + gpu.prover.verify per rep",
                             "detail": detail,
                             "evidence": {"reps": [{"t_total": r["t_total"], "buckets": r["buckets"], "python_verifier": r["python_verifier_s"],
                                                    "proof_sha256": r["proof_sha256"]} for r in rows],
                                          "rust": [{k: v[k] for k in ("rep", "accepted", "circuit_pinned", "verify_seconds", "error", "command")}
                                                   for v in verdicts],
                                          "soundness": budget}},
              "measurements": meas,
              "measurement_files": ["prove_reps.json", "verify_independent.json"] + [v["json"] for v in verdicts],
              "artifacts": [r["proof_file"] for r in rows] + [f"statement/{n}" for n in (*STATEMENT_FILES, "public.bin")]
                           + ([f"statement/{n}" for n in BIND_FILES] if bind else [])
                           + (["statement/commitment.txt"] if commitment else [])}
    problems = contract.validate(result)
    result["contract_problems"] = problems
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, default=str))
    os.replace(tmp, run_dir / "result.json")
    print(json.dumps({"t_total": t_total, "buckets": b, "verify_wall": rust_med, "proof_bytes": med["proof_bytes"],
                      "soundness_log2": ach, "rust_accepted": f"{n_acc}/{len(rows)}", "status": result["validation"]["status"],
                      "contract_problems": problems}, indent=1))
    return 0 if ok and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
