"""Frozen input sets on silicon: execute every instance of a target's benchmark set on the target's anchor GPU, exactly as
a real GEMM would -- the K-steps chained on the tensor core with the FP32 accumulator carried between ``mma`` / ``wgmma``
calls in the target's program order, the target's epilogue applied to the last word -- and compare every hardware word with
the set's expected (model-computed) word: every intermediate accumulator where the set records them, and the final word.

    python benchmarks/tc_probe/instances_hw.py --instances bench-instances/v1/vu-k1536 --root /workspace/bench-instances/v1 --out $RESEARCH_RUN_DIR --sass-check
    python benchmarks/tc_probe/instances_hw.py --instances bench-instances/v1/tu-k16 --root ... --out ...
    python benchmarks/tc_probe/instances_hw.py --instances bench-instances/v1/vu-k1536-neg --out ...          # committed arrays: no --root needed
    python benchmarks/tc_probe/instances_hw.py --instances fp8-ada-synthetic/vu-k1536-fp8 --out ...          # generated from the seed

Input sets (``--instances DATASET/TIER``; the instruction defaults to the target's and may be overridden with ``--instruction``
to run the same words through another architecture's instruction of the same shape)::

    bench-instances/v1/vu-k1536        FIRST (sm80.mma.m16n8k16.bf16)   4,096 inputs x 96 steps from c_0 = +0, expected acc after every
                                                                         step (vu-k1536.acc.u32) and y = f32_to_bf16 RNE (vu-k1536.y.u16)
    bench-instances/v1/tu-k16          FIRST                             4,096 transitions (c, a[16], b[16]) -> out (tu-k16.out.u32)
    bench-instances/v1/vu-k1536-neg    FIRST                             52 claimed-WRONG or out-of-domain (reject) inputs: the harness must
                                                                         detect every one (hardware word != claimed, or the domain check)
    fp8-ada-synthetic/vu-k1536-fp8     FP8_ADA_MMA (sm89.mma.m16n8k32.e4m3)  4,096 inputs x 48 steps of uniform finite E4M3 bytes from
                                                                       seed 20260922 (archive.direct.ligero.fp8.chain.instances_fp8,
                                                                       regenerated here from the same recipe), accumulators recorded
                                                                       by ADA_E4M3_M16N8K32, no epilogue (the bare FP32 word)
    bench-instances-bf16-hopper/v1/vu-k1536-bf16-hopper   BF16_HOPPER (sm90.mma.m16n8k16.bf16)  4,096 inputs x 96 steps of random_finite_bf16
                                                                       words from seed 20260922, HOPPER_BF16_M16N8K16 accumulators,
                                                                       f32_to_bf16 epilogue (relchain.instances)
    bench-instances-fp8-hopper/v1/vu-k1536-fp8-hopper     FP8_HOPPER_WGMMA (sm90.wgmma.m64n8k32.e4m3)  4,096 inputs x 48 E4M3 steps from
                                                                       seed 20260922, HOPPER_E4M3_K32 accumulators, bare FP32 word

``bench-instances/v1`` is loaded from ``--root`` (the built arrays; ``python -m verity_numerical.bench.instances build``) after every
array's sha256 is checked against the pinned ``--manifest`` (default the repository's ``fixtures/bench-instances/v1/manifest.json``);
the manifest's own sha256 is the recorded ``manifest_sha256``.  The synthetic sets have no files: their ``manifest_sha256`` is
``instances_digest(n)`` (a digest of the recipe: relation, seed, n, K), as the chain benches record it.

One instance per tile at position ``(r % M, (r // M) % 8)`` with the rest of the tile zero (as ``tc_probe --replay-instances``): every
other output word must come back ``+0`` (``other_positions_nonzero``).  Step ``s`` of every instance is one launch; the hardware's own
D of step ``s`` is the C of step ``s + 1`` -- a mismatch propagates, as it would in a real chain, and is located at its first step.

Domain check (``verity.ml.circuits.targets.target``: finite operands and finite accumulators at every step; a saturation *inside the last*
transition is a valid output, D8): instances with a non-finite operand word or a non-finite hardware accumulator before the last step
are ``out of domain``.  Positive tiers must have none; the negative tier's ``reject`` rows must trip it (its ``wrong`` rows must
have ``hardware != claimed`` and ``hardware == correct``).

Outputs under ``--out``: ``probe_results.json``, ``capture_<tier>.npz`` (the hardware accumulator of every instance at every step,
the final words, the ids), ``sass_<kernel>_<arch>.txt`` with ``--sass-check``, and in a recorded run (``RESEARCH_RUN_ID`` set) a ``result.json`` of kind
``tc-evidence/v1`` whose ``tc_evidence`` is the meta contract: ``{instruction, target, instances_dataset, instances_tier, range,
manifest_sha256, elements, mismatches, first_mismatch | null, per_step_checked, sass, device, driver, cuda, model_agrees, ...}``.
``mismatches`` counts hardware words that differ from the reference words: the tier's expected (model-recorded) words, and on the
negative tier the *correct* word of every claimed-wrong row (``negative.claimed_words_contradicted`` counts hardware words that differ
from the *claimed* ones -- all of them, if the harness detects every wrong claim; ``negative.undetected`` is that tier's failure count).
``validation.status`` is ``passed`` iff a positive tier has 0 mismatches, 0 other-position words and 0 out-of-domain instances, or
the negative tier has 0 undetected rows; ``failed`` otherwise.  ``--repeat N`` runs the whole replay N times and counts words that
differed between repetitions (nondeterminism); ``--only-ids`` re-runs chosen instances alone (confirming a mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from verity_config.settings import SHARED

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tc_probe as P  # sibling module: kernels, formats, instruction table, SASS check, result envelope
from verity.ml.circuits.silicon import silicon as S
from verity.ml.circuits.silicon.cast import f32_to_bf16_word
from verity.ml.circuits.targets import target as T

REPO = Path(__file__).resolve().parents[2]
N = P.N
FIRST_MISMATCHES = 20
BENCH_INSTANCES = "bench-instances/v1"
FP8_ADA_DATASET_ID = "fp8-ada-synthetic"
#: exactly the ``instances.dataset`` string ``archive.direct.ligero.fp8.chain.bench_vu_fp8`` records in its fingerprint
FP8_ADA_DATASET = "fp8-ada synthetic (uniform finite E4M3 bytes, model-recorded accumulators)"
FP8_ADA_TIER = "vu-k1536-fp8"
SYNTHETIC_SEED = 20260922  # every seeded synthetic set of the ligero chain runners (fp8-ada, bf16-hopper, fp8-hopper)
K_VU = 1536
SYNTHETIC_N = 4096
FP8_ADA_SEED, FP8_ADA_N = SYNTHETIC_SEED, SYNTHETIC_N
EPILOGUE_BF16 = "f32_to_bf16"


# --- input sets ---------------------------------------------------------------------------------------------------------------------


@dataclass
class InstanceSet:
    """A tier of one dataset as tiles-to-be: ``a``/``b`` are ``(n, steps, k)`` operand words in the instruction's format, ``c0`` the
    ``(n,)`` incoming FP32 words of step 0, ``acc`` the ``(n, steps)`` expected accumulator after every step (``None`` when the set
    records none), ``y`` the ``(n,)`` expected final word after ``epilogue`` (``None``: the bare FP32 accumulator).  ``negative``
    carries the claimed-wrong tier's rows (``verdict`` wrong/reject, ``correct``)."""

    dataset: str
    tier: str
    instruction: str
    target: str
    model: str
    manifest_sha256: str
    source: str
    ids: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c0: np.ndarray
    acc: np.ndarray | None
    y: np.ndarray
    epilogue: str | None
    total_n: int
    negative: dict | None = None

    @property
    def n(self) -> int:
        return int(self.ids.size)

    @property
    def steps(self) -> int:
        return int(self.a.shape[1])

    @property
    def k(self) -> int:
        return int(self.a.shape[2])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slice(lo: int | None, hi: int | None, n: int) -> tuple[int, int]:
    lo, hi = (0 if lo is None else lo), (n if hi is None else hi)
    if not 0 <= lo < hi <= n:
        raise ValueError(f"range [{lo}, {hi}) outside [0, {n})")
    return lo, hi


def load_bench_instances(tier: str, root: Path | None, manifest_path: Path, lo: int | None = None, hi: int | None = None,
                         only_ids: list[int] | None = None) -> InstanceSet:
    """``bench-instances/v1`` tier from the built arrays under ``root`` (committed files may also be read from the manifest's own
    directory), every array digest-checked against the pinned manifest first."""

    from verity_numerical.bench import instances as I

    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"] != BENCH_INSTANCES or tier not in manifest["tiers"]:
        raise ValueError(f"{manifest_path} is not a {BENCH_INSTANCES} manifest with tier {tier!r}")
    files = manifest["tiers"][tier]["files"]
    root = Path(root) if root is not None else manifest_path.parent
    where: dict[str, Path] = {}
    bad = []
    for name, rec in files.items():
        cands = [root / name] + ([manifest_path.parent / name] if rec["committed"] else [])
        p = next((c for c in cands if c.is_file()), None)
        if p is None:
            raise FileNotFoundError(f"{name} not found under {root} (built arrays: python -m verity_numerical.bench.instances build --out {root})")
        if _sha256(p) != rec["sha256"]:
            bad.append(name)
        where[name] = p
    if bad:
        raise RuntimeError(f"{tier} arrays differ from the pinned manifest {manifest_path}: {bad}")
    if root != manifest_path.parent:  # the loader reads one directory: make sure that directory has the committed files too
        for name, p in where.items():
            if p.parent != root:
                (root / name).write_bytes(p.read_bytes())
    data = I.load(tier, root)
    n_total = data.n
    if only_ids is not None:
        ids = np.array(sorted(set(only_ids)), dtype=np.int64)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= n_total:
            raise ValueError(f"--only-ids outside [0, {n_total})")
    else:
        lo, hi = _slice(lo, hi, n_total)
        ids = np.arange(lo, hi, dtype=np.int64)
    common = {"dataset": manifest["dataset"], "instruction": T.FIRST.instruction, "target": T.FIRST.name, "model": manifest["target"]["pipeline"],
              "manifest_sha256": _sha256(manifest_path), "source": f"{root} (sha256 of every array == {manifest_path})", "ids": ids, "total_n": n_total}
    if tier == "tu-k16":
        k = I.STEP
        a = np.frombuffer(data.a.tobytes(), np.uint16).reshape(n_total, 1, k)[ids]
        b = np.frombuffer(data.b.tobytes(), np.uint16).reshape(n_total, 1, k)[ids]
        c = np.frombuffer(data.c.tobytes(), np.uint32)[ids]
        out = np.frombuffer(data.out.tobytes(), np.uint32)[ids]
        return InstanceSet(tier=tier, a=a, b=b, c0=c, acc=out[:, None].copy(), y=out.copy(), epilogue=None, **common)
    k, steps = I.STEP, I.STEPS
    x = np.frombuffer(data.x.tobytes(), np.uint16).reshape(n_total, steps, k)[ids]
    w = np.frombuffer(data.w.tobytes(), np.uint16).reshape(n_total, steps, k)[ids]
    y = np.frombuffer(data.y.tobytes(), np.uint16)[ids]
    c0 = np.zeros(ids.size, np.uint32)
    if tier == "vu-k1536":
        acc = np.frombuffer(data.acc.tobytes(), np.uint32).reshape(n_total, steps)[ids]
        return InstanceSet(tier=tier, a=x, b=w, c0=c0, acc=acc.copy(), y=y.copy(), epilogue=EPILOGUE_BF16, **common)
    # vu-k1536-neg: y = the claimed (wrong) word; the index carries verdict and the correct word (the reference for the compare)
    cols = data.columns
    rows = [dict(zip(cols, r)) for r in data.index]
    neg = {"columns": cols, "rows": [rows[int(i)] for i in ids]}
    return InstanceSet(tier=tier, a=x, b=w, c0=c0, acc=None, y=y.copy(), epilogue=EPILOGUE_BF16, negative=neg, **common)


def fp8_random_finite_operands(rng: np.random.Generator, shape) -> np.ndarray:
    """Uniform E4M3 bytes without the NaN codes (``archive.direct.ligero.fp8.witness.random_finite_operands``, verbatim)."""

    w = rng.integers(0, 256, size=shape, dtype=np.int64)
    nan = (w & 0x7F) == 0x7F
    w[nan] = w[nan] & ~0x08  # 0x7f -> 0x77 (the largest finite), keeps the sign
    return w


def bf16_random_finite_operands(rng: np.random.Generator, shape) -> np.ndarray:
    """Finite BF16 words with exponent fields in [96, 160), one in ten a signed zero, one in ten a subnormal
    (``archive.direct.ligero.relations.random_finite_bf16``, verbatim)."""

    sign = rng.integers(0, 2, size=shape, dtype=np.int64) << 15
    t = rng.integers(96, 160, size=shape, dtype=np.int64)
    f = rng.integers(0, 128, size=shape, dtype=np.int64)
    kind = rng.integers(0, 10, size=shape)
    w = sign | (t << 7) | f
    w = np.where(kind == 0, sign, w)
    w = np.where(kind == 1, sign | np.maximum(f, 1), w)
    return w


@dataclass(frozen=True)
class Synthetic:
    """A seeded synthetic set of a ligero chain relation (``archive.direct.ligero.relchain.instances`` / ``instances_digest``,
    and before it ``fp8.chain.instances_fp8``): input ``i`` draws ``rng = default_rng([seed, i])``, ``a`` then ``b`` (``K_VU`` words
    each) with ``operands``, and records ``tc_dot(model)`` chained from ``+0`` after every step.  The set has no files: its
    ``manifest_sha256`` is ``sha256("<relation> synthetic|seed=<seed>|n=<n>|K=1536")``."""

    dataset_id: str        # the --instances dataset token
    dataset: str           # the instances_dataset string the relation's bench records
    tier: str
    relation: str          # the digest prefix
    target: str
    instruction: str
    model: object          # verity.ml.circuits.silicon.models.GroupSum
    operands: object       # (rng, shape) -> int64 words
    dtype: object          # the operand word dtype
    epilogue: str | None
    recipe: str

    @property
    def spec(self) -> str:
        return f"{self.dataset_id}/{self.tier}"

    @property
    def k(self) -> int:
        return int(self.model.k)

    @property
    def steps(self) -> int:
        return K_VU // self.k

    def digest(self, n: int, seed: int = SYNTHETIC_SEED) -> str:
        return hashlib.sha256(f"{self.relation} synthetic|seed={seed}|n={n}|K={K_VU}".encode()).hexdigest()

    def instance(self, i: int, seed: int = SYNTHETIC_SEED) -> tuple[np.ndarray, np.ndarray, list[int]]:
        rng = np.random.default_rng([seed, i])
        a = self.operands(rng, (K_VU,))
        b = self.operands(rng, (K_VU,))
        al, bl, k = a.tolist(), b.tolist(), self.k
        accs, c = [], 0
        for s in range(self.steps):
            c = S.tc_dot(self.model, c, al[k * s : k * (s + 1)], bl[k * s : k * (s + 1)])
            accs.append(c)
        return a, b, accs


# The BF16_HOPPER target is not in every tree this script runs from: its name and instruction as pinned there
_BF16_HOPPER = getattr(T, "BF16_HOPPER", None)
BF16_HOPPER_TARGET = _BF16_HOPPER.name if _BF16_HOPPER else "bf16-hopper-mma-draft/2026-09-22"
BF16_HOPPER_INSTRUCTION = _BF16_HOPPER.instruction if _BF16_HOPPER else "sm90.mma.m16n8k16.bf16"

FP8_ADA_SYNTHETIC = Synthetic(
    dataset_id=FP8_ADA_DATASET_ID, dataset=FP8_ADA_DATASET, tier=FP8_ADA_TIER, relation="fp8-ada",
    target=T.FP8_ADA_MMA.name, instruction=T.FP8_ADA_MMA.instruction, model=S.ADA_E4M3_M16N8K32,
    operands=fp8_random_finite_operands, dtype=np.uint8, epilogue=None,
    recipe="a, b = uniform finite E4M3 bytes (NaN codes -> 0x77 | sign); accs = tc_dot(ADA_E4M3_M16N8K32) chained from +0 "
           "(archive.direct.ligero.fp8.chain.instances_fp8)")
BF16_HOPPER_SYNTHETIC = Synthetic(
    dataset_id="bench-instances-bf16-hopper/v1", dataset="bench-instances-bf16-hopper/v1", tier="vu-k1536-bf16-hopper", relation="bf16-hopper",
    target=BF16_HOPPER_TARGET, instruction=BF16_HOPPER_INSTRUCTION, model=S.HOPPER_BF16_M16N8K16,
    operands=bf16_random_finite_operands, dtype=np.uint16, epilogue=EPILOGUE_BF16,
    recipe="a, b = random_finite_bf16 (exponent fields [96, 160), 1/10 signed zeros, 1/10 subnormals); accs = tc_dot(HOPPER_BF16_M16N8K16) "
           "chained from +0 (archive.direct.ligero.relchain.instances, relation bf16-hopper)")
FP8_HOPPER_SYNTHETIC = Synthetic(
    dataset_id="bench-instances-fp8-hopper/v1", dataset="bench-instances-fp8-hopper/v1", tier="vu-k1536-fp8-hopper", relation="fp8-hopper",
    target=T.FP8_HOPPER_WGMMA.name, instruction=T.FP8_HOPPER_WGMMA.instruction, model=S.HOPPER_E4M3_K32,
    operands=fp8_random_finite_operands, dtype=np.uint8, epilogue=None,
    recipe="a, b = uniform finite E4M3 bytes (NaN codes -> 0x77 | sign); accs = tc_dot(HOPPER_E4M3_K32) chained from +0 "
           "(archive.direct.ligero.relchain.instances, relation fp8-hopper)")
SYNTHETIC = {s.spec: s for s in (FP8_ADA_SYNTHETIC, BF16_HOPPER_SYNTHETIC, FP8_HOPPER_SYNTHETIC)}


def fp8_instance(i: int, seed: int = FP8_ADA_SEED) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Input ``i`` of the FP8 Ada synthetic set: ``(a (1536,), b (1536,), accs (48,))`` exactly as ``instances_fp8`` forms it."""

    return FP8_ADA_SYNTHETIC.instance(i, seed)


def fp8_instances_digest(n: int, seed: int = FP8_ADA_SEED) -> str:
    """``archive.direct.ligero.fp8.chain.instances_digest``: a digest of the recipe (the set has no files)."""

    return FP8_ADA_SYNTHETIC.digest(n, seed)


def _synthetic_worker(job: tuple[str, int]):
    spec, i = job
    syn = SYNTHETIC[spec]
    a, b, accs = syn.instance(i)
    return i, a.astype(syn.dtype), b.astype(syn.dtype), np.array(accs, np.uint32)


def load_synthetic(syn: Synthetic, lo: int | None = None, hi: int | None = None, only_ids: list[int] | None = None,
                   total_n: int = SYNTHETIC_N, procs: int = 1) -> InstanceSet:
    """Generate the chosen instances of a synthetic set (``procs`` processes) as an ``InstanceSet``."""

    if only_ids is not None:
        ids = np.array(sorted(set(only_ids)), dtype=np.int64)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= total_n:
            raise ValueError(f"--only-ids outside [0, {total_n})")
    else:
        lo, hi = _slice(lo, hi, total_n)
        ids = np.arange(lo, hi, dtype=np.int64)
    steps, k = syn.steps, syn.k
    a = np.zeros((ids.size, steps, k), syn.dtype)
    b = np.zeros((ids.size, steps, k), syn.dtype)
    acc = np.zeros((ids.size, steps), np.uint32)
    jobs = [(syn.spec, int(i)) for i in ids]
    if procs > 1 and len(jobs) > 1:
        import multiprocessing as mp

        with mp.Pool(procs) as pool:
            results = pool.map(_synthetic_worker, jobs, chunksize=16)
    else:
        results = [_synthetic_worker(j) for j in jobs]
    for r, (i, ai, bi, acci) in enumerate(results):
        assert i == ids[r]
        a[r] = ai.reshape(steps, k)
        b[r] = bi.reshape(steps, k)
        acc[r] = acci
    return InstanceSet(
        dataset=syn.dataset, tier=syn.tier, instruction=syn.instruction, target=syn.target, model=syn.model.name,
        manifest_sha256=syn.digest(total_n), source=f"generated: rng = default_rng([{SYNTHETIC_SEED}, i]); {syn.recipe}",
        ids=ids, a=a, b=b, c0=np.zeros(ids.size, np.uint32), acc=acc, y=epilogue_words(syn.epilogue, acc[:, -1]), epilogue=syn.epilogue,
        total_n=total_n,
    )


def load_fp8_ada(lo: int | None = None, hi: int | None = None, only_ids: list[int] | None = None, total_n: int = FP8_ADA_N,
                 procs: int = 1) -> InstanceSet:
    return load_synthetic(FP8_ADA_SYNTHETIC, lo, hi, only_ids, total_n, procs)


DATASETS = {
    f"{BENCH_INSTANCES}/vu-k1536": "FIRST: 4,096 inputs x 96 BF16 steps, per-step accumulators + f32_to_bf16 epilogue",
    f"{BENCH_INSTANCES}/tu-k16": "FIRST: 4,096 single transitions with arbitrary finite c",
    f"{BENCH_INSTANCES}/vu-k1536-neg": "FIRST: 52 claimed-wrong / out-of-domain inputs (the harness must detect all)",
    FP8_ADA_SYNTHETIC.spec: "FP8_ADA_MMA: 4,096 inputs x 48 E4M3 steps generated from seed 20260922, bare FP32 word",
    BF16_HOPPER_SYNTHETIC.spec: "BF16_HOPPER: 4,096 inputs x 96 BF16 steps generated from seed 20260922, f32_to_bf16 epilogue",
    FP8_HOPPER_SYNTHETIC.spec: "FP8_HOPPER_WGMMA: 4,096 inputs x 48 E4M3 steps generated from seed 20260922, bare FP32 word",
}


def load_instances(spec: str, args) -> InstanceSet:
    if spec not in DATASETS:
        raise SystemExit(f"unknown input set {spec!r}; known: {', '.join(DATASETS)}")
    dataset, _, tier = spec.rpartition("/")
    lo, hi = (None, None)
    if args.range:
        lo_s, hi_s = args.range.split(":")
        lo, hi = int(lo_s), int(hi_s)
    only = [int(t) for t in args.only_ids.split(",") if t] if args.only_ids else None
    if dataset == BENCH_INSTANCES:
        manifest = Path(args.manifest) if args.manifest else REPO / "fixtures" / "bench-instances" / "v1" / "manifest.json"
        return load_bench_instances(tier, Path(args.root) if args.root else None, manifest, lo, hi, only)
    if spec in SYNTHETIC:
        return load_synthetic(SYNTHETIC[spec], lo, hi, only, procs=args.procs)
    raise SystemExit(f"unknown input set {spec!r}; known: {', '.join(DATASETS)}")


# --- the replay ------------------------------------------------------------------------------------------------------------------------


def _is_finite_fp32(w: np.ndarray) -> np.ndarray:
    return ((np.asarray(w, np.uint32) >> 23) & 0xFF) != 0xFF


def epilogue_words(name: str | None, acc: np.ndarray) -> np.ndarray:
    if name is None:
        return np.asarray(acc, np.uint32)
    if name == EPILOGUE_BF16:
        return np.array([f32_to_bf16_word(int(v)) for v in acc], dtype=np.uint16)  # the target's own cast, word by word
    raise ValueError(f"unknown epilogue {name!r}")


def replay(probe, ins: P.Instruction, inst: InstanceSet, log=print) -> dict:
    """Run every instance of ``inst`` through ``probe`` (one instruction per step per tile) and compare every word."""

    M, K, fmt = ins.m, ins.k, ins.fmt
    if inst.k != K or fmt.dtype != inst.a.dtype:
        raise RuntimeError(f"{inst.dataset}/{inst.tier} holds k={inst.k} {inst.a.dtype} words; {ins.id} takes k={K} {fmt.dtype}")
    n, steps = inst.n, inst.steps
    ar = np.arange(n)
    rows, cols = ar % M, (ar // M) % N
    carried = inst.c0.astype(np.uint32).copy()
    hw_acc = np.zeros((n, steps), np.uint32)
    per_step: list[int] = []
    first: list[dict] = []
    # a non-finite operand word poisons the other words of its tile row (NaN x 0 = NaN): those tiles are out of the domain anyway
    # and are counted apart
    finite_ops = fmt.is_finite(inst.a).all(axis=(1, 2)) & fmt.is_finite(inst.b).all(axis=(1, 2))
    other_nonzero = other_nonzero_nonfinite = 0
    t0 = time.time()
    for s in range(steps):
        A = np.zeros((n, M, K), fmt.dtype)
        Bt = np.zeros((n, N, K), fmt.dtype)
        C = np.zeros((n, M, N), np.uint32)
        A[ar, rows] = inst.a[:, s]
        Bt[ar, cols] = inst.b[:, s]
        C[ar, rows, cols] = carried
        D = probe.tiles(A, Bt, C)
        got = D[ar, rows, cols].copy()
        D[ar, rows, cols] = 0
        other_nonzero += int(np.count_nonzero(D[finite_ops]))  # zero rows x anything on a zero accumulator: every other word is +0
        other_nonzero_nonfinite += int(np.count_nonzero(D[~finite_ops]))
        hw_acc[:, s] = got
        if inst.acc is not None:
            bad = np.nonzero(got != inst.acc[:, s])[0]
            per_step.append(int(bad.size))
            for r in bad[: max(0, FIRST_MISMATCHES - len(first))]:
                first.append({"index": int(inst.ids[r]), "step": s, "expected": f"{int(inst.acc[r, s]):08x}", "got": f"{int(got[r]):08x}",
                              **P.hexrec(carried[r], inst.a[r, s], inst.b[r, s], got[r], fmt.hex_width)})
        carried = got  # the hardware's own accumulator carries on
    gpu_seconds = time.time() - t0
    y_hw = epilogue_words(inst.epilogue, carried)
    claimed_contradicted = None
    if inst.negative is None:
        y_ref = inst.y
        y_bad = np.nonzero(y_hw != y_ref)[0]
    else:
        # the reference word of a claimed-wrong row is its CORRECT word (the claimed one is what the harness must contradict); a reject
        # row has no reference word -- the domain check judges it (negative_verdicts)
        rows = inst.negative["rows"]
        is_wrong = np.array([r["verdict"] == "wrong" for r in rows], bool)
        y_ref = np.array([int(r["correct_y"]) if r["verdict"] == "wrong" else int(r["claimed_y"]) for r in rows], inst.y.dtype)
        y_bad = np.nonzero(is_wrong & (y_hw != y_ref))[0]
        claimed_contradicted = int(np.count_nonzero(y_hw != inst.y))
    y_width = 4 if inst.y.dtype == np.uint16 else 8
    first_y = [{"index": int(inst.ids[r]), "step": "epilogue" if inst.epilogue else steps - 1, "expected": f"{int(y_ref[r]):0{y_width}x}",
                "got": f"{int(y_hw[r]):0{y_width}x}", "acc": f"{int(carried[r]):08x}"} for r in y_bad[:FIRST_MISMATCHES]]

    # domain: non-finite operand words, non-finite accumulators before the last step (D8: the last step may saturate)
    nonfinite_operand = ~finite_ops
    nonfinite_c0 = ~_is_finite_fp32(inst.c0)
    nonfinite_mid = ~_is_finite_fp32(hw_acc[:, :-1]).all(axis=1) if steps > 1 else np.zeros(n, bool)
    out_of_domain = nonfinite_operand | nonfinite_c0 | nonfinite_mid

    intermediate = int(n * (steps - 1)) if inst.acc is not None else 0
    final = int(n) if inst.acc is not None else 0
    epilogue_n = int(n) if inst.epilogue is not None or inst.acc is None else 0  # the y compare is a separate word only with an epilogue
    if inst.acc is not None and inst.epilogue is None:
        # y is the bare last accumulator: already counted as the final word (the y compare must then agree with step steps-1)
        assert int(y_bad.size) == per_step[-1], "y (bare accumulator) disagrees with the last step's compare"
    mism_intermediate = int(sum(per_step[:-1])) if inst.acc is not None else 0
    mism_final = int(per_step[-1]) if inst.acc is not None else 0
    mism_epilogue = int(y_bad.size) if epilogue_n else 0
    mismatches = mism_intermediate + mism_final + mism_epilogue
    first_all = (first + first_y)[:FIRST_MISMATCHES] if inst.acc is not None else first_y

    report = {
        "instances": n,
        "steps": steps,
        "ids": [int(inst.ids[0]), int(inst.ids[-1])] if n else [],
        "words": {"intermediate": intermediate, "final": final, "epilogue": epilogue_n, "total": intermediate + final + epilogue_n},
        "mismatches": {"intermediate": mism_intermediate, "final": mism_final, "epilogue": mism_epilogue, "total": mismatches},
        "steps_with_mismatches": {str(s): m for s, m in enumerate(per_step) if m},
        "first_mismatches": first_all,
        "other_positions_nonzero": other_nonzero,
        "other_positions_nonzero_in_nonfinite_operand_tiles": other_nonzero_nonfinite,
        "domain": {
            "nonfinite_operand_instances": int(nonfinite_operand.sum()),
            "nonfinite_c0_instances": int(nonfinite_c0.sum()),
            "nonfinite_intermediate_acc_instances": int(nonfinite_mid.sum()),
            "out_of_domain_instances": int(out_of_domain.sum()),
            "out_of_domain_ids": [int(i) for i in inst.ids[out_of_domain][:64]],
            "final_saturated_instances": int((~_is_finite_fp32(carried)).sum()),
        },
        "epilogue": inst.epilogue or "none (the bare FP32 accumulator word)",
        "gpu_seconds": round(gpu_seconds, 3),
        "_hw_acc": hw_acc,
        "_y_hw": y_hw,
        "_out_of_domain": out_of_domain,
    }
    if inst.negative is not None:
        report["negative"] = negative_verdicts(inst, y_hw, out_of_domain)
        report["negative"]["claimed_words_contradicted"] = claimed_contradicted
    return report


def negative_verdicts(inst: InstanceSet, y_hw: np.ndarray, out_of_domain: np.ndarray) -> dict:
    """The claimed-wrong tier: a ``wrong`` row is detected when the hardware word differs from the claimed word (and the hardware
    word is the row's correct word); a ``reject`` row (out of the supported domain) is detected by the domain check."""

    rows = inst.negative["rows"]
    per_row = []
    wrong = reject = wrong_detected = reject_detected = correct_reproduced = 0
    for r, row in enumerate(rows):
        hw = int(y_hw[r])
        claimed = int(row["claimed_y"])
        entry = {"id": int(row["id"]), "mistake": row["mistake"], "verdict": row["verdict"], "claimed": f"{claimed:04x}", "hardware": f"{hw:04x}"}
        if row["verdict"] == "wrong":
            wrong += 1
            correct = int(row["correct_y"])
            entry["correct"] = f"{correct:04x}"
            entry["detected"] = hw != claimed
            entry["correct_reproduced"] = hw == correct
            wrong_detected += entry["detected"]
            correct_reproduced += entry["correct_reproduced"]
        else:
            reject += 1
            entry["detected"] = bool(out_of_domain[r])
            entry["hardware_equals_claimed"] = hw == claimed
            reject_detected += entry["detected"]
        per_row.append(entry)
    return {
        "claimed_words": len(rows),
        "wrong": wrong, "wrong_detected": wrong_detected, "wrong_correct_reproduced": correct_reproduced,
        "reject": reject, "reject_detected": reject_detected,
        "detected": wrong_detected + reject_detected,
        "undetected": len(rows) - wrong_detected - reject_detected,
        "rows": per_row,
    }


# --- main ------------------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="frozen input sets on silicon (see the module docstring)")
    ap.add_argument("--instances", required=True, metavar="DATASET/TIER", help=", ".join(DATASETS))
    ap.add_argument("--instruction", default=None, help="instruction id (default: the set's target's; see tc_probe.INSTRUCTIONS)")
    ap.add_argument("--arch", default=None, help="override the instruction's nvcc arch")
    ap.add_argument("--root", default=None, help="bench-instances/v1 directory holding the built arrays (default: the manifest's directory)")
    ap.add_argument("--manifest", default=None, help="the pinned bench-instances/v1 manifest (default fixtures/bench-instances/v1/manifest.json)")
    ap.add_argument("--range", default=None, metavar="LO:HI", help="instance ids [LO, HI) (default: the whole tier)")
    ap.add_argument("--only-ids", default=None, metavar="ID,ID,...", help="re-run these instances alone")
    ap.add_argument("--repeat", type=int, default=1, help="run the replay this many times and count words that differ between repetitions")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="processes for generating the FP8 set")
    ap.add_argument("--sass-check", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    inst = load_instances(args.instances, args)
    ins_id = args.instruction or inst.instruction
    if ins_id not in P.INSTRUCTIONS:
        ap.error(f"unknown instruction {ins_id!r}; known: {', '.join(P.INSTRUCTIONS)}")
    ins = P.INSTRUCTIONS[ins_id]
    arch = args.arch or ins.arch
    print("instances", json.dumps({"dataset": inst.dataset, "tier": inst.tier, "n": inst.n, "of": inst.total_n, "steps": inst.steps, "k": inst.k,
                                   "ids": [int(inst.ids[0]), int(inst.ids[-1])], "manifest_sha256": inst.manifest_sha256,
                                   "target": inst.target, "model": inst.model, "instruction": ins.id, "epilogue": inst.epilogue}), flush=True)

    so, nvcc = P.build(ins.tu, arch, out)
    results: dict = {"schema": "instances-hw/v1", "instruction": ins.describe(), "arch": arch}
    if args.sass_check:
        results["sass"] = P.sass_check(so, ins.kernel, out, ins.sass_mma)
        print("sass", json.dumps({k: v for k, v in results["sass"].items() if k in ("ok", "kernel", "tile_kernel_mma_count", "tile_kernel_functions", "error")}), flush=True)
    probe = P.Probe(so, ins)
    identity = probe.identity()
    identity["nvcc"] = nvcc
    identity["arch"] = arch
    results["identity"] = identity
    results["layout_check"] = P.layout_check(probe, ins)
    print("device", json.dumps(identity), flush=True)
    print("layout_check", json.dumps(results["layout_check"]), flush=True)

    results["instances"] = {"dataset": inst.dataset, "tier": inst.tier, "instruction": ins.id, "target": inst.target, "model": inst.model,
                            "manifest_sha256": inst.manifest_sha256, "range": [int(inst.ids[0]), int(inst.ids[-1]) + 1] if args.only_ids is None else None,
                            "only_ids": [int(i) for i in inst.ids] if args.only_ids else None, "n": inst.n, "total_n": inst.total_n,
                            "steps": inst.steps, "k": inst.k, "source": inst.source, "epilogue": inst.epilogue}
    reps = []
    for rep in range(max(1, args.repeat)):
        reps.append(replay(probe, ins, inst))
        r = reps[-1]
        print(f"replay[{rep}]", json.dumps({k: v for k, v in r.items() if k in ("instances", "steps", "words", "mismatches", "steps_with_mismatches",
                                                                                "other_positions_nonzero", "domain", "gpu_seconds")}), flush=True)
        if "negative" in r:
            print("negative", json.dumps({k: v for k, v in r["negative"].items() if k != "rows"}), flush=True)
    rep0 = reps[0]
    nondet = 0
    for r in reps[1:]:
        nondet += int(np.count_nonzero(r["_hw_acc"] != rep0["_hw_acc"])) + int(np.count_nonzero(r["_y_hw"] != rep0["_y_hw"]))
    results["replay"] = {k: v for k, v in rep0.items() if not k.startswith("_")}
    results["repeat"] = {"runs": len(reps), "words_differing_between_runs": nondet}
    np.savez_compressed(out / f"capture_{inst.tier}.npz", ids=inst.ids, hw_acc=rep0["_hw_acc"], y_hw=rep0["_y_hw"],
                        out_of_domain=rep0["_out_of_domain"], **({"expected_acc": inst.acc} if inst.acc is not None else {}), expected_y=inst.y)

    failures: list[str] = []
    mism = rep0["mismatches"]["total"]
    if inst.negative is None:
        if mism:
            failures.append(f"{mism} hardware words differ from {inst.dataset}/{inst.tier}")
        if rep0["domain"]["out_of_domain_instances"]:
            failures.append(f"{rep0['domain']['out_of_domain_instances']} instances out of the domain")
    else:
        if rep0["negative"]["undetected"]:
            failures.append(f"{rep0['negative']['undetected']} of {rep0['negative']['claimed_words']} negatives not detected")
        if rep0["negative"]["wrong_correct_reproduced"] != rep0["negative"]["wrong"]:
            failures.append(f"hardware reproduced the correct word for {rep0['negative']['wrong_correct_reproduced']}/{rep0['negative']['wrong']} wrong rows")
    if rep0["other_positions_nonzero"]:
        failures.append(f"other_positions_nonzero={rep0['other_positions_nonzero']}")
    if nondet:
        failures.append(f"{nondet} words differed between repetitions")
    if "sass" in results and not results["sass"]["ok"]:
        failures.append(f"sass tile_kernel_mma_count={results['sass'].get('tile_kernel_mma_count')} expected={ins.sass_mma}")
    results["failures"] = failures
    results["seconds"] = round(time.time() - t_all, 1)
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))

    model_agrees = not failures if inst.negative is None else rep0["negative"]["undetected"] == 0 and rep0["negative"]["wrong_correct_reproduced"] == rep0["negative"]["wrong"]
    sass = results.get("sass")
    meta = {
        "instruction": ins.id,
        "ptx": ins.ptx,
        "arch": arch,
        "target": inst.target,
        "model": inst.model,
        "instances_dataset": inst.dataset,
        "instances_tier": inst.tier,
        "range": results["instances"]["range"],
        "only_ids": results["instances"]["only_ids"],
        "manifest_sha256": inst.manifest_sha256,
        "elements": rep0["words"]["total"],
        "words": rep0["words"],
        "mismatches": mism,
        "mismatches_by_kind": rep0["mismatches"],
        "first_mismatch": rep0["first_mismatches"][0] if rep0["first_mismatches"] else None,
        "first_mismatches": rep0["first_mismatches"],
        "steps_with_mismatches": rep0["steps_with_mismatches"],
        "per_step_checked": inst.acc is not None and inst.steps > 1,
        "steps": inst.steps,
        "instances": inst.n,
        "other_positions_nonzero": rep0["other_positions_nonzero"],
        "domain": {k: v for k, v in rep0["domain"].items() if k != "out_of_domain_ids"},
        "negative": {k: v for k, v in rep0["negative"].items() if k != "rows"} if inst.negative is not None else None,
        "repeat": results["repeat"],
        "layout_check": results["layout_check"],
        "sass": None if sass is None else {"ok": sass["ok"], "tile_kernel_mma_count": sass["tile_kernel_mma_count"], "single_instruction": sass["single_instruction"],
                                          "mma": sorted({m for f in sass["tile_kernel_functions"].values() for m in f["mma"]}), "file": sass.get("sass_file")},
        "device": identity["device"],
        "compute_capability": identity["compute_capability"],
        "driver": identity["driver"],
        "cuda": nvcc,
        "nvcc": nvcc,
        "hostname": identity["hostname"],
        "epilogue": inst.epilogue,
        "model_agrees": bool(model_agrees),
        "mismatch_semantics": ("hardware word != the reference word: the tier's expected (model-recorded) word; on the negative tier the "
                               "CORRECT word of a claimed-wrong row (reject rows have none: the domain check judges them, negative.reject_detected); "
                               "negative.claimed_words_contradicted counts hardware words != the claimed words, negative.undetected that tier's failures"),
        "source": inst.source,
        "gpu_seconds": rep0["gpu_seconds"],
        "wall": results["seconds"],
    }
    run_id = SHARED.RESEARCH_RUN_ID
    if run_id:
        status = "failed" if failures else "passed"
        checker = (f"instances_hw: hardware words vs {inst.dataset}/{inst.tier} expected words ({inst.model})" if inst.negative is None else
                   f"instances_hw: every claimed-wrong word of {inst.dataset}/{inst.tier} detected (hardware != claimed, == correct; reject rows by the domain check)")
        detail = "; ".join(failures) or (f"every hardware word reproduced ({meta['elements']} words, {inst.n} instances x {inst.steps} steps)" if inst.negative is None
                                          else f"{rep0['negative']['detected']}/{rep0['negative']['claimed_words']} negatives detected")
        measurements = [
            {"name": "wall", "value": results["seconds"], "unit": "s"},
            {"name": "gpu_seconds", "value": rep0["gpu_seconds"], "unit": "s"},
            {"name": "elements", "value": meta["elements"], "unit": "words"},
            {"name": "mismatches", "value": mism, "unit": "words"},
            {"name": "instances", "value": inst.n, "unit": "instances"},
            {"name": "steps", "value": inst.steps, "unit": "steps"},
            {"name": "out_of_domain_instances", "value": rep0["domain"]["out_of_domain_instances"], "unit": "instances"},
            {"name": "other_positions_nonzero", "value": rep0["other_positions_nonzero"], "unit": "words"},
        ]
        if inst.negative is not None:
            measurements += [{"name": "negatives_detected", "value": rep0["negative"]["detected"], "unit": "instances"},
                             {"name": "negatives_undetected", "value": rep0["negative"]["undetected"], "unit": "instances"}]
        if sass is not None:
            measurements.append({"name": "tile_kernel_mma_count", "value": sass.get("tile_kernel_mma_count"), "unit": "instructions"})
        artifacts = [f"capture_{inst.tier}.npz"] + ([sass["sass_file"]] if sass and sass.get("sass_file") else [])
        P.write_result(out, {
            "schema": "research/result/v0.1",
            "run_id": run_id,
            "kind": "tc-evidence/v1",
            "workload_fingerprint": {
                "probe": "benchmarks/tc_probe/instances_hw.py", "mode": "instances", "instruction": ins.id, "ptx": ins.ptx, "arch": arch,
                "sku": identity["device"], "device": identity["device"], "compute_capability": identity["compute_capability"],
                "driver": identity["driver"], "nvcc": nvcc, "target": inst.target, "model": inst.model,
                "instances": {"dataset": inst.dataset, "tier": inst.tier, "range": meta["range"], "only_ids": meta["only_ids"], "manifest_sha256": inst.manifest_sha256},
                "repeat": args.repeat,
            },
            "validation": {"status": status, "checker": checker, "detail": detail, "evidence": ["probe_results.json"]},
            "measurements": measurements,
            "measurement_files": ["probe_results.json"],
            "artifacts": artifacts,
            "tc_evidence": meta,
        })
    print("FAILURES" if failures else "ALL HARDWARE WORDS REPRODUCED" if inst.negative is None else "ALL NEGATIVES DETECTED",
          failures or "", f"{results['seconds']}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
