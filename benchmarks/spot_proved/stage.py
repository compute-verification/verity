"""Stage sampled proof blocks of one captured step (`capture.py`) as the one-stage audit's saved values (`a11.py --load-values`).

    python benchmarks/spot_proved/stage.py --capture DIR --step S --out DIR [--per-type 8] [--types T,...]
    python benchmarks/spot_proved/stage.py --capture DIR --step S --out DIR --every-call N [--jobs J] [--deadline T]
        [--rows T,... | --rows-past-keys K]

``--every-call`` stages nothing: it runs the match gate on N instances of every recorded call of every type, over every layer
of a step the capture kept whole (``--keep all``, whose ``kept_params`` hold every layer's norm weights; FP8 linears, MoE
launches and cuBLAS GEMMs only in the opened layers, whose weights are written), and writes ``gate.json``: per type its calls, agreements,
disagreements, errors and first failures, the recorded ops no type gates, and apart from them the ops the Program binds to no
value (`SCHEDULE_OPS`). ``--rows`` cuts every population to some token
rows, and ``--rows-past-keys 2048`` to the rows attending to more than 2,048 keys (DSA's long rows), read off the indexer's
top-k inputs (`long_rows`).

* **Commitment.** Each rank's ``ops.jsonl`` and ``params.json`` are hashed again and held to its ``meta.json``'s rank root,
  the ranks' roots to the step root every rank's ``commit.json`` names, and the coins to one value on every rank. Every blob
  is held to its SHA-512 name when read, and every opened parameter to its SHA-512 in ``params.json`` and ``commit.json``.
* **Population.** A proof block type's instances are the recorded calls of its kernel in the opened layers, on every rank,
  over the step's real tokens: a group quant's token row, a FlashInfer FP8 linear's (token, output column) when the call has
  fewer than 32 rows (the op's FlashInfer branch), an MoE w13 or routed w2 launch's (slot, output column), a fused allreduce +
  RMSNorm's token for its residual half-rows and its norm, a plain allreduce's (token, half-row) on symm-mem's multimem route
  or FlashInfer's MNNVL one (its received row the same on every rank; a call on NCCL's route is counted, not staged), an
  unbiased bf16 cuBLAS GEMM's token
  row (``aten::mm`` against a bf16 weight's transposed view, or ``aten::linear``) at the route its call's M rows take (a call past
  ``CUBLAS_ROUTES["m_max"]`` is counted, not staged), MLA's absorbs (``aten::bmm``), the first layer's ``_C::rms_norm``, the
  MoE block's ``aten::add`` of its shared and routed outputs, the router's grouped top-k, ``silu_and_mul`` and ``moe_sum`` per row,
  and FA3's split combine per (token, head). Instances are sampled from it
  by the capture's coins, ``--per-type`` of each type.
* **Match gate.** Each sampled instance's Definition is evaluated on the captured inputs and must return the captured outputs
  bit for bit (completeness, not soundness: an instance that disagrees is reported and not staged).
* **Wires.** Each value a staged instance reads has a source under the step root (`Wires`): the recorded call that output it
  (or the calls that wrote its real rows through strided views, one or several: a padded batch's rows past the step's tokens
  are in no population, so they cost nothing), a parameter held to ``params.json``, a `GAPS` input (a value the step
  computes that the capture doesn't record: a trace gap, counted at the parts of it staged instances read and printed as gap
  bits, never charged: wiring it takes it to 0), or an `ADVICE` input (a choice nothing computes, charged the same way; none today). Of a
  split's K/V rows, a token's slots past its keys and the rows no staged split reads cost nothing. A split's K/V row is
  wired by its page-table slot (`Wires.gathered`) to the step's cache write that put it there, else to the window's
  carried cache (`HISTORY_PORT`, counted as history bits), and stays a gap only where neither is established. Where the
  capture records the top-k's conversion to slots (`CONVERT`) and its rank root commits the batch (`batch_sha512`), a row
  is wired by its request and position instead, as the Program reads it, and no page-table entry is read
  (`Wires.by_position`). A token past
  the top-k's keys adds its selection's settings (selection bits). A MoE pair's
  expert is the router's output (`ROUTERS`), never the align's schedule. An instance that reads anything else is not staged, and its reason names the fix.
  Staging checks the wiring against the committed capture; a11 proves the staged units alone and doesn't read it.
* **Out.** ``values.json``, ``units.json``, ``<label>.npz`` (a11's saved-values format), ``kinds.json`` (a11's table of the
  staged labels), ``window.json`` (what the replay block is, how its values link to serving, and the wiring: each staged
  instance's inputs with their sources, the advice, gap, history and selection bits, the calls wired to that no population gates,
  and the unwired) and
  ``stage.json`` (populations, samples, the gate's results and every type not staged, with its reason).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

#: the tree a11.py comes from: this file's, else (sent alone into a run's inputs) the working directory's
TREE = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[1] / "one_stage").is_dir() else Path.cwd()
for _p in ("benchmarks/one_stage", "tools/verity/circuit_check/src", "benchmarks/numerical/python", "adapters/vllm",
           "verity/protocols/flock/prover/python", "."):
    sys.path.insert(0, str(TREE / _p))
sys.setrecursionlimit(100000)

NP = {"torch.bfloat16": np.uint16, "torch.float16": np.uint16, "torch.float32": np.uint32, "torch.float8_e4m3fn": np.uint8,
      "torch.uint8": np.uint8, "torch.int8": np.int8, "torch.int32": np.int32, "torch.int64": np.int64, "torch.bool": np.uint8}
CK_BITS = {"F8_E4M3": 8, "BF16": 16, "F32": 32}
NUM_SM = 132  # the H200's SMs, FA3's split count reads it
#: labels of one family that differ only in a size: one is staged, the census's type when the step has it, else the
#: smallest (SplitPart past T=64 needs more than Flock's 2^27-bit block)
FAMILIES = {"SparseMlaSplitPart_T": "SparseMlaSplitPart_T64", "SparseMlaCombine_N": "SparseMlaCombine_N5"}


#: the file beside a step's ``params.json`` holding the node's multimem reduction tree (`capture.py site --allreduce-tree`)
TREE_FILE = "allreduce_tree.json"


def parse_tree(text: str, world: int):
    """A reduction tree over WORLD ranks, as JSON pairs or as kernels' probe writes it, ``(((((((0,2),3),1),5),6),4),7)``."""
    from verity.definitions.registry import tp_allreduce as AR
    return AR.multimem_tree(json.loads(text.replace("(", "[").replace(")", "]")), world)


def multimem_trees(world: int, tree=None) -> str:
    """TREES of `AllReduceMultimem_v1` at WORLD ranks, one per owner: TREE (the node's, read off its switch), else the
    registry's H200 tree at 8 and a chain otherwise."""
    from verity.definitions.registry import tp_allreduce as AR
    tree = tree if tree is not None else AR.GLM53_H200_TREE if world == 8 else AR.chain(world)
    return json.dumps([AR.multimem_tree(tree, world)] * world, separators=(",", ":"))


def load_roles(params: dict) -> dict:
    """The served fields vLLM rewrites at load, by `verity_vllm.check.fp8_ue8m0`'s roles: each block-FP8 linear outside the routed
    experts (a DeepGEMM linear: its weight and ``weight_scale_inv`` requantised to ue8m0), and MLA's ``W_UK_T`` / ``W_UV``
    (absorbed from the requantised ``kv_b_proj``)."""
    roles = {}
    for n, x in params.items():
        if ".experts." in n:
            continue
        if n.endswith(".weight") and x["dtype"] == "torch.float8_e4m3fn" and n + "_scale_inv" in params:
            roles[n], roles[n + "_scale_inv"] = {"role": "fp8_w_requant_ue8m0"}, {"role": "fp8_ws_requant_ue8m0"}
        elif n.endswith((".W_UK_T", ".W_UV")):
            roles[n] = {"role": "mla.absorb"}
    return roles


def batch_sha512(req_ids, scheduled) -> str:
    """SHA-512 over a step's batch as `capture.py` commits it under the rank root: its request ids in batch order and each
    request's scheduled tokens."""
    doc = json.dumps({"req_ids": req_ids, "scheduled": scheduled}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha512(f"verity/spot/batch\0{doc}".encode()).hexdigest()


class Capture:
    """One step of a capture directory, its commitment checked on load: a rank root over ``ops.jsonl``, ``params.json`` and,
    where ``meta.json`` names one, the batch (`batch_sha512`)."""

    def __init__(self, d: Path, step: int) -> None:
        self.d, self.step = d, step
        self.ranks = sorted(int(p.name[4:]) for p in d.glob("rank*") if (p / f"step{step}" / "meta.json").exists())
        if not self.ranks:
            raise SystemExit(f"{d}: no rank recorded step {step}")
        self.meta, self.commit, self.events, self.params = {}, {}, {}, {}
        #: each rank's request ids in batch order where its rank root commits them (`batch_sha512`), else None
        self.req_ids: dict[int, list[str] | None] = {}
        for r in self.ranks:
            sd = d / f"rank{r}" / f"step{step}"
            ops = (sd / "ops.jsonl").read_bytes()
            pj = next(p for p in (sd / "params.json", d / f"rank{r}" / "params.json") if p.is_file()).read_bytes()
            meta = json.loads((sd / "meta.json").read_text())
            ops_sha, params_sha = hashlib.sha512(ops).hexdigest(), hashlib.sha512(pj).hexdigest()
            if (ops_sha, params_sha) != (meta["ops_sha512"], meta["params_sha512"]):
                raise SystemExit(f"rank {r}: ops.jsonl or params.json is not what meta.json committed")
            batch = meta.get("batch_sha512")
            if batch is not None and batch != batch_sha512(meta.get("req_ids"), meta.get("scheduled")):
                raise SystemExit(f"rank {r}: meta.json's req_ids or scheduled counts are not the batch it committed")
            if hashlib.sha512("\0".join(["verity/spot/rank-root", ops_sha, params_sha, *([batch] if batch else [])]).encode()
                              ).hexdigest() != meta["rank_root"]:
                raise SystemExit(f"rank {r}: rank root {meta['rank_root'][:16]} is not SHA-512 of its files")
            self.req_ids[r] = list(meta["req_ids"]) if batch else None
            self.meta[r], self.commit[r] = meta, json.loads((sd / "commit.json").read_text())
            self.events[r] = [json.loads(x) for x in ops.decode().splitlines() if x]
            self.params[r] = json.loads(pj)
        c0 = self.commit[self.ranks[0]]
        for r in self.ranks:
            c = self.commit[r]
            if (c["root"], c["coins"], c["rank_roots"]) != (c0["root"], c0["coins"], c0["rank_roots"]):
                raise SystemExit(f"rank {r}: commit.json disagrees with rank {self.ranks[0]}'s")
            if c["rank_roots"][r] != self.meta[r]["rank_root"]:
                raise SystemExit(f"rank {r}: the step root's leaf is not its rank root")
        if hashlib.sha512(("verity/spot/step-root\0" + "\0".join(c0["rank_roots"])).encode()).hexdigest() != c0["root"]:
            raise SystemExit("the step root is not SHA-512 over the ranks' roots")
        self.root, self.coins = c0["root"], bytes.fromhex(c0["coins"])
        self.layers = sorted(x for v in c0["layers"].values() for x in v)
        self.whole = all(self.commit[r].get("whole", self.commit[r].get("kept") == "every recorded tensor") for r in self.ranks)
        trees = {r: json.loads(p.read_text()) for r in self.ranks
                 if (p := d / f"rank{r}" / f"step{step}" / TREE_FILE).is_file()}
        if trees and (len(trees) != len(self.ranks) or len({json.dumps(t["tree"]) for t in trees.values()}) != 1):
            raise SystemExit(f"step {step}: the ranks' {TREE_FILE} are missing or disagree")
        #: the node's multimem tree and where it was read (``--allreduce-tree`` replaces it), else None: the registry's
        self.tree: dict | None = trees[self.ranks[0]] if trees else None
        self.every = False
        self.per_call = 0
        #: with ``--rows`` / ``--rows-past-keys``, the token rows every population is cut to (each with its keys, or None)
        self.only: dict[int, int | None] | None = None
        self.checkpoint: str | None = None
        self._params: dict = {}
        self._composers: dict = {}
        self._ck = None
        self._lock = threading.Lock()
        #: while set (`traced`), every recorded tensor read is appended here as (rank, its recorded reference)
        self.reads: list | None = None

    def tensor(self, r: int, ref: dict) -> np.ndarray:
        """A recorded tensor's words (its logical order), checked against its SHA-512."""
        if self.reads is not None:
            self.reads.append((r, ref))
        if "param" in ref:
            return self.param_view(r, ref)
        if "sha512" not in ref:
            raise LookupError(f"not recorded with its bytes: {ref}")
        if not self.kept(r, ref):
            raise LookupError(f"not kept: rank {r}'s {ref['dtype']} {ref['shape']} (over the capture's keep cap, outside the "
                              "opened layers)")
        h = ref["sha512"]
        raw = (self.d / f"rank{r}" / "blobs" / h[:2] / h).read_bytes()
        if hashlib.sha512(raw).hexdigest() != h:
            raise SystemExit(f"rank {r}: blob {h[:16]} is not its name")
        return np.frombuffer(raw, dtype=NP[ref["dtype"]]).reshape(ref["shape"])

    def part(self, r: int, ref: dict, key: tuple) -> np.ndarray:
        """``tensor(r, ref)[key]``, ``key`` ints and unit-step slices over the leading axes, its read recorded with the part it
        takes, as [start, stop) per axis: a `GAPS` or `ADVICE` input is counted at the parts staged instances read."""
        reads, self.reads = self.reads, None
        try:
            x = self.tensor(r, ref)
        finally:
            self.reads = reads
        if reads is not None:
            reads.append((r, ref, _part(ref["shape"], key)))
        return x[key]

    def kept(self, r: int, ref: dict) -> bool:
        """Whether a recorded tensor's bytes are in the capture."""
        h = ref.get("sha512")
        return h is not None and (self.d / f"rank{r}" / "blobs" / h[:2] / h).is_file()

    def param(self, r: int, name: str) -> np.ndarray:
        if (r, name) not in self._params:
            od = self.d / f"rank{r}" / f"step{self.step}" / "open"
            meta = self.commit[r]["opened"].get(name) or self.commit[r].get("kept_params", {}).get(name)
            if (meta is None or not (od / f"{name}.bin").is_file()) and self.checkpoint is not None and name in self.params[r]:
                self._params[(r, name)] = self.from_checkpoint(r, name)
                return self._params[(r, name)]
            if meta is None:
                raise LookupError(f"parameter {name} is not in an opened layer")
            raw = (od / f"{name}.bin").read_bytes()
            h = hashlib.sha512(raw).hexdigest()
            if h != meta["sha512"] or (name in self.params[r] and self.params[r][name]["sha512"] != h):
                raise SystemExit(f"rank {r}: opened {name} is not the committed bytes")
            self._params[(r, name)] = np.frombuffer(raw, dtype=NP[meta["dtype"]]).reshape(meta["shape"])
        return self._params[(r, name)]

    def from_checkpoint(self, r: int, name: str) -> np.ndarray:
        """Rank ``r``'s served ``name`` composed from ``checkpoint`` by `verity_vllm.check.weights_of_record` (TP slices, fused
        parts, the experts' scale grid), with vLLM's load-time ue8m0 requant of each DeepGEMM linear and MLA's absorbed
        ``W_UK_T`` / ``W_UV``, held to the SHA-512 the step's ``params.json`` committed."""
        with self._lock:
            if r not in self._composers:
                self._composers[r] = self._composer(r)
        rec = self.params[r][name]
        raw = np.ascontiguousarray(self._composers[r](name)).tobytes()
        if hashlib.sha512(raw).hexdigest() != rec["sha512"]:
            raise LookupError(f"rank {r}: {name} composed from the checkpoint is not the committed bytes")
        return np.frombuffer(raw, dtype=NP[rec["dtype"]]).reshape(rec["shape"])

    def _composer(self, r: int):
        from verity_vllm.check import fp8_ue8m0
        from verity_vllm.check.checkpoint_index import CheckpointIndex
        from verity_vllm.check.weights_of_record import compose
        if self._ck is None:
            self._ck = CheckpointIndex(self.checkpoint, verify_shards=False)
        ck, P, world = self._ck, self.params[r], len(self.ranks)
        dims = {n: (x["shape"], 8 * np.dtype(NP[x["dtype"]]).itemsize) for n, x in P.items()}

        def raw(f: str):
            # MLA's absorbed W_UK_T / W_UV are composed from kv_b_proj, which a decode-only step never reads, so it isn't among
            # the step's parameters: it is then the rank's dim-0 slice of the checkpoint tensor, at the checkpoint's width
            shape, bits = dims[f] if f in dims else ([ck.tensors[f][2][0] // world, *ck.tensors[f][2][1:]], CK_BITS[ck.dtype(f)])
            return compose(f, shape, bits, ck, world=world, rank=r)
        rw = fp8_ue8m0.Rewrites(list(dims), load_roles(P), raw, ck.config)
        return lambda n: (rw.compose(n, dims[n][0]) if rw.covers(n) else compose(n, *dims[n], ck, world=world, rank=r))[0]

    def param_view(self, r: int, ref: dict) -> np.ndarray:
        base = self.param(r, ref["param"])
        if "stride" not in ref:
            return base.reshape(ref["shape"])
        item = base.itemsize
        flat = base.reshape(-1)
        off = ref["byte_offset"] // item
        return np.lib.stride_tricks.as_strided(flat[off:], shape=ref["shape"], strides=[s * item for s in ref["stride"]])

    def opened(self, e: dict, weights: bool = False) -> bool:
        """Whether a call's values are here: an opened layer's, or (`every`, a step kept whole) any layer's call that reads
        no weights past the kept small parameters."""
        return e.get("layer") in self.layers or (self.every and (not weights or self.checkpoint is not None))

    def cells(self, label: str, call: tuple, *dims: int, slots: int = 1):
        """A call's instance indices over ``dims``: every one, or with `per_call` set (``--every-call``) that many sampled by
        the coins, so a gate over every call never enumerates a call's rows times its columns. With `only` set, the first
        dimension is cut to those tokens' rows (``slots`` rows a token, token-major)."""
        firsts = None
        if self.only is not None and dims:
            firsts = [t * slots + k for t in sorted(self.only) for k in range(slots) if t * slots + k < dims[0]]
            dims = (len(firsts), *dims[1:])
        n = int(np.prod(dims, dtype=np.int64)) if dims else 1
        for i in (draw(self.coins, f"cells/{label}/{call}", n, self.per_call) if self.per_call else range(n)):
            idx = tuple(int(x) for x in np.unravel_index(i, dims))
            yield idx if firsts is None else (firsts[idx[0]], *idx[1:])

    def tokens(self, r: int) -> int:
        return sum(self.meta[r]["scheduled"].values())

    def rows(self, r: int, rows: int) -> list[int]:
        """A call's real rows of one token each (`real_rows`), cut to `only`'s tokens when set."""
        return [t for t in range(self.real_rows(r, rows)) if self.only is None or t in self.only]

    def slots(self, r: int, rows: int) -> int:
        """Rows a token in a call of ``rows`` rows: the MoE's (token, slot) rows run token-major, a row a slot."""
        n, real = self.tokens(r), self.real_rows(r, rows)
        return real // n if n and real % n == 0 else 1

    def padded(self, r: int) -> int | None:
        """The forward's rows, its real tokens padded (the recorded ``model_output``'s)."""
        return next((e["out"]["hidden"]["shape"][0] for e in reversed(self.events[r]) if e["op"] == "model_output"), None)

    def real_rows(self, r: int, rows: int) -> int:
        """Of a call's ``rows``, those of the step's real tokens: the forward's padded rows (or a multiple of them, one per
        token and slot) cut to the scheduled tokens; any other count is taken whole."""
        padded = self.padded(r)
        n = self.tokens(r)
        if padded and rows % padded == 0:
            return n * (rows // padded)
        return min(rows, n) if rows >= n else rows


def long_rows(cap: Capture, keys: int) -> dict[int, int]:
    """The step's token rows attending to more than ``keys`` keys, each with its keys, read off the DSA indexer's top-k inputs
    on the first rank: the decode rows, which lead the batch (``_C::cooperative_topk``'s ``lengths``), and the prefill
    rows, which end it (``_C::top_k_per_row_prefill``'s ``rowEnds - rowStarts``)."""
    r = cap.ranks[0]
    n = cap.tokens(r)
    dec = next((e for e in cap.events[r] if e["op"] == "_C::cooperative_topk"), None)
    pre = next((e for e in cap.events[r] if e["op"] == "_C::top_k_per_row_prefill"), None)
    if dec is None and pre is None:
        raise SystemExit(f"step {cap.step}: no DSA indexer top-k was recorded, so no row's keys are known")
    m = int(pre["in"]["numRows"]) if pre is not None else 0
    out: dict[int, int] = {}
    if dec is not None:
        lengths = cap.tensor(r, dec["in"]["lengths"]).reshape(-1).astype(np.int64)
        out |= {t: int(lengths[t]) for t in range(min(len(lengths), n - m)) if lengths[t] > keys}
    if pre is not None:
        span = cap.tensor(r, pre["in"]["rowEnds"]).astype(np.int64) - cap.tensor(r, pre["in"]["rowStarts"]).astype(np.int64)
        out |= {n - m + i: int(span[i]) for i in range(m) if span[i] > keys}
    return dict(sorted(out.items()))


def words(a) -> np.ndarray:
    """A tensor's elements as their bit patterns, one word each."""
    a = np.ascontiguousarray(a)
    if a.dtype.kind in "fi":
        a = a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.itemsize])
    return a.astype(np.uint64).reshape(-1)


def block_fn(block: str):
    from verity_vllm.pipeline import serving_rows as SRW
    return SRW.template(block).fn


# ---- populations: each type's instances, as (label, block, key, args(), expected()) ----------------------------------------

def quant_instances(cap: Capture):
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_C::per_token_group_fp8_quant" or not cap.opened(e):
                continue
            a = e["in"]
            if not a.get("scale_ue8m0") or a.get("group_size") != 128:
                continue
            K = a["input"]["shape"][-1]
            block = f"Fp8GroupQuantUe8m0_v1{{K={K},G=128}}"
            for t in cap.rows(r, a["input"]["shape"][0]):
                yield (f"Fp8GroupQuantUe8m0_K{K}", block, (r, e["seq"], t),
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["in"]["input"])[t]],
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["out"]["output_q"])[t], cap.tensor(r, e["out"]["output_s"])[t]])


def fp8_linear_instances(cap: Capture):
    """The block-scaled FP8 linear per output word, on the branch vLLM takes at its row count M: FlashInfer's below 32 rows,
    DeepGEMM's from 32 (deepgemm.py's module docstring), whose quantised row and scales, its inner
    ``_C::per_token_group_fp8_quant``'s outputs, are outputs of its Definition too."""
    for r in cap.ranks:
        quant = {x["parent"]: x for x in cap.events[r] if x["op"] == "_C::per_token_group_fp8_quant" and x.get("parent") is not None}
        for e in cap.events[r]:
            if e["op"] != "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm" or not cap.opened(e, weights=True):
                continue
            a = e["in"]
            M, K = a["input"]["shape"]
            if "param" not in a["weight"] or (M >= 32 and e["seq"] not in quant):
                continue
            N = a["weight"]["shape"][0]
            q = quant[e["seq"]] if M >= 32 else None
            name = "FlashInferFp8Linear" if M < 32 else "DeepGemmFp8Linear"
            block = f"{name}_v1{{K={K},N=1}}"
            for t, col in cap.cells(name, (r, e["seq"]), cap.real_rows(r, M), N):
                yield (f"{name}_K{K}", block, (r, e["seq"], t, col),
                       lambda r=r, e=e, t=t, col=col: [cap.tensor(r, e["in"]["input"])[t],
                                                       cap.tensor(r, e["in"]["weight"])[col:col + 1],
                                                       cap.tensor(r, e["in"]["weight_scale"])[col // 128:col // 128 + 1]],
                       lambda r=r, e=e, t=t, col=col, q=q: (
                           [] if q is None else [cap.tensor(r, q["out"]["output_q"])[t], cap.tensor(r, q["out"]["output_s"])[t]])
                       + [cap.tensor(r, e["out"]["ret0"])[t, col:col + 1]])


def moe_instances(cap: Capture, profile):
    """A block-FP8 `fused_moe_kernel` launch's (token, slot) pairs: the w13 launch (`MoeExpertFp8Gemm`, A one row a token) and
    the w2 launch with the routed weight (`MoeExpertFp8GemmW`, A one row a pair)."""
    from verity_vllm.program.frontend import targets as T
    for r in cap.ranks:
        router = None
        for e in cap.events[r]:
            if e["op"] in ROUTERS:
                router = e
            if e["op"] != "moe_launch" or not cap.opened(e, weights=True):
                continue
            a, cfg = e["in"], e["in"]["config"]
            routed = bool(a.get("mul_routed_weight"))
            E, N, K = a["B"]["shape"]
            S = a["A"]["shape"][0]
            G = a.get("block_shape", [128, 128])[1]
            bm, bn, kb, warps = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"], cfg.get("num_warps", 4)
            # an FP8 launch takes SWAP_AB from vLLM's `enable_swap_ab`, not from the tuned config, which names none
            swap = bm < 64 and bn >= 64
            why = T.moe_fp8_tuned_tile_refusal(profile, bm, bn, kb, warps, swap)
            if why:
                raise LookupError(f"moe_launch (seq {e['seq']}): {why}")
            spec = T.moe_expert_fp8_gemm_spec(profile, E, K, 1, G, kb, swap, BM=bm, BN=bn, WARPS=warps, routed=routed)
            topk = a["topk_weights"]["shape"][1] if routed else int(a["top_k"])
            dot = re.search(r"DOT=(\w+?)Dot", spec.id)
            kind = "MoeExpertFp8GemmW" if routed else "MoeExpertFp8Gemm"
            label = f"{kind}_K{K}_{'swap' if swap else 'noswap'}_{dot[1] if dot else 'dot'}"
            if router is None or router.get("layer") != e.get("layer"):
                raise LookupError(f"moe_launch (seq {e['seq']}): no router call ({sorted(ROUTERS)}) precedes it in its layer, "
                                  "so its pairs' experts would come from the align's schedule, which no Definition computes")
            for j, col in cap.cells(kind, (r, e["seq"]), cap.real_rows(r, S) * (1 if routed else topk), N, slots=topk):
                yield (label, spec, (r, e["seq"], j, col),
                       lambda r=r, e=e, j=j, col=col, router=router: moe_args(cap, r, e, j, col, router),
                       lambda r=r, e=e, j=j, col=col, topk=topk, N=N: [cap.tensor(r, e["out"]["C"]).reshape(-1, topk, N)[j // topk, j % topk, col:col + 1]])


#: the MoE router's calls, each with the output holding its (token, slot) experts, token-major
ROUTERS = {"_moe_C::grouped_topk": "ret1"}


def moe_args(cap: Capture, r: int, e: dict, j: int, col: int, router: dict) -> list:
    """Pair j's operands: its A row (the w13 launch reads token j // top_k's), its expert, the w2 launch's routed weight, and
    weight column `col` with its scale block. The w13 launch has no routed weight (vLLM passes ``topk_weights=None``) and
    its ``top_k`` is the slots a token. The expert is the router's output for the pair (`ROUTERS`), not the align's
    ``expert_ids``, a schedule no Definition computes: a launch that used another expert fails the match gate."""
    a = e["in"]
    routed = bool(a.get("mul_routed_weight"))
    t = j if routed else j // int(a["top_k"])
    A, As = cap.tensor(r, a["A"]), cap.tensor(r, a["A_scale"])
    B, Bs = cap.tensor(r, a["B"]), cap.tensor(r, a["B_scale"])
    ex = int(cap.tensor(r, router["out"][ROUTERS[router["op"]]]).reshape(-1)[j])
    g = [np.array([cap.tensor(r, a["topk_weights"]).reshape(-1)[j]])] if routed else []
    return [A[t], As[t], np.array([ex], np.uint64), *g, B[:, col:col + 1, :], Bs[:, col // 128:col // 128 + 1, :]]


def router_instances(cap: Capture, unstaged: dict | None = None):
    """`_moe_C::grouped_topk` per token row on `MoeRouterSigmoidBias_v1` (one group, sigmoid scores, the selection bias, the
    routed scaling inside), at the kernel its launch takes at its rows; any other configuration is counted in ``unstaged``."""
    from verity.definitions.registry import moe as M
    unstaged = {} if unstaged is None else unstaged
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_moe_C::grouped_topk" or not cap.opened(e, weights=True):
                continue
            a, sc = e["in"], e["in"]["scores"]
            rows, E, topk = sc["shape"][0], sc["shape"][-1], int(a["topk"])
            kernel = M.router_kernel(E, rows, topk)
            got = (a.get("n_group"), a.get("topk_group"), a.get("renormalize"), a.get("scoring_func"), sc["dtype"])
            if got != (1, 1, True, 1, "torch.float32") or a["bias"].get("dtype") != "torch.float32" or kernel is None:
                rec = unstaged.setdefault(f"MoeRouter_E{E}", {"population": 0, "calls": [], "why": (
                    "grouped_topk off MoeRouterSigmoidBias_v1's form: (n_group, topk_group, renormalize, scoring_func, "
                    "scores dtype) must be (1, 1, True, 1, float32) with a float32 bias, at a launch router_kernel pins")})
                rec["population"] += cap.real_rows(r, rows)
                rec["calls"].append({"rank": r, "seq": e["seq"], "args": [*got[:4], sc["dtype"], kernel]})
                continue
            scale = float(a["routed_scaling_factor"]).hex()
            label = f"MoeRouterSigmoidBias_E{E}_TOPK{topk}_{kernel}"
            block = f'MoeRouterSigmoidBias_v1{{E={E},TOPK={topk},SCALE={{"f64":"{scale}"}},KERNEL="{kernel}"}}'
            for (t,) in cap.cells(label, (r, e["seq"]), cap.real_rows(r, rows)):
                yield (label, block, (r, e["seq"], t),
                       lambda r=r, a=a, t=t: [cap.tensor(r, a["scores"])[t], cap.tensor(r, a["bias"]).reshape(-1)],
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["out"]["ret0"])[t], cap.tensor(r, e["out"]["ret1"])[t]])


def silu_mul_instances(cap: Capture):
    """`_C::silu_and_mul` per row on `SiluMul_v2`: the MoE's (a row a token and slot) and the dense MLPs'."""
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_C::silu_and_mul" or not cap.opened(e):
                continue
            x, y = e["in"]["input"], e["out"]["result"]
            I = x["shape"][-1] // 2
            if x["dtype"] != "torch.bfloat16" or y["shape"][-1] != I:
                raise LookupError(f"silu_and_mul (seq {e['seq']}): input {x['dtype']} {x['shape']}, result {y['shape']}")
            label = f"SiluMul_I{I}"
            for (t,) in cap.cells(label, (r, e["seq"]), cap.real_rows(r, x["shape"][0]), slots=cap.slots(r, x["shape"][0])):
                yield (label, f"SiluMul_v2{{I={I}}}", (r, e["seq"], t),
                       lambda r=r, x=x, t=t: [cap.tensor(r, x)[t]], lambda r=r, y=y, t=t: [cap.tensor(r, y)[t]])


def moe_sum_instances(cap: Capture):
    """`_moe_C::moe_sum` per token row on `MoeSum_v1`: the sum over its slots of the w2 launch's weighted rows."""
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_moe_C::moe_sum" or not cap.opened(e):
                continue
            x, y = e["in"]["input"], e["out"]["output"]
            if x["dtype"] != "torch.bfloat16" or len(x["shape"]) != 3 or e["in"].get("expert_map") is not None:
                raise LookupError(f"moe_sum (seq {e['seq']}): input {x['dtype']} {x['shape']}, an expert map "
                                  f"{e['in'].get('expert_map') is not None}; MoeSum_v1 sums every slot of a bf16 row")
            rows, topk, H = x["shape"]
            label = f"MoeSum_TOPK{topk}_H{H}"
            for (t,) in cap.cells(label, (r, e["seq"]), cap.real_rows(r, rows)):
                yield (label, f"MoeSum_v1{{TOPK={topk},H={H}}}", (r, e["seq"], t),
                       lambda r=r, x=x, t=t: [cap.tensor(r, x)[t]], lambda r=r, y=y, t=t: [cap.tensor(r, y)[t]])


def aligned(cap: Capture, op: str):
    """The calls of ``op`` in the opened layers, the i-th of each rank together (TP ranks run the same calls in order)."""
    per = {r: [e for e in cap.events[r] if e["op"] == op and cap.opened(e)] for r in cap.ranks}
    n = min(len(v) for v in per.values())
    for i in range(n):
        yield {r: per[r][i] for r in cap.ranks}


def fused_ar_norm_outputs(e: dict) -> tuple[dict, dict]:
    """The recorded buffers holding a ``vllm::flashinfer_trtllm_fused_allreduce_norm`` call's (new residual, normed rows) after it
    ran. vLLM d9105ea80's op hands FlashInfer ``residual_out=allreduce_in`` when ``norm_out`` is given (`fused_allreduce_rms_norm`'s
    path), else ``residual_out=residual`` and ``norm_out=allreduce_in``. Given ``norm_out``, ``residual`` is only FlashInfer's
    ``residual_in`` and holds the old residual after the call, though the op's schema marks it written."""
    if e["in"].get("norm_out") is None:
        return e["out"]["residual"], e["out"]["allreduce_in"]
    return e["out"]["allreduce_in"], e["out"]["norm_out"]


def fused_ar_norm_instances(cap: Capture):
    from verity.definitions.registry.tp_allreduce import mnnvl_grid, mnnvl_shot
    W = len(cap.ranks)
    r0 = cap.ranks[0]
    for calls in aligned(cap, "vllm::flashinfer_trtllm_fused_allreduce_norm"):
        e0 = calls[r0]
        T, H = e0["in"]["allreduce_in"]["shape"]
        half = H // 2
        shot, cluster = mnnvl_shot(T, H, W), mnnvl_grid(T, H)[1]
        new_res, normed = fused_ar_norm_outputs(e0)
        for t in cap.rows(r0, T):
            for c in range(2):
                sl = slice(c * half, (c + 1) * half)
                yield (f"AllReduceResidualMnnvl_W{W}_S{shot}", f"AllReduceResidualMnnvl_v1{{WORLD={W},N={half},SHOT={shot}}}",
                       (e0["seq"], t, c),
                       lambda calls=calls, t=t, sl=sl: [np.stack([cap.tensor(r, calls[r]["in"]["allreduce_in"])[t, sl] for r in cap.ranks]),
                                                        cap.tensor(r0, calls[r0]["in"]["residual"])[t, sl]],
                       lambda new_res=new_res, t=t, sl=sl: [cap.tensor(r0, new_res)[t, sl]])
            eps = e0["in"]["rms_eps"]
            yield (f"RmsNormMnnvl_W{W}_S{shot}",
                   f'RmsNormMnnvl_v1{{N={H},SHOT={shot},CLUSTER={cluster},EPS={{"f64":"{float(eps).hex()}"}},FMA=true}}',
                   (e0["seq"], t),
                   lambda calls=calls, new_res=new_res, t=t: [cap.tensor(r0, new_res)[t], cap.tensor(r0, calls[r0]["in"]["rms_gamma"])],
                   lambda normed=normed, t=t: [cap.tensor(r0, normed)[t]])


def allreduce_route(cap: Capture, e0: dict) -> str:
    """The communicator a plain allreduce ran on: symm-mem's when the recorder saw its multimem op inside the call. Neither
    FlashInfer's MNNVL kernel nor NCCL is an op the recorder sees, so a call without one took the route
    `CudaCommunicator.all_reduce` tries first for its bytes: FlashInfer's (``fi_ar_comm``) within `tp_allreduce.FI_MAX_BYTES`,
    else NCCL's."""
    from verity.definitions.registry.tp_allreduce import FI_MAX_BYTES
    kids = {x["op"] for x in cap.events[cap.ranks[0]] if x.get("parent") == e0["seq"]}
    if any("multimem" in k for k in kids):
        return "symm_mem_comm"
    T, H = e0["in"]["tensor"]["shape"]
    return "fi_ar_comm" if T * H * 2 <= FI_MAX_BYTES else "pynccl_comm"


def same_on_every_rank(cap: Capture, calls: dict, t: int, sl: slice) -> np.ndarray:
    """Rank 0's received row, once every rank's equals it: each rank's view of the allreduce is its own output."""
    want = cap.tensor(cap.ranks[0], calls[cap.ranks[0]]["out"]["ret0"])[t, sl]
    for r in cap.ranks[1:]:
        got = cap.tensor(r, calls[r]["out"]["ret0"])[t, sl]
        if not np.array_equal(got, want):
            raise LookupError(f"rank {r} received other words than rank {cap.ranks[0]} (seq {calls[r]['seq']}, token {t})")
    return want


def allreduce_instances(cap: Capture, unstaged: dict | None = None):
    """A plain allreduce's (token, half-row) on symm-mem's multimem route or FlashInfer's MNNVL one. A call on NCCL's is
    counted in ``unstaged`` under its label, with why, and the step's other calls are still staged."""
    from verity.definitions.registry import tp_allreduce as AR
    W = len(cap.ranks)
    unstaged = {} if unstaged is None else unstaged
    for calls in aligned(cap, "vllm::all_reduce"):
        e0 = calls[cap.ranks[0]]
        shape = e0["in"]["tensor"]["shape"]
        if len(shape) != 2:
            continue
        T, H = shape
        half = H // 2
        route, rows = allreduce_route(cap, e0), cap.real_rows(cap.ranks[0], T)
        if route == "fi_ar_comm":
            shot = AR.mnnvl_shot(T, H, W)
            for t in cap.rows(cap.ranks[0], T):
                for c in range(2):
                    sl = slice(c * half, (c + 1) * half)
                    yield (f"AllReduceMnnvl_W{W}_S{shot}", f"AllReduceMnnvl_v1{{WORLD={W},N={half},SHOT={shot}}}", (e0["seq"], t, c),
                           lambda calls=calls, t=t, sl=sl: [np.stack([cap.tensor(r, calls[r]["in"]["tensor"])[t, sl] for r in cap.ranks])],
                           lambda calls=calls, t=t, sl=sl: [same_on_every_rank(cap, calls, t, sl)])
            continue
        if route != "symm_mem_comm":
            rec = unstaged.setdefault(f"AllReduceNccl_W{W}", {"population": 0, "calls": [], "why": (
                f"vllm::all_reduce on {route}: no Definition models NCCL's all-reduce at world {W} on this GPU "
                f"(`tp_allreduce.covered` names no {route} entry at world {W})")})
            rec["population"] += 2 * rows
            rec["calls"].append({"seq": e0["seq"], "rows": T, "cols": H, "bytes": T * H * 2})
            continue
        if AR.covered("defaults", H, route, T) is None:
            past = unstaged.setdefault(f"AllReduceMultimem_W{W}", {"past_captured": []})
            past["past_captured"].append({"seq": e0["seq"], "rows": T})
        trees = multimem_trees(W, cap.tree["tree"] if cap.tree else None)
        for t in cap.rows(cap.ranks[0], T):
            for c in range(2):
                sl = slice(c * half, (c + 1) * half)
                yield (f"AllReduceMultimem_W{W}", f"AllReduceMultimem_v1{{WORLD={W},SIZE={half},TREES={trees},FTZ=false}}", (e0["seq"], t, c),
                       lambda calls=calls, t=t, sl=sl: [np.stack([cap.tensor(r, calls[r]["in"]["tensor"])[t, sl] for r in cap.ranks])],
                       lambda calls=calls, t=t, sl=sl: [same_on_every_rank(cap, calls, t, sl)])


def cublas_route(M: int, K: int, N: int, f32: bool) -> tuple[str, dict] | str:
    """The role and statics the Program binds an unbiased bf16 GEMM of M rows against a [N, K] weight at, cuBLAS's captured
    vLLM-default route at those M (for f32 out, the route `vllm_bindings.bf16_gemm.f32out_route` binds; the linear's
    `default_cublas_route` for bf16 out), or why it has none."""
    from verity.definitions.registry import bf16_gemm_check as BC
    if f32:
        phase = "f32out" if f"f32out {K} {N}" in BC.CUBLAS_ROUTES["routes"] else "router"
        kernel = BC.dispatch_sm90(M, K, N)[0]
        if phase == "router" and kernel != "cublas":
            return f"vLLM dispatches the {kernel} tier at M={M}, K={K}, N={N}, not aten.mm.dtype"
        route = BC.default_cublas_route(phase, M, K, N)
        if route is None:
            return f"no captured vLLM-default cuBLAS {phase} route for M={M}, K={K}, N={N}"
        return ("cublas_splitk_gemm_f32out" if route[0] == "splitk" else "cublas_gemm_f32out"), route[1]
    route = BC.default_cublas_route("linear", M, K, N)
    if route is None:
        return f"no captured vLLM-default cuBLAS linear route for M={M}, K={K}, N={N}"
    return ("cublas_splitk_gemm" if route[0] == "splitk" else "gemm_dot"), route[1]


def cublas_instances(cap: Capture, profile, unstaged: dict | None = None):
    """An unbiased bf16 GEMM's token row on cuBLAS: ``aten::mm`` of the activation rows against a bf16 weight's transposed view
    (GLM-5.3's router gate past the ll_bf16 tier, f32 out), or ``aten::linear`` against the weight itself (its indexer's wk +
    weights projection, which vLLM runs as ``F.linear``, bf16 out), at the Definition
    the Program binds each row's call to at the M rows cuBLAS saw, in the opened layers (the weights are past the kept small
    parameters). A call whose M has no captured route, and any other ``aten::mm``, is counted in ``unstaged`` under its label,
    with why, and the step's other calls are still staged."""
    from verity.definitions.registry import bf16_gemm_check as BC
    from verity_vllm.program.frontend import targets as T
    from verity_vllm.program.predict.common import kind
    unstaged = {} if unstaged is None else unstaged
    dot = T.tensor_core_dot(profile)
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] not in ("aten::mm", "aten::linear") or not cap.opened(e, weights=True):
                continue
            mm = e["op"] == "aten::mm"
            x, w = (e["in"].get("self"), e["in"].get("mat2")) if mm else (e["in"].get("input"), e["in"].get("weight"))
            y = e["out"].get("ret0")
            shaped = all(isinstance(v, dict) and len(v.get("shape", ())) == 2 for v in (x, w, y)) and "param" in w
            K = x["shape"][1] if shaped else None
            if mm:
                plain = shaped and w["shape"][0] == K and w.get("stride") == [1, K]
            else:
                plain = shaped and e["in"].get("bias") is None and w["shape"][1] == K and w.get("stride") in (None, [K, 1])
            if not plain or (x["dtype"], w["dtype"]) != ("torch.bfloat16",) * 2:
                name = "AtenMm_other" if mm else "AtenLinear_other"
                rec = unstaged.setdefault(name, {"population": 0, "calls": [], "why": (
                    f"an {e['op']} that is no bf16 activation against a bf16 weight's transposed view: no population reads it"
                    if mm else "an aten::linear that is no unbiased bf16 activation against a bf16 weight: no population reads it")})
                rec["population"] += 1
                rec["calls"].append({"rank": r, "seq": e["seq"]})
                continue
            (M, K), N = x["shape"], (w["shape"][1] if mm else w["shape"][0])
            f32 = y["dtype"] == "torch.float32"
            rows, route = cap.real_rows(r, M), cublas_route(M, K, N, f32)
            if isinstance(route, str):
                rec = unstaged.setdefault(f"CublasGemm{'F32Out' if f32 else 'Bf16'}_K{K}_N{N}", {"population": 0, "calls": [], "why": (
                    f"{route} (`bf16_gemm_check.CUBLAS_ROUTES` maps M to {BC.CUBLAS_ROUTES['m_max']})")})
                rec["population"] += rows
                rec["calls"].append({"rank": r, "seq": e["seq"], "rows": M})
                continue
            role, statics = route
            spec = kind(role, K=K, N=N, DOT=dot, **statics)
            label = f"{spec.id.split('{')[0]}_K{K}_N{N}" + (f"_S{statics['SPLIT_K']}" if "SPLIT_K" in statics else "")
            for t in cap.rows(r, M):
                yield (label, spec, (r, e["seq"], t),
                       lambda r=r, x=x, w=w, t=t, mm=mm: [cap.tensor(r, x)[t],
                                                          np.ascontiguousarray(cap.tensor(r, w).T if mm else cap.tensor(r, w))],
                       lambda r=r, y=y, t=t: [cap.tensor(r, y)[t]])


def token_rows(cap: Capture, r: int, rows: int, prefill: int) -> list[tuple[int, int]]:
    """A call's (row, token) pairs over the step's real tokens: a row a token (`real_rows`), or, for a call whose rows are the
    DSA indexer's decode-padded layout (its decode rows padded, then the step's ``prefill`` rows), the real decode rows and the
    prefill rows, padding left out."""
    n, padded = cap.tokens(r), cap.padded(r)
    dec, dec_padded = n - prefill, rows - prefill
    if prefill and rows != n and not (padded and rows % padded == 0) and 0 <= dec < dec_padded:
        return [(i, i) for i in range(dec)] + [(dec_padded + j, dec + j) for j in range(prefill)]
    return [(t, t) for t in range(cap.real_rows(r, rows))]


def bmm_instances(cap: Capture, profile, unstaged: dict | None = None):
    """MLA's latent absorbs, ``aten::bmm`` of a row's heads against ``W_UK_T`` or ``W_UV`` in bf16, per row on the Program's
    unsplit `BatchedGemm_v1` (`predict.dsa`'s ``absorb``), ``w[b][n][k]`` the weight at ``[b, k, n]``, in the opened layers;
    any other ``aten::bmm`` is counted in ``unstaged``."""
    from verity_vllm.program.frontend import targets as T
    from verity_vllm.program.predict.common import kind
    unstaged = {} if unstaged is None else unstaged
    dot = T.tensor_core_dot(profile)
    for r in cap.ranks:
        prefill = next((int(e["in"]["numRows"]) for e in cap.events[r] if e["op"] == "_C::top_k_per_row_prefill"), 0)
        for e in cap.events[r]:
            if e["op"] != "aten::bmm" or not cap.opened(e, weights=True):
                continue
            x, w, y = e["in"].get("self"), e["in"].get("mat2"), e["out"].get("ret0")
            ok = (all(isinstance(v, dict) and len(v.get("shape", ())) == 3 and v.get("dtype") == "torch.bfloat16" for v in (x, w, y))
                  and "param" in w and w["shape"][:2] == [x["shape"][0], x["shape"][2]]
                  and y["shape"] == [x["shape"][0], x["shape"][1], w["shape"][2]])
            if not ok:
                rec = unstaged.setdefault("AtenBmm_other", {"population": 0, "calls": [], "why": (
                    "an aten::bmm that is no bf16 batch of rows against a bf16 weight: no population reads it")})
                rec["population"] += 1
                rec["calls"].append({"rank": r, "seq": e["seq"]})
                continue
            (B, R, K), N = x["shape"], w["shape"][2]
            spec = kind("batched_gemm", B=B, K=K, N=N, DOT=dot)
            for i, t in token_rows(cap, r, R, prefill):
                if cap.only is not None and t not in cap.only:
                    continue
                yield (f"BatchedGemm_B{B}_K{K}_N{N}", spec, (r, e["seq"], t),
                       lambda r=r, x=x, w=w, i=i: [cap.tensor(r, x)[:, i, :], np.ascontiguousarray(cap.tensor(r, w).transpose(0, 2, 1))],
                       lambda r=r, y=y, i=i: [cap.tensor(r, y)[:, i, :]])


def rms_norm_instances(cap: Capture):
    """``_C::rms_norm`` (the first layer's input norm, before any residual) per token row on `RMSNormCudaVector8_v1` at its
    launch block."""
    from verity.definitions.registry.cuda_rmsnorm import launch_block
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_C::rms_norm" or not cap.opened(e):
                continue
            a = e["in"]
            T, N = a["input"]["shape"]
            block = f'RMSNormCudaVector8_v1{{N={N},EPS={{"f64":"{float(a["epsilon"]).hex()}"}},BLOCK={launch_block(T, N, fused=False)}}}'
            for t in cap.rows(r, T):
                yield (f"RMSNormCudaVector8_N{N}", block, (r, e["seq"], t),
                       lambda r=r, a=a, t=t: [cap.tensor(r, a["input"])[t], cap.tensor(r, a["weight"])],
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["out"]["result"])[t]])


def add_instances(cap: Capture, unstaged: dict | None = None):
    """``aten::add`` of two bf16 rows (the MoE block's shared and routed experts' outputs) per token row on `ResidualAdd_v1`,
    torch's f32 add rounded to bf16; any other ``aten::add`` is counted in ``unstaged``."""
    unstaged = {} if unstaged is None else unstaged
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "aten::add" or not cap.opened(e):
                continue
            x, o, y = e["in"].get("self"), e["in"].get("other"), e["out"].get("ret0")
            if not (all(isinstance(v, dict) and len(v.get("shape", ())) == 2 and v.get("dtype") == "torch.bfloat16" for v in (x, o, y))
                    and x["shape"] == o["shape"] == y["shape"] and e["in"].get("alpha") in (None, 1)):
                rec = unstaged.setdefault("AtenAdd_other", {"population": 0, "calls": [], "why": (
                    "an aten::add that is no sum of two bf16 rows of one shape: no population reads it")})
                rec["population"] += 1
                rec["calls"].append({"rank": r, "seq": e["seq"]})
                continue
            T, N = x["shape"]
            for t in cap.rows(r, T):
                yield (f"ResidualAdd_N{N}", f"ResidualAdd_v1{{N={N}}}", (r, e["seq"], t),
                       lambda r=r, x=x, o=o, t=t: [cap.tensor(r, x)[t], cap.tensor(r, o)[t]],
                       lambda r=r, y=y, t=t: [cap.tensor(r, y)[t]])


def fused_add_rms_norm_instances(cap: Capture):
    from verity.definitions.registry.cuda_rmsnorm import launch_block
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] != "_C::fused_add_rms_norm" or not cap.opened(e):
                continue
            a = e["in"]
            T, N = a["input"]["shape"]
            eps = float(a["epsilon"]).hex()
            block = f'RMSNormFusedCudaVector8_v3{{N={N},EPS={{"f64":"{eps}"}},BLOCK={launch_block(T, N, fused=True)}}}'
            for t in cap.rows(r, T):
                yield (f"RMSNormFusedCudaVector8_N{N}", block, (r, e["seq"], t),
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["in"]["input"])[t], cap.tensor(r, e["in"]["residual"])[t],
                                              cap.tensor(r, e["in"]["weight"])],
                       lambda r=r, e=e, t=t: [cap.tensor(r, e["out"]["input"])[t], cap.tensor(r, e["out"]["residual"])[t]])


def fa3_calls(cap: Capture):
    """FA3's sparse MLA calls in the opened layers: (rank, the op's event, the tap's event or None, each token's valid keys and
    split count)."""
    from verity.definitions.registry import sparse_mla_split as SP
    for r in cap.ranks:
        for e in cap.events[r]:
            if e["op"] == "_vllm_fa3_C::fwd" and e["in"].get("q_v") is not None:
                if not cap.opened(e):
                    continue
                o = e["out"]
                if "sha512" not in (o.get("ret2") or {}) or "sha512" not in (o.get("ret3") or {}):
                    continue
                tap = None
                for x in cap.events[r][cap.events[r].index(e) + 1:]:
                    if x["op"] == "fa3_sparse_mla":
                        tap = x
                        break
                    if x["op"] == "_vllm_fa3_C::fwd":
                        break
                T = e["in"]["q"]["shape"][0]
                cu = cap.tensor(r, e["in"]["cu_seqlens_q"]).astype(np.int64)
                if list(cu) != list(range(T + 1)):
                    raise LookupError(f"rank {r} seq {e['seq']}: cu_seqlens_q is not one query per sequence")
                valid = [int(x) for x in cap.tensor(r, e["in"]["seqused_k"])]
                bound = o["ret2"]["shape"][0]
                yield r, e, tap, valid, SP.dynamic_num_splits(valid, NUM_SM, bound)


def mla_combine_instances(cap: Capture):
    for r, e, _, valid, nss in fa3_calls(cap):
        bound, H, T, DV = e["out"]["ret2"]["shape"]
        for t in range(min(cap.tokens(r), T)):
            if cap.only is not None and t not in cap.only:
                continue
            from verity.definitions.registry import sparse_mla_split as SP
            ns = len(SP.split_blocks(valid[t], nss[t]))
            if ns < 2:
                continue
            for (h,) in cap.cells(f"SparseMlaCombine/{t}", (r, e["seq"]), H):
                yield (f"SparseMlaCombine_N{ns}", f'SparseMlaCombine_v1{{NSPLIT={ns},DV={DV},INV={{"fn":"Fa3InvSum_v1"}}}}',
                       (r, e["seq"], t, h),
                       lambda r=r, e=e, t=t, h=h, ns=ns: [cap.tensor(r, e["out"]["ret3"])[:ns, h, t],
                                                          cap.tensor(r, e["out"]["ret2"])[:ns, h, t, :]],
                       lambda r=r, e=e, t=t, h=h: [cap.tensor(r, e["out"]["ret0"])[t, h, :]])


def mla_split_part_instances(cap: Capture):
    """Each split of a token whose KV rows the tap kept (``kv_gathered_rows``, else its first ``kv_gathered_tokens``), per head:
    its keys' rows in, FA3's (o_partial, lse) for that split out."""
    from verity.definitions.registry import sparse_mla_split as SP
    for r, e, tap, valid, nss in fa3_calls(cap):
        if tap is None or "sha512" not in (tap["in"].get("kv_gathered") or {}):
            continue
        bound, H, T, DV = e["out"]["ret2"]["shape"]
        DR = e["in"]["q"]["shape"][-1]
        BN = SP.BLOCK_N
        scale = float(tap["in"]["softmax_scale"]).hex()
        kept = tap["in"].get("kv_gathered_rows") or list(range(int(tap["in"]["kv_gathered_tokens"])))
        for i, t in enumerate(kept):
            if t >= min(cap.tokens(r), T) or (cap.only is not None and t not in cap.only):
                continue
            splits = SP.split_blocks(valid[t], nss[t])
            if len(splits) < 2:
                continue
            for s, h in cap.cells(f"SparseMlaSplitPart/{t}", (r, e["seq"]), len(splits), H):
                b0, b1 = splits[s]
                lo, hi = b0 * BN, min(b1 * BN, valid[t])
                block = (f'SparseMlaSplitPart_v1{{T={hi - lo},DR={DR},DV={DV},BN={BN},DOT={{"fn":"HopperBF16WgmmaDot16_v1"}},'
                         f'INV={{"fn":"Fa3InvSum_v1"}},SCALE={{"f64":"{scale}"}}}}')
                yield (f"SparseMlaSplitPart_T{hi - lo}", block, (r, e["seq"], t, h, s),
                       lambda r=r, e=e, tap=tap, i=i, t=t, h=h, lo=lo, hi=hi: [
                           cap.tensor(r, tap["in"]["q_v"])[t, h], cap.tensor(r, tap["in"]["q"])[t, h],
                           cap.part(r, tap["in"]["kv_gathered"], (i, slice(lo, hi)))],
                       lambda r=r, e=e, t=t, h=h, s=s: [cap.tensor(r, e["out"]["ret2"])[s, h, t, :],
                                                        cap.tensor(r, e["out"]["ret3"])[s, h, t:t + 1]])


# ---- sample, gate, stage ---------------------------------------------------------------------------------------------------

def draw(coins: bytes, label: str, n: int, k: int) -> list[int]:
    """``k`` distinct indices of ``n`` from the capture's coins (all of them when ``n`` <= ``k``), in the order sampled, so that
    any prefix is itself a uniform sample (a11's ``--per-kind`` takes one)."""
    if n <= k:
        return list(range(n))
    out: list[int] = []
    i = 0
    while len(out) < k:
        j = int.from_bytes(hashlib.sha512(coins + f"stage/{label}/{i}".encode()).digest(), "big") % n
        if j not in out:
            out.append(j)
        i += 1
    return out


def gate_instance(spec, key, args_of, want_of, full_gates: int) -> tuple[dict, dict | None]:
    """One instance through the match gate: its Definition evaluated on the captured inputs against the captured outputs."""
    import a11
    try:
        args = [words(x) for x in args_of()]
        want = np.concatenate([words(x) for x in want_of()])
        ev = a11.evaluate_units(spec, spec, args, 1, full_gates)
        got = np.asarray([int(x) for x in ev["call_out"]], dtype=np.uint64)
        same = got.shape == want.shape and bool(np.array_equal(got, want))
        first = None if same else int(np.nonzero(got != want)[0][0]) if got.shape == want.shape else "shape"
        return {"key": list(key), "agree": same, "words": int(want.size), "first_diff": first,
                **({} if same or first == "shape" else {"got": int(got[first]), "want": int(want[first]),
                                                         "differ": int(np.count_nonzero(got != want))})}, ev
    except Exception as ex:  # noqa: BLE001
        return {"key": list(key), "agree": None, "error": f"{type(ex).__name__}: {str(ex)[:300]}",
                "trace": traceback.format_exc()[-800:]}, None


#: the recorded ops some population reads; every other recorded op no such op encloses is listed by ``--every-call`` as not
#: gated (`op_coverage`)
GATED_OPS = {"_C::per_token_group_fp8_quant", "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm", "moe_launch",
             "vllm::flashinfer_trtllm_fused_allreduce_norm", "vllm::all_reduce", "_C::fused_add_rms_norm", "_vllm_fa3_C::fwd",
             "fa3_sparse_mla", "aten::mm", "aten::linear", "aten::bmm", "_moe_C::grouped_topk", "_C::silu_and_mul",
             "_moe_C::moe_sum", "_C::rms_norm", "aten::add"}


def _gated(e: dict) -> bool:
    """Whether a population reads the recorded call ``e``: its op is in `GATED_OPS`, and an FA3 call is its sparse MLA form
    (`fa3_calls`: with ``q_v``), never the dense prefill attention, which no population reads."""
    return e["op"] in GATED_OPS and (e["op"] != "_vllm_fa3_C::fwd" or e["in"].get("q_v") is not None)


#: recorded ops the Program binds to no value, each with the rule that says so; ``--every-call`` lists them apart from the ops
#: not gated
SCHEDULE_OPS = {"_moe_C::moe_align_block_size": (
    "a schedule marker (verity_vllm's MoeAlignBlockSizeRule): the fused MoE launches use its sorted order only for addressing, "
    "and the Program is the unpermuted pair set, which the MoE launches' per-pair gates sample")}
#: inputs a population reads that the step computes but no recorded call of it outputs (trace gaps), each counted at the parts
#: staged instances read (`Capture.part`), else at its width, and printed as gap bits, not charged: wiring it to what
#: computes it takes it to 0. (the recorded op, its argument) -> why
GAPS = {("fa3_sparse_mla", "kv_gathered"): (
    "the K/V rows a split reads, each the output of a cache write, this step's or an earlier one's: the step's record holds "
    "the cache by shape only, so a row is a gap until it is wired to its write or to the window's carried cache"),
        ("fa3_sparse_mla", "block_table"): (
    "the global slot each gathered K/V row is read from, which vLLM computes from the indexer's top-k through the request's "
    "page table (`CONVERT`), read only where the capture doesn't record that call and commit its batch: with both, each row is "
    "wired by its request and position, as the Program reads it, and no entry is read. Once a row's write and its read are both computed "
    "from one page table, its physical blocks reach no value; a prefix-cache hit, the one choice in it that does, is the "
    "schedule's advice")}
#: inputs a population reads that nothing computes (a free choice: the schedule, a prefix-cache hit, an allocator's choice),
#: each charged as advice at the parts staged instances read, else at its width; no staged population reads one today
ADVICE: dict[tuple[str, str], str] = {}
#: `GAPS` inputs whose rows are gathered from the paged cache through the tap's page table, each row wired to what wrote it
#: (`Wires.gathered`): the step's cache write, else an earlier step's (the window's `HISTORY_PORT`)
GATHERS = {("fa3_sparse_mla", "kv_gathered")}
#: the source kinds of a `GAPS` and an `ADVICE` input, each counted at the parts of it staged instances read
UNRECORDED = ("gap", "advice")
#: the call that writes the step's K/V cache rows (`capture.py`'s ``PY_KERNELS``): its op, its slot map, and the outputs a cache
#: row concatenates, in the row's word order
CACHE_WRITE = ("fused_norm_rope", "slot_mapping", ("kv_c_out", "k_pe_out"))
#: the paged MLA cache's block size: a request's position p sits at slot b * KV_BLOCK + p % KV_BLOCK of its block b
KV_BLOCK = 64
#: the call that turns each token's top-k positions into the global slots the tap reads, through its request's page table
#: (`capture.py`'s ``PY_KERNELS``; `convert_rows`): its inputs name each gathered row's request and position
CONVERT = "triton_convert_req_index_to_global_index"
#: the previous record's port that carries the K/V rows earlier steps wrote: the one-stage verifier holds a window's carried
#: value to it (R7's port-only form)
HISTORY_PORT = "edge.kv"
HISTORY_WHY = ("K/V rows an earlier step wrote: the window's carried cache, held to the previous record's port of this name, "
               "not chosen by the prover")
SELECTION_WHY = ("a row past the top-k's keys reads its rows through DsaWaksmanRows_v1, whose switch settings are a run input "
                 "the prover brings: free bits until the settings are proved unique for the checked order")
#: the indexer's top-k ops, whose shared output each later layer's page table converts
TOPK_OPS = ("_C::cooperative_topk", "_C::top_k_per_row_prefill")
#: the candidates (latest first) and the offsets in each that `Wires.view` tries before it calls an input unwired
VIEW_CANDIDATES, VIEW_OFFSETS = 64, 256
_EVERY: dict = {}


class Unwired(LookupError):
    """A value an instance reads that is no recorded call's output, no parameter and no `GAPS` or `ADVICE` input."""


def _part(shape: list[int], key) -> list[list[int]]:
    """``key`` (ints and unit-step slices over the leading axes of a tensor of ``shape``) as [start, stop) per axis."""
    out = []
    for n, k in zip(shape, key if isinstance(key, tuple) else (key,)):
        if isinstance(k, slice):
            lo, hi, step = k.indices(n)
            if step != 1:
                raise ValueError(f"a part's slice steps by 1, not {step}")
            out.append([lo, max(lo, hi)])
        else:
            i = int(k) + (n if int(k) < 0 else 0)
            out.append([i, i + 1])
    return out


def _read_bits(width: int, shape: list[int] | None, parts: list | None) -> int:
    """The bits read of a `GAPS` or `ADVICE` input or of the carried cache: its ``width`` in bits, or with ``parts``
    (`Capture.part`) the elements their union covers."""
    if parts is None or not shape:
        return width
    d = max(len(p) for p in parts)
    mask = np.zeros(shape[:d], bool)
    for p in parts:
        mask[tuple(slice(lo, hi) for lo, hi in p)] = True
    return int(mask.sum()) * (width // int(np.prod(shape[:d], dtype=np.int64)))


def _row_parts(cells: list[tuple[int, int]]) -> list[list[list[int]]]:
    """(token, row) cells as parts (`_part`): one per run of consecutive rows of a token."""
    out: list = []
    for i, j in sorted(cells):
        if out and out[-1][0] == [i, i + 1] and out[-1][1][1] == j:
            out[-1][1][1] = j + 1
        else:
            out.append([[i, i + 1], [j, j + 1]])
    return out


def own_rows(row: np.ndarray, n: int, block: int) -> set[int] | None:
    """The rows of a page-table row that can hold its token's own position, for a row that reads every key of its token: its
    ``n`` keys' global slots in any order, then -1. By the paged layout (a request's position p at slot b * block + p % block
    of its block b) the keys fill whole blocks from residue 0 and at most one block partly, which holds the newest position at
    its highest residue: the row reading that slot when ``n`` isn't a multiple of ``block``, else every row at a block's last
    residue. None where the row breaks the layout."""
    s = row[:n].astype(np.int64)
    if (row[n:] != -1).any() or (s < 0).any() or len(set(s.tolist())) != n:
        return None
    blocks: dict[int, set[int]] = {}
    for x in s.tolist():
        blocks.setdefault(x // block, set()).add(x % block)
    if any(res != set(range(len(res))) for res in blocks.values()):
        return None
    part = [b for b, res in blocks.items() if len(res) < block]
    if len(part) != (1 if n % block else 0):
        return None
    hit = s == part[0] * block + n % block - 1 if part else s % block == block - 1
    return {int(j) for j in np.flatnonzero(hit)}


def convert_rows(topk: np.ndarray, req: np.ndarray, table: np.ndarray, block: int, stride: int) -> tuple[list, np.ndarray, np.ndarray]:
    """`CONVERT` as GLM-5.3 serves it (vLLM's ``sparse_utils.py``: one tile a row, valid counts returned, no prefill
    workspace or DCP): token t's top-k entry p is kept where p >= 0 and its block p // ``block`` is in the request's row of
    ``table``, as ``table[req[t], p // block] * stride + p % block`` in int32; the kept entries move to the front in order
    and the rest are -1. Returns each token's kept positions, the slots and each row's count of kept entries."""
    topk, req = np.asarray(topk, np.int64), np.asarray(req, np.int64)
    nb = table.shape[1]
    blk = np.where(topk >= 0, topk // block, -1)
    ok = (topk >= 0) & (blk < nb)
    g = table[req[:, None], np.clip(blk, 0, nb - 1)].astype(np.int64) * stride + topk % block
    g = (g & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    slots = np.full(topk.shape, -1, np.int32)
    kept = []
    for t in range(topk.shape[0]):
        slots[t, :int(ok[t].sum())] = g[t][ok[t]]
        kept.append(topk[t][ok[t]])
    return kept, slots, ok.sum(axis=1).astype(np.int32)


def _runs(xs: list[int]) -> list[list[int]]:
    """Sorted ``xs`` as [lo, hi) runs of consecutive values."""
    out: list[list[int]] = []
    for x in sorted(xs):
        if out and out[-1][1] == x:
            out[-1][1] = x + 1
        else:
            out.append([x, x + 1])
    return out


def traced(cap: Capture, args_of, reads: list):
    """``args_of`` with each recorded tensor it reads appended to ``reads`` (`Capture.reads`)."""
    def run():
        cap.reads = reads
        try:
            return args_of()
        finally:
            cap.reads = None
    return run


class Wires:
    """Where each value an instance reads comes from, under the step's root: a staged unit's input is wired to the recorded call
    that output it (``block`` when a population reads that call or one enclosing it, `_gated`; ``call`` when none does:
    committed but not proved), read from a parameter held to ``params.json`` (``registered``), or a `GAPS` input (``gap``) or
    an `ADVICE` input (``advice``) counted at the parts of it read (`of`), whose gathered rows are wired row by row
    (`gathered`: the step's cache write, ``history``, ``selection``). An input of a call is wired to the latest earlier call
    on its rank (one enclosing it excepted, whose outputs come after) that output the same bytes, passing over one that only
    passed them on from its own input (`passes`), else that output a tensor it is a strided view of (its recorded strides
    from some offset), else to the latest earlier calls whose outputs are strided views into it from its real rows, together
    each element of those once (`cover`: an ``out=`` buffer a later call reads whole, or one several calls wrote in parts, a
    source each, padding rows written or not); an output read is its own call's. Anything else is `Unwired`."""

    def __init__(self, capture: Capture) -> None:
        self.capture = capture
        self.at: dict[int, tuple[int, int, str, str]] = {}
        self.made: dict[tuple[int, str], list[tuple[int, str]]] = {}
        self.outs: dict[int, list[tuple[int, str, dict]]] = {}
        self.found: dict[int, dict | list[dict] | Unwired] = {}
        self.by_seq = {r: {e["seq"]: e for e in capture.events[r]} for r in capture.ranks}
        for r in capture.ranks:
            for e in capture.events[r]:
                for side in ("in", "out"):
                    for name, ref in _refs(e.get(side) or {}):
                        self.at[id(ref)] = (r, e["seq"], side, name)
                        if side == "out" and "sha512" in ref:
                            self.made.setdefault((r, ref["sha512"]), []).append((e["seq"], name))
                            self.outs.setdefault(r, []).append((e["seq"], name, ref))

    def enclosing(self, r: int, seq: int) -> list[dict]:
        out, x = [], self.by_seq[r].get(self.by_seq[r][seq].get("parent"))
        while x is not None:
            out.append(x)
            x = self.by_seq[r].get(x.get("parent"))
        return out

    def of(self, reads: list) -> list[dict]:
        """Each distinct reference in ``reads`` (`traced`), with its sources (one each, or one per call that wrote a part of
        it: `cover`); a `GAPS` or `ADVICE` source names the ``parts`` of it the reads took (`Capture.part`) and is counted at
        their union, unless a read took it whole."""
        order: dict[int, tuple[int, dict]] = {}
        parts: dict[int, list | None] = {}
        for r, ref, *part in reads:
            order.setdefault(id(ref), (r, ref))
            got = parts.get(id(ref), [])
            parts[id(ref)] = None if not part or got is None else got + part
        out = []
        for k, (r, ref) in order.items():
            s = self.source(r, ref)
            if isinstance(s, list):
                out += s
                continue
            if s["kind"] in UNRECORDED and parts[k] is not None:
                if (s["op"], s["in"]) in GATHERS:
                    out += self.gathered(r, s, ref, parts[k])
                    continue
                s = {**s, "parts": parts[k], "bits": _read_bits(s["width"], s["shape"], parts[k])}
            out.append(s)
        return out

    def gathered(self, r: int, adv: dict, ref: dict, parts: list) -> list[dict]:
        """The sources of the rows ``parts`` read of a `GATHERS` input (rows of a token, ``[[i, i + 1], [lo, hi]]``), each row by
        the global slot the tap's page table names for it: the step's cache write (`CACHE_WRITE`, ``rows``: [part, row, its
        row] when the write put it at that slot, its bytes checked), else an earlier step's (``history``, `HISTORY_PORT`); the
        page-table entries those rows are read through are the tap's ``block_table``'s source, counted at the entries read. A
        row neither is established for stays a ``gap``. Without the step's cache write recorded, a token whose request
        schedules only it (`single_tokens`) writes only its own position this step, so of a row reading every key of it (fewer
        than the top-k) the rows that can't hold that position (`own_rows`) are history. With the write's positions recorded, a
        token past the top-k's keys reads its rows through `DsaWaksmanRows_v1`, whose settings are free (``selection``, once
        per token and top-k); without them such a token's rows stay gaps. With the step's `CONVERT` call recorded and its batch
        committed as well, each row is wired by its request and position instead (`by_position`), and no page-table entry is
        read."""
        rec, tap = self.capture, self.by_seq[r][adv["seq"]]
        table, used = tap["in"].get("block_table"), tap["in"].get("seqused_k")
        if not all(isinstance(x, dict) and rec.kept(r, x) for x in (table, used)):
            return [{**adv, "parts": parts, "bits": _read_bits(adv["width"], adv["shape"], parts)}]
        from verity.definitions.registry.dsa_topk import waksman_switches
        kept = tap["in"].get("kv_gathered_rows") or list(range(int(tap["in"]["kv_gathered_tokens"])))
        page, used = rec.tensor(r, table), rec.tensor(r, used).astype(np.int64)
        topk, width = page.shape[1], adv["shape"][-1]
        write = self.cache_write(r, tap)
        conv = self.conversion(r, tap) if write is not None and write[2] is not None else None
        if conv is not None:
            return self.by_position(r, adv, ref, parts, tap, conv, write)
        single = self.single_tokens(r, tap) if write is None else set()
        read = None
        wrote: list = []
        history: list = []
        left: list = []
        via: list = []
        picks: dict = {}
        for p in parts:
            (i0, i1), (lo, hi) = p if len(p) == 2 else (p[0], [0, adv["shape"][1]])
            for i in range(i0, i1):
                t, n = kept[i], int(used[kept[i]])
                own = own_rows(page[t], n, KV_BLOCK) if write is None and t in single and n < topk else None
                for j in range(lo, hi):
                    s = int(page[t, j])
                    if s < 0 or j >= n or (write is None and (own is None or j in own)):
                        left.append((i, j))
                        continue
                    via.append((t, j))
                    if write is None or s not in write[1]:
                        history.append((i, j))
                    else:
                        ev, tw = write[0], write[1][s]
                        if read is None:
                            read = rec.tensor(r, ref)
                        if not np.array_equal(read[i, j], write[3][tw]):
                            raise Unwired(f"rank {r} {tap['op']} (seq {tap['seq']}) reads {adv['in']}[{i}] row {j} at slot {s}, "
                                          f"which {ev['op']} (seq {ev['seq']}) wrote from its row {tw}, but not those bytes")
                        wrote.append([i, j, tw])
                if write is not None and n >= topk:
                    keys = write[2]
                    if keys is None:
                        raise Unwired(f"rank {r} {tap['op']} (seq {tap['seq']}): token {t} reads {topk} rows, and its keys (the "
                                      f"cache write's positions) aren't recorded, so whether a selection routes them isn't known")
                    if keys[t] > topk:
                        k = self.topk_seq(r, tap)
                        picks[t] = {"kind": "selection", "rank": r, "seq": k, "token": t,
                                    "via": f"DsaWaksmanRows_v1{{N={keys[t]},K={topk},W={width},E=16}}",
                                    "bits": waksman_switches(keys[t], topk), "why": SELECTION_WHY}
        out = []
        if wrote:
            ev = write[0]
            out.append({**self.call(r, ev["seq"], "+".join(CACHE_WRITE[2])),
                        "into": {"seq": tap["seq"], "op": tap["op"], "in": adv["in"]}, "rows": wrote})
        if history:
            hp = _row_parts(history)
            out.append({**{k: adv[k] for k in ("rank", "seq", "op", "in", "shape", "width")}, "kind": "history",
                        "port": HISTORY_PORT, "parts": hp, "bits": _read_bits(adv["width"], adv["shape"], hp), "why": HISTORY_WHY})
        if left:
            lp = _row_parts(left)
            out.append({**adv, "parts": lp, "bits": _read_bits(adv["width"], adv["shape"], lp)})
        if via:
            src = self.source(r, table)
            if isinstance(src, dict) and src["kind"] in UNRECORDED:
                vp = _row_parts(via)
                src = {**src, "parts": vp, "bits": _read_bits(src["width"], src["shape"], vp)}
            out += src if isinstance(src, list) else [src]
        return out + list(picks.values())

    def conversion(self, r: int, tap: dict) -> tuple[np.ndarray, list] | None:
        """From the recorded `CONVERT` call whose outputs the tap reads as its page table and valid counts: the request of each
        of the step's tokens, an index into the committed batch's request ids (`Capture.req_ids`) laid out by their scheduled
        counts in that order, as vLLM lays out ``req_id_per_token``; and the top-k positions each token's row keeps, in order
        (`convert_rows`). None where no such call is recorded with its inputs kept, or the rank root doesn't commit the batch.
        Refuses a call that takes one of the batch's tokens as another request than that layout does (the attention call's
        tokens lead the step; any past the batch's are padding, which `by_position` wires no row of), whose outputs aren't
        `convert_rows` of its inputs, or whose counts aren't the tap's ``seqused_k``, and a batch whose request ids aren't its
        scheduled requests, each once."""
        rec, table, used = self.capture, tap["in"]["block_table"], tap["in"]["seqused_k"]
        ev = next((self.by_seq[r][s] for s, o in reversed(self.made.get((r, table.get("sha512")), []))
                   if s < tap["seq"] and o == "r0" and self.by_seq[r][s]["op"] == CONVERT), None)
        ids = rec.req_ids.get(r)
        if ev is None or ids is None:
            return None
        a = ev["in"]
        if not all(isinstance(a.get(k), dict) and rec.kept(r, a[k]) for k in ("req_id", "block_table", "token_indices")):
            return None
        where = f"rank {r} {CONVERT} (seq {ev['seq']})"
        if (ev["out"].get("r1") or {}).get("sha512") != used.get("sha512"):
            raise Unwired(f"{where}: {tap['op']} (seq {tap['seq']}) reads its slots but not its valid counts as seqused_k")
        if a.get("HAS_PREFILL_WORKSPACE"):
            raise Unwired(f"{where} maps prefill tokens to a workspace, which convert_rows doesn't model")
        block = int(a.get("BLOCK_SIZE") or KV_BLOCK)
        stride = int(a.get("BLOCK_STRIDE_ROWS") or block)
        sched = rec.meta[r]["scheduled"]
        if len(set(ids)) != len(ids) or set(ids) != set(sched):
            raise Unwired(f"rank {r}: the committed batch's request ids aren't its scheduled requests, each once")
        layout = np.repeat(np.arange(len(ids)), [int(sched[x]) for x in ids])
        req = rec.tensor(r, a["req_id"]).reshape(-1).astype(np.int64)
        if (t := next((t for t in range(min(len(req), len(layout))) if req[t] != layout[t]), None)) is not None:
            raise Unwired(f"{where} takes token {t} as request {int(req[t])}, but the committed batch puts it at request "
                          f"{int(layout[t])} ({ids[int(layout[t])]})")
        kept, slots, counts = convert_rows(rec.tensor(r, a["token_indices"]), req, rec.tensor(r, a["block_table"]), block, stride)
        if not (np.array_equal(slots, rec.tensor(r, table)) and np.array_equal(counts, rec.tensor(r, used).reshape(-1))):
            raise Unwired(f"{where} output slots or counts that its inputs don't give (convert_rows)")
        return layout, kept

    def by_position(self, r: int, adv: dict, ref: dict, parts: list, tap: dict, conv: tuple, write: tuple) -> list[dict]:
        """`gathered` where the step records its `CONVERT` call, its cache write's positions and its committed batch. Each row a
        token reads is the position its top-k row keeps there, of its own request (`conversion`'s layout of the batch), which
        is how the Program reads it. That position is wired to the step's write of it (its bytes checked), or, below the
        request's first position this step, to the carried cache (``history``, naming each token's request and positions). No
        page-table entry is read. The served slot of each row is held to that wiring: the write's slot for a position this
        step wrote, else no slot this step wrote. Refuses a token past the batch's (padding), a valid count other than min(the
        token's keys, the top-k), a row at most the top-k's keys long that doesn't read its keys in order (the Program's
        public map), a longer one that isn't a selection of its keys (which `DsaWaksmanRows_v1`'s settings, charged once per
        token, choose), and a position the step computes whose write isn't recorded."""
        from verity.definitions.registry.dsa_topk import waksman_switches
        rec, (req, rows) = self.capture, conv
        wev, slots, keys, made = write
        page, used = rec.tensor(r, tap["in"]["block_table"]), rec.tensor(r, tap["in"]["seqused_k"]).reshape(-1)
        topk, width = page.shape[1], adv["shape"][-1]
        kept = tap["in"].get("kv_gathered_rows") or list(range(int(tap["in"]["kv_gathered_tokens"])))
        names = rec.req_ids[r]
        where = f"rank {r} {tap['op']} (seq {tap['seq']})"
        slot_of = {tw: s for s, tw in slots.items()}
        mine = {(int(req[tw]), int(keys[tw]) - 1): tw for tw in slot_of if tw < len(req)}
        first: dict[int, int] = {}
        for q, k in zip(req.tolist(), keys.tolist()):
            first[q] = min(first.get(q, k - 1), k - 1)
        read = None
        wrote, history, left = [], [], []
        at: dict[int, tuple] = {}
        picks: dict = {}
        for p in parts:
            (i0, i1), (lo, hi) = p if len(p) == 2 else (p[0], [0, adv["shape"][1]])
            for i in range(i0, i1):
                t = kept[i]
                if t >= len(req):
                    raise Unwired(f"{where}: token {t} is past the committed batch's {len(req)} tokens, so it has no request")
                n, q, ps = int(keys[t]), int(req[t]), rows[t]
                if int(used[t]) != min(n, topk):
                    raise Unwired(f"{where}: token {t} at position {n - 1} has {int(used[t])} valid keys, not min({n}, {topk})")
                if n <= topk and not np.array_equal(ps, np.arange(n)):
                    raise Unwired(f"{where}: token {t}'s top-k row doesn't read its {n} keys in order, the Program's map")
                if n > topk and (ps.max(initial=0) >= n or len(np.unique(ps)) != len(ps)):
                    raise Unwired(f"{where}: token {t}'s top-k row isn't a selection of its {n} keys")
                if n > topk:
                    picks[t] = {"kind": "selection", "rank": r, "seq": self.topk_seq(r, tap), "token": t,
                                "via": f"DsaWaksmanRows_v1{{N={n},K={topk},W={width},E=16}}",
                                "bits": waksman_switches(n, topk), "why": SELECTION_WHY}
                for j in range(lo, hi):
                    if j >= len(ps):
                        left.append((i, j))
                        continue
                    x, s = int(ps[j]), int(page[t, j])
                    tw = mine.get((q, x))
                    if tw is not None:
                        if s != slot_of[tw]:
                            raise Unwired(f"{where}: token {t} reads position {x} of its request at slot {s}, but "
                                          f"{wev['op']} (seq {wev['seq']}) wrote it at slot {slot_of[tw]}")
                        if read is None:
                            read = rec.tensor(r, ref)
                        if not np.array_equal(read[i, j], made[tw]):
                            raise Unwired(f"{where} reads {adv['in']}[{i}] row {j}, position {x} of its request, which "
                                          f"{wev['op']} (seq {wev['seq']}) wrote from its row {tw}, but not those bytes")
                        wrote.append([i, j, tw])
                    elif x >= first[q]:
                        raise Unwired(f"{where}: token {t} reads position {x} of its request, which the step computes, but "
                                      f"{wev['op']} (seq {wev['seq']}) records no write of it")
                    elif s in slots:
                        raise Unwired(f"{where}: token {t} reads position {x} of its request, an earlier step's, at slot {s}, "
                                      f"which {wev['op']} (seq {wev['seq']}) filled this step")
                    else:
                        history.append((i, j))
                        at.setdefault(i, (names[q], []))[1].append(x)
        out = []
        if wrote:
            out.append({**self.call(r, wev["seq"], "+".join(CACHE_WRITE[2])),
                        "into": {"seq": tap["seq"], "op": tap["op"], "in": adv["in"]}, "rows": wrote})
        if history:
            hp = _row_parts(history)
            out.append({**{k: adv[k] for k in ("rank", "seq", "op", "in", "shape", "width")}, "kind": "history",
                        "port": HISTORY_PORT, "parts": hp, "bits": _read_bits(adv["width"], adv["shape"], hp), "why": HISTORY_WHY,
                        "at": [[i, name, _runs(xs)] for i, (name, xs) in sorted(at.items())]})
        if left:
            lp = _row_parts(left)
            out.append({**adv, "parts": lp, "bits": _read_bits(adv["width"], adv["shape"], lp)})
        return out + list(picks.values())

    def cache_write(self, r: int, tap: dict) -> tuple[dict, dict[int, int], np.ndarray | None, np.ndarray] | None:
        """The step's cache write of the tap's layer on rank ``r`` (`CACHE_WRITE`, the latest before the tap), with each slot it
        wrote and the row it wrote it from, each row's keys (its position plus one) when its positions are recorded, and the
        cache rows it wrote; None when the step's record has no such call with its slot map and rows."""
        op, slot_name, outs = CACHE_WRITE
        ev = next((e for e in reversed(self.capture.events[r]) if e["seq"] < tap["seq"] and e["op"] == op
                   and e.get("layer") == tap.get("layer")), None)
        if ev is None or not all(self.capture.kept(r, x) for x in [ev["in"].get(slot_name) or {}, *(ev["out"].get(o) or {} for o in outs)]):
            return None
        slots: dict[int, int] = {}
        for t, s in enumerate(self.capture.tensor(r, ev["in"][slot_name]).reshape(-1).tolist()):
            if s >= 0:
                if s in slots:
                    raise Unwired(f"rank {r} {op} (seq {ev['seq']}) writes slot {s} from rows {slots[s]} and {t}")
                slots[int(s)] = t
        pos = ev["in"].get("positions")
        keys = self.capture.tensor(r, pos).reshape(-1).astype(np.int64) + 1 if isinstance(pos, dict) and self.capture.kept(r, pos) else None
        made = [self.capture.tensor(r, ev["out"][o]) for o in outs]
        return ev, slots, keys, np.concatenate([x.reshape(x.shape[0], -1) for x in made], axis=1)

    def single_tokens(self, r: int, tap: dict) -> set[int]:
        """The step's tokens whose request schedules only them: the decode rows of the indexer's top-k, which lead the batch
        (`long_rows`), when exactly that many of the step's requests schedule one token; else none."""
        dec = next((e for e in reversed(self.capture.events[r]) if e["seq"] < tap["seq"] and e["op"] == "_C::cooperative_topk"), None)
        if dec is None:
            return set()
        n, counts = int(dec["in"]["lengths"]["shape"][0]), list(self.capture.meta[r]["scheduled"].values())
        return set(range(n)) if counts.count(1) == n and sum(counts) == self.capture.tokens(r) else set()

    def topk_seq(self, r: int, tap: dict) -> int:
        """The indexer top-k whose output the tap's page table converts: the latest before it (layers that share a top-k read
        the last one computed), else the tap itself."""
        return next((e["seq"] for e in reversed(self.capture.events[r]) if e["seq"] < tap["seq"] and e["op"] in TOPK_OPS), tap["seq"])

    def source(self, r: int, ref: dict) -> dict | list[dict]:
        """``ref``'s source, found once per recorded reference (instances of one call read the same ones): a list, one per call,
        when several calls wrote its real rows in parts (`cover`)."""
        if id(ref) not in self.at:
            return self._source(r, ref)
        if id(ref) not in self.found:
            try:
                self.found[id(ref)] = self._source(r, ref)
            except Unwired as ex:
                self.found[id(ref)] = ex
        got = self.found[id(ref)]
        if isinstance(got, Unwired):
            raise got
        return got

    def _source(self, r: int, ref: dict) -> dict | list[dict]:
        if "param" in ref:
            return {"kind": "registered", "rank": r, "param": ref["param"],
                    "sha512": self.capture.params[r].get(ref["param"], {}).get("sha512")}
        if id(ref) not in self.at:
            raise Unwired(f"rank {r}: a {ref.get('dtype')} {ref.get('shape')} read that is no recorded call's argument")
        r, seq, side, name = self.at[id(ref)]
        e = self.by_seq[r][seq]
        if side == "out":
            return self.call(r, seq, name)
        for kind, table in (("gap", GAPS), ("advice", ADVICE)):
            if (e["op"], name) in table:
                bits = int(np.prod(ref["shape"], dtype=np.int64)) * 8 * np.dtype(NP[ref["dtype"]]).itemsize
                return {"kind": kind, "rank": r, "seq": seq, "op": e["op"], "in": name, "shape": list(ref["shape"]),
                        "width": bits, "bits": bits, "why": table[(e["op"], name)]}
        skip = {x["seq"] for x in self.enclosing(r, seq)}
        makers = [(s, o) for s, o in self.made.get((r, ref.get("sha512")), []) if s < seq and s not in skip]
        made = next(((s, o) for s, o in reversed(makers) if not self.passes(r, s, ref["sha512"])), makers[-1] if makers else None)
        view = None if made else self.view(r, seq, ref, skip)
        hits = [(*made, {})] if made else [view] if view else self.cover(r, seq, ref, skip)
        if not hits:
            raise Unwired(f"rank {r} {e['op']} (seq {seq}) reads {name}, a {ref.get('dtype')} {ref.get('shape')} that no earlier "
                          f"recorded call output, whole or as a strided view, or wrote through one, and that is no parameter: "
                          f"wire it (record the call that writes it in capture.py), or declare ({e['op']!r}, {name!r}) in "
                          f"stage.GAPS if the step computes it (printed as a gap until wired) or in stage.ADVICE if "
                          f"nothing does (a free choice, charged at what instances read of it)")
        got = [{**self.call(r, s, o), "into": {"seq": seq, "op": e["op"], "in": name}, **how} for s, o, how in hits]
        return got[0] if len(got) == 1 else got

    def passes(self, r: int, seq: int, sha: str) -> bool:
        """Whether the recorded call ``seq`` on rank ``r`` output ``sha``'s bytes as one of its own inputs brought them in (a
        same-dtype ``aten::to``, ``copy_``'s source, FA3 returning ``q``): it computed none of them, so an earlier call that
        output them is wired to instead, where one did."""
        return any(x.get("sha512") == sha for _, x in _refs(self.by_seq[r][seq].get("in") or {}))

    def call(self, r: int, seq: int, name: str) -> dict:
        e = self.by_seq[r][seq]
        gated = any(_gated(x) for x in [e, *self.enclosing(r, seq)])
        return {"kind": "block" if gated else "call", "rank": r, "seq": seq, "op": e["op"], "out": name}

    def view(self, r: int, seq: int, ref: dict, skip: set) -> tuple[int, str, dict] | None:
        """The latest earlier output on rank ``r`` that ``ref`` is a strided view of, by its recorded strides, from the offset
        where every one of a few probed elements and then the whole view agree, with ``{"view": {offset, stride}}``."""
        stride, shape = ref.get("stride"), ref.get("shape")
        if not self.capture.kept(r, ref) or not stride or not shape or min(stride) < 0 or not all(shape):
            return None
        x = self.capture.tensor(r, ref)
        flat = x.reshape(-1)
        reach = sum((d - 1) * s for d, s in zip(shape, stride))
        probes = np.unique(np.linspace(0, flat.size - 1, num=min(16, flat.size)).astype(np.int64))
        at = np.asarray([sum(int(i) * s for i, s in zip(np.unravel_index(p, shape), stride)) for p in probes], np.int64)
        tried = 0
        for s, o, out in reversed(self.outs.get(r, [])):
            if s >= seq or s in skip or out.get("dtype") != ref.get("dtype") or out["sha512"] == ref["sha512"]:
                continue
            if int(np.prod(out["shape"], dtype=np.int64)) <= reach or not self.capture.kept(r, out):
                continue
            tried += 1
            if tried > VIEW_CANDIDATES:
                break
            y = self.capture.tensor(r, out).reshape(-1)
            offs = np.flatnonzero(y[:y.size - reach] == flat[0])
            for p, a in zip(probes[1:], at[1:]):
                offs = offs[y[offs + a] == flat[p]]
            for off in offs[:VIEW_OFFSETS]:
                v = np.lib.stride_tricks.as_strided(y[off:], shape=shape, strides=[st * y.itemsize for st in stride])
                if np.array_equal(v, x):
                    return s, o, {"view": {"offset": int(off), "stride": list(stride)}}
        return None

    def cover(self, r: int, seq: int, ref: dict, skip: set) -> list[tuple[int, str, dict]] | None:
        """The latest earlier outputs on rank ``r`` that are strided views, by their recorded strides, into a row-major
        ``ref``, each of whole rows from a row's start (``offset``, found where a few probed elements and then the whole view
        agree) and starting in its real rows (`Capture.real_rows`), together every element of those once and nothing past
        ``ref``, and equal to it there, each with ``{"cover": {stride, offset, rows}}``: one ``out=`` buffer a later call reads
        whole, or one several calls wrote in parts (a mixed step's attention output: its decode rows from the absorb's
        ``out=``, its prefill rows copied in from FA3's). A view may run on into the padding rows: a full-graph step runs
        attention at the padded batch, so the absorb's ``out=`` writes every row. An earlier output that overlaps a later one
        taken is not taken. The real rows are the whole of what an instance reads of ``ref``: every population draws real rows
        only, so a padded batch's rows past them (written by the padded forward, or by nothing) reach no staged unit and are
        neither wired nor charged."""
        shape = ref.get("shape")
        if not shape or not all(shape) or ref.get("stride") != [int(np.prod(shape[i + 1:])) for i in range(len(shape))]:
            return None
        real, row = self.capture.real_rows(r, shape[0]), int(np.prod(shape[1:]))
        size, full, x, tried, pieces = real * row, shape[0] * row, None, 0, []
        if not size:
            return None
        hit = np.zeros(full, bool)
        for s, o, out in reversed(self.outs.get(r, [])):
            stride, n = out.get("stride"), int(np.prod(out["shape"], dtype=np.int64))
            if (s >= seq or s in skip or out.get("dtype") != ref.get("dtype") or not stride or min(stride) < 0 or not n
                    or n % row or n > full):
                continue
            reach = sum((d - 1) * st for d, st in zip(out["shape"], stride))
            mark = np.zeros(reach + 1, bool)
            np.lib.stride_tricks.as_strided(mark, shape=out["shape"], strides=stride)[...] = True
            if reach >= full or int(mark.sum()) != n or not self.capture.kept(r, out) or not self.capture.kept(r, ref):
                continue
            tried += 1
            if tried > VIEW_CANDIDATES:
                break
            x = self.capture.tensor(r, ref).reshape(-1) if x is None else x
            y = self.capture.tensor(r, out)
            first = np.unravel_index(np.arange(y.size)[::max(1, y.size // 16)], out["shape"])
            at = sum(i * st for i, st in zip(first, stride))
            offs = np.arange(0, min(size, full - reach), row)
            for a, want in zip(at, y[first]):
                offs = offs[x[offs + a] == want]
            for off in offs[:VIEW_OFFSETS]:
                if hit[off:off + reach + 1][mark].any():
                    continue
                v = np.lib.stride_tricks.as_strided(x[off:], shape=out["shape"], strides=[st * x.itemsize for st in stride])
                if np.array_equal(v, y):
                    hit[off:off + reach + 1] |= mark
                    pieces.append((s, o, {"cover": {"stride": list(stride), "offset": int(off), "rows": real}}))
                    break
            if hit[:size].all():
                return pieces
        return None


def wiring(types: dict) -> dict:
    """``window.json``'s wiring: each staged instance's inputs with their sources (`Wires`), the advice they read (free choices:
    each argument charged once, at the union of the parts staged instances read, or at its width if one read it whole), the
    trace gaps they read (``gap_bits``: computed values not wired yet, counted the same way and not charged), the rows they read
    of the window's carried cache (history, counted the same way), the selection settings (once per token and top-k), the
    calls they're wired to that no population gates, and the drawn instances left unstaged for an unwired input."""
    inputs: dict = {}
    charged: dict = {"advice": {}, "gap": {}, "history": {}}
    selection: dict = {}
    unproved: dict = {}
    for label, rec in types.items():
        for g in rec.get("gate", ()):
            if "sources" not in g:
                continue
            inputs.setdefault(label, []).append({"key": g["key"], "sources": g["sources"]})
            for s in g["sources"]:
                if s["kind"] in charged:
                    k, into = (s["rank"], s["seq"], s["in"]), charged[s["kind"]]
                    width, shape, got = into.get(k, (s.get("width", s["bits"]), s.get("shape"), []))
                    into[k] = (width, shape, None if got is None or "parts" not in s else got + s["parts"])
                elif s["kind"] == "selection":
                    selection[(s["rank"], s["seq"], s["token"])] = s["bits"]
                elif s["kind"] == "call":
                    unproved.setdefault(s["op"], set()).add((s["rank"], s["seq"]))
    unwired = {label: [g["unwired"] for g in rec.get("gate", ()) if "unwired" in g] for label, rec in types.items()}
    return {"inputs": inputs, "advice_bits": sum(_read_bits(*v) for v in charged["advice"].values()),
            "gap_bits": sum(_read_bits(*v) for v in charged["gap"].values()),
            "history_bits": sum(_read_bits(*v) for v in charged["history"].values()),
            "selection_bits": sum(selection.values()),
            "unproved_calls": {op: len(v) for op, v in sorted(unproved.items())}, "unwired": {k: v for k, v in unwired.items() if v}}


def _refs(d: dict, prefix: str = ""):
    """Each recorded tensor reference among an event's arguments or returns, by name (a list's ``name[i]``)."""
    for k, v in d.items() if isinstance(d, dict) else enumerate(d):
        name = f"{prefix}[{k}]" if prefix else str(k)
        if isinstance(v, dict) and ("sha512" in v or "param" in v):
            yield name, v
        elif isinstance(v, list):
            yield from _refs(v, name)


def call_of(label: str, key) -> tuple:
    """The kernel call an instance is of: ``(seq,)`` for the types aligned across ranks, else ``(rank, seq)``."""
    return tuple(key[:1]) if label.startswith(("AllReduce", "RmsNormMnnvl")) else tuple(key[:2])


def _gate_task(task: tuple[str, int]) -> tuple[str, dict]:
    label, i = task
    _, _, key, args_of, want_of = _EVERY["pops"][label][i]
    return label, gate_instance(_EVERY["specs"][label], key, args_of, want_of, _EVERY["full_gates"])[0]


def preload(cap: Capture, pops: dict, tasks: list[tuple[str, int]], jobs: int) -> dict:
    """Compose from ``--checkpoint``, before the fork, each unkept weight a sampled call reads, so the workers share it; a weight
    whose composition is not the committed bytes is left to fail its instances."""
    if cap.checkpoint is None:
        return {}
    from concurrent.futures import ThreadPoolExecutor
    by_seq = {(r, e["seq"]): e for r in cap.ranks for e in cap.events[r]}
    want = set()
    for label, i in tasks:
        key = pops[label][i][2]
        e = by_seq.get(tuple(key[:2])) if isinstance(key, tuple) and len(key) > 1 else None
        want |= {(key[0], v["param"]) for v in (e or {}).get("in", {}).values() if isinstance(v, dict) and "param" in v}

    def one(k: tuple[int, str]) -> str | None:
        try:
            cap.param(*k)
        except LookupError as ex:
            return str(ex)
        return None
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        fails = [x for x in ex.map(one, sorted(want)) if x]
    return {"weights": len(want), "composed": len(want) - len(fails), "refused": fails[:20]}


def every_call(cap: Capture, pops: dict, a, unstaged: dict | None = None) -> dict:
    """The match gate on ``--every-call`` instances of every recorded call of every type (sampled by the coins within the call),
    over every layer when the step was kept whole (weight-reading types over the opened layers only), in ``--jobs``
    processes, stopping at ``--deadline``."""
    import multiprocessing as mp
    import random
    rep: dict = {"capture": a.capture, "step": a.step, "ranks": cap.ranks, "root": cap.root, "whole": cap.whole,
                 "layers": "every layer" if cap.whole else cap.layers, "weight_layers": "every layer (--checkpoint)" if cap.checkpoint else cap.layers, "per_call": a.every_call,
                 "rows": "every real row" if cap.only is None else cap.only, "types": {}}
    specs: dict = {}
    tasks: list[tuple[str, int]] = []
    for label, insts in sorted(pops.items()):
        rec = rep["types"][label] = {"instances": len(insts)}
        try:
            specs[label] = insts[0][1] if not isinstance(insts[0][1], str) else block_fn(insts[0][1])
            rec["unit"] = specs[label].id
        except Exception as ex:  # noqa: BLE001
            rec["why"] = f"no Definition binds: {type(ex).__name__}: {str(ex)[:300]}"
            continue
        calls: dict[tuple, list[int]] = {}
        for i, inst in enumerate(insts):
            calls.setdefault(call_of(label, inst[2]), []).append(i)
        rec["calls"] = len(calls)
        for c, idx in sorted(calls.items()):
            tasks += [(label, idx[j]) for j in draw(cap.coins, f"every/{label}/{c}", len(idx), a.every_call)]
    random.Random(cap.coins).shuffle(tasks)
    rep["from_checkpoint"] = preload(cap, pops, tasks, a.jobs)
    _EVERY.update(pops=pops, specs=specs, full_gates=a.full_gates)
    done: dict[str, list[dict]] = {k: [] for k in specs}
    rep["instances"] = len(tasks)
    with mp.get_context("fork").Pool(a.jobs) as pool:
        it = pool.imap_unordered(_gate_task, tasks, chunksize=1)
        for _ in tasks:
            try:
                label, g = it.next(timeout=max(1.0, a.deadline - time.time()) if a.deadline else None)
            except mp.TimeoutError:
                rep["deadline"] = f"stopped at the deadline after {sum(map(len, done.values()))} of {len(tasks)} instances"
                break
            done[label].append(g)
        pool.terminate()
    for label, gs in done.items():
        def kept(g: dict) -> bool:
            return not (g["agree"] is None and g["error"].startswith("LookupError: not kept"))
        bad = [g for g in gs if g["agree"] is not True and kept(g)]
        not_kept = [g for g in gs if not kept(g)]
        rep["types"][label].update(gated=len(gs), calls_gated=len({call_of(label, g["key"]) for g in gs}),
                                   agree=sum(g["agree"] is True for g in gs), disagree=sum(g["agree"] is False for g in gs),
                                   errors=sum(g["agree"] is None for g in gs) - len(not_kept), not_kept=len(not_kept),
                                   failures=bad[:8] + not_kept[:2])
    # read_ops: what a population reads, not what this run gated (--types, the deadline or an unbound Definition can
    # leave some ungated); the per-type counts above say what was
    rep["read_ops"], rep["not_gated_ops"], schedule = op_coverage(cap, unstaged)
    rep["schedule_ops"] = {op: {"calls": n, "why": SCHEDULE_OPS[op]} for op, n in schedule.items()}
    return rep


def op_coverage(cap: Capture, unstaged: dict | None = None) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """The step's recorded ops, nested ones included, that some population reads; those that none reads and no op it reads
    encloses (an op inside an ungated one, such as the router's top-k inside the MoE block, is listed itself); and, apart from
    them, the `SCHEDULE_OPS`. A call a population refused (listed in ``unstaged``, by rank and seq or by seq on every rank)
    counts as not gated."""
    refused = {(c.get("rank", r), c["seq"]) for rec in (unstaged or {}).values() for c in rec.get("calls", ())
               for r in ([c["rank"]] if "rank" in c else cap.ranks)}
    gated: dict[str, int] = {}
    other: dict[str, int] = {}
    schedule: dict[str, int] = {}
    for r in cap.ranks:
        by_seq = {e["seq"]: e for e in cap.events[r]}
        for e in cap.events[r]:
            if not (cap.whole or e.get("layer") in cap.layers):
                continue
            x, inside = by_seq.get(e.get("parent")), False
            while x is not None and not inside:
                inside, x = _gated(x) and (r, x["seq"]) not in refused, by_seq.get(x.get("parent"))
            if _gated(e) and (r, e["seq"]) not in refused:
                gated[e["op"]] = gated.get(e["op"], 0) + 1
            elif not inside:
                into = schedule if e["op"] in SCHEDULE_OPS else other
                into[e["op"]] = into.get(e["op"], 0) + 1
    return dict(sorted(gated.items())), dict(sorted(other.items())), dict(sorted(schedule.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--types", default=None,
                    help="comma-separated labels (default: every type found); with --every-call, label prefixes")
    ap.add_argument("--cc", default="9.0", help="the serving GPUs' compute capability (the MoE launches' e4m3 step)")
    ap.add_argument("--full-gates", type=int, default=4_000_000)
    ap.add_argument("--of", default="", help="what the replay block is (window.json)")
    ap.add_argument("--link", default="", help="how its values link to serving (window.json)")
    ap.add_argument("--row", default="spot-glm53")
    ap.add_argument("--every-call", type=int, default=0, metavar="N",
                    help="instead of staging, gate N instances of every call of every type (OUT/gate.json)")
    ap.add_argument("--jobs", type=int, default=1, help="--every-call's processes")
    ap.add_argument("--checkpoint", default=None,
                    help="the served checkpoint: a weight the capture didn't keep is composed from it and held to params.json")
    ap.add_argument("--deadline", type=float, default=0.0, help="--every-call stops at this Unix time (0: none)")
    ap.add_argument("--allreduce-tree", default=None, metavar="TREE",
                    help=f"the node's multimem reduction tree, replacing the capture's {TREE_FILE} (default: that file's, "
                         "else the registry's GLM53_H200_TREE, which is another node's)")
    ap.add_argument("--rows", default=None, metavar="T,...", help="cut every population to these token rows")
    ap.add_argument("--rows-past-keys", type=int, default=0, metavar="K",
                    help="cut every population to the token rows attending to more than K keys (the DSA indexer's top-k "
                         "inputs say which: 2048 gives the long rows)")
    a = ap.parse_args()
    import a11
    import a11_kinds as K
    from verity.circuits.codec import canonical_json, encode_program
    from verity.protocols.spot.verifier.one_stage import partition as P
    from verity_vllm import TargetProfile

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap = Capture(Path(a.capture), a.step)
    cap.every, cap.per_call = bool(a.every_call) and cap.whole, a.every_call
    cap.checkpoint = a.checkpoint
    if a.allreduce_tree:
        cap.tree = {"tree": parse_tree(a.allreduce_tree, len(cap.ranks)), "source": "--allreduce-tree"}
    if a.rows:
        cap.only = {int(t): None for t in a.rows.split(",")}
    elif a.rows_past_keys:
        cap.only = long_rows(cap, a.rows_past_keys)
        print(f"rows past {a.rows_past_keys} keys (token: keys): {cap.only}", flush=True)
    profile = TargetProfile(compute_capability=tuple(map(int, a.cc.split("."))), batch_invariant=False)
    report: dict = {"capture": a.capture, "step": a.step, "ranks": cap.ranks, "root": cap.root, "coins": cap.coins.hex(),
                    "layers": cap.layers, "checks": {r: cap.meta[r]["check"] for r in cap.ranks}, "types": {},
                    "allreduce_tree": cap.tree or "the registry's GLM53_H200_TREE, not read off this node",
                    "rows": "every real row" if cap.only is None else cap.only}
    pops: dict[str, list] = {}
    unstaged: dict[str, dict] = {}
    sources = [quant_instances(cap), fp8_linear_instances(cap), moe_instances(cap, profile),
               fused_ar_norm_instances(cap), allreduce_instances(cap, unstaged), cublas_instances(cap, profile, unstaged),
               bmm_instances(cap, profile, unstaged), fused_add_rms_norm_instances(cap), rms_norm_instances(cap),
               add_instances(cap, unstaged), router_instances(cap, unstaged), silu_mul_instances(cap),
               moe_sum_instances(cap), mla_combine_instances(cap), mla_split_part_instances(cap)]
    for src in sources:
        try:
            for inst in src:
                pops.setdefault(inst[0], []).append(inst)
        except Exception as ex:  # noqa: BLE001
            report.setdefault("population_errors", []).append(f"{type(ex).__name__}: {ex}"[:400])
    for label, rec in unstaged.items():
        if "why" in rec:
            report["types"][label] = {**rec, "staged": False}
    if a.every_call:
        if a.types:
            pops = {k: v for k, v in pops.items() if k.startswith(tuple(a.types.split(",")))}
        rep = every_call(cap, pops, a, unstaged)
        rep["population_errors"] = report.get("population_errors", [])
        rep["unstaged"] = unstaged
        rep["allreduce_tree"] = report["allreduce_tree"]
        rep["seconds"] = round(time.time() - t0, 1)
        (out / "gate.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
        for label, rec in rep["types"].items():
            print(f"gate {label}: {rec.get('agree', 0)} agree, {rec.get('disagree', 0)} disagree, {rec.get('errors', 0)} errors, "
                  f"{rec.get('not_kept', 0)} not kept over {rec.get('calls_gated', 0)}/{rec.get('calls', 0)} calls "
                  f"{rec.get('why', '')}", flush=True)
        print(f"gate: not gated ops {rep['not_gated_ops']}; schedule ops {sorted(rep['schedule_ops'])}; "
              f"not modelled {sorted(k for k, v in unstaged.items() if 'why' in v)}; population errors {rep['population_errors']}",
              flush=True)
        return 0
    for family, prefer in FAMILIES.items():
        members = sorted((k for k in pops if k.startswith(family)), key=lambda k: (k != prefer, int(k[len(family):]), k))
        for k in members[1:]:
            report["types"][k] = {"population": len(pops.pop(k)), "staged": False,
                                  "why": f"one Definition of the family {family} is staged: {members[0]}"}
    wanted = set(a.types.split(",")) if a.types else None
    staged, rows_of_kind, kinds, wires = {}, {}, {}, Wires(cap)
    for label, insts in sorted(pops.items()):
        if wanted and label not in wanted:
            continue
        rec: dict = {"population": len(insts), **unstaged.get(label, {})}
        report["types"][label] = rec
        try:
            spec = insts[0][1] if not isinstance(insts[0][1], str) else block_fn(insts[0][1])
        except Exception as ex:  # noqa: BLE001
            rec.update(staged=False, why=f"no Definition binds: {type(ex).__name__}: {str(ex)[:300]}")
            continue
        rec["unit"] = spec.id
        rows_spec = K.rows_of(spec)
        if rows_spec is None:
            rec.update(staged=False, why=f"{spec.id}: the parameters' bits are not whole u16 words")
            continue
        picks = draw(cap.coins, label, len(insts), a.per_type)
        rec["drawn"] = [list(insts[i][2]) for i in picks]
        ins, outs, gate = [], [], []
        for i in picks:
            _, _, key, args_of, want_of = insts[i]
            reads: list = []
            g, ev = gate_instance(spec, key, traced(cap, args_of, reads), want_of, a.full_gates)
            gate.append(g)
            if g["agree"]:
                try:
                    g["sources"] = wires.of(reads)
                except Unwired as ex:
                    g["unwired"] = str(ex)
                    continue
                ins += ev["in"]
                outs += ev["out"]
        rec["gate"] = gate
        if not ins:
            unwired = next((g["unwired"] for g in gate if "unwired" in g), None)
            rec.update(staged=False, why=f"every drawn instance that passed the match gate reads an unwired input: {unwired}"
                       if unwired else "no drawn instance passed the match gate")
            continue
        rows, routs = a11.unit_rows(spec, rows_spec, ins, outs)
        rows_of_kind[label] = {"unit": spec, "rows": rows, "outs": routs}
        n = len(ins)
        agree = [g["agree"] for g in gate]
        staged[label] = {"staged": True, "unit": spec.id, "n": n, "name": label, "step": a.step,
                         "event": rec["drawn"][0], "calls": rec["drawn"],
                         "twin": {"agree": all(x is True for x in agree), "calls": len(agree), "compared": sum(g.get("words", 0) for g in gate)}}
        kinds[label] = {"gates": spec.gates, "outcome": "lowers"}
        rec.update(staged=True, n=n)
        print(f"stage {label}: {n}/{len(picks)} drawn of {len(insts)} pass the gate ({spec.id[:80]})", flush=True)
    for label, rec in report["types"].items():
        if not rec.get("staged"):
            print(f"stage {label}: not staged: {rec.get('why')}", flush=True)
    units = {kv["unit"].id: kv["unit"] for kv in rows_of_kind.values()}
    if units:
        (out / "units.json").write_bytes(canonical_json(encode_program(P.mixed_population_program([(u, 1) for u in units.values()]))))
    for f, kv in rows_of_kind.items():
        np.savez(out / f"{f}.npz", **{f"i.{r}": np.asarray(v, dtype=np.uint16) for r, v in kv["rows"].items()},
                 **{f"o.{o}": np.asarray(v, dtype=np.uint16) for o, v in kv["outs"].items()})
    weights = {"from": "the parameters each rank held while serving, written by the capture for the opened layers and held to "
                       "their committed SHA-512s", "layers": cap.layers}
    (out / "values.json").write_text(json.dumps({"staged": staged, "weights": weights, "values": a.capture, "steps": str(a.step),
                                                 "layers": ",".join(map(str, cap.layers)), "per_kind": a.per_type},
                                                indent=1, sort_keys=True, default=str))
    (out / "kinds.json").write_text(json.dumps({"kinds": kinds}, indent=1, sort_keys=True))
    wired = wiring(report["types"])
    (out / "window.json").write_text(json.dumps({"of": a.of, "link": a.link, "row": a.row, "root": cap.root, **wired}, indent=1))
    kinds_of = [s["kind"] for v in wired["inputs"].values() for g in v for s in g["sources"]]
    shown = ('block', 'call', 'registered', 'history', 'selection', 'gap', 'advice')
    print(f"wires: {', '.join(f'{kinds_of.count(k)} {k}' for k in shown)} sources; advice (free choices) {wired['advice_bits']} "
          f"bits; trace gaps (computed, not wired yet; not charged) {wired['gap_bits']} bits; history {wired['history_bits']} "
          f"bits ({HISTORY_PORT}); selection {wired['selection_bits']} bits; calls wired to but not proved "
          f"{wired['unproved_calls']}; unwired {sorted(wired['unwired'])}", flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    (out / "stage.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    return 0 if staged else 1


if __name__ == "__main__":
    raise SystemExit(main())
