"""Sampled VUs of a passing row, exported as content-addressed benchmark instance sets (`vllm-vu-set/v1`).

Opt-in and not on the record: the Commit stage calls `after_replay` once pair 0's replay of record has passed and its value source
is recorded; it exports only when `--vu-export-dir` (env `VERITY_VU_EXPORT_DIR`) is given.  `limits.json` (the `Limits` fields) and
`provenance.json` (extra provenance: the row number, the research run) in that directory are read when present.  The draw is a
stratified sample over the population the replay samples (every evaluable VU of the executed prefix, the replay's strata), keyed by
`verity.randomness.derive` from the run root.  It is a benchmark selection, not a verifier challenge: fixed by the commitment and
independent of the replay's own sample, nothing more.  Each drawn VU is
evaluated exactly as the replay evaluates it (`check.replay.evaluate.replay_vu`: committed operands read through the binding map,
registered weights, the family's row kernel), with an evaluation tap recording the operands and outputs of every row it
evaluates, interior producers included.  A VU is exported only when it replays bit-exactly against its committed outputs.

Each evaluated row is decomposed into the subcircuit a backend proves, and every instance is re-evaluated under that subcircuit's
relation before it is written (a decomposition that does not reproduce the row's words is an error, never an instance):

  Gemm_v1{K,N} / Gemm_v2{K,N,DOT}  -> GemmCoordinate<K>: x[K], w[K] (one weight row), y[1]; `gemm-coordinate-<dot>-k<K>`
  Attention_v3 / Attention_v2      -> one query head over its KV head: q[D], k[T*D], v[T*D], out[D]; `attention-head-<dot>-d<D>-bn<BN>`
  RoPE_v1{NHEADS,D}                -> one head: x[D], cs[D], out[D]; `rope-head-d<D>`
  any other evaluable family       -> the row itself: in<g> per arg group, out.<member>; `<family>-<int statics>`

`Embedding_v1` is not exported (a commitment opening of the table, not a subcircuit).

Layout of a set `<out>/sets/<name>/`: flat little-endian arrays `<port>.<u8|u16|u32|u64>`, instance-major (the
`fixtures/bench-instances/v1` convention); a port whose width varies across instances has `<port>.offsets.u64` (n + 1 word
offsets).  `manifest.json` holds the relation, the ports, the sha256 of every file, the content digest (relation + ports + file
digests, provenance excluded) and the provenance (row, model, run, commitment root, Program and manifest digests).  `index.json`
has one row per instance (the VU it came from and the sub-index inside it).  `<out>/export.json` summarises the export and
`<out>/vus.jsonl` lists every drawn VU with its operand identities and outcome.  `verify(<out>)` re-checks every file digest and
re-evaluates every instance.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from verity.randomness import Key, derive

from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.check.replay.population import phase_anchor, population, split_not_executed
from verity_vllm.commit import challenge as CH
from verity_vllm.program.kernels import derived_rows as D
from verity_vllm.program.kernels import rows as RK
from verity_vllm.query.manifest import format as RM

SET_SCHEMA = "vllm-vu-set/v1"
EXPORT_SCHEMA = "vllm-vu-export/v1"
SELECTION_DOMAIN = "verity/vllm-vu-export/select/v1"
SELECTION_SOURCE = b"verity-vllm benchmark selection"

NOT_EXPORTED = {"Embedding_v1": "a commitment opening of the embedding table (spec), not a subcircuit"}

_DOT_SLUG = {"AmpereBF16TcDot16_v1": "ampere-bf16", "HopperBF16WgmmaDot16_v1": "hopper-bf16-wgmma"}
_WORD = {1: ("u8", np.uint8), 2: ("u16", np.uint16), 4: ("u32", np.uint32), 8: ("u64", np.uint64)}


@dataclasses.dataclass
class Limits:
    per_family: int = 256                    # VUs drawn per family (round-robin across the replay's strata)
    per_family_override: dict[str, int] = dataclasses.field(default_factory=dict)
    gemm_coords_per_row: int = 32            # GemmCoordinate instances taken from one Gemm row
    heads_per_row: int = 4                   # attention / RoPE heads taken from one row
    max_instances_per_set: int = 8192
    max_row_words: int = 1 << 22             # a generic row whose operands exceed this is counted, not written
    max_seconds: float = 600.0               # evaluation budget: the draw stops (named, per family) once it is spent
    workers: int = 8                         # > 1: a forked pool (host-retained stores only; a device-retained one runs serial)

    @classmethod
    def from_dir(cls, out_dir: str | Path) -> "Limits":
        f = Path(out_dir) / "limits.json"
        return cls(**json.loads(f.read_text())) if f.is_file() else cls()

    def for_family(self, fam: str) -> int:
        return int(self.per_family_override.get(fam, self.per_family))


class DecompositionMismatch(RuntimeError):
    """A subcircuit instance did not reproduce the words of the row it was cut from."""


@dataclasses.dataclass
class Instance:
    set_name: str
    relation: dict[str, Any]
    ports: dict[str, np.ndarray]
    sub: dict[str, Any]


def selection_key(run_root: bytes) -> Key:
    return derive(SELECTION_SOURCE, SELECTION_DOMAIN, {"run_root": bytes(run_root)})


class _Picker:
    """`k` of `n` sub-indices of one evaluated row (coordinates, heads), drawn from the selection key at the row's own index."""

    def __init__(self, key: Key, index: tuple) -> None:
        self.key, self.index = key, index

    def __call__(self, n: int, k: int) -> list[int]:
        return list(range(n)) if k >= n else list(self.key.subset(k, n, self.index))


def select(pool: list[dict], limits: Limits, key: Key) -> list[dict]:
    """Per family: `limits.for_family` VUs, drawn per stratum (`challenge.stratum_picks` under the family's shard of `key`) and emitted
    round-robin across the family's strata, so a small cap still reaches every stratum it can.  The families are interleaved (one VU
    of each in turn), so a spent time budget truncates every family alike."""
    by_fam: dict[str, dict[str, list[dict]]] = {}
    for v in pool:
        if v["family"] in NOT_EXPORTED:
            continue
        by_fam.setdefault(v["family"], {}).setdefault(v["stratum"], []).append(v)
    per: list[list[dict]] = []
    for fam in sorted(by_fam):
        cap = limits.for_family(fam)
        fkey = key.shard(("family", fam))
        # the strata in a keyed order: in name order a cap below the stratum count takes one shape's strata first (all K=1536 of #4's GEMMs)
        names = sorted(by_fam[fam], key=lambda k: fkey.stream(("stratum-order", k), 16))
        picks = CH.stratum_picks([by_fam[fam][k] for k in names], cap, None, fkey)
        per.append(_round_robin(picks)[:cap])
    return _round_robin(per)


def _round_robin(lists: list[list[dict]]) -> list[dict]:
    return [ls[j] for j in range(max((len(ls) for ls in lists), default=0)) for ls in lists if j < len(ls)]


def _unsigned(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a)).reshape(-1)
    if a.dtype.kind == "u":
        return a
    if a.dtype.itemsize not in _WORD:
        raise TypeError(f"no word type for {a.dtype}")
    return a.view(_WORD[a.dtype.itemsize][1])


def _outputs(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The row kernel's members with its aliases dropped (`Gemm_v1` returns the same array as `0` and `out`)."""
    seen: dict[int, str] = {}
    res = {}
    for k, v in out.items():
        if id(v) in seen:
            continue
        seen[id(v)] = k
        res[k] = _unsigned(v)
    return res


def _one(out: dict[str, np.ndarray]) -> np.ndarray:
    o = _outputs(out)
    if len(o) != 1:
        raise DecompositionMismatch(f"expected one output member, the kernel returned {sorted(o)}")
    return next(iter(o.values()))


def _gemm_dot(fam: str, st: dict[str, Any]) -> tuple[str, Callable]:
    if fam == "Gemm_v1":
        return "AmpereBF16TcDot16_v1", D.dot_bf16
    dot_id = RK.gemm_dot_prim(st)
    return dot_id, D.dot_bf16_for(dot_id)


def gemm_coordinate(dot_id: str, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y = f2fp_bf16(ascending k16 chain of `dot_id` from +0) for rows w (n, K) against x (K,): (y words, non-finite lanes)."""
    acc, nf = D.dot_bf16_for(dot_id)(np.asarray(x, dtype=np.uint16)[None, :], np.asarray(w, dtype=np.uint16))
    return D.f2fp_bf16(acc).astype(np.uint16).reshape(-1), np.asarray(nf).reshape(-1)


def _decompose_gemm(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    K, N = int(st["K"]), int(st["N"])
    dot_id, _dot = _gemm_dot(fam, st)
    x = np.asarray(ops[0], dtype=np.uint16).reshape(K)
    w = np.asarray(ops[1], dtype=np.uint16).reshape(N, K)
    y = _one(out).astype(np.uint16)
    sel = np.asarray(pick(N, limits.gemm_coords_per_row), dtype=np.int64)
    yy, nf = gemm_coordinate(dot_id, x, w[sel])
    rel = {"subcircuit": f"GemmCoordinate<{K}>", "definition": fam, "dot": dot_id, "statics": {"K": K},
           "rule": "y = f2fp_bf16(ascending k16 chain of DOT over (x, w) from +0)"}
    for j, n in enumerate(sel.tolist()):
        if nf[j]:
            continue                                                         # outside the vectorised twin's domain: not an instance
        if yy[j] != y[n]:
            raise DecompositionMismatch(f"{fam} coordinate {n}: {int(yy[j]):#06x} != row word {int(y[n]):#06x}")
        yield Instance(f"gemm-coordinate-{_DOT_SLUG.get(dot_id, dot_id.lower())}-k{K}", rel,
                       {"x": x.copy(), "w": w[n].copy(), "y": y[n:n + 1].copy()}, {"n": int(n), "N": N})


def _attn_prims(fam: str, st: dict[str, Any]) -> tuple[str, str]:
    if fam == "Attention_v2":
        return RK.prim_static(st, "DOT", fam), RK.prim_static(st, "INV", fam)
    return "AmpereBF16TcDot16_v1", "Fa2InvSum_v1"


def attention_head(dot_id: str, inv_id: str, BN: int, q: np.ndarray, k: np.ndarray, v: np.ndarray, Dh: int) -> tuple[np.ndarray, bool]:
    """One head: q (D,), k / v (T*D,) -> (out (D,) bf16 words, non-finite)."""
    T = int(np.asarray(k).size) // Dh
    o, nf = D.attention_rows(np.asarray(q, dtype=np.uint16).reshape(1, Dh), np.asarray(k, dtype=np.uint16).reshape(T, 1, Dh),
                             np.asarray(v, dtype=np.uint16).reshape(T, 1, Dh), BN, dot_prim=dot_id, inv_prim=inv_id)
    return np.asarray(o, dtype=np.uint16).reshape(Dh), bool(np.asarray(nf).any())


def _decompose_attention(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    T, NH, KVH, Dh, BN = (int(st[k]) for k in ("T", "NH", "KVH", "D", "BN"))
    dot_id, inv_id = _attn_prims(fam, st)
    q = np.asarray(ops[0], dtype=np.uint16).reshape(NH, Dh)
    kc = np.asarray(ops[1], dtype=np.uint16).reshape(T, KVH, Dh)
    vc = np.asarray(ops[2], dtype=np.uint16).reshape(T, KVH, Dh)
    o = _one(out).astype(np.uint16).reshape(NH, Dh)
    rel = {"subcircuit": f"AttentionHead<T,{Dh}>", "definition": fam, "dot": dot_id, "inv": inv_id, "statics": {"D": Dh, "BN": BN},
           "rule": "one query at position T-1 over keys 0..T-1 of its KV head, ceil(T/BN) key blocks last -> first (the Definition's head)"}
    name = (f"attention-head-fa2-d{Dh}-bn{BN}" if fam == "Attention_v3"
            else f"attention-head-{_DOT_SLUG.get(dot_id, dot_id.lower())}-{inv_id.lower()}-d{Dh}-bn{BN}")
    for h in pick(NH, limits.heads_per_row):
        g = h // (NH // KVH)
        ports = {"q": q[h].copy(), "k": kc[:, g].reshape(-1).copy(), "v": vc[:, g].reshape(-1).copy(), "out": o[h].copy()}
        got, nf = attention_head(dot_id, inv_id, BN, ports["q"], ports["k"], ports["v"], Dh)
        if nf:
            continue
        if not np.array_equal(got, ports["out"]):
            raise DecompositionMismatch(f"{fam} head {h}: {int(np.count_nonzero(got != ports['out']))} words differ from the row")
        yield Instance(name, rel, ports, {"head": int(h), "kv_head": int(g), "T": T, "NH": NH, "KVH": KVH})


def _decompose_rope(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    NH, Dh = int(st["NHEADS"]), int(st["D"])
    x = np.asarray(ops[0], dtype=np.uint16).reshape(NH, Dh)
    cs = np.asarray(ops[1], dtype=np.uint16).reshape(Dh)
    o = _one(out).astype(np.uint16).reshape(NH, Dh)
    rel = {"subcircuit": f"RoPEHead<{Dh}>", "definition": fam, "statics": {"D": Dh},
           "rule": "out[i] = bf16_rn(RN32(x[i]*c[i] - x[i+R]*s[i])), out[i+R] = bf16_rn(RN32(x[i+R]*c[i] + x[i]*s[i])), cs = [c | s], R = D/2"}
    for h in pick(NH, limits.heads_per_row):
        got = D.rope_rows(x[h:h + 1], cs).reshape(Dh)
        if not np.array_equal(got, o[h]):
            raise DecompositionMismatch(f"{fam} head {h}: differs from the row")
        yield Instance(f"rope-head-d{Dh}", rel, {"x": x[h].copy(), "cs": cs.copy(), "out": o[h].copy()}, {"head": int(h), "NHEADS": NH})


def _row_set_name(fam: str, st: dict[str, Any]) -> str:
    ints = "-".join(f"{k.lower()}{v}" for k, v in sorted(st.items()) if isinstance(v, int))
    other = {k: v for k, v in st.items() if not isinstance(v, int)}
    tag = hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()[:8] if other else ""
    return "-".join(p for p in (fam.lower().replace("_", "-"), ints, tag) if p)


def _decompose_row(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    ports = {f"in{g}": _unsigned(a).copy() for g, a in enumerate(ops)}
    ports.update({f"out.{k}": v.copy() for k, v in _outputs(out).items()})
    if sum(int(a.size) for a in ports.values()) > limits.max_row_words:
        return
    rel = {"subcircuit": fam, "definition": fam, "statics": st, "rule": f"the registered row kernel of {fam} (verity_vllm.program.kernels.rows)"}
    yield Instance(_row_set_name(fam, st), rel, ports, {})


DECOMPOSE: dict[str, Callable[..., Iterator[Instance]]] = {"Gemm_v1": _decompose_gemm, "Gemm_v2": _decompose_gemm,
                                                           "Attention_v3": _decompose_attention, "Attention_v2": _decompose_attention,
                                                           "RoPE_v1": _decompose_rope}


def template_of(spec: str) -> tuple[str, dict[str, Any], int]:
    """A Call's Definition spec -> (subcircuit template, bound parameters, instances per Call): `decompose`'s cut, read off the spec."""
    fam, st = RM.spec_family(spec), RM.spec_statics(spec)
    try:
        if fam in ("Gemm_v1", "Gemm_v2"):
            return "GemmCoordinate", {"K": st["K"], "dot": _gemm_dot(fam, st)[0]}, int(st["N"])
        if fam in ("Attention_v3", "Attention_v2"):
            dot, inv = _attn_prims(fam, st)
            return "AttentionHead", {"D": st["D"], "BN": st["BN"], "dot": dot, "inv": inv}, int(st["NH"])
        if fam == "RoPE_v1":
            return "RoPEHead", {"D": st["D"]}, int(st["NHEADS"])
    except (KeyError, RK.Declined):
        pass
    return fam, st, 1


def decompose(fam: str, st: dict[str, Any], ops: list[np.ndarray], out: dict[str, np.ndarray], pick: Callable[[int, int], list[int]],
              limits: Limits) -> list[Instance]:
    if fam in NOT_EXPORTED:
        return []
    return list(DECOMPOSE.get(fam, _decompose_row)(fam, st, ops, out, pick, limits))


def capture(pi: ProgramIndex, store: CommittedStore, weights: Callable, vu: dict, *, arrive: int) -> tuple[dict, list[tuple]]:
    """The replay's evaluation of one VU, with every row evaluation it performs (interior producers first, the VU's own row last)."""
    evals: list[tuple] = []
    tap = lambda fam, st, ops, out: evals.append((fam, dict(st), ops, out))  # noqa: E731
    EV.EVALUATION_TAPS.append(tap)
    try:
        rec = EV.replay_vu(pi, store, weights, vu, arrive=arrive)
    finally:
        EV.EVALUATION_TAPS.remove(tap)
    return rec, evals


class _SetWriter:
    def __init__(self, name: str, relation: dict[str, Any]) -> None:
        self.name, self.relation = name, relation
        self.ports: dict[str, list[np.ndarray]] = {}
        self.index: list[list[Any]] = []

    def add(self, inst: Instance, ref: list[Any]) -> None:
        if self.ports and set(inst.ports) != set(self.ports):
            raise DecompositionMismatch(f"set {self.name}: ports {sorted(inst.ports)} != {sorted(self.ports)}")
        for k, a in inst.ports.items():
            self.ports.setdefault(k, []).append(_unsigned(a))
        self.index.append([len(self.index), *ref, inst.sub])

    def write(self, root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
        d = root / self.name
        d.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, Any]] = {}
        ports: dict[str, dict[str, Any]] = {}

        def put(fname: str, arr: np.ndarray) -> None:
            b = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
            (d / fname).write_bytes(b)
            files[fname] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}

        for k in sorted(self.ports):
            arrs = self.ports[k]
            dt = {a.dtype for a in arrs}
            if len(dt) != 1:
                raise DecompositionMismatch(f"set {self.name} port {k}: word types {sorted(map(str, dt))}")
            wname = _WORD[arrs[0].dtype.itemsize][0]
            sizes = {int(a.size) for a in arrs}
            put(f"{k}.{wname}", np.concatenate(arrs))
            p: dict[str, Any] = {"dtype": wname, "file": f"{k}.{wname}", "words": sizes.pop() if len(sizes) == 1 else None}
            if p["words"] is None:
                offs = np.concatenate([[0], np.cumsum([a.size for a in arrs])]).astype(np.uint64)
                put(f"{k}.offsets.u64", offs)
                p["offsets"] = f"{k}.offsets.u64"
            ports[k] = p
        idx = {"schema": f"{SET_SCHEMA}/index", "set": self.name,
               "columns": ["id", "vu", "request", "engine_step", "op_path", "row", "phase", "source", "sub"], "rows": self.index}
        ib = (json.dumps(idx, separators=(",", ":")) + "\n").encode()
        (d / "index.json").write_bytes(ib)
        files["index.json"] = {"sha256": hashlib.sha256(ib).hexdigest(), "bytes": len(ib)}
        body = {"schema": SET_SCHEMA, "set": self.name, "relation": self.relation, "n": len(self.index), "ports": ports,
                "files": {k: v for k, v in files.items() if k != "index.json"}}
        content = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        man = dict(body, files=files, content_digest=content, provenance=provenance,
                   content_digest_rule="sha256 of the canonical JSON of {schema, set, relation, n, ports, files without index.json}")
        (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str) + "\n")
        return {"set": self.name, "n": len(self.index), "bytes": sum(v["bytes"] for v in files.values()), "content_digest": content,
                "relation": self.relation.get("subcircuit")}


_FORK: dict[str, Any] = {}          # the per-VU context the forked workers read (set in the parent right before the fork)


def _one_vu(k: int) -> tuple[dict[str, Any], list[tuple[Instance, str]], float]:
    """Evaluate VU `k` of the draw and decompose every row it evaluated: (log entry, [(instance, source)], seconds)."""
    c = _FORK
    vu, tv = c["picked"][k], time.perf_counter()
    rid = vu["request_id"]
    rec, evals = capture(c["indices"][rid], c["store"], c["weights"], vu, arrive=int(c["arrivals"].get(rid, 0)))
    entry = {"vu": k, "request": rid, "i": vu["i"], "family": vu["family"], "spec": vu["spec"], "stratum": vu["stratum"],
             "engine_step": rec.get("engine_step"), "op_path": rec.get("op_path"), "row": rec.get("row"), "phase": vu["phase"],
             "result": rec.get("result"), "why": rec.get("why"), "inputs": rec.get("inputs")}
    insts: list[tuple[Instance, str]] = []
    if rec.get("result") is True:
        try:
            for j, (efam, est, ops, eout) in enumerate(evals):
                pick = _Picker(c["key"], ("row", str(rid), int(vu["i"]), j))
                insts += [(inst, "committed" if j == len(evals) - 1 else "interior") for inst in decompose(efam, est, ops, eout, pick, c["limits"])]
        except DecompositionMismatch as e:
            entry["decomposition_error"], insts = str(e), []
    return entry, insts, time.perf_counter() - tv


def _run_draw(n: int, workers: int, budget_s: float, t0: float) -> Iterator[tuple[dict[str, Any], list[tuple[Instance, str]], float] | None]:
    """The draw's results in order; None for a VU the spent time budget left undrawn.  `workers` > 1: a forked pool in chunks (the
    replay's pattern: weights and twin libraries pre-warmed in the parent, the heap frozen before the fork, children never touch CUDA)."""
    if workers <= 1:
        for k in range(n):
            yield None if time.perf_counter() - t0 > budget_s else _one_vu(k)
        return
    import gc
    import multiprocessing as mp
    from verity_vllm.check.replay import driver as SR
    SR._prewarm_weights(_FORK["indices"], _FORK["picked"], _FORK["weights"])
    SR._prewarm_evaluators(_FORK["picked"])
    gc.collect()
    gc.freeze()
    try:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            chunk = workers * 4
            for lo in range(0, n, chunk):
                if time.perf_counter() - t0 > budget_s:
                    yield from (None for _ in range(lo, n))
                    return
                yield from pool.map(_one_vu, range(lo, min(n, lo + chunk)))
    finally:
        gc.unfreeze()


def export_vus(programs: dict[str, dict[str, Any]], bmap: dict[str, Any], committed: Callable, weights: Callable, *, out_dir: str | Path,
               run_root: bytes, arrivals: dict[str, int] | None = None, manifest: dict[str, Any] | None = None,
               executed_prefix: dict[str, dict] | None = None, program_dirs: dict[str, str] | None = None,
               provenance: dict[str, Any] | None = None, limits: Limits | None = None) -> dict[str, Any]:
    """Draw, evaluate, decompose and write the sets.  Returns the export summary (also written to `<out_dir>/export.json`)."""
    t0 = time.perf_counter()
    limits = limits or Limits()
    arrivals = arrivals or {}
    out = Path(out_dir)
    (out / "sets").mkdir(parents=True, exist_ok=True)
    indices: dict[str, ProgramIndex] = {}
    by_inst: dict[int, ProgramIndex] = {}
    all_vus: list[dict] = []
    for rid, inst in programs.items():
        pi = by_inst.get(id(inst)) or ProgramIndex(inst, program_dir=(program_dirs or {}).get(rid))
        by_inst[id(inst)] = indices[rid] = pi
        last = int(phase_anchor(pi.n_steps, (executed_prefix or {}).get(str(rid)))["last_step"])
        vus, _ne, _nef = population(pi, rid, arrive=int(arrivals.get(rid, 0)), last_step=last)
        all_vus.extend(vus)
    pool, _not_exec = split_not_executed(all_vus, executed_prefix)
    key = selection_key(run_root)
    picked = select(pool, limits, key)
    _FORK.update(picked=picked, indices=indices, store=CommittedStore.of_record(bmap, committed, manifest), weights=weights,
                 arrivals=arrivals, key=key, limits=limits)
    writers: dict[str, _SetWriter] = {}
    fam_stats: dict[str, Counter] = {}
    fam_time: Counter = Counter()
    vus_log = []
    try:
        for k, res in enumerate(_run_draw(len(picked), int(limits.workers), float(limits.max_seconds), t0)):
            st = fam_stats.setdefault(picked[k]["family"], Counter())
            if res is None:
                st["not_drawn_time_budget"] += 1
                continue
            entry, insts, dt = res
            st["drawn"] += 1
            fam_time[entry["family"]] += dt
            if entry.get("decomposition_error"):
                st["decomposition_error"] += 1
            if entry["result"] is True:
                st["replayed_equal"] += 1
            else:
                st["not_exported_" + ("mismatch" if entry["result"] is False else "not_evaluated")] += 1
            got: Counter = Counter()
            for inst, source in insts:
                w = writers.get(inst.set_name) or writers.setdefault(inst.set_name, _SetWriter(inst.set_name, inst.relation))
                if len(w.index) >= limits.max_instances_per_set:
                    st["instances_over_cap"] += 1
                    continue
                w.add(inst, [k, entry["request"], entry["engine_step"], entry["op_path"], entry["row"], entry["phase"], source])
                got[inst.set_name] += 1
                st["instances"] += 1
            vus_log.append(dict(entry, instances=dict(got)))
            if k % 100 == 0:
                print(f"[vu-export] {k}/{len(picked)} VUs, {sum(len(x.index) for x in writers.values())} instances, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        _FORK.clear()
    prov = dict(provenance or {}, run_root=bytes(run_root).hex(),
                selection=f"verity.randomness.derive({SELECTION_SOURCE!r}, {SELECTION_DOMAIN!r}, {{run_root}}); per family key.shard(('family', "
                          "family)) through challenge.stratum_picks; sub-indices key.subset(k, n, ('row', request, i, evaluation))")
    sets = [w.write(out / "sets", prov) for _n, w in sorted(writers.items())]
    with (out / "vus.jsonl").open("w") as f:
        for e in vus_log:
            f.write(json.dumps(e, separators=(",", ":"), default=str) + "\n")
    summary = {"schema": EXPORT_SCHEMA, "provenance": prov, "limits": dataclasses.asdict(limits),
               "population": {"vus": len(all_vus), "pool": len(pool), "not_executed": len(_not_exec),
                              "pool_by_family": dict(Counter(v["family"] for v in pool)), "not_exported_families": NOT_EXPORTED},
               "by_family": {f: dict(c) | {"seconds": round(fam_time[f], 3)} for f, c in sorted(fam_stats.items())},
               "sets": sets, "wall_s": round(time.perf_counter() - t0, 3)}
    (out / "export.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def _read_port(d: Path, p: dict[str, Any], n: int) -> list[np.ndarray]:
    arr = np.frombuffer((d / p["file"]).read_bytes(), dtype=np.dtype(_WORD[int(p["dtype"][1:]) // 8][1]).newbyteorder("<"))
    if p.get("words") is not None:
        return list(arr.reshape(n, int(p["words"])))
    offs = np.frombuffer((d / p["offsets"]).read_bytes(), dtype="<u8").astype(np.int64)
    return [arr[offs[i]:offs[i + 1]] for i in range(n)]


def verify_set(d: Path) -> dict[str, Any]:
    """Every file digest against the manifest, then every instance re-evaluated under the set's relation."""
    man = json.loads((d / "manifest.json").read_text())
    for fname, f in man["files"].items():
        if hashlib.sha256((d / fname).read_bytes()).hexdigest() != f["sha256"]:
            return {"set": man["set"], "ok": False, "why": f"{fname}: sha256 differs from the manifest"}
    n, rel = int(man["n"]), man["relation"]
    P = {k: _read_port(d, p, n) for k, p in man["ports"].items()}
    idx = json.loads((d / "index.json").read_text())["rows"]
    bad = []
    for i in range(n):
        fam = rel["definition"]
        if fam in ("Gemm_v1", "Gemm_v2"):
            y, _nf = gemm_coordinate(rel["dot"], P["x"][i], P["w"][i][None, :])
            ok = int(y[0]) == int(P["y"][i][0])
        elif fam in ("Attention_v3", "Attention_v2"):
            got, _nf = attention_head(rel["dot"], rel["inv"], int(rel["statics"]["BN"]), P["q"][i], P["k"][i], P["v"][i], int(rel["statics"]["D"]))
            ok = np.array_equal(got, P["out"][i])
        elif fam == "RoPE_v1":
            ok = np.array_equal(D.rope_rows(P["x"][i][None, :], P["cs"][i]).reshape(-1), P["out"][i])
        else:
            ins = sorted((k for k in P if k.startswith("in")), key=lambda k: int(k[2:]))
            got = _outputs(EV.evaluate(fam, rel["statics"], [[P[k][i]] for k in ins], [None] * len(ins)))
            ok = all(k[4:] in got and np.array_equal(got[k[4:]].astype(P[k][i].dtype), P[k][i]) for k in P if k.startswith("out."))
        if not ok:
            bad.append(idx[i][0])
    return {"set": man["set"], "ok": not bad, "n": n, "bad": bad[:16], "bad_n": len(bad)}


def verify(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir) / "sets"
    res = [verify_set(d) for d in sorted(root.iterdir()) if (d / "manifest.json").is_file()]
    return {"ok": all(r["ok"] for r in res) and bool(res), "sets": res}


def _limits_for(out: Path, sr: dict[str, Any]) -> Limits:
    lim = Limits.from_dir(out)
    if (sr.get("evaluator") or {}).get("retention") == "device":
        lim.workers = 1                                                      # forked children cannot read device-retained blocks
    return lim


def after_replay(out_dir: str | None, sr: dict[str, Any], pair: int, run_root: bytes, programs: dict[str, dict[str, Any]], bmap: dict[str, Any],
                 reader: Callable, weights: Callable, arrivals: dict[str, int], manifest: Any, replay_kw: dict[str, Any], loaded: dict[str, dict],
                 program_digests: dict[str, str], versions: dict[str, Any], eng: Any) -> dict[str, Any] | None:
    """The Commit stage's call, right after pair 0's replay record and value source are taken: exports when `out_dir` is given and the
    replay passed.  A failure is printed and returned by name; nothing here reaches the record or a verdict."""
    if not out_dir or pair != 0 or sr.get("result") is not True:
        return None
    out = Path(out_dir)
    extra = json.loads((out / "provenance.json").read_text()) if (out / "provenance.json").is_file() else {}
    man = manifest if isinstance(manifest, dict) else None
    prov = dict(extra, model=(getattr(eng, "cp", None) or {}).get("repo"), revision=(getattr(eng, "cp", None) or {}).get("revision"),
                program_digests=program_digests, manifest_digest=(man or {}).get("manifest_digest"), pair=pair,
                versions={k: v for k, v in versions.items() if isinstance(v, (str, int, float))})
    try:
        s = export_vus(programs, bmap, reader, weights, out_dir=out, run_root=run_root, arrivals=arrivals, manifest=man,
                       executed_prefix=replay_kw.get("executed_prefix"), program_dirs={k: v["dir"] for k, v in loaded.items() if v.get("dir")},
                       provenance=prov, limits=_limits_for(out, sr))
    except Exception as e:  # noqa: BLE001  -- not part of the record: named, never a verdict
        print(f"[vu-export] FAILED: {e!r}"[:400], flush=True)
        return {"ok": False, "why": repr(e)[:400]}
    print(f"[vu-export] {len(s['sets'])} sets, {sum(x['n'] for x in s['sets'])} instances, {sum(x['bytes'] for x in s['sets'])} bytes "
          f"in {s['wall_s']} s -> {out}", flush=True)
    return s
