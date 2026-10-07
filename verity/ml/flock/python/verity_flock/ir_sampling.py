"""C-Flock lowering of ``gumbel-top-p-token-select`` (``GumbelTopPTokenSelect_v1{V}``, one vocabulary row) and its frame-v3
instance file for the statement ``verity/flock-ir-sampling/v1`` (``live/src/ir_sampling.rs``, binary ``flock-ir-sampling``,
format ``flock-ir-sampling/v1``).

The Definition is ``TemperatureScale -> TopPMask -> GumbelSelectF32``: three scans over the V lanes of the row around three
row-level values (the temperature's reciprocal, the top-p keep word, the Gumbel stream key). The lowering keeps the scans'
lane bodies in the circuit and the row-level values out of it:

- **unit** (one per lane v, all identical, lowered gate by gate from the IR composites): ``TemperatureLane(x_v, temp, rcp,
  skip) -> scaled_v``; ``TopPMaskStep``'s select ``SelectF32(kbit_v, scaled_v, -inf) -> masked_v``, its ``BitAtx{V}(keep, v)``
  resolved by the arrangement (lane v reads bit v); ``GumbelSelectStep((best, best_i, i), masked_v, key, noisy)``, its
  ``GumbelNoiseLane(key, i)`` resolved the same way (lane v reads ``g_v``), gives the carry lane v hands on. Lane 0 (``i ==
  0``) takes the scan's init ``(y_0, 0, 1)`` instead of the step, as the Definition does (``y_0`` is the step's own ``y``).
- **cut words** (public, per instance, :data:`SCALARS`, :data:`INIT`, then :data:`LANE` per lane): the row scalars ``temp``,
  ``rcp = DivFullRcp(temp)``, ``skip = temp in {0, 1}``, ``noisy = temp != 0``; the initial carry ``(0, 0, 0)``; per lane
  ``kbit_v`` (bit v of ``TopPMaskWordx{V}(scaled, top_p, splits)``), ``g_v = GumbelNoiseLane(GumbelStreamKey(seed, pos), v)``,
  ``scaled_v`` and the carry ``(best_v, best_i_v, i_v + 1)``. A unit computes ``scaled_v`` and its carry; every other cut
  word is **native**: the verifier computes it (:func:`native_words`) from the public words. The top-p keep word is native
  because its split pipeline runs the measured MUFU EX2 / RCP tables over whole-row reductions, which no circuit here
  lowers; the noise needs only the public key.
- **ports:** the logits row is a private frame-v3 row (``blake3-keyed/row/v2``, hashed in the proof like every frame-v3 row);
  the scalar operands (top_p, seed, pos, temp, splits) are public, their frame-v3 row digests computed natively; the token
  (lane V-1's ``best_i``, a u64 as the row kernel returns it) is public.

The unit netlist does not depend on V (only the arrangement does) and is pinned by :data:`PIN`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import ir_frame as IF
from . import ir_lower as IL
from .gf2 import NOT, Circuit, mux, or_reduce, sub

TEMPLATE = "gumbel-top-p-token-select"
UNIT = "gumbel-top-p-token-select/fp32/lane"
PIN = "c9aeca364cb08b68a174fa9811ff505306c6a8f9af7458689ff23e66409cc1b6"
FORMAT = "flock-ir-sampling/v1"
STATEMENT = "verity/flock-ir-sampling/v1"
SCHEME = IF.SCHEME

SCALARS = ("temp", "rcp", "skip", "noisy")
INIT = ("best", "best_i", "i")
LANE = ("kbit", "g", "scaled", "best", "best_i", "i_next")
#: unit ports: the logit (the leaf), then the cut inputs; outputs are all cut words
IN_PORTS = ("x", "temp", "rcp", "skip", "noisy", "kbit", "g", "best", "best_i", "i")
OUT_PORTS = ("scaled", "best", "best_i", "i_next")
PUBLIC_PORTS = ("in1", "in2", "in3", "in4", "in5")       # top_p, seed, pos, temp, splits
LOGITS, OUTPUT = "in0", "out.sampled_token_ids"
PT_LOCAL = 16
NEG_INF = 0xFF800000
ONE_F32 = 0x3F800000


def words(V: int) -> int:
    return len(SCALARS) + len(INIT) + len(LANE) * V


def lane_word(v: int, name: str) -> int:
    return len(SCALARS) + len(INIT) + len(LANE) * v + LANE.index(name)


def lane_ports(v: int) -> tuple[list[int], list[int]]:
    """Lane v's cut input words (``IN_PORTS[1:]``) and cut output words (``OUT_PORTS``)."""
    carry = [len(SCALARS) + k for k in range(3)] if v == 0 else [lane_word(v - 1, n) for n in ("best", "best_i", "i_next")]
    return ([0, 1, 2, 3, lane_word(v, "kbit"), lane_word(v, "g"), *carry],
            [lane_word(v, n) for n in OUT_PORTS[:1]] + [lane_word(v, n) for n in ("best", "best_i", "i_next")])


# --- pieces: ir_lower's (the lane's F32 ops, compare, selects and integer add are tail_pieces', NaN payloads included) --------

PIECES = IL.PIECES


# --- the lane unit, lowered from the IR composites ------------------------------------------------------------------------------

def _composites(V: int):
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import sampling as SM

    return SM.TemperatureLane, bind(SM.TopPMaskStep, V=V), SM.GumbelSelectStep


def _walk(C, target, args, resolved: dict):
    """Lower ``target``'s gates on ``args`` (one word per argument, ``None`` for an argument only resolved gates read); a gate
    whose primitive id is in ``resolved`` takes that word instead of a piece. Returns ``(call, wires)``."""
    _prog, call = IL.standalone(target)
    wires = {a: w for a, w in zip(call.args, args) if w is not None}
    todo = []
    for g in call:
        if g.prim.id in resolved:
            wires[g.index] = list(resolved[g.prim.id])
        else:
            todo.append(g)
    IL.lower_gates(C, todo, wires, PIECES)
    return call, wires


def lane_circuit(V: int) -> Circuit:
    """The lane unit: inputs ``in0`` (the bf16 logit) and ``in1 .. in9`` (``IN_PORTS[1:]``, 32 bits each; skip, noisy and kbit
    use bit 0), outputs ``out0 .. out3`` (``OUT_PORTS``)."""
    temp_lane, mask_step, select_step = _composites(V)
    C = Circuit(cse=True)
    x = C.input("in0", 16)
    temp, rcp, skip, noisy, kbit, g, best, best_i, i = (C.input(f"in{j}", 32) for j in range(1, 10))
    call, w = _walk(C, temp_lane, [x, temp, rcp, skip[:1]], {})
    scaled = w[call.returns[0]]
    call, w = _walk(C, mask_step, [i, scaled, None], {f"BitAtx{V}_v1": kbit[:1]})
    masked = w[call.returns[1]]
    call, w = _walk(C, select_step, [best, best_i, i, masked, None, noisy[:1]], {"GumbelNoiseLane_v1": g})
    nb, ni, nxt = (w[r] for r in call.returns[:3])
    (gt,) = [gg for gg in call if gg.prim.id == "F32GtStrict_v1"]
    y = w[gt.operands[0]]
    first = NOT(or_reduce(C, i))
    C.output("out0", scaled)
    C.output("out1", mux(C, first, y, nb))
    C.output("out2", mux(C, first, i, ni))
    C.output("out3", nxt)
    return C


@dataclass
class Lane:
    """The lane unit's circuit and its ``flock-ir-unit/v2`` layout (the leaf group, one cut input group, one cut output group)."""

    V: int
    circuit: Circuit
    layout: IL.Layout

    @property
    def name(self) -> str:
        return UNIT

    @property
    def units(self):
        """What ``ir_bench`` reads of a lowering's units: the count per instance, and cut words (so the MUFU tables, for the
        native ``DivFullRcp``)."""
        from types import SimpleNamespace

        return SimpleNamespace(n=self.V, cut=True)

    @property
    def cut(self) -> dict:
        """The cut structure the netlist pins: the word count, no tail program (every unit-less cut word is native), the native
        rule, and per lane its cut ports as ``[column, cut word]``."""
        lay = self.layout
        cin = lay.in_port_cols[1:]
        cout = [c - lay.out_col * IL.WORD for c in lay.out_port_cols]
        ins, outs = [], []
        for v in range(self.V):
            a, b = lane_ports(v)
            ins.append([[c, k] for c, k in zip(cin, a)])
            outs.append([[c, k] for c, k in zip(cout, b)])
        return {"words": words(self.V), "tail": [], "native": STATEMENT, "in": ins, "out": outs}

    @property
    def text(self) -> str:
        return IL.netlist(self.layout, self.name, self.cut)

    @property
    def unit_text(self) -> str:
        """The netlist without its cut line: what does not depend on V (:data:`PIN`)."""
        return IL.netlist(self.layout, self.name)

    @property
    def sha256(self) -> str:
        return IL.digest(self.text)

    def evaluate(self, ports: list[np.ndarray], flip: tuple[int, int] | None = None):
        """``(output words per port, lane mask of satisfied lanes)`` of the laid-out rows over lane-major port words."""
        return IL.evaluate_layout(self.layout, ports, len(ports[0]), flip)


@lru_cache(maxsize=None)
def lane(V: int) -> Lane:
    C = lane_circuit(V)
    lay = IL.layout(C, [f"in{j}" for j in range(len(IN_PORTS))], [f"out{j}" for j in range(len(OUT_PORTS))], [1], None)
    return Lane(V, C, lay)


# --- the words of one row, as the verifier computes them ------------------------------------------------------------------------

def _prim(name: str):
    from verity_vllm.program.registry import prims as P
    from verity.ml.circuits.ops import scalar as S

    return getattr(P, name, None) or getattr(S, name)


def row_scalars(temp: int) -> list[int]:
    """``[temp, rcp, skip, noisy]`` by the IR primitives (``TemperatureScale``'s and ``GumbelSelectF32``'s row values)."""
    eq = _prim("F32Eq").evaluate
    rcp = _prim("DivFullRcp").evaluate(temp)
    return [temp & 0xFFFFFFFF, rcp, eq(temp, 0) | eq(temp, ONE_F32), 1 - eq(temp, 0)]


def _add_ftz_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``F32AddFtz_v1`` lane for lane: the vectorised twin, and the primitive itself on every lane with a NaN operand."""
    from verity_vllm.program.registry import sampling_rows as SR

    out = np.asarray(SR.add_ftz(a, b), dtype=np.uint32)
    nan = ((a & 0x7F800000) == 0x7F800000) & ((a & 0x7FFFFF) != 0) | ((b & 0x7F800000) == 0x7F800000) & ((b & 0x7FFFFF) != 0)
    prim = _prim("F32AddFtz").evaluate
    for v in np.flatnonzero(nan):
        out[v] = prim(int(a[v]), int(b[v]))
    return out


def _scan(y: np.ndarray):
    """``GumbelSelectF32``'s scan over y: every lane's carry out ``(best, best_i, i + 1)`` (strict ``>``, NaN never wins, a
    NaN ``y_0`` pins the best)."""
    V = len(y)
    idx = np.arange(V, dtype=np.int64)
    yf = y.view(np.float32)
    if np.isnan(yf[0]):
        return np.full(V, y[0], np.uint32), np.zeros(V, np.uint32), (idx + 1).astype(np.uint32)
    cand = np.where(np.isnan(yf), np.float32(-np.inf), yf)
    run = np.maximum.accumulate(cand)
    new = np.zeros(V, bool)
    new[0] = True
    new[1:] = cand[1:] > run[:-1]
    best_i = np.maximum.accumulate(np.where(new, idx, 0))
    return y[best_i].astype(np.uint32), best_i.astype(np.uint32), (idx + 1).astype(np.uint32)


def row_words(logits: np.ndarray, top_p: int, seed: int, pos: int, temp: int, splits: int) -> tuple[np.ndarray, int]:
    """Every cut word of one instance (:func:`words` of them) and its token: the units' words by the vectorised twins of the
    IR primitives (``verity_vllm.program.registry.sampling_rows``, held word-equal to them), the native ones by the primitives
    and the top-p split pipeline (``TopPMaskWordx{V}``'s own reference)."""
    from verity_vllm.program.registry import sampling as SM
    from verity_vllm.program.registry import sampling_rows as SR

    logits = np.asarray(logits, dtype=np.uint16).reshape(-1)
    V = len(logits)
    temp_, rcp, skip, noisy = row_scalars(temp)
    scaled = np.asarray(SR.temperature_scale_row(logits, temp_), dtype=np.uint32)
    keep = SM.topp_keep(scaled.view(np.float32), SM._f(int(top_p)), int(splits))
    masked = np.where(keep, scaled, np.uint32(NEG_INF)).astype(np.uint32)
    g = np.asarray(SR.noise_row(SM.stream_key(int(seed), int(pos)), V), dtype=np.uint32)
    y = _add_ftz_rows(masked, g) if noisy else masked
    best, best_i, i_next = _scan(y)
    out = np.zeros(words(V), dtype=np.uint32)
    out[:4] = [temp_, rcp, skip, noisy]
    lane = out[len(SCALARS) + len(INIT):].reshape(V, len(LANE))
    for k, col in enumerate((keep.astype(np.uint32), g, scaled, best, best_i, i_next)):
        lane[:, k] = col
    return out, int(best_i[-1])


def native_words(cuts: np.ndarray, top_p: int, seed: int, pos: int, temp: int, splits: int) -> np.ndarray:
    """The native cut words of one instance recomputed from its public words (the scalar ports and the units' ``scaled``
    words): what :func:`check_native` holds a file's cut words to."""
    from verity_vllm.program.registry import sampling as SM
    from verity_vllm.program.registry import sampling_rows as SR

    V = (len(cuts) - len(SCALARS) - len(INIT)) // len(LANE)
    lane = np.asarray(cuts[len(SCALARS) + len(INIT):], dtype=np.uint32).reshape(V, len(LANE))
    scaled = lane[:, LANE.index("scaled")].copy()
    out = np.asarray(cuts, dtype=np.uint32).copy()
    out[:4] = row_scalars(temp)
    out[4:7] = 0
    nl = out[len(SCALARS) + len(INIT):].reshape(V, len(LANE))
    nl[:, LANE.index("kbit")] = SM.topp_keep(scaled.view(np.float32), SM._f(int(top_p)), int(splits))
    nl[:, LANE.index("g")] = SR.noise_row(SM.stream_key(int(seed), int(pos)), V)
    return out


# --- the instance file ----------------------------------------------------------------------------------------------------------

def _scalar(input_set, name: str, lo: int, hi: int) -> np.ndarray:
    return np.asarray(input_set.port(name, lo, hi), dtype=np.uint64).reshape(hi - lo, -1)[:, 0]


def instances(input_set, lo: int, hi: int):
    """``(logits (n, V) u16, scalars {port: (n,) u64}, outputs (n,) u64)`` of ids [lo, hi)."""
    n = hi - lo
    logits = np.asarray(input_set.port(LOGITS, lo, hi), dtype=np.uint16).reshape(n, -1)
    scalars = {p: _scalar(input_set, p, lo, hi) for p in PUBLIC_PORTS}
    return logits, scalars, _scalar(input_set, OUTPUT, lo, hi)


def check_template(sub) -> None:
    names = [p.name for p in sub.inputs]
    if names != [LOGITS, *PUBLIC_PORTS] or [p.name for p in sub.outputs] != [OUTPUT] or sub.inputs[0].dtype != "u16":
        raise ValueError(f"{sub.id}: not GumbelTopPTokenSelect_v1's ports")


def plan(low: Lane, V: int, k_max: int = IF.K_MAX):
    """The frame-v3 block plan for rows of V bf16 logits: runs of ``nb`` compressions, each read by its ``32 nb`` lanes (every
    run of every chunk the same shape, the last chunk of a row possibly shorter), and the packing with the best fill."""
    row_bytes = 2 * V
    if row_bytes % IF.BLOCK:
        raise ValueError(f"a {row_bytes}-byte row is not whole 64-byte blocks")
    blocks = [min(16, (row_bytes - IF.CHUNK * c) // IF.BLOCK) for c in range(-(-row_bytes // IF.CHUNK))]
    ul = max(7, (low.layout.useful - 1).bit_length())
    # a unit port group's region claim has 7 + log2(its words) + log2(units per block) free bits, at most PT_LOCAL (16)
    max_units = 1 << (PT_LOCAL - 7 - max(w for _, w, _ in (*low.layout.in_groups, *low.layout.out_groups)).bit_length() + 1)
    nb = 16
    while nb >= 1:
        if all(b % nb == 0 for b in blocks):
            upc = 32 * nb
            best = None
            for G in range(1, IF.MAX_SLOTS + 1):
                Q, units = 1 << (G - 1).bit_length(), G * upc
                C = Q * nb
                if C > IF.MAX_SLOTS or units > min(IF.MAX_SLOTS, max_units):
                    break
                if units & (units - 1):
                    continue
                pos0 = -(-(C << IF.SUB_LOG) // (1 << ul))
                pos0 = -(-pos0 // units) * units
                k_log = max(IF.K_MIN, (((pos0 + units) << ul) - 1).bit_length())
                if k_log > k_max:
                    continue
                fill = ((C << IF.SUB_LOG) + units * low.layout.useful) / (1 << k_log)
                if best is None or fill > best[0] + 1e-9:
                    best = (fill, G, C, pos0, units, k_log)
            if best is not None:
                _, G, C, pos0, units, k_log = best
                return {"nb": nb, "chunk_blocks": blocks, "G": G, "comp_slots": C, "unit_log": ul, "unit_pos0": pos0,
                        "units_per_block": units, "k_log": k_log, "units_per_comp": upc}
        nb //= 2
    raise ValueError(f"no run of a chunk's compressions fits a block of at most 2^{k_max} bits with 32 lanes per compression")


def statement(input_set, lo: int, hi: int, logits: np.ndarray, scalars: dict, outs: np.ndarray):
    """The frame-v3 roots over instances [lo, hi): the logits rows and each scalar port as ``blake3-keyed/row/v2`` row leaves
    (scalar ports as rows of 32-bit words: a u64 is two), the token as ``u64`` word leaves; each tree under a domain bound
    to the set, the range, the port and its schema (``ir_frame.BINDING_TAG``). Returns the header's ``frame_v3`` entry and
    the logits row digests."""
    import blake3
    from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
    from verity.ml.commitments.identity import identity_digest
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain
    from verity.ml.commitments.rowleaf import ROLE_X, SCHEMA_BLAKE3_ROW, blake3_row_key

    n, V = logits.shape
    widths = {p: (2 if input_set.subcircuit.inputs[[q.name for q in input_set.subcircuit.inputs].index(p)].dtype == "u64" else 1)
              for p in PUBLIC_PORTS}
    leaves = {LOGITS: RowLeaf("blake3", ROLE_X, 16, V), **{p: RowLeaf("blake3", ROLE_X, 32, widths[p]) for p in PUBLIC_PORTS},
              OUTPUT: WordLeaf("u64", 8)}
    schemas = {p: (SCHEMA_BLAKE3_ROW if p != OUTPUT else "u64") for p in leaves}
    bind, doms = {}, {}
    for name in leaves:
        bind[name] = bytes.fromhex(identity_digest(IF.BINDING_TAG, {"set": input_set.name, "content_digest": input_set.content_digest,
                                                                   "lo": int(lo), "hi": int(hi), "port": name, "schema": schemas[name]}))
        doms[name] = CommitmentDomain(bind[name], IF.OWNER, RangeIndexedDomain(0, n))
    fv = FrameV3({name: Port(doms[name], leaf) for name, leaf in leaves.items()})
    key = blake3_row_key(ROLE_X)
    digests = np.frombuffer(b"".join(blake3.blake3(np.ascontiguousarray(logits[i], dtype="<u2").tobytes(), key=key).digest()
                                     for i in range(n)), dtype=np.uint8).reshape(n, 32)
    roots = {LOGITS: fv.root(LOGITS, [fv.tree_leaf(LOGITS, i, digests[i].tobytes()) for i in range(n)]).hex()}
    for p in PUBLIC_PORTS:
        rows = [int(v).to_bytes(4 * widths[p], "little") for v in scalars[p].tolist()]
        roots[p] = fv.root(p, [fv.tree_leaf(p, i, blake3.blake3(r, key=key).digest()) for i, r in enumerate(rows)]).hex()
    roots[OUTPUT] = fv.root(OUTPUT, [fv.leaf_layout(OUTPUT, i).digest(int(v).to_bytes(8, "big")) for i, v in enumerate(outs.tolist())]).hex()
    return ({"roots": roots, "domain_ids": {k: d.domain_id.hex() for k, d in doms.items()}, "bindings": {k: b.hex() for k, b in bind.items()},
             "owner": IF.OWNER, "schemas": schemas, "key": key.hex(), "public_words": {p: widths[p] for p in PUBLIC_PORTS}}, digests)


def write(low: Lane, input_set, lo: int, hi: int, path, k_max: int = IF.K_MAX) -> dict:
    """``flock-ir-sampling/v1``: one JSON header line (the plan, every block's runs as ``[instance, port 0, chunk, run]`` and
    lanes, the frame-v3 statement), then every instance's logits row (u16 LE), its row digest (32 bytes), its public port
    words (u32 LE, ``public_words`` per port: a u64 as lo, hi) and its cut words (u32 LE)."""
    sub = input_set.subcircuit
    check_template(sub)
    logits, scalars, outs = instances(input_set, lo, hi)
    n, V = logits.shape
    if V != low.V:
        raise ValueError(f"the lane netlist is for V = {low.V}, the set's rows have {V} logits")
    cuts = np.zeros((n, words(V)), dtype=np.uint32)
    for i in range(n):
        cuts[i], tok = row_words(logits[i], *(int(scalars[p][i]) for p in PUBLIC_PORTS))
        if tok != int(outs[i]):
            raise ValueError(f"instance {lo + i}: the words give token {tok}, the set records {int(outs[i])}")
    fv, digests = statement(input_set, lo, hi, logits, scalars, outs)
    pl = plan(low, V, k_max)
    nb, upc, G = pl["nb"], pl["units_per_comp"], pl["G"]
    runs = [(c, r) for c, b in enumerate(pl["chunk_blocks"]) for r in range(b // nb)]
    comps = [(i, c, r) for i in range(n) for c, r in runs]
    chunk_slots = pl["comp_slots"] // nb
    blocks = []
    for b in range(-(-len(comps) // G)):
        chunks, units = [], []
        for gi in range(b * G, (b + 1) * G):
            if gi >= len(comps):
                chunks.append(None)
                units += [None] * upc
                continue
            i, c, r = comps[gi]
            chunks.append([i, 0, c, r])
            first = (IF.CHUNK * c + IF.BLOCK * nb * r) // 2
            units += [i * V + first + k for k in range(upc)]
        blocks.append([chunks + [None] * (chunk_slots - len(chunks)), units])
    lay = low.layout
    wiring = [[[k // upc, 2 * (k % upc)]] for k in range(pl["units_per_block"])]
    head = {"format": FORMAT, "statement": STATEMENT, "scheme": SCHEME, "unit": low.name, "unit_sha256": low.sha256,
            "set": input_set.name, "subcircuit": sub.id, "content_digest": input_set.content_digest, "range": [lo, hi], "instances": n,
            "units_per_instance": V, "units": n * V, "in_ports": [[LOGITS, V]], "out_ports": [],
            "public_ports": [[p, fv["public_words"][p]] for p in PUBLIC_PORTS], "out_cut": [[OUTPUT, lane_word(V - 1, "best_i")]],
            "layout": {"sub_log": IF.SUB_LOG, "comp_slots": pl["comp_slots"], "nb": nb, "chunk_nb": 16, "chunk_blocks": pl["chunk_blocks"],
                       "chunks_per_block": chunk_slots, "units_per_block": pl["units_per_block"], "unit_log": pl["unit_log"],
                       "unit_pos0": pl["unit_pos0"], "k_log": pl["k_log"], "G": G, "chunks_per_comp": 1, "units_per_comp": upc,
                       "wiring": wiring, "leaf_cols": [lay.in_port_cols[0]], "ret_group": [lay.out_col, 0], "ret_cols": [],
                       "out_leaf": [[] for _ in range(V)]},
            "blocks": blocks, "cut_words": words(V), "frame_v3": fv}
    pub = np.stack([scalars[p] for p in PUBLIC_PORTS], axis=1)
    with open(path, "wb") as f:
        f.write(json.dumps(head, sort_keys=True).encode() + b"\n")
        f.write(np.ascontiguousarray(logits, dtype="<u2").tobytes())
        f.write(digests.tobytes())
        for i in range(n):
            for k, p in enumerate(PUBLIC_PORTS):
                f.write(int(pub[i, k]).to_bytes(4 * fv["public_words"][p], "little"))
        f.write(np.ascontiguousarray(cuts, dtype="<u4").tobytes())
    return head


def stage(low: Lane, input_set, lo: int, hi: int, out_dir, k_max: int = IF.K_MAX) -> Path:
    """Write ids [lo, hi) of ``input_set`` as ``<out_dir>/frame-<n>.bin`` and the lane netlist as ``<out_dir>/net.txt``: the
    verifier's own staging (red-team-flock-2 IR2), every cut word the verifier's own computation."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "net.txt").write_text(low.text)
    path = out / f"frame-{hi - lo}.bin"
    write(low, input_set, lo, hi, path, k_max)
    return path


def read(path):
    """``(header, logits (n, V), digests (n, 32), public words {port: (n,) u64}, cuts (n, words))`` of an instance file."""
    b = Path(path).read_bytes()
    nl = b.index(b"\n")
    head = json.loads(b[:nl])
    n, V = head["instances"], head["units_per_instance"]
    at = nl + 1
    logits = np.frombuffer(b, "<u2", n * V, at).reshape(n, V)
    at += 2 * n * V
    digests = np.frombuffer(b, np.uint8, 32 * n, at).reshape(n, 32)
    at += 32 * n
    pw = [(p, w) for p, w in head["public_ports"]]
    per = 4 * sum(w for _, w in pw)
    pub = {p: np.zeros(n, np.uint64) for p, _ in pw}
    for i in range(n):
        o = at + i * per
        for p, w in pw:
            pub[p][i] = int.from_bytes(b[o:o + 4 * w], "little")
            o += 4 * w
    at += n * per
    cuts = np.frombuffer(b, "<u4", n * head["cut_words"], at).reshape(n, head["cut_words"])
    if at + 4 * n * head["cut_words"] != len(b):
        raise ValueError("instance file length")
    return head, logits, digests, pub, cuts


def check_native(path) -> list[str]:
    """The verifier's check of a file's native cut words: each recomputed from the file's public words (the scalar ports and
    the units' ``scaled`` words), and the token the file's out port commits equal to lane V-1's ``best_i`` (empty: all hold).
    ``flock-ir-sampling`` checks the units' words in the proof; this is the part of the statement it does not evaluate."""
    head, _logits, _d, pub, cuts = read(path)
    bad = []
    for i in range(head["instances"]):
        want = native_words(cuts[i], *(int(pub[p][i]) for p in PUBLIC_PORTS))
        diff = np.flatnonzero(want != cuts[i])
        if len(diff):
            bad.append(f"instance {i}: {len(diff)} native cut words differ from the verifier's (first: word {int(diff[0])})")
    return bad


def public_sha256(path) -> str:
    """sha256 of the file's public bytes (the header line and everything after the logits rows): the Σ input
    ``flock-ir-sampling`` computes, and what ``serve --native-checked`` names."""
    b = Path(path).read_bytes()
    nl = b.index(b"\n")
    head = json.loads(b[:nl])
    return hashlib.sha256(b[:nl] + b[nl + 1 + 2 * head["instances"] * head["units_per_instance"]:]).hexdigest()


def serve_args(path) -> list[str]:
    """``flock-ir-sampling serve`` refuses a sampling file until its native words are checked: :func:`check_native`, then
    ``--native-checked <public sha256>``."""
    bad = check_native(path)
    if bad:
        raise ValueError(f"{path}: {bad[0]}")
    return ["--native-checked", public_sha256(path)]


def check_staged(low: Lane, input_set, path) -> list[str]:
    """What differs between the instance file at ``path`` and the verifier's own staging of the same range (IR2)."""
    import tempfile

    head = json.loads(Path(path).read_bytes().split(b"\n", 1)[0])
    lo, hi = head["range"]
    with tempfile.TemporaryDirectory() as d:
        mine = stage(low, input_set, lo, hi, d)
        if mine.read_bytes() == Path(path).read_bytes():
            return []
        a, b = mine.read_bytes().split(b"\n", 1), Path(path).read_bytes().split(b"\n", 1)
        return [f"{what} differs from the verifier's own staging of {input_set.name} [{lo}, {hi})"
                for what, x, y in (("header", a[0], b[0]), ("body", a[1], b[1])) if x != y]


def unit_ports(cuts: np.ndarray, logits: np.ndarray) -> list[np.ndarray]:
    """Per lane-unit input port, the lane-major words of every (instance, lane) (instance-major): what the units read."""
    n, V = logits.shape
    ports = [np.asarray(logits, dtype=np.uint64).reshape(-1)]
    idx = np.array([lane_ports(v)[0] for v in range(V)], dtype=np.int64)            # (V, 9)
    for j in range(idx.shape[1]):
        ports.append(np.asarray(cuts, dtype=np.uint64)[:, idx[:, j]].reshape(-1))
    return ports


def unit_outputs(cuts: np.ndarray, V: int) -> list[np.ndarray]:
    idx = np.array([lane_ports(v)[1] for v in range(V)], dtype=np.int64)
    return [np.asarray(cuts, dtype=np.uint64)[:, idx[:, j]].reshape(-1) for j in range(idx.shape[1])]


def pin_digest(low: Lane) -> str:
    return hashlib.sha256(low.unit_text.encode()).hexdigest()
