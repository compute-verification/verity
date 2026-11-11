"""GLM-5.3's MoE expert GEMMs cut per expert: a (token, slot) pair's GEMM over its expert's rows alone, each block of weight rows
read at the router's slot id from a table registered one expert a row, so no proof block holds another expert's weights.

A MoE layer's pair (t, k), for each projection (w13, and w2 with the routed weight g):

    MoeRouterSigmoidBiasSlots_v2{E, TOPK, SCALE, KERNEL}  (logits, bias) -> (g_0 .. g_{TOPK-1}, e_0 .. e_{TOPK-1})  a unit a token
    EmbeddingRead_v1{V=E, H=N K / 2}      (tok, root, domain, row, salt, path) -> (h, ok)  rows [bN, bN + N) of expert tok
    EmbeddingRead_v1{V=E, H=2 K / G}      (tok, root, domain, row, salt, path) -> (h, ok)  their G-block's scale row
    MoeExpertFp8BlockGemm(W)_v4{K, N, G, KB, SWAP, DOT}   (xq, sx, [g,] w, ws) -> [N] bf16

A table is a registered parameter of its own (`interface.Table`): its row e is the block's rows of expert e as their e4m3 bytes
in u16 words (a row's little-endian bytes, `interface.words`), or the scale row's f32 words. Its root and domain id are table
reads, and its row, salt and path an opening at the hidden key ``tok``, which the verdict ``o1`` holds to the root. ``tok`` is
the router's row ``o{TOPK + k}``, ``g`` its row ``o{k}``, and ``w`` and ``ws`` the reads' ``h``, each a wire. Fused, the
weight rows' read and its GEMM are one unit, `MoeExpertFp8ReadGemm(W)_v1{READ, K, N, G, KB, SWAP, DOT}` (tok, root, domain,
row, salt, path, xq, sx, [g,] ws) -> (y, ok) with READ = EmbeddingRead_v1{V=E, H=N K / 2}, and its table declared on it, so
no unit outputs a weight row.

* the cut computes the gather form: the router's slots, the reads of the experts they name and the per-expert GEMMs over the
  rows read are `MoeExpertFp8Gemm(W)_v1` over the whole stack, at small statics; a row read at another expert does not open;
* a fused unit is its read then its GEMM, evaluated on its gates at small statics;
* the window interface takes it at GLM-5.3's TP8 statics, read and GEMM apart or fused: each template is a proof block Flock
  composes (`check_proof_block`), the wires and declarations check (`Circuit`), a window commits its tables, and with the
  tables' parameters pinned the free input bits are the router's logits and bias and the GEMMs' activations alone, which the
  served circuit wires from the blocks before; unpinned, the openings are free too. Fused, the weight rows are no unit's
  output.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.evaluation.bits import evaluate_bits
from verity.definitions.ops.boolean import gemm as BG
from verity.definitions.ops.prims import HopperE4m3MmaV2Dot32 as MV
from verity.definitions.ops.prims import HopperE4m3QgmmaDot32 as HP
from verity.commitments import row_table as RT
from verity.commitments.gates import embedding as EMB
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import bit_row
from verity.definitions.registry import boolean_fp8_moe as BM
from verity.definitions.registry import boolean_moe as BMOE
from verity.definitions.registry import fp8_moe as FM
from verity.definitions.registry import fp8_moe_experts as X
from verity.definitions.registry import moe as MOE
from verity.commitments.windows import interface as WI


# --- the cut computes the gather form --------------------------------------------------------------------------------------------

def _e4m3(rng: np.random.Generator, *shape: int) -> np.ndarray:
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] ^= 1
    return b


def _f32(rng: np.random.Generator, lo: float, hi: float, *shape: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32).view(np.uint32)


class _Table:
    """``rows`` (one an expert, u16 words) registered as a table: its dense domain (owner -2), salts and frame-v3 tree."""

    def __init__(self, rng: np.random.Generator, rows: np.ndarray) -> None:
        self.rows, V = rows, len(rows)
        self.dom = CommitmentDomain(rng.bytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
        self.salts = [rng.bytes(EMB.SALT_BYTES) for _ in range(V)]
        self.tree = RT.commit(self.dom, [bit_row((r.tolist(), 16)) for r in rows], self.salts)

    def read(self, tok: int, at: int | None = None) -> tuple[list[int], int]:
        """`EmbeddingRead_v1`'s (h, ok) at key ``tok`` on the opening of row ``at`` (``tok``'s own by default)."""
        e = tok if at is None else at
        p = self.tree.path(e)
        return EMB.reference(len(self.rows), tok, self.tree.root, self.dom.domain_id, self.rows[e].tolist(), self.salts[e],
                             [p[k:k + 64] for k in range(0, len(p), 64)])


def _tables(rng: np.random.Generator, w: np.ndarray, ws: np.ndarray, N: int) -> tuple[list[_Table], list[_Table]]:
    """A projection's tables: one a block of N weight rows (its rows' e4m3 bytes as words) and one a G-block's scale row."""
    E_, R, K = w.shape
    rows = [_Table(rng, WI.words(w[:, b * N:(b + 1) * N, :], 8, E_, N * K // 2)) for b in range(R // N)]
    return rows, [_Table(rng, WI.words(ws[:, nb, :], 32, E_, 2 * ws.shape[2])) for nb in range(ws.shape[1])]


def _per_expert(gemm, rows: list[_Table], scales: list[_Table], e: int, lead: tuple, N: int, G: int) -> tuple[int, ...]:
    """The pair's output words by the cut: per block, its rows and its G-block's scale row read at ``e``, then ``gemm``."""
    out = []
    for b, table in enumerate(rows):
        (h, ok), (hs, oks) = table.read(e), scales[b * N // G].read(e)
        assert ok == oks == 1
        w = np.array(h, "<u2").view(np.uint8)
        out += evaluate(gemm, *lead, w.tolist(), np.array(hs, "<u2").view("<u4").tolist())
    return tuple(out)


@pytest.mark.parametrize("routed", [False, True], ids=["w13", "w2"])
def test_the_cut_is_the_gather_form(routed: bool) -> None:
    """At E = 8 experts, 2 slots, K = 64, G = KB = 32 and 64 weight rows in reads of 8 (two G-blocks of four reads): for each
    token, the router's slot ids name the rows read, and the per-expert GEMMs over them are the gather form's words over the
    whole stack and the same id; each read opens (ok = 1) at its expert's row and at no other's."""
    rng = np.random.default_rng(0xC164 + routed)
    E_, TOPK, K, G, KB, R, N = 8, 2, 64, 32, 32, 64, 8
    SWAP, DOT = (True, HP) if routed else (False, MV)
    st = {"K": K, "G": G, "KB": KB, "SWAP": SWAP, "DOT": DOT}
    w, ws = _e4m3(rng, E_, R, K), _f32(rng, 1e-4, 1e-2, E_, R // G, K // G)
    rows, scales = _tables(rng, w, ws, N)
    router = bind(MOE.MoeRouterSigmoidBiasSlots, E=E_, TOPK=TOPK, SCALE=2.5, KERNEL="block")
    gather = bind(X.MoeExpertFp8GemmW if routed else X.MoeExpertFp8Gemm, E=E_, N=R, **st)
    block = bind(FM.MoeExpertFp8BlockGemmWV3 if routed else FM.MoeExpertFp8BlockGemmV3, N=N, **st)
    seen = set()
    for _token in range(3):
        logits = (rng.standard_normal(E_) * 2).astype(np.float32).view(np.uint32).tolist()
        bias = (rng.standard_normal(E_) * 1e-2).astype(np.float32).view(np.uint32).tolist()
        slots = evaluate(router, logits, bias)
        for k in range(TOPK):
            g, e = slots[k], slots[TOPK + k]
            assert 0 <= e < E_
            seen.add(e)
            xq, sx = _e4m3(rng, K).tolist(), _f32(rng, 1e-3, 1e-1, K // G).tolist()
            lead = (xq, sx, g) if routed else (xq, sx)
            want = evaluate(gather, xq, sx, e, *lead[2:], w.reshape(-1).tolist(), ws.reshape(-1).tolist())
            assert _per_expert(block, rows, scales, e, lead, N, G) == tuple(want)
            other = (e + 1) % E_
            assert [t.read(other, at=e)[1] for t in rows + scales] == [0] * (len(rows) + len(scales))
    assert len(seen) > 1


def _bits(words, width: int) -> list[int]:
    return [(int(w) >> j) & 1 for w in words for j in range(width)]


@pytest.mark.slow
@pytest.mark.parametrize("routed", [False, True], ids=["w13", "w2"])
def test_a_fused_unit_is_its_read_then_its_gemm(routed: bool) -> None:
    """Minutes, most of it tracing the read's SHA-512 parts: the only evaluation of the fused unit's wiring (the read's h into
    its GEMM's w, byte for byte), at its sidecar's statics, two experts and w13's one 32-byte row or w2's two. Read at each
    expert's key, y is the GEMM over that expert's rows and ok = 1; at the other expert's key, ok = 0; at a key past V, y is
    the GEMM over the read's fill and ok = 1."""
    rng = np.random.default_rng(0xC187 + routed)
    V, K, G = 2, 32, 32
    N, SWAP, DOT, WORD = (2, True, BG.HopperE4m3QgmmaDot32, HP) if routed else (1, False, BG.HopperE4m3MmaV2Dot32, MV)
    fused = bind(BM.MoeExpertFp8ReadGemmWV1 if routed else BM.MoeExpertFp8ReadGemmV1, READ=EMB.read(V, N * K // 2), K=K, N=N, G=G,
                 KB=G, SWAP=SWAP, DOT=DOT)
    gemm = bind(FM.MoeExpertFp8BlockGemmWV3 if routed else FM.MoeExpertFp8BlockGemmV3, K=K, N=N, G=G, KB=G, SWAP=SWAP, DOT=WORD)
    table = _Table(rng, WI.words(_e4m3(rng, V, N, K), 8, V, N * K // 2))
    cases, want = [], []
    for tok, at in ((0, 0), (1, 1), (1, 0), (V, 0)):
        xq, sx, g, ws = _e4m3(rng, K), _f32(rng, 1e-3, 1e-1, K // G), _f32(rng, 0.1, 1.0, 1), _f32(rng, 1e-4, 1e-2, K // G)
        p = table.tree.path(at)
        read = EMB.arguments(V, tok, table.tree.root, table.dom.domain_id, table.rows[at].tolist(), table.salts[at],
                             [p[k:k + 64] for k in range(0, len(p), 64)])
        lead = (_bits(xq, 8), _bits(sx, 32)) + ((_bits(g, 32),) if routed else ())
        cases.append((*read, *lead, _bits(ws, 32)))
        h, ok = table.read(tok, at)
        w = np.array(h, "<u2").view(np.uint8).tolist()
        want.append((tuple(evaluate(gemm, xq.tolist(), sx.tolist(), *g.tolist()[:routed], w, ws.tolist())), ok))
    got = evaluate_bits(fused, *[np.array([c[k] for c in cases], dtype=np.uint8) for k in range(len(cases[0]))])
    got = got.reshape(len(cases), -1)
    assert [(tuple(int(sum(int(b) << j for j, b in enumerate(r[16 * n:16 * n + 16]))) for n in range(N)), int(r[16 * N]))
            for r in got] == want
    assert [ok for _y, ok in want] == [1, 1, 0, 1]


# --- the window interface takes it at GLM-5.3's statics --------------------------------------------------------------------------

#: GLM-5.3 at TP8, a rank's expert shard: E experts, TOPK slots, w13's 512 rows over K = 6144 and w2's 6144 rows over K = 256
E, TOPK, G = 256, 8, 128
UP = {"K": 6144, "R": 512, "N": 1}
DOWN = {"K": 256, "R": 6144, "N": 32}
ROUTER = bind(BMOE.MoeRouterSigmoidBiasSlots, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block")
GEMMS = {"up": bind(BM.MoeExpertFp8BlockGemmV4, K=UP["K"], N=UP["N"], G=G, KB=G, SWAP=False, DOT=BG.HopperE4m3MmaV2Dot32),
         "down": bind(BM.MoeExpertFp8BlockGemmWV4, K=DOWN["K"], N=DOWN["N"], G=G, KB=G, SWAP=True, DOT=BG.HopperE4m3QgmmaDot32)}
#: each GEMM over the rows it reads itself, at its GEMM's statics
FUSED = {p: bind(fused, READ=EMB.read(E, sh["N"] * sh["K"] // 2), **GEMMS[p].bindings)
         for p, fused, sh in (("up", BM.MoeExpertFp8ReadGemmV1, UP), ("down", BM.MoeExpertFp8ReadGemmWV1, DOWN))}
#: the ports the cut leaves to the circuit around it: the router's logits and bias, each GEMM's activation and its scales
OPEN_PORTS = {"router": ("logits", "bias"), "up": ("xq", "sx"), "down": ("xq", "sx")}


def _cut(layers: int, tokens: int, fused: bool = False) -> tuple[WI.Circuit, dict[str, int]]:
    """``layers`` MoE layers of ``tokens`` tokens cut per expert, and each registered parameter's leaves; ``fused``, each
    GEMM's weight rows read by the GEMM's own unit (`FUSED`), the tables and the scale reads as they are.

    A read template's units are table-major, ((l B + b) TOPK + k) T + t, so a table's opening and table reads are one
    declaration each and its tok wires one a slot; w13's GEMM units are its rows read's, w2's slot-major, ((l TOPK + k) B + b)
    T + t, so its routed weight is one wire a slot. A fused unit is its table's reader, so table-major, its routed weight one
    wire a slot as its tok is."""
    L, T = layers, tokens
    rid = descriptor_id(ROUTER)
    templates: list[tuple[object, int]] = [(ROUTER, L * T)]
    wires: list[WI.Wire] = []
    params: dict[str, int] = {}
    tables, roots, openings = [], [], []

    def read(name: str, H: int, B: int, fn=None, routed: bool = False) -> str:
        fn = fn or EMB.read(E, H)
        tid = descriptor_id(fn)
        templates.append((fn, L * B * TOPK * T))
        for l in range(L):
            for b in range(B):
                at, param = (l * B + b) * TOPK * T, f"moe.{l}.{name}.{b}@{H}/table"
                params[param] = H * E
                tables.append(WI.Table(param, param, 0, H, E))
                roots.extend(WI.TableRead(tid, p, param, p, at, TOPK * T) for p in ("root", "domain"))
                openings.append(WI.Opening(tid, param, "tok", ("row", "salt", "path"), "o1", at, TOPK * T))
                wires.extend(WI.Wire(tid, "tok", rid, f"o{TOPK + k}", at + k * T, (T,), l * T, (1,)) for k in range(TOPK))
                if routed:
                    wires.extend(WI.Wire(tid, "g", rid, f"o{k}", at + k * T, (T,), l * T, (1,)) for k in range(TOPK))
        return tid

    for proj, sh in (("up", UP), ("down", DOWN)):
        K, R, N = sh["K"], sh["R"], sh["N"]
        B, NBG, per = R // N, R // G, G // N
        if fused:
            scales = read(f"{proj}.ws", 2 * K // G, NBG)
            fid = read(f"{proj}.w", N * K // 2, B, FUSED[proj], routed=proj == "down")
            wires.append(WI.Wire(fid, "ws", scales, "o0", 0, (L, NBG, per, TOPK * T), 0, (NBG * TOPK * T, TOPK * T, 0, 1)))
            continue
        rows, scales, gemm = read(f"{proj}.w", N * K // 2, B), read(f"{proj}.ws", 2 * K // G, NBG), GEMMS[proj]
        gid = descriptor_id(gemm)
        templates.append((gemm, L * B * TOPK * T))
        if proj == "up":
            wires += [WI.Wire(gid, "w", rows, "o0", 0, (L * B * TOPK * T,), 0, (1,)),
                      WI.Wire(gid, "ws", scales, "o0", 0, (L, NBG, per, TOPK * T), 0, (NBG * TOPK * T, TOPK * T, 0, 1))]
        else:
            wires += [WI.Wire(gid, "w", rows, "o0", 0, (L, TOPK, B, T), 0, (B * TOPK * T, T, TOPK * T, 1)),
                      WI.Wire(gid, "ws", scales, "o0", 0, (L, TOPK, NBG, per, T), 0, (NBG * TOPK * T, T, TOPK * T, 0, 1))]
            wires += [WI.Wire(gid, "g", rid, f"o{k}", (l * TOPK + k) * B * T, (B, T), l * T, (0, 1)) for l in range(L)
                      for k in range(TOPK)]
    reg = WI.Registered(params, tuple(tables), (), tuple(roots), tuple(openings))
    return WI.Circuit("glm53-moe-per-expert", tuple(templates), wires=tuple(WI.coalesce(wires)), registered=reg), params


@pytest.mark.parametrize("fused", [False, True], ids=["read-then-gemm", "fused"])
def test_the_cut_is_a_circuit_the_window_interface_takes(fused: bool) -> None:
    """Two layers of two tokens: the templates' proof blocks compose, the wires and declarations check, the tables commit, and
    the free input bits are the ports left open alone where the tables are pinned, those and the openings where they aren't."""
    L, T = 2, 2
    circuit, params = _cut(L, T, fused)
    lays, gemms = circuit.layouts, FUSED if fused else GEMMS
    for fn, _n in circuit.templates:
        WI.check_proof_block(fn)
    WI._committable(circuit.registered, (), circuit.name)
    assert len(circuit.units) == (5 if fused else 7) and circuit.units[descriptor_id(ROUTER)] == L * T
    assert len(params) == L * (UP["R"] // UP["N"] + UP["R"] // G + DOWN["R"] // DOWN["N"] + DOWN["R"] // G)

    def bits(tid: str, ports) -> int:
        return 16 * sum(dict(lays[tid][0].rows)[p] for p in ports) * lays[tid][1]

    opened = sum(bits(o.reader, o.ports) // lays[o.reader][1] * o.units for o in circuit.registered.openings)
    table_reads = sum(16 * dict(lays[r.reader][0].rows)[r.port] * r.units for r in circuit.registered.roots)
    left = bits(descriptor_id(ROUTER), OPEN_PORTS["router"]) + sum(bits(descriptor_id(gemms[p]), OPEN_PORTS[p]) for p in gemms)
    pins = {p: f"{i:0128x}" for i, p in enumerate(sorted(params))}
    held = WI.input_count(circuit.description(), pins=pins, roots=pins)
    assert held["registered"] == opened + table_reads and held["ingress"] == 0
    assert held["free"] == left and held["wired"] == held["inputs"] - left - opened - table_reads
    unpinned = WI.input_count(circuit.description())
    assert unpinned["registered"] == 0 and unpinned["free"] == left + opened + table_reads


def test_a_read_is_a_wire_of_the_gemms_weight_and_scale_rows() -> None:
    """Each read's h is its GEMM's w or ws row word for word (a wire reads its writer's row whole), and the router's slot rows
    are a read's tok and w2's g."""
    sizes = {p: dict(WI.layout(fn).rows) for p, fn in GEMMS.items()}
    for p, sh in (("up", UP), ("down", DOWN)):
        K, N = sh["K"], sh["N"]
        assert dict(WI.layout(EMB.read(E, N * K // 2)).outs)["o0"] == sizes[p]["w"]
        assert dict(WI.layout(EMB.read(E, 2 * K // G)).outs)["o0"] == sizes[p]["ws"]
    outs, tok = dict(WI.layout(ROUTER).outs), dict(WI.layout(EMB.read(E, 4)).rows)["tok"]
    assert [outs[f"o{j}"] for j in range(2 * TOPK)] == [2] * 2 * TOPK == [sizes["down"]["g"]] * TOPK + [tok] * TOPK


def test_a_fused_unit_has_its_reads_ports_and_outputs_then_its_gemms() -> None:
    """A fused unit's rows are its read's (tok .. path) then its GEMM's without w, and its outputs the GEMM's y as o0 and the
    read's verdict as o1, so a table's declarations on the read carry over to it unchanged; a READ of other than the block's
    N K / 2 words, or not a read, is refused."""
    for p, sh in (("up", UP), ("down", DOWN)):
        read, gemm, fused = (WI.layout(fn) for fn in (EMB.read(E, sh["N"] * sh["K"] // 2), GEMMS[p], FUSED[p]))
        assert fused.rows == read.rows + tuple(r for r in gemm.rows if r[0] != "w")
        assert fused.outs == (("o0", dict(gemm.outs)["o0"]), ("o1", dict(read.outs)["o1"]))
    for wrong in (EMB.read(E, 2 * UP["K"] // G), GEMMS["up"]):
        with pytest.raises(ValueError, match="not a read of N K / 2 = 3072 u16 words"):
            bind(BM.MoeExpertFp8ReadGemmV1, **{**FUSED["up"].bindings, "READ": wrong}).signature()


def test_fusing_takes_the_weight_rows_out_of_the_outputs() -> None:
    """The fused cut's units output every word the read-then-GEMM cut's do but the weight rows' reads' h: the H words of each
    (token, slot, block) of either projection, most of what the cut outputs."""
    L, T = 2, 2

    def out_words(circuit: WI.Circuit) -> int:
        return sum(sum(w for _o, w in lay.outs) * n for lay, n in circuit.layouts.values())

    rows = sum(L * sh["R"] // sh["N"] * TOPK * T * sh["N"] * sh["K"] // 2 for sh in (UP, DOWN))
    apart, fused = out_words(_cut(L, T)[0]), out_words(_cut(L, T, fused=True)[0])
    assert apart - fused == rows and rows > 0.99 * apart
