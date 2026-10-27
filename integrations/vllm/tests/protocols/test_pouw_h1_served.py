"""`protocol_options.pouw_h1_served`: an H1 window that mixes K, served under compute accounting's window Program
(`h1window.program` on vllm's 32 × 32 tile, one template per K), registers its tables as its calls are recorded: each
step's A table of K ``a/<K>@<s>`` (M slot rows of each of the step's calls of K, call after call, each row its
codes then its f32 s_a) at the step's last call of K, each call's z table ``zy@<c>`` (one z/y record a proof block, in tile
order), and the weights' ``b/<K>`` (each linear's rows once, codes then s_w), as ``hm96-sha512/row/v2`` trees whose rows
and roots recompute here from the served tensors, under domains bound to the window's open root. Each root is in the
manifest row of the call that completes it, so its round's root commits it before the round's coin. A call's statement file
(`h1window.table_reads_file`) resolves each of its proof blocks' ports (`h1window.window_ports`, each K's own; ``a_k<K>`` →
``a_k<K>_0`` for a tile of one row, ``b_k<K>`` → ``b_k<K>_0`` for one column) to its own tables and the weights', and no
other call's proof block reads them; a read opened with another row is refused. A weight that changes, blocks that aren't one
per linear, an odd K, n off C and a served call with no z/y record register nothing; a second window reusing the first's
key is refused (R7-held). Every window of a period declares one Program (the period's prefix), its call tables its own
through their window-bound domains, and its ``b/<K>`` the period's registered weights, which a window serving
another code or s_w is refused for. A stand-in tile proof block of the zy proof block's parameter shape (`h1zy.ROW` rows, its `h1zy.RECORD`, a
receipt sink and ``ok``) keeps the Program small."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw import weights as Wt  # noqa: E402
from verity.core.protocols.pouw.audit import hm96_salt  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind  # noqa: E402
from verity.ml.circuits.ports import In, Out, Ports  # noqa: E402
from verity.ml.circuits.refs import Coll, coll_concat  # noqa: E402
from verity.ml.circuits.types import Array, Record, Value  # noqa: E402
from verity.ml.commitments import hm96  # noqa: E402
from verity.ml.commitments.merkle import tree_levels  # noqa: E402
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest  # noqa: E402
from verity_vllm.engine import int8_ops as OPS  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import registered as NR  # noqa: E402
from verity_vllm.protocols.pouw.h1 import requests as RQ  # noqa: E402
from verity_vllm.protocols.pouw.h1 import served as SV  # noqa: E402

TILE = PH.TILE
T, STEPS, SERVED = 64, 2, 5
MODULES = {0: "q", 1: "k", 2: "o"}
#: q and k at one K on one input, o at another K, as down_proj's K is the intermediate size
SHAPES = [(64, 16), (32, 16), (32, 32)]
BLOCKS = (0, 0, 1)
KS = (16, 32)
SALT = hashlib.sha512(b"salt").digest()
W1, W2 = hashlib.sha512(b"window 1").digest(), hashlib.sha512(b"window 2").digest()
WKEY = bytes(range(32))
V1, V16 = Value(1), Value(16)
XOR1 = PrimitiveDefinition("H1ServedTestXor1", 1, (("a", V1), ("b", V1)), V1, lambda a, b: a ^ b, register=False)
LOW1 = PrimitiveDefinition("H1ServedTestLow1", 1, (("v", V16),), V1, lambda v: v & 1, register=False)
FOLD1 = PrimitiveDefinition("H1ServedTestFold1", 1, (("key", V1), ("x", V1)), V1, lambda key, x: key ^ x,
                            register=False, ports=Ports(params={"key": In(kind="coin")}, ret=Out(sink=True)))
#: the stand-in's receipt: 16 bits, one u16 word a proof block (a receipt is bits, as the tile's u32 words are)
RECEIPT = Array(16, V1)
OUT = Record(("receipt", RECEIPT), ("ok", V1))


def _sig(S):
    n = Wt.chunks(S.K)
    return ((("a", Array(n * S.R, Z.ROW(S.K // n))), ("b", Array(n * S.C, Z.ROW(S.K // n))), ("zy", Z.RECORD(S.R, S.C)),
             ("salt", Array(4, V1)), ("position", V16), ("roots", Array(2, Array(4, V16)))), OUT)


def _first(c):
    return Coll(V1, c.refs.slice(0, 1))


def _body(B, S, a, b, zy, salt, position, roots):
    ok = B.call(XOR1, B.call(XOR1, _first(a[0]), _first(b[0])), _first(zy))
    x = B.call(XOR1, B.call(LOW1, position), B.call(LOW1, roots[0][0]))
    receipt = coll_concat(RECEIPT, [B.call(FOLD1, _first(salt), B.call(XOR1, x, Coll(V1, a[0].refs.slice(i, i + 1))))
                                    for i in range(16)])
    return coll_concat(OUT, [receipt, ok])


StandIn = CompositeDefinition("H1ServedTestTile", 0, ("K", "R", "C"), _sig, _body, register=False,
                              ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")},
                                          ret={"receipt": Out(sink=True)}))


def _j10r_sig(S):
    params, ret = _sig(S)
    return params + (("call", SD.POS_BITS), ("blocks", Array(2, SD.POS_BITS))), ret


def _j10r_body(B, S, a, b, zy, salt, position, roots, call, blocks):
    return _body(B, S, a, b, zy, salt, position, roots)


#: the stand-in under J10-R: `h1window.TILE_PARAMS_J10R`, its call and blocks typed as `h1tile.PouwH1Tile5`'s
StandInJ10R = CompositeDefinition("H1ServedTestTileJ10R", 0, ("K", "R", "C"), _j10r_sig, _j10r_body, register=False,
                                  ports=StandIn.ports)


def _unit(tile=TILE, defn=StandIn):
    return lambda k: bind(defn, K=k, R=tile[0], C=tile[1])


def _codes(rng, n, k):
    return torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8))


def _weights(seed: int = 0, shapes=SHAPES):
    """Each linear's int8 codes and f32 scales (n, 1)."""
    rng, g = np.random.default_rng(1000 + seed), torch.Generator().manual_seed(seed)
    return [(_codes(rng, n, k), torch.rand(n, 1, generator=g) + 0.5) for n, k in shapes]


def _period(ws, shapes=SHAPES, modules=MODULES, blocks=BLOCKS, defn=StandIn):
    """The period's `Weights` over `ws`, registered under the declared Program's digest with `WKEY`."""
    circuit = P.program_digest(SV.declared(shapes, modules, T, STEPS, TILE, blocks, unit=_unit(defn=defn)))
    return SV.register_weights(circuit, shapes, {w: (c.numpy(), s.numpy().reshape(-1)) for w, (c, s) in enumerate(ws)},
                               WKEY)


def _window(key: bytes, seed: int = 0, served_key: bytes | None = None, shapes=SHAPES, blocks=BLOCKS, steps=STEPS,
            served=1, changed=None, modules=MODULES, tile=TILE, rows=T, no_z=False, ws=None, weights=None, receipts=None,
            order=None, defn=StandIn):
    """A window: its first ``served`` steps serve 5 rows each, q and k on one input x, o on another, every step with the
    same weights ``ws`` (`_weights` of ``seed`` by default; step 1's q on ``changed`` when given), each call's z/y records
    `scaled_mm_zy`'s over its T slots, its linears called in ``order`` (module order by default); the later steps are
    empty. With ``weights`` (`_period`) its ``b/<K>`` are the period's. Its proof blocks are ``defn``'s (`StandInJ10R` for a
    J10-R window). Returns the run, each linear's weight and scales, and each served call's (a, s_a, z, y) by its index,
    z its exact product and y its served y."""
    rng = np.random.default_rng(seed)
    run = PH.H1Run("t", SALT, modules, shapes, rows, steps, key=bytes([seed + 1]) * 32,
                   served=SV.Served(tile, _unit(tile, defn), blocks, window=served_key or key, weights=weights,
                                    receipts=receipts))
    ws = _weights(seed, shapes) if ws is None else ws
    got = {}
    for s in range(served):
        xs = {}
        for b, (_n, k) in zip(blocks, shapes):
            xs[b] = xs[b] if b in xs else _codes(rng, SERVED, k)
        inputs = [xs[b] for b in blocks]
        for w in range(len(inputs)) if order is None else order:
            a, (wt, sw) = inputs[w], ws[w]
            wt = changed if (s, w) == (1, 0) and changed is not None else wt
            n = wt.shape[0]
            s_a = torch.rand(SERVED) + 0.5
            y, zy = OPS.scaled_mm_zy(a, s_a, wt, sw, rows) if n % PH.TILE[1] == 0 else (None, None)
            call = run.record(w, a, s_a, wt, sw, z=None if no_z else zy)
            got[call.index] = (a, s_a, a.to(torch.int32) @ wt.to(torch.int32).T, y)
    run.close()
    return run, ws, got


def _tree(reg: G.Registered, rows: list[bytes], key: bytes, first: int) -> str:
    """`reg`'s tree recomputed from each row's raw bytes, its bit row's memory, under row j's salt at nonce ``first``."""
    dom, inst = reg.domain, hm96.SHA512
    bcs = [inst.commit_string(inst.default_key, sha512_row_v2_digest(row, 8 * len(row)), hm96_salt(key, first, j))
           for j, row in enumerate(rows)]
    return tree_levels(dom, [row_leaf(dom, j, inst.tree_leaf(inst.default_key, bc), SV.V2)
                             for j, bc in enumerate(bcs)])[-1][0].hex()


def _a_rows(a, s_a, m, k):
    """A table rows from the tensors: each served row's codes then its f32 scale, then zero rows of scale 0."""
    rows = [a[i].numpy().tobytes() + np.float32(s_a[i]).tobytes() for i in range(len(a))]
    return rows + [bytes(k + 4)] * (m - len(rows))


def _records(z, y, m, n, tile=TILE):
    """z table rows from the tensors, tile by tile: each cell's z (s32 LE), row-major, then each cell's y (bf16 bits)."""
    (r, c), zf, yf = tile, np.zeros((m, n), "<i4"), np.zeros((m, n), "<i2")
    zf[:len(z)], yf[:len(y)] = z.numpy(), y.view(torch.int16).numpy()
    out = []
    for ti in range(m // r):
        for tj in range(n // c):
            cells = [(ti * r + i, tj * c + j) for i in range(r) for j in range(c)]
            out.append(b"".join(zf[p].tobytes() for p in cells) + b"".join(yf[p].tobytes() for p in cells))
    return out


def _served_call(got, c, n, k):
    """Call c's served (a, s_a, z, y), zeros for an empty call."""
    return got.get(c) or (torch.zeros(0, k, dtype=torch.int8), torch.zeros(0), torch.zeros(0, n), torch.zeros(0, n).bfloat16())


def test_each_call_registers_its_tables_as_it_is_recorded():
    """Each step's A table of K is the Program's `h1window.table` value ``a/<K>@<s>``: M rows of K/2 + 2 words for each of
    the step's calls of K, call after call (`credit.step_tables`), the j-th call's rows from row M·j
    (`h1window.step_of`), registered at the step's last call of K and in that call's manifest row only. Each call's z
    table is one record of 3 R C words a proof block. Both are under the registered-value domain of the Program's digest bound
    to the window's open root; their roots recompute from the served tensors (an empty call's from zeros), so the round's
    root is over them."""
    run, ws, got = _window(W1)
    served, pre = run.served, SV.PREFIX
    prog, digest = served.program, P.program_digest(served.program)
    assert served.program_digest == digest and served.root == W1.hex() and served.m == T and served.steps == STEPS
    assert served.linears == [("q", 16, 64), ("k", 16, 32), ("o", 32, 32)]
    assert served.calls == [(m, s) for s in range(STEPS) for m in ("q", "k", "o")]
    assert list(served.registered) == [pre + v for v in ("zy@0", "a/16@0", "zy@1", "b/16", "pub@0", "pub@1", "a/32@0",
                                                         "zy@2", "b/32", "pub@2", "zy@3", "a/16@1", "zy@4", "pub@3",
                                                         "pub@4", "a/32@1", "zy@5", "pub@5")]
    tables = CR.step_tables({"calls": served.calls, "linears": served.linears, "m": served.m}, STEPS)
    assert tables == [(0, 16, [0, 1]), (0, 32, [2]), (1, 16, [3, 4]), (1, 32, [5])]
    for s, k, cs in tables:
        ra = served.registered[HW.table(prog, "a", cs[0], STEPS)]
        assert ra.value == pre + f"a/{k}@{s}" and (ra.leaves, ra.words) == (T * len(cs), k // 2 + 2)
        assert ra.domain.domain_id.hex() == G.domain(digest, ra.value, ra.words, ra.leaves, window=W1.hex()).domain_id.hex()
        assert ra.domain.domain_id != G.domain(digest, ra.value, ra.words, ra.leaves).domain_id
        rows = [row for c in cs for row in _a_rows(*_served_call(got, c, 0, k)[:2], T, k)]
        assert ra.root.hex() == _tree(ra, rows, run._key, SV.nonce(cs[0], "a"))
        assert [c.index for c in run.calls if ra.value in run.tables[c.index]["registered"]] == [cs[-1]]
        assert run.tables[cs[-1]]["registered"][ra.value] == ra.root.hex()
        assert [HW.step_of(prog, c, STEPS) for c in cs] == [(s, len(cs), T * j) for j in range(len(cs))]
    for c in run.calls:
        n, k = run.shapes_[c.layer]
        a, s_a, z, y = _served_call(got, c.index, n, k)
        rz = served.registered[HW.table(prog, "zy", c.index)]
        assert (rz.leaves, rz.words) == ((T // 32) * (n // 32), 3 * 32 * 32)
        assert rz.root.hex() == _tree(rz, _records(z, y, T, n), run._key, SV.nonce(c.index, "zy"))
        assert {rz.value: rz.root.hex()}.items() <= run.tables[c.index]["registered"].items()
    for r, log in enumerate(run.round_log):
        assert RQ.step_digest(run.salt, r, run._rows([c for c in run.calls if c.step == r])).hex() == log["calls"]


@pytest.mark.parametrize("steps", [2, 3])
def test_b_is_each_linears_weight_once_a_window(steps):
    """b/<K> is Σn rows of K's linears, codes then s_w, whatever the window's steps, registered at the call that completes
    K's weights in step 0 and in that call's manifest row only; every step's proof blocks read the same rows."""
    run, ws, _got = _window(W1, steps=steps, served=2)
    served, pre = run.served, SV.PREFIX
    b16, b32 = served.registered[pre + "b/16"], served.registered[pre + "b/32"]
    assert (b16.leaves, b32.leaves) == (64 + 32, 32)
    rows = [w[j].numpy().tobytes() + np.float32(s[j, 0]).tobytes() for w, s in ws[:2] for j in range(len(w))]
    assert b16.root.hex() == _tree(b16, rows, run._key, SV.B_NONCE)
    assert [c.index for c in run.calls if pre + "b/16" in run.tables[c.index]["registered"]] == [1]
    assert [c.index for c in run.calls if pre + "b/32" in run.tables[c.index]["registered"]] == [2]
    tp = P.evaluate(served.program, served.query())
    per = tp.population // steps
    first = [next(v for key, v in HW.table_reads(served.program, tp, u, steps).items() if key.startswith("b_k") and
                  key.endswith("[0]")) for u in range(tp.population)]
    assert all(first[u] == first[u % per] for u in range(tp.population))


def test_each_expert_gemm_registers_its_tables_as_a_dense_linears_call_does():
    """A MoE layer's expert GEMMs (`engine.int8_experts`), wired as the option wires every recorded layer
    (`pouw_h1_hooks.wire`: z/y records over T, observe, apply), hand H1 each call's A rows, s_a and `scaled_mm_zy`'s z/y
    records as a dense linear's call does: each (step, expert, projection) puts its rows in its step's A table of its K
    ``a/<K>@<s>`` (the step's experts' calls of K, call after call) and registers ``zy@<c>``, whose roots recompute from
    each expert's own served rows (an idle expert's zero-row call from zeros), and its manifest row carries its served
    rows m."""
    from verity_vllm.engine import hooks
    from verity_vllm.engine import int8_experts as X
    from verity_vllm.protocols.pouw.h1 import hooks as PHK
    g, rng = torch.Generator().manual_seed(3), np.random.default_rng(3)
    w13, w2 = _codes(rng, 2 * 32, 32).reshape(2, 32, 32), _codes(rng, 2 * 32, 16).reshape(2, 32, 16)
    s13, s2 = (torch.rand(2, 32, 1, generator=g) * 0.01 + 1e-4 for _ in range(2))
    sites = X.sites("moe", w13, s13, w2, s2)
    shapes, modules = [tuple(s.weight.shape) for s in sites], {w: s.name for w, s in enumerate(sites)}
    run = PH.H1Run("t", SALT, modules, shapes, T, STEPS, key=bytes([7]) * 32,
                   served=SV.Served(TILE, _unit(TILE), window=W1))
    steps = [torch.tensor([[0, 1], [1, 0], [0, 1]]), torch.tensor([[1, X.UNROUTED], [1, X.UNROUTED]])]
    got = {}
    with hooks.Hooks("t") as h:
        PHK.wire(run, h, sites, None, T)
        for s, ids in enumerate(steps):
            x = (torch.randn(len(ids), 32, generator=g) * 3).bfloat16()
            X.forward(sites, x, torch.rand(ids.shape, generator=g), ids)
            for e, pos in enumerate(X.routes(ids, 2)):
                a = x.index_select(0, pos // ids.shape[1])
                for w, wt, sw in ((2 * e, w13[e], s13[e]), (2 * e + 1, w2[e], s2[e])):
                    n, k = wt.shape
                    xq, s_a, y = (torch.zeros(0, k, dtype=torch.int8), torch.zeros(0), torch.zeros(0, n).bfloat16()) \
                        if not len(a) else OPS.linear_zy(a, wt, sw.reshape(-1, 1), T)[:3]
                    got[s * len(sites) + w] = (xq, s_a, xq.to(torch.int32) @ wt.to(torch.int32).T, y)
                    a = X.silu_and_mul(y)
    run.close()
    served, prog = run.served, run.served.program
    assert not served.refused and [c.m for c in run.calls] == [3, 3, 3, 3, 0, 0, 2, 2]
    assert [c.summary()["served"] for c in run.calls] == [3, 3, 3, 3, 0, 0, 2, 2]
    assert [c.module for c in run.calls[:4]] == ["moe.0.gate_up", "moe.0.down", "moe.1.gate_up", "moe.1.down"]
    for _s, k, cs in CR.step_tables({"calls": served.calls, "linears": served.linears, "m": served.m}, STEPS):
        ra = served.registered[HW.table(prog, "a", cs[0], STEPS)]
        rows = [row for c in cs for row in _a_rows(*got[c][:2], T, k)]
        assert ra.root.hex() == _tree(ra, rows, run._key, SV.nonce(cs[0], "a"))
        assert run.tables[cs[-1]]["registered"][ra.value] == ra.root.hex()
    for c in run.calls:
        n, _k = run.shapes_[c.layer]
        _a, _s_a, z, y = got[c.index]
        rz = served.registered[HW.table(prog, "zy", c.index)]
        assert rz.root.hex() == _tree(rz, _records(z, y, T, n), run._key, SV.nonce(c.index, "zy"))
        assert {rz.value: rz.root.hex()}.items() <= run.tables[c.index]["registered"].items()


def test_a_calls_statement_reads_its_units_rows_at_its_tables_and_refuses_another_row():
    """Each call's file names its K's ports, ``a_k<K>_0`` .. ``a_k<K>_31``, ``b_k<K>_0`` .. ``b_k<K>_31`` and ``zy``, each in
    row/v2 with its tree's value, root, rows, width and domain (the window's): the A ports at its step's A table of K,
    the record at the call's z table, B at the weights'. Only the call's proof blocks read them, at the rows
    `h1window.table_reads` gives (its A rows from its call's offset in the step table), and each read opens its tree at
    its row, while the same path with another row is refused."""
    run, _ws, _got = _window(W1)
    served = run.served
    prog, tp = served.program, P.evaluate(served.program, served.query())
    reg = SV.register(run)
    for c, (u0, n) in enumerate(HW.call_units(prog)):
        k, f = run.shapes_[run.calls[c].layer][1], reg.files[c]
        assert set(f["ports"]) == {f"{t}_k{k}_{i}" for t in "ab" for i in range(32)} | {"zy", HW.PUB_PORT, HW.OK_PORT}
        assert {(p["row_schema"], p["word_bits"]) for p in f["ports"].values()} == {(SV.V2, 16)}
        assert {f["ports"][f"a_k{k}_0"]["value"], f["ports"]["zy"]["value"], f["ports"][f"b_k{k}_0"]["value"]} == {
            HW.table(prog, "a", c, STEPS), HW.table(prog, "zy", c), SV.PREFIX + f"b/{k}"}
        at = HW.step_of(prog, c, STEPS)[2]
        assert f["format"] == G.READS_V1 and all(p[G.READS_FROM] == u0 and len(p["reads"]) == n for p in f["ports"].values())
        assert f["ports"][f"a_k{k}_0"]["reads"][0] == at and at == T * served.step_calls(c).index(c)
        assert f["ports"][HW.OK_PORT]["value"] == SV.PREFIX + "ok" and SV.PREFIX + "ok" not in served.registered
        for port, p in f["ports"].items():
            if port == HW.OK_PORT:
                continue
            r = served.registered[p["value"]]
            assert (p["root"], p["words"], p["leaves"], p["domain"]) == (
                r.root.hex(), r.words, r.leaves, r.domain.domain_id.hex())
            if port == HW.PUB_PORT:
                continue
            key = port if port == "zy" else f"{port.rsplit('_', 1)[0]}[{port.rsplit('_', 1)[1]}]"
            for u, row in enumerate(CC.port_reads(p)):
                if not u0 <= u < u0 + n:
                    assert row is None
                    continue
                assert HW.table_reads(prog, tp, u, STEPS)[key] == (p["value"], row)
                assert G.verify_read(r.domain, r.root, row, r.bcs[row], r.path(row), SV.V2)
                assert not G.verify_read(r.domain, r.root, row, r.bcs[(row + 1) % r.leaves], r.path(row), SV.V2)


@pytest.mark.parametrize("tile, lone", [((1, 32), "a"), ((32, 1), "b")], ids=["R=1", "C=1"])
def test_a_tile_of_one_row_or_column_maps_its_lone_read_to_port_0(tile, lone):
    """With R = 1 `registered.reads` and `h1window.table_reads` both key a proof block's lone A row by its parameter's bare name
    ``a_k<K>`` (`h1window.row_key`), and the window's ports map it to ``a_k<K>_0`` with no ``[0]`` alias; likewise B's
    with C = 1. Each call's file names its K's port, read by each of its proof blocks."""
    run, _ws, _got = _window(W1, shapes=[(32, 16), (32, 16), (32, 32)], steps=1, tile=tile, rows=32)
    served = run.served
    ports = served.ports()
    assert all(ports[f"{lone}_k{k}"] == f"{lone}_k{k}_0" and f"{lone}_k{k}[0]" not in ports for k in KS)
    assert ports == HW.window_ports(served.program)
    reg = SV.register(run)
    other = "b" if lone == "a" else "a"
    for c, (u0, n) in enumerate(HW.call_units(served.program)):
        k = run.shapes_[c][1]
        assert set(reg.files[c]["ports"]) == {f"{lone}_k{k}_0", "zy", HW.PUB_PORT, HW.OK_PORT} | {f"{other}_k{k}_{i}" for i in range(32)}
        reads = CC.port_reads(reg.files[c]["ports"][f"{lone}_k{k}_0"])
        assert [u for u, r in enumerate(reads) if r is not None] == list(range(u0, u0 + n))


def test_a_k32_and_k322_window_names_every_read_its_own_port():
    """At K = 32 and 322 a port without `window_ports`' separator would make ``a_k32[20]`` and ``a_k322[0]`` one port
    ``a_k3220``; the window's 4 × 32 reads are 4 × 32 ports, ``a_k32_20`` and ``a_k322_0`` among them."""
    prog = HW.program([("q", 32, 32), ("o", 322, 32)], [("q", 0), ("o", 0)], T, TILE, SV.PREFIX, _unit())
    ports = SV.Served(program=prog).ports()
    assert len(ports) == len(set(ports.values())) == 4 * 32
    assert (ports["a_k32[20]"], ports["a_k322[0]"]) == ("a_k32_20", "a_k322_0")


def test_calls_root_binds_the_tables():
    """A call's registered roots are in the manifest, and the served digest beside it: another root moves calls_root."""
    run, _ws, _got = _window(W1)
    served, got = run.served, run.calls_root()
    run.served = None
    assert got == hashlib.sha512(run.calls_root() + served.digest()).digest()
    run.served = served
    value = next(iter(run.tables[0]["registered"]))
    run.tables[0]["registered"][value] = "00" * 64
    assert run.calls_root() != got


@pytest.mark.parametrize("why, kw", [
    ("weight changed within the window", {"served": 2, "changed": torch.zeros(64, 16, dtype=torch.int8)}),
    ("one input block for each", {"blocks": (0, 1)}),
    ("whole u16 words", {"shapes": [(64, 15), (32, 15), (32, 32)]}),
    ("whole blocks", {"shapes": [(48, 16), (32, 16), (32, 32)]}),
    ("no z/y record", {"no_z": True}),
], ids=["weight-changed", "blocks-off", "odd-k", "n-off-c", "no-record"])
def test_a_window_off_the_programs_layout_is_refused(why, kw):
    """A weight changed between steps, blocks that aren't one per linear, an odd K, n off the tile's C and a served call
    with no z/y record: no table from then on, and `trees` says why."""
    run, _ws, _got = _window(W1, **kw)
    assert any(why in r for r in run.served.refused)
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


def test_a_second_window_reusing_the_first_windows_key_is_refused():
    """R7-held: every window of a period registers under the period's one Program and its names, so a window's values are
    its own through their domains, which bind its open root. A later window keyed as the first commits under the first's
    domains with other roots, and `register` refuses it; keyed as its own window over the period's weights, its call
    tables' domains are its own and its ``b/<K>`` the period's, entry for entry, and it registers. A window that
    registers weights of its own beside the period's commits ``b/<K>`` under the domain that binds no window with
    another root, and is refused."""
    ws = _weights()
    period = _period(ws)
    first, _ws, _got = _window(W1, ws=ws, weights=period)
    log = [{"registered": SV.register(first).registered}]
    again, _ws, _got = _window(W2, seed=1, served_key=W1, ws=ws, weights=period)
    with pytest.raises(ValueError, match="R7-held"):
        SV.register(again, log=log)
    own, _ws, _got = _window(W2, seed=1, ws=ws, weights=period)
    SV.register(own, log=log)
    mine, theirs = SV.trees(own), log[0]["registered"]
    assert {t.value for t in mine} == {e["value"] for e in theirs}
    held = [e for e in theirs if CR.is_held(e["value"])]
    assert held == period.entries() == [t.entry() for t in mine if CR.is_held(t.value)]
    assert not {t.domain for t in mine if not CR.is_held(t.value)} & {e["domain"] for e in theirs}
    other, _ws, _got = _window(W2, seed=1)
    with pytest.raises(ValueError, match="R7-held"):
        SV.register(other, log=log)


def test_the_window_key_is_its_window_open_root():
    commits = {"weights_root": "11" * 64, "requests_root": "22" * 64, "session": {"session": "s-1"}}
    key = SV.Served().key(commits)
    assert key == PS.window_open_root(bytes.fromhex("11" * 64), bytes.fromhex("22" * 64), "s-1")
    with pytest.raises(ValueError, match="the window has no key"):
        SV.Served().key({"weights_root": "11" * 64})


def test_a_window_whose_rows_arent_registered_values_is_refused():
    run = PH.H1Run("t", SALT, MODULES, SHAPES, T, STEPS)
    run.close()
    with pytest.raises(ValueError, match="aren't registered values"):
        SV.trees(run)


def test_register_writes_each_calls_file(tmp_path):
    """`register` gives the record's entries, one a tree, and writes each asked call's file as `file_name`, canonical
    JSON of `reads_file`."""
    run, _ws, _got = _window(W1)
    reg = SV.register(run, calls=[0, 2], directory=tmp_path)
    assert reg.window_root == W1.hex() and [e["value"] for e in reg.registered] == list(run.served.registered)
    assert sorted(p.name for p in tmp_path.iterdir()) == [SV.file_name(0), SV.file_name(2)] == [
        "pouw-h1-registered-c0.json", "pouw-h1-registered-c2.json"]
    from verity.ml.circuits.codec import canonical_json
    assert (tmp_path / SV.file_name(2)).read_bytes() == canonical_json(SV.reads_file(run, 2)) == canonical_json(reg.files[2])


def test_two_windows_of_one_layout_declare_one_program():
    """`declared` from the window's public shapes, modules, rows and steps alone, under the period's `PREFIX`, is the
    Program every window of the period serves: two windows, two open roots, one `program_digest`, the period's
    ``--circuit``."""
    run, _ws, _got = _window(W1)
    other, _ws, _got = _window(W2, seed=1)
    decl = SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, unit=_unit())
    assert run.served.root != other.served.root
    assert P.program_digest(decl) == run.served.program_digest == other.served.program_digest
    assert decl.fn.bindings["PREFIX"] == SV.PREFIX
    assert decl.fn.bindings["CALLS"] == tuple(run.served.calls) and decl.fn.bindings["LINEARS"] == tuple(run.served.linears)
    assert hashlib.sha512(SV.descriptor(decl)).hexdigest() == run.served.program_digest


def test_a_periods_program_and_digest_given_are_not_declared_again(monkeypatch):
    """A window handed its period's Program and digest declares neither again (at FILL the digest alone takes minutes),
    and one handed only the Program takes its digest."""
    decl = SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, unit=_unit())
    digest = P.program_digest(decl)
    taken = SV.Served(program=decl).declare(SHAPES, MODULES, T, STEPS)
    monkeypatch.setattr(SV, "declared", lambda *a, **k: pytest.fail("declared again"))
    monkeypatch.setattr(SV.P, "program_digest", lambda *a, **k: pytest.fail("digested again"))
    served = SV.Served(program=decl, program_digest=digest)
    assert taken == digest == served.declare(SHAPES, MODULES, T, STEPS) and served.calls == list(decl.fn.bindings["CALLS"])


def test_a_window_without_a_unit_takes_each_ks_tile_unit_on_its_cards_arch(monkeypatch):
    """Without ``unit``, `declared` takes `h1window.tile_unit` on ``arch``, the serving card's H1 record, whose domain
    refuses a K that card's kernels don't serve: K 16 and 32 on sm_90 (K >= 384 and 512 | K), before any tile is built.
    It hands each K and the tile to `tile_unit` with that arch, as does a `Served` of that arch, and one with no arch keeps
    `tile_unit`'s own (sm_120's, which states no domain). The real proof blocks' Program is the same on sm_90 and sm_120 (the
    record's step is one function); a stand-in of their parameter shape keeps it small here."""
    with pytest.raises(ValueError, match="outside H1's domain on sm_90"):
        SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, arch="sm_90")
    asked, stand_in = [], _unit()

    def tile_unit(k, tile, definition=None, arch="sm_120"):
        asked.append((k, tuple(tile), definition, arch))
        return stand_in(k)

    monkeypatch.setattr(SV.HW, "tile_unit", tile_unit)
    decl = SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, arch="sm_90")
    assert sorted(asked) == [(16, TILE, None, "sm_90"), (32, TILE, None, "sm_90")]
    assert P.program_digest(decl) == P.program_digest(SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, unit=stand_in))
    asked.clear()
    SV.Served(TILE, None, BLOCKS, arch="sm_90").declare(SHAPES, MODULES, T, STEPS)
    assert sorted(asked) == [(16, TILE, None, "sm_90"), (32, TILE, None, "sm_90")]
    asked.clear()
    SV.Served(TILE, None, BLOCKS).declare(SHAPES, MODULES, T, STEPS)
    assert sorted(asked) == [(16, TILE, None, "sm_120"), (32, TILE, None, "sm_120")]


@pytest.mark.parametrize("capability, arch", [((9, 0), "sm_90"), ((12, 0), "sm_120"), ((8, 9), None)])
def test_card_arch_is_the_cards_h1_record(monkeypatch, capability, arch):
    """`card_arch` names the current card by its H1 record (`h1device.DEVICES`): an H200's compute capability 9.0 is
    sm_90's, the RTX PRO 6000's 12.0 sm_120's, and a card with no record (8.9) is refused."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    if arch is None:
        with pytest.raises(ValueError, match="no device record for 'sm_89'"):
            SV.card_arch()
    else:
        assert SV.card_arch() == arch


def test_a_replayed_windows_values_are_refused_through_their_window_bound_domains():
    """Two windows of a period share every name, so a window's call tables are its own only through their domains, which
    bind its open root: window 1's, presented as window 2's, are not under the domains window 2's root gives them
    (`registered.domain`, as credit's `domain_of` and `h1window.table_reads_file` compute them), and no row of them opens
    under window 2's domain; window 2's statement file names window 2's domains whatever roots it is handed. The
    weights' ``b/<K>`` bind no window and are the period's in both."""
    ws = _weights()
    period = _period(ws)
    one, _ws, _got = _window(W1, ws=ws, weights=period)
    two, _ws, _got = _window(W2, seed=1, ws=ws, weights=period)
    digest = one.served.program_digest
    for v, r in one.served.registered.items():
        theirs = two.served.registered[v]
        under_two = G.domain(digest, v, r.words, r.leaves, window=None if CR.is_held(v) else W2.hex())
        assert under_two.domain_id == theirs.domain.domain_id
        if CR.is_held(v):
            assert under_two.domain_id == r.domain.domain_id and theirs.root == r.root
            continue
        assert under_two.domain_id != r.domain.domain_id
        assert G.verify_read(r.domain, r.root, 0, r.bcs[0], r.path(0), SV.V2)
        assert not G.verify_read(under_two, r.root, 0, r.bcs[0], r.path(0), SV.V2)
    tp = P.evaluate(two.served.program, two.served.query())
    f = HW.table_reads_file(two.served.program, tp, 0, one.served.roots(), two.served.ports(), prog=digest, window=W2.hex(),
                            steps=STEPS)
    assert {p["domain"] for p in f["ports"].values()} == {
        two.served.registered[p["value"]].domain.domain_id.hex() for p in f["ports"].values()}
    assert not {p["domain"] for p in f["ports"].values() if not CR.is_held(p["value"])} & {
        r.domain.domain_id.hex() for r in one.served.registered.values()}


def test_the_periods_weights_are_registered_once_and_every_window_registers_them():
    """Each K's ``b/<K>`` is K's linears' rows in order, each its K codes then its f32 s_w, registered once under
    the period's declared Program with no window and salts from the period's key; `Weights.root` is
    `credit.weights_root` of their roots, and every window of the period registers those same entries, its open root
    over that root."""
    ws = _weights()
    period = _period(ws)
    assert sorted(period.registered) == list(KS)
    for k, r in period.registered.items():
        assert (r.value, r.words) == (SV.PREFIX + f"b/{k}", Wt.leaf_words(k))
        assert r.domain.domain_id == G.domain(period.circuit, r.value, r.words, r.leaves).domain_id
    rows = [w[j].numpy().tobytes() + np.float32(s[j, 0]).tobytes() for w, s in ws[:2] for j in range(len(w))]
    assert period.registered[16].root.hex() == _tree(period.registered[16], rows, WKEY, SV.B_NONCE)
    assert period.root() == CR.weights_root({k: r.root for k, r in period.registered.items()})
    for i, key in enumerate((W1, W2)):
        run, _ws, _got = _window(key, seed=i, ws=ws, weights=period)
        assert not run.served.refused
        assert [e for e in run.served.entries() if CR.is_held(e["value"])] == period.entries()
        assert run.served.program_digest == period.circuit


def test_the_periods_weights_under_the_registrations_key_are_the_auditors_registration():
    """The auditor registers a model's b/<K> with core's `weights.weights_roots` (``model-register``)
    under its own salt key, and the served path registers the served rows under that key (``model-key``,
    `register_weights`): the same rows, salts and domains, so each K's root, entry and the weights' root are the
    registration's."""
    ws = _weights()
    period = _period(ws)
    linears = [(n, k) for n, k in SHAPES]
    rows = [[Wt.row_bytes(w[j].numpy().tobytes(), np.float32(s[j, 0]).tobytes()) for j in range(len(w))] for w, s in ws]
    roots = Wt.weights_roots(period.circuit, SV.PREFIX, linears, lambda w: rows[w], WKEY)
    assert SV.B_NONCE == Wt.B_NONCE == RQ.REQUESTS_NONCE + (1 << 42) + (1 << 40)
    assert {k: r.root for k, r in period.registered.items()} == roots
    assert period.root() == Wt.weights_root(roots)
    leaves = {k: sum(n for n, kw in linears if kw == k) for k in roots}
    assert period.entries() == [Wt.entry(period.circuit, SV.PREFIX, k, leaves[k], roots[k]) for k in sorted(roots)]


@pytest.mark.parametrize("edit", ["s_w", "code"])
def test_a_window_serving_other_weights_than_the_periods_is_refused(edit):
    """A window serves the weights its period registered before its salt. One row's s_w, or one code, other than
    the period's refuses the window at the call that completes its K (`Served.commit`, `Weights.matches`: vllm's own
    check), with no table from then on, and `trees` says why."""
    ws = _weights()
    period = _period(ws)
    codes, scales = ws[1][0].clone(), ws[1][1].clone()
    if edit == "s_w":
        scales[3, 0] += 1
    else:
        codes[3, 5] = codes[3, 5] ^ 1
    run, _ws, _got = _window(W1, ws=[ws[0], (codes, scales), ws[2]], weights=period)
    assert any("codes or s_w are not their rows of the period's registered" in r for r in run.served.refused)
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


def test_round_r_lists_step_rs_a_tables_whole_under_its_root():
    """Each round's root lists step r's A tables ``a/<K>@<r>``, one a K, whole, as aligned subtrees (`credit.cover`) whose
    tree is the table's registered root, and no other round lists them (`credit.round_root`)."""
    run, _ws, _got = _window(W1)
    served = run.served
    for r, log in enumerate(run.round_log):
        segs = served.segments(run, r)
        assert list(segs) == [f"{SV.PREFIX}a/{k}@{r}" for k in KS]
        for v, got in segs.items():
            reg = served.registered[v]
            assert CR.tree_root(reg.domain.domain_id, reg.leaves, {(lv, ix): h for lv, ix, h in got}) == reg.root
        assert log["a"] == {v: [[lv, ix, h.hex()] for lv, ix, h in got] for v, got in segs.items()}
        assert CR.round_root(r, bytes.fromhex(log["inner"]), segs).hex() == log["root"]


class _Receipts(dict):
    """A stand-in receipt producer: each proof block of the call one receipt of the stand-in's width (one u16 word), drawn from
    the call's served codes and weight, kept by call (with the codes it was given, and the order it was asked in) so a
    test recomputes each table from them."""

    def __init__(self, dtype=np.uint16, width=1):
        super().__init__()
        self.dtype, self.width, self.order, self.codes = dtype, width, [], {}

    def __call__(self, call, codes, s_a, w, s_w):
        seed = hashlib.sha512(b"%d" % call.index + np.asarray(codes).tobytes() + np.asarray(w).tobytes()).digest()
        got = np.random.default_rng(int.from_bytes(seed[:8], "little")).integers(
            0, np.iinfo(self.dtype).max, ((T // TILE[0]) * (call.n // TILE[1]), self.width), dtype=self.dtype)
        self[call.index], self.codes[call.index] = got, np.array(codes)
        self.order.append(call.index)
        return got


def test_receipt_words_are_each_u32_word_little_endian_low_half_first():
    got = SV.receipt_words(np.array([[0x11223344, 0xAABBCCDD], [1, 2]], np.uint32))
    assert got.dtype == np.uint16 and got.tolist() == [[0x3344, 0x1122, 0xCCDD, 0xAABB], [1, 0, 2, 0]]
    assert SV.receipt_words(np.array([[7, 8]], np.uint16)).tolist() == [[7, 8]]
    with pytest.raises(ValueError, match="u32 or u16"):
        SV.receipt_words(np.zeros(3, np.uint32))


def test_each_calls_receipts_are_its_receipt_table_once_its_roots_are_registered():
    """With a receipt producer, each call's ``receipt@<c>`` (BRIDGE2) is registered at its step's last call of its K, after
    the roots its proof blocks' seeds take (its step's ``a/<K>@<s>`` and ``b/<K>``), the producer asked for each of the step's
    calls of K in call order with that call's own served codes: one ``row/v2`` row a proof block of the call in its tile order,
    the proof block's receipt (the stand-in's 16 bits, one u16 word), its root recomputed here from the call's receipts under row
    j's salt at `RECEIPT_NONCE` + c, its domain the registered-value domain of the Program's digest bound to the window's
    open root (as ``zy@<c>``'s), and its root in the manifest row of the call that registered it. Round r lists step r's
    calls' receipt tables whole under its root, which core's `credit._a_link` ties to the call's round, `tables` lists
    every call's, and each call's ``--registered`` file names it at its output row o0 (`h1window.RECEIPT_PORT`), proof block u
    of its t-th tile at row t."""
    made = _Receipts()
    run, _ws, got = _window(W1, receipts=made)
    served, pre = run.served, SV.PREFIX
    prog, digest = served.program, P.program_digest(served.program)
    assert not served.refused and sorted(made) == list(range(len(run.calls)))
    names = list(served.registered)
    assert made.order == sorted(range(len(run.calls)), key=lambda c: (served.step_calls(c)[-1], c))
    for c in run.calls:
        rc, last = served.registered[pre + f"receipt@{c.index}"], served.step_calls(c.index)[-1]
        assert names.index(rc.value) > max(names.index(served.a_table(c.index)), names.index(pre + f"b/{c.k}"),
                                           names.index(pre + f"zy@{last}"))
        assert np.array_equal(made.codes[c.index], _served_call(got, c.index, c.n, c.k)[0].numpy())
        assert (rc.leaves, rc.words) == (HW.call_units(prog)[c.index][1], 1) == served.receipt_shape(c.index)
        assert rc.root.hex() == _tree(rc, [w.tobytes() for w in made[c.index].astype("<u2")], run._key,
                                      SV.RECEIPT_NONCE + c.index)
        assert rc.domain.domain_id == G.domain(digest, rc.value, rc.words, rc.leaves, window=W1.hex()).domain_id
        assert rc.domain.domain_id != G.domain(digest, rc.value, rc.words, rc.leaves).domain_id
        assert served.root == W1.hex() and rc.window == served.registered[pre + f"zy@{c.index}"].window == W1.hex()
        assert run.tables[last]["registered"][rc.value] == rc.root.hex()
        assert all(rc.value not in run.tables[d]["registered"] for d in range(len(run.calls)) if d != last)
    assert served.tables()[-len(run.calls):] == [pre + f"receipt@{c}" for c in range(len(run.calls))]
    for r, log in enumerate(run.round_log):
        segs = served.segments(run, r)
        assert list(segs) == [f"{pre}a/{k}@{r}" for k in KS] + [pre + f"receipt@{c.index}" for c in run.calls if c.step == r]
        for v, got in segs.items():
            reg = served.registered[v]
            assert CR.tree_root(reg.domain.domain_id, reg.leaves, {(lv, ix): h for lv, ix, h in got}) == reg.root
        assert CR.round_root(r, bytes.fromhex(log["inner"]), segs).hex() == log["root"]
    assert CR._a_link(run.round_log, served.entries(), len(run.calls)) is None
    reg = SV.register(run)
    for c, (u0, n) in enumerate(HW.call_units(prog)):
        p, rc = reg.files[c]["ports"][HW.RECEIPT_PORT], served.registered[pre + f"receipt@{c}"]
        assert (p["value"], p["root"], p["leaves"], p["words"], p["domain"], p["row_schema"]) == (
            rc.value, rc.root.hex(), rc.leaves, rc.words, rc.domain.domain_id.hex(), SV.V2)
        assert [(u, row) for u, row in enumerate(CC.port_reads(p)) if row is not None] == [(u0 + t, t) for t in range(n)]
        assert all(G.verify_read(rc.domain, rc.root, t, rc.bcs[t], rc.path(t), SV.V2) for t in range(n))


def test_each_calls_public_table_and_the_ok_rows_are_the_ones_credit_builds_and_a_row_off_opens_none_of_them():
    """Each call's ``pub@<c>`` is registered at its step's last call of K, once root(``a/<K>@<s>``) and root(``b/<K>``)
    exist, and is the public table core's `credit.registered_file` builds from the descriptor, the window's entries, its
    open root and its Program salt: the same value, root, rows, words and domain, row t `core_check.public_words` of the
    salt, u0 + t and the call's two roots under `core_check.public_row_salt` of the open root; and the call's
    ``--registered`` file is credit's, the verifier-built ok rows (o1) included, every pub row opening credit's root. A table with one row off, under the
    same salts, has another root, which `credit._verified` refuses a release for, and its row opens none of credit's.
    At close the window commits its ok rows (`Served.ok`, o1) at credit's root, in no entry: a window whose entries
    name them is refused. Its linears' K are 32 and 64: credit's layout refuses a tile at a K its proof doesn't cover
    (``tile-k``, K a multiple of 32)."""
    run, _ws, _got = _window(W1, receipts=_Receipts(), shapes=[(64, 32), (32, 32), (32, 64)])
    served, pre = run.served, SV.PREFIX
    assert not served.refused and served.tables()[-2 * len(run.calls):-len(run.calls)] == [
        pre + f"pub@{c}" for c in range(len(run.calls))]
    raw = SV.descriptor(served.program)
    (lay, why), program, shape = CR.layout_of(raw), hashlib.sha512(raw).hexdigest(), CR.shape_of(raw)
    assert why is None and program == served.program_digest
    roots = {}
    for e in served.entries():
        roots.setdefault(e["value"], []).append(e["root"])
    files = SV.register(run).files
    for c, (u0, n) in enumerate(HW.call_units(served.program)):
        k, last = run.calls[c].k, served.step_calls(c)[-1]
        r = served.registered[pre + f"pub@{c}"]
        assert run.tables[last]["registered"][r.value] == r.root.hex()
        mine, why = CR.registered_file(program, lay, shape, c, roots, served.root, served.steps, run._salt.hex())
        assert why is None, why
        pub = mine["ports"][CR.PUB_PORT]
        assert files[c] == mine and HW.OK_PORT in mine["ports"]
        assert (pub["value"], pub["root"], pub["leaves"], pub["words"], pub["domain"]) == (
            r.value, r.root.hex(), n, r.words, r.domain.domain_id.hex())
        ab = (served.registered[served.a_table(c)].root, served.registered[pre + f"b/{k}"].root)
        assert pub["known"]["rows"] == [list(row) for row in r.rows] == [
            CC.public_words(run._salt, u0 + t, ab) for t in range(n)]
        assert pub["known"]["salts"] == [s.hex() for s in r.salts] == [
            CC.public_row_salt(served.root, u0 + t).hex() for t in range(n)]
        assert all(G.verify_read(r.domain, bytes.fromhex(pub["root"]), t, r.bcs[t], r.path(t), SV.V2) for t in range(n))
        t = c % n
        words = np.array(pub["known"]["rows"], np.uint16)
        words[t, c % words.shape[1]] ^= 1
        off = NR.commit(program, r.value, words, run._key, 0, True, b"".join(r.salts), window=served.root)
        assert off.root.hex() != pub["root"] and off.domain.domain_id == r.domain.domain_id
        assert not G.verify_read(r.domain, bytes.fromhex(pub["root"]), t, off.bcs[t], off.path(t), SV.V2)
        ok = mine["ports"][CR.OK_PORT]
        assert (ok["value"], ok["root"], ok["leaves"], ok["domain"]) == (
            served.ok.value, served.ok.root.hex(), served.ok.leaves, served.ok.domain.domain_id.hex())
        assert all(G.verify_read(served.ok.domain, bytes.fromhex(ok["root"]), u, served.ok.bcs[u], served.ok.path(u),
                                 SV.V2) for u in range(u0, u0 + n))
    assert served.ok.value == pre + "ok" and served.ok.value not in [e["value"] for e in served.entries()]
    assert [list(row) for row in served.ok.rows] == [list(CC.OK_ROW)] * served.ok.leaves
    assert CR.registered_file(program, lay, shape, 0, {**roots, served.ok.value: [served.ok.root.hex()]}, served.root,
                              served.steps, run._salt.hex())[1][0] == "registration"


class _Prepared(_Receipts):
    """The stand-in with a ``prepare``: each batch it was handed (with the asks made before it), and each ask's
    arguments, by call."""

    def __init__(self, fail=None):
        super().__init__()
        self.batches, self.asked, self.fail = [], {}, fail

    def prepare(self, batch):
        if self.fail:
            raise ValueError(self.fail)
        self.batches.append((len(self.order), batch))

    def __call__(self, call, codes, s_a, w, s_w):
        self.asked[call.index] = (call, codes, s_a, w, s_w)
        return super().__call__(call, codes, s_a, w, s_w)


def test_a_producer_with_prepare_is_handed_each_steps_calls_of_one_k_before_it_is_asked_them():
    """A receipt producer with a ``prepare`` is handed each step's calls of one K together before any of them is asked:
    the same calls with the same arguments, in the order they are then asked, so it may compute them in one launch; each
    receipt table is still the producer's ask's words. A ``prepare`` that refuses refuses the window, naming the calls."""
    made = _Prepared()
    run, _ws, _got = _window(W1, receipts=made)
    served, at = run.served, 0
    assert not served.refused and made.batches
    for asked_before, batch in made.batches:
        calls = [args[0].index for args in batch]
        assert asked_before == at and made.order[at:at + len(calls)] == calls == served.step_calls(calls[-1])
        assert all(x is y for args in batch for x, y in zip(args, made.asked[args[0].index], strict=True))
        at += len(calls)
    assert at == len(run.calls)
    for c in run.calls:
        rc = served.registered[SV.PREFIX + f"receipt@{c.index}"]
        assert rc.root.hex() == _tree(rc, [w.tobytes() for w in made[c.index].astype("<u2")], run._key,
                                      SV.RECEIPT_NONCE + c.index)
    refused = _window(W1, receipts=_Prepared(fail="no launch"))[0].served.refused
    assert any(r.startswith("calls [") and r.endswith("]: no launch") for r in refused), refused


class _OnDevice(torch.Tensor):
    """A host tensor that says it is on a CUDA device, as the engine's ``a`` is."""

    @property
    def is_cuda(self):
        return True


class _Staged(_Prepared):
    """The stand-in with a ``stage`` too: each call it was handed, with the asks made before it and the rows."""

    def __init__(self, fail=None):
        super().__init__()
        self.staged, self.stage_fail = [], fail

    def stage(self, call, a):
        if self.stage_fail:
            raise ValueError(self.stage_fail)
        self.staged.append((len(self.order), call.index, a))


def test_a_producer_with_stage_is_handed_each_served_calls_device_rows_before_its_asked_them(monkeypatch):
    """A receipt producer with a ``stage`` is handed each served call and its rows ``a`` where they are on a CUDA
    device: the engine's tensor itself, in call order, before any of its step's calls of K are asked; each receipt table
    is still the producer's ask's words. Rows on the host stage nothing, and a ``stage`` that refuses refuses the
    window, naming the call."""
    commit, seen = SV.Served.commit, {}

    def on_device(self, run, call, a, *rest):
        a = seen[call.index] = a.as_subclass(_OnDevice)
        return commit(self, run, call, a, *rest)

    monkeypatch.setattr(SV.Served, "commit", on_device)
    made = _Staged()
    run, _ws, _got = _window(W1, receipts=made)
    served = run.served
    assert not served.refused
    assert [c for _asked, c, _a in made.staged] == [c.index for c in run.calls if c.m]
    for asked, c, a in made.staged:
        assert a is seen[c] and c not in made.order[:asked] and asked == made.order.index(served.step_calls(c)[0])
    for c in run.calls:
        rc = served.registered[SV.PREFIX + f"receipt@{c.index}"]
        assert rc.root.hex() == _tree(rc, [w.tobytes() for w in made[c.index].astype("<u2")], run._key,
                                      SV.RECEIPT_NONCE + c.index)
    refused = _window(W1, receipts=_Staged(fail="no copy"))[0].served.refused
    first = min(c for _asked, c, _a in made.staged)
    assert f"call {first}: no copy" in refused, refused
    monkeypatch.undo()
    host = _Staged()
    assert not _window(W1, receipts=host)[0].served.refused and host.batches and not host.staged


def test_zy_words_are_record_words_of_the_records_split():
    """`zy_words` is `record_words` of `pouw_h1_records.split`: the records' own words where the Program's tile is their
    proof block and they cover every slot, taken apart otherwise (fewer rows than slots, another tile)."""
    from verity_vllm.protocols.pouw.h1 import records as ZR

    rng = np.random.default_rng(3)
    for rows, n, slots, tile in ((64, 96, 64, (32, 32)), (32, 64, 64, (32, 32)), (64, 64, 64, (16, 32))):
        records = rng.integers(0, 256, ((rows // 32) * (n // 32), ZR.ZY_BYTES), dtype=np.uint8)
        got = SV.zy_words(records, rows, n, slots, tile)
        assert got.dtype == np.uint16 and np.array_equal(got, SV.record_words(*ZR.split(records, rows, n), slots, tile))


def test_a_forward_out_of_module_order_lays_out_each_step_table_and_its_receipts_in_call_order():
    """GLM-5.3's DSA attention runs a step's linears out of module order. Here each step calls k before q, which
    share K = 16, then o: each step's A table of K is still its calls' rows in call order, registered at the call that
    completes it (q's, the last of K to come, not k's, the last in call order), and the producer is handed, then asked,
    that step's calls of K in call order there, each receipt table after the roots its proof blocks' seeds take."""
    made = _Prepared()
    run, _ws, got = _window(W1, served=2, receipts=made, order=(1, 0, 2))
    served, prog = run.served, run.served.program
    assert not served.refused and run.order_ok and run.static()
    tables = CR.step_tables({"calls": served.calls, "linears": served.linears, "m": served.m}, STEPS)
    assert tables == [(0, 16, [0, 1]), (0, 32, [2]), (1, 16, [3, 4]), (1, 32, [5])]
    for _s, k, cs in tables:
        ra = served.registered[HW.table(prog, "a", cs[0], STEPS)]
        rows = [row for c in cs for row in _a_rows(*_served_call(got, c, 0, k)[:2], T, k)]
        assert ra.root.hex() == _tree(ra, rows, run._key, SV.nonce(cs[0], "a"))
        assert [c.index for c in run.calls if ra.value in run.tables[c.index]["registered"]] == [cs[0]]
    assert [[args[0].index for args in batch] for _n, batch in made.batches] == [cs for _s, _k, cs in tables]
    assert made.order == list(range(len(run.calls)))
    names = list(served.registered)
    for c in run.calls:
        rc = served.registered[SV.PREFIX + f"receipt@{c.index}"]
        assert names.index(rc.value) > names.index(served.a_table(c.index))
        assert rc.root.hex() == _tree(rc, [w.tobytes() for w in made[c.index].astype("<u2")], run._key,
                                      SV.RECEIPT_NONCE + c.index)


def test_receipts_off_the_units_receipt_width_are_refused():
    """Receipts of another width than the tile proof block's receipt (u32 words where the stand-in's is one u16 word) register
    nothing, and `trees` says why."""
    run, _ws, _got = _window(W1, receipts=_Receipts(np.uint32, 2))
    assert any("its receipts are 4 rows of 4 u16 words, not its 4 units' receipts of 1" in r for r in run.served.refused)
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


#: Llama-3.1-8B's two contractions on a small slice of it: up_proj's K = 4096, one leaf a row, and down_proj's
#: K = 14336, 4 chunk leaves a row (SPLITK)
LLAMA = {"shapes": [(64, 4096), (32, 14336)], "modules": {0: "up_proj", 1: "down_proj"}, "blocks": (0, 1)}


def _chunk_rows(rows: list[bytes], k: int) -> list[bytes]:
    """Whole rows (each its K codes, then its f32 scale's 4 bytes) as SPLITK's leaves: row r's chunk c, at leaf n r + c,
    is ``codes[c K_c : (c + 1) K_c] ‖ scale``, n = `weights.chunks` (K) and K_c = K / n."""
    n = Wt.chunks(k)
    kc = k // n
    return [row[c * kc:(c + 1) * kc] + row[k:] for row in rows for c in range(n)]


@pytest.mark.parametrize("k, n", [(64, 1), (4096, 1), (5120, 2), (6144, 2), (14336, 4), (16384, 4)])
def test_chunk_words_are_the_specs_leaf_bytes(k, n):
    """SPLITK §1: `chunk_words` at n = `weights.chunks` (K) gives row r's chunk c at leaf n r + c, its words the bytes
    ``codes[r, c K_c : (c + 1) K_c] ‖ scale[r]`` little-endian, `weights.leaf_words` (K) of them; at K <= 4096 it is
    `row_words`, and word for word core's `h1zy.chunk_words`. An n that doesn't cut K into whole u16 words is refused."""
    rng = np.random.default_rng(k)
    codes, scales = rng.integers(-128, 128, (3, k), dtype=np.int8), (rng.random(3) + 0.5).astype("<f4")
    got = SV.chunk_words(codes, scales, Wt.chunks(k))
    assert Wt.chunks(k) == n and got.dtype == np.uint16 and got.shape == (3 * n, Wt.leaf_words(k))
    rows = [codes[r].tobytes() + scales[r].tobytes() for r in range(3)]
    assert [w.astype("<u2").tobytes() for w in got] == _chunk_rows(rows, k)
    assert got.tolist() == Z.chunk_words(rows, k)
    assert (n == 1) == np.array_equal(got, SV.row_words(codes, scales))
    with pytest.raises(ValueError, match="chunks of whole u16 words"):
        SV.chunk_words(codes, scales, 3 if k % 3 else 7)


def test_k_up_to_4096_roots_are_unchanged():
    """SPLITK leaves every K <= 4096 as it was: the mixed-K window's A tables and ``b/<K>`` (K = 16 and 32), served
    from a seeded torch, have the roots they had before chunk rows. Every root is registered under the declared Program's
    digest, so a change to the window's layout repins them."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        ws = _weights()
        period = _period(ws)
        run, _ws, _got = _window(W1, weights=period)
    got = {v: r.root.hex()[:32] for v, r in run.served.registered.items() if "/a/" in v or "/b/" in v}
    assert period.root().hex()[:32] == "0b889e954429d1cdac365f8e23e8153c"
    assert got == {SV.PREFIX + v: h for v, h in {
        "a/16@0": "04b9eb08e826e1494240fc36ab0bc5e4", "a/16@1": "e899d34fbfa90038cbdf2585410f56b5",
        "a/32@0": "3087ac7de2b46030df6a89c0dde76fca", "a/32@1": "9d7cca0084a5daad92b6536fba6fd31c",
        "b/16": "9af66bcd442455a00f1bbc66995815c7", "b/32": "da3a215e48a1fa5f72401e28e5791cc6"}.items()}


#: kernels' SPLITK vectors (``tests/vectors/splitk_rows.json``, `splitk_rows_vectors`): their
#: inputs from SHAKE256 of `VECTORS_LABEL`, each case (case, value, K, rows, nonce, binds the window) and its pinned
#: (leaves, words, domain, root)
VECTORS_LABEL = b"verity/pouw/splitk-rows-vectors/v1"
VECTORS = [
    ("a-14336-s0", "a/14336@0", 14336, 64, (1 << 61) + (1 << 42), True, 256, 1794,
     "8acce5e4bf00be8d31e84814e2b349dd5e0906103d55748662898e15e1501c93a50a993b29b5add3c21e89a8e984942ec4a943b0ee4df17e5ea4e239bee20e56",
     "b2e05799b4fad2b3795b165d5e262d2b35e8776ef3d41eabc132aa160a0fbed537e53927727063f844054ddcce12cb1a5e0e941e4442a4ba50231f7ab72cc369"),
    ("b-6144", "b/6144", 6144, 64, Wt.B_NONCE, False, 128, 1538,
     "ee06cb8014a9ea97fa6e3c319a61ac9094dfa6727b6a8bca92c9c6daf8098080499f505edeb236c03264fb695e79f2f2c9985ae9250712ae33de36591fbf5358",
     "f93f07478d0900b061ffc2a6d6b24aa342a7b0247380bd95b2aa3f80fffc8567bed58d4be497b41321d288a33bdd910062e394dbf70444582a7e8f014b49e7ad"),
]


def _shake(part: bytes, n: int) -> bytes:
    return hashlib.shake_256(b"\0".join((VECTORS_LABEL, part))).digest(n)


def _vector_rows(case: str, k: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    """A vector case's int8 codes (rows, K) and f32 scales: row r the r-th K + 4 bytes of SHAKE256(label, NUL, case),
    its scale's bits set into [2^-7, 2^-6)."""
    b = np.frombuffer(_shake(case.encode(), rows * (k + 4)), np.uint8).reshape(rows, k + 4)
    bits = 0x3C000000 | (np.ascontiguousarray(b[:, k:]).view("<u4").reshape(rows) & 0x007FFFFF)
    return np.ascontiguousarray(b[:, :k]).view(np.int8), bits.astype("<u4").view("<f4")


@pytest.mark.parametrize("case", VECTORS, ids=[c[0] for c in VECTORS])
def test_the_served_paths_chunk_leaves_are_kernels_vectors(case):
    """Two SPLITK test vectors through the served path's own code: ``a/14336@0`` (64 rows, n = 4) is
    `Served.a_words` of the rows over a 64-slot table, committed as `Served` commits a step's A table (at call 0's A
    nonce, `nonce`, binding the window), and ``b/6144`` (64 rows, n = 2) is `register_weights`' ``b/<K>`` (at `B_NONCE`,
    binding none); each has the vectors' leaves, words, domain and root."""
    name, value, k, rows, nonce, windowed, leaves, words, domain, root = case
    codes, scales = _vector_rows(name, k, rows)
    program, window, key = _shake(b"program", 64).hex(), _shake(b"window", 64).hex(), _shake(b"key", 32)
    if windowed:
        assert SV.nonce(0, "a") == nonce
        served = SV.Served()
        served.m = rows
        reg = NR.commit(program, value, served.a_words(SimpleNamespace(k=k), codes, scales), key, nonce, window=window)
    else:
        reg = SV.register_weights(program, [(rows, k)], {0: (codes, scales)}, key, prefix="").registered[k]
    assert (reg.value, reg.leaves, reg.words) == (value, leaves, words)
    assert (reg.domain.domain_id.hex(), reg.root.hex()) == (domain, root)


def _llama(ws=None, edit=None, defn=StandIn):
    """`LLAMA`'s window over its period's weights (`_period`), down_proj's served codes ``edit`` of them when given, its
    proof blocks ``defn``'s."""
    ws = _weights(0, LLAMA["shapes"]) if ws is None else ws
    period = _period(ws, defn=defn, **LLAMA)
    served = ws if edit is None else [ws[0], edit(ws[1])]
    run, _ws, got = _window(W1, ws=served, weights=period, defn=defn, **LLAMA)
    return run, ws, got, period


def test_a_chunked_llama_window_registers_chunk_leaves_and_core_credit_ties_it():
    """SPLITK §4 on a Llama-shaped window: up_proj at K = 4096 keeps one leaf a row, down_proj at K = 14336 is 4.
    Each step's ``a/<K>@<s>`` is M·|calls|·n leaves of `weights.leaf_words` (K) words, credit's `table_rows`, its root
    the spec's leaf bytes of the served rows; ``b/<K>`` is the period's n·Σn leaves (`register_weights`), its root
    the chunk leaves of the weights, and `Weights.matches` holds a served linear to them. Core's credit ties the window:
    `_a_link` (each table whole under its round's root, at the layout's rows) and `_weights` (the period's entries,
    under the window's open root)."""
    run, ws, got, period = _llama()
    served, pre = run.served, SV.PREFIX
    assert not served.refused
    layout = CR.layout_of(SV.descriptor(served.program))[0]
    rows = CR.table_rows(layout, STEPS, pre)
    for s, k, cs in CR.step_tables(layout, STEPS):
        ra = served.registered[pre + f"a/{k}@{s}"]
        assert (ra.leaves, ra.words) == (rows[ra.value], Wt.leaf_words(k)) == (Wt.chunks(k) * T * len(cs), Wt.b_words(k // Wt.chunks(k)))
        whole = [row for c in cs for row in _a_rows(*_served_call(got, c, 0, k)[:2], T, k)]
        assert ra.root.hex() == _tree(ra, _chunk_rows(whole, k), run._key, SV.nonce(cs[0], "a"))
    b4, b14 = served.registered[pre + "b/4096"], served.registered[pre + "b/14336"]
    assert (b4.leaves, b4.words, b14.leaves, b14.words) == (64, 2050, 4 * 32, 1794)
    assert b14 is period.registered[14336] and b4 is period.registered[4096]
    whole = [w[j].numpy().tobytes() + np.float32(s[j, 0]).tobytes() for w, s in ws[1:] for j in range(len(w))]
    assert b14.root.hex() == _tree(b14, _chunk_rows(whole, 14336), WKEY, SV.B_NONCE + 1)
    assert period.at == {0: (4096, 0), 1: (14336, 0)} and period.matches(1, ws[1][0].numpy(), ws[1][1].numpy())
    assert CR._a_link(run.round_log, served.entries(), len(run.calls), rows) is None
    requests, session = hashlib.sha512(b"requests").digest(), "llama-session"
    window = {"root": PS.window_open_root(period.root(), requests, session).hex()}
    assert CR._weights(window, layout, served.entries(), requests.hex(), session, served.program_digest, pre) == (
        period.root(), None)
    whole = [e if e["value"] != pre + "b/14336" else {**e, "leaves": 32, "words": Wt.b_words(14336)}
             for e in served.entries()]
    assert CR._weights(window, layout, whole, requests.hex(), session, served.program_digest, pre)[1] == (
        "weights", CR.other_weights_entry(pre + "b/14336", 14336, 32))


def _last_chunk_code(w):
    codes = w[0].clone()
    codes[3, 14335] ^= 1
    return codes, w[1]


@pytest.mark.parametrize("defn", [StandIn, StandInJ10R], ids=["v1", "j10r"])
def test_a_chunked_window_serving_a_code_off_its_periods_weights_is_refused(defn):
    """`Weights.matches` at n = 4 reads down_proj's rows from leaf n·lo: one code in a row's last chunk other than the
    period's refuses the window, v1's or J10-R's, at the call that completes K = 14336, and `trees` says why."""
    run, _ws, _got, _period_w = _llama(edit=_last_chunk_code, defn=defn)
    assert any("linears [1]' codes or s_w are not their rows of the period's registered h1/b/14336" in r
               for r in run.served.refused), run.served.refused
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


def test_a_j10r_llama_window_at_k_14336_is_credited_at_its_whole_k_and_core_ties_it():
    """J10-R over chunk rows: `LLAMA`'s window over J10-R proof blocks (`StandInJ10R`, as `h1zy.PouwH1TileZyChunks_v2` presents
    J10-R's tile at a chunked K) has down_proj at K = 14336 in 4 chunk rows a row, and credit reads it as J10-R (`shape_of`'s
    ``j10r``, n = 4). Each call's ``--registered`` file is the one credit builds from the descriptor, its 4 R A chunk
    ports included, and its public rows hold each proof block's call and blocks at the call's whole K (`core_check.j10r_blocks`
    of K = 14336, one noise stream a call): a public table at a chunk's K (3584) has another root and opens none of
    credit's rows. Core's credit ties the window (`_a_link`, `_weights` at the period's root) and refuses ``b/14336``
    registered as whole rows."""
    real = HW._chunked(14336, HW.tile_unit(14336, TILE, HT.PouwH1Tile5))
    assert real.defn is Z.PouwH1TileZyChunks2 and real.bindings["CHUNKS"] == 4
    got, want = dict(_unit(defn=StandInJ10R)(14336).signature()[0]), dict(real.signature()[0])
    assert list(got) == list(want) == list(HW.TILE_PARAMS_J10R)
    assert all(got[p] == want[p] for p in ("a", "b", "zy", "call", "blocks"))
    run, _ws, _got, period = _llama(defn=StandInJ10R)
    served, pre = run.served, SV.PREFIX
    assert not served.refused and HW.is_j10r(served.program)
    raw = SV.descriptor(served.program)
    (layout, why), program, shape = CR.layout_of(raw), hashlib.sha512(raw).hexdigest(), CR.shape_of(raw)
    assert why is None and program == served.program_digest
    assert {k: (t["j10r"], t["n"]) for k, t in shape["tiles"].items()} == {4096: (True, 1), 14336: (True, 4)}
    roots = {}
    for e in served.entries():
        roots.setdefault(e["value"], []).append(e["root"])
    files = SV.register(run).files
    for c, (u0, n) in enumerate(HW.call_units(served.program)):
        k, cols = run.calls[c].k, run.calls[c].n // TILE[1]
        mine, why = CR.registered_file(program, layout, shape, c, roots, served.root, served.steps, run._salt.hex())
        assert why is None and files[c] == mine, why
        blocks = [CC.j10r_blocks(T, k, TILE, t // cols, t % cols) for t in range(n)]
        assert HW.call_blocks(served.program, c) == blocks
        ab = (served.registered[served.a_table(c)].root, served.registered[pre + f"b/{k}"].root)
        pub, r = mine["ports"][CR.PUB_PORT], served.registered[pre + f"pub@{c}"]
        assert pub["known"]["rows"] == [list(row) for row in r.rows] == [
            CC.public_words(run._salt, u0 + t, ab, (u0, blocks[t])) for t in range(n)]
        if k == 14336:
            assert {f"a_k14336_{i}" for i in range(4 * TILE[0])} <= set(mine["ports"])
            chunk = [CC.public_words(run._salt, u0 + t, ab, (u0, CC.j10r_blocks(T, k // 4, TILE, t // cols, t % cols)))
                     for t in range(n)]
            assert chunk != pub["known"]["rows"]
            off = NR.commit(program, r.value, np.array(chunk, np.uint16), run._key, 0, True, b"".join(r.salts),
                            window=served.root)
            assert off.root.hex() != pub["root"]
            assert not any(G.verify_read(r.domain, bytes.fromhex(pub["root"]), t, off.bcs[t], off.path(t), SV.V2)
                           for t in range(n))
    assert CR._a_link(run.round_log, served.entries(), len(run.calls), CR.table_rows(layout, STEPS, pre)) is None
    requests, session = hashlib.sha512(b"requests").digest(), "llama-session"
    window = {"root": PS.window_open_root(period.root(), requests, session).hex()}
    assert CR._weights(window, layout, served.entries(), requests.hex(), session, served.program_digest, pre) == (
        period.root(), None)
    whole = [e if e["value"] != pre + "b/14336" else {**e, "leaves": 32, "words": Wt.b_words(14336)}
             for e in served.entries()]
    assert CR._weights(window, layout, whole, requests.hex(), session, served.program_digest, pre)[1] == (
        "weights", CR.other_weights_entry(pre + "b/14336", 14336, 32))
