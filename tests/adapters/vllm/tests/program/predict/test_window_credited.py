"""verity_vllm.program.predict.window_circuit with credited linears (`Credited`): at each, Pearl+'s grid of
`PearlPlusTileCoreWords_v2` units in the product's place, its x widened to A rows by wire and its y handed on to every
reader by wire; with none credited the circuit is today's byte for byte. Pinned on TINY with one layer whose MLP is 256
wide, so down_proj (K = 256, the least the tile takes) is creditable, at WINDOW's requests of 5 and 4 positions: position
3's two rows (each request's, live from its last prompt position on) a whole row tile, position 4's one row a row tile
with a filler row."""
from __future__ import annotations

import random
from collections import Counter
from functools import cache
from typing import Any

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.program import is_input
from verity.circuits.refs import runs
from verity.commitments.periods import interface as IF
from verity.protocols.resource_accounting.compute.pouw.prover import schemes as pouw_schemes
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import inert_bound
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c as C
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_u as CU

from tests.program.predict import test_steps as S
from tests.program.predict.test_window_circuit import CLASS, GRAIN, WINDOW, made_up, wired
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_class import Credited, PearlTables

SCHEME = "pearl-c-h100-v1-h3"
CFG = {**S.TINY, "intermediate_size": 256, "num_hidden_layers": 1}
#: down_proj, a position's fourth `Gemm_v2`
DOWN = 3
CREDITED = Credited(SCHEME, frozenset({(3, DOWN), (4, DOWN)}), tile=(2, 8))
TILE = "PearlPlusTileCoreWords_v2"
K, N, DIGEST = 256, 64, 512


@cache
def _window(cfg: str = "mlp256") -> tuple[Any, Any]:
    reg = S._registered(CFG if cfg == "mlp256" else S.TINY)
    return reg, S.compile_steps(reg, *S._session(*WINDOW))


def _lower(credited: Credited | None) -> WC.Lowered:
    reg, comp = _window()
    return WC.lower(reg, CLASS, None, GRAIN, requests=WC.takes(comp, CLASS, None), credited=credited)


def _bytes(c: IF.Circuit) -> bytes:
    return IF.circuit_bytes(c.description())


def _units(c: IF.Circuit, family: str) -> list[tuple[Any, int]]:
    return [(fn, n) for fn, n in c.templates if fn.id.startswith(family + "{")]


def test_with_no_credited_linear_the_circuit_is_todays():
    reg, comp = _window("tiny")
    none = Credited(SCHEME)
    assert _bytes(WC.window_circuit(reg, comp, CLASS, GRAIN, credited=none)) == _bytes(WC.window_circuit(reg, comp, CLASS, GRAIN))
    assert _bytes(WC.class_circuit(reg, CLASS, None, GRAIN, credited=none)) == _bytes(WC.class_circuit(reg, CLASS, None, GRAIN))


def test_a_credited_linear_is_its_tiles_their_a_rows_and_their_y_rows_in_the_slices_place():
    """Position 3's two rows are one row tile of 8 column tiles, position 4's one row another with a filler row: 3 A rows,
    16 units of two templates and 24 row splits, a tile row each, replace the 3 · 64 / 4 slices of `Gemm_v2{K=256,N=4}`; the certificates
    and the filler row are the only free rows, the salt and the weight's root are supplied, the rest registered."""
    plain, c = _lower(None).circuit, _lower(CREDITED).circuit
    old, new = Counter(dict((fn.id, n) for fn, n in plain.templates)), Counter(dict((fn.id, n) for fn, n in c.templates))
    (down,) = [i for i in old if i.startswith("Gemm_v2{K=256,N=4,")]
    assert old[down] - new[down] == 3 * N // 4
    assert sum(n for _fn, n in _units(c, "LinearARow_v3")) == 3
    assert sorted((fn.bindings["TR"], fn.bindings["I"], n) for fn, n in _units(c, "LinearYRow_v1")) == [(1, 0, 8), (2, 0, 8), (2, 1, 8)]
    assert sorted((fn.bindings["TR"], n) for fn, n in _units(c, TILE)) == [(1, 8), (2, 8)]
    assert sorted((f.port, f.units, f.cut) for f in c.free) == [("a", 8, (1, 1)), ("cert", 8, None), ("cert", 8, None)]
    prog, _q = wired(c)
    assert len(prog.annotations["free"]) == 3
    assert sorted(m["supplied"] for m in c.registered.supplied.values()) == ["pouw.pearlplus.salt/v1", "registered.root/v1"]
    assert [v.split("@")[0] for v in c.registered.values if v.startswith("pearlplus/")] == [
        "pearlplus/" + p for p in ("unit", "fa", "fb", "g", "eb", "rows")]


def test_a_credited_linear_is_refused_beside_tables_and_where_its_tile_cant_be():
    reg, _comp = _window()
    with pytest.raises(Unexpressible, match="reading no tables"):
        WC.class_circuit(reg, CLASS, None, GRAIN, tables=PearlTables(SCHEME, words=16), credited=CREDITED)
    with pytest.raises(Unexpressible, match="promotion windows"):
        _lower(Credited(SCHEME, frozenset({(0, 0)}), tile=(2, 8)))
    with pytest.raises(Unexpressible, match="don't divide"):
        _lower(Credited(SCHEME, frozenset({(3, DOWN)}), tile=(2, 2)))


def _flat(x: Any) -> list[int]:
    if isinstance(x, (bytes, bytearray)):
        return list(x)
    if isinstance(x, int):
        return [x]
    return [w for v in x for w in _flat(v)]


def _rows(requests: list[Any], position: int) -> int:
    return sum(1 for r in requests if r.positions > position)


def _tiles(requests: list[Any]) -> dict[tuple[str, int], tuple[int, Any]]:
    """Each tile unit, by template and unit, as (its credited linear's index, its tile), in `credit`'s allocation order."""
    out, seen = {}, Counter()
    for i, (p, _call) in enumerate(sorted(CREDITED.linears)):
        for t in PC.tiles(SCHEME, K, _rows(requests, p), N, CREDITED.tile):
            out[(descriptor_id(t.core), seen[t.core.id])] = (i, t)
            seen[t.core.id] += 1
    return out


def _public(tiles: dict, salt: bytes, root: bytes) -> dict[str, list[int]]:
    """The public row as its registered values' leaves, in `credit`'s order: one row a linear of unit, fa, fb and g, a
    column tile's of eb, a row tile's of rows; a port of 32-bit leaves its leaves, any other its 16-bit words."""
    rows: dict[str, dict[tuple[int, int], list[int]]] = {p: {} for p in ("unit", "fa", "fb", "g", "eb", "rows")}
    for i, t in tiles.values():
        pub, types, words = PC.public(SCHEME, K, t, 7 + i, salt, root), dict(t.core.signature()[0]), dict(IF.layout(t.core).rows)
        for p in rows:
            key = (i, t.col if p == "eb" else t.rows[0] if p == "rows" else 0)
            leaves = _flat(pub[p])
            rows[p][key] = leaves if types[p].homogeneous_width() == 32 else IF.rows_of_leaves(
                np.asarray([leaves], dtype=np.uint64), types[p], words[p])[0].tolist()
    return {p: [w for _k, r in sorted(rs.items()) for w in r] for p, rs in rows.items()}


def _honest(t: Any, unit: int, salt: bytes, root: bytes, a: np.ndarray, b: np.ndarray, m: int) -> tuple[list[int], list[int]]:
    """Tile ``t``'s certificates on A rows ``a`` (FP32 words) and weight columns ``b`` (BF16 words), and its y."""
    tm, tn = len(t.rows), len(t.cols)
    s = pouw_schemes.get(SCHEME, k_min=K, tile=(tm, tn))
    _d, ng = inert_bound._promoted(s.device.name, K)
    ctx = C.Context(salt, C.Matmul(unit, C.Shape(-(-m // tm) * tm, K, N), 0), bytes(64), root)
    ops = s.encode(ctx, s.noise(ctx), [C.f32_row(r) for r in a.reshape(tm, K).tolist()],
                   [C.f32_row([v << 16 for v in r]) for r in b.reshape(tn, K).tolist()], C.Tile(t.rows, t.cols))
    checked = s.checked(ctx, ops)
    cert = [[[list(inert_bound.certificates(ops.a.codes[i], ops.b.codes[j], s.device.name))[2 * z:2 * z + 2]
              for z in range(ng - 1)] for j in range(tn)] for i in range(tm)]
    return _flat(cert), [CU.y(checked.u[i][j], ops.a.alpha[i], ops.b.alpha[j]) for i in range(t.real) for j in range(tn)]


@pytest.mark.slow
def test_on_honest_values_every_tile_is_ok_writes_the_schemes_y_and_its_readers_take_it_by_wire():
    """Node by node on the wired Program: weights small BF16 words, every other parameter made up, the public row, the
    salt and the root as the verifier fills them, each tile's certificates the prover's on the A rows and weight columns it
    is wired and read, the filler row zero. Each tile's ok is 1 and its y the scheme's (`pearl_c_u.y`), each row's split
    hands that row's words on to down_proj's reader; a weight word or a certificate word changed makes the tile's ok 0. Slow (16
    tiles of about 280 million gates): only evaluating a tile on the rows its wires and reads gather shows that `credit`'s
    reads and the registered values' layout give it its inputs."""
    c = _lower(CREDITED).circuit
    prog, _q = wired(c)
    requests = WC.takes(_window()[1], CLASS, None)
    ids = [descriptor_id(fn) for fn, _ in c.templates]
    tiles = _tiles(requests)
    rng = random.Random(5)
    salt, root = rng.randbytes(32), rng.randbytes(64)
    sig = dict(prog.fn.signature()[0])
    inputs = made_up(prog)
    for name, t in sig.items():
        if name.startswith("weights/") and "/table/" not in name and not name.endswith("/root") and t.homogeneous_width() == 16:
            inputs[name] = np.asarray([int(np.float32(rng.uniform(-0.25, 0.25)).view(np.uint32)) >> 16
                                       for _ in range(t.leaves)], dtype=np.uint64)
    port = dict(_units(c, TILE)[0][0].signature()[0])
    for v, mark in c.registered.supplied.items():
        p, raw = ("salt", salt) if mark["supplied"] == "pouw.pearlplus.salt/v1" else ("root", root)
        inputs[v] = IF.rows_of_leaves(np.frombuffer(raw, np.uint8).astype(np.uint64)[None], port[p], sig[v].leaves)[0]
    for p, words in _public(tiles, salt, root).items():
        (v,) = [n for n in sig if n.startswith(f"pearlplus/{p}@")]
        inputs[v] = np.asarray(words, dtype=np.uint64)
    tile_ix = {j for j, tid in enumerate(ids) if tid.startswith(TILE)}
    for j in tile_ix:
        if f"{j}.a" in inputs:
            inputs[f"{j}.a"][:] = 0

    names, vals, at = list(sig), [], {}
    order = iter(c.description()["order"])
    cur: list[Any] = []
    ok, ys, bad, split, readers = [], [], [], [], set()
    splits: set[int] = set()

    def gather(node: Any) -> list[np.ndarray]:
        return [np.concatenate([vals[i][lo:hi] for _s, i, lo, hi in runs(a.refs)]).reshape(node.n, -1) for a in node.args]

    for k, node in enumerate(prog.root.body.nodes):
        if is_input(node.fn):
            at[names[len(at)]] = k
            vals.append(np.asarray(inputs[names[len(at) - 1]], dtype=np.uint64).reshape(-1))
            continue
        if not cur or cur[2] == 0:
            cur = list(next(order))
        tid, lo = cur[0], cur[1]
        assert tid == descriptor_id(node.fn)
        cur[1], cur[2] = lo + node.n, cur[2] - node.n
        j, ports = ids.index(tid), [p for p, _t in node.fn.signature()[0]]
        if j in tile_ix:
            args = gather(node)
            certs, want = vals[at[f"{j}.cert"]], []
            per = certs.size // c.templates[j][1]
            for q in range(node.n):
                i, t = tiles[(tid, lo + q)]
                cert, y = _honest(t, 7 + i, salt, root, args[ports.index("a")][q], args[ports.index("b")][q],
                                  _rows(requests, sorted(CREDITED.linears)[i][0]))
                certs[(lo + q) * per:(lo + q + 1) * per] = cert
                want.append(y)
            args = gather(node)
            w_bad, c_bad = [x[:1].copy() for x in args], [x[:1].copy() for x in args]
            w_bad[ports.index("b")][0, 5] ^= 1 << 3
            c_bad[ports.index("cert")][0, 0] ^= 1
            out = np.asarray(evaluate_batch(node.fn, *(np.concatenate(x) for x in zip(args, w_bad, c_bad))),
                             dtype=np.uint64).reshape(node.n + 2, -1)
            ok += out[:node.n, -1].tolist()
            ys += [out[q, DIGEST:-1].tolist() == want[q] for q in range(node.n)]
            bad += out[node.n:, -1].tolist()
            vals.append(out[:node.n].reshape(-1))
            continue
        args = gather(node)
        out = np.asarray(evaluate_batch(node.fn, *args), dtype=np.uint64).reshape(-1)
        if tid.startswith("LinearYRow_v1"):
            splits.add(k)
            tr, tn = node.fn.bindings["TR"], node.fn.bindings["TN"]
            split.append(np.array_equal(out, args[0].reshape(node.n, tr, tn)[:, node.fn.bindings["I"]].reshape(-1)))
        readers |= {tid.split("{")[0] for a in node.args for _s, i, _lo, _hi in runs(a.refs) if i in splits}
        vals.append(out)

    print("readers of y", sorted(readers), "ok", ok, "tampered", bad)
    assert ok == [1] * 16 and all(ys) and bad == [0] * len(bad) and len(bad) >= 4, (ok, ys, bad)
    assert split and all(split), split
    assert readers and not readers & {"LinearYRow_v1", TILE}, readers
