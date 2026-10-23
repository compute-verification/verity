"""`protocol_options.pouw_h1_served`: an H1 window that mixes K, served under compute accounting's window Program
(`h1window.program` on vllm's 32 × 32 tile, one template per K), registers each call's tables as it is recorded: its A
table ``a/<K>@<c>`` (M slot rows, each its codes then its f32 s_a) and its z table ``zy@<c>`` (one z/y record a unit, in
tile order), and the weights' ``b/<K>`` (each linear's rows once, codes then s_w), as ``hm96-sha512/row/v2`` trees whose
rows and roots recompute here from the served tensors, under domains bound to the window's open root. Each root is in its
call's manifest row, so its round's root commits it before the round's coin. A call's statement file
(`h1window.table_reads_file`) resolves each of its units' ports (`h1window.window_ports`, each K's own; ``a_k<K>`` →
``a_k<K>_0`` for a tile of one row, ``b_k<K>`` → ``b_k<K>_0`` for one column) to its own tables and the weights', and no
other call's unit reads them; a read opened with another row is refused. A weight that changes, blocks that aren't one
per linear, an odd K, n off C and a served call with no z/y record register nothing; a second window reusing the first's
key is refused (R7-held). Every window of a period declares one Program (the period's prefix), its call tables its own
through their window-bound domains, and its ``b/<K>`` the period's registered weights (VLLM-71), which a window serving
another code or s_w is refused for. A stand-in tile unit of the zy unit's parameter shape (`h1zy.ROW` rows, its `h1zy.RECORD`, a
receipt sink and ``ok``) keeps the Program small."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw.audit import hm96_salt  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind  # noqa: E402
from verity.ml.circuits.ports import In, Out, Ports  # noqa: E402
from verity.ml.circuits.refs import Coll, coll_concat  # noqa: E402
from verity.ml.circuits.types import Array, Record, Value  # noqa: E402
from verity.ml.commitments import hm96  # noqa: E402
from verity.ml.commitments.merkle import tree_levels  # noqa: E402
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest  # noqa: E402
from verity_vllm.engine import int8_ops as OPS  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_requests as RQ  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_served as SV  # noqa: E402

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
XOR16 = PrimitiveDefinition("H1ServedTestXor16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: a ^ b, register=False)
FOLD16 = PrimitiveDefinition("H1ServedTestFold16", 1, (("key", V16), ("x", V16)), V16, lambda key, x: key ^ x,
                             register=False, ports=Ports(params={"key": In(kind="coin")}, ret=Out(sink=True)))
OUT = Record(("receipt", V16), ("ok", V1))


def _sig(S):
    return ((("a", Array(S.R, Z.ROW(S.K))), ("b", Array(S.C, Z.ROW(S.K))), ("zy", Z.RECORD(S.R, S.C)),
             ("salt", Array(4, V16)), ("position", V16), ("roots", Array(2, Array(4, V16)))), OUT)


def _first(c):
    return Coll(V1, c.refs.slice(0, 1))


def _body(B, S, a, b, zy, salt, position, roots):
    ok = B.call(XOR1, B.call(XOR1, _first(a[0]), _first(b[0])), _first(zy))
    return coll_concat(OUT, [B.call(FOLD16, salt[0], B.call(XOR16, position, roots[0][0])), ok])


StandIn = CompositeDefinition("H1ServedTestTile", 0, ("K", "R", "C"), _sig, _body, register=False,
                              ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")},
                                          ret={"receipt": Out(sink=True)}))


def _unit(tile=TILE):
    return lambda k: bind(StandIn, K=k, R=tile[0], C=tile[1])


def _codes(rng, n, k):
    return torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8))


def _weights(seed: int = 0, shapes=SHAPES):
    """Each linear's int8 codes and f32 scales (n, 1)."""
    rng, g = np.random.default_rng(1000 + seed), torch.Generator().manual_seed(seed)
    return [(_codes(rng, n, k), torch.rand(n, 1, generator=g) + 0.5) for n, k in shapes]


def _period(ws, shapes=SHAPES):
    """The period's `Weights` over `ws`, registered under the declared Program's digest with `WKEY`."""
    circuit = P.program_digest(SV.declared(shapes, MODULES, T, STEPS, TILE, BLOCKS, unit=_unit()))
    return SV.register_weights(circuit, shapes, {w: (c.numpy(), s.numpy().reshape(-1)) for w, (c, s) in enumerate(ws)},
                               WKEY)


def _window(key: bytes, seed: int = 0, served_key: bytes | None = None, shapes=SHAPES, blocks=BLOCKS, steps=STEPS,
            served=1, changed=None, modules=MODULES, tile=TILE, rows=T, no_z=False, ws=None, weights=None):
    """A window: its first ``served`` steps serve 5 rows each, q and k on one input x, o on another, every step with the
    same weights ``ws`` (`_weights` of ``seed`` by default; step 1's q on ``changed`` when given), each call's z/y records
    `scaled_mm_zy`'s over its T slots; the later steps are empty. With ``weights`` (`_period`) its ``b/<K>`` are the
    period's. Returns the run, each linear's weight and scales, and each served call's (a, s_a, z, y) by its index, z its
    exact product and y its served y."""
    rng = np.random.default_rng(seed)
    run = PH.H1Run("t", SALT, modules, shapes, rows, steps, key=bytes([seed + 1]) * 32,
                   served=SV.Served(tile, _unit(tile), blocks, window=served_key or key, weights=weights))
    ws = _weights(seed, shapes) if ws is None else ws
    got = {}
    for s in range(served):
        x = _codes(rng, SERVED, shapes[0][1])
        inputs = [x, x] + [_codes(rng, SERVED, k) for _n, k in shapes[2:]]
        for w, (a, (wt, sw)) in enumerate(zip(inputs, ws)):
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


def test_each_call_registers_its_tables_as_it_is_recorded():
    """Every call's A and z tables are the Program's `h1window.table` values, M rows of K/2 + 2 words and one record of
    3 R C words a unit, under the registered-value domain of the Program's digest bound to the window's open root; their
    roots recompute from the served tensors (an empty call's from zeros), and each is in its call's manifest row, so its
    round's root is over it."""
    run, ws, got = _window(W1)
    served, pre = run.served, SV.PREFIX
    prog, digest = served.program, P.program_digest(served.program)
    assert served.program_digest == digest and served.root == W1.hex() and served.m == T
    assert served.linears == [("q", 16, 64), ("k", 16, 32), ("o", 32, 32)]
    assert served.calls == [(m, s) for s in range(STEPS) for m in ("q", "k", "o")]
    assert list(served.registered) == [pre + "a/16@0", pre + "zy@0", pre + "a/16@1", pre + "zy@1", pre + "b/16",
                                       pre + "a/32@2", pre + "zy@2", pre + "b/32",
                                       *(pre + v for c, k in ((3, 16), (4, 16), (5, 32)) for v in (f"a/{k}@{c}", f"zy@{c}"))]
    for c in run.calls:
        n, k = run.shapes_[c.layer]
        a, s_a, z, y = got.get(c.index) or (torch.zeros(0, k, dtype=torch.int8), torch.zeros(0), torch.zeros(0, n),
                                             torch.zeros(0, n).bfloat16())
        ra, rz = served.registered[HW.table(prog, "a", c.index)], served.registered[HW.table(prog, "zy", c.index)]
        assert (ra.leaves, ra.words, rz.leaves, rz.words) == (T, k // 2 + 2, (T // 32) * (n // 32), 3 * 32 * 32)
        assert ra.domain.domain_id.hex() == G.domain(digest, ra.value, ra.words, ra.leaves, window=W1.hex()).domain_id.hex()
        assert ra.domain.domain_id != G.domain(digest, ra.value, ra.words, ra.leaves).domain_id
        assert ra.root.hex() == _tree(ra, _a_rows(a, s_a, T, k), run._key, SV.nonce(c.index, "a"))
        assert rz.root.hex() == _tree(rz, _records(z, y, T, n), run._key, SV.nonce(c.index, "zy"))
        assert {ra.value: ra.root.hex(), rz.value: rz.root.hex()}.items() <= run.tables[c.index]["registered"].items()
    for r, log in enumerate(run.round_log):
        assert RQ.step_digest(run.salt, r, run._rows([c for c in run.calls if c.step == r])).hex() == log["calls"]


@pytest.mark.parametrize("steps", [2, 3])
def test_b_is_each_linears_weight_once_a_window(steps):
    """b/<K> is Σn rows of K's linears, codes then s_w, whatever the window's steps, registered at the call that completes
    K's weights in step 0 and in that call's manifest row only; every step's units read the same rows."""
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
    first = [next(v for key, v in HW.table_reads(served.program, tp, u).items() if key.startswith("b_k") and
                  key.endswith("[0]")) for u in range(tp.population)]
    assert all(first[u] == first[u % per] for u in range(tp.population))


def test_a_calls_statement_reads_its_units_rows_at_its_tables_and_refuses_another_row():
    """Each call's file names its K's ports, ``a_k<K>_0`` .. ``a_k<K>_31``, ``b_k<K>_0`` .. ``b_k<K>_31`` and ``zy``, each in
    row/v2 with its tree's value, root, rows, width and domain (the window's): the A ports and the record at the call's
    own tables, B at the weights'. Only the call's units read them, at the rows `h1window.table_reads` gives, and each read
    opens its tree at its row, while the same path with another row is refused."""
    run, _ws, _got = _window(W1)
    served = run.served
    prog, tp = served.program, P.evaluate(served.program, served.query())
    reg = SV.register(run)
    for c, (u0, n) in enumerate(HW.call_units(prog)):
        k, f = run.shapes_[run.calls[c].layer][1], reg.files[c]
        assert set(f["ports"]) == {f"{t}_k{k}_{i}" for t in "ab" for i in range(32)} | {"zy"}
        assert {(p["row_schema"], p["word_bits"]) for p in f["ports"].values()} == {(SV.V2, 16)}
        assert {f["ports"][f"a_k{k}_0"]["value"], f["ports"]["zy"]["value"], f["ports"][f"b_k{k}_0"]["value"]} == {
            HW.table(prog, "a", c), HW.table(prog, "zy", c), SV.PREFIX + f"b/{k}"}
        for port, p in f["ports"].items():
            r = served.registered[p["value"]]
            assert (p["root"], p["words"], p["leaves"], p["domain"]) == (
                r.root.hex(), r.words, r.leaves, r.domain.domain_id.hex())
            key = port if port == "zy" else f"{port.rsplit('_', 1)[0]}[{port.rsplit('_', 1)[1]}]"
            for u, row in enumerate(p["reads"]):
                if not u0 <= u < u0 + n:
                    assert row is None
                    continue
                assert HW.table_reads(prog, tp, u)[key] == (p["value"], row)
                assert G.verify_read(r.domain, r.root, row, r.bcs[row], r.path(row), SV.V2)
                assert not G.verify_read(r.domain, r.root, row, r.bcs[(row + 1) % r.leaves], r.path(row), SV.V2)


@pytest.mark.parametrize("tile, lone", [((1, 32), "a"), ((32, 1), "b")], ids=["R=1", "C=1"])
def test_a_tile_of_one_row_or_column_maps_its_lone_read_to_port_0(tile, lone):
    """With R = 1 `registered.reads` and `h1window.table_reads` both key a unit's lone A row by its parameter's bare name
    ``a_k<K>`` (`h1window.row_key`), and the window's ports map it to ``a_k<K>_0`` with no ``[0]`` alias; likewise B's
    with C = 1. Each call's file names its K's port, read by each of its units."""
    run, _ws, _got = _window(W1, shapes=[(32, 16), (32, 16), (32, 32)], steps=1, tile=tile, rows=32)
    served = run.served
    ports = served.ports()
    assert all(ports[f"{lone}_k{k}"] == f"{lone}_k{k}_0" and f"{lone}_k{k}[0]" not in ports for k in KS)
    assert ports == HW.window_ports(served.program)
    reg = SV.register(run)
    other = "b" if lone == "a" else "a"
    for c, (u0, n) in enumerate(HW.call_units(served.program)):
        k = run.shapes_[c][1]
        assert set(reg.files[c]["ports"]) == {f"{lone}_k{k}_0", "zy"} | {f"{other}_k{k}_{i}" for i in range(32)}
        reads = reg.files[c]["ports"][f"{lone}_k{k}_0"]["reads"]
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
    f = HW.table_reads_file(two.served.program, tp, 0, one.served.roots(), two.served.ports(), prog=digest, window=W2.hex())
    assert {p["domain"] for p in f["ports"].values()} == {
        two.served.registered[p["value"]].domain.domain_id.hex() for p in f["ports"].values()}
    assert not {p["domain"] for p in f["ports"].values() if not CR.is_held(p["value"])} & {
        r.domain.domain_id.hex() for r in one.served.registered.values()}


def test_the_periods_weights_are_registered_once_and_every_window_registers_them():
    """VLLM-71: each K's ``b/<K>`` is K's linears' rows in order, each its K codes then its f32 s_w, registered once under
    the period's declared Program with no window and salts from the period's key; `Weights.root` is
    `credit.weights_root` of their roots, and every window of the period registers those same entries, its open root
    over that root."""
    ws = _weights()
    period = _period(ws)
    assert sorted(period.registered) == list(KS)
    for k, r in period.registered.items():
        assert (r.value, r.words) == (SV.PREFIX + f"b/{k}", CR.b_words(k))
        assert r.domain.domain_id == G.domain(period.circuit, r.value, r.words, r.leaves).domain_id
    rows = [w[j].numpy().tobytes() + np.float32(s[j, 0]).tobytes() for w, s in ws[:2] for j in range(len(w))]
    assert period.registered[16].root.hex() == _tree(period.registered[16], rows, WKEY, SV.B_NONCE)
    assert period.root() == CR.weights_root({k: r.root for k, r in period.registered.items()})
    for i, key in enumerate((W1, W2)):
        run, _ws, _got = _window(key, seed=i, ws=ws, weights=period)
        assert not run.served.refused
        assert [e for e in run.served.entries() if CR.is_held(e["value"])] == period.entries()
        assert run.served.program_digest == period.circuit


@pytest.mark.parametrize("edit", ["s_w", "code"])
def test_a_window_serving_other_weights_than_the_periods_is_refused(edit):
    """VLLM-71: a window serves the weights its period registered before its salt. One row's s_w, or one code, other than
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
    """Each round's root lists step r's calls' A tables ``a/<K>@<c>`` whole, as aligned subtrees (`credit.cover`) whose
    tree is the table's registered root, and no other round lists them (`credit.round_root`)."""
    run, _ws, _got = _window(W1)
    served, per = run.served, len(SHAPES)
    for r, log in enumerate(run.round_log):
        segs = served.segments(run, r)
        assert set(segs) == {HW.table(served.program, "a", c) for c in range(r * per, (r + 1) * per)}
        for v, got in segs.items():
            reg = served.registered[v]
            assert CR.tree_root(reg.domain.domain_id, reg.leaves, {(lv, ix): h for lv, ix, h in got}) == reg.root
        assert log["a"] == {v: [[lv, ix, h.hex()] for lv, ix, h in got] for v, got in segs.items()}
        assert CR.round_root(r, bytes.fromhex(log["inner"]), segs).hex() == log["root"]
