"""`protocol_options.pouw_h1_served`: an H1 window that mixes K, committed as compute accounting's window Program's
parameters (`h1window.program` on vllm's 32 × 32 tile, one template per K), is each K's A codes and B rows as
``hm96-sha512/row/v2`` trees whose shapes, rows and roots are the ones the Program's units read (`h1window.unit_reads`),
recomputed here from the served tensors. B is each linear's weight rows once a window. One `registered.reads` and
`registered.reads_file` over the whole window resolve every unit's ports (`h1window.window_ports`), each K's own
(``a_k<K>_0`` .. ``a_k<K>_31``, ``b_k<K>_0`` .. ``b_k<K>_31``; ``a_k<K>`` → ``a_k<K>_0`` for a tile of one row,
``b_k<K>`` → ``b_k<K>_0`` for one column),
to the trees' rows, and a read opened with a row of another A block or another B row is refused; `register` registers
the window once, in one file, through `register_served` under the window's open root. A call whose codes aren't its shared
block's is refused; y, z and the scale stay under the round roots only; a second window reusing the first's key is
refused (R7-held); `calls_root` binds the trees. A stand-in tile unit of the H1 tile's parameter shape (one-bit A and B
leaves, 8K a row, its receipt a sink of the salt as `test_circuit_h1window`'s) keeps the Program small."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw.audit import hm96_salt  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind  # noqa: E402
from verity.ml.circuits.ports import In, Out, Ports  # noqa: E402
from verity.ml.circuits.refs import coll_concat  # noqa: E402
from verity.ml.circuits.types import Array, Record, Value  # noqa: E402
from verity.ml.commitments import hm96  # noqa: E402
from verity.ml.commitments.merkle import tree_levels  # noqa: E402
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest  # noqa: E402
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
V1, V16 = Value(1), Value(16)
XOR1 = PrimitiveDefinition("H1ServedTestXor1", 1, (("a", V1), ("b", V1)), V1, lambda a, b: a ^ b, register=False)
XOR16 = PrimitiveDefinition("H1ServedTestXor16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: a ^ b, register=False)
FOLD16 = PrimitiveDefinition("H1ServedTestFold16", 1, (("key", V16), ("x", V16)), V16, lambda key, x: key ^ x,
                             register=False, ports=Ports(params={"key": In(kind="coin")}, ret=Out(sink=True)))
OUT = Record(("z", V1), ("receipt", V16))


def _sig(S):
    return ((("a", Array(S.R, Array(8 * S.K, V1))), ("b", Array(S.C, Array(8 * S.K, V1))), ("salt", Array(4, V16)),
             ("position", V16), ("roots", Array(2, Array(4, V16)))), OUT)


def _body(B, S, a, b, salt, position, roots):
    return coll_concat(OUT, [B.call(XOR1, a[0][0], b[0][0]),
                             B.call(FOLD16, salt[0], B.call(XOR16, position, roots[0][0]))])


StandIn = CompositeDefinition("H1ServedTestTile", 0, ("K", "R", "C"), _sig, _body, register=False,
                              ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")},
                                          ret={"receipt": Out(sink=True)}))


def _unit(tile=TILE):
    return lambda k: bind(StandIn, K=k, R=tile[0], C=tile[1])


def _codes(rng, n, k):
    return torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8))


def _window(key: bytes, seed: int = 0, served_key: bytes | None = None, shapes=SHAPES, blocks=BLOCKS, k_input=None,
            steps=STEPS, served=1, changed=None, modules=MODULES, tile=TILE, rows=T):
    """A window: its first ``served`` steps serve 5 rows each, q and k on one input x (k on ``k_input`` when given), o
    on another, every step with the same weights (step 1's q on ``changed`` when given); the later steps are empty.
    Returns the run, each linear's weight and each served step's inputs."""
    rng = np.random.default_rng(seed)
    run = PH.H1Run("t", SALT, modules, shapes, rows, steps, key=bytes([seed + 1]) * 32,
                   served=SV.Served(tile, _unit(tile), blocks, window=served_key or key))
    ws = [_codes(rng, n, k) for n, k in shapes]
    steps_in = []
    for s in range(served):
        x = _codes(rng, SERVED, shapes[0][1])
        inputs = [x, x if k_input is None else k_input] + [_codes(rng, SERVED, k) for _n, k in shapes[2:]]
        for w, (a, wt) in enumerate(zip(inputs, ws)):
            wt = changed if (s, w) == (1, 0) and changed is not None else wt
            n = wt.shape[0]
            z = torch.from_numpy(rng.integers(-(1 << 20), 1 << 20, (SERVED, n), dtype=np.int32))
            run.record(w, a, torch.rand(SERVED) + 0.5, wt, torch.ones(n, 1), z=z, y=torch.randn(SERVED, n).bfloat16())
        steps_in.append(inputs)
    run.close()
    return run, ws, steps_in


def _expected(run, ws, steps_in, tile=TILE):
    """Each unit's (K, its A codes, its B codes), written out from the served tensors independently of the Program: call
    (s, w)'s row block ti of its T slots and its linear's weight rows [C tj, C tj + C)."""
    (r, c), m, out = tile, HW.slot(run.rows, tile), []
    for s in range(run.steps):
        for w, (n, k) in enumerate(run.shapes_):
            codes = np.zeros((m, k), np.int8)
            if s < len(steps_in):
                codes[:SERVED] = steps_in[s][w].numpy()
            out += [(k, codes[r * ti:r * ti + r], ws[w].numpy()[c * tj:c * tj + c]) for ti in range(m // r)
                    for tj in range(n // c)]
    return out


def _root(reg: G.Registered, codes: np.ndarray, key: bytes, i: int) -> str:
    """`reg`'s tree recomputed from the raw int8 bytes of `codes`: a row's bit row is its codes' memory."""
    dom, inst = reg.domain, hm96.SHA512
    bcs = [inst.commit_string(inst.default_key, sha512_row_v2_digest(row.tobytes(), 8 * len(row)), hm96_salt(key, SV.nonce(i), j))
           for j, row in enumerate(codes)]
    return tree_levels(dom, [row_leaf(dom, j, inst.tree_leaf(inst.default_key, bc), SV.V2)
                             for j, bc in enumerate(bcs)])[-1][0].hex()


def test_a_mixed_k_window_commits_the_rows_its_programs_units_read():
    """Each K's A then B, named as the Program's parameters, its rows K/2 words under the registered-value domain of the
    Program's digest; every unit's A rows (`unit_reads`) hold its call's codes (the served rows, zero past them, step 1
    all zero) and its B rows its linear's weight rows, and every root recomputes from the tensors."""
    run, ws, steps_in = _window(W1)
    served, trees = run.served, SV.trees(run)
    prog, pre = served.program, SV.prefix(W1)
    sig, digest = dict(prog.fn.signature()[0]), P.program_digest(prog)
    assert served.m == T and served.linears == [("q", 16, 64), ("k", 16, 32), ("o", 32, 32)]
    assert served.calls == [(m, s) for s in range(STEPS) for m in ("q", "k", "o")]
    assert [t.value for t in trees] == [pre + r for r in HW.roles(KS)[:4]]
    assert [(t.leaves, t.words, t.word_bits, t.row_schema) for t in trees] == [
        (sig[t.value].n, k // 2, 16, SV.V2) for t, k in zip(trees, (16, 16, 32, 32))]
    assert served.root == W1.hex()
    assert [t.domain for t in trees] == [G.domain(digest, t.value, t.words, t.leaves, window=W1.hex()).domain_id.hex()
                                         for t in trees]
    assert not {t.domain for t in trees} & {G.domain(digest, t.value, t.words, t.leaves).domain_id.hex() for t in trees}
    tp = P.evaluate(prog, served.query())
    want = _expected(run, ws, steps_in)
    assert tp.population == len(want) == STEPS * (2 * 2 + 2 + 2)
    rows = {r.value: np.asarray(r.rows, np.uint16) for r in served.registered}
    for u, (k, a, b) in enumerate(want):
        got = HW.unit_reads(prog, tp, u)
        for i in range(TILE[0]):
            v, row = got[f"a_k{k}[{i}]"]
            assert v == pre + f"a/{k}" and (rows[v][row] == SV.row_words(a)[i]).all()
        for j in range(TILE[1]):
            v, row = got[f"b_k{k}[{j}]"]
            assert v == pre + f"b/{k}" and (rows[v][row] == SV.row_words(b)[j]).all()
        assert got["position"] == (pre + "position", u) and got["z"] == (pre + "z", u)
    x0, o0 = steps_in[0][0].numpy(), steps_in[0][2].numpy()
    a16, a32 = np.zeros((STEPS * T, 16), np.int8), np.zeros((STEPS * T, 32), np.int8)
    a16[:SERVED], a32[:SERVED] = x0, o0
    for i, codes in enumerate((a16, np.concatenate([ws[0].numpy(), ws[1].numpy()]), a32, ws[2].numpy())):
        assert trees[i].root == _root(served.registered[i], codes, run._key, i)


@pytest.mark.parametrize("steps", [2, 3])
def test_b_is_each_linears_weight_once_a_window(steps):
    """b/<K> is Σn rows of K's linears whatever the window's steps, every step's units reading the same rows, and a
    second served step with the same weight commits nothing more."""
    run, _ws, _inputs = _window(W1, steps=steps, served=2)
    pre, tp = SV.prefix(W1), P.evaluate(run.served.program, run.served.query())
    assert {t.value: t.leaves for t in SV.trees(run) if "/b/" in t.value} == {pre + "b/16": 64 + 32, pre + "b/32": 32}
    per = tp.population // steps
    assert all(_b0(HW.unit_reads(run.served.program, tp, u)) == _b0(HW.unit_reads(run.served.program, tp, u % per))
               for u in range(tp.population))


def _b0(reads: dict) -> tuple[str, int]:
    """A unit's read of its first B row, under its K's template's key."""
    return next(v for key, v in reads.items() if key.startswith("b_k") and key.endswith("[0]"))


def test_one_read_resolves_every_units_ports_across_k_and_refuses_another_row():
    """One `registered.reads` over the whole mixed-K window (bit rows count K/2 words as 8K one-bit leaves) gives each unit
    its K's ``a_k<K>[i]`` and ``b_k<K>[j]`` rows, exactly those `unit_reads` gives, and none of the other K's; each K's A
    and B is one value, B once a window. `reads_file` names them as each K's own ports ``a_k<K>_0`` .. ``a_k<K>_31`` and
    ``b_k<K>_0`` .. ``b_k<K>_31`` in row/v2, each with its value's root, width and domain; each read opens its tree at its
    row, while the same position opened with a row of another A block or the next B row is refused."""
    run, _ws, _inputs = _window(W1)
    served = run.served
    prog, q = served.program, served.query()
    tp = P.evaluate(prog, q)
    want = [HW.unit_reads(prog, tp, u) for u in range(tp.population)]
    assert served.ks() == list(KS)
    got = G.reads(prog, q, served.values(), served.schemas())
    assert set(got) == set(served.ports()) == {f"{t}_k{k}[{i}]" for t in "ab" for k in KS for i in range(32)}
    for key, entry in got.items():
        assert entry["rows"] == [w[key][1] if key in w else None for w in want], key
        assert all(w[key][0] == entry["value"] for w in want if key in w), key
    assert [sum(r is not None for r in got[f"a_k{k}[0]"]["rows"]) for k in KS] == [STEPS * 6, STEPS * 2]
    by = {r.value: r for r in served.registered}
    pre = SV.prefix(W1)
    assert sorted(by) == sorted(pre + f"{t}/{k}" for t in "ab" for k in KS), "each K's A and B is one value"
    roots = {r.value: r.root.hex() for r in served.registered}
    f = G.reads_file(prog, q, served.values(), roots, ports=served.ports(), schemas=served.schemas(), window=served.root)
    assert set(f["ports"]) == {f"{t}_k{k}_{i}" for t in "ab" for k in KS for i in range(32)}
    assert {(p["row_schema"], p["word_bits"]) for p in f["ports"].values()} == {(SV.V2, 16)}
    for k in KS:
        for t in "ab":
            p, reg = f["ports"][f"{t}_k{k}_0"], by[pre + f"{t}/{k}"]
            assert (p["value"], p["root"], p["words"], p["leaves"], p["domain"]) == (
                reg.value, reg.root.hex(), k // 2, reg.leaves, reg.domain.domain_id.hex())
        for port, (key, other) in {f"a_k{k}_0": (f"a_k{k}[0]", T), f"b_k{k}_0": (f"b_k{k}[0]", 1)}.items():
            reg = by[f["ports"][port]["value"]]
            for u, row in enumerate(f["ports"][port]["reads"]):
                if row is None:
                    assert key not in want[u]
                    continue
                assert want[u][key] == (reg.value, row)
                wrong = (row + other) % reg.leaves
                assert G.verify_read(reg.domain, reg.root, row, reg.bcs[row], reg.path(row), SV.V2)
                assert not G.verify_read(reg.domain, reg.root, row, reg.bcs[wrong], reg.path(row), SV.V2)


@pytest.mark.parametrize("tile, lone", [((1, 32), "a"), ((32, 1), "b")], ids=["R=1", "C=1"])
def test_a_tile_of_one_row_or_column_maps_its_lone_read_to_port_0(tile, lone):
    """With R = 1 `registered.reads` keys a unit's A row ``a_k<K>``, not ``a_k<K>[0]``, and `window_ports` maps it to
    ``a_k<K>_0``; likewise B's with C = 1. One read over a mixed-K window still names each K's port."""
    run, _ws, _inputs = _window(W1, shapes=[(32, 16), (32, 16), (32, 32)], steps=1, tile=tile, rows=32)
    served = run.served
    ports = served.ports()
    assert all(ports[f"{lone}_k{k}"] == f"{lone}_k{k}_0" and f"{lone}_k{k}[0]" not in ports for k in KS)
    prog, q = served.program, served.query()
    got = G.reads(prog, q, served.values(), served.schemas())
    assert set(got) == set(served.ports()) and all(f"{lone}_k{k}" in got for k in KS)
    f = G.reads_file(prog, q, served.values(), {r.value: r.root.hex() for r in served.registered}, ports=served.ports(),
                     schemas=served.schemas())
    other = "b" if lone == "a" else "a"
    assert set(f["ports"]) == {f"{lone}_k{k}_0" for k in KS} | {f"{other}_k{k}_{i}" for k in KS for i in range(32)}
    reads = {k: f["ports"][f"{lone}_k{k}_0"]["reads"] for k in KS}
    if lone == "a":
        assert reads[16] == list(range(32)) * 2 + [None] * 32 and reads[32] == [None] * 64 + list(range(32))
    else:
        assert reads[16] == list(range(32)) + list(range(32, 64)) + [None] * 32 and reads[32] == [None] * 64 + list(range(32))


def test_a_k32_and_k322_window_names_every_read_its_own_port():
    """At K = 32 and 322 a port without `window_ports`' separator would make ``a_k32[20]`` and ``a_k322[0]`` one port
    ``a_k3220``, which `reads_file` would merge; the window's 4 × 32 reads are 4 × 32 ports, ``a_k32_20`` and
    ``a_k322_0`` among them."""
    prog = HW.program([("q", 32, 32), ("o", 322, 32)], [("q", 0), ("o", 0)], T, TILE, SV.prefix(W1), _unit())
    ports = SV.Served(program=prog).ports()
    assert len(ports) == len(set(ports.values())) == 4 * 32
    assert (ports["a_k32[20]"], ports["a_k322[0]"]) == ("a_k32_20", "a_k322_0")


def test_the_declared_program_is_the_served_windows_before_it_serves():
    """`declared` from the window's public shapes, modules, rows and steps alone is the Program `close` built, under the
    window's prefix; with none it names no window, so two windows' served Programs differ and their declared one doesn't."""
    run, _ws, _inputs = _window(W1)
    other, _ws, _inputs = _window(W2, seed=1)
    decl = SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, unit=_unit())
    assert P.program_digest(SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, SV.prefix(W1), _unit())) == \
        P.program_digest(run.served.program) != P.program_digest(other.served.program)
    assert P.program_digest(decl) == P.program_digest(SV.declared(SHAPES, MODULES, T, STEPS, TILE, BLOCKS, unit=_unit()))
    assert decl.fn.bindings["CALLS"] == tuple(run.served.calls) and decl.fn.bindings["LINEARS"] == tuple(run.served.linears)


def test_y_z_and_the_scale_stay_under_the_round_roots_alone():
    """The registered rows are A's codes and B's: no y, z or scale. Each call's A (with its scale), y and z tables stay
    committed, and each round's root is over them."""
    run, _ws, _inputs = _window(W1)
    assert all(r.words == k // 2 for r, k in zip(run.served.registered, (16, 16, 32, 32)))
    assert not any(t.value.endswith(("/z", "/receipt")) for t in SV.trees(run))
    assert all(set(run.tables[c.index]) == {"a", "y", "z"} and None not in run.tables[c.index].values() for c in run.calls)
    for r, log in enumerate(run.round_log):
        assert RQ.step_digest(run.salt, r, run._rows([c for c in run.calls if c.step == r])).hex() == log["calls"]


def test_calls_root_binds_the_trees():
    run, _ws, _inputs = _window(W1)
    served, got = run.served, run.calls_root()
    run.served = None
    assert got == hashlib.sha512(run.calls_root() + served.digest()).digest()
    run.served = served
    a = served.registered[0]
    served.registered[0] = G.Registered.commit(a.program, a.value, [[w ^ 1 for w in r] for r in a.rows], a.salts,
                                               schema=SV.V2, window=a.window)
    assert run.calls_root() != got


@pytest.mark.parametrize("why, kw", [
    ("aren't those of its block 0", {"k_input": torch.zeros(SERVED, 16, dtype=torch.int8)}),
    ("weight changed within the window", {"served": 2, "changed": torch.zeros(64, 16, dtype=torch.int8)}),
    ("one input block for each", {"blocks": (0, 1)}),
    ("whole u16 words", {"shapes": [(64, 15), (32, 15), (32, 32)]}),
    ("whole blocks", {"shapes": [(48, 16), (32, 16), (32, 32)]}),
], ids=["a-block-mismatch", "weight-changed", "blocks-off", "odd-k", "n-off-c"])
def test_a_window_off_the_programs_layout_commits_no_tree(why, kw):
    """k on another input than q's shared block, a weight changed between steps, blocks that aren't one per linear, an
    odd K and n off the tile's C: `close` commits nothing, and `trees` says why."""
    run, _ws, _inputs = _window(W1, **kw)
    assert run.served.registered == [] and any(why in r for r in run.served.refused)
    with pytest.raises(ValueError, match="trees were refused"):
        SV.trees(run)


def test_a_second_window_reusing_the_first_windows_key_is_refused():
    """R7-held: a later window keyed as the first commits under the first's names and domains with other roots, and
    `register_served` refuses it; keyed as its own window its names and domains are its own, and it registers."""
    first, _ws, _inputs = _window(W1)
    log = [{"registered": SV.register(first).registered}]
    again, _ws, _inputs = _window(W2, seed=1, served_key=W1)
    with pytest.raises(ValueError, match="R7-held"):
        SV.register(again, log=log)
    own, _ws, _inputs = _window(W2, seed=1)
    SV.register(own, log=log)
    mine, theirs = SV.trees(own), log[0]["registered"]
    assert not {t.value for t in mine} & {e["value"] for e in theirs}
    assert not {t.domain for t in mine} & {e["domain"] for e in theirs}


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


def test_register_registers_the_whole_window_once_in_one_file(tmp_path, monkeypatch):
    """`register` calls `register_served` once, over every K's A and B tree, with the window's ports (each K's own), and
    writes its one ``--registered`` file; ``ports`` overrides them."""
    run, _ws, _inputs = _window(W1)
    calls = []

    class Reg:
        def write(self, path):
            path.write_text("registered")

    def fake(program, query, served, log=(), ports=None, window_root=None):
        calls.append((program, query, [t.value for t in served], ports, window_root))
        return Reg()

    monkeypatch.setattr(SV, "register_served", fake)
    SV.register(run, directory=tmp_path)
    pre = SV.prefix(W1)
    assert calls == [(run.served.program, run.served.query(), [pre + r for r in HW.roles(KS)[:4]],
                      HW.window_ports(run.served.program), W1.hex())]
    assert sorted(p.name for p in tmp_path.iterdir()) == [SV.FILE] == ["pouw-h1-registered.json"]
    SV.register(run, directory=tmp_path, ports={"a_k16[0]": "a0"})
    assert calls[-1][3] == {"a_k16[0]": "a0"}


def test_the_window_registers(tmp_path):
    """`register_served` takes the window's row/v2 trees under its root: one registered entry a tree, each port of the
    window's `--registered` file in row/v2 with its tree's root and domain, and the file written as `FILE`."""
    run, _ws, _inputs = _window(W1)
    reg = SV.register(run, directory=tmp_path)
    by = {t.value: t for t in SV.trees(run)}
    assert reg.window_root == W1.hex() and [e["value"] for e in reg.registered] == list(by)
    assert set(reg.ports) == set(HW.window_ports(run.served.program).values())
    assert all((p["row_schema"], p["root"], p["domain"]) == (SV.V2, by[p["value"]].root, by[p["value"]].domain)
               for p in reg.ports.values())
    assert (tmp_path / SV.FILE).read_bytes() == reg.file_bytes()
