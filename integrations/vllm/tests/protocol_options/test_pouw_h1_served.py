"""`protocol_options.pouw_h1_served`: an H1 window that mixes K, committed as compute accounting's window Program's
parameters (`h1window.program` on vllm's 32 × 32 tile, one template per K), is each K's A codes and B rows as
``hm96-sha512/row/v2`` trees whose shapes, rows and roots are the ones the Program's units read (`h1window.unit_reads`),
recomputed here from the served tensors. B is each linear's weight rows once a window. For each K, `registered.reads` and
`registered.reads_file` resolve every unit of its template's ports (``a0`` .. ``a31``, ``b0`` .. ``b31``; ``a`` → ``a0``
for a tile of one row) to the trees' rows, and a read opened with a row of another A block or another B row is refused;
one file over both K is refused, as both templates read under the same ports. A call whose codes
aren't its shared block's is refused; y, z and the scale stay under the round roots only; a second window reusing the
first's names is refused (R7-held); `calls_root` binds the trees.

The registration through `register_served` is a strict xfail: its served trees are ``hm96-sha512/row/v1`` alone
(`served.READ`) until proofs' REGV2 follow-up. A stand-in tile unit of the H1 tile's parameter shape (one-bit A and B
leaves, 8K a row) keeps the Program small."""
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
from verity.ml.circuits.types import Array, Value  # noqa: E402
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


def _sig(S):
    return ((("a", Array(S.R, Array(8 * S.K, V1))), ("b", Array(S.C, Array(8 * S.K, V1))), ("salt", Array(4, V16)),
             ("position", V16), ("roots", Array(2, Array(4, V16)))), V1)


StandIn = CompositeDefinition("H1ServedTestTile", 0, ("K", "R", "C"), _sig,
                              lambda B, S, a, b, salt, position, roots: B.call(XOR1, a[0][0], b[0][0]), register=False)


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
    assert [t.domain for t in trees] == [G.domain(digest, t.value, t.words, t.leaves).domain_id.hex() for t in trees]
    tp = P.evaluate(prog, served.query())
    want = _expected(run, ws, steps_in)
    assert tp.population == len(want) == STEPS * (2 * 2 + 2 + 2)
    rows = {r.value: np.asarray(r.rows, np.uint16) for r in served.registered}
    for u, (k, a, b) in enumerate(want):
        got = HW.unit_reads(prog, tp, u)
        for i in range(TILE[0]):
            assert got[f"a[{i}]"][0] == pre + f"a/{k}" and (rows[pre + f"a/{k}"][got[f"a[{i}]"][1]] == SV.row_words(a)[i]).all()
        for j in range(TILE[1]):
            assert got[f"b[{j}]"][0] == pre + f"b/{k}" and (rows[pre + f"b/{k}"][got[f"b[{j}]"][1]] == SV.row_words(b)[j]).all()
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
    assert all(HW.unit_reads(run.served.program, tp, u)["b[0]"] == HW.unit_reads(run.served.program, tp, u % per)["b[0]"]
               for u in range(tp.population))


def test_registered_reads_resolve_every_units_ports_and_refuse_another_row():
    """Over one K's values, `registered.reads` (bit rows count K/2 words as 8K one-bit leaves) gives each unit of K's
    template the ``a[i]`` and ``b[j]`` rows `unit_reads` does, and the other template's units none; `reads_file` names them
    as the tile's ports ``a0`` .. ``a31`` and ``b0`` .. ``b31`` in row/v2, with the trees' roots; each read opens its tree
    at its row, while the same position opened with a row of another A block or the next B row is refused. Over both K's
    values at once, the port ``a[0]`` would read two values, and `reads` refuses it."""
    run, _ws, _inputs = _window(W1)
    served = run.served
    prog, q = served.program, served.query()
    tp = P.evaluate(prog, q)
    want = [HW.unit_reads(prog, tp, u) for u in range(tp.population)]
    with pytest.raises(ValueError, match="in one template and"):
        G.reads(prog, q, served.values(), served.schemas())
    assert served.ks() == list(KS)
    by = {r.value: r for r in served.registered}
    for k in KS:
        got = G.reads(prog, q, served.values(k), served.schemas(k))
        assert set(got) == set(SV.ports(TILE)) - {"a", "b"}
        for u in range(tp.population):
            mine = want[u]["a[0]"][0] in served.values(k)
            assert all(got[p]["rows"][u] == (want[u][p][1] if mine else None) and got[p]["value"] in served.values(k)
                       for p in got)
            assert all(got[p]["value"] == want[u][p][0] for p in got if mine)
        roots = {r.value: r.root.hex() for r in served.of(k)}
        f = G.reads_file(prog, q, served.values(k), roots, ports=SV.ports(TILE), schemas=served.schemas(k))
        assert set(f["ports"]) == {f"a{i}" for i in range(32)} | {f"b{j}" for j in range(32)}
        assert {(p["row_schema"], p["word_bits"]) for p in f["ports"].values()} == {(SV.V2, 16)}
        for port, (key, other) in {"a0": ("a[0]", T), "b0": ("b[0]", 1)}.items():
            reg = by[f["ports"][port]["value"]]
            assert f["ports"][port]["root"] == reg.root.hex()
            for u, row in enumerate(f["ports"][port]["reads"]):
                if row is None:
                    continue
                assert want[u][key] == (reg.value, row)
                wrong = (row + other) % reg.leaves
                assert G.verify_read(reg.domain, reg.root, row, reg.bcs[row], reg.path(row), SV.V2)
                assert not G.verify_read(reg.domain, reg.root, row, reg.bcs[wrong], reg.path(row), SV.V2)


def test_a_tile_of_one_row_maps_its_lone_a_read_to_port_a0():
    """With R = 1 `registered.reads` keys a unit's A row ``a``, not ``a[0]``, and `ports` maps it to ``a0``."""
    tile = (1, 32)
    assert SV.ports(tile)["a"] == SV.ports(tile)["a[0]"] == "a0" and "b" not in SV.ports(tile)
    run, _ws, _inputs = _window(W1, shapes=[(32, 16)] * 3, steps=1, tile=tile, rows=32)
    served = run.served
    prog, q = served.program, served.query()
    assert "a" in G.reads(prog, q, served.values(), served.schemas())
    f = G.reads_file(prog, q, served.values(), {r.value: r.root.hex() for r in served.registered}, ports=SV.ports(tile),
                     schemas=served.schemas())
    assert set(f["ports"]) == {"a0"} | {f"b{j}" for j in range(32)}
    assert f["ports"]["a0"]["reads"] == list(range(32)) * 2 + list(range(32, 64))


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
                                               schema=SV.V2)
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


def test_a_second_window_reusing_the_first_windows_names_is_refused():
    """R7-held: a later window keyed as the first commits under the first's names and other roots, and is refused before
    the registration; keyed as its own window its names are its own."""
    first, _ws, _inputs = _window(W1)
    log = [{"registered": first.served.entries()}]
    again, _ws, _inputs = _window(W2, seed=1, served_key=W1)
    with pytest.raises(ValueError, match="R7-held"):
        SV.held(SV.trees(again), log)
    with pytest.raises(ValueError, match="R7-held"):
        SV.register(again, log=log)
    own, _ws, _inputs = _window(W2, seed=1)
    SV.held(SV.trees(own), log)
    assert not {t.value for t in SV.trees(own)} & {e["value"] for e in log[0]["registered"]}


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


@pytest.mark.xfail(reason="REGV2 follow-up (after #2971 and #2979 land): register_served refuses hm96-sha512/row/v2 served "
                          "trees (served.READ is row/v1 alone) and doesn't pass their schemas to registered.reads",
                   raises=ValueError, strict=True)
def test_the_window_registers():
    run, _ws, _inputs = _window(W1)
    SV.register(run)
