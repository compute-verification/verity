"""`protocol_options.pouw_h1_served`: an H1 window's A codes and B rows, committed as the parameters ``a`` and ``b`` of
compute accounting's window Program (`h1window.program`), are two ``hm96-sha512/row/v2`` trees whose shapes, rows and
roots are the ones the Program's units read (`h1window.unit_reads`), recomputed here from the served tensors; q and k
share an A block and a call whose codes aren't its block's is refused; y and z stay under the round roots only; a second
window reusing the first's names is refused (R7-held); `calls_root` binds both trees.

The registration itself (`register_served`) is a strict xfail: it reads ``hm96-sha512/row/v1`` rows alone (REGPORT),
and its reads can't express the tile's yet (REGV2). A stand-in tile of the H1 tile's parameter shape (`h1tile`'s one-bit A
and B leaves) keeps the Program small."""
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

K, N, T, STEPS = 16, 16, 32, 2
MODULES = {0: "q", 1: "k", 2: "o"}
SHAPES = [(N, K)] * len(MODULES)
#: q and k read one input, o another
BLOCKS = (0, 0, 1)
CALLS = STEPS * len(SHAPES)
R, C = 4, 8
SALT = hashlib.sha512(b"salt").digest()
W1, W2 = hashlib.sha512(b"window 1").digest(), hashlib.sha512(b"window 2").digest()
V1, V16 = Value(1), Value(16)
XOR1 = PrimitiveDefinition("H1ServedTestXor1", 1, (("a", V1), ("b", V1)), V1, lambda a, b: a ^ b, register=False)


def _tile(k: int):
    """The H1 tile's parameters at contraction k: R A rows and C B rows of 8k one-bit leaves, the salt, position, roots."""
    def sig(S):
        return ((("a", Array(R, Array(8 * k, V1))), ("b", Array(C, Array(8 * k, V1))), ("salt", Array(4, V16)),
                 ("position", V16), ("roots", Array(2, Array(4, V16)))), V1)

    return bind(CompositeDefinition(f"H1ServedTestTile{k}", 0, (), sig,
                                    lambda B, S, a, b, salt, position, roots: B.call(XOR1, a[0][0], b[0][0]), register=False))


def _weights(rng, shapes=SHAPES):
    return [torch.from_numpy(rng.integers(-127, 128, (n, k), dtype=np.int8)) for n, k in shapes]


def _window(key: bytes, seed: int = 0, served_key: bytes | None = None, shapes=SHAPES, blocks=BLOCKS, k_input=None):
    """A served window: step 0 serves 5 rows, q and k on one input x (k on ``k_input`` when given), o on another; step 1
    is empty. Returns the run, each linear's weight and each step-0 call's A."""
    rng = np.random.default_rng(seed)
    run = PH.H1Run("t", SALT, MODULES, shapes, T, STEPS, key=bytes([seed + 1]) * 32,
                   served=SV.Served(_tile, blocks, window=served_key or key))
    ws = _weights(rng, shapes)
    x = torch.from_numpy(rng.integers(-127, 128, (5, shapes[0][1]), dtype=np.int8))
    o = torch.from_numpy(rng.integers(-127, 128, (5, shapes[2][1]), dtype=np.int8))
    inputs = [x, x if k_input is None else k_input, o]
    for w, (a, wt) in enumerate(zip(inputs, ws)):
        z = torch.from_numpy(rng.integers(-(1 << 20), 1 << 20, (5, wt.shape[0]), dtype=np.int32))
        run.record(w, a, torch.rand(5) + 0.5, wt, torch.ones(wt.shape[0], 1), z=z, y=torch.randn(5, wt.shape[0]).bfloat16())
    run.close()
    return run, ws, inputs


def _root(digest: str, value: str, codes: np.ndarray, key: bytes, nonce: int) -> str:
    """The tree of `codes`' rows, recomputed from the raw int8 bytes: a row's bit row is its codes' memory."""
    dom = G.domain(digest, value, codes.shape[1] // 2, len(codes))
    inst = hm96.SHA512
    bcs = [inst.commit_string(inst.default_key, sha512_row_v2_digest(row.tobytes(), 8 * len(row)), hm96_salt(key, nonce, i))
           for i, row in enumerate(codes)]
    return tree_levels(dom, [row_leaf(dom, i, inst.tree_leaf(inst.default_key, bc), SV.V2)
                             for i, bc in enumerate(bcs)])[-1][0].hex()


def test_the_trees_are_the_rows_the_window_programs_units_read():
    """Two trees, A then B, named as the Program's ``a`` and ``b``, of its parameters' rows, K/2 words each under the
    registered-value domain of the Program's digest; each unit's A and B rows (`unit_reads`) hold its call's codes
    (the served rows, zero past them, step 1 all zero) and its call's weight rows, and both roots recompute."""
    run, ws, inputs = _window(W1)
    served, trees = run.served, SV.trees(run)
    prog = served.program
    va, vb = SV.names(W1)[:2]
    sig, digest = dict(prog.fn.signature()[0]), P.program_digest(prog)
    assert served.m == HW.slot(T, R) == T
    assert served.calls == [(f"s{s}/{m}", N, 2 * s + b) for s in range(STEPS) for m, b in zip(MODULES.values(), BLOCKS)]
    assert [(t.value, t.leaves, t.words, t.word_bits, t.row_schema) for t in trees] == [
        (va, sig[va].n, K // 2, 16, SV.V2), (vb, sig[vb].n, K // 2, 16, SV.V2)]
    assert (sig[va].n, sig[vb].n) == (2 * STEPS * T, CALLS * N)
    assert [t.domain for t in trees] == [G.domain(digest, t.value, K // 2, t.leaves).domain_id.hex() for t in trees]
    tp = P.evaluate(prog, served.query())
    per = (T // R) * (N // C)
    assert tp.population == CALLS * per
    a_words, b_words = (t.words for t in served.trees)
    for u in range(tp.population):
        c, ti, tj = u // per, u % per // (N // C), u % (N // C)
        s, w = divmod(c, len(SHAPES))
        codes = np.zeros((T, K), np.int8)
        if s == 0:
            codes[:5] = inputs[w].numpy()
        got = HW.unit_reads(prog, tp, u, outputs=SV.outputs(W1))
        for i in range(R):
            param, row = got[f"a[{i}]"]
            assert param == va and (a_words[row] == SV.row_words(codes[R * ti + i:R * ti + i + 1])[0]).all()
        for j in range(C):
            param, row = got[f"b[{j}]"]
            assert param == vb and (b_words[row] == SV.row_words(ws[w].numpy()[C * tj + j:C * tj + j + 1])[0]).all()
        assert got["z"] == (SV.outputs(W1)[0], u) and got["position"] == (SV.names(W1)[3], u)
    a = np.zeros((2 * STEPS * T, K), np.int8)
    a[:5], a[T:T + 5] = inputs[0].numpy(), inputs[2].numpy()
    b = np.concatenate([wt.numpy() for _s in range(STEPS) for wt in ws])
    assert trees[0].root == _root(digest, va, a, run._key, SV.A_NONCE)
    assert trees[1].root == _root(digest, vb, b, run._key, SV.B_NONCE)


def test_y_z_and_the_scale_stay_under_the_round_roots_alone():
    """The registered rows are A's codes and B's: no y, z or scale. Each call's A (with its scale), y and z tables stay
    committed, and each round's root is over them."""
    run, _ws, _inputs = _window(W1)
    assert [t.value for t in SV.trees(run)] == list(SV.names(W1)[:2])
    assert all(t.words.shape[1] == K // 2 for t in run.served.trees)
    assert all(set(run.tables[c.index]) == {"a", "y", "z"} and None not in run.tables[c.index].values() for c in run.calls)
    for r, log in enumerate(run.round_log):
        assert RQ.step_digest(run.salt, r, run._rows([c for c in run.calls if c.step == r])).hex() == log["calls"]


def test_calls_root_binds_the_trees():
    run, _ws, _inputs = _window(W1)
    served, got = run.served, run.calls_root()
    run.served = None
    assert got == hashlib.sha512(run.calls_root() + served.digest()).digest()
    run.served = served
    a = served.trees[0]
    served.trees[0] = SV.Tree.commit(P.program_digest(served.program), a.value, a.words ^ 1, a.salts)
    assert run.calls_root() != got


@pytest.mark.parametrize("why, kw", [
    ("aren't those of its block 0", {"k_input": torch.zeros(5, K, dtype=torch.int8)}),
    ("one K", {"shapes": [(N, K), (N, K + 2), (N, K)]}),
    ("no input block", {"blocks": (0, 2, 2)}),
    ("whole blocks of", {"shapes": [(12, K)] * 3}),
], ids=["a-block-mismatch", "two-k", "blocks-off", "n-off-c"])
def test_a_window_off_the_programs_layout_commits_no_tree(why, kw):
    """k on another input than q's shared block, calls of two K, blocks that skip one, and n off the tile's C: `close`
    commits nothing, and `trees` says why."""
    run, _ws, _inputs = _window(W1, **kw)
    assert run.served.trees == [] and any(why in r for r in run.served.refused)
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
    key = SV.Served(_tile).key(commits)
    assert key == PS.window_open_root(bytes.fromhex("11" * 64), bytes.fromhex("22" * 64), "s-1")
    with pytest.raises(ValueError, match="the window has no key"):
        SV.Served(_tile).key({"weights_root": "11" * 64})


def test_a_window_whose_rows_arent_registered_values_is_refused():
    run = PH.H1Run("t", SALT, MODULES, SHAPES, T, STEPS)
    run.close()
    with pytest.raises(ValueError, match="aren't registered values"):
        SV.trees(run)


@pytest.mark.xfail(reason="REGPORT and REGV2: register_served reads hm96-sha512/row/v1 rows alone (served.READ), and "
                          "registered.reads can't express the tile's reads: a template parameter of R or C rows, not one, "
                          "and one-bit leaves packed K/2 words to a row (shape and width count them as 16-bit leaves)",
                   raises=ValueError, strict=True)
def test_the_window_registers():
    run, _ws, _inputs = _window(W1)
    SV.register(run, ports={f"a[{i}]": f"a{i}" for i in range(R)} | {f"b[{j}]": f"b{j}" for j in range(C)})
