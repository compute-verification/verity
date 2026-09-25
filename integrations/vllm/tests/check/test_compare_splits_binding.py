"""[R16 stoch/global, COORD M-0169 (4) / stoch M-0254] compare binds the fold's `Const32[S]` splits operand of every top-p TokenSelect to
the request Program's request-owned `splits` Input entry of the same sampling event; GM-01 reads S from the bound operand.

The fold (patterns.TokenSelectGumbelTopP._stage3) emits `Const32[S]` "splits" as the 6th operand of `GumbelTopPTokenSelect_v1{V}` from the
split kernels' constexpr; the request Program (vllm_meta.StochasticRequest) takes `splits: int32 [T+1]` as an INPUT after the seed and
event s reads `splits[s:s+1]`.  Neither side has S as a static.  These tests build the two instance sequences directly (no torch, no
fold) and run the real `compare.Prog` binding + `compare_steps`."""

from __future__ import annotations

from verity_vllm.check.match import program_compare as C
from verity_vllm.check.match import global_match as gm

SEL = "GumbelTopPTokenSelect_v1{V=16}"
LP, T = 4, 2          # 3 sampling events


def _const(width: int, value: int) -> str:
    return f"Const{width}[0x{value:x}]_v1"


def _prog(label: str, *, splits_form: str, S: list[int], tok=0, seed=1, splits_node=None, weights=None, read_entry=None,
          shared_const: bool = False) -> C.Prog:
    """An instance sequence of `len(S)` sampling events.  Per event: a Gemm reading the prompt/previous select + the weights, the three
    constant operands of the select (top_p, pos, temperature), then the select.  `splits_form`: "const" (the fold: Const32[S] before the
    select, 6th operand) | "input" (the Program: 6th operand reads splits[read_entry(e)] of the `splits_node` Input) | "static" (the R15
    {V,S} form: five operands)."""
    P = C.Prog.__new__(C.Prog)
    P.label, P.digest = label, f"digest-{label}"
    P.inputs = {tok: ("Input32_v1", LP), seed: ("Input64_v1", 1)}
    if splits_node is not None:
        P.inputs[splits_node] = ("Input32_v1", len(S))
    P.inputs[weights] = ("Input16_v1", 64)
    P.tok_node, P.seed_nodes, P.seed_index = tok, {seed}, {seed: 0}
    P.weight_fields = {weights: "model.lm_head.weight"}
    P.param_names, P.oracle_rows, P.n_nodes = [], None, 0
    inst, k = [], max(P.inputs) + 1
    prev_sel = None
    shared_k = None
    for e, s_val in enumerate(S):
        src = [("n", tok, 0, LP)] if prev_sel is None else [("n", prev_sel, 0, 1)]
        inst.append({"k": k, "spec": "Gemm_v1{M=1,N=16,K=8}", "form": "call", "args": [src, [("n", weights, 0, 64)]], "out": "bf16[16]"})
        gemm = k
        k += 1
        consts = []
        for w, v in ((32, 0x3f733333), (64, LP - 1 + e), (32, 0x3f4ccccd)):
            inst.append({"k": k, "spec": _const(w, v), "form": "call", "args": [], "out": f"u{w}"})
            consts.append(k)
            k += 1
        args = [[("n", gemm, 0, 16)], [("n", consts[0], 0, 1)], [("n", seed, 0, 1)], [("n", consts[1], 0, 1)], [("n", consts[2], 0, 1)]]
        spec = SEL
        if splits_form == "const":
            if shared_const and shared_k is not None:
                args.append([("n", shared_k, 0, 1)])
            else:
                inst.append({"k": k, "spec": _const(32, s_val), "form": "call", "args": [], "out": "u32"})
                args.append([("n", k, 0, 1)])
                shared_k = k
                k += 1
        elif splits_form == "input":
            ent = read_entry(e) if read_entry else e
            args.append([("n", splits_node, ent, ent + 1)])
        elif splits_form == "static":
            spec = f"GumbelTopPTokenSelect_v1{{V=16,S={s_val}}}"
        inst.append({"k": k, "spec": spec, "form": "call", "args": args, "out": "i32"})
        prev_sel = k
        k += 1
    P.inst = inst
    P.k2i = {r["k"]: i for i, r in enumerate(P.inst)}
    P._bind_splits()
    P._segment()
    return P


def test_fold_const32_splits_binds_to_the_programs_splits_input_entry_of_the_same_event():
    S = [32, 16, 8]
    F = _prog("fold", splits_form="const", S=S, weights=2)                        # [prompt, seed, weights]: the fold has no splits Input
    A = _prog("program", splits_form="input", S=S, splits_node=2, weights=3)      # [prompt, seed, splits, weights]
    # the fold's three constants are bound (removed from the sequence, values kept by select node); the Program's Input 2 is the splits
    assert F.n_splits_consts_bound == 3 and sorted(F.splits_values.values()) == sorted(S) and not F.splits_nodes
    assert A.splits_nodes == {2} and not A.splits_values and A.n_splits_consts_bound == 0
    assert len(F.inst) == len(A.inst) == 5 * 3 and len(F.steps) == len(A.steps) == 3
    for e, (i_f, i_a) in enumerate(zip((st[-1] for st in F.steps), (st[-1] for st in A.steps))):
        assert F.inst[i_f]["spec"] == A.inst[i_a]["spec"] == SEL
        assert F.canon(i_f)[C.SPLITS_OPERAND] == A.canon(i_a)[C.SPLITS_OPERAND] == [("SPLITS", e, e + 1)]
        # the constant's value is what GM-01 reads for the relation check; the static-form reader still works for R15 Programs
        assert gm.top_p_split_count(F.inst[i_f]["spec"], F, F.inst[i_f]) == S[e]
        assert gm.top_p_split_count(A.inst[i_a]["spec"], A, A.inst[i_a]) is None
    res = C.compare_steps(F, A)
    assert res["totals"]["steps_spec_equal"] == 3 and res["totals"]["operands_different"] == 0 and res["totals"]["operands_equal"] > 0
    assert res["canonical_sha256"]["equal"], res
    assert res["a"]["splits_bound"]["selects"] == 3 and res["a"]["splits_bound"]["values"] == [8, 16, 32]
    assert res["b"]["splits_bound"]["nodes"] == [2]
    # the values are NOT compared here: another launch geometry on the fold is the same STRUCTURE (GM-01 G6 checks S_t per engine step)
    F2 = _prog("fold2", splits_form="const", S=[4, 4, 4], weights=2)
    assert C.compare_steps(F2, A)["canonical_sha256"]["equal"]
    assert sorted(F2.splits_values.values()) == [4, 4, 4]


def test_a_program_reading_the_wrong_splits_entry_or_an_unbound_fold_constant_is_a_named_difference():
    S = [32, 16, 8]
    F = _prog("fold", splits_form="const", S=S, weights=2)
    # (a) event e reads splits[e+1]: the operand differs at every select (the Program's schedule is shifted against the fold's launches)
    A_shift = _prog("program-shifted", splits_form="input", S=S, splits_node=2, weights=3, read_entry=lambda e: (e + 1) % 3)
    res = C.compare_steps(F, A_shift)
    assert not res["canonical_sha256"]["equal"] and res["totals"]["operands_different"] == 3
    assert all(st["wiring"]["different_by_spec_arg"] == {f"{SEL}#arg{C.SPLITS_OPERAND}": 1} for st in res["steps"]), res["steps"]
    # (b) one Const32 read by every select (shared, readers 3): NOT the fold's per-select splits constant -> not bound, stays an instance,
    # and the spec sequence differs from the Program's (one instance more at events 0; the operand is an N-ref, not SPLITS)
    F_shared = _prog("fold-shared", splits_form="const", S=[16, 16, 16], weights=2, shared_const=True)
    assert F_shared.n_splits_consts_bound == 0 and not F_shared.splits_values
    A = _prog("program", splits_form="input", S=S, splits_node=2, weights=3)
    res = C.compare_steps(F_shared, A)
    assert not res["canonical_sha256"]["equal"] and res["totals"]["steps_spec_equal"] < 3
    # (c) the R15 static form {V,S} against the operand-form Program: different specs, no binding on either side, named
    F_static = _prog("fold-static", splits_form="static", S=S, weights=2)
    assert F_static.n_splits_consts_bound == 0 and gm.top_p_split_count(F_static.inst[F_static.steps[0][-1]]["spec"], F_static,
                                                                          F_static.inst[F_static.steps[0][-1]]) == 32
    res = C.compare_steps(F_static, A)
    assert not res["canonical_sha256"]["equal"] and res["totals"]["steps_spec_equal"] == 0


def test_a_fold_without_top_p_selects_binds_nothing():
    """Greedy folds (TokenSelect_v1, five-operand R15 stochastic forms) are untouched: no instance removed, no splits node."""
    P = _prog("greedy-like", splits_form="static", S=[1, 1, 1], weights=2)
    assert P.n_splits_consts_bound == 0 and not P.splits_values and not P.splits_nodes and len(P.inst) == 5 * 3
    # the bound fold keeps every other instance and its node ids (k2i is rebuilt over the reduced sequence)
    P2 = _prog("fold", splits_form="const", S=[8, 8, 8], weights=2)
    assert P2.n_splits_consts_bound == 3 and len(P2.inst) == 5 * 3 and all(P2.k2i[r["k"]] == i for i, r in enumerate(P2.inst))
    assert set(P2.splits_values) == {P2.inst[st[-1]]["k"] for st in P2.steps}   # keyed by the select's node id
