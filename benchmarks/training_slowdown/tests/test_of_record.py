import hashlib
import json
import math

import pytest
from synthetic import SHAPE, tiny_registry, write_config

from training_slowdown import as_math, codecs, config, library, of_record, starts

CTX, D = SHAPE["ctx"], SHAPE["d_model"]
X, S, G = [1, CTX, D], [1, CTX, 1], [D]
HEADS, HEAD = SHAPE["n_head"], D // SHAPE["n_head"]
#: CUDA's fused attention kernel, whose backward's name adds `_backward`
FLASH = "_scaled_dot_product_flash_attention"


def log(*entries) -> dict:
    """A step's log as `per_row` gives it, from `(phase, kind, op, operand shapes, count, MACs)`."""
    return {(phase, kind, op, json.dumps(shapes)): [count, macs] for phase, kind, op, shapes, count, macs in entries}


def step(log_: dict, updated=((D, 0, 16),)) -> dict:
    return {"rows": 1, "updated": sorted(updated), "log": log_}


#: products both sides have: a weight's product and its gradient's
BOTH = (("fwd", "updated", "mm", [[CTX, D], [D, D]], 4, 4 * CTX * D * D),
        ("bwd", "grad", "mm", [[D, CTX], [CTX, D]], 4, 4 * CTX * D * D))


def test_a_steps_rows_a_forward_are_divided_out_of_its_token_shapes_and_their_macs():
    macs = 3 * CTX * D * D
    struct = {"updated": [[D * D, D, macs, 3 * CTX, 16], [D, 0, 3 * CTX * D, 3 * CTX, 32]], "log": [
        ["fwd", "updated", "mm", json.dumps([[3 * CTX, D], [D, D]]), 2, 2 * macs],
        ["fwd", "act", "mul", json.dumps([[3, CTX, D], [3, CTX, 1]]), 5, 5 * 3 * CTX * D],
        ["other", "param", "_foreach_mul_", json.dumps([[D, D], [D], [3, D]]), 1, 2 * D * D + D],
        ["other", "param", "mm", json.dumps([[D, 3], [3, D]]), 7, 7 * 3 * D * D]]}
    got = of_record.per_row(struct, SHAPE)
    assert got["rows"] == 3 and got["updated"] == [(D, 0, 32), (D * D, D, 16)]
    assert got["log"] == log(("fwd", "updated", "mm", [[CTX, D], [D, D]], 2, 2 * CTX * D * D),
                             ("fwd", "act", "mul", [X, S], 5, 5 * CTX * D),
                             ("other", "param", "_foreach_mul_", [[3, D], [D], [D, D]], 1, 2 * D * D + D),
                             ("other", "param", "mm", [[D, 3], [3, D]], 7, 7 * 3 * D * D))
    with pytest.raises(ValueError, match="not one whole number of rows of 1024"):
        of_record.per_row(struct | {"updated": [*struct["updated"], [D, 0, CTX * D, CTX, 16]]}, SHAPE)


def attention(rows: int, calls: int) -> tuple:
    """Attention as the counter writes a fused kernel at `rows` rows a forward, `calls` calls a step, each with a
    backward call that asks for q's, k's and v's gradients: a call's two scalings and two products, and a backward
    call's four products and two scalings."""
    n, each, numel = rows * HEADS, calls * rows * HEADS * CTX * CTX * HEAD, calls * rows * HEADS * CTX * HEAD
    qk, pv, dq = [[n, CTX, HEAD], [n, HEAD, CTX]], [[n, CTX, CTX], [n, CTX, HEAD]], [[n, HEAD, CTX], [n, CTX, CTX]]
    q, k = [[rows, HEADS, CTX, HEAD]], [[rows, HEADS, HEAD, CTX]]
    return (("fwd", "act", "bmm", qk, calls, each), ("fwd", "act", "bmm", pv, calls, each),
            ("fwd", "theta", "mul", q, calls, numel), ("fwd", "theta", "mul", k, calls, numel),
            ("bwd", "grad", "bmm", pv, 2 * calls, 2 * each), ("bwd", "grad", "bmm", qk, calls, each),
            ("bwd", "grad", "bmm", dq, calls, each),
            ("bwd", "grad", "mul", q, calls, numel), ("bwd", "grad", "mul", k, calls, numel))


def fused(rows: int, calls: int, backward: int | None = None) -> list:
    """Attention as a record holds it that was written before the counter wrote the kernel as the math kernel's ops:
    one row a kernel, CUDA's, at torch's formula, the backward's with its trailing operands, `backward` calls a
    step (as many as the forward's unless given)."""
    q, each = [rows, HEADS, CTX, HEAD], rows * HEADS * CTX * CTX * HEAD
    back = calls if backward is None else backward
    return [["fwd", "act", FLASH, json.dumps([q] * 3), calls, 2 * calls * each]] + [
        ["bwd", "grad", FLASH + "_backward", json.dumps([q] * 5 + [q[:3], [2], []]), back, 5 * back * each]] * bool(back)


def a_step(rows: int, *entries) -> dict:
    """A step structure as a record stores it, at `rows` rows a forward: a weight's product and `entries`."""
    return {"updated": [[D * D, D, rows * CTX * D * D, rows * CTX, 16]], "log": [
        ["fwd", "updated", "mm", json.dumps([[rows * CTX, D], [D, D]]), 4, 4 * rows * CTX * D * D],
        *([*e[:3], json.dumps(e[3]), *e[4:]] if isinstance(e, tuple) else e for e in entries)]}


MM = ("fwd", "updated", "mm", [[CTX, D], [D, D]], 4, 4 * CTX * D * D)


def test_the_rows_are_divided_out_of_attentions_products_which_lead_with_rows_times_heads():
    """At two rows a forward: the kernel's products, batched over rows x heads, become the heads' and their MACs a
    row's. A `bmm` that leads with rows x heads over anything but contexts and a head's width, and a product that is
    no `bmm`, keep that dimension."""
    other = ("fwd", "act", "bmm", [[2 * HEADS, CTX, D], [2 * HEADS, D, CTX]], 1, 2 * HEADS * CTX * D * CTX)
    weight = ("bwd", "grad", "mm", [[2 * HEADS, 2 * CTX], [2 * CTX, D]], 1, 2 * HEADS * 2 * CTX * D)
    got = of_record.per_row(a_step(2, *attention(2, calls=3), other, weight), SHAPE)
    assert got["rows"] == 2 and got["log"] == log(MM, *attention(1, calls=3), other, (
        "bwd", "grad", "mm", [[2 * HEADS, CTX], [CTX, D]], 1, HEADS * 2 * CTX * D))
    # one row a forward, as a config is counted: nothing to divide
    assert of_record.per_row(a_step(1, *attention(1, calls=3)), SHAPE)["log"] == log(MM, *attention(1, calls=3))


def test_a_records_fused_attention_is_compared_as_the_counter_writes_it_under_an_ask_that_is_read_or_stated():
    """A record's fused rows at two rows a forward, three calls a step, beside a config's step as the counter writes
    it, whose every backward call asks for q's, k's and v's. The ask is the trace's where the record holds one, else
    the stated one; neither is a refusal in `as_math`'s words; a stated ask that is wrong is a difference."""
    mine = step(log(MM, *attention(1, calls=3)), ((D * D, D, 16),))
    struct, at = a_step(2, *fused(2, calls=3)), f"[{2 * HEADS}, {CTX}, {HEAD}]"

    def compared(rec: dict, said: str | None) -> tuple[dict, str, list]:
        rewritten, says, unread = of_record.as_counted(struct, rec, said)
        assert rewritten["updated"] == struct["updated"]
        return of_record.compare(mine, of_record.per_row(rewritten, SHAPE), SHAPE), says, unread

    with pytest.raises(ValueError, match=r"holds no attention backward at \[4, 1024, 8\] .*: say --partial none"):
        of_record.as_counted(struct, {}, None)
    got, says, unread = compared({}, "none")
    assert got["differences"] == [] and got["expected"] == [] and got["products"] == 10 and unread == []
    assert f"`none` at {at}, of_record.toml's word (`partial`) and not the record's, which holds no trace" in says
    got, says, unread = compared({}, "q:1")
    each = HEADS * CTX * CTX * HEAD  # a product's MACs at one row a forward
    assert says.count("`q:1`") == 1 and unread == [] and got["differences"] == [
        f"log, unequal: bwd/grad bmm [[2,1024,1024], [2,1024,8]] x6, {6 * each:,} MACs in the config's step, x5, "
        f"{5 * each:,} MACs in the record's",
        f"log, unequal: bwd/grad bmm [[2,8,1024], [2,1024,1024]] x3, {3 * each:,} MACs in the config's step, x2, "
        f"{2 * each:,} MACs in the record's",
        "log, unequal: bwd/grad mul [[1,2,8,1024]] x3, 49,152 MACs in the config's step, x2, 32,768 MACs in the "
        "record's"]

    def product(m: int, k: int, n: int, count: int) -> dict:
        return {"op": "bmm", "phase": "bwd", "count": count, "dims": {"batch": 2 * HEADS, "m": m, "k": k, "n": n}}

    trace = {"trace": {"rows": [product(CTX, HEAD, CTX, 3), product(CTX, CTX, HEAD, 6), product(HEAD, CTX, CTX, 3)]}}
    got, says, unread = compared(trace, None)
    assert got["differences"] == [] and unread == [] and f"`none` at {at}, read from its trace" in says
    with pytest.raises(ValueError, match="trace says --partial none at .* and the command says --partial q:1"):
        of_record.as_counted(struct, trace, "q:1")
    with pytest.raises(ValueError, match="--partial 1: not `none` nor SUBSET:CALLS"):
        of_record.as_counted(struct, {}, 1)
    # nothing to read an ask for: a record the counter wrote, and a step whose attention has no backward
    plain = a_step(2, *attention(2, calls=3))
    assert of_record.as_counted(plain, {}, None) == (plain, None, [])
    assert of_record.as_counted(plain, {}, "none") == (plain, None, [
        'of_record.toml states `partial = "none"` and the record\'s step has no fused attention backward row to read '
        'it for'])
    forward = a_step(2, *fused(2, calls=3, backward=0))
    rewritten, says, unread = of_record.as_counted(forward, {}, None)
    assert says.endswith(": it has no backward") and unread == []
    assert of_record.per_row(rewritten, SHAPE)["log"] == log(MM, *attention(1, calls=3)[:4])
    assert of_record.RULES == (of_record.norm, of_record.loss, of_record.average)


def norms(updated: int, frozen: int, scales: int) -> tuple:
    """The norm's products on a CPU: `updated` gains' multiplies, `frozen` of θ's gains, `scales` normalisations and
    a backward through each."""
    n = CTX * D
    return (("fwd", "updated", "mul", [X, G], updated, updated * n),
            ("fwd", "theta", "mul", [X, G], frozen, frozen * n), ("fwd", "act", "mul", [X, S], scales, scales * n),
            ("bwd", "grad", "div", [X], scales, scales * n), ("bwd", "grad", "mul", [X, X], 2 * scales, 2 * scales * n),
            ("bwd", "grad", "mul", [S, S], scales, scales * CTX))


def test_the_cpus_norm_is_expected_for_cudas_fused_kernel_with_one_normalisation_a_gain():
    n = CTX * D
    cuda = log(*BOTH, ("fwd", "updated", "_fused_rms_norm", [X, G], 3, 3 * n))
    got = of_record.compare(step(log(*BOTH, *norms(3, 2, 5))), step(cuda), SHAPE)
    assert got["differences"] == []
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [
        ("norm", 3 * n, 3 * n + 2 * n + 5 * n + 5 * n + 10 * n + 5 * CTX)]
    assert "3 multiplies by an updated gain" in got["expected"][0]["what"]
    # a normalisation with no gain's multiply, another count of updated gains, a product of the backward's the
    # record has too: none is the norm's exact case, so each side's entries are differences
    for cpu, record in ((norms(3, 2, 6), cuda), (norms(4, 1, 5), cuda),
                        (norms(3, 2, 5), cuda | log(("bwd", "grad", "mul", [X, X], 1, n)))):
        got = of_record.compare(step(log(*BOTH, *cpu)), step(record), SHAPE)
        assert got["expected"] == [] and len(got["differences"]) >= 7
        assert any("fwd/updated _fused_rms_norm" in d for d in got["differences"])
    # a product over another shape is no part of the norm
    wide = log(*BOTH, *norms(3, 2, 5), ("fwd", "act", "mul", [X, X], 1, n))
    got = of_record.compare(step(wide), step(cuda), SHAPE)
    assert [e["rule"] for e in got["expected"]] == ["norm"]
    assert got["differences"] == ["log, the config's step only: fwd/act mul [[1,1024,16] x2] x1, 16,384 MACs"]


def test_the_cpus_norm_is_expected_where_the_step_trains_no_gain_and_the_record_has_nothing_of_it():
    """A step that trains no gain: CUDA's counter is shown nothing of the fused kernel, so the record has no entry of
    the norm, and the CPU's side has every gain's multiply as θ's. Expected, with one normalisation a gain; not where
    a normalisation has no gain's multiply, where the record has one of the backward's products too, or where the
    config multiplies by an updated gain the record never counted."""
    n = CTX * D
    frozen = norms(0, 5, 5)[1:]
    got = of_record.compare(step(log(*BOTH, *frozen)), step(log(*BOTH)), SHAPE)
    assert got["differences"] == [] and got["products"] == len(BOTH)
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [
        ("norm", 0, 5 * n + 5 * n + 5 * n + 10 * n + 5 * CTX)]
    assert "no gain is updated" in got["expected"][0]["what"] and "5 by θ's gains" in got["expected"][0]["what"]
    for cpu, record in ((norms(0, 5, 6)[1:], log(*BOTH)), (norms(1, 4, 5), log(*BOTH)),
                        (frozen, log(*BOTH, ("bwd", "grad", "mul", [X, X], 1, n)))):
        got = of_record.compare(step(log(*BOTH, *cpu)), step(record), SHAPE)
        assert got["expected"] == [] and len(got["differences"]) >= 5
    # a step with no norm at all on either side is equal, with no rule fired
    got = of_record.compare(step(log(*BOTH)), step(log(*BOTH)), SHAPE)
    assert got["expected"] == [] and got["differences"] == []


def test_a_plans_codecs_must_each_be_of_the_records_kind_of_encoding(monkeypatch):
    """A record's format may have several codec kinds (here a made-up second one for `quant`). A plan of either, or
    of both, is that kind; one that keeps a tensor whole is not, and neither is any plan beside a format this table
    doesn't know. A program's own model's tensors whole (`model`) are the whole tensors a plan of `identity` keeps."""
    assert of_record.e_kind(["scaledquant"], "quant") == [] == of_record.e_kind(["identity"], "full")
    assert of_record.e_kind(["identity"], "model") == [] and of_record.e_kind(["scaledquant"], "model") != []
    monkeypatch.setitem(of_record.E_KINDS, "quant", ("scaledquant", "coded"))
    for kinds in (["scaledquant"], ["coded"], ["scaledquant", "coded"]):
        assert of_record.e_kind(kinds, "quant") == []
    assert of_record.e_kind(["identity", "coded"], "quant") == [
        "E's kind: the config's plan is identity, coded, the record's format quant (scaledquant or coded)"]
    assert of_record.e_kind(["coded"], "full") == [
        "E's kind: the config's plan is coded, the record's format full (identity)"]
    assert of_record.e_kind(["identity"], "codes") == [
        "E's kind: the config's plan is identity, the record's format codes (none known here)"]
    assert of_record.e_kind([], "full") == ["E's kind: the config's plan is empty, the record's format full (identity)"]
    # the sizes are printed and not compared
    assert of_record.e_sizes(["scaledquant"], 1200, {"format": "quant", "e_bytes": 1000}, 30) == (
        "scaledquant, 1,200 B after 2 steps; the record's quant, 1,000 B after 30")


def test_sparse_ternary_codes_are_a_records_quant_and_their_packing_is_named():
    """The codes a run trained are deltas in codes with a scale a group, a record's `quant`, so a plan of the
    library's sparse ternary codec is of that kind, alone or beside a rounding of the trained delta. The codec packs
    them another way than the records' program did, which the E line says beside the two sizes; every codec the
    tables name is one the library has."""
    kind = codecs.SparseTernary(0.5, 0).kind
    assert of_record.e_kind([kind], "quant") == [] == of_record.e_kind(sorted(["scaledquant", kind]), "quant")
    assert of_record.e_kind([kind], "full") == [
        "E's kind: the config's plan is sparseternary, the record's format full (identity)"]
    assert of_record.e_sizes([kind], 1200, {"format": "quant", "e_bytes": 1000}, 30) == (
        "sparseternary, 1,200 B after 2 steps; the record's quant, 1,000 B after 30; expected to differ in size by "
        "the packing too: sparseternary is one xz stream a matrix at one setting, where the record's program kept "
        "the smallest of several settings' streams of all its matrices")
    library_kinds = {c.kind for c in vars(codecs).values() if isinstance(c, type) and issubclass(c, codecs.Codec)}
    assert {k for kinds in of_record.E_KINDS.values() for k in kinds} | set(of_record.E_PACKING) <= library_kinds


def losses(n: int) -> tuple[tuple, tuple]:
    """The loss's products `n` times a step: as a program arranges it that means each row, then as a route does."""
    return ((("bwd", "grad", "div", [[1, CTX]], n, n * CTX), ("fwd", "theta", "div", [[]], n, n),
             ("bwd", "grad", "div", [[]], n, n)),
            (("fwd", "theta", "mul", [[]], n, n), ("bwd", "grad", "mul", [[]], n, n)))


def test_the_routes_loss_is_expected_for_a_programs_mean_of_each_row_where_no_row_is_dropped():
    program, route = losses(3)

    def compared(cpu=route, record=program, args=("--batch", "96")):
        return of_record.compare(step(log(*BOTH, *cpu)), step(log(*BOTH, *record)), SHAPE, args=args)

    got = compared()
    assert got["differences"] == []
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [("loss", 3 * CTX + 6, 6)]
    assert "3 times a step" in got["expected"][0]["what"] and "state no --trunc" in got["expected"][0]["what"]
    # no truncation stated as 0 is none; a record that keeps no arguments is not checked, and the output says so
    assert [e["rule"] for e in compared(args=("--trunc", "0.0"))["expected"]] == ["loss"]
    assert "keeps no arguments of its program, so that was not checked" in compared(args=())["expected"][0]["what"]
    # a run that drops rows, or whose argument states no number: its loss is another one
    for args in (("--trunc", "0.1"), ("--batch", "96", "--trunc=0.1"), ("--trunc",), ("--trunc", "nan"),
                 ("--trunc", "some")):
        got = compared(args=args)
        assert got["expected"] == [] and len(got["differences"]) == 5
    # another count on one side, the route's product in the record too, a line missing: none is the exact case
    for cpu, record in ((losses(2)[1], program), (route, (*program, route[0])), (route, program[:2]),
                        (route[:1], program), ((*route, program[2]), program)):
        got = compared(cpu, record)
        assert got["expected"] == [] and got["differences"]


def averaged(tensors: list, times: int = 1) -> tuple:
    """Route `ema`'s average: a `lerp_` of each of `tensors` with one of its shape, `times` each."""
    return tuple(("other", "param", "lerp_", [t, t], times * tensors.count(t), times * tensors.count(t) * math.prod(t))
                 for t in sorted({tuple(t) for t in tensors}) for t in [list(t)])


def test_the_routes_average_of_its_trained_tensors_is_expected_for_a_programs_of_every_tensor():
    trained, frozen = [[D], [D], [D, D]], [[259, D]]
    macs = 2 * D + D * D
    program = ("other", "param", "_foreach_lerp_", sorted((trained + frozen) * 2), 1, macs + 259 * D)
    states = ("--batch", "64", "--ema-frac", "0.1")

    def compared(cpu, record, args=states):
        return of_record.compare(step(log(*BOTH, *cpu)), step(log(*BOTH, record)), SHAPE, args=args)

    got = compared(averaged(trained), program)
    assert got["differences"] == []
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [
        ("average", macs + 259 * D, macs)]
    what = got["expected"][0]["what"]
    assert "3 of them" in what and f"1 more that the route does not train ({259 * D:,} MACs a step)" in what
    # a route that trains every tensor averages the program's own
    got = compared(averaged(trained + frozen), program)
    assert got["differences"] == [] and got["expected"][0]["record_macs"] == got["expected"][0]["config_macs"]
    assert got["expected"][0]["what"].endswith("the same 4")
    assert [e["rule"] for e in compared(averaged(trained), program, ("--ema-frac=1e-1",))["expected"]] == ["average"]
    # not its exact case: a program that states no average, none that is kept, or no number for it; a tensor the
    # program's op lacks; a `lerp_` twice a tensor; another product of an unfused AdamW step on the config's side; the
    # op twice a step, or priced otherwise
    for cpu, record, args in (
            (averaged(trained), program, ()), (averaged(trained), program, ("--ema-frac", "0.0")),
            (averaged(trained), program, ("--ema-frac", "inf")), (averaged(trained), program, ("--ema-frac",)),
            (averaged([*trained, [2 * D]]), program, states), (averaged(trained, 2), program, states),
            ((*averaged(trained), ("other", "param", "addcmul_", [[D]] * 3, 2, 2 * D)), program, states),
            (averaged(trained), (*program[:4], 2, 2 * program[5]), states),
            (averaged(trained), (*program[:5], program[5] + 1), states)):
        got = compared(cpu, record, args)
        assert got["expected"] == [] and any("_foreach_lerp_" in d for d in got["differences"])


def test_a_use_of_an_updated_weight_is_the_records_by_its_bits_too():
    """A route casts to bf16 on a CPU as on CUDA, so a use that reads its weight at other bits than the record's is
    a difference, 32 for the record's 16 as much as 16 for its 32."""
    record = [(D * D, D, 16), (D * D, D, 16), (D, 0, 32)]
    same = log(*BOTH)
    got = of_record.compare(step(same, record), step(same, record), SHAPE)
    assert got["differences"] == got["expected"] == [] and got["uses"] == 3
    got = of_record.compare(step(same, [(D * D, D, 32), (D * D, D, 32), (D, 0, 32)]), step(same, record), SHAPE)
    assert got["expected"] == [] and got["uses"] == 0 and got["differences"] == [
        "updated, the weights' bits: the config's (256, 16) at 32 x2; the record's (256, 16) at 16 x2"]
    got = of_record.compare(step(same, [(D * D, D, 16), (D * D, D, 16), (D, 0, 16)]), step(same, record), SHAPE)
    assert got["expected"] == [] and got["differences"] == [
        "updated, the weights' bits: the config's (16, 0) at 16 x1; the record's (16, 0) at 32 x1"]
    got = of_record.compare(step(same, [(D * D, 2 * D, 32), (D * D, D, 32), (D, 0, 32)]), step(same, record), SHAPE)
    assert got["expected"] == [] and got["uses"] == 0 and got["differences"] == [
        "updated, the config's step only: (256 MACs a token, 32 upstream) x1",
        "updated, the record's step only: (256 MACs a token, 16 upstream) x1"]


def test_a_step_stands_for_no_run_when_the_capture_refuses_it_or_it_reads_a_value_back():
    trace = {"data_dependent": [], "carried_scalars": []}
    assert of_record.refusal({"trace": trace, "invalid": []}) is None
    refused = {"refused": {"reason": "dynamic_control_flow", "op": "aten.item", "message": "why"}}
    assert of_record.refusal({"trace": refused, "invalid": []}) == (
        "the capture refused the step (dynamic_control_flow, aten.item: why)")
    assert "reads a value back" in of_record.refusal({"trace": trace | {"data_dependent": ["a clip"]}, "invalid": []})
    assert of_record.refusal({"trace": trace, "invalid": ["no step"]}) == "the count is invalid (no step)"


def test_every_held_entry_is_an_entry_of_the_library_with_its_record_named():
    runs = of_record.held()
    assert set(runs) <= {p.stem for p in library.entries()}
    for h in runs.values():
        assert set(h) - {"record_only", "partial", "step"} == {"run", "art", "path", "sha256"}
        assert "step" not in h or (set(h["step"]) == {"counted", "why"} and h["step"]["counted"] > of_record.STEPS)
        assert len(h["sha256"]) == 64 and h["art"].startswith("art:")
        assert "partial" not in h or as_math.parse(h["partial"])  # an ask `as_math` reads
        assert all(set(line) == {"phase", "kind", "op", "shapes", "count", "macs", "why"}
                   for line in h.get("record_only", []))


#: a line of a record's own program: a scale times a multiplier, scalars both
SCALE = {"phase": "fwd", "kind": "act", "op": "mul", "shapes": [[], []], "count": 3, "macs": 3,
         "why": "its multiplier is 1"}


def test_a_line_the_entry_names_is_the_records_over_the_configs_and_one_the_record_lacks_is_a_difference():
    scale = ("fwd", "act", "mul", [[], []])
    record = log(*BOTH, (*scale, 3, 3))
    got = of_record.compare(step(log(*BOTH)), step(record), SHAPE, [SCALE])
    assert got["differences"] == [] and got["expected"] == [] and got["products"] == 2
    assert got["named"] == [{"line": "fwd/act mul [[] x2] x3, 3 MACs", "why": "its multiplier is 1"}]
    # a line the config has one of and the record four: the three over the config's are the named ones
    got = of_record.compare(step(log(*BOTH, (*scale, 1, 1))), step(log(*BOTH, (*scale, 4, 4))), SHAPE, [SCALE])
    assert got["differences"] == [] and [e["line"] for e in got["named"]] == ["fwd/act mul [[] x2] x3, 3 MACs"]
    # unnamed, the line is a difference; named with another count, or where the config computes it too, the name is
    assert of_record.compare(step(log(*BOTH)), step(record), SHAPE)["differences"] == [
        "log, the record's step only: fwd/act mul [[] x2] x3, 3 MACs"]
    stale = "of_record.toml names a line the record's step does not have over the config's as named: fwd/act mul "
    got = of_record.compare(step(log(*BOTH)), step(record), SHAPE, [SCALE | {"count": 2, "macs": 2}])
    assert got["named"] == [] and got["differences"] == [
        stale + "[[] x2] x2, 2 MACs", "log, the record's step only: fwd/act mul [[] x2] x3, 3 MACs"]
    got = of_record.compare(step(log(*BOTH, (*scale, 2, 2))), step(record), SHAPE, [SCALE])
    assert got["named"] == [] and got["differences"] == [
        stale + "[[] x2] x3, 3 MACs",
        "log, unequal: fwd/act mul [[] x2] x2, 2 MACs in the config's step, x3, 3 MACs in the record's"]
    got = of_record.compare(step(record), step(record), SHAPE, [SCALE])
    assert got["named"] == [] and got["differences"] == [stale + "[[] x2] x3, 3 MACs"] and got["products"] == 3


def count(*more, updated=((D * D, D),), macs=None) -> dict:
    """A config's counted record made up here: every step of one structure, `BOTH` and `more` its log."""
    entries = (*BOTH, *more)
    struct = {"updated": [[p, u, CTX * p, CTX, 16] for p, u in updated],
              "log": [[*e[:3], json.dumps(e[3]), *e[4:]] for e in entries]}
    total = sum(e[5] for e in entries) if macs is None else macs
    return {"invalid": [], "trace": {"data_dependent": [], "carried_scalars": []}, "checkpoints": [{"e_bytes": 10}],
            "steps": [{"struct": "s", "macs": total}] * of_record.STEPS, "structures": {"s": struct}}


def as_cuda(struct: dict, rows: int) -> dict:
    """A CPU step's structure as the records hold CUDA's step: attention one fused kernel a call at torch's formula
    (`fused`), the norm one fused kernel counted at its updated gains alone, each weight read at the bits this step
    reads it at (a route casts on a CPU as on CUDA). The step's every attention backward asks for q's, k's and v's,
    or this refuses to write it."""
    x, s = [rows, CTX, D], [rows, CTX, 1]
    kernel = {(*e[:3], json.dumps(e[3])): e[4] for e in attention(rows, calls=1)}
    calls, out = {}, []
    for phase, kind, op, shapes, count, macs in struct["log"]:
        at = json.loads(shapes)
        if (phase, kind, op, shapes) in kernel:
            calls[phase, op, shapes] = count // kernel[phase, kind, op, shapes]
            continue
        if (phase, kind, op, at) == ("fwd", "updated", "mul", [x, G]):
            op = "_fused_rms_norm"
        elif op in ("mul", "div") and all(o in (x, s, G) for o in at) and phase != "other":
            continue
        out.append([phase, kind, op, json.dumps(at), count, macs])
    forward, backward = ({n for (phase, *_), n in calls.items() if phase == which} for which in ("fwd", "bwd"))
    assert len(calls) == len(kernel) and len(forward) == len(backward) == 1  # every row of the kernel's, of one count
    return {"updated": struct["updated"], "log": out + fused(rows, forward.pop(), backward.pop())}


def held_to(root, entry: str, rec: dict, record_only=(), partial: str | None = None, step: dict | None = None):
    """`rec` as `entry`'s run of record under `root`: where its records are, and the data file naming it, the
    gradients `partial` says its attention backward asked for, the lines `record_only` of its program and the step
    `step` says is compared."""
    run, path = "r-made-up", "out/run/run.json"
    (root / "records" / run / "out" / "run").mkdir(parents=True)
    raw = json.dumps(rec).encode()
    (root / "records" / run / path).write_bytes(raw)
    lines = "".join(f'[[held."{entry}".record_only]]\n' + "".join(f"{k} = {json.dumps(v)}\n" for k, v in line.items())
                    for line in record_only)
    (root / "held.toml").write_text(f'[held."{entry}"]\nrun = "{run}"\nart = "art:{"0" * 64}"\npath = "{path}"\n'
                                    f'sha256 = "{hashlib.sha256(raw).hexdigest()}"\n'
                                    + (f'partial = "{partial}"\n' if partial else "")
                                    + (f'step = {{ counted = {step["counted"]}, why = {json.dumps(step["why"])} }}\n'
                                       if step else "") + lines)
    return root / "records", root / "held.toml"


@pytest.mark.slow
def test_the_command_names_each_expected_difference_and_exits_1_on_any_other(tmp_path, capsys):
    """A made-up run of record: the config's own counted step at two rows a forward, written as the records hold CUDA's
    step, attention one fused row a kernel and no trace. With the ask stated, the command writes that kernel as the
    counter does, divides the rows out, counts the config at one row a forward and names the expected difference.
    With no ask stated the entry is refused, naming the key; with a wrong one, a call for q's alone where the step's
    every call asks for all three, it differs. A config that trains another set, beside a record of another start and
    length, differs and exits 1. It counts and traces a route's steps, so slow; what the rules' own tests can't show is
    that a route's real step produces the entries each rule names, that the counter's rows of a fused kernel are the
    ones a rewritten record holds, and that the command reads a record the way a job writes one."""
    reg = tiny_registry(tmp_path / "reg")
    budget, dials = {"steps": 3, "batch": 4}, {"trainable": "blocks.*,norm", "opt": "muon", "lr": 1e-2,
                                               "micro_batches": 2, "fused": True}
    path = write_config(tmp_path / "a.toml", budget=budget, dials=dials)
    cfg = config.load(path, starts.registry(reg))
    cpu = of_record.counted(cfg, 1, tmp_path / "count")
    assert cpu["invalid"] == [] and cpu["checkpoints"][-1]["step"] == 1
    first, second = cpu["steps"]
    tables = [kind for _, kind, op, shapes, *_ in cpu["structures"][second["struct"]]["log"]
              if op == "mul" and json.loads(shapes)[-1] == [CTX, 4]]
    assert set(tables) == {"theta", "grad"}  # a product with a position table is θ's, as the records count it
    rec = {k: v for k, v in cpu.items() if k != "trace"} | {
        "steps": [first, second, second], "start_record": {"name": "tiny", "digest": cfg.start.digest},
        "structures": cpu["structures"] | {second["struct"]: as_cuda(cpu["structures"][second["struct"]], rows=2)}}
    records, held = held_to(tmp_path, "a", rec, partial="none")
    argv = ["--records", str(records), "--held", str(held), "--starts", str(reg)]
    assert of_record.main([str(path), *argv, "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal, 1 expected difference\n")
    assert "  expected  norm: " in out and out.count("  expected  ") == 1
    assert "  attention the record holds a fused kernel as one row: compared as the ops the math kernel" in out
    assert "`none` at [4, 1024, 8], of_record.toml's word (`partial`) and not the record's" in out
    assert "2 forwards of one row" in out and "2 of its 3 steps have, 2 rows a forward divided out" in out
    assert "DIFFERS" not in out and out.rstrip().endswith(of_record.FAIR) and "1 entry: 1 equal, 0 differ" in out
    kept = json.loads((tmp_path / "out" / "of_record.json").read_text())
    assert kept["schema"] == of_record.SCHEMA and kept["rows"][0]["differences"] == []
    own = json.loads((tmp_path / "out" / "a" / "run.json").read_text())
    assert own["invalid"] == [] and own["trace"]["tokens"] == 4 * CTX and own["steps"][1]["tokens"] == 2 * CTX

    records, held = held_to(tmp_path / "unsaid", "a", rec)
    assert of_record.main([str(path), "--records", str(records), "--held", str(held), "--starts", str(reg)]) == 1
    out = capsys.readouterr().out
    assert out.startswith(
        "a, held to r-made-up (out/run/run.json): refused: its record's fused attention rows can't be compared as "
        "the math kernel's ops: the record's trace holds no attention backward at [4, 1024, 8] to read the gradients "
        "asked from: say --partial none (every call asks for q's, k's and v's) or --partial SUBSET:CALLS (what "
        "`--partial` would say is the entry's `partial` in of_record.toml)\n")
    assert "1 entry: 0 equal, 0 differ, 1 refused." in out

    records, held = held_to(tmp_path / "wrong", "a", rec, partial="q:1")
    assert of_record.main([str(path), "--records", str(records), "--held", str(held), "--starts", str(reg)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): differs, 3 differences\n")
    assert out.count("  DIFFERS   log, unequal: bwd/grad ") == 3 and "`q:1` at [4, 1024, 8], of_record.toml's" in out
    assert ("  DIFFERS   log, unequal: bwd/grad bmm [[2,8,1024], [2,1024,1024]] x2, 33,554,432 MACs in the config's "
            "step, x1, 16,777,216 MACs in the record's\n") in out

    other = write_config(tmp_path / "b" / "a.toml", budget=budget, dials=dials | {"trainable": "blocks.*"})
    rec["start_record"]["digest"] = "e" * 64
    records, held = held_to(tmp_path / "elsewhere", "a", rec | {"steps": rec["steps"] * 2}, partial="none")
    assert of_record.main([str(other), "--records", str(records), "--held", str(held), "--starts", str(reg)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): differs, ")
    assert "  DIFFERS   the start: the config's is tiny (dddddddddddd), the record's tiny (eeeeeeeeeeee)" in out
    assert "  DIFFERS   the steps: the config's 3, the record's 6" in out
    assert "  DIFFERS   updated, the record's step only: " in out and "1 entry: 0 equal, 1 differ" in out
    assert "  DIFFERS   log, the record's step only: fwd/updated _fused_rms_norm" in out


def test_an_entry_that_states_its_step_is_counted_to_it_and_the_output_says_which_and_why(tmp_path, capsys,
                                                                                         monkeypatch):
    """An entry whose trained weights leave θ after the second step states the step compared (`step`): the config is
    counted to it, that step is the one set beside the record's, and the output prints it with the entry's why. An
    entry that states none is counted to `STEPS`."""
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    cfg = config.load(path, starts.registry(reg))
    mine, asked = count(), []

    def counted(cfg, rows, out, n=of_record.STEPS):
        asked.append(n)
        other = {"updated": [], "log": []}
        return mine | {"steps": [{"struct": "o"}] * (n - 1) + [{"struct": "s"}],
                       "structures": mine["structures"] | {"o": other}}

    monkeypatch.setattr(of_record, "counted", counted)
    rec = {"start_record": {"name": "tiny", "digest": cfg.start.digest}, "structures": dict(mine["structures"]),
           "steps": [{"struct": "s"}] * cfg.budget["steps"], "checkpoints": [{"format": "full", "e_bytes": 20}]}
    why = "a gain behind a zero matrix gets no gradient at the first step"
    records, held = held_to(tmp_path, "a", rec, step={"counted": 3, "why": why})
    assert of_record.main([str(path), "--records", str(records), "--held", str(held), "--starts", str(reg)]) == 0
    out = capsys.readouterr().out
    assert asked == [3] and out.startswith("a, held to r-made-up (out/run/run.json): equal\n")
    assert ("  step      the config's step 3, 2 forwards of one row, counted on a CPU, beside the record's modal "
            f"structure, which 3 of its 3 steps have, 1 row a forward divided out; of_record.toml states step 3: "
            f"{why}\n") in out and "identity, 10 B after 3 steps" in out
    records, held = held_to(tmp_path / "unstated", "a", rec)
    assert of_record.main([str(path), "--records", str(records), "--held", str(held), "--starts", str(reg)]) == 0
    assert asked == [3, of_record.STEPS] and "the config's step 2," in capsys.readouterr().out


def test_a_difference_no_rule_names_is_printed_and_the_command_exits_1(tmp_path, capsys, monkeypatch):
    """The command on a count made up here, so no step is counted: a step equal to its record's exits 0, and so does
    one whose record has a line more that the data file names; a product the record lacks, which no rule names, is
    printed as a difference and exits 1; so does a step the capture refused, which is compared with nothing."""
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    cfg = config.load(path, starts.registry(reg))
    mine = count()
    monkeypatch.setattr(of_record, "counted", lambda *a: mine)
    rec = {"start_record": {"name": "tiny", "digest": cfg.start.digest}, "structures": dict(mine["structures"]),
           "steps": [{"struct": "s"}] * cfg.budget["steps"], "checkpoints": [{"format": "full", "e_bytes": 20}]}
    records, held = held_to(tmp_path, "a", rec)
    argv = [str(path), "--records", str(records), "--held", str(held), "--starts", str(reg)]
    assert of_record.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal\n") and "DIFFERS" not in out
    assert ("  step      the config's step 2, 2 forwards of one row, counted on a CPU, beside the record's modal "
            "structure, which 3 of its 3 steps have, 1 row a forward divided out\n") in out
    assert "  updated   1 of the record's 1 uses" in out and "  log       2 of the record's 2 products" in out
    assert "  E         identity, 10 B after 2 steps; the record's full, 20 B after 3" in out

    scaled = rec | {"structures": count(("fwd", "act", "mul", [[], []], 3, 3))["structures"]}
    records, named = held_to(tmp_path / "named", "a", scaled, [SCALE])
    assert of_record.main([str(path), "--records", str(records), "--held", str(named), "--starts", str(reg)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal, 1 named line of the record's program "
                          "alone\n") and "DIFFERS" not in out and "1 entry: 1 equal, 0 differ, 0 refused." in out
    assert ("  named     in the record's step and not the config's: fwd/act mul [[] x2] x3, 3 MACs; of_record.toml "
            "names it: its multiplier is 1\n") in out and "  log       2 of the record's 3 products" in out

    mine["structures"] = count(("fwd", "act", "matmul", [[CTX, D], [D, D]], 1, CTX * D * D))["structures"]
    assert of_record.main(argv) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): differs, 1 difference\n")
    assert "  DIFFERS   log, the config's step only: fwd/act matmul [[1024,16], [16,16]] x1, 262,144 MACs" in out
    assert "1 entry: 0 equal, 1 differ, 0 refused." in out

    mine["trace"] = {"refused": {"reason": "dynamic_control_flow", "op": "aten.item", "message": "why"}}
    assert of_record.main(argv) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): refused: the capture refused the step")
    assert "DIFFERS" not in out and "1 entry: 0 equal, 0 differ, 1 refused." in out


def test_a_record_that_is_missing_or_another_file_is_refused_with_its_fetch_line(tmp_path):
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    records, held = held_to(tmp_path, "a", {"steps": []})
    argv = [str(path), "--held", str(held), "--starts", str(reg), "--records"]
    with pytest.raises(SystemExit, match=r"a's record is not at .*nowhere/r-made-up/out/run/run.json \(`research "
                                         r"data fetch art:0+ --to .*nowhere/r-made-up --path out/run/run.json`\)"):
        of_record.main([*argv, str(tmp_path / "nowhere")])
    (records / "r-made-up" / "out" / "run" / "run.json").write_text("{}")
    with pytest.raises(SystemExit, match="is not the record of_record.toml names for a: its sha256 is 44136fa3"):
        of_record.main([*argv, str(records)])
    with pytest.raises(SystemExit, match="held.toml holds b to no run"):
        of_record.main([str(write_config(tmp_path / "b.toml")), *argv[1:], str(records)])
