"""training_slowdown.capture.products: a step's matrix products as the products term reads them. On a toy step at
H = 256, past the K at which the GEMM rate rule changes (3^1.5·√(k/32), 235 at the console's k)."""
from __future__ import annotations

import contextlib
import json
import math

import pytest
import torch
import torch.nn.functional as tF
from torch import nn

from training_slowdown import calculator, policy, work
from training_slowdown.capture import CaptureRefused, capture_step, capture_steps, products, trace

V, H, B, CTX = 16, 256, 4, 8
SHAPES = {"emb": [V, H], "fc": [H, H], "head": [V, H]}


class Toy(nn.Module):
    """An embedding, one tanh layer and an untied head."""

    def __init__(self):
        super().__init__()
        self.emb, self.fc, self.head = (nn.Parameter(torch.randn(*SHAPES[n])) for n in ("emb", "fc", "head"))

    def forward(self, x):
        return tF.linear(torch.tanh(tF.linear(tF.embedding(x, self.emb), self.fc)), self.head)


def test_a_steps_rows_and_their_kinds():
    """The forward's head product is fresh and its fc product an expansion (a lookup at the token ids is paid at no
    rate); in the backward three products are fresh and one an expansion. The rows carry the capture's version, the
    order unknown and the step's |C| (`step_gates`)."""
    k = policy.CONSOLE[1]
    model = Toy()
    g = capture_step(model, torch.optim.SGD(model.parameters(), lr=1e-3),
                     lambda m, b: tF.cross_entropy(m(b[0]).flatten(0, 1), b[1].flatten()),
                     (torch.zeros(B, CTX, dtype=torch.long), torch.zeros(B, CTX, dtype=torch.long)))
    got = products.rows(g, k)
    assert (got["capture"], got["order"]) == (products.CAPTURE, "unknown")
    assert got["gates"] == products.step_gates(g) > 0 and all(r["count"] == 1 for r in got["rows"])
    assert sorted((r["phase"], work.product_kind(r)) for r in got["rows"]) == sorted(
        [("forward", "fresh"), ("forward", "expansion"), ("backward", "expansion")] + [("backward", "fresh")] * 3)


def test_the_capture_and_the_stored_records_capture_are_the_two_that_are_read():
    """Every stored record was traced by the capture's 4.12.9 rows, which are read still, by that capture's rule
    (`priced_dtype`); from `PRICED_AS_TRACED_FROM` on a product is priced in the dtype its trace's |C| counts it in,
    and this capture's rows state the attention call each is of. Moving `CAPTURE` moves the grader's with it, and
    dropping 4.12.9 from `READS` needs every stored record traced again."""
    assert products.READS == ("4.12.9", products.CAPTURE)
    assert (products.PRICED_AS_TRACED_FROM, products.CAPTURE) == ("4.23.0", "4.25.0")


def test_a_product_is_priced_in_its_traced_dtype_and_an_earlier_captures_wider_one_in_bf16():
    """One rule: a product's price is the Definition of the dtype the trace's |C| counts it in: its traced dtype's,
    and for a product of an attention call the dtype the call computes in (`attention`), whatever it was traced in.
    The rule of the captures before `PRICED_AS_TRACED_FROM` stays for their traces alone: a forward or backward
    product they traced wider than bf16 (attention's, which every capture traces in float32) is priced in bf16, an
    optimizer's in its own, and a trace that states no capture is read as theirs."""
    from training_slowdown import gates

    def row(phase: str, dtype: str, attention: str | None = None) -> dict:
        return {"count": 1, "phase": phase, "op": "bmm", "dims": {"batch": 2, "m": 4, "k": 8, "n": 4},
                "token_dims": ["batch"], "dtypes": [dtype, dtype], "attention": attention, "affected": [True, True],
                "widths": [32, 32], "self": False, "gathered": [False, False], "rates": [32.0, 32.0]}

    f32, bf16 = gates.per_mac("gemm/f32"), gates.per_mac("gemm/bf16")
    assert f32 > bf16
    for phase in ("forward", "backward", "optimizer"):
        wide, narrow = row(phase, "float32"), row(phase, "bfloat16")
        assert products.traced_dtype(wide) == products.priced_dtype(wide, products.CAPTURE) == "float32"
        assert products.priced(wide, products.CAPTURE)["gates_per_mac"] == f32
        assert all(products.priced(narrow, c)["gates_per_mac"] == bf16 for c in (*products.READS, None))
        for earlier in ("4.12.9", "4.12.11", "4.22.0", None, "a tree's own"):
            repriced = phase != "optimizer"
            assert products.priced_dtype(wide, earlier) == ("bfloat16" if repriced else "float32")
            assert products.priced(wide, earlier)["gates_per_mac"] == (bf16 if repriced else f32)
    assert products.priced_dtype(row("forward", "float32"), "4.24.1") == "float32"
    for phase in ("forward", "backward"):
        called = row(phase, "float32", "bfloat16")
        assert (products.traced_dtype(called), products.counted_dtype(called)) == ("float32", "bfloat16")
        assert products.priced_dtype(called, products.CAPTURE) == "bfloat16"
        assert products.priced(called, products.CAPTURE)["gates_per_mac"] == bf16
        assert products.priced(row(phase, "float32", "float32"), products.CAPTURE)["gates_per_mac"] == f32
    # a row of a capture that states no call (no `attention`) is priced as traced
    unstated = {k: v for k, v in row("forward", "float32").items() if k != "attention"}
    assert products.priced_dtype(unstated, "4.23.0") == products.priced_dtype(unstated, products.CAPTURE) == "float32"


def test_a_products_traced_dtype_is_its_results_by_the_counters_rule():
    """One rule with the counter (`flops.dtype_of`): the dtype a product computes in is its result's. A raw row holds
    its operands' dtypes, and torch computes a product in its operands' one dtype, so the row's dtype is operand A's
    wherever the two agree; a row whose operands differ is in the dtype torch promotes them to, the wider, whichever
    operand is first; and a kernel that casts its result (`_scaled_mm`) is in its first operand's."""
    def row(op: str, a: str, b: str) -> dict:
        return {"phase": "forward", "op": op, "dtypes": [a, b]}

    assert products.traced_dtype(row("mm", "bfloat16", "bfloat16")) == "bfloat16"
    assert products.traced_dtype(row("mm", "float32", "bfloat16")) == "float32"
    assert products.traced_dtype(row("mm", "bfloat16", "float32")) == "float32"
    assert products.traced_dtype(row("_scaled_mm", "float8_e4m3fn", "float8_e5m2")) == "float8_e4m3fn"
    a, b = torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(3, 2)
    with pytest.raises(RuntimeError, match="dtype"):
        a @ b


RAW = {"count", "phase", "op", "dims", "token_dims", "dtypes", "attention", "affected", "widths", "self", "gathered",
       "rates"}
PRICE = {"macs", "outputs", "paid", "b_c", "gates_per_mac", "cell_gates"}


def test_a_raw_row_has_no_price_and_priced_makes_it_the_row_rows_states():
    """What a record keeps (`raw_on_dial`) holds each row's operands' dtypes, the attention call it is of (none in
    this step) and nothing a price decides, at the
    console's k and at each other k of its dial, and pricing it after a round trip through JSON gives `rows`' rows at
    that k: a record's rows read where the grade prices them are the rows the capture would have stated."""
    model = Toy()
    g = capture_step(model, torch.optim.SGD(model.parameters(), lr=1e-3),
                     lambda m, b: tF.cross_entropy(m(b[0]).flatten(0, 1), b[1].flatten()),
                     (torch.zeros(B, CTX, dtype=torch.long), torch.zeros(B, CTX, dtype=torch.long)))
    raw = json.loads(json.dumps(products.raw_on_dial(g)))
    assert set(raw) == set(products.rows(g)) - {"gates"} | {"at"}
    assert sorted(map(int, raw["at"])) == sorted(set(products.DIAL_K) - {raw["k"]})
    assert raw["at"][str(800_000_000)]["rows"] != raw["rows"]
    for a in [raw] + list(raw["at"].values()):
        b = products.rows(g, a["k"])
        assert set(a) >= {"k", "rows", "p_match"} and a["p_match"] == b["p_match"]
        assert all(set(r) == RAW and r["dtypes"] == ["float32", "float32"] and r["attention"] is None
                   for r in a["rows"])
        assert [products.priced(r, raw["capture"]) for r in a["rows"]] == b["rows"]
    assert raw["capture"] == products.CAPTURE
    for r in raw["rows"]:
        d, got = r["dims"], products.priced(r, raw["capture"])
        assert set(got) == RAW - {"dtypes", "attention"} | PRICE
        assert (got["macs"], got["outputs"]) == (d["batch"] * d["m"] * d["k"] * d["n"], d["batch"] * d["m"] * d["n"])
        assert got["paid"] == [work.is_paid(x, w) for x, w in zip(r["rates"], r["widths"])]
        assert got["b_c"] == products.ACC_BITS
        # an fp32 product is priced in fp32, the dtype it was traced in
        assert got["gates_per_mac"] > products.priced(r | {"dtypes": ["bfloat16", "bfloat16"]},
                                                      raw["capture"])["gates_per_mac"]


@pytest.mark.parametrize("wide, op, dtype", [
    (lambda h, w: tF.linear(h.double(), w.double()).float(), "mm", "float64"),
    (lambda h, w: tF.linear(h, w) * (torch.arange(32) * 2).sum(), "mul", "int64")])
def test_a_step_with_an_op_in_a_dtype_no_definition_prices_is_refused(wide, op, dtype):
    """A layer cast to fp64, and an integer multiplication beside an fp32 layer: the step's circuit has an op no
    Definition prices, and the capture refuses the step with the op and the dtype, where it priced such an op as a
    bf16 one."""
    class Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1, self.w2, self.w3 = (nn.Parameter(torch.randn(32, 32)) for _ in range(3))

        def forward(self, x):
            h = tF.linear(x, self.w1)
            return tF.linear(h, self.w2) + wide(h, self.w3)

    model = Wide()
    with pytest.raises(CaptureRefused, match=f"unpriced_dtype: node .*: {op} in {dtype}: no Definition") as e:
        capture_step(model, torch.optim.SGD(model.parameters(), lr=1e-3), lambda m, b: m(b).square().mean(),
                     torch.randn(4, 32))
    assert (e.value.reason, e.value.op) == ("unpriced_dtype", f"aten.{op}.{'default' if op == 'mm' else 'Tensor'}")


class Fan(nn.Module):
    """Three trained layers; the second's output reaches the third's operand through one column repeated across the
    row (`fan`), through a row norm (`norm`), or directly."""

    def __init__(self, d: int, via: str) -> None:
        super().__init__()
        self.w1, self.w2, self.w3 = (nn.Parameter(torch.randn(d, d)) for _ in range(3))
        self.via = via

    def forward(self, x):
        h = (x @ self.w1) @ self.w2
        if self.via == "fan":
            h = h[:, :1].expand(-1, h.shape[1])
        elif self.via == "norm":
            h = h / h.norm(dim=-1, keepdim=True)
        return h @ self.w3


@pytest.mark.parametrize("via, broken", (("direct", []), ("norm", []),
                                         ("fan", [{"op": "expand", "fan_out": 16.0, "count": 1}])))
def test_p_match_fails_where_one_entry_is_handed_to_several_on_the_way_into_an_operand(via, broken):
    """A product's output carried into the next product's operand one to one (directly, or through a row norm whose
    statistic is charged nothing and whose broadcast its reduction pays for) keeps P-match; one column repeated across
    its row hands each entry to 16, so P-match fails at the expand and the products term is unmeasured with the
    reason."""
    torch.manual_seed(0)
    m = Fan(16, via)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: tF.mse_loss(mm(b[0]), b[1]),
                     (torch.randn(6, 16), torch.randn(6, 16)))
    k = policy.CONSOLE[1]
    got = products.rows(g, k)
    p = {"k": k, "p_match": got["p_match"], "he": work.he_of_rows(got["rows"], k)}
    assert got["p_match"] == broken
    entry = calculator.slowdown(dict(zip(("c_gates", "k_bits"), policy.CONSOLE)), p, {"gates": got["gates"]})
    term, why = entry["term"], entry["why"]
    if broken:
        assert term is None and f"breaks the premise that {work.P_MATCH}" in why and "expand (fan-out 16" in why
    else:
        assert why is None and term is not None


class Embedded(nn.Module):
    """An embedding's rows into two trained layers, wide enough that the first's outputs are paid, with a position's
    table sliced to the context between them (`pos`) or not."""

    def __init__(self, pos: bool, d: int = 256) -> None:
        super().__init__()
        self.e = nn.Embedding(16, d)
        self.w1, self.w2 = nn.Parameter(torch.randn(d, d) / d ** 0.5), nn.Parameter(torch.randn(d, 16) / d ** 0.5)
        self.register_buffer("table", torch.randn(64, d))
        self.pos = pos

    def forward(self, ids):
        h = self.e(ids) @ self.w1
        if self.pos:
            h = h * self.table[: ids.shape[-1]]
        return h @ self.w2


@pytest.mark.parametrize("pos", (False, True))
def test_products_that_read_an_embeddings_rows_before_a_position_makes_them_token_specific_are_dropped(pos):
    """Tokens that share an id share its row, and so do their first layer's outputs, so the second layer's forward
    product and its weight gradient read gathered rows unless a position's table makes them token-specific first. A
    trace's rows say which operands do (`gathered`), and the products term leaves those products out of M (never
    |C|). The capture's P-match check finds that the gather breaks nothing."""
    torch.manual_seed(0)
    m = Embedded(pos)

    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1),
                     lambda mm, ids: tF.cross_entropy(mm(ids[:, :-1]).reshape(-1, 16), ids[:, 1:].reshape(-1)),
                     torch.randint(0, 16, (2, 9)))
    k = policy.CONSOLE[1]
    got = products.rows(g, k)
    flags = sorted((r["phase"], r["dims"]["k"], tuple(r["gathered"])) for r in got["rows"] if any(r["gathered"]))
    first = [("backward", 16, (True, False)), ("forward", 256, (True, False))]
    second = [("backward", 16, (True, False)), ("forward", 256, (True, False))]
    assert flags == sorted(first if pos else first + second)
    assert got["p_match"] == []
    second_fresh = [r for r in got["rows"] if r["dims"]["n"] == 16 and work.product_kind(r) == "fresh"]
    assert second_fresh and all(any(r["gathered"]) != pos for r in second_fresh)
    gathered = [r for r in got["rows"] if all(r["affected"]) and not r["self"] and any(r["gathered"])]
    assert work.products_of_trace(got)["he"]["gathered_macs"] == sum(r["count"] * r["macs"] for r in gathered) > 0


def _product(form: str, u, v):
    """u @ v as a GEMM, as a broadcast multiplication and a sum, or as v's entries gathered, multiplied and added by
    index (a sparse matrix holding every entry of v)."""
    if form == "matmul":
        return u @ v
    if form == "broadcast":
        return (u.unsqueeze(-1) * v).sum(-2)
    if form.startswith("hidden-"):
        return _Hidden.apply(u, v, form.removeprefix("hidden-"))
    rows, cols = (x.flatten() for x in torch.meshgrid(torch.arange(v.shape[0]), torch.arange(v.shape[1]), indexing="ij"))
    return torch.zeros(*u.shape[:-1], v.shape[1], dtype=u.dtype).index_add(-1, cols, u[..., rows] * v.flatten())


def _through(t, how: str):
    """`t` as it is, or passed through a map (neg twice), a scaling and an addition, or a copy in place."""
    if how == "neg":
        return t.neg().neg()
    if how == "scale":
        return t * 1.0 + 0.0
    if how == "copy":
        buf = torch.empty_like(t)
        buf.copy_(t)
        return buf
    return t


def _hide(u, v, how: str):
    """u @ v as a broadcast multiplication and a sum, the broadcast passed through `how` first (`_through`, or a
    multiplication by ones, which don't depend on θ′)."""
    e = u.unsqueeze(-1).expand(*u.shape, v.shape[-1])
    e = u.unsqueeze(-1) * torch.ones(v.shape[-1], dtype=u.dtype) if how == "ones" else _through(e, how)
    return (e * v).sum(-2)


class _Hidden(torch.autograd.Function):
    """u @ v written by `_hide` in the forward and the backward both, so no op autograd writes shows the product."""

    @staticmethod
    def forward(ctx, u, v, how):
        ctx.save_for_backward(u, v)
        ctx.how = how
        return _hide(u, v, how)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        return (_hide(g, v.t(), ctx.how), _hide(u.reshape(-1, u.shape[-1]).t(), g.reshape(-1, g.shape[-1]), ctx.how),
                None)


class Adapted(nn.Module):
    """Two stacked rank-16 adapters on frozen 512 x 512 matrices, the adapters' products written `form`; `sparse` adds
    a trained sparse matrix with one entry per row, applied by gather, multiplication and index_add, so no entry of
    its operand is read twice, the gather's rows and the product each passed through `how` (`_through`)."""

    def __init__(self, form: str, d: int = 512, r: int = 16, how: str = "plain") -> None:
        super().__init__()
        self.form, self.how = form, how
        self.w = nn.ParameterList(nn.Parameter(torch.randn(d, d) / d ** 0.5, requires_grad=False) for _ in range(2))
        self.a = nn.ParameterList(nn.Parameter(torch.randn(d, r) / d ** 0.5) for _ in range(2))
        self.b = nn.ParameterList(nn.Parameter(torch.randn(r, d) / r ** 0.5) for _ in range(2))
        self.vals = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.register_buffer("rows", torch.randperm(d))
        self.register_buffer("cols", torch.randint(0, d, (d,)))

    def forward(self, x):
        for w, a, b in zip(self.w, self.a, self.b):
            if self.form == "sparse":
                rows = _through(x.index_select(-1, self.rows), self.how)
                s = torch.zeros_like(x).index_add(-1, self.cols, _through(rows * self.vals, self.how))
                x = x @ w + (x @ a) @ b + s
            else:
                x = x @ w + _product(self.form, _product(self.form, x, a), b)
        return x


@pytest.mark.parametrize("form, flagged", (("matmul", []), ("broadcast", ["mul"]), ("index", ["mul"]),
                                           ("sparse", ["index_add"])))
def test_products_written_with_elementwise_ops_are_found_and_are_no_rows(form, flagged):
    """A bf16 step of two stacked rank-16 adapters over 8 x 256 tokens. Written with @, three of its products are fresh
    (50.3M MACs, the second layer's down projection and both up projections), with nothing written elementwise.
    Written as broadcast multiplications and sums, or as gathers, multiplications and index_adds, the adapters'
    products are no rows, so none is fresh and M holds none of them; the capture finds each (their multiplications,
    each broadcasting both operands, in the forward and the backward). A sparse product whose gather reads each entry
    once is found at its sum by index, in the forward and the backward; the first layer's reads the frozen input, so
    its product is θ's and not one."""
    torch.manual_seed(0)
    m = Adapted(form).to(torch.bfloat16)
    x = torch.randn(8, 256, 512, dtype=torch.bfloat16)
    g = capture_step(m, torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1),
                     lambda mm, b: tF.mse_loss(mm(b[0]), b[1]), (x, x.clone()))
    got = products.rows(g, policy.CONSOLE[1])
    found = got["elementwise_products"]
    assert [x["op"] for x in found] == flagged
    fresh = [r for r in got["rows"] if work.product_kind(r) == "fresh"]
    if form in ("matmul", "sparse"):
        assert sum(r["count"] for r in fresh) == 3 and sum(r["count"] * r["macs"] for r in fresh) == 50_331_648
    else:
        assert fresh == [] and found[0]["count"] == 9 and found[0]["macs"] == 150_994_944
    if form == "sparse":
        assert found[0]["count"] == 2 and found[0]["macs"] == 2 * 8 * 256 * 512


def _elementwise(m, x, y=None) -> list[dict]:
    g = capture_step(m, torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1),
                     lambda mm, b: tF.mse_loss(mm(b[0]), b[1]), (x, x.clone() if y is None else y))
    return products.rows(g, policy.CONSOLE[1])["elementwise_products"]


@pytest.mark.parametrize("how", ("plain", "neg", "scale", "ones", "copy"))
def test_a_broadcast_product_is_found_whatever_its_broadcast_passes_through(how):
    """The adapters' products as custom autograd functions that write each, forward and backward, as a broadcast
    multiplication and a sum, the broadcast first passed through a map, a scaling and an addition, a multiplication by
    ones, or a copy in place. 4.12.7's check lost a broadcast at any op but a view and found none of these; a value's
    entries are its sources' (`products._sources`), so each is found as the plain form is, all ten products."""
    torch.manual_seed(0)
    found = _elementwise(Adapted(f"hidden-{how}").to(torch.bfloat16), torch.randn(2, 16, 512, dtype=torch.bfloat16))
    assert found == [{"op": "mul", "count": 10, "macs": 2_621_440}]


@pytest.mark.parametrize("how", ("plain", "neg", "scale", "copy"))
def test_a_sparse_product_is_found_whatever_its_rows_pass_through(how):
    """sparse's gather, multiplication and index_add, the gathered rows and their product each passed through a map,
    a scaling and an addition, or a copy in place before the sum by index: found at both of its sums, as written
    plainly."""
    torch.manual_seed(0)
    found = _elementwise(Adapted("sparse", how=how).to(torch.bfloat16), torch.randn(2, 16, 512, dtype=torch.bfloat16))
    assert found == [{"op": "index_add", "count": 2, "macs": 2 * 2 * 16 * 512}]


class Looked(nn.Module):
    """A trained embedding of 64 rows of 32, looked up `form`, then a norm, a layer and a head: by a lookup, by
    indexing, by a computed index, from a table the step builds of the embedding (`built`, as a grown model's copy of
    θ's table is), or, the control, from a table the step builds as a trained vector's broadcast (`spread`)."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.form = form
        self.emb = nn.Embedding(64, 32)
        if form == "spread":
            self.lane = nn.Parameter(torch.randn(32))
        self.g = nn.Parameter(torch.ones(32))
        self.w = nn.Linear(32, 32, bias=False)
        self.head = nn.Linear(32, 64, bias=False)

    def forward(self, tok):
        x = (self.emb(tok) if self.form == "embedding" else self.emb.weight[tok] if self.form == "index" else
             tF.embedding(tok, self.emb.weight * 1.0) if self.form == "built" else
             tF.embedding(tok, self.lane.expand(64, 32) * 1.0) if self.form == "spread" else self.emb(tok + 0))
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g
        return self.head(x + self.w(h))


@pytest.mark.parametrize("form, found", (("embedding", []), ("index", []), ("built", []),
                                         ("computed", [{"op": "mul", "count": 3, "macs": 3 * 16_384}]),
                                         ("spread", [{"op": "mul", "count": 3, "macs": 3 * 16_384}])))
def test_an_embeddings_rows_are_no_broadcast(form, found):
    """512 tokens of a 64-row embedding: its rows repeat as the tokens do, which is the data and no broadcast, so the
    norm's multiplication of a token's row by its scale is elementwise (4.12.7's check took it for a product once a
    step had more tokens than the table has rows, which every large-batch dense step does). A lookup is a table's rows
    by a graph input (`products._lookup`, `products._rows_by_data`): an index the program computes is none, so there
    it is found. A table the step builds is one where its entries are its own, as a grown model's copy of θ's table
    is (4.12.13's check took it for a broadcast, and found its norm at every micro-batch), and none where it is a
    trained vector's broadcast, which is found."""
    torch.manual_seed(0)
    tok = torch.randint(0, 64, (8, 64))
    m = Looked(form)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1),
                     lambda mm, b: tF.cross_entropy(mm(b[0]).flatten(0, 1), b[1].flatten()), (tok, tok.clone()))
    assert products.rows(g, policy.CONSOLE[1])["elementwise_products"] == found


class Lanes(nn.Module):
    """A token's value spread over 8 lanes by constants, as an adapter's decoder unpacks a packed word, times
    the token's own scale (`same`) or a trained vector of the lanes (`outer`), then a head."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.form = form
        self.w = nn.Linear(32, 32, bias=False)
        self.lanes = nn.Parameter(torch.randn(8))
        self.head = nn.Linear(8, 8, bias=False)
        self.register_buffer("freqs", torch.arange(1.0, 9.0))

    def forward(self, x):
        h = self.w(x)
        code = torch.sin(h.sum(-1, keepdim=True) * self.freqs)
        return self.head(code * (h.pow(2).mean(-1, keepdim=True) if self.form == "same" else self.lanes))


@pytest.mark.parametrize("form, found", (("same", []), ("outer", [{"op": "mul", "count": 1, "macs": 2 * 16 * 8}])))
def test_two_broadcasts_along_the_same_dim_are_elementwise(form, found):
    """A token's code over 8 lanes times its own scale: both are broadcasts along the lanes, so neither is one along a
    dim the other isn't, and the multiplication is elementwise (`products._flats`), where its entries alone would make
    it a product (4.12.9's check first found such a decoder so). Times a trained vector of the lanes,
    it is an outer product, found."""
    torch.manual_seed(0)
    assert _elementwise(Lanes(form), torch.randn(2, 16, 32), torch.randn(2, 16, 8)) == found


class Dequantized(nn.Module):
    """A quant-format refresh: a trained 4 x 8 weight rounded once a step onto a constant 4-level codebook by
    each row's f16 RMS and decoded, cb[codes] * s, its update the decode minus the weight (`decode`); or, the control,
    the weight plus a trained vector's outer product with its column means (`outer`)."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.form = form
        self.A = nn.Parameter(torch.randn(4, 8))
        if form == "outer":
            self.u = nn.Parameter(torch.randn(4))
        self.register_buffer("cb", torch.tensor([-1.5, -0.5, 0.5, 1.5]))
        self.register_buffer("dA", torch.zeros(4, 8))

    def forward(self, x):
        if self.form == "outer":
            return tF.linear(x, self.A + self.u[:, None] * self.A.mean(0)[None, :])
        with torch.no_grad():
            s = self.A.pow(2).mean(1).sqrt().half().float()[:, None]
            codes = torch.bucketize(self.A / s, (self.cb[1:] + self.cb[:-1]) / 2)
            self.dA.copy_(self.cb[codes] * s - self.A)
        return tF.linear(x, self.A + self.dA)


@pytest.mark.parametrize("form, found", (("decode", []), ("outer", [{"op": "mul", "count": 1, "macs": 32}])))
def test_a_codebook_decode_is_elementwise_and_an_outer_product_beside_it_is_found(form, found):
    """A codebook decode: each decoded entry is its own code's level times its row's scale, so the decode is
    elementwise (4.12.7's check found it, as a mul of 32 multiplications; 4.12.9's doesn't), while the weight plus an
    outer product of a trained vector and its column means is one, found by both."""
    torch.manual_seed(0)
    assert _elementwise(Dequantized(form), torch.randn(2, 16, 8), torch.randn(2, 16, 4)) == found


class Messages(nn.Module):
    """A coded gradient message: a trained 4 x 64 weight in two groups of 128, each group's entries coded at
    `bits` by its absmax, packed 8 // bits to a byte and unpacked, so the update is the decode minus the weight; the
    decode is each level times its group's scale (`scale`), or, the control, the levels times a trained vector of a
    group's 128 entries (`vector`), which varies along the lanes the levels are flat along."""

    def __init__(self, bits: int, form: str = "scale") -> None:
        super().__init__()
        self.bits, self.form = bits, form
        self.A = nn.Parameter(torch.randn(4, 64))
        if form == "vector":
            self.v = nn.Parameter(torch.randn(128))
        self.register_buffer("shifts", torch.arange(8 // bits) * bits)
        self.register_buffer("dA", torch.zeros(4, 64))

    def forward(self, x):
        bits, per, L = self.bits, 8 // self.bits, 2 ** (self.bits - 1) - 1
        with torch.no_grad():
            g = self.A.reshape(1, 2, 128)
            sc = g.abs().amax(-1, keepdim=True)
            c = (torch.round(g / sc * L) + L).long().reshape(1, 256 // per, per)
            codes = (c << self.shifts).sum(-1).to(torch.uint8)
            q = (((codes.long()[..., None] >> self.shifts) & (2 ** bits - 1)).float() - L) / L
        if self.form == "vector":
            return tF.linear(x, self.A + (q.view(1, 2, 128) * self.v).view(4, 64))
        with torch.no_grad():
            self.dA.copy_((q.view(1, 2, 128) * sc).view(4, 64) - self.A)
        return tF.linear(x, self.A + self.dA)


class Flattened(nn.Module):
    """An outer product of two trained vectors, u ⊗ w, written as a multiplication of their broadcasts each flattened
    by a view, as an update to a frozen weight."""

    def __init__(self) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        self.u, self.w = nn.Parameter(torch.randn(4)), nn.Parameter(torch.randn(8))

    def forward(self, x):
        uw = self.u[:, None].expand(4, 8).reshape(32) * self.w[None, :].expand(4, 8).reshape(32)
        return tF.linear(x, self.A + uw.view(4, 8))


@pytest.mark.parametrize("bits", (2, 4, 8))
def test_a_coded_messages_levels_times_their_groups_scale_are_elementwise_through_the_view_that_groups_them(bits):
    """A coded message's decode: unpacked levels [1, 256 // per, per] are flat along their per lanes, and the view to the
    groups [1, 2, 128] merges the lanes with 128 // per bytes into one dim, which no dim of its own can say flat; their
    multiplication by each group's scale [1, 2, 1] is elementwise, since on the grid both share ([1, 2, 128 // per,
    per], `products._cells`) the scale is flat along the bytes and the lanes, the levels along the lanes. 4.12.9's
    check found it at 2 bits (a mul of 256 multiplications), where the levels have fewer entries than the step's, and
    not at 4 and 8."""
    torch.manual_seed(0)
    assert _elementwise(Messages(bits), torch.randn(2, 16, 64), torch.randn(2, 16, 4)) == []


@pytest.mark.parametrize("make, found", ((lambda: Messages(2, "vector"), [{"op": "mul", "count": 1, "macs": 256}]),
                                         (Flattened, [{"op": "mul", "count": 1, "macs": 32}])))
def test_a_product_behind_a_view_that_groups_its_dims_is_found(make, found):
    """The grid that exempts a coded message's decode still finds a product: the 2-bit levels times a trained vector
    of the group's entries, which varies along the lanes the levels are flat along and is flat along the groups, and
    an outer product whose two broadcasts are each flattened by a view."""
    torch.manual_seed(0)
    m = make()
    x = torch.randn(2, 16, m.A.shape[1])
    assert _elementwise(m, x, torch.randn(2, 16, 4)) == found


class MatVec(nn.Module):
    """A layer plus a bias made by a matrix-vector product of its weight and a trained vector (aten.mv)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.w, self.v = nn.Parameter(torch.randn(d, d)), nn.Parameter(torch.randn(d))

    def forward(self, x):
        return x @ self.w + torch.mv(self.w, self.v)


def test_a_bf16_steps_matrix_vector_products_are_priced_as_gemms():
    """aten.mv is a GEMM by its name (training_slowdown.opnames' GEMM_LIKE), as capture.structure counts it, so a bf16
    step with torch.mv is priced at the bf16 GEMM's Definition through the capture's rows and |C|, never as an
    elementwise op, whose price has two Definitions that a GEMM's can't take."""
    from training_slowdown import gates

    torch.manual_seed(0)
    m = MatVec(16).to(torch.bfloat16)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: tF.mse_loss(mm(b[0]), b[1]),
                     (torch.empty(6, 16, dtype=torch.bfloat16), torch.empty(6, 16, dtype=torch.bfloat16)))
    got = products.rows(g, policy.CONSOLE[1])
    mv = [r for r in got["rows"] if r["op"] == "mv"]
    assert mv and all(r["gates_per_mac"] == gates.per_mac("gemm/bf16") for r in mv)
    assert {r["op"] for r in got["rows"]} == {"mm", "mv"} and got["gates"] == products.step_gates(g) > 0


class Packed(nn.Module):
    """A forward code passed through bytes: a layer's outputs, each token's 64 on 2^bits levels by a bf16 step
    (2 bits: their RMS's, 4: their absmax's), the codes passed through bytes (eight to `bits` bytes, unpacked over 8
    lanes and viewed back to the token's 64), decoded (j + 0.5) * step and passed straight through (`scale`); or, the
    control, the levels times a trained vector of the 64 (`vector`)."""

    def __init__(self, bits: int, form: str = "scale") -> None:
        super().__init__()
        self.bits, self.form = bits, form
        self.w = nn.Linear(32, 64, bias=False)
        self.head = nn.Linear(64, 8, bias=False)
        if form == "vector":
            self.v = nn.Parameter(torch.randn(64))
        self.register_buffer("lanes", bits * torch.arange(8))   # each lane's shift, and each byte's
        self.register_buffer("octets", 8 * torch.arange(bits))

    def forward(self, x):
        bits, half = self.bits, 2 ** (self.bits - 1)
        y = self.w(x)
        with torch.no_grad():
            g = y.reshape(-1, 1, 64)
            step = (g.pow(2).mean(-1, keepdim=True).sqrt() * 0.9957 if bits == 2 else
                    g.abs().amax(-1, keepdim=True) * (2.0 / 2 ** bits)).to(torch.bfloat16).float().clamp_min(1e-30)
            c = (torch.floor(g / step).clamp(-half, half - 1) + half).long()
            word = (c.reshape(*c.shape[:-1], -1, 8) << self.lanes).sum(-1)
            packed = ((word.unsqueeze(-1) >> self.octets) & 255).to(torch.uint8)
            word = (packed.long() << self.octets).sum(-1)
            j = ((word.unsqueeze(-1) >> self.lanes) & (2 ** bits - 1)).reshape(c.shape).float() - half
        if self.form == "vector":
            return self.head(y + ((j + 0.5) * self.v).reshape(y.shape))
        return self.head(y + (((j + 0.5) * step).reshape(y.shape) - y).detach())


@pytest.mark.parametrize("bits, form, found", ((2, "scale", []), (4, "scale", []),
                                               (2, "vector", [{"op": "mul", "count": 1, "macs": 2 * 16 * 64}])))
def test_a_code_passed_through_bytes_and_decoded_by_its_step_is_elementwise(bits, form, found):
    """The unpacked levels [32, 1, 8, 8] are flat along their 8 lanes, and the view back to [32, 1, 64] keeps that run
    of the index, however it merges the lanes with the bytes, through the cast and the maps that follow; their
    multiplication by each token's step [32, 1, 1], flat along all 64, is elementwise. 4.12.12's check followed a
    view that regroups dims only where it is the multiplication's own operand, so it found such a decode,
    a map after the view, at both widths. Times a trained vector of the 64, which varies along the lanes the levels
    are flat along, it is an outer product, found."""
    torch.manual_seed(0)
    assert _elementwise(Packed(bits, form), torch.randn(2, 16, 32), torch.randn(2, 16, 8)) == found


class Clipped(nn.Module):
    """A trained 16 x 8 weight of which a layer reads 4 rows, so its gradient is a padded one (`clip`); or, the
    control, a frozen 16 x 8 weight plus the outer product of a trained 4-vector padded to 16 and a trained 8-vector
    (`outer`)."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.form = form
        self.A = nn.Parameter(torch.randn(16, 8), requires_grad=form == "clip")
        if form == "outer":
            self.u, self.w = nn.Parameter(torch.randn(4)), nn.Parameter(torch.randn(8))

    def forward(self, x):
        if self.form == "clip":
            return tF.linear(x, self.A[:4])
        return tF.linear(x, self.A + tF.pad(self.u, (0, 12))[:, None] * self.w[None, :])


@pytest.mark.parametrize("form, found", (("clip", []), ("outer", [{"op": "mul", "count": 1, "macs": 128}])))
def test_a_gradient_scaled_by_its_clip_coefficient_is_elementwise(form, found):
    """clip_grad_norm_ scales each gradient in place by one coefficient, a norm's, which has one entry and so is flat
    along the whole index: it mixes with nothing, whatever the gradient's runs are. 4.12.12's check found an
    adapter's clipped gradients so, each padded by its slice's backward, whose runs it can't tell. A
    padded vector, whose runs it can't tell either, times a trained vector's broadcast is still an outer product."""
    torch.manual_seed(0)

    def step(mm, b, opts):
        for o in opts:
            o.zero_grad()
        tF.mse_loss(mm(b[0]), b[1]).backward()
        torch.nn.utils.clip_grad_norm_([p for p in mm.parameters() if p.requires_grad], 1.0)
        for o in opts:
            o.step()

    m = Clipped(form)
    g = capture_steps([step] * 3, m, (torch.randn(2, 16, 8), torch.randn(2, 16, 4 if form == "clip" else 16)), measured=1,
                      optimizers=torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1))
    assert products.rows(g, policy.CONSOLE[1])["elementwise_products"] == found


# --- an attention call's price -----------------------------------------------------------------------------------------
AH, HEADS, AB, AS = 32, 2, 3, 8
#: a call's head dim, its batch of heads, and what its ops count: one product's MACs, one scaling's (of q or of k),
#: and one op's on the weights (dropout's, a trained mask's add)
AE, AN = AH // HEADS, AB * HEADS
PRODUCT, SCALING, WEIGHTS = AN * AS * AS * AE, AN * AS * AE, AN * AS * AS
#: what each step's calls count, forward and backward, as (products, scalings, ops on the weights): a call's forward
#: is two products and two scalings, its backward four and two; dropout scales the weights twice forward and once
#: backward; a trained mask is added to the scores once; a call under no_grad has no backward
CALLS = {"plain": (6, 4, 0), "dropout": (6, 4, 3), "mask": (6, 4, 1), "module": (6, 4, 0), "no grad": (8, 6, 0),
         "wide": (6, 4, 0), "upcast": (6, 4, 0)}
#: torch's own builtin for the call, bound before any capture
BOUND = torch._C._nn.scaled_dot_product_attention


def by_bound(q, k, v):
    """A function torch hands to a function mode, as it does its own (`handle_torch_function`), which makes the call
    through `BOUND`."""
    if torch.overrides.has_torch_function((q, k, v)):
        return torch.overrides.handle_torch_function(by_bound, (q, k, v), q, k, v)
    return BOUND(q, k, v, is_causal=True)


class Attending(nn.Module):
    """An embedding, q, k and v projections, one attention call and a head. `how`: the call as it is, with dropout,
    with a trained bias as its mask, made by `nn.MultiheadAttention`, made a second time under no_grad, followed by
    a product that the step computes in float32 with autocast turned off around it, or made with q and k cast up to
    float32, which autocast casts back for the kernel. And two that make no call the capture marks: attention by
    hand (`nn.MultiheadAttention` asked for its weights), and the call made through `BOUND` inside `by_bound`."""

    def __init__(self, how: str) -> None:
        super().__init__()
        self.how = how
        self.emb = nn.Embedding(V, AH)
        self.q, self.k, self.v, self.head = (nn.Linear(AH, AH, bias=False) for _ in range(4))
        if how == "mask":
            self.bias = nn.Parameter(torch.zeros(1, 1))
        if how in ("module", "weights"):
            self.att = nn.MultiheadAttention(AH, HEADS, bias=False, batch_first=True)
        if how == "wide":
            self.side = nn.Parameter(torch.randn(AH, AH))

    def forward(self, ids):
        x = self.emb(ids)
        if self.how in ("module", "weights"):
            return self.head(self.att(x, x, x, need_weights=self.how == "weights")[0])
        b, s, d = x.shape
        q, k, v = (f(x).view(b, s, HEADS, d // HEADS).transpose(1, 2) for f in (self.q, self.k, self.v))
        if self.how == "upcast":
            q, k = q.float(), k.float()
        if self.how == "bound":
            a = by_bound(q, k, v)
        elif self.how == "mask":
            a = tF.scaled_dot_product_attention(q, k, v, attn_mask=self.bias.expand(s, s))
        else:
            a = tF.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.5 * (self.how == "dropout"))
        if self.how == "no grad":
            with torch.no_grad():
                again = tF.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a + again
        a = a.transpose(1, 2).reshape(b, s, d)
        if self.how == "wide":
            with torch.autocast("cpu", enabled=False):
                a = a + (a.float() @ self.side).to(a.dtype)
        return self.head(a)


def attending(how: str, autocast: bool, marked: bool = True):
    """The capture of `Attending(how)`'s step, its forward under bf16 autocast or in float32; `marked` False: as a
    capture that states no call takes it, every op in the dtype it was traced in."""
    def loss(m, b):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out = m(b[0])
        return tF.cross_entropy(out.float().flatten(0, 1), b[1].flatten())

    torch.manual_seed(0)
    m = Attending(how)
    with pytest.MonkeyPatch.context() as patch:
        if not marked:
            patch.setattr(trace, "_marked_attention", contextlib.nullcontext)
        return capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), loss,
                            (torch.empty(AB, AS, dtype=torch.long), torch.empty(AB, AS, dtype=torch.long)))


def _priced(g) -> list[dict]:
    return [products.priced(r, products.CAPTURE) for r in products.raw_on_dial(g)["rows"]]


@pytest.mark.parametrize("how", CALLS)
def test_a_bf16_calls_ops_are_priced_at_16_bits_and_every_other_op_as_traced(how):
    """A step under bf16 autocast: torch's math kernel, which the capture runs, computes the call's products in
    float32 from operands it casts up, and the rows state the bf16 the call was made in (`attention`), which is
    their price; no other row states a call, and every other row is priced as a capture that marks no call prices
    it, a float32 product inside autocast at float32. The step's |C| is under that capture's by exactly the call's
    ops (its products, its scalings, dropout's and a trained mask's add, forward and backward: the ops the counter
    counts in a call, `flops.Counter._attention`) at the difference of the two dtypes' prices, and by nothing else;
    its MACs are the same. Whatever makes the call: `nn.MultiheadAttention`, and a call that has no backward. A
    call whose q and k reach it in float32 is marked bf16 too, the dtype autocast hands the kernel."""
    from training_slowdown import gates

    g, unmarked = attending(how, True), attending(how, True, marked=False)
    raw = products.raw_on_dial(g)["rows"]
    assert {(r["op"], tuple(r["dtypes"]), r["attention"]) for r in raw} == {
        ("mm", ("bfloat16", "bfloat16"), None), ("bmm", ("float32", "float32"), "bfloat16"),
        *([("mm", ("float32", "float32"), None)] if how == "wide" else [])}
    assert sum(r["count"] * math.prod(r["dims"].values()) for r in raw if r["attention"]) == CALLS[how][0] * PRODUCT
    bf16, f32 = gates.per_mac("gemm/bf16"), gates.per_mac("gemm/f32")
    priced, before = _priced(g), _priced(unmarked)
    assert {(r["op"], r["gates_per_mac"]) for r in priced} == {
        ("mm", bf16), ("bmm", bf16), *([("mm", f32)] if how == "wide" else [])}
    assert [r for r in priced if r["op"] != "bmm"] == [r for r in before if r["op"] != "bmm"]
    assert {r["gates_per_mac"] for r in before if r["op"] == "bmm"} == {f32}
    n_products, n_scalings, n_weights = CALLS[how]
    elementwise = gates.per_mac("elementwise/f32") - gates.per_mac("elementwise/bf16")
    assert products.step_gates(unmarked) - products.step_gates(g) == (
        n_products * PRODUCT * (f32 - bf16) + (n_scalings * SCALING + n_weights * WEIGHTS) * elementwise)
    assert (g.macs, g.elementwise_macs, g.c_macs) == (unmarked.macs, unmarked.elementwise_macs, unmarked.c_macs)
    assert products.rows(g)["gates"] == products.step_gates(g)


@pytest.mark.parametrize("how", ("plain", "dropout", "mask"))
def test_a_float32_calls_ops_stay_at_32_bits(how):
    """A step in float32: the call is made in float32, its rows say so, and its price is the traced one: the rows
    and the step's |C| are those of a capture that marks no call."""
    from training_slowdown import gates

    g, unmarked = attending(how, False), attending(how, False, marked=False)
    raw = products.raw_on_dial(g)["rows"]
    assert {(r["op"], tuple(r["dtypes"]), r["attention"]) for r in raw} == {
        ("mm", ("float32", "float32"), None), ("bmm", ("float32", "float32"), "float32")}
    assert _priced(g) == _priced(unmarked) and {r["gates_per_mac"] for r in _priced(g)} == {gates.per_mac("gemm/f32")}
    assert products.step_gates(g) == products.step_gates(unmarked) > 0


@pytest.mark.parametrize("how, dtype", [("weights", "bfloat16"), ("bound", "float32")])
def test_attention_the_capture_marks_no_call_of_is_priced_as_traced(how, dtype):
    """Under bf16 autocast, attention that is no call the capture marks: written by hand (`nn.MultiheadAttention`
    asked for its weights, a `bmm` and a softmax), whose products autocast runs in bf16; and a call made through a
    builtin bound before the capture inside a function torch hands to the capture's mode, which the mode does not
    see there, so its products are the math kernel's float32. No row states a call, and the step is priced as a
    capture that marks none prices it."""
    g, unmarked = attending(how, True), attending(how, True, marked=False)
    raw = products.raw_on_dial(g)["rows"]
    assert {r["attention"] for r in raw} == {None}
    assert {tuple(r["dtypes"]) for r in raw if r["op"] == "bmm"} == {(dtype, dtype)}
    assert _priced(g) == _priced(unmarked) and products.step_gates(g) == products.step_gates(unmarked)


def test_a_call_that_returns_in_another_dtype_than_it_is_marked_is_refused(monkeypatch):
    """The marker's dtype is the one the call computes in, and either kernel returns in it: a call whose output is
    in another dtype than its marker would have its ops priced in a dtype it did not compute in, and the capture is
    refused, naming the call and both dtypes."""
    monkeypatch.setattr(trace, "_computes_in", lambda query: torch.float16)
    with pytest.raises(CaptureRefused) as e:
        attending("plain", True)
    assert (e.value.reason, e.value.op) == ("trace", "scaled_dot_product_attention")
    assert "an attention call marked torch.float16 returned torch.bfloat16" in str(e.value)
