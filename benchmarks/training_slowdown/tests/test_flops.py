import itertools

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from conftest import D, PER_TOKEN, VOCAB, Tiny, loss_of
from training_slowdown import flops, opnames
from training_slowdown.annotate import annotate
from training_slowdown.flops import Counter, flags, forward_macs_per_token, gather_selects, guard_exits, leaked, mux_tree

IDX = torch.randint(0, VOCAB, (3, 8), generator=torch.Generator().manual_seed(3))  # 24 tokens: no activation is weight-shaped
TOKENS = IDX.numel()
LOOKUP = mux_tree(VOCAB, 32) * D  # the updated tied table's lookup per token: 58 multiplexers per word
SCALED = 2 * D  # attention's scalings of q and of kᵀ per token, each by a value made in the step


def train(m, theta, n=3, params=None, lr=0.1, body=None):
    c = Counter(theta)
    c.register(m.state_dict(keep_vars=True))
    opt = torch.optim.SGD(params if params is not None else m.parameters(), lr=lr)
    with c:
        for _ in range(n):
            c.begin_step()
            if body:
                body(m)
            else:
                loss_of(m, IDX).backward()
                opt.step()
                opt.zero_grad()
            c.end_step()
    c.finish()
    return c.record()


def test_counts_and_updated_structure(tiny):
    m, theta = tiny
    rec = train(m, theta)
    s0, s1 = rec["steps"][0], rec["steps"][1]
    assert s0["tokens"] == s1["tokens"] == TOKENS
    assert s0["by"]["fwd/theta"] == (PER_TOKEN + SCALED) * TOKENS  # step 0 reads θ itself
    assert "fwd/updated" not in s0["by"]
    assert s1["by"]["fwd/updated"] == (PER_TOKEN + LOOKUP) * TOKENS
    assert sorted(s1["weights"]) == ["emb", "w1.weight", "w2.weight"]
    ann = annotate(rec)
    st = ann.steps[1]
    assert st.updated_per_token == PER_TOKEN + LOOKUP
    assert st.depth == LOOKUP + 2 * D * D  # the head, after the updated table's lookup, w1 and w2
    assert [u for u, _ in st.profile] == [0, LOOKUP, LOOKUP + D * D, LOOKUP + 2 * D * D]
    assert ann.macs == rec["totals"]["macs"] == sum(s["macs"] for s in rec["steps"])
    assert st.bits == 32


def test_a_gathers_price_is_the_ands_of_the_circuits_gather():
    """`gather_selects` against `verity.circuits.boolean.gather` on one-bit words: one tree by the token's 32 bits up to a
    chunk, and past it the registry's chunks by the low 10 bits, then their results by the high 22."""
    from verity.circuits.boolean.forms import Circuit
    from verity.circuits.boolean.gather import gather

    def ands(n):
        C = Circuit()
        idx = C.input("idx", 32)
        words = [C.input(f"w{i}", 1) for i in range(n)]
        if n <= 1024:
            gather(C, idx, words, 0)
        else:
            chunks = [gather(C, idx[:10], words[i:i + 1024], 0) for i in range(0, n, 1024)]
            gather(C, idx[10:], chunks, 0)
        return C.kind.count("and")

    for n in (1, 2, 3, 5, 32, 1000, 1024, 2048, 2051):
        assert gather_selects(n) == ands(n), n
    assert gather_selects(VOCAB) == mux_tree(VOCAB, 32) == 58
    assert gather_selects(50304) == 50325  # GPT-2's table: about one multiplexer per row


def test_frozen_weights_stay_theta(tiny):
    m, theta = tiny
    rec = train(m, theta, params=[m.w2.weight])
    st = annotate(rec).steps[2]
    assert st.updated_macs == D * D * TOKENS
    assert list(st.weights) == ["w2.weight"]
    assert st.profile == ((0, D * D * TOKENS),)  # w1 is θ's: nothing updated upstream of w2
    assert st.depth == 0


def test_an_update_of_the_tied_table_alone_is_read_at_its_lookup(tiny):
    """The lookup reads the updated table at position 0, an updated node priced as the circuit's gather,
    so the tied head has it upstream; w1 and w2 are θ's and add nothing."""
    m, theta = tiny
    rec = train(m, theta, params=[m.emb])
    st = annotate(rec).steps[2]
    assert list(st.weights) == ["emb"]
    assert st.updated_macs == (LOOKUP + VOCAB * D) * TOKENS
    assert st.profile == ((0, LOOKUP * TOKENS), (LOOKUP, VOCAB * D * TOKENS))
    assert st.depth == LOOKUP


def test_new_weights_are_updated(tiny):
    m, theta = tiny
    for p in m.parameters():
        p.requires_grad_(False)
    a = torch.zeros(4, D, requires_grad=True)
    b = torch.zeros(D, 4, requires_grad=True)
    with torch.no_grad():
        a.normal_()
    opt = torch.optim.SGD([a, b], lr=0.1)

    def body(m):
        x = F.embedding(IDX, m.emb)
        y = x + (x @ a.t()) @ b.t()
        F.cross_entropy(F.linear(y, m.emb).reshape(-1, VOCAB), IDX.reshape(-1)).backward()
        opt.step()
        opt.zero_grad()

    rec = train(m, theta, body=body)
    st = annotate(rec).steps[1]
    assert st.updated_macs == 2 * 4 * D * TOKENS  # the factors' two products; the head is θ's
    assert sorted(st.weights) == ["new/0", "new/1"]


def test_a_value_carried_across_steps_is_a_weight(tiny):
    m, theta = tiny
    for p in m.parameters():
        p.requires_grad_(False)
    memory = []

    def body(m):
        x = F.embedding(IDX, m.emb).reshape(-1, D)
        if memory:
            x = x @ memory[0]
        memory[:] = [(x.t() @ x)[:D, :D] / TOKENS]

    rec = train(m, theta, body=body)
    assert "fwd/updated" not in rec["steps"][0]["by"]
    assert rec["steps"][1]["by"]["fwd/updated"] == D * D * TOKENS


def test_a_gradient_by_hand_is_a_weight(tiny):
    m, theta = tiny
    for p in m.parameters():
        p.requires_grad_(False)

    def body(m):
        x = F.embedding(IDX, m.emb).reshape(-1, D)
        dw = x.t() @ x  # a weight-shaped sum over tokens, computed in the forward
        w2 = m.w1.weight - 0.01 * dw
        _ = x @ w2.t()

    rec = train(m, theta, n=1, body=body)
    by = rec["steps"][0]["by"]
    assert by["fwd/updated"] == D * D * TOKENS
    assert by["fwd/act"] == D * D * TOKENS


def test_a_stream_batch_is_token_data_through_any_op_and_counts_once():
    """`Counter.batch`: an index lookup (`embed[x]`) beside a one-token embedding, the targets read by a gather, and a
    batch fetched in one step and read in the next."""
    torch.manual_seed(0)
    theta = {"w": torch.randn(D, D), "embed": torch.randn(VOCAB, D)}
    w, embed = (torch.nn.Parameter(theta[k].clone()) for k in ("w", "embed"))
    c = Counter(theta, digests=False)
    c.register({"w": w, "embed": embed})
    gen = torch.Generator().manual_seed(1)
    with c, torch.no_grad():
        c.begin_step()
        w.add_(0.01)
        b = c.batch(torch.randint(0, VOCAB, (3, 9), generator=gen))
        x, y = b[:, :-1], b[:, 1:]
        F.embedding(x[:, :1], embed)
        torch.log_softmax(embed[x] @ w.T @ embed.T, -1).gather(-1, y[..., None])
        ahead = c.batch(torch.randint(0, VOCAB, (3, 9), generator=gen))
        c.end_step()
        embed[ahead[:, :-1]] @ w.T
        c.end_step()
    c.finish()
    s0, s1 = c.record()["steps"]
    assert s0["tokens"] == 3 * 9  # every position an op reads: x's 8 and the gather's last target
    assert s1["tokens"] == 3 * 8
    assert s0["by"]["fwd/updated"] == s1["by"]["fwd/updated"] == 3 * 8 * D * D


def test_disjoint_reads_of_one_batch_add_up_and_overlapping_ones_count_each_position_once():
    """training_slowdown 4.0's counter: a micro-batched step (`x = rows[i:i + m]`, `fwd(x[:, :-1])`) consumes every micro-batch's
    tokens, where the rule before counted one at the largest read; windows overlapping by a token can't chain into one
    read either."""
    torch.manual_seed(0)
    theta = {"w": torch.randn(D, D), "embed": torch.randn(VOCAB, D)}
    w, embed = (torch.nn.Parameter(theta[k].clone()) for k in ("w", "embed"))
    c = Counter(theta, digests=False)
    c.register({"w": w, "embed": embed})
    gen = torch.Generator().manual_seed(1)
    with c, torch.no_grad():
        c.begin_step()
        w.add_(0.01)
        rows = c.batch(torch.randint(0, VOCAB, (4, 9), generator=gen))
        for i in range(0, 4, 2):
            x = rows[i:i + 2]
            F.embedding(x[:, :-1], embed) @ w
        c.end_step()
        stream = c.batch(torch.randint(0, VOCAB, (20,), generator=gen))
        for lo, hi in ((0, 8), (7, 15), (14, 20), (0, 20)):
            F.embedding(stream[lo:hi], embed) @ w
        c.end_step()
    c.finish()
    s0, s1 = c.record()["steps"]
    assert s0["tokens"] == 4 * 8 and s0["by"]["fwd/updated"] == 4 * 8 * D * D
    assert s1["tokens"] == 20


def one_step(body) -> dict:
    """The record of one step that moves `w` off θ, then runs `body(w, embed)`."""
    torch.manual_seed(0)
    theta = {"w": torch.randn(D, D), "embed": torch.randn(VOCAB, D)}
    w, embed = (torch.nn.Parameter(theta[k].clone()) for k in ("w", "embed"))
    c = Counter(theta, digests=False)
    c.register({"w": w, "embed": embed})
    with c, torch.no_grad():
        c.begin_step()
        w.add_(0.01)
        body(w, embed)
        c.end_step()
    c.finish()
    return c.record()


def embedded(embed):
    return F.embedding(IDX, embed)


def test_a_product_written_through_any_op_is_an_updated_node():
    """A quotient by the reciprocal, an in-place accumulating matmul, a sum then exp, and a distance each combine the
    step's tokens with the updated weight, at the matmul's MACs."""
    cases = {"div": lambda w, e: (embedded(e).unsqueeze(-1) / w.T.reciprocal()).sum(-2),
             "addmm_": lambda w, e: torch.zeros(TOKENS, D).addmm_(embedded(e).reshape(-1, D), w.T),
             "add_exp": lambda w, e: torch.exp(embedded(e).unsqueeze(-1) + w.T).sum(-2),
             "cdist": lambda w, e: torch.cdist(embedded(e).reshape(-1, D), w)}
    for name, body in cases.items():
        (s,) = one_step(body)["steps"]
        assert s["by"].get("fwd/updated") == TOKENS * D * D, (name, s["by"])
    (s,) = one_step(lambda w, e: F.layer_norm(embedded(e), (D,), weight=w[0]))["steps"]
    assert s["by"]["fwd/updated"] == TOKENS * D  # a trained gain


def test_a_token_rows_work_is_per_token():
    """An operand reshaped to one value per row has no more rows than the step has tokens."""
    rec = one_step(lambda w, e: embedded(e).reshape(-1, 1) @ w[:1])
    (s,) = rec["steps"]
    assert rec["structures"][s["struct"]]["updated"] == [[D * D, 0, TOKENS * D * D, TOKENS, 32]]


def test_ops_the_counter_cannot_price_are_recorded():
    rec = one_step(lambda w, e: F.bilinear(embedded(e), embedded(e), w.unsqueeze(0)))
    assert list(rec["unknown_ops"]) == ["aten._trilinear.default"] and rec["foreign_ops"] == {}


def test_token_data_read_out_of_torch_is_counted_in_its_step():
    guard_exits()
    rec = one_step(lambda w, e: (embedded(e).numpy(), embedded(e).sum(-1).tolist(), w.detach().numpy()))
    (s,) = rec["steps"]
    assert s["exited"] == TOKENS * D + TOKENS and leaked(s)
    (s,) = one_step(lambda w, e: embedded(e).sum((1, 2)).tolist())["steps"]
    assert s["exited"] == 3 and not leaked(s)


def test_a_program_made_token_source_is_one_source_by_tensor_not_by_address():
    """Tokens keyed by an index tensor's address would count a fresh index tensor per chunk at a freed one's address
    once, so one score per chunk read out would be a value per token. Each new tensor over the
    same memory counts (here a numpy buffer refilled per chunk, every chunk at one address), as does a buffer an op
    rewrites; the same source read again, or through a view, doesn't."""
    guard_exits()
    chunks, per = 4, 6
    gen = torch.Generator().manual_seed(5)
    data = torch.randint(0, VOCAB, (chunks, per), generator=gen).numpy()

    def fresh(w, e):
        buf = data[0].copy()
        for k in range(chunks):
            buf[:] = data[k]
            (F.embedding(torch.from_numpy(buf), e) @ w).sum().tolist()

    (s,) = one_step(fresh)["steps"]
    assert s["tokens"] == chunks * per and s["exited"] == chunks and not leaked(s)

    def rewritten(w, e):
        buf = torch.zeros(per, dtype=torch.long)
        for k in range(chunks):
            buf.copy_(torch.from_numpy(data[k]))
            F.embedding(buf, e) @ w

    (s,) = one_step(rewritten)["steps"]
    assert s["tokens"] == chunks * per

    def reread(w, e):
        idx = torch.from_numpy(data[0])
        F.embedding(idx, e) @ w
        F.embedding(idx[: per // 2], e) @ w
        F.embedding(idx, e) @ w

    (s,) = one_step(reread)["steps"]
    assert s["tokens"] == per


def test_zero_mac_lookups_raise_the_steps_tokens_and_its_read_out_allowance():
    """Lookups of θ's table by fresh index tensors cost nothing but raise the step's tokens, so a value per token read
    out stays under the counter's allowance."""
    guard_exits()

    def run(w, e):
        F.embedding(IDX, e) @ w
        for _ in range(3):
            F.embedding(torch.randint(0, VOCAB, (TOKENS,)), e)
        F.embedding(IDX, e).sum(-1).numpy()

    (s,) = one_step(run)["steps"]
    assert s["tokens"] == 4 * TOKENS and s["exited"] == TOKENS and not leaked(s)


def test_writes_through_data_are_seen(tiny):
    m, theta = tiny

    def body(m):
        m.w1.weight.data.add_(1.0)
        F.linear(F.embedding(IDX, m.emb), m.w1.weight)

    rec = train(m, theta, n=1, body=body)
    assert rec["steps"][0]["by"]["fwd/updated"] == D * D * TOKENS
    assert rec["unseen_writes"] == []


def tabled():
    """`Tiny` with two buffers its θ doesn't hold: a table outside the state dict and a zero matrix in it."""
    torch.manual_seed(0)
    m = Tiny()
    theta = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.register_buffer("table", torch.linspace(0.5, 1.5, D), persistent=False)
    m.register_buffer("mean", torch.zeros(D, D))
    return m, theta


def scaled(m, t=None):
    """A forward whose embedded tokens are scaled by the table: D MACs a token, then the head's."""
    return F.linear(F.embedding(IDX, m.emb) * m.table, m.emb)


def counted(m, theta, body, n=2, buffers=True):
    c = Counter(theta)
    c.register(m.state_dict(keep_vars=True), dict(m.named_buffers()) if buffers else None)
    with c:
        for t in range(n):
            c.begin_step()
            body(m, t)
            c.end_step()
    c.finish()
    return c, c.record()


def test_a_models_buffers_that_theta_lacks_are_constants_and_a_product_with_one_is_thetas():
    """The model's buffers θ doesn't hold, in the state dict or not, are held under their names beside θ, which stays
    the caller's: a product of token data with a table is θ's, in every step, and the table is no weight the program
    made. A counter given no buffers takes the table for one, and calls the same product updated from the first
    step on; the step's MACs are the same either way."""
    m, theta = tabled()
    c, rec = counted(m, theta, scaled)
    assert c.constants == {"table", "mean"} and set(c.theta_cpu) == set(theta) | c.constants
    assert set(theta) == {"emb", "w1.weight", "w2.weight"} and c.weights["table"] is m.table
    for s in rec["steps"]:
        assert s["by"] == {"fwd/theta": (D + VOCAB * D) * TOKENS} and s["weights"] == {}
    assert rec["new_weights"] == [] and rec["unseen_writes"] == []

    m, theta = tabled()
    c, without = counted(m, theta, scaled, buffers=False)
    assert c.constants == set() and without["new_weights"] == ["new/0"]
    for s in without["steps"]:
        assert s["by"] == {"fwd/theta": VOCAB * D * TOKENS, "fwd/updated": D * TOKENS} and list(s["weights"]) == ["new/0"]
    assert without["totals"]["macs"] == rec["totals"]["macs"]


def test_a_buffer_a_step_writes_is_an_updated_weight_from_that_write_on():
    """A constant is checked as a θ weight is: once a step writes the table through torch, a product with it is an
    updated weight's, listed under the buffer's own name, in that step and after."""
    def body(m, t):
        if t == 1:
            m.table.mul_(2.0)
        scaled(m)

    c, rec = counted(*tabled(), body, n=3)
    first, *later = rec["steps"]
    assert first["by"] == {"fwd/theta": (D + VOCAB * D) * TOKENS} and first["weights"] == {}
    for s in later:
        assert s["by"]["fwd/updated"] == D * TOKENS and s["by"]["fwd/theta"] == VOCAB * D * TOKENS
        assert list(s["weights"]) == ["table"] and s["shapes"] == {"table": ["float32", [D]]}
    assert rec["new_weights"] == [] and rec["unseen_writes"] == []


def test_a_write_into_a_buffer_that_the_dispatcher_does_not_see_is_an_unseen_write():
    """As for a weight of θ: a buffer the step read at its registered value and that has another at the step's end,
    with no op having written it, was written where the counter can't see (here through numpy)."""
    def body(m, t):
        scaled(m)
        if t == 1:
            m.table.numpy()[0] += 1.0

    _, rec = counted(*tabled(), body, n=2)
    assert rec["unseen_writes"] == [(1, "table")]


def test_no_tensor_the_counter_has_not_seen_is_taken_for_a_buffer():
    """A tensor made before the step that equals a buffer of the model, an optimizer's zero state beside a zero
    average, is not that buffer: it is a weight the program made, and the buffer's name stays the buffer's. A tensor
    that holds a weight of θ is still that weight."""
    m, theta = tabled()
    zeros, twin = torch.zeros(D, D), theta["w1.weight"].clone()

    def body(m, t):
        x = F.embedding(IDX, m.emb).reshape(-1, D)
        _ = x @ zeros, x @ twin

    c, rec = counted(m, theta, body, n=1)
    assert rec["new_weights"] == ["new/0"] and c.new_tensors["new/0"] is zeros and c.weights["mean"] is m.mean
    assert rec["steps"][0]["by"] == {"fwd/updated": D * D * TOKENS, "fwd/theta": D * D * TOKENS}


def test_paused_work_is_not_counted(tiny):
    m, theta = tiny
    c = Counter(theta)
    c.register(m.state_dict(keep_vars=True))
    with c:
        with c.pause():
            m(IDX)
        c.begin_step()
        m(IDX)
        c.end_step()
    assert c.setup.macs == 0
    assert c.steps[0].by["fwd/theta"] == (PER_TOKEN + SCALED) * TOKENS


HEADS, L, S, E = 2, 5, 6, 4
#: who needs a gradient among q, k and v
ASKED = [(q, k, v) for q in (False, True) for k in (False, True) for v in (False, True)]


def counted_attention(body, math: bool, monkeypatch) -> tuple[dict, int]:
    """One step's record of `body(c, w)`, with `w` an updated weight of θ, under the math attention kernel or the one
    torch picks, and how many fused kernels the counter saw."""
    seen = []
    inner = Counter._attention
    monkeypatch.setattr(Counter, "_attention", lambda self, func, *a: seen.append(str(func)) or inner(self, func, *a))
    torch.manual_seed(0)
    theta = {"w": torch.randn(E, E)}
    w = torch.nn.Parameter(theta["w"].clone())
    c = Counter(theta, digests=False)
    c.register({"w": w})
    with c:
        c.begin_step()
        with torch.no_grad():
            w.add_(0.01)
        if math:
            with sdpa_kernel([SDPBackend.MATH]):
                body(c, w)
        else:
            body(c, w)
        c.end_step()
    c.finish()
    rec = c.record()
    (step,) = rec["steps"]
    keep = ("tokens", "macs", "matmul_macs", "elementwise_macs", "approx_macs", "by", "shapes", "device_macs")
    return {k: step[k] for k in keep} | rec["structures"][step["struct"]] | {"weights": sorted(step["weights"])}, len(seen)


def tokens(c: Counter, *shape) -> torch.Tensor:
    return c.batch(torch.randn(*shape))


@pytest.mark.parametrize("asked", ASKED, ids=lambda a: "".join(n for n, x in zip("qkv", a) if x) or "none")
def test_a_fused_attention_is_counted_as_the_math_kernels_ops_for_the_gradients_asked(asked, monkeypatch):
    """A step's record under the fused attention kernel is its record under the math kernel, the one a trace runs:
    every count, the op log's rows and the updated nodes. In the forward two scalings and two products; in the
    backward, by which of q, k and v need a gradient, v's product, the weights' where q or k asks, and for each of q
    and k its product and its scaling."""
    n = 3 * HEADS

    def body(c, w):
        q, k, v = (tokens(c, 3, HEADS, t, E).requires_grad_(g) for t, g in zip((L, S, S), asked))
        out = F.scaled_dot_product_attention(q @ w.detach(), k, v)
        if any(asked):
            out.sum().backward()

    fused, kernels = counted_attention(body, False, monkeypatch)
    assert kernels == 1 + any(asked) and (fused, 0) == counted_attention(body, True, monkeypatch)
    gq, gk, gv = asked
    bwd = {tuple(row[2:4]): row[5] for row in fused["log"] if row[0] == "bwd"}
    scalings = {("mul", f"[[3, {HEADS}, {L}, {E}]]"): n * L * E, ("mul", f"[[3, {HEADS}, {E}, {S}]]"): n * E * S}
    products = {("bmm", f"[[{n}, {S}, {L}], [{n}, {L}, {E}]]"): gv, ("bmm", f"[[{n}, {L}, {E}], [{n}, {E}, {S}]]"): gq or gk,
                ("bmm", f"[[{n}, {L}, {S}], [{n}, {S}, {E}]]"): gq, ("bmm", f"[[{n}, {E}, {L}], [{n}, {L}, {S}]]"): gk}
    asked_scalings = dict(zip(scalings.items(), (gq, gk)))
    assert {k: v for k, v in bwd.items() if k[0] == "bmm"} == {k: n * L * S * E for k, on in products.items() if on}
    assert {k: v for k, v in bwd.items() if k[0] == "mul" and k in scalings} == {k: v for (k, v), on in asked_scalings.items() if on}
    fwd = {tuple(row[2:4]): row[5] for row in fused["log"] if row[0] != "bwd"}
    assert scalings.items() <= fwd.items()
    assert fwd["bmm", f"[[{n}, {L}, {E}], [{n}, {E}, {S}]]"] == fwd["bmm", f"[[{n}, {L}, {S}], [{n}, {S}, {E}]]"] == n * L * S * E


VARIANTS = {
    "causal": lambda c, w: F.scaled_dot_product_attention(*(tokens(c, 3, HEADS, L, E) @ w for _ in "qkv"), is_causal=True),
    "a scale": lambda c, w: F.scaled_dot_product_attention(tokens(c, 3, HEADS, L, E) @ w, tokens(c, 3, HEADS, S, E),
                                                           tokens(c, 3, HEADS, S, E), scale=0.3),
    "three dims": lambda c, w: F.scaled_dot_product_attention(tokens(c, 3, L, E) @ w, tokens(c, 3, S, E), tokens(c, 3, S, E)),
    "grouped queries": lambda c, w: F.scaled_dot_product_attention(tokens(c, 3, 4, L, E), tokens(c, 3, 2, S, E) @ w,
                                                                   tokens(c, 3, 2, S, E), enable_gqa=True),
    "a bool mask": lambda c, w: F.scaled_dot_product_attention(tokens(c, 3, HEADS, L, E), tokens(c, 3, HEADS, S, E) @ w,
                                                               tokens(c, 3, HEADS, S, E), attn_mask=torch.ones(L, S).tril() > 0),
    # an updated weight as the mask: its sum with the scores is an updated node of its own
    "an updated mask": lambda c, w: F.scaled_dot_product_attention(tokens(c, 3, HEADS, E, E), tokens(c, 3, HEADS, E, E),
                                                                   tokens(c, 3, HEADS, E, E), attn_mask=w.detach()),
    # an updated weight as the queries, attending over token data: the scores are the updated node
    "updated queries": lambda c, w: F.scaled_dot_product_attention(w.detach().expand(3, HEADS, E, E), tokens(c, 3, HEADS, S, E),
                                                                   tokens(c, 3, HEADS, S, E)),
    # weights from an updated weight alone, applied to token data: the second product is the updated node
    "updated weights": lambda c, w: F.scaled_dot_product_attention(w.detach().expand(3, HEADS, E, E), w.detach().expand(3, HEADS, E, E),
                                                                   tokens(c, 3, HEADS, E, E)),
    "no token": lambda c, w: F.scaled_dot_product_attention(*(w.detach().expand(3, HEADS, E, E) for _ in "qkv")),
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_fused_attentions_record_is_the_math_kernels_whatever_its_operands_are(variant, monkeypatch):
    """The same record under either kernel where the operands' provenance differs (token data, an updated weight, both),
    with a mask, a scale, three dims and grouped queries: each op inside the kernel is classified by its own operands,
    as the math kernel's ops are."""
    fused, kernels = counted_attention(VARIANTS[variant], False, monkeypatch)
    assert kernels == 1 and (fused, 0) == counted_attention(VARIANTS[variant], True, monkeypatch)
    kinds = {row[1] for row in fused["log"]}
    assert ("updated" in kinds) == (variant not in ("no token",)), kinds
    if variant == "an updated mask":
        assert ["fwd", "updated", "add", f"[[3, {HEADS}, {E}, {E}], [{E}, {E}]]", 1, 3 * HEADS * E * E] in fused["log"]


def test_a_derived_operand_inside_a_fused_kernel_is_listed_without_a_digest():
    """Attention weights computed from gradients alone and applied to token data: the second product is an updated node
    whose weight operand is derived. Under the math kernel that operand is a tensor, digested; a fused kernel never
    holds it, so the step lists it at its shape with no digest, and still counts as reading a changed weight."""
    def step(math):
        w = torch.nn.Parameter(torch.randn(3, HEADS, E, E))
        c = Counter()
        with c:
            c.begin_step()
            (w * w).sum().backward()
            with sdpa_kernel([SDPBackend.MATH]) if math else torch.no_grad():
                F.scaled_dot_product_attention(w.grad, w.grad, tokens(c, 3, HEADS, E, E))
            c.end_step()
        c.finish()
        (s,) = c.record()["steps"]
        return s

    fused, math = step(False), step(True)
    assert fused["shapes"] == math["shapes"] == {"new/0": ["float32", [3, HEADS, E, E]],
                                                 "derived/0": ["float32", [3 * HEADS, E, E]]}
    assert fused["weights"]["derived/0"] == [""] and all(len(d) == 16 for d in (
        fused["weights"]["new/0"][0], math["weights"]["new/0"][0], math["weights"]["derived/0"][0]))
    assert fused["by"] == math["by"] and fused["by"]["fwd/updated"] == 3 * HEADS * E * E * E


def test_a_kernels_dropout_and_its_word_on_the_gradients_asked_are_read():
    """What only the CUDA kernels take, read from the kernel's own operands by its schema's names: dropout inside
    attention (two scalings of the weights in the forward, one in the backward where q or k asks), and the
    memory-efficient kernel's `grad_input_mask`, its word on which of q, k, v and the bias the step asks for, which
    stands where the tensors' `requires_grad` says more. Against the math kernel's record of the same attention."""
    aten, n = torch.ops.aten, 3 * HEADS
    q, k, v = (torch.randn(3, HEADS, t, E, requires_grad=True) for t in (L, S, S))
    bias = torch.zeros(L, S)

    def kernel(asked):
        c = Counter(digests=False)
        tags = {id(t): flops.NONE for t in (q, k, v, bias)}
        c._attention(aten._scaled_dot_product_efficient_attention.default, (q, k, v, bias, True, 0.5), {}, tags, False)
        if any(asked):
            c._attention(aten._scaled_dot_product_efficient_attention_backward.default,
                         (q, q, k, v, bias, q, q, q, q, 0.5, asked), {}, tags, True)
        return c.setup

    def math(asked):
        c = Counter(digests=False)
        ins = [t.detach().requires_grad_(g) for t, g in zip((q, k, v, bias), asked)]
        with c, sdpa_kernel([SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(*ins[:3], attn_mask=ins[3], dropout_p=0.5)
            if any(asked):
                out.sum().backward()
        return c.setup

    for asked in ([True, True, True, False], [False, False, True, False], [True, False, False, False],
                  [False, False, False, True], [False, False, False, False]):
        a, b = kernel(asked), math(asked)
        assert a.log == b.log and (a.matmul_macs, a.elementwise_macs, a.by) == (b.matmul_macs, b.elementwise_macs, b.by), asked
    assert kernel([True] * 4).elementwise_macs == n * (2 * (L * E + E * S) + 3 * L * S)


def test_every_fused_attention_kernel_names_the_operands_the_counter_reads():
    """`Counter._attention` reads a kernel's operands by its schema's names. Every fused kernel torch has names q, k and
    v as it reads them, and dropout, a mask and the gradients asked where it takes them: a kernel torch adds, or an
    operand it renames, fails here. The kernels of devices no job runs on (another backend's, MPS's) are none of them:
    a run that reaches one is invalid, as for any op named for products that the counter has no count of."""
    read = {"dropout_p", "attn_mask", "attn_bias", "grad_input_mask"}
    names = {str(p).removeprefix("aten._scaled_dot_product_"): {a.name for o in p.overloads()
                                                                for a in getattr(p, o)._schema.arguments}
             for p in flops.FUSED_ATTENTION}
    assert all({"query", "key", "value"} <= n for n in names.values())
    assert {k: sorted(n & read) for k, n in names.items()} == {
        "flash_attention": ["dropout_p"], "flash_attention_backward": ["dropout_p"],
        "flash_attention_for_cpu": ["attn_mask", "dropout_p"],
        "flash_attention_for_cpu_backward": ["attn_mask", "dropout_p"],
        "efficient_attention": ["attn_bias", "dropout_p"],
        "efficient_attention_backward": ["attn_bias", "dropout_p", "grad_input_mask"],
        "cudnn_attention": ["attn_bias", "dropout_p"], "cudnn_attention_backward": ["attn_bias", "dropout_p"]}
    every = {s.name.removeprefix("aten::") for s in torch._C._jit_get_all_schemas() if "scaled_dot_product" in s.name}
    others = every - {f"_scaled_dot_product_{k}" for k in names}
    assert others == {"scaled_dot_product_attention", "_scaled_dot_product_attention_math",
                      "_scaled_dot_product_attention_math_for_mps", "_scaled_dot_product_fused_attention_overrideable",
                      "_scaled_dot_product_fused_attention_overrideable_backward"}
    assert all(opnames.gemm_like(n) for n in others)


RATES, ON = (1e-3, torch.tensor(1e-3)), (False, True)


def settings(**options):
    return [dict(zip(options, values)) for values in itertools.product(*options.values())]


#: each optimizer that has a fused kernel: the kernels it steps in, its unfused step's largest count a parameter, and
#: every setting of its options a CPU runs (state made `capturable` needs a GPU)
FUSED_STEPS = {
    "SGD": ({"_fused_sgd_"}, 4, [
        s for s in settings(lr=RATES, momentum=(0.0, 0.9), dampening=(0.0, 0.1), nesterov=ON, weight_decay=(0.0, 0.1),
                            maximize=ON)
        if not s["nesterov"] or (s["momentum"] and not s["dampening"])]),  # Nesterov momentum takes no other setting
    "Adam": ({"_fused_adam_", "_fused_adamw_"}, 6, settings(
        lr=RATES, weight_decay=(0.0, 0.1), amsgrad=ON, maximize=ON, decoupled_weight_decay=ON)),
    "AdamW": ({"_fused_adamw_"}, 6, settings(lr=RATES, weight_decay=(0.0, 0.1), amsgrad=ON, maximize=ON)),
    "Adagrad": ({"_fused_adagrad_"}, 4, settings(
        lr=RATES, lr_decay=(0.0, 0.1), weight_decay=(0.0, 0.1), maximize=ON, initial_accumulator_value=(0.0, 0.5))),
}


@pytest.mark.parametrize("name", FUSED_STEPS)
def test_choosing_a_fused_optimizer_kernel_never_lowers_a_steps_count(name):
    """The rule `opnames.FUSED_OPTIMIZER_MACS` holds. At every setting of an optimizer's options its fused kernel
    counts its constant a parameter, and its unfused step, a tensor at a time or foreach, counts at most that: SGD's
    and Adagrad's constants are their unfused steps' largest, Adam's and AdamW's are above theirs. Between them the
    four optimizers step in every kernel of the table.

    The one exception is at most one multiplication a step, with a tensor learning rate and weight decay: Adagrad's
    unfused step then divides the rate itself, whatever the parameters' size, which its kernel does inside."""
    n = 32
    kernels, largest, steps = *FUSED_STEPS[name][:2], FUSED_STEPS[name][2]
    assert set().union(*(k for k, _, _ in FUSED_STEPS.values())) == set(opnames.FUSED_OPTIMIZER_MACS)

    def count(options):
        """The largest of three steps' counts of a parameter of n values (the first step makes the optimizer's
        state), and the fused kernels they ran."""
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(n))
        opt, c, macs = getattr(torch.optim, name)([w], **options), Counter(digests=False), []
        for _ in range(3):
            w.grad = torch.randn(n)
            before = c.setup.elementwise_macs
            with c:
                opt.step()
            macs.append(c.setup.elementwise_macs - before)
        return max(macs), {row[2] for row in c.setup.log} & set(opnames.FUSED_OPTIMIZER_MACS)

    ran, most = set(), 0
    for s in steps:
        tensor = isinstance(s["lr"], torch.Tensor)
        fused, (kernel,) = count(s | {"fused": True})
        assert fused == opnames.FUSED_OPTIMIZER_MACS[kernel] * n, s
        ran.add(kernel)
        for foreach in ON:
            if tensor and foreach and name.startswith("Adam"):
                continue  # torch has no foreach step of Adam with a tensor rate
            unfused, none = count(s | {"foreach": foreach})
            assert not none and unfused <= fused + (name == "Adagrad" and tensor and s["weight_decay"] > 0), (s, foreach)
            most = max(most, 0 if tensor else unfused)
    assert ran == kernels and most == largest * n


def test_forward_macs_per_token_and_flags(tiny):
    m, _ = tiny
    attn = 2 * 8 * D // 2  # one sequence of 8 tokens: QK^T and AV, per token on average over the causal mask (full here)
    assert forward_macs_per_token(m, 8, VOCAB) >= PER_TOKEN
    assert forward_macs_per_token(m, 8, VOCAB) <= PER_TOKEN + 2 * attn * 2
    t = {"cpu_s": 100.0, "wall_s": 50.0, "gpu_s": 10.0, "macs": 1, "device_macs": {"cpu": 5}, "children_cpu_s": 0.0}
    out = flags(t, peak_macs_per_s=1e12)
    assert any(f.startswith("cpu_heavy") for f in out)
    assert any(f.startswith("cpu_parallel") for f in out)
    assert any(f.startswith("gpu_unexplained") for f in out)
    assert any(f.startswith("cpu_macs") for f in out)


def test_the_keyed_digest_is_exact_and_keyed():
    from training_slowdown.flops import KeyedDigest

    t = torch.randn(5, 7, generator=torch.Generator().manual_seed(1))
    d = KeyedDigest(b"k" * 16)
    assert d(t) == d(t.clone()) == d(t.t().contiguous().t())
    flipped = t.clone()
    flipped.view(torch.int32)[2, 3] ^= 1
    assert d(flipped) != d(t)
    assert d(t.reshape(7, 5)) != d(t)  # the shape is mixed in
    assert d(t.bfloat16()) != d(t.half())  # so is the dtype
    assert KeyedDigest(b"j" * 16)(t) != d(t)
    assert d(torch.zeros(3, dtype=torch.bfloat16)) != d(torch.zeros(3, dtype=torch.float16))
