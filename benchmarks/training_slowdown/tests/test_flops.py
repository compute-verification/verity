import torch
import torch.nn.functional as F

from conftest import D, PER_TOKEN, VOCAB, loss_of
from training_slowdown.annotate import annotate
from training_slowdown.flops import Counter, flags, forward_macs_per_token, gather_selects, guard_exits, leaked, mux_tree

IDX = torch.randint(0, VOCAB, (3, 8), generator=torch.Generator().manual_seed(3))  # 24 tokens: no activation is weight-shaped
TOKENS = IDX.numel()
LOOKUP = mux_tree(VOCAB, 32) * D  # the updated tied table's lookup per token: 58 multiplexers per word


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
    assert s0["by"]["fwd/theta"] == PER_TOKEN * TOKENS  # step 0 reads θ itself
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


def test_a_record_counted_before_lookups_were_priced_gets_them_at_ingest(tiny, monkeypatch):
    """`annotate.with_lookups` on a record counted as 3.5 did (no lookup priced) gives 3.6's annotation of the same run."""
    from training_slowdown import flops
    from training_slowdown.annotate import with_lookups

    m, theta = tiny
    want = annotate(train(m, theta))
    torch.manual_seed(0)
    m2 = type(m)()
    monkeypatch.setattr(flops, "gather_macs", lambda *a: 0)
    old = train(m2, theta)
    monkeypatch.undo()
    assert annotate(old).steps[1].depth == 2 * D * D
    got = annotate(with_lookups(old, ["emb"]))
    assert got.macs == want.macs
    assert [(s.profile, s.depth, s.updated_per_token) for s in got.steps] == [
        (s.profile, s.depth, s.updated_per_token) for s in want.steps]
    assert annotate(with_lookups(old, ["w1.weight"])).steps[1].depth > annotate(old).steps[1].depth
    assert annotate(with_lookups(old, ["head"])) == annotate(old)  # names no weight the record lists


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
    assert c.steps[0].by["fwd/theta"] == PER_TOKEN * TOKENS


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
