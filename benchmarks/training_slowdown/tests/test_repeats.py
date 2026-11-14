"""The positions of a step that repeat another's words by the batch's tokens: the counter's `repeats` on each step
(`flops.Counter._repeats`), held to the definition by brute force; M leaving them out (`work.he_of_rows`,
`work.batch_count`); and a job's record carrying them to its grade, a stopped job's as the unbroken one's."""
import numpy as np
import pytest
import synthetic
import test_resume
import torch
import torch.nn.functional as F
from test_formalism import he_row
from test_grade import CONFIG, registry

from conftest import D, VOCAB
from training_slowdown import config, flops, run, starts, work
from training_slowdown import formalism as F_
from training_slowdown import grade as G
from training_slowdown.data import Data
from training_slowdown.flops import Counter


def definition(rows: list[list[int]]) -> int:
    """The repeats of `rows` (equal-length lists of token ids, one a row) as the rule states them: a position's key is
    its row's tokens up to it, its own included, and inside a row's leading run of one token that one token; the
    repeats are the positions past each row's first whose key is a first position's or an earlier position's."""
    def key(r, i):
        p = tuple(r[: i + 1])
        return p[:1] if len(set(p)) == 1 else p
    first = {key(r, 0) for r in rows}
    past = {key(r, i) for r in rows for i in range(1, len(r))}
    return sum(len(r) - 1 for r in rows) - len(past - first)


def random_rows(gen: np.random.Generator, n: int, length: int, vocab: int) -> np.ndarray:
    """`n` rows of `length` tokens of `vocab`, with planted repeats: some rows copy a prefix of an earlier row, and
    some start with a run of one token."""
    rows = gen.integers(0, vocab, (n, length))
    for i in range(n):
        if i and gen.random() < 0.5:
            cut = gen.integers(1, length + 1)
            rows[i, :cut] = rows[gen.integers(0, i), :cut]
        if gen.random() < 0.4:
            rows[i, : gen.integers(1, length + 1)] = rows[i, 0]
    return rows


@pytest.mark.parametrize("seed", range(4))
def test_the_sorted_count_is_the_definition(seed):
    """`flops.repeats`, counted on the sorted rows, against the definition on random batches with planted prefixes and
    runs, over small vocabularies (many chance repeats) and large ones (few), and a row of one token."""
    gen = np.random.default_rng(seed)
    for _ in range(250):
        n, length, vocab = int(gen.integers(1, 9)), int(gen.integers(1, 12)), int(gen.choice([2, 3, 5, 50]))
        rows = random_rows(gen, n, length, vocab)
        assert flops.repeats(rows) == definition(rows.tolist())
    assert flops.repeats(np.array([[7] * 6])) == definition([[7] * 6]) == 5


#: rows 0 and 1 share their first four tokens, row 2 starts with a run of five of one token and row 3 with row 2's
#: first token; ROWS[:, :-1], what a step reads as its inputs, has 3 + 4 = 7 repeats: row 1's places 1 to 3 and row
#: 2's places 1 to 4
ROWS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                     [1, 2, 3, 4, 9, 8, 7, 6, 5],
                     [3, 3, 3, 3, 3, 4, 5, 6, 7],
                     [3, 10, 11, 12, 13, 14, 15, 16, 17],
                     [20, 21, 22, 23, 24, 25, 26, 27, 28],
                     [29, 30, 31, 0, 1, 2, 3, 4, 5]])


def counted(body, batch: torch.Tensor = ROWS) -> tuple[dict, dict]:
    """The record of one step that moves `w` off θ and runs `body(rows, w, embed)` on `batch` as a batch of the data
    stream, beside the step with no reads of it."""
    torch.manual_seed(0)
    theta = {"w": torch.randn(D, D), "embed": torch.randn(VOCAB, D)}
    w, embed = (torch.nn.Parameter(theta[k].clone()) for k in ("w", "embed"))
    c = Counter(theta, digests=False)
    c.register({"w": w, "embed": embed})
    with c, torch.no_grad():
        c.begin_step()
        w.add_(0.01)
        body(c.batch(batch.clone()), w, embed)
        c.end_step()
    c.finish()
    rec = c.record()
    return rec["steps"][0], rec


def test_one_read_of_the_batch_counts_its_repeats_over_the_positions_its_tokens_count():
    """One read of the batch's inputs: the step's `repeats` are the definition's over them, the setup and the tail
    record none, and reading the tokens is no op of the step: its MACs, ops and token data read out of torch are the
    read's alone."""
    s, rec = counted(lambda rows, w, embed: F.embedding(rows[:, :-1], embed) @ w)
    assert s["tokens"] == ROWS[:, :-1].numel() and s["repeats"] == definition(ROWS[:, :-1].tolist()) == 7
    assert s["batch_rows"] == len(ROWS)
    assert all(rec[x]["repeats"] is None and rec[x]["batch_rows"] is None for x in ("setup", "tail"))
    assert s["exited"] == 0 and s["by"] == {"fwd/updated": ROWS[:, :-1].numel() * D * D}


def test_micro_batches_of_one_step_count_as_the_step_together():
    """Two micro-batches of one step, each read in its own forward: the repeats are the whole batch's, so the prefix
    rows 0 and 1 share, split between the two, counts as in one read."""
    def body(rows, w, embed):
        for x in (rows[:1], rows[1:]):
            F.embedding(x[:, :-1], embed) @ w

    s, _ = counted(body)
    assert s["tokens"] == ROWS[:, :-1].numel() and s["repeats"] == definition(ROWS[:, :-1].tolist()) == 7


def test_a_slice_read_twice_counts_once():
    """The inputs read twice (two lookups) and two of their rows a third time: each position once."""
    def body(rows, w, embed):
        x = rows[:, :-1]
        F.embedding(x, embed) @ w
        F.embedding(x, embed) @ w
        F.embedding(x[:2], embed) @ w

    s, _ = counted(body)
    assert s["tokens"] == ROWS[:, :-1].numel() and s["repeats"] == definition(ROWS[:, :-1].tolist()) == 7
    assert s["batch_rows"] == len(ROWS)


def test_some_rows_read_count_over_those_rows():
    """Two of the six rows read: the repeats are those rows' alone."""
    s, _ = counted(lambda rows, w, embed: F.embedding(rows[2:4, :-1], embed) @ w)
    assert s["tokens"] == 2 * 8 and s["repeats"] == definition(ROWS[2:4, :-1].tolist()) == 4 and s["batch_rows"] == 2


@pytest.mark.parametrize("read", [
    lambda rows: [rows[:, :-1], rows[:, 1:]],        # the targets read by index: a read from the second position
    lambda rows: [rows[:, :-1], rows[:2, :3]],       # a read of each row's first three of a step's eight
    lambda rows: [rows[:3, :-1], rows[3:, :-2]],     # two lengths
    lambda rows: [rows[:, ::2]],                     # a gap
    lambda rows: [rows[:, :-1] + 0],                 # a value computed from the batch, read as indices
    lambda rows: [rows.reshape(-1)[:20]],            # the batch as one stream, cut into a window
], ids=["targets", "shorter", "two-lengths", "gap", "computed", "window"])
def test_a_read_that_is_not_rows_from_their_first_position_records_none(read):
    """Reads that aren't each, in every row they touch, the row's first positions up to one length over the step, or
    that aren't of the batch's own storage: the step records no repeats, and its tokens as before. The inputs and
    the targets together cover whole rows, but the targets' read starts past each row's first position."""
    def body(rows, w, embed):
        for x in read(rows):
            F.embedding(x, embed) @ w

    s, _ = counted(body)
    assert s["tokens"] > 0 and s["repeats"] is None and s["batch_rows"] is None


@pytest.mark.parametrize("write", [
    lambda rows: rows.fill_(3),                                          # to one token: every position would repeat
    lambda rows: rows.copy_(torch.arange(rows.numel()).view_as(rows) % VOCAB),  # to other tokens: the repeats read missed
    lambda rows: rows[1:2, :2].copy_(rows[4:5, :2]),                     # a row already read, through a view
], ids=["one-token", "other-tokens", "a-read-row"])
def test_a_batch_written_after_its_read_records_none(write):
    """The count reads the batch's tokens at the step's end, so a batch written after the step's first read of it
    records no repeats, whichever way the write would move them: not the words the model read. The same write before
    the step's reads is counted, as the batch the model reads."""
    def body(rows, w, embed):
        F.embedding(rows[:3, :-1], embed) @ w
        write(rows)
        F.embedding(rows[3:, :-1], embed) @ w

    s, _ = counted(body)
    assert s["tokens"] == ROWS[:, :-1].numel() and s["repeats"] is None and s["batch_rows"] is None
    written = ROWS.clone()
    write(written)
    assert counted(lambda rows, w, embed: (write(rows), F.embedding(rows[:, :-1], embed) @ w))[0]["repeats"] == \
        definition(written[:, :-1].tolist()) != definition(ROWS[:, :-1].tolist())


def test_a_batch_marked_by_the_program_and_a_one_dimensional_stream_record_none():
    """Token data the program marks (`Counter.tokens`) is no batch of the data stream, and a 1-D batch has no rows:
    neither step records repeats."""
    s, _ = counted(lambda rows, w, embed: F.embedding(rows[:8], embed) @ w, ROWS.reshape(-1))
    assert s["repeats"] is None
    torch.manual_seed(0)
    theta = {"w": torch.randn(D, D), "embed": torch.randn(VOCAB, D)}
    w, embed = (torch.nn.Parameter(theta[k].clone()) for k in ("w", "embed"))
    c = Counter(theta, digests=False)
    c.register({"w": w, "embed": embed})
    with c, torch.no_grad():
        c.begin_step()
        F.embedding(c.tokens(ROWS.clone())[:, :-1], embed) @ w
        c.end_step()
    c.finish()
    assert c.record()["steps"][0]["repeats"] is None


#: a step of 1,024 tokens in rows of 256 positions (attention's widest token dim): a forward product and a weight
#: gradient token-indexed, a product no token indexes, one reading a gather's rows, one array times itself, and the
#: optimizer's and attention's, which M leaves out
ROWS_OF_A_STEP = [he_row("forward", ["m"], 1024, 64, 64, 2), he_row("backward", ["k"], 64, 1024, 64),
                  he_row("forward", [], 64, 64, 64), he_row("forward", ["m"], 1024, 64, 48) | {"gathered": [True, False]},
                  he_row("backward", ["m"], 1024, 64, 32) | {"self": True}, he_row("optimizer", [], 64, 64, 64, 4),
                  he_row("forward", ["batch", "m", "n"], 256, 64, 256, 1, 4)]
#: 100 of 4,096 tokens counted as repeats, over the 4,096 a count covers
COUNT = {"positions": 100, "tokens": 4096, "measured": 4096}


def test_m_leaves_out_the_counted_share_of_every_token_indexed_product_and_nothing_else():
    """Token-indexed products keep 1 − 1/n − p/T of their MACs with a count, 1 − 1/n without; a product no token
    indexes moves not at all, nor do the products M leaves out (gathered, one array times itself, the optimizer's and
    attention's), nor any output. Without a count M is what it was before the count, key for key: no `repeats`."""
    k, n = 1e6, 256
    before, after = (work.he_of_rows(ROWS_OF_A_STEP, k, 2.0, 1024, b) for b in (None, COUNT))
    assert before == work.he_of_rows(ROWS_OF_A_STEP, k, 2.0, 1024) and "repeats" not in before
    assert after.pop("repeats") == COUNT and after.keys() == before.keys()
    for x in ("candidates_macs", "self_macs", "gathered_macs", "left_out_macs", "r_step", "trained_weights"):
        assert after[x] == before[x]
    assert before["gathered_macs"] > 0 and before["self_macs"] > 0 and before["left_out_macs"] > 0
    share = (1 - 1 / n - COUNT["positions"] / COUNT["tokens"]) / (1 - 1 / n)
    token = [not all(g["weight"]) for g in before["groups"]]
    assert sum(token) == 2 and len(token) == 3
    for t, a, b in zip(token, after["groups"], before["groups"]):
        assert {**a, "macs": 0} == {**b, "macs": 0}
        assert a["macs"] == (pytest.approx(b["macs"] * share, rel=1e-12) if t else b["macs"])


def test_a_steps_count_is_read_only_where_its_tokens_are_its_rows_times_the_products_positions():
    """`work.batch_count` over a run's steps: a step counted over rows of the products' n positions adds its repeats;
    one counted over rows of another length, and one with no count (a record made before it, or a step whose reads
    gave none), adds its tokens alone; with no n (no product with two token dims) every count is read; with no
    step's count read there is none, and M is the premise's."""
    steps = [{"tokens": 2 * 256, "repeats": 5, "batch_rows": 2}, {"tokens": 2 * 255, "repeats": 3, "batch_rows": 2},
             {"tokens": 512, "repeats": None, "batch_rows": None}, {"tokens": 512}]
    assert work.batch_count(steps, 256) == {"positions": 5, "tokens": 2046, "measured": 512}
    assert work.batch_count(steps, None) == {"positions": 8, "tokens": 2046, "measured": 1022}
    assert work.batch_count(steps, 128) is None and work.batch_count(steps[2:], 256) is None
    assert work.row_positions(ROWS_OF_A_STEP) == 256 and work.row_positions(ROWS_OF_A_STEP[:-1]) is None
    rows = [r | {"dims": r["dims"] | {"m": 255, "n": 255}} if len(r["token_dims"]) > 1 else r for r in ROWS_OF_A_STEP]
    assert work.he_of_rows(rows, 1e6, 2.0, 1024, work.batch_count(steps[1:2], work.row_positions(rows)))["groups"] \
        != work.he_of_rows(rows, 1e6, 2.0, 1024)["groups"]
    assert "repeats" not in work.he_of_rows(rows, 1e6, 2.0, 1024, work.batch_count(steps[:1], work.row_positions(rows)))


#: what every window of the planted stream starts with: a run of four of one token, then another. In a batch of R
#: rows each row's places 1 to 3 repeat its first (a leading run), and place 4 repeats in all rows but one (a prefix
#: they share)
PLANT = [7, 7, 7, 7, 9]
TOKENS = synthetic.tokens


def planting(monkeypatch) -> None:
    """`synthetic.tokens` writing `PLANT` at the start of every window the stream cuts at the tiny start's context,
    in the training parts only (`synthetic.tiny_data` makes them first): the held-out pools stay their documents'
    bytes."""
    parts = iter(range(len(synthetic.PARTS)))

    def planted(rng: np.random.Generator, n: int, alphabet: int | None = None) -> np.ndarray:
        t = TOKENS(rng, n, alphabet)
        if next(parts, None) is not None:
            for i in range(0, n, synthetic.SHAPE["ctx"]):
                t[i:i + len(PLANT)] = PLANT[: n - i]
        return t

    monkeypatch.setattr(synthetic, "tokens", planted)


def served(cfg, reg, steps: int) -> list[int]:
    """The repeats of each of the first `steps` batches the job's stream serves, by the definition, over each row's
    inputs (all its tokens but the last target)."""
    stream = run.stream_of(cfg, Data.of(cfg.start.data, starts.registry(reg)), 0, reg.parent / "starts")
    return [definition(stream.next(cfg.budget["batch"])[:, :-1].tolist()) for _ in range(steps)]


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """One search job of the tiny start on the planted stream, run as a node runs it: its record, config and
    registry."""
    with pytest.MonkeyPatch.context() as mp:
        planting(mp)
        rec, cfg, reg, _ = synthetic.run_job(tmp_path_factory.mktemp("planted"))
    return rec, cfg, reg


def without_count(rec: dict) -> dict:
    """`rec` as a record made before the count writes it: no step has `repeats` or `batch_rows`."""
    return rec | {k: [{x: v for x, v in s.items() if x not in ("repeats", "batch_rows")} for s in rec[k]]
                  for k in ("steps",)} | {k: {x: v for x, v in rec[k].items() if x not in ("repeats", "batch_rows")}
                                          for k in ("setup", "tail")}


@pytest.mark.slow
def test_a_jobs_record_carries_its_steps_repeats_to_its_grade_and_the_record_without_them_grades_as_before(planted):
    """A job of the shipped route on a stream whose windows start alike (`PLANT`): every step records the repeats
    its batch holds, over its rows of the context's positions; the grade's M leaves out their share of the steps
    after the first update, and its strings say the count was taken on the job's batch and name the premise that
    remains (`work.P_REPEATS`). The same record without the field grades as a record before the count: M is the
    premise's, with no `repeats`, and its strings name `work.P_BATCH`."""
    rec, cfg, reg = planted
    ctx, batch = synthetic.SHAPE["ctx"], cfg.budget["batch"]
    got = [s["repeats"] for s in rec["steps"]]
    assert got == served(cfg, reg, len(rec["steps"])) and min(got) >= 3 * batch + batch - 1
    assert all(s["batch_rows"] == batch and s["tokens"] == batch * ctx for s in rec["steps"])
    config = CONFIG | {"start": "tiny", "seeds": {"search": [0], "final": []}}
    reg_ = registry("tiny", synthetic.SHAPE)
    after, before = (G.grade(config, reg_, [{"seed": 0, "split": "search", "record": r}], [
        {"seed": 0, "split": "search", "record": r}]) for r in (rec, without_count(rec)))
    row = G.rows_of(rec)[-1]
    he, he0 = (G.traced(r, row)["formalism"]["products"]["he"] for r in (rec, without_count(rec)))
    changed = G.changed_steps(rec, None)
    assert he["repeats"] == {"positions": sum(s["repeats"] for s in changed), "tokens": batch * ctx * len(changed),
                             "measured": batch * ctx * len(changed)} and "repeats" not in he0
    share = (1 - 1 / ctx - he["repeats"]["positions"] / he["repeats"]["tokens"]) / (1 - 1 / ctx)
    for a, b in zip(he["groups"], he0["groups"]):
        assert a["macs"] == (pytest.approx(b["macs"] * share, rel=1e-12) if not all(b["weight"]) else b["macs"])
    assert any(not all(g["weight"]) for g in he["groups"])
    assert after["rho"]["terms"]["products"] is not None and before["rho"]["terms"]["products"] is not None
    shape = synthetic.SHAPE | {"forward_macs_per_token": rec["forward_macs_per_token"]}
    for r, s, premise, other in ((rec, after, work.P_REPEATS, work.P_BATCH),
                                 (without_count(rec), before, work.P_BATCH, work.P_REPEATS)):
        bound = F_.bound(F_.measure(G.traced(r, row), shape), s["policy"]["c_gates"], s["policy"]["k_bits"])
        published = [s["rho"]["products_rests_on"], bound["rests_on"]["products"],
                     G.per_period(1.0, s["rho"] | {"binding_kind": "products"})["parts"]["proved"]]
        assert all(premise in x and other not in x for x in published)
    assert work.COUNTED in after["rho"]["products_rests_on"]
    assert G.traced(rec, row)["formalism"]["labels"]["products"].endswith(F_.COUNTED_PRODUCTS)
    assert F_.COUNTED_PRODUCTS not in G.traced(without_count(rec), row)["formalism"]["labels"]["products"]


def test_a_stopped_and_continued_job_counts_the_unbroken_jobs_repeats(tmp_path, monkeypatch):
    """A job stopped after step 1 and continued records each step's repeats as the unbroken job does, which are its
    batch's: the count reads a step's batch and nothing else."""
    planting(monkeypatch)
    reg = synthetic.tiny_registry(tmp_path / "reg")
    tiny = (tmp_path, reg, synthetic.write_config(tmp_path / "a.toml", budget=test_resume.BUDGET))
    whole = test_resume.segment(tiny)
    stopped = test_resume.segment(tiny, stop_after=1)
    cut = run.joined(run.joined(None, stopped["record"]), test_resume.segment(tiny, {"step": 2}, stopped["state"]))
    cfg = config.load(tiny[2], starts.registry(reg))
    want = served(cfg, reg, test_resume.BUDGET["steps"])
    assert [s["repeats"] for s in cut["steps"]] == [s["repeats"] for s in whole["steps"]] == want
    assert min(want) >= 4 * test_resume.BUDGET["batch"] - 1
