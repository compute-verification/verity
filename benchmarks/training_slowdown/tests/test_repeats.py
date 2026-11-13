"""The positions of a step that repeat another's words by the batch's tokens: the counter's `repeats` on each step
(`flops.Counter._repeats`), held to the definition by brute force."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import D, VOCAB
from training_slowdown import flops
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


def drawn(gen: np.random.Generator, n: int, length: int, vocab: int) -> np.ndarray:
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
    """`flops.repeats`, counted on the sorted rows, against the definition on drawn batches with planted prefixes and
    runs, over small vocabularies (many chance repeats) and large ones (few), and a row of one token."""
    gen = np.random.default_rng(seed)
    for _ in range(250):
        n, length, vocab = int(gen.integers(1, 9)), int(gen.integers(1, 12)), int(gen.choice([2, 3, 5, 50]))
        rows = drawn(gen, n, length, vocab)
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
