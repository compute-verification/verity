"""The premise the products count takes on a batch (`work.P_BATCH`), and the three batches that break it, each held on
the model's forward: positions the count charges as words of their own that hold equal words, to rounding."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from training_slowdown import model, work
from training_slowdown.model import LM, Size

#: a tiny lf model: three blocks, so "at every layer" has a first, a middle and a last
SIZE, VOCAB, CTX = Size(d_model=32, n_layer=3, n_head=2, d_ff=48), 16, 8
#: the products a block's forward runs outside attention, in its order: q, k and v before its attention, then the
#: attention's output, the MLP's gate and up, and its down
BLOCK = 7
#: how far apart two positions' words may be and still be equal: their largest difference over the larger one's
#: largest entry. float32 sums one word at two positions in two orders, so the two agree to rounding and not bit for
#: bit; the bound is under a hundredth of a bf16 word's step. Over seeds 0 to 999 of `words`, the worst equal pair
#: is 1.6e-6 apart
EQUAL = 2e-5
#: how far apart a control pair's words are at the least. Over seeds 0 to 999 of `words`, the worst control is
#: 0.0201 apart
APART = 500 * EQUAL
#: how far apart the farthest of the proportions' equal pairs is at the least, at every product after the second
#: attention: the repeat stops there. Over seeds 0 to 999 of `words`, the worst is 0.0120 apart
PART = 100 * EQUAL


class Operands(TorchDispatchMode):
    """The token-indexed operand of each matrix product run under it outside attention, as the dispatcher sees one: a
    linear layer's `mm` of the batch's positions by the layer's width."""

    def __init__(self, positions: int):
        super().__init__()
        self.positions, self.seen = positions, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket is torch.ops.aten.mm and args[0].shape[0] == self.positions:
            self.seen.append(args[0].detach().clone())
        return func(*args, **(kwargs or {}))


def words(rows: list[list[int]], seed: int = 0) -> list[torch.Tensor]:
    """The words every position of the batch `rows` holds at each product the model's forward runs outside
    attention, in the forward's order (`BLOCK` a block, then the head's): each product's token-indexed operand, as
    rows by positions by width. The weights are random, each matrix at a scale that keeps a word's entries of order 1
    and attention far from uniform."""
    gen = torch.Generator().manual_seed(seed)
    lm = LM(SIZE, VOCAB, CTX)
    for p in lm.parameters():
        if p.dim() == 2:
            torch.nn.init.normal_(p, std=p.shape[1] ** -0.5, generator=gen)
    idx = torch.tensor(rows)
    with torch.no_grad(), Operands(idx.numel()) as operands:
        lm(idx)
    assert len(operands.seen) == BLOCK * SIZE.n_layer + 1
    return [x.view(*idx.shape, -1) for x in operands.seen]


def apart(x: torch.Tensor, a: tuple[int, int], b: tuple[int, int]) -> float:
    """How far apart the words of two positions (row, place) are in one operand `x`: their largest difference over
    the larger one's largest entry."""
    u, v = x[a], x[b]
    return ((u - v).abs().max() / torch.maximum(u.abs().max(), v.abs().max())).item()


def spread(at: list[torch.Tensor], pairs: list[tuple]) -> tuple[float, float]:
    """The least and the largest `apart` of `pairs` over the operands `at`."""
    every = [apart(x, a, b) for x in at for a, b in pairs]
    return min(every), max(every)


#: rows 0 and 1 share their first three tokens; row 2 shares none with them
PREFIX = [[1, 2, 3, 4, 5, 6, 7, 8],
          [1, 2, 3, 9, 5, 6, 7, 8],
          [10, 11, 12, 13, 14, 15, 0, 9]]
#: row 0 starts with a run of five of one token; row 1 has no token twice
RUN = [[1, 1, 1, 1, 1, 2, 3, 4],
       [5, 6, 7, 8, 9, 10, 11, 12],
       [13, 14, 15, 0, 1, 2, 3, 4]]
#: token 2 after tokens 1 and 2 in equal numbers: row 0's places 1, 3 and 5, and row 1's place 3 in another order.
#: Row 2's place 3 holds it after one 1 and three 2s
PROPORTIONS = [[1, 2, 1, 2, 1, 2, 3, 4],
               [1, 1, 2, 2, 5, 6, 7, 8],
               [1, 2, 2, 2, 9, 10, 11, 12]]


def test_two_rows_that_share_their_first_tokens_hold_equal_words_at_the_shared_positions():
    """Rows 0 and 1 share their first three tokens. A position reads only the tokens up to it, so the two rows hold
    equal words at places 0, 1 and 2 at every product of every layer. The count leaves out place 0
    (`work.SHARED_PREFIX`) and charges places 1 and 2 in both rows. The control is the next place, where the rows'
    tokens differ."""
    at = words(PREFIX)
    assert spread(at, [((0, p), (1, p)) for p in range(work.SHARED_PREFIX, 3)])[1] <= EQUAL
    assert spread(at, [((0, 3), (1, 3))])[0] >= APART


@pytest.mark.parametrize("rotation", [True, False])
def test_a_row_that_starts_with_a_run_of_one_token_holds_equal_words_along_the_run(rotation, monkeypatch):
    """Row 0 starts with five of one token. Attention at a place of the run reads keys and values of that one token
    alone, and a weighted mean of one value is that value whatever the weights, so the rotation, which only moves the
    weights, changes nothing: the five places hold equal words at every product of every layer, and the count
    charges the four past the first (`work.SHARED_PREFIX`). The control is the same places of a row with no token
    twice."""
    if not rotation:
        monkeypatch.setattr(model, "rope", lambda x, cos, sin: x)
    at = words(RUN)
    assert spread(at, [((0, 0), (0, p)) for p in range(work.SHARED_PREFIX, 5)])[1] <= EQUAL
    assert spread(at, [((1, 0), (1, p)) for p in range(work.SHARED_PREFIX, 5)])[0] >= APART


def test_without_the_rotation_one_token_after_the_same_proportions_holds_equal_words_after_the_first_attention(
        monkeypatch):
    """With no rotation an attention's weights read tokens and not places, so its output at a place is a mean of the
    earlier tokens' values by their numbers: two places with one token whose earlier tokens are in the same
    proportions hold equal words from the first attention to the second, in any order of those tokens. The second
    attention reads the words of the places before, which differ between the two, so they part at every product
    after it, which the last line holds. The control is that token after other proportions."""
    monkeypatch.setattr(model, "rope", lambda x, cos, sin: x)
    at = words(PROPORTIONS)
    first, second = at[3:BLOCK + 3], at[BLOCK + 3:]
    same = [((0, 1), (0, 3)), ((0, 1), (0, 5)), ((0, 1), (1, 3))]
    assert spread(first, same)[1] <= EQUAL
    assert spread(first, [((0, 1), (2, 3))])[0] >= APART
    assert all(spread([x], same)[1] >= PART for x in second)
