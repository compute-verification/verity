import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "breakeven" / "pool_window_hits.py"
spec = importlib.util.spec_from_file_location("pool_window_hits", HERE)
pw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pw)

TOKENS = {"linear": 1, "core": 0, "idx": 0, "head": 0, "topk": 2048}


def _trace():
    """Three requests in 4-token blocks: B reuses A's two blocks, C one of them; ingress buckets 0, 5 and 15."""
    a = (list(range(1, 9)) + [9, 10], 8, 0)
    b = (list(range(1, 9)) + [20, 21] + [22], 10, 5)
    c = ([1, 2, 3, 4, 30, 31, 32, 33] + [34], 8, 15)
    steps = [[(0, 0, 8)], [(0, 8, 1)], [(1, 8, 2)], [(1, 10, 1), (2, 4, 4)], [(2, 8, 1)]]
    return [a, b, c], steps


def test_a_window_keeps_only_its_own_requests_blocks():
    reqs, steps = _trace()
    got = pw.replay(reqs, steps, 4, TOKENS, [(1e9, 0.0), (10.0, 0.0), (10.0, 5.0)])
    assert (got["saved_tokens"], got["hit_requests"]) == (12, 2)
    # one window: nothing lost; windows [0, 10), [10, 20): C's block is A's and B's, a window before;
    # windows [-5, 5), [5, 15), [15, 25): every holder is in another window
    assert got["lost_tokens"] == [0, 4, 12]
    assert got["lost_requests"] == [0, 1, 2]
    assert got["lost_flops"] == got["lost_tokens"]


def test_a_request_holds_the_blocks_it_recomputed():
    """D, a window after A, recomputes A's blocks; E, in D's window, hits them on D."""
    a = (list(range(1, 9)) + [9], 8, 0)
    d = (list(range(1, 9)) + [40, 41] + [42], 10, 12)
    e = (list(range(1, 9)) + [50, 51] + [52], 10, 14)
    steps = [[(0, 0, 8)], [(1, 8, 2)], [(2, 8, 2)]]
    got = pw.replay([a, d, e], steps, 4, TOKENS, [(10.0, 0.0)])
    assert got["lost_tokens"] == [8] and got["lost_requests"] == [1]
