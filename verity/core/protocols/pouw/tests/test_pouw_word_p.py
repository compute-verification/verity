"""The word-granular production P v1.1 at the k the MVP uses.

Its repairs guarantee clause 1 of the rejection rule and no cycle of π² inside a step.  The rule's other two clauses and
λ₃ ≥ 16 (the edge connectivity of the step multigraph, which gives MinRank 16 under the MinRank argument's hypotheses)
are not guaranteed by construction, so each is certified here, exactly, per k.
"""

from __future__ import annotations

import numpy as np
import pytest

from verity.ml.protocols.pouw.schemes.ncp import D, admissible_perm, perm_digest, word_perm

# An independent generator's vectors; k = 4096 and 11008 are v1 unchanged.
DIGESTS = {
    1088: "536757a293b17fdbecab9cca20d9a4ed76331d12424f6f47855190345b92546f",
    2048: "7f00a9b74d4bd8b36019e74ee31ae12d918dd65861957c94cfc8af8699139732",
    4096: "24e68f30edbe538e2842266ac6b917a8a0ccf3ee0b4f8bf2cb1d8fe5c5943854",
    11008: "a57ea054ea3e404469f0c5b9055a0f6789fb1b2ce4c02b2fe83914828c040eb8",
}
MVP_K = (1088, 2048, 11008)    # the domain's floor and Qwen2.5-3B's two depths


def edge_connectivity(f: np.ndarray) -> int:
    """Stoer-Wagner's exact global min cut of the step multigraph: nodes are the k/16 steps, one edge per j."""
    k = f.size
    u, v = np.arange(k) // D, f // D
    w = np.zeros((k // D, k // D), dtype=np.int64)
    np.add.at(w, (u, v), 1)
    np.add.at(w, (v, u), 1)
    np.fill_diagonal(w, 0)
    active = list(range(w.shape[0]))
    best = None
    while len(active) > 1:
        idx = np.array(active)
        sub = w[np.ix_(idx, idx)]
        conn, added = sub[0].copy(), np.zeros(idx.size, dtype=bool)
        added[0] = True
        prev = last = 0
        for _ in range(idx.size - 1):
            nxt = int(np.argmax(np.where(added, -1, conn)))
            prev, last, cut = last, nxt, int(conn[nxt])
            added[nxt] = True
            conn += sub[nxt]
        best = cut if best is None else min(best, cut)
        s, t = int(idx[prev]), int(idx[last])
        w[s, :] += w[t, :]
        w[:, s] += w[:, t]
        w[s, s] = 0
        active.remove(t)
    return best


def inside_step_cycle(f: np.ndarray) -> bool:
    g = f[f]
    nxt = {int(a): int(g[a]) for a in np.flatnonzero(g // D == np.arange(f.size) // D)}
    for s in nxt:
        seen, x = {s}, s
        while x in nxt:
            x = nxt[x]
            if x in seen:
                return True
            seen.add(x)
    return False


@pytest.mark.parametrize("k", sorted(DIGESTS))
def test_digest_is_the_red_teams(k):
    assert perm_digest(word_perm(k)[0]) == DIGESTS[k]


@pytest.mark.parametrize("k", MVP_K)
def test_certified_at_the_mvps_k(k):
    pi, sigma = word_perm(k)
    f = np.array(pi)
    j = np.arange(k)
    assert admissible_perm(k, pi, sigma)                                   # all three clauses of Lean `Admissible`
    assert not np.any(f == j) and not np.any((f[f] == j) & (f != j))       # (a) no fixed point, no 2-cycle
    assert not np.any(f // D == j // D)                                    # (b) clause 1
    assert not inside_step_cycle(f)                                        # (c) no cycle of π² inside a step
    assert edge_connectivity(f) >= 16                                      # λ₃ ≥ 16, so MinRank = 16


def test_edge_connectivity_on_a_known_graph():
    """The shift by 16 joins step t to t + 1 only: a cycle of k/16 steps with 16 parallel edges each way, λ₃ = 32."""
    k = 256
    assert edge_connectivity((np.arange(k) + 16) % k) == 32
