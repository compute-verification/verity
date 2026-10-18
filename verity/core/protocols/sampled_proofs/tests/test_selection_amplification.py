"""How far one wrong unit reaches through in-circuit selection (red-team RT-R1R3, format/policy table, rows R1 and R3).

A toy MoE layer in the two padded forms the served rows use: token-major, the expert's weights gathered by the routed id
(`fp8_moe_experts.MoeExpertFp8Gemm`, GLM-5.3's row of record), and fixed slots, each expert's weights read at its own slot and
the router selecting among slot outputs (`moe_pad`, `LMoeFfn_v1`). Words are 16 bits, one wrong unit's whole output.
"""
from __future__ import annotations

import random

E, D, MASK = 8, 16, 0xFFFF
BOT = None


def matvec(w, x):
    return [sum(w[r][c] * x[c] for c in range(D)) & MASK for r in range(D)]


def router(x, gate):
    return max(range(E), key=lambda e: (sum(g * v for g, v in zip(gate[e], x)), -e))


def token_major(x, ident, experts):
    return matvec(experts[ident], x)


def fixed_slots(x, ident, experts):
    enables = [ident == e for e in range(E)]
    slots = [matvec(experts[e], x) if enables[e] else BOT for e in range(E)]
    return slots[ident]


def mux_c5(index, words):
    return sum((index == v) * words[v] for v in range(len(words))) & MASK


def model(seed):
    rng = random.Random(seed)
    experts = [[[rng.randrange(1 << 16) for _ in range(D)] for _ in range(D)] for _ in range(E)]
    gate = [[rng.randrange(-8, 8) for _ in range(D)] for _ in range(E)]
    return experts, gate


def test_one_wrong_router_unit_selects_any_expert_in_both_forms():
    experts, gate = model(1)
    x = [random.Random(2).randrange(1 << 16) for _ in range(D)]
    honest = router(x, gate)
    for wrong in range(E):
        assert token_major(x, wrong, experts) == fixed_slots(x, wrong, experts) == matvec(experts[wrong], x)
    assert token_major(x, honest, experts) != token_major(x, (honest + 1) % E, experts)


def test_one_wrong_router_unit_dumps_a_chosen_expert_column():
    experts, _ = model(3)
    for target, k in [(5, 0), (2, 9)]:
        one_hot = [int(c == k) for c in range(D)]
        out = fixed_slots(one_hot, target, experts)
        assert out == [experts[target][r][k] for r in range(D)]


def test_a_mux_over_secrets_is_c5_legal_after_a_rewrite():
    rng = random.Random(4)
    secrets = [rng.randrange(1 << 16) for _ in range(64)]
    assert all(mux_c5(i, secrets) == secrets[i] for i in range(64))


def test_one_wrong_activation_word_moves_a_whole_weight_column():
    experts, _ = model(5)
    w = experts[0]
    x = [random.Random(6).randrange(1 << 16) for _ in range(D)]
    k, delta = 3, 1
    bad = list(x)
    bad[k] = (bad[k] + delta) & MASK
    diff = [(a - b) & MASK for a, b in zip(matvec(w, bad), matvec(w, x))]
    assert diff == [w[r][k] * delta & MASK for r in range(D)]
