"""verity_vllm.program.prune: a body's dead units (the nodes no output reads) and the body without them, each reference renumbered
in its own form; on one rank, the request wrapper's dead units are the last layer's rows before the prefill's last."""
from __future__ import annotations

from collections import Counter

from verity_vllm.predict import build as PB
from verity_vllm.predict.phi import Phi
from verity_vllm.program.prune import _read, _renumbered, dead_units, without_dead_units

from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided

TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
PHI = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})


def _shape(body) -> list:
    return [(n.fn.id, [a.refs.to_json() for a in n.args]) for n in body.nodes] + [body.ret.to_json()]


def test_the_request_wrappers_dead_units_are_the_last_layers_rows_before_the_prefills_last(monkeypatch):
    """TINY's request of 8 prompt rows: the last layer's attention, o projection, MLP and their norms on rows 0-6 (and the rotary
    of their queries) reach no sample; the build drops those 56 units and nothing else, as pruning the unpruned body does."""
    pruned = PB.build_program(PHI, "request", LP=8, T=3)
    monkeypatch.setattr(PB, "without_dead_units", lambda b: (b, list(range(len(b.nodes)))))
    full = PB.build_program(PHI, "request", LP=8, T=3)
    body = full.program.fn.body
    dead = dead_units(body)
    assert Counter(body.nodes[k].fn.id.split("{")[0] for k in dead) == {"Gemm_v2": 21, "RMSNormFusedCuda_v2": 14, "RoPE_v1": 7,
                                                                         "Attention_v5": 7, "SiluMul_v2": 7}
    again, kept = without_dead_units(body)
    assert kept == [k for k in range(len(body.nodes)) if k not in set(dead)]
    assert _shape(again) == _shape(pruned.program.fn.body) and again.offsets[-1] == body.offsets[-1] - sum(body.nodes[k].gates for k in dead)
    assert dead_units(pruned.program.fn.body) == [] and without_dead_units(pruned.program.fn.body)[0] is pruned.program.fn.body
    assert pruned.digest != full.digest
    step = PB.build_program(PHI, "step", M=1, pos0=8)
    assert dead_units(step.program.fn.body) == []


def test_each_reference_is_renumbered_in_its_own_form():
    new = [0, 1, 1, 3, 2]  # nodes 1 and 3 dead: 0, 2, 4 become 0, 1, 2
    memo: dict = {}
    a, p, empty = Affine("n", 2, 5, 3), Affine("p", 2, 0, 3), Affine("n", 3, 0, 0)
    s = Strided("n", 4, 1, 2, 8, 1, 6)
    e = Explicit((("n", 4, 0), ("p", 4, 1), ("n", 2, 3)))
    c = Concat([Affine("n", 0, 0, 2), Concat([s, e])])
    got = {k: _renumbered(r, new, memo) for k, r in {"a": a, "p": p, "empty": empty, "s": s, "e": e, "c": c}.items()}
    assert got["p"] is p and got["empty"] is empty
    assert got["a"].to_json() == Affine("n", 1, 5, 3).to_json()
    assert type(got["s"]) is Strided and got["s"].to_json() == Strided("n", 2, 1, 2, 8, 1, 6).to_json()
    assert got["e"].to_json() == Explicit((("n", 2, 0), ("p", 4, 1), ("n", 1, 3))).to_json()
    assert type(got["c"]) is Concat and type(got["c"].parts[1]) is Concat and got["c"].parts[0] is c.parts[0]
    assert got["c"].parts[1].parts == [got["s"], got["e"]]
    unchanged = Concat([Affine("n", 0, 0, 2), p])
    assert _renumbered(unchanged, new, memo) is unchanged
    read: set[int] = set()
    _read(c, read)
    _read(empty, read)
    assert read == {0, 2, 4}
