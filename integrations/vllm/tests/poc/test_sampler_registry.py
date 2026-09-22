"""sampler.select plugged into W2's relation registry (skips when verity_vllm is not importable).

Run with both packages on sys.path, e.g. from a checkout containing W2's harness branch:
    PYTHONPATH=<harness>/vllm-poc:<sampler>/vllm-poc pytest tests/poc/test_sampler_registry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

verity_vllm = pytest.importorskip("verity_vllm")
from verity_vllm import dtypes, identity, relations, synthetic  # noqa: E402
from verity_vllm.capture import Value, json_value  # noqa: E402
from verity_vllm.description import Description, Node, NodeInstance  # noqa: E402

from verity_vllm_sampler import registry, select  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def installed():
    """Fill the slot for this module only; W2's registry tests expect it empty again afterwards."""
    registry.install()
    yield
    registry.uninstall()
    assert not relations.get(registry.RELATION_INDEPENDENT).implemented


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return synthetic.build_toy_run(tmp_path_factory.mktemp("toy") / "run", seed=3)


def test_slot_is_filled_by_install():
    rel = relations.get(registry.RELATION_INDEPENDENT)
    assert rel.implemented and rel.owner == "W3" and rel.klass == "independent"
    srt = relations.get(registry.RELATION_SAME_RUNTIME)
    assert srt.implemented and srt.klass == "same_runtime" and srt.includes_full_op
    assert registry.RELATION_INDEPENDENT not in relations.ids(implemented=False)
    # idempotent
    assert registry.install() == [registry.RELATION_INDEPENDENT, registry.RELATION_SAME_RUNTIME]


def _repoint(description: Description) -> Description:
    doc = description.to_json()
    for n in doc["nodes"]:
        if n["op"] == "sampler.select":
            n["relation"] = registry.RELATION_INDEPENDENT
            n["args"] = [
                {"name": "logits", "source": "node", "op": "lm_head.gemm", "port": 0},
                {"name": "sampling_params", "source": "boundary", "field": "sampling_params"},
                {"name": "seed", "source": "boundary", "field": "seed"},
            ]
    return Description(doc)


def test_greedy_path_through_w2_runner_is_exact(toy):
    """Toy capture is greedy (temperature 0): every committed sampler output is reproduced from the
    authenticated lm_head row through W2's runner, with the boundary seed/params leaves opened."""
    reader = toy.reader()
    desc = _repoint(toy.description)
    sample = []
    for rid in reader.request_ids():
        b = reader.boundary(rid)
        sample += [cid for cid, inst in desc.expected_ids(b, reader.profile_id).items() if inst.node.op == "sampler.select"]
    assert sample
    report = relations.check(desc, reader, toy.commitments, sample)
    assert report.all_exact(), [r.to_json() for r in report.rejected()]
    fam = report.families()["sampler"]["independent"]
    assert fam.exact == fam.attempted == len(sample)
    assert all(r.inputs_opened == 3 for r in report.results)  # logits + sampling_params + seed


def test_checker_depends_on_its_declared_input_operand(toy):
    """Contract §15.2(b): mutate the authenticated input operand (the lm_head row) of a selected sampler
    VU and confirm the executable relation predicts a different output, i.e. the checker is not
    returning the captured answer or ignoring its inputs."""
    from verity_vllm.commit import TreeCache

    reader = toy.reader()
    desc = _repoint(toy.description)
    rid = reader.request_ids()[0]
    b = reader.boundary(rid)
    cid, inst = next((c, i) for c, i in desc.expected_ids(b, reader.profile_id).items() if i.node.op == "sampler.select")
    cache = TreeCache(reader)
    committed = cache.open(cid).value
    resolved = desc.resolve_args(inst, b, reader.profile_id)
    inputs = {"__symbols__": desc.symbols}
    for name, ref in resolved.items():
        inputs[name] = cache.open(ref).as_value()
    rel = relations.get(registry.RELATION_INDEPENDENT)
    assert rel.checker(inst, inputs) == committed
    # move the maximum to another token: the relation must follow the operand, not the commitment
    row = inputs["logits"].float32().reshape(-1).copy()
    winner = int(np.argmax(row))
    other = (winner + 1) % row.size
    row[other] = row[winner] + 1.0
    mutated = Value(inputs["logits"].canonical_id, "bfloat16", inputs["logits"].shape, dtypes.f32_to_bf16_bytes(row), "captured")
    out = rel.checker(inst, {**inputs, "logits": mutated})
    assert out != committed and int(np.frombuffer(out, "<i4")[0]) == other


def _inst(t: int, dtype: str = "int32") -> NodeInstance:
    node = Node(op="sampler.select", layer="sampler", op_type="sampler", dtype=dtype, shape=(),
                relation=registry.RELATION_INDEPENDENT, positions="sampled")
    return NodeInstance(node, "req-x", t, "decode", 0, None, None)


def _inputs(logits_f32: np.ndarray, params: dict, seed, *, dtype: str = "bfloat16") -> dict:
    pid = "0123456789ab"
    lid = str(identity.value_id(pid, "req-x", 0, "decode", "lm_head", "lm_head.gemm", 0))
    if dtype == "bfloat16":
        raw = dtypes.f32_to_bf16_bytes(logits_f32)
    else:
        raw = dtypes.array_to_bytes(logits_f32.astype(np.float32))
    return {
        "__symbols__": {},
        "logits": Value(lid, dtype, (logits_f32.size,), raw, "captured"),
        "sampling_params": json_value(str(identity.boundary_id(pid, "req-x", "sampling_params")), params),
        "seed": json_value(str(identity.boundary_id(pid, "req-x", "seed")), seed),
    }


def test_stochastic_path_direct_matches_select_and_encodes_int32():
    rng = np.random.default_rng(11)
    V = 4096
    logits = (rng.standard_normal(V) * 3).astype(np.float32)
    logits[[5, 77, 900]] += 12.0  # a peaked head so certification is likely
    params = {"temperature": 1.0, "top_p": 0.9, "top_k": 50, "max_tokens": 8}
    rel = relations.get(registry.RELATION_INDEPENDENT)
    found_certified = False
    for pos in range(40, 60):
        x = dtypes.bf16_bits_to_f32(dtypes.f32_to_bf16_bits(logits))  # the bf16-rounded row the checker sees
        ref = select.independent_select(x, 1.0, 50, 0.9, 2, pos)
        inst = _inst(pos)
        if ref.certified:
            found_certified = True
            out = rel.checker(inst, _inputs(logits, params, 2))
            assert np.frombuffer(out, dtype="<i4")[0] == ref.predicted
        else:
            with pytest.raises(relations.RelationError, match="uncertified"):
                rel.checker(inst, _inputs(logits, params, 2))
    assert found_certified


def test_seed_from_params_when_no_seed_arg_and_misassociation_changes_token():
    rng = np.random.default_rng(5)
    V = 2048
    logits = rng.standard_normal(V).astype(np.float32)
    params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": 7}
    rel = relations.get(registry.RELATION_INDEPENDENT)
    inputs = _inputs(logits, params, 7, dtype="float32")
    del inputs["seed"]
    tokens = set()
    for pos in range(100, 130):
        try:
            tokens.add(int(np.frombuffer(rel.checker(_inst(pos), inputs), dtype="<i4")[0]))
        except relations.RelationError:
            pass
    assert len(tokens) > 1  # N09: the same row under another position's randomness selects other tokens


def test_unsupported_params_and_bad_inputs_are_errors_not_passes():
    rel = relations.get(registry.RELATION_INDEPENDENT)
    logits = np.zeros(64, np.float32)
    with pytest.raises(relations.RelationError, match="min_p"):
        rel.checker(_inst(3), _inputs(logits, {"temperature": 1.0, "min_p": 0.1}, 1))
    with pytest.raises(relations.RelationError, match="no seed"):
        rel.checker(_inst(3), _inputs(logits, {"temperature": 1.0}, None))
    with pytest.raises(relations.RelationError):
        rel.checker(_inst(3, dtype="bfloat16"), _inputs(logits, {"temperature": 0.0}, None))
    # greedy with ties: first index wins, encoded int64 on request
    logits[[9, 40]] = 5.0
    out = rel.checker(_inst(3, dtype="int64"), _inputs(logits, {"temperature": 0.0}, None, dtype="float32"))
    assert np.frombuffer(out, dtype="<i8")[0] == 9
