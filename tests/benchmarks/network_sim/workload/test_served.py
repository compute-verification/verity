"""verity_vllm's serving Program as the workload: it is `Serve_v1` with registered weights."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("verity_vllm")

from workload import served as S

CFG = S.Config()


@pytest.fixture(scope="module")
def model():
    return S.load(CFG)


def test_the_program_is_serve_v1_with_registered_weights(model):
    assert model.report.family["family"].startswith("Serve_v1{")
    kinds = {model.kind[g] for g in model.input_gates(1)}
    assert kinds == {"registered"} and {model.kind[g] for g in model.input_gates(0)} == {"run"}
    assert len(model.input_gates(0)) == CFG.prompt and len(model.outputs) == CFG.steps + 1
    rounds = [c.request for c in model.calls]
    assert rounds == sorted(rounds) and set(rounds) == set(range(CFG.steps + 1))
    for r in range(CFG.steps + 1):
        layers = [c.layer for c in model.calls if c.request == r]
        assert layers == sorted(layers) and layers[0] == 0 and layers[-1] == CFG.layers - 1
    tokens = model.reference(0)
    assert len(tokens) == CFG.steps + 1 and all(0 <= t < CFG.vocab for t in tokens)
