"""The integer MLP's Boolean Program computes its integer reference."""
from workload import model as M

SPEC = M.Spec(d=2, layers=3, requests=2, groups=2)


def test_the_circuit_computes_the_integer_reference():
    model = M.Model(SPEC)
    x, ws = M.requests(SPEC, 1), M.weights(SPEC, 1)
    t = model.bind_inputs(x, ws)
    model.evaluate(t)
    assert model.output_words(t) == M.reference(SPEC, x, ws)
