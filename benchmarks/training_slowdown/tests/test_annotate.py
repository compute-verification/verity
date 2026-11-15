"""training_slowdown.annotate: a run's annotation through a step as a prefix of its count."""
from training_slowdown.annotate import annotate


def record(n_steps: int) -> dict:
    """A harness record of a toy run: one traced structure, every step the same."""
    struct = {"updated": [[100, i * 100, 100 * 8, 8, 16] for i in range(4)], "log": [], "beta": 2.0, "cut_bits": 800}
    steps = [{"step": t, "tokens": 8, "macs": 3 * 400 * 8, "by": {}, "struct": "s", "weights": {}, "shapes": {}}
             for t in range(n_steps)]
    return {"setup": {"macs": 7, "tokens": 0, "by": {}}, "tail": {"macs": 5, "tokens": 0, "by": {}}, "steps": steps,
            "structures": {"s": struct},
            "checkpoints": [{"step": s, "bpb": 1.0 - 0.01 * s} for s in (4, 9)]}


def test_an_annotation_through_a_step_is_a_prefix():
    rec = record(10)
    a4 = annotate(rec, through=4)
    assert a4.macs == 7 + 5 * 9600
    assert annotate(rec).macs == 7 + 10 * 9600 + 5
    assert a4.depth() == 300
