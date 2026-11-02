"""te_core.score: a run's checkpoints as prefixes of its count, and te_core.annotate's traced structures."""
from te_core.annotate import annotate, with_structures
from te_core.score import C_PER_FORWARD_TOKEN, block_c, checkpoints


def test_c_is_4096_forward_passes():
    assert block_c(10) == 10 * C_PER_FORWARD_TOKEN == 40_960


def record(n_steps: int) -> dict:
    """A harness record of a toy run: one traced structure, every step the same."""
    struct = {"updated": [[100, i * 100, 100 * 8, 8, 16] for i in range(4)], "log": [], "beta": 2.0, "cut_bits": 800}
    steps = [{"step": t, "tokens": 8, "macs": 3 * 400 * 8, "by": {}, "struct": "s", "weights": {}, "shapes": {}}
             for t in range(n_steps)]
    return {"setup": {"macs": 7, "tokens": 0, "by": {}}, "tail": {"macs": 5, "tokens": 0, "by": {}}, "steps": steps,
            "structures": {"s": struct},
            "checkpoints": [{"step": s, "val_bpb": 1.0 - 0.01 * s} for s in (4, 9)]}


def test_checkpoints_are_prefixes():
    rec = record(10)
    a4 = annotate(rec, through=4)
    assert a4.macs == 7 + 5 * 9600
    assert annotate(rec).macs == 7 + 10 * 9600 + 5
    subs = checkpoints(rec)
    assert [s["macs"] for s in subs] == [7 + 5 * 9600, 7 + 10 * 9600 + 5]
    assert subs[0]["annotation"]["steps"] == 5
    assert subs[0]["annotation"]["beta"] == [2.0, 2.0] and subs[0]["annotation"]["untraced"] == 0
    assert a4.depth() == 300


def test_with_structures_scales_the_traced_rows_and_reprices_the_families():
    """The runtime counter's rows (4 nodes of 100 MACs per token, no cut) give way to a traced structure of 2 nodes of 50
    per token, scaled to each step's tokens; step 0 updates nothing and keeps its own; a family's β_t is bits(E_t) over
    the traced updated MACs per token block (4x its old value here), and |C| is the runtime count."""
    rec = record(3)
    rec["structures"] = {"s": {"updated": [[100, i * 100, 800, 8, 16] for i in range(4)], "log": [["x"]]},
                         "z": {"updated": [], "log": []}}
    rec["steps"][0]["struct"] = "z"
    rec["steps"][2]["tokens"] = 4
    rec["families"] = [{"covers": [1, 2], "beta": {"1": 1.0, "2": 2.0}}]
    traced = {"updated": [[50, 0, 50 * 16, 16, 32], [50, 50, 50 * 16, 16, 16]], "beta": 3.0, "cut_bits": 300}
    out = with_structures(rec, {"s": traced, "z": traced})
    ann = annotate(out)
    assert ann.macs == annotate(rec).macs
    assert [s.updated_macs for s in ann.steps] == [0, 100 * 8, 100 * 4]
    assert [s.beta for s in ann.steps] == [None, 3.0, 3.0] and ann.depth() == 50 and ann.bits() == 16
    assert out["families"][0]["beta"] == {"1": 4.0, "2": 8.0}
    assert out["structures"][out["steps"][1]["struct"]]["log"] == [["x"]]
    assert rec["structures"].keys() == {"s", "z"} and rec["families"][0]["beta"] == {"1": 1.0, "2": 2.0}
