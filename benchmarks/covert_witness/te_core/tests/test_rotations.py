"""te_core.rotations: a run's rotation products are named `tf32` from its record, and a row ingested before is
re-priced to what ingest writes now."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from te_core import gates, ledger, rotations, tiles
from te_core.ledger import Ledger, ingest
from te_core.rotations import Unread, labelled, named, repriced

#: the 45M start's shape (te_core.policy's `SHAPES`): d 512, d_ff 1408, 6 blocks, vocabulary 50,304
PARAMS, D, H, V = 45_029_888, 512, 1408, 50_304
T = 32768
DD, DOWN, UP = [[T, D], [D, D]], [[T, H], [H, D]], [[T, D], [D, H]]
TF32, BF16 = gates.per_mac("gemm/tf32"), gates.per_mac("gemm/bf16")


def row(phase, role, shape, n, op="mm"):
    macs = shape[0][0] * shape[0][1] * shape[1][1] * n
    return [phase, role, op, json.dumps(shape), n, macs]


def record(log, args, island=("wildcard", "train.py"), weights=("blocks.4.wq.weight", "blocks.5.wq.weight", "norm"),
           setup=(), steps=10):
    """A run of `steps` steps of one structure with `log`, two checkpoints, and `setup`'s log before them."""
    by = {}
    for r in log:
        by[f"{r[0]}/{r[1]}"] = by.get(f"{r[0]}/{r[1]}", 0) + r[5]
    structures = {"s": {"log": log, "updated": [[100, 0, 100 * T, T, 16]]}, "0": {"log": list(setup), "updated": []}}
    return {"island": island[0], "program": island[1], "script_args": list(args), "te_core": "3.20",
            "start": {"name": "s100", "digest": "d", "val_bpb": 1.2, "parameters": PARAMS},
            "setup": {"step": -1, "tokens": 0, "macs": sum(r[5] for r in setup), "struct": "0",
                      "by": {"other/param": sum(r[5] for r in setup)}},
            "tail": {"macs": 0, "tokens": 0, "by": {}, "struct": None},
            "steps": [{"step": t, "tokens": T, "macs": sum(by.values()), "by": dict(by), "struct": "s",
                       "weights": {w: 1 for w in weights}, "shapes": {}} for t in range(steps)],
            "structures": structures, "tier": 0, "seed": 1, "theta_sha256": "t", "c": 10,
            "totals": {"cpu_s": 1, "wall_s": 1, "gpu_s": 1},
            "checkpoints": [{"step": s, "val_bpb": 1.1 - 0.01 * s, "e_bytes": 30, "source": "encode"} for s in (4, 9)]}


def shallow(cut="4:q8,5:q8,6:q8"):
    """Blocks 4 and 5 of six trained, two micro-batches a step: each block's four attention products and three MLP
    products a pass, forward and backward for the trained two, and three cuts' rotations, the one at boundary 4 with
    no backward pass."""
    return record([row("fwd", "theta", DD, 4 * 4 * 2 + 3 * 2 * 2), row("fwd", "updated", DD, 2 * 4 * 2),
                   row("fwd", "theta", DOWN, 4 * 2), row("fwd", "updated", DOWN, 2 * 2),
                   row("fwd", "theta", UP, 4 * 2 * 2), row("fwd", "updated", UP, 2 * 2 * 2),
                   row("fwd", "theta", [[T, D], [D, V]], 2),
                   row("bwd", "grad", DD, 2 * 4 * 2 + 2 * 2 * 2), row("bwd", "grad", [[D, T], [T, D]], 2 * 4 * 2),
                   row("bwd", "grad", [[T, V], [V, D]], 2), row("bwd", "grad", UP, 2 * 2),
                   row("other", "param", [[D, D], [D, D]], 30)], ["--set", "blocks.[45]..*,norm", "--cut", cut])


def own_basis(cut="all:q4c"):
    """Every block trained, two micro-batches a step, each cut's forward coded in the residual's own basis: no forward
    rotation, and the gradient rotated across each of the six cuts."""
    return record([row("fwd", "updated", DD, 6 * 4 * 2), row("fwd", "updated", DOWN, 6 * 2),
                   row("fwd", "updated", UP, 6 * 2 * 2), row("fwd", "theta", [[T, D], [D, V]], 2),
                   row("bwd", "grad", DD, 6 * 4 * 2 + 6 * 2 * 2), row("bwd", "grad", [[D, T], [T, D]], 6 * 4 * 2),
                   row("bwd", "grad", [[T, V], [V, D]], 2)], ["--set", "blocks\\..*,norm", "--cut", cut],
                  weights=("blocks.0.wq.weight", "blocks.5.wq.weight", "norm"))


def keys(log):
    return {" ".join(str(x) for x in r[:4]): r[4:] for r in log}


def test_a_cuts_rotations_are_the_d_by_d_products_beyond_the_blocks_own():
    rec = shallow()
    out, counts = named(rec)
    assert counts == {"s": {"fwd": 12, "bwd": 8}}
    got, was = keys(out["structures"]["s"]["log"]), keys(rec["structures"]["s"]["log"])
    one = T * D * D
    assert got[f"fwd theta mm {json.dumps(DD)} tf32"] == [12, 12 * one]
    assert got[f"fwd theta mm {json.dumps(DD)}"] == [32, 32 * one]
    assert got[f"bwd grad mm {json.dumps(DD)} tf32"] == [8, 8 * one]
    assert got[f"bwd grad mm {json.dumps(DD)}"] == [16, 16 * one]
    assert sum(m for _, m in got.values()) == sum(m for _, m in was.values())
    assert {k: v for k, v in got.items() if "tf32" not in k and k in was and was[k] == v}.keys() == was.keys() - {
        f"fwd theta mm {json.dumps(DD)}", f"bwd grad mm {json.dumps(DD)}"}
    assert rec["structures"]["s"]["log"][0][4] == 44, "the record itself is left as it was"
    assert labelled(out)["structures"]["s"] is out["structures"]["s"], "a named structure is named once"
    assert rotations.macs(rec, through=4) == 5 * 20 * one and rotations.macs(rec) == 10 * 20 * one


def test_a_code_in_the_residuals_own_basis_rotates_only_the_gradient():
    rec = own_basis()
    out, counts = named(rec)
    assert counts == {"s": {"bwd": 24}}
    got = keys(out["structures"]["s"]["log"])
    one = T * D * D
    assert got[f"bwd grad mm {json.dumps(DD)} tf32"] == [24, 24 * one] and got[f"bwd grad mm {json.dumps(DD)}"] == [48, 48 * one]
    assert [k for k in got if k.endswith("tf32")] == [f"bwd grad mm {json.dumps(DD)} tf32"]
    assert rotations.cuts("all:q4c", 6) == ([1, 2, 3, 4, 5, 6], 0) and rotations.cuts("5:dq2cb4,6:q4", 6) == ([5, 6], 1)
    with pytest.raises(Unread, match="0 forward ones, which are no two a cut a pass"):
        named(own_basis("all:q4"))                   # six rotating cuts' forward products aren't in the log
    with pytest.raises(Unread, match="12 forward d x d products beyond the blocks' own"):
        named(shallow("4:q8c,5:q8c,6:q8c"))          # and a log that has them is no own-basis code's


def test_the_stored_arguments_have_to_agree_with_the_log():
    with pytest.raises(Unread, match="no two a cut a pass"):
        named(shallow("5:q8,6:q8,3:q4,2:q4,1:q4"))   # 12 forward products are no two a pass for five cuts
    with pytest.raises(Unread, match="1 cuts have a trained weight below"):
        named(shallow("3:q8,4:q8,5:q8"))            # two backward passes cross two cuts, and only one is above block 4
    with pytest.raises(Unread, match="p<keep>"):
        named(shallow("5:dq2p64b4,6:dq2p64b4"))
    with pytest.raises(Unread, match="doesn't know the program"):
        named(shallow() | {"island": "shallow"})
    with pytest.raises(Unread, match="no step of the record has a rotation product"):
        named(record([row("fwd", "theta", DD, 16), row("fwd", "theta", DOWN, 4)], ["--cut", "6:q4"]))
    plain = shallow() | {"script_args": ["--set", "blocks.[45]..*,norm"]}
    assert named(plain) == (plain, {}) and labelled(plain) is plain


def test_a_packed_updates_and_an_adapters_rotations_are_rows_of_their_own():
    packed = record([row("fwd", "updated", DD, 48), row("other", "param", [[2048, 64], [64, 64]], 54),
                     row("other", "param", [[4096, 64], [64, 64]], 72), row("other", "param", [[D, D], [D, D]], 75),
                     row("other", "param", [[2048, 128], [128, 128]], 3)],
                    ["--rotate", "--group", "64", "--opt", "muon"], ("lowrank-restart", "delta_grad.py"),
                    setup=[row("other", "param", [[2048, 64], [64, 64]], 18)])
    out, counts = named(packed)
    assert counts == {"s": {"other": 126}, "0": {"other": 18}}
    got = keys(out["structures"]["s"]["log"])
    assert sorted(k for k in got if k.endswith("tf32")) == [
        "other param mm [[2048, 64], [64, 64]] tf32", "other param mm [[4096, 64], [64, 64]] tf32"]
    assert named(packed | {"script_args": ["--rotate"]})[1] == {"s": {"other": 3}}, "--group is 128 unless given"
    z = record([row("fwd", "theta", DD, 48), row("fwd", "theta", DOWN, 12), row("fwd", "theta", [[T, 256], [256, 256]], 4),
                row("bwd", "grad", [[T, 256], [256, 256]], 4), row("fwd", "updated", [[T, D], [D, 256]], 2)],
               ["--z-rotate", "--adapter", "256"], weights=("new/0", "new/1"))
    assert named(z)[1] == {"s": {"fwd": 4, "bwd": 4}}
    with pytest.raises(Unread, match="not one of its own"):
        named(z | {"script_args": ["--z-rotate", "--adapter", str(D)]})


def test_a_named_row_is_priced_and_placed_as_a_tf32_gemm():
    rec = shallow()
    was, now = ledger._macs_log(rec, None), ledger._macs_log(labelled(rec), None)
    assert gates.of_log(now)["gates"] - gates.of_log(was)["gates"] == pytest.approx(rotations.macs(rec) * (TF32 - BF16))
    assert gates.of_log(now)["macs"] == gates.of_log(was)["macs"]
    wide = [o for o in tiles.of_log(now) if o["widths"] == [32, 32]]
    assert sorted(o["phase"] for o in wide) == ["bwd", "fwd"] and not [o for o in tiles.of_log(was) if o["widths"] == [32, 32]]
    assert all(o["products"] == [[1, T, D, D]] and o["gates"] == o["macs"] * TF32 for o in wide)
    assert tiles.unplaced(tiles.of_log(now)) == tiles.unplaced(tiles.of_log(was)) == 0


TRACED = {"updated": [[50, 0, 50 * 16, 16, 32]], "beta": 3.0, "cut_bits": 300,
          "formalism": {"sigma": 0.9, "affected_share": 0.4, "token_factor": 0.5, "delta_grad_bits": 64,
                        "tail_share": 0.0, "u_in": [[0, 0], [1, 10]], "tau_ids": 1, "tau_other_bits": 0, "u1_bits": 10}}


@pytest.mark.parametrize("structures", [None, {"s": TRACED}], ids=["counted", "traced"])
@pytest.mark.parametrize("which", ["cuts", "group"])
def test_a_repriced_row_is_what_ingest_writes_for_the_named_record(tmp_path, monkeypatch, structures, which):
    rec = shallow() if which == "cuts" else record(
        [row("fwd", "updated", DD, 48), row("other", "param", [[2048, 64], [64, 64]], 54)],
        ["--rotate", "--group", "64"], ("lowrank-restart", "delta_grad.py"),
        setup=[row("other", "param", [[2048, 64], [64, 64]], 18)])
    rec["steps"][0]["struct"] = "0"   # a first step before any update, which no trace renames
    rec["steps"][0]["by"] = dict(rec["setup"]["by"])
    run = tmp_path / "rec" / "run"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(rec))
    kw = {"direction": "d", "mechanism": "m", "run_id": "run", "structures": structures}
    new = ingest(run, Ledger(tmp_path / "new", "a"), **kw)
    monkeypatch.setattr(ledger, "labelled", lambda r: r)
    old = [json.loads(json.dumps(r)) for r in ingest(run, Ledger(tmp_path / "old", "a"), **kw)]
    assert [r["macs_log"] for r in old] != [r["macs_log"] for r in new]
    for a, b in zip(old, new, strict=True):
        got = repriced(a, rec)
        assert got["macs_log"] == json.loads(json.dumps(b["macs_log"]))
        assert len(got["formalism"]["parts"]) == len(b["formalism"]["parts"]) >= 2
        for p, q in zip(got["formalism"]["parts"], b["formalism"]["parts"]):
            assert p.keys() == q.keys()
            assert {k: v for k, v in p.items() if v is not None} == pytest.approx({k: v for k, v in q.items() if v is not None},
                                                                                rel=1e-12)
        assert {k: v for k, v in got.items() if k not in ("macs_log", "formalism")} == {
            k: v for k, v in a.items() if k not in ("macs_log", "formalism")}
        assert {k: v for k, v in got["formalism"].items() if k != "parts"} == {
            k: v for k, v in a["formalism"].items() if k != "parts"}
        more = sum(q["gates"] for q in b["formalism"]["parts"]) - sum(p["gates"] for p in a["formalism"]["parts"])
        through = None if a["checkpoint_step"] >= 9 else a["checkpoint_step"]
        assert more == pytest.approx(rotations.macs(rec, through) * (TF32 - BF16)) and more > 0
    with pytest.raises(Unread, match="another macs_log"):
        repriced(old[0] | {"checkpoint_step": 3}, rec)
    with pytest.raises(Unread, match="are not its record's: parts of"):
        f = old[0]["formalism"]
        repriced(old[0] | {"formalism": f | {"parts": [f["parts"][0] | {"gates": 1.0}, *f["parts"][1:]]}}, rec)


def test_a_row_ingested_with_its_lookups_priced_is_repriced_the_same_way(tmp_path, monkeypatch):
    # the lookups' names are read from covert_witness's `prepare`, whichever `prepare` a test before this one loaded
    spec = importlib.util.spec_from_file_location("cw_prepare_rotations", Path(__file__).resolve().parents[2] / "prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    monkeypatch.setitem(sys.modules, "prepare", prepare)
    rec = record([row("fwd", "updated", DD, 48), row("other", "param", [[2048, 64], [64, 64]], 54)],
                 ["--rotate", "--group", "64"], ("lowrank-restart", "delta_grad.py"), weights=("embed", "norm"))
    rec["te_core"] = "3.5.1"
    for s in rec["steps"]:
        s["shapes"] = {"embed": ["bfloat16", [V, D]]}
    run = tmp_path / "rec" / "run"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(rec))
    kw = {"direction": "d", "mechanism": "m", "run_id": "run", "lookups": True}
    new = ingest(run, Ledger(tmp_path / "new", "a"), **kw)
    monkeypatch.setattr(ledger, "labelled", lambda r: r)
    old = [json.loads(json.dumps(r)) for r in ingest(run, Ledger(tmp_path / "old", "a"), **kw)]
    assert old[0]["macs_log"]["by"]["fwd/updated"] > 5 * rec["steps"][0]["by"]["fwd/updated"], "the lookups are priced"
    for a, b in zip(old, new, strict=True):
        got = repriced(a, rec)
        assert got["macs_log"] == json.loads(json.dumps(b["macs_log"])) != a["macs_log"]
        assert got["formalism"] == json.loads(json.dumps(b["formalism"])) != a["formalism"]
