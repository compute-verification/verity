import hashlib
import json

import pytest
from synthetic import SHAPE, tiny_registry, write_config

from training_slowdown import config, library, of_record, starts

CTX, D = SHAPE["ctx"], SHAPE["d_model"]
X, S, G = [1, CTX, D], [1, CTX, 1], [D]
Q = [1, 2, CTX, 8]
ATTENTION = list(of_record.ATTENTION.items())


def log(*entries) -> dict:
    """A step's log as `per_row` gives it, from `(phase, kind, op, operand shapes, count, MACs)`."""
    return {(phase, kind, op, json.dumps(shapes)): [count, macs] for phase, kind, op, shapes, count, macs in entries}


def step(log_: dict, updated=((D, 0, 16),)) -> dict:
    return {"rows": 1, "updated": sorted(updated), "log": log_}


#: products both sides have: a weight's product and its gradient's
BOTH = (("fwd", "updated", "mm", [[CTX, D], [D, D]], 4, 4 * CTX * D * D),
        ("bwd", "grad", "mm", [[D, CTX], [CTX, D]], 4, 4 * CTX * D * D))


def test_a_steps_rows_a_forward_are_divided_out_of_its_token_shapes_and_their_macs():
    macs = 3 * CTX * D * D
    struct = {"updated": [[D * D, D, macs, 3 * CTX, 16], [D, 0, 3 * CTX * D, 3 * CTX, 32]], "log": [
        ["fwd", "updated", "mm", json.dumps([[3 * CTX, D], [D, D]]), 2, 2 * macs],
        ["fwd", "act", "mul", json.dumps([[3, CTX, D], [3, CTX, 1]]), 5, 5 * 3 * CTX * D],
        ["other", "param", "_foreach_mul_", json.dumps([[D, D], [D], [3, D]]), 1, 2 * D * D + D],
        ["other", "param", "mm", json.dumps([[D, 3], [3, D]]), 7, 7 * 3 * D * D]]}
    got = of_record.per_row(struct, CTX)
    assert got["rows"] == 3 and got["updated"] == [(D, 0, 32), (D * D, D, 16)]
    assert got["log"] == log(("fwd", "updated", "mm", [[CTX, D], [D, D]], 2, 2 * CTX * D * D),
                             ("fwd", "act", "mul", [X, S], 5, 5 * CTX * D),
                             ("other", "param", "_foreach_mul_", [[3, D], [D], [D, D]], 1, 2 * D * D + D),
                             ("other", "param", "mm", [[D, 3], [3, D]], 7, 7 * 3 * D * D))
    with pytest.raises(ValueError, match="not one whole number of rows of 1024"):
        of_record.per_row(struct | {"updated": [*struct["updated"], [D, 0, CTX * D, CTX, 16]]}, CTX)


def test_the_cpus_attention_kernels_are_expected_for_cudas_of_the_same_count_and_macs():
    fwd, bwd = [Q] * 3, [Q] * 5 + [Q[:3]]
    cuda = log(*BOTH, ("fwd", "act", ATTENTION[0][0], fwd, 2, 900), ("bwd", "grad", ATTENTION[1][0],
                                                                     bwd + of_record.CUDA_ONLY_OPERANDS, 2, 1900))
    cpu = log(*BOTH, ("fwd", "act", ATTENTION[0][1], fwd, 2, 900), ("bwd", "grad", ATTENTION[1][1], bwd, 2, 1900))
    got = of_record.compare(step(cpu), step(cuda), SHAPE)
    assert got["differences"] == [] and got["products"] == 2
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [("attention", 2800, 2800)]
    once = log(*BOTH, ("fwd", "act", ATTENTION[0][1], fwd, 1, 450), ("bwd", "grad", ATTENTION[1][1], bwd, 2, 1900))
    got = of_record.compare(step(once), step(cuda), SHAPE)
    assert [e["rule"] for e in got["expected"]] == ["attention"]  # the backward's pair alone
    assert got["differences"] == [
        f"log, the record's step only: fwd/act {ATTENTION[0][0]} [[1,2,1024,8] x3] x2, 900 MACs",
        f"log, the config's step only: fwd/act {ATTENTION[0][1]} [[1,2,1024,8] x3] x1, 450 MACs"]


def norms(updated: int, frozen: int, scales: int) -> tuple:
    """The norm's products on a CPU: `updated` gains' multiplies, `frozen` of θ's gains, `scales` normalisations and
    a backward through each."""
    n = CTX * D
    return (("fwd", "updated", "mul", [X, G], updated, updated * n),
            ("fwd", "theta", "mul", [X, G], frozen, frozen * n), ("fwd", "act", "mul", [X, S], scales, scales * n),
            ("bwd", "grad", "div", [X], scales, scales * n), ("bwd", "grad", "mul", [X, X], 2 * scales, 2 * scales * n),
            ("bwd", "grad", "mul", [S, S], scales, scales * CTX))


def test_the_cpus_norm_is_expected_for_cudas_fused_kernel_with_one_normalisation_a_gain():
    n = CTX * D
    cuda = log(*BOTH, ("fwd", "updated", "_fused_rms_norm", [X, G], 3, 3 * n))
    got = of_record.compare(step(log(*BOTH, *norms(3, 2, 5))), step(cuda), SHAPE)
    assert got["differences"] == []
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [
        ("norm", 3 * n, 3 * n + 2 * n + 5 * n + 5 * n + 10 * n + 5 * CTX)]
    assert "3 multiplies by an updated gain" in got["expected"][0]["what"]
    # a normalisation with no gain's multiply, another count of updated gains, a product of the backward's the
    # record has too: none is the norm's exact case, so each side's entries are differences
    for cpu, record in ((norms(3, 2, 6), cuda), (norms(4, 1, 5), cuda),
                        (norms(3, 2, 5), cuda | log(("bwd", "grad", "mul", [X, X], 1, n)))):
        got = of_record.compare(step(log(*BOTH, *cpu)), step(record), SHAPE)
        assert got["expected"] == [] and len(got["differences"]) >= 7
        assert any("fwd/updated _fused_rms_norm" in d for d in got["differences"])
    # a product over another shape is no part of the norm
    wide = log(*BOTH, *norms(3, 2, 5), ("fwd", "act", "mul", [X, X], 1, n))
    got = of_record.compare(step(wide), step(cuda), SHAPE)
    assert [e["rule"] for e in got["expected"]] == ["norm"]
    assert got["differences"] == ["log, the config's step only: fwd/act mul [[1,1024,16] x2] x1, 16,384 MACs"]


def test_the_cpus_norm_is_expected_where_the_step_trains_no_gain_and_the_record_has_nothing_of_it():
    """A step that trains no gain: CUDA's counter is shown nothing of the fused kernel, so the record has no entry of
    the norm, and the CPU's side has every gain's multiply as θ's. Expected, with one normalisation a gain; not where
    a normalisation has no gain's multiply, where the record has one of the backward's products too, or where the
    config multiplies by an updated gain the record never counted."""
    n = CTX * D
    frozen = norms(0, 5, 5)[1:]
    got = of_record.compare(step(log(*BOTH, *frozen)), step(log(*BOTH)), SHAPE)
    assert got["differences"] == [] and got["products"] == len(BOTH)
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [
        ("norm", 0, 5 * n + 5 * n + 5 * n + 10 * n + 5 * CTX)]
    assert "no gain is updated" in got["expected"][0]["what"] and "5 by θ's gains" in got["expected"][0]["what"]
    for cpu, record in ((norms(0, 5, 6)[1:], log(*BOTH)), (norms(1, 4, 5), log(*BOTH)),
                        (frozen, log(*BOTH, ("bwd", "grad", "mul", [X, X], 1, n)))):
        got = of_record.compare(step(log(*BOTH, *cpu)), step(record), SHAPE)
        assert got["expected"] == [] and len(got["differences"]) >= 5
    # a step with no norm at all on either side is equal, with no rule fired
    got = of_record.compare(step(log(*BOTH)), step(log(*BOTH)), SHAPE)
    assert got["expected"] == [] and got["differences"] == []


def test_a_plans_codecs_must_each_be_of_the_records_kind_of_encoding(monkeypatch):
    """A record's format may have several codec kinds (here a made-up second one for `quant`). A plan of either, or
    of both, is that kind; one that keeps a tensor whole is not, and neither is any plan beside a format this table
    doesn't know."""
    assert of_record.e_kind(["scaledquant"], "quant") == [] == of_record.e_kind(["identity"], "full")
    monkeypatch.setitem(of_record.E_KINDS, "quant", ("scaledquant", "coded"))
    for kinds in (["scaledquant"], ["coded"], ["scaledquant", "coded"]):
        assert of_record.e_kind(kinds, "quant") == []
    assert of_record.e_kind(["identity", "coded"], "quant") == [
        "E's kind: the config's plan is identity, coded, the record's format quant (scaledquant or coded)"]
    assert of_record.e_kind(["coded"], "full") == [
        "E's kind: the config's plan is coded, the record's format full (identity)"]
    assert of_record.e_kind(["identity"], "model") == [
        "E's kind: the config's plan is identity, the record's format model (none known here)"]
    assert of_record.e_kind([], "full") == ["E's kind: the config's plan is empty, the record's format full (identity)"]
    # the sizes are printed and not compared; a codec that packs the same encoding another way is named as that
    last = {"format": "quant", "e_bytes": 1000}
    assert of_record.e_sizes(["scaledquant"], 1200, last, 30) == (
        "scaledquant, 1,200 B after 2 steps; the record's quant, 1,000 B after 30")
    monkeypatch.setitem(of_record.E_PACKING, "coded", "one stream a matrix")
    assert of_record.e_sizes(["coded"], 1200, last, 30) == (
        "coded, 1,200 B after 2 steps; the record's quant, 1,000 B after 30; expected to differ in size by the "
        "packing too: coded is one stream a matrix")


def unfused(params: list, times: int = 1) -> tuple:
    return tuple(("other", "param", op, [p] * operands, times * params.count(p), times * params.count(p) * p[0])
                 for p in ([D], [2 * D]) if p in params for op, operands in of_record.UNFUSED_ADAMW)


def test_the_unfused_adamw_is_expected_for_cudas_fused_kernel_over_the_same_parameters():
    params = [[D], [D], [2 * D]]
    fused = ("other", "param", "_fused_adamw_", params * 4 + [[]] * 3, 1, 8 * 4 * D)
    got = of_record.compare(step(log(*BOTH, *unfused(params))), step(log(*BOTH, fused)), SHAPE)
    assert got["differences"] == []
    assert [(e["rule"], e["record_macs"], e["config_macs"]) for e in got["expected"]] == [("adamw", 32 * D, 20 * D)]
    # other parameters, a product twice a parameter (a weight decay), a kernel priced otherwise: not its exact case
    for cpu, record in ((unfused(params[:2]), fused), (unfused(params, 2), fused),
                        (unfused(params), (*fused[:5], 7 * 4 * D))):
        got = of_record.compare(step(log(*BOTH, *cpu)), step(log(*BOTH, record)), SHAPE)
        assert got["expected"] == [] and any("_fused_adamw_" in d for d in got["differences"])


def test_32_bits_for_the_records_16_is_expected_and_any_other_use_differs():
    record = [(D * D, D, 16), (D * D, D, 16), (D, 0, 32)]
    same = log(*BOTH)
    got = of_record.compare(step(same, [(D * D, D, 32), (D * D, D, 32), (D, 0, 32)]), step(same, record), SHAPE)
    assert got["differences"] == [] and got["uses"] == 3
    assert [e["rule"] for e in got["expected"]] == ["width"] and got["expected"][0]["what"].startswith("2 of the uses")
    assert of_record.compare(step(same, record), step(same, record), SHAPE)["expected"] == []
    got = of_record.compare(step(same, [(D * D, D, 16), (D * D, D, 16), (D, 0, 16)]), step(same, record), SHAPE)
    assert got["expected"] == [] and got["differences"] == [
        "updated, the weights' bits: the config's (16, 0) at 16 x1; the record's (16, 0) at 32 x1"]
    got = of_record.compare(step(same, [(D * D, 2 * D, 32), (D * D, D, 32), (D, 0, 32)]), step(same, record), SHAPE)
    assert got["expected"] == [] and got["uses"] == 0 and got["differences"] == [
        "updated, the config's step only: (256 MACs a token, 32 upstream) x1",
        "updated, the record's step only: (256 MACs a token, 16 upstream) x1"]


def test_a_step_stands_for_no_run_when_the_capture_refuses_it_or_it_reads_a_value_back():
    trace = {"data_dependent": [], "carried_scalars": []}
    assert of_record.refusal({"trace": trace, "invalid": []}) is None
    refused = {"refused": {"reason": "dynamic_control_flow", "op": "aten.item", "message": "why"}}
    assert of_record.refusal({"trace": refused, "invalid": []}) == (
        "the capture refused the step (dynamic_control_flow, aten.item: why)")
    assert "reads a value back" in of_record.refusal({"trace": trace | {"data_dependent": ["a clip"]}, "invalid": []})
    assert of_record.refusal({"trace": trace, "invalid": ["no step"]}) == "the count is invalid (no step)"


def test_every_held_entry_is_an_entry_of_the_library_with_its_record_named():
    runs = of_record.held()
    assert set(runs) <= {p.stem for p in library.entries()}
    for h in runs.values():
        assert set(h) - {"record_only"} == {"run", "art", "path", "sha256"}
        assert len(h["sha256"]) == 64 and h["art"].startswith("art:")
        assert all(set(line) == {"phase", "kind", "op", "shapes", "count", "macs", "why"}
                   for line in h.get("record_only", []))


#: a line of a record's own program: a scale times a multiplier, scalars both
SCALE = {"phase": "fwd", "kind": "act", "op": "mul", "shapes": [[], []], "count": 3, "macs": 3,
         "why": "its multiplier is 1"}


def test_a_line_the_entry_names_is_the_records_over_the_configs_and_one_the_record_lacks_is_a_difference():
    scale = ("fwd", "act", "mul", [[], []])
    record = log(*BOTH, (*scale, 3, 3))
    got = of_record.compare(step(log(*BOTH)), step(record), SHAPE, [SCALE])
    assert got["differences"] == [] and got["expected"] == [] and got["products"] == 2
    assert got["named"] == [{"line": "fwd/act mul [[] x2] x3, 3 MACs", "why": "its multiplier is 1"}]
    # a line the config has one of and the record four: the three over the config's are the named ones
    got = of_record.compare(step(log(*BOTH, (*scale, 1, 1))), step(log(*BOTH, (*scale, 4, 4))), SHAPE, [SCALE])
    assert got["differences"] == [] and [e["line"] for e in got["named"]] == ["fwd/act mul [[] x2] x3, 3 MACs"]
    # unnamed, the line is a difference; named with another count, or where the config computes it too, the name is
    assert of_record.compare(step(log(*BOTH)), step(record), SHAPE)["differences"] == [
        "log, the record's step only: fwd/act mul [[] x2] x3, 3 MACs"]
    stale = "of_record.toml names a line the record's step does not have over the config's as named: fwd/act mul "
    got = of_record.compare(step(log(*BOTH)), step(record), SHAPE, [SCALE | {"count": 2, "macs": 2}])
    assert got["named"] == [] and got["differences"] == [
        stale + "[[] x2] x2, 2 MACs", "log, the record's step only: fwd/act mul [[] x2] x3, 3 MACs"]
    got = of_record.compare(step(log(*BOTH, (*scale, 2, 2))), step(record), SHAPE, [SCALE])
    assert got["named"] == [] and got["differences"] == [
        stale + "[[] x2] x3, 3 MACs",
        "log, unequal: fwd/act mul [[] x2] x2, 2 MACs in the config's step, x3, 3 MACs in the record's"]
    got = of_record.compare(step(record), step(record), SHAPE, [SCALE])
    assert got["named"] == [] and got["differences"] == [stale + "[[] x2] x3, 3 MACs"] and got["products"] == 3


def count(*more, updated=((D * D, D),), macs=None) -> dict:
    """A config's counted record made up here: every step of one structure, `BOTH` and `more` its log."""
    entries = (*BOTH, *more)
    struct = {"updated": [[p, u, CTX * p, CTX, 16] for p, u in updated],
              "log": [[*e[:3], json.dumps(e[3]), *e[4:]] for e in entries]}
    total = sum(e[5] for e in entries) if macs is None else macs
    return {"invalid": [], "trace": {"data_dependent": [], "carried_scalars": []}, "checkpoints": [{"e_bytes": 10}],
            "steps": [{"struct": "s", "macs": total}] * of_record.STEPS, "structures": {"s": struct}}


def test_the_position_tables_are_expected_where_the_two_counts_differ_in_the_weights_kind_alone(tmp_path, monkeypatch):
    """The harness's rule on counts made up here: the harness's own count has a product with a table as an updated
    weight's and more uses; the count as the records' harness took it has that product as θ's."""
    cfg = config.load(write_config(tmp_path / "a.toml"), starts.registry(tiny_registry(tmp_path / "reg")))
    rope = [[1, 2, CTX, 4], [CTX, 4]]
    own = count(("fwd", "updated", "mul", rope, 8, 8 * CTX * 8), updated=((D * D, D), (8, 0), (8, D)))
    own["structures"]["s"]["updated"][2][3] = 2 * CTX  # a table's use in a step's second forward: no row's tokens
    as_recorded = count(("fwd", "theta", "mul", rope, 8, 8 * CTX * 8))
    asked = []

    def counted(cfg, rows, out, harness, trace):
        asked.append((rows, out, harness, trace))
        return as_recorded

    monkeypatch.setattr(of_record, "counted", counted)
    again, found, differences = of_record.tables(cfg, 1, tmp_path / "a", own)
    assert again is as_recorded and differences == []
    assert asked == [(1, tmp_path / "a" / "tables", of_record.AsRecorded, own["trace"])]
    assert found["rule"] == "tables" and (
        "8 of the forward's products (65,536 MACs a step at one row a forward) are an updated weight's in the "
        "harness's own count and θ's here, and its own count has 3 uses of an updated weight for these 1; the step's "
        "MACs are 2,162,688 on both counts") in found["what"]
    # counts that agree: nothing to name; counts that differ in another product, or in the step's MACs: a difference
    assert of_record.tables(cfg, 1, tmp_path / "a", as_recorded) == (as_recorded, None, [])
    for other in (count(("fwd", "act", "mul", rope, 8, 8 * CTX * 8)),
                  count(("fwd", "updated", "mul", rope, 8, 8 * CTX * 8), macs=1)):
        again, found, differences = of_record.tables(cfg, 1, tmp_path / "a", other)
        assert found is None and len(differences) == 1 and differences[0].startswith(
            "the position tables: the harness's own count and the count with them checked like θ's weights differ in "
            "more than which of the forward's products are an updated weight's (the step's MACs: ")
    assert of_record.HARNESS == (of_record.tables,)


def as_cuda(struct: dict, rows: int, gains: int) -> dict:
    """A CPU step's structure as CUDA's counter writes the step: attention under CUDA's names, the norm one fused
    kernel counted at its updated gains alone, AdamW one fused op over the `gains`, the weights read at 16 bits."""
    x, s = [rows, CTX, D], [rows, CTX, 1]
    cpu, steps, out = {v: k for k, v in ATTENTION}, {op for op, _ in of_record.UNFUSED_ADAMW}, []
    for phase, kind, op, shapes, count, macs in struct["log"]:
        at = json.loads(shapes)
        if op in cpu:
            at += of_record.CUDA_ONLY_OPERANDS if op.endswith("backward") else []
            op = cpu[op]
        elif (phase, kind, op, at) == ("fwd", "updated", "mul", [x, G]):
            op = "_fused_rms_norm"
        elif op in ("mul", "div") and all(o in (x, s, G) for o in at) and phase != "other":
            continue
        elif phase == "other" and op in steps and all(o == G for o in at):
            continue
        out.append([phase, kind, op, json.dumps(at), count, macs])
    out.append(["other", "param", "_fused_adamw_", json.dumps([G] * 4 * gains + [[]] * gains), 1, 8 * gains * D])
    return {"updated": [[*u[:4], 16] for u in struct["updated"]], "log": out}


def held_to(root, entry: str, rec: dict, record_only=()):
    """`rec` as `entry`'s run of record under `root`: where its records are, and the data file naming it and the
    lines `record_only` of its program."""
    run, path = "r-made-up", "out/run/run.json"
    (root / "records" / run / "out" / "run").mkdir(parents=True)
    raw = json.dumps(rec).encode()
    (root / "records" / run / path).write_bytes(raw)
    lines = "".join(f'[[held."{entry}".record_only]]\n' + "".join(f"{k} = {json.dumps(v)}\n" for k, v in line.items())
                    for line in record_only)
    (root / "held.toml").write_text(f'[held."{entry}"]\nrun = "{run}"\nart = "art:{"0" * 64}"\npath = "{path}"\n'
                                    f'sha256 = "{hashlib.sha256(raw).hexdigest()}"\n{lines}')
    return root / "records", root / "held.toml"


@pytest.mark.slow
def test_the_command_names_each_expected_difference_and_exits_1_on_any_other(tmp_path, capsys):
    """A made-up run of record: the config's own counted step at two rows a forward, counted as the records' harness
    counted and written as CUDA's counter writes it. The command divides its rows out, counts the config at one row a
    forward and names the five expected differences; a config that trains another set, beside a record of another
    start and length, differs and exits 1. It counts and traces a route's steps, so slow; what the rules' own tests
    can't show is that a route's real step produces the entries each rule names and that the command reads a record
    the way a job writes one."""
    reg = tiny_registry(tmp_path / "reg")
    budget, dials = {"steps": 3, "batch": 4}, {"trainable": "blocks.*,norm", "opt": "muon", "lr": 1e-2,
                                               "micro_batches": 2}
    path = write_config(tmp_path / "a.toml", budget=budget, dials=dials)
    cfg = config.load(path, starts.registry(reg))
    cpu = of_record.counted(cfg, 1, tmp_path / "count", of_record.AsRecorded)
    assert cpu["invalid"] == [] and cpu["checkpoints"][-1]["step"] == 1
    first, second = cpu["steps"]
    tables = [kind for _, kind, op, shapes, *_ in cpu["structures"][second["struct"]]["log"]
              if op == "mul" and json.loads(shapes)[-1] == [CTX, 4]]
    assert set(tables) == {"theta", "grad"}  # a product with a position table is θ's, as the records count it
    rec = cpu | {"steps": [first, second, second], "start_record": {"name": "tiny", "digest": cfg.start.digest},
                 "structures": cpu["structures"] | {
                     second["struct"]: as_cuda(cpu["structures"][second["struct"]], rows=2, gains=3)}}
    records, held = held_to(tmp_path, "a", rec)
    argv = ["--records", str(records), "--held", str(held), "--starts", str(reg)]
    assert of_record.main([str(path), *argv, "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal, 5 expected differences\n")
    for rule in ("tables: ", "width: ", "attention: ", "norm: ", "adamw: "):
        assert f"  expected  {rule}" in out
    assert "are an updated weight's in the harness's own count and θ's here" in out
    assert "2 forwards of one row" in out and "2 of its 3 steps have, 2 rows a forward divided out" in out
    assert "DIFFERS" not in out and out.rstrip().endswith(of_record.FAIR) and "1 entry: 1 equal, 0 differ" in out
    kept = json.loads((tmp_path / "out" / "of_record.json").read_text())
    assert kept["schema"] == of_record.SCHEMA and kept["rows"][0]["differences"] == []
    own, compared = (json.loads((tmp_path / "out" / "a" / at / "run.json").read_text()) for at in ("", "tables"))
    assert own["trace"] == compared["trace"] and own["trace"]["tokens"] == 4 * CTX
    own, compared = own["steps"][1], compared["steps"][1]
    assert own["macs"] == compared["macs"] and own["by"] != compared["by"]

    other = write_config(tmp_path / "b" / "a.toml", budget=budget, dials=dials | {"trainable": "blocks.*"})
    rec["start_record"]["digest"] = "e" * 64
    records, held = held_to(tmp_path / "elsewhere", "a", rec | {"steps": rec["steps"] * 2})
    assert of_record.main([str(other), "--records", str(records), "--held", str(held), "--starts", str(reg)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): differs, ")
    assert "  DIFFERS   the start: the config's is tiny (dddddddddddd), the record's tiny (eeeeeeeeeeee)" in out
    assert "  DIFFERS   the steps: the config's 3, the record's 6" in out
    assert "  DIFFERS   updated, the record's step only: " in out and "1 entry: 0 equal, 1 differ" in out
    assert "  DIFFERS   log, the record's step only: fwd/updated _fused_rms_norm" in out


def test_a_difference_no_rule_names_is_printed_and_the_command_exits_1(tmp_path, capsys, monkeypatch):
    """The command on a count made up here, so no step is counted: a step equal to its record's exits 0, and so does
    one whose record has a line more that the data file names; a product the record lacks, which no rule names, is
    printed as a difference and exits 1; so does a step the capture refused, which is compared with nothing."""
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    cfg = config.load(path, starts.registry(reg))
    mine = count()
    monkeypatch.setattr(of_record, "counted", lambda *a: mine)
    rec = {"start_record": {"name": "tiny", "digest": cfg.start.digest}, "structures": dict(mine["structures"]),
           "steps": [{"struct": "s"}] * cfg.budget["steps"], "checkpoints": [{"format": "full", "e_bytes": 20}]}
    records, held = held_to(tmp_path, "a", rec)
    argv = [str(path), "--records", str(records), "--held", str(held), "--starts", str(reg)]
    assert of_record.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal\n") and "DIFFERS" not in out
    assert ("  step      the config's step 2, 2 forwards of one row, counted on a CPU, beside the record's modal "
            "structure, which 3 of its 3 steps have, 1 row a forward divided out\n") in out
    assert "  updated   1 of the record's 1 uses" in out and "  log       2 of the record's 2 products" in out
    assert "  E         identity, 10 B after 2 steps; the record's full, 20 B after 3" in out

    scaled = rec | {"structures": count(("fwd", "act", "mul", [[], []], 3, 3))["structures"]}
    records, named = held_to(tmp_path / "named", "a", scaled, [SCALE])
    assert of_record.main([str(path), "--records", str(records), "--held", str(named), "--starts", str(reg)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal, 1 named line of the record's program "
                          "alone\n") and "DIFFERS" not in out and "1 entry: 1 equal, 0 differ, 0 refused." in out
    assert ("  named     in the record's step and not the config's: fwd/act mul [[] x2] x3, 3 MACs; of_record.toml "
            "names it: its multiplier is 1\n") in out and "  log       2 of the record's 3 products" in out

    mine["structures"] = count(("fwd", "act", "matmul", [[CTX, D], [D, D]], 1, CTX * D * D))["structures"]
    assert of_record.main(argv) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): differs, 1 difference\n")
    assert "  DIFFERS   log, the config's step only: fwd/act matmul [[1024,16], [16,16]] x1, 262,144 MACs" in out
    assert "1 entry: 0 equal, 1 differ, 0 refused." in out

    mine["trace"] = {"refused": {"reason": "dynamic_control_flow", "op": "aten.item", "message": "why"}}
    assert of_record.main(argv) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): refused: the capture refused the step")
    assert "DIFFERS" not in out and "1 entry: 0 equal, 0 differ, 1 refused." in out


def test_a_record_that_is_missing_or_another_file_is_refused_with_its_fetch_line(tmp_path):
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    records, held = held_to(tmp_path, "a", {"steps": []})
    argv = [str(path), "--held", str(held), "--starts", str(reg), "--records"]
    with pytest.raises(SystemExit, match=r"a's record is not at .*nowhere/r-made-up/out/run/run.json \(`research "
                                         r"data fetch art:0+ --to .*nowhere/r-made-up --path out/run/run.json`\)"):
        of_record.main([*argv, str(tmp_path / "nowhere")])
    (records / "r-made-up" / "out" / "run" / "run.json").write_text("{}")
    with pytest.raises(SystemExit, match="is not the record of_record.toml names for a: its sha256 is 44136fa3"):
        of_record.main([*argv, str(records)])
    with pytest.raises(SystemExit, match="held.toml holds b to no run"):
        of_record.main([str(write_config(tmp_path / "b.toml")), *argv[1:], str(records)])
