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
        assert set(h) == {"run", "art", "path", "sha256"} and len(h["sha256"]) == 64 and h["art"].startswith("art:")


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


def held_to(root, entry: str, rec: dict):
    """`rec` as `entry`'s run of record under `root`: where its records are, and the data file naming it."""
    run, path = "r-made-up", "out/run/run.json"
    (root / "records" / run / "out" / "run").mkdir(parents=True)
    raw = json.dumps(rec).encode()
    (root / "records" / run / path).write_bytes(raw)
    (root / "held.toml").write_text(f'[held."{entry}"]\nrun = "{run}"\nart = "art:{"0" * 64}"\npath = "{path}"\n'
                                    f'sha256 = "{hashlib.sha256(raw).hexdigest()}"\n')
    return root / "records", root / "held.toml"


@pytest.mark.slow
def test_the_command_names_each_expected_difference_and_exits_1_on_any_other(tmp_path, capsys):
    """A made-up run of record: the config's own counted step at two rows a forward, written as CUDA's counter
    writes it. The command divides its rows out, counts the config at one row a forward and names the four expected
    differences; a config that trains another set, beside a record of another start and length, differs and exits 1.
    It counts and traces three steps, so slow; what the rules' own tests can't show is that a route's real step
    produces the entries each rule names and that the command reads a record the way a job writes one."""
    reg = tiny_registry(tmp_path / "reg")
    budget, dials = {"steps": 3, "batch": 4}, {"trainable": "blocks.*,norm", "opt": "muon", "lr": 1e-2,
                                               "micro_batches": 2}
    path = write_config(tmp_path / "a.toml", budget=budget, dials=dials)
    cfg = config.load(path, starts.registry(reg))
    cpu = of_record.counted(cfg, 1, tmp_path / "count")
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
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal, 4 expected differences\n")
    for rule in ("width: ", "attention: ", "norm: ", "adamw: "):
        assert f"  expected  {rule}" in out
    assert "2 forwards of one row" in out and "2 of its 3 steps have, 2 rows a forward divided out" in out
    assert "DIFFERS" not in out and out.rstrip().endswith(of_record.FAIR) and "1 entry: 1 equal, 0 differ" in out
    kept = json.loads((tmp_path / "out" / "of_record.json").read_text())
    assert kept["schema"] == of_record.SCHEMA and kept["rows"][0]["differences"] == []
    assert json.loads((tmp_path / "out" / "a" / "run.json").read_text())["trace"]["tokens"] == 4 * CTX

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
    """The command on a count made up here, so no step is counted: a step equal to its record's exits 0; a product
    the record lacks, which no rule names, is printed as a difference and exits 1; so does a step the capture
    refused, which is compared with nothing."""
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    cfg = config.load(path, starts.registry(reg))

    def struct(*more) -> dict:
        return {"updated": [[D * D, D, CTX * D * D, CTX, 16]],
                "log": [[*e[:3], json.dumps(e[3]), *e[4:]] for e in (*BOTH, *more)]}

    count = {"invalid": [], "trace": {"data_dependent": [], "carried_scalars": []}, "checkpoints": [{"e_bytes": 10}],
             "steps": [{"struct": "s"}] * of_record.STEPS, "structures": {"s": struct()}}
    monkeypatch.setattr(of_record, "counted", lambda cfg, rows, out: count)
    rec = {"start_record": {"name": "tiny", "digest": cfg.start.digest}, "structures": {"s": struct()},
           "steps": [{"struct": "s"}] * cfg.budget["steps"], "checkpoints": [{"format": "full", "e_bytes": 20}]}
    records, held = held_to(tmp_path, "a", rec)
    argv = [str(path), "--records", str(records), "--held", str(held), "--starts", str(reg)]
    assert of_record.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): equal\n") and "DIFFERS" not in out
    assert "  updated   1 of the record's 1 uses" in out and "  log       2 of the record's 2 products" in out
    assert "  E         identity, 10 B after 2 steps; the record's full, 20 B after 3" in out

    count["structures"]["s"] = struct(("fwd", "act", "matmul", [[CTX, D], [D, D]], 1, CTX * D * D))
    assert of_record.main(argv) == 1
    out = capsys.readouterr().out
    assert out.startswith("a, held to r-made-up (out/run/run.json): differs, 1 difference\n")
    assert "  DIFFERS   log, the config's step only: fwd/act matmul [[1024,16], [16,16]] x1, 262,144 MACs" in out
    assert "1 entry: 0 equal, 1 differ, 0 refused." in out

    count["trace"] = {"refused": {"reason": "dynamic_control_flow", "op": "aten.item", "message": "why"}}
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
