import copy
import json

import pytest
import torch
from synthetic import JOBS, LORA, job_counted
from torch.utils import flop_counter

from training_slowdown import as_math, flops, grade, run

#: each job's attention backward: the tiny lora's adapter is on q alone in the one block, with nothing trained below it
ASKED = {"lora": "q:1", "whole": "none"}
OTHER = {"lora": "none", "whole": "q:1"}
#: the same lora with its adapter on k: the products of its backward are also those of two calls asking for k's
#: gradient with q's or with v's
TRIED = {job: JOBS[job] for job in ASKED} | {
    "k": {"route": "lora", "dials": LORA | {"targets": "blocks.0.wk.weight,blocks.0.w_up.weight"}}}


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    """Jobs of the tiny start, each as the counter writes it and as a counter that holds a fused attention kernel as
    one row at torch's formula wrote it, that one with the job's trace and, as such a counter wrote, with no dtype in
    its op log: (fused, now) by job."""
    aten, formulas = torch.ops.aten, flop_counter.flop_registry
    one_row = dict(formulas) | {
        aten._scaled_dot_product_flash_attention_for_cpu: formulas[aten._scaled_dot_product_flash_attention],
        aten._scaled_dot_product_flash_attention_for_cpu_backward:
            formulas[aten._scaled_dot_product_flash_attention_backward]}
    out = {}
    for job, over in TRIED.items():
        root = tmp_path_factory.mktemp(job)
        now, (cfg, route, model, optimizers) = job_counted(root / "now", 3, **over)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(flops, "FUSED_ATTENTION", set())
            patch.setattr(flops, "FLOP_FORMULAS", one_row)
            fused, _ = job_counted(root / "fused", 3, **over)
        out[job] = (grade.without_dtypes(fused) | {"trace": run.trace_step(cfg, route, model, optimizers, "cpu")},
                    now)
    return out


def counts(rec: dict) -> tuple:
    """Every step's counts with its op log by row, whatever dtype a row names, and the record's totals."""
    keep = ("step", "tokens", "macs", "matmul_macs", "elementwise_macs", "approx_macs", "by", "device_macs")
    rec = grade.without_dtypes(rec)
    return ([{k: s.get(k) for k in keep} | {"log": {tuple(r[:4]): r[4:] for r in rec["structures"][s["struct"]]["log"]}}
             for s in (rec["setup"], *rec["steps"], rec["tail"])], rec["totals"]["macs"], rec["totals"]["device_macs"])


def rewritten(rec: dict, flag: str | None = None, **over) -> dict:
    rec = copy.deepcopy(rec) | over
    as_math.record(rec, as_math.parse(flag))
    return rec


def fused_rows(rec: dict) -> int:
    return sum("scaled_dot_product" in row[2] for st in rec["structures"].values() for row in st["log"])


@pytest.mark.parametrize("job", ASKED)
def test_the_gradients_asked_are_read_from_the_records_trace(records, job):
    """With no `--partial`, a record whose trace holds its attention's backward is rewritten at the ask the trace
    shows, and is then the record the counter writes of that job, row for row and count for count: for the job whose
    backward asks for q's gradient alone as for the one that asks for all three. The record says it was rewritten,
    and where its ask is from."""
    fused, now = records[job]
    assert fused_rows(fused) and not fused_rows(now) and counts(fused) != counts(now)
    got = rewritten(fused)
    assert counts(got) == counts(now) and not fused_rows(got)
    (how,) = got["provenance"]["as_math"]["gradients_asked"].values()
    assert how == {"asked": ASKED[job], "from": "its trace"} and "provenance" not in now


@pytest.mark.parametrize("job", ASKED)
def test_a_partial_that_disagrees_with_the_trace_is_refused_with_both_stated(records, job):
    fused, now = records[job]
    with pytest.raises(ValueError, match=f"trace says --partial {ASKED[job]} at .* command says --partial {OTHER[job]}"):
        rewritten(fused, OTHER[job])
    assert counts(rewritten(fused, ASKED[job])) == counts(now)


@pytest.mark.parametrize("job", ASKED)
@pytest.mark.parametrize("trace", [None, {"refused": {"reason": "dynamic_control_flow"}}, {"rows": []}], ids=str)
def test_a_record_with_no_trace_rows_is_refused_unless_partial_says_the_ask(records, job, trace):
    """Nothing is assumed of a record whose trace doesn't show its attention's backward: no `--partial` is refused,
    and `none` is a value like any other. The ask a person gives is the one taken, and a wrong one is a wrong count:
    all three where the step asked for q's alone counts more than the step ran."""
    fused, now = records[job]
    with pytest.raises(ValueError, match="holds no attention backward at .* say --partial none"):
        rewritten(fused, trace=trace)
    got = rewritten(fused, ASKED[job], trace=trace)
    assert counts(got) == counts(now)
    assert [h["from"] for h in got["provenance"]["as_math"]["gradients_asked"].values()] == ["--partial"]
    wrong = rewritten(fused, OTHER[job], trace=trace)
    assert (wrong["totals"]["macs"] > now["totals"]["macs"]) == (job == "lora") != (
        wrong["totals"]["macs"] < now["totals"]["macs"])


def test_a_trace_that_leaves_more_than_one_ask_open_is_refused_until_partial_picks_one(records):
    """A step whose attention backward asks for k's gradient alone: its trace's products are also those of twice the
    calls asking for k's with q's or with v's, so the rewrite names the readings and assumes none. `--partial` picks
    one of them, and then the record is the counter's; an ask the trace doesn't leave open is refused, and so is one
    of more calls than the record's row holds."""
    fused, now = records["k"]
    assert counts(fused) != counts(now)
    with pytest.raises(ValueError, match="trace leaves --partial k:1 or qk:2 or kv:2 open at .*: say which"):
        rewritten(fused)
    got = rewritten(fused, "k:1")
    assert counts(got) == counts(now)
    assert list(got["provenance"]["as_math"]["gradients_asked"].values()) == [{"asked": "k:1", "from": "its trace"}]
    for flag in ("none", "q:1", "k:2"):
        with pytest.raises(ValueError, match=f"trace says --partial k:1 or qk:2 or kv:2 at .* says --partial {flag}"):
            rewritten(fused, flag)
    with pytest.raises(ValueError, match="2 calls a step ask for qk's alone at .* the record's row holds 1"):
        rewritten(fused, "qk:2")
    with pytest.raises(ValueError, match="2 calls a step ask for q's alone at .* the record's row holds 1"):
        rewritten(fused, "q:2", trace=None)


def test_a_fused_row_that_names_a_dtype_is_refused_and_other_rows_keep_theirs(records):
    """A record whose op log names dtypes was written by the counter that writes the kernel's ops, so a fused row
    that names one is no job's: the kernel's results aren't all in the dtype it computes in (its CPU form returns a
    float32 beside a bf16 result), and the rewrite refuses it. A row that isn't a kernel's keeps the dtype it
    names."""
    fused, _ = records["whole"]
    named = copy.deepcopy(fused)
    for st in named["structures"].values():
        st["log"] = [[*row, "bfloat16+float32" if "scaled_dot_product" in row[2] else "float32"] for row in st["log"]]
    with pytest.raises(ValueError, match=r"_scaled_dot_product_flash_attention_for_cpu\w* .*: a fused row that names "
                                         r"a dtype \(bfloat16\+float32\)"):
        rewritten(named)
    rows, _, _ = as_math.rewrite([["other", "param", "mul", "[[4]]", 2, 8, "float32"],
                                  ["other", "param", "mul", "[[4]]", 1, 4, "bfloat16"]], [])
    assert rows == [["other", "param", "mul", "[[4]]", 2, 8, "float32"],
                    ["other", "param", "mul", "[[4]]", 1, 4, "bfloat16"]]


SHAPE = (2, 3, 16, 4)  # b, h, s, d


def trace_of(asks: list[str]) -> list:
    """The backward product rows a trace holds of attention calls, one for each ask (of the letters q, k, v)."""
    rows = {}
    for asked in asks:
        for shapes, n in as_math.backward(asked, *SHAPE)[0].items():
            rows[shapes] = rows.get(shapes, 0) + n
    return [{"op": "bmm", "phase": "backward", "count": n,
             "dims": dict(zip(("batch", "m", "k", "n"), (*a, b[2])))} for shapes, n in rows.items()
            for a, b in [json.loads(shapes)]]


K = [("k", 1), ("qk", 2), ("kv", 2)]


@pytest.mark.parametrize("subset, open_", [("q", [("q", 2)]), ("v", [("v", 2)]), ("qv", [("qv", 2)]),
                                           ("k", [("k", 2), ("qk", 4), ("kv", 4)]), ("qk", K), ("kv", K)])
def test_the_trace_shows_each_subset_the_rewrite_writes(subset, open_):
    """The reading of a trace (`traced`) against the rows the rewrite writes for an ask (`backward`): of five calls,
    two that ask for a subset's gradients alone read as that subset twice. Where k's is asked the products leave more
    than one reading (k's alone in half the calls, or k's with q's or with v's), which `--partial` then settles and
    nothing else does. All three read as `none`, forward rows and other shapes as nothing, and two different subsets
    in one step as no reading."""
    n, s, d = SHAPE[0] * SHAPE[1], *SHAPE[2:]
    rows = trace_of(["qkv"] * 3 + [subset] * 2)
    assert as_math.traced(rows, n, s, d) == open_
    assert as_math.traced(trace_of(["qkv"] * 5), n, s, d) == [as_math.ALL]
    assert as_math.traced([r | {"phase": "forward"} for r in rows], n, s, d) is None
    assert as_math.traced(rows, n + 1, s, d) is None and as_math.traced(rows, n, s, s) is None
    if len(open_) > 1:
        with pytest.raises(ValueError, match=f"leaves --partial {' or '.join(map(as_math.said, open_))} open"):
            as_math.settle(open_, None, "here")
        with pytest.raises(ValueError, match="trace says .* command says --partial v:2"):
            as_math.settle(open_, ("v", 2), "here")
    assert as_math.settle(open_, (subset, 2), "here") == ((subset, 2), "its trace")
    with pytest.raises(ValueError, match="no one subset's"):
        as_math.traced(trace_of(["qkv", "q", "v", "v"]), n, s, d)


@pytest.mark.parametrize("flag", ["qkv:1", "x:1", "q", "q:0", "q:-1", "", "qq:1"])
def test_a_partial_that_is_no_ask_is_refused(flag):
    with pytest.raises(ValueError, match="not `none` nor SUBSET:CALLS"):
        as_math.parse(flag)
    assert as_math.parse(None) is None and as_math.parse("none") == as_math.ALL and as_math.parse("vq:2") == ("qv", 2)


def test_the_command_writes_nothing_where_a_record_is_refused(records, tmp_path, capsys):
    """Through the command: a directory of records, one with a trace and one without. With no `--partial` the one
    without is refused by name and nothing is written; with the ask said, both are written, the traced one's ask
    checked against its trace."""
    fused, now = records["lora"]
    src, dst = tmp_path / "src", tmp_path / "dst"
    for name, rec in (("0/search", fused), ("0/final", fused | {"trace": None})):
        (src / name).mkdir(parents=True)
        (src / name / "run.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="0/final/run.json: refused: .* say --partial none"):
        as_math.main([str(src), str(dst)])
    with pytest.raises(SystemExit, match="0/search/run.json: refused: .*trace says --partial q:1"):
        as_math.main([str(src), str(dst), "--partial", "none"])
    assert not dst.exists()
    assert as_math.main([str(src), str(dst), "--partial", "q:1"]) == 0
    assert sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()) == ["0/final/run.json",
                                                                                      "0/search/run.json"]
    assert all(counts(json.loads(p.read_text())) == counts(json.loads(json.dumps(now))) for p in dst.rglob("run.json"))
    assert '"from": "its trace"' in capsys.readouterr().out
