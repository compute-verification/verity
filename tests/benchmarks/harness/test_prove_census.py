"""`verity bench prove` on a workload of several requests (`prove_census`): each request's census summed into the row
as served, the census table with its gaps, the totals refused by name, and GLM-5.3's reading unchanged. The censuses
are made up in the form `block_census.sh --request` writes, one `REQUEST/census.json` a request."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from prove_helpers import (
    FIXTURES,
    REPO,
    Store,
    _outside_a_research_run,  # noqa: F401  (autouse)
    designed,
    recorded,
)
from verity_bench import kit, spot_status
from verity_bench import prove as P
from verity_bench import prove_census as PC
from verity_bench import prove_records as R

RUNNER = REPO / "benchmarks" / "prove" / "run.py"
ONE_B = R.WORKLOADS["llama32-1b"]
GEMM = 'ScaledMmS8Coordinate_v1{K=2048,DOT={"fn":"BlackwellS8ImmaDot32_v1"}}'
NORM = 'RMSNormFusedCuda_v2{N=2048,EPS={"f64":"0x1.4f8b588e368f1p-17"}}'
ATTN = 'Attention_v4{T=12,NH=32,KVH=8,D=64,BN=192,DOT={"fn":"HopperBF16WgmmaDot16_v1"},INV={"fn":"Fa3InvSum_v1"},MASKED_FROM=0}'
EMBED = "Embedding_v1{V=128256,H=2048}"
SELECT = "TokenSelect_v4{V=128256}"
#: GLM-5.3's reading on the prove fixtures' records, as `run.py --system spot --workload glm53` wrote it before a
#: workload could hold several requests: its table and its measurements, each but its host
GLM_READING = FIXTURES / "glm53-reading.json"


def request(prefill: int, decode: int, gemm: int, norm: int, attn: int = 0) -> dict:
    """A made-up request's census: two block types and a third where `attn`, and two gaps, an embedding row and a token
    selection a token."""

    tokens = prefill + decode
    definitions = {
        GEMM: {"family": "ScaledMmS8Coordinate_v1", "k_log": 23, "ands": 273417, "macs": 0,
               "of": {"ScaledMmS8_v1": "split"}, "prefill": gemm * prefill, "decode": gemm * decode},
        NORM: {"family": "RMSNormFusedCuda_v2", "k_log": 26, "ands": 19668368, "macs": 0,
               "of": {"RMSNormFusedCuda_v2": "row"}, "prefill": norm * prefill, "decode": norm * decode}}
    if attn:
        definitions[ATTN] = {"family": "Attention_v4", "k_log": 27, "ands": 37433280, "macs": 0,
                             "of": {"Attention_v4": "row"}, "prefill": attn, "decode": 0}
    return {"tokens": {"prefill": prefill, "decode": decode}, "prompt_len": prefill, "steps": decode + 1,
            "per_request": {"circuit_macs": 2000 * tokens, "macs": 1500 * tokens},
            "definitions": definitions,
            "gaps": {EMBED: {"family": "Embedding_v1", "why": "its rows alone need a 2^39 block", "ands": 4202739712,
                             "prefill": prefill, "decode": decode},
                     SELECT: {"family": "TokenSelect_v4", "why": "its rows alone need a 2^28 block", "ands": 22603069,
                              "prefill": 1, "decode": decode}}}


REQUESTS = {"r0": request(5, 3, gemm=4, norm=1), "r1": request(2, 1, gemm=4, norm=1, attn=16)}


def census_art(s: Store, requests: dict[str, dict], model: str = ONE_B.model) -> str:
    """The censuses in one tree, as `block_census.sh` lays them out, labelled the workload's census of record."""

    root = s._scratch()
    for name, doc in requests.items():
        (root / name).mkdir()
        (root / name / R.CENSUS_FILE).write_text(json.dumps(doc))
        (root / name / "table.md").write_text("made up")
    art = s.art(R.CENSUS_KIND, {"model": model, "run": "r20261117-000000-0000"}, tree=root)
    s.label(art, R.RUN_OF_RECORD, ONE_B.record(R.CENSUS))
    return art


def gemm_rated(s: Store, tmp: Path, run: str = "r20261117-000000-0001") -> str:
    """A run whose proof server rate rates `GEMM` at one prover, labelled the workload's steady-state record."""

    row = {"type": "gemm-s8", "provers": 1, "n": 4096, "warm": 1, "sessions": 12,
           "stage": {"definition": "ScaledMmS8Coordinate_v1{K=2048}", "word": GEMM, "k_log": 23, "blocks": 4096, "full": 4096},
           "exits": {"verifier-gemm-s8": 0, "prover-gpu0": 0}, "accepted": 13, "refused_n": 0,
           "warm_per_prover": {"prover-gpu0": 1}, "unfiled": 0, "gpus": 1, "flat": True, "statements_per_s": 0.25,
           "throttled": {"nr_throttled": 0, "throttled_usec": 0}, "idle_provers": 0, "verify_s": 3.0}
    files = tmp / "run-files" / run
    (files / R.RATES).parent.mkdir(parents=True)
    (files / R.RATES).write_text(json.dumps([row]))
    s.ran(run, files)
    s.label(run, R.RUN_OF_RECORD, ONE_B.record(R.STEADY))
    return run


def run(monkeypatch, capsys, *argv: str, within: Path | None = None) -> tuple[int, str]:
    if within is not None:
        monkeypatch.setenv("RESEARCH_RUN_DIR", str(within))
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(RUNNER), run_name="__main__")
    return done.value.code, capsys.readouterr().out


def test_the_requests_sum_to_the_row_as_served():
    """Each block type's and each gap's rows are summed over the requests, a request that has none of a type counting
    0 of it, and the summed census prices a block type's blocks over every request's tokens."""

    docs = list(REQUESTS.items())
    found = PC.table(docs, {})
    assert found["requests"] == {"r0": {"prefill": 5, "decode": 3}, "r1": {"prefill": 2, "decode": 1}}
    assert found["blocks"] == {"r0": 40, "r1": 31}
    rows = {r["block"]: r for r in found["rows"]}
    assert [r["block"] for r in found["rows"]] == [GEMM, ATTN, NORM]
    assert rows[GEMM]["requests"] == {"r0": 32, "r1": 12} and rows[GEMM]["total"] == 44
    assert rows[GEMM]["gates"] == 44 * 273417 and rows[GEMM]["rate"] is None
    assert rows[ATTN]["requests"] == {"r0": 0, "r1": 16} and rows[NORM]["total"] == 11
    gaps = {g["block"]: g for g in found["gaps"]}
    assert list(gaps) == [EMBED, SELECT]
    assert gaps[EMBED]["requests"] == {"r0": 8, "r1": 3} and gaps[EMBED]["total"] == 11
    assert gaps[SELECT]["requests"] == {"r0": 4, "r1": 2} and gaps[SELECT]["why"] == "its rows alone need a 2^28 block"
    whole = P.Census.of(P.summed([d for _, d in docs]))
    assert {b.name: b.per_token for b in whole.blocks} == pytest.approx({GEMM: 44 / 11, NORM: 11 / 11, ATTN: 16 / 11})
    assert whole.served_macs_per_token == pytest.approx(2000) and whole.macs_in_blocks == pytest.approx(0.75)
    assert REQUESTS["r0"]["definitions"][GEMM]["prefill"] == 20


def test_the_command_prints_the_census_and_refuses_the_ratio_by_name(tmp_path, monkeypatch, capsys):
    """With only the census and a steady rate of one block type labelled, the command prints each block type's blocks
    in each request and summed, which type is rated, each gap as a gap, and refuses the totals naming each type with no
    rate, each gap and each unlabelled slot with its label; it exits 1, and under a research run writes the census
    table as a measurement file and no measurement."""

    s = Store(tmp_path)
    art = census_art(s, REQUESTS)
    rated = gemm_rated(s, tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    code, out = run(monkeypatch, capsys, "--system", "spot", "--workload", "llama32-1b")
    assert code == 1 and out.startswith("# verity bench prove: spot on llama32-1b (Llama-3.2-1B)\n\n## The census of record\n")
    assert (f"{art} (`run_of_record=prove-llama32-1b-census`, run r20261117-000000-0000), 2 requests, summed: the row "
            "as served.") in out
    assert "| r0 | 5 | 3 | 40 |\n| r1 | 2 | 1 | 31 |\n| summed | 7 | 4 | 71 |\n" in out
    assert (f"| `{GEMM}` | 23 | 273,417 | 32 | 12 | 44 | 1.203e+07 | `gemm-s8` at 1 prover(s), 24 cores ({rated}) |"
            in out)
    assert f"| `{ATTN}` | 27 | 37,433,280 | 0 | 16 | 16 | 5.989e+08 | none |" in out
    assert "Rated: 1 of 3 block types, 44 of 71 proof blocks." in out
    assert f"| `{EMBED}` | its rows alone need a 2^39 block | 4,202,739,712 | 8 | 3 | 11 |" in out
    assert f"| `{SELECT}` | its rows alone need a 2^28 block | 22,603,069 | 4 | 2 | 6 |" in out
    refusal = out.split("## Prover GPUs per serving GPU: refused\n", 1)[1]
    assert refusal.startswith("\nNo total is given while any of these stands:\n\n- 2 block type(s) with blocks and no "
                              "steady-state rate at 1 prover(s) and 10 cores or more (label each run of its rate: "
                              "`research data label TARGET run_of_record prove-llama32-1b-flock-steady --by LANE`):\n"
                              f"  - `{ATTN}`, 1.455 blocks a served token\n  - `{NORM}`, 1 blocks a served token\n"
                              "- 2 gap(s), served rows no block holds, each with its rows and why in the gaps above: "
                              f"`{EMBED}`, `{SELECT}`\n")
    for slot, what in (("cut", "cut"), ("flock-single", "single-session card"), ("spot-sampling", "sampling plan"),
                       ("lean-verify", "Lean verifier run"), ("flock-warm", "warm-session run")):
        assert (f"- {slot} (`prove-llama32-1b-{slot}`): no {what} of record: nothing in the store carries "
                f"`run_of_record=prove-llama32-1b-{slot}`") in refusal
        assert f"(label one: `research data label TARGET run_of_record prove-llama32-1b-{slot} --by LANE`)" in refusal
    assert "gpus_per_serving_gpu" not in out

    code, out = run(monkeypatch, capsys, "--system", "spot", "--workload", "llama32-1b", "--json")
    doc = json.loads(out)
    assert code == 1 and "totals" not in doc and [m["block"] for m in doc["refusal"]["no_rate"]] == [ATTN, NORM]
    assert [g["block"] for g in doc["refusal"]["gaps"]] == [EMBED, SELECT]

    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261117-000000-0002")
    monkeypatch.setattr(kit, "host_stamp", lambda: {"card": "none", "gpus": [], "driver": None, "cuda": None,
                                                    "node": "cpu-node-0", "commit": "c" * 40, "why": "no nvidia-smi"})
    within = tmp_path / "run"
    code, printed = run(monkeypatch, capsys, "--system", "spot", "--workload", "llama32-1b", within=within)
    result = json.loads((within / kit.RESULT).read_text())
    assert code == 1 and printed.startswith("# verity bench prove: spot on llama32-1b")
    assert result["measurement_files"] == [PC.CENSUS_TABLE] and result.get("measurements", []) == []
    kept = json.loads((within / PC.CENSUS_TABLE).read_text())
    assert kept["census"]["blocks"] == {"r0": 40, "r1": 31} and kept["refusal"] == doc["refusal"]
    assert not (within / R.TABLE).exists()


def test_a_census_of_another_model_or_with_no_census_file_is_refused(tmp_path):
    s = Store(tmp_path)
    census_art(s, REQUESTS, model="GLM-5.3")
    with pytest.raises(ValueError, match="isn't a block census of Llama-3.2-1B"):
        R.build("spot", "llama32-1b", store=s.fresh(), root=REPO)
    s = Store(tmp_path / "other")
    art = s.tree(R.CENSUS_KIND, "census-rank0.json", REQUESTS["r0"], {"model": ONE_B.model})
    s.label(art, R.RUN_OF_RECORD, ONE_B.record(R.CENSUS))
    with pytest.raises(ValueError, match=r"holds no `census.json` and no `REQUEST/census.json`"):
        R.build("spot", "llama32-1b", store=s.fresh(), root=REPO)


def glm_reading(tmp_path: Path, monkeypatch, capsys) -> dict:
    """GLM-5.3's reading on the prove fixtures' records: the table `run.py --system spot` prints and the measurements
    it writes under a research run, each but its host."""

    s, _ = recorded(tmp_path)
    designed(s, tmp_path)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "reader"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261112-000000-0000")
    monkeypatch.setattr(spot_status, "commit_known", lambda root, commit: False)
    monkeypatch.setattr(spot_status, "LOCK", Path("tests/benchmarks/harness/fixtures/spot_status/lock.json"))
    monkeypatch.setattr(kit, "host_stamp", lambda: {"card": "none", "gpus": [], "driver": None, "cuda": None,
                                                    "node": "cpu-node-0", "commit": "c" * 40, "why": "no nvidia-smi"})
    within = tmp_path / "run"
    code, text = run(monkeypatch, capsys, "--system", "spot", "--workload", "glm53", within=within)
    assert code == 0
    result = json.loads((within / kit.RESULT).read_text())
    return {"text": text, "measurement_files": result["measurement_files"],
            "measurements": [{k: v for k, v in m.items() if k != "host"} for m in result["measurements"]]}


def test_glm53s_reading_is_unchanged(tmp_path, monkeypatch, capsys):
    """GLM-5.3's one census reads as before: its printed table and every measurement it writes are what they were
    before a workload could hold several requests, and none carries a member. Its measurements as a workload with a
    member's carry that member, with the same values and marks."""

    got = glm_reading(tmp_path, monkeypatch, capsys)
    assert got == json.loads(GLM_READING.read_text())
    doc = json.loads((tmp_path / "run" / R.TABLE).read_text())
    metrics = runpy.run_path(str(RUNNER))["METRICS"]
    plain = R.measurements(doc, metrics)
    assert all("member" not in m for m in plain)
    as_member = R.measurements(doc | {"workload": "llama32-1b"}, metrics)
    assert [(m["name"], m["value"], m["mark"], m["member"]) for m in as_member] == [
        (m["name"], m["value"], m["mark"], ONE_B.member) for m in plain]
