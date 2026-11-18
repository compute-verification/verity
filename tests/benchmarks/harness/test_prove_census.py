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
    HEAD_END,
    HEAD_END_BITS,
    REPO,
    Store,
    _outside_a_research_run,  # noqa: F401  (autouse)
    designed,
    first,
    on_bits,
    recorded,
    steady,
    without_names,
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
    assert (f"| `{GEMM}` | 1 | 23 | 273,417 | 32 | 12 | 44 | 12,030,348 | `gemm-s8` at 1 prover(s), 24 cores ({rated}) |"
            in out)
    assert f"| `{ATTN}` | 1 | 27 | 37,433,280 | 0 | 16 | 16 | 598,932,480 | none |" in out
    assert "a block may be at most 2^27." in out and "System `spot`: only the refusal's `spot-sampling` slot" in out
    assert "Rated: 1 of 3 block types, 44 of 71 proof blocks." in out
    assert f"| `{EMBED}` | 1 | its rows alone need a 2^39 block | 4,202,739,712 | 8 | 3 | 11 |" in out
    assert f"| `{SELECT}` | 1 | its rows alone need a 2^28 block | 22,603,069 | 4 | 2 | 6 |" in out
    refusal = out.split("## Prover GPUs per serving GPU: refused\n", 1)[1]
    assert refusal.startswith("\nNo total is given while any of these stands:\n\n- 2 block type(s) with blocks and no "
                              "steady-state rate at 1 prover(s) and 10 cores or more (label each run of its rate: "
                              "`research data label TARGET run_of_record prove-llama32-1b-flock-steady --by LANE`), "
                              "each with its proof blocks a served token, a served token being a prompt or a decoded "
                              "token (as GLM-5.3's table counts it), and runs as above:\n"
                              f"  - `{ATTN}`, 1.455 blocks a served token\n  - `{NORM}`, 1 block a served token\n"
                              "- 2 gaps, Definitions with 17 served rows no block holds, by family and reason; each is "
                              "named, with its served rows and why, in the gaps table above and in the census table kept "
                              "as `out/prove-census.json` under a research run (`refusal.gaps`; `--json` prints it):\n"
                              "  - `Embedding_v1`, its rows alone need a 2^39 block: 1 Definition, 11 served rows\n"
                              "  - `TokenSelect_v4`, its rows alone need a 2^28 block: 1 Definition, 6 served rows\n")
    for slot, what in (("cut", "cut"), ("flock-single", "single-session card"), ("spot-sampling", "sampling plan")):
        assert (f"- {slot} (`prove-llama32-1b-{slot}`): no {what} of record: nothing in the store carries "
                f"`run_of_record=prove-llama32-1b-{slot}`") in refusal
    for slot, what in (("lean-verify", "Lean verifier run"), ("flock-warm", "warm-session run")):
        assert (f"- {slot} (`prove-llama32-1b-{slot}`): no {what} of record: no run labelled "
                f"`run_of_record=prove-llama32-1b-{slot}` that is not superseded") in refusal
    for slot in ("cut", "flock-single", "spot-sampling", "lean-verify", "flock-warm"):
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


def census_page(found: dict, refusal: dict | None = None) -> list[str]:
    return PC.render({"census": found | {"art": "art:census", "record": ONE_B.record(R.CENSUS), "run": "r0",
                                         "steady": ONE_B.record(R.STEADY), "provers": 1, "cores": 10},
                      "refusal": refusal})


def test_a_block_types_rate_is_its_own_ids_else_its_proved_as_else_its_bounded_by_and_the_table_says_which():
    """A word block type proved as its Definition on bits takes that one's rate, and the table names it beside the
    block; one whose own Definition on bits has none takes its kind's largest block's, and the table says `bounded by`
    and that the rated count includes it, an upper bound; one whose own id has a rate takes it and prints as it did
    before the census named either. The refusal reads the rates the same way, and with neither field the word blocks
    have no rate, as before."""

    big, small, end = first(190, 192, 2), first(5, 128, 2), P.definition_id(HEAD_END)
    rates = {P.definition_id(big): steady("attn-first", P.definition_id(big), "r20261118-000000-0001", 8, 4.0),
             end: steady("attn-end", end, "r20261118-000000-0002", 8, 1.0),
             P.definition_id(HEAD_END_BITS): steady("attn-end2", P.definition_id(HEAD_END_BITS), "r20261118-000000-0003",
                                                    8, 2.0)}
    doc = on_bits()
    found = PC.table([(None, doc)], rates)
    rate = {r["block"]: r["rate"] for r in found["rows"]}
    assert rate == {
        first(190, 192): {"record": "r20261118-000000-0001", "type": "attn-first", "provers": 1, "cores": 10, "proved_as": big},
        first(5, 128): {"record": "r20261118-000000-0001", "type": "attn-first", "provers": 1, "cores": 10, "bounded_by": big},
        HEAD_END: {"record": "r20261118-000000-0002", "type": "attn-end", "provers": 1, "cores": 10}}
    assert PC.refusal(P.Census.of(doc), found, rates, []) is None
    lines = census_page(found)
    assert (f"| `{first(5, 128)}` proved as `{small}` | 1 | 22 | 100,000 | 30 | 30 | 3,000,000 | `attn-first` at 1 "
            f"prover(s), 10 cores (r20261118-000000-0001), bounded by `{big}` |") in lines
    assert (f"| `{first(190, 192)}` proved as `{big}` | 1 | 26 | 10,000,000 | 2 | 2 | 20,000,000 | `attn-first` at 1 "
            "prover(s), 10 cores (r20261118-000000-0001) |") in lines
    assert (f"| `{HEAD_END}` | 1 | 20 | 1,000 | 4 | 4 | 4,000 | `attn-end` at 1 prover(s), 10 cores "
            "(r20261118-000000-0002) |") in lines
    assert ("Rated: 3 of 3 block types, 36 of 36 proof blocks, 1 block type(s) of them (30 proof blocks) bounded by a "
            "larger block's rate, an upper bound on their cost and not a measurement of it.") in lines

    plain = without_names(doc)
    bare = PC.table([(None, plain)], rates)
    assert {r["block"]: r["rate"] for r in bare["rows"]} == {first(5, 128): None, HEAD_END: rate[HEAD_END],
                                                              first(190, 192): None}
    assert [r["block"] for r in PC.refusal(P.Census.of(plain), bare, rates, [])["no_rate"]] == [first(5, 128), first(190, 192)]
    text = "\n".join(census_page(bare))
    assert "proved as" not in text and "bounded by" not in text and "Rated: 1 of 3 block types, 4 of 36 proof blocks.\n" in text
    assert [x for x in lines if HEAD_END in x] == [x for x in census_page(bare) if HEAD_END in x]


def test_a_block_type_whose_definition_on_bits_has_no_rate_and_no_bound_names_it_and_has_none():
    """With no rate at its own id, its `proved_as` or its `bounded_by`, a block type has no rate, the refusal names it
    by its census name, and the table names its Definition on bits, the one a measurement would rate it by."""

    doc = on_bits()
    found = PC.table([(None, doc)], {})
    assert all(r["rate"] is None for r in found["rows"])
    assert {r["block"] for r in PC.refusal(P.Census.of(doc), found, {}, [])["no_rate"]} == set(doc["definitions"])
    assert (f"| `{HEAD_END}` proved as `{HEAD_END_BITS}` | 1 | 20 | 1,000 | 4 | 4 | 4,000 | none |") in census_page(found)


def test_a_run_bounded_by_one_larger_block_says_so_once_and_names_its_definitions_on_bits_as_a_run():
    """First blocks of 1 to 4 keys, each proved as its own Definition on bits and none rated, take the 190-key one's
    rate: they share it, so they print as one run line that says `bounded by` once and names their Definitions on bits
    in the run's own notation."""

    def block(nvis: int) -> dict:
        return {"k_log": 22, "ands": 1000 * nvis, "of": {"Attention_v4": "split"}, "prefill": 3, "decode": 0, "macs": 0,
                "proved_as": first(nvis, 128, 2), "bounded_by": first(190, 192, 2)}

    big = first(190, 192, 2)
    doc = {"tokens": {"prefill": 3, "decode": 0}, "per_request": {"circuit_macs": 30, "macs": 30},
           "definitions": {first(190, 192): block(190) | {"k_log": 26, "proved_as": big, "bounded_by": None}}
           | {first(n, 128): block(n) for n in range(1, 5)}}
    del doc["definitions"][first(190, 192)]["bounded_by"]
    rates = {P.definition_id(big): steady("attn-first", P.definition_id(big), "r20261118-000000-0001", 8, 4.0)}
    lines = census_page(PC.table([(None, doc)], rates))
    run_of = 'AttnBlockFirst_v{}{{D=64,NVIS=1..4,BN=128,DOT={{"fn":"HopperBF16WgmmaDot16_v{}"}}}}'
    assert (f"| `{run_of.format(1, 1)}` proved as `{run_of.format(3, 2)}` | 4 | 22 | 1,000 to 4,000 | 12 | 12 | 30,000 | "
            f"`attn-first` at 1 prover(s), 10 cores (r20261118-000000-0001), bounded by `{big}` |") in lines
    assert sum("bounded by" in x for x in lines if x.startswith("| `")) == 1


def attention(t: int, masked: int = 0, bn: int = 128) -> str:
    return (f'Attention_v4{{T={t},NH=32,KVH=8,D=64,BN={bn},DOT={{"fn":"HopperBF16WgmmaDot16_v1"}},'
            f'INV={{"fn":"Fa3InvSum_v1"}},MASKED_FROM={masked}}}')


def test_the_text_groups_runs_exactly():
    """A run of one family's Definitions differing in one static, with one reason (or one `k_log` and rate) and served
    rows in the same requests, prints as one line; a hole, another reason, another pattern of requests or another
    static parts runs, a Definition alone keeps its name, and the lines' Definitions and served rows, in each request
    and in all, are the table's exactly."""

    def gap(why: str, ands: int, rows: int) -> dict:
        return {"family": "Attention_v4", "why": why, "ands": ands, "prefill": rows, "decode": 0}

    def block(k_log: int, rows: int) -> dict:
        return {"family": "Attention_v4", "k_log": k_log, "ands": 1000 * k_log, "macs": 0,
                "of": {"Attention_v4": "row"}, "prefill": rows, "decode": 0}

    both, first = {}, {}
    for t in range(22, 49):
        both[attention(t)] = gap(f"a 2^28 block at {1000 * t:,} ANDs", 1000 * t, 15)
    for t in range(49, 52):
        both[attention(t)] = gap(f"a 2^29 block at {1000 * t:,} ANDs", 1000 * t, 15)
    for t in (60, 61, 63):
        both[attention(t)] = gap("its rows alone need a 2^28 block", 1000 * t, 15)
    for t in (52, 53):
        first[attention(t)] = gap(f"a 2^29 block at {1000 * t:,} ANDs", 1000 * t, 15)
    for t in (60, 61):
        both[attention(t, masked=1)] = gap("its rows alone need a 2^28 block", 1000 * t, 15)
    blocks = {attention(t): block(26 if t < 9 else 27, 15) for t in range(1, 12)}
    blocks[attention(12, bn=192)] = block(27, 15)

    def doc(gaps: dict) -> dict:
        return {"tokens": {"prefill": 1, "decode": 0}, "definitions": blocks, "gaps": gaps}

    found = PC.table([("r0", doc(both | first)), ("r1", doc(both))], {})
    gaps, rows = PC.printed(found["gaps"], PC._gap_kind), PC.printed(found["rows"], PC._block_kind)
    for lines, kept in ((gaps, found["gaps"]), (rows, found["rows"])):
        assert sum(u["definitions"] for u in lines) == len(kept)
        assert sum(u["total"] for u in lines) == sum(r["total"] for r in kept)
        for n in ("r0", "r1"):
            assert sum(u["requests"][n] for u in lines) == sum(r["requests"][n] for r in kept)
        assert sorted(r["block"] for u in lines for r in u["of"]) == sorted(r["block"] for r in kept)
    assert {(u["block"], u["definitions"]) for u in gaps} == {
        (attention("22..48"), 27), (attention("49..51"), 3), (attention("52..53"), 2), (attention("60..61"), 2),
        (attention(63), 1), (attention("60..61", masked=1), 2)}
    assert {(u["block"], u["definitions"]) for u in rows} == {
        (attention("1..8"), 8), (attention("9..11"), 3), (attention(12, bn=192), 1)}
    assert next(u for u in gaps if u["definitions"] == 27)["total"] == 27 * 30
    no_rate = [{"block": r["block"], "per_token": r["total"] / 2} for r in found["rows"]]
    assert {u["block"] for _, u in PC.unrated(found["rows"], no_rate)} == {u["block"] for u in rows}


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
