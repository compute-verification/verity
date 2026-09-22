"""Configuration-conformance harness (lane CH, `verity_vllm_adapter.conformance`).

CPU-only: port validation (negatives name the offending field), the level rules (evidence -> level, over-claim
refused, never above the evidence), `--from-results` end to end on one small committed bundle, matrix generation.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # vllm-poc/
REPO = ROOT.parent
for p in (ROOT, REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from verity_vllm_adapter.conformance import SCHEMA_LADDER  # noqa: E402
from verity_vllm_adapter.conformance import evidence as E  # noqa: E402
from verity_vllm_adapter.conformance import families as F  # noqa: E402
from verity_vllm_adapter.conformance import ladder as LD  # noqa: E402
from verity_vllm_adapter.conformance import level as LV  # noqa: E402
from verity_vllm_adapter.conformance import run as R  # noqa: E402
from verity_vllm_adapter.conformance import taxonomy as TX  # noqa: E402
from verity_vllm_adapter.conformance.port import PortError, list_ports, load_port, write_bindings  # noqa: E402

PROFILES = ROOT / "profiles"
SMALL_BUNDLE = ROOT / "results" / "B0-d48d41099714-20260907T1826Z"  # W1d's compiled-fx B0 row (3.7 MB): the smallest complete row

needs_small_bundle = pytest.mark.skipif(not SMALL_BUNDLE.is_dir(), reason="committed B0 compiled-fx results not present")


# ---------------------------------------------------------------------------------------------------------
# port validation
# ---------------------------------------------------------------------------------------------------------


def test_every_committed_port_validates():
    names = list_ports()
    assert {"b0-eager", "b1-eager", "b0-compiled", "b1-compiled", "m1-eager", "b1-hopper", "b1-tp2"} <= set(names)
    for name in names:
        port = load_port(name)
        assert port.name == name
        # a port either claims Level 2 (program constructible, every composite bound) or says why not
        if port.level2_claimed:
            assert port.bindings, name
        else:
            assert port.level2_reason, name


def test_m1_port_does_not_claim_level2():
    port = load_port("m1-eager")
    assert not port.level2_claimed
    assert "lane CM" in port.level2_reason
    # MoE block families per W9's moe.md
    fams = port.expectations.families
    assert fams["router"].status == "supported-exact" and "moe.topk.ids.v1" in fams["router"].relations
    assert fams["expert"].status == "same-runtime"


def test_hopper_port_claims_level2_with_hopper_bindings():
    """Lane HP's real port (replaced CH's worked-example placeholder): Hopper hardware class, Level 2 claimed, the GEMM and
    attention families bound to the Hopper-registered functions, everything else TA1's."""
    hp = load_port("b1-hopper")
    assert hp.config.hardware.gpu_generation == "hopper" and hp.level2_claimed and hp.mapping_implemented
    b = hp.bindings
    assert b["GemmHopper"].registered == "GemmHopper@1" and b["AttentionFA3"].registered == "AttentionFA3@1"
    # "everything else TA1's" is the *current* registry: RMSNormFusedCuda@2 (B1Q's generic-kernel model) since the HP merge;
    # HP's recorded 142/142 RMSNorm rows were checked under @1 and re-checked under @2 on the H100 (hopper-profile.md §7)
    assert b["RMSNormFusedCuda"].registered == "RMSNormFusedCuda@2" and b["Embedding"].registered == "Embedding@1"
    assert "Gemm" not in b and "Attention" not in b  # the port's program uses the Hopper-bound names, not TA1's
    raw = hp.static_config.bindings()  # the port's own record names the TA1 definition each Hopper-bound composite re-instantiates
    assert raw["GemmHopper"]["structure_from"] == "Gemm@1" and raw["AttentionFA3"]["structure_from"] == "Attention@1"
    assert "structure_from" not in raw["RMSNormFusedCuda"]
    assert hp.expectations.families["gemm"].status == "supported-exact" and "wgmma" in hp.expectations.families["gemm"].reason
    assert hp.expectations.families["attn"].status == "supported-exact" and "FlashAttention-3" in hp.expectations.families["attn"].reason


def test_tp2_port_claims_a_typed_program_of_shard_and_collective_functions():
    """Lane TP: the TP=2 serving program is TA1's B1 structure with the sharded functions written as compositions of
    the registered functions plus AllReduce2 / AllGather2 / EmbeddingShard (no new IR construct)."""
    tp = load_port("b1-tp2")
    assert tp.config.tensor_parallel_size == 2 and tp.config.capture_per_rank and tp.level2_claimed and tp.mapping_implemented
    b = tp.bindings
    assert {"AllReduce2", "AllGather2", "EmbeddingShard", "Gemm", "Attention", "RMSNormFusedCuda", "TokenSelect"} <= set(b)
    assert b["AllReduce2"].family == "all_reduce" and b["AllGather2"].family == "all_gather" and b["EmbeddingShard"].family == "embedding"
    assert b["Gemm"].registered == "Gemm@1"  # TA1's Gemm, unchanged, called per rank on a static view of the weight
    prog = tp.program(LP=2, STEPS=1)
    assert prog.fn.id.startswith("ServeTP2_v1{")
    # the TP=2 program has the same named root calls as B1's (embed / pre / attn / post / final per position)
    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    p1 = Program(bind(b1.Serve, C=dict(b1.B1_CONFIG), LP=2, STEPS=1))
    names2 = {c.node.name for c in prog.circuit.root_calls() if c.node.name}
    names1 = {c.node.name for c in p1.circuit.root_calls() if c.node.name}
    assert names2 == names1 and prog.gates > p1.gates and prog.input_gates == p1.input_gates
    # the mapping routes a per-rank identity to the rank's member / call and a replicated one to the shared call
    cm = tp.circuit_map(prog)
    pid = "p=7f26cdcedcd0/r=R/t=0/ph=prefill"
    call_r0, gates_r0 = cm.locate(f"{pid}/l=0/op=layers.0.self_attn.o_proj.gemm/port=0/s=0")
    call_r1, gates_r1 = cm.locate(f"{pid}/l=0/op=layers.0.self_attn.o_proj.gemm/port=0/s=1")
    assert call_r0.name == call_r1.name == "Gemm" and call_r0.member == 0 and call_r1.member == 1 and set(gates_r0).isdisjoint(gates_r1)
    ar0 = cm.locate_kind(f"{pid}/l=0/op=layers.0.self_attn.o_proj.all_reduce/port=0/s=0")
    ar1 = cm.locate_kind(f"{pid}/l=0/op=layers.0.self_attn.o_proj.all_reduce/port=0/s=1")
    assert ar0[0].name == "AllReduce2" and ar0[0] is ar1[0] and ar0[1] == ar1[1] and ar0[2] == ar1[2] == "replicated"
    flags = R.shell_flags(tp)
    assert "CONF_TP=2" in flags and "CONF_PER_RANK=1" in flags and "--tensor-parallel-size 2" in flags


def _copy_port(tmp_path: Path, name: str) -> Path:
    dst = tmp_path / name
    shutil.copytree(PROFILES / name, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _edit_json(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=1))


def test_missing_field_is_named(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")

    def drop(doc):
        del doc["hardware_class"]

    _edit_json(d / "vllm_config.json", drop)
    with pytest.raises(PortError, match="hardware_class"):
        load_port(d)


def test_unknown_execution_mode_is_refused(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")

    def bad(doc):
        doc["execution_mode"] = "jit"

    _edit_json(d / "vllm_config.json", bad)
    with pytest.raises(PortError, match="execution_mode"):
        load_port(d)


def test_expectations_naming_unknown_family_refused(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")

    def bad(doc):
        doc["families"]["layernorm"] = {"status": "supported-exact", "reason": "typo for rmsnorm", "relations": []}

    _edit_json(d / "expectations.json", bad)
    with pytest.raises(PortError, match="unknown family 'layernorm'"):
        load_port(d)


def test_expectations_unknown_status_refused(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")

    def bad(doc):
        doc["families"]["gemm"]["status"] = "probably-fine"

    _edit_json(d / "expectations.json", bad)
    with pytest.raises(PortError, match="status"):
        load_port(d)


def test_binding_to_unregistered_version_refused(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")
    sc = d / "static_config.py"
    text = sc.read_text()
    assert 'f"{comp.name}@{comp.version}"' in text
    sc.write_text(text.replace('f"{comp.name}@{comp.version}"', 'f"{comp.name}@{comp.version + 7}"'))
    with pytest.raises(PortError, match="unregistered version"):
        load_port(d)


def test_binding_missing_composite_refused(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")
    sc = d / "static_config.py"
    text = sc.read_text()
    sc.write_text(text.replace("    return out\n", "    out.pop('Gemm')\n    return out\n"))
    with pytest.raises(PortError, match="Gemm"):
        load_port(d)


def test_stale_bindings_json_refused(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")
    gen = d / "bindings.json"
    if not gen.exists():
        write_bindings(load_port(d))
    doc = json.loads(gen.read_text())
    doc["bindings"]["Gemm"]["status"] = "not-yet-established"
    gen.write_text(json.dumps(doc))
    with pytest.raises(PortError, match="stale"):
        load_port(d)


def test_static_config_without_program_is_refused(tmp_path):
    d = _copy_port(tmp_path, "b1-compiled")
    (d / "static_config.py").write_text("CONFIG = {}\n")
    with pytest.raises(PortError, match="program"):
        load_port(d)


def test_known_families_match_adapter_when_torch_importable():
    try:
        cap = importlib.import_module("verity_vllm_adapter.capture")
    except Exception:  # noqa: BLE001 — torch absent on the CPU box
        pytest.skip("torch not importable")
    adapter = {fam for fam, _ in cap.OP_FAMILY.values()} if isinstance(next(iter(cap.OP_FAMILY.values())), tuple) else set(cap.OP_FAMILY.values())
    assert adapter <= set(F.KNOWN_FAMILIES)


# ---------------------------------------------------------------------------------------------------------
# level rules
# ---------------------------------------------------------------------------------------------------------


def _l1(prod_status: str, replay_status: str, *, problems=(), missing=()):
    production = {"status": prod_status, "method": "two processes", "productions": [], "pairs": []}
    replay = {"status": replay_status, "replays": [{"commit": "x/commit", "profile_id": "abc", "status": replay_status,
                                                    "problems": list(problems), "missing": list(missing)}],
              "rows": {replay_status: 1}}
    return LV.level1(production, replay)


def test_level1_needs_both_halves():
    assert _l1("reproducible", "reproducible").met
    assert not _l1("reproducible", "not-reproducible", problems=["compare.json: commit capture differs from the references"]).met
    assert not _l1("not-reproducible", "reproducible").met
    r = _l1("absent", "reproducible")
    assert not r.met and "commit store" in r.reason
    r = _l1("reproducible", "incomplete", missing=["compare.json missing"])
    assert not r.met and "compare.json missing" in r.reason


def test_level2_not_claimed_and_mismatch_rules():
    assert LV.level2({"status": "absent", "reason": "none", "files": [], "summary": {}, "records": [], "mismatches": [],
                      "expectations": {"consistent": True}}, False).status == "not-claimed"
    ok = {"status": "established", "reason": None, "files": [{"path": "c.json", "summary": {"exact": 3, "mismatch": 0}}],
          "summary": {"exact": 3, "mismatch": 0, "sampled": 0}, "records": 3, "mismatches": [], "missing": {"derived_inputs": []},
          "expectations": {"consistent": True, "problems": []}}
    assert LV.level2(ok, True).met
    # a mismatch is allowed only when recorded and the composite's binding status carries the divergence
    rec = dict(ok, summary={"exact": 2, "mismatch": 1, "sampled": 0},
               mismatches=[{"canonical_id": "p=a/r=b/t=1/ph=prefill/l=0/op=x.rmsnorm/port=0", "composite": "RMSNormFusedCuda_v1", "status": "mismatch", "words": [],
                            "binding": {"registered": "RMSNormFusedCuda@1", "status": "exact-model-tested", "divergence_recorded": "1 word"},
                            "recorded_in_binding": True}])
    assert LV.level2(rec, True).met
    unrec = dict(rec, mismatches=[dict(rec["mismatches"][0], recorded_in_binding=False, binding={"registered": "RMSNormFusedCuda@1", "status": "exact-model-tested"})],
                 status="not-established", reason="a mismatch is not recorded on its composite's binding (status not downgraded)")
    r = LV.level2(unrec, True)
    assert not r.met and r.status == "not-met" and "not recorded" in r.reason


def test_verdict_headline_is_highest_met_and_gaps_named():
    res = [LV.LevelResult(1, True, "met", "ok", []), LV.LevelResult(2, False, "not-attempted", "none", []),
           LV.LevelResult(3, True, "met", "ok", []), LV.LevelResult(4, False, "not-attempted", "none", [])]
    v = LV.verdict("p", res, utc="x")
    assert v["level"] == 3 and v["met"] == [1, 3] and v["gaps"] == [2]
    assert v["summary"] == "L3 (gaps: L2)"
    assert v["levels"]["2"]["status"] == "not-attempted"
    LV.assert_claim(v, 3)
    LV.assert_claim(v, 1)
    with pytest.raises(LV.OverClaim):
        LV.assert_claim(v, 2)
    with pytest.raises(LV.OverClaim):
        LV.assert_claim(v, 4)


def test_verdict_with_no_evidence_is_level_zero():
    res = [LV.LevelResult(n, False, "absent", "nothing", []) for n in (1, 2, 3, 4)]
    v = LV.verdict("p", res, utc="x")
    assert v["level"] == 0 and v["summary"] == "none"
    with pytest.raises(LV.OverClaim):
        LV.assert_claim(v, 1)


# ---------------------------------------------------------------------------------------------------------
# --from-results end to end (one small committed bundle) and the matrix
# ---------------------------------------------------------------------------------------------------------


@needs_small_bundle
def test_from_results_end_to_end_small_bundle(tmp_path):
    port = load_port("b0-compiled-fx")
    out = tmp_path / "CONF-b0-compiled-fx-test"
    level = R.run_port(port, out, from_results=[str(SMALL_BUNDLE)], correspondence=[], obligations=[], proof_records=[])
    # the nine standard outputs (level.json kept, ladder.json beside it) + CONFORMANCE.md + run.json
    for name in ("production_repro.json", "replay_repro.json", "program.json", "correspondence.json", "missing_values.json",
                 "mismatches.jsonl", "descriptor_stats.json", "level.json", "ladder.json", "CONFORMANCE.md", "run.json"):
        assert (out / name).exists(), name
    assert level["level"] == 3 and level["met"] == [1, 3] and level["gaps"] == [2]
    assert level["levels"]["2"]["status"] == "not-claimed"
    assert level["levels"]["4"]["status"] == "not-attempted"
    doc = json.loads((out / "level.json").read_text())
    assert doc["port"] == "b0-compiled-fx"
    assert doc["inputs"]["results"][0]["profile_id"] == "d48d41099714"
    rr = json.loads((out / "replay_repro.json").read_text())
    assert rr["status"] == "reproducible" and rr["replays"][0]["replay_exec"] == "compiled-fx"
    pr = json.loads((out / "program.json").read_text())
    assert pr["status"] == "not-claimed"
    # the run refuses a Level 2 claim
    with pytest.raises(LV.OverClaim):
        LV.assert_claim(level, 2)


@needs_small_bundle
def test_results_dir_of_another_port_is_rejected(tmp_path):
    port = load_port("b1-eager")  # the small bundle is a B0 row: its manifest violates the port
    rd = E.discover(SMALL_BUNDLE, port)
    assert not rd.matches_port and rd.manifest_violations
    out = tmp_path / "CONF-b1-eager-test"
    level = R.run_port(port, out, from_results=[str(SMALL_BUNDLE)], correspondence=[], obligations=[], proof_records=[])
    assert level["level"] == 0 and level["levels"]["1"]["status"] == "absent"


@needs_small_bundle
def test_per_rank_pod_row_yields_one_results_dir_per_rank(tmp_path):
    """A pod row replayed per rank (`run_case.sh --rank-tag r`): `commit/rank<r>/`, `ref1/rank<r>/`,
    `compare.rank<r>.json`, `verify-security/rank<r>/` — built here from the small bundle with symlinks."""
    import os

    row = tmp_path / "B0" / "tp-test"
    row.mkdir(parents=True)
    os.symlink(SMALL_BUNDLE / "production", row / "production")
    for r in (0, 1):
        for mode in ("commit", "ref1", "ref2"):
            (row / mode).mkdir(exist_ok=True)
            os.symlink(SMALL_BUNDLE / ("bundle" if mode == "commit" else mode), row / mode / f"rank{r}")
        shutil.copy(SMALL_BUNDLE / "compare.json", row / f"compare.rank{r}.json")
        vd = row / "verify-security" / f"rank{r}"
        vd.mkdir(parents=True)
        for name in ("verify.json", "table1.jsonl", "table2.jsonl"):
            shutil.copy(SMALL_BUNDLE / name, vd / name)
    port = load_port("b0-compiled-fx")
    dirs = E.discover_all(row, port)
    assert [d.rank for d in dirs] == [0, 1]
    assert all(d.layout == "pod" and d.matches_port and d.compare is not None and d.verify is not None for d in dirs)
    assert dirs[1].commit == row / "commit" / "rank1" and dirs[1].references == [row / "ref1" / "rank1", row / "ref2" / "rank1"]
    rr = E.replay_repro(dirs)
    assert rr["status"] == "reproducible" and len(rr["replays"]) == 2
    level = R.run_port(port, tmp_path / "out", from_results=[str(row)], correspondence=[], obligations=[], proof_records=[])
    assert level["level"] == 3 and [d["rank"] for d in level["inputs"]["results"]] == [0, 1]


def test_matrix_generation(tmp_path):
    results = tmp_path / "results"
    for port, headline, met in (("b0-eager", 3, [1, 3]), ("m1-eager", 1, [1])):
        d = results / f"CONF-{port}-20260908T0000Z"
        d.mkdir(parents=True)
        levels = {}
        for n in (1, 2, 3, 4):
            levels[str(n)] = {"level": n, "name": LV.LEVEL_NAMES[n], "met": n in met, "status": "met" if n in met else "not-attempted",
                              "reason": "test", "evidence": [{"kind": "production", "path": f"r/{port}/production"}] if n == 1 else []}
        gaps = [g for g in range(1, headline) if g not in met]
        summary = f"L{headline}" + (f" (gaps: {', '.join(f'L{g}' for g in gaps)})" if gaps else "")
        (d / "level.json").write_text(json.dumps({"port": port, "utc": "20260908T0000Z", "level": headline, "met": met,
                                                  "gaps": gaps, "summary": summary, "levels": levels, "claims": []}))
    # level.json-only runs (HP/TP's pre-ladder harness) are lifted to the ladder: C1–C4 mapped, C0/C5 not-attempted
    rows = R.collect_levels(results)
    text = R.render_matrix(rows, ["b0-eager", "m1-eager", "b1-hopper"])
    assert "| `b0-eager` — **C3 (gaps: C0, C2)** |" in text  # C0 is a named gap: the lifted run never evaluated it
    assert "| `m1-eager` — **C1 (gaps: C0)** |" in text
    assert "| `b1-hopper` | hardware semantics" in text and "| not run | not run | not run | not run | not run | not run | — | `NOT_ATTEMPTED` | — |" in text
    assert "`r/b0-eager/production`" in text
    assert "not-attempted: run predates the ladder" in text
    assert "| Profile | axis | C0 runs | C1 replay | C2 representation | C3 protocol | C4 proof | C5 vertical slice | Main unsupported reason | Blocker (next rung) | run |" in text
    assert "| C4: `NOT_ATTEMPTED` — " in text  # b0-eager lifted: C3 reached, C4 not-attempted is the next rung
    # the CLI lists every registered port (unrun ports as "not run") and --check compares the written file
    out = tmp_path / "matrix.md"
    assert R.main(["matrix", "--results", str(results), "--out", str(out)]) == 0
    assert out.read_text() == R.render_matrix(R.collect_ladders(results), list_ports())
    assert "| `b1-tp2` | distribution:" in out.read_text()
    assert R.main(["matrix", "--results", str(results), "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text() + "edited\n")
    assert R.main(["matrix", "--results", str(results), "--out", str(out), "--check"]) == 1


def test_render_shows_conformance_runs(tmp_path):
    from verity_vllm import render as VR

    results = tmp_path / "results"
    d = results / "CONF-b0-eager-20260908T0000Z"
    d.mkdir(parents=True)
    levels = {str(n): {"status": "met" if n in (1, 3) else "not-attempted", "met": n in (1, 3)} for n in (1, 2, 3, 4)}
    (d / "level.json").write_text(json.dumps({"port": "b0-eager", "utc": "20260908T0000Z", "level": 3, "levels": levels}))
    collected = VR.collect(results)
    assert collected.issues == [] and len(collected.conformance) == 1
    text = VR.render_markdown(collected, "x")
    assert "## Configuration conformance (profile ports)" in text
    assert "| b0-eager | L3 | met | not-attempted | met | not-attempted |" in text
    # an over-claiming level.json is an issue, never rendered as a higher level
    (d / "level.json").write_text(json.dumps({"port": "b0-eager", "utc": "20260908T0000Z", "level": 4, "levels": levels}))
    collected = VR.collect(results)
    assert any("headline level 4 is not the highest met level (3)" in i for i in collected.issues)


# ---------------------------------------------------------------------------------------------------------
# the promotion ladder C0–C5 (lane CH2): both spellings, ladder.json, C0/C5 contracts, the closed taxonomy
# ---------------------------------------------------------------------------------------------------------


def test_rung_spellings_both_accepted_output_is_c():
    for spelled, want in (("C0", "C0"), ("c2", "C2"), ("L1", "C1"), ("l4", "C4"), ("Level 3", "C3"), (2, "C2"), ("2", "C2"), ("C5", "C5")):
        assert LD.normalize_rung(spelled) == want, spelled
    for bad in ("L0", "L5", "C6", 0, 5, "x", "", True):
        with pytest.raises(ValueError):
            LD.normalize_rung(bad)
    assert [LD.RUNG_OF_LEVEL[n] for n in (1, 2, 3, 4)] == ["C1", "C2", "C3", "C4"]
    assert LD.RUNGS == ("C0", "C1", "C2", "C3", "C4", "C5") and set(LD.RUNG_NAMES) == set(LD.RUNGS)


def _ladder_doc(c0="not-attempted", c5="not-attempted", levels=((1, True), (2, False), (3, True), (4, False))):
    res = [LV.LevelResult(n, met, "met" if met else "not-attempted", "t", []) for n, met in levels]
    level = LV.verdict("p", res, utc="x")
    return LD.ladder("p", level, LD.RungResult("C0", c0 == "met", c0, "t", []), LD.RungResult("C5", c5 == "met", c5, "t", []),
                     "unclassified", None, utc="x", run_dir=None, axis="q")


def test_ladder_headline_is_highest_rung_met_and_gaps_named():
    d = _ladder_doc(c0="met")
    assert d["schema"] == SCHEMA_LADDER and d["reached"] == "C3" and d["met"] == ["C0", "C1", "C3"] and d["gaps"] == ["C2"]
    assert d["summary"] == "C3 (gaps: C2)" and list(d["rungs"]) == list(LD.RUNGS)
    assert all(d["rungs"][r]["status"] in LD.STATUSES for r in LD.RUNGS)
    assert d["rungs"]["C2"]["level"] == 2 and d["rungs"]["C0"].get("level") is None and d["rung_of_level"] == {"L1": "C1", "L2": "C2", "L3": "C3", "L4": "C4"}
    # a C0 gap under a C3 headline is named, never implied
    d2 = _ladder_doc(c0="not-met")
    assert d2["reached"] == "C3" and d2["gaps"] == ["C0", "C2"]
    # C5 alone met with nothing below: reached C5, every lower rung a gap (never implied)
    d3 = _ladder_doc(c5="met", levels=((1, False), (2, False), (3, False), (4, False)))
    assert d3["reached"] == "C5" and d3["gaps"] == ["C0", "C1", "C2", "C3", "C4"]
    none = _ladder_doc(levels=((1, False), (2, False), (3, False), (4, False)))
    assert none["reached"] is None and none["summary"] == "none" and none["met"] == []
    # over-claims refused in either spelling
    LD.assert_claim(d, "C3")
    LD.assert_claim(d, "L3")
    LD.assert_claim(d, 1)
    for claim in ("C2", "L2", "C4", "C5", "L4"):
        with pytest.raises(LV.OverClaim):
            LD.assert_claim(d, claim)
    with pytest.raises(ValueError):
        LD.assert_claim(d, "C9")


def test_ladder_from_level_json_only_run():
    """HP/TP's runs write level.json only: the matrix lifts them (C1–C4 mapped, C0/C5 not-attempted with the reason)."""
    levels = {str(n): {"level": n, "name": LV.LEVEL_NAMES[n], "met": n in (1, 3), "status": "met" if n in (1, 3) else ("not-claimed" if n == 2 else "not-attempted"),
                       "reason": "unsupported:primitive-semantics-missing FA3 wgmma m64n8k16" if n == 2 else "t", "evidence": []} for n in (1, 2, 3, 4)}
    doc = LD.ladder_from_level({"port": "b1-hopper", "utc": "x", "level": 3, "summary": "L3 (gaps: L2)", "levels": levels}, axis="hw")
    assert doc["reached"] == "C3" and doc["gaps"] == ["C0", "C2"] and doc["derived_from"] == "level.json" and doc["axis"] == "hw"
    assert doc["rungs"]["C0"]["status"] == "not-attempted" and "predates the ladder" in doc["rungs"]["C0"]["reason"]
    assert doc["rungs"]["C5"]["status"] == "not-attempted" and "int-e2e-regression" in doc["rungs"]["C5"]["reason"]
    assert doc["rungs"]["C1"]["met"] and doc["rungs"]["C2"]["status"] == "not-claimed"
    # a taxonomy-prefixed level-2 reason is read as the unsupported outcome
    assert doc["outcome"] == TX.PRIMITIVE_SEMANTICS_MISSING and doc["unsupported"]["detail"] == "FA3 wgmma m64n8k16" and doc["unsupported"]["rung"] == "C2"
    # a free-text reason stays unclassified (pending work, not an unsupported outcome)
    levels["2"]["reason"] = "typed M1 program: lane CM"
    doc = LD.ladder_from_level({"port": "m1-eager", "utc": "x", "level": 3, "levels": levels})
    assert doc["outcome"] == "unclassified" and doc["unsupported"] is None


def test_taxonomy_enum_is_closed_and_parse_prefix():
    assert TX.OUTCOMES == ("supported-exactly", "unsupported:primitive-semantics-missing", "unsupported:compact-representation-missing",
                           "unsupported:replay-cannot-obtain-canonical-value", "unsupported:commitment-layout-cannot-authenticate")
    assert TX.parse("unsupported:primitive-semantics-missing FA3 wgmma m64n8k16") == (TX.PRIMITIVE_SEMANTICS_MISSING, "FA3 wgmma m64n8k16")
    assert TX.parse("unsupported:compact-representation-missing: the all-reduce partial") == (TX.COMPACT_REPRESENTATION_MISSING, "the all-reduce partial")
    assert TX.parse("typed M1 program: lane CM") is None and TX.parse(None) is None and TX.parse("supported-exactly x") is None
    with pytest.raises(TX.TaxonomyError):
        TX.check_unsupported_reason("unsupported:kernel-is-weird", "here")
    with pytest.raises(TX.TaxonomyError):
        TX.check_unsupported_reason("supported-exactly", "here")  # not an *unsupported* reason
    with pytest.raises(TX.TaxonomyError):
        TX.unsupported_record("C2", "opaque", "d", "s")
    # the correspondence mapping
    assert TX.classify_correspondence_record({"check": {"status": "exact"}, "value_identity": {}, "authentication_location": {"leaf": 1}}) == TX.SUPPORTED_EXACTLY
    assert TX.classify_correspondence_record({"check": {"status": "missing-operand"}}) == TX.REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE
    assert TX.classify_correspondence_record({"check": {"status": "shape-mismatch"}}) == TX.COMPACT_REPRESENTATION_MISSING
    assert TX.classify_correspondence_record({"check": {"status": "exact"}, "value_identity": {}, "authentication_location": None}) == TX.COMMITMENT_LAYOUT_CANNOT_AUTHENTICATE
    assert TX.classify_correspondence_record({"check": {"status": "mismatch"}}) is None


def test_expectations_unsupported_requires_taxonomy_reason(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")

    def no_reason(doc):
        doc["families"]["attn"] = {"status": "unsupported", "kind": "opaque", "reason": "FA3 not registered", "relations": []}

    _edit_json(d / "expectations.json", no_reason)
    with pytest.raises(PortError, match="unsupported_reason"):
        load_port(d)

    def bad_reason(doc):
        doc["families"]["attn"]["unsupported_reason"] = "unsupported:because"

    _edit_json(d / "expectations.json", bad_reason)
    with pytest.raises(PortError, match="unsupported:primitive-semantics-missing"):
        load_port(d)

    def good(doc):
        doc["families"]["attn"]["unsupported_reason"] = "unsupported:primitive-semantics-missing"

    _edit_json(d / "expectations.json", good)
    port = load_port(d)
    assert port.expectations.families["attn"].unsupported_reason == TX.PRIMITIVE_SEMANTICS_MISSING
    outcome, un = LD.unsupported_outcome(port, None, {})
    assert outcome == TX.PRIMITIVE_SEMANTICS_MISSING and un["source"] == "expectations.json families.attn" and un["rung"] == "C2"

    def on_supported(doc):
        doc["families"]["gemm"]["unsupported_reason"] = "unsupported:primitive-semantics-missing"

    _edit_json(d / "expectations.json", on_supported)
    with pytest.raises(PortError, match="only for status 'unsupported'"):
        load_port(d)


def test_expectations_unsupported_functions_fail_closed(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")

    def bad(doc):
        doc["unsupported_functions"] = [{"function": "FA3 wgmma m64n8k16", "reason": "not-registered", "detail": "x"}]

    _edit_json(d / "expectations.json", bad)
    with pytest.raises(PortError, match=r"unsupported_functions\[0\].*reason"):
        load_port(d)

    def missing_detail(doc):
        doc["unsupported_functions"] = [{"function": "FA3", "reason": "unsupported:primitive-semantics-missing"}]

    _edit_json(d / "expectations.json", missing_detail)
    with pytest.raises(PortError, match="detail"):
        load_port(d)

    def bad_rung(doc):
        doc["unsupported_functions"] = [{"function": "FA3", "reason": "unsupported:primitive-semantics-missing", "detail": "x", "rung": "C9"}]

    _edit_json(d / "expectations.json", bad_rung)
    with pytest.raises(PortError, match="rung"):
        load_port(d)

    def good(doc):
        doc["unsupported_functions"] = [{"function": "FA3 wgmma m64n8k16", "reason": "unsupported:primitive-semantics-missing", "detail": "sm90 attention kernel", "rung": "L2"}]

    _edit_json(d / "expectations.json", good)
    port = load_port(d)
    assert port.expectations.unsupported_functions[0]["rung"] == "C2"  # L2 accepted on input, C2 on output
    outcome, un = LD.unsupported_outcome(port, None, {})
    assert outcome == TX.PRIMITIVE_SEMANTICS_MISSING and un["detail"].startswith("FA3 wgmma m64n8k16: ") and un["classified"]


def test_static_config_not_implemented_taxonomy_prefix(tmp_path):
    d = _copy_port(tmp_path, "m1-eager")
    sc = d / "static_config.py"
    sc.write_text(sc.read_text().replace('"typed M1 program: lane CM"', '"unsupported:primitive-semantics-missing fused_moe_kernel per-slot expert output"'))
    port = load_port(d)
    assert not port.level2_claimed and port.level2_taxonomy == (TX.PRIMITIVE_SEMANTICS_MISSING, "fused_moe_kernel per-slot expert output")
    outcome, un = LD.unsupported_outcome(port, None, {"C2": LD.RungResult("C2", False, "not-claimed", port.level2_reason, [])})
    assert outcome == TX.PRIMITIVE_SEMANTICS_MISSING and un["rung"] == "C2" and "NotImplementedError" in un["source"]
    # the committed m1 port's free-text reason is pending work: unclassified, no unsupported record
    m1 = load_port("m1-eager")
    assert m1.level2_taxonomy is None
    assert LD.unsupported_outcome(m1, None, {"C2": LD.RungResult("C2", False, "not-claimed", m1.level2_reason, [])}) == ("unclassified", None)


def test_every_committed_port_declares_its_axis():
    for name in list_ports():
        port = load_port(name)
        assert port.config.axis and ":" in port.config.axis, name  # "<axis>: <what the profile tests>"
        assert port.as_json()["axis"] == port.config.axis
    assert load_port("b1-hopper").config.axis.startswith("hardware semantics")
    assert load_port("b1-tp2").config.axis.startswith("distribution")
    assert load_port("m1-eager").config.axis.startswith("dynamic routing")
    for name in ("b0-eager", "b0-compiled", "b0-compiled-fx", "b1-eager", "b1-compiled", "b1-a100-eager", "b0-a100-eager"):
        assert load_port(name).config.axis.startswith("execution mode"), name


def test_port_without_axis_is_refused(tmp_path):
    d = _copy_port(tmp_path, "b0-eager")

    def drop(doc):
        del doc["axis"]

    _edit_json(d / "vllm_config.json", drop)
    with pytest.raises(PortError, match="axis"):
        load_port(d)

    def empty(doc):
        doc["axis"] = "  "

    _edit_json(d / "vllm_config.json", empty)
    with pytest.raises(PortError, match="axis"):
        load_port(d)


def test_compiled_ports_declare_primitive_semantics_missing():
    for name in ("b0-compiled", "b0-compiled-fx", "b1-compiled"):
        port = load_port(name)
        assert not port.level2_claimed
        outcome, un = LD.unsupported_outcome(port, None, {"C2": LD.RungResult("C2", False, "not-claimed", port.level2_reason, [])})
        assert outcome == TX.PRIMITIVE_SEMANTICS_MISSING and "triton_per_fused_fused_add_rms_norm" in un["detail"], name


@needs_small_bundle
def test_ladder_json_written_with_c0_from_production_manifest(tmp_path):
    port = load_port("b0-compiled-fx")
    out = tmp_path / "CONF-b0-compiled-fx-test"
    level = R.run_port(port, out, from_results=[str(SMALL_BUNDLE)], correspondence=[], obligations=[], proof_records=[])
    doc = json.loads((out / "ladder.json").read_text())
    assert doc == level["ladder"] and doc["schema"] == SCHEMA_LADDER and doc["port"] == "b0-compiled-fx"
    assert doc["reached"] == "C3" and doc["met"] == ["C0", "C1", "C3"] and doc["gaps"] == ["C2"] and doc["summary"] == "C3 (gaps: C2)"
    assert doc["axis"] == port.config.axis
    # level.json is unchanged in shape and agrees
    lv = json.loads((out / "level.json").read_text())
    assert lv["level"] == 3 and lv["summary"] == "L3 (gaps: L2)" and "ladder" not in lv and doc["level_json"] == {"level": 3, "summary": "L3 (gaps: L2)"}
    # C0: the production manifest's execution choices, every required field present; the compiled kernel identity from the bundle
    c0 = doc["rungs"]["C0"]
    assert c0["met"] and c0["status"] == "met"
    man = [e for e in c0["evidence"] if e["kind"] == "production-manifest"]
    assert len(man) == 1 and man[0]["missing"] is None and man[0]["path"].endswith("production/manifest.json")
    xc = man[0]["execution_choices"]
    assert set(p for p, _ in LD.C0_REQUIRED_FIELDS) <= set(xc) and xc["compilation.enforce_eager"] is False and xc["dtype"] == "bfloat16"
    assert xc["backends.attention"].endswith("FlashAttentionImpl") and xc["vllm.wheel_sha256"] and xc["model.weights_sha256"]
    kid = [e for e in c0["evidence"] if e["kind"] == "kernel-record"]
    assert kid and kid[0]["required"] is True and kid[0]["identifies_production_kernels"] and kid[0]["kernels_identical"] is True
    # C5: not attempted, with the INT pointer
    assert doc["rungs"]["C5"]["status"] == "not-attempted" and "int-e2e-regression" in doc["rungs"]["C5"]["reason"]
    # the taxonomy outcome comes from the port's declaration
    assert doc["outcome"] == TX.PRIMITIVE_SEMANTICS_MISSING and doc["unsupported"]["rung"] == "C2" and doc["unsupported"]["classified"]
    # over-claims refused in either spelling
    LD.assert_claim(doc, "C3")
    LD.assert_claim(doc, "L1")
    for claim in ("C2", "L2", "C4", "C5"):
        with pytest.raises(LV.OverClaim):
            LD.assert_claim(doc, claim)
    # CONFORMANCE.md carries the ladder
    md = (out / "CONFORMANCE.md").read_text()
    assert "C3 (gaps: C2)" in md and "## C0 — runs" in md and "## C5 — vertical slice" in md and "unsupported:primitive-semantics-missing" in md
    # the matrix cites the production manifest for C0
    text = R.render_matrix(R.collect_ladders(tmp_path), ["b0-compiled-fx"])
    assert "| `b0-compiled-fx` — **C3 (gaps: C2)** |" in text and "production/manifest.json`" in text
    assert "`unsupported:primitive-semantics-missing` — triton_per_fused_fused_add_rms_norm" in text


@needs_small_bundle
def test_c0_not_met_when_execution_choice_missing(tmp_path):
    """A production manifest without a recorded backend sinks C0 (C1 still judges the outputs)."""
    row = tmp_path / "row"
    shutil.copytree(SMALL_BUNDLE, row, symlinks=True)
    mf = row / "production" / "manifest.json"
    doc = json.loads(mf.read_text())
    del doc["backends"]["attention"]
    del doc["gpu"]["driver"]
    mf.write_text(json.dumps(doc))
    port = load_port("b0-compiled-fx")
    level = R.run_port(port, tmp_path / "out", from_results=[str(row)], correspondence=[], obligations=[], proof_records=[])
    c0 = level["ladder"]["rungs"]["C0"]
    assert not c0["met"] and c0["status"] == "not-met" and "backends.attention" in c0["reason"] and "gpu.driver" in c0["reason"]
    assert level["ladder"]["rungs"]["C1"]["met"] and level["ladder"]["reached"] == "C3" and level["ladder"]["gaps"] == ["C0", "C2"]


def test_c5_int_record_contract(tmp_path):
    """The C5 reader against INT's `run_composed` layout (branch int-e2e-regression, read at 3e7712f): manifest.json, summary.json,
    reverify/reverify.json; a PASS re-verified by a second process meets C5, anything less names the problem."""
    port = load_port("b0-eager")
    rec = tmp_path / "INT-B0-x"
    (rec / "reverify").mkdir(parents=True)
    (rec / "manifest.json").write_text(json.dumps({"schema": "verity-vllm/e2e-composed/v1", "case": "B0", "command": "python -m e2e.run_composed --case B0 --out DIR",
                                                   "inputs": {"production_manifest": {"engine_flags": port.config.environment_flags, "dtype": "bfloat16",
                                                                                      "compilation": {"enforce_eager": True, "cudagraph_mode": "NONE"},
                                                                                      "gpu": {"compute_capability": "8.9", "name": "NVIDIA GeForce RTX 4090"},
                                                                                      "model": {"repo": port.config.model_repo}}}}))
    stages = {k: {"status": "ok"} for k in ("program", "plan", "production", "preparation", "verify", "obligations", "prove", "bundle")}
    (rec / "summary.json").write_text(json.dumps({"case": "B0", "verdict": "PASS", "partial": False, "stages": stages}))
    (rec / "reverify" / "reverify.json").write_text(json.dumps({"schema": "verity-vllm/e2e-composed/reverify/v1", "recorded_verdict": "PASS", "fresh_verdict": "PASS",
                                                                "agrees": True, "rows": {k: {"status": "ok"} for k in ("program", "plan", "records", "proofs", "bundle")}}))
    r = LD.rung_c5([rec], port)
    assert r.met and r.status == "met" and r.evidence[0]["kind"] == "int-record" and r.evidence[0]["fresh_verdict"] == "PASS"
    # a record of another case, or one whose second process disagrees, does not
    assert not LD.rung_c5([rec], load_port("b1-eager")).met
    rv = json.loads((rec / "reverify" / "reverify.json").read_text())
    rv["fresh_verdict"], rv["agrees"] = "FAIL", False
    (rec / "reverify" / "reverify.json").write_text(json.dumps(rv))
    r = LD.rung_c5([rec], port)
    assert not r.met and r.status == "not-met" and "fresh_verdict 'FAIL'" in r.reason
    (rec / "reverify" / "reverify.json").unlink()
    r = LD.rung_c5([rec], port)
    assert not r.met and "no second-process verification" in r.reason
    assert LD.rung_c5([], port).status == "not-attempted"


# ---------------------------------------------------------------------------------------------------------
# blocker classes: why a profile did not advance (GUIDANCE §24)
# ---------------------------------------------------------------------------------------------------------


def test_blocker_enum_is_closed_and_every_unmet_rung_carries_one():
    assert LD.BLOCKERS == ("SEMANTICS_MISSING", "REPRESENTATION_MISSING", "CAPTURE_MISSING", "AUTHENTICATION_MISSING",
                           "PROOF_LOWERING_MISSING", "IMPLEMENTATION_BUG", "NOT_ATTEMPTED")
    d = _ladder_doc(c0="met")  # C0, C1, C3 met; C2 not-attempted; C4 not-attempted; C5 not-attempted
    for r in LD.RUNGS:
        rec = d["rungs"][r]
        assert (rec["blocker"] is None) == rec["met"], r
        if rec["blocker"]:
            assert rec["blocker"]["class"] in LD.BLOCKERS
    assert d["rungs"]["C2"]["blocker"]["class"] == "NOT_ATTEMPTED"
    # the headline blocker is the first unmet rung above the headline (C3 reached -> C4)
    assert d["blocker"]["rung"] == "C4" and d["blocker"]["class"] == "NOT_ATTEMPTED"
    assert any("[NOT_ATTEMPTED]" in c for c in d["claims"])


def test_blocker_derivation_by_rung_and_taxonomy():
    mk = lambda rung, status, reason=None: LD.RungResult(rung, False, status, reason, [])
    assert LD.derive_blocker(mk("C0", "not-met", "no manifest"), None)["class"] == "CAPTURE_MISSING"
    assert LD.derive_blocker(mk("C1", "not-met"), None)["class"] == "CAPTURE_MISSING"
    assert LD.derive_blocker(mk("C3", "not-met"), None)["class"] == "AUTHENTICATION_MISSING"
    assert LD.derive_blocker(mk("C4", "not-met"), None)["class"] == "PROOF_LOWERING_MISSING"
    assert LD.derive_blocker(mk("C5", "not-met", "stage failed"), None)["class"] == "IMPLEMENTATION_BUG"
    assert LD.derive_blocker(mk("C5", "not-attempted"), None)["class"] == "NOT_ATTEMPTED"
    assert LD.derive_blocker(mk("C2", "absent"), None)["class"] == "NOT_ATTEMPTED"
    # C2 follows the closed taxonomy
    for reason, cls in ((TX.PRIMITIVE_SEMANTICS_MISSING, "SEMANTICS_MISSING"), (TX.COMPACT_REPRESENTATION_MISSING, "REPRESENTATION_MISSING"),
                        (TX.REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE, "CAPTURE_MISSING"), (TX.COMMITMENT_LAYOUT_CANNOT_AUTHENTICATE, "AUTHENTICATION_MISSING")):
        b = LD.derive_blocker(mk("C2", "not-met", "x"), {"reason": reason, "detail": "f"})
        assert b["class"] == cls and b["detail"] == "f", reason
    assert LD.derive_blocker(mk("C2", "not-claimed", "unsupported:primitive-semantics-missing FA3 wgmma"), None)["class"] == "SEMANTICS_MISSING"
    assert LD.derive_blocker(mk("C2", "not-claimed", "typed M1 program: lane CM"), None)["class"] == "REPRESENTATION_MISSING"
    assert LD.derive_blocker(mk("C2", "not-met", "3 mismatches not recorded"), None)["class"] == "SEMANTICS_MISSING"
    assert LD.derive_blocker(LD.RungResult("C2", True, "met", None, []), None) is None


def test_declared_blocker_wins_and_a_resolved_one_becomes_history():
    res = [LV.LevelResult(n, n in (1, 2, 3), "met" if n in (1, 2, 3) else "not-attempted", "t", []) for n in (1, 2, 3, 4)]
    level = LV.verdict("b1-hopper", res, utc="x")
    declared = [{"rung": "C3", "class": "IMPLEMENTATION_BUG", "detail": "profile selection refused Qwen2", "resolved": "select_semantic_profile"},
                {"rung": "C4", "class": "PROOF_LOWERING_MISSING", "detail": "no Rust Hopper primitive"}]
    d = LD.ladder("b1-hopper", level, LD.RungResult("C0", True, "met", "t", []), LD.RungResult("C5", False, "not-attempted", "t", []),
                  TX.SUPPORTED_EXACTLY, None, utc="x", run_dir=None, axis="hw", declared_blockers=declared)
    assert d["reached"] == "C4" if False else d["reached"] == "C3"
    assert d["rungs"]["C3"]["blocker"] is None  # met: the declaration is history, not a blocker
    assert d["blocker_history"] == [{"rung": "C3", "class": "IMPLEMENTATION_BUG", "detail": "profile selection refused Qwen2",
                                     "resolved": "select_semantic_profile", "source": "expectations.json blockers (rung now met)"}]
    assert d["rungs"]["C4"]["blocker"]["class"] == "PROOF_LOWERING_MISSING" and d["rungs"]["C4"]["blocker"]["source"] == "expectations.json blockers"
    assert d["blocker"] == {"rung": "C4", **d["rungs"]["C4"]["blocker"]}
    # the same through the level.json lift
    d2 = LD.ladder_from_level({"port": "b1-hopper", "utc": "x", "level": 3, "levels": {str(n): {"level": n, "met": n in (1, 2, 3), "status": "met" if n in (1, 2, 3) else "not-attempted", "reason": "t", "evidence": []} for n in (1, 2, 3, 4)}},
                              axis="hw", declared_blockers=declared)
    assert d2["blocker_history"][0]["class"] == "IMPLEMENTATION_BUG" and d2["rungs"]["C4"]["blocker"]["class"] == "PROOF_LOWERING_MISSING"


def test_expectations_blockers_are_validated_and_committed_ports_declare_them(tmp_path):
    from verity_vllm_adapter.conformance import port as PT
    base = json.loads((PROFILES / "b1-hopper" / "expectations.json").read_text())
    assert [b["rung"] for b in base["blockers"]] == ["C3", "C4"] and base["blockers"][0]["class"] == "IMPLEMENTATION_BUG" and base["blockers"][0]["resolved"]
    tp = json.loads((PROFILES / "b1-tp2" / "expectations.json").read_text())
    assert tp["blockers"][0] == {**tp["blockers"][0], "rung": "C3", "class": "REPRESENTATION_MISSING"}
    ok = PT._parse_expectations(base, "x")
    assert ok.blockers[0]["rung"] == "C3" and ok.as_json()["blockers"] == list(ok.blockers)
    bad = json.loads(json.dumps(base)); bad["blockers"][0]["class"] = "BAD_CLASS"
    with pytest.raises(PT.PortError, match="field 'class' must be one of"):
        PT._parse_expectations(bad, "x")
    bad = json.loads(json.dumps(base)); bad["blockers"][1]["rung"] = "C3"
    with pytest.raises(PT.PortError, match="declared twice"):
        PT._parse_expectations(bad, "x")
    bad = json.loads(json.dumps(base)); bad["blockers"][0]["rung"] = "C9"
    with pytest.raises(PT.PortError, match="field 'rung'"):
        PT._parse_expectations(bad, "x")
    # L-spelling is accepted and normalised
    alt = json.loads(json.dumps(base)); alt["blockers"][1]["rung"] = "L4"
    assert PT._parse_expectations(alt, "x").blockers[1]["rung"] == "C4"


def test_matrix_has_a_blocker_column():
    text = R.render_matrix(R.collect_ladders(ROOT / "results"), list_ports())
    header = next(l for l in text.splitlines() if l.startswith("| Profile |"))
    assert "| Blocker (next rung) |" in header
    rows = [l for l in text.splitlines() if l.startswith("| `")]
    assert rows and all(("`" in l.split("|")[-3]) for l in rows), [l.split("|")[-3] for l in rows]  # every row names a class (or none)


def test_correspondence_evidence_must_name_the_bound_definition_versions():
    """IT-F3 (`docs/vllm-poc/independent-tests.md`): program identity hashes the static configuration, not the definition
    versions, so TP's `RMSNormFusedCuda_v1` correspondence kept b1-tp2's C2 "met" after the port moved to `@2`. Every
    record names the composite version it was evaluated under; `typed.correspond` now refuses evidence whose versions
    are not the port's, and the ladder classes that rung NOT_ATTEMPTED (nothing is known under the current binding)."""
    from verity_vllm_adapter.conformance import typed as T
    port = load_port(PROFILES / "b1-tp2")
    run = ROOT / "results" / "B1-7f26cdcedcd0-20260908T0229Z"
    dirs = [E.discover(run, port)]
    current = T.correspond(port, sorted((run / "correspondence").glob("*.json")), dirs, None)
    assert current["status"] == "established" and all(not f["definition_version_drift"] for f in current["files"]), current["reason"]
    stale = T.correspond(port, sorted((run / "correspondence-rmsnormv1").glob("*.json")), dirs, None)
    assert stale["status"] == "not-established" and "definition-version drift" in stale["reason"], stale["reason"]
    assert "RMSNormFusedCuda_v1 but the port binds RMSNormFusedCuda@2" in stale["reason"]
    drift = [f["definition_version_drift"]["RMSNormFusedCuda"] for f in stale["files"]]
    assert [d["recorded"] for d in drift] == [1, 1] and [d["bound"] for d in drift] == [2, 2] and sum(d["records"] for d in drift) == 380
    b = LD.derive_blocker(LD.RungResult("C2", False, "not-met", stale["reason"], []), None)
    assert b["class"] == "NOT_ATTEMPTED" and "predates" in b["source"]
