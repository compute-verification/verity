"""SAMPLER (2026-09-10): sampling against the validated partition interface -- population/v2 == population/v1, law equality of the
two-stage sampler over dense and compact populations (and with the legacy e2e.security_sample), refusals (partition digest, other
plan, substituted population, second draw, unsupported predicate, tampered ru_of), predicates, expected work.  TINY only (LP=4,
STEPS=8; < 400 MB)."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

POC = Path(__file__).resolve().parents[1]
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from e2e import expected_work as ew  # noqa: E402
from e2e import partition_sample as ps  # noqa: E402
from e2e import population as popmod  # noqa: E402
from e2e import population_v2 as v2mod  # noqa: E402
from e2e import predicates as pr  # noqa: E402
from e2e import protocol_instance as pi  # noqa: E402
from e2e import security_sample as ss  # noqa: E402

TINY_IDENTITY = {"family": "Serve_v1", "config_sha12": "b60e37f2db33", "case": "TINY", "LP": 4, "STEPS": 8, "function": "Serve_v1{C={b60e37f2db33},LP=4,STEPS=8}"}
TINY_VOCAB = {"digest": "0" * 64, "schema": "verity-ir/registry-version/v0", "profile_name": "tiny-test", "profile_sha256": "1" * 64, "source": "test placeholder"}
TINY_OUTPUTS = [{"name": "token_slot", "multiplicity": 9, "domain_bits_declared": 16, "wire_bits": 32, "enforcement": "test", "source": "STEPS+1 slots"}]
SEED = "ab" * 32


@pytest.fixture(scope="module")
def tiny():
    from veritor.plans.validate import validate_plan

    program, adapted, circuit, pdigest = popmod.build(popmod.TINY, popmod.TINY_LP, popmod.TINY_STEPS)
    plan, meta = popmod.make_plan("P1/cu=operator", adapted, circuit)
    v = validate_plan(circuit, plan)
    p3, meta3 = popmod.make_plan("P3/cu=block", adapted, circuit)
    v3 = validate_plan(circuit, p3)
    return dict(program=program, adapted=adapted, circuit=circuit, pdigest=pdigest, plan=plan, meta=meta, verdict=v, p3=p3, meta3=meta3, verdict3=v3)


@pytest.fixture(scope="module")
def popdir(tiny, tmp_path_factory):
    """A population directory (maps.json, population.json = v1, population.compact.json = v2, compact_validation.json) of the SAME
    validated compact P1/cu=operator plan, as e2e.population maps --population-v2 writes it."""

    out = tmp_path_factory.mktemp("p1op")
    Q, circuit, plan, v, pdigest, meta = tiny["adapted"].program, tiny["circuit"], tiny["plan"], tiny["verdict"], tiny["pdigest"], tiny["meta"]
    popmod.write_maps(out, Q, circuit, plan, v, pdigest, {"vu_level": meta["vu_level"], "cu_strategy": meta["cu_strategy"]}, tables=False)
    art = popmod.validation_artifact(Q, plan, v, pdigest, meta, seconds=0.0, maxrss_mb=None, verdict_source="test")
    (out / "compact_validation.json").write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
    doc = v2mod.build(plan, circuit, program_digest=pdigest)
    (out / "population.compact.json").write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
    # a population/v2 of ANOTHER plan of the same program (P3/cu=block)
    doc3 = v2mod.build(tiny["p3"], circuit, program_digest=pdigest)
    (out / "population.p3.compact.json").write_text(json.dumps(doc3, sort_keys=True, separators=(",", ":")) + "\n")
    return out


@pytest.fixture(scope="module")
def instance(popdir, tmp_path_factory):
    doc = pi.pre_challenge_instance(instance_id="TINY-sampler", population_dir=popdir, identity=dict(TINY_IDENTITY), vocabulary=dict(TINY_VOCAB),
                                    outputs=[dict(o) for o in TINY_OUTPUTS], q="1/2", s="1/64", delta="1/1000000", input_gates=14884)
    f = tmp_path_factory.mktemp("inst") / "protocol_instance.json"
    f.write_text(json.dumps(doc, indent=1, sort_keys=True))
    assert pi.validate_instance(pi.load_instance(f)) == []
    return f


def _sel(out: Path):
    ev = {e["name"]: e for e in json.loads((out / "chronology.json").read_text())["events"]}
    return ev["r_q"]["selection"], ev["r_s"]["selection"], ev["r_q"]["digest"], ev["r_s"]["digest"]


def _run(mod, inst, pop, out, seed=SEED):
    mod.stage1(inst, pop, "11" * 32, out, seed)
    mod.stage2(inst, pop, "33" * 32, out)
    mod.reveal(out)
    assert mod.verify(inst, pop, out) == []
    return _sel(out)


# -- deliverable 1: population/v2 == population/v1 ----------------------------------------------------------------------------------


def test_compact_population_equals_dense(popdir, tiny):
    v1 = json.loads((popdir / "population.json").read_text())
    pop = v2mod.CompactPopulation(json.loads((popdir / "population.compact.json").read_text()))
    assert pop.check(digest=True)["digest_checked"] is True
    cmp = v2mod.compare_with_v1(pop, v1, plan=tiny["plan"])
    assert cmp["ru_of_equal"] and cmp["vu_count"] == 53113 and cmp["ru_count"] == 305 and cmp["intervals_checked"] == 53113
    from veritor.plans.summary import partition_digest

    assert pop.partition_digest == v1["partition_digest"] == partition_digest(tiny["plan"])
    assert pop.plan_digest == v1["plan_digest"] and pop.program_digest == v1["program_digest"]
    assert len(pop.rows) < pop.vu_count / 10  # O(runs): 53,113 VUs in a few thousand rows
    # O(log rows) lookups agree with the dense list on a spread of VUs; kind_counts covers every VU
    for vu in (0, 1, 14883, 14884, 20000, 33333, 53112):
        assert pop.ru_of(vu) == v1["ru_of"][vu]
        lo, hi = pop.gate_interval(vu)
        assert lo < hi and pop.kind_of(vu).size == hi - lo
    assert sum(pop.kind_counts().values()) == pop.vu_count


def test_row_sink_streams_the_same_rows(tiny):
    """RowSink (whole-B0 path: rows streamed to disk, O(1) rows in memory) yields exactly the in-memory rows."""

    kinds, rows, cu_ivs, n = v2mod.build_rows(tiny["plan"], tiny["circuit"])
    streamed: list[list[int]] = []
    sink = v2mod.RowSink(streamed.append)
    kinds2, sink_rows, cu_ivs2, n2 = v2mod.build_rows(tiny["plan"], tiny["circuit"], sink=sink)
    assert sink_rows is sink and sink.count == len(rows) == len(streamed) == 2658 and n == n2 == 53113
    assert streamed == rows and [k.to_json() for k in kinds] == [k.to_json() for k in kinds2] and cu_ivs == cu_ivs2


def test_population_v1_still_readable_and_v2_loader(popdir, instance):
    d = ps.load_population(popdir / "population.json")
    c = ps.load_population(popdir / "population.compact.json", check="full")
    assert d.schema == v2mod.POPULATION_V1 and c.schema == v2mod.POPULATION_V2
    assert (d.vu_count, d.ru_count, d.partition_digest) == (c.vu_count, c.ru_count, c.partition_digest)
    assert all(d.ru_of(vu) == c.ru_of(vu) for vu in range(0, d.vu_count, 97))
    assert all(d.ru_size(r) == c.ru_size(r) for r in range(d.ru_count))
    assert d.kind_of(5) is None and c.kind_of(5) is not None


# -- deliverable 2: the law over the partition --------------------------------------------------------------------------------------


def test_law_equality_dense_compact_and_legacy_sampler(popdir, instance, tmp_path):
    """Same seed => same J, same Q, same challenge digests: legacy security_sample on v1, partition_sample on v1, partition_sample on v2."""

    v1, v2 = popdir / "population.json", popdir / "population.compact.json"
    a = _run(ss, instance, v1, tmp_path / "legacy_v1")
    b = _run(ps, instance, v1, tmp_path / "new_v1")
    c = _run(ps, instance, v2, tmp_path / "new_v2")
    assert a == b == c
    assert 0 < len(a[0]) < 305 and 0 < len(a[1])
    # the legacy verifier accepts the new sampler's chronology on the dense population (same event structure and digests)
    assert ss.verify(instance, v1, tmp_path / "new_v1") == []
    # selected.json lists exactly Q with RU, kind and gate interval of the validated partition
    sel = json.loads((tmp_path / "new_v2" / "selected.json").read_text())
    pop = v2mod.CompactPopulation(json.loads(v2.read_text()))
    assert [r["vu_index"] for r in sel["selected"]] == list(c[1]) and sel["count"] == len(c[1])
    J = set(c[0])
    for r in sel["selected"]:
        assert r["ru_index"] == pop.ru_of(r["vu_index"]) and r["ru_index"] in J
        assert r["partition_digest"] == pop.partition_digest and r["kind"]["fn"] and r["gate_interval"] == list(pop.gate_interval(r["vu_index"]))
    assert sum(sel["kinds"].values()) == sel["count"]


def test_stage2_law_is_independent_per_vu_not_per_row(popdir, instance):
    """draw_Q ranks the VUs of J block by block and draws bernoulli_subset over that ranking: the ranking is the same object for both
    representations (an explicit cross-check of the ranking primitive, independent of the chronology)."""

    d = ps.load_population(popdir / "population.json")
    c = ps.load_population(popdir / "population.compact.json")
    seed = bytes.fromhex(SEED)
    J = tuple(range(0, 305, 7))
    phase = "cd" * 32
    assert ps.draw_Q(seed, phase, J, d, Fraction(1, 16)) == ps.draw_Q(seed, phase, J, c, Fraction(1, 16))
    vus_by_ru = [[] for _ in range(305)]
    for vu, ru in enumerate(json.loads((popdir / "population.json").read_text())["ru_of"]):
        vus_by_ru[ru].append(vu)
    assert ss.draw_Q(seed, phase, J, vus_by_ru, Fraction(1, 16)) == ps.draw_Q(seed, phase, J, c, Fraction(1, 16))


# -- deliverable 5: refusals --------------------------------------------------------------------------------------------------------------


def test_refuse_population_partition_digest_mismatch(popdir, instance, tmp_path):
    doc = json.loads((popdir / "population.compact.json").read_text())
    doc["partition_digest"] = "0" * 64
    doc["content_sha256"] = v2mod.content_sha256(doc)
    f = tmp_path / "wrong_partition.json"
    f.write_text(json.dumps(doc))
    with pytest.raises((ss.ChronologyError, v2mod.PopulationError), match="partition"):
        ps.stage1(instance, f, "11" * 32, tmp_path / "c", SEED)
    assert not (tmp_path / "c" / "chronology.json").exists()


def test_refuse_population_of_a_different_plan(popdir, instance, tmp_path):
    """P3/cu=block population of the same program against the P1/cu=operator instance: another partition -> REFUSE (both the plan
    digest and the partition digest differ; the counts too)."""

    with pytest.raises(ss.ChronologyError, match="instance partition|plan digest|partition_digest"):
        ps.stage1(instance, popdir / "population.p3.compact.json", "11" * 32, tmp_path / "c", SEED)
    assert not (tmp_path / "c").exists() or not (tmp_path / "c" / "chronology.json").exists()


def test_refuse_second_stage2_draw(popdir, instance, tmp_path):
    v2 = popdir / "population.compact.json"
    out = tmp_path / "c"
    ps.stage1(instance, v2, "11" * 32, out, SEED)
    with pytest.raises(ss.ChronologyError, match="reroll"):
        ps.stage1(instance, v2, "22" * 32, out, SEED)
    ps.stage2(instance, v2, "33" * 32, out)
    with pytest.raises(ss.ChronologyError, match="reroll"):
        ps.stage2(instance, v2, "44" * 32, out)
    doc = json.loads((out / "chronology.json").read_text())
    assert [e["name"] for e in doc["events"] if e["kind"] == "challenge"] == ["r_q", "r_s"] and doc["rerolls"] == 0


def test_refuse_population_substituted_between_stages(popdir, instance, tmp_path):
    """Stage 1 pins the population file (content sha256); stage 2 with the other representation of the SAME partition is refused --
    a population swapped between the draws is never accepted silently (S9)."""

    out = tmp_path / "c"
    ps.stage1(instance, popdir / "population.compact.json", "11" * 32, out, SEED)
    with pytest.raises(ss.ChronologyError, match="pinned at stage 1"):
        ps.stage2(instance, popdir / "population.json", "33" * 32, out)
    ps.stage2(instance, popdir / "population.compact.json", "33" * 32, out)
    ps.reveal(out)
    assert ps.verify(instance, popdir / "population.compact.json", out) == []
    problems = ps.verify(instance, popdir / "population.json", out)
    assert problems and "pinned at stage 1" in problems[0]


def test_tampered_ru_of_is_detected(popdir, instance, tmp_path):
    """v2: moving one row to another CU breaks the CU containment / the re-derived partition digest -> the self-check refuses before
    stage 1.  v1: editing ru_of after stage 1 changes the pinned content -> stage 2 / verify refuse."""

    doc = json.loads((popdir / "population.compact.json").read_text())
    row = next(r for r in doc["rows"] if r[2] > 0)
    row[2] -= 1  # claim the row's VUs belong to the previous CU
    doc["content_sha256"] = v2mod.content_sha256(doc)
    f = tmp_path / "tampered.json"
    f.write_text(json.dumps(doc))
    with pytest.raises(v2mod.PopulationError):
        ps.load_population(f, check="full")
    with pytest.raises((ss.ChronologyError, v2mod.PopulationError)):
        ps.stage1(instance, f, "11" * 32, tmp_path / "c1", SEED)
    # v2: re-deriving the digest catches a swap of two rows' CUs that keeps every count (the intervals change)
    doc = json.loads((popdir / "population.compact.json").read_text())
    pop = v2mod.CompactPopulation(doc)
    assert pop.check(digest=True)["digest_checked"]
    # v1 after stage 1
    v1 = json.loads((popdir / "population.json").read_text())
    out = tmp_path / "c2"
    ps.stage1(instance, popdir / "population.json", "11" * 32, out, SEED)
    v1["ru_of"][100], v1["ru_of"][20000] = v1["ru_of"][20000], v1["ru_of"][100]
    g = tmp_path / "v1_tampered.json"
    g.write_text(json.dumps(v1))
    with pytest.raises(ss.ChronologyError, match="pinned at stage 1"):
        ps.stage2(instance, g, "33" * 32, out)


# -- deliverable 3: predicates --------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def preds(tiny, popdir):
    pop = v2mod.CompactPopulation(json.loads((popdir / "population.compact.json").read_text()))
    return pr.Predicates(tiny["circuit"], tiny["plan"], pop, pr.Operators(tiny["circuit"], tiny["p3"], tiny["verdict3"]), verdict=tiny["verdict"])


def _first_vu(preds, prefix: str, form: str | None = None, operator: str | None = None) -> int:
    pop = preds.pop
    for vu in range(pop.vu_count):
        k = pop.kind_of(vu)
        if k.fn.startswith(prefix) and (form is None or k.form == form):
            if operator is None or preds.ops.kind(preds.ops.unit_of_gate(pop.gate_interval(vu)[0])).fn.startswith(operator):
                return vu
    raise AssertionError(f"no VU of kind {prefix}")


def test_unsupported_kind_refuses_with_exact_code(preds):
    """Every attention-interior kind REFUSES with PREDICATE_UNSUPPORTED: interface not committed; pending S55 -- no head-row cone."""

    for prefix in ("DotBf16", "F32FmaSubFtz", "MufuEx2Ftz", "F32Max", "F32AddFtz", "F32MulFtz", "Fa2InvSum", "GuardNegInfZero", "AmpereBF16TcDot16"):
        vu = _first_vu(preds, prefix, operator="Attention")
        p = preds.predicate(vu)
        assert p.status == "REFUSED" and p.code == pr.PREDICATE_UNSUPPORTED, (prefix, p)
        assert p.reason.startswith("interface not committed; pending S55"), (prefix, p.reason)
        assert p.obligation is None and p.adapter is None
    # the loud refusal on a selection
    vu = _first_vu(preds, "DotBf16", operator="Attention")
    sel = {"partition_digest": preds.pop.partition_digest, "selected": [{"vu_index": vu, "ru_index": preds.pop.ru_of(vu)}]}
    with pytest.raises(pr.PredicateError) as ei:
        preds.for_selected(sel, strict=True)
    assert str(ei.value).startswith("PREDICATE_UNSUPPORTED: interface not committed; pending S55")
    assert ei.value.code == pr.PREDICATE_UNSUPPORTED and ei.value.vu == vu
    # interior of a non-attention operator: same code, a different reason (no S55), and no many-to-one adapter
    p = preds.predicate(_first_vu(preds, "F32Add_v1", operator="RMSNorm"))
    assert p.status == "REFUSED" and p.code == pr.PREDICATE_UNSUPPORTED and "interface not committed" in p.reason and "S55" not in p.reason
    p = preds.predicate(_first_vu(preds, "ArgmaxStep"))
    assert p.status == "REFUSED" and p.code == pr.PREDICATE_UNSUPPORTED and p.interface["status"] == "interior"


def test_gemm_coordinate_supported_and_adapters_explicit(preds):
    vu = _first_vu(preds, "GemmCoordinate")
    p = preds.predicate(vu)
    assert p.status == "SUPPORTED" and p.code is None and p.adapter is None
    assert p.interface["status"] == "committed" and p.obligation["checker"]["name"] == "gemm-coordinate" and p.obligation["checker"]["proof"] == "zk"
    assert p.obligation["checker"]["elf_sha256"] == pr.GEMM_COORDINATE_ELF
    opens = p.obligation["open"]
    assert sum(o["role"] == "operand" for o in opens) == 2 * 32 and sum(o["role"] == "output" for o in opens) == 1  # K=32: x row + w row + the word
    assert p.obligation["relation"]["parameters"] == {"K": "32"}
    # committed interface but no approved checker -> REFUSED, exact code, reason names the kind program
    p = preds.predicate(_first_vu(preds, "SiluMulBf16"))
    assert p.status == "REFUSED" and p.code == pr.PREDICATE_UNSUPPORTED and p.interface["status"] == "committed" and "no approved checker" in p.reason
    # the norm output coordinate: ADAPTER (cone), never silently SUPPORTED
    p = preds.predicate(_first_vu(preds, "F2fpBf16", "batch", operator="RMSNorm"))
    assert p.status == "ADAPTER" and p.adapter["unit"] == "cone" and p.interface["status"] == "output-committed" and p.adapter["interior_inputs"]
    # the attention output cast is NOT given the cone adapter (S55)
    p = preds.predicate(_first_vu(preds, "F2fpBf16", "batch", operator="Attention"))
    assert p.status == "REFUSED" and "S55" in p.reason
    # Input leaves: the input equation (verifier check of the committed leaf)
    p = preds.predicate(0)
    assert p.status == "SUPPORTED" and p.obligation["checker"]["name"] == "input-equation" and p.obligation["checker"]["proof"] == "verifier"


def test_predicates_refuse_selection_of_another_partition(preds):
    with pytest.raises(pr.PredicateError, match="PARTITION_MISMATCH"):
        preds.for_selected({"partition_digest": "f" * 64, "selected": []})
    vu = _first_vu(preds, "GemmCoordinate")
    with pytest.raises(pr.PredicateError, match="RU_MISMATCH"):
        preds.for_selected({"partition_digest": preds.pop.partition_digest, "selected": [{"vu_index": vu, "ru_index": preds.pop.ru_of(vu) + 1}]})


# -- deliverable 4: expected work -------------------------------------------------------------------------------------------------------------


def test_expected_work_counts_and_unpriced(popdir):
    counts, meta = ew.kinds_from_population(popdir / "population.compact.json")
    assert sum(counts.values()) == 53113
    t3 = ROOT / "docs" / "data" / "table3a" / "B0-serve-lp32-s15.json"
    doc = ew.expected_work(counts, meta, q=Fraction(1, 2), s=Fraction(1, 64), ru_count=305, instances=1, coefs=ew.coefficients(t3))
    assert doc["E_J"]["value"] == pytest.approx(152.5) and doc["E_Q"]["value"] == pytest.approx(53113 / 128, abs=1e-3)
    assert sum(r["E_Q"] for r in doc["rows"]) == pytest.approx(doc["E_Q"]["value"], abs=0.05)
    unpriced = [r for r in doc["rows"] if not r["priced"]]
    assert unpriced and all(r["expected_gpu_s"] is None and r["cost_source"].startswith("UNPRICED") for r in unpriced)
    assert doc["totals"]["unpriced_E_Q"] > 0 and doc["totals"]["unpriced_kinds"] == len(unpriced)
    # the B0 census population: 166,196,840 VUs (31,157,512 relation + 135,039,328 Input), 3 priced kinds
    counts, meta = ew.kinds_from_census(t3, None)
    assert sum(counts.values()) == 166_196_840 and meta["input_vus"] == 135_039_328
    doc = ew.expected_work(counts, meta, q=Fraction(1, 2), s=Fraction(1, 4096), ru_count=4, instances=4, coefs=ew.coefficients(t3))
    assert doc["E_Q"]["value"] == pytest.approx(166_196_840 * 4 / 8192, rel=1e-6)
    assert doc["totals"]["priced_kinds"] == 3 and doc["totals"]["unpriced_kinds"] == 35
