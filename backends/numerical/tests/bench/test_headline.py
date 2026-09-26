"""The per-workload headline (``bench.headline``): units from the census workload's template mix, native work from the
census work models in both bases, coverage only by cleared results on the same subcircuit and the served hardware, and the
work-weighted overhead over the covered share; uncovered subcircuits listed, never imputed."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from verity_numerical.bench import census
from verity_numerical.bench import headline as H
from verity_numerical.bench import views as V

pytest.importorskip("research.store.local", reason="research.store (tools/research) is not importable")
from research.store.local import LocalStore  # noqa: E402
from test_tables import REDTEAM, store  # noqa: E402, F401  (store: the fixture)
from test_views import NONGEMM, committed, nongemm  # noqa: E402

ROW101 = next(w for w in census.raw("workloads") if w["source"]["row"] == 101)
SPEC = dataclasses.replace(V.SPEC, interaction_validated=False)
L40S = census.HARDWARE["l40s-48gb/bf16"]


def served_on(workload: dict, hardware: str) -> dict:
    """``workload`` as if it had been served on another census hardware line (same mix)."""
    w = copy.deepcopy(workload)
    w["id"], w["serving"]["hardware"] = f"{w['id']}@{hardware}", hardware
    return w


def by(rec: dict, **kw) -> dict:
    return next(g for g in rec["subcircuits"] if all(g[k] == v for k, v in kw.items()))


def test_units_and_native_work_of_row_101_from_the_census() -> None:
    gs = H.groups(ROW101)
    assert sum(g["units"] for g in gs) == sum(m["vus"] * m["subcircuits_per_vu"] for m in ROW101["template_mix"])
    k2048 = next(g for g in gs if g["subcircuit"] == "gemm-coordinate/k2048/sm80-mma-bf16")
    assert k2048["units"] == 4592 * (16384 + 2048 + 3072) + 32 * 128256 and k2048["vus"] == 3 * 4592 + 32
    assert k2048["native_seconds"]["tensor-peak"] == pytest.approx((k2048["units"] * 4096 / L40S.flop_per_second,) * 2)
    assert k2048["native_seconds"]["memory-bound"] == pytest.approx((k2048["units"] * 4096 / 864e9,) * 2)
    # attention: T only to its block -- last=full is NB*BN exactly, partial is ((NB-1)*BN, NB*BN) open
    att = next(g for g in gs if g["template"] == "attention-head")
    lo = hi = 0
    for m in (m for m in ROW101["template_mix"] if m["subcircuit_template"] == "attention-head"):
        NB, BN, D, n = m["parameters"]["NB"], m["parameters"]["BN"], m["parameters"]["D"], m["vus"] * m["subcircuits_per_vu"]
        tlo, thi = (NB * BN, NB * BN) if m["parameters"]["last"] == "full" else ((NB - 1) * BN + 1, NB * BN - 1)
        lo, hi = lo + n * (4 * D + 4 * tlo * D), hi + n * (4 * D + 4 * thi * D)
    assert att["native_seconds"]["memory-bound"] == pytest.approx((lo / 864e9, hi / 864e9))
    assert att["subcircuit"] == "attention-head/d64-bn128/sm80-fa2-bf16"
    assert att["native_seconds"]["tensor-peak"] == att["native_seconds"]["memory-bound"]     # not a tensor template
    rope = next(g for g in gs if g["subcircuit"] == "rope-head/d64/neox-bf16")
    assert rope["units"] == 4592 * (32 + 8) and rope["vu_families"] == ["RoPE_v1"]


def test_nothing_is_covered_without_results_and_every_subcircuit_is_listed(store: LocalStore) -> None:
    rec = next(r for r in H.headline(V.build(store, SPEC)) if r["workload"] == ROW101["id"])
    for row in rec["backends"]:
        assert row["covered"] == [] and len(row["uncovered"]) == len(rec["subcircuits"]) == 9
        for b in H.BASES:
            assert row["bases"][b.id]["share"] == (0.0, 0.0) and row["bases"][b.id]["overhead"] is None
        reasons = {u["display"]: u["reason"] for u in row["uncovered"]}
        assert reasons[by(rec, template="attention-head")["display"]] == "no result"
        assert reasons[by(rec, template="embedding-row")["display"]] == "commitment opening"
        assert set(reasons.values()) == {"no result", "commitment opening"}
        assert reasons[by(rec, subcircuit="silu-mul/i8192/bf16")["display"]] == "no result"
    for b in H.BASES:                                           # the per-subcircuit shares bracket the whole run
        lo, hi = (sum(g["share"][b.id][i] for g in rec["subcircuits"]) for i in (0, 1))
        assert lo <= 1 + 1e-12 and hi >= 1 - 1e-12


def test_results_on_other_hardware_are_listed_not_counted(store: LocalStore) -> None:
    for sub in NONGEMM:
        nongemm(store, sub)                                     # C-Flock on an H100; #101 was served on an L40S
    rec = next(r for r in H.headline(V.build(store, SPEC)) if r["workload"] == ROW101["id"])
    row = next(r for r in rec["backends"] if r["backend"] == "C-interactive")
    assert row["covered"] == [] and row["bases"]["tensor-peak"]["share"] == (0.0, 0.0)
    elsewhere = [u for u in row["uncovered"] if u["subcircuit"] in NONGEMM]
    assert len(elsewhere) == 4 and all(u["reason"] == "only on H100 SXM5" for u in elsewhere)
    assert all(e["prover_hardware"] == "h100-sxm5-80gb/bf16" and e["cleared"] for u in elsewhere for e in u["results_elsewhere"])


def test_covered_share_and_work_weighted_overhead(store: LocalStore) -> None:
    arts = {sub: nongemm(store, sub, t_total=0.4 + 0.1 * i) for i, sub in enumerate(NONGEMM)}
    views = V.build(store, SPEC)
    w = served_on(ROW101, "h100-sxm5-80gb/bf16")
    rec = H.headline(views, [w])[0]
    row = next(r for r in rec["backends"] if r["backend"] == "C-interactive")
    assert {x["subcircuit"]: x["artifact"] for x in row["covered"]} == arts
    P = {c.subcircuit: c.throughput["instances_per_second"] for c in views.admissible}
    for b in H.BASES:
        cov = [by(rec, subcircuit=s)["native_seconds"][b.id] for s in NONGEMM]
        allw = [g["native_seconds"][b.id] for g in rec["subcircuits"]]
        prove = sum(by(rec, subcircuit=s)["units"] / P[s] for s in NONGEMM)
        clo, chi = sum(x[0] for x in cov), sum(x[1] for x in cov)
        tlo, thi = sum(x[0] for x in allw), sum(x[1] for x in allw)
        got = row["bases"][b.id]
        assert got["share"] == pytest.approx((clo / (clo + (thi - chi)), chi / (chi + (tlo - clo))))
        assert got["overhead"] == pytest.approx((prove / chi, prove / clo)) and got["proving_seconds"] == pytest.approx(prove)
        assert got["share"][0] < got["share"][1]                # attention's T leaves the uncovered work a range
    assert {u["reason"] for u in row["uncovered"]} == {"commitment opening", "no result"}
    # the other backends have no result on these subcircuits
    assert all(r["covered"] == [] for r in rec["backends"] if r["backend"] != "C-interactive")


def test_a_provisional_result_does_not_cover(store: LocalStore) -> None:
    w = served_on(ROW101, "rtx-4090/e4m3")
    w["template_mix"] = [{"vu_family": "Gemm_v1", "parameters": {"K": 1536, "N": 8}, "vus": 10, "subcircuit_template": "gemm-coordinate",
                          "subcircuit_parameters": {"K": 1536, "datatype": "e4m3", "semantics": "sm89.mma.m16n8k32.e4m3"}, "subcircuits_per_vu": 8}]
    committed(store, t_total=0.4)
    row = next(r for r in H.headline(V.build(store), [w])[0]["backends"] if r["backend"] == "B-interactive")
    assert row["covered"] == [] and row["uncovered"][0]["reason"] == "not cleared"
    committed(store, t_total=0.9, labels=[("proof_class", "COMPLETE_ZK_BACKEND", REDTEAM, "2026-09-25T01:00:00Z")])
    views = V.build(store)
    rec = H.headline(views, [w])[0]
    row = next(r for r in rec["backends"] if r["backend"] == "B-interactive")
    best = max((c for c in views.admissible if c.line == V.FP8_ADA_MMA.name), key=lambda c: c.throughput["instances_per_second"])
    assert [x["artifact"] for x in row["covered"]] == [best.artifact] and row["bases"]["tensor-peak"]["share"] == (1.0, 1.0)
    N = census.HARDWARE["rtx-4090/e4m3"].flop_per_second / 3072
    assert row["bases"]["tensor-peak"]["overhead"] == pytest.approx((N / best.throughput["instances_per_second"],) * 2)
    assert row["bases"]["tensor-peak"]["overhead"][0] == pytest.approx(best.overhead)       # one subcircuit: Table 2's cell


def test_a_family_without_a_census_template_is_listed_and_left_out() -> None:
    w = copy.deepcopy(ROW101)
    w["template_mix"].append({"vu_family": "Mystery_v1", "parameters": {"N": 4}, "vus": 7, "subcircuit_template": None,
                              "subcircuit_parameters": None, "subcircuits_per_vu": 1})
    rec = H.headline(V.Views(rules=V.SPEC), [w])[0]
    odd = by(rec, template=None)
    assert odd["native_seconds"] == {"tensor-peak": None, "memory-bound": None} and odd["share"]["tensor-peak"] is None
    assert odd["unmodelled"]["memory-bound"] == "no census template" and odd["display"] == "Mystery_v1 (no census template)"
    assert rec["native_seconds"] == H.headline(V.Views(rules=V.SPEC), [ROW101])[0]["native_seconds"]     # left out of both sums
    assert all(odd["display"] in r["bases"]["memory-bound"]["unmodelled"] for r in rec["backends"])
    assert all(u["reason"] == "no census subcircuit" for r in rec["backends"] for u in r["uncovered"] if u["template"] is None)
    assert "not modelled: no census template" in "\n".join(H.render_markdown([rec]))


def test_the_router_has_a_bytes_model_from_its_logits_datatype() -> None:
    wm = census.SUBCIRCUIT_TEMPLATES["moe-router-topk"].work_model
    assert wm.port_datatypes.logits == "bf16" and wm.bytes_per_input == "2*E + 8*TOPK"
    olmoe = next(w for w in census.raw("workloads") if any(m["subcircuit_template"] == "moe-router-topk" for m in w["template_mix"]))
    rec = H.headline(V.Views(rules=V.SPEC), [olmoe])[0]
    router = by(rec, template="moe-router-topk")
    bw = census.HARDWARE[olmoe["serving"]["hardware"]].memory_bytes_per_second
    assert router["native_seconds"]["memory-bound"] == pytest.approx((router["units"] * (2 * 64 + 8 * 8) / bw,) * 2)
    assert all(not g["unmodelled"] for g in rec["subcircuits"])       # every template of every workload is modelled now


def test_101s_bound_subcircuits_carry_bench_spines_ids() -> None:
    from verity_numerical.bench import templates as TM

    for tpl, params in (("attention-head", {"D": 64, "BN": 128}), ("gumbel-top-p-token-select", {"V": 128256}),
                        ("rope-head", {"D": 64}), ("silu-mul", {"I": 8192})):
        spine = TM.subcircuit(tpl, **params)
        assert spine.id in census.SUBCIRCUITS and census.SUBCIRCUITS[spine.id].display == spine.display
    att = census.SUBCIRCUITS["attention-head/d64-bn128/sm80-fa2-bf16"]
    assert (att.variable_ranges.T.min, att.variable_ranges.T.max) == (1, 383)
    lo = min(H._bounds(census.SUBCIRCUIT_TEMPLATES["attention-head"].work_model, m["parameters"])["T"][0]
             for m in ROW101["template_mix"] if m["subcircuit_template"] == "attention-head")
    hi = max(H._bounds(census.SUBCIRCUIT_TEMPLATES["attention-head"].work_model, m["parameters"])["T"][1]
             for m in ROW101["template_mix"] if m["subcircuit_template"] == "attention-head")
    assert (lo, hi) == (1, 383)
    assert {g["subcircuit"] for g in H.groups(ROW101)} >= {"attention-head/d64-bn128/sm80-fa2-bf16", "gumbel-top-p-token-select/v128256/fp32",
                                                          "embedding-row/v128256-h2048/bf16"}
    assert all(g["subcircuit"] for g in H.groups(ROW101))


def test_the_render_and_the_site_json_carry_the_headline_and_legacy_does_not(store: LocalStore) -> None:
    for sub in NONGEMM:
        nongemm(store, sub)
    views = V.build(store, SPEC)
    md = V.render_markdown(views)
    assert "## Workload headline — share of a served run's native work covered, and its work-weighted overhead" in md
    assert f"### {ROW101['display']} — served on L40S · BF16" in md
    sec = md[md.index(f"### {ROW101['display']}"):]
    sec = sec[:sec.index("\n### ")]
    row = next(x for x in sec.splitlines() if x.startswith("| C-interactive |"))
    assert row.startswith("| C-interactive | 0 of 9 | 0% | — | 0% | — | no result: AttentionHead<T,64> · FA2 BN=128 · sm80 mma · BF16, ")
    assert "; commitment opening: Embedding_v1<128256, 2048> · BF16" in row and "; only on H100 SXM5: " in row
    assert md.index("## Workload headline") > md.index("## Table 3") and md.index("## Workload headline") < md.index("Rejected (")
    ent = V.render_entities(views, rendered_at="2026-09-26T08:00:00Z", commit="abc")
    wh = ent["workload_headline"]
    assert [b["id"] for b in wh["bases"]] == ["tensor-peak", "memory-bound"] and ROW101["id"] in ent["census"]["workloads"]
    rec = next(r for r in wh["workloads"] if r["id"] == ROW101["id"])
    assert rec["served_hardware"] == "l40s-48gb/bf16" and len(rec["subcircuits"]) == 9
    row = next(r for r in rec["backends"] if r["backend"] == "C-interactive")
    assert row["bases"]["tensor-peak"]["share"] == {"min": 0.0, "max": 0.0, "display": "0%"} and row["bases"]["tensor-peak"]["overhead"] is None
    json.dumps(wh)                                              # plain JSON
    covered = H.to_json(H.headline(views, [served_on(ROW101, "h100-sxm5-80gb/bf16")]), {r["artifact"]: r["id"] for r in ent["results"]})
    got = next(r for r in covered["workloads"][0]["backends"] if r["backend"] == "C-interactive")["covered"]
    assert got and all(x["result"] and x["result"].startswith("C-interactive@") for x in got)
    legacy = V.build(store, V.LEGACY)
    assert "Workload headline" not in V.render_markdown(legacy) and V.render_entities(legacy)["workload_headline"] is None


def test_the_any_hardware_preview_counts_the_cells_on_other_gpus_and_the_published_rules_do_not(store: LocalStore) -> None:
    import dataclasses

    for sub in NONGEMM:
        nongemm(store, sub)
    strict = next(r for r in H.headline(V.build(store, SPEC)) if r["workload"] == ROW101["id"])
    loose = next(r for r in H.headline(V.build(store, dataclasses.replace(SPEC, headline_any_hardware=True))) if r["workload"] == ROW101["id"])
    row = lambda rec: next(r for r in rec["backends"] if r["backend"] == "C-interactive")  # noqa: E731
    assert row(strict)["covered"] == [] and not SPEC.headline_any_hardware
    assert {c["subcircuit"] for c in row(loose)["covered"]} == set(NONGEMM)


# --- #101 on the L40S: the hardware line the served run divides by, the same semantics ids --------------------------------------

L40S_GPU = "NVIDIA L40S"


def l40s_cell(store: LocalStore, sub: str, *, profile: str | None = None, relation: str | None = None, t_total: float = 0.4,
              redteam: bool = True) -> str:
    """A C-Flock frame-v3 keyed-BLAKE3 result proved on an L40S over ``sub`` (#101's statements), stamped as ``bench.cell``
    registers it, its inputs a registered captured set of ``sub``, verified by a non-producer, red-team cleared unless not."""
    import hashlib

    from test_tables import LANE, VERIFIER, attempt, hashed
    from test_views import FLOCK_LIVE, FLOCK_NAME, PROTOCOL, SWEEP

    digest = hashlib.sha256(f"l40s-{sub}".encode()).hexdigest()
    params = vars(census.SUBCIRCUITS[sub].parameters)
    store.put_artifact("input-set/v1", {"set": sub.replace("/", "-"), "subcircuit": sub, "source": "captured", "n": 4096,
                                        "content_digest": digest, "schema": "vllm-vu-set/v1", "parameters": params})
    doc = hashed("blake3", t_total=t_total, security={"target": -128, "achieved_log2": -195.5})
    fp = doc["workload_fingerprint"]
    fp["software"]["backend"]["name"] = FLOCK_NAME
    fp.update(proof_class="NON_ZK_PROOF", zk_mode="none", hardware={"gpu": {"name": L40S_GPU, "driver": "570.211.01", "cuda": "12.8", "memory_total_bytes": 46068 << 20}, "cpu": {"model": "c"}},
              instances={"dataset": sub.replace("/", "-"), "tier": f"vllm-vu-set/v1:{sub}", "range": [0, 4096], "manifest_sha256": digest,
                         "source": "captured"})
    fp.pop("profile", None)
    if profile:
        fp["profile"] = profile
    if "K" in params:
        fp["K"] = params["K"]
    if relation:
        fp["relation"] = relation
    doc["measurements"] = [m for m in doc["measurements"] if not m["name"].startswith(("rate.", "overhead."))]
    doc["measurements"].append({"name": V.COMMIT, "value": 0.1, "unit": "s"})
    for name, value in FLOCK_LIVE.items():
        if name in ("rtt_method", "t_total_includes_wait", "loopback_method"):
            doc.setdefault("interaction", {})[name] = value
        else:
            doc["measurements"].append({"name": name, "value": value, "unit": {"rounds.sequential_depth": "count", "net.bytes_out": "B",
                                                                                "net.bytes_in": "B", "net.rtt_ms": "ms"}.get(name, "s")})
    doc["protocol"], doc["sweep"] = dict(PROTOCOL), dict(SWEEP)
    doc["cell"] = {"subcircuit": sub, "statement": f"{sub}+frame-v3/blake3-keyed", "backend": "C-interactive"}
    files = store.put_artifact("run-files/v1", {"run_id": doc["run_id"], "dump": "proofs"})
    store.index.set_replica(files, "remote", "2026-09-25T00:00:00Z")
    art = store.put_artifact(T_RESULT_KIND, doc, refs={"run_files": files})
    store.put_attempt(attempt(doc["run_id"], art, LANE))
    store.add_label(art, "candidate", "X", LANE, ts="2026-09-22T10:00:00Z")
    store.add_label(art, "independently_verified", True, VERIFIER, ref=f"{doc['run_id']}:verify", ts="2026-09-22T10:01:00Z")
    if redteam:
        store.add_label(art, "proof_class", "NON_ZK_PROOF", "red-team-flock-2", ts="2026-09-26T07:30:00Z")
    return art


from verity_numerical.bench.tables import RESULT_KIND as T_RESULT_KIND  # noqa: E402

GEMM101 = ("gemm-coordinate/k2048/sm80-mma-bf16", "gemm-coordinate/k8192/sm80-mma-bf16")


def test_l40s_cells_land_on_the_l40s_line_with_101s_statement_ids(store: LocalStore) -> None:
    from verity.proofs.target import FIRST

    gemm = {sub: l40s_cell(store, sub, profile=FIRST.name, relation="bf16-ampere") for sub in GEMM101}
    elem = {sub: l40s_cell(store, sub, profile=FIRST.name) for sub in NONGEMM}               # the prover GPU picks the line
    views = V.build(store, SPEC)
    assert not [(r.get("subcircuit"), r["codes"]) for r in views.rejected if r["artifact"] in {*gemm.values(), *elem.values()}]
    for sub, art in {**gemm, **elem}.items():
        ln = next(x for x in views.lines if x.get("subcircuit") == sub)
        assert ln["line"] == V.L40S_BF16.name and ln["device"] == "NVIDIA L40S 48GB"
        cell = next(c for c in ln["cells"].values() if c)
        assert cell.artifact == art and not cell.provisional
    k2048 = next(x for x in views.lines if x.get("subcircuit") == GEMM101[0])["native"]
    assert k2048["instances_per_second"] == pytest.approx(362.05e12 / 4096)
    rope = next(x for x in views.lines if x.get("subcircuit") == NONGEMM[0])["native"]
    assert rope["unit"] == "memory" and rope["instances_per_second"] == pytest.approx(864e9 / (6 * 64))
    # no L40S row at the targets' own K = 1536, and the targets' rows are untouched
    assert all(x.get("subcircuit") for x in views.lines if x["line"] == V.L40S_BF16.name)
    ent = V.render_entities(views, rendered_at="2026-09-26T09:00:00Z", commit="abc")
    r = next(x for x in ent["results"] if x["artifact"] == gemm[GEMM101[0]])
    assert r["hardware"] == "l40s-48gb/bf16" and r["prover_hardware"] == "l40s-48gb/bf16" and r["subcircuit"] == GEMM101[0]
    # the headline: #101's GEMM and elementwise work is covered by C-interactive, attention and sampling are not, embedding never
    rec = next(x for x in H.headline(views) if x["workload"] == ROW101["id"])
    row = next(x for x in rec["backends"] if x["backend"] == "C-interactive")
    assert {x["subcircuit"] for x in row["covered"]} == {*GEMM101, *NONGEMM}
    assert {u["subcircuit"]: u["reason"] for u in row["uncovered"]} == {
        "attention-head/d64-bn128/sm80-fa2-bf16": "no result", "gumbel-top-p-token-select/v128256/fp32": "no result",
        "embedding-row/v128256-h2048/bf16": "commitment opening"}
    tp, mb = row["bases"]["tensor-peak"], row["bases"]["memory-bound"]
    assert 0.15 < tp["share"][0] < tp["share"][1] < 0.4 and 0.98 < mb["share"][0] < 0.99
    assert tp["overhead"] is not None and mb["overhead"] is not None


def test_an_l40s_sampling_cell_with_no_tensor_datatype_finds_the_l40s_line(store: LocalStore) -> None:
    from verity.proofs.target import FIRST

    art = l40s_cell(store, "gumbel-top-p-token-select/v128256/fp32", profile=FIRST.name)
    views = V.build(store, SPEC)
    ln = next(x for x in views.lines if x.get("subcircuit") == "gumbel-top-p-token-select/v128256/fp32")
    assert ln["line"] == V.L40S_BF16.name and next(c for c in ln["cells"].values() if c).artifact == art
    assert ln["native"]["instances_per_second"] == pytest.approx(864e9 / (2 * 128256 + 36))


def test_an_l40s_cell_is_still_red_team_gated_and_an_h100_profile_does_not_pull_it_off_its_device(store: LocalStore) -> None:
    from verity.proofs.target import BF16_HOPPER

    uncleared = l40s_cell(store, NONGEMM[1], profile=BF16_HOPPER.name, redteam=False)
    profiled = l40s_cell(store, NONGEMM[2], profile=BF16_HOPPER.name)                       # a stale H100 profile on an L40S proof
    views = V.build(store, SPEC)
    why = next(r for r in views.rejected if r["artifact"] == uncleared)
    assert why["line"] == V.L40S_BF16.name and why["codes"][0][0] == "U"
    ln = next(x for x in views.lines if x.get("subcircuit") == NONGEMM[2])
    assert ln["line"] == V.L40S_BF16.name and next(c for c in ln["cells"].values() if c).artifact == profiled
