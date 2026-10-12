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

PROVEN = [s for s in NONGEMM if not s.startswith("rmsnorm")]   # the RMSNorms leave their row tail to the verifier (census)

ROW101 = next(w for w in census.raw("workloads") if w["source"]["row"] == 101)
SPEC = dataclasses.replace(V.SPEC, interaction_validated=False, headline_strict_verifier=False)   # the footnote machinery; V.SPEC is strict
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
    exact = sum(16 * 32 * (4 * 64 + 4 * T * 64) for T in range(1, 288)) / 864e9      # 16 VUs x 32 heads at every T = 1..287
    assert att["native_seconds"]["memory-bound"] == pytest.approx((exact, exact)) and lo / 864e9 < exact < hi / 864e9
    assert sorted(att["values"]) == list(range(1, 288)) and all(x["units"] == 16 * 32 for x in att["values"].values())
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
    assert {x["subcircuit"]: x["artifact"] for x in row["covered"]} == {s: a for s, a in arts.items() if s in PROVEN}
    P = {c.subcircuit: c.throughput["instances_per_second"] for c in views.admissible}
    for b in H.BASES:
        cov = [by(rec, subcircuit=s)["native_seconds"][b.id] for s in PROVEN]
        allw = [g["native_seconds"][b.id] for g in rec["subcircuits"]]
        prove = sum(by(rec, subcircuit=s)["units"] / P[s] for s in PROVEN)
        clo, chi = sum(x[0] for x in cov), sum(x[1] for x in cov)
        tlo, thi = sum(x[0] for x in allw), sum(x[1] for x in allw)
        got = row["bases"][b.id]
        assert got["share"] == pytest.approx((clo / (clo + (thi - chi)), chi / (chi + (tlo - clo))))
        assert got["overhead"] == pytest.approx((prove / chi, prove / clo)) and got["proving_seconds"] == pytest.approx(prove)
        assert got["share"][0] == pytest.approx(got["share"][1])  # #101's per-T counts make attention's work exact
    assert {u["reason"] for u in row["uncovered"]} == {"commitment opening", "no result", "omits verifier-evaluated parts"}
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
    lo, hi = (f(H._bounds(census.SUBCIRCUIT_TEMPLATES["attention-head"].work_model, m["parameters"])["T"][i]
                for m in ROW101["template_mix"] if m["subcircuit_template"] == "attention-head") for f, i in ((min, 0), (max, 1)))
    assert (lo, hi) == (1, 383)                                 # #101's own strata; the bound subcircuit's range spans every row it serves
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
    assert {c["subcircuit"] for c in row(loose)["covered"]} == set(PROVEN)


# --- #101 on the L40S: the hardware line the served run divides by, the same semantics ids --------------------------------------

L40S_GPU = "NVIDIA L40S"


def l40s_cell(store: LocalStore, sub: str, *, profile: str | None = None, relation: str | None = None, t_total: float = 0.4,
              redteam: bool = True, T: int | None = None, set_T: int | None = None, klass: tuple[int, int] | None = None,
              key_counts: dict | None = None, per_key_count: dict | None = None, source: str = "captured") -> str:
    """A C-Flock frame-v3 keyed-BLAKE3 result proved on an L40S over ``sub`` (#101's statements), stamped as ``bench.cell``
    registers it, its inputs a registered captured set of ``sub``, verified by a non-producer, red-team cleared unless not."""
    import hashlib

    from test_tables import LANE, VERIFIER, attempt, hashed
    from test_views import FLOCK_LIVE, FLOCK_NAME, PROTOCOL, SWEEP

    digest = hashlib.sha256(f"l40s-{sub}-{T}-{set_T}-{klass}".encode()).hexdigest()
    params = vars(census.SUBCIRCUITS[sub].parameters)
    name = sub.replace("/", "-") + (f"-t{set_T if set_T is not None else T}" if T is not None else
                                    f"-class-t{klass[0]}-{klass[1]}" if klass else "")
    store.put_artifact("input-set/v1", {"set": name, "subcircuit": sub, "source": source, "n": 4096,
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
    if T is not None:                   # as flock-ir-lowering's ir_bench records an attention cell: no GEMM target, T in the unit name
        fp.update(profile=None, K=None, subcircuit=sub, subcircuit_template=sub.split("/")[0])
        fp["software"]["backend"]["unit"] = f"attention-head/fa2-bf16/tc-step/d64-bn128/t{T}"
        fp["instances"]["tier"] = name
    if klass is not None:               # as ir_bench records a class cell (4eb3b991): key_counts, key_class; per_key_count on the doc
        fp.update(profile=None, K=None, subcircuit=sub, subcircuit_template=sub.split("/")[0],
                  key_counts={str(t): h for t, h in key_counts.items()},
                  key_class={"T": list(klass), "pin": "2f102216" + "0" * 56, "manifest": "flock-ir-class/v1"})
        fp["software"]["backend"]["unit"] = f"attention-head/fa2-bf16/tc-step/d64-bn128/class-t{klass[0]}-{klass[1]}"
        fp["instances"]["tier"] = name
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
    if per_key_count is not None:
        doc["per_key_count"] = {str(t): {"heads": h, "e2e_s": e, "prove_s": 0.8 * e, "verify_s": 0.1 * e} for t, (h, e) in per_key_count.items()}
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
    from verity.ml.circuits.targets.target import FIRST

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
    assert {x["subcircuit"] for x in row["covered"]} == {*GEMM101, *PROVEN}
    assert {u["subcircuit"]: u["reason"] for u in row["uncovered"]} == {
        "attention-head/d64-bn128/sm80-fa2-bf16": "no result", "gumbel-top-p-token-select/v128256/fp32": "no result",
        "embedding-row/v128256-h2048/bf16": "commitment opening",
        **{s: "omits verifier-evaluated parts" for s in NONGEMM if s not in PROVEN}}
    tp, mb = row["bases"]["tensor-peak"], row["bases"]["memory-bound"]
    assert 0.15 < tp["share"][0] < 0.4 and tp["share"][0] == pytest.approx(tp["share"][1]) and 0.98 < mb["share"][0] < 0.995
    assert tp["overhead"] is not None and mb["overhead"] is not None


def test_an_l40s_sampling_cell_with_no_tensor_datatype_finds_the_l40s_line(store: LocalStore) -> None:
    from verity.ml.circuits.targets.target import FIRST

    art = l40s_cell(store, "gumbel-top-p-token-select/v128256/fp32", profile=FIRST.name)
    views = V.build(store, SPEC)
    ln = next(x for x in views.lines if x.get("subcircuit") == "gumbel-top-p-token-select/v128256/fp32")
    assert ln["line"] == V.L40S_BF16.name and next(c for c in ln["cells"].values() if c).artifact == art
    assert ln["native"]["instances_per_second"] == pytest.approx(864e9 / (2 * 128256 + 36))


def test_an_l40s_cell_is_still_red_team_gated_and_an_h100_profile_does_not_pull_it_off_its_device(store: LocalStore) -> None:
    from verity.ml.circuits.targets.target import BF16_HOPPER

    uncleared = l40s_cell(store, NONGEMM[1], profile=BF16_HOPPER.name, redteam=False)
    profiled = l40s_cell(store, NONGEMM[2], profile=BF16_HOPPER.name)                       # a stale H100 profile on an L40S proof
    views = V.build(store, SPEC)
    why = next(r for r in views.rejected if r["artifact"] == uncleared)
    assert why["line"] == V.L40S_BF16.name and why["codes"][0][0] == "U"
    ln = next(x for x in views.lines if x.get("subcircuit") == NONGEMM[2])
    assert ln["line"] == V.L40S_BF16.name and next(c for c in ln["cells"].values() if c).artifact == profiled


# --- every served workload bound the same way ------------------------------------------------------------------------------------

#: Templates (and attention semantics) the bench spine binds but cannot generate synthetic sets for yet, or does not bind at all:
#: the census binds them in the spine's id form.  A change here is a spine change (a new generator) or a new served template.
NO_SPINE_GENERATOR: set[str] = set()


def _spine_key(sid: str) -> str:
    tpl, _, rest = sid.partition("/")
    return f"{tpl}/{rest.rsplit('/', 1)[1]}" if tpl == "attention-head" and not rest.endswith("sm80-fa2-bf16") else tpl


def test_every_served_workload_resolves_to_bound_subcircuits_in_the_spines_id_form() -> None:
    from verity_numerical.bench import templates as TM

    seen = set()
    for w in census.raw("workloads"):
        for g in H.groups(w):
            assert g["subcircuit"], (w["id"], g["display"])                   # every template of every workload is bound
            seen.add(g["subcircuit"])
    missing = set()
    for sid in seen:
        try:
            spine = TM.parse_subcircuit_id(sid)
        except (KeyError, ValueError):
            spine = None
        if spine is None or spine.id != sid:                    # parse_subcircuit_id reads FA3's id as FA2's: not the spine's
            missing.add(_spine_key(sid))
            continue
        assert spine.display == census.SUBCIRCUITS[sid].display
    assert missing == NO_SPINE_GENERATOR
    for sid in seen:                                            # the spine's id form: <template>/<parameters>/<semantics or datatype>
        tpl, params, tail = sid.split("/")
        assert tpl == census.SUBCIRCUITS[sid].template and params and tail
    for s in census.SUBCIRCUITS.values():
        if s.template == "attention-head":
            hits = [m for w in census.raw("workloads") for m in w["template_mix"]
                    if m["subcircuit_template"] == "attention-head" and m["subcircuit_parameters"] == vars(s.parameters)]
            bounds = [H._bounds(census.SUBCIRCUIT_TEMPLATES["attention-head"].work_model, m["parameters"])["T"] for m in hits]
            assert (s.variable_ranges.T.min, s.variable_ranges.T.max) == (min(b[0] for b in bounds), max(b[1] for b in bounds))


def test_each_workloads_gaps_are_named_by_subcircuit(store: LocalStore) -> None:
    recs = H.headline(V.build(store, SPEC))
    assert len(recs) == len(census.raw("workloads"))
    for rec in recs:
        for row in rec["backends"]:
            assert {u["reason"] for u in row["uncovered"]} <= {"no result", "commitment opening"}
            assert all(u["subcircuit"] for u in row["uncovered"])


@pytest.mark.parametrize("sid", ["gemm-coordinate/k2304/sm80-mma-bf16", "gemm-coordinate/k9728/sm90-wgmma-bf16",
                                 "rmsnorm-triton/n128-eps1e-06/bf16", "rope-head/d256/neox-bf16", "silu-mul/i1024/bf16"])
def test_the_spine_generates_synthetic_sets_for_other_workloads_subcircuits(sid: str, tmp_path) -> None:
    from verity_numerical.bench import generate as G
    from verity_numerical.bench import templates as TM

    if not sid.startswith("gemm-coordinate/"):
        pytest.importorskip("verity_vllm", reason="non-GEMM templates evaluate through the vLLM integration's Definitions")
    out = G.generate(TM.parse_subcircuit_id(sid), 2, 0, tmp_path)
    assert out["draws"] >= 2


def test_below_bar_proofs_on_a_workloads_inputs_are_listed_and_never_counted(store: LocalStore) -> None:
    from test_tables import put, result

    res = result(candidate="SP1", profile=None, subcircuit="Gemm_v1", instances=None)
    res["workload_fingerprint"].update(definition={"id": "GemmCoordinate_v1{K=2048}"}, modes=["core"],
                                       instances={"set": "gemm-set", "set_art": "art:" + "b" * 64, "row": ROW101["source"]["row"]})
    res["measurements"].append({"name": "core.soundness_bits", "value": 95.8, "unit": "count"})
    put(store, res, labels=[("verified", "accepted", "verify-night-3", "2026-09-26T10:10:00Z"),
                            ("below_bar", "true", "verify-night-3", "2026-09-26T10:10:00Z")])
    rec = next(r for r in H.headline(V.build(store, SPEC)) if r["workload"] == ROW101["id"])
    assert [x["definition"] for x in rec["below_bar"]] == ["GemmCoordinate_v1{K=2048}"] and rec["below_bar"][0]["soundness_bits"] == 95.8
    assert all(row["covered"] == [] for row in rec["backends"])
    assert "Below the 128-bit bar, verified but not counted: 1 proofs" in "\n".join(H.render_markdown([rec]))


# --- attention per key count: a cell covers #101's attention units at its own T, weighted by the template mix -------------------

ATT = "attention-head/d64-bn128/sm80-fa2-bf16"
DRAWN = (1, 2, 3, 4, 128, 129, 130, 131, 132, 256, 257, 258, 259, 260, 261, 287)       # the captured set's 16 key counts (art:6312cb50)
DONE = (1, 2, 3, 4, 128, 129, 130, 131, 132, 256, 257, 258, 287)                        # 13 of them with cells


def _register_drawn(store: LocalStore) -> None:
    import hashlib

    for t in DRAWN:
        store.put_artifact("input-set/v1", {"set": f"{ATT.replace('/', '-')}-captured-t{t}", "subcircuit": ATT, "source": "captured", "n": 64,
                                            "row": 101, "schema": "vllm-vu-set/v1", "parameters": vars(census.SUBCIRCUITS[ATT].parameters),
                                            "content_digest": hashlib.sha256(f"drawn-{t}".encode()).hexdigest()})


def test_the_key_count_is_read_where_the_cells_record_it() -> None:
    sets = {"d": ("art:set", {"set": "attention-head-d64-bn128-sm80-fa2-bf16-t4", "subcircuit": ATT})}
    unit = {"workload_fingerprint": {"software": {"backend": {"unit": "attention-head/fa2-bf16/tc-step/d64-bn128/t4"}},
                                     "instances": {"content_digest": "d"}}}
    assert V.input_variables(unit, ATT, sets) == ({"T": 4}, None)
    assert V.input_variables({"workload_fingerprint": {"statics": {"T": 258}}}, ATT, {}) == ({"T": 258}, None)
    assert V.input_variables({"workload_fingerprint": {}}, ATT, {})[1].startswith("its work model needs T")
    clash = {"workload_fingerprint": {"software": {"backend": {"unit": "x/t4"}}, "statics": {"T": 5}}}
    assert "recorded differently: 4 (unit); 5 (statics)" in V.input_variables(clash, ATT, {})[1]
    assert V.input_variables({}, "rope-head/d64/neox-bf16", {}) == ({}, None)                  # no per-input variable


def test_attention_cells_cover_101_per_key_count_and_nothing_between(store: LocalStore) -> None:
    arts = {t: l40s_cell(store, ATT, T=t, t_total=0.2 + 0.01 * t) for t in DONE}
    l40s_cell(store, ATT, T=300)                                                                 # a T #101 never serves
    clash = l40s_cell(store, ATT, T=5, set_T=6)                                                  # unit says 5, set name 6
    _register_drawn(store)
    views = V.build(store, SPEC)
    assert "recorded differently" in next(r for r in views.rejected if r["artifact"] == clash)["codes"][0][1]
    rows = {ln["variables"]["T"]: ln for ln in views.lines if ln.get("subcircuit") == ATT}
    assert set(rows) == {*DONE, 300} and rows[4]["subcircuit_display"].endswith("· T = 4")
    assert rows[258]["native"]["instances_per_second"] == pytest.approx(864e9 / (4 * 64 + 4 * 258 * 64))
    assert next(c for c in rows[4]["cells"].values() if c).to_json()["variables"] == {"T": 4}
    rec = next(r for r in H.headline(views) if r["workload"] == ROW101["id"])
    row = next(x for x in rec["backends"] if x["backend"] == "C-interactive")
    att = next(x for x in row["covered"] if x["subcircuit"] == ATT)
    assert att["covered_values"] == list(DONE) and att["drawn_values"] == list(DRAWN) and att["population_values"] == 287
    assert att["values_elsewhere"] == [300] and [x["artifact"] for x in att["per_value"]] == [arts[t] for t in DONE]
    assert all(x["units"] == 16 * 32 for x in att["per_value"]) and att["units"] == 13 * 16 * 32
    P = {c.variables["T"]: c.throughput["instances_per_second"] for c in views.admissible if c.subcircuit == ATT}
    native = {t: 16 * 32 * (4 * 64 + 4 * t * 64) / 864e9 for t in DONE}
    assert att["native_seconds"]["memory-bound"] == pytest.approx(sum(native.values()))
    assert att["proving_seconds"] == pytest.approx(sum(16 * 32 / P[t] for t in DONE))
    gap = next(u for u in row["uncovered"] if u["subcircuit"] == ATT)
    assert gap["reason"] == "not at every T" and gap["missing_values"] == 287 - 13 and gap["units"] == (287 - 13) * 16 * 32
    total = rec["native_seconds"]["memory-bound"][0]
    assert row["bases"]["memory-bound"]["share"][0] == pytest.approx(sum(native.values()) / total)
    assert row["bases"]["memory-bound"]["overhead"][0] == pytest.approx(att["proving_seconds"] / sum(native.values()))
    assert att["verifier_evaluated"]["share"] == "4–11% of a head's scalar ops, all of its exp2 and reciprocal work"
    md = "\n".join(H.render_markdown([rec]))
    assert "C-interactive† at T = 1–4, 128–132, 256–258, 287 (13 of 287)" in md
    assert "drawn T = 1–4, 128–132, 256–261, 287; 259, 260, 261 drawn but not covered" in md
    assert ("† Verifier-evaluated, AttentionHead<T,64> · FA2 BN=128 · sm80 mma · BF16: the softmax" in md
            and "is checked natively by the verifier, not proven; counted: red-team-flock-3 granted attention" in md)
    ent = H.to_json([rec])["workloads"][0]
    got = next(x for x in next(b for b in ent["backends"] if b["backend"] == "C-interactive")["covered"] if x["subcircuit"] == ATT)
    assert len(got["per_value"]) == 13 and got["covered_values"] == list(DONE)
    full = V.render_markdown(views)
    assert "· AttentionHead<T,64> · FA2 BN=128 · sm80 mma · BF16 · T = 258 ·" in full
    site = V.render_entities(views, rendered_at="2026-09-26T12:00:00Z", commit="abc")
    res = [r for r in site["results"] if r["subcircuit"] == ATT]
    assert len({r["id"] for r in res}) == len(res) == 14 and {r["variables"]["T"] for r in res} == {*DONE, 300}   # one per T
    assert V.parity(store) == []


def test_without_per_key_count_counts_attention_cells_are_not_extrapolated(store: LocalStore) -> None:
    l40s_cell(store, ATT, T=4)
    w = copy.deepcopy(ROW101)
    for m in w["template_mix"]:
        m.pop("variable_counts", None)
    row = next(x for x in H.headline(V.build(store, SPEC), [w])[0]["backends"] if x["backend"] == "C-interactive")
    assert not [x for x in row["covered"] if x["subcircuit"] == ATT]
    gap = next(u for u in row["uncovered"] if u["subcircuit"] == ATT)
    assert gap["reason"] == "no per-T counts" and "never extrapolated" in gap["text"]


def test_sampling_stays_uncounted_until_the_one_verifier_evaluated_switch(store: LocalStore) -> None:
    from verity.ml.circuits.targets.target import FIRST

    gumbel = "gumbel-top-p-token-select/v128256/fp32"
    art = l40s_cell(store, gumbel, profile=FIRST.name)
    views = V.build(store, SPEC)
    row = next(x for x in next(r for r in H.headline(views) if r["workload"] == ROW101["id"])["backends"] if x["backend"] == "C-interactive")
    gap = next(u for u in row["uncovered"] if u["subcircuit"] == gumbel)
    assert gap["reason"] == "omits verifier-evaluated parts" and "most of the template" in gap["text"]
    flipped = V.build(store, dataclasses.replace(SPEC, headline_count_omitted=True))
    rec = next(r for r in H.headline(flipped) if r["workload"] == ROW101["id"])
    row = next(x for x in rec["backends"] if x["backend"] == "C-interactive")
    hit = next(x for x in row["covered"] if x["subcircuit"] == gumbel)
    assert hit["artifact"] == art and hit["verifier_evaluated"]["ruling"].startswith("counted by the verifier-evaluated switch")
    assert "† Verifier-evaluated, GumbelTopPTokenSelect_v1<128256>" in "\n".join(H.render_markdown([rec]))


def test_a_class_cell_credits_each_listed_key_count_at_its_own_throughput(store: LocalStore) -> None:
    """flock-ir-lowering's class-pin statement (red-team-flock-3's grant, one pin per class T 1–128, 129–256, 257–512): one cell
    spans several T; each T in its key_counts is credited at heads / e2e_s from per_key_count, native N per T; nothing else."""
    low = l40s_cell(store, ATT, klass=(1, 128), key_counts={1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 128: 64, 200: 64},
                    per_key_count={1: (64, 0.5), 2: (64, 0.5), 3: (64, 0.6), 4: (64, 0.7), 128: (64, 2.0), 200: (64, 1.0)})
    high = l40s_cell(store, ATT, klass=(257, 512), key_counts={257: 64, 258: 64, 287: 64, 300: 64},
                     per_key_count={257: (64, 3.0), 258: (64, 3.0), 287: (64, 3.2), 300: (64, 3.4)})
    single = l40s_cell(store, ATT, T=4, t_total=0.05)                                            # faster at T = 4 than the class cell
    views = V.build(store, SPEC)
    assert not [r for r in views.rejected if r["artifact"] in (low, high, single)]
    rows = {ln["variables"]["T"]: ln for ln in views.lines if ln.get("subcircuit") == ATT}
    assert {"1–128", "257–512", 4} <= set(rows) and rows["1–128"]["subcircuit_display"].endswith("· T = 1–128")
    cell = next(c for c in rows["1–128"]["cells"].values() if c)
    per_n = {t: 864e9 / (4 * 64 + 4 * t * 64) for t in (1, 2, 3, 4, 128)}
    n_eff = 5 * 64 / sum(64 / per_n[t] for t in per_n)                           # work-weighted over the cell's credited T
    assert cell.artifact == low and cell.native == pytest.approx(n_eff) and cell.overhead == pytest.approx(n_eff / cell.throughput["instances_per_second"])
    rec = next(r for r in H.headline(views) if r["workload"] == ROW101["id"])
    row = next(x for x in rec["backends"] if x["backend"] == "C-interactive")
    att = next(x for x in row["covered"] if x["subcircuit"] == ATT)
    assert att["covered_values"] == [1, 2, 3, 4, 128, 257, 258, 287]           # T 5 (no timing) and 200 (outside 1–128) not credited
    assert att["values_elsewhere"] == [300]                                      # listed, but #101 never serves it
    pv = {x["T"]: x for x in att["per_value"]}
    assert pv[1]["artifact"] == low and pv[1]["P"] == pytest.approx(64 / 0.5) and pv[1]["class"] == [1, 128]
    assert pv[287]["artifact"] == high and pv[287]["P"] == pytest.approx(64 / 3.2)
    assert pv[4]["artifact"] == single and "class" not in pv[4]                  # the single-T cell is faster there
    assert pv[128]["proving_seconds"] == pytest.approx(16 * 32 / (64 / 2.0))
    assert att["native_seconds"]["memory-bound"] == pytest.approx(sum(16 * 32 * (4 * 64 + 4 * t * 64) for t in pv) / 864e9)
    assert att["verifier_evaluated"]["share"].startswith("4–11%")
    md = "\n".join(H.render_markdown([rec]))
    assert "C-interactive† at T = 1–4, 128, 257–258, 287 (8 of 287)" in md and "† Verifier-evaluated, AttentionHead<T,64>" in md
    site = V.render_entities(views, rendered_at="2026-09-26T14:00:00Z", commit="abc")
    r = next(x for x in site["results"] if x["artifact"] == low)
    assert r["key_class"]["T"] == [1, 128] and set(r["key_class"]["key_counts"]) == {"1", "2", "3", "4", "128"}
    assert r["key_class"]["not_credited"] == {"5": "no per_key_count timing", "200": "outside the class 1–128"}
    assert V.parity(store) == []


def test_key_counts_proved_on_synthetic_inputs_count_with_a_provenance_footnote(store: LocalStore) -> None:
    """red-team-flock-3's CP6 (coordinator ruling 2026-09-26, open to Daniel): a class cell on the spine generator's synthetic
    set covers its T like a captured one, and the covered entry says which T are synthetic."""
    syn = l40s_cell(store, ATT, klass=(1, 128), key_counts={1: 64, 2: 64, 3: 64}, source="synthetic",
                    per_key_count={1: (64, 0.5), 2: (64, 0.5), 3: (64, 0.6)})
    cap = l40s_cell(store, ATT, T=4, t_total=0.05)
    rec = next(r for r in H.headline(V.build(store, SPEC)) if r["workload"] == ROW101["id"])
    att = next(x for x in next(b for b in rec["backends"] if b["backend"] == "C-interactive")["covered"] if x["subcircuit"] == ATT)
    assert att["covered_values"] == [1, 2, 3, 4]
    pv = {x["T"]: x for x in att["per_value"]}
    assert pv[1]["artifact"] == syn and pv[1]["source"] == "synthetic" and pv[4]["artifact"] == cap and pv[4]["source"] == "captured"
    assert att["provenance"]["synthetic_values"] == [1, 2, 3] and "3 of the 4 covered T values" in att["provenance"]["text"]
    md = "\n".join(H.render_markdown([rec]))
    assert "C-interactive†‡ at T = 1–4" in md and "‡ Synthetic inputs, AttentionHead<T,64>" in md and "proved on synthetic inputs" in md


def test_a_broken_class_cell_is_rejected_not_guessed(store: LocalStore) -> None:
    none = l40s_cell(store, ATT, klass=(1, 128), key_counts={1: 64}, per_key_count={})
    views = V.build(store, SPEC)
    why = next(r for r in views.rejected if r["artifact"] == none)["codes"][0]
    assert why[0] == "X" and "no creditable key count (T 1: no per_key_count timing)" in why[1]


def test_the_strict_verifier_preview_leaves_every_verifier_evaluated_template_uncovered(store: LocalStore) -> None:
    """Daniel's standing rule (2026-09-26), as a preview: the verifier evaluates no part of the computation, so attention (the
    softmax), the RMSNorms (the row tail) and sampling are not covered; RoPE, which leaves nothing to the verifier, still is."""
    from verity.ml.circuits.targets.target import FIRST

    rms = "rmsnorm-fused-cuda/n2048-eps1e-05/bf16"
    l40s_cell(store, ATT, T=4)
    l40s_cell(store, rms, profile=FIRST.name)
    l40s_cell(store, "rope-head/d64/neox-bf16", profile=FIRST.name)
    l40s_cell(store, "gumbel-top-p-token-select/v128256/fp32", profile=FIRST.name)

    def row(rules):
        rec = next(r for r in H.headline(V.build(store, rules)) if r["workload"] == ROW101["id"])
        return next(x for x in rec["backends"] if x["backend"] == "C-interactive")

    loose = {x["subcircuit"] for x in row(SPEC)["covered"]}            # the RMSNorm row tail is never counted (census, 2026-09-26)
    assert {ATT, "rope-head/d64/neox-bf16"} <= loose and rms not in loose
    assert V.SPEC.headline_strict_verifier                                # the published rules are the strict reading
    strict = row(dataclasses.replace(SPEC, headline_strict_verifier=True, headline_count_omitted=True))
    assert {x["subcircuit"] for x in strict["covered"]} == {"rope-head/d64/neox-bf16"}
    gaps = {u["subcircuit"]: u for u in strict["uncovered"]}
    assert gaps[ATT]["reason"] == gaps[rms]["reason"] == "omits verifier-evaluated parts"
    assert "row tail" in gaps[rms]["text"] and "softmax" in gaps[ATT]["text"]
    assert gaps["gumbel-top-p-token-select/v128256/fp32"]["reason"] == "omits verifier-evaluated parts"
