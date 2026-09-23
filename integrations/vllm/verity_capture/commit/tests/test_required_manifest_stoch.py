"""[R15 stoch] The required-value manifest of a STOCHASTIC request Program (seeded temperature + top-p): the sampling event is
`GumbelTopPTokenSelect_v1{V}` (manbind's `is_token_select` / `is_stochastic_select`, M-0107 / M-0349), so every step requires the
sampled id AND the consumed seed (`runner.sampler/seed`, family `sampler_seed`); the Program's constants (`Const32[..]_v1` /
`Const64[..]_v1`: temperature / top_p / position written in at Build) bind nothing and do not make the manifest incomplete.  The
binding-map structure grows the `seed` member under `runner.sampler` for a stochastic workload only (commit_delta.binding_record)."""
from __future__ import annotations

import copy
import types

from verity_capture.commit import binding as B
from verity_capture.commit.required_manifest import build_manifest, coverage_check, is_stochastic_select, is_token_select, segment_steps

from .test_required_manifest import _layouts_from, _program


def _stochastic_program(**kw) -> dict:
    """The greedy synthetic Program with its `TokenSelect_v1{V}` rows replaced by the derived stochastic select and its per-row constants
    (`vllm_sampling.GumbelTopPTokenSelectRule` emits Const32[top_p], Const64[pos], Const32[temperature] then the select, per sampled row)."""
    P = copy.deepcopy(_program(**kw))
    rows = []
    for r in P["instances"]["rows"]:
        if r["spec"].startswith("TokenSelect_v1"):
            V = r["spec"].split("V=", 1)[1].rstrip("}")
            base = r["name"].split("[", 1)[0]
            for spec, nm in ((f"Const32[0x3f733333]_v1", f"{base}/top_p"), (f"Const64[0x0000000000000003]_v1", f"{base}/pos"),
                             (f"Const32[0x3f4ccccd]_v1", f"{base}/temperature")):
                rows.append({"i": len(rows), "spec": spec, "form": "call", "name": nm, "args": []})
            rows.append(dict(r, i=len(rows), spec=f"GumbelTopPTokenSelect_v1{{V={V}}}", name=f"{base}/gumbel_topp_token_select[0]"))
        else:
            rows.append(dict(r, i=len(rows)))
    P["instances"]["rows"] = rows
    # [R16 stoch, coord M-0169] the request-owned derived-S Input `splits` (int32, one entry per sampling event) follows the seed
    n_events = sum(1 for r in rows if r["spec"].startswith("GumbelTopPTokenSelect_v1"))
    P["instances"]["params"] = [P["instances"]["params"][0], ["seed", "Array<1,Value<64>>"], ["splits", f"Array<{n_events},Value<32>>"],
                                P["instances"]["params"][1]]
    return P


def test_stochastic_select_is_the_step_boundary_and_requires_the_seed():
    assert is_token_select("GumbelTopPTokenSelect_v1") and is_stochastic_select("GumbelTopPTokenSelect_v1")
    assert is_token_select("TokenSelect_v1") and not is_stochastic_select("TokenSelect_v1")
    P = _stochastic_program(T=2)
    G = _program(T=2)
    assert len(segment_steps(P["instances"]["rows"])) == len(segment_steps(G["instances"]["rows"])) == 3
    man = build_manifest(P)
    assert man["complete"] is True and man["unmodelled"] == {}, man["unmodelled"]
    fam = man["populations"]["by_family"]
    assert fam["sampler_seed"] == 3 and fam["sampled_token_ids"] == 3, fam
    seeds = [r for r in man["identities"] if r["family"] == "sampler_seed"]
    assert all(r["op_path"] == "runner.sampler" and r["output_member"] == "seed" and r["dtype"] == "i64" and r["spec"] == "GumbelTopPTokenSelect_v1" for r in seeds)
    assert sorted(r["step"] for r in seeds) == [0, 1, 2]
    assert "sampler_seed" in man["query"]["required_families"] and man["query"]["stochastic_selectors"] == ["GumbelTopPTokenSelect_v1"]
    # the greedy Program of the same shape differs ONLY by the seed + splits rows (constants bind nothing)  [R16 stoch REQ-03: +3 splits]
    gman = build_manifest(G)
    assert man["populations"]["identities"] == gman["populations"]["identities"] + 3 + 3
    assert man["manifest_digest"] != gman["manifest_digest"]


def _static_s_program(**kw) -> dict:
    """The R15 static-S stochastic Program: `GumbelTopPTokenSelect_v1{V,S=16}` and NO `splits` param (seed only)."""
    P = _stochastic_program(**kw)
    for r in P["instances"]["rows"]:
        if r["spec"].startswith("GumbelTopPTokenSelect_v1"):
            r["spec"] = r["spec"][:-1] + ",S=16}"
    P["instances"]["params"] = [p for p in P["instances"]["params"] if p[0] != "splits"]
    return P


def test_derived_s_program_requires_the_sampler_splits_per_step_static_s_does_not():
    """[R16 stoch REQ-03 / reqval M-0305] the derived-S Program declares `splits: Array<T+1, Value<32>>` (request-owned input; entry s = the
    split count of sampling event s) -> every step requires `runner.sampler/splits` (family `sampler_splits`, i32, one per sequence) beside
    the seed; the R15 static-S Program (`{V,S}`, no `splits` param) is unchanged: no such family, no such rows."""
    P = _stochastic_program(T=2)
    man = build_manifest(P)
    assert man["complete"] is True and man["unmodelled"] == {}, man["unmodelled"]
    fam = man["populations"]["by_family"]
    assert fam["sampler_splits"] == fam["sampler_seed"] == fam["sampled_token_ids"] == 3, fam
    rows = [r for r in man["identities"] if r["family"] == "sampler_splits"]
    assert all(r["op_path"] == "runner.sampler" and r["output_member"] == "splits" and r["dtype"] == "i32" and r["spec"] == "GumbelTopPTokenSelect_v1"
               and r["element_range"] == [0, 1] for r in rows), rows
    assert sorted(r["step"] for r in rows) == [0, 1, 2]
    assert "sampler_splits" in man["query"]["required_families"]
    assert "SplitsFor_v1" in man["query"]["why"]["sampler_splits"] and "F-stoch-01" in man["query"]["why"]["sampler_splits"]
    # the R15 static-S Program: seed rows only, and the manifest stays complete
    sman = build_manifest(_static_s_program(T=2))
    assert sman["complete"] is True
    assert "sampler_splits" not in sman["populations"]["by_family"] and "sampler_splits" not in sman["query"]["required_families"]
    assert sman["populations"]["by_family"]["sampler_seed"] == 3
    assert man["manifest_digest"] != sman["manifest_digest"]


def test_a_splits_param_no_stochastic_selector_reads_emits_no_family():
    """A Program that declares `splits` with a GREEDY selector: the inventory emits no `sampler_splits` (the family is the stochastic
    selector's); the input nothing names is the traversal engine's `unnamed_input` finding (reqval 16922fc2e), not the inventory's."""
    P = _stochastic_program(T=1)
    for r in P["instances"]["rows"]:
        if r["spec"].startswith("GumbelTopPTokenSelect_v1"):
            r["spec"] = "TokenSelect_v1" + r["spec"][len("GumbelTopPTokenSelect_v1"):]
    man = build_manifest(P)
    assert "sampler_splits" not in man["populations"]["by_family"] and "sampler_seed" not in man["populations"]["by_family"]
    assert "sampler_splits" not in man["query"]["required_families"]


def test_coverage_needs_the_bound_splits_per_step():
    """[REQ-03] the family without its collector binding fails C1 coverage closed at every step; bound at all but one step names the step."""
    man = build_manifest(_stochastic_program(T=2))
    lay = _layouts_from(man)
    assert coverage_check(man, lay)["ok"]
    lay2 = {s: [e for e in ents if e["name"] != "runner.sampler/splits"] for s, ents in lay.items()}
    r = coverage_check(man, lay2)
    assert not r["ok"] and r["families_missing_entirely"] == ["sampler_splits"] and r["missing_n"] == 3, r
    lay3 = {s: [e for e in ents if not (s == 2 and e["name"] == "runner.sampler/splits")] for s, ents in lay.items()}
    r3 = coverage_check(man, lay3)
    assert not r3["ok"] and r3["missing_n"] == 1 and r3["missing"][0]["step"] == 2 and r3["missing"][0]["output_member"] == "splits"


def test_coverage_needs_the_bound_seed_per_step():
    man = build_manifest(_stochastic_program(T=2))
    lay = _layouts_from(man)
    assert coverage_check(man, lay)["ok"]
    # a collector that binds the sampled ids but never the seed (the greedy collector) fails coverage by name, every step
    lay2 = {s: [e for e in ents if e["name"] != "runner.sampler/seed"] for s, ents in lay.items()}
    r = coverage_check(man, lay2)
    assert not r["ok"] and r["families_missing_entirely"] == ["sampler_seed"] and r["missing_n"] == 3, r
    # the seed bound at all but one step -> that step is named
    lay3 = {s: [e for e in ents if not (s == 1 and e["name"] == "runner.sampler/seed")] for s, ents in lay.items()}
    r3 = coverage_check(man, lay3)
    assert not r3["ok"] and r3["missing_n"] == 1 and r3["missing"][0]["step"] == 1 and r3["missing"][0]["output_member"] == "seed"


def test_binding_structure_has_the_seed_member_for_stochastic_workloads_only():
    com = types.SimpleNamespace(_modules=[], _module_class=lambda n: "Linear", runner=types.SimpleNamespace(sample=lambda *a: None), sources=[], weights=None)
    greedy = B.structure_of(com)
    stoch = B.structure_of(com, stochastic=True)
    g = next(d for d in greedy["modules"] if d["op_path"] == "runner.sampler")
    s = next(d for d in stoch["modules"] if d["op_path"] == "runner.sampler")
    assert g["members"] == ["sampled_token_ids"] and s["members"] == ["sampled_token_ids", "seed"]
    wl = {"program_digest": "p" * 64, "requests": ["r0"], "steps": 2, "ranks": [0]}
    pop_g, pop_s = list(B.expected_population(greedy, wl)), list(B.expected_population(stoch, wl))
    assert len(pop_s) == len(pop_g) + 2
    assert any(i[4] == "runner.sampler" and i[5] == "seed" for i in pop_s) and not any(i[5] == "seed" for i in pop_g)
    # [R16 stoch REQ-03] the derived-S row grows the `splits` member too; a greedy workload never does, whatever the flag says
    d = next(m for m in B.structure_of(com, stochastic=True, derived_splits=True)["modules"] if m["op_path"] == "runner.sampler")
    assert d["members"] == ["sampled_token_ids", "seed", "splits"]
    gd = next(m for m in B.structure_of(com, stochastic=False, derived_splits=True)["modules"] if m["op_path"] == "runner.sampler")
    assert gd["members"] == ["sampled_token_ids"]
