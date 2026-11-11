"""The required-value manifest FORMAT (`manifest_format`): name / spec parsing, the identity digest, the coverage check's negatives
the manifest makes detectable BY CONSTRUCTION (omit a whole producer family; omit one step; drop one member and bind another twice; a member
committed short) and the binding loader's two engine-step shapes -- over manifests of record built by the v2 query
(`test_sampled_replay.v2_manifest`) on a synthetic wired Program (no torch, no pod)."""
from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.commit import binding as B
from verity_vllm.query.manifest import format as MF
from verity_vllm.query.manifest.bind import coverage_check
from verity_vllm.query.manifest.format import canonical_identity, manifest_digest, spec_statics, split_name
from tests.check.test_sampled_replay import _program, v2_manifest

EPS = 'EPS={"f64":"0x1.4f8b588e368f1p-17"}'


def _layouts_from(man: dict) -> dict[int, list[dict]]:
    """A collector's per-step layouts that cover the manifest exactly (the positive case), in the committed-name vocabulary."""
    lay: dict[int, list[dict]] = {}
    for r in man["identities"]:
        if r["family"] == "weights":
            continue
        lay.setdefault(int(r["step"]), []).append({"name": f"{r['op_path']}/{r['output_member']}", "numel": r["element_range"][1], "inv": r["invocation"]})
    for step in lay:                     # committed order: a module's k-th call is the k-th occurrence in the step
        lay[step].sort(key=lambda m: (m["name"], m["inv"]))
    return lay


def test_parsing():
    assert spec_statics("Gemm_v1{K=576,N=960}") == {"K": 576, "N": 960}
    assert spec_statics(f"RMSNormFusedCuda_v2{{N=576,{EPS}}}")["N"] == 576
    assert split_name("model.model.layers.3.self_attn.rotary_emb/rotary_embedding_3/key[7]") == ("model.model.layers.3.self_attn.rotary_emb", "rotary_embedding_3", "key", 7)
    assert split_name("argmax_2[0]") == ("", "argmax_2", "", 0)


def test_digest_determinism():
    inst, _woff = _program()
    man = v2_manifest(inst)
    assert man["complete"] is True and man["steps"] == [0, 1, 2] and man["engine_steps"] == [0, 1, 2]
    assert v2_manifest(inst)["manifest_digest"] == man["manifest_digest"]                     # rebuild -> same
    ids = copy.deepcopy(man["identities"])
    for r in ids:                                                                             # annotations do not enter
        r.pop("geometry", None)
        r.pop("rows", None)
    assert manifest_digest(ids) == man["manifest_digest"]
    assert v2_manifest(inst, digest="q" * 64)["manifest_digest"] != man["manifest_digest"]   # the Program digest does
    assert len(canonical_identity(man["identities"][0])) == 8


def test_coverage_positive_and_negatives():
    inst, _woff = _program()
    man = v2_manifest(inst)
    lay = _layouts_from(man)
    ok = coverage_check(man, lay)
    assert ok["ok"], ok["missing"][:3]
    checked = [r for r in man["identities"] if r["family"] != "weights"]
    assert ok["committed_unmatched_n"] == 0 and ok["checked"] == len(checked)

    # (1) omit a WHOLE producer family: the sampler never committed -> coverage failure naming the family
    sampled = {f"{r['op_path']}/{r['output_member']}" for r in man["identities"] if r["family"] == "sampled_token_ids"}
    lay1 = {s: [m for m in ms if m["name"] not in sampled] for s, ms in lay.items()}
    r1 = coverage_check(man, lay1)
    assert not r1["ok"] and r1["families_missing_entirely"] == ["sampled_token_ids"] and r1["missing_by_family"] == {"sampled_token_ids": len(man["steps"])}

    # (2) omit one step (the last decode step never committed) -> coverage failure on exactly that step
    last = man["steps"][-1]
    r2 = coverage_check(man, {s: ms for s, ms in lay.items() if s != last})
    assert not r2["ok"] and r2["steps_with_missing"] == [last] and r2["missing_n"] == sum(1 for r in checked if r["step"] == last)

    # (3) drop one output member of a two-member module and commit the other twice: same count, same module set -> coverage failure on
    # it.  The members are the fused norm's ports `0` / `1`, committed as tuple slots: one slot for two members places neither
    # (`verity_vllm.query.manifest.bind`), so both are missing and both tensors are extras
    two = next((s, op, sorted(ms)) for (s, op), ms in _members_by_step_op(checked).items() if len(ms) >= 2)
    s3, op3, (keep, drop) = two[0], two[1], two[2][:2]
    lay3 = copy.deepcopy(lay)
    k = next(i for i, m in enumerate(lay3[s3]) if m["name"] == f"{op3}/{drop}")
    lay3[s3][k] = dict(lay3[s3][k], name=f"{op3}/{keep}")
    r3 = coverage_check(man, lay3)
    assert not r3["ok"] and r3["missing_n"] == 2 and {(m["output_member"], m["step"]) for m in r3["missing"]} == {(keep, s3), (drop, s3)}
    assert r3["committed_extras_n"] == 2 and r3["committed_unmatched_n"] == 0

    # (4) a member committed SHORT (wrong element range: one row fewer) -> coverage failure (containment)
    tall = next(r for r in checked if r["family"] == "instance_outputs" and int(r.get("rows") or 1) > 1)
    lay4 = copy.deepcopy(lay)
    m = next(m for m in lay4[int(tall["step"])] if m["name"] == f"{tall['op_path']}/{tall['output_member']}")
    m["numel"] -= m["numel"] // int(tall["rows"])
    r4 = coverage_check(man, lay4)
    assert not r4["ok"] and r4["missing"][0]["op_path"] == tall["op_path"]


def test_a_batch_hidden_row_is_covered_by_the_planes_no_manifest_row_spells():
    """A B>1 batch-level hidden row (request `*`, one FA2 launch per layer per engine step) names the padded whole-step geometry.  The
    tap commits per-sequence planes at a segmented prefill and one whole launch at decode, under spellings no row has, so the binder
    binds none of them; coverage holds them to the launch's whole stream: one plane set per executing sequence at its scheduled rows, every tensor the words its name's geometry declares."""
    rows_of = {0: {"r0": 64, "r1": 48}, 1: {"r0": 1, "r1": 1}}
    geo = {0: (8, 64, 1, 128, 64), 1: (8, 1, 1, 128, 64)}                     # fa2_geometry: (HB, M, NB, BN, D), 4 heads a sequence
    rows = [{"family": "fa2_hidden_m1_stream", "request_id": "*", "requests": ["r0", "r1"], "rank": 0, "step": "*", "engine_step": s,
             "op_path": "fa2.m1/L0", "output_member": f"g{geo[s][0]}x{geo[s][1]}x1", "invocation": 0, "element_range": [0, MF.mat_total_words(*geo[s])],
             "geometry": dict(zip(("HB", "M", "NB", "BN", "D"), geo[s]), seqlen_k=64 + s, rows=geo[s][1] * 2, n_seq=2)} for s in (0, 1)]
    man = {"complete": True, "manifest_digest": "m" * 64, "identities": rows,
           "engine_steps": [{"engine_step": s, "members": [{"request_id": q, "rows": n} for q, n in rows_of[s].items()], "rows": sum(rows_of[s].values()),
                             "n_seq": 2} for s in (0, 1)]}
    widths = MF.hidden_stream_widths(1, 128, 64)
    lay = {0: [{"name": f"fa2.m1/L0/s{i}/g4x{sl}x1/{c}", "numel": 4 * sl * w} for i, sl in enumerate(rows_of[0].values()) for c, w in widths.items()],
           1: [{"name": "fa2.m1/L0/g8x1x1", "numel": MF.mat_total_words(*geo[1])}]}
    cov = coverage_check(man, lay)
    assert cov["ok"] and cov["missing_n"] == 0, cov["missing"]
    assert cov["committed_extras_n"] == 0 and cov["committed_unmatched_n"] == 0
    dropped = coverage_check(man, {0: [m for m in lay[0] if "/s1/" not in m["name"]], 1: lay[1]})
    assert not dropped["ok"] and dropped["missing_by_family"] == {"fa2_hidden_m1_stream": 1} and dropped["missing"][0]["engine_step"] == 0
    assert "the planes of sequences ['s0'] committed, 2 sequence(s) executing" in dropped["missing"][0]["why"]
    one_short = copy.deepcopy(lay)
    one_short[1][0]["numel"] -= 1
    short = coverage_check(man, one_short)
    assert not short["ok"] and short["missing"][0]["engine_step"] == 1 and "words committed, its geometry declares" in short["missing"][0]["why"]


def _seq_planes(rows: list[int], *, h: int = 4, nb: int = 2, ms: bool = False) -> list[tuple[str, int]]:
    return [(f"s{i}/g{h}x{sl}x{nb}/{c}", h * sl * w) for i, sl in enumerate(rows) for c, w in MF.hidden_stream_widths(nb, 128, 64, ms).items()]


@pytest.mark.parametrize("active, members, why", [
    ([3, 1], _seq_planes([3, 1]), None),
    ([3, 1], _seq_planes([3, 1], ms=True), "s0 carries"),                                   # MS without the guarded-max policy
    ([3], _seq_planes([3], nb=1), None),                                                     # one of two executing: its key blocks may be fewer
    ([3], _seq_planes([3], nb=3), "NB 3"),                                                   # ... never more than the launch declares
    ([3, 1], _seq_planes([3, 1], nb=1), "NB 1"),                                             # every declared sequence executing: the launch's NB
    ([3, 1], _seq_planes([3, 1]) + [("g8x3x2", MF.mat_total_words(8, 3, 2, 128, 64))], "beside the per-sequence planes"),
    ([3, 1], _seq_planes([3, 1]) + _seq_planes([3])[:1], "s0/g4x3x2/S committed twice"),
    ([3, 1], _seq_planes([3, 1]) + [("s0/g4x3x2/bot", 1)], None),                           # a ⊥ padding plane is not the launch's stream
    ([3, 1], [("g8x3x2", MF.mat_total_words(8, 3, 2, 128, 64))], None),
    ([3, 1], [("g8x12x2", MF.mat_total_words(8, 12, 2, 128, 64))], None),                # FA3's rows with its KV group's heads, an old manifest's M without
    ([3, 1], [("g8x2x2", MF.mat_total_words(8, 2, 2, 128, 64))], "2 executing sequence(s) of this launch stage g8x>=3x2"),
    ([3], [("g4x3x2", MF.mat_total_words(4, 3, 2, 128, 64))], None),
    ([3], [("g8x3x2", MF.mat_total_words(8, 3, 2, 128, 64))], "1 executing sequence(s) of this launch stage g4x>=3x<=2")])
def test_batch_launch_shortfall_names_what_the_committed_stream_lacks(active, members, why):
    """`format.batch_launch_shortfall` over a two-sequence FA2 launch (4 heads a sequence, NB 2, the legacy tile): None when the committed
    tensors are its whole stream at the executing sequences' rows, else the shortfall named."""
    geometry = {"HB": 8, "M": 3, "NB": 2, "BN": 128, "D": 64, "n_seq": 2}
    got = MF.batch_launch_shortfall(geometry, active, members)
    assert (got is None) if why is None else (why in (got or "")), got


def _members_by_step_op(ids: list[dict]) -> dict[tuple[int, str], set[str]]:
    out: dict[tuple[int, str], set[str]] = {}
    for r in ids:
        if r["family"] == "instance_outputs" and int(r["invocation"]) == 0:
            out.setdefault((int(r["step"]), r["op_path"]), set()).add(r["output_member"])
    return out


def test_fresh_b1_manifest_loads_in_binding_and_composition_skips(tmp_path):
    """A B=1 manifest writes engine_steps as the request's engine-step NUMBERS (ints); binding.load_required_manifest
    keeps them as numbers and declares no composition (composition_check skips, named).  A B>=2 GLOBAL manifest declares the dict
    composition and the loader keeps it; a manifest mixing the two shapes is refused by name."""
    inst, _woff = _program()
    man = v2_manifest(inst)
    assert man["engine_steps"] and all(isinstance(e, int) for e in man["engine_steps"]), man["engine_steps"]
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(man))
    pop, hdr = B.load_required_manifest(str(p), man["program_digest"], requests=["r0"], steps=len(man["steps"]), ranks=[0], arrivals={"r0": 0})
    assert len(pop) > 0 and hdr["engine_steps"] == [] and hdr["engine_step_numbers"] == man["engine_steps"] and hdr["global"] is False
    comp = B.composition_check(hdr, {"entries": []})
    assert comp["ok"] and comp["checked_steps"] == 0 and "no engine-step composition" in comp["skipped"]
    g = v2_manifest(inst, ("r0", "r1"))
    pg = tmp_path / "global.json"
    pg.write_text(json.dumps(g))
    _popg, hdrg = B.load_required_manifest(str(pg), g["program_digest"], requests=["r0", "r1"], steps=len(man["steps"]), ranks=[0], arrivals={"r0": 0, "r1": 0})
    assert hdrg["global"] and len(hdrg["engine_steps"]) == len(g["engine_steps"]) and hdrg["engine_step_numbers"] == [e["engine_step"] for e in g["engine_steps"]]
    assert all(isinstance(e["members"], list) and e["rows"] is not None for e in hdrg["engine_steps"])
    bad = dict(man, engine_steps=[0, {"engine_step": 1, "members": []}])
    pb = tmp_path / "bad.json"
    pb.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="engine_steps mixes"):
        B.load_required_manifest(str(pb), man["program_digest"], requests=["r0"], steps=2, ranks=[0], arrivals={"r0": 0})
