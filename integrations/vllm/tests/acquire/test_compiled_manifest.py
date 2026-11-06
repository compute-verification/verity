"""P_compiled x query required manifest (verity_vllm/acquire/compiled.py) -- torch-free.

The manifest is built from a process's Inductor output code (the wrappers' `Runner.call` launch programs + buffer allocations + the split
graph's piece wiring), never from what the collector emitted.  Pinned against a B1 workload's committed output code of BOTH arms
(`p_compiled_{control,treatment}_r1_output_code`): the same 17-piece / 354-per-step population, program digest (kernel set x launch
table x dataflow) and manifest digest the pod built
(compiled_manifest*.json, commit/summary.json program_digests) -- reproducible offline, and the treatment's own code derives the same Program.
"""
from __future__ import annotations

import json
import os

import pytest

from verity_vllm.acquire import compiled as CM
from verity_vllm.program.compiled_relations import OUTPUT_ROLES, roles_of

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-compiled"
ATT = os.path.join(ROOT, "out", "gen", "sweep", "evidence", "compiled", ROW, "r15_0100_b1_dataflow_row")
CODE = os.path.join(ATT, "p_compiled_control_r1_output_code")
CODE_T = os.path.join(ATT, "p_compiled_treatment_r1_output_code")
SG = os.path.join(ATT, "split_graph_control_r1", "rank_0_0", "backbone", "computation_graph.py")
SG_T = os.path.join(ATT, "split_graph_treatment_r1", "rank_0_0", "backbone", "computation_graph.py")
KSET = "d6b788f28401e9b9ef546160783b92c6a6286d636cf66a5b956a8a240206c8ff"

pytestmark = pytest.mark.skipif(not os.path.isdir(CODE), reason="row-of-record output code not present")


def _build(steps=32, prefill=256, code=CODE, sg=SG, v1=True):
    # v1 = the digest of the records built before weight routing entered program_digest; the pins below are v1 because that is the code
    # that ran them.  _build_v2 = the current default (weight routing in the digest).
    return CM.build(code, pieces=17, steps=steps, prefill_tokens=prefill, kernel_set_sha256=KSET, split_graph=sg, weights_in_dataflow=not v1)


def _build_v2(**kw):
    return _build(v1=False, **kw)


def test_f_pm_29_weight_routing_is_in_the_v2_program_digest(tmp_path):
    """Exchanging which served weight a norm launch consumes (post_attention_layernorm.weight <->
    the next input_layernorm.weight, both [2048] bf16) must change the Program: v2 digests differ, v1 (the recorded digest) does not."""
    import re, shutil
    v1, v2 = _build(steps=4, prefill=16), _build_v2(steps=4, prefill=16)
    assert v1["program_digest"] != v2["program_digest"]
    df2 = v2["program"]["dataflow"]
    assert df2["weights_declared"] is True and df2["digest_version"] == 2 and df2["n_weight_edges"] > 0 and df2["n_undefined"] == 0 and v2["complete"] is True
    # every weight edge names a served parameter through the split graph
    names = {e["src"].get("name") for rows in df2["edges"] for r in rows for e in r["edges"] if e.get("weight")}
    assert all(n for n in names) and any("layernorm" in n for n in names) and any("proj" in n for n in names), sorted(names)[:8]
    assert v1["populations"] == v2["populations"]                              # the required-value population is the same (identities carry
    assert [{k: r[k] for k in r if k != "program_digest"} for r in v1["identities"]] == [{k: r[k] for k in r if k != "program_digest"} for r in v2["identities"]]   # the Program, so manifest_digest differs)
    # the mutant: swap the two norm launches' WEIGHT arguments (the positions compiled_relations declares as weights) in every wrapper
    from verity_vllm.program.compiled_relations import roles_of
    k0, k2 = "triton_red_fused_fused_add_rms_norm_t_0", "triton_red_fused_fused_add_rms_norm_t_2"
    (w0pos,), (w2pos,) = sorted(roles_of(k0)[1]), sorted(roles_of(k2)[1])
    mut = tmp_path / "code"; shutil.copytree(CODE, mut)
    swapped = 0
    for f in mut.rglob("*.py"):
        src = f.read_text()
        L0 = re.findall(rf"{k0}\.run\(([^\n]*)\)", src); L2 = re.findall(rf"{k2}\.run\(([^\n]*)\)", src)
        if not (L0 and L2):
            continue
        a0 = [x.strip() for x in L0[0].split(",")]; a2 = [x.strip() for x in L2[0].split(",")]
        w0, w2 = a0[w0pos], a2[w2pos]
        assert re.fullmatch(r"arg\d+_1", w0) and re.fullmatch(r"arg\d+_1", w2) and w0 != w2, (a0, a2)
        def _swap(line: str, a: str, b: str) -> str:
            parts = [x.strip() for x in line.split(",")]
            return ", ".join(b if x == a else a if x == b else x for x in parts)
        new = re.sub(rf"({k0}\.run\()([^\n]*)(\))", lambda m: m.group(1) + _swap(m.group(2), w0, w2) + m.group(3), src)
        new = re.sub(rf"({k2}\.run\()([^\n]*)(\))", lambda m: m.group(1) + _swap(m.group(2), w0, w2) + m.group(3), new)
        assert new != src
        f.write_text(new); swapped += 1
    assert swapped >= 1
    m1, m2_ = _build(steps=4, prefill=16, code=str(mut)), _build_v2(steps=4, prefill=16, code=str(mut))
    assert m1["program_digest"] == v1["program_digest"]                        # v1 cannot see the swap
    assert m2_["program_digest"] != v2["program_digest"]                       # v2 does


def test_manifest_reproduces_the_row_of_record_digests():
    m = _build()
    rec = json.load(open(os.path.join(ATT, "compiled_manifest.json")))
    assert m["program_digest"] == rec["program_digest"]
    # the population is the record's row for row; the digest formula is the consumer's (required_manifest.manifest_digest;
    # binding.load_required_manifest recomputes it over the rows) -- the record's own field predates that formula
    from verity_vllm.query.manifest import format as RM
    assert m["schema"] == RM.SCHEMA and m["manifest_digest"] == RM.manifest_digest(rec["identities"]) == RM.manifest_digest(m["identities"])
    assert [RM.canonical_identity(r) for r in m["identities"]] == [RM.canonical_identity(r) for r in rec["identities"]]
    s = json.load(open(os.path.join(ATT, "commit", "summary.json")))
    assert s["program_digests"]["equal"] is True and s["program_digests"]["control"] == m["program_digest"]
    assert m["program"]["dataflow"]["wiring_declared"] is True and m["program"]["dataflow"]["n_undefined"] == 0
    assert m["complete"] is True and m["unmodelled"] == {}
    assert m["populations"] == {"identities": 11328, "by_family": {"compiled_kernel_buffers": 11328}, "steps": 32, "per_step": 354}
    lt = m["program"]["launch_table"]
    assert [L["kernel"] for L in lt["first"]] == ["triton_red_fused__to_copy_embedding_rms_norm_t_0", "matmul_kernel_persistent_0", "triton_poi_fused_1"]
    assert len(lt["middle"]) == 8 and len(lt["last"]) == 6 and lt["last"][-1]["kernel"] == "triton_red_fused_fused_add_rms_norm_2"


def test_treatment_output_code_derives_the_same_program():
    """The treatment's wrappers carry Inductor's debug-printer statements (the seam the hooks ride) -- not launches; the
    Program derived from them (kernel set x launch table x dataflow) is the control's, so the treatment executed the Program of record."""
    c, t = _build(steps=4, prefill=16), _build(steps=4, prefill=16, code=CODE_T, sg=SG_T)
    assert c["program_digest"] == t["program_digest"] and c["manifest_digest"] == t["manifest_digest"]
    rec_t = json.load(open(os.path.join(ATT, "compiled_manifest_treatment.json")))
    assert _build(code=CODE_T, sg=SG_T)["program_digest"] == rec_t["program_digest"] == _build()["program_digest"]
    # and the consumer's own integrity checks admit it (schema, digest recomputed over the rows, no zero-width identity)
    import tempfile
    from verity_vllm.commit import binding as B
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(t, f)
    pop, hdr = B.load_required_manifest(f.name, t["program_digest"], requests=["r0"], steps=4, ranks=[0], arrivals={"r0": 0})
    assert hdr["manifest_digest_verified"] is True and hdr["dataflow_in_digest"] is True and len(pop) == t["populations"]["identities"]
    # the wrapper TEXT does differ between the arms (the seam), which is why the merge compares derived Programs, not call text
    wc = sorted(p for p in __import__("glob").glob(os.path.join(CODE, "*", "*.py")) if "def call(" in open(p).read())
    wt = sorted(p for p in __import__("glob").glob(os.path.join(CODE_T, "*", "*.py")) if "def call(" in open(p).read())
    assert len(wc) == len(wt) > 0
    assert any("_hook" in open(p).read() or "intermediate" in open(p).read() for p in wt)


def test_identities_are_program_x_query_with_widths_from_the_wrappers():
    m = _build()
    ids = m["identities"]
    step0 = [r for r in ids if r["step"] == 0]
    step1 = [r for r in ids if r["step"] == 1]
    assert len(step0) == len(step1) == 354                                  # the same launch structure every step
    by = {(r["op_path"], r["output_member"]): r for r in step0}
    # prefill: 256 tokens x width; decode: 1 token x width -- the GEMM 'a' of the down_proj is [tokens, 8192] bf16
    a0 = by[("c01.matmul_kernel_persistent_2", "a")]["element_range"]
    a1 = {(r["op_path"], r["output_member"]): r for r in step1}[("c01.matmul_kernel_persistent_2", "a")]["element_range"]
    assert a0 == [0, 256 * 8192] and a1 == [0, 8192]
    assert by[("c00.triton_red_fused__to_copy_embedding_rms_norm_t_0", "ids")]["element_range"] == [0, 256]
    assert by[("c00.triton_red_fused__to_copy_embedding_rms_norm_t_0", "ids")]["kind"] == "input"
    assert by[("c16.triton_red_fused_fused_add_rms_norm_2", "out")]["kind"] == "output"
    # every identity's role is a registered role of its kernel (the manifest's role map, resolved from the generated kernel's signature --
    # `triton_poi_fused_1` names both the Llama RoPE and the Qwen RoPE+bias, so the name alone is unmodelled); weights are never identities
    rm = m["program"]["role_maps"]
    for r in ids[:400]:
        piece, kernel = r["op_path"].split(".", 1)
        kind = "first" if piece == "c00" else ("last" if piece == f"c{m['program']['pieces'] - 1:02d}" else "middle")
        e = next(x for x in rm[kind] if x["kernel"] == kernel)
        assert r["output_member"] in e["roles"].values() and r["output_member"] not in {e["roles"].get(str(i)) for i in e["weight_args"]}
        assert r["kind"] == ("output" if r["output_member"] in OUTPUT_ROLES else "input")
    assert roles_of("triton_poi_fused_1") == ({}, set())                                         # two arities share the name: no guess
    assert roles_of("triton_poi_fused_1", ["in_ptr0", "in_ptr1", "in_ptr2", "out_ptr0", "out_ptr1", "xnumel_0", "xnumel_1", "XBLOCK : tl.constexpr"])[0] == {0: "qkv", 1: "pos", 3: "k", 4: "q"}
    assert roles_of("triton_poi_fused_3", arity=6)[0] == {0: "qkv", 1: "bias", 2: "pos", 4: "k", 5: "qkv_b"}
    assert roles_of("triton_poi_fused_3", arity=7) == ({}, set())
    assert roles_of("matmul_kernel_persistent_2") == ({0: "a", 2: "out"}, {1})                  # one arity: the name suffices


def test_program_digest_is_the_artifact_not_the_workload():
    a, b = _build(steps=32, prefill=256), _build(steps=8, prefill=64)
    assert a["program_digest"] == b["program_digest"]                          # P_compiled = the compiled artifact's launch program + widths
    assert a["manifest_digest"] != b["manifest_digest"]                        # the query (steps, tokens) changes the population
    assert b["populations"]["identities"] == 8 * 354


# ---------------------------------------------------------------------------------------------- dataflow identity

def _copy_code(tmp_path, edit=None):
    """A copy of the row-of-record output code with an optional line edit applied to every wrapper (returns the copy's root)."""
    import shutil
    root = tmp_path / "code"
    shutil.copytree(CODE, root)
    if edit:
        for f in root.glob("*/*.py"):
            t = f.read_text()
            t2 = edit(t)
            if t2 != t:
                f.write_text(t2)
    return str(root)


def test_dataflow_edges_are_declared_and_complete():
    m = _build()
    df = m["program"]["dataflow"]
    assert df["n_edges"] > 0 and df["n_undefined"] == 0 and m["complete"] is True
    mid = df["edges"][1]                                                        # a middle piece: every launch's inputs have a named producer
    by = {r["kernel"]: {e["role"]: e["src"] for e in r["edges"]} for r in mid}
    assert by["triton_red_fused_fused_add_rms_norm_t_0"]["x0"]["kernel"] == "matmul_kernel_persistent_0"       # o_proj GEMM output
    assert by["triton_poi_fused_mul_silu_slice_t_1"]["x"]["kernel"] == "matmul_kernel_persistent_1"            # gate_up GEMM output
    assert by["matmul_kernel_persistent_2"]["a"]["kernel"] == "triton_poi_fused_mul_silu_slice_t_1"           # silu*up -> down GEMM
    assert by["triton_red_fused_fused_add_rms_norm_t_2"]["x1"]["kernel"] == "matmul_kernel_persistent_0"


def test_n1_rewiring_changes_the_program_digest(tmp_path):
    """Swap the two inputs of the second fused add+rmsnorm in the middle wrappers (which buffer is the residual) -> a different
    compiled implementation must have a different P_compiled identity."""
    swapped = _copy_code(tmp_path, lambda t: t.replace("triton_red_fused_fused_add_rms_norm_t_2.run(buf6, buf0, ", "triton_red_fused_fused_add_rms_norm_t_2.run(buf0, buf6, "))
    a = CM.build(CODE, pieces=17, steps=4, prefill_tokens=16, kernel_set_sha256=KSET, split_graph=SG)
    b = CM.build(swapped, pieces=17, steps=4, prefill_tokens=16, kernel_set_sha256=KSET, split_graph=SG)
    assert a["program_digest"] != b["program_digest"]
    assert a["populations"] == b["populations"]                                 # same population, different dataflow


def test_n3_unparsed_launch_is_refused(tmp_path):
    """A `.run(*args, ...)` line the parser cannot read is an undeclared part of the implementation -> refusal, not silence."""
    bad = _copy_code(tmp_path, lambda t: t.replace("triton_poi_fused_mul_silu_slice_t_1.run(buf3, buf7, ", "triton_poi_fused_mul_silu_slice_t_1.run(*args, ") if "def call(self, args)" in t else t)
    with pytest.raises(SystemExit, match="did not read"):
        CM.build(bad, pieces=17, steps=4, prefill_tokens=16, kernel_set_sha256=KSET, split_graph=SG)


def test_n4_kernel_set_is_required():
    with pytest.raises(SystemExit, match="kernel-set"):
        CM.build(CODE, pieces=17, steps=4, prefill_tokens=16)
