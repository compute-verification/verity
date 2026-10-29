"""END TO END: the sampled replay over a Program whose rotary rows live under a per-layer path that is an
ALIAS of the shared module's canonical identity (vLLM's one RotaryEmbedding instance: the binding map holds
`model.layers.0.self_attn.rotary_emb` at invocation = layer; the manifest records `model.layers.1.self_attn.rotary_emb -> [.., 1]`).

  * WITHOUT the manifest (the pre-fix reading, and any caller that passes none): every VU that reads or produces the aliased identity is
    'no committed identity for (.., model.layers.1.self_attn.rotary_emb/..)' BY NAME -- the failing record, reproduced: partial, never PASS;
  * WITH the manifest: the replay is complete and bit-exact (the RoPE rows' own outputs and, where the tree registers it, the Attention_v3
    row's q / K-history inputs are read at invocation 1 of the canonical path);
  * an alias table that points at the WRONG invocation (0, where nothing of layer 1 is bound... but layer 0's words would be) never reads
    a neighbour's words: the identity is missing by name (invocation 0 is not bound under this world) -- fail closed.
"""
from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest

from verity_vllm.check.replay import driver as SR, index as IX, opening as OP
from verity.ml.definitions.numerics import rows as RK
from tests.check.test_sampled_replay import Honest, _bf16, _program, _programs, _weights

# needs the rope / attention test Program (`_program(rope=...)`); a tree without it cannot run this end to end
pytestmark = pytest.mark.skipif("rope" not in inspect.signature(_program).parameters,
                                reason="dense's rope/attention test Program is not on this tree; run on the int line")

CANON = "model.layers.0.self_attn.rotary_emb"
ALIAS = "model.layers.1.self_attn.rotary_emb"


def _aliased_world():
    """The rope (+ attention where registered) Program with its rotary rows named under LAYER 1's module path; the committed identities of
    those rows are re-addressed to the canonical path at invocation 1, exactly as the required manifest binds a shared module."""
    attn = "Attention_v3" in RK.EVALUATORS
    inst, woff = _program(attn=True) if attn else _program(rope=True)
    inst = copy.deepcopy(inst)
    n_ren = 0
    for r in inst["rows"]:
        if "model.model.layers.0.self_attn.rotary_emb/" in r["name"]:
            r["name"] = r["name"].replace("model.model.layers.0.self_attn.rotary_emb/", "model.model.layers.1.self_attn.rotary_emb/")
            n_ren += 1
    assert n_ren > 0
    rng = np.random.default_rng(23)
    n_w = max(hi for lo, hi in woff.values())
    weights = _bf16(rng, n_w, 0.5)
    weights[woff["model.model.layers.0.input_layernorm.weight"][0]:woff["model.model.layers.0.input_layernorm.weight"][1]] = 0x3F80
    weights[woff["model.model.norm.weight"][0]:woff["model.model.norm.weight"][1]] = 0x3F80
    lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
    weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    world = Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})
    bmap, committed = world.bmap_and_reader()
    n_re = 0
    for e in bmap["entries"]:
        if e["identity"]["op_path"] == ALIAS:
            e["identity"]["op_path"], e["identity"]["invocation"] = CANON, 1
            n_re += 1
    assert n_re > 0 and not any(e["identity"]["op_path"] == ALIAS for e in bmap["entries"])
    fam_of = lambda e: ("input_token_ids" if e["identity"]["output_member"] == "in" else                                     # noqa: E731
                        ("sampled_token_ids" if e["identity"]["op_path"] == IX.SAMPLER_OP else e["family"]))
    ids = [{"family": fam_of(e), "request_id": e["identity"]["request_id"], "step": e["identity"]["step"], "op_path": e["identity"]["op_path"],
            "output_member": e["identity"]["output_member"], "invocation": e["identity"]["invocation"], "element_range": e["identity"]["element_range"],
            "rank": 0} for e in bmap["entries"]]
    manifest = {"query": {"id": "Q_module_v1"}, "op_path_aliases": {ALIAS: [CANON, 1]}, "identities": ids}
    return world, bmap, committed, manifest, attn


def test_without_the_alias_table_the_aliased_identities_are_missing_by_name_and_the_replay_is_partial():
    world, bmap, committed, _manifest, attn = _aliased_world()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=5)
    assert rep["result"] is None and rep["grade"] == "partial", json.dumps(rep, indent=1, default=str)[:2000]
    ne = rep["not_evaluated"]
    assert ne and all(x["why_class"] == "missing-input" for x in ne), ne[:3]
    whys = " ".join(str(x.get("why") or "") for x in ne)
    assert f"no committed identity for (r0, step 0, {ALIAS}/" in whys or f"no committed identity for (r1, step 0, {ALIAS}/" in whys, whys[:1500]
    assert "invocation" not in whys                                                # no alias table: nothing resolved, nothing collapsed
    fams = {x["spec"].split("{")[0] for x in ne}
    assert fams == ({"Attention_v3", "RoPE_v1"} if attn else {"RoPE_v1"}), fams  # the rows that PRODUCE and the rows that READ the alias
    assert any(k.startswith("RoPE_v1") for k in rep["strata_unevaluated"])


def test_with_the_manifest_alias_table_the_replay_reads_invocation_one_of_the_canonical_path_and_is_complete():
    world, bmap, committed, manifest, attn = _aliased_world()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=5, manifest=manifest)
    assert rep["result"] is True and rep["grade"] == "complete", json.dumps(rep, indent=1, default=str)[:3000]
    assert rep["sample"]["mismatch_n"] == 0 and rep["sample"]["evaluated"] == rep["sample"]["picked"]
    rope = {k: v for k, v in rep["strata"].items() if k.startswith("RoPE_v1")}
    assert rope and all(v["evaluated"] >= 1 and v["evaluated"] == v["sampled"] for v in rope.values()) and all(f"|{ALIAS}|" in k for k in rope), sorted(rope)
    if attn:
        at = {k: v for k, v in rep["strata"].items() if k.startswith("Attention_v3")}
        assert at and all(v["evaluated"] >= 1 and v["evaluated"] == v["sampled"] for v in at.values()), at
    pop = rep["population"]
    assert pop["query_checks"]["identities_without_rows"]["n"] == 0 and pop["query_checks"]["vus_outside_query"]["n"] == 0, pop["query_checks"]


@pytest.mark.slow
def test_a_wrong_invocation_in_the_alias_table_never_reads_a_neighbours_words():
    world, bmap, committed, manifest, attn = _aliased_world()
    # (a) the alias points at invocation 0 -- nothing of the shared module is bound there in this world: missing by name, not layer 1's words
    bad = dict(manifest, op_path_aliases={ALIAS: [CANON, 0]})
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=5, manifest=bad)
    assert rep["result"] is None and rep["grade"] == "partial"
    whys = " ".join(str(x.get("why") or "") for x in rep["not_evaluated"])
    assert f"[alias -> {CANON} invocation 0]" in whys and "bound invocations" in whys and "[1]" in whys, whys[:1500]
    # (b) a second module bound at invocation 0 under the canonical path with DIFFERENT words: layer 1 still reads invocation 1 exactly
    bmap2 = copy.deepcopy(bmap)
    extra = []
    for e in bmap2["entries"]:
        if e["identity"]["op_path"] == CANON and e["identity"]["invocation"] == 1:
            e0 = copy.deepcopy(e)
            e0["identity"]["invocation"] = 0
            e0["name"] = e["name"] + "#inv0"
            extra.append(e0)
    bmap2["entries"].extend(extra)
    ids0 = [dict(i, invocation=0) for i in manifest["identities"] if i["op_path"] == CANON]
    man2 = dict(manifest, identities=manifest["identities"] + ids0)

    def committed2(step, name, lo, hi):
        if name.endswith("#inv0"):
            b = committed(step, name[: -len("#inv0")], lo, hi)
            return None if b is None else bytes(x ^ 0x5A for x in b)               # invocation 0 = layer 0's DIFFERENT words
        return committed(step, name, lo, hi)

    rep2 = SR.sampled_replay(_programs(world), bmap2, committed2, _weights(world), per_stratum=8, seed=5, manifest=man2)
    assert rep2["result"] is True and rep2["sample"]["mismatch_n"] == 0, json.dumps(rep2, indent=1, default=str)[:2000]
    # and the store, asked for invocation 0 through a layer-0 path, gets the OTHER words (never layer 1's)
    st = OP.CommittedStore.of_record(bmap2, committed2, man2)
    w1, _ = st.words("r0", 0, ALIAS, "query", 0, 4)
    w0, _ = st.words("r0", 0, CANON, "query", 0, 4)
    assert w1 is not None and w0 is not None and w1.tolist() != w0.tolist()
