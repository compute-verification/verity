"""[F-r16-15, rev M-0131 / coord M-0130 (c) F-r16-MOE-PIN] the fused-expert stacks under vLLM 0.28's RoutedExperts submodule.

The int-1 canary's OLMoE Commit FAILed closed at the weights pin: vLLM 0.28.1rc1 (the pods' engine) owns the fused expert stacks on a
RoutedExperts submodule of the FusedMoE layer -- served name `model.layers.N.mlp.experts.routed_experts.w13_weight` -- and MAN-09's
composition rule matched only `<layer>.mlp.experts.(w13|w2)_*`, so all 32 stacks of OLMoE (about 90 percent of its parameters) were
'no checkpoint tensor and no composition rule' (compared 180 / unresolved 32 / complete False).  A missing composition rule, named
correctly, not a mismatch (COORD ruling 5: repair from evidence, log it).

The rule now accepts the OPTIONAL `routed_experts` segment (nothing else), keeps the container the served name declares (`mlp.experts`,
Mixtral's `block_sparse_moe.experts`) as the home of the checkpoint's `<container>.<e>.<part>.<suffix>` tensors, and takes the per-expert
part names from what the checkpoint carries for expert 0: HF gate_proj|up_proj|down_proj (OLMoE, Qwen3-MoE) or Mixtral's w1|w3|w2 -- the
same layout vLLM's RoutedExperts._load_w13 / _load_w2 fill (w1 = gate first half, w3 = up second half, w2 = down; read on cp-r4's
vllm 0.28.1rc1.dev472 routed_experts.py).  Negatives (coord (c)): one element of one expert flipped, two experts permuted, an expert
replaced by another's tensors, an unknown segment -- every one refused BY NAME; the honest served stacks pass with every field compared.
CPU, synthetic checkpoint (real safetensors), no torch.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_vllm.check import weights_of_record as WR
from tests.check.test_weights_of_record import _bf16, live_digests, write_safetensors

H, I, E = 8, 6, 4          # hidden, intermediate, experts


def _ty(shape, bits):
    t = f"Value<{bits}>"
    for n in reversed(shape):
        t = f"Array<{n},{t}>"
    return t


def _program(fields):
    struct_ = "{" + ", ".join(f"{n}: {_ty(s, b)}" for n, s, b in fields) + "}"
    return {"instances": {"rows": [], "params": [["weights", struct_]], "digest": "e" * 128}, "result": {"program_digest": "e" * 128}, "dir": "synthetic"}


@pytest.fixture(params=["olmoe", "mixtral"])
def moe(request, tmp_path):
    """A tiny MoE checkpoint in the HF layout of the family: OLMoE / Qwen3-MoE (mlp.experts.<e>.gate_proj|up_proj|down_proj) or Mixtral
    (block_sparse_moe.experts.<e>.w1|w3|w2); the served names carry vLLM 0.28's routed_experts segment."""
    rng = np.random.default_rng(11)
    fam = request.param
    container = "model.layers.0.mlp.experts" if fam == "olmoe" else "model.layers.0.block_sparse_moe.experts"
    gate, up, down = ("gate_proj", "up_proj", "down_proj") if fam == "olmoe" else ("w1", "w3", "w2")
    t = {"model.embed_tokens.weight": ("BF16", _bf16(rng, 10, H)), "model.norm.weight": ("BF16", _bf16(rng, H))}
    for e in range(E):
        t[f"{container}.{e}.{gate}.weight"] = ("BF16", _bf16(rng, I, H))
        t[f"{container}.{e}.{up}.weight"] = ("BF16", _bf16(rng, I, H))
        t[f"{container}.{e}.{down}.weight"] = ("BF16", _bf16(rng, H, I))
    d = tmp_path / "snap"; d.mkdir()
    write_safetensors(str(d / "model.safetensors"), t)
    json.dump({"tie_word_embeddings": False, "hidden_size": H}, open(d / "config.json", "w"))
    man = tmp_path / "checkpoints.json"
    json.dump({"checkpoints": [{"repo": f"test/{fam}", "revision": "abc", "local_path": str(d),
                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / 'model.safetensors'))}}}]}, open(man, "w"))
    served_w13 = f"{container}.routed_experts.w13_weight"
    served_w2 = f"{container}.routed_experts.w2_weight"
    fields = [("model.model.embed_tokens.weight", [10, H], 16), ("model." + served_w13, [E, 2 * I, H], 16), ("model." + served_w2, [E, H, I], 16),
              ("model.model.norm.weight", [H], 16)]

    def A(n):
        return WR.to_served(t[n][1], t[n][0], 16)[0]
    served = {"model.embed_tokens.weight": A("model.embed_tokens.weight"), "model.norm.weight": A("model.norm.weight"),
              served_w13: np.stack([np.concatenate([A(f"{container}.{e}.{gate}.weight"), A(f"{container}.{e}.{up}.weight")]) for e in range(E)]),
              served_w2: np.stack([A(f"{container}.{e}.{down}.weight") for e in range(E)])}
    return {"fam": fam, "dir": d, "man": man, "t": t, "fields": fields, "served": served, "container": container, "w13": served_w13, "w2": served_w2,
            "parts": (gate, up, down)}


def _record(m):
    return WR.weights_of_record(_program(m["fields"]), WR.CheckpointIndex(str(m["dir"]), checkpoints_manifest=str(m["man"])), prefix="model.")


def test_routed_experts_stacks_are_composed_from_the_per_expert_checkpoint_tensors_and_the_rule_names_it(moe):
    rec = _record(moe)
    assert rec["complete"] is True and rec["unresolved"] == [], rec["unresolved"]
    assert rec["n_pinned"] == rec["n_fields"] == len(moe["fields"])
    by = {f["field"]: f for f in rec["fields"]}
    gate, up, down = moe["parts"]
    r13 = by["model." + moe["w13"]]
    assert r13["rule"] == f"fused experts w13 (vLLM RoutedExperts submodule) <- {gate}|{up} per expert ({moe['container']}.<e>.<part>.weight)"
    assert r13["sources"] == [f"{moe['container']}.{e}.{p}.weight" for e in range(E) for p in (gate, up)]
    r2 = by["model." + moe["w2"]]
    assert r2["rule"].startswith("fused experts w2 (vLLM RoutedExperts submodule) <- " + down) and len(r2["sources"]) == E
    # the honest served stacks (vLLM's _load_w13: gate first half, up second; _load_w2: down) pass with EVERY field compared
    ok = WR.check(rec, live_digests(moe["served"]), program_digests_of_record={"e" * 128}, checkpoint_of_record={"repo": f"test/{moe['fam']}", "revision": "abc"})
    assert ok["result"] is True and ok["compared"] == len(moe["fields"]) and ok["mismatched"] == [] and ok["missing"] == [], ok
    # the stacks are a function of the checkpoint bytes: the plain (pre-0.28) served name composes the SAME digests
    plain_fields = [(n.replace(".routed_experts.", "."), s, b) for n, s, b in moe["fields"]]
    rec_plain = WR.weights_of_record(_program(plain_fields), WR.CheckpointIndex(str(moe["dir"]), checkpoints_manifest=str(moe["man"])), prefix="model.")
    by_plain = {f["field"].replace(".routed_experts.", "."): f["digest"] for f in rec_plain["fields"]}
    assert rec_plain["complete"] and all(by_plain[f["field"].replace(".routed_experts.", ".")] == f["digest"] for f in rec["fields"])
    assert "(vLLM RoutedExperts submodule)" not in next(f["rule"] for f in rec_plain["fields"] if f["field"].endswith("w13_weight"))


def test_unknown_segments_and_absent_parts_are_refused_by_name_never_guessed(moe):
    d, man = moe["dir"], moe["man"]
    ck = WR.CheckpointIndex(str(d), checkpoints_manifest=str(man))
    # any other segment between the container and the stack: no composition rule
    with pytest.raises(KeyError, match="no checkpoint tensor and no composition rule"):
        WR.compose(f"{moe['container']}.shared.w13_weight", [E, 2 * I, H], 16, ck)
    with pytest.raises(KeyError, match="no checkpoint tensor and no composition rule"):
        WR.compose(f"{moe['container']}.routed_experts.extra.w2_weight", [E, H, I], 16, ck)
    # a container whose expert 0 carries neither HF nor Mixtral part names: refused naming both candidate sets
    with pytest.raises(KeyError, match=r"no per-expert checkpoint tensors model.layers.7.mlp.experts.0.\{gate_proj\|up_proj\|w1\|w3\}.weight"):
        WR.compose("model.layers.7.mlp.experts.routed_experts.w13_weight", [E, 2 * I, H], 16, ck)
    # a stack that declares MORE experts than the checkpoint has: the missing expert is named (not zero-filled)
    with pytest.raises(KeyError, match=f"expert {E} sources missing"):
        WR.compose(moe["w13"], [E + 1, 2 * I, H], 16, ck)
    # in a record: the unresolved field is named with its reason and the record is incomplete (the checker FAILs 'weights of record incomplete')
    bad_fields = moe["fields"] + [("model.model.layers.0.mlp.experts.other.w13_weight", [E, 2 * I, H], 16)]
    rec = WR.weights_of_record(_program(bad_fields), WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix="model.")
    assert rec["complete"] is False and [u["field"] for u in rec["unresolved"]] == ["model.model.layers.0.mlp.experts.other.w13_weight"]
    assert "no composition rule" in rec["unresolved"][0]["why"]


def test_expert_level_substitutions_fail_the_pin_by_name(moe):
    """coord M-0130 (c): wrong-expert / permuted-expert negatives -- a served stack that is not the checkpoint's experts in the checkpoint's
    order is a named FAIL, whatever the scalar statistics (a permutation keeps every byte, only the order changes)."""
    rec = _record(moe)
    honest = moe["served"]
    ctx = {"program_digests_of_record": {"e" * 128}, "checkpoint_of_record": {"repo": f"test/{moe['fam']}", "revision": "abc"}}
    assert WR.check(rec, live_digests(honest), **ctx)["result"] is True

    def _fails(served, field):
        res = WR.check(rec, live_digests(served), **ctx)
        assert res["result"] is False and [m["served_name"] for m in res["mismatched"]] == [field], res
        assert "1 served weight(s) differ from the checkpoint of record: " + field in res["why"]
        return res
    # (1) one element of ONE expert's slice flipped (rev M-0131 acceptance: flipweight on ...routed_experts.w2_weight)
    s = dict(honest); w = s[moe["w2"]].copy(); w[2, 3, 1] ^= 0x0040; s[moe["w2"]] = w
    _fails(s, moe["w2"])
    # (2) two experts permuted (same bytes, wrong order): the stack is not the checkpoint's experts by index
    s = dict(honest); w = s[moe["w13"]].copy(); w[[0, 1]] = w[[1, 0]]; s[moe["w13"]] = w
    assert np.array_equal(np.sort(w.ravel()), np.sort(honest[moe["w13"]].ravel()))          # a permutation: every word kept
    _fails(s, moe["w13"])
    # (3) an expert replaced by another expert's tensors (a duplicated expert)
    s = dict(honest); w = s[moe["w2"]].copy(); w[3] = w[0]; s[moe["w2"]] = w
    _fails(s, moe["w2"])
    # (4) gate|up halves swapped inside one expert (up first): the layout vLLM fills is gate first (w1), up second (w3)
    s = dict(honest); w = s[moe["w13"]].copy(); w[1] = np.concatenate([w[1, I:], w[1, :I]]); s[moe["w13"]] = w
    _fails(s, moe["w13"])
    # (5) a stack with one expert fewer: shape differs -> named as a mismatch of that field, never compared partially
    s = dict(honest); s[moe["w13"]] = honest[moe["w13"]][:E - 1]
    res = WR.check(rec, live_digests(s), **ctx)
    assert res["result"] is False and [m["served_name"] for m in res["mismatched"]] == [moe["w13"]]
