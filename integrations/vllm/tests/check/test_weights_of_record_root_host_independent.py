"""root_of_record is a function of (weights struct, checkpoint bytes, config.json) -- not of the host.

Llama-3.2-1B B1 records derived on cp-r1 (EPYC 9354, numpy AVX512 dispatch) and cp-a100 (EPYC 7763, no AVX512) agreed on all
162 byte-digest fields and differed in ONE: the analytic rotary entry's `inv_freq_sha256` -- numpy's fp32 `np.power(base, exps)` takes
SVML on AVX512 hosts and returns 7 of 32 exponents one ulp off the correctly rounded value (measured on both pods).  v1 keyed the
analytic entry of the root by those bytes, so one model had two roots.  The pin VERDICT never depended on it (check_cos_sin is a
bounded check against the analytic function), but a root that names the weights of record must not name the deriving CPU.

v2 keys the analytic entry by the sha256 of its EXACT inputs (table name, declared shape, the config.json values the function reads);
the host's inv_freq bytes stay on the record as information.  Tests (CPU, no torch): perturbing what the host computes leaves the root
unchanged; changing an input the function reads changes it; the checkpoint bytes still change it; a v1 record's root is not re-rooted
by the reader (records are never hand-edited: the row_pod reuse rule re-derives them, test_row_pod_weights_of_record_reuse).
"""
from __future__ import annotations

import json

import numpy as np

from verity_vllm.program import model as analytic
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import cli
from tests.check.test_weights_of_record import PREFIX, _program, checkpoint  # noqa: F401  (fixture)

FIELD = "model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"


def _rotary_setup(d, theta=10000.0):
    cfg = json.load(open(d / "config.json"))
    cfg.update({"model_type": "llama", "num_hidden_layers": 1, "intermediate_size": 12, "vocab_size": 20, "rope_theta": theta,
                "max_position_embeddings": 64, "head_dim": 4, "rms_norm_eps": 1e-5, "torch_dtype": "bfloat16"})
    json.dump(cfg, open(d / "config.json", "w"))
    C, _ = analytic.config_of(cfg)
    MAXP, ROT = int(C["MAX_POS"]), analytic.rotary_dim(C)
    prog, _fields = _program()
    prog["instances"]["params"] = [["weights", "{" + f"{FIELD}: Array<{MAXP},Array<{ROT},Value<16>>>, model.model.norm.weight: Array<8,Value<16>>" + "}"]]
    return prog


def _record(checkpoint):
    d, man, _t = checkpoint
    return WR.weights_of_record(_rotary_setup(d), WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)


def test_the_root_does_not_move_when_the_host_computes_inv_freq_differently(checkpoint, monkeypatch):
    rec = _record(checkpoint)
    assert rec["schema"] == WR.SCHEMA == "verity/commit/weights-of-record/v2" and rec["analytic_fields"] == [FIELD]
    ent = {f["field"]: f for f in rec["fields"]}[FIELD]
    assert len(ent["analytic"]["inputs_sha256"]) == 64 and len(ent["analytic"]["inv_freq_sha256"]) == 64 and "host" in ent["analytic"]["inv_freq_note"]
    # the other host: every inv_freq word one ulp off (what AVX512 SVML did to 7 of 32 on cp-r1, here to all of them)
    real = analytic._inv_freq

    def other_host(C):
        inv = real(C)
        return (inv.view(np.uint32) + np.uint32(1)).view(np.float32)
    monkeypatch.setattr(analytic, "_inv_freq", other_host)
    rec2 = _record(checkpoint)
    ent2 = {f["field"]: f for f in rec2["fields"]}[FIELD]
    assert ent2["analytic"]["inv_freq_sha256"] != ent["analytic"]["inv_freq_sha256"]          # the host's value did change ...
    assert ent2["analytic"]["inputs_sha256"] == ent["analytic"]["inputs_sha256"]              # ... the inputs did not ...
    assert rec2["root_of_record"] == rec["root_of_record"]                                    # ... and the root names the same weights of record
    assert [f["digest"] for f in rec2["fields"]] == [f["digest"] for f in rec["fields"]]


def test_the_root_moves_with_the_analytic_inputs_and_with_the_checkpoint_bytes(checkpoint):
    d, man, _t = checkpoint
    rec = _record(checkpoint)
    # another theta in config.json: a different analytic function -> a different key, a different root (the served table would FAIL too)
    prog = _rotary_setup(d, theta=20000.0)
    rec_theta = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    e0 = {f["field"]: f for f in rec["fields"]}[FIELD]["analytic"]; e1 = {f["field"]: f for f in rec_theta["fields"]}[FIELD]["analytic"]
    assert e1["inputs_sha256"] != e0["inputs_sha256"] and e1["C"]["ROPE_THETA"] == 20000.0 and rec_theta["root_of_record"] != rec["root_of_record"]
    # the key is exactly the sha256 of the canonical JSON of (table, expected_shape, C): recomputable from the record alone
    import hashlib
    key = hashlib.sha256(json.dumps({"table": "cos_sin", "expected_shape": e0["expected_shape"], "C": e0["C"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert key == e0["inputs_sha256"]
    # a byte of the checkpoint (the norm) changed: the root moves through the field digest (unchanged v1 behaviour)
    from tests.check.test_weights_of_record import write_safetensors
    t = dict(_t); w = t["model.norm.weight"][1].copy(); w[0] ^= 0x0001; t["model.norm.weight"] = ("BF16", w)
    write_safetensors(str(d / "model.safetensors"), t)
    man2 = d.parent / "checkpoints2.json"
    json.dump({"checkpoints": [{"repo": "test/x", "revision": "abc", "local_path": str(d), "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}, open(man2, "w"))
    rec_bytes = WR.weights_of_record(_rotary_setup(d), WR.CheckpointIndex(str(d), checkpoints_manifest=str(man2)), prefix=PREFIX)
    assert rec_bytes["root_of_record"] != rec["root_of_record"]
    assert {f["field"]: f for f in rec_bytes["fields"]}[FIELD]["analytic"]["inputs_sha256"] == e0["inputs_sha256"]


def test_a_v1_record_is_read_as_it_is_and_the_checker_still_pins_by_field(checkpoint):
    """The reader never re-roots an older record (repairs are logged derivations, done by row_pod's reuse rule)."""
    rec = _record(checkpoint)
    v1 = json.loads(json.dumps(rec)); v1["schema"] = WR.SCHEMA_V1; v1["root_of_record"] = "11" * 32
    for f in v1["fields"]:
        if f.get("analytic"):
            f["analytic"].pop("inputs_sha256"); f["analytic"].pop("inv_freq_note")
    d, _man, t = checkpoint
    cfg = json.load(open(d / "config.json")); C, _ = analytic.config_of(cfg)
    norm = WR.to_served(t["model.norm.weight"][1], t["model.norm.weight"][0], 16)[0]
    import hashlib
    live = {"params": {"model.layers.0.self_attn.rotary_emb.cos_sin_cache": {"digest": "x", "dtype": "bfloat16", "shape": list(analytic.cos_sin_table(C).shape)},
                       "model.norm.weight": {"digest": hashlib.sha256(np.ascontiguousarray(norm).tobytes()).hexdigest(), "dtype": "bfloat16", "shape": [8]}},
            "raw": {"model.layers.0.self_attn.rotary_emb.cos_sin_cache": analytic.cos_sin_table(C)}}
    ok = WR.check(v1, live)
    assert ok["result"] is True and ok["compared"] == 2 and ok["root_of_record"] == "11" * 32        # reported as recorded, never recomputed


def test_reroot_and_diff_are_readers_that_name_host_only_differences_apart_from_weight_differences(checkpoint, tmp_path, capsys):
    """Report compares same-model rows across pods under schema v1 by the fields: `--diff A B` names one-sided fields,
    fields whose digest / shape / bits / analytic INPUTS differ (the weights differ), and host-only differences (inv_freq_sha256) apart;
    both records are re-rooted under the current schema without being changed.  reroot() of a v2 record == its root_of_record."""
    rec = _record(checkpoint)
    assert WR.reroot(rec) == rec["root_of_record"]                                            # v2: the root is recomputable from the fields
    # the other pod's v1 record of the SAME weights: v1 schema, another root, another inv_freq_sha256, no inputs_sha256
    v1 = json.loads(json.dumps(rec)); v1["schema"] = WR.SCHEMA_V1; v1["root_of_record"] = "22" * 32
    for f in v1["fields"]:
        if f.get("analytic"):
            f["analytic"].pop("inputs_sha256"); f["analytic"]["inv_freq_sha256"] = "ab" * 32
    d = WR.diff_records(rec, v1)
    assert d["same_weights_of_record"] is True and d["differ"] == [] and d["only_a"] == d["only_b"] == []
    assert [h["field"] for h in d["host_only"]] == [FIELD] and d["reroot"][0] == d["reroot"][1] == rec["root_of_record"]
    assert d["schema"] == [WR.SCHEMA, WR.SCHEMA_V1] and d["root_of_record"][1] == "22" * 32                # as recorded, never rewritten
    # a record whose norm digest differs (another checkpoint byte): the weights differ, named by field and key; roots differ
    other = json.loads(json.dumps(rec))
    nf = next(f for f in other["fields"] if f["field"] == "model.model.norm.weight"); nf["digest"] = "cd" * 32
    d2 = WR.diff_records(rec, other)
    assert d2["same_weights_of_record"] is False and [x["field"] for x in d2["differ"]] == ["model.model.norm.weight"] and d2["differ"][0]["keys"] == ["digest"]
    assert d2["reroot"][0] != d2["reroot"][1] and d2["host_only"] == []
    # another rope theta: the analytic INPUTS differ -> a weights difference (not host-only)
    theta = json.loads(json.dumps(rec))
    af = next(f for f in theta["fields"] if f["field"] == FIELD); af["analytic"]["C"]["ROPE_THETA"] = 20000.0; af["analytic"].pop("inputs_sha256")
    d3 = WR.diff_records(rec, theta)
    assert d3["same_weights_of_record"] is False and d3["differ"][0]["keys"] == ["analytic inputs"]
    # a field on one side only
    short = json.loads(json.dumps(rec)); short["fields"] = [f for f in short["fields"] if f["field"] != "model.model.norm.weight"]
    d4 = WR.diff_records(rec, short)
    assert d4["same_weights_of_record"] is False and d4["only_a"] == ["model.model.norm.weight"]
    # the CLI: exit 0 same / 1 different; nothing written; no --checkpoint-dir needed
    pa, pb, pc = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    pa.write_text(json.dumps(rec)); pb.write_text(json.dumps(v1)); pc.write_text(json.dumps(other))
    before = {p.name: p.stat().st_mtime_ns for p in (pa, pb, pc)}
    assert cli.main(["weights-of-record", "--diff", str(pa), str(pb)]) == 0
    out = capsys.readouterr().out
    assert "same weights of record: True" in out and "host-only " + FIELD in out and "re-root under " + WR.SCHEMA in out
    assert cli.main(["weights-of-record", "--diff", str(pa), str(pc)]) == 1
    assert "DIFFER model.model.norm.weight: digest" in capsys.readouterr().out
    assert {p.name: p.stat().st_mtime_ns for p in (pa, pb, pc)} == before
    # deriving still needs --checkpoint-dir (a manual check now that --diff exists without one)
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["weights-of-record", "--program-dir", str(tmp_path)])
    assert "--checkpoint-dir is required" in capsys.readouterr().err
