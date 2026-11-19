"""`gate call-boundaries` on a row dir after its Build, torch-free: a row passes when every `call_boundaries` identity of its manifest has
a source its Commit attaches -- covered by the call-boundary source's host plan over the Build dirs, or claimed by an attached source --
and stops naming the first one without (an identity the plan cannot cover, a claim naming a source the Commit does not attach, any call
boundary of a TP manifest); a manifest naming none passes."""
from __future__ import annotations

import json

from tests.acquire.call_boundary_programs import LP, PF, build_dir, dense_program
from verity_vllm.acquire import gate as G
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import taps
from verity_vllm.pipeline import cli


def _row_dir(tmp_path, **kw):
    """A row dir after its Build: the dense stand-in's Build dir and its manifest.json."""
    b, _at = dense_program(**kw)
    _d, man = build_dir(tmp_path, b.doc(LP), [PF[len("model."):]])
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return man


def _gate(tmp_path, capsys):
    rc = cli.main(["gate", "call-boundaries", str(tmp_path), "--out", str(tmp_path / "call_boundaries.json")])
    return rc, capsys.readouterr().out.strip().splitlines()[-1], json.loads((tmp_path / "call_boundaries.json").read_text())


def _cb(man):
    return sum(1 for r in man["identities"] if r["family"] == CB.FAMILY)


def test_a_row_whose_every_call_boundary_the_plan_covers_passes(tmp_path, capsys):
    man = _row_dir(tmp_path)
    n = _cb(man)
    rc, line, rec = _gate(tmp_path, capsys)
    assert rc == 0 and rec["pass"] is True and n > 0
    assert (rec["identities"], rec["claimed"], rec["covered"], rec["uncovered"], rec["unattached"]) == (n, 0, n, 0, 0)
    assert rec["sources"] == ["call_boundary"] and rec["program_dirs"] == {"r0": str(tmp_path / "build_request")}
    assert line.startswith(f"identities={n} claimed=0 covered={n} uncovered=0 unattached=0 sources=call_boundary wall=") and "first:" not in line


def test_an_identity_the_plan_cannot_cover_stops_the_row_by_name(tmp_path, capsys):
    man = _row_dir(tmp_path, embed=PF)
    plan = CB.plan_of(man, str(tmp_path))
    assert plan.uncovered
    rc, line, rec = _gate(tmp_path, capsys)
    assert rc == 1 and rec["pass"] is False
    assert (rec["identities"], rec["covered"], rec["uncovered"]) == (_cb(man), plan.covered, len(plan.uncovered))
    assert rec["named"]["uncovered"][0] == plan.uncovered[0]
    assert f"uncovered={len(plan.uncovered)} " in line and line.endswith(f"first: {plan.uncovered[0]}")
    assert line.split("first: ")[1].startswith(f"{PF[len('model.'):]}/") and "reads Program input 'prompt'" in line


def test_a_claim_naming_a_source_the_commit_does_not_attach_stops_the_row_by_name(tmp_path, capsys, monkeypatch):
    man = _row_dir(tmp_path)
    logits = [r for r in man["identities"] if r["family"] == CB.FAMILY and r["op_path"].startswith("logits_processor")]
    assert logits
    monkeypatch.setattr(CB, "CLAIMS", [lambda r: "logits_tap" if r["op_path"].startswith("logits_processor") else None])
    rc, line, rec = _gate(tmp_path, capsys)
    n = _cb(man)
    assert rc == 1 and rec["pass"] is False
    assert (rec["claimed"], rec["covered"], rec["uncovered"], rec["unattached"]) == (len(logits), n - len(logits), 0, len(logits))
    r = logits[0]
    assert line.endswith(f"first: {r['op_path']}/{r['output_member']}: claimed by logits_tap, a source the Commit does not attach")


QUANT = {"family": CB.FAMILY, "request_id": "r0", "engine_step": 0, "op_path": "model.layers.0.mlp.down_proj/int8_row_quant",
         "output_member": "0", "spec": "Int8RowQuant_v1"}


def test_an_int8_row_quant_call_boundary_is_claimed_by_the_int8_source_the_commit_attaches(tmp_path):
    man = _row_dir(tmp_path)
    n = _cb(man)
    man = {**man, "identities": [*man["identities"], QUANT]}
    assert taps.attached(G.classes_of_identities(man)) == ["int8"]
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is True and (rec["identities"], rec["claimed"], rec["covered"], rec["uncovered"]) == (n + 1, 1, n, 0)
    assert rec["sources"] == ["int8", "call_boundary"]


def test_a_claim_naming_an_attached_source_passes(tmp_path, monkeypatch):
    man = _row_dir(tmp_path)
    man = {**man, "identities": [*man["identities"], QUANT]}
    monkeypatch.setattr(CB, "CLAIMS", [lambda c: "int8" if c["op_path"].startswith(("logits_processor", "model.layers.0.mlp.down_proj"))
                                       else None])
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is True and rec["claimed"] > 1 and rec["covered"] == rec["identities"] - rec["claimed"]


def test_a_manifest_naming_no_call_boundaries_passes(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(json.dumps({"identities": [{"family": "instance_outputs"}, {"family": "weights"}]}))
    rc, line, rec = _gate(tmp_path, capsys)
    assert rc == 0 and rec["pass"] is True and rec["identities"] == 0
    assert rec["schema"] == "verity-vllm/call-boundaries-gate/v1"
    assert line.startswith("identities=0 claimed=0 covered=0 uncovered=0 unattached=0 sources=none wall=")


def test_every_call_boundary_of_a_tp_manifest_stops_the_row(tmp_path):
    man = {**_row_dir(tmp_path), "ranks": 2}
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is False and rec["uncovered"] == rec["identities"] == _cb(man) and "call_boundary" not in rec["sources"]
    assert rec["named"]["uncovered"][0].endswith(": a manifest of 2 ranks (the source serves one rank)")


def test_a_sparse_mla_call_boundary_is_claimed_by_no_source_and_a_tp_manifest_s_stops_the_row(tmp_path):
    ids = [{"family": CB.FAMILY, "request_id": "r0", "step": s, "engine_step": s, "invocation": 0, "rank": k, "output_member": "out",
            "op_path": f"inner.model.model.layers.0.self_attn/dsa_indexer_attn_{s}", "dtype": "bf16", "spec": "SparseMlaSplit_v1",
            "element_range": [0, 4096]} for k in range(8) for s in range(2)]
    man = {"ranks": 8, "fixture": {"model": "zai-org/GLM-5.3"}, "identities": ids}
    assert taps.attached(G.classes_of_identities(man)) == [] and all(CB.claimed(c) is None for c in G.classes_of_identities(man))
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is False and (rec["claimed"], rec["uncovered"], rec["unattached"]) == (0, len(ids), 0)
