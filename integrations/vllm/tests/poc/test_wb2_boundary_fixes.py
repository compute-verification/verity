"""WB2 (2026-09-07): fixes after WA's second audit of the security-mode baseline
(``docs/verified-backend/audit-security-baseline.md``).

Fix 1 -- the chosen logprobs are part of the protected production boundary (finding 1, WA's M2):
one ``float32`` ``output_logprob`` leaf per produced token, the IEEE-754 binary32 word of the value the
API reported (never rounded); bundles that predate the leaves verify as before and say ``unbound``;
negative N19 falsifies one logprob and must be rejected; ``derive_logprobs`` rewrites an old archive's
boundary trees from its own production record; ``verify_cpu`` checks the result from the archive alone.

Fix 2 -- no crash without a verdict (finding 4, WA's M6 / M8b / M12): every structural exception of the
pipeline is a ``FAIL structural: <check>: …`` row; a tampered index leaf or stored weight value is an
``auth_failed`` opening, never an uncaught ``CommitError``.

Also: ``verification_evidence`` row labels (GUIDANCE §15.2) and the render's ``bindings`` /
``verification_evidence`` columns.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

from verity_vllm import commit as vcommit
from verity_vllm import relations, rows, synthetic
from verity_vllm.capture import LOGPROB_DTYPE, Boundary, CaptureReader, binary32_exact, boundary_tree_id
from verity_vllm.commit import MerkleTree, TreeCache
from verity_vllm.description import Description
from verity_vllm import rowcheck
from verity_vllm_adapter import binding as vbinding
from verity_vllm_adapter import derive_logprobs, verify, verify_cpu
from verity_vllm_adapter import verify_bindings as VB

TOY_PLUGIN_INSTALL = synthetic.install


PID = "0123456789ab"


def f32(x: float) -> float:
    """The nearest binary32 value of ``x`` as a Python float (what vLLM's float32 logprob reads as)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def write_production(run_dir: Path, path: Path, *, logprobs: bool = True) -> Path:
    reader = CaptureReader(run_dir)
    with open(path, "w") as handle:
        for i, rid in enumerate(reader.request_ids()):
            b = reader.boundary(rid)
            rec = {"request_id": rid, "prompt_token_ids": b.prompt_token_ids, "output_token_ids": list(b.output_token_ids),
                   "finish_reason": b.finish_reason, "sampling_params": b.sampling_params}
            if logprobs:
                rec["chosen_logprobs"] = [f32(-0.37 * (k + 1) - 0.11 * i) for k in range(len(b.output_token_ids))]
            handle.write(json.dumps(rec) + "\n")
    return path


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("wb2")
    t = synthetic.build_toy_run(root / "good", seed=3)
    prod_dir = root / "prod"
    prod_dir.mkdir()
    write_production(t.run_dir, prod_dir / "production.jsonl", logprobs=True)
    legacy_prod = root / "prod-legacy"
    legacy_prod.mkdir()
    write_production(t.run_dir, legacy_prod / "production.jsonl", logprobs=False)
    derived = root / "derived"
    derive_logprobs.derive(t.run_dir, prod_dir / "production.jsonl", derived, source_label="toy")
    return {"root": root, "toy": t, "run": t.run_dir, "prod": prod_dir, "prod_legacy": legacy_prod, "derived": derived / "commit",
            "derived_root": derived}


# --- Fix 1: the leaf encoding -----------------------------------------------------------------------------


def test_logprob_leaves_are_bit_exact_float32_words():
    steps = [{"step": 0, "phase": "prefill", "positions": [0, 1, 2, 3]}, {"step": 1, "phase": "decode", "positions": [4]}, {"step": 2, "phase": "decode", "positions": [5]}]
    lps = [f32(-1.2345), f32(-0.001), f32(-7.5)]
    b = Boundary.build("r", [1, 2, 3, 4], [5, 6, 7], {"max_tokens": 3}, seed=None, finish_reason="length", steps=steps, chosen_logprobs=lps)
    legacy = Boundary.build("r", [1, 2, 3, 4], [5, 6, 7], {"max_tokens": 3}, seed=None, finish_reason="length", steps=steps)
    assert b.binds_logprobs and not legacy.binds_logprobs and b.validate() == []
    leaves = {v.canonical_id: v for v in b.leaves(PID)}
    lp_leaves = {cid: v for cid, v in leaves.items() if "/f=output_logprob/" in cid or "output_logprob" in cid}
    assert len(lp_leaves) == 3
    for v, x in zip(sorted(lp_leaves.values(), key=lambda v: v.canonical_id), lps, strict=True):
        assert v.dtype == LOGPROB_DTYPE == "float32" and tuple(v.shape) == ()
        assert v.data == struct.pack("<f", x)  # the binary32 word, little-endian: never a rounded decimal
    # the legacy leaf set is exactly the bound set minus the logprob leaves (same ids, same bytes)
    legacy_leaves = {v.canonical_id: v.data for v in legacy.leaves(PID)}
    assert legacy_leaves == {cid: v.data for cid, v in leaves.items() if cid not in lp_leaves}
    # one ulp moves the root
    lps2 = list(lps)
    lps2[1] = struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", lps[1]))[0] + 1))[0]
    b2 = Boundary.build("r", [1, 2, 3, 4], [5, 6, 7], {"max_tokens": 3}, seed=None, finish_reason="length", steps=steps, chosen_logprobs=lps2)
    assert MerkleTree.from_values(b2.leaves(PID), "boundary/r").root != MerkleTree.from_values(b.leaves(PID), "boundary/r").root
    # JSON round trip keeps the field only when bound; a legacy document has no key
    doc = b.to_json()
    assert doc["chosen_logprobs"] == lps and "chosen_logprobs" not in legacy.to_json()
    again = Boundary.from_json(doc)
    assert again.chosen_logprobs == lps and again.leaves(PID)[-1].data == b.leaves(PID)[-1].data
    # never rounded, never None, never a wrong count: structural problems, not leaves
    for bad in ([-0.0001, f32(-1.0), f32(-1.0)], [None, f32(-1.0), f32(-1.0)], [f32(-1.0)]):
        nb = Boundary.build("r", [1, 2, 3, 4], [5, 6, 7], {"max_tokens": 3}, seed=None, finish_reason="length", steps=steps, chosen_logprobs=bad)
        assert nb.validate(), bad
    assert binary32_exact(f32(-0.1)) and not binary32_exact(-0.1) and not binary32_exact(float("nan")) and not binary32_exact(None)


def test_status_unbound_for_old_trees_bound_for_derived(toy):
    reader = CaptureReader(toy["run"])
    com = vcommit.load_commitments(reader)
    production = {json.loads(l)["request_id"]: json.loads(l) for l in (toy["prod"] / "production.jsonl").read_text().splitlines()}
    _, status = rowcheck.production_boundaries_with_status(production, com)
    assert set(status.values()) == {"unbound"}  # the production record carries logprobs; the committed trees do not: never upgraded
    dreader = CaptureReader(toy["derived"])
    _, dstatus = rowcheck.production_boundaries_with_status(production, vcommit.load_commitments(dreader))
    assert set(dstatus.values()) == {"bound"}
    d = json.loads((toy["derived_root"] / derive_logprobs.DERIVED_FILE).read_text())
    assert d["tool"] == derive_logprobs.TOOL and "not a capture" in d["kind"]
    for r in d["change"]["requests"].values():
        assert r["root_after"] != r["root_before"] and r["leaf_count_after"] == r["leaf_count_before"] + r["logprob_leaves"]
    # everything but the boundary trees is byte-identical
    before, after = com["trees"], vcommit.load_commitments(dreader)["trees"]
    assert {t: r for t, r in before.items() if not t.startswith("boundary/")} == {t: r for t, r in after.items() if not t.startswith("boundary/")}


def test_output_binding_and_n19(toy, tmp_path):
    reader = CaptureReader(toy["run"])
    com = vcommit.load_commitments(reader)
    prod = toy["prod"] / "production.jsonl"
    legacy = vbinding.output_binding(reader, com, prod)
    assert legacy.ok and legacy.data["logprobs"] == "unbound" and legacy.data["logprob_leaves"] == 0
    dreader = CaptureReader(toy["derived"])
    dcom = vcommit.load_commitments(dreader)
    bound = vbinding.output_binding(dreader, dcom, prod)
    assert bound.ok and bound.data["logprobs"] == "bound" and bound.data["logprob_leaves"] > 0
    assert bound.data["logprob_leaf_dtype"] == "float32" and "binary32" in bound.data["logprob_leaf_encoding"]
    # WA's M2: one falsified logprob in production.jsonl -- structural output-binding failure once bound
    recs = [json.loads(l) for l in prod.read_text().splitlines()]
    recs[0]["chosen_logprobs"][0] = f32(recs[0]["chosen_logprobs"][0] - 0.5)
    m2 = tmp_path / "m2.jsonl"
    m2.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert not vbinding.output_binding(dreader, dcom, m2).ok
    assert vbinding.output_binding(reader, com, m2).ok  # the old archive cannot see it: its row says unbound
    # a record without logprobs against a bound boundary is a problem too
    assert not vbinding.output_binding(dreader, dcom, toy["prod_legacy"] / "production.jsonl").ok
    # N19 rejects every variant on the bound bundle; inapplicable (not NOT_RUN) on the legacy one
    n19 = vbinding.binding_negatives(dreader, dcom, cp=None, production_path=prod, out_dir=tmp_path / "neg", ids=["N19"])
    assert len(n19) == 1 and n19[0]["attempted"] >= 4 and n19[0]["rejected"] == n19[0]["attempted"] and not n19[0].get("inapplicable")
    old = vbinding.binding_negatives(reader, com, cp=None, production_path=prod, out_dir=tmp_path / "neg2", ids=["N19"])
    assert old[0]["attempted"] == 0 and old[0]["inapplicable"]
    assert "N19" in vbinding.BINDING_NEGATIVE_IDS


def test_derive_refuses_inconsistent_or_already_bound_sources(toy, tmp_path):
    with pytest.raises(derive_logprobs.DeriveError, match="already binds"):
        derive_logprobs.derive(toy["derived"], toy["prod"] / "production.jsonl", tmp_path / "again", source_label="derived")
    with pytest.raises(derive_logprobs.DeriveError, match="no chosen_logprobs"):
        derive_logprobs.derive(toy["run"], toy["prod_legacy"] / "production.jsonl", tmp_path / "nolp", source_label="toy")


def test_verify_cpu_rows_say_bound_or_unbound(toy, tmp_path):
    rc = verify_cpu.run([str(toy["derived"]), "--production", str(toy["prod"]), "--out", str(tmp_path / "d"), "--case", "B0", "--worker", "WB2"])
    row = json.loads((tmp_path / "d" / "table1.jsonl").read_text())
    assert rc == 0 and row["result"] == "NOT_RUN" and rows.logprobs_label(row) == "bound"
    assert row["verification_evidence"]["label"] == rows.EVIDENCE_DERIVED and row["structural"]["result"] == "ok"
    assert row["negative"]["attempted"] == row["negative"]["rejected"] > 0 and rows.validate_row(1, row) == []
    assert rows.result_label(row) == "NOT_RUN"  # a blocker kind is not qualified by the (absent) challenge
    t3 = [json.loads(l) for l in (tmp_path / "d" / "table3.jsonl").read_text().splitlines()]
    assert {r["negative_id"] for r in t3} == {"N17", "N19"}
    rc = verify_cpu.run([str(toy["run"]), "--production", str(toy["prod"]), "--out", str(tmp_path / "l"), "--case", "B0"])
    row = json.loads((tmp_path / "l" / "table1.jsonl").read_text())
    assert rc == 0 and rows.logprobs_label(row) == "unbound" and row["verification_evidence"]["label"] == rows.EVIDENCE_CPU_ONLY
    assert row["challenge"]["negatives_inapplicable"]["N19"]
    # M2 through the CPU verifier: FAIL structural: output binding
    recs = [json.loads(l) for l in (toy["prod"] / "production.jsonl").read_text().splitlines()]
    recs[1]["chosen_logprobs"][-1] = f32(-0.25)
    (tmp_path / "m2").mkdir()
    (tmp_path / "m2" / "production.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    verify_cpu.run([str(toy["derived"]), "--production", str(tmp_path / "m2"), "--out", str(tmp_path / "m2out"), "--case", "B0"])
    row = json.loads((tmp_path / "m2out" / "table1.jsonl").read_text())
    assert row["result"] == "FAIL" and row["first_blocker"].startswith("structural: output binding")


# --- Fix 2: no crash without a verdict ----------------------------------------------------------------------


def test_m8b_dropped_layer_is_a_fail_row_with_the_check_named(toy, tmp_path, capsys):
    m8b = tmp_path / "m8b"
    shutil.copytree(toy["run"], m8b)
    d = json.loads((m8b / "description.json").read_text())
    d["nodes"] = [n for n in d["nodes"] if n.get("layer") != "0"]  # WA's M8b: every node of one layer removed
    (m8b / "description.json").write_text(json.dumps(d))
    out = tmp_path / "out"
    out.mkdir()
    (out / "table1.jsonl").write_text('{"stale": true}\n')  # a stale row from an earlier run must be replaced, not kept
    rc = verify.run([str(m8b), "--production", str(toy["prod"]), "--out", str(out), "--case", "B0", "--worker", "WB2"])
    captured = capsys.readouterr()
    row = json.loads((out / "table1.jsonl").read_text())
    assert rc == 1 and row["result"] == "FAIL" and row["structural"]["result"] == "failed"
    assert row["first_blocker"].startswith("structural: description validation: node ") and "refers to" in row["first_blocker"]
    assert row["relation_checks"] == {"same_runtime": {"exact": 0, "attempted": 0}, "independent": {"exact": 0, "attempted": 0}}
    assert row["authentication"] == {"valid": 0, "attempted": 0} and row["worker"] == "WB2"
    assert captured.out.startswith("RESULT FAIL first_blocker=structural: description validation")
    assert "Traceback" in captured.err and "DescriptionError" in captured.err  # the traceback is on stderr, never the only output
    vj = json.loads((out / "verify.json").read_text())
    assert vj["structural"]["result"] == "failed" and "DescriptionError" in vj["exception"]


def test_structural_reason_and_refusal_row_name_the_check(tmp_path, capsys):
    assert VB.structural_reason(VB.StructuralFailure("commitments", "unexpected schema")) == "commitments: unexpected schema"
    assert VB.structural_reason(vcommit.CommitError("w=x: value does not match the committed leaf")).startswith("commitment: CommitError: w=x")
    assert VB.structural_reason(FileNotFoundError("required.json")).startswith("missing file: FileNotFoundError")
    assert VB.structural_reason(RuntimeError("boom")) == "verifier exception: RuntimeError: boom"
    t1 = VB.refusal_row(tmp_path, case="B0", profile_id="p", worker="WB2", adapter_revision="x", reason="commitment: CommitError: w=x", exception="tb", result="failed")
    assert t1["result"] == "FAIL" and t1["first_blocker"] == "structural: commitment: CommitError: w=x" and t1["structural"]["result"] == "failed"
    assert t1["production_changes"].startswith("not assessed: a structural check failed") and rows.validate_row(1, t1) == []
    assert capsys.readouterr().out.startswith("RESULT FAIL first_blocker=structural: commitment")


@pytest.mark.parametrize("recompute", [True, False])
def test_m6_m12_tampered_weights_are_auth_failures_not_crashes(toy, tmp_path, recompute):
    TOY_PLUGIN_INSTALL()
    reader = CaptureReader(toy["run"])
    sample = [c for c in reader.ru("req-0", 0).ids() if "mlp" in c or "embed" in c]
    assert sample

    def check(run_dir: Path) -> dict[str, int]:
        r = CaptureReader(run_dir)
        rep = relations.check(Description(r.description), r, vcommit.load_commitments(r), sample, cache=TreeCache(r, recompute_leaves=recompute))
        return rep.to_json()["statuses"]

    assert check(toy["run"]) == {"exact": len(sample)}
    m6 = tmp_path / "m6"  # WA's M6: the index's recorded leaf of one weight altered, sha untouched
    shutil.copytree(toy["run"], m6)
    w = json.loads((m6 / "weights" / "index.json").read_text())
    w["entries"][0]["leaf"] = "00" + w["entries"][0]["leaf"][2:]
    (m6 / "weights" / "index.json").write_text(json.dumps(w))
    s6 = check(m6)  # no CommitError escapes; trusting the index -> the openings do not authenticate;
    assert set(s6) <= {"exact", "auth_failed"}  # rehashing the bytes -> the relation side is clean and the model binding (C3a) catches it
    if not recompute:
        assert s6.get("auth_failed", 0) > 0
    m12 = tmp_path / "m12"  # WA's M12: one bit of a stored weight value flipped, index untouched
    shutil.copytree(toy["run"], m12)
    e = json.loads((m12 / "weights" / "index.json").read_text())["entries"][0]
    with open(m12 / "weights" / "values.bin", "r+b") as f:
        f.seek(e["offset"] + 1)
        b = f.read(1)
        f.seek(e["offset"] + 1)
        f.write(bytes([b[0] ^ 1]))
    s12 = check(m12)
    assert s12.get("auth_failed", 0) > 0 and set(s12) <= {"exact", "auth_failed"}


def test_load_description_structure_then_relation_ids(toy):
    reader = CaptureReader(toy["run"])
    d = verify.load_description(reader, known_relations=())  # structure only (before the runtime relations are registered)
    assert d.nodes
    with pytest.raises(VB.StructuralFailure, match="description validation: node .*unknown relation"):
        verify.load_description(reader, known_relations={"nothing"}, description=d)


# --- verification evidence + render -------------------------------------------------------------------------


def test_verification_evidence_labels_and_audit_overlay():
    sec = {"challenge": {"mode": "security", "bindings": {"logprobs": "bound"}}, "worker": "WB", "result": "PASS"}
    assert rows.verification_evidence(sec)["label"] == rows.EVIDENCE_TRANSCRIPT and rows.logprobs_label(sec) == "bound"
    diag = {"challenge": {"mode": "diagnostic"}, "worker": "W1", "result": "PASS"}
    assert rows.verification_evidence(diag)["label"] == rows.EVIDENCE_DIAGNOSTIC and rows.logprobs_label(diag) == "unbound"
    assert rows.logprobs_label({}) == "unbound"  # a row from before the field: never upgraded
    audit = {"verification_evidence": {"label": rows.EVIDENCE_INDEPENDENT, "verifier": "WA", "report": "docs/verified-backend/audit-security-baseline.md"}}
    ev = rows.verification_evidence(sec, audit)
    assert ev["label"] == rows.EVIDENCE_INDEPENDENT and ev["verifier"] == "WA" and ev["report"].endswith("audit-security-baseline.md")
    with pytest.raises(rows.RowError):
        rows.verification_evidence(sec, {"verification_evidence": {"label": "verified by vibes"}})
    assert rows.result_label({"result": "NOT_RUN"}) == "NOT_RUN" and rows.result_label({"result": "PASS"}) == "PASS (diagnostic challenge)"


def test_render_shows_logprob_status_and_evidence(tmp_path):
    from verity_vllm import render

    src = Path(__file__).resolve().parents[1] / "results" / "B0-18b3c0ef5d91-20260907T1956Z"
    if not (src / "table1.jsonl").exists():
        pytest.skip("the small security row is not in results/")
    results = tmp_path / "results"
    old = results / "B0-18b3c0ef5d91-20260907T1956Z"
    old.mkdir(parents=True)
    shutil.copy(src / "table1.jsonl", old / "table1.jsonl")
    (old / "audit.json").write_text(json.dumps({"verification_evidence": {"label": rows.EVIDENCE_INDEPENDENT, "verifier": "WA",
                                                                          "report": "docs/verified-backend/audit-security-baseline.md"}}))
    row = json.loads((src / "table1.jsonl").read_text().splitlines()[0])
    new = results / "B0-x-derived"
    new.mkdir()
    row2 = dict(row)
    row2["experiment_id"] = "B0-x-derived"
    row2["challenge"] = {**row["challenge"], "bindings": {**row["challenge"]["bindings"], "logprobs": "bound"}}
    row2.pop("verification_evidence", None)
    rows.write_jsonl(new / "table1.jsonl", [row2])
    out, collected = render.render(results, tmp_path / "results.md")
    text = out.read_text()
    assert "| bindings | verification_evidence |" in text
    assert "logprobs: unbound | independently verified (WA: `docs/verified-backend/audit-security-baseline.md`)" in text
    assert "logprobs: bound | transcript-verified (security-mode staging)" in text
    assert not [i for i in collected.issues if "audit" in i]
