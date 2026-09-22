"""The seams added for the composed end-to-end command (lane INT), CPU only, and ``verify_composed`` on
the preserved artefacts of one pod run.

* ``veritor.plans.verity_ir_adapter``: TA1's ``LayerPost`` bridged onto RA's plan IR keeps the gate
  count and validates (both partitions the command uses) on a tiny static configuration;
* ``verity_vllm_adapter.obligations.select_coordinate_units``: deterministic in the challenge digest,
  shares ``x`` rows and ``W[j]`` rows as declared;
* ``verity_vllm_adapter.weight_sources.checkpoint_row_words``: reads one row of a (concatenated) vLLM
  parameter from a safetensors file without torch;
* the composed export on a tiny v2 bundle (two coordinates of one GEMM row + a pinned weight word + a
  pinned boundary token): record accepted, statement re-lowers, witness accepted by the transparent
  backend, the composition report names the shared leaf;
* ``e2e.verify_composed`` on ``fixtures/e2e-composed/INT-B0-*``: program/plan/records rows re-verified,
  proofs' binding re-checked on the recorded public values (no host binary here), verdict agrees;
* IT's artefact-side negatives (``fixtures/it/int_negatives.py`` cases a / b1 / b2 / b3 / c, applied by
  ``e2e.artefact_negatives`` to temp copies of the committed @2 records): each refused at the ``records``
  row by the check that is *about* the edit -- coverage (IT-F1), placement (IT-F2) or arithmetic -- not only
  by the recorded proof's digest.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from _helpers import TINY_MANIFEST

POC = Path(__file__).resolve().parents[1]
ROOT = POC.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("veritor.interface")
pytest.importorskip("verity.ir")

from verity_vllm import commit, identity  # noqa: E402
from verity_vllm.capture import Boundary, CaptureReader, RunWriter, StepRecord  # noqa: E402
from verity_vllm.commit import TreeCache  # noqa: E402
from verity_vllm.description import Description  # noqa: E402
from verity_vllm_adapter import obligations as OB  # noqa: E402
from verity_vllm_adapter.weight_sources import checkpoint_row_words  # noqa: E402

from e2e.run_composed import layer_plans, plans_digest, static_configuration  # noqa: E402

FIXTURES = POC / "fixtures" / "e2e-composed"
TINY_CONFIG = {**static_configuration("B0"), "H": 16, "NH": 2, "KVH": 1, "D": 8, "I": 32, "GU": 64, "QKV": 32, "V": 64, "NL": 2}


# -- the verity.ir -> plan IR bridge and the two partitions -----------------------------------------


def test_layer_plans_validate_on_tiny_config() -> None:
    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    from veritor.plans.validate import validate_plan

    layer = Program(bind(b1.LayerPost, C=TINY_CONFIG))
    Q, c, partitions = layer_plans(TINY_CONFIG)
    assert c.size == layer.gates, "the bridge must keep the canonical layout (one gate per gate)"
    names = [n.fn.name for n in Q.root.nodes]
    assert names[:3] == ["Input", "Input", "Input"] and sum(n.startswith("Gemm") for n in names) == 3
    statuses = {}
    for tag, plan in partitions:
        v = validate_plan(c, plan)
        statuses[tag] = (v.status, len(plan.verification_units))
        assert v.status == "valid", (tag, v.codes, v.reasons[:2])
    assert statuses["P1"][1] == 3 + 5  # three Input batches + five function calls
    assert statuses["P2"][1] > statuses["P1"][1]  # every GEMM refined to its coordinates
    d1 = plans_digest(Q, partitions)
    assert d1 == plans_digest(*layer_plans(TINY_CONFIG)[::2]), "the plan digest is a function of the plans' content"
    assert d1 != plans_digest(*layer_plans({**TINY_CONFIG, "H": 32})[::2])


def test_gemm_plan_units_validate_and_name_the_layer_shapes() -> None:
    """INT3: a `gemm:<K>x<N>` unit is the bare `Gemm{K,N}` program (x | w | the coordinate batch) with P1 = one call and
    P2 = one VU per coordinate; the layer's four shapes are listed in call order; the layer unit's digest is unchanged."""
    from e2e.run_composed import layer_gemm_shapes, unit_plans, unit_program

    from veritor.plans.validate import validate_plan

    shapes = layer_gemm_shapes(TINY_CONFIG)
    assert shapes == ["gemm:16x32", "gemm:16x16", "gemm:16x64", "gemm:32x16"]
    Q, c, partitions = unit_plans(TINY_CONFIG, "gemm:32x16")
    assert c.size == unit_program(TINY_CONFIG, "gemm:32x16").gates
    assert [n.fn.name for n in Q.root.nodes][:2] == ["Input", "Input"] and Q.root.nodes[2].fn.name.startswith("GemmCoordinate")
    for tag, plan in partitions:
        v = validate_plan(c, plan)
        assert v.status == "valid", (tag, v.codes, v.reasons[:2])
    assert len(partitions[0][1].verification_units) == 2 + 1 and len(partitions[1][1].verification_units) == 2 + 16
    assert plans_digest(*unit_plans(TINY_CONFIG, "layer")[::2]) == plans_digest(*layer_plans(TINY_CONFIG)[::2])
    with pytest.raises(ValueError):
        unit_program(TINY_CONFIG, "attention")


def test_bridge_refuses_nothing_silently_on_b0_layer_shape() -> None:
    """The bridged B0 layer has the node forms the command's report lists (three Inputs, five calls)."""
    Q, _, partitions = layer_plans(static_configuration("B0"))
    forms = [(n.form, n.fn.name) for n in Q.root.nodes]
    assert [f for f, _ in forms[:3]] == ["batch"] * 3
    assert [n.split("_v")[0] for _, n in forms[3:]] == ["Gemm", "RMSNormFusedCuda", "Gemm", "SiluMul", "Gemm"]
    assert len(partitions[1][1].verification_units) == 3 + 5 - 3 + (576 + 3072 + 576)


# -- deterministic coordinate selection ----------------------------------------------------------


def _rows(n_ops: int = 2, rows_per_op: int = 3, N: int = 64) -> list[OB.SampledRow]:
    rows = []
    for o in range(n_ops):
        for r in range(rows_per_op):
            rows.append(OB.SampledRow(f"p=x/r=req-{r}/u=0/prefill/L{o}/op{o}/0", f"layers.{o}.mlp.down_proj.gemm", "gemm", "gemm.bf16", "prefill", str(o),
                                      f"req-{r}", N, ("certified",) if r == 0 else ("coverage",), "exact"))
    return rows


def test_select_coordinate_units_is_deterministic_and_shares_values() -> None:
    rows = _rows()
    ch = {"digest": "ab" * 32}
    a = OB.select_coordinate_units(rows, ch, rows_sharing_x=2, coords_per_row=2, ops_sharing_w=1)
    b = OB.select_coordinate_units(rows, ch, rows_sharing_x=2, coords_per_row=2, ops_sharing_w=1)
    assert a.to_json() == b.to_json()
    other = OB.select_coordinate_units(rows, {"digest": "cd" * 32}, rows_sharing_x=2, coords_per_row=2, ops_sharing_w=1)
    assert other.to_json() != a.to_json(), "seeded by the challenge digest"
    assert len(a.shared_x) == 2 and all(len(s["j"]) == 2 for s in a.shared_x), a.shared_x
    for s in a.shared_x:  # two coordinates of one row share its x
        assert sum(1 for cid, _ in a.units if cid == s["row"]) >= 2
    assert len(a.shared_w) == 1
    sw = a.shared_w[0]
    assert len({cid for cid in sw["rows"]}) == 2 and all((cid, sw["j"]) in set(a.units) for cid in sw["rows"])
    assert len(set(a.units)) == len(a.units)


def test_select_prefers_rows_the_verifier_accepted() -> None:
    rows = _rows(1, 3)
    rows[0] = OB.SampledRow(**{**rows[0].__dict__, "verifier_status": "mismatch"})
    sel = OB.select_coordinate_units(rows, {"digest": "01" * 32}, rows_sharing_x=2, coords_per_row=1, ops_sharing_w=0)
    assert rows[0].canonical_id not in {cid for cid, _ in sel.units}
    assert OB.select_coordinate_units([], {"digest": ""}).note


def test_select_draws_a_chain_row_first_when_the_sample_offers_one() -> None:
    """INT3: a row whose producer VU is itself sampled is drawn first (the output-feeds-input chain); the rest of the
    draw is still seeded by the challenge digest, and without candidates the selection is INT's unchanged."""
    rows = _rows(2, 3)
    ch = {"digest": "ab" * 32}
    target = rows[4]  # a coverage row: without the chain preference the first rows come from the certified sample
    cand = {target.canonical_id: {"producer": "p=x/r=req-1/u=0/prefill/L1/act/0", "producer_op": "act", "producer_samples": ["adjacent"], "producer_verifier_status": "exact"}}
    sel = OB.select_coordinate_units(rows, ch, rows_sharing_x=2, coords_per_row=2, ops_sharing_w=0, chain_candidates=cand)
    assert sel.shared_x[0]["row"] == target.canonical_id and "chain row" in sel.note
    assert len(sel.shared_x) == 2 and sel.shared_x[1]["row"] != target.canonical_id
    plain = OB.select_coordinate_units(rows, ch, rows_sharing_x=2, coords_per_row=2, ops_sharing_w=0, chain_candidates={})
    assert plain.to_json() == OB.select_coordinate_units(rows, ch, rows_sharing_x=2, coords_per_row=2, ops_sharing_w=0).to_json()
    assert OB.select_coordinate_units(rows, ch, rows_sharing_x=2, coords_per_row=2, ops_sharing_w=0, chain_candidates=cand).to_json() == sel.to_json()


# -- the checkpoint row reader --------------------------------------------------------------------


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs = []
    off = 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def test_checkpoint_row_words_reads_concatenated_rows_without_torch(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    q, k, v = (rng.integers(0, 1 << 16, size=(n, 8), dtype=np.uint16) for n in (4, 2, 2))
    _write_safetensors(tmp_path / "model.safetensors", {"model.layers.0.self_attn.q_proj.weight": q, "model.layers.0.self_attn.k_proj.weight": k,
                                                        "model.layers.0.self_attn.v_proj.weight": v, "model.layers.0.self_attn.o_proj.weight": q})
    data, src = checkpoint_row_words(tmp_path, "layers.0.self_attn.qkv_proj.weight", 5)  # row 5 of cat(q,k,v) = k[1]
    assert data == k[1].tobytes() and src["tensor"].endswith("k_proj.weight") and src["local_row"] == 1
    data, src = checkpoint_row_words(tmp_path, "layers.0.self_attn.o_proj.weight", 3)
    assert data == q[3].tobytes()
    with pytest.raises(Exception, match="sha256|differs|mismatch"):
        checkpoint_row_words(tmp_path, "layers.0.self_attn.o_proj.weight", 0, expected_file_sha256={"model.safetensors": "00" * 32})
    with pytest.raises(Exception):
        checkpoint_row_words(tmp_path, "layers.0.self_attn.qkv_proj.weight", 8)  # past the end


# -- the composed export on a tiny v2 bundle ------------------------------------------------------

K, N = 32, 8


def bf16_int(v: int) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16


def _description(profile_id: str) -> dict:
    return {
        "schema": "verity-vllm/description/v1", "profile_id": profile_id, "model": "toy", "symbols": {"H": K, "N": N},
        "nodes": [
            {"op": "layers.0.input_layernorm.rmsnorm", "layer": "0", "op_type": "rmsnorm", "dtype": "bfloat16", "shape": ["H"], "args": [],
             "relation": "unresolved", "origin": "captured", "decompositions": [{"id": "row", "interface": {"input_bytes": 0, "output_bits": 16 * K}}], "runtime_impl": "toy"},
            {"op": "layers.0.self_attn.o_proj.gemm", "layer": "0", "op_type": "gemm", "dtype": "bfloat16", "shape": ["N"],
             "args": [{"name": "x", "source": "node", "op": "layers.0.input_layernorm.rmsnorm", "port": 0}, {"name": "w", "source": "weight", "param": "layers.0.self_attn.o_proj.weight"}],
             "relation": "gemm.same_runtime.v1", "origin": "captured", "decompositions": [{"id": "row", "interface": {"input_bytes": 2 * K * (N + 1), "output_bits": 16 * N}}],
             "runtime_impl": "triton:vllm.model_executor.determinism.batch_invariant.matmul_persistent (via linear_batch_invariant)", "runtime_op": "verity.bi_mm",
             "module_path": "layers.0.self_attn.o_proj", "consts": {}, "arg_cols": {}},
        ],
    }


@pytest.fixture()
def tiny_bundle(tmp_path: Path) -> tuple[Path, np.ndarray, str]:
    rng = np.random.default_rng(3)
    x_int = rng.integers(-3, 4, size=(K,))
    w_int = rng.integers(-3, 4, size=(N, K))
    out_int = w_int @ x_int
    assert np.all(np.abs(out_int) < 256)
    x = np.array([bf16_int(int(v)) for v in x_int], np.uint16)
    w = np.array([[bf16_int(int(v)) for v in row] for row in w_int], np.uint16)
    out = np.array([bf16_int(int(v)) for v in out_int], np.uint16)
    run = RunWriter(tmp_path / "run", TINY_MANIFEST)
    pid = run.profile_id
    ru = run.ru("req-0", 0, phase="prefill", positions=[0, 1, 2])
    x_id = str(identity.value_id(pid, "req-0", 0, "prefill", "0", "layers.0.input_layernorm.rmsnorm", 0))
    out_id = str(identity.value_id(pid, "req-0", 0, "prefill", "0", "layers.0.self_attn.o_proj.gemm", 0))
    ru.add(x_id, "bfloat16", [K], x)
    ru.add(out_id, "bfloat16", [N], out)
    ru.finish()
    boundary = Boundary.build("req-0", [11, 12, 13], [7], {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1, "seed": None},
                              seed=None, finish_reason="stop", steps=[StepRecord(0, "prefill", (0, 1, 2))])
    run.write_boundary(boundary)
    ww = run.weights(store_values=True, format_version=2)
    ww.add("layers.0.self_attn.o_proj.weight", "bfloat16", [N, K], w, rows_per_leaf=1)
    ww.finish()
    run.write_description(_description(pid))
    run.finish()
    commit.build_commitments(CaptureReader(tmp_path / "run"))
    (tmp_path / "ckpt").mkdir()
    _write_safetensors(tmp_path / "ckpt" / "model.safetensors", {"model.layers.0.self_attn.o_proj.weight": w})
    return tmp_path, w, out_id


def test_composed_export_shares_x_and_pins_inputs(tiny_bundle) -> None:
    from veritor.interface import TypedObligationSet, check_binding, check_relations, statement_bytes, validate
    from veritor.interface.binding import bound_public_values

    root, w, out_id = tiny_bundle
    reader = CaptureReader(root / "run")
    description = Description(reader.description)
    commitments = json.loads((root / "run" / "commitments.json").read_text())
    cache = TreeCache(reader)
    challenge = {"digest": "77" * 32, "certified_sample": [out_id], "coverage_sample": [], "adjacent_vus": []}
    rows = OB.sampled_gemm_rows(challenge, description, reader, [{"canonical_id": out_id, "status": "exact"}])
    assert len(rows) == 1 and rows[0].N == N and rows[0].samples == ("certified",)
    sel = OB.select_coordinate_units(rows, challenge, rows_sharing_x=1, coords_per_row=2, ops_sharing_w=0)
    assert len(sel.units) == 2 and len(sel.shared_x) == 1
    units, problems = OB.open_units(description, reader, commitments, cache, sel)
    assert not problems and len(units) == 2
    assert units[0].x.canonical_id == units[1].x.canonical_id, "two coordinates of one row open the same x"
    weight_eqs, wreport = OB.input_equations(units, challenge, model_dir=root / "ckpt", expected_file_sha256=None, per_unit=2)
    assert len(weight_eqs) == 4 and all(r["source"]["tensor"].endswith("o_proj.weight") for r in wreport)
    prod_rows = [{"request_id": "req-0", "prompt_token_ids": [11, 12, 13], "output_token_ids": [7]}]
    boundary_eqs, breport = OB.boundary_input_equations(reader, commitments, cache, prod_rows, challenge, request_ids=["req-0"], prompt_per_request=1, outputs_per_request=1)
    assert len(boundary_eqs) == 4 and {r["field"] for r in breport} == {"prompt_token", "output_token"}
    assert all(r["expected"] == r["committed"] for r in breport)
    export, record = OB.build(commitments, units, weight_eqs + boundary_eqs, profile_id=reader.profile_id, name="tiny", description="tiny composed", source="test")
    assert record["verdict"] == "accept" and record["relations"] == "accept" and record["witness_verdict"] == "accept", {k: record.get(k) for k in ("verdict", "code", "detail", "relations", "witness_verdict")}
    obj = TypedObligationSet.from_json(record["object"])
    validate(obj)
    assert len(obj.obligations) == 2 + 4 + 4  # two coordinates, four weight pins, two tokens x two halves
    assert statement_bytes(obj).hex() == record["statement"]
    assert check_relations(obj, record["values"])
    comp = record["provenance"]["composition"]
    assert comp["shared_x_rows"] and len(comp["input_equations"]) == 8, comp
    pv = bound_public_values(obj)
    assert len(pv) == 65 and check_binding(obj, pv).ok
    assert not check_binding(obj, pv[:33]).ok
    # a wrong expected token -> the relation rejects (the Input equation's pin is the API's word, not the committed one)
    bad = [OB.InputEquation(boundary_eqs[0].leaf, boundary_eqs[0].offset, (boundary_eqs[0].expected + 1) & 0xFFFF, "wrong")]
    _, rec2 = OB.build(commitments, units, weight_eqs + bad, profile_id=reader.profile_id, name="tiny-bad", description="", source="test")
    assert rec2["relations"] == "reject" or rec2["witness_verdict"] != "accept"


def test_chain_is_realised_only_when_the_producer_is_sampled(tiny_bundle) -> None:
    """INT3: the GEMM row's ``x`` is the rmsnorm output.  With that VU in the sample the chain record names the link
    (the same committed leaf is the producer's checked output and the coordinate's input); without it, none."""
    root, _w, out_id = tiny_bundle
    reader = CaptureReader(root / "run")
    description = Description(reader.description)
    commitments = json.loads((root / "run" / "commitments.json").read_text())
    cache = TreeCache(reader)
    x_id = str(identity.value_id(reader.profile_id, "req-0", 0, "prefill", "0", "layers.0.input_layernorm.rmsnorm", 0))
    results = [{"canonical_id": out_id, "status": "exact"}, {"canonical_id": x_id, "status": "exact"}]
    with_producer = {"digest": "77" * 32, "certified_sample": [out_id], "coverage_sample": [x_id], "adjacent_vus": [x_id]}
    rows = OB.sampled_gemm_rows(with_producer, description, reader, results)
    producers = OB.sampled_producers(rows, description, reader, with_producer, results)
    assert set(producers) == {out_id} and producers[out_id]["producer"] == x_id and sorted(producers[out_id]["producer_samples"]) == ["adjacent", "coverage"]
    sel = OB.select_coordinate_units(rows, with_producer, rows_sharing_x=1, coords_per_row=2, ops_sharing_w=0, chain_candidates=producers)
    units, problems = OB.open_units(description, reader, commitments, cache, sel)
    assert not problems
    chain = OB.chain_record(sel, units, producers, results, with_producer)
    assert chain["realised"] and chain["candidates"] == 1 and len(chain["links"]) == 2  # both coordinates of the row share the producer's leaf
    assert all(l["x_id"] == x_id and l["producer_verifier_status"] == "exact" for l in chain["links"])
    without = {"digest": "77" * 32, "certified_sample": [out_id], "coverage_sample": [], "adjacent_vus": []}
    rows2 = OB.sampled_gemm_rows(without, description, reader, results)
    producers2 = OB.sampled_producers(rows2, description, reader, without, results)
    assert producers2 == {}
    chain2 = OB.chain_record(sel, units, producers2, results, without)
    assert not chain2["realised"] and chain2["candidates"] == 0 and "no sampled GEMM row's producer" in chain2["note"]


# -- the selective statement's third-party checks --------------------------------------------------


def test_selective_statement_checks_accept_j_and_refuse_interiors_outside_j(tiny_bundle) -> None:
    """INT3 (additive to RC's selective.py): WB's statement checks require every committed replay unit to have been
    declared before T_0; a selective statement declares none and the final commitments hold J's.  The selective
    variant accepts exactly that and refuses an interior outside J, a non-empty declaration, or a phase-0 record
    that changed."""
    from verity_vllm import selective
    from verity_vllm.capture import WEIGHTS_TREE_ID

    root, _w, _out = tiny_bundle
    reader = CaptureReader(root / "run")
    description = Description(reader.description)
    commitments = json.loads((root / "run" / "commitments.json").read_text())
    trees = commitments["trees"]
    boundary = {tid: {"root": r["root"], "leaf_count": r["leaf_count"], "ids_digest": r.get("ids_digest")} for tid, r in sorted(trees.items()) if r.get("kind") == "boundary"}
    weights = {k: trees[WEIGHTS_TREE_ID].get(k) for k in ("root", "leaf_count", "ids_digest", "weights_sha256")}
    statement = {"schema": selective.SELECTIVE_SCHEMA, "profile_id": reader.profile_id,
                 "manifest_digest": __import__("hashlib").sha256((root / "run" / "manifest.json").read_bytes()).hexdigest(),
                 "description_digest": description.digest().hex(), "required_digest": None, "boundary_roots": boundary, "weights_root": weights,
                 "replay_units_declared": [], "first_stage": {"mode": "selective", "requests": 1, "interiors_committed_before_r_q": 0},
                 "selective": {"population": "requests", "candidates": 1, "selection": {"kind": "fixed_size", "k": 1}}, "bindings": {}, "policy": {}, "source": {}}
    phase0 = {"trees": {tid: r for tid, r in trees.items() if r.get("kind") != "ru"}}
    common = dict(reader=reader, description=description, required_doc=None, commitments=commitments, production_sha256=None)
    assert selective.statement_checks(statement, selected=["req-0"], commitments_phase0=phase0, **common) == []
    outside = selective.statement_checks(statement, selected=[], commitments_phase0=phase0, **common)
    assert any("outside J" in p for p in outside), outside
    declared = selective.statement_checks({**statement, "replay_units_declared": ["ru/req-0/0"]}, selected=["req-0"], commitments_phase0=phase0, **common)
    assert any("not empty" in p for p in declared), declared
    changed = json.loads(json.dumps(phase0))
    next(iter(changed["trees"].values()))["root"] = "00" * 32
    moved = selective.statement_checks(statement, selected=["req-0"], commitments_phase0=changed, **common)
    assert any("phase-0 record" in p for p in moved), moved


# -- the second process on one run's preserved artefacts -------------------------------------------


def _preserved_runs() -> list[Path]:
    """The positive runs whose full artefact set (program, plan, records, proofs) is committed."""
    return sorted(p for p in FIXTURES.glob("INT-*") if (p / "program" / "program.json").exists()) if FIXTURES.exists() else []


def _preserved_negatives() -> list[Path]:
    """The negatives: only their summary / manifest / log are committed (the mutated bundles are on the pod)."""
    return sorted(p for p in FIXTURES.glob("INT-*-neg-*") if (p / "summary.json").exists()) if FIXTURES.exists() else []


@pytest.mark.parametrize("run_dir", _preserved_negatives() or [pytest.param(None, marks=pytest.mark.skip(reason="no preserved negative"))])
def test_preserved_negatives_fail_at_the_verifier(run_dir: Path) -> None:
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["verdict"] == "FAIL" and s["caught_by_stage"] == "verify" and s["scope"]["negative"] == run_dir.name.split("-neg-")[1]
    assert s["stages"]["mutation"]["status"] == "ok"  # the fault was injected by WA's labelled mutation
    blocker = s["stages"]["verify"]["first_blocker"]
    selective = s["scope"]["preparation"].startswith("RC selective")
    if selective:
        # the mutation is applied at its handshake point (recorded); a mutated description is refused before r_q
        # (structural, or the C3 architecture binding -- whichever the verifier reaches first), a flipped weight bit
        # fails authentication of J's openings after phase 2
        assert s["scope"]["negative_point"] == {"wa-m12": "READY_PHASE2", "wa-m8b": "READY_PHASE0"}[s["scope"]["negative"]]
        assert s["stages"]["mutation"]["applied_at"].startswith("handshake " + s["scope"]["negative_point"])
        expected = {"wa-m12": ("authentication",), "wa-m8b": ("structural", "binding")}[s["scope"]["negative"]]
    else:
        expected = {"wa-m12": ("authentication",), "wa-m8b": ("structural",)}[s["scope"]["negative"]]
    assert blocker.startswith(expected), blocker
    assert s["stages"]["prove"]["status"] == "skipped"  # nothing is proved when the verifier's row failed


@pytest.mark.parametrize("run_dir", _preserved_runs() or [pytest.param(None, marks=pytest.mark.skip(reason="no preserved run under fixtures/e2e-composed"))])
def test_verify_composed_on_preserved_artefacts(run_dir: Path, tmp_path: Path) -> None:
    from e2e import verify_composed

    summary = json.loads((run_dir / "summary.json").read_text())
    # the tarball and weights (hundreds of MB) are archived on the pod, not here: the bundle row is skipped;
    # no host binary here: the proofs' binding is re-checked on the recorded public values (row says so).
    # A B1 record's Serve program (410 MB descriptor, 6 GB to build) is never reconstructed on this machine
    # (int3-brief §4): `--program recorded` checks the preserved descriptor's sha256 / digest and rebuilds only the
    # layer program; the pod's second process (its own reverify.json, committed beside) did the reconstruction.
    program_mode = "rebuild" if summary["case"] == "B0" else "recorded"
    # [PROTO-A v3, 2026-09-09] these records predate the protocol instance: they verify under the explicit --historical-v2 path,
    # whose reverify document says it attests no instance. Without the flag the 'instance' row FAILs by design (brief A.4).
    rc = verify_composed.main(["--dir", str(run_dir), "--out", str(tmp_path), "--allow-recorded-proofs", "--skip-bundle", "--program", program_mode, "--plan", program_mode,
                               "--historical-v2"])
    doc = json.loads((tmp_path / "reverify.json").read_text())
    rows = doc["rows"]
    # [REC S4, user ruling 11:54 PT] the legacy path is UNCERTIFIED in every output: row status, top-level flag, certification_status
    assert rows["instance"]["status"] == "UNCERTIFIED" and doc["process"]["attests_protocol_instance"] is False
    assert doc.get("UNCERTIFIED") and doc.get("certification_status") == "UNCERTIFIED"
    if program_mode == "recorded":
        assert rows["program"]["status"] == "ok" and rows["program"]["program_reconstructed"] is False, rows["program"]
        assert rows["plan"]["status"] == "ok" and rows["plan"]["plan_revalidated"] is False, rows["plan"]
        assert rows["records"]["status"] == "ok" and rows["proofs"]["status"] == "ok"
        assert doc["fresh_verdict"] == summary["verdict"] == "PASS" and doc["agrees"] and rc == 0
        pod = run_dir / "reverify" / "reverify.json"  # the pod's second process: reconstruction, re-validation, host verification, bundle
        assert pod.exists(), "a B1 record is committed with the pod's own second-process reverify.json"
        pdoc = json.loads(pod.read_text())
        assert pdoc["fresh_verdict"] == "PASS" and pdoc["agrees"]
        assert pdoc["process"].get("program_reconstructed", True) is True and pdoc["process"].get("plan_revalidated", True) is True
        assert pdoc["rows"]["program"]["status"] == "ok" and pdoc["rows"]["program"]["digest"] == summary["stages"]["program"]["program_digest"]
        assert pdoc["rows"]["plan"]["status"] == "ok" and pdoc["rows"]["plan"]["plan_digest"] == summary["stages"]["plan"]["plan_digest"]
        return
    assert rows["plan"]["status"] == "ok" and all(p["status"] == "valid" for p in rows["plan"]["partitions"].values()), rows["plan"]
    assert rows["records"]["status"] == "ok" and rows["records"]["h_O"] == summary["stages"]["obligations"]["h_O"], rows["records"]
    assert rows["proofs"]["status"] == "ok" and all(m["binding_rechecked"]["ok"] for m in rows["proofs"]["modes"].values()), rows["proofs"]
    assert rows["proofs"]["modes"]["core"]["re_verified"] is False
    recorded_digest = summary["stages"]["program"]["program_digest"]
    problems = rows["program"]["problems"]
    # The record must be internally consistent whatever the current registry says B0 is.
    assert not any(p.startswith("preserved descriptor sha256") or p.startswith("preserved descriptor's digest") or p.startswith("preserved layer") for p in problems), problems
    if rows["program"]["digest"] == recorded_digest:
        assert rows["program"]["status"] == "ok", rows["program"]
        assert doc["fresh_verdict"] == summary["verdict"] == "PASS" and doc["agrees"] and rc == 0
    else:
        # Static-configuration drift: `INT-B0-20260908T0235Z` ran `Serve` with `RMSNormFusedCuda@1` (INT branched before
        # B1Q's `@2`, 7da4eda); the merged registry's B0 binds `@2`, so the program the code now calls B0 is not the program
        # this audit was over. The second process must say so — not by a bare digest mismatch but by DC's denotational
        # comparison of the preserved descriptor against the reconstruction, which names the one differing definition.
        # A fresh B0 record under `@2` (INT3 step A) flips this branch back to the one above; the old record is kept as is.
        assert rows["program"]["status"] == "FAIL", rows["program"]
        transition = rows["program"]["codec_transition"]["program"]
        assert transition["verdict"] == "MISMATCH" and "definition sets differ" in transition["detail"], transition
        assert "RMSNormFusedCuda_v1" in transition["detail"] and "RMSNormFusedCuda_v2" in transition["detail"], transition
        assert rows["program"]["gates"] == summary["stages"]["program"]["gates"] if "gates" in summary["stages"]["program"] else True
        assert doc["fresh_verdict"] == "FAIL" and summary["verdict"] == "PASS" and not doc["agrees"] and rc != 0


# --- IT's artefact-side negatives against the committed @2 records (int3-brief addendum; independent-tests.md §5) ---

AT2_RECORDS = [FIXTURES / "INT-B0-20260908T0725Z", FIXTURES / "INT-B1-20260908T0941Z"]
IT_CASES = ("a", "b1", "b2", "b3", "c")


def _at2_records() -> list[Path]:
    return [p for p in AT2_RECORDS if (p / "obligations" / "composed.json").exists()]


@pytest.mark.parametrize("run_dir", _at2_records() or [pytest.param(None, marks=pytest.mark.skip(reason="no @2 record under fixtures/e2e-composed"))])
def test_verify_composed_records_row_covers_the_selection_and_places_every_leaf(run_dir: Path, tmp_path: Path) -> None:
    """On the untouched records the two new checks hold: the re-derived selection (from the challenge digest and
    the verifier's rows) is covered by the object and equals the recorded selection.json; every leaf's id sits
    at its authenticated rank with position == rank; the boundary and weights roots are the verifier's."""
    from e2e import verify_composed

    rc = verify_composed.main(["--dir", str(run_dir), "--out", str(tmp_path), "--allow-recorded-proofs", "--skip-bundle", "--program", "recorded", "--plan", "recorded",
                               "--historical-v2"])  # pre-instance record (PROTO-A v3)
    rec = json.loads((tmp_path / "reverify.json").read_text())["rows"]["records"]
    assert rc == 0 and rec["status"] == "ok", rec["detail"]
    cov, pl = rec["coverage"], rec["placement"]
    assert cov["status"] == "ok" and cov["demanded_units"] == cov["covered_units"] > 0 and not cov["missing"] and not cov["extra_obligations_not_demanded"], cov
    assert cov["recorded_selection_agrees"] is True and cov["roots_checked_against_verifier"] == 3, cov  # two boundary trees + weights
    assert pl["status"] == "ok" and pl["leaves_checked"] == pl["leaves"] == rec["leaves"], pl
    assert "coverage" in rec["detail"] and "placement ok" in rec["detail"]


@pytest.mark.parametrize("case", IT_CASES)
@pytest.mark.parametrize("run_dir", _at2_records() or [pytest.param(None, marks=pytest.mark.skip(reason="no @2 record under fixtures/e2e-composed"))])
def test_it_artefact_negatives_are_refused_by_the_check_about_the_edit(run_dir: Path, case: str, tmp_path: Path) -> None:
    """IT's cases (`fixtures/it/int_negatives.py`) on a temp copy of the committed record, every digest re-derived, then
    the recorded-modes second process: the ``records`` row FAILs by the check that is *about* the edit, before (or
    without) the arithmetic and independently of the recorded proof's digest (which a re-proving adversary avoids).

    a  wrong result            -> relations / transparent backend RELATION_REJECTED (intended: arithmetic)
    b1 same-layout leaf swap   -> placement: the authenticated leaf at the claimed rank carries the other id (IT-F2)
    b2 different-layout swap   -> placement first, then the backend's INVALID_OPENING on the leaf size
    b3 position alone swapped  -> placement: position != rank; nothing else refuses (IT-F2)
    c  omitted obligation      -> coverage: the challenge-selected unit is named as missing (IT-F1)
    """
    from e2e import artefact_negatives as AN

    dst = tmp_path / case
    edit = AN.build(run_dir, case, dst)
    assert json.loads((run_dir / "obligations" / "composed.json").read_text())["digest"] != json.loads((dst / "obligations" / "composed.json").read_text())["digest"]
    sp = AN.second_process(dst)
    rec = sp["records"]
    assert rec["status"] == "FAIL" and sp["fresh_verdict"] == "FAIL" and sp["recorded_verdict"] == "PASS" and sp["exit"] != 0
    refused = AN.refused_by(rec)
    if case == "a":
        assert refused == "relations" and rec["coverage"] == "ok" and rec["placement"] == "ok"
        assert rec["relations"] == "reject" and rec["witness_verdict"] == "reject:RELATION_REJECTED"
    elif case in ("b1", "b2", "b3"):
        assert refused == "placement" and rec["coverage"] == "ok", rec["problems"]
        ids = list(edit["leaves"])
        assert all(any(i in p for i in ids) for p in rec["placement_problems"]) and len(rec["placement_problems"]) == 2, rec["placement_problems"]
        if case == "b1":
            assert all("the authenticated leaf at that rank is" in p for p in rec["placement_problems"])
            assert rec["witness_verdict"] == "reject:RELATION_REJECTED"  # arithmetic still refuses it, after placement did
        elif case == "b2":
            assert all("the authenticated leaf at that rank is" in p for p in rec["placement_problems"])
            assert rec["witness_verdict"] == "reject:INVALID_OPENING"  # the size check IT saw, now second
        else:
            assert all("position" in p and "but rank" in p for p in rec["placement_problems"])
            assert rec["relations"] == "accept" and rec["witness_verdict"] == "accept"  # IT-F2: nothing else sees it
            assert all(p.startswith("placement:") for p in rec["problems"])
    else:
        assert refused == "coverage" and rec["placement"] == "ok" and rec["relations"] == "accept" and rec["witness_verdict"] == "accept"
        removed = edit["removed_obligation"]["out"][0]  # "<row canonical id>[j]"
        row_id, j = removed[:removed.rindex("[")], removed[removed.rindex("[") + 1:-1]
        assert rec["coverage_missing"] == [f"{row_id}[j={j}]"], (rec["coverage_missing"], removed)
        assert f"{row_id}[j={j}]" in rec["detail"]  # the records row names the missing unit
    # the recorded proof's digest refusal is still there (E_OBJECT_DIGEST_MISMATCH) but is no longer the only barrier
    assert sp["proofs"]["status"] == "FAIL"
