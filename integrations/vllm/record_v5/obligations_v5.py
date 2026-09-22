#!/usr/bin/env python
"""[A9 contract 2026-09-10] Selected VUs (Q, over J \\ F) -> proof obligations, under the contract (adapted from L1's
``scripts/rec_p1_obligations.py``; its ``ProgramLocator`` is imported unchanged).

Differences from L1/D4:

* the class of a selected VU comes from ``expressibility_v5.json`` (record_v5/coverage_v5: A5/6's table x the v5 population):
  ``checker`` -> ``provable (approved ELF)`` (a GemmCoordinate: located in the pinned program, operand/weight/output rows opened, one
  typed-obligation unit); everything else -> ``unsupported (no checker)`` / ``unsupported (not in coverage table)``.  NEVER
  ``exempt (charged)``: an unsupported selected VU of a RETAINED RU is recorded in ``obligations/unsupported_selected.json`` and makes the
  verifier's outcome FAIL (the check cannot be established; the prover did not exempt the RU) -- see verify_v5;
* no Input kinds exist in the population (inputs are not VUs); if one is selected the script REFUSES (``E_LEGACY_INPUTS_AS_VUS``);
* ``N_V`` per instance and the program identity come from the v5 population file, not ``maps.json``;
* cost classes: program build + locator + mapping are ``verifier_expected`` (class 7: the verifier's derivation of expected statements);
  openings + witness + statement bytes are ``prove_witness`` (class 6: proof generation, witness side) -- two costlog rows.

    python -m record_v5.obligations_v5 --record REC --population population.v5.json --expressibility REC/expressibility_v5.json --lp 32 --steps 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from record_v5 import POC, deps, freeze
from record_v5 import population_v5 as p5
from record_v5 import coverage_v5 as cov5

deps.install()

import rec_p1_obligations as L1  # noqa: E402  (scripts; read-only, imported)
from e2e.verifier_v3 import locator as vloc  # noqa: E402  (A3's verifier-side locator: the shared kind sets)
from e2e.coverage_v1.attn_locator import locate_attention as vloc_locate_attention  # noqa: E402

utc, sha256_file, ProgramLocator = L1.utc, L1.sha256_file, L1.ProgramLocator

def _is_gemm_unit(u: dict[str, Any]) -> bool:
    L = u.get("locate")
    return str(u["class"]).startswith("provable") and isinstance(L, dict) and "refused" not in L and "builder" not in L


PROVABLE = cov5.CLASS_PROVABLE  # == the expressibility class string: the verifier compares mapping.units.jsonl per unit (V5.prover_mapping)
UNSUPPORTED = cov5.CLASS_NO_CHECKER
ELEM_CHUNK = 50  # elementwise units per format-3 statement (integrator 23:05Z: chunks <= 50 units)
NOT_IN_TABLE = cov5.CLASS_NOT_IN_TABLE


def _pinned_identity(rec: Path) -> tuple[str | None, dict | None]:
    """[E7 D54/D57] (`family`, `attention`) of the record's protocol_instance.json program identity ((None, None) -> Serve_v1, legacy geometry)."""
    try:
        doc = json.loads((rec / "protocol_instance.json").read_text())
    except (OSError, ValueError):
        return None, None
    ident = (doc.get("program") or {}).get("identity") or {}
    fam = ident.get("family")
    return (fam if isinstance(fam, str) else None), (ident.get("attention") if isinstance(ident.get("attention"), dict) else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--population", required=True, help="population.v5.json (rows basis: gate intervals exist)")
    ap.add_argument("--expressibility", required=True, help="REC/expressibility_v5.json (record_v5.coverage_v5)")
    ap.add_argument("--case", default="B0")
    ap.add_argument("--lp", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--statement-format", type=int, choices=(2, 3), default=3)
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--only-chunks", type=int, default=0)
    ap.add_argument("--skip-program-digest", action="store_true")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    freeze.check(rec, "obligations")
    odir = rec / "obligations"
    odir.mkdir(exist_ok=True)
    sel = json.loads((rec / "chronology" / "selected.json").read_text())
    proto = json.loads((rec / "protocol.json").read_text())
    ru_map = {r["ru"]: r["request_id"] for r in proto["ru_map"]["rus"]}
    retained = set(proto.get("retained") or [])
    pop_doc = json.loads(Path(a.population).read_text())
    if pop_doc.get("schema") != p5.SCHEMA:
        raise SystemExit(f"E_CONFIG: {a.population} is not a {p5.SCHEMA} population")
    if pop_doc.get("basis_kind") != "rows":
        raise SystemExit("UNAVAILABLE: obligations need gate intervals; a rules-basis population (no plan at this shape) has none -- report as coverage, do not model")
    if pop_doc["content_sha256"] != proto["population_file_sha256"] and sha256_file(Path(a.population)) != proto["population_file_sha256"]:
        raise SystemExit("NOT BOUND: population file is not the one pinned by protocol.json")
    n_v = int(pop_doc["vu_count"])
    expr = json.loads(Path(a.expressibility).read_text())
    klass = {r["kind"]: r["class"] for r in expr["rows"]}
    T: dict[str, float] = {}

    # --- the pinned program (verifier side: class 7) ---
    t0 = time.perf_counter()
    from compact_whole_b0 import build_config  # scripts/
    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    C, _ = build_config(a.case, 0, [])
    fam, geom = _pinned_identity(rec)  # [E7 D54/D57] Serve_v1 (pre-D54 records) | Serve_v4 + the identity's declared geometry
    program = Program(b1.serve_program(C, a.lp, a.steps, family=fam, attention=geom))
    T["program_s"] = round(time.perf_counter() - t0, 2)
    prog_check = {"gates": program.gates, "input_gates": program.input_gates, "expected_gates": pop_doc["gates"], "expected_digest": pop_doc["program_digest"]}
    if program.gates != pop_doc["gates"]:
        raise SystemExit(f"NOT BOUND: program gates {program.gates} != population's {pop_doc['gates']}")
    if not a.skip_program_digest:
        from verity.ir import encode_program, program_digest

        t1 = time.perf_counter()
        d = program_digest(encode_program(program))
        prog_check["digest"] = d
        T["program_digest_s"] = round(time.perf_counter() - t1, 1)
        if d != pop_doc["program_digest"]:
            raise SystemExit(f"NOT BOUND: program digest {d} mismatch (population pins {pop_doc['program_digest']})")
    t1 = time.perf_counter()
    loc = ProgramLocator(program, C)
    T["locator_s"] = round(time.perf_counter() - t1, 2)
    print(f"[obligations_v5] program {program.gates} gates ({program.input_gates} input) built in {T['program_s']} s; locator {T['locator_s']} s", flush=True)

    # --- map every selected unit ---
    t1 = time.perf_counter()
    # [A3 ckpt 8 / CK-ELEM] ONE derivation on both sides: the prover maps with the verifier's own `locate_selected` (contract class
    # vocabulary; GemmCoordinate / attention-body locations; `provable (BUILDERS: <family>)` for CK-ELEM's elementwise kinds), so
    # V5.prover_mapping compares like with like.  Inputs are not VUs here (refused if selected).
    for row in sel["selected"]:
        if row["kind"]["fn"] == "Input":
            raise SystemExit("E_LEGACY_INPUTS_AS_VUS: an Input kind was selected; inputs are not VUs (the population must be population_v5)")
    units: list[dict[str, Any]] = vloc.locate_selected(loc, sel["selected"], klass=klass, ru_request=ru_map, n_v=n_v)
    for u in units:
        u["retained_ru"] = int(u["ru"]) in retained
    by_class: Counter = Counter(u["class"] for u in units)
    refused: list[dict[str, Any]] = [u for u in units if isinstance(u.get("locate"), dict) and "refused" in u["locate"]]
    # [E10 NIGHT 2026-09-12, G1 (b)] VERIFIER-COVERED kinds (`GatherBf16x<V>_v1|batch`, rule gather-equality/v1: E7 `verifier_covered_rule`, E9
    # `steps.gather_equality_check` = V6.gather in both entry points): no checker, no proof, no statement from the prover -- the verifier's own
    # authenticated openings establish the VU.  They are NOT unsupported selections; recorded by name in obligations/verifier_covered.json.
    from e2e.coverage_v1.builders import verifier_covered_rule as _vc_rule
    for u in units:
        r_ = _vc_rule(str(u.get("kind", "")))
        if r_ is not None:
            u["verifier_covered_rule"] = r_
    verifier_covered: list[dict[str, Any]] = [{k: u[k] for k in ("vu_index", "ru", "request_id", "kind", "class", "gate_interval", "retained_ru", "verifier_covered_rule")}
                                              for u in units if u.get("verifier_covered_rule")]
    unsupported_selected: list[dict[str, Any]] = [{k: u[k] for k in ("vu_index", "ru", "request_id", "kind", "class", "gate_interval", "retained_ru")}
                                                  for u in units if not str(u["class"]).startswith("provable") and not u.get("verifier_covered_rule")]
    T["map_s"] = round(time.perf_counter() - t1, 2)
    mapping = {"schema": "veritor/a9/mapping-v5/v1", "utc": utc(), "program": prog_check, "instances": proto["instances"], "n_v": n_v, "count": len(units), "by_class": dict(by_class),
               "refused": len(refused), "refused_examples": refused[:20], "unsupported_selected": len(unsupported_selected), "verifier_covered": len(verifier_covered), "timing": T,
               "gemm_ops": dict(Counter(f"{u['locate'].get('role')}#{u['locate'].get('gemm_index')}" for u in units if _is_gemm_unit(u)))}
    (rec / "mapping.json").write_text(json.dumps(mapping, indent=1, sort_keys=True, default=str) + "\n")
    (rec / "mapping.units.jsonl").write_text("".join(json.dumps(u, sort_keys=True) + "\n" for u in units))
    # Inputs are not VUs in the v5 population, so there are no verifier-side input checks; A3's reverify still reads the legacy file
    # (reverify.py:438, unconditional) -- an empty file states exactly that (W-A9-5: make it optional)
    (odir / "input_checks.jsonl").write_text("")
    (odir / "unsupported_selected.json").write_text(json.dumps({
        "schema": "veritor/a9/unsupported-selected/v1", "count": len(unsupported_selected), "retained_ru_count": sum(1 for x in unsupported_selected if x["retained_ru"]),
        "by_kind": dict(Counter(x["kind"] for x in unsupported_selected)), "units": unsupported_selected,
        "meaning": "selected VUs whose kind has no checker in this configuration and whose RU was NOT exempted: the verifier cannot establish their check; never 'exempt (charged)'"}, indent=1) + "\n")
    (odir / "verifier_covered.json").write_text(json.dumps({
        "schema": "veritor/e10/verifier-covered-selected/v1", "count": len(verifier_covered), "by_rule": dict(Counter(x["verifier_covered_rule"] for x in verifier_covered)),
        "units": verifier_covered, "meaning": "selected VUs of a kind the VERIFIER covers by its own rule (gather-equality/v1: V6.gather in reverify.py and verify_v5); "
                                              "the prover emits no statement for them; they are neither unsupported nor uncovered"}, indent=1) + "\n")
    print(f"[obligations_v5] mapped {len(units)} units in {T['map_s']} s: {dict(by_class)}; refused {len(refused)}; unsupported selected {len(unsupported_selected)}; "
          f"verifier-covered {len(verifier_covered)}", flush=True)
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "verifier: pinned program build + structural locator + expected positions of the selected VUs", "class": "verifier_expected",
                            "seconds": round(T["program_s"] + T.get("program_digest_s", 0.0) + T["locator_s"] + T["map_s"], 2), "batch": f"{len(units)} selected VUs", "label": "MEASURED",
                            "note": "class 7 (verifier derivation of expected statements): what the verifier computes from the fixed circuit and Q alone"}) + "\n")
    if a.map_only:
        return 0 if not refused else 1

    # --- open, build statements (prover side: witness; class 6) ---
    from verity_vllm import identity
    from verity_vllm.capture import CaptureReader
    from verity_vllm.commit import TreeCache, load_commitments
    from verity_vllm.description import Description
    from verity_vllm_adapter import obligations as O
    from verity_vllm_adapter.coordinate_check import coordinate_openings
    from veritor.interface import TypedObligationSet, to_statement
    from veritor.interface.binding import bound_public_values
    from veritor.interface.lowering import statement_bytes_v3
    from veritor.protocol.proofs.wire import encode_statement, statement_digest

    sel_dir = rec / "adapter" / "selective"
    reader = CaptureReader(sel_dir)
    desc = Description(reader.description)
    comm = load_commitments(reader)
    cache = TreeCache(reader)
    t1 = time.perf_counter()
    cid_index: dict[tuple[str, int, str], str] = {}
    lm_head_ops = {n.op for n in desc.nodes_of_type("lm_head")}
    for ru_r in reader.rus():
        for cid_s in ru_r.ids():
            c = identity.parse(cid_s)
            if c.kind == "value" and c.op and (c.op.endswith(".gemm") or c.op in lm_head_ops) and (c.port or 0) == 0 and c.coordinate is None:
                cid_index[(c.request_id, c.t, c.op)] = cid_s
    T["cid_index_s"] = round(time.perf_counter() - t1, 2)
    coord_units = []
    problems: list[dict[str, Any]] = []
    t1 = time.perf_counter()
    for u in units:
        L = u.get("locate")
        if not _is_gemm_unit(u):  # attention-body units carry a (t, layer, head) locate, elementwise units a builder; neither is a coordinate
            continue
        rid = u["request_id"]
        op = L["op_suffix"] and f"layers.{L['layer']}.{L['op_suffix']}"
        if op is None:
            op = next(iter(lm_head_ops))
        cid_s = cid_index.get((rid, L["t"], op))
        if cid_s is None:
            problems.append({**u, "problem": f"no committed output row for ({rid[:8]}, t={L['t']}, {op})"})
            continue
        try:
            co = coordinate_openings(desc, reader, comm, cache, cid_s, L["j"])
        except Exception as e:  # noqa: BLE001
            problems.append({**u, "problem": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        u["canonical_id"], u["j"] = cid_s, L["j"]
        coord_units.append((u, co))
    T["open_s"] = round(time.perf_counter() - t1, 2)
    print(f"[obligations_v5] opened {len(coord_units)} coordinates, {len(problems)} problems in {T['open_s']} s", flush=True)

    t1 = time.perf_counter()
    chunks: list[list[Any]] = []
    n_chunks = max(1, -(-len(coord_units) // a.chunk))
    for c in range(n_chunks):
        chunks.append(coord_units[c::n_chunks])
    statements = []
    for c, chunk in enumerate(chunks):
        if a.only_chunks and c >= a.only_chunks:
            break
        if not chunk:
            continue
        tc = time.perf_counter()
        export, record = O.build(comm, [co for _, co in chunk], [], profile_id=reader.profile_id, name=f"REC-V5-{a.case}-chunk{c:02d}",
                                 description=f"record V5 chunk {c}: {len(chunk)} GemmCoordinate units of the selected VUs (Q over J minus F) of the v5 population",
                                 source=str(rec / "chronology" / "selected.json"))
        obj = TypedObligationSet.from_json(record["object"])
        v2 = encode_statement(to_statement(obj))
        assert v2.hex() == record["statement"]
        cdir = odir / f"chunk{c:02d}"
        cdir.mkdir(exist_ok=True)
        (cdir / "witness.bin").write_bytes(bytes.fromhex(record["witness"]))
        if a.statement_format == 3:
            v3 = statement_bytes_v3(obj)
            assert v3.hex() == record["statement_v3"]
            (cdir / "statement.bin").write_bytes(v3)
            record["statement_v2_digest"] = record["statement_digest"]
            record["statement_digest"] = statement_digest(v3).hex()
            assert record["statement_digest"] == record["statement_v3_digest"]
            record["binding"] = {**record["binding"], "statement_digest": record["statement_digest"], "statement_format": 3,
                                 "public_values": bound_public_values(record["object"], statement_format=3).hex()}
            record["statement_format"] = 3
            v2 = v3
        else:
            (cdir / "statement.bin").write_bytes(v2)
            record["statement_format"] = 2
            record["binding"] = {**record["binding"], "statement_format": 2}
        assert statement_digest((cdir / "statement.bin").read_bytes()).hex() == record["statement_digest"]
        record["selected_vus"] = {"coordinates": [u["vu_index"] for u, _ in chunk], "inputs": []}
        record["expected_positions_source"] = [{"vu_index": u["vu_index"], "ru": u["ru"], "request_id": u["request_id"], "canonical_id": u["canonical_id"], "j": u["j"],
                                                "locate": {k: u["locate"].get(k) for k in ("role", "gemm_index", "layer", "op_suffix", "t", "j")}} for u, _ in chunk]
        (cdir / "composed.json").write_text(json.dumps(record, sort_keys=True) + "\n")
        statements.append({"chunk": c, "dir": str(cdir), "coordinates": len(chunk), "input_equations": 0, "obligations": len(obj.obligations), "leaves": len(obj.leaves),
                           "roots": [r.id for r in obj.roots], "h_O": record["digest"], "h_L": record["statement_digest"], "statement_bytes": len(v2), "statement_format": a.statement_format,
                           "witness_bytes": len(record["witness"]) // 2, "relations": record["relations"], "witness_verdict": record["witness_verdict"], "verdict": record["verdict"],
                           "sha256": {"statement.bin": sha256_file(cdir / "statement.bin"), "witness.bin": sha256_file(cdir / "witness.bin")}})
        print(f"[obligations_v5] chunk {c}: {len(chunk)} coords -> {len(obj.obligations)} obligations, statement {len(v2)} B, witness {len(record['witness']) // 2} B, "
              f"witness {record['witness_verdict']}, {round(time.perf_counter() - tc, 1)} s", flush=True)
    # [CK-ELEM, A3 ckpt 8] selected VUs of a kind whose registered builder derives an expectation: one format-3 statement per family per
    # chunk of ELEM_CHUNK units (further obligations/chunkNN dirs; proofs.json lists them); a refused unit is a problem row, never dropped.
    # RSS is logged per chunk (integrator 23:05Z).
    from e2e.coverage_v1 import ck_elem as CE
    from e2e.coverage_v1.builders import expectation_builders
    import resource as _res

    # [E10, EOD 2026-09-12] THE registry path (E7 interface 6 + E3 interface 5): every selected VU whose kind has a builder in E7's ONE
    # registry (`elem_builders.ALL_BUILDERS`: ck_elem's one-gate kinds, the FTZ/select families, the call kinds, the PV step) is resolved by
    # `elem_builders.resolve_unit` over a record handle whose word oracle is E3's `PositionOracle` (boundary rows, weights, prompt tokens,
    # program literals, and E2's `interior/<req>/<step>` leaves -- the same oracle the verifier derives expectations from) and whose opener
    # serves E2's interior leaves from the prover's trees; statements are packaged per family by `package_batches` (openings authenticated
    # against commitments.json).  The pre-EOD handle (`ck_elem.RecordHandle` over boundary rows only) refused every unit with an interior
    # operand -- attempt 8, 00:38Z: 235 x "is an interior value the record does not commit" although the record commits E2's tree.
    registry = None
    try:
        from e2e.coverage_v1 import elem_builders as EB
        from e2e.interior import tree as e2tree
        from e2e.verifier_v3 import expected as ex3
        from e2e.verifier_v3 import locator as vloc3
        from e2e.verifier_v3 import steps as vsteps
        if all(hasattr(EB, n) for n in ("open_record", "package_batches", "batches_document", "ALL_BUILDERS")):
            registry = EB
    except ImportError as e:  # an older tree: the pre-EOD path below, by name
        print(f"[obligations_v5] registry path unavailable ({type(e).__name__}: {e}); ck_elem boundary-row path", flush=True)
    kinds_with_builder = set(registry.ALL_BUILDERS) if registry is not None else set(expectation_builders())
    # [E10 NIGHT 2026-09-12, n2 09:14Z] the registry path RESOLVES E4's FTZ kinds too (`elem_builders.resolve_unit` -> `resolve_ftz_unit`, family
    # veritor.ftz-f32@1, tables pinned by E6 since wave 3), but they are not in `ALL_BUILDERS` (the one-gate elementwise dict) -> n2 refused its 21
    # ftz units "no registered builder" although E7's ONE registry (`builders._registry`) folds `ftz_builders.BUILDERS` in and E3's census counts
    # them covered.  The set of kinds the registry path can build is ALL_BUILDERS + ftz_kinds(); a kind the packaging cannot build still refuses by name.
    if registry is not None and hasattr(registry, "ftz_kinds"):
        try:
            kinds_with_builder |= set(registry.ftz_kinds())
        except Exception as e_ftz:  # noqa: BLE001
            print(f"[obligations_v5] ftz_kinds() unavailable ({type(e_ftz).__name__}: {e_ftz}); FTZ kinds refuse by name", flush=True)
    elem_units = [u for u in units if isinstance(u.get("locate"), dict) and "builder" in u["locate"] and u["kind"] in kinds_with_builder]
    # a unit the mapping placed under a BUILDERS family whose kind has NO registered builder today (attempt 8: the 18 `veritor.ftz-f32@1`
    # units -- E4's FTZ table kinds wait on D23's pinned `commitments.tables`) is a refusal BY NAME, never a silent drop: uncovered for V5.
    for u in units:
        if isinstance(u.get("locate"), dict) and "builder" in u["locate"] and u["kind"] not in kinds_with_builder:
            problems.append({**u, "problem": f"builder refused: {u['kind']}: no registered builder for this kind today (mapping named family "
                                             f"{u['locate'].get('builder')}; registry: {sorted(kinds_with_builder)[:3]}...); refused by name, left uncovered for V5",
                             "by_name": True})
    if any(p_.get("by_name") for p_ in problems):
        print(f"[obligations_v5] {sum(1 for p_ in problems if p_.get('by_name'))} unit(s) with no registered builder -> refused by name: "
              f"{sorted({p_['kind'] for p_ in problems if p_.get('by_name')})}", flush=True)
    T["elementwise_units"] = len(elem_units)
    T["elementwise_path"] = "E7 registry (elem_builders.package_batches) + E3 PositionOracle" if registry is not None else "ck_elem.RecordHandle (boundary rows only)"
    if elem_units and registry is not None:
        te = time.perf_counter()
        inst_doc = json.loads((rec / "protocol_instance.json").read_text())
        c2 = json.loads((sel_dir / "commitments.json").read_text())
        retained_rids = sorted({str(u["request_id"]) for u in elem_units})
        pinfo = inst_doc.get("program") or {}
        pprog, pC, pprob = vloc3.build_pinned_program(pinfo.get("identity") or {}, expected_digest=pop_doc["program_digest"],
                                                      expected_gates=(pinfo.get("gates") or {}).get("value"), builder=pinfo.get("builder"))
        if pprog is None or pprob:
            raise SystemExit(f"NOT BOUND: the pinned program did not build for the registry path: {list(pprob or [])[:2]}")
        if pprog.gates != program.gates:
            raise SystemExit(f"NOT BOUND: pinned program {pprog.gates} gates != obligations program {program.gates}")
        ploc = vloc3.make_locator(pprog, pC)
        fam0 = (inst_doc.get("epoch") or {}).get("family") or "veritor.tc-ampere-bf16@2"
        ci = ex3.coverage_inputs(sel_dir, c2, retained_request_ids=set(retained_rids), units=[], checks=[], family=fam0, recompute_leaves=True, config=pC,
                                 program=pprog, locator=ploc, population=pop_doc, interior_layout=vsteps.interior_layout_pin(inst_doc))
        positions = ci.get("positions")
        if positions is None:
            raise SystemExit(f"UNAVAILABLE: E3's PositionOracle did not build (tree problems {list(ci.get('tree_problems') or [])[:2]})")
        # E2's interior trees of the retained requests (the prover's, under <selective>/interior/<req>/<step>/) -> openings of their leaves
        irank: dict[str, tuple[Any, int]] = {}
        committed_ids: dict[str, list[str]] = {}
        for rid in retained_rids:
            for d in sorted((sel_dir / "interior" / rid).glob("*")) if (sel_dir / "interior" / rid).is_dir() else []:
                if (d / "index.json").exists():
                    t = e2tree.load_tree(d, recompute=False)
                    committed_ids[t.tree_id] = list(t.layout.ids)
                    for i, cid in enumerate(t.layout.ids):
                        irank[cid] = (t, i)

        def open_leaf(cid: str):
            hit = irank.get(cid)
            return hit[0].open(hit[1]) if hit is not None else cache.open(cid)

        # [E10 NIGHT 2026-09-12, n2b 09:20Z] the instance's PINS go to the handle by name: `commitments.literals` (E6 D36, rule prescribed-literal/v1;
        # without it the registry refused 15 units LITERAL_UNPINNED although V3.literal_table had PASSED on the same pin) and `commitments.tables`
        # (E6 D23 MUFU table roots; without them ck_ftz refused "no committed table tree in the record").  The handle rebuilds the literal table from
        # the ids and CHECKS it against the pinned roots; E2's TableWords serve the table rows from the pinned public tables.
        _cm = inst_doc.get("commitments") or {}
        handle = registry.open_record(pprog, c2, oracles=[positions], opener=open_leaf, C=pC, locator=ploc, committed_ids=committed_ids,
                                      profile_id=str(reader.profile_id), reader=reader, include_boundary_rows=False,
                                      tables=_cm.get("tables") or None, literals=_cm.get("literals") or None)
        max_units = {f: min(ELEM_CHUNK, int(n)) for f, n in getattr(registry, "MAX_UNITS_BY_FAMILY", {}).items()}
        # [E10 NIGHT 2026-09-12] The attempt-8 pre-screen that refused BY NAME every unit whose operands alias one committed word (x*x of the
        # RMSNorm mean square) is WITHDRAWN: E7's registry expresses those units itself under `aliased-body/v1` (D28/D30, identity-keyed; the
        # census reads all 419,904 aliased VUs `ok`), and `package_batches` records any unit it cannot build as a `refused` row with its reason
        # (never a family-wide exception any more).  Every located unit therefore goes to the registry; refusals come back by name below.
        buildable: list[dict[str, Any]] = list(elem_units)
        packaged = registry.package_batches(buildable, handle, max_units=max_units or ELEM_CHUNK, name=f"V5-{rec.name}", authenticate=True, transparent_check=True)
        by_vu = {int(u.get("vu_index", -1)): u for u in elem_units}
        for r in packaged.refused:
            problems.append({**by_vu.get(int(r["vu_index"]), {"vu_index": r["vu_index"], "kind": r["kind"], "request_id": r.get("request_id")}),
                             "problem": f"builder refused: {str(r['reason'])[:300]}"})
        T["registry_s"] = round(time.perf_counter() - te, 1)
        print(f"[obligations_v5] registry path: {len(elem_units)} units -> {len(packaged.batches)} batch(es) {packaged.by_family}, {len(packaged.refused)} refused, "
              f"interior trees {len(committed_ids)} ({sum(len(v) for v in committed_ids.values())} leaves), {T['registry_s']} s", flush=True)
        (odir / "registry_batches.json").write_text(json.dumps(registry.batches_document(packaged, sampled_vu_indices=[int(u.get("vu_index", -1)) for u in elem_units]),
                                                               indent=1, sort_keys=True) + "\n")
        for b in packaged.batches:
            c = len(statements)
            tc = time.perf_counter()
            fam, batch, built = b.family, list(b.built.units), b.built
            cdir = odir / f"chunk{c:02d}"
            cdir.mkdir(exist_ok=True)
            (cdir / "statement.bin").write_bytes(built.statement_bytes)
            (cdir / "witness.bin").write_bytes(built.witness_bytes)
            record = dict(built.record)
            record["statement_digest"] = statement_digest(built.statement_bytes).hex()
            record["binding"] = {"statement_digest": record["statement_digest"], "statement_format": 3,
                                 "public_values": bound_public_values(record["object"], statement_format=3).hex()}
            record["selected_vus"] = {"elementwise": [int(v) for v in b.vu_indices]}
            record["witness_verdict"] = record.get("witness_verdict") or CE.verify_transparent(built)
            record["registry"] = {"path": T["elementwise_path"], "batch": b.name, "kinds": list(b.kinds)}
            (cdir / "composed.json").write_text(json.dumps(record, sort_keys=True) + "\n")
            rss_mb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024)
            statements.append({"chunk": c, "dir": str(cdir), "elementwise": len(batch), "family": fam, "kinds": sorted(b.kinds),
                               "obligations": len(built.obj.obligations), "leaves": len(built.obj.leaves), "roots": [r.id for r in built.obj.roots],
                               "h_O": record["digest"], "h_L": record["statement_digest"], "statement_bytes": len(built.statement_bytes), "statement_format": 3,
                               "witness_bytes": len(built.witness_bytes), "witness_verdict": record["witness_verdict"], "verdict": "accept",
                               "seconds": round(time.perf_counter() - tc, 2), "max_rss_mb": round(rss_mb, 1),
                               "sha256": {"statement.bin": sha256_file(cdir / "statement.bin"), "witness.bin": sha256_file(cdir / "witness.bin")}})
            print(f"[obligations_v5] chunk {c}: {len(batch)} registry units ({fam}: {', '.join(sorted(b.kinds))[:120]}) -> {len(built.obj.obligations)} obligations, "
                  f"statement {len(built.statement_bytes)} B, witness {len(built.witness_bytes)} B, transparent {record['witness_verdict']}, max RSS {rss_mb:.0f} MB", flush=True)
        T["elementwise_s"] = round(time.perf_counter() - te, 1)
        T["elementwise_refused"] = sum(1 for p_ in problems if str(p_.get("problem", "")).startswith("builder refused"))
        with (rec / "costlog.jsonl").open("a") as f:
            f.write(json.dumps({"stage": "prover: registry statements (E7 elem_builders over E3 positions: resolve + build + transparent check, format 3)", "class": "prove_witness",
                                "seconds": T["elementwise_s"], "batch": f"{len(elem_units)} registry units in {len(packaged.batches)} chunk(s)", "label": "MEASURED",
                                "note": "class 6 witness side of the registry statements; the SP1 proof time is measured by rec_p1_prove"}) + "\n")
    elif elem_units:
        te = time.perf_counter()
        handle = CE.RecordHandle.open(sel_dir, program, C, loc)
        by_family: dict[str, list] = {}
        for u in elem_units:
            try:
                r = CE.resolve(u, handle)
            except CE.Refusal as e:
                problems.append({**u, "problem": f"builder refused: {str(e)[:200]}"})
                continue
            by_family.setdefault(r.spec.family, []).append(r)
        for fam, rs in sorted(by_family.items()):
            for c0 in range(0, len(rs), ELEM_CHUNK):
                c = len(statements)
                tc = time.perf_counter()
                batch = rs[c0:c0 + ELEM_CHUNK]
                built = CE.build_many(batch, handle, name=f"V5-{rec.name}-chunk{c:02d}-{fam}")
                cdir = odir / f"chunk{c:02d}"
                cdir.mkdir(exist_ok=True)
                (cdir / "statement.bin").write_bytes(built.statement_bytes)
                (cdir / "witness.bin").write_bytes(built.witness_bytes)
                record = dict(built.record)
                record["statement_digest"] = statement_digest(built.statement_bytes).hex()
                record["binding"] = {"statement_digest": record["statement_digest"], "statement_format": 3,
                                     "public_values": bound_public_values(record["object"], statement_format=3).hex()}
                record["selected_vus"] = {"elementwise": [r.vu_index for r in batch]}
                record["witness_verdict"] = CE.verify_transparent(built)
                (cdir / "composed.json").write_text(json.dumps(record, sort_keys=True) + "\n")
                rss_mb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024)
                statements.append({"chunk": c, "dir": str(cdir), "elementwise": len(batch), "family": fam, "kinds": sorted({r.kind for r in batch}),
                                   "obligations": len(built.obj.obligations), "leaves": len(built.obj.leaves), "roots": [r.id for r in built.obj.roots],
                                   "h_O": record["digest"], "h_L": record["statement_digest"], "statement_bytes": len(built.statement_bytes), "statement_format": 3,
                                   "witness_bytes": len(built.witness_bytes), "witness_verdict": record["witness_verdict"], "verdict": "accept",
                                   "seconds": round(time.perf_counter() - tc, 2), "max_rss_mb": round(rss_mb, 1),
                                   "sha256": {"statement.bin": sha256_file(cdir / "statement.bin"), "witness.bin": sha256_file(cdir / "witness.bin")}})
                print(f"[obligations_v5] chunk {c}: {len(batch)} elementwise units ({fam}) -> {len(built.obj.obligations)} obligations, statement {len(built.statement_bytes)} B, "
                      f"witness {len(built.witness_bytes)} B, transparent {record['witness_verdict']}, {round(time.perf_counter() - tc, 1)} s, max RSS {rss_mb:.0f} MB", flush=True)
        T["elementwise_s"] = round(time.perf_counter() - te, 1)
        T["elementwise_refused"] = sum(1 for p_ in problems if str(p_.get("problem", "")).startswith("builder refused"))
        with (rec / "costlog.jsonl").open("a") as f:
            f.write(json.dumps({"stage": "prover: elementwise statements (ck_elem: resolve + build + transparent check, format 3)", "class": "prove_witness",
                                "seconds": T["elementwise_s"], "batch": f"{len(elem_units)} elementwise units in {sum(1 for s_ in statements if 'family' in s_)} chunk(s)", "label": "MEASURED",
                                "note": "CK-ELEM families; separate row from the GemmCoordinate openings"}) + "\n")
    T["statements_s"] = round(time.perf_counter() - t1, 1)
    out = {"schema": "veritor/a9/obligations-v5/v1", "utc": utc(), "instance_digest": proto["instance_digest"], "selected": len(units), "by_class": dict(by_class),
           "mapped_refused": len(refused), "coordinates_opened": len(coord_units), "unsupported_selected": len(unsupported_selected),
           "unsupported_selected_retained": sum(1 for x in unsupported_selected if x["retained_ru"]), "problems": problems, "statements": statements, "timing": T,
           "provenance": {"label": "MEASURED", "script": "record_v5/obligations_v5.py (L1 rec_p1_obligations.ProgramLocator imported)", "program": prog_check,
                          "note": "no 'exempt (charged)' class exists; unsupported selections are recorded and fail verification unless their RU was exempted (F)"}}
    (odir / "obligations.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "prover: open committed rows for the selected coordinates + witness + statement bytes", "class": "prove_witness",
                            "seconds": round(T["cid_index_s"] + T["open_s"] + T["statements_s"], 2), "batch": f"{len(coord_units)} coordinates -> {len(statements)} statements", "label": "MEASURED",
                            "note": "class 6 (proof generation), witness side; the SP1 prove wall is a separate row"}) + "\n")
    hard = [p_ for p_ in problems if not p_.get("by_name")]
    out["refused_by_name"] = len(problems) - len(hard)
    (odir / "obligations.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[obligations_v5] DONE: {len(statements)} statements; {len(problems)} problems ({out['refused_by_name']} refused by name -> uncovered for V5); "
          f"unsupported selected {len(unsupported_selected)} ({out['unsupported_selected_retained']} in retained RUs); {T}")
    # rc 0 = every buildable statement was built; units refused BY NAME stay uncovered and are the verifier's to name at V5 (D23: no redraw).
    return 0 if not hard and not refused else 1


if __name__ == "__main__":
    sys.exit(main())
