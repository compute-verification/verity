#!/usr/bin/env python
# ============================================================================================================================
# LEGACY (DIRECTIVE v4) — NOT THE CONTRACT PATH.  Kept only to reproduce the REC-P1 historical record.
#   * Assigns the class "exempt (charged)" to selected VUs of an inexpressible kind and counts them into B_ex — the contract's
#     exemption is a selected RU in F under an RU-count budget, never a VU class (docs/PROTOCOL.md; PROTOCOL-CONFORMANCE R8).
#   * Treats selected Input gates as VUs with "input unit (verifier)" checks — the contract has no input VUs (R2).
#   * Writes provenance["units"], which the legacy verifier's V5 then trusts (R4, A2's M1).
# Refuses to run unless --legacy-directive-v4 is passed (R14).  With the flag, behaviour is unchanged.  Contract path:
# record_v5/coverage_v5.py (expressibility -> unsupported kinds), verifier_v3 + src/veritor/interface/expected.py (A2).  [A10 2026-09-10]
# ============================================================================================================================
"""[L1 RECORD 2026-09-10] Selected VUs of the VALIDATED P1/cu=whole partition -> their proof obligations / verifier checks (DIRECTIVE v3 §1.1 step 6).

LEGACY (DIRECTIVE v4): see the banner above; requires ``--legacy-directive-v4``.

Every row of ``chronology/selected.json`` is one unit of the pinned program's coordinate-level partition (a gate interval).  This script
locates each selected unit IN THE PROGRAM (the verity.ir ``Serve`` circuit of the certified B0 program; the root calls are named
structurally by ``verity_vllm.correspondence.program.StructuralCircuitMap``, never by annotation) and, by the expressibility table's class
of its kind:

* ``provable (approved ELF)`` -- a ``GemmCoordinate`` call: the enclosing ``Gemm`` call names the production op (qkv / o / gate_up /
  down / lm_head) at position t of request r (the RU), the member index is the output coordinate j; the committed operand row x,
  weight row W[j,:] and output row are opened (``verity_vllm_adapter.coordinate_check.coordinate_openings``) and the coordinate becomes one
  ``typed-obligation/v0`` unit of a statement proved under the approved guest;
* ``input unit (verifier)`` -- an ``Input`` gate: a prescribed value (per-RU-copy formulation).  A weight gate is located in the
  program's weights collection -> (weights leaf, word offset); the check opens the prover's weights tree at that leaf (root == the
  registered weight commitment, asserted by rec_p1_protocol) and compares the word with the pinned checkpoint's word; a prompt-token
  gate opens the request commitment's ``prompt_token[t]`` leaf and compares with the served prompt.  Both are ALSO written into the
  statement as ``Input`` equations, so the proof binds them; the verifier re-checks them independently;
* ``exempt (charged)`` -- counted into B_ex; nothing opened.

The mapping is checked, never assumed: a GemmCoordinate unit whose gate interval is not exactly one ``GemmCoordinate`` member of a
``Gemm`` call, or an Input gate in no collection, is a REFUSAL recorded per unit (and the record's Bound is NOT BOUND if any unit of a
provable/input class cannot be mapped).  Statements are chunked (``--chunk``) so each proof stays within the guest's memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# [A10 2026-09-10] R14 legacy guard: this script is DIRECTIVE-v4 legacy and refuses to run unless the flag is passed explicitly.
LEGACY_FLAG = "--legacy-directive-v4"
LEGACY_POINTER = (
    f"{Path(__file__).name}: LEGACY (DIRECTIVE v4) — not the contract path; refusing to run without {LEGACY_FLAG}.\n"
    "  'exempt (charged)' VU classes counted into B_ex and Input gates as VUs are not in the contract (docs/PROTOCOL.md R2/R8).\n"
    "  Contract path: record_v5/coverage_v5.py + e2e/verifier_v3 + src/veritor/interface/expected.py.\n"
    f"  Historical reproduction of the REC-P1 record: re-run with {LEGACY_FLAG} (behaviour is otherwise unchanged)."
)


def legacy_gate(argv: list[str] | None) -> list[str] | None:
    """Return ``argv`` without the legacy flag, or ``None`` (after printing the pointer) when it is absent.  ``-h/--help`` pass."""
    args = list(sys.argv[1:] if argv is None else argv)
    if LEGACY_FLAG in args:
        return [x for x in args if x != LEGACY_FLAG]
    if any(x in ("-h", "--help") for x in args):
        return args
    print(LEGACY_POINTER, file=sys.stderr)
    return None


# [A3, integrator 19:50Z] ``ProgramLocator`` / ``GEMM_OPS`` moved to ``e2e.verifier_v3.locator`` (shared: the verifier RUNS the locator
# over its own program build instead of reading this script's ``mapping.units.jsonl``).  Legacy pointers kept for the record scripts.
from e2e.verifier_v3.locator import GEMM_OPS, ProgramLocator  # noqa: E402,F401


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    argv = legacy_gate(argv)
    if argv is None:
        return 2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(LEGACY_FLAG, action="store_true", help="REQUIRED: acknowledge this is the DIRECTIVE-v4 legacy path (see banner)")
    ap.add_argument("--record", required=True)
    ap.add_argument("--instance-dir", required=True)
    ap.add_argument("--expressibility", required=True, help="out/record/expressibility.json (kind -> class)")
    ap.add_argument("--case", default="B0")
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--chunk", type=int, default=4000, help="max selected units (coordinates + input equations) per statement")
    ap.add_argument("--statement-format", type=int, choices=(2, 3), default=3, help="3 = packed position rows (what the approved guest reads; ~100x smaller than 2)")
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--only-chunks", type=int, default=0, help="build only the first n chunks (timing runs)")
    ap.add_argument("--skip-program-digest", action="store_true")
    a = ap.parse_args(argv)
    rec = Path(a.record)
    odir = rec / "obligations"
    odir.mkdir(exist_ok=True)
    sel = json.loads((rec / "chronology" / "selected.json").read_text())
    proto = json.loads((rec / "protocol.json").read_text())
    ru_map = {r["ru"]: r["request_id"] for r in proto["ru_map"]["rus"]}
    maps = json.loads((rec / "maps.json").read_text())
    expr = json.loads(Path(a.expressibility).read_text())
    klass = {r["kind"]: r["class"] for r in expr["rows"]}
    n_v = int(maps["plan"]["vus"]) if "plan" in maps and "vus" in maps["plan"] else None
    T: dict[str, float] = {}

    # --- the pinned program ---
    t0 = time.perf_counter()
    from compact_whole_b0 import build_config  # scripts/
    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    C, _ = build_config(a.case, 0, [])
    fam, geom = _pinned_identity(rec)  # [E7 D54/D57] Serve_v1 (pre-D54 records) | Serve_v4 + the identity's declared geometry
    program = Program(b1.serve_program(C, a.lp, a.steps, family=fam, attention=geom))
    T["program_s"] = round(time.perf_counter() - t0, 2)
    prog_check = {"gates": program.gates, "input_gates": program.input_gates, "expected_gates": maps["program"]["gates"], "expected_digest": maps["program"]["digest"]}
    if program.gates != maps["program"]["gates"]:
        raise SystemExit(f"NOT BOUND: program gates {program.gates} != population's {maps['program']['gates']}")
    if not a.skip_program_digest:
        from verity.ir import encode_program, program_digest

        t1 = time.perf_counter()
        d = program_digest(encode_program(program))
        prog_check["digest"] = d
        T["program_digest_s"] = round(time.perf_counter() - t1, 1)
        if d != maps["program"]["digest"]:
            raise SystemExit(f"NOT BOUND: program digest {d} mismatch (population pins {maps['program']['digest']})")
    t1 = time.perf_counter()
    loc = ProgramLocator(program, C)
    T["locator_s"] = round(time.perf_counter() - t1, 2)
    if n_v is None:
        n_v = int(expr["vus_per_request"])
    print(f"[rec_p1_obligations] program {program.gates} gates ({program.input_gates} input) built in {T['program_s']} s; locator {T['locator_s']} s; "
          f"prompt gates {loc.prompt_range}; {len(loc.w_ranges)} weight collections", flush=True)

    from e2e.coverage_v1.builders import BUILDERS, BuildRefusal, LocatedUnit, dispatch  # [A5/6 patch]

    # --- map every selected unit ---
    t1 = time.perf_counter()
    # [CK-ELEM dispatch, re-targeted by A3 22:20Z] ONE derivation on both sides: the prover maps the selected rows with the same
    # `locate_selected` the verifier runs (e2e.verifier_v3.locator) -- contract class vocabulary (W-A9-6; the table's row kept in
    # `expressibility_class`), GemmCoordinate / Input / attention-body locations, and `provable (BUILDERS: <family>)` for every kind whose
    # registered builder derives an expectation (CK-ELEM's four elementwise kinds), so the verifier's `V5.prover_mapping` compares like with like.
    from e2e.verifier_v3.locator import locate_selected

    units: list[dict[str, Any]] = locate_selected(loc, sel["selected"], klass=klass, ru_request=ru_map, n_v=n_v)
    by_class: Counter = Counter(u["class"] for u in units)
    refused: list[dict[str, Any]] = [u for u in units if isinstance(u.get("locate"), dict) and "refused" in u["locate"]]
    T["map_s"] = round(time.perf_counter() - t1, 2)
    mapping = {"schema": "veritor/rec/p1-mapping/v1", "utc": utc(), "program": prog_check, "instances": proto["instances"], "n_v": n_v, "count": len(units), "by_class": dict(by_class),
               "refused": len(refused), "refused_examples": refused[:20], "timing": T,
               "gemm_ops": Counter(f"{u['locate'].get('role')}#{u['locate'].get('gemm_index')}" for u in units if u["class"].startswith("provable") and "refused" not in u["locate"]),
               "input_kinds": Counter(u["locate"].get("kind", "refused") for u in units if u["class"].startswith("input")),
               "input_fields": Counter((u["locate"].get("field") or "").split("].")[-1] for u in units if u["class"].startswith("input") and u["locate"].get("kind") == "weight")}
    (rec / "mapping.json").write_text(json.dumps(mapping, indent=1, sort_keys=True, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)) + "\n")
    (rec / "mapping.units.jsonl").write_text("".join(json.dumps(u, sort_keys=True) + "\n" for u in units))
    print(f"[rec_p1_obligations] mapped {len(units)} units in {T['map_s']} s: {dict(by_class)}; refused {len(refused)}; gemm ops {dict(mapping['gemm_ops'])}; "
          f"inputs {dict(mapping['input_kinds'])}", flush=True)
    if refused:
        for r in refused[:5]:
            print("  REFUSED", r["kind"], r["gate_interval"], r["locate"]["refused"])
    if a.map_only:
        return 0 if not refused else 1

    # --- open, check, build statements ---
    import numpy as np
    from verity_vllm import identity
    from verity_vllm.capture import CaptureReader
    from verity_vllm.commit import Opening, TreeCache, load_commitments
    from verity_vllm.description import Description
    from verity_vllm.typed_obligation_export import InputEquation
    from verity_vllm_adapter import obligations as O
    from verity_vllm_adapter import profile as vprofile
    from verity_vllm_adapter.coordinate_check import coordinate_openings
    from verity_vllm_adapter.weight_sources import hf_tensors
    from veritor.interface import TypedObligationSet, to_statement
    from veritor.interface.binding import bound_public_values
    from veritor.interface.lowering import statement_bytes_v3
    from veritor.protocol.proofs.wire import encode_statement, statement_digest

    sel_dir = rec / "adapter" / "selective"
    reader = CaptureReader(sel_dir)
    desc = Description(reader.description)
    comm = load_commitments(reader)
    cache = TreeCache(reader)
    prod = {r["request_id"]: r for r in (json.loads(l) for l in (rec / "adapter" / "production" / "production.jsonl").read_text().splitlines() if l.strip())}
    cp = vprofile.checkpoint_for(a.case, a.checkpoints)
    registered = json.loads((rec / "weights_authorized.json").read_text())
    reg_leaf = {e["canonical_id"]: e for e in registered["leaf_table"]}
    # capture ids of committed GEMM outputs: (request, t, op) -> canonical id
    t1 = time.perf_counter()
    cid_index: dict[tuple[str, int, str], str] = {}
    lm_head_ops = {n.op for n in desc.nodes_of_type("lm_head")}
    for ru_r in reader.rus():
        for cid_s in ru_r.ids():
            c = identity.parse(cid_s)
            if c.kind == "value" and c.op and (c.op.endswith(".gemm") or c.op in lm_head_ops) and (c.port or 0) == 0 and c.coordinate is None:
                cid_index[(c.request_id, c.t, c.op)] = cid_s
    T["cid_index_s"] = round(time.perf_counter() - t1, 2)
    print(f"[rec_p1_obligations] {len(cid_index)} committed GEMM output rows indexed ({T['cid_index_s']} s); lm_head ops {sorted(lm_head_ops)}", flush=True)

    # the pinned checkpoint, read once without torch (safetensors header + raw bytes); the file sha256 is checked once against the manifest
    import struct

    st_path = Path(cp["local_path"]) / "model.safetensors"
    st_sha = sha256_file(st_path.resolve())
    if (cp.get("safetensors_sha256") or {}).get("model.safetensors") not in (None, st_sha):
        raise SystemExit(f"NOT BOUND: checkpoint sha256 {st_sha} != manifest {cp.get('safetensors_sha256')}")
    with st_path.open("rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        st_header = json.loads(f.read(hlen))
    st_base = 8 + hlen
    tensor_bytes: dict[str, bytes] = {}

    def checkpoint_word(cid: str, offset: int) -> tuple[int | None, str]:
        param = cid[2:]
        hf = hf_tensors(param)
        if hf is None:
            return None, "derived table (not a checkpoint tensor): leaf hash compared with the registered leaf table only"
        if cid not in tensor_bytes:
            parts = []
            with st_path.open("rb") as f:
                for name in hf[0]:
                    s0, e0 = st_header[name]["data_offsets"]
                    assert st_header[name]["dtype"] == "BF16", (name, st_header[name]["dtype"])
                    f.seek(st_base + s0)
                    parts.append(f.read(e0 - s0))
            tensor_bytes[cid] = b"".join(parts)
        b = tensor_bytes[cid]
        return int.from_bytes(b[2 * offset:2 * offset + 2], "little"), f"checkpoint:model.safetensors@{st_sha[:12]}:{'+'.join(hf[0])}[{offset}]"

    # [A5/6 patch] dispatch every located attention unit through BUILDERS[kind].build(located_unit, record): a Built (statement +
    # witness + expectation) per unit, or a BuildRefusal naming the uncommitted interior word -- recorded per unit, never silent
    from e2e.coverage_v1.capture_record import CaptureRecord

    record = CaptureRecord(sel_dir, comm, config=C, session=bytes.fromhex(proto["session"]) if "session" in proto else b"\0" * 32,
                           compiled=bytes.fromhex(maps["program"]["digest"]))
    built_attention: list[tuple[dict[str, Any], Any]] = []
    for u in units:
        if u["kind"] in BUILDERS and u.get("locate") and "refused" not in u["locate"] and "builder" not in u["locate"]:  # elem units: ck_elem below
            L = u["locate"]
            lu = LocatedUnit(str(u["vu_index"]), u["kind"], u["request_id"], int(L["t"]), layer=int(L["layer"]), head=int(L["head"]), ru_index=int(u["ru"]),
                             extra={"unit": L.get("unit"), "chain": L.get("chain", [])})
            try:
                built_attention.append((u, dispatch(lu, record)))
            except BuildRefusal as e:
                u["build_refused"] = e.reason
                u["missing_words"] = [list(m) for m in e.missing]
    T["attention_builds"] = len(built_attention)
    T["attention_refusals"] = sum(1 for u in units if "build_refused" in u)
    (rec / "mapping.units.jsonl").write_text("".join(json.dumps(u, sort_keys=True) + "\n" for u in units))
    # each Built is written as its own raw format-3 statement batch (fa2 family; not a typed object -- Q-A56-4) beside the GEMM chunks
    for k, (u, b) in enumerate(built_attention):
        from veritor.protocol.proofs.packed import encode_packed_witness
        from veritor.protocol.proofs.wire import encode_statement_v3

        (odir / f"attn-{k:05d}.statement.bin").write_bytes(encode_statement_v3(b.statement))
        (odir / f"attn-{k:05d}.witness.bin").write_bytes(encode_packed_witness(b.witness))
        (odir / f"attn-{k:05d}.json").write_text(json.dumps({"vu_index": u["vu_index"], "kind": b.kind, "notes": b.notes, "programs": b.programs}, indent=1))

    coord_units = []
    eqs_by_unit: list[tuple[dict[str, Any], list[InputEquation]]] = []
    checks: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []
    t1 = time.perf_counter()
    for u in units:
        L = u.get("locate")
        if not L or "refused" in L:
            continue
        rid = u["request_id"]
        if u["class"].startswith("provable"):
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
        elif L["kind"] == "weight":
            cid = L["canonical_id"]
            w = reader.weights
            ent = w.entry(cid)
            width = 4 if ent.dtype == "float32" else 2
            if ent.layout.kind == "rows":
                K = int(np.prod(ent.shape[1:]))
                row, k = divmod(L["offset"], K)
                leaf = w.leaf_id_for_row(cid, row)
            else:
                leaf, k = cid, L["offset"]
            try:
                o = cache.open(leaf)
            except Exception as e:  # noqa: BLE001
                problems.append({**u, "problem": f"{leaf}: {type(e).__name__}: {str(e)[:160]}"})
                continue
            if not isinstance(o, Opening):
                problems.append({**u, "problem": f"{leaf}: not a single leaf opening"})
                continue
            words = np.frombuffer(o.value, dtype="<u2" if width == 2 else "<u4").reshape(-1)
            committed = int(words[k])
            expected, src = checkpoint_word(cid, L["offset"]) if width == 2 else (None, "float32 derived table: leaf hash compared with the registered leaf table only")
            leaf_ok = reg_leaf.get(cid, {}).get("tensor_sha256") == ent.sha256
            ok = leaf_ok and (expected is None or expected == committed)
            checks.append({"vu_index": u["vu_index"], "ru": u["ru"], "kind": "weight", "leaf": leaf, "word": k, "committed": committed, "expected": expected, "source": src,
                           "registered_leaf_sha256_match": leaf_ok, "ok": bool(ok)})
            if expected is not None:
                prefix: tuple[int, ...] = (0,) if len(o.shape) == 2 and int(o.shape[0]) == 1 else ()
                eqs_by_unit.append((u, [InputEquation(o, (*prefix, k), expected, src)]))
            else:
                eqs_by_unit.append((u, []))
        elif L["kind"] == "prompt_token":
            t = L["t"]
            rec_row = prod[rid]
            prompt = list(rec_row.get("prompt_token_ids") or reader.boundary(rid).prompt_token_ids)
            if t >= len(prompt):
                problems.append({**u, "problem": f"prompt position {t} beyond the served prompt ({len(prompt)} tokens)"})
                continue
            cid_b = str(identity.boundary_id(reader.profile_id, rid, "prompt_token", t))
            o = cache.open(cid_b)
            assert isinstance(o, Opening)
            expected = int(prompt[t]) & 0xFFFFFFFF
            committed = int.from_bytes(o.value, "little", signed=True) & 0xFFFFFFFF
            checks.append({"vu_index": u["vu_index"], "ru": u["ru"], "kind": "prompt_token", "leaf": cid_b, "t": t, "committed": committed, "expected": expected,
                           "source": f"production.jsonl:{rid}:prompt_token[{t}]", "ok": committed == expected})
            eqs_by_unit.append((u, [InputEquation(o, (0,), expected & 0xFFFF, f"production.jsonl:{rid}:prompt_token[{t}] low half"),
                                    InputEquation(o, (1,), expected >> 16, f"production.jsonl:{rid}:prompt_token[{t}] high half")]))
    T["open_s"] = round(time.perf_counter() - t1, 2)
    n_ok = sum(c["ok"] for c in checks)
    print(f"[rec_p1_obligations] opened {len(coord_units)} coordinates, {len(checks)} input checks ({n_ok} ok), {len(problems)} problems in {T['open_s']} s", flush=True)
    (odir / "input_checks.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in checks))

    # --- chunked statements ---
    t1 = time.perf_counter()
    items: list[tuple[str, Any]] = [("coord", cu) for cu in coord_units] + [("input", e) for e in eqs_by_unit]
    # interleave so every chunk carries coordinates and inputs
    chunks: list[list[tuple[str, Any]]] = []
    n_chunks = max(1, -(-len(items) // a.chunk))
    for c in range(n_chunks):
        chunks.append(items[c::n_chunks])
    statements = []
    for c, chunk in enumerate(chunks):
        if a.only_chunks and c >= a.only_chunks:
            break
        tc = time.perf_counter()
        cus = [cu for k, cu in chunk if k == "coord"]
        ins = [e for k, e in chunk if k == "input"]
        eqs = [q for _, qs in ins for q in qs]
        if not cus:
            problems.append({"chunk": c, "problem": "chunk without a coordinate unit; typed-obligation export needs one -- rechunk"})
            continue
        export, record = O.build(comm, [co for _, co in cus], eqs, profile_id=reader.profile_id, name=f"REC-P1-{a.case}-chunk{c:02d}",
                                 description=f"record REC-P1 chunk {c}: {len(cus)} GemmCoordinate units and {len(eqs)} Input equations of the selected VUs of the validated P1/cu=whole partition",
                                 source=str(rec / "chronology" / "selected.json"))
        obj = TypedObligationSet.from_json(record["object"])
        v2 = encode_statement(to_statement(obj))
        assert v2.hex() == record["statement"]
        cdir = odir / f"chunk{c:02d}"
        cdir.mkdir(exist_ok=True)
        (cdir / "witness.bin").write_bytes(bytes.fromhex(record["witness"]))
        if a.statement_format == 3:
            # format 3 (`statement_bytes_v3`): the same lowering with uniform position rows as ranges; h_L = sha256 of the bytes handed to the guest
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
        record["selected_vus"] = {"coordinates": [u["vu_index"] for u, _ in cus], "inputs": [u["vu_index"] for u, _ in ins]}
        (cdir / "composed.json").write_text(json.dumps(record, sort_keys=True) + "\n")
        statements.append({"chunk": c, "dir": str(cdir), "coordinates": len(cus), "input_equations": len(eqs), "input_units": len(ins), "obligations": len(obj.obligations),
                           "leaves": len(obj.leaves), "roots": [r.id for r in obj.roots], "h_O": record["digest"], "h_L": record["statement_digest"],
                           "statement_bytes": len(v2), "statement_format": a.statement_format, "witness_bytes": len(record["witness"]) // 2, "relations": record["relations"], "witness_verdict": record["witness_verdict"],
                           "verdict": record["verdict"], "sha256": {"statement.bin": sha256_file(cdir / "statement.bin"), "witness.bin": sha256_file(cdir / "witness.bin")}})
        print(f"[rec_p1_obligations] chunk {c}: {len(cus)} coords + {len(eqs)} input eqs -> {len(obj.obligations)} obligations, statement {len(v2)} B, witness {len(record['witness']) // 2} B, "
              f"relations {record['relations']}, witness {record['witness_verdict']}, {len(obj.roots)} roots, {round(time.perf_counter() - tc, 1)} s", flush=True)
    # [CK-ELEM dispatch, re-targeted 22:20Z] selected VUs of a kind whose registered builder derives an expectation (coverage_v1.builders
    # .expectation_builders(): CK-ELEM's elementwise kinds): one statement per family per chunk of `a.chunk` units, written as further chunk
    # directories (statement.bin = format 3, witness.bin, composed.json with digest = h_O).  A unit the builder refuses (an interior word the
    # record does not commit) is a problem row, never dropped.
    from e2e.coverage_v1 import ck_elem as CE
    from e2e.coverage_v1.builders import expectation_builders

    elem_units = [u for u in units if isinstance(u.get("locate"), dict) and "builder" in u["locate"] and u["kind"] in expectation_builders()]
    if elem_units:
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
            for c0 in range(0, len(rs), a.chunk):
                c = len(statements)
                batch = rs[c0:c0 + a.chunk]
                built = CE.build_many(batch, handle, name=f"REC-P1-{a.case}-chunk{c:02d}-{fam}")
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
                statements.append({"chunk": c, "dir": str(cdir), "elementwise": len(batch), "family": fam, "kinds": sorted({r.kind for r in batch}),
                                   "obligations": len(built.obj.obligations), "leaves": len(built.obj.leaves), "roots": [r.id for r in built.obj.roots],
                                   "h_O": record["digest"], "h_L": record["statement_digest"], "statement_bytes": len(built.statement_bytes), "statement_format": 3,
                                   "witness_bytes": len(built.witness_bytes), "witness_verdict": record["witness_verdict"], "verdict": "accept",
                                   "sha256": {"statement.bin": sha256_file(cdir / "statement.bin"), "witness.bin": sha256_file(cdir / "witness.bin")}})
                print(f"[rec_p1_obligations] chunk {c}: {len(batch)} elementwise units ({fam}) -> {len(built.obj.obligations)} obligations, "
                      f"statement {len(built.statement_bytes)} B, witness {len(built.witness_bytes)} B, transparent {record['witness_verdict']}", flush=True)
        T["elementwise_s"] = round(time.perf_counter() - te, 1)
        T["elementwise_units"] = len(elem_units)
        T["elementwise_refused"] = sum(1 for p_ in problems if str(p_.get("problem", "")).startswith("builder refused"))
    T["statements_s"] = round(time.perf_counter() - t1, 1)
    out = {"schema": "veritor/rec/p1-obligations/v1", "utc": utc(), "instance_digest": proto["instance_digest"], "selected": len(units), "by_class": dict(by_class),
           "mapped_refused": len(refused), "coordinates_opened": len(coord_units), "input_checks": {"count": len(checks), "ok": n_ok, "by_kind": dict(Counter(c["kind"] for c in checks))},
           "problems": problems, "statements": statements, "timing": T,
           "exempt_selected": sum(v for k, v in by_class.items() if k.startswith("exempt")),
           "provenance": {"label": "MEASURED", "script": "scripts/rec_p1_obligations.py", "hardware": "record-gpu CPU",
                          "program": prog_check, "note": "units located in the pinned program by structural root-call names + Gemm member index; inputs by the program's parameter collections"}}
    (odir / "obligations.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    with (rec / "costlog.jsonl").open("a") as f:
        f.write(json.dumps({"stage": "obligations (locate selected VUs in the program, open commitments, build statements)", "class": "verifier", "seconds": round(sum(T.values()), 1),
                            "hardware": "record-gpu CPU", "batch": f"{len(units)} selected VUs -> {len(statements)} statements", "label": "MEASURED"}) + "\n")
    print(f"[rec_p1_obligations] DONE: {len(statements)} statements; {len(problems)} problems; exempt selected {out['exempt_selected']}; {T}")
    return 0 if not problems and not refused else 1


if __name__ == "__main__":
    sys.exit(main())
