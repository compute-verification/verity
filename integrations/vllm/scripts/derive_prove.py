#!/usr/bin/env python
"""S5 of `scripts/derive_e2e.sh` (lane CAPCORR, GUIDANCE §43.5): from S4's address map, a NONEMPTY selection of the
record's obligations is re-stated FROM THE PROGRAM'S GATES, its witness is reconstructed from the record's
authenticated interfaces (the adapter's opening path = WPREP's per-leaf pipeline), the statement is proved with
the record's own pinned SP1 host (ELF/vk = manifest pins) and verified, and `derived_proof_record.json` binds all
of it: program digest + kind-vocabulary version + profile + record sha + obligations (program addresses) + ELF/vk
+ times + host.

    python scripts/derive_prove.py --program <spec> --record <dir> --s4 <S4 out dir> --bundle <commit dir> \
        --checkpoint-dir <dir with model.safetensors> --host <veritor-zk-host> --out <dir> [--modes core,compressed]
    python scripts/derive_prove.py ... --verify-only --proof <bin> [--proof-record <json>]   # verifier side
    python scripts/derive_prove.py ... --self-test-refusals                                    # negatives

Exit codes: 0 proved+verified+bound; 2 binding refused (program/record/proof mismatch); 3 statement/witness/proof
failure; 4 usage/IO.  Every refusal names both sides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for p in (REPO, REPO / "src", REPO / "vllm-poc", HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import derive_corr as S4  # noqa: E402  (same directory: binding stage, record pins, host info)
from verity_vllm.correspondence.core import CaptureIdentity, NotInProgram  # noqa: E402
from verity_vllm.correspondence.program import (StructuralCircuitMap, TC_LOWERING_RULES, binding_check, canonical_reference,  # noqa: E402
                                              load_program, locate_coordinate, lower_call_to_subcircuit, program_identity, record_pins)

SCHEMA = "veritor/derived-proof-record/v1"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(p: Path, doc: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))


def gpu_info() -> dict[str, Any] | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            name, tot, used, drv = [s.strip() for s in r.stdout.strip().splitlines()[0].split(",")]
            return {"name": name, "memory_total": tot, "memory_used_at_start": used, "driver": drv}
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------------------------------------
# torch-free checkpoint resolver (the bundle references weights by checkpoint sha; values.bin is omitted)
# ---------------------------------------------------------------------------------------------------------


def numpy_resolver(model_dir: Path, index_sha256: dict[str, str] | None = None):
    """`WeightsReader.resolver` without torch: the HF tensors vLLM concatenates along dim 0 are byte-concatenated
    (row-major, dim-0 concat == byte concat); the safetensors file is hashed once against the recorded sha (the
    entry's `source`, or — for a bundle that references the whole values.bin by checkpoint sha and records no
    per-entry source — the index's `source_sha256` + `weight_sources.hf_tensors(param)`). Derived buffers (cos_sin
    tables etc.) have no checkpoint tensor and raise: they are never opened by a GEMM-coordinate obligation."""
    from verity_vllm_adapter.weight_sources import hf_tensors
    verified: dict[str, str] = {}
    headers: dict[str, tuple[dict, int]] = {}

    def resolve(entry: Any) -> bytes:
        src = entry.source or {}
        if src.get("kind") != "checkpoint":
            hf = hf_tensors(entry.param)
            if hf is None or not index_sha256:
                raise KeyError(f"{entry.canonical_id}: no checkpoint source recorded and not a checkpoint tensor")
            (fname, sha), = list(index_sha256.items())[:1]
            src = {"kind": "checkpoint", "file": fname, "file_sha256": sha, "tensors": hf[0], "cat_dim": hf[1]}
        fname = src["file"]
        path = model_dir / fname
        if fname not in verified:
            sha = sha256_file(path.resolve())
            if sha != src["file_sha256"]:
                raise ValueError(f"{path}: sha256 {sha} != recorded {src['file_sha256']}")
            verified[fname] = sha
            with path.open("rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                headers[fname] = (json.loads(f.read(hlen)), 8 + hlen)
        header, base = headers[fname]
        if int(src.get("cat_dim", 0)) != 0:
            raise KeyError(f"{entry.canonical_id}: cat_dim {src.get('cat_dim')} not handled without torch")
        out = bytearray()
        with path.open("rb") as f:
            for name in src["tensors"]:
                start, end = header[name]["data_offsets"]
                f.seek(base + start)
                out += f.read(end - start)
        return bytes(out)

    return resolve


# ---------------------------------------------------------------------------------------------------------
# statement from the program's gates + witness from the authenticated interfaces
# ---------------------------------------------------------------------------------------------------------


class Refused(Exception):
    """An explicit refusal: (exit code, reason)."""

    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code, self.reason = code, reason


def guarded_build(a: argparse.Namespace, out: Path, lp, amap: dict[str, Any], stage: str) -> dict[str, Any]:
    """`build_from_program` with the record-side failures turned into explicit refusals: an opening that disagrees with
    commitments.json (altered/missing commitment) -> exit 2; a reconstructed witness the transparent backend rejects
    (a committed value that does not satisfy its obligation, e.g. an altered output manifest) -> exit 3."""
    from verity_vllm.commit import CommitError
    from verity_vllm.typed_obligation_export import ExportError
    try:
        return build_from_program(a, out, lp, amap)
    except (ExportError, CommitError) as err:
        raise Refused(2, f"authenticated opening disagrees with the record's commitments: {err}") from err
    except SystemExit as err:
        msg = str(err.code) if isinstance(err.code, str) else ""
        if "transparent backend rejects" in msg:
            raise Refused(3, msg) from err
        raise


def build_from_program(a: argparse.Namespace, out: Path, lp, amap: dict[str, Any]) -> dict[str, Any]:
    from verity_vllm.capture import CaptureReader
    from verity_vllm.commit import TreeCache, load_commitments
    from verity_vllm.description import Description
    from verity_vllm.typed_obligation_export import CoordinateUnit, InputEquation, export_composed, packed_witness_of
    from verity_vllm_adapter.coordinate_check import coordinate_openings
    from verity_vllm_adapter.weight_sources import checkpoint_row_words
    from veritor.interface import TypedObligationSet, family_gate_set, to_statement
    from veritor.interface.typed_obligation import object_digest
    from veritor.protocol.messages import Reject
    from veritor.protocol.proofs import TransparentBackend
    from veritor.protocol.proofs.packed import encode_packed_witness
    from veritor.protocol.proofs.wire import encode_statement
    import dataclasses

    program = lp.program
    cmap = StructuralCircuitMap(program)
    t0 = time.time()
    sel_gemm = [e for e in amap["entries"] if e["class"] == "gemm-proof"]
    sel_eq = [e for e in amap["entries"] if e["class"] == "input-eq"]
    if a.max_units:
        sel_gemm = sel_gemm[: a.max_units]
    if not sel_gemm:
        raise SystemExit("S5: the address map has no GEMM-coordinate obligations to prove (selection must be nonempty)")
    bad = [e["id"] for e in sel_gemm if not e.get("kind_program_equal") or e.get("x_operand_is_program_value") is False]
    if bad:
        raise SystemExit(f"S5: refusing to prove obligations whose program lowering or x-operand did not check in S4: {bad}")
    # keep only the InputEqs that pin a value of the selected units (weight rows of the selected coordinates + boundary tokens)
    sel_targets = {e["id"] for e in sel_gemm}

    # --- witness side: the record's authenticated interfaces
    bundle = Path(a.bundle)
    reader = CaptureReader(bundle)
    desc = Description(reader.description)
    comm = load_commitments(reader)
    # trees from the committed leaf hashes (leaves.bin / ru index): the bundle omits weights/values.bin (referenced by
    # checkpoint sha); only the OPENED rows are rebuilt from the pinned checkpoint, and the guest re-hashes them against
    # the committed root — a wrong rebuild is a rejected proof, never a silently accepted one
    cache = TreeCache(reader, recompute_leaves=False)
    reader.weights.resolver = numpy_resolver(Path(a.checkpoint_dir), dict(reader.weights.index.source_sha256))
    units = []
    unit_addr = {}
    for e in sel_gemm:
        m = re.match(r"^(?P<cid>p=.*?/port=\d+)/Gemm\[(?P<j>\d+)\]$", e["id"])
        cid, j = m.group("cid"), int(m.group("j"))
        u = coordinate_openings(desc, reader, comm, cache, cid, j)
        units.append(u)
        unit_addr[f"{cid}/Gemm[{j}]"] = e
    t_open = time.time() - t0
    # input equations: weight words (expected = the CHECKPOINT's word) + boundary token halves (expected = production's token)
    eqs: list[InputEquation] = []
    eq_report = []
    units_by_w = {(u.param, u.j): u for u in units}
    prod = [json.loads(l) for l in (bundle.parent / "production" / "production.jsonl").read_text().splitlines() if l.strip()]
    prod_by_rid = {r["request_id"]: r for r in prod}
    for e in sel_eq:
        tid = e["id"]
        m = re.match(r"^w=(?P<param>[^/]+)/c=(?P<j>\d+)\[0\]\[(?P<k>\d+)\]/InputEq$", tid)
        if m:
            param, j, k = m.group("param"), int(m.group("j")), int(m.group("k"))
            u = units_by_w.get((param, j))
            if u is None:
                continue  # pins a weight row of a unit outside --max-units
            data, source = checkpoint_row_words(Path(a.checkpoint_dir), param, j)
            expected = int.from_bytes(data[k * 2:(k + 1) * 2], "little")
            eqs.append(InputEquation(u.w_row, (0, k), expected, f"checkpoint:{source['file']}@{source['file_sha256'][:12]}:{source['tensor']}[{source['local_row']}][{k}]"))
            eq_report.append({"target": tid, "expected": expected, "source": "checkpoint", "program_address": e["address"]})
            continue
        m = re.match(r"^(?P<leaf>p=[^/]+/r=(?P<r>[^/]+)/b=(?P<f>prompt_token|output_token)/t=(?P<t>\d+))\[(?P<h>[01])\]/InputEq$", tid)
        if not m:
            raise SystemExit(f"S5: unrecognised InputEq target {tid}")
        leaf_id, rid, fld, t, h = m.group("leaf"), m.group("r"), m.group("f"), int(m.group("t")), int(m.group("h"))
        if rid not in {u.request_id for u in units}:
            continue
        rec = prod_by_rid[rid]
        b = reader.boundary(rid)
        if fld == "prompt_token":
            expected_tok = int((rec.get("prompt_token_ids") or b.prompt_token_ids)[t])
        else:
            outputs = list(rec.get("output_token_ids") or [])
            present = [(k, r) for k, r in enumerate(b.output_records) if r.present]
            k = next(k for k, r in present if int(r.t) == t)
            expected_tok = int(outputs[k]) if k < len(outputs) else int(present[k][1].token_id)
        o = cache.open(leaf_id)
        u32 = expected_tok & 0xFFFFFFFF
        half = (u32 & 0xFFFF) if h == 0 else (u32 >> 16)
        eqs.append(InputEquation(o, (h,), half, f"production.jsonl:{rid}:{fld}[{t}] {'low' if h == 0 else 'high'} half"))
        eq_report.append({"target": tid, "expected": half, "token": expected_tok, "source": "production.jsonl", "program_address": e["address"]})
    cus = [CoordinateUnit(u.canonical_id, u.j, u.x, u.w_row, u.out_row, u.x_cols) for u in units]
    export = export_composed(comm, cus, eqs, profile_id=reader.profile_id)
    t_export = time.time() - t0 - t_open

    # --- statement side: the subcircuit bodies come from the PROGRAM'S GATES (capcorr.tc-lowering@1), not from the
    # interface's pinned table; equality of digests with the exporter's table is asserted, then the program-lowered
    # bodies REPLACE the table (same digest => same canonical body => same h_O; the swap makes the provenance literal)
    prog_subs: dict[str, Any] = {}
    prog_addr: dict[str, dict[str, Any]] = {}
    for u in units:
        gemm, coord = locate_coordinate(cmap, u.canonical_id, u.j)
        sc = lower_call_to_subcircuit(coord)
        prog_subs[sc.digest] = sc
        prog_addr[f"{u.canonical_id}/Gemm[{u.j}]"] = {"coordinate": canonical_reference(coord, coord.returns), "gemm": canonical_reference(gemm),
                                                      "program_lowered_subcircuit": sc.digest}
    table = {s.digest: s for s in export.obj.subcircuits}
    for ob in export.obj.obligations:
        if ob.target.id.endswith("/InputEq"):
            continue
        if ob.subcircuit not in prog_subs:
            raise SystemExit(f"S5: obligation {ob.target.id} uses subcircuit {ob.subcircuit[:16]} which the program lowering did not produce")
    swapped = tuple(prog_subs.get(s.digest, s) for s in export.obj.subcircuits)
    for s_old, s_new in zip(export.obj.subcircuits, swapped, strict=True):
        if s_old.digest != s_new.digest or (s_old.incoming, s_old.gates, s_old.outgoing) != (s_new.incoming, s_new.gates, s_new.outgoing):
            raise SystemExit(f"S5: program-lowered subcircuit {s_new.digest[:16]} differs from the exporter's body")
    obj = dataclasses.replace(export.obj, subcircuits=swapped)
    h_o = object_digest(obj.to_json())
    statement = to_statement(obj)
    raw_stmt = encode_statement(statement)
    witness = packed_witness_of(export.openings.values(), statement)
    raw_wit = encode_packed_witness(witness)
    gate_set = family_gate_set(obj.family.id, obj.family.width)
    try:
        TransparentBackend(gate_set).verify(statement, raw_wit)
        transparent = "accept"
    except Reject as err:
        transparent = f"reject:{err.code.name}"
    h_l = hashlib.sha256(raw_stmt).hexdigest()
    (out / "statement.bin").write_bytes(raw_stmt)
    (out / "witness.bin").write_bytes(raw_wit)
    write_json(out / "typed_object.json", obj.to_json())
    rd = Path(a.record)
    rec_stmt = (rd / "obligations" / "statement.bin").read_bytes() if (rd / "obligations" / "statement.bin").exists() else None
    rec_wit = (rd / "obligations" / "witness.bin").read_bytes() if (rd / "obligations" / "witness.bin").exists() else None
    rec_comp = json.loads((rd / "obligations" / "composed.json").read_text())
    doc = {"units": len(units), "input_equations": len(eqs), "obligations": len(obj.obligations), "leaves": len(obj.leaves), "references": len(obj.references),
           "roots": [r.id for r in obj.roots], "subcircuits": {s.digest: (s.definition.name if s.definition else None) for s in obj.subcircuits},
           "h_O": h_o, "h_L": h_l, "statement_bytes": len(raw_stmt), "witness_bytes": len(raw_wit), "transparent_verdict": transparent,
           "statement_equals_record": (rec_stmt == raw_stmt) if rec_stmt is not None else None,
           "witness_equals_record": (rec_wit == raw_wit) if rec_wit is not None else None,
           "h_O_equals_record": h_o == rec_comp.get("digest"), "record_h_O": rec_comp.get("digest"), "record_h_L": rec_comp.get("statement_digest"),
           "program_addresses": prog_addr, "input_equation_report": eq_report,
           "lowering_rule": TC_LOWERING_RULES, "openings": len(export.openings), "seconds": {"open": round(t_open, 2), "export": round(t_export, 2)},
           "witness_reconstruction": ("adapter opening path (coordinate_openings: x row, W[j,:] leaf, out row; boundary leaf; weights rebuilt "
                                      "from the pinned checkpoint by sha) -> export_composed -> packed_witness_of (format-3 packed leaves + paths)")}
    if transparent != "accept":
        raise SystemExit(f"S5: transparent backend rejects the reconstructed witness: {transparent}")
    return doc


# ---------------------------------------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------------------------------------


def host_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("host printed no JSON")


def host(a: argparse.Namespace, out: Path, name: str, *hargs: str, prover: str | None = None, timeout: int = 3600) -> dict[str, Any]:
    cmd = [a.host, *hargs]
    env = dict(os.environ)
    env["SP1_PROVER"] = prover if prover is not None else a.sp1_prover
    env.setdefault("RUST_LOG", "info")
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / f"host-{name}.log").write_text(r.stderr + "\n" + r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"host {name} exited {r.returncode}: {r.stderr[-1200:]}")
    d = host_json(r.stdout)
    d["wall_seconds"] = round(time.perf_counter() - t, 3)
    return d


def prove_and_verify(a: argparse.Namespace, out: Path, built: dict[str, Any], pins: dict[str, Any]) -> dict[str, Any]:
    from veritor.interface import TypedObligationSet, check_binding
    obj = TypedObligationSet.from_json(json.loads((out / "typed_object.json").read_text()))
    st, wt = str(out / "statement.bin"), str(out / "witness.bin")
    info = host(a, out, "info", "info", prover="cpu")
    prover_ok = info.get("elf_sha256") == pins.get("elf_sha256") and info.get("vk_hash") == pins.get("vk_hash")
    res: dict[str, Any] = {"info": info, "host_sha256": sha256_file(Path(a.host)), "host_sha256_equals_record": sha256_file(Path(a.host)) == pins.get("host_sha256"),
                           "elf_vk_equal_record_pins": prover_ok, "modes": {}}
    if not prover_ok:
        res["refusal"] = (f"prover identity mismatch: host ELF {str(info.get('elf_sha256'))[:16]} / vk {str(info.get('vk_hash'))[:18]} != record pins "
                          f"{str(pins.get('elf_sha256'))[:16]} / {str(pins.get('vk_hash'))[:18]}")
        return res
    common = ["--statement", st, "--witness", wt, "--typed-object-digest", built["h_O"]]
    ex = host(a, out, "execute", "execute", *common, prover="cpu")
    res["execute"] = {k: ex.get(k) for k in ("verdict", "total_cycles", "cycle_tracker", "public_values", "execute_seconds", "wall_seconds", "unsound")}
    all_ok = bool(ex.get("verdict")) and not ex.get("unsound")
    for mode in a.modes:
        proof = out / f"proof-{mode}.bin"
        try:
            pr = host(a, out, f"prove-{mode}", "prove", *common, "--out", str(proof), "--mode", mode, timeout=a.timeout)
        except (RuntimeError, subprocess.TimeoutExpired) as err:
            res["modes"][mode] = {"ok": False, "error": str(err)[-1500:]}
            all_ok = False
            continue
        vr = host(a, out, f"verify-{mode}", "verify", "--proof", str(proof), "--statement", st, prover="cpu")
        pv = bytes.fromhex(vr["public_values"])
        b = check_binding(obj, pv)
        ok = bool(vr.get("ok")) and bool(vr.get("statement_match")) and bool(vr.get("verdict")) and not vr.get("unsound") and vr.get("vk_hash") == info.get("vk_hash") and b.ok
        res["modes"][mode] = {"ok": ok, "proof_bytes": pr.get("proof_bytes"), "proof_sha256": sha256_file(proof), "shards": pr.get("shards"),
                              "setup_seconds": pr.get("setup_seconds"), "prove_seconds": pr.get("prove_seconds"), "prove_wall_seconds": pr.get("wall_seconds"),
                              "host_verify_seconds": pr.get("verify_seconds"), "verify_seconds": vr.get("verify_seconds"), "verify_wall_seconds": vr.get("wall_seconds"),
                              "verify_ok": vr.get("ok"), "statement_match": vr.get("statement_match"), "verdict": vr.get("verdict"), "vk_hash": vr.get("vk_hash"),
                              "public_values": vr.get("public_values"), "binding": {"ok": b.ok, "code": str(getattr(b, "code", "")), "detail": getattr(b, "detail", "")}}
        all_ok &= ok
    res["ok"] = all_ok
    return res


# ---------------------------------------------------------------------------------------------------------
# verifier side
# ---------------------------------------------------------------------------------------------------------


def verify_only(a: argparse.Namespace, out: Path) -> int:
    """A second party with the program, the record and a proof (+ its record): rebuild the statement from the program
    and the record's interfaces, check the proof against THAT statement and the pins. Refuses on any mismatch."""
    from veritor.interface import TypedObligationSet, check_binding
    pins = record_pins(Path(a.record))
    lp = load_program(a.program, LP=a.LP or pins.get("LP"), STEPS=a.STEPS or pins.get("STEPS"), repo_root=REPO)
    ident = program_identity(lp)
    reasons = binding_check(ident, pins, expect_vocabulary=a.expect_vocabulary, expect_profile_sha256=a.expect_profile_sha256)
    prec = json.loads(Path(a.proof_record).read_text()) if a.proof_record else None
    if prec is not None:
        if prec["program"]["digest"] != ident["digest"]:
            reasons.append(f"proof record program digest {prec['program']['digest'][:16]} != supplied program {ident['digest'][:16]}")
        if prec["program"]["vocabulary"]["digest"] != ident["vocabulary"]["digest"]:
            reasons.append(f"proof record kind-vocabulary {prec['program']['vocabulary']['digest'][:16]} != this evaluation's {ident['vocabulary']['digest'][:16]}")
        if prec["record"]["certificate_sha256"] != pins.get("certificate_sha256"):
            reasons.append(f"proof record certificate sha {prec['record']['certificate_sha256'][:16]} != this record's {str(pins.get('certificate_sha256'))[:16]}")
        if prec["prover"]["elf_sha256"] != pins.get("elf_sha256") or prec["prover"]["vk_hash"] != pins.get("vk_hash"):
            reasons.append("proof record ELF/vk != record manifest pins")
    if reasons:
        for r in reasons:
            print(f"[verify] REFUSED: {r}", file=sys.stderr)
        write_json(out / "verify_result.json", {"ok": False, "refused": reasons, "utc": utc()})
        return 2
    amap = json.loads((Path(a.s4) / "l3" / "address_map.json").read_text())
    if amap["program"]["digest"] != ident["digest"]:
        print(f"[verify] REFUSED: address map program digest {amap['program']['digest'][:16]} != supplied program {ident['digest'][:16]}", file=sys.stderr)
        return 2
    try:
        built = guarded_build(a, out, lp, amap, "verify")
    except Refused as r:
        print(f"[verify] REFUSED: {r.reason}", file=sys.stderr)
        write_json(out / "verify_result.json", {"ok": False, "refused": [r.reason], "utc": utc()})
        return r.code
    if prec is not None and prec["statement"]["h_L"] != built["h_L"]:
        print(f"[verify] REFUSED: proof record h_L {prec['statement']['h_L'][:16]} != statement rebuilt from the program {built['h_L'][:16]}", file=sys.stderr)
        write_json(out / "verify_result.json", {"ok": False, "refused": ["h_L mismatch"], "built": built, "utc": utc()})
        return 2
    obj = TypedObligationSet.from_json(json.loads((out / "typed_object.json").read_text()))
    try:
        vr = host(a, out, "verify-only", "verify", "--proof", a.proof, "--statement", str(out / "statement.bin"), prover="cpu")
    except RuntimeError as err:
        print(f"[verify] REFUSED: host verify failed: {str(err)[-400:]}", file=sys.stderr)
        write_json(out / "verify_result.json", {"ok": False, "refused": [f"host verify failed: {str(err)[-800:]}"], "built": built, "utc": utc()})
        return 2
    pv = bytes.fromhex(vr.get("public_values") or "")
    b = check_binding(obj, pv) if pv else None
    problems = []
    if not vr.get("ok"):
        problems.append("host: proof does not verify")
    if not vr.get("statement_match"):
        problems.append(f"host: proof public values name statement {str(vr.get('statement_digest'))[:16]}, not the program-built statement {built['h_L'][:16]}")
    if vr.get("vk_hash") != pins.get("vk_hash"):
        problems.append(f"vk {vr.get('vk_hash')} != record pin {pins.get('vk_hash')}")
    if b is None or not b.ok:
        problems.append(f"IB binding: {getattr(b, 'code', 'no public values')} {getattr(b, 'detail', '')}")
    if not vr.get("verdict"):
        problems.append("guest verdict is reject")
    doc = {"ok": not problems, "refused": problems, "host_verify": vr, "built": {k: v for k, v in built.items() if k not in ("program_addresses", "input_equation_report")},
           "program": ident, "record": pins["record"], "utc": utc()}
    write_json(out / "verify_result.json", doc)
    if problems:
        for r in problems:
            print(f"[verify] REFUSED: {r}", file=sys.stderr)
        return 2
    print(f"[verify] OK: proof binds program {ident['digest'][:16]} x record {pins['record']} statement {built['h_L'][:16]} (vk {vr.get('vk_hash')[:18]})")
    return 0


def self_test_refusals(a: argparse.Namespace, out: Path) -> int:
    """Verifier-side negatives. Each must exit 2 with the intended reason; the control must pass."""
    base = [sys.executable, str(HERE), "--record", a.record, "--s4", a.s4, "--bundle", a.bundle, "--checkpoint-dir", a.checkpoint_dir, "--host", a.host]
    proof = str(out / f"proof-{a.modes[0]}.bin")
    prec = str(out / "derived_proof_record.json")
    cases = []

    def run(name: str, extra: list[str], expect: int, must_contain: str, *, mode: str = "--verify-only", bundle: str | None = None) -> None:
        cmd = [*base, mode, "--out", str(out / "refusal-tmp"), *extra]
        if bundle is not None:
            cmd[cmd.index("--bundle") + 1] = bundle
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        log = r.stderr + r.stdout
        cases.append({"case": name, "expect_exit": expect, "exit": r.returncode, "ok": r.returncode == expect and must_contain in log, "log_tail": log[-400:]})

    def mirrored_bundle(tmp: Path, *, edit_production=None, edit_commitments=None) -> str:
        """A bundle directory of symlinks to the real files, with production.jsonl and/or commitments.json replaced by
        an edited copy (the record itself is never modified)."""
        src = Path(a.bundle)
        mb = tmp / "bundle"
        mb.mkdir(parents=True)
        for f in src.iterdir():
            if f.name == "commitments.json" and edit_commitments is not None:
                doc = json.loads(f.read_text())
                edit_commitments(doc)
                (mb / f.name).write_text(json.dumps(doc))
            else:
                (mb / f.name).symlink_to(f.resolve())
        pd = tmp / "production"
        pd.mkdir()
        lines = [json.loads(l) for l in (src.parent / "production" / "production.jsonl").read_text().splitlines() if l.strip()]
        if edit_production is not None:
            edit_production(lines)
        (pd / "production.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
        for f in (src.parent / "production").iterdir():
            if f.name != "production.jsonl":
                (pd / f.name).symlink_to(f.resolve())
        return str(mb)

    run("control: this proof, this program, this record", ["--program", a.program, "--proof", proof, "--proof-record", prec], 0, "[verify] OK")
    run("different program (Serve_v1 STEPS=7) offered for this record", ["--program", "registry:B0", "--STEPS", "7", "--proof", proof, "--proof-record", prec], 2, "program digest mismatch")
    other = a.other_proof or str(REPO / "fixtures/e2e-composed/INT-B0-20260908T0235Z/proofs/proof-core.bin")
    if Path(other).exists():
        run("a proof from a DIFFERENT record (INT-B0-20260908T0235Z) offered as this record's", ["--program", a.program, "--proof", other], 2, "REFUSED")
    covp = sorted(Path(a.record).glob("proofs/proof-cov/embed-*/proof-compressed.bin"))
    if covp:
        run("this record's embed-gather coverage proof offered as the derived GEMM statement's proof", ["--program", a.program, "--proof", str(covp[0])], 2, "REFUSED")
    # a tampered proof record: program digest edited
    import tempfile
    with tempfile.TemporaryDirectory(prefix="capcorr-s5-") as tmp:
        d = json.loads(Path(prec).read_text())
        d["program"]["digest"] = "00" * 32
        p = Path(tmp) / "rec.json"
        p.write_text(json.dumps(d))
        run("proof record whose program digest was edited", ["--program", a.program, "--proof", proof, "--proof-record", str(p)], 2, "proof record program digest")
        d = json.loads(Path(prec).read_text())
        d["program"]["vocabulary"]["digest"] = "11" * 32
        p.write_text(json.dumps(d))
        run("proof record whose kind-vocabulary version was edited", ["--program", a.program, "--proof", proof, "--proof-record", str(p)], 2, "kind-vocabulary")
        # missing required proof: a proof that covers only a SUBSET of the selection (S5 --max-units 1) offered as the selection's proof
        subset = out / "subset" / f"proof-{a.modes[0]}.bin"
        if subset.exists():
            run("missing required proof: a proof covering 1 of the selected GEMM coordinates offered as the whole selection's proof",
                ["--program", a.program, "--proof", str(subset)], 2, "not the program-built statement")
        # altered committed output: one request's produced tokens edited in (a mirror of) the output manifest production.jsonl
        d = json.loads(Path(prec).read_text())
        rids = sorted({e["target"].split("/r=")[1].split("/")[0] for e in d["selection"]["input_equation_report"] if e.get("source") == "production.jsonl"})
        if rids:
            rid = rids[0]

            def flip_outputs(lines):
                for l in lines:
                    if l["request_id"] == rid:
                        l["output_token_ids"] = [int(t) ^ 1 for t in (l.get("output_token_ids") or [])]

            mb = mirrored_bundle(Path(tmp) / "alt-output", edit_production=flip_outputs)
            run(f"altered committed output (request {rid[:8]} output tokens edited in production.jsonl): the record's proof offered against it",
                ["--program", a.program, "--proof", proof, "--proof-record", prec], 3, "transparent backend rejects the reconstructed witness", bundle=mb)
            run(f"altered committed output (request {rid[:8]}): statement+witness rebuilt over the edited manifest",
                ["--program", a.program], 3, "transparent backend rejects the reconstructed witness", mode="--build-only", bundle=mb)
        # altered committed root: the boundary tree root of a request in the selection edited in (a mirror of) commitments.json
        roots = [t for t in (d["statement"].get("roots") or []) if str(t).startswith("boundary/")]
        if roots:
            tid = roots[0]

            def flip_root(doc):
                r = doc["trees"][tid]["root"]
                doc["trees"][tid]["root"] = ("0" if r[0] != "0" else "1") + r[1:]

            mb = mirrored_bundle(Path(tmp) / "alt-root", edit_commitments=flip_root)
            run(f"altered commitment ({tid} root edited in commitments.json): the record's proof offered against it",
                ["--program", a.program, "--proof", proof, "--proof-record", prec], 2, "REFUSED: authenticated opening disagrees", bundle=mb)
    shutil.rmtree(out / "refusal-tmp", ignore_errors=True)
    ok = all(c["ok"] for c in cases)
    write_json(out / "refusals.json", {"schema": SCHEMA + "/refusals", "utc": utc(), "cases": cases, "all_as_intended": ok})
    for c in cases:
        print(f"[s5-refusals] {'ok ' if c['ok'] else 'BAD'} exit {c['exit']} (expect {c['expect_exit']}): {c['case']}")
    print(f"[s5-refusals] {'PASS' if ok else 'FAIL'}: {sum(c['ok'] for c in cases)}/{len(cases)}")
    return 0 if ok else 3


# ---------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--s4", required=True, help="S4 output dir (binding.json, l3/address_map.json)")
    ap.add_argument("--bundle", required=True, help="extracted commit dir (holds commitments.json; ../production/production.jsonl beside it)")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--host", required=True, help="veritor-zk-host (the record's bin/veritor-zk-host: ELF/vk must equal the manifest pins)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="core", help="comma list of core,compressed")
    ap.add_argument("--sp1-prover", default=os.environ.get("SP1_PROVER", "cuda"))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--max-units", type=int, default=0, help="prove only the first N coordinates (0 = the record's whole selection)")
    ap.add_argument("--expect-vocabulary", default=None)
    ap.add_argument("--expect-profile-sha256", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--LP", type=int, default=None)
    ap.add_argument("--STEPS", type=int, default=None)
    ap.add_argument("--build-only", action="store_true", help="stop after the statement/witness are built (no host)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--proof", default=None)
    ap.add_argument("--proof-record", default=None)
    ap.add_argument("--other-proof", default=None)
    ap.add_argument("--self-test-refusals", action="store_true")
    a = ap.parse_args(argv)
    a.modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.verify_only:
        if not a.proof:
            print("--verify-only needs --proof", file=sys.stderr)
            return 4
        return verify_only(a, out)
    if a.self_test_refusals:
        return self_test_refusals(a, out)
    t0 = time.time()
    bind_doc, lp = S4.stage_bind(a, out)
    if not bind_doc["bound"]:
        return 2
    pins = bind_doc["record"]
    amap = json.loads((Path(a.s4) / "l3" / "address_map.json").read_text())
    if amap["program"]["digest"] != bind_doc["program"]["digest"]:
        print(f"[s5] REFUSED: address map was built for program {amap['program']['digest'][:16]}, not {bind_doc['program']['digest'][:16]}", file=sys.stderr)
        return 2
    try:
        built = guarded_build(a, out, lp, amap, "s5")
    except Refused as r:
        print(f"[s5] REFUSED: {r.reason}", file=sys.stderr)
        write_json(out / "built.json", {"ok": False, "refused": [r.reason], "utc": utc()})
        return r.code
    print(f"[s5] statement from program gates: {built['obligations']} obligations ({built['units']} GemmCoordinate + {built['input_equations']} InputEq), "
          f"h_O {built['h_O'][:16]} h_L {built['h_L'][:16]}, statement==record {built['statement_equals_record']}, witness==record {built['witness_equals_record']}, "
          f"transparent {built['transparent_verdict']} ({built['seconds']})")
    if a.build_only:
        write_json(out / "built.json", built)
        print("[s5] --build-only: statement + witness written; no proof")
        return 0 if built["transparent_verdict"] == "accept" else 3
    pv = prove_and_verify(a, out, built, pins)
    s4 = json.loads((Path(a.s4) / "s4.json").read_text()) if (Path(a.s4) / "s4.json").exists() else {}
    l2 = (s4.get("stages") or {}).get("l2") or {}
    doc = {"schema": SCHEMA, "utc": utc(), "label": bind_doc["label"],
           "program": bind_doc["program"], "record": {k: pins.get(k) for k in ("record", "program_digest", "certificate_program_digest", "certificate_sha256", "manifest_program_digest",
                                                                            "semantic_profile", "semantic_profile_sha256", "elf_sha256", "vk_hash", "host_sha256", "LP", "STEPS",
                                                                            "production_record_sha256", "served_tokens_sha256", "sampling_policy_sha256", "sampling_policy",
                                                                            "commitments_sha256", "plan_digest", "challenge_digest", "capture_profile_id")},
           "capture_configuration": bind_doc["capture_configuration_note"],
           "s4": {"binding": bind_doc["bound"], "l2": {r: {k: v.get(k) for k in ("status", "exact", "mismatch", "missing_operand", "not_in_program", "committed_leaves")} for r, v in (l2.get("requests") or {}).items()},
                  "l3": (s4.get("stages") or {}).get("l3")},
           "selection": {"policy": "the record's own composed selection (obligations/composed.json), addressed by S4's L3 map", "obligations": built["obligations"],
                         "gemm_coordinates": built["units"], "input_equations": built["input_equations"], "program_addresses": built["program_addresses"],
                         "input_equation_report": built["input_equation_report"]},
           "statement": {k: built[k] for k in ("h_O", "h_L", "statement_bytes", "witness_bytes", "transparent_verdict", "statement_equals_record", "witness_equals_record",
                                               "h_O_equals_record", "record_h_O", "record_h_L", "roots", "leaves", "references", "subcircuits", "lowering_rule", "openings", "seconds",
                                               "witness_reconstruction")},
           "prover": {"host": a.host, "host_sha256": pv["host_sha256"], "host_sha256_equals_record": pv["host_sha256_equals_record"], "elf_sha256": pv["info"].get("elf_sha256"),
                      "vk_hash": pv["info"].get("vk_hash"), "sp1_version": pv["info"].get("sp1_version"), "guest_features": pv["info"].get("guest_features"),
                      "elf_vk_equal_record_pins": pv["elf_vk_equal_record_pins"], "sp1_prover": a.sp1_prover},
           "execute": pv.get("execute"), "proofs": pv["modes"], "ok": bool(pv.get("ok")), "refusal": pv.get("refusal"),
           "host_machine": {**S4.host_info(), "gpu": gpu_info()}, "seconds_total": round(time.time() - t0, 1)}
    write_json(out / "derived_proof_record.json", doc)
    status = "PASS" if doc["ok"] else "FAIL"
    print(f"[s5] {status}: " + "; ".join(f"{m}: {'ok' if r.get('ok') else 'FAIL'} prove {r.get('prove_seconds')}s verify {r.get('verify_seconds')}s {r.get('proof_bytes')} B"
                                        for m, r in pv["modes"].items()) + (f"; {pv.get('refusal')}" if pv.get("refusal") else "") + f" -> {out / 'derived_proof_record.json'}")
    return 0 if doc["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
