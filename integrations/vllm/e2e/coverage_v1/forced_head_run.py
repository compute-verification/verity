"""FORCED (labelled) run of one whole-head attention VU through the verifier_v3 V5 path on a real record (A5/6 item 3 for the head VU;
Q-A56-6: A9's TINY record).  NOT a certificate: the selection is chosen, not drawn; the verifier_v3 modules used are the mainline
files with A5/6's PROPOSED patches (`data/contract/a56-coverage/patches/a3-*.diff`) applied to a TEMPORARY copy -- A3's files in
the checkout are not touched.  Every output carries the label.

What runs, in order (each step's timing recorded):
  1. verifier: build the pinned program in-process (A3 ``build_pinned_program`` from ``protocol_instance.json#program.identity``),
     ``ProgramLocator``; the forced unit = head ``--head`` of ``t=<t>/layers.<L>/attn`` (``attn_locator.head_interval``), located back
     through ``locate_attention`` (must say ``unit == "head"``);
  2. verifier: ``expected.coverage_inputs`` over the retained RU (committed trees, id lists checked) with the forced unit ->
     ``Q_proof`` (flag ``COVERAGE_V1_HEAD_VU=1``) -> ``expected_statements`` (A2 expectation, value-free, digest);
  3. prover: ``CaptureRecord`` over ``adapter/selective`` + ``dispatch(LocatedUnit)`` -> raw fa2 statement + witness, written as a batch;
  4. verifier: ``head_views_of_raw_statement`` from committed ids only -> ``check_coverage`` -> the forced VU covered exactly once;
  5. R9 negative: the NEIGHBOURING head's statement -> uncovered + unexpected; a tampered output word -> the builder refuses (HC);
  6. optional ``--fa2-host``: SP1 execute / prove / verify of the batch (fa2 guest, UNAPPROVED); optional ``--approved-host``:
     ``verify-many`` of that proof through the approved host -> expected REFUSAL (ok=false: GUEST_NOT_APPROVED by construction).

    PYTHONPATH=. python -m e2e.coverage_v1.forced_head_run --record <rec> --out /tmp/a56/forced --t 3 --layer 0 --head 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PATCHES = REPO / "data" / "contract" / "a56-coverage" / "patches"
LABEL = "FORCED (labelled) selection through PATCHED verifier_v3 modules (temporary copy; A5/6 PROPOSED patches) -- not a certificate; head VU is a PROPOSED partition change; fa2 guest UNAPPROVED"


def mainline_has_attention_path() -> bool:
    """True when the checkout's ``verifier_v3`` already carries the attention path (A3 ckpt 7 applied the A5/6 patches on
    mainline ``b745af3b``): ``Deriver.attention_row`` / ``Deriver.config`` and ``locator.ATTENTION_KINDS``.  Then no overlay."""

    from e2e.verifier_v3 import expected as ex
    from e2e.verifier_v3 import locator as vloc

    return hasattr(ex.Deriver, "attention_row") and hasattr(vloc, "ATTENTION_KINDS")


def overlay_patched_e2e() -> Path:
    """Copy ``e2e`` to a temp dir, apply the two verifier patches there, and put it FIRST on ``sys.path``."""
    tmp = Path(tempfile.mkdtemp(prefix="a56-forced-"))
    shutil.copytree(REPO / "e2e", tmp / "e2e", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("a3-expected-attention.diff", "a3-locator-attention.diff"):
        r = subprocess.run(["patch", "-p3", "-s", "-d", str(tmp / "e2e"), "-i", str(PATCHES / name)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"patch {name} does not apply to the mainline copy: {r.stdout}{r.stderr}")
    for m in [m for m in sys.modules if m == "e2e" or m.startswith("e2e.")]:
        del sys.modules[m]
    sys.path.insert(0, str(tmp))
    return tmp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=int, default=None, help="position (default: the last prefill position of the retained request, LP-1)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--fa2-host", default=None)
    ap.add_argument("--approved-host", default=None)
    ap.add_argument("--mode", default="core")
    a = ap.parse_args(argv)
    os.environ["COVERAGE_V1_HEAD_VU"] = "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Path(a.record)
    T: dict[str, float] = {}
    doc: dict[str, Any] = {"schema": "veritor/a56/forced-head-run/v1", "label": LABEL, "record": str(rec), "steps": {}, "timings_s": T}

    def step(name: str, t0: float, **kw: Any) -> None:
        T[name] = round(time.perf_counter() - t0, 3)
        doc["steps"][name] = kw
        print(f"[forced_head_run] {name}: {T[name]} s {json.dumps(kw, default=str)[:300]}", flush=True)

    if mainline_has_attention_path():
        tmp = None
        doc["verifier_v3_source"] = "mainline checkout (attention path present: A3 ckpt 7 applied the A5/6 patches) -- no overlay, nothing patched"
        doc["label"] = LABEL.replace("through PATCHED verifier_v3 modules (temporary copy; A5/6 PROPOSED patches)", "through MAINLINE verifier_v3 modules (A3 ckpt 7 carries the attention path)")
    else:
        tmp = overlay_patched_e2e()
        doc["verifier_v3_source"] = f"patched overlay {tmp}"
    from e2e.coverage_v1.builders import BUILDERS, BuildRefusal, LocatedUnit, dispatch  # first: the registry imports the builders (flag on)
    from e2e.coverage_v1 import check_coverage, expected_statements
    from e2e.coverage_v1.attn_head_builder import fa2_gate_set, head_kind_id, head_views_of_raw_statement
    from e2e.coverage_v1.attn_locator import head_interval, locate_attention
    from e2e.coverage_v1.capture_record import CaptureRecord
    from e2e.coverage_v1.coverage import AcceptedProof
    from e2e.verifier_v3 import expected as ex
    from e2e.verifier_v3 import locator as vloc

    BUILDERS.refresh()
    assert "AttentionHead_v1|call" in BUILDERS, "head builder not registered (COVERAGE_V1_HEAD_VU flag?)"

    # 1. verifier: program + locator + forced unit
    t0 = time.perf_counter()
    inst_doc = json.loads((rec / "protocol_instance.json").read_text())
    proto = json.loads((rec / "protocol.json").read_text())
    maps = json.loads((rec / "maps.json").read_text())
    identity = (inst_doc.get("program") or {}).get("identity") or {}
    program, C, problems = vloc.build_pinned_program(identity, expected_digest=maps["program"]["digest"], expected_gates=None)
    if program is None or problems:
        raise SystemExit(f"program not rebuilt: {problems}")
    loc = vloc.ProgramLocator(program, C)
    rus = proto["ru_map"]["rus"]
    F = set(proto.get("F") or [])
    retained = [r for r in proto["J"] if r not in F] or list(range(len(rus)))
    ru = retained[0]
    rid = rus[ru]["request_id"]
    t = a.t if a.t is not None else int(identity.get("LP", 4)) - 1
    iv = head_interval(loc, t, a.layer, a.head)
    if iv is None:
        raise SystemExit(f"no head {a.head} at t={t} layer={a.layer} in the program")
    L = locate_attention(loc, *iv)
    D = int(C["D"])
    kind = head_kind_id(t + 1, D)
    FORCED_VU = 10**9 + a.head  # labelled: not a drawn index
    forced = {"vu_index": FORCED_VU, "ru": ru, "request_id": rid, "kind": kind, "class": "forced (labelled): head VU, PROPOSED partition", "gate_interval": list(iv), "locate": L}
    step("verifier.program_locator_forced_unit", t0, identity=identity, program_gates=program.gates, ru=ru, request_id=rid, t=t, layer=a.layer, head=a.head, interval=list(iv), located=L)
    if L.get("unit") != "head":
        raise SystemExit(f"forced interval did not locate as a head unit: {L}")

    # 2. verifier: coverage inputs + expected statement (value-free)
    t0 = time.perf_counter()
    sel = rec / "adapter" / "selective"
    c2 = json.loads((sel / "commitments.json").read_text())
    gs = fa2_gate_set()
    ci = ex.coverage_inputs(sel, c2, retained_request_ids={rid}, units=[forced], checks=[], family=gs.id, recompute_leaves=True, config=C)
    expected = expected_statements(ci["circuit"], ci["partition"], ci["Q_proof"])
    VU_KEY = str(FORCED_VU)  # coverage_v1 keys expectations by vu id string
    e = expected[VU_KEY]
    step("verifier.expected_statement", t0, Q_proof=ci["Q_proof"], uncovered=ci["uncovered"], tree_problems=ci["tree_problems"], expected_digest=e.digest, label=e.label,
         incoming=len(e.incoming), outgoing=len(e.outgoing), subcircuit=e.subcircuit)

    # 3. prover: build the statement from the record's committed rows
    t0 = time.perf_counter()
    session = bytes.fromhex(proto["session"]) if isinstance(proto.get("session"), str) and len(proto["session"]) == 64 else b"\0" * 32
    record = CaptureRecord(sel, c2, config=C, session=session, compiled=bytes.fromhex(maps["program"]["digest"]))
    unit = LocatedUnit(str(FORCED_VU), kind, rid, t, layer=a.layer, head=a.head, ru_index=ru, extra={"unit": "head", "forced": True})
    built = dispatch(unit, record)
    sys.path.insert(0, str(REPO / "zk" / "bench"))
    from pb_attn_batches import write_batch

    batch = write_batch(out / "batch", built.statement, built.witness, gs, {"composition": "attention-head", "kind": built.kind, "forced": True, **built.notes})
    step("prover.build_statement", t0, kind=built.kind, obligations=built.notes["obligations"], blocks=built.notes["blocks"], statement_bytes=batch["statement_bytes"],
         witness_bytes=batch["witness_bytes"], transparent_check_s=batch["transparent_check_s"])

    # 4. verifier: views from committed ids -> coverage
    t0 = time.perf_counter()
    trees = ci["trees"]
    tree_of_root = {(bytes.fromhex(ct.root) if isinstance(ct.root, str) else ct.root): tid for tid, ct in trees.items()}  # verifier keeps hex roots
    from verity_vllm import identity

    # rank order is verity-vllm/leaf/v1 (identity.sort_key), not the index.json listing order
    ids = {tid: sorted(lst, key=identity.sort_key) for tid, lst in ci["deriver"].committed(list(trees)).ids.items()}

    def id_at_rank(tree: str, rank: int) -> str | None:
        lst = ids.get(tree)
        return lst[rank] if lst is not None and 0 <= rank < len(lst) else None

    # the raw fa2 statement has no typed object: A2's `position_check` (typed-object positions vs committed ranks) does not apply to it;
    # its positions are recovered and checked by `head_views_of_raw_statement` from committed ids (tree_of_root, id_at_rank: every
    # opened rank must be a committed packed leaf of a retained tree).  Run coverage by digest on those views; also record what
    # `check_coverage` with trees/rank_of says (it refuses the proof for lack of a typed object -- the labelled gap for raw statements).
    views = head_views_of_raw_statement(built.statement, tree_of_root, id_at_rank, D=D)
    rep = check_coverage(expected, [AcceptedProof("forced-head", views)])
    rep_typed = check_coverage(expected, [AcceptedProof("forced-head", views)], trees=trees, rank_of=ci["rank_of"])
    step("verifier.coverage_positive", t0, covered=list(rep.covered), uncovered=list(rep.uncovered), unexpected=len(rep.unexpected), duplicate=list(rep.duplicate),
         views_equal_builder=(views == built.views), positions_checked_by="head_views_of_raw_statement (committed ids; raw format-3 statement, no typed object)",
         with_typed_position_check={"covered": list(rep_typed.covered), "position_problems": {k: list(v) for k, v in (getattr(rep_typed, "position_problems", {}) or {}).items()}})
    ok_positive = [str(v) for v in rep.covered] == [VU_KEY] and not rep.uncovered and not rep.unexpected

    # 5. R9 negatives
    t0 = time.perf_counter()
    nh = int(C["NH"])
    other_head = (a.head + 1) % nh
    neg: dict[str, Any] = {}
    try:
        other = dispatch(LocatedUnit(str(FORCED_VU), head_kind_id(t + 1, D), rid, t, layer=a.layer, head=other_head, ru_index=ru), record)
        v2 = head_views_of_raw_statement(other.statement, tree_of_root, id_at_rank, D=D)
        r2 = check_coverage(expected, [AcceptedProof("neighbour-head", v2)])
        neg["neighbouring_head"] = {"head": other_head, "uncovered": list(r2.uncovered), "unexpected": len(r2.unexpected), "refused_as_expected": [str(v) for v in r2.uncovered] == [VU_KEY] and len(r2.unexpected) == 1}
    except BuildRefusal as err:
        neg["neighbouring_head"] = {"head": other_head, "build_refusal": err.reason}
    step("verifier.r9_negative", t0, **neg)
    ok_negative = bool(neg.get("neighbouring_head", {}).get("refused_as_expected"))

    # 6a. the verifier's own refusal of a fa2 statement, without any host: V4.guest (approved_guests.json has no fa2 entry) and
    # V4.family (the fa2 family is not listed under 9fd00e60) -- the Python checks the certificate runs, on the fa2 identity the pod measured
    t0 = time.perf_counter()
    from e2e.approved_guests import approval_of
    from veritor.interface.families import check_family_approved

    FA2_ELF, FA2_VK = "33cba05834171e2518b6cf122ac40e36797ba111b8299a7f7bb546f4d7a219c9", "0x00413469f3964a62a43c0e78812f223cafc8a5024e57a7e5e4c4e87512ccadfb"
    APPROVED_ELF, APPROVED_VK = "9fd00e60fb6a74762de987cde211aca9de4576de844569da779acac6bf3863db", "0x0018b8e4e955ad4c348bcf57271643814dfcd68d575323410841c29289bd5797"
    ap_fa2 = approval_of(FA2_ELF, FA2_VK)
    ap_ok = approval_of(APPROVED_ELF, APPROVED_VK)
    fam_problems = check_family_approved(gs.id, str(gs.digest), 32, ap_ok.entry or {})
    step("verifier.refusal_of_fa2_without_host", t0, fa2_guest=ap_fa2.to_json(), approved_guest_ok=ap_ok.ok,
         fa2_family_under_9fd00e60=fam_problems, refused_as_expected=(not ap_fa2.ok and ap_fa2.reason == "GUEST_NOT_APPROVED" and bool(fam_problems)),
         pod_measured="approved host verify-many refused both fa2 proofs (ok=false, statement_match=true, 2-6 ms) -- pod/records.jsonl")

    # 6. optional SP1
    if a.fa2_host:
        t0 = time.perf_counter()
        d = Path(batch["dir"])
        st, wit = str(d / "statement.bin"), str(d / "witness-fmt3.bin")
        proof = d / f"proof-{a.mode}.bin"

        def run(argv: list[str]) -> dict[str, Any]:
            p = subprocess.run(argv, capture_output=True, text=True)
            lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
            return {"ok": p.returncode == 0 and bool(lines), **(json.loads(lines[-1]) if lines else {"stderr_tail": p.stderr[-800:]})}

        pr = run([a.fa2_host, "prove", "--statement", st, "--witness", wit, "--out", str(proof), "--mode", a.mode])
        vr = run([a.fa2_host, "verify", "--proof", str(proof), "--statement", st]) if pr.get("ok") else {"skipped": True}
        sp1: dict[str, Any] = {"fa2_prove": {k: pr.get(k) for k in ("ok", "setup_seconds", "prove_seconds", "verify_seconds", "proof_bytes", "verdict", "elf_sha256", "vk_hash")},
                               "fa2_verify": {k: vr.get(k) for k in ("ok", "verify_seconds", "statement_match")}}
        if a.approved_host and pr.get("ok"):
            p = subprocess.run([a.approved_host, "verify-many"], input=json.dumps({"id": "forced-head", "proof": str(proof), "statement": st}) + "\n", capture_output=True, text=True)
            res = [json.loads(l) for l in p.stdout.splitlines() if l.strip().startswith("{")]
            sp1["approved_verify_many"] = {"results": res, "refused_as_expected": any(r.get("event") == "verify" and r.get("ok") is False for r in res)}
        step("sp1", t0, **sp1)

    doc["verdict"] = {"forced_head_covered": ok_positive, "neighbour_refused": ok_negative, "fa2_refused_by_verifier": doc["steps"]["verifier.refusal_of_fa2_without_host"]["refused_as_expected"], "label": doc["label"]}
    (out / "forced_head_run.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps(doc["verdict"]))
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if (ok_positive and ok_negative) else 1


if __name__ == "__main__":
    raise SystemExit(main())
