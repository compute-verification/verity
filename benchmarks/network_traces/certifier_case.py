"""certifier-case: ``verity/certifier/v0``'s case (`NetworkCheck.Period`'s formats) of a certifier session's run and the
period's compute audits, which ``network-period check`` (`block_certifier.check`) decides and `period_case`, its Python
mirror, re-decides.

* `run_windows`: a run's link-windows as N2's verifier directory (`n2_zk.member`) and the client's certificates hold
  them, with the link's one-stage v2 parameters and circuit file.
* `audit_answer`: a period's compute audit as one-stage answers it at δ, from the integrity check's answer and the
  audit record.
* `case_of` (the clock form) and `rows_case_of` (the rows form of a warden run's link-windows and their scans): the
  case, its layout terms (`layout`: each audit's free inputs and Program, and the link's observation horizon).
* `term_delta` and `holds_at`: the period check's ``eps`` is the sum of its terms' δ, so a case whose answer is to
  hold at δ decides each term at its share, and an accepted answer whose ``eps`` isn't δ had a term the split missed.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import certifier_serve as cs

from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R
from verity.ml.circuits.codec import canonical_json

PROTOCOL = "verity/certifier/v0"
ONE_STAGE = R.PROTOCOL


def audit_answer(reg: Mapping[str, Any], integrity: Mapping[str, Any], record: Mapping[str, Any] | None,
                 delta: Fraction) -> dict[str, Any]:
    """The period's compute audit as one-stage answers it at ``delta``: the integrity check's refusal as it is; on its
    acceptance the audit record's draw and proved units, with one-stage's profile and bound at ``delta``."""
    if integrity["verdict"] != "accept":
        return dict(integrity)
    if record is None or record.get("accepted") is not True or R.digest(record["registration"]) != R.digest(reg):
        raise RuntimeError("the integrity check accepted, but its audit record is missing or for another registration")
    draw = record["draw"]
    profile = A.profile(reg, draw, delta).to_json()
    profile["delta"] = str(delta)
    wrong = A.wrong_units(reg, draw, delta)
    return {"commitment": R.digest(reg), "profile": profile, "protocol": ONE_STAGE,
            "public": {"bound": wrong["bound"], "drawn": sorted(draw["units"]), "per_template": wrong.get("per_template"),
                       "proved": sorted(record["proved_units"])}, "reason": None, "verdict": "accept"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _cert(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {k: doc[k] for k in ("digest", "direction", "key_id", "link", "signature", "window")}


def run_windows(sess: Mapping[str, Any], strategy: str) -> dict[str, Any]:
    """A run of the session: its deployment, plan and calibration, N2's one-stage v2 parameters for the link
    (``one_stage``, `n2_zk.one_stage_params`) and its circuit file, and per link-window N2's registration, one-stage v2
    evidence (`n2_zk.one_stage_evidence`, None for a link-window N2 decided without an audit record, a late one), the
    salts its prover delivered for the ``pub`` and ``o0`` rows, the files its proof names (``blobs``: the public file
    and the sessions' files), and the client's certificates (egress first) and clock certificate."""
    run = Path(sess["dir"]) / "rehearsal" / strategy
    v = Path(sess["dir"]) / "n2zk" / strategy / "verifier"
    member = json.loads((Path(sess["dir"]) / "n2zk" / "summary.json").read_text())["members"].get(strategy) or {}
    decided = {r["window"]: r for r in member.get("windows") or []}
    certs, clocks = _jsonl(run / "client" / "certificates.jsonl"), _jsonl(run / "client" / "clocks.jsonl")
    windows = {}
    for w in sorted(decided):
        pre = f"w{w:03d}-"
        one = v / f"{pre}one-stage.json"
        mine = {c["direction"]: _cert(c) for c in certs if c.get("window") == w and "digest" in c}
        clock = next((c for c in clocks if c.get("window") == w and "digest" in c), None)
        windows[w] = {"registration": json.loads((v / f"{pre}registration.json").read_text()),
                      "one_stage": json.loads(one.read_text()) if one.is_file() else None,
                      "salts": json.loads((v / f"{pre}delivered.json").read_text())["salts"],
                      "blobs": [str(v / f"{pre}pub-1.bin"), *sorted(str(f) for f in v.glob(f"{pre}session-*"))],
                      "certificates": [mine[d] for d in ("egress", "ingress") if d in mine],
                      "clock": None if clock is None else clock_evidence(clock)}
    return {"run": run, "deployment": json.loads((run / "deployment.json").read_text()),
            "plan": json.loads((run / "plan.json").read_text()),
            "calibration": json.loads((run / "calibration.json").read_text()),
            "one_stage": json.loads((v / "one-stage.json").read_text()) if (v / "one-stage.json").is_file() else None,
            "circuit": str(v / "circuit.txt"), "windows": windows}


def clock_evidence(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """A link-window's clock in the period check's evidence: the client's salted clock certificate alone (a line of
    its ``clocks.jsonl`` is the session and the certificate's fields), which signs the sync's clock digest under the
    certifier's salt (`certificate.clock_commitment`); the sync and its salt are witness and never in the case."""
    return _cert(certificate)


def layout(free: Sequence[int], programs: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    """The layout terms: each audit's free inputs and Program, and the link's observation horizon."""
    if len(free) != len(programs):
        raise ValueError(f"{len(free)} free-input counts for {len(programs)} audits' Programs")
    return {"audits": [{"free": f, "program": p, "steps": horizon} for f, p in zip(free, programs)], "cells": [],
            "jitter": 1}


def _end(grid: Any, profile: Sequence[Sequence[int]] | None) -> dict[str, Any]:
    """An end of the link as certifier/v0 reads it: its grid (`grid.GridParams`) and occupancy profile, or null."""
    return {"grid": {k: getattr(grid, k) for k in ("allocation", "bucket_ms", "buckets", "frame_bytes", "queue_capacity")},
            "profile": None if profile is None else [list(row) for row in profile]}


def case_of(rw: Mapping[str, Any], windows: Sequence[int], audits: Sequence[tuple[dict, dict]], first: int,
            free: Sequence[int], programs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``verity/certifier/v0``'s case in the clock form of the link-windows ``windows`` of a run (`run_windows`) and
    the compute ``audits`` (registration, answer), each with its ``free`` inputs and its Program; the observation
    horizon is the link-windows' buckets. The link is N2's on the run's planned profile (`n2_zk.link`): its ends' grids
    and occupancy profiles, its run, and N2's one-stage v2 parameters with ``log`` the registrations of the run's
    link-windows before ``first``, which the period check runs v2 under on each link-window's evidence and delivered
    salts. Its ``blobs`` are the paths of the files those proofs name (the circuit file, each link-window's public file
    and sessions' files), which `block_certifier.check` hands ``network-period check``."""
    if len(free) != len(audits):
        raise ValueError(f"{len(free)} free-input counts for {len(audits)} audits")
    if rw["one_stage"] is None:
        raise RuntimeError("N2 wrote no one-stage v2 parameters for the run (n2_zk.one_stage_params)")
    dep, setup = rw["deployment"], cs.ar.Setup(rw["calibration"])
    key = cs.read_public(dep["certifier_key"])
    N, W, H = dep["certifier_key"]["params"]
    occupancy = rw["plan"].get("occupancy") or {}
    xs = [rw["windows"][w] for w in windows]
    log = [x["registration"] for w, x in sorted(rw["windows"].items()) if w < first]
    params = {**layout(free, programs, len(xs) * cs.WINDOW_BUCKETS),
              "egress": _end(setup.egress_params, occupancy.get("egress")), "first": first,
              "ingress": _end(setup.ingress_params, occupancy.get("ingress")), "key": key.public.hex(),
              "link": dep["links"]["n2"], "lms": {"H": H, "N": N, "W": W, "toy": False},
              "one_stage": {**rw["one_stage"], "log": log}, "run": dep["run_id"], "session": dep["session"],
              "sync_set": {"deltas": list(setup.sync_set.deltas),
                           "rhos": [[r.numerator, r.denominator] for r in setup.sync_set.rhos]},
              "windows": len(xs)}
    commitment = canonical_json({"audits": [r for r, _ in audits], "windows": [x["registration"] for x in xs]})
    evidence = canonical_json({"audits": [a for _, a in audits],
                               "windows": [{"certificates": x["certificates"], "clock": x["clock"],
                                            "one_stage": x["one_stage"], "salts": x["salts"]} for x in xs]})
    blobs = [rw["circuit"], *(b for x in xs for b in x["blobs"])]
    return {"blobs": blobs, "commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}


def rows_case_of(rw: Mapping[str, Any], windows: Sequence[int], scans: Mapping[int, tuple[dict, dict, dict]],
                 audits: Sequence[tuple[dict, dict]], first: int, free: Sequence[int],
                 programs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``verity/certifier/v0``'s case in the rows form (`NetworkCheck.Period`'s) of a warden run's link-windows
    ``windows``, each with its scan's registration, answer and Program's descriptor (``scans[w]``), and the compute
    ``audits`` with their ``free`` inputs and Programs: the network certifier's one schedule and key from the run's
    deployment, the scan's descriptor, and the audits' horizon the link-windows' buckets."""
    import warden_serve as WS

    if len(free) != len(audits):
        raise ValueError(f"{len(free)} free-input counts for {len(audits)} audits")
    dep = rw["deployment"]
    key, T = WS.cs.read_public(dep["key"]), dep["params"]["window"]
    N, W, H = dep["key"]["params"]
    xs, sc = [rw["windows"][w] for w in windows], [scans[w] for w in windows]
    program, descs = {r["program"] for r, _, _ in sc}, {canonical_json(d) for _, _, d in sc}
    if len(program) != 1 or len(descs) != 1:
        raise RuntimeError(f"the link-windows' scans register {len(program)} Programs and {len(descs)} descriptors, "
                           "not the scan's one")
    params = {**layout(free, programs, len(xs) * T), "key": key.public.hex(), "link": WS.LINK,
              "lms": {"H": H, "N": N, "W": W, "toy": False}, "program": program.pop(),
              "scan": sc[0][2], "schedule": dict(dep["params"]),
              "session": dep["session"], "windows": len(xs), "first": first}
    commitment = canonical_json({"audits": [r for r, _ in audits], "windows": [r for r, _, _ in sc]})
    evidence = canonical_json({"audits": [a for _, a in audits], "windows": [
        {"answer": a, "certificates": [x["certificates"][n] for n in (WS.INGRESS, WS.EGRESS, WS.FAILS)],
         "registrations": [x["registrations"][d] for d in ("ingress", "egress")]} for x, (_, a, _) in zip(xs, sc)]})
    return {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}


def term_delta(delta: Fraction, terms: int) -> Fraction:
    """The δ each of a case's ``terms`` is decided at for its answer to hold at ``delta``: `NetworkCheck.Period`'s
    ``eps`` is the sum of its terms' δ (the clock form's compute audits', or the rows form's link-windows' scans', whose
    compute audits are acceptance conditions in no term; the clock form's link-window answers add none), so each takes
    an equal share."""
    if terms < 1:
        raise ValueError(f"a case has at least one term, not {terms}")
    return delta / terms


def holds_at(index: int, a: Mapping[str, Any], delta: Fraction) -> None:
    """An accepted answer's ``eps`` is the ``delta`` its terms were split for (`term_delta`), else an error."""
    if a["verdict"] == "accept" and Fraction(a["public"]["eps"]) != delta:
        raise RuntimeError(f"period {index}: eps {a['public']['eps']} for δ = {delta}: the case has a term its split "
                           "(term_delta) didn't count")
