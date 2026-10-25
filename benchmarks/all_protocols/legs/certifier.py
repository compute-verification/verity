"""The communication leg: the network certifier over the node's link, decided per period by `NetworkCheck.Period`
(``verity/certifier/v0``, the ``network-period`` executable) on a case built from one certifier session that spans the
run's periods (`verity_all_protocols.table`'s interface: ``leg(period_dir, node, out) -> answer``).

**The session** (`session`, once a run, in the runner's scratch beside the period's window, ``certifier-session/``). At
the first period the leg reads the period's requests off its value files (`requests_of`: each request's prompt token is
the one whose embedding the loop's first linear took, its steps the loop's) and serves the run's periods of them
(``periods`` in ``period.json``) through the network certifier: `certifier_serve.py rehearse` on the fixed traffic and
N2's planned profile, each period's requests one group in a link-window of its own (``--prompts``,
``--window-per-group``), so period ``i`` is link-window ``i``, and the loop's model behind it (`torch_loop_stub.py`, the
loop's W8A8 MLP, each step's 256 output bytes as hex text). Beside the honest run, the same plan with the shim's
`CERTIFIER_TAMPER` (``extra`` by default: one event more after window 0's first response). The client keeps and verifies
each window's certificates and salted clock certificates. Then N2's verdict of record on every link-window of both runs,
each proved ``--zk`` behind the Lean firewall (`n2_zk.py prove`): its exit and leak scan are in ``summary.json``
(`n2_report`), and a leak, a scan whose control missed a planted piece, or a nonzero exit refuses the session
(`n2_refusal`). The session is one LMS key (height `CERTIFIER_HEIGHT`) and one certifier session per run, its leaf state
kept across both. Its records (heads and salts, syncs, request and response records, the shim's events, the plan's
prompts) are witness and stay in the scratch; the runner's ``--out/certifier-session/`` gets each run's deployment,
calibration and the client's certificates and clock certificates alone (`publish`, `PUBLIC_SESSION`), and
``summary.json`` the session's counts.

**A period** (`leg`): its link-window, and for the first period every link-window N2 decided before it and for the last
every one after (`period_windows`), so each certified link-window is some period's. Their certified outputs (the egress
records' canonical events, each response F of them in the plan's order) must be the period's registered outputs, the
value files' last-linear outputs, request by request (`io`, the leg's own check, written to ``io.json``): while the
integrity leg accepts, a mismatch is the leg's error; when it refuses (a loop-tampered period), the mismatch is recorded
(``refused``) and the case decides on that audit. The case (`case_of`) is the period's link-windows, each with its N2
registration, its one-stage answer from N2's outcome of record (Selection All: its one unit, `answer_of_outcome`), its
two certificates and its clock certificate as the client kept them, the salted certificate alone (`clock_evidence`: the
sync and its salt are witness); and the period's compute audit, the integrity leg's registration and audit record
re-decided by one-stage's own bound at δ (`audit_answer`). Its answer is ``network-period check``'s at δ = 1/100
(`DELTAS`), and ``out/`` also holds the case and answer at δ = 10⁻⁶ and the tampered run's (``tamper.*``). The period
holding the tampered link-window (`TAMPERED_WINDOW`) must reject there, which the leg asserts; the tampered run's other
periods may accept.

Layout terms (`layout`): the audit's free inputs (`free_inputs`, from the window's ``circuit.json``: every unit's
input-row bits, less the rows wires feed or ``external`` names), the observation horizon the link's (its link-windows'
buckets, `certifier_serve.WINDOW_BUCKETS` each), jitter bound 1 (N2 holds the planned profile's egress counts exactly,
so no observation moves), and no connection through the network certifier but the link, whose counts the profile fixes
(``cells`` empty).
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R
from verity.ml.circuits.codec import canonical_json

PROTOCOL = "verity/certifier/v0"
ONE_STAGE = R.PROTOCOL
ROOT = Path(__file__).resolve().parents[3]
TRACES = ROOT / "benchmarks" / "network_traces"
LEAN = ROOT / "verity" / "core" / "protocols" / "network" / "lean"
SESSION = "certifier-session"
#: the δ of every compute audit's bound in a period's case: the answer's, then the one written beside it
DELTAS = (Fraction(1, 100), Fraction(1, 10 ** 6))
#: N2's stream decides at `certifier_serve.DELTA`
N2_DELTA = Fraction(2) ** -40
#: what the leg's verdict still stands on that isn't the network certifier's own, each with its fix (``stubbed`` in
#: ``summary.json``)
STUBBED = (
    {"gap": "the records are certifier_serve.py's certifier-only path (active.Proxy), not the network certifier's own "
            "cells as one device",
     "fix": "the gate PR: warden-rehearse --prompts FILE --window-per-group on vy-challenger-gate, the loop stub "
            "answering id-form (io_tie.id_event, P = 503), and the leg reading the device's cells.jsonl and certified "
            "records"},
    {"gap": "the device runs off-host, beside the server on the node, not on the auditor's machine (vy-challenger-1) "
            "through the gate",
     "fix": "the same gate PR: the leg serves the stub on a peer and runs warden-rehearse through the gate with the "
            "lane's gate key"})
#: a canonical event's most bytes (`active.TOKEN_BYTES`)
TOKEN_BYTES = 64
#: what `publish` copies of each run of the session into the runner's ``--out``: nothing the prover holds as witness
PUBLIC_SESSION = ("deployment.json", "calibration.json", "client/certificates.jsonl", "client/clocks.jsonl")
#: the link-window each shim tamper changes (`certifier_serve.py`'s ``late``, ``extra``, ``drop``), whose period's case
#: must reject; ``token`` changes bytes only the I/O tie reads, which the network check accepts
TAMPERED_WINDOW = {"late": 0, "extra": 0, "drop": 0}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


# --- the period's records ----------------------------------------------------------------------------------------------

def loop_units(period_dir: Path) -> dict[int, tuple[dict, dict[str, np.ndarray]]]:
    """Every unit of the period's window: its label (`index.json`: request, step, layer) and its rows by port, u16 words,
    read off the member's value file (``m<k>-private.bin``: every instance's rows, ports in order, then the salts)."""
    win = json.loads((period_dir / "window.json").read_text())
    labels = {m["name"]: m["labels"] for m in json.loads((period_dir / "index.json").read_text())["members"]}
    out: dict[int, tuple[dict, dict[str, np.ndarray]]] = {}
    for m in win["members"]:
        names, widths = [p for p, _ in m["ports"]], [w for _, w in m["ports"]]
        W, n = sum(widths), m["n"]
        rows = np.frombuffer((period_dir / f"m{m['k']}-private.bin").read_bytes()[:2 * n * W], "<u2").reshape(n, W)
        at = dict(zip(names, np.cumsum([0, *widths[:-1]]).tolist()))
        units = json.loads((period_dir / m["units"]).read_text())["indices"]
        label = {}
        for lo, hi, lab in labels[m["name"]]:
            label.update({u: lab for u in range(lo, hi)})
        for i, u in enumerate(units):
            out[u] = (label[u], {p: rows[i, at[p]:at[p] + w] for p, w in zip(names, widths)})
    return out


def free_inputs(circuit: Mapping[str, Any]) -> dict[str, Any]:
    """The period's free inputs, ``#(freeInputs \\ reg)`` of `NetworkCheck.Period`'s count, from the window's
    ``circuit.json``: every unit's input-row bits (each template's units times its statement's ``in_rows`` widths), less
    each unit's rows whose every port a wire feeds from another unit's output (``wires``) or ``external`` names as the
    system's I/O. ``bits`` the total, ``per_template`` each template's. A row the registration's ``anchors`` register
    still counts: anchors name no rows yet, and counting a registered row free only overcounts K."""
    from verity.ml.windows.interface import Wire

    units = {t: n for t, n in circuit["templates"]}
    covered: dict[tuple[str, str], set[int] | None] = {}
    for e in circuit["external"]:
        t, port = e.rsplit("/", 1)
        if t not in units:
            raise ValueError(f"external port {e!r} is no template's")
        covered[t, port] = None
    for w in map(Wire.of, circuit["wires"]):
        if w.reader not in units:
            raise ValueError(f"a wire's reader {w.reader!r} is no template")
        fed = covered.setdefault((w.reader, w.port), set())
        if fed is not None:
            fed.update(range(w.at, w.at + w.units))
    per = {}
    for t, n in units.items():
        bits = 0
        for _, _, width, ports in circuit["statements"][t]["in_rows"]:
            fed = [covered.get((t, p), set()) for p in ports]
            gone = 0 if not fed else n if all(f is None for f in fed) else len(set(range(n)).intersection(
                *(set(range(n)) if f is None else f for f in fed)))
            bits += (n - gone) * width
        per[t] = bits
    return {"bits": sum(per.values()), "per_template": per}


def registered_outputs(units: Mapping[int, tuple[dict, dict]]) -> dict[tuple[str, int], bytes]:
    """Each request's step outputs as the window registered them: the last linear's units' ``o0`` words in unit order,
    2H bytes a step, by (request, step)."""
    last = max(lab["layer"] for lab, _ in units.values())
    got: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for u, (lab, rows) in units.items():
        if lab["layer"] == last:
            got.setdefault((lab["request"], lab["step"]), []).append((u, int(rows["o0"][0])))
    return {k: np.array([w for _, w in sorted(v)], "<u2").tobytes() for k, v in sorted(got.items())}


def requests_of(period_dir: Path, units: Mapping[int, tuple[dict, dict]] | None = None) -> tuple[list[int], int, int]:
    """(each request's prompt token, the loop's steps, the MLP's width H): a request's token is the one whose quantized
    embedding is the ``x`` row its first linear took at step 0 (`loop.TorchLoop.run`)."""
    sys.path.insert(0, str(TRACES))
    import torch_loop_stub as S

    units = loop_units(period_dir) if units is None else units
    name = json.loads((period_dir / "circuit.json").read_text())["name"]
    h = int(re.fullmatch(r"torch-loop/mlp-w8a8-h(\d+)", name).group(1))
    model = S.L.Model(h=h)
    table = {S.L.bits(S.L.quantize(model.embed[t][None, :])[0]).tobytes(): t for t in range(S.L.VOCAB)}
    first: dict[str, tuple[int, bytes]] = {}
    steps = 1 + max(lab["step"] for lab, _ in units.values())
    for u, (lab, rows) in sorted(units.items()):
        if lab["step"] == 0 and lab["layer"] == 0 and lab["request"] not in first:
            first[lab["request"]] = (u, rows["x"].astype("<u2").tobytes())
    tokens = []
    for r in sorted(first, key=lambda r: int(r.lstrip("r"))):
        x = first[r][1]
        if x not in table:
            raise RuntimeError(f"request {r}'s first input is no token's embedding under the loop's model")
        tokens.append(table[x])
    return tokens, steps, h


# --- the session ---------------------------------------------------------------------------------------------------------

def _tools(period_dir: Path) -> dict[str, str]:
    return json.loads((period_dir / "period.json").read_text())["tools"]


def lean_exes() -> dict[str, str]:
    """``network-period`` and ``network-records``, built in the tree."""
    p = subprocess.run(["lake", "build", "network-period", "network-records"], cwd=LEAN, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"lake build network-period network-records: {p.stdout[-2000:]}{p.stderr[-2000:]}")
    return {n: str(LEAN / ".lake" / "build" / "bin" / n) for n in ("network-period", "network-records")}


def _stub(steps: int, h: int, log: Path) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen([sys.executable, str(TRACES / "torch_loop_stub.py"), "--steps", str(steps), "--h", str(h)],
                            stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, cwd=ROOT)
    line = json.loads(proc.stdout.readline() or "{}")
    if "listening" not in line:
        proc.kill()
        raise RuntimeError(f"the stub exited before listening ({log})")
    return proc, line["listening"]


def _keep(src: Path, dst: Path) -> None:
    """`n2_zk.sh`'s keep: everything but a prover's file (``inst-*``, ``prover-instances.bin``) or a circuit."""
    for f in src.rglob("*"):
        if f.is_file() and f.name != "circuit.txt" and not f.name.startswith("inst-") and f.name != "prover-instances.bin":
            (dst / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst / f.relative_to(src))


def _run(argv: Sequence[str], log: Path, env: Mapping[str, str] | None = None) -> int:
    with open(log, "w") as f:
        return subprocess.run(list(argv), cwd=ROOT, env=dict(env or os.environ), stdout=f, stderr=subprocess.STDOUT).returncode


def session(period_dir: Path) -> dict[str, Any]:
    """The run's one certifier session (module docstring), made at its first period in the runner's scratch beside the
    period's window (``period_dir.parent``); its record ``session.json``."""
    s = period_dir.parent / SESSION
    if (s / "session.json").exists():
        return json.loads((s / "session.json").read_text())
    s.mkdir(parents=True, exist_ok=True)
    tools, exes = _tools(period_dir), lean_exes()
    prompts, steps, h = requests_of(period_dir)
    period = json.loads((period_dir / "period.json").read_text())
    if not isinstance(period.get("periods"), int) or period["periods"] < 1:
        raise RuntimeError("period.json names no period count (`periods`), which the session's plan serves")
    periods, height = period["periods"], _env_int("CERTIFIER_HEIGHT", 10)
    tamper = os.environ.get("CERTIFIER_TAMPER", "extra")
    strategies = ["honest", *([tamper] if tamper else [])]
    frames = steps * math.ceil(4 * h / TOKEN_BYTES)
    env = {**os.environ, "VERITY_NETWORK_RECORDS_EXE": exes["network-records"],
           "PYTHONPATH": os.pathsep.join(filter(None, [tools["m0-pythonpath"], str(TRACES),
                                                       str(ROOT / "tools" / "verity" / "circuit_check" / "src"),
                                                       os.environ.get("PYTHONPATH")]))}
    stub, url = _stub(steps, h, s / "stub.log")
    try:
        rc = _run([sys.executable, str(TRACES / "certifier_serve.py"), "rehearse", "--upstream", url,
                   "--out", str(s / "rehearsal"), "--traffic", "fixed", "--profile", "planned",
                   "--prompts", ";".join(map(str, prompts)), "--requests", str(len(prompts) * periods),
                   "--in-flight", str(len(prompts)), "--tokens", str(steps), "--response-frames", str(frames),
                   "--probes", str(len(prompts)), "--window-per-group", "--strategies", ",".join(strategies),
                   "--height", str(height)], s / "rehearse.log", env)
    finally:
        stub.terminate()
        stub.wait()
    if not (s / "rehearsal" / "summary.json").exists():
        raise RuntimeError(f"certifier_serve.py rehearse exited {rc} with no summary ({s / 'rehearse.log'})")
    scratch = Path(tempfile.mkdtemp(prefix="certifier-leg-n2-"))
    try:
        n2 = [sys.executable, str(TRACES / "n2_zk.py"), "prove", "--run", str(s / "rehearsal" / "honest"),
              "--rounds", ",".join(strategies), "--bin", tools["flock-circuit"], "--lean", tools["flock-verify"],
              "--out", str(scratch / "n2zk"), "--jobs", str(len(strategies))]
        if tamper in ("extra", "late"):
            n2 += [f"--{tamper}", str(s / "rehearsal" / tamper)]
        n2_rc = _run(n2, s / "n2zk.log", {**env, "FC_FIREWALL_LEAN": tools["flock-firewall"]})
        _keep(scratch / "n2zk", s / "n2zk")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if not (s / "n2zk" / "summary.json").exists():
        raise RuntimeError(f"n2_zk.py prove exited {n2_rc} with no summary ({s / 'n2zk.log'})")
    rec = {"prompts": prompts, "steps": steps, "h": h, "periods": periods, "height": height, "strategies": strategies,
           "tamper": tamper or None, "frames": frames, "rehearse_exit": rc, "n2_exit": n2_rc, "exes": exes,
           "dir": str(s)}
    (s / "session.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def publish(sess: Mapping[str, Any], out: Path) -> None:
    """The session's public files into ``out`` (the runner's ``--out/certifier-session``), each run's alone: its
    deployment, calibration and the client's certificates and clock certificates (`PUBLIC_SESSION`). The records with
    their heads and salts, the syncs, the request and response records, the shim's events and the plan's prompts stay
    in the session, in the runner's scratch."""
    for strategy in sess["strategies"]:
        run = Path(sess["dir"]) / "rehearsal" / strategy
        for f in PUBLIC_SESSION:
            if (run / f).is_file():
                (out / strategy / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(run / f, out / strategy / f)


def n2_report(sess: Mapping[str, Any]) -> dict[str, Any]:
    """N2's prove over the session: its exit, and each member's ok and leak scan (``leaked``, the witness pieces found
    in what the verifier received, ``structured``, and whether the scan's control found every piece it planted)."""
    s = json.loads((Path(sess["dir"]) / "n2zk" / "summary.json").read_text())
    members = {}
    for name, m in (s.get("members") or {}).items():
        rec = m.get("received") or {}
        members[name] = {"ok": m.get("ok"), "leaked": rec.get("leaked"), "verifier_found": rec.get("verifier_found"),
                         "structured": rec.get("structured"),
                         "control_holds": (rec.get("control") or {}).get("holds_all")}
    return {"exit": sess["n2_exit"], "ok": s.get("ok"), "members": members}


def n2_refusal(report: Mapping[str, Any]) -> str | None:
    """Why N2's prove refuses the session, or None: a member whose scan found witness in what the verifier received, or
    whose control didn't find every piece it planted (no scan to trust), or a nonzero exit."""
    for name, m in report["members"].items():
        if m["leaked"] or m["verifier_found"] or m["structured"]:
            return (f"n2_zk.py prove: member {name} leaked witness to the verifier (leaked {m['leaked']}, found "
                    f"{m['verifier_found']}, structured {m['structured']})")
        if m["control_holds"] is not True:
            return f"n2_zk.py prove: member {name}'s leak scan control didn't find every piece it planted"
    if report["exit"] != 0:
        bad = [n for n, m in report["members"].items() if not m["ok"]]
        return f"n2_zk.py prove exited {report['exit']}: members {bad} not as expected"
    return None


# --- the case ------------------------------------------------------------------------------------------------------------

def answer_of_outcome(reg: Mapping[str, Any], draw: Mapping[str, Any], accepted: bool, reason: str | None,
                      delta: Fraction = N2_DELTA) -> dict[str, Any]:
    """One-stage's answer for a link-window's registration from N2's outcome of record: on acceptance its one unit drawn
    and proved, the profile and bound one-stage's (`audit.profile`, `audit.wrong_units`) at the stream's δ."""
    commitment = R.digest(reg)
    if not accepted:
        return {"commitment": commitment, "profile": None, "protocol": ONE_STAGE, "public": None,
                "reason": reason or "N2's verifier of record refused the link-window", "verdict": "reject"}
    law = dict(reg["law"])
    units = sorted(draw.get("units") or range(reg["population"]))
    d = {**law, "units": units}
    profile = A.profile(reg, d, delta).to_json()
    profile["delta"] = str(delta)
    wrong = A.wrong_units(reg, d, delta)
    return {"commitment": commitment, "profile": profile, "protocol": ONE_STAGE,
            "public": {"bound": wrong["bound"], "drawn": units, "per_template": wrong.get("per_template"),
                       "proved": units}, "reason": None, "verdict": "accept"}


def audit_answer(reg: Mapping[str, Any], integrity: Mapping[str, Any], record: Mapping[str, Any] | None,
                 delta: Fraction) -> dict[str, Any]:
    """The period's compute audit as one-stage answers it at ``delta``: the integrity leg's refusal as it is; on its
    acceptance the audit record's draw and proved units, with one-stage's profile and bound at ``delta``."""
    if integrity["verdict"] != "accept":
        return dict(integrity)
    if record is None or record.get("accepted") is not True or R.digest(record["registration"]) != R.digest(reg):
        raise RuntimeError("the integrity leg accepted, but its audit record is missing or for another registration")
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
    """A run of the session: its deployment, plan and calibration, and per link-window N2's registration, one-stage
    answer, the client's certificates (egress first) and clock certificate."""
    run = Path(sess["dir"]) / "rehearsal" / strategy
    v = Path(sess["dir"]) / "n2zk" / strategy / "verifier"
    member = json.loads((Path(sess["dir"]) / "n2zk" / "summary.json").read_text())["members"].get(strategy) or {}
    decided = {r["window"]: r for r in member.get("windows") or []}
    certs, clocks = _jsonl(run / "client" / "certificates.jsonl"), _jsonl(run / "client" / "clocks.jsonl")
    windows = {}
    for w, row in sorted(decided.items()):
        reg = json.loads((v / f"w{w:03d}-registration.json").read_text())
        draw = json.loads((v / f"w{w:03d}-draw.json").read_text())
        mine = {c["direction"]: _cert(c) for c in certs if c.get("window") == w and "digest" in c}
        clock = next((c for c in clocks if c.get("window") == w and "digest" in c), None)
        windows[w] = {"registration": reg,
                      "answer": answer_of_outcome(reg, draw, bool(row.get("accepted")),
                                                  f"N2's verifier of record: {row.get('public')}"),
                      "certificates": [mine[d] for d in ("egress", "ingress") if d in mine],
                      "clock": None if clock is None else clock_evidence(clock)}
    return {"run": run, "deployment": json.loads((run / "deployment.json").read_text()),
            "plan": json.loads((run / "plan.json").read_text()),
            "calibration": json.loads((run / "calibration.json").read_text()), "windows": windows}


def clock_evidence(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """A link-window's clock in the period check's evidence: the client's salted clock certificate alone (a line of
    its ``clocks.jsonl`` is the session and the certificate's fields), which signs the sync's clock digest under the
    certifier's salt (`certificate.clock_commitment`); the sync and its salt are witness and never in the case."""
    return _cert(certificate)


def period_windows(plan: Mapping[str, Any], period: int, per: int, periods: int, decided: Sequence[int]) -> list[int]:
    """The link-windows of period ``period`` of ``periods``: those its ``per`` requests are released in (the plan's
    model releases), which ``--window-per-group`` makes one; the first period also takes the link-windows N2 decided
    (``decided``) before its own, and the last those after, so every certified link-window is some period's."""
    T = _window_buckets(plan)
    rel = {int(k): v for k, v in plan["model_release"].items()}
    streams = [q["stream"] for q in plan["requests"][period * per:(period + 1) * per]]
    if len(streams) < per or period >= periods:
        raise RuntimeError(f"the certifier session carries {min(periods, len(plan['requests']) // per)} periods, not "
                           f"period {period}")
    ws = sorted({rel[s] // T for s in streams})
    lo = min([ws[0], *decided]) if period == 0 else ws[0]
    hi = max([ws[-1], *decided]) if period == periods - 1 else ws[-1]
    return list(range(lo, hi + 1))


def _window_buckets(plan: Mapping[str, Any]) -> int:
    sys.path.insert(0, str(TRACES))
    import certifier_serve as cs

    return cs.WINDOW_BUCKETS


def layout(free: Sequence[int], horizon: int) -> dict[str, Any]:
    """The layout terms (module docstring): each audit's free inputs and the link's observation horizon."""
    return {"audits": [{"free": f, "steps": horizon} for f in free], "cells": [], "jitter": 1}


def case_of(rw: Mapping[str, Any], windows: Sequence[int], audits: Sequence[tuple[dict, dict]], first: int,
            free: Sequence[int]) -> dict[str, Any]:
    """``verity/certifier/v0``'s case (`NetworkCheck.Period`'s formats) of the link-windows ``windows`` of a run
    (`run_windows`) and the compute ``audits`` (registration, answer), each with its ``free`` inputs (`free_inputs`);
    the observation horizon is the link-windows' buckets."""
    sys.path.insert(0, str(TRACES))
    import certifier_serve as cs

    if len(free) != len(audits):
        raise ValueError(f"{len(free)} free-input counts for {len(audits)} audits")
    dep, setup = rw["deployment"], cs.ar.Setup(rw["calibration"])
    key = cs.read_public(dep["certifier_key"])
    N, W, H = dep["certifier_key"]["params"]
    xs = [rw["windows"][w] for w in windows]
    program = {x["registration"]["program"] for x in xs}
    if len(program) != 1:
        raise RuntimeError(f"the link-windows register {len(program)} Programs, not the link's one")
    params = {**layout(free, len(xs) * cs.WINDOW_BUCKETS), "key": key.public.hex(), "link": dep["links"]["n2"],
              "lms": {"H": H, "N": N, "W": W, "toy": False}, "program": program.pop(), "session": dep["session"],
              "sync_set": {"deltas": list(setup.sync_set.deltas),
                           "rhos": [[r.numerator, r.denominator] for r in setup.sync_set.rhos]},
              "windows": len(xs), "first": first}
    commitment = canonical_json({"audits": [r for r, _ in audits], "windows": [x["registration"] for x in xs]})
    evidence = canonical_json({"audits": [a for _, a in audits],
                               "windows": [{"answer": x["answer"], "certificates": x["certificates"],
                                            "clock": x["clock"]} for x in xs]})
    return {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}


def check(exe: str, case: Mapping[str, Any]) -> dict[str, Any]:
    """``network-period check`` on the case: its answer."""
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, data in (("params.json", canonical_json(case["params"])),
                           ("commitment.bin", bytes.fromhex(case["commitment"])),
                           ("evidence.json", bytes.fromhex(case["evidence"]))):
            (Path(d) / name).write_bytes(data)
            files.append(str(Path(d) / name))
        p = subprocess.run([exe, "check", *files], capture_output=True, text=True)
    if p.returncode not in (0, 1, 2) or not p.stdout.strip():
        raise RuntimeError(f"network-period check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


# --- the certified outputs -----------------------------------------------------------------------------------------------

def certified_events(run: Path, frame_bytes: int, windows: Sequence[int]) -> list[bytes]:
    """The egress's canonical events in link-windows ``windows`` as the certifier recorded them, each one's bytes
    (`active.event`), in record order, from ``responses.records``: a fixed plan's one canonical stream."""
    sys.path.insert(0, str(TRACES))
    import active_replay as ar
    from verity.core.protocols.certifier.active import DATA, EVENT_HEAD, EVENT_TEXT, decode_frame

    out = []
    for record in sorted((ar.unpack_record(b, frame_bytes) for b in ar.read_blobs(run / "responses.records")),
                         key=lambda r: r.window):
        if record.window not in windows:
            continue
        for row in record.grid:
            for slot in row:
                if slot is None:
                    continue
                kind, _, payload = decode_frame(slot)
                if kind == DATA and payload.startswith(EVENT_HEAD):
                    out.append(base64.b64decode(payload[len(EVENT_HEAD):len(EVENT_HEAD) + EVENT_TEXT].rstrip(b"\0")))
    return out


def _hex(text: bytes) -> bytes | None:
    try:
        return bytes.fromhex(text.decode())
    except ValueError:
        return None


def io(rw: Mapping[str, Any], period: int, windows: Sequence[int], outputs: Mapping[tuple[str, int], bytes],
       steps: int, frames: int) -> dict:
    """The period's certified outputs against its registered ones (`match_outputs`), read off the egress events of
    its link-windows."""
    sys.path.insert(0, str(TRACES))
    import certifier_serve as cs

    setup = cs.ar.Setup(rw["calibration"])
    events = certified_events(rw["run"], setup.egress_params.frame_bytes, windows)
    per = len({r for r, _ in outputs})
    return match_outputs(rw["plan"]["requests"][period * per:(period + 1) * per], events, frames, outputs, steps)


def match_outputs(requests: Sequence[Mapping[str, Any]], events: Sequence[bytes], frames: int,
                  outputs: Mapping[tuple[str, int], bytes], steps: int) -> dict:
    """The period's plan ``requests`` against its registered requests ``r0, r1, …`` in order: the egress carries each
    response as ``frames`` events, back to back in the plan's order, its text the hex of the request's step outputs,
    2H bytes a step. ``by`` is ``position`` when the request's response there is its outputs, ``content`` when another
    response is; ``equal`` when every request's is, and the egress holds exactly the period's responses."""
    reqs = sorted({r for r, _ in outputs}, key=lambda r: int(r.lstrip("r")))
    if len(requests) != len(reqs):
        raise ValueError(f"{len(requests)} plan requests for the period's {len(reqs)} registered requests")
    texts = [_hex(b"".join(events[j * frames:(j + 1) * frames])) for j in range(-(-len(events) // frames))]
    want = {r: b"".join(outputs[r, k] for k in range(steps)) for r in reqs}
    rows = []
    for j, (q, r) in enumerate(zip(requests, reqs)):
        got = texts[j] if j < len(texts) else None
        by = "position" if got == want[r] else "content" if want[r] in texts else None
        rows.append({"request": r, "session": q["session"], "stream": q["stream"], "by": by,
                     "certified_bytes": len(got or b"")})
    return {"equal": all(x["by"] for x in rows) and len(events) == frames * len(reqs), "events": len(events),
            "requests": rows}


# --- the leg -------------------------------------------------------------------------------------------------------------

def _audits(period_dir: Path, out: Path, delta: Fraction) -> list[tuple[dict, dict]]:
    reg = json.loads((period_dir / "registration.json").read_text())
    integrity = json.loads((period_dir / "answers" / "sampled-proofs.json").read_text())
    rec_file = out.parent / "integrity" / "audit" / "window" / "audit.json"
    record = json.loads(rec_file.read_text()) if rec_file.exists() else None
    return [(reg, audit_answer(reg, integrity, record, delta))]


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The period's network certifier check (module docstring) into ``out``; its answer at δ = 1/100."""
    index = json.loads((period_dir / "period.json").read_text())["period"]
    sess = session(period_dir)
    publish(sess, out.parent.parent / SESSION)
    n2 = n2_report(sess)
    summary: dict[str, Any] = {
        "session": {"requests": len(sess["prompts"]), **{k: sess[k] for k in (
            "steps", "h", "periods", "height", "strategies", "tamper", "rehearse_exit")}},
        "n2": n2, "stubbed": list(STUBBED)}
    if why := n2_refusal(n2):
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        raise RuntimeError(why)
    units = loop_units(period_dir)
    outputs = registered_outputs(units)
    free = free_inputs(json.loads((period_dir / "circuit.json").read_text()))
    integrity = json.loads((period_dir / "answers" / "sampled-proofs.json").read_text())["verdict"]
    per, exe = len(sess["prompts"]), sess["exes"]["network-period"]
    summary["free_inputs"] = free
    answer = None
    for strategy in sess["strategies"]:
        rw = run_windows(sess, strategy)
        ws = period_windows(rw["plan"], index, per, sess["periods"], sorted(rw["windows"]))
        missing = [w for w in ws if w not in rw["windows"]]
        if missing:
            raise RuntimeError(f"{strategy}: N2 decided no link-window {missing} of period {index}")
        row = {"windows": ws, "io": io(rw, index, ws, outputs, sess["steps"], sess["frames"])}
        tag = "" if strategy == "honest" else "tamper."
        if strategy == "honest" and not row["io"]["equal"]:
            (out / "io.json").write_text(json.dumps(row["io"], indent=1, sort_keys=True))
            if integrity == "accept":
                raise RuntimeError(f"period {index}: the certified outputs are not the period's registered outputs, "
                                   "though the integrity leg accepted them")
            row["io"]["refused"] = (f"the certified outputs are not the period's registered outputs; the integrity leg "
                                    f"answered {integrity}, so the case decides on its audit")
        for k, delta in enumerate(DELTAS):
            case = case_of(rw, ws, _audits(period_dir, out, delta), ws[0], [free["bits"]])
            a = check(exe, case)
            suffix = "" if k == 0 else f".delta-{delta.denominator}"
            name = "certifier" if strategy == "honest" and k == 0 else f"{tag}certifier{suffix}"
            (out / f"{name}.case.json").write_text(json.dumps(case, sort_keys=True))
            (out / f"{name}.answer.json").write_text(json.dumps(a, sort_keys=True))
            row[str(delta)] = {"verdict": a["verdict"], "reason": a.get("reason"),
                               "K": (a.get("public") or {}).get("K"), "eps": (a.get("public") or {}).get("eps")}
            if strategy == "honest" and k == 0:
                answer = a
            if TAMPERED_WINDOW.get(strategy) in ws and a["verdict"] == "accept":
                (out / "summary.json").write_text(json.dumps(summary | {strategy: row}, indent=1, sort_keys=True))
                raise RuntimeError(f"period {index}: the {strategy} run's case accepts at δ = {delta}, though its "
                                   f"link-window {TAMPERED_WINDOW[strategy]} is tampered")
        summary[strategy] = row
    (out / "io.json").write_text(json.dumps(summary["honest"]["io"], indent=1, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    (out / "runs.json").write_text(json.dumps([]))
    return answer


__all__ = ["DELTAS", "PROTOCOL", "PUBLIC_SESSION", "STUBBED", "TAMPERED_WINDOW", "answer_of_outcome", "audit_answer",
           "case_of", "certified_events", "check", "clock_evidence", "free_inputs", "io", "layout", "leg", "loop_units",
           "match_outputs", "n2_refusal", "n2_report", "period_windows", "publish", "registered_outputs",
           "requests_of", "session"]
