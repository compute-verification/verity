"""The communication leg: the network certifier over the node's link, decided per period by `NetworkCheck.Period`
(``verity/certifier/v0``, the ``network-period`` executable) on a case built from one certifier session that spans the
run's periods (`verity_all_protocols.table`'s interface: ``leg(period_dir, node, out) -> answer``).

**The session** (`session`, once a run, under the runner's ``--out`` as ``certifier-session/``). At the first period the
leg reads the period's requests off its value files (`requests_of`: each request's prompt token is the one whose
embedding the loop's first linear took, its steps the loop's) and serves `CERTIFIER_PERIODS` periods of them through the
network certifier: `certifier_serve.py rehearse` on the fixed traffic and N2's planned profile, each period's requests one
group in a link-window of its own (``--prompts``, ``--window-per-group``), so period ``i`` is link-window ``i``, and the
loop's model behind it (`torch_loop_stub.py`, the loop's W8A8 MLP, each step's 256 output bytes as hex text). Beside
the honest run, the same plan with the shim's `CERTIFIER_TAMPER` (``extra`` by default: one event more after window
0's first response). The client keeps and verifies each window's certificates and clock certificates. Then N2's verdict
of record on every link-window of both runs, each proved ``--zk`` behind the Lean firewall (`n2_zk.py prove`). The
session is one LMS key (height `CERTIFIER_HEIGHT`) and one certifier session per run, its leaf state kept across both.

**A period** (`leg`): the certified outputs of its link-window (the egress records' canonical events, by stream) must be
the period's registered outputs, the value files' last-linear outputs, request by request (`io`, the leg's own check,
written to ``io.json`` beside the answer: a refusal there is the leg's error). The case (`case_of`) is the period's
link-windows, each with its N2 registration, its one-stage answer from N2's outcome of record (Selection All: its one
unit, `answer_of_outcome`), its two certificates and its clock certificate as the client kept them; and the period's
compute audit, the integrity leg's registration and audit record re-decided by one-stage's own bound at δ
(`audit_answer`). Its answer is ``network-period check``'s at δ = 1/100 (`DELTAS`), and ``out/`` also holds the case
and answer at δ = 10⁻⁶ and, for the tampered run, its case and answer (``tamper.*``), which must reject.

Layout terms (`layout`): no free input (the loop's every input is registered), the audit's horizon the loop's steps,
jitter bound 1 (N2 holds the planned profile's egress counts exactly, so no observation moves), and no warden
connection but the link, whose counts the profile fixes (``cells`` empty).
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
#: a canonical event's most bytes (`active.TOKEN_BYTES`)
TOKEN_BYTES = 64


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


def session(root: Path, period_dir: Path) -> dict[str, Any]:
    """The run's one certifier session (module docstring), made at its first period; its record ``session.json``."""
    s = root / SESSION
    if (s / "session.json").exists():
        return json.loads((s / "session.json").read_text())
    s.mkdir(parents=True, exist_ok=True)
    tools, exes = _tools(period_dir), lean_exes()
    prompts, steps, h = requests_of(period_dir)
    periods, height = _env_int("CERTIFIER_PERIODS", 1), _env_int("CERTIFIER_HEIGHT", 10)
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
           "frames": frames, "rehearse_exit": rc, "n2_exit": n2_rc, "exes": exes, "dir": str(s)}
    (s / "session.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


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
        clock = next((c for c in clocks if c["certificate"]["window"] == w), None)
        windows[w] = {"registration": reg,
                      "answer": answer_of_outcome(reg, draw, bool(row.get("accepted")),
                                                  f"N2's verifier of record: {row.get('public')}"),
                      "certificates": [mine[d] for d in ("egress", "ingress") if d in mine],
                      "clock": None if clock is None else {
                          "certificate": _cert(clock["certificate"]),
                          "sync": [clock["sync"][0], *_ratio(clock["sync"][1])]}}
    return {"run": run, "deployment": json.loads((run / "deployment.json").read_text()),
            "plan": json.loads((run / "plan.json").read_text()),
            "calibration": json.loads((run / "calibration.json").read_text()), "windows": windows}


def _ratio(s: str) -> list[int]:
    f = Fraction(s)
    return [f.numerator, f.denominator]


def period_windows(plan: Mapping[str, Any], period: int, per: int) -> list[int]:
    """The link-windows period ``period``'s ``per`` requests are released in (the plan's model releases), which
    ``--window-per-group`` makes one."""
    T = _window_buckets(plan)
    rel = {int(k): v for k, v in plan["model_release"].items()}
    streams = [q["stream"] for q in plan["requests"][period * per:(period + 1) * per]]
    if len(streams) < per:
        raise RuntimeError(f"the certifier session carries {len(plan['requests']) // per} periods, not period {period}"
                           " (CERTIFIER_PERIODS)")
    ws = sorted({rel[s] // T for s in streams})
    return list(range(ws[0], ws[-1] + 1))


def _window_buckets(plan: Mapping[str, Any]) -> int:
    sys.path.insert(0, str(TRACES))
    import certifier_serve as cs

    return cs.WINDOW_BUCKETS


def layout(steps: int, audits: int) -> dict[str, Any]:
    """The layout terms (module docstring)."""
    return {"audits": [{"free": 0, "steps": steps} for _ in range(audits)], "cells": [], "jitter": 1}


def case_of(rw: Mapping[str, Any], windows: Sequence[int], audits: Sequence[tuple[dict, dict]], steps: int,
            first: int) -> dict[str, Any]:
    """``verity/certifier/v0``'s case (`NetworkCheck.Period`'s formats) of the link-windows ``windows`` of a run
    (`run_windows`) and the compute ``audits`` (registration, answer)."""
    sys.path.insert(0, str(TRACES))
    import certifier_serve as cs

    dep, setup = rw["deployment"], cs.ar.Setup(rw["calibration"])
    key = cs.read_public(dep["certifier_key"])
    N, W, H = dep["certifier_key"]["params"]
    xs = [rw["windows"][w] for w in windows]
    program = {x["registration"]["program"] for x in xs}
    if len(program) != 1:
        raise RuntimeError(f"the link-windows register {len(program)} Programs, not the link's one")
    params = {**layout(steps, len(audits)), "key": key.public.hex(), "link": dep["links"]["n2"],
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

def certified_outputs(run: Path, frame_bytes: int, windows: Sequence[int]) -> dict[int, bytes]:
    """Each egress stream's text in link-windows ``windows`` as the certifier recorded it: its canonical events' bytes in
    record order (`active.event`), from ``responses.records``."""
    sys.path.insert(0, str(TRACES))
    import active_replay as ar
    from verity.core.protocols.certifier.active import DATA, EVENT_HEAD, EVENT_TEXT, decode_frame

    text: dict[int, bytearray] = {}
    for record in sorted((ar.unpack_record(b, frame_bytes) for b in ar.read_blobs(run / "responses.records")),
                         key=lambda r: r.window):
        if record.window not in windows:
            continue
        for row in record.grid:
            for slot in row:
                if slot is None:
                    continue
                kind, stream, payload = decode_frame(slot)
                if kind != DATA or not payload.startswith(EVENT_HEAD):
                    continue
                b64 = payload[len(EVENT_HEAD):len(EVENT_HEAD) + EVENT_TEXT].rstrip(b"\0")
                text.setdefault(stream, bytearray()).extend(base64.b64decode(b64))
    return {s: bytes(t) for s, t in text.items()}


def _hex(text: bytes) -> bytes | None:
    try:
        return bytes.fromhex(text.decode())
    except ValueError:
        return None


def io(rw: Mapping[str, Any], period: int, windows: Sequence[int], outputs: Mapping[tuple[str, int], bytes],
       steps: int, h: int) -> dict:
    """The period's certified outputs against its registered ones, request by request: request ``r`` of the period is
    the plan's request ``period · n + r``, whose egress stream in the period's link-windows carries its steps'
    outputs as hex, 2H bytes a step (``by: stream``); else, for each request, a stream there that carries them
    (``by: content``)."""
    sys.path.insert(0, str(TRACES))
    import certifier_serve as cs

    setup = cs.ar.Setup(rw["calibration"])
    got = {s: _hex(t) for s, t in certified_outputs(rw["run"], setup.egress_params.frame_bytes, windows).items()}
    reqs = sorted({r for r, _ in outputs}, key=lambda r: int(r.lstrip("r")))
    want = {r: b"".join(outputs[r, k] for k in range(steps)) for r in reqs}
    rows = []
    for i, r in enumerate(reqs):
        q = rw["plan"]["requests"][period * len(reqs) + i]
        by = "stream" if got.get(q["stream"]) == want[r] else \
            "content" if want[r] in got.values() else None
        rows.append({"request": r, "session": q["session"], "stream": q["stream"], "by": by,
                     "certified_bytes": len(got.get(q["stream"]) or b"")})
    return {"equal": all(x["by"] for x in rows), "streams": {str(s): None if t is None else len(t)
                                                            for s, t in sorted(got.items())},
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
    sess = session(out.parent.parent, period_dir)
    units = loop_units(period_dir)
    outputs = registered_outputs(units)
    per, exe = len(sess["prompts"]), sess["exes"]["network-period"]
    summary: dict[str, Any] = {"session": {k: sess[k] for k in ("prompts", "steps", "h", "periods", "height",
                                                                 "strategies", "rehearse_exit", "n2_exit")}}
    answer = None
    for strategy in sess["strategies"]:
        rw = run_windows(sess, strategy)
        ws = period_windows(rw["plan"], index, per)
        missing = [w for w in ws if w not in rw["windows"]]
        if missing:
            raise RuntimeError(f"{strategy}: N2 decided no link-window {missing} of period {index}")
        row = {"windows": ws, "io": io(rw, index, ws, outputs, sess["steps"], sess["h"])}
        tag = "" if strategy == "honest" else "tamper."
        if strategy == "honest" and not row["io"]["equal"]:
            (out / "io.json").write_text(json.dumps(row["io"], indent=1, sort_keys=True))
            raise RuntimeError(f"period {index}: the certified outputs are not the period's registered outputs")
        for k, delta in enumerate(DELTAS):
            case = case_of(rw, ws, _audits(period_dir, out, delta), sess["steps"], ws[0])
            a = check(exe, case)
            suffix = "" if k == 0 else f".delta-{delta.denominator}"
            name = "certifier" if strategy == "honest" and k == 0 else f"{tag}certifier{suffix}"
            (out / f"{name}.case.json").write_text(json.dumps(case, sort_keys=True))
            (out / f"{name}.answer.json").write_text(json.dumps(a, sort_keys=True))
            row[str(delta)] = {"verdict": a["verdict"], "reason": a.get("reason"),
                               "K": (a.get("public") or {}).get("K"), "eps": (a.get("public") or {}).get("eps")}
            if strategy == "honest" and k == 0:
                answer = a
        summary[strategy] = row
    (out / "io.json").write_text(json.dumps(summary["honest"]["io"], indent=1, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    (out / "runs.json").write_text(json.dumps([]))
    return answer


__all__ = ["DELTAS", "PROTOCOL", "answer_of_outcome", "audit_answer", "case_of", "certified_outputs", "check", "io",
           "layout", "leg", "loop_units", "period_windows", "registered_outputs", "requests_of", "session"]
