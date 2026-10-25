"""The communication leg: the network certifier over the period (`verity/certifier/v0`, `NetworkCheck.Period`, onsite's
communication component), on a link that carries the torch loop's inputs and outputs (`verity_all_protocols.table`'s
interface: ``leg(period_dir, node, out) -> answer``).

* **The link.** A completions endpoint serves the loop's requests and outputs as read back from the period's window
  (`loop_io`: each request's token id, and each decode step's output, the second linear's 128 bf16 coordinates, 256
  bytes): the prompt is the loop's token ids, and the response streams every request's step outputs in order as hex, one
  server-sent event per 32 bytes, which the certifier's shim cuts as one canonical event each (`active.TOKEN_BYTES`).
  certifier-serve's planned profile sends every session one body and doesn't forward its ``user``, so every session
  carries the whole loop: one session per loop request. The run is `n2_zk.virtual` (an injected clock, fresh LMS keys,
  the certifier signing each link-window's two certified records and each egress window's clock) on the prover's node,
  a rehearsal: the auditor-trusted device is vy-challenger-1's gate (``vy-challenger-gate certifier-rehearse``), whose
  run directory ``ALLP_CERTIFIER_RUN`` names in its place, made against ``python legs/certifier.py serve --window DIR``.
* **Each link-window** is registered on the link's own stream (`service.LinkWindows`, Selection All, its one unit) and
  decided by N2's verifier of record: by default its unit proved by C-Flock ``prove --zk`` behind the Lean firewall and
  held by the verifier's Lean ``verify --zk`` (`n2_zk.py prove`); with ``ALLP_CERTIFIER_PROVE=0``, in the clear
  (`LinkWindows.unproved_outcome`), which reads the opening, so the window's answer is ``undecided``. `window_answer`
  turns the outcome into one-stage's answer for the window's registration.
* **The period** (`case_of`): the link-windows' registrations and the integrity leg's (``answers/sampled-proofs.json``,
  the compute audit) committed together, the certificates and clock certificates from the certifier's log as evidence,
  and `NetworkCheck.Period`'s parameters: the certifier's key and session, the link, the Program, Σ_sync, and per audit
  its free input bits (`free_bits`) and steps. ``network-period check`` (Lean, built in the tree unless
  ``VERITY_NETWORK_PERIOD`` names it) decides, and its answer is the leg's; acceptance publishes ``K``.
* **The month** (`month`): ``NetTiming.Guarantees.HorizonLeakAccepted``'s count (#4367) over a month of periods of
  ``ALLP_PERIOD_S`` seconds (3600 by default, the runner's), each a session of its own: the faulty units paid once,
  ``⌈ln(1/δ)/q⌉ · (16 + log2(N + 1))`` bits, plus the schedules, cells, free inputs and jitter of every period, at
  δ in `DELTAS`. It is conditional on the theorem's two per-window inputs, neither discharged.

Kept in ``out``: ``certifier.case.json`` (the port's case, for onsite), ``certifier.month.json``,
``certifier.detail.json`` (each link-window's decision, the prover's exit and leak scan when proved, what the units
reduce to here), the link's public files under
``link/`` and, when proved, what the verifier received under ``verifier/``. The link's wire records and the prover's
files stay in scratch beside the period's window.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import verity_difftest as VD
from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R

ROOT = Path(__file__).resolve().parents[3]
TRACES = ROOT / "benchmarks" / "network_traces"
LOOP = ROOT / "integrations" / "torch_loop"
NETWORK_LEAN = ROOT / "verity" / "core" / "protocols" / "network" / "lean"
for _p in (TRACES, LOOP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import certifier_serve as cs  # noqa: E402
import n2_zk as NZ  # noqa: E402
from verity.core.protocols.certifier import service as SVC  # noqa: E402
from verity.core.protocols.certifier.certificate import Certificate, Direction  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402

PROTOCOL = "verity/certifier/v0"
INTEGRITY = "sampled-proofs"
MODEL = "torch-loop"
#: bytes of loop output a server-sent event carries: 64 hex characters, one canonical event
CHUNK_BYTES = cs.ar.TOKEN_BYTES // 2
#: the observations' jitter bound: the certifier's shaper runs on a fixed clock (`capacity`'s one jitter class)
JITTER = 1
#: a proof unit's outputs are at most 16 bits
OUT_BITS = 16
DELTAS = ("1/100", "1/1000000")
MONTH_S = 30 * 86400
REDUCES_TO = (
    "The tiny loop's link is certifier-serve's planned profile against an endpoint that answers at once: every session "
    "sends one body and receives every step output of the loop, so each link-window's egress counts are the plan's "
    "occupancy. Each link-window's unit (NetworkLinkWindow_v5) reduces to its two head checks: each end's head opened "
    "from the registered root and its signed digest, the egress head's fields and its counts held to the plan's "
    "occupancy (the ingress end checks no counts). There is no served-count check, no advice and no I/O tie, so the "
    "bytes on the wire are not tied to the outputs the compute audit committed: a prover that sends other bytes at the "
    "plan's pace is accepted, as certifier-serve's `--tamper token` is. vLLM's link adds the served profile's counts "
    "(decode pacing) and, advised, the requests and token ids tied to the plan.")


# --- the loop's inputs and outputs, from its window ----------------------------------------------------------------------

def loop_io(period_dir: Path) -> tuple[list[int], list[list[bytes]]]:
    """The loop's requests (token ids, in request order) and each request's step outputs (each the second linear's bf16
    coordinates, little-endian), read from the window's prover files and its labels."""
    win = json.loads((period_dir / "window.json").read_text())
    circuit = json.loads((period_dir / "circuit.json").read_text())
    if circuit.get("row_tables"):
        raise ValueError("the window has row tables, whose rows this reader doesn't lay out")
    labels = {m["name"]: m["labels"] for m in json.loads((period_dir / "index.json").read_text())["members"]}
    units: dict[tuple[int, int, int], list[tuple[int, np.ndarray]]] = {}
    cols: dict[str, slice] = {}
    for m in win["members"]:
        widths = [w for _, w in m["ports"]]
        at = np.cumsum([0, *widths]).tolist()
        cols = {p: slice(at[i], at[i + 1]) for i, (p, _) in enumerate(m["ports"])}
        raw = (period_dir / f"m{m['k']}-private.bin").read_bytes()
        rows = np.frombuffer(raw[: m["n"] * at[-1] * 2], "<u2").reshape(m["n"], at[-1])
        for lo, hi, lab in labels[m["name"]]:
            key = (int(lab["request"].removeprefix("r")), int(lab["step"]), int(lab["layer"]))
            units.setdefault(key, []).extend((g, rows[g - m["base"]]) for g in range(lo, hi))
    requests = sorted({r for r, _, _ in units})
    steps = sorted({s for _, s, _ in units})
    if requests != list(range(len(requests))) or steps != list(range(len(steps))):
        raise ValueError(f"the window's labels name requests {requests} and steps {steps}")

    def ordered(key) -> list[np.ndarray]:
        return [row for _, row in sorted(units[key], key=lambda u: u[0])]

    outputs = [[np.concatenate([row[cols["o0"]] for row in ordered((r, s, 1))]).astype("<u2").tobytes() for s in steps]
               for r in requests]
    return [token_of(ordered((r, 0, 0))[0][cols["x"]]) for r in requests], outputs


def token_of(x_words: np.ndarray) -> int:
    """The token id whose quantized embedding is the first linear's input row ``x_words`` (its s8 codes as u16 words)."""
    from verity_torch_loop import loop as L

    h = 2 * len(x_words)
    codes, _ = L.quantize(L.Model(h=h).embed)
    got = [t for t in range(codes.shape[0]) if L.bits(codes[t]).tobytes() == x_words.astype("<u2").tobytes()]
    if len(got) != 1:
        raise ValueError(f"the first step's input row is the embedding of {len(got)} tokens, not one")
    return got[0]


def transcript(outputs: Sequence[Sequence[bytes]]) -> list[str]:
    """Every request's step outputs, in order, as the response's event texts: 64 hex characters each."""
    data = b"".join(b"".join(steps) for steps in outputs)
    return [data[i:i + CHUNK_BYTES].hex() for i in range(0, len(data), CHUNK_BYTES)]


def endpoint(tokens: Sequence[int], texts: Sequence[str]):
    """An `asyncio.start_server` handler: an OpenAI-style completions endpoint that serves ``MODEL`` (``/v1/models``),
    tokenizes any prompt as the loop's ``tokens`` (``/tokenize``) and answers every completion at once with ``texts``,
    one server-sent event each, then ``[DONE]``, chunked."""
    events = [b"data: " + json.dumps({"choices": [{"text": t}]}).encode() + b"\n\n" for t in texts] + [b"data: [DONE]\n\n"]

    async def handle(reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            n = re.search(rb"content-length: (\d+)", head, re.I)
            if n:
                await reader.readexactly(int(n.group(1)))
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            writer.close()
            return
        path = head.split(b" ", 2)[1] if head.count(b" ") >= 2 else b""
        if path.endswith(b"/completions"):
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                         + b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in events) + b"0\r\n\r\n")
        else:
            doc = {"data": [{"id": MODEL}]} if path.endswith(b"/models") else {"tokens": list(tokens)}
            body = json.dumps(doc).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                         % (len(body), body))
        await writer.drain()
        writer.close()

    return handle


def link_run(period_dir: Path, scratch: Path) -> Path:
    """The run directory of the link that carried the period's loop: ``ALLP_CERTIFIER_RUN``'s (its ``honest/`` when it has
    one), held to the loop's requests and outputs, or else `n2_zk.virtual`'s honest run in ``scratch``."""
    tokens, outputs = loop_io(period_dir)
    texts = transcript(outputs)
    if named := os.environ.get("ALLP_CERTIFIER_RUN"):
        run = Path(named) / "honest" if (Path(named) / "honest").is_dir() else Path(named)
        body = json.loads((run / "plan.json").read_text()).get("body") or {}
        if body.get("prompt") != tokens or body.get("max_tokens") != len(texts):
            raise ValueError(f"{run}: its requests are not this period's loop (prompt {body.get('prompt')}, "
                             f"{body.get('max_tokens')} events)")
        return run
    shutil.rmtree(scratch / "link", ignore_errors=True)
    NZ.virtual(scratch / "link", requests=len(tokens), tokens=len(texts), strategies=("honest",),
               endpoint=endpoint(tokens, texts), body=cs.request_body(MODEL, len(texts), prompt=list(tokens)))
    return scratch / "link" / "honest"


# --- each link-window's decision --------------------------------------------------------------------------------------

def decided_in_the_clear(run: Path) -> list[tuple[dict, dict]]:
    """Each link-window's registration record and outcome body, decided in the clear on the link's own stream."""
    rd = NZ.link_windows(run)
    st = rd.lw.stream(budget=rd.plan["windows"], delta=cs.DELTA, challenger="all-protocols", coins=cs._no_coins,
                      clock=lambda: 0.0)
    out, previous = [], None
    for x in rd.windows:
        reg = rd.lw.register(st, rd.lw.registration(x.opening, previous))
        previous = reg.digest
        draw = st.select(reg)
        st.deliver(draw, [rd.lw.session(x.opening, draw)]) or rd.lw.unproved_outcome(st, draw, x.opening, rd.syncs)
        out.append((reg.record, st.records[-1]))
    return out


def decided_by_proofs(run: Path, scratch: Path, tools: Mapping[str, str],
                      out: Path) -> tuple[list[tuple[dict, dict]], dict[str, Any]]:
    """Each link-window's registration record and outcome body, its unit proved ``--zk`` behind the firewall and decided
    by the verifier of record (`n2_zk.py prove`, the honest round), and the prover's exit and its scan of what the
    verifier received (`leak_scan`: the pieces leaked, and whether its control found every piece in the prover's own
    files). The scan isn't the verifier's, so it decides nothing here; what the verifier received is kept in ``out``."""
    work = scratch / "n2"
    shutil.rmtree(work, ignore_errors=True)
    path = [tools.get("m0-pythonpath"), str(ROOT / "tools/verity/circuit_check/src"), os.environ.get("PYTHONPATH")]
    env = {**os.environ, "FC_FIREWALL_LEAN": tools["flock-firewall"], "PYTHONPATH": os.pathsep.join(filter(None, path)),
           "VERITY_NETWORK_RECORDS_EXE": lean_exe("network-records")}
    with open(out / "n2_zk.log", "w") as log:
        p = subprocess.run([sys.executable, str(TRACES / "n2_zk.py"), "prove", "--run", str(run), "--rounds", "honest",
                            "--bin", tools["flock-circuit"], "--lean", tools["flock-verify"], "--out", str(work),
                            "--jobs", "1"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    vdir = work / "honest" / "verifier"
    if vdir.is_dir():
        shutil.copytree(vdir, out / "verifier", dirs_exist_ok=True)
    prover: dict[str, Any] = {"exit": p.returncode, "scan": None}
    if (work / "summary.json").exists():
        shutil.copy(work / "summary.json", out / "n2_zk.summary.json")
        member = json.loads((work / "summary.json").read_text())["members"].get("honest") or {}
        received = member.get("received") or {}
        prover["scan"] = {"ok": member.get("ok"), "leaked": received.get("leaked"),
                          "control_holds_all": (received.get("control") or {}).get("holds_all"),
                          "control_missing": (received.get("control") or {}).get("missing")}
    windows = json.loads((run / "plan.json").read_text())["windows"]
    got = []
    for w in range(windows):
        files = [vdir / f"w{w:03d}-{f}.json" for f in ("registration", "outcome")]
        if not all(f.exists() for f in files):
            raise RuntimeError(f"n2_zk.py prove exited {p.returncode} with no outcome for link-window {w} "
                               f"(its log: {out / 'n2_zk.log'})")
        got.append(tuple(json.loads(f.read_text()) for f in files))
    return got, prover


def _refusal(verdict: str, reason: str, commitment: str) -> dict[str, Any]:
    return {"commitment": commitment, "profile": None, "protocol": R.PROTOCOL, "public": None, "reason": reason,
            "verdict": verdict}


def window_answer(registration: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    """One-stage's answer for a link-window's registration from its outcome body: accept only when the verifier of record
    decided it by a proof and its audit accepted; ``undecided`` when it was decided in the clear (which reads the
    opening, never a verdict of record); reject otherwise (a refused proof, or none delivered by the deadline)."""
    commitment = R.digest(registration)
    audit = body.get("audit")
    if audit is None:
        return _refusal("reject", f"outcome {body.get('symbol')}: no audit", commitment)
    decided = ((audit.get("verdicts") or {}).get(SVC.VERIFIER) or {}).get("decided_by")
    if decided != "proof":
        return _refusal("undecided", f"decided by {decided or 'nothing'}, not by a proof", commitment)
    if body.get("symbol") != "accept" or audit.get("accepted") is not True:
        why = sorted(c for c, v in (audit.get("checks") or {}).items() if not (v.get("ok") if isinstance(v, Mapping) else v))
        return _refusal("reject", f"outcome {body.get('symbol')}: " + (", ".join(why) or "not accepted"), commitment)
    reg, draw, wrong = audit["registration"], audit["draw"], audit["wrong_units"]
    profile = A.profile(reg, draw, wrong["delta"]).to_json()
    profile["delta"] = str(Fraction(wrong["delta"]))
    public = {"bound": wrong["bound"], "drawn": sorted(draw["units"]), "per_template": wrong.get("per_template"),
              "proved": sorted(audit["proved_units"])}
    return {"commitment": commitment, "profile": profile, "protocol": R.PROTOCOL, "public": public, "reason": None,
            "verdict": "accept"}


# --- the period's case ------------------------------------------------------------------------------------------------

def free_bits(period_dir: Path) -> int:
    """The audit's free input bits: every unit's input rows, an upper bound on `freeInputs \\ reg` (no wire or registered
    input is subtracted)."""
    c = json.loads((period_dir / "circuit.json").read_text())
    n = dict(c["templates"])
    return sum(n[t] * sum(row[2] for row in st["in_rows"]) for t, st in c["statements"].items())


def certificate_json(c: Certificate) -> dict[str, Any]:
    return {"digest": c.digest.hex(), "direction": c.direction.value, "key_id": c.key_id.hex(), "link": c.link,
            "signature": c.signature.hex(), "window": c.window}


def case_of(run: Path, period_dir: Path, windows: Sequence[tuple[dict, dict]], answers: Sequence[dict]) -> dict:
    """``NetworkCheck.Period``'s case (``{"commitment": hex, "evidence": hex, "params": JSON}``): the link-windows'
    registrations and the compute audit's committed, the certifier's certificates and clock certificates and each
    window's answer as evidence."""
    lw, plan = NZ.link(run)
    dep = json.loads((run / "deployment.json").read_text())
    setup = cs.ar.Setup(json.loads((run / "calibration.json").read_text()))
    e = setup.egress_params
    m = plan["windows"]
    if len(windows) != m:
        raise ValueError(f"{len(windows)} link-windows decided of the plan's {m}")
    certified, _refused = cs.read_certified(run / "certified.jsonl")
    clocks = {c.certificate.window: c for c in cs.read_clocks(run / "clocks.jsonl")}
    try:
        audit_answer = json.loads((period_dir / "answers" / f"{INTEGRITY}.json").read_text())
    except OSError:
        raise RuntimeError("the integrity leg left no answer for the period's compute audit") from None
    lp = lw.key.params
    params = {"audits": [{"free": free_bits(period_dir), "steps": m * e.buckets}], "cells": [m * e.buckets * e.allocation],
              "jitter": JITTER, "key": lw.key.public.hex(), "link": lw.link,
              "lms": {"H": lp.H, "N": lp.N, "W": lp.W, "toy": bool(lp.toy)}, "program": lw.partition["program"],
              "session": dep["session"],
              "sync_set": {"deltas": list(lw.sync_set.deltas),
                           "rhos": [[r.numerator, r.denominator] for r in lw.sync_set.rhos]},
              "windows": m}
    evidence_windows = []
    for w, a in enumerate(answers):
        pair = [certified.get((d, w)) for d in (Direction.EGRESS, Direction.INGRESS)]
        if None in pair or w not in clocks:
            missing = "certified record" if None in pair else "clock certificate"
            raise ValueError(f"link-window {w}: the certifier's log has no {missing}")
        s: ClockSync = clocks[w].sync
        evidence_windows.append({"answer": a, "certificates": [certificate_json(r.certificate) for r in pair],
                                 "clock": {"certificate": certificate_json(clocks[w].certificate),
                                           "sync": [s.delta, s.rho.numerator, s.rho.denominator]}})
    registration = json.loads((period_dir / "registration.json").read_text())
    commitment = {"audits": [registration], "windows": [reg for reg, _ in windows]}
    evidence = {"audits": [audit_answer], "windows": evidence_windows}
    return {"commitment": VD.canonical(commitment).encode().hex(), "evidence": VD.canonical(evidence).encode().hex(),
            "params": params}


def lean_exe(name: str) -> str:
    """The network package's executable ``name``: ``VERITY_NETWORK_PERIOD`` for ``network-period``, else built in the
    tree (``lake build``, under a lock in the package's ``.lake``)."""
    if name == "network-period" and (named := os.environ.get("VERITY_NETWORK_PERIOD")):
        return named
    exe = NETWORK_LEAN / ".lake" / "build" / "bin" / name
    (NETWORK_LEAN / ".lake").mkdir(exist_ok=True)
    with open(NETWORK_LEAN / ".lake" / f"{name}.build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = subprocess.run(["lake", "build", name], cwd=NETWORK_LEAN, capture_output=True, text=True, check=False)
    if p.returncode != 0 or not exe.exists():
        raise RuntimeError(f"lake build {name} exited {p.returncode}: {(p.stdout + p.stderr)[-3000:]}")
    return str(exe)


def period_answer(case: Mapping[str, Any], exe: str, scratch: Path) -> dict[str, Any]:
    """``network-period check`` on the case: its answer line, parsed."""
    files = []
    for name, data in (("params.json", VD.canonical(case["params"]).encode()),
                       ("commitment.bin", bytes.fromhex(case["commitment"])),
                       ("evidence.json", bytes.fromhex(case["evidence"]))):
        (scratch / name).write_bytes(data)
        files.append(str(scratch / name))
    p = subprocess.run([exe, "check", *files], capture_output=True, text=True, check=False)
    if p.returncode not in (0, 1, 2):
        raise RuntimeError(f"network-period check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


# --- the month --------------------------------------------------------------------------------------------------------

def month(params: Mapping[str, Any], audits: Sequence[Mapping[str, Any]], period_s: float,
          month_s: float = MONTH_S) -> dict[str, Any]:
    """``HorizonLeakAccepted``'s count over a month of ``period_s`` periods, each period the same link-windows and compute
    audits (``audits``: their accepted answers) as a session of its own, at each δ of `DELTAS`: log2 of
    ``#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(16·Fmax) · ∏_i 2^free_i · Dj^k_i``, ``Fmax = ⌈ln(1/δ)/q⌉``,
    ``q = min kd_i / n_i``, ``N = Σ_i n_i``. Each term is rounded up."""
    n = max(1, int(month_s // period_s))
    sizes = [(a["profile"]["levels"][0]["size"], a["profile"]["levels"][0]["draw"]["subset"]) for a in audits]
    q = min(Fraction(k, size) for size, k in sizes)
    units = n * sum(size for size, _ in sizes)
    syncs = len(params["sync_set"]["deltas"]) * len(params["sync_set"]["rhos"])
    terms = {"schedules": math.ceil(n * params["windows"] * math.log2(syncs)),
             "cells": n * sum(math.ceil(math.log2(cap + 1)) for cap in params["cells"]),
             "free": n * sum(a["free"] for a in params["audits"]),
             "jitter": math.ceil(n * sum(a["steps"] for a in params["audits"]) * math.log2(params["jitter"]))}
    rows = {}
    for d in DELTAS:
        fmax = math.ceil(math.log(1 / Fraction(d)) / q)
        faulty = math.ceil(fmax * (OUT_BITS + math.log2(units + 1)))
        rows[d] = {"Fmax": fmax, "faulty": faulty, "K": faulty + sum(terms.values())}
    return {"periods": n, "period_s": period_s, "q": str(q), "units": units, "terms": terms, "at": rows,
            "statement": "NetTiming.Guarantees.HorizonLeakAccepted (#4367)",
            "conditional": "on its two per-window inputs, hesc (Flock.Guarantees.SampledKSLive at each window's "
                           "history) and hleak (the live SampledLeakAccepted), neither discharged; with probability at "
                           "most δ + Σ e + Σ e'"}


# --- the leg ----------------------------------------------------------------------------------------------------------

#: the link's files that are public: the grids and calibration, the deployment (the certifier's key, its session, the
#: links) and the clock certificates. The plan holds the requests' body and the certified records the heads and salts.
PUBLIC_LINK = ("calibration.json", "deployment.json", "clocks.jsonl")


def leg(period_dir: Path, node: str, out: Path, check: Callable[[dict], dict] | None = None) -> dict[str, Any]:
    """The period's communication audit (module docstring) into ``out``; ``check``'s answer on the period's case
    (``network-period check`` by default)."""
    p = json.loads((period_dir / "period.json").read_text())
    scratch = period_dir.parent / f"{period_dir.name}-certifier"
    scratch.mkdir(parents=True, exist_ok=True)
    _tokens, outputs = loop_io(period_dir)
    run = link_run(period_dir, scratch)
    (out / "link").mkdir(parents=True, exist_ok=True)
    for f in PUBLIC_LINK:
        if (run / f).exists():
            shutil.copy(run / f, out / "link" / f)
    prove = os.environ.get("ALLP_CERTIFIER_PROVE", "1") != "0"
    windows, prover = decided_by_proofs(run, scratch, p["tools"], out) if prove else (decided_in_the_clear(run), None)
    answers = [window_answer(reg, body) for reg, body in windows]
    case = case_of(run, period_dir, windows, answers)
    (out / "certifier.case.json").write_text(json.dumps(case, sort_keys=True))
    if check is None:
        exe = lean_exe("network-period")

        def check(c: dict) -> dict:
            return period_answer(c, exe, scratch)

    answer = check(case)
    audits = [a for a in json.loads(bytes.fromhex(case["evidence"]))["audits"] if a.get("verdict") == "accept"]
    if audits:
        (out / "certifier.month.json").write_text(json.dumps(
            {**month(case["params"], audits, float(os.environ.get("ALLP_PERIOD_S", "3600"))),
             "period_verdict": answer["verdict"]}, indent=1, sort_keys=True))
    (out / "certifier.detail.json").write_text(json.dumps({
        "certifier": f"gate run {run}" if os.environ.get("ALLP_CERTIFIER_RUN") else
        "n2_zk.virtual on the prover's node (a rehearsal; the auditor-trusted device is vy-challenger-1's gate)",
        "decided": "proofs" if prove else "in the clear", "node": node, "prover": prover,
        "loop": {"requests": len(outputs), "steps": len(outputs[0]), "bytes": sum(len(s) for r in outputs for s in r)},
        "windows": [{"window": w, "registration": R.digest(reg), "verdict": a["verdict"], "reason": a["reason"],
                     "decided_by": ((body.get("audit") or {}).get("verdicts", {}).get(SVC.VERIFIER) or {}).get("decided_by")}
                    for w, ((reg, body), a) in enumerate(zip(windows, answers))],
        "reduces_to": REDUCES_TO}, indent=1, sort_keys=True))
    return answer


def serve(argv: Sequence[str] | None = None) -> int:
    """``serve --window DIR [--host H] [--port P]``: the endpoint (`endpoint`) for the period window ``DIR``, for the
    gate's ``certifier-rehearse`` to put the certifier in front of. Prints its URL, then serves until killed."""
    import argparse

    ap = argparse.ArgumentParser(prog="certifier.py serve")
    ap.add_argument("--window", required=True, type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args(argv)
    tokens, outputs = loop_io(a.window)

    async def go():
        server = await asyncio.start_server(endpoint(tokens, transcript(outputs)), a.host, a.port)
        print(json.dumps({"url": "http://%s:%d" % server.sockets[0].getsockname()[:2], "model": MODEL,
                          "tokens": len(transcript(outputs))}), flush=True)
        async with server:
            await server.serve_forever()

    asyncio.run(go())
    return 0


__all__ = ["DELTAS", "PROTOCOL", "REDUCES_TO", "case_of", "decided_by_proofs", "decided_in_the_clear", "endpoint",
           "free_bits", "leg", "link_run", "loop_io", "month", "period_answer", "transcript", "window_answer"]

if __name__ == "__main__":
    if sys.argv[1:2] != ["serve"]:
        raise SystemExit("usage: certifier.py serve --window DIR [--host H] [--port P]")
    raise SystemExit(serve(sys.argv[2:]))
