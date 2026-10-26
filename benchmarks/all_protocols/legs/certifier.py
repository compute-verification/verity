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
each proved ``--zk`` behind the Lean firewall one session at a time (`n2_zk.py prove --jobs 1`,
`n2_zk.ONE_SESSION`): its exit and leak scan are in ``summary.json``
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
re-decided by one-stage's own bound at its term's δ (`audit_answer`). Its answer is ``network-period check``'s at
δ = 1/100 (`DELTAS`), and ``out/`` also holds the case and answer at δ = 10⁻⁶ and the tampered run's (``tamper.*``).
At each δ every term of the case is decided at δ over the case's term count (`term_delta`: the answer's ``eps`` is the
sum of its terms' δ), so an accepted answer's ``eps`` is δ itself, which the leg asserts (`holds_at`). The period
holding the tampered link-window (`TAMPERED_WINDOW`) must reject there, which the leg asserts; the tampered run's other
periods may accept.

**The warden path** (``CERTIFIER_PATH=warden``, `warden_leg`; the default ``certifier`` is the path above, whose
verdict the table keeps until this one decides every period): the records are the network certifier's own cells as one
device.
`warden_session` serves the same prompts through `warden_serve.py rehearse --ids --prompts --window-per-group` at the
scan's schedule (`warden_argv`, the gate's scan rehearsal), the stub answering id-form, a group of a period's requests
at a time, once honest and once for each of the stub's tampers (`WARDEN_TAMPER`, `WARDEN_TAMPERS`: its first id request
answered ``late``, ``byte`` or ``extra``, a request of period `WARDEN_TAMPERED_PERIOD`'s). A period's link-windows are
its group's in the run's ``groups.json`` (`warden_period_windows`), each with its two row registrations and its three
records' certificates (`warden_windows`). The certified ids are checked against the period's registered outputs in the
clear (`warden_io`), and the scan's chain over link-window 0 in the clear (`scan_dry`). Honest must pass both; the
tampered period must be refused, ``late`` by the scan (the network certifier failed its connection), ``byte`` and
``extra`` by the output tie alone (the scan reads its outputs off the egress). Only the gate's public files reach the
runner's ``--out/warden-session/`` (`warden_publish`, `WARDEN_PUBLIC`, the certificates without what they sign). The
salts, the records, the client's log, the traffic and the prompts stay in the scratch. Each run the dry scan accepts
is then proved on link-window 0 (`scan_prove`: `scan_zk.py prove --warden-run --draw` `SCAN_DRAW`, its units ``--zk``
behind the Lean firewall; a round not as expected or witness in what its verifier received refuses the session), and
its audit record is one-stage's answer for its registration, re-decided at its term's δ (`scan_window`: the rows form's
terms are the audits and each link-window's scan, `term_delta`). A period whose
every link-window has a scan answer is decided in certifier/v0's rows form (`rows_case_of`, ``network-period check``):
per link-window the scan's registration in the commitment (its ``warden/<side>`` roots the rows' registrations' roots
and domains, `scan_zk.warden_roots`) and, in the evidence, its answer, the three records' certificates and the two rows'
registrations, under the deployment's key and its one schedule; the scan's layout terms
its free inputs (`scan_free`, every scan row but the network certifier's own cells, an overcount) over a link-window's
buckets. A period with a link-window the scan didn't answer raises (`WARDEN_PENDING`: link-window 0 alone is proved
yet, and a ``late`` run's is refused before the scan registers it).

Layout terms (`layout`): the audit's free inputs (`free_inputs`, from the window's ``circuit.json``: every unit's
input-row bits, less the rows wires feed or ``external`` names), the observation horizon the link's (its link-windows'
buckets, `certifier_serve.WINDOW_BUCKETS` each), jitter bound 1 (N2 holds the planned profile's egress counts exactly,
so no observation moves), and no connection through the network certifier but the link, whose counts the profile fixes
(``cells`` empty).

**The block record** (`block_leg`): a period's accepted answer, of either path, as the certifier leg of
``verity/block-record/v1`` (CMP-95 item 6): K in bits, its error the answer's exact ``eps``, and the certificate at
δ = 1/100 and 10⁻⁶ wherever ``eps`` is at most δ. A loop's leg and the H200 fallback's write that one shape.
"""
from __future__ import annotations

import base64
import fcntl
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import audit as A, registration as R
from verity.ml.circuits.codec import canonical_json
from verity.ml.windows.rows import BC_BYTES, SALT_BYTES

PROTOCOL = "verity/certifier/v0"
ONE_STAGE = R.PROTOCOL
ROOT = Path(__file__).resolve().parents[3]
TRACES = ROOT / "benchmarks" / "network_traces"
LEAN = ROOT / "verity" / "core" / "protocols" / "network" / "lean"
SESSION = "certifier-session"
#: the δ a period's certificate holds at: its answer's, then the one written beside it, each split across the case's
#: terms (`term_delta`)
DELTAS = (Fraction(1, 100), Fraction(1, 10 ** 6))
#: N2's stream decides at `certifier_serve.DELTA`
N2_DELTA = Fraction(2) ** -40
#: the leg's records, by ``CERTIFIER_PATH``: ``certifier`` (the default), `certifier_serve.py`'s certifier-only path
#: (`session`, `certifier_leg`); ``warden``, the network certifier's own cells as one device (`warden_session`,
#: `warden_leg`), which `verity/certifier/v0` decides once it has the device's form
PATHS = ("certifier", "warden")
_PROXY_GAP = {
    "gap": "the records are certifier_serve.py's certifier-only path (active.Proxy), not the network certifier's own "
           "cells as one device",
    "fix": "CERTIFIER_PATH=warden: the device's cells.jsonl and certified records (warden_serve.py rehearse --ids "
           "--prompts --window-per-group), decided by certifier/v0's form for them"}
_OFF_HOST_GAP = {
    "gap": "the device runs off-host, beside the server on the node, not on the auditor's machine (vy-challenger-1) "
           "through the gate",
    "fix": "the gate PR: the leg serves the stub on a peer and runs warden-rehearse --scan --prompts "
           "--window-per-group through the gate with the lane's gate key"}
_TIE_GAP = {
    "gap": "the output tie is computed in the clear, not proved: the leg's own check of the certified outputs against "
           "the registered ones (io, warden_io)",
    "fix": "NET-68's advised tie: the certified ids proved to be the registered outputs' (io_tie) inside the period's "
           "proofs"}
_SCAN_GAP = {
    "gap": "the scan's layout terms are the leg's overcount (its free inputs every scan row but the network certifier's "
           "own cells, `scan_free`), and its draw the verifier's own Lean coins on link-window 0 alone",
    "fix": "LEAK-LEAN's exact K terms against the Lean, and IO-TIE's scan_zk --window W on the challenger's coins "
           "(#4387)"}
#: what each path's verdict still stands on that isn't the network certifier's own, each with its fix (``stubbed`` in
#: ``summary.json``)
STUBBED = {"certifier": (_PROXY_GAP, _OFF_HOST_GAP, _TIE_GAP), "warden": (_OFF_HOST_GAP, _TIE_GAP, _SCAN_GAP)}
#: a canonical event's most bytes (`active.TOKEN_BYTES`)
TOKEN_BYTES = 64
#: what `publish` copies of each run of the session into the runner's ``--out``: nothing the prover holds as witness
PUBLIC_SESSION = ("deployment.json", "calibration.json", "client/certificates.jsonl", "client/clocks.jsonl")
#: the link-window each shim tamper changes (`certifier_serve.py`'s ``late``, ``extra``, ``drop``), whose period's case
#: must reject; ``token`` changes bytes only the I/O tie reads, which the network check accepts
TAMPERED_WINDOW = {"late": 0, "extra": 0, "drop": 0}
WARDEN_SESSION = "warden-session"
#: what `warden_publish` copies of each warden run, beside its certificates' public parts: the gate's
#: (`auditor.WARDEN_PUBLIC`, `auditor.warden_outcome`)
WARDEN_PUBLIC = ("deployment.json", "cells.jsonl", "groups.json")
#: the warden session's tampered runs (``WARDEN_TAMPER``, of `torch_loop_stub.TAMPERS`): the stub's answer to its first
#: id request, a request of group 0 and so of `WARDEN_TAMPERED_PERIOD`. ``late`` fails the connection at the warden, so
#: the scan refuses its link-window; ``byte`` and ``extra`` keep the egress one the scan accepts (its ``prog`` is read
#: off the egress), so only the output tie refuses them
WARDEN_TAMPERS = ("late", "byte")
WARDEN_TAMPERED_PERIOD = 0
#: a warden rehearsal's traffic flags as the gate's (`auditor.WARDEN_RATE`, ``WARDEN_SEED``, ``WARDEN_TAIL_S``), at the
#: greedy sampling the stub's id requests need (`torch_loop_stub.id_refusal`)
WARDEN_TRAFFIC = ("--rate", "0.5", "--seed", "0", "--tail", "60", "--temperature", "0", "--top-p", "1")
#: why a warden period past link-window 0 isn't decided yet
WARDEN_PENDING = "scan_zk.py proves link-window 0 alone (no --window W), so no later link-window has a scan answer"
#: the scan's sampled draw (``SCAN_DRAW``): Lean ``draw --stratified K`` of the window's 2T units
SCAN_DRAW = 16
#: what `scan_prove` keeps of a scan's ``--out``: its summary and its honest round's registration, draw and audit record
SCAN_KEEP = ("summary.json", "honest/registration.json", "honest/draw.json", "honest/audit.json")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


# --- the period's records ----------------------------------------------------------------------------------------------

def member_rows(period_dir: Path, m: Mapping[str, Any]) -> np.ndarray:
    """Every unit's rows of member ``m`` (its ``window.json`` entry), ports in order, u16 words (units x words), off its
    value files (`rows.Committed`'s bodies): ``m<k>-private.bin`` is every instance's rows then the salts, or for a
    shared-row member (``shared_rows``, each port's table size: a registered read's table holds the registered rows its
    units read) each port's table then the salts, each unit's rows its refs into the tables, the u32 per instance and
    port that end ``m<k>-public.bin``."""
    widths, n = [w for _, w in m["ports"]], m["n"]
    priv = (period_dir / f"m{m['k']}-private.bin").read_bytes()
    if "shared_rows" not in m:
        if len(priv) != n * (2 * sum(widths) + SALT_BYTES * len(widths)):
            raise ValueError(f"m{m['k']}-private.bin: {len(priv)} bytes, not {n} instances' rows {widths} and salts")
        return np.frombuffer(priv[:2 * n * sum(widths)], "<u2").reshape(n, sum(widths))
    sizes = [int(m["shared_rows"][p]) for p, _ in m["ports"]]
    pub = (period_dir / f"m{m['k']}-public.bin").read_bytes()
    if len(priv) != sum(s * (2 * w + SALT_BYTES) for s, w in zip(sizes, widths)) or \
            len(pub) != BC_BYTES * sum(sizes) + 4 * n * len(widths):
        raise ValueError(f"m{m['k']}: {len(priv)} private and {len(pub)} public bytes, not a shared-row member's of "
                         f"tables {dict(zip((p for p, _ in m['ports']), sizes))}, rows {widths} and {n} instances")
    refs = np.frombuffer(pub[BC_BYTES * sum(sizes):], "<u4").reshape(n, len(widths))
    if (refs >= np.array(sizes)).any():
        raise ValueError(f"m{m['k']}: a ref past its port's table")
    tables = np.split(np.frombuffer(priv, np.uint8), np.cumsum([2 * s * w for s, w in zip(sizes, widths)]))
    return np.concatenate([t.view("<u2").reshape(s, w)[refs[:, j]]
                           for j, (t, s, w) in enumerate(zip(tables, sizes, widths))], axis=1)


def loop_units(period_dir: Path) -> dict[int, tuple[dict, dict[str, np.ndarray]]]:
    """Every unit of the period's window: its label (`index.json`: request, step, layer) and its rows by port, u16 words,
    read off the member's value files (`member_rows`)."""
    win = json.loads((period_dir / "window.json").read_text())
    labels = {m["name"]: m["labels"] for m in json.loads((period_dir / "index.json").read_text())["members"]}
    out: dict[int, tuple[dict, dict[str, np.ndarray]]] = {}
    for m in win["members"]:
        names, widths = [p for p, _ in m["ports"]], [w for _, w in m["ports"]]
        rows = member_rows(period_dir, m)
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
    each unit's rows whose every port a wire feeds from another unit's output (``wires``), a registered read fills from
    the value the verifier pinned before the run (``registered``'s uncut ``reads``: their reader units' port) or
    ``external`` names as the system's I/O. ``bits`` the total, ``per_template`` each template's. A row the
    registration's ``anchors`` register, or a registered table's root, an opening or an ingress fills, still counts: none
    is committed yet, and counting a registered row free only overcounts K."""
    from verity.ml.windows.interface import Wire

    units = {t: n for t, n in circuit["templates"]}
    covered: dict[tuple[str, str], set[int] | None] = {}
    for e in circuit["external"]:
        t, port = e.rsplit("/", 1)
        if t not in units:
            raise ValueError(f"external port {e!r} is no template's")
        covered[t, port] = None
    fills = [(w.reader, w.port, w.at, w.units, "a wire") for w in map(Wire.of, circuit["wires"])]
    fills += [(r["reader"], r["port"], r["at"], math.prod(r["radix"]), "a registered read")
              for r in (circuit.get("registered") or {}).get("reads", ()) if r.get("cut", [0, 1]) == [0, 1]]
    for reader, port, at, n, what in fills:
        if reader not in units:
            raise ValueError(f"{what}'s reader {reader!r} is no template")
        fed = covered.setdefault((reader, port), set())
        if fed is not None:
            fed.update(range(at, at + n))
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


def _stub(steps: int, h: int, log: Path, tamper: str | None = None) -> tuple[subprocess.Popen, str]:
    tampered = ["--tamper", tamper, "--tamper-at", "0"] if tamper else []
    proc = subprocess.Popen([sys.executable, str(TRACES / "torch_loop_stub.py"), "--steps", str(steps), "--h", str(h),
                             *tampered], stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, cwd=ROOT)
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


def _periods(period_dir: Path) -> int:
    period = json.loads((period_dir / "period.json").read_text())
    if not isinstance(period.get("periods"), int) or period["periods"] < 1:
        raise RuntimeError("period.json names no period count (`periods`), which the session's plan serves")
    return period["periods"]


def once(s: Path, make: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
    """The run's one session in ``s``: its record ``s/session.json``, made by ``make(s)`` while holding ``s``'s lock, so
    that of the windows the runner audits at once (``--jobs``) one makes it and the others read it."""
    s.mkdir(parents=True, exist_ok=True)
    with open(s.parent / f"{s.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (s / "session.json").exists():
            return json.loads((s / "session.json").read_text())
        return make(s)


def session(period_dir: Path) -> dict[str, Any]:
    """The run's one certifier session (module docstring), made at its first period in the runner's scratch beside the
    period's window (``period_dir.parent``); its record ``session.json`` (`once`)."""
    return once(period_dir.parent / SESSION, lambda s: _session(period_dir, s))


def _session(period_dir: Path, s: Path) -> dict[str, Any]:
    tools, exes = _tools(period_dir), lean_exes()
    prompts, steps, h = requests_of(period_dir)
    periods, height = _periods(period_dir), _env_int("CERTIFIER_HEIGHT", 10)
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
              "--out", str(scratch / "n2zk"), "--jobs", "1"]
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


# --- the warden session --------------------------------------------------------------------------------------------------

def _traces_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(TRACES), os.environ.get("PYTHONPATH")]))}


def scan_schedule(env: Mapping[str, str] | None = None) -> list[str]:
    """The scan's schedule as `warden_serve.py`'s flags (`scan_zk.py schedule`, the gate's ``SCAN_SCHEDULE``), the only
    one `scan_zk.py prove --warden-run` reads."""
    p = subprocess.run([sys.executable, str(TRACES / "scan_zk.py"), "schedule"], cwd=ROOT, env=dict(env or os.environ),
                       capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.split():
        raise RuntimeError(f"scan_zk.py schedule exited {p.returncode}: {p.stderr[-2000:]}")
    return p.stdout.split()


def warden_argv(upstream: str, out: Path, prompts: Path, requests: int, steps: int, height: int,
                schedule: Sequence[str]) -> list[str]:
    """`warden_serve.py rehearse` as the gate runs a scan rehearsal of prompts in groups (`auditor.warden_command` of a
    request with ``scan``, ``prompts``, ``window_per_group`` and ``tokens``): the network certifier and its client
    here, its upstream the stub at ``upstream``, each of the ``requests`` exactly ``steps`` tokens."""
    return [sys.executable, str(TRACES / "warden_serve.py"), "rehearse", "--upstream", upstream.removeprefix("http://"),
            "--out", str(out), "--height", str(height), "--requests", str(requests), *WARDEN_TRAFFIC,
            "--ids", *schedule, "--prompts", str(prompts), "--window-per-group", "--max-tokens", str(steps)]


def scan_dry(run: Path, log: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The scan's chain over a warden run's link-window 0 in the clear (`scan_zk.py prove --dry --warden-run`), nothing
    proved: ``accepted``, else its refusal (the last line it wrote)."""
    with tempfile.TemporaryDirectory(prefix="certifier-leg-scan-") as d:
        rc = _run([sys.executable, str(TRACES / "scan_zk.py"), "prove", "--dry", "--warden-run", str(run),
                   "--out", d], log, env)
    lines = [x for x in log.read_text().splitlines() if x.strip()]
    return {"window": 0, "exit": rc, "accepted": rc == 0, "reason": None if rc == 0 else (lines or [""])[-1][-500:]}


def warden_rehearse(s: Path, prompts: Sequence[int], steps: int, h: int, periods: int, height: int,
                    strategies: Sequence[str], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Each strategy's warden run of the prompts into ``s/rehearsal/<strategy>``, a group of them a period, each
    behind its own stub (``honest``, else the stub's ``--tamper``), then the scan's dry chain over its link-window 0
    (`scan_dry`); the prompts file (``s/prompts.json``) and every run's records stay in ``s``."""
    env = dict(env or _traces_env())
    (s / "prompts.json").write_text(json.dumps([[t] for t in prompts]))
    schedule = scan_schedule(env)
    exits, accepted, scans = {}, {}, {}
    for strategy in strategies:
        run = s / "rehearsal" / strategy
        stub, url = _stub(steps, h, s / f"stub-{strategy}.log", None if strategy == "honest" else strategy)
        try:
            exits[strategy] = _run(warden_argv(url, run, s / "prompts.json", len(prompts) * periods, steps, height,
                                               schedule), s / f"rehearse-{strategy}.log", env)
        finally:
            stub.terminate()
            stub.wait()
        if not (run / "check.json").exists():
            raise RuntimeError(f"warden_serve.py rehearse ({strategy}) exited {exits[strategy]} with no check.json "
                               f"({s / f'rehearse-{strategy}.log'})")
        accepted[strategy] = json.loads((run / "check.json").read_text()).get("accepted")
        scans[strategy] = scan_dry(run, s / f"scan-{strategy}.log", env)
    return {"prompts": list(prompts), "steps": steps, "h": h, "periods": periods, "height": height,
            "strategies": list(strategies), "schedule": schedule, "rehearse_exit": exits, "warden_accepted": accepted,
            "scan": scans, "dir": str(s)}


def _free_ports(n: int) -> int:
    """A base port with the ``n`` ports from it free now (`scan_zk.py prove --port`: its sessions' ports)."""
    import socket

    for _ in range(64):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        if base + n >= 65536:
            continue
        try:
            for p in range(base, base + n):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", p))
            return base
        except OSError:
            continue
    raise RuntimeError(f"no {n} free ports in a row")


def scan_prove(run: Path, s: Path, strategy: str, tools: Mapping[str, str], draw: int) -> dict[str, Any]:
    """The scan's proof of a warden run's link-window 0 (`scan_zk.py prove --warden-run --draw`, its units ``--zk``
    behind the Lean firewall, the draw the verifier's Lean's): its exit, and `SCAN_KEEP` of its ``--out`` in
    ``s/scan/<strategy>`` (the rest, provers' files and sessions, in a scratch removed after)."""
    env = {**os.environ, "FC_FIREWALL_LEAN": tools["flock-firewall"],
           "PYTHONPATH": os.pathsep.join(filter(None, [tools["m0-pythonpath"], str(TRACES),
                                                       str(ROOT / "tools" / "verity" / "circuit_check" / "src"),
                                                       os.environ.get("PYTHONPATH")]))}
    keep = s / "scan" / strategy
    keep.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="certifier-leg-scan-"))
    try:
        rc = _run([sys.executable, str(TRACES / "scan_zk.py"), "prove", "--warden-run", str(run), "--draw", str(draw),
                   "--bin", tools["flock-circuit"], "--lean", tools["flock-verify"], "--out", str(scratch),
                   "--port", str(_free_ports(16))], s / f"scan-prove-{strategy}.log", env)
        for f in SCAN_KEEP:
            if (scratch / f).is_file():
                (keep / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(scratch / f, keep / f)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    summary = json.loads((keep / "summary.json").read_text()) if (keep / "summary.json").exists() else {}
    return {"exit": rc, "ok": summary.get("ok"), "dir": str(keep), "draw": draw}


def scan_free(summary: Mapping[str, Any]) -> int:
    """The scan's free inputs a link-window, the leg's overcount: each of its T buckets' ingress and egress units' rows
    (``summary.json``'s ``bits``) but the network certifier's own cells (``row``), which its rows' registrations bind."""
    return summary["T"] * sum(b for k, b in summary["bits"].items() if k.rsplit("/", 1)[-1] != "row")


def scan_window(sess: Mapping[str, Any], strategy: str, delta: Fraction) -> tuple[dict, dict, int] | None:
    """Link-window 0's scan registration, its one-stage answer re-decided at ``delta`` (`audit_answer`; a refused
    audit is a reject for that registration), and its free inputs (`scan_free`); None when the run's scan proved
    nothing (refused before it registered, or not run)."""
    sp = (sess.get("scan_prove") or {}).get(strategy)
    if not sp or not (Path(sp["dir"]) / "honest" / "registration.json").exists():
        return None
    d = Path(sp["dir"])
    reg = json.loads((d / "honest" / "registration.json").read_text())
    record = json.loads((d / "honest" / "audit.json").read_text()) if (d / "honest" / "audit.json").exists() else None
    if record is not None and record.get("accepted") is True:
        answer = audit_answer(reg, {"verdict": "accept"}, record, delta)
    else:
        answer = {"commitment": R.digest(reg), "profile": None, "protocol": ONE_STAGE, "public": None,
                  "reason": f"the scan's audit of link-window 0 refused (scan_zk.py exited {sp['exit']})",
                  "verdict": "reject"}
    return reg, answer, scan_free(json.loads((d / "summary.json").read_text()))


def warden_tampers() -> list[str]:
    """``WARDEN_TAMPER``'s tampered runs, comma-separated (empty: none), `WARDEN_TAMPERS` when it is unset."""
    sys.path.insert(0, str(TRACES))
    import torch_loop_stub as S

    got = WARDEN_TAMPERS if os.environ.get("WARDEN_TAMPER") is None else [
        t for t in os.environ["WARDEN_TAMPER"].split(",") if t]
    if bad := [t for t in got if t not in S.TAMPERS]:
        raise RuntimeError(f"WARDEN_TAMPER: {bad} not of the stub's tampers {S.TAMPERS}")
    return list(got)


def warden_session(period_dir: Path) -> dict[str, Any]:
    """The run's one warden session (module docstring), made at its first period in the runner's scratch beside the
    period's window (``period_dir.parent``); its record ``session.json`` (`once`)."""
    return once(period_dir.parent / WARDEN_SESSION, lambda s: _warden_session(period_dir, s))


def _warden_session(period_dir: Path, s: Path) -> dict[str, Any]:
    tools, exes = _tools(period_dir), lean_exes()
    prompts, steps, h = requests_of(period_dir)
    rec = warden_rehearse(s, prompts, steps, h, _periods(period_dir), _env_int("CERTIFIER_HEIGHT", 10),
                          ["honest", *warden_tampers()])
    draw = _env_int("SCAN_DRAW", SCAN_DRAW)
    rec["scan_prove"] = {st: scan_prove(s / "rehearsal" / st, s, st, tools, draw) for st in rec["strategies"]
                         if rec["scan"][st]["accepted"]}
    rec["exes"] = exes
    (s / "session.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def warden_publish(sess: Mapping[str, Any], out: Path) -> None:
    """The warden session's public files into ``out`` (the runner's ``--out/warden-session``), each run's alone, as
    the gate publishes a warden rehearsal's (`auditor.warden_outcome`): its deployment, its rows' registrations
    (``cells.jsonl``) and its groups (`WARDEN_PUBLIC`), and ``certificates.jsonl``, each certificate of
    ``certified.jsonl`` without the head and salt it signs. The salts, the records, the client's log, the traffic and
    the prompts stay in the session, in the runner's scratch."""
    for strategy in sess["strategies"]:
        run, dst = Path(sess["dir"]) / "rehearsal" / strategy, out / strategy
        dst.mkdir(parents=True, exist_ok=True)
        for f in WARDEN_PUBLIC:
            if (run / f).is_file():
                shutil.copy(run / f, dst / f)
        (dst / "certificates.jsonl").write_text("".join(
            json.dumps({k: d[k] for k in ("certificate", "refused") if k in d}, sort_keys=True) + "\n"
            for d in _jsonl(run / "certified.jsonl")))


def warden_period_windows(groups: Mapping[str, Any], period: int, per: int) -> list[int]:
    """The link-windows of period ``period``: group ``period``'s in ``groups.json`` (`warden_serve.groups`, half-open),
    whose requests must be the period's ``per``, request ``period·per + i`` the period's ``r<i>``. The groups' windows
    tile the run, so every certified link-window is some period's."""
    g = next((g for g in groups["groups"] if g["group"] == period), None)
    if g is None:
        raise RuntimeError(f"the warden session carries {len(groups['groups'])} groups, not period {period}'s")
    if g["requests"] != list(range(period * per, (period + 1) * per)):
        raise RuntimeError(f"group {period}'s requests are {g['requests']}, not period {period}'s {per}")
    lo, hi = g["windows"]
    return list(range(lo, hi))


def warden_windows(sess: Mapping[str, Any], strategy: str) -> dict[str, Any]:
    """A warden run of the session: its deployment and groups, and per link-window the network certifier's two row
    registrations (``cells.jsonl``, by direction) and its three records' certificates (``certified.jsonl``, by record:
    ``requests``, ``responses``, ``fails``), None where it has none."""
    sys.path.insert(0, str(TRACES))
    import warden_serve as WS

    run = Path(sess["dir"]) / "rehearsal" / strategy
    groups = json.loads((run / WS.GROUPS).read_text())
    regs = [d["registration"] for d in _jsonl(run / "cells.jsonl") if "registration" in d]
    certs = [d["certificate"] for d in _jsonl(run / "certified.jsonl") if "certificate" in d]
    windows = {}
    for w in range(groups["groups"][-1]["windows"][1]):
        rows = {d: next((r for r in regs if (r["link"], r["direction"], r["window"]) == (WS.LINK, d, w)), None)
                for d in ("ingress", "egress")}
        records = {}
        for name, (link, d) in WS.CERTIFIED.items():
            at = (link, d.value, w)
            records[name] = next((_cert(c) for c in certs if (c["link"], c["direction"], c["window"]) == at), None)
        windows[w] = {"registrations": rows, "certificates": records}
    return {"run": run, "deployment": json.loads((run / "deployment.json").read_text()), "groups": groups,
            "windows": windows}


def warden_ids(run: Path, windows: Sequence[int]) -> dict[int, list[int]]:
    """Each connection's ids on the network certifier's egress in link-windows ``windows``, by its stream (the
    warden's ordinal), in cell order: the id-form data cells of ``responses.records`` (`io_tie.parse_cell`)."""
    sys.path.insert(0, str(TRACES))
    import warden_serve as WS
    from verity.core.protocols.certifier import io_tie as IT, warden as W, warden_rows as WR

    params = W.WardenParams(**json.loads((run / "deployment.json").read_text())["params"])
    B, ids = params.cell_bytes, {}
    records = (WS.ar.unpack_record(b, B) for b in WS.ar.read_blobs(run / f"{WS.EGRESS}.records"))
    for rec in sorted((r for r in records if r.window in windows), key=lambda r: r.window):
        for row in WR.window_rows(rec, params):
            for s in range(params.lanes * params.slots):
                cell = IT.parse_cell(row[s * B:(s + 1) * B])
                if cell is not None and cell.data:
                    ids.setdefault(cell.stream, []).append(cell.id)
    return ids


def match_ids(ordinals: Sequence[int | None], ids: Mapping[int, Sequence[int]],
              outputs: Mapping[tuple[str, int], bytes], steps: int) -> dict:
    """The period's registered requests ``r0, r1, …`` against the egress's ids: request ``r<i>``'s connection
    (``ordinals[i]``, None when the client has none) must carry each step's `torch_loop_stub.scan_id` of its outputs
    and nothing else. ``by`` is ``position`` when it does, ``content`` when another connection does; ``equal`` when
    every request's is, and no other connection sent ids in the period's link-windows (``others``)."""
    sys.path.insert(0, str(TRACES))
    import torch_loop_stub as S

    reqs = sorted({r for r, _ in outputs}, key=lambda r: int(r.lstrip("r")))
    if len(ordinals) != len(reqs):
        raise ValueError(f"{len(ordinals)} connections for the period's {len(reqs)} registered requests")
    rows = []
    for r, o in zip(reqs, ordinals):
        want = [S.scan_id(outputs[r, k]) for k in range(steps)]
        got = list(ids.get(o, ())) if o is not None else None
        by = "position" if got == want else "content" if want in [list(v) for v in ids.values()] else None
        rows.append({"request": r, "ordinal": o, "ids": len(got or ()), "by": by})
    others = sorted(set(ids) - set(ordinals))
    return {"equal": all(x["by"] for x in rows) and not others, "others": others, "requests": rows}


def warden_io(run: Path, period: int, windows: Sequence[int], outputs: Mapping[tuple[str, int], bytes],
              steps: int) -> dict:
    """The period's certified ids against its registered outputs (`match_ids`), in the clear: request ``period·per +
    i`` of the run is the period's ``r<i>``, its connection ``client.json``'s ordinal for it."""
    per = len({r for r, _ in outputs})
    conns = {c["index"]: c for c in json.loads((run / "client.json").read_text())["connections"]}
    ordinals = [conns.get(period * per + i, {}).get("ordinal") for i in range(per)]
    return match_ids(ordinals, warden_ids(run, windows), outputs, steps)


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


def rows_case_of(rw: Mapping[str, Any], windows: Sequence[int], scans: Mapping[int, tuple[dict, dict, int]],
                 audits: Sequence[tuple[dict, dict]], first: int, free: Sequence[int]) -> dict[str, Any]:
    """``verity/certifier/v0``'s case in the rows form (`NetworkCheck.Period`'s) of a warden run's link-windows
    ``windows`` (`warden_windows`), each with its scan's registration and answer (``scans[w]``, `scan_window`), and
    the compute ``audits`` with their ``free`` inputs: the network certifier's one schedule and key from the run's
    deployment, the scan's layout terms its free inputs over one link-window's buckets, and the audits' horizon the
    link-windows' buckets."""
    sys.path.insert(0, str(TRACES))
    import warden_serve as WS

    if len(free) != len(audits):
        raise ValueError(f"{len(free)} free-input counts for {len(audits)} audits")
    dep = rw["deployment"]
    key, T = WS.cs.read_public(dep["key"]), dep["params"]["window"]
    N, W, H = dep["key"]["params"]
    xs, sc = [rw["windows"][w] for w in windows], [scans[w] for w in windows]
    program, scan_free_bits = {r["program"] for r, _, _ in sc}, {f for _, _, f in sc}
    if len(program) != 1 or len(scan_free_bits) != 1:
        raise RuntimeError(f"the link-windows' scans register {len(program)} Programs and {len(scan_free_bits)} "
                           "layouts, not the scan's one")
    params = {**layout(free, len(xs) * T), "key": key.public.hex(), "link": WS.LINK,
              "lms": {"H": H, "N": N, "W": W, "toy": False}, "program": program.pop(),
              "scan": {"free": scan_free_bits.pop(), "steps": T}, "schedule": dict(dep["params"]),
              "session": dep["session"], "windows": len(xs), "first": first}
    commitment = canonical_json({"audits": [r for r, _ in audits], "windows": [r for r, _, _ in sc]})
    evidence = canonical_json({"audits": [a for _, a in audits], "windows": [
        {"answer": a, "certificates": [x["certificates"][n] for n in (WS.INGRESS, WS.EGRESS, WS.FAILS)],
         "registrations": [x["registrations"][d] for d in ("ingress", "egress")]} for x, (_, a, _) in zip(xs, sc)]})
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

def term_delta(delta: Fraction, terms: int) -> Fraction:
    """The δ each of a case's ``terms`` is decided at for its answer to hold at ``delta``: `NetworkCheck.Period`'s
    ``eps`` is the sum of its terms' δ (each compute audit's, and in the rows form each link-window's scan's; the clock
    form's link-window answers add none), so each takes an equal share."""
    if terms < 1:
        raise ValueError(f"a case has at least one term, not {terms}")
    return delta / terms


def holds_at(index: int, a: Mapping[str, Any], delta: Fraction) -> None:
    """An accepted answer's ``eps`` is the ``delta`` its terms were split for (`term_delta`), else the leg's error."""
    if a["verdict"] == "accept" and Fraction(a["public"]["eps"]) != delta:
        raise RuntimeError(f"period {index}: eps {a['public']['eps']} for δ = {delta}: the case has a term its split "
                           "(term_delta) didn't count")


def _audit_records(period_dir: Path, out: Path) -> list[tuple[dict, dict, dict | None]]:
    """The period's compute audits, undecided: the registration, the integrity leg's answer and its audit record."""
    reg = json.loads((period_dir / "registration.json").read_text())
    integrity = json.loads((period_dir / "answers" / "sampled-proofs.json").read_text())
    rec_file = out.parent / "integrity" / "audit" / "window" / "audit.json"
    record = json.loads(rec_file.read_text()) if rec_file.exists() else None
    return [(reg, integrity, record)]


def _audits(records: Sequence[tuple[dict, dict, dict | None]], delta: Fraction) -> list[tuple[dict, dict]]:
    return [(reg, audit_answer(reg, integrity, record, delta)) for reg, integrity, record in records]


def leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The period's network certifier check (module docstring) into ``out``, on the path ``CERTIFIER_PATH`` names
    (`PATHS`); its answer at δ = 1/100."""
    path = os.environ.get("CERTIFIER_PATH") or PATHS[0]
    if path not in PATHS:
        raise RuntimeError(f"CERTIFIER_PATH: one of {PATHS}, not {path!r}")
    return (warden_leg if path == "warden" else certifier_leg)(period_dir, node, out)


def warden_leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The period on the network certifier's own cells (module docstring, "The warden path"): its link-windows, their
    registrations and certificates, the certified ids against the registered outputs, the scan's dry chain, and where
    every link-window has a scan answer the rows form's case decided by ``network-period check``; the tampered runs'
    tampered period must be refused. Its answer at δ = 1/100; a period with a link-window the scan didn't answer
    raises (`WARDEN_PENDING`)."""
    index = json.loads((period_dir / "period.json").read_text())["period"]
    sess = warden_session(period_dir)
    warden_publish(sess, out.parent.parent / WARDEN_SESSION)
    summary: dict[str, Any] = {
        "path": "warden", "stubbed": list(STUBBED["warden"]),
        "session": {"requests": len(sess["prompts"]), **{k: sess[k] for k in (
            "steps", "h", "periods", "height", "strategies", "rehearse_exit", "warden_accepted")}}}
    outputs = registered_outputs(loop_units(period_dir))
    summary["free_inputs"] = free_inputs(json.loads((period_dir / "circuit.json").read_text()))
    integrity = json.loads((period_dir / "answers" / "sampled-proofs.json").read_text())["verdict"]
    per = len(sess["prompts"])

    def fail(why: str) -> RuntimeError:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        return RuntimeError(why)

    summary["scan_prove"] = sess.get("scan_prove") or {}
    for name, sp in summary["scan_prove"].items():
        if sp["exit"] != 0:
            raise fail(f"scan_zk.py prove of the {name} run's link-window 0 exited {sp['exit']}: a round not as "
                       f"expected, or witness in what the verifier received ({sp['dir']})")
    exe, answer, records = sess["exes"]["network-period"], None, _audit_records(period_dir, out)
    for strategy in sess["strategies"]:
        rw = warden_windows(sess, strategy)
        ws = warden_period_windows(rw["groups"], index, per)
        scan = sess["scan"][strategy] if 0 in ws else None
        row = {"windows": ws, "io": warden_io(rw["run"], index, ws, outputs, sess["steps"]), "scan": scan,
               "unsigned": {w: sorted([*(d for d, r in rw["windows"][w]["registrations"].items() if r is None),
                                       *(n for n, c in rw["windows"][w]["certificates"].items() if c is None)])
                            for w in ws if None in (*rw["windows"][w]["registrations"].values(),
                                                    *rw["windows"][w]["certificates"].values())}}
        summary[strategy] = row
        unanswered = [w for w in ws if w != 0 or scan_window(sess, strategy, DELTAS[0]) is None]
        if unanswered:
            row["pending"] = (f"link-windows {unanswered}: {WARDEN_PENDING}" if unanswered != [0] else
                              "the scan refused link-window 0 before it registered it: no scan answer to decide on")
        else:
            for k, delta in enumerate(DELTAS):
                d = term_delta(delta, len(records) + len(ws))
                case = rows_case_of(rw, ws, {0: scan_window(sess, strategy, d)}, _audits(records, d), ws[0],
                                    [summary["free_inputs"]["bits"]])
                a = check(exe, case)
                holds_at(index, a, delta)
                name = ("" if strategy == "honest" else f"{strategy}.") + "certifier" + (
                    "" if k == 0 else f".delta-{delta.denominator}")
                (out / f"{name}.case.json").write_text(json.dumps(case, sort_keys=True))
                (out / f"{name}.answer.json").write_text(json.dumps(a, sort_keys=True))
                pub = a.get("public") or {}
                row[str(delta)] = {"verdict": a["verdict"], "reason": a.get("reason"), "term_delta": str(d),
                                   "K": pub.get("K"), "eps": pub.get("eps"), "scans": pub.get("scans")}
                if strategy == "honest" and k == 0:
                    answer = a
        if strategy == "honest":
            if row["unsigned"]:
                raise fail(f"period {index}: the honest run's link-windows {sorted(row['unsigned'])} lack a "
                           f"registration or certificate: {row['unsigned']}")
            if scan is not None and not scan["accepted"]:
                raise fail(f"period {index}: the scan refuses the honest run's link-window 0: {scan['reason']}")
            if not row["io"]["equal"]:
                if integrity == "accept":
                    raise fail(f"period {index}: the certified ids are not the period's registered outputs', though "
                               "the integrity leg accepted them")
                row["io"]["refused"] = (f"the certified ids are not the period's registered outputs'; the integrity "
                                        f"leg answered {integrity}")
        elif index == WARDEN_TAMPERED_PERIOD:
            port = [row[str(d)]["verdict"] for d in DELTAS if str(d) in row]
            refused = [*(["period check"] if port and "accept" not in port else []),
                       *(["scan"] if scan is not None and not scan["accepted"] else []),
                       *(["tie"] if not row["io"]["equal"] else [])]
            row["refused_by"] = refused
            if not refused or (strategy == "late" and "scan" not in refused):
                raise fail(f"period {index}: the {strategy} run's tampered period is refused by {refused or 'nothing'}"
                           f", not {'the scan' if strategy == 'late' else 'the output tie'}")
    (out / "io.json").write_text(json.dumps(summary["honest"]["io"], indent=1, sort_keys=True))
    (out / "runs.json").write_text(json.dumps([]))
    if answer is None:
        raise fail(f"period {index}: {summary['honest']['pending']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return answer


def certifier_leg(period_dir: Path, node: str, out: Path) -> dict[str, Any]:
    """The period on `certifier_serve.py`'s certifier-only path (module docstring); its answer at δ = 1/100."""
    index = json.loads((period_dir / "period.json").read_text())["period"]
    sess = session(period_dir)
    publish(sess, out.parent.parent / SESSION)
    n2 = n2_report(sess)
    summary: dict[str, Any] = {
        "path": "certifier",
        "session": {"requests": len(sess["prompts"]), **{k: sess[k] for k in (
            "steps", "h", "periods", "height", "strategies", "tamper", "rehearse_exit")}},
        "n2": n2, "stubbed": list(STUBBED["certifier"])}
    if why := n2_refusal(n2):
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        raise RuntimeError(why)
    units = loop_units(period_dir)
    outputs = registered_outputs(units)
    free = free_inputs(json.loads((period_dir / "circuit.json").read_text()))
    integrity = json.loads((period_dir / "answers" / "sampled-proofs.json").read_text())["verdict"]
    per, exe, records = len(sess["prompts"]), sess["exes"]["network-period"], _audit_records(period_dir, out)
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
            d = term_delta(delta, len(records))
            case = case_of(rw, ws, _audits(records, d), ws[0], [free["bits"]])
            a = check(exe, case)
            holds_at(index, a, delta)
            suffix = "" if k == 0 else f".delta-{delta.denominator}"
            name = "certifier" if strategy == "honest" and k == 0 else f"{tag}certifier{suffix}"
            (out / f"{name}.case.json").write_text(json.dumps(case, sort_keys=True))
            (out / f"{name}.answer.json").write_text(json.dumps(a, sort_keys=True))
            row[str(delta)] = {"verdict": a["verdict"], "reason": a.get("reason"), "term_delta": str(d),
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


# --- the block record ----------------------------------------------------------------------------------------------------

#: the block record's δs (``verity/block-record/v1``'s ``at``): a leg's certificate holds at δ where its error is at
#: most δ. The leg splits each of its `DELTAS` across a period's terms (`term_delta`), so its answer at δ has error δ.
BLOCK_DELTAS = (Fraction(1, 100), Fraction(1, 10 ** 6))
#: where the block record says a leg's error is read from: the runner row's ``quantity`` (the answer's public)
BLOCK_ERROR_FROM = "quantity.eps"


def block_leg(answer: Mapping[str, Any], *, server: str, runs: Sequence[str], source: str = "real",
              owner: str | None = "network-accounting") -> dict[str, Any]:
    """The block record's certifier leg (``legs.certifier`` of ``verity/block-record/v1``, CMP-95 item 6) of one
    period's ``network-period check`` answer, in the clock form or the rows form: the answer's verdict, reason,
    commitment and public; the quantity K, the bits a covert sender in ``server`` can move over the link in the
    period's ``windows`` link-windows; the error the answer's exact ``eps``; and the certificate at each δ of
    `BLOCK_DELTAS`, K wherever ``eps`` ≤ δ, else not established. ``runs``: the run ids it comes from; ``source``:
    ``real`` or ``fixture``. Only an accepted answer certifies a K, so any other is a ValueError: the record's
    quantity has no shape for a period that certifies nothing."""
    if answer.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol {answer.get('protocol')!r}: not {PROTOCOL}")
    if answer.get("verdict") != "accept":
        raise ValueError(f"a {answer.get('verdict')} period certifies no K ({answer.get('reason')})")
    if source not in ("real", "fixture"):
        raise ValueError(f"source {source!r}: real or fixture")
    if not runs:
        raise ValueError("a leg comes from at least one run")
    pub = answer["public"]
    K, eps = int(pub["K"]), Fraction(pub["eps"])
    at = {str(d): {"holds": True, "value": K, "epsilon": None, "by": "run"} if eps <= d else
          {"holds": False, "why": f"the run established it except with {eps}, more than {d}"} for d in BLOCK_DELTAS}
    return {"protocol": PROTOCOL, "owner": owner, "server": server, "source": source,
            "answer": {k: answer.get(k) for k in ("verdict", "reason", "commitment", "public")},
            "quantity": {"symbol": "K", "value": K, "unit": "bits", "windows": int(pub["windows"])},
            "error": str(eps), "error_from": BLOCK_ERROR_FROM, "at": at, "runs": list(runs)}


__all__ = ["BLOCK_DELTAS", "BLOCK_ERROR_FROM", "DELTAS", "PATHS", "PROTOCOL", "PUBLIC_SESSION", "STUBBED",
           "TAMPERED_WINDOW", "WARDEN_PENDING", "WARDEN_PUBLIC", "WARDEN_TAMPERED_PERIOD", "WARDEN_TAMPERS",
           "answer_of_outcome", "audit_answer", "block_leg", "case_of", "certified_events", "certifier_leg", "check",
           "clock_evidence", "free_inputs", "holds_at", "io", "layout", "leg", "loop_units", "match_ids",
           "match_outputs", "n2_refusal", "n2_report", "once", "period_windows", "publish", "registered_outputs", "requests_of",
           "rows_case_of", "SCAN_DRAW", "SCAN_KEEP", "scan_dry", "scan_free", "scan_prove", "scan_schedule",
           "scan_window", "session", "term_delta", "warden_argv", "warden_ids", "warden_io", "warden_leg",
           "warden_period_windows", "warden_publish", "warden_rehearse", "warden_session", "warden_tampers",
           "warden_windows"]
