"""network-e2e: network accounting end to end on a served model, as one report. A certifier-serve rehearsal's runs (the
network certifier in front of a served endpoint, `certifier_serve.py`), N2's verdict of record on every link-window and
the link's network verdict (`LinkWindows.verify` through the proof service's stream, `network_verdict`), every
link-window's head checks proved by C-Flock ``prove --zk`` behind the Lean firewall and held by Lean (`n2_zk.prove`), and
the negatives (``late``, ``extra`` and a tampered head) refused, in `report.json` and `report.md` under ``--out``.

    python benchmarks/network_e2e/e2e.py --from SRC [SRC ...] --out DIR (--bin flock-circuit --lean flock-verify |
                                         --proofs N2ZK_DIR | --no-prove)
    python benchmarks/network_e2e/e2e.py (--upstream URL | --vllm MODEL) --out DIR [...] [certifier_serve.py rehearse's options]

`run.sh` is the runner (`verity bench network_e2e`): it builds the binaries and the firewall's frame as `n2_zk.sh` does,
then runs this. The README says what the report establishes and what it doesn't.

* **The runs** (`collect`). Each ``--from`` SRC is a rehearsal (`certifier_serve.py rehearse --out`'s directory: its
  `summary.json` and one run directory per strategy), a gate run (`vy-challenger-gate fetch RUN`'s directory: `end.json`
  and the rehearsal in `out/`), one strategy's run directory (its `certified.jsonl`), or a tarball of any of these. With
  ``--upstream`` or ``--vllm`` the rehearsal is run first (`certifier_serve.py rehearse`, honest, late and extra, every
  option this file doesn't take passed on) into ``DIR/rehearsal``. An honest run is required: the tampered head is its
  window 0's egress head one count off (`n2_zk.one_count`).
* **N2.** Per run, each link-window committed under fresh salts, registered on N2's stream, selected (Selection All) and
  decided by N2's verifier of record on the host, with the Lean audit (`n2_zk.link_windows`), then `network_verdict`.
  Honest, tampered and extra are `n2_zk.prove`'s own rounds, so their verdicts are on the roots it proves under; any other
  run's (late) are taken here the same way.
* **Proofs** (planned profile). `n2_zk.prove` on the honest run and, when its plan has the honest run's windows, the
  extra run: each member (honest egress and ingress, tampered egress, extra egress) one statement of n units, drawn by the
  verifier's Lean, proved ``--zk`` behind the firewall and held by Lean; the verifier opens each unit's ``pub`` row to its
  own copy and its verdict row. With ``--proofs`` they are a run's of its own, adopted (`adopt`: each proved unit's
  certificate digest must be these runs'), so that what N2-ZK's run proved isn't proved again. On the served profile
  nothing is proved: its count check is `NetworkServedCounts_v1`, which `n2_zk` doesn't prove (the claim says so).
* **Cost.** Each member's session: the prover's time (``prove_total_s``), Lean's ``verify --zk`` time and the proof's
  bytes, each divided by its n units for a link-window's end, and the peak RSS of the largest process the driver has run
  so far (``max_child_rss_gb``, `RUSAGE_CHILDREN`, which only grows: n2_zk proves the honest members first, so the cost,
  read from the honest run, is theirs); the driver's own peak (the unit's lowering) beside them.

Exit 0 when the honest run is accepted everywhere it was checked and each negative is refused (`report["ok"]`).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "network_traces"))

import n2_zk  # noqa: E402
from verity.core.protocols.certifier import head_check as HC  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402

FORMAT = "verity-network-e2e/report/v1"
HONEST, TAMPERED = "honest", "tampered"
NEGATIVES = ("late", "extra", TAMPERED)
README = "verity/core/protocols/certifier/README.md"

CLAIM = {
    "statement": (
        "Counts only. A report marked OK establishes, for the rehearsal's one link: on every link-window of the honest "
        "run N2's verdict of record accepted (LinkWindows.verify, through the proof service's stream with Selection "
        "All), and the link's network verdict, the AND over its link-windows with no epsilon and no share credit "
        "(LinkWindows.network_verdict), accepted, so each window's per-bucket frame counts are the planned occupancy "
        "profile's on egress (on a served link, the ones its ingress head's counts derive under the declared sync), "
        "which puts the link's timing at the profile; and late, extra and a tampered head were each refused. Where "
        "proofs ran, the head checks (NetworkHeadCheck_v2: the head's hiding opening under the registered root, its "
        "fields, its counts against the profile, and its record digest under the certifier's salt against the digest "
        "its certificate signs) of every honest link-window at both ends, and the egress head checks of the tampered "
        "run and, when its plan has the honest run's windows, the extra run, were proved by C-Flock prove --zk behind "
        "the Lean firewall and held by Lean verify --zk, the verifier opening only each unit's public row and its "
        "verdict row. Late's head checks are not proved: N2 alone refuses it."),
    "not_covered": [
        "N3 (content): the frames inside the counts, their bytes and their order, are not checked. N3 is a design and "
        "the simulator's behaviour, not a proof-service unit, so nothing here holds the egress grid to the Program's.",
        "The proofs cover each end's head check only: not the certificates' LMS check (NetworkCertificatesCheck, on the "
        "host by design, over public fields and the signed digest), not the audit of each record's status, signature "
        "and declared syncs (on the host, over the opening), and on a served link not NetworkServedCounts_v1, which "
        "has no prover here: a served run gets N2's verdict of record and no proof.",
        "The rehearsal is a rehearsal: the client's link is loopback, its check plays the firewall and the challenger "
        "in one process, and it publishes the records, certificates and certifier's salts (witness) so that its check "
        "can be replayed (certifier_serve.py's docstring). Off the gate the certifier is a process on the host that "
        "ran the rehearsal, which against an endpoint on its own loopback is the prover's node, not an auditor host "
        "(the report's 'where' says which). n2_zk runs the prover, the firewall and the verifier, whose live coins "
        "the proofs take, on one node. The model is the one the plan names: whether it was cut or full, and the "
        "response length, are the run's (the report's 'tokens').",
    ],
    "zero_knowledge_gaps": [
        "N2's unit isn't proved as the verifier of record (README, 'Not yet zero-knowledge'): the verdict of record is "
        "LinkWindows.verify on the host, which runs head_check.reference over the opening and so reads the heads and "
        "the certifier's salts. The proofs here run beside it, on the roots it registered; they don't replace it.",
        "The proofs here are of n2_zk's row adapter around the unchanged NetworkHeadCheck_v2, not of N2's registered "
        "unit, so LinkWindows.proved, which refuses a proof of any Program but the unit's (NET-43), doesn't take them. "
        "Lean holds each proof to the program and partition the driver wrote.",
        "The certificate's link to its head (README, NET-42): until the unit is proved, the verifier of record checks "
        "each signed digest against its head and the certifier's salt on the host, over the opening "
        "(CertifiedRecord.verify).",
    ],
    "source": README,
}


def log(msg: str) -> None:
    print(f"E2E {time.strftime('%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --- the runs ---------------------------------------------------------------------------------------------------------

def _unpack(src: Path, work: Path) -> Path:
    """A tarball's single top directory, or the directory it was extracted into."""
    dest = Path(tempfile.mkdtemp(prefix="from-", dir=work))
    with tarfile.open(src) as tar:
        tar.extractall(dest, filter="data")
    tops = [p for p in dest.iterdir()]
    return tops[0] if len(tops) == 1 and tops[0].is_dir() else dest


def _check(doc: dict | None) -> dict | None:
    """A run's own check (`certifier_serve.check`'s, at the rehearsal): its verdict and parts."""
    if not doc:
        return None
    return {"verdict": doc.get("verdict"), "network": (doc.get("network_verdict") or {}).get("accepted"),
            "audit": doc.get("audit_accepted"), "salt": doc.get("salt_accepted"), "salt_again": doc.get("salt_again")}


def collect(sources: list[Path], work: Path) -> dict[str, dict]:
    """Each strategy's run directory, with what its rehearsal's own check and the gate's `end.json` said of it."""
    found: dict[str, dict] = {}
    for src in sources:
        d = _unpack(src, work) if src.is_file() else src
        gate = _read(d / "end.json")
        if gate is not None:
            d = d / "out"
        summary = _read(d / "summary.json") or {}
        if (d / "certified.jsonl").exists():
            items = [((_read(d / "check.json") or {}).get("strategy") or d.name, d)]
        else:
            items = [(s, d / s) for s in summary.get("runs", {})] or \
                    [(p.name, p) for p in sorted(d.iterdir()) if (p / "certified.jsonl").exists()]
        if not items:
            raise SystemExit(f"{src}: no run (a rehearsal's directory, a gate run's, a run's certified.jsonl)")
        for s, run in items:
            if s in found:
                raise SystemExit(f"two {s} runs: {found[s]['dir']} and {run}")
            if not (run / "certified.jsonl").exists():
                raise SystemExit(f"{run}: no certified.jsonl")
            check = _check(_read(run / "check.json")) or _check((summary.get("runs") or {}).get(s))
            found[s] = {"dir": run, "source": str(src), "check": check,
                        "gate": None if gate is None else {"run": gate.get("run"), "rc": gate.get("rc"),
                                                           "ended": gate.get("ended"), "pinned": gate.get("pinned"),
                                                           **((gate.get("runs") or {}).get(s) or {})}}
    if HONEST not in found:
        raise SystemExit(f"no honest run among {sorted(found)}: the tampered head is the honest run's")
    return found


def where(runs: dict[str, dict]) -> dict:
    """Where the honest run's certifier ran: on the gate's host (the auditor's machine, through its tunnel to the
    endpoint), or off the gate on the host that ran the rehearsal, which against a loopback endpoint is the endpoint's."""
    upstream = (_read(runs[HONEST]["dir"] / "deployment.json") or {}).get("upstream")
    if runs[HONEST]["gate"] is not None:
        return {"upstream": upstream, "where": "gate: the certifier on the gate's host, the auditor's, tunnelled to the "
                                               "endpoint"}
    host = urllib.parse.urlsplit(upstream or "").hostname or ""
    local = host == "localhost" or host.startswith("127.") or host == "::1"
    return {"upstream": upstream, "where": "off the gate: the certifier on the host that ran the rehearsal" +
            (", the endpoint's own (loopback), so the prover's node, not an auditor host" if local else "")}


def rehearse(args, rest: list[str], out: Path) -> Path:
    """`certifier_serve.py rehearse` into ``out``, honest, late and extra unless ``rest`` names its strategies."""
    me = [sys.executable, str(ROOT / "benchmarks" / "network_traces" / "certifier_serve.py"), "rehearse", "--out",
          str(out), *(["--upstream", args.upstream] if args.upstream else ["--vllm", args.vllm]), *rest]
    log(f"rehearsal: {' '.join(me[2:])}")
    rc = subprocess.run(me, cwd=ROOT).returncode
    log(f"rehearsal exited {rc} (1: a verdict not as expected; the report says which)")
    if not (out / "summary.json").exists():
        raise SystemExit(f"the rehearsal wrote no summary.json (exit {rc})")
    return out


# --- N2 and the proofs ------------------------------------------------------------------------------------------------

def parts(rd: SimpleNamespace, x: SimpleNamespace) -> dict[str, list[str] | None]:
    """Each end's head-check parts that fail on link-window x, by the reference in the clear
    (`head_check.reference_parts`), and on a served link the served counts (`LinkWindows.served_check`)."""
    out = {end: None if c is None else [k for k, ok in HC.reference_parts(c).items() if not ok]
           for end, c in zip(n2_zk.ENDS, x.checks)}
    if rd.lw.served is not None and not rd.lw.served_check(x.opening, [ClockSync(d, Fraction(r))
                                                                         for d, r in rd.plan["syncs"]]):
        out["egress"] = (out["egress"] or []) + ["served-counts"]
    return out


def host(rd: SimpleNamespace) -> dict:
    """`n2_zk.link_windows`'s result as `n2_zk.prove` records N2's verdicts (its summary's ``host``), with each
    link-window's failed parts (`parts`)."""
    return {"windows": [{"window": x.window, "registration": x.digest, **(x.host or {}), "parts": parts(rd, x)}
                        for x in rd.windows], "network": rd.network, "issuer_refused": rd.issuer_refused}


def _ports(n: int) -> int:
    """A base port with ``n`` free loopback ports after it (`n2_zk.prove` takes member j's session at base + j)."""
    for _ in range(64):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = s.getsockname()[1]
        if base + n > 65535:
            continue
        try:
            for p in range(base, base + n):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", p))
        except OSError:
            continue
        return base
    raise SystemExit(f"no {n} free loopback ports in a row")


def prove(runs: dict[str, dict], args, work: Path) -> dict:
    """`n2_zk.prove` on the honest run and, when its plan has the honest run's windows, the extra run; its summary."""
    honest, extra = runs[HONEST]["dir"], runs.get("extra", {}).get("dir")
    if extra is not None and (_read(extra / "plan.json") or {}).get("windows") != (_read(honest / "plan.json") or {}).get(
            "windows"):
        log("extra: its plan's windows aren't the honest run's, so n2_zk proves it in no member; N2's verdict only")
        extra = None
    out = work / "n2zk"
    a = SimpleNamespace(run=honest, extra=extra, bin=args.bin, lean=args.lean, out=str(out), port=_ports(8),
                        delta=args.delta, points=[], dry=False)
    try:
        rc = n2_zk.prove(a)
    except SystemExit as e:
        rc, why = 1, str(e)
    else:
        why = None
    summary = _read(out / "summary.json") or {}
    return {"rc": rc, "why": why, "dir": out, "summary": summary, "extra_proved": extra is not None}


def adopt(d: Path, runs: dict[str, dict]) -> dict:
    """`n2_zk.prove`'s output from a run of its own (its ``--out``, or `n2_zk.sh`'s ``$OUT/n2zk``), as `prove` returns
    it, so that what N2-ZK's run proved isn't proved again. Refused unless every proved member is of these runs: the
    certificate digest each of its units' public row carries (``signed``, in the verifier's ``n2.json``) is the one
    that run's certified record of that end and link-window carries (the tampered head keeps the honest certificate)."""
    summary = _read(d / "summary.json") or {}
    members = summary.get("members") or {}
    if not members:
        raise SystemExit(f"{d}: no n2_zk members (n2_zk.prove's --out, or n2_zk.sh's $OUT/n2zk)")
    for k, m in members.items():
        if not m.get("units"):
            continue
        name, end = k.rsplit("-", 1)
        run = runs.get(HONEST if name == TAMPERED else name)
        sent = _read(d / k / "verifier" / "n2.json")
        if run is None or sent is None:
            raise SystemExit(f"{d}: {k} was proved, but " + (f"--from gives no {name} run" if run is None
                                                              else f"{k}/verifier/n2.json isn't there"))
        certified, _ = n2_zk.cs.read_certified(run["dir"] / "certified.jsonl")
        direction = Direction.EGRESS if end == "egress" else Direction.INGRESS
        mine = [certified[(direction, x["window"])].certificate.digest.hex() if (direction, x["window"]) in certified
                else None for x in sent]
        if [x["signed"] for x in sent] != mine:
            raise SystemExit(f"{d}: {k}'s units carry certificates other than {run['dir']}'s: proofs of another run")
    ok = summary.get("ok")
    why = None if ok else "n2_zk's run didn't finish (its summary has no ok)" if ok is None else "n2_zk's summary: not ok"
    return {"rc": 0 if ok else 1, "why": why, "dir": d, "summary": summary, "adopted": str(d),
            "extra_proved": bool((members.get("extra-egress") or {}).get("units"))}


def keep(src: Path, dest: Path) -> None:
    """n2_zk's public files into ``dest``, as `n2_zk.sh` keeps them: never a prover's instance file, a circuit or a
    session's archive."""
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_file() and f.name != "circuit.txt" and not f.name.startswith("inst-") and "archive" not in rel.parts:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / rel)


# --- the report -------------------------------------------------------------------------------------------------------

def _n2_row(h: dict) -> str:
    return "accept" if h.get("accepted") else "refuse"


def _proof(unit: dict | None) -> str | None:
    """A unit's proved verdict: ``accept`` or ``refuse`` (its proof held by Lean and its verdict row opened), else
    ``malformed``."""
    if unit is None:
        return None
    return {"accept": "accept", "reject": "refuse"}.get(unit["verdict"], "malformed")


def _member(m: dict) -> dict:
    """One n2_zk member's proofs: held by Lean or not, what the verifier received, and the session's cost."""
    t, rec = m.get("timing") or {}, m.get("received") or {}
    return {"held": bool(m.get("accepted")) and all((m.get("verdicts") or {}).values()), "lean": m.get("verdicts"),
            "audit": m.get("audit"), "n": m.get("n"), "ands": m.get("ands"), "stage_s": m.get("stage_s"),
            "session_s": t.get("session_s"), "prove_total_s": (t.get("timing") or {}).get("prove_total_s"),
            "lean_verify_s": m.get("lean_verify_s"), "per_unit": m.get("per_unit"),
            "peak_child_rss_gb": (t.get("process") or {}).get("max_child_rss_gb"),
            "verifier_bytes": rec.get("verifier_bytes"), "witness_found_at_verifier": rec.get("verifier_found"),
            "public_pieces_left_out": None if rec.get("public_pieces") is None else len(rec["public_pieces"]),
            "control_holds_all": (rec.get("control") or {}).get("holds_all")}


def report(runs: dict[str, dict], n2: dict[str, dict], proofs: dict, meta: dict) -> dict:
    """The report (`FORMAT`): per run its network verdict, per link-window N2's verdict and the proofs' with their cost,
    the negatives' refusals and the claim. ``proofs``: `prove`'s result, or ``{"ran": False, "why": ...}``."""
    summary = proofs.get("summary") or {}
    members = summary.get("members") or {}
    names = [HONEST] + [s for s in NEGATIVES if s in n2] + sorted(set(n2) - {HONEST, *NEGATIVES})
    out_runs, rows = {}, []
    for name in names:
        h, given = n2[name], runs.get(name, {})
        ends = {e: _member(members[f"{name}-{e}"]) | {"units": members[f"{name}-{e}"]["units"]}
                for e in n2_zk.ENDS if (members.get(f"{name}-{e}") or {}).get("units")}
        out_runs[name] = {"dir": str(given.get("dir", runs[HONEST]["dir"])), "check": given.get("check"),
                          "gate": given.get("gate"),
                          "network": {k: (h["network"] or {}).get(k) for k in ("accepted", "windows", "refused", "digest")},
                          "issuer_refused": h.get("issuer_refused"),
                          "proofs": {e: {k: v for k, v in m.items() if k != "units"} for e, m in ends.items()} or None}
        peak = max((m["peak_child_rss_gb"] for m in ends.values() if m["peak_child_rss_gb"] is not None), default=None)
        for x in h["windows"]:
            w = x["window"]
            proved = {e: _proof(next((u for u in m["units"] if u["window"] == w), None)) for e, m in ends.items()}
            per = [m.get("per_unit") or {} for m in ends.values()]
            rows.append({
                "run": name, "window": w, "n2": _n2_row(x), "failed_parts": x.get("parts"), "proof": proved or None,
                "head_checks": None if not proved else "pass" if all(v == "accept" for v in proved.values())
                else "malformed" if None in proved.values() or "malformed" in proved.values() else "refuse",
                "prove_s": _sum(p.get("prove_total_s") for p in per), "verify_s": _sum(p.get("lean_verify_s") for p in per),
                "proof_bytes": _sum(p.get("proof_bytes") for p in per), "peak_gb": peak})
    negatives = {name: negative(out_runs[name], [r for r in rows if r["run"] == name])
                 for name in NEGATIVES if name in out_runs}
    honest, honest_rows = out_runs[HONEST], [r for r in rows if r["run"] == HONEST]
    proved_ok = not proofs.get("ran") or (
        proofs.get("rc") == 0 and bool(honest["proofs"]) and all(p["held"] for p in honest["proofs"].values())
        and all(r["head_checks"] == "pass" for r in honest_rows))
    ok = (bool(honest["network"]["accepted"]) and bool(honest_rows) and all(r["n2"] == "accept" for r in honest_rows)
          and (honest["check"] is None or honest["check"]["verdict"] == "accepted") and proved_ok
          and len(negatives) == len(NEGATIVES) and all(n["refused"] for n in negatives.values()))
    return {"format": FORMAT, **meta, "claim": CLAIM, "ok": ok, "runs": out_runs, "link_windows": rows,
            "negatives": negatives,
            "proofs": {"ran": bool(proofs.get("ran")), "adopted": proofs.get("adopted"), "why": proofs.get("why"),
                       "rc": proofs.get("rc"), "extra_proved": proofs.get("extra_proved"), "n2_zk_ok": summary.get("ok"),
                       "stagers": summary.get("stagers"), "driver_peak_gb": proofs.get("driver_peak_gb"),
                       "cost_per_link_window": cost(honest_rows)}}


def _sum(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs), 3) if xs else None


def cost(rows: list[dict]) -> dict | None:
    """The honest link-windows' cost, both ends: the mean prover seconds, Lean verify seconds and proof bytes."""
    got = [r for r in rows if r["prove_s"] is not None]
    if not got:
        return None
    mean = {k: _sum(r[k] for r in got) for k in ("prove_s", "verify_s", "proof_bytes")}
    return {"link_windows": len(got), **{k: None if v is None else round(v / len(got), 3) for k, v in mean.items()},
            "peak_gb": max((r["peak_gb"] for r in got if r["peak_gb"] is not None), default=None)}


def negative(run: dict, rows: list[dict]) -> dict:
    """Whether the negative was refused, and by what: N2's network verdict and its link-windows' verdicts of record,
    the rehearsal's own check, and where proved, its head checks' proved verdicts (each proof held by Lean). Refused
    only when every one of them that ran refuses it."""
    by, check = [], run.get("check")
    n2 = [r["window"] for r in rows if r["n2"] == "refuse"]
    if run["network"]["accepted"] is False and n2:
        by.append(f"N2 (network verdict; link-windows {n2})")
    if check is not None and check["verdict"] == "refused":
        by.append("the rehearsal's own check")
    proved = [r["window"] for r in rows if r["head_checks"] == "refuse"]
    held = all(p["held"] for p in (run["proofs"] or {}).values())
    if proved and held:
        by.append(f"the proved head checks (link-windows {proved})")
    refused = (run["network"]["accepted"] is False and bool(n2) and (check is None or check["verdict"] == "refused")
               and (not run["proofs"] or (held and bool(proved))))
    return {"refused": refused, "by": by, "n2_refused_windows": n2, "proved_refused_windows": proved,
            "failed_parts": {r["window"]: r["failed_parts"] for r in rows if r["n2"] == "refuse"},
            "proved": bool(run["proofs"]), "proofs_held": held if run["proofs"] else None}


def _cell(x) -> str:
    return "—" if x is None else str(x)


def markdown(r: dict) -> str:
    """The report as Markdown: the verdicts, one row per link-window, the negatives and the claim."""
    p = r.get("proofs") or {}
    lines = [f"# Network accounting end to end: {'OK' if r['ok'] else 'NOT OK'}", "",
             f"Source {', '.join(r['from'])}; model {_cell(r.get('model'))} (as the plan names it), "
             f"{_cell(r.get('tokens'))} tokens a response, {_cell(r.get('requests'))} requests; endpoint "
             f"{_cell(r.get('upstream'))}, {_cell(r.get('where'))}; N2's profile {_cell(r.get('profile'))}; "
             f"{_cell(r.get('windows'))} link-windows planned; commit {_cell(r.get('commit'))}; "
             f"proofs {'run' if p.get('ran') else 'not run'}.", "",
             "## Runs", "",
             "| run | rehearsal's check | gate | N2 network verdict | refused windows | proofs held (Lean) |",
             "|---|---|---|---|---|---|"]
    for name, x in r["runs"].items():
        c, g = x.get("check"), x.get("gate")
        held = x.get("proofs")
        lines.append(f"| {name} | {_cell(c and c['verdict'])} | {_cell(g and g.get('verdict'))} | "
                     f"{'accept' if x['network']['accepted'] else 'refuse'} | {_cell(x['network']['refused'] or None)} | "
                     f"{', '.join(f'{e}: {v['held']}' for e, v in held.items()) if held else '—'} |")
    lines += ["", "## Link-windows", "",
              "N2: the verdict of record (host). Proof: each end's head check proved --zk and held by Lean, accept or "
              "refuse as its opened verdict row says. Failed parts: the head check's parts that fail, evaluated in the "
              "clear on the rehearsal's published witness (a diagnostic, not the verdict). Prove and verify seconds and "
              "proof bytes: the member session's divided by its units, summed over the proved ends (a session proves "
              "every link-window of an end at once); peak: the largest process the driver had run by the end of the "
              "run's last session (RUSAGE_CHILDREN only grows, so the tampered and extra rows' include the honest "
              "sessions').",
              "", "| run | window | N2 | proof egress | proof ingress | head checks | failed parts | prove s | verify s "
              "| proof bytes | peak GB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["link_windows"]:
        pr = x["proof"] or {}
        failed = "; ".join(f"{e} {'+'.join(f)}" for e, f in (x["failed_parts"] or {}).items() if f) or None
        lines.append(f"| {x['run']} | {x['window']} | {_cell(x['n2'])} | {_cell(pr.get('egress'))} | "
                     f"{_cell(pr.get('ingress'))} | {_cell(x['head_checks'])} | {_cell(failed)} | {_cell(x['prove_s'])} "
                     f"| {_cell(x['verify_s'])} | {_cell(x['proof_bytes'])} | {_cell(x['peak_gb'])} |")
    lines += ["", "## Negatives", ""]
    for name, n in r["negatives"].items():
        lines.append(f"- **{name}**: {'refused' if n['refused'] else 'NOT refused'} by "
                     f"{'; '.join(n['by']) or 'nothing'}{'' if n['proved'] else ' (not proved: N2 only)'}")
    scanned = [(name, e, m) for name, x in r["runs"].items() for e, m in (x.get("proofs") or {}).items()]
    if scanned:
        lines += ["", "## What the verifier received", "",
                  "Each member's verifier directory, scanned for every piece of every head, head salt, sibling and "
                  "certifier salt but the pieces the public circuit holds (n2_zk's `public_pieces`); the prover's "
                  "instance file is the control.", ""]
        lines += [f"- {name} {e}: {_cell(m['verifier_bytes'])} bytes; witness pieces found: "
                  f"{m['witness_found_at_verifier'] or 'none'}; left out as public: {_cell(m['public_pieces_left_out'])}; "
                  f"control holds all: {_cell(m['control_holds_all'])}" for name, e, m in scanned]
    c = p.get("cost_per_link_window")
    if c:
        lines += ["", "## Cost per link-window (honest, both ends)", "",
                  f"{c['prove_s']} s proving, {c['verify_s']} s Lean verify, {c['proof_bytes']} proof bytes "
                  f"(each session's divided by its {c['link_windows']} link-windows); peak {c['peak_gb']} GB (the largest "
                  f"process by the end of the honest sessions); the driver's own peak (lowering) "
                  f"{_cell(p.get('driver_peak_gb'))} GB."]
    if p.get("adopted"):
        lines += ["", f"Proofs: adopted from {p['adopted']}, n2_zk's output of its own run, each proved unit's "
                      "certificate digest matched to these runs' certified records."]
    if p.get("why") or not p.get("ran"):
        lines += ["", f"Proofs: {p.get('why') or 'not run'}"]
    lines += ["", "## Claim", "", r["claim"]["statement"], "", "Not covered:", ""]
    lines += [f"- {x}" for x in r["claim"]["not_covered"]]
    lines += ["", f"Zero-knowledge gaps (`{r['claim']['source']}`):", ""]
    lines += [f"- {x}" for x in r["claim"]["zero_knowledge_gaps"]]
    return "\n".join(lines) + "\n"


def _commit() -> str | None:
    sha = os.environ.get("RESEARCH_SOURCE_SHA")
    if sha:
        return sha
    p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() or None


def run(args, rest: list[str], clock=time.monotonic) -> dict:
    """The runs, N2's verdicts, the proofs and the report, written under ``args.out``; returns the report. ``clock`` is
    the stream's for the runs N2 decides here (`n2_zk.prove` keeps its own)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="network-e2e-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        sources = [Path(s) for s in args.sources] if args.sources else [rehearse(args, rest, out / "rehearsal")]
        runs = collect(sources, work)
        plan = _read(runs[HONEST]["dir"] / "plan.json") or {}
        profile = (plan.get("occupancy") or {}).get("kind")
        log(f"runs: {', '.join(f'{s} {r['dir']}' for s, r in runs.items())}; N2's profile {profile}")
        if args.no_prove:
            proofs = {"ran": False, "why": "not run (--no-prove)"}
        elif profile != "planned":
            proofs = {"ran": False, "why": f"not run: the {profile} profile's count check is NetworkServedCounts_v1, "
                                           "which n2_zk doesn't prove"}
        elif args.proofs:
            src = Path(args.proofs)
            proofs = adopt(_unpack(src, work) if src.is_file() else src, runs) | {"ran": True, "adopted": str(src)}
            log(f"proofs: adopted from {args.proofs}{': ' + proofs['why'] if proofs['why'] else ''}")
        else:
            proofs = prove(runs, args, work) | {"ran": True}
            proofs["driver_peak_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)
            keep(proofs["dir"], out / "n2zk")
            log(f"proofs: n2_zk exited {proofs['rc']}{': ' + proofs['why'] if proofs['why'] else ''}")
        summary = proofs.get("summary") or {}
        n2, refs = dict(summary.get("host") or {}), summary.get("reference") or {}
        for name, h in n2.items():
            for i, x in enumerate(h["windows"]):
                x["parts"] = {e: refs[f"{name}-{e}"]["failed"][i] for e in n2_zk.ENDS if f"{name}-{e}" in refs}
        for name, r in runs.items():
            if name not in n2:
                n2[name] = host(n2_zk.link_windows(r["dir"], clock=clock))
        if TAMPERED not in n2:
            n2[TAMPERED] = host(n2_zk.link_windows(runs[HONEST]["dir"], edit=n2_zk.one_count, clock=clock))
        served = plan.get("served") or {}
        meta = {"from": [str(s) for s in sources], "commit": _commit(), "model": plan.get("live_model"),
                "tokens": served.get("tokens"), "requests": served.get("requests"), **where(runs),
                "profile": profile, "windows": plan.get("windows"), "served": plan.get("served")}
        r = report(runs, n2, proofs, meta)
        (out / "report.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
        (out / "report.md").write_text(markdown(r))
        log(f"report {out / 'report.md'}: {'ok' if r['ok'] else 'NOT ok'}")
        return r
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="e2e.py", description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="sources", nargs="+", metavar="SRC",
                     help="a rehearsal's directory, a gate run's (vy-challenger-gate fetch), a run's, or a tarball")
    src.add_argument("--upstream", help="rehearse first against this endpoint, http(s)://HOST[:PORT]")
    src.add_argument("--vllm", metavar="MODEL", help="rehearse first, starting vLLM for MODEL on this host's GPU")
    ap.add_argument("--out", default=os.environ.get("OUT"), help="the report's directory (default $OUT)")
    ap.add_argument("--work", help="scratch, kept (default: a temporary directory, removed)")
    ap.add_argument("--bin", help="C-Flock's flock-circuit (run.sh builds it)")
    ap.add_argument("--lean", help="the Lean flock-verify (run.sh builds it)")
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    how = ap.add_mutually_exclusive_group()
    how.add_argument("--no-prove", action="store_true", help="N2's verdicts and the head checks' references only")
    how.add_argument("--proofs", metavar="DIR", help="adopt n2_zk.prove's output of these runs (n2_zk.sh's $OUT/n2zk, or a "
                                                     "tarball of it) instead of proving")
    args, rest = ap.parse_known_args(argv)
    if rest and args.sources:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if not args.out:
        ap.error("--out, or $OUT")
    if not (args.no_prove or args.proofs) and not (args.bin and args.lean):
        ap.error("proving needs --bin and --lean (run.sh builds them), --proofs DIR, or --no-prove")
    r = run(args, rest)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
