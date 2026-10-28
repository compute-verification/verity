"""network-e2e: network accounting end to end on a served model, as one report. A certifier-serve rehearsal's runs (the
network certifier in front of a served endpoint, `certifier_serve.py`), N2's verdict of record on every link-window and
the link's network verdict (`LinkWindows.verify` through the proof system's stream, `network_verdict`), every
link-window's head checks proved by C-Flock ``prove --zk`` behind the Lean firewall and held by Lean (`n2_zk.prove`), and
the negatives (``late``, ``extra`` and a tampered head) refused, in `report.json` and `report.md` under ``--out``; then
the chain's stages (`chain`): the advice's bits per link-window, the prove path's audit and the negatives
(`negatives.py`), and what is still stubbed.

    python benchmarks/network_e2e/e2e.py --from SRC [SRC ...] --out DIR (--bin flock-circuit --lean flock-verify |
                                         --proofs N2ZK_DIR | --no-prove) [--advice FILE [--serving-config FILE] |
                                         --endpoint-run DIR] [--ids FILE] [--prove-path DIR ...
                                         [--prove-path-registered FILE]] [--no-negatives]
    python benchmarks/network_e2e/e2e.py (--upstream URL | --vllm MODEL | --endpoint-peer NAME:PORT) --out DIR [...]
                                         [--endpoint-run DIR] [certifier_serve.py rehearse's options]

`run.sh` is the runner (`verity bench network_e2e`): it builds the binaries and the firewall's frame as `n2_zk.sh` does,
then runs this. The README says what the report establishes and what it doesn't.

* **The runs** (`collect`). Each ``--from`` SRC is a rehearsal (`certifier_serve.py rehearse --out`'s directory: its
  `summary.json` and one run directory per strategy), a gate run (`vy-challenger-gate fetch RUN`'s directory: `end.json`
  and the rehearsal in `out/`), one strategy's run directory (its `certified.jsonl`), or a tarball of any of these. With
  ``--upstream``, ``--vllm`` or ``--endpoint-peer`` the rehearsal is run first (`certifier_serve.py rehearse`, honest,
  late and extra, every option this file doesn't take passed on) into ``DIR/rehearsal``; ``--endpoint-peer NAME:PORT``
  reaches another node's endpoint on its loopback through an ssh forward (`forward`), so the certifier runs off the
  prover's node, though not on the auditor host. The negatives' drop, when the rehearsal ran it, is a negative (refused
  by N2's counts, and on the advice profile by the rehearsal's own check), and its token is refused only by the I/O tie
  (``PENDING``): on the advice profile, where the tie is required (`chain.tie`, `chain.tie_gate`), token is a negative
  it must refuse at the link-windows its changed ids fall in; on the others, where the tie doesn't run, a row pending
  it. An honest run is required: the tampered head is its window 0's egress head one count off (`n2_zk.one_count`).
* **N2.** Per run, each link-window committed under fresh salts, registered on N2's stream, selected (Selection All) and
  decided in the clear, with the Lean audit (`n2_zk.link_windows` with a clock: `LinkWindows.unproved_outcome`, as for
  any run that proves nothing), then `network_verdict`. A summary `n2_zk.prove` wrote before N2's registered unit was
  proved (adopted) carries honest's, tampered's and extra's verdicts on the roots it proved under; any other run's are
  taken here.
* **Proofs** (planned profile). `n2_zk.prove` on the honest run and, when its plan has the honest run's windows, the
  extra run: each member (honest egress and ingress, tampered egress, extra egress) one statement of n units, drawn by the
  verifier's Lean, proved ``--zk`` behind the firewall and held by Lean; the verifier opens each unit's ``pub`` row to its
  own copy and its verdict row. With ``--proofs`` they are a run's of its own, adopted (`adopt`: each proved unit's
  certificate digest must be these runs'), so that what n2_zk's own run proved isn't proved again. Not yet on `n2_zk` as
  it now proves: one member a round, each link-window decided by its proof of N2's registered unit; `prove` refuses and
  `adopt` refuses such a summary, each saying so (``--no-prove`` takes N2's verdicts only). On the served profile
  nothing is proved: its count check is `NetworkServedCounts_v1`, which `n2_zk` doesn't prove (the claim says so).
* **Cost.** A member proves all n of its link-windows in one session, one statement and one pair of proofs, so a
  link-window has no time or proof of its own. Per session: the prover's time (``prove_total_s``), Lean's ``verify --zk``
  time, the bytes of its two proofs (``circuit.rep0.bin``, ``circuit.rep1.bin``) and the peak RSS of the largest process
  the driver had run by its end (``max_child_rss_gb``, `RUSAGE_CHILDREN`, which only grows: n2_zk proves the honest
  members first). Beside them, n2_zk's cost pass (``summary.json``'s ``cost``): one honest link-window proved alone, per
  end (`alone`); and the driver's own peak (the unit's lowering).
* **Varied traffic** (``rehearse --traffic varied``, passed on): N2's profile is ``advice``, on which N2 gives no
  verdict without N2's advised counts, so nothing is proved and the rehearsal's own check (certified, the certified I/O,
  audit, salt) is each run's verdict, late, extra and drop its negatives (`CLAIM_ADVICE`).
* **The chain** (`stages_of`, `chain`'s docstring): ``--advice`` and ``--serving-config``, or ``--endpoint-run``, the
  endpoint's records, whose own advice for the honest run's Commit windows `verity-vllm advice` writes
  (`endpoint_advice`, `chain.served_advice`), ``--prove-path`` (beside the endpoint's records, ``prove_path.py audit
  --serve-run``'s audits of its Commit windows, `chain.served_commit`), the negatives registry on the runs, and on the
  advice profile the I/O tie on each run (`chain.tie`, its sampled ids the endpoint's records' or ``--ids``'); each that
  ran must pass for the report to be OK, the tie on the advice profile whether it ran or not (`chain.tie_gate`), and
  every ``--serve-run`` arm's advice held by content to the report's advice of its window whether its prove path binds
  or not (`chain.held_to`); ``stubbed`` lists what is still a stand-in.

Exit 0 when the honest run is accepted everywhere it was checked, each negative is refused, every chain stage that
ran passed, every ``--serve-run`` arm's advice is the report's and, on the advice profile, the I/O tie accepted the
honest run (`report["ok"]`).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "benchmarks" / "network_traces")]

import chain  # noqa: E402
import n2_zk  # noqa: E402
from verity.core.protocols.certifier import head_check as HC  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction  # noqa: E402
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402

FORMAT = "verity-network-e2e/report/v3"
HONEST, TAMPERED = "honest", "tampered"
NEGATIVES = ("late", "extra", TAMPERED)
# the live tampers of `negatives.py` that only the I/O tie refuses: a negative of the tie on the advice profile, where
# the tie is required (`chain.tie_gate`), and pending it on the others, where it doesn't run
PENDING = {"token": "the I/O tie"}
TIE_STAGE = "the I/O tie (`io_tie`'s reference in the clear, on stand-ins)"
#: each negative's own violation in the rehearsal's own check (`_check`'s codes), the verdict on the advice profile,
#: where N2 gives none: late's held session faults its window (the audit's status), extra's response runs past its
#: max_tokens, and drop's cut response stops short of it without EOS (the certified I/O's finish rule)
OWN_CODE = {"late": "audit.status:coverage-never-established", "extra": "io:longer than max_tokens",
            "drop": "io:short of max_tokens without EOS"}
README = "verity/core/protocols/certifier/README.md"
PEERS = Path("/etc/vy/peers")
ADVICE = [sys.executable, "-m", "verity_vllm.pipeline.cli", "advice"]
GATE_HOST = "vy-challenger-1"

CLAIM = {
    "statement": (
        "Counts only. A report marked OK establishes, for the rehearsal's one link: on every link-window of the honest "
        "run N2's verdict of record accepted (LinkWindows.unproved_outcome, in the clear, on the stream with Selection "
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
        "N3 (content): the frames inside the counts, their bytes and their order, are not checked. N3 is retired, and "
        "the network certifier's step-indexed scan that supersedes it doesn't run here, so nothing here holds the "
        "egress grid to the Program's.",
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
        "N2's verdicts here are decided in the clear (README, 'A run that proves nothing'): "
        "LinkWindows.unproved_outcome runs LinkWindows.clear over the opening, and so reads the heads and the "
        "certifier's salts. n2_zk decides each link-window on a --zk proof of N2's registered unit instead; this "
        "report doesn't take those verdicts yet.",
        "The proofs an adopted summary carries are of n2_zk's earlier row adapter around the unchanged "
        "NetworkHeadCheck_v2, not of N2's registered unit, so LinkWindows.proved, which refuses a proof of any Program "
        "but the unit's, doesn't take them. Lean holds each proof to the program and partition the driver wrote.",
        "The certificate's link to its head (README, `CertifiedRecord.salt`): decided in the clear, each signed digest "
        "is checked against its head and the certifier's salt on the host, over the opening (CertifiedRecord.verify).",
    ],
    "source": README,
}

CLAIM_ADVICE = {
    "statement": (
        "Certified I/O only, no counts. On the advice profile (varied traffic) N2 gives no verdict: N2's advised "
        "counts (NetworkAdvisedCounts_v1), which derive each window's egress counts from its ingress head and the "
        "advice, aren't run here. A report marked OK establishes, for the rehearsal's one link: every link-window's "
        "records certified under the certifier's key and session; the certified I/O well formed (every request an id "
        "request, every egress frame an id event of a request released by its bucket, every response ending at its "
        "first EOS or at its max_tokens); the certifier's audit of both records and the salt audit accepted (the "
        "rehearsal's own check); and late and extra, and drop where it ran, each refused by that check for its own "
        "violation (late: its window's coverage fault; extra: a response longer than its max_tokens; drop: one short "
        "of it without EOS). Where an advice was given, it is these runs' (every advice request a certified one) and "
        "rebuilt from their certified I/O. The I/O tie (`io_tie`'s reference in the clear, on the negatives' "
        "stand-ins), required here, accepted the honest run, its certified ids the endpoint's sampled ids for every "
        "one of its certified requests, and refused token, whose certified ids aren't, by its content part at exactly "
        "the link-windows token changed. The advice's sections say what was checked of the advice, and the prove "
        "path's what it proved."),
    "not_covered": [
        "N2's counts: no link-window's head is held to a profile or to the advice (N2's advised counts), and nothing "
        "is proved by n2_zk, whose head check is the planned profile's; a tampered head is not refused.",
        "The tie (`io_tie`): no proof holds the certified tokens to the committed values. The tie is `io_tie`'s "
        "reference in the clear on stand-ins: the certified frames under fake salts in place of the warden's cells, and "
        "the endpoint's sampled ids from its records in place of the Commit's port. The advice's rebuild from the "
        "certified I/O here runs in the clear, on the rehearsal's published witness.",
        CLAIM["not_covered"][2],
    ],
    "zero_knowledge_gaps": [
        "The rehearsal's check reads the records, certificates and certifier's salts in the clear, and the advice is "
        "handed to the report whole: nothing on the advice profile is zero-knowledge yet.",
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
    """A run's own check (`certifier_serve.check`'s, at the rehearsal): its verdict and parts, and the codes of what it
    found: each window's audit status (``audit.status:…``) and each certified I/O violation (``io:…``)."""
    if not doc:
        return None
    io = doc.get("io")
    codes = {f"audit.{v}" for w in doc.get("rows") or doc.get("windows") or () for x in (w.get("audit") or {}).values()
             for v in x.values() if v} | {f"io:{v[1]}" for v in (io or {}).get("violations") or ()}
    return {"verdict": doc.get("verdict"), "codes": sorted(codes),
            "network": (doc.get("network_verdict") or {}).get("accepted"),
            "audit": doc.get("audit_accepted"), "salt": doc.get("salt_accepted"), "salt_again": doc.get("salt_again"),
            "certified": doc.get("certified_accepted"),
            "io": None if io is None else {k: io.get(k) for k in ("accepted", "requests", "tokens", "finish",
                                                                  "violations")}}


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
                        "shim_requests": (_read(run / "server.json") or {}).get("requests"),
                        "gate": None if gate is None else {"run": gate.get("run"), "rc": gate.get("rc"),
                                                           "ended": gate.get("ended"), "pinned": gate.get("pinned"),
                                                           **((gate.get("runs") or {}).get(s) or {})}}
    if HONEST not in found:
        raise SystemExit(f"no honest run among {sorted(found)}: the tampered head is the honest run's")
    return found


def where(runs: dict[str, dict], peer: str | None = None) -> dict:
    """Where the honest run's certifier ran: on the gate's host (the auditor's machine, through its tunnel to the
    endpoint), on the host that ran the rehearsal reaching another node's endpoint through `forward` (``peer``), or off
    the gate on the host that ran the rehearsal, which against a loopback endpoint is the endpoint's."""
    upstream = (_read(runs[HONEST]["dir"] / "deployment.json") or {}).get("upstream")
    if runs[HONEST]["gate"] is not None:
        req = _read(Path(runs[HONEST]["dir"]).parents[1] / "request.json") or {}
        gates = {s: x["gate"] for s, x in runs.items() if x.get("gate")}
        off = sorted(set(runs) - set(gates))
        at = f"{req['peer']}:{req['port']}" if req.get("peer") and req.get("port") else None
        return {"upstream": f"{at} (its 127.0.0.1, tunnelled to {upstream})" if at and upstream else at or upstream,
                "where": f"gate: the certifier on {GATE_HOST}, the auditor's host, through its gate (vy-challenger-gate "
                         f"certifier-rehearse), tunnelled to {at or 'the endpoint'}; gate runs "
                         f"{', '.join(f'{s} {g.get('run')}' for s, g in gates.items())}; pinned to main "
                         f"{', '.join(sorted({str(g.get('pinned')) for g in gates.values()}))}"
                         + (f"; {', '.join(off)} off the gate" if off else "")}
    if peer is not None:
        return {"upstream": f"{peer} (its 127.0.0.1, forwarded to {upstream})",
                "where": f"off the prover's node: better than the loopback, though not yet the auditor host (the "
                         f"certifier on {socket.gethostname()}, the endpoint on {peer.rsplit(':', 1)[0]} through an "
                         f"ssh forward)"}
    host = urllib.parse.urlsplit(upstream or "").hostname or ""
    local = host == "localhost" or host.startswith("127.") or host == "::1"
    return {"upstream": upstream, "where": "off the gate: the certifier on the host that ran the rehearsal" +
            (", the endpoint's own (loopback), so the prover's node, not an auditor host" if local else "")}


def _ssh_to(peer: str, peers: Path) -> list[str]:
    """ssh's arguments to node NAME (``peer``, NAME:PORT), from the nodes' own ssh to each other (``peers``)."""
    name = peer.rsplit(":", 1)[0]
    to = next((line.split()[1:] for line in peers.read_text().splitlines() if line.split()[:1] == [name]), None)
    if not to:
        raise SystemExit(f"--endpoint-peer {peer}: no {name} in {peers}")
    return to


def forward(peer: str, log_to: Path, peers: Path = PEERS) -> tuple[str, subprocess.Popen]:
    """An ssh forward from a free port on this host's 127.0.0.1 to NAME's 127.0.0.1:PORT (``peer``), through the
    nodes' own ssh to each other (``peers``, as `benchmarks/remote_session/run.sh` opens it), and the upstream URL it
    gives here, once it connects."""
    name, port = peer.rsplit(":", 1)
    to = _ssh_to(peer, peers)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        here = s.getsockname()[1]
    with open(log_to, "ab") as f:
        p = subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o",
                              "ServerAliveInterval=15", "-L", f"127.0.0.1:{here}:127.0.0.1:{int(port)}", *to,
                              "sleep 86400"], stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
    for _ in range(120):
        if p.poll() is not None:
            raise SystemExit(f"--endpoint-peer {peer}: ssh exited {p.returncode} ({log_to})")
        try:
            socket.create_connection(("127.0.0.1", here), timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    else:
        p.kill()
        raise SystemExit(f"--endpoint-peer {peer}: the forward never listened ({log_to})")
    log(f"endpoint: {name}'s 127.0.0.1:{port} forwarded to 127.0.0.1:{here}")
    return f"http://127.0.0.1:{here}", p


def endpoint_records(src: str, peer: str | None, dest: Path, peers: Path = PEERS) -> Path:
    """The endpoint's records (`chain.ENDPOINT`) copied into ``dest``: from ``src`` on ``peer``'s node through the
    nodes' own ssh, or from a local ``src``."""
    dest.mkdir(parents=True, exist_ok=True)
    if peer is None:
        if Path(src).resolve() != dest.resolve():
            for f in chain.ENDPOINT:
                shutil.copyfile(Path(src) / f, dest / f)
        return dest
    p = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", *_ssh_to(peer, peers),
                        f"cd {shlex.quote(src)} && tar cf - {' '.join(chain.ENDPOINT)}"], capture_output=True, check=False)
    if p.returncode:
        raise OSError(f"ssh to {peer.rsplit(':', 1)[0]} exited {p.returncode}: {p.stderr.decode()[-400:]}")
    with tarfile.open(fileobj=BytesIO(p.stdout)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def endpoint_advice(ep: Path | None, why: str | None, honest: Path, plan: dict) -> tuple[dict | None, dict]:
    """The endpoint's own advice for the honest run: `verity-vllm advice` (`ADVICE`, one advice a closed Commit
    window) over its records ``ep`` into ``ep/advice`` at the
    certifier's bucket, unless an index is there, then `chain.served_advice` on the honest run's certified I/O. Without
    the records or the advice (``why``, or the command's own failure) the advice is not these runs'."""
    T, dep = chain.window_buckets(honest), _read(honest / "deployment.json")
    io_ = (_read(honest / "io.json") or {}).get("requests")
    eos = (plan.get("varied") or {}).get("eos_token_ids")

    def none(why: str) -> tuple[None, dict]:
        return None, {"T": T, "served": None, "config": None, "bound": False, "bound_why": why, "public_fields": None,
                      "rebuilt": None, "why": None, "v0_total": 0, "commit_windows": [], "windows": []}

    if ep is None:
        return none(f"no endpoint records: {why}")
    if io_ is None or dep is None or not eos:
        return none("the honest run has no certified I/O, deployment or EOS ids (only a varied run's check writes them)")
    adv = ep / "advice"
    if not (adv / "windows.json").exists():
        p = subprocess.run([*ADVICE, str(ep), "--out", str(adv), "--tau", repr(dep["window_s"] / T)],
                           capture_output=True, text=True, check=False)
        (ep / "advice.log").write_text(p.stdout + p.stderr)
        log(f"endpoint advice: verity-vllm advice exited {p.returncode} ({ep / 'advice.log'})")
        if not (adv / "windows.json").exists():
            return none(f"verity-vllm advice made no index (exit {p.returncode}: {(p.stdout + p.stderr)[-300:].strip()})")
    try:
        return chain.served_advice(ep, adv, dep, io_, T, eos)
    except (OSError, KeyError, ValueError) as err:
        return none(f"the endpoint's advice: {err!r}")


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
           for end, c in zip(n2_zk.ENDS, x.insts)}
    if rd.lw.served is not None and not rd.lw.served_check(x.opening, [ClockSync(d, Fraction(r))
                                                                         for d, r in rd.plan["syncs"]]):
        out["egress"] = (out["egress"] or []) + ["served-counts"]
    return out


def host(rd: SimpleNamespace) -> dict:
    """`n2_zk.link_windows`'s result on its stream as an adopted `n2_zk.prove` summary records N2's verdicts (its
    ``host``), with each link-window's failed parts (`parts`)."""
    return {"windows": [{"window": x.window, "registration": x.digest, **(x.host or {}), "parts": parts(rd, x)}
                        for x in rd.windows], "network": rd.network, "issuer_refused": rd.issuer_refused}


#: why `prove` and `adopt` take no proof of `n2_zk` as it now proves
NOT_YET = ("n2_zk proves N2's registered unit, one member a round, each link-window decided by its proof; this report "
           "reads the per-end members n2_zk proved before, and doesn't read these yet (--no-prove: N2's verdicts only)")


def prove(runs: dict[str, dict], args, work: Path) -> dict:
    """`n2_zk.prove`'s result as this report reads it: refused before anything is proved (`NOT_YET`)."""
    return {"rc": 1, "why": NOT_YET, "dir": work / "n2zk", "summary": {}, "extra_proved": False}


def adopt(d: Path, runs: dict[str, dict]) -> dict:
    """`n2_zk.prove`'s output from a run of its own (its ``--out``, or `n2_zk.sh`'s ``$OUT/n2zk``), as `prove` returns
    it, so that what n2_zk's own run proved isn't proved again. Refused unless every proved member is of these runs: the
    certificate digest each of its units' public row carries (``signed``, in the verifier's ``n2.json``) is the one
    that run's certified record of that end and link-window carries (the tampered head keeps the honest certificate)."""
    summary = _read(d / "summary.json") or {}
    members = summary.get("members") or {}
    if not members:
        raise SystemExit(f"{d}: no n2_zk members (n2_zk.prove's --out, or n2_zk.sh's $OUT/n2zk)")
    if any(k in n2_zk.ROUNDS for k in members):
        raise SystemExit(f"{d}: {NOT_YET}")
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


def _round(x, places: int = 3):
    return None if x is None else round(x, places)


def _member(m: dict) -> dict:
    """One n2_zk member's proofs: held by Lean or not, what the verifier received, and its session's cost (all n
    link-windows in one session and one pair of proofs)."""
    t, rec = m.get("timing") or {}, m.get("received") or {}
    proofs = (t.get("timing") or {}).get("proof_bytes")
    return {"held": bool(m.get("accepted")) and all((m.get("verdicts") or {}).values()), "lean": m.get("verdicts"),
            "audit": m.get("audit"), "n": m.get("n"), "ands": m.get("ands"), "stage_s": m.get("stage_s"),
            "session_s": t.get("session_s"), "prove_s": _round((t.get("timing") or {}).get("prove_total_s")),
            "verify_s": m.get("lean_verify_s"), "proof_bytes": sum(proofs) if proofs else None,
            "proof_files": len(proofs) if proofs else None, "peak_gb": (t.get("process") or {}).get("max_child_rss_gb"),
            "verifier_bytes": rec.get("verifier_bytes"), "witness_found_at_verifier": rec.get("verifier_found"),
            "public_pieces_left_out": None if rec.get("public_pieces") is None else len(rec["public_pieces"]),
            "control_holds_all": (rec.get("control") or {}).get("holds_all")}


def report(runs: dict[str, dict], n2: dict[str, dict], proofs: dict, meta: dict, stages: dict | None = None) -> dict:
    """The report (`FORMAT`): per run its network verdict, per link-window N2's verdict and the proofs' with their cost,
    the negatives' refusals, the chain's stages (``stages``: `chain`'s advice, bits, prove path and the negatives
    (`negatives.py`), each when it ran) and the claim. ``proofs``: `prove`'s result, or ``{"ran": False, "why": ...}``.
    On the advice profile N2 gives no verdict (``n2`` has no windows), so the rehearsal's own check is the verdict, and
    late and extra its negatives; the I/O tie (``stages``' ``tie``) is required there (`chain.tie_gate`), and token is
    its negative, refused by the tie's own row whatever the tie gave the honest run. Drop, where it ran, is a negative
    on every profile."""
    advised = meta.get("profile") == chain.ADVICE
    s = stages or {}
    t = s.get("tie") or {}
    ties = t.get("runs") or {}
    expected = (NEGATIVES[:2] if advised else NEGATIVES) + tuple(s for s in ("drop",) if s in n2) \
        + tuple(s for s in PENDING if s in n2 and advised)
    pending = {s: w for s, w in PENDING.items() if s in n2 and not advised}
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
                          "shim_requests": (given if name in runs else runs[HONEST]).get("shim_requests"),
                          "network": {k: (h["network"] or {}).get(k) for k in ("accepted", "windows", "refused", "digest")},
                          "issuer_refused": h.get("issuer_refused"),
                          "proofs": {e: {k: v for k, v in m.items() if k != "units"} for e, m in ends.items()} or None}
        for x in h["windows"]:
            w = x["window"]
            proved = {e: _proof(next((u for u in m["units"] if u["window"] == w), None)) for e, m in ends.items()}
            rows.append({
                "run": name, "window": w, "n2": _n2_row(x), "failed_parts": x.get("parts"), "proof": proved or None,
                "head_checks": None if not proved else "pass" if all(v == "accept" for v in proved.values())
                else "malformed" if None in proved.values() or "malformed" in proved.values() else "refuse"})
    negatives = {name: negative(out_runs[name], [r for r in rows if r["run"] == name], name, ties.get(name))
                 for name in expected if name in out_runs}
    honest, honest_rows = out_runs[HONEST], [r for r in rows if r["run"] == HONEST]
    proved_ok = not proofs.get("ran") or (
        proofs.get("rc") == 0 and bool(honest["proofs"]) and all(p["held"] for p in honest["proofs"].values())
        and all(r["head_checks"] == "pass" for r in honest_rows))
    n2_ok = advised or (bool(honest["network"]["accepted"]) and bool(honest_rows)
                        and all(r["n2"] == "accept" for r in honest_rows))
    check_ok = (honest["check"] or {}).get("verdict") == "accepted" if advised or honest["check"] is not None else True
    b, p, ng = s.get("bits"), s.get("prove_path"), s.get("negatives")
    # on the advice profile the I/O tie is the only check of the served content, so it is required there
    tie_failed = chain.tie_gate(s.get("tie")) if advised else []
    # the derived bits and the residual's own check (`public_fields`) are under the serving config the job was given,
    # not one the endpoint registered, so they stay out of the OK; the advice's binding and its rebuild from the
    # certified I/O don't rest on that config's being the endpoint's
    advice_ok = b is None or (b.get("bound") is True and b.get("rebuilt") is True)
    why_not = ([] if n2_ok else ["N2 didn't accept the honest run on every link-window"]) \
        + ([] if check_ok else ["the rehearsal's own check didn't accept the honest run"]) \
        + ([] if proved_ok else ["the honest run's proofs didn't all run and hold"]) \
        + ([] if b is None or b.get("bound") is True else ["the advice isn't these runs'"]) \
        + ([] if b is None or b.get("rebuilt") is True else ["the advice wasn't rebuilt from the certified I/O"]) \
        + ([] if p is None or not p.get("bound") or p["ok"] else ["the prove path failed"]) \
        + ([] if p is None or p.get("tied") is not False else
           [f"the prove path's audited advice isn't the report's: {p['tied_why']}"]) \
        + ([] if ng is None or not ng.get("ran") or ng["ok"] else ["the negatives registry failed"]) \
        + tie_failed \
        + [f"no {s} run" for s in expected if s not in negatives] \
        + [f"{s} not refused" for s, n in negatives.items() if not n["refused"]]
    ok = not why_not
    ran = {"N2's verdicts of record": any(r["n2"] in ("accept", "refuse") for r in honest_rows),
           "the proofs (n2_zk, held by Lean)": bool(proofs.get("ran")),
           "the rehearsal's own check": honest["check"] is not None,
           "the advice, these runs' and rebuilt from their certified I/O": b is not None and advice_ok,
           "the prove path, of these runs": bool(p and p.get("bound")),
           "the negatives registry": bool(ng and ng.get("ran"))}
    # the tie is never "not run": where it didn't run, `tie_failed` says why the report isn't OK
    on = [k for k, v in ran.items() if v] + [TIE_STAGE] * bool(advised and t.get("ran"))
    waiting = {name: {"waits": w, "check": (out_runs[name]["check"] or {}).get("verdict"),
                      "n2": out_runs[name]["network"]["accepted"],
                      "ran": isinstance(out_runs[name]["shim_requests"], int) and out_runs[name]["shim_requests"] != 0,
                      "why": t.get("why")}
               for name, w in pending.items()}
    return {"format": FORMAT, **meta, "claim": CLAIM_ADVICE if advised else CLAIM, "ok": ok, "why_not": why_not,
            "ran": on, "not_run": [k for k, v in ran.items() if not v],
            "runs": out_runs,
            "link_windows": rows, "negatives": negatives, "pending": waiting, "chain": stages,
            "proofs": {"ran": bool(proofs.get("ran")), "adopted": proofs.get("adopted"), "why": proofs.get("why"),
                       "rc": proofs.get("rc"), "extra_proved": proofs.get("extra_proved"), "n2_zk_ok": summary.get("ok"),
                       "stagers": summary.get("stagers"), "driver_peak_gb": proofs.get("driver_peak_gb"),
                       "alone": alone(summary, proofs.get("dir"))}}


def alone(summary: dict, d: Path | None) -> dict | None:
    """n2_zk's cost pass, per end: one honest link-window proved by itself (a session whose verifier draws one unit of
    the n staged), with its prover and Lean seconds, its two proofs' bytes (read from ``d``, n2_zk's output, when
    there) and the peak, as the report's sessions."""
    out = {}
    for end, rows in (summary.get("cost") or {}).items():
        x = next((r for r in rows if r.get("units") == 1 and "lean" in r), None)
        if x is None:
            continue
        files = sorted((Path(d) / "cost" / end / f"{x['arm']}-k{x['k']}-r{x['rep']}").glob("sessions/*/*.rep*.bin")) \
            if d else []
        out[end] = {"held": bool(x.get("accepted")) and x["lean"].get("accepted") is True, "staged": x.get("staged"),
                    "prove_s": _round(x.get("prove_total_s")), "verify_s": x["lean"].get("verify_s"),
                    "proof_bytes": sum(f.stat().st_size for f in files) or None, "proof_files": len(files) or None,
                    "peak_gb": x.get("max_child_rss_gb")}
    return out or None


def negative(run: dict, rows: list[dict], name: str | None = None, tie: dict | None = None) -> dict:
    """Whether negative ``name`` was refused, and by what: N2's network verdict and its link-windows' verdicts of
    record (a link-window the certifier certified nothing of, `Issuer.refused`, has no verdict of its own, and the
    network verdict refuses it unregistered), the rehearsal's own check, and where proved, its head checks' proved
    verdicts (each proof held by Lean).
    Refused only when every one of them that ran refuses it. Where N2 gives no verdict (the advice profile), the check
    refuses it only by its own violation (`OWN_CODE`): a refusal for another reason is not this negative's. A pending
    negative (`PENDING`) is refused by the I/O tie alone (``tie``, its row of `chain.tie`) when its verdict is the one it
    owes; on another negative the tie's owed refusal is listed beside the rest. A run whose shim received no request, or
    that has no ``server.json`` to say, didn't run."""
    by, check = [], run.get("check")
    if name in PENDING:
        ran = isinstance(run.get("shim_requests"), int) and run["shim_requests"] != 0
        refused = bool(tie and tie["ok"]) and ran
        return {"refused": refused, "ran": ran, "by": [_tie_by(tie)] if refused else [], "own_code": None,
                "check_codes": None if check is None else check.get("codes"), "n2_refused_windows": [],
                "proved_refused_windows": [], "failed_parts": {}, "proved": False, "proofs_held": None, "tie": tie}
    n2 = sorted({r["window"] for r in rows if r["n2"] == "refuse"} | set(run["network"].get("refused") or ()))
    decided = run["network"]["accepted"] is not None
    own = None if decided else OWN_CODE.get(name)
    check_refused = check is not None and check["verdict"] == "refused"
    by_own = check_refused and own is not None and own in (check.get("codes") or ())
    issuer = sorted({w for *_, w in run.get("issuer_refused") or ()})
    if issuer:
        by.append(f"the certifier (Issuer.refused; link-windows {issuer})")
    if run["network"]["accepted"] is False and n2:
        by.append(f"N2 (network verdict; link-windows {n2})")
    if check_refused and (decided or by_own):
        by.append("the rehearsal's own check" + (f" ({own})" if own else ""))
    proved = [r["window"] for r in rows if r["head_checks"] == "refuse"]
    held = all(p["held"] for p in (run["proofs"] or {}).values())
    if proved and held:
        by.append(f"the proved head checks (link-windows {proved})")
    if tie and tie["ok"]:
        by.append(_tie_by(tie))
    ran = isinstance(run.get("shim_requests"), int) and run["shim_requests"] != 0
    refused = ((run["network"]["accepted"] is False and bool(n2) if decided else by_own)
               and (check is None or check_refused) and (not run["proofs"] or (held and bool(proved))) and ran)
    return {"refused": refused, "ran": ran, "by": by if ran else [], "own_code": own,
            "check_codes": None if check is None else check.get("codes"), "n2_refused_windows": n2,
            "proved_refused_windows": proved,
            "failed_parts": {r["window"]: r["failed_parts"] for r in rows if r["n2"] == "refuse"},
            "proved": bool(run["proofs"]), "proofs_held": held if run["proofs"] else None, "tie": tie}


def _tie_by(tie: dict) -> str:
    """The I/O tie as what refused a negative: its codes, the link-windows they refused, and token's changed ids'."""
    at = f" at link-windows {tie['windows']}" if tie["windows"] else ""
    lost = f", {tie['unplaced']} committed outputs no certified cell carries" if tie.get("unplaced") else ""
    changed = (f", the link-windows its changed ids fall in" if tie.get("changed") == tie["windows"] and tie["windows"]
               else "")
    return f"the I/O tie ({', '.join(tie['codes'])}{at}{changed}{lost})"


def _cell(x) -> str:
    return "—" if x is None else str(x)


def markdown(r: dict) -> str:
    """The report as Markdown: the verdicts, one row per link-window, the negatives and the claim."""
    p, v = r.get("proofs") or {}, r.get("varied")
    traffic = (f"varied traffic: {_cell(v.get('requests'))} requests, max_tokens drawn per request up to "
               f"{_cell(v.get('max_tokens'))}, prompts of {'–'.join(map(str, v.get('prompt_tokens') or ())) or '—'} "
               f"tokens, EOS {_cell(v.get('eos_token_ids'))}" if v else
               f"{_cell(r.get('tokens'))} tokens a response, {_cell(r.get('requests'))} requests")
    on = r.get("ran") or []
    lines = [f"# Network accounting end to end: {'OK' if r['ok'] else 'NOT OK'} on "
             f"{', '.join(on[:-1]) + ' and ' + on[-1] if len(on) > 1 else on[0] if on else 'nothing that ran'}"
             + (f" (not run: {', '.join(r['not_run'])})" if r.get("not_run") else ""), "",
             *([f"Not OK: {'; '.join(r['why_not'])}.", ""] if r.get("why_not") else []),
             f"Source {', '.join(r['from'])}; model {_cell(r.get('model'))} (as the plan names it), {traffic}; endpoint "
             f"{_cell(r.get('upstream'))}, {_cell(r.get('where'))}; N2's profile {_cell(r.get('profile'))}; "
             f"{_cell(r.get('windows'))} link-windows planned; commit {_cell(r.get('commit'))}; "
             f"proofs {'run' if p.get('ran') else 'not run'}.", "",
             "## Runs", "",
             "| run | rehearsal's check | gate | N2 network verdict | refused windows | proofs held (Lean) |",
             "|---|---|---|---|---|---|"]
    for name, x in r["runs"].items():
        c, g = x.get("check"), x.get("gate")
        held, accepted = x.get("proofs"), x["network"]["accepted"]
        io = (c or {}).get("io")
        lines.append(f"| {name} | {_cell(c and c['verdict'])}{f' (I/O {io['accepted']})' if io else ''} | "
                     f"{_cell(g and g.get('verdict'))} | "
                     f"{'—' if accepted is None else 'accept' if accepted else 'refuse'} | "
                     f"{_cell(x['network']['refused'] or None)} | "
                     f"{', '.join(f'{e}: {v['held']}' for e, v in held.items()) if held else '—'} |")
    lines += ["", "## Link-windows", ""]
    if not r["link_windows"]:
        lines.append("No N2 verdict on the advice profile (N2's advised counts aren't run here): the rehearsal's own "
                     "check is the verdict.")
    else:
        lines += ["N2: the verdict of record (host). Proof: each end's head check proved --zk and held by Lean, accept "
                  "or refuse as its opened verdict row says. Failed parts: the head check's parts that fail, evaluated in "
                  "the clear on the rehearsal's published witness (a diagnostic, not the verdict). Each run's end proves "
                  "all its link-windows in one session, so a link-window has no proving time or proof of its own: the "
                  "sessions' are under Proving sessions.",
                  "", "| run | window | N2 | proof egress | proof ingress | head checks | failed parts |",
                  "|---|---|---|---|---|---|---|"]
    for x in r["link_windows"]:
        pr = x["proof"] or {}
        failed = "; ".join(f"{e} {'+'.join(f)}" for e, f in (x["failed_parts"] or {}).items() if f) or None
        lines.append(f"| {x['run']} | {x['window']} | {_cell(x['n2'])} | {_cell(pr.get('egress'))} | "
                     f"{_cell(pr.get('ingress'))} | {_cell(x['head_checks'])} | {_cell(failed)} |")
    lines += ["", "## Negatives", ""]
    for name, n in r["negatives"].items():
        other = (f"; the rehearsal's own check refused it for {', '.join(n['check_codes'] or ()) or 'no code'}, not "
                 f"its own violation {n['own_code']}" if not n["refused"] and n.get("own_code") and n.get("check_codes")
                 is not None and r["runs"][name]["check"]["verdict"] == "refused" else "")
        t = n.get("tie")
        if name in PENDING and t is None:
            other += (f"; the I/O tie, which alone refuses it, didn't run "
                      f"({((r.get('chain') or {}).get('tie') or {}).get('why') or 'no tie stage'})")
        elif name in PENDING and t["verdict"] not in ("accepted", "refused"):
            other += f"; the I/O tie, which alone refuses it, gave it no verdict ({t['why']})"
        elif t and not t["ok"] and name in PENDING:
            other += (f"; the I/O tie {t['verdict']}" + (f" ({', '.join(t['codes'])} at link-windows {t['windows']})"
                                                          if t["codes"] else "")
                      + (f", its changed ids in link-windows {t['changed']}" if t.get("changed") is not None else
                         ", its shim logged no changed id"))
        lines.append(f"- **{name}**: " + (
            f"{'refused' if n['refused'] else 'NOT refused'} by {'; '.join(n['by']) or 'nothing'}"
            f"{'' if n['proved'] else ' (not proved)'}{other}" if n.get("ran", True)
            else "NOT refused: not run, its shim received no request (or no server.json says it did), so its tamper "
                 "never ran"))
    for name, n in (r.get("pending") or {}).items():
        n2 = n["n2"]
        lines.append(f"- **{name}** (the registry's, pending {n['waits']}, which alone refuses it here): " + (
            f"the rehearsal's check {_cell(n['check'])}, N2 {'—' if n2 is None else 'accept' if n2 else 'refuse'}"
            if n["ran"] else "not run: its shim received no request, so its tamper never ran")
            + (f"; the I/O tie not run ({n['why']})" if n.get("why") else ""))
    scanned = [(name, e, m) for name, x in r["runs"].items() for e, m in (x.get("proofs") or {}).items()]
    if scanned:
        lines += ["", "## Proving sessions", "",
                  "One session per run and end: its link-windows are one statement, proved --zk in one pair of proofs "
                  "and verified by Lean at once. Proof bytes: the pair's. Peak: the largest process n2_zk had run by "
                  "the session's end (RUSAGE_CHILDREN only grows, and n2_zk runs the honest sessions first, so the "
                  "tampered and extra sessions' include theirs).",
                  "", "| run | end | link-windows | held (Lean) | prove s | Lean verify s | proof bytes | peak GB |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {name} | {e} | {_cell(m['n'])} | {m['held']} | {_cell(m['prove_s'])} | {_cell(m['verify_s'])} | "
                  f"{_cell(m['proof_bytes'])} | {_cell(m['peak_gb'])} |" for name, e, m in scanned]
    one = p.get("alone")
    if one:
        lines += ["", "## One link-window proved alone", "",
                  "n2_zk's cost pass: a session on the honest run's statement whose verifier draws one link-window, "
                  "per end.", "", "| end | link-windows staged | held (Lean) | prove s | Lean verify s | proof bytes "
                  "| peak GB |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {e} | {_cell(m['staged'])} | {m['held']} | {_cell(m['prove_s'])} | {_cell(m['verify_s'])} | "
                  f"{_cell(m['proof_bytes'])} | {_cell(m['peak_gb'])} |" for e, m in one.items()]
    if p.get("driver_peak_gb") is not None:
        lines += ["", f"The driver's own peak (lowering): {p['driver_peak_gb']} GB."]
    if scanned:
        lines += ["", "## What the verifier received", "",
                  "Each member's verifier directory, scanned for every piece of every head, head salt, sibling and "
                  "certifier salt but the pieces the public circuit holds (n2_zk's `public_pieces`); the prover's "
                  "instance file is the control.", ""]
        lines += [f"- {name} {e}: {_cell(m['verifier_bytes'])} bytes; witness pieces found: "
                  f"{m['witness_found_at_verifier'] or 'none'}; left out as public: {_cell(m['public_pieces_left_out'])}; "
                  f"control holds all: {_cell(m['control_holds_all'])}" for name, e, m in scanned]
    if p.get("adopted"):
        lines += ["", f"Proofs: adopted from {p['adopted']}, n2_zk's output of its own run, each proved unit's "
                      "certificate digest matched to these runs' certified records."]
    if p.get("why") or not p.get("ran"):
        lines += ["", f"Proofs: {p.get('why') or 'not run'}"]
    lines += chain.markdown(r.get("chain") or {})
    if r.get("stubbed"):
        lines += ["", "## What's still stubbed", ""] + [f"- {x}" for x in r["stubbed"]]
    lines += ["", "## Claim", "", r["claim"]["statement"], "", "Not covered:", ""]
    lines += [f"- {x}" for x in r["claim"]["not_covered"]]
    lines += ["", f"Zero-knowledge gaps (`{r['claim']['source']}`):", ""]
    lines += [f"- {x}" for x in r["claim"]["zero_knowledge_gaps"]]
    return "\n".join(lines) + "\n"


def stages_of(runs: dict[str, dict], plan: dict, args, work: Path, clock) -> dict:
    """The chain's stages (`chain`): the advice and its bits per link-window when ``--advice`` gives it (T, the
    certified I/O and the windows from the honest run), or the endpoint's own when ``--endpoint-run`` names its records
    (`endpoint_advice`), the prove path when ``--prove-path`` does, and the negatives (`negatives.py`) on the runs
    unless ``--no-negatives`` (not on the advice profile: its rows take N2's verdict), with the I/O tie's verdict when
    ``--ids`` gives each request's sampled ids. On the advice profile, the I/O tie on each run (`chain.tie`), its
    sampled ids ``--ids``' or the endpoint's records', which the report requires there (`chain.tie_gate`)."""
    honest, out, adv = runs[HONEST]["dir"], {}, None
    if getattr(args, "endpoint_run", None):
        out["advice"], out["bits"] = endpoint_advice(getattr(args, "endpoint_dir", None),
                                                     getattr(args, "endpoint_why", None), honest, plan)
        log(f"endpoint advice: bound {out['bits']['bound']}, rebuilt {out['bits']['rebuilt']}"
            + (f" ({out['bits']['bound_why']})" if out["bits"]["bound_why"] else ""))
    elif args.advice:
        adv, out["advice"] = chain.advice(Path(args.advice))
        config = _read(Path(args.serving_config)) if args.serving_config else None
        if args.serving_config and not isinstance(config, dict):
            raise SystemExit(f"--serving-config {args.serving_config}: not a JSON object")
        out["bits"] = chain.bits(adv, chain.window_buckets(honest), plan.get("windows") or 0,
                                 config and config.get("config", config), (_read(honest / "io.json") or {}).get("requests"))
    if args.prove_path:
        srcs = [Path(s) for s in args.prove_path]
        given = getattr(args, "prove_path_registered", None)
        registered = _read(Path(given)) if given else None
        if given and not isinstance(registered, dict):
            raise SystemExit(f"--prove-path-registered {given}: not a JSON object")
        b = out.get("bits") or {}
        out["prove_path"] = chain.prove_path([_unpack(s, work) if s.is_file() else s for s in srcs], out.get("advice"),
                                             b.get("bound"), b, getattr(args, "endpoint_dir", None), registered) \
            | {"dir": ", ".join(map(str, srcs))}
        log(f"prove path: adopted from {', '.join(map(str, srcs))}, ok {out['prove_path']['ok']}, bound "
            f"{out['prove_path']['bound']}")
    if (plan.get("occupancy") or {}).get("kind") == chain.ADVICE:
        out["negatives"] = {"ran": False, "why": "its rows take N2's verdict, which the advice profile has none of "
                                                 "without N2's advised counts"}
        out["tie"] = chain.tie({s: r["dir"] for s, r in runs.items()}, getattr(args, "endpoint_dir", None),
                               Path(args.ids) if args.ids else None, getattr(args, "endpoint_why", None))
        log(f"I/O tie: {', '.join(f'{s} {x['verdict']} (ok {x['ok']})' for s, x in out['tie']['runs'].items())}"
            if out["tie"]["ran"] else f"I/O tie: didn't run, which the advice profile requires ({out['tie']['why']})")
    elif not args.no_negatives:
        out["negatives"] = chain.negatives({s: r["dir"] for s, r in runs.items()}, work / "chain", adv, clock,
                                           ids=Path(args.ids) if args.ids else None)
        log(f"negatives (the registry): ok {out['negatives']['ok']}")
    return out


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
    tunnel = None
    try:
        if getattr(args, "endpoint_peer", None):
            args.upstream, tunnel = forward(args.endpoint_peer, out / "forward.log")
        sources = [Path(s) for s in args.sources] if args.sources else [rehearse(args, rest, out / "rehearsal")]
        if tunnel is not None:
            tunnel.kill()
        if getattr(args, "endpoint_run", None):
            try:
                args.endpoint_dir = endpoint_records(args.endpoint_run, getattr(args, "endpoint_peer", None),
                                                     out / "endpoint")
                log(f"endpoint records: {args.endpoint_run} copied to {args.endpoint_dir}")
            except (OSError, SystemExit) as err:
                args.endpoint_dir, args.endpoint_why = None, f"{args.endpoint_run} not copied: {err}"
        runs = collect(sources, work)
        plan = _read(runs[HONEST]["dir"] / "plan.json") or {}
        profile = (plan.get("occupancy") or {}).get("kind")
        advised = profile == chain.ADVICE
        log(f"runs: {', '.join(f'{s} {r['dir']}' for s, r in runs.items())}; N2's profile {profile}")
        if advised:
            proofs = {"ran": False, "why": "not run: on the advice profile the counts are N2's advised counts "
                                           "(NetworkAdvisedCounts_v1), which this report doesn't run, and n2_zk "
                                           "proves the planned profile's head check"}
        elif profile != "planned":
            proofs = {"ran": False, "why": f"not run: the {profile} profile's count check is NetworkServedCounts_v1, "
                                           "which n2_zk doesn't prove"}
        elif args.no_prove:
            proofs = {"ran": False, "why": "not run (--no-prove)"}
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
                n2[name] = {"windows": [], "network": None} if advised else \
                    host(n2_zk.link_windows(r["dir"], clock=clock))
        if TAMPERED not in n2 and not advised:
            n2[TAMPERED] = host(n2_zk.link_windows(runs[HONEST]["dir"], edit=n2_zk.one_count, clock=clock))
        served = plan.get("served") or plan.get("varied") or {}
        meta = {"from": [str(s) for s in sources], "commit": _commit(), "model": plan.get("live_model"),
                "tokens": served.get("tokens", served.get("max_tokens")), "requests": served.get("requests"),
                **where(runs, getattr(args, "endpoint_peer", None)), "profile": profile, "windows": plan.get("windows"), "served": plan.get("served"),
                "varied": plan.get("varied")}
        stages = stages_of(runs, plan, args, work, clock)
        r = report(runs, n2, proofs, meta, stages)
        r["stubbed"] = chain.stubs(r, hasattr(n2_zk.cs, "TRAFFIC"))
        (out / "report.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
        (out / "report.md").write_text(markdown(r))
        log(f"report {out / 'report.md'}: {'ok' if r['ok'] else 'NOT ok'}")
        return r
    finally:
        if tunnel is not None:
            tunnel.kill()
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="e2e.py", description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="sources", nargs="+", metavar="SRC",
                     help="a rehearsal's directory, a gate run's (vy-challenger-gate fetch), a run's, or a tarball")
    src.add_argument("--upstream", help="rehearse first against this endpoint, http(s)://HOST[:PORT]")
    src.add_argument("--vllm", metavar="MODEL", help="rehearse first, starting vLLM for MODEL on this host's GPU")
    src.add_argument("--endpoint-peer", metavar="NAME:PORT", help="rehearse first against node NAME's endpoint on its "
                                                                  "127.0.0.1:PORT, through an ssh forward (/etc/vy/peers)")
    ap.add_argument("--out", default=os.environ.get("OUT"), help="the report's directory (default $OUT)")
    ap.add_argument("--work", help="scratch, kept (default: a temporary directory, removed)")
    ap.add_argument("--bin", help="C-Flock's flock-circuit (run.sh builds it)")
    ap.add_argument("--lean", help="the Lean flock-verify (run.sh builds it)")
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    how = ap.add_mutually_exclusive_group()
    how.add_argument("--no-prove", action="store_true", help="N2's verdicts and the head checks' references only")
    how.add_argument("--proofs", metavar="DIR", help="adopt n2_zk.prove's output of these runs (n2_zk.sh's $OUT/n2zk, or a "
                                                     "tarball of it) instead of proving")
    ap.add_argument("--advice", metavar="FILE", help="the advice: I1 v0's JSON or encode bytes, or a prove-path "
                                                     "stand-in's (chain.advice); its bits per link-window")
    ap.add_argument("--serving-config", metavar="FILE", help="the serving config the derived bits take (derived.Config's "
                                                             "fields, or a step trace's config)")
    ap.add_argument("--endpoint-run", metavar="DIR", help="the endpoint's records (a verity-vllm serve-openai run's "
                                                          "out/serve; on --endpoint-peer's node, copied after the "
                                                          "rehearsal): its own advice for the honest run's Commit "
                                                          "windows, by verity-vllm advice (endpoint_advice)")
    ap.add_argument("--prove-path", metavar="DIR", action="append",
                    help="adopt prove_path.py audit's output (its run.sh's $OUT, or a tarball of it); again for another "
                         "(each --serve-run audit one of the served run's Commit windows)")
    ap.add_argument("--prove-path-registered", metavar="FILE",
                    help="who registered each --serve-run audit's window advice, by window: {K: {by, registration, "
                         "record, first_draw, ref}}, registration the verity-vllm advice directory it was registered in "
                         "when not the report's own, record the serving run's own run directory, first_draw ISO 8601 "
                         "UTC; when is read off the record's out/advice_windows.jsonl stamp, else the registration's "
                         "file time: at serve time if before its draw and no later than the next window's close (the "
                         "session's end, per the record, for its last window)")
    ap.add_argument("--ids", metavar="FILE", help="each request's sampled ids by its advice id (the endpoint's "
                                                  "requests.jsonl: user, output_ids), for the I/O tie's verdict on the "
                                                  "negatives it reads, and on the advice profile on every run in place "
                                                  "of --endpoint-run's records")
    ap.add_argument("--no-negatives", action="store_true", help="skip the negatives registry (negatives.suite)")
    args, rest = ap.parse_known_args(argv)
    if rest and args.sources:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if not args.out:
        ap.error("--out, or $OUT")
    if args.endpoint_run and (args.advice or args.serving_config):
        ap.error("--endpoint-run takes the endpoint's own advice and reads its config off its records: no --advice or "
                 "--serving-config beside it")
    if not (args.no_prove or args.proofs) and not (args.bin and args.lean):
        ap.error("proving needs --bin and --lean (run.sh builds them), --proofs DIR, or --no-prove")
    r = run(args, rest)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
