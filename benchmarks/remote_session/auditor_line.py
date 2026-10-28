"""The memory line on the auditor's clock, the analysis of record for a gate run: each accepted paced session's line
(time-held and link-charged, each read held over time and at the passes, exact) with its groups' times read from what
the auditor's host signed rather than from the verifier's stamps.

**Who computes it** (`line_by`, `LINE_BY`). The Lean verifier (`remote-audit-verifier`'s `line_record` request,
`RemoteAudit.Line.lineOfRecordOn` at the record's `groups_per_coin` g, with its W from `RemoteAudit.Pooled` over the
record's draws) computes each session's line off its record from
the inputs `verifier.recorded_inputs` reads, and `remote_bandwidth.line` computes it beside it from the same JSON
(`line.line_of_record`):
the two are compared exactly, every rational as text in lowest terms and every integer, and a session where they
differ refuses the run's line, naming both values (`Disagree`). Floats appear only as the exact shares correctly
rounded for display. `--line-by python` computes it in `line.py` alone, a reading that is never of record: the
`memory-line-of-record` label needs `line_by` lean.

Group g's coin σ_g left the challenger at its release log's `released_at` for challenge coin g, and window g's last
root reached it at the first `received_at` of a request naming it by place (`auditor_seen`). Openings coin g's request
names it (`audit.Session.close` asks for the coin after the window's last root), and a paced window is drawn at its
k-th root (its boundary leaves before it), so that request leaves as the root arrives; for k = 1, σ_(g+2)'s request names the same root too. At
one window in flight (`audit.Pace.in_flight` 1) σ_(g+1)'s request names it and openings coin g's leaves once σ_(g+1)
is back, so this reading, which doesn't take f, ends L_g about a round trip to the challenger past s_(g+1). Drawn
G > 1 windows a coin (`groups_per_coin`), openings coin c's request names only the last root of draw c's last window,
and another window's A_g is the first request that leaves after its last root, σ_(g+2)'s at the earliest, which above
k = 1 names a root of window g + 1 and so ends L_g later than the window did. Both
times are the challenger's monotonic clock in its release log, whose lines its close signs (`challenger.session_of`),
and whose close, record and refused hellos the auditor's ledger sealed at the session's end (`remote.trusted_session`).
So L_g = max(A_g, s_(g+1)) − s_g on the challenger's clock holds group g's reads: they come after its coin, whose seed
they take, and before its last root, which chains them.

Beside the times the line reads the Lean verifier's record of the session (`verifier.lean_record`, `verifier.json`'s
`lean_record`): each group's passes, its windows' receipts and draws, whose pooled bound (`remote_bandwidth.pooled`)
is its wrong receipts, and its registered blocks, the registration's; and the verifier's verdict, and each group's
held bytes by kind (`ledger.memory`), which the record doesn't hold (they need each pass's plan and the store's
regions). The verifier runs on the audit's machine, not
the auditor's host, so these rest on its verdict, which the session's transcript (`verifier.py --transcript`) lets the
auditor replay offline (the remote audit's `lean/tests/replay.py transcript`). The verifier's stamps are read only to
cross-check the challenger's: each coin interval on both clocks, their rates, and each group's L on both.

The ledger check: its longest head verified under the pinned key (`ledger_head`, `verify_ledger`, `verify_periods`);
the period's lane, cap and circuit, and that it closed; each session reserved and opened by the lane for the peer under
the pin at epoch k, opened at its window root R_k, ended closed with no refused hello, inside the period; σ_0 asked
after R_k, and each coin released after its root's receipt. The peer is the verifier machine's name at the challenger
host, as the run's own record gives it (`peer_of`): the node's own name for a verifier on a node, the verifier host's
for a host run. `--peer` overrides it, and a gate run whose record names none is refused. A session served by `--challenger serve`
(the verifier's machine) has no ledger: its log is read under the key its run wrote (`session_of`) and its line is not
the auditor's.

**Credit.** The line of record (`line`) counts a session only when all five hold, each reported per session
(`credit`), and gives any other session zero credit with the reasons:

- *credited*: its window reads through `challenger.credited_session` (its pin first, as *pinned* reads it) with the
  descriptor of the verifier that ran it, `verity/remote-bandwidth/v0 verifier tree <tree>`, `<tree>` being
  `git rev-parse COMMIT:verity/core/protocols/remote_bandwidth` of the run's recorded commit (`--commit`), so it counts
  only toward a period that committed that verifier; and that commit is on the auditor's first-parent list
  of main (`--main`), as the challenger host's pin must be, so the verdict is that of a tree main's check ran on. A host
  run's request.json beside OUT records the commit its verifier ran at (`pinned`), and a `--commit` other than it is
  refused;
- *complete*: its window's end, and its period's close, are complete: the challenger released every coin it committed
  to, in every window of the period;
- *accepted*: the Lean's verdict (the record's `verdict` lean or both, no disagreement) accepted it, since a refused
  session can still close complete;
- *pinned*: its window reads through `challenger.pinned_session` with the challenger host's pins log (`--pins`, its
  lines as the gate served them) and the auditor's own first-parent list of main (`--main FILE`, `git rev-list
  --first-parent origin/main`): its open sealed the log as it stood, one of the sealed lines records the open's
  `pinned`, and that commit is on main's first-parent line, a tree main's check ran on. With no
  log or no copy of main, no session is pinned or credited;
- *bound*: the registered store is the model's (`bound_check`, the run's, so every session's): the verifier bound it
  (`verifier.py --model`, verifier.json's `binding`), refusing any hello whose payload root isn't that of the store the
  pinned checkpoint lays out (`remote_bandwidth.store`, in the tree *credited* names, so the period's circuit commits to
  the code that decided it), the Lean verifier deciding it (`hold`, the binding's `lean`), the binding's root is the
  registration's, and its model, name and
  checkpoint_sha512, is the one the audit's period signs (its open's `model`), or with no period's, `--model-pin`'s.
  With neither, no session is bound.

At each T the line is the least share among the credited sessions there, and zero when none is, with every session's
reasons; beside each share of weights held is the same session's share of filler held (the rest of HBM and the host
tier, `line`'s `filler`), which the line doesn't count as weights. Each form charges the node's rates (`RATES`) and
their bursts (`rates_at`; `--bursts none` for the reading of gate runs up to 7), and deducts the chips' S_chip at no
less than the auditor's (`verifier.recorded_chip_bytes`). What every accepted paced session
measured, credited or not, is `measured`. `--table` writes each session's five checks and the line as Markdown.
Each session's `sketch` is its δs at the outside-compute factor F its verifier charged, Q = F·C (`verifier.sketch_at`),
and `sketch_outside` every F the run's sessions were charged at.

**Error.** Each accepted paced session's line fails, crediting more than the node held, with probability at most its
`error`'s total (`session_error`): the memory line's δ as the ledger computed it, read from the record, with its six
parts beside it: the windows' profile δ, AuditRO's 2^-slack a pass (the slack its ledger was charged at: 67 bits, and
41 in a record that doesn't say, where it dominates: 2^-29.95 over a 60 s session's 2118 passes), the coin
commitment's hiding terms n·2^-191, the receipts' binding term at the session's bounds, δs, the sketch's, and the
draw's rejection Σ_t k_t/2^64, k_t the session's `samples` (the hiding, the binding and the rejection added to the
total of a record whose ledger didn't charge them), plus the windows' PRF advantages by name (`NAMED_TERMS`, claim
prf/sha-512), which have no number and are in no total. The line at each T, and the line of record over every T
(`credit`'s `error`), fail with at most the sum of their credited sessions' totals: the least share is read off
whichever session the run made least, so the union bound holds where a maximum wouldn't.

**Conditional on** (`conditional_on`, `CONDITIONAL_ON`). The line's statement (`LineHeld`) takes `NoOutsideCompute`,
statement C in full, as a premise that no record of the run discharges, so a line says it is conditional on it, until
certifiers cover the gate node's boundary. A relay prover, a thin node forwarding each coin to a helper that holds
the store and computes the receipts, breaks that premise alone. The error's binding term takes the gap `binding-split`
as closed (split_t = 0), which the deployed protocol doesn't yet ensure, so every line is conditional on that too,
grounded or not. The `memory-line-of-record` label names what the file names (`line_label`: `lean conditional on
NoOutsideCompute, binding-split`), and the evidence store's labeller refuses one that doesn't.

**The lockdown** (`--lockdown FILE`, `read_lockdown`, `lockdown_check`). Infra's record of the provider's lockdown of
the gate node: the security group's allow list, each interval it was applied, with infra's records, and whether the
node mounts a shared filesystem (null while infra's answer is pending). A gate session whose span as the challenger
signed it on the ledger (its open's `started_utc` to its end's `ended_utc`) lies strictly inside one applied interval
is grounded by it (`no_relay`): its `NoOutsideCompute` rests on `ProviderLockdown`, under `trusted-host`,
which names what the lockdown doesn't close. A line whose credited sessions are all grounded (every gate session's,
with none credited) is conditional on `ProviderLockdown` in its place; any other is conditional on
`NoOutsideCompute`, as without the file. It changes no credit.

**From the transcript** (`line_of`). Where the run's out/ keeps a session's transcript (`transcripts/`, a host run
fetched with `--transcripts`), the Lean makes the session's record itself, `Shared.run` at the session's
`groups_per_coin` on the transcript (`Session.run` at one; `Line.lineOfRun`, `LineHeld`'s `lineOf`), and refuses a kept
record that isn't it; elsewhere it reads the kept
record. The verifier's clock's reading holds the times seen to the record's; the challenger's clock's doesn't.

**The boundary** (`--boundary FILE --boundary-pin SHA512`, off by default; `boundary_check`). With it, a sixth check
refuses a session's credit when the network certifiers' records of the gate node's boundary links over its credited
span, from σ_0's release to its last root's arrival on the challenger's clock, show ingress other than the verifier's
messages. It reads the certifier's `CertifiedRecord` of each boundary link's ingress window covering the span
(`certifier.certificate`: the head's `counts` and `root`), each held to one record with a complete status, signed under
the certifier's key in its session (`certifier.audit.Audit.check_ingress_link`), and compares its real frames with the
verifier's own messages framed as the link frames them: their count and their root (`certificate.frames_root`), which
must be equal. FILE names the links, the certifier's key (whose SHA-512 is `--boundary-pin`), its session and ingress
grid, each window's span on the challenger's clock, the verifier's frames and the records (`read_boundary`). It reads the
heads in the clear, a prototype; that the links are all of the node's boundary is `certifier-coverage`, which it doesn't
check, so the line stays conditional on `NoOutsideCompute` with it on.

    python auditor_line.py OUT --host-bytes B [--pin SHA512 --period P --commit SHA [--pins LOG --main FILE]]
        [--ledger HEAD ...] [--peer NAME] [--model-pin FILE] [--disk-bytes-per-s B/S] [--link-bytes-per-s B/S]
        [--bursts stated|none] [--line-by lean|python] [--lean-exe PATH] [--table FILE]
        [--boundary FILE --boundary-pin SHA512] [--lockdown FILE]
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import statistics
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import challengers as CH  # noqa: E402
import lean_verifier as lv  # noqa: E402
import verifier as V  # noqa: E402
import wire  # noqa: E402
from verity.core.protocols.certifier import audit as caudit  # noqa: E402
from verity.core.protocols.certifier import certificate as cert  # noqa: E402
from verity.core.protocols.certifier.grid import GridParams, Reason, Status  # noqa: E402
from verity.core.protocols.remote_bandwidth import audit, coins, ledger, line, pooled, receipts, timing  # noqa: E402
from verity.core.protocols.remote_bandwidth import store as rstore  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as Ch  # noqa: E402
from verity.core.protocols.sampled_proofs import remote  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402

RATES = {"disk_bytes_per_s": 4.4e9, "link_bytes_per_s": 5e10}
"""The node's ceilings a gate run's line is read at, bytes a second (`line.Rates`): the node's NICs into its memory,
which Identity leaves in, charged in both forms as `ledger.Node`'s disks are, and its boundary links beyond them,
link-charged (54.4e9 in all)."""
BURSTS = ("stated", "none")
"""What the line charges beyond the rates (`rates_at`): `ledger.Node.stated`'s bursts, the line of record's
(`verifier.ingress_bursts`), or none, the reading of gate runs up to 7, whose recorded lines it reproduces at the S_chip
they were recorded at (`test_auditor_line_lean`; a burst of 0 states the rate down to one nanosecond)."""
LANE = "memory-accounting"
VERIFIER_TREE = "verity/core/protocols/remote_bandwidth"
LEAN_VERDICTS = ("lean", "both")
CHECKS = ("credited", "complete", "accepted", "pinned", "bound")
"""What a session's credit needs, each reported per session in this order."""
SHARES = ("time", "snapshot", "filler_time", "filler_snapshot")
"""A session's shares of the node in a form (`shares`): weights held, over time and at the passes, and filler held in
the same two readings."""
LINE_BY = ("lean", "python")
"""Who computes the line (`line_of_record`): the Lean verifier, held exactly to `line.py` on the same inputs, which is
of record, or `line.py` alone."""
NAMED_TERMS = {"prf_advantage": {"claim": ledger.PRF_CLAIM, "term": "Σ_t Adv(prf t)", "value": None,
                                 "in_total": False}}
"""`LineHeld`'s error beyond δ: each window's PRF advantage, one oracle-access distinguisher a window, which its claim
prices and no number here does. Every error records it by name, outside its total."""
CONDITIONAL_ON = {"NoOutsideCompute": (
    "statement C in full (Accounting.Remote.Held.Assumptions.NoOutsideCompute, a premise of LineHeld): no device "
    "outside the node computes on a registered block or a receipt during the session; assumed until certifiers cover "
    "the gate node's boundary (certifier-coverage) and certify that nothing but the verifier's messages entered it over "
    "the credited span"),
    "binding-split": (
    "the receipts' binding term (ledger.binding, Accounting.Guarantees.Remote.ReceiptsBindingOfCR) is charged with "
    "split_t, the chance that two runs of the prover from window t's cut on independent openings coins root the window "
    "differently, taken as 0; the prover is sent no coin commitment before the session's end (coins.Released, "
    "audit.Session.disclose), so its roots can't depend on an openings coin before it is released, and what remains "
    "is the statement that the play fixes each window's receipts by its cut (PROTOCOL.md §8)")}
"""What the line's error rests on that no record of the run discharges, each with what it says: the premises of its
statement and the gaps in its charge (`GAPS`). A line (`auditor_line.json`'s `conditional_on`, `credit.md`) is
conditional on them, `NoOutsideCompute` unless a lockdown grounds it (`GROUNDED`), and the `memory-line-of-record`
label names exactly the file's (`vocab.check_line_of_record`)."""
GAPS = ("binding-split",)
"""The gaps in the line's charge (`CONDITIONAL_ON`'s), which every line is conditional on, grounded or not."""
LOCKDOWN = "ProviderLockdown"
GROUNDED = {LOCKDOWN: (
    "the provider's lockdown of the gate node (Accounting.Remote.Held.Assumptions.ProviderLockdown, under "
    "trusted-host): while infra's records show it applied, no device outside the node computes on a registered "
    "block or a receipt, which gives NoOutsideCompute for a session whose signed span lies inside an applied interval "
    "(noOutsideCompute_of_lockdown); it assumes what the security group doesn't close: the provider's resolver, flows "
    "opened before it was applied, and the provider's own paths (metadata, management plane, a cluster's fabric, a "
    "shared filesystem)")}
"""What grounds `NoOutsideCompute` for a session (`lockdown_check`), each with what it says: a line whose sessions are
all grounded is conditional on it in its place."""
BOUNDARY = "boundary"
"""The check `--boundary` adds to `CHECKS` (`boundary_check`)."""


class Disagree(ValueError):
    """The Lean's line and `line.py`'s differ on the same inputs, so there is no line: where, and both values."""


def descriptor(tree: str) -> bytes:
    """The descriptor of the circuit a session of the verifier at git tree `tree` (`VERIFIER_TREE`'s) proves: what a
    period's `--circuit` is the SHA-512 of."""
    return f"{audit.PROTOCOL_ID} verifier tree {tree}".encode()


def verifier_tree(commit: str, repo: Path = HERE.parents[1]) -> tuple[str, str]:
    """The verifier run's recorded commit `commit` in full, and `VERIFIER_TREE`'s git tree there, read in `repo`'s
    git."""
    got = []
    for rev in (f"{commit}^{{commit}}", f"{commit}^{{commit}}:{VERIFIER_TREE}"):
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", rev], capture_output=True,
                           text=True)
        if p.returncode or len(p.stdout.strip()) != 40:
            raise SystemExit(f"auditor_line: {repo}'s git has no {VERIFIER_TREE} at commit {commit}: run where git has "
                             "the verifier run's commit (`git fetch origin COMMIT`), or pass --repo")
        got.append(p.stdout.strip())
    return got[0], got[1]


def requested_pin(out: Path) -> str | None:
    """The commit a host run's verifier ran at: the `pinned` of the request.json beside `out` (host.py `gate_run`,
    which runs it only from that tree), or None for a run with no request."""
    req = out.parent / "request.json"
    return json.loads(req.read_text()).get("pinned") if req.is_file() else None


def peer_of(out: Path, v: dict) -> str:
    """The challenger peer the verifier ran as (`verifier.py --challenger-peer`), from the run's record: each gate
    session's in `v` (verifier.json's `challenger.peer`, `challengers.Gate.finish`), and a host run's `request.json`
    beside `out` (its `argv`; its `peer` is the audited node). Refused unless the record names exactly one."""
    named = {(s.get("challenger") or {}).get("peer") for s in v["sessions"]
             if (s.get("challenger") or {}).get("challenger") == "gate"}
    req = out.parent / "request.json"
    if req.is_file():
        argv = json.loads(req.read_text()).get("argv") or []
        if "--challenger-peer" in argv[:-1]:
            named.add(argv[argv.index("--challenger-peer") + 1])
    named.discard(None)
    if len(named) != 1:
        got = f"challenger peers {', '.join(sorted(named))}" if named else "no challenger peer"
        raise SystemExit(f"auditor_line: the run's record names {got} (verifier.json's gate sessions, {req}'s "
                         "--challenger-peer): pass --peer NAME, the verifier machine's name in the challenger host's "
                         "/etc/vy/peers")
    return named.pop()


def release_log(lines: list[str]) -> dict[str, dict[str, list]]:
    """A release log's lines after its head, by family: each coin's root (`after`), its request's receipt and the
    coin's release, on the challenger's clock, in index order (`challenger.released_roots_of` checks the form)."""
    fam: dict[str, dict[str, list]] = {}
    for i in range(1, len(lines) - 1, 2):
        root, release = json.loads(lines[i]), json.loads(lines[i + 1])
        f = fam.setdefault(root["label"], {"after": [], "received": [], "released": []})
        f["after"].append(root["root"])
        f["received"].append(root["received_at"])
        f["released"].append(release["released_at"])
    return fam


def auditor_seen(lines: list[str], seen: list[dict], per_coin: int = 1) -> tuple[list[dict], dict]:
    """`seen` (a session's `windows_seen`) with each group's coin and last root on the challenger's clock: `released_ns`
    challenge coin g's release, `arrivals_ns` the first receipt of the requests that name the window's last root by
    their place, openings coin g's and, when it names the same root, σ_(g+2)'s; and which family's came first. Both
    leave only once that root has come, so a root the prover repeats can't make A_g early, as an earlier request
    naming the same value (σ_(g+1)'s, or another window's) would.

    Drawn `per_coin` G > 1 windows a coin (`groups_per_coin`), only a draw's last window closes, so openings coin c's
    request names the last root of draw c's last window and no request names another window's by place. A_g is then
    the first receipt of the requests that leave only once window g's last root has come: openings coin g // G's and
    σ_(g+2)'s, which at k = 1 names window g's root and above it a root of window g + 1, whose roots come after g's.
    That time is at or after A_g, so L_g is at least window g's own and the line credits no more."""
    fam = release_log(lines)
    ch, op = fam.get("challenge"), fam.get("openings")
    n = len(seen)
    draws = -(-n // per_coin)
    if ch is None or op is None or len(ch["released"]) < n or len(op["after"]) < draws:
        want = (f"one of each for each of {n} groups" if per_coin == 1 else
                f"one challenge coin for each of {n} groups and one openings coin for each of their {draws} draws")
        raise ValueError(f"the release log holds {len((ch or {}).get('released', ()))} challenge and "
                         f"{len((op or {}).get('after', ()))} openings coins, not {want}")
    out, by = [], {}
    for g, w in enumerate(seen):
        if per_coin == 1:
            t, label = op["received"][g], "openings"
            if g + 2 < len(ch["after"]) and ch["after"][g + 2] == op["after"][g] and ch["received"][g + 2] < t:
                t, label = ch["received"][g + 2], "challenge"
        else:
            t, label = op["received"][g // per_coin], "openings"
            if g + 2 < n and ch["received"][g + 2] < t:
                t, label = ch["received"][g + 2], "challenge"
        by[label] = by.get(label, 0) + 1
        out.append({**w, "released_ns": ch["released"][g], "arrivals_ns": [t] if w["arrivals_ns"] else []})
    return out, {"last_root_first_named_by": by}


def q(xs: list[float], scale: float = 1e6) -> dict | None:
    xs = sorted(x / scale for x in xs)
    if not xs:
        return None
    at = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 3), "median": round(at(0.5), 3), "p90": round(at(0.9), 3),
            "p99": round(at(0.99), 3), "max": round(xs[-1], 3)}


def group_l(seen: list[dict]) -> list[int]:
    """Each group's L but the last's (`paced_line`'s L_g)."""
    s, n = [w["released_ns"] for w in seen], len(seen)
    return [max(seen[g]["arrivals_ns"][-1], s[g + 1]) - s[g] for g in range(n - 1)]


def rates_at(disk_bytes_per_s: float = RATES["disk_bytes_per_s"],
             link_bytes_per_s: float = RATES["link_bytes_per_s"], bursts: str = "stated") -> dict:
    """The node's rates `verifier.recorded_rates` takes beside the record's squarings: its disks' and links' ceilings,
    and with `bursts` stated `ledger.Node.stated`'s bursts (one rebuild batch on the chips, a second of the disks, one
    link window: `verifier.ingress_bursts` in each form)."""
    if bursts not in BURSTS:
        raise ValueError(f"bursts are one of {', '.join(BURSTS)}, not {bursts}")
    out = {"disk_bytes_per_s": disk_bytes_per_s, "link_bytes_per_s": link_bytes_per_s}
    if bursts == "stated":
        node = ledger.Node.stated(disk_bytes_per_s=disk_bytes_per_s, link_bytes_per_s=link_bytes_per_s)
        out |= {"rebuild_burst_bytes": node.rebuild_burst_bytes, "disk_burst_bytes": node.disk_burst_bytes,
                "link_burst_bytes": node.link_burst_bytes}
    return out


NOTHING = object()
"""What `differ` names as the value of a key that one side's JSON lacks."""


def said(x: object) -> str:
    return "nothing" if x is NOTHING else json.dumps(x)


def differ(lean: object, python: object, at: str = "") -> tuple[str, object, object] | None:
    """The first place two JSON values differ, its path and each one's value there, or None when they are equal: the
    same keys, lists of the same length, and leaves of the same type and value."""
    if isinstance(lean, dict) and isinstance(python, dict):
        for k in sorted(set(lean) | set(python)):
            if k not in lean or k not in python:
                return f"{at}.{k}", lean.get(k, NOTHING), python.get(k, NOTHING)
            if (d := differ(lean[k], python[k], f"{at}.{k}")) is not None:
                return d
        return None
    if isinstance(lean, list) and isinstance(python, list):
        if len(lean) != len(python):
            return f"{at} (its length)", len(lean), len(python)
        return next((d for i, (x, y) in enumerate(zip(lean, python)) if (d := differ(x, y, f"{at}[{i}]"))), None)
    return None if type(lean) is type(python) and lean == python else (at or ".", lean, python)


def line_of_record(inputs: dict, lean: lv.Lean | None) -> dict:
    """The line of `inputs` as `line.run_json` writes it, read off the Lean verifier's record in the record form
    (`line.record_inputs_json`'s, the Lean verifier's `line_record`), or of groups and windows given
    (`line.inputs_json`'s, its `line`): with `lean`, the Lean verifier's answer, held exactly to `line.py`'s on the
    same inputs (`differ`), or `Disagree` naming both values; with none, `line.py`'s alone. A line both refuse, with
    the same reason, is `line.Refused`."""
    op, run = ("line_record", line.record_run_json) if "record" in inputs else ("line", line.run_json)
    python = run(inputs)
    got = python
    if lean is not None:
        got = lean.call({"op": op, **inputs})
        if not isinstance(got, dict) or ("refused" in got) != ("refused" in python):
            raise Disagree(f"the Lean's line and line.py's disagree: the Lean's {outcome(got)}, line.py's "
                           f"{outcome(python)}")
        if (d := differ(got, python)) is not None:
            at, a, b = d
            raise Disagree(f"the Lean's line and line.py's disagree at {at}: the Lean's {said(a)}, line.py's "
                           f"{said(b)}")
    if "refused" in got:
        raise line.Refused(got["refused"])
    return got


def outcome(got: object) -> str:
    """A line's answer in a few words: its refusal, or each form's weights share held over time."""
    if isinstance(got, dict) and "refused" in got:
        return f"refusal {json.dumps(got['refused'])}"
    try:
        return "line, weights held " + ", ".join(f"{f} {got[f]['time']['weights_share']}" for f in line.FORMS)
    except (KeyError, TypeError):
        return json.dumps(got)


def shares(session: dict, registration: dict, host_bytes: int, rates: dict, lean: lv.Lean | None = None, *,
           clock: str = "record", transcript: dict | None = None) -> dict:
    """A session's shares of the node in each form at `rates` (`rates_at`), on the clock its `windows_seen` read
    (`clock`: the record's, which holds its times, or another, `line.CLOCKS`), from its line (`line_of_record`, the
    Lean's with `lean`, off the record `Session.run` makes of `transcript` where given; `shares_of`)."""
    return shares_of(line_of_record(V.recorded_inputs(session, registration, host_bytes=host_bytes,
                                                      rates=V.recorded_rates(session, registration, **rates),
                                                      clock=clock, transcript=transcript), lean))


def transcript_of(out: Path, s: dict) -> dict | None:
    """Session `s`'s transcript (`lean_verifier.Transcript`'s file) where the run's out/ keeps it: its name under
    `out/transcripts/`, gzipped (`host.gzip_transcripts`) or not, held to the sha256 verifier.json gives it; None where
    there is none (a host run fetched without `--transcripts`)."""
    t = s.get("transcript") or {}
    if not t.get("path") or not t.get("sha256"):
        return None
    name = Path(t["path"]).name
    for p, read in ((out / "transcripts" / f"{name}.gz", gzip.open), (out / "transcripts" / name, open)):
        if p.is_file():
            with read(p, "rb") as f:
                raw = f.read()
            if hashlib.sha256(raw).hexdigest() != t["sha256"]:
                raise SystemExit(f"auditor_line: {p} is not session {s['k']}'s transcript: its sha256 isn't the one "
                                 "verifier.json gives it")
            return json.loads(raw)
    return None


def shares_of(got: dict) -> dict:
    """A line's (`line.run_json`'s form) shares of the node in each form (`line.FORMS`): weights and filler held, over
    time and at the passes (`SHARES`), each exact (`exact`) and the float nearest it beside it, and the worst stretch
    over time."""
    row = {}
    for f in line.FORMS:
        j = got[f]
        exact = {"time": j["time"]["weights_share"], "snapshot": j["snapshot"]["weights_share"],
                 "filler_time": j["time"]["filler_share"], "filler_snapshot": j["snapshot"]["filler_share"]}
        worst = next(x for x in j["stretches"] if x["stretch"] == j["time"]["stretch"])
        row[f] = {**{k: float(line.number(x)) for k, x in exact.items()}, "exact": exact,
                  "worst_stretch": worst["stretch"], "worst_l_max_ms": worst["l_max_ns"] / 1e6,
                  "worst_span_s": worst["span_ns"] / 1e9}
    return row


def session_error(session: dict) -> dict:
    """An accepted session's failure probability, the chance its line credits more than the node held: `total`, the
    memory line's δ as the ledger computed it (`ledger.memory`'s `Line.delta`, every form's in the record's
    `declared`), read and never recomputed, so it charges whatever the ledger does (the F the node states among it);
    beside it its six parts (`ledger.delta_parts`): `delta_profile`, the windows' profile δ, `delta_slack`, AuditRO's
    2^-slack a pass, `delta_hiding`, the coin commitment's Σ_t hide_t, `delta_binding`, the receipts' binding term,
    `delta_sketch`, δs, and `delta_rejection`, the draw's Σ_t k_t/2^64. A record that carries the parts (verifier.json's
    `delta`) must sum to the total as the ledger adds them, or there is no line (`Disagree`); one written before them
    gives the profile δ from its windows (`recorded_windows`) and the slack from its passes at the slack its ledger was
    charged at (its `inputs`' `slack_log2`, else `V.UNRECORDED_SLACK_LOG2`), both exact, and δs as the rest. A ledger
    that didn't charge the hiding, the binding or the rejection (no such part) has it added to its total: the hiding
    from its windows (`recorded_windows`, as `ledger.hiding` reads them), the binding at the session's bounds
    (`recorded_binding`, as `ledger.binding` reads them), reading F·C off the record's sketch term (`sketch_failure`),
    so a record with none has no binding term here, and the rejection at its windows' requested draws (the session's
    `samples` in each, as `ledger.rejection` reads them off the record). Each in floats and exact beside them
    (`exact`), the total's log2, and `named_terms`, the error's terms with no numeric value (`NAMED_TERMS`), in no
    total."""
    lg = session["ledger"]
    totals = {m["delta"] for m in lg["declared"]["memory"].values() if "delta" in m}
    if len(totals) != 1:
        raise Disagree(f"session {session['k']}'s memory line forms record δ {sorted(totals)}: the ledger computes one")
    total = Fraction(totals.pop())
    hiding = binding = rejection = None
    if (d := lg.get("delta")) is not None:
        added = 0.0
        for n in ledger.DELTA_PARTS:
            added += d.get(n, 0.0)
        if added != float(total):
            raise Disagree(f"session {session['k']}'s δ parts {d} don't sum to its memory line's δ {float(total)!r}")
        profile, slack, sketch = (Fraction(d[n]) for n in ("profile", "slack", "sketch"))
        if "hiding" in d:
            hiding = Fraction(d["hiding"])
        if "binding" in d:
            binding = Fraction(d["binding"])
        if "rejection" in d:
            rejection = Fraction(d["rejection"])
    else:
        profile = pooled.number(V.recorded_windows(session)[1])
        slack = Fraction(lg["inputs"]["passes"], 2 ** lg["inputs"].get("slack_log2", V.UNRECORDED_SLACK_LOG2))
        sketch = total - profile - slack
    if hiding is None:
        hiding = Fraction(ledger.hiding_over(len(V.recorded_windows(session)[0])))
        total += hiding
    if binding is None:
        binding = recorded_binding(session)
        total += binding
    if rejection is None:
        windows = len(V.recorded_windows(session)[0])
        rejection = Fraction(windows * session["samples"], 2 ** ledger.REJECTION_LOG2)
        total += rejection
    exact = {"delta_profile": profile, "delta_slack": slack, "delta_hiding": hiding, "delta_binding": binding,
             "delta_sketch": sketch, "delta_rejection": rejection, "total": total}
    return {**{k: float(x) for k, x in exact.items()}, "exact": {k: line.text(x) for k, x in exact.items()},
            "log2_total": timing.log2(exact["total"]), "named_terms": NAMED_TERMS}


def recorded_append_coins(session: dict) -> int | None:
    """The append coins an accepted session's coin commitment reserved: its record's (`append_coins`), or, before the
    record kept them, what its challenger released when it released every coin it committed to (`complete`); None
    when neither says."""
    if "append_coins" in session:
        return session["append_coins"]
    end = (session.get("challenger") or {}).get("end") or {}
    return end["released"].get(coins.APPEND, 0) if end.get("complete") and "released" in end else None


def recorded_binding(session: dict) -> Fraction:
    """`ledger.binding` from an accepted session's record, at the session's bounds (`ledger.binding_over`): its draws
    (`V.recorded_windows`, a window each at `groups_per_coin` g = 1), the most receipts a draw of it can have (g times
    `ledger.window_receipts`, at its pace's k, its registered blocks and, when its append coins aren't recorded
    (`recorded_append_coins`), the codec's `max_blocks`) and its draw cap `samples`, F·C the queries its sketch term
    (`sketch_failure`) charged; 0 for a record with no sketch term, which states no C."""
    sf = (session.get("ledger") or {}).get("sketch_failure")
    if not sf:
        return Fraction(0)
    public = session["record"]["public"]
    pace = public["pace"]
    appends = recorded_append_coins(session)
    L = ledger.window_receipts(None if pace is None else audit.Pace(pace["k"], pace["window_ns"]), public["blocks"],
                               1 if appends is None else appends, wire.codec_named(public["codec"]).max_blocks)
    return Fraction(ledger.binding_over(len(V.recorded_windows(session)[0]), public.get("groups_per_coin", 1) * L,
                                        session["samples"], sf.get("outside", 1) * sf["compressions"]))


def union_error(rows: list[dict]) -> dict:
    """The chance some session of `rows` fails (`session_error`), at most the sum of their totals: the sessions, the
    sum and its exact value, its log2 (None for none), and the terms they name with no numeric value
    (`NAMED_TERMS`, none for no session), in no total."""
    rows = sorted(rows, key=lambda r: r["k"])
    total = sum((line.number(r["error"]["exact"]["total"]) for r in rows), Fraction(0))
    return {"sessions": [r["k"] for r in rows], "total": float(total), "exact": line.text(total),
            "log2": timing.log2(total) if total else None, "named_terms": NAMED_TERMS if rows else {}}


def bound_check(v: dict, model: dict | None, source: str | None) -> dict:
    """Whether the run's registered store is `model`'s ({name, checkpoint_sha512}, from `source`): the verifier's
    binding (verifier.json's `binding`, `remote_bandwidth.store.Binding.record`) held the hello's payload root to the root of the
    store the pinned checkpoint lays out under the verifier's payload key, the Lean verifier deciding it (`lean`, its
    `hold`'s answer: bound, at that root), that root is the registration's, and the binding's model is `model`."""
    b, root = v.get("binding"), (v.get("registration") or {}).get("payload_root")
    out = {"model": model, "source": source, "binding": None if b is None else {
        k: b.get(k) for k in ("name", "checkpoint_sha512", "payload_root", "expected_root", "bound", "lean")}}
    if model is None:
        why = "no model to bind the store to: the audit's period signs none and no --model-pin names one"
    elif b is None:
        why = ("the verifier bound the store to no model (verifier.py --model ROLE --checkpoint DIR --pin FILE), so its "
               "payload root is the prover's choice")
    elif b.get("bound") is not True:
        why = f"the verifier didn't hold the hello's payload root to {b.get('name')}'s store"
    elif b.get("expected_root") != root or b.get("payload_root") != root:
        why = "the binding's payload root is not the registration's"
    elif not isinstance(lean := b.get("lean"), dict) or "regions" not in lean or lean.get("expected_root") != root:
        why = ("the Lean verifier didn't hold the hello's payload root to the store (`hold`): the binding is "
               "store.py's alone")
    elif (b.get("name"), b.get("checkpoint_sha512")) != (model["name"], model["checkpoint_sha512"]):
        why = (f"the verifier bound the store to {b.get('name')} ({str(b.get('checkpoint_sha512'))[:16]}...), not "
               f"{model['name']} ({model['checkpoint_sha512'][:16]}...)")
    else:
        why = None
    return {**out, "ok": why is None, "reason": why}


def certified_record_json(r: cert.CertifiedRecord) -> dict:
    """A network certifier's certified record as `read_boundary` reads it: every bytes field as hex, the status as its
    reason (None: complete)."""
    h, c = r.head, r.certificate
    return {"head": {"link": h.link, "direction": h.direction.value, "window": h.window, "buckets": h.buckets,
                     "allocation": h.allocation, "frame_bytes": h.frame_bytes, "bucket_ms": h.bucket_ms,
                     "status": None if h.status.reason is None else h.status.reason.value, "deadline": h.deadline,
                     "parameters": h.parameters.hex(), "counts": list(h.counts), "root": h.root.hex()},
            "salt": r.salt.hex(),
            "certificate": {"key_id": c.key_id.hex(), "link": c.link, "direction": c.direction.value,
                            "window": c.window, "digest": c.digest.hex(), "signature": c.signature.hex()}}


def certified_record(d: dict) -> cert.CertifiedRecord:
    """The certified record `d` (`certified_record_json`'s form) is."""
    h, c = d["head"], d["certificate"]
    head = cert.RecordHead(h["link"], cert.Direction(h["direction"]), h["window"], h["buckets"], h["allocation"],
                           h["frame_bytes"], h["bucket_ms"], Status(None if h["status"] is None else Reason(h["status"])),
                           h["deadline"], bytes.fromhex(h["parameters"]), tuple(h["counts"]), bytes.fromhex(h["root"]))
    return cert.CertifiedRecord(head, bytes.fromhex(d["salt"]), cert.Certificate(
        bytes.fromhex(c["key_id"]), c["link"], cert.Direction(c["direction"]), c["window"], bytes.fromhex(c["digest"]),
        bytes.fromhex(c["signature"])))


def read_boundary(path: Path, pin: str) -> dict:
    """The boundary file `--boundary` names (JSON), its certifier's key held to `pin`, the SHA-512 of the key's public
    bytes: `links`, the gate node's boundary links; `key` ({N, W, H, public}) and `session`, the network certifier's LMS
    key and its certifier session's id; `grid`, its ingress grid (`GridParams`'s fields); `windows`, each ingress
    window's number and span `[start_ns, end_ns)` on the challenger's clock; `verifier_frames`, by link and window, the
    frames that carried the verifier's messages, as the link frames them; and `records`, the certified records
    (`certified_record_json`)."""
    raw = path.read_bytes()
    d = json.loads(raw)
    k = d["key"]
    key = cert.LmsKey(lms.Params(k["N"], k["W"], k["H"]), bytes.fromhex(k["public"]))
    if hashlib.sha512(key.public).hexdigest() != pin:
        raise SystemExit(f"auditor_line: {path}'s certifier key is not the one --boundary-pin names ({pin[:16]}...)")
    if not d["links"]:
        raise SystemExit(f"auditor_line: {path} names no boundary link")
    params = GridParams(**d["grid"])
    frames = {link: {int(w): [bytes.fromhex(f) for f in fs] for w, fs in by.items()}
              for link, by in d.get("verifier_frames", {}).items()}
    if any(len(f) != params.frame_bytes for by in frames.values() for fs in by.values() for f in fs):
        raise SystemExit(f"auditor_line: {path}'s verifier frames are not all the grid's {params.frame_bytes} bytes")
    return {"sha512": hashlib.sha512(raw).hexdigest(), "links": list(d["links"]), "key": key,
            "session": bytes.fromhex(d["session"]), "params": params,
            "windows": [(w["window"], w["start_ns"], w["end_ns"]) for w in d["windows"]], "frames": frames,
            "records": [certified_record(r) for r in d["records"]]}


def boundary_check(b: dict, seen: list[dict]) -> dict:
    """Whether the gate node's boundary links let in nothing but the verifier's messages over a session's credited span,
    from σ_0's release to its last root's arrival on the challenger's clock (`seen`, `auditor_seen`'s), by the network
    certifiers' records `b` (`read_boundary`'s): the ingress windows that meet the span cover it, and each boundary
    link's record of each is one record with a complete status, signed under the certifier's key in its session, on its
    grid (`Audit.check_ingress_link`), whose real frames are as many as the verifier's messages framed for that
    link-window, under the same root (`frames_root`)."""
    lo = seen[0]["released_ns"]
    hi = max([lo, *(t for w in seen for t in w["arrivals_ns"])])
    meet = sorted(((s, e, w) for w, s, e in b["windows"] if s <= hi and lo < e))
    at = lo
    for s, e, _ in meet:
        if s <= at:
            at = max(at, e)
    why = [] if at > hi else [f"the certifiers' windows cover its span [{lo}, {hi}] ns only to {at}"]
    windows = sorted({w for _, _, w in meet})
    for link in b["links"]:
        verdicts = {v.window: v for v in caudit.Audit().check_ingress_link(
            link, b["records"], max(windows, default=-1) + 1, b["key"], b["params"], b["session"])}
        heads = {r.window: r.head for r in b["records"] if r.link == link and r.direction is cert.Direction.INGRESS}
        for w in windows:
            if not verdicts[w].accepted:
                why.append(f"{link}'s ingress window {w} has no record to read ({verdicts[w].violation})")
                continue
            mine = b["frames"].get(link, {}).get(w, [])
            n = sum(heads[w].counts)
            if n != len(mine):
                why.append(f"{link}'s ingress window {w} let in {n} frames, the verifier's messages {len(mine)}")
            elif heads[w].root != cert.frames_root(mine, b["params"]):
                why.append(f"{link}'s ingress window {w} let in frames other than the verifier's messages (its root "
                           "is not theirs)")
    return {"ok": not why, "reason": "; ".join(why) or None, "span_ns": [lo, hi], "windows": windows,
            "links": b["links"]}


def _utc(x: object) -> bool:
    return isinstance(x, str) and Ch.later(x, 0) == x


def read_lockdown(path: Path) -> dict:
    """The provider's lockdown of the gate node as infra recorded it (`--lockdown FILE`): `{"node", "allow": [{"host",
    "ip"}, ...], "intervals": [{"applied_utc", "lifted_utc", "records": ["art:...", ...]}, ...],
    "shared_filesystem"}`, each interval's times `YYYY-MM-DDTHH:MM:SSZ` with applied before lifted and its records
    infra's (the rules before, during and after), and `shared_filesystem` infra's answer to whether the node mounts
    one (true, false, or null while it is pending), with the file's SHA-512. Refused (SystemExit) where it isn't of
    that form."""
    raw = path.read_bytes()
    try:
        d = json.loads(raw)
    except ValueError:
        d = None
    if not (isinstance(d, dict) and isinstance(d.get("node"), str) and isinstance(d.get("allow"), list) and d["allow"]
            and all(isinstance(a, dict) and isinstance(a.get("host"), str) and isinstance(a.get("ip"), str)
                    for a in d["allow"])
            and isinstance(d.get("intervals"), list)
            and all(isinstance(i, dict) and _utc(i.get("applied_utc")) and _utc(i.get("lifted_utc"))
                    and i["applied_utc"] < i["lifted_utc"] and isinstance(i.get("records"), list)
                    and all(isinstance(r, str) and r.startswith("art:") for r in i["records"]) for i in d["intervals"])
            and "shared_filesystem" in d and d["shared_filesystem"] in (True, False, None)):
        raise SystemExit(f"auditor_line: {path} is not a lockdown record (`read_lockdown`'s form)")
    return {**d, "sha512": hashlib.sha512(raw).hexdigest()}


def signed_span(w: dict | None) -> list[str] | None:
    """A gate session's span in UTC as the challenger signed it on the ledger, which holds its credited passes: its
    open's `started_utc` to its end's `ended_utc`; None without both."""
    span = [(((w or {}).get(e) or {}).get("body") or {}).get(f) for e, f in (("open", "started_utc"),
                                                                            ("end", "ended_utc"))]
    return span if all(map(_utc, span)) else None


def lockdown_check(lockdown: dict | None, span: list[str] | None) -> dict:
    """What grounds a gate session's `NoOutsideCompute` (`no_relay`): the provider's lockdown (`ProviderLockdown`) where
    its signed span lies strictly inside one applied interval at whole seconds (applied before its start and its end
    before the lift, so the times floored to the second still bound it) and infra hasn't answered that the node mounts
    a shared filesystem; otherwise nothing, and its line is conditional on `NoOutsideCompute` itself, with why."""
    def bare(why: str) -> dict:
        return {"grounded_by": None, "conditional_on": "NoOutsideCompute", "span_utc": span, "reason": why}

    if lockdown is None:
        return bare("no lockdown record (--lockdown)")
    if span is None:
        return bare("the ledger signs no span for it (its open's started_utc, its end's ended_utc)")
    if lockdown["shared_filesystem"] is True:
        return bare(f"{lockdown['node']} mounts a shared filesystem, a path the lockdown doesn't close")
    inside = next((i for i in lockdown["intervals"] if i["applied_utc"] < span[0] and span[1] < i["lifted_utc"]), None)
    if inside is None:
        return bare(f"its span {span[0]} to {span[1]} lies inside no interval the lockdown was applied")
    return {"grounded_by": LOCKDOWN, "conditional_on": LOCKDOWN, "span_utc": span,
            "interval": [inside["applied_utc"], inside["lifted_utc"]], "records": inside["records"],
            "shared_filesystem": "pending (infra)" if lockdown["shared_filesystem"] is None else "none mounted"}


def conditional_on(rows: list[dict]) -> dict:
    """What the line is conditional on (`CONDITIONAL_ON`, `GROUNDED`): the assumption each credited session's
    `NoOutsideCompute` rests on (`no_relay`), or each gate session's when none is credited, `NoOutsideCompute` with
    none; then the gaps in its charge (`GAPS`)."""
    xs = [r for r in rows if r["credit"]["ok"]] or rows
    names = sorted({r["no_relay"]["conditional_on"] for r in xs if "no_relay" in r}) or ["NoOutsideCompute"]
    return {n: {**CONDITIONAL_ON, **GROUNDED}[n] for n in [*names, *GAPS]}


def line_label(doc: dict) -> str:
    """The `memory-line-of-record` value naming the doc's line: who computed it and what it is conditional on, as
    the evidence store's labeller holds it to this file (`vocab.check_line_of_record`)."""
    return f"{doc['line_by']} conditional on {', '.join(doc['conditional_on'])}"


def with_check(credit: dict, name: str, check: dict) -> dict:
    """`credit` (`credit_check`'s) with one more check, which its credit then needs too."""
    zero = [*credit["zero"], *([] if check["ok"] else [f"not {name}: {check['reason']}"])]
    return {**credit, name: check, "ok": credit["ok"] and check["ok"], "zero": zero}


def clock_rate_ppm(Rv: list[int], Ra: list[int]) -> float | None:
    """The auditor's clock's rate over the verifier's, less 1, in ppm, from each coin's release on both (`Rv`, `Ra`):
    the median of the rates between each coin and the one half the session after it. The verifier stamps a coin it
    fetches alone (σ_0 and σ_1 at k = 1) before asking for it, and one asked ahead (`coins.Coins.ahead`) after it came
    back (`audit.Session._release`), so those first coins sit a round trip off the rest. A rate taken end to end reads
    that round trip over the session's span (−441 ppm at 60 s and −8989 ppm at 1 s on gate run 7); here it moves two
    of the pairs. None for a session of fewer than four coins."""
    h = len(Rv) // 2
    rates = [((Ra[i + h] - Ra[i]) / (Rv[i + h] - Rv[i]) - 1) * 1e6 for i in range(h) if Rv[i + h] != Rv[i]]
    return statistics.median(rates) if h >= 2 and rates else None


def cross_check(verifier_seen: list[dict], auditor: list[dict]) -> dict:
    """The challenger's clock held to the verifier's stamps: coin intervals, rate and each group's L on both."""
    Rv, Ra = [w["released_ns"] for w in verifier_seen], [w["released_ns"] for w in auditor]
    n = len(Rv)
    Lv, La = group_l(verifier_seen), group_l(auditor)
    return {"coin_interval_auditor_minus_verifier_ms": q([(Ra[g + 1] - Ra[g]) - (Rv[g + 1] - Rv[g])
                                                          for g in range(n - 1)]),
            "rate_auditor_over_verifier_minus_1_ppm": clock_rate_ppm(Rv, Ra),
            "L_auditor_ms": q(La), "L_verifier_ms": q(Lv), "L_auditor_minus_verifier_ms": q(
                [a - v for a, v in zip(La, Lv)])}


def log_checks(lines: list[str], window_root: str) -> dict:
    """What the release log shows about the order of coins and roots: each coin released after its request's receipt,
    the clock never running back, and σ_0 asked after the session's window root."""
    fam = release_log(lines)
    times = [t for i in range(1, len(lines) - 1, 2)
             for t in (json.loads(lines[i])["received_at"], json.loads(lines[i + 1])["released_at"])]
    return {"coins_released": {label: len(f["released"]) for label, f in fam.items()},
            "coin_before_its_root": [(label, i) for label, f in fam.items()
                                     for i, (r, s) in enumerate(zip(f["received"], f["released"])) if s < r],
            "clock_monotone": all(a <= b for a, b in zip(times, times[1:])),
            "sigma0_after_window_root": bool(fam.get("challenge", {}).get("after"))
            and fam["challenge"]["after"][0] == window_root}


def ledger_check(heads: list[Path], key: Ch.ChallengerKey, period: str, lane: str
                 ) -> tuple[list[str], dict, dict, dict]:
    """The auditor's ledger: its longest head under `key`, verified, with its windows by root and its periods."""
    lines = Ch.ledger_head([remote.ledger_lines(p) for p in heads], key, CH.lms_verify)
    raw = [line.encode() for line in lines]
    verify = lambda m, sig: CH.lms_verify(key, m, sig)  # noqa: E731
    windows = Ch.verify_ledger(raw, key.public[:16], verify)
    periods = Ch.verify_periods(raw, key.public[:16], verify)
    p = periods.get(period)
    doc = {"entries": len(lines), "sha512": hashlib.sha512("".join(x + "\n" for x in lines).encode()).hexdigest(),
           "verify_ledger": "passed", "heads": len(heads)}
    doc["period"] = None if p is None else {
        "name": period, "lane": p["lane"], "cap": p["cap"], "circuit": p["circuit"], "windows": len(p["windows"]),
        "opened_by": p["open"]["body"].get("by") if p.get("open") else None,
        "opened_utc": p["open"]["body"].get("opened_utc") if p.get("open") else None,
        "closed": p.get("close") is not None,
        "closed_utc": p["close"]["body"].get("closed_utc") if p.get("close") else None,
        "complete": p["close"]["body"].get("complete") if p.get("close") else None,
        "ok": p["lane"] == lane and p.get("close") is not None}
    return lines, windows, periods, doc


def credit_check(s: dict, R: str, lines: list[str], windows: dict, periods: dict, *, key: Ch.ChallengerKey,
                 period: str, desc: bytes, commit: str, pins: list[str] | None, main: list[str] | None,
                 bound: dict) -> dict:
    """Whether session `s`, whose window root is R, earns credit: its window credited toward the audit's period for
    the verifier `desc` describes (`credited_session`), that verifier's `commit` on `main`, its window and its period
    closed complete, its verdict the
    Lean's acceptance, its open's pin one the host's pins log `pins` records on the auditor's first-parent list of
    main `main` (`pinned_session`; None: no log, or no copy of main), and the run's store bound to the model (`bound`,
    `bound_check`'s); each check with its reason, and `zero` the
    reasons it earns none."""
    sid = (s.get("challenger") or {}).get("session")
    credited: dict = {"descriptor_sha512": hashlib.sha512(desc).hexdigest()}
    try:
        got = Ch.credited_session(lines, {"root": R}, key, pins, main, desc, CH.lms_verify)
        credited.update(period=got.period, ok=got.period == period, reason=None if got.period == period else (
            f"its window counts toward period {got.period}, not the audit's period {period}"))
        if credited["ok"] and commit not in (main or ()):
            credited.update(ok=False, reason=(
                f"its verifier ran commit {commit}, which is not on main's first-parent line in the auditor's copy "
                "(`git rev-list --first-parent origin/main`), so no tree main's check ran on: repin the verifier "
                "host to a commit on main's first-parent line (vy-verifier-pin) and rerun the window there"))
    except Ch.CoinError as e:
        credited.update(ok=False, reason=str(e))
    end = ((windows.get(R) or {}).get("end") or {}).get("body") or {}
    close = ((periods.get(period) or {}).get("close") or {}).get("body")
    outcome = next((w for w in (close or {}).get("windows", []) if w.get("session") == sid), None)
    complete = {"window": end.get("complete") is True, "released": end.get("released"),
                "period_closed": close is not None, "in_close": None if outcome is None else outcome.get("complete"),
                "period": None if close is None else close.get("complete") is True}
    why = []
    if not complete["window"]:
        reserved = (((windows.get(R) or {}).get("reserve") or {}).get("body") or {}).get("families") or []
        got = end.get("released") or {}
        short = [f"{got.get(label, 0)} of its {n} {label} coins" for label, n in reserved if got.get(label, 0) < n]
        why.append("its window has no end on the ledger" if not end else "its window closed incomplete: the "
                   f"challenger released {', '.join(short) or 'fewer coins than it reserved'}")
    if close is None:
        why.append(f"period {period} hasn't closed")
    elif not complete["period"]:
        short = sum(not w.get("complete") for w in close.get("windows", []))
        why.append(f"period {period} closed incomplete ({short} of its {len(close.get('windows', []))} windows "
                   "incomplete)")
    elif outcome is None or outcome.get("complete") is not True:
        why.append(f"period {period}'s close doesn't list its window complete")
    complete.update(ok=not why, reason="; ".join(why) or None)
    verdict = s.get("verdict")
    if verdict not in LEAN_VERDICTS:
        reason = f"its verdict is {verdict}'s, not the Lean's"
    elif s.get("disagreement") is not None:
        reason = "the two verifiers disagreed"
    elif s.get("accepted") is not True:
        reason = f"the Lean refused it ({(s.get('failure') or {}).get('failure')})"
    else:
        reason = None
    accepted = {"ok": reason is None, "verdict": verdict, "failure": s.get("failure"), "reason": reason}
    opened = (((windows.get(R) or {}).get("open") or {}).get("body")) or {}
    pinned: dict = {"commit": opened.get("pinned"), "seal": opened.get("pins")}
    try:
        Ch.pinned_session(lines, {"root": R}, key, pins, main, CH.lms_verify)
        pinned.update(ok=True, reason=None)
    except Ch.CoinError as e:
        pinned.update(ok=False, reason=str(e))
    checks = {"credited": credited, "complete": complete, "accepted": accepted, "pinned": pinned, "bound": bound}
    zero = [f"not {name}: {c['reason']}" for name, c in checks.items() if not c["ok"]]
    return {**checks, "ok": not zero, "zero": zero}


def session_check(s: dict, k: int, R: str, windows: dict, periods: dict, *, period: str, lane: str, peer: str,
                  pin: str) -> dict:
    """Session k's entries on the ledger: its reserve, open and end, held to the lane, the peer, the pin, its epoch and
    the period."""
    sid = (s.get("challenger") or {}).get("session")
    w = windows.get(R)
    if w is None:
        return {"window_root": R, "ok": False, "problem": "the ledger opens no session for this session's window root"}
    rs, op, end = w.get("reserve"), w["open"], w.get("end")
    b, ob = (rs or {}).get("body", {}), op["body"]
    chk = {"window_root": R, "session": op["session"], "is_the_verifiers": op["session"] == sid,
           "reserve_seq": (rs or {}).get("seq"), "open_seq": op["seq"], "end_seq": (end or {}).get("seq"),
           "lane": Ch.lane_of(op), "reserve_by": b.get("by"), "open_by": ob.get("by"), "peer": b.get("peer"),
           "open_peer": ob.get("peer"), "key_is_pin": ob.get("key_sha512") == pin and b.get("key_sha512") == pin,
           "scheme": b.get("scheme"), "epoch": b.get("epoch"), "families": b.get("families"),
           "in_period": op["session"] in ((periods.get(period) or {}).get("windows") or []),
           "end": None if end is None else {f: end["body"].get(f) for f in
                                            ("closed", "complete", "rc", "refused_hellos", "released", "ended_utc")}}
    chk["ok"] = bool(chk["is_the_verifiers"] and chk["lane"] == lane and chk["open_by"] == lane and chk["peer"] == peer
                     and chk["open_peer"] == peer and chk["key_is_pin"] and chk["scheme"] == coins.SCHEME
                     and chk["epoch"] == k and chk["in_period"] and end is not None
                     and end["body"].get("closed") is True and end["body"].get("refused_hellos") == 0)
    return chk


def signed_lines(d: Path, kind: str, R: str, heads: list[Path], key: Ch.ChallengerKey | None,
                 period: str | None) -> tuple[list[str], dict]:
    """The release log's lines from the session's directory `d`, each line the one the challenger's close signs, read
    through the ledger (`trusted_session`, a gate's session) or under the key the run wrote (`session_of`, a served
    one)."""
    files = remote.session_files(d)
    if kind == "gate":
        remote.trusted_session(d, heads, {"root": R}, key, CH.lms_verify)
        got = Ch.audited_session(Ch.ledger_head([remote.ledger_lines(p) for p in heads], key, CH.lms_verify),
                                 {"root": R}, key, CH.lms_verify)
        if got.period != period:
            raise Ch.CoinError(f"the ledger holds the session's window in period {got.period}, not {period}")
        trust = {"read": "trusted_session", "period": got.period, "cap": got.cap, "circuit": got.circuit}
    else:
        key = Ch.ChallengerKey.from_json(json.loads((d / remote.KEY).read_text()))
        Ch.session_of(key, files["record"], files["log"], files["close"], CH.lms_verify)
        trust = {"read": "session_of", "key_sha512": hashlib.sha512(key.public).hexdigest(),
                 "note": "a served challenger on the verifier's machine: no ledger, not the auditor's clock"}
    close = Ch.ChallengerClose.from_json(files["close"])
    if len(files["log"]) != close.lines or Ch.log_chain(files["log"]) != close.chain:
        raise Ch.CoinError("the release log read is not the one the close signs")
    return files["log"], trust


def analyse(out: Path, *, host_bytes: int, pin: str | None, period: str | None, heads: list[Path] | None,
            lane: str = LANE, peer: str | None = None, rates: dict | None = None,
            desc: bytes | None = None, commit: str | None = None, pins: list[str] | None = None,
            main: list[str] | None = None, model: dict | None = None, lean: lv.Lean | None = None,
            boundary: dict | None = None, lockdown: dict | None = None) -> dict:
    """The run's line: `desc` is the descriptor of the verifier that ran it (`descriptor` of `commit`'s tree, `commit`
    in full), `pins` the challenger host's pins log, its lines (None: the gate served none), `main` the auditor's first-parent list of
    main (None: none given), `model` the model the store must be bound to ({name, checkpoint_sha512},
    `--model-pin`'s) when the audit's period signs none, `peer` the gate sessions' peer (None: the record's,
    `peer_of`), `rates` the node's (`rates_at`; None: its defaults), `lean` the Lean verifier that computes every
    session's line (`line_of_record`; None: `line.py` alone, `line_by` python), and `boundary` the certifiers' records
    of the gate node's boundary that each gate session's credit is then held to (`read_boundary`, `boundary_check`;
    None: not checked), and `lockdown` the provider's lockdown of the gate node, which grounds a gate session's
    `NoOutsideCompute` where its signed span lies inside an applied interval (`read_lockdown`, `lockdown_check`;
    None: none does). Each session's line is the Lean's off the record `Session.run` makes of its transcript where
    out/ keeps it (`transcript_of`, `line_of`), else off its kept record."""
    rates = rates_at() if rates is None else rates
    v = json.loads((out / "verifier.json").read_text())
    reg = v["registration"]
    D = bytes.fromhex(reg["digest"])
    ch_dir = out / "challenger"
    kinds = {(s.get("challenger") or {}).get("challenger") for s in v["sessions"]}
    key = windows = periods = None
    lines: list[str] = []
    doc: dict = {"registration": reg["digest"], "host_bytes": host_bytes, "rates": rates,
                 "line_by": "python" if lean is None else "lean", "conditional_on": dict(CONDITIONAL_ON),
                 "boundary": {"checked": False, "reason": "off by default (--boundary)"} if boundary is None else {
                     "checked": True, "sha512": boundary["sha512"], "links": boundary["links"]},
                 "lockdown": {"read": False, "reason": "no lockdown record (--lockdown)"} if lockdown is None else {
                     "read": True, **{k: lockdown[k] for k in ("sha512", "node", "allow", "intervals",
                                                               "shared_filesystem")}},
                 "commit": commit, "descriptor": None if desc is None else desc.decode(),
                 "pins": None if pins is None else {"lines": len(pins), "sha512": Ch.log_digest(pins)},
                 "verifier_rtt_ms": {k: x / 1e6 for k, x in v.get("rtt", {}).items() if k.endswith("_ns")}}
    if "gate" in kinds:
        if pin is None or period is None or desc is None or commit is None:
            raise SystemExit("auditor_line: a gate run's line needs --pin, --period and --commit (the verifier run's "
                             "recorded commit, whose verifier the period must have committed)")
        ran = requested_pin(out)
        if ran is not None and ran != commit:
            raise SystemExit(f"auditor_line: the host run's request.json ran its verifier at {ran} (`pinned`), not "
                             f"--commit {commit}: pass --commit {ran}, the commit whose verifier decided its sessions")
        doc["peer"] = {"name": peer, "from": "--peer"} if peer else {"name": peer_of(out, v), "from": "record"}
        peer = doc["peer"]["name"]
        heads = [*(heads or []), *sorted(ch_dir.glob("*.ledger.jsonl"))]
        key = next((k for k in (Ch.ChallengerKey.from_json(json.loads(p.read_text()))
                                for p in sorted(ch_dir.glob("*/out/key.json")))
                    if hashlib.sha512(k.public).hexdigest() == pin), None)
        if key is None:
            raise SystemExit(f"auditor_line: no session's key.json under {ch_dir} is the pinned key {pin[:16]}…")
        lines, windows, periods, doc["ledger"] = ledger_check(heads, key, period, lane)
        doc["ledger"]["heads_read"] = [str(p) for p in heads]
    signed = ((((periods or {}).get(period) or {}).get("open") or {}).get("body") or {}).get("model")
    if isinstance(signed, dict):
        bound = bound_check(v, signed, f"period {period}'s open")
    else:
        bound = bound_check(v, model, None if model is None else "--model-pin")
    doc["bound"] = bound
    rows, measured = [], {}
    for s in v["sessions"]:
        k, c = s["k"], s.get("challenger") or {}
        R = coins.window(D, k)["root"]
        row = {"k": k, "accepted": s["accepted"], "failure": s.get("failure"), "pace": s.get("pace"),
               "challenger": c.get("challenger"), "challenger_session": c.get("session"),
               "verifier_coin_check": c.get("reason"), "sketch": V.sketch_at(s, reg)}
        rows.append(row)
        if c.get("challenger") == "gate":
            row["ledger"] = session_check(s, k, R, windows, periods, period=period, lane=lane, peer=peer, pin=pin)
            row["credit"] = credit_check(s, R, lines, windows, periods, key=key, period=period, desc=desc,
                                         commit=commit, pins=pins, main=main, bound=bound)
            row["no_relay"] = lockdown_check(lockdown, signed_span(windows.get(R)))
            d = ch_dir / c["session"] / "out"
        elif c.get("challenger") == "serve":
            row["credit"] = no_credit("a served challenger (the verifier's machine) has no auditor's ledger, so no "
                                      "period credits its window", bound)
            d = Path(c["dir"]) if Path(c["dir"]).is_dir() else ch_dir / c["session"]
        else:
            row["credit"] = no_credit("its coins were drawn in the verifier's process: no challenger, no ledger", bound)
            row["skipped"] = "no challenger log: coins drawn in the verifier's process"
            continue
        try:
            log, row["trust"] = signed_lines(d, c["challenger"], R, heads or [], key, period)
        except (Ch.CoinError, OSError, ValueError, KeyError) as e:
            row["refused"] = f"the challenger's session is not readable as signed: {e}"
            row["credit"] = {**row["credit"], "ok": False, "zero": [*row["credit"]["zero"], row["refused"]]}
            continue
        row["log"] = log_checks(log, R)
        good = row.get("ledger", {"ok": True})["ok"] and not row["log"]["coin_before_its_root"] \
            and row["log"]["clock_monotone"] and row["log"]["sigma0_after_window_root"] and c.get("reason") is None
        row["of_record"] = bool(good and c.get("challenger") == "gate")
        if c.get("challenger") == "gate" and not good:
            why = ("not of record: its ledger entries, its release log or its coins fail the checks beside it (ledger, "
                   "log, verifier_coin_check)")
            row["credit"] = {**row["credit"], "ok": False, "zero": [*row["credit"]["zero"], why]}
        if not s.get("pace"):
            continue
        t = f"T={s['pace']['t_ms'] / 1000:g}s"
        row["T"] = t
        if not s["accepted"]:
            continue
        seen, row["named"] = auditor_seen(log, s["windows_seen"], s.get("groups_per_coin", 1))
        if boundary is not None and c.get("challenger") == "gate":
            row["credit"] = with_check(row["credit"], BOUNDARY, boundary_check(boundary, seen))
        a = copy.deepcopy(s)
        a["lean_record"], a["windows_seen"] = V.lean_record(s), seen
        row["groups"] = len(seen)
        tr = transcript_of(out, s)
        row["line_of"] = "transcript" if tr is not None and lean is not None else "record"
        row["shares_auditor_clock"] = shares(a, reg, host_bytes, rates, lean, clock="challenger", transcript=tr)
        row["shares_verifier_clock"] = shares(s, reg, host_bytes, rates, lean, transcript=tr)
        row["error"] = session_error(s)
        row["cross_check"] = cross_check(s["windows_seen"], seen)
        for clock in ("auditor", "verifier"):
            for f, x in row[f"shares_{clock}_clock"].items():
                cur = measured.setdefault(t, {}).setdefault(clock, {}).get(f)
                if cur is None or x["time"] < cur["time"]:
                    measured[t][clock][f] = {"k": k, **{name: x[name] for name in SHARES},
                                             "of_record": row["of_record"], "credited": row["credit"]["ok"]}
    for r in rows:
        if boundary is not None and r["challenger"] == "gate" and BOUNDARY not in r["credit"]:
            r["credit"] = with_check(r["credit"], BOUNDARY, {"ok": False, "reason": (
                "no credited span to hold to the certifiers' records: not an accepted paced session read as signed")})
    doc["sessions"], doc["measured"] = rows, measured
    doc["line"], doc["credit"] = credited_line(rows)
    doc["conditional_on"] = conditional_on(rows)
    doc["sketch_outside"] = sorted({r["sketch"]["outside"] for r in rows if r["sketch"]})
    return doc


def no_credit(why: str, bound: dict | None = None) -> dict:
    out = {c: {"ok": False, "reason": why} for c in CHECKS}
    if bound is not None:
        out["bound"] = bound
    return {**out, "ok": False, "zero": [f"not credited: {why}"]}


def credited_line(rows: list[dict]) -> tuple[dict, dict]:
    """The line of record, on the auditor's clock: at each T and in each form the least share of weights held among
    the credited sessions there (exact, the first on a tie), with that session's filler held beside it, or zero with
    every session's reasons, and its error (`union_error` of those sessions); and who earned credit, who didn't and
    why, in a sentence, with the error of the line over every T."""
    out: dict = {}
    lined = []
    for t in sorted({r["T"] for r in rows if r.get("T")}):
        at = [r for r in rows if r.get("T") == t]
        ok = [r for r in at if r["credit"]["ok"]]
        lined += ok
        if ok:
            out[t] = {f: min(({"k": r["k"], **{n: r["shares_auditor_clock"][f][n] for n in (*SHARES, "exact")}}
                              for r in ok), key=lambda x: line.number(x["exact"]["time"])) for f in line.FORMS}
            out[t]["credited_sessions"] = [r["k"] for r in ok]
        else:
            out[t] = {**{f: {**dict.fromkeys(SHARES, 0.0), "exact": dict.fromkeys(SHARES, "0")} for f in line.FORMS},
                      "credited_sessions": [], "zero": {r["k"]: r["credit"]["zero"] for r in at}}
        out[t]["error"] = union_error(ok)
    credited = [r["k"] for r in rows if r["credit"]["ok"]]
    zero = {r["k"]: r["credit"]["zero"] for r in rows if not r["credit"]["ok"]}
    if not credited:
        said = "the line credits zero: no session is credited, complete, accepted, pinned and bound"
    else:
        said = (f"the line credits session(s) {', '.join(map(str, credited))}"
                + (f"; session(s) {', '.join(map(str, zero))} earn none" if zero else ""))
    return out, {"credited_sessions": credited, "zero": zero, "error": union_error(lined), "summary": said + "".join(
        f"; session {k}: {'; '.join(why)}" for k, why in zero.items())}


def table(doc: dict) -> str:
    """`analyse`'s doc as Markdown: each session's five checks (`CHECKS`), with the reason of each that failed, and
    its credit, weights held then filler held; then the line at each T, time-held (labelled as under Identity, which
    nothing the verifier checks establishes) / link-charged with the snapshot forms, weights and filler apart, and its
    error when it credits a session."""
    b = line.FORMS
    checks = (*CHECKS, BOUNDARY) if doc["boundary"]["checked"] else CHECKS
    out = ["| session | T | " + " | ".join(checks) + " | credit |", "|---|---|" + "---|" * len(checks) + "---|"]
    for s in doc["sessions"]:
        c = s["credit"]
        cells = ["-" if n not in c else "yes" if c[n]["ok"] else f"no: {c[n]['reason']}" for n in checks]
        mine = {x: (s.get("shares_auditor_clock") or {}).get(x) for x in b}
        credit = ("credited: weights " + " / ".join(f"{mine[x]['time']:.4%}" for x in b) + ", filler "
                  + " / ".join(f"{mine[x]['filler_time']:.4%}" for x in b) if c["ok"] and all(mine.values())
                  else "zero")
        out.append(f"| {s['k']} | {s.get('T') or '-'} | " + " | ".join(cells) + f" | {credit} |")
    out.append("")
    for t, row in doc["line"].items():
        out.append(f"line {t}: credited {row['credited_sessions']}, time-held (under Identity: no block over the "
                   "boundary links) / link-charged weights "
                   + " / ".join(f"{row[x]['time']:.4%}" for x in b) + ", filler "
                   + " / ".join(f"{row[x]['filler_time']:.4%}" for x in b) + " (snapshot weights "
                   + " / ".join(f"{row[x]['snapshot']:.4%}" for x in b) + ", filler "
                   + " / ".join(f"{row[x]['filler_snapshot']:.4%}" for x in b) + ")"
                   + (f", error {row['error']['total']:.3g} (2^{row['error']['log2']:.2f}): the ledger's δ "
                      "(profile δ + P·2^-slack + Σ_t hide_t + bind + δs + Σ_t k_t/2^64) summed over the credited sessions, "
                      "beside "
                      "Σ_t Adv(prf t) (claim prf/sha-512), named and in no total"
                      if row["error"]["sessions"] else ""))
    out.append(f"line by {doc['line_by']}; {doc['credit']['summary']}")
    out += [f"conditional on {name}: {what}" for name, what in doc["conditional_on"].items()]
    out += [f"no relay, session {s['k']}: " + (
        f"{s['no_relay']['grounded_by']}, its span inside the lockdown applied {' to '.join(s['no_relay']['interval'])}"
        if s["no_relay"]["grounded_by"] else f"conditional on NoOutsideCompute, {s['no_relay']['reason']}")
        for s in doc["sessions"] if "no_relay" in s]
    out.append(f"boundary: held to the certifiers' records {doc['boundary']['sha512'][:16]}... of "
               f"{', '.join(doc['boundary']['links'])}" if doc["boundary"]["checked"] else
               f"boundary: not held to certifiers' records, {doc['boundary']['reason']}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the verifier's out/ (verifier.json, challenger/)")
    ap.add_argument("--host-bytes", type=int, required=True, help="the node's host DRAM, bytes (verifier.py's)")
    ap.add_argument("--pin", help="SHA-512 of the challenger's public key (a gate run)")
    ap.add_argument("--period", help="the audit's period on the ledger (a gate run)")
    ap.add_argument("--ledger", type=Path, action="append",
                    help="a published head of the auditor's ledger, read with each session's (challenger/*.ledger.jsonl): "
                         "all are checked to extend one another, and the longest is read")
    ap.add_argument("--commit", help="the verifier run's recorded commit (its attempt's source.commit): its "
                    f"{VERIFIER_TREE} tree names the verifier a gate run's period must have committed")
    ap.add_argument("--repo", type=Path, default=HERE.parents[1], help="a git checkout holding --commit")
    ap.add_argument("--pins", type=Path, help="the challenger host's pins log as the gate served it (`pins`), a line a "
                    "pin: without it no session is pinned")
    ap.add_argument("--main", type=Path, metavar="FILE", help="the auditor's own first-parent list of main, a commit a "
                    "line (`git rev-list --first-parent origin/main` in its checkout): without it no session is pinned")
    ap.add_argument("--model-pin", type=Path, metavar="FILE",
                    help="the model the store must be bound to when the audit's period signs none: a pin in the "
                         "auditor's model registry's shape (`remote_bandwidth.store.read_pin`); without either no session is "
                         "bound")
    ap.add_argument("--table", type=Path, help="write each session's checks and the line here, as Markdown")
    ap.add_argument("--lane", default=LANE)
    ap.add_argument("--peer", help="override the gate sessions' challenger peer, which is otherwise the run's record's "
                    "(`peer_of`: verifier.json's, else a host run's request.json beside OUT)")
    ap.add_argument("--disk-bytes-per-s", type=float, default=RATES["disk_bytes_per_s"], metavar="B/S",
                    help="the ceiling into node memory both forms charge (default: the NICs', `RATES`)")
    ap.add_argument("--link-bytes-per-s", type=float, default=RATES["link_bytes_per_s"], metavar="B/S",
                    help="the boundary links' ceiling the link-charged form adds (default: 400 Gb/s, `RATES`)")
    ap.add_argument("--bursts", choices=BURSTS, default="stated",
                    help="charge `ledger.Node.stated`'s bursts beyond the rates (the line of record) or none (gate runs "
                         "up to 7)")
    ap.add_argument("--line-by", choices=LINE_BY, default="lean",
                    help="compute each line in the Lean verifier, held exactly to line.py's (the line of record), or in "
                         "line.py alone (a reading, not of record)")
    ap.add_argument("--lean-exe", type=Path, metavar="PATH",
                    help="the remote-audit-verifier executable (default: `lake build` it in the remote audit's package)")
    ap.add_argument("--boundary", type=Path, metavar="FILE",
                    help="the network certifiers' records of the gate node's boundary links (`read_boundary`): refuse "
                         "a session's credit when they show ingress other than the verifier's messages over its span "
                         "(off by default)")
    ap.add_argument("--boundary-pin", metavar="SHA512", help="SHA-512 of the certifier's public key --boundary's "
                    "records must verify under")
    ap.add_argument("--lockdown", type=Path, metavar="FILE",
                    help="infra's record of the provider's lockdown of the gate node (`read_lockdown`): a gate "
                         "session whose signed span lies inside an applied interval is conditional on "
                         "ProviderLockdown in place of NoOutsideCompute (default: none is)")
    a = ap.parse_args(argv)
    if (a.boundary is None) != (a.boundary_pin is None):
        ap.error("--boundary and --boundary-pin go together")
    boundary = None if a.boundary is None else read_boundary(a.boundary, a.boundary_pin)
    commit, desc = None, None
    if a.commit is not None:
        commit, tree = verifier_tree(a.commit, a.repo)
        desc = descriptor(tree)
    pins = None if a.pins is None else [x for x in a.pins.read_text().split("\n") if x]
    first_parent = None if a.main is None else remote.main_lines(a.main)
    model = None
    if a.model_pin is not None:
        try:
            p = rstore.read_pin(a.model_pin)
        except rstore.Refused as e:
            raise SystemExit(f"auditor_line: {e}") from None
        model = {"name": p["name"], "checkpoint_sha512": p["checkpoint_sha512"]}
    lean = None
    if a.line_by == "lean":
        try:
            lean = lv.Lean(a.lean_exe or lv.build(), lambda _: None)
        except (lv.LeanError, OSError) as e:
            raise SystemExit(f"auditor_line: no Lean verifier to compute the line ({e}): pass --lean-exe, or "
                             "--line-by python for a reading that is not of record") from None
    try:
        doc = analyse(a.out, host_bytes=a.host_bytes, pin=a.pin, period=a.period, heads=a.ledger, lane=a.lane,
                      peer=a.peer, rates=rates_at(a.disk_bytes_per_s, a.link_bytes_per_s, a.bursts), desc=desc,
                      commit=commit, pins=pins, main=first_parent, model=model, lean=lean, boundary=boundary,
                      lockdown=None if a.lockdown is None else read_lockdown(a.lockdown))
    except (Disagree, lv.LeanError) as e:
        raise SystemExit(f"auditor_line: no line: {e}") from None
    finally:
        if lean is not None:
            lean.close()
    print(json.dumps(doc, indent=1, default=str))
    if a.table is not None:
        a.table.write_text(table(doc))
    print(f"auditor_line: line by {doc['line_by']}; {doc['credit']['summary']}", file=sys.stderr)
    print(f"auditor_line: conditional on {', '.join(doc['conditional_on'])} (its memory-line-of-record label: "
          f"`{line_label(doc)}`)", file=sys.stderr)
    bound = doc["bound"]
    print(f"auditor_line: bound: {'yes, to ' + bound['model']['name'] if bound['ok'] else 'no: ' + bound['reason']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
