"""The memory line on the auditor's clock, the analysis of record for a gate run: each accepted paced session's line
(`remote_bandwidth.line`, exact, through `verifier.recorded_line`: time-held and link-charged, each read held over time
and at the passes) with its groups' times read from what the auditor's host signed rather than from the verifier's
stamps.

Group g's coin σ_g left the challenger at its release log's `released_at` for challenge coin g, and window g's last
root reached it at the first `received_at` of a request naming that root (`auditor_seen`). Openings coin g's request
names it (`audit.Session.close` asks for the coin after the window's last root), and a paced window is drawn at its
k-th root (its boundary leaves before it), so that request leaves as the root arrives; for k = 1, σ_(g+2)'s request names the same root too. Both
times are the challenger's monotonic clock in its release log, whose lines its close signs (`challenger.session_of`),
and whose close, record and refused hellos the auditor's ledger sealed at the session's end (`remote.trusted_session`).
So L_g = max(A_g, s_(g+1)) − s_g on the challenger's clock holds group g's reads: they come after its coin, whose seed
they take, and before its last root, which chains them.

Beside the times the line reads the verifier's record of the session (`verifier.json`): its verdict, each group's
passes and held bytes (`ledger.memory`), and its windows' receipts and draws, whose pooled bound
(`remote_bandwidth.pooled`) is its wrong receipts. The verifier runs on the audit's machine, not
the auditor's host, so these rest on its verdict, which the session's transcript (`verifier.py --transcript`) lets the
auditor replay offline (the remote audit's `lean/tests/replay.py transcript`). The verifier's stamps are read only to
cross-check the challenger's: each coin interval on both clocks, their rates, and each group's L on both.

The ledger check: its longest head verified under the pinned key (`ledger_head`, `verify_ledger`, `verify_periods`);
the period's lane, cap and circuit, and that it closed; each session reserved and opened by the lane for the peer under
the pin at epoch k, opened at its window root R_k, ended closed with no refused hello, inside the period; σ_0 asked
after R_k, and each coin released after its root's receipt. The peer is the verifier machine's name at the challenger
host, as the run's own record gives it (`peer_of`): vy-nebius-1 for a verifier there, vy-auditor-1 for a host run.
`--peer` overrides it, and a gate run whose record names none is refused. A session served by `--challenger serve`
(the verifier's machine) has no ledger: its log is read under the key its run wrote (`session_of`) and its line is not
the auditor's.

**Credit.** The line of record (`line`) counts a session only when all five hold, each reported per session
(`credit`), and gives any other session zero credit with the reasons:

- *credited*: its window reads through `challenger.credited_session` (its pin first, as *pinned* reads it) with the
  descriptor of the verifier that ran it, `verity/remote-bandwidth/v0 verifier tree <tree>`, `<tree>` being
  `git rev-parse COMMIT:verity/core/protocols/remote_bandwidth` of the run's recorded commit (`--commit`), so it counts
  only toward a period that committed that verifier (PRF-126);
- *complete*: its window's end, and its period's close, are complete: the challenger released every coin it committed
  to, in every window of the period (PRF-132);
- *accepted*: the Lean's verdict (the record's `verdict` lean or both, no disagreement) accepted it, since a refused
  session can still close complete;
- *pinned*: its window reads through `challenger.pinned_session` with the challenger host's pins log (`--pins`, its
  lines as the gate served them) and the auditor's own first-parent list of main (`--main FILE`, `git rev-list
  --first-parent origin/main`): its open sealed the log as it stood, one of the sealed lines records the open's
  `pinned`, and that commit is on main's first-parent line, a tree main's check ran on (RT4-INT-40, CMP-74). With no
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
their bursts (`rates_at`; `--bursts none` for the reading of gate runs up to 7). What every accepted paced session
measured, credited or not, is `measured`. `--table` writes each session's five checks and the line as Markdown.

    python auditor_line.py OUT --host-bytes B [--pin SHA512 --period P --commit SHA [--pins LOG --main FILE]]
        [--ledger HEAD ...] [--peer NAME] [--model-pin FILE] [--disk-bytes-per-s B/S] [--link-bytes-per-s B/S]
        [--bursts stated|none] [--table FILE]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import challengers as CH  # noqa: E402
import verifier as V  # noqa: E402
from verity.core.protocols.remote_bandwidth import audit, coins, ledger, line  # noqa: E402
from verity.core.protocols.remote_bandwidth import store as rstore  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as Ch  # noqa: E402
from verity.core.protocols.sampled_proofs import remote  # noqa: E402

RATES = {"disk_bytes_per_s": 4.4e9, "link_bytes_per_s": 5e10}
"""The node's ceilings a gate run's line is read at, bytes a second (`line.Rates`): the node's NICs into its memory,
which Identity leaves in, charged in both forms as `ledger.Node`'s disks are, and its boundary links beyond them,
link-charged (54.4e9 in all)."""
BURSTS = ("stated", "none")
"""What the line charges beyond the rates (`rates_at`): `ledger.Node.stated`'s bursts, the line of record's
(`verifier.ingress_bursts`), or none, the reading of gate runs up to 7, whose recorded lines it reproduces (a burst of 0
states the rate down to one nanosecond)."""
LANE = "memory-accounting"
VERIFIER_TREE = "verity/core/protocols/remote_bandwidth"
LEAN_VERDICTS = ("lean", "both")
CHECKS = ("credited", "complete", "accepted", "pinned", "bound")
"""What a session's credit needs, each reported per session in this order."""
SHARES = ("time", "snapshot", "filler_time", "filler_snapshot")
"""A session's shares of the node in a form (`shares`): weights held, over time and at the passes, and filler held in
the same two readings."""


def descriptor(tree: str) -> bytes:
    """The descriptor of the circuit a session of the verifier at git tree `tree` (`VERIFIER_TREE`'s) proves: what a
    period's `--circuit` is the SHA-512 of."""
    return f"{audit.PROTOCOL_ID} verifier tree {tree}".encode()


def verifier_tree(commit: str, repo: Path = HERE.parents[1]) -> str:
    """`VERIFIER_TREE`'s git tree at `commit`, the verifier run's recorded commit, read in `repo`'s git."""
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet",
                        f"{commit}^{{commit}}:{VERIFIER_TREE}"], capture_output=True, text=True)
    if p.returncode or len(p.stdout.strip()) != 40:
        raise SystemExit(f"auditor_line: {repo}'s git has no {VERIFIER_TREE} at commit {commit}: run where git has the "
                         "verifier run's commit (`git fetch origin COMMIT`), or pass --repo")
    return p.stdout.strip()


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


def auditor_seen(lines: list[str], seen: list[dict]) -> tuple[list[dict], dict]:
    """`seen` (a session's `windows_seen`) with each group's coin and last root on the challenger's clock: `released_ns`
    challenge coin g's release, `arrivals_ns` the first receipt of a request naming openings coin g's root, the
    window's last; and which family's request named each last root first."""
    fam = release_log(lines)
    ch, op = fam.get("challenge"), fam.get("openings")
    n = len(seen)
    if ch is None or op is None or len(ch["released"]) < n or len(op["after"]) < n:
        raise ValueError(f"the release log holds {len((ch or {}).get('released', ()))} challenge and "
                         f"{len((op or {}).get('after', ()))} openings coins, not one of each for each of {n} groups")
    first: dict[str, tuple[int, str]] = {}
    for label, f in fam.items():
        for root, t in zip(f["after"], f["received"]):
            if root not in first or t < first[root][0]:
                first[root] = (t, label)
    out, by = [], {}
    for g, w in enumerate(seen):
        t, label = first[op["after"][g]]
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


def shares(session: dict, registration: dict, host_bytes: int, rates: dict) -> dict:
    """A session's shares of the node in each form (`line.FORMS`) at `rates` (`rates_at`), on the clock its
    `windows_seen` read: weights and filler held, over time and at the passes (`SHARES`), in floats and each exact
    beside them (`exact`), and the worst stretch over time."""
    got = V.recorded_line(session, registration, host_bytes=host_bytes,
                          rates=V.recorded_rates(session, registration, **rates))
    row = {}
    for f in line.FORMS:
        exact = {"time": got.share(f), "snapshot": got.share(f, line.SNAPSHOT),
                 "filler_time": got.share(f, line.TIME, "filler"),
                 "filler_snapshot": got.share(f, line.SNAPSHOT, "filler")}
        worst = got.form(f).worst(line.TIME)
        row[f] = {**{k: float(x) for k, x in exact.items()}, "exact": {k: line.text(x) for k, x in exact.items()},
                  "worst_stretch": worst.stretch, "worst_l_max_ms": worst.l_max_ns / 1e6,
                  "worst_span_s": worst.span_ns / 1e9}
    return row


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
                 period: str, desc: bytes, pins: list[str] | None, main: list[str] | None, bound: dict) -> dict:
    """Whether session `s`, whose window root is R, earns credit: its window credited toward the audit's period for
    the verifier `desc` describes (`credited_session`), its window and its period closed complete, its verdict the
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
            main: list[str] | None = None, model: dict | None = None) -> dict:
    """The run's line: `desc` is the descriptor of the verifier that ran it (`descriptor` of `commit`'s tree), `pins`
    the challenger host's pins log, its lines (None: the gate served none), `main` the auditor's first-parent list of
    main (None: none given), `model` the model the store must be bound to ({name, checkpoint_sha512},
    `--model-pin`'s) when the audit's period signs none, `peer` the gate sessions' peer (None: the record's,
    `peer_of`), and `rates` the node's (`rates_at`; None: its defaults)."""
    rates = rates_at() if rates is None else rates
    v = json.loads((out / "verifier.json").read_text())
    reg = v["registration"]
    D = bytes.fromhex(reg["digest"])
    ch_dir = out / "challenger"
    kinds = {(s.get("challenger") or {}).get("challenger") for s in v["sessions"]}
    key = windows = periods = None
    lines: list[str] = []
    doc: dict = {"registration": reg["digest"], "host_bytes": host_bytes, "rates": rates,
                 "commit": commit, "descriptor": None if desc is None else desc.decode(),
                 "pins": None if pins is None else {"lines": len(pins), "sha512": Ch.log_digest(pins)},
                 "verifier_rtt_ms": {k: x / 1e6 for k, x in v.get("rtt", {}).items() if k.endswith("_ns")}}
    if "gate" in kinds:
        if pin is None or period is None or desc is None:
            raise SystemExit("auditor_line: a gate run's line needs --pin, --period and --commit (the verifier run's "
                             "recorded commit, whose verifier the period must have committed)")
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
               "verifier_coin_check": c.get("reason")}
        rows.append(row)
        if c.get("challenger") == "gate":
            row["ledger"] = session_check(s, k, R, windows, periods, period=period, lane=lane, peer=peer, pin=pin)
            row["credit"] = credit_check(s, R, lines, windows, periods, key=key, period=period, desc=desc, pins=pins,
                                         main=main, bound=bound)
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
        seen, row["named"] = auditor_seen(log, s["windows_seen"])
        a = copy.deepcopy(s)
        a["windows_seen"] = seen
        row["groups"] = len(seen)
        row["shares_auditor_clock"] = shares(a, reg, host_bytes, rates)
        row["shares_verifier_clock"] = shares(s, reg, host_bytes, rates)
        row["cross_check"] = cross_check(s["windows_seen"], seen)
        for clock in ("auditor", "verifier"):
            for f, x in row[f"shares_{clock}_clock"].items():
                cur = measured.setdefault(t, {}).setdefault(clock, {}).get(f)
                if cur is None or x["time"] < cur["time"]:
                    measured[t][clock][f] = {"k": k, **{name: x[name] for name in SHARES},
                                             "of_record": row["of_record"], "credited": row["credit"]["ok"]}
    doc["sessions"], doc["measured"] = rows, measured
    doc["line"], doc["credit"] = credited_line(rows)
    return doc


def no_credit(why: str, bound: dict | None = None) -> dict:
    out = {c: {"ok": False, "reason": why} for c in CHECKS}
    if bound is not None:
        out["bound"] = bound
    return {**out, "ok": False, "zero": [f"not credited: {why}"]}


def credited_line(rows: list[dict]) -> tuple[dict, dict]:
    """The line of record, on the auditor's clock: at each T and in each form the least share of weights held among
    the credited sessions there (exact, the first on a tie), with that session's filler held beside it, or zero with
    every session's reasons; and who earned credit, who didn't and why, in a sentence."""
    out: dict = {}
    for t in sorted({r["T"] for r in rows if r.get("T")}):
        at = [r for r in rows if r.get("T") == t]
        ok = [r for r in at if r["credit"]["ok"]]
        if ok:
            out[t] = {f: min(({"k": r["k"], **{n: r["shares_auditor_clock"][f][n] for n in (*SHARES, "exact")}}
                              for r in ok), key=lambda x: line.number(x["exact"]["time"])) for f in line.FORMS}
            out[t]["credited_sessions"] = [r["k"] for r in ok]
        else:
            out[t] = {**{f: {**dict.fromkeys(SHARES, 0.0), "exact": dict.fromkeys(SHARES, "0")} for f in line.FORMS},
                      "credited_sessions": [], "zero": {r["k"]: r["credit"]["zero"] for r in at}}
    credited = [r["k"] for r in rows if r["credit"]["ok"]]
    zero = {r["k"]: r["credit"]["zero"] for r in rows if not r["credit"]["ok"]}
    if not credited:
        said = "the line credits zero: no session is credited, complete, accepted, pinned and bound"
    else:
        said = (f"the line credits session(s) {', '.join(map(str, credited))}"
                + (f"; session(s) {', '.join(map(str, zero))} earn none" if zero else ""))
    return out, {"credited_sessions": credited, "zero": zero, "summary": said + "".join(
        f"; session {k}: {'; '.join(why)}" for k, why in zero.items())}


def table(doc: dict) -> str:
    """`analyse`'s doc as Markdown: each session's five checks (`CHECKS`), with the reason of each that failed, and
    its credit, weights held then filler held; then the line at each T, time-held / link-charged with the snapshot
    forms, weights and filler apart."""
    b = line.FORMS
    out = ["| session | T | " + " | ".join(CHECKS) + " | credit |", "|---|---|" + "---|" * len(CHECKS) + "---|"]
    for s in doc["sessions"]:
        c = s["credit"]
        cells = ["yes" if c[n]["ok"] else f"no: {c[n]['reason']}" for n in CHECKS]
        mine = {x: (s.get("shares_auditor_clock") or {}).get(x) for x in b}
        credit = ("credited: weights " + " / ".join(f"{mine[x]['time']:.4%}" for x in b) + ", filler "
                  + " / ".join(f"{mine[x]['filler_time']:.4%}" for x in b) if c["ok"] and all(mine.values())
                  else "zero")
        out.append(f"| {s['k']} | {s.get('T') or '-'} | " + " | ".join(cells) + f" | {credit} |")
    out.append("")
    for t, row in doc["line"].items():
        out.append(f"line {t}: credited {row['credited_sessions']}, time-held / link-charged weights "
                   + " / ".join(f"{row[x]['time']:.4%}" for x in b) + ", filler "
                   + " / ".join(f"{row[x]['filler_time']:.4%}" for x in b) + " (snapshot weights "
                   + " / ".join(f"{row[x]['snapshot']:.4%}" for x in b) + ", filler "
                   + " / ".join(f"{row[x]['filler_snapshot']:.4%}" for x in b) + ")")
    out.append(doc["credit"]["summary"])
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
                    "pin: without it no session is pinned (RT4-INT-40)")
    ap.add_argument("--main", type=Path, metavar="FILE", help="the auditor's own first-parent list of main, a commit a "
                    "line (`git rev-list --first-parent origin/main` in its checkout): without it no session is pinned "
                    "(CMP-74)")
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
    a = ap.parse_args(argv)
    desc = None if a.commit is None else descriptor(verifier_tree(a.commit, a.repo))
    pins = None if a.pins is None else [x for x in a.pins.read_text().split("\n") if x]
    first_parent = None if a.main is None else remote.main_lines(a.main)
    model = None
    if a.model_pin is not None:
        try:
            p = rstore.read_pin(a.model_pin)
        except rstore.Refused as e:
            raise SystemExit(f"auditor_line: {e}") from None
        model = {"name": p["name"], "checkpoint_sha512": p["checkpoint_sha512"]}
    doc = analyse(a.out, host_bytes=a.host_bytes, pin=a.pin, period=a.period, heads=a.ledger, lane=a.lane, peer=a.peer,
                  rates=rates_at(a.disk_bytes_per_s, a.link_bytes_per_s, a.bursts), desc=desc, commit=a.commit,
                  pins=pins, main=first_parent, model=model)
    print(json.dumps(doc, indent=1, default=str))
    if a.table is not None:
        a.table.write_text(table(doc))
    print(f"auditor_line: {doc['credit']['summary']}", file=sys.stderr)
    bound = doc["bound"]
    print(f"auditor_line: bound: {'yes, to ' + bound['model']['name'] if bound['ok'] else 'no: ' + bound['reason']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
