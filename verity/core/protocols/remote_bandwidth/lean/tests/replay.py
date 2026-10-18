"""A recorded remote session replayed through both verifiers, `RemoteAudit.Session` (the definition, served by
`remote-audit-difftest` as `session`) and `audit.Session` (`ra_cases.session`), their verdicts and reasons compared.

A record is one of two kinds:

* A transcript: the difftest's case, every event the verifier's `Session` saw with the clock it read, its 2W coins and
  its append coins. Both verifiers replay all of it and must give every event the same result, and the session the same
  outcome and record. The outcome is the verdict, and a rejection's failure, window and draw are its reason.
* A remote session's `verifier.json` (`verity/remote-session/verifier/v0`). Per session it keeps the registration,
  the parameters, each window's challenge release, boundary and pass arrivals on the verifier's clock, its draws, and
  for an accepted session each pass's length and step. It keeps no roots, plans, coins or openings, and of appends and
  releases only their counts. `case_of` makes a case of what it keeps: each pass a root (synthetic) at its arrival, the
  registered store its plan (an accepted pass's recorded length when that is shorter) and its step the recorded one
  (else its count in the session), each boundary at its time and each window drawn at its time, under synthetic coins
  and with no appends, releases or openings.
  Both verifiers replay that case as above, and `held` holds their verdict to the recorded one:
  - an accepted session draws every window, each as many receipts as recorded;
  - one rejected for a failure the record keeps the cause of (`KEPT`: too few passes in a window) is rejected the same
    at the same window;
  - one rejected for any other failure is rejected nowhere in what is replayed, and draws as many receipts as recorded
    in each window it drew. Its cause is in what the record doesn't keep: its openings, plans, appends or steps, or
    the root it rejected, whose arrival the verifier records only once `Session.receive` takes it (so not a late one).
  Both verifiers also recompute the registration's digest from its fields, which must be the recorded one, unless the
  registration was recorded before it named its layout and rows (`LATER`), when the digest bound neither.

    uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py transcript CASE.json ...
    uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py record VERIFIER.json ... [--out ROWS.json]

One row a session, printed and with `--out` written as JSON. Exit status 1 if the verifiers differ on any session, or a
verdict isn't the recorded one.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import audit

SESSION = D.Port(M.EXE, "session", M.session, None, package=M.PACKAGE)
#: The failures whose cause a `verifier.json` keeps: a window's passes.
KEPT = frozenset({"few"})
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")


def server() -> D.Server:
    return D.Server(SESSION.argv())


def _digest_py(reg: dict) -> D.Outcome:
    try:
        with M.decoding():
            return "output", M.registration_of(copy.deepcopy(reg)).digest.hex()
    except D.Reject as e:
        return "reject", str(e)


def digests(reg: dict, s: D.Server) -> tuple[D.Outcome, D.Outcome]:
    """The registration's digest by each verifier: Lean's, then Python's."""
    kind, text = s.call("registration_digest", reg)
    return (kind, json.loads(text) if kind == "output" else text), _digest_py(reg)


def summary(outcome: D.Outcome) -> dict:
    """A session's verdict and reasons from one verifier's answer: its outcome (None while a window is unverified), the
    first rejection (failure, window, draw), how many calls it refused, and each window's draw count."""
    kind, text = outcome
    if kind != "output":
        return {kind: text[:500]}
    out = json.loads(text)
    results = out["results"]
    rejected = next(({"failure": r["failure"], "window": r["window"], "at": r["at"]} for r in results
                     if "failure" in r), None)
    return {"outcome": out["outcome"], "rejected": rejected, "refused": sum(1 for r in results if "refused" in r),
            "drawn": [len(r["draws"]) for r in results if "draws" in r]}


def difference(case: dict, lean: D.Outcome, python: D.Outcome) -> dict:
    """Where the two answers first differ: the event and each side's result, or else the outcome and record."""
    if "output" != lean[0] or "output" != python[0]:
        return {"lean": f"{lean[0]} {lean[1][:300]}", "python": f"{python[0]} {python[1][:300]}"}
    a, b = json.loads(lean[1]), json.loads(python[1])
    for k, (x, y) in enumerate(zip(a["results"], b["results"])):
        if x != y:
            e = case["events"][k]
            return {"event": k, "kind": e["kind"], "lean": D.canonical(x)[:300], "python": D.canonical(y)[:300]}
    if len(a["results"]) != len(b["results"]):
        return {"results": [len(a["results"]), len(b["results"])]}
    return {"lean": D.canonical({k: a[k] for k in ("outcome", "record")})[:600],
            "python": D.canonical({k: b[k] for k in ("outcome", "record")})[:600]}


def replay(case: dict, s: D.Server) -> dict:
    """Both verifiers on one case: each one's verdict and reasons, whether they agree and, if not, where they part."""
    python = SESSION.python(case)
    try:
        lean = SESSION.lean(s, case)
    except D.ProtocolError as e:
        lean = ("error", str(e))
    agree = "error" not in (lean[0], python[0]) and SESSION.key(lean) == SESSION.key(python)
    row = {"lean": summary(lean), "python": summary(python), "agree": agree}
    if not agree:
        row["difference"] = difference(case, lean, python)
    return row


# --- a verifier.json ------------------------------------------------------------------------------------------------

def codec_of(name: str) -> dict:
    """A codec's JSON from its name (`Codec.name`: `p1024/k3/masked`, `p512/k2`)."""
    field, rounds, *masked = name.split("/")
    if not rounds.startswith("k") or masked not in ([], ["masked"]):
        raise ValueError(f"not a codec's name: {name}")
    return {"field": field, "masked": masked == ["masked"], "rounds": int(rounds[1:])}


#: What a registration has named, and its digest bound, since 13 Oct 09:24Z (36e0ea654). One recorded before names
#: neither: it had the one layout and its kind's rows, and its digest is one neither verifier computes now.
LATER = ("layout", "rows")


def registration_of(reg: dict) -> dict:
    return {"bind": reg["bind"], "blocks": reg["blocks"], "codec": codec_of(reg["codec"]), "codec_seed": reg["codec_seed"],
            "layout": reg.get("layout", receipts.LAYOUT), "node": reg["node"], "payload_root": reg["payload_root"],
            "receipt": reg["receipt"], "rows": reg["rows"] if "rows" in reg else receipts.sketch_rows(reg["receipt"])}


def _synthetic(label: bytes, n: int) -> str:
    return hashlib.sha512(label).digest()[:n].hex()


def case_of(reg: dict, session: dict) -> tuple[dict, dict]:
    """The case of what a `verifier.json` session keeps (the module's docstring), and how each part of it was made."""
    W, N = session["windows"], reg["blocks"]
    record = session.get("record") or {}
    profile = (record.get("profile") or {}).get("windows") or []
    lifetime = (record.get("public") or {}).get("lifetime") or audit.LIFETIME
    seen = session["windows_seen"]
    drawn = {int(t) for t in session.get("draws") or {}}
    failure = session.get("failure") or {}
    made = {"plans": "registered store", "steps": "recorded" if profile and "steps" in profile[0] else "counted"}
    made |= {k: f"{session[k]} recorded, none replayed" for k in ("appended", "released") if session.get(k)}
    timed: list[tuple[int, int, int, int, dict]] = []
    step = 0
    for t, w in enumerate(seen):
        lengths = profile[t]["lengths"] if t < len(profile) else None
        steps = profile[t].get("steps") if t < len(profile) else None
        if len(w["arrivals_ns"]) != w["passes"] or (lengths is not None and len(lengths) != w["passes"]):
            raise ValueError(f"window {t}: {w['passes']} passes, {len(w['arrivals_ns'])} arrivals")
        for i, now in enumerate(w["arrivals_ns"]):
            n = lengths[i] if lengths is not None else N
            if n < N:
                made["plans"] = "recorded lengths"
            timed.append((now, 1, t, i, {"kind": "receive", "t": t, "i": i,
                                         "root": _synthetic(b"replay root %d %d" % (t, i), 64),
                                         "runs": None if n >= N else [[0, n]], "change": None,
                                         "step": steps[i] if steps else step, "now": now}))
            step += 1
        if w.get("boundary_ns") is not None:
            timed.append((w["boundary_ns"], 0, t, 0, {"kind": "boundary", "now": w["boundary_ns"]}))
        if t in drawn:
            at = profile[t]["opened_ns"] if t < len(profile) else w["draws_sent_ns"]
        elif failure.get("failure") == "few" and failure.get("t") == t and w.get("deadline_ns") is not None:
            at = w["deadline_ns"] + 1
        else:
            continue
        if type(at) is not int:
            raise ValueError(f"window {t} was drawn at no recorded time")
        timed.append((at, 2, t, 0, {"kind": "close", "t": t, "now": at}))
    start = seen[0]["released_ns"]
    events = [{"kind": "start", "now": start}] + [e for *_, e in sorted(timed, key=lambda x: x[:4])]
    case = {"registration": registration_of(reg), "not_before_ns": start, "append_coins": [],
            "coins": [_synthetic(b"replay coin %d" % j, 32) for j in range(2 * W)], "events": events,
            "params": {"deadline_ns": session["allowance_ns"], "deferred": False, "lifetime": lifetime,
                       "min_passes": 1, "samples": session["samples"], "window_ns": session["window_ns"],
                       "windows": W}}
    return case, made


def recorded(session: dict) -> str:
    f = session.get("failure")
    if session.get("accepted"):
        return "accepted"
    return f"{f['failure']} at window {f['t']}" + (f", draw {f['at']}" if f.get("at") else "") if f else "unended"


def held(session: dict, verdict: dict) -> tuple[bool, str]:
    """Whether a replayed record's verdict (`summary`) is the recorded one, as the module's docstring says, and why."""
    if "outcome" not in verdict:
        return False, "the case was refused"
    if verdict["refused"]:
        return False, f"{verdict['refused']} calls refused: the record's events are out of the session's turn"
    want = [len(session["draws"][k]) for k in sorted(session.get("draws") or {}, key=int)]
    f = session.get("failure")
    if f and f["failure"] in KEPT:
        r = verdict["rejected"]
        ok = r is not None and (r["failure"], r["window"]) == (f["failure"], f["t"])
        return ok, f"replayed {r['failure']} at window {r['window']}" if r else "replayed with no rejection"
    if verdict["rejected"] is not None:
        r = verdict["rejected"]
        return False, f"replayed {r['failure']} at window {r['window']}, where the record has {recorded(session)}"
    if verdict["drawn"] != want:
        return False, f"drew {verdict['drawn']}, the record {want}"
    if session.get("accepted") and len(want) != session["windows"]:
        return False, f"an accepted session with {len(want)} of {session['windows']} windows drawn"
    if session.get("accepted"):
        return True, "every window drawn, as many as recorded"
    return True, "no rejection in what the record keeps; its failure is in what it doesn't"


def run_of(path: Path) -> str:
    found = RUN_ID.findall(str(path))
    return found[-1] if found else str(path)


def replay_record(path: Path, s: D.Server) -> list[dict]:
    v = json.loads(path.read_text())
    if v.get("schema") != "verity/remote-session/verifier/v0":
        raise ValueError(f"{path}: not a verity/remote-session/verifier/v0 record")
    reg = registration_of(v["registration"])
    lean_digest, py_digest = digests(reg, s)
    earlier = [k for k in LATER if k not in v["registration"]]
    rows = []
    for session in v.get("sessions", []):
        row = {"run": run_of(path), "session": session.get("k"), "recorded": recorded(session),
               "windows": session["windows"], "passes": sum(w["passes"] for w in session.get("windows_seen") or []),
               "window_ns": session["window_ns"],
               "digest": None if earlier else lean_digest == py_digest == ("output", v["registration"]["digest"])
               and session.get("registration") == v["registration"]["digest"]}
        if not session.get("windows_seen"):
            rows.append(row | {"agree": None, "held": None, "why": "no window seen"})
            continue
        case, made = case_of(v["registration"], session)
        if earlier:
            made["registration"] = f"recorded before it named {' and '.join(earlier)}, its digest not today's"
        row |= replay(case, s) | {"made": made}
        row["held"], row["why"] = held(session, row["lean"]) if row["agree"] else (False, "the verifiers differ")
        rows.append(row)
    return rows


def replay_transcript(path: Path, s: D.Server) -> list[dict]:
    row = {"file": str(path)} | replay(json.loads(path.read_text()), s)
    row["held"] = row["agree"]
    return [row]


def line(row: dict) -> str:
    who = row.get("run", row.get("file"))
    v = row.get("lean", {})
    got = ("accepted" if v.get("outcome") == "accepted" else
           f"{v['rejected']['failure']} at window {v['rejected']['window']}" if v.get("rejected") else
           f"drew {len(v['drawn'])} windows" if "drawn" in v else next(iter(v.items()), ("?", ""))[0])
    head = f"{who} #{row['session']}: recorded {row['recorded']}; " if "session" in row else f"{who}: "
    return (f"{head}replayed {got}; agree {row['agree']}, held {row['held']}"
            + (f", digest {row['digest']}" if "digest" in row else "") + (f" ({row['why']})" if row.get("why") else "")
            + (f"\n    difference: {D.canonical(row['difference'])}" if row.get("difference") else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=("transcript", "record"))
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ns = ap.parse_args(argv)
    rows: list[dict] = []
    with server() as s:
        for path in ns.files:
            try:
                got = (replay_transcript if ns.kind == "transcript" else replay_record)(path, s)
            except (ValueError, KeyError, TypeError) as e:
                got = [{"run": run_of(path), "agree": None, "held": False, "why": f"unreadable: {type(e).__name__}: {e}"}]
            for row in got:
                print(line(row), flush=True)
            rows += got
    if ns.out:
        ns.out.write_text(json.dumps(rows, indent=1) + "\n")
    bad = [r for r in rows if r.get("agree") is False or r.get("held") is False or r.get("digest") is False]
    print(f"{len(rows)} sessions replayed; {sum(1 for r in rows if r.get('agree') is False)} where the verifiers "
          f"differ; {sum(1 for r in rows if r.get('held') is False)} not the recorded verdict")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
