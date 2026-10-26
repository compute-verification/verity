"""A recorded remote session replayed through both verifiers, `RemoteAudit.Session` (the definition, served by
`remote-audit-difftest` as `session`) and `audit.Session` (`ra_cases.session`), their verdicts and reasons compared.

A record is one of two kinds:

* A transcript: the difftest's case, every event the verifier's `Session` saw with the clock it read, its W + ⌈W/g⌉
  coins (g its `groups_per_coin`, 1 when absent) and its append coins. Both verifiers replay all of it and must give every event the same result, and the session the same
  outcome and record. The outcome is the verdict, and a rejection's failure, window and draw are its reason. Given
  the `verifier.json` that recorded it (`--record`), the outcome must be the recorded one. A transcript written before
  it named its session's pace (20 Oct) replays only with the pace its `verifier.json` gives, and only if its bytes are
  the ones whose sha256 that record keeps: replayed unpaced, a paced session's windows would be held to no cap on their
  roots.
* A remote session's `verifier.json` (`verity/remote-session/verifier/v0`). Per session it keeps the registration,
  the parameters, each window's challenge release, boundary and pass arrivals on the verifier's clock, its draws, and
  for an accepted session each pass's length and step. It keeps no roots, plans, coins or openings, and of appends and
  releases only their counts. `case_of` makes a case of what it keeps: each pass a root (synthetic) at its arrival, the
  registered store its plan (an accepted pass's recorded length when that is shorter) and its step the recorded one
  (else its count in the session), each boundary at its time and each draw closed at its last window at its time
  (every window at `groups_per_coin` 1), under synthetic coins and with no appends, releases or openings. A session
  recorded before the window was stated (13 Oct, 36e0ea654) may have a window shorter than its `window_ns`, a
  boundary today's `Session` refuses; its case states its shortest. A session drawn more than one window a coin at
  one window in flight isn't replayed: no property covers it, and the driver refuses it. Both verifiers replay that
  case as above, and `held` holds their verdict to the recorded one:
  - an accepted session draws every draw, each as many receipts as recorded over its windows;
  - one rejected for a failure the record keeps the cause of (`KEPT`: too few passes in a window) is rejected the same
    at the same window;
  - one rejected for any other failure is rejected nowhere in what is replayed, and draws as many receipts as recorded
    in each window it drew. Its cause is in what the record doesn't keep: its openings, plans, appends or steps, or
    the root it rejected, whose arrival the verifier records only once `Session.receive` takes it (so not a late one).
  Both verifiers also recompute the registration's digest from its fields, which must be the recorded one, unless the
  registration was recorded before it named its layout and rows (`LATER`), when the digest bound neither. A record
  that keeps the verifier's registry (`--registry`) has both verifiers read it back, which must write it out as
  recorded. A record of a registration the registry refused (a second one for a node with one live) has no session:
  its refusal replays on the registry as it stood before (the recorded one, less the refusal's count), where both
  must refuse the node again, by its live registration, and leave the recorded registry.

    uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py transcript CASE.json ... \
        [--record VERIFIER.json ...] [--out ROWS.json]
    uv run python verity/core/protocols/remote_bandwidth/lean/tests/replay.py record VERIFIER.json ... [--out ROWS.json]

One row a session (one for a record with none), printed and with `--out` written as JSON. Exit status 1 if the verifiers
differ on any row, a verdict isn't the recorded one, or a recorded digest or registry isn't the verifiers'.
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
from verity.core.protocols.remote_bandwidth import audit, coins, receipts

SESSION = D.Port(M.EXE, "session", M.session, None, package=M.PACKAGE, of_record=M.OF_RECORD)
REGISTRY = D.Port(M.EXE, "registry", M.registry, None, package=M.PACKAGE, of_record=M.OF_RECORD, rejects=M.REFUSALS)
#: The failures whose cause a `verifier.json` keeps: a window's passes.
KEPT = frozenset({"few"})
RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
#: `Registry.register`'s refusal, which a refused record keeps.
REFUSED = re.compile(r"node ([0-9a-f]+) has a live registration, ([0-9a-f]+)…")


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


def pace_json(session: dict) -> dict | None:
    """The pace a `verifier.json` session ran under (`audit.Pace`, as `params_json` writes it): its record's, else the
    session's own, either the `audit.Pace` the `--pace K` driver wrote or `pace_of`'s k and T (which
    `verifier.session` makes its `Pace`), else None; with `in_flight` when it is 1."""
    public = ((session.get("record") or {}).get("public") or {}).get("pace")
    p = public if public is not None else session.get("pace")
    if p is None:
        return None
    one = {"in_flight": 1} if p.get("in_flight", 2) == 1 else {}
    if public is not None:
        return {"k": public["k"], "window_ns": public["window_ns"]} | one
    return {"k": p["k"], "window_ns": p["window_ns"] if "window_ns" in p else round(p["t_ms"] * 1e6)} | one


def last_of(t: int, g: int, W: int) -> int:
    """The last window of window t's draw at `groups_per_coin` g: min((t // g + 1)g, W) − 1, t itself at g = 1."""
    return min((t // g + 1) * g, W) - 1


def case_of(reg: dict, session: dict) -> tuple[dict, dict]:
    """The case of what a `verifier.json` session keeps (the module's docstring), and how each part of it was made. Drawn
    `groups_per_coin` g windows a coin, each draw closes at its last window, at the time its draws were sent, under
    W + ⌈W/g⌉ coins."""
    W, N, g = session["windows"], reg["blocks"], session.get("groups_per_coin", 1)
    record = session.get("record") or {}
    profile = (record.get("profile") or {}).get("windows") or []
    lifetime = ((record.get("public") or {}).get("lifetime") or session.get("lifetime")
                or audit.lifetime_windows(session["window_ns"]))
    seen = session["windows_seen"]
    drawn = {last_of(int(t), g, W) for t in session.get("draws") or {}}
    failure = session.get("failure") or {}
    few = last_of(failure["t"], g, W) if failure.get("failure") == "few" and failure.get("t") is not None else None
    made = {"plans": "registered store", "steps": "recorded" if profile and "steps" in profile[0] else "counted"}
    made |= {k: f"{session[k]} recorded, none replayed" for k in ("appended", "released") if session.get(k)}
    window_ns = session["window_ns"]
    shortest = min((w["boundary_ns"] - w["released_ns"] for w in seen if w.get("boundary_ns") is not None),
                   default=window_ns)
    if shortest < window_ns:
        made["window_ns"] = f"{shortest}, its shortest window: recorded before the stated window"
        window_ns = shortest
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
        elif few == t and w.get("deadline_ns") is not None:
            at = w["deadline_ns"] + 1
        else:
            continue
        if type(at) is not int:
            raise ValueError(f"window {t} was drawn at no recorded time")
        timed.append((at, 2, t, 0, {"kind": "close", "t": t, "now": at}))
    start = seen[0]["released_ns"]
    events = [{"kind": "start", "now": start}] + [e for *_, e in sorted(timed, key=lambda x: x[:4])]
    case = {"registration": registration_of(reg), "not_before_ns": start, "append_coins": [],
            "coins": [_synthetic(b"replay coin %d" % j, 32) for j in range(W + coins.openings_coins(W, g))],
            "events": events,
            "params": {"deadline_ns": session["allowance_ns"], "deferred": False, "lifetime": lifetime,
                       "min_passes": 1, "pace": pace_json(session), "samples": session["samples"],
                       "window_ns": window_ns, "windows": W} | ({"groups_per_coin": g} if g != 1 else {})}
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
    g, W = session.get("groups_per_coin", 1), session["windows"]
    by_draw: dict[int, int] = {}
    for k, ds in (session.get("draws") or {}).items():
        by_draw[int(k) // g] = by_draw.get(int(k) // g, 0) + len(ds)
    want = [by_draw[c] for c in sorted(by_draw)]
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
    if session.get("accepted") and len(want) != coins.openings_coins(W, g):
        return False, f"an accepted session with {len(want)} of {coins.openings_coins(W, g)} draws drawn"
    if session.get("accepted"):
        return True, "every draw drawn, as many as recorded"
    return True, "no rejection in what the record keeps; its failure is in what it doesn't"


def run_of(path: Path) -> str:
    found = RUN_ID.findall(str(path))
    return found[-1] if found else str(path)


def registries(case: dict, s: D.Server) -> tuple[bool, dict | None]:
    """Both verifiers' registries on one case: whether they agree, and the Lean's answer."""
    python = REGISTRY.python(case)
    try:
        lean = REGISTRY.lean(s, case)
    except D.ProtocolError as e:
        lean = ("error", str(e))
    agree = "error" not in (lean[0], python[0]) and REGISTRY.key(lean) == REGISTRY.key(python)
    return agree, json.loads(lean[1]) if lean[0] == "output" else None


def read_back(registry: dict | None, s: D.Server) -> bool | None:
    """Whether both verifiers read a record's registry/v0 back and write it out as recorded; None if it keeps none."""
    if registry is None:
        return None
    agree, out = registries({"source": "00", "registry": registry, "ops": []}, s)
    return agree and out is not None and out["registry"] == registry


def refusal(v: dict, s: D.Server) -> dict:
    """A refused registration's record replayed: its node registered again on the registry as it stood before (the
    recorded one, the node's refusals one fewer), which both verifiers must refuse, leaving the recorded registry."""
    m, after = REFUSED.match(v["refused"]), v.get("registry")
    if m is None or after is None or m[1] not in after.get("refused", {}):
        return {"agree": None, "held": None, "why": "the record keeps no registry to replay its refusal in"}
    node, prefix = m[1], m[2]
    before = copy.deepcopy(after)
    before["refused"][node] -= 1
    if not before["refused"][node]:
        del before["refused"][node]
    op = {"op": "register", "node": node, "blocks": 1, "payload_root": "00" * 64, "codec": M.codec_json(M.P512),
          "receipt": receipts.DEFAULT, "bind": False, "layout": receipts.LAYOUT,
          "rows": receipts.sketch_rows(receipts.DEFAULT), "nonce": "00" * audit.NONCE_BYTES}
    agree, out = registries({"source": "00", "registry": before, "ops": [op]}, s)
    if not agree:
        return {"agree": False, "held": False, "why": "the verifiers differ"}
    live = after["live"].get(node, {}).get("registration", "")
    ok = (out["results"] == [{"refused": True}] and out["registry"] == after and live.startswith(prefix))
    return {"agree": True, "held": ok, "why": "refused again, by the node's live registration, the recorded registry "
            "after it" if ok else f"replayed {D.canonical(out['results'])}, the registry after it "
            f"{'as' if out['registry'] == after else 'not as'} recorded"}


def replay_record(path: Path, s: D.Server) -> list[dict]:
    v = json.loads(path.read_text())
    if v.get("schema") != "verity/remote-session/verifier/v0":
        raise ValueError(f"{path}: not a verity/remote-session/verifier/v0 record")
    if "registration" not in v and "refused" in v:
        return [{"run": run_of(path), "recorded": "registration refused", "registry": read_back(v.get("registry"), s)}
                | refusal(v, s)]
    registry = read_back(v.get("registry"), s)
    reg = registration_of(v["registration"])
    lean_digest, py_digest = digests(reg, s)
    earlier = [k for k in LATER if k not in v["registration"]]
    rows = []
    for session in v.get("sessions", []):
        row = {"run": run_of(path), "session": session.get("k"), "recorded": recorded(session),
               "windows": session["windows"], "passes": sum(w["passes"] for w in session.get("windows_seen") or []),
               "window_ns": session["window_ns"],
               "digest": None if earlier else lean_digest == py_digest == ("output", v["registration"]["digest"])
               and session.get("registration") == v["registration"]["digest"], "registry": registry}
        if not session.get("windows_seen"):
            rows.append(row | {"agree": None, "held": None, "why": "no window seen"})
            continue
        if session.get("groups_per_coin", 1) != 1 and (pace_json(session) or {}).get("in_flight") == 1:
            rows.append(row | {"agree": None, "held": None, "why": "drawn more than one window a coin "
                               "(groups_per_coin) at one window in flight, which no property covers"})
            continue
        case, made = case_of(v["registration"], session)
        if earlier:
            made["registration"] = f"recorded before it named {' and '.join(earlier)}, its digest not today's"
        row |= replay(case, s) | {"made": made}
        row["held"], row["why"] = held(session, row["lean"]) if row["agree"] else (False, "the verifiers differ")
        rows.append(row)
    return rows or [{"run": run_of(path), "recorded": "no session", "agree": None, "held": None,
                     "digest": None if earlier else lean_digest == py_digest == ("output", v["registration"]["digest"]),
                     "registry": registry, "why": "the registration ran no session"}]


def recorded_transcripts(records: list[Path]) -> dict[str, dict]:
    """Each transcript's file name, from the `verifier.json`s that recorded it, to its session's pace (`pace_json`),
    the sha256 of its bytes and the outcome recorded for it (accepted, a failure, or None for an unended session)."""
    known = {}
    for path in records:
        for session in json.loads(path.read_text()).get("sessions", []):
            t = session.get("transcript") or {}
            if "sha256" in t:
                f = session.get("failure") or {}
                known[Path(t["path"]).name] = {"pace": pace_json(session), "sha256": t["sha256"],
                                               "outcome": "accepted" if session.get("accepted") else f.get("failure")}
    return known


def replay_transcript(path: Path, s: D.Server, records: dict[str, dict] | None = None) -> list[dict]:
    """A transcript replayed through both verifiers, held to the outcome its `verifier.json` recorded when `records`
    (`recorded_transcripts`) has it. One that names no pace takes it from there, and only if its bytes are the
    recorded ones."""
    data = path.read_bytes()
    case, row = json.loads(data), {"file": str(path)}
    named = "pace" in case.get("params", {})
    sha256 = None if named else hashlib.sha256(data).hexdigest()
    del data
    known = (records or {}).get(path.name)
    if not named:
        if known is None:
            return [row | {"agree": None, "held": False, "why": "the transcript names no pace and no verifier.json "
                           "given records its session's: replayed unpaced, a paced session's windows would be held to "
                           "no cap on their roots"}]
        if sha256 != known["sha256"]:
            return [row | {"agree": None, "held": False,
                           "why": "not the recorded transcript: sha256 differs from its verifier.json's"}]
        case["params"]["pace"] = known["pace"]
        row["pace"] = f"{D.canonical(known['pace'])}, from its verifier.json"
    row |= replay(case, s)
    row["held"] = row["agree"] and "outcome" in row["lean"]
    if known is not None:
        row["recorded"] = known["outcome"]
        if row["held"] and row["lean"]["outcome"] != known["outcome"]:
            row["held"], row["why"] = False, f"replayed {row['lean']['outcome']}, the record {known['outcome']}"
    return [row]


def line(row: dict) -> str:
    who = row.get("run", row.get("file"))
    v = row.get("lean", {})
    got = ("nothing" if not v else "accepted" if v.get("outcome") == "accepted" else
           f"{v['rejected']['failure']} at window {v['rejected']['window']}" if v.get("rejected") else
           f"drew {len(v['drawn'])} windows" if "drawn" in v else next(iter(v.items()), ("?", ""))[0])
    head = (f"{who} #{row['session']}: " if "session" in row else f"{who}: ") + (
        f"recorded {row['recorded']}; " if "recorded" in row else "")
    return (f"{head}replayed {got}; agree {row['agree']}, held {row['held']}"
            + (f", digest {row['digest']}" if "digest" in row else "")
            + (f", registry {row['registry']}" if row.get("registry") is not None else "")
            + (f" ({row['why']})" if row.get("why") else "")
            + (f"\n    difference: {D.canonical(row['difference'])}" if row.get("difference") else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=("transcript", "record"))
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--record", nargs="+", type=Path, default=[], metavar="VERIFIER.json",
                    help="the verifier.json records of the transcripts, for the pace of one that names none")
    ap.add_argument("--out", type=Path)
    ns = ap.parse_args(argv)
    records = recorded_transcripts(ns.record)
    rows: list[dict] = []
    with server() as s:
        for path in ns.files:
            try:
                got = replay_transcript(path, s, records) if ns.kind == "transcript" else replay_record(path, s)
            except (ValueError, KeyError, TypeError) as e:
                got = [{"run": run_of(path), "agree": None, "held": False, "why": f"unreadable: {type(e).__name__}: {e}"}]
            for row in got:
                print(line(row), flush=True)
            rows += got
    if ns.out:
        ns.out.write_text(json.dumps(rows, indent=1) + "\n")
    bad = [r for r in rows if False in (r.get("agree"), r.get("held"), r.get("digest"), r.get("registry"))]
    refusals = [r for r in rows if r.get("recorded") == "registration refused" and r.get("agree") is not None]
    print(f"{sum(1 for r in rows if 'session' in r and r.get('agree') is not None)} recorded sessions, "
          f"{sum(1 for r in rows if 'file' in r and r.get('agree') is not None)} transcripts and {len(refusals)} "
          f"refused registrations replayed; "
          f"{sum(1 for r in rows if r.get('agree') is False)} where the verifiers differ; "
          f"{sum(1 for r in rows if r.get('held') is False)} not the recorded verdict; "
          f"{sum(1 for r in rows if r.get('agree') is None)} records with no verdict to replay; "
          f"{sum(1 for r in rows if r.get('registry') is not None)} rows of a recorded registry, "
          f"{sum(1 for r in rows if r.get('registry') is False)} not read back as recorded")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
