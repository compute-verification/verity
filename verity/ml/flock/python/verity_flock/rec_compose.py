"""`flock-verify rec-compose` in Python (`Flock/RecCompose.lean`, `RecCompose.check`), reason for reason, which
`tests/test_lean_rec_compose.py` holds to the Lean binary on honest and tampered part sets. The Lean is the verifier;
this is for writing part sets and their verdicts (``verdict``) in tests and stand-ins.

A recursive statement's V* parts each ran ``flock-verify verify --zk --rec-part J`` once. Their logs compose when there
is one log per V* statement with one ``VERDICT``; ``rec_part`` 0 to n - 1 each once; every ``VERDICT`` accepts, ran
``--zk``, names no identifier and the composition's own rows file's SHA-512; the first part's ``rec_shared`` holds every
input the compiled verifier's parts read at the composition's own inner shape, and every part's is that one; and no two
sessions share a coin key. Then the challenger's interactive-run record (``verity/rec-run/v1``, `RecCompose.runOf`,
written by `verity.core.protocols.sampled_proofs.challenger.RunRecord`) must be one run of them (`RecCompose.checkRun`,
C10 ``OneRun``): the staging first, at the composition's rows and the shared set every part read; exactly one session
opened per part and none for another; every opened session a composed part's and every part's opened; the close last."""
from __future__ import annotations

import hashlib
import json

#: ``rec_shared``'s fields in `RecCompose.Shared`'s order: key, and what it is (`Shared.diff`'s words)
SHARED = (
    ("rec_inner", "the verifier's inner shape, --rec-inner"),
    ("inner_circuit", "the inner statement's circuit, --rec-inner-circuit"),
    ("inner_public", "the inner statement's public file, --rec-inner-public"),
    ("inner_partition", "the inner statement's partition, --rec-inner-partition"),
    ("inner_coins", "a record of check 5's coins, --rec-inner-coins"),
    ("inner_session", "the inner session's record, --rec-inner-session"),
    ("inner_transcript", "the coin server's transcript, --rec-inner-transcript"),
    ("inner_registration", "the circuit registration, --rec-inner-registration"),
    ("inner_statement", "how the inner statement is proved, --rec-inner-statement"),
    ("inner_statement_coins", "the inner statement's coins, --rec-inner-statement-coins"),
    ("inner_tables", "the inner statement's lookup tables, --rec-inner-tables"),
    ("inner_draw", "the inner statement's draw, --rec-inner-draw"),
)


def digest(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def shared(**files) -> dict:
    """``rec_shared`` as ``verify`` writes it (`Shared.toJson`): each key of ``SHARED`` given as bytes (its SHA-512), a
    string (``inner_statement``), ``"os"``/``"seed"`` (``inner_statement_coins``) or ``[(name, bytes)]``
    (``inner_tables``); ``None`` when absent."""
    out = {}
    for k, _ in SHARED:
        v = files.get(k)
        if k == "inner_tables" and v is not None:
            v = [[n, digest(b)] for n, b in v]
        elif isinstance(v, bytes):
            v = digest(v)
        out[k] = v
    return out


def verdict(j: int, rows: bytes, rec_shared: dict, coin_key: str, accepted: bool = True, why: str = "",
            zk: bool = True, **extra) -> str:
    """A part's ``VERDICT`` line as ``verify --zk --rec-part J --rec-rows F`` writes it."""
    v = {"session": f"s{j}", "accepted": accepted, "why": why, "rec_part": j, "rec_rows_sha512": digest(rows),
         "rec_shared": rec_shared, "zk": zk, "coin_key": coin_key, **extra}
    return "VERDICT " + json.dumps(v, separators=(",", ":"))


def _nat(v) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


def _str(v) -> str | None:
    return v if isinstance(v, str) else None


def _read_shared(j) -> tuple | None:
    """`Shared.ofJson?`: the fields in order, or None where one is absent or not in its form."""
    if not isinstance(j, dict):
        return None
    out = []
    for k, _ in SHARED:
        if k not in j:
            return None
        v = j[k]
        if k == "inner_statement_coins":
            if v is not None and v not in ("os", "seed"):
                return None
            v = None if v is None else v == "os"
        elif k == "inner_tables":
            if v is not None:
                if not isinstance(v, list) or not all(isinstance(t, list) and len(t) == 2 and all(isinstance(x, str) for x in t)
                                                     for t in v):
                    return None
                v = tuple((t[0], t[1]) for t in v)
        elif v is not None and not isinstance(v, str):
            return None
        out.append(v)
    return tuple(out)


def _read(j) -> dict:
    """`Verdict.ofJson`."""
    o = j if isinstance(j, dict) else {}
    return {"accepted": o.get("accepted") is True, "why": _str(o.get("why")) or "", "part": _nat(o.get("rec_part")),
            "rows": _str(o.get("rec_rows_sha512")), "shared": _read_shared(o["rec_shared"]) if "rec_shared" in o else None,
            "zk": o.get("zk") if isinstance(o.get("zk"), bool) else None, "coin_key": _str(o.get("coin_key")),
            "identifier": "identifier" in o}


class Refused(Exception):
    pass


def _lean_str(k: str, v) -> str:
    """The value as `Shared.diff` shows it (Lean's `toString`, `none` when absent)."""
    if v is None:
        return "none"
    if k == "inner_statement_coins":
        return "true" if v else "false"
    if k == "inner_tables":
        return "[" + ", ".join(f"({n}, {d})" for n, d in v) + "]"
    return v


def _diff(s: tuple, t: tuple) -> str:
    for (k, what), a, b in zip(SHARED, s, t):
        if a != b:
            return f"its {k} ({what}) is {_lean_str(k, a)}, not {_lean_str(k, b)} as the first part's"
    return "its rec_shared is not the first part's"


def _part_of(name: str, log: str) -> tuple[str, dict]:
    vs = []
    for line in log.split("\n"):
        if line.startswith("VERDICT "):
            try:
                vs.append(json.loads(line[8:]))
            except ValueError as e:
                raise Refused(f"{name}: a VERDICT line that is not JSON: {e}") from None
    if not vs:
        raise Refused(f"{name}: no VERDICT")
    if len(vs) > 1:
        raise Refused(f"{name}: {len(vs)} VERDICTs: a V* part is one session (RecDeploy.V runs one per part)")
    return name, _read(vs[0])


def _check_one(n: int, rows: str, name: str, v: dict) -> None:
    if not v["accepted"]:
        raise Refused(f"{name}: a session is not accepted ({v['why']})")
    j = v["part"]
    if j is None:
        raise Refused(f"{name}: its VERDICT names no rec_part: not one of V*'s parts (--rec-part J)")
    if n <= j:
        raise Refused(f"rec_part {j}: V* has {n} statements")
    if v["zk"] is not True:
        raise Refused(f"{name}: not a --zk session: V*'s parts are masked (RecDeploy.insO), the only form the recursion's "
                      "proof reads")
    if v["identifier"]:
        raise Refused(f"{name}: its VERDICT names an identifier (--identifier): V*'s parts set up without one "
                      "(RecIns.setup), the only form the recursion's proof reads")
    h = v["rows"]
    if h is None:
        raise Refused(f"{name}: its VERDICT names no rec_rows_sha512: it was not checked against the shared rows (--rec-rows)")
    if h != rows:
        raise Refused(f"{name}: checked against the rows file {h}, not the one staged rows file {rows}")


def _held(name: str, what: str) -> str:
    return f"{name}: its verify held no {what}, which every part of the compiled verifier reads"


def _check_form(name: str, inner: str, s: tuple) -> None:
    rec_inner, circuit, public, partition, coins, session, transcript, _, statement, statement_coins, tables, _ = s
    if rec_inner is None:
        raise Refused(f"{name}: its VERDICT names no rec_inner: it was not verified at V*'s inner shape (--rec-inner)")
    if rec_inner != inner:
        raise Refused(f"{name}: verified at the inner shape {rec_inner}, not the composition's own --rec-inner {inner}")
    for v, what in ((circuit, "inner statement's circuit (--rec-inner-circuit)"),
                    (public, "inner statement's public file (--rec-inner-public)"),
                    (partition, "inner statement's partition (--rec-inner-partition)")):
        if v is None:
            raise Refused(_held(name, what))
    if coins is not None:
        raise Refused(f"{name}: it read a record of check 5's coins (--rec-inner-coins), which no part of the compiled "
                      "verifier reads")
    for v, what in ((session, "inner session's record (--rec-inner-session)"),
                    (transcript, "coin server's transcript (--rec-inner-transcript)"),
                    (statement, "word on how the inner statement is proved (--rec-inner-statement)"),
                    (statement_coins, "inner statement's coins (--rec-inner-statement)"),
                    (tables, "inner statement's lookup tables (--rec-inner-statement)")):
        if v is None:
            raise Refused(_held(name, what))


def _check_verdicts(n: int, rows: str, inner: str, vs: list[tuple[str, dict]]) -> None:
    if n == 0:
        raise Refused("no V* statement")
    for name, v in vs:
        _check_one(n, rows, name, v)
    if not vs:
        raise Refused(f"0 parts, not V*'s {n} statements")
    name, v = vs[0]
    sh = v["shared"]
    if sh is None:
        raise Refused(f"{name}: its VERDICT names no rec_shared: the inputs it shares with the other parts are not named")
    _check_form(name, inner, sh)
    for name, v in vs:
        if v["shared"] is None:
            raise Refused(f"{name}: its VERDICT names no rec_shared: the inputs it shares with the other parts are not named")
        if v["shared"] != sh:
            raise Refused(f"{name}: {_diff(v['shared'], sh)}")
    for j in range(n):
        c = sum(1 for _, v in vs if v["part"] == j)
        if c != 1:
            raise Refused(f"rec_part {j}: {c} logs, not one")
    if len(vs) != n:
        raise Refused(f"{len(vs)} parts, not V*'s {n} statements")
    for i, (name, v) in enumerate(vs):
        k = v["coin_key"]
        if k is None:
            raise Refused(f"{name}: its VERDICT names no coin_key: its session's coins are not named")
        if any(w["coin_key"] == k for _, w in vs[i + 1:]):
            raise Refused(f"{name}: its session's coin key is another part's: a session key is never reused "
                          "(CoinTree.checkFresh)")


RUN_KIND = "verity/rec-run/v1"


def _event(j) -> tuple | None:
    """`Event.ofJson?`: ``("stage", rows, shared)``, ``("open", session, part, coin key)`` or ``("close",)``."""
    o = j if isinstance(j, dict) else {}
    kind = o.get("event")
    if kind == "stage":
        rows, sh = _str(o.get("rec_rows_sha512")), _read_shared(o["rec_shared"]) if "rec_shared" in o else None
        return None if rows is None or sh is None else ("stage", rows, sh)
    if kind == "open":
        session, part, key = _str(o.get("session")), _nat(o.get("rec_part")), _str(o.get("coin_key"))
        return None if session is None or part is None or key is None else ("open", session, part, key)
    return ("close",) if kind == "close" else None


def _run_of(text: str) -> list[tuple]:
    """`RecCompose.runOf`: the record's events in order."""
    try:
        j = json.loads(text)
    except ValueError:
        raise Refused("run record: not JSON") from None
    if not isinstance(j, dict) or j.get("kind") != RUN_KIND:
        raise Refused(f"run record: not a {RUN_KIND} record")
    if not isinstance(j.get("events"), list):
        raise Refused("run record: no list of events")
    out = []
    for i, e in enumerate(j["events"]):
        ev = _event(e)
        if ev is None:
            raise Refused(f"run record: its event {i} is not a stage, an open or a close in its form")
        out.append(ev)
    return out


def _opens_of(evs: list[tuple]) -> list[tuple[str, int, str]]:
    """`RecCompose.opensOf`: the sessions opened up to the close, which is the last event."""
    opens = []
    for i, e in enumerate(evs):
        if e[0] == "open":
            opens.append(e[1:])
        elif e[0] == "stage":
            raise Refused("run record: a second staging: V*'s statements are staged once, before the first part's Hello")
        elif i != len(evs) - 1:
            raise Refused("run record: an event after its close")
        else:
            return opens
    raise Refused("run record: not closed: the challenger closes a run's record after its last session, and an open "
                  "record may gain sessions")


def _check_run(n: int, rows: str, vs: list[tuple[str, dict]], evs: list[tuple]) -> None:
    """`RecCompose.checkRun`."""
    if not evs or evs[0][0] != "stage":
        raise Refused("run record: its first event is not the staging: V*'s statements are staged before the first "
                      "part's Hello")
    _, r, sh = evs[0]
    if r != rows:
        raise Refused(f"run record: it staged the rows file {r}, not the composition's {rows}")
    for name, v in vs:
        if v["shared"] != sh:
            raise Refused(f"{name}: its rec_shared is not the one the run record staged before the first Hello")
    opens = _opens_of(evs[1:])
    for session, part, _ in opens:
        if part >= n:
            raise Refused(f"run record: session {session} was opened for part {part}: V* has {n} statements")
    for j in range(n):
        c = sum(1 for _, part, _ in opens if part == j)
        if c != 1:
            raise Refused(f"part {j}: {c} sessions opened in the run record, not one: a part is never retried")
    for session, part, key in opens:
        if not any(v["part"] == part and v["coin_key"] == key for _, v in vs):
            raise Refused(f"run record: session {session} was opened for part {part}, and it is not the session "
                          "composed for that part")
    for name, v in vs:
        if not any(v["part"] == part and v["coin_key"] == key for _, part, key in opens):
            raise Refused(f"{name}: its session is not one the run record opened for its part")


def check(n: int, rows: str, inner: str, run: str, parts: list[tuple[str, str]]) -> tuple[bool, str]:
    """`RecCompose.check` at V*'s ``n`` statements, the composition's own rows file's SHA-512 ``rows`` and inner
    shape's ``inner``, over the challenger's interactive-run record ``run`` (its text) and ``parts``' logs (name, text):
    (accepted, the refusal's reason or "")."""
    try:
        vs = [_part_of(name, log) for name, log in parts]
        _check_verdicts(n, rows, inner, vs)
        _check_run(n, rows, vs, _run_of(run))
    except Refused as e:
        return False, str(e)
    return True, ""
