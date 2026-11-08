"""The Lean verifier's caps on a recursive session (`Flock.RecBudget`), for the stagers: a V* session's
``registered.json`` names at most `MAX_PORTS` ports, each of a tree of at most `MAX_DEPTH` levels (`regs`), and its table
holds at most `MAX_INSTANCES` instances (`table`); V*'s list at the inner shape holds at most `MAX_STATEMENTS` statements
(`count`), the V* parts the recursion's soundness bound covers, and records at most `MAX_RECORDED` hm96 calls (`recorded`,
each statement's counted by `recorded_at`); and the inner statement's public file holds at most `MAX_INNER_ROWS`
instances and shared-row table rows (`inner`). `verify` refuses a session past any of them, so a stager refuses to write
one (`check`, `check_inner`), with the verifier's message."""
from __future__ import annotations

MAX_INSTANCES = 4096
MAX_PORTS = 512
MAX_DEPTH = 64
MAX_STATEMENTS = 16
MAX_RECORDED = 2 ** 32
MAX_INNER_ROWS = 131072
TAIL = "the recursion's call bound covers no more"


def log2ceil(n: int) -> int:
    """`Flock.log2ceil`: the levels of a tree over ``n`` leaves."""
    return 0 if n <= 1 else (n - 1).bit_length()


def regs(registered: dict) -> str | None:
    """`RecBudget.regs` on a ``registered.json`` object: its refusal, or None. Its ports in the verifier's order, by name."""
    ports = registered["ports"]
    if len(ports) > MAX_PORTS:
        return f"refused: a V* session's registered reads name {len(ports)} ports, more than {MAX_PORTS}: {TAIL}"
    for name in sorted(ports):
        leaves = ports[name]["leaves"]
        if log2ceil(leaves) > MAX_DEPTH:
            return (f"refused: a V* session's registered port {name} has {leaves} leaves, a tree of {log2ceil(leaves)} levels, "
                    f"more than {MAX_DEPTH}: {TAIL}")
    return None


def table(unit: str, instances: int) -> str | None:
    """`RecBudget.table` on a table of ``instances`` instances of the unit ``unit``: its refusal, or None."""
    if instances > MAX_INSTANCES:
        return f"{unit}: refused: a V* session's table of {instances} instances, more than {MAX_INSTANCES}: {TAIL}"
    return None


def count(statements: int) -> str | None:
    """`RecBudget.count` on V*'s list of ``statements`` statements: its refusal, or None."""
    if statements > MAX_STATEMENTS:
        return (f"refused: V*'s list at the verifier's inner shape holds {statements} statements, more than {MAX_STATEMENTS}: "
                f"the recursion's soundness bound covers at most {MAX_STATEMENTS} V* parts")
    return None


def recorded_at(kind: str, n: int = 0, own_rows: int = 0, acc_reads: int = 0) -> int:
    """`RecBudget.recordedAt`: the hm96 calls a V* statement of kind ``kind`` (``inner``, ``rec-open``,
    ``rec-open-stacked``, ``cap``, or another) over ``n`` instances records at most, two at each port and instance it
    reads: the ``own_rows`` rows its message-round ports read; an algebra part's `v`, `out`, `acc0` and `acc1` at its two
    instances, a level's `out`, `coef` and `dirs` at each opening (R's level's `out` alone), a RecCap's `top` and `out`
    at each node; and each of the ``acc_reads`` rows of `rec-acc` its units read, at the unit's `acc` and `acc_out`."""
    per = {"inner": 16, "rec-open": 6 * n, "rec-open-stacked": 2 * n, "cap": 4 * n}.get(kind, 0)
    return 2 * own_rows + per + 4 * acc_reads


def recorded_calls(sh) -> int:
    """`RecBudget.recordedCalls` at a plain inner shape (a `rec_algebra.Shape` with no RecSparse, R or link table): each
    level's RecOpen over its ``2 q`` openings, whose units read ``2 q`` rows of `rec-acc`; each of the algebra's parts, its
    message-round ports ``m<k>`` at both reps' rows; then RecCap over each later level's ``2 · 2^cap`` tops and over level
    0's ``2^cap`` nodes."""
    from . import rec_residuals as RR

    lv = sh.sched.levels
    calls = sum(recorded_at("rec-open", 2 * l.q, acc_reads=2 * l.q) for l in lv)
    for p in RR.parts(RR.spec(sh)):
        calls += recorded_at("inner", 2, own_rows=2 * sum(nm[:1] == "m" and nm[1:].isdigit() for nm, _ in p.S["ports"]))
    return calls + sum(recorded_at("cap", 2 << l.cap) for l in lv[1:]) + recorded_at("cap", 1 << lv[0].cap)


def recorded(calls: int) -> str | None:
    """`RecBudget.recorded` on V*'s list recording ``calls`` hm96 calls: its refusal, or None."""
    if calls > MAX_RECORDED:
        return (f"refused: V*'s list at the verifier's inner shape records {calls} hm96 calls, more than {MAX_RECORDED}: "
                f"{TAIL}")
    return None


def inner(instances: int, counts: list[int] | None = None) -> str | None:
    """`RecBudget.innerRows` on the inner statement's public file of ``instances`` instances and, a shared-row file, tables
    of ``counts`` rows: its refusal, or None."""
    if instances > MAX_INNER_ROWS:
        return f"refused: the inner statement's public file holds {instances} instances, more than {MAX_INNER_ROWS}: {TAIL}"
    if any(r > MAX_INNER_ROWS for r in counts or ()):
        return f"refused: the inner statement's public file has a shared-row table of more than {MAX_INNER_ROWS} rows: {TAIL}"
    return None


def check(registered: dict, unit: str, instances: int) -> None:
    """Raise ValueError with the verifier's refusal of a V* session it would refuse."""
    if why := regs(registered) or table(unit, instances):
        raise ValueError(why)


def check_inner(instances: int, counts: list[int] | None = None) -> None:
    """Raise ValueError with the verifier's refusal of an inner statement it would refuse."""
    if why := inner(instances, counts):
        raise ValueError(why)
