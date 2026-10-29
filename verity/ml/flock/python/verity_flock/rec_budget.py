"""The Lean verifier's caps on a V* session's registered reads (`Flock.RecBudget`), for the stagers: a V* session's
``registered.json`` names at most `MAX_PORTS` ports, each of a tree of at most `MAX_DEPTH` levels (`regs`), and its table
holds at most `MAX_INSTANCES` instances (`table`); V*'s list at the inner shape holds at most `MAX_STATEMENTS` statements
(`count`). `verify` refuses a session past any of them, so a stager refuses to write one (`check`), with the verifier's
message."""
from __future__ import annotations

MAX_INSTANCES = 65536
MAX_PORTS = 4096
MAX_DEPTH = 64
MAX_STATEMENTS = 4096
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
                f"{TAIL}")
    return None


def check(registered: dict, unit: str, instances: int) -> None:
    """Raise ValueError with the verifier's refusal of a V* session it would refuse."""
    if why := regs(registered) or table(unit, instances):
        raise ValueError(why)
