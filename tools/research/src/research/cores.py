"""Whole cores for a queued CPU run on a shared node (`research run --queue`), claimed ahead of the run's allocation."""
from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from pathlib import Path
from typing import Callable

CLAIMS = Path(os.environ.get("RESEARCH_CPU_CLAIMS_DIR", "/workspace/research/locks/cpu-claims"))
CPU_ROOT = Path(os.environ.get("RESEARCH_CPU_ROOT", "/sys/devices/system/cpu"))
POLL_S = 15


def parse(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        if lo.isdigit() and (not hi or hi.isdigit()):
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def fmt(cpus: set[int]) -> str:
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and runs[-1][1] == c - 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs)


def core(c: int, root: Path) -> frozenset[int]:
    try:
        return frozenset(parse((root / f"cpu{c}" / "topology" / "thread_siblings_list").read_text())) or frozenset({c})
    except OSError:
        return frozenset({c})


def carve(free: set[int], n: int, root: Path) -> set[int] | None:
    """Whole cores, lowest first; None when they don't fit."""
    got: set[int] = set()
    for c in sorted(free):
        sib = core(c, root)
        if len(got) < n and c == min(sib) and sib <= free:
            got |= sib
    return got if len(got) >= n else None


def held(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def live(d: Path, pattern: str) -> list[Path]:
    out = []
    for f in sorted(d.glob(pattern)):
        if held(f):
            out.append(f)
        else:
            f.unlink(missing_ok=True)
    return out


def claimed(d: Path) -> dict[str, set[int]]:
    out = {}
    for f in live(d, "*.cpus"):
        try:
            out[f.name.removesuffix(".cpus")] = parse(f.read_text())
        except OSError:
            continue
    return out


def locked(path: Path, text: str) -> int:
    tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, text.encode())
        os.replace(tmp, path)
    except OSError:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    return fd


def take(n: int, share: set[int], who: str, d: Path, root: Path) -> tuple[set[int], int] | None:
    with open(d / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        live_claims = claimed(d)
        got = carve(share - set().union(*live_claims.values()), n, root)
        if got is None or who in live_claims:
            return None
        return got, locked(d / f"{who}.cpus", f"{fmt(got)}\n")


def ticket_name(rank: int, ns: int, who: str) -> str:
    if rank > 0:
        return f"wait.!{9999 - min(rank, 9999):04d}.{ns:020d}.{who}"
    if rank < 0:
        return f"wait.~{min(-rank, 9999):04d}.{ns:020d}.{who}"
    return f"wait.{ns:020d}.{who}"


def hold(n: int, share: set[int], who: str, *, rank: int = 0, d: Path = CLAIMS, root: Path = CPU_ROOT, poll_s: float = POLL_S,
         sleep: Callable[[float], None] = time.sleep, clock_ns: Callable[[], int] = time.time_ns,
         log: Callable[[str], None] = lambda s: print(s, file=sys.stderr, flush=True),
         on_hold: Callable[[], None] = lambda: None) -> tuple[set[int], int]:
    if carve(share, n, root) is None:
        raise ValueError(f"{n} CPUs in whole cores don't fit in {fmt(share)}")
    d.mkdir(parents=True, exist_ok=True)
    ticket, tfd, polls = d / ticket_name(rank, clock_ns(), who), None, 0
    try:
        while True:
            ahead = [t for t in live(d, "wait.*") if t.name < ticket.name]
            if not ahead and (got := take(n, share, who, d, root)):
                if polls:
                    log(f"research cpu-hold: took CPUs {fmt(got[0])} after {polls} poll(s)")
                return got
            if tfd is None:
                tfd = locked(ticket, f"{n}\n")
                busy = claimed(d)
                log(f"research cpu-hold: waiting for {n} CPUs in whole cores of {fmt(share)}: {len(busy)} claim(s) hold "
                    f"{fmt(set().union(*busy.values()) & share) or 'none of them'}, {len(ahead)} waiter(s) ahead; polled every "
                    f"{poll_s:g} s")
                on_hold()
            polls += 1
            sleep(poll_s)
    finally:
        if tfd is not None:
            ticket.unlink(missing_ok=True)
            os.close(tfd)


def pinned(cmd: list[str], k: int, share: str, got: str) -> list[str]:
    head = [got if a == share else a[:-len(share)] + got if a.endswith(f"={share}") else a for a in cmd[:k]]
    return head + cmd[k:]


def main(argv: list[str], *, sleep: Callable[[float], None] = time.sleep,
         execvp: Callable[[str, list[str]], None] = os.execvp) -> int:
    ap = argparse.ArgumentParser(prog="research cpu-hold")
    ap.add_argument("--cpus", type=int, required=True, help="how many CPUs (rounded up to whole cores)")
    ap.add_argument("--from", dest="share", required=True, metavar="LIST", help="the CPUs to take them from, as the wrapper names them")
    ap.add_argument("--wrap", type=int, required=True, metavar="K", help="CMD's first K arguments are the wrapper that names LIST")
    ap.add_argument("--who", required=True, help="the claim's name (the run id)")
    ap.add_argument("--rank", type=int, default=0, metavar="R", help="its class's priority less the default class's: the line "
                    "takes higher ranks first, then the oldest (default 0)")
    ap.add_argument("--ack", default=None, metavar="FIFO", help="the launch's acknowledgement FIFO, told when a wait starts")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    cmd = ns.cmd[1:] if ns.cmd[:1] == ["--"] else ns.cmd
    if not cmd:
        print("research cpu-hold: give the command after --", file=sys.stderr)
        return 2

    def on_hold() -> None:
        if ns.ack:
            from .remote import ack_launch

            ack_launch(ns.ack, "cpu-hold")

    d = Path(os.environ.get("RESEARCH_CPU_CLAIMS_DIR", CLAIMS))
    try:
        got, fd = hold(max(ns.cpus, 1), parse(ns.share), ns.who, rank=ns.rank,
                       d=d, root=Path(os.environ.get("RESEARCH_CPU_ROOT", CPU_ROOT)), sleep=sleep, on_hold=on_hold)
    except ValueError as e:
        print(f"research cpu-hold: {e}", file=sys.stderr)
        return 2
    os.set_inheritable(fd, True)
    cmd = pinned(cmd, ns.wrap, ns.share, fmt(got))
    print(f"research cpu-hold: run {ns.who} on CPUs {fmt(got)} (claim {d / (ns.who + '.cpus')})", file=sys.stderr, flush=True)
    execvp(cmd[0], cmd)
    return 127


__all__ = ["CLAIMS", "carve", "claimed", "hold", "main", "pinned", "take", "ticket_name"]
