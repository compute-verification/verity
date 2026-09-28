#!/usr/bin/env python3
"""`python3 tools/verity/check/lean_slot.py [--pool NAME] [--by NAME] -- CMD...`: CMD in one of this node's Lean audit slots.

A Lean audit builds its packages against Mathlib on every core it gets, so a node runs only so many at once (vy-nebius-1, 3 Oct:
a cap of 4, kept by hand and overrun). The steward sets the cap in SLOTS (`/workspace/research/locks/lean-slots`; LEAN_SLOTS
overrides), one pool per line, which it may rewrite at any time:

    NAME N      # e.g. `check 2` (check's lean-audit step), `audit 2` (`tools/verity/lean/check.py --build` run by
                # hand) and `build 2` (`lean_changed.py --records`, the fast path)

A request takes a slot of its pool (--pool, else LEAN_SLOT_POOL, else `audit`), or of pool `any` when SLOTS doesn't list its
own; SLOTS listing neither refuses it, naming the fix. Slot I of pool P is a `flock` on `lean-P-I.lock` beside SLOTS, I < N.
CMD inherits the descriptor, so the slot is held while CMD runs and freed when it exits, however it ends: the kernel drops a
dead process's flocks, after SIGTERM (`research cancel`) or SIGKILL too. CMD runs with LEAN_SLOT=P/I, so a request nested in
it (audit.py under check's step) runs at once instead of taking a second slot. Without SLOTS (a laptop) CMD runs as it is.

With every slot of its pool taken, a request waits, and never gives up: it queues in arrival order in `lean-line-P` beside
SLOTS (`slot.py`'s line), re-reads SLOTS every POLL_S, and prints each slot's holder (pid, since when, run, who, the time it
took the slot) and the waiters ahead of it whenever that changes, and every REPORT_EVERY polls otherwise. Who asked is --by,
else RESEARCH_LANE, else whoever holds the check slot this runs in (slot.py's `check-NAME.who`), else RESEARCH_AGENT_URL.

`python3 tools/verity/check/lean_slot.py --status [--json]`: each pool's slots and holders, then its waiters; --json, the same as one
JSON object (`status_doc`), which `check.py --slots` reads from every machine. Stdlib only: it runs before uv.

From Python (`tools/verity/lean/check.py --build`, `lean_changed.py --records`): `hold(pool)` takes a slot for the calling
process and returns its descriptor, which the process keeps open until it exits (None: no cap here, or a slot is held
already); `hold(pool, or_any=False)` never takes pool `any`'s.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SLOTS = Path(os.environ.get("LEAN_SLOTS", "/workspace/research/locks/lean-slots"))
POLL_S = 20
#: polls between two listings of the same holders: 3 min
REPORT_EVERY = 9
DEFAULT_POOL = "audit"
ANY = "any"
#: set in CMD's environment to the slot it holds, `POOL/I`
HELD = "LEAN_SLOT"

#: slot.py, from beside this file, unless a caller loaded it under this name already: `check.py --slots` sends both files to a
#: machine on stdin, where neither is on disk
SLOT_MODULE = "check_slot_for_lean_slot"
if SLOT_MODULE in sys.modules:
    SLOT = sys.modules[SLOT_MODULE]
else:
    _spec = importlib.util.spec_from_file_location(SLOT_MODULE, Path(__file__).resolve().parent / "slot.py")
    SLOT = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(SLOT)


class Refused(Exception):
    """SLOTS exists but grants this request nothing it can wait for: a guard never fails open, so the audit doesn't run."""


def parse(text: str) -> dict[str, int]:
    pools: dict[str, int] = {}
    for line in text.splitlines():
        f = line.split("#")[0].split()
        if not f:
            continue
        if len(f) != 2 or not f[1].isdigit():
            raise Refused(f"{SLOTS}: {line.strip()!r} is not `POOL N`")
        pools[f[0]] = int(f[1])
    return pools


def resolve(pools: dict[str, int], pool: str, or_any: bool = True) -> tuple[str, int]:
    """The pool a request for ``pool`` takes from (``or_any``: pool `any` when SLOTS doesn't list it), and its slot count."""
    for p in (pool, ANY) if or_any else (pool,):
        if p in pools:
            return p, pools[p]
    if not or_any:
        raise Refused(f"{SLOTS} lists no pool {pool!r}, the only one this request takes; the steward adds `{pool} N` there")
    raise Refused(f"{SLOTS} lists neither pool {pool!r} nor {ANY!r}; the steward adds `{pool} N` or `{ANY} N` there "
                  f"(or pass --pool / LEAN_SLOT_POOL naming one of {', '.join(sorted(pools)) or 'none'})")


def lock_path(d: Path, pool: str, i: int) -> Path:
    return d / f"lean-{pool}-{i}.lock"


def line_name(pool: str) -> str:
    return f"lean-line-{pool}"


def take(d: Path, pool: str, n: int) -> tuple[int, int] | None:
    """The first free slot of ``pool`` (its index and the descriptor holding its lock), or None."""
    for i in range(n):
        fd = os.open(lock_path(d, pool, i), os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        return i, fd
    return None


def pool_docs(pools: dict[str, int], d: Path, now: datetime, proc: Path = Path("/proc"), only: str | None = None,
              before: str | None = None) -> dict[str, dict]:
    """Each pool (``only``: that one) by name: its N, its slots with their holders (`slot.holding`), and its live waiters
    (``before``: those ahead of that ticket). A slot past N that is still held (the steward lowered N) is listed too."""
    out = {}
    for pool, n in sorted(pools.items()):
        if only is not None and pool != only:
            continue
        held = {int(p.stem.rsplit("-", 1)[1]) for p in d.glob(f"lean-{pool}-*.lock") if p.stem.rsplit("-", 1)[1].isdigit()}
        slots = []
        for i in sorted(set(range(n)) | {i for i in held if i >= n}):
            h = SLOT.holding(lock_path(d, pool, i), lock_path(d, pool, i).with_suffix(".who"), now, proc)
            if i < n or h is not None:
                slots.append({"index": i, "past_n": i >= n, "holder": h})
        out[pool] = {"n": n, "slots": slots, "waiters": SLOT.waiting(d / line_name(pool), now, before)}
    return out


def status(pools: dict[str, int], d: Path, now: datetime, proc: Path = Path("/proc"), only: str | None = None,
           before: str | None = None) -> list[str]:
    """`pool_docs` as lines: each slot with its holder, then the pool's waiters."""
    out = []
    for pool, p in pool_docs(pools, d, now, proc, only, before).items():
        out += [f"{pool}/{s['index']}{' (past N)' if s['past_n'] else ''}  {SLOT.held_text(s['holder'])}" for s in p["slots"]]
        out += [f"{pool}: {SLOT.waiter_text(w)}" for w in p["waiters"]]
    return out


def status_doc(now: datetime, proc: Path = Path("/proc")) -> dict:
    """`--status --json`: SLOTS and its pools (`pool_docs`; null without SLOTS, so audits here run uncapped)."""
    return {"file": str(SLOTS), "now": f"{now:%Y-%m-%dT%H:%MZ}",
            "pools": pool_docs(parse(SLOTS.read_text()), SLOTS.parent, now, proc) if SLOTS.exists() else None}


def asker(by: str | None = None) -> dict:
    """slot.py's asker, naming who asked as the module docstring says."""
    if not by and not (by := os.environ.get("RESEARCH_LANE")) and (name := os.environ.get("CHECK_SLOT")):
        by = SLOT._read(SLOT.SLOTS.parent / f"check-{name}.who").get("by")
    return SLOT.asker(by)


def wait(pool: str, by: str | None = None, now=lambda: datetime.now(timezone.utc), sleep=time.sleep,
         log=lambda s: print(s, file=sys.stderr, flush=True), or_any: bool = True) -> tuple[str, int, int]:
    """Wait for a slot of ``pool`` (SLOTS exists; `resolve`); return the pool it came from, its index and the descriptor
    holding it."""
    d = SLOTS.parent
    pools = parse(SLOTS.read_text())
    name, n = resolve(pools, pool, or_any)
    ticket, tfd = SLOT.join(d, who=asker(by), line=line_name(name))
    told, since = None, 0
    try:
        while True:
            try:
                pools = parse(SLOTS.read_text()) or pools
            except (OSError, Refused):
                pass                                    # mid-rewrite: the last pools it read
            n = pools.get(name, n)
            ahead = SLOT.ahead(ticket)
            got = None if ahead else take(d, name, n)
            if got:
                break
            why = f"{ahead} Lean audit(s) ahead in line" if ahead else f"all {n} Lean audit slot(s) of pool {name} are taken"
            lines = status(pools, d, now(), only=name, before=ticket.name)
            seen = (why, [re.sub(r" \(\d+ min\)", "", ln) for ln in lines])       # the same holders, a minute older
            if seen != told or since >= REPORT_EVERY:
                log(f"lean-slot: {why}; waiting, with no time limit, and retrying every {POLL_S} s ({SLOTS})")
                for ln in lines:
                    log(f"lean-slot:   {ln}")
                told, since = seen, 0
            since += 1
            sleep(POLL_S)
    finally:
        ticket.unlink(missing_ok=True)
        os.close(tfd)
    i, fd = got
    who = {**asker(by), "took": f"{now():%Y-%m-%dT%H:%MZ}"}
    path = lock_path(d, name, i).with_suffix(".who")
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(who) + "\n")
    tmp.chmod(0o664)
    os.replace(tmp, path)
    log(f"lean-slot: Lean audit slot {name}/{i} of {n} ({SLOTS})")
    return name, i, fd


def hold(pool: str | None = None, by: str | None = None, **kw) -> int | None:
    """Take a slot for this process (see the module docstring; ``kw`` goes to `wait`); its descriptor, which the caller keeps
    open, or None."""
    if os.environ.get(HELD) or not SLOTS.exists():
        return None
    name, i, fd = wait(pool or os.environ.get("LEAN_SLOT_POOL") or DEFAULT_POOL, by, **kw)
    os.environ[HELD] = f"{name}/{i}"
    return fd


def main(argv: list[str], now=lambda: datetime.now(timezone.utc), sleep=time.sleep) -> int:
    if argv[:1] == ["--status"]:
        if argv[1:] == ["--json"]:
            print(json.dumps(status_doc(now())))
        elif argv[1:]:
            sys.exit("usage: lean_slot.py --status [--json]")
        elif not SLOTS.exists():
            print(f"no Lean audit cap on this machine ({SLOTS})")
        else:
            print("\n".join(status(parse(SLOTS.read_text()), SLOTS.parent, now())))
        return 0
    opts, cmd = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else ([], argv)
    val = {k: opts[opts.index(k) + 1] for k in ("--pool", "--by") if k in opts[:-1]}
    if not cmd or len(opts) != 2 * len(val):
        sys.exit("usage: lean_slot.py [--pool NAME] [--by NAME] -- CMD...")
    if os.environ.get(HELD):
        print(f"lean-slot: this already runs in Lean audit slot {os.environ[HELD]}", file=sys.stderr, flush=True)
        os.execvp(cmd[0], cmd)
    if not SLOTS.exists():
        os.execvp(cmd[0], cmd)
    try:
        name, i, fd = wait(val.get("--pool") or os.environ.get("LEAN_SLOT_POOL") or DEFAULT_POOL, val.get("--by"), now, sleep)
    except Refused as e:
        sys.exit(f"lean-slot: {e}")
    os.set_inheritable(fd, True)
    os.execvpe(cmd[0], cmd, {**os.environ, HELD: f"{name}/{i}"})


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
