"""`PousCheck.Obligation.verdict` against `obligation.verdict` through the core difftest harness (`pous-obligation-difftest`).

The named cases are `obligation_cases.CASES` (read by path from the PoUS suite): the honest run, a stale page (its old
version answers a read), content equal to its keystream (committed again after the coin, or kept under the first
commitment) and a page encoded before its coin (stored before the coin, or after it under a guessed one). Each must give
the same verdict in Lean and Python, and the reason the case names. The drawn logs are those cases at other seeds and
sizes, each under one event edit (an event dropped, duplicated, moved or swapped with the next, a hex field's byte
flipped, a number moved by one, a read made late) or one malformation (a key dropped, hex uppercased or cut odd, a salt
or coin of the wrong length, an unknown op, a count out of range), which both must refuse.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.pous import obligation as O

LEAN = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

_spec = importlib.util.spec_from_file_location("obligation_cases",
                                               ROOT / "verity/core/protocols/pous/tests/obligation_cases.py")
C = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def port(inputs) -> D.Port:
    return D.Port("pous-obligation-difftest", "verdict", O.verdict, inputs, package=LEAN, rejects=(ValueError,))


server = D.server_fixture(port(st.none()))


@pytest.mark.parametrize("name", sorted(C.CASES))
def test_named_cases_agree(server, name):
    """Each case at seeds 0 to 2: Lean's verdict is Python's, accepted for the honest run and rejected for the reason the
    case names otherwise."""
    build, reason = C.CASES[name]
    for seed in range(3):
        kind, text = D.replay(port(st.none()), build(seed), server)
        assert kind == "output", (kind, text)
        got = json.loads(text)
        assert got["verdict"] == ("accept" if reason is None else "reject"), got
        assert reason is None or got["reason"] == reason, got


def _flip(h: str, i: int) -> str:
    if not h:
        return "00"
    k = 2 * (i % (len(h) // 2))
    return h[:k] + f"{int(h[k:k + 2], 16) ^ (1 << (i % 8)):02x}" + h[k + 2:]


def _event_edit(log: dict, kind: int, i: int, x: int) -> dict:
    log = copy.deepcopy(log)
    ev = log["events"]
    if not ev:
        return log
    i %= len(ev)
    e = ev[i]
    match kind:
        case 0:
            del ev[i]
        case 1:
            ev.insert(i, copy.deepcopy(e))
        case 2:
            ev.insert(x % len(ev), ev.pop(i))
        case 3:
            if i + 1 < len(ev):
                ev[i], ev[i + 1] = ev[i + 1], ev[i]
        case 4:
            keys = sorted(k for k, v in e.items() if isinstance(v, str) and k != "op")
            lists = sorted(k for k, v in e.items() if isinstance(v, list) and v)
            if lists and x % 2:
                k = lists[x % len(lists)]
                e[k][x % len(e[k])] = _flip(e[k][x % len(e[k])], x)
            elif keys:
                k = keys[x % len(keys)]
                e[k] = _flip(e[k], x)
        case 5:
            nums = sorted(k for k, v in e.items() if type(v) is int)
            if nums:
                k = nums[x % len(nums)]
                e[k] = max(0, e[k] + (1 if x % 2 else -1))
        case _:
            reads = [r for r in ev if r["op"] == "read"]
            if reads:
                reads[x % len(reads)]["ns"] = log["limit_ns"] + 1 + x % 3
    return log


def _malformed(log: dict, kind: int, x: int) -> dict:
    log = copy.deepcopy(log)
    ev = log["events"]
    e = ev[x % len(ev)]
    match kind:
        case 0:
            e.pop(sorted(e)[x % len(e)])
        case 1:
            k = next((k for k, v in sorted(e.items()) if isinstance(v, str) and k != "op" and v), None)
            if k is None:
                log["salt"] = log["salt"].upper() + "a"
            else:
                e[k] = e[k][:-1] if x % 2 else e[k].upper() + "AB"
        case 2:
            log["salt"] = log["salt"][: -2 * (1 + x % 3)] if x % 2 else log["salt"] + "00"
        case 3:
            coins = [c for c in ev if c["op"] == "coin"]
            coins[x % len(coins)]["coin"] += "00"
        case 4:
            ev.insert(x % (len(ev) + 1), {"op": ["erase", "COMMIT", 7][x % 3]})
        case _:
            log[["pages", "blocks"][x % 2]] = [0, O.MAX + 1, -1, True][x % 4]
    return log


NAMES = sorted(C.CASES)
xs = st.integers(0, 2 ** 16)
logs = st.builds(lambda n, seed, pages, blocks: C.CASES[n][0](seed, pages=pages, blocks=blocks),
                 st.sampled_from(NAMES), st.integers(0, 2 ** 16), st.integers(1, 3), st.integers(1, 5))
edited = st.builds(_event_edit, logs, st.integers(0, 6), xs, xs)
malformed = st.builds(_malformed, logs, st.integers(0, 5), xs)


@pytest.mark.slow
def test_drawn_logs_agree(server):
    """The cases at drawn seeds and sizes, as they are, under one event edit, and malformed."""
    stats = D.run(port(st.one_of(logs, edited, edited, edited, malformed)), examples=300, server=server)
    assert stats.outputs > 150 and stats.rejects > 15, stats
