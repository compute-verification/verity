"""N3Window_v5 with the canonical framing's templates (`workload.n3_canonical`) on the active replay's raw records, on an
injected clock: an honest run is accepted on every window, egress and ingress, and so is one whose tokens are longer
than an event (cut over events of `TOKEN_BYTES`), empty or of every byte value, and whose prompt takes five frames. A
byte flipped in a frame's text, header, suffix or fill, a request byte that isn't the peer's, or one more event than
the Program has, is refused in its own window and nowhere else."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import active, calibration
from verity.core.protocols.certifier.grid import Record

HERE = Path(__file__).resolve().parents[1]
B = 4096


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", [str(HERE), *sys.path])
        spec = importlib.util.spec_from_file_location("network_sim_mockdc", HERE / "mockdc.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        mod.C = importlib.import_module("workload.n3_canonical")
        yield mod


def replay(m, doc: dict, out: Path, edit=None, tokens: dict | None = None) -> tuple:
    report = calibration.calibrate(doc)
    s = m.ar.Setup(report)
    plan = m.ar.plan_configuration(doc["configurations"][0], s)
    if edit is not None:
        plan = edit(plan, s)
    if tokens is not None:
        plan = plan | {"tokens": {k: [t.hex() for t in v] for k, v in tokens.items()}}
    assert m.ar.run_virtual(plan, report, out) == {"server": 0, "proxy": 0, "peer": 0}
    return plan, s


def records(m, run: Path) -> dict:
    return {link: [m.ar.unpack_record(b, B) for b in m.ar.read_blobs(run / f"{link}.records")]
            for link in ("responses", "requests")}


def verdicts(m, plan, s, run, recs=None) -> dict[tuple[str, int], bool]:
    n3, windows = m.C.replay_windows(plan, s, run, recs)
    return {(r["port"], r["window"]): r["ok"] for r in m.C.verdicts(n3, windows)}


EVERY = {(p, w): True for p in ("responses", "requests") for w in range(3)}


@pytest.fixture(scope="module")
def honest(m, tmp_path_factory):
    run = tmp_path_factory.mktemp("honest")
    plan, s = replay(m, m.trace(), run)
    return plan, s, run


def test_an_honest_run_is_accepted_on_every_window(m, honest):
    assert verdicts(m, *honest) == EVERY


def long_trace(m) -> dict:
    """The synthetic trace with window 0's first request five frames long."""
    doc = m.trace()
    ingress = doc["configurations"][0]["ingress"][0]
    first = ingress["arrival_ms"][0]
    ingress["arrival_ms"] = [first] * 4 + ingress["arrival_ms"]
    return doc


def test_long_tokens_and_a_long_prompt_are_accepted_on_every_window(m, tmp_path):
    """A token of 200 bytes takes four events (64, 64, 64 and 8 bytes); one of 65 takes two; an empty token is an event
    of no bytes; the tokens hold NUL, `=` and 0xff. The first request is five frames, its length in the first."""
    long = bytes(range(200))
    tokens = {"s0.0": [long[i:i + 64] for i in range(0, 200, 64)],
              "s0.1": [b"\xff" * 64, b"\xff", b"", b"="], "s1.2": [b"\0" * 63, b" t1", b"=" * 64, b"\0"]}
    assert tokens["s0.0"] == [bytes.fromhex(h) for h in m.ar.token_frames(long)]
    plan, s = replay(m, long_trace(m), tmp_path, tokens=tokens)
    assert max(q["frames"] for q in plan["requests"]) == 5
    assert verdicts(m, plan, s, tmp_path) == EVERY


def flip(recs: dict, link: str, window: int, k: int, at: int) -> dict:
    """The records with byte `at` of the k-th frame link's window `window` released flipped in its low bit."""
    out = dict(recs)
    edited = []
    for r in recs[link]:
        if r.window == window:
            seen, grid = 0, []
            for row in r.grid:
                new = []
                for f in row:
                    real = f is not None and any(f)
                    if real and seen == k:
                        f = f[:at] + bytes([f[at] ^ 1]) + f[at + 1:]
                    seen += real
                    new.append(f)
                grid.append(tuple(new))
            r = Record(r.link, r.window, tuple(grid), r.status)
        edited.append(r)
    out[link] = edited
    return out


P, S, _, _ = active.event_placement(B)
TAMPERS = {"event-text": ("responses", 0, 0, P), "event-header": ("responses", 1, 2, 3),
           "event-suffix": ("responses", 0, 1, S + 2), "event-fill": ("responses", 2, 0, B - 1),
           "request-byte": ("requests", 0, 0, 20), "request-header": ("requests", 1, 0, 8),
           "request-fill": ("requests", 2, 0, B - 1)}


@pytest.mark.parametrize("name", TAMPERS)
def test_a_tampered_frame_is_refused_in_its_window(m, honest, name):
    link, window, *_ = TAMPERS[name]
    plan, s, run = honest
    got = verdicts(m, plan, s, run, flip(records(m, run), *TAMPERS[name]))
    assert got == EVERY | {(link, window): False}


def test_an_extra_event_is_refused_in_its_window(m, tmp_path):
    """mockdc's `extra` server: honest, but for one more event on the first session at the position after the
    Program's last, so the last window holds it; declared against the honest Program."""
    plan, s = replay(m, m.trace(), tmp_path, edit=lambda p, s: m.extra_plan(p))
    honest_plan = {k: v for k, v in plan.items() if k not in ("strategy", "writes")}
    assert verdicts(m, honest_plan, s, tmp_path) == EVERY | {("responses", 2): False}
