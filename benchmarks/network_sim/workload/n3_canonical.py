"""N3Window_v5 with its templates on the canonical framing (`verity.core.protocols.certifier.active`): every frame a
port records is its direction's template on bytes of the committed outboxes, so v5 with the template accepts honest
traffic on every window, and refuses a frame off the template or off the outboxes.

- **Templates** (`template`): an egress frame is a canonical event (`active.event`), BASE64 at `active.event_placement`
  on outbox rows of `active.TOKEN_BYTES`; an ingress frame is `active.data_frames`' (a request's length and bytes on the
  proxy's links), RAW at `active.request_placement` on rows of B. A port's direction is its link's end (`Setup.links`:
  0 egress, 1 ingress), and each direction has its own layout on its own rows (`n3_v5.prepare` with its ports).
- **Pieces** (`event_pieces`, `data_pieces`): a frame as `n3_unit` declares it, from runs of the units' outboxes.
- **The replay** (`replay_windows`): an `active_replay.py` run as two units and two ports. The server's outbox is the
  bytes every egress frame of the Program carries, in canonical order (`active_replay.frame_data`, or what the live
  shim's logged events carry); the peer's is every request in link order, each its length (u32be) and bytes, as
  `active.request_frames` sends them. Each record's released frames are a window, a salt frame's payload the
  certifier's `active.PLACEHOLDER`.
- **The verdict** (`verdicts`): per window, v5's reference with its direction's template; a window no unit runs is ok
  only if it recorded nothing (`n3_unit.Window.check`).

    python -m workload.n3_canonical replay RUN_DIR --out FILE   v5's verdicts on an active_replay run's windows
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from verity.core.protocols.certifier import active as A
from verity.core.protocols.certifier.grid import Record, occupancy, row_count

from . import n3_unit as U
from . import n3_v5 as V5

EGRESS, INGRESS = 0, 1
RESPONSES, REQUESTS = "responses", "requests"
PROGRAM = b"verity/netsim/n3-replay"


def template(direction: int, B: int) -> tuple[tuple, int]:
    """(TPL, RB) of a direction's frames of B bytes."""
    if direction == EGRESS:
        return V5.tpl_of(U.B64, *A.event_placement(B)), A.TOKEN_BYTES
    return V5.tpl_of(U.RAW, *A.request_placement(B)), B


def event_pieces(unit: str, offset: int, n: int, B: int) -> list[tuple]:
    """The canonical event of bytes offset ... offset + n - 1 of the unit's outbox (n <= `TOKEN_BYTES`), as pieces."""
    if n > A.TOKEN_BYTES:
        raise ValueError(f"an event carries at most {A.TOKEN_BYTES} bytes, not {n}")
    _, s, head, tail = A.event_placement(B)
    text = [("b", (("r", (unit,), 8 * offset, 8 * n),), n)] if n else []
    pieces = [U.const(head), *text, U.zeros(A.EVENT_TEXT - 4 * -(-n // 3)), U.const(tail), U.zeros(B - s - len(tail))]
    return [p for p in pieces if U.width(p)]


def data_pieces(runs: Sequence[tuple[str, int, int]], B: int) -> list[list[tuple]]:
    """`active.data_frames` of the bytes the runs (unit, byte offset, bytes) of outboxes give in order, as each frame's
    pieces."""
    room = B - A.HEADER.size
    head = U.const(A.request_placement(B)[2])
    stream = [("r", (u,), 8 * o, 8 * n) for u, o, n in runs if n]
    total = sum(n for _, _, n in runs)
    out = []
    for lo in range(0, max(1, total), room):
        hi = min(lo + room, total)
        out.append([head, *U._slice(stream, 8 * lo, 8 * hi), U.zeros(room - (hi - lo))])
    return out


def released(rec: Record) -> list[bytes]:
    return [f for row in rec.grid for f in row[:row_count(row)]]


def windows_of(records: dict[str, Sequence[Record]]) -> list[U.Window]:
    """Each port's records as windows: its released frames, counts, status and grid, and on a window holding a salt
    frame the certifier's `PLACEHOLDER` as its payload."""
    out = []
    for port, recs in records.items():
        for rec in sorted(recs, key=lambda r: r.window):
            frames = released(rec)
            salt = A.PLACEHOLDER if any(U._is_salt(f) for f in frames) else None
            out.append(U.Window(port, rec.window, frames, occupancy(rec.grid),
                                rec.status.reason.value if rec.status.reason else "", salt, rec.grid))
    return out


def _ar():
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import records  # noqa: F401, PLC0415  (puts network_traces on the path, for active_replay)

    import active_replay as ar  # noqa: PLC0415

    return ar


def replay_units(plan: dict, setup, release: dict[int, int], events: Sequence | None = None) -> tuple[dict, dict]:
    """The replay's outboxes (`server`, `peer`) and each port's declared frames: the Program's egress frames in
    canonical order at `release` (or the live shim's `events`), and the plan's requests in link order."""
    ar = _ar()
    e = setup.egress_params
    B = e.frame_bytes
    if events is not None:
        tokens = [A.event_data(A.event_body(ev)) or b"" for *_, ev in sorted(events)]
    else:
        keys = ar.program_keys(plan["sessions"], release, plan["windows"] * e.buckets)
        tokens = [ar.frame_data(plan, plan["sessions"][rank][0], k) for _, _, rank, k in keys]
    server, egress = b"", []
    for t in tokens:
        egress.append(event_pieces("server", len(server), len(t), B))
        server += t
    sender = ar.Peer(plan, ar.wire_params(setup))
    peer, ingress = b"", []
    for q in sorted(plan["requests"], key=lambda q: (q["enqueue"], q["send_ms"], q["stream"])):
        request = sender.request(q)
        data = A.LENGTH.pack(len(request)) + request
        ingress += data_pieces([("peer", len(peer), len(data))], B)
        peer += data
    return {"server": server, "peer": peer}, {RESPONSES: egress, REQUESTS: ingress}


def replay_windows(plan: dict, setup, run: Path, records: dict[str, Sequence[Record]] | None = None,
                   seed: bytes = b"n3-replay") -> tuple[U.Setup, list[U.Window]]:
    """An active_replay run's N3 setup and windows: its raw records (`records`, else the run's), the units and frames
    `replay_units` declares, released as the ingress records say."""
    ar = _ar()
    e, i = setup.egress_params, setup.ingress_params
    B = e.frame_bytes
    if records is None:
        records = {link: [ar.unpack_record(b, B) for b in ar.read_blobs(run / f"{link}.records")]
                   for link in (RESPONSES, REQUESTS)}
    release, _ = ar.ingress_releases(records[REQUESTS], i, e)
    events = ar.read_events(run / "server.events.gz") if "live_model" in plan else None
    outboxes, declared = replay_units(plan, setup, release, events)
    units = U.UnitTable.register(outboxes, B, PROGRAM, seed)
    layout = U.register_layout(declared, [RESPONSES, REQUESTS], B, units, PROGRAM, b"", seed)
    links = {RESPONSES: ("replay", EGRESS), REQUESTS: ("replay", INGRESS)}
    n3 = U.Setup(None, units, layout, links, {RESPONSES: e, REQUESTS: i}, seed,  # type: ignore[arg-type]
                 sync_set=setup.sync_set, named={p: p for p in links})
    return n3, n3.attach(windows_of(records))


def verdicts(setup: U.Setup, windows: Sequence[U.Window], kinds: dict[str, int] | None = None) -> list[dict]:
    """v5's verdict on each window with its port's template, by its unit's reference or by public data: `kinds` gives
    each port's, `EGRESS` (events) or `INGRESS` (`data_frames`), by default its link's end."""
    B = setup.fmt.B
    kinds = kinds or {p: d for p, (_, d) in setup.links.items()}
    ports: dict[int, list[str]] = {}
    for p, d in kinds.items():
        ports.setdefault(d, []).append(p)
    out = []
    for w in windows:
        d = kinds[w.port]
        r = {"port": w.port, "window": w.window, "recorded": len(w.frames)}
        if w.instance is None:
            out.append(r | {"by": "public data", "ok": w.why == U.EMPTY and not w.frames, "why": w.why})
            continue
        tpl, RB = template(d, B)
        inst = V5.instance(setup, w, tpl=tpl, RB=RB, ports=ports[d])
        out.append(r | {"by": "unit", "ok": V5.reference(inst), "rows": inst.fmt.B, "N": inst.N})
    return out


def summary(checked: Sequence[dict]) -> dict:
    return {"windows": len(checked), "accepted": sum(r["ok"] for r in checked),
            "refused": [f"{r['port']}[{r['window']}]" + (f" ({r['why']})" if r.get("why") else "")
                        for r in checked if not r["ok"]]}


def _replay(args) -> dict:
    ar = _ar()
    run = Path(args.run)
    plan = json.loads((run / "plan.json").read_text())
    setup = ar.Setup(json.loads((Path(args.report) if args.report else run / "report.json").read_text()))
    n3, windows = replay_windows(plan, setup, run)
    checked = verdicts(n3, windows)
    return {"run": str(run), "summary": summary(checked), "windows": checked}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = ap.add_subparsers(dest="cmd", required=True)
    r = sub_.add_parser("replay", help="v5's verdicts with the templates on an active_replay run's windows")
    r.add_argument("run", help="the run's directory, with its plan.json and records")
    r.add_argument("--report", help="the calibration report (default: the run's report.json)")
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(fn=_replay)
    args = ap.parse_args(argv)
    result = args.fn(args)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
