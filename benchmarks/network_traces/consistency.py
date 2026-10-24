"""A first check of the consistency relation between a circuit and its network certificates, on real certified frames.

The relation has two directions (`note:network-accounting/20261006T0633Z-draft-circuit-certificate-consistency`): every
crossing value appears in its link's certified traffic at the position the framing φ gives it, and every certified frame
is explained by crossing values or by declared protocol traffic D. This reads the raw network certificates of one
`active_replay.py` run (`requests.records`, `responses.records`, and the server's event log `server.events.gz`) and
checks what can be checked without a Build of the same requests:

- **φ, the framing.** Every real slot is one canonical frame (`active`'s framing): kind `DATA` on `CANONICAL_STREAM`, a
  length that fits, and zero fill. Padding is elided or all zero bytes. On ingress the frames are cut into requests by
  the length each starts with (`active.Requests`), the k-th request stream k, and the fill after a request's last byte
  is zero.
- **Ingress, both directions.** Each stream's request is rebuilt from the workload it names
  (`active_replay.live_body`). The certified prompt and `max_tokens` must equal those run inputs, which are the
  circuit's external inputs. Every other byte is either D (the length, the HTTP head, the JSON syntax, the constant
  fields) or not explained by anything (the session label `user`, and the padding that fills the last frame).
- **Egress, direction 2.** Each frame must be one canonical event (`active.event_data`), and the frames in certified
  order must be the server's log in position order, each what follows its `id:` line (`active.event_body`). Every byte
  is then counted as one of:
  - `crossing`: a function of the circuit's outputs, the event's text (the base64 of its token's bytes, padded to
    `EVENT_TEXT`);
  - `declared`: D, the event's constant head and tail;
  - `unexplained`: a frame that isn't a canonical event.
  The direction 1 check on egress needs the Build's output tokens for these requests, which no run has, so it isn't
  done here.

    python consistency.py RUN_DIR [RUN_DIR ...] --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import active_replay as ar
from verity.core.protocols.certifier.active import (
    CANONICAL_STREAM,
    DATA,
    EVENT_TEXT,
    HEADER,
    LENGTH,
    Requests,
    event_body,
    event_data,
)


def _compact(v) -> bytes:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode()


def deframe(records: list, frame_bytes: int) -> tuple[list[bytes], dict]:
    """The link's frame payloads, in certified order, and φ's tally: real frames, padding, and every slot that isn't a
    canonical frame."""
    payloads: list[bytes] = []
    tally = {"link_windows": len(records), "complete": sum(r.status.complete for r in records), "frames": 0,
             "padding": 0, "non_canonical": []}
    room = frame_bytes - HEADER.size
    for r in sorted(records, key=lambda r: r.window):
        for t, row in enumerate(r.grid):
            for j, f in enumerate(row):
                if f is None or f == bytes(frame_bytes):
                    tally["padding"] += 1
                    continue
                kind, stream, n = HEADER.unpack_from(f)
                if (len(f) != frame_bytes or kind != DATA or stream != CANONICAL_STREAM or n > room
                        or any(f[HEADER.size + n :])):
                    tally["non_canonical"].append([r.window, t, j])
                    continue
                tally["frames"] += 1
                payloads.append(f[HEADER.size : HEADER.size + n])
    return payloads, tally


def requests(payloads: list[bytes]) -> tuple[dict[int, bytes], list[int]]:
    """The requests the ingress payloads carry, by stream (`active.Requests`), and the streams whose last frame's fill
    isn't zero."""
    split, out, dirty = Requests(), defaultdict(bytearray), []
    for p in payloads:
        start = not split.left
        stream, data, end = split.feed(p)
        out[stream] += data
        if end and any(p[(LENGTH.size if start else 0) + len(data) :]):
            dirty.append(stream)
    return {s: bytes(r) for s, r in out.items()}, dirty


def ingress(payloads: list[bytes], model: str | None) -> tuple[dict[int, str], dict]:
    """Each stream's session, and the ingress tally: bytes by class, and the requests whose certified inputs are, or
    aren't, the workload's run inputs."""
    streams, dirty = requests(payloads)
    sessions, by_class = {}, Counter()
    checked, mismatched, unparsed = 0, [], []
    for s, raw in sorted(streams.items()):
        head_end = raw.find(b"\r\n\r\n")
        try:
            body_raw = raw[head_end + 4 :]
            body = json.loads(body_raw)
        except ValueError:
            unparsed.append(s)
            continue
        session = body.get("user")
        sessions[s] = session
        crossing = sum(len(_compact(body[k])) for k in ("prompt", "max_tokens") if k in body)
        label = len(_compact(session)) if session is not None else 0
        filler = len(body_raw) - len(body_raw.rstrip(b" "))
        by_class["crossing"] += crossing
        by_class["unexplained"] += label + filler
        by_class["declared"] += LENGTH.size + len(raw) - crossing - label - filler
        if session is None or model is None:
            continue
        checked += 1
        expected = ar.live_body(session, model)
        if {k: v for k, v in body.items() if k != "user"} != expected:
            mismatched.append(session)
    return sessions, {"requests": len(streams), "bytes": dict(by_class), "run_inputs_checked": checked,
                      "run_inputs_mismatched": mismatched, "unparsed_streams": unparsed, "fill_not_zero": dirty}


def egress(payloads: list[bytes], log: list) -> dict:
    """The egress tally: bytes by class, the frames that aren't canonical events, and the sessions whose events differ
    from the server's log."""
    logged = [(session, event_body(event)) for _p, session, _k, _b, event in sorted(log)]
    by_class, differs, unparsed = Counter(), [], []
    for i, body in enumerate(payloads):
        session = logged[i][0] if i < len(logged) else None
        if event_data(body) is None:
            unparsed.append(i)
            by_class["unexplained"] += len(body)
        else:
            by_class["crossing"] += EVENT_TEXT
            by_class["declared"] += len(body) - EVENT_TEXT
        if logged and (i >= len(logged) or logged[i][1] != body) and session not in differs:
            differs.append(session)
    total = sum(by_class.values())
    return {"events": len(payloads), "bytes": dict(by_class),
            "share": {k: v / total for k, v in by_class.items()} if total else {},
            "events_compared_with_server_log": min(len(payloads), len(logged)) if logged else 0,
            "sessions_differing_from_server_log": differs, "unparsed_events": unparsed}


def check_run(run: Path, frame_bytes: int, model: str | None) -> dict:
    out = {"run": str(run)}
    in_payloads, out["phi_ingress"] = deframe([ar.unpack_record(b, frame_bytes) for b in ar.read_blobs(run / "requests.records")], frame_bytes)
    eg_payloads, out["phi_egress"] = deframe([ar.unpack_record(b, frame_bytes) for b in ar.read_blobs(run / "responses.records")], frame_bytes)
    _sessions, out["ingress"] = ingress(in_payloads, model)
    log = ar.read_events(run / "server.events.gz") if (run / "server.events.gz").exists() else []
    out["egress"] = egress(eg_payloads, log)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="active_replay.py run directories")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frame-bytes", type=int, default=4096)
    ap.add_argument("--model", default="served", help="the served model name the requests carry (None: skip the run-input check)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = [check_run(r, args.frame_bytes, None if args.model == "None" else args.model) for r in args.runs]
    (args.out / "consistency.json").write_text(json.dumps(results, indent=1, default=sorted))
    for r in results:
        print(json.dumps({"run": r["run"], "non_canonical": len(r["phi_ingress"]["non_canonical"]) + len(r["phi_egress"]["non_canonical"]),
                          "inputs_mismatched": len(r["ingress"]["run_inputs_mismatched"]),
                          "egress_share": {k: round(v, 4) for k, v in r["egress"]["share"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
