"""A first check of the consistency relation between a circuit and its network certificates, on real certified frames.

The relation has two directions (`note:network-accounting/20261006T0633Z-draft-circuit-certificate-consistency`): every
crossing value appears in its link's certified traffic at the position the framing φ gives it, and every certified frame
is explained by crossing values or by declared protocol traffic D. This reads the raw network certificates of one
`active_replay.py` run (`requests.records`, `responses.records`, and the server's event log `server.events.gz`) and
checks what can be checked without a Build of the same requests:

- **φ, the framing.** Every real slot is one canonical frame: kind `DATA` or `LAST`, a length that fits, and zero fill.
  Padding is elided or all zero bytes.
- **Ingress, both directions.** Each stream's request is rebuilt from the workload it names
  (`active_replay.live_body`). The certified prompt and `max_tokens` must equal those run inputs, which are the
  circuit's external inputs. Every other byte is either D (the HTTP head, the JSON syntax, the constant fields) or not
  explained by anything (the session label `user`, and the padding that fills the last frame).
- **Egress, direction 2.** Each stream's bytes are cut into server-sent events and must equal the server's log. Every
  byte is then counted as one of:
  - `crossing`: a function of the circuit's outputs, which are the completion text, `finish_reason` and `stop_reason`;
  - `declared`: D, meaning the HTTP head, the event syntax, the JSON keys and syntax, the constant fields and
    `data: [DONE]`;
  - `advice`: the event's `id:`, its canonical position, which the schedule fixes;
  - `unexplained`: the completion id, which is random per request, and `created`, the server's wall-clock second.
  The direction 1 check on egress needs the Build's output tokens for these requests, which no run has, so it isn't
  done here.

    python consistency.py RUN_DIR [RUN_DIR ...] --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import active_replay as ar
from verity.core.protocols.certifier.active import DATA, HEADER, LAST, PAD, EVENT_END

CROSSING_CHOICE = ("text", "finish_reason", "stop_reason")
UNEXPLAINED_TOP = ("id", "created")
COMPLETION_ID = re.compile(r"^cmpl-([0-9a-f]+)$")


def _compact(v) -> bytes:
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode()


def deframe(records: list, frame_bytes: int) -> tuple[dict[int, bytearray], dict]:
    """Each stream's payload bytes, in certified order, and φ's tally: real frames by kind, padding, and every slot
    that isn't a canonical frame."""
    streams: dict[int, bytearray] = defaultdict(bytearray)
    tally = {"link_windows": len(records), "complete": sum(r.status.complete for r in records), "frames": Counter(),
             "padding": 0, "non_canonical": []}
    room = frame_bytes - HEADER.size
    for r in sorted(records, key=lambda r: r.window):
        for t, row in enumerate(r.grid):
            for j, f in enumerate(row):
                if f is None or f == bytes(frame_bytes):
                    tally["padding"] += 1
                    continue
                kind, stream, n = HEADER.unpack_from(f)
                if len(f) != frame_bytes or kind not in (DATA, LAST) or n > room or any(f[HEADER.size + n:]):
                    tally["non_canonical"].append([r.window, t, j])
                    continue
                tally["frames"][{DATA: "DATA", LAST: "LAST"}[kind]] += 1
                streams[stream] += f[HEADER.size : HEADER.size + n]
    tally["frames"] = dict(tally["frames"])
    return streams, tally


def ingress(streams: dict[int, bytearray], model: str | None) -> tuple[dict[int, str], dict]:
    """Each stream's session, and the ingress tally: bytes by class, and the requests whose certified inputs are, or
    aren't, the workload's run inputs."""
    sessions, by_class = {}, Counter()
    checked, mismatched, unparsed = 0, [], []
    for s, raw in sorted(streams.items()):
        head_end = raw.find(b"\r\n\r\n")
        try:
            body_raw = bytes(raw[head_end + 4 :])
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
        by_class["declared"] += len(raw) - crossing - label - filler
        if session is None or model is None:
            continue
        checked += 1
        expected = ar.live_body(session, model)
        if {k: v for k, v in body.items() if k != "user"} != expected:
            mismatched.append(session)
    return sessions, {"requests": len(streams), "bytes": dict(by_class), "run_inputs_checked": checked,
                      "run_inputs_mismatched": mismatched, "unparsed_streams": unparsed}


def egress(streams: dict[int, bytearray], sessions: dict[int, str], log: list) -> dict:
    """The egress tally: bytes by class, the events that differ from the server's log, and the unexplained fields'
    values per request."""
    logged = defaultdict(list)
    for position, session, k, _bucket, event in sorted(log):
        logged[session].append(event)
    by_class, fields = Counter(), {"completion_id_hex_digits": Counter(), "created_per_request": Counter(),
                                   "created_values": set(), "events": 0, "done": 0, "reserialized_exactly": 0}
    differs, unparsed, compared = [], [], 0
    for s, raw in sorted(streams.items()):
        head_end = raw.find(b"\r\n\r\n") + 4 if raw.startswith(b"HTTP/") else 0
        by_class["declared"] += head_end
        events = ar.Events().feed(bytes(raw[head_end:]))
        rest = len(raw) - head_end - sum(map(len, events))
        by_class["unexplained"] += rest
        session = sessions.get(s)
        certified = [bytes(raw[:head_end]) + events[0], *events[1:]] if events else []
        if session is not None and log:
            compared += 1
            if certified != logged.get(session, [])[: len(events)]:
                differs.append(session)
        ids, created = set(), set()
        for e in events:
            fields["events"] += 1
            lines = e[: -len(EVENT_END)].split(b"\n")
            data = next((ln[len(b"data: ") :] for ln in lines if ln.startswith(b"data: ")), None)
            position = next((ln for ln in lines if ln.startswith(b"id:")), b"")
            by_class["advice"] += len(position)
            if data == b"[DONE]":
                fields["done"] += 1
                by_class["declared"] += len(e) - len(position)
                continue
            try:
                chunk = json.loads(data)
            except (TypeError, ValueError):
                unparsed.append(session)
                by_class["unexplained"] += len(e) - len(position)
                continue
            fields["reserialized_exactly"] += _compact(chunk) == data
            crossing = sum(len(_compact(c.get(k))) for c in chunk.get("choices", []) for k in CROSSING_CHOICE if k in c)
            unexplained = sum(len(_compact(chunk[k])) for k in UNEXPLAINED_TOP if k in chunk)
            by_class["crossing"] += crossing
            by_class["unexplained"] += unexplained
            by_class["declared"] += len(e) - len(position) - crossing - unexplained
            if m := COMPLETION_ID.match(str(chunk.get("id", ""))):
                ids.add(m.group(1))
            if "created" in chunk:
                created.add(chunk["created"])
        for i in ids:
            fields["completion_id_hex_digits"][len(i)] += 1
        fields["created_per_request"][len(created)] += 1
        fields["created_values"] |= created
    total = sum(by_class.values())
    return {"requests": len(streams), "bytes": dict(by_class), "share": {k: v / total for k, v in by_class.items()} if total else {},
            "events": fields["events"], "done_events": fields["done"],
            "events_reserialized_exactly": fields["reserialized_exactly"],
            "completion_id_hex_digits": dict(fields["completion_id_hex_digits"]),
            "distinct_created_per_request": dict(fields["created_per_request"]),
            "created_span_s": (max(fields["created_values"]) - min(fields["created_values"])) if fields["created_values"] else 0,
            "streams_compared_with_server_log": compared, "streams_differing_from_server_log": differs,
            "unparsed_events": unparsed}


def check_run(run: Path, frame_bytes: int, model: str | None) -> dict:
    out = {"run": str(run)}
    in_streams, out["phi_ingress"] = deframe([ar.unpack_record(b, frame_bytes) for b in ar.read_blobs(run / "requests.records")], frame_bytes)
    eg_streams, out["phi_egress"] = deframe([ar.unpack_record(b, frame_bytes) for b in ar.read_blobs(run / "responses.records")], frame_bytes)
    sessions, out["ingress"] = ingress(in_streams, model)
    log = ar.read_events(run / "server.events.gz") if (run / "server.events.gz").exists() else []
    out["egress"] = egress(eg_streams, sessions, log)
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
