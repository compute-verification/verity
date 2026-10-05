"""A worker for `flock-audit`'s tests (`test_lean_audit.py`): one process serving the developer, the serving side and the
audit, each request a frame on stdin (an 8-byte big-endian length and the bytes) and its answer the next frame on stdout. It
replays a recorded session into the auditor's content store; `--tamper` makes one of its answers a dishonest one. Every
request it reads is appended to `--log` as a JSON line."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TAMPER = ("redraw", "draw", "hash", "proof", "canon", "long", "exit", "population")


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def put(archive: Path, b: bytes) -> str:
    key = hashlib.sha512(b).hexdigest()
    (archive / "sha512").mkdir(parents=True, exist_ok=True)
    (archive / "sha512" / key).write_bytes(b)
    return key


def read_frame(f) -> bytes | None:
    head = f.read(8)
    if len(head) < 8:
        return None
    return f.read(int.from_bytes(head, "big"))


def write_frame(f, b: bytes) -> None:
    f.write(len(b).to_bytes(8, "big") + b)
    f.flush()


def session(archive: Path, recorded: Path, tamper: str | None) -> dict:
    files = {}
    for p in sorted(recorded.glob("*.rep*.bin")):
        b = p.read_bytes()
        if tamper == "proof" and p.name.endswith(".rep0.bin"):
            b = b[:len(b) // 2] + bytes([b[len(b) // 2] ^ 1]) + b[len(b) // 2 + 1:]
        files[p.name] = put(archive, b)
    record = (recorded / "session.json").read_bytes()
    if tamper == "draw":
        j = json.loads(record)
        d = j["unit_draw"]
        d["units"] = sorted(set(d["units"][1:]) | {min(set(range(d["population"])) - set(d["units"]))})
        record = json.dumps(j).encode()
    key = put(archive, record)
    if tamper == "hash":
        (archive / "sha512" / key).write_bytes(record + b" ")
    return {"files": files, "record": key}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--session", type=Path, required=True, help="the recorded session the serving side answers with")
    ap.add_argument("--public", type=Path, required=True, help="the developer's registered public file")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--tamper", choices=TAMPER)
    a = ap.parse_args()
    pub = a.public.read_bytes()
    if a.tamper == "population":
        line, rest = pub.split(b"\n", 1)
        header = json.loads(line)
        header["instances"] *= 2
        pub = json.dumps(header).encode() + b"\n" + rest
    public = put(a.archive, pub)
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while (req := read_frame(inp)) is not None:
        ask = json.loads(req)
        with a.log.open("a") as f:
            f.write(json.dumps(ask) + "\n")
        if ask["ask"] == "register":
            if a.tamper == "canon":
                write_frame(out, json.dumps({"public": public}).encode())
                continue
            ans = {"public": public}
        elif ask["ask"] == "session":
            if a.tamper == "exit":
                return 0
            if a.tamper == "long":
                out.write((1 << 40).to_bytes(8, "big"))
                out.flush()
                continue
            ans = {"redraw": True} if a.tamper == "redraw" else session(a.archive, a.session, a.tamper)
        elif ask["ask"] == "audit":
            ans = {"accepted": ask["verdict"]["accepted"], "report": "received"}
        else:
            ans = {"error": f"unknown ask {ask['ask']!r}"}
        write_frame(out, canon(ans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
