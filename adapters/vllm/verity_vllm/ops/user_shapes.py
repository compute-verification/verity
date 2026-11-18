"""Counts stored runs' request users by shape, printing counts only, never a user: a request_users.jsonl line's ``user``,
a seals.jsonl line's ``requests`` (each the body's ``user``, else the server's HTTP id).

    python3 user_shapes.py LIST    LIST: one path a line, each a copy of request_users.jsonl or seals.jsonl
"""
import json
import re
import sys
from collections import Counter

#: the server's HTTP id (serve_admit.new_http_id), then the labels our load tools make up, then the generic shapes
SHAPES = (
    ("server-made HTTP id (cmpl-/chatcmpl- + hex)", r"(chat)?cmpl-[0-9a-f]{32}(-\d+)?"),
    ("UUID", r"[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}"),
    ("synthetic <word>.<n> (serve_load varied./fixed.)", r"(varied|fixed)\.\d+(\.t\d+)?"),
    ("synthetic user-<n> (network front, advice_run)", r"user-\d+"),
    ("synthetic req-<n> (trace corpus, serve_tax)", r"req-\d+"),
    ("synthetic b<k>-<i> (serve_load bursts)", r"b\d+-\d+"),
    ("synthetic <tag>-<n> (attester_load reps, attester warm-ups)", r"(warmup-.+|.+-rep\d+)-\d+"),
    ("synthetic <letter><n> (u<n>, q<n>, r<n>)", r"[uqr]\d+"),
    ("synthetic <word>-smoke[-<n>] (llama1b_serve_seal.sh's smokes)", r"[a-z]+-smoke(-\d+)?"),
    ("synthetic none-<n> (active_replay's sessionless requests)", r"none-\d+"),
    ("synthetic <name>.<n> (the network certifier front's sessions, certifier_serve.py)", r"[A-Za-z0-9_-]+\.\d+"),
    ("email-like", r"[^@\s]+@[^@\s]+\.[^@\s]+"),
    ("contains whitespace", r".*\s.*"),
)
COMPILED = [(name, re.compile(rx, re.S)) for name, rx in SHAPES]


def length_class(n: int) -> str:
    return "1-8" if n <= 8 else "9-32" if n <= 32 else "33-128" if n <= 128 else ">128"


def shape(user) -> str:
    if user is None:
        return "none sent"
    if not isinstance(user, str):
        return "not a string"
    return next((name for name, rx in COMPILED if rx.fullmatch(user)),
                f"matches no shape, length {length_class(len(user))}")


def users_of(path: str, record: dict) -> list:
    if path.endswith("seals.jsonl"):
        return list(record.get("requests") or []) if isinstance(record, dict) else []
    return [record.get("user")] if isinstance(record, dict) else []


def main(listing: str) -> int:
    by_kind: dict[str, Counter] = {}
    files, unreadable, bad_lines = Counter(), Counter(), Counter()
    for path in filter(None, map(str.strip, open(listing))):
        kind = "seals.jsonl" if path.endswith("seals.jsonl") else "request_users.jsonl"
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            unreadable[kind] += 1
            continue
        files[kind] += 1
        counts = by_kind.setdefault(kind, Counter())
        for line in filter(str.strip, text.splitlines()):
            try:
                counts.update(shape(u) for u in users_of(path, json.loads(line)))
            except ValueError:
                bad_lines[kind] += 1
    print(json.dumps({"files_read": dict(files), "unreadable": dict(unreadable), "bad_lines": dict(bad_lines),
                      "by_shape": {k: dict(c.most_common()) for k, c in by_kind.items()},
                      "totals": {k: sum(c.values()) for k, c in by_kind.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
