"""Public IPv4 addresses, cloud resource ids and endpoints, and Slack, Notion or Linear ids and links, found in text."""
from __future__ import annotations

import ipaddress
import os
import re
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

RESOLVERS = {"1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9"}
# : tests/test_repository.py's credential scan reads this list too.
BINARY = frozenset({".bin", ".npy", ".npz", ".lgto", ".neg", ".proof", ".gz", ".png", ".pdf", ".olean", ".ilean", ".tar", ".zip"})
# : `uv.lock`'s dotted numbers are package versions; the test's toy files name each form on purpose.
SKIP = frozenset({"uv.lock", "tests/test_literal_addresses.py"})

_DIGITS = "0123456789"
_WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_QUAD_MIDDLE = re.compile(r"\.\d{1,3}\.\d{1,3}\.\d")
QUAD = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_VERSION_BEFORE = re.compile(
    r"(?:§|(?<=\w)-|[=<>~!]=\s*|\b(?:ISA|PTX|CUDA|cuBLAS|cuDNN|NCCL|version|v|section)\s*[=:]?\s*[\"']?)$", re.I)

_NEBIUS = ("computeinstance", "computedisk", "computeimage", "computefilesystem", "computegpucluster", "project", "tenant",
           "serviceaccount", "vpcnetwork", "vpcsubnet", "vpcallocation", "vpcsecuritygroup", "publickey", "accesskey",
           "storagebucket", "mk8scluster", "mk8snodegroup", "registry", "iamgroup")
_STOP = r"[^\s)\]\"'`>]"

KINDS = {
    "provider id": [("-", r"\b(?:" + "|".join(_NEBIUS) + r")-[eu]0\d[a-z0-9]{12,}"),
                    ("prj_", r"\bprj_[A-Za-z0-9]{20,}"), ("team_", r"\bteam_[A-Za-z0-9]{20,}")],
    "endpoint": [(".r2.cloudflarestorage.com", r"\b[0-9a-f]{32}\.r2\.cloudflarestorage\.com"),
                 (".r2.dev", r"\bpub-[0-9a-f]{32}\.r2\.dev"),
                 (".workers.dev", r"[a-z0-9-]+\.workers\.dev"), (".pages.dev", r"[a-z0-9-]+\.pages\.dev"),
                 (".vercel.app", r"[a-z0-9-]+\.vercel\.app"), (".proxy.runpod.net", r"[a-z0-9-]+\.proxy\.runpod\.net"),
                 ("@ssh.runpod.io", r"[a-z0-9-]+@ssh\.runpod\.io"),
                 (".iam.gserviceaccount.com", r"[a-z0-9-]+\.iam\.gserviceaccount\.com"), ("arn:aws:", r"arn:aws:[a-z0-9-]+:"),
                 (".neon.tech", r"\bep-[a-z0-9-]+\.[a-z0-9-]+\.aws\.neon\.tech"),
                 (".amazonaws.com", r"[a-z0-9.-]+\.s3[.-](?:[a-z0-9-]+\.)?amazonaws\.com")],
    "slack id": [("0", r"(?<![A-Za-z0-9])[CDGSTUWB]0[A-Z0-9]{7,10}(?![A-Za-z0-9])"),
                 (".slack.com", r"\b(?!(?:api|app|hooks|files|status|www|edgeapi|docs)\.)[a-z0-9-]{3,}\.slack\.com\b")],
    "notion link": [("notion.", rf"(?<![\w.])(?:www\.|app\.)?notion\.(?:so|site|com)/(?!v1/){_STOP}{{3,}}")],
    "linear link": [("linear.app/", rf"\blinear\.app/(?!oauth/)[a-z0-9-]{{3,}}/{_STOP}*")],
}
_ANCHORS = {"slack id": re.compile(r"0[A-Z0-9]{7,10}"), "provider id": re.compile(r"-[eu]0\d")}
_KINDS = [(kind, hint, re.compile(rx), _ANCHORS[kind] if len(hint) == 1 else None)
          for kind, rules in KINDS.items() for hint, rx in rules]
_FAKE = re.compile(r"example|fake|dummy|acme", re.I)


def quads(text: str):
    seen = set()
    for m in _QUAD_MIDDLE.finditer(text):
        s = m.start()
        while s > 0 and m.start() - s < 3 and text[s - 1] in _DIGITS:
            s -= 1
        q = QUAD.match(text, s)
        if s == m.start() or s in seen or not q:
            continue
        seen.add(s)
        before, after = text[s - 1:s], text[q.end():q.end() + 2]
        if (before and (before in _WORD or before in ".§")) or (after[:1] and (after[0] in _WORD or after[0] == "-")) \
                or (after[:1] == "." and after[1:2] in set(_DIGITS)) or _VERSION_BEFORE.search(text[max(0, s - 16):s]):
            continue
        yield s, q.group()


def _public(quad: str) -> bool:
    try:
        return ipaddress.IPv4Address(quad).is_global and quad not in RESOLVERS
    except ValueError:
        return False


def findings(text: str) -> list[tuple[str, int]]:
    found = [("ipv4", s) for s, quad in quads(text) if _public(quad)]
    seen = set()
    for kind, hint, rx, anchor in _KINDS:
        if hint not in text:
            continue
        spans = [(max(0, a.start() - 20), a.start() + 20) for a in anchor.finditer(text)] if anchor else [(0, len(text))]
        for at, end in spans:
            for m in rx.finditer(text, at, end):
                if not _FAKE.search(m.group()) and (kind, m.start()) not in seen:
                    seen.add((kind, m.start()))
                    found.append((kind, m.start()))
    return [(kind, text.count("\n", 0, at) + 1) for kind, at in sorted(found, key=lambda x: x[1])]


def scan(files: Iterable[str], read: Callable[[str], str | None]) -> dict[str, list]:
    out = {}
    for f in files:
        if f in SKIP or os.path.splitext(f)[1] in BINARY:
            continue
        if (text := read(f)) and (found := findings(text)):
            out[f] = found
    return out


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def reader(root: Path) -> Callable[[str], str | None]:
    """A file's text, or None when it is deleted in the working tree or is a submodule's directory."""
    def read(f: str) -> str | None:
        path = root / f
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    return read
