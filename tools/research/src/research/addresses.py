"""Addresses and service ids written in text: a public IPv4 address, a cloud provider's resource id or endpoint, or a Slack,
Notion or Linear id or link.  `findings` reads one text and `scan` a list of files; `tests/test_literal_addresses.py`
holds the tree to them, and `benchmarks/security_findings/open_findings.py` counts the tracked files holding a public
address with them.  A documentation address (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24), a private one, a public
resolver and an id that says it is fake (`example`, `fake`, `dummy`) are none.  Each pattern starts with a literal, so a
whole tree scans in seconds.
"""
from __future__ import annotations

import ipaddress
import os
import re
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

#: Public resolvers a script routes toward to find its own interface; they are nobody's machine.
RESOLVERS = {"1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9"}
#: Binary data, never read: the tree's one list, which `tests/test_repository.py`'s scan for credentials reads too.
BINARY = frozenset({".bin", ".npy", ".npz", ".lgto", ".neg", ".proof", ".gz", ".png", ".pdf", ".olean", ".ilean", ".tar", ".zip"})
#: Files never scanned: `uv.lock`, whose dotted numbers are package versions, and the test's toy files, which name each form.
SKIP = frozenset({"uv.lock", "tests/test_literal_addresses.py"})

_DIGITS = "0123456789"
_WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
#: the middle of a dotted quad (`.0.2.4` of `192.0.2.4`), from which `quads` widens to the whole
_QUAD_MIDDLE = re.compile(r"\.\d{1,3}\.\d{1,3}\.\d")
#: any dotted quad, address or not
QUAD = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
#: A dotted quad that is a version or a section number, not an address: `PTX ISA 9.7.14.5`, `§2.1.3.2`, `section 1.4.1.1`,
#: `==12.9.1.4`, `pkg-13.0.3.0`. A hyphen marks one only after a word: `${H:-ADDRESS}` is an address.
_VERSION_BEFORE = re.compile(
    r"(?:§|(?<=\w)-|[=<>~!]=\s*|\b(?:ISA|PTX|CUDA|cuBLAS|cuDNN|NCCL|version|v|section)\s*[=:]?\s*[\"']?)$", re.I)

_NEBIUS = ("computeinstance", "computedisk", "computeimage", "computefilesystem", "computegpucluster", "project", "tenant",
           "serviceaccount", "vpcnetwork", "vpcsubnet", "vpcallocation", "vpcsecuritygroup", "publickey", "accesskey",
           "storagebucket", "mk8scluster", "mk8snodegroup", "registry", "iamgroup")
_STOP = r"[^\s)\]\"'`>]"

#: kind: [(a substring the file holds, the pattern then tried)]; a pattern that starts with a character class is tried only
#: near each match of its anchor (`_ANCHORS`)
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
    """(start, the quad) of each dotted quad of ``text`` that reads as an address: not part of a longer dotted token or a
    word's hyphenated one, and not a version or section number."""
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
    """(kind, line) of each address or service id ``text`` names, in order; a public IPv4 address's kind is ``ipv4``."""
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
    """{file: its findings} for each file of ``files`` that has some, each read with ``read`` (None: nothing to read),
    leaving out `SKIP` and `BINARY`."""
    out = {}
    for f in files:
        if f in SKIP or os.path.splitext(f)[1] in BINARY:
            continue
        if (text := read(f)) and (found := findings(text)):
            out[f] = found
    return out


def tracked(root: Path) -> list[str]:
    """The files git tracks in the checkout at ``root``, relative to it."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def reader(root: Path) -> Callable[[str], str | None]:
    """A `scan` reader of the checkout at ``root``: a file's text, or None when it is deleted in the working tree or is a
    submodule's directory."""
    def read(f: str) -> str | None:
        path = root / f
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
    return read
