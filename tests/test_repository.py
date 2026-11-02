"""Repository invariants (AGENTS.md "Where writing goes"): the tree holds executable code, tests, small fixtures and a
few maintained current-truth documents.  Lane reports, handoffs, findings and drafts live in the notes (``~/.research/notes``),
never in Git.  Generated output (rendered tables, reports, plots, evidence copies, run results) is never tracked: it is
regenerated from code plus the evidence store, or fetched from the store by ``art:`` id.  Model weights live in artifact
storage too: every tracked blob stays under ``BLOB_LIMIT`` bytes.

Enforced here rather than remembered.  ``.gitignore`` keeps the report-genre paths out of ``git add``; this test is the
backstop for anything that slips past it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Basenames a tracked markdown file may have.  Everything else is a report, a handoff or a draft and belongs in the
#: notes store (cite it from here by ``note:<id>``).
ALLOWED_MD_BASENAMES = frozenset({"README.md", "AGENTS.md", "ARCHITECTURE.md", "PROTOCOL.md", "DESIGN.md", "SKILL.md",
                                  "DISCREPANCIES.md", "CHANGELOG.md"})
#: The Glossary's second file, at the root only: no directory holds a glossary.
ROOT_ONLY_MD = frozenset({"GLOSSARY.md"})
#: Layout/spec documents left in place by the layout migration; a human folds them into PROTOCOL.md / README.md.
ALLOWED_EXTRA = (
    "Security/Proofs/Flock/Soundness/ASSUMPTIONS.md",
)
#: One file per entry behind an index document, so that no one document outgrows its cap: {directory: its index}. Flock
#: soundness's detail behind its named assumptions (its ``ASSUMPTIONS.md`` states them in full) and the network certifier's sections (``protocol/<topic>.md``, each
#: keeping its heading in ``PROTOCOL.md`` with a one-line pointer).
SPLIT_DOCUMENT_DIRS = {"Security/Proofs/Flock/Soundness/assumption-notes/": "Security/Proofs/Flock/Soundness/ASSUMPTIONS.md",
                       "verity/core/protocols/certifier/protocol/": "verity/core/protocols/certifier/PROTOCOL.md"}
#: Generated output, by where it sits or what it is: no tracked path has one of these directories, suffixes or stamps.
GENERATED_DIRS = frozenset({"reports", "results", "renders", "previews", "plots", "evidence", "assets", "outputs"})
GENERATED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pdf", ".html", ".htm"})
#: ``<YYYYMMDDTHHMMZ>-...``: a stamped render (the steward's ``<stamp>-tables.json``) or a lane note.
STAMPED = re.compile(r"^\d{8}T\d{4}Z[-_.]")

REPORT_GENRE = re.compile(r"handoff|report|urgent|draft|triage|status|^results?_", re.IGNORECASE)
#: Source files are exempt from the basename rule (``report.py`` is a script, not a report).
SOURCE_SUFFIXES = frozenset({".py", ".rs", ".toml", ".sh", ".cu", ".c", ".h", ".lock"})

MD_SIZE_CAP = 48 * 1024
PROTOCOL_SIZE_CAP = 256 * 1024
#: A skill is loaded into an agent's context whole, so it states the current procedure and rules, never their history
#: (`.agents/skills/authoring-skills/`).
SKILL_SIZE_CAP = 16 * 1024
#: Skills over SKILL_SIZE_CAP when it was set, each capped at its size then (the largest at any open PR's head, rounded up,
#: plus 1 KiB for PRs queued against it). An entry only shrinks, and goes once its skill is under SKILL_SIZE_CAP; its owner
#: compacts it.
OVERSIZE_SKILLS: dict[str, int] = {}
BLOB_LIMIT = 256 * 1024  # a tracked file holds at most this, except source code (CODE_SUFFIXES), ALLOWLIST and audit records
#: Suffixes of source, exempt from BLOB_LIMIT (a generated module or a pinned Lean file can be larger); markdown has its own caps.
CODE_SUFFIXES = SOURCE_SUFFIXES | {".lean", ".cuh", ".cpp", ".md"}
#: Data over BLOB_LIMIT that is not a test fixture: runtime tables a module loads, the generated census, the vLLM checkpoint
#: manifest (append-only; most of its readers don't read the sibling manifests yet). Each is capped at its size rounded up
#: to 64 KiB, so none can grow past that and nothing joins them silently. A test fixture over BLOB_LIMIT is registered in
#: fixtures/artifacts.json instead and fetched (`research data fetch-fixtures`).
ALLOWLIST = {
    "benchmarks/census/workloads.json": 384 * 1024,
    "integrations/vllm/manifests/checkpoints.json": 320 * 1024,
    "integrations/vllm/verity_vllm/check/tables/B0-divergence-20260907T1604Z/cos_sin_cache.npy": 1088 * 1024,
    "verity/ml/circuits/ops/tables/mufu_tanh_sm89.xzblocks": 448 * 1024,
}
AUDIT_RECORD = "lean-audit.json"
#: A Lean package's audit record must stay in Git, since the audit compares the build against it and `--update`'s diff is
#: what the spec alert shows when a guarantee changes; it grows 3-5 KiB per pin and with every definition under `meaning` (following the whole
#: executable verifier, `Flock`, took soundness's record past 2 MiB, and `Security`'s holds every package's guarantees).
AUDIT_RECORD_LIMIT = 6 * 1024 * 1024
#: Tracked data (everything but CODE_SUFFIXES and audit records) in the whole tree, so that many small files cannot
#: add up to what one large one may not be. Audit records are left out: they must stay in Git and each has its own
#: AUDIT_RECORD_LIMIT, so counting them here would make every new guarantee spend the fixtures' budget. 48 MiB:
#: the E2E demo's fixtures took the tree to 40.1 MiB; register the next large one rather than raising it again.
DATA_BUDGET = 48 * 1024 * 1024
REGISTRY = "fixtures/artifacts.json"


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def test_tracked_markdown_is_maintained_documentation() -> None:
    offenders = [p for p in _tracked() if p.endswith(".md") and os.path.basename(p) not in ALLOWED_MD_BASENAMES
                 and p not in ROOT_ONLY_MD
                 and not p.endswith(".template.md") and p not in ALLOWED_EXTRA
                 and not p.startswith(tuple(SPLIT_DOCUMENT_DIRS))]
    assert not offenders, (
        "tracked markdown that is not a maintained document (reports, handoffs and drafts belong in ~/.research/notes):\n"
        + "\n".join(offenders)
    )


def test_no_report_genre_basenames() -> None:
    offenders = [p for p in _tracked() if REPORT_GENRE.search(os.path.basename(p))
                 and os.path.splitext(p)[1] not in SOURCE_SUFFIXES]
    assert not offenders, "report-genre files belong in ~/.research/notes/lanes/<lane>/:\n" + "\n".join(offenders)


def generated_output(path: str) -> str | None:
    """Why ``path`` looks like generated output, or None."""
    parts = path.split("/")
    if hit := GENERATED_DIRS.intersection(parts[:-1]):
        return f"under {sorted(hit)[0]}/"
    if os.path.splitext(path)[1].lower() in GENERATED_SUFFIXES:
        return f"a {os.path.splitext(path)[1].lower()} render"
    if STAMPED.match(parts[-1]):
        return "a stamped render or note"
    return None


def test_generated_output_patterns() -> None:
    assert generated_output("backends/numerical/reports/ledger/b-zk.jsonl") == "under reports/"
    assert generated_output("renders/daily/20260927T0857Z-tables.json") == "under renders/"
    assert generated_output("lanes/x/evidence/run.log") == "under evidence/"
    assert generated_output("benchmarks/numerical/plot_overhead.png") == "a .png render"
    assert generated_output("integrations/vllm/20260929T0300Z-tables.json") == "a stamped render or note"
    assert generated_output("benchmarks/numerical/python/verity_numerical/bench/contract.py") is None
    assert generated_output("tools/research/tests/fixtures/run_v0.2/out/result.json") is None


def test_no_generated_output() -> None:
    """Rendered tables, reports, plots, evidence copies and run results are regenerated from code plus the evidence store, or
    fetched from it by ``art:`` id (``research data put --preserve``); none is tracked."""
    offenders = [f"{p} ({why})" for p in _tracked() if (why := generated_output(p))]
    assert not offenders, (
        "generated output belongs in the evidence store, not Git (AGENTS.md \"Where writing goes\"):\n" + "\n".join(offenders)
    )


def split_dir(path: str) -> str | None:
    """The `SPLIT_DOCUMENT_DIRS` directory behind ``path``: the one it sits in, or the one it indexes (``ASSUMPTIONS.md``
    indexes ``assumption-notes/``); None for any other document."""
    return next((d for d, index in SPLIT_DOCUMENT_DIRS.items() if path.startswith(d) or path == index), None)


def markdown_cap(path: str) -> int:
    """The markdown file ``path``'s cap in bytes.  The merge queue's landing gate (`tools/verity/check/gates/markdown_caps.py`)
    reads it and `markdown_over_cap` from this file at `main`, so keep both importable with the standard library alone."""
    if os.path.basename(path) == "SKILL.md":
        return OVERSIZE_SKILLS.get(path, SKILL_SIZE_CAP)
    return PROTOCOL_SIZE_CAP if os.path.basename(path) == "PROTOCOL.md" else MD_SIZE_CAP


def markdown_over_cap(path: str, size: int) -> str | None:
    """Why the markdown file ``path`` of ``size`` bytes is over its cap, and where its detail goes; None within it."""
    if size <= (cap := markdown_cap(path)):
        return None
    if os.path.basename(path) == "SKILL.md":
        return (f"{path}: {size / 1024:.0f} KB > {cap // 1024} KB, {size - cap} bytes over; compact it: keep the current "
                "procedure and rules, drop what AGENTS.md or another skill states, and move dates and history to the "
                "decisions log or a note (.agents/skills/authoring-skills/SKILL.md)")
    where = f"one file per entry under {d}" if (d := split_dir(path)) else "the notes store (cite it by note:<id>)"
    return f"{path}: {size / 1024:.0f} KB > {cap // 1024} KB, {size - cap} bytes over; move detail to {where}"


def test_markdown_over_cap_says_by_how_much_and_where_detail_goes(monkeypatch) -> None:
    soundness = "Security/Proofs/Flock/Soundness/"
    assert markdown_over_cap("README.md", MD_SIZE_CAP) is None and markdown_over_cap("a/PROTOCOL.md", MD_SIZE_CAP + 1) is None
    assert markdown_over_cap("a/README.md", MD_SIZE_CAP + 100) == (
        "a/README.md: 48 KB > 48 KB, 100 bytes over; move detail to the notes store (cite it by note:<id>)")
    for p in (soundness + "ASSUMPTIONS.md", soundness + "assumption-notes/e2e-checklist.md"):
        assert markdown_over_cap(p, MD_SIZE_CAP + 7).endswith(f", 7 bytes over; move detail to one file per entry under {soundness}assumption-notes/")
    assert markdown_over_cap(soundness + "README.md", MD_SIZE_CAP + 1).endswith("the notes store (cite it by note:<id>)")
    certifier = "verity/core/protocols/certifier/"
    assert markdown_over_cap(certifier + "PROTOCOL.md", PROTOCOL_SIZE_CAP + 1).endswith(f"one file per entry under {certifier}protocol/")
    assert markdown_over_cap(certifier + "protocol/calibration.md", MD_SIZE_CAP + 1).endswith(f"{certifier}protocol/")
    assert markdown_over_cap(".agents/skills/x/SKILL.md", SKILL_SIZE_CAP) is None
    assert markdown_over_cap(".agents/skills/x/SKILL.md", SKILL_SIZE_CAP + 5).startswith(
        ".agents/skills/x/SKILL.md: 16 KB > 16 KB, 5 bytes over; compact it")
    big, cap = ".agents/skills/x/SKILL.md", SKILL_SIZE_CAP + 1024
    monkeypatch.setitem(OVERSIZE_SKILLS, big, cap)
    assert markdown_over_cap(big, cap) is None and "1 bytes over" in markdown_over_cap(big, cap + 1)


def test_oversize_skills_only_shrink() -> None:
    """An OVERSIZE_SKILLS entry names a skill that exists and is still over SKILL_SIZE_CAP; once it's under, the entry goes,
    so the list only shrinks."""
    stale = [f"{p}: {'missing' if not (ROOT / p).exists() else f'{os.path.getsize(ROOT / p)} bytes, under the cap'}"
             for p in OVERSIZE_SKILLS if not (ROOT / p).exists() or os.path.getsize(ROOT / p) <= SKILL_SIZE_CAP]
    assert not stale, "delete these OVERSIZE_SKILLS entries in tests/test_repository.py:\n" + "\n".join(stale)


def test_markdown_size_caps() -> None:
    offenders = []
    for p in _tracked():
        if not p.endswith(".md"):
            continue
        try:
            size = os.path.getsize(ROOT / p)
        except FileNotFoundError:  # tracked but deleted in the working tree
            continue
        if why := markdown_over_cap(p, size):
            offenders.append(why)
    assert not offenders, "maintained documents are small:\n" + "\n".join(offenders)


#: Test files that need torch, a GPU or cargo, and were never part of `check`.
UNSUITED_TEST_DIRS = ("verity/ml/kernels/",)


def _testpaths(tracked: list[str]) -> list[str]:
    """Every suite's testpaths, relative to the root: a suite is a directory whose pyproject.toml configures pytest."""
    testpaths = []
    for p in tracked:
        if os.path.basename(p) == "pyproject.toml":
            ini = tomllib.loads((ROOT / p).read_text()).get("tool", {}).get("pytest", {}).get("ini_options")
            if ini is not None:
                testpaths += [os.path.normpath(os.path.join(os.path.dirname(p), t)) for t in ini.get("testpaths", ["."])]
    return testpaths


def test_every_test_file_is_in_a_suite() -> None:
    """`tools/verity/check/suites.py` runs suites, so a test file outside every suite's testpaths would never run. A suite is a
    directory whose pyproject.toml configures pytest; give a new package one, or put the test in the package it tests."""
    tracked = _tracked()
    testpaths = _testpaths(tracked)
    tests = [p for p in tracked if p.endswith(".py") and (os.path.basename(p).startswith("test_") or p.endswith("_test.py"))]
    offenders = [p for p in tests if not any(t == "." or p.startswith(t + "/") for t in testpaths) and not p.startswith(UNSUITED_TEST_DIRS)]
    assert not offenders, "test files that no suite runs:\n" + "\n".join(offenders)


#: Credential formats with a fixed shape, as secret scanners (GitGuardian's PR check among them) recognize them.
CREDENTIAL_FORMATS = {
    "an AWS access key id": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "a private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "a GitHub token": r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})",
    "a Slack token": r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    "a JSON web token": r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.",
    "a RunPod API key": r"\brpa_[A-Za-z0-9]{20,}",
    "an R2 account endpoint": r"\b[0-9a-f]{32}\.r2\.cloudflarestorage\.com",
}
_FORMATS = re.compile("|".join(f"(?P<f{i}>{p})" for i, p in enumerate(CREDENTIAL_FORMATS.values())))
#: A file can match a format only if it holds one of these.
_FORMAT_HINTS = ("AKIA", "ASIA", "PRIVATE KEY", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "xox", "eyJ", "rpa_",
                 "r2.cloudflarestorage.com")
#: A value assigned to a credential name: `name = "value"`, `"name": "value"`, or `NAME=value` inside a string.
CREDENTIAL_ASSIGNMENTS = (re.compile(r"(?<![\w-])(?P<name>[A-Za-z_][\w-]*)[\"']?\s*[:=]\s*[bru]{0,2}[\"'](?P<value>[^\"'\s]{12,})[\"']"),
                          re.compile(r"\b(?P<name>[A-Z][A-Z0-9_]*)=(?P<value>[^\"'\s]{12,})"))
CREDENTIAL_WORDS = {"secret", "token", "password", "passwd", "credential", "credentials"}
CREDENTIAL_PAIRS = {("api", "key"), ("access", "key"), ("secret", "key"), ("private", "key"), ("auth", "token")}
#: A file holds a credential name only if its lowercased text holds one of these.
_NAME_HINTS = ("secret", "token", "passw", "credential", "apikey", "api_key", "api-key", "accesskey", "access_key", "access-key",
               "privatekey", "private_key", "private-key")
#: Binary data, never read; and data files (vectors, fixtures), read only for the fixed credential formats.
_BINARY = frozenset({".bin", ".npy", ".npz", ".lgto", ".neg", ".proof", ".gz", ".png", ".pdf", ".olean", ".ilean", ".tar", ".zip"})
_DATA = frozenset({".json", ".jsonl", ".csv"})
#: Values that say they are not real: documented examples (AWS's SigV4 test vector keys contain EXAMPLE) and placeholders.
PLACEHOLDER = re.compile(r"EXAMPLE|example|dummy|DUMMY|fake|FAKE|placeholder|redacted|xxxx|XXXX|\*\*\*\*|[<{$]")


def _credential_name(name: str) -> bool:
    parts = [p.lower() for p in re.split(r"[_\-.]|(?<=[a-z0-9])(?=[A-Z])", name) if p]
    return bool(parts) and (parts[-1] in CREDENTIAL_WORDS or tuple(parts[-2:]) in CREDENTIAL_PAIRS)


def _looks_random(value: str) -> bool:
    counts = [value.count(c) for c in set(value)]
    entropy = -sum(n / len(value) * math.log2(n / len(value)) for n in counts)
    classes = sum(bool(re.search(p, value)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[^A-Za-z0-9]"))
    return entropy >= 3.5 and classes >= 3


def test_no_secret_like_literals() -> None:
    """No tracked file holds something a secret scanner reads as a credential. A dummy credential is short and says what it is
    (`"dummy"`, `"not-to-be-copied"`); a test that needs a realistic one, to test redaction or scanning, splits it into adjacent
    string literals, which Python joins and no scanner reads as one."""
    offenders, formats = [], list(CREDENTIAL_FORMATS)
    for p in _tracked():
        path, suffix = ROOT / p, os.path.splitext(p)[1]
        try:
            if suffix in _BINARY or path.stat().st_size > 1024 * 1024:
                continue
            text = path.read_text()
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
            continue
        found = [(m, formats[int(m.lastgroup[1:])]) for m in _FORMATS.finditer(text) if not PLACEHOLDER.search(m.group(0))] \
            if any(h in text for h in _FORMAT_HINTS) else []
        low = text.lower()
        if suffix not in _DATA and any(w in low for w in _NAME_HINTS):
            found += [(m, f"a random-looking {m['name']}") for pattern in CREDENTIAL_ASSIGNMENTS for m in pattern.finditer(text)
                      if _credential_name(m["name"]) and _looks_random(m["value"]) and not PLACEHOLDER.search(m["value"])]
        offenders += [f"{p}:{text.count(chr(10), 0, m.start()) + 1}: {what}" for m, what in found]
    assert not offenders, "credential-like literals (see this test's docstring for what to write instead):\n" + "\n".join(offenders)


def test_no_tracked_blob_exceeds_limit() -> None:
    offenders = []
    for p in _tracked():
        if os.path.splitext(p)[1] in CODE_SUFFIXES:
            continue
        try:
            size = os.path.getsize(ROOT / p)
        except FileNotFoundError:  # tracked but deleted in the working tree
            continue
        limit = ALLOWLIST.get(p, AUDIT_RECORD_LIMIT if os.path.basename(p) == AUDIT_RECORD else BLOB_LIMIT)
        if size > limit:
            offenders.append(f"{p}: {size / 1024:.0f} KiB > {limit // 1024} KiB")
    assert not offenders, (
        f"tracked files over {BLOB_LIMIT // 1024} KiB: register a fixture in {REGISTRY} and gitignore its path "
        "(`research data put --kind fixture/v1 --preserve`, then `research data export`); evidence and run outputs belong in "
        "artifact storage, not Git:\n" + "\n".join(offenders)
    )


def test_allowlists_name_tracked_files() -> None:
    tracked = set(_tracked())
    stale = sorted(p for p in ALLOWLIST if p not in tracked)
    assert not stale, "ALLOWLIST names files that are no longer tracked; remove them:\n" + "\n".join(stale)


def test_owned_paths_exist() -> None:
    """Each handle's `owns` entries in the messaging registry name tracked paths, so a move updates its owners' map."""
    tracked = _tracked()
    reg = json.loads((ROOT / ".agents/skills/using-slack/registry.json").read_text())
    stale = sorted(f"@{h['handle']}: {p}" for h in reg["handles"] for p in h.get("owns", [])
                   if not any(t == p or t.startswith(p if p.endswith("/") else p + "/") for t in tracked))
    assert not stale, "registry `owns` entries that match no tracked path; update them:\n" + "\n".join(stale)


def test_every_path_has_owner() -> None:
    """`research msg owners` names a handle for every tracked path, so a report about any directory reaches someone."""
    from research.slack import owner

    reg = json.loads((ROOT / ".agents/skills/using-slack/registry.json").read_text())
    unowned = sorted({os.path.dirname(p) + "/" for p in _tracked() if owner(reg, p) is None})
    assert not unowned, ("directories holding tracked files no handle owns; add each to its owner's `owns` in "
                         ".agents/skills/using-slack/registry.json:\n" + "\n".join(unowned))


def test_tracked_data_within_budget() -> None:
    total = 0
    for p in _tracked():
        if os.path.splitext(p)[1] in CODE_SUFFIXES or os.path.basename(p) == AUDIT_RECORD:
            continue
        try:
            total += os.path.getsize(ROOT / p)
        except FileNotFoundError:
            continue
    assert total <= DATA_BUDGET, (f"tracked data is {total / 2**20:.1f} MiB, over the {DATA_BUDGET / 2**20:.0f} MiB budget: register large "
                                  f"fixtures in {REGISTRY} rather than tracking them")


def test_fetched_fixtures_are_gitignored() -> None:
    """An entry of the fixture registry that git does not track is fetched (`research data fetch-fixtures`), so its path must be
    ignored: a fetched copy must never be committed back."""
    if not (ROOT / ".git").exists():
        return
    doc = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    tracked = set(_tracked())
    fetched = sorted(v for k, v in doc.items() if k.startswith("art:") and isinstance(v, str)
                     and v not in tracked and not any(t.startswith(v.rstrip("/") + "/") for t in tracked))
    if not fetched:
        return
    out = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], cwd=ROOT, input="\n".join(fetched) + "\n",
                         capture_output=True, text=True).stdout.split("\n")
    unignored = sorted(set(fetched) - {x for x in out if x})
    assert not unignored, "registry entries git does not track must be gitignored (add them to .gitignore):\n" + "\n".join(unignored)
