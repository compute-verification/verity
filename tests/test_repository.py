"""Repository invariants (AGENTS.md "Where writing goes"): the tree holds executable code, tests, small fixtures and a
few maintained current-truth documents.  Lane reports, handoffs, findings, drafts and their evidence (explore JSON,
plots, gate outputs, run results) live in the notes store (``~/.research/notes``), never in Git.  Experimental evidence,
run outputs and model weights live in artifact storage: every tracked blob stays under ``BLOB_LIMIT`` bytes.

Enforced here rather than remembered.  ``.gitignore`` keeps the report-genre paths out of ``git add``; this test is the
backstop for anything that slips past it.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Basenames a tracked markdown file may have.  Everything else is a report, a handoff or a draft and belongs in the
#: notes store (cite it from here by ``note:<id>``).
ALLOWED_MD_BASENAMES = frozenset({"README.md", "AGENTS.md", "PROTOCOL.md", "DESIGN.md", "SKILL.md", "DISCREPANCIES.md",
                                  "CHANGELOG.md"})
#: Layout/spec documents left in place by the 2026-09-22 migration; a human folds them into PROTOCOL.md / README.md.
ALLOWED_EXTRA = (
    "backends/direct/V2_AIR.md",
    "backends/direct/encode/V2_SHAPE.md",
    "backends/vole/ASSUMPTIONS.md",
    "backends/flock/verifier/lean/soundness/ASSUMPTIONS.md",
    "backends/numerical/python/verity_numerical/redteam/TRANSCRIPT_CHECKLIST.md",
)
#: One file per entry behind an index document, so that no one document outgrows its cap: the Flock soundness package's
#: detail behind its named assumptions (its ``ASSUMPTIONS.md`` states them in full) and ligero-verify's discrepancies
#: (indexed by its ``DISCREPANCIES.md``).
SPLIT_DOCUMENT_DIRS = ("backends/flock/verifier/lean/soundness/assumptions/", "backends/ligero-verify/discrepancies/")
#: Evidence files that code still reads (``backends/shared/anchor_a100/report.py`` inputs, ``reprice_v2.py --v1``,
#: ``tensor/fused/bench.py`` PREVIOUS_RUN).  Pinned until those scripts read from the notes store; nothing else may
#: live under ``reports/`` or ``results/``.
PINNED_EVIDENCE = (
    "backends/numerical/reports/b_measured_4090.json",
    "backends/numerical/reports/fused_logup_4090.json",
    "backends/numerical/reports/fused_sumcheck_4090.json",
    "backends/numerical/reports/packed_sumcheck_h100.json",
    "backends/numerical/reports/tc_sumcheck_4090.json",
    "backends/direct/ligero/results/result_unit_b4096.json",
    "backends/direct/ligero/results/result_vu_42.json",
    "backends/direct/ligero/results/result_vu_85.json",
    "backends/gkr/packed/results/r20260922-064917-c293/bench.json",
    "backends/gkr/packed/results/r20260922-065412-73be/bench.json",
    "backends/gkr/packed/results/r20260922-073439-1f15/logup_bench.json",
)
PINNED_EVIDENCE_DIRS = ("backends/shared/anchor_a100/results/",)

REPORT_GENRE = re.compile(r"handoff|report|urgent|draft|triage|status", re.IGNORECASE)
#: Source files are exempt from the basename rule (``report.py`` is a script, not a report).
SOURCE_SUFFIXES = frozenset({".py", ".rs", ".toml", ".sh", ".cu", ".c", ".h", ".lock"})

MD_SIZE_CAP = 48 * 1024
PROTOCOL_SIZE_CAP = 128 * 1024
BLOB_LIMIT = 5 * 1024 * 1024  # 5 MiB; the largest fixture today is 4.5 MB
#: Must-reject proofs the ligero verifiers' regression tests read from the tree (steps-pin H1/H2, red-team-leaf-2 G1), merged
#: 2026-09-23 in lane/steps-pin; each is capped at its own size, so they cannot grow and nothing else joins them silently.
OVERSIZE_FIXTURES = {
    "backends/ligero-verify/fixtures/fp8-ada-ajtai-n64-g1-decoy/sub_00.proof": 6 * 1024 * 1024,
    "backends/ligero-verify/fixtures/steps-pin/bf16-hopper-ajtai-n128-collide-P/sub_00.proof": 6 * 1024 * 1024,
    "backends/ligero-verify/fixtures/steps-pin/bf16-hopper-ajtai-n128-collide-Q/sub_00.proof": 6 * 1024 * 1024,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-ajtai-n64-collide-P/sub_00.proof": 6 * 1024 * 1024,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-ajtai-n64-collide-Q/sub_00.proof": 6 * 1024 * 1024,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-ajtai-n64-steps32/sub_00.proof": 6 * 1024 * 1024,
    "backends/ligero-verify/fixtures/steps-pin/fp8-ada-poseidon2-steps32/sub_00.proof": 7 * 1024 * 1024,
}


def _tracked() -> list[str]:
    if not (ROOT / ".git").exists():                # a shipped `git archive` (research run --source): its files are the commit's,
        out = []                                    # beside what the machine adds (the runner's READY.json, uv's .venv, bytecode)
        for d, dirs, files in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in (".venv", "__pycache__")]
            out += [os.path.relpath(os.path.join(d, f), ROOT) for f in files]
        return sorted(p for p in out if p != "READY.json")
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def _is_pinned(path: str) -> bool:
    return path in PINNED_EVIDENCE or path.startswith(PINNED_EVIDENCE_DIRS)


def test_tracked_markdown_is_maintained_documentation() -> None:
    offenders = [p for p in _tracked() if p.endswith(".md") and os.path.basename(p) not in ALLOWED_MD_BASENAMES
                 and not p.endswith(".template.md") and p not in ALLOWED_EXTRA
                 and not p.startswith(SPLIT_DOCUMENT_DIRS)]
    assert not offenders, (
        "tracked markdown that is not a maintained document (reports, handoffs and drafts belong in ~/.research/notes):\n"
        + "\n".join(offenders)
    )


def test_no_report_genre_basenames() -> None:
    offenders = [p for p in _tracked() if REPORT_GENRE.search(os.path.basename(p))
                 and os.path.splitext(p)[1] not in SOURCE_SUFFIXES]
    assert not offenders, "report-genre files belong in ~/.research/notes/lanes/<lane>/:\n" + "\n".join(offenders)


def test_no_evidence_under_reports_or_results() -> None:
    offenders = []
    for p in _tracked():
        parts = p.split("/")
        if "results" in parts[:-1] and not _is_pinned(p):
            offenders.append(p)
        elif "reports" in parts[:-1]:
            if re.fullmatch(r".*/reports/ledger/[^/]+\.jsonl", p) or _is_pinned(p):
                continue
            offenders.append(p)
    assert not offenders, (
        "evidence under reports/ or results/ belongs in ~/.research/notes/campaigns/<campaign>/assets/ "
        "(only reports/ledger/*.jsonl and PINNED_EVIDENCE stay):\n" + "\n".join(offenders)
    )


def test_pinned_evidence_still_exists() -> None:
    """Drop entries from PINNED_EVIDENCE when the scripts that read them move to the notes store."""
    tracked = set(_tracked())
    missing = [p for p in PINNED_EVIDENCE if p not in tracked]
    assert not missing, "PINNED_EVIDENCE lists files that are no longer tracked; remove them:\n" + "\n".join(missing)


def test_markdown_size_caps() -> None:
    offenders = []
    for p in _tracked():
        if not p.endswith(".md"):
            continue
        cap = PROTOCOL_SIZE_CAP if os.path.basename(p) == "PROTOCOL.md" else MD_SIZE_CAP
        try:
            size = os.path.getsize(ROOT / p)
        except FileNotFoundError:  # tracked but deleted in the working tree
            continue
        if size > cap:
            offenders.append(f"{p}: {size / 1024:.0f} KB > {cap // 1024} KB")
    assert not offenders, "maintained documents are small; move detail to the notes store:\n" + "\n".join(offenders)


def test_no_tracked_blob_exceeds_limit() -> None:
    offenders = []
    for p in _tracked():
        try:
            size = os.path.getsize(ROOT / p)
        except FileNotFoundError:  # tracked but deleted in the working tree
            continue
        if size > OVERSIZE_FIXTURES.get(p, BLOB_LIMIT):
            offenders.append(f"{p}: {size / 1048576:.1f} MiB")
    assert not offenders, (
        "tracked files over the blob limit (evidence and run outputs belong in artifact storage, not Git):\n"
        + "\n".join(offenders)
    )
