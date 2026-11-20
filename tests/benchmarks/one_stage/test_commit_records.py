"""A Commit run publishes only listed files: `adapters/vllm/tools/served_run/published.sh` lists each with the reason it holds
nothing of a request. The list never matches a file known to hold a value, and a file the published set gains without an entry
fails `unlisted`."""
from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PUBLISHED_SH = ROOT / "adapters" / "vllm" / "tools" / "served_run" / "published.sh"
ROW = "llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager"
#: what every file a stage leaves carries here, and no path can
PLANTED = "a planted request value"

BUNDLE, KEEP = "commit/replay_bundle_p0", "commit/replay_slim_p0"
#: a row dir's files that hold values, by the stage that leaves them: the prompts and the access trace, the token records
#: and the capture, the Commit's records, bundle and keep
VALUES = {
    "match": ("match/spec.json", "match/accesses.jsonl.gz", "match/capture/tokens.json", "match/control/tokens.json",
              "match/capture/log.jsonl.gz"),
    "commit": ("commit/tokens_p0.json", "commit/verdict.json", "commit/versions.json", "commit/runs.jsonl",
               f"{BUNDLE}/MANIFEST.json", f"{BUNDLE}/context.json", f"{BUNDLE}/store/salts/s0.bin", f"{BUNDLE}/weights/tree.json"),
    "replay": ("commit/sampled_replay_p0.json", f"{KEEP}/slim.json", f"{KEEP}/context.json", f"{KEEP}/store/raw/r0.bin"),
}
#: the row's records and logs whose failing copy can quote a served token, and the ones that carry the requests' shape
QUOTING = {
    "build": ("build.log", "build_summary.json", "manifest.json", "manifest.log", "strict_word.json", "effective_config.json",
              "build_request_LP4096_T32/digest.txt"),
    "match": ("match.log", "match/run.json", "match/program.json", "match/instances.jsonl", "match/CARD.md", "match/resolution.md"),
    "commit": ("commit.log", "verdict.json"),
    "replay": ("replay.log", "config_record.json"),
}
#: the records every stage appends to
EVERY_STAGE = ("stages.txt", "row.log")
STAGES = ("build", "match", "commit", "replay")
#: a name with no entry, as a stage, a later change or a stray tool could leave it
STRAYS = ("stray.json", "match/stray.json")

def entries() -> list[tuple[str, str, str]]:
    """published.sh's list, as bash reads it: (whose, pattern, why) per entry."""
    got = subprocess.run(["bash", "-c", f'. {shlex.quote(str(PUBLISHED_SH))} && printf "%s\\n" "${{PUBLISHED[@]}}"'],
                         capture_output=True, text=True, check=True)
    return [tuple(line.split("|", 2)) for line in got.stdout.splitlines()]  # type: ignore[misc]


def has_entry(rel: str, whose: str | None = None) -> bool:
    return any(fnmatch.fnmatchcase(rel, pattern) for w, pattern, _ in entries() if whose in (None, w))


def row_records() -> list[str]:
    """The row dir's files the list names outright, which a stand-in row dir is given."""
    return sorted(pattern for whose, pattern, _ in entries() if whose == "row" and not re.search(r"[*?\[]", pattern))


def unlisted(published: Path) -> list[str]:
    """The published dir's files with no entry in published.sh, and any file at all that carries a planted value."""
    files = {p.relative_to(published).as_posix(): p for p in sorted(published.rglob("*")) if not p.is_dir()}
    return sorted(name for name, p in files.items()
                  if p.is_symlink() or not has_entry(name) or PLANTED.encode() in p.read_bytes())


def test_every_entry_says_why_and_none_matches_a_file_that_holds_a_value():
    es = entries()
    assert es and all(len(e) == 3 for e in es)
    for whose, pattern, why in es:
        assert whose in ("row", "run") and pattern and len(why.split()) >= 4, f"an entry is WHOSE|PATTERN|why: {whose}|{pattern}|{why}"
    private = [*(rel for group in (VALUES, QUOTING) for rels in group.values() for rel in rels), *EVERY_STAGE, *STRAYS,
               *(f"{s}.out" for s in STAGES), "sizes.txt", "result.json", "evidence/row.log", "evidence/match/run.json"]
    assert [rel for rel in private if has_entry(rel)] == []
    assert row_records(), "the stand-in row dir gets every record the list names"


def test_a_published_file_with_no_entry_is_found(tmp_path):
    """What the tests below hold a published dir to: a file with no entry, a link, or a listed file carrying a value."""
    (tmp_path / "match").mkdir()
    for rel, text in {"machine.txt": "a node", "stray.json": "{}", "match/spec.json": "{}", "steps.txt": PLANTED}.items():
        (tmp_path / rel).write_text(text)
    (tmp_path / "verdicts.txt").symlink_to(tmp_path / "stray.json")
    assert unlisted(tmp_path) == ["match/spec.json", "steps.txt", "stray.json", "verdicts.txt"]

