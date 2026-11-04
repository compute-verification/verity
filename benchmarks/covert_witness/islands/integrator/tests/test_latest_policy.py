"""latest_policy.py picks the newest refresh run's algorithm-policy table at te_core 4.13 or later from both of the
scorer's meta forms (the table's `what` with `source` and `refresh`, and a refresh's "... uncut tables ..." `what`),
skips older versions, arts missing a file or a run, and malformed arts, and prints FALLBACK when none qualify or the
store can't be read."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("latest_policy", Path(__file__).resolve().parents[1] / "latest_policy.py")
LP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LP)

FILES = ["algorithm-policy.json", "coverage.json", "frontier-fit.json", "frontier-points.json"]


def table(art, version, refresh, files=FILES):
    return {"id": art, "manifest": {"payload": {"files": [{"path": p} for p in files]}, "meta": {
        "what": LP.WHAT, "source": f"python -m te_core.algorithm_policy (te_core {version}; pin 024b)",
        "refresh": f"{refresh} on vy-nebius-cpu-7" if refresh else ""}}}


def refresh(art, version, run, files=FILES):
    return {"id": art, "manifest": {"payload": {"files": [{"path": p} for p in files]}, "meta": {
        "what": f"refresh {run} uncut tables (algorithm-policy, coverage, frontier-fit), te_core {version} (bbed995)"}}}


TABLE = table("art:table", "4.13.1", "r20261104-114655-3fc3")
REFRESH = refresh("art:refresh", "4.13.1", "r20261104-141224-e7fc")


def test_newest_refresh_run_wins_across_both_forms():
    assert LP.pick([TABLE, REFRESH]) == "art:refresh"
    assert LP.pick([REFRESH, TABLE]) == "art:refresh"
    assert LP.pick([TABLE]) == "art:table"
    assert LP.pick([REFRESH]) == "art:refresh"


def test_skips_old_versions_missing_files_runs_and_malformed_arts():
    newer = "r20261104-180000-aaaa"
    assert LP.pick([TABLE, table("art:old", "4.12.14", newer), refresh("art:old2", "4.1", newer)]) == "art:table"
    assert LP.pick([TABLE, refresh("art:nofit", "4.13.3", newer, files=FILES[:2])]) == "art:table"
    assert LP.pick([TABLE, table("art:norun", "4.13.3", None)]) == "art:table"
    assert LP.pick([{"id": "art:bad"}, TABLE, {"id": "art:bad2", "manifest": {"meta": {}}}]) == "art:table"


def test_fallback_when_nothing_qualifies_or_no_store(monkeypatch, capsys):
    assert LP.pick([]) == LP.FALLBACK
    assert LP.pick([table("art:old", "4.12.14", "r20261104-180000-aaaa")]) == LP.FALLBACK

    def broken(*where):
        raise OSError("no store")

    monkeypatch.setattr(LP, "select", broken)
    assert LP.main() == 0
    assert capsys.readouterr().out.strip() == LP.FALLBACK


def test_main_reads_both_selects(monkeypatch, capsys):
    other = {"id": "art:other", "manifest": {"payload": {"files": []}, "meta": {"what": "something else"}}}
    monkeypatch.setattr(LP, "select", lambda *where: [TABLE] if where else [TABLE, REFRESH, other])
    LP.main()
    assert capsys.readouterr().out.strip() == "art:refresh"
