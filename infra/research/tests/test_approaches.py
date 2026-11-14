"""The approach registry (`research.approaches`, `research notes claim | approach | approaches`): records in the evidence store whose
id is a function of the name, everything else labels; claims refuse a second live owner and a silent retry of a killed approach;
a claim race is decided by the fold, not by who wrote last; the public view is one generated file per campaign that shows what the
campaign's brief allows."""
from __future__ import annotations

import datetime as dt

import pytest

from research import approaches as A
from research import notes
from research.store import labels_sync as LS
from research.store import vocab
from research.store.local import LocalStore
from research.store.model import Label

NAME, OTHER = "pous/band-d12", "pous/dense"
NEW = dict(title="Band graph, d = 12", hypothesis="The band labelling meets the timed storage game", type_="scheme")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "CLAIM_SETTLE_S", 0)
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\nwrite_through = true\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    root = tmp_path / "notes"
    for d in ("lanes/pous-mvp", "lanes/pous-ec", "lanes/coordinator", "campaigns/pous"):
        (root / d).mkdir(parents=True)
    (root / "campaigns/pous/BRIEF.md").write_text("---\nid: pous/brief\nkind: brief\nregistry: titles\n---\n\n# POUS\n")
    return tmp_path


def store(env, name="store") -> LocalStore:
    return LocalStore(env / name, config_path=env / "store.toml")


def remote_keys(st: LocalStore, name: str) -> list[str]:
    return [Label.from_doc(r["label"]).key for r in LS.compare(st, A.record_id(name)) if r["where"] in ("both", "remote")]


def test_the_record_id_is_a_function_of_the_name_alone():
    assert A.record_id(NAME) == A.record_id(NAME) != A.record_id(OTHER)
    assert A.manifest(NAME).to_doc()["meta"] == {"campaign": "pous", "slug": "band-d12"}
    for bad in ("pous", "Pous/x", "pous/", "pous/a b", "pous/x/y"):
        with pytest.raises(A.Refused):
            A.record_id(bad)


def test_claim_writes_the_record_and_its_labels_through_to_the_remote(env):
    st = store(env)
    rc, line = A.claim(st, NAME, "pous-mvp", cites=["verity#166"], **NEW)
    assert rc == 0 and "claimed pous/band-d12 for pous-mvp" in line
    assert set(remote_keys(st, NAME)) == {"approach_status", "owner", "title", "hypothesis", "approach_type", "cites"}
    elsewhere = store(env, "store2")                        # another machine: nothing local, the claim is on the remote
    a = A.fold(NAME, A.current_labels(elsewhere, A.record_id(NAME)))
    assert (a.status, a.owner, a.type, a.cites) == ("live", "pous-mvp", "scheme", ["verity#166"])
    assert A.exists(elsewhere, NAME) and not A.exists(elsewhere, OTHER)


def test_without_a_remote_the_claim_is_a_pending_file_the_steward_records(env, tmp_path, monkeypatch, capsys):
    root = env / "notes"
    with pytest.raises(A.NoRemote):
        A.claim(LocalStore(tmp_path / "s", config_path=tmp_path / "absent.toml"), NAME, "pous-mvp", **NEW)
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    assert notes.main(["claim", NAME, "--lane", "pous-mvp", "--title", NEW["title"], "--hypothesis", NEW["hypothesis"], "--type",
                       "scheme", "--cite", "verity#166", "--root", str(root), "--store", str(tmp_path / "s")]) == 4
    assert "PENDING: no usable remote" in capsys.readouterr().err
    [p] = (root / "lanes/pous-mvp").glob("*-claim-pous--band-d12.md")
    assert A.read_pending(p)["approach"] == NAME and A.read_pending(p)["cites"] == "verity#166"
    assert notes.main(["claim", "pous/other", "--lane", "pous-ec", "--root", str(root), "--store", str(tmp_path / "s")]) == 4
    done: list[str] = []
    now = dt.datetime(2026, 9, 28, 6, 0, tzinfo=dt.timezone.utc)
    lines = A.import_pending(store(env), root, now, done)
    assert lines[0].startswith("06:00Z CLAIM-REFUSED lanes/pous-ec/") and "refused: pous/other is new" in lines[0]
    assert lines[1].startswith("06:00Z CLAIM lanes/pous-mvp/") and "claimed pous/band-d12 for pous-mvp" in lines[1]
    assert list((root / "lanes/pous-ec").glob("*-handoff-steward-claim.md"))
    a = A.fold(NAME, A.current_labels(store(env), A.record_id(NAME)))
    assert (a.status, a.owner, a.cites) == ("live", "pous-mvp", ["verity#166"])
    assert A.import_pending(store(env), root, now, done) == []        # recorded once


def test_a_new_name_needs_title_hypothesis_and_type(env):
    with pytest.raises(A.Refused, match="is new"):
        A.claim(store(env), NAME, "pous-mvp", title="t")


def test_a_second_live_claim_is_refused_and_names_the_owner(env):
    A.claim(store(env), NAME, "pous-mvp", **NEW)
    with pytest.raises(A.Refused, match="live under pous-mvp .* write to lanes/pous-mvp/"):
        A.claim(store(env, "store2"), NAME, "pous-ec")
    assert A.claim(store(env), NAME, "pous-mvp")[0] == 0     # the owner again: already yours


def test_killed_needs_a_reason_and_reopening_needs_why(env):
    st = store(env)
    A.claim(st, NAME, "pous-mvp", **NEW)
    with pytest.raises(A.Refused, match="killed needs a reason"):
        A.set_status(st, NAME, "killed", "pous-mvp")
    A.set_status(st, NAME, "killed", "pous-mvp", reason="window attack beats it at 128 KB", cites=["note:pous/x"])
    with pytest.raises(A.Refused, match="is killed: window attack .* --reopen WHY"):
        A.claim(store(env, "store2"), NAME, "pous-ec")
    rc, _ = A.claim(store(env, "store2"), NAME, "pous-ec", reopen="five layers instead of four")
    a = A.fold(NAME, A.current_labels(st, A.record_id(NAME)))
    assert rc == 0 and (a.status, a.owner) == ("live", "pous-ec")
    assert [(h["status"], h["owner"], h["reason"]) for h in a.history] == [
        ("live", "pous-mvp", None), ("killed", "pous-mvp", "window attack beats it at 128 KB"),
        ("live", "pous-ec", "five layers instead of four")]


def test_merged_names_its_evidence(env):
    st = store(env)
    A.claim(st, NAME, "pous-mvp", **NEW)
    with pytest.raises(A.Refused, match="merged needs evidence"):
        A.set_status(st, NAME, "merged", "pous-mvp", reason="landed")
    A.set_status(st, NAME, "merged", "pous-mvp", reason="landed", cites=["verity#166"])


def test_a_parked_approach_can_be_claimed_as_it_is(env):
    A.claim(store(env), NAME, "pous-mvp", **NEW)
    A.set_status(store(env), NAME, "parked", "pous-mvp", reason="waiting for Daniel's timing decision")
    assert A.claim(store(env, "store2"), NAME, "pous-ec")[0] == 0


def test_a_claim_race_is_decided_by_the_fold_and_the_loser_exits_3(env, monkeypatch):
    A.claim(store(env), NAME, "pous-mvp", **NEW)
    real = A.current_labels
    calls = []

    def blind_first(st, art):                               # the loser read the remote before the winner's claim landed
        calls.append(art)
        return [] if len(calls) == 1 else real(st, art)

    monkeypatch.setattr(A, "current_labels", blind_first)
    monkeypatch.setattr(A, "utc_now", lambda: "2099-01-01T00:00:00Z")   # and stamped its claim after the winner's
    rc, line = A.claim(store(env, "store2"), NAME, "pous-ec", **NEW)
    assert rc == 3 and "lost pous/band-d12 to pous-mvp" in line
    a = A.fold(NAME, real(store(env), A.record_id(NAME)))
    assert (a.owner, [x["by"] for x in a.ignored]) == ("pous-mvp", ["pous-ec"])


def test_a_hand_over_needs_a_reason_and_is_not_written_by_the_new_owner(env):
    st = store(env)
    A.claim(st, NAME, "pous-mvp", **NEW)
    with pytest.raises(A.Refused, match="hands it over"):
        A.set_status(st, NAME, "live", "pous-ec", owner="pous-ec", reason="mine now")
    A.set_status(st, NAME, "live", "coordinator", owner="pous-ec", reason="pous-mvp went final")
    assert A.fold(NAME, A.current_labels(st, A.record_id(NAME))).owner == "pous-ec"


def test_superseded_by_and_attacks_must_name_an_existing_approach(env):
    st = store(env)
    A.claim(st, NAME, "pous-mvp", **NEW)
    with pytest.raises(A.Refused, match="superseded needs superseded_by"):
        A.set_status(st, NAME, "superseded", "pous-mvp", reason="replaced")
    with pytest.raises(A.Refused, match="superseded_by 'pous/dense' names no other approach"):
        A.set_status(st, NAME, "superseded", "pous-mvp", reason="replaced", superseded_by=OTHER)
    with pytest.raises(A.Refused, match="an attack needs attacks"):
        A.claim(st, "pous/window-attack", "pous-ec", title="Window attack", hypothesis="A window of layers regenerates", type_="attack")
    A.claim(st, "pous/window-attack", "pous-ec", title="Window attack", hypothesis="A window of layers regenerates", type_="attack",
            attacks=NAME)
    A.claim(st, OTHER, "pous-mvp", title="Dense graph", hypothesis="Dense labelling meets the game", type_="scheme")
    A.set_status(st, NAME, "superseded", "pous-mvp", reason="dense chosen", superseded_by=OTHER)


def test_fields_are_one_line_and_evidence_tokens_have_a_known_form(env):
    st = store(env)
    with pytest.raises(A.Refused, match="one line of at most 240"):
        A.claim(st, NAME, "pous-mvp", **{**NEW, "title": "x" * 241})
    with pytest.raises(A.Refused, match="evidence 'see the doc' is of no known form"):
        A.claim(st, NAME, "pous-mvp", cites=["see the doc"], **NEW)
    good = ["r20260928-014051-da08", "art:1a2b3c4d", "drv:" + "e" * 64, "note:pous/x", "notes-asset:campaigns/pous/a.json",
            "approach:pous/dense", "verity#166", "verity@6746f408", "https://arxiv.org/abs/1"]
    assert all(A.EVIDENCE_RE.match(t) for t in good)
    assert not any(A.EVIDENCE_RE.match(t) for t in ("art:xyz", "#12", "http://x", "store:pous/docs/pouw/new-crypto.md", "r2026-1"))
    with pytest.raises(A.Refused, match="private agent store's path: copy it into the notes"):
        A.claim(st, NAME, "pous-mvp", cites=["store:pous/docs/pouw/new-crypto.md"], **NEW)
    with pytest.raises(A.Refused, match="not one of live"):
        A.set_status(st, NAME, "dead", "pous-mvp", **NEW)


def test_a_change_of_evidence_alone_retracts_a_cite_and_leaves_status_reason_and_history(env):
    st = store(env)
    A.claim(st, NAME, "pous-mvp", cites=["verity#166"], **NEW)
    A.set_status(st, NAME, "killed", "pous-mvp", reason="a window attack regenerates it")
    old = "store:pous/docs/p3-scheme.md"                    # written before `store:` left EVIDENCE_RE
    st.add_label(A.record_id(NAME), vocab.CITES, old, "pous-mvp", ts="2026-09-30T00:00:00Z")
    before = A.fold(NAME, A.current_labels(st, A.record_id(NAME)))
    assert any("private agent store" in p for p in A.problems(before))
    with pytest.raises(A.Refused, match="private agent store"):
        A.set_status(st, NAME, None, "verity-root", cites=["note:x"])
    with pytest.raises(A.Refused, match="does not cite 'note:y'"):
        A.set_status(st, NAME, None, "verity-root", cites=["note:x"], uncites=[old, "note:y"])
    with pytest.raises(A.Refused, match="both cited and uncited"):
        A.set_status(st, NAME, None, "verity-root", cites=[old], uncites=[old])
    with pytest.raises(A.Refused, match="only --cite and --uncite"):
        A.set_status(st, NAME, None, "verity-root", reason="x", cites=["note:x"])
    with pytest.raises(A.Refused, match="nothing to change"):
        A.set_status(st, NAME, None, "verity-root")
    with pytest.raises(A.Refused, match="not recorded"):
        A.set_status(st, OTHER, None, "verity-root", cites=["note:x"])
    after = A.set_status(st, NAME, None, "verity-root", cites=["note:x"], uncites=[old])
    elsewhere = A.fold(NAME, A.current_labels(store(env, "store2"), A.record_id(NAME)))
    for a in (after, elsewhere):
        assert a.cites == ["verity#166", "note:x"] and not A.problems(a)
        assert (a.status, a.reason, a.owner, a.updated, a.history) == \
            (before.status, before.reason, before.owner, before.updated, before.history)
    assert A.main(["approach", NAME, "--by", "verity-root", "--uncite", "note:x"], env / "notes") == 0
    assert A.fold(NAME, A.current_labels(st, A.record_id(NAME))).cites == ["verity#166"]


def test_backfill_records_past_work_at_its_time_and_history_is_append_only(env):
    st = store(env)
    A.set_status(st, OTHER, "live", "pous", at="2026-09-27T14:30Z", title="Dense graph", hypothesis="Dense labelling meets the game",
                 type_="scheme")
    A.claim(st, NAME, "pous-mvp", **NEW)
    A.set_status(st, OTHER, "superseded", "pous", at="2026-09-28T03:40Z", reason="band d = 12 is secure-first", superseded_by=NAME)
    a = A.fold(OTHER, A.current_labels(st, A.record_id(OTHER)))
    assert (a.claimed, a.updated, a.status, a.owner) == ("2026-09-27T14:30:00Z", "2026-09-28T03:40:00Z", "superseded", "pous")
    with pytest.raises(A.Refused, match="history is append-only"):
        A.set_status(st, OTHER, "killed", "pous", at="2026-09-28T01:00Z", reason="x")
    with pytest.raises(A.Refused, match="in the future"):
        A.set_status(st, OTHER, "killed", "pous", at="2099-01-01T00:00Z", reason="x")


def test_render_shows_what_the_brief_allows(env):
    st, root = store(env), env / "notes"
    A.claim(st, NAME, "pous-mvp", cites=["verity#166"], **NEW)
    A.claim(st, OTHER, "pous-mvp", title="Dense graph", hypothesis="SECRET-HYPOTHESIS", type_="scheme")
    A.set_status(st, OTHER, "killed", "pous-mvp", reason="SECRET-REASON")
    assert [p.name for p in A.render(st, root)] == ["APPROACHES.md"]
    text = (root / "campaigns/pous/APPROACHES.md").read_text()
    assert "## live" in text and "**pous/band-d12** (scheme) · owner `pous-mvp`" in text and "`verity#166`" in text
    assert "## killed" in text and "SECRET" not in text and "registry: titles" in text
    assert A.render(st, root) == []                          # unchanged content: nothing to commit
    (root / "campaigns/pous/BRIEF.md").write_text("---\nregistry: full\n---\n")
    A.render(st, root)
    text = (root / "campaigns/pous/APPROACHES.md").read_text()
    assert "hypothesis: SECRET-HYPOTHESIS" in text and "why: SECRET-REASON" in text


def test_check_reports_errors_and_orphans(env):
    st, root = store(env), env / "notes"
    A.claim(st, NAME, "pous-mvp", **NEW)
    assert A.check(st, root) == ([], [])
    notes.checkpoint(root, "pous-mvp", "final", "done", commit="abc1234")
    errors, warnings = A.check(st, root)
    assert errors == [] and warnings == ["pous/band-d12: ORPHAN, live but owner lane pous-mvp is final: park it or hand it over"]
    A.claim(st, "other/x", "nobody-here", title="x", hypothesis="y", type_="route")
    errors, warnings = A.check(st, root)
    assert errors == ["campaigns/other/BRIEF.md is missing: a campaign with approaches needs its brief"]
    assert "other/x: ORPHAN, live but owner lane nobody-here has no folder lanes/nobody-here/: park it or hand it over" in warnings
    st.add_label(A.record_id(NAME), vocab.APPROACH_STATUS, "killed", "pous-mvp")    # a plain `research data label`, no reason
    assert "pous/band-d12: killed needs a reason" in A.check(st, root)[0]


def test_the_notes_cli_claims_refuses_and_lists(env, capsys):
    root = env / "notes"
    args = ["--root", str(root), "--store", str(env / "store")]
    assert notes.main(["claim", NAME, "--lane", "pous-mvp", "--title", NEW["title"], "--hypothesis", NEW["hypothesis"],
                       "--type", "scheme", "--cite", "verity#166", *args]) == 0
    assert "claimed pous/band-d12" in capsys.readouterr().out
    assert notes.main(["claim", NAME, "--lane", "pous-ec", *args]) == 2
    assert "refused: pous/band-d12 is live under pous-mvp" in capsys.readouterr().err
    assert notes.main(["approach", NAME, "killed", "--by", "pous-mvp", "--reason", "target held", *args]) == 0
    assert notes.main(["approaches", "--campaign", "pous", "--grep", "band", *args]) == 0
    out = capsys.readouterr().out
    assert "pous/band-d12" in out and "killed" in out and "-- target held" in out
    assert notes.main(["approaches", "check", *args]) == 0
    assert notes.main(["approaches", "render", *args]) == 0 and (root / "campaigns/pous/APPROACHES.md").is_file()


def test_the_steward_rule_renders_on_its_interval(env):
    st, root = store(env), env / "notes"
    A.claim(st, NAME, "pous-mvp", **NEW)
    (root / "steward.toml").write_text(f'[approaches]\nstore = "{env / "steward-store"}"\nevery_min = 10\n')
    state, now = {}, dt.datetime(2026, 9, 28, 6, 0, tzinfo=dt.timezone.utc)
    assert A.steward_lines(root, now, state) == ["06:00Z RENDERED campaigns/pous/APPROACHES.md"]
    assert A.steward_lines(root, now + dt.timedelta(minutes=5), state) == []        # inside the interval
    assert A.steward_lines(root, now + dt.timedelta(minutes=11), state) == []       # nothing changed
    (root / "steward.toml").write_text("")
    assert A.steward_lines(root, now + dt.timedelta(minutes=30), {}) == []           # no [approaches] table: the rule is off
