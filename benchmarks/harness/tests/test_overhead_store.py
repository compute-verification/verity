"""`verity bench overhead`'s store side: versioned points and the source check on a real store with a directory remote."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from overhead_standins import (
    CUBIN,
    CUBIN_ART,
    LEASE,
    REPO,
    Node,
    Store,
    brute_force,
    node_points,
    node_run,
    stored_points,
    timed,
)
from overhead_standins import scheme as scheme
from overhead_standins import stamped as stamped
from verity_bench import overhead as O
from verity_bench import overhead_records as R
from verity_bench.cli import main


def test_points_are_stored_versioned_and_the_newest_supersedes_the_last(tmp_path):
    s = Store(tmp_path)
    first = stored_points(tmp_path, s, "r20261104-020000-0001")
    assert R.current_points(s.store, "h200") == first
    meta = s.store.get_manifest(first).meta
    assert meta["version"] == 1 and meta["supersedes"] is None and (meta["c"], meta["l"]) == (brute_force().c,
                                                                                             brute_force().lat)
    second = stored_points(tmp_path, s, "r20261104-020000-0002")
    assert second != first
    fresh = s.fresh()
    from research.store import labels_sync

    labels_sync.sync(fresh, push=False)
    assert R.current_points(fresh, "h200") == second and R.current_points(fresh, "rtx-pro-6000") is None
    assert [lab.value for lab in fresh.labels(first, key="superseded_by")] == [second]
    assert {"version": 2, "supersedes": first, "run": "r20261104-020000-0002"}.items() <= fresh.get_manifest(
        second).meta.items()

    node_points(tmp_path, Node(tmp_path), "r20261104-020000-0003")
    s.ran("r20261104-020000-0003", tmp_path / "r20261104-020000-0003", sub="points", validation="failed", rc=1)
    with pytest.raises(ValueError, match="isn't a passing run"):
        R.store_points(s.store, "r20261104-020000-0003")


def test_only_a_run_of_the_door_s_own_job_can_be_stored_or_recorded(tmp_path):
    """The source check's first step: the run's recorded command is the node job the door launches, with the door's options
    and no variable but the door's, so a sweep or timed run under a wrapper script, a patched interpreter, an option the
    node job sets itself (another nsys) or another environment never becomes a record; nor do points swept under another
    rule."""

    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    node_points(tmp_path, Node(tmp_path), "r20261104-020000-0002")
    wrapped = ["bash", "-c", 'bash "$RESEARCH_RUN_DIR/inputs/points.sh"']
    for i, (kw, why) in enumerate((
            ({"argv": wrapped}, "isn't a run of the door"),
            ({"argv": [*LEASE, "python", "-c", "import verity_bench.overhead_node"]}, "isn't a run of the door"),
            ({"sub": "run"}, "isn't a run of the door"),
            ({"argv": [*LEASE, "bash", R.NODE_JOB, "points", "--card", "h200", "--uuid=GPU-0"]}, "isn't a run of the door"),
            ({"sub": "points", "env": {"VERITY_BENCH_SETUP": "patch.sh"}}, "set VERITY_BENCH_SETUP with --env"))):
        run = f"r20261104-020000-001{i}"
        s.ran(run, tmp_path / "r20261104-020000-0002", **kw)
        with pytest.raises(ValueError, match=why):
            R.store_points(s.store, run)
    # the lease `research run --gpus 1` puts before it, where gpu-lease hands out every GPU, holds the same job
    gpus = ["gpu-lease", "1", "--wait", "--mem-gb", "0", "--", "bash", R.NODE_JOB, "points", "--card=h200"]
    s.ran("r20261104-020000-0020", tmp_path / "r20261104-020000-0002", argv=gpus,
          env={"VERITY_BENCH_VENV": "/workspace/jobs/venv312"})
    assert R.store_points(s.store, "r20261104-020000-0020").startswith("art:")

    points = tmp_path / "r20261104-020000-0002" / "out" / "points.json"
    doc = json.loads(points.read_text())
    points.write_text(json.dumps(doc | {"rule": O.RULE | {"m_max": 64}}))
    s.ran("r20261104-020000-0021", tmp_path / "r20261104-020000-0002", sub="points")
    with pytest.raises(ValueError, match="'m_max': 64"):
        R.store_points(s.store, "r20261104-020000-0021")

    sent = tmp_path / "sent-points.json"
    sent.write_bytes(Path(s.store.fetch(art)).read_bytes())
    rc, run_dir = node_run(tmp_path, Node(tmp_path), sent, "stock", art, run="r20261104-030000-0001")
    assert rc == 0
    for i, kw in enumerate(({"sub": "points"}, {"argv": [*LEASE, "bash", R.NODE_JOB, "run", "--nsys", "/tmp/nsys"]})):
        run = f"r20261104-030000-001{i}"
        s.ran(run, run_dir, **kw)
        with pytest.raises(ValueError, match="isn't a run of the door"):
            R.record_run(s.store, run)
        assert not R.of_record(s.store, run)[0]


def test_the_table_shows_only_labelled_runs_of_record_and_refuses_the_rest_with_why(tmp_path, scheme):
    """The source check: a γ stands only from a labelled run of the door, timed at the points of record. A run that
    doesn't stand as a whole can't be labelled; a point that fails the door's rules stays in the table, refused; points
    superseded after the run refuse its γ; and with no labelled run the table holds no γ."""

    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    run = timed(tmp_path, s, art, "pearl-plus", "r20261104-030000-0002")
    stock = timed(tmp_path, s, art, "stock", "r20261104-030000-0003")
    noisy = timed(tmp_path, s, art, "pearl-plus", "r20261104-030000-0004", Node(tmp_path, reps=12))

    assert R.gamma_rows(s.store) == []
    assert R.record_run(s.store, run) == [] and R.record_run(s.store, stock) == []
    rows = {(r["scheme"], r["point"]): r for r in R.gamma_rows(s.store)}
    assert set(rows) == {("pearl-plus", "C"), ("pearl-plus", "L"), ("stock", "C"), ("stock", "L")}
    assert all(r["of_record"] and r["why"] is None for r in rows.values())
    got = rows["pearl-plus", "L"]
    assert got["gamma"]["gamma"] == pytest.approx(0.55, rel=1e-4) and got["run"] == run and got["points_art"] == art

    # a fresh machine renders the table from the store's remote alone, each period beside the window
    table = R.build(s.fresh(), root=REPO)
    assert all(r["of_record"] for r in table["rows"]) and table["census"]["gpus"] == 100_000
    by = {(r["scheme"], r["point"]): r for r in table["rows"]}
    assert "periods" not in by["stock", "C"] and by["pearl-plus", "C"]["periods"]["share"] == pytest.approx(0.55 / 1.55)
    text = R.render(table)
    assert "| pearl-plus | h200 | L |" in text and "| pearl-plus | h200 | C | 35.48% | year |" in text
    assert "Refused" not in text

    # a labelled run whose points fail the door's rules: shown refused with why, never as a figure
    assert all("fewer than 20" in w for w in R.record_run(s.store, noisy))
    noisy_c = {(r["scheme"], r["point"]): r for r in R.gamma_rows(s.store)}["pearl-plus", "C"]
    assert not noisy_c["of_record"] and "fewer than 20" in noisy_c["why"]
    text = R.render(R.build(s.fresh("second"), root=REPO))
    assert "Refused (not of record):" in text and "fewer than 20" in text and "| pearl-plus | h200 | L |" not in text

    # a run that doesn't stand as a whole can't be labelled: from a dirty tree, or timed at points of no stored art
    sent = tmp_path / "r20261104-030000-0002-points.json"
    dirty = "r20261104-030000-0005"
    s.ran(dirty, node_run(tmp_path, Node(tmp_path), sent, "pearl-plus", art, run=dirty)[1], dirty=True)
    with pytest.raises(ValueError, match="clean tree"):
        R.record_run(s.store, dirty)
    loose = "r20261104-030000-0006"
    s.ran(loose, node_run(tmp_path, Node(tmp_path), sent, "pearl-plus", run=loose)[1])
    with pytest.raises(ValueError, match="no stored art"):
        R.record_run(s.store, loose)

    # nor can one whose cubin art isn't the binary its SASS gate passed, or isn't in the store
    for i, (cubin, cubin_art) in enumerate(((b"another cubin", s.cubin), (CUBIN, CUBIN_ART))):
        other = timed(tmp_path, s, art, "pearl-plus", f"r20261104-030000-001{i}", cubin=cubin, cubin_art=cubin_art)
        with pytest.raises(ValueError, match="SASS gate passed" if i == 0 else "isn't in the store"):
            R.record_run(s.store, other)

    # new points supersede the ones the runs were timed at: none of their γ stands
    stored_points(tmp_path, s, "r20261104-020000-0007")
    third = s.fresh("third")
    later = R.build(third, root=REPO)
    assert not any(r["of_record"] for r in later["rows"])
    assert all("superseded" in r["why"] for r in later["rows"])


def test_a_row_carries_the_note_of_its_own_run_and_the_same_numbers_with_it_or_without(tmp_path, scheme, monkeypatch, capsys):
    """Made-up records of one stand-in arm under three entries. The fixture's pearl-plus entry states no `commits_y`, so its
    C and L rows carry
    `WITHOUT_Y` in both rendered tables and the JSON. A made-up scheme whose entry commits y carries no note, and every
    number of its rows is pearl-plus's. A run recorded before the field (its `arm` record without `commits_y`) carries the
    note whatever today's entry says: the note is the run's, not the entry's. stock, with no arm, carries none. The
    table's coverage line names the layer's linears and what the model's entry leaves out."""

    with_y = dataclasses.replace(scheme, id="pearl-plus-y", commits_y=True)
    before = dataclasses.replace(with_y, id="pearl-plus-before")
    for sch in (with_y, before):
        monkeypatch.setitem(O.SCHEMES, sch.id, sch)
    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    runs = {"pearl-plus": timed(tmp_path, s, art, "pearl-plus", "r20261104-030000-0002"),
            with_y.id: timed(tmp_path, s, art, with_y.id, "r20261104-030000-0003"),
            before.id: timed(tmp_path, s, art, before.id, "r20261104-030000-0004",
                             edit=lambda doc: doc["arm"].pop("commits_y")),
            "stock": timed(tmp_path, s, art, "stock", "r20261104-030000-0005")}
    assert all(R.record_run(s.store, run) == [] for run in runs.values())

    table = R.build(s.fresh(), root=REPO)
    by = {(r["scheme"], r["point"]): r for r in table["rows"]}
    assert set(by) == {(name, point) for name in runs for point in O.POINTS} and all(r["of_record"] for r in by.values())
    ours = {"scheme", "run", "commits_y", "note"}
    for point in O.POINTS:
        without, committed, old, stock = (by[name, point] for name in runs)
        assert (without["commits_y"], without["note"]) == (False, O.WITHOUT_Y) == (old["commits_y"], old["note"])
        assert (committed["commits_y"], committed["note"]) == (True, None)
        assert (stock["commits_y"], stock["note"]) == (None, None)
        numbers = {k: v for k, v in without.items() if k not in ours}
        assert numbers["gamma"]["gamma"] == pytest.approx(0.55, rel=1e-4) and numbers["periods"]["share"] > 0
        assert all({k: v for k, v in r.items() if k not in ours} == numbers for r in (committed, old))

    lines = R.render(table).splitlines()
    rows = {name: [ln for ln in lines if ln.startswith(f"| {name} | ")] for name in runs}
    # a row of the γ table at each point and, for a scheme with an arm, a row a period (and the window) at each point
    assert [len(rows[name]) for name in runs] == [2 * (2 + len(O.PERIODS_H))] * 3 + [2]
    for name in ("pearl-plus", before.id):
        assert all(ln.endswith(f" | {O.WITHOUT_Y} |") for ln in rows[name])
    for name in (with_y.id, "stock"):
        assert all(ln.endswith(" |  |") and O.WITHOUT_Y not in ln for ln in rows[name])

    def bare(name: str, note: str) -> list[str]:
        """A scheme's rows without their note, under pearl-plus's name and no run id."""

        return [ln.removesuffix(f" {note} |").replace(f"| {name} |", "| pearl-plus |").replace(runs[name], "RUN")
                for ln in rows[name]]

    assert bare("pearl-plus", O.WITHOUT_Y) == bare(with_y.id, "") == bare(before.id, O.WITHOUT_Y)

    assert O.WITHOUT_Y == "without y's commitment (y tree not timed)"
    assert table["covers"] == O.covers() == ("Each γ is over the 4 linears of one llama3.1-70b layer (qkv, o, gate_up, "
                                             "down) and leaves the LM head out.")
    assert lines.count(table["covers"]) == 1
    monkeypatch.setenv("COLUMNS", "400")
    with pytest.raises(SystemExit):
        main(["overhead", "table", "--help"])
    said = capsys.readouterr().out
    assert table["covers"] in said and O.WITHOUT_Y in said
    monkeypatch.setitem(O.LEFT_OUT, O.KERNEL_MODEL, ())
    assert O.covers() == "Each γ is over the 4 linears of one llama3.1-70b layer (qkv, o, gate_up, down)."
