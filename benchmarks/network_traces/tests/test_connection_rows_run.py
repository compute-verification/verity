"""A relay run's connection rows (`connection_rows_run`) without the wall clock: the schedule's lengths rows and
content rows as flags that `rehearse` passes to its `serve`, and a run without them what it was; the hand as the
publisher queue's items, a slot stride each; a run of the network certifier on a `VirtualClock` whose publisher,
inline and in `serve`'s process, lays, commits and signs each handed execution phase's tables, which the offline
check holds to the records' own reading of the same cells and refuses when a registration, a salt, a kept line or a
record isn't the run's; an execution phase that isn't handed has no line; and the cost's load and its measurement
at a small schedule."""

from __future__ import annotations

import dataclasses
import json

import pytest

from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier.warden import Released, WardenParams

from test_warden_serve import _run, ws

crr = ws.crr
#: a schedule with lengths rows and three classes, two of one size: 16 buckets an execution phase, a slot stride of 2
PLAIN = WardenParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=6, window=16, ingress_cells=4,
                     ingress_buckets=3, egress_cells=9, every=2)
CLASSES = ((4, 4, 2), (128, 8, 3), (128, 8, 1))
ROWS = dataclasses.replace(PLAIN, lengths=1, content=CLASSES)
SERVE = ["serve", "--upstream", "h:1", "--listen", "h:0", "--out", "o", "--key", "k", "--session", "00", "--windows", "1"]
REHEARSE = ["rehearse", "--fake", "--out", "o", "--ids"]
SCHEDULE = [x for name in ("cell_bytes", "slots", "delay", "lanes", "queue", "window", "ingress_cells", "ingress_buckets",
                           "egress_cells", "every") for x in (f"--{name.replace('_', '-')}", str(getattr(PLAIN, name)))]
#: each request's prompt ids and the ids its server answers: six connections on two lanes, so over two execution phases
PROMPTS = ((1,), (2, 3, 4, 5, 6), (7,), (8, 9), (10,), (11, 12, 13))
ANSWERS = ((11, 22, 33), tuple(range(8)), (), (7,), (5, 6), (9,))


def schedule_of(argv: list[str]) -> WardenParams:
    args = ws.parser().parse_args(argv)
    return crr.schedule(ws._params(args), args)


def test_the_connection_rows_are_schedule_flags_rehearse_passes_to_serve():
    """`--lengths 1` and `--content P,O,ROWS ...` give the schedule its lengths rows and its content rows' classes, in
    order, two of one size among them; `rehearse` passes its schedule's to `serve`, which reads the same schedule;
    without them the schedule is the one it was and no flag is passed; and a class that isn't three integers, or
    content rows without lengths rows, is refused."""
    rows = ["--lengths", "1", "--content", *(",".join(map(str, c)) for c in CLASSES)]
    got = schedule_of(REHEARSE + SCHEDULE + rows)
    assert got == ROWS and got.digest != PLAIN.digest
    assert crr.flags_of(got) == rows and schedule_of(SERVE + SCHEDULE + crr.flags_of(got)) == got
    lengths = schedule_of(REHEARSE + SCHEDULE + ["--lengths", "1"])
    assert (lengths.lengths, lengths.content) == (1, None) and crr.flags_of(lengths) == ["--lengths", "1"]
    for argv in (REHEARSE, SERVE):
        plain = schedule_of(argv + SCHEDULE)
        assert plain == ws._params(ws.parser().parse_args(argv + SCHEDULE)) and crr.flags_of(plain) == []
        assert plain.digest == PLAIN.digest
    for bad in (["--content", "4,4"], ["--content", "4,4,x"], ["--lengths", "2"]):
        with pytest.raises(SystemExit):
            ws.parser().parse_args(REHEARSE + SCHEDULE + bad)
    with pytest.raises(ValueError, match="has lengths rows"):
        schedule_of(REHEARSE + SCHEDULE + ["--content", "4,4,2"])


def test_a_hand_is_queued_a_slot_stride_an_item_and_kept_whole():
    """`hand` gives one item a slot stride that holds a release and then the close with the strides' count, and
    `Keeper`, called with them in the queue's order, keeps the hand it was given."""
    a, b, c = (Released(n, 20 + n, 2, False, bytes([n + 1]) * (20 + n), bytes(8)) for n in range(3))
    strides = ((), (a, b), (), (c,), (), (), (), ())
    items = list(crr.hand(3, strides))
    assert items == [(crr.HAND, (3, 1), (a, b)), (crr.HAND, (3, 3), (c,)), (crr.HAND, (3, None), 8)]
    assert list(crr.hand(4, ((),) * 8)) == [(crr.HAND, (4, None), 8)]
    kept = []
    keeper = crr.Keeper(None, None, ROWS, ws.LINK)
    keeper.keep = lambda window, got: kept.append((window, got))
    for _, at, data in [*items[:2], *crr.hand(4, ((),) * 8), items[2]]:
        keeper(at, data)
    assert kept == [(4, ((),) * 8), (3, strides)] and keeper.open == {}


def served(out, key, *, windows: int = 4, process: bool = False, params: WardenParams = ROWS, requests=None):
    """A run of `params` with `PROMPTS`' requests (or `requests`), each answered with its `ANSWERS`."""
    B = params.cell_bytes
    out.mkdir()
    requests = requests or [ws.id_request(model="m", user=f"user-{j}", prompt=list(p), max_tokens=8, frame_bytes=B,
                                          offset=0) for j, p in enumerate(PROMPTS)]
    responses = [ws.id_response(list(ids), params.payload) for ids in ANSWERS]
    return _run(out, key, requests, responses, windows=windows, params=params, id_payload=params.payload,
                process=process), requests


@pytest.mark.parametrize("process", [False, True])
def test_a_run_keeps_each_handed_execution_phases_rows_and_the_check_holds_them_to_the_records(tmp_path, process):
    """Six connections over two execution phases of a run of four, published inline and by `serve`'s publisher
    process: every execution phase but the last has one line of `rows.jsonl`, its registration, signed once, and one
    blob of salts; the tables are the schedule's, whatever each execution phase held; each release's rows are what it
    sent and was answered; the check accepts the run and counts three of three registered; and no tick is late."""
    out = tmp_path / "run"
    device, requests = served(out, tmp_path / "key", process=process)
    assert len(device.log) == len(PROMPTS) and all(c["failed"] is None and not c["cut"] for c in device.log)
    assert max(device.lateness) == 0
    kept = crr.read(out, ROWS)
    assert [doc["registration"]["window"] for doc, _ in kept] == [0, 1, 2]
    listed = CR.tables_of(ROWS)
    assert [name for name, _, _ in listed] == [CR.LENGTHS, CR.FAILS, *(n for i in range(3) for n in CT.names(i))]
    for doc, salts in kept:
        assert [(t["name"], t["rows"], 2 * t["words"]) for t in doc["registration"]["tables"]] == list(listed)
        assert doc["cost"]["tables"] == len(listed) and doc["cost"]["rows"] == sum(rows for _, rows, _ in listed)
        assert doc["cost"]["row_bytes"] == sum(rows * size for _, rows, size in listed)
        assert {name: len(s) for name, s in salts.items()} == {name: rows for name, rows, _ in listed}
    records = ws.read_records(out, ROWS)
    held = CR.of_records(ROWS, records["requests"], records["responses"])
    releases = [r for w in sorted(held) for stride in held[w] for r in stride]
    order = sorted(range(len(PROMPTS)), key=lambda j: device.log[j]["release"])
    assert [c["ordinal"] for c in device.log] == list(range(1, len(PROMPTS) + 1))
    assert [(r.request, CT.ids_of(r.ids), r.failed) for r in releases] == \
        [(requests[j], ANSWERS[j], False) for j in order]
    assert len({w for w in held if any(held[w])}) == 2
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    s = result["summary"]["connection_rows"]
    assert (s["registered"], s["of"], s["tables"], s["rows"]) == (3, 3, [len(listed)], [sum(r for _, r, _ in listed)])
    assert s["build_ms"]["n"] == s["commit_ms"]["n"] == s["sign_ms"]["n"] == 3
    assert result["summary"]["cell_rows"]["registered"] == 8 and result["summary"]["certified_windows"] == 12


def test_the_check_refuses_rows_that_arent_the_runs(tmp_path):
    """The check refuses a registration with another root, a row's salt changed, a line lost, a line of an execution
    phase that has none, and a record whose cells aren't the rows' (an id changed in the egress record), each by the
    execution phase it names."""
    out = tmp_path / "run"
    served(out, tmp_path / "key")
    assert ws.check(out)["accepted"]

    def refused(edit) -> list[str]:
        saved = {p: p.read_bytes() for p in out.iterdir()}
        edit()
        try:
            return ws.check(out)["problems"]
        finally:
            for p, data in saved.items():
                p.write_bytes(data)

    def edit_rows(change) -> None:
        lines = [json.loads(x) for x in (out / crr.ROWS).read_text().splitlines()]
        salts = ws.ar.read_blobs(out / crr.SALTS)
        change(lines, salts)
        (out / crr.ROWS).write_text("".join(json.dumps(x) + "\n" for x in lines))
        (out / crr.SALTS).unlink()
        for blob in salts:
            ws.ar.write_blobs(out / crr.SALTS, blob)

    def other_root(lines, salts) -> None:
        lines[1]["registration"]["tables"][3]["root"] = "00" * 64

    def other_salt(lines, salts) -> None:
        salts[1] = salts[1][:-1] + bytes([salts[1][-1] ^ 1])

    def lost(lines, salts) -> None:
        del lines[1], salts[1]

    def moved(lines, salts) -> None:
        lines[2]["registration"]["window"] = 3

    assert "execution phase 1's connection rows aren't registered" in refused(lambda: edit_rows(other_root))
    assert "execution phase 1's connection rows aren't its records' releases" in refused(lambda: edit_rows(other_salt))
    problems = refused(lambda: edit_rows(lost))
    assert "execution phase 1's connection rows aren't registered" in problems
    assert not any("execution phase 0" in p or "execution phase 2" in p for p in problems)
    problems = refused(lambda: edit_rows(moved))
    assert "execution phase 2's connection rows aren't registered" in problems
    assert any(p.startswith("execution phase 3 has connection rows") for p in problems)
    assert any("connection rows: " in p and "don't read" in p for p in refused(
        lambda: edit_rows(lambda lines, salts: salts.pop())))

    def other_id() -> None:
        records = ws.read_records(out, ROWS)["responses"]
        record = records[0]
        row = next(b for b, r in enumerate(record.grid) if any(f is not None and f[0] == ws.DATA for f in r))
        frames = list(record.grid[row])
        k = next(i for i, f in enumerate(frames) if f is not None and f[0] == ws.DATA)
        read = ws.IT.parse_cell(frames[k])
        frames[k] = ws.IT.data_cell(read.id + 1, read.stream, ROWS.cell_bytes)
        grid = record.grid[:row] + (tuple(frames),) + record.grid[row + 1:]
        (out / "responses.records").unlink()
        for r in (ws.replace(record, grid=grid), *records[1:]):
            ws.ar.write_blobs(out / "responses.records", ws.ar.pack_record(r))

    assert "execution phase 0's connection rows aren't its records' releases" in refused(other_id)


@pytest.mark.parametrize("process", [False, True])
def test_an_execution_phase_that_isnt_handed_has_no_rows(tmp_path, process):
    """Six connections at once on two lanes and a queue of five: the sixth is refused, so execution phase 0's records
    are withheld and it is not handed; its releases get no line, no head and no signature, inline and in the publisher
    process. Execution phases 1 and 2 have theirs, and the check, which refuses the run on execution phase 0's
    certificates, counts two of two registered and asks none of execution phase 0."""
    out = tmp_path / "run"
    device, _ = served(out, tmp_path / "key", process=process, params=dataclasses.replace(ROWS, queue=5))
    assert (device.refused, device.refused_windows, len(device.log)) == (1, {0}, 5)
    assert [doc["registration"]["window"] for doc, _ in crr.read(out, dataclasses.replace(ROWS, queue=5))] == [1, 2]
    result = ws.check(out)
    s = result["summary"]["connection_rows"]
    assert (s["registered"], s["of"]) == (2, 2)
    assert not any("connection rows" in p for p in result["problems"]) and not result["accepted"]


def test_a_run_without_lengths_rows_writes_and_checks_what_it_did(tmp_path):
    """A schedule without lengths rows: the run writes no `rows.jsonl`, takes no leaf for one, and its check's summary
    has no entry for them; rows found beside it are no part of its check."""
    out = tmp_path / "run"
    served(out, tmp_path / "key", windows=2, params=PLAIN)
    assert not (out / crr.ROWS).exists() and not (out / crr.SALTS).exists()
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    assert "connection_rows" not in result["summary"]
    assert len((out / "certified.jsonl").read_text().splitlines()) == 6
    assert crr.check(out, {}, PLAIN, [], [], [], ws.LINK) == (None, [])


def test_the_costs_load_is_the_classes_rows_or_what_the_grids_carry_and_its_close_is_measured(tmp_path):
    """`loaded`: with `rows`, every row of every class, each release its class's longest request and most ids, filed
    in its class by the network certifier's rule; with `grid`, no more than an execution phase's grids carry; a
    schedule without content rows is refused. `cost` at a small schedule hands that load to the publisher process
    through the device's own hand and reports each step once an execution phase."""
    p = dataclasses.replace(PLAIN, window=32, every=2, lanes=2, slots=2, ingress_cells=4, egress_cells=9, lengths=1,
                            content=((4, 4, 2), (128, 8, 3)))
    rows = crr.loaded(p, "rows")
    filed = [CT.file(p, Released(0, size, n, False, bytes(size), bytes(4 * n))) for size, n in rows]
    assert sorted(filed) == [0, 0, 1, 1, 1] and len({r for r, f in zip(rows, filed) if f == 1}) == 1
    assert max(n for _, n in rows) == 8 and max(size for size, _ in rows) == min(CT.request_room(512, 128), 4 * p.payload)
    big = dataclasses.replace(p, content=((128, 8, 30),))
    cells = big.slots * big.lanes * big.window
    grid = crr.loaded(big, "grid")
    assert len(grid) == cells // 9 < len(crr.loaded(big, "rows")) == 30
    assert sum(n + 1 for _, n in grid) <= cells and sum(-(-size // big.payload) for size, _ in grid) <= cells
    with pytest.raises(ValueError, match="content rows"):
        crr.loaded(dataclasses.replace(p, content=None), "rows")
    got = crr.cost(p, "rows", phases=2, height=4)
    listed = CR.tables_of(p)
    assert (got["releases"], got["tables"], got["rows"]) == (5, len(listed), sum(r for _, r, _ in listed))
    assert got["handed_bytes"] == sum(size + 4 * n for size, n in rows)
    assert {got[k]["n"] for k in ("hand_ms", "pickle_ms", "pickle_longest_item_ms", "build_ms", "commit_ms",
                                  "sign_ms")} == {2}
    out = tmp_path / "cost.json"
    assert crr.main(["cost", "--out", str(out), "--classes", "4,4,2", "128,8,3", "--load", "rows", "--phases", "1",
                     "--height", "4", "--slots", "2", "--lanes", "2", "--window", "32", "--every", "2", "--delay", "5",
                     "--ingress-cells", "4", "--egress-cells", "9"]) == 0
    assert json.loads(out.read_text())["releases"] == 5
