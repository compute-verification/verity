"""A relay run's connection rows (`connection_rows_run`) without the wall clock: the schedule's lengths rows and
content rows as flags that `rehearse` passes to its `serve`, and a run without them what it was; the hand as the
publisher queue's items, a slot stride each; a run of the network certifier on a `VirtualClock` whose publisher,
inline and in `serve`'s process, lays, commits and signs each handed execution phase's tables, which the offline
check holds to the records' own reading of the same cells and refuses when a registration, a salt, a kept line or a
record isn't the run's; an execution phase that isn't handed has no line; under the lengths rows' form 2 every
execution phase has one line, its registration or the network certifier's word that it has no head, the run's last
included, and the check holds each to the one that is due, also past an execution phase that isn't complete; and
the cost's load and its measurement at a small schedule."""

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
#: the same schedule at the lengths rows' form 2, each class its two sizes
ENDS = dataclasses.replace(ROWS, lengths=2, content=tuple(c[:2] for c in CLASSES))
T = PLAIN.window
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
    order, two of one size among them, and `--lengths 2` with `--content P,O ...` the lengths rows' form 2, another
    schedule; `rehearse` passes its schedule's to `serve`, which reads the same schedule; without them the schedule
    is the one it was and no flag is passed; and a class that isn't two or three integers, a form that is none, a
    class of the other form's numbers, or content rows without lengths rows, is refused. Under form 2 a rehearsal
    runs long enough to admit its last request (`closing_windows`)."""
    rows = ["--lengths", "1", "--content", *(",".join(map(str, c)) for c in CLASSES)]
    pairs = ["--content", *(",".join(map(str, c[:2])) for c in CLASSES)]
    got = schedule_of(REHEARSE + SCHEDULE + rows)
    assert got == ROWS and got.digest != PLAIN.digest
    assert crr.flags_of(got) == rows and schedule_of(SERVE + SCHEDULE + crr.flags_of(got)) == got
    lengths = schedule_of(REHEARSE + SCHEDULE + ["--lengths", "1"])
    assert (lengths.lengths, lengths.content) == (1, None) and crr.flags_of(lengths) == ["--lengths", "1"]
    ends = schedule_of(REHEARSE + SCHEDULE + ["--lengths", "2", *pairs])
    assert ends == ENDS and ends.digest not in (ROWS.digest, PLAIN.digest)
    assert crr.flags_of(ends) == ["--lengths", "2", *pairs]
    assert schedule_of(SERVE + SCHEDULE + crr.flags_of(ends)) == ends
    for argv in (REHEARSE, SERVE):
        plain = schedule_of(argv + SCHEDULE)
        assert plain == ws._params(ws.parser().parse_args(argv + SCHEDULE)) and crr.flags_of(plain) == []
        assert plain.digest == PLAIN.digest
    for bad in (["--content", "4"], ["--content", "4,4,2,1"], ["--content", "4,4,x"], ["--lengths", "3"],
                ["--lengths", "0"]):
        with pytest.raises(SystemExit):
            ws.parser().parse_args(REHEARSE + SCHEDULE + bad)
    with pytest.raises(ValueError, match="drop its third number"):
        schedule_of(REHEARSE + SCHEDULE + ["--lengths", "2", *rows[2:]])
    with pytest.raises(ValueError, match="give the rows of each of its tables"):
        schedule_of(REHEARSE + SCHEDULE + ["--lengths", "1", *pairs])
    with pytest.raises(ValueError, match="has lengths rows"):
        schedule_of(REHEARSE + SCHEDULE + ["--content", "4,4,2"])
    # a request sent in bucket b is admitted by bucket b + 2, which is at most the run's last bucket less the span
    assert ENDS.span == 9 and T == 16
    assert [ws.closing_windows(ENDS, at) for at in (0.05, 0.45, 0.55, 2.0, 2.1)] == [1, 1, 2, 2, 3]


def test_a_hand_is_queued_a_slot_stride_an_item_and_kept_whole():
    """`hand` gives one item a slot stride that holds a release and then the close with the strides' count, and
    `Keeper`, called with them in the queue's order, keeps the hand it was given. An execution phase that ends
    without releases is its close alone, with no count, and is kept as none: not as one that released nothing."""
    a, b, c = (Released(n, 20 + n, 2, False, bytes([n + 1]) * (20 + n), bytes(8)) for n in range(3))
    strides = ((), (a, b), (), (c,), (), (), (), ())
    items = list(crr.hand(3, strides))
    assert items == [(crr.HAND, (3, 1), (a, b)), (crr.HAND, (3, 3), (c,)), (crr.HAND, (3, None), 8)]
    assert list(crr.hand(4, ((),) * 8)) == [(crr.HAND, (4, None), 8)]
    assert list(crr.hand(5, None)) == [(crr.HAND, (5, None), None)]
    kept = []
    keeper = crr.Keeper(None, None, ROWS, ws.LINK)
    keeper.keep = lambda window, got: kept.append((window, got))
    for _, at, data in [*items[:2], *crr.hand(4, ((),) * 8), *crr.hand(5, None), items[2]]:
        keeper(at, data)
    assert kept == [(4, ((),) * 8), (5, None), (3, strides)] and keeper.open == {}


def served(out, key, *, windows: int = 4, process: bool = False, params: WardenParams = ROWS, prompts=PROMPTS,
           answers=ANSWERS, **kw):
    """A run of `params` with `prompts`' requests, each answered with its `answers`; `kw` is `_run`'s."""
    B = params.cell_bytes
    out.mkdir()
    requests = [ws.id_request(model="m", user=f"user-{j}", prompt=list(p), max_tokens=8, frame_bytes=B, offset=0)
                for j, p in enumerate(prompts)]
    responses = [ws.id_response(list(ids), params.payload) for ids in answers]
    return _run(out, key, requests, responses, windows=windows, params=params, id_payload=params.payload,
                process=process, **kw), requests


def refused(out, edit) -> list[str]:
    """The check's problems of the run in `out` once `edit` has changed its files, which are then put back."""
    saved = {p: p.read_bytes() for p in out.iterdir()}
    edit()
    try:
        return ws.check(out)["problems"]
    finally:
        for p in set(out.iterdir()) - set(saved):
            p.unlink()
        for p, data in saved.items():
            p.write_bytes(data)


def edit_rows(out, change) -> None:
    """`rows.jsonl`'s lines and `rows.salts`' blobs as `change(lines, salts)` leaves them."""
    lines = [json.loads(x) for x in (out / crr.ROWS).read_text().splitlines()]
    salts = ws.ar.read_blobs(out / crr.SALTS)
    change(lines, salts)
    (out / crr.ROWS).write_text("".join(json.dumps(x) + "\n" for x in lines))
    (out / crr.SALTS).unlink()
    for blob in salts:
        ws.ar.write_blobs(out / crr.SALTS, blob)


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

    def other_root(lines, salts) -> None:
        lines[1]["registration"]["tables"][3]["root"] = "00" * 64

    def other_salt(lines, salts) -> None:
        salts[1] = salts[1][:-1] + bytes([salts[1][-1] ^ 1])

    def lost(lines, salts) -> None:
        del lines[1], salts[1]

    def moved(lines, salts) -> None:
        lines[2]["registration"]["window"] = 3

    def edited(change) -> list[str]:
        return refused(out, lambda: edit_rows(out, change))

    assert "execution phase 1's connection rows aren't registered" in edited(other_root)
    assert "execution phase 1's connection rows aren't its records' releases" in edited(other_salt)
    problems = edited(lost)
    assert "execution phase 1's connection rows aren't registered" in problems
    assert not any("execution phase 0" in p or "execution phase 2" in p for p in problems)
    problems = edited(moved)
    assert "execution phase 2's connection rows aren't registered" in problems
    assert any(p.startswith("execution phase 3 has connection rows") for p in problems)
    assert any("connection rows: " in p and "don't read" in p for p in edited(lambda lines, salts: salts.pop()))

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

    assert "execution phase 0's connection rows aren't its records' releases" in refused(out, other_id)


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


def phases_of(out, params) -> list[tuple[int, str]]:
    """Each line of the run's `rows.jsonl` in order: its execution phase and what it holds."""
    return [(crr._phase(doc), next(k for k in ("registration", crr.NO_HEAD, "refused") if k in doc))
            for doc, _ in crr.read(out, params)]


def device_key(out):
    """The key and the session of the run's deployment record."""
    dep = json.loads((out / "deployment.json").read_text())
    return ws.cs.read_public(dep["key"]), bytes.fromhex(dep["session"])


@pytest.mark.parametrize("process", [False, True])
def test_under_form_2_every_execution_phase_has_one_line_and_the_runs_last_its_head(tmp_path, process):
    """The six connections at the lengths rows' form 2, inline and in the publisher process: each of the run's four
    execution phases has one line and one blob, in order, and each its registration, the last one's signed at the
    run's last close. Its lengths and fails tables are the schedule's, and each class's three tables, their salts
    and the line's cost are of the rows the execution phase's releases filed in the class: the last two released
    nothing, so each of their content tables has no row and the zero root. The check accepts the run, counts four of
    four registered, and no word that an execution phase has no head, of none due."""
    out = tmp_path / "run"
    device, requests = served(out, tmp_path / "key", process=process, params=ENDS)
    assert len(device.log) == len(PROMPTS) and all(c["failed"] is None and not c["cut"] for c in device.log)
    assert phases_of(out, ENDS) == [(w, "registration") for w in range(4)]
    listed = CR.tables_of(ENDS)
    assert [rows for _, rows, _ in listed[2:]] == [ENDS.phase_releases] * 9
    records = ws.read_records(out, ENDS)
    held = CR.of_records(ENDS, records["requests"], records["responses"])
    assert sorted(held) == [0, 1, 2, 3] and sorted(w for w in held if any(held[w])) == [0, 1]
    assert sum(len(stride) for w in held for stride in held[w]) == len(PROMPTS)
    counts = []
    for w, (doc, salts) in enumerate(crr.read(out, ENDS)):
        filed = [CT.file(ENDS, r) for stride in held[w] for r in stride]
        counts.append([filed.count(i) for i in range(3)])
        want = [(name, rows) for name, rows, _ in listed[:2]]
        want += [(name, counts[w][i]) for i in range(3) for name in CT.names(i)]
        entries = doc["registration"]["tables"]
        assert [(t["name"], t["rows"]) for t in entries] == want
        assert {name: len(s) for name, s in salts.items()} == dict(want)
        assert doc["cost"]["rows"] == sum(rows for _, rows in want)
        assert all((t["root"] == CR.NO_ROWS_ROOT.hex()) == (not t["rows"]) for t in entries)
    assert [sum(c) for c in counts] == [sum(map(len, held[w])) for w in range(4)] and counts[2:] == [[0, 0, 0]] * 2
    assert len({tuple(c) for c in counts}) == 3
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    s = result["summary"]["connection_rows"]
    assert (s["registered"], s["of"], s[crr.NO_HEAD]) == (4, 4, {"signed": 0, "of": 0})
    assert s["sign_ms"]["n"] == 4 and result["summary"]["certified_windows"] == 12


@pytest.mark.parametrize("process", [False, True])
def test_under_form_2_an_execution_phase_that_isnt_handed_has_the_signed_word_and_no_head(tmp_path, process):
    """The sixth of six connections at once is refused, so execution phase 0's records are withheld: under form 2 it
    and the one after it, which releases a connection too, end without releases. Each has one line, the network
    certifier's signed word that it has no head, an empty blob and no registration, inline and in the publisher
    process; the two after them have their heads, the first of them one release's. The check, which refuses the run
    on execution phase 0's certificates, takes both words and both heads, and no release of the first two execution
    phases is in a head."""
    out = tmp_path / "run"
    params = dataclasses.replace(ENDS, queue=5)
    device, _ = served(out, tmp_path / "key", process=process, params=params, prompts=(*PROMPTS, (14,), (15, 16)),
                       answers=(*ANSWERS[:5], (3,), (4, 5)), arrive={6: T + 2, 7: 2 * T + 2})
    assert (device.refused, device.refused_windows, len(device.log)) == (1, {0}, 7)
    assert phases_of(out, params) == [(0, crr.NO_HEAD), (1, crr.NO_HEAD), (2, "registration"), (3, "registration")]
    key, session = device_key(out)
    for w, (doc, salts) in enumerate(crr.read(out, params)[:2]):
        assert sorted(doc) == ["cost", crr.NO_HEAD] and sorted(doc["cost"]) == ["sign_ms"] and salts == {}
        CR.check_no_head(doc[crr.NO_HEAD], key, session, params, ws.LINK, w)
        with pytest.raises(ValueError, match="the word's window"):
            CR.check_no_head(doc[crr.NO_HEAD], key, session, params, ws.LINK, 1 - w)
    assert ws.ar.read_blobs(out / crr.SALTS)[:2] == [b"", b""]
    records = ws.read_records(out, params)
    held = CR.of_records(params, records["requests"], records["responses"])
    assert [sum(map(len, held[w])) for w in range(4)] == [5, 1, 1, 0]
    result = ws.check(out)
    s = result["summary"]["connection_rows"]
    assert (s["registered"], s["of"], s[crr.NO_HEAD]) == (2, 2, {"signed": 2, "of": 2})
    assert not any("connection rows" in p or "no head" in p for p in result["problems"]) and not result["accepted"]


def test_under_form_2_a_run_through_its_last_buckets_passes_the_check_which_counts_the_arrivals_turned_away(tmp_path):
    """Eight requests at form 2 over a run of two execution phases, the last four from the first bucket that admits
    nothing to the run's last: the network certifier serves the first four and turns the last four away, each with
    one cell, in its bucket; it refuses none, so both execution phases are certified and each has its head. The
    run's log counts the four on their own, and the check accepts the run, its summary and its table saying that the
    four clients whose outcome is refused were turned away."""
    out = tmp_path / "run"
    first = 2 * T - ENDS.span
    late = (first, first, first + 3, 2 * T - 1)
    device, _ = served(out, tmp_path / "key", windows=2, params=ENDS,
                       prompts=(*PROMPTS[:4], (14,), (15, 16), (17,), (18,)), answers=ANSWERS[:4],
                       arrive=dict(enumerate((0, 0, T, first - 1, *late))))
    (out / "warden.json").write_text(json.dumps({"connections": device.log, **ws.counts(device)}))
    assert ws.counts(device) == {"ordinals": 4, "refused": 0, "refused_windows": [], "turned_away": 4,
                                 "turned_away_reset": 0, "ticks": 2 * T}
    assert [(c["admitted"], c["failed"], c["cut"]) for c in device.log] == \
        [(t, None, False) for t in (0, 0, T, first - 1)]
    assert phases_of(out, ENDS) == [(0, "registration"), (1, "registration")]
    clients = json.loads((out / "client.json").read_text())["connections"]
    assert [(c["outcome"], c["kinds"], c["arrivals_ms"]) for c in clients[4:]] == \
        [("refused", ws.KIND_CHARS[ws.REFUSED], [t * ENDS.bucket_ms + 1]) for t in late]
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["requests"], s["ended"], s["refused"], s["turned_away"], s["withheld_windows"]) == (8, 4, 4, 4, [])
    assert (s["certified_windows"], s["connection_rows"]["registered"], s["connection_rows"]["of"]) == (6, 2, 2)
    assert "4/8 ended, 4 refused or turned away (4 turned away)," in ws.table(result)


def test_under_form_2_the_check_holds_each_execution_phase_to_the_one_that_is_due(tmp_path):
    """The run above, its lines changed: the check refuses an execution phase with neither a head nor the word, one
    with a head where the word is due, one with the word where a head is due, one with two lines, a word signed for
    another execution phase, a word whose signature is changed, and a word with salts beside it, each by the
    execution phase it names."""
    out = tmp_path / "run"
    params = dataclasses.replace(ENDS, queue=5)
    served(out, tmp_path / "key", params=params)

    def edited(change) -> list[str]:
        return [p for p in refused(out, lambda: edit_rows(out, change)) if "execution phase" in p]

    assert edited(lambda lines, salts: None) == []
    neither = "execution phase {} has neither connection rows nor the network certifier's word that it has none"
    unhanded = ("execution phase {} has connection rows, and it or an execution phase beside it wasn't published "
                "complete")
    unregistered = "execution phase {}'s connection rows aren't registered"

    def lost(lines, salts) -> None:
        del lines[1], salts[1]

    def head_for_word(lines, salts) -> None:
        del lines[0], salts[0]
        lines[-1]["registration"]["window"] = 0

    def word_for_head(lines, salts) -> None:
        del lines[2], salts[2]
        lines[0][crr.NO_HEAD]["window"] = 2

    def twice(lines, salts) -> None:
        lines.append(lines[1])
        salts.append(salts[1])

    def of_another(lines, salts) -> None:
        lines[1][crr.NO_HEAD] = lines[0][crr.NO_HEAD] | {"window": 1}

    def other_signature(lines, salts) -> None:
        signed = lines[1][crr.NO_HEAD]["certificate"]
        signed["signature"] = signed["signature"][:-2] + ("00" if signed["signature"][-2:] != "00" else "01")

    def refusal(lines, salts) -> None:
        lines[1] = {"refused": [ws.LINK, "egress", 1], "cost": lines[1]["cost"]}

    assert edited(lost) == [neither.format(1)]
    assert edited(head_for_word) == [unhanded.format(0), unregistered.format(3)]
    assert edited(word_for_head) == [neither.format(0), unregistered.format(2)]
    assert edited(twice) == ["execution phase 1 has two lines of connection rows"]
    (problem,) = edited(of_another)
    assert problem.startswith("execution phase 1's word that it has no head: ") and "doesn't sign that" in problem
    (problem,) = edited(other_signature)
    assert problem.startswith("execution phase 1's word that it has no head: ") and "doesn't verify" in problem
    assert edited(refusal) == [neither.format(1)]

    def salted(lines, salts) -> None:
        salts[1] = bytes(crr.WR.SALT_BYTES)

    word = f"execution phase 1 has the network certifier's word that it has no head, and {crr.WR.SALT_BYTES} bytes"
    assert any("don't read" in p and word in p for p in edited(salted))


def test_under_form_2_the_check_reads_a_tables_salts_by_the_rows_its_registration_counts(tmp_path):
    """Under form 2 a line's blob of salts is one a row of each table as its registration counts them. The check
    takes the run; with a salt lost from a blob, or a count changed and its salts not, the files no longer read;
    with a class's count one more in its three entries and a salt more for each, the registration isn't the
    schedule's signed head, by the execution phase it names; and with a salt changed the rows aren't its records'
    releases."""
    out = tmp_path / "run"
    served(out, tmp_path / "key", params=ENDS)
    assert ws.check(out)["accepted"]
    size = crr.WR.SALT_BYTES

    def edited(change) -> list[str]:
        return [p for p in refused(out, lambda: edit_rows(out, change)) if "connection rows" in p]

    def lost(lines, salts) -> None:
        salts[0] = salts[0][:-size]

    def recounted(lines, salts) -> None:
        lines[0]["registration"]["tables"][2]["rows"] += 1

    def one_more(lines, salts) -> None:
        for t in lines[0]["registration"]["tables"][2:5]:
            t["rows"] += 1
        salts[0] += b"".join(bytes([k + 1]) * size for k in range(3))

    def other_salt(lines, salts) -> None:
        salts[0] = salts[0][:-1] + bytes([salts[0][-1] ^ 1])

    for change in (lost, recounted):
        assert any("don't read" in p and "execution phase 0's blob" in p for p in edited(change))
    problems = edited(one_more)
    assert len(problems) == 2 and problems[0].startswith("execution phase 0's connection rows: ")
    assert problems[1] == "execution phase 0's connection rows aren't registered"
    assert edited(other_salt) == ["execution phase 0's connection rows aren't its records' releases",
                                  "execution phase 0's connection rows aren't registered"]


@pytest.mark.parametrize("form", [1, 2])
def test_the_check_reads_the_release_order_past_an_execution_phase_that_isnt_complete(tmp_path, form):
    """A tick past its slack faults the second of five execution phases after its third bucket, and a connection
    released in it after that has lost its `END` in the records, so they count the later releases one short. The
    network certifier counted it: its heads of the later execution phases hold each release at its place in the
    release order. With the run's `warden.json` the check lays the same tables and takes every head, under form 1
    and form 2, and under form 2 the words of the faulted execution phase and the ones beside it. Without it the
    check can't place the later releases and takes no head past the fault; and with a log that gives a later
    release another place, the head isn't the tables'. The run is refused in each case on the faulted execution
    phase's own records."""
    params = {1: ROWS, 2: ENDS}[form]
    out = tmp_path / "run"
    prompts, answers = PROMPTS[:4], ANSWERS[:4]
    arrive = {0: 0, 1: T + 4, 2: 2 * T + 2, 3: 3 * T + 2}
    device, _ = served(out, tmp_path / "key", windows=5, params=params, prompts=prompts, answers=answers,
                       arrive=arrive, late={T + 3: 0.05})
    log = {c["ordinal"]: c for c in device.log}
    assert [log[k]["release"] for k in (1, 2, 3, 4)] == [0, 1, 2, 3] and all(not c["cut"] for c in device.log)
    assert T + 3 < log[2]["ingress_end"] < 2 * T
    records = ws.read_records(out, params)
    assert [r.status.complete for r in records["requests"]] == [True, False, True, True, True]
    handed = {1: [2, 3], 2: [3, 4]}[form]
    lines = phases_of(out, params)
    assert [w for w, kind in lines if kind == "registration"] == handed
    assert [w for w, kind in lines if kind == crr.NO_HEAD] == {1: [], 2: [0, 1, 2]}[form]
    short = CR.of_records(params, records["requests"], records["responses"])
    assert sorted(short) == [0, 1]

    def rows_problems() -> list[str]:
        result = ws.check(out)
        assert not result["accepted"]
        return [p for p in result["problems"] if "connection rows" in p or "no head" in p]

    unregistered = [f"execution phase {w}'s connection rows aren't registered" for w in handed]
    assert rows_problems() == unregistered
    (out / "warden.json").write_text(json.dumps({"connections": device.log}))
    assert rows_problems() == []
    s = ws.check(out)["summary"]["connection_rows"]
    assert (s["registered"], s["of"]) == (2, 2)
    assert s.get(crr.NO_HEAD) == {1: None, 2: {"signed": 3, "of": 3}}[form]
    held = CR.of_records(params, records["requests"], records["responses"],
                         {c["ordinal"]: c["release"] for c in device.log})
    assert [(w, r.order) for w in sorted(held) for stride in held[w] for r in stride] == [(0, 0), (2, 2), (3, 3)]
    swapped = [c | {"release": {2: 3, 3: 2}.get(c["release"], c["release"])} for c in device.log]
    (out / "warden.json").write_text(json.dumps({"connections": swapped}))
    wrong = {1: [2, 3], 2: [3]}[form]
    assert rows_problems() == [p for w in wrong for p in (
        f"execution phase {w}'s connection rows aren't its records' releases", unregistered[handed.index(w)])]


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


#: the cost command's flags of a small schedule
SMALL_COST = ["--slots", "2", "--lanes", "2", "--window", "32", "--every", "2", "--delay", "5", "--ingress-cells", "4",
              "--egress-cells", "9"]


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
                     "--height", "4", *SMALL_COST]) == 0
    assert json.loads(out.read_text())["releases"] == 5


#: pinned from the runner before it took `--lengths`: the digest of the schedule the cost command measures by
#: default, and its load's releases and handed bytes, to the grid and to the rows
COSTED_DIGEST = ("a45d70304d29e91ffab9df9860e7fb57fb63454452add2369af50be06d8ff1bebbf3d9828fa4d1bea82d48f0716592e9"
                 "51cf58267774e62a6a05f35b35c2b64b")
COSTED_LOADS = {"grid": (90, 1195378), "rows": (96, 1360768)}


def test_the_cost_is_of_form_1_by_default_and_of_form_2_a_classes_rows_are_the_execution_phases(tmp_path):
    """Without `--lengths` the cost command measures the schedule and the load it measured before it took the flag,
    held to that digest and those loads. With `--lengths 2` the default classes are the same sizes, each (P, O), and
    the schedule another. Its `rows` load is the costliest close the schedule has: R = `phase_releases` releases, all
    of the class whose rows are the most bytes, each its longest request and most ids. Its `grid` load is the
    classes in turn and no more than R releases or than the grids carry. At a small schedule the close hands
    execution phase 0 alone, and its tables are the lengths and fails tables' and the one class's R rows three
    times, the other class's of no row; a class of the other form's numbers is refused by the schedule."""
    default = crr.costed(crr.parser().parse_args(["cost", "--out", "x"]))
    assert (default.lengths, default.content, default.digest.hex()) == (1, crr.CLASSES, COSTED_DIGEST)
    for load, (releases, handed) in COSTED_LOADS.items():
        got = crr.loaded(default, load)
        assert (len(got), sum(size + 4 * n for size, n in got)) == (releases, handed)
    held = crr.costed(crr.parser().parse_args(["cost", "--out", "x", "--lengths", "2"]))
    assert (held.lengths, held.content) == (2, tuple(c[:2] for c in crr.CLASSES)) and held.digest != default.digest
    widest = (min(CT.request_room(512, 4096), 96 * held.payload), 4096)
    assert crr.loaded(held, "rows") == [widest] * held.phase_releases and held.phase_releases == 336
    cells = held.slots * held.lanes * held.window
    grid = crr.loaded(held, "grid")
    assert len(set(grid)) == 6 and len(grid) <= held.phase_releases and sum(n + 1 for _, n in grid) <= cells
    small = crr.costed(crr.parser().parse_args(["cost", "--out", "x", "--lengths", "2", "--classes", "4,4", "128,8",
                                                *SMALL_COST]))
    R = small.phase_releases
    rows = crr.loaded(small, "rows")
    filed = [CT.file(small, Released(0, size, n, False, bytes(size), bytes(4 * n))) for size, n in rows]
    assert (R, len(rows), filed) == (12, R, [1] * R)
    assert [CT.file(small, Released(0, size, n, False, bytes(size), bytes(4 * n)))
            for size, n in crr.loaded(small, "grid")] == [0, 1] * (R // 2)
    out = tmp_path / "cost.json"
    assert crr.main(["cost", "--out", str(out), "--lengths", "2", "--classes", "4,4", "128,8", "--load", "rows",
                     "--phases", "2", "--height", "4", *SMALL_COST]) == 0
    got = json.loads(out.read_text())
    slots = sum(n for _, n, _ in CR.tables_of(small)[:2])
    assert (got["parameters"], got["classes"]) == (small.digest.hex(), [[4, 4], [128, 8]])
    assert (got["releases"], got["tables"], got["rows"]) == (R, 8, slots + 3 * R)
    assert {got[k]["n"] for k in ("hand_ms", "pickle_ms", "build_ms", "commit_ms", "sign_ms")} == {2}
    for form, numbers, fix in (("2", "4,4,2", "drop its third number"),
                               ("1", "4,4", "give the rows of each of its tables")):
        with pytest.raises(ValueError, match=fix):
            crr.main(["cost", "--out", str(out), "--lengths", form, "--classes", numbers, *SMALL_COST])
