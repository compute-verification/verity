"""A relay run's connection rows (`connection_rows_run`) without the wall clock: the schedule's lengths rows and
content rows as flags that `rehearse` passes to its `serve`, and a run without them what it was; the hand as the
publisher queue's items, a slot stride each; a run of the network certifier on a `VirtualClock` whose publisher,
inline and in `serve`'s process, lays, commits and signs each handed execution phase's tables, which the offline
check holds to the records' own reading of the same cells and refuses when a registration, a salt, a kept line or a
record isn't the run's; the registrations' file, which a check that found no problem writes when it is handed the
SHA-512 of the network certifier's public key and its session: a first line that names the two, then a line for
each registration it held and each the publisher's own bytes, which the reader of the stop verdicts takes its
heads from, so that a deployment record of another key or session, a registration with a wrong signature or with
tables that aren't its records' leaves no file and every class at its whole schedule, and a check handed neither
is refused the file; an execution phase that isn't handed has no line; under the lengths rows' form 2
every execution phase has one line, its registration or the network certifier's word that it has no head, the run's
last included, and the check holds each to the one that is due, also past an execution phase that isn't complete;
and the cost's load and its measurement at a small schedule."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier.warden import Released, WardenParams

import test_stop_verdicts as TSV
from test_warden_serve import _run, ws

crr = ws.crr
#: the reader of the registrations' file, and its tests' records
SV = TSV.SV
#: a schedule with lengths rows and three classes, two of one size: 16 buckets an execution phase, a slot stride of 2
PLAIN = WardenParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=6, window=16, ingress_cells=4,
                     ingress_buckets=3, egress_cells=9, every=2)
CLASSES = ((4, 4, 2), (128, 8, 3), (128, 8, 1))
ROWS = dataclasses.replace(PLAIN, lengths=1, content=CLASSES)
#: the same schedule at the lengths rows' form 2
ENDS = dataclasses.replace(ROWS, lengths=2)
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
    order, two of one size among them, and `--lengths 2` the lengths rows' form 2, another schedule; `rehearse`
    passes its schedule's to `serve`, which reads the same schedule; without them the schedule is the one it was and
    no flag is passed; and a class that isn't three integers, a form that is none, or content rows without lengths
    rows, is refused. Under form 2 a rehearsal runs long enough to admit its last request (`closing_windows`)."""
    rows = ["--lengths", "1", "--content", *(",".join(map(str, c)) for c in CLASSES)]
    got = schedule_of(REHEARSE + SCHEDULE + rows)
    assert got == ROWS and got.digest != PLAIN.digest
    assert crr.flags_of(got) == rows and schedule_of(SERVE + SCHEDULE + crr.flags_of(got)) == got
    lengths = schedule_of(REHEARSE + SCHEDULE + ["--lengths", "1"])
    assert (lengths.lengths, lengths.content) == (1, None) and crr.flags_of(lengths) == ["--lengths", "1"]
    ends = schedule_of(REHEARSE + SCHEDULE + ["--lengths", "2", *rows[2:]])
    assert ends == ENDS and ends.digest not in (ROWS.digest, PLAIN.digest)
    assert crr.flags_of(ends) == ["--lengths", "2", *rows[2:]]
    assert schedule_of(SERVE + SCHEDULE + crr.flags_of(ends)) == ends
    for argv in (REHEARSE, SERVE):
        plain = schedule_of(argv + SCHEDULE)
        assert plain == ws._params(ws.parser().parse_args(argv + SCHEDULE)) and crr.flags_of(plain) == []
        assert plain.digest == PLAIN.digest
    for bad in (["--content", "4,4"], ["--content", "4,4,x"], ["--lengths", "3"], ["--lengths", "0"]):
        with pytest.raises(SystemExit):
            ws.parser().parse_args(REHEARSE + SCHEDULE + bad)
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


def refused(out, edit, registrations=None, under=None) -> list[str]:
    """The check's problems of the run in `out` once `edit` has changed its files, which are then put back; the
    check is given `registrations` as its registrations' file and is handed `under`."""
    saved = {p: p.read_bytes() for p in out.iterdir()}
    edit()
    try:
        return ws.check(out, registrations, under)["problems"]
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


def answer(tmp, file, registrations: list[dict]) -> TSV.LB.Stop:
    """The reader's answer (`stop_verdicts.of_runs`) on a run that holds the registrations' file as the check left
    it at `file`, or none where it left none, with the four verdicts of every class of each of `registrations`'
    heads, their verifiers' controls and the serve's registrations."""
    heads = [SV.head_of(doc) for doc in registrations]
    files = {"out/the-verdicts.jsonl": TSV.jsonl([r for h in heads for cls in SV.classes(h) for r in TSV.four(h, cls)]),
             "out/the-controls.jsonl": TSV.jsonl(TSV.the_controls()),
             SV.SERVES_FILE: TSV.jsonl(TSV.with_streams(TSV.SERVED))}
    if file.exists():
        files[SV.REGISTRATIONS_FILE] = file.read_text()
    return SV.of_runs(TSV.Store(tmp, {"a run": files}), ["a run"])


def handed(out) -> tuple[bytes, bytes]:
    """What the auditor hands the check of a run made under the network certifier's own key: the SHA-512 of the key
    the run's deployment record names, and its session."""
    key, session = device_key(out)
    return crr.key_sha512(key), session


def command(out, under=None, file=None) -> list[str]:
    """The `check` command of the run in `out`, handed `under` and given `file` as its registrations' file."""
    return ["check", "--run", str(out), *([] if under is None else ["--key-sha512", under[0].hex(), "--session",
                                                                    under[1].hex()]),
            *([] if file is None else ["--registrations", str(file)])]


def test_the_check_writes_what_it_was_checked_under_then_each_registration_as_the_publisher_kept_it(
        tmp_path, monkeypatch):
    """Handed the SHA-512 of the network certifier's public key and the session, and given a file, a check that
    accepts the run writes it. The first line names the hash and the session it was handed, and the reader's
    `checked_under` reads them from it; the hash is over the LMS public key ``I ‖ T[1]``, the bytes the deployment
    record writes in hex. Each line after it is one registration the check held to both, every handed execution
    phase's, in order: byte for byte the registration as the publisher kept it in `rows.jsonl`, and the head the
    reader's `head_of` reads from it is the head `check_registration` verified under the key, the session and the
    schedule. The reader, handed that file with the four verdicts of each head's classes, holds all three
    execution phases and prints what the file names. `check --key-sha512 HEX --session HEX --registrations FILE`
    writes the same file; a check that is given no file writes none, handed the two or not."""
    monkeypatch.setattr(SV, "FORMATS", {form: kind for kind, form in TSV.FORMATS.items()})
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "key")
    under = handed(out)
    public = bytes.fromhex(json.loads((out / "deployment.json").read_text())["key"]["public"])
    assert under[0] == hashlib.sha512(public).digest() and len(public) == 16 + 64
    assert ws.check(out)["accepted"] and ws.check(out, None, under)["accepted"] and not file.parent.exists()
    assert ws.check(out, file, under)["accepted"]
    written, kept = file.read_text(), (out / crr.ROWS).read_text().splitlines()
    first, *lines = written.splitlines()
    names = (under[0].hex(), under[1].hex())
    assert first == json.dumps({"checked_under": {"key_sha512": names[0], "session": names[1]}})
    assert SV.checked_under(first) == names
    assert len(lines) == len(kept) == 3 and written == "".join(line + "\n" for line in (first, *lines))
    key, session = device_key(out)
    for w, (line, publishers) in enumerate(zip(lines, kept)):
        assert publishers.startswith(f'{{"registration": {line}, "cost": ')
        verified, _ = CR.check_registration(json.loads(line), key, session, ROWS, ws.LINK, w)
        assert SV.head_of(json.loads(line)) == verified and verified.window == w
    held = answer(tmp_path / "read", file, [json.loads(line) for line in lines])
    assert held.held == 3 and TSV.names_under("a run", under=names) in held.why, held.why
    by_command = tmp_path / "out" / "by-command.jsonl"
    assert ws.main(command(out, under, by_command)) == 0 and by_command.read_text() == written
    assert ws.main(command(out)) == ws.main(command(out, under)) == 0
    assert sorted(p.name for p in file.parent.iterdir()) == ["by-command.jsonl", "registrations.jsonl"]


def test_a_run_under_another_key_gets_no_file_under_the_keys_hash_and_one_under_its_own(tmp_path, monkeypatch):
    """A run directory whose deployment record names another key than the network certifier's, a rehearsal's own:
    its certificates and its registrations are signed under that key and verify under it. Handed the hash of the
    network certifier's key, the check names the problem, the key and not the session, as its only one, and leaves
    no file; so does its command, which exits 1, with a file asked or not. Handed the hash of the run's own key, the
    same directory is accepted and its file is written, naming that hash: the reader holds all three execution
    phases on it and prints the hash, which it is not told to refuse."""
    monkeypatch.setattr(SV, "FORMATS", {form: kind for kind, form in TSV.FORMATS.items()})
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "a rehearsal's key")
    own, session = handed(out)
    ws.cs.load_key(tmp_path / "key", 5)
    true = crr.key_sha512(ws.cs.read_public(json.loads((tmp_path / "key" / "public.json").read_text())))
    assert true != own and ws.check(out, file, (own, session))["accepted"]
    earlier = file.read_text()
    result = ws.check(out, file, (true, session))
    assert result["problems"] == ["the deployment record's key's SHA-512 is not the one the check was handed: "
                                  f"{own.hex()}, not {true.hex()}"]
    s = result["summary"]["connection_rows"]
    assert (s["registered"], s["of"]) == (3, 3) and not result["accepted"] and not file.exists()
    assert ws.main(command(out, (true, session), file)) == 1 and not file.exists()
    assert json.loads((out / "check.json").read_text())["problems"] == result["problems"]
    # the command names the problem with no file asked too, and a check handed nothing holds the run to no key
    assert ws.main(command(out, (true, session))) == 1 and ws.main(command(out)) == 0
    # under its own key's hash: the file, which says so
    assert ws.main(command(out, (own, session), file)) == 0 and file.read_text() == earlier
    first, *lines = earlier.splitlines()
    assert SV.checked_under(first) == (own.hex(), session.hex()) and len(lines) == 3
    held = answer(tmp_path / "read", file, [json.loads(line) for line in lines])
    assert held.held == 3 and TSV.names_under("a run", under=(own.hex(), session.hex())) in held.why


def test_a_deployment_record_of_another_session_than_the_one_handed_gets_no_file(tmp_path):
    """Handed the hash of the run's key and another session than its deployment record names, the check names the
    problem, the session and not the key, as its only one, and leaves no file, removing an earlier check's; handed
    another key's hash too, it names both. A deployment record changed to name another session than the one handed
    is the same problem by name, beside its certificates, which verify under no other session."""
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "key")
    key, session = handed(out)
    other, wrong = bytes([session[0] ^ 1]) + session[1:], bytes([key[0] ^ 1]) + key[1:]
    of_session = ("the deployment record's session is not the one the check was handed: "
                  f"{session.hex()}, not {other.hex()}")
    assert ws.check(out, file, (key, session))["accepted"] and file.exists()
    result = ws.check(out, file, (key, other))
    assert result["problems"] == [of_session] and not result["accepted"] and not file.exists()
    assert ws.check(out, file, (key, session))["accepted"] and file.exists()
    assert ws.check(out, file, (wrong, other))["problems"] == [
        f"the deployment record's key's SHA-512 is not the one the check was handed: {key.hex()}, not {wrong.hex()}",
        of_session] and not file.exists()
    assert ws.main(command(out, (key, other), file)) == 1 and not file.exists()

    def another_session() -> None:
        dep = json.loads((out / "deployment.json").read_text())
        (out / "deployment.json").write_text(json.dumps(dep | {"session": other.hex()}))

    assert ws.check(out, file, (key, session))["accepted"] and file.exists()
    problems = refused(out, another_session, file, (key, session))
    assert ("the deployment record's session is not the one the check was handed: "
            f"{other.hex()}, not {session.hex()}") in problems and not file.exists()
    assert "responses window 0 isn't certified" in problems and not any("key's SHA-512" in p for p in problems)


def test_a_registrations_file_without_the_hash_or_the_session_is_refused_before_the_run_is_read(tmp_path):
    """A check given a registrations' file and not handed the key's hash and the session refuses, by name, before
    it reads the run: of a directory that holds no run it is that refusal, not the missing deployment record. A
    file already at the path is removed first, so none stands from an earlier check. The command refuses the file
    without both, with the hash alone and with the session alone, and it refuses the one without the other, and a
    hash that is no SHA-512, whether a file is asked or not; it writes no `check.json`."""
    none, file = tmp_path / "no run", tmp_path / "out" / "registrations.jsonl"
    file.parent.mkdir()
    key, session = bytes(64), bytes(32)
    for check in (lambda: ws.check(none, file), lambda: crr.check(none, {}, PLAIN, [], [], [], ws.LINK, file)):
        file.write_text("an earlier check's\n")
        with pytest.raises(ValueError, match="only by a check that is handed the SHA-512 of the network certifier's "
                                             "public key and its session"):
            check()
        assert not file.exists()
    with pytest.raises(FileNotFoundError):
        ws.check(none, file, (key, session))
    for flags in ([], ["--key-sha512", key.hex()], ["--session", session.hex()]):
        for asked in (["--registrations", str(file)], []):
            if not flags and not asked:
                continue
            file.write_text("an earlier check's\n")
            with pytest.raises(SystemExit) as refusal:
                ws.main(["check", "--run", str(none), *flags, *asked])
            assert refusal.value.code == 2 and file.exists() == (not asked), (flags, asked)
    for short in (key.hex()[2:], "5a", key.hex() + "00", "no hex"):
        with pytest.raises(SystemExit) as refusal:
            ws.main(["check", "--run", str(none), "--key-sha512", short, "--session", session.hex()])
        assert refusal.value.code == 2
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["out", "registrations.jsonl"]


def test_a_registration_with_a_wrong_signature_gives_no_file_and_the_reader_holds_nothing(tmp_path, monkeypatch):
    """A registration whose signature is changed still names its own head's digest, so the reader's `head_of` reads
    the head it read before, and a reader handed that registration would hold its execution phase: the reader
    checks no signature. The check refuses it under the key, so it writes no file and removes the one an earlier
    check wrote, with the lines of the two execution phases it still holds; the reader, handed what the check left,
    charges every class its whole schedule and says no registration was read. With the run's files put back the
    check writes the file it wrote before."""
    monkeypatch.setattr(SV, "FORMATS", {form: kind for kind, form in TSV.FORMATS.items()})
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "key")
    under = handed(out)
    assert ws.check(out, file, under)["accepted"]
    written = file.read_text()
    signed = [json.loads(line) for line in written.splitlines()[1:]]

    def other_signature(lines, salts) -> None:
        cert = lines[1]["registration"]["certificate"]
        cert["signature"] = ("1" if cert["signature"][0] == "0" else "0") + cert["signature"][1:]
        forged.append(json.loads(json.dumps(lines[1]["registration"])))

    forged: list[dict] = []
    problems = refused(out, lambda: edit_rows(out, other_signature), file, under)
    assert forged[0] != signed[1] and SV.head_of(forged[0]) == SV.head_of(signed[1])
    assert ("execution phase 1's connection rows: the registration's certificate doesn't verify under the network "
            "certifier's key in the session") in problems
    assert not any("execution phase 0" in p or "execution phase 2" in p for p in problems)
    assert not file.exists()
    left = answer(tmp_path / "left", file, signed)
    assert left.held == 0 and "verdict record(s) were read and no registration" in left.why[0]
    # the reader alone holds a head nobody signed, under a first line anybody wrote: only the check's file keeps it
    # from one
    file.write_text(TSV.checked(*forged))
    assert answer(tmp_path / "forged", file, forged).held == 1
    assert ws.check(out, file, under)["accepted"] and file.read_text() == written


def test_a_registration_whose_tables_arent_its_records_releases_gives_no_file(tmp_path):
    """A registration that verifies under the key and whose tables, laid again from the run's own records under the
    kept salts, aren't its head's (a row's salt changed; an id changed in the egress record) fails the check's
    second hold: the check writes no file and removes the one an earlier check wrote."""
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "key")
    key, session = device_key(out)

    def other_salt(lines, salts) -> None:
        salts[1] = salts[1][:-1] + bytes([salts[1][-1] ^ 1])
        CR.check_registration(lines[1]["registration"], key, session, ROWS, ws.LINK, 1)

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

    under = handed(out)
    for w, edit in ((1, lambda: edit_rows(out, other_salt)), (0, other_id)):
        assert ws.check(out, file, under)["accepted"] and len(file.read_text().splitlines()) == 1 + 3
        assert f"execution phase {w}'s connection rows aren't its records' releases" in refused(out, edit, file, under)
        assert not file.exists()


def test_a_check_that_finds_any_problem_writes_no_file(tmp_path):
    """The registrations' file is a whole check's or none: a run whose first execution phase's records were withheld
    has two registrations that the check holds, with no problem of the connection rows, and the check refuses the
    run, so it writes no file, removes one that was there, and its command exits 1. So does a run whose every
    registration holds and whose client says a connection failed, a problem the check finds after its records'. A
    run whose one execution phase isn't handed, and a run without lengths rows, have no registration, so their
    checks, which accept them, leave no file either, not one of the first line alone; a run without lengths rows is
    held to the session it was handed all the same. Nor does a check that doesn't end leave a file, of a run
    without its deployment record, and none leaves a part of a file beside it."""
    out, file = tmp_path / "run", tmp_path / "out" / "registrations.jsonl"
    served(out, tmp_path / "key", params=dataclasses.replace(ROWS, queue=5))
    file.parent.mkdir()
    file.write_text("an earlier check's\n")
    under = handed(out)
    result = ws.check(out, file, under)
    s = result["summary"]["connection_rows"]
    assert (s["registered"], s["of"]) == (2, 2) and not any("connection rows" in p for p in result["problems"])
    assert not result["accepted"] and not file.exists()
    assert ws.main(command(out, under, file)) == 1 and not file.exists()
    # a problem of the client's side alone, which the check finds past the certificates and the records
    whole = tmp_path / "whole"
    served(whole, tmp_path / "whole-key")
    under = handed(whole)
    assert ws.check(whole, file, under)["accepted"] and len(file.read_text().splitlines()) == 1 + 3

    def failed() -> None:
        client = json.loads((whole / "client.json").read_text())
        client["connections"][0]["error"] = "the connection was cut"
        (whole / "client.json").write_text(json.dumps(client))

    problems = refused(whole, failed, file, under)
    assert problems == ["client connection 0: the connection was cut"] and not file.exists()
    # a run with lengths rows whose one execution phase isn't handed: accepted, no registration, and so no file
    lone = tmp_path / "lone"
    served(lone, tmp_path / "lone-key", windows=1, prompts=PROMPTS[:1], answers=ANSWERS[:1])
    file.write_text("an earlier check's\n")
    result = ws.check(lone, file, handed(lone))
    s = result["summary"]["connection_rows"]
    assert result["accepted"] and (s["registered"], s["of"]) == (0, 0) and not file.exists()
    plain = tmp_path / "plain"
    served(plain, tmp_path / "plain-key", windows=2, params=PLAIN)
    under = handed(plain)
    file.write_text("an earlier check's\n")
    assert ws.check(plain, file, under)["accepted"] and not file.exists()
    # a run without lengths rows is held to what the check was handed all the same
    other = (under[0], bytes([under[1][0] ^ 1]) + under[1][1:])
    assert ws.check(plain, None, other)["problems"] == [
        f"the deployment record's session is not the one the check was handed: {under[1].hex()}, not {other[1].hex()}"]
    file.write_text("an earlier check's\n")
    (plain / "deployment.json").unlink()
    with pytest.raises(FileNotFoundError):
        ws.check(plain, file, under)
    assert list(file.parent.iterdir()) == []


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
    run's last close. The last two released nothing and their tables are the schedule's all the same. The check
    accepts the run, counts four of four registered, and no word that an execution phase has no head, of none due."""
    out = tmp_path / "run"
    device, requests = served(out, tmp_path / "key", process=process, params=ENDS)
    assert len(device.log) == len(PROMPTS) and all(c["failed"] is None and not c["cut"] for c in device.log)
    assert phases_of(out, ENDS) == [(w, "registration") for w in range(4)]
    listed = CR.tables_of(ENDS)
    for doc, salts in crr.read(out, ENDS):
        assert [(t["name"], t["rows"]) for t in doc["registration"]["tables"]] == [(n, rows) for n, rows, _ in listed]
        assert {name: len(s) for name, s in salts.items()} == {name: rows for name, rows, _ in listed}
    records = ws.read_records(out, ENDS)
    held = CR.of_records(ENDS, records["requests"], records["responses"])
    assert sorted(held) == [0, 1, 2, 3] and sorted(w for w in held if any(held[w])) == [0, 1]
    assert sum(len(stride) for w in held for stride in held[w]) == len(PROMPTS)
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


def test_under_form_2_the_check_holds_each_execution_phase_to_the_one_that_is_due(tmp_path):
    """The run above, its lines changed: the check refuses an execution phase with neither a head nor the word, one
    with a head where the word is due, one with the word where a head is due, one with two lines, a word signed for
    another execution phase, and a word whose signature is changed, each by the execution phase it names."""
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
    params = dataclasses.replace(ROWS, lengths=form)
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
