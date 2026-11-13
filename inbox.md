[2026-11-13T17:58Z] 7f54ea3b7a01
TERMGB to the lead, Fri 13 Nov 9:58 AM PST. #9501's head is `b2e51a3184bb`: three commits on `0ab489582d96`, each a fast-forward [ran; read from origin].

1. `054e3990b50d`, `FORMATS` empty: the reader's line is `FORMATS: dict[str, str] = {}`, its comment and the docstring say no format is listed, the tie's among them, since no statement registers what the tie reads. Its test asserts `SV.FORMATS == {}`, no verdict read on, the four verdicts missing, 4 unlisted records and 4 unlisted controls. `test_stop_verdicts.py`: 27 passed on the VM [ran].

2. `bca84e3fdd3b`, the unfiled request's mend, **ingress refusing to the byte**. In `warden.py`: the schedule's `request_bound`, the cells' bytes and, in a schedule with content rows, no more than its largest class holds; ingress refuses past it with the problem `ingress`, as it refused past the cells. A schedule without content rows is as it was. Docs: `warden.py`, `content_rows.py`, the certifier's `PROTOCOL.md` ("The filing"). Its two tests, in `test_network_certifier_content_tables.py`:
   - `test_a_request_longer_than_the_largest_class_holds_is_refused_at_ingress_and_never_released`: the longest request a class holds (1,797 bytes at the tests' schedule) is released and filed; one a prompt id longer (1,808, within the 4 cells' 2,012) fails closed with nothing released, under either form; a schedule without content rows releases it. With ingress at its old bound this test fails: the longer request is released [ran, earlier this turn on `main`'s tree].
   - `test_an_execution_phase_a_longer_request_reached_releases_only_what_its_classes_count`: under form 2, the phase's head is committed and the rule (`class_charge.charge`) is asked with four accepted, complete verdicts on every class. With the bound the phase releases 1 and its head counts 1. With the bound put back to the cells' alone, inside the test, the same traffic's phase releases 2, its head counts 1, and the rule charges it 0 schedules: the hole, shown.

3. `b2e51a3184bb`, the response's side, at the schedule: `content_rows.classes` refuses a schedule unless a class of its largest P holds the most ids a response releases (`content_rows.most_released`: `egress_cells` − 1, or the schedule's bound on `max_tokens`). The message names the class, both numbers and what to change. Its test, `test_a_schedule_is_refused_unless_a_class_of_its_largest_p_holds_the_most_ids_a_response_releases`: four such schedules handed under each form, each refused; the same classes accepted once a response is held to what they hold; the longest request answered with the most ids is released and filed. One existing test's schedule of two small classes, which the check refuses, takes a response bound of 2.

**What ran on the VM at `b2e51a3184bb`** [ran]: the certifier's 14 test files of the device, its rows and the rule, 281 passed; the cost runner's `test_connection_rows_run.py`, 19 passed; ruff clean on the files changed. At `bca84e3fdd3b`: `tests/test_boundaries.py`, `tests/test_doc_paths.py` and the retired-terms lint, 44 passed.

**Not closed, and said in the commits:** form 1 (a release filed in a class whose rows are all held has no content row). I did not make the tables refuse an unfiled release: the device no longer makes one, and a refusal there ends the publisher's process.

**Two things to weigh.**
- The retired-terms lint counts the code's own names: the third commit adds 5 sites in `content_rows.py` and 3 in the test, each `max_tokens_cap` or `WardenParams` as an identifier. The lint warns tree-wide today (its allowlist is at the old paths), so nothing turns red [ran].
- `connection_rows.tables` already refuses a release of `egress_cells` outputs or more as "no release of the schedule" [read]. The same line could hold a release's request to `request_bound`. I left it: say if you want it.

**Runs of record, launched 9:57 AM PST:** `r20261113-175703-1fe5` at #9533's `10414e54a035` and `r20261113-175741-0c1b` at #9501's `b2e51a3184bb`, each `suites.py --changed d119240998be --quick`. #9501's will show the harness's eight red tests.

Next: the comment on NET-431 (form 1), the writer onto this head, the two bodies when the runs end, then #9305 and #9329.

