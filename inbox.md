[2026-11-12T19:20Z] 023d398adc65
THREADS, Thu 12 Nov, 11:20 AM PST.

**`run.py` fits decisions 1 to 6 as written. Two small additions follow. I keep the suite at one thread by holding `run.THREADS` at 1 in the fixture, and by resetting the count after each test.**

Against the decisions (READ: `benchmarks/training_slowdown/training_slowdown/run.py`, `of_record.py`):

1. Both of a job's processes pass through `run.main`. The route's process is `-m training_slowdown.run *argv --inside`, and `limit_cpu` runs right after `parse`, before `inside` or `outside`. Nothing before that computes on torch: `parse` only calls `torch.cuda.is_available()`. `of_record.main` today calls neither `limit_cpu` nor a thread count. Its torch work starts in `counted`, under `row`. (READ)
2. The route's process gets the job's argv verbatim, so the count reaches it by the same default. Addition: the job's own process appends `--threads N` (its own count) to the route's command. Then the two processes hold one count even where the default is read otherwise (the tests' fixture, below). (READ; the addition is mine)
3. `"device"` is written in `Harness.so_far`, at line 535. I write `"threads": torch.get_num_threads()` there, which is what the process holds when it writes. `of_record.counted` reaches it through `finish`. (READ)
4. `Kept.load` runs in the job's own process, not the route's, beside its VERSION, seed and split checks. It compares the record's count with that process's. A state with no count loads. The joined record is `rec | ...`, so it states the new segment's count. Addition: `joined` also refuses a segment whose count differs from a kept record that names one, as it already does for seed and split. (READ; the addition is mine)
5. and 6. `of_record.row` reads the record and then calls `counted`. Just before `counted`, I set the record's count, or `run.THREADS` where it names none. A fact line `threads` says which: "N, the record's" or "8: the record names no count". (READ)
7. VERSION stays. `grade.rows_of`, `library.py` and `evaluate.py` read a record by named keys. The grader reads submissions, not job records. The one closed set is `stop_record`'s, which `so_far` doesn't make. (READ: grep across the package and `grader/`)

The fixture (READ: `tests/conftest.py`, and the callers of `run.main` and `of_record.main`):

- `one_thread` is session-scoped. It sets 1 and restores the count only at session end.
- Tests call `run.main` and `of_record.main` in the pytest process: `test_resume`, `test_run`, `test_sandbox`, `test_windows`, `synthetic.run_job` and `test_of_record`.
- With a default of 8, the first such call would move the whole session to 8 and leave it there. Every later module fixture and test would then compute at 8, where the fixture says 1, and every route subprocess would run 8 threads on a 4-CPU slot.

What I'm doing about it (INFERRED):

- The fixture holds `run.THREADS` at 1 for the session (`MonkeyPatch.context`). `parse` reads the default when it is called, so a job a test starts without `--threads` runs at 1 in both processes; the route's process gets the count by the explicit pass in addition 2.
- An autouse function-scoped teardown sets 1 back after each test, so a test that passes `--threads N` leaves no other test at N.
- The tests of the field and of the refusals use counts other than 1, passed explicitly.

Next: build it, then run the suite on a node at my head merged with `main`. Draft PR by 6 PM PST.

