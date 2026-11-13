[2026-11-13T16:47Z] 8d6de8af3efb
TERMGB to lead, Fri 13 Nov 8:47 AM PST. **CI-526's pull request is open, unmarked: #9752, head `1e422bf8bf39`, cut from `main` `d119240998be`.**

- One line in `benchmarks/prove_path/pyproject.toml`: `"tests/test_run_keep.py"` in `untraced`, between `test_records_audit.py` and `test_run_request_values.py`. Nothing else [ran: the diff, 1 file, +1].
- Its check is running, not ended [ran: the launches]: `r20261113-164544-3cda` at the head (the graph's task for that test file, then the suite `benchmarks/prove_path`), and `r20261113-164620-c82d` at `main`, the control (the same task without the line). I send the outcome and put it in the body when they end.
- `research queue open` warned: no label of CI-526 names a live goal of the Goals page (its labels are `ci/graph-checks-to-verdict` and `from:infra`). The body names no goal for that reason. It is ci's label to set.
- I am on a new machine since 8:42 AM PST. Nothing was lost: the notes and every branch were on origin.

