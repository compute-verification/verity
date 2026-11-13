**Outside `check`, the `tools/verity/circuit_check` suite took about 24 minutes longer at the branch's head than at its base on the same machine. The cause is not the new entries or the spots: it is the suite's catalog being rebuilt from scratch because `targets.py` changed. Inside `check` the catalog's records are made for the tree being checked, so I expect the change to add no time there. Fewer spots would not change this, because the spots don't enter the suite at all.**

What I measured. In run `r20261113-015932-d77d` (vy-nebius-cpu-4, one check slot), the suite ran whole and fresh three times in a row:

- at the head: 1,833 s;
- at the base (the three changed files put back as at `1885cf1fe08c`): 400 s;
- at the head again: 2,056 s.

The summed per-test time differs by only about 330 s, and the tests that grew are unrelated to FP8. Examples are `test_metadata_file[verity/definitions/ops/metadata.json]` (0.02 s to 94 s) and `test_split_prims_serve.py` (44 s to 100 s each). In each worker, the first test that reads the catalog is charged for building it.

Why. The suite's catalog (`tests/conftest.py`, `_from_records`) is taken from the catalog bindings' records in the host's `~/.cache/verity-check/circuit-check`. A record counts only while every input it read is unchanged, and editing `targets.py` changes an input of every binding. With no record counting, every session builds each binding itself, which takes about 25 minutes. The base was fast on vy-nebius-cpu-4 because that host's records matched it.

Two runs support this:

- In run `r20261113-034352-d28a` (vy-nebius-cpu-6), whose host records don't match the base either, the base itself took 1,753 s.
- The same run then makes records for the head with `circuit-check --all --records-only` and hands them to the suite (`CIRCUIT_CHECK_RECORDS`, as the graph's test-file runner does). @@HEADREC@@

`circuit-check --all` itself, as `check` runs it, checks the same 17,189 targets at the base and at the head. Only the two coordinate targets already in the catalog differ: they are now checked as Calls, about 5 s each.

Next: the pull request, against `main` and unmarked.
