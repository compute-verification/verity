**Run on its own, the `tools/verity/circuit_check` suite takes about 24 minutes longer at the branch's head than at its base on the same machine. The cause is not the new entries or the spots. Editing `targets.py` means no stored catalog record counts, so the suite rebuilds circuit-check's catalog itself. A one-comment edit to `targets.py` costs the same. Fewer spots would save nothing, because neither the suite nor `check`'s circuit-check step runs the spots.**

What I measured:

- Run `r20261113-015932-d77d` (vy-nebius-cpu-4, one check slot) ran the suite whole and fresh three times in a row: 1,833 s at the head, then 400 s at the base (the three changed files put back as at `1885cf1fe08c`), then 2,056 s at the head again.
- The summed per-test time differs by only about 330 s, and the tests that grew are unrelated to FP8. For example, `test_metadata_file[verity/definitions/ops/metadata.json]` went from 0.02 s to 94 s, and each `test_split_prims_serve.py` test from 44 s to 100 s. In each worker, the first test that reads the catalog is charged for building it.

Why. The suite takes its catalog from the catalog bindings' records in the host's `~/.cache/verity-check/circuit-check` (`tests/conftest.py`, `_from_records`). A record counts only while every input it read is unchanged, and `targets.py` is an input of every binding. When none counts, the session builds every binding itself, which takes about 25 minutes. Run `r20261113-034352-d28a` (vy-nebius-cpu-6, whose host records don't match the base either) shows this:

- the unchanged base took 1,753 s;
- the base plus one comment line in `targets.py` took 1,784 s.

What I could not show is the suite at the head with records made for the head. In run `r20261113-055542-22f5`, I stored records with `circuit-check --all --records-only` and handed them to the suite. The session refused them, before any test ran: `verity.definitions.kernels.tc.circuit-check#_core_roots's record doesn't count (a read changed, or it lists other roots than its binding makes now)`. That is in the cache and conftest code, which this issue leaves to its owners. Whether `check` itself pays the rebuild therefore depends on whether its suite finds records that count for the tree it checks. Any PR that edits `targets.py` or a sidecar is in the same position as this one.

Elsewhere, the change adds nothing measurable:

- `circuit-check --all`, as `check` runs it, checks the same 17,189 targets at the base and at the head: 9,678 s and 10,862 s on two different machines. The head's catalog prologue alone was 577 s slower, and the per-target time was 13 % slower across all targets.
- Only the two coordinate targets already in the catalog check differently: they are now checked as Calls, about 5 s each.
- The two opt-in walk tests took 4,104 s at the base and 4,075 s at the head.

On the smallest set of spots that still asks for the entry: any one of the three wide GEMMs should be enough to make the walk ask for the coordinate (and `F32Mul_v3`), since each is over the budget and calls both. That is inferred from the code; I did not run the walk with fewer spots. But removing spots changes none of the times above, so I'm keeping all four served sizes. I'm opening the pull request next, against `main` and unmarked.
