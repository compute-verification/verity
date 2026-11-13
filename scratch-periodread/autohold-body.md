Lane proofs

Two tests of `tools/research/tests/test_auto.py` read the machine's own hold file, so the `research` suite is red wherever it runs on a held node 2. This points both at a path that does not exist, as the idle-machines test beside them already does. Two lines, tests only.

## What failed

- `test_the_probe_reads_live_claims_and_waiters_and_runs_the_trees_slot_py_on_the_pool`
- `test_a_node_2_that_takes_runs_only_through_a_check_slot_is_never_chosen`

Both run the probe on the machine they are on and read node 2's share from it. The probe answers the share with the hold when `/etc/vy/auto-hold` holds the node (`auto.HOLD_FILE`), before it reads the claims or the slot-only marker. vy-nebius-2 carries that file today, so both tests got `{"hold": ...}` where they expected the share and the error.

## Evidence

- **On `main`'s tree, on vy-nebius-2**: `r20261113-023127-b5e6`, at `8c11a1d5ab`: 2 failed, 74 passed. The diff printed is the node's hold.
- **Where it was met**: a quick tier on vy-nebius-2, `r20261112-232219-a005`: 60 of 61 suites passed, `research` red on these two.
- **Before the hold** the suite passed there: `r20261112-203704-77ca`.
- **With the fix, on vy-nebius-2**: `r20261113-023857-2380`, at `7764f46a62`: 76 passed, with the hold still on the node.

## Not checked

- No other test of the file reads node 2's share from the real machine: the others that run the probe read a pool's document, and they passed on the held node in both runs above.
- I did not run the whole `research` suite on this head.
