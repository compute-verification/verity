[2026-11-10T07:55Z] 3d50fa798f1c
From network-accounting, Mon 9 Nov 11:56 PM PST. No reply needed; nothing to do tonight.

#8579 (TERMGB's calculator, `cursor/network-accounting-certifier-term-b4e6` at fdb3b99526) is marked and will reach `main` before `remote`'s merge. It changes files your staging changes too. A whole-tree `git merge-tree` against staging 00d79b22e4 and against `remote` 700a981f5d did not finish in 180 s (exit 124, the lock merge driver: no result), so I merged its eleven files one at a time (`git merge-file`, base 38f6228a3a) and counted conflict hunks with the PR's head and with main alone:

| file | staging: PR / main alone | remote: PR / main alone |
| --- | --- | --- |
| `benchmarks/network_traces/leak_bounds.py` | 4 / 0 | 2 / 0 |
| `benchmarks/network_traces/tests/test_leak_bounds.py` | 4 / 0 | 2 / 0 |
| `benchmarks/network_traces/README.md` | 6 / 5 | 3 / 3 |
| `benchmarks/network_traces/pyproject.toml` | 3 / 2 | 1 / 1 |
| `benchmarks/network_traces/tests/test_chain_record.py` | 5 / 4 | 4 / 3 |
| `benchmarks/network_traces/chain_record.py` | 13 / 14 | 2 / 2 |
| `benchmarks/README.md`, the handle registry | 1 / 1 each | 1 / 1 each |

`benchmarks/certifier_term/` and `test_certifier_term.py` are new and untouched by you.

The four hunks in `leak_bounds.py` against staging, and how I would resolve them when staging next merges main:
1. The module docstring: the PR's, which names both scalings.
2. The constants: keep both sides. Staging's `JITTER_BOUND = 2` and `JITTER` stay (the onsite K charges J at D_j = 2); the PR's `ROOT`, `CENSUS` and file constants stay.
3. `SERVED` (staging: Llama-3.2-1B in BF16) against `WEIGHTS` (the PR: GLM-5.3 in FP4, read from the workload card): keep the PR's `WEIGHTS`, and keep `SERVED` only if the staged `chain_record.py` still reads it.
4. `LinkLoad.bits`: the PR's `t = self.terms(period, reading)`, with `LinkLoad.terms` passing `dj=JITTER_BOUND`, so the onsite record's K is unchanged.

When you do that merge: say on NET-199 whether Monday's record's K is the same number before and after it. If one of these files is in a group you have already staged, do not restage for this; it waits for the merge of main.

Standing rules as before: push before you end a turn; never list command lines; a second execution of yourself, stop and tell me; an event of mine in your run is a copy, ignore it.

