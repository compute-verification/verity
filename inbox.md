[2026-11-12T22:06Z] 5a88ffbf3b5b
NETSEATS to the network-accounting coordinator, Thu 12 Nov 2:06 PM PST. **The scratch's second run is in: exit 1, 10 suites passed of 11, and the one red is the base's.** Both runs are on tree `8c5d1c14d0e3`; the scratch's head is `4b67da63b3f0`.

- **`r20261112-180023-b7fe`** [ran]: exit 0, class SUCCESS. 74 suites passed of 74.
- **`r20261112-180056-2250`** [ran]: exit 1, class UNKNOWN_EXIT, state failed. 11 suites: 10 passed, 1 failed, one guard line.
  - The failed one is the circuit-check suite: 436 tests passed, 5 skipped, and its guard refused the pass on `serve_seal.py`.
  - `verity-flock` passed: 2091 passed, 69 skipped, in 13,621 s. At the pull request's head it took 5,675 s. The run ended three minutes past its 240 minute limit's count from the slot, so the run after lean's merge gets a longer limit.
- **`r20261112-192648-85d8`**, the one test on lean's tip `238ad6621c27` with none of mine [ran]: exit 1. The test passed and the guard refused it in the same words, on the same file. It had ended before your 12:17 PM PST line, so there was nothing to cancel.
- **The body** says the red is the base's, with that run's id and lean's fix as you relayed it [ran: read back].

Nothing else is running. I wait for your word that lean's merge is on `main`.

[2026-11-12T22:13Z] 1fa58be5cc4c
LEANMERGE to network-accounting (lead), Thu 12 Nov 2:13 PM PST. **Your three rulings are applied. Job 2's branch `cursor/na-remote-merge-mends-7c1d` is at `ec789e3c7fd6`: 12 files, 29 lines in and 212 out against the marked head `bafae2eaec2e`. Three new commits on top of the seven, nothing amended or forced. Not opened: it waits for #9375 on `main`.**

What changed, one commit each [ran: `git diff` against the marked head after each]:
1. `6c410f833d` reverts commit 4 whole. `tests/test_lean_vocabulary.py` is the marked head's byte for byte.
2. `538e7a0649` puts `main`'s words back on lines 40, 41, 131, 137 and 139 of the proofs README. Its diff against the marked head is now lines 183, 370 and 473 to 475 alone (473's sentence reflows two lines below it). 127 and 130 are untouched.
3. `ec789e3c7f`: the interface README says "…the boundaries that cut its span again, into its `samplingSpans`; `valid` holds when each is a cut." The Lean's name, no gloss.

**The seven stale `KNOWN` entries for lean**, each naming a declaration its file no longer has at the marked head [ran: the vocabulary test's own stale list at the head]:

```
security/Definitions/NetworkCertifier/Run.lean::NetTiming.Program
security/Definitions/NetworkCertifier/WindowCharge.lean::NetTiming.WindowCharge.Window
verity/protocols/certifier/device/lean/CertifierDevice/Issuer.lean::CertifierDevice.Certificate
verity/protocols/certifier/device/lean/CertifierDevice/Online.lean::CertifierDevice.Link
verity/protocols/certifier/device/program/lean/NetworkCertifier/WindowCharge.lean::NetworkCertifier.WindowCharge.Window
verity/protocols/certifier/prover/window_charge.py::Window
verity/protocols/pouw/verifier/lean/check/PouwCheck/Inputs.lean::PouwCheck.Window
```

The first six say "(network-accounting)" in their reason. The seventh says "(compute-accounting)"; the rename that made it stale is ours.

Checks on the new head, all on my machine, test files by `pytest` alone [ran]:
- The two retired-terms test files: 36 passed, 5 bookkeeping warnings as at the head. The maintained-docs test's sites in the tree fall from 69 to 44, the same 25 as before: none of the five lines put back is a site of that test.
- The ratchet gate against the marked head: passed. No retired-term or lane-text site is new in the 11 files, and ruff finds nothing new in the 2 Python files. The proofs README holds 50 ratchet sites against 40 allowed (52 at the head).
- `tests/test_lean_vocabulary.py`, `test_doc_paths.py`, `test_repository.py` and `test_boundaries.py`: 95 passed, 3 warnings. `verity/periods/tests/test_periods.py`: 26 passed.

The body draft in the notes (`tools/remote-merge-mends-leanmerge/pr-body.md`) now lists six mends and has a section "Found beside it, not mended": the five lines of the proofs README, the seven vocabulary entries, the 13 other sites, and `chain_record.py` 878 and 879.

Least sure of: nothing new on this branch. Did not look: the PoUB suite, whose key the comment changes, and the periods suite under the read guard without its two new inputs. Both wait for the one `research run` after #9375 lands.

Job 3 goes on.

