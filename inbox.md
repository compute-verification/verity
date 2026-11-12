[2026-11-12T14:58Z] 34f737817d28
TERMGB to the network-accounting coordinator, Thu 12 Nov, 7:00 AM PT. **NET-369: pull request #9329 is open and unmarked at `c77b01b8b79af859bdfd33ea7e64b150e9717977`, on `cursor/na-forged-no-proof-b4e6`. The stand-in ran at that head under both modes, exit 0 each, and prints the new lines.**

**The runs, at `c77b01b8b79a`, exits as their statuses say** [ran]

- `r20261112-144142-7045`, the stand-in under `--zk`: exit 0. The forged rank passes on `no_proof_released`; `lean_refused` is `null`, with the reason beside it.
- `r20261112-144225-5ed3`, the stand-in under `PP_PLAIN=1`: exit 0. The forged rank passes on `lean_refused`; `no_proof_released` is `null`, "the sessions ran behind no ZK server".
- `r20261112-144251-1977`, the real two-member forged test with the Lean verifier, the ZK server and the prover built in the run: exit 0, 1 passed.
- `r20261112-144330-b96d`, `suites.py --changed 2d5f7a169d85 --quick`: exit 0, 6 suites passed, 0 failed.
- The same four ran at `ce6d38906ec1`, one docstring sentence earlier, each exit 0.

**What `no_proof_released` reads**

Two files of the forged member's own session, both of which the audit already writes. Both must hold.

1. The verifier's record of the session (`session-m<k>/sessions/*/session.json`) lists no proof.
2. The prover's record of it (the `LIVE` line of `session-m<k>/prove.log`) gives `prover_stopped` opening with `FIREWALL-REFUSED`, the ZK server's own text for a message it holds back.

It has a value only where the summary says the sessions ran behind the ZK server. `prove_bundle.py` 273-275 already applies the same rule to its own tampered sessions.

**The test fails on `main`** [ran, on my VM]: with only the test files' changes on `main`, all 15 new cases fail, by signature. So I also ran `main`'s own rule on each case's records: it differs from what the tests expect in 10 of 14 cases, this head's in none. Six mutations of the rule are each caught.

**What I am least sure of**

- **The check reads that the ZK server refused, not which of its checks refused.** Any refusal by it of the forged member's session, with no proof recorded, passes. The honest arm of the same run is the control.
- **It reads `a0.run`'s log format**, since a5 keeps `prover_stopped` nowhere else. If that format changes a forged `--zk` run fails. It fails closed.
- **`NO_SESSION_FILES` sits in `records_audit.py`**, not in `a0.lean_refused`, which still counts a `session files:` refusal. That is the line you tell proofs.
- **`records_audit.py audit --forged` is told no forged member**, so any sampled member's held proof passes it.
- In this suites run `benchmarks/one_stage` shows 2 skipped. I take them to be the two-member module's two tests, which skip without the built binaries [argued; the node's log not read]. The separate run built them.

**For you**

- `queue open` warned that NET-369 has no goal label. The body's `Goal network/slice-clean` is its parent's. I changed no label.
- The change also rewrites one sentence of `benchmarks/network_e2e/chain.py`'s docstring, which stated the old rule. That file reads only the arm's `ok`. It is the sixth file.
- The ratchet gate over #9305's head exits 1 for 3 retired-term sites, each the code's own text. It is off the merge gate; I touched no allowlist.
- It merges clean with `main` at `7d296de0f3e9` and with #9237 at `9d4aec0111fb` [`git merge-tree`]. Against lean's tip `f3837f57d113` it adds no conflict to #9305's own, by count.
- #9329 does not change `served_verdicts.sh`, so the follow-on's refusal words are left for you to list, as you said at 6:55 AM PST.
- #9305: head unmoved at `2d5f7a169d85`. I wait for your word that lean's landing is on `main`.

Not run: the whole check, `circuit-check`, any Lean check, a11.

TIMER: none

