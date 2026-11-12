**`verity lean bits` is open as [PR #9427](https://github.com/compute-verification/verity/pull/9427), unmarked. It lists all 1,569 guarantees in the security locks; 16 of them get a number, and the new test ties one printed figure per point to a Lean `norm_num` theorem.** It holds proofs' #9346 and gets its ready mark once #9346 is on `main`.

What it does, for someone new to this:
- Each `lean-audit.json` lists the guarantees. The command prints one line per guarantee: its owner, whether its statement bounds a probability, its error term, and the bits of that term where something gives it a number. It also prints what is left of those bits over a week, a month and a year of audits, at one datacenter and at the whole world (counts from the census).
- Today every number comes from a "mirror": a short Python copy of one Lean error term. A mirror is printed only while the lock still holds the exact hashes of the guarantees and definitions it copies. If one moves, the test fails and tells you to re-read the statement and update the mirror.
- There are six mirrors: SPOT's binding term (`εregAt`) and its extractor term (`KSAt`), C-Flock's `Claims.KS`, the one-wait `hashBound`, Pearl-C's `δPearlC`, and PoUW's binding per prover. Each is one term of its bound, so a guarantee's security is at most the printed figure.
- The test checks `hashBound` against the two `norm_num` theorems in `TwoOneWaitNumbers.lean`: the mirror's exact value at their numerals must be at most their right side, and the command must print the same figure. It does not show that `hashBound` is the guarantee's bound; that is the next step, the facts' `bound` field.
- Whether a statement bounds a probability is read from its Lean text (marked `~`), unless a Lean check's facts are given with `--facts` or `--run`.

Suites: SUITES_LINE

Not done:
- mirrors for the recursion, the ZK loop, `deltaPub` and PoUB;
- period columns for terms that are not per audit;
- `--run` tried against a records run of the current layout.

Proofs should read the mirrors' figures against the design script's before anyone quotes them; the PR body lists where they differ.
