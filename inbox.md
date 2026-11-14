[2026-11-14T17:21Z] 890a5702759c
model: Claude Opus 5.5 (this turn's system prompt; run-info's `originalModelName` is null); subscriptions removed: none held.

READMAPS to NAMES2 and training, Sat 14 Nov 9:20 AM PST: two commits on `cursor/training-rename-2-a32b` (draft #10164), read at its head 1f03e873b722188465a9f92b84e5047508b7acd4. The head is the branch's tip on origin.

**The README cure is NOT READY 1: every name it now lists is kept, but `exposure.py`'s `eta` and `fit.py`'s and `library.py`'s `efficiency` keys are older names of the same term, and the list still leaves them out. The remote verifier's README commit is READY: it changes the same lines, and its sentence is true at the new base. All four patch-id pairs are equal.**

1. 529a760113dd0e86179051fca92dd2c688bac078, `benchmarks/nci/README.md`
- READ, at the head: each listed name is where NAMES2 says.
  - `bracket.py`: `"rho"` at lines 109–115, and `def efficiency` at 168.
  - `tiers.py`: `DEFAULTS` at 103–104 holds `"unmonitored"` and `"residual"`. `evaluate` (301) reads `p["unmonitored"]` (307), has `unmonitored` and `residual` in its terms (351) and returns `"efficiency"` (357). The printed columns are at 380.
  - `exposure.py` reads `p["unmonitored"]` at 198 and 205, and `statement.py` reads it at 83, 108, 150 and 351.
  - `fit.py`: `def dampener` at 200, `def attack_dampener` at 208, and the output keys `D_closed`, `D_fit`, `D_fit_low`, `D_fit_high`, `D_fit_train`, `D_replica`, `D_path` and `D_none`.
  - `statement.py`: `def dampener` at 126 and `def fit_dampener` at 155.
- READ: the glossary rows (`GLOSSARY.md`, Older names, at the head) cover every listed name: row 212 for `"rho"` and "training efficiency" η, row 213 for "dampener" D, row 227 for "unmonitored", and row 229 for "residual compute" and "monitored residual".
- RAN: the lint at the head (`python3 -m pytest tests/lint/test_repo_retired_terms.py`, in a worktree) passes, 21 passed. Its one bookkeeping warning names no file under `benchmarks/nci/`. The README's sites rise from 12 to 18 (one `unmonitored`, five `dampener`), within its allowlist count of 20.
- READ, still left out (scanned with an identifier search over every Python file under `benchmarks/nci/` at the head):
  - `exposure.py`'s `eta`. It is the function `def eta(p, tier)` (145) and the output keys `eta` (215) and `eta_T1`…`eta_T5` (209), and `statement.py` calls it as `X.eta` (398, 399, 422). Its docstring says "compute efficiency eta (`tiers.py`)": it is `tiers.py`'s `efficiency` under the symbol row 212 retires ("training efficiency" η). Today's η is the success threshold of T_𝒢(η; S), so the old name misreads.
  - `fit.py`'s `"efficiency"` key, which its `replica` and `path` outputs carry (227) and `attack_dampener` reads (257, 258). It is the same key the sentence lists for two other files.
  - `library.py`'s output keys `efficiency_achievable` and `efficiency_ceiling` (156–157, printed at 254). They are m / slowdown, the quantity of `bracket.py`'s `efficiency`. My earlier read counted these as covered; they are their own keys.
  - `statement.py`'s own output keys `D_fit_T5` (324) and `D` (351). The sentence gives `D_*` to `fit.py` only.
  - Names built on `residual`: `tiers.py`'s output key `memory_caps_residual` (357), and `residual_share`, `audited_residual` and `residual_mfu` in `tiers.py`, `exposure.py` and `statement.py`, and `fit.py`'s `residual_with_lag`.
- INFERRED, not older names of the slowdown's terms: `tiers.py`'s `rho_T`, `rho_PoS` and `rho_NCI` are rates (row 212 keeps ρ as a resource rate). `wipe_residual` is memory the fill misses, `kappa.py`'s `RESIDUAL` is the model's residual stream, `reference_efficiency` is the current assumption's name, and the "monitored compute" and "dark" words are prose in labels and docstrings.

NOT READY 1 529a760113dd0e86179051fca92dd2c688bac078

2. 11e4300b5a1eeeea0c39cbbd5a447a6f225695b9, `verity/protocols/compose/verifier/remote/README.md`, formerly b9fb40b93402f0d19975f67c21ec0e7bdcf752f9
- RAN: the two commits' added and removed lines are identical (a `diff` of their `+` and `-` lines is empty). The hunk header moved from `-14,8` to `-15,8`. The first context line changed from "premise until proved at remote." to "and the composed statement still takes it.", and the blob ids differ. The patch-ids differ for that reason: 21ce1b669ebc0700 against 5ba64f7061980c28.
- READ: the commit that moved the hunk is d7f7c9d7f52 ("security and remote verifier READMEs: the inference-only notes trimmed under the Markdown cap after the moves"), between b9fb40b9's parent and aea3218430d.
- READ: the sentence it leaves is "Lean keeps some older names: `*Horizon` means a monitoring period's pooled bound, `Session*` a run of a subprotocol, and `Window*` a period." It is true at 11e4300b and at the head:
  - `*Horizon`: `IntegrityHorizon` and the rest are in `security/`, and the README cites `Flock.Guarantees.IntegrityHorizon`. Glossary row 178 retires the Lean's `*Horizon*`.
  - `Session*`: the README cites `Accounting.Guarantees.TS.SessionSoundAt`. Row 207 retires "session" outside Flock's and the ZK server's sense.
  - `Window*`: `WindowCharge`, `WindowCoins`, `WindowRecord` and others are in `security/`. Row 204 retires `Verity.Window`.
  - Dropping "`Slowdown*` means overhead" is right, since slowdown is the current term.
- READ: no later commit on the chain changes this README (`git diff 11e4300b 1f03e873` is empty for it).

READY 11e4300b5a1eeeea0c39cbbd5a447a6f225695b9

3. The four pairs (RAN `git show X | git patch-id --verbatim`):
- 1f69e8facb88 and 355d2886100a: EQUAL (15d356fe41f3b976).
- a26c12c0ae73 and ce9996354cce: EQUAL (6a06ef3cfe8f05ca).
- a4535709e5ac and 106c29f81921: EQUAL (bf47a8de26982159).
- dfb06d585e92 and 4e22274fec8f: EQUAL (5dbf7eaf5dad0d2e).
- RAN, beyond the brief: my other earlier READYs also carry. a6066d3c981 and bd9c4adab4d are EQUAL, a19f6d475ae and cf6c64d054e are EQUAL, and 9f88db49cc5 and aef07856d46 are EQUAL.
- READ: all seven new commits are on the chain under 1f03e873.

MUST FIX
- READ: `benchmarks/nci/README.md`'s list of kept older names leaves out `exposure.py`'s `eta` (the function and its `eta` and `eta_T*` output keys, which `statement.py` calls), `fit.py`'s `"efficiency"` key in its `replica` and `path` outputs, and `library.py`'s `efficiency_achievable` and `efficiency_ceiling`. Cure, my suggestion: stop listing by file and say the code and its outputs keep the names built on `rho`, `efficiency` or `eta`, `residual`, `unmonitored`, and `dampener` or `D`. This also covers the SHOULD FIX item, and the next file to gain such a name.

SHOULD FIX
- READ: if the list stays per file, it also leaves out `statement.py`'s own `D_fit_T5` and `D` keys, `tiers.py`'s `memory_caps_residual` key, and the names built on `residual` (`residual_share`, `audited_residual`, `residual_mfu`, `residual_with_lag`).
- READ: the remote verifier's README still says "overhead" for the slowdown at lines 28 ("the overhead bound's remote regime"), 396 and 397 ("The overhead over remote, per period", "The overhead over a monitoring period"). Row 212 retires "overhead" in that sense. These lines are outside 11e4300b's hunk and were in b9fb40b9's base too, which I passed. A stale statement to fix on sight, in this chain or the README owner's next PR; it doesn't hold 11e4300b.

NOTE
- READ: the chain's base aea3218430d is a queue candidate, not yet on main; main's tip is 723560c29ec.
- INFERRED: my first read of this README named only `unmonitored` and `dampener`, and missed `eta` and the other `efficiency` keys. This time the scan covered every identifier and string key in the directory's Python.

For the lead to read
- READ: an event addressed to me, NAMES2's inbox message at 9:13 AM PST, which matches your brief. I saw none new addressed to @training this turn.
- RAN: no Lean was run and no run was launched. The worktree `/agent/repos/wt-rename3` (detached at 1f03e873) is left in place.

