[2026-11-13T21:28Z] c3c24882ff82
WATERLINE to network-accounting (lead), Fri 13 Nov 1:29 PM PST. Both done.

- **The note:** `20261113T2126Z-checkpoint-net310-step-two-cut-script.md`, synced (notes commit `222993b39d`).
- **The push:** the script as it stands, on a scratch branch of my own, `cursor/na-net310-cut-script-e388` at `fbb784f94712` (three files under `tools/move/cut/`, cut from #9754's head `a1d0bc0f8e48`; my checkout's identity; no pull request is made of it). The mend's branch `cursor/na-second-request-refusal-e388` is at `9520fcdf4b74`, pushed before noon and unchanged since.
- **4 PM PST still holds** for my line. A note and a push each hour from here.
- The note's items 3 to 5 want your ruling and will be in that line: the ratchet gate reads a definition's move as new sites where it arrives, whatever the lists say (off the merge gate; STAMPS's split passes because its new files hold none).

[2026-11-13T21:29Z] 85052e8c179f
PERIODPROOF to the lead, Fri 13 Nov 1:29 PM PST. **#9641's records commit is pushed: head `7236dd249c296835abb406129c6d8ca0329f6588`** (fast-forward over `4343ae38d5ce`; two locks, 7 lines for 8), and its body is set at that head.
- What the commit holds [run: `r20261113-184419-f08c`'s kept records against the head's, by section]: the `reads` record of `NetworkCheck.Period` in both locks; the `type_hash` of two honest witnesses in the properties' lock, each statement printed the same before and after [read: the review]; and the one `trust_gap` list in `main`'s order, the same 12 names. That last line is not a record: the run wrote it, I did not. Say if you want it out.
- The sections' order is not rewritten; the body says so in one sentence, without the issue's id.
- #8829: the mend of its `K` lemmas is tried on a node now (`r20261113-212813-cf2f`, one file). What it changes: `rowsTermsFails`, `K_failsOfFn`, `rowsKBoundsFails` and `rowsKBoundsFailPatterns` take `main`'s form, as `main`'s `rowsKBoundsCount` does: a link-window's triple with its `faulty` entries' bits, the factor `2 ^ faulty i` in each window's product, and the `charged` audits. 21 lines for 18, one file, no guarantee and no lock line.

[2026-11-13T21:31Z] 31bd37627adf
GATERELAY to network-accounting (lead), Fri 13 Nov 1:33 PM PST

**The earliest hour is now: the pull request opens today, unmarked, within the hour, with the verb, the runner's `session`, the piece and their tests. Nothing is cut.**

- Head `98d6493c0f2b348e0661ae7863d25b7518467a0f`, pushed at 12:23 PM PST. It holds `audit_windows` as its own commit (`cda61ac8013`) with its test, and the default height in the piece's contract: 13 for the longest session, `--minutes` 66 for 59 min 40 s and 126 for 119 min 21 s.
- The node's run `r20261113-202346-2d1a` is still going [read from its log at 1:31 PM PST]: the two suites whole have passed, the gate's 689 and the network traces' 746; of `suites.py --changed fa3e619ee146 --quick`, 45 of 47 suites have passed or were cached, none failed. I open the pull request when it ends, so the body says what ran, and send you its number.
- The body says, as you asked: the two names that change on the rename's day, where the contract is stated; under "For the pin", that the sha is `main`'s commit holding the pull request and that you send it, with no sha and no branch named.
- #9782: nothing pushed, and nothing will be until you say #9667 is on `main`.

One thing I changed that you did not ask for. The no-growth lint counted 150 retired-term sites in what I added. I rewrote my own text to say "execution phase" where it said "link-window", as `warden_serve.py` already does in places, and renamed my tests' local `warden` to `relay`. 68 sites are left, each a use of a name the code already has (`warden_*`, `WardenParams`, `"device": "warden"`, the verb `warden-rehearse`), and the body says so. The gate's refusal messages for a session now read "execution phases", where a rehearsal's still read "link-windows".

[2026-11-13T21:32Z] 6425539f306c
TERMGB to network-accounting (lead), Fri 13 Nov 1:32 PM PST. **The open point on `weights-tie-other` is settled by a run: under #9329's rule that tie's audit exits 1, and the mend cannot change it. Your choice is needed before #9329 is marked. I build nothing on it.**

**What ran** [ran, `r20261113-194903-d5c7`, the tie's arm at `10b60872b98f`; read: each tie's `RECORDS` line].
- `weights-tie`, the honest tie: accepted, 613 of 613 units sampled, pins written (`pinned: true`, 16 weights). #9305's flag `work-rate:613/613` is the rule it ran under.
- `weights-tie-other`, audited with `--forged`: the record rejects, every check before the sample passed, and the audit exits 1. `no_proof_released: false`, `lean_refused: null`. No root is pinned (`pinned: false`), which is the tie's right end; the arm still counts the tie failed, because a refused tie's audit must exit 0.
  - All 57 sessions have 2 proofs in the verifier's record. No proof was held back, so reading the proof server's log changes nothing here.
  - Lean refused three sessions, each at setup and on the registered reads: "setup: registered reads: port o0: table row 0 does not open … registered root at position 0". That is a refusal before any proof is read, so `lean_refused` has no value.
- `weights-tie-forged`: its audit was running when I read the job. Its lines follow when it ends, at `10b60872b98f` and at the mended `6476be339903`.

**What it means.** On `main` and at #9305's head a `--forged` audit passes when any Lean verdict is not accepted, so this tie passes there. Under #9329's rule it passes on neither check. If #9329 lands as it is, `PP_WEIGHTS_TIE=1 run.sh` exits non-zero on this tie. No suite runs that arm.

**The choices, corrected by the run.** My line of Thursday gave three; the third is gone.
- **A. A mode of its own for a refused read.** `records_audit.py audit` takes a flag that `run.sh` gives to `weights-tie-other` alone. It passes when every check before the sample passed, the record rejects, and Lean refused a session at setup on its registered reads; proofs may have left, since no unit is false. About 15 lines in `records_audit.py`, one in `run.sh`, one test, as one commit on #9329's head. Thursday I described it as "Lean refused a proof it read": the run shows the refusal is at setup, so the mode reads that reason, not a proof's.
- **B. Leave it and say so.** #9329's body names the tie as exiting 1 under its rule, and the arm stays red on that tie until a mode exists.
- **C, pass on either check, does not exist here:** `lean_refused` has no value on a refusal at setup, so it would pass nothing more.
- **My advice: A,** because B lands a change that turns a green arm of `run.sh` red. It touches no file of the tie's.

[2026-11-13T21:35Z] 89c9131f6bc3
PERIODPROOF to the lead, Fri 13 Nov 1:35 PM PST. Your 1:30 PM PST word is read; the rehearsal of #9641 with #8708 and #9664 waits for the head you send, behind #9663's push and #8829.
- **#8829's mend is pushed: `c441db46a4c4a93aa9ec6db85ccd3dcd0f1ae5ff`**, one file, 21 lines for 18, as my 1:36 PM line said. `verity lean try` on it: Lean exit 0, none of its 9 theorems uses an axiom beyond the three [run: `r20261113-212813-cf2f`]. Its job waits for #9663's new head, which it merges, and its names commit.
- **#9663, for the fact you relay from POLICYROWS:** its records run now (`r20261113-212246-2df8`) is a script's step, since it gives the one moved name. When the two commits are pushed I launch the plain `lean_changed.py --records security --update` at the pushed head by its own command line, the form your 12:51 PM line names, so the store holds a pass on the head's own Lean inputs. I send you its id with the head.

