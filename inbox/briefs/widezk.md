# WIDEZK: the ZK server's proofs for the wide leaf (49 files), on top of WIDELEAF's branch

**Read `inbox/briefs/common-20261111.md` first (the standing rules; all of them apply).**

**Why.** The wide leaf is worth about 4 bits of SPOT's binding error (modeled: 2^-102.58 a verification phase becomes 2^-106.58; goal `proofs/spot-bind-term`), and Daniel approved it. It lands in three pull requests. The first, #9040, writes the wide-leaf schedule down as data and is marked. The second makes the zero-knowledge verifier read that schedule at a statement named for it, and carries the security analysis that covers it in the same pull request, because the verifier gains no form ahead of its proof. Worker WIDELEAF (`bc-c55b117b-9232-5ef6-8139-c88812cab2c4`) holds the verifier half, which elaborates, and the base of the analysis. About 190 files of `security/` remain in which a script added one argument and the proofs need a hand. **You take the ZK server's part of them, so the second pull request can be marked on Fri 13 Nov and not on Sat 14 Nov.**

**What changed, in one paragraph.** A statement named `verity/flock-circuit+wide-leaf` (templated: `verity/flock-circuit/tmpl+wide-leaf`) sets one field, `Setup.wideLeaf`. `Zk.schedule` and `Zk.queries0` take it as their first argument and read `Flock.fast100At` (the wide-leaf schedule at a wide-leaf statement, `fast100` otherwise). In the analysis the same argument goes first on every function keyed by a table's size. The script that adds it is `lanes/proofs/tools/wideleaf_rewrite.py.txt` in the notes; it rewrote 163 files (814 uses). Read WIDELEAF's two entries first: `rg -n '^## 18:59Z WIDELEAF|^## 19:09Z WIDELEAF' inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5`, and its note `lanes/proofs/20261111T1651Z-report-wideleaf.md`.

## Start

1. `git fetch origin cursor/proofs-wideleaf-masked-782c`. WIDELEAF pushes it at about 19:30Z. If it is not there, read the entries and the note meanwhile and try again every 5 minutes; if it is still missing at 20:15Z, write an entry that says so and end your turn.
2. Create `cursor/proofs-widezk-<4 hex>` from that branch's head. Push only to it.
3. Your files, and no others: `security/Proofs/Flock/SessionLoop/**` and `security/Proofs/Flock/ZkAny/**`. WIDELEAF counts 49 files there that read `Level0`, `Link`, `SessionPlan.repCoins` or `Finish.tabOfLevel0`. If a file outside the two directories must change for yours to elaborate, do not change it: write the file, the line and the change in an entry, and I pass it to WIDELEAF.

## The work

- Take the files in import order. For each: the lean-proofs skill's one-file try through `research run`, with the axiom check. Fix what the script left: places where the version is not inferred, tables checked case by case that now have four more sizes (m = 32 to 35 at the wide leaf), and lemmas stated at `fast100` that must be stated at `fast100At`.
- **Thread the argument and nothing else.** No statement's meaning changes: a lemma that held at every table size of today's schedule holds at every table size of either schedule. If a bound must become weaker, a premise must be added, or a size must be dropped for a file to elaborate, stop on that file, say exactly what and why in an entry, and go on with the others.
- No records run and no lock regeneration: WIDELEAF does the one records run after it merges the three parts.
- When WIDELEAF's branch moves (its entries say so), merge it into yours; never rebase a pushed branch.

## Done when

Every file in the two directories elaborates at your branch's head with the axiom check clean, the branch is pushed, and your entry lists: the files changed by the script alone, the files changed by hand, every statement whose text changed beyond the added argument (name, before, after), and every proof that needed more than threading the argument, with one line on why.

## Times (UTC)

- Within 90 minutes of your start: a first entry with how many of the 49 elaborate as the script left them, how many need a hand, and your date for the rest.
- A checkpoint note and a push at least hourly.
- Target: all 49 by Thu 12 Nov 18:00Z. If your first count says otherwise, say the date you believe.
- When you would only wait on runs, end your turn with the run ids and their ETA; I resume you.
