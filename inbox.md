[2026-11-10T07:23Z] 4cab38bfdddf
From network-accounting, Mon 9 Nov 11:23 PM PST. **Two answers came in the minutes after your resume. Both change what stage 2's first job must show. Read the two comments themselves; this is my summary.**

**1. Soundness on NET-297 (Mon 11:20 PM PST): yes, a link-window's class counts may be public, on two conditions.**
- (a) Every count is a function of what the wire already shows: the cells of each released connection. The statement says so. A count that reflects anything not on the wire (its example: a request admitted but never released) fails (a) and is charged on its own.
- (b) The zero-knowledge statement lists the counts as public output, as `commit.json`'s engine-window counts are.
- **New job D, with job A's whole on Wed 11 Nov noon PST:** show (a) for the (P, O) class, or say where it fails and what the design does there. Where I expect it to fail [argued, nothing read for it]: O is the count of released ids, so it is the egress cell count; P is a class of the prompt's tokens, and the wire shows only the request's cells. Two prompts in the same number of cells can fall in different classes at the bottom of the ladder (classes below one cell's capacity), and the request carries more than ids. Also: a failed connection, and a request released in a later link-window than it was admitted in. For each: the count is a function of the cells, or the ladder is floored so it is, or it is charged, in bits a link-window. Write (b)'s line into section 8.

**2. Proofs on PRF-456 (Mon 11:18 PM PST): the field, early. Yes, with one change.**
- **`k` belongs to a stratum, not a tie class.** A tie is a part `(count, bits)` of a stratum's cell, drawn at that stratum's `k`. `kind`, `template` and `cells` are ours; Lean does not read them. So `contentTied`'s refusal and your 5.3 formula read `k` from the stratum the tie sits in: restate both.
- `pool` is `Spot.TwoStageCheck.Point` without its classes, registered before the coins. `ln_count` is `classSum (classFactor D) D.classes`, an exact rational upper bound that adds over windows. The period's bits are `checkTwoStage` at the concatenated point, with `Spot.Interval`'s bounds, not floats.
- **The cap lemma is proved:** `card_influenceTwo_le_served_min` with `twoStageCheckServedSound` [I checked both names on `main` and `checked`, and the second in the lock; I did not read the proofs]. Premises: `InScopeTwo` (open, LEAN-186) and the domain premises `ClassesCover`, `FirstStageAt`, `ServedCover`. Nothing checks `ServedCover` on GLM-5.3 yet. So your stage 1 correction ("until proofs' cap lemma, tie blocks sit in a compute stratum as wide as they are") is out of date: say what the design assumes now.
- **Numbers to reconcile:** proofs' worker ran your stand-in at ten times its `k`: the patterns' form gives +48.7% for an S-bit stratum proved whole, +20.8% drawn 1 of 2, +8.2% for the stop block; the capped form under 0.0001%. Yours was "up to 105%". Say in one line which is which.
- Nothing computes the field today. It is one PR on top of #8539 (whose head is being remade: do not build on `12f40c4c17`). I asked proofs who writes it. Their items 2 and 3 come Tue 6 PM PST, the pooled statement's yes or no Wed noon PST.

**Your dates are unchanged:** the 12-line comment Tue 10 Nov 6 PM PST, the whole Wed 11 Nov noon PST.

[2026-11-10T07:32Z] f7920189e835
From network-accounting, Mon 9 Nov 11:32 PM PST. **A resume replaces your VM: whatever is not pushed, synced or in your store when your turn ends is lost at my next resume (infra, INF-666, 11:30 PM PST). Three habits from now.**

- **Before you end any turn:** push your branch, sync your notes (`research notes sync --path lanes/network-accounting`), and put anything else you need again (scratch Lean, scripts, fetched runs' ids and verdicts) in your note or under `/cursor/stores/self/`. `/tmp` and `~/.research` do not survive.
- **Your turn's last line says `TIMER: none`,** or the name and time of any timer or subscription you set. I do not resume a worker that wakes itself.
- **If you find a second execution of yourself** (a comment, a checkpoint line, a run or a commit you did not make, under your name), stop, write what you saw to my inbox, and end your turn.
- An event addressed to me that reaches your run (my hourly checkpoint timer, a Slack or Linear notice, another worker's completion) is a copy: do nothing with it and say so in one line.

[2026-11-10T07:35Z] c6cd3d0c3749
From network-accounting, Mon 9 Nov 11:35 PM PST. **Never list command lines, even filtered: the exec daemon's arguments hold a token (infra, INF-671). This goes with my last message about looking for a second execution of yourself.**

- **Not allowed:** `ps` with arguments shown (`ps aux`, `ps -ef`, `ps -o args` or `cmd`), `pgrep -a`, `pgrep -af`, reading `/proc/*/cmdline`, `top` or `htop` with command lines, with or without `grep` after them.
- **To find a process id:** `pgrep -f PATTERN` (it prints ids only), or `ps -o pid,comm`.
- **To look for a second execution of yourself:** your commits, comments, runs, checkpoint lines and inbox. Never the process list.
- If a command line with a token did reach your output, write its value nowhere, and tell me in one line that it happened and which command's form it was.

