# PRF-525 ROM

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker ROM. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** Walk back "no random oracle" for the commitments only, and keep live coins. In the random-oracle model a Merkle commitment is extractable from the prover's query log with no rewinding (Ben-Sasson, Chiesa and Spooner 2016; the interactive form, IBCS, in Chiesa and Yogev's book). Then the committed trace is whatever the log determines at commitment time, and the binding term is a collision or a guessed preimage among the prover's queries, about $(Q^2+Q)/2^{512}$.

**Answer:**
1. The exact bound for a Merkle tree with our leaves (hm96 rows: a leaf masks its row's digest with $As$ and publishes the hash of the salt; frame-v3-sha512; `verity/ml/commitments`, and the Lean under `Security/Definitions` and `Security/Proofs/Flock` that models them). Is a leaf's content extractable from the log, or only its digest?
2. Does the factor $N$ disappear, and the square root? Give the whole replacement for $N\sqrt{\beta'}+\beta'$, and the bits at the three points.
3. The base argument: C-Flock's knowledge soundness is proved by rewinding today ($e_s$ reruns a session, a fork finder under a square root, $\sqrt{\rho\gamma\beta}$). In this model does its extractor become straight-line too? What do $\kappa$ and that term become?
4. What the Lean already has. The game on `main` logs every hash call of both parties (`pLog`, `vLog`), and the finders read those logs. How far is a premise "H is a uniformly random function", or a narrower falsifiable one stated over the log, from what `Security/Definitions/Core/Interaction.lean` models today? Name the definitions that change.
5. What is given up, in plain words Daniel can weigh: one theorem about a fixed public hash becomes a theorem about an idealized one. Which published systems state their security this way, and which do not?
6. Zero knowledge and the recursion: does anything there break when H is an oracle (a verifier circuit that computes SHA-512 inside a proof)?

**Give each option as a card, in your comment and in your final message:**

~~~text
OPTION <name>
changes:      premise | statement | proof only | protocol | parameters | code   (which, in one line each)
walks back:   <the design rule or the draft's sentence it reverses, or "none">
bound:        <the formula, every symbol defined>
bits:         floor / B / A, each at n_sess = 1, 30, n_PU; T = 2^80 and 2^64; k = 1507 and the largest k
reaches 100:  yes | no | only at <points>
rests on:     <each premise by name: on main today | new, so Daniel's decision>
costs:        prover time, proof and commitment size, verifier, Lean (files, lemmas, a size), code
kills it:     <what would make it fail, and the cheapest step that would show it>
~~~

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **ROM**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-525 --file F.md --worker ROM` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
