# PRF-527 NFACTOR

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker NFACTOR. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** The term is $N\sqrt{\beta'}$ because the committed trace is a plurality over coins at each of $N$ gates, and a run that opens a node to a non-plurality value is paired with one independent second run. Find every way to lose less.

**Answer:**
1. Where $N$ and the square root come from, line by line: `Security/Proofs/Flock/SamplingGeneric/Reg.lean`, `Security/Proofs/Core/Game/Rewinding.lean`, and the definition of the committed value (`Freg`, `regVal`) in `Security/Definitions/Claims/SamplingGeneric.lean`. Is $N$ a union bound over every gate, over boundary nodes, or over nodes the sample can reach?
2. Each alternative, with its bound and its bits: (a) route 3b, a rewinding count of $\sqrt{N\cdot Q}$ (TWOPROVE is scoping it; read its entries and do not repeat them, extend them); (b) a union over the sampled blocks' boundary only; (c) the number of blocks $n_{PU}$ or of rows in place of gates; (d) a committed trace defined by a threshold above one half, or by one fixed second tape, in place of the plurality; (e) the tape-averaged form of REGX (`cursor/proofs-regx-efe1`): what does it change in the number?
3. What each alternative does to the statements that rest on the committed trace being fixed before the sample: Corollary 3.16, Claim 3.19 (repeated audits) and Section 4 of the draft, and `WeightSecurity` in the Lean.
4. The best this direction can reach without changing the premise, and with `StrictBirthday`.

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

- You are a cloud worker of the lane **proofs**, named **NFACTOR**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-527 --file F.md --worker NFACTOR` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
