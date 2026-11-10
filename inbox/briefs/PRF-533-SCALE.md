# PRF-533 SCALE

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker SCALE. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** Whatever fixes the floor, today's bound gets worse with the number of sessions $n_{sess}$ and the circuit's size $N$, and is vacuous at a class audit ($N=2^{58.8}$, $n_{sess}=n_{PU}=2^{35.1}$). Find the structure that keeps both small at the served sizes.

**Answer:**
1. Where $n_{sess}$ enters and why the finder extracts every session of both runs (`runExt`, `Security/Proofs/Flock/SamplingGeneric/Efficient.lean`; TWOPROVE's C2 and C5). Would a finder that guesses one session, or extracts only sessions that touch one node, pay $\sqrt{n_{sess}}$ in probability in place of $n_{sess}$ in cost? Under which premise form is that better?
2. The session count the served system actually has, one-stage and two-stage, and what the Lean instance has (`compiledTwo`, `TwoDeployed.lean:138-140`; `SpotFlock/Code.lean:599-615`; TWOPROVE answers part of this by Wed 17:00Z: read its entries, do not wait for it). What is the smallest $n_{sess}$ a design can have: one session a template, one a sample, batching every sampled block into one session?
3. $N$ at scale. Is a class audit one commitment of $2^{58.8}$ gates, or can it be many commitments, each small, whose errors add (Claim 3.19)? Compare $N\sqrt{\beta'}$ for one big audit with the sum over $m$ audits of size $N/m$, under each premise form, and say what the sampling bound $\delta_s$ loses.
4. For each of the swarm's main routes (a linear binding term; no factor $N$; a random oracle; a wider hash), what is left of the dependence on $n_{sess}$ and $N$?
5. The served design you would recommend so that points A and B meet the target, and what it changes for network-accounting (PRF-456) and for vllm's serving.

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

- You are a cloud worker of the lane **proofs**, named **SCALE**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-533 --file F.md --worker SCALE` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
