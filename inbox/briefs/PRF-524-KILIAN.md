# PRF-524 KILIAN

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker KILIAN. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** SPOT's Protocol 1 is Kilian's protocol with the sampling PCP: commit to the proof string, sample, answer under the commitment. The literature has tight concrete analyses of exactly this in the standard model, with a vector commitment's position binding as the premise. Find what they give us.

**Start from** (from the coordinator's memory, so check each): Chiesa, Dall'Agnol, Guan, Spooner and Yogev, "Untangling the Security of Kilian's Protocol: Upper and Lower Bounds" (TCC 2024); Chiesa, Dall'Agnol, Guan and Spooner, "On the Security of Succinct Interactive Arguments from Vector Commitments" (2023); Barak and Goldreich's universal arguments; the IBCS chapter of Chiesa and Yogev's book. Add what you find.

**Answer:**
1. The exact form of their bound: for an adversary of size $t$ and success $\varepsilon_{PCP}+\varepsilon$, a position-binding adversary of what size and what success? Is the loss in time (a factor like $\ell/\varepsilon$), in probability (a square root), or either?
2. Transplanted to Protocol 1: our "answers" are not openings sent in the clear but openings the base argument's extractor recovers, at $e_s$ expected reruns a session. What does that do to the bound?
3. The number. Under `StrictBirthday` (a finder of cost at most $q$ succeeds with probability at most $q^2/2^{513}$), a reduction that turns success $\varepsilon$ into a finder of cost about $T\cdot w/\varepsilon$ and constant success gives $\varepsilon\lesssim T\,w/2^{256.5}$, and one with success about $\varepsilon$ gives $\varepsilon^3\lesssim T^2w^2/2^{513}$. Which is it, what is $w$ for us (does $N$ enter, or only the number of queried positions?), and what are the bits at the three points?
4. The lower bounds: is a loss of this kind forced on every black-box reduction to collision resistance, and does it apply to our setting?
5. The theorem that would replace B.2 or sit beside it, in the draft's notation, and what it takes in Lean: which of `Security/Proofs/Core/Game/Rewinding.lean` and `Security/Proofs/Flock/SamplingGeneric/Reg.lean` survives.

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

- You are a cloud worker of the lane **proofs**, named **KILIAN**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-524 --file F.md --worker KILIAN` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
