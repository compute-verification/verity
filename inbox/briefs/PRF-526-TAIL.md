# PRF-526 TAIL

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker TAIL. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** Keep the theorem's shape and ask what the cost accounting can and cannot buy. The context's claim is that no cap reaches 100 while the term is $N\sqrt{\cdot}$: the knowledge error $2k/K_0$ times the birthday term does not depend on the cap. Prove that or break it.

**Answer:**
1. Where the cap enters. `Security/Definitions/Claims.lean:314-323` (`ERep`, $\kappa$ as `2 * D.k x / D.cap x`), `Security/Proofs/Flock/KS/Flock.lean` (`KS_of_oneWait`), `KS/Deploy.lean` (`cap_pos`, the deployed cap $2^{256}$). Is $2k/\text{cap}$ a Markov bound that a sharper tail could replace, or the knowledge error itself, which no rewinding extractor that needs $k$ accepting transcripts can beat with fewer than about $k/\kappa$ reruns? State the lower bound if it is one.
2. The consequence, if it is one: knowledge error $2^{-100}$ by rewinding needs a finder of cost about $2^{100}\,T$ or more, so `ExpectedCR` (linear, $2^{256}$) can never give it at $T=2^{80}$, and only a quadratic premise (`StrictBirthday`) or a larger exponent can. Confirm or refute, with the numbers.
3. The draft's Theorem B.3 has an extractor with strict limits and exponential tails ($\eta=(1+\alpha)2^{-127}+e^{-m_o/8}+\alpha e^{-m_i/8}$). Could the base argument's extractor be given the same shape, so its failure is exponentially small in the cap rather than $2k/K_0$? What would that take in `Security/Proofs/Flock/KS/`?
4. TWOPROVE's restatement (its 23:10Z entry): C-Flock's knowledge soundness at a second cap is one lemma, `KS_of_oneWait` at `deployZkAt K0`. Given that, write the best bound each of these gives as a function of $K_0$, and its optimum: capped under `StrictBirthday`; capped with `AdaptiveBirthday` (ASK-272); each with the square root and $N$ removed (as the other workers' routes would).
5. One table: for each pair (premise form, shape of the binding term), the best bits at the three points. This is the map the coordinator hands Daniel, so make each cell checkable.

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

- You are a cloud worker of the lane **proofs**, named **TAIL**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-526 --file F.md --worker TAIL` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
