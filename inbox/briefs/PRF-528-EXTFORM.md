# PRF-528 EXTFORM

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker EXTFORM. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** The draft states soundness twice. Theorem B.2 (the prover is bound to a trace fixed before the sample) carries $N\sqrt{\beta'}$. Theorem B.3 (an extractor outputs a witness for $\mathcal R_k$) carries only a linear $\beta_{ext}$: one run of an extractor that collects $m_o\,m_i$ accepted reruns, with a collision among its own hash calls as the only binding failure. Nobody has evaluated B.3 at the deployed sizes, or asked whether the applications could rest on it.

**Answer:**
1. Find B.3 in the Lean on `main` (the knowledge-soundness form of the compiler, `CompilerKSAt` and what it reads; `Security/Proofs/Flock/SamplingGeneric/`), and evaluate every term at the floor and at points A and B: $\alpha^2(\delta_s(k)+\kappa)$, $\eta$, $\sqrt{\alpha^4\rho\gamma\beta}$, $\beta_{ext}$. The extractor's expected reruns are $1+m_o+m_o\,m_i\,(1+q\mu)$ with $m_o\ge 2048\times$ the number of blocks: what is $\beta_{ext}$ under `ExpectedCR` and under `StrictBirthday`? PRF-466 and BINDTERM's finding of 8 Nov (in the worker log) priced the linear term once: read them and say what still holds.
2. $\eta$ contains $2^{-127}$ and is fixed by the analysis. Is the whole of B.3 at or under $2^{-100}$ anywhere? Where not, which term is in the way?
3. The real question: the draft says the applications need B.2 because B.3's witness may depend on the verifier's coins. Can the committed trace be defined as what this extractor collects with the prover's tape fixed (so it is fixed before the sample), with the binding failure the linear $\beta_{ext}$? Write the statement that would replace Theorem 3.15 and Corollary 3.16, say what Claim 3.19 and Section 4 need from it, and where it fails if it fails.
4. What it costs in Lean, and what it does to the two-stage form (#8512).

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

- You are a cloud worker of the lane **proofs**, named **EXTFORM**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-528 --file F.md --worker EXTFORM` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
