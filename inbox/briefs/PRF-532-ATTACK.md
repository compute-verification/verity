# PRF-532 ATTACK

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker ATTACK. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** The proved bound is $2^{-33.6}$ and the plain birthday bound at the same budget is $2^{-353}$. Say where the truth is. Word everything neutrally: this is an analysis of our own protocol.

**Answer:**
1. The best strategy you can construct for a prover with $2^{80}$ SHA-512 calls against Protocol 1 as deployed (the tree, the hm96 leaves, the base argument's sessions, the two-stage design): what success probability, at the floor and at points A and B? Is there any strategy that does better than finding one collision or guessing the sample?
2. Is any of the proof's loss matched by a strategy: the factor $N$, the square root, the extractor's reruns, the number of sessions? For example, does a prover that equivocates at one of $N$ positions with small probability each gain anything? Does the plurality definition admit a prover whose openings disagree with the committed trace without any collision (a prover that opens rarely)?
3. The lower bounds in the literature for Kilian-type protocols (Chiesa, Dall'Agnol, Guan, Spooner and Yogev, TCC 2024, is the starting point; check it): do they exhibit a commitment scheme, binding in the standard sense, under which the protocol is as weak as the bound says? If so, is our tree such a scheme, or does its structure rule that out?
4. What an honest statement of "proved against conjectured" would read like for us: the proved bound, the best known strategy, and the assumption under which they meet.
5. Anything in the deployed code that is weaker than the model the theorem is about (a truncated digest, a reused salt, a tree shared across audits, an opening the verifier does not check). Read `verity/ml/commitments`, `verity/core/service/spot` and the verifier of record.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **ATTACK**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-532 --file F.md --worker ATTACK` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
