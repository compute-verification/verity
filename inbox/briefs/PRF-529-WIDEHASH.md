# PRF-529 WIDEHASH

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker WIDEHASH. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** Walk back "one hash: SHA-512 with its whole output". Every bound today has $2^{256}$ or $2^{513}$ in it because the digest is 512 bits. A digest of 768 or 1,024 bits with collision resistance to match moves the exponent, and the theorems keep their shape: with $2^{512}$ in place of $2^{256}$ under `ExpectedCR`, today's $2^{-33.6}$ becomes about $2^{-161.6}$.

**Answer:**
1. Which constructions give collision resistance of $2^{384}$ or $2^{512}$ honestly: a sponge with capacity 1,024 and a 1,024-bit output (Keccak's permutation at rate 576), SHAKE256 (no: capacity 512 caps it at $2^{256}$; confirm), a wide-pipe or double-length construction, a concatenation of two hashes (Joux's multicollisions say no for iterated hashes; confirm). What does each cost a byte hashed, on a CPU and on our GPUs?
2. The premise: `ExpectedCR` and `StrictBirthday` at the new exponent. Is $E[\text{cost}]/2^{512}$ believable for a 1,024-bit sponge (the generic attacks, the best known ones)?
3. The numbers at the three points under each premise form, for 768 and 1,024 bits, with everything else as today.
4. What changes across the stack, with sizes: the trees (frame-v3, `verity/ml/commitments`, the multiproofs), hm96's leaves (digest 512, salt 1,536, the matrix; draft Appendix A), the GPU hash kernels (`verity/ml/kernels/hash_gpu/`), hashing inside proofs and the recursion, LMS signatures at $n=64$, the challenge derivation, proof and commitment sizes, and the exfiltration capacity if any committed or opened bytes count toward it. What does the prover's overhead become (hashing is already a large share of it)?
5. A middle path: only the tree's inner nodes at the wider digest, or only the trees whose binding the sampling theorem reads.

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

- You are a cloud worker of the lane **proofs**, named **WIDEHASH**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-529 --file F.md --worker WIDEHASH` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
