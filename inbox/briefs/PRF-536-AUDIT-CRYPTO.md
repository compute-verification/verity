# PRF-536 AUDIT-CRYPTO

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker AUDIT-CRYPTO. Deadline: Wed 11 Nov 06:00Z (Tue 10 PM PST). First answer within 90 minutes of your start.**

**Direction.** The primitives under every protocol. For each, the security it is used at, in bits, at a budget of $2^{80}$, against a 100-bit target.

**Cover:**
1. hm96 leaves (draft Appendix A; `verity/ml/commitments`; the Lean that proves hiding for a random key): the digest 512, the salt 1,536, the matrix $512\times1536$. The draft splits $2^{-257}$ into "all but a $2^{-64}$ fraction of matrices" and $2^{-192}$ a pair of leaves: **a fixed public matrix that is bad with probability $2^{-64}$ is a 64-bit number under a 100-bit target.** Is that split forced? What split meets 100 bits, and what does $L\cdot2^{-192}$ become at the deployed number of leaves?
2. The trees (frame-v3-sha512) and the multiproofs: domain separation between leaves and inner nodes, between trees, audits and positions; any truncation; second-preimage against collision; what each binding claim assumes.
3. LMS signatures at $n=64$ (`verity/core/primitives/lms/`): the parameters, the number of signatures a key makes, the proved or cited security at $2^{80}$, multi-key and multi-target loss.
4. The challenge derivation and every seed (`verity/ml/randomness`, `verity/core/primitives/randomness`): lengths, the live coins' entropy, the two Fiat-Shamir exceptions of AGENTS.md (the remote bandwidth audit's pass seeds; PoUW's per-matmul noise seeds with the grinding term $\mu^m$): each evaluated at what a prover with $2^{80}$ hash calls can grind.
5. Duo receipts: keyed u32 multiply, xor and rotate folds, binding under the premise H-μ⁺ (premise ledger row R1). What is the binding in bits of one receipt and of an epoch's, and is a 32-bit word anywhere the security parameter?
6. Pearl's FP8 scheme as Pearl defines it: its own concrete security.
7. BLAKE3 or any other hash still on a served path.

**Give each finding as a `SMELL` block** (the smell-check skill's format), then one table of everything you evaluated: the guarantee, the term, its formula, the deployed parameters you put in and where each comes from, the number, and whether a Lean theorem on `main` states that number or only the formula. "None found" needs the list of what you read. A term nobody has evaluated at deployed sizes is a finding even when the number turns out fine.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **AUDIT-CRYPTO**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-536 --file F.md --worker AUDIT-CRYPTO` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
