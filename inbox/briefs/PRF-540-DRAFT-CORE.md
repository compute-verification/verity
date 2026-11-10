# PRF-540 DRAFT-CORE

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker DRAFT-CORE. Deadline: the table by Wed 11 Nov 03:00Z (Tue 7 PM PST); the resolutions by 06:00Z. First answer within 90 minutes of your start.**

**Direction.** Daniel: "this is our current draft … Should find differences and resolve them." Compare the draft with the Lean on `main`, statement by statement, for the one-stage protocol.

**The draft:** `git fetch origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/attachments/spot-security-definitions.md` (fetched 23:16Z Tue 10 Nov from https://app.notion.com/p/SPOT-security-definitions-413399515d9e8162bbb7f8de48ca23e4 ; fetch it again through the Notion connector if you have it, and say if it changed).

**Cover:**
1. Definitions 1.1 to 2.4 and Lemma 2.5 (circuits, blocks, traces, correct, overwriting) against `Security/Definitions/` (the sampling definitions, `Security/Definitions/Claims/Sampling*.lean`, and what they read).
2. Section 3.1 (the sampling PCP, $\delta_s$), 3.2 (commitments, binding, hiding), 3.3 (the relation $\mathcal R_k$, completeness, knowledge soundness, zero knowledge), 3.4 (Protocol 1, the base argument).
3. Section 3.5: Proposition 3.13, Definition 3.14 (the committed trace: plurality over $r$ and $\zeta$, ties, partial), Theorem 3.15, Corollary 3.16, Theorem 3.18, Claim 3.19, Claim 3.20.
4. Appendix A (the instantiation) and Appendix B: Definition B.1 (parameters $(\kappa,\gamma)$, the fork finder, the rounds $\rho$), Theorem B.2 against `Compiled.KSReg` and `εreg` symbol by symbol (is $\rho\gamma$ the Lean's `k * γs`? is $N$ `nG`, gates or something else? is $\beta'$ `Bconf` as the draft describes it: "runs the prover twice, independently, at one value of the prover's randomness"?), Theorem B.3 against the Lean's knowledge-soundness form with every constant ($\alpha$, $\eta$, $m_o$, $m_i$, the rerun bound).
5. Quantifiers and order: what is fixed before what, what the probability is over, "efficient" and "negligible" where the Lean has an explicit finder, "for every prover strategy" against the Lean's prover type, what the extractor is given.
6. What the draft does not say and the Lean needs (premises, side conditions such as `Scope`, `Sessions`, `OpenCallsLe`, `Binds`), and what the draft says that no Lean statement backs.
7. The gap that started this: the draft gives no number for the explicit bounds at the instantiation of Appendix A. Draft the paragraph and table that would state them (leave the cells that depend on the swarm's outcome marked).

**Give one table of differences**, one row each: the draft's item (its number and sentence); the Lean's (name, file and line at the SHA you read, and whether it is on `main`, in review or absent); the difference in one line; which side is right and why; the resolution as exact text: the replacement sentence for the draft, or the Lean change with its size and whether it changes what a guarantee says or rests on (then it is Daniel's, through the coordinator). Rows with no difference are listed in one line at the end, so the coordinator can see what was compared. You edit neither the Notion page nor a locked statement: the coordinator applies the draft's changes as suggested edits and files the Lean's.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **DRAFT-CORE**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-540 --file F.md --worker DRAFT-CORE` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
