# PRF-534 AUDIT-SPOT

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker AUDIT-SPOT. Deadline: Wed 11 Nov 06:00Z (Tue 10 PM PST). First answer within 90 minutes of your start.**

**Direction.** The binding term was a formula nobody evaluated. Evaluate every other term of SPOT's own theorems, in the draft and in the Lean, at the floor and at points A and B.

**Cover:**
1. Theorem B.2's first two terms as SPOT uses them: $\kappa$ and $\sqrt{\rho\gamma\beta}$ (in the Lean, `κs` and `Real.sqrt (k * γs * PrB)`). The second is a square root of a collision probability as well. TWOPROVE reads $\kappa_s\le2^{-202.98}$ and the fork finder's term as $2^{-141.5}$ at $T=2^{80}$ (PRF-517): check both, say which premise form the second uses (under `ExpectedCR` it would be about $2^{-87.5}\sqrt{k\gamma_s}$), and give $\rho$, $\gamma_s$ and $k$ as deployed.
2. Theorem B.3: $\alpha$, $\eta$ (it holds a fixed $2^{-127}$), $m_o$ and $m_i$, and the extractor's expected reruns $1+m_o+m_o m_i(1+q\mu)$ with $m_o\ge 2048\times$ the number of blocks. At point A that is more than $2^{46}$ reruns of a whole audit: is the extractor's work counted anywhere it must be, and is any term vacuous at the deployed sizes?
3. The sampling term $\delta_s(k)$, Remark 3.4 and the non-interactive comparison ("about 19 times the sample"): the arithmetic, and the rates the served system uses.
4. Completeness, $q\,\delta_c$: $\delta_c$ of the base argument as deployed, and $q$.
5. Claim 3.19, repeated audits: $\sum_t\varepsilon_t+\theta^M$ at the number of audits in a period and in the post's world-year.
6. Zero knowledge of Protocol 1 (Claim 3.20): $\varepsilon_h$ plus $\varepsilon_b$ a run, at the deployed number of leaves and runs.
7. The two-stage forms (Claims 3.22 and 3.23, #8512): the same terms.
8. `Security/Proofs/Flock/SamplingGeneric/Bits.lean`, `SpotFlock/Numbers.lean`, `Security/Proofs/Properties/Spot/Bits.lean`: which numbers are theorems, at which parameters, and are those the deployed ones?

**Give each finding as a `SMELL` block** (the smell-check skill's format), then one table of everything you evaluated: the guarantee, the term, its formula, the deployed parameters you put in and where each comes from, the number, and whether a Lean theorem on `main` states that number or only the formula. "None found" needs the list of what you read. A term nobody has evaluated at deployed sizes is a finding even when the number turns out fine.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **AUDIT-SPOT**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-534 --file F.md --worker AUDIT-SPOT` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
