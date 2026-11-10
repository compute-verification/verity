# PRF-535 AUDIT-FLOCK

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker AUDIT-FLOCK. Deadline: Wed 11 Nov 06:00Z (Tue 10 PM PST). First answer within 90 minutes of your start.**

**Direction.** C-Flock is the base argument under everything. Evaluate every error term of its knowledge soundness and its zero knowledge at the deployed parameters and at a budget of $2^{80}$ SHA-512 calls.

**Cover:**
1. `Claims.KS` and its record (`Security/Definitions/Claims.lean`, `Security/Proofs/Flock/KS/`, `Security/Proofs/Flock/Claims/Numbers*.lean`, `Bits.lean`): $\kappa$ (`2k/cap`, quoted as $2^{-231}$), $k=\lceil 4N_0/e\rceil$, `εtable`, `N₀`, $\gamma$, the weight and the waits, the fork finder's term, the field's size and the number of queries. For each: the formula, the deployed value, and whether `Numbers` proves it at the deployed record or at another.
2. The figures on the Goals page until today, "$2^{-90}$ a run" and "$2^{-45.5}$ a year" (PRF-466): what were they bounds on, which premise form, and do they survive?
3. Which collision premise each term rests on, linear or quadratic, and whether the finder's cost is counted in full (the extractor's reruns, the verifier's calls, the prover's own hashing). Smell-check question 8: an uncounted extractor or simulator.
4. Zero knowledge: the simulator's error terms (`zkSimForkLeCapTau`, `zkSimForkLePads`, `zkViewClashQP`, the pads, the Goldreich and Kahan commitment of the verifier's coins), each evaluated; the simulator's running time; the "one bit a session" leak of the ZK server and the counted bits of where a session stopped.
5. The recursion (`Security/Proofs/Flock/RecGeneric/`, `Security/Proofs/Properties/Verity/Recursive/Numbers.lean`): how the errors compound over levels and outer statements, evaluated.
6. Any security parameter chosen for a target under 100 bits: a field of $2^{64}$ elements somewhere, a query count set for 80 bits, a 128-bit seed.

**Give each finding as a `SMELL` block** (the smell-check skill's format), then one table of everything you evaluated: the guarantee, the term, its formula, the deployed parameters you put in and where each comes from, the number, and whether a Lean theorem on `main` states that number or only the formula. "None found" needs the list of what you read. A term nobody has evaluated at deployed sizes is a finding even when the number turns out fine.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **AUDIT-FLOCK**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-535 --file F.md --worker AUDIT-FLOCK` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
