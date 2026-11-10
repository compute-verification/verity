# PRF-538 AUDIT-HEADLINE

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker AUDIT-HEADLINE. Deadline: Wed 11 Nov 06:00Z (Tue 10 PM PST). First answer within 90 minutes of your start.**

**Direction.** The binding figure was quoted on the Goals page for days ("$2^{-90}$ a run") while the theorem that mattered gave $2^{-33.6}$. Find every other quoted security number and check what backs it.

**Cover:**
1. The Goals page (`research goals show --json`): every description and blocker with a probability, a "negligible", a number of bits or a "proved". For each: the theorem or run it cites, and whether that theorem, evaluated at the deployed parameters, gives the number.
2. The SPOT post's claims ledger (the goal "Every claim traced to evidence", `proofs/spot-claims-checked`, and the note it links): the 34 claims; mark each security claim evaluated, formula only, or unbacked.
3. The draft (`inbox/attachments/spot-security-definitions.md` on `cursor/proofs-inbox`): every "negligible", every number (the $1\%$ and $95\%$ examples, "about 19 times", $2^{-257}$, $2^{-64}$, $2^{-192}$, $2^{-127}$, $\alpha^2\approx1.00135$), and Appendix D's table against `main`.
4. Weight security and the exfiltration capacity (`Security/Proofs/WeightSecurity/`, `Security/Models/`, the 300 GB target of the tier-1 goal): how the audit's error enters the capacity. Is the capacity conditional on no binding failure, and what does $2^{-33.6}$ an audit, or $2^{-1}$ at point A, do to the headline over a world-year of $1.78\times10^{16}$ requests?
5. DESIGN.md, the READMEs and docstrings on `main`: `git grep` for "negligible", "2^-", "bits of security", "128-bit", "256-bit" and the like; list each with what backs it.
6. The lock's `why` sentences for open premises that quote a number.

**Give each finding as a `SMELL` block** (the smell-check skill's format), then one table of everything you evaluated: the guarantee, the term, its formula, the deployed parameters you put in and where each comes from, the number, and whether a Lean theorem on `main` states that number or only the formula. "None found" needs the list of what you read. A term nobody has evaluated at deployed sizes is a finding even when the number turns out fine.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **AUDIT-HEADLINE**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-538 --file F.md --worker AUDIT-HEADLINE` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
