# PRF-539 AUDIT-TABLE

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker AUDIT-TABLE. Deadline: Wed 11 Nov 06:00Z (Tue 10 PM PST) for the inventory and the design; the code after the coordinator's word. First answer within 90 minutes of your start.**

**Direction.** The cause of today's miss is structural: a guarantee can be locked with an error that is a formula, and nothing evaluates the formula at the deployed record. Build the inventory and design the fix. AGENTS.md: "Enforce deterministic requirements in code, with the fix in the failure message, rather than in prose."

**Deliver:**
1. **The inventory.** Every guarantee in every lock on `main` (`git ls-tree -r --name-only origin/main | grep lean-audit.json`, the fixtures apart) whose statement bounds a probability. For each: its error term, the parameters it reads, whether a theorem evaluates it at a deployed record (the `Numbers*.lean` and `Bits.lean` files), at which sizes, and its owner. A table, sorted with the unevaluated ones first.
2. **The design** of one command (a `verity bench` runner or a `verity` subcommand; read `tools/verity/cli/` and `benchmarks/`) that prints, for each guarantee, its bits at each deployed point and at a named budget, from the Lean's own definitions where it can (an `#eval` or a `decide`-checked bound in a small package without Mathlib) and from a script that mirrors them where it cannot, with the mirror held to the Lean by a test.
3. **The check** that would have caught today: a guarantee whose statement has a probability bound and no evaluated row fails, with the row to add in its message. Assess it with `.agents/skills/process-design/SKILL.md` first (it runs on other people's work), and propose it; do not land a gate. The non-gating table can land.
4. How the goal "Every plotted number checked in Lean" (`proofs/spot-numbers-lean-checked`) relates: is this the same work, a superset, or separate?
5. A first version of the table itself, by script, for SPOT and C-Flock, from the other auditors' comments as they arrive (their issues are siblings of this one).

Code, if the coordinator says go: a new branch `cursor/proofs-security-table-<hex>` from `origin/main`, a PR opened unmarked with `research queue open`, `Lane proofs` on its first line.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **AUDIT-TABLE**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-539 --file F.md --worker AUDIT-TABLE` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
