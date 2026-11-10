# PRF-537 AUDIT-RESOURCE

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker AUDIT-RESOURCE. Deadline: Wed 11 Nov 06:00Z (Tue 10 PM PST). First answer within 90 minutes of your start.**

**Direction.** PoUW, PoUS, the bandwidth audit, the remote bandwidth audit, the network certifier, NCI and the joint accounting guarantee each state an error. Evaluate each at the parameters the demos run. These guarantees belong to other leads (compute-accounting, memory-accounting, network-accounting, soundness); you read them and report, and the coordinator sends each finding to its owner.

**Cover, for each protocol:**
1. Its guarantee statements in `Security/Models/<Area>/` and the locks (`Security/lean-audit.json`, `Security/Proofs/lean-audit.json`, and each package's own): every probability in them, and what "$\varepsilon$ at $\delta$" the draw establishes at the sample sizes actually used (AGENTS.md: "It surfaces what its draw established").
2. The number at the deployed or demo parameters, and whether any Lean theorem or any test states that number.
3. Every place a guarantee inherits SPOT's or C-Flock's error (the sampled proofs under PoUW and the certifier): does it inherit the binding term, and is its own quoted figure still true with $2^{-33.6}$ in it?
4. Union bounds over time: windows, epochs, links, machines. Is each stated for one window or for a period, and what does a year do?
5. PoUW's grinding term ($\mu^m$ a try, AGENTS.md "Live coins only"): the deployed $\mu$ and $m$, and the number of tries $2^{80}$ hash calls allow.
6. Hardware constants read as security parameters (f_max, peak rates from the data sheet): where a bound's size rests on one.

**Give each finding as a `SMELL` block** (the smell-check skill's format), then one table of everything you evaluated: the guarantee, the term, its formula, the deployed parameters you put in and where each comes from, the number, and whether a Lean theorem on `main` states that number or only the formula. "None found" needs the list of what you read. A term nobody has evaluated at deployed sizes is a finding even when the number turns out fine.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **AUDIT-RESOURCE**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-537 --file F.md --worker AUDIT-RESOURCE` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
