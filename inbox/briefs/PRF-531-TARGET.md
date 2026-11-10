# PRF-531 TARGET

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker TARGET. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** Daniel asked for "100 bits of security". Pin down what that must mean for SPOT so every option is measured against one target, and so the draft can say it.

**Answer:**
1. The definitions in use: an error of at most $2^{-100}$ at a fixed budget $T$; every attacker of time $T$ succeeds with probability at most $T/2^{100}$ (the ratio); Micciancio and Walter's bit security; "provable" against "conjectured" security as the STARK papers state it (ethSTARK, the FRI soundness papers), Ligero, Brakedown, Binius, and what the systems we compare ourselves to claim. Under each, what is today's number at the floor, B and A? (At $T=2^{80}$ today's bound is $2^{-33.6}$, yet the ratio $T/\varepsilon$ is $2^{113}$ there and about $2^{74}$ for small $T$: say what that means and whether it misleads.)
2. The budget. What is counted in $T$ on `main` (`BconfAt`'s docstring on `cursor/proofs-bconf-cost-5987`: both parties' calls in every run)? Is $2^{80}$ the right budget for an interactive audit whose commitment must be opened inside a window? Does hashing done before the audit count, and would a live salt in the tree remove it?
3. The scale. Claim 3.19 sums the per-audit errors. The world of the post is $1.78\times10^{16}$ requests a year on $46.87$ million machines. If an audit is a request, or a link-window, how many audits are there in a period and in a year, and what per-audit error makes the year's sum $2^{-40}$, $2^{-64}$? Where do network-accounting's targets ($2^{-61.0}$ a class audit, $2^{-53.4}$ an audit; PRF-456) come from, and are they consistent with "100 bits"?
4. What an undetected binding failure buys the prover: one audit's worth of unsampled incorrect blocks, or more? This sets how small the number has to be.
5. Your recommendation: the one sentence the draft should carry as the target, the budget, and the level (audit, period, year) at which it is stated. It goes to Daniel as a recommendation, not a decision.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **TARGET**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-531 --file F.md --worker TARGET` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
