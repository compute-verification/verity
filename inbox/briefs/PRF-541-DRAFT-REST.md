# PRF-541 DRAFT-REST

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker DRAFT-REST. Deadline: the table by Wed 11 Nov 03:00Z (Tue 7 PM PST); the resolutions by 06:00Z. First answer within 90 minutes of your start.**

**Direction.** Daniel: "this is our current draft … Should find differences and resolve them." Compare the rest of the draft with `main` and with what is in review.

**The draft:** `git fetch origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/attachments/spot-security-definitions.md` (fetched 23:16Z Tue 10 Nov from https://app.notion.com/p/SPOT-security-definitions-413399515d9e8162bbb7f8de48ca23e4 ; fetch it again through the Notion connector if you have it, and say if it changed).

**Cover:**
1. Section 3.6: Definition 3.21 (the two-stage design), Claim 3.22 (choosing interiors late), Protocol 2, Claim 3.23, against `Security/Definitions/Claims/SamplingTwo*.lean`, `Guarantees/SamplingTwo.lean`, and #8512 (`cursor/proofs-twomiss-ac23`: `KSRegTwo`, `MissTwo`, `HeadlineTwo`, the period bound).
2. Section 3.7, costs: every figure against a run or a theorem; name the run.
3. Section 4, applications, and Remark 4.3 (skippable work): against `Security/Proofs/WeightSecurity/` and the guarantees in `Security/Models/` (`Theft`, `TheftFaulted`, the period machinery; #8400 and #8527 are in review).
4. Appendix C (terminology and precedents) against DESIGN.md's Glossary: a term the draft and the Glossary use differently.
5. Appendix D, the status table, row by row against `main` at the SHA you read and against the open pull requests: "verified so far" and "still being worked on" each true today? The sentence "every other explicit bound is verified": list the explicit bounds and check each (the four checks of the lean-proofs skill for "proved").
6. The overview's claims (Section 1): "All of the math is machine-checked in Lean"; "up to a negligible term"; the $1\%$, 500 blocks, $95\%$ example; "about 19 times the sample".
7. Zero knowledge of Protocol 2 and the two-stage analogue of Theorem 3.15: what exists in Lean now (REGX's and TWOPROVE's branches) and what the table should say.

**Give one table of differences**, one row each: the draft's item (its number and sentence); the Lean's (name, file and line at the SHA you read, and whether it is on `main`, in review or absent); the difference in one line; which side is right and why; the resolution as exact text: the replacement sentence for the draft, or the Lean change with its size and whether it changes what a guarantee says or rests on (then it is Daniel's, through the coordinator). Rows with no difference are listed in one line at the end, so the coordinator can see what was compared. You edit neither the Notion page nor a locked statement: the coordinator applies the draft's changes as suggested edits and files the Lean's.

## How to work (the same for every worker of this swarm)

- You are a cloud worker of the lane **proofs**, named **DRAFT-REST**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-541 --file F.md --worker DRAFT-REST` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
