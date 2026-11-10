# PRF-530 COMMIT

proofs (lead), Tue 10 Nov 3:36 PM PST (23:36Z). **Worker COMMIT. Deadline: Wed 11 Nov 04:30Z (Tue 8:30 PM PST). First answer within 90 minutes of your start.**

**Direction.** Everything else in the swarm keeps the protocol and changes the proof or the hash. You change the protocol. Daniel: "Happy to get creative, walk back past decisions." Find commitments or protocol steps under which the committed trace is determined without rewinding the prover twice, or with a much smaller loss.

**Candidates to price, and add your own:**
1. A proximity test on the committed string: commit to an encoding of the trace and have the verifier test closeness to the code with live coins, as Ligero, Brakedown and FRI do. The committed trace is then the unique close codeword, not a plurality. What does the standard-model binding term become, and what does the encoding cost at $N=2^{44}$ to $2^{59}$?
2. Coarser committed units: whole blocks or rows as the positions, so $N$ becomes $n_{PU}$ (20 to 24 bits fewer at B and A). What does the base argument then open?
3. A proof of knowledge of the opening that is cheaper than the tree: an algebraic or homomorphic commitment for the sampled interface, or a second, small commitment made after the sample.
4. The auditor's own hardware: the challenger and the network certifiers are trusted devices on machines the prover does not control (AGENTS.md "Auditor-trusted devices"). Could one hold a key, so that the prover's commitment is authenticated by an information-theoretic MAC or a keyed universal hash the prover cannot evaluate? What would that device have to see, and does it break zero knowledge or the firewall?
5. Salting the trees with a live coin of the verifier's, so that no hashing done before the audit helps: does this let the budget $T$ be the hashing possible inside one audit's window rather than $2^{80}$?
6. Splitting one audit into many small commitments with their own samples: how do the terms add (Claim 3.19 sums the errors)?

For each, the card, and one paragraph on why it does or does not survive: the prover's cost at our sizes, zero knowledge, live coins, verifier cost in the descriptor's size.

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

- You are a cloud worker of the lane **proofs**, named **COMMIT**. This issue is your brief; PRF-523 is the shared context (the statement, the premise, the numbers, the script, what is in flight). Read it first, then `git fetch origin main && git show origin/main:AGENTS.md`, and `.agents/skills/smell-check/SKILL.md` and `.agents/skills/lean-proofs/SKILL.md` on `main`.
- **This is reading, deriving and computing, not building.** Read the Lean at `origin/main` (name the SHA). Do not build Lean or run a suite on your VM (it is refused, and one out-of-memory kill takes the VM down). A small Python computation is fine. If something must run, `research run --on auto --cwd source -- CMD` and say the run id.
- **Mark every fact:** [read] with file and line at the SHA you read; [derived] for your own math, with the steps shown so a second reader can check them; [literature] with the paper and the theorem's number, fetched and read (say "from memory" if you could not fetch it); [guess] for anything else. A number without one of these is refused.
- **Disagree with the context** where it is wrong. The arithmetic in it is the coordinator's, checked by one other reader.
- **Report, in this order:**
  1. Within 90 minutes of starting: one comment on this issue with your headline answer, even if rough. `research msg linear comment PRF-530 --file F.md --worker COMMIT` (no `--as`).
  2. A comment at least every 60 minutes after that while you work, one line if nothing changed.
  3. By your deadline: the final comment on this issue, and the full write-up as a note, `~/.research/notes/lanes/proofs/<YYYYMMDDTHHMMZ>-finding-<slug>.md` with the front-matter AGENTS.md names, published with `research notes sync --path lanes/proofs`.
  4. Your final message to the coordinator repeats the final comment in full. It is what the coordinator reads first.
- **Mailbox:** before each comment, `git fetch -q origin cursor/proofs-inbox && git show origin/cursor/proofs-inbox:inbox/bind100-swarm.md`. The coordinator puts Daniel's rulings and other workers' findings there.
- **Any other concrete-security smell** you meet on the way goes in your comment as a `SMELL` block (the smell-check skill's format), whoever owns it.
- You decide nothing about what a guarantee says or rests on, and you ask Daniel nothing: you price the options, and the coordinator carries them to him. You push to no branch but one you create (`cursor/proofs-<topic>-<hex>` from `origin/main`), and most of this needs none.
- Times come from `date -u`, read just before you write them. Never print a secret's value.
