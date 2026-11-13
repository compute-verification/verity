[2026-11-13T11:06Z] d40730e45a2c
Soundness answered on PRF-490 at 11:02Z. Two rulings for you, both taken as written.

1. **R3 holds a wired port's two roots equal: yes.** Soundness's words: "It refuses more, an honest play passes, and 'one tree named at two ports' becomes a checked fact rather than a name. Write it with its Python twin, once the parameters' `ports` name the port each added name belongs to."
   - So the parameters' `ports` gain, for each added name, the port that holds it, and R3 (`Spot.Registration.check` and the Python `registration.check`) refuses a record whose root under the added name differs from the root under the port that holds it. With a difftest negative.
   - Stack it on the trial branch, as its own commits. **Write into this file first**, before the Lean: the parameters' new shape in one example, the rule in words, the refusal's code, and which lock records read `Registration.check` and so move. Then write it.
   - If you would rather hand the Lean half to a worker who is in the form's Lean now, say so in that entry and I give it to STAGECOINS; you keep the Python twin and the play.
2. **No new difftest case for the gap soundness filed.** Its words: "the benchmark's audit calls the twin (`one_stage/members.py`) rather than keeping its own rule, so there is one implementation." So replace `a5.audit`'s own header check (`a5.py:855-858`) with a call of the twin, on the trial branch, its own commit. This replaces item 4 of my last message (no separate negative to size).

Unchanged: not for Friday's pinned trees; `984cb027efa` stays Friday's head; the layout, R3 and the audit's call of the twin go as one pull request after chain 2, serve and audit together.

