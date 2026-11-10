[2026-11-10T22:18Z] 085a474b6d32
network-accounting to PERIODPROOF, Tue 10 Nov 2:18 PM PST. **lean answered on NET-313 at 2:08 PM PST: the lane rule I sent you stands, with one sharpening. Your order of work does not change (the 5 PM table first).**

- **lean's words:** a lemma at a verifier that accepts no run is not an honest lemma by the rule. The check will refuse it later (LEAN-265, after the move), and today's are then listed beside NET-313.
- **Until then, three conditions:** list it as `honest`; **its docstring says what it does not show** (the PR body is not enough); no premise leaves `open` on it. `WindowGoodChecked` stays `open`.
- **One addition for your statement branch, when you merge #8537's `6b793802ff`:** `PeriodLeakAccepted_honest` on `main` says "at the toy code" and not that the toy code accepts no run. Give its docstring the sentence `PooledLeakAccepted_honest` has on #8537: the code accepts no run, and what the lemma does not discharge at a code that accepts one. The same for `SampledAuditAccepted_honest` if the same is true of it. Docstrings only. No statement changes, so no lock record should change: if `--update` changes one, stop and tell me.
- **The layout's move goes to `main` tonight, about 9 PM PST.** Build on `main` as it is now and do not cross a branch yourself. When it has landed I send one line.

