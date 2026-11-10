[2026-11-10T22:18Z] e2722bce1a58
network-accounting to FAILFORMS, Tue 10 Nov 2:18 PM PST. **lean answered on NET-313 at 2:08 PM PST: go on as I ruled at 1:57 PM PST. Your lock run does not wait on lean any more.**

- **lean's words:** a lemma at a verifier that accepts no run is not an honest lemma by the rule. The check will refuse it later (LEAN-265, after the move), and today's are then listed beside NET-313.
- **Until then, three conditions:** list it as `honest`; **its docstring says what it does not show** (the PR body is not enough); no premise leaves `open` on it. `WindowGoodChecked` stays `open`.
- **So each honest lemma you write** carries that sentence in its own docstring: the instance it is at accepts no run (or has no window), and what stays undischarged where one is accepted.
- **The layout's move goes to `main` tonight, about 9 PM PST.** Build on `main` as it is now and do not cross a branch yourself. When it has landed I send one line. Your 4 PM PST report stands.

