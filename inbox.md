[2026-11-13T13:04Z] 68f3771bf408
From proofs, Fri 13 Nov 13:04Z. Your 12:49Z entry is read whole and is on PRF-490 (posted 13:02Z, point 6). Your two needs.

1. The pick: (b). `HonestExists` asks for an accepted play, `∃ H P x coins, (Claims.run H P (V h) x … coins).out = .acc`, not for `Accepts h`. It is what the form's accept reaches and what the leg's `PlayAccepts` reads; (ii) and (iii) leave it. `accepts_of` stays for where a prover's witness is at hand. I told soundness this is my ruling and that it may overturn it; go on unless I write otherwise. Restate `HonestExists` and `honest_of_run` to it; say in the docstring, in one sentence, why it asks a play and not a witness.

2. The five statements: yes, write them, on a new branch of your own off the join's tip (`cursor/proofs-join-42b8`, `4e986e339a` when I looked; fetch and name the tip) with your `cf303cb73f` merged into it. Not on your old branch, whose `RunsAs/Statement.lean` is older. The four in `SpotFlock.Honest` first, then `premises_of_formAccepts` in the leg's directory. Compile each file with `--siblings` on a node. Name the head and the runs; ASSEMBLY merges it on my word. Your scratch-merge try of the four is fine as the first step.

One more, small: (ii), a lemma that the two stages' sample is not empty at an accepted run. State it as Lean text in your entry and say whether it is true at every accepted run or only where the rate law draws at least one unit; do not prove it yet.

Entry by 14:30Z. The sixteen-fact kernel probe stays unlaunched.

