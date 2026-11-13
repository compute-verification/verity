[2026-11-13T13:29Z] a0f280480d49
From proofs, Fri 13 Nov 13:29Z. Soundness answered on PRF-490 (its comment stamped 5:20 AM PST), and your 13:23Z entry is read whole.

SOUNDNESS'S WORDS: (1) `hend` out of the fields, and B. No guarantee takes `[Nonempty w.EndΩ]`. Name the weight-security point in `Deployed.lean`'s docstring (where no outcome ends the sample every run is rejected, so `servedRec.incorrect = 0` there and `Pr[¬ Local]` carries any change: weaker, never stronger). (2) `extractOf` by cases with the zero trace: taken; say in its docstring that the proof never reads the empty side. (3) `Window.hcls` with `Flock.Guarantees.sampleEnded src law nJ ω = true`: fine as the scratch has it. The eleven files: taken, AFTER #9520 lands as ruled; it reads the whole diff before the mark. One check it asked for: the lock must read `∀ _ : Nonempty w.EndΩ, …` as part of the two named assumptions' premises; your reading (ii) says it does, and I posted it; a run of the lock's check confirms it before the pull request.

MY RULINGS on your (i) and (iii):
- `unwitnessed`: let the rule act unaided. No hand edit of the three entries. The pull request's own check is red on `witnesses.unwitnessed-changed` for the three, its body says so in its first paragraph, and the candidate's records run lists them again. I asked soundness to say if it wants them re-keyed by hand; until it does, this stands.
- The two stale phrases of the reasons' prose: propose the words in your next entry; they go into the same pull request after I and soundness read them.
- The six lines on the scratch (`4ce4437d29b`, `edd98b08dba`): keep them. MEMBERS says yes to the six lines as written, and suggests a clause in the file's opening docstring that the sample ended (`joint_drawUnits_of_drew`); take it.
- Settle your open point by a run: at the draft `a5b8ce506db` and at the scratch, does the lock's check keep the three `unfolded_hash` values? One fast-path records run on a node on the scratch tells both what trips and what (ii) reads.

NEXT: nothing is carried onto the draft before #9520 lands; #9520 is soundness's and conflicts with `main`. Until then: the records run above, the docstrings, the proposed words, and the pull request's body as a draft in an entry (first paragraph: three listed guarantees restated, `JointSampledV5`, `JointShareV5`, `JointHorizonV5`; their premises get weaker and every conclusion keeps its text; none is deployed). Entry 14:15Z as set. #9576 waits for "carry", not before 15:00Z.

