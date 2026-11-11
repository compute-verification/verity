[2026-11-11T22:14Z] 6c9bfddd6252
LEANMERGE, from network-accounting's coordinator, Wed 11 Nov 2:14 PM PST. STOP job 5 where it is (push what you have; say its state in one line) and do this first. It blocks lean's lock run on LEAN-249.

**The failure** [lean's worker MAINMERGE on LEAN-249, 1:53 PM PST; the two lines below read by me at the crossed head `3ff11d118f93`]
- `security/Proofs` does not build at the crossed head. Four modules fail beside the `Causal.lean` you mended. One is ours: `security/Proofs/NetworkCertifier/WitnessToys.lean`, "261:43: Type mismatch": `[]` "has type `List Bool` but is expected to have type `List (Obs 1 ?m.8 1) → List Bool`".
- Line 261, in `pooled_bound_fails`, hands `PooledGood` an `op` of three arguments: `(fun _ _ _ => ([] : List Bool))`. `main`'s `PooledGood` (`security/Definitions/NetworkCertifier/PooledLeak.lean:71-73`) takes `op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool`, four. [read by me]
- Behind it and not built: `NetworkCertifier/CountWitnesses`, `NetworkCertifier/PeriodWitnesses` and the root `Proofs.NetworkCertifier`. Whether they build once this one does is not known.

**The job, as for `Causal.lean`**
1. One commit on a branch of ours cut from the crossed tip as it is when you start (fetch it; say the sha): `cursor/na-cross-mend-witnesstoys-<4 hex>`. Proofs and witnesses only: no file under `security/Definitions`, no guarantee's statement, no `--update`, no lock.
2. Mend `WitnessToys.lean`. The theorem must still say what its docstring says about `main`'s `PooledGood`: the pooled bound fails at an audit that always accepts. If the fourth argument (the history of observations) changes what the witness shows, stop and tell me rather than weaken it. Look for the same three-argument `op` elsewhere in the file and in the two modules behind it.
3. Try the file (`lean_changed.py --try … --siblings`), then `CountWitnesses.lean`, `PeriodWitnesses.lean` and the root `security/Proofs/NetworkCertifier.lean`, each `--on auto`. Mend what fails in those for the same cause in the same commit, each listed; anything of another cause is listed and left.
4. Comment on LEAN-249 with `--worker LEANMERGE` when it is pushed, as you did at 1:18 PM PST: the sha, its parent, the files, what changed, each try's run id and its last lines, what is not built. Read LEAN-249's newest comments in that minute first. I ring lean myself.

**Times:** the sha and the tries by 3:00 PM PST, by inbox, even if a try has no verdict. Job 5's first word moves to 4:15 PM PST.

Rules as before: masking on every read (lean's comments name a node: never repeat it), `--on auto` only, PIDs only, the clock from `TZ=America/Los_Angeles date`, [ran]/[read]/[guess] on each claim.

