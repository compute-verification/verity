import Proofs.Flock.Soundness.Discharge.ZkHidden.View

/-!
# C-Flock's one-stage audit under hidden outputs is zero knowledge (trusted)

`ZeroKnowledgeHidden` is the zero-knowledge half of `EndToEndHidden`, on the same plan: the verifier's joint view of the
`J` tables of a hidden-output `--zk` session is simulatable. Its proof is
`Flock.SecurityProofs.ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden`.

C-Flock's definitions are still in its proofs package, so this module imports
`Proofs.Flock.Soundness.Discharge.ZkHidden.View`, and `Specs.lean`, which the `security` package builds without
`security_proofs`, does not import it: `Proofs.Flock.ZeroKnowledgeHidden` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge.Exec FlockSoundness.Refine FlockSoundness.Discharge.ZkExec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.DecodeZL
open scoped ENNReal

/-- **C-Flock's one-stage audit under hidden outputs is zero knowledge, at the coins the prover doesn't refuse.** At
every draw `S` at which `verify --zk` sets every one of the session's `J = nTab I` tables up (`ys`), and every
registration `R`, the joint view of the session's plan `planZHJ dj y₀` (the plan `EndToEndHidden` is sound on), table
`j` at its executable statement, is within `Σ_j 2·|Hid_j|·2^-193` of the per-table simulators' joint view, both ways
(`ZkViewAtJ`): at any coin vector per table that the honest prover doesn't refuse (`CoinedSetup.Admits`), for any
encoding of the joint view and any event on it, the real view at valid witnesses whose registered values are the public
ones, and the simulators' at any witnesses, each table's prover randomness independent. `|Hid_j|` is the number of
table `j`'s leaves the final message doesn't open, and the prover `P` is any prover of Ligerito's levels ≥ 1
(`LigProver`).

Which verifiers: honest-verifier zero knowledge at each fixed coin vector, a zerocheck outer coin `1` among the refused.
It covers a verifier that chooses its coins because the coin server commits every coin at `Hello`, before the prover's
first message, under `keyed-cr`, which no statement here models yet.

Its named assumption, a hypothesis:
* `hash-derived-key` (`Assumptions.Zk.HashDerivedKeyHm96`): the pinned hm96 key hides at `2^-193`.

And one hypothesis no check implies yet: every table's statement avoids the mask slot (`avoidsMask`), until the
verifier's D11 refusal in `Net.ofRows` discharges it. The rest is an untyped statement. -/
def ZeroKnowledgeHidden : Prop :=
  ∀ (n : ℕ) (Reg : Type) (P : LigProver)
    -- the session's statement, and the record's draw file and setup at registration
    (I : Inputs) (_ht : I.tags.typed = false)
    (mPts : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    -- every table's statement avoids the mask slot
    (_hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    -- hash-derived-key
    (_hKey : Assumptions.Zk.HashDerivedKeyHm96)
    -- a draw at which `verify --zk` sets every table up, and a registration
    (S : Finset (Fin n)) (R : Reg)
    (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
    ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
      (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ))

end Flock.Guarantees
