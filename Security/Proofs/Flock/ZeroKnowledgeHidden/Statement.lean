import Proofs.Flock.Soundness.Discharge.ZkHidden.View

/-!
# C-Flock's one-stage audit under hidden outputs is zero knowledge (trusted)

`ZeroKnowledgeHidden` is the zero-knowledge half of `EndToEndHidden`, on the same plan: the verifier's joint view of the
`J` tables of a hidden-output `--zk` session is simulatable. Its proof is
`Flock.SecurityProofs.ZeroKnowledgeHidden : Flock.Guarantees.ZeroKnowledgeHidden`.

C-Flock's definitions are still in its proofs package, so this module imports
`Proofs.Flock.Soundness.Discharge.ZkHidden.View`, and `Models.lean`, which the `security` package builds without
`security_proofs`, does not import it: `Proofs.Flock.ZeroKnowledgeHidden` does.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge.Exec FlockSoundness.Refine FlockSoundness.Discharge.ZkExec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.DecodeZL
open scoped ENNReal

/-- **C-Flock's one-stage audit under hidden outputs is zero knowledge, at the coins the prover doesn't refuse, on runs
its self-check doesn't stop.** At
every draw `S` at which `verify --zk` sets every one of the session's `J = nTab I` tables up (`ys`), and every
registration `R`, the joint view of the session's plan `planZHJ dj y₀` (the plan `EndToEndHidden` is sound on), table
`j` at its executable statement, is within `Σ_j 2·|Hid_j|·δ_{K*}` of the per-table simulators' joint view, both ways
(`ZkViewAtJ`): at any coin vector per table that the honest prover doesn't refuse (`CoinedSetup.Admits`), for any
encoding of the joint view and any event on it, the real view at valid witnesses and the simulators' at any
witnesses, each table's prover randomness independent. `|Hid_j|` is the number of
table `j`'s leaves the final message doesn't open, and the prover `P` is any prover of Ligerito's levels ≥ 1
(`LigProver`).

The honest prover also stops when its own batched constraint fails (`zk_veil.rs`'s `SelfCheck::Stop`, when
`⟨c, h⟩ + t ≠ 0`, after the batching coins and before `τ`'s commitment), and that stop depends on the witness, not only
on the coins. The real view here never stops there, so this is a statement about the runs where it doesn't fire. At a
valid witness and admitted coins it never fires if the clear protocol is complete (`ZK/Complete.lean`'s `InnerHolds`,
which `inner_complete` takes as a hypothesis), and nothing here proves that.

Which verifiers: honest-verifier zero knowledge at each fixed coin vector, a zerocheck outer coin `1` among the refused.
It covers a verifier that chooses its coins because the coin server commits every coin at `Hello`, before the prover's
first message, under `keyed-cr`, which no statement here models yet.

`δ_{K*}` is `hidingGap My saltHash`, the commitments' hiding gap at hm96-sha512's pinned key `K*` (`ZkView.My`,
`ZkView.saltHash`): the statistical distance of `(M(K*)·y, H(sp ‖ y))` from `(U, H(sp ‖ y))` for a uniform salt `y`.
Nothing here proves it small for `K*`: it is below `2^-193` for all but `2^-64` of keys (`CROnly.hm96_sha512_tail`), and
that `K*` is not one of the others is unproved. No premise is taken on the key or on SHA-512.

One hypothesis no check implies yet: every table's statement avoids the mask slot (`avoidsMask`), until the
verifier's D11 refusal in `Net.ofRows` discharges it. The rest are an untyped statement and `EndToEndHidden`'s
reference setup `y₀`: table 0's, at `mPts` link points, at some draw `d₀`, which every draw flock-verify accepts gives
(`Flock.SecurityProofs.EndToEndHidden_refSetup`). -/
def ZeroKnowledgeHidden : Prop :=
  ∀ (n : ℕ) (Reg : Type) (P : LigProver)
    -- the session's statement, the record's draw file, and the reference setup
    (I : Inputs) (_ht : I.tags.typed = false)
    (mPts : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    -- every table's statement avoids the mask slot
    (_hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    -- a draw at which `verify --zk` sets every table up, and a registration
    (S : Finset (Fin n)) (R : Reg)
    (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
    ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
      (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st)
      (ENNReal.ofReal (CROnly.hidingGap FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash))

end Flock.Guarantees
