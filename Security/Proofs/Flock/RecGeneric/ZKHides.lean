module

public import Proofs.Flock.RecGeneric.ZK
public import Proofs.Flock.Claims.Recursion.SeqPrefix

/-!
# Z2's proof: hm96 leaves hide

At hm96, `Leaves.commits` is `ZK.adaptiveLeaves` of the messages' digests under the masks `M·y` (`hm96_commits_eq`), so
`ZK.ideal_leaves_adaptive` is `Leaves.Hides` at the key's hiding gap, read in the claims core's `Pr`
(`ZKMal.Pr_le_of_prCoin`).

`Flock.SecurityProofs.Hm96Hides` is Z2.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims FlockSoundness.Discharge

/-- **hm96's commitments to adaptive messages are its adaptive leaves**: leaf `i` is
`enc(SHA-512(message i) ⊕ M·y_i, c(y_i))`, message `i` reading the leaves before it. -/
theorem hm96_commits_eq (enc : ZkView.Bits × ByteArray → Msg) {n : ℕ} (msgs : (i : Fin n) → (Fin i → Msg) → Msg)
    (y : Fin n → (Leaves.hm96 enc).Salt) :
    (Leaves.hm96 enc).commits msgs y =
      FlockSoundness.ZK.adaptiveLeaves enc (fun i prev => ZkView.bitsOf (Flock.Sha512.hash ⟨(msgs i prev).toArray⟩))
        (fun i => (ZkView.My (y i), ZkView.saltHash (y i))) :=
  FlockSoundness.ZK.adaptiveLeaves_unique _ _ _ _ fun i => by
    -- `simp only`, not `rfl`: unifying `(π i).1` with `ZkView.My (y i)` unfolds `My`'s 1536-term sum past `maxRecDepth`
    rw [Leaves.commits]
    simp only [Leaves.hm96]

end Claims.RecGeneric

open Claims Claims.RecGeneric FlockSoundness.Discharge in
/-- **Z2** (`Claims.RecGeneric.Hm96Hides`): `ZK.ideal_leaves_adaptive` at the digests of the messages. -/
theorem Flock.SecurityProofs.Hm96Hides : Claims.RecGeneric.Hm96Hides := by
  intro enc δ hT1 n Ω _ _ msgs G
  have h := Claims.ZKMal.Pr_le_of_prCoin (Claims.ZKMal.nat_mul_ne_top n δ)
    (FlockSoundness.ZK.ideal_leaves_adaptive ZkView.My ZkView.saltHash hT1 enc n
      (fun ω i prev => ZkView.bitsOf (Flock.Sha512.hash ⟨(msgs ω i prev).toArray⟩)) G)
  rw [Claims.ZKMal.toReal_nat_mul] at h
  simp only [hm96_commits_eq]
  exact h
