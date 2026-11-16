module

public import Proofs.Certifier.WitnessToys

/-!
# The ingress count's honest and nontrivial lemmas

The ingress count takes that every status is complete (`StatusComplete`, acceptance as it bears on ingress), which no
claim or verifier reads, so its entry names its kind. Its honest lemma concludes it for `inI`, one frame a bucket on
its one link, with W2, W4 at allocation `1` below the link's two slots, and J; its nontrivial lemma decodes every
message of `fxLeakI`, a sender whose secret is its frame's length, from its records, against two content assignments.
-/

@[expose] public section

namespace NetTiming.Witness


/-! ## Into X -/

/-- **`inI`'s run has every status complete, and it meets W4, J and W2**: one frame a bucket on its one link, left
packed, at allocation `1`, jitter class `0`, the wire its record. -/
theorem Ingress_honest :
    StatusComplete 4 (inI.obsRun () 4) ∧ Packed inI.certifier ∧ Allocated inI.certifier (fun _ => 1) ∧
      JitterBounded inI.certifier 1 ∧ Regenerates inI.certifier (fun _ o => o) :=
  ⟨statusComplete_inI 4, fun _ _ _ _ => oneRow_packed, fun _ _ _ _ => oneRow_count, fun _ _ _ => Nat.one_pos,
    fun _ _ _ => rfl⟩

/-- **`EncardIngressObsDecodableLeContents`'s bound fails at the verifier that accepts everything**, at two content
assignments. -/
theorem EncardIngressObsDecodableLeContents_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 4) ∧ fxLeakDec (fxLeakI.obsRun z 4) = z}.encard ≤
      ({fun _ => [[true]], fun _ => [[false]]} : Set (Fin 1 → List (List Bool))).encard *
        (((∏ _ℓ : Fin 1, (1 + 1)) ^ 4 * 1 ^ 4 : ℕ) : ℕ∞) := by
  rw [twoContents_encard]
  exact fxLeakObs_not_le (by norm_num) _ (two_mul_ne_top _)

end NetTiming.Witness
