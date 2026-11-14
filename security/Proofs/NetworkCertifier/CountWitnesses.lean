module

public import Proofs.NetworkCertifier.WitnessToys

/-!
# The sampled and ingress counts' honest and nontrivial lemmas

Each count the lock holds to witnesses has an honest and a nontrivial lemma, each at a positive instance: several
steps, a possible incorrect set that is not empty.

The count under a sampled audit (`EncardAccSeqsSampledLe`) takes that the layout fits the circuit. Its honest lemma
concludes that at `sampProg`, with the free inputs' count, the faulty-bits bound at `δ = 1/4`, the incorrect set `{0}`
possible there, and the acceptance of the step `obs1`; its nontrivial lemma is the same count at `acceptAll`, against
the bound's value at those parameters (`accSeqsAt_acceptAll_not_le`).

The six ingress counts take that every status is complete (`StatusComplete`, acceptance as it bears on ingress), which
no claim or verifier reads, so their entries name their kind. Their honest lemma concludes it for `inI`, one frame a
bucket on its one link, with W2, W4 at allocation `1` below the link's two slots, and J; their nontrivial lemmas decode
every message of `fxLeakI`, a sender whose secret is its frame's length, from its records or its wire, against two
content assignments where the bound has them.
-/

@[expose] public section

namespace NetTiming.Witness

/-! ## Under a sampled audit -/

/-- **`sampProg`'s audit meets the sampled count's premises and accepts its honest step**: `notLay` fits `sampProg`
over one step, its one free input admits two values, each possible incorrect set at `δ = 1/4` commits at most three bits,
the incorrect set `{0}` is possible there, and the step `obs1` is accepted. -/
theorem EncardAccSeqsSampledLe_honest :
    notLay.Fits sampProg 1 ∧ (AsReg sampProg notLay 1 ∅ fun _ => false).encard ≤ ((2 ^ 1 : ℕ) : ℕ∞) ∧
      (∀ F ∈ faultySets sampDesign sampPart (1 / 4), faultyBits sampProg sampPart F ≤ 3) ∧
      ({0} : Finset ℕ) ∈ faultySets sampDesign sampPart (1 / 4) ∧
      [obs1] ∈ AccSeqsSampled sampProg sampPart sampDesign notLay 1 (AsReg sampProg notLay 1 ∅ fun _ => false) (1 / 4)
        (fun _ _ => [false]) 1 :=
  ⟨sampLay_fits, sampAs_encard, sampBits, single_mem_sampFaulty, obs1_sampAcc⟩

/-- **`EncardAccSeqsSampledLe`'s bound fails at the verifier that accepts everything**, at `sampProg`'s parameters:
`c = 1`, `β = 3`, its possible incorrect sets at `δ = 1/4`, `D = 1`. -/
theorem EncardAccSeqsSampledLe_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤
      ((2 ^ 1 * (faultySets sampDesign sampPart (1 / 4)).card * 2 ^ 3 * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-! ## Into X -/

/-- **`inI`'s run has every status complete, and it meets W4, J and W2**: one frame a bucket on its one link, left
packed, at allocation `1`, jitter class `0`, the wire its record. -/
theorem Ingress_honest :
    StatusComplete 4 (inI.obsRun () 4) ∧ Packed inI.certifier ∧ Allocated inI.certifier (fun _ => 1) ∧
      JitterBounded inI.certifier 1 ∧ Regenerates inI.certifier (fun _ o => o) :=
  ⟨statusComplete_inI 4, fun _ _ _ _ => oneRow_packed, fun _ _ _ _ => oneRow_count, fun _ _ _ => Nat.one_pos,
    fun _ _ _ => rfl⟩

/-- **`EncardIngressObsDecodableLe`'s bound fails at the verifier that accepts everything**, at one frame a bucket on
one link over four steps: there a decoder of `fxLeakI`'s records recovers every message. -/
theorem EncardIngressObsDecodableLe_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 4) ∧ fxLeakDec (fxLeakI.obsRun z 4) = z}.encard ≤
      (((∏ _ℓ : Fin 1, (1 + 1)) ^ 4 * 1 ^ 4 : ℕ) : ℕ∞) :=
  fxLeakObs_not_le (by norm_num) _ (ENat.coe_ne_top _)

/-- **`EncardIngressDecodableLe`'s bound fails at the verifier that accepts everything**: a decoder of `fxLeakI`'s
wire recovers every message. -/
theorem EncardIngressDecodableLe_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 4) ∧ fxLeakDec (fxLeakI.wireRun z 4) = z}.encard ≤
      (((∏ _ℓ : Fin 1, (1 + 1)) ^ 4 * 1 ^ 4 : ℕ) : ℕ∞) :=
  fxLeak_not_le (by norm_num) _ (ENat.coe_ne_top _)

/-- **`EncardIngressObsDecodableLeContents`'s bound fails at the verifier that accepts everything**, at two content
assignments. -/
theorem EncardIngressObsDecodableLeContents_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 4) ∧ fxLeakDec (fxLeakI.obsRun z 4) = z}.encard ≤
      ({fun _ => [[true]], fun _ => [[false]]} : Set (Fin 1 → List (List Bool))).encard *
        (((∏ _ℓ : Fin 1, (1 + 1)) ^ 4 * 1 ^ 4 : ℕ) : ℕ∞) := by
  rw [twoContents_encard]
  exact fxLeakObs_not_le (by norm_num) _ (two_mul_ne_top _)

/-- **`EncardIngressDecodableLeContents`'s bound fails at the verifier that accepts everything**, at two content
assignments. -/
theorem EncardIngressDecodableLeContents_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 4) ∧ fxLeakDec (fxLeakI.wireRun z 4) = z}.encard ≤
      ({fun _ => [[true]], fun _ => [[false]]} : Set (Fin 1 → List (List Bool))).encard *
        (((∏ _ℓ : Fin 1, (1 + 1)) ^ 4 * 1 ^ 4 : ℕ) : ℕ∞) := by
  rw [twoContents_encard]
  exact fxLeak_not_le (by norm_num) _ (two_mul_ne_top _)

/-- **`EncardIngressObsDecodableLePerExecutionPhase`'s bound fails at the verifier that accepts everything**, over two windows
of two buckets. -/
theorem EncardIngressObsDecodableLePerExecutionPhase_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (2 * 2)) ∧ fxLeakDec (fxLeakI.obsRun z (2 * 2)) = z}.encard ≤
      ((((∏ _ℓ : Fin 1, (1 + 1) ^ 2) * 1 ^ 2) ^ 2 : ℕ) : ℕ∞) :=
  fxLeakObs_not_le (by norm_num) _ (ENat.coe_ne_top _)

/-- **`EncardIngressObsDecodableLeContentsPerExecutionPhase`'s bound fails at the verifier that accepts everything**, over two
windows of two buckets at two content assignments. -/
theorem EncardIngressObsDecodableLeContentsPerExecutionPhase_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (2 * 2)) ∧ fxLeakDec (fxLeakI.obsRun z (2 * 2)) = z}.encard ≤
      ({fun _ => [[true]], fun _ => [[false]]} : Set (Fin 1 → List (List Bool))).encard *
        ((((∏ _ℓ : Fin 1, (1 + 1) ^ 2) * 1 ^ 2) ^ 2 : ℕ) : ℕ∞) := by
  rw [twoContents_encard]
  exact fxLeakObs_not_le (by norm_num) _ (two_mul_ne_top _)

end NetTiming.Witness
