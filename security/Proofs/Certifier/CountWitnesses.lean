module

public import Proofs.Certifier.WitnessToys

/-! The sampled and ingress counts' honest and nontrivial lemmas, each at a positive instance; the ingress counts take `StatusComplete`, which no claim or verifier reads. -/

@[expose] public section

namespace NetTiming.Witness


/-- **`sampProg`'s audit meets the sampled count's premises and accepts its honest step**: `notLay` fits `sampProg`
over one step, its one free input admits two values, each possible incorrect set at `δ = 1/4` commits at most three bits,
the incorrect set `{0}` is possible there, and the step `obs1` is accepted. -/
theorem EncardAccSeqsSampledLe_honest :
    notLay.Fits sampProg 1 ∧ (AsReg sampProg notLay 1 ∅ fun _ => false).encard ≤ ((2 ^ 1 : ℕ) : ℕ∞) ∧
      (∀ F ∈ incorrectSets sampDesign sampPart (1 / 4), faultyBits sampProg sampPart F ≤ 3) ∧
      ({0} : Finset ℕ) ∈ incorrectSets sampDesign sampPart (1 / 4) ∧
      [obs1] ∈ AccSeqsSampled sampProg sampPart sampDesign notLay 1 (AsReg sampProg notLay 1 ∅ fun _ => false) (1 / 4)
        (fun _ _ => [false]) 1 :=
  ⟨sampLay_fits, sampAs_encard, sampBits, single_mem_sampFaulty, obs1_sampAcc⟩

/-- **`EncardAccSeqsSampledLe`'s bound fails at the verifier that accepts everything**, at `sampProg`'s parameters:
`c = 1`, `β = 3`, its possible incorrect sets at `δ = 1/4`, `D = 1`. -/
theorem EncardAccSeqsSampledLe_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤
      ((2 ^ 1 * (incorrectSets sampDesign sampPart (1 / 4)).card * 2 ^ 3 * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _


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
