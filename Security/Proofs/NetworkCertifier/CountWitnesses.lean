import Proofs.NetworkCertifier.WitnessToys

/-!
# The egress and ingress counts' honest and nontrivial lemmas

Each count the lock holds to witnesses has an honest and a nontrivial lemma, each at a positive instance: four steps,
two execution blocks of one advice bit each, a possible faulty set that is not empty.

The four egress counts over `AccSeqs` (the fixed advice budget and the reference schedule's three) take no premise
outside a named assumption's module, so their honest lemmas say that the deployed Program accepts the honest
certifier's run: at the constant-rate Program over two execution blocks of two buckets (`fxRate`), the honest
certifier's four steps are among the counted sequences, and from the first block's start (`e₀ = 0`) they continue the
empty prefix. Their nontrivial lemmas say that each bound fails at the verifier that accepts everything: there, one
link of two slots carries any payload at every step, so the sequences counted are infinitely many
(`accSeqsAt_acceptAll_not_le`), against the bound's value at `fxRate`'s parameters.

The count under a sampled audit (`EncardAccSeqsSampledLe`) takes that the layout fits the circuit. Its honest lemma
concludes that at `sampProg`, with the free inputs' count, the faulty-bits bound at `δ = 1/4`, the faulty set `{0}`
possible there, and the acceptance of the step `obs1`; its nontrivial lemma is the same count at `acceptAll`, against
the bound's value at those parameters.

The two counts through the wire (`EncardDecodableLeConstantRate` and its per-execution-block form) take the run's
acceptance, which their honest lemmas conclude for the honest certifier `fxHonestI` at `fxRate`, with W2 and J, its two
named assumptions, holding there too; the per-execution-block form's, from the first block's start, where every run's
prefix is empty. At the verifier that accepts everything a sender whose secret is its frame's length (`fxLeakI`) has
every secret decoded (`fxLeak_not_le`).

The six ingress counts take that every status is complete (`StatusComplete`, acceptance as it bears on ingress), which
no claim or verifier reads, so their entries name their kind. Their honest lemma concludes it for `inI`, one frame a
bucket on its one link, with W2, W4 at allocation `1` below the link's two slots, and J; their nontrivial lemmas decode
every message of `fxLeakI` from its records or its wire, against two content assignments where the bound has them.
-/

namespace NetTiming.Witness

/-! ## Egress -/

/-- **The reference schedule accepts the honest certifier's run**: four steps, both execution blocks of `fxRate`, under
the advice `fxAdv`. -/
theorem ConstantRate_honest :
    (honestI (K := 1) fxRate fxAdv).obsRun () 4 ∈ AccSeqs fxRate (unitOp (K := 1)) 4 1 :=
  honestI_mem fxRate fxAdv 4

/-- **From the first execution block's start, the reference schedule accepts the honest certifier's run**: at
`e₀ = 0`, `m = 2`, `fxRate` accepts its two blocks under `fxAdv`, and the run continues the empty prefix. -/
theorem ConstantRatePerWindow_honest :
    (honestI (K := 1) fxRate fxAdv).obsRun () ((0 + 2) * (1 * 2)) ∈
        AccSeqs fxRate (unitOp (K := 1)) ((0 + 2) * (1 * 2)) 1 ∧
      ((honestI (K := 1) fxRate fxAdv).obsRun () ((0 + 2) * (1 * 2))).take (0 * (1 * 2)) = [] :=
  ⟨honestI_mem fxRate fxAdv _, by first | exact List.take_zero | rfl | (simp; done)⟩

/-- **`EncardAccSeqsLeAdviceBits`'s bound fails at the verifier that accepts everything**, at `fxRate`'s parameters:
two execution blocks of one advice bit each, four steps. -/
theorem EncardAccSeqsLeAdviceBits_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤ ((2 ^ (2 * 1) * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeConstantRate`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters: `E = 2` execution blocks, `k = 1`, four steps. -/
theorem EncardAccSeqsLeConstantRate_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤ ((2 ^ (2 * 1) * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeConstantRateWindows`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters with four steps within its first `m = 2` execution blocks. -/
theorem EncardAccSeqsLeConstantRateWindows_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤ ((2 ^ (min 2 2 * 1) * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeConstantRatePerWindow`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters from the first execution block's start (`e₀ = 0`, the prefix `[]`) over its `m = 2` blocks. -/
theorem EncardAccSeqsLeConstantRatePerWindow_nontrivial :
    ¬ {o | o ∈ AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) ((0 + 2) * (1 * 2)) 1 ∧
        o.take (0 * (1 * 2)) = []}.encard ≤ (((2 ^ 1 * 1 ^ (1 * 2)) ^ 2 : ℕ) : ℕ∞) := by
  have h : {o | o ∈ AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) ((0 + 2) * (1 * 2)) 1 ∧
      o.take (0 * (1 * 2)) = []} = AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) ((0 + 2) * (1 * 2)) 1 := by
    ext o
    first
      | exact ⟨fun h => h.1, fun h => ⟨h, List.take_zero⟩⟩
      | (simp; done)
  rw [h]
  exact accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-! ## Under a sampled audit -/

/-- **`sampProg`'s audit meets the sampled count's premises and accepts its honest step**: `notLay` fits `sampProg`
over one step, its one free input admits two values, each possible faulty set at `δ = 1/4` commits at most three bits,
the faulty set `{0}` is possible there, and the step `obs1` is accepted. -/
theorem EncardAccSeqsSampledLe_honest :
    notLay.Fits sampProg 1 ∧ (AsReg sampProg notLay 1 ∅ fun _ => false).encard ≤ ((2 ^ 1 : ℕ) : ℕ∞) ∧
      (∀ F ∈ faultySets sampDesign sampPart (1 / 4), faultyBits sampProg sampPart F ≤ 3) ∧
      ({0} : Finset ℕ) ∈ faultySets sampDesign sampPart (1 / 4) ∧
      [obs1] ∈ AccSeqsSampled sampProg sampPart sampDesign notLay 1 (AsReg sampProg notLay 1 ∅ fun _ => false) (1 / 4)
        (fun _ _ => [false]) 1 :=
  ⟨sampLay_fits, sampAs_encard, sampBits, single_mem_sampFaulty, obs1_sampAcc⟩

/-- **`EncardAccSeqsSampledLe`'s bound fails at the verifier that accepts everything**, at `sampProg`'s parameters:
`c = 1`, `β = 3`, its possible faulty sets at `δ = 1/4`, `D = 1`. -/
theorem EncardAccSeqsSampledLe_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤
      ((2 ^ 1 * (faultySets sampDesign sampPart (1 / 4)).card * 2 ^ 3 * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-! ## Through the wire -/

/-- **The reference schedule accepts the honest certifier's run, which meets W2 and J**: `fxHonestI`'s four steps are
accepted at `fxRate`, its wire is its record, and its jitter class is below `D = 1`. -/
theorem DecodableConstantRate_honest :
    AcceptedUpTo fxRate fxHonestI.op 4 (fxHonestI.obsRun () 4) ∧ Regenerates fxHonestI.certifier (fun _ o => o) ∧
      JitterBounded fxHonestI.certifier 1 :=
  ⟨⟨fxAdv, honestI_accepted fxRate fxAdv 4⟩, fun _ _ _ => rfl, fun _ _ _ => Nat.one_pos⟩

/-- **From the first execution block's start, the reference schedule accepts the honest certifier's run, which meets
W2 and J**: at `e₀ = 0` the run's prefix is `[]`, `fxHonestI`'s `m = 2` blocks are accepted at `fxRate`, its wire is
its record, and its jitter class is below `D = 1`. -/
theorem DecodableConstantRatePerWindow_honest :
    fxHonestI.obsRun () (0 * (1 * 2)) = [] ∧
      AcceptedUpTo fxRate fxHonestI.op ((0 + 2) * (1 * 2)) (fxHonestI.obsRun () ((0 + 2) * (1 * 2))) ∧
      Regenerates fxHonestI.certifier (fun _ o => o) ∧ JitterBounded fxHonestI.certifier 1 :=
  ⟨List.eq_nil_of_length_eq_zero (Interaction.length_obsRun _ () _), ⟨fxAdv, honestI_accepted fxRate fxAdv _⟩,
    fun _ _ _ => rfl, fun _ _ _ => Nat.one_pos⟩

/-- **`EncardDecodableLeConstantRate`'s bound fails at the verifier that accepts everything**: there `fxLeakI`'s
decoder recovers every secret, against the bound's value at `fxRate`'s parameters within its first `m = 2` execution
blocks. -/
theorem EncardDecodableLeConstantRate_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 4) ∧ fxLeakDec (fxLeakI.wireRun z 4) = z}.encard ≤
      ((2 ^ (min 2 2 * 1) * 1 ^ 4 : ℕ) : ℕ∞) :=
  fxLeak_not_le (by norm_num) _ (ENat.coe_ne_top _)

/-- **`EncardDecodableLeConstantRatePerWindow`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters from the first execution block's start (`e₀ = 0`, where every secret's prefix is `[]`) over its `m = 2`
blocks. -/
theorem EncardDecodableLeConstantRatePerWindow_nontrivial :
    ¬ {z : ℕ | fxLeakI.obsRun z (0 * (1 * 2)) = [] ∧ acceptAll (fxLeakI.obsRun z ((0 + 2) * (1 * 2))) ∧
        fxLeakDec (fxLeakI.wireRun z ((0 + 2) * (1 * 2))) = z}.encard ≤ (((2 ^ 1 * 1 ^ (1 * 2)) ^ 2 : ℕ) : ℕ∞) := by
  have h : {z : ℕ | fxLeakI.obsRun z (0 * (1 * 2)) = [] ∧ acceptAll (fxLeakI.obsRun z ((0 + 2) * (1 * 2))) ∧
      fxLeakDec (fxLeakI.wireRun z ((0 + 2) * (1 * 2))) = z} =
      {z : ℕ | acceptAll (fxLeakI.obsRun z ((0 + 2) * (1 * 2))) ∧
        fxLeakDec (fxLeakI.wireRun z ((0 + 2) * (1 * 2))) = z} :=
    Set.ext fun z => ⟨fun h => h.2, fun h => ⟨List.eq_nil_of_length_eq_zero (Interaction.length_obsRun _ z _), h⟩⟩
  rw [h]
  exact fxLeak_not_le (by norm_num) _ (ENat.coe_ne_top _)

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

/-- **`EncardIngressObsDecodableLePerWindow`'s bound fails at the verifier that accepts everything**, over two windows
of two buckets. -/
theorem EncardIngressObsDecodableLePerWindow_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (2 * 2)) ∧ fxLeakDec (fxLeakI.obsRun z (2 * 2)) = z}.encard ≤
      ((((∏ _ℓ : Fin 1, (1 + 1) ^ 2) * 1 ^ 2) ^ 2 : ℕ) : ℕ∞) :=
  fxLeakObs_not_le (by norm_num) _ (ENat.coe_ne_top _)

/-- **`EncardIngressObsDecodableLeContentsPerWindow`'s bound fails at the verifier that accepts everything**, over two
windows of two buckets at two content assignments. -/
theorem EncardIngressObsDecodableLeContentsPerWindow_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (2 * 2)) ∧ fxLeakDec (fxLeakI.obsRun z (2 * 2)) = z}.encard ≤
      ({fun _ => [[true]], fun _ => [[false]]} : Set (Fin 1 → List (List Bool))).encard *
        ((((∏ _ℓ : Fin 1, (1 + 1) ^ 2) * 1 ^ 2) ^ 2 : ℕ) : ℕ∞) := by
  rw [twoContents_encard]
  exact fxLeakObs_not_le (by norm_num) _ (two_mul_ne_top _)

end NetTiming.Witness
