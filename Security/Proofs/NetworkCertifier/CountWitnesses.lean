import Proofs.NetworkCertifier.WitnessToys

/-!
# The egress and ingress counts' honest and nontrivial lemmas

Each count the lock holds to witnesses has an honest and a nontrivial lemma, each at a positive instance: four steps,
the content advice `Bool`, a sync set of two values, a possible faulty set that is not empty.

The six egress counts over `AccSeqs` take no premise outside a named assumption's module, so their honest lemmas say
that the deployed Program accepts the honest certifier's run: at the constant-rate Program over two windows of two
buckets (`fxRate`, and `fxFixed` with its clock fixed), the honest certifier's four steps are among the counted
sequences. `EncardAccSeqsLeConstantRateFixedClock` takes a sync set of one value, so its honest lemma is at `fxRate1`,
over `syncOne`. Their nontrivial lemmas say that each bound fails at the verifier that accepts everything: there, one
link of two slots carries any payload at every step, so the sequences counted are infinitely many
(`accSeqsAt_acceptAll_not_le`), against the bound's value at `fxRate`'s parameters.

The count under a sampled audit (`EncardAccSeqsSampledLe`) takes that the layout fits the circuit. Its honest lemma
concludes that at `sampProg`, with the free inputs' count, the faulty-bits bound at `δ = 1/4`, the faulty set `{0}`
possible there, and the acceptance of the step `obs1`; its nontrivial lemma is the same count at `acceptAll`, against
the bound's value at those parameters.

The two counts through the wire (`EncardDecodableLeConstantRate` and its per-window form) take the run's acceptance,
which their honest lemma concludes for the honest certifier `fxHonestI` at `fxRate`, with W2 and J, its two named
assumptions, holding there too. At the verifier that accepts everything a sender whose secret is its frame's length
(`fxLeakI`) has every secret decoded (`fxLeak_not_le`).

The six ingress counts take that every status is complete (`StatusComplete`, acceptance as it bears on ingress), which
no claim or verifier reads, so their entries name their kind. Their honest lemma concludes it for `inI`, one frame a
bucket on its one link, with W2, W4 at allocation `1` below the link's two slots, and J; their nontrivial lemmas decode
every message of `fxLeakI` from its records or its wire, against two content assignments where the bound has them.
-/

namespace NetTiming.Witness

/-! ## Egress -/

/-- **The shipped schedule accepts the honest certifier's run**: four steps of `fxRate` under the advice `fxAdv`. -/
theorem ConstantRate_honest :
    (honestI (K := 1) fxRate fxAdv).obsRun () 4 ∈ AccSeqs fxRate (unitOp (K := 1)) 4 1 :=
  honestI_mem fxRate fxAdv 4

/-- **With its clock fixed, the shipped schedule accepts the honest certifier's run.** -/
theorem FixedClock_honest :
    (honestI (K := 1) fxFixed fxAdvU).obsRun () 4 ∈ AccSeqs fxFixed (unitOp (K := 1)) 4 1 :=
  honestI_mem fxFixed fxAdvU 4

/-- **At a sync set of one value, the shipped schedule accepts the honest certifier's run**: `syncOne` has one value,
and `fxRate1` accepts four steps under `fxAdv1`. -/
theorem ConstantRateFixedClock_honest :
    syncOne.size = 1 ∧ (honestI (K := 1) fxRate1 fxAdv1).obsRun () 4 ∈ AccSeqs fxRate1 (unitOp (K := 1)) 4 1 :=
  ⟨by simp [SyncSet.size, syncOne], honestI_mem fxRate1 fxAdv1 4⟩

/-- **`EncardAccSeqsLeSync`'s bound fails at the verifier that accepts everything**, at `fxRate`'s parameters. -/
theorem EncardAccSeqsLeSync_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤
      ((Fintype.card Bool * Fintype.card syncTwo.Elem ^ 2 * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeFixedClock`'s bound fails at the verifier that accepts everything**, at `fxFixed`'s parameters. -/
theorem EncardAccSeqsLeFixedClock_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤ ((Fintype.card Bool * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeConstantRate`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters. -/
theorem EncardAccSeqsLeConstantRate_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤
      ((Fintype.card Bool * syncTwo.size ^ 2 * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeConstantRateFixedClock`'s bound fails at the verifier that accepts everything**, at `fxRate1`'s
parameters, whose sync set has one value. -/
theorem EncardAccSeqsLeConstantRateFixedClock_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤ ((Fintype.card Bool * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeConstantRateWindows`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters over its two windows. -/
theorem EncardAccSeqsLeConstantRateWindows_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) 4 1).encard ≤
      ((Fintype.card Bool * syncTwo.size ^ min 2 2 * 1 ^ 4 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

/-- **`EncardAccSeqsLeConstantRatePerWindow`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters over its two windows. -/
theorem EncardAccSeqsLeConstantRatePerWindow_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 2)) (2 * 2) 1).encard ≤
      ((Fintype.card Bool * (syncTwo.size * 1 ^ 2) ^ 2 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le (by norm_num) (by norm_num) _

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

/-- **The shipped schedule accepts the honest certifier's run, which meets W2 and J**: `fxHonestI`'s four steps are
accepted at `fxRate`, its wire is its record, and its jitter class is below `D = 1`. -/
theorem DecodableConstantRate_honest :
    AcceptedUpTo fxRate fxHonestI.op 4 (fxHonestI.obsRun () 4) ∧ Regenerates fxHonestI.certifier (fun _ o => o) ∧
      JitterBounded fxHonestI.certifier 1 :=
  ⟨⟨fxAdv, honestI_accepted fxRate fxAdv 4⟩, fun _ _ _ => rfl, fun _ _ _ => Nat.one_pos⟩

/-- **`EncardDecodableLeConstantRate`'s bound fails at the verifier that accepts everything**: there `fxLeakI`'s
decoder recovers every secret, against the bound's value at `fxRate`'s parameters. -/
theorem EncardDecodableLeConstantRate_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 4) ∧ fxLeakDec (fxLeakI.wireRun z 4) = z}.encard ≤
      ((Fintype.card Bool * syncTwo.size ^ min 2 2 * 1 ^ 4 : ℕ) : ℕ∞) :=
  fxLeak_not_le (by norm_num) _ (ENat.coe_ne_top _)

/-- **`EncardDecodableLeConstantRatePerWindow`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters over its two windows. -/
theorem EncardDecodableLeConstantRatePerWindow_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (2 * 2)) ∧ fxLeakDec (fxLeakI.wireRun z (2 * 2)) = z}.encard ≤
      ((Fintype.card Bool * (syncTwo.size * 1 ^ 2) ^ 2 : ℕ) : ℕ∞) :=
  fxLeak_not_le (by norm_num) _ (ENat.coe_ne_top _)

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
