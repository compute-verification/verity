import Proofs.NetworkCertifier.WitnessToys

/-!
# The egress counts' honest and nontrivial lemmas

Each count over `Accepted` that a doc cites as a result has an honest and a nontrivial lemma (lean's ruling on rule 3,
27 Oct). The six egress counts over `AccSeqs` take no premise outside a named assumption's module, so their honest
lemmas say that the deployed Program accepts the honest certifier's step: at the constant-rate Program at one bucket
(`fxRate`, and `fxFixed` with its clock fixed), the honest step `fxHonest` is among the counted sequences. Their
nontrivial lemmas say that each bound fails at the verifier that accepts everything: there, one step of one link of
one slot carries any payload, so the sequences counted are infinitely many (`accSeqsAt_acceptAll_not_le`), against
the bound's value at `fxRate`'s parameters.

The count under a sampled audit (`EncardAccSeqsSampledLe`) takes that the layout fits the circuit. Its honest lemma
concludes that at the negation `notStmt`, with the faulty-bits premise and the acceptance of the step `obs1`; its
nontrivial lemma is the same count at `acceptAll`, against the bound's value at the negation's one faulty set.

The two counts through the wire (`EncardDecodableLeConstantRate` and its per-window form) take the run's acceptance,
which their honest lemma concludes for the honest certifier `fxHonestI` at `fxRate`, with W2 and J, its two named
assumptions, holding there too. At the verifier that accepts everything a sender whose secret is its frame's length
(`fxLeakI`) has every secret decoded (`fxLeak_not_le`).

The six ingress counts take that every status is complete (`StatusComplete`, acceptance as it bears on ingress), which
#4871's rule doesn't hold to witnesses, so their entries name their kind. Their honest lemma concludes it for
`fxHonestI`, with W2, W4 and J; their nontrivial lemmas decode every message of `fxLeakI` from its records or its wire.
-/

namespace NetTiming.Witness

/-- **The shipped schedule accepts the honest certifier's step**: `fxHonest` is among `fxRate`'s accepted sequences of
one step. -/
theorem ConstantRate_honest : fxHonest ∈ AccSeqs fxRate fxOp 1 1 :=
  fxHonest_accepted.1

/-- **With its clock fixed, the shipped schedule accepts the honest certifier's step.** -/
theorem FixedClock_honest : fxHonest ∈ AccSeqs fxFixed fxOp 1 1 :=
  fxHonest_accepted.2

/-- **`EncardAccSeqsLeSync`'s bound fails at the verifier that accepts everything**, at `fxRate`'s parameters. -/
theorem EncardAccSeqsLeSync_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤
      ((Fintype.card Unit * Fintype.card syncOne.Elem ^ 1 * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le _

/-- **`EncardAccSeqsLeFixedClock`'s bound fails at the verifier that accepts everything**, at `fxFixed`'s parameters. -/
theorem EncardAccSeqsLeFixedClock_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤ ((Fintype.card Unit * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le _

/-- **`EncardAccSeqsLeConstantRate`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters. -/
theorem EncardAccSeqsLeConstantRate_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤
      ((Fintype.card Unit * syncOne.size ^ 1 * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le _

/-- **`EncardAccSeqsLeConstantRateFixedClock`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters, whose sync set has one value. -/
theorem EncardAccSeqsLeConstantRateFixedClock_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤ ((Fintype.card Unit * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le _

/-- **`EncardAccSeqsLeConstantRateWindows`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters over its first window. -/
theorem EncardAccSeqsLeConstantRateWindows_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤
      ((Fintype.card Unit * syncOne.size ^ min 1 1 * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le _

/-- **`EncardAccSeqsLeConstantRatePerWindow`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters over its one window. -/
theorem EncardAccSeqsLeConstantRatePerWindow_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) (1 * 1) 1).encard ≤
      ((Fintype.card Unit * (syncOne.size * 1 ^ 1) ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le _

/-! ## Under a sampled audit -/

/-- **The negation's audit meets the sampled count's premises and accepts its honest step**: `notLay` fits `notProg`
over one step, the one possible faulty set at `δ = 0` commits no bits, and the step `obs1` is accepted. -/
theorem EncardAccSeqsSampledLe_honest :
    notLay.Fits notProg 1 ∧ (∀ F ∈ faultySets notStmt.design notPart 0, faultyBits notProg notPart F ≤ 0) ∧
      [obs1] ∈ AccSeqsSampled notProg notPart notStmt.design notLay 1 {fun _ => false} 0 (fun _ _ => [false]) 1 :=
  ⟨notLay_fits, fun F hF => by simp [notFaulty_empty hF, faultyBits], obs1_acc⟩

/-- **`EncardAccSeqsSampledLe`'s bound fails at the verifier that accepts everything**, at the negation's parameters:
`c = β = 0`, its one possible faulty set, `D = 1`. -/
theorem EncardAccSeqsSampledLe_nontrivial :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := 1)) 1 1).encard ≤
      ((2 ^ 0 * (faultySets notStmt.design notPart 0).card * 2 ^ 0 * 1 ^ 1 : ℕ) : ℕ∞) :=
  accSeqsAt_acceptAll_not_le _

/-! ## Through the wire -/

/-- **The shipped schedule accepts the honest certifier's run, which meets W2 and J**: `fxHonestI`'s first step is
accepted at `fxRate`, its wire is its record, and its jitter class is below `D = 1`. -/
theorem DecodableConstantRate_honest :
    AcceptedUpTo fxRate fxHonestI.op 1 (fxHonestI.obsRun () 1) ∧ Regenerates fxHonestI.certifier (fun _ o => o) ∧
      JitterBounded fxHonestI.certifier 1 :=
  ⟨fxHonest_accepted.1.2.2, fx_wire.1, fx_wire.2.1⟩

/-- **`EncardDecodableLeConstantRate`'s bound fails at the verifier that accepts everything**: there `fxLeakI`'s
decoder recovers every secret, against the bound's value at `fxRate`'s parameters. -/
theorem EncardDecodableLeConstantRate_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.wireRun z 1) = z}.encard ≤
      ((Fintype.card Unit * syncOne.size ^ min 1 1 * 1 ^ 1 : ℕ) : ℕ∞) :=
  fxLeak_not_le _

/-- **`EncardDecodableLeConstantRatePerWindow`'s bound fails at the verifier that accepts everything**, at `fxRate`'s
parameters over its one window. -/
theorem EncardDecodableLeConstantRatePerWindow_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (1 * 1)) ∧ fxLeakDec (fxLeakI.wireRun z (1 * 1)) = z}.encard ≤
      ((Fintype.card Unit * (syncOne.size * 1 ^ 1) ^ 1 : ℕ) : ℕ∞) :=
  fxLeak_not_le _

/-! ## Into X -/

/-- **The honest certifier's run has every status complete, and it meets W2, W4 and J**: at one frame a bucket on its
one link, with the wire its record. -/
theorem Ingress_honest :
    StatusComplete 1 (fxHonestI.obsRun () 1) ∧ Packed fxHonestI.certifier ∧
      Allocated fxHonestI.certifier (fun _ => 1) ∧ JitterBounded fxHonestI.certifier 1 ∧
      Regenerates fxHonestI.certifier (fun _ o => o) :=
  ⟨fxHonestI_complete, fxHonestI_w4.1, fxHonestI_w4.2, fx_wire.2.1, fx_wire.1⟩

/-- **`EncardIngressObsDecodableLe`'s bound fails at the verifier that accepts everything**, at one frame a bucket on
one link: there a decoder of `fxLeakI`'s records recovers every message. -/
theorem EncardIngressObsDecodableLe_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.obsRun z 1) = z}.encard ≤
      (((∏ _ℓ : Fin 1, (1 + 1)) ^ 1 * 1 ^ 1 : ℕ) : ℕ∞) :=
  fxLeakObs_not_le _

/-- **`EncardIngressDecodableLe`'s bound fails at the verifier that accepts everything**: a decoder of `fxLeakI`'s
wire recovers every message. -/
theorem EncardIngressDecodableLe_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.wireRun z 1) = z}.encard ≤
      (((∏ _ℓ : Fin 1, (1 + 1)) ^ 1 * 1 ^ 1 : ℕ) : ℕ∞) :=
  fxLeak_not_le _

/-- **`EncardIngressObsDecodableLeContents`'s bound fails at the verifier that accepts everything**, at one content
assignment. -/
theorem EncardIngressObsDecodableLeContents_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.obsRun z 1) = z}.encard ≤
      ({fun _ => []} : Set (Fin 1 → List (List Bool))).encard *
        (((∏ _ℓ : Fin 1, (1 + 1)) ^ 1 * 1 ^ 1 : ℕ) : ℕ∞) := by
  rw [Set.encard_singleton, one_mul]
  exact fxLeakObs_not_le _

/-- **`EncardIngressDecodableLeContents`'s bound fails at the verifier that accepts everything**, at one content
assignment. -/
theorem EncardIngressDecodableLeContents_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z 1) ∧ fxLeakDec (fxLeakI.wireRun z 1) = z}.encard ≤
      ({fun _ => []} : Set (Fin 1 → List (List Bool))).encard *
        (((∏ _ℓ : Fin 1, (1 + 1)) ^ 1 * 1 ^ 1 : ℕ) : ℕ∞) := by
  rw [Set.encard_singleton, one_mul]
  exact fxLeak_not_le _

/-- **`EncardIngressObsDecodableLePerWindow`'s bound fails at the verifier that accepts everything**, over one window
of one bucket. -/
theorem EncardIngressObsDecodableLePerWindow_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (1 * 1)) ∧ fxLeakDec (fxLeakI.obsRun z (1 * 1)) = z}.encard ≤
      ((((∏ _ℓ : Fin 1, (1 + 1) ^ 1) * 1 ^ 1) ^ 1 : ℕ) : ℕ∞) :=
  fxLeakObs_not_le _

/-- **`EncardIngressObsDecodableLeContentsPerWindow`'s bound fails at the verifier that accepts everything**, over one
window of one bucket at one content assignment. -/
theorem EncardIngressObsDecodableLeContentsPerWindow_nontrivial :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z (1 * 1)) ∧ fxLeakDec (fxLeakI.obsRun z (1 * 1)) = z}.encard ≤
      ({fun _ => []} : Set (Fin 1 → List (List Bool))).encard *
        ((((∏ _ℓ : Fin 1, (1 + 1) ^ 1) * 1 ^ 1) ^ 1 : ℕ) : ℕ∞) := by
  rw [Set.encard_singleton, one_mul]
  exact fxLeakObs_not_le _

end NetTiming.Witness
