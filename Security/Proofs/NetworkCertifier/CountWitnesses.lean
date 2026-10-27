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

The two counts through the wire (`EncardDecodableLeConstantRate` and its per-window form) take the run's acceptance,
which their honest lemma concludes for the honest certifier `fxHonestI` at `fxRate`, with W2 and J, its two named
assumptions, holding there too. At the verifier that accepts everything a sender whose secret is its frame's length
(`fxLeakI`) has every secret decoded (`fxLeak_not_le`).
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

end NetTiming.Witness
