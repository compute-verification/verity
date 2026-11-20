module

public import Properties.Poub.Remote.Shared
public import Proofs.Properties.Poub.Remote.RecordAccepts

/-!
# Witnesses of the window's, the sample's and the appends' properties

The honest witnesses are at `RecordAccepts`'s accepted session: the one-block registration `reg1`, which
`Registration.make?` accepts at the deployed codec, sketch-sha512 receipts of 16 rows (`reg1_make`), and its window 0 of
one pass over the whole store under challenge `zeros 64`, whose one receipt a sample of one draws on 128 zero bytes
(`receiptSample_one`) and whose opening `checkWindow` accepts (`checkWindow_one`).

* `verifyWindow_one`: `verifyWindow` accepts that window, its one opening and its sample's coins: the premise of
  `AcceptChecksOpenings`, `AcceptOpensSamples` and `AcceptSamplesOpenable`.
* `sampleIsSubset_honest`, `Shared.acceptSamplesOpenable_honest`: the sample of that window, alone or as the one window of
  its sample, is drawn.
* `appendsInOrder_honest`: the session's run, which accepts (`run_accepts`).

The nontrivial witnesses: on a sample's coins of no bytes the draw of one receipt among one runs out on its first read
(`sampleUnits_empty`), so no sample is drawn (`receiptSample_empty`) and no values are the draw's (`sampleValues_empty`),
while a verifier that accepted every window would accept it (also the shared sample's `SampleIsSubset`,
`RunSamplesRecords` and `AcceptedSamplesOpenable`, whose honest witnesses need a run's state at a sample's close); and a
state whose store does not end where its registration
and appends end fails `AppendsInOrder`'s first conjunct.
-/

@[expose] public section

open _root_.Poub

namespace Properties.Poub

open Properties.Poub.Line.Honest

/-- The draw of one unit among one on no bytes throws `RAN_OUT` at its first read, before its loop. -/
theorem sampleUnits_empty : Spot.Draw.sampleUnits (.subset 1) 1 ByteArray.empty = .error Spot.Draw.RAN_OUT := by
  unfold Spot.Draw.sampleUnits Flock.Draw.drawWith Flock.Draw.onWith
  simp only [width_one]
  rfl

theorem receiptSample_empty : receiptSample 1 ByteArray.empty #[1] = none := by
  unfold receiptSample receiptSampleRead
  simp [sampleUnits_empty]

theorem sampleValues_empty (us : Array ℕ) : ¬ SampleValues 1 1 ByteArray.empty us := by
  rintro (⟨h, -⟩ | ⟨-, read, h⟩)
  · exact absurd h (by decide)
  · rw [show min 1 1 = 1 from rfl, sampleUnits_empty] at h
    exact Except.noConfusion h

/-- `RecordAccepts`'s window 0 under challenge `zeros 64`. -/
def window0 : WindowRecord := window1 reg1 (zeros 64) (root1 reg1 (seed (zeros 64) 0 reg1.digest))

theorem window0_lengths : window0.passes.map (·.plan.length) = #[1] := rfl

theorem verifyWindow_one : verifyWindow reg1 #[] window0 (zeros 128) 1 #[opening1 reg1 (zeros 64)] = some .accept := by
  have hs : receiptSample 1 (zeros 128) (window0.passes.map (·.plan.length)) = some #[(0, 0)] := by
    rw [window0_lengths]
    exact receiptSample_one
  have hc := checkWindow_one reg1 (zeros 64) rfl rfl rfl
  unfold verifyWindow
  rw [hs]
  exact congrArg some hc

/-- **`AcceptChecksOpenings`' premise holds at the one-block registration.** `verifyWindow` accepts window 0 of one
pass over the whole store of a registration `Registration.make?` accepts, on a sample of one receipt and its opening. -/
theorem acceptChecksOpenings_honest :
    ∃ (reg : Registration) (w : WindowRecord) (coin : ByteArray) (openings : Array Opening),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        verifyWindow reg #[] w coin 1 openings = some .accept :=
  ⟨reg1, window0, zeros 128, #[opening1 reg1 (zeros 64)], reg1_make, verifyWindow_one⟩

/-- **`AcceptChecksOpenings` fails without the draw.** On a sample's coins of no bytes no sample is drawn over the
window's one receipt, which a verifier that accepted every window would accept. -/
theorem acceptChecksOpenings_nontrivial : ∀ sampled, ¬ receiptSample 1 ByteArray.empty #[1] = some sampled := by
  intro sampled h
  rw [receiptSample_empty] at h
  exact Option.noConfusion h

/-- **`AcceptOpensSamples`' premise holds at the one-block registration** (`acceptChecksOpenings_honest`). -/
theorem acceptOpensSamples_honest :
    ∃ (reg : Registration) (w : WindowRecord) (coin : ByteArray) (openings : Array Opening),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        verifyWindow reg #[] w coin 1 openings = some .accept :=
  acceptChecksOpenings_honest

/-- **`AcceptOpensSamples` fails without the draw** (`acceptChecksOpenings_nontrivial`). -/
theorem acceptOpensSamples_nontrivial : ∀ sampled, ¬ receiptSample 1 ByteArray.empty #[1] = some sampled :=
  acceptChecksOpenings_nontrivial

/-- **`AcceptSamplesOpenable`' premise holds at the one-block registration** (`acceptChecksOpenings_honest`). -/
theorem acceptSamplesOpenable_honest :
    ∃ (reg : Registration) (w : WindowRecord) (coin : ByteArray) (openings : Array Opening),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        verifyWindow reg #[] w coin 1 openings = some .accept :=
  acceptChecksOpenings_honest

/-- **`AcceptSamplesOpenable` fails without the draw.** On a sample's coins of no bytes no values among the window's one
receipt are the draw's. -/
theorem acceptSamplesOpenable_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`SampleIsSubset`'s premise holds at the one-block registration's window**: its one receipt is drawn on 128 zero
bytes. -/
theorem sampleIsSubset_honest :
    ∃ (src : ByteArray) (sampled : Array (ℕ × ℕ)), receiptSample 1 src (window0.passes.map (·.plan.length)) =
      some sampled :=
  ⟨zeros 128, #[(0, 0)], by rw [window0_lengths]; exact receiptSample_one⟩

/-- **`SampleIsSubset` fails without the draw.** On no bytes no values among one receipt are the draw's. -/
theorem sampleIsSubset_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`AppendsInOrder`'s premise holds at the one-block registration**: the session's run over window 0, its one pass,
its sample and its opening accepts (`run_accepts`). -/
theorem appendsInOrder_honest :
    ∃ (reg : Registration) (coins : Array ByteArray) (events : Array _root_.Poub.Session.Event)
      (rs : Array _root_.Poub.Session.Result) (s : _root_.Poub.Session.State),
      Registration.make? (zeros 1) 1 reg.payloadRoot (zeros Codec.SEED_BYTES) honestCodec .sketch false LAYOUT 16 =
          .ok reg ∧
        _root_.Poub.Session.run (_root_.Poub.Session.State.init params1 reg coins #[] 0) events = .ok (rs, s) ∧
        s.outcome = some "accepted" := by
  have hroot : (root1 reg1 (seed (zeros 64) 0 reg1.digest)).size = 64 := by
    simp only [root1, zeroRoot, climb_size]
  obtain ⟨rs, s, hr, ho, -⟩ := run_accepts reg1 rfl (zeros 64) _ hroot _ (checkWindow_one reg1 (zeros 64) rfl rfl rfl)
  exact ⟨reg1, _, _, rs, s, reg1_make, hr, ho⟩

/-- A state of the one-block registration whose store ends at block 2, with no appends. -/
def overrun : _root_.Poub.Session.State :=
  { _root_.Poub.Session.State.init params1 reg1 #[] #[] 0 with extent := 2 }

/-- **`AppendsInOrder` fails without the session's checks.** A state whose store ends past its registration's one block
with no appends, which a verifier that accepted every session would leave, has its store not ending where its
registration and appends do. -/
theorem appendsInOrder_nontrivial : ¬ overrun.extent = reg1.blocks + (overrun.appends.toList.map (·.count)).sum := by
  intro h
  have h' : (2 : ℕ) = 1 + 0 := h
  omega

namespace Shared

/-- **`Shared.AcceptSamplesOpenable`'s premise holds at the one-block registration**: a sample whose one window is its
window 0 draws that window's one receipt on 128 zero bytes. -/
theorem acceptSamplesOpenable_honest :
    ∃ (coin : ByteArray) (sampled : Array (ℕ × ℕ)), receiptSample 1 coin (lengthsOf [window0]) = some sampled :=
  ⟨zeros 128, #[(0, 0)], by rw [show lengthsOf [window0] = #[1] from rfl]; exact receiptSample_one⟩

/-- **`Shared.AcceptSamplesOpenable` fails without the draw.** On no bytes no values among one receipt are the
draw's. -/
theorem acceptSamplesOpenable_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`Shared.SampleIsSubset` fails without the draw.** On no bytes no values among one receipt are the draw's. -/
theorem sampleIsSubset_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

/-- **`Shared.RunSamplesRecords` fails without the draw.** On a sample's coins of no bytes no sample is drawn over one
receipt. -/
theorem runSamplesRecords_nontrivial : ∀ sampled, ¬ receiptSample 1 ByteArray.empty #[1] = some sampled :=
  acceptChecksOpenings_nontrivial

/-- **`Shared.AcceptedSamplesOpenable` fails without the draw.** On no bytes no values among one receipt are the
draw's. -/
theorem acceptedSamplesOpenable_nontrivial : ∀ us, ¬ SampleValues 1 1 ByteArray.empty us :=
  sampleValues_empty

end Shared

end Properties.Poub
