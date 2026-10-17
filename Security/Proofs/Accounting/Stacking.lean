import Mathlib
import Models.Accounting.Guarantees.Stacking
import Proofs.Accounting.RateAudit.DeployedSketchSum

/-!
# Stacking: the carrier lemma and the stacked session

`Accounting.SecurityProofs.CarrierBits` counts a circuit's values: a step of `x` bits splits the stores into at most
`2^x` classes, in each of which the rest of the provenance has `x` bits fewer, so the values take at most `2^n`
contents when every run has at most `n` input and encoded bits (`Provenance.card_image_le`). A carrier's values are its
sectors at every store, which take every value, so its sectors are at most `n` bits (`Carrier.card_le`), and under
`CircuitEncodeHardness` a rounding budget bounds the encoded bits (`Provenance.encoded_mul_le`).

`Accounting.SecurityProofs.StackedSessionConservationSketchDeployed` then reads each stacked device as the same device
at on-chip state `S + C` (`SketchDeployedSession.lift`, which changes no run) and applies the deployed guarantee there.
-/

namespace Accounting.RateSketch

open Pous (Bits)

namespace Provenance

variable {N m : ℕ} {α : Type}

/-- One step of `x` bits: the stores split by the step's answer `g`, and the values in each class are at most
`2^(n − x)`, so at most `2^n` in all. -/
theorem card_image_step [DecidableEq α] {x n : ℕ} (T : Finset (SketchStore N m)) (g : SketchStore N m → Bits x)
    (F : Bits x → SketchStore N m → α) (hx : x ≤ n)
    (hF : ∀ v, ((T.filter fun E => g E = v).image (F v)).card ≤ 2 ^ (n - x)) :
    (T.image fun E => F (g E) E).card ≤ 2 ^ n := by
  calc (T.image fun E => F (g E) E).card
      ≤ (Finset.univ.biUnion fun v => (T.filter fun E => g E = v).image (F v)).card := by
        apply Finset.card_le_card
        intro a ha
        obtain ⟨E, hE, rfl⟩ := Finset.mem_image.mp ha
        exact Finset.mem_biUnion.mpr ⟨g E, Finset.mem_univ _,
          Finset.mem_image.mpr ⟨E, Finset.mem_filter.mpr ⟨hE, rfl⟩, rfl⟩⟩
    _ ≤ ∑ v : Bits x, ((T.filter fun E => g E = v).image (F v)).card := Finset.card_biUnion_le
    _ ≤ ∑ _v : Bits x, 2 ^ (n - x) := Finset.sum_le_sum fun v _ => hF v
    _ = 2 ^ n := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul]
        simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]
        rw [← pow_add, Nat.add_sub_cancel' hx]

/-- **The circuit's values take at most `2^n` contents** over a set of stores at each of which it has at most `n`
input and encoded bits. -/
theorem card_image_le [DecidableEq α] (c : Provenance N m α) :
    ∀ (T : Finset (SketchStore N m)) (n : ℕ), (∀ E ∈ T, c.inputs E + c.encoded E ≤ n) →
      (T.image fun E => c.run E).card ≤ 2 ^ n := by
  induction c with
  | done a =>
    intro T n _
    calc (T.image fun E => (done a : Provenance N m α).run E).card ≤ ({a} : Finset α).card := by
          apply Finset.card_le_card
          intro b hb
          obtain ⟨E, _, rfl⟩ := Finset.mem_image.mp hb
          exact Finset.mem_singleton_self a
      _ = 1 := Finset.card_singleton a
      _ ≤ 2 ^ n := Nat.one_le_two_pow
  | input x f k ih =>
    intro T n hT
    rcases T.eq_empty_or_nonempty with rfl | ⟨E₀, hE₀⟩
    · simp
    have hx : x ≤ n := by
      have := hT E₀ hE₀
      simp only [inputs, encoded] at this
      omega
    exact card_image_step T f (fun v E => (k v).run E) hx fun v =>
      ih v _ _ fun E hE => by
        obtain ⟨hET, rfl⟩ := Finset.mem_filter.mp hE
        have := hT E hET
        simp only [inputs, encoded] at this
        omega
  | encode t x _ f k ih =>
    intro T n hT
    rcases T.eq_empty_or_nonempty with rfl | ⟨E₀, hE₀⟩
    · simp
    have hx : x ≤ n := by
      have := hT E₀ hE₀
      simp only [inputs, encoded] at this
      omega
    exact card_image_step T (fun E => f fun u => E (t, u)) (fun v E => (k v).run E) hx fun v =>
      ih v _ _ fun E hE => by
        obtain ⟨hET, rfl⟩ := Finset.mem_filter.mp hE
        have := hT E hET
        simp only [inputs, encoded] at this
        omega

/-- **A rounding budget bounds the encoded bits**: under `CircuitEncodeHardness` at a tile's `m·256` bits, the encoded
bits times `rEnc` are at most `m·256` times the encodes' roundings. -/
theorem encoded_mul_le {rEnc : ℕ} {cost : ℕ → ℕ}
    (hE : Assumptions.CircuitEncodeHardness (m * 256) rEnc cost) (c : Provenance N m α) (E : SketchStore N m) :
    c.encoded E * rEnc ≤ m * 256 * c.roundings cost E := by
  induction c with
  | done a => simp [encoded, roundings]
  | input x f k ih => simpa [encoded, roundings] using ih (f E)
  | encode t x hx f k ih =>
    simp only [encoded, roundings]
    rw [add_mul, mul_add]
    exact add_le_add (ih _) (hE x hx)

end Provenance

namespace Carrier

variable {N m : ℕ}

/-- **A carrier's sectors are at most its input and encoded bits.** Its values are its sectors at every store, and
those take every value, `2^(256·|carried|)` of them. -/
theorem card_le (K : Carrier N m) {n : ℕ} (hK : ∀ E, K.provenance.inputs E + K.provenance.encoded E ≤ n) :
    K.carried.card * 256 ≤ n := by
  classical
  have h := K.provenance.card_image_le Finset.univ n fun E _ => hK E
  have himg : (Finset.univ.image fun E => K.provenance.run E) = Finset.univ := by
    refine Finset.eq_univ_of_forall fun v => Finset.mem_image.mpr ?_
    refine ⟨fun p => if hp : p ∈ K.carried then v ⟨p, hp⟩ else fun _ => 0, Finset.mem_univ _, ?_⟩
    rw [K.holds]
    funext p
    simp [p.2]
  rw [himg, Finset.card_univ, Fintype.card_fun, Fintype.card_coe] at h
  have hs : Fintype.card Sector = 2 ^ 256 := by simp [Sector]
  rw [hs, ← pow_mul, mul_comm] at h
  exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h

/-- A carrier whose circuit takes in at most `I` bits and spends at most `R` roundings on its encodes carries at most
`I + m·256·R/rEnc` bits, under `CircuitEncodeHardness` at a tile's `m·256` bits. -/
theorem card_le_of_budget (K : Carrier N m) {I R rEnc : ℕ} {cost : ℕ → ℕ} (hr : 0 < rEnc)
    (hE : Assumptions.CircuitEncodeHardness (m * 256) rEnc cost)
    (hK : ∀ E, K.provenance.inputs E ≤ I ∧ K.provenance.roundings cost E ≤ R) :
    K.carried.card * 256 ≤ I + m * 256 * R / rEnc := by
  refine K.card_le fun E => ?_
  have h2 : K.provenance.encoded E ≤ m * 256 * R / rEnc := by
    rw [Nat.le_div_iff_mul_le hr]
    exact (Provenance.encoded_mul_le hE K.provenance E).trans (Nat.mul_le_mul_left _ (hK E).2)
  have h1 := (hK E).1
  omega

end Carrier

/-- The same device at a larger on-chip budget `S'`, which its held sectors fit. -/
def Device.lift {N m ρ h S S' L q w : ℕ} {X Sig Dg : Type} (B : Device N m ρ h S L q w X Sig Dg)
    (hB : B.held.card * w ≤ S') : Device N m ρ h S' L q w X Sig Dg :=
  ⟨B.held, hB, B.chip, B.bounded⟩

/-- The same session at on-chip budget `S'`, which every device's held sectors fit: each device lifted, each pick
unchanged, so every run, receipt and read is the session's. -/
def SketchDeployedSession.lift {n P N m ρ h T L q ks G d : ℕ} (A : SketchDeployedSession n P N m ρ h T L q ks G d)
    (S' : ℕ) (hA : ∀ w η j ι c, ((A.window w η).device j ι c).held.card * 256 ≤ S') :
    SketchDeployedSession n P N m ρ h S' L q ks G d where
  window w η :=
    { device := fun j ι c => ((A.window w η).device j ι c).lift (hA w η j ι c)
      pick := (A.window w η).pick }

end Accounting.RateSketch

namespace Accounting.SecurityProofs

open Accounting.RateSketch

/-- **The carrier lemma**, `Accounting.Guarantees.CarrierBits`. -/
theorem CarrierBits : Accounting.Guarantees.CarrierBits := by
  intro N m I X α c hc
  classical
  have h := c.card_image_le Finset.univ (I + X) fun E _ => add_le_add (hc E).1 (hc E).2
  rwa [Nat.card_coe_set_eq, Set.ncard_eq_toFinset_card', Set.toFinset_range]

/-- **The remote audit at the deployed parameters, with stacking**,
`Accounting.Guarantees.StackedSessionConservationSketchDeployed`. -/
theorem StackedSessionConservationSketchDeployed :
    Accounting.Guarantees.StackedSessionConservationSketchDeployed := by
  intro n P N ρ T S L q G X qtot Rin cρ reads life rEnc cost ε A hn hP hN hr hrank hE hX hB hK C
  have hA : ∀ w η j ι c, ((A.window w η).device j ι c).held.card * 256 ≤ S + C := by
    intro w η j ι c
    obtain ⟨K, hS, hKb⟩ := hK w η j ι c
    have hK' : K.carried.card * 256 ≤ C := K.card_le_of_budget hr hE hKb
    calc ((A.window w η).device j ι c).held.card * 256
        ≤ ((((A.window w η).device j ι c).held \ K.carried).card + K.carried.card) * 256 :=
          Nat.mul_le_mul_right _ Finset.card_le_card_sdiff_add_card
      _ = (((A.window w η).device j ι c).held \ K.carried).card * 256 + K.carried.card * 256 := add_mul _ _ _
      _ ≤ S + C := add_le_add hS hK'
  exact RemoteSessionConservationSketchDeployed n P N ρ (S + C) L q G X qtot ε (A.lift (S + C) hA) hn hP hN
    hrank hX hB

end Accounting.SecurityProofs
