import Properties.Pous.Period
import Proofs.Pous.EndToEnd
import Proofs.Properties.Pous.Run
import Proofs.Accounting.Prob
import Proofs.Accounting.Capacity
import Proofs.Accounting.PerRunJoint

/-!
# Proof of PoUS's per-period guarantee, and that it is the joint's capacity guarantee

At a value `o` of the other audits' coins, a certificate `S` with `S · len > 0` and `B = ⌊S · len⌋₊`: on a capacity coin
where the verifier accepts with no break in the period's data, the period is certified (`certified_of`, from
`runCertifies`), and where the capacity account held fewer than `S · len` bit-ticks, it held at most `B` bits at
every tick of the window (`Accounting.Run.mem_le_held`). `B + S_chip + c ≤ S₀`, so `RingDevice` gives one ring-model
server passing on every such coin, and `endToEndRing` bounds its pass by `δ + ε_max`. With `S · len ≤ 0` the event is
empty. A run whose capacity verdict is that acceptance outside a break then has `CapacitySoundPerRun`, and with it
`CapacitySound` (`Accounting.Run.capacitySound_of_perRun`).
-/

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

theorem certified_of {ΩW ΩQ : Type} (P : Audits ΩW ΩQ)
    {ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    (ha : P.Accepts ω) (hb : ¬ P.Break ω) : P.Certified ω := by
  obtain ⟨hdec, hpub, haud, hsid, htr, hrun⟩ :=
    Pous.EndToEnd.SecurityProofs.runCertifies P.trusted (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1))
      (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.session ω) P.sid (P.device ω) ha
  refine ⟨hdec, hpub, haud, hsid, htr, ?_⟩
  rcases hrun with h | h | h
  · exact h
  · exact absurd (Or.inl h) hb
  · exact absurd (Or.inr h) hb

theorem δ_nonneg : (0 : ℝ) ≤ (Params.δ : ℝ) := by
  have : (0 : ℚ) ≤ Params.δ := by norm_num [Params.δ]
  exact_mod_cast this

theorem εMax_ne_top : Params.εMax ≠ ⊤ := by
  unfold Params.εMax
  exact ENNReal.pow_ne_top (ENNReal.inv_ne_top.2 two_ne_zero)

theorem periodHeld : PeriodHeld := by
  intro D Q L hP ι α ΩW ΩQ _ _ d r P round S hdev hS o
  by_cases hpos : 0 < S * (r.win .capacity).len
  · set B := ⌊S * (r.win .capacity).len⌋₊
    have hfl : (B : ℝ) ≤ S * (r.win .capacity).len := Nat.floor_le hpos.le
    have hn : B + d.chip + ∑ i, d.cap i * round ≤ Pous.EndToEnd.S := by
      have : ((B + d.chip + ∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S := by
        rw [Nat.cast_add, Nat.cast_add]
        linarith
      exact_mod_cast this
    obtain ⟨A, hA⟩ := hdev o B hn
    refine le_trans (Pouw.pr_mono (P' := Passes P.sid (P.W o) A) ?_) ?_
    · rintro x ⟨ha, hb, hh⟩
      exact hA x (fun t ht => Nat.le_floor ((Nat.cast_le.2 (r.mem_le_held ht)).trans hh.le))
        (certified_of P ha hb)
    · have hfin : ENNReal.ofReal (Params.δ : ℝ) + Params.εMax ≠ ⊤ :=
        ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, εMax_ne_top⟩
      calc Pouw.pr (Passes P.sid (P.W o) A)
          = (Pous.pr (Passes P.sid (P.W o) A)).toReal := Accounting.pr_eq_toReal _
        _ ≤ (ENNReal.ofReal (Params.δ : ℝ) + Params.εMax).toReal :=
          ENNReal.toReal_mono hfin (Pous.EndToEnd.SecurityProofs.endToEndRing D Q L hP _ _ A)
        _ = _ := by rw [ENNReal.toReal_add ENNReal.ofReal_ne_top εMax_ne_top, ENNReal.toReal_ofReal δ_nonneg]
  · rw [Pouw.pr_eq_zero fun x hx => hpos (lt_of_le_of_lt (Nat.cast_nonneg _) hx.2.2)]
    exact add_nonneg δ_nonneg ENNReal.toReal_nonneg

theorem periodCapacity : PeriodCapacity := by
  intro D Q L hP ι α ΩW ΩQ _ _ _ _ d r P round S hdev hS hv
  have h : r.CapacitySoundPerRun S ((Params.δ : ℝ) + Params.εMax.toReal) := fun x q => by
    refine le_trans (Pouw.pr_mono ?_) (periodHeld D Q L hP ι α ΩW ΩQ d r P round S hdev hS (x, q))
    rintro s ⟨hacc, hheld⟩
    exact ⟨(hv _ hacc).1, (hv _ hacc).2, hheld⟩
  exact ⟨h, r.capacitySound_of_perRun h⟩

/-! ### The honest and nontrivial lemmas

The device premise, `RingDevice`, holds at an honest device: one whose capacity account holds more than `S₀` bits at a
tick of the capacity window on every coin, as a server keeping all of `C` (`FULL > S₀` bits) does, is a ring-model
server for every period's audit, since no bound `B` within `S₀` holds its account. A verifier accepting every period
fails the conclusion: on a run whose account holds nothing, its event holds on every coin, with probability 1, above
`δ + ε_max`. -/

/-- A server that keeps nothing and answers every index with the zero block, with no query. -/
def nullServer (D Q L : ℕ) : Server Pous.EndToEnd.S D Q L :=
  ⟨fun _ _ _ _ _ => false, fun _ _ _ σ _ => .ret (fun _ => false, σ),
    fun _ _ _ _ _ _ => ⟨Nat.zero_le _, Nat.zero_le _, Nat.zero_le _⟩⟩

/-- **`RingDevice` holds at an honest device**, one whose capacity account exceeds `S₀` at a tick of the capacity
window on every coin, at every `(D, Q, L)`. -/
theorem ringDevice_honest {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (D Q L round : ℕ)
    (hfull : ∀ ω, ∃ t ∈ (r.win .capacity).ticks, Pous.EndToEnd.S < (r.schedule ω t).mem (r.acct .capacity)) :
    RingDevice r P D Q L round := by
  intro o B hB
  refine ⟨nullServer D Q L, fun x hx _ => ?_⟩
  obtain ⟨t, ht, hlt⟩ := hfull (o.1, o.2, x)
  have := hx t ht
  exfalso
  omega

theorem bound_toReal_lt_one : (Params.δ : ℝ) + Params.εMax.toReal < 1 := by
  have hfin : ENNReal.ofReal (Params.δ : ℝ) + Params.εMax ≠ ⊤ :=
    ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, εMax_ne_top⟩
  have h := (ENNReal.toReal_lt_toReal hfin ENNReal.one_ne_top).2 Pous.EndToEnd.SecurityProofs.bound_lt_one
  rwa [ENNReal.toReal_add ENNReal.ofReal_ne_top εMax_ne_top, ENNReal.toReal_ofReal δ_nonneg,
    ENNReal.toReal_one] at h

/-- **`periodHeld` holds at the honest instance**: its device premise, `ringDevice_honest`. -/
theorem periodHeld_honest {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (D Q L round : ℕ)
    (hfull : ∀ ω, ∃ t ∈ (r.win .capacity).ticks, Pous.EndToEnd.S < (r.schedule ω t).mem (r.acct .capacity)) :
    RingDevice r P D Q L round :=
  ringDevice_honest r P D Q L round hfull

/-- **`periodHeld` is nontrivial**: an event that holds on every capacity coin, a verifier accepting every period on a
run whose account holds nothing, exceeds its bound `δ + ε_max`. -/
theorem periodHeld_nontrivial :
    ¬ Pouw.pr (fun _ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      (Params.δ : ℝ) + Params.εMax.toReal := by
  rw [Pouw.pr_eq_one fun _ => trivial]
  exact not_le.2 bound_toReal_lt_one

/-- **`periodCapacity` holds at the honest instance**: `periodHeld_honest`, its premises being `periodHeld`'s. -/
theorem periodCapacity_honest {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (D Q L round : ℕ)
    (hfull : ∀ ω, ∃ t ∈ (r.win .capacity).ticks, Pous.EndToEnd.S < (r.schedule ω t).mem (r.acct .capacity)) :
    RingDevice r P D Q L round :=
  ringDevice_honest r P D Q L round hfull

/-- **`periodCapacity` is nontrivial**: `periodHeld_nontrivial`, its bound being `periodHeld`'s. -/
theorem periodCapacity_nontrivial :
    ¬ Pouw.pr (fun _ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      (Params.δ : ℝ) + Params.εMax.toReal :=
  periodHeld_nontrivial

end Pous.Period.SecurityProofs
