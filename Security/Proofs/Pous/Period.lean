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
  intro hH D Q L hP ι α ΩW ΩQ _ _ d r P round S hdev hS o
  by_cases hpos : 0 < S * (r.win .capacity).len
  · set B := ⌊S * (r.win .capacity).len⌋₊
    have hfl : (B : ℝ) ≤ S * (r.win .capacity).len := Nat.floor_le hpos.le
    have hn : B + d.chip + ∑ i, d.cap i * round ≤ Pous.EndToEnd.S := by
      have : ((B + d.chip + ∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S := by
        rw [Nat.cast_add, Nat.cast_add]
        linarith
      exact_mod_cast this
    obtain ⟨A, hA⟩ := hdev o B hn
    refine le_trans (Pouw.pr_mono (P' := Passes P.sid (P.pub o).root (P.W o) A) ?_) ?_
    · rintro x ⟨ha, hb, hh⟩
      exact hA x (fun t ht => Nat.le_floor ((Nat.cast_le.2 (r.mem_le_held ht)).trans hh.le))
        (certified_of P ha hb)
    · have hfin : ENNReal.ofReal (Params.δ : ℝ) + Params.εMax ≠ ⊤ :=
        ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, εMax_ne_top⟩
      calc Pouw.pr (Passes P.sid (P.pub o).root (P.W o) A)
          = (Pous.pr (Passes P.sid (P.pub o).root (P.W o) A)).toReal := Accounting.pr_eq_toReal _
        _ ≤ (ENNReal.ofReal (Params.δ : ℝ) + Params.εMax).toReal :=
          ENNReal.toReal_mono hfin (Pous.EndToEnd.SecurityProofs.endToEndRing hH D Q L hP _ _ _ A)
        _ = _ := by rw [ENNReal.toReal_add ENNReal.ofReal_ne_top εMax_ne_top, ENNReal.toReal_ofReal δ_nonneg]
  · rw [Pouw.pr_eq_zero fun x hx => hpos (lt_of_le_of_lt (Nat.cast_nonneg _) hx.2.2)]
    exact add_nonneg δ_nonneg ENNReal.toReal_nonneg

theorem periodCapacity : PeriodCapacity := by
  intro hH D Q L hP ι α ΩW ΩQ _ _ _ _ d r P round S hdev hS hv
  have h : r.CapacitySoundPerRun S ((Params.δ : ℝ) + Params.εMax.toReal) := fun x q => by
    refine le_trans (Pouw.pr_mono ?_) (periodHeld hH D Q L hP ι α ΩW ΩQ d r P round S hdev hS (x, q))
    rintro s ⟨hacc, hheld⟩
    exact ⟨(hv _ hacc).1, (hv _ hacc).2, hheld⟩
  exact ⟨h, r.capacitySound_of_perRun h⟩

end Pous.Period.SecurityProofs
