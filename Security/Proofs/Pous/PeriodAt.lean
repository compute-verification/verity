import Properties.Pous.PeriodAt
import Proofs.Pous.Period
import Proofs.Pous.EndToEndAt
import Proofs.Properties.Pous.RunAt

/-!
# Proof of PoUS's per-period guarantee at any point and block count

`periodHeld`'s proof (`Proofs.Pous.Period`) with the point's certificate (`certified_ofAt`, from
`Pous.EndToEnd.SecurityProofs.runCertifiesAt`) and the ring game at `N`
(`Pous.EndToEnd.SecurityProofs.endToEndRingOfRoundSecureWide`) in place of the deployed ones. `periodHeldDemo` is its
instance at Demo B's `N`, with the state bound `Pous.SecurityProofs.stateBound_p2ring_demo` gives.
-/

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

theorem certified_ofAt {B w p : ℕ} [NeZero p] {sch : Scheme (ringRO B w p)} {ΩW ΩQ : Type} (P : AuditsOf sch ΩW ΩQ)
    (pt : Run.Point)
    {ω : ΩW × ΩQ × ((ringRO B w p).Ω × HmCoins × Streams)} (ha : P.Accepts pt ω) (hb : ¬ P.Break pt ω) :
    P.Certified pt ω := by
  obtain ⟨hdec, hpub, haud, hsid, htr, hrun⟩ :=
    Pous.EndToEnd.SecurityProofs.runCertifiesAt pt P.trusted (P.commitment (ω.1, ω.2.1))
      (P.store (ω.1, ω.2.1)) (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.session ω) P.sid (P.device ω) ha
  refine ⟨hdec, hpub, haud, hsid, htr, ?_⟩
  rcases hrun with h | h | h
  · exact h
  · exact absurd (Or.inl h) hb
  · exact absurd (Or.inr h) hb

theorem periodHeldAt : PeriodHeldAt := by
  intro p ε₀ D Q L hnum hW pt N _ h1 h2 hH ι α ΩW ΩQ _ _ d r P round S hdev hS o
  by_cases hpos : 0 < S * (r.win .capacity).len
  · set B := ⌊S * (r.win .capacity).len⌋₊
    have hfl : (B : ℝ) ≤ S * (r.win .capacity).len := Nat.floor_le hpos.le
    have hn : B + d.chip + ∑ i, d.cap i * round ≤ SAt N := by
      have : ((B + d.chip + ∑ i, d.cap i * round : ℕ) : ℝ) ≤ SAt N := by
        rw [Nat.cast_add, Nat.cast_add]
        linarith
      exact_mod_cast this
    obtain ⟨A, hA⟩ := hdev o B hn
    refine le_trans (Pouw.pr_mono (P' := PassesAt N P.sid (P.pub o).root (P.W o) A) ?_) ?_
    · rintro x ⟨ha, hb, hh⟩
      exact hA x (fun t ht => Nat.le_floor ((Nat.cast_le.2 (r.mem_le_held ht)).trans hh.le))
        (certified_ofAt P pt ha hb)
    · have hfin : ENNReal.ofReal (Params.δ : ℝ) + Params.εMax ≠ ⊤ :=
        ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, εMax_ne_top⟩
      calc Pouw.pr (PassesAt N P.sid (P.pub o).root (P.W o) A)
          = (Pous.pr (PassesAt N P.sid (P.pub o).root (P.W o) A)).toReal := Accounting.pr_eq_toReal _
        _ ≤ (ENNReal.ofReal (Params.δ : ℝ) + Params.εMax).toReal :=
          ENNReal.toReal_mono hfin (Pous.EndToEnd.SecurityProofs.endToEndRingOfRoundSecureWide p ε₀ D Q L hnum hW N
            h1 h2 hH _ _ _ A)
        _ = _ := by rw [ENNReal.toReal_add ENNReal.ofReal_ne_top εMax_ne_top, ENNReal.toReal_ofReal δ_nonneg]
  · rw [Pouw.pr_eq_zero fun x hx => hpos (lt_of_le_of_lt (Nat.cast_nonneg _) hx.2.2)]
    exact add_nonneg δ_nonneg ENNReal.toReal_nonneg

theorem periodHeldDemo : PeriodHeldDemo := by
  intro p ε₀ D Q L hnum hW pt N _ hN hH ι α ΩW ΩQ _ _ d r P round S hdev hS o
  have hSN : SAt N = 1060008348186 := Pous.SecurityProofs.stateBound_p2ring_demo N _ _ hN
  exact periodHeldAt p ε₀ D Q L hnum hW pt N (by rw [hN]; norm_num) (by rw [hN]; norm_num) hH ι α ΩW ΩQ d r P
    round S hdev (by rw [hSN, Nat.cast_ofNat]; exact hS) o

end Pous.Period.SecurityProofs
