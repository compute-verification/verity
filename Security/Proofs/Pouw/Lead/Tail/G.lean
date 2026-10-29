import Proofs.Pouw.Prob
import Proofs.Pouw.Lead.Compose.G
import Proofs.Pouw.Lead.Tail.Table
import Proofs.Pouw.Lead.Tail.Repeats

/-!
# The lead's tail form from its pieces

`G_tail`'s proof, at each coin value in its event: as in `Compose.G`, the egress-free twin (`OutsourceWithinEgressLead`)
is accepted and hashed no collision there, so `Compose.perCoin` and the twin's saving give
`32 · (honestRegs − tileWords · (repeats + tableHits) − networkGap) ≤ work`. Work below
`32 · (honestRegs − gapFixed)` then needs a table hit of the twin's or `tileWords · repeats` above the coin-branch mean
and slack, the head and `p*` being nonnegative. The first is within `tableTail` (`Tail.table_tail`, on the twin), the
second within `coinTail` (its supremum over the data).

`tailAt`: `tableTail` at `2^30` tiles is `2^−41` and a term under `2^−100`, and `coinTail` at most `2^−42`
(`Tail.coinTail_le`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `coinTail` bounds each call's coin-branch tail, at its data. -/
theorem pr_repeats_le_coinTail (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    Pouw.pr (fun ω : Coins L => coinBranchMean L + coinSlack L < tileWords * (repeats L κ ω : ℚ)) ≤ coinTail L :=
  le_ciSup (f := fun κ : Fin L.m → Fin L.S → Fin L.n → ℤ =>
    Pouw.pr fun ω : Coins L => coinBranchMean L + coinSlack L < tileWords * (repeats L κ ω : ℚ))
    ⟨1, by rintro _ ⟨κ', rfl⟩; exact Pouw.pr_le_one _⟩ κ

/-- **The lead's work guarantee, tail form** (`Pouw.Guarantees.Lead.G_tail`). -/
theorem G_tail : Pouw.Guarantees.Lead.G_tail := by
  intro pr H T m n K κ stream C isA can Real hC h32 hw1 htile hone hcard hsha hhon hout σ hcan
  obtain ⟨σ', hσ', -, htwin⟩ := hout σ (hcard σ hcan)
  have htw : (tileWords : ℚ) = 128 := by norm_num [tileWords]
  have hβ : 0 ≤ betaLead (lead T m n K) C := by
    unfold betaLead nMax wMac honestRegs wordsBlocks wordsRegs aBlocks shaWordsRegs shaARegs
    positivity
  have hnet : 32 * networkGap (lead T m n K) C = betaLead (lead T m n K) C * (lead T m n K).callBits := by
    unfold networkGap
    ring
  have hhead : 0 ≤ headGap (lead T m n K) C := by unfold headGap; positivity
  have hps : 0 ≤ pStarGap (lead T m n K) := by unfold pStarGap stockRegs; positivity
  have hsub : ∀ ω, (Accepted (lead T m n K) H C κ σ ω ∧ ¬ Break (lead T m n K) H C κ σ ω ∧
        (σ.work pr ω : ℚ) < 32 * (honestRegs (lead T m n K) C - gapFixed (lead T m n K) C)) →
      0 < tableHits (lead T m n K) κ σ' stream ω ∨
        coinBranchMean (lead T m n K) + coinSlack (lead T m n K) < tileWords * (repeats (lead T m n K) κ ω : ℚ) := by
    rintro ω ⟨hacc, hnb, hwork⟩
    by_contra hcon
    push Not at hcon
    obtain ⟨h0, hrep⟩ := hcon
    have hacc' : Accepted (lead T m n K) H C κ σ' ω := by
      obtain ⟨he, hr, -, -⟩ := htwin ω
      obtain ⟨hadm, hroot, hbits⟩ := hacc
      refine ⟨hadm, hr.trans hroot, ?_⟩
      rw [he]
      exact (Nat.cast_nonneg _).trans hbits
    have hnb' : ¬ Break (lead T m n K) H C κ σ' ω := fun hb => by
      unfold Break Hash.CollisionIn at hb
      obtain ⟨x, hx, y, hy, hxy⟩ := hb
      have hsub : ∀ z ∈ (σ'.exec ω).hashed ++ (C.msgs H (word (lead T m n K) κ ω) ++ C.aMsgs),
          z ∈ (σ.exec ω).hashed ++ (C.msgs H (word (lead T m n K) κ ω) ++ C.aMsgs) := fun z hz => by
        rcases List.mem_append.1 hz with hz | hz
        · rcases (htwin ω).2.2.1 z hz with h | h
          · exact List.mem_append_left _ h
          · exact List.mem_append_right _ h
        · exact List.mem_append_right _ hz
      apply hnb
      unfold Break Hash.CollisionIn
      exact ⟨x, hsub x hx, y, hsub y hy, hxy⟩
    have h1 := Compose.perCoin pr H T m n K κ stream C isA Real hC h32 hw1 htile hone hsha hhon σ' hσ' ω
      (htwin ω).1 hacc' hnb'
    have h2 := (htwin ω).2.2.2
    have h3 := mul_le_mul_of_nonneg_left hacc.2.2 hβ
    have hth : (tableHits (lead T m n K) κ σ' stream ω : ℚ) = 0 := by exact_mod_cast Nat.le_zero.mp h0
    unfold gapFixed at hwork
    rw [htw, hth] at h1
    rw [htw] at hrep
    linarith
  calc Pouw.pr _ ≤ Pouw.pr fun ω : Coins (lead T m n K) => 0 < tableHits (lead T m n K) κ σ' stream ω ∨
        coinBranchMean (lead T m n K) + coinSlack (lead T m n K) < tileWords * (repeats (lead T m n K) κ ω : ℚ) :=
        Pouw.pr_mono hsub
    _ ≤ _ := Pouw.pr_or_le _ _
    _ ≤ etaLead (lead T m n K) :=
        add_le_add (table_tail rfl rfl (by simp only [Layout.rowBlocks, lead, padRows]; omega)
          (by simp only [Layout.colBlocks, lead, padCols]; omega) κ σ' stream)
          (pr_repeats_le_coinTail _ κ)

/-- **The tail at the three points and decode** (`Pouw.Guarantees.Lead.TailAt`). -/
theorem tailAt : Pouw.Guarantees.Lead.TailAt := by
  intro T n K hn hK L hL htiles
  have hmem : L.mem = 2 ^ 34 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hL
    rcases hL with rfl | rfl | rfl | rfl <;> rfl
  have hcoin := coinTail_le T n K hn hK L hL (htiles.trans (by norm_num))
  have htab : tableTail L ≤ 1 / 2 ^ 41 + 1 / 2 ^ 103 := by
    have ht : (L.rowBlocks * L.colBlocks * L.S : ℚ) ≤ 2 ^ 30 := by exact_mod_cast htiles
    unfold tableTail
    rw [hmem]
    calc (L.rowBlocks * L.colBlocks * L.S : ℚ) * ((((2 ^ 34 : ℕ) : ℚ) + 1) / 2 ^ 168 + ((2 ^ 34 : ℕ) : ℚ) / 2 ^ 105)
        ≤ 2 ^ 30 * ((((2 ^ 34 : ℕ) : ℚ) + 1) / 2 ^ 168 + ((2 ^ 34 : ℕ) : ℚ) / 2 ^ 105) :=
          mul_le_mul_of_nonneg_right ht (by positivity)
      _ ≤ 1 / 2 ^ 41 + 1 / 2 ^ 103 := by norm_num
  have htab' : (tableTail L : ℝ) ≤ ((1 / 2 ^ 41 + 1 / 2 ^ 103 : ℚ) : ℝ) := Rat.cast_le.mpr htab
  push_cast at htab'
  unfold etaLead
  linarith

end Pouw.SecurityProofs.Lead.Tail
