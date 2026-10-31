import Proofs.Pouw.Prob
import Proofs.Pouw.Lead.Compose.G
import Proofs.Pouw.Lead.Tail.Table
import Proofs.Pouw.Lead.Tail.Repeats

/-!
# The lead's tail form from its pieces

`G_tail`'s proof, at each coin value in its event: as in `Compose.G_coins`, the egress-free twin
(`OutsourceWithinEgressLead`) is accepted and hashed no collision there, so `Compose.perCoin` and the twin's saving
give `32 · (honestRegs − tileWords · (repeats + tableHits)) ≤ work + betaLead · egress`. Work below
`32 · (honestRegs − gapFixed)` then needs a table hit of the twin's or `tileWords · repeats` above the coin-branch mean
and slack, the head and `p*` being nonnegative. The first is within `tableTail` (`Tail.table_tail`, on the twin), the
second within `coinTail` (its supremum over the data). That is `G_tail_coins`, over the coin values; `G_tail` reads it
at the coins the certifier releases, which `CertifierCoins` makes `coinsOf` of the entropy, an equivalence
(`Pouw.pr_equiv`).

`tailAt`: `tableTail` at `2^30` tiles and the node's `2^40` stored tiles is `2^−35` and a term under `2^−97`, and
`coinTail` at most `2^−42` (`Tail.coinTail_le`).
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Dimension.H100 Pouw.Dimension.Pointwise

/-- `coinTail` bounds each call's coin-branch tail, at its data. -/
theorem pr_repeats_le_coinTail (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    Pouw.pr (fun ω : Coins L => coinBranchMean L + coinSlack L < tileWords * (repeats L κ ω : ℚ)) ≤ coinTail L :=
  le_ciSup (f := fun κ : Fin L.m → Fin L.S → Fin L.n → ℤ =>
    Pouw.pr fun ω : Coins L => coinBranchMean L + coinSlack L < tileWords * (repeats L κ ω : ℚ))
    ⟨1, by rintro _ ⟨κ', rfl⟩; exact Pouw.pr_le_one _⟩ κ

/-- **The lead's tail form over the coin values**, from `perCoin` on the twin and the two tails. -/
theorem G_tail_coins (pr : Prices100) (H : Hash.Msg → Hash.Digest) (T m n K : ℕ)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (stream : CoinStream (lead T m n K)) (C : Pouw.Lead.Commit (lead T m n K))
    (isA : Hash.Msg → Bool) (can Real : Strategy (lead T m n K) → Prop) (hC : CommitL1w H C isA κ)
    (h32 : H32 pr) (hw1 : W1 pr Real) (htile : TileGrain H C κ stream Real) (hone : OneInstructionMinimal Real)
    (hcard : CardWithinAccelW1 can Real) (hsha : ShaWrites isA Real) (hhon : HonestHashRuled H C κ Real)
    (hout : OutsourceWithinEgressLead pr (betaLead (lead T m n K) C) H C κ Real) (σ : Strategy (lead T m n K))
    (hcan : can σ) :
    Pouw.pr (fun ω : Coins (lead T m n K) =>
        Accepted (lead T m n K) H C κ σ ω ∧ ¬ Break (lead T m n K) H C κ σ ω ∧
          (σ.work pr ω : ℚ) + betaLead (lead T m n K) C * (σ.exec ω).egress <
            32 * (honestRegs (lead T m n K) C - gapFixed (lead T m n K) C)) ≤
      etaLead (lead T m n K) := by
  obtain ⟨σ', hσ', -, haroot', htwin⟩ := hout σ (hcard σ hcan)
  have htw : (tileWords : ℚ) = 128 := by norm_num [tileWords]
  have hhead : 0 ≤ headGap (lead T m n K) C := by unfold headGap; positivity
  have hps : 0 ≤ pStarGap (lead T m n K) := by unfold pStarGap stockRegs; positivity
  have hsub : ∀ ω, (Accepted (lead T m n K) H C κ σ ω ∧ ¬ Break (lead T m n K) H C κ σ ω ∧
        (σ.work pr ω : ℚ) + betaLead (lead T m n K) C * (σ.exec ω).egress <
          32 * (honestRegs (lead T m n K) C - gapFixed (lead T m n K) C)) →
      0 < tableHits (lead T m n K) κ σ' stream ω ∨
        coinBranchMean (lead T m n K) + coinSlack (lead T m n K) < tileWords * (repeats (lead T m n K) κ ω : ℚ) := by
    rintro ω ⟨hacc, hnb, hwork⟩
    by_contra hcon
    push Not at hcon
    obtain ⟨h0, hrep⟩ := hcon
    have hacc' : Accepted (lead T m n K) H C κ σ' ω := by
      obtain ⟨-, hr, -, -⟩ := htwin ω
      obtain ⟨hadm, haroot, hroot⟩ := hacc
      exact ⟨hadm, haroot'.trans haroot, hr.trans hroot⟩
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
    have hth : (tableHits (lead T m n K) κ σ' stream ω : ℚ) = 0 := by
      have : tableHits (lead T m n K) κ σ' stream ω = 0 := by omega
      exact_mod_cast this
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

/-- **The lead's work guarantee, tail form** (`Pouw.Guarantees.Lead.G_tail`): `G_tail_coins` at the coins the
certifier releases, which `CertifierCoins` makes `coinsOf` of the entropy, an equivalence, so the share of entropy
values is the share of coin values. -/
theorem G_tail : Pouw.Guarantees.Lead.G_tail := by
  intro pr H T m n K κ stream C isA can Real Device Ent _ cert coinsOf runC hcert hC h32 hw1 htile hone hcard hsha
    hhon hout σ hcan
  have hrel : ∀ e, released cert runC σ e = coinsOf e := fun e => by
    unfold released
    rw [hcert (σ.aRoot, e)]
    rfl
  have h := G_tail_coins pr H T m n K κ stream C isA can Real hC h32 hw1 htile hone hcard hsha hhon hout σ hcan
  have heq := Pouw.pr_equiv coinsOf (fun ω : Coins (lead T m n K) =>
    Accepted (lead T m n K) H C κ σ ω ∧ ¬ Break (lead T m n K) H C κ σ ω ∧
      (σ.work pr ω : ℚ) + betaLead (lead T m n K) C * (σ.exec ω).egress <
        32 * (honestRegs (lead T m n K) C - gapFixed (lead T m n K) C))
  simp only [hrel]
  exact heq.le.trans h

/-- **The tail at the three points and decode** (`Pouw.Guarantees.Lead.TailAt`). -/
theorem tailAt : Pouw.Guarantees.Lead.TailAt := by
  intro T n K hn hK L hL htiles
  have hmem : L.mem = 2 ^ 40 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hL
    rcases hL with rfl | rfl | rfl | rfl <;> rfl
  have hcoin := coinTail_le T n K hn hK L hL htiles
  have htab : tableTail L ≤ 1 / 2 ^ 35 + 1 / 2 ^ 97 := by
    have ht : (L.rowBlocks * L.colBlocks * L.S : ℚ) ≤ 2 ^ 30 := by exact_mod_cast htiles
    unfold tableTail
    rw [hmem]
    calc (L.rowBlocks * L.colBlocks * L.S : ℚ) * ((((2 ^ 40 : ℕ) : ℚ) + 1) / 2 ^ 168 + ((2 ^ 40 : ℕ) : ℚ) / 2 ^ 105)
        ≤ 2 ^ 30 * ((((2 ^ 40 : ℕ) : ℚ) + 1) / 2 ^ 168 + ((2 ^ 40 : ℕ) : ℚ) / 2 ^ 105) :=
          mul_le_mul_of_nonneg_right ht (by positivity)
      _ ≤ 1 / 2 ^ 35 + 1 / 2 ^ 97 := by norm_num
  have htab' : (tableTail L : ℝ) ≤ ((1 / 2 ^ 35 + 1 / 2 ^ 97 : ℚ) : ℝ) := Rat.cast_le.mpr htab
  push_cast at htab'
  unfold etaLead
  linarith

end Pouw.SecurityProofs.Lead.Tail
