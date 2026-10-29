import Proofs.Accounting.Compose.ServiceCovers
import Proofs.Accounting.Compose.CircuitWork
import Definitions.Accounting.ComposePeriod

/-!
# The period layer's covers: `ServiceCovers` with a window index

`PeriodCoversStmt` is the one piece of `CircuitWorkPeriod`'s proof (`Proofs.Accounting.ComposePeriod`) stated here,
and proved here as `periodCovers`. The assembly reads it as a hypothesis (`ComposePeriod.circuitWorkPeriod_of_covers`).

It is `CoversProof.serviceCovers`' argument with a window index. Every position, cut and writer of window `w` of the
period layer (`Declared.periodLayer`) is window `w`'s, so a late edge (`DrawnLayer.Late`) stays in its window and
lies in the IR's late cone (`Declared.Reaches`) under `Declared.Wf` and `LayerCommitted`, as `reaches_of_late` shows
for one window. A credited window's live tiles have its own run's modes under `periodMode`, and its dead tiles are
skipped whole. So a live tile of words that isn't good is reached by a tile its window's run didn't pay for: either a
live one, whose `liveCone` the window's round charges and which holds the tile's live words, or a dead one, which
`DeadConeDead` says reaches no live words.

The proof reads the one-window layer (`Declared.layer`) at any run-dependent half (`CoversProof.anyRun`): window `w`'s
positions, cuts, coins and writers are that layer's tagged by `w`, so a late edge of the period layer is one of that
layer's inside its window (`CoversProof.late_one`), and `Compose.CoversProof.reaches_of_rtg` and
`Compose.WorkProof.writerReads`, which read only the static half, give the cone and the writers.
-/

namespace Accounting.SecurityProofs.ComposePeriod

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

open Classical in
/-- **The period layer's covers on a credited window's live tiles** (proved as `periodCovers`): the period layer's
writers read their positions (`DrawnLayer.WriterReads`), and on every path `x`, in every credited window `w` whose dead
tiles' cones hold no live words (`DeadConeDead`), the live words of the tiles that aren't good under the period's modes
(`periodMode`) are at most `N_w` times the `liveCone` of the tiles window `w`'s run didn't pay for, at its rounds'
coins. The right side is window `w`'s round's deficiency in `periodGame` at weights `liveCone` and costs `1, 1`. -/
def PeriodCoversStmt : Prop :=
  ∀ {ι K C Ξ V : Type} (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (pr : Outcomes C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V),
    Cd.Wf → LayerCommitted Cd →
    (Cd.periodLayer hr hK pr).WriterReads ∧
      ∀ (x : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n), pr.Credited x w →
        DeadConeDead Sc Cd Np (Wp.liveAt x w) →
        (∑ t : Fin Np, if (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t then (0 : ℝ)
          else (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ)) ≤
          (∑ u : Fin Np, if (Wp.windowAt x w).modeAt (tileStage Cd Np) (coinAt x w).1 u = .paid then 0
            else liveCone Sc Cd Np (Wp.liveAt x w) u) * (wordsOf Sc Cd : ℝ)

namespace CoversProof

/-- A run-dependent half of the one-window layer (`Declared.layer`), any one: the lemmas this file reuses read only
the layer's static half. -/
def anyRun (C B : Type) (m s Np : ℕ) : ProofRun C B m s Np Unit :=
  ⟨fun _ _ _ => True, fun _ _ _ _ => (), fun _ _ _ _ => True, fun _ _ _ _ => True⟩

section Layer

variable {ι C Ξ V : Type} (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
  (pr : Outcomes C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V)

theorem periodLayer_src (ℓ : Fin n × ℕ) : (Cd.periodLayer hr hK pr).src ℓ =
    if h : ℓ.2 < Cd.dag.length ∧ Cd.binding ℓ.2 ≠ some .coin ∧ Cd.unit ℓ.2 < Np then
      some (ℓ.1, ⟨Cd.unit ℓ.2, h.2.2⟩) else none := rfl

/-- **A late edge of the period layer stays in its window**, and is the one-window layer's edge there. -/
theorem late_one {u v : Fin n × Fin Np} (h : (Cd.periodLayer hr hK pr).Late u v) :
    u.1 = v.1 ∧ (Cd.layer hr hK (anyRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np)).Late (0, u.2) (0, v.2) := by
  obtain ⟨k, hsrc, hcut⟩ := h
  rw [periodLayer_src] at hsrc
  split at hsrc
  · rename_i hs
    obtain rfl := Option.some.inj hsrc
    refine ⟨rfl, k, (Accounting.SecurityProofs.Compose.CoversProof.src_eq Cd hr hK
      (anyRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np) _).trans (dite_eq_left hs), ?_⟩
    rcases hcut with h0 | h1
    · exact absurd h0 (lt_irrefl _)
    · exact Or.inr ⟨rfl, h1.2⟩
  · cases hsrc

theorem rtg_one {u v : Fin n × Fin Np} (h : Relation.ReflTransGen (Cd.periodLayer hr hK pr).Late u v) :
    u.1 = v.1 ∧ Relation.ReflTransGen
      (Cd.layer hr hK (anyRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np)).Late (0, u.2) (0, v.2) := by
  induction h with
  | refl => exact ⟨rfl, Relation.ReflTransGen.refl⟩
  | tail _ hbc ih =>
    obtain ⟨h1, h2⟩ := late_one Cd hr hK pr hbc
    exact ⟨ih.1.trans h1, ih.2.tail h2⟩

/-- **The period layer's late cone lies in its window and in the IR's late cone**: under `Declared.Wf` and
`LayerCommitted`. -/
theorem periodLayer_cone (hw : Cd.Wf) (hc : LayerCommitted Cd) {u v : Fin n × Fin Np}
    (h : Relation.ReflTransGen (Cd.periodLayer hr hK pr).Late u v) : u.1 = v.1 ∧ Cd.Reaches u.2.val v.2.val :=
  ⟨(rtg_one Cd hr hK pr h).1, Accounting.SecurityProofs.Compose.CoversProof.reaches_of_rtg Cd hr hK
    (anyRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np) hw hc (rtg_one Cd hr hK pr h).2⟩

theorem periodLayer_writerReads : (Cd.periodLayer hr hK pr).WriterReads := by
  intro ℓ u hu
  rw [periodLayer_src] at hu
  split at hu
  · rename_i h
    obtain rfl := Option.some.inj hu
    obtain ⟨k, hk⟩ := Accounting.SecurityProofs.Compose.WorkProof.writerReads Cd hr hK
      (anyRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np) ℓ.2 (0, ⟨Cd.unit ℓ.2, h.2.2⟩)
      ((Accounting.SecurityProofs.Compose.CoversProof.src_eq Cd hr hK _ ℓ.2).trans (dite_eq_left h))
    exact ⟨k, Prod.ext rfl hk⟩
  · cases hu

end Layer

/-- A tile's live words are among the circuit's. -/
theorem liveWords_le_wordsOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool)
    (t : Fin Np) : liveWords Sc Cd Np live t ≤ wordsOf Sc Cd := by
  unfold liveWords
  split
  · exact Accounting.SecurityProofs.Compose.CoversProof.tileWords_le_wordsOf Sc Cd Np t
  · exact Nat.zero_le _

open Classical in
/-- **The live words a tile's late cone reaches weigh at most `N_w` times its `liveCone`** (`0` when `N_w = 0`). -/
theorem sum_reaches_le {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool)
    (u : Fin Np) :
    (∑ t : Fin Np, if Cd.Reaches u.val t.val then (liveWords Sc Cd Np live t : ℝ) else 0) ≤
      liveCone Sc Cd Np live u * (wordsOf Sc Cd : ℝ) := by
  unfold liveCone
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun t _ => ?_
  by_cases hR : Cd.Reaches u.val t.val
  · rw [ite_eq_left hR, ite_eq_left hR]
    rcases Nat.eq_zero_or_pos (wordsOf Sc Cd) with h0 | hpos
    · have h00 : liveWords Sc Cd Np live t = 0 := Nat.le_zero.mp (h0 ▸ liveWords_le_wordsOf Sc Cd Np live t)
      simp [h0, h00]
    · exact (div_mul_cancel₀ _ (Nat.cast_pos.mpr hpos).ne').ge
  · simp [hR]

open Classical in
/-- **A share that charges each tile the run didn't pay for its `liveCone` covers**: `covers_of_cone`'s count at live
words, when each live tile of words that isn't good is reached by a tile whose mode isn't paid. -/
theorem sum_not_good_le {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool)
    (md : Fin Np → Mode) (G : Fin Np → Prop)
    (h : ∀ t, ¬ G t → liveWords Sc Cd Np live t ≠ 0 → ∃ u, md u ≠ .paid ∧ Cd.Reaches u.val t.val) :
    (∑ t : Fin Np, if G t then (0 : ℝ) else (liveWords Sc Cd Np live t : ℝ)) ≤
      (∑ u : Fin Np, if md u = .paid then 0 else liveCone Sc Cd Np live u) * (wordsOf Sc Cd : ℝ) := by
  have key : ∀ t : Fin Np, (if G t then (0 : ℝ) else (liveWords Sc Cd Np live t : ℝ)) ≤
      ∑ u : Fin Np, if md u = .paid then (0 : ℝ) else
        (if Cd.Reaches u.val t.val then (liveWords Sc Cd Np live t : ℝ) else 0) := by
    intro t
    have hnn : 0 ≤ ∑ u : Fin Np, (if md u = .paid then (0 : ℝ) else
        (if Cd.Reaches u.val t.val then (liveWords Sc Cd Np live t : ℝ) else 0)) :=
      Finset.sum_nonneg fun u _ => by split_ifs <;> simp
    split_ifs with hg
    · exact hnn
    · by_cases ht : liveWords Sc Cd Np live t = 0
      · have h0 : (liveWords Sc Cd Np live t : ℝ) = 0 := by rw [ht, Nat.cast_zero]
        exact h0.le.trans hnn
      obtain ⟨u, hu, hR⟩ := h t hg ht
      calc (liveWords Sc Cd Np live t : ℝ) = (if md u = .paid then (0 : ℝ) else
            (if Cd.Reaches u.val t.val then (liveWords Sc Cd Np live t : ℝ) else 0)) := by
            rw [ite_eq_right hu, ite_eq_left hR]
        _ ≤ _ := Finset.single_le_sum (f := fun u => if md u = .paid then (0 : ℝ) else
              (if Cd.Reaches u.val t.val then (liveWords Sc Cd Np live t : ℝ) else 0))
            (fun u _ => by split_ifs <;> simp) (Finset.mem_univ u)
  calc (∑ t : Fin Np, if G t then (0 : ℝ) else (liveWords Sc Cd Np live t : ℝ))
      ≤ ∑ t : Fin Np, ∑ u : Fin Np, (if md u = .paid then (0 : ℝ) else
          (if Cd.Reaches u.val t.val then (liveWords Sc Cd Np live t : ℝ) else 0)) :=
        Finset.sum_le_sum fun t _ => key t
    _ = ∑ u : Fin Np, ∑ t : Fin Np, (if md u = .paid then (0 : ℝ) else
          (if Cd.Reaches u.val t.val then (liveWords Sc Cd Np live t : ℝ) else 0)) := Finset.sum_comm
    _ ≤ ∑ u : Fin Np, (if md u = .paid then 0 else liveCone Sc Cd Np live u) * (wordsOf Sc Cd : ℝ) := by
        refine Finset.sum_le_sum fun u _ => ?_
        split_ifs
        · simp
        · exact sum_reaches_le Sc Cd Np live u
    _ = (∑ u : Fin Np, if md u = .paid then 0 else liveCone Sc Cd Np live u) * (wordsOf Sc Cd : ℝ) :=
        (Finset.sum_mul ..).symm

end CoversProof

open Classical in
/-- `PeriodCoversStmt`: the writers from the one-window layer's (`Compose.WorkProof.writerReads`), and the covers by
`CoversProof.sum_not_good_le`. A tile that isn't good has a tile of its period cone that isn't paid; that cone lies in
its window and in the IR's (`CoversProof.periodLayer_cone`), and in a credited window a tile that isn't paid is dead,
so reaches no live words (`DeadConeDead`), or live and not paid in the window's run. -/
theorem periodCovers : PeriodCoversStmt := by
  intro ι K C Ξ V Sc Cd n Np Kd hr hK Wp pr hw hc
  refine ⟨CoversProof.periodLayer_writerReads Cd hr hK pr, fun x w hx hdead => ?_⟩
  refine CoversProof.sum_not_good_le Sc Cd Np (Wp.liveAt x w)
    ((Wp.windowAt x w).modeAt (tileStage Cd Np) (coinAt x w).1)
    (fun t => (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t) fun t hg ht => ?_
  obtain ⟨u, hu, hb⟩ : ∃ u, Relation.ReflTransGen (Cd.periodLayer hr hK pr).Late u (w, t) ∧
      periodMode Cd Wp pr x u ≠ .paid :=
    Classical.byContradiction fun hno => hg fun u hu => Classical.byContradiction fun hb => hno ⟨u, hu, hb⟩
  obtain ⟨h1, hR⟩ := CoversProof.periodLayer_cone Cd hr hK pr hw hc hu
  refine ⟨u.2, fun hp => ?_, hR⟩
  cases hl : Wp.liveAt x w u.2 with
  | false => exact ht (hdead u.2 t hl hR)
  | true =>
    have h1' : u.1 = w := h1
    refine hb ?_
    unfold periodMode
    rw [h1']
    split
    · exact hp
    · rename_i hn
      exact (hn ⟨hx, hl⟩).elim

end Accounting.SecurityProofs.ComposePeriod
