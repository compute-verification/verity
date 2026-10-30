import Proofs.Accounting.RemoteHeld.Line
import Proofs.Properties.RemoteAudit.P2Line
import Proofs.Properties.RemoteAudit.Pooled

/-!
# PoUB-P2's line of record, proved (`Properties.RemoteAudit.P2Line`: `LineHeld`)

v0's argument (`Properties.RemoteAudit.Line.SecurityProofs.lineHeld`) at the P2 subprotocol's credit. At each outcome
where the verifier accepts the record's session (so the trace of record accepts, `ofRecord`) and the certifiers observed
nothing but the verifier's messages entering the node, the receipts' sketch holds and the trace's wrong receipts are at
most the session's pooled bound at the credit's δ (`holdsAnd` of `sketchBound_of_covered` and `Draw.pooledDraw` at the
records read through the coin commitment), so at most every pooled bound the credit reads (`wrong_le`:
`PooledCoversSession` at the record's draw, whose windows an accepted record's credit pools, `pooledWindows_eq`), and
the credit claims no more bits read than the passes read (`readAt_of`):

* each of the `P` passes reads every unit of the store but its wrong receipts
  (`Properties.RemoteAudit.Line.SecurityProofs.read_card`), and they are among the credited passes (`Paced` at
  `recordPace`, `passesBefore_mono`);
* the units the record's passes read are at most `N` each (`units_le`, `summary_facts`), so at most `P·N`;
* the credit's wrong receipts, `⌈wrong⌉` of the pooled bound, are at least it (`le_ceil`), so `q_bytes` is at most the
  units read less the trace's wrong receipts, `UNIT_BYTES` bytes each (`qBytes_le`).

The accepted request takes a pass a window (`minPasses_pos`), which the record's facts read (`summary_facts`).
-/

namespace Properties.RemoteAudit.P2Line.SecurityProofs

open Finset Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Accounting.SecurityProofs.Remote

theorem passesBefore_succ (ws : Array RemoteAudit.Session.WindowSummary) (g : ℕ) :
    passesBefore ws (g + 1) = passesBefore ws g + ws[g]!.lengths.size := by
  simp [passesBefore, List.range_succ]

theorem passesBefore_mono (ws : Array RemoteAudit.Session.WindowSummary) : Monotone (passesBefore ws) :=
  monotone_nat_of_le_succ fun g => by rw [passesBefore_succ]; omega

theorem passesBefore_size (ws : Array RemoteAudit.Session.WindowSummary) :
    passesBefore ws ws.size = (ws.toList.map fun w => w.lengths.size).sum := by
  unfold passesBefore
  congr 1
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
    simp at h1
    rw [getElem!_pos ws i h1]

theorem units_le_list {N : ℕ} :
    ∀ (l : List RemoteAudit.Session.WindowSummary), (∀ w ∈ l, ∀ L ∈ w.lengths, L ≤ N) →
      (l.map fun w => w.lengths.toList.sum).sum ≤ (l.map fun w => w.lengths.size).sum * N
  | [], _ => by simp
  | w :: l, h => by
    simp only [List.map_cons, List.sum_cons, add_mul]
    refine Nat.add_le_add ?_ (units_le_list l fun w' hw' => h w' (List.mem_cons_of_mem _ hw'))
    have := List.sum_le_length_nsmul w.lengths.toList N fun L hL => h w List.mem_cons_self L (by simpa using hL)
    simpa using this

/-- The units a record's passes read are at most `N` for each of its `P` passes. -/
theorem units_le {ws : Array RemoteAudit.Session.WindowSummary} {N : ℕ} (h : ∀ w ∈ ws.toList, ∀ L ∈ w.lengths, L ≤ N) :
    ((ws.map fun w => w.lengths.toList.sum).toList.sum) ≤ passesBefore ws ws.size * N := by
  rw [passesBefore_size, Array.toList_map]
  exact units_le_list _ h

/-- The P2 credit's wrong receipts, `⌈wrong⌉` of the pooled bound, are at least it. -/
theorem le_ceil (q : ℚ) : (q : ℝ) ≤ ((q.num.toNat + q.den - 1) / q.den : ℕ) := by
  have hd : 0 < q.den := q.den_pos
  set k := (q.num.toNat + q.den - 1) / q.den
  have hk : q.num.toNat ≤ k * q.den := by
    have h := Nat.lt_div_mul_add (a := q.num.toNat + q.den - 1) hd
    generalize (q.num.toNat + q.den - 1) / q.den * q.den = m at h
    generalize q.num.toNat = a at h
    have : k * q.den = (q.num.toNat + q.den - 1) / q.den * q.den := rfl
    omega
  have hq : q ≤ (k : ℚ) := by
    rw [← Rat.num_div_den q, div_le_iff₀ (by exact_mod_cast hd)]
    have : (q.num : ℚ) ≤ (q.num.toNat : ℚ) := by exact_mod_cast Int.self_le_toNat q.num
    exact this.trans (by exact_mod_cast hk)
  exact_mod_cast hq

/-- What the P2 credit credits is at most `n` units, `UNIT_BYTES` bytes each, where the units its passes read are at
most `n` plus the wrong receipts of each pooled bound it reads. -/
theorem qBytes_le {sm : RemoteAudit.Session.Summary} {n : ℕ}
    (h : ∀ b, RemoteAudit.Pooled.pooled (pooledWindows sm) RemoteAudit.P2Session.DELTA = .ok b →
      (((sm.windows.map fun w => w.lengths.toList.sum).toList.sum : ℕ) : ℝ) ≤ n + b.wrong) :
    (RemoteAudit.P2Session.credit sm).qBytes ≤ n * RemoteAudit.P2.UNIT_BYTES := by
  unfold RemoteAudit.P2Session.credit
  dsimp only
  split
  · rename_i b hb
    refine Nat.mul_le_mul_right _ ?_
    have h1 := h b (by rw [pooledWindows]; simpa [Array.map_map] using hb)
    have h2 := le_ceil b.wrong
    generalize (b.wrong.num.toNat + b.wrong.den - 1) / b.wrong.den = K at h2 ⊢
    generalize hu : (sm.windows.map fun w => w.lengths.toList.sum).toList.sum = u at h1
    have h3 : (u : ℝ) ≤ ((n + K : ℕ) : ℝ) := by rw [Nat.cast_add]; linarith
    have h4 := (Nat.cast_le (α := ℝ)).1 h3
    have hu' : ((sm.windows.map fun w => (w.lengths.toList.sum, w.drawn)).map (·.1)).toList.sum = u := by
      rw [← hu, Array.map_map]; rfl
    rw [hu']
    omega
  · simp

/-- On an accepting outcome of a trace over the store's units whose passes plan every unit, where the receipts' sketch
holds and the trace's wrong receipts are at most every pooled bound the record's credit reads, the credit claims no
more bits read than its `P` passes read: each pass reads every unit but its wrong receipts
(`Properties.RemoteAudit.Line.SecurityProofs.read_card`), the passes are among the credited ones (`Paced`), and the
units the record's passes read are at most `N` each (`summary_facts`). -/
theorem readAt_of {Ω : Type} {T : Trace Ω} {node : Audited} {rec : Ω → Replay} (hm : 1 ≤ node.params.minPasses)
    (hp : Paced T (recordPace node rec)) (hcov : Covers T) (hN : T.blocks = node.blocks) {ω : Ω} (ha : T.accepts ω)
    (hk : ∀ p < T.passes, ∀ j ∈ T.correct ω p, ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ j ∈ T.present ω t)
    (hw : ∀ sm, (rec ω).summary node = some sm → ∀ b,
      RemoteAudit.Pooled.pooled (pooledWindows sm) RemoteAudit.P2Session.DELTA = .ok b → (T.wrong ω : ℝ) ≤ b.wrong) :
    ReadAt T ω node (rec ω) := by
  intro sm hsm
  obtain ⟨s, -, -, -, hW, hsW, -, hwin⟩ := summary_facts hm hsm
  have hws : (rec ω).windows node = sm.windows := by unfold Replay.windows; rw [hsm]
  have hPle : passes sm ≤ T.passes := by
    obtain ⟨-, -, hfP, -⟩ := hp ω ha
    have hfP' : passesBefore ((rec ω).windows node) (node.windows - 1 + 1) ≤ T.passes := hfP
    rw [hws] at hfP'
    exact le_trans (passesBefore_mono _ (by omega)) hfP'
  set P := passes sm with hP
  have h1 : ∑ p ∈ range P, ((T.read ω p).card + (T.plan ω p \ T.correct ω p).card) = P * T.blocks := by
    rw [Finset.sum_const_nat fun p hp' => Line.SecurityProofs.read_card hcov ha
      (lt_of_lt_of_le (Finset.mem_range.1 hp') hPle) (hk p (lt_of_lt_of_le (Finset.mem_range.1 hp') hPle)),
      Finset.card_range]
  have h2 : ∑ p ∈ range P, (T.plan ω p \ T.correct ω p).card ≤ T.wrong ω :=
    Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 hPle)
  have hsum : P * T.blocks ≤ T.reads ω P + T.wrong ω := by
    rw [Finset.sum_add_distrib] at h1
    unfold Trace.reads
    rw [← h1]
    exact Nat.add_le_add_left h2 _
  have hus : (sm.windows.map fun w => w.lengths.toList.sum).toList.sum ≤ P * node.blocks := by
    refine units_le fun w hw' L hL => ?_
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hw'
    simp only [Array.length_toList] at hi
    have hi' : i < s.windows.size := by omega
    have := (hwin i hi').2.2.2
    rw [getElem!_pos sm.windows i hi] at this
    exact this L (by simpa using hL)
  have hq := qBytes_le (sm := sm) (n := T.reads ω P) fun b hb => by
    have hb' := hw sm hsm b hb
    have e1 : ((P * T.blocks : ℕ) : ℝ) ≤ ((T.reads ω P + T.wrong ω : ℕ) : ℝ) := by exact_mod_cast hsum
    have e2 : (((sm.windows.map fun w => w.lengths.toList.sum).toList.sum : ℕ) : ℝ) ≤ ((P * node.blocks : ℕ) : ℝ) := by
      exact_mod_cast hus
    rw [hN] at e1
    push_cast at e1 e2 ⊢
    linarith
  have hq' : ((RemoteAudit.P2Session.credit sm).qBytes : ℝ) ≤ (T.reads ω P : ℝ) * RemoteAudit.P2.UNIT_BYTES := by
    exact_mod_cast hq
  linarith

/-- On an accepted record, the windows the P2 credit pools are the record's draw's at `ω`: each window's units read
are its receipts, below `RECEIPT_LIMIT` so the draw's size, and its draws `min(s, U_t)` (`summary_facts`). -/
theorem pooledWindows_eq {κ : ℕ} {X : Type} {node : Audited} {rec : Coins node.draws κ X → Replay}
    {correct : Fin node.draws → Coins node.draws κ X → Finset ℕ} {ω : Coins node.draws κ X}
    (hm : 1 ≤ node.params.minPasses) {sm : RemoteAudit.Session.Summary} (hsm : (rec ω).summary node = some sm) :
    pooledWindows sm = Array.ofFn fun t =>
      (((recordDraw node rec correct).size t ω : ℕ), (recordDraw node rec correct).draws t ω) := by
  obtain ⟨s, hrun, -, -, hW, hsW, -, hwin⟩ := summary_facts hm hsm
  have hreach := reached_of_run hrun
  apply Array.ext
  · simp [pooledWindows, hW, Audited.draws, Audited.windows]
  · intro i h1 h2
    simp only [pooledWindows, Array.size_map] at h1
    have hi : i < s.windows.size := by omega
    obtain ⟨hl, hlim, hd, -⟩ := hwin i hi
    rw [getElem!_pos sm.windows i h1] at hl hd
    have hrec : (rec ω).receiptsAt node i = s.windows[i].receipts := by
      unfold Replay.receiptsAt
      rw [hreach, Array.getElem?_eq_getElem hi]
      rfl
    have hmin : min ((rec ω).receiptsAt node i) Line.receiptMax = s.windows[i].receipts := by
      rw [hrec]
      unfold Line.receiptMax
      omega
    simp only [pooledWindows, Array.getElem_map, Array.getElem_ofFn, Session.draws, recordDraw, hmin, hl, hd]

/-- On an accepted record, the trace's wrong receipts below the record's draw's pooled bound at the P2 credit's `δ` are
below every pooled bound the credit reads (`Properties.RemoteAudit.PooledCoversSession` at the record's draw). -/
theorem wrong_le {κ : ℕ} {X : Type} {node : Audited} {T : Trace (Coins node.draws κ X)}
    {rec : Coins node.draws κ X → Replay} {correct : Fin node.draws → Coins node.draws κ X → Finset ℕ}
    {ω : Coins node.draws κ X} (hm : 1 ≤ node.params.minPasses)
    (hw : (T.wrong ω : ℝ) ≤ (recordDraw node rec correct).pooled (RemoteAudit.P2Session.DELTA : ℝ) ω) :
    ∀ sm, (rec ω).summary node = some sm → ∀ b,
      RemoteAudit.Pooled.pooled (pooledWindows sm) RemoteAudit.P2Session.DELTA = .ok b → (T.wrong ω : ℝ) ≤ b.wrong := by
  intro sm hsm b hb
  rw [pooledWindows_eq (correct := correct) hm hsm] at hb
  exact hw.trans (Properties.RemoteAudit.pooledCoversSession node.draws Line.receiptMax _ κ X
    (recordDraw node rec correct) ω RemoteAudit.P2Session.DELTA b hb)

/-- `δ`, 2^-40, is nonnegative in the reals. -/
theorem delta_nonneg : (0 : ℝ) ≤ (RemoteAudit.P2Session.DELTA : ℝ) := by
  unfold RemoteAudit.P2Session.DELTA
  push_cast
  positivity

/-- **PoUB-P2's line of record, in bits read** (`LineHeld`), proved: v0's argument (`Properties.RemoteAudit.Line`'s
`lineHeld`) with the credit's ceiling of the pooled bound in place of the line's W, the accepted request taking a pass a
window (`minPasses_pos`). -/
theorem lineHeld : LineHeld := by
  intro node κ X Cm _ prf rest recs correct com T R δs δnet r F O I hp hcov hk hC hI hL hR hCnt
  have hH := holdsAnd (sketchBound_of_covered hk hC hI)
    (Draw.pooledDraw prf rest F com (ofRecord T node r) R delta_nonneg Line.SecurityProofs.receiptMax_lt hL hR hCnt)
  refine le_trans (Pouw.pr_mono fun ω hω => ⟨hω.1, fun h => hω.2 fun hq => ?_⟩) hH
  have hm := minPasses_pos hω.1.1
  exact readAt_of hm hp hcov rfl hω.1 (h.1 hq) (wrong_le (correct := fun t ω => correct (com ω) t ω) hm h.2)

end Properties.RemoteAudit.P2Line.SecurityProofs
