import Properties.NetworkPeriod.Placed
import Proofs.Properties.NetworkScan

/-!
# An accepted scan's proof blocks each commit at most `max(b_ing, b_eg)` bits

`ScanWindowsPlaced` reads a window's statement only through `StmtBits`: its gates are some members' joint program
(`jointOf`), each member's circuit file one `fs` lists at a template, and each proof block a member's unit's
(`Bits.blockOf`). The bound is per block, from its member's file alone, then summed over an incorrect set:

* `committed_blockOf` (Lemma A, the file): a unit's block commits at most `16 · (paramWords + retWords)` bits of the
  Definition its file is checked against (`Bits.check`), reading no input padding (`Bits.readsWhy` at no parameter):
  its committed gates are the source gates it owns (`insOf`, at most 16 a port word: a port's bits past its first word
  are the zero, `insOf_length`) and its returned copy rows (16 a word of the return, `units_committed`); no gate
  outside it reads one of its rows (`Lowered.reads_jointUnit`).
* `words_le_terms` (Lemma A, the scan): each template's `16 · (paramWords + retWords)` is at most
  `max sc.ingress sc.egress`, `scanProg`'s terms (`vuBits`): bucket 0's and the later ingress buckets' at `ingress`
  (the later template returns bucket 0's return), the egress template's at `egress`.
* `committed_le_terms` (Lemma A): the two, at a member's file at a template index.
* `faultyBits_le` (Lemma B): an incorrect set of at most `b` blocks, each committing at most `β`, commits at most
  `b · β`.
-/

namespace Properties.NetworkPeriod.Placed

open SampledProofs.Bits
open Properties.NetworkScan
open NetworkCheck.Period (Scan Schedule scanProg templateNodes paramWords vuBits)
open NetworkCheck.Circuit (templateDef)

/-! ## Lengths -/

theorem length_le_of_nodup {α : Type} {l₁ l₂ : List α} (hd : l₁.Nodup) (h : l₁ ⊆ l₂) : l₁.length ≤ l₂.length := by
  obtain ⟨l, hp, hs⟩ := hd.subperm h
  rw [← hp.length_eq]
  exact hs.length_le

theorem length_flatMap_range_le {α : Type} (f : ℕ → List α) {k : ℕ} (h : ∀ j, (f j).length ≤ k) :
    ∀ n, ((List.range n).flatMap f).length ≤ n * k
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, List.length_append, Nat.succ_mul]
    have := length_flatMap_range_le f h n
    have := h n
    omega

theorem paramWords_eq (ps : Array (String × Flock.Program.Ty)) :
    paramWords ps = (ps.toList.map (wordsOf ·.2)).sum := by
  unfold paramWords
  rw [← Array.foldl_toList]
  suffices h : ∀ (l : List (String × Flock.Program.Ty)) (a : ℕ),
      l.foldl (fun n p => n + wordsOf p.2) a = a + (l.map (wordsOf ·.2)).sum by
    rw [h, Nat.zero_add]
  intro l
  induction l with
  | nil => intro a; simp
  | cons x l ih => intro a; rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]; omega

/-! ## A unit's lowered gates -/

theorem mem_seg {kp : KeyProg} {u g y : ℕ}
    (h : y ∈ if unitOf kp g == some u then (List.range (gateSize kp g)).map (off kp g + ·) else []) :
    unitOf kp g = some u ∧ off kp g ≤ y ∧ y < off kp g + gateSize kp g := by
  split at h
  · rename_i hu
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 h
    exact ⟨beq_iff_eq.1 hu, Nat.le_add_right _ _, by have := List.mem_range.1 hj; omega⟩
  · cases h

theorem mem_unitGates {kp : KeyProg} {u p : ℕ} :
    p ∈ unitGates kp u ↔ ∃ g < kp.size, unitOf kp g = some u ∧ off kp g ≤ p ∧ p < off kp g + gateSize kp g := by
  unfold unitGates
  rw [List.mem_flatMap]
  constructor
  · rintro ⟨g, hg, hp⟩
    exact ⟨g, List.mem_range.1 hg, mem_seg hp⟩
  · rintro ⟨g, hg, hu, h1, h2⟩
    refine ⟨g, List.mem_range.2 hg, ?_⟩
    rw [ite_eq_left (beq_iff_eq.2 hu)]
    exact List.mem_map.2 ⟨p - off kp g, List.mem_range.2 (by omega), by omega⟩

/-- A unit's lowered gates are each its gates' once: two gates' never meet (`Lowered.seg_unique`). -/
theorem unitGates_nodup (kp : KeyProg) (u : ℕ) : (unitGates kp u).Nodup := by
  unfold unitGates List.Nodup
  rw [List.pairwise_flatMap]
  refine ⟨fun g _ => ?_, List.Pairwise.imp_of_mem ?_ List.nodup_range⟩
  · split
    · exact List.Nodup.map (f := (off kp g + ·)) (fun a b h => Nat.add_left_cancel h) List.nodup_range
    · exact List.Pairwise.nil
  · intro g g' hg hg' hne y hy y' hy' hyy
    obtain ⟨-, h1, h2⟩ := mem_seg hy
    obtain ⟨-, h1', h2'⟩ := mem_seg hy'
    subst hyy
    exact hne (Lowered.seg_unique kp (List.mem_range.1 hg) (List.mem_range.1 hg') h1 h2 h1' h2')

theorem unitOf_some {kp : KeyProg} {g u : ℕ} (h : unitOf kp g = some u) :
    kp.inputs ≤ g ∧ g < kp.size ∧ 0 < kp.perUnit ∧ u = (g - kp.inputs) / kp.perUnit := by
  unfold unitOf at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, not_or] at hc
    exact ⟨by omega, by omega, by omega, (Option.some.inj h).symm⟩

theorem unitGate_of_unitOf {c : Flock.Circuit} {n g u : ℕ} (h : unitOf (keyProg c n) g = some u) :
    Lowered.UnitGate c n u g ∧ (keyProg c n).inputs ≤ g := by
  obtain ⟨h1, h2, -, rfl⟩ := unitOf_some h
  obtain ⟨⟨a, b⟩, hr⟩ := Lowered.row?_some _ h1 h2
  exact ⟨(Lowered.row_unit c hr).1, h1⟩

theorem unitOf_of_unitGate {c : Flock.Circuit} {n g u : ℕ} (h : Lowered.UnitGate c n u g)
    (hg : (keyProg c n).inputs ≤ g) : unitOf (keyProg c n) g = some u := by
  obtain ⟨hu, h⟩ := h
  have hI := Lowered.keyProg_inputs c n
  have hB := Lowered.mul_add_le (B := Lowered.srcs c) (i := Lowered.srcs c) hu (Nat.le_refl _)
  rcases h with ⟨-, h2⟩ | ⟨h1, h2⟩
  · omega
  · have := Lowered.unitOf_row c hu (i := g - (keyProg c n).inputs - u * (keyProg c n).perUnit) (by omega)
    rwa [show (keyProg c n).inputs + (u * (keyProg c n).perUnit +
      (g - (keyProg c n).inputs - u * (keyProg c n).perUnit)) = g by omega] at this

/-- **A unit's lowered gate it commits is returned**: none is an input, and a gate outside the block that reads one is
the unit's (`Lowered.reads_jointUnit`), so one of its rows, in the block, or one of its source gates, which reads
nothing. -/
theorem unit_boundary {ms : Members} {m u p : ℕ} {L : List ℕ} (hm : m < ms.length)
    (hp : p ∈ unitGates (keyProg (ms.getD m default).1 (ms.getD m default).2.size) u)
    (hL : ∀ q ∈ unitGates (keyProg (ms.getD m default).1 (ms.getD m default).2.size) u, base ms m + q ∈ L)
    (hb : (jointOf ms).Boundary L (base ms m + p) = true) : base ms m + p ∈ (jointOf ms).outs := by
  obtain ⟨g, -, hgu, h1, h2⟩ := mem_unitGates.1 hp
  obtain ⟨hU, hgi⟩ := unitGate_of_unitOf hgu
  have hpos : Lowered.JointUnit ms m u (base ms m + p) := ⟨hm, p, rfl, g, hU, h1, h2⟩
  have hpl : p < (memberProg (ms.getD m default)).1.length :=
    Lowered.unitPos_lt_member (ci := ms.getD m default) ⟨g, hU, h1, h2⟩
  unfold Claims.Sampling.Prog.Boundary at hb
  simp only [Bool.or_eq_true, decide_eq_true_eq, List.contains_iff_mem, List.any_eq_true, List.mem_range,
    Bool.and_eq_true] at hb
  rcases hb with (ho | hin) | ⟨g', -, hg'L, hread⟩
  · exact ho
  · exfalso
    rw [Lowered.getElem?_jointOf ms hm hpl, Option.map_eq_some_iff] at hin
    obtain ⟨G, hG, hGi⟩ := hin
    rw [Lowered.bgateAt_input] at hGi
    subst hGi
    have hi := (Lowered.prog_input_iff _ _ p).1 hG
    have ho := Lowered.off_mono (keyProg (ms.getD m default).1 (ms.getD m default).2.size) hgi
    rw [Lowered.off_input _ (Nat.le_refl _)] at ho
    omega
  · exfalso
    rcases hG : (jointOf ms).gates[g']? with _ | G
    · rw [hG] at hread; simp at hread
    · rw [hG] at hread
      have hread' : base ms m + p ∈ G.reads := by simpa using hread
      obtain ⟨-, x, rfl, g'', hU'', h1'', h2''⟩ := Lowered.reads_jointUnit hG hread' hpos
      by_cases hgi'' : (keyProg (ms.getD m default).1 (ms.getD m default).2.size).inputs ≤ g''
      · exact hg'L (hL x (mem_unitGates.2 ⟨g'', Lowered.unitGate_lt_size hU'', unitOf_of_unitGate hU'' hgi'',
          h1'', h2''⟩))
      · have hgi2 : g'' < (keyProg (ms.getD m default).1 (ms.getD m default).2.size).inputs :=
          Nat.lt_of_not_le hgi''
        have hs1 := Lowered.gateSize_input _ hgi2
        have ho := Lowered.off_input _ (Nat.le_of_lt hgi2)
        obtain rfl : g'' = x := by omega
        have hxl : g'' < (memberProg (ms.getD m default)).1.length :=
          Lowered.unitPos_lt_member (ci := ms.getD m default) ⟨g'', hU'', h1'', h2''⟩
        rw [Lowered.getElem?_jointOf ms hm hxl, Option.map_eq_some_iff] at hG
        obtain ⟨G1, hG1, rfl⟩ := hG
        rw [Lowered.bgateAt_reads] at hread'
        have hlow := Lowered.getElem?_prog (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) (Lowered.unitGate_lt_size hU'') (j := 0)
          (by omega)
        rw [Nat.add_zero, ho] at hlow
        unfold memberProg at hG1
        rw [hlow] at hG1
        have hnil : G1.reads = [] := by
          rw [lower, Lowered.row?_input _ hgi2] at hG1
          rcases hκ : consts (ms.getD m default).1 (ms.getD m default).2.size g'' with _ | v <;>
            rw [hκ] at hG1 <;> simp only [List.getElem?_cons_zero, Option.some.injEq] at hG1 <;> subst hG1 <;> rfl
        rw [hnil] at hread'
        simp at hread'

/-- **A unit's lowered gates commit at most 16 bits a word of the return**: each it commits is returned
(`unit_boundary`), an output copy bit of its one instance (`Check.Facts.upv`), of a leaf among the return's words
(`Check.Facts.ret`). -/
theorem units_committed {P : Flock.Program.Prog} {tmpl : String} {d : Flock.Program.Defn} {ms : Members} {m u : ℕ}
    {L : List ℕ} (hm : m < ms.length) (hF : Check.Facts (ms.getD m default).1 P tmpl)
    (hd : P.defs.find? (·.key == tmpl) = some d)
    (hL : ∀ q ∈ unitGates (keyProg (ms.getD m default).1 (ms.getD m default).2.size) u, base ms m + q ∈ L) :
    (((unitGates (keyProg (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)).filter
      fun g => (jointOf ms).Boundary L g).length ≤ 16 * retWords d.ret := by
  have hnd : (((unitGates (keyProg (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)).filter
      fun g => (jointOf ms).Boundary L g).Nodup :=
    ((unitGates_nodup _ u).map (f := (base ms m + ·)) fun a b h => Nat.add_left_cancel h).filter _
  have hsub : (((unitGates (keyProg (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)).filter
      fun g => (jointOf ms).Boundary L g) ⊆ (List.range (retWords d.ret)).flatMap fun j =>
        (List.range 16).filterMap fun t =>
          (outBit (ms.getD m default).1 (ms.getD m default).2.size u 0 j t).map (base ms m + ·) := by
    intro y hy
    rw [List.mem_filter, List.mem_map] at hy
    obtain ⟨⟨p, hp, rfl⟩, hb⟩ := hy
    have ho := unit_boundary hm hp hL hb
    obtain ⟨m', hm', u', k, j, t, x, hx, he⟩ := (Lowered.mem_outs_jointOf_iff ms _).1 ho
    obtain ⟨g, -, hgu, h1, h2⟩ := mem_unitGates.1 hp
    obtain ⟨hU, -⟩ := unitGate_of_unitOf hgu
    have hj1 : Lowered.JointUnit ms m u (base ms m + p) := ⟨hm, p, rfl, g, hU, h1, h2⟩
    have hj2 : Lowered.JointUnit ms m' u' (base ms m + p) := ⟨hm', x, he, Lowered.outBit_unitPos hx⟩
    obtain ⟨rfl, rfl⟩ := Lowered.jointUnit_unique hj1 hj2
    obtain rfl : p = x := by omega
    have hx' := hx
    simp only [outBit, Option.map_eq_some_iff] at hx'
    obtain ⟨g', hg', -⟩ := hx'
    obtain ⟨-, hk, hjs, ht, -⟩ := Lowered.outGate_some hg'
    rw [hF.upv] at hk
    obtain rfl : k = 0 := by omega
    have hjr : j < retWords d.ret := by
      have hs : 0 < (ms.getD m default).1.leavesOut.size := by rw [hF.leaves, hF.upv]; omega
      have h0 : (ms.getD m default).1.leavesOut.getD 0 #[] ∈ (ms.getD m default).1.leavesOut := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs, Option.getD_some]
        exact Array.getElem_mem hs
      rw [← hF.ret d hd _ h0]
      exact hjs
    rw [List.mem_flatMap]
    exact ⟨j, List.mem_range.2 hjr, List.mem_filterMap.2 ⟨t, List.mem_range.2 ht, by rw [hx, Option.map_some]⟩⟩
  refine (length_le_of_nodup hnd hsub).trans ?_
  exact (length_flatMap_range_le _ (k := 16) (fun j => (List.length_filterMap_le _ _).trans (by simp)) _).trans
    (Nat.le_of_eq (Nat.mul_comm _ _))

/-! ## A unit's inputs -/

theorem set_getD {m : Array Bool} {x y : ℕ} :
    (m.set! x true).getD y false = true ↔ m.getD y false = true ∨ (x = y ∧ y < m.size) := by
  rw [Array.set!_eq_setIfInBounds, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds,
    Array.getD_eq_getD_getElem?]
  by_cases hxy : x = y
  · subst hxy
    by_cases hx : x < m.size
    · simp [hx]
    · simp [hx]
  · simp [hxy]

theorem fold_set_size {N : ℕ} : ∀ (r : List ℕ) (m : Array Bool),
    (r.foldl (fun m x => if x < N then m.set! x true else m) m).size = m.size
  | [], _ => rfl
  | x :: r, m => by
    rw [List.foldl_cons, fold_set_size r]
    split <;> simp

theorem fold_set_mono {N y : ℕ} : ∀ (r : List ℕ) (m : Array Bool), m.getD y false = true →
    (r.foldl (fun m x => if x < N then m.set! x true else m) m).getD y false = true
  | [], _, h => h
  | x :: r, m, h => by
    rw [List.foldl_cons]
    apply fold_set_mono r
    try dsimp only
    split
    · exact set_getD.2 (Or.inl h)
    · exact h

theorem fold_set_hit {N y : ℕ} (hy : y < N) : ∀ (r : List ℕ) (m : Array Bool), y ∈ r → y < m.size →
    (r.foldl (fun m x => if x < N then m.set! x true else m) m).getD y false = true
  | [], _, hm, _ => by simp at hm
  | x :: r, m, hm, hs => by
    rw [List.foldl_cons]
    rcases List.mem_cons.1 hm with rfl | hm'
    · apply fold_set_mono r
      try dsimp only
      rw [ite_eq_left hy]
      exact set_getD.2 (Or.inr ⟨rfl, hs⟩)
    · apply fold_set_hit hy r _ hm'
      try dsimp only
      split <;> simp [hs]

theorem fold_rows_mono {N y : ℕ} : ∀ (rs : List (List ℕ)) (m : Array Bool), m.getD y false = true →
    (rs.foldl (fun m r => r.foldl (fun m x => if x < N then m.set! x true else m) m) m).getD y false = true
  | [], _, h => h
  | r :: rs, m, h => by
    rw [List.foldl_cons]
    exact fold_rows_mono rs _ (fold_set_mono r m h)

theorem fold_rows_hit {N y : ℕ} (hy : y < N) : ∀ (rs : List (List ℕ)) (m : Array Bool) (r : List ℕ), r ∈ rs →
    y ∈ r → y < m.size →
      (rs.foldl (fun m r => r.foldl (fun m x => if x < N then m.set! x true else m) m) m).getD y false = true
  | [], _, _, hr, _, _ => by simp at hr
  | r' :: rs, m, r, hr, hy', hs => by
    rw [List.foldl_cons]
    rcases List.mem_cons.1 hr with rfl | hr'
    · exact fold_rows_mono rs _ (fold_set_hit hy r m hy' hs)
    · exact fold_rows_hit hy rs _ r hr' hy' (Nat.lt_of_lt_of_eq hs (fold_set_size r' m).symm)

/-- **A column a computed row reads is a read input column** (`Reads.unitReads_true`'s converse). -/
theorem unitReads_hit {c : Flock.Circuit} {i x : ℕ} (hi : i < (Rows.ofNet c.unit).nComp)
    (hx : x ∈ (Rows.ofNet c.unit).a.getD i [] ++ (Rows.ofNet c.unit).b.getD i [])
    (hxn : x < (Rows.ofNet c.unit).nIn) : (unitReads c).getD x false = true := by
  have hsa : (Rows.ofNet c.unit).a.size = (Rows.ofNet c.unit).nComp := by simp [Rows.ofNet]
  have hsb : (Rows.ofNet c.unit).b.size = (Rows.ofNet c.unit).nComp := by simp [Rows.ofNet]
  unfold unitReads
  try dsimp only
  rw [← Array.foldl_toList]
  have hs : x < (Array.replicate (Rows.ofNet c.unit).nIn false).size := by simpa using hxn
  rw [Array.toList_append]
  rcases List.mem_append.1 hx with ha | hb
  · refine fold_rows_hit hxn _ _ ((Rows.ofNet c.unit).a.getD i []) (List.mem_append_left _ ?_) ha hs
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
    exact Array.mem_def.1 (Array.getElem_mem _)
  · refine fold_rows_hit hxn _ _ ((Rows.ofNet c.unit).b.getD i []) (List.mem_append_right _ ?_) hb hs
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
    exact Array.mem_def.1 (Array.getElem_mem _)

/-- Where each step of the loop body `f` yields, keeping `Q` or meeting an element with `P`, a loop ending in `Q`
started in `Q` or met one. -/
theorem forIn_ne {α β : Type} {f : α → β → Id (ForInStep β)} {Q : β → Prop} {P : α → Prop}
    (hf : ∀ e m, ∃ m', (f e m).run = .yield m' ∧ (Q m' → Q m ∨ P e)) :
    ∀ (l : List α) (m : β), Q (forIn l m f).run → Q m ∨ ∃ e ∈ l, P e
  | [], m, h => by rw [List.forIn_nil] at h; exact Or.inl h
  | e :: l, m, h => by
    obtain ⟨m', hm', hq⟩ := hf e m
    rw [List.forIn_cons, Id.run_bind, hm'] at h
    rcases forIn_ne hf l m' h with h1 | ⟨e', he', hp⟩
    · rcases hq h1 with h2 | h2
      · exact Or.inl h2
      · exact Or.inr ⟨e, List.mem_cons.2 (Or.inl rfl), h2⟩
    · exact Or.inr ⟨e', List.mem_cons_of_mem _ he', hp⟩

theorem set_getElem! {m : Array Bool} {y x : ℕ} (h : (m.set! y true)[x]! = true) : m[x]! = true ∨ y = x := by
  rw [Array.getElem!_eq_getD] at h ⊢
  rcases set_getD.1 h with h1 | ⟨h1, -⟩
  · exact Or.inl h1
  · exact Or.inr h1

/-- **A column a row reads is a port's**: where `readsWhy` finds nothing wrong, each unit input column `x` the reads mark
lies in one of the unit's input ports. -/
theorem readsWhy_held {c : Flock.Circuit} {d : Flock.Program.Defn} {reads : Array Bool} {names : List String}
    (h : readsWhy c d reads names = none) {x : ℕ} (hx : x < min (c.unit.inWords * Flock.WORD) c.unit.constPos)
    (hrx : reads.getD x false = true) :
    ∃ p < (Flock.portCols c.unit.inGroups).size, ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 ≤ x ∧
      x < ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2 := by
  unfold readsWhy at h
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', ← Array.forIn_toList] at h
  split at h
  · cases h
  try simp only [Id.run_bind] at h
  split at h
  · rename_i r hr
    obtain ⟨m, rfl⟩ := Reads.forIn_some hr (by
      intro y s
      try dsimp only
      repeat' split
      all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩)
    cases h
  rename_i hr
  have hb := Reads.forIn_each (by
    intro y s
    try dsimp only
    repeat' split
    all_goals first | exact Or.inl ⟨_, rfl⟩ | exact Or.inr ⟨_, _, rfl⟩) _ hr x (by
    simp only [List.mem_range'_1, Std.Legacy.Range.size]
    omega)
  try dsimp only at hb
  split at hb
  · cases hb
  rename_i hc
  have hrx' : reads[x]! = true := by rw [Array.getElem!_eq_getD]; exact hrx
  rw [hrx', Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hc
  rcases forIn_ne (Q := fun m : Array Bool => m[x]! = true) (P := fun e : ℕ × ℕ => e.1 ≤ x ∧ x < e.1 + e.2)
      (by
        intro e m
        obtain ⟨a0, w⟩ := e
        refine ⟨_, rfl, fun hm => ?_⟩
        rcases forIn_ne (Q := fun m : Array Bool => m[x]! = true) (P := fun t : ℕ => a0 + t = x)
            (by
              intro t m
              refine ⟨_, rfl, fun hm => ?_⟩
              exact set_getElem! hm) _ _ hm with h0 | ⟨t, ht, hxt⟩
        · exact Or.inl h0
        · right
          simp only [List.mem_range'_1, Std.Legacy.Range.size] at ht
          try dsimp only at hxt ⊢
          omega) _ _ hc with h0 | ⟨e, he, hpe⟩
  · exfalso
    try dsimp only at h0
    rw [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, Array.getElem?_replicate] at h0
    split at h0 <;> simp at h0
  · obtain ⟨p, hp, rfl⟩ := List.getElem_of_mem he
    try dsimp only at hpe
    have hp' : p < (Flock.portCols c.unit.inGroups).size := by simpa using hp
    have hg : (Flock.portCols c.unit.inGroups).getD p (0, 0) = (Flock.portCols c.unit.inGroups).toList[p] := by
      rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, List.getElem?_eq_getElem hp, Option.getD_some]
    exact ⟨p, hp', by rw [hg]; exact hpe.1, by rw [hg]; exact hpe.2⟩

theorem fold_contains {f : Std.HashMap (ℕ × ℕ) ℕ → ℕ × ℕ × ℕ → Std.HashMap (ℕ × ℕ) ℕ} {key : ℕ × ℕ}
    (hmono : ∀ m e, m.contains key = true → (f m e).contains key = true)
    (hhit : ∀ m e, (e.1, e.2.1) = key → (f m e).contains key = true) :
    ∀ (l : List (ℕ × ℕ × ℕ)) (m : Std.HashMap (ℕ × ℕ) ℕ),
      (m.contains key = true ∨ ∃ e ∈ l, (e.1, e.2.1) = key) → (l.foldl f m).contains key = true
  | [], _, h => by
    rcases h with h | ⟨e, he, -⟩
    · exact h
    · cases he
  | e :: l, m, h => by
    rw [List.foldl_cons]
    apply fold_contains hmono hhit l
    rcases h with h | ⟨e', he', hk⟩
    · exact Or.inl (hmono m e h)
    · rcases List.mem_cons.1 he' with rfl | he''
      · exact Or.inl (hhit m e' hk)
      · exact Or.inr ⟨e', he'', hk⟩

/-- **A leaf cut's port has a first cut** in `Index.of`'s map. -/
theorem cuts_isSome {c : Flock.Circuit} {k p w : ℕ} (h : (k, p, w) ∈ c.leafCuts) :
    ∃ w', (Index.of c).cuts[(k, p)]? = some w' := by
  apply Option.isSome_iff_exists.1
  rw [Std.HashMap.isSome_getElem?_eq_contains]
  unfold Index.of
  dsimp only
  rw [← Array.foldl_toList]
  refine fold_contains ?_ ?_ _ _ (Or.inr ⟨(k, p, w), Array.mem_def.1 h, rfl⟩)
  · intro m e hm
    obtain ⟨k', p', w'⟩ := e
    dsimp only
    split
    · exact hm
    · rw [Std.HashMap.contains_insert, hm, Bool.or_true]
  · intro m e he
    obtain ⟨k', p', w'⟩ := e
    dsimp only at he ⊢
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    split
    · assumption
    · rw [Std.HashMap.contains_insert, beq_self_eq_true, Bool.true_or]

/-- **A later port's bit past its first word has the zero as its source**: a port of the first group is one word wide
(`Check.Facts.first`), and a later port's source is its leaf cut's (`Check.Facts.cut`), a message bit only below 16. -/
theorem src_lt16 {c : Flock.Circuit} {P : Flock.Program.Prog} {tmpl : String} (hF : Check.Facts c P tmpl) {p τ : ℕ}
    (hp : p < (Flock.portCols c.unit.inGroups).size) (hτ : τ < ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2)
    (hz : srcAt c (Index.of c) 0 p τ ≠ .zero) : τ < 16 := by
  by_cases h1 : p < (c.unit.inGroups.getD 0 default).bits.size
  · have := hF.first p h1
    omega
  · by_contra h16
    obtain ⟨w, hw⟩ := hF.cut p hp (Nat.not_lt.1 h1)
    obtain ⟨w', hw'⟩ := cuts_isSome hw
    apply hz
    simp only [srcAt, h1, ↓reduceIte, hw']
    simp [msgAt, h16]

theorem stack_one {R : Rows} {i col : ℕ} (hi : i < (R.stack 1).nComp)
    (hcol : col ∈ (R.stack 1).a.getD i [] ++ (R.stack 1).b.getD i []) (hlt : col < (R.stack 1).nIn) :
    ∃ i0 < R.nComp, col ∈ R.a.getD i0 [] ++ R.b.getD i0 [] ∧ col < R.nIn := by
  have hN : (R.stack 1).nComp = 1 * (R.nComp + 1) := rfl
  have hI : (R.stack 1).nIn = 1 * R.nIn := rfl
  rw [hN] at hi
  rw [hI] at hlt
  rcases Nat.eq_zero_or_pos i with rfl | hpos
  · exfalso
    have h0 : ∀ f : Array (List ℕ), ((Array.range (1 * (R.nComp + 1))).map (R.stackRow 1 f)).getD 0 [] =
        [1 * R.nIn + 1 * (R.nComp + 1)] := by
      intro f
      have hlt0 : 0 < 1 * (R.nComp + 1) := by omega
      simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, hlt0, ↓reduceIte,
        Option.map_some, Option.getD_some]
      simp [Rows.stackRow]
    have ha : (R.stack 1).a.getD 0 [] = [1 * R.nIn + 1 * (R.nComp + 1)] := h0 R.a
    have hb : (R.stack 1).b.getD 0 [] = [1 * R.nIn + 1 * (R.nComp + 1)] := h0 R.b
    rw [ha, hb] at hcol
    simp at hcol
    omega
  · obtain ⟨i0, rfl⟩ : ∃ i0, i = 0 * (R.nComp + 1) + (1 + i0) := ⟨i - 1, by omega⟩
    have hi0 : i0 < R.nComp := by omega
    have ha : (R.stack 1).a.getD (0 * (R.nComp + 1) + (1 + i0)) [] = (R.a.getD i0 []).map (R.stackCol 1 0) :=
      Lowered.stackRows_getD R R.a (K := 1) (k := 0) (by omega) hi0
    have hb : (R.stack 1).b.getD (0 * (R.nComp + 1) + (1 + i0)) [] = (R.b.getD i0 []).map (R.stackCol 1 0) :=
      Lowered.stackRows_getD R R.b (K := 1) (k := 0) (by omega) hi0
    rw [ha, hb, ← List.map_append, List.mem_map] at hcol
    obtain ⟨x, hx, rfl⟩ := hcol
    by_cases hxn : x < R.nIn
    · rw [Lowered.stackCol_input R (K := 1) (k := 0) hxn] at hlt ⊢
      exact ⟨i0, hi0, by rw [Nat.zero_mul, Nat.zero_add]; exact hx, by omega⟩
    · exfalso
      unfold Rows.stackCol at hlt
      rw [ite_eq_right hxn] at hlt
      split at hlt <;> omega

theorem wireOf_input {c : Flock.Circuit} {n u col g : ℕ} (hw : Lowered.wireOf c n u col = g)
    (hg : g < (keyProg c n).inputs) (h0 : g ≠ 0) (h1 : g ≠ 1) :
    col < Lowered.srcs c ∧ ∃ b, keyIdx c (Lowered.srcs c) col = some b ∧ g = 2 + u * Lowered.srcs c + b := by
  have hI := Lowered.keyProg_inputs c n
  unfold Lowered.wireOf at hw
  split at hw
  · rename_i hc
    refine ⟨hc, ?_⟩
    split at hw
    · rename_i b hb
      refine ⟨b, ?_, hw.symm⟩
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range, ite_eq_left hc] at hb
      simpa [Lowered.srcs] using hb
    · exact absurd hw.symm h1
  · split at hw
    · unfold Lowered.srcs at hI
      omega
    · exact absurd hw.symm h0

/-- **A unit's input is a source gate of a port bit below 16**: it is the source gate of a column its rows read
(`owner`), a column of an input port (`readsWhy_held`), and a key's (`keyIdx`) only below the port's 16th bit
(`src_lt16`). -/
theorem insOf_mem {c : Flock.Circuit} {n u g : ℕ} {P : Flock.Program.Prog} {tmpl : String} {d : Flock.Program.Defn}
    (hF : Check.Facts c P tmpl) (hr : readsWhy c d (unitReads c) [] = none)
    (hg : g ∈ insOf (keyProg c n) (consts c n) u) :
    g ∈ (List.range (Flock.portCols c.unit.inGroups).size).flatMap fun p => (List.range 16).filterMap fun τ =>
      (keyIdx c (Lowered.srcs c) (((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + τ)).map
        (2 + u * Lowered.srcs c + ·) := by
  unfold insOf at hg
  rw [List.mem_filter, List.mem_range] at hg
  obtain ⟨hgs, hgc⟩ := hg
  simp only [Bool.and_eq_true, Option.isNone_iff_eq_none, beq_iff_eq] at hgc
  obtain ⟨⟨hrow, hκ⟩, hown⟩ := hgc
  have hgi : g < (keyProg c n).inputs := by
    by_contra hc
    obtain ⟨ab, hab⟩ := Lowered.row?_some _ (Nat.not_lt.1 hc) hgs
    rw [hab] at hrow
    cases hrow
  have h0 : g ≠ 0 := by rintro rfl; simp [consts] at hκ
  have h1 : g ≠ 1 := by rintro rfl; simp [consts] at hκ
  unfold owner at hown
  have hfind := List.find?_some hown
  simp only [List.any_eq_true, List.mem_range, Bool.and_eq_true, beq_iff_eq, List.contains_iff_mem] at hfind
  obtain ⟨h, -, hhu, hga⟩ := hfind
  obtain ⟨hh1, -, hP, hu⟩ := unitOf_some hhu
  have hun : u < n := (unitGate_of_unitOf hhu).1.1
  have hdm : (h - (keyProg c n).inputs) / (keyProg c n).perUnit * (keyProg c n).perUnit +
      (h - (keyProg c n).inputs) % (keyProg c n).perUnit = h - (keyProg c n).inputs := by
    rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
  have hi := Nat.mod_lt (h - (keyProg c n).inputs) hP
  rw [← hu] at hdm
  have hrow := Lowered.keyProg_row? c hun hi
  rw [show (keyProg c n).inputs + (u * (keyProg c n).perUnit + (h - (keyProg c n).inputs) % (keyProg c n).perUnit) =
    h by omega] at hrow
  simp only [args, hrow] at hga
  rw [← List.map_append, List.mem_map] at hga
  obtain ⟨col, hcol, hw⟩ := hga
  obtain ⟨hcolB, b, hb, rfl⟩ := wireOf_input hw hgi h0 h1
  have hiR : (h - (keyProg c n).inputs) % (keyProg c n).perUnit < ((Rows.ofNet c.unit).stack c.upv).nComp := by
    rw [← Lowered.keyProg_perUnit c n]; exact hi
  have hcolR : col < ((Rows.ofNet c.unit).stack c.upv).nIn := hcolB
  rw [hF.upv] at hiR hcol hcolR
  obtain ⟨i0, hi0, hcol0, hcol0n⟩ := stack_one hiR hcol hcolR
  obtain ⟨p, hp, hp1, hp2⟩ := readsWhy_held hr hcol0n (unitReads_hit hi0 hcol0 hcol0n)
  have hW : col < c.unit.inWords * Flock.WORD := by rw [← hF.inWords]; exact hcol0n
  obtain ⟨τ, rfl⟩ : ∃ τ, col = ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1 + τ :=
    ⟨col - ((Flock.portCols c.unit.inGroups).getD p (0, 0)).1, by omega⟩
  have hτ : τ < ((Flock.portCols c.unit.inGroups).getD p (0, 0)).2 := by omega
  have hsk := Reads.srcKey_port (ix := Index.of c) hF.ordered hp hτ hW
  have hz : srcAt c (Index.of c) 0 p τ ≠ .zero := by
    rw [← hsk]
    intro hz0
    unfold keyIdx at hb
    dsimp only at hb
    rw [hz0] at hb
    simp at hb
  have h16 := src_lt16 hF hp hτ hz
  rw [List.mem_flatMap]
  exact ⟨p, List.mem_range.2 hp, List.mem_filterMap.2 ⟨τ, List.mem_range.2 h16, by rw [hb, Option.map_some]⟩⟩

/-- **A unit owns at most 16 inputs a parameter word** (`insOf_mem`): at most 16 a port, and the ports are the
parameters' words (`Check.Facts.ports`). -/
theorem insOf_length {c : Flock.Circuit} {n u : ℕ} {P : Flock.Program.Prog} {tmpl : String}
    {d : Flock.Program.Defn} (hF : Check.Facts c P tmpl) (hd : P.defs.find? (·.key == tmpl) = some d)
    (hr : readsWhy c d (unitReads c) [] = none) :
    (insOf (keyProg c n) (consts c n) u).length ≤ 16 * paramWords d.params := by
  have hnd : (insOf (keyProg c n) (consts c n) u).Nodup := by
    unfold insOf
    exact List.nodup_range.filter _
  refine (length_le_of_nodup hnd fun g hg => insOf_mem hF hr hg).trans ?_
  refine (length_flatMap_range_le _ (k := 16) (fun p => (List.length_filterMap_le _ _).trans (by simp)) _).trans ?_
  rw [(hF.ports d hd).1, ← paramWords_eq]
  omega

/-- **Lemma A, the file alone**: where member `m`'s circuit file passes Bits' check as the template `tmpl`, whose
Definition is `d`, and reads no input padding, its unit `u`'s proof block (`blockOf`, at the member's constants and
first position) commits at most `16 · (paramWords d.params + retWords d.ret)` bits of the members' joint program. -/
theorem committed_blockOf {P : Flock.Program.Prog} {tmpl : String} {d : Flock.Program.Defn} {ms : Members} {m u : ℕ}
    (hm : m < ms.length) (hc : check (ms.getD m default).1 P tmpl = none) (hd : P.defs.find? (·.key == tmpl) = some d)
    (hr : readsWhy (ms.getD m default).1 d (unitReads (ms.getD m default).1) [] = none) :
    Claims.Sampling.committed (jointOf ms)
        ((blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ≤
      16 * (paramWords d.params + retWords d.ret) := by
  have hF := Check.facts hc
  unfold Claims.Sampling.committed blockOf
  rw [List.map_append, List.filter_append, List.length_append, Nat.mul_add,
    Nat.add_comm (16 * paramWords d.params)]
  refine Nat.add_le_add (units_committed hm hF hd fun q hq => List.mem_append_left _ (List.mem_map_of_mem hq)) ?_
  refine (List.length_filter_le _ _).trans ?_
  rw [List.length_map, List.length_map]
  exact insOf_length hF hd hr

open Properties.NetworkScan.Fits (callee egressNode bind_ok need_ok needSome_ok ingressTemplates_ok split_two
  split_three)
open NciCheck.Properties.SecurityProofs (BodiesOk.ty_beq)

/-- **An accepted scan's terms**: `egress` is the egress template's `vuBits`, and `ingress` bucket 0's template's, at
the larger parameter words of it and the later ingress template, which returns what it returns. -/
theorem scanProg_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    (h : scanProg P program s = .ok sc) :
    sc.egress = vuBits 1 (paramWords (callee P (egressNode P)).params + retWords (callee P (egressNode P)).ret) ∧
    ((∃ a e, templateNodes P = [a, e] ∧
        sc.ingress = vuBits 1 (max (paramWords (callee P a).params) 0 + retWords (callee P a).ret)) ∨
      ∃ a b e, templateNodes P = [a, b, e] ∧ (callee P b).ret = (callee P a).ret ∧
        sc.ingress = vuBits 1 (max (paramWords (callee P a).params) (paramWords (callee P b).params) +
          retWords (callee P a).ret)) := by
  unfold scanProg at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨root, hroot, h⟩ := bind_ok h
  replace hroot := needSome_ok hroot
  obtain rfl : root = P.defs[P.root]! := by
    obtain ⟨hlt, e⟩ := Array.getElem?_eq_some_iff.1 hroot
    rw [getElem!_pos P.defs _ hlt, e]
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨⟨xe, de⟩, hback, h⟩ := bind_ok h
  replace hback := needSome_ok hback
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨⟨⟨xf, df⟩, later⟩, hm, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, h6, h⟩ := bind_ok h
  replace h6 := need_ok h6
  iterate 12 obtain ⟨_, -, h⟩ := bind_ok h
  have hsc := (Except.ok.inj h).symm
  subst hsc
  have hxe : xe = Fits.node P (egressNode P) ∧ de = callee P (egressNode P) := by
    rw [Array.back?_map, List.back?_toArray, Option.map_eq_some_iff] at hback
    obtain ⟨e, he, hge⟩ := hback
    have hen : egressNode P = e := by simp [egressNode, he]
    rw [hen]
    simp only [Prod.mk.injEq] at hge
    exact ⟨hge.1.symm, hge.2.symm⟩
  obtain ⟨rfl, rfl⟩ := hxe
  refine ⟨rfl, ?_⟩
  rcases ingressTemplates_ok hm with ⟨f, hf, hx⟩ | ⟨f, b, hf, hx⟩
  · simp only [Array.toList_pop, Array.toList_map] at hf
    obtain ⟨a, e, hts, hga⟩ := split_two hf
    simp only [Prod.mk.injEq] at hx
    obtain ⟨hx1, rfl⟩ := hx
    have hfa : (xf, df) = (Fits.node P a, callee P a) := by rw [hx1, ← hga]
    simp only [Prod.mk.injEq] at hfa
    obtain ⟨rfl, rfl⟩ := hfa
    exact .inl ⟨a, e, hts, rfl⟩
  · simp only [Array.toList_pop, Array.toList_map] at hf
    obtain ⟨a, a', e, hts, hga, hgb⟩ := split_three hf
    simp only [Prod.mk.injEq] at hx
    obtain ⟨hx1, rfl⟩ := hx
    have hfa : (xf, df) = (Fits.node P a, callee P a) := by rw [hx1, ← hga]
    have hba : b = (Fits.node P a', callee P a') := by rw [← hgb]
    simp only [Prod.mk.injEq] at hfa
    obtain ⟨rfl, rfl⟩ := hfa
    subst hba
    simp only [Option.all_some, Bool.and_eq_true] at h6
    exact .inr ⟨a, a', e, hts, BodiesOk.ty_beq h6.2.2, rfl⟩

/-- **Lemma A, the scan's terms**: at an accepted scan, each template's `16 · (paramWords + retWords)` is at most
`max sc.ingress sc.egress`. -/
theorem words_le_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    (h : scanProg P program s = .ok sc) {i : ℕ} (hi : i < (templateNodes P).length) :
    16 * (paramWords (templateDef P ((templateNodes P).getD i 0)).params +
      retWords (templateDef P ((templateNodes P).getD i 0)).ret) ≤ max sc.ingress sc.egress := by
  obtain ⟨heg, hin⟩ := scanProg_terms h
  have hing : ∀ {k}, 16 * (paramWords (callee P k).params + retWords (callee P k).ret) ≤ sc.ingress →
      16 * (paramWords (callee P k).params + retWords (callee P k).ret) ≤ max sc.ingress sc.egress :=
    fun h => h.trans (le_max_left _ _)
  have hegr : 16 * (paramWords (callee P (egressNode P)).params + retWords (callee P (egressNode P)).ret) ≤
      max sc.ingress sc.egress := by
    refine le_trans ?_ (le_max_right _ _)
    rw [heg, vuBits]; omega
  rcases hin with ⟨a, e, hts, hin⟩ | ⟨a, b, e, hts, hba, hin⟩
  · have hen : egressNode P = e := by simp [egressNode, hts]
    rw [hen] at hegr
    rw [hts] at hi ⊢
    have hi2 : i = 0 ∨ i = 1 := by simp only [List.length_cons, List.length_nil] at hi; omega
    rcases hi2 with rfl | rfl
    · show 16 * (paramWords (callee P a).params + retWords (callee P a).ret) ≤ _
      refine hing ?_
      rw [hin, vuBits]
      have := Nat.le_max_left (paramWords (callee P a).params) 0
      omega
    · exact hegr
  · have hen : egressNode P = e := by simp [egressNode, hts]
    rw [hen] at hegr
    rw [hts] at hi ⊢
    have hi3 : i = 0 ∨ i = 1 ∨ i = 2 := by simp only [List.length_cons, List.length_nil] at hi; omega
    rcases hi3 with rfl | rfl | rfl
    · show 16 * (paramWords (callee P a).params + retWords (callee P a).ret) ≤ _
      refine hing ?_
      rw [hin, vuBits]
      have := Nat.le_max_left (paramWords (callee P a).params) (paramWords (callee P b).params)
      omega
    · show 16 * (paramWords (callee P b).params + retWords (callee P b).ret) ≤ _
      refine hing ?_
      rw [hin, vuBits, hba]
      have := Nat.le_max_right (paramWords (callee P a).params) (paramWords (callee P b).params)
      omega
    · exact hegr

/-- **Lemma A**: at an accepted scan whose templates' files `fs` Bits lays out (`FilesOf`) reading no input padding
(`PaddingUnread`), a unit's proof block of a member whose file is the one `fs` lists at a template index commits at
most `max sc.ingress sc.egress` bits of the members' joint program, whatever the other members are. -/
theorem committed_le_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    {fs : List Flock.Circuit} (hscan : scanProg P program s = .ok sc) (hfs : FilesOf P fs) (hpad : PaddingUnread P fs)
    {ms : Members} {m u : ℕ} (hm : m < ms.length) {i : ℕ} (hi : i < (templateNodes P).length)
    (hfile : (ms.getD m default).1 = fs.getD i default) :
    Claims.Sampling.committed (jointOf ms)
        ((blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ≤
      max sc.ingress sc.egress := by
  have hi' : i < fs.length := hfs.1 ▸ hi
  have hc := hfs.2 i hi'
  have hr := hpad i hi'
  rw [← hfile] at hc hr
  have hk : (templateNodes P).getD i 0 ∈ templateNodes P := by
    rw [Query.list_getD_of_lt _ hi]; exact List.getElem_mem hi
  have hd := find?_key (Properties.NetworkPeriod.keysUnique_nodup (Fits.scanProg_accepted hscan).keys)
    (templateDef_mem hk)
  exact (committed_blockOf hm hc hd hr).trans (words_le_terms hscan hi)

/-- **Lemma B**: a set `F` of the partition's blocks, at most `b` of them, each committing at most `β` bits, commits at
most `b · β` (`faultyBits`). -/
theorem faultyBits_le {C : Claims.Sampling.Prog} {Q : Claims.Sampling.Partition} {F : Finset ℕ} {β b : ℕ}
    (hF : F ⊆ Finset.range Q.blocks.length)
    (hβ : ∀ v < Q.blocks.length, Claims.Sampling.committed C (Q.blocks.getD v []) ≤ β) (hb : F.card ≤ b) :
    NetTiming.faultyBits C Q F ≤ b * β := by
  unfold NetTiming.faultyBits
  calc ∑ v ∈ F, Claims.Sampling.committed C (Q.blocks.getD v []) ≤ ∑ _v ∈ F, β :=
        Finset.sum_le_sum fun v hv => hβ v (Finset.mem_range.1 (hF hv))
    _ = F.card * β := by rw [Finset.sum_const, smul_eq_mul]
    _ ≤ b * β := Nat.mul_le_mul_right β hb

end Properties.NetworkPeriod.Placed
