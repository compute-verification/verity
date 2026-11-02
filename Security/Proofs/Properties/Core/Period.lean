import Properties.Core.Period

/-!
# Proofs that the code's periods are the spec's (`Properties.Core.Period`)

A code cut is its boundaries, an array; a spec cut is a monotone function from `Fin (n + 1)`. An array of boundaries
that is `isCut` of a span is `Array.ofFn` of exactly one cut of it (`isCut_iff`): its first and last boundaries are the
span's start and stop, and `ordered` is monotone (`ordered_ofFn`). The parts of `Array.ofFn f` are the spans between
consecutive values of `f` (`parts_ofFn`), which is `Cut.part`. A monitoring period then reads cut by cut.
-/

namespace Properties.Period

open _root_.Core (TickSpan MonitoringPeriod parts)

theorem head?_ofFn {n : ℕ} (f : Fin (n + 1) → ℕ) : (List.ofFn f).head? = some (f 0) := by
  rw [List.ofFn_succ]; rfl

theorem getLast?_ofFn {n : ℕ} (f : Fin (n + 1) → ℕ) : (List.ofFn f).getLast? = some (f (Fin.last n)) := by
  rw [List.getLast?_eq_getElem?, List.getElem?_ofFn]
  simp [Fin.last]

theorem ordered_ofFn {n : ℕ} (f : Fin (n + 1) → ℕ) :
    TickSpan.ordered (List.ofFn f) = true ↔ Monotone f := by
  rw [Fin.monotone_iff_le_succ]
  induction n with
  | zero => simp [List.ofFn_succ, TickSpan.ordered]
  | succ n ih =>
    have step : TickSpan.ordered (List.ofFn f) =
        (decide (f 0 ≤ f 1) && TickSpan.ordered (List.ofFn fun i : Fin (n + 1) => f i.succ)) := by
      rw [List.ofFn_succ (f := f), List.ofFn_succ (f := fun i : Fin (n + 1) => f i.succ)]
      rfl
    rw [step, Bool.and_eq_true, decide_eq_true_eq, ih (fun i => f i.succ), Fin.forall_fin_succ]
    simp only [Fin.castSucc_zero, Fin.succ_zero_eq_one]
    exact Iff.rfl

/-- `isCut` of `Array.ofFn f`: `f` starts at the span's start, ends at its stop, and is monotone. -/
theorem isCut_ofFn {n : ℕ} (p : TickSpan) (f : Fin (n + 1) → ℕ) :
    p.isCut (Array.ofFn f) = true ↔ f 0 = p.start ∧ f (Fin.last n) = p.stop ∧ Monotone f := by
  simp only [TickSpan.isCut, Array.toList_ofFn, head?_ofFn, getLast?_ofFn, Bool.and_eq_true, beq_iff_eq,
    Option.some.injEq, ordered_ofFn, and_assoc]

theorem eq_ofFn (bs : Array ℕ) : bs = #[] ∨ ∃ n, ∃ f : Fin (n + 1) → ℕ, bs = Array.ofFn f := by
  rcases h : bs.size with _ | n
  · exact .inl (Array.size_eq_zero_iff.mp h)
  · refine .inr ⟨n, fun i => bs[i.1], ?_⟩
    apply Array.ext <;> simp [h]

/-- **A code cut is a spec cut's boundaries.** `bs` is a cut of the code span `⟨q.start, q.len⟩` exactly when it is
`boundaries` of a cut of `q`. -/
theorem isCut_iff (q : Verity.TickSpan) (bs : Array ℕ) :
    TickSpan.isCut ⟨q.start, q.len⟩ bs = true ↔ ∃ n, ∃ c : q.Cut n, boundaries c = bs := by
  constructor
  · intro h
    rcases eq_ofFn bs with rfl | ⟨n, f, rfl⟩
    · simp [TickSpan.isCut] at h
    · obtain ⟨first, last, mono⟩ := (isCut_ofFn _ f).mp h
      exact ⟨n, ⟨f, mono, first, last⟩, rfl⟩
  · rintro ⟨n, c, rfl⟩
    exact (isCut_ofFn _ c.boundary).mpr ⟨c.first, c.last, c.mono⟩

theorem zipWith_ofFn {α β γ : Type} {n : ℕ} (g : α → β → γ) (a : Fin n → α) (b : Fin n → β) :
    List.zipWith g (List.ofFn a) (List.ofFn b) = List.ofFn fun j => g (a j) (b j) := by
  apply List.ext_getElem <;> simp

/-- The parts of `Array.ofFn f`: the spans from each value of `f` to the next. -/
theorem parts_ofFn {n : ℕ} (f : Fin (n + 1) → ℕ) :
    parts (Array.ofFn f) = List.ofFn fun j : Fin n => (⟨f j.castSucc, f j.succ - f j.castSucc⟩ : TickSpan) := by
  apply List.ext_getElem
  · simp [parts]
  · intro i h₁ h₂
    simp only [parts, Array.toList_ofFn, List.getElem_zipWith, List.getElem_tail, List.getElem_ofFn]
    rfl

theorem parts_boundaries {q : Verity.TickSpan} {n : ℕ} (c : q.Cut n) :
    parts (boundaries c) = List.ofFn fun j => (⟨(c.part j).start, (c.part j).len⟩ : TickSpan) :=
  parts_ofFn c.boundary

theorem spec_mk (q : Verity.TickSpan) : spec ⟨q.start, q.len⟩ = q := rfl

theorem tickSpanIsSpec : TickSpanIsSpec := by
  refine ⟨⟨fun p p' h => ?_, fun q => ⟨⟨q.start, q.len⟩, rfl⟩⟩, fun _ => rfl, fun p => ?_, fun p q => ?_⟩
  · cases p; cases p'; cases h; rfl
  · ext t
    simp [TickSpan.ticks, Verity.TickSpan.ticks, spec, Verity.TickSpan.stop, List.mem_range'_1]
  · simp only [TickSpan.within, Bool.and_eq_true, decide_eq_true_eq]
    exact Iff.rfl

/-- The spans of `code M`'s cuts and the sampling cuts it holds, read through `isCut`. -/
theorem valid_code (M : Verity.MonitoringPeriod) : (code M).valid = true := by
  simp only [MonitoringPeriod.valid, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, id_eq]
  refine ⟨⟨(isCut_iff M.span _).mpr ⟨_, M.executionCut, rfl⟩, by simp [code, boundaries]⟩, ?_⟩
  intro b hb
  simp only [MonitoringPeriod.executionSpans, code, parts_boundaries, Array.toList_ofFn, zipWith_ofFn,
    List.mem_ofFn] at hb
  obtain ⟨j, rfl⟩ := hb
  exact (isCut_iff _ _).mpr ⟨_, M.samplingCut j, rfl⟩

theorem valid_iff (m : MonitoringPeriod) : m.valid = true ↔ ∃ M, code M = m := by
  refine ⟨fun h => ?_, fun ⟨M, hM⟩ => hM ▸ valid_code M⟩
  obtain ⟨s, ex, sa⟩ := m
  simp only [MonitoringPeriod.valid, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, id_eq] at h
  obtain ⟨⟨hex, hsize⟩, hsa⟩ := h
  obtain ⟨n, c, rfl⟩ := (isCut_iff (spec s) ex).mp hex
  have hn : sa.size = n := by simpa [boundaries] using hsize
  subst hn
  have hsa' : sa.toList = List.ofFn fun j : Fin sa.size => sa[j.1] := by
    rw [← Array.toList_ofFn, Array.ofFn_getElem]
  simp only [MonitoringPeriod.executionSpans, parts_boundaries, hsa', zipWith_ofFn, List.mem_ofFn] at hsa
  have hcut : ∀ j : Fin sa.size, ∃ k, ∃ d : (c.part j).Cut k, boundaries d = sa[j.1] :=
    fun j => (isCut_iff _ _).mp (hsa _ ⟨j, rfl⟩)
  choose k d hd using hcut
  refine ⟨⟨spec s, sa.size, c, k, d⟩, ?_⟩
  simp only [code, MonitoringPeriod.mk.injEq, true_and]
  refine ⟨rfl, ?_⟩
  apply Array.ext
  · simp
  · intro i h₁ h₂
    simp only [Array.getElem_ofFn]
    exact hd ⟨i, h₂⟩

theorem cut_ext {q : Verity.TickSpan} {n : ℕ} {c c' : q.Cut n} (h : c.boundary = c'.boundary) : c = c' := by
  cases c; cases c'; cases h; rfl

theorem boundaries_inj {q : Verity.TickSpan} {n n' : ℕ} {c : q.Cut n} {c' : q.Cut n'}
    (h : boundaries c = boundaries c') : n = n' ∧ ∀ e : n = n', e ▸ c = c' := by
  have hn : n = n' := by simpa [boundaries] using congrArg Array.size h
  refine ⟨hn, fun e => ?_⟩
  subst e
  apply cut_ext
  funext i
  have := congrArg (fun a : Array ℕ => a[i.1]?) h
  simpa [boundaries] using this

theorem code_injective : Function.Injective code := by
  rintro ⟨s, n, c, k, d⟩ ⟨s', n', c', k', d'⟩ h
  simp only [code, MonitoringPeriod.mk.injEq, TickSpan.mk.injEq] at h
  obtain ⟨⟨hs, hl⟩, hc, hsa⟩ := h
  have hs : s = s' := by cases s; cases s'; simp_all
  subst hs
  obtain ⟨rfl, hc⟩ := boundaries_inj hc
  obtain rfl := hc rfl
  have hj : ∀ j, boundaries (d j) = boundaries (d' j) := fun j => by
    simpa using congrArg (fun a : Array (Array ℕ) => a[j.1]?) hsa
  obtain rfl : k = k' := funext fun j => (boundaries_inj (hj j)).1
  obtain rfl : d = d' := funext fun j => (boundaries_inj (hj j)).2 rfl
  rfl

theorem monitoringPeriodIsSpec : MonitoringPeriodIsSpec := by
  refine ⟨valid_iff, code_injective, fun M => ?_, fun M => ?_⟩
  · simp only [MonitoringPeriod.executionSpans, code, parts_boundaries, List.map_ofFn]
    rfl
  · simp only [MonitoringPeriod.samplingSpans, code, Array.toList_ofFn, List.map_ofFn]
    congr 1
    funext j
    simp only [Function.comp, parts_boundaries, List.map_ofFn]
    rfl

end Properties.Period
