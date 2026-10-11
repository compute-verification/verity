import Properties.Proofs.CertifierNetwork.Bucketer

/-!
# The records a run publishes

Over a trace, the records `step` publishes on the egress link are of windows `0, 1, …` in order, one per window the
shaper closed, and likewise on the ingress link for the bucketer; each is of its link's grid: `T` packed rows
(`Inv`, kept by every step, `step_inv`). On a closed run both counted all `W` windows (`closed_records`), so each link
has exactly one record of each window `w < W` (`filter_one`).
-/

namespace Properties.CertifierNetwork.Proofs

open _root_.CertifierDevice
open Properties.CertifierDevice (Packed horizon outputs after)
open Properties.CertifierDevice.Proofs

/-- A record of the egress grid on the egress link, or of the ingress grid on the ingress link: `T` packed rows. -/
def OfGrid (c : Config) (r : Record) : Prop :=
  (r.link = c.egressLink ∧ r.grid.size = c.egress.buckets ∧ ∀ row ∈ r.grid, Packed c.egress row) ∨
    (r.link = c.ingressLink ∧ r.grid.size = c.ingress.buckets ∧ ∀ row ∈ r.grid, Packed c.ingress row)

/-- The windows of the records of `P` on link `l`, in order. -/
def windowsOn (l : String) (P : List Record) : List Nat := (P.filter fun r => decide (r.link = l)).map Record.window

theorem windowsOn_append (l : String) (P Q : List Record) :
    windowsOn l (P ++ Q) = windowsOn l P ++ windowsOn l Q := by
  simp [windowsOn, List.filter_append]

theorem windowsOn_eq {l : String} {r : Record} (h : r.link = l) : windowsOn l [r] = [r.window] := by
  simp [windowsOn, h]

theorem windowsOn_ne {l : String} {r : Record} (h : r.link ≠ l) : windowsOn l [r] = [] := by
  simp [windowsOn, h]

/-- What a run's prefix is: the device's state, the bucketer of the ingress grid's shape, both counts within their
horizons, and the records published so far, `P`, one per window each grid closed, in order, each of its grid. -/
structure Inv (c : Config) (st : State) (P : List Record) : Prop where
  good : Good c st
  bu : BuGood c st.bucketer
  sh_le : st.shaper.t ≤ horizon c
  bu_le : st.bucketer.t ≤ c.windows * c.ingress.buckets
  eg : windowsOn c.egressLink P = List.range (st.shaper.t / c.egress.buckets)
  ig : windowsOn c.ingressLink P = List.range (st.bucketer.t / c.ingress.buckets)
  grid : ∀ r ∈ P, OfGrid c r

theorem init_inv (c : Config) : Inv c (init c) [] := by
  have hb : (init c).bucketer.t = 0 := by simp [init]
  refine ⟨init_good c, init_buGood c, by rw [init_t]; exact Nat.zero_le _, by rw [hb]; exact Nat.zero_le _, ?_, ?_,
    fun r hr => nomatch hr⟩
  · rw [init_t]; simp [windowsOn]
  · rw [hb]; simp [windowsOn]

theorem publishes_egress (p q : Prop) [Decidable p] [Decidable q] (w : ByteArray) (r : Record) (g : Params) :
    publishes ((if p then [Action.writeLink w] else []) ++ if q then [Action.publish r g] else []) =
      if q then [r] else [] := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq, publishes]

theorem range_close {t T : Nat} (h : (t + 1) % T = 0) : List.range ((t + 1) / T) = List.range (t / T) ++ [t / T] := by
  rw [succ_div_of_mod h, List.range_succ]

/-- **One step keeps `Inv`**, the records it publishes appended. -/
theorem step_inv {c : Config} (hd : Distinct c) {st : State} {P : List Record} (h : Inv c st P)
    (ev : Event) : Inv c (step st ev).1 (P ++ publishes (step st ev).2) := by
  by_cases hl : ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c
  · obtain ⟨now, backlog, rfl, ht⟩ := hl
    have hb := step_egress_bucketer h.good ht (now := now) (backlog := backlog)
    rw [step_egress_fst h.good ht] at hb ⊢
    rw [step_egress_snd h.good ht]
    obtain ⟨e1, e2, row, status, hrow, -, -, e4, -, -⟩ := egress_spec h.good now backlog
    rw [e4, publishes_egress]
    have hsh := h.good.shaper
    by_cases hm : (st.shaper.t + 1) % c.egress.buckets = 0
    · simp only [hm, ↓reduceIte]
      obtain ⟨hT0, hlast, -⟩ := mod_last hm
      generalize hrec : (⟨c.egressLink, st.shaper.t / c.egress.buckets, st.shaper.certifier.rows.push row, status⟩ :
        Record) = rec
      have hl : rec.link = c.egressLink := by rw [← hrec]
      have hw : rec.window = st.shaper.t / c.egress.buckets := by rw [← hrec]
      refine ⟨e1, h.bu.rel hb, by rw [e2]; exact ht, by rw [hb.2]; exact h.bu_le, ?_, ?_, ?_⟩
      · rw [windowsOn_append, windowsOn_eq hl, h.eg, e2, range_close hm, hw]
      · rw [windowsOn_append, windowsOn_ne (by rw [hl]; exact hd), h.ig, hb.2, List.append_nil]
      · intro r hr
        rcases List.mem_append.1 hr with hr | hr
        · exact h.grid r hr
        · rw [List.mem_singleton] at hr
          subst hr hrec
          refine Or.inl ⟨rfl, ?_, fun row' hr' => ?_⟩
          · simp only [Array.size_push, hsh.rows, hlast]
            omega
          · rcases Array.mem_push.1 hr' with hr' | rfl
            · exact hsh.packed row' hr'
            · exact hrow
    · simp only [hm, ↓reduceIte, List.append_nil]
      refine ⟨e1, h.bu.rel hb, by rw [e2]; exact ht, by rw [hb.2]; exact h.bu_le, ?_, ?_, h.grid⟩
      · rw [h.eg, e2, succ_div_of_not_mod hm]
      · rw [h.ig, hb.2]
  · obtain ⟨q, -⟩ := step_quiet h.good ev hl
    have hgood := h.good.quiet q
    have hst : (step st ev).1.shaper.t = st.shaper.t := q.2.1.2.1
    by_cases hi : ∃ now backlog, ev = .tick .ingress now backlog ∧ st.bucketer.t < c.windows * c.ingress.buckets
    · obtain ⟨now, backlog, rfl, ht⟩ := hi
      obtain ⟨b0, hb0, e1, e2⟩ := step_ingress h.good ht (now := now) (backlog := backlog)
      have hg0 := h.bu.rel hb0
      obtain ⟨k1, k2, k3, k4⟩ := bucketerTick_spec hg0 (Nat.pos_of_ne_zero fun h0 => by simp [h0] at ht)
      rw [e2]
      have hbt : (step st (.tick .ingress now backlog)).1.bucketer.t = st.bucketer.t + 1 := by rw [e1, k2, hb0.2]
      cases hr : b0.tick.2.2 with
      | none =>
        have hm : ¬ (b0.t + 1) % c.ingress.buckets = 0 := fun hm => by simpa [hr] using k3 hm
        rw [hb0.2] at hm
        refine ⟨hgood, e1 ▸ k1, by rw [hst]; exact h.sh_le, by rw [hbt]; exact ht, ?_, ?_, ?_⟩
        · rw [Option.toList_none, List.append_nil, h.eg, hst]
        · rw [Option.toList_none, List.append_nil, h.ig, hbt, succ_div_of_not_mod hm]
        · simpa using h.grid
      | some r =>
        obtain ⟨hm, hlink, hw, hsize, hpk⟩ := k4 r hr
        rw [hb0.2] at hm hw
        rw [Option.toList_some]
        refine ⟨hgood, e1 ▸ k1, by rw [hst]; exact h.sh_le, by rw [hbt]; exact ht, ?_, ?_, ?_⟩
        · rw [windowsOn_append, windowsOn_ne (by rw [hlink]; exact Ne.symm hd), h.eg, hst, List.append_nil]
        · rw [windowsOn_append, windowsOn_eq hlink, h.ig, hbt, range_close hm, hw]
        · intro r' hr'
          rcases List.mem_append.1 hr' with hr' | hr'
          · exact h.grid r' hr'
          · rw [List.mem_singleton] at hr'
            subst hr'
            exact Or.inr ⟨hlink, hsize, hpk⟩
    · obtain ⟨hb, hp⟩ := step_other h.good ev hl hi
      rw [hp, List.append_nil]
      exact ⟨hgood, h.bu.rel hb, by rw [hst]; exact h.sh_le, by rw [hb.2]; exact h.bu_le, by rw [hst]; exact h.eg,
        by rw [hb.2]; exact h.ig, h.grid⟩

theorem inv_after {c : Config} (hd : Distinct c) :
    ∀ (tr : List Event) (st : State) (P : List Record), Inv c st P →
      Inv c (after st tr) (P ++ (outputs st tr).flatMap publishes)
  | [], st, P, h => by simpa [after, outputs] using h
  | ev :: tr, st, P, h => by
    have := inv_after hd tr _ _ (step_inv hd h ev)
    show Inv c (after (step st ev).1 tr) (P ++ ((step st ev).2 :: outputs (step st ev).1 tr).flatMap publishes)
    rwa [List.flatMap_cons, ← List.append_assoc]

/-- **A closed run's records**: on each link, one of each window `w < W`, in order; each of its link's grid. -/
theorem closed_records {c : Config} {tr : List Event} (hT : Ticking c) (hd : Distinct c) (hc : Closed c tr) :
    windowsOn c.egressLink (published c tr) = List.range c.windows ∧
      windowsOn c.ingressLink (published c tr) = List.range c.windows ∧ ∀ r ∈ published c tr, OfGrid c r := by
  have h := inv_after hd tr (init c) [] (init_inv c)
  rw [List.nil_append] at h
  change Inv c (after (init c) tr) (published c tr) at h
  unfold Closed State.done State.ticks at hc
  rw [h.good.config] at hc
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
  have he : (after (init c) tr).shaper.t = c.windows * c.egress.buckets := le_antisymm h.sh_le hc.1
  have hi : (after (init c) tr).bucketer.t = c.windows * c.ingress.buckets := le_antisymm h.bu_le hc.2
  refine ⟨?_, ?_, h.grid⟩
  · rw [h.eg, he, Nat.mul_div_cancel _ hT.1]
  · rw [h.ig, hi, Nat.mul_div_cancel _ hT.2]

theorem range_filter_eq {W w : Nat} (hw : w < W) : (List.range W).filter (fun x => decide (x = w)) = [w] := by
  induction W with
  | zero => omega
  | succ W ih =>
    rw [List.range_succ, List.filter_append]
    by_cases h : w < W
    · rw [ih h]
      simp [Nat.ne_of_gt h]
    · have hw' : w = W := by omega
      subst hw'
      rw [List.filter_eq_nil_iff.2 (fun x hx => by simp at hx; simp; omega)]
      simp

/-- **One record of each window**: if the windows on `l` are `0, …, W − 1`, window `w < W` has exactly one. -/
theorem filter_one {P : List Record} {l : String} {W w : Nat} (h : windowsOn l P = List.range W) (hw : w < W) :
    ∃ r, P.filter (fun r => decide (r.link = l ∧ r.window = w)) = [r] := by
  have e : P.filter (fun r => decide (r.link = l ∧ r.window = w)) =
      (P.filter fun r => decide (r.link = l)).filter fun r => decide (r.window = w) := by
    rw [List.filter_filter]
    congr 1
    funext r
    simp [Bool.and_comm]
  rw [e]
  have hm : ((P.filter fun r => decide (r.link = l)).filter fun r => decide (r.window = w)).map Record.window = [w] := by
    have e2 := List.filter_map (f := Record.window) (p := fun x => decide (x = w))
      (l := P.filter fun r => decide (r.link = l))
    rw [show ((fun x => decide (x = w)) ∘ Record.window) = (fun r : Record => decide (r.window = w)) from rfl] at e2
    unfold windowsOn at h
    rw [← e2, h, range_filter_eq hw]
  obtain ⟨r, hr, -⟩ := List.map_eq_singleton_iff.1 hm
  exact ⟨r, hr⟩

end Properties.CertifierNetwork.Proofs
