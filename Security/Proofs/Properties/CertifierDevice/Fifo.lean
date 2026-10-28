import Proofs.Properties.CertifierDevice.Trace

/-!
# The certifier device, proved: complete records are the Program's

The egress certifier is a FIFO queue of the frames the shaper handed. A shaper tick hands the frames at the next
canonical positions (`handList`), enqueues them, and releases `min(queue, r)` from the queue's front (`fifo_tick`). Over a
trace (`fifo_inv`), the shaper has handed `handed`'s frames, its `queued` count is the Program's queue (`NetworkCertifier.queue`)
of the hand counts, and an unfaulted certifier's queue is the last `queued` frames handed. A complete record's window
never faults (by `FailClosed`), so each of its rows is the FIFO's release at the buckets that handed its frames
(`Complete`), which is `NetworkCertifier.expectedRow` when each frame is handed at the bucket
`NetworkCertifier.handingBuckets` gives it (`completeRows_honest`).
-/

namespace Properties.CertifierDevice.Proofs

open _root_.CertifierDevice
open NetworkCertifier (Reason Status)

/-! ## Enqueues and faults -/

theorem enqueue_none {c : Certifier} {f : ByteArray} (h : (c.enqueue f).fault = none) :
    c.fault = none ∧ c.enqueue f = { c with queue := c.queue.push f } := by
  unfold Certifier.enqueue at h ⊢
  by_cases h1 : c.fault.isSome
  · simp only [h1, ↓reduceIte] at h
    simp [h] at h1
  · simp only [h1, Bool.false_eq_true, ↓reduceIte] at h ⊢
    by_cases h2 : f.size ≠ c.params.frameBytes
    · simp only [h2, ne_eq, not_false_eq_true, ↓reduceIte] at h
      have := doFault_fault_isSome c .malformed
      rw [h] at this
      cases this
    · simp only [h2, ↓reduceIte] at h ⊢
      by_cases h3 : c.queue.size ≥ c.params.queueCapacity
      · simp only [h3, ↓reduceIte] at h
        have := doFault_fault_isSome c .overflow
        rw [h] at this
        cases this
      · simp only [h3, ↓reduceIte]
        exact ⟨by simpa using h1, by simp⟩

theorem CertRel.of_none {c c' : Certifier} (h : CertRel c c') (hf : c'.fault = none) : c' = c := by
  rcases h with rfl | ⟨r, rfl⟩
  · rfl
  · have := doFault_fault_isSome c r
    rw [hf] at this
    cases this

/-! ## The handing loop -/

/-- The frames at canonical positions `s.next … m − 1` that `s` holds, in order. -/
def handList (s : Shaper) (m : Nat) : List Frame :=
  (List.range' s.next (m - s.next)).filterMap fun k => (s.pending.get? k).map (·.1)

/-- What the handing loop has done since `s1`: handed the positions `s1.next … s.next − 1`, each held, and, unless it
left the certifier faulted, enqueued their frames. -/
def HandInv (s1 s : Shaper) : Prop :=
  s1.next ≤ s.next ∧ s.queued = s1.queued + (s.next - s1.next) ∧ s.t = s1.t ∧
    (∀ k, s.pending.get? k = if s1.next ≤ k ∧ k < s.next then none else s1.pending.get? k) ∧
    (handList s1 s.next).length = s.next - s1.next ∧
    (s.certifier.fault = none → s.certifier = ((handList s1 s.next).map Frame.bytes).foldl Certifier.enqueue s1.certifier)

theorem handList_succ (s1 : Shaper) {m : Nat} (hm : s1.next ≤ m) {frame : Frame} {ready : Int}
    (h : s1.pending.get? m = some (frame, ready)) : handList s1 (m + 1) = handList s1 m ++ [frame] := by
  unfold handList
  rw [show m + 1 - s1.next = (m - s1.next) + 1 by omega, List.range'_concat, List.filterMap_append]
  have h' : s1.pending[m]? = some (frame, ready) := h
  simp [show s1.next + (m - s1.next) = m by omega, h']

theorem HandInv.step {T : Nat} {s1 s s' : Shaper} (hi : HandInv s1 s) (h : HandStep T s s') : HandInv s1 s' := by
  obtain ⟨i1, i2, i3, i4, i5, i6⟩ := hi
  obtain ⟨frame, ready, sync, hh, hand, hget, -, -, -, -, rfl⟩ := h
  have hget1 : s1.pending.get? s.next = some (frame, ready) := by
    rw [i4 s.next] at hget
    simpa using hget
  have hl := handList_succ s1 i1 hget1
  unfold HandInv
  dsimp only
  refine ⟨by omega, by omega, i3, fun k => ?_, ?_, ?_⟩
  · rw [Std.HashMap.get?_erase, i4 k]
    by_cases hk : s.next = k
    · subst hk
      simp only [beq_self_eq_true, ↓reduceIte]
      rw [ite_eq_left_of_eq_true _ _ (eq_true ⟨i1, by omega⟩)]
    · have : (s.next == k) = false := by simpa using hk
      simp only [this, Bool.false_eq_true, ↓reduceIte]
      congr 1
      apply propext
      omega
  · rw [hl, List.length_append, i5]
    simp only [List.length_singleton]
    omega
  · intro hf
    obtain ⟨h1, h2⟩ := enqueue_none hf
    have e : lateCert s hh = s.certifier := (lateCert_rel s hh).of_none h1
    rw [e] at h1 h2 ⊢
    rw [hl, List.map_append, List.foldl_append, ← i6 h1]
    rfl

theorem handStage_inv (T : Nat) (s1 : Shaper) : HandInv s1 (handStage T s1) := by
  have := handStage_reach T s1
  generalize handStage T s1 = s2 at this ⊢
  induction this with
  | refl =>
    refine ⟨le_rfl, by omega, rfl, fun k => ?_, by simp [handList], fun _ => by simp [handList]⟩
    rw [ite_eq_right_iff.2]
    intro h
    omega
  | tail _ hst ih => exact ih.step hst

/-! ## The certifier's queue -/

/-- Enqueues that leave the certifier unfaulted pushed every frame. -/
theorem enqueue_fold (fs : List ByteArray) (c : Certifier) (h : (fs.foldl Certifier.enqueue c).fault = none) :
    c.fault = none ∧ (fs.foldl Certifier.enqueue c).queue = c.queue ++ fs.toArray := by
  induction fs generalizing c with
  | nil => exact ⟨h, by simp⟩
  | cons f fs ih =>
    obtain ⟨h1, h2⟩ := ih (c.enqueue f) h
    obtain ⟨h3, h4⟩ := enqueue_none h1
    rw [List.foldl_cons, h2, h4]
    exact ⟨h3, by simp⟩

theorem certTick_none {c : Certifier} (h : c.fault = none) :
    let n := min c.queue.size c.params.allocation
    c.tick.2 = (c.queue.extract 0 n).map some ++ Array.replicate (c.params.allocation - n) none ∧
      c.tick.1.queue = c.queue.extract n c.queue.size := by
  unfold Certifier.tick
  simp [h]

theorem certTick_queue_of_fault {c : Certifier} (h : c.fault.isSome) : c.tick.1.queue = c.queue := by
  unfold Certifier.tick
  simp [h]

theorem fault_checkAdvice_isSome (s : Shaper) (r : Reason) : (s.fault r).checkAdvice.certifier.fault.isSome := by
  rcases (checkAdvice_rel (s.fault r)).1 with h | ⟨r', h⟩ <;> rw [h] <;> exact doFault_fault_isSome _ _

/-- The row and the close, for the queue: the hand count and the queued count kept and released, and the certifier's
queue the tick's unless the window closes, where an unfaulted next window keeps it, and has nothing queued unless the
closing window was unfaulted. -/
theorem finishStage_fifo (T : Nat) (s : Shaper) :
    let x := finishStage T s
    x.1.next = s.next ∧ x.1.queued = s.queued - min s.queued s.certifier.params.allocation ∧
      ((s.t + 1) % T ≠ 0 → x.1.certifier = s.certifier.tick.1) ∧
      ((s.t + 1) % T = 0 → x.1.certifier.fault = none →
        x.1.certifier.queue = s.certifier.tick.1.queue ∧ (x.1.queued = 0 ∨ s.certifier.fault = none)) := by
  obtain ⟨-, t2, -, -, t5⟩ := certTick_keeps s.certifier
  unfold finishStage
  dsimp only
  rw [t2]
  by_cases hc : (s.t + 1) % T = 0
  · simp only [hc, ne_eq, not_true_eq_false, ↓reduceIte, false_implies, true_implies, true_and]
    have hcl : ∀ (s' : Shaper), CertRel s.certifier.tick.1.closeSamplingBlock.1 s'.certifier →
        s'.certifier.fault = none → s'.certifier.queue = s.certifier.tick.1.queue := by
      intro s' hr hf
      rw [hr.of_none hf]
      rfl
    split
    · rename_i hq
      split
      · rename_i r hr
        refine ⟨by rw [(checkAdvice_rel _).2]; rfl, by rw [(checkAdvice_rel _).2]; rfl, fun hf => ?_⟩
        exfalso
        exact Option.isSome_iff_ne_none.1 (fault_checkAdvice_isSome _ r) hf
      · rename_i hr
        refine ⟨by rw [(checkAdvice_rel _).2], by rw [(checkAdvice_rel _).2], fun hf =>
          ⟨hcl _ (checkAdvice_rel _).1 hf, Or.inr ?_⟩⟩
        simp only [Certifier.closeSamplingBlock, t5] at hr
        cases hfs : s.certifier.fault with
        | none => rfl
        | some r => rw [hfs] at hr; exact absurd rfl (hr r)
    · rename_i hq
      refine ⟨by rw [(checkAdvice_rel _).2], by rw [(checkAdvice_rel _).2], fun hf =>
        ⟨hcl _ (checkAdvice_rel _).1 hf, Or.inl ?_⟩⟩
      rw [(checkAdvice_rel _).2]
      dsimp only
      omega
  · simp only [hc, ne_eq, not_false_eq_true, ↓reduceIte, false_implies, and_true, true_implies]

/-- **One tick, for the queue.** It hands `L`, the frames at the next canonical positions, counts them queued, and
releases `min(queued, r)`; if it leaves the certifier clean (unfaulted, or at a window's close a complete record), the
certifier was clean before it and the row is the front of its queue with `L` enqueued, which keeps the rest; a
certifier left unfaulted otherwise has nothing queued. -/
theorem fifo_tick {link : String} {p : Params} (s : Shaper) (hs : ShGood link p s) :
    let x := s.tick
    let L := handList s x.1.next
    let Q := s.certifier.queue ++ (L.map Frame.bytes).toArray
    let m := min Q.size p.allocation
    x.1.next = s.next + L.length ∧
      x.1.queued = s.queued + L.length - min (s.queued + L.length) p.allocation ∧
      ((if (s.t + 1) % p.buckets = 0 then x.2.2.map Record.status = some .complete else x.1.certifier.fault = none) →
        s.certifier.fault = none ∧ x.2.1 = (Q.extract 0 m).map some ++ Array.replicate (p.allocation - m) none) ∧
      (x.1.certifier.fault = none → (s.certifier.fault = none ∧ x.1.certifier.queue = Q.extract m Q.size) ∨
        (x.1.certifier.queue = #[] ∧ x.1.queued = 0)) := by
  obtain ⟨c1, prover, syncs, hc1, -, e1⟩ := declareStage_eq s
  have hT : s.certifier.params.buckets = p.buckets := by rw [hs.shape.params]
  have hs1 : Shape c1 link p := hc1.shape hs.shape
  have hi := handStage_inv p.buckets { s with certifier := c1, prover, syncs }
  obtain ⟨k1, -, k3, k4, hs2, -⟩ := handStage_keeps p.buckets { s with certifier := c1, prover, syncs } hs1
  dsimp only
  rw [tick_eq, e1, hT]
  have hL : ∀ m, handList s m = handList { s with certifier := c1, prover, syncs } m := fun _ => rfl
  generalize handStage p.buckets { s with certifier := c1, prover, syncs } = s2 at hi k1 k3 k4 hs2 ⊢
  obtain ⟨i1, i2, -, -, i5, i6⟩ := hi
  rw [← hL] at i5 i6
  dsimp only at i1 i2 i5 i6 k1 k3 k4
  have hg2 : ShGood link p s2 := ⟨hs2, by rw [k4, hc1.rows, k1, hs.rows], by rw [k3, hc1.window, k1, hs.window],
    by rw [k4, hc1.rows]; exact hs.packed⟩
  obtain ⟨f1, f2, f3, f4⟩ := finishStage_fifo p.buckets s2
  obtain ⟨-, -, g3, g4⟩ := finishStage_spec s2 hg2
  have hp2 : s2.certifier.params = p := hs2.params
  rw [hp2] at f2
  rw [k1] at f3 f4 g4
  generalize finishStage p.buckets s2 = x at f1 f2 f3 f4 g3 g4 ⊢
  rw [f1]
  generalize handList s s2.next = L at i5 i6 ⊢
  have key : s2.certifier.fault = none →
      s.certifier.fault = none ∧ s2.certifier.queue = s.certifier.queue ++ (L.map Frame.bytes).toArray := by
    intro hf
    have e6 := i6 hf
    rw [e6] at hf ⊢
    obtain ⟨h1, h2⟩ := enqueue_fold _ _ hf
    have e := hc1.of_none h1
    subst e
    exact ⟨h1, h2⟩
  have hrel : s2.certifier.fault = none →
      let Q := s.certifier.queue ++ (L.map Frame.bytes).toArray
      let m := min Q.size p.allocation
      s.certifier.fault = none ∧ x.2.1 = (Q.extract 0 m).map some ++ Array.replicate (p.allocation - m) none ∧
        s2.certifier.tick.1.queue = Q.extract m Q.size := by
    intro hf
    obtain ⟨h1, h2⟩ := key hf
    obtain ⟨t1, t2⟩ := certTick_none hf
    refine ⟨h1, ?_, ?_⟩
    · rw [g3, t1, h2, hp2]
    · rw [t2, h2, hp2]
  refine ⟨by omega, by rw [f2, i2]; omega, fun hc => ?_, fun hf => ?_⟩
  · have hf : s2.certifier.fault = none := by
      by_cases hcl : (s.t + 1) % p.buckets = 0
      · simp only [hcl, ↓reduceIte] at hc g4
        rw [g4] at hc
        cases hfs : s2.certifier.fault with
        | none => rfl
        | some r => rw [hfs] at hc; cases hc
      · simp only [hcl, ↓reduceIte] at hc
        rw [f3 hcl, (certTick_keeps _).2.2.2.2] at hc
        exact hc
    exact ⟨(hrel hf).1, (hrel hf).2.1⟩
  · by_cases hcl : (s.t + 1) % p.buckets = 0
    · obtain ⟨q1, q2⟩ := f4 hcl hf
      cases hfs : s2.certifier.fault with
      | none => exact Or.inl ⟨(hrel hfs).1, by rw [q1, (hrel hfs).2.2]⟩
      | some r =>
        have hsome : s2.certifier.fault.isSome := by rw [hfs]; rfl
        refine Or.inr ⟨by rw [q1, certTick_queue_of_fault hsome, hs2.dropped hsome], ?_⟩
        rcases q2 with q2 | q2
        · exact q2
        · rw [hfs] at q2; cases q2
    · rw [f3 hcl] at hf ⊢
      rw [(certTick_keeps _).2.2.2.2] at hf
      exact Or.inl ⟨(hrel hf).1, (hrel hf).2.2⟩

/-! ## An egress tick's shaper -/

theorem egressPre_pending (st : State) (now backlog : Nat) :
    (egressPre st now backlog).shaper.pending = st.shaper.pending := by
  unfold egressPre
  by_cases h1 : st.isLate st.config.egress.bucketMs st.shaper.t now = true
  all_goals by_cases h2 : backlog > st.config.egress.allocation * st.config.egress.frameBytes
  all_goals simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]
  all_goals rfl

theorem checkForwarding_pending (st : State) (now : Nat) :
    (st.checkForwarding now).shaper.pending = st.shaper.pending := by
  unfold State.checkForwarding
  dsimp only
  refine Id.bind_inv (fun (r : State × Array Nat) => r.1.shaper.pending = st.shaper.pending)
    (fun (r : State) => r.shaper.pending = st.shaper.pending) _ _ ?_ ?_
  · refine forIn_inv_arr (fun (r : State × Array Nat) => r.1.shaper.pending = st.shaper.pending) rfl ?_
    intro s _ b hb
    split
    · exact hb
    · split
      · split
        · exact hb
        · exact hb
      · exact hb
  · intro r hr
    split
    · exact hr
    · exact hr

theorem egressMid_rel (st : State) (now backlog : Nat) : ShRel st.shaper (egressMid st now backlog) :=
  ((egressPre_quiet st now backlog).trans (checkForwarding_quiet _ now)).2.1

theorem egressMid_pending (st : State) (now backlog : Nat) :
    (egressMid st now backlog).pending = st.shaper.pending := by
  unfold egressMid
  rw [checkForwarding_pending, egressPre_pending]

theorem egressTail_shaper (p : Params) (now backlog t : Nat) (st : State) :
    ∃ l, (egressTail p now backlog t st).1.shaper = { (st.checkForwarding now).shaper.tick.1 with prover := l } := by
  rw [egressTail_eq]
  obtain ⟨l, hl⟩ := startAt_shaper p t (egressShape p now backlog st).1
  refine ⟨l, hl.trans ?_⟩
  unfold egressShape
  dsimp only
  generalize st.checkForwarding now = st1
  generalize st1.shaper.tick = x
  rcases x with ⟨sh, row, record⟩
  dsimp only
  split <;> split <;> rfl

/-- A live egress tick leaves the shaper `egressMid`'s tick leaves, but for the online prover's choice. -/
theorem step_egress_shaper {c : Config} {st : State} (hg : Good c st) {now backlog : Nat}
    (ht : st.shaper.t < horizon c) :
    ∃ l, (step st (.tick .egress now backlog)).1.shaper = { (egressMid st now backlog).tick.1 with prover := l } := by
  rw [step_egress_fst hg ht, egressTick_eq]
  exact egressTail_shaper _ _ _ _ _

/-- The frames a live egress tick hands are `egressMid`'s `handList`. -/
theorem handedAt_egress {c : Config} {st : State} (hg : Good c st) {now backlog : Nat}
    (ht : st.shaper.t < horizon c) :
    handedAt st (.tick .egress now backlog) =
      handList (egressMid st now backlog) (egressMid st now backlog).tick.1.next := by
  obtain ⟨l, hl⟩ := step_egress_shaper hg ht (now := now) (backlog := backlog)
  unfold handedAt handList
  rw [hl]
  dsimp only
  rw [(egressMid_rel st now backlog).2.2.1, egressMid_pending]

theorem handedAt_quiet {st : State} {ev : Event} (h : (step st ev).1.shaper.next = st.shaper.next) :
    handedAt st ev = [] := by
  unfold handedAt
  rw [h, Nat.sub_self]
  rfl

/-! ## Sums -/

/-- `f 0 + … + f (t − 1)`. -/
def rsum (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | t + 1 => rsum f t + f t

theorem rsum_congr {f g : Nat → Nat} {t : Nat} (h : ∀ u < t, f u = g u) : rsum f t = rsum g t := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [rsum]; rw [ih (fun u hu => h u (by omega)), h t (by omega)]

theorem rsum_add (f g : Nat → Nat) (t : Nat) : rsum (fun u => f u + g u) t = rsum f t + rsum g t := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [rsum, ih]; omega

theorem rsum_ind (a t : Nat) : rsum (fun u => if a = u then 1 else 0) t = if a < t then 1 else 0 := by
  induction t with
  | zero => simp [rsum]
  | succ t ih => simp only [rsum, ih]; split_ifs <;> omega

/-- The hand counts of frames all handed before bucket `t` sum to their number. -/
theorem rsum_count (l : List Nat) (t : Nat) (h : ∀ x ∈ l, x < t) : rsum (fun u => l.count u) t = l.length := by
  induction l with
  | nil =>
    have : ∀ t, rsum (fun _ => 0) t = 0 := by
      intro t; induction t with
      | zero => rfl
      | succ t ih => simp [rsum, ih]
    simpa using this t
  | cons a l ih =>
    have e : rsum (fun u => (a :: l).count u) t =
        rsum (fun u => l.count u + if a = u then 1 else 0) t := rsum_congr fun u _ => by simp [List.count_cons]
    rw [e, rsum_add, ih (fun x hx => h x (List.mem_cons_of_mem _ hx)), rsum_ind]
    simp [h a (List.mem_cons_self ..)]

theorem queue_congr {S : Nat} {f g : Nat → Nat} {t : Nat} (h : ∀ u < t, f u = g u) :
    NetworkCertifier.queue S f t = NetworkCertifier.queue S g t := by
  induction t with
  | zero => rfl
  | succ t ih => simp only [NetworkCertifier.queue]; rw [ih (fun u hu => h u (by omega)), h t (by omega)]

/-- The FIFO's conservation: what it released before bucket `t` and what it holds are what was enqueued before `t`. -/
theorem released_conserve (S : Nat) (f : Nat → Nat) (t : Nat) :
    rsum (fun u => (NetworkCertifier.released S f u : Nat)) t + NetworkCertifier.queue S f t = rsum f t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [rsum, NetworkCertifier.queue, NetworkCertifier.released] at ih ⊢
    omega

theorem sum_finRange (g : Nat → Nat) (t : Nat) : ((List.finRange t).map fun u : Fin t => g u).sum = rsum g t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [List.finRange_succ_last, List.map_append, List.map_map, List.sum_append]
    simp only [Function.comp_def, Fin.val_castSucc, ih, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      Fin.val_last, rsum]
    omega

theorem offset_last (S : Nat) (f : Nat → Nat) (t : Nat) :
    NetworkCertifier.offset (fun u : Fin (t + 1) => NetworkCertifier.released S f u) (Fin.last t) =
      rsum (fun u => (NetworkCertifier.released S f u : Nat)) t := by
  unfold NetworkCertifier.offset
  rw [List.finRange_succ_last, List.filter_append, List.filter_map]
  have e1 : List.filter ((fun u => decide (u < Fin.last t)) ∘ Fin.castSucc) (List.finRange t) = List.finRange t := by
    rw [List.filter_eq_self]
    intro u _
    simp [Fin.castSucc_lt_last]
  rw [e1]
  simp only [lt_self_iff_false, decide_false, Bool.false_eq_true, not_false_eq_true, List.filter_cons_of_neg,
    List.filter_nil, List.append_nil, List.map_map]
  exact sum_finRange (fun u => (NetworkCertifier.released S f u : Nat)) t

/-! ## Over a trace -/

/-- What the shaper's FIFO is after handing `hp`: it handed them all before its bucket, its `queued` count is the
Program's queue of their hand counts, and an unfaulted certifier holds the last `queued` frames handed. -/
structure FifoInv (S : Nat) (hp : List (Frame × Nat)) (s : Shaper) : Prop where
  next : s.next = hp.length
  past : ∀ x ∈ hp, x.2 < s.t
  queued : s.queued = NetworkCertifier.queue S (fun u => (hp.map Prod.snd).count u) s.t
  le : s.queued ≤ hp.length
  queue : s.certifier.fault = none → s.certifier.queue.toList = (hp.map (·.1.bytes)).drop (hp.length - s.queued)

theorem FifoInv.rel {S : Nat} {hp : List (Frame × Nat)} {s s' : Shaper} (h : FifoInv S hp s) (hr : ShRel s s') :
    FifoInv S hp s' := by
  obtain ⟨r1, r2, r3, r4, -⟩ := hr
  refine ⟨r3.trans h.next, fun x hx => r2 ▸ h.past x hx, by rw [r4, r2]; exact h.queued, r4 ▸ h.le, fun hf => ?_⟩
  rw [r1.of_none hf, r4]
  exact h.queue (by rw [← r1.of_none hf]; exact hf)

theorem handed_append (st : State) (a b : List Event) : handed st (a ++ b) = handed st a ++ handed (after st a) b := by
  induction a generalizing st with
  | nil => rfl
  | cons ev a ih =>
    simp only [List.cons_append, handed, ih, List.append_assoc]
    rfl

theorem handed_take_succ (c : Config) (tr : List Event) (n : Nat) (h : n < tr.length) :
    handed (init c) (tr.take (n + 1)) =
      handed (init c) (tr.take n) ++ (handedAt (pre c tr n) tr[n]).map (·, (pre c tr n).shaper.t) := by
  rw [List.take_succ_eq_append_getElem h, handed_append]
  simp [handed]

/-- Every frame handed from `st` on is handed at a bucket at least `st`'s. -/
theorem handed_ge {c : Config} {st : State} (hg : Good c st) (tr : List Event) :
    ∀ x ∈ handed st tr, st.shaper.t ≤ x.2 := by
  induction tr generalizing st with
  | nil => simp [handed]
  | cons ev tr ih =>
    obtain ⟨g1, t1⟩ := step_good hg ev
    have hle : st.shaper.t ≤ (step st ev).1.shaper.t := by rw [t1]; unfold run; split <;> omega
    intro x hx
    rcases List.mem_append.1 hx with hx | hx
    · obtain ⟨f, -, rfl⟩ := List.mem_map.1 hx
      exact le_rfl
    · exact hle.trans (ih g1 x hx)

theorem init_fifo (c : Config) : FifoInv c.egress.allocation [] (init c).shaper := by
  unfold init Shaper.make
  dsimp only
  obtain ⟨hr, e⟩ := checkAdvice_rel
    { certifier := { link := c.egressLink, params := c.egress }, syncs := c.syncs, syncSet := c.syncSet,
      prover := c.prover.map fun p =>
        let l : LaggedChoices :=
          { prover := Prover.make c.syncSet c.egress.buckets p.rule p.envelope p.prior p.lag }
        if c.windows > 0 then l.start else l,
      windows := c.windows }
  refine ⟨by rw [e]; rfl, by simp, by rw [e]; rfl, by rw [e]; exact le_rfl, fun hf => ?_⟩
  rw [hr.of_none hf]
  simp

/-- **One step, for the FIFO.** -/
theorem step_fifo {c : Config} {st : State} {hp : List (Frame × Nat)} (hg : Good c st)
    (hi : FifoInv c.egress.allocation hp st.shaper) (ev : Event) :
    FifoInv c.egress.allocation (hp ++ (handedAt st ev).map (·, st.shaper.t)) (step st ev).1.shaper := by
  by_cases hl : ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c
  · obtain ⟨now, backlog, rfl, ht⟩ := hl
    have hr := egressMid_rel st now backlog
    have hgs : ShGood c.egressLink c.egress (egressMid st now backlog) := hg.shaper.rel hr
    have hi' := hi.rel hr
    obtain ⟨l, hl⟩ := step_egress_shaper hg ht (now := now) (backlog := backlog)
    rw [handedAt_egress hg ht, hl, ← hr.2.1]
    refine FifoInv.rel (s := (egressMid st now backlog).tick.1) ?_ ⟨CertRel.refl _, rfl, rfl, rfl, rfl, rfl⟩
    have tk := (tick_spec _ hgs).2.1
    obtain ⟨f1, f2, -, f4⟩ := fifo_tick _ hgs
    generalize egressMid st now backlog = s at hr hgs hi' tk f1 f2 f4 ⊢
    generalize s.tick = x at tk f1 f2 f4 ⊢
    generalize handList s x.1.next = L at f1 f2 f4 ⊢
    obtain ⟨i1, i2, i3, i4, i5⟩ := hi'
    have hcnt : ∀ u, ((hp ++ L.map (·, s.t)).map Prod.snd).count u =
        (hp.map Prod.snd).count u + if s.t = u then L.length else 0 := by
      intro u
      simp [List.map_append, List.count_append, Function.comp_def, List.count_replicate]
    have hzero : (hp.map Prod.snd).count s.t = 0 :=
      List.count_eq_zero_of_not_mem fun hm => by
        obtain ⟨y, hy, e⟩ := List.mem_map.1 hm
        have := i2 y hy
        omega
    refine ⟨by rw [f1, i1]; simp, fun y hy => ?_, ?_, ?_, fun hf => ?_⟩
    · rw [tk]
      rcases List.mem_append.1 hy with hy | hy
      · have := i2 y hy; omega
      · obtain ⟨f, -, rfl⟩ := List.mem_map.1 hy; simp
    · rw [tk, NetworkCertifier.queue, hcnt, hzero, f2]
      rw [queue_congr (g := fun u => (hp.map Prod.snd).count u) (fun u hu => by
        rw [hcnt]; simp [Nat.ne_of_gt hu])] 
      rw [← i3]
      simp
    · rw [f2]; simp; omega
    · have hlen : ((hp.map (·.1.bytes)).drop (hp.length - s.queued)).length = s.queued := by simp; omega
      rcases f4 hf with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · rw [h2, Array.toList_extract, List.extract_eq_take_drop, List.take_of_length_le (by simp),
          Array.toList_append, i5 h1, Array.size_append, ← Array.length_toList, i5 h1, hlen, f2]
        simp only [List.map_append, List.map_map, Function.comp_def, List.length_append, List.length_map,
          List.size_toArray]
        rw [← List.drop_append_of_le_length (by simp), List.drop_drop]
        congr 1
        omega
      · rw [h1, h2]
        simp
  · obtain ⟨q, -⟩ := step_quiet hg ev hl
    rw [handedAt_quiet q.2.1.2.2.1, List.map_nil, List.append_nil]
    exact hi.rel q.2.1

theorem fifo_inv (c : Config) (tr : List Event) (n : Nat) (hn : n ≤ tr.length) :
    FifoInv c.egress.allocation (handed (init c) (tr.take n)) (pre c tr n).shaper := by
  induction n with
  | zero =>
    rw [List.take_zero]
    exact init_fifo c
  | succ n ih =>
    have hn' : n < tr.length := by omega
    rw [handed_take_succ c tr n hn', pre_succ c tr n hn']
    exact step_fifo (prefix_good c tr n).1 (ih (by omega)) _

theorem getElem!_eq {α : Type _} [Inhabited α] {a : Array α} {j : Nat} {v : α} (h : a[j]? = some v) : a[j]! = v := by
  rw [getElem!_def, h]

/-- **The row a clean tick releases**: slot `i` holds the frame `i` places after the oldest one queued, among those
handed through this tick, if fewer than `min(queued, r)` precede it in the row, and is padding otherwise. -/
theorem release {c : Config} {st : State} {hp : List (Frame × Nat)} (hg : Good c st)
    (hi : FifoInv c.egress.allocation hp st.shaper) {now backlog : Nat} (ht : st.shaper.t < horizon c)
    (hc : if (st.shaper.t + 1) % c.egress.buckets = 0 then
        (egressMid st now backlog).tick.2.2.map Record.status = some .complete
      else (egressMid st now backlog).tick.1.certifier.fault = none) :
    ∀ i < c.egress.allocation, (egressMid st now backlog).tick.2.1[i]? =
      some (if i < min (st.shaper.queued + (handedAt st (.tick .egress now backlog)).length) c.egress.allocation then
        ((hp ++ (handedAt st (.tick .egress now backlog)).map (·, st.shaper.t)).map (·.1.bytes))[hp.length -
          st.shaper.queued + i]?
      else none) := by
  intro i hia
  have hr := egressMid_rel st now backlog
  have hgs : ShGood c.egressLink c.egress (egressMid st now backlog) := hg.shaper.rel hr
  have hi' := hi.rel hr
  rw [handedAt_egress hg ht]
  rw [← hr.2.1] at hc ⊢
  rw [← hr.2.2.2.1]
  obtain ⟨-, -, f3, -⟩ := fifo_tick _ hgs
  obtain ⟨h1, h2⟩ := f3 hc
  generalize egressMid st now backlog = s at hr hgs hi' h1 h2 ⊢
  generalize s.tick = x at h2 ⊢
  generalize handList s x.1.next = L at h2 ⊢
  obtain ⟨-, -, -, i4, i5⟩ := hi'
  have hq := i5 h1
  have hsize : s.certifier.queue.size = s.queued := by
    rw [← Array.length_toList, hq]; simp; omega
  rw [h2, ← Array.getElem?_toList]
  simp only [Array.toList_append, Array.toList_map, Array.toList_extract, List.extract_eq_take_drop,
    Array.toList_replicate, hq, Array.size_append, hsize, List.size_toArray, List.length_map, Nat.sub_zero,
    List.drop_zero]
  by_cases hm : i < min (s.queued + L.length) c.egress.allocation
  · rw [List.getElem?_append_left (by simp; omega)]
    simp only [hm, ↓reduceIte, List.getElem?_map, List.getElem?_take, List.map_append, List.map_map,
      Function.comp_def]
    rw [← List.drop_append_of_le_length (by simp), List.getElem?_drop, List.getElem?_eq_getElem (by simp; omega)]
    rfl
  · rw [List.getElem?_append_right (by simp; omega)]
    simp only [hm, ↓reduceIte, List.getElem?_replicate, List.length_map, List.length_take, List.length_append,
      List.length_drop, List.length_map]
    split_ifs <;> first | rfl | (exfalso; omega)

/-- **The discrete intermediate value theorem**: every egress bucket up to one that runs runs. -/
theorem egressAt_exists {c : Config} {tr : List Event} {n k b : Nat} (hE : EgressAt c tr n k b) {t : Nat}
    (ht : t ≤ k) : ∃ m b', m ≤ n ∧ EgressAt c tr m t b' := by
  have hn := egressAt_lt hE
  have hk1 : (pre c tr (n + 1)).shaper.t = k + 1 := by
    rw [t_succ c tr n hn]
    obtain ⟨now, he⟩ := egressAt_get hE
    rw [he, egressAt_t hE]
    simp [run, tickView, hE.2.2]
  have hex : ∃ i, t < (pre c tr i).shaper.t := ⟨n + 1, by omega⟩
  have hi : t < (pre c tr (Nat.find hex)).shaper.t := Nat.find_spec hex
  have hle : Nat.find hex ≤ n + 1 := Nat.find_min' hex (by omega)
  have h0 : Nat.find hex ≠ 0 := by
    intro h
    rw [h] at hi
    simp [pre, after, init_t] at hi
  obtain ⟨m, hm⟩ : ∃ m, Nat.find hex = m + 1 := ⟨Nat.find hex - 1, by omega⟩
  have hnot : ¬ t < (pre c tr m).shaper.t := Nat.find_min hex (by omega)
  rw [hm] at hi hle
  have hm' : m < tr.length := by omega
  have hs := t_succ c tr m hm'
  unfold run at hs
  split at hs
  · rename_i hv
    obtain ⟨hv1, hv2⟩ := hv
    have htm : (pre c tr m).shaper.t = t := by omega
    rw [← htm]
    cases he : tr[m] with
    | tick g now b' =>
      cases g
      · exact ⟨m, b', by omega, egressAt_of hm' he hv2⟩
      · rw [he] at hv1; simp [tickView] at hv1
    | _ => rw [he] at hv1; simp [tickView] at hv1
  · omega

/-! ## Complete records -/

/-- Complete records over the whole trace, given that each frame is handed at the bucket
`NetworkCertifier.handingBuckets` gives it under the syncs the trace declared. -/
def CompleteRowsHonest : Prop :=
  ∀ (c : Config) (tr : List Event),
    let hs := handed (init c) tr
    let syncs := (after (init c) tr).shaper.syncs
    let frames : List NetworkCertifier.LogicalFrame := hs.map fun x => ⟨bitsOf x.1.bytes, x.1.anchor, x.1.index⟩
    let T := c.egress.buckets
    hs.map Prod.snd = NetworkCertifier.handingBuckets T syncs.size (fun w => syncs[w]) frames →
    ∀ n k backlog acts rec p, EgressAt c tr n k backlog → (outputs (init c) tr)[n]? = some acts →
      .publish rec p ∈ acts → rec.status = .complete → rec.window < syncs.size →
      ∀ j < T, ∀ (s : Nat) (hs : s < c.egress.allocation),
        NetworkCertifier.expectedRow c.egress.allocation T syncs.size (fun w => syncs[w]) frames (rec.window * T + j)
          ⟨s, hs⟩ = (rec.grid[j]!)[s]!.map bitsOf

theorem completeRows_honest : CompleteRowsHonest := by
  intro c tr
  dsimp only
  intro H n k backlog acts rec p hE hacts hmem hst hw j hj i hia
  rw [egress_acts hE] at hacts
  obtain rfl := Option.some.inj hacts
  unfold egressActions at hmem
  obtain ⟨hc, rfl⟩ : (k + 1) % c.egress.buckets = 0 ∧
      rec = ⟨c.egressLink, k / c.egress.buckets,
        Array.ofFn fun j : Fin c.egress.buckets => rowOf c tr (k / c.egress.buckets * c.egress.buckets + j),
        statusOf c tr (k / c.egress.buckets)⟩ := by
    rcases List.mem_append.1 hmem with h | h
    · split at h <;> simp at h
    · split at h
      · simp only [List.mem_singleton, Action.publish.injEq] at h
        exact ⟨‹_›, h.1⟩
      · simp at h
  dsimp only at hst hw ⊢
  obtain ⟨hT0, hkm, hlast⟩ := mod_last hc
  generalize hT : c.egress.buckets = T at *
  generalize hwd : k / T = w at *
  have hgrid : (Array.ofFn fun j : Fin T => rowOf c tr (w * T + j))[j]! = rowOf c tr (w * T + j) :=
    getElem!_eq (by simp [hj])
  rw [hgrid]
  have htk : w * T + j ≤ k := by omega
  generalize htd : w * T + j = t at htk ⊢
  obtain ⟨m, b', hmn, hEm⟩ := egressAt_exists hE htk
  obtain ⟨now', he'⟩ := egressAt_get hEm
  have hm := egressAt_lt hEm
  have hgm := (prefix_good c tr m).1
  have htm := egressAt_t hEm
  have hth : (pre c tr m).shaper.t < horizon c := htm ▸ hEm.2.2
  obtain ⟨l, hl⟩ := step_egress_shaper hgm hth (now := now') (backlog := b')
  have hpre : (pre c tr (m + 1)).shaper = { (egressMid (pre c tr m) now' b').tick.1 with prover := l } := by
    rw [pre_succ c tr m hm, he', hl]
  have ht1 : (pre c tr (m + 1)).shaper.t = t + 1 := by
    rw [hpre, (tick_spec _ (hgm.shaper.rel (egressMid_rel _ now' b'))).2.1, (egressMid_rel _ now' b').2.1, htm]
  rw [rowOf_eq hEm, he']
  simp only [evRow]
  have hclean : if ((pre c tr m).shaper.t + 1) % c.egress.buckets = 0 then
      (egressMid (pre c tr m) now' b').tick.2.2.map Record.status = some .complete
    else (egressMid (pre c tr m) now' b').tick.1.certifier.fault = none := by
    rw [htm, hT]
    by_cases hj1 : j = T - 1
    · have htk' : t = k := by omega
      subst htk'
      obtain rfl : m = n := egressAt_unique hEm hE
      have hcl : (t + 1) % T = 0 := hc
      simp only [hcl, ↓reduceIte]
      obtain ⟨-, -, row, status, -, -, hs, -⟩ := egress_spec hgm now' b'
      have hs' := hs (by rw [htm, hT]; exact hcl)
      have hso := statusOf_eq hE w (by rw [hT]; omega)
      rw [he', hst] at hso
      simp only [evStatus, hs', Option.getD_some] at hso
      rw [hs', ← hso]
    · have hmod : (t + 1) % T = j + 1 := by
        rw [show t + 1 = (j + 1) + w * T by omega, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
      simp only [hmod, Nat.add_one_ne_zero, ↓reduceIte]
      cases hf : (egressMid (pre c tr m) now' b').tick.1.certifier.fault with
      | none => rfl
      | some r =>
        exfalso
        have hf' : (after (init c) (tr.take (m + 1))).shaper.certifier.fault = some r := by
          change (pre c tr (m + 1)).shaper.certifier.fault = _
          rw [hpre, hf]
        have hne : m ≠ n := by
          rintro rfl
          have := egressAt_t hE
          omega
        have hwin : k / c.egress.buckets = (after (init c) (tr.take (m + 1))).shaper.t / c.egress.buckets := by
          change _ = (pre c tr (m + 1)).shaper.t / _
          rw [ht1, hT, hwd, show t + 1 = (j + 1) + w * T by omega, Nat.add_mul_div_right _ _ hT0,
            Nat.div_eq_of_lt (by omega)]
          omega
        obtain ⟨grid, -, hout⟩ := failClosed_proof c tr (m + 1) r hf' n k backlog (by omega) hE hwin
        rw [egress_acts hE] at hout
        have := Option.some.inj hout
        unfold egressActions at this
        rw [hT] at this
        simp only [hc, ↓reduceIte] at this
        split at this
        · simp only [List.cons_append, List.nil_append, List.cons.injEq, Action.publish.injEq,
            Record.mk.injEq] at this
          simp [hwd, hst] at this
        · simp only [List.nil_append, List.cons.injEq, Action.publish.injEq, Record.mk.injEq] at this
          simp [hwd, hst] at this
  have hfi := fifo_inv c tr m hm.le
  have hrel := release hgm hfi hth hclean i hia
  rw [getElem!_eq hrel]
  have hfull : handed (init c) tr = handed (init c) (tr.take m) ++
      (handedAt (pre c tr m) (.tick .egress now' b')).map (·, t) ++ handed (pre c tr (m + 1)) (tr.drop (m + 1)) := by
    conv_lhs => rw [← List.take_append_drop (m + 1) tr]
    rw [handed_append, handed_take_succ c tr m hm, he', htm]
  have hrest : ∀ x ∈ handed (pre c tr (m + 1)) (tr.drop (m + 1)), t + 1 ≤ x.2 :=
    fun x hx => ht1 ▸ handed_ge (prefix_good c tr (m + 1)).1 _ x hx
  have hpast : ∀ x ∈ (handed (init c) (tr.take m)).map Prod.snd, x < t := by
    intro x hx
    obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
    exact htm ▸ hfi.past y hy
  have hle := hfi.le
  have hqq : (pre c tr m).shaper.queued = NetworkCertifier.queue c.egress.allocation
      (fun u => ((handed (init c) (tr.take m)).map Prod.snd).count u) t := htm ▸ hfi.queued
  rw [htm]
  generalize handed (init c) (tr.take m) = hp at hfull hpast hle hqq ⊢
  generalize handedAt (pre c tr m) (.tick .egress now' b') = L at hfull ⊢
  generalize (pre c tr m).shaper.queued = q at hle hqq ⊢
  have hWT : t < (after (init c) tr).shaper.syncs.size * T := by
    have := Nat.mul_le_mul_right T (show w + 1 ≤ _ from hw)
    rw [Nat.succ_mul] at this
    omega
  have hcount : ∀ u ≤ t, NetworkCertifier.enqueued T (after (init c) tr).shaper.syncs.size
      (fun w => (after (init c) tr).shaper.syncs[w])
      ((handed (init c) tr).map fun x => ⟨bitsOf x.1.bytes, x.1.anchor, x.1.index⟩) u =
      (hp.map Prod.snd).count u + if t = u then L.length else 0 := by
    intro u hu
    unfold NetworkCertifier.enqueued
    simp only [show u < (after (init c) tr).shaper.syncs.size * T by omega, ↓reduceIte]
    rw [← H, hfull]
    have hz : ((handed (pre c tr (m + 1)) (tr.drop (m + 1))).map Prod.snd).count u = 0 :=
      List.count_eq_zero_of_not_mem fun hm => by
        obtain ⟨y, hy, e⟩ := List.mem_map.1 hm
        have := hrest y hy
        omega
    simp only [List.map_append, List.count_append, hz, List.map_map, Function.comp_def, List.map_const',
      List.count_replicate, beq_iff_eq, Nat.add_zero]
  unfold NetworkCertifier.expectedRow NetworkCertifier.fifoGrid NetworkCertifier.fill
  simp only [Fin.val_last]
  rw [offset_last]
  generalize NetworkCertifier.enqueued T (after (init c) tr).shaper.syncs.size
    (fun w => (after (init c) tr).shaper.syncs[w])
    ((handed (init c) tr).map fun x => ⟨bitsOf x.1.bytes, x.1.anchor, x.1.index⟩) = E at hcount ⊢
  have hzero : (hp.map Prod.snd).count t = 0 := List.count_eq_zero_of_not_mem fun h => by
    have := hpast t h
    omega
  have hlt : ∀ u < t, E u = (hp.map Prod.snd).count u := fun u hu => by
    rw [hcount u hu.le]
    simp [Nat.ne_of_gt hu]
  have hEt : E t = L.length := by rw [hcount t le_rfl, hzero]; simp
  have hqE : NetworkCertifier.queue c.egress.allocation E t = q := by rw [queue_congr hlt, hqq]
  have hrelt : (NetworkCertifier.released c.egress.allocation E t : Nat) = min (q + L.length) c.egress.allocation := by
    simp only [NetworkCertifier.released]
    rw [hqE, hEt]
  have hoff : rsum (fun u => (NetworkCertifier.released c.egress.allocation E u : Nat)) t = hp.length - q := by
    have h1 := released_conserve c.egress.allocation E t
    rw [rsum_congr hlt, rsum_count _ t hpast, hqE, List.length_map] at h1
    omega
  rw [hrelt, hoff]
  split_ifs with h1
  · have hk : hp.length - q + i < (hp ++ L.map (·, t)).length := by simp; omega
    obtain ⟨v, hv⟩ : ∃ v, (hp ++ L.map (·, t))[hp.length - q + i]? = some v := ⟨_, List.getElem?_eq_getElem hk⟩
    rw [List.getElem?_map, hv]
    simp only [Option.map_some]
    unfold NetworkCertifier.payloads
    rw [List.getElem?_map, hfull, List.getElem?_append_left hk, hv]
    rfl
  · rfl

end Properties.CertifierDevice.Proofs
