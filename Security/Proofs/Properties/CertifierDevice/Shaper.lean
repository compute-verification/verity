import Proofs.Properties.CertifierDevice.Basic

/-!
# The certifier device, proved: one shaper tick

`Shaper.tick` is three stages (`tick_eq`): the prover's declaration (`declareStage`), the handing loop (`handStage`, one
`HandStep` per frame handed) and the certifier's row with the window's close (`finishStage`). `tick_spec` is what a tick
does to a shaper whose certifier has the egress grid's shape: it releases a packed row, padding once faulted, and at a
window's last bucket returns the window's record, its rows the window's and its status the window's first fault.
-/

namespace Properties.CertifierDevice.Proofs

open _root_.CertifierDevice
open NetworkCertifier (Reason Status)

/-! ## The stages -/

/-- `Shaper.tick`'s declaration. -/
def declareStage (s : Shaper) : Shaper :=
  match s.prover with
  | some p =>
    if (s.t % s.certifier.params.buckets = 0 && s.t / s.certifier.params.buckets = s.syncs.size &&
        s.syncs.size < s.windows) = true then
      let s1 := if p.declare.2.2 = true then s.fault .monitorUnavailable else s
      Shaper.checkAdvice { s1 with prover := some p.declare.1, syncs := s1.syncs.push p.declare.2.1 }
    else s
  | none => s

/-- The certifier a frame is enqueued in: faulted `coverage-never-established` if the frame is handed after its `h`. -/
def lateCert (s : Shaper) (h : Nat) : Certifier :=
  if h < s.t then s.certifier.doFault .coverageNeverEstablished else s.certifier

theorem lateCert_rel (s : Shaper) (h : Nat) : CertRel s.certifier (lateCert s h) := by
  unfold lateCert; split
  · exact Or.inr ⟨_, rfl⟩
  · exact CertRel.refl _

/-- `Shaper.tick`'s handing loop body, buckets of `T`. -/
def handBody (T : Nat) (_ : Nat) (s : Shaper) : Id (ForInStep Shaper) :=
  match s.pending.get? s.next with
  | none => pure (.done s)
  | some (frame, ready) =>
    match s.syncs[frame.anchor / T]? with
    | none => pure (.done s)
    | some sync =>
      let h := max (eligibleAt frame.anchor frame.index sync) s.h
      let hand := (max (max (h : Int) ready) s.last).toNat
      if hand > s.t then pure (.done s)
      else pure (.yield { s with
        pending := s.pending.erase s.next, h, last := hand, next := s.next + 1, hands := s.hands.push hand,
        certifier := (lateCert s h).enqueue frame.bytes, queued := s.queued + 1 })

/-- `Shaper.tick`'s handing loop. -/
def handStage (T : Nat) (s : Shaper) : Shaper := (forIn (List.range' 0 s.pending.size) s (handBody T)).run

/-- `Shaper.tick`'s row and the window's close. -/
def finishStage (T : Nat) (s : Shaper) : Shaper × Slots × Option BlockGrid :=
  let s1 : Shaper := { s with
    certifier := s.certifier.tick.1, queued := s.queued - min s.queued s.certifier.tick.1.params.allocation,
    t := s.t + 1 }
  if s1.t % T ≠ 0 then (s1, s.certifier.tick.2, none) else
  let s2 : Shaper := { s1 with certifier := s1.certifier.closeSamplingBlock.1 }
  if s2.queued > 0 then
    match Status.reason? s1.certifier.closeSamplingBlock.2.status with
    | some r => ((s2.fault r).checkAdvice, s.certifier.tick.2, some s1.certifier.closeSamplingBlock.2)
    | _ => (s2.checkAdvice, s.certifier.tick.2, some s1.certifier.closeSamplingBlock.2)
  else (s2.checkAdvice, s.certifier.tick.2, some s1.certifier.closeSamplingBlock.2)

theorem range_size (n : Nat) : [:n].size = n := by simp [Std.Legacy.Range.size]

theorem tick_eq (s : Shaper) :
    s.tick = finishStage s.certifier.params.buckets (handStage s.certifier.params.buckets (declareStage s)) := by
  unfold Shaper.tick
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size]
  rcases hp : s.prover with _ | p
  · simp only [declareStage, hp]
    rfl
  · simp only [declareStage, hp]
    split
    · split <;> rfl
    · rfl

/-! ## Faults and the declaration -/

theorem checkAdvice_eq (s : Shaper) : ∃ c', CertRel s.certifier c' ∧ s.checkAdvice = { s with certifier := c' } := by
  unfold Shaper.checkAdvice
  split
  · split
    · exact ⟨_, CertRel.refl _, rfl⟩
    · exact ⟨_, Or.inr ⟨_, rfl⟩, rfl⟩
  · exact ⟨_, CertRel.refl _, rfl⟩

theorem fault_eq (s : Shaper) (r : Reason) : s.fault r = { s with certifier := s.certifier.doFault r } := rfl

/-- What the declaration changes: the certifier, faulted at most, the prover and the syncs, one pushed at most. -/
theorem declareStage_eq (s : Shaper) : ∃ c' prover syncs, CertRel s.certifier c' ∧
    (syncs = s.syncs ∨ ∃ x, syncs = s.syncs.push x) ∧
    declareStage s = { s with certifier := c', prover, syncs } := by
  unfold declareStage
  split
  · rename_i p _
    split
    · obtain ⟨c1, h1, e1⟩ : ∃ c1, CertRel s.certifier c1 ∧
          (if p.declare.2.2 = true then s.fault .monitorUnavailable else s) = { s with certifier := c1 } := by
        split
        · exact ⟨_, Or.inr ⟨_, rfl⟩, rfl⟩
        · exact ⟨_, CertRel.refl _, rfl⟩
      dsimp only
      rw [e1]
      obtain ⟨c2, h2, e2⟩ := checkAdvice_eq
        { s with certifier := c1, prover := some p.declare.1, syncs := s.syncs.push p.declare.2.1 }
      rw [e2]
      exact ⟨c2, _, _, h1.trans h2, Or.inr ⟨_, rfl⟩, rfl⟩
    · exact ⟨_, _, _, CertRel.refl _, Or.inl rfl, rfl⟩
  · exact ⟨_, _, _, CertRel.refl _, Or.inl rfl, rfl⟩

/-! ## The handing loop -/

/-- One frame handed: the frame at the next canonical position enqueued at hand time `hand`, its handing bucket `h` the
rule's under the sync of its anchor's window, the certifier faulted first if the bucket is after `h`. -/
def HandStep (T : Nat) (s s' : Shaper) : Prop :=
  ∃ frame ready sync h hand, s.pending.get? s.next = some (frame, ready) ∧ s.syncs[frame.anchor / T]? = some sync ∧
    h = max (eligibleAt frame.anchor frame.index sync) s.h ∧ h ≤ s.t ∧ hand ≤ s.t ∧
    s' = { s with pending := s.pending.erase s.next, h, last := hand, next := s.next + 1, hands := s.hands.push hand,
                  certifier := (lateCert s h).enqueue frame.bytes, queued := s.queued + 1 }

theorem handStage_reach (T : Nat) (s : Shaper) : Relation.ReflTransGen (HandStep T) s (handStage T s) := by
  unfold handStage
  refine forIn_inv (fun x => Relation.ReflTransGen (HandStep T) s x) Relation.ReflTransGen.refl ?_
  intro i _ x hx
  unfold handBody
  split
  · exact hx
  · rename_i frame ready hget
    split
    · exact hx
    · rename_i sync hsync
      dsimp only
      split
      · exact hx
      · rename_i hle
        exact hx.tail ⟨frame, ready, sync, _, _, hget, hsync, rfl, by omega, by omega, rfl⟩

/-- What the handing loop keeps. -/
theorem HandStep.keeps {T : Nat} {s s' : Shaper} (h : HandStep T s s') :
    s'.t = s.t ∧ s'.syncs = s.syncs ∧ s'.certifier.link = s.certifier.link ∧
      s'.certifier.params = s.certifier.params ∧ s'.certifier.window = s.certifier.window ∧
      s'.certifier.rows = s.certifier.rows ∧ (∀ r, s.certifier.fault = some r → s'.certifier = s.certifier) := by
  obtain ⟨frame, ready, sync, hh, hand, -, -, -, -, -, rfl⟩ := h
  have hl := lateCert_rel s hh
  refine ⟨rfl, rfl, (enqueue_link _ _).trans hl.link, (enqueue_params _ _).trans hl.params,
    (enqueue_window _ _).trans hl.window, (enqueue_rows _ _).trans hl.rows, fun r hr => ?_⟩
  dsimp only
  rw [hl.of_fault hr]
  exact enqueue_of_fault hr _

theorem HandStep.shape {T : Nat} {s s' : Shaper} {link : String} {p : Params} (h : HandStep T s s')
    (hs : Shape s.certifier link p) : Shape s'.certifier link p := by
  obtain ⟨frame, ready, sync, hh, hand, -, -, -, -, -, rfl⟩ := h
  exact ((lateCert_rel s hh).shape hs).enqueue _

theorem handStage_keeps (T : Nat) (s : Shaper) {link : String} {p : Params} (hs : Shape s.certifier link p) :
    let s' := handStage T s
    s'.t = s.t ∧ s'.syncs = s.syncs ∧ s'.certifier.window = s.certifier.window ∧
      s'.certifier.rows = s.certifier.rows ∧ Shape s'.certifier link p ∧
      (∀ r, s.certifier.fault = some r → s'.certifier = s.certifier) := by
  have := handStage_reach T s
  dsimp only
  generalize handStage T s = s' at this ⊢
  induction this with
  | refl => exact ⟨rfl, rfl, rfl, rfl, hs, fun _ _ => rfl⟩
  | tail _ hst ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih
    obtain ⟨k1, k2, -, -, k5, k6, k7⟩ := hst.keeps
    refine ⟨k1.trans h1, k2.trans h2, k5.trans h3, k6.trans h4, hst.shape h5, fun r hr => ?_⟩
    rw [k7 r (by rw [h6 r hr]; exact hr), h6 r hr]

/-! ## One tick -/

/-- A shaper whose certifier has the egress grid's shape, holds the open window's rows, packed, and is at its window. -/
structure ShGood (link : String) (p : Params) (s : Shaper) : Prop where
  shape : Shape s.certifier link p
  rows : s.certifier.rows.size = s.t % p.buckets
  window : s.certifier.window = s.t / p.buckets
  packed : ∀ row ∈ s.certifier.rows, Packed p row

theorem certTick_row {c : Certifier} {link : String} {p : Params} (h : Shape c link p) : Packed p c.tick.2 := by
  unfold Certifier.tick
  split
  · rw [h.params]; exact packed_replicate p
  · exact packed_tick h

theorem certTick_fault {c : Certifier} {r : Reason} (h : c.fault = some r) :
    c.tick.2 = Array.replicate c.params.allocation none ∧ c.tick.1 = { c with rows := c.rows.push c.tick.2 } := by
  unfold Certifier.tick
  simp [h]

theorem certTick_keeps (c : Certifier) :
    c.tick.1.link = c.link ∧ c.tick.1.params = c.params ∧ c.tick.1.window = c.window ∧
      c.tick.1.rows = c.rows.push c.tick.2 ∧ c.tick.1.fault = c.fault := by
  unfold Certifier.tick
  split <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩

theorem Shape.tick {c : Certifier} {link : String} {p : Params} (h : Shape c link p) : Shape c.tick.1 link p := by
  refine ⟨(certTick_keeps c).1.trans h.link, (certTick_keeps c).2.1.trans h.params, ?_, ?_⟩
  · unfold Certifier.tick
    split
    · exact h.sizes
    · intro f hf
      exact h.sizes f (mem_of_mem_extract hf)
  · intro hs
    unfold Certifier.tick
    have hs' : c.fault.isSome := by
      have := (certTick_keeps c).2.2.2.2
      rwa [this] at hs
    simp [hs', h.dropped hs']

theorem Shape.closeSamplingBlock {c : Certifier} {link : String} {p : Params} (h : Shape c link p) :
    Shape c.closeSamplingBlock.1 link p :=
  ⟨h.link, h.params, h.sizes, by simp [Certifier.closeSamplingBlock]⟩

theorem checkAdvice_rel (s : Shaper) : CertRel s.certifier s.checkAdvice.certifier ∧
    s.checkAdvice = { s with certifier := s.checkAdvice.certifier } := by
  obtain ⟨c', h, e⟩ := checkAdvice_eq s
  rw [e]; exact ⟨h, rfl⟩

theorem checkAdvice_t (s : Shaper) : s.checkAdvice.t = s.t := by rw [(checkAdvice_rel s).2]

theorem succ_div_of_mod {a T : Nat} (h : (a + 1) % T = 0) : (a + 1) / T = a / T + 1 :=
  Nat.succ_div_of_dvd (Nat.dvd_of_mod_eq_zero h)

theorem succ_div_of_not_mod {a T : Nat} (h : ¬ (a + 1) % T = 0) : (a + 1) / T = a / T :=
  Nat.succ_div_of_not_dvd (fun hd => h (Nat.mod_eq_zero_of_dvd hd))

theorem succ_mod_of_not_mod {a T : Nat} (h : ¬ (a + 1) % T = 0) : (a + 1) % T = a % T + 1 := by
  have e := succ_div_of_not_mod h
  have h1 := Nat.div_add_mod (a + 1) T
  have h2 := Nat.div_add_mod a T
  rw [e] at h1
  omega

/-- The row and the window's close, on a shaper whose certifier has the grid's shape. -/
theorem finishStage_spec {link : String} {p : Params} (s : Shaper) (hs : ShGood link p s) :
    let x := finishStage p.buckets s
    ShGood link p x.1 ∧ x.1.t = s.t + 1 ∧ x.2.1 = s.certifier.tick.2 ∧
      if (s.t + 1) % p.buckets = 0 then
        x.2.2 = some ⟨link, s.t / p.buckets, s.certifier.rows.push x.2.1,
          match s.certifier.fault with | none => .complete | some r => .notEstablished r⟩
      else x.2.2 = none ∧ x.1.certifier.rows = s.certifier.rows.push x.2.1 ∧
        x.1.certifier.fault = s.certifier.fault := by
  obtain ⟨t1, t2, t3, t4, t5⟩ := certTick_keeps s.certifier
  have hsh := hs.shape.tick
  have hrow := certTick_row hs.shape
  unfold finishStage
  dsimp only
  by_cases hc : (s.t + 1) % p.buckets = 0
  · simp only [hc, ne_eq, not_true_eq_false, ↓reduceIte]
    have hw := succ_div_of_mod hc
    have key : ∀ s' : Shaper, CertRel s.certifier.tick.1.closeSamplingBlock.1 s'.certifier → s'.t = s.t + 1 →
        ShGood link p s' := by
      intro s' hr ht
      refine ⟨hr.shape hsh.closeSamplingBlock, ?_, ?_, ?_⟩
      · rw [hr.rows, ht, hc]; rfl
      · rw [hr.window, ht, hw]; simp [Certifier.closeSamplingBlock, t3, hs.window]
      · rw [hr.rows]; simp [Certifier.closeSamplingBlock]
    have hrec : s.certifier.tick.1.closeSamplingBlock.2 = ⟨link, s.t / p.buckets,
        s.certifier.rows.push s.certifier.tick.2,
        match s.certifier.fault with | none => .complete | some r => .notEstablished r⟩ := by
      simp only [Certifier.closeSamplingBlock, t1, t3, t4, t5, hs.shape.link, hs.window]
      rfl
    split
    · split
      · refine ⟨key _ ?_ (checkAdvice_t _), checkAdvice_t _, rfl, congrArg some hrec⟩
        exact (CertRel.refl _).doFault _ |>.trans (checkAdvice_rel _).1
      · exact ⟨key _ (checkAdvice_rel _).1 (checkAdvice_t _), checkAdvice_t _, rfl, congrArg some hrec⟩
    · exact ⟨key _ (checkAdvice_rel _).1 (checkAdvice_t _), checkAdvice_t _, rfl, congrArg some hrec⟩
  · simp only [hc, ne_eq, not_false_eq_true, ↓reduceIte]
    refine ⟨⟨hsh, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
    rotate_left 3
    all_goals try first | trivial | exact t4 | exact t5
    · rw [t4, Array.size_push, hs.rows]; exact (succ_mod_of_not_mod hc).symm
    · rw [t3, hs.window]; exact (succ_div_of_not_mod hc).symm
    · intro row hr
      rw [t4] at hr
      rcases Array.mem_push.1 hr with hr | rfl
      · exact hs.packed row hr
      · exact hrow

/-- **One tick.** -/
theorem tick_spec {link : String} {p : Params} (s : Shaper) (hs : ShGood link p s) :
    let x := s.tick
    ShGood link p x.1 ∧ x.1.t = s.t + 1 ∧ Packed p x.2.1 ∧
      (∀ r, s.certifier.fault = some r → x.2.1 = Array.replicate p.allocation none) ∧
      if (s.t + 1) % p.buckets = 0 then
        ∃ status, x.2.2 = some ⟨link, s.t / p.buckets, s.certifier.rows.push x.2.1, status⟩ ∧
          ∀ r, s.certifier.fault = some r → status = .notEstablished r
      else x.2.2 = none ∧ x.1.certifier.rows = s.certifier.rows.push x.2.1 ∧
        ∀ r, s.certifier.fault = some r → x.1.certifier.fault = some r := by
  obtain ⟨c1, prover, syncs, hc1, -, e1⟩ := declareStage_eq s
  have hT : s.certifier.params.buckets = p.buckets := by rw [hs.shape.params]
  rw [tick_eq, e1, hT]
  have hs1 : Shape c1 link p := hc1.shape hs.shape
  obtain ⟨k1, -, k3, k4, hs2, k6⟩ := handStage_keeps p.buckets { s with certifier := c1, prover, syncs } hs1
  generalize hs2e : handStage p.buckets { s with certifier := c1, prover, syncs } = s2 at k1 k3 k4 hs2 k6
  have hfault : ∀ r, s.certifier.fault = some r → s2.certifier = s.certifier := by
    intro r hr
    have e : c1 = s.certifier := hc1.of_fault hr
    subst e
    exact k6 r hr
  have hw2 : s2.certifier.window = s.certifier.window := k3.trans hc1.window
  have hr2 : s2.certifier.rows = s.certifier.rows := k4.trans hc1.rows
  have ht2 : s2.t = s.t := k1
  have hg2 : ShGood link p s2 := ⟨hs2, by rw [hr2, ht2, hs.rows], by rw [hw2, ht2, hs.window],
    by rw [hr2]; exact hs.packed⟩
  obtain ⟨f1, f2, f3, f4⟩ := finishStage_spec s2 hg2
  rw [ht2, hr2] at f4
  refine ⟨f1, f2.trans (by rw [ht2]), f3 ▸ certTick_row hs2, fun r hr => ?_, ?_⟩
  · rw [f3, hfault r hr, (certTick_fault hr).1, hs.shape.params]
  · split
    · rename_i hc
      simp only [hc, ↓reduceIte] at f4
      refine ⟨_, f4, fun r hr => ?_⟩
      rw [hfault r hr, hr]
    · rename_i hc
      simp only [hc, ↓reduceIte] at f4
      refine ⟨f4.1, f4.2.1, fun r hr => ?_⟩
      rw [f4.2.2, hfault r hr, hr]

end Properties.CertifierDevice.Proofs
