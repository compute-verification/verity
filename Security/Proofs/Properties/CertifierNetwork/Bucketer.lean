import Proofs.Properties.CertifierNetwork.Defs
import Proofs.Properties.CertifierDevice.Trace

/-!
# The ingress bucketer over a step

What a step does to the ingress bucketer and which records it publishes: a live ingress tick ticks the bucketer (faulted
at most first) and publishes its record, if any (`step_ingress`); a live egress tick leaves the bucketer faulted at most,
at the same tick (`step_egress_bucketer`); any other step does that and publishes nothing (`step_other`). A bucketer
of the ingress grid's shape (`BuGood`) keeps it through a tick, and a tick at a window's last bucket returns the window's
record: its `T` rows, each packed (`bucketerTick_spec`).
-/

namespace Properties.CertifierNetwork.Proofs

open _root_.CertifierDevice
open _root_.NetworkCertifier (Reason Status)
open Properties.CertifierDevice (Packed horizon)
open Properties.CertifierDevice.Proofs

/-! ## The bucketer, faulted at most -/

/-- The bucketer, faulted at most, at the same tick. -/
def BuRel (b b' : Bucketer) : Prop := CertRel b.certifier b'.certifier ∧ b'.t = b.t

theorem BuRel.refl (b : Bucketer) : BuRel b b := ⟨CertRel.refl _, rfl⟩

theorem BuRel.trans {b b' b'' : Bucketer} (h : BuRel b b') (h' : BuRel b' b'') : BuRel b b'' :=
  ⟨h.1.trans h'.1, h'.2.trans h.2⟩

theorem BuRel.of_eq {b b' : Bucketer} (h : b' = b) : BuRel b b' := h ▸ BuRel.refl b

theorem BuRel.fault (b : Bucketer) (r : Reason) : BuRel b { b with certifier := b.certifier.doFault r } :=
  ⟨Or.inr ⟨r, rfl⟩, rfl⟩

/-- The records `acts` publish are none. -/
theorem publishes_nil {acts : List Action} (h : ∀ a ∈ acts, ∀ r p, a ≠ .publish r p) : publishes acts = [] := by
  unfold publishes
  rw [List.filterMap_eq_nil_iff]
  intro a ha
  cases a with
  | publish r p => exact absurd rfl (h _ ha r p)
  | _ => rfl

theorem publishes_append (a b : List Action) : publishes (a ++ b) = publishes a ++ publishes b :=
  List.filterMap_append

/-! ## The handlers -/

theorem pump_bucketer (st : State) (s : Nat) : (st.pump s).1.bucketer = st.bucketer := by
  unfold State.pump
  dsimp only
  split
  · rfl
  · exact Id.bind_inv (fun _ => True) (fun (r : State × Array Action) => r.1.bucketer = st.bucketer) _ _ trivial
      (fun _ _ => rfl)

theorem pump_publishes (st : State) (s : Nat) : publishes (st.pump s).2.toList = [] := by
  refine publishes_nil fun a ha r p he => ?_
  obtain ⟨s, b, d, rfl⟩ := (pump_spec st s).2 a (Array.mem_toList_iff.1 ha)
  cases he

theorem saltPump_publishes (st : State) : publishes st.saltPump.2.toList = [] := by
  refine publishes_nil fun a ha r p he => ?_
  obtain ⟨b, rfl⟩ := (saltPump_keeps st).2.2.2 a (Array.mem_toList_iff.1 ha)
  cases he

theorem checkForwarding_bucketer (st : State) (now : Nat) : BuRel st.bucketer (st.checkForwarding now).bucketer := by
  unfold State.checkForwarding
  dsimp only
  refine Id.bind_inv (fun (r : State × Array Nat) => BuRel st.bucketer r.1.bucketer)
    (fun (r : State) => BuRel st.bucketer r.bucketer) _ _ ?_ ?_
  · refine forIn_inv_arr (fun (r : State × Array Nat) => BuRel st.bucketer r.1.bucketer) (BuRel.refl _) ?_
    intro s _ b hb
    split
    · exact hb
    · split
      · split
        · exact hb.trans (BuRel.fault _ _)
        · exact hb
      · exact hb
  · intro r hr
    split
    · exact hr
    · exact hr

theorem linkRead_bucketer (st : State) (now : Nat) (data : ByteArray) :
    BuRel st.bucketer (st.linkRead now data).bucketer := by
  unfold State.linkRead
  dsimp only
  split
  · exact BuRel.refl _
  · refine Id.bind_inv (fun (b : Bucketer) => BuRel st.bucketer b) (fun (r : State) => BuRel st.bucketer r.bucketer) _ _ ?_
      (fun _ h => h)
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    exact forIn_inv (fun (b : Bucketer) => BuRel st.bucketer b) (BuRel.refl _) (fun _ _ b hb => hb)

theorem egressFrame_bucketer (st : State) (s position : Nat) (bytes : ByteArray) (ready : Int) :
    (st.egressFrame s position bytes ready).bucketer = st.bucketer := by
  unfold State.egressFrame
  dsimp only
  split <;> rfl

theorem egressEvent_bucketer (st : State) (s : Nat) (event : ByteArray) (ready : Int) :
    (st.egressEvent s event ready).bucketer = st.bucketer := by
  unfold State.egressEvent
  split
  · exact egressFrame_bucketer _ _ _ _ _
  · rfl

theorem egressRecord_bucketer (st : State) (s position : Nat) (record : ByteArray) (ready : Int) :
    (st.egressRecord s position record ready).bucketer = st.bucketer := by
  unfold State.egressRecord
  split
  · exact egressFrame_bucketer _ _ _ _ _
  · rfl

theorem upstreamEnd_bucketer (st : State) (s now : Nat) : (st.upstreamEnd s now).bucketer = st.bucketer := by
  unfold State.upstreamEnd
  dsimp only
  split <;> rfl

theorem upstreamRead_bucketer (st : State) (s now : Nat) (data : ByteArray) :
    (st.upstreamRead s now data).bucketer = st.bucketer := by
  unfold State.upstreamRead
  dsimp only
  split
  · exact upstreamEnd_bucketer _ _ _
  · split
    · refine Id.bind_inv (fun (r : State) => r.bucketer = st.bucketer) (fun (r : State) => r.bucketer = st.bucketer) _ _
        ?_ (fun _ h => h)
      refine forIn_inv_arr (fun (r : State) => r.bucketer = st.bucketer) rfl (fun u _ b hb => ?_)
      rcases u with ⟨position, record⟩
      exact (egressRecord_bucketer _ _ _ _ _).trans hb
    · refine Id.bind_inv (fun (r : State) => r.bucketer = st.bucketer) (fun (r : State) => r.bucketer = st.bucketer) _ _
        ?_ (fun _ h => h)
      exact forIn_inv_arr (fun (r : State) => r.bucketer = st.bucketer) rfl
        (fun _ _ b hb => (egressEvent_bucketer _ _ _ _).trans hb)

theorem opened_bucketer (st : State) (s : Nat) : (st.opened s).1.bucketer = st.bucketer := by
  rw [opened_fst, pump_bucketer]
  rfl

theorem opened_publishes (st : State) (s : Nat) : publishes (st.opened s).2.toList = [] := by
  refine publishes_nil fun a ha r p he => ?_
  obtain ⟨s, b, d, rfl⟩ := (opened_spec st s).2 a (Array.mem_toList_iff.1 ha)
  cases he

theorem egressPre_bucketer (st : State) (now backlog : Nat) : (egressPre st now backlog).bucketer = st.bucketer := by
  unfold egressPre
  by_cases h1 : st.isLate st.config.egress.bucketMs st.shaper.t now = true
  all_goals by_cases h2 : backlog > st.config.egress.allocation * st.config.egress.frameBytes
  all_goals simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]

theorem startAt_bucketer (p : Params) (t : Nat) (st : State) : (startAt p t st).bucketer = st.bucketer := by
  unfold startAt
  split
  · split <;> rfl
  · rfl

theorem egressShape_bucketer (p : Params) (now backlog : Nat) (st : State) :
    (egressShape p now backlog st).1.bucketer = (st.checkForwarding now).bucketer := by
  unfold egressShape
  dsimp only
  generalize st.checkForwarding now = st1
  generalize st1.shaper.tick = x
  rcases x with ⟨sh, row, record⟩
  dsimp only
  split <;> split <;> rfl

theorem egressTick_bucketer (st : State) (now backlog : Nat) :
    BuRel st.bucketer (st.egressTick now backlog).1.bucketer := by
  rw [egressTick_eq, egressTail_eq]
  dsimp only
  rw [startAt_bucketer, egressShape_bucketer]
  have := checkForwarding_bucketer (egressPre st now backlog) now
  rwa [egressPre_bucketer] at this

/-- `ingressTail` ticks the bucketer and publishes its record, if any. -/
theorem ingressTail_bucketer (st : State) (t : Nat) :
    (ingressTail st t).1.bucketer = st.bucketer.tick.1 ∧
      publishes (ingressTail st t).2.toList = st.bucketer.tick.2.2.toList := by
  unfold ingressTail
  dsimp only
  generalize st.bucketer.tick = x
  rcases x with ⟨b, row, record⟩
  dsimp only
  refine Id.bind_inv (fun (r : State × Array State.Started) => r.1.bucketer = b)
    (fun (r : State × Array Action) => r.1.bucketer = b ∧ publishes r.2.toList = record.toList) _ _ ?_ ?_
  · refine forIn_inv_arr (fun (r : State × Array State.Started) => r.1.bucketer = b) rfl ?_
    intro slot _ y hy
    repeat' split
    all_goals first
      | exact hy
      | exact (releaseSalt_keeps _ _).2.2.trans hy
  · intro y hy
    have h0 : ∀ acts : Array Action, publishes acts.toList = record.toList →
        (Id.run (forIn y.2 (y.1, acts) fun x r => match x with
          | .connect s => pure (ForInStep.yield (r.1, r.2.push (.connect s)))
          | .pump s => match r.1.pump s with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more))
          | .saltConnect => pure (ForInStep.yield (r.1, r.2.push .connectSalt))
          | .saltPump => match r.1.saltPump with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more)))).1.bucketer = b ∧
          publishes (Id.run (forIn y.2 (y.1, acts) fun x r => match x with
          | .connect s => pure (ForInStep.yield (r.1, r.2.push (.connect s)))
          | .pump s => match r.1.pump s with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more))
          | .saltConnect => pure (ForInStep.yield (r.1, r.2.push .connectSalt))
          | .saltPump => match r.1.saltPump with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more)))).2.toList = record.toList := by
      intro acts hacts
      refine forIn_inv_arr (fun (r : State × Array Action) => r.1.bucketer = b ∧
        publishes r.2.toList = record.toList) ⟨hy, hacts⟩ ?_
      intro x _ r hr
      split
      · refine ⟨hr.1, ?_⟩
        show publishes (r.2.push _).toList = record.toList
        rw [Array.toList_push, publishes_append, hr.2]
        exact List.append_nil _
      · rename_i s _
        have p1 := pump_bucketer r.1 s
        have p2 := pump_publishes r.1 s
        generalize r.1.pump s = z at p1 p2 ⊢
        rcases z with ⟨st', more⟩
        refine ⟨p1.trans hr.1, ?_⟩
        show publishes (r.2 ++ more).toList = record.toList
        rw [Array.toList_append, publishes_append, hr.2, p2, List.append_nil]
      · refine ⟨hr.1, ?_⟩
        show publishes (r.2.push _).toList = record.toList
        rw [Array.toList_push, publishes_append, hr.2]
        exact List.append_nil _
      · have p1 := (saltPump_keeps r.1).2.2.1
        have p2 := saltPump_publishes r.1
        generalize r.1.saltPump = z at p1 p2 ⊢
        rcases z with ⟨st', more⟩
        refine ⟨p1.trans hr.1, ?_⟩
        show publishes (r.2 ++ more).toList = record.toList
        rw [Array.toList_append, publishes_append, hr.2, p2, List.append_nil]
    cases record with
    | none => exact h0 #[] rfl
    | some r => exact h0 #[.publish r _] rfl

/-! ## Steps -/

theorem done_false_ingress {st : State} (ht : st.bucketer.t < st.ticks st.config.ingress) : st.done = false := by
  unfold State.done
  simp [Nat.not_le.2 ht]

/-- **A live ingress tick** ticks the bucketer, faulted at most, and publishes its record, if any. -/
theorem step_ingress {c : Config} {st : State} (hg : Good c st) {now backlog : Nat}
    (ht : st.bucketer.t < c.windows * c.ingress.buckets) :
    ∃ b0, BuRel st.bucketer b0 ∧ (step st (.tick .ingress now backlog)).1.bucketer = b0.tick.1 ∧
      publishes (step st (.tick .ingress now backlog)).2 = b0.tick.2.2.toList := by
  have ht' : st.bucketer.t < st.ticks st.config.ingress := by unfold State.ticks; rw [hg.config]; exact ht
  have hs : step st (.tick .ingress now backlog) = ((st.ingressTick now).1, (st.ingressTick now).2.toList) := by
    unfold step
    rw [done_false_ingress ht']
    simp only [Bool.false_eq_true, ↓reduceIte, ht']
  rw [hs]
  dsimp only
  rw [ingressTick_eq]
  split
  · obtain ⟨h1, h2⟩ := ingressTail_bucketer
      { st with bucketer := { st.bucketer with certifier := st.bucketer.certifier.doFault .monitorUnavailable } }
      st.bucketer.t
    exact ⟨_, BuRel.fault _ _, h1, h2⟩
  · obtain ⟨h1, h2⟩ := ingressTail_bucketer st st.bucketer.t
    exact ⟨_, BuRel.refl _, h1, h2⟩

/-- A live egress tick leaves the bucketer faulted at most, at the same tick. -/
theorem step_egress_bucketer {c : Config} {st : State} (hg : Good c st) {now backlog : Nat}
    (ht : st.shaper.t < horizon c) : BuRel st.bucketer (step st (.tick .egress now backlog)).1.bucketer := by
  rw [step_egress_fst hg ht]
  exact egressTick_bucketer _ _ _

/-- **Any other step** leaves the bucketer faulted at most, at the same tick, and publishes nothing. -/
theorem step_other {c : Config} {st : State} (hg : Good c st) (ev : Event)
    (he : ¬ ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c)
    (hi : ¬ ∃ now backlog, ev = .tick .ingress now backlog ∧ st.bucketer.t < c.windows * c.ingress.buckets) :
    BuRel st.bucketer (step st ev).1.bucketer ∧ publishes (step st ev).2 = [] := by
  have none_ok : BuRel st.bucketer st.bucketer ∧ publishes (#[] : Array Action).toList = [] := ⟨BuRel.refl _, rfl⟩
  have pump_ok : ∀ (st' : State) (s : Nat) (y : State × Array Action), st'.bucketer = st.bucketer → st'.pump s = y →
      BuRel st.bucketer y.1.bucketer ∧ publishes y.2.toList = [] := by
    intro st' s y hq hy
    subst hy
    exact ⟨BuRel.of_eq ((pump_bucketer st' s).trans hq), pump_publishes st' s⟩
  have opened_ok : ∀ (s : Nat) (y : State × Array Action), st.opened s = y →
      BuRel st.bucketer y.1.bucketer ∧ publishes y.2.toList = [] := by
    intro s y hy
    subst hy
    exact ⟨BuRel.of_eq (opened_bucketer st s), opened_publishes st s⟩
  have saltPump_ok : ∀ (st' : State) (y : State × Array Action), st'.bucketer = st.bucketer → st'.saltPump = y →
      BuRel st.bucketer y.1.bucketer ∧ publishes y.2.toList = [] := by
    intro st' y hq hy
    subst hy
    exact ⟨BuRel.of_eq ((saltPump_keeps st').2.2.1.trans hq), saltPump_publishes st'⟩
  unfold step
  split
  · exact ⟨BuRel.refl _, rfl⟩
  · split
    rename_i st' acts heq
    split at heq
    · rename_i now backlog
      split at heq
      · exact absurd ⟨now, backlog, rfl, by unfold State.ticks at *; rwa [hg.config] at *⟩ he
      · cases heq; exact none_ok
    · rename_i now backlog
      split at heq
      · exact absurd ⟨now, backlog, rfl, by unfold State.ticks at *; rwa [hg.config] at *⟩ hi
      · cases heq; exact none_ok
    · split at heq
      · cases heq; exact ⟨linkRead_bucketer _ _ _, rfl⟩
      · cases heq; exact none_ok
    · split at heq
      · exact opened_ok _ _ heq
      · cases heq; exact none_ok
    · split at heq
      · cases heq; exact ⟨BuRel.fault _ _, rfl⟩
      · cases heq; exact none_ok
    · split at heq
      · cases heq; exact ⟨BuRel.of_eq (upstreamRead_bucketer _ _ _ _), rfl⟩
      · cases heq; exact none_ok
    · dsimp only at heq
      split at heq
      · exact pump_ok _ _ _ (by rfl) heq
      · cases heq; exact none_ok
    · split at heq
      · split at heq
        · refine saltPump_ok _ _ ?_ heq
          rfl
        · cases heq; exact none_ok
      · cases heq; exact none_ok
    · split at heq
      · split at heq
        · cases heq; exact ⟨BuRel.fault _ _, rfl⟩
        · cases heq; exact none_ok
      · cases heq; exact none_ok
    · split at heq
      · split at heq
        · refine saltPump_ok _ _ ?_ heq
          rfl
        · cases heq; exact none_ok
      · cases heq; exact none_ok

/-! ## A bucketer tick -/

/-- A bucketer whose certifier has the ingress grid's shape, holds the open window's rows, packed, and is at its
window. -/
structure BuGood (c : Config) (b : Bucketer) : Prop where
  shape : Shape b.certifier c.ingressLink c.ingress
  rows : b.certifier.rows.size = b.t % c.ingress.buckets
  window : b.certifier.window = b.t / c.ingress.buckets
  packed : ∀ row ∈ b.certifier.rows, Packed c.ingress row

theorem BuGood.rel {c : Config} {b b' : Bucketer} (h : BuGood c b) (hr : BuRel b b') : BuGood c b' :=
  ⟨hr.1.shape h.shape, by rw [hr.1.rows, hr.2]; exact h.rows, by rw [hr.1.window, hr.2]; exact h.window,
    by rw [hr.1.rows]; exact h.packed⟩

theorem init_buGood (c : Config) : BuGood c (init c).bucketer :=
  ⟨⟨rfl, rfl, by simp [init], by simp [init]⟩, by simp [init], by simp [init], by simp [init]⟩

/-- **A bucketer tick**: it keeps the grid's shape, counts the tick, and at a window's last bucket (only there) returns
the window's record, of the ingress link and the window, its `T` rows packed. -/
theorem bucketerTick_spec {c : Config} {b : Bucketer} (hb : BuGood c b) (hT : 0 < c.ingress.buckets) :
    BuGood c b.tick.1 ∧ b.tick.1.t = b.t + 1 ∧
      ((b.t + 1) % c.ingress.buckets = 0 → b.tick.2.2.isSome) ∧
      ∀ r, b.tick.2.2 = some r → (b.t + 1) % c.ingress.buckets = 0 ∧ r.link = c.ingressLink ∧
        r.window = b.t / c.ingress.buckets ∧ r.grid.size = c.ingress.buckets ∧
        ∀ row ∈ r.grid, Packed c.ingress row := by
  obtain ⟨ht1, -, -, hr1, hw1, hs⟩ := pre_rel b
  have hs1 := hs _ _ hb.shape
  unfold Bucketer.tick
  dsimp only
  generalize b.takeOwn.enter = b1 at ht1 hr1 hw1 hs1 ⊢
  simp only [follow_certifier, follow_t, ht1]
  generalize b1.certifier = c1 at hr1 hw1 hs1 ⊢
  obtain ⟨t1, t2, t3, t4, t5⟩ := certTick_keeps c1
  have hrow := certTick_row hs1
  have hsh := hs1.tick
  generalize hx : c1.tick = x at t1 t2 t3 t4 t5 hrow hsh ⊢
  rcases x with ⟨c2, row⟩
  dsimp only at t1 t2 t3 t4 t5 hrow hsh ⊢
  have hP : c2.params.buckets = c.ingress.buckets := by rw [hsh.params]
  rw [hP]
  by_cases hm : (b.t + 1) % c.ingress.buckets = 0
  · simp only [hm, ne_eq, not_true_eq_false, ↓reduceIte, Id.run_pure]
    have hw := succ_div_of_mod hm
    obtain ⟨-, hlast, -⟩ := mod_last hm
    refine ⟨⟨hsh.closeSamplingBlock, by simp [Certifier.closeSamplingBlock, hm], ?_,
      by simp [Certifier.closeSamplingBlock]⟩, trivial, fun _ => rfl, ?_⟩
    · simp only [Certifier.closeSamplingBlock, t3, hw1, hb.window, hw]
    · rintro r ⟨⟩
      refine ⟨trivial, by simp [Certifier.closeSamplingBlock, t1, hs1.link],
        by simp [Certifier.closeSamplingBlock, t3, hw1, hb.window], ?_, ?_⟩
      · simp only [Certifier.closeSamplingBlock, t4, hr1, Array.size_push, hb.rows, hlast]
        omega
      · intro row' hr'
        simp only [Certifier.closeSamplingBlock, t4, hr1] at hr'
        rcases Array.mem_push.1 hr' with hr' | rfl
        · exact hb.packed row' hr'
        · exact hrow
  · simp only [hm, ne_eq, not_false_eq_true, ↓reduceIte, Id.run_pure]
    refine ⟨⟨hsh, ?_, ?_, ?_⟩, trivial, fun h => h.elim, fun r hr => nomatch hr⟩
    · rw [t4, hr1, Array.size_push, hb.rows]; exact (succ_mod_of_not_mod hm).symm
    · rw [t3, hw1, hb.window]; exact (succ_div_of_not_mod hm).symm
    · intro row' hr'
      rw [t4, hr1] at hr'
      rcases Array.mem_push.1 hr' with hr' | rfl
      · exact hb.packed row' hr'
      · exact hrow

end Properties.CertifierNetwork.Proofs
