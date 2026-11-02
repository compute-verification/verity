import Proofs.Properties.CertifierDevice.Shaper

/-!
# The certifier device, proved: one step

A device state is `Good` for its configuration when its egress certifier has the grid's shape and holds the open
window's rows, and the ingress certifier is the configured link's. A step that is not a live egress tick is `Quiet`: it
keeps the configuration, faults the egress certifier at most and leaves its tick count, and keeps the ingress
certifier's link and parameters; its actions never write the link and publish only the ingress grid's records
(`step_quiet`). A live egress tick ticks the shaper once and writes its row, `r · B` bytes, then publishes the window's
record at its last bucket (`step_egress`).
-/

namespace Properties.CertifierDevice.Proofs

open _root_.CertifierDevice
open NetworkCertifier (Reason Status)

/-! ## Loops and folds -/

theorem Id.bind_inv {α β : Type _} (P : α → Prop) (Q : β → Prop) (x : Id α) (f : α → Id β) (hx : P x.run)
    (hf : ∀ a, P a → Q (f a).run) : Q (x >>= f).run := hf _ hx

theorem forIn_inv_arr {α β : Type _} (P : β → Prop) {l : Array α} {b : β} {f : α → β → Id (ForInStep β)}
    (h0 : P b) (hf : ∀ a ∈ l, ∀ b, P b → P (f a b).run.value) : P (forIn l b f).run := by
  rw [← Array.forIn_toList]
  exact forIn_inv P h0 (fun a ha => hf a (Array.mem_toList_iff.1 ha))

theorem foldl_inv {α β : Type _} (P : β → Prop) {l : Array α} {b : β} {f : β → α → β}
    (h0 : P b) (hf : ∀ a ∈ l, ∀ b, P b → P (f b a)) : P (l.foldl f b) := by
  rw [← Array.foldl_toList]
  have key : ∀ (m : List α) (b : β), P b → (∀ a ∈ m, a ∈ l) → P (m.foldl f b) := by
    intro m
    induction m with
    | nil => intro b hb _; exact hb
    | cons a m ih =>
      intro b hb hm
      exact ih _ (hf a (hm a (by simp)) b hb) (fun a' ha' => hm a' (by simp [ha']))
  exact key _ _ h0 (fun a ha => Array.mem_toList_iff.1 ha)

/-! ## Quiet steps -/

/-- The egress shaper, faulted at most, at the same tick, having handed and queued as many frames, at the same handing
bucket and with the same syncs. -/
def ShRel (s s' : Shaper) : Prop := CertRel s.certifier s'.certifier ∧ s'.t = s.t ∧ s'.next = s.next ∧
  s'.queued = s.queued ∧ s'.h = s.h ∧ s'.syncs = s.syncs

theorem ShRel.refl (s : Shaper) : ShRel s s := ⟨CertRel.refl _, rfl, rfl, rfl, rfl, rfl⟩
theorem ShRel.trans {s s' s'' : Shaper} (h : ShRel s s') (h' : ShRel s' s'') : ShRel s s'' :=
  ⟨h.1.trans h'.1, h'.2.1.trans h.2.1, h'.2.2.1.trans h.2.2.1, h'.2.2.2.1.trans h.2.2.2.1,
    h'.2.2.2.2.1.trans h.2.2.2.2.1, h'.2.2.2.2.2.trans h.2.2.2.2.2⟩
theorem ShRel.fault (s : Shaper) (r : Reason) : ShRel s (s.fault r) := ⟨Or.inr ⟨r, rfl⟩, rfl, rfl, rfl, rfl, rfl⟩
theorem ShRel.arrive (s : Shaper) (position : Nat) (frame : Frame) (ready : Int) :
    ShRel s (s.arrive position frame ready) := by
  unfold Shaper.arrive
  split
  · exact ShRel.fault _ _
  · split
    · exact ShRel.fault _ _
    · exact ⟨CertRel.refl _, rfl, rfl, rfl, rfl, rfl⟩

theorem ShGood.rel {link : String} {p : Params} {s s' : Shaper} (h : ShGood link p s) (hr : ShRel s s') :
    ShGood link p s' :=
  ⟨hr.1.shape h.shape, by rw [hr.1.rows, hr.2.1]; exact h.rows, by rw [hr.1.window, hr.2.1]; exact h.window,
    by rw [hr.1.rows]; exact h.packed⟩

/-- What a step other than a live egress tick keeps. -/
def Quiet (st st' : State) : Prop :=
  st'.config = st.config ∧ ShRel st.shaper st'.shaper ∧ st'.bucketer.certifier.link = st.bucketer.certifier.link ∧
    st'.bucketer.certifier.params = st.bucketer.certifier.params

theorem Quiet.refl (st : State) : Quiet st st := ⟨rfl, ShRel.refl _, rfl, rfl⟩
theorem Quiet.trans {st st' st'' : State} (h : Quiet st st') (h' : Quiet st' st'') : Quiet st st'' :=
  ⟨h'.1.trans h.1, h.2.1.trans h'.2.1, h'.2.2.1.trans h.2.2.1, h'.2.2.2.trans h.2.2.2⟩

theorem Quiet.same {st st' : State} (h1 : st'.config = st.config) (h2 : st'.shaper = st.shaper)
    (h3 : st'.bucketer.certifier = st.bucketer.certifier) : Quiet st st' :=
  ⟨h1, by rw [h2]; exact ShRel.refl _, by rw [h3], by rw [h3]⟩

theorem Quiet.shaper {st : State} {s : Shaper} (h : ShRel st.shaper s) : Quiet st { st with shaper := s } :=
  ⟨rfl, h, rfl, rfl⟩

theorem Quiet.bucketFault (st : State) (r : Reason) :
    Quiet st { st with bucketer := { st.bucketer with certifier := st.bucketer.certifier.doFault r } } :=
  ⟨rfl, ShRel.refl _, doFault_link _ _, doFault_params _ _⟩

theorem Quiet.setStream (st : State) (s : Nat) (x : Stream) : Quiet st (st.setStream s x) := Quiet.refl st

/-- The device's state, for its configuration `c`. -/
structure Good (c : Config) (st : State) : Prop where
  config : st.config = c
  shaper : ShGood c.egressLink c.egress st.shaper
  link : st.bucketer.certifier.link = c.ingressLink
  params : st.bucketer.certifier.params = c.ingress

theorem Good.quiet {c : Config} {st st' : State} (h : Good c st) (hq : Quiet st st') : Good c st' :=
  ⟨hq.1.trans h.config, h.shaper.rel hq.2.1, hq.2.2.1.trans h.link, hq.2.2.2.trans h.params⟩

theorem init_good (c : Config) : Good c (init c) := by
  refine ⟨rfl, ?_, rfl, rfl⟩
  unfold init Shaper.make
  dsimp only
  obtain ⟨hr, e⟩ := checkAdvice_rel
    { certifier := { link := c.egressLink, params := c.egress }, syncs := c.syncs, syncSet := c.syncSet,
      prover := c.prover.map fun p =>
        let l : LaggedChoices :=
          { prover := Prover.make c.syncSet c.egress.buckets p.rule p.envelope p.prior p.lag }
        if c.windows > 0 then l.start else l,
      windows := c.windows }
  refine ShGood.rel ⟨⟨rfl, rfl, by simp, by simp⟩, by simp, by simp, by simp⟩
    ⟨hr, checkAdvice_t _, by rw [e], by rw [e], by rw [e], by rw [e]⟩

/-! ## The handlers -/

/-- An action that is not the egress link's: no link write, and a record only of the ingress grid. -/
def IngressAct (link : String) (p : Params) (a : Action) : Prop :=
  (∀ b, a ≠ .writeLink b) ∧ ∀ r q, a = .publish r q → r.link = link ∧ q = p

theorem IngressAct.upstream (link : String) (p : Params) (s : Nat) (b : ByteArray) (d : Bool) :
    IngressAct link p (.writeUpstream s b d) := ⟨fun _ h => (nomatch h), fun _ _ h => (nomatch h)⟩

theorem IngressAct.connect (link : String) (p : Params) (s : Nat) : IngressAct link p (.connect s) :=
  ⟨fun _ h => (nomatch h), fun _ _ h => (nomatch h)⟩

theorem IngressAct.connectSalt (link : String) (p : Params) : IngressAct link p .connectSalt :=
  ⟨fun _ h => (nomatch h), fun _ _ h => (nomatch h)⟩

theorem IngressAct.writeSalt (link : String) (p : Params) (b : ByteArray) : IngressAct link p (.writeSalt b) :=
  ⟨fun _ h => (nomatch h), fun _ _ h => (nomatch h)⟩

/-- `releaseSalt` touches only the salt connection and its count. -/
theorem releaseSalt_keeps (st : State) (p : Nat) :
    (st.releaseSalt p).1.config = st.config ∧ (st.releaseSalt p).1.shaper = st.shaper ∧
      (st.releaseSalt p).1.bucketer = st.bucketer := by
  unfold State.releaseSalt
  dsimp only
  split <;> exact ⟨rfl, rfl, rfl⟩

theorem releaseSalt_quiet (st : State) (p : Nat) : Quiet st (st.releaseSalt p).1 :=
  have h := releaseSalt_keeps st p
  Quiet.same h.1 h.2.1 (by rw [h.2.2])

/-- `saltPump` touches only the salt connection, and writes salt frames only. -/
theorem saltPump_keeps (st : State) :
    st.saltPump.1.config = st.config ∧ st.saltPump.1.shaper = st.shaper ∧ st.saltPump.1.bucketer = st.bucketer ∧
      ∀ a ∈ st.saltPump.2, ∃ b, a = .writeSalt b := by
  unfold State.saltPump
  split
  · split
    · exact ⟨rfl, rfl, rfl, fun a ha => ⟨_, by simpa using ha⟩⟩
    · exact ⟨rfl, rfl, rfl, fun a ha => absurd ha (Array.not_mem_empty a)⟩
  · exact ⟨rfl, rfl, rfl, fun a ha => absurd ha (Array.not_mem_empty a)⟩

theorem saltPump_spec (st : State) : Quiet st st.saltPump.1 ∧ ∀ a ∈ st.saltPump.2, ∃ b, a = .writeSalt b :=
  have h := saltPump_keeps st
  ⟨Quiet.same h.1 h.2.1 (by rw [h.2.2.1]), h.2.2.2⟩

def PumpOK (st : State) (r : State × Array Action) : Prop :=
  Quiet st r.1 ∧ ∀ a ∈ r.2, ∃ s b d, a = .writeUpstream s b d

theorem pump_spec (st : State) (s : Nat) : PumpOK st (st.pump s) := by
  unfold State.pump
  dsimp only
  split
  · exact ⟨Quiet.refl _, fun a ha => absurd ha (Array.not_mem_empty a)⟩
  · refine Id.bind_inv (fun (r : Stream × Array Action × Nat) => ∀ a ∈ r.2.1, ∃ s b d, a = .writeUpstream s b d)
      _ _ _ ?_ ?_
    · refine forIn_inv_arr (fun (r : Stream × Array Action × Nat) => ∀ a ∈ r.2.1, ∃ s b d,
        a = .writeUpstream s b d) (fun a ha => absurd ha (Array.not_mem_empty a)) ?_
      intro y _ b hb
      split
      all_goals
        intro a ha
        rcases Array.mem_push.1 ha with ha | rfl
        · exact hb a ha
        · exact ⟨_, _, _, rfl⟩
    · intro r hr
      exact ⟨Quiet.setStream _ _ _, hr⟩

theorem opened_fst (st : State) (s : Nat) :
    (st.opened s).1 = ((st.setStream s { st.stream s with phase := .open }).pump s).1 := by
  unfold State.opened
  generalize (st.setStream s { st.stream s with phase := .open }).pump s = y
  rcases y with ⟨_, _⟩
  rfl

/-- An opened connection's actions: its ordinal header, or what `pump` writes. -/
theorem opened_mem {st : State} {s : Nat} {a : Action} (h : a ∈ (st.opened s).2) :
    a = .writeUpstream s (ordinalHeader s) false ∨
      a ∈ ((st.setStream s { st.stream s with phase := .open }).pump s).2 := by
  unfold State.opened at h
  generalize (st.setStream s { st.stream s with phase := .open }).pump s = y at h ⊢
  rcases y with ⟨_, acts⟩
  dsimp only at h ⊢
  split at h
  · rcases Array.mem_append.1 h with h | h
    · exact Or.inl (by simpa using h)
    · exact Or.inr h
  · exact Or.inr h

theorem opened_spec (st : State) (s : Nat) : PumpOK st (st.opened s) := by
  obtain ⟨p1, p2⟩ := pump_spec (st.setStream s { st.stream s with phase := .open }) s
  refine ⟨by rw [opened_fst]; exact (Quiet.setStream _ _ _).trans p1, fun a ha => ?_⟩
  rcases opened_mem ha with rfl | ha
  · exact ⟨_, _, _, rfl⟩
  · exact p2 a ha

theorem checkForwarding_quiet (st : State) (now : Nat) : Quiet st (st.checkForwarding now) := by
  unfold State.checkForwarding
  dsimp only
  refine Id.bind_inv (fun (r : State × Array Nat) => Quiet st r.1) _ _ _ ?_ ?_
  · refine forIn_inv_arr (fun (r : State × Array Nat) => Quiet st r.1) (Quiet.refl st) ?_
    intro s _ b hb
    split
    · exact hb
    · split
      · split
        · exact hb.trans ((Quiet.bucketFault _ _).trans (Quiet.refl _))
        · exact hb
      · exact hb
  · intro r hr
    split
    · exact hr.trans (Quiet.shaper (ShRel.fault _ _))
    · exact hr

theorem linkRead_quiet (st : State) (now : Nat) (data : ByteArray) : Quiet st (st.linkRead now data) := by
  unfold State.linkRead
  dsimp only
  split
  · exact Quiet.refl _
  · refine Id.bind_inv (fun (b : Bucketer) => b.certifier = st.bucketer.certifier) (Quiet st) _ _ ?_ ?_
    · rw [Std.Legacy.Range.forIn_eq_forIn_range']
      exact forIn_inv (fun (b : Bucketer) => b.certifier = st.bucketer.certifier) rfl (fun _ _ b hb => hb)
    · intro b hb
      exact Quiet.same rfl rfl hb

theorem egressFrame_quiet (st : State) (s position : Nat) (bytes : ByteArray) (ready : Int) :
    Quiet st (st.egressFrame s position bytes ready) := by
  unfold State.egressFrame
  dsimp only
  split
  · exact Quiet.shaper (ShRel.arrive _ _ _ _)
  · exact Quiet.shaper (ShRel.fault _ _)

theorem egressEvent_quiet (st : State) (s : Nat) (event : ByteArray) (ready : Int) :
    Quiet st (st.egressEvent s event ready) := by
  unfold State.egressEvent
  split
  · exact egressFrame_quiet _ _ _ _ _
  · exact Quiet.shaper (ShRel.fault _ _)

theorem egressRecord_quiet (st : State) (s position : Nat) (record : ByteArray) (ready : Int) :
    Quiet st (st.egressRecord s position record ready) := by
  unfold State.egressRecord
  split
  · exact egressFrame_quiet _ _ _ _ _
  · exact Quiet.shaper (ShRel.fault _ _)

theorem upstreamEnd_quiet (st : State) (s now : Nat) : Quiet st (st.upstreamEnd s now) := by
  unfold State.upstreamEnd
  dsimp only
  split
  · exact Quiet.setStream _ _ _
  · exact (Quiet.shaper (ShRel.fault _ _)).trans (Quiet.setStream _ _ _)

theorem upstreamRead_quiet (st : State) (s now : Nat) (data : ByteArray) : Quiet st (st.upstreamRead s now data) := by
  unfold State.upstreamRead
  dsimp only
  split
  · exact upstreamEnd_quiet _ _ _
  · split
    · refine Id.bind_inv (Quiet st) (Quiet st) _ _ ?_ (fun _ h => h)
      refine forIn_inv_arr _ (Quiet.setStream _ _ _) (fun u _ b hb => ?_)
      rcases u with ⟨position, record⟩
      exact hb.trans (egressRecord_quiet _ _ _ _ _)
    · refine Id.bind_inv (Quiet st) (Quiet st) _ _ ?_ (fun _ h => h)
      exact forIn_inv_arr _ (Quiet.setStream _ _ _) (fun _ _ b hb => hb.trans (egressEvent_quiet _ _ _ _))

/-! ## The bucketer before its certifier ticks -/

/-- `c'` is `c` with frames enqueued: the same link, parameters, rows and window, and `c`'s shape kept. -/
def EnqRel (c c' : Certifier) : Prop :=
  c'.link = c.link ∧ c'.params = c.params ∧ c'.rows = c.rows ∧ c'.window = c.window ∧
    ∀ link p, Shape c link p → Shape c' link p

theorem EnqRel.refl (c : Certifier) : EnqRel c c := ⟨rfl, rfl, rfl, rfl, fun _ _ h => h⟩

theorem EnqRel.trans {c c' c'' : Certifier} (h : EnqRel c c') (h' : EnqRel c' c'') : EnqRel c c'' :=
  ⟨h'.1.trans h.1, h'.2.1.trans h.2.1, h'.2.2.1.trans h.2.2.1, h'.2.2.2.1.trans h.2.2.2.1,
    fun l p hs => h'.2.2.2.2 l p (h.2.2.2.2 l p hs)⟩

theorem EnqRel.enqueue (c : Certifier) (f : ByteArray) : EnqRel c (c.enqueue f) :=
  ⟨enqueue_link c f, enqueue_params c f, enqueue_rows c f, enqueue_window c f, fun _ _ h => h.enqueue f⟩

theorem EnqRel.foldl (c : Certifier) (l : Array (Int × ByteArray)) :
    EnqRel c (l.foldl (fun c x => match x with | (_, f) => c.enqueue f) c) :=
  foldl_inv (EnqRel c) (EnqRel.refl c) (fun x _ c' h => by
    rcases x with ⟨_, f⟩
    exact h.trans (EnqRel.enqueue c' f))

theorem enqueue_first_eq (c : Certifier) (frame : ByteArray) :
    c.enqueue frame true = c ∨ (∃ r, c.enqueue frame true = c.doFault r) ∨
      (c.fault = none ∧ frame.size = c.params.frameBytes ∧
        c.enqueue frame true = { c with queue := #[frame] ++ c.queue }) := by
  unfold Certifier.enqueue
  by_cases h1 : c.fault.isSome
  · simp [h1]
  · simp only [h1, Bool.false_eq_true, ite_false]
    by_cases h2 : frame.size ≠ c.params.frameBytes
    · exact Or.inr (Or.inl ⟨.malformed, by simp [h2]⟩)
    · by_cases h3 : c.queue.size ≥ c.params.queueCapacity
      · exact Or.inr (Or.inl ⟨.overflow, by simp [h2, h3]⟩)
      · exact Or.inr (Or.inr ⟨by simpa using h1, by simpa using h2, by simp [h2, h3]⟩)

theorem Shape.enqueueFirst {c : Certifier} {link : String} {p : Params} (h : Shape c link p) (f : ByteArray) :
    Shape (c.enqueue f true) link p := by
  rcases enqueue_first_eq c f with he | ⟨r, he⟩ | ⟨hn, hsz, he⟩ <;> rw [he]
  · exact h
  · exact h.doFault r
  · refine ⟨h.link, h.params, ?_, ?_⟩
    · intro g hg
      rcases Array.mem_append.1 hg with hg | hg
      · rw [Array.mem_singleton.1 hg, hsz, h.params]
      · exact h.sizes g hg
    · simp [hn]

theorem EnqRel.enqueueFirst (c : Certifier) (f : ByteArray) : EnqRel c (c.enqueue f true) := by
  refine ⟨?_, ?_, ?_, ?_, fun _ _ h => h.enqueueFirst f⟩ <;>
    rcases enqueue_first_eq c f with he | ⟨r, he⟩ | ⟨-, -, he⟩ <;> rw [he] <;>
    simp [doFault_link, doFault_params, doFault_rows, doFault_window]

theorem takeOwn_rel (b : Bucketer) : b.takeOwn.t = b.t ∧ EnqRel b.certifier b.takeOwn.certifier := by
  unfold Bucketer.takeOwn
  dsimp only
  split
  · split
    · exact ⟨rfl, EnqRel.enqueueFirst _ _⟩
    · exact ⟨rfl, EnqRel.refl _⟩
  · exact ⟨rfl, EnqRel.refl _⟩

theorem admit_rel (a : Admission) (b : Bucketer) :
    (Bucketer.admit a b).t = b.t ∧ EnqRel b.certifier (Bucketer.admit a b).certifier := by
  unfold Bucketer.admit
  dsimp only
  have h0 : ∀ b' : Bucketer, b' = (if b.t % b.certifier.params.buckets = 0 then { b with counts := #[] } else b) →
      b'.t = b.t ∧ EnqRel b.certifier b'.certifier := by
    intro b' hb
    subst hb
    split <;> exact ⟨rfl, EnqRel.refl _⟩
  generalize hb : (if b.t % b.certifier.params.buckets = 0 then { b with counts := #[] } else b) = b0
  have hb0 := h0 b0 hb.symm
  refine Id.bind_inv (fun (b' : Bucketer) => b'.t = b.t ∧ EnqRel b.certifier b'.certifier)
    (fun (b' : Bucketer) => b'.t = b.t ∧ EnqRel b.certifier b'.certifier) _ _ ?_ (fun _ h => h)
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  refine forIn_inv (fun (b' : Bucketer) => b'.t = b.t ∧ EnqRel b.certifier b'.certifier) hb0 ?_
  intro _ _ b' hb'
  split
  · exact ⟨hb'.1, hb'.2.trans (EnqRel.foldl _ _)⟩
  · exact hb'

theorem enter_rel (b : Bucketer) : b.enter.t = b.t ∧ EnqRel b.certifier b.enter.certifier := by
  unfold Bucketer.enter
  split
  · exact admit_rel _ _
  · exact ⟨rfl, EnqRel.foldl _ _⟩

/-- What a bucketer tick enqueues before its certifier ticks: its own frame and the frames that enter. -/
theorem pre_rel (b : Bucketer) : b.takeOwn.enter.t = b.t ∧ EnqRel b.certifier b.takeOwn.enter.certifier :=
  ⟨(enter_rel _).1.trans (takeOwn_rel b).1, (takeOwn_rel b).2.trans (enter_rel _).2⟩

theorem follow_certifier (b : Bucketer) (row : Slots) : (b.follow row).certifier = b.certifier := by
  unfold Bucketer.follow; split <;> rfl

theorem follow_t (b : Bucketer) (row : Slots) : (b.follow row).t = b.t := by
  unfold Bucketer.follow; split <;> rfl

theorem bucketer_tick (b : Bucketer) :
    b.tick.1.certifier.link = b.certifier.link ∧ b.tick.1.certifier.params = b.certifier.params ∧
      ∀ r, b.tick.2.2 = some r → r.link = b.certifier.link := by
  obtain ⟨-, hc⟩ := pre_rel b
  unfold Bucketer.tick
  dsimp only
  generalize b.takeOwn.enter = b1 at hc ⊢
  obtain ⟨t1, t2, -, -, -⟩ := certTick_keeps b1.certifier
  generalize b1.certifier.tick = x at t1 t2 ⊢
  rcases x with ⟨c', row⟩
  dsimp only at t1 t2 ⊢
  simp only [follow_certifier]
  split
  · exact ⟨t1.trans hc.1, t2.trans hc.2.1, fun r h => nomatch h⟩
  · simp only [Certifier.closeSamplingBlock]
    exact ⟨t1.trans hc.1, t2.trans hc.2.1, fun r h => by cases h; exact t1.trans hc.1⟩

/-- `ingressTick` after its late check. -/
def ingressTail (st : State) (t : Nat) : State × Array Action := Id.run do
  let p := st.config.ingress
  let e := st.config.egress
  let mut st := st
  let (bucketer, row, record) := st.bucketer.tick
  st := { st with bucketer }
  let anchor := t * p.bucketMs / e.bucketMs
  let mut started := #[]
  for (slot, i) in (row.extract 0 (rowCount row)).zipIdx do
    if let some (_, w) := bucketer.own.find? (·.1 == i) then
      let salt := st.releaseSalt (w / (st.config.salt.map (·.pouwWindow)).getD 1)
      st := salt.1
      if let some x := salt.2 then started := started.push x
      continue
    let (kind, _, payload) := decodeFrame (slot.getD ByteArray.empty)
    if kind != kindData then continue
    let (requests, s, data, last) := st.requests.feed payload
    st := { st with requests }
    if !st.streams.contains s then
      st := st.setStream s {}
      started := started.push (State.Started.connect s)
    let mut x := st.stream s
    if x.phase == .open && !x.draining && x.out.isEmpty then started := started.push (.pump s)
    x := { x with out := x.out.push (data, last) }
    if last && x.anchor.isNone then
      x := { x with anchor := some anchor, window := some (t / p.buckets),
                    deadline := some (st.config.origin +
                      min (t * p.bucketMs * nsPerMs + st.config.slack) ((anchor + 1) * e.bucketMs * nsPerMs)) }
      st := { st with released := st.released.push s, requestsReleased := st.requestsReleased + 1 }
    st := st.setStream s x
  let mut acts := #[]
  if let some r := record then acts := acts.push (.publish r p)
  for x in started do
    match x with
    | .connect s => acts := acts.push (.connect s)
    | .pump s =>
      let (st', more) := st.pump s
      st := st'
      acts := acts ++ more
    | .saltConnect => acts := acts.push .connectSalt
    | .saltPump =>
      let (st', more) := st.saltPump
      st := st'
      acts := acts ++ more
  return (st, acts)

set_option maxHeartbeats 1000000 in
theorem ingressTick_eq (st : State) (now : Nat) :
    st.ingressTick now = ingressTail (if st.isLate st.config.ingress.bucketMs st.bucketer.t now then
      { st with bucketer := { st.bucketer with
        certifier := st.bucketer.certifier.doFault .monitorUnavailable } } else st) st.bucketer.t := by
  unfold State.ingressTick
  by_cases h : st.isLate st.config.ingress.bucketMs st.bucketer.t now = true
  · simp only [h, ↓reduceIte]
    exact rfl
  · simp only [h, Bool.false_eq_true, ↓reduceIte]
    exact rfl

theorem ingressTail_spec (st : State) (t : Nat) :
    Quiet st (ingressTail st t).1 ∧
      ∀ a ∈ (ingressTail st t).2, IngressAct st.bucketer.certifier.link st.config.ingress a := by
  obtain ⟨b1, b2, b3⟩ := bucketer_tick st.bucketer
  unfold ingressTail
  dsimp only
  generalize st.bucketer.tick = x at b1 b2 b3 ⊢
  rcases x with ⟨b, row, record⟩
  dsimp only at b1 b2 b3 ⊢
  have q1 : Quiet st { st with bucketer := b } := ⟨rfl, ShRel.refl _, b1, b2⟩
  refine Id.bind_inv (fun (r : State × Array State.Started) => Quiet st r.1)
    (fun (r : State × Array Action) => Quiet st r.1 ∧
      ∀ a ∈ r.2, IngressAct st.bucketer.certifier.link st.config.ingress a) _ _ ?_ ?_
  · refine forIn_inv_arr (fun (r : State × Array State.Started) => Quiet st r.1) q1 ?_
    intro slot _ y hy
    repeat' split
    all_goals first
      | exact hy
      | exact hy.trans (Quiet.same rfl rfl rfl)
      | exact hy.trans (releaseSalt_quiet _ _)
  · intro y hy
    have h0 : ∀ acts : Array Action, (∀ a ∈ acts, IngressAct st.bucketer.certifier.link st.config.ingress a) →
        Quiet st (Id.run (forIn y.2 (y.1, acts) fun x r => match x with
          | .connect s => pure (ForInStep.yield (r.1, r.2.push (.connect s)))
          | .pump s => match r.1.pump s with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more))
          | .saltConnect => pure (ForInStep.yield (r.1, r.2.push .connectSalt))
          | .saltPump => match r.1.saltPump with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more)))).1 ∧
          ∀ a ∈ (Id.run (forIn y.2 (y.1, acts) fun x r => match x with
          | .connect s => pure (ForInStep.yield (r.1, r.2.push (.connect s)))
          | .pump s => match r.1.pump s with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more))
          | .saltConnect => pure (ForInStep.yield (r.1, r.2.push .connectSalt))
          | .saltPump => match r.1.saltPump with
            | (st', more) => pure (ForInStep.yield (st', r.2 ++ more)))).2,
            IngressAct st.bucketer.certifier.link st.config.ingress a := by
      intro acts hacts
      refine forIn_inv_arr (fun (r : State × Array Action) => Quiet st r.1 ∧
        ∀ a ∈ r.2, IngressAct st.bucketer.certifier.link st.config.ingress a) ⟨hy, hacts⟩ ?_
      intro x _ r hr
      split
      · refine ⟨hr.1, fun a ha => ?_⟩
        rcases Array.mem_push.1 ha with ha | rfl
        · exact hr.2 a ha
        · exact IngressAct.connect _ _ _
      · rename_i s _
        obtain ⟨p1, p2⟩ := pump_spec r.1 s
        generalize r.1.pump s = z at p1 p2 ⊢
        rcases z with ⟨st', more⟩
        refine ⟨hr.1.trans p1, fun a ha => ?_⟩
        rcases Array.mem_append.1 ha with ha | ha
        · exact hr.2 a ha
        · obtain ⟨s, b, d, rfl⟩ := p2 a ha
          exact IngressAct.upstream _ _ _ _ _
      · refine ⟨hr.1, fun a ha => ?_⟩
        rcases Array.mem_push.1 ha with ha | rfl
        · exact hr.2 a ha
        · exact IngressAct.connectSalt _ _
      · obtain ⟨p1, p2⟩ := saltPump_spec r.1
        generalize r.1.saltPump = z at p1 p2 ⊢
        rcases z with ⟨st', more⟩
        refine ⟨hr.1.trans p1, fun a ha => ?_⟩
        rcases Array.mem_append.1 ha with ha | ha
        · exact hr.2 a ha
        · obtain ⟨b, rfl⟩ := p2 a ha
          exact IngressAct.writeSalt _ _ _
    split
    · refine h0 _ (fun a ha => ?_)
      rcases Array.mem_push.1 ha with ha | rfl
      · exact absurd ha (Array.not_mem_empty a)
      · exact ⟨fun _ h => (nomatch h), fun r' q h => by cases h; exact ⟨b3 _ rfl, rfl⟩⟩
    · exact h0 _ (fun a ha => absurd ha (Array.not_mem_empty a))

theorem ingressTick_spec (st : State) (now : Nat) :
    Quiet st (st.ingressTick now).1 ∧
      ∀ a ∈ (st.ingressTick now).2, IngressAct st.bucketer.certifier.link st.config.ingress a := by
  rw [ingressTick_eq]
  split
  · obtain ⟨h1, h2⟩ := ingressTail_spec
      { st with bucketer := { st.bucketer with certifier := st.bucketer.certifier.doFault .monitorUnavailable } }
      st.bucketer.t
    refine ⟨(Quiet.bucketFault _ _).trans h1, fun a ha => ?_⟩
    have := h2 a ha
    rwa [doFault_link] at this
  · exact ingressTail_spec st st.bucketer.t

/-! ## Egress ticks -/

/-- `egressTick` after its late and backlog checks, for the tick `t`. -/
def egressTail (p : Params) (now backlog t : Nat) (st : State) : State × Array Action := Id.run do
  let mut st := st.checkForwarding now
  let (shaper, row, record) := st.shaper.tick
  st := { st with shaper }
  let mut acts := #[]
  if backlog ≤ 4 * (p.allocation * p.frameBytes) then
    let n := rowCount row
    let bytes := (row.extract 0 n).foldl (fun acc f => acc ++ f.getD ByteArray.empty) ByteArray.empty
    acts := acts.push (.writeLink (bytes ++ ⟨Array.replicate ((p.allocation - n) * p.frameBytes) 0⟩))
  if let some l := st.shaper.prover then
    if (t + l.prover.lag) % p.buckets = 0 && (t + l.prover.lag) / p.buckets < st.config.windows then
      st := { st with shaper := { st.shaper with prover := some l.start } }
  if let some r := record then acts := acts.push (.publish r p)
  return (st, acts)

/-- `egressTail` without the online prover's choice. -/
def egressShape (p : Params) (now backlog : Nat) (st : State) : State × Array Action := Id.run do
  let mut st := st.checkForwarding now
  let (shaper, row, record) := st.shaper.tick
  st := { st with shaper }
  let mut acts := #[]
  if backlog ≤ 4 * (p.allocation * p.frameBytes) then
    let n := rowCount row
    let bytes := (row.extract 0 n).foldl (fun acc f => acc ++ f.getD ByteArray.empty) ByteArray.empty
    acts := acts.push (.writeLink (bytes ++ ⟨Array.replicate ((p.allocation - n) * p.frameBytes) 0⟩))
  if let some r := record then acts := acts.push (.publish r p)
  return (st, acts)

/-- The online prover's choice at the end of the egress tick `t`. -/
def startAt (p : Params) (t : Nat) (st : State) : State :=
  match st.shaper.prover with
  | some l =>
    if (t + l.prover.lag) % p.buckets = 0 && (t + l.prover.lag) / p.buckets < st.config.windows then
      { st with shaper := { st.shaper with prover := some l.start } }
    else st
  | none => st

theorem egressTail_eq (p : Params) (now backlog t : Nat) (st : State) :
    egressTail p now backlog t st =
      (startAt p t (egressShape p now backlog st).1, (egressShape p now backlog st).2) := by
  unfold egressTail egressShape
  dsimp only
  generalize st.checkForwarding now = st1
  generalize st1.shaper.tick = x
  rcases x with ⟨sh, row, record⟩
  unfold startAt
  by_cases hb : backlog ≤ 4 * (p.allocation * p.frameBytes)
  all_goals simp only [hb, ↓reduceIte]
  all_goals rcases hp : sh.prover with _ | l
  all_goals rcases record with _ | r
  all_goals simp only [Id.run_pure, hp]
  all_goals try split_ifs
  all_goals rfl

theorem startAt_shaper (p : Params) (t : Nat) (st : State) :
    ∃ l, (startAt p t st).shaper = { st.shaper with prover := l } := by
  unfold startAt
  split
  · split
    · exact ⟨_, rfl⟩
    · exact ⟨st.shaper.prover, rfl⟩
  · exact ⟨st.shaper.prover, rfl⟩

theorem startAt_config (p : Params) (t : Nat) (st : State) : (startAt p t st).config = st.config := by
  unfold startAt
  split
  · split <;> rfl
  · rfl

theorem startAt_quiet (p : Params) (t : Nat) (st : State) : Quiet st (startAt p t st) := by
  obtain ⟨l, hl⟩ := startAt_shaper p t st
  refine ⟨startAt_config p t st, by rw [hl]; exact ⟨CertRel.refl _, rfl, rfl, rfl, rfl, rfl⟩, ?_, ?_⟩
  all_goals unfold startAt; split
  all_goals first | rfl | split <;> rfl

/-- `egressTick`'s late and backlog checks. -/
def egressPre (st : State) (now backlog : Nat) : State :=
  let p := st.config.egress
  let st1 := if st.isLate p.bucketMs st.shaper.t now then
    { st with shaper := st.shaper.fault .monitorUnavailable } else st
  if backlog > p.allocation * p.frameBytes then
    { st1 with shaper := st1.shaper.fault .deliveryRefused } else st1

theorem egressTick_eq (st : State) (now backlog : Nat) :
    st.egressTick now backlog =
      egressTail st.config.egress now backlog st.shaper.t (egressPre st now backlog) := by
  unfold State.egressTick egressPre
  by_cases h1 : st.isLate st.config.egress.bucketMs st.shaper.t now = true
  all_goals by_cases h2 : backlog > st.config.egress.allocation * st.config.egress.frameBytes
  all_goals simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]
  all_goals rfl

theorem egressPre_quiet (st : State) (now backlog : Nat) : Quiet st (egressPre st now backlog) := by
  unfold egressPre
  dsimp only
  have h1 : Quiet st (if st.isLate st.config.egress.bucketMs st.shaper.t now then
      { st with shaper := st.shaper.fault .monitorUnavailable } else st) := by
    split
    · exact Quiet.shaper (ShRel.fault _ _)
    · exact Quiet.refl _
  generalize (if st.isLate st.config.egress.bucketMs st.shaper.t now then
      { st with shaper := st.shaper.fault .monitorUnavailable } else st) = st1 at h1 ⊢
  split
  · exact h1.trans (Quiet.shaper (ShRel.fault _ _))
  · exact h1

/-- What one live egress tick does: the next shaper tick `y`, its row written as `r · B` bytes when the link's backlog
allows, and at the window's last bucket the window's record. -/
def EgressSpec (c : Config) (st : State) (backlog : Nat) (y : Shaper × Slots × Option BlockGrid)
    (x : State × Array Action) : Prop :=
  let T := c.egress.buckets
  Good c x.1 ∧ x.1.shaper.t = st.shaper.t + 1 ∧
    ∃ row status, Packed c.egress row ∧ row = y.2.1 ∧
      ((st.shaper.t + 1) % T = 0 → y.2.2.map BlockGrid.status = some status) ∧
      x.2.toList = (if backlog ≤ 4 * bucketBytes c.egress then [.writeLink (wire c.egress row)] else []) ++
        (if (st.shaper.t + 1) % T = 0 then
          [.publish ⟨c.egressLink, st.shaper.t / T, st.shaper.certifier.rows.push row, status⟩ c.egress]
        else []) ∧
      (∀ r, st.shaper.certifier.fault = some r → row = Array.replicate c.egress.allocation none ∧
        status = .notEstablished r) ∧
      ((st.shaper.t + 1) % T ≠ 0 → x.1.shaper.certifier.rows = st.shaper.certifier.rows.push row ∧
        ∀ r, st.shaper.certifier.fault = some r → x.1.shaper.certifier.fault = some r)

theorem egressShape_spec {c : Config} {st0 st : State} (hg : Good c st0) (hq : Quiet st0 st) (now backlog : Nat) :
    EgressSpec c st0 backlog (st.checkForwarding now).shaper.tick (egressShape c.egress now backlog st) := by
  have hq' := hq.trans (checkForwarding_quiet st now)
  have hg' := hg.quiet hq'
  have hsh := hg'.shaper
  obtain ⟨f1, f2, f3, f4, f5⟩ := tick_spec _ hsh
  have hof : ∀ r, st0.shaper.certifier.fault = some r →
      (st.checkForwarding now).shaper.certifier = st0.shaper.certifier := fun r hr => hq'.2.1.1.of_fault hr
  have ht : (st.checkForwarding now).shaper.t = st0.shaper.t := hq'.2.1.2.1
  have hrows : (st.checkForwarding now).shaper.certifier.rows = st0.shaper.certifier.rows := hq'.2.1.1.rows
  unfold egressShape
  dsimp only
  generalize (st.checkForwarding now) = st1 at hg' hsh f1 f2 f3 f4 f5 hof ht hrows ⊢
  rw [ht, hrows] at f5
  rw [ht] at f2
  generalize st1.shaper.tick = x at f1 f2 f3 f4 f5 ⊢
  rcases x with ⟨sh, row, record⟩
  dsimp only at f1 f2 f3 f4 f5 ⊢
  have hgood : Good c { st1 with shaper := sh } := ⟨hg'.config, f1, hg'.link, hg'.params⟩
  have hwire : codeBytes c.egress row = wire c.egress row := codeBytes_packed f3
  have hcfg : st1.config.egress = c.egress := by rw [hg'.config]
  have hfr : ∀ r, st0.shaper.certifier.fault = some r → row = Array.replicate c.egress.allocation none :=
    fun r hr => f4 r (by rw [hof r hr]; exact hr)
  have hlink : st1.shaper.certifier.link = c.egressLink := hsh.shape.link
  unfold EgressSpec
  by_cases hb : backlog ≤ 4 * (c.egress.allocation * c.egress.frameBytes)
  all_goals simp only [hb, ↓reduceIte]
  all_goals have hb' : (backlog ≤ 4 * bucketBytes c.egress) = (backlog ≤ 4 * (c.egress.allocation * c.egress.frameBytes)) := rfl
  all_goals split at f5
  · rename_i hc
    obtain ⟨status, hrec, hst⟩ := f5
    subst hrec
    refine ⟨hgood, f2, row, status, f3, rfl, fun _ => rfl, ?_,
      fun r hr => ⟨hfr r hr, hst r (by rw [hof r hr]; exact hr)⟩, fun h => absurd hc h⟩
    simp only [hb', hb, hc, ↓reduceIte, List.singleton_append, ← hwire]
    rfl
  · rename_i hc
    obtain ⟨hrec, hr1, hr2⟩ := f5
    subst hrec
    refine ⟨hgood, f2, row, match st0.shaper.certifier.fault with | none => .complete | some r => .notEstablished r,
      f3, rfl, fun h => absurd h hc, ?_, fun r hr => ⟨hfr r hr, by rw [hr]⟩,
      fun _ => ⟨hr1, fun r hr => hr2 r (by rw [hof r hr]; exact hr)⟩⟩
    simp only [hb', hb, hc, ↓reduceIte, List.append_nil, ← hwire]
    rfl
  · rename_i hc
    obtain ⟨status, hrec, hst⟩ := f5
    subst hrec
    refine ⟨hgood, f2, row, status, f3, rfl, fun _ => rfl, ?_,
      fun r hr => ⟨hfr r hr, hst r (by rw [hof r hr]; exact hr)⟩, fun h => absurd hc h⟩
    simp only [hb', hb, hc, ↓reduceIte, List.nil_append]
    rfl
  · rename_i hc
    obtain ⟨hrec, hr1, hr2⟩ := f5
    subst hrec
    refine ⟨hgood, f2, row, match st0.shaper.certifier.fault with | none => .complete | some r => .notEstablished r,
      f3, rfl, fun h => absurd h hc, ?_, fun r hr => ⟨hfr r hr, by rw [hr]⟩,
      fun _ => ⟨hr1, fun r hr => hr2 r (by rw [hof r hr]; exact hr)⟩⟩
    simp only [hb', hb, hc, ↓reduceIte, List.append_nil]
    rfl

theorem EgressSpec.start {c : Config} {st : State} {backlog : Nat} {y : Shaper × Slots × Option BlockGrid}
    {x : State × Array Action} (h : EgressSpec c st backlog y x) (p : Params) (t : Nat) :
    EgressSpec c st backlog y (startAt p t x.1, x.2) := by
  obtain ⟨l, hl⟩ := startAt_shaper p t x.1
  obtain ⟨g, ht, rest⟩ := h
  refine ⟨g.quiet (startAt_quiet p t x.1), by rw [hl]; exact ht, ?_⟩
  rw [hl]
  exact rest

theorem egressTail_spec {c : Config} {st0 st : State} (hg : Good c st0) (hq : Quiet st0 st) (now backlog t : Nat) :
    EgressSpec c st0 backlog (st.checkForwarding now).shaper.tick (egressTail c.egress now backlog t st) := by
  rw [egressTail_eq]
  exact (egressShape_spec hg hq now backlog).start c.egress t

/-! ## Steps -/

theorem done_false {c : Config} {st : State} (hg : Good c st) (ht : st.shaper.t < horizon c) : st.done = false := by
  unfold State.done State.ticks
  rw [hg.config]
  unfold horizon at ht
  simp [ht]

theorem step_live {c : Config} {st : State} (hg : Good c st) {now backlog : Nat} (ht : st.shaper.t < horizon c)
    (x : State × Array Action) (hx : st.egressTick now backlog = x) :
    step st (.tick .egress now backlog) = (x.1, x.2.toList) := by
  have ht' : st.shaper.t < st.ticks st.config.egress := by unfold State.ticks; rw [hg.config]; exact ht
  unfold step
  rw [done_false hg ht]
  simp only [Bool.false_eq_true, ↓reduceIte, ht', hx]

theorem step_live_fst {c : Config} {st : State} (hg : Good c st) {now backlog : Nat} (ht : st.shaper.t < horizon c)
    (x : State × Array Action) (hx : st.egressTick now backlog = x) : (step st (.tick .egress now backlog)).1 = x.1 := by
  rw [step_live hg ht x hx]

theorem step_live_snd {c : Config} {st : State} (hg : Good c st) {now backlog : Nat} (ht : st.shaper.t < horizon c)
    (x : State × Array Action) (hx : st.egressTick now backlog = x) :
    (step st (.tick .egress now backlog)).2 = x.2.toList := by
  rw [step_live hg ht x hx]

/-- **A live egress tick** is `egressTick`: its state, and its actions. Stated by component: the kernel would unfold
`egressTick` to compare a pair of its projections with it. -/
theorem step_egress_fst {c : Config} {st : State} (hg : Good c st) {now backlog : Nat}
    (ht : st.shaper.t < horizon c) : (step st (.tick .egress now backlog)).1 = (st.egressTick now backlog).1 :=
  step_live_fst hg ht _ rfl

theorem step_egress_snd {c : Config} {st : State} (hg : Good c st) {now backlog : Nat}
    (ht : st.shaper.t < horizon c) :
    (step st (.tick .egress now backlog)).2 = (st.egressTick now backlog).2.toList :=
  step_live_snd hg ht _ rfl

/-- The shaper an egress tick ticks: after the late and backlog checks and the forwarding check. -/
def egressMid (st : State) (now backlog : Nat) : Shaper := ((egressPre st now backlog).checkForwarding now).shaper

theorem egress_spec {c : Config} {st : State} (hg : Good c st) (now backlog : Nat) :
    EgressSpec c st backlog (egressMid st now backlog).tick (st.egressTick now backlog) := by
  rw [egressTick_eq, hg.config]
  have := egressPre_quiet st now backlog
  exact egressTail_spec hg this now backlog _

/-- **A step other than a live egress tick** is quiet, and its actions are the ingress grid's or the servers'. -/
theorem step_quiet {c : Config} {st : State} (hg : Good c st) (ev : Event)
    (hl : ¬ ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c) :
    Quiet st (step st ev).1 ∧ ∀ a ∈ (step st ev).2, IngressAct c.ingressLink c.ingress a := by
  have hlink : st.bucketer.certifier.link = c.ingressLink := hg.link
  have hin : st.config.ingress = c.ingress := by rw [hg.config]
  have none_ok : Quiet st st ∧ ∀ a ∈ (#[] : Array Action).toList, IngressAct c.ingressLink c.ingress a :=
    ⟨Quiet.refl _, fun a ha => by simp at ha⟩
  have pump_ok : ∀ st' s y, Quiet st st' → st'.pump s = y → Quiet st y.1 ∧
      ∀ a ∈ y.2.toList, IngressAct c.ingressLink c.ingress a := by
    intro st' s y hq hy
    subst hy
    obtain ⟨p1, p2⟩ := pump_spec st' s
    refine ⟨hq.trans p1, fun a ha => ?_⟩
    obtain ⟨s, b, d, rfl⟩ := p2 a (Array.mem_toList_iff.1 ha)
    exact IngressAct.upstream _ _ _ _ _
  have opened_ok : ∀ s y, st.opened s = y → Quiet st y.1 ∧
      ∀ a ∈ y.2.toList, IngressAct c.ingressLink c.ingress a := by
    intro s y hy
    subst hy
    obtain ⟨p1, p2⟩ := opened_spec st s
    refine ⟨p1, fun a ha => ?_⟩
    obtain ⟨s, b, d, rfl⟩ := p2 a (Array.mem_toList_iff.1 ha)
    exact IngressAct.upstream _ _ _ _ _
  have saltPump_ok : ∀ st' y, Quiet st st' → st'.saltPump = y → Quiet st y.1 ∧
      ∀ a ∈ y.2.toList, IngressAct c.ingressLink c.ingress a := by
    intro st' y hq hy
    subst hy
    obtain ⟨p1, p2⟩ := saltPump_spec st'
    refine ⟨hq.trans p1, fun a ha => ?_⟩
    obtain ⟨b, rfl⟩ := p2 a (Array.mem_toList_iff.1 ha)
    exact IngressAct.writeSalt _ _ _
  unfold step
  split
  · exact ⟨Quiet.refl _, fun a ha => by simp at ha⟩
  · split
    rename_i st' acts heq
    split at heq
    · rename_i now backlog
      split at heq
      · exact absurd ⟨now, backlog, rfl, by unfold State.ticks at *; rwa [hg.config] at *⟩ hl
      · cases heq; exact none_ok
    · split at heq
      · obtain ⟨i1, i2⟩ := ingressTick_spec st _
        rw [heq] at i1 i2
        refine ⟨i1, fun a ha => ?_⟩
        have := i2 a (Array.mem_toList_iff.1 ha)
        rwa [hlink, hin] at this
      · cases heq; exact none_ok
    · split at heq
      · cases heq; exact ⟨linkRead_quiet _ _ _, fun a ha => by simp at ha⟩
      · cases heq; exact none_ok
    · split at heq
      · exact opened_ok _ _ heq
      · cases heq; exact none_ok
    · split at heq
      · cases heq; exact ⟨(Quiet.bucketFault _ _).trans (Quiet.setStream _ _ _), fun a ha => by simp at ha⟩
      · cases heq; exact none_ok
    · split at heq
      · cases heq; exact ⟨upstreamRead_quiet _ _ _ _, fun a ha => by simp at ha⟩
      · cases heq; exact none_ok
    · dsimp only at heq
      split at heq
      · exact pump_ok _ _ _ (Quiet.setStream _ _ _) heq
      · cases heq; exact none_ok
    · split at heq
      · split at heq
        · refine saltPump_ok _ _ ?_ heq
          exact Quiet.same rfl rfl rfl
        · cases heq; exact none_ok
      · cases heq; exact none_ok
    · split at heq
      · split at heq
        · cases heq
          exact ⟨⟨rfl, ShRel.refl _, doFault_link _ _, doFault_params _ _⟩, fun a ha => by simp at ha⟩
        · cases heq; exact none_ok
      · cases heq; exact none_ok
    · split at heq
      · split at heq
        · refine saltPump_ok _ _ ?_ heq
          exact Quiet.same rfl rfl rfl
        · cases heq; exact none_ok
      · cases heq; exact none_ok

end Properties.CertifierDevice.Proofs
