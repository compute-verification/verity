import Properties.Proofs.CertifierDevice.Fifo

/-!
# The certifier device, proved: handing buckets and late frames

The shaper's `h` is the Program's handing rule (`NetworkCertifier.handingFrom`) run over the frames it handed, under any
syncs extending the ones it declared (`Extends`). Each frame is handed at a bucket at or after its `h`, and a frame handed
after its `h` leaves the certifier faulted: the tick that hands it is not clean (`tick_hand`). Over a trace
(`hinv`), the handing buckets of the frames handed are their `h`s, each at most the bucket that handed it, and while the
certifier is unfaulted its `queued` count is the Program's queue of those `h`s: the Program releases no later than the
device (`released_mono`), and a window whose start is unfaulted has caught up.
-/

namespace Properties.CertifierDevice.Proofs

open _root_.CertifierDevice
open NetworkCertifier (Reason Status ClockSync LogicalFrame)

/-! ## The handing rule -/

/-- A frame as the Program reads it: its bits, anchor and index. -/
def logical (f : Frame) : LogicalFrame := ⟨bitsOf f.bytes, f.anchor, f.index⟩

/-- One step of `NetworkCertifier.handingFrom` under the syncs `S`. -/
def hstep (T : Nat) (S : Array ClockSync) (last : Nat) (f : LogicalFrame) : Nat :=
  if hw : f.anchor / T < S.size then max (NetworkCertifier.eligible f S[f.anchor / T]) last
  else max (max last (S.size * T)) f.anchor

/-- `NetworkCertifier.handingFrom` under the syncs `S`. -/
abbrev hfrom (T : Nat) (S : Array ClockSync) (last : Nat) (fs : List LogicalFrame) : List Nat :=
  NetworkCertifier.handingFrom T S.size (fun w => S[w]) last fs

/-- The handing bucket of the last of `fs`, from `last`. -/
def hlast (T : Nat) (S : Array ClockSync) (last : Nat) (fs : List LogicalFrame) : Nat := fs.foldl (hstep T S) last

theorem hfrom_cons (T : Nat) (S : Array ClockSync) (last : Nat) (f : LogicalFrame) (fs : List LogicalFrame) :
    hfrom T S last (f :: fs) = hstep T S last f :: hfrom T S (hstep T S last f) fs := rfl

theorem hfrom_append (T : Nat) (S : Array ClockSync) (last : Nat) (xs ys : List LogicalFrame) :
    hfrom T S last (xs ++ ys) = hfrom T S last xs ++ hfrom T S (hlast T S last xs) ys := by
  induction xs generalizing last with
  | nil => rfl
  | cons x xs ih => rw [List.cons_append, hfrom_cons, hfrom_cons, ih]; rfl

theorem hfrom_length (T : Nat) (S : Array ClockSync) (last : Nat) (fs : List LogicalFrame) :
    (hfrom T S last fs).length = fs.length := by
  induction fs generalizing last with
  | nil => rfl
  | cons f fs ih => rw [hfrom_cons, List.length_cons, ih, List.length_cons]

theorem hlast_append (T : Nat) (S : Array ClockSync) (last : Nat) (xs ys : List LogicalFrame) :
    hlast T S last (xs ++ ys) = hlast T S (hlast T S last xs) ys := by
  unfold hlast; rw [List.foldl_append]

/-- `S` extends the syncs `syncs`: it agrees with them wherever they are declared. -/
def Extends (S syncs : Array ClockSync) : Prop := ∀ i < syncs.size, S[i]? = syncs[i]?

theorem Extends.refl (S : Array ClockSync) : Extends S S := fun _ _ => rfl

theorem Extends.trans {S S' S'' : Array ClockSync} (h : Extends S S') (h' : Extends S' S'') : Extends S S'' := by
  intro i hi
  have e := h' i hi
  have hi' : i < S'.size := by
    rcases Nat.lt_or_ge i S'.size with h1 | h1
    · exact h1
    · rw [Array.getElem?_eq_none h1, Array.getElem?_eq_getElem hi] at e; cases e
  rw [h i hi', e]

theorem Extends.push (S : Array ClockSync) (x : ClockSync) : Extends (S.push x) S := by
  intro i hi
  rw [Array.getElem?_push]
  simp [Nat.ne_of_lt hi]

theorem Extends.get {S syncs : Array ClockSync} (h : Extends S syncs) {i : Nat} {x : ClockSync}
    (hx : syncs[i]? = some x) : ∃ hi : i < S.size, S[i] = x := by
  have hi : i < syncs.size := by
    rcases Nat.lt_or_ge i syncs.size with h1 | h1
    · exact h1
    · rw [Array.getElem?_eq_none h1] at hx; cases hx
  have e := h i hi
  rw [hx] at e
  obtain ⟨hi', e'⟩ := Array.getElem?_eq_some_iff.1 e
  exact ⟨hi', e'⟩

theorem hstep_of {T : Nat} {S syncs : Array ClockSync} (h : Extends S syncs) {f : Frame} {sync : ClockSync}
    (hs : syncs[f.anchor / T]? = some sync) (last : Nat) :
    hstep T S last (logical f) = max (eligibleAt f.anchor f.index sync) last := by
  obtain ⟨hi, e⟩ := h.get hs
  have key : ∀ (i : Nat) (hi : i < S.size), i = f.anchor / T →
      max (NetworkCertifier.eligible (logical f) S[i]) last = max (eligibleAt f.anchor f.index sync) last := by
    rintro i hi rfl
    rw [e]
    rfl
  unfold hstep
  split
  · exact key _ _ rfl
  · exact absurd hi ‹_›

/-! ## The handing loop -/

theorem lateCert_none {s : Shaper} {h : Nat} (hf : (lateCert s h).fault = none) : s.t ≤ h := by
  unfold lateCert at hf
  split at hf
  · have := doFault_fault_isSome s.certifier .coverageNeverEstablished
    rw [hf] at this
    cases this
  · omega

/-- What the handing loop does to `h`: since `s1`, it ran the handing rule over the frames handed, each bucket at most
the tick, and at least the tick while the certifier stays unfaulted. -/
def HandH (T : Nat) (S : Array ClockSync) (s1 s : Shaper) : Prop :=
  s.h = hlast T S s1.h ((handList s1 s.next).map logical) ∧
    ∀ x ∈ hfrom T S s1.h ((handList s1 s.next).map logical), x ≤ s1.t ∧ (s.certifier.fault = none → s1.t ≤ x)

theorem handStage_h (T : Nat) (S : Array ClockSync) (s1 : Shaper) (hS : Extends S s1.syncs) :
    HandH T S s1 (handStage T s1) := by
  have hr := handStage_reach T s1
  have hi := handStage_inv T s1
  suffices ∀ s, Relation.ReflTransGen (HandStep T) s1 s → HandInv s1 s ∧ s.syncs = s1.syncs ∧ HandH T S s1 s from
    (this _ hr).2.2
  intro s hs
  induction hs with
  | refl =>
    refine ⟨?_, rfl, ?_, ?_⟩
    · refine ⟨le_rfl, by omega, rfl, fun k => ?_, by simp [handList], fun _ => by simp [handList]⟩
      rw [ite_eq_right_iff.2]
      intro h
      omega
    · simp [handList, hlast]
    · simp [handList, hfrom, NetworkCertifier.handingFrom]
  | @tail s s' _ hst ih =>
    obtain ⟨i, sy, hh1, hh2⟩ := ih
    have i' := i.step hst
    obtain ⟨i1, -, i3, i4, -, -⟩ := i
    obtain ⟨k1, k2, -⟩ := hst.keeps
    refine ⟨i', k2.trans sy, ?_⟩
    obtain ⟨frame, ready, sync, h, hand, hget, hsync, rfl, hht, -, rfl⟩ := hst
    have hget1 : s1.pending.get? s.next = some (frame, ready) := by
      rw [i4 s.next] at hget
      simpa using hget
    have hl := handList_succ s1 i1 hget1
    rw [sy] at hsync
    have hstp := hstep_of hS hsync s.h
    unfold HandH
    dsimp only
    rw [hl, List.map_append, hfrom_append, hlast_append, ← hh1]
    simp only [List.map_cons, List.map_nil, hfrom_cons]
    refine ⟨by rw [hlast, List.foldl_cons, List.foldl_nil, hstp], fun x hx => ?_⟩
    rcases List.mem_append.1 hx with hx | hx
    · refine ⟨(hh2 x hx).1, fun hf => (hh2 x hx).2 ?_⟩
      obtain ⟨h1, -⟩ := enqueue_none hf
      rw [← (lateCert_rel s _).of_none h1]
      exact h1
    · rcases List.mem_cons.1 hx with rfl | hx
      swap
      · simp [hfrom, NetworkCertifier.handingFrom] at hx
      rw [hstp]
      refine ⟨i3 ▸ hht, fun hf => ?_⟩
      obtain ⟨h1, -⟩ := enqueue_none hf
      have := lateCert_none h1
      omega

/-! ## One tick -/

theorem finishStage_h (T : Nat) (s : Shaper) :
    (finishStage T s).1.h = s.h ∧ (finishStage T s).1.syncs = s.syncs := by
  unfold finishStage
  dsimp only
  split
  · exact ⟨rfl, rfl⟩
  · split
    · split
      · rw [(checkAdvice_rel _).2]; exact ⟨rfl, rfl⟩
      · rw [(checkAdvice_rel _).2]; exact ⟨rfl, rfl⟩
    · rw [(checkAdvice_rel _).2]; exact ⟨rfl, rfl⟩

/-- A tick is clean when it leaves the certifier unfaulted, or at a window's close publishes a complete record. -/
def Clean (T : Nat) (s : Shaper) (x : Shaper × Slots × Option Record) : Prop :=
  if (s.t + 1) % T = 0 then x.2.2.map Record.status = some .complete else x.1.certifier.fault = none

/-- **One tick, for the handing buckets.** Under syncs `S` extending the ones it leaves, a tick runs the handing rule from
`h` over the frames it hands, each bucket at most the tick; a frame handed after its bucket makes the tick unclean and
leaves nothing queued unless it leaves the certifier faulted. -/
theorem tick_hand {link : String} {p : Params} (s : Shaper) (hs : ShGood link p s) (S : Array ClockSync)
    (hS : Extends S s.tick.1.syncs) :
    let x := s.tick
    let hl := hfrom p.buckets S s.h ((handList s x.1.next).map logical)
    x.1.h = hlast p.buckets S s.h ((handList s x.1.next).map logical) ∧ Extends x.1.syncs s.syncs ∧
      ∀ y ∈ hl, y ≤ s.t ∧ (y < s.t → ¬ Clean p.buckets s x ∧ (x.1.certifier.fault = none → x.1.queued = 0)) := by
  obtain ⟨c1, prover, syncs, hc1, hsy, e1⟩ := declareStage_eq s
  have hT : s.certifier.params.buckets = p.buckets := by rw [hs.shape.params]
  have hs1 : Shape c1 link p := hc1.shape hs.shape
  obtain ⟨k1, k2, k3, k4, hs2, -⟩ := handStage_keeps p.buckets { s with certifier := c1, prover, syncs } hs1
  unfold Clean
  dsimp only
  rw [tick_eq, e1, hT] at hS ⊢
  have hL : ∀ m, handList s m = handList { s with certifier := c1, prover, syncs } m := fun _ => rfl
  obtain ⟨fh, fs⟩ := finishStage_h p.buckets (handStage p.buckets { s with certifier := c1, prover, syncs })
  rw [fs, k2] at hS
  have hH := handStage_h p.buckets S { s with certifier := c1, prover, syncs } hS
  generalize handStage p.buckets { s with certifier := c1, prover, syncs } = s2 at k1 k2 k3 k4 hs2 hH fh fs ⊢
  obtain ⟨hH1, hH2⟩ := hH
  dsimp only at k1 k2 k3 k4 hH1 hH2
  have hg2 : ShGood link p s2 := ⟨hs2, by rw [k4, hc1.rows, k1, hs.rows], by rw [k3, hc1.window, k1, hs.window],
    by rw [k4, hc1.rows]; exact hs.packed⟩
  obtain ⟨f1, -, -, f4⟩ := finishStage_fifo p.buckets s2
  obtain ⟨-, -, -, g4⟩ := finishStage_spec s2 hg2
  rw [k1] at f4 g4
  rw [f1, fh]
  refine ⟨hH1, ?_, fun y hy => ⟨(hH2 y hy).1, fun hlt => ?_⟩⟩
  · rw [fs, k2]
    rcases hsy with rfl | ⟨z, rfl⟩
    · exact Extends.refl _
    · exact Extends.push _ _
  have hf2 : s2.certifier.fault ≠ none := fun hf => absurd ((hH2 y hy).2 hf) (by omega)
  obtain ⟨r, hr⟩ := Option.ne_none_iff_exists'.1 hf2
  by_cases hcl : (s.t + 1) % p.buckets = 0
  · simp only [hcl, ↓reduceIte] at g4 ⊢
    refine ⟨?_, fun hf => ?_⟩
    · rw [g4, hr]
      simp
    · rcases (f4 hcl hf).2 with h | h
      · exact h
      · exact absurd h hf2
  · simp only [hcl, ↓reduceIte] at g4 ⊢
    rw [g4.2.2, hr]
    exact ⟨by simp, fun h => by cases h⟩

/-! ## The Program releases no later -/

theorem rsum_zero (t : Nat) : rsum (fun _ => 0) t = 0 := by
  induction t with
  | zero => rfl
  | succ t ih => simp [rsum, ih]

/-- The counts of `l` before `t` sum to how many of `l` are before `t`. -/
theorem rsum_count_lt (l : List Nat) (t : Nat) : rsum (fun u => l.count u) t = l.countP (· < t) := by
  induction l with
  | nil => simpa using rsum_zero t
  | cons a l ih =>
    have e : rsum (fun u => (a :: l).count u) t =
        rsum (fun u => l.count u + if a = u then 1 else 0) t := rsum_congr fun u _ => by simp [List.count_cons]
    rw [e, rsum_add, ih, rsum_ind, List.countP_cons]
    simp

theorem countP_lt_mono {hs as : List Nat} (h : List.Forall₂ (· ≤ ·) hs as) (t : Nat) :
    as.countP (· < t) ≤ hs.countP (· < t) := by
  induction h with
  | nil => simp
  | @cons a b _ _ hab _ ih =>
    simp only [List.countP_cons, decide_eq_true_eq]
    by_cases hb : b < t
    · simp only [hb, show a < t by omega, ↓reduceIte]; omega
    · simp only [hb, ↓reduceIte]; split <;> omega

/-- **The Program releases no later**: a FIFO whose arrivals are never behind another's has released at least as many. -/
theorem released_mono (S : Nat) (f g : Nat → Nat) (h : ∀ u, rsum g u ≤ rsum f u) (t : Nat) :
    rsum (fun u => (NetworkCertifier.released S g u : Nat)) t ≤
      rsum (fun u => (NetworkCertifier.released S f u : Nat)) t := by
  induction t with
  | zero => exact le_rfl
  | succ t ih =>
    have cf := released_conserve S f t
    have cg := released_conserve S g t
    have h1 := h (t + 1)
    simp only [rsum, NetworkCertifier.released] at cf cg h1 ih ⊢
    omega

theorem forall₂_append {α β : Type _} {R : α → β → Prop} {a c : List α} {b d : List β} (h : List.Forall₂ R a b)
    (h' : List.Forall₂ R c d) : List.Forall₂ R (a ++ c) (b ++ d) := by
  induction h with
  | nil => exact h'
  | cons hab _ ih => exact List.Forall₂.cons hab ih

theorem forall₂_const {α : Type _} {R : Nat → Nat → Prop} (t : Nat) :
    ∀ (l : List Nat) (m : List α), l.length = m.length → (∀ y ∈ l, R y t) → List.Forall₂ R l (m.map fun _ => t)
  | [], [], _, _ => .nil
  | y :: l, _ :: m, hl, h => .cons (h y (by simp)) (forall₂_const t l m (by simpa using hl) (fun z hz => h z (by simp [hz])))
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

theorem forall₂_lt {hs as : List Nat} (h : List.Forall₂ (· ≤ ·) hs as) {t : Nat} (ha : ∀ a ∈ as, a < t) :
    ∀ y ∈ hs, y < t := by
  induction h with
  | nil => simp
  | @cons a b _ _ hab _ ih =>
    intro y hy
    rcases List.mem_cons.1 hy with rfl | hy
    · exact lt_of_le_of_lt hab (ha b (by simp))
    · exact ih (fun x hx => ha x (by simp [hx])) y hy

theorem forall₂_eq_of {R : Nat → Nat → Prop} {l m : List Nat} (h : List.Forall₂ R l m) (hR : ∀ a b, R a b → b ∈ m → a = b) :
    l = m := by
  induction h with
  | nil => rfl
  | @cons a b _ _ hab _ ih =>
    rw [hR a b hab (by simp), ih (fun x y hx hy => hR x y hx (by simp [hy]))]

/-! ## Over a trace -/

/-- The handing buckets the Program gives the frames handed, `hp`, under the syncs `S`. -/
abbrev hlist (T : Nat) (S : Array ClockSync) (hp : List (Frame × Nat)) : List Nat :=
  hfrom T S 0 (hp.map fun x => logical x.1)

/-- What the shaper's handing is after handing `hp`, under syncs `S` extending its own: its `h` is the last of the
Program's handing buckets of `hp`, each at most the bucket that handed it, and while its certifier is unfaulted its
`queued` count is the Program's queue of them. -/
structure HInv (c : Config) (S : Array ClockSync) (hp : List (Frame × Nat)) (s : Shaper) : Prop where
  h : s.h = hlast c.egress.buckets S 0 (hp.map fun x => logical x.1)
  le : List.Forall₂ (· ≤ ·) (hlist c.egress.buckets S hp) (hp.map Prod.snd)
  queue : s.certifier.fault = none →
    NetworkCertifier.queue c.egress.allocation (fun u => (hlist c.egress.buckets S hp).count u) s.t = s.queued

theorem init_hinv (c : Config) (S : Array ClockSync) : HInv c S [] (init c).shaper := by
  unfold init Shaper.make
  dsimp only
  obtain ⟨-, e⟩ := checkAdvice_rel
    { certifier := { link := c.egressLink, params := c.egress }, syncs := c.syncs, syncSet := c.syncSet,
      prover := c.prover.map fun p =>
        let l : LaggedChoices :=
          { prover := Prover.make c.syncSet c.egress.buckets p.rule p.envelope p.prior p.lag }
        if c.windows > 0 then l.start else l,
      windows := c.windows }
  refine ⟨by rw [e]; rfl, by simp [hlist, hfrom, NetworkCertifier.handingFrom], fun _ => by rw [e]; rfl⟩

/-- The syncs only grow. -/
theorem step_syncs {c : Config} {st : State} (hg : Good c st) (ev : Event) :
    Extends (step st ev).1.shaper.syncs st.shaper.syncs := by
  by_cases hl : ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c
  · obtain ⟨now, backlog, rfl, ht⟩ := hl
    have hr := egressMid_rel st now backlog
    obtain ⟨l, hl⟩ := step_egress_shaper hg ht (now := now) (backlog := backlog)
    rw [hl, ← hr.2.2.2.2.2]
    exact (tick_hand _ (hg.shaper.rel hr) _ (Extends.refl _)).2.1
  · obtain ⟨q, -⟩ := step_quiet hg ev hl
    rw [q.2.1.2.2.2.2.2]
    exact Extends.refl _

theorem pre_extends (c : Config) (tr : List Event) {m N : Nat} (hm : m ≤ N) (hN : N ≤ tr.length) :
    Extends (pre c tr N).shaper.syncs (pre c tr m).shaper.syncs := by
  induction N with
  | zero => obtain rfl : m = 0 := by omega
            exact Extends.refl _
  | succ N ih =>
    rcases Nat.lt_or_ge m (N + 1) with h | h
    · rw [pre_succ c tr N (by omega)]
      exact (step_syncs (prefix_good c tr N).1 _).trans (ih (by omega) (by omega))
    · obtain rfl : m = N + 1 := by omega
      exact Extends.refl _

/-- **One step, for the handing buckets.** -/
theorem step_hinv {c : Config} {st : State} {hp : List (Frame × Nat)} {S : Array ClockSync} (hg : Good c st)
    (hf : FifoInv c.egress.allocation hp st.shaper) (hi : HInv c S hp st.shaper) (ev : Event)
    (hS : Extends S (step st ev).1.shaper.syncs) :
    HInv c S (hp ++ (handedAt st ev).map (·, st.shaper.t)) (step st ev).1.shaper := by
  have hf' := step_fifo hg hf ev
  by_cases hl : ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c
  · obtain ⟨now, backlog, rfl, ht⟩ := hl
    have hr := egressMid_rel st now backlog
    have hgs : ShGood c.egressLink c.egress (egressMid st now backlog) := hg.shaper.rel hr
    obtain ⟨l, hl⟩ := step_egress_shaper hg ht (now := now) (backlog := backlog)
    rw [handedAt_egress hg ht] at hf' ⊢
    rw [hl] at hS hf' ⊢
    obtain ⟨r1, r2, -, r4, r5, -⟩ := hr
    rw [← r2]
    rw [← r2] at hf'
    generalize egressMid st now backlog = s at hgs r1 r2 r4 r5 hS hf' ⊢
    have tk := (tick_spec _ hgs).2.1
    obtain ⟨-, f2, -, f4⟩ := fifo_tick _ hgs
    obtain ⟨th1, -, th3⟩ := tick_hand s hgs S hS
    generalize s.tick = x at tk f2 f4 th1 th3 hS hf' ⊢
    generalize handList s x.1.next = L at f2 f4 th1 th3 hf' ⊢
    dsimp only at tk f2 f4 th1 th3 hf' ⊢
    have hmap : (hp ++ L.map (·, s.t)).map (fun x => logical x.1) = hp.map (fun x => logical x.1) ++ L.map logical := by
      simp [Function.comp_def]
    have hsnd : (hp ++ L.map (·, s.t)).map Prod.snd = hp.map Prod.snd ++ L.map fun _ => s.t := by
      simp [Function.comp_def]
    have hl' : hlist c.egress.buckets S (hp ++ L.map (·, s.t)) =
        hlist c.egress.buckets S hp ++ hfrom c.egress.buckets S s.h (L.map logical) := by
      rw [hlist, hmap, hfrom_append, ← hi.h, r5]
    have hpast : ∀ y ∈ hlist c.egress.buckets S hp, y < s.t := by
      refine forall₂_lt hi.le fun a ha => ?_
      obtain ⟨z, hz, rfl⟩ := List.mem_map.1 ha
      rw [r2]
      exact hf.past z hz
    refine ⟨?_, ?_, fun hx => ?_⟩
    · rw [th1, hmap, hlast_append, ← hi.h, r5]
    · rw [hl', hsnd]
      refine forall₂_append hi.le (forall₂_const _ _ _ (by rw [hfrom_length, List.length_map]) ?_)
      intro y hy
      exact (th3 y hy).1
    · show NetworkCertifier.queue _ _ x.1.t = x.1.queued
      rw [tk]
      by_cases hq0 : x.1.queued = 0
      · rw [hq0]
        have hle' : List.Forall₂ (· ≤ ·) (hlist c.egress.buckets S (hp ++ L.map (·, s.t)))
            ((hp ++ L.map (·, s.t)).map Prod.snd) := by
          rw [hl', hsnd]
          refine forall₂_append hi.le (forall₂_const _ _ _ (by rw [hfrom_length, List.length_map]) ?_)
          intro y hy
          exact (th3 y hy).1
        generalize hP : hp ++ L.map (·, s.t) = P at hle' hf' ⊢
        have hlenP : (hlist c.egress.buckets S P).length = P.length := by
          rw [hlist, hfrom_length, List.length_map]
        have hAf : rsum (fun u => (hlist c.egress.buckets S P).count u) (s.t + 1) = P.length := by
          rw [rsum_count_lt, List.countP_eq_length.2, hlenP]
          intro y hy
          have := forall₂_lt hle' (t := s.t + 1) (fun a ha => by
            obtain ⟨z, hz, rfl⟩ := List.mem_map.1 ha
            have := hf'.past z hz
            rw [tk] at this
            exact this) y hy
          simpa using this
        have hAg : rsum (fun u => (P.map Prod.snd).count u) (s.t + 1) = P.length := by
          rw [rsum_count _ _ (fun y hy => by
            obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hy
            have := hf'.past z hz
            rw [tk] at this
            exact this), List.length_map]
        have hmono := released_mono c.egress.allocation (fun u => (hlist c.egress.buckets S P).count u)
          (fun u => (P.map Prod.snd).count u) (fun u => by
            rw [rsum_count_lt, rsum_count_lt]; exact countP_lt_mono hle' u) (s.t + 1)
        have cf := released_conserve c.egress.allocation (fun u => (hlist c.egress.buckets S P).count u) (s.t + 1)
        have cg := released_conserve c.egress.allocation (fun u => (P.map Prod.snd).count u) (s.t + 1)
        have hq := hf'.queued
        rw [hq0, tk] at hq
        rw [hAf] at cf
        rw [hAg, ← hq] at cg
        omega
      · have hall : ∀ y ∈ hfrom c.egress.buckets S s.h (L.map logical), y = s.t := by
          intro y hy
          obtain ⟨h1, h2⟩ := th3 y hy
          by_contra hne
          exact hq0 ((h2 (by omega)).2 hx)
        have hs0 : s.certifier.fault = none := by
          rcases f4 hx with ⟨h1, -⟩ | ⟨-, h2⟩
          · exact h1
          · exact absurd h2 hq0
        have hst0 : st.shaper.certifier.fault = none := by rw [← r1.of_none hs0]; exact hs0
        have hq := hi.queue hst0
        rw [← r2, ← r4] at hq
        rw [hl']
        have hlen : (hfrom c.egress.buckets S s.h (L.map logical)).length = L.length := by
          rw [hfrom_length, List.length_map]
        generalize hfrom c.egress.buckets S s.h (L.map logical) = hl at hall hlen
        simp only [NetworkCertifier.queue]
        have hcnt : ∀ u, (hlist c.egress.buckets S hp ++ hl).count u =
            (hlist c.egress.buckets S hp).count u + if u = s.t then L.length else 0 := by
          intro u
          rw [List.count_append]
          congr 1
          split
          · rename_i hu; subst hu; rw [List.count_eq_length.2 (fun y hy => (hall y hy).symm), hlen]
          · rename_i hu
            exact List.count_eq_zero_of_not_mem fun hm => hu (hall u hm)
        rw [queue_congr (g := fun u => (hlist c.egress.buckets S hp).count u) (fun u hu => by
          rw [hcnt]; simp [Nat.ne_of_lt hu]), hq, hcnt]
        have hz : (hlist c.egress.buckets S hp).count s.t = 0 :=
          List.count_eq_zero_of_not_mem fun hm => absurd (hpast _ hm) (lt_irrefl _)
        rw [hz, f2]
        simp
  · obtain ⟨q, -⟩ := step_quiet hg ev hl
    rw [handedAt_quiet q.2.1.2.2.1, List.map_nil, List.append_nil]
    obtain ⟨r1, r2, -, r4, r5, -⟩ := q.2.1
    refine ⟨by rw [r5]; exact hi.h, hi.le, fun hx => ?_⟩
    rw [r2, r4]
    exact hi.queue (by rw [← r1.of_none hx]; exact hx)

theorem hinv (c : Config) (tr : List Event) {N : Nat} (hN : N ≤ tr.length) :
    ∀ m ≤ N, HInv c (pre c tr N).shaper.syncs (handed (init c) (tr.take m)) (pre c tr m).shaper := by
  intro m hm
  induction m with
  | zero => rw [List.take_zero]; exact init_hinv c _
  | succ m ih =>
    have hm' : m < tr.length := by omega
    rw [handed_take_succ c tr m hm']
    have hS := pre_extends c tr hm hN
    rw [pre_succ c tr m hm'] at hS ⊢
    exact step_hinv (prefix_good c tr m).1 (fifo_inv c tr m (by omega)) (ih (by omega)) _ hS

/-! ## A complete window -/

/-- **A complete window's ticks are clean.** If egress bucket `k`, its window's last, closes the window complete, every
egress tick of that window up to `k` is clean. -/
theorem clean_of_complete {c : Config} {tr : List Event} {n k backlog : Nat} (hE : EgressAt c tr n k backlog)
    (hc : (k + 1) % c.egress.buckets = 0) (hso : statusOf c tr (k / c.egress.buckets) = .complete)
    {m t b' now' : Nat} (hEm : EgressAt c tr m t b') (he' : tr[m]'(egressAt_lt hEm) = .tick .egress now' b')
    (htk : t ≤ k) (hwin : t / c.egress.buckets = k / c.egress.buckets) :
    Clean c.egress.buckets (egressMid (pre c tr m) now' b') (egressMid (pre c tr m) now' b').tick := by
  have hgm := (prefix_good c tr m).1
  have htm := egressAt_t hEm
  have hth : (pre c tr m).shaper.t < horizon c := htm ▸ hEm.2.2
  have hm := egressAt_lt hEm
  obtain ⟨l, hl⟩ := step_egress_shaper hgm hth (now := now') (backlog := b')
  have hpre : (pre c tr (m + 1)).shaper = { (egressMid (pre c tr m) now' b').tick.1 with prover := l } := by
    rw [pre_succ c tr m hm, he', hl]
  have hmid : (egressMid (pre c tr m) now' b').t = t := by rw [(egressMid_rel _ now' b').2.1, htm]
  have ht1 : (pre c tr (m + 1)).shaper.t = t + 1 := by
    rw [hpre]
    change (egressMid (pre c tr m) now' b').tick.1.t = _
    rw [(tick_spec _ (hgm.shaper.rel (egressMid_rel _ now' b'))).2.1, hmid]
  obtain ⟨hT0, hkm, -⟩ := mod_last hc
  unfold Clean
  rw [hmid]
  by_cases htk' : t = k
  · subst htk'
    obtain rfl : m = n := egressAt_unique hEm hE
    simp only [hc, ↓reduceIte]
    obtain ⟨-, -, row, status, -, -, hs, -⟩ := egress_spec hgm now' b'
    have hs' := hs (by rw [htm]; exact hc)
    have hso' := statusOf_eq hE (t / c.egress.buckets) (by have := Nat.div_add_mod t c.egress.buckets; rw [Nat.mul_comm]; omega)
    rw [hso, he'] at hso'
    simp only [evStatus, hs', Option.getD_some] at hso'
    rw [hs', ← hso']
  · have hmod : ¬ (t + 1) % c.egress.buckets = 0 := by
      intro h
      have h1 := (mod_last h).2.1
      have h2 := Nat.div_add_mod t c.egress.buckets
      have h3 := Nat.div_add_mod k c.egress.buckets
      rw [hwin] at h2
      omega
    simp only [hmod, ↓reduceIte]
    cases hf : (egressMid (pre c tr m) now' b').tick.1.certifier.fault with
    | none => rfl
    | some r =>
      exfalso
      have hf' : (after (init c) (tr.take (m + 1))).shaper.certifier.fault = some r := by
        change (pre c tr (m + 1)).shaper.certifier.fault = _
        rw [hpre]
        exact hf
      have hmn : m < n := by
        by_contra hnm
        have := t_mono c tr n (m - n) (by omega)
        rw [show n + (m - n) = m by omega, egressAt_t hE, htm] at this
        omega
      have hwin' : k / c.egress.buckets = (after (init c) (tr.take (m + 1))).shaper.t / c.egress.buckets := by
        change _ = (pre c tr (m + 1)).shaper.t / _
        rw [ht1, succ_div_of_not_mod hmod, hwin]
      obtain ⟨grid, -, hout⟩ := failClosed_proof c tr (m + 1) r hf' n k backlog (by omega) hE hwin'
      rw [egress_acts hE] at hout
      have := Option.some.inj hout
      unfold egressActions at this
      simp only [hc, ↓reduceIte] at this
      split at this
      · simp only [List.cons_append, List.nil_append, List.cons.injEq, Action.publish.injEq,
          Record.mk.injEq] at this
        simp [hso] at this
      · simp only [List.nil_append, List.cons.injEq, Action.publish.injEq, Record.mk.injEq] at this
        simp [hso] at this

/-- **The handing buckets of a complete window**: every frame handed by its close at a bucket of the window was handed
at its handing bucket. -/
theorem window_hands {c : Config} {tr : List Event} {n k backlog : Nat} (hE : EgressAt c tr n k backlog)
    (hc : (k + 1) % c.egress.buckets = 0) (hso : statusOf c tr (k / c.egress.buckets) = .complete) :
    ∀ m ≤ n + 1, List.Forall₂ (fun h a => k / c.egress.buckets * c.egress.buckets ≤ a → h = a)
      (hlist c.egress.buckets (pre c tr (n + 1)).shaper.syncs (handed (init c) (tr.take m)))
      ((handed (init c) (tr.take m)).map Prod.snd) := by
  have hn := egressAt_lt hE
  intro m hm
  induction m with
  | zero => simp [hlist, hfrom, NetworkCertifier.handingFrom, handed]
  | succ m ih =>
    have hm' : m < tr.length := by omega
    have ih := ih (by omega)
    have hi := hinv c tr (N := n + 1) (by omega) m (by omega)
    rw [handed_take_succ c tr m hm']
    rw [hlist, List.map_append, hfrom_append, ← hi.h, List.map_append]
    refine forall₂_append ih ?_
    have hgm := (prefix_good c tr m).1
    by_cases hl : ∃ now b', tr[m] = .tick .egress now b' ∧ (pre c tr m).shaper.t < horizon c
    · obtain ⟨now', b', he', ht⟩ := hl
      have hr := egressMid_rel (pre c tr m) now' b'
      have hgs := hgm.shaper.rel hr
      rw [he', handedAt_egress hgm ht]
      simp only [List.map_map, Function.comp_def]
      by_cases hw : k / c.egress.buckets * c.egress.buckets ≤ (pre c tr m).shaper.t
      · have hEm := egressAt_of hm' he' ht
        have htk : (pre c tr m).shaper.t ≤ k := by
          by_contra hlt
          have := t_mono c tr m (n - m) (by omega)
          have hmn : m ≤ n := by omega
          rcases Nat.lt_or_ge m n with h | h
          · have h1 := t_mono c tr (m + 1) (n - (m + 1)) (by omega)
            rw [show m + 1 + (n - (m + 1)) = n by omega, egressAt_t hE, t_succ c tr m hm', he'] at h1
            simp only [run, tickView, Option.map_some, ht, and_true, ↓reduceIte] at h1
            omega
          · obtain rfl : m = n := by omega
            rw [egressAt_t hE] at hlt
            omega
        have hwin : (pre c tr m).shaper.t / c.egress.buckets = k / c.egress.buckets := by
          obtain ⟨hT0, -, -⟩ := mod_last hc
          apply le_antisymm (Nat.div_le_div_right htk)
          exact (Nat.le_div_iff_mul_le hT0).2 hw
        have hcl := clean_of_complete hE hc hso hEm he' htk hwin
        have hS : Extends (pre c tr (n + 1)).shaper.syncs (egressMid (pre c tr m) now' b').tick.1.syncs := by
          have := pre_extends c tr (m := m + 1) (N := n + 1) (by omega) (by omega)
          rw [pre_succ c tr m hm', he'] at this
          obtain ⟨l, hl⟩ := step_egress_shaper hgm ht (now := now') (backlog := b')
          rw [hl] at this
          exact this
        obtain ⟨-, -, th3⟩ := tick_hand _ hgs _ hS
        rw [hr.2.2.2.2.1, hr.2.1] at th3
        refine forall₂_const _ _ _ (by rw [hfrom_length, List.length_map]) fun y hy _ => ?_
        obtain ⟨h1, h2⟩ := th3 y hy
        by_contra hne
        exact (h2 (by omega)).1 hcl
      · refine forall₂_const _ _ _ (by rw [hfrom_length, List.length_map]) fun y _ h => absurd h hw
    · obtain ⟨q, -⟩ := step_quiet hgm tr[m] (by
        rintro ⟨now, b', he, ht⟩
        exact hl ⟨now, b', he, ht⟩)
      rw [handedAt_quiet q.2.1.2.2.1]
      simp [hfrom, NetworkCertifier.handingFrom]

end Properties.CertifierDevice.Proofs
