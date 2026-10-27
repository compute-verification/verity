import Proofs.Flock.Claims.ZK.FrameAsks
import Proofs.Flock.Claims.ZK.FrameSched

/-!
# A rep's scheduled rounds against its plan

The loop requests a rep's rounds at the plan's counts (`Session.repPlan`), and the frame checks the device against the
rep's schedule so far (`Loop.schedSoFar`): the algebraic phase's rounds (`Skeleton.rounds`, counted in `FrameCount`),
then the opening's (`Drive.openStep`), fed the coins released so far, with the round it asks for next. This file shows
that schedule's counts are a prefix of the plan's.

* `NA`, `naLift`, `piop_na`: on the algebraic phase's coins alone (`stepAsks coins false`) the PIOP asks for nothing,
  through `Sched.repPiop_sim` at a `Sched.Lift` that ignores its right side, and returns the extras and two claims more.
* `repOpening_ot`, `openStep_sp`: past the PIOP's coin check, `openStep` asks for the ring switch's counts, then
  Ligerito's (`openList`).
* `feed_prefix`: fed coins, a step program's rounds and its next ask are a prefix of its asks.
* `sched_prefix`: the schedule so far squeezes a prefix of `replayCoins` without its zeros, then `openList`.
* `plan_sched`: at the placeholder at a link of `spec.points` points, each round of the schedule so far squeezes the
  plan's count for it (`repPlan_list`: the plan is those asks, then the inner proof's; `sched_facts`: every `--zk`
  schedule has a lane round and counts within a squeeze's, by `decide +kernel`).
-/

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)

/-- **A step program that asks for nothing**: where it ends, with a value in `Q`. -/
def NA {α : Type} (Q : α → Prop) : Step α → Prop
  | .done a => Q a
  | .fail _ => True
  | .ask .. => False

theorem na_bind {α β : Type} {Q : α → Prop} {R : β → Prop} {f : α → Step β} (hf : ∀ a, Q a → NA R (f a)) :
    ∀ s : Step α, NA Q s → NA R (s.bind f)
  | .done a, h => hf a h
  | .fail _, _ => trivial
  | .ask .., h => h.elim

theorem na_mono {α : Type} {Q Q' : α → Prop} (h : ∀ a, Q a → Q' a) : ∀ s : Step α, NA Q s → NA Q' s
  | .done a, hs => h a hs
  | .fail _, _ => trivial
  | .ask .., hs => hs.elim

theorem step_pure_bind {α β : Type} (a : α) (f : α → Step β) : (pure a >>= f : Step β) = f a := rfl

/-- **The opening's runs on the algebraic phase's coins alone** (`stepAsks coins false`) as a `Lift`: they ask for
nothing. -/
def naLift (coins : Got) : Sched.Lift (M := OpenM) (stepAsks coins false) where
  R S x _ := ∀ got, NA (fun r => ∃ b, S r.1 b) (x.run got)
  pure h := fun _ => ⟨_, h⟩
  bind := by
    intro α β γ δ S T x y f g hx hf got
    rw [StateT.run_bind]
    exact na_bind (fun r ⟨b, hb⟩ => hf r.1 b hb r.2) _ (hx got)
  throw _ _ := fun _ => trivial
  liftOk h := fun _ => ⟨_, h⟩
  round := by
    intro c c' _ got
    unfold stepAsks
    dsimp only
    simp only [StateT.run_bind, StateT.run_get, step_pure_bind]
    rcases coins[got.size]? with _ | cs
    · exact trivial
    · simp only [StateT.run_bind, StateT.run_pure, step_pure_bind]
      split
      · exact trivial
      · exact ⟨_, rfl⟩

/-- **The PIOP on the algebraic phase's coins asks for nothing**, and where it ends, its claims are the extras and
two more. -/
theorem piop_na (coins : Got) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hd : st.digest.size < 2 ^ 64) (hcn : cap.size < 2 ^ 64)
    (hr : root.size < 2 ^ 64) (hm : st.m < 2 ^ 64) (t : Transcript) (ht : Sched.TrAlike t t) (got : Got) :
    NA (fun r => r.1.1.2.2.size = extras.size + 2)
      (((_root_.Flock.Firewall.Finish.repPiop (stepAsks coins false) st lay cap root extras mk).run t).run got) :=
  na_mono (fun _ ⟨_, ⟨_, hs⟩, _⟩ => hs) _
    (Sched.repPiop_sim (L := naLift coins) st lay cap cap root extras mk rfl hd hcn hr hm t t ht got)

theorem sp_mono {α : Type} {Q Q' : α → List ℕ → Prop} (h : ∀ a ns, Q a ns → Q' a ns) :
    ∀ (s : Step α) (ns : List ℕ), SP Q s ns → SP Q' s ns
  | .done a, ns, hs => h a ns hs
  | .fail _, _, _ => trivial
  | .ask _ _ k, _, ⟨ns', e, hk⟩ => ⟨ns', e, fun cs => sp_mono h (k cs) ns' (hk cs)⟩

theorem sp_na_bind {α β : Type} {Q : α → Prop} {R : β → List ℕ → Prop} {f : α → Step β} {ns : List ℕ}
    (hf : ∀ a, Q a → SP R (f a) ns) : ∀ s : Step α, NA Q s → SP R (s.bind f) ns
  | .done a, h => hf a h
  | .fail _, _ => trivial
  | .ask .., h => h.elim

theorem sp_run' {α : Type} {p : TMW OpenM α} {t : Transcript} {s : Got} {ns : List ℕ}
    (h : SP (fun _ _ => True) ((p.run t).run s) ns) : SP (fun _ _ => True) ((p.run' t).run' s) ns := by
  refine sp_bind (Q := fun _ _ => True) (fun _ _ _ => ?_) _ _ (sp_bind (fun _ _ _ => ?_) _ _ h) <;> exact trivial

theorem sp_getThe_bind {β : Type} {R : β × Transcript × Got → List ℕ → Prop} {f : Got → TMW OpenM β} {t : Transcript}
    {got : Got} {ns : List ℕ} (h : SP (fun r => R (r.1.1, r.1.2, r.2)) (((f got).run t).run got) ns) :
    SP (fun r => R (r.1.1, r.1.2, r.2)) (((getThe Got >>= f).run t).run got) ns := h

/-- **A rep's opening past the algebraic phase**: the ring switch's asks, then Ligerito's. -/
theorem repOpening_ot (coins : Got) (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule)
    (lay : _root_.Flock.Zk.PadLayout) (tPad rep : ℕ) (g : Array F128) (cm0 : _root_.Flock.Firewall.Finish.Commit)
    (claims : Array _root_.Flock.Claim) (h : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (hc : claims.size < 2 ^ 64) (hk : 1 ≤ sch.k0) (hn : sch.n < 2 ^ 62)
    (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 63) :
    OT (Past coins) ((List.replicate claims.size [7]).flatten ++ sq claims.size ++
        (_root_.Flock.Firewall.Session.ligeritoCoins sch).toList.filter (· != 0)) (fun _ => True)
      (_root_.Flock.Firewall.Finish.repOpening (stepAsks coins true)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra
        lanePad salt) := by
  unfold _root_.Flock.Firewall.Finish.repOpening
  apply ot_eq
  · apply ot_bind (ringSwitchP_ot coins _ claims hc); intro r _
    obtain ⟨ss, T, sws⟩ := r
    dsimp only
    apply ot_bind (ligeritoPZk_ot coins _ sch _ _ _ _ _ _ _ _ _ _ _ _ hk hn hq); intro r _
    obtain ⟨lg, zl, rec⟩ := r
    exact ot_pure trivial
  · simp only [List.append_nil, List.append_assoc]

theorem openStep_sp (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (coins : Got) (hdom : dom.size < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hm : t.st.m < 2 ^ 64)
    (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) :
    SP (fun _ _ => True) (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi coins)
      ((List.replicate (xs.size + 2) [7]).flatten ++ sq (xs.size + 2) ++
        (_root_.Flock.Firewall.Session.ligeritoCoins t.sch).toList.filter (· != 0)) := by
  obtain ⟨st, sch, tPad, g, extra, lanePad, cm0⟩ := t
  unfold _root_.Flock.Firewall.Drive.openStep
  dsimp only
  refine sp_run' ?_
  rw [StateT.run_bind, StateT.run_bind]
  refine sp_na_bind (Q := fun r => r.1.1.2.2.size = (_root_.Flock.Firewall.Finish.claimsOf xs).size + 2)
    (fun r hr => ?_) _ (piop_na coins st _ _ _ _ _ hdg hcn (by decide) hm _ (Sched.trAlike_init dom hdom) _)
  have hc : r.1.1.2.2.size = xs.size + 2 := by
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map] at hr
    exact hr
  refine sp_getThe_bind (R := fun _ _ => True) ?_
  split
  · exact trivial
  · rename_i hne
    have hP : Past coins r.2 := by
      simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hne
      unfold Past; omega
    have ho := repOpening_ot coins st sch (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) tPad rep g cm0 r.1.1.2.2
      (Array.replicate (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).len 0) extra lanePad
      (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty)) (by omega) hk hn hq r.1.2 r.2
      [] hP
    rw [List.append_nil, hc] at ho
    rw [StateT.run_bind, StateT.run_bind]
    refine sp_bind (fun _ _ _ => ?_) _ _ ho
    exact trivial

/-- The count a step program asks for next, none where it has ended. -/
def pend {α : Type} : Step α → List ℕ
  | .ask _ n _ => [n]
  | _ => []

/-- **Fed coins, a step program's rounds are its asks so far**: the rounds it sent, and the round it asks for next, are
a prefix of its asks after the rounds it started with. -/
theorem feed_prefix {α : Type} {Q : α → List ℕ → Prop} : ∀ (s : Step α) (cs : List (Array F128))
    (rs : Array (ByteArray × ℕ)) (ns : List ℕ), SP Q s ns →
    (s.feed cs rs).2.toList.map Prod.snd ++ pend (s.feed cs rs).1 <+: rs.toList.map Prod.snd ++ ns
  | .done _, _, rs, ns, _ => by
    show rs.toList.map Prod.snd ++ [] <+: _
    rw [List.append_nil]; exact List.prefix_append _ _
  | .fail _, _, rs, ns, _ => by
    show rs.toList.map Prod.snd ++ [] <+: _
    rw [List.append_nil]; exact List.prefix_append _ _
  | .ask c n k, [], rs, ns, ⟨ns', e, _⟩ => by
    show rs.toList.map Prod.snd ++ [n] <+: _
    rw [e]; exact (List.prefix_append_right_inj _).2 ⟨ns', rfl⟩
  | .ask c n k, x :: rest, rs, ns, ⟨ns', e, hk⟩ => by
    have := feed_prefix (k x) rest (rs.push (c, n)) ns' (hk x)
    simp only [Array.toList_push, List.map_append, List.map_cons, List.map_nil, List.append_assoc,
      List.singleton_append] at this
    rw [e]
    exact this

/-- The opening's asks at `claims` claims and the schedule `sch`: the ring switch's, then Ligerito's. -/
def openList (claims : ℕ) (sch : _root_.Flock.Schedule) : List ℕ :=
  (List.replicate claims [7]).flatten ++ sq claims ++
    (_root_.Flock.Firewall.Session.ligeritoCoins sch).toList.filter (· != 0)

/-- **A rep's scheduled rounds so far squeeze a prefix of its asks**: the algebraic phase's counts (`replayCoins`
without its zeros), then the opening's (`openList`). -/
theorem sched_prefix (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (cr : Array (Array F128)) (sched : Array (ByteArray × ℕ))
    (h : Loop.schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) :
    sched.toList.map Prod.snd <+:
      (_root_.Flock.Firewall.Session.replayCoins t.st.m t.st.fold.kLog).toList.filter (· != 0) ++
        openList (xs.size + 2) t.sch := by
  unfold Loop.schedSoFar at h
  rcases ha : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom
    with e | alg
  · rw [ha] at h; cases h
  · rw [ha] at h
    have hc := rounds_counts hm ha
    rw [piop_counts] at hc
    have hsp := feed_prefix _ (cr.extract alg.size cr.size).toList #[] _
      (openStep_sp t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
    simp only [List.map_nil, List.nil_append] at hsp
    simp only [bind, Except.bind] at h
    split at h
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      rw [hc]; exact List.prefix_append _ _
    · rcases hf : (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
        (cr.extract alg.size cr.size).toList #[] with ⟨s', rs⟩
      rw [hf] at h hsp
      cases s' with
      | ask c n k =>
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        simp only [pend, Array.toList_append, Array.toList_push, List.map_append, List.map_cons, List.map_nil] at hsp ⊢
        rw [hc]
        exact (List.prefix_append_right_inj _).2 hsp
      | done a =>
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        simp only [pend, List.append_nil, Array.toList_append, List.map_append] at hsp ⊢
        rw [hc]; exact (List.prefix_append_right_inj _).2 hsp
      | fail e => cases h

theorem tabOfLevel0_ok {st : _root_.Flock.Setup} {l : _root_.Flock.Firewall.Level0.Level0}
    {t : _root_.Flock.Firewall.Finish.Tab} (h : _root_.Flock.Firewall.Finish.tabOfLevel0 st l = .ok t) :
    t.st = st ∧ ∃ tp, _root_.Flock.Zk.schedule st.m = .ok (t.sch, tp) := by
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0 at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  split at h
  · cases h
  rcases hs : _root_.Flock.Zk.schedule st.m with e | ⟨sch, tPad⟩
  · rw [hs] at h; cases h
  · rw [hs] at h
    simp only at h
    split at h
    · cases h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, tPad, rfl⟩

/-- **The placeholder's table** is the statement's, at its `--zk` schedule, and its extra claims are at the link's
points. -/
theorem placeholder_tab {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {k : ℕ} {link : Array F128}
    {c' : _root_.Flock.Firewall.Frame.Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} (h : Loop.placeholder spec sts k link = .ok (c', t, dom, xs, hi)) :
    t.st = _root_.Flock.Setup.ofCircuit sts[k / 2]! ∧
      (∃ tp, _root_.Flock.Zk.schedule sts[k / 2]!.m = .ok (t.sch, tp)) ∧
      xs = _root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! link) := by
  have bok : ∀ {α β : Type} {x : V α} {f : α → V β} {b : β}, x >>= f = .ok b → ∃ a, x = .ok a ∧ f a = .ok b := by
    intro α β x f b hb
    cases x with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  unfold Loop.placeholder at h
  obtain ⟨⟨lay, mask⟩, -, h⟩ := bok h
  obtain ⟨⟨sch, tPad⟩, -, h⟩ := bok h
  obtain ⟨l0, -, h⟩ := bok h
  obtain ⟨t0, ht0, h⟩ := bok h
  cases h
  exact ⟨(tabOfLevel0_ok ht0).1, (tabOfLevel0_ok ht0).2, rfl⟩

theorem extras_size (st : _root_.Flock.Stmt) (pts : Array (Array F128)) :
    (_root_.Flock.Firewall.Drive.extras st pts).size = st.regions.size * pts.size := by
  unfold _root_.Flock.Firewall.Drive.extras
  rw [← Array.length_toList, Array.toList_flatMap]
  simp only [Array.toList_map, List.length_flatMap, List.length_map, Array.length_toList, List.map_const',
    List.sum_replicate, smul_eq_mul]

theorem linkPoints_size (st : _root_.Flock.Stmt) (c : Array F128) :
    (_root_.Flock.Firewall.Drive.linkPoints st c).size = c.size / st.mPts := by
  simp [_root_.Flock.Firewall.Drive.linkPoints]

/-- A `--zk` schedule's facts the opening's asks read: a lane round before the pair, and its counts within a squeeze's. -/
def schedOk (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule m with
  | .ok (sch, _) => decide (1 ≤ sch.k0) && decide (sch.n < 2 ^ 62) &&
      sch.levels.all (fun lv => decide (lv.q < 2 ^ 63)) && decide ((default : _root_.Flock.Level).q < 2 ^ 63)
  | .error _ => true

/-- Every `--zk` schedule, `m < 36`, by the kernel's evaluation. -/
theorem schedOk_lt : ∀ m, m < 36 → schedOk m = true := by decide +kernel

theorem sched_facts {m tp : ℕ} {sch : _root_.Flock.Schedule} (h : _root_.Flock.Zk.schedule m = .ok (sch, tp)) :
    m < 36 ∧ 1 ≤ sch.k0 ∧ sch.n < 2 ^ 62 ∧ ∀ i, (sch.levels.getD i default).q < 2 ^ 63 := by
  have hm : m < 36 := by
    by_contra hc
    obtain ⟨e, he⟩ := Loop.schedule_ge (m := m) (by omega)
    rw [he] at h; cases h
  have hok := schedOk_lt m hm
  unfold schedOk at hok
  rw [h] at hok
  simp only [Bool.and_eq_true, decide_eq_true_eq, Array.all_eq_true] at hok
  obtain ⟨⟨⟨hk, hn⟩, hl⟩, hd⟩ := hok
  refine ⟨hm, hk, hn, fun i => ?_⟩
  rw [Array.getD_eq_getD_getElem?]
  rcases hi : sch.levels[i]? with _ | lv
  · exact hd
  · obtain ⟨hi', rfl⟩ := Array.getElem?_eq_some_iff.1 hi
    exact hl i hi'

theorem seven_bne : ((7 : ℕ) != 0) = true := rfl

theorem flatten_seven (n : ℕ) : (List.replicate n [7]).flatten = List.replicate n 7 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [List.replicate_succ, List.flatten_cons, ih]; rfl

/-- **A rep's plan**, as a list: the algebraic phase's counts, the opening's, then the inner proof's. -/
theorem repPlan_list {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (h : _root_.Flock.Zk.schedule st.m = .ok (sch, tp)) :
    (_root_.Flock.Firewall.Session.repPlan spec st).toList =
      (_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0) ++
        openList (2 + spec.points * st.regions.size) sch ++
        [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0) := by
  simp only [_root_.Flock.Firewall.Session.repPlan, Loop.repCoins_ok h, Except.toOption, Option.getD_some,
    Array.toList_filter, Array.toList_append, List.filter_append, Array.toList_replicate, List.filter_replicate,
    openList, flatten_seven, List.append_assoc, seven_bne, ↓reduceIte, filter_cons_ne, List.filter_nil,
    List.append_nil]

theorem getD_of_prefix {P : Array ℕ} {sched : Array (ByteArray × ℕ)} {i : ℕ}
    (h : sched.toList.map Prod.snd <+: P.toList) (hi : i < sched.size) :
    P.getD i 0 = (sched.getD i (.empty, 0)).2 := by
  have hl : i < (sched.toList.map Prod.snd).length := by simpa using hi
  have hPl : i < P.size := by
    have := h.length_le
    simp only [List.length_map, Array.length_toList] at this
    omega
  have e := h.getElem hl
  simp only [List.getElem_map, Array.getElem_toList] at e
  simp only [Array.getD, hi, hPl, dite_true]
  exact e.symm

/-- **The plan's count is the schedule's**: at a placeholder at the link points of `spec.points` points, each round of
the rep's schedule so far squeezes the plan's count for it. -/
theorem plan_sched (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) (k i : ℕ) (link : Array F128)
    (c' : _root_.Flock.Firewall.Frame.Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr : Array (Array F128)) (sched : Array (ByteArray × ℕ))
    (hph : Loop.placeholder spec sts k link = .ok (c', t, dom, xs, hi))
    (hs : Loop.schedSoFar t (k % 2) dom xs hi cr = .ok sched) (hlt : i < sched.size)
    (hpts : link.size / sts[k / 2]!.mPts = spec.points) (hdom : dom.size < 2 ^ 64)
    (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) :
    (_root_.Flock.Firewall.Session.repPlan spec sts[k / 2]!).getD i 0 = (sched.getD i (.empty, 0)).2 := by
  obtain ⟨hts, ⟨tp, hsch⟩, hxs⟩ := placeholder_tab hph
  have hxn : xs.size = spec.points * sts[k / 2]!.regions.size := by
    rw [hxs, extras_size, linkPoints_size, hpts, Nat.mul_comm]
  obtain ⟨hm, hk, hn, hq⟩ := sched_facts hsch
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts]
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hpre := sched_prefix t (k % 2) dom xs hi cr sched hs hdom hdg hcn (by rw [hts]; exact Nat.lt_trans hm (by decide))
    hx hk hn hq
  rw [hts, hxn] at hpre
  refine getD_of_prefix (hpre.trans ?_) hlt
  rw [repPlan_list hsch, Nat.add_comm]
  exact List.prefix_append _ _

end Claims.ZKServer.Count
