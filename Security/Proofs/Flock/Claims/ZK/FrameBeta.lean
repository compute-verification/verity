import Proofs.Flock.Claims.ZK.FrameAdmits
import Proofs.Flock.Claims.ZK.FrameTaken

/-!
# The frames refuse `β = 0` (`FrameUnrefused`'s `β` fault)

At a `Live` statement the loop serves, on a live history of `refK x` openings whose last coins the frame takes, no
table's rep has `β = 0` among the opened coins (`Masked.Thrown`'s third disjunct): the rep's frame takes `β`'s coins
at its round, in the inner proof's round 2, where `Masker.answer` stops at `β = 0`.

* `Count.openStep_end`, `Count.feed_full`, **`Count.schedSoFar_size`**: the opening asks for exactly `openList`'s counts,
  so fed at least that many coins its schedule so far is the algebraic phase's rounds and all of the opening's.
* `Loop.locate_take_sum`, `Loop.repStartZk_eq`: rep round `Σ_{q' < q} rs[q'] + i` is rep stream `q`'s round `i`, and
  rep `q` starts at opening `1 + Σ_{q' < q} rs[q']`.
* `Loop.answer_beta`: where the frame's answer moves the inner proof to its round 3, the coins were one nonzero `β`.
* **`LeanRelease.beta_unrefused`**: the fault. The opening at `Masked.betaSlot x j r` is rep `2j + r`'s round
  `piopRoundsZk + k₀ + 7 + 11(R − 1) + 2`, whose state is the inner proof's round 2 (`repState_own_inner`) once the
  schedule so far is complete (`schedSoFar_size`, at the plan's counts `repPlan_list`, `repPlan_size`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_BETA INNER_ROUNDS)

namespace Count

open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)

theorem sp_run'_end {α : Type} {p : TMW OpenM α} {t : Transcript} {s : Got} {ns : List ℕ}
    (h : SP (fun _ ns => ns = []) ((p.run t).run s) ns) : SP (fun _ ns => ns = []) ((p.run' t).run' s) ns := by
  refine sp_bind (Q := fun _ ns => ns = []) (fun _ _ h => ?_) _ _ (sp_bind (fun _ _ h => ?_) _ _ h) <;> exact h

/-- **The opening asks for exactly `openList`'s counts**: past the PIOP's coin check, `openStep` asks for the ring
switch's counts, then Ligerito's, and ends with none left. -/
theorem openStep_end (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (coins : Got) (hdom : dom.size < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hm : t.st.m < 2 ^ 64)
    (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) :
    SP (fun _ ns => ns = []) (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi coins)
      (openList (xs.size + 2) t.sch) := by
  obtain ⟨st, sch, tPad, g, extra, lanePad, cm0⟩ := t
  unfold openList _root_.Flock.Firewall.Drive.openStep
  dsimp only
  refine sp_run'_end ?_
  rw [StateT.run_bind, StateT.run_bind]
  refine sp_na_bind (Q := fun r => r.1.1.2.2.size = (_root_.Flock.Firewall.Finish.claimsOf xs).size + 2)
    (fun r hr => ?_) _ (piop_na coins st _ _ _ _ _ hdg hcn (by decide) hm _ (Sched.trAlike_init dom hdom) _)
  have hc : r.1.1.2.2.size = xs.size + 2 := by
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map] at hr
    exact hr
  refine sp_getThe_bind (R := fun _ ns => ns = []) ?_
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
    refine sp_bind (fun _ _ h => ?_) _ _ ho
    exact h.2.2

/-- **Fed at least the coins it asks for, a step program that ends with no asks left ends**, having sent exactly
its asks' rounds. -/
theorem feed_full {α : Type} : ∀ (s : Step α) (cs : List (Array F128)) (rs : Array (ByteArray × ℕ)) (ns : List ℕ),
    SP (fun _ ns => ns = []) s ns → ns.length ≤ cs.length →
    (∀ c n k rs', s.feed cs rs ≠ (.ask c n k, rs')) ∧
      ∀ a rs', s.feed cs rs = (.done a, rs') → rs'.size = rs.size + ns.length
  | .done a, cs, rs, ns, h, _ => by
    have e : (Step.done a).feed cs rs = (.done a, rs) := by cases cs <;> rfl
    rw [e]
    have h0 : ns = [] := h
    subst h0
    refine ⟨fun _ _ _ _ h' => ?_, fun _ _ h' => ?_⟩
    · simp at h'
    · obtain ⟨-, rfl⟩ := Prod.mk.inj h'
      simp
  | .fail e, cs, rs, ns, _, _ => by
    have e' : (Step.fail e : Step α).feed cs rs = (.fail e, rs) := by cases cs <;> rfl
    rw [e']
    refine ⟨fun _ _ _ _ h' => ?_, fun _ _ h' => ?_⟩ <;> simp at h'
  | .ask _ _ _, [], _, _, ⟨ns', e, _⟩, hl => by
    subst e
    simp at hl
  | .ask c n k, x :: rest, rs, ns, ⟨ns', e, hk⟩, hl => by
    subst e
    have ih := feed_full (k x) rest (rs.push (c, n)) ns' (hk x) (by simp only [List.length_cons] at hl; omega)
    have e' : (Step.ask c n k).feed (x :: rest) rs = (k x).feed rest (rs.push (c, n)) := rfl
    rw [e']
    refine ⟨ih.1, fun a rs' h' => ?_⟩
    rw [ih.2 a rs' h', Array.size_push, List.length_cons]
    omega

/-- **The schedule so far, complete**: fed the coins of the algebraic phase's rounds and of all the opening's, a rep's
schedule so far is those rounds. -/
theorem schedSoFar_size (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (cr : Array (Array F128)) {alg sched : Array (ByteArray × ℕ)}
    (ha : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (h : Loop.schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hle : alg.size + (openList (xs.size + 2) t.sch).length ≤ cr.size) :
    sched.size = alg.size + (openList (xs.size + 2) t.sch).length := by
  unfold Loop.schedSoFar at h
  rw [ha] at h
  simp only [bind, Except.bind] at h
  split at h
  · omega
  · have hf := feed_full _ (cr.extract alg.size cr.size).toList #[] _
      (openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
      (by simp only [Array.length_toList, Array.size_extract]; omega)
    rcases hfe : (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
      (cr.extract alg.size cr.size).toList #[] with ⟨s', rs⟩
    rw [hfe] at h hf
    cases s' with
    | ask c n k => exact absurd rfl (hf.1 c n k rs)
    | done a =>
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      rw [Array.size_append, hf.2 a rs rfl, Array.size_empty, Nat.zero_add]
    | fail e => cases h

end Count

namespace Loop

/-- **Rep round `Σ_{q' < q} rs[q'] + i` is rep stream `q`'s round `i`.** -/
theorem locate_take_sum : ∀ (rs : List ℕ) (k q i : ℕ), q < rs.length → i < rs.getD q 0 →
    ZKMal.CoinLayer.locate rs k ((rs.take q).sum + i) = (k + q, i)
  | [], _, _, _, h, _ => by simp at h
  | c :: cs, k, 0, i, _, hi => by
    rw [List.take_zero, List.sum_nil, Nat.zero_add, Nat.add_zero]
    exact locate_cons_lt (by simpa using hi)
  | c :: cs, k, q + 1, i, hq, hi => by
    rw [List.take_succ_cons, List.sum_cons, locate_cons_ge (by omega),
      show c + (cs.take q).sum + i - c = (cs.take q).sum + i by omega,
      locate_take_sum cs (k + 1) q i (by simpa using hq) (by simpa using hi)]
    simp only [Prod.mk.injEq, and_true]
    omega

/-- **Rep `q` starts at opening `1 + Σ_{q' < q} rs[q']`.** -/
theorem repStartZk_eq (spec : SessionSpec) (sts : Array Stmt) {q : ℕ} (hq : q ≤ spec.repStreams.size) :
    FlockComplete.repStartZk spec sts q = 1 + ((roundsOf spec sts).take q).sum := by
  unfold FlockComplete.repStartZk roundsOf
  rw [← List.map_take, List.take_range, Nat.min_eq_left hq]

theorem innerCoins_beta {a : Array F128} (h : _root_.Flock.Firewall.Frame.innerCoins INNER_BETA a = .ok ()) :
    a.size = 1 ∧ a[0]!.isZero = false := by
  unfold _root_.Flock.Firewall.Frame.innerCoins at h
  generalize hb : a[0]! = b at h
  by_cases h1 : a.size = 1
  · by_cases h2 : b.isZero = true
    · simp [h1, h2, throw, throwThe, MonadExceptOf.throw] at h
    · exact ⟨h1, by simpa using h2⟩
  · simp [h1, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h

/-- Where the frame's answer at the inner proof's round `k` takes the coins, `innerCoins` passed them. -/
theorem answer_innerCoins {c : Cfg} {s m : Masker} {cs : Array F128} {k : ℕ} (hs : s.inner = some k)
    (h : s.answer c cs = .ok m) : _root_.Flock.Firewall.Frame.innerCoins k cs = .ok () := by
  rcases hic : _root_.Flock.Firewall.Frame.innerCoins k cs with e | u
  · exfalso
    unfold Masker.answer at h
    simp only [bind, Except.bind, hs, hic] at h
    repeat' split at h
    all_goals cases h
  · rfl

/-- **Where the frame's answer moves the inner proof to its round 3, the coins were one nonzero `β`.** -/
theorem answer_beta {c : Cfg} {s m : Masker} {cs : Array F128} (h : s.answer c cs = .ok m)
    (hm : m.inner = some (INNER_BETA + 1)) : cs.size = 1 ∧ cs[0]!.isZero = false := by
  rcases hs : s.inner with _ | k
  · rw [Sched.answer_inner hs h] at hm
    cases hm
  · have hk := answer_inner_some hs h
    rw [hm, Option.some.injEq] at hk
    have hk2 : k = INNER_BETA := by omega
    subst hk2
    exact innerCoins_beta (answer_innerCoins hs h)

/-- Where the loop's state at a rep's round takes coins, the rep's algebraic phase and schedule so far are set up. -/
theorem repState_ok {c : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {i : ℕ} {c' : Cfg}
    {cs : Array F128} {m : Masker} (ha : (repState c t rep dom xs hi cr0 cr ns i).answer c' cs = .ok m) :
    ∃ alg sched, _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg ∧ schedSoFar t rep dom xs hi cr.pop = .ok sched := by
  rcases hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom
    with e | alg
  · exfalso
    unfold repState at ha
    simp only [hal, bind, Except.bind] at ha
    exact absurd ha (answer_owed_none cs rfl)
  · rcases hs : schedSoFar t rep dom xs hi cr.pop with e | sched
    · exfalso
      unfold repState at ha
      simp only [hal, hs, bind, Except.bind] at ha
      exact absurd ha (answer_owed_none cs rfl)
    · exact ⟨alg, sched, rfl, rfl⟩

/-- The placeholder's level-0 cap is within a `u64`'s count. -/
theorem placeholder_cap {spec : SessionSpec} {sts : Array Stmt} {k : ℕ} {link : Array F128} {c' : Cfg}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts k link = .ok (c', t, dom, xs, hi)) :
    (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64 := by
  have bok : ∀ {α β : Type} {x : V α} {f : α → V β} {b : β}, x >>= f = .ok b → ∃ a, x = .ok a ∧ f a = .ok b := by
    intro α β x f b hb
    cases x with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hR : Count.RealTabs sts (fun k' => if k' = k then .ok (t, hi) else .error "") := by
    intro k' tr hr he
    by_cases hk : k' = k
    · subst hk
      simp only [↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at he
      obtain ⟨rfl, -⟩ := he
      have h := hph
      unfold placeholder at h
      obtain ⟨_, -, h⟩ := bok h
      obtain ⟨_, -, h⟩ := bok h
      obtain ⟨l0, hl0, h⟩ := bok h
      obtain ⟨t0, ht0, h⟩ := bok h
      cases h
      exact ⟨_, _, l0, hl0, ht0⟩
    · simp only [hk, ↓reduceIte] at he
      cases he
  have hT := Count.real_tabAlike hR hph (tr := t) (hr := hi) (by simp)
  rw [← hT.2.2.2.2.1]
  exact hT.2.2.2.2.2

end Loop

namespace LeanRelease

open Loop

/-- **The frames refuse `β = 0`**: at a `Live` statement the loop serves, on a live history of `refK x` openings whose
last coins the frame takes, table `j`'s rep `r`'s `β`, where it is among the opened coins, is nonzero. The opening at
`betaSlot` is the rep's inner round 2, its frame there at the inner proof's round 2 (the schedule so far complete,
`schedSoFar_size`), and the frame took its coins (`coins_taken_of_live`), which it does only at a nonzero `β`
(`answer_beta`). -/
theorem beta_unrefused (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hb : Masked.betaSlot x j r < refK x) :
    (Masked.betaAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r).isZero = false := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [show (2 * j + r) / 2 = j by omega]
    exact getElem!_pos _ j hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hs : Masked.betaSlot x j r = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + N + 2 := by
    unfold Masked.betaSlot
    rw [Masked.repSlot_eq hsx hz, hN]
    omega
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrq : (roundsOf spec (Masked.stsOf x)).getD (2 * j + r) 0 = N + 4 := by
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq2, Option.map_some,
      Option.getD_some]
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hs1 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) := by
    unfold FlockComplete.repStartZk; omega
  obtain ⟨s, hsd⟩ : ∃ s, Masked.betaSlot x j r = s := ⟨_, rfl⟩
  rw [hsd] at hs hb
  have hlast : (ofLoop T L₀ real).last x = refK x + 1 := loop_last loopNr T (withFrames T L₀ real) x
  obtain ⟨-, m, ha⟩ := coins_taken_of_live (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive
    (by rw [hlast]; simp [hl]) hr (i := s + 2) (by omega) (by simp only [List.length_cons]; omega)
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
  have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
  have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
    rw [List.take_succ_cons, htk]
  have hts : (ops.take s).length = s := by rw [List.length_take]; omega
  rw [hh] at ha
  change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => ZKMal.CoinLayer.rd p.2)).answer _
    (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
  have hca : coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)])) =
      (ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold coinsAt
    simp only [List.tail_cons, List.getLast?_append, List.getLast?_singleton, List.length_append,
      hts, List.length_singleton, Nat.add_sub_cancel]
    exact feedOf_rd _ _
  rw [hca, List.map_append, List.map_cons, List.map_nil] at ha
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1) = (2 * j + r, N + 2) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) (N + 2) (by rw [hrsl]; exact hq2)
      (by rw [hrq]; omega)
    rw [Nat.zero_add] at e
    rw [← e, hs, repStartZk_eq spec _ hq2.le]
    congr 1
    omega
  rw [progOf_snoc x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (ZKMal.CoinLayer.rd da) hsx (q := s - 1)
    (by rw [List.length_map, hts]; omega), hloc] at ha
  dsimp only at ha
  rcases hph : placeholder spec (Masked.stsOf x) (2 * j + r)
      (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0) with e | ⟨c', t, dom, xs, hi⟩
  · rw [hph] at ha
    exact absurd ha (answer_owed_none _ rfl)
  rw [hph] at ha
  dsimp only at ha
  obtain ⟨alg, sched, hal, hsc⟩ := repState_ok ha
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  obtain ⟨hm36, hk0, hn, hq⟩ := Count.sched_facts (htsch ▸ hz : _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad))
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨-, -, -, -, hdm, -⟩ := placeholder_ok hph
  have hls : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    obtain ⟨p0, rest, hpr⟩ : ∃ p0 rest, ops.take s = p0 :: rest := by
      cases h' : ops.take s with
      | nil => rw [h', List.length_nil] at hts; omega
      | cons p0 rest => exact ⟨p0, rest, rfl⟩
    rw [hpr]
    simp [wordsAt, FlockDeploy.Coin.vals]
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hpts : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size /
      (Masked.stsOf x)[(2 * j + r) / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  have hxn : xs.size = spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by
    rw [hxs, Count.extras_size, Count.linkPoints_size, hpts, Nat.mul_comm]
  have hxb : xs.size + 2 < 2 ^ 64 := by
    rw [hxs, Loop.extras_size]
    have := Nat.mul_le_mul_left (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size
      ((linkPoints_size_le (Masked.stsOf x)[(2 * j + r) / 2]! _).trans (hls ▸ Nat.min_le_right _ _))
    omega
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : t.st.m < 2 ^ 64 := by rw [hts']; exact Nat.lt_trans hm36 (by decide)
  have hA : alg.size = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stsOf x)[(2 * j + r) / 2]!.m
      (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog).toList.filter (· != 0)).length := by
    have e := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts', List.length_map, Array.length_toList] at e
    exact e
  have hplan := congrArg List.length (Count.repPlan_list (spec := spec) (htsch ▸ hz :
    _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad)))
  rw [Array.length_toList, repPlan_size] at hplan
  simp only [repRounds_true, hz, List.length_append] at hplan
  have h4 : ([1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0)).length = 4 := by
    decide
  have hFO : alg.size + (Count.openList (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)
      t.sch).length = N := by
    rw [h4] at hplan
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [hA, hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hx2 : xs.size + 2 = 2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by omega
  have hS := Count.schedSoFar_size t _ dom xs hi _ hal hsc (hdm ▸ hdom') hdg (placeholder_cap hph) hmb hxb hk0 hn hq
    (by
      rw [hx2, hFO]
      simp only [Array.size_pop, Array.size_push, Array.size_map, Array.size_range]
      omega)
  rw [hx2, hFO] at hS
  rw [show N + 2 + 1 = sched.size + 2 + 1 by omega] at ha
  have hI := repState_own_inner c' _ t _ dom xs hi _ _ _ hal hsc (i := 2) (by decide) ha
  obtain ⟨hB1, hB2⟩ := answer_beta ha (by rw [hI]; rfl)
  have hn1 : min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX = 1 := by
    simpa [FlockDeploy.Coin.vals] using hB1
  have h0 : ((ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX))[0]! =
      (ZKMal.CoinLayer.rd da).at 0 := by
    rw [hn1]
    simp [FlockDeploy.Coin.vals]
  unfold Masked.betaAt Masked.slotAt
  rw [hsd, dif_pos hb]
  show (((ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2).getD s default).at 0).isZero = false
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, hd, Option.map_some, Option.getD_some]
  show ((ZKMal.CoinLayer.rd da).at 0).isZero = false
  rw [← h0]
  exact hB2

end LeanRelease

end Claims.ZKServer
