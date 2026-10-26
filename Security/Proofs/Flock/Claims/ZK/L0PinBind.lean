import Proofs.Flock.Claims.ZK.L0PinSwitch
import Proofs.Flock.Claims.ZK.L0PinRec

/-!
# The frame at a rep's binding

Level 0's binding (`LeanRelease.answered_openBind`) needs the frame's state where it takes the coins of the ring
switch's last round, `γ`'s: outside every masked phase, past the algebraic phase's opening, before level 0 and the inner
proof, at the algebraic phase's last position. This file gives it at the loop's state (`repState_opened`), from the
rep's schedule so far:

* **The opening's rounds so far** (`schedSoFar_sentQ`, `feed_part`): past the algebraic phase's rounds, a rep's schedule
  is what its opening sent, fed, with the round it asks for next; fed fewer coins than it asks for, one more round than
  coins.
* **The ring switch cut at `γ`** (`ringSwitchCut_gamma`, `openStep_switch`): the cut run stops at `γ`'s squeeze having
  sent the switch's bodies (`Switch.rsBody`), so the opening's first `claims + 1` rounds are those, after the PIOP's
  empty pending bytes.
* **Their counts** (`sched_count`, `openList_lt`, `openList_at`): `7` a claim, then `claims` for `γ`.
* **The flags** (`repState_opened`): the switch's first round is the opening's (`Flags.PcsRound`), the rest quiet, so
  `Flags.sched_flags` carries the frame there, and the coins keep it.
* **Level 0's rounds** (`sched_binding`, `sched_l0`): fed level 0's first `j` coins as well, the schedule's next
  rounds are level 0's first `j + 1` bodies at zero pads (`openStep_l0`), the binding first, with the opening's counts.
-/

namespace Claims.ZKServer.Bind

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)
open _root_.Flock.Firewall.Finish (repOpening repPiop commitFw claimsOf ringSwitchP_eq switchVec)
open FlockSoundness.Refine (Frame encs)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (bind_go liftTM_ok)
open FlockComplete.LigLock.Fed (feedR sentQ pendC feed_eq CoinAt repOpening_feed cutAsks cutAsks_toC openStep_sent)
open Switch (rsBody)
open Loop

/-! ## The opening's rounds so far -/

/-- **Fed fewer coins than it asks for**, a step program that has not failed sent a round for each and asks for one
more. -/
theorem feed_part {α : Type} : ∀ (s : Step α) (cs : List (Array F128)) (ns : List ℕ),
    Count.SP (fun _ ns => ns = []) s ns → cs.length < ns.length → (∀ e, (feedR s cs).1 ≠ .fail e) →
    (sentQ (feedR s cs)).length = cs.length + 1
  | .done _, _, ns, h, hl, _ => by
    have h0 : ns = [] := h
    subst h0
    simp at hl
  | .fail e, cs, _, _, _, hf => (hf e (by cases cs <;> rfl)).elim
  | .ask _ _ _, [], _, _, _, _ => rfl
  | .ask c n k, z :: rest, _, ⟨ns', e, hk⟩, hl, hf => by
    subst e
    have ih := feed_part (k z) rest ns' (hk z) (by simp only [List.length_cons] at hl; omega) hf
    have e2 : sentQ (feedR (.ask c n k) (z :: rest)) = c :: sentQ (feedR (k z) rest) := rfl
    rw [e2, List.length_cons, ih, List.length_cons]

/-- **A rep's schedule past its algebraic phase** is what its opening, fed the rest of the coins, sent and asks for
next, and the feeding has not failed. -/
theorem schedSoFar_sentQ {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hA : alg.size ≤ cr.size) :
    (sched.toList.map Prod.fst).drop alg.size =
        sentQ (feedR (openStep t rep dom xs hi (cr.extract 0 alg.size)) (cr.extract alg.size cr.size).toList) ∧
      ∀ e, (feedR (openStep t rep dom xs hi (cr.extract 0 alg.size)) (cr.extract alg.size cr.size).toList).1 ≠
        .fail e := by
  unfold schedSoFar at hs
  rw [hal] at hs
  simp only [bind, Except.bind] at hs
  split at hs
  · omega
  · rw [feed_eq] at hs
    rcases hF : feedR (openStep t rep dom xs hi (cr.extract 0 alg.size)) (cr.extract alg.size cr.size).toList with
      ⟨s', sent, rest⟩
    rw [hF] at hs
    cases s' with
    | ask c n k =>
      simp only [pure, Except.pure, Except.ok.injEq] at hs
      subst hs
      refine ⟨?_, fun e he => by (try rw [hF] at he); cases he⟩
      try rw [hF]
      rw [Array.toList_append, List.map_append, List.drop_left' (by simp)]
      simp [sentQ, pendC]
    | done a =>
      simp only [pure, Except.pure, Except.ok.injEq] at hs
      subst hs
      refine ⟨?_, fun e he => by (try rw [hF] at he); cases he⟩
      try rw [hF]
      rw [Array.toList_append, List.map_append, List.drop_left' (by simp)]
      simp [sentQ, pendC]
    | fail e => cases hs

/-- A schedule's entry past `A`, from the rounds past `A`. -/
theorem sched_entry {sched : Array (ByteArray × ℕ)} {A : ℕ} {q : List ByteArray}
    (h : (sched.toList.map Prod.fst).drop A = q) {i : ℕ} (hi : i < q.length) :
    (sched.getD (A + i) (.empty, 0)).1 = q[i] := by
  have hsz : A + i < sched.size := by
    have := congrArg List.length h
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  subst h
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hsz, Option.getD_some]
  simp only [List.getElem_drop, List.getElem_map, Array.getElem_toList]

/-- A schedule's entry past `A`, from the rounds past `A` and their first `N + 1` bodies. -/
theorem sched_bytes {sched : Array (ByteArray × ℕ)} {A N : ℕ} {q : List ByteArray} {B : ℕ → Claims.Msg}
    (h : (sched.toList.map Prod.fst).drop A = q)
    (hq : (q.take (N + 1)).map Claims.toMsg = (List.range (N + 1)).map B) (hlen : N < q.length) {i : ℕ}
    (hi : i ≤ N) : (sched.getD (A + i) (.empty, 0)).1.data.toList = B i := by
  have e1 : Claims.toMsg q[i] = B i := by
    have := List.getElem_of_eq hq (i := i) (by simp only [List.length_map, List.length_take]; omega)
    simp only [List.getElem_map, List.getElem_take, List.getElem_range] at this
    exact this
  rw [sched_entry h (by omega)]
  exact e1

/-! ## The opening's counts -/

theorem openList_eq {N : ℕ} (sch : _root_.Flock.Schedule) (hN : N ≠ 0) :
    Count.openList N sch = List.replicate N 7 ++ N ::
      (_root_.Flock.Firewall.Session.ligeritoCoins sch).toList.filter (· != 0) := by
  unfold Count.openList Count.sq
  rw [Count.flatten_seven, if_neg hN, List.append_assoc, List.singleton_append]

theorem openList_len {N : ℕ} (sch : _root_.Flock.Schedule) (hN : N ≠ 0) : N < (Count.openList N sch).length := by
  rw [openList_eq sch hN]
  simp

theorem openList_lt {N i : ℕ} (sch : _root_.Flock.Schedule) (hN : N ≠ 0) (hi : i < N) :
    (Count.openList N sch).getD i 0 = 7 := by
  rw [openList_eq sch hN, List.getD_eq_getElem?_getD, List.getElem?_append_left (by simpa using hi)]
  simp [hi]

theorem openList_at {N : ℕ} (sch : _root_.Flock.Schedule) (hN : N ≠ 0) : (Count.openList N sch).getD N 0 = N := by
  rw [openList_eq sch hN, List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp), List.length_replicate,
    Nat.sub_self]
  rfl

theorem ligeritoCoins_cons (sch : _root_.Flock.Schedule) :
    ∃ rest, (_root_.Flock.Firewall.Session.ligeritoCoins sch).toList = sch.n :: rest := by
  unfold _root_.Flock.Firewall.Session.ligeritoCoins
  simp only [Array.toList_append, List.append_assoc, List.toList_toArray, List.cons_append, List.nil_append]
  exact ⟨_, rfl⟩

/-- Past the ring switch's, the opening's first count is level 0's binding's, `n`. -/
theorem openList_bind {N : ℕ} (sch : _root_.Flock.Schedule) (hN : N ≠ 0) (hn0 : sch.n ≠ 0) :
    N + 1 < (Count.openList N sch).length ∧ (Count.openList N sch).getD (N + 1) 0 = sch.n := by
  obtain ⟨rest, hr⟩ := ligeritoCoins_cons sch
  rw [openList_eq sch hN, hr, List.filter_cons_of_pos (by simpa using hn0)]
  constructor
  · simp
  · rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp), List.length_replicate,
      show N + 1 - N = 1 by omega]
    rfl

/-- **A rep's schedule past its algebraic phase squeezes the opening's counts.** -/
theorem sched_count {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) {i : ℕ} (hi' : alg.size + i < sched.size) :
    (sched.getD (alg.size + i) (.empty, 0)).2 = (Count.openList (xs.size + 2) t.sch).getD i 0 := by
  have hpre := Count.sched_prefix t rep dom xs hi cr sched hs hdom hdg hcn hm hx hk hn hq
  have hc := Count.rounds_counts hm hal
  rw [Count.piop_counts] at hc
  rw [← hc] at hpre
  have e := Count.getD_of_prefix (P := (alg.toList.map Prod.snd ++ Count.openList (xs.size + 2) t.sch).toArray)
    (by rw [List.toList_toArray]; exact hpre) hi'
  rw [← e, Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_append_right (by simp),
    List.length_map, Array.length_toList, Nat.add_sub_cancel_left, List.getD_eq_getElem?_getD]

/-! ## The ring switch cut at `γ` -/

/-- **The cut ring switch, cut at `γ`'s squeeze**: from `l`, cut at its last ask, it stops having sent `rsBody` after
`t`'s pending bytes, `γ`'s squeeze its last body. -/
theorem ringSwitchCut_gamma (coin : ℕ → ℕ → F128) (n : ℕ) (sv : ℕ → Array F128)
    (claims : Array _root_.Flock.Claim) (hN : claims.size ≠ 0) (t : Transcript) (l : Array Claims.Msg)
    (hl : n = l.size + claims.size) :
    ∃ L', ((ringSwitchCut (asksCut coin n) sv claims).run t).go l = (.error "past the openings", L') ∧
      L'.toList = l.toList ++ (List.range (claims.size + 1)).map (rsBody sv claims.size t.pending.data.toList) := by
  unfold ringSwitchCut
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  try dsimp only
  refine Switch.forIn_bind_go _ (Switch.rsLoopT sv coin l.size (Switch.pcsT t))
    (Switch.rsLoopB sv coin l.size (Switch.pcsT t)) l.size claims.size rfl ?hbody _ _ l rfl
    (fun r => ∃ L', r = (.error "past the openings", L') ∧
      L'.toList = l.toList ++ (List.range (claims.size + 1)).map (rsBody sv claims.size t.pending.data.toList)) ?_
  case hbody =>
    intro i hi s l' hl'
    try dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    rw [bind_go (sampleVecP_cut_ok coin (by decide) _ (by omega))]
    rw [hl']
    exact ⟨_, rfl⟩
  intro s
  try dsimp only
  rw [FlockComplete.bind_go_err (x := FlockComplete.sampleVecP (asksCut coin n) claims.size)
    (FlockComplete.squeezeP_cut_err coin 2 hN _ (by simp; omega))]
  refine ⟨_, rfl, ?_⟩
  rw [Array.toList_push, Array.toList_append, List.toList_toArray, Switch.rsLoopB_map, List.range_succ,
    List.map_append, List.map_cons, List.map_nil, List.append_assoc]
  rw [Switch.rsLast sv coin l.size (Switch.pcsT t) hN t.pending.data.toList]

/-- **The opening's first rounds are the ring switch's**: fed past `γ`'s squeeze, `openStep`'s first `claims + 1`
rounds are the switch's bodies from empty pending bytes, each claim's switched vector of `128` words. -/
theorem openStep_switch (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (coins : Array (Array F128)) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (cs : List (Array F128))
    (hlen : xs.size + 2 < (sentQ (feedR (openStep t rep dom xs hi coins) cs)).length) :
    ∃ sv : ℕ → Array F128, (∀ k, (sv k).size = 128) ∧
      ((sentQ (feedR (openStep t rep dom xs hi coins) cs)).take (xs.size + 2 + 1)).map Claims.toMsg =
        (List.range (xs.size + 2 + 1)).map (rsBody sv (xs.size + 2) []) := by
  rcases openStep_sent t rep dom xs hi coins hdom hdg hcn hm cs with h | ⟨r, hp, hr, hQ⟩
  · rw [h] at hlen
    simp at hlen
  have hna : r.1.1.2.2.size = xs.size + 2 := by
    have := Count.piop_na coins t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) (fun _ => 0) hdg hcn (by decide) hm _
      (Sched.trAlike_init dom hdom) #[]
    rw [hp] at this
    have h' : r.1.1.2.2.size = (claimsOf xs).size + 2 := this
    simpa only [claimsOf, Array.size_map] using h'
  have hpe : r.1.2.pending = ByteArray.empty := Switch.endE_repPiop _ _ _ _ _ _ _ _ _ _ hp
  refine ⟨fun k => switchVec t.g r.1.1.2.2[k]!, fun k => by simp [switchVec], ?_⟩
  have hc : CoinAt (fun j i => (cs.getD j #[])[i]!) (#[] : Array Claims.Msg).size cs := by
    intro j hj
    funext i
    simp [List.getD_eq_getElem?_getD, hj]
  obtain ⟨-, hF⟩ := repOpening_feed (fun j i => (cs.getD j #[])[i]!) (xs.size + 2) coins t.st hmk t.sch
    (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) t.tPad rep t.g t.cm0 r.1.1.2.2
    (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty)) r.1.2 r.2 #[] cs (by omega) hc
  rw [hQ] at hlen ⊢
  have e := hF (by simp) (by rw [Array.size_empty, Nat.zero_add]; exact hlen)
  obtain ⟨L', hL, hL'⟩ := ringSwitchCut_gamma (fun j i => (cs.getD j #[])[i]!) (xs.size + 2)
    (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2 (by omega) r.1.2 #[] (by simp [hna])
  have hro : ((repOpening (cutAsks (fun j i => (cs.getD j #[])[i]!) (xs.size + 2))
      (fun lv g s => pure (commitFw lv g s)) t.st t.sch (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      t.tPad rep t.g t.cm0 r.1.1.2.2
      (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) t.extra t.lanePad
      (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))).run r.1.2).go #[] =
      (.error "past the openings", L') := by
    unfold repOpening
    rw [ringSwitchP_eq, cutAsks_toC]
    exact FlockComplete.bind_go_err hL
  have e2 := congrArg Array.toList (Prod.mk.inj (hro.symm.trans e)).2
  rw [hL', hpe, empty_toList, hna] at e2
  simp only [Array.toList_append, List.toList_toArray, Array.toList_empty, List.nil_append, Array.size_empty,
    Nat.sub_zero] at e2
  exact e2.symm

/-! ## The frame at the binding -/

/-- **The loop's state at the coins of a rep's `γ`**: after the algebraic phase's rounds and the ring switch's
`xs.size + 2` claim rounds and `γ`'s, the frame, where it takes `γ`'s coins at the layout `c''`, is outside every masked
phase, past the algebraic phase's opening, before level 0 and the inner proof, at the algebraic phase's last position
`c.nPiop`, and owes nothing. -/
theorem repState_opened {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hcr : cr.size = alg.size + (xs.size + 2) + 1) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2) + 1)).answer c'' cs = .ok m) :
    m.done = true ∧ m.on = false ∧ m.l0 = false ∧ m.doneL0 = false ∧ m.inner = none ∧ m.pos = c.nPiop ∧
      m.owed = none := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hAle : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  have hpop : cr.pop.size = alg.size + (xs.size + 2) := by rw [Array.size_pop, hcr]; omega
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.pop.extract alg.size cr.pop.size).toList.length = xs.size + 2 := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.pop.extract 0 alg.size) hdom hdg hcap hm hx hk hn hq)
    (by rw [hcsl]; exact openList_len t.sch (by omega)) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = alg.size + (xs.size + 2) + 1 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  obtain ⟨sv, hsv, hbody⟩ := openStep_switch t rep dom xs hi (cr.pop.extract 0 alg.size) hdom hdg hcap hm hmk
    (cr.pop.extract alg.size cr.pop.size).toList (by omega)
  have hB : ∀ i, i ≤ xs.size + 2 →
      (sched.getD (alg.size + i) (.empty, 0)).1.data.toList = rsBody sv (xs.size + 2) [] i :=
    fun i hi' => sched_bytes hdrop hbody (by omega) hi'
  have hC : ∀ i, i ≤ xs.size + 2 →
      (sched.getD (alg.size + i) (.empty, 0)).2 = (Count.openList (xs.size + 2) t.sch).getD i 0 :=
    fun i hi' => sched_count hal hs hdom hdg hcap hm hx hk hn hq (by omega)
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have hN0 : xs.size + 2 ≠ 0 := by omega
  have hsv' : ∀ i, (sv i).size < 2 ^ 64 := fun i => by rw [hsv]; decide
  have hp : Flags.PcsRound (sched.getD alg.size (.empty, 0)) := by
    have h7 := hC 0 (by omega)
    rw [openList_lt t.sch hN0 (by omega), Nat.add_zero] at h7
    have hb := hB 0 (by omega)
    rw [Nat.add_zero] at hb
    rw [show sched.getD alg.size (.empty, 0) = ((sched.getD alg.size (.empty, 0)).1, 7) from Prod.ext rfl h7]
    exact Switch.pcsRound_rs sv hN0 (hsv' 0) hb
  have hq' : ∀ k, 1 ≤ k → k ≤ alg.size + (xs.size + 2) → k ≠ alg.size →
      Quiet.QRound (sched.getD k (.empty, 0)) := by
    intro k h1 hkJ hkA
    by_cases hkl : k < alg.size
    · rw [hg k hkl, Array.getElem?_eq_getElem hkl, Option.getD_some]
      exact Quiet.rounds_quiet hm hdg hcap hdom hal k _ h1 (Array.getElem?_eq_getElem hkl)
    · obtain ⟨i, rfl⟩ : ∃ i, k = alg.size + i := ⟨k - alg.size, by omega⟩
      have hcnt := hC i (by omega)
      have hb := hB i (by omega)
      by_cases hiN : i < xs.size + 2
      · rw [openList_lt t.sch hN0 hiN] at hcnt
        rw [show sched.getD (alg.size + i) (.empty, 0) = ((sched.getD (alg.size + i) (.empty, 0)).1, 7) from
          Prod.ext rfl hcnt]
        exact Switch.qRound_rs sv (by omega) hiN (hsv' i) hb
      · obtain rfl : i = xs.size + 2 := by omega
        rw [openList_at t.sch hN0] at hcnt
        rw [show sched.getD (alg.size + (xs.size + 2)) (.empty, 0) =
          ((sched.getD (alg.size + (xs.size + 2)) (.empty, 0)).1, xs.size + 2) from Prod.ext rfl hcnt]
        exact Switch.qRound_gamma sv hN0 hx hb
  rw [repState_at c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2)) hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr (alg.size + (xs.size + 2)))).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    rcases Flags.sched_flags hc _ sched cr hdg hcap hdom h0 (by omega) hp _ hq' h2 with ho | ⟨-, hF⟩
    · exact absurd ha (answer_owed_none cs ho)
    · have e := answer_alg c'' _ cs hF.2.2.1 hF.1 ha
      subst e
      obtain ⟨a1, a2, a3, -, a5⟩ := hF
      obtain ⟨b1, b2, b3⟩ := a5 (by omega)
      exact ⟨b1, b2, a1, a2, a3, b3, rfl⟩
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

/-! ## Level 0's binding in the schedule -/

theorem coinAt_getD (cs : List (Array F128)) :
    CoinAt (fun j i => (cs.getD j #[])[i]!) (#[] : Array Claims.Msg).size cs := by
  intro j hj
  funext i
  simp [List.getD_eq_getElem?_getD, hj]

/-- **A rep's schedule holds level 0's binding**: fed the coins of the algebraic phase's rounds, the ring switch's and
`γ`'s, a rep's schedule asks next for level 0's binding, at the switch's target `T` and the level-0 cap, with `z`'s
squeeze of `n` coins. -/
theorem sched_binding {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hn0 : t.sch.n ≠ 0) (hcr : cr.size = alg.size + (xs.size + 2) + 1) :
    alg.size + (xs.size + 2) + 1 < sched.size ∧
      ∃ T, (sched.getD (alg.size + (xs.size + 2) + 1) (.empty, 0)).1.data.toList =
          encs (openBind T (FlockComplete.flat t.cm0.toC.cap)) ++ (Frame.sq 2 t.sch.n).enc ∧
        (sched.getD (alg.size + (xs.size + 2) + 1) (.empty, 0)).2 = t.sch.n := by
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hAle : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.extract alg.size cr.size).toList.length = xs.size + 2 + 1 := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
    (by rw [hcsl]; exact (openList_bind t.sch (by omega) hn0).1) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = alg.size + (xs.size + 2) + 2 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  refine ⟨by omega, ?_⟩
  rw [show alg.size + (xs.size + 2) + 1 = alg.size + (xs.size + 2 + 1) by omega]
  have hcnt := sched_count hal hs hdom hdg hcn hm hx hk hn hq (i := xs.size + 2 + 1) (by omega)
  rw [(openList_bind t.sch (by omega) hn0).2] at hcnt
  have hc := coinAt_getD (cr.extract alg.size cr.size).toList
  obtain ⟨r, hp, -, hL0⟩ := FlockComplete.LigLock.Fed.openStep_l0 t rep dom xs hi (cr.extract 0 alg.size) hdom hdg
    hcn hm hmk (cr.extract alg.size cr.size).toList
    (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!) #[] hc
    (by intro h0; rw [h0] at hlenF; simp at hlenF)
  have hna : r.1.1.2.2.size = xs.size + 2 := by
    have := Count.piop_na (cr.extract 0 alg.size) t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) (fun _ => 0) hdg hcn (by decide) hm _
      (Sched.trAlike_init dom hdom) #[]
    rw [hp] at this
    have h' : r.1.1.2.2.size = (claimsOf xs).size + 2 := this
    simpa only [claimsOf, Array.size_map] using h'
  obtain ⟨⟨ss, T, sws⟩, t', L', hL, hLl, ht'⟩ := Switch.ringSwitchCut_bodies
    (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!) (xs.size + 2 + 1)
    (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2 (by omega) r.1.2 #[] (by simp [hna])
  have hLs : L'.size = xs.size + 2 + 1 := by
    have := congrArg List.length hLl
    simp only [Array.length_toList, Array.toList_empty, List.nil_append, List.length_map, List.length_range,
      hna] at this
    exact this
  have key := hL0 (j := 0) (ss := ss) (T := T) (sws := sws) (tr₂ := t') (l₂ := L') (by omega) hk hn0 (by simp)
    (by rw [hLs, Nat.add_zero]; exact hL) (by rw [hLs]; simp only [Array.size_empty, Nat.zero_add]; omega)
  rw [hLl, hLs] at key
  simp only [Array.toList_empty, List.nil_append, Array.size_empty, Nat.sub_zero, Nat.add_zero, hna] at key
  have hk2 := List.getElem_of_eq key (i := xs.size + 2 + 1)
    (by simp only [List.length_map, List.length_take]; omega)
  rw [List.getElem_append_right (by simp)] at hk2
  simp only [List.getElem_map, List.getElem_take, List.length_map, List.length_range, Nat.sub_self,
    List.getElem_range] at hk2
  rw [FlockComplete.LigLock.l0BodyAt_zero, ht', empty_toList, List.nil_append] at hk2
  refine ⟨T, ?_, hcnt⟩
  rw [sched_entry hdrop (by omega)]
  unfold Claims.toMsg at hk2
  rw [hk2, openBind]

/-! ## Level 0's rounds in the schedule -/

theorem ligeritoCoins_l0 (sch : _root_.Flock.Schedule) :
    ∃ rest, (_root_.Flock.Firewall.Session.ligeritoCoins sch).toList =
      sch.n :: 1 :: (List.replicate sch.k0 2 ++ 2 :: 2 :: rest) := by
  unfold _root_.Flock.Firewall.Session.ligeritoCoins
  simp only [Array.toList_append, List.append_assoc, List.toList_toArray, Array.toList_replicate, List.cons_append,
    List.nil_append]
  exact ⟨_, rfl⟩

/-- Past the ring switch's, the opening's counts reach past level 0's first `k₀ + 4` rounds. -/
theorem openList_len_l0 {N : ℕ} (sch : _root_.Flock.Schedule) (hN : N ≠ 0) (hn0 : sch.n ≠ 0) :
    N + 1 + (sch.k0 + 4) ≤ (Count.openList N sch).length := by
  obtain ⟨rest, hr⟩ := ligeritoCoins_l0 sch
  have hf : (sch.n :: 1 :: (List.replicate sch.k0 2 ++ 2 :: 2 :: rest)).filter (· != 0) =
      sch.n :: 1 :: (List.replicate sch.k0 2 ++ 2 :: 2 :: rest.filter (· != 0)) := by
    simp [List.filter_cons, List.filter_append, List.filter_replicate, hn0]
  rw [openList_eq sch hN, hr, hf]
  simp only [List.length_append, List.length_replicate, List.length_cons]
  omega

/-- **A rep's schedule holds level 0's rounds**: fed the coins of the algebraic phase's rounds, the ring switch's, `γ`'s
and level 0's first `j`, a rep's schedule past the ring switch's rounds is level 0's first `j + 1` bodies at zero pads,
at the fed coins from the binding on, the switch's target and basis, with the opening's counts. -/
theorem sched_l0 {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hn0 : t.sch.n ≠ 0) (hm14 : 14 ≤ t.st.m) {j : ℕ} (hj : j < t.sch.k0 + 4)
    (hcr : cr.size = alg.size + (xs.size + 2) + 1 + j) :
    alg.size + (xs.size + 2) + 1 + j < sched.size ∧
      ∃ T sws, (∃ zc lc, FlockComplete.LigLock.Fed.PtsAt t.st.m t.st.fold.kLog
            (FlockComplete.LigLock.Fed.recCoin (cr.extract 0 alg.size)) 0 zc lc ∧
          sws = FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims t.st.m t.st.kLog zc lc ++ claimsOf xs)
            ⟨fun k i => ((cr.extract alg.size cr.size).toList.getD k #[])[i]!,
              fun i => ((cr.extract alg.size cr.size).toList.getD (xs.size + 2) #[])[i]!⟩) ∧ ∀ i ≤ j,
        (sched.getD (alg.size + (xs.size + 2) + 1 + i) (.empty, 0)).1.data.toList =
          FlockComplete.LigLock.l0BodyAt (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!)
            (xs.size + 2 + 1) [] T (FlockComplete.flat t.cm0.toC.cap) t.sch t.g
            (FlockComplete.basisVals sws (t.st.m - 7)) (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
            (Array.replicate (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).len 0) rep t.extra
            (fun _ _ => 0) i ∧
        (sched.getD (alg.size + (xs.size + 2) + 1 + i) (.empty, 0)).2 =
          (Count.openList (xs.size + 2) t.sch).getD (xs.size + 2 + 1 + i) 0 := by
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hAle : alg.size ≤ sched.size := by rw [hrest, Array.size_append]; omega
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs (by omega)
  have hcsl : (cr.extract alg.size cr.size).toList.length = xs.size + 2 + 1 + j := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hlenF := feed_part _ _ _
    (Count.openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
    (by rw [hcsl]; have := openList_len_l0 t.sch (N := xs.size + 2) (by omega) hn0; omega) hnf
  rw [hcsl] at hlenF
  have hsz : sched.size = alg.size + (xs.size + 2) + 1 + j + 1 := by
    have := congrArg List.length hdrop
    simp only [List.length_drop, List.length_map, Array.length_toList] at this
    omega
  refine ⟨by omega, ?_⟩
  have hc := coinAt_getD (cr.extract alg.size cr.size).toList
  obtain ⟨r, hp, -, hL0⟩ := FlockComplete.LigLock.Fed.openStep_l0 t rep dom xs hi (cr.extract 0 alg.size) hdom hdg
    hcn hm hmk (cr.extract alg.size cr.size).toList
    (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!) #[] hc
    (by intro h0; rw [h0] at hlenF; simp at hlenF)
  have hna : r.1.1.2.2.size = xs.size + 2 := by
    have := Count.piop_na (cr.extract 0 alg.size) t.st (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.flat t.cm0.cap) ByteArray.empty (claimsOf xs) (fun _ => 0) hdg hcn (by decide) hm _
      (Sched.trAlike_init dom hdom) #[]
    rw [hp] at this
    have h' : r.1.1.2.2.size = (claimsOf xs).size + 2 := this
    simpa only [claimsOf, Array.size_map] using h'
  obtain ⟨⟨ss, T, sws⟩, t', L', hL, hLl, ht'⟩ := Switch.ringSwitchCut_bodies
    (fun j i => ((cr.extract alg.size cr.size).toList.getD j #[])[i]!) (xs.size + 2 + 1 + j)
    (fun k => switchVec t.g r.1.1.2.2[k]!) r.1.1.2.2 (by omega) r.1.2 #[] (by simp [hna])
  have hLs : L'.size = xs.size + 2 + 1 := by
    have := congrArg List.length hLl
    simp only [Array.length_toList, Array.toList_empty, List.nil_append, List.length_map, List.length_range,
      hna] at this
    exact this
  have key := hL0 (j := j) (ss := ss) (T := T) (sws := sws) (tr₂ := t') (l₂ := L') hj hk hn0 (by simp)
    (by rw [hLs]; exact hL) (by rw [hLs]; simp only [Array.size_empty, Nat.zero_add]; omega)
  obtain ⟨zc, lc, hcl, hpts⟩ := FlockComplete.LigLock.Fed.repPiop_rec _ _ _ _ _ _ _ hm14 _ hp
  obtain ⟨hsw, -, -⟩ := FlockComplete.ringSwitchCut_sws _ _ _ _ (by omega) hL
  rw [hLl, hLs, ht', empty_toList] at key
  simp only [Array.toList_empty, List.nil_append, Array.size_empty, Nat.sub_zero, hna] at key
  refine ⟨T, sws, ⟨zc, lc, hpts, ?_⟩, fun i hi => ⟨?_, ?_⟩⟩
  · rw [hsw, ← hcl]
    simp only [Array.size_empty, Nat.zero_add, hna]
  · have hk2 := List.getElem_of_eq key (i := xs.size + 2 + 1 + i)
      (by simp only [List.length_map, List.length_take]; omega)
    rw [List.getElem_append_right (by simp)] at hk2
    simp only [List.getElem_map, List.getElem_take, List.length_map, List.length_range, Nat.add_sub_cancel_left,
      List.getElem_range] at hk2
    rw [show alg.size + (xs.size + 2) + 1 + i = alg.size + (xs.size + 2 + 1 + i) by omega,
      sched_entry hdrop (by omega)]
    unfold Claims.toMsg at hk2
    rw [hk2]
  · rw [show alg.size + (xs.size + 2) + 1 + i = alg.size + (xs.size + 2 + 1 + i) by omega]
    exact sched_count hal hs hdom hdg hcn hm hx hk hn hq (by omega)

end Claims.ZKServer.Bind
