module

public import Proofs.Flock.ZkAny.RelayLoopCur
public import Proofs.Flock.ZkAny.RelayRoundFirst
public import Proofs.Flock.ZkAny.RelayPrior
public import Proofs.Flock.ZkAny.RelaySwitch

/-!
# The frame round's own rep one round on, from its stream's first round

`loopKept_round`'s `hcur` at a round that is its stream's first, where the next round is the same stream's:

* `roundPlace_first_ok`: at a stream's first round that `roundPlace` places, the session's entry, the table's
  statement, name and level 0 are there and `withAlg` schedules the frame's algebraic phase.
  `roundPlace_rep0`: at rep 1's, rep 0's frame is there too.
* `roundRun_alg_first`: where the algebraic phase has a round, the frame the session's next request finds has not
  begun its opening.
* **`repJ_same_first`**: so the model's coin step takes the round's coins, and where the next round is the same
  stream's the frame the session's next request finds keeps `SameInv` for the stream's first round
  (`roundRun_same_first`), the readied frame being the rep's run from its start (`repRun_first0`, `roundPlace_prior`)
  and the model's state the rep's start (`ofLoop_answered_switch`), its placeholder, configuration and schedule the
  model's (`freshOf_ok`, `ofLoop_cfgAlike`, `Sched.schedSoFar_alike`), the round's count the plan's, and the table's
  level 0 the one the loop drew (`BuiltAt`), so giving a table whose commitment's cap is its root
  (`tab_of_placeholder`, `cap_of_drawn`).
* `tab_of_same`, `tab_of_placeholder`: a level 0 at another's `m` and `t_pad` gives a table where it does, so one
  sampled at the table's `m` gives one where the rep's placeholder is there.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Masker Place Req)

set_option maxHeartbeats 1000000 in
/-- **At a stream's first round `roundPlace` places, what it read is there**: the session's entry, the table's
statement, name and level 0 bound for the session, and the frame `withAlg` gives on the found frame. -/
theorem roundPlace_first_ok {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hfst : (p.coins.isEmpty && !p.st.begun) = true) :
    ∃ (s : Release) (name : String) (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt) (p' : Rep),
      R.sessions.find? (·.sid == sid) = some s ∧ s.sts[i]? = some stm ∧ s.spec.tables[i]? = some name ∧
        _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0 ∧
        _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p' := by
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  rcases hs : R.sessions.find? (·.sid == sid) with _ | s
  · simp only [hs] at h <;> cases h
  simp only [hs] at h
  rcases ht : s.sts[i]? with _ | stm
  · simp only [ht] at h <;> cases h
  simp only [ht] at h
  rcases hn : s.spec.tables[i]? with _ | name
  · simp only [hn] at h <;> cases h
  simp only [hn] at h
  rcases hl : _root_.Flock.Firewall.Level0.ofSession L0 sid name with _ | l0
  · simp only [hl] at h <;> cases h
  simp only [hl] at h
  rcases hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p with e | p'
  · exfalso
    repeat' (first
      | simp only [hfst, hw, bind, Except.bind, pure, Except.pure, Bool.false_eq_true, ↓reduceIte] at h
      | split at h)
    all_goals first
      | (cases h; done)
      | exact absurd (by assumption) throw_ne_ok
  · exact ⟨s, name, l0, stm, p', by first | rfl | exact hs, by first | exact ht | rfl, by first | exact hn | rfl,
      by first | exact hl | rfl, hw⟩

set_option maxHeartbeats 1000000 in
/-- **At rep 1's first round rep 0's frame is there**: `roundPlace` refuses rep 1's first round before rep 0's
frame. -/
theorem roundPlace_rep0 {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i 1 = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i 1 dom content n = .ok (q, pl, f, nd))
    (hfst : (p.coins.isEmpty && !p.st.begun) = true) : ∃ j0 p0, R.find sid i 0 = some (j0, p0) := by
  rcases h0 : R.find sid i 0 with _ | ⟨j0, p0⟩
  · exfalso
    have h11 : ((1 : Nat) == 1) = true := rfl
    unfold _root_.Flock.Firewall.Frames.roundPlace at h
    simp only [hf, bind, Except.bind, pure, Except.pure] at h
    repeat' (first
      | simp only [hfst, h11, h0, bind, Except.bind, pure, Except.pure, Bool.false_eq_true, ↓reduceIte] at h
      | split at h)
    all_goals first
      | (cases h; done)
      | exact absurd (by assumption) throw_ne_ok
  · exact ⟨j0, p0, by first | rfl | exact h0⟩

/-- **At a stream's first round, where the algebraic phase has a round, the round leaves the rep's opening
unbegun**: on a fresh frame, `placeOf` places the round on the frame `roundPlace` readied as it is
(`Loop.placeOf_alg`), which `withAlg` and rep 1's prior step leave with the fresh frame's opening, and the frame the
session's next request finds has not begun it either. -/
theorem roundRun_alg_first {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) {pre : List Req} {ev : List (Nat × ByteArray × Place × Array F128)}
    (hR : RepRun pre ev q) {p : Rep} (hfa : frameAt R sid i r = some p) (hp : FreshRep p) {s : Release}
    {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0} (hss : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    {p' : Rep} (hw : _root_.Flock.Firewall.Frames.withAlg s l0 name p = .ok p') (hpos : 0 < p'.alg.size) :
    ∃ p₃, frameAt R₃ sid i r = some p₃ ∧ p₃.opening = none ∧ p₃.opened = false := by
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hfr : (frameAt R sid i r).isSome := by
    rw [hfa]
    rfl
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  obtain ⟨p'', pl₀, n₀, hpo, -, hp''⟩ := roundPlace_first_of hf hrp hfst hss hn hl0 hw
  obtain ⟨-, -, hco, hop, hod, hto⟩ := withAlg_keep hw
  have hka : p''.coins = p'.coins ∧ p''.alg = p'.alg ∧ p''.opening = p'.opening ∧ p''.opened = p'.opened ∧
      p''.toOpening = p'.toOpening := by
    rcases hp'' with rfl | ⟨st₁, rfl⟩
    · exact ⟨rfl, rfl, rfl, rfl, rfl⟩
    · exact ⟨rfl, rfl, rfl, rfl, rfl⟩
  have hlt : p''.coins.size < p''.alg.size := by
    rw [hka.1, hka.2.1, hco, hp.coins, Array.size_empty]
    exact hpos
  rw [Loop.placeOf_alg R.salts l0 s p'' _ hlt] at hpo
  have hq : q = p'' := congrArg (fun x => x.2.2) (Except.ok.inj hpo).symm
  subst hq
  obtain ⟨st₂, -, hat, -⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hc hR
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl') with ⟨a', st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  have hto' : q.toOpening = false := by rw [hka.2.2.2.2, hto, hp.toOpening]
  have hop' : q.opening = none := by rw [hka.2.2.1, hop, hp.opening]
  have hod' : q.opened = false := by rw [hka.2.2.2.1, hod, hp.opened]
  have key : ∀ (p' : Rep) (st'' : _root_.Flock.Firewall.Frame.State) (cs : Array F128), p'.toOpening = false →
      p'.opening = none → (Loop.coinsOf p' st'' cs).opening = none := fun p' st'' cs h1 h2 => by
    simp only [Loop.coinsOf, h1, h2]
  exact ⟨_, hat, key { q with st := st', vals := v } _ _ hto' hop', hod'⟩

/-- **A level 0 of another's `m` and `t_pad` gives a table where it does.** -/
theorem tab_of_same {st : _root_.Flock.Setup} {l l' : _root_.Flock.Firewall.Level0.Level0}
    {t : _root_.Flock.Firewall.Finish.Tab} (h : _root_.Flock.Firewall.Finish.tabOfLevel0 st l = .ok t)
    (hm : l'.m = l.m) (htp : l'.tPad = l.tPad) : ∃ t', _root_.Flock.Firewall.Finish.tabOfLevel0 st l' = .ok t' := by
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0 at h ⊢
  rw [hm, htp]
  simp only [bind, Except.bind] at h ⊢
  split at h
  · cases h
  rename_i h1
  rw [ite_eq_right h1]
  split at h
  · cases h
  rename_i h2
  rw [ite_eq_right h2]
  rcases hs : _root_.Flock.Zk.schedule st.m with e | ⟨sch, tPad⟩
  · rw [hs] at h
    cases h
  · rw [hs] at h
    simp only at h ⊢
    split at h
    · cases h
    rename_i h3
    rw [ite_eq_right h3]
    exact ⟨_, rfl⟩

/-- **A level 0 sampled at the table's `m` gives a table**, where rep stream `k`'s placeholder is there: the
placeholder's level 0 gives one, at the same `m` and the `t_pad` of its schedule (`Level0.ofDraws_ok`). -/
theorem tab_of_placeholder {spec : SessionSpec} {sts : Array Stmt} {k : Nat} {link : Array F128} {c' : Cfg}
    {t' : _root_.Flock.Firewall.Finish.Tab} {dom' : ByteArray} {xs' : Array (Array F128)} {hi' : Array ByteArray}
    (hph : Loop.placeholder spec sts k link = .ok (c', t', dom', xs', hi')) {g : Array F128} {d : ByteArray}
    {l : _root_.Flock.Firewall.Level0.Level0} (hl : _root_.Flock.Firewall.Level0.ofDraws sts[k / 2]!.m g d = .ok l) :
    ∃ t, _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit sts[k / 2]!) l = .ok t := by
  have bok : ∀ {α β : Type} {x : V α} {f : α → V β} {b : β}, x >>= f = .ok b → ∃ a, x = .ok a ∧ f a = .ok b := by
    intro α β x f b hb
    cases x with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  unfold Loop.placeholder at hph
  obtain ⟨⟨lay, mask⟩, -, hph⟩ := bok hph
  obtain ⟨⟨sch, tPad⟩, -, hph⟩ := bok hph
  obtain ⟨l₀, hl₀, hph⟩ := bok hph
  obtain ⟨t₀, ht₀, -⟩ := bok hph
  obtain ⟨hm, -, ⟨s, hz, -⟩, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl
  obtain ⟨hm₀, -, ⟨s₀, hz₀, -⟩, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl₀
  exact tab_of_same ht₀ (hm.trans hm₀.symm) (Prod.mk.inj (Except.ok.inj (hz.symm.trans hz₀))).2

open Loop LeanRelease in
/-- **The round's coins taken, and the round's own rep one round on from its stream's first round where the next round
is the same stream's**: at the loop's own frames (`hrel`), the session's entry being the setup's with the `Commit`'s
words as its link (`hK2`), the model's own table at the rep the one the rep's level 0 gives (`hK3`; of the placeholder's
shape where `real` answers the loop's own, `hR`, within the bounds `hS` and `hP` give), and every level 0 bound to the
session the one the loop drew for its table (`BuiltAt`), the rep streams' names distinct (`hnd`), where the round is the
first of table `i`'s rep `r`'s stream (`hk`), the rep's frame fresh (`FreshAt`) and readied at the model's state but for
`stopped` (`hu₀`): the frame the session's next request finds keeps `SameInv` for the stream's first round
(`roundRun_same_first`), the frame `roundPlace` readied being the rep's run from its start (`repRun_first0` at rep 0,
`roundPlace_prior` at rep 1), the model's state the rep's start (`ofLoop_answered_switch`), its placeholder,
configuration and schedule the model's (`freshOf_ok`, `ofLoop_cfgAlike`, `Sched.schedSoFar_alike`), the table's level 0
the one sampled at its statement's `m` (`tables_inj`), so giving a table as the placeholder's does (`tab_of_placeholder`)
whose commitment's cap is its root (`cap_of_drawn`), and the round's count the plan's (`Loop.loopNr_succ`,
`Count.plan_sched`), with its opening unbegun (`roundRun_alg_first`); the rep's salts are not read, the algebraic phase
having a round past the first's coins. -/
theorem repJ_same_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)}
    (hrel : M.L = ofLoop T L₀ real) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    (hS : ∀ (k : Nat) (link : Array F128) (c' : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray)
      (xs : Array (Array F128)) (hi : Array ByteArray), k < spec.repStreams.size →
      link.size ≤ _root_.Flock.Firewall.Frames.COIN_MAX → placeholder spec sts k link = .ok (c', t, dom, xs, hi) →
      dom.size < 2 ^ 64 ∧ xs.size + 2 < 2 ^ 64)
    (hR : Count.RealTabs sts (real M.x M.w M.ρ))
    (hP : ∀ k, k < spec.repStreams.size → sts[k / 2]!.mPts = spec.mPts ∧ 0 < spec.mPts ∧
      spec.points * spec.mPts ≤ _root_.Flock.Firewall.Frames.COIN_MAX)
    {sid : Nat} {st : State} {pl : Placed} {stream : String} {i r : Nat} {dom : ByteArray} {v : Claims.Msg} {e : Env}
    {σ σ' : LoopAt} {e0 : Claims.Msg × Claims.Msg} {h1 : List (Claims.Msg × Claims.Msg)} {q₀ : Nat}
    (hmem : (i, r, stream) ∈ spec.repStreams.toList) (hh : M.hist σ.c = e0 :: h1) (hq : h1.length = q₀ + 1)
    (hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, 0)) {m₀ : Masker}
    (hA₀ : M.L.answered M.x M.ρ (M.hist σ.c) = .ok m₀) (hlast : (M.hist σ.c).length < M.L.last M.x)
    (hK2 : ∀ rel : Release, sessAt σ.R sid = some rel → rel.spec = spec ∧ rel.sts = sts ∧
      rel.link = some (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0))
    (hK3 : ∀ (rel : Release) (name : String) (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt)
      (t : _root_.Flock.Firewall.Finish.Tab), sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
      ∃ hr, real M.x M.w M.ρ (2 * i + r) = .ok (t, hr))
    (hB : BuiltAt M spec sts σ.L0 sid) (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hfr : FreshAt σ.R sid i r) (hfs : (frameAt σ.R sid i r).isSome)
    (hpc : ∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c))
    (hr : RoundRun sid st pl stream (loopNr M.x h1.length) i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hreq : st.request pl (.round stream (loopNr M.x h1.length)) = none) {q : Rep} {pl' : Place} {f : Bool}
    {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ (loopNr M.x h1.length) =
      .ok (q, pl', f, nd))
    (hu₀ : unstop q.st.m = unstop m₀)
    {o : ByteArray} (ho : e.round = .ok o) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr M.x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {m' : Claims.Msg} (hh' : M.hist σ'.c = M.hist σ.c ++ [(m', a.data.toList)]) (hSt : Stays sid σ σ') :
    M.L.fed M.x M.ρ (M.hist σ'.c) = true ∧
      ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 = 2 * i + r → RepJ M sid spec sts i r σ') := by
  have hlen : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := congrArg Prod.fst hk
    unfold nextAt at this
    rwa [hlen] at this
  have hk2 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = 0 := by
    have := congrArg Prod.snd hk
    unfold nextAt at this
    rwa [hlen] at this
  have hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀ = (2 * i + r, 0) := Prod.ext hk1 hk2
  have hr2 := rep_lt_two hmem
  have hmod : (2 * i + r) % 2 = r := by omega
  have hdiv : (2 * i + r) / 2 = i := by omega
  obtain ⟨p, hfa⟩ := Option.isSome_iff_exists.mp hfs
  have hp : FreshRep p := hfr p hfa
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hf)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hf)
  have hfst : (p.coins.isEmpty && !p.st.begun) = true := by
    rw [hp.coins, hp.st]
    rfl
  obtain ⟨rel, name, l0, stm, p', hss, hstm', hn, hl0, hw⟩ := roundPlace_first_ok hf hrp hfst
  have hsa : sessAt σ.R sid = some rel := hss
  obtain ⟨hsp, hsts, hlk⟩ := hK2 rel hsa
  rw [hh, List.tail_cons] at hlk
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  have hRq : ∃ pre : List Req, RepRun pre [] q := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · obtain ⟨p'', pl₀, n₀, hpo, -, hp''⟩ := roundPlace_first_of hf hrp hfst hss hn hl0 hw
      obtain ⟨-, -, hk2'⟩ := placeOf_run hpo
      obtain ⟨-, -, hk1'⟩ := withAlg_run hw
      have hcq : q.coins = #[] := by
        rw [hk2']
        rcases hp'' with rfl | ⟨st₁, rfl⟩
        · rw [hk1', hp.coins]
        · show p'.coins = #[]
          rw [hk1', hp.coins]
      exact ⟨[], repRun_start (by rw [roundPlace_st hf hrp (by decide), hp.st]) hcq⟩
    · obtain ⟨j0, p0, h0⟩ := roundPlace_rep0 hf hrp hfst
      exact ⟨_, (roundPlace_prior hf hrp (fresh_repRun hp) hss hn hl0 hw h0).2⟩
  obtain ⟨pre, hRq⟩ := hRq
  -- the model's side
  have hA₁ : (ofLoop T L₀ real).answered M.x M.ρ (e0 :: h1) = .ok m₀ := by
    rw [← hrel, ← hh]
    exact hA₀
  have hlast' : (e0 :: h1).length < (ofLoop T L₀ real).last M.x := by
    rw [← hrel, ← hh]
    exact hlast
  have hsw : 2 ≤ h1.length → repOf M.x (h1.length + 1) ≠ repOf M.x h1.length := by
    intro h2 he
    have e1 : repOf M.x (h1.length + 1) = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 := by
      rw [show h1.length + 1 = q₀ + 2 by omega]
      exact repOf_add_two hx q₀
    have e2 : repOf M.x h1.length = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ - 1)).1 := by
      rw [show h1.length = q₀ - 1 + 2 by omega]
      exact repOf_add_two hx (q₀ - 1)
    rw [e1, e2] at he
    have h' := Loop.locate_succ (roundsOf spec sts) 0 (q₀ - 1)
      (by rw [show q₀ - 1 + 1 = q₀ by omega]; exact he)
    rw [show q₀ - 1 + 1 = q₀ by omega, hk2] at h'
    omega
  have hm := ofLoop_answered_switch T L₀ real M.x M.ρ e0 h1 hsw hA₁
  have hcs1 : (h1.map fun y => ZKMal.CoinLayer.rd y.2).length = q₀ + 1 := by rw [List.length_map, hq]
  obtain ⟨c', t', dom', xs', hip, -, hph, -, -⟩ := Loop.freshOf_ok M.x _ hx hcs1 hm
  have hph' : placeholder spec sts (2 * i + r) (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0) =
      .ok (c', t', dom', xs', hip) := by
    rw [← hk1]
    exact hph
  have hq2 : (e0 :: h1).length = q₀ + 2 := by simp only [List.length_cons, hq]
  have hc := ofLoop_cfgAlike T L₀ real M.x M.ρ (e0 :: h1) hx hq2 hph
  obtain ⟨lay, mask, -, -, hdm, hxe⟩ := Loop.placeholder_ok hph'
  have hname : spec.tables[i]! = name := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hn
    rw [← hsp, getElem!_pos rel.spec.tables i hlt, he]
  have hstmb : sts[i]! = stm := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hstm'
    rw [← hsts, getElem!_pos rel.sts i hlt, he]
  obtain ⟨jb, gb, db, _, hjb, -, hdb⟩ := hB name l0 hl0
  rw [tables_inj (j := i) hnd hjb (by rw [← hsp]; exact hn), hstmb] at hdb
  obtain ⟨t, htab⟩ : ∃ t, _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t := by
    have h := tab_of_placeholder (g := gb) (d := db) (l := l0) hph' (by rw [hdiv, hstmb]; exact hdb)
    rwa [hdiv, hstmb] at h
  have hcap : _root_.Flock.Firewall.Finish.flat t.cm0.cap = l0.root := cap_of_drawn hdb htab
  have hdom : (rel.spec.domain name r).toUTF8 = dom' := by rw [hdm, hdiv, hmod, hname, hsp]
  have hkl : 2 * i + r < spec.repStreams.size := rep_lt_size hmem
  have hls := wordsAt_zero_size M.x (cs := h1.map fun y => ZKMal.CoinLayer.rd y.2)
    (by rw [List.length_map, hq]; omega)
  obtain ⟨hdm', hxs2⟩ := hS (2 * i + r) _ c' t' dom' xs' hip hkl (by rw [hls]; exact Nat.min_le_right _ _) hph'
  obtain ⟨hmp, hmp0, hpc'⟩ := hP (2 * i + r) hkl
  have hnr0 : loopNr M.x 0 = spec.points * spec.mPts := by
    show (Loop.reqOf M.x 0).2 = _
    rw [Loop.reqOf_eq hx]
    rfl
  have hpts : (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0).size / sts[(2 * i + r) / 2]!.mPts =
      spec.points := by
    rw [hls, hnr0, Nat.min_eq_left hpc', hmp, Nat.mul_div_cancel _ hmp0]
  obtain ⟨hr', hre⟩ := hK3 rel name l0 stm t hsa hn hl0 hstm' htab
  have hTa := Count.real_tabAlike hR hph' hre
  obtain ⟨hdg, hmb, hnb, hlq, hpb, hdb⟩ := Count.real_bounds hph' hTa
  -- the frame's schedule on no coins, the algebraic phase `withAlg` scheduled
  obtain ⟨xs, hxsd⟩ : ∃ xs, xs = _root_.Flock.Firewall.Drive.extras stm
      (_root_.Flock.Firewall.Drive.linkPoints stm (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0)) :=
    ⟨_, rfl⟩
  have hxx : xs = xs' := by rw [hxsd, hxe, hdiv, hstmb]
  have hI' := repInv_withAlg hp hw hst' htab hcap xs #[]
  have hpos : 0 < p'.alg.size := Loop.rounds_pos hI'.alg
  obtain ⟨-, hrw, hcw, -, -, -⟩ := withAlg_keep hw
  have hsc : schedSoFar t p.rep (rel.spec.domain name r).toUTF8 xs #[] p.coins = .ok p'.alg := by
    have e1 := Loop.schedSoFar_alg t _ xs #[] p' hI'.alg (by rw [hcw, hp.coins, Array.size_empty]; exact hpos)
    rw [hrw, hcw, hpr] at e1
    rw [hpr]
    exact e1
  have hsc2 : schedSoFar t r dom' xs' #[] #[] = .ok p'.alg := by
    have e2 := hsc
    rw [hpr, hp.coins, hdom, hxx] at e2
    exact e2
  have hcn : (_root_.Flock.Firewall.Finish.flat t'.cm0.cap).size < 2 ^ 64 := by
    rw [← hTa.2.2.2.2.1]
    exact hTa.2.2.2.2.2
  have hal := Sched.schedSoFar_alike t t' r dom' xs' #[] hip #[] hTa.1 hTa.2.1 hTa.2.2.1 hTa.2.2.2.1
    hTa.2.2.2.2 hdm' hdg hmb hxs2 hnb hlq hpb hdb
  rw [hsc2] at hal
  cases hs' : schedSoFar t' r dom' xs' hip #[] with
  | error e' =>
    rw [hs'] at hal
    exact (hal : False).elim
  | ok sched' =>
  rw [hs'] at hal
  have hlt' : 0 < sched'.size := schedSoFar_nil_pos hs'
  have hqs : q₀ < (roundsOf spec sts).sum :=
    LeanRelease.loop_round_in_plan T (withFrames T L₀ real) M.x M.ρ (e0 :: h1) hA₁ (Nat.ne_of_lt hlast') hx hq2
  have hn' : loopNr M.x h1.length = (sched'.getD 0 (.empty, 0)).2 := by
    rw [hq]
    exact (Loop.loopNr_succ hx hki hqs).trans (Count.plan_sched spec sts (2 * i + r) 0 _ c' t' dom' xs' hip
      #[] sched' hph' (by rw [hmod]; exact hs') hlt' hpts hdm' hcn hxs2)
  have ho' := (hr.round hreq).symm.trans ho
  obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
  have hcf : p.cfg = (ofLoop T L₀ real).cfg M.x M.ρ (e0 :: h1) := by
    rw [hpc p hfa, hrel, hh]
  have hlink : rel.link = some (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0) := hlk
  obtain ⟨hfd, hsm⟩ := roundRun_same_first T L₀ real M.x M.ρ e0 h1 hx hq hk2 hph hc
    (by rw [hk1, hmod]; exact hs') hn' hr hreq hrp ho' hput ha hb hs hco hRq hfa hp hss hn hl0 hw hst' hlink
    (r0 := ByteArray.empty) (q0 := 0) (hi := #[]) (fun h0 => absurd h0 (by omega)) htab hcap hxsd hsc hal hpos hcf
    hm hu₀
  have hsnd : (e0 :: (h1 ++ [(v, a.data.toList)])).map Prod.snd = (M.hist σ'.c).map Prod.snd := by
    rw [hh', hh]
    simp
  refine ⟨by rw [hrel]; exact (ofLoop_fed_snd T L₀ real M.x M.ρ hsnd).symm.trans hfd, fun hsame => ?_⟩
  have hj2 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).2 = 0 + 1 := by
    rw [Loop.locate_succ _ _ _ (hsame.trans hk1.symm), hk2]
  obtain ⟨-, p₃, hat, hI3⟩ := hsm _ (by rw [hq]; exact hj2)
  have hI4 := sameInv_snd hsnd hI3
  have hcs : ((h1 ++ [(v, a.data.toList)]).map fun y => ZKMal.CoinLayer.rd y.2) =
      (M.hist σ'.c).tail.map fun y => ZKMal.CoinLayer.rd y.2 := by
    rw [hh', hh, List.cons_append, List.tail_cons]
    simp only [List.map_append, List.map_cons, List.map_nil]
  rw [hcs, ← hrel] at hI4
  have hc3 : p₃.coins.size = 0 + 1 := by rw [hI4.coins, Array.size_map, Array.size_range]
  have hpos3 : 0 < p₃.alg.size := Loop.rounds_pos hI4.inv.alg
  have hl' : (M.hist σ'.c).length - 2 = q₀ + 1 := by
    rw [hh', hh, List.length_append, List.length_cons, hq, List.length_singleton]
    omega
  refine ⟨p₃, rel, name, l0, hat, ?_, hn, ?_, ?_⟩
  · rw [sessAt_stay (hSt.2 name l0 hl0) (by rw [hsa]; rfl)]
    exact hsa
  · rw [hSt.1 name]
    exact hl0
  · unfold StreamAt
    rw [hl']
    refine ⟨fun _ => ?_, fun he _ => absurd (hsame.symm.trans he) (by omega)⟩
    rw [hmod]
    refine ⟨0, _, stm, t, xs, #[], pre, _, hj2, hlink, by rw [hdiv]; exact hstm', htab, hcap, hxsd, hI4,
      fun hlt3 => absurd hlt3 (by omega), fun _ => ?_⟩
    obtain ⟨p₅, hat5, hop5, hod5⟩ := roundRun_alg_first hr hreq hrp ho' hput ha hb hs hco hRq hfa hp hss hn hl0 hw
      hpos
    obtain rfl := Option.some.inj (hat5.symm.trans hat)
    exact ⟨hop5, hod5⟩

end Claims.ZKServer.Relay
