module

public import Proofs.Flock.ZkAny.RelayLoopKept
public import Proofs.Flock.ZkAny.RelayLoopSalts
public import Proofs.Flock.ZkAny.RelayGap

/-!
# The frame round's own rep one round on

`loopKept_round` takes `RepJ` of the round's own rep at the session's next request. Here it is, case by case on the
model's next round:

* `rep_lt_two`: a rep stream's rep is `0` or `1`. `LoopModel.hist_ans`: the history one message on, where the auditor
  answered coins as `V*` did, ends with their bytes.
* `sched_of_placeOf`: where `placeOf` places the rep's next round, its schedule so far is one (the converse of
  `Loop.placeOf_sched`).
* **`repJ_next`**: past its stream's first round, where the next round is another stream's, the frame the session's
  next request finds holds the history's words for each of its stream's rounds (`roundRun_last`, `count_last`).
* `roundRun_alg`: in its algebraic phase a round leaves its rep's opening unbegun. `sameInv_salts_kept`: so a frame
  keeping `SameInv` keeps it at salts the round lemmas read, the rep's from its opening's first round on.
* `ofLoop_placeholder`, `ofLoop_cfgAlike`: past a rep's first round the loop's frame takes coins only at the rep's
  placeholder, whose configuration is alike the tape's.
* **`repJ_same`**: past its stream's first round and within the rep's schedule, at the loop's own frames, the
  session's entry being the setup's and the model's own table the rep's level 0's, the model's coin step takes the
  round's coins, and where the next round is the same stream's the frame the session's next request finds keeps
  `SameInv` one round more of the rep (`roundRun_same`).
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed byte?)
open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Masker Place)

/-- **A rep stream is among the setup's rep streams' indices.** -/
theorem rep_lt_size {spec : SessionSpec} {i r : Nat} {s : String} (h : (i, r, s) ∈ spec.repStreams.toList) :
    2 * i + r < spec.repStreams.size := by
  have hme := rep_index h
  by_contra hc
  rw [Array.getElem?_eq_none (by omega)] at hme
  cases hme

/-- **A rep stream's rep is `0` or `1`.** -/
theorem rep_lt_two {spec : SessionSpec} {i r : Nat} {s : String} (h : (i, r, s) ∈ spec.repStreams.toList) :
    r < 2 := by
  have hme := rep_index h
  have hm := rep_lt_size h
  rw [Array.getElem?_eq_getElem hm, FlockDeploy.repStreams_getElem spec hm, Option.some.injEq, Prod.mk.injEq,
    Prod.mk.injEq] at hme
  obtain ⟨-, hr, -⟩ := hme
  omega

open Claims.ZKMal in
/-- **The history one message on ends with the coins' bytes** where the auditor answered coins to it as `V*` did. -/
theorem LoopModel.hist_ans {W Rr : Type} (M : LoopModel W Rr) {j : Nat} {e : Env} (h : M.Ans j e) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) :
    ∃ m', M.hist (j + 1) = M.hist j ++ [(m', a.data.toList)] := by
  obtain ⟨hj, hA⟩ := h
  refine ⟨(M.τ[j]'hj).1, ?_⟩
  unfold LoopModel.hist histOf
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, List.map_append, List.map_cons,
    List.map_nil]
  simp only [hA a ha hb, msgOf]

open Loop in
/-- **Where `placeOf` places the rep's next round, its schedule so far is one**: at a frame keeping its schedule so far
(`Loop.RepInv`) on the table and extras its session gives it and the salts the loop's salts give it where `placeOf`
reads them, the opening `placeOf` goes on with has not failed. -/
theorem sched_of_placeOf {salts : Array (ByteArray × Nat × Array ByteArray)}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {s : Release} {p : Rep} {dom : ByteArray}
    {t : _root_.Flock.Firewall.Finish.Tab} {xs : Array (Array F128)} {hi : Array ByteArray} {stm : Stmt}
    {c : Array F128} {r0 : ByteArray} {q0 : Nat} (hst : s.sts[p.table]? = some stm) (hlink : s.link = some c)
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hxs : xs = _root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm c))
    (hI : RepInv t dom xs hi p) {v : Place × Nat × Rep}
    (hv : _root_.Flock.Firewall.Frames.placeOf salts l0 s p dom = .ok v) :
    ∃ sched, schedSoFar t p.rep dom xs hi p.coins = .ok sched := by
  obtain ⟨hal, -, hop⟩ := hI
  by_cases hlt : p.coins.size < p.alg.size
  · exact ⟨p.alg, schedSoFar_alg t dom xs hi p hal hlt⟩
  · rw [schedSoFar_open t dom xs hi p hal hlt]
    rcases hop with ⟨hn, ho, hle⟩ | ⟨hle, ho, hS⟩ | ⟨hle, ho, hn, x, hx⟩
    · have hF : openFed t dom xs hi p = (_root_.Flock.Firewall.Drive.openStepM t p.rep dom xs hi
          (p.coins.extract 0 p.alg.size), #[]) := by
        unfold openFed
        rw [Array.extract_empty_of_size_le_start hle]
        exact feed_nil _ _
      rw [placeOf_none salts l0 s p dom hlt ho hn hst hlink (hsalt (by omega) ho hn) htab, ← hxs] at hv
      rw [hF]
      revert hv
      generalize _root_.Flock.Firewall.Drive.openStepM t p.rep dom xs hi (p.coins.extract 0 p.alg.size) = S
      intro hv
      cases S with
      | ask c' n k' => exact ⟨_, rfl⟩
      | done x' => exact ⟨_, rfl⟩
      | fail e => simp [stepPlace] at hv
    · rw [placeOf_some salts l0 s p dom hlt ho hS] at hv
      generalize openFed t dom xs hi p = F at hv ⊢
      obtain ⟨S, rs⟩ := F
      cases S with
      | ask c' n k' => exact ⟨_, rfl⟩
      | done x' => exact ⟨_, rfl⟩
      | fail e => simp [stepPlace] at hv
    · generalize openFed t dom xs hi p = F at hx ⊢
      obtain ⟨S, rs⟩ := F
      dsimp only at hx
      subst hx
      exact ⟨_, rfl⟩

/-- **The round's own rep one round on, where the next round is another stream's**: past its stream's first round,
at a round of table `i`'s rep `r` that releases and whose coins the rep's frame took, where the history one message on
ends with the coins' bytes, the frame the session's next request finds is the rep's, on the session's entry, table
and level 0 as they stay, holding the history's words for each of its stream's rounds where the next round is the
next stream's first (`roundRun_last`, `count_last`); `StreamAt` asks nothing else there. -/
theorem repJ_next {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {st : State} {pl : Placed} {stream : String} {i r : Nat} {dom : ByteArray} {v : Claims.Msg} {e : Env}
    {σ σ' : LoopAt} {e0 : Claims.Msg × Claims.Msg} {h1 : List (Claims.Msg × Claims.Msg)} {q₀ j₀ : Nat}
    (hmem : (i, r, stream) ∈ spec.repStreams.toList) (hh : M.hist σ.c = e0 :: h1) (hq : h1.length = q₀ + 1)
    (hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, j₀ + 1))
    (hne : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₀ + 1)).1 ≠ 2 * i + r)
    (hr : RoundRun sid st pl stream (Loop.loopNr M.x h1.length) i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hreq : st.request pl (.round stream (Loop.loopNr M.x h1.length)) = none) {q : Rep} {pl' : Place} {f : Bool}
    {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ (Loop.loopNr M.x h1.length) =
      .ok (q, pl', f, nd))
    {o : ByteArray} (ho : e.round = .ok o) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (Loop.loopNr M.x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {m' : Claims.Msg} (hh' : M.hist σ'.c = M.hist σ.c ++ [(m', a.data.toList)]) (hS : Stays sid σ σ')
    (hJ : RepJ M sid spec sts i r σ) : RepJ M sid spec sts i r σ' := by
  obtain ⟨p, rel, name, l0, hfa, hss, hn, hl0, hA⟩ := hJ
  have hss' : σ.R.sessions.find? (·.sid == sid) = some rel := hss
  have hlen : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := congrArg Prod.fst hk
    unfold nextAt at this
    rwa [hlen] at this
  have hk2 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = j₀ + 1 := by
    have := congrArg Prod.snd hk
    unfold nextAt at this
    rwa [hlen] at this
  obtain ⟨j, c, stm, t, xs, hi, pre, ev, hj, hlink, hstm, htab, -, hxs, hI, -, -⟩ := hA.1 (by rw [hlen]; exact hk1)
  rw [hlen, hk2] at hj
  obtain rfl : j₀ = j := by omega
  rw [hlen, hh, List.tail_cons] at hI
  have hr2 := rep_lt_two hmem
  have hmod : (2 * i + r) % 2 = r := by omega
  have hdiv : (2 * i + r) / 2 = i := by omega
  rw [hmod] at hI
  rw [hdiv] at hstm
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hf)
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm
  have hsz : p.coins.size = j₀ + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  have hne' : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  obtain ⟨r0, q0, hi', hI', hsalt⟩ := sameInv_salts hI hfa hrp hss' hn hl0
  obtain ⟨w, hw⟩ := roundPlace_placeOf hf hrp (by rw [hne', Bool.false_and]) hss' hn hl0
  obtain ⟨sched, hsc⟩ := sched_of_placeOf hst' hlink hsalt htab hxs hI'.inv hw
  have ho' := (hr.round hreq).symm.trans ho
  obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
  obtain ⟨p₃, hat, -, hcw, -⟩ := roundRun_last M.x e0 h1 v hq hk2 hr hreq hrp ho' hput ha hb hs hco hfa hss' hn
    hl0 hst' hlink hsalt htab hxs hI' hsc
  refine ⟨p₃, rel, name, l0, hat, ?_, hn, ?_, ?_⟩
  · rw [sessAt_stay (hS.2 name l0 hl0) (by rw [hss]; rfl)]
    exact hss
  · rw [hS.1 name]
    exact hl0
  · have hl' : (M.hist σ'.c).length - 2 = q₀ + 1 := by
      rw [hh', hh, List.length_append, List.length_cons, hq, List.length_singleton]
      omega
    unfold StreamAt
    rw [hl']
    refine ⟨fun he => absurd he hne, fun he _ => ?_⟩
    have hnext : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (q₀ + 1)).1 =
        (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 + 1 := by rw [he, hk1]
    have hcnt := count_last hk2 hnext
    rw [he, Nat.add_sub_cancel] at hcnt
    rw [hcw, hcnt, hh', hh, List.cons_append, List.tail_cons]
    simp only [List.map_append, List.map_cons, List.map_nil]

/-- **In its algebraic phase a round leaves its rep's opening unbegun**: past the stream's first round, where the
found frame has fewer coins than its algebraic phase's rounds and has not begun its opening, `placeOf` places the round
on the frame as it is (`Loop.placeOf_alg`), and the frame the session's next request finds has not begun it either. -/
theorem roundRun_alg {sid : Nat} {st : State} {pl : Placed} {stream : String} {n i r : Nat}
    {dom content : ByteArray} {e : Env} {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {R₃ : Sessions}
    (hr : RoundRun sid st pl stream n i r dom content e R L0 R₃) (hreq : st.request pl (.round stream n) = none)
    {q p₁ : Rep} {pl' : Place} {f : Bool} {nd : Option String} {o : ByteArray}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl', f, nd))
    (ho : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).out = .ok o)
    (hput : (_root_.Flock.Firewall.Frames.roundStep R L0 sid i r dom content n).put = some p₁) {a : ByteArray}
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) {pre : List _root_.Flock.Firewall.Frame.Req}
    {ev : List (Nat × ByteArray × Place × Array F128)} (hR : RepRun pre ev q) {p : Rep}
    (hfa : frameAt R sid i r = some p) (hl : f = false) {s : Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} (hss : R.sessions.find? (·.sid == sid) = some s)
    (hn : s.spec.tables[i]? = some name) (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hlt : p.coins.size < p.alg.size) (hto : p.toOpening = false) (hop : p.opening = none)
    (hod : p.opened = false) :
    ∃ p₃, frameAt R₃ sid i r = some p₃ ∧ p₃.opening = none ∧ p₃.opened = false := by
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hfr : (frameAt R sid i r).isSome := by
    rw [hfa]
    rfl
  obtain ⟨hq, -⟩ := roundPlace_later_of hf hrp hl hss hn hl0 (Loop.placeOf_alg R.salts l0 s p _ hlt)
  obtain ⟨st₂, -, hat, -⟩ := roundRun_next hr hreq hfr hrp ho hput ha hb hs hc hR
  rcases hres : _root_.Flock.Firewall.Frame.step q.cfg q.st
    (.round n (if nd.isSome then ByteArray.empty else content) pl') with ⟨a', st'⟩
  obtain ⟨v, hv⟩ := roundStep_put hrp hres
  rw [hv, Option.some.injEq] at hput
  subst hput
  have hq' : q = p := hq
  subst hq'
  have key : ∀ (p' : Rep) (st'' : _root_.Flock.Firewall.Frame.State) (cs : Array F128), p'.toOpening = false →
      p'.opening = none → (Loop.coinsOf p' st'' cs).opening = none := fun p' st'' cs h1 h2 => by
    simp only [Loop.coinsOf, h1, h2]
  exact ⟨_, hat, key { q with st := st', vals := v } _ _ hto hop, hod⟩

/-- **A frame keeping `SameInv` at a round past its stream's first keeps it at salts the round lemmas read, which are
the rep's in the loop's frames from its opening's first round on**: as `sameInv_salts`, where the frame's salts are
the rep's once its opening has begun, which it has not where its coins are as many as its algebraic phase's rounds
(`StreamAt`'s). -/
theorem sameInv_salts_kept {W Rr : Type} {L : LeanRelease W Rr} {x : Msg} {ρ : Rr} {h : List (Msg × Msg)}
    {cs : List FlockDeploy.Coin} {q₀ j : Nat} {t : _root_.Flock.Firewall.Finish.Tab} {dom' : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {pre : List _root_.Flock.Firewall.Frame.Req}
    {ev : List (Nat × ByteArray × Place × Array F128)} {p : Rep} (hI : SameInv L x ρ h cs q₀ j t dom' xs hi pre ev p)
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r : Nat} {dom content : ByteArray} {n : Nat}
    {q : Rep} {pl : Place} {f : Bool} {nd : Option String} (hfa : frameAt R sid i r = some p)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0)
    (hsa : p.alg.size < p.coins.size → saltsAt R l0.root p.rep = some hi)
    (hord : p.alg.size = p.coins.size → p.opening = none ∧ p.opened = false) :
    ∃ r0 q0 hi', SameInv L x ρ h cs q₀ j t dom' xs hi' pre ev p ∧
      (p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
        R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi')) ∧
      (p.alg.size ≤ p.coins.size → saltsAt R l0.root p.rep = some hi') := by
  by_cases hlt : p.alg.size < p.coins.size
  · obtain ⟨r0, q0, hf⟩ := find?_of_saltsAt (hsa hlt)
    exact ⟨r0, q0, hi, hI, fun _ _ _ => hf, fun _ => hsa hlt⟩
  · obtain ⟨r0, q0, hi', hI', hs⟩ := sameInv_salts hI hfa hrp hss hn hl0
    refine ⟨r0, q0, hi', hI', hs, fun hle => ?_⟩
    obtain ⟨hop, hod⟩ := hord (by omega)
    exact saltsAt_of_find? (hs hle hod hop)

/-- **The `Commit`'s words are as many as the loop requests**, within `Frames.COIN_MAX`. -/
theorem wordsAt_zero_size (x : Msg) {cs : List FlockDeploy.Coin} (h : 0 < cs.length) :
    (Loop.wordsAt x cs 0).size = min (Loop.loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
  cases cs with
  | nil => simp at h
  | cons c cs => simp [Loop.wordsAt, FlockDeploy.Coin.vals]

open Loop LeanRelease in
/-- **Past a rep's first round the loop's frame takes coins only at the rep's placeholder**: where it takes the coins
after `e0 :: h1`, the round after being past its stream's first, the frame state the rep's last round left owes them
(`takeAt_ok`), which it does only where the rep's placeholder is one (`progOf_placeholder`). -/
theorem ofLoop_placeholder (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker} (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    {q j : Nat} (hq : h1.length = q + 1) (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) :
    ∃ c' t dom xs hi, placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1
      (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi) := by
  have hF : (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) = .ok m := by
    unfold LeanRelease.answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [LeanRelease.ofLoop_frameAnswer] at hF
  have e1 : (e0 :: h1).length = q + 2 := by simp only [List.length_cons, hq]
  have hcs : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := by
    rcases q with _ | q'
    · have := Loop.locate_zero (roundsOf spec sts) 0
      rw [hj] at this
      exact absurd this (Nat.succ_ne_zero _)
    · exact ⟨q', rfl⟩
  have hp := Loop.locate_pred (roundsOf spec sts) 0 q' j hj
  rcases LeanRelease.takeAt_ok hF with ⟨hn, -⟩ | ⟨-, -, ha⟩ | ⟨-, hb, -, -⟩
  · rw [e1] at hn
    omega
  · have ho : (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun ho =>
      Loop.answer_owed_none _ ho ha
    obtain ⟨c', t, dom, xs, hi, hph⟩ := Loop.progOf_placeholder x _ hx (by rw [hcs]) ho
    rw [hp] at hph
    exact ⟨c', t, dom, xs, hi, hph⟩
  · exfalso
    rw [e1, show q' + 1 + 2 - 1 = q' + 2 by omega, LeanRelease.repOf_add_two hx, LeanRelease.repOf_add_two hx,
      hp] at hb
    simp at hb

open Loop LeanRelease in
/-- **The configuration at the loop's own frames is of the placeholder's shape**: past `Hello` and the `Commit`, at
the rep of the round after `h`, the tape's samples and the placeholder's zeros configure the rep alike
(`cfgAlike_ofBytes`). -/
theorem ofLoop_cfgAlike (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (h : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) {q : Nat} (hq : h.length = q + 2) {link : Array F128}
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 link = .ok (c', t, dom, xs, hi)) :
    CfgAlike ((ofLoop T L₀ real).cfg x ρ h) c' := by
  obtain ⟨lay, mask, hst, hc', -, -⟩ := Loop.placeholder_ok hph
  have hrk : (ofLoop T L₀ real).rep x h.length = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 := by
    show repOf x h.length = _
    simp only [repOf, hx, hq, Nat.add_sub_cancel]
  have hso : startOf x (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .ok (lay, mask) := by
    unfold startOf
    rw [hx]
    exact hst
  unfold LeanRelease.cfg LeanRelease.drawsAt
  rw [hrk, hc']
  have e1 : (ofLoop T L₀ real).lay x (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = lay := by
    show ((startOf x (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1).toOption.map (·.1)).getD default = lay
    rw [hso]
    rfl
  have e2 : (ofLoop T L₀ real).slot x (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = some mask := by
    show (startOf x (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1).toOption.map (·.2) = some mask
    rw [hso]
    rfl
  rw [e1, e2]
  exact Loop.cfgAlike_ofBytes lay _ _ (some mask)

open Loop LeanRelease in
/-- **The round's coins taken, and the round's own rep one round on where the next round is the same stream's**: the
model's coin step takes the round's coins (`roundRun_same`, `ofLoop_fed_snd`), and as `repJ_next`, where the round
after it is the rep's next (`hsame`), at the loop's own frames (`hrel`), the session's entry being the setup's with
the `Commit`'s words as its link (`hK2`), the model's own table at the rep the one the rep's level 0 gives (`hK3`; it
is of the placeholder's shape where `real` answers the loop's own, `hR`, within the bounds `hS` and `hP` give), and the
round within the rep's schedule, not the inner proof's (`hin`): the frame the session's next request finds keeps
`SameInv` one round more of the rep (`roundRun_same`), the placeholder, its configuration and its schedule being the
model's (`ofLoop_placeholder`, `ofLoop_cfgAlike`, `Sched.schedSoFar_alike`) and the round's count the plan's
(`Loop.loopNr_succ`, `Count.plan_sched`), at the rep's salts once its opening has begun (`sameInv_salts_kept`), not
begun where its coins are as many as its algebraic phase's rounds (`roundRun_alg`). -/
theorem repJ_same (T : ZKMal.CoinLayer.TableData)
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
    {σ σ' : LoopAt} {e0 : Claims.Msg × Claims.Msg} {h1 : List (Claims.Msg × Claims.Msg)} {q₀ j₀ : Nat}
    (hmem : (i, r, stream) ∈ spec.repStreams.toList) (hh : M.hist σ.c = e0 :: h1) (hq : h1.length = q₀ + 1)
    (hk : nextAt spec sts (M.hist σ.c) = (2 * i + r, j₀ + 1)) {m₀ : Masker}
    (hA₀ : M.L.answered M.x M.ρ (M.hist σ.c) = .ok m₀) (hlast : (M.hist σ.c).length < M.L.last M.x)
    (hK2 : ∀ rel : Release, sessAt σ.R sid = some rel → rel.spec = spec ∧ rel.sts = sts ∧
      rel.link = some (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0))
    (hK3 : ∀ (rel : Release) (name : String) (l0 : _root_.Flock.Firewall.Level0.Level0) (stm : Stmt)
      (t : _root_.Flock.Firewall.Finish.Tab), sessAt σ.R sid = some rel → rel.spec.tables[i]? = some name →
      _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 → rel.sts[i]? = some stm →
      _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t →
      ∃ hr, real M.x M.w M.ρ (2 * i + r) = .ok (t, hr))
    (hin : ∀ c' t' dom' xs' hi' sched', placeholder spec sts (2 * i + r)
      (wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2) 0) = .ok (c', t', dom', xs', hi') →
      schedSoFar t' r dom' xs' hi' ((Array.range (j₀ + 1)).map fun k =>
        wordsAt M.x ((M.hist σ.c).tail.map fun y => ZKMal.CoinLayer.rd y.2)
          ((M.hist σ.c).length - 2 - (j₀ + 1) + k + 1)) = .ok sched' → j₀ + 1 < sched'.size)
    (hr : RoundRun sid st pl stream (loopNr M.x h1.length) i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R)
    (hreq : st.request pl (.round stream (loopNr M.x h1.length)) = none) {q : Rep} {pl' : Place} {f : Bool}
    {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom ⟨v.toArray⟩ (loopNr M.x h1.length) =
      .ok (q, pl', f, nd))
    {o : ByteArray} (ho : e.round = .ok o) {a : ByteArray} (ha : e.ans = some a) (hb : byte? a 0 = some 2)
    (hs : (st.answer (.round stream (loopNr M.x h1.length)) a).1.stop = none) (hco : e.coins = .ok ())
    {m' : Claims.Msg} (hh' : M.hist σ'.c = M.hist σ.c ++ [(m', a.data.toList)]) (hSt : Stays sid σ σ')
    (hJ : RepJ M sid spec sts i r σ) : M.L.fed M.x M.ρ (M.hist σ'.c) = true ∧
      ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).1 = 2 * i + r → RepJ M sid spec sts i r σ') := by
  obtain ⟨p, rel, name, l0, hfa, hss, hn, hl0, hA⟩ := hJ
  have hss' : σ.R.sessions.find? (·.sid == sid) = some rel := hss
  obtain ⟨hsp, hsts, hlk⟩ := hK2 rel hss
  have hlen : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := congrArg Prod.fst hk
    unfold nextAt at this
    rwa [hlen] at this
  have hk2 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = j₀ + 1 := by
    have := congrArg Prod.snd hk
    unfold nextAt at this
    rwa [hlen] at this
  rw [hlen, hh, List.tail_cons] at hin
  rw [hh, List.tail_cons] at hlk
  rw [hrel, hh] at hA₀ hlast
  obtain ⟨j, c, stm, t, xs, hi, pre, ev, hj, hlink, hstm, htab, hcap, hxs, hI, hsa, hord⟩ :=
    hA.1 (by rw [hlen]; exact hk1)
  rw [hlen, hk2] at hj
  obtain rfl : j₀ = j := by omega
  rw [hlen, hh, List.tail_cons] at hI
  have hr2 := rep_lt_two hmem
  have hmod : (2 * i + r) % 2 = r := by omega
  have hdiv : (2 * i + r) / 2 = i := by omega
  rw [hmod] at hI
  rw [hmod] at hsa
  have hstm' : rel.sts[i]? = some stm := by
    have h' := hstm
    rw [hdiv] at h'
    exact h'
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hf)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hf)
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm'
  have hsz : p.coins.size = j₀ + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  have hne' : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  rw [← hpr] at hsa
  obtain ⟨r0, q0, his, hI', hsalt, hsa'⟩ := sameInv_salts_kept hI hfa hrp hss' hn hl0 hsa hord
  obtain ⟨hl, hRq, -⟩ := sameInv_place hI' hfa hrp
  obtain ⟨w, hw⟩ := roundPlace_placeOf hf hrp (by rw [hne', Bool.false_and]) hss' hn hl0
  obtain ⟨sched, hsc⟩ := sched_of_placeOf hst' hlink hsalt htab hxs hI'.inv hw
  have hq2 : (e0 :: h1).length = q₀ + 2 := by simp only [List.length_cons, hq]
  obtain ⟨c', t', dom', xs', hip, hph⟩ := ofLoop_placeholder T L₀ real M.x M.ρ e0 h1 hA₀ hx hq hk2
  have hph' : placeholder spec sts (2 * i + r) (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0) =
      .ok (c', t', dom', xs', hip) := by
    rw [← hk1]
    exact hph
  have hc := ofLoop_cfgAlike T L₀ real M.x M.ρ (e0 :: h1) hx hq2 hph
  obtain ⟨lay, mask, -, -, hdm, hxe⟩ := Loop.placeholder_ok hph'
  have hname : spec.tables[i]! = name := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hn
    rw [← hsp, getElem!_pos rel.spec.tables i hlt, he]
  have hstmb : sts[i]! = stm := by
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hstm'
    rw [← hsts, getElem!_pos rel.sts i hlt, he]
  have hdom : (rel.spec.domain name r).toUTF8 = dom' := by rw [hdm, hdiv, hmod, hname, hsp]
  have hcw : c = wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0 := Option.some.inj (hlink.symm.trans hlk)
  have hxx : xs = xs' := by rw [hxs, hxe, hdiv, hstmb, hcw]
  have hkl : 2 * i + r < spec.repStreams.size := rep_lt_size hmem
  have hls := wordsAt_zero_size M.x (cs := h1.map fun y => ZKMal.CoinLayer.rd y.2)
    (by rw [List.length_map, hq]; omega)
  obtain ⟨hdm', hxs2⟩ := hS (2 * i + r) _ c' t' dom' xs' hip hkl (by rw [hls]; exact Nat.min_le_right _ _) hph'
  obtain ⟨hmp, hmp0, hpc⟩ := hP (2 * i + r) hkl
  have hnr0 : loopNr M.x 0 = spec.points * spec.mPts := by
    show (Loop.reqOf M.x 0).2 = _
    rw [Loop.reqOf_eq hx]
    rfl
  have hpts : (wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) 0).size / sts[(2 * i + r) / 2]!.mPts =
      spec.points := by
    rw [hls, hnr0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  obtain ⟨hr', hre⟩ := hK3 rel name l0 stm t hss hn hl0 hstm' htab
  have hT := Count.real_tabAlike hR hph' hre
  obtain ⟨hdg, hmb, hnb, hlq, hpb, hdb⟩ := Count.real_bounds hph' hT
  have hcn : (_root_.Flock.Firewall.Finish.flat t'.cm0.cap).size < 2 ^ 64 := by
    rw [← hT.2.2.2.2.1]
    exact hT.2.2.2.2.2
  have hsc2 : schedSoFar t r dom' xs' his p.coins = .ok sched := by
    rw [← hpr, ← hdom, ← hxx]
    exact hsc
  have hal := Sched.schedSoFar_alike t t' r dom' xs' his hip p.coins hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2
    hdm' hdg hmb hxs2 hnb hlq hpb hdb
  rw [hsc2] at hal
  cases hs' : schedSoFar t' r dom' xs' hip p.coins with
  | error e' =>
    rw [hs'] at hal
    exact (hal : False).elim
  | ok sched' =>
  rw [hs'] at hal
  have hs0 : schedSoFar t' r dom' xs' hip ((Array.range (j₀ + 1)).map fun k =>
      wordsAt M.x (h1.map fun y => ZKMal.CoinLayer.rd y.2) (q₀ - (j₀ + 1) + k + 1)) = .ok sched' := by
    rw [← hI.coins]
    exact hs'
  have hlt' := hin c' t' dom' xs' hip sched' hph' hs0
  have hlt : j₀ + 1 < sched.size := by
    rw [Loop.roundsAt_size hal]
    exact hlt'
  have hqs : q₀ < (roundsOf spec sts).sum :=
    LeanRelease.loop_round_in_plan T (withFrames T L₀ real) M.x M.ρ (e0 :: h1) hA₀ (Nat.ne_of_lt hlast) hx hq2
  have hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀ = (2 * i + r, j₀ + 1) := Prod.ext hk1 hk2
  have hn' : loopNr M.x h1.length = (sched'.getD (j₀ + 1) (.empty, 0)).2 := by
    rw [hq]
    exact (Loop.loopNr_succ hx hki hqs).trans (Count.plan_sched spec sts (2 * i + r) (j₀ + 1) _ c' t' dom' xs' hip
      p.coins sched' hph' (by rw [hmod]; exact hs') hlt' hpts hdm' hcn hxs2)
  have ho' := (hr.round hreq).symm.trans ho
  obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
  rw [hrel] at hI'
  obtain ⟨hfd, hsm⟩ := roundRun_same T L₀ real M.x M.ρ e0 h1 hx hq hk2 hph hc
    (by rw [hk1, hmod]; exact hs0) hn' hr hreq hrp ho' hput ha hb hs hco hfa hss' hn hl0 hst' hlink hsalt htab hxs
    hI' hsc hal hlt hA₀
  have hsnd : (e0 :: (h1 ++ [(v, a.data.toList)])).map Prod.snd = (M.hist σ'.c).map Prod.snd := by
    rw [hh', hh]
    simp
  refine ⟨by rw [hrel]; exact (ofLoop_fed_snd T L₀ real M.x M.ρ hsnd).symm.trans hfd, fun hsame => ?_⟩
  have hj2 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q₀ + 1)).2 = j₀ + 1 + 1 := by
    rw [Loop.locate_succ _ _ _ (hsame.trans hk1.symm), hk2]
  obtain ⟨-, p₃, hat, hI3⟩ := hsm _ (by rw [hq]; exact hj2)
  have hI4 := sameInv_snd hsnd hI3
  have hcs : ((h1 ++ [(v, a.data.toList)]).map fun y => ZKMal.CoinLayer.rd y.2) =
      (M.hist σ'.c).tail.map fun y => ZKMal.CoinLayer.rd y.2 := by
    rw [hh', hh, List.cons_append, List.tail_cons]
    simp only [List.map_append, List.map_cons, List.map_nil]
  rw [hcs, ← hrel] at hI4
  have halg : p₃.alg = p.alg := Except.ok.inj (hI4.inv.alg.symm.trans hI'.inv.alg)
  have hc3 : p₃.coins.size = j₀ + 1 + 1 := by rw [hI4.coins, Array.size_map, Array.size_range]
  have hl' : (M.hist σ'.c).length - 2 = q₀ + 1 := by
    rw [hh', hh, List.length_append, List.length_cons, hq, List.length_singleton]
    omega
  refine ⟨p₃, rel, name, l0, hat, ?_, hn, ?_, ?_⟩
  · rw [sessAt_stay (hSt.2 name l0 hl0) (by rw [hss]; rfl)]
    exact hss
  · rw [hSt.1 name]
    exact hl0
  · unfold StreamAt
    rw [hl']
    refine ⟨fun _ => ?_, fun he _ => absurd (hsame.symm.trans he) (by omega)⟩
    rw [hmod]
    refine ⟨j₀ + 1, c, stm, t, xs, his, pre, _, hj2, hlink, hstm, htab, hcap, hxs, hI4, fun hlt3 => ?_,
      fun heq3 => ?_⟩
    · rw [halg, hc3] at hlt3
      have h1 := hsa' (by omega)
      rw [hpr] at h1
      rw [saltsAt_stay (hSt.2 name l0 hl0) r (by rw [h1]; rfl), h1]
    · rw [halg, hc3] at heq3
      rcases hI'.inv.opening with ⟨hn0, ho0, -⟩ | ⟨hle, -⟩ | ⟨hle, -⟩
      · obtain ⟨p₅, hat5, hop5, hod5⟩ := roundRun_alg hr hreq hrp ho' hput ha hb hs hco hRq hfa hl hss' hn hl0
          (by omega) hI'.inv.toOpening hn0 ho0
        obtain rfl := Option.some.inj (hat5.symm.trans hat)
        exact ⟨hop5, hod5⟩
      · omega
      · omega

end Claims.ZKServer.Relay
