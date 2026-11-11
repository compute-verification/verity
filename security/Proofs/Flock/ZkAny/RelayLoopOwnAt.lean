module

public import Proofs.Flock.ZkAny.RelayLoopInner
public import Proofs.Flock.ZkAny.RelayLoopOwn
public import Proofs.Flock.ZkAny.RelayLoopFacts
public import Proofs.Flock.SessionLoop.LiveInner

/-!
# The frame's own inner round is the model's scheduled round

Past a rep's schedule so far `placeOf` places the frame's next round at the inner proof's own, and `roundPlace` readies
it at `Frame.innerSched` on the frame's configuration, masking state and the rounds the loop's finish computed. At the
deployed model, whose release has the session loop's own rounds of the inner proof (`LeanRelease.withInner` at the
finish's rounds `fin`), the model's scheduled round there is `Loop.innerOf`'s, `Frame.innerSched` at the model's
configuration and `ε` and `β` the first words of the coins of the inner proof's rounds 0 and 2. They are one round:

* `innerSched_fin_congr`: the frame's own inner round reads the finish's rounds only at round 0's (at round 0) and
  round 2's (at round 2). `roundPlace_own_bounds`: `roundPlace` readies an inner round only within the inner proof and
  where the loop's finish computed it.
* `ofLoop_answered_own`: past the rep's schedule on its own table and salts the model's masking state has begun no
  inner round where the schedule ends at the round, and is at the inner proof's round past the last it took coins at
  otherwise, `ε` and `β` the first words of the coins of its rounds 0 and 2 (`repState_own_inner`, `repState_own_eb`).
* `roundPlace_scheduled_own`: at the deployed model, where `placeOf` places the frame's round at the inner proof's own,
  the frame at the model's configuration and masking state but for `stopped`, and each round of the inner proof the
  loop's finish computed so far the model's finish's, the round's place is the model's scheduled round
  (`Loop.innerOf_eq`, `scheduled_innerAt`, `innerSched_congr`, `innerSched_fin_congr`).
* **`placeSched_own`**: so at a round of a stream seen, where `LoopFacts` and `LoopJ` hold, the model's tables and
  salts are the loop's (`RealAt`) and the rep's rounds of the inner proof the model's finish's (`InnerSrc`, the
  model's `fin` the release's), the round's place is the model's scheduled round wherever `placeOf` places it at the
  inner proof's own: `placeFacts_seen`'s `hown`.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed reqAt)
open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep Release)
open _root_.Flock.Firewall.Frame (Cfg Masker Place INNER_ROUNDS INNER_BETA)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- An extract's `k`-th entry is the array's `s + k`-th, within the extract. -/
theorem extract_getD {α : Type} {a : Array α} {s e k : Nat} (d : α) (h : s + k < e) (he : e ≤ a.size) :
    (a.extract s e).getD k d = a.getD (s + k) d := by
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_extract, if_pos (by omega)]

/-- A byte array is its bytes' list read back. -/
theorem bytes_toList_toArray (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := by
  rfl

/-- Where each entry `a` holds is `b`'s, `a` and `b` read alike within `a`. -/
theorem getD_of_agree {a b : Array ByteArray} (h : ∀ (k : Nat) (x : ByteArray), a[k]? = some x → b[k]? = some x)
    {k : Nat}
    (hk : k < a.size) (d : ByteArray) : a.getD k d = b.getD k d := by
  have ha : a[k]? = some a[k] := Array.getElem?_eq_getElem hk
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, h k _ ha, ha]

open _root_.Flock.Firewall.Frame (innerSched ownRound) in
/-- **The frame's own inner round reads the finish's rounds only at round 0's and round 2's**, the first at round 0
and the second at round 2: on rounds alike there, it is one round. -/
theorem innerSched_fin_congr (c : Cfg) (s : Masker) {fin fin' : Array ByteArray} (n : Nat) :
    ∀ {k : Nat}, k < INNER_ROUNDS → (k = 0 → fin.getD 0 .empty = fin'.getD 0 .empty) →
      (k = 2 → fin.getD 2 .empty = fin'.getD 2 .empty) → innerSched c s fin n k = innerSched c s fin' n k
  | 0, _, h0, _ => by simp only [innerSched, ownRound, h0 rfl]
  | 1, _, _, _ => by simp only [innerSched, ownRound]
  | 2, _, _, h2 => by simp only [innerSched, ownRound, h2 rfl]
  | 3, _, _, _ => by simp only [innerSched, ownRound]
  | k + 4, h, _, _ => absurd h (by unfold INNER_ROUNDS; omega)

set_option maxHeartbeats 1000000 in
/-- **At the inner proof's own place `roundPlace` readies only a round the loop's finish computed**: where `placeOf`
places the found frame's next round at the inner proof's own, the readied frame is at an inner round within the inner
proof that the rounds the loop's finish computed so far hold; at any other it refuses. -/
theorem roundPlace_own_bounds {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid i r j : Nat}
    {dom content : ByteArray} {n : Nat} {p q : Rep} {pl : Place} {f : Bool} {nd : Option String}
    (hf : R.find sid i r = some (j, p))
    (h : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pl, f, nd))
    (hl : f = false) {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hs : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) {k : Nat} {q' : Rep}
    (hv : _root_.Flock.Firewall.Frames.placeOf R.salts l0 s p (s.spec.domain name r).toUTF8 = .ok (.own, k, q')) :
    q.st.m.inner.getD 0 < INNER_ROUNDS ∧ q.st.m.inner.getD 0 < (R.innerOf l0.root r).size := by
  have hfst : (p.coins.isEmpty && !p.st.begun) = false := (roundPlace_first hf h).symm.trans hl
  unfold _root_.Flock.Firewall.Frames.roundPlace at h
  simp only [hf, bind, Except.bind, pure, Except.pure] at h
  repeat' (first
    | simp only [hs, hn, hl0, hfst, hv, Bool.false_eq_true, ↓reduceIte] at h
    | split at h)
  all_goals first
    | (cases h; done)
    | exact absurd (by assumption) throw_ne_ok
    | (cases h; exact ⟨by omega, by omega⟩)

open Loop LeanRelease in
/-- **The model's masking state past a rep's schedule**: where the frame's coin step takes the coins after `e0 :: h1`
at `m`, before the rank stop (`answeredR`), the next round is rep stream `k`'s round `j + 1`, at a setup passing
`sizesOk`, and the schedule so far on its own table and salts (`rsched`, `Count.RealTabs`) ends at or before the
round: where it ends at the round, `m` has begun no inner round (`repState_inner`); before it, `m` is at the inner
proof's round `j - rsched.size + 1`, within the inner proof (`repState_own_inner`; past its last round the rep's state
owes no coins, `repState_past`), its `ε` the first word of the coins of the inner proof's round 0 and, past its round
2, its `β` the first word of that round's (`repState_own_eb`). -/
theorem ofLoop_answered_own (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker}
    (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: h1) = .ok m)
    {cs : List FlockDeploy.Coin} (hcs : h1.map (fun p => ZKMal.CoinLayer.rd p.2) = cs)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q k j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hz : _root_.Flock.Firewall.Session.sizesOk spec sts = true) (hR : Count.RealTabs sts (real x w ρ))
    (hq : h1.length = q + 1) (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1))
    (hk : k < spec.repStreams.size)
    {tr : _root_.Flock.Firewall.Finish.Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr))
    {rsched : Array (ByteArray × Nat)}
    (hrs0 : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (wordsAt x cs 0))) hr ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok rsched)
    (hle : rsched.size ≤ j + 1) :
    (rsched.size = j + 1 → m.inner = none) ∧
      (rsched.size < j + 1 → j - rsched.size < INNER_ROUNDS ∧ m.inner = some (j - rsched.size + 1) ∧
        m.eps = (((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).getD rsched.size #[])[0]! ∧
        (INNER_BETA ≤ j - rsched.size → m.beta =
          (((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).getD
            (rsched.size + INNER_BETA) #[])[0]!)) := by
  have hq1 : cs.length = q + 1 := by rw [← hcs, List.length_map, hq]
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  have hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1 := by rw [hki]
  have hjq : j + 1 ≤ q := by
    have := locate_snd_le (roundsOf spec sts) 0 q
    rw [hj] at this
    exact this
  obtain ⟨m0, r0⟩ := e0
  have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x cs := by
    rw [← hcs]
    exact loop_coinsRead loopNr T (withFrames T L₀ real) x m0 r0 h1
  have hF : (ofLoop T L₀ real).frameAnswerR x ρ ((m0, r0) :: h1) = .ok m := by
    unfold answeredR at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswerR_same T L₀ real x ρ (m0, r0) h1 hx hq hj, hcs, hcr, readCoins_eq x cs hq1] at hF
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := by
    rcases q with _ | q'
    · have := locate_zero (roundsOf spec sts) 0
      rw [hki] at this
      exact absurd this (Nat.succ_ne_zero _)
    · exact ⟨q', rfl⟩
  have hp := locate_pred (roundsOf spec sts) 0 q' j hj
  obtain ⟨c', t, dom, xs, hi, hph⟩ :=
    progOf_placeholder x cs hx (p := q') (by rw [hq1]) (fun ho => answer_owed_none _ ho hF)
  rw [hp, hki] at hph
  have hph' : placeholder spec sts k (wordsAt x cs 0) = .ok (c', t, dom, xs, hi) := hph
  rw [progOf_at x cs hx hq1 hj (by rw [hk1]; exact hph'), hj, hk1] at hF
  obtain ⟨lay, mask, -, -, hdm, hxe⟩ := placeholder_ok hph'
  obtain ⟨-, hd, -, -⟩ := sizesOk_at hz hk
  have hls : (wordsAt x cs 0).size = min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    rw [← hcs]
    cases h1 with
    | nil => simp at hq
    | cons p h2 => simp [wordsAt, FlockDeploy.Coin.vals]
  have hdom : dom.size < 2 ^ 64 := by rw [hdm]; exact hd
  have hxs : xs.size + 2 < 2 ^ 64 := by
    rw [hxe, extras_size]
    have := Nat.mul_le_mul_left sts[k / 2]!.regions.size
      ((linkPoints_size_le sts[k / 2]! (wordsAt x cs 0)).trans (hls.trans_le (Nat.min_le_right _ _)))
    have h2 := (sizesOk_at hz hk).2.2.1
    omega
  have hT := Count.real_tabAlike hR hph' hreal
  obtain ⟨hdg, hmb, hn, hlq, hpd, hdd⟩ := Count.real_bounds hph' hT
  have hR' := Sched.schedSoFar_alike tr t (k % 2) dom xs hr hi
    ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1))
    hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2 hdom hdg hmb hxs hn hlq hpd hdd
  rw [← hdm, ← hxe] at hrs0
  rw [hrs0] at hR'
  cases hs' : schedSoFar t (k % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1)) with
  | error e => rw [hs'] at hR'; exact (hR' : False).elim
  | ok sched' =>
  rw [hs'] at hR'
  have hsz := roundsAt_size hR'
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨sched, hs⟩ := schedSoFar_pop_ok hs'
  have hss : sched' = sched := schedSoFar_pop_eq (j := j) hs hs' (by simp) (by omega)
  rw [hss] at hsz hR'
  have hg : ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1)).getD j #[] =
      wordsAt x cs (q' + 1) := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by simp), Option.getD_some, Array.getElem_map,
      Array.getElem_range]
    congr 1
    omega
  refine ⟨fun hS => ?_, fun hlt => ?_⟩
  · exact Sched.answer_inner (repState_inner c' t _ dom xs hi _ _ _ j hal hs (by omega)
      fun i hi' => roundsAt_framed hR' (by omega)) hF
  · obtain ⟨i, rfl⟩ : ∃ i, j = sched.size + i := ⟨j - sched.size, by omega⟩
    rw [show sched.size + i - rsched.size = i by omega, hsz]
    rcases Nat.lt_or_ge i INNER_ROUNDS with h4 | h4
    · obtain ⟨he, hb⟩ := repState_own_eb c' _ t (k % 2) dom xs hi _ _ _ hal hs h4 hF
      refine ⟨h4, repState_own_inner c' _ t (k % 2) dom xs hi _ _ _ hal hs h4 hF, ?_, fun hi2 => ?_⟩
      · rw [he]
        by_cases hi0 : i = 0
        · subst hi0
          rw [if_pos rfl, ← hg, Nat.add_zero sched.size]
        · rw [if_neg hi0]
      · rw [hb hi2]
        by_cases hi2' : i = INNER_BETA
        · subst hi2'
          rw [if_pos rfl, ← hg]
        · rw [if_neg hi2']
    · exact absurd hF (answer_owed_none _ (repState_past c' t (k % 2) dom xs hi _ _ _ hal hs (by omega)))

open Loop LeanRelease in
/-- **The frame's own inner round is the deployed model's scheduled round**: at the deployed model, whose release
has the session loop's own rounds of the inner proof (`withInner` at the finish's rounds `fin`), after the history
`e0 :: h1` whose next round is table `i`'s rep `r`'s `j + 2`-th, where the rep's frame keeps its schedule so far
(`Loop.RepInv`) on the loop's table, the session's extras at the model's link and the salts the loop holds where
`placeOf` reads them, which are the model's (`real`), holds the history's words for the rep's rounds so far, and
`placeOf` places its round at the inner proof's own (`hown`), where the readied frame is at the model's configuration
and masking state but for `stopped`, the round squeezes the model's coin count and each round of the inner proof the
loop's finish computed so far is the model's finish's (`hfin`): the round's place is the model's scheduled round
there. `roundPlace_own_of` gives the place, `Frame.innerSched` at the frame's state on the inner proof's round the frame
is at, which is within the inner proof and one the loop's finish computed (`roundPlace_own_bounds`);
`ofLoop_answered_own` that round, the model's (`Loop.innerOf_eq`, `scheduled_innerAt`), its `ε` and `β`, the model's
where the round reads them (`innerSched_congr`), and the finish's rounds it reads, the model's (`innerSched_fin_congr`).
-/
theorem roundPlace_scheduled_own
    (L₁ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Rand))
    (src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (fin : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Nonce ×
      (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : String → Option ByteArray)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q₀ j i r : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hz : _root_.Flock.Firewall.Session.sizesOk spec sts = true) (hRT : Count.RealTabs sts (real x w ρ))
    (hq : h1.length = q₀ + 1) (hk : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).1 = 2 * i + r)
    (hr2 : r < 2) (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀).2 = j + 1)
    (hks : 2 * i + r < spec.repStreams.size)
    {R : Sessions} {L0 : _root_.Flock.Firewall.Level0.Slots} {sid : Nat} {dom content : ByteArray} {n : Nat}
    {p q : Rep} {pc : Place} {f : Bool} {nd : Option String} (hfa : frameAt R sid i r = some p)
    (hrp : _root_.Flock.Firewall.Frames.roundPlace R L0 sid i r dom content n = .ok (q, pc, f, nd))
    {s : Release} {name : String} {l0 : _root_.Flock.Firewall.Level0.Level0}
    (hss : R.sessions.find? (·.sid == sid) = some s) (hn : s.spec.tables[i]? = some name)
    (hl0 : _root_.Flock.Firewall.Level0.ofSession L0 sid name = some l0) (hspec : s.spec = spec)
    (hsts : s.sts = sts) (hlink : s.link = some (wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))
    {stm : Stmt} {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {r0 : ByteArray} {q0 : Nat}
    (hst : sts[i]? = some stm)
    (htab : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit stm) l0 = .ok t)
    (hreal : real x w ρ (2 * i + r) = .ok (t, hi))
    (hI : RepInv t (spec.domain name r).toUTF8 (_root_.Flock.Firewall.Drive.extras stm
      (_root_.Flock.Firewall.Drive.linkPoints stm (wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi p)
    (hco : p.coins = (Array.range (j + 1)).map fun k =>
      wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) (q₀ - (j + 1) + k + 1))
    (hsalt : p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
      R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hi))
    {sched : Array (ByteArray × Nat)}
    (hsch : schedSoFar t r (spec.domain name r).toUTF8 (_root_.Flock.Firewall.Drive.extras stm
      (_root_.Flock.Firewall.Drive.linkPoints stm (wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
      p.coins = .ok sched)
    (hown : (placeAt sched (j + 1)).1 = .own)
    (hcq : q.cfg = (ofLoop ZKMal.Masked.dataE (withFiles (withLink (withInner ZKMal.Masked.dataE L₁ real fin) src) src)
      real).cfg x ρ (e0 :: h1))
    {m : Masker}
    (hA : (ofLoop ZKMal.Masked.dataE (withFiles (withLink (withInner ZKMal.Masked.dataE L₁ real fin) src) src)
      real).answeredR x ρ (e0 :: h1) = .ok m)
    (huq : unstop q.st.m = unstop m) (hnr : n = loopNr x (q₀ + 1))
    (hfin : ∀ (k : Nat) (b : ByteArray), (R.innerOf l0.root r)[k]? = some b → (fin x w ρ (e0 :: h1))[k]? = some b) :
    pc = .sched ⟨((ofLoop ZKMal.Masked.dataE (withFiles (withLink (withInner ZKMal.Masked.dataE L₁ real fin) src) src)
      real).scheduled x w ρ (e0 :: h1) vals).toArray⟩ := by
  obtain ⟨_, hfd⟩ := find_of_frameAt hfa
  obtain ⟨-, htb, hrr⟩ : p.sid = sid ∧ p.table = i ∧ p.rep = r := by
    have h := find_key hfd
    simp only [keyOf, Prod.mk.injEq] at h
    exact h
  have hsz : p.coins.size = j + 1 := by rw [hco, Array.size_map, Array.size_range]
  have hne : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  obtain ⟨hf, -, -, -⟩ := roundPlace_next hfa hne hrp
  obtain ⟨q', hv, -⟩ := Loop.placeOf_sched R.salts l0 s p (s.spec.domain name r).toUTF8
    (by rw [hsts, htb]; exact hst) hlink hsalt htab rfl (by rw [hspec]; exact hI) (by rw [hspec, hrr]; exact hsch)
  rw [hsz, hown] at hv
  obtain ⟨hqq, hpc⟩ := roundPlace_own_of hfd hrp hf hss hn hl0 hv
  rw [← hqq] at hpc
  have hle : sched.size ≤ j + 1 := by
    rcases hsj : sched[j + 1]? with _ | ⟨c0, n0⟩
    · exact Array.getElem?_eq_none_iff.mp hsj
    · simp [placeAt, hsj] at hown
  have h2r : (2 * i + r) % 2 = r := by omega
  have h2i : (2 * i + r) / 2 = i := by omega
  have htn : spec.tables[i]! = name := by
    have hn' : spec.tables[i]? = some name := by
      rw [← hspec]
      exact hn
    obtain ⟨hl, he⟩ := Array.getElem?_eq_some_iff.mp hn'
    rw [getElem!_pos spec.tables i hl, he]
  have hsti : sts[i]! = stm := by
    obtain ⟨hl, he⟩ := Array.getElem?_eq_some_iff.mp hst
    rw [getElem!_pos sts i hl, he]
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q₀ = (2 * i + r, j + 1) := Prod.ext hk hj
  have hcs : (h1.map fun e => ZKMal.CoinLayer.rd e.2).length = q₀ + 1 := by rw [List.length_map, hq]
  have hsch' : schedSoFar t ((2 * i + r) % 2)
      (spec.domain (spec.tables[(2 * i + r) / 2]!) ((2 * i + r) % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[(2 * i + r) / 2]! (_root_.Flock.Firewall.Drive.linkPoints
        sts[(2 * i + r) / 2]! (wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hi
      ((Array.range (j + 1)).map fun k =>
        wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) (q₀ - (j + 1) + k + 1)) = .ok sched := by
    rw [h2r, h2i, htn, hsti, ← hco]
    exact hsch
  have hnone : schedOf (real x w ρ) x (h1.map fun e => ZKMal.CoinLayer.rd e.2) = none := by
    rw [schedOf_at hx hcs hloc, hreal]
    simp only [hsch', Array.getElem?_eq_none hle, Option.map_none]
  obtain ⟨hA0, hA1⟩ := ofLoop_answered_own ZKMal.Masked.dataE
    (withFiles (withLink (withInner ZKMal.Masked.dataE L₁ real fin) src) src) real x w ρ e0 h1 hA rfl hx hz hRT hq
    hloc hks hreal hsch' hle
  have hqi : q.st.m.inner = m.inner := by
    have h := congrArg Masker.inner huq
    exact h
  have hqe : q.st.m.eps = m.eps := by
    have h := congrArg Masker.eps huq
    exact h
  have hqb : q.st.m.beta = m.beta := by
    have h := congrArg Masker.beta huq
    exact h
  obtain ⟨hb4, hbf⟩ := roundPlace_own_bounds hfd hrp hf hss hn hl0 hv
  have hdone : q.st.m.inner ≠ some INNER_ROUNDS := by
    intro hin4
    rw [hin4, Option.getD_some] at hb4
    exact Nat.lt_irrefl _ hb4
  have hkey : j + 1 - sched.size < INNER_ROUNDS ∧ q.st.m.inner.getD 0 = j + 1 - sched.size ∧
      (j + 1 - sched.size = 1 → q.st.m.eps = (((Array.range (j + 1)).map fun k =>
        wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) (q₀ - (j + 1) + k + 1)).getD sched.size #[])[0]!) ∧
      (j + 1 - sched.size = 3 → q.st.m.beta = (((Array.range (j + 1)).map fun k =>
        wordsAt x (h1.map fun e => ZKMal.CoinLayer.rd e.2) (q₀ - (j + 1) + k + 1)).getD
          (sched.size + INNER_BETA) #[])[0]!) := by
    rcases Nat.lt_or_ge sched.size (j + 1) with hlt' | hge
    · obtain ⟨h4, hin, heps, hbeta⟩ := hA1 hlt'
      have hne3 : j - sched.size ≠ 3 := fun h3 => hdone (by rw [hqi, hin, h3]; rfl)
      refine ⟨by unfold INNER_ROUNDS at h4 ⊢; omega, by rw [hqi, hin, Option.getD_some]; omega,
        fun _ => hqe.trans heps, fun h3 => hqb.trans (hbeta (by unfold INNER_BETA; omega))⟩
    · have hS : sched.size = j + 1 := by omega
      refine ⟨by rw [hS, Nat.sub_self]; decide, by rw [hqi, hA0 hS, hS, Nat.sub_self]; rfl,
        fun h => by omega, fun h => by omega⟩
  obtain ⟨hlt4, hidx, he, hb⟩ := hkey
  have hin := Loop.innerOf_eq (real x w ρ) ((withFrames ZKMal.Masked.dataE L₁ real).cfg x ρ (e0 :: h1))
    (fin x w ρ (e0 :: h1)) x _ hx hcs hloc hreal hsch' hle hlt4
  have hsc : (ofLoop ZKMal.Masked.dataE (withFiles (withLink (withInner ZKMal.Masked.dataE L₁ real fin) src) src)
      real).scheduled x w ρ (e0 :: h1) vals = _ := scheduled_innerAt L₁ src real fin x w ρ e0 h1 hnone hin vals
  have hcfg : (ofLoop ZKMal.Masked.dataE (withFiles (withLink (withInner ZKMal.Masked.dataE L₁ real fin) src) src)
      real).cfg x ρ (e0 :: h1) = (withFrames ZKMal.Masked.dataE L₁ real).cfg x ρ (e0 :: h1) := by
    unfold LeanRelease.cfg LeanRelease.drawsAt
    rfl
  rw [hpc, hsc, bytes_toList_toArray, hcq, hcfg, hnr, hidx]
  rw [hidx] at hbf
  refine congrArg Place.sched (Eq.trans (innerSched_fin_congr _ q.st.m (fin := R.innerOf l0.root r)
    (fin' := fin x w ρ (e0 :: h1)) _ hlt4 (fun h0 => ?_) (fun h2 => ?_))
    (innerSched_congr _ _ _ hlt4 (fun h1' => ?_) (fun h3 => ?_)))
  · exact getD_of_agree hfin (by omega) _
  · exact getD_of_agree hfin (by omega) _
  · dsimp only
    rw [he h1', getElem!_zero_getD, extract_getD (k := 0) _ (by omega) (by simp), Nat.add_zero sched.size]
  · dsimp only
    rw [hb h3, getElem!_zero_getD, extract_getD (k := 2) _ (by omega) (by simp)]
    rfl

/-- **`placeFacts_seen`'s `hown`**: at the slot of a round of `stream`, table `i`'s rep `r`, of the deployed loop whose
release has the session loop's own rounds of the inner proof (`withInner` at the finish's rounds `fin`), where
`LoopFacts` and `LoopJ` hold, the model's tables and salts are the loop's (`RealAt`), its coin step takes the coins
before the round (`answeredR`), and each round of the inner proof the loop's finish computed so far for the rep is the
model's finish's (`InnerSrc`, at the model's `fin` the release's), wherever `seen` has a round of `stream` and
`placeOf` places it at the inner proof's own, the round's place is the model's scheduled round. `RepJ` gives the rep's
`StreamAt` on the frame, entry, name and level 0 `hown` reads (`Option.some.inj`), `SessOn` the entry's setup,
`LinkOn` the model's link and `SeenAt` the round past the stream's first; the frame keeps its schedule so far at
`real`'s salts as in `placeSched_later`, and `SameInv` puts the readied frame at the model's configuration and masking
state but for `stopped` (`sameInv_place`); the round's coin count is the model's (`Loop.reqOf_eq`, `LoopJ.plan`), so
`roundPlace_scheduled_own` places it. -/
theorem placeSched_own
    {L₁ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array F128 × ByteArray × Array ByteArray)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)}
    {fin : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M (LeanRelease.withFiles (LeanRelease.withLink
      (LeanRelease.withInner ZKMal.Masked.dataE L₁ real fin) src) src) real s0 spec sts)
    (hRT : Count.RealTabs sts (real M.x M.w M.ρ))
    {sid : Nat} {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hR : RealAt M real sid σ) (hans : ∃ m, M.L.answeredR M.x M.ρ (M.hist σ.c) = .ok m)
    (hin : InnerSrc M sid i r σ) (hfin : M.fin = fin M.x M.w M.ρ) {dom content : ByteArray} {q : Rep} {pc : Place}
    {f : Bool} {nd : Option String}
    (hrp : _root_.Flock.Firewall.Frames.roundPlace σ.R σ.L0 sid i r dom content n = .ok (q, pc, f, nd)) :
    seen.contains stream = true → ∀ (p : Rep) (rel : Release) (name : String)
      (l0 : _root_.Flock.Firewall.Level0.Level0) n' q', frameAt σ.R sid i r = some p → sessAt σ.R sid = some rel →
      rel.spec.tables[i]? = some name → _root_.Flock.Firewall.Level0.ofSession σ.L0 sid name = some l0 →
      _root_.Flock.Firewall.Frames.placeOf σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 = .ok (.own, n', q') →
      pc = .sched ⟨(M.L.scheduled M.x M.w M.ρ (M.hist σ.c)
        (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c)))).toArray⟩ := by
  intro hseen p rel name l0 n' q' hfa hss hn hl0 hpo
  obtain ⟨hsent, hop, -, hpq⟩ := slot_round hsl
  obtain ⟨hk0, hmem⟩ := round_rep hsl hF hJ hrep hlt hD.names
  have hr2 := rep_lt_two hmem
  have hlen : (M.hist σ.c).length = opened st + 1 := by
    rw [M.length_hist (Nat.le_of_lt hlt), hF.count hsent]
  obtain ⟨e0, h1, hh⟩ : ∃ e0 h1, M.hist σ.c = e0 :: h1 := by
    cases hc : M.hist σ.c with
    | nil =>
      rw [hc, List.length_nil] at hlen
      omega
    | cons e0 h1 => exact ⟨e0, h1, rfl⟩
  obtain ⟨q₀, hq⟩ : ∃ q₀, h1.length = q₀ + 1 := by
    refine ⟨opened st - 1, ?_⟩
    rw [hh, List.length_cons] at hlen
    omega
  have hop1 : opened st = q₀ + 1 := by
    rw [hh, List.length_cons, hq] at hlen
    omega
  have h2 : 0 < (nextAt spec sts (M.hist σ.c)).2 := by
    rcases (hJ.seenAt i r stream hmem).1 hseen with h | ⟨-, h⟩
    · omega
    · exact h
  obtain ⟨j, hkj⟩ : ∃ j, nextAt spec sts (M.hist σ.c) = (2 * i + r, j + 1) := by
    refine ⟨(nextAt spec sts (M.hist σ.c)).2 - 1, Prod.ext hk0.symm ?_⟩
    show (nextAt spec sts (M.hist σ.c)).2 = (nextAt spec sts (M.hist σ.c)).2 - 1 + 1
    omega
  obtain ⟨p₁, rel₁, name₁, l0₁, hfa₁, hss₁, hn₁, hl0₁, hA⟩ := hJ.reps i r stream hmem hseen
  have ep : p₁ = p := Option.some.inj (hfa₁.symm.trans hfa)
  have er : rel₁ = rel := Option.some.inj (hss₁.symm.trans hss)
  rw [er] at hn₁
  have en : name₁ = name := Option.some.inj (hn₁.symm.trans hn)
  rw [en] at hl0₁
  have el : l0₁ = l0 := Option.some.inj (hl0₁.symm.trans hl0)
  rw [ep, er, en, el] at hA
  obtain ⟨hspec, hsts⟩ := (hJ.sess hsent).eq hss
  have hc2 : 2 ≤ σ.c := by
    rw [hF.count hsent]
    omega
  have hl := hJ.link hc2 name l0 hl0 rel hss
  rw [hh, List.tail_cons] at hl
  have hss' : σ.R.sessions.find? (·.sid == sid) = some rel := hss
  have hl2 : (M.hist σ.c).length - 2 = q₀ := by
    rw [hh, List.length_cons, hq]
    omega
  have hk1 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).1 = 2 * i + r := by
    have := congrArg Prod.fst hkj
    unfold nextAt at this
    rwa [hl2] at this
  have hk2 : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 q₀).2 = j + 1 := by
    have := congrArg Prod.snd hkj
    unfold nextAt at this
    rwa [hl2] at this
  obtain ⟨j', c, stm, t, xs, hi, pre, ev, hj, hlc, hstm, htab, -, hxs, hI, hsa, hord⟩ :=
    hA.1 (by rw [hl2]; exact hk1)
  rw [hl2, hk2] at hj
  obtain rfl : j = j' := by omega
  rw [hl2, hh, List.tail_cons] at hI
  have hmod : (2 * i + r) % 2 = r := by omega
  have hdiv : (2 * i + r) / 2 = i := by omega
  rw [hmod] at hI hsa
  rw [hdiv] at hstm
  rw [hl, Option.some.injEq] at hlc
  subst hlc hxs
  obtain ⟨jf, hf⟩ := find_of_frameAt hfa
  have hpt : p.table = i := congrArg (fun x => x.2.1) (find_key hf)
  have hpr : p.rep = r := congrArg (fun x => x.2.2) (find_key hf)
  have hsz : p.coins.size = j + 1 := by rw [hI.coins, Array.size_map, Array.size_range]
  have hne' : p.coins.isEmpty = false := by simp [Array.isEmpty, hsz]
  obtain ⟨hiR, hreal, hRs⟩ := hR i r rel name l0 stm t hr2 hss hn hl0 hstm htab
  obtain ⟨r0, q0, hIR, hsalt⟩ : ∃ r0 q0, Loop.RepInv t (rel.spec.domain name r).toUTF8
      (_root_.Flock.Firewall.Drive.extras stm (_root_.Flock.Firewall.Drive.linkPoints stm
        (Loop.wordsAt M.x (h1.map fun e => ZKMal.CoinLayer.rd e.2) 0))) hiR p ∧
      (p.alg.size ≤ p.coins.size → p.opened = false → p.opening = none →
        σ.R.salts.find? (fun x => x.1 == l0.root && x.2.1 == p.rep) = some (r0, q0, hiR)) := by
    by_cases hc : p.opening = none ∧ p.opened = false
    · obtain ⟨r0, q0, hiF, hs⟩ := roundPlace_salts hfa hne' hrp hss' hn hl0
      refine ⟨r0, q0, repInv_hi hI.inv hc.1 hc.2, fun hle ho' hop' => ?_⟩
      have hf' := hs hle ho' hop'
      rw [hpr] at hf' ⊢
      rw [hf', hRs r0 q0 hiF hf']
    · have hle : p.alg.size ≤ p.coins.size := by
        rcases hI.inv.opening with h1' | ⟨h2', -⟩ | ⟨h3, -⟩
        · exact absurd ⟨h1'.1, h1'.2.1⟩ hc
        · exact h2'
        · exact h3
      have hlt' : p.alg.size < p.coins.size := by
        rcases Nat.lt_or_eq_of_le hle with h | h
        · exact h
        · exact absurd (hord h) hc
      obtain ⟨r0, q0, hf'⟩ := find?_of_saltsAt (hsa hlt')
      have hhi := hRs r0 q0 hi hf'
      subst hhi
      exact ⟨.empty, 0, hI.inv, fun _ ho' hop' => absurd ⟨hop', ho'⟩ hc⟩
  have hst' : rel.sts[p.table]? = some stm := by
    rw [hpt]
    exact hstm
  obtain ⟨w, hw⟩ := roundPlace_placeOf hf hrp (by rw [hne', Bool.false_and]) hss' hn hl0
  obtain ⟨sched, hsc⟩ := sched_of_placeOf hst' hl hsalt htab rfl hIR hw
  have hsc' := hsc
  rw [hpr, hspec] at hsc'
  obtain ⟨q'', hq'', -⟩ := Loop.placeOf_sched σ.R.salts l0 rel p (rel.spec.domain name r).toUTF8 hst' hl
    hsalt htab rfl hIR hsc
  rw [hsz] at hq''
  have hown : (Loop.placeAt sched (j + 1)).1 = .own := by
    have h := hpo.symm.trans hq''
    simp only [Except.ok.injEq, Prod.mk.injEq] at h
    exact h.1.symm
  obtain ⟨m, hAm⟩ := hans
  have hAm1 : M.L.answeredR M.x M.ρ (e0 :: h1) = .ok m := by
    rw [← hh]
    exact hAm
  have huq := (sameInv_place hI hfa hrp).2.2.2.1 m hAm1
  have hcq := (sameInv_place hI hfa hrp).2.2.1
  rw [hD.rel] at hcq hAm1
  have hnr : n = Loop.loopNr M.x (q₀ + 1) := by
    symm
    rw [← hop1]
    unfold Loop.loopNr
    rw [Loop.reqOf_eq hD.setup, ← hJ.plan hsent]
    unfold reqAt
    rw [ite_eq_right hop, Array.getD_eq_getD_getElem?, hpq, Option.getD_some]
  have hfin' := hin rel name l0 hss hn hl0
  rw [hfin, hh] at hfin'
  have := roundPlace_scheduled_own L₁ src real fin M.x M.w M.ρ e0 h1
    (computedOf (M.L.computed M.x M.w M.ρ (M.hist σ.c))) hD.setup (hJ.sizes hsent) hRT hq hk1 hr2 hk2
    (rep_lt_size hmem) hfa hrp hss' hn hl0 hspec hsts hl (by rw [← hsts]; exact hstm) htab hreal
    (by rw [← hspec]; exact hIR) hI.coins hsalt hsc' hown hcq hAm1 huq hnr hfin'
  rw [this, hD.rel, hh]

end Claims.ZKServer.Relay
