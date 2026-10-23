import Proofs.Flock.Claims.ZK.Refine
import Flock.Firewall.SessionPlan
import Flock.Firewall.Frames

/-!
# The release at the session loop (`flock-firewall session`)

The session loop (`Flock.Firewall.Session`, live PROTOCOL.md §10.4) decides what crosses to the auditor from pure functions
of the statement's setup (`Flock.Firewall.SessionPlan`): the `Hello` (`Hello.msg`), what its coin check reads of it
(`helloOf`: the tree, the link points and the nonce), each opening's stream and count (`planOf`, `reqAt`), and when
`Finish` leaves (`State.finish`: once the schedule's last opening has). `LeanRelease.loop` is a release with those, at the
statement `x`'s setup (`FlockDeploy.setupOf`), and here the transport's trusts on them are theorems:

* `LeanRelease.loop_helloIs`: its `Hello` is `CoinLayer.helloMsg`, by `rfl`.
* `LeanRelease.loop_helloParses`: its coin check reads the tape's nonce, and reads no `Hello` only where the schedule has no
  place for coin `0` (`helloOf` and `CoinLayer.slot` both refuse a tree whose streams are not the session's, `streamsOk`).
* `LeanRelease.loop_reqsFollow`: its requests follow the schedule (`CoinOpen.Follows`) at the coin counts it asks for
  (`Loop.loopNr`). A rep's planned rounds are the verifier's (`Loop.repPlan_size`: `repCoins` has
  `FlockDeploy.repRounds true` entries, by `decide +kernel` over the `--zk` schedules, `Loop.ligOk_lt`), so the plan has
  `refK x - 1` openings (`Loop.refK_plan`), and opening `j + 1`'s stream and round are `CoinLayer.slot`'s (`Loop.fst_plan`,
  `Loop.count_opened`).
* `LeanRelease.loop_last`: `Finish` is round `refK x + 1`.
* `LeanRelease.loop_coinsRead`: the coins a frame takes are the loop's reading of the auditor's last answer
  (`Frames.feedOf`, which the loop feeds the round's frame, `Frames.coins`), the decoded coin's words (`Loop.readCoins`).

**`Flock.zk_server_lean_loop`** is `Flock.zk_server_lean_coinLayer` at that release, with the transport's other trusts
(`ProgReads`, `LinkRoundEq`, `FinalRoundEq`) and the tables' side as before. What it still reads of the loop is that it
runs at `x`'s setup: the loop sets up from the auditor's draw (`Hello.Setup.setup`), which is `x`'s where `x` reads
(`FlockDeploy.Ins.setup`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock.Firewall.Session (INNER_CONS replayCoins ligeritoCoins repCoins repPlan repName planOf reqAt streamsOk
  helloOf)
open _root_.Flock.Firewall.Frame (Cfg Masker Draws MaskSlot)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

/-! ## A rep's rounds -/

/-- Ligerito's rounds at `m`'s `--zk` schedule, less those of no coins, are the verifier's `k0 + 7 + 11 R`, and a
schedule needs `m ≥ 25`. -/
def ligOk (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule m with
  | .ok (sch, _) =>
    decide (25 ≤ m) && ((ligeritoCoins sch).filter (· != 0)).size == sch.k0 + 7 + 11 * (sch.levels.size - 1)
  | .error _ => true

/-- Every `--zk` schedule, `m < 36`, by the kernel's evaluation. -/
theorem ligOk_lt : ∀ m, m < 36 → ligOk m = true := by decide +kernel

theorem queries0_ge {m : ℕ} (h : 36 ≤ m) : ∃ e, _root_.Flock.Zk.queries0 m = .error e := by
  unfold _root_.Flock.Zk.queries0
  have h1 : ¬ m ≤ 24 := by omega
  have h2 : (m == 25) = false := by rw [beq_eq_false_iff_ne]; omega
  have h3 : (m == 26) = false := by rw [beq_eq_false_iff_ne]; omega
  have h4 : (m == 27) = false := by rw [beq_eq_false_iff_ne]; omega
  have h5 : ¬ m ≤ 31 := by omega
  have h6 : ¬ m ≤ 35 := by omega
  simp only [h1, h2, h3, h4, h5, h6, ↓reduceIte, Bool.false_eq_true]
  exact ⟨_, rfl⟩

theorem schedule_ge {m : ℕ} (h : 36 ≤ m) : ∃ e, _root_.Flock.Zk.schedule m = .error e := by
  obtain ⟨e, he⟩ := queries0_ge h
  unfold _root_.Flock.Zk.schedule
  rw [he]
  exact ⟨_, rfl⟩

theorem ligOk_all (m : ℕ) : ligOk m = true := by
  by_cases h : m < 36
  · exact ligOk_lt m h
  · obtain ⟨e, he⟩ := schedule_ge (m := m) (by omega)
    unfold ligOk
    rw [he]

theorem size_filter_toList (p : ℕ → Bool) (xs : Array ℕ) : (xs.filter p).size = (xs.toList.filter p).length := by
  rw [← Array.toList_filter, Array.length_toList]

theorem repCoins_ok {m kLog claims : ℕ} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (h : _root_.Flock.Zk.schedule m = .ok (sch, tp)) :
    repCoins m kLog claims = .ok ((replayCoins m kLog ++ Array.replicate claims 7 ++ #[claims] ++ ligeritoCoins sch ++
      #[1, INNER_CONS, 1, _root_.Flock.Zk.Q]).filter (· != 0)) := by
  unfold repCoins
  rw [h]
  rfl

theorem repCoins_size {m kLog claims : ℕ} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (h : _root_.Flock.Zk.schedule m = .ok (sch, tp)) (hc : claims ≠ 0) :
    ((repCoins m kLog claims).toOption.getD #[]).size =
      (m + (kLog - 6)) + (claims + 1) + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 4 := by
  have hlig := ligOk_all m
  unfold ligOk at hlig
  rw [h] at hlig
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hlig
  obtain ⟨h25, hl⟩ := hlig
  rw [size_filter_toList] at hl
  rw [repCoins_ok h]
  simp only [Except.toOption, Option.getD_some]
  rw [size_filter_toList]
  simp only [Array.toList_append, List.filter_append, List.length_append]
  rw [hl]
  have hm13 : ¬ m - 13 = 0 := by omega
  have hm13' : ¬ m ≤ 13 := by omega
  simp [replayCoins, INNER_CONS, _root_.Flock.Zk.Q, hm13, hc]
  omega

theorem repRounds_true (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) :
    FlockDeploy.repRounds true spec st =
      match _root_.Flock.Zk.schedule st.m with
      | .ok (sch, _) => (st.m + (st.c.kLog - 6)) + ((2 + spec.points * st.regions.size) + 1) +
          (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 4
      | .error _ => 0 := rfl

/-- **A rep's planned openings are the verifier's rounds**: `repPlan` has `FlockDeploy.repRounds true` entries. -/
theorem repPlan_size (spec : _root_.Flock.SessionSpec) (st : _root_.Flock.Stmt) :
    (repPlan spec st).size = FlockDeploy.repRounds true spec st := by
  rw [repRounds_true]
  unfold repPlan
  cases h : _root_.Flock.Zk.schedule st.m with
  | error e =>
    have hc : repCoins st.m st.c.kLog (2 + spec.points * st.regions.size) = .error e := by
      unfold repCoins
      rw [h]
      rfl
    rw [hc]
    rfl
  | ok p =>
    obtain ⟨sch, tp⟩ := p
    exact repCoins_size h (by omega)

/-! ## Locating a rep round -/

theorem locate_cons_lt {c i k : ℕ} {cs : List ℕ} (h : i < c) : CoinLayer.locate (c :: cs) k i = (k, i) := by
  show (if i < c then (k, i) else CoinLayer.locate cs (k + 1) (i - c)) = _
  simp only [h, ↓reduceIte]

theorem locate_cons_ge {c i k : ℕ} {cs : List ℕ} (h : ¬ i < c) :
    CoinLayer.locate (c :: cs) k i = CoinLayer.locate cs (k + 1) (i - c) := by
  show (if i < c then (k, i) else CoinLayer.locate cs (k + 1) (i - c)) = _
  simp only [h, ↓reduceIte]

theorem locate_fst_ge : ∀ (rs : List ℕ) (k i : ℕ), k ≤ (CoinLayer.locate rs k i).1
  | [], _, _ => Nat.le_refl _
  | c :: cs, k, i => by
    by_cases h : i < c
    · rw [locate_cons_lt h] <;> exact Nat.le_refl _
    · rw [locate_cons_ge h]; exact Nat.le_of_succ_le (locate_fst_ge cs (k + 1) (i - c))

theorem locate_lt : ∀ (rs : List ℕ) (k i : ℕ), i < rs.sum → (CoinLayer.locate rs k i).1 < k + rs.length
  | [], _, _, h => by simp at h
  | c :: cs, k, i, h => by
    by_cases hc : i < c
    · rw [locate_cons_lt hc]; simp
    · rw [locate_cons_ge hc]
      have := locate_lt cs (k + 1) (i - c) (by simp at h; omega)
      simp only [List.length_cons]; omega

theorem count_locate : ∀ (rs : List ℕ) (k i : ℕ), i < rs.sum →
    ((List.range i).map fun i' => (CoinLayer.locate rs k i').1).count (CoinLayer.locate rs k i).1 =
      (CoinLayer.locate rs k i).2
  | [], _, _, h => by simp at h
  | c :: cs, k, i, h => by
    by_cases hc : i < c
    · rw [locate_cons_lt hc]
      have e : ((List.range i).map fun i' => (CoinLayer.locate (c :: cs) k i').1) = List.replicate i k := by
        apply List.ext_getElem (by simp)
        intro n h1 h2
        simp only [List.getElem_map, List.getElem_range, List.getElem_replicate]
        simp only [List.length_map, List.length_range] at h1
        rw [locate_cons_lt (by omega)]
      rw [e, List.count_replicate_self]
    · rw [locate_cons_ge hc]
      have hi : i = c + (i - c) := by omega
      have ih := count_locate cs (k + 1) (i - c) (by simp at h; omega)
      have hK := locate_fst_ge cs (k + 1) (i - c)
      rw [hi, List.range_add, List.map_append, List.count_append, ← hi]
      have e1 : ((List.range c).map fun i' => (CoinLayer.locate (c :: cs) k i').1) = List.replicate c k := by
        apply List.ext_getElem (by simp)
        intro n h1 h2
        simp only [List.getElem_map, List.getElem_range, List.getElem_replicate]
        simp only [List.length_map, List.length_range] at h1
        rw [locate_cons_lt h1]
      have e2 : (((List.range (i - c)).map (c + ·)).map fun i' => (CoinLayer.locate (c :: cs) k i').1) =
          (List.range (i - c)).map fun i' => (CoinLayer.locate cs (k + 1) i').1 := by
        rw [List.map_map]
        apply List.map_congr_left
        intro n _
        simp only [Function.comp]
        rw [locate_cons_ge (by omega)]
        congr 2
        omega
      rw [e1, e2, ih, List.count_replicate]
      have : (k == (CoinLayer.locate cs (k + 1) (i - c)).1) = false := by
        rw [beq_eq_false_iff_ne]; omega
      simp [this]

theorem fst_getElem?_flatten {α : Type} (nm : ℕ → String) :
    ∀ (ls : List (List (String × α))) (k₀ i : ℕ) (p : String × α),
      (∀ k (hk : k < ls.length), ∀ q ∈ ls[k], q.1 = nm (k₀ + k)) → ls.flatten[i]? = some p →
      p.1 = nm (CoinLayer.locate (ls.map List.length) k₀ i).1
  | [], _, _, _, _, h => by simp at h
  | l :: ls, k₀, i, p, hyp, h => by
    rw [List.flatten_cons] at h
    by_cases hc : i < l.length
    · rw [List.getElem?_append_left hc] at h
      simp only [List.map_cons]
      rw [locate_cons_lt hc]
      have := hyp 0 (by simp) p (List.mem_of_getElem? h)
      simpa using this
    · rw [List.getElem?_append_right (by omega)] at h
      simp only [List.map_cons]
      rw [locate_cons_ge hc]
      refine fst_getElem?_flatten nm ls (k₀ + 1) (i - l.length) p ?_ h
      intro k hk q hq
      have := hyp (k + 1) (by simp; omega) q (by simpa using hq)
      rw [this]; congr 1; omega

theorem sum_half (f : ℕ → ℕ) : ∀ n,
    ((List.range (2 * n)).map fun k => f (k / 2)).sum = ((List.range n).map fun i => 2 * f i).sum
  | 0 => rfl
  | n + 1 => by
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by omega, List.range_succ, List.range_succ, List.range_succ]
    simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    rw [sum_half f n, show (2 * n + 1) / 2 = n by omega, show 2 * n / 2 = n by omega]
    omega

theorem foldl_add {β : Type} (g : β → ℕ) : ∀ (l : List β) (a : ℕ),
    l.foldl (fun acc x => acc + g x) a = a + (l.map g).sum
  | [], a => by simp
  | x :: l, a => by
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [foldl_add g l]; omega

theorem map_range_getElem! {β : Type} [Inhabited β] (a : Array β) (h : β → ℕ) :
    (List.range a.size).map (fun i => h a[i]!) = a.toList.map h := by
  apply List.ext_getElem (by simp)
  intro n h1 h2
  simp only [List.length_map, List.length_range] at h1
  simp [h1]

/-! ## The loop's plan -/

/-- The rep streams' round counts in `SessionSpec.repStreams` order, as `CoinLayer.slot` reads them. -/
def roundsOf (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) : List ℕ :=
  (List.range spec.repStreams.size).map fun k => FlockDeploy.repRounds true spec sts[k / 2]!

/-- `planOf`'s openings stream by stream. -/
def plansOf (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) : List (List (String × ℕ)) :=
  (List.range spec.repStreams.size).map fun k => (repPlan spec sts[k / 2]!).toList.map (repName spec k, ·)

theorem planOf_toList (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) :
    (planOf spec sts).toList = (plansOf spec sts).flatten := by
  simp only [planOf, plansOf, Array.toList_flatMap, Array.toList_range, Array.toList_map, List.flatMap]

theorem plansOf_length (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) :
    (plansOf spec sts).map List.length = roundsOf spec sts := by
  simp only [plansOf, roundsOf, List.map_map]
  apply List.map_congr_left
  intro k _
  simp [repPlan_size]

theorem plan_size (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) :
    (planOf spec sts).size = (roundsOf spec sts).sum := by
  rw [← Array.length_toList, planOf_toList, List.length_flatten, plansOf_length]

theorem rounds_eq {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hN : spec.repStreams.size = 2 * sts.size) : FlockDeploy.rounds true spec sts = (roundsOf spec sts).sum := by
  unfold FlockDeploy.rounds roundsOf
  rw [hN]
  have e1 := sum_half (fun i => FlockDeploy.repRounds true spec sts[i]!) sts.size
  have e2 := map_range_getElem! sts (fun st => 2 * FlockDeploy.repRounds true spec st)
  have e3 := foldl_add (fun st => 2 * FlockDeploy.repRounds true spec st) sts.toList 0
  rw [e1, e2, ← Array.foldl_toList, e3, Nat.zero_add]

theorem refK_of_setup {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) : refK x = 1 + FlockDeploy.rounds true spec sts := by
  unfold refK
  simp only [FlockDeploy.V, hs]
  omega

/-! ## The streams -/

theorem streamsOk_eq {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (h : streamsOk spec sts ct = true) :
    ct.streams.toList = _root_.Flock.Firewall.Coins.SESSION :: spec.repStreams.toList.map (·.2.2) ∧
      ct.streams.toList.Nodup ∧ spec.repStreams.size = 2 * sts.size := by
  simp only [streamsOk, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

theorem refK_plan {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (hs : FlockDeploy.setupOf true x = some (s, spec, sts))
    (hok : streamsOk spec sts ct = true) : refK x = 1 + (planOf spec sts).size := by
  rw [refK_of_setup hs, rounds_eq (streamsOk_eq hok).2.2, plan_size]

theorem repName_eq {spec : _root_.Flock.SessionSpec} {k : ℕ} (hk : k < spec.repStreams.size) :
    repName spec k = spec.repStreams[k].2.2 := by
  simp [repName, hk]

theorem idx_repName {spec : _root_.Flock.SessionSpec} {ct : _root_.Flock.CoinTree.Spec} {k : ℕ}
    (hs : ct.streams.toList = _root_.Flock.Firewall.Coins.SESSION :: spec.repStreams.toList.map (·.2.2))
    (hnd : ct.streams.toList.Nodup) (hk : k < spec.repStreams.size) :
    ct.streams.toList.idxOf (repName spec k) = k + 1 := by
  have hl : k + 1 < ct.streams.toList.length := by rw [hs]; simp [hk]
  have e : repName spec k = ct.streams.toList[k + 1] := by
    simp only [hs, List.getElem_cons_succ, List.getElem_map, Array.getElem_toList]
    exact repName_eq hk
  rw [e, List.Nodup.idxOf_getElem hnd _ hl]

theorem idx_session {spec : _root_.Flock.SessionSpec} {ct : _root_.Flock.CoinTree.Spec}
    (hs : ct.streams.toList = _root_.Flock.Firewall.Coins.SESSION :: spec.repStreams.toList.map (·.2.2)) :
    ct.streams.toList.idxOf _root_.Flock.Firewall.Coins.SESSION = 0 := by
  rw [hs]
  exact List.idxOf_cons_self

theorem ne_session {spec : _root_.Flock.SessionSpec} {ct : _root_.Flock.CoinTree.Spec} {k : ℕ}
    (hs : ct.streams.toList = _root_.Flock.Firewall.Coins.SESSION :: spec.repStreams.toList.map (·.2.2))
    (hnd : ct.streams.toList.Nodup) (hk : k < spec.repStreams.size) :
    (repName spec k == _root_.Flock.Firewall.Coins.SESSION) = false := by
  rw [beq_eq_false_iff_ne]
  intro e
  have h1 := idx_repName hs hnd hk
  rw [e, idx_session hs] at h1
  omega

/-! ## The schedule's places -/

theorem slot_zero {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (hs : FlockDeploy.setupOf true x = some (s, spec, sts))
    (hc : spec.coinTree = some ct) (hok : streamsOk spec sts ct = true) :
    CoinLayer.slot x 0 = some (spec, ct, 0, 0) := by
  unfold CoinLayer.slot
  simp only [hs, hc, hok, ↓reduceIte]

theorem slot_succ {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (hs : FlockDeploy.setupOf true x = some (s, spec, sts))
    (hc : spec.coinTree = some ct) (hok : streamsOk spec sts ct = true) (j : ℕ) :
    CoinLayer.slot x (j + 1) = some (spec, ct,
      ct.streams.toList.idxOf (repName spec (CoinLayer.locate (roundsOf spec sts) 0 j).1),
      (CoinLayer.locate (roundsOf spec sts) 0 j).2) := by
  unfold CoinLayer.slot
  simp only [hs, hc, hok, ↓reduceIte, Nat.add_one_ne_zero, Nat.add_sub_cancel, roundsOf]

theorem fst_plan {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {i : ℕ}
    (hi : i < (roundsOf spec sts).sum) :
    ((planOf spec sts).getD i ("", 0)).1 = repName spec (CoinLayer.locate (roundsOf spec sts) 0 i).1 := by
  have hi' : i < (planOf spec sts).size := by rw [plan_size]; exact hi
  have h2 : (plansOf spec sts).flatten[i]? = some (planOf spec sts)[i] := by
    rw [← planOf_toList, Array.getElem?_toList]
    simp [hi']
  have hg : (planOf spec sts).getD i ("", 0) = (planOf spec sts)[i] := by simp [hi']
  rw [hg, ← plansOf_length]
  refine fst_getElem?_flatten (repName spec) (plansOf spec sts) 0 i _ ?_ h2
  intro k hk q hq
  simp only [plansOf, List.getElem_map, List.getElem_range, List.mem_map] at hq
  obtain ⟨_, _, rfl⟩ := hq
  simp

/-! ## The loop's requests -/

/-- **The loop's request for opening `j`** at `x`: `reqAt` at the plan and link points `x`'s setup gives. -/
def reqOf (x : Msg) (j : ℕ) : String × ℕ :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, sts) => reqAt (planOf spec sts) (spec.points * spec.mPts) j
  | none => ("", 0)

/-- The coin counts the loop requests: rep round `j`'s, `reqOf`'s. -/
def loopNr (x : Msg) (j : ℕ) : ℕ := (reqOf x j).2

theorem reqOf_eq {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (j : ℕ) :
    reqOf x j = reqAt (planOf spec sts) (spec.points * spec.mPts) j := by
  unfold reqOf
  rw [hs]

/-- The streams' indices in the tree of the first `j` openings `req` asks for. -/
def openedAt (ct : _root_.Flock.CoinTree.Spec) (req : ℕ → String × ℕ) (j : ℕ) : List ℕ :=
  (List.range j).map fun i => ct.streams.toList.idxOf (req i).1

theorem count_succ_map (l : List ℕ) (k : ℕ) : (l.map (· + 1)).count (k + 1) = l.count k := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.count_cons, ih]

theorem count_opened {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (hs : FlockDeploy.setupOf true x = some (s, spec, sts))
    (hok : streamsOk spec sts ct = true) {i : ℕ} (hi : i < (roundsOf spec sts).sum) :
    (openedAt ct (reqOf x) (i + 1)).count ((CoinLayer.locate (roundsOf spec sts) 0 i).1 + 1) =
      (CoinLayer.locate (roundsOf spec sts) 0 i).2 := by
  obtain ⟨hst, hnd, hN⟩ := streamsOk_eq hok
  have e : openedAt ct (reqOf x) (i + 1) =
      0 :: ((List.range i).map fun i' => (CoinLayer.locate (roundsOf spec sts) 0 i').1).map (· + 1) := by
    unfold openedAt
    rw [List.range_succ_eq_map, List.map_cons, List.map_map, List.map_map]
    have h0 : (reqOf x 0).1 = _root_.Flock.Firewall.Coins.SESSION := by rw [reqOf_eq hs]; rfl
    rw [h0, idx_session hst]
    congr 1
    apply List.map_congr_left
    intro i' hi'
    rw [List.mem_range] at hi'
    have hlt : i' < (roundsOf spec sts).sum := by omega
    have hk : (CoinLayer.locate (roundsOf spec sts) 0 i').1 < spec.repStreams.size := by
      have := locate_lt (roundsOf spec sts) 0 i' hlt
      simpa [roundsOf] using this
    simp only [Function.comp, Nat.succ_eq_add_one]
    rw [reqOf_eq hs]
    show ct.streams.toList.idxOf ((planOf spec sts).getD i' ("", 0)).1 = _
    rw [fst_plan hlt, idx_repName hst hnd hk]
  rw [e, List.count_cons, count_succ_map, count_locate _ 0 i hi]
  simp

theorem follows_head {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (hs : FlockDeploy.setupOf true x = some (s, spec, sts))
    (hc : spec.coinTree = some ct) (hok : streamsOk spec sts ct = true) (j : ℕ)
    (hj : j < (planOf spec sts).size + 1) :
    CoinLayer.place loopNr x j = CoinOpen.placeOf ct (ct.streams.toList.idxOf (reqOf x j).1)
      ((openedAt ct (reqOf x) j).count (ct.streams.toList.idxOf (reqOf x j).1))
      (if (reqOf x j).1 == _root_.Flock.Firewall.Coins.SESSION then spec.points * spec.mPts else (reqOf x j).2) := by
  obtain ⟨hst, hnd, hN⟩ := streamsOk_eq hok
  cases j with
  | zero =>
    rw [CoinOpen.place_eq (slot_zero hs hc hok)]
    have h0 : (reqOf x 0).1 = _root_.Flock.Firewall.Coins.SESSION := by rw [reqOf_eq hs]; rfl
    rw [h0, idx_session hst]
    simp [openedAt]
  | succ i =>
    have hi : i < (roundsOf spec sts).sum := by rw [← plan_size]; omega
    have hk : (CoinLayer.locate (roundsOf spec sts) 0 i).1 < spec.repStreams.size := by
      have := locate_lt (roundsOf spec sts) 0 i hi
      simpa [roundsOf] using this
    have h1 : (reqOf x (i + 1)).1 = repName spec (CoinLayer.locate (roundsOf spec sts) 0 i).1 := by
      rw [reqOf_eq hs]
      exact fst_plan hi
    rw [CoinOpen.place_eq (slot_succ hs hc hok i), h1, idx_repName hst hnd hk, ne_session hst hnd hk,
      count_opened hs hok hi]
    simp [loopNr]

theorem follows_gen {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (hs : FlockDeploy.setupOf true x = some (s, spec, sts))
    (hc : spec.coinTree = some ct) (hok : streamsOk spec sts ct = true) :
    ∀ (ms : List Msg) (j : ℕ), j + ms.length ≤ (planOf spec sts).size + 1 →
      CoinOpen.Follows (CoinLayer.place loopNr x) ct (spec.points * spec.mPts) j (openedAt ct (reqOf x) j)
        (reqsFrom (reqOf x) j ms)
  | [], _, _ => trivial
  | _ :: ms, j, h => by
    simp only [List.length_cons] at h
    show _ ∧ _
    refine ⟨follows_head hs hc hok j (by omega), ?_⟩
    have e : openedAt ct (reqOf x) j ++ [ct.streams.toList.idxOf (reqOf x j).1] = openedAt ct (reqOf x) (j + 1) := by
      simp [openedAt, List.range_succ]
    rw [e]
    exact follows_gen hs hc hok ms (j + 1) (by omega)

/-! ## The loop's `Hello` -/

/-- **What the loop's coin check reads of the session's `Hello`** at `x` and the nonce `ν`: `helloOf` at `x`'s setup and
the nonce's bytes. -/
def helloChkAt (x : Msg) (ν : CoinLayer.Nonce) : Except String _root_.Flock.Firewall.Coins.Hello :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, sts) =>
    match helloOf spec sts ⟨(CoinLayer.nb ν).toArray⟩ with
    | some hl => .ok hl
    | none => .error "the session has no coin tree of its own streams"
  | none => .error "the statement sets up no session"

theorem helloChkAt_ok {x : Msg} {ν : CoinLayer.Nonce} {hl : _root_.Flock.Firewall.Coins.Hello}
    (h : helloChkAt x ν = .ok hl) :
    ∃ s spec sts ct, FlockDeploy.setupOf true x = some (s, spec, sts) ∧ spec.coinTree = some ct ∧
      streamsOk spec sts ct = true ∧
      hl = { spec := ct, links := spec.points * spec.mPts, nonce := CoinLayer.nb ν } := by
  unfold helloChkAt at h
  rcases hs : FlockDeploy.setupOf true x with _ | ⟨s, spec, sts⟩
  · simp only [hs] at h
    cases h
  · simp only [hs, helloOf] at h
    rcases hc : spec.coinTree with _ | ct
    · simp only [hc] at h
      cases h
    · by_cases hok : streamsOk spec sts ct = true
      · simp only [hc, hok, ↓reduceIte, Except.ok.injEq] at h
        exact ⟨s, spec, sts, ct, rfl, hc, hok, by rw [← h]⟩
      · simp only [hc, hok, ↓reduceIte, Bool.false_eq_true] at h
        cases h

theorem helloChkAt_error {x : Msg} {ν : CoinLayer.Nonce} {e : String} (h : helloChkAt x ν = .error e) :
    CoinLayer.slot x 0 = none := by
  unfold helloChkAt at h
  unfold CoinLayer.slot
  rcases hs : FlockDeploy.setupOf true x with _ | ⟨s, spec, sts⟩
  · rfl
  · simp only [hs, helloOf] at h
    simp only
    rcases hc : spec.coinTree with _ | ct
    · rfl
    · by_cases hok : streamsOk spec sts ct = true
      · simp only [hc, hok, ↓reduceIte] at h
        cases h
      · simp only [hok, Bool.false_eq_true, ↓reduceIte]

/-! ## The loop's `Finish` -/

/-- **The round the loop's `Finish` leaves at** at `x`: past `Hello` and every opening of the schedule (`State.finish`),
the plan's size `+ 2`. Where `x` sets up no session with a tree of its own streams no opening leaves, and neither does
`Finish`; the round is then the protocol's. -/
def lastOf (x : Msg) : ℕ :=
  match FlockDeploy.setupOf true x with
  | some (_, spec, sts) =>
    match spec.coinTree with
    | some ct => if streamsOk spec sts ct then (planOf spec sts).size + 2 else refK x + 1
    | none => refK x + 1
  | none => refK x + 1

/-! ## The coins the loop feeds a frame -/

/-- **The coins the loop feeds the frame** after `h`: `Frames.feedOf` of the last opening at its request's count. -/
def coinsAt (x : Msg) (h : List (Msg × Msg)) : Array _root_.Flock.F128 :=
  match h.tail.getLast? with
  | some (_, a) => _root_.Flock.Firewall.Frames.feedOf (loopNr x (h.tail.length - 1)) a
  | none => #[]

/-- The coins `coinsAt` reads, from the decoded coins: the last one's words, as many as its request's count. -/
def readCoins (x : Msg) (cs : List FlockDeploy.Coin) : Array _root_.Flock.F128 :=
  match cs.getLast? with
  | some c => c.vals (min (loopNr x (cs.length - 1)) _root_.Flock.Firewall.Frames.COIN_MAX)
  | none => #[]

/-- `feedOf` of an opening is its decoded coin's words. -/
theorem feedOf_rd (n : ℕ) (m : Msg) :
    _root_.Flock.Firewall.Frames.feedOf n m = (CoinLayer.rd m).vals (min n _root_.Flock.Firewall.Frames.COIN_MAX) := by
  unfold _root_.Flock.Firewall.Frames.feedOf _root_.Flock.Firewall.Frame.words FlockDeploy.Coin.vals
  apply Array.ext (by simp)
  intro i h1 _
  simp only [Array.size_map, Array.size_range] at h1
  have hi : i < 65536 := by
    have : min n _root_.Flock.Firewall.Frames.COIN_MAX ≤ 65536 := Nat.min_le_right _ _
    omega
  simp [FlockDeploy.Coin.at, hi, CoinLayer.rd, CoinLayer.coinOfD, FlockComplete.coinOf, Claims.ofMsg]

theorem lastOf_eq (x : Msg) : lastOf x = refK x + 1 := by
  unfold lastOf
  rcases hs : FlockDeploy.setupOf true x with _ | ⟨s, spec, sts⟩
  · rfl
  · simp only
    rcases hc : spec.coinTree with _ | ct
    · rfl
    · by_cases hok : streamsOk spec sts ct = true
      · simp only [hok, ↓reduceIte]
        rw [refK_plan hs hok]
        omega
      · simp only [hok, Bool.false_eq_true, ↓reduceIte]

end Loop

namespace LeanRelease

/-- **The release at the session loop**: `L₀`'s, with the loop's `Hello` at the tape's nonce, its coin check's reading of
that `Hello` (`Loop.helloChkAt`), its requests (`Loop.reqOf`) and its `Finish`'s round (`Loop.lastOf`), each at the
statement's setup. -/
noncomputable def loop (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand)) :
    LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand) :=
  { L₀ with
    hello := fun x ρ => CoinLayer.helloMsg x ρ.1
    helloChk := fun x ρ => Loop.helloChkAt x ρ.1
    req := Loop.reqOf
    coinsOf := Loop.coinsAt
    last := Loop.lastOf }

/-- **The coins the loop's frames take read the history only through the decoded coins**: `Loop.readCoins` of them. -/
theorem loop_coinsRead (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (x : Msg) :
    CoinsRead (CoinLayer.session nr T) (loop nr T L₀) Loop.readCoins x := by
  intro m₀ ρ₀ ops
  show Loop.coinsAt x ((m₀, ρ₀) :: ops) = Loop.readCoins x (ops.map fun p => CoinLayer.rd p.2)
  unfold Loop.coinsAt Loop.readCoins
  simp only [List.tail_cons, List.getLast?_map, List.length_map]
  rcases ops.getLast? with _ | ⟨a, b⟩
  · rfl
  · exact Loop.feedOf_rd _ b

/-- **The loop's `Hello` is `flock-firewall hello`'s.** -/
theorem loop_helloIs (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (x : Msg) :
    HelloIs nr T (loop nr T L₀) x :=
  fun _ _ => rfl

/-- **The loop's coin check reads the session's `Hello` with the tape's nonce.** -/
theorem loop_helloParses (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (x : Msg) :
    HelloParses nr T (loop nr T L₀) x := by
  intro ν t
  show (match Loop.helloChkAt x ν with
    | .ok hl => hl.nonce = CoinLayer.nb ν
    | .error _ => CoinLayer.slot x 0 = none)
  cases h : Loop.helloChkAt x ν with
  | ok hl =>
    obtain ⟨_, _, _, _, _, _, _, rfl⟩ := Loop.helloChkAt_ok h
    exact rfl
  | error e => exact Loop.helloChkAt_error h

/-- **The loop requests each opening at the schedule's place**, at the coin counts it asks for (`Loop.loopNr`). -/
theorem loop_reqsFollow (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)) (x : Msg) :
    ReqsFollow Loop.loopNr T (loop Loop.loopNr T L₀) x := by
  intro ν t hl h ms hms
  obtain ⟨s, spec, sts, ct, hs, hc, hok, rfl⟩ := Loop.helloChkAt_ok h
  rw [Loop.refK_plan hs hok] at hms
  exact Loop.follows_gen hs hc hok ms 0 (by omega)

/-- **The loop's `Finish` is one past the last coin.** -/
theorem loop_last (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand))
    (x : Msg) :
    (loop nr T L₀).last x = refK x + 1 :=
  Loop.lastOf_eq x

end LeanRelease

open FlockSoundness.Discharge

/-- **`Flock.zk_server_lean_coinLayer` at the session loop** (`LeanRelease.loop`): its `Hello`, its coin check's reading
of it, its requests, the coins its frames take and its `Finish`'s round are theorems (`loop_helloIs`, `loop_helloParses`,
`loop_reqsFollow`, `loop_coinsRead`, `loop_last`), at the coin counts the loop requests (`Loop.loopNr`) and the coins'
reading `Loop.readCoins`; the transport's other trusts (`ProgReads`, `LinkRoundEq`, `FinalRoundEq`) and the tables'
side are `Flock.zk_server_lean_coinLayer`'s. -/
theorem Flock.zk_server_lean_loop
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (T : CoinLayer.TableData)
    (vs : (CoinLayer.session Loop.loopNr T).SwVals) (t : ℕ) (ht : 0 < t)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (hS : (CoinLayer.session Loop.loopNr T).Tables)
    (hV : (CoinLayer.session Loop.loopNr T).Complete (Flock.R (FlockDeploy.deployZk rel).parse))
    (hD : (CoinLayer.session Loop.loopNr T).DummyReg) (hc : (CoinLayer.session Loop.loopNr T).Causal)
    (hsc : (CoinLayer.session Loop.loopNr T).SwCausal vs) (hT : (CoinLayer.session Loop.loopNr T).TapeOK)
    (L₀ L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (hL : L = LeanRelease.loop Loop.loopNr T L₀)
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (hR : Flock.R (FlockDeploy.deployZk rel).parse x w)
    (pc : Msg → List (CoinLayer.session Loop.loopNr T).Coin → Masker)
    (fc : Msg → List (CoinLayer.session Loop.loopNr T).Coin → Except String Masker)
    (hPr : LeanRelease.ProgReads (CoinLayer.session Loop.loopNr T) L pc fc x)
    (hLk : LeanRelease.LinkRoundEq (CoinLayer.session Loop.loopNr T) vs L x w)
    (hF : LeanRelease.FinalRoundEq (CoinLayer.session Loop.loopNr T) vs L CoinOpen.shaH wk x w)
    (hE : LeanRelease.LiveRoundEq (CoinLayer.session Loop.loopNr T) vs L CoinOpen.shaH wk x w)
    (hEf : LeanRelease.FillRoundEq (CoinLayer.session Loop.loopNr T) vs L CoinOpen.shaH wk x w)
    {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = (FlockDeploy.deployZk rel).V.k x + 1)
    (Ev : List ((Msg × Out) × ℕ) → Prop) :
    Pr (fun ω : ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand) × L.toSrv.Chk =>
        Ev (L.toSrv.timed x
          (runStar CoinOpen.shaH (L.toSrv.proverD L.fillOf (fun x => (FlockDeploy.deployZk rel).V.k x + 1) wk w) Vs x ω
            θ).msgs)) ≤
      2 * (Pr (fun c : (leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) L pc fc Loop.readCoins) L.stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) L pc fc Loop.readCoins) L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).C x =>
          Ev (L.toSrv.timed x ((leaky1 ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) L pc fc Loop.readCoins) L.stop).refP.withSim fun x _ c =>
              ((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) L pc fc Loop.readCoins) L.stop).coinModel x c).simulatorE t (t * 3 ^ t))
            (fun x => (FlockDeploy.deployZk rel).V.k x + 1) L.toSrv.stop).out CoinOpen.shaH Vs θ x c))) +
          (8 * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) L pc fc Loop.readCoins)
              L.stop).nLeaves x * δ₁.toReal +
            8 * t * ((CoinLayer.session Loop.loopNr T).withStops (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) L pc fc Loop.readCoins)
              L.stop).nLeaves x * δ₁.toReal + 2 * (1 / 3) ^ t) +
          4 * t * 3 ^ t *
            (Fork CoinOpen.shaH ((((CoinLayer.session Loop.loopNr T).withStops
                (LeanRelease.rfOf (CoinLayer.session Loop.loopNr T) L pc fc Loop.readCoins) L.stop).swP vs).prover w) Vs θ x).Pr
              CoinOpen.shaH) := by
  subst hL
  exact Flock.zk_server_lean_coinLayer rel Loop.loopNr T vs t ht hT1 hδ₁ hS hV hD hc hsc hT _ wk x w hR
    (LeanRelease.loop_last Loop.loopNr T L₀ x) pc fc Loop.readCoins ⟨hPr, LeanRelease.loop_coinsRead Loop.loopNr T L₀ x⟩
    (LeanRelease.loop_helloIs Loop.loopNr T L₀ x)
    (LeanRelease.loop_helloParses Loop.loopNr T L₀ x) (LeanRelease.loop_reqsFollow T L₀ x) hLk hF hE hEf Vs θ hk Ev

end Claims.ZKServer
