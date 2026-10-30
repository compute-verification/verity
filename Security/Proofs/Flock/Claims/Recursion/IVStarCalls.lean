import Proofs.Flock.Claims.Recursion.AcceptedScope
import Proofs.Flock.Claims.Recursion.BoundRun
import Proofs.Flock.Claims.Recursion.BoundBridge
import Proofs.Flock.Claims.Recursion.SchedCount
import Proofs.Flock.Claims.Recursion.NextRRows
import Proofs.Flock.Claims.Bridge.Committed
import Proofs.Flock.Recursive.HoloRepCoins
import Proofs.Flock.Claims.Recursion.LastCallsR

/-!
# The inner verifier's own calls at the deployed bound record (`hU`)

`BitsLin2`'s `n` bounds the compiled record's inner-verifier calls (`Deploy.innerCalls`): `IVStar`'s run on the inner
messages the outer witnesses give, at any coins. Those are the verifier's own hash evaluations, not the prover's, so
they are counted here rather than assumed.

* `IVStar` calls nothing before its last round (`Bound.run_ivstar`). At its last it opens each earlier round's commit
  roundStrings, two calls each, as many as the schedule fixes (`Bound.nAt`), then replays the element-level verifier on the
  opened messages (`Bound.agrees`), stopping at the first step that answers other than the run's coin. An element-level
  step that answers a coin calls nothing, so the replay costs at most one step's calls (`Bound.agrees_calls_le`), and
  the run at most `2 · Σ_{j < k-1} nAt j + c_E` (`innerCallsOf_ivstar_le`).
* At the deployed schedule (`nAtSched`) round 0 carries each table's level-0 top, `2^{cap₀} ≤ 128` roundStrings
  (`level0Nodes_le`), and each later round at most one top of `2^{cap} ≤ 128` strings and one row
  (`repRounds_caps`); a rep has at most `2200` rounds (`repSchedAt_length_le`). At a plain session in scope
  (`PlainIn`: at most `200` tables, `k_log ≤ 27`, `1026` regions at two link points) the schedule's roundStrings number at
  most `2^27` (`sum_nAtSched_plainIn`).
* At the deployed plain verifier's steps a coin answer calls nothing (`plainNext_msg`) and a decision calls its setup's
  `setupCallsOf` and its verdict's `shaCallsOf` (`plainNext_calls_le`).

So at a statement the compiled verifier accepts, the inner-verifier calls are at most `2^28 + c_B + c_V`
(`innerCalls_plain_le_acc`), `c_B` bounding the inner setup's calls and `c_V` its verdict's at every oracle, both counts
of the verifier's own code; `lamLin2_plain_ge_acc` is `λ₂ ≥ 100` there wherever `2^28 + c_B + c_V ≤ 2^55`.
-/

namespace Claims.Recursion

open Claims

theorem bindOC_calls (H : Msg → Msg) {α β : Type} (p : OC α) (f : α → OC β) :
    ((bindOC p f).eval H).2.length = (p.eval H).2.length + ((f (p.eval H).1).eval H).2.length := by
  simp only [eval_bindOC_full, List.length_append]

theorem allOC_cons_calls (H : Msg → Msg) (p : OC Bool) (ps : List (OC Bool)) :
    ((allOC (p :: ps)).eval H).2.length =
      (p.eval H).2.length + (if (p.eval H).1 = true then ((allOC ps).eval H).2.length else 0) := by
  show ((bindOC p fun b => if b then allOC ps else .ret false).eval H).2.length = _
  rw [bindOC_calls]
  cases (p.eval H).1
  · first | rfl | simp [OC.eval]
  · first | rfl | simp [OC.eval]

/-- `allOC`'s calls are at most its programs'. -/
theorem allOC_calls_le_sum (H : Msg → Msg) : ∀ ps : List (OC Bool),
    ((allOC ps).eval H).2.length ≤ (ps.map fun p => (p.eval H).2.length).sum
  | [] => by simp [allOC, OC.eval]
  | p :: ps => by
    rw [allOC_cons_calls, List.map_cons, List.sum_cons]
    have ih := allOC_calls_le_sum H ps
    split <;> omega

/-- **`allOC` stops at the first `false`**: where each program calls at most `c` and each before the last that returns
`true` calls nothing, the whole calls at most `c`. -/
theorem allOC_calls_le (H : Msg → Msg) (c : ℕ) : ∀ ps : List (OC Bool),
    (∀ p ∈ ps, (p.eval H).2.length ≤ c) →
    (∀ (i : ℕ) (hi : i + 1 < ps.length), (ps[i].eval H).1 = true → (ps[i].eval H).2 = []) →
    ((allOC ps).eval H).2.length ≤ c
  | [], _, _ => by simp [allOC, OC.eval]
  | p :: ps, hc, h0 => by
    rw [allOC_cons_calls]
    have hp := hc p List.mem_cons_self
    split
    · rename_i hb
      by_cases hps : ps = []
      · subst hps
        simpa [allOC, OC.eval] using hp
      · have hp0 : (p.eval H).2 = [] :=
          h0 0 (by cases ps with
            | nil => exact absurd rfl hps
            | cons _ _ => simp) hb
        have ih := allOC_calls_le H c ps (fun q hq => hc q (List.mem_cons_of_mem _ hq))
          (fun i hi hq => h0 (i + 1) (by simp only [List.length_cons]; omega) hq)
        rw [hp0, List.length_nil]
        omega
    · omega

section Bound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

theorem Bound.opens_calls (Bd : Bound Coin Col T) (H : Msg → Msg) (a row : Msg) (y : Fin Bd.saltLen → UInt8) :
    ((Bd.opens a row y).eval H).2.length = 2 := by
  first | rfl | simp [Bound.opens, OC.eval]

theorem Bound.opensRound_calls_le (Bd : Bound Coin Col T) (H : Msg → Msg) (a : Msg)
    (os : List (Msg × (Fin Bd.saltLen → UInt8))) : ((Bd.opensRound a os).eval H).2.length ≤ 2 * os.length := by
  unfold Bound.opensRound
  split
  · refine (allOC_calls_le_sum H _).trans ((List.sum_le_card_nsmul _ 2 fun y hy => ?_).trans ?_)
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hy
      obtain ⟨i, hi, rfl⟩ := List.mem_mapIdx.1 hp
      exact le_of_eq (Bd.opens_calls H _ _ _)
    · first
      | (simp only [List.length_map, List.length_mapIdx, smul_eq_mul]; omega)
      | simp [Nat.mul_comm]
  · simp [OC.eval]

/-- Each earlier round's openings call twice a row. -/
theorem Bound.opensRounds_calls_le (Bd : Bound Coin Col T) (H : Msg → Msg) :
    ∀ (as : List Msg) (os : List (List (Msg × (Fin Bd.saltLen → UInt8)))),
      ((allOC ((as.zip os).map fun p => Bd.opensRound p.1 p.2)).eval H).2.length ≤ 2 * (os.map List.length).sum
  | [], _ => by simp [allOC, OC.eval]
  | _ :: _, [] => by simp [allOC, OC.eval]
  | a :: as, o :: os => by
    simp only [List.zip_cons_cons, List.map_cons, List.sum_cons, allOC_cons_calls]
    have h1 := Bound.opensRound_calls_le Bd H a o
    have h2 := Bound.opensRounds_calls_le Bd H as os
    split <;> omega

/-- **The element-level replay calls at most one step's calls**: each step before the last that matches the run's
answer answered a coin, which calls nothing (`h0`). -/
theorem Bound.agrees_calls_le (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (cE : ℕ)
    (hE : ∀ as cs, ((Bd.DE.V.next x as cs).eval H).2.length ≤ cE)
    (h0 : ∀ as cs m, ((Bd.DE.V.next x as cs).eval H).1 = .msg m → ((Bd.DE.V.next x as cs).eval H).2 = [])
    (ms : List Msg) (cs : List Coin) : ((Bd.agrees x ms cs).eval H).2.length ≤ cE := by
  unfold Bound.agrees
  refine allOC_calls_le H cE _ (fun p hp => ?_) (fun i hi hpt => ?_)
  · obtain ⟨j, -, rfl⟩ := List.mem_map.1 hp
    simp only [bindOC_calls, OC.eval, List.length_nil, Nat.add_zero]
    exact hE _ _
  · have hi' : i + 1 < ms.length := by simpa using hi
    simp only [List.getElem_map, List.getElem_range] at hpt ⊢
    rw [eval_bindOC_full] at hpt ⊢
    simp only [OC.eval, List.append_nil, decide_eq_true_eq, hi', ↓reduceIte] at hpt ⊢
    exact h0 _ _ _ hpt

theorem sum_map_length_eq {β : Type} : ∀ (f : ℕ → ℕ) (os : List (List β)),
    (∀ j < os.length, (os.getD j []).length = f j) → (os.map List.length).sum = ∑ j ∈ Finset.range os.length, f j
  | _, [], _ => by simp
  | f, o :: os, h => by
    have ih := sum_map_length_eq (fun j => f (j + 1)) os fun j hj => by
      simpa using h (j + 1) (by simp only [List.length_cons]; omega)
    have h0 : o.length = f 0 := by simpa using h 0 (by simp)
    rw [List.map_cons, List.sum_cons, List.length_cons, Finset.sum_range_succ', ih]
    try beta_reduce
    omega

/-- **`IVStar`'s step calls**: nothing before the last round; at the last, two a commit string the schedule fixes for
the earlier rounds, then at most one element-level step's. -/
theorem Bound.ivstar_calls_le (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (cE : ℕ)
    (hE : ∀ as cs, ((Bd.DE.V.next x as cs).eval H).2.length ≤ cE)
    (h0 : ∀ as cs m, ((Bd.DE.V.next x as cs).eval H).1 = .msg m → ((Bd.DE.V.next x as cs).eval H).2 = [])
    (as : List Msg) (cs : List Coin) :
    ((Bd.IVStar.next x as cs).eval H).2.length ≤ 2 * ∑ j ∈ Finset.range (as.length - 1), Bd.nAt x j + cE := by
  first
  | simp only [Bound.IVStar]
  | (unfold Bound.IVStar; dsimp only)
  split
  · simp [OC.eval]
  · rcases hf : Bd.final (as.getLastD []) with _ | ⟨os, last⟩
    · simp [OC.eval]
    · dsimp only
      split
      · rename_i hlen
        have hl1 := hlen.1
        have hA := Bd.opensRounds_calls_le H as.dropLast os
        have hS : (os.map List.length).sum = ∑ j ∈ Finset.range (as.length - 1), Bd.nAt x j := by
          rw [sum_map_length_eq (Bd.nAt x) os hlen.2, show os.length = as.length - 1 by omega]
        simp only [bindOC_calls]
        split
        · have hG := Bd.agrees_calls_le H x cE hE h0 (Bd.opened os last) cs
          simp only [bindOC_calls, OC.eval, List.length_nil, Nat.add_zero]
          omega
        · simp only [OC.eval, List.length_nil, Nat.add_zero]
          omega
      · simp [OC.eval]

end Bound

end Claims.Recursion

namespace RecDeploy

open Flock
open Claims (Msg Out Prover run coinSeq)
open FlockDeploy (Coin Ins)

/-- **`IVStar`'s calls on the inner messages, at any coins**: at most two a commit string the schedule fixes before the
last round, and one element-level step's. -/
theorem innerCallsOf_ivstar_le {Col T : Type} [DecidableEq Col] (Bd : Claims.Recursion.Bound Coin Col T)
    (H : Msg → Msg) (rd : Readers) (x : Msg) (cE : ℕ)
    (hE : ∀ as cs, ((Bd.DE.V.next (xIOf x) as cs).eval H).2.length ≤ cE)
    (h0 : ∀ as cs m, ((Bd.DE.V.next (xIOf x) as cs).eval H).1 = .msg m →
      ((Bd.DE.V.next (xIOf x) as cs).eval H).2 = [])
    (c : ℕ → Coin) (ws : List Claims.Flock.Wit) :
    (innerCallsOf H Bd.IVStar rd x c ws).length ≤
      2 * ∑ j ∈ Finset.range (Bd.DE.V.k (xIOf x) - 1), Bd.nAt (xIOf x) j + cE := by
  unfold innerCallsOf
  rw [List.length_map]
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k (xIOf x)) with hk | hk
  · rw [Bd.run_ivstar_zero H _ _ _ hk]
    exact Nat.zero_le _
  · rw [Bd.run_ivstar H _ _ _ hk]
    have h := Bd.ivstar_calls_le H (xIOf x) cE hE h0
      ((List.range (Bd.DE.V.k (xIOf x))).map fun i => (rd.msgs x ws).getD i [])
      ((List.range (Bd.DE.V.k (xIOf x))).map
        (coinSeq fun i : Fin (Bd.IVStar.k (xIOf x)) => c i.val))
    rw [List.length_map, List.length_range] at h
    exact h

/-! ## The deployed schedule's commit roundStrings -/

/-- A table's level-0 top: `2^{cap₀} ≤ 2^7` nodes. -/
theorem level0Nodes_le (st : Stmt) : level0Nodes st ≤ 128 := by
  unfold level0Nodes
  cases hf : fast100 st.m with
  | none => simp
  | some s =>
    have hb := FlockDeploy.Rows.fast100_bound hf
    show 2 ^ (s.levels.getD 0 default).cap ≤ 128
    calc 2 ^ (s.levels.getD 0 default).cap ≤ 2 ^ 7 := Nat.pow_le_pow_right (by norm_num) (hb.2.2.2 0 (by norm_num))
      _ = 128 := by norm_num

/-- **A rep's rounds commit at most one top of at most `2^7` nodes each.** -/
theorem repRounds_caps {m kLog claims : ℕ} {out : Array RecVStar.SchedRound}
    (h : RecVStar.repRounds m kLog claims = some out) : ∀ r ∈ out, r.2.1.toList.sum ≤ 128 := by
  unfold RecVStar.repRounds at h
  obtain ⟨sch, hf, h⟩ := Flock.Guarantees.HoloRepCoins.obind_some h
  have hb := FlockDeploy.Rows.fast100_bound hf
  have hcapU : ∀ i, 2 ^ (sch.levels.getD i default).cap ≤ 128 := by
    intro i
    have h7 : (sch.levels.getD i default).cap ≤ 7 := by
      by_cases hi : i < 7
      · exact hb.2.2.2 i hi
      · have hsz : sch.levels.size ≤ i := by have := hb.1; omega
        first
        | (unfold Array.getD; rw [dif_neg (by omega)]; first | decide | rfl | simp)
        | (rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none hsz]; first | decide | rfl | simp)
        | simp [Array.getD_eq_getD_getElem?, hsz]
    calc 2 ^ (sch.levels.getD i default).cap ≤ 2 ^ 7 := Nat.pow_le_pow_right (by norm_num) h7
      _ = 128 := by norm_num
  have hnil : ∀ (n z : ℕ) (b : Bool), ((n, #[], z, b) : RecVStar.SchedRound).2.1.toList.sum ≤ 128 :=
    fun _ _ _ => by simp
  have hcap : ∀ (n i z : ℕ) (b : Bool),
      ((n, #[2 ^ (sch.levels.getD i default).cap], z, b) : RecVStar.SchedRound).2.1.toList.sum ≤ 128 :=
    fun _ i _ _ => by simpa using hcapU i
  try dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  obtain ⟨st, hst, h⟩ := Flock.Guarantees.HoloRepCoins.obind_some h
  first
  | cases h
  | (simp only [pure, Option.some.injEq] at h; subst h)
  | (simp at h; subst h)
  refine Flock.Guarantees.HoloRepCoins.oforIn_inv hst (fun _ => True)
    (fun st => ∀ x ∈ st.1, x.2.1.toList.sum ≤ 128) (fun _ _ => trivial) ?_ ?_
  · intro a _ r y hr hy
    have hr' : ∀ x ∈ r.1, x.2.1.toList.sum ≤ 128 := hr
    try dsimp only at hy
    repeat' (first | (obtain ⟨_, _, hy⟩ := Flock.Guarantees.HoloRepCoins.obind_some hy) | (split at hy))
    all_goals first
      | contradiction
      | (simp at hy; done)
      | (first
         | cases hy
         | (injection hy with hy; subst hy)
         | (simp only [pure, Option.some.injEq] at hy; subst hy)
         refine ⟨_, rfl, ?_⟩
         intro x hx
         try dsimp only at hx
         simp only [Array.mem_append, Array.mem_replicate, List.mem_toArray, List.mem_cons, List.not_mem_nil,
           or_false] at hx
         repeat' (first | (refine Or.elim hx (fun hx => ?_) (fun hx => ?_)) | (replace hx := hx.2))
         all_goals first
           | exact hr' x hx
           | (subst hx
              first | exact hnil _ _ _ | exact hcap _ _ _ _ | (simp; done) | simpa using hcapU _))
  · intro x hx
    try dsimp only at hx
    simp only [Array.mem_append, Array.mem_replicate, List.mem_toArray, List.mem_cons, List.not_mem_nil,
      or_false] at hx
    repeat' (first | (refine Or.elim hx (fun hx => ?_) (fun hx => ?_)) | (replace hx := hx.2))
    all_goals
      subst hx
      first | exact hnil _ _ _ | exact hcap _ _ _ _ | (simp; done)

/-- **A rep's rounds**, at `k_log ≤ 27` and at most `2054` claims: at most `2200`. -/
theorem repSchedAt_length_le (spec : SessionSpec) (st : Stmt) (hk : st.c.kLog ≤ 27)
    (hc : 2 + spec.points * st.regions.size ≤ 2054) : (repSchedAt spec st).length ≤ 2200 := by
  unfold repSchedAt
  cases h : RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size) with
  | none => simp
  | some rs =>
    obtain ⟨sch, hs, hsz⟩ := repRounds_size h
    obtain ⟨-, hm, -, -, -⟩ := fast100_sizes hs
    have hb := FlockDeploy.Rows.fast100_bound hs
    have hloop : ((List.range' 0 (sch.levels.size - 1)).map (loopG (sch.levels.size - 1))).sum ≤ 66 := by
      have := List.sum_le_card_nsmul ((List.range' 0 (sch.levels.size - 1)).map (loopG (sch.levels.size - 1))) 11
        (fun y hy => by
          obtain ⟨i, -, rfl⟩ := List.mem_map.1 hy
          unfold loopG
          split <;> omega)
      simp only [List.length_map, List.length_range', smul_eq_mul] at this
      have h7 := hb.1
      omega
    show rs.toList.length ≤ 2200
    rw [Array.length_toList]
    have hk0 := hb.2.2.1
    generalize spec.points * st.regions.size = c at hc hsz
    omega

/-- The schedule's recorded rounds in order: each table's two reps. -/
def schedRoundsAll (spec : SessionSpec) (sts : Array Stmt) : List RecVStar.SchedRound :=
  sts.toList.flatMap fun st => repSchedAt spec st ++ repSchedAt spec st

/-- A round's commit strings: its tops' cap nodes, then one row where it absorbs. -/
def roundStrings (r : RecVStar.SchedRound) : ℕ := r.2.1.toList.sum + if r.2.2.2 then 1 else 0

theorem roundStrings_le {spec : SessionSpec} {sts : Array Stmt} : ∀ r ∈ schedRoundsAll spec sts, roundStrings r ≤ 129 := by
  intro r hr
  unfold schedRoundsAll at hr
  obtain ⟨st, -, hr⟩ := List.mem_flatMap.1 hr
  have hr' : r ∈ repSchedAt spec st := by
    rcases List.mem_append.1 hr with h | h <;> exact h
  unfold repSchedAt at hr'
  cases h : RecVStar.repRounds st.m st.c.kLog (2 + spec.points * st.regions.size) with
  | none => rw [h] at hr'; simp at hr'
  | some out =>
    rw [h] at hr'
    have hmem : r ∈ out := Array.mem_toList_iff.1 hr'
    have := repRounds_caps h r hmem
    unfold roundStrings
    split <;> omega

theorem schedRoundsAll_length_le {spec : SessionSpec} {sts : Array Stmt}
    (hk : ∀ st ∈ sts.toList, (repSchedAt spec st).length ≤ 2200) : (schedRoundsAll spec sts).length ≤ 4400 * sts.size := by
  unfold schedRoundsAll
  rw [List.length_flatMap]
  have := List.sum_le_card_nsmul (sts.toList.map fun st => (repSchedAt spec st ++ repSchedAt spec st).length) 4400
    (fun y hy => by
      obtain ⟨st, hst, rfl⟩ := List.mem_map.1 hy
      rw [List.length_append]
      have := hk st hst
      omega)
  simp only [List.length_map, Array.length_toList, smul_eq_mul] at this
  omega

theorem nAtSched_zero {x : Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : FlockDeploy.setupOf false x = some (s, spec, sts)) :
    nAtSched x 0 = (sts.toList.map level0Nodes).sum := by
  first
  | (unfold nAtSched; rw [hset]; rfl)
  | (simp only [nAtSched, hset, eq_self_iff_true, ↓reduceIte]; done)
  | simp [nAtSched, hset]

theorem nAtSched_succ {x : Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : FlockDeploy.setupOf false x = some (s, spec, sts)) (i : ℕ) :
    nAtSched x (i + 1) = ((schedRoundsAll spec sts)[i]?.map roundStrings).getD 0 := by
  simp only [nAtSched, hset, Nat.add_one_ne_zero, ↓reduceIte, Nat.add_sub_cancel, schedRoundsAll]
  first
  | (cases (sts.toList.flatMap fun st => repSchedAt spec st ++ repSchedAt spec st)[i]? <;> rfl)
  | (split <;> simp_all [roundStrings])

theorem sum_getD_roundStrings_le (L : List RecVStar.SchedRound) (hL : ∀ r ∈ L, roundStrings r ≤ 129) :
    ∀ N, ∑ i ∈ Finset.range N, (L[i]?.map roundStrings).getD 0 ≤ 129 * min N L.length
  | 0 => by simp
  | N + 1 => by
    rw [Finset.sum_range_succ]
    have ih := sum_getD_roundStrings_le L hL N
    by_cases hN : N < L.length
    · rw [List.getElem?_eq_getElem hN, Option.map_some, Option.getD_some]
      have h1 := hL _ (List.getElem_mem hN)
      have e1 : min N L.length = N := by omega
      have e2 : min (N + 1) L.length = N + 1 := by omega
      rw [e1] at ih
      rw [e2]
      omega
    · rw [List.getElem?_eq_none (by omega), Option.map_none, Option.getD_none]
      have e1 : min N L.length = L.length := by omega
      have e2 : min (N + 1) L.length = L.length := by omega
      rw [e1] at ih
      rw [e2]
      omega

/-- **The schedule's commit strings over any rounds**: each table's level-0 top, and at most `129` a recorded round. -/
theorem sum_nAtSched_le {x : Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : FlockDeploy.setupOf false x = some (s, spec, sts)) (N : ℕ) :
    ∑ j ∈ Finset.range N, nAtSched x j ≤ 128 * sts.size + 129 * (schedRoundsAll spec sts).length := by
  have h0 : nAtSched x 0 ≤ 128 * sts.size := by
    rw [nAtSched_zero hset]
    have := List.sum_le_card_nsmul (sts.toList.map level0Nodes) 128 (fun y hy => by
      obtain ⟨st, -, rfl⟩ := List.mem_map.1 hy
      exact level0Nodes_le st)
    simp only [List.length_map, Array.length_toList, smul_eq_mul] at this
    omega
  cases N with
  | zero => simp
  | succ N =>
    rw [Finset.sum_range_succ']
    have h1 : ∑ i ∈ Finset.range N, nAtSched x (i + 1) ≤ 129 * (schedRoundsAll spec sts).length := by
      rw [Finset.sum_congr rfl fun i _ => nAtSched_succ hset i]
      exact (sum_getD_roundStrings_le _ roundStrings_le N).trans (Nat.mul_le_mul_left _ (Nat.min_le_right _ _))
    omega

/-- **At a plain session in scope the schedule's commit roundStrings number at most `2^27`**: `200` tables, each a level-0
top of `128` and two reps of at most `2200` rounds, `129` roundStrings each. -/
theorem sum_nAtSched_plainIn {x : Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hin : FlockDeploy.PlainIn x s spec sts) (N : ℕ) : ∑ j ∈ Finset.range N, nAtSched x j ≤ 134217728 := by
  have h1 := sum_nAtSched_le hin.setup N
  have hlen := schedRoundsAll_length_le (spec := spec) (sts := sts) fun st hst => by
    obtain ⟨-, -, hr⟩ := hin.tab st hst
    have hr' : st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1026 ∧ spec.mPts ≤ 64 := hr
    have hp : spec.points * st.regions.size ≤ 2 * 1026 := Nat.mul_le_mul hin.points hr'.2.1
    exact repSchedAt_length_le spec st hr'.1 (by omega)
  have hsz := hin.size_le
  omega

/-! ## At the deployed plain verifier's steps -/

/-- **A plain step that answers a coin calls nothing.** -/
theorem plainNext_msg (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List Coin) (m : Msg)
    (h : ((FlockDeploy.next false x as cs).eval H).1 = .msg m) : ((FlockDeploy.next false x as cs).eval H).2 = [] := by
  cases ha : FlockDeploy.act false x as cs with
  | rej => rw [FlockDeploy.next_rej H ha]
  | coin c => rw [FlockDeploy.next_coin H ha]
  | decide s spec sts link c₁ mids f =>
    exfalso
    exact Claims.Bridge.outOf_ne_msg H (FlockDeploy.decision false s x link c₁ mids f) m
      (by unfold FlockDeploy.next at h; rw [ha] at h; exact h)

/-- **A plain step's calls**: a decision's setup calls, then its verdict's. -/
theorem plainNext_calls_le (H : Msg → Msg) (x : Msg) (cB cV : ℕ)
    (hB : ∀ s spec sts, FlockDeploy.setupOf false x = some (s, spec, sts) →
      FlockDeploy.setupCallsOf (Claims.hOf H) s.read ≤ cB)
    (hV : ∀ s spec sts, FlockDeploy.setupOf false x = some (s, spec, sts) → ∀ spec' sts' record proofs,
      FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
      FlockDeploy.shaCallsOf false (Claims.hOf H) spec' (FlockDeploy.tabsOf sts') record proofs ≤ cV)
    (as : List Msg) (cs : List Coin) : ((FlockDeploy.next false x as cs).eval H).2.length ≤ cB + cV := by
  cases ha : FlockDeploy.act false x as cs with
  | rej => rw [FlockDeploy.next_rej H ha]; exact Nat.zero_le _
  | coin c => rw [FlockDeploy.next_coin H ha]; exact Nat.zero_le _
  | decide s spec sts link c₁ mids f =>
    have hs := FlockDeploy.act_decide ha
    rw [FlockDeploy.next_decide_log H ha]
    have h1 := hB s spec sts hs
    rcases hw : FlockDeploy.setupWith (Claims.hOf H) s.read with e | ⟨spec', sts'⟩
    · first
      | (show FlockDeploy.setupCallsOf (Claims.hOf H) s.read + 0 ≤ cB + cV; omega)
      | (simp only [hw]; omega)
      | (simp [hw]; omega)
    · have h2 := hV s spec sts hs spec' sts' f.record f.proofs hw
      first
      | (show FlockDeploy.setupCallsOf (Claims.hOf H) s.read +
          FlockDeploy.shaCallsOf false (Claims.hOf H) spec' (FlockDeploy.tabsOf sts') f.record f.proofs ≤ cB + cV
         omega)
      | (simp only [hw]; omega)
      | (simp [hw]; omega)

/-! ## At an accepted statement -/

/-- **`hU` at the deployed bound record, at an accepted statement**: on a run the compiled verifier accepts, the
record's inner-verifier calls at any coins and witnesses are at most `2^28` openings' and one element-level step's
(`c_E`), wherever the element-level steps answer a coin with no call (`h0`). -/
theorem innerCalls_deployedB_le_acc (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (cE : ℕ) (hE : ∀ as cs, ((nextE (xIOf x) as cs).eval H).2.length ≤ cE)
    (h0 : ∀ as cs m, ((nextE (xIOf x) as cs).eval H).1 = .msg m → ((nextE (xIOf x) as cs).eval H).2 = []) :
    ∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤ 2 ^ 28 + cE := by
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_run_acc hacc
  intro cs ws
  have h := innerCallsOf_ivstar_le (bdDeployed rel nextE colsE wE join rowIn final w) H L.readers x cE hE h0 cs ws
  have hs : ∑ j ∈ Finset.range ((bdDeployed rel nextE colsE wE join rowIn final w).DE.V.k (xIOf x) - 1),
      (bdDeployed rel nextE colsE wE join rowIn final w).nAt (xIOf x) j ≤ 134217728 :=
    sum_nAtSched_plainIn hin _
  have h28 : (2 : ℕ) ^ 28 = 268435456 := by norm_num
  refine le_trans h ?_
  omega

/-- **`hU` at the deployed plain verifier's steps, at an accepted statement**: `innerCalls_deployedB_le_acc` with
`c_E` a decision's setup calls (`c_B`) and verdict calls (`c_V`), both bounds on the verifier's own code at `H`. -/
theorem innerCalls_plain_le_acc (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (cB cV : ℕ)
    (hB : ∀ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) →
      FlockDeploy.setupCallsOf (Claims.hOf H) s.read ≤ cB)
    (hV : ∀ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) → ∀ spec' sts' record proofs,
      FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
      FlockDeploy.shaCallsOf false (Claims.hOf H) spec' (FlockDeploy.tabsOf sts') record proofs ≤ cV) :
    ∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤
        2 ^ 28 + (cB + cV) :=
  innerCalls_deployedB_le_acc rel b L (FlockDeploy.V false).next colsE wE join rowIn final w H hacc (cB + cV)
    (fun as cs => plainNext_calls_le H (xIOf x) cB cV hB hV as cs)
    (fun as cs m hm => plainNext_msg H (xIOf x) as cs m hm)

/-- **Route 2C's `λ₂ ≥ 100` at the deployed plain steps, at an accepted statement, with `hU` counted**:
`lamLin2_deployedB_ge_acc` at `n = 2^28 + c_B + c_V` (`innerCalls_plain_le_acc`), wherever that is at most `2^55`.
What is left is `hS` and the two counts of the verifier's own code, `c_B` (its inner setup's calls at `H`) and `c_V`
(its verdict's). -/
theorem lamLin2_plain_ge_acc (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) (ps : List (Msg × Out)) (cB cV : ℕ)
    (hB : ∀ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) →
      FlockDeploy.setupCallsOf (Claims.hOf H) s.read ≤ cB)
    (hV : ∀ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) → ∀ spec' sts' record proofs,
      FlockDeploy.setupWith (Claims.hOf H) s.read = .ok (spec', sts') →
      FlockDeploy.shaCallsOf false (Claims.hOf H) spec' (FlockDeploy.tabsOf sts') record proofs ≤ cV)
    (hc : 2 ^ 28 + (cB + cV) ≤ 2 ^ 55) :
    (∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤
        2 ^ 28 + (cB + cV)) ∧
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record
        (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).deploy
        decI)
      FlockDeploy.KS.CpDep x (2 ^ 28 + (cB + cV)) (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L
        (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).IVStar).vstar x ps)).length ≤
        2 ^ 55 :=
  ⟨innerCalls_plain_le_acc rel b L colsE wE join rowIn final w H hacc cB cV hB hV,
    lamLin2_deployedB_ge_acc rel b L (FlockDeploy.V false).next colsE wE join rowIn final w decI H hacc hS ps hc⟩

/-! ## At the deployed reader's steps (`nextR`, the element-level steps KS's bound record reads) -/

/-- **A step of `nextR` that answers a coin calls nothing.** -/
theorem nextR_msg (rd : Msg → Option FlockDeploy.FinalP) (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List Coin)
    (m : Msg) (h : ((FlockDeploy.nextR rd x as cs).eval H).1 = .msg m) :
    ((FlockDeploy.nextR rd x as cs).eval H).2 = [] := by
  cases ha : FlockDeploy.actR rd x as cs with
  | rej => first | rfl | simp [FlockDeploy.nextR, ha, Claims.OC.eval]
  | coin c => first | rfl | simp [FlockDeploy.nextR, ha, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    exfalso
    exact Claims.Bridge.outOf_ne_msg H (FlockDeploy.decisionSF s x link c₁ mids f) m
      (by rw [FlockDeploy.nextR_decide ha] at h; exact h)

/-- **A step of `nextR`'s calls**: a decision's setup calls at `H` (`c_B`), then its verdict's at whatever session the
setup returns there (`c_V`: the files' digests and the plain verdict at the reader's record). -/
theorem nextR_calls_le (rd : Msg → Option FlockDeploy.FinalP) (H : Msg → Msg) (x : Msg) (cB cV : ℕ)
    (hB : ∀ v, FlockDeploy.setupOf false x = some v → FlockDeploy.setupCallsOf (Claims.hOf H) v.1.read ≤ cB)
    (hV : ∀ s spec sts link c₁ mids f,
      ((FlockDeploy.decisionAtS s spec sts x link c₁ mids f).run.eval H).2.length ≤ cV)
    (as : List Msg) (cs : List Coin) : ((FlockDeploy.nextR rd x as cs).eval H).2.length ≤ cB + cV := by
  cases ha : FlockDeploy.actR rd x as cs with
  | rej => simp [FlockDeploy.nextR, ha, Claims.OC.eval]
  | coin c => simp [FlockDeploy.nextR, ha, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨v, _, _, _, _, _, _, hset, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some (FlockDeploy.actR_decide ha)
    have hs : s = v.1 := congrArg FlockDeploy.DecR.s hd
    have h1 := hB v hset
    rw [← hs] at h1
    have hl := (FlockDeploy.setupM_eval H s.read).2
    have key : ((FlockDeploy.nextR rd x as cs).eval H).2 = ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).2 ++
        (match ((FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H).1 with
         | .ok w => ((FlockDeploy.decisionAtS s w.1 w.2 x link c₁ mids f).run.eval H).2
         | .error _ => []) := by
      simp only [FlockDeploy.nextR, ha, FlockDeploy.KS.outOf_log, FlockDeploy.decisionSF, ExceptT.run_bind,
        Claims.OC.bind_eq, Claims.OC.eval_bind]
      rcases hv : (FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
      cases r with
      | error e => first | rfl | simp [Claims.OC.eval]
      | ok w =>
        obtain ⟨a, b⟩ := w
        first | rfl | simp [Claims.OC.eval]
    rw [key, List.length_append, hl]
    split
    · rename_i w _
      have h2 := hV s w.1 w.2 link c₁ mids f
      omega
    · first | omega | (simp only [List.length_nil, Nat.add_zero]; omega) | (simp; omega)

/-- **`hU` at KS's bound record, at an accepted statement**: `innerCalls_deployedB_le_acc` at the deployed reader's
steps `nextR rd`, with `c_E` a decision's setup calls (`c_B`) and verdict calls (`c_V`). -/
theorem innerCalls_R_le_acc (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (rd : Msg → Option FlockDeploy.FinalP)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (cB cV : ℕ)
    (hB : ∀ v, FlockDeploy.setupOf false (xIOf x) = some v → FlockDeploy.setupCallsOf (Claims.hOf H) v.1.read ≤ cB)
    (hV : ∀ s spec sts link c₁ mids f,
      ((FlockDeploy.decisionAtS s spec sts (xIOf x) link c₁ mids f).run.eval H).2.length ≤ cV) :
    ∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤
        2 ^ 28 + (cB + cV) :=
  innerCalls_deployedB_le_acc rel b L (FlockDeploy.nextR rd) colsE wE join rowIn final w H hacc (cB + cV)
    (fun as cs => nextR_calls_le rd H (xIOf x) cB cV hB hV as cs)
    (fun as cs m hm => nextR_msg rd H (xIOf x) as cs m hm)

/-- **Route 2C's `λ₂ ≥ 100` at KS's bound record (the deployed reader's steps), at an accepted statement, with `hU`
counted**: `lamLin2_deployedB_ge_acc` at `nextE := nextR rd` and `n = 2^28 + c_B + c_V`, wherever that is at most `2^55`.
What is left is `hS` and the two counts of the verifier's own code, `c_B` and `c_V`. -/
theorem lamLin2_R_ge_acc (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (rd : Msg → Option FlockDeploy.FinalP)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {IV : Claims.Verifier Coin} {H' : Msg → Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
    {r : Fin ((V IV (deployedB b).st).k x) → Coin} (hacc : (run H' P (V IV (deployedB b).st) x ω r).out = .acc)
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) (ps : List (Msg × Out)) (cB cV : ℕ)
    (hB : ∀ v, FlockDeploy.setupOf false (xIOf x) = some v → FlockDeploy.setupCallsOf (Claims.hOf H) v.1.read ≤ cB)
    (hV : ∀ s spec sts link c₁ mids f,
      ((FlockDeploy.decisionAtS s spec sts (xIOf x) link c₁ mids f).run.eval H).2.length ≤ cV)
    (hc : 2 ^ 28 + (cB + cV) ≤ 2 ^ 55) :
    (∀ cs ws, (((recBoundAt rel (deployedB b) L).record
      (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).innerCalls H x cs ws).length ≤
        2 ^ 28 + (cB + cV)) ∧
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record
        (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x (2 ^ 28 + (cB + cV)) (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L
        (bdDeployed rel (FlockDeploy.nextR rd) colsE wE join rowIn final w).IVStar).vstar x ps)).length ≤
        2 ^ 55 :=
  ⟨innerCalls_R_le_acc rel b L rd colsE wE join rowIn final w H hacc cB cV hB hV,
    lamLin2_deployedB_ge_acc rel b L (FlockDeploy.nextR rd) colsE wE join rowIn final w decI H hacc hS ps hc⟩

end RecDeploy
