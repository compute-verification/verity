import Proofs.Flock.C3.Reps.RoundSel

/-!
# A rep's message rounds along `holoRepShape`

`holoSel_parts`' `hm` puts message `m` of a rep's kinds `σ := holoRepShape …` at the reps' round `(msgRounds rs)[m]`:
`roundOf σ m = heldBefore (.rep 0) rs ((msgRounds rs)[m])`, at `holoRounds`' `rs`. It holds at the messages of the
zerocheck, the lincheck and the opening, V*'s parts' (`m + kLog + nExtra - 6` of them at `nExtra + 2` claims), and not
past them: compiled Ligerito's rounds absorb its tops, which `ligShape` has no message for, and a round that commits a
top and absorbs elements is two messages there.

* `roundOf_eq_skel`: `roundOf` reads only which kinds are messages (`isMsg`).
* `roundOfB_rskel`: along a round list's messages and coins in its order (`rskel`: a round's message, then its coins as
  one step, as `heldBefore` groups a rep's rounds), message `i`'s round is the coin rounds before its round.
* `skel_head`: the zerocheck's, lincheck's and opening's kinds are `holoRounds`' head in that order (`holoHead`,
  `repHead` with z_partial's round split), by replicate algebra.
* `roundOf_holoRepShape`: `hm`'s equation at each of those messages, at any Ligerito schedule (`roundOfB_append`: a
  message of the head is placed by the head alone).
-/

namespace FlockC3.Reps.RoundOfHolo

open FlockSoundness.Discharge.Exec (Kind zcShape opShape ligShape pairRounds switches)
open Flock.RecVStar (SchedRound)
open Flock.Guarantees (holoRounds msgRounds heldBefore HoloStream zpRound)
open FlockC3.Compose (repHead repRounds_tail repRounds_m)
open Flock.Guarantees.HoloRepCoins (obind_some)

/-! ## Skeletons -/

/-- Whether a kind is a message. -/
def isMsg : Kind → Bool
  | .msg _ => true
  | .coin _ => false

/-- `roundOf` on a skeleton: the coins (`false`) before message `i` (the `i`-th `true`). -/
def roundOfB : List Bool → ℕ → ℕ
  | [], _ => 0
  | true :: _, 0 => 0
  | true :: s, i + 1 => roundOfB s i
  | false :: s, i => roundOfB s i + 1

theorem roundOf_eq_skel : ∀ (σ : List Kind) (i : ℕ), FlockC3.Sparse.roundOf σ i = roundOfB (σ.map isMsg) i
  | [], _ => rfl
  | .msg _ :: _, 0 => rfl
  | .msg _ :: σ, i + 1 => roundOf_eq_skel σ i
  | .coin _ :: σ, i => congrArg (· + 1) (roundOf_eq_skel σ i)

/-- A round's steps: its message, if it absorbs one, then its coins, as one step, if it draws any. -/
def rstep (q : SchedRound) : List Bool := (if q.2.2.2 then [true] else []) ++ (if q.1 = 0 then [] else [false])

/-- A round list's steps in its order. -/
def rskel (l : List SchedRound) : List Bool := l.flatMap rstep

/-! ## Message rounds on lists -/

/-- `msgRounds` on a list. -/
def msgL : List SchedRound → List ℕ
  | [] => []
  | q :: l => (if q.2.2.2 then [0] else []) ++ (msgL l).map (· + 1)

/-- `heldBefore` at a rep's stream on a list. -/
def heldL : List SchedRound → ℕ → ℕ
  | [], _ => 0
  | _ :: _, 0 => 0
  | q :: l, k + 1 => min q.1 1 + heldL l k

theorem filter_range_succ (n : ℕ) (p : ℕ → Bool) :
    (List.range (n + 1)).filter p =
      (if p 0 then [0] else []) ++ ((List.range n).filter (p ∘ (· + 1))).map (· + 1) := by
  rw [List.range_succ_eq_map, List.filter_cons, List.filter_map]
  split <;> rfl

theorem msgRounds_toArray : ∀ l : List SchedRound, msgRounds l.toArray = msgL l
  | [] => rfl
  | q :: l => by
    have ih := msgRounds_toArray l
    unfold msgRounds at ih ⊢
    rw [List.size_toArray, List.length_cons, filter_range_succ, msgL, ← ih, List.size_toArray]
    simp [Function.comp_def]

theorem heldBefore_toArray : ∀ (l : List SchedRound) (k : ℕ), heldBefore (.rep 0) l.toArray k = heldL l k
  | [], k => by simp [heldBefore, heldL]
  | _ :: _, 0 => by simp [heldBefore, heldL]
  | q :: l, k + 1 => by
    have ih := heldBefore_toArray l k
    unfold heldBefore at ih ⊢
    rw [List.size_toArray, List.length_cons, Nat.min_def] at *
    rw [heldL, ← ih]
    split <;> split <;> first
      | omega
      | (rw [List.range_succ_eq_map, List.map_cons, List.sum_cons, List.map_map]
         simp [HoloStream.group, Function.comp_def, Nat.min_def])

/-- **Message `i`'s round is the coin rounds before its round**, along a round list's steps. -/
theorem roundOfB_rskel : ∀ (l : List SchedRound) (i : ℕ) (hi : i < (msgL l).length),
    roundOfB (rskel l) i = heldL l ((msgL l)[i]'hi)
  | [], _, hi => absurd hi (by simp [msgL])
  | q :: l, i, hi => by
    have hc : ∀ s : List Bool, ∀ j, roundOfB ((if q.1 = 0 then [] else [false]) ++ s) j =
        roundOfB s j + min q.1 1 := by
      intro s j
      by_cases h0 : q.1 = 0
      · simp [h0]
      · simp [h0, roundOfB, Nat.min_def]
        omega
    have hsk : rskel (q :: l) =
        (if q.2.2.2 then [true] else []) ++ ((if q.1 = 0 then [] else [false]) ++ rskel l) := by
      simp only [rskel, List.flatMap_cons, rstep, List.append_assoc]
    rw [hsk]
    by_cases ha : q.2.2.2 = true
    · have hm : msgL (q :: l) = 0 :: (msgL l).map (· + 1) := by simp [msgL, ha]
      simp only [ha, ↓reduceIte, List.singleton_append]
      cases i with
      | zero => simp [roundOfB, hm, heldL]
      | succ j =>
        have hj : j < (msgL l).length := by simp [hm] at hi; omega
        simp only [roundOfB]
        rw [hc, roundOfB_rskel l j hj]
        simp only [hm, List.getElem_cons_succ, List.getElem_map, heldL]
        omega
    · have hm : msgL (q :: l) = (msgL l).map (· + 1) := by simp [msgL, ha]
      have hj : i < (msgL l).length := by simpa [hm] using hi
      simp only [ha, Bool.false_eq_true, ↓reduceIte, List.nil_append]
      rw [hc, roundOfB_rskel l i hj]
      simp only [hm, List.getElem_map, heldL]
      omega

/-! ## `holoRounds`' steps -/

/-- `holoRounds`' head: `repHead` with z_partial's round split into one that absorbs it and one that absorbs h and
draws `r′`. -/
def holoHead (m kLog claims : ℕ) : List SchedRound :=
  [(6, #[], 1, false), (m - 13, #[], 0, false), (1, #[], 0, true)] ++ List.replicate (m - 5) (1, #[], 0, true) ++
    [(1, #[], 0, false)] ++ List.replicate (kLog - 6) (1, #[], 0, true) ++ [(0, #[], 0, true), (1, #[], 0, true)] ++
    List.replicate claims (7, #[], 0, true) ++ [(claims, #[], 0, false)]

/-- `repHead` before z_partial's round. -/
def zpPre (m kLog : ℕ) : List SchedRound :=
  [(6, #[], 1, false), (m - 13, #[], 0, false), (1, #[], 0, true)] ++ List.replicate (m - 5) (1, #[], 0, true) ++
    [(1, #[], 0, false)] ++ List.replicate (kLog - 6) (1, #[], 0, true)

theorem zpPre_length {m kLog : ℕ} (hm : 6 ≤ m) (hk : 6 ≤ kLog) : (zpPre m kLog).length = zpRound m kLog := by
  simp [zpPre, zpRound]
  omega

theorem repHead_toList (m kLog claims : ℕ) :
    (repHead m kLog claims).toList = zpPre m kLog ++ (1, #[], 0, true) ::
      (List.replicate claims (7, #[], 0, true) ++ [(claims, #[], 0, false)]) := by
  simp [repHead, zpPre]

/-- **`holoRounds` is `holoHead` and `repRounds`' tail**, at `fast100`'s tables. -/
theorem holoRounds_toList {m kLog claims : ℕ} {rs : Array SchedRound} (hk : 6 ≤ kLog)
    (h : holoRounds m kLog claims = some rs) :
    22 ≤ m ∧ ∃ T : List SchedRound, rs.toList = holoHead m kLog claims ++ T := by
  unfold holoRounds at h
  obtain ⟨out, hout, h⟩ := obind_some h
  obtain ⟨T, rfl, -⟩ := repRounds_tail hout
  have hm := repRounds_m hout
  refine ⟨hm.1, T.toList, ?_⟩
  dsimp only at h
  split at h
  · cases h
  cases h
  have hl := zpPre_length (m := m) (kLog := kLog) (by omega) hk
  have hA : (repHead m kLog claims ++ T).toList = zpPre m kLog ++ ((1, #[], 0, true) ::
      (List.replicate claims (7, #[], 0, true) ++ [(claims, #[], 0, false)] ++ T.toList)) := by
    rw [Array.toList_append, repHead_toList]
    simp
  simp only [Array.toList_append, Array.toList_extract, hA, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero,
    ← hl, Array.size_append]
  have hd : ∀ (P R : List SchedRound) (x : SchedRound), (P ++ x :: R).drop (P.length + 1) = R := by
    intro P R x
    induction P <;> simp_all
  have hs : (repHead m kLog claims).size = (zpPre m kLog).length + 1 + (claims + 1) := by
    rw [← Array.length_toList, repHead_toList]
    simp
    omega
  rw [List.take_left' rfl, hd, List.take_of_length_le (by simp [hs]; omega)]
  simp [holoHead, zpPre]

/-! ## `holoRepShape`'s steps -/

/-- `n` rounds that absorb a message and draw a coin. -/
def pairsB : ℕ → List Bool
  | 0 => []
  | n + 1 => true :: false :: pairsB n

theorem skel_pairRounds (F : Type) : ∀ n, (pairRounds F n).map isMsg = pairsB n
  | 0 => rfl
  | n + 1 => by simp [pairRounds, pairsB, isMsg, skel_pairRounds F n]

theorem skel_switches (F : Type) : ∀ n, (switches F n).map isMsg = pairsB n
  | 0 => rfl
  | n + 1 => by simp [switches, pairsB, isMsg, skel_switches F n]

theorem rskel_replicate (q : SchedRound) (ha : q.2.2.2 = true) (hc : q.1 ≠ 0) :
    ∀ n, rskel (List.replicate n q) = pairsB n
  | 0 => rfl
  | n + 1 => by
    rw [List.replicate_succ, rskel, List.flatMap_cons, ← rskel, rskel_replicate q ha hc n]
    simp [rstep, ha, hc, pairsB]

theorem rskel_append (l l' : List SchedRound) : rskel (l ++ l') = rskel l ++ rskel l' := List.flatMap_append

theorem pairsB_succ' : ∀ n, pairsB (n + 1) = pairsB n ++ [true, false]
  | 0 => rfl
  | n + 1 => congrArg (fun s => true :: false :: s) (pairsB_succ' n)

/-- **The heads agree**: the zerocheck's, lincheck's (with z_partial's split round) and opening's steps are
`holoHead`'s, at the opening's two claims and two per region. -/
theorem skel_head {m kLog nExtra : ℕ} (hm : 22 ≤ m) (F : Type) :
    (zcShape F m ++ (FlockC3.HoloValues.holoLcShape F kLog ++ opShape F (nExtra + 2))).map isMsg =
      rskel (holoHead m kLog (nExtra + 2)) := by
  obtain ⟨m, rfl⟩ : ∃ k, m = k + 22 := ⟨m - 22, by omega⟩
  have e1 : m + 22 - 5 = (m + 16) + 1 := by omega
  simp only [holoHead, rskel_append, e1]
  rw [rskel_replicate _ rfl (by decide), rskel_replicate _ rfl (by decide), rskel_replicate _ rfl (by decide)]
  simp [zcShape, FlockC3.HoloValues.holoLcShape, opShape, isMsg, skel_pairRounds, skel_switches, rskel, rstep,
    pairsB_succ']

/-! ## `hm` -/

theorem count_rskel : ∀ l : List SchedRound, (rskel l).count true = (msgL l).length
  | [] => rfl
  | q :: l => by
    rw [rskel, List.flatMap_cons, ← rskel, List.count_append, count_rskel l, msgL, List.length_append,
      List.length_map]
    unfold rstep
    by_cases ha : q.2.2.2 = true <;> by_cases hc : q.1 = 0 <;> simp [ha, hc]

theorem msgL_length : ∀ l : List SchedRound, (msgL l).length = l.countP fun q => q.2.2.2
  | [] => rfl
  | q :: l => by
    rw [msgL, List.length_append, List.length_map, msgL_length l, List.countP_cons]
    by_cases ha : q.2.2.2 = true <;> simp [ha, Nat.add_comm]

/-- **A message of the head is placed by the head alone.** -/
theorem roundOfB_append : ∀ (s₁ s₂ s₂' : List Bool) (i : ℕ), i < s₁.count true →
    roundOfB (s₁ ++ s₂) i = roundOfB (s₁ ++ s₂') i
  | [], _, _, _, h => absurd h (by simp)
  | true :: _, _, _, 0, _ => rfl
  | true :: s, s₂, s₂', i + 1, h => roundOfB_append s s₂ s₂' i (by simp at h; omega)
  | false :: s, s₂, s₂', i, h => congrArg (· + 1) (roundOfB_append s s₂ s₂' i (by simpa using h))

/-- **`hm`'s equation**: at `holoRounds`' `rs`, message `i` of `holoRepShape`'s kinds, one of the zerocheck's, the
lincheck's and the opening's (`i < m + kLog + nExtra - 6`), is at the reps' round `(msgRounds rs)[i]`, at any Ligerito
schedule. -/
theorem roundOf_holoRepShape {F K D : Type} {m kLog nExtra : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    {rs : Array SchedRound} (hk : 6 ≤ kLog) (h : holoRounds m kLog (nExtra + 2) = some rs) {i : ℕ}
    (hi : i < m + kLog + nExtra - 6) :
    ∃ hi' : i < (msgRounds rs).length,
      FlockC3.Sparse.roundOf (FlockC3.HoloValues.holoRepShape F K D m kLog nExtra sch) i =
        heldBefore (.rep 0) rs ((msgRounds rs)[i]'hi') := by
  obtain ⟨hm, T, hT⟩ := holoRounds_toList hk h
  have hH : (msgL (holoHead m kLog (nExtra + 2))).length = m + kLog + nExtra - 6 := by
    rw [msgL_length]
    simp [holoHead, List.countP_append, List.countP_replicate]
    omega
  obtain rfl : rs = (holoHead m kLog (nExtra + 2) ++ T).toArray := by rw [← hT]
  have hi' : i < (msgL (holoHead m kLog (nExtra + 2) ++ T)).length := by
    rw [msgL_length, List.countP_append, ← msgL_length, hH]
    omega
  refine ⟨by rwa [msgRounds_toArray], ?_⟩
  have hσ : (FlockC3.HoloValues.holoRepShape F K D m kLog nExtra sch).map isMsg =
      rskel (holoHead m kLog (nExtra + 2)) ++ (ligShape F K D sch).map isMsg := by
    rw [← skel_head hm F]
    simp only [FlockC3.HoloValues.holoRepShape, List.map_append, List.append_assoc]
  rw [roundOf_eq_skel, hσ, roundOfB_append _ _ (rskel T) _ (by rw [count_rskel, hH]; exact hi), ← rskel_append,
    roundOfB_rskel _ i hi', heldBefore_toArray]
  simp only [msgRounds_toArray]

/-- **`holoSel_parts`' `hm`** at `holoRepShape`, where each part's `m<m>` port names a message of the zerocheck, the
lincheck or the opening. -/
theorem hm_of_names {F K D : Type} {m kLog nExtra : ℕ} {sch : FlockSoundness.Accounting.Schedule}
    {rs : Array SchedRound} (hk : 6 ≤ kLog) (h : holoRounds m kLog (nExtra + 2) = some rs) {S : ℕ}
    {Vs : Fin S → FlockSoundness.Discharge.Recursion.VStmt} {sh : Flock.Shape} {sess : ℕ → Fin S}
    (hnm : ∀ p < (FlockVBridge.Family.partsOf sh).size, ∀ (j : ℕ) (q : Fin (Vs (sess p)).c.ports.size),
      (Vs (sess p)).c.ports[q].name = s!"m{j}" → j < m + kLog + nExtra - 6) :
    ∀ p < (FlockVBridge.Family.partsOf sh).size, ∀ (j : ℕ) (q : Fin (Vs (sess p)).c.ports.size),
      (Vs (sess p)).c.ports[q].name = s!"m{j}" → ∃ hj : j < (msgRounds rs).length,
        FlockC3.Sparse.roundOf (FlockC3.HoloValues.holoRepShape F K D m kLog nExtra sch) j =
          heldBefore (.rep 0) rs ((msgRounds rs)[j]'hj) :=
  fun p hp j q hq => roundOf_holoRepShape hk h (hnm p hp j q hq)

end FlockC3.Reps.RoundOfHolo
