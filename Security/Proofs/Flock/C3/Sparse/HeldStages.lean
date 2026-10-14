import Proofs.Flock.C3.Compose.LaterHeldW
import Proofs.Flock.C3.Sparse.MsgsLayers

/-!
# The sparse stages' coins from the inner session's records (`h2`, `h3`)

`coins_stepsOf`: a sparse stage's coin steps are its coins, one a step (`weave_coins`).
`stageStarts`: the round each stage past the reps starts at, at the map read, is one past the history rounds of the
streams before it in `holoStreams`' order (`laterCoins_of_held`'s `h2`, `h3` and R's `StageHeld` starts);
`coinRounds_take_sum`: within a sparse stream, the coins of its coin rounds before the `p`-th take the history rounds
before it (`heldBefore`).
`stageHeldW_sparse`: at the verifier's record of the session (`hh`), its count check of the sparse streams
(`SparseRoundsCounted`) and their draws at the map (`SparseDrawnAt`), phase `x`'s stage at `sparseCoins`' coins is
`StageHeldW` from its start round: its coins are the held rounds of coins (`coinRounds_held`), flattened
(`SparseCsFlat`).
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Exec Refine
open Flock.Guarantees (HoloStream heldBefore coinRounds sparseRounds)
open Flock.RecVStar (SchedRound InnerCommits)
open FlockC3.Reps (nCoins StageHeldW)
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Assumptions (heldCoins)
open FlockVBridge.Sparse (nc phaseMsgs coinTape)
open FlockC3.Compose (C3Run)

/-! ## A sparse stage's coin steps -/

theorem weave_coins : ∀ (σ : List Kind) (ms cs : List Msg), (σ.filterMap kMsgTy).length = ms.length →
    (σ.filterMap kCoinTy).length = cs.length → (weave σ ms cs).filterMap Sum.getRight? = cs
  | [], _, [], _, _ => rfl
  | [], _, _ :: _, _, h => absurd h (by simp)
  | .msg _ :: _, [], _, h, _ => absurd h (by simp [kMsgTy])
  | .msg _ :: σ, _ :: ms, cs, hm, hc => by
      simpa [weave, List.filterMap_cons] using weave_coins σ ms cs (by simpa [kMsgTy, List.filterMap_cons] using hm)
        (by simpa [kCoinTy, List.filterMap_cons] using hc)
  | .coin _ :: _, _, [], _, h => absurd h (by simp [kCoinTy])
  | .coin _ :: σ, ms, c :: cs, hm, hc => by
      simpa [weave] using weave_coins σ ms cs (by simpa [kMsgTy, List.filterMap_cons] using hm)
        (by simpa [kCoinTy, List.filterMap_cons] using hc)

theorem phaseMsgs_length (sh : Model.SparseShape) (w : Array Flock.F128) :
    (phaseMsgs sh w).length = (msgTys sh).length := by
  unfold phaseMsgs msgTys FlockVBridge.Sparse.vtTape FlockVBridge.Sparse.msgTape
  split_ifs <;> simp <;> omega

/-- **A sparse stage's coin steps are its coins**, one a step. -/
theorem coins_stepsOf (sh : Model.SparseShape) (w cs : Array Flock.F128) :
    (stepsOf sh w cs).filterMap Sum.getRight? = (coinTape cs 0 (nc sh)).map (coinArr fOf) :=
  weave_coins _ _ _ (by rw [sparseKinds_msgs, phaseMsgs_length])
    (by rw [sparseKinds_coins]; simp [coinTape])

theorem nCoins_stepsOf_nc (sh : Model.SparseShape) (w cs : Array Flock.F128) : nCoins (stepsOf sh w cs) = nc sh := by
  show ((stepsOf sh w cs).filterMap Sum.getRight?).length = _
  rw [coins_stepsOf]
  simp [coinTape]

/-! ## The start sums (for R's `StageHeld` part 3 and `h2`, `h3`) -/

/-- **The stages' start rounds at the map read**: where each rep's coin steps take its stream's history rounds (`hts`,
the reps' schedule `rs`) and each sparse stage is a phase's steps at the verifier's RecSparse shape `sp`, each sparse
stage takes its stream's history rounds, sparse phase `x` starts one past both reps' and, for phase 1, phase 0's
(`roundAt_sparseRounds`' sum), and the stages past them start one past all four streams' (R's `roundIn_stacked`
sum at `holoStreams`' first four). -/
theorem stageStarts {e : ExecHolo} {u : C3Run e} {rs : Array SchedRound} {sp : Flock.RecSparse.Shape}
    (hts : ∀ i : Fin 2, nCoins (u.ts i) = heldBefore (.rep i) rs rs.size)
    (hsp : ∀ x : Fin 2, ∃ w cs, u.sp x = stepsOf (FlockC3.Compose.sparseOf sp) w cs) :
    (∀ x : Fin 2, nCoins (u.sp x) = heldBefore (.sparse x) (sparseRounds sp) (sparseRounds sp).size) ∧
      1 + nCoins (u.ts 0) + nCoins (u.ts 1) =
        1 + 2 * heldBefore (.rep 0) rs rs.size + (0 : Fin 2).val * nc (FlockC3.Compose.sparseOf sp) ∧
      1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) =
        1 + 2 * heldBefore (.rep 0) rs rs.size + (1 : Fin 2).val * nc (FlockC3.Compose.sparseOf sp) ∧
      1 + nCoins (u.ts 0) + nCoins (u.ts 1) + nCoins (u.sp 0) + nCoins (u.sp 1) =
        1 + ([((.rep 0 : HoloStream), rs), (.rep 1, rs), (.sparse 0, sparseRounds sp), (.sparse 1, sparseRounds sp)].map
          fun e => heldBefore e.1 e.2 e.2.size).sum := by
  have hn : ∀ x : Fin 2, nCoins (u.sp x) = nc (FlockC3.Compose.sparseOf sp) := fun x => by
    obtain ⟨w, cs, h⟩ := hsp x
    rw [h, nCoins_stepsOf_nc]
  have h1 : heldBefore (.rep 1) rs rs.size = heldBefore (.rep 0) rs rs.size := rfl
  have hs := FlockC3.Compose.heldBefore_sparse_size
  refine ⟨fun x => (hn x).trans (hs x sp).symm, ?_, ?_, ?_⟩
  · rw [hts 0, hts 1, h1]; simp; omega
  · rw [hts 0, hts 1, h1, hn 0]; simp; omega
  · simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    rw [hts 0, hts 1, h1, hn 0, hn 1, hs 0 sp, hs 1 sp]
    omega

theorem sum_filter_pos (f : ℕ → ℕ) : ∀ q, (((List.range q).filter fun k => 0 < f k).map f).sum =
    ((List.range q).map f).sum
  | 0 => rfl
  | q + 1 => by
    rw [List.range_succ, List.filter_append, List.map_append, List.map_append, List.sum_append, List.sum_append,
      sum_filter_pos f q]
    by_cases h : 0 < f q
    · simp [h]
    · simp [h]; omega

theorem range_split {n q : ℕ} (hq : q < n) :
    List.range n = List.range q ++ q :: List.range' (q + 1) (n - (q + 1)) := by
  obtain ⟨k, rfl⟩ : ∃ k, n = q + 1 + k := ⟨n - (q + 1), by omega⟩
  rw [show q + 1 + k - (q + 1) = k by omega, List.range_add, List.range_succ, List.range'_eq_map_range,
    List.append_assoc]
  rfl

/-- **A sparse stream's coin rounds before its `p`-th take the history rounds before it**: the coins of its first `p`
coin rounds sum to `heldBefore` at the `p`-th, the rounds of no coins between them taking none. -/
theorem coinRounds_take_sum (x : Fin 2) (rs : Array SchedRound) {p q : ℕ} (hq : (coinRounds rs)[p]? = some q) :
    (((coinRounds rs).take p).map fun k => (rs.getD k default).1).sum = heldBefore (.sparse x) rs q := by
  have hmem : q ∈ coinRounds rs := List.mem_of_getElem? hq
  have hq' : q < rs.size ∧ 0 < (rs.getD q default).1 := by
    unfold coinRounds at hmem
    simpa using hmem
  have hsplit : coinRounds rs = (List.range q).filter (fun k => 0 < (rs.getD k default).1) ++
      q :: ((List.range' (q + 1) (rs.size - (q + 1))).filter fun k => 0 < (rs.getD k default).1) := by
    unfold coinRounds
    rw [range_split hq'.1, List.filter_append, List.filter_cons_of_pos (by simpa using hq'.2)]
  have hp : p = ((List.range q).filter fun k => 0 < (rs.getD k default).1).length := by
    have hnd : (coinRounds rs).Nodup := (List.nodup_range).filter _
    have hA : (coinRounds rs)[((List.range q).filter fun k => 0 < (rs.getD k default).1).length]? = some q := by
      rw [hsplit, List.getElem?_append_right le_rfl, Nat.sub_self, List.getElem?_cons_zero]
    obtain ⟨h₁, e₁⟩ := List.getElem?_eq_some_iff.1 hq
    obtain ⟨h₂, e₂⟩ := List.getElem?_eq_some_iff.1 hA
    exact (hnd.getElem_inj_iff (hi := h₁) (hj := h₂)).1 (e₁.trans e₂.symm)
  rw [hsplit, hp, List.take_left, sum_filter_pos (fun k => (rs.getD k default).1)]
  unfold heldBefore
  rw [Nat.min_eq_left hq'.1.le]
  rfl

/-! ## Phase `x`'s stage -/

/-- **Hypothesis: the sparse twin of `RepRoundsCounted`** (H's count check in `checkCommitRounds`, round for round):
each sparse phase's stream the verifier holds draws, round by round, the coins its schedule `sparseRounds sp` has. -/
def SparseRoundsCounted (cs : InnerCommits) (sp : Flock.RecSparse.Shape) : Prop :=
  ∀ (x : Fin 2) (st : Flock.RecVStar.InnerStream), cs.stream? s!"{Flock.RecVStar.SPARSE}/rep{x.val}" = some st →
    st.coins.map (·.size) = (sparseRounds sp).map (·.1)

/-- **Stub: phase `x`'s rounds of coins are drawn at the map read from round `j`** (80–120 lines: `holoSelWords` at
`holoStreams`' first four streams, `roundAt_sparseRounds` at `stageStarts`' start, and `holoForms`' forms `none` there;
the bound from `holoKinds`' coin count): the stage's `nc` rounds are below `r`, and the stream's coin round `q` is
drawn at the history round `j` past its rounds before `q` and its own coins, less one (`selW`). -/
def SparseDrawnAt (selW : ℕ → String → ℕ → Prop) (r : ℕ) (sp : Flock.RecSparse.Shape) (x : Fin 2) (j : ℕ) : Prop :=
  j + nc (FlockC3.Compose.sparseOf sp) ≤ r ∧ ∀ q ∈ coinRounds (sparseRounds sp),
    selW (j + heldBefore (.sparse x) (sparseRounds sp) q + ((sparseRounds sp).getD q default).1 - 1)
      s!"{Flock.RecVStar.SPARSE}/rep{x.val}" q

/-- **Stub: `sparseCoins`' layout** (40–50 lines, `sizes_get` at each part as `sparseCoins_size` reads them): where
it accepts phase `x`'s held coins `H`, its coins in the order drawn are `H`'s rounds of coins, flattened. -/
def SparseCsFlat (c3 : Flock.RecCoins.C3) (unit : String) : Prop :=
  ∀ (x : Fin 2) sc H, Flock.RecCoins.sparseCoins c3 unit x.val = .ok sc →
    c3.coins unit s!"{Flock.RecVStar.SPARSE}/rep{x.val}" = .ok H → sparseCs sc = (H.filter (·.size != 0)).flatten

theorem map_range_getD {α : Type} (H : Array (Array α)) : (List.range H.size).map (H.getD · #[]) = H.toList := by
  refine List.ext_getElem (by simp) fun n _ h₂ => ?_
  have hn : n < H.size := by simpa using h₂
  simp [hn]

theorem map_range'_get! {α : Type} [Inhabited α] (a : Array α) : (List.range' 0 a.size).map (a[·]!) = a.toList := by
  refine List.ext_getElem (by simp) fun n h₁ _ => ?_
  have hn : n < a.size := by simpa using h₁
  simp [getElem!_pos a n hn]

/-- **`h2`, `h3` at `sparseOf`'s stages**: where the verifier holds the coin server's record (`hh`), counts its sparse
streams' rounds (`SparseRoundsCounted`) and `sparseCoins` accepts phase `x`'s coins `sc`, the stage of any words at
them (`stepsOf … (sparseCs sc)`) is `StageHeldW` from a round `j` its coin rounds are drawn from (`SparseDrawnAt`):
its coin steps are the held rounds of coins (`coinRounds` of `sparseRounds`), flattened (`SparseCsFlat`). -/
theorem stageHeldW_sparse {r : ℕ} {CT : Fin r → Type} {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {selW : ℕ → String → ℕ → Prop}
    {c3 : Flock.RecCoins.C3} {unit : String} {sp : Flock.RecSparse.Shape} {x : Fin 2}
    {sc : Flock.RecCoins.SparseCoins} (w : Array Flock.F128) {j : ℕ}
    (hh : held (wr t) = .ok c3.commits) (hcnt : SparseRoundsCounted c3.commits sp) (hflat : SparseCsFlat c3 unit)
    (hsh : c3.sparse = sp) (hsc : Flock.RecCoins.sparseCoins c3 unit x.val = .ok sc)
    (hdr : SparseDrawnAt selW r sp x j) :
    StageHeldW t wr held selW j (stepsOf (FlockC3.Compose.sparseOf sp) w (sparseCs sc)) := by
  have hsc' := hsc
  unfold Flock.RecCoins.sparseCoins at hsc'
  obtain ⟨H, hH, -⟩ := FlockVBridge.PartS.bind_ok hsc'
  obtain ⟨st, hst, rfl⟩ : ∃ st, c3.commits.stream? s!"{Flock.RecVStar.SPARSE}/rep{x.val}" = some st ∧ st.coins = H := by
    unfold Flock.RecCoins.C3.coins at hH
    split at hH
    · rename_i st hs
      exact ⟨st, hs, (Except.ok.inj hH)⟩
    · cases hH
  have hc := hcnt x st hst
  have hsz : st.coins.size = (sparseRounds sp).size := by simpa using congrArg Array.size hc
  have hqs : ∀ q < (sparseRounds sp).size, (st.coins.getD q #[]).size = ((sparseRounds sp).getD q default).1 := by
    intro q hq
    have e := congrArg (·[q]?) hc
    simp only [Array.getElem?_map, Array.getElem?_eq_getElem hq, Array.getElem?_eq_getElem (hsz ▸ hq),
      Option.map_some] at e
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (hsz ▸ hq), Option.getD_some,
      Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
    exact Option.some.inj e
  have hks : coinRounds (sparseRounds sp) = (List.range st.coins.size).filter fun q => (st.coins.getD q #[]).size != 0 := by
    unfold coinRounds
    rw [hsz]
    refine List.filter_congr fun q hq => ?_
    rw [hqs q (List.mem_range.1 hq)]
    generalize ((sparseRounds sp).getD q default).1 = a
    cases a <;> rfl
  have hfl : ((coinRounds (sparseRounds sp)).map fun q => (st.coins.getD q #[]).toList).flatten = (sparseCs sc).toList := by
    have e : (coinRounds (sparseRounds sp)).map (st.coins.getD · #[]) = (st.coins.filter (·.size != 0)).toList := by
      rw [hks, Array.toList_filter, ← map_range_getD st.coins, List.filter_map]
      rfl
    rw [hflat x sc st.coins hsc hH, Array.toList_flatten, ← e, List.map_map]
    rfl
  have hcsz : (sparseCs sc).size = nc (FlockC3.Compose.sparseOf sp) := by
    rw [sparseCoins_size hsc, hsh, FlockC3.Compose.degrees_length]
    unfold nc FlockVBridge.Sparse.nR
    rw [FlockC3.Compose.pLog_sparseOf]
    rfl
  refine ⟨coinRounds (sparseRounds sp), (st.coins.getD · #[]), s!"{Flock.RecVStar.SPARSE}/rep{x.val}", id, ?_, ?_,
    ?_, ?_⟩
  · rw [coins_stepsOf, hfl, ← hcsz, coinTape, List.map_map, ← map_range'_get! (sparseCs sc), List.map_map]
    rfl
  · intro q c hq
    rw [hh] at hq
    simp only [heldCoins, hst, Option.bind_some, id] at hq
    show st.coins.getD q #[] = c
    rw [Array.getD_eq_getD_getElem?, hq, Option.getD_some]
  · rw [hfl, Array.length_toList, hcsz]
    exact hdr.1
  · intro p q hp _
    have hmem : q ∈ coinRounds (sparseRounds sp) := List.mem_of_getElem? hp
    have hlt : ∀ k ∈ coinRounds (sparseRounds sp), k < (sparseRounds sp).size := fun k hk => by
      unfold coinRounds at hk
      exact List.mem_range.1 (List.mem_filter.1 hk).1
    have e : ((coinRounds (sparseRounds sp)).take p).map (fun q => (st.coins.getD q #[]).size) =
        ((coinRounds (sparseRounds sp)).take p).map fun k => ((sparseRounds sp).getD k default).1 :=
      List.map_congr_left fun k hk => hqs k (hlt k (List.mem_of_mem_take hk))
    rw [e, coinRounds_take_sum x _ hp, hqs q (hlt q hmem)]
    exact hdr.2 q hmem

end

end FlockC3.Sparse

#print axioms FlockC3.Sparse.weave_coins
#print axioms FlockC3.Sparse.coins_stepsOf
#print axioms FlockC3.Sparse.stageStarts
#print axioms FlockC3.Sparse.coinRounds_take_sum
#print axioms FlockC3.Sparse.stageHeldW_sparse
