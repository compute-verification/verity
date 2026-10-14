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
(`SparseRoundsCounted`) and their draws at the map (`SparseDrawnAt`, `sparseDrawnAt_holo` at `holoSelWords`), phase
`x`'s stage at `sparseCoins`' coins is `StageHeldW` from its start round: its coins are the held rounds of coins,
flattened (`sparseCsFlat`).
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

/-- **Phase `x`'s rounds of coins are drawn at the map read from round `j`**: the stage's `nc` rounds are below `r`,
and the stream's coin round `q` is drawn at the history round `j` past its rounds before `q` and its own coins, less
one (`selW`). `sparseDrawnAt_holo` gives it at `holoSelWords`. -/
def SparseDrawnAt (selW : ℕ → String → ℕ → Prop) (r : ℕ) (sp : Flock.RecSparse.Shape) (x : Fin 2) (j : ℕ) : Prop :=
  j + nc (FlockC3.Compose.sparseOf sp) ≤ r ∧ ∀ q ∈ coinRounds (sparseRounds sp),
    selW (j + heldBefore (.sparse x) (sparseRounds sp) q + ((sparseRounds sp).getD q default).1 - 1)
      s!"{Flock.RecVStar.SPARSE}/rep{x.val}" q

/-- **`sparseCoins`' layout**: where it accepts phase `x`'s held coins `H`, its coins in the order drawn are `H`'s
rounds of coins, flattened (`sparseCsFlat`). -/
def SparseCsFlat (c3 : Flock.RecCoins.C3) (unit : String) : Prop :=
  ∀ (x : Fin 2) sc H, Flock.RecCoins.sparseCoins c3 unit x.val = .ok sc →
    c3.coins unit s!"{Flock.RecVStar.SPARSE}/rep{x.val}" = .ok H → sparseCs sc = (H.filter (·.size != 0)).flatten

/-! ## `SparseDrawnAt` at the map read -/

theorem toString_lt_two {n : ℕ} (hn : n < 2) : (toString n : String).toList.length = 1 := by
  rcases (show n = 0 ∨ n = 1 by omega) with rfl | rfl <;> decide

theorem name_ne (t P : String) {a b : ℕ} (ha : a < 2) (hb : b < 2) (ht : t ≠ s!"{P}/") :
    s!"{t}rep{a}" ≠ s!"{P}/rep{b}" := by
  intro h
  have h := congrArg String.toList h
  simp only [String.toList_append] at h
  have hn := congrArg List.length h
  simp only [List.length_append, toString_lt_two ha, toString_lt_two hb] at hn
  obtain ⟨h1, -⟩ := List.append_inj h (by simp only [List.length_append]; omega)
  have h1' : (toString t).toList ++ "rep".toList = ((toString P).toList ++ "/".toList) ++ "rep".toList := by
    rw [List.append_assoc]; exact h1
  apply ht
  apply String.ext
  simp only [String.toList_append]
  exact List.append_cancel_right h1'

/-- **A rep's stream is no sparse phase's** where the table's prefix is not the sparse streams' (`htbl`): its name
`{tbl}rep{i}` would be `sparse/rep{x}`. -/
theorem repName_ne_sparse {tbl : String} (htbl : tbl ≠ s!"{Flock.RecVStar.SPARSE}/") (i x : Fin 2) :
    (HoloStream.rep i).name tbl ≠ (HoloStream.sparse x).name tbl :=
  name_ne tbl _ i.isLt x.isLt htbl

theorem sparseName_ne (tbl : String) : (HoloStream.sparse 0).name tbl ≠ (HoloStream.sparse 1).name tbl := by
  intro h
  have h := congrArg String.toList h
  simp only [HoloStream.name, String.toList_append] at h
  exact absurd (List.append_cancel_left h) (by decide)

/-- **`holoStreams`' order finds sparse phase `x` by its name**, past both reps' (`repName_ne_sparse`). -/
theorem find?_sparseName {tbl : String} (htbl : tbl ≠ s!"{Flock.RecVStar.SPARSE}/") (rs S : Array SchedRound)
    (R : List (HoloStream × Array SchedRound)) (x : Fin 2) :
    ((HoloStream.rep 0, rs) :: (HoloStream.rep 1, rs) :: (.sparse 0, S) :: (.sparse 1, S) :: R).find?
      (fun e => e.1.name tbl == (HoloStream.sparse x).name tbl) = some (.sparse x, S) := by
  let p := fun e : HoloStream × Array SchedRound => e.1.name tbl == (HoloStream.sparse x).name tbl
  have hr : ∀ i : Fin 2, ¬p (.rep i, rs) = true := fun i h => repName_ne_sparse htbl i x (beq_iff_eq.1 h)
  rw [List.find?_cons_of_neg (p := p) (hr 0), List.find?_cons_of_neg (p := p) (hr 1)]
  rcases x with ⟨_ | _ | n, hn⟩
  · exact List.find?_cons_of_pos (beq_iff_eq.2 rfl)
  · exact (List.find?_cons_of_neg (p := p) fun h => sparseName_ne tbl (beq_iff_eq.1 h)).trans
      (List.find?_cons_of_pos (beq_iff_eq.2 rfl))
  · exact absurd hn (by omega)

/-- **`SparseDrawnAt` at the map read**: at `holoStreams`' order (`holoStreams_sparse_eq`), sparse phase `x`'s
stream, found by its name (`find?_sparseName`, the table's prefix not the sparse streams', `htbl`), starts one past
both reps' rounds and, for phase 1, phase 0's (`roundAt_sparseRounds`, `stageStarts`' start), its rounds of forms
`none` (`holoForms_sparse_none`), and its `nc` rounds inside `holoForms` past both reps' (`heldBefore_rep_forms`). -/
theorem sparseDrawnAt_holo {st : Flock.Stmt} {hp : HoloParams} {ish : Flock.RecVStar.Inner} {tbl : String}
    {sp : Flock.RecSparse.Shape} {rs : Array SchedRound} (x : Fin 2) (hsp : ish.sparse = some sp)
    (hr : Flock.Guarantees.holoRounds ish.m ish.kLog ish.claims = some rs)
    (hS : ∃ L, Flock.Guarantees.holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L)
    (hc : 0 < ish.claims)
    (hA : FlockC3.Compose.repFormsAt (holoAt st hp) = FlockC3.Compose.repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = FlockC3.Compose.sparseOf sp) (htbl : tbl ≠ s!"{Flock.RecVStar.SPARSE}/") :
    SparseDrawnAt (FlockC3.Compose.holoSelWords st hp ish tbl)
      (1 + (kindCoinTys (Flock.Guarantees.holoKinds (holoAt st hp))).length) sp x
      (1 + 2 * heldBefore (.rep 0) rs rs.size + x.val * nc (FlockC3.Compose.sparseOf sp)) := by
  obtain ⟨L, hS⟩ := hS
  have hS' := hS
  rw [hsp] at hS'
  obtain ⟨rs', R, hr', -, rfl⟩ := FlockC3.Compose.holoStreams_sparse_eq hS'
  obtain rfl : rs' = rs := Option.some.inj (hr'.symm.trans hr)
  have hrep := FlockC3.Compose.heldBefore_rep_forms 0 hc hr
  have hn : nc (FlockC3.Compose.sparseOf sp) =
      (FlockC3.Compose.sparseForms (FlockC3.Compose.sparseOf sp)).length := by
    simp only [FlockC3.Compose.sparseForms, List.length_replicate, nc, FlockVBridge.Sparse.nR]
    omega
  refine ⟨?_, fun q hq => ?_⟩
  · rw [FlockC3.Compose.length_kindCoinTys_holoKinds, FlockC3.Compose.holoForms_eq, hA, hshape]
    simp only [List.length_append]
    have : x.val * nc (FlockC3.Compose.sparseOf sp) ≤ 1 * nc (FlockC3.Compose.sparseOf sp) :=
      Nat.mul_le_mul_right _ (by omega)
    omega
  have hq' : q < (sparseRounds sp).size ∧ 0 < ((sparseRounds sp).getD q default).1 := by
    unfold coinRounds at hq
    simpa using hq
  refine ⟨_, (.sparse x, sparseRounds sp), hS, find?_sparseName htbl _ _ R x, hq'.2, ?_,
    FlockC3.Compose.holoForms_sparse_none hS hsp hc hA hshape (by fin_cases x <;> simp) rfl⟩
  rw [FlockC3.Compose.roundAt_sparseRounds]
  show _ = _ + ((sparseRounds sp).getD q default).1
  omega

/-! ## `sparseCoins`' layout -/

theorem map_range_get! {α : Type} [Inhabited α] (a : Array α) : (List.range a.size).map (a[·]!) = a.toList := by
  refine List.ext_getElem (by simp) fun n h₁ _ => ?_
  have hn : n < a.size := by simpa using h₁
  simp [getElem!_pos a n hn]

theorem flatten_ones {α : Type} [Inhabited α] (f : ℕ → Array α) : ∀ n, (∀ i < n, (f i).size = 1) →
    ((List.range n).map fun i => (f i).toList).flatten = (List.range n).map fun i => (f i)[0]!
  | 0, _ => rfl
  | n + 1, h => by
    have e : (f n).toList = [(f n)[0]!] := by
      refine List.ext_getElem (by simp [h n (by omega)]) fun k _ h₂ => ?_
      have hk : k = 0 := by simpa using h₂
      subst hk
      simp [getElem!_pos (f n) 0 (by rw [h n (by omega)]; omega)]
    rw [List.range_succ, List.map_append, List.map_append, List.flatten_append,
      flatten_ones f n fun i hi => h i (by omega)]
    simp [e]

/-- **Rounds of one coin, flattened**: an array of rounds whose `1 … n` hold one coin each is round 0, those coins,
round `n + 1` and the `t` rounds past it. -/
theorem flatten_split {α : Type} [Inhabited α] {cs : Array (Array α)} {n t : ℕ} (hs : cs.size = 1 + (n + (1 + t)))
    (h1 : ∀ i < n, (cs[1 + i]!).size = 1) :
    cs.flatten.toList = cs[0]!.toList ++ ((List.range n).map fun i => (cs[1 + i]!)[0]!) ++ (cs[1 + n]!).toList ++
      ((List.range t).map fun i => (cs[1 + (n + (1 + i))]!).toList).flatten := by
  rw [Array.toList_flatten, ← map_range_get! cs, hs, List.range_add, List.range_add, List.range_add]
  simp only [List.map_append, List.map_map, List.flatten_append, List.range_one, List.map_cons, List.map_nil,
    List.flatten_cons, List.flatten_nil, List.append_nil, Function.comp_def, Nat.add_zero]
  rw [flatten_ones (fun i => cs[1 + i]!) n h1]
  simp only [List.append_assoc]

theorem want_one {a b n i : ℕ} (T : Array ℕ) (hi : i < n) : (#[a] ++ Array.replicate n 1 ++ #[b] ++ T)[1 + i]! = 1 := by
  rw [FlockSoundness.Discharge.ZkExec.append_get!_left _ _ (by simp; omega),
    FlockSoundness.Discharge.ZkExec.append_get!_left _ _ (by simp; omega),
    FlockSoundness.Discharge.ZkExec.append_get!_right _ _ (j := i) (by simp) (by simp; omega),
    getElem!_pos _ i (by simp; omega)]
  simp

/-- **`sparseCoins`' layout holds**: its count check gives its rounds of coins one round for `ζ_c`, one coin a round
for H2's and H3's, one for `ζ_p` and, at T > 1, one for `ρ_t` (`flatten_split`), which it reads in that order. -/
theorem sparseCsFlat (c3 : Flock.RecCoins.C3) (unit : String) : SparseCsFlat c3 unit := by
  intro x sc H hsc hH
  unfold Flock.RecCoins.sparseCoins at hsc
  obtain ⟨H', hH', h⟩ := FlockVBridge.PartS.bind_ok hsc
  rw [hH] at hH'
  cases hH'
  dsimp only at h
  generalize H.filter (fun x => x.size != 0) = cs at h ⊢
  by_cases hst : c3.sparse.stacked = true
  all_goals first | rw [ite_eq_left hst, ite_eq_left hst] at h | rw [ite_eq_right hst, ite_eq_right hst] at h
  all_goals
    split at h
    · cases h
    rename_i hb
    cases h
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hb
    have hsz := congrArg Array.size hb
    simp only [Array.size_map, Array.size_append, Array.size_replicate, Array.size_singleton, Array.size_empty] at hsz
    have h1 : ∀ i < Flock.RecSparse.PRE_ROUNDS + c3.sparse.degrees.length, (cs[1 + i]!).size = 1 :=
      fun i hi => (sizes_get hb _ (by simp; omega)).trans (want_one _ hi)
    generalize Flock.RecSparse.PRE_ROUNDS = p at hsz h1 ⊢
    generalize c3.sparse.degrees.length = d at hsz h1 ⊢
    apply Array.toList_inj.1
    first | rw [flatten_split (n := p + d) (t := 1) (by omega) h1] | rw [flatten_split (n := p + d) (t := 0) (by omega) h1]
    simp only [sparseCs, Array.toList_append, Array.toList_map, Array.toList_range, List.range_add, List.map_append,
      List.map_map, Function.comp_def, List.append_assoc, List.range_one, List.map_nil, List.map_cons,
      List.flatten_nil, List.flatten_cons, List.append_nil]
    have e1 : (fun x => (cs[x + 1]!)[0]!) = fun i => (cs[1 + i]!)[0]! := funext fun x => by rw [Nat.add_comm]
    have e2 : (fun x => (cs[x + p + 1]!)[0]!) = fun x => (cs[1 + (p + x)]!)[0]! :=
      funext fun x => by rw [show x + p + 1 = 1 + (p + x) by omega]
    rw [e1, e2, show p + 1 + d = 1 + (p + d) by omega]
    first | rw [show p + 2 + d = 1 + (p + d + (1 + 0)) by omega] | simp

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
its coin steps are the held rounds of coins (`coinRounds` of `sparseRounds`), flattened (`sparseCsFlat`). -/
theorem stageHeldW_sparse {r : ℕ} {CT : Fin r → Type} {Cm : Type} {t : List (Cm × ((k : Fin r) → CT k))}
    {wr : List (Cm × ((k : Fin r) → CT k)) → Lean.Json × String}
    {held : Lean.Json × String → Except String InnerCommits} {selW : ℕ → String → ℕ → Prop}
    {c3 : Flock.RecCoins.C3} {unit : String} {sp : Flock.RecSparse.Shape} {x : Fin 2}
    {sc : Flock.RecCoins.SparseCoins} (w : Array Flock.F128) {j : ℕ}
    (hh : held (wr t) = .ok c3.commits) (hcnt : SparseRoundsCounted c3.commits sp)
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
    rw [sparseCsFlat c3 unit x sc st.coins hsc hH, Array.toList_flatten, ← e, List.map_map]
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
