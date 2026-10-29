import Proofs.Flock.Claims.Recursion.ResFrames
import Proofs.Flock.Claims.Recursion.RestOpenedSized
import Proofs.Flock.Claims.Recursion.OutTopRows

/-!
# A committing round's elements are its chunk's values

`framesRead` reads a rep's elements as `keyVal` at `rounds`' keys, flattened (`ResFrames.framesRead_fits`); the opened
rows hold the values each chunk absorbs (`FitsRep.elems`). This file lines the two up round by round: the `kk`-th
round of `rounds` reads, element by element, the values of chunk `t`, the `kk`-th committing round (`commitRounds`).
`rounds`' sizes are V\*'s (`rounds_sizes_alg`), whose rounds are `schedV`'s nonzero counts (`Cnt.schedV_filter`), and
chunk `t` absorbs `schedV`'s `t`-th count (`Cnt.cntPat_repFrames`), so the flattened lists cut the same way.
-/

namespace RecDeploy.ResFrames

open Flock
open FlockDeploy (FitsRep)
open Flock.RecAlgebra (Key msgNames ligRounds rounds hRound)
open FlockDecodeTies (keyVal ligLevels_nil ligLevels_cons)
open FlockC3.Reps (ligF1 ligF2 lig0 ligMid ligLast ligLevels ligRounds_eq)
open FlockSoundness.Discharge.Exec (chunks repFrames)

/-! ## `rounds`' sizes -/

theorem msgNames_sz (n : ℕ) : (msgNames n).size = 4 := rfl

theorem oodMsg_sz (o n : ℕ) : (#[Key.ood o] ++ msgNames n).size = 5 := rfl

theorem sz_push (out : Array (Array Key)) (r : Array Key) :
    (out.push r).toList.map Array.size = out.toList.map Array.size ++ [r.size] := by
  simp

theorem ligF1_sz : ∀ (l : List ℕ) (b : ℕ × Array (Array Key)),
    (l.foldl ligF1 b).2.toList.map Array.size = b.2.toList.map Array.size ++ List.replicate l.length 4
  | [], b => by simp
  | _ :: l, b => by
    rw [List.foldl_cons, ligF1_sz l]
    simp only [ligF1, sz_push, msgNames_sz, List.length_cons, List.replicate_succ, List.append_assoc,
      List.singleton_append]

theorem lig0_sz (sch : Schedule) :
    (lig0 sch).2.2.toList.map Array.size = [1, 1] ++ List.replicate (sch.k0 + 1) 4 ++ [5, 5, 4] := by
  have a := ligF1_sz (List.range' 0 (sch.k0 + 1)) (0, #[#[Key.T], #[Key.ood 0]])
  simp only [lig0]
  generalize List.foldl ligF1 (0, #[#[Key.T], #[Key.ood 0]]) (List.range' 0 (sch.k0 + 1)) = A at a ⊢
  rw [show List.range' 0 2 = [0, 1] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, ligF2, sz_push, a, msgNames_sz, oodMsg_sz, List.length_range']
  simp

theorem ligMid_sz (s : ℕ × ℕ × Array (Array Key)) :
    (ligMid s).2.2.toList.map Array.size = s.2.2.toList.map Array.size ++ [4, 4, 4, 4, 5, 5, 4] := by
  obtain ⟨mi, oi, out⟩ := s
  simp only [ligMid]
  rw [show List.range' 0 4 = [0, 1, 2, 3] from rfl, show List.range' 0 2 = [0, 1] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, ligF1, ligF2, sz_push, msgNames_sz, oodMsg_sz, List.append_assoc,
    List.cons_append, List.nil_append]

theorem ligLast_sz (ext : ℕ) (s : ℕ × ℕ × Array (Array Key)) :
    (ligLast ext s).2.2.toList.map Array.size = s.2.2.toList.map Array.size ++ [4, 4, 4, 4 + (2 <<< ext)] := by
  obtain ⟨mi, oi, out⟩ := s
  simp only [ligLast]
  rw [show List.range' 0 3 = [0, 1, 2] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, ligF1, sz_push, msgNames_sz, Array.size_append, Array.size_map,
    Array.size_range, List.append_assoc, List.cons_append, List.nil_append]

theorem ligLevels_sz (R ext : ℕ) : ∀ (l : List ℕ) (init : ℕ × ℕ × Array (Array Key)),
    (ligLevels R ligMid (ligLast ext) l init).2.2.toList.map Array.size =
      init.2.2.toList.map Array.size ++
        l.flatMap fun i => if i + 1 < R then [4, 4, 4, 4, 5, 5, 4] else [4, 4, 4, 4 + (2 <<< ext)]
  | [], init => by rw [ligLevels_nil]; simp
  | i :: l, init => by
    rw [ligLevels_cons, ligLevels_sz R ext l, List.flatMap_cons]
    by_cases h : i + 1 < R
    · simp only [h, ↓reduceIte, ligMid_sz, List.append_assoc]
    · simp only [h, ↓reduceIte, ligLast_sz, List.append_assoc]

theorem ligRounds_sz (sch : Schedule) :
    (ligRounds sch).toList.map Array.size = [1, 1] ++ List.replicate (sch.k0 + 1) 4 ++ [5, 5, 4] ++
      (List.range' 0 (sch.levels.size - 1)).flatMap fun i =>
        if i + 1 < sch.levels.size - 1 then [4, 4, 4, 4, 5, 5, 4] else [4, 4, 4, 4 + (2 <<< sch.ext)] := by
  rw [ligRounds_eq, ligLevels_sz, lig0_sz]

/-- **`rounds`' sizes are V\*'s**, at a plain shape. -/
theorem rounds_sizes_alg (sh : Shape) (hh : sh.holo = false) :
    (rounds sh).toList.map Array.size = VStarSizes.sizesV ⟨sh.m, sh.kLog, sh.claims⟩ sh.sched := by
  simp only [rounds, hRound, hh, Bool.false_eq_true, ↓reduceIte, Array.toList_append, List.map_append, ligRounds_sz,
    Array.toList_map, Array.toList_range, List.map_map]
  simp [VStarSizes.sizesV, Function.comp_def, List.map_const', List.append_assoc]

/-! ## Two cuts of one list -/

theorem filter_len_idx {α : Type} : ∀ (B : List (List α)) (k t : ℕ),
    ((List.range B.length).filter (fun i => (B.getD i []).length != 0))[k]? = some t →
      (B.filter (fun l => l.length != 0))[k]? = some (B.getD t [])
  | [], k, t, h => by simp at h
  | a :: B, k, t, h => by
    rw [List.length_cons, List.range_succ_eq_map, List.filter_cons, List.filter_map] at h
    have e : (List.filter ((fun i => ((a :: B).getD i []).length != 0) ∘ Nat.succ) (List.range B.length)) =
        (List.range B.length).filter (fun i => (B.getD i []).length != 0) := by
      congr 1
    rw [e] at h
    rw [List.filter_cons]
    by_cases ha : (a.length != 0) = true
    · simp only [List.getD_cons_zero, ha, ↓reduceIte] at h ⊢
      cases k with
      | zero =>
        simp at h
        subst h
        simp
      | succ k =>
        simp only [List.getElem?_cons_succ, List.getElem?_map] at h ⊢
        obtain ⟨t', ht', rfl⟩ := Option.map_eq_some_iff.1 h
        rw [filter_len_idx B k t' ht']
        simp
    · simp only [List.getD_cons_zero, ha, Bool.false_eq_true, ↓reduceIte] at h ⊢
      rw [List.getElem?_map] at h
      obtain ⟨t', ht', rfl⟩ := Option.map_eq_some_iff.1 h
      rw [filter_len_idx B k t' ht']
      simp

theorem flatten_filter_ne {α : Type} : ∀ B : List (List α), (B.filter fun l => l.length != 0).flatten = B.flatten
  | [] => rfl
  | a :: B => by
    rw [List.filter_cons]
    by_cases ha : (a.length != 0) = true
    · simp only [ha, ↓reduceIte, List.flatten_cons, flatten_filter_ne B]
    · obtain rfl : a = [] := by simpa using ha
      simp [flatten_filter_ne B]

/-- **Two cuts of one list**: where `A`'s pieces have `V`'s nonzero lengths and `B`'s have `V`'s, with the same
flattening, `A`'s `kk`-th piece is `B`'s `t`-th, `t` the `kk`-th index of a nonzero length. -/
theorem align {α : Type} {A B : List (List α)} {V : List ℕ} (hA : A.map List.length = V.filter (· != 0))
    (hB : B.map List.length = V) (hf : A.flatten = B.flatten) {kk t : ℕ}
    (ht : ((List.range V.length).filter fun i => V.getD i 0 != 0)[kk]? = some t) :
    A[kk]? = some (B.getD t []) := by
  subst hB
  have hAB : A = B.filter fun l => l.length != 0 := by
    refine RecDeploy.eq_of_flatten_lengths ?_ (hf.trans (flatten_filter_ne B).symm)
    rw [hA, List.filter_map]
    rfl
  rw [hAB]
  refine filter_len_idx B kk t ?_
  have e : (fun i => (B.map List.length).getD i 0 != 0) = fun i => (B.getD i []).length != 0 := by
    funext i
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
    cases B[i]? <;> rfl
  rw [List.length_map, e] at ht
  exact ht

/-! ## The committing rounds' chunks -/

/-- **A committing round's elements are its chunk's values**: at a rep whose schedule is its frames' and a proof that
fits its rows, `rounds`' `kk`-th round reads, at `keyVal`, the values chunk `t` absorbs, where `t` is the `kk`-th
committing round. -/
theorem round_chunk {c : FlockDeploy.RepCtx} {rows : List (List ByteArray)} {p : Flock.Proof} {T : F128}
    {rs : Array RecVStar.SchedRound} {kk t : ℕ} (hsc : FlockDeploy.SchedOf c) (hF : FitsRep c rows p T)
    (hrs : c.sched = rs.toList) (ht : (RecVStar.commitRounds rs)[kk]? = some t) :
    ((rounds (FlockSoundness.Refine.shapeOf c.st c.sch (c.n - 2))).toList.map fun a => a.toList.map (keyVal p T))[kk]? =
      some (FlockDeploy.vals ((FlockDeploy.chunksOf c p T).getD t [])) := by
  obtain ⟨hs, hkl, rs', hrr, hsc'⟩ := hsc
  rw [Array.toList_inj.1 (hsc'.symm.trans hrs)] at hrr
  obtain ⟨e, hne, hps⟩ := hF.sizes
  obtain ⟨h22, -, hn7, hk0, -⟩ := RecDeploy.fast100_sizes hs
  have hk : 0 < c.sch.k0 := by rw [hk0]; split_ifs <;> omega
  have hL : 2 ≤ c.sch.levels.size := hF.levels
  have hyr : p.ligerito.finalProof.yr.size = 2 <<< c.sch.ext := FlockDecodeTies.yr_size hps
  have hcnt := FlockDeploy.Cnt.cntPat_repFrames (st := c.st) (sch := c.sch) c.dom e p T (by omega) hk
    (FlockDeploy.q_ne_zero hs) (by omega) (by omega) hps.r1ab hps.r1c hps.zPartial
    (fun k hk => FlockDecodeTies.rsEach_get hps (by simp [FlockSoundness.Refine.shapeOf]; omega))
  rw [hyr] at hcnt
  have hch : FlockDeploy.chunksOf c p T = (chunks (repFrames c.st c.sch c.dom (2 + e) p T)).1 := by
    unfold FlockDeploy.chunksOf
    rw [hne]
  have hB : ((FlockDeploy.chunksOf c p T).map FlockDeploy.vals).map List.length =
      FlockDeploy.Cnt.schedV c.sch c.st.m c.st.fold.kLog (2 + e) (2 <<< c.sch.ext) := by
    rw [hch, List.map_map]
    exact congrArg Prod.fst hcnt
  have hflat := frames_chunks hF hkl
  rw [show c.n - 2 = e by omega] at hflat ⊢
  rw [Array.toList_flatten, List.map_flatten, List.map_map] at hflat
  have hA : ((rounds (FlockSoundness.Refine.shapeOf c.st c.sch e)).toList.map
      fun a => a.toList.map (keyVal p T)).map List.length =
        (FlockDeploy.Cnt.schedV c.sch c.st.m c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)).filter (· != 0) := by
    rw [FlockDeploy.Cnt.schedV_filter, hkl, List.map_map]
    refine Eq.trans ?_ (rounds_sizes_alg (FlockSoundness.Refine.shapeOf c.st c.sch e) rfl)
    congr 1
    funext a
    simp
  rw [hne] at hrr
  have hfl : rs.toList.map (·.2.2.2) =
      (FlockDeploy.Cnt.schedV c.sch c.st.m c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)).map (· != 0) := by
    rw [FlockDeploy.map_flag, FlockDeploy.repRounds_flagTop hs hrr,
      FlockDeploy.Cnt.schedPat_flags c.sch (show 6 ≤ c.st.m by omega) c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)]
  rw [FlockDeploy.Cnt.commitRounds_eq hfl] at ht
  rw [align hA hB hflat ht, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases (FlockDeploy.chunksOf c p T)[t]? <;> rfl

end RecDeploy.ResFrames
