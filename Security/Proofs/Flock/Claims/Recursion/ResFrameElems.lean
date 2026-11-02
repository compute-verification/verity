import Proofs.Flock.Claims.Recursion.ResFrameRounds
import Proofs.Flock.Claims.Recursion.LayoutBDeployed
import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.Claims.Recursion.DecodeTies
import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.C3.Reps.RoundLinks

/-!
# A frame port's element, at a decision

The pieces `framePorts_D` (`ResFramePorts`) puts together, at a dense decision `D` of the opened messages:

* `DecAt.frame_elem`: element `o` of `rounds`' `kk`-th round, read off the messages (`framesRead`) at its flat index,
  is element `o` of the values chunk `t` absorbs, `t` the `kk`-th committing round, and the rep's rows hold those values
  at round `t` (`RowsAt`).
* `DecAt.rows_getD`: rep `r`'s round `t` rows, at `Bound.opened`'s messages and two reps of `n` rounds each, are the
  opened round `r n + t + 1`'s.
* `Rows.elemReadC_frame`: below the frame values' length, `elemReadC` is `framesRead`'s element at the row port.
* `ResLower.mem_heldRows_of`: a part's port `m<k>` at index `i` holds row `rep · R + t` of its circuit's port `i`, `t`
  the `k`-th committing round.
-/

namespace RecDeploy.ResFrames

open Flock
open Flock.RecAlgebra (rounds)

theorem sum_decomp : ∀ (L : List ℕ) (e : ℕ), e < L.sum →
    ∃ i, ∃ hi : i < L.length, ∃ o, o < L[i] ∧ e = (L.take i).sum + o
  | [], e, h => by simp at h
  | a :: L, e, h => by
    by_cases hea : e < a
    · exact ⟨0, by simp, e, by simpa using hea, by simp⟩
    · obtain ⟨i, hi, o, ho, he⟩ := sum_decomp L (e - a) (by simp at h; omega)
      refine ⟨i + 1, by simpa using hi, o, by simpa using ho, ?_⟩
      simp only [List.take_succ_cons, List.sum_cons]
      omega

theorem flatten_getElem? {α : Type} : ∀ (A : List (List α)) (k o : ℕ) (hk : k < A.length), o < A[k].length →
    A.flatten[((A.map List.length).take k).sum + o]? = A[k][o]?
  | [], _, _, hk, _ => absurd hk (Nat.not_lt_zero _)
  | a :: A, 0, o, _, ho => by
    simp only [List.flatten_cons, List.map_cons, List.take_zero, List.sum_nil, Nat.zero_add,
      List.getElem_cons_zero] at ho ⊢
    exact List.getElem?_append_left ho
  | a :: A, k + 1, o, hk, ho => by
    simp only [List.flatten_cons, List.map_cons, List.take_succ_cons, List.sum_cons, List.getElem_cons_succ] at ho ⊢
    rw [Nat.add_assoc, List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]
    exact flatten_getElem? A k o (Nat.lt_of_succ_lt_succ hk) ho

theorem flatten_map_getElem? {α β : Type} (R : Array (Array α)) (f : α → β) (kk o : ℕ) (hkk : kk < R.size)
    (ho : o < R[kk].size) :
    (R.flatten.toList.map f)[((R.toList.map Array.size).take kk).sum + o]? = some (f R[kk][o]) := by
  rw [Array.toList_flatten, List.map_flatten, List.map_map]
  have e := flatten_getElem? (R.toList.map (List.map f ∘ Array.toList)) kk o (by simpa using hkk)
    (by simpa using ho)
  have hl : (R.toList.map (List.map f ∘ Array.toList)).map List.length = R.toList.map Array.size := by
    simp [Function.comp_def]
  rw [hl] at e
  rw [e]
  simp [ho]

theorem flatten_size {α : Type} (R : Array (Array α)) : R.flatten.size = (R.toList.map Array.size).sum := by
  rw [← Array.length_toList, Array.toList_flatten, List.length_flatten, List.map_map]
  rfl

theorem take_sizes_sum {α : Type} (RR : Array (Array α)) (extra : Array ℕ) {kk : ℕ} (h : kk ≤ RR.size) :
    (((RR.map (·.size)) ++ extra).toList.take kk).sum = ((RR.toList.map Array.size).take kk).sum := by
  rw [Array.toList_append, Array.toList_map, List.take_append_of_le_length (by simpa using h)]

theorem take_sizes_full {α : Type} (RR : Array (Array α)) (extra : Array ℕ) :
    (((RR.map (·.size)) ++ extra).toList.take RR.size).sum = RR.flatten.size := by
  rw [take_sizes_sum RR extra (Nat.le_refl _), List.take_of_length_le (by simp), flatten_size]

theorem filterIdx_length (V : List ℕ) :
    ((List.range V.length).filter fun i => V.getD i 0 != 0).length = (V.filter (· != 0)).length := by
  have hV : V = (List.range V.length).map (V.getD · 0) := by
    apply List.ext_getElem (by simp)
    intro n h1 h2
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h1]
  conv => rhs; rw [hV]
  rw [List.filter_map, List.length_map]
  rfl

theorem cutLens_two_getD {α : Type} (l : List α) (n : ℕ) {k t : ℕ} (hk : k < 2) (ht : t < n) (d : α) :
    ((FlockDeploy.cutLens [n, n] l).getD k []).getD t d = l.getD (k * n + t) d := by
  match k, hk with
  | 0, _ => simp [FlockDeploy.cutLens, List.getD_eq_getElem?_getD, ht]
  | 1, _ => simp [FlockDeploy.cutLens, List.getD_eq_getElem?_getD, List.getElem?_drop, ht]

theorem getD_toList_eq {α : Type} [Inhabited α] (rs : Array α) (t : ℕ) :
    rs.toList.getD t default = rs.getD t default := by
  simp [List.getD_eq_getElem?_getD]

theorem portNames_m {sh : Shape} {k : ℕ} (hk : k < (rounds sh).size) : (RecVStar.portNames sh)[k]! = s!"m{k}" := by
  unfold RecVStar.portNames
  rw [getElem!_pos _ _ (by simp; omega), Array.getElem_append_left (by simpa using hk)]
  simp

/-- **`rounds`' `kk`-th round commits**: at a rep whose schedule is its frames' and a proof that fits its rows, every
round of `rounds` has a committing round of the schedule (`commitRounds`). -/
theorem commit_of_round {c : FlockDeploy.RepCtx} {rows : List (List ByteArray)} {p : Flock.Proof} {T : F128}
    {rs : Array RecVStar.SchedRound} {kk : ℕ} (hsc : FlockDeploy.SchedOf c) (hF : FlockDeploy.FitsRep c rows p T)
    (hrs : c.sched = rs.toList)
    (hkk : kk < (rounds (FlockSoundness.Refine.shapeOf c.st c.sch (c.n - 2))).size) :
    ∃ t, (RecVStar.commitRounds rs)[kk]? = some t := by
  obtain ⟨hs, hkl, rs', hrr, hsc'⟩ := hsc
  rw [Array.toList_inj.1 (hsc'.symm.trans hrs)] at hrr
  obtain ⟨e, hne, hps⟩ := hF.sizes
  obtain ⟨h22, -, hn7, hk0, -⟩ := RecDeploy.fast100_sizes hs
  rw [show c.n - 2 = e by omega] at hkk
  have hA : (rounds (FlockSoundness.Refine.shapeOf c.st c.sch e)).toList.map Array.size =
      (FlockDeploy.Cnt.schedV c.sch c.st.m c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)).filter (· != 0) := by
    rw [FlockDeploy.Cnt.schedV_filter, hkl]
    exact rounds_sizes_alg (FlockSoundness.Refine.shapeOf c.st c.sch e) rfl
  rw [hne] at hrr
  have hfl : rs.toList.map (·.2.2.2) =
      (FlockDeploy.Cnt.schedV c.sch c.st.m c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)).map (· != 0) := by
    rw [FlockDeploy.map_flag, FlockDeploy.repRounds_flagTop hs hrr,
      FlockDeploy.Cnt.schedPat_flags c.sch (show 6 ≤ c.st.m by omega) c.st.fold.kLog (2 + e) (2 <<< c.sch.ext)]
  rw [FlockDeploy.Cnt.commitRounds_eq hfl]
  refine ⟨_, List.getElem?_eq_getElem ?_⟩
  rw [filterIdx_length, ← hA, List.length_map, Array.length_toList]
  exact hkk

end RecDeploy.ResFrames

namespace RecDeploy.Rows.DecAt

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf repCtx repLens cutLens chunksOf FitsRep SchedOf vals)
open FlockSoundness.Discharge.Recursion (HClass)
open Flock.RecAlgebra (rounds)
open FlockDecodeTies (keyVal)

section

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass)

/-- **Rep `2 i + r`'s schedule is its frames'**, where row G holds at table `i`. -/
theorem sched_ctx (acc : ℕ → Array F128) {i : ℕ} (hG : D.RowG cls acc i) (hi : i < D.sts.size) (r : Fin 2) :
    SchedOf (D.ctx (2 * i + r)) := by
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have h2 : (2 * i + (r : ℕ)) / 2 = i := rep_div_two i r
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← D.points]
    exact hG.extra
  have hft : fast100 D.sts[i].m = some (D.files cls i).sch := by
    have h0 := hG.fast
    rw [D.files_st cls hi] at h0
    exact h0
  have hex : D.sts[i].extraClaims (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    have h0 := hext
    rw [show (tabsOf D.sts)[i]! = (D.files cls i).st from rfl, D.files_st cls hi] at h0
    exact h0
  unfold DecAt.ctx
  refine FlockDeploy.schedOf_repCtx (extras := (D.files cls i).extras) (by rw [h2, ← hsts]; exact hi) ?_ ?_
  · simp only [h2, ← hsts]
    rw [hft]
    rfl
  · simp only [h2, ← hsts, ← hsp]
    exact hex

/-- **A frame element is its chunk's value**: at a dense decision, element `o` of `rounds`' `kk`-th round, read off the
messages at its flat index, is element `o` of the values `vs` chunk `t` absorbs, where `t` is the `kk`-th committing
round of the rep's schedule `rs`, and the rep's round-`t` rows hold `vs` (`RowsAt`). -/
theorem frame_elem {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (h0 : 0 < D.sts.size) (r : Fin 2)
    {Xs : Shape} (hX : D.sh cls 0 = Xs) {rs : Array RecVStar.SchedRound}
    (hrs : (D.ctx (2 * 0 + (r : ℕ))).sched = rs.toList) {kk o : ℕ} (hkk : kk < (rounds Xs).size)
    (ho : o < (rounds Xs)[kk].size) :
    ∃ t vs, (RecVStar.commitRounds rs)[kk]? = some t ∧ t < rs.size ∧ o < vs.length ∧
      (framesRead x ms (2 * 0 + (r : ℕ)))[(((rounds Xs).toList.map Array.size).take kk).sum + o]? = vs[o]? ∧
      FlockDeploy.RowsAt (rs.toList.getD t default) ((D.rows (2 * 0 + (r : ℕ))).getD t []) vs := by
  subst hX
  have hG := D.rowG cls (fun _ => #[]) hst hf hx h0
  have hk : 2 * 0 + (r : ℕ) < D.d.f.ps.size := D.rep_lt h0 r
  have hF := D.ctx_rows_fit hk
  have hsc := D.sched_ctx cls (fun _ => #[]) hG h0 r
  have hext : (tabsOf D.sts)[0]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls 0).extras := by
    rw [← D.points]
    exact hG.extra
  have hst0 : (D.ctx (2 * 0 + (r : ℕ))).st = (D.files cls 0).st := D.ctx_st cls (fun _ => #[]) 0 r hG
  have hsch : (D.ctx (2 * 0 + (r : ℕ))).sch = (D.files cls 0).sch := D.ctx_sch cls (fun _ => #[]) 0 r hG
  have hn : (D.ctx (2 * 0 + (r : ℕ))).n = 2 + (D.files cls 0).extras.size :=
    D.ctx_n cls (fun _ => #[]) 0 r hG hext
  have hshC : FlockSoundness.Refine.shapeOf (D.ctx (2 * 0 + (r : ℕ))).st (D.ctx (2 * 0 + (r : ℕ))).sch
      ((D.ctx (2 * 0 + (r : ℕ))).n - 2) = D.sh cls 0 := by
    rw [hst0, hsch, hn, Nat.add_sub_cancel_left]
    all_goals rfl
  obtain ⟨t, ht⟩ := ResFrames.commit_of_round hsc hF hrs (by rw [hshC]; exact hkk)
  have hrc := ResFrames.round_chunk hsc hF hrs ht
  rw [hshC, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hkk] at hrc
  have hvs : (rounds (D.sh cls 0))[kk].toList.map (keyVal D.d.f.ps[2 * 0 + (r : ℕ)]! (D.T (2 * 0 + (r : ℕ)))) =
      vals ((chunksOf (D.ctx (2 * 0 + (r : ℕ))) D.d.f.ps[2 * 0 + (r : ℕ)]! (D.T (2 * 0 + (r : ℕ)))).getD t []) :=
    Option.some.inj hrc
  have htl : t < rs.size := by
    have hm := List.mem_of_getElem? ht
    unfold RecVStar.commitRounds at hm
    exact List.mem_range.1 (List.mem_filter.1 hm).1
  refine ⟨t, vals ((chunksOf (D.ctx (2 * 0 + (r : ℕ))) D.d.f.ps[2 * 0 + (r : ℕ)]! (D.T (2 * 0 + (r : ℕ)))).getD t []),
    ht, htl, ?_, ?_, ?_⟩
  · rw [← hvs, List.length_map, Array.length_toList]
    exact ho
  · rw [D.framesRead_eq cls hst hf hx h0 r, ResFrames.flatten_map_getElem? _ _ kk o hkk ho, ← hvs,
      List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem ho]
    all_goals rfl
  · have hlt : t < (D.rows (2 * 0 + (r : ℕ))).length := by
      rw [← hF.sched, hrs, Array.length_toList]
      exact htl
    have h := hF.elems t hlt
    rw [hrs] at h
    exact h

end

/-- **Rep `r`'s rounds are the opened rounds after its link**: at `Bound.opened`'s messages at the deployed join, with
two reps of `n` rounds each, rep `r`'s round `t` rows are opened round `r n + t + 1`'s. -/
theorem rows_getD {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
    (D : DecAt rd x ms cs) {o₀ : List (Claims.Msg × Salt)} {ot : List (List (Claims.Msg × Salt))}
    {last : Claims.Msg} (hms : ms = ((o₀ :: ot).map fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]) {n : ℕ}
    (hlens : repLens D.d.spec D.d.sts = [n, n]) (r : Fin 2) {t : ℕ} (ht : t < n) :
    (D.rows (2 * 0 + (r : ℕ))).getD t [] = RestRead.rowsR ((o₀ :: ot).getD ((r : ℕ) * n + t + 1) []) := by
  subst hms
  have h1 : (((o₀ :: ot).map fun o => FlockDeploy.joinR (o.map Prod.fst)) ++ [last]).tail.dropLast.mapM
      FlockDeploy.rowsOf = some (ot.map RestRead.rowsR) := by
    simp only [List.map_cons, List.cons_append, List.tail_cons, List.dropLast_concat]
    exact RestRead.mapM_map_some (fun o => FlockDeploy.rowsOf_joinR' _) ot
  have e2 : D.rss = ot.map RestRead.rowsR := by
    unfold DecAt.rss
    rw [h1, Option.getD_some]
  unfold DecAt.rows
  rw [hlens, e2, show 2 * 0 + (r : ℕ) = r by omega, ResFrames.cutLens_two_getD _ _ r.isLt ht, List.getD_cons_succ]
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases ot[(r : ℕ) * n + t]? <;> rfl

end RecDeploy.Rows.DecAt

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin setupOf)

/-- **`elemReadC` below the frame values**: at a row port `j` inside rep `o + g`'s frame values, `elemReadC` reads
`framesRead`'s element `j`, whatever the accumulators. -/
theorem elemReadC_frame (acc : Claims.Msg → List Claims.Msg → List Coin → ℕ → Array F128) {x : Claims.Msg}
    {ms : List Claims.Msg} {cs : List Coin} {key : ℕ × ℕ × ℕ × ℕ} {e j : ℕ}
    {v : FlockDeploy.Ins × SessionSpec × Array Stmt} (hv : setupOf false x = some v)
    (hpa : portAt (shapeNoCoin v.2.1 v.2.2) key.1 e = some j)
    (hj : j < (framesRead x ms (key.2.1 + key.2.2.1)).length) :
    elemReadC acc x ms cs key e = ((framesRead x ms (key.2.1 + key.2.2.1))[j]?).getD 0 := by
  have hj' : j < (framesRead x ms (key.2.1 + key.2.2.1)).toArray.size := by simpa using hj
  unfold elemReadC
  simp only [hv, hpa]
  rw [el_append_left hj']
  unfold RecAlgebra.el
  rw [Array.getD_eq_getD_getElem?, List.getElem?_toArray]

end RecDeploy.Rows

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)

/-- **A part's message port holds its round's row**: where part `s`'s V\* statement is an algebra part whose port `i`
is `m<k>`, at index `i` of its circuit's ports, and `t` is the `k`-th committing round, part `s` holds row
`rep · R + t` (`heldRows`) at port `m<k>`, index `i`, instance `rep`, for either rep. -/
theorem mem_heldRows_of {r : XR} {s : ℕ} {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm pt)) {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {rs : Array RecVStar.SchedRound}
    (hrs : repSched r.shape = some rs) {i k t : ℕ} (hi : i < pt.S.ports.length) (hnm : pt.S.ports[i].1 = s!"m{k}")
    (hfind : c.ports.findIdx? (·.name == pt.S.ports[i].1) = some i)
    (hct : (RecVStar.commitRounds rs)[k]? = some t) (rf : Fin 2) :
    ((rf : ℕ) * rs.size + t, s!"m{k}", i, (rf : ℕ)) ∈ heldRows r s := by
  have hst : (RecVStar.statements r.shape).toOption.bind (fun vs => vs[s]?) = some (.inner j nm pt) := by
    rw [hss]
    exact hsv
  unfold heldRows
  rw [hst, hc, hrs]
  refine List.mem_flatMap.2 ⟨(s!"m{k}", s!"rec-c{k}", #[some 0, some 1]), ?_, ?_⟩
  · refine List.mem_filterMap.2 ⟨pt.S.ports[i], List.getElem_mem hi, ?_⟩
    show (RecVStar.mIndex pt.S.ports[i].1).map (fun k' => (pt.S.ports[i].1, s!"rec-c{k'}", #[some 0, some 1])) = _
    rw [hnm, FlockC3.Reps.mIndex_m, Option.map_some]
  · have e1 : msgRound rs (rf : ℕ) k = some ((rf : ℕ) * rs.size + t) := by
      unfold msgRound
      rw [hct]
      all_goals rfl
    have e2 : c.ports.findIdx? (·.name == s!"m{k}") = some i := by
      rw [← hnm]
      exact hfind
    have hci : RecVStar.cIndex (s!"m{k}", s!"rec-c{k}", (#[some 0, some 1] : Array (Option ℕ))).2.1 = some k :=
      FlockC3.Reps.cIndex_c k
    rw [hci]
    refine List.mem_filterMap.2 ⟨some (rf : ℕ), ?_, ?_⟩
    · rcases (show (rf : ℕ) = 0 ∨ (rf : ℕ) = 1 by omega) with h | h <;> rw [h] <;> simp
    · show ((msgRound rs (rf : ℕ) k).bind fun j =>
        (c.ports.findIdx? (·.name == s!"m{k}")).bind fun qi => some (j, s!"m{k}", qi, (rf : ℕ))) = _
      rw [e1, e2]
      all_goals rfl

end RecDeploy.ResLower
