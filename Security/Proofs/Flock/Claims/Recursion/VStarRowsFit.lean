import Proofs.Flock.Claims.Recursion.HpEarlyStaged
import Proofs.Flock.Claims.Recursion.VStarFitRow
import Proofs.Flock.Claims.Recursion.VStarTopsOk
import Proofs.Flock.Claims.Recursion.RoundWords
import Proofs.Flock.Claims.Recursion.SchedDense

/-!
# The fitted rows at every round with pieces (`rowsFit_tops`), and hp before the last round from them (`hpEarly_tops`)

`RowsFit` at round `i` asks that round `i`'s (part, segment) list at `layR elemR` (`flatPieces` of `pieceAt`) is as long
as the round's rows. At a dense statement that holds only where the round has pieces: round 0 is RecCap's `N` rows and
a held stream round with no tops is one row, but `layR`'s `pieceAt` is empty at both (`stepPieces`; those rounds are
held whole, `msgAt`). There `piecesOf` is empty too, so hp is vacuous, and `hpEarly_tops` takes `RowsFit` only where
`pieceAt` is not empty.

* **The rounds with pieces** are the stream rounds that commit tops (`stepPieces`, `topsPiecesAt`): each top `e` at its
  cap's part, its `n` segments in order (`capFlat`), then the absorbed row at its holder's part (`rowPiece`). The
  fitted round's slots are in the same order (`slotsAt_tops`: `capSlots` per top, then `rowSlot`), so the lists are as
  long (`capFlat_length`, `rowSlot_length`).
* **A top's `t`-th segment** (`capFlat_fit`) is the hm96 slot of its cap's port `top` at instance `j / R · n + t`, which
  reads the round's row `(sizes.take e).sum + t` at the fitted port inputs (`portRow_capl`, `roundRows_cap`), where the
  two reps' tops differ (`hTops`).
* **The absorbed row's segment** is its holder's port's slot, which reads the round's last row (`fit_rowPiece`).

`rowsFit_tops` holds at every history holding both reps' stream rounds (`hH`), with every part staged and accepted by
the stager (`hS`, `hok`), which the honest run has (`honest_half`'s `hH`, `stagedPart_run`, `hok_honest`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-! ## Lists -/

theorem range_flatMap_split {β : Type} (F : ℕ → List β) :
    ∀ (n k : ℕ), k < ((List.range n).flatMap F).length →
      ∃ e < n, ∃ t < (F e).length, k = ((List.range e).flatMap F).length + t
  | 0, k, h => by simp at h
  | n + 1, k, h => by
    have h' : k < ((List.range n).flatMap F).length + (F n).length := by
      simp only [List.range_succ, List.flatMap_append, List.flatMap_cons, List.flatMap_nil, List.append_nil,
        List.length_append] at h
      exact h
    by_cases hk : k < ((List.range n).flatMap F).length
    · obtain ⟨e, he, t, ht, rfl⟩ := range_flatMap_split F n k hk
      exact ⟨e, by omega, t, ht, rfl⟩
    · exact ⟨n, by omega, k - ((List.range n).flatMap F).length, by omega, by omega⟩

theorem flatMap_length_congr {α β γ : Type} (F : α → List β) (G : α → List γ)
    (h : ∀ a, (F a).length = (G a).length) : ∀ L : List α, (L.flatMap F).length = (L.flatMap G).length
  | [] => rfl
  | a :: L => by
    rw [List.flatMap_cons, List.flatMap_cons, List.length_append, List.length_append, h a,
      flatMap_length_congr F G h L]

theorem findSome?_toList_length {α β γ : Type} (f : α → Option β) (g : α → Option γ)
    (h : ∀ a, (f a).isSome = (g a).isSome) :
    ∀ L : List α, (L.findSome? f).toList.length = (L.findSome? g).toList.length
  | [] => rfl
  | a :: L => by
    rw [List.findSome?_cons, List.findSome?_cons]
    have ha := h a
    cases hf : f a with
    | none =>
      cases hg : g a with
      | none => exact findSome?_toList_length f g h L
      | some _ => simp [hf, hg] at ha
    | some _ =>
      cases hg : g a with
      | none => simp [hf, hg] at ha
      | some _ => rfl

/-! ## A round that commits tops -/

/-- **The absorbed row's slot and piece** are both there or both not (`rowSlot`, `rowPiece`: the first holder). -/
theorem rowSlot_length (r : XR) (j : ℕ) : (rowSlot r j).length = (rowPiece r j).length := by
  unfold rowSlot rowPiece
  refine findSome?_toList_length _ _ (fun s => ?_) _
  try dsimp only
  rcases h1 : (heldRows r s).find? (·.1 == j) with _ | h
  · simp [h1]
  · rcases h2 : r.w.parts[s]?.bind partCircuit with _ | c
    · simp [h1, h2]
    · simp [h1, h2]

/-- **Top `e` of stream round `j` as (part, segment) pairs**: its cap's part with each of its segments
(`capPieceAt`). -/
def capFlat (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) : List (ℕ × (Msg × List HmNets.Lin)) :=
  (capPieceAt r rs j e).2.map ((capPieceAt r rs j e).1, ·)

theorem capFlat_length (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) :
    (capFlat r rs j e).length = (capSlots r rs j e).length := by
  simp [capFlat, capPieceAt, topPiece, capSlots]

theorem flatPieces_map {α : Type} (f : α → ℕ × List (Msg × List HmNets.Lin)) :
    ∀ L : List α, HpEarly.flatPieces (L.map f) = L.flatMap fun a => (f a).2.map ((f a).1, ·)
  | [] => rfl
  | a :: L => by
    have ih := flatPieces_map f L
    unfold HpEarly.flatPieces at ih ⊢
    rw [List.map_cons, List.flatMap_cons, List.flatMap_cons, ih]

/-- **A round that commits tops as (part, segment) pairs**: each top's (`capFlat`), then the absorbed row's. -/
theorem flatPieces_tops {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ}
    (he : ((rs.getD (j % rs.size) default).2.1).isEmpty = false) :
    HpEarly.flatPieces (topsPiecesAt r rs j) =
      (List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capFlat r rs j) ++
        HpEarly.flatPieces (rowPiece r j) := by
  unfold topsPiecesAt
  rw [if_neg (Bool.ne_true_of_eq_false he)]
  have h := flatPieces_map (capPieceAt r rs j) (List.range ((rs.getD (j % rs.size) default).2.1).size)
  unfold HpEarly.flatPieces at h ⊢
  rw [List.flatMap_append, h]
  try rfl

/-- `RowsFit` read off `getElem?`. -/
theorem rowsFit_of_getElem? {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {h : List (Msg × Msg)}
    {i : ℕ}
    (hl : (HpEarly.flatPieces ((DeployedR.layR OpenRows.elemR).pieceAt x i)).length = (roundRows I w ρ h i).length)
    (hk : ∀ (k : ℕ) (v : ℕ × (Msg × List HmNets.Lin)), (HpEarly.flatPieces ((DeployedR.layR OpenRows.elemR).pieceAt x i))[k]? = some v →
      ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn)
        (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
        partAt x v.1 = some (st, c) ∧ srcOf I x w ρ h = some src ∧
        _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧ v.2 = (([] : Msg), hmCommit c q n) ∧
        c.ports.findIdx? (·.name == nm) = some q ∧ portRow (fitPin c pin) nm n = some rw ∧
        rowHidden c nm n = some ps ∧ (roundRows I w ρ h i)[k]? = some rw) :
    HpEarly.RowsFit I x w ρ h i := by
  unfold HpEarly.RowsFit
  exact ⟨hl, fun k hk' => hk k _ (List.getElem?_eq_getElem hk')⟩

/-- **Top `e`'s `t`-th segment of stream round `j` is fitted to the round's row** `(sizes.take e).sum + t`: its cap's
port `top` (index 0) at instance `j / R · n + t`, at the fitted port inputs (`portRow_capl`, `roundRows_cap`). -/
theorem capFlat_fit {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape} {H : List (Msg × Msg)}
    {stg : Stage} {r : XR} {rs : Array RecVStar.SchedRound} {j e t : ℕ} (hr : readX x = some r)
    (hrs : repSched r.shape = some rs) (hj : j < 2 * rs.size) (hH : 2 * rs.size ≤ H.length)
    (hS : ∀ s < nS x, ∃ xO, StagedPart stg s x xO)
    (hTops : ∀ src, srcOf (Ir.fit x) x w ρ H = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false)
    (he : e < ((rs.getD (j % rs.size) default).2.1).size)
    (ht : t < ((rs.getD (j % rs.size) default).2.1).getD e 0) :
    ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn)
      (ps : List ℕ) (rw : _root_.VStar.In.Row),
      partAt x (capsFrom r rs + topsBefore rs (j % rs.size) + e) = some (st, c) ∧
      srcOf (Ir.fit x) x w ρ H = some src ∧ _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧
      (capFlat r rs j e)[t]? = some (capsFrom r rs + topsBefore rs (j % rs.size) + e,
        (([] : Msg), hmCommit c 0 (j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t))) ∧
      c.ports.findIdx? (·.name == "top") = some 0 ∧
      portRow (fitPin c pin) "top" (j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t) = some rw ∧
      rowHidden c "top" (j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t) = some ps ∧
      (roundRows (Ir.fit x) w ρ H (j + 1))[((rs.getD (j % rs.size) default).2.1.toList.take e).sum + t]? =
        some rw := by
  have hkR : j % rs.size < rs.size := Nat.mod_lt _ (by omega)
  have hg := topSizes_at hkR he
  obtain ⟨c, hc, hcp⟩ := capPart_parsed hr hrs hg
  have hst := capStmt_at hr hrs hg
  rw [← Nat.add_assoc] at hc hst
  obtain ⟨p0, p1, hp, h0, hw, -⟩ := cap_ports hcp
  have hq0 := capPort_cap hp h0
  have hpa : partAt x (capsFrom r rs + topsBefore rs (j % rs.size) + e) =
      some (.cap (topsBefore rs (j % rs.size) + e + 1) (2 * ((rs.getD (j % rs.size) default).2.1)[e]), c) := by
    simp [partAt, hr, hst, hc]
  obtain ⟨xO, hS0⟩ := hS _ (HpEarly.lt_nS_of_partAt hpa)
  obtain ⟨ps, hps⟩ := (rowHidden_staged hS0 hr hc).2 _ _ hst 0 hq0
    (j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t)
  have hsrc0 : srcOf (Ir.fit x) x w ρ H = some { (Ir.fit x).rest w ρ.1 H with
      msgs := msgsOf rs (roundRows (Ir.fit x) w ρ H), tops := topsOf rs (roundRows (Ir.fit x) w ρ H) } := by
    simp [srcOf, hr, hrs]
  obtain ⟨src, pin, hsrc, hpin, hrow⟩ := portRow_capl (I := Ir) (x := x) (w := w) (ρ := ρ) (hist := H)
    (N := 2 * ((rs.getD (j % rs.size) default).2.1)[e]) hr hrs hj hH he hc hcp (by rw [arrGetD_of_lt _ he])
    (fun a b => hTops _ hsrc0 (topsBefore rs (j % rs.size) + e + 1) (Nat.succ_pos _) a b)
  have hRR := roundRows_cap (I := Ir) (x := x) (w := w) (ρ := ρ) (hist := H) hr hrs
    (show j + 1 ≤ H.length by omega) he ht hc hp h0 hw
  refine ⟨_, c, src, pin, ps, _, hpa, hsrc, hpin, ?_, hq0, (hrow t ht).trans hRR, hps, hRR⟩
  have ht' := ht
  simp only [Array.getD_eq_getD_getElem?] at ht'
  simp [capFlat, capPieceAt, topPiece, List.getElem?_map, List.getElem?_range ht', hc, hq0]

/-! ## `RowsFit` at the rounds with pieces -/

/-- **The fitted rows at every round with pieces**: at a dense statement, each inner round `i < kI x - 1` whose pieces
at `layR elemR` are not empty (a stream round committing tops) has `RowsFit` at the fitted records, where the history
`H` holds both reps' stream rounds (`hH`), every part is staged (`hS`) and the stager accepts every part (`hok`), and
the two reps' tops differ at each later level (`hTops`, the tape exception). -/
theorem rowsFit_tops {Ir : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (Ir.fit x).Tape} {H : List (Msg × Msg)}
    {st : Claims.Recursion.Stmt} {stg : Stage} (hx : parse x = some st) (hf : Claims.Recursion.Form.Dense st.form)
    (hH : ∀ r rs, readX x = some r → repSched r.shape = some rs → 2 * rs.size ≤ H.length)
    (hS : ∀ s < nS x, ∃ xO, StagedPart stg s x xO)
    (hok : ∀ s sv c src, partAt x s = some (sv, c) → srcOf (Ir.fit x) x w ρ H = some src →
      ∃ pin, _root_.VStar.In.partIn src (progOf c) sv = .ok pin)
    (hTops : ∀ src, srcOf (Ir.fit x) x w ρ H = some src → ∀ l, 0 < l → ∀ a b,
      (src.tops[0]!)[l]? = some a → (src.tops[1]!)[l]? = some b → _root_.VStar.In.sameTop a b = false) :
    ∀ i < RestRead.kI x - 1, (DeployedR.layR OpenRows.elemR).pieceAt x i ≠ [] →
      HpEarly.RowsFit (Ir.fit x) x w ρ H i := by
  intro i hi hne
  obtain ⟨r, rs, hr, hrs, hkR⟩ := schedMatches_dense x st hx hf
  have hkI : RestRead.kI x = 2 + 2 * rs.size := hkR
  have hHl := hH r rs hr hrs
  have hnS : nS x = r.w.parts.length := by simp [nS, hr]
  cases i with
  | zero =>
    exfalso
    apply hne
    show stepPieces _ x 0 = []
    have h1 : ¬ (0 + 1 = layoutOf.rounds x) := fun h => by
      have h' : 0 + 1 = RestRead.kI x := h
      omega
    unfold stepPieces
    rw [if_neg h1, if_pos rfl]
  | succ j =>
    have hj : j < 2 * rs.size := by omega
    have hpc : (DeployedR.layR OpenRows.elemR).pieceAt x (j + 1) = topsPiecesAt r rs j := by
      show stepPieces _ x (j + 1) = _
      have hc : ¬ (j + 1 + 1 = layoutOf.rounds x) := fun h => by
        have h' : j + 1 + 1 = RestRead.kI x := h
        omega
      have htp : topsPieces x j = topsPiecesAt r rs j := by simp [topsPieces, hr, hrs]
      simp only [stepPieces, if_neg hc, if_neg (Nat.add_one_ne_zero j), Nat.add_sub_cancel, htp]
    cases hte : ((rs.getD (j % rs.size) default).2.1).isEmpty with
    | true => exact absurd (hpc.trans (topsPiecesAt_empty hte)) hne
    | false =>
      have hRRlen : (roundRows (Ir.fit x) w ρ H (j + 1)).length =
          ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capSlots r rs j)).length +
            (rowSlot r j).length := by
        rw [roundRows, List.length_mapIdx, fit_rows_length Ir x w ρ.1 (H.take (j + 1)), List.length_take,
          Nat.min_eq_left (show j + 1 ≤ H.length by omega), slotsAt_tops hr hrs hte, List.length_append]
      have hAlen : ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capFlat r rs j)).length =
          ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capSlots r rs j)).length :=
        flatMap_length_congr _ _ (capFlat_length r rs j) _
      have hA : ∀ (k : ℕ) (v : ℕ × (Msg × List HmNets.Lin)),
          ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capFlat r rs j))[k]? = some v →
          ∃ (st : _root_.VStar.Stmt) (c : Circuit) (src : _root_.VStar.In.Src) (pin : _root_.VStar.In.PartIn)
            (nm : String) (q n : ℕ) (ps : List ℕ) (rw : _root_.VStar.In.Row),
            partAt x v.1 = some (st, c) ∧ srcOf (Ir.fit x) x w ρ H = some src ∧
            _root_.VStar.In.partIn src (progOf c) st = .ok pin ∧ v.2 = (([] : Msg), hmCommit c q n) ∧
            c.ports.findIdx? (·.name == nm) = some q ∧ portRow (fitPin c pin) nm n = some rw ∧
            rowHidden c nm n = some ps ∧ (roundRows (Ir.fit x) w ρ H (j + 1))[k]? = some rw := by
        intro k v hv
        obtain ⟨e, he, t, ht, rfl⟩ := range_flatMap_split (capFlat r rs j) _ k (List.getElem?_eq_some_iff.1 hv).1
        have ht' : t < ((rs.getD (j % rs.size) default).2.1).getD e 0 := by
          rw [capFlat_length, capSlots_length] at ht
          exact ht
        obtain ⟨sv, c, src, pin, ps, rw, hpa, hsrc, hpin, hcf, hq, hrow, hps, hRR⟩ :=
          capFlat_fit (stg := stg) hr hrs hj hHl hS hTops he ht'
        rw [flatMap_range_getElem? (capFlat r rs j) he ht, ← List.getElem?_eq_getElem ht, hcf] at hv
        obtain rfl := Option.some.inj hv
        have hpre : ((List.range e).flatMap (capFlat r rs j)).length =
            ((rs.getD (j % rs.size) default).2.1.toList.take e).sum := by
          rw [flatMap_length_congr _ _ (capFlat_length r rs j), capSlots_before r rs j he.le]
        exact ⟨sv, c, src, pin, "top", 0, _, ps, rw, hpa, hsrc, hpin, rfl, hq, hrow, hps, by rw [hpre]; exact hRR⟩
      obtain ⟨o, ho⟩ : ∃ o, rowPiece r j = Option.toList o := ⟨_, rfl⟩
      have hslot := rowSlot_length r j
      cases o with
      | none =>
        have hB : HpEarly.flatPieces (rowPiece r j) = [] := by simp [ho, HpEarly.flatPieces]
        have hsl : (rowSlot r j).length = 0 := by simp [hslot, ho]
        refine rowsFit_of_getElem? ?_ ?_
        · rw [hpc, flatPieces_tops hte, hB, List.append_nil, hRRlen, hsl, Nat.add_zero, hAlen]
        · rw [hpc, flatPieces_tops hte, hB, List.append_nil]
          exact hA
      | some p =>
        obtain ⟨s0, segs⟩ := p
        have hP : rowPiece r j = [(s0, segs)] := ho
        have hs0 : s0 < nS x := by
          rw [hnS]
          exact rowPiece_lt (r := r) (j := j) (p := (s0, segs)) (by rw [hP]; exact List.mem_singleton_self _)
        obtain ⟨xO, hS0⟩ := hS s0 hs0
        obtain ⟨sv, c, src, pin, nm, q, n, ps, rw, hpa, hsrc, hpin, hseg, hq, hrow, hps, hlast⟩ :=
          fit_rowPiece (I := Ir) (x := x) (w := w) (ρ := ρ) (hist := H) (K := H.length) hr hrs hS0
            (show j + 1 ≤ H.length by omega) (show j + 1 ≤ H.length by omega) hj hte
            (fun sv c src hp hs => hok s0 sv c src hp (by rwa [List.take_of_length_le (Nat.le_refl _)] at hs))
            (heldRows_words hr) hP
        rw [List.take_of_length_le (Nat.le_refl _)] at hsrc
        subst hseg
        have hB : HpEarly.flatPieces (rowPiece r j) = [(s0, (([] : Msg), hmCommit c q n))] := by
          simp [hP, HpEarly.flatPieces]
        have hsl : (rowSlot r j).length = 1 := by simp [hslot, hP]
        refine rowsFit_of_getElem? ?_ ?_
        · rw [hpc, flatPieces_tops hte, hB, List.length_append, List.length_singleton, hRRlen, hsl, hAlen]
        · rw [hpc, flatPieces_tops hte, hB]
          intro k v hv
          by_cases hk : k < ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap (capFlat r rs j)).length
          · rw [List.getElem?_append_left hk] at hv
            exact hA k v hv
          · rw [List.getElem?_append_right (Nat.le_of_not_lt hk)] at hv
            have hlt := (List.getElem?_eq_some_iff.1 hv).1
            rw [List.length_singleton] at hlt
            have hk0 : k - ((List.range ((rs.getD (j % rs.size) default).2.1).size).flatMap
                (capFlat r rs j)).length = 0 := by omega
            rw [hk0, List.getElem?_cons_zero] at hv
            obtain rfl := Option.some.inj hv
            have hkk : k = (roundRows (Ir.fit x) w ρ H (j + 1)).length - 1 := by
              rw [hRRlen, hsl, ← hAlen]
              omega
            rw [List.getLast?_eq_getElem?, ← hkk] at hlast
            exact ⟨sv, c, src, pin, nm, q, n, ps, rw, hpa, hsrc, hpin, rfl, hq, hrow, hps, hlast⟩

end RecDeploy.VStarBuild

namespace RecDeploy.HpEarly

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild

/-- **hp before the last round on the honest run, with `RowsFit` only where the round has pieces**: `hpEarly_staged`
where `layR elemR`'s `pieceAt x i` is not empty; vacuous where it is, since part `s` then holds no piece of round `i`
(`piecesOf`). `rowsFit_tops` gives the `RowsFit` it takes. -/
theorem hpEarly_tops (mask : ℕ → ℕ → Bool) (I : InnerRec) (IV : Claims.Verifier FlockDeploy.Coin) (b : Brings)
    (w : Claims.Flock.Wit) {x : Msg} {xr : XR} (hx : readX x = some xr) (hk : IV.k (xIOf x) = RestRead.kI x)
    (ω : RecDeploy.Tape (finB b (A0 mask) I x))
    (r : Fin ((RecDeploy.V IV (finB b (A0 mask) I x).st).k x) → FlockDeploy.Coin) {s i : ℕ}
    (hs : s < nS x) (hi : i < RestRead.kI x - 1)
    (hfit : (DeployedR.layR OpenRows.elemR).pieceAt x i ≠ [] →
      RowsFit I x w ω.1 ((((Claims.run FlockDeploy.sha512 (refP IV (finB b (A0 mask) I x) w)
        (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω r).msgs.take (IV.k (xIOf x))).map histOf).take
          (RestRead.kI x - 1)) i)
    {st : Stage} (hS : ∀ s sv c, partAt x s = some (sv, c) → ∃ xO, StagedPart st s x xO) :
    ∀ o m, (o, m) ∈ (DeployedR.layR OpenRows.elemR).piecesOf x s
      (witFin (A0 mask) I x w ω.1 (((Claims.run FlockDeploy.sha512
        (refP IV (finB b (A0 mask) I x) w) (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω
          r).msgs.take (IV.k (xIOf x))).map histOf) s) i →
      ((((Claims.run FlockDeploy.sha512 (refP IV (finB b (A0 mask) I x) w)
        (RecDeploy.V IV (finB b (A0 mask) I x).st) x ω r).msgs.map Prod.fst).getD i []).drop o).take
          m.length = m := by
  by_cases h : (DeployedR.layR OpenRows.elemR).pieceAt x i = []
  · intro o m hm
    rw [Claims.Recursion.VStarB.LayoutB.piecesOf, h] at hm
    simp at hm
  · exact hpEarly_staged mask I IV b w hx hk ω r hs hi (hfit h) hS

end RecDeploy.HpEarly
