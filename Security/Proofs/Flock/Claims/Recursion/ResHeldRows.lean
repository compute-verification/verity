import Proofs.Flock.Claims.Recursion.OutTopRows
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Claims.Recursion.CapLower
import Proofs.Flock.Claims.Recursion.SlotsCalls
import Proofs.Flock.Claims.Recursion.RestLens

/-!
# Every holder of a stream round's row holds the row its first holder opens

At `layR` a stream round `j` that absorbs a message holds one hm96 commit string past its tops, and every part with a
held row of round `j` (`heldRows`) commits to it there: its part's check reads that round's bytes as its link's
commit (`round_view`, `heldRow_bytes`, `CmtAgrees`). The round's row is opened at its first holder (`rowSlot`), so up
to a collision among the recorded calls every holder's row is the opened one.

* `held_commit_ms`: a holder's commit string is the readers' round `j + 1` past its tops' bytes.
* `held_rows_tie`: two holders with one commit string hold one row (`rows_eq_of_calls'`, prefixes of one length).
* `held_rowSlot`: at the joint witnesses, every holder's row is its round's `rowSlot`'s.
-/

namespace RecDeploy.ResHeld

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

/-- **A holder's commit string is its round's past the tops**: at a staged part whose check passes and whose readers'
messages agree with its commit strings (`CmtAgrees`), each of its held rows `h` is of a stream round `h.1 < 2R`, and
round `h.1 + 1` of the messages holds, past its tops' `128 · n` bytes, the holder's slot's commit string. -/
theorem held_commit_ms (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) {st : Stage} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {ms : List Msg}
    (hA : (layoutBTops fin).CmtAgrees (cmtTr recRead) x xO ms) {r : XR} (hr : readX x = some r)
    {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) (hK : layoutOf.rounds x = 2 + 2 * rs.size)
    {c : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c) {h : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s) :
    h.1 < 2 * rs.size ∧
      ((ms.getD (h.1 + 1) []).drop ((((rs.getD (h.1 % rs.size) default).2.1).toList.map (128 * ·)).sum)).take 128 =
        readForms w (hmCommit c h.2.2.1 h.2.2.2) := by
  obtain ⟨i, lk, cs, hi, hl, hcs, hv⟩ := round_view hS hr hrs
  obtain ⟨kc, rs', kk, cc, hrs', hmr, hrep, hkc, hb⟩ := heldRow_bytes hS hck hr hpc hi hcs hh
  rw [hrs] at hrs'
  obtain rfl : rs = rs' := Option.some.inj hrs'
  unfold msgRound at hmr
  rcases ht : (RecVStar.commitRounds rs)[kc]? with _ | t
  · rw [ht] at hmr; cases hmr
  rw [ht, Option.map_some, Option.some.injEq] at hmr
  have htl := commitRounds_lt ht
  obtain ⟨y, hmsg, hlens, htops, hcm⟩ := hv h.2.2.2 hrep t htl
  have hyc := hcm kc kk cc ht hkc
  have hj : h.1 < 2 * rs.size := by
    have : h.2.2.2 * rs.size ≤ 1 * rs.size := Nat.mul_le_mul_right _ (by omega)
    omega
  have hjm : h.1 % rs.size = t := by
    rw [← hmr, Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt htl]
  refine ⟨hj, ?_⟩
  rw [hjm]
  have hab : (rs.getD t default).2.2.2 = true := by
    have := RestRead.heldRows_absorbs hh hrs
    rwa [hjm] at this
  have hsr : streamRound x h.1 = some (rs.getD t default) := by
    simp [streamRound, hr, hrs, hj, hjm]
  have hfx : (layoutBTops fin).fixedAt x (h.1 + 1) = none := by
    show (streamRound x h.1).bind _ = none
    rw [hsr]
    show (if ((rs.getD t default).2.1.isEmpty && !(rs.getD t default).2.2.2) = true then some [] else none) = none
    rw [hab]
    simp
  have hpc' : (layoutBTops fin).pieceAt x (h.1 + 1) = topsPiecesAt r rs h.1 := by
    show stepPieces fin x (h.1 + 1) = _
    unfold stepPieces
    rw [if_neg (by omega), if_neg (by omega), Nat.add_sub_cancel]
    simp [topsPieces, hr, hrs]
  have hA1 := hA (h.1 + 1) (by show h.1 + 1 + 1 < layoutOf.rounds x; omega) hfx
  have hmsg' : cmtTr recRead x (h.1 + 1) xO = (y.tops.map List.flatten).flatten ++ cc.toList := by
    rw [← hmr, hmsg, hyc, List.flatten_singleton]
  have hTL : (y.tops.map List.flatten).flatten.length =
      (((rs.getD t default).2.1).toList.map (128 * ·)).sum := by
    rw [List.length_flatten]
    congr 1
    apply List.ext_getElem (by simp [tops_length hlens])
    intro e h1 h2
    have he' : e < y.tops.length := by simpa using h1
    obtain ⟨kt, top, -, hye, h128⟩ := htops e he'
    have he2 : e < ((rs.getD t default).2.1).size := by rw [← tops_length hlens]; exact he'
    simp only [List.getElem_map, Array.getElem_toList]
    rw [hye, flatten_nodes_length h128, tops_size hlens he' hye, getD_lt he2]
  have hcl : cc.toList.length = 128 := by
    rw [← hb, readForms_length, hmCommit_length]
  have hlen : ((y.tops.map List.flatten).flatten ++ cc.toList).length =
      (((rs.getD t default).2.1).toList.map (128 * ·)).sum + 128 := by
    rw [List.length_append, hTL, hcl]
  have hT : (ms.getD (h.1 + 1) []).take ((((rs.getD t default).2.1).toList.map (128 * ·)).sum + 128) =
      (y.tops.map List.flatten).flatten ++ cc.toList := by
    by_cases he : ((rs.getD t default).2.1).isEmpty = true
    · have hpe : (layoutBTops fin).pieceAt x (h.1 + 1) = [] := by
        rw [hpc']
        unfold topsPiecesAt
        rw [hjm, if_pos he]
      rw [hA1.1 hpe, hmsg', List.take_of_length_le (le_of_eq hlen)]
    · have hpt : topsPiecesAt r rs h.1 = (List.range ((rs.getD t default).2.1).size).map (capPieceAt r rs h.1) ++
          rowPiece r h.1 := by
        unfold topsPiecesAt
        rw [hjm, if_neg he]
      have hcap : ((List.range ((rs.getD t default).2.1).size).map (capPieceAt r rs h.1)).map
          (fun p => pieceLen p.2) = ((rs.getD t default).2.1).toList.map (128 * ·) := by
        apply List.ext_getElem (by simp)
        intro e h1 h2
        have he2 : e < ((rs.getD t default).2.1).size := by simpa using h1
        simp only [List.getElem_map, List.getElem_range, capPieceAt, pieceLen_topPiece, hjm, Array.getElem_toList]
        rw [getD_lt he2]
      have hsl : s < r.w.parts.length := by
        rcases hps : r.w.parts[s]? with _ | p
        · rw [hps] at hpc; cases hpc
        · exact (List.getElem?_eq_some_iff.1 hps).1
      have hrne : rowPiece r h.1 ≠ [] := by
        unfold rowPiece
        intro h0
        rw [Option.toList_eq_nil_iff, List.findSome?_eq_none_iff] at h0
        have h0s := h0 s (List.mem_range.2 hsl)
        obtain ⟨h', hf⟩ : ∃ h', (heldRows r s).find? (·.1 == h.1) = some h' :=
          Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨h, hh, by simp⟩)
        rw [hf, Option.bind_some, hpc, Option.map_some] at h0s
        cases h0s
      have hrow : ((rowPiece r h.1).map fun p => pieceLen p.2).sum = 128 := by
        have e : (rowPiece r h.1).map (fun p => pieceLen p.2) = (rowPiece r h.1).map fun _ => 128 :=
          List.map_congr_left fun p hp => by
            obtain ⟨h1, -, -, c1, -, hp2⟩ := rowPiece_mem hp
            rw [hp2]
            simp [pieceLen, hmCommit_length]
        have hle : (rowPiece r h.1).length ≤ 1 := toList_length_le _
        have hpos : 0 < (rowPiece r h.1).length := List.length_pos_iff.2 hrne
        rw [e, sum_map_const]
        have : (rowPiece r h.1).length = 1 := by omega
        rw [this]
      have hPL : (layoutBTops fin).piecesLen x (h.1 + 1) = (((rs.getD t default).2.1).toList.map (128 * ·)).sum + 128 := by
        unfold LayoutB.piecesLen
        rw [hpc', hpt, List.map_append, List.sum_append, hcap, hrow]
      have h2 := hA1.2
      rw [hPL, hmsg', List.take_of_length_le (le_of_eq hlen)] at h2
      exact h2
  have h3 := congrArg (List.drop ((((rs.getD t default).2.1).toList.map (128 * ·)).sum)) hT
  rw [List.drop_take, Nat.add_sub_cancel_left, List.drop_left' hTL] at h3
  rw [h3, hb]

/-- **A held row's slot calls are among `layR`'s**, at its part's witness. -/
theorem held_calls_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    {r : XR} (hr : readX x = some r) {s : ℕ} {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {ws : List Claims.Flock.Wit} {w : Claims.Flock.Wit} (hw : ws[s]? = some w) {h : ℕ × String × ℕ × ℕ}
    (hh : h ∈ heldRows r s) :
    (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c h.2.2.1 h.2.2.2),
        readForms w ((hmCommit c h.2.2.1 h.2.2.2).drop 512)) ∈ (DeployedR.layR elem).allCalls x ws ∧
      (rowPfx c h.2.2.1 ++ readForms w (((rowHidden c h.2.1 h.2.2.2).getD []).map HmNets.Lin.var),
        readForms w (hmDigest c h.2.2.1 h.2.2.2)) ∈ (DeployedR.layR elem).allCalls x ws := by
  have hm : ∀ p ∈ hmCalls c h.2.1 h.2.2.1 h.2.2.2, (p.1 ++ readForms w p.2.1, readForms w p.2.2) ∈
      (DeployedR.layR elem).allCalls x ws := fun p hp => by
    refine CapLower.mem_allCalls hw ?_
    unfold LayoutB.calls
    refine List.mem_map.2 ⟨p, OpenSlots.mem_layR_of_dep elem ?_, rfl⟩
    simp only [layoutBDeployed, hr, Option.bind_some, hc, Option.map_some, Option.getD_some]
    exact List.mem_flatMap.2 ⟨h, hh, hp⟩
  refine ⟨?_, ?_⟩
  · have e := hm (Hm96.Default512.saltPrefix.data.toList, hmSalt c h.2.2.1 h.2.2.2, (hmCommit c h.2.2.1 h.2.2.2).drop 512)
      (by unfold hmCalls; exact List.mem_cons_self)
    dsimp only at e ⊢
    exact e
  · have e := hm ((c.ports[h.2.2.1]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD [],
      ((rowHidden c h.2.1 h.2.2.2).getD []).map HmNets.Lin.var, hmDigest c h.2.2.1 h.2.2.2)
      (by unfold hmCalls; exact List.mem_cons_of_mem _ List.mem_cons_self)
    unfold rowPfx
    dsimp only at e ⊢
    exact e

theorem rowPfx_length {c : Circuit} {q : ℕ} (hq : q < c.ports.size) : (rowPfx c q).length = 128 := by
  simp only [rowPfx, Array.getElem?_eq_getElem hq, Option.map_some, Option.getD_some]
  exact FlockSoundness.Discharge.Hm.length_rowPrefix _

/-- `CapLower.rows_eq_of_calls` at row prefixes of one length rather than one prefix. -/
theorem rows_eq_of_calls' {l : List (Msg × Msg)} (hT : ∀ e ∈ l, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl l).isSome = false) {c c' : Circuit} {q q' n n' : ℕ} {w w' : Claims.Flock.Wit}
    {row row' : List HmNets.Lin}
    (hs : (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n),
      readForms w ((hmCommit c q n).drop 512)) ∈ l)
    (hr : (rowPfx c q ++ readForms w row, readForms w (hmDigest c q n)) ∈ l)
    (hs' : (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n'),
      readForms w' ((hmCommit c' q' n').drop 512)) ∈ l)
    (hr' : (rowPfx c' q' ++ readForms w' row', readForms w' (hmDigest c' q' n')) ∈ l)
    (hpfx : (rowPfx c q).length = (rowPfx c' q').length)
    (hcm : readForms w (hmCommit c q n) = readForms w' (hmCommit c' q' n')) :
    readForms w row = readForms w' row' := by
  have hsp : FlockSoundness.Refine.K512.saltPrefix = Hm96.Default512.saltPrefix.data.toList := by
    unfold FlockSoundness.Refine.K512; rfl
  have hs1 : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n)) =
      readForms w ((hmCommit c q n).drop 512) := by
    have h := hT _ hs; exact h
  have hr1 : FlockDeploy.sha512 (rowPfx c q ++ readForms w row) = readForms w (hmDigest c q n) := by
    have h := hT _ hr; exact h
  have hs1' : FlockDeploy.sha512 (Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n')) =
      readForms w' ((hmCommit c' q' n').drop 512) := by
    have h := hT _ hs'; exact h
  have hr1' : FlockDeploy.sha512 (rowPfx c' q' ++ readForms w' row') = readForms w' (hmDigest c' q' n') := by
    have h := hT _ hr'; exact h
  have e := CapLower.commit_of_true (c := c) (q := q) (n := n) (w := w) (R := rowPfx c q ++ readForms w row) hs1 hr1
  have e' := CapLower.commit_of_true (c := c') (q := q') (n := n') (w := w') (R := rowPfx c' q' ++ readForms w' row')
    hs1' hr1'
  have hy0 : Hm96.Default512.saltPrefix.data.toList ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst :=
    List.mem_map.2 ⟨_, hs, rfl⟩
  have hy0' : Hm96.Default512.saltPrefix.data.toList ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst :=
    List.mem_map.2 ⟨_, hs', rfl⟩
  have hy : FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c q n) ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0
  have hy' : FlockSoundness.Refine.K512.saltPrefix ++ readForms w' (hmSalt c' q' n') ∈ l.map Prod.fst := by
    rw [hsp]; exact hy0'
  have hr0 : rowPfx c q ++ readForms w row ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr, rfl⟩
  have hr0' : rowPfx c' q' ++ readForms w' row' ∈ l.map Prod.fst := List.mem_map.2 ⟨_, hr', rfl⟩
  have hb := CapLower.commit_bind (r := rowPfx c q ++ readForms w row) (r' := rowPfx c' q' ++ readForms w' row')
    (y := readForms w (hmSalt c q n)) (y' := readForms w' (hmSalt c' q' n')) hT hnc hr0 hr0' hy hy'
    (e.symm.trans (hcm.trans e'))
  exact (List.append_inj hb hpfx).2

/-- **Two holders with one commit string hold one row**, at `layR` and the joint witnesses: where every part's
recorded calls are true and free of collisions, held rows `h` of part `s` and `h'` of part `s'` whose slots carry one
commit string have one row at their ports' unit input columns. -/
theorem held_rows_tie (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    {s s' : ℕ} {c c' : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    (hc' : r.w.parts[s']?.bind partCircuit = some c') {w w' : Claims.Flock.Wit} (hw : ws[s]? = some w)
    (hw' : ws[s']? = some w') {h h' : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s) (hh' : h' ∈ heldRows r s')
    (hcm : readForms w (hmCommit c h.2.2.1 h.2.2.2) = readForms w' (hmCommit c' h'.2.2.1 h'.2.2.2)) :
    readForms w (((rowHidden c h.2.1 h.2.2.2).getD []).map HmNets.Lin.var) =
      readForms w' (((rowHidden c' h'.2.1 h'.2.2.2).getD []).map HmNets.Lin.var) := by
  obtain ⟨h1, h2⟩ := held_calls_layR elem hr hc hw hh
  obtain ⟨h3, h4⟩ := held_calls_layR elem hr hc' hw' hh'
  exact rows_eq_of_calls' hT hnc h1 h2 h3 h4
    ((rowPfx_length (heldRows_port hc hh)).trans (rowPfx_length (heldRows_port hc' hh')).symm) hcm

/-- **Every holder's row is its round's opened row**, at `layR` and the joint witnesses: where every part's recorded
calls are true and free of collisions and every part is staged, checked and agrees with the readers' messages, each
held row `h` of part `s` is the row of round `h.1`'s first holder's slot (`rowSlot`), at that part's witness. -/
theorem held_rowSlot (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) {x : Msg}
    {ws : List Claims.Flock.Wit}
    (hT : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).allCalls x ws)).isSome = false) {r : XR} (hr : readX x = some r)
    (hl : ws.length = r.w.parts.length)
    (hpart : ∀ s (hs : s < ws.length), ∃ (st : Stage) (xO : Msg), StagedPart st s x xO ∧
      Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO ws[s] = true ∧
      (layoutBTops (finOpen (DeployedR.restR elem))).CmtAgrees (cmtTr recRead) x xO ((DeployedR.layR elem).msgs x ws))
    {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) (hK : layoutOf.rounds x = 2 + 2 * rs.size)
    {s : ℕ} {w : Claims.Flock.Wit} (hw : ws[s]? = some w) {c : Circuit}
    (hc : r.w.parts[s]?.bind partCircuit = some c) {h : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s) :
    ∃ σ, OpenSlots.rowSlot r h.1 = some σ ∧ ∃ wσ, ws[σ.s]? = some wσ ∧
      readForms w (((rowHidden c h.2.1 h.2.2.2).getD []).map HmNets.Lin.var) = readForms wσ σ.row := by
  have hsl : s < r.w.parts.length := by
    rcases hps : r.w.parts[s]? with _ | p
    · rw [hps] at hc; cases hc
    · exact (List.getElem?_eq_some_iff.1 hps).1
  have hne : OpenSlots.rowSlot r h.1 ≠ none := by
    unfold OpenSlots.rowSlot
    intro h0
    rw [List.findSome?_eq_none_iff] at h0
    have h0s := h0 s (List.mem_range.2 hsl)
    unfold OpenSlots.rowSlotAt at h0s
    obtain ⟨h', hf⟩ : ∃ h', (heldRows r s).find? (·.1 == h.1) = some h' :=
      Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨h, hh, by simp⟩)
    rw [hf, Option.bind_some, hc, Option.map_some] at h0s
    cases h0s
  obtain ⟨σ, hσ⟩ := Option.ne_none_iff_exists'.1 hne
  refine ⟨σ, hσ, ?_⟩
  obtain ⟨s0, hs0, hσ0⟩ := List.exists_of_findSome?_eq_some hσ
  have hs0l : s0 < ws.length := by rw [hl]; exact List.mem_range.1 hs0
  unfold OpenSlots.rowSlotAt at hσ0
  obtain ⟨h0, hf0, hσ0⟩ := Option.bind_eq_some_iff.1 hσ0
  obtain ⟨c0, hc0, rfl⟩ := Option.map_eq_some_iff.1 hσ0
  have hh0 : h0 ∈ heldRows r s0 := List.mem_of_find?_eq_some hf0
  have hj0 : h0.1 = h.1 := by simpa using List.find?_some hf0
  have hw0 : ws[s0]? = some ws[s0] := List.getElem?_eq_getElem hs0l
  refine ⟨ws[s0], hw0, ?_⟩
  have hsw : s < ws.length := (List.getElem?_eq_some_iff.1 hw).1
  have hw' : ws[s] = w := (List.getElem?_eq_some_iff.1 hw).2
  obtain ⟨st, xO, hS, hck, hA⟩ := hpart s hsw
  obtain ⟨st0, xO0, hS0, hck0, hA0⟩ := hpart s0 hs0l
  rw [hw'] at hck
  obtain ⟨-, e1⟩ := held_commit_ms _ hS hck hA hr hrs hK hc hh
  obtain ⟨-, e2⟩ := held_commit_ms _ hS0 hck0 hA0 hr hrs hK hc0 hh0
  rw [hj0] at e2
  exact held_rows_tie elem hT hnc hr hc hc0 hw hw0 hh hh0 (e1.symm.trans e2)

end RecDeploy.ResHeld
