import Proofs.Flock.Claims.Recursion.NonVacuousRows
import Proofs.Flock.Claims.Recursion.RowDLevels
import Proofs.Flock.Claims.Recursion.VPiecesDeployed

/-!
# `TargetDMerkle` from what the parts accept

`TargetDMerkle` (`NonVacuousRows`, the `DecompB` assembly's row D leaf) is asked at every decision `At` admits: every
inner statement, messages and coins at which the base checks and the parts' pieces `P` hold, an adversarial prover's
included (`hpieces_of_targets`, then `DecompB`'s `∀ c ws`). No honest prover's equation holds there, and the inner
verifier's acceptance is what `DecompB` concludes, so neither can give it. What the decision does carry is the parts'
own checks: row D's `rec-open` pieces among `P` (`openHolds`, each query's `QueryOpens`) and row C's decode (from the
`shaped` base check), which fixes the caps' and openings' shapes.

* `Rows.levelCheck_of_queries`: a level's Merkle check passes where its caps and openings have their shapes
  (`CapsAt`, `LevelShaped`) and each query opens (`QueryOpens`), by `merkleCheck_of`.
* `targetDMerkle_of_accept`: `TargetDMerkle` at any `P` that holds row D's pieces at a side and position (`hP`), where
  those read the decision's side (`hside`) and the verifier rows' positions (`hpos`). The coins are the rep's recorded
  rounds (`cfOf`) from Ligerito's first round (`lgStart`), and the positions are `DecAt.htie`.
* `openHolds_of_vrows`: the parts' pieces `vrows` hold their `rec-open` pieces (`hP` at the assembly's `PD`).
* `targetDMerkle_of_accept_rows`: with nothing open, at the parts' pieces read at `Rows.sideAt rd` and `Rows.posAt`
  (`DecAt.sideAt_eq`, `DecAt.posAt_eq`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt levelCheck)
open RecDeploy.OpenedPlain (QueryOpens LevelShaped levelShaped_of_decode saltLen_plain)
open FlockVBridge.Family (CapsAt cfOf lgStart)
open FlockComplete.Replay.Lg (qOff)

namespace Rows

/-- **A level's check passes where each of its queries opens**: at caps of the schedule's shape (`CapsAt`) and
openings of `pos.size` queries' shape (`LevelShaped`), where every query opens at its position (`QueryOpens`). -/
theorem levelCheck_of_queries {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof} {ℓ : ℕ}
    {pos : Array ℕ} (hℓ : ℓ < sch.levels.size) (hcaps : CapsAt sch cap p) (hsh : LevelShaped sch p ℓ pos.size)
    (hq : ∀ j < pos.size, QueryOpens sch cap p ℓ j pos[j]!) : levelCheck sch cap p ℓ pos = .ok () := by
  obtain ⟨hc1, hc2⟩ := hcaps ℓ hℓ
  obtain ⟨h2, h3, h6, h4⟩ := hsh
  unfold levelCheck
  rw [h4]
  refine FlockVBridge.Family.Conss.merkleCheck_of hc1 h2 h3 rfl hc2 h6 fun j hj => ?_
  obtain ⟨hs, hc⟩ := hq j hj
  unfold opens
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨hs, ?_⟩, by simp; rfl⟩, hc⟩
  have hle : j * ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) + ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) ≤
      pos.size * ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) := by
    have := Nat.mul_le_mul_right ((sch.levels[ℓ]!).d - (sch.levels[ℓ]!).cap) hj
    rwa [Nat.succ_mul] at this
  simp only [sibsOf, Array.length_toList, Array.size_extract]
  omega

end Rows

namespace Top

variable {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop}
  {acc : Msg → List Msg → List Coin → ℕ → Array F128}

/-- **`TargetDMerkle` from the parts' `rec-open` pieces**: wherever `P` holds row D's pieces at `side` and `posAt`
(`hP`), and at a decision those read each rep's decided side (`hside`) and the positions its verifier rows draw
(`hpos`). Each rep's proof passes `MerkleOk` at its recorded rounds from Ligerito's first round: row C's decode gives
the caps' and openings' shapes, and the pieces open every query at the rows' positions (`DecAt.rowDPieces_cover`). -/
theorem targetDMerkle_of_accept (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), P x ms cs →
      ∀ q ∈ Rows.rowDPieces key x, openHolds side posAt q x ms cs)
    (hside : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass)
      (i : ℕ), i < D.sts.size → ∀ r : Fin 2, ∀ q ∈ Rows.rowDPieces key x, q.table = i → q.rep = r.val →
        side x ms q = D.side cls i r)
    (hpos : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass)
      (i : ℕ), i < D.sts.size → ∀ r : Fin 2, ∀ v info kappa,
        verifierRows (D.sh cls i) (D.rep cls (acc x ms cs) i r).rounds (D.rep cls (acc x ms cs) i r).extras
          (D.rep cls (acc x ms cs) i r).comb = .ok (v, info, kappa) →
        ∀ q ∈ Rows.rowDPieces key x, q.table = i → q.rep = r.val →
          posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!) :
    TargetDMerkle rd P acc := by
  intro x ms cs hA D cls i hi r
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, hr, hb, hPx⟩ := id hA
  have hG := D.rowG cls (acc x ms cs) hst hf hx hi
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  have hC := D.rowC_of cls (acc x ms cs) i r hG D.points_eq (idx_lt D hi r)
    (shaped_of_base hr hb D _ (idx_lt D hi r))
  have hd : (D.files cls i).st.merkle.digestLen = 64 := (congrArg MerkleScheme.digestLen hG.merkle).trans rfl
  have hs : (D.files cls i).st.merkle.saltBytes = 0 := (congrArg MerkleScheme.saltBytes hG.merkle).trans rfl
  have hcaps : CapsAt (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito :=
    Rows.capsAt_of_decode hd hC.decode
  have hshape : ∀ ℓ < (D.files cls i).sch.levels.size,
      LevelShaped (D.files cls i).sch ((D.files cls i).proof r).ligerito ℓ ((D.files cls i).sch.levels[ℓ]!).q :=
    fun ℓ hℓ => levelShaped_of_decode (saltLen_plain hs) hd hC.decode ℓ hℓ
  obtain ⟨S, gs, hS, -⟩ := D.structure_ok cls hst hf hx hi D.points_eq
  obtain ⟨⟨v, info, kappa⟩, hv⟩ := D.rowsRun_of cls (acc x ms cs) hti hS (hsz acc hA D cls hi hS) r
  have htie := D.htie cls (acc x ms cs) hG hti r
  have hseam := FlockSoundness.Refine.fast100_level _ _ hG.fast
  refine ⟨cfOf (D.rep cls (acc x ms cs) i r).rounds, lgStart (D.sh cls i) (D.files cls i).extras.size, ?_, htie⟩
  intro ℓ hℓ
  have hpsz : (info.lg.conss[ℓ]!).pos.size = ((D.files cls i).sch.levels[ℓ]!).q := by
    rw [htie v info kappa hv ℓ hℓ]
    exact FlockSoundness.Refine.positions_size _
      (hseam _ (by rw [getElem!_pos _ ℓ hℓ]; exact Array.getElem_mem hℓ)) _
  have hc : levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
      (info.lg.conss[ℓ]!).pos = .ok () := by
    refine Rows.levelCheck_of_queries hℓ hcaps (by rw [hpsz]; exact hshape ℓ hℓ) fun j hj => ?_
    obtain ⟨q, hq, hqt, hqr, hql, hqj⟩ :=
      D.rowDPieces_cover cls key hi r ℓ hℓ j (by rw [← hpsz]; exact hj)
    have h := hP x ms cs hPx q hq
    unfold openHolds at h
    rw [hside x ms cs hA D cls i hi r q hq hqt hqr, hpos x ms cs hA D cls i hi r v info kappa hv q hq hqt hqr,
      hql, hqj] at h
    exact h
  rw [htie v info kappa hv ℓ hℓ] at hc
  rw [Rows.plain_opsWith_std, With.ofScheme_merkleCheck, Rows.opening_eq]
  simpa only [levelCheck, FlockVBridge.Family.Conss.getD_default] using hc

/-- **The parts' pieces hold their `rec-open` pieces**: where every piece of `vrows` holds as `vholds` says, every
`rec-open` piece holds (`targetDMerkle_of_accept`'s `hP` at the assembly's `DecompBR.PD`). -/
theorem openHolds_of_vrows {resP : Msg → List ResPiece} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ}
    {capP : Msg → List CapPiece} {resH : ResPiece → Msg → List Msg → List Coin → Prop}
    {side : Msg → List Msg → OpenPiece → OpenSide} {posAt : Msg → List Coin → OpenPiece → ℕ}
    {capH : CapPiece → Msg → List Msg → List Coin → Prop} {x : Msg} {ms : List Msg} {cs : List Coin}
    (h : RowsHold (VPieces.vrows resP (Rows.rowDPieces key) capP)
      (VPieces.vholds resH (openHolds side posAt) capH) x ms cs) :
    ∀ q ∈ Rows.rowDPieces key x, openHolds side posAt q x ms cs := fun q hq =>
  h (.opn q) (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr (List.mem_map.2 ⟨q, hq, rfl⟩)))))

/-- **`TargetDMerkle` at the parts' pieces, with nothing open**: at pieces whose `rec-open` ones are row D's read at
`Rows.sideAt rd` and `Rows.posAt` (the assembly's `PD … (Rows.sideAt rd) Rows.posAt`), the side and position ties are
`DecAt.sideAt_eq` and `DecAt.posAt_eq`. -/
theorem targetDMerkle_of_accept_rows {resP : Msg → List ResPiece} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ}
    {capP : Msg → List CapPiece} {resH : ResPiece → Msg → List Msg → List Coin → Prop}
    {capH : CapPiece → Msg → List Msg → List Coin → Prop} :
    TargetDMerkle rd (RowsHold (VPieces.vrows resP (Rows.rowDPieces key) capP)
      (VPieces.vholds resH (openHolds (Rows.sideAt rd) Rows.posAt) capH)) acc :=
  targetDMerkle_of_accept key (Rows.sideAt rd) Rows.posAt (fun _ _ _ h => openHolds_of_vrows h)
    (fun _ _ _ _ D cls i hi r q _ ht hr => D.sideAt_eq cls i hi r q ht hr)
    (fun x ms cs hA D cls i hi r _ _ _ hv q _ ht hr => by
      obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
      have hti : i < D.spec.tables.size := by
        obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
        omega
      exact D.posAt_eq cls (acc x ms cs) i (D.names_nodup hst hf hx) hi hti r hv q ht hr)

end Top

end RecDeploy
