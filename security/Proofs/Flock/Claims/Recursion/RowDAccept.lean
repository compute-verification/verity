module

public import Proofs.Flock.Claims.Recursion.NonVacuousRows
public import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
public import Proofs.Flock.Claims.Recursion.RowDLevels
public import Proofs.Flock.Claims.Recursion.VPiecesDeployed
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.DecodeTies
import Proofs.Flock.Claims.Recursion.OpenedDecode
import Proofs.Flock.Claims.Recursion.RoundSizes
import Proofs.Flock.Claims.Recursion.VerdictLoop
import Proofs.Flock.Claims.Recursion.VerdictRowD
import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.Claims.Recursion.VerdictRowRDeployed
import Proofs.Flock.Soundness.Refine.Seam

/-! `TargetDMerkle` from what the parts accept, since at an adversarial decision no honest prover's equation holds and the inner verifier's acceptance is what `DecompB` concludes. -/

@[expose] public section

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

theorem queryOpens_congr {s s' : OpenSide} (hs : s.sch = s'.sch)
    (hc : ∀ ℓ, FlockSoundness.Refine.capAt s.cap s.lig ℓ = FlockSoundness.Refine.capAt s'.cap s'.lig ℓ)
    (ho : ∀ R ℓ, FlockSoundness.Refine.openAt s.lig R ℓ = FlockSoundness.Refine.openAt s'.lig R ℓ) {ℓ j pos : ℕ}
    (h : QueryOpens s.sch s.cap s.lig ℓ j pos) : QueryOpens s'.sch s'.cap s'.lig ℓ j pos := by
  first
    | (unfold QueryOpens at h ⊢; rwa [← hs, ← hc, ← ho])
    | (simp only [QueryOpens, hs, hc, ho] at h; exact h)

theorem readRep_absorbed {c : FlockDeploy.RepCtx} {rows : List (List ByteArray)} {q p : Flock.Proof}
    {ms : List Msg} (h : FlockDeploy.readRep c rows q = some (p, ms)) : ∃ a, p = FlockDeploy.absorbedFrom c a q := by
  by_cases hx : ∃ pT : Flock.Proof × F128, FlockDeploy.FitsRep c rows pT.1 pT.2
  · rw [FlockDeploy.readRep, dif_pos hx] at h
    exact ⟨_, (Prod.mk.inj (Option.some.inj h)).1.symm⟩
  · rw [FlockDeploy.readRep, dif_neg hx] at h
    cases h

namespace DecAt

variable {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ)

theorem sideAtR_opens (hi : i < D.sts.size) (r : Fin 2) (q : OpenPiece) (ht : q.table = i) (hr : q.rep = r.val) :
    (OpenLower.sideAtR rd x ms q).sch = (D.side cls i r).sch ∧
      (∀ ℓ, FlockSoundness.Refine.capAt (OpenLower.sideAtR rd x ms q).cap (OpenLower.sideAtR rd x ms q).lig ℓ =
        FlockSoundness.Refine.capAt (D.side cls i r).cap (D.side cls i r).lig ℓ) ∧
      ∀ R ℓ, FlockSoundness.Refine.openAt (OpenLower.sideAtR rd x ms q).lig R ℓ =
        FlockSoundness.Refine.openAt (D.side cls i r).lig R ℓ := by
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, hfl, -, hr₀, hrss, -, hpf, hd⟩ := decR_last D.dec
  obtain ⟨-, hsts⟩ := D.setup_eq
  have hsts' : D.sts = v.2.2 := by rw [hsts, hd]
  have hk : 2 * q.table + q.rep < 2 * v.2.2.size := by
    have := r.isLt
    rw [← hsts']
    omega
  have hk2 : (2 * q.table + q.rep) / 2 < v.2.2.size := by omega
  obtain ⟨ms', hrep⟩ := readPs_get hpf hk
  obtain ⟨T, hfit, -⟩ := FlockDeploy.readRep_fits hrep
  obtain ⟨a, ha⟩ := readRep_absorbed hrep
  obtain ⟨raw, hraw, -⟩ := Option.map_eq_some_iff.1 hr₀
  have hs := sideOf_repCtx (spec := v.2.1) hk2 c₁ r₀ cs.tail.dropLast
    ((FlockDeploy.cutLens (FlockDeploy.repLens v.2.1 v.2.2) rss).getD (2 * q.table + q.rep) [])
    fl.ps[2 * q.table + q.rep]!
  rw [sideOf, hrep, Option.map_some] at hs
  have hRS : sideAt rd x ms q = ⟨(fast100 (FlockDeploy.tabsOf v.2.2)[q.table]!.m).getD default,
      (pf.1[2 * q.table + q.rep]!).cap, (pf.1[2 * q.table + q.rep]!).ligerito⟩ := by
    unfold sideAt schOf capLigAt
    rw [hv, hfl, hr₀, hrss]
    dsimp only
    rw [← hs]
    all_goals rfl
  have h1 : (pf.1[2 * q.table + q.rep]!).cap.toList =
      (FlockDeploy.repCtx v.2.1 v.2.2 default r₀ [] (2 * q.table + q.rep)).cap0 := hfit.cap
  have h2 : (pf.1[2 * q.table + q.rep]!).ligerito.recursiveCaps.toList.map Array.toList =
      FlockDeploy.topGroups (FlockDeploy.repCtx v.2.1 v.2.2 default raw [] (2 * q.table + q.rep)).sched
        ((FlockDeploy.cutLens (FlockDeploy.repLens v.2.1 v.2.2) rss).getD (2 * q.table + q.rep) []) := hfit.tops
  have hcap : OpenLower.capR x ms q = (pf.1[2 * q.table + q.rep]!).cap := by
    unfold OpenLower.capR
    rw [hv, hr₀]
    exact (congrArg List.toArray h1).symm.trans (by simp)
  have hrc : (OpenLower.ligR rd x ms q).recursiveCaps = (pf.1[2 * q.table + q.rep]!).ligerito.recursiveCaps := by
    unfold OpenLower.ligR
    rw [hv, hfl, hraw, hrss]
    first
      | exact (congrArg (fun l : List (List ByteArray) => (l.map List.toArray).toArray) h2).symm.trans
          (by simp [Function.comp_def])
      | (dsimp only; rw [← h2]; simp [Function.comp_def])
  have ho : ∀ R ℓ, FlockSoundness.Refine.openAt (OpenLower.ligR rd x ms q) R ℓ =
      FlockSoundness.Refine.openAt (pf.1[2 * q.table + q.rep]!).ligerito R ℓ := by
    intro R ℓ
    unfold OpenLower.ligR
    rw [hv, hfl, hraw, hrss, ha]
    first
      | rfl
      | (unfold FlockSoundness.Refine.openAt; rfl)
  rw [← D.sideAt_eq cls i hi r q ht hr, hRS]
  refine ⟨?_, fun ℓ => ?_, fun R ℓ => ho R ℓ⟩
  · show OpenLower.schAt x q.table = _
    first
      | (unfold OpenLower.schAt; rw [hv]; all_goals rfl)
      | simp only [OpenLower.schAt, hv]
  · show FlockSoundness.Refine.capAt (OpenLower.capR x ms q) (OpenLower.ligR rd x ms q) ℓ =
      FlockSoundness.Refine.capAt (pf.1[2 * q.table + q.rep]!).cap (pf.1[2 * q.table + q.rep]!).ligerito ℓ
    unfold FlockSoundness.Refine.capAt
    rw [hcap, hrc]

end DecAt

end Rows

namespace Top

variable {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop}
  {acc : Msg → List Msg → List Coin → ℕ → Array F128}

theorem targetDMerkle_of_accept (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ)
    (hP : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), P x ms cs →
      ∀ q ∈ Rows.rowDPieces key x, openHolds side posAt q x ms cs)
    (hside : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass)
      (i : ℕ), i < D.sts.size → ∀ r : Fin 2, ∀ q ∈ Rows.rowDPieces key x, q.table = i → q.rep = r.val →
        (side x ms q).sch = (D.side cls i r).sch ∧
          (∀ ℓ, FlockSoundness.Refine.capAt (side x ms q).cap (side x ms q).lig ℓ =
            FlockSoundness.Refine.capAt (D.side cls i r).cap (D.side cls i r).lig ℓ) ∧
          ∀ R ℓ, FlockSoundness.Refine.openAt (side x ms q).lig R ℓ =
            FlockSoundness.Refine.openAt (D.side cls i r).lig R ℓ)
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
    obtain ⟨hs1, hs2, hs3⟩ := hside x ms cs hA D cls i hi r q hq hqt hqr
    have h' := Rows.queryOpens_congr hs1 hs2 hs3 h
    rw [hpos x ms cs hA D cls i hi r v info kappa hv q hq hqt hqr, hql, hqj] at h'
    exact h'
  rw [htie v info kappa hv ℓ hℓ] at hc
  rw [Rows.plain_opsWith_std, With.ofScheme_merkleCheck, Rows.opening_eq]
  simpa only [levelCheck, FlockVBridge.Family.Conss.getD_default] using hc

theorem openHolds_of_vrows {resP : Msg → List ResPiece} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ}
    {capP : Msg → List CapPiece} {resH : ResPiece → Msg → List Msg → List Coin → Prop}
    {side : Msg → List Msg → OpenPiece → OpenSide} {posAt : Msg → List Coin → OpenPiece → ℕ}
    {capH : CapPiece → Msg → List Msg → List Coin → Prop} {x : Msg} {ms : List Msg} {cs : List Coin}
    (h : RowsHold (VPieces.vrows resP (Rows.rowDPieces key) capP)
      (VPieces.vholds resH (openHolds side posAt) capH) x ms cs) :
    ∀ q ∈ Rows.rowDPieces key x, openHolds side posAt q x ms cs := fun q hq =>
  h (.opn q) (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr (List.mem_map.2 ⟨q, hq, rfl⟩)))))

theorem targetDMerkle_of_accept_rows {resP : Msg → List ResPiece} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ}
    {capP : Msg → List CapPiece} {resH : ResPiece → Msg → List Msg → List Coin → Prop}
    {capH : CapPiece → Msg → List Msg → List Coin → Prop} :
    TargetDMerkle rd (RowsHold (VPieces.vrows resP (Rows.rowDPieces key) capP)
      (VPieces.vholds resH (openHolds (Rows.sideAt rd) Rows.posAt) capH)) acc :=
  targetDMerkle_of_accept key (Rows.sideAt rd) Rows.posAt (fun _ _ _ h => openHolds_of_vrows h)
    (fun _ _ _ _ D cls i hi r q _ ht hr => by
      rw [D.sideAt_eq cls i hi r q ht hr]
      exact ⟨rfl, fun _ => rfl, fun _ _ => rfl⟩)
    (fun x ms cs hA D cls i hi r _ _ _ hv q _ ht hr => by
      obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
      have hti : i < D.spec.tables.size := by
        obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
        omega
      exact D.posAt_eq cls (acc x ms cs) i (D.names_nodup hst hf hx) hi hti r hv q ht hr)

theorem targetDMerkle_of_accept_sideAtR {resP : Msg → List ResPiece} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ}
    {capP : Msg → List CapPiece} {resH : ResPiece → Msg → List Msg → List Coin → Prop}
    {capH : CapPiece → Msg → List Msg → List Coin → Prop} :
    TargetDMerkle rd (RowsHold (VPieces.vrows resP (Rows.rowDPieces key) capP)
      (VPieces.vholds resH (openHolds (OpenLower.sideAtR rd) Rows.posAt) capH)) acc :=
  targetDMerkle_of_accept key (OpenLower.sideAtR rd) Rows.posAt (fun _ _ _ h => openHolds_of_vrows h)
    (fun _ _ _ _ D cls i hi r q _ ht hr => D.sideAtR_opens cls i hi r q ht hr)
    (fun x ms cs hA D cls i hi r _ _ _ hv q _ ht hr => by
      obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
      have hti : i < D.spec.tables.size := by
        obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
        omega
      exact D.posAt_eq cls (acc x ms cs) i (D.names_nodup hst hf hx) hi hti r hv q ht hr)

end Top

end RecDeploy
