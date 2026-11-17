module

public import Proofs.Flock.C3.Opening.SrcOpening
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
import Proofs.Flock.Level3.Assembly

/-! The claim tie, `ClaimRel` from `claimO` to the run's claims, from a point tie and a value tie. -/

@[expose] public section

namespace FlockC3.Opening

section Ties

open Flock Flock.RecAlgebra

theorem shape_m {m c : ℕ} {sh : Flock.Shape} (h : Flock.RecOpening.shape? m c = some sh) : sh.m = m := by
  unfold Flock.RecOpening.shape? at h
  cases hf : Flock.fast100 m with
  | none => simp [hf] at h
  | some s =>
    by_cases hc : c < 1
    · simp [hf, hc] at h
    · simp [hf, hc] at h
      rw [← h]

theorem claimRel_of_ties {c3 : RecCoins.C3} {unit : String} {sh : Shape} {k : ℕ} {a : AClaim Lin}
    {ssh : FlockSoundness.Model.SparseShape} {c : FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh} {wv : F128}
    (ha : coinClaim c3 unit sh k = .ok a) (hm : sh.m = ssh.mReg)
    (hpt : ∀ sc, RecCoins.sparseCoins c3 unit k = .ok sc → ∀ j < ssh.mReg,
      c.coord j = FlockSoundness.Refine.fOf (coinPt c3 sh sc)[j]!)
    (hv : c.value = FlockSoundness.Refine.fOf wv) :
    FlockSoundness.Refine.ClaimRel ssh.mReg (⟨wv, a.skip, a.x⟩ : AClaim F128).toClaim c.toClaim := by
  obtain ⟨sc, hsc, hsize, hpre, hskip, hx⟩ := coinClaim_ok ha
  have h6 : RecSparse.PRE_ROUNDS = 6 := rfl
  obtain ⟨p, hp⟩ : ∃ p, p = coinPt c3 sh sc := ⟨_, rfl⟩
  rw [h6, ← hp] at hskip hx
  rw [h6] at hpre
  rw [← hp] at hsize
  have hpt' : ∀ j < ssh.mReg, c.coord j = FlockSoundness.Refine.fOf p[j]! := by
    rw [hp]
    exact hpt sc hsc
  have hsz6 : (p.extract 0 6).size = 6 := by
    rw [Array.size_extract]
    omega
  refine ⟨hv, ?_, ?_, ?_, ?_⟩
  · show a.skip.size = 64
    rw [hskip, FlockLevel3.eqTable_size, hsz6]
    all_goals rfl
  · intro s
    show FlockSoundness.Model.eqAt (fun j : Fin 6 => c.coord j.val) s.val = FlockSoundness.Refine.fOf a.skip[s.val]!
    rw [hskip, FlockSoundness.Refine.eqTable_vecOf _ 6 hsz6 s.val (lt_of_lt_of_eq s.isLt (by norm_num))]
    congr 1
    funext j
    have hj := j.isLt
    rw [hpt' j.val (by omega)]
    show FlockSoundness.Refine.fOf p[j.val]! = FlockSoundness.Refine.fOf (p.extract 0 6)[j.val]!
    rw [FlockLevel3.extract_get! p 0 6 j.val (by omega) (by omega), Nat.zero_add]
  · show a.x.size = ssh.mReg - 6
    rw [hx, Array.size_extract]
    omega
  · intro i
    have hi := i.isLt
    show c.coord (i.val + 6) = FlockSoundness.Refine.fOf a.x[i.val]!
    rw [hx, FlockLevel3.extract_get! p 6 p.size i.val (by omega) le_rfl, hpt' (i.val + 6) (by omega),
      Nat.add_comm i.val 6]

def PointTie (c3 : RecCoins.C3) (unit : String) (sh : Shape) {ssh : FlockSoundness.Model.SparseShape}
    (c : Fin 2 → FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh) : Prop :=
  ∀ i : Fin 2, ∀ sc, RecCoins.sparseCoins c3 unit i.val = .ok sc → ∀ j < ssh.mReg,
    (c i).coord j = FlockSoundness.Refine.fOf (coinPt c3 sh sc)[j]!

def ValueTie (sh : Shape) (w : Array F128) {ssh : FlockSoundness.Model.SparseShape}
    (c : Fin 2 → FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh) : Prop :=
  ∀ i : Fin 2, (c i).value = FlockSoundness.Refine.fOf w[RecOpening.claimW sh i]!

theorem claimRel_claimO {c3 : RecCoins.C3} {unit : String} {sh : Shape} {a b : AClaim Lin} {w : Array F128}
    {ssh : FlockSoundness.Model.SparseShape} {c : Fin 2 → FlockSoundness.Model.RegClaim FlockLevel3.GF128 ssh}
    (ha : coinClaim c3 unit sh 0 = .ok a) (hb : coinClaim c3 unit sh 1 = .ok b) (hm : sh.m = ssh.mReg)
    (hpt : PointTie c3 unit sh c) (hv : ValueTie sh w c) (i : Fin 2) :
    FlockSoundness.Refine.ClaimRel ssh.mReg (claimO sh w ![a, b] i).toClaim (c i).toClaim := by
  have hi : coinClaim c3 unit sh i.val = .ok (![a, b] i) := by
    fin_cases i
    · exact ha
    · exact hb
  exact claimRel_of_ties hi hm (hpt i) (hv i)

end Ties

section Src

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

theorem openingReadAt_src_ties
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} (hsh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh) (unit : String)
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → RecCoins.C3)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockVBridge.AccChain.Openings)
    (hsess : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      SessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh unit (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ)))
    (hpt : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      PointTie (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit sh
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).claim)
    (hv : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ValueTie sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).claim)
    (htie : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      OpenTie sh (w R ω t (layers zs dg N k Rw R ω t R₂ τ)) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (src R ω t (layers zs dg N k Rw R ω t R₂ τ))) :
    AtSrc zs dg N k Rw src fun _ u => OpeningReadAt (runSteps u) u :=
  openingReadAt_src hsh unit c3 w ops hsess
    (fun R ω t R₂ τ o hacc _ _ ha hb => claimRel_claimO ha hb (shape_m hsh) (hpt R ω t R₂ τ o hacc)
      (hv R ω t R₂ τ o hacc))
    htie

end Src

end FlockC3.Opening
