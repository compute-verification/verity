module

public import Proofs.Flock.C3.Reps.RepsRead
public import Proofs.Flock.C3.Reps.Maps
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

/-! The values the reps' forms read at a stage, at L's decoder. -/

@[expose] public section

namespace FlockC3.Reps

noncomputable section

open Flock

section algebra

def LinReads (f : Flock.RecAlgebra.Lin) (P : Flock.RecAlgebra.Key → Prop) : Prop :=
  (∀ kv ∈ f.t, P kv.1) ∧ ∀ kv ∈ f.q, P kv.1.1 ∧ P kv.1.2

theorem linEval_congr {val val' : Flock.RecAlgebra.Key → F128} {f : Flock.RecAlgebra.Lin}
    {P : Flock.RecAlgebra.Key → Prop} (hf : LinReads f P) (h : ∀ k, P k → val k = val' k) :
    Flock.RecAlgebra.Lin.eval val f = Flock.RecAlgebra.Lin.eval val' f := by
  unfold Flock.RecAlgebra.Lin.eval
  rw [Flock.RecAlgebra.tEval_congr f.t fun kv hkv => h kv.1 (hf.1 kv hkv),
    Flock.RecAlgebra.tEval_congr (g := fun ab => val ab.1 * val ab.2) (g' := fun ab => val' ab.1 * val' ab.2) f.q
      fun kv hkv => by
        show val kv.1.1 * val kv.1.2 = val' kv.1.1 * val' kv.1.2
        rw [h _ (hf.2 kv hkv).1, h _ (hf.2 kv hkv).2]]

theorem valOf_cons {sh : Flock.Shape} {w comb : Array F128} {c c' : ℕ → ℕ → F128} {x : Flock.RecAlgebra.Key}
    (hx : ∀ ℓ l, x ≠ .cons ℓ l) : Flock.RecAlgebra.valOf sh w comb c x = Flock.RecAlgebra.valOf sh w comb c' x := by
  cases x with
  | cons ℓ l => exact absurd rfl (hx ℓ l)
  | _ => rfl

theorem valOf_read {sh : Flock.Shape} {w comb w' comb' : Array F128} {c c' : ℕ → ℕ → F128}
    {P : Flock.RecAlgebra.Key → Prop}
    (hwc : ∀ x, P x → (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh w comb (fun _ _ => 0) x = Flock.RecAlgebra.valOf sh w' comb' (fun _ _ => 0) x)
    (hc : ∀ ℓ l, P (.cons ℓ l) → c ℓ l = c' ℓ l) :
    ∀ x, P x → Flock.RecAlgebra.valOf sh w comb c x = Flock.RecAlgebra.valOf sh w' comb' c' x := by
  intro x hx
  cases x with
  | cons ℓ l => exact hc ℓ l hx
  | _ => exact (valOf_cons (fun _ _ h => by cases h)).trans ((hwc _ hx fun _ _ h => by cases h).trans
      (valOf_cons (fun _ _ h => by cases h)))

theorem consAt_congr {rows rows' : Array (Array F128)} {q : ℕ} {α α' : Array F128} (l : ℕ)
    (h : ∀ j < q, ∀ l : ℕ, (rows[j]!)[l]! = (rows'[j]!)[l]!) (ha : α = α') :
    Flock.RecAlgebra.consAt rows q α l = Flock.RecAlgebra.consAt rows' q α' l := by
  rw [← ha]
  exact Flock.RecAlgebra.rangeSum_congr fun j hj => by rw [h j hj l]

end algebra

def colsAt (Q : FlockSoundness.Discharge.Recursion.VStmt) (bits : Q.Bits) (u : Fin Q.nv) {j : ℕ} (hj : j < Q.c.upv) :
    ℕ → Bool :=
  FlockVBridge.copyVal (FlockVBridge.keyInst Q.hU Q.nv (FlockSoundness.Discharge.Layout.outsOf Q.c Q.hU Q.nv) u) bits hj

theorem rows_congr {Q : FlockSoundness.Discharge.Recursion.VStmt} {bits bits' : Q.Bits} {u : Fin Q.nv} {j LANES : ℕ}
    {r r' a a' cf cf' A A' : Array F128}
    (h : FlockVBridge.Family.OpensAtL Q bits u j LANES r a cf A)
    (h' : FlockVBridge.Family.OpensAtL Q bits' u j LANES r' a' cf' A')
    (hc : ∀ hj : j < Q.c.upv, ∀ t < 128 * LANES, colsAt Q bits u hj t = colsAt Q bits' u hj t) :
    ∀ l : ℕ, r[l]! = r'[l]! := by
  obtain ⟨hj, -, -, _, -, hs, -, hr, -⟩ := h
  obtain ⟨hj', -, -, _, -, hs', -, hr', -⟩ := h'
  intro l
  by_cases hl : l < LANES
  · refine FlockVBridge.AccChain.getElem!_of_bitL fun t ht => ?_
    have hb : 128 * l + t < 8 * (MerkleScheme.rowBytes r).size := by rw [FlockVBridge.size_rowBytes, hs]; omega
    have hb' : 128 * l + t < 8 * (MerkleScheme.rowBytes r').size := by rw [FlockVBridge.size_rowBytes, hs']; omega
    rw [← hr _ hb, ← hr' _ hb', Nat.zero_add]
    exact hc hj _ (by omega)
  · rw [getElem!_neg r l (by omega), getElem!_neg r' l (by omega)]

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf cpO)
open FlockVBridge.AccChain (levelLanes)
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {stmt : Fin S → VStar.Stmt} {sh : Flock.Shape} {Ω C : Type}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}

local notation "ℐ" => holoInnerAt T CT st hp

theorem rep_w (M : RepMaps stmt Vs sh Reg Ω C) (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits)
    (ρ : Fin 2) : (M.rep R ω t l ρ).w = (M.rep₀ l ρ).w :=
  rfl

theorem rep_comb (M : RepMaps stmt Vs sh Reg Ω C) (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits)
    (ρ : Fin 2) : (M.rep R ω t l ρ).comb = (M.rep₀ l ρ).comb := rfl

theorem rep_cons (M : RepMaps stmt Vs sh Reg Ω C) (R : Reg) (ω : Ω) (t : List C) (l : (j : Fin S) → (Vs j).Bits)
    (ρ : Fin 2) :
    (M.rep R ω t l ρ).cons = fun ℓ i => Flock.RecAlgebra.consAt (M.rows l ρ ℓ) (M.q ℓ) (M.alpha R ω t ρ ℓ) i := rfl

def WCombRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun _ _ _ Lb _ _ _ Lb' =>
    ∀ x, K s ρ x → (∀ ℓ l, x ≠ .cons ℓ l) →
      Flock.RecAlgebra.valOf sh (M.rep₀ Lb ρ).w (M.rep₀ Lb ρ).comb (fun _ _ => 0) x =
        Flock.RecAlgebra.valOf sh (M.rep₀ Lb' ρ).w (M.rep₀ Lb' ρ).comb (fun _ _ => 0) x

def RowColsRead (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (M : RepMaps stmt Vs sh Reg Ω C) (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2) ℓ l, K s ρ (.cons ℓ l) → ℓ < sh.sched.levels.size ∧ ∀ j < M.q ℓ,
    ∀ (hj : cpO ℓ ρ j < (Vs (M.sessO ℓ)).c.upv) (Lb Lb' : (x : Fin S) → (Vs x).Bits),
      maskOf Vs (skOf Vs own sel) s Lb = maskOf Vs (skOf Vs own sel) s Lb' → ∀ t < 128 * levelLanes sh ℓ,
        colsAt (Vs (M.sessO ℓ)) (Lb (M.sessO ℓ)) (M.unO ℓ ρ j) hj t =
          colsAt (Vs (M.sessO ℓ)) (Lb' (M.sessO ℓ)) (M.unO ℓ ρ j) hj t

/-- `α` is the history's, not the layers', so its read is a hypothesis. -/
def AlphaRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin))
    (K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop) : Prop :=
  ∀ s (ρ : Fin 2), ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun R ω t _ R' ω' t' _ =>
    ∀ ℓ l, K s ρ (.cons ℓ l) → M.alpha R ω t ρ ℓ = M.alpha R' ω' t' ρ ℓ

theorem formsRead_maps (M : RepMaps stmt Vs sh Reg L.Ω (Cm × (ℐ).Coin)) {K : ℕ → Fin 2 → Flock.RecAlgebra.Key → Prop}
    (hu : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      ∀ (ρ : Fin 2), ∀ ℓ < sh.sched.levels.size,
      ∀ j < M.q ℓ, ∃ a cf A, FlockVBridge.Family.OpensAtL (Vs (M.sessO ℓ))
        (layers zs dg N k Rw R ω t R₂ τ (M.sessO ℓ)) (M.unO ℓ ρ j) (cpO ℓ ρ j) (levelLanes sh ℓ)
        ((M.rows (layers zs dg N k Rw R ω t R₂ τ) ρ ℓ)[j]!) a cf A)
    (hwc : WCombRead (st := st) (hp := hp) zs dg N k Rw own sel M K) (hcols : RowColsRead own sel M K)
    (hal : AlphaRead (st := st) (hp := hp) zs dg N k Rw own sel M K) (s : ℕ) (ρ : Fin 2) :
    ReadAt (st := st) (hp := hp) zs dg N k Rw own sel s fun R ω t Lb R' ω' t' Lb' =>
      ∀ f : Flock.RecAlgebra.Lin, LinReads f (K s ρ) →
        Flock.RecAlgebra.Lin.eval
            (Flock.RecAlgebra.valOf sh (M.rep R ω t Lb ρ).w (M.rep R ω t Lb ρ).comb (M.rep R ω t Lb ρ).cons) f =
          Flock.RecAlgebra.Lin.eval
            (Flock.RecAlgebra.valOf sh (M.rep R' ω' t' Lb' ρ).w (M.rep R' ω' t' Lb' ρ).comb
              (M.rep R' ω' t' Lb' ρ).cons) f := by
  intro R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX f hf
  simp only [rep_w, rep_comb, rep_cons]
  apply linEval_congr hf
  apply valOf_read (hwc s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX)
  intro ℓ l hk
  obtain ⟨hℓ, hc⟩ := hcols s ρ ℓ l hk
  refine consAt_congr l (fun j hj => ?_) (hal s ρ R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX ℓ l hk)
  obtain ⟨a, cf, A, h⟩ := hu R ω t R₂ τ o hA ρ ℓ hℓ j hj
  obtain ⟨a', cf', A', h'⟩ := hu R' ω' t' R₂' τ' o' hA' ρ ℓ hℓ j hj
  exact rows_congr h h' fun hj' => hc j hj hj' _ _ hX

end

end FlockC3.Reps
