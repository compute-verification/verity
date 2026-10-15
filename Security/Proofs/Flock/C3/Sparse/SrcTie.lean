import Proofs.Flock.C3.Sparse.Src
import Proofs.Flock.VBridge.Sparse.Laid
import Proofs.Flock.VBridge.Sparse.Unit
import Proofs.Flock.VBridge.AccUnits

/-!
# The source's sparse phases from V*'s RecSparse units

`SrcSparseTie`: at every accepted outer proof, each rep's sparse stage `u.sp i` is the steps (`stepsOf`) of message
words `w` and coins `c` at the rep's values, held by a RecSparse unit V*'s check accepted (`SparseTie`): the circuit
passes `RecSparse.check` at the session's shape (`unitShape`), the laid-out unit's rows hold at `v`, its input columns
hold `w`, the verifier row at `c` and the slots' types (`tauArr`), and its residuals' output columns are zero. The tie
states the rows as `LaidRows` of `RecSparse.unitLaid`; `SrcLayer` reads them off V*'s layers, as `ResLayer` does for
`RecResiduals` (`sparseTie_of_layer`). `srcSparseOK_of_tie`: then the source's sparse phases play (`SrcSparseOK`).
`check_sparse` bounds `T_LOG`, and `laid_sparse` reads the zero columns as the unit's residuals being zero, so the unit
accepts and plays its phase on its steps (`playsT_of_unit`).
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockSoundness.Discharge.Hm (LaidRows inRowsOf)
open FlockVBridge.Sparse (SCoins nw nc verifierRow outOf outOf_eq tail_iff shOf out_eq SparseIn sparseInBits sparseBase
  laid_sparse check_sparse)
open FlockVBridge.OfVStar (bitOf_zero)
open FlockVBridge.AccChain (eq_of_bitOf)

/-- The verifier's `RecSparse.Shape` at the session's class (`shOf` of it is `e.cls.shape`). -/
def unitShape (e : ExecHolo) : Flock.RecSparse.Shape := ⟨e.cls.nu, e.cls.slotLog, e.cls.sLog, e.cls.tLog⟩

/-- The slots' types, as the unit's `tau` port holds them. -/
def tauArr (e : ExecHolo) : Array ℕ := Array.ofFn fun σ => (e.τ σ).val

/-- **Rep `i`'s sparse stage is a RecSparse unit V*'s check accepted**: words `w` and coins `c` at the rep's values
(all of `SparseUnit` but its acceptance), whose steps are the stage, and a circuit `RecSparse.check` accepted at the
session's shape whose laid-out unit's rows hold at `v`, with its input columns holding `w`, the verifier row at `c` and
the slots' types, and its residuals' output columns zero (`SparseTieAt`, at those `w` and `c`). -/
abbrev SparseTieAt (e : ExecHolo) (u : C3Run e) (i : Fin 2) (w : Array Flock.F128) (c : SCoins) : Prop :=
  ∃ (cc : Flock.Circuit) (v : ℕ → Bool),
    w.size = nw e.cls.shape ∧ nc e.cls.shape ≤ c.cs.size ∧
    FlockVBridge.Sparse.pubOf e.cls.shape c =
      holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2 ∧
    (∀ s : Fin 64, fOf w[s.val]! = (u.rep i).2.h s) ∧ u.sp i = stepsOf e.cls.shape w c.cs ∧
    Flock.RecSparse.check cc = .ok () ∧ Flock.RecSparse.params cc.unit.name = some (unitShape e) ∧
    LaidRows (Flock.RecSparse.unitLaid (unitShape e)) (inRowsOf (sparseInBits (unitShape e))) v ∧
    SparseIn (unitShape e) (w[·]!) ((verifierRow e.cls.shape c)[·]!) ((tauArr e)[·]!) v ∧
    ∀ t < 128 * (Flock.RecSparse.residualNames (unitShape e)).length, v (sparseBase (unitShape e) + t) = false

@[inherit_doc SparseTieAt]
def SparseTie (e : ExecHolo) (u : C3Run e) (i : Fin 2) : Prop :=
  ∃ (w : Array Flock.F128) (c : SCoins), SparseTieAt e u i w c

theorem outOf_size (sh : Flock.RecSparse.Shape) (rs w v : Array Flock.F128) (τ : Array ℕ) :
    (outOf (shOf sh) rs w v τ).1.size = (Flock.RecSparse.residualNames sh).length := by
  rw [outOf_eq]
  by_cases ht : sh.tLog = 0
  · have hst : sh.stacked = false := by simp [Flock.RecSparse.Shape.stacked, ht]
    simp [ht, hst, Flock.RecSparse.residualNames]
  · have hst : sh.stacked = true := by
      simp only [Flock.RecSparse.Shape.stacked, decide_eq_true_eq]
      omega
    simp [ht, hst, Flock.RecSparse.residualNames]

/-- **A rep tied to an accepted unit plays its sparse phase on its stage** to some claim. -/
theorem plays_of_tie {e : ExecHolo} {u : C3Run e} {i : Fin 2} (h : SparseTie e u i) :
    ∃ x, PlaysT (sparseAt e u i) (u.sp i) (some x) := by
  obtain ⟨w, c, cc, v, hw, hc, hpub, hh, hs, hck, hpar, hL, hin, h0⟩ := h
  have hv := (check_sparse hck hpar).1
  simp only [Flock.RecSparse.Shape.valid, Bool.and_eq_true, decide_eq_true_eq] at hv
  obtain ⟨hbits, -⟩ := laid_sparse (unitShape e) hv.2 v hL w (verifierRow (shOf (unitShape e)) c) (tauArr e) hin
  rw [hs]
  refine playsT_of_unit (τ := tauArr e) ⟨hw, hc, hpub, fun σ => arr_ofFn_get (fun σ => (e.τ σ).val) σ.isLt,
    hh, _, out_eq (shOf (unitShape e)) w (tauArr e) c hw, ⟨fun x hx => ?_, fun y hy => ?_⟩⟩
  · obtain ⟨r, hr, rfl⟩ := Array.mem_iff_getElem.1 hx
    refine eq_of_bitOf fun k hk => (hbits r hr k hk).symm.trans ((h0 _ ?_).trans (bitOf_zero k).symm)
    have := Nat.lt_of_lt_of_eq hr (outOf_size (unitShape e) _ _ _ _)
    omega
  · rw [outOf_eq] at hy
    split at hy
    · simp at hy
    · exact (tail_iff (shOf (unitShape e)) (tauArr e)).2
        (fun s hs => (arr_ofFn_get (fun σ => (e.τ σ).val) hs).trans_lt (e.τ _).isLt) y hy

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

variable {S : ℕ} {Vs : Fin S → VStmt}

/-- **The source tied to V*'s RecSparse units**: at every accepted outer proof, each rep's sparse stage is the steps
of a RecSparse unit V*'s check accepted, at the rep's values (`SparseTie`). -/
def SrcSparseTie (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => ∀ i : Fin 2, SparseTie (holoAt st hp) u i

/-- **The source's sparse phases play** at a source tied to V*'s RecSparse units. -/
theorem srcSparseOK_of_tie {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : SrcSparseTie zs dg N k Rw src) : SrcSparseOK zs dg N k Rw src :=
  fun R ω t R₂ τ o hacc i => plays_of_tie (h R ω t R₂ τ o hacc i)

end

end FlockC3.Sparse
