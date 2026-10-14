import Proofs.Flock.C3.Sparse.MsgsLayers

/-!
# The sparse tie at the tied source, from V*'s sessions

`srcSparseTie_of_sessions`: `SrcSparseTie` at a source where, at every accepted outer proof, each rep's sparse phase is
in V*'s sessions at the coins record `c3`, the words `w` and the coins `c` read off the run (`SparseSessionAt`), and the
source reads them there (`SparseSrcAt`): `sparseTie_of_session` at the sessions' layers, good where the outer proof is
accepted.
`sparseOf`: `stp` with each rep's sparse stage the steps of `w` at `c`'s phase coins (`stepsOf`), so `SparseSrcAt`'s
stage conjunct holds at its tied source by definition, and its tied reps are `stp`'s. `srcSparseTie_sparseOf`:
`SrcSparseTie` at `tiedSrc (sparseOf stp w c)` from the sessions and `SparseRepTie` at `tiedSrc stp`, the tied rep's
public values being the coins' (`pubOf`, check 5's `rec-sparse-v`) and its `h` w's first 64 elements (RecSparse's `h`
registered as the table's h round, `RecVStar.ownPorts`).
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge.Sparse (SCoins)

/-- **Hypothesis: rep `i`'s output is the RecSparse unit's at w and `c`** (`SparseSrcAt` but its stage): the rep's
public values are the coins `c`'s (`pubOf`), and its `h` is w's first 64 elements. -/
def SparseRepTie (e : ExecHolo) (u : C3Run e) (i : Fin 2) (w : Array Flock.F128) (c : SCoins) : Prop :=
  FlockVBridge.Sparse.pubOf e.cls.shape c =
      holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2 ∧
    ∀ s : Fin 64, fOf w[s.val]! = (u.rep i).2.h s

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`SrcSparseTie` from V*'s sessions**: at every accepted outer proof, rep `i`'s sparse phase is in V*'s sessions at
the coins record `c3`, the words `w` and the phase's coins `c` read off the run (`SparseSessionAt`), and the source reads
them there (`SparseSrcAt`). -/
theorem srcSparseTie_of_sessions
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3) (unit : String)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins)
    (h : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
          (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) ∧
        SparseSrcAt (holoAt st hp) (src R ω t (layers zs dg N k Rw R ω t R₂ τ)) i
          (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i)) :
    SrcSparseTie zs dg N k Rw src := by
  intro R ω t R₂ τ o hA i
  obtain ⟨hs, hsrc⟩ := h R ω t R₂ τ o hA i
  exact sparseTie_of_session (fun s => (hA.1 s).2) hs hsrc

/-- **The source with each rep's sparse stage at the words `w` and coins `c`**: `stp` with rep `i`'s sparse stage the
steps of `w i` at `c i`'s phase coins (`stepsOf`). -/
def sparseOf (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins) :
    Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp) := fun R ω t Lb =>
  { stp R ω t Lb with sp := fun i => stepsOf (holoAt st hp).cls.shape (w R ω t Lb i) (c R ω t Lb i).cs }

/-- **`SrcSparseTie` at the tied source of `sparseOf`**: its stages are the steps of `w` at `c` by definition and its
tied reps are `stp`'s (`tie` reads no sparse stage before the reps' outputs), so V*'s sessions at `c3`, `w` and `c`
(`SparseSessionAt`) and the tied reps' ties to the unit (`SparseRepTie` at `tiedSrc stp`) give it. -/
theorem srcSparseTie_sparseOf
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3) (unit : String)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array Flock.F128)
    (c : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → SCoins)
    (h : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
          (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i) ∧
        SparseRepTie (holoAt st hp) (tiedSrc stp R ω t (layers zs dg N k Rw R ω t R₂ τ)) i
          (w R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (c R ω t (layers zs dg N k Rw R ω t R₂ τ) i)) :
    SrcSparseTie zs dg N k Rw (tiedSrc (sparseOf stp w c)) := by
  refine srcSparseTie_of_sessions c3 unit w c fun R ω t R₂ τ o hA i => ?_
  obtain ⟨hs, hpub, hh⟩ := h R ω t R₂ τ o hA i
  exact ⟨hs, hpub, hh, rfl⟩

end

end FlockC3.Sparse
