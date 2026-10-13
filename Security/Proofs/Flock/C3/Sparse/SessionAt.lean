import Proofs.Flock.C3.Sparse.SrcLayer

/-!
# The sparse tie from V*'s sessions

`SessionAt` says what V*'s sessions give for R's opening; `SparseSessionAt` says it for rep `i`'s sparse phase, in its
style. The verifier's coins accept the phase at its shape (`RecCoins.sparseCoins` at the verifier's RecSparse shape,
`unitShape`), the phase's coins `c.cs` being the coin server's in the order drawn (`sparseCs`). Some copy of the
`rec-sparse` unit in some session (`SparseCopyAt`, which `MsgsLayers` reads at a fixed copy per rep) registers w at the
message columns some computed row reads, the verifier row at `c`
on `v`'s and with `T > 1` the slots' types on `tau`'s, its residuals' output columns registered zero (`RegOut`). The
unit is one whole net (`RecSparse.check` against `unitNet`), so unlike R's opening there is no part to choose.

* `sparseCoins_size`: `sparseCoins`' count check fixes the coins' number, `12 + NU + S_LOG + P_LOG + T_LOG`.
* `SparseSrcAt`: what the source reads there: rep `i`'s public values are the coins' (`pubOf`), its `h` is w's first 64
  elements, and its sparse stage is the steps of w at the coins (`stepsOf`).
* `sparseTieL_of_session`, `sparseTie_of_session`: the two give `SparseTieL` at that session's layer, its outputs from
  their registered zeros (`copyVal_regOut`), so `SparseTie` (`sparseTie_of_layer`).
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge (wbit copyVal keyInst)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.OfVStar (ReadCol rowCol)
open FlockVBridge.AccChain (RegOut copyVal_regOut)
open FlockVBridge.Sparse (SCoins nw nc verifierRow tauBase vBaseS sparseBase check_sparse inWords_sparse shOf)

/-- Rep `r`'s sparse-phase coins in the order `SCoins.cs` holds them: `ζ_c`, `ζ_r`, H3's, `ζ_p`, `ρ_t`. -/
def sparseCs (sc : Flock.RecCoins.SparseCoins) : Array Flock.F128 :=
  sc.zetaC ++ sc.pre ++ sc.rounds ++ sc.zetaP ++ sc.rho

theorem sizes_get {cs : Array (Array Flock.F128)} {want : Array ℕ} (h : cs.map (·.size) = want) (k : ℕ)
    (hk : k < want.size) : (cs[k]!).size = want[k]! := by
  have hk' : k < cs.size := by rw [← h, Array.size_map] at hk; exact hk
  have hm : (cs.map (·.size))[k]! = want[k]! := by rw [h]
  rw [getElem!_pos cs k hk']
  rw [getElem!_pos _ k (by rw [Array.size_map]; exact hk'), Array.getElem_map] at hm
  exact hm

theorem want_get (a b n : ℕ) (t : Array ℕ) :
    (#[a] ++ Array.replicate n 1 ++ #[b] ++ t)[0]! = a ∧ (#[a] ++ Array.replicate n 1 ++ #[b] ++ t)[n + 1]! = b ∧
      (#[a] ++ Array.replicate n 1 ++ #[b] ++ t)[n + 2]! = t[0]! := by
  refine ⟨?_, ?_, ?_⟩
  · rw [getElem!_pos _ 0 (by simp only [Array.size_append, Array.size_replicate, Array.size_singleton]; omega)]
    simp
  · rw [getElem!_pos _ (n + 1) (by simp only [Array.size_append, Array.size_replicate, Array.size_singleton]; omega)]
    simp [Array.getElem_push]
  · by_cases ht : 0 < t.size
    · rw [getElem!_pos _ (n + 2) (by simp only [Array.size_append, Array.size_replicate, Array.size_singleton]; omega),
        getElem!_pos t 0 ht]
      simp
    · rw [getElem!_neg _ (n + 2) (by simp only [Array.size_append, Array.size_replicate, Array.size_singleton]; omega),
        getElem!_neg t 0 ht]

/-- **`sparseCoins`' count check fixes the coins' number**: 6 for `ζ_c`, one a round for H2's 6 and H3's
`NU + S_LOG`, `P_LOG` for `ζ_p` and `T_LOG` for `ρ_t`. -/
theorem sparseCoins_size {c3 : Flock.RecCoins.C3} {unit : String} {r : ℕ} {sc : Flock.RecCoins.SparseCoins}
    (h : Flock.RecCoins.sparseCoins c3 unit r = .ok sc) :
    (sparseCs sc).size = 12 + c3.sparse.degrees.length + c3.sparse.pLog + c3.sparse.tLog := by
  unfold Flock.RecCoins.sparseCoins at h
  obtain ⟨cs0, -, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  generalize Array.filter (fun x => x.size != 0) cs0 = cs at h
  by_cases hc : (cs.map (·.size) != #[Flock.RecSparse.PRE_ROUNDS] ++
      Array.replicate (Flock.RecSparse.PRE_ROUNDS + c3.sparse.degrees.length) 1 ++ #[c3.sparse.pLog] ++
      (if c3.sparse.stacked then #[c3.sparse.tLog] else #[])) = true
  · simp only [hc, ↓reduceIte] at h
    cases h
  · simp only [hc, Bool.false_eq_true, ↓reduceIte] at h
    cases h
    have hw := bne_eq_false_iff_eq.1 (Bool.not_eq_true _ ▸ hc)
    obtain ⟨e0, ep, et⟩ := want_get Flock.RecSparse.PRE_ROUNDS c3.sparse.pLog
      (Flock.RecSparse.PRE_ROUNDS + c3.sparse.degrees.length) (if c3.sparse.stacked then #[c3.sparse.tLog] else #[])
    have h0 := (sizes_get hw 0
      (by simp only [Array.size_append, Array.size_replicate, Array.size_singleton]; omega)).trans e0
    have hp := (sizes_get hw (Flock.RecSparse.PRE_ROUNDS + 1 + c3.sparse.degrees.length)
      (by simp only [Array.size_append, Array.size_replicate, Array.size_singleton]; omega)).trans
      ((congrArg _ (by omega)).trans ep)
    have hP : Flock.RecSparse.PRE_ROUNDS = 6 := rfl
    simp only [sparseCs, Array.size_append, Array.size_map, Array.size_range]
    by_cases hst : c3.sparse.stacked = true
    · have ht := (sizes_get hw (Flock.RecSparse.PRE_ROUNDS + 2 + c3.sparse.degrees.length)
        (by simp only [Array.size_append, Array.size_replicate, Array.size_singleton, hst, ↓reduceIte]; omega)).trans
        ((congrArg _ (by omega)).trans et)
      have e1 : (#[c3.sparse.tLog] : Array ℕ)[0]! = c3.sparse.tLog := rfl
      simp only [hst, ↓reduceIte] at ht ⊢
      omega
    · have h0' : c3.sparse.tLog = 0 := by simp [Flock.RecSparse.Shape.stacked] at hst; omega
      simp only [hst, Bool.false_eq_true, ↓reduceIte, Array.size_empty]
      omega

/-- **What the source reads at the sparse session**: rep `i`'s public values are the coins `c`'s, its `h` is w's first
64 elements, and its sparse stage is the steps of w at `c`'s phase coins. -/
def SparseSrcAt (e : ExecHolo) (u : C3Run e) (i : Fin 2) (w : Array Flock.F128) (c : SCoins) : Prop :=
  FlockVBridge.Sparse.pubOf e.cls.shape c =
      holoPub (e.cls.pubStatement e.regs) e.cls.shape e.cls.units (u.rep i).1 (u.rep i).2 ∧
    (∀ s : Fin 64, fOf w[s.val]! = (u.rep i).2.h s) ∧ u.sp i = stepsOf e.cls.shape w c.cs

section Sessions

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

/-- **Copy `j` of unit `un` in session `s` holds a sparse phase** at w and the phase's coins `c`: it is of a circuit
`RecSparse.check` accepted at the verifier's RecSparse shape, its message columns some computed row reads registered
bits of w, `v`'s of the verifier row at `c`, with `T > 1` `tau`'s of the slots' types, and its residuals' output columns
registered 0 (`RegOut`). -/
def SparseCopyAt (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (e : ExecHolo)
    (w : Array Flock.F128) (c : SCoins) (s : Fin nS) (un : Fin (Vs s).nv) (j : ℕ) (hj : j < (Vs s).c.upv) : Prop :=
  Flock.RecSparse.check (Vs s).c = .ok () ∧ Flock.RecSparse.params (Vs s).c.unit.name = some (unitShape e) ∧
    Lb s (FlockSoundness.Discharge.Const.Prog.oneAt
      (FlockSoundness.Discharge.Placed.keyProg (Vs s).c (Vs s).hU (Vs s).nv) un) = true ∧
    (Vs s).c.unit.unitLog ≤ 32 ∧
    (∀ q < 128 * ((Flock.RecSparse.messagePorts (unitShape e)).map (·.2)).sum,
      ReadCol (FlockSoundness.Rows.ofNet (Vs s).hU) (rowCol (Flock.RecSparse.messagePorts (unitShape e)) 0 q) →
        RegBit (zs s) un j (rowCol (Flock.RecSparse.messagePorts (unitShape e)) 0 q) (wbit (w[·]!) q)) ∧
    (∀ q < 128 * Flock.RecSparse.nv (unitShape e),
      RegBit (zs s) un j (vBaseS (unitShape e) + q) (wbit ((verifierRow e.cls.shape c)[·]!) q)) ∧
    ((unitShape e).stacked = true → ∀ k < (unitShape e).S, ∀ b < 16,
      RegBit (zs s) un j (tauBase (unitShape e) + (16 * k + b)) (((tauArr e)[k]!).testBit b)) ∧
    ∀ t < 128 * (Flock.RecSparse.residualNames (unitShape e)).length,
      RegOut (zs s) un hj (sparseBase (unitShape e) + t) false

/-- **(n7) Rep `i`'s sparse phase in V*'s sessions**, at the sessions `zs` and their layers `Lb`, the verifier's coins
`c3`, w and the phase's coins `c`: `RecCoins.sparseCoins` accepts rep `i`'s coins at the verifier's RecSparse shape and
`c.cs` is them, w has its elements, and some copy holds the phase (`SparseCopyAt`). -/
def SparseSessionAt (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (e : ExecHolo)
    (c3 : Flock.RecCoins.C3) (unit : String) (i : Fin 2) (w : Array Flock.F128) (c : SCoins) : Prop :=
  (∃ sc, Flock.RecCoins.sparseCoins c3 unit i.val = .ok sc ∧ c3.sparse = unitShape e ∧ c.cs = sparseCs sc) ∧
    w.size = nw e.cls.shape ∧
    ∃ (s : Fin nS) (un : Fin (Vs s).nv) (j : ℕ) (hj : j < (Vs s).c.upv), SparseCopyAt zs Lb e w c s un j hj

/-- **(n7's sparse tie from V*'s sessions)** At good layers, the sparse session and what the source reads there give
`SparseTieL` at the session's layer: the coins' count is `nc`'s (`sparseCoins_size`), and the residuals' output
columns, past the unit's inputs, are 0 in the layer (`copyVal_regOut`). -/
theorem sparseTieL_of_session {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits}
    {e : ExecHolo} {c3 : Flock.RecCoins.C3} {unit : String} {i : Fin 2} {w : Array Flock.F128} {c : SCoins}
    {u : C3Run e} (hG : ∀ s, (zs s).Good (Lb s)) (hs : SparseSessionAt zs Lb e c3 unit i w c)
    (hsrc : SparseSrcAt e u i w c) : ∃ s, SparseTieL (zs s) (Lb s) e u i := by
  obtain ⟨⟨sc, hsc, hsh, hcs⟩, hw, s, un, j, hj, hck, hpar, hone, hu, hrw, hrv, hrt, h0⟩ := hs
  obtain ⟨hpub, hh, hst⟩ := hsrc
  have hnc : nc e.cls.shape ≤ c.cs.size := by
    refine Nat.le_of_eq ?_
    show nc (shOf (unitShape e)) = c.cs.size
    rw [hcs, sparseCoins_size hsc, hsh, FlockVBridge.Sparse.degs_length, FlockVBridge.Sparse.nc,
      FlockVBridge.Sparse.pLog_shOf, FlockVBridge.Sparse.shOf_tLog]
  obtain ⟨-, -, -, hn⟩ := check_sparse hck hpar
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName (unitShape e)) (Flock.RecSparse.unitLaid (unitShape e)) =
      .ok (Vs s).c.unit := hn
  have hbase : (FlockSoundness.Rows.ofNet (Vs s).hU).nIn ≤ sparseBase (unitShape e) := by
    rw [FlockSoundness.Discharge.Layout.ofNet_nIn_eq, inWords_sparse hn']
    exact FlockVBridge.Decode.inRows_le_outStart _ _ _
  exact ⟨s, w, c, un, j, hj, hw, hnc, hpub, hh, hst, hck, hpar, hone, hu, hrw, hrv, hrt, fun t ht =>
    copyVal_regOut (zs s) (hG s).2 (FlockVBridge.keyProg_correct (Vs s).hU (Vs s).nv (hG s).1 un) hj (by omega)
      (h0 t ht)⟩

/-- **The sparse tie from V*'s sessions** (`sparseTie_of_layer` at the session's layer). -/
theorem sparseTie_of_session {zs : (s : Fin nS) → (Vs s).Outer Reg'} {Lb : (s : Fin nS) → (Vs s).Bits}
    {e : ExecHolo} {c3 : Flock.RecCoins.C3} {unit : String} {i : Fin 2} {w : Array Flock.F128} {c : SCoins}
    {u : C3Run e} (hG : ∀ s, (zs s).Good (Lb s)) (hs : SparseSessionAt zs Lb e c3 unit i w c)
    (hsrc : SparseSrcAt e u i w c) : SparseTie e u i :=
  (sparseTieL_of_session hG hs hsrc).elim fun s h => sparseTie_of_layer (zs s) (hG s) h

end Sessions

end

end FlockC3.Sparse
