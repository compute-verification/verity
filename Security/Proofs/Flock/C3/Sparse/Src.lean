import Proofs.Flock.C3.Sparse.SrcTape
import Proofs.Flock.C3.Compose.Decoder

/-!
# The read fact from the source's sparse phases

`SrcSparseOK`: at every accepted outer proof, each rep's sparse phase plays on the source's run's stage `u.sp i` to
some claim. `sparseReadAt_of_ok`: then `SparseReadAt` holds on the run's own steps, which is what
`sparseRead_c3DecOf` takes. Each play is a RecSparse unit's steps (`unit_of_playsT`; the slots' types are below `T` as
`ExecHolo` types them), and on `runSteps u` the reps' sparse stages sit right past the reps' steps (`sps_prefix`), so
no decode of the steps is needed.
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose FlockC3.PubRows
open FlockVBridge.InnerAccepts (firstSteps)

/-- **On a run's own steps, past the reps' steps, come the reps' sparse stages.** -/
theorem sps_prefix {e : ExecHolo} (u : C3Run e) : u.sp 0 ++ u.sp 1 <+: afterReps (runSteps u) u := by
  have hl : (firstSteps (u.ts 0) ++ u.ts 1).length = 2 * (u.ts 0).length + (u.ts 1).length := by
    rw [List.length_append, firstSteps_length]
  rw [afterReps, runSteps, holoSteps_eq, stagesOf]
  simp only [List.drop_succ_cons, List.drop_zero, List.append_assoc]
  rw [← List.append_assoc (firstSteps (u.ts 0)), List.drop_left' hl]
  exact ⟨_, List.append_assoc _ _ _⟩

/-- **The read fact on a run's own steps**, from each rep's sparse phase playing on its stage to some claim. -/
theorem sparseReadAt_of_plays {e : ExecHolo} {u : C3Run e}
    (hpl : ∀ i : Fin 2, ∃ x, PlaysT (sparseAt e u i) (u.sp i) (some x)) : SparseReadAt e (runSteps u) u := by
  have key := fun i => (hpl i).elim fun _ hx => unit_of_playsT e.cls.shape hx fun _ σ => (e.τ σ).isLt
  choose w τ c hu hs using key
  refine ⟨w, τ, c, hu, ?_⟩
  rw [hs 0, hs 1]
  exact sps_prefix u

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section source

variable {S : ℕ} {Vs : Fin S → VStmt}

/-- **The source's sparse phases play**: at every accepted outer proof, each rep's sparse phase plays on the source's
run's stage `u.sp i` to some claim. -/
def SrcSparseOK (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)) : Prop :=
  AtSrc zs dg N k Rw src fun _ u => ∀ i : Fin 2, ∃ x, PlaysT (sparseAt (holoAt st hp) u i) (u.sp i) (some x)

/-- **The read fact on the source's runs' own steps**, from its sparse phases. -/
theorem sparseReadAt_of_ok {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (h : SrcSparseOK zs dg N k Rw src) :
    AtSrc zs dg N k Rw src fun _ u => SparseReadAt (holoAt st hp) (runSteps u) u :=
  fun R ω t R₂ τ o hacc => sparseReadAt_of_plays (h R ω t R₂ τ o hacc)

end source

section honest

variable {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParses Flock.Tags.circuit ish bind Vs}
  {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).OuterR Reg'}
  {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {tr : Lean.Json} {record : String} {reg : Lean.Json} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {src : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
    C3Run (holoAt st hp)}

local notation "𝒵" => Flock.Guarantees.custodied zr hCust h
local notation "𝓓" => C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc
  (c3DecOf 𝒵 dg N k Rw src own sel) tr record reg

/-- **`SparseRead` at the fixed decoder**, from the source's obligations and its sparse phases. -/
theorem sparseRead_of_ok (hs : SrcOK 𝒵 dg N k Rw src own sel) (hsp : SrcSparseOK 𝒵 dg N k Rw src) :
    SparseRead 𝓓 dg N k Rw :=
  sparseRead_c3DecOf hs (sparseReadAt_of_ok hsp)

end honest

end

end FlockC3.Sparse
