import Proofs.Flock.VBridge.Compiled
import Proofs.Flock.VBridge.InnerAccepts
import Proofs.Flock.Soundness.Refine.RepConverse
import Proofs.Flock.VBridge.Full

/-!
# VBridge G item 5: `StagesAccept` from one table's files and both reps' runs

`stagesAccept_of_runs` proves `StagesAccept` at the compiled inner protocol on the verifier's own definitions
(`flockInnerCL` at `execArith`, SHA-512 and the schedule `schedOf sch`) from two statements about V*'s layers:

* `TableFiles.Valid` (the file half): V*'s layers hold, for one table, the setup, the session record, the proof files
  and their decodes, the PCS parameters, the committed cap, both reps' tapes in the inner transcript, and the
  transcript's steps (`InnerAccepts.tableTape`). These are facts about V*'s registered reads, like `VDecodes`.
* `TableFiles.Runs` (the algebra half): each rep's transcript program (`Refine.repProg`) returns from its stream's
  replay transcript and consumes every recorded round. `Refine.prog_of_checks` gives it from the stages' checks, and
  E's stage theorems (`zerocheck_of_residuals`, `lincheck_of_residuals`, `ringSwitch_of_residuals`,
  `ligerito_of_residuals`) give the checks from `AlgebraHolds` and `AccHolds`, once V*'s rep values are the decoded
  proof's and the stream's coins are `verifierRows`' coins.

From them, `Refine.verifyRep_of_checks` makes both reps' checks accept with the committed cap, and
`InnerAccepts.inner_accepts_of_reps` makes `flockInnerCL` accept the inner transcript.
-/

namespace FlockVBridge.Stages

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts

local notation "𝔽" => FlockLevel3.GF128

/-- What V*'s layers hold for one table of the inner verifier's session. -/
structure TableFiles (cls : HClass) where
  spec : SessionSpec
  st : Setup
  sess : Session
  sch : Flock.Schedule
  pts : Fin 2 → Fin cls.mPts → 𝔽
  i : ℕ
  extras : Array Claim
  bytes : ℕ → ByteArray
  proof : ℕ → Proof
  stream : ℕ → StreamRec
  tape : ℕ → List Game.Step

namespace TableFiles

variable {cls : HClass}

/-- A rep's stream, as `verifyRep` finds it. -/
def Finds (F : TableFiles cls) (r : ℕ) : Prop :=
  F.sess.streams.find? (fun x => x.name == toString F.spec.tables[F.i]! ++ toString "/rep" ++ toString r) =
    some (F.stream r)

/-- **The file half**: at the hidden statement `M`, the setup realizes `M`'s statement, the session's extra claims are
`extras` and realize the ones at the link points, each rep's stream is in the record, its proof file decodes at the
shape `verifyRep` decodes with, carries the pinned PCS parameters and the committed cap, its steps are a tape of its
spine, and the inner transcript `t`'s steps are the level-0 cap, the link points, both tapes and the openings. -/
def Valid (F : TableFiles cls) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls) {nT nR : ℕ}
    (T : Fin nT → Type) (CT : Fin nR → Type) (t : List (List (SMsg T) × ((k : Fin nR) → CT k))) : Prop :=
  F.st.m = (M.toStatement regs).m ∧ F.st.kLog = (M.toStatement regs).kLog ∧
    FoldRealizes F.st.fold (M.toStatement regs) ∧
    F.st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" ∧
    Flock.fast100 F.st.m = some F.sch ∧
    ExtraRealizes F.st (M.toStatement regs) F.sess.points
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts F.pts) ∧
    (∀ s ∈ F.sess.streams.toList, RoundsWF s.rounds) ∧
    F.st.extra F.sess.points = .ok F.extras ∧
    (∀ r < 2, F.Finds r ∧ decodeProof (shapeOf F.st F.sch F.extras.size) (F.bytes r) = .ok (F.proof r) ∧
      ((F.proof r).params != pcsParams F.st.merkle F.sch) = false ∧
      ((F.proof r).cap.foldl (· ++ ·) ByteArray.empty == F.sess.rootB[F.i]!) = true ∧
      RepTape F.st (M.toStatement regs) F.sch (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts F.pts).length
        (F.proof r) (F.stream r) (F.tape r)) ∧
    schedSteps T CT 0 t = .inl ⟨Cap Dig, capDig (F.proof 0).cap⟩ :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, F.pts⟩ ::
      tableTape (F.tape 0) (F.tape 1) (opensOf (F.proof 0) (F.proof 1) F.sch, saltsOf (F.proof 0) (F.proof 1) F.sch)

/-- **The algebra half**: each rep's transcript program returns from its stream's replay transcript, having consumed
every recorded round. -/
def Runs (F : TableFiles cls) : Prop :=
  ∀ r < 2, ∃ t₄ : Flock.Transcript,
    repProg F.st F.sch (F.proof r) F.extras
        (Transcript.init (F.stream r).rounds (F.spec.domain F.spec.tables[F.i]! r).toUTF8) = .ok ((), t₄) ∧
      t₄.k = (F.stream r).rounds.size

/-- **Both halves make the compiled inner verifier accept `t`.** -/
theorem accepts (F : TableFiles cls) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls) {nT nR : ℕ}
    (T : Fin nT → Type) (CT : Fin nR → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)] (sch : Flock.Schedule)
    (hsch : F.sch = sch) (t : List (List (SMsg T) × ((k : Fin nR) → CT k))) (hv : F.Valid regs M T CT t)
    (hr : F.Runs) : (flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)).accepts M t := by
  subst hsch
  obtain ⟨hmS, hkS, hfold, hms, hf, hext, hwf, hex, hrep, ht⟩ := hv
  have hsize := (hext F.extras hex).1
  have acc : ∀ r < 2, verifyRep F.spec F.st F.sch F.sess F.i r (F.bytes r) = .ok F.sess.rootB[F.i]! := by
    intro r hr2
    obtain ⟨hs, hdec, hpar, hcap, -⟩ := hrep r hr2
    obtain ⟨t₄, hprog, hk⟩ := hr r hr2
    have e := verifyRep_of_checks F.spec F.st F.sch F.sess F.i r (F.bytes r) (F.stream r) F.extras (F.proof r) t₄
      hs hex hdec hpar hprog hk hcap
    rwa [bytes_eq_of_beq hcap] at e
  obtain ⟨hs₀, hdec₀, -, -, ht₀⟩ := hrep 0 (by omega)
  obtain ⟨hs₁, hdec₁, -, -, ht₁⟩ := hrep 1 (by omega)
  rw [hsize] at hdec₀ hdec₁
  exact inner_accepts_of_reps T CT cls regs M F.spec F.st _ rfl hmS hkS hfold hms F.pts F.sess hext hwf F.sch hf F.i
    (F.bytes 0) (F.bytes 1) _ (acc 0 (by omega)) (acc 1 (by omega)) (F.proof 0) (F.proof 1) (F.stream 0) (F.stream 1)
    hdec₀ hdec₁ hs₀ hs₁ (F.tape 0) (F.tape 1) ht₀ ht₁ t ht

end TableFiles

/-! ## `StagesAccept` -/

section outer

variable {n : ℕ} {L : Law n} {Reg : Type} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Flock.Schedule)

/-- The compiled inner protocol on the verifier's own definitions. -/
local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

/-- **G item 5 from one table's files and both reps' runs.** When, at every accepted outer proof, V*'s layers hold a
valid table (`files`, at a schedule `sch`), and, where the reps also meet E's preconditions, both reps' programs run,
then the compiled inner verifier at `schedOf sch` accepts the transcript V*'s layers hold. -/
theorem stagesAccept_of_runs
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (ℐ).Msg)
    (files : ((j : Fin S) → (Vs j).Bits) → TableFiles cls) (hsch : ∀ l, (files l).sch = sch)
    (hfiles : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (files (layers zs dg N k Rw R ω t R₂ τ)).Valid regs (x R ω) T CT
        ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (hruns : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (∀ r, AlgebraHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) ∧
        AccHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r)) →
      (files (layers zs dg N k Rw R ω t R₂ τ)).Runs) :
    StagesAccept (I := ℐ) zs dg N k Rw sh rep x wmsg := by
  intro R ω t R₂ τ o h halg
  exact (files (layers zs dg N k Rw R ω t R₂ τ)).accepts regs (x R ω) T CT sch (hsch _)
    ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)) (hfiles R ω t R₂ τ o h) (hruns R ω t R₂ τ o h halg)

/-- **VBridge G on the verifier's own definitions**, from the pieces still open: `vbridge_compiled` at `ℐ`, with
`StagesAccept` from one table's files and both reps' runs (`stagesAccept_of_runs`) and `VFullR` from V*'s full draw
(`vFullR_of_read`). -/
theorem vbridge_exec
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (ℐ).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (ℐ).Msg)
    (sh : Shape) (rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    -- G items 1–3 and 4
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh fun _ _ _ => rep)
    (hacc : AccChains (I := ℐ) zs dg N k Rw sh fun _ _ _ => rep)
    -- G item 5: one table's files and both reps' runs
    (files : ((j : Fin S) → (Vs j).Bits) → TableFiles cls) (hsch : ∀ l, (files l).sch = sch)
    (hfiles : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (files (layers zs dg N k Rw R ω t R₂ τ)).Valid regs (x R ω) T CT
        ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (hruns : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (∀ r, AlgebraHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r) ∧
        AccHolds sh (rep (layers zs dg N k Rw R ω t R₂ τ) r)) →
      (files (layers zs dg N k Rw R ω t R₂ τ)).Runs)
    -- V*'s public files (G item 6)
    (hcm : VCommits zs Sk cmt) (hdec : VDecodesR wmsg (fun _ _ a b => a = b) Sk msg) :
    Flock.Assumptions.VBridge (I := ℐ) zs dg N k Rw x wmsg Sk cmt msg :=
  vbridge_compiled T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch) zs dg N k Rw x wmsg Sk cmt msg sh rep
    halg hacc (stagesAccept_of_runs T CT cls regs sch zs dg N k Rw sh rep x wmsg files hsch hfiles hruns) hcm hdec
    (vFullR_of_read _ Sk)

end outer

end FlockVBridge.Stages
