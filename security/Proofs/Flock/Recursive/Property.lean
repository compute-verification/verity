module

public import Proofs.Flock.Recursive.ExecClass
public import Proofs.Flock.Recursive.ZkServer
public import Proofs.Flock.Recursive.Description
public import Proofs.Flock.Recursive.VStarCheck
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

/-! The recursive audit's property as the verifier runs it, `RecursiveAudit`: (i) soundness in fork form and (ii) zero knowledge with circuit privacy (trusted). -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

def VerifiesV {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (zs : Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsV zs o ∧ ∀ j, FailClosed.ScopeZR (zs o.1 j).I

def VerifiesVR {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsVR zr o ∧ ∀ j, FailClosed.ScopeZR (zr o.1 j).I

/-- **The verifier accepts every one of V*'s sessions as V*'s statement of its index**, as the prover
supplies them: each session's proof and the check that it is V*'s statement `j` (`AcceptsVRV`: verity-verify-proof with
`--rec-inner` at `i`, `--rec-part j`, `--rec-bindings` at `bind j` and `--rec-rows` at `rows`, `RecVStar.checkStmt`),
and its statement's form (`FailClosed.ScopeZR`), at the registration the outcome records. -/
def VerifiesVStar {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (i : RecVStar.Inner) (bind : Fin S → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port)) (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  AcceptsVRV i bind rows zr o ∧ ∀ j, FailClosed.ScopeZR (zr o.1 j).I

/-- **Flock's compiled inner protocol as the executable runs it**: `flockInnerCL` at the executable's arithmetic
(`execArith`) and the inner statement's Merkle scheme, SHA-512 with plain leaves (`Refine.enc512`, `Refine.leafPlain`
under `Refine.H512`: `flock-leaf/sha512-unsalted`, which `RecVStar.innerTags` names), on the class `cls`, its regions
and the schedule `sch`. -/
noncomputable abbrev innerExec {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (cls : HClass) (regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog)))
    (sch : FlockSoundness.Accounting.Schedule) : Inner :=
  flockInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain cls regs sch

/-- **The number of V*'s outer statements** over an inner session of a table of `m` variables, `k_log` and `claims`
claims, as V*'s staging lays them out (`rec_vstage.stage`, `VStar.statements`) and the verifier groups the
algebra's residuals (`Flock.RecResiduals.groups`, `rec_residuals.parts`): one `RecOpen` per Ligerito level of
`fast100 m`, then one `InnerRepCheck` per part of the rep's algebra, each residual in S's order in the first part that
stays within `Flock.RecResiduals.MAX_SLOTS` sha512x3 slots, `MAX_REGIONS` regions and 16 residuals, then one `RecCap` a
level: at each later level `l` over both reps' `2^cap_l` nodes of its tops (`VStar.levelCapStatements`,
`RecVStar.checkTopsListed`), and last over the `2^cap₀` nodes of `root_B`'s level-0 cap (`VStar.capStatement`,
`RecVStar.checkCapListed`). Zero at a shape the verifier refuses, where V* stages nothing. -/
def outerParts (m kLog claims : ℕ) : ℕ :=
  match Flock.RecResiduals.shape? m kLog claims with
  | none => 0
  | some sh =>
    match Flock.RecAlgebra.structureOf sh >>= fun S =>
        Flock.RecResiduals.groups S (Flock.RecResiduals.portSizes sh) with
    | .ok gs => sh.sched.levels.size + gs.size + sh.sched.levels.size
    | .error _ => 0

/-- **The number of V*'s outer statements over the class `cls` and its regions `regs`**: `outerParts` at the class's
`m` and `k_log`, and the inner session's claims, two of the protocol's and two a region (`Flock.Stmt.extraClaims`). -/
abbrev outerStatements (cls : HClass) (regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog))) : ℕ :=
  outerParts cls.m cls.kLog (2 + 2 * regs.length)

/-- **Flock's compiled inner protocol at the inner statement `st`**: `innerExec` at its class, regions and schedule
(`execClassAt st`). -/
noncomputable abbrev innerAt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) : Inner :=
  innerExec T CT (execClassAt st).cls (execClassAt st).regs (execClassAt st).sch

/-- **The number of V*'s outer statements over the inner statement `st`**: `outerStatements` at its class and
regions. -/
noncomputable abbrev outerStatementsAt (st : Flock.Stmt) : ℕ :=
  outerStatements (execClassAt st).cls (execClassAt st).regs

/-- **V*'s typed sessions** (`Recursion.zsOf`), at the records' custody (`Flock.Premises.RecordCustody`) and a typing
of every index (`Recursion.Typed`): each session `zr R ω t R₂ j` that types, with the record (`RecordsLiveZKJ`) its
custody gives, and at one that doesn't, a session of its index that types, refusing everything. -/
noncomputable abbrev custodied {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg') (hC : Flock.Premises.RecordCustody zr) (h : Typed zr) :
    A → B → C → D → (j : Fin S) → (Vs j).Outer Reg' :=
  zsOf zr (fun R ω t R₂ j => And.left (hC R ω t R₂ j)) h

/-- **The prover's strategy against the typed sessions** (`Recursion.recT`): its strategy against the sessions as it
supplies them, with the same registrations and rounds. -/
noncomputable abbrev transported {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hC : Flock.Premises.RecordCustody zr) (h : Typed zr)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N)) :
    Strategy (recGame L Reg I Cm fun R ω t => outerV (custodied zr hC h R ω t) dg N) :=
  recT zr (fun R ω t R₂ j => And.left (hC R ω t R₂ j)) h σ

abbrev stmtAt {Ω H : Type} (rt : Flock.Registered.Port) (stmt : (Fin rt.leaves → ByteArray) → Ω → H) :
    (ℕ → ByteArray) → Ω → H :=
  fun q ω => stmt (fun r => q r) ω

abbrev hiddenAt {Ω H : Type} (st : Flock.Stmt) (rt : Flock.Registered.Port)
    (stmt : (Fin rt.leaves → ByteArray) → Ω → H) {Reg : Type} : Reg → Ω → H :=
  fun _ ω => stmtAt rt stmt (Flock.Registered.rowsOf st.c st.pub rt) ω

/-- **The sampling design of a finite space of samples with an element**: `SamplingDesign` at `Ω` and the units each sample takes, nonempty by
the element. Every sampling design is one (`Classical.inhabited_of_nonempty`); (i) binds the space, its element and the sample rather
than a sampling design, whose `nonempty` field would be a hypothesis of the statement. -/
abbrev sampleDesign {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] (draw : Ω → Finset (Fin n)) : SamplingDesign n := ⟨Ω, draw⟩

/-- Obligations, not premises: `VStarParses` and `TagsFound` in (i), and `ZkServerSends` and V*'s staging in (ii).
(i) is in fork form and states no bits; `KSLin2` is proved at the deployment's dense statements
(`Flock.SecurityProofs.RecursiveKS`). -/
def RecursiveAudit : Prop :=
  (∀ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Inhabited (CT k)]
    {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] {draw : Ω → Finset (Fin n)} {Reg Cm : Type} {Reg' Reg₂ : Type}
    (st : Flock.Stmt)
    {Vs : Fin (outerStatementsAt st) → VStmt}
    (zr : Reg → (sampleDesign draw).Ω → List (Cm × (innerAt T CT st).Coin) → Reg₂ →
      (j : Fin (outerStatementsAt st)) → (Vs j).OuterR Reg')
    (_hCust : Flock.Premises.RecordCustody zr)
    (ish : RecVStar.Inner) (bind : Fin (outerStatementsAt st) → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port))
    -- obligation: V*'s statement `j`'s file under `bind j` parses to `Vs j`'s circuit
    (_hob : VStarParses Flock.Tags.circuit ish bind Vs)
    -- obligation on our code: every session's inputs carry the tags their name names, as `verifyCmd` forms them
    (_hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Strategy (recGame (sampleDesign draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N))
    (wmsg : ((j : Fin (outerStatementsAt st)) → (Vs j).Bits) → List (innerAt T CT st).Msg)
    (Sk : ℕ → (j : Fin (outerStatementsAt st)) →
      Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin (outerStatementsAt st)) → Cm →
      PosK (Vs j).c (Vs j).hU (Vs j).nv →
      (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (outerStatementsAt st)) →
      PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      (innerAt T CT st).Msg)
    (rt : Flock.Registered.Port) (stmt : (Fin rt.leaves → ByteArray) → (sampleDesign draw).Ω → Hidden (execClassAt st).cls)
    (_hA3 : Flock.Premises.UniformRandomBytes Vs)
    (p path₀ : ℕ → ByteArray) (_hreg : ∀ r < rt.leaves, Flock.Registered.opens rt (p r) r (path₀ r) = true)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S₀
      (d₀ : Lean.Json) (y₀ : ZkHidden.SampleSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    (_hbr : ∀ h : Typed zr,
      Flock.Premises.VBridgeC (custodied zr _hCust h) dg N k Rw (hiddenAt st rt stmt) wmsg Sk cmt msg),
    prob (fun o => AcceptsStmt st ∧ AcceptsDescription rt st ∧
          VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ (innerAt T CT st).holds (stmtAt rt stmt p o.2.1))
        (recGame (sampleDesign draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N) σ = 0 ∨
    (∃ r < rt.leaves, Flock.Registered.rowsOf st.c st.pub rt r ≠ p r ∧
      Flock.Registered.Collides (Flock.Registered.bcPair rt r (Flock.Registered.rowsOf st.c st.pub rt r)
        (Flock.Registered.pathOf st.pub rt r) (p r) (path₀ r))) ∨
    ∃ h : Typed zr,
      (regFinder (custodied zr _hCust h) dg N (transported zr _hCust h σ)).Finds Refine.H512 ∨
      RoundFork (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) cmt ∨
      SessionFork (custodied zr _hCust h) dg N k Rw M (transported zr _hCust h σ) ∨
      InnerForkRec (custodied zr _hCust h) dg N k Rw (transported zr _hCust h σ) (hiddenAt st rt stmt) wmsg
        Refine.H512 (clashPickCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (execClassAt st).cls
          (execClassAt st).regs (execClassAt st).sch) ∨
      prob (fun o => AcceptsStmt st ∧ AcceptsDescription rt st ∧
            VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ¬ (innerAt T CT st).holds (stmtAt rt stmt p o.2.1))
          (recGame (sampleDesign draw) Reg (innerAt T CT st) Cm fun R ω t => outerVR (zr R ω t) dg N) σ ≤
        expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
            (fun R₂ j τ =>
              (custodied zr _hCust h σ.1 ω t R₂ j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          ((innerAt T CT st).r : ℝ≥0∞) *
            expAt (transported zr _hCust h σ) (fun ω t s => sumV (custodied zr _hCust h σ.1 ω t) dg N
              (fun R₂ j τ => (custodied zr _hCust h σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (Accounting.tableError (execClassAt st).sch
            ⟨(execClassAt st).cls.kLog, (execClassAt st).regs.length, (execClassAt st).cls.mPts⟩)) ∧
  (∀ {Out Prog Wit Rv Ωa Ωp Pub : Type} [Fintype Ωa] [Inhabited Ωa] [Fintype Ωp] [Inhabited Ωp]
    (s : Flock.ZkServer.Schedule)
    (nr : ℕ) (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop)
    (nh : ℕ) (reqs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (T : (Fin nh → ByteArray) → Array Flock.ZkServer.Sent)
    (V : OuterZK.Sessions (OuterZK.HCtx Prog Wit Ωa Ωp nr nh) Pub)
    (pubS : Out → Rv → Ωa → (Fin nh → ByteArray) → Pub) (stopped : OuterZK.Stopped)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit)
    -- what it takes of the code, each an obligation
    (_hF : ZkServerSends s reqs msgs T) (_hpub : SessionsPublic rows pub honest msgs V pubS) (_hC : SessionsCoined V)
    (_hH : SessionsHonest honest msgs V), honest o P w →
    ∀ E : Rv × Ωa × (Array Flock.ZkServer.Sent × Bool) × V.View → Prop,
      fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E ≤ fwSimPr rows pub T V pubS o p₀ E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) ∧
        fwSimPr rows pub T V pubS o p₀ E ≤ fwRealPr rows pub s reqs msgs V (OuterZK.proverExe stopped) P w E +
          (2 * nr * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) +
            V.errAt (ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) +
            nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)))

end Flock.Guarantees
