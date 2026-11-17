module

public import Proofs.Flock.Recursive.PublicAudit
public import Proofs.Flock.Recursive.HoloExec
public import Proofs.Flock.Recursive.HoloCoins
public import Proofs.Flock.Recursive.HoloMap
public import Proofs.Flock.Recursive.TopFinder
public import Proofs.Flock.Recursive.ValueSrc
public import Proofs.Flock.Recursive.LeafFinder

/-! The recursive audit at candidate 3's inner protocol, the Holo design (`RecursiveAuditHolo`), whose non-vacuity needs VBridgeCOff at candidate 3 (trusted). -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

def c3Of (ish : RecVStar.Inner) (cs : Option Flock.RecVStar.InnerCommits) : Option Flock.RecCoins.C3 := do
  let c ← cs
  let sp ← ish.sparse
  pure { sparse := sp, stacked := ish.stacked.or c.stacked, commits := c }

def innerHeld {Cm Coin : Type} (wr : List (Cm × Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (t : List (Cm × Coin)) : Option Flock.RecVStar.InnerCommits :=
  (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption

/-- `Zk.shapeOk` is not read here. -/
def ChecksC3 (ish : RecVStar.Inner) (cs : Option Flock.RecVStar.InnerCommits) (coins : Option Flock.RecCoins.Source)
    (j : ℕ) (f : VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (I : FlockSoundness.Discharge.Exec.Inputs) (own : Array Flock.Registered.Port)
    (t : FlockSoundness.Discharge.Exec.Transcript) : Prop :=
  ∃ spec sts, FlockSoundness.Discharge.Exec.setupOf (FlockSoundness.Discharge.ZkExec.zkInputs I) t.record =
      .ok (spec, sts) ∧
    sts.forM (Flock.RecVStar.checkStmt (some ish) I.tags I.tables · I.circuitFile I.partition I.program (some own)
      (some j) (some f) rows) = .ok () ∧
    sts.forM (Flock.RecCoins.check · (some own) coins (c3Of ish cs)) = .ok () ∧
    sts.forM (Flock.RecVStar.checkPubRows · (some own) cs ish.sparse) = .ok () ∧
    Flock.RecVStar.checkLabels (FlockSoundness.Discharge.Exec.recordDraw t.record) I.count sts = .ok () ∧
    ∀ c, cs = some c → (Flock.RecVStar.checkCovers ish c).isOk = true

theorem checksC3_of_recordSample {ish : RecVStar.Inner} {cs : Option Flock.RecVStar.InnerCommits}
    {coins : Option Flock.RecCoins.Source} {j : ℕ} {f : VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)} {I : FlockSoundness.Discharge.Exec.Inputs}
    {own : Array Flock.Registered.Port} {t t' : FlockSoundness.Discharge.Exec.Transcript}
    (h : ChecksC3 ish cs coins j f rows I own t)
    (hrec : FlockSoundness.Discharge.Exec.recordDraw t'.record = FlockSoundness.Discharge.Exec.recordDraw t.record) :
    ChecksC3 ish cs coins j f rows I own t' := by
  have hset : FlockSoundness.Discharge.Exec.setupOf (FlockSoundness.Discharge.ZkExec.zkInputs I) t'.record =
      FlockSoundness.Discharge.Exec.setupOf (FlockSoundness.Discharge.ZkExec.zkInputs I) t.record := by
    unfold FlockSoundness.Discharge.Exec.setupOf
    rw [hrec]
  unfold ChecksC3 at h ⊢
  rw [hset, hrec]
  exact h

def ChecksVR {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} (ish : RecVStar.Inner)
    (cs : Option Flock.RecVStar.InnerCommits) (coins : Option Flock.RecCoins.Source) (bind : Fin S → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port)) (zr : Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    {dg : ByteArray → ByteArray} {N : ℕ} (o : Reg₂ × VRec Vs Reg' dg N) : Prop :=
  ∀ j : Fin S, ChecksC3 ish cs coins j (bind j) rows (zr o.1 j).I (zr o.1 j).own (o.2 j).2.2.2

open Classical in
noncomputable def keepOwn {Q : VStmt} {Reg' : Type} (z : Q.Outer Reg')
    (P : FlockSoundness.Discharge.Exec.Inputs → Array Flock.Registered.Port →
      FlockSoundness.Discharge.Exec.Transcript → Prop) : Q.Outer Reg' :=
  { z with
    own := if ∃ R' t, z.Writes R' t ∧ AcceptsZKR z.I z.own t ∧ P z.I z.own t then z.own else freshOwn Q.c
    hown := fun ω R' o h => by
      by_cases hp : ∃ R' t, z.Writes R' t ∧ AcceptsZKR z.I z.own t ∧ P z.I z.own t
      · rw [ite_eq_left hp] at h ⊢
        exact z.hown ω R' o h
      · rw [ite_eq_right hp] at h
        exact (not_acceptsZKR_fresh z.I z.ht z.hTags z.hcH _ h).elim }

noncomputable abbrev checkedC3 {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B D : Type} {I : Inner} {Cm : Type}
    (ish : RecVStar.Inner) (cs : List (Cm × I.Coin) → Option Flock.RecVStar.InnerCommits)
    (coins : List (Cm × I.Coin) → Option Flock.RecCoins.Source)
    (bind : Fin S → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    (zr : A → B → List (Cm × I.Coin) → D → (j : Fin S) → (Vs j).OuterR Reg') (hC : Flock.Premises.RecordCustody zr)
    (h : Typed zr) : A → B → List (Cm × I.Coin) → D → (j : Fin S) → (Vs j).Outer Reg' :=
  fun R ω t R₂ j => keepOwn (custodied zr hC h R ω t R₂ j) fun J own tr =>
    I.r ≤ t.length ∧ ChecksC3 ish (cs t) (coins t) j (bind j) rows J own tr

noncomputable def checkedT {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ} (ish : RecVStar.Inner)
    (cs : List (Cm × I.Coin) → Option Flock.RecVStar.InnerCommits)
    (coins : List (Cm × I.Coin) → Option Flock.RecCoins.Source) (bind : Fin S → VStar.Bindings)
    (rows : Option (Array Flock.Registered.Port))
    (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hC : Flock.Premises.RecordCustody zr) (h : Typed zr)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N)) :
    Strategy (recGame L Reg I Cm fun R ω t => outerV (checkedC3 ish cs coins bind rows zr hC h R ω t) dg N) :=
  transported zr hC h σ

def RunOpensReg {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    (st : Flock.Stmt) (hp : HoloParams) {n : ℕ} {L : SamplingDesign n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt}
    {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) (pos : ℕ) (col : List FlockLevel3.GF128) : Prop :=
  ∃ o : Reg × L.Ω × List (Cm × (holoInnerAt T CT st hp).Coin) × (Reg₂ × VRec Vs Reg' dg N),
    Game.Outcome (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
    o.2.1 = ω ∧ ∃ out : OutHL FlockLevel3.GF128 Refine.Dig (List UInt8),
      outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)
          (HoloReg.mk (fun _ _ => 0) cap) (decodedV zs dg N k Rw wmsg o.1 o.2.1 o.2.2.1 o.2.2.2) = some out ∧
        ∃ y ∈ slotOpensL (out.reg cap) 0 0 (holoAt st hp).rsch.sch.level₀.queries, y.1 = pos ∧
          VerifiedL Refine.H512 Refine.enc512 Refine.leafPlain cap (holoAt st hp).rsch.sch.level₀.lanes
            (pathLen (holoAt st hp).rsch.sch.level₀) y ∧ y.2.1 = col

open Classical in
noncomputable def regWord {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) {n : ℕ} {L : SamplingDesign n} {Reg Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) : Oracle FlockLevel3.GF128 := fun ℓ pos =>
  if h : ∃ col, RunOpensReg T CT st hp zs dg N k Rw wmsg σ cap ω pos col then h.choose.getD ℓ 0 else 0

noncomputable def regStmt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) {n : ℕ} {L : SamplingDesign n} {Reg Cm : Type} {S : ℕ}
    {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAt T CT st hp).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAt T CT st hp) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) : Reg → L.Ω → HoloReg FlockLevel3.GF128 Refine.Dig :=
  fun _ ω => HoloReg.mk (regWord T CT st hp zs dg N k Rw wmsg σ cap ω) cap

/-- Each application is explicitly at its own typing's protocol, so no statement of it compares two typings' protocols. -/
def RunOpensRegτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
    (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) {n : ℕ} {L : SamplingDesign n}
    {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAtτ T CT st hp τv).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) (pos : ℕ) (col : List FlockLevel3.GF128) : Prop :=
  ∃ o : Reg × L.Ω × List (Cm × (holoInnerAtτ T CT st hp τv).Coin) × (Reg₂ × VRec Vs Reg' dg N),
    Game.Outcome (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
    o.2.1 = ω ∧ ∃ out : OutHL FlockLevel3.GF128 Refine.Dig (List UInt8),
      outHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAtτ st hp τv)
          (HoloReg.mk (fun _ _ => 0) cap) (decodedV zs dg N k Rw wmsg o.1 o.2.1 o.2.2.1 o.2.2.2) = some out ∧
        ∃ y ∈ slotOpensL (out.reg cap) 0 0 (holoAtτ st hp τv).rsch.sch.level₀.queries, y.1 = pos ∧
          VerifiedL Refine.H512 Refine.enc512 Refine.leafPlain cap (holoAtτ st hp τv).rsch.sch.level₀.lanes
            (pathLen (holoAtτ st hp τv).rsch.sch.level₀) y ∧ y.2.1 = col

open Classical in
noncomputable def regWordτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) {n : ℕ}
    {L : SamplingDesign n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAtτ T CT st hp τv).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) (ω : L.Ω) : Oracle FlockLevel3.GF128 := fun ℓ pos =>
  if h : ∃ col, RunOpensRegτ T CT st hp τv zs dg N k Rw wmsg σ cap ω pos col then h.choose.getD ℓ 0 else 0

noncomputable def regStmtτ {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) {n : ℕ}
    {L : SamplingDesign n} {Reg Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
    (zs : Reg → L.Ω → List (Cm × (holoInnerAtτ T CT st hp τv).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (holoInnerAtτ T CT st hp τv).Msg)
    (σ : Strategy (recGame L Reg (holoInnerAtτ T CT st hp τv) Cm fun R ω t => outerV (zs R ω t) dg N))
    (cap : Cap Refine.Dig) : Reg → L.Ω → HoloReg FlockLevel3.GF128 Refine.Dig :=
  fun _ ω => HoloReg.mk (regWordτ T CT st hp τv zs dg N k Rw wmsg σ cap ω) cap

noncomputable def holoTys (st : Flock.Stmt) (hp : HoloParams) : Fin 16 → Type :=
  ![FlockSoundness.Model.Cap Refine.Dig, Bool,
    FlockSoundness.Model.Opens FlockLevel3.GF128 Refine.Dig × FlockSoundness.Model.Salts (List UInt8),
    (Fin 64 → FlockLevel3.GF128) × (Fin 64 → FlockLevel3.GF128), FlockLevel3.GF128 × FlockLevel3.GF128,
    Fin 64 → FlockLevel3.GF128, Fin 128 → FlockLevel3.GF128, FlockLevel3.GF128,
    FlockLevel3.GF256 × FlockLevel3.GF256, Fin (2 ^ ((holoAt st hp).sch.ext + 1)) → FlockLevel3.GF128,
    Fin (2 ^ ((holoAt st hp).rsch.sch.ext + 1)) → FlockLevel3.GF128,
    FlockLevel3.GF128 × (Fin 1 → FlockLevel3.GF128),
    FlockLevel3.GF128 × (Fin (2 * (holoAt st hp).cls.shape.ell) → FlockLevel3.GF128),
    FlockLevel3.GF128 × (Fin (2 * (holoAt st hp).cls.shape.ell + 1) → FlockLevel3.GF128),
    Fin (2 ^ (holoAt st hp).cls.shape.pLog) → FlockLevel3.GF128,
    Fin (2 ^ (holoAt st hp).cls.shape.tLog) → FlockLevel3.GF128]

/-- The slots' types are bound existentially because the verifier does not hold `hp.τ` (partitions are private); their
range `Fin (2^tLog)` is the one the zero outputs pin (check 5's `rec-sparse-zero` rows). -/
def HoloSatSome (st : Flock.Stmt) (hp : HoloParams) (x : HoloReg FlockLevel3.GF128 Refine.Dig) : Prop :=
  ∃ (τ : Fin (2 ^ (holoAt st hp).cls.sLog) → Fin (2 ^ (holoAt st hp).cls.tLog))
    (R : RegTable (holoAt st hp).cls),
    (holoAt st hp).rsch.Close execArith x.w R ∧ ∃ z, (R.toStatement τ (holoAt st hp).regs).Satisfies z

/- `holoAt` stays applied, never unfolded: each application at a typing is at that typing's protocol, so no defeq compares
two typings' protocols, which would unfold `holoTableCL`'s body. The attribute makes an accidental crossing fail fast. -/
attribute [local irreducible] holoAt

/-- Candidate 3 is not deployed: only its tests' fixture runs its inner session. Check 5's source `coins` is any, though no
reader builds one and the verifier accepts none at candidate 3 (`FlockC3.VStarList.hcoins_of_acceptsC3`). -/
def RecursiveAuditC3At {r : Flock.Stmt → HoloParams → ℕ}
    (CT : (st : Flock.Stmt) → (hp : HoloParams) → Fin (r st hp) → Type) [∀ st hp k, Fintype (CT st hp k)]
    [∀ st hp k, Inhabited (CT st hp k)]
    (im : (st : Flock.Stmt) → (hp : HoloParams) → (Cm : Type) → RecVStar.Inner → String →
      (Cm → String → Nat → List UInt8) →
        FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin (r st hp)) → CT st hp k)) :
    Prop :=
  ∀
    {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] {draw : Ω → Finset (Fin n)} {Reg Cm : Type} {Reg' Reg₂ : Type}
    (obj : Lean.Json) (prog : ByteArray) (st : Flock.Stmt) (ish : RecVStar.Inner),
    -- the sparse phase's parameters are derived from that accepted shape and statement, not another verifier input
    let hp := holoParamsAt ish st
    ∀
    (reg : Lean.Json) (wr : List (Cm × ((k : Fin (r st hp)) → CT st hp k)) → Lean.Json × String)
    (coins : String → Option Flock.RecCoins.Source)
    -- R's cap and the salts the prover's workers committed R's level-0 top rows under, which they hold
    -- (`registrant-holds-openings`: the guarantee is applied at the held salts)
    (cap : Cap Refine.Dig) (salt : ℕ → ByteArray) (bcs' : Array ByteArray)
    (_hstk : StackedTopAt (holoAt st hp).rsch.sch.level₀ st reg cap salt bcs')
    {Vs : Fin (vstarCount ish) → VStmt}
    (zr : Reg → (sampleDesign draw).Ω → List (Cm × ((k : Fin (r st hp)) → CT st hp k)) → Reg₂ →
      (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg')
    (_hCust : Flock.Premises.RecordCustody zr)
    (_hRC : Flock.Premises.RegistrantCovers zr)
    (tbl : String) (bc : Cm → String → Nat → List UInt8)
    (_hIC : Flock.Premises.InnerRecordCustody wr (fun w => innerCommitsAt w.1 w.2 st reg) ish.sparse
      (im st hp Cm ish tbl bc))
    (_htbl : ∀ i : Fin 2, (((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.SPARSE}/" ||
      ((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.STACKED}/") = false)
    (bind : Fin (vstarCount ish) → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    -- obligation: where V*'s statement `j`'s file under `bind j` parses into scope, as the verifier's acceptance parses it,
    -- it parses to `Vs j`'s circuit
    (_hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    -- obligation on our code: every session's inputs carry the tags their name names
    (_hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover, at the audit's game written out (`recGame` at the inner protocol's parameter-level coin
    -- type and round count), so its type reads no typing's protocol
    (σ : Strategy (Game.send Reg fun R => Game.coin (sampleDesign draw).Ω fun ω =>
      rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
        (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)))
    (wmsg : ((j : Fin (vstarCount ish)) → (Vs j).Bits) → List (List (SMsg (holoTys st hp))))
    (Sk : ℕ → (j : Fin (vstarCount ish)) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv →
      (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (vstarCount ish)) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      List (SMsg (holoTys st hp)))
    (_hA3 : Flock.Premises.UniformRandomBytes Vs)
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S₀
      (d₀ : Lean.Json) (y₀ : ZkHidden.SampleSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    -- VBridge at the committed rounds, a premise, at some typing `τv` of the slots: its discharge takes the registered one,
    -- which the zero outputs pin below `2^tLog`
    (_hbr : ∀ h : Typed zr, ∃ τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog),
      Flock.Premises.VBridgeCOff
        (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv)
        (fun R ω t R₂ o =>
          ValueColl (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h)
            (valueSrcAll (fun t => coins (wr t).2) fun t => c3Of ish (innerHeld wr st reg t)) R ω t R₂ ∨
          LeafColl (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) R ω t R₂ o ∨
          SaltColl (holoAt st hp).rsch.sch.level₀ cap salt
            (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h) R ω t R₂ ∨
          TopColl wr st reg bcs' t)
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg
        N k Rw (regStmtτ (holoTys st hp) (CT st hp) st hp τv
          (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
          (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap)
        wmsg Sk cmt msg),
    prob (fun o => (AcceptsPublic obj prog st ∧ AcceptsC3 ish st (wr o.2.2.1).1 (wr o.2.2.1).2 reg) ∧
          VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ChecksVR ish (innerHeld wr st reg o.2.2.1) (coins (wr o.2.2.1).2) bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2)
        (Game.send Reg fun R => Game.coin (sampleDesign draw).Ω fun ω =>
          rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
            (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)) σ = 0 ∨
    ∃ h : Typed zr,
      (regFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (topFinder wr st reg bcs'
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (valueFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (valueSrcAll (fun t => coins (wr t).2) fun t => c3Of ish (innerHeld wr st reg t)) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (leafFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (saltFinder (holoAt st hp).rsch.sch.level₀ cap salt
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      RoundFork (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ) cmt ∨
      SessionFork (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw M
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ) ∨
      InnerForkRec (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)
        (regStmtτ (holoTys st hp) (CT st hp) st hp hp.τ
          (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
          (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap)
        wmsg Refine.H512
        (clashPickHL (holoTys st hp) (CT st hp) execArith Refine.H512 Refine.enc512 Refine.leafPlain
          (holoAtτ st hp hp.τ)) ∨
      prob (fun o => (AcceptsPublic obj prog st ∧ AcceptsC3 ish st (wr o.2.2.1).1 (wr o.2.2.1).2 reg) ∧
            VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ChecksVR ish (innerHeld wr st reg o.2.2.1) (coins (wr o.2.2.1).2) bind rows (zr o.1 o.2.1 o.2.2.1)
              o.2.2.2 ∧
            ¬ HoloSatSome st hp
              (regStmtτ (holoTys st hp) (CT st hp) st hp hp.τ
                (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
                (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap o.1 o.2.1))
          (Game.send Reg fun R => Game.coin (sampleDesign draw).Ω fun ω =>
            rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
              (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)) σ ≤
        expAt (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) (fun ω t s =>
            sumV (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t) dg N
            (fun R₂ j τ =>
              (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t R₂
                j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          ((r st hp + 1 : ℕ) : ℝ≥0∞) *
            expAt (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ) (fun ω t s =>
              sumV (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t) dg N
              (fun R₂ j τ =>
                (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t R₂
                  j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp))

/-- **`RecursiveAuditC3At` with no A3**: `RecursiveAuditC3At` without its A3 premise, at the same bound. V*'s sampling designs are
the full sample (`--draw subset:<units>`), which samples every unit at every value of the operating system's bytes
(`execOS_subset_full_mem`), so no session's miss reads those bytes. -/
def RecursiveAuditC3AtNoA3 {r : Flock.Stmt → HoloParams → ℕ}
    (CT : (st : Flock.Stmt) → (hp : HoloParams) → Fin (r st hp) → Type) [∀ st hp k, Fintype (CT st hp k)]
    [∀ st hp k, Inhabited (CT st hp k)]
    (im : (st : Flock.Stmt) → (hp : HoloParams) → (Cm : Type) → RecVStar.Inner → String →
      (Cm → String → Nat → List UInt8) →
        FlockSoundness.Assumptions.InnerRecordMap Cm ((k : Fin (r st hp)) → CT st hp k)) :
    Prop :=
  ∀
    {n : ℕ} {Ω : Type} [Fintype Ω] [Inhabited Ω] {draw : Ω → Finset (Fin n)} {Reg Cm : Type} {Reg' Reg₂ : Type}
    (obj : Lean.Json) (prog : ByteArray) (st : Flock.Stmt) (ish : RecVStar.Inner),
    -- the sparse phase's parameters are derived from that accepted shape and statement, not another verifier input
    let hp := holoParamsAt ish st
    ∀
    (reg : Lean.Json) (wr : List (Cm × ((k : Fin (r st hp)) → CT st hp k)) → Lean.Json × String)
    (coins : String → Option Flock.RecCoins.Source)
    -- R's cap and the salts the prover's workers committed R's level-0 top rows under, which they hold
    -- (`registrant-holds-openings`: the guarantee is applied at the held salts)
    (cap : Cap Refine.Dig) (salt : ℕ → ByteArray) (bcs' : Array ByteArray)
    (_hstk : StackedTopAt (holoAt st hp).rsch.sch.level₀ st reg cap salt bcs')
    {Vs : Fin (vstarCount ish) → VStmt}
    (zr : Reg → (sampleDesign draw).Ω → List (Cm × ((k : Fin (r st hp)) → CT st hp k)) → Reg₂ →
      (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg')
    (_hCust : Flock.Premises.RecordCustody zr)
    (_hRC : Flock.Premises.RegistrantCovers zr)
    (tbl : String) (bc : Cm → String → Nat → List UInt8)
    (_hIC : Flock.Premises.InnerRecordCustody wr (fun w => innerCommitsAt w.1 w.2 st reg) ish.sparse
      (im st hp Cm ish tbl bc))
    (_htbl : ∀ i : Fin 2, (((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.SPARSE}/" ||
      ((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.STACKED}/") = false)
    (bind : Fin (vstarCount ish) → VStar.Bindings) (rows : Option (Array Flock.Registered.Port))
    -- obligation: where V*'s statement `j`'s file under `bind j` parses into scope, as the verifier's acceptance parses it,
    -- it parses to `Vs j`'s circuit
    (_hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    -- obligation on our code: every session's inputs carry the tags their name names
    (_hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the recursive prover, at the audit's game written out (`recGame` at the inner protocol's parameter-level coin
    -- type and round count), so its type reads no typing's protocol
    (σ : Strategy (Game.send Reg fun R => Game.coin (sampleDesign draw).Ω fun ω =>
      rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
        (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)))
    (wmsg : ((j : Fin (vstarCount ish)) → (Vs j).Bits) → List (List (SMsg (holoTys st hp))))
    (Sk : ℕ → (j : Fin (vstarCount ish)) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv →
      (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin (vstarCount ish)) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) →
      List (SMsg (holoTys st hp)))
    {k Rw M : ℕ} (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ R ω t R₂ j S₀
      (d₀ : Lean.Json) (y₀ : ZkHidden.SampleSetupZHJ (inputsJ (zr R ω t R₂ j).I 0) (Vs j).mPts S₀ d₀) S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zr R ω t R₂ j).dj y₀ S' R') i k ≤ ρ)
    -- VBridge at the committed rounds, a premise, at some typing `τv` of the slots: its discharge takes the registered one,
    -- which the zero outputs pin below `2^tLog`
    (_hbr : ∀ h : Typed zr, ∃ τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog),
      Flock.Premises.VBridgeCOff
        (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv)
        (fun R ω t R₂ o =>
          ValueColl (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h)
            (valueSrcAll (fun t => coins (wr t).2) fun t => c3Of ish (innerHeld wr st reg t)) R ω t R₂ ∨
          LeafColl (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) R ω t R₂ o ∨
          SaltColl (holoAt st hp).rsch.sch.level₀ cap salt
            (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h) R ω t R₂ ∨
          TopColl wr st reg bcs' t)
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg
        N k Rw (regStmtτ (holoTys st hp) (CT st hp) st hp τv
          (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
          (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp τv) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap)
        wmsg Sk cmt msg),
    prob (fun o => (AcceptsPublic obj prog st ∧ AcceptsC3 ish st (wr o.2.2.1).1 (wr o.2.2.1).2 reg) ∧
          VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
          ChecksVR ish (innerHeld wr st reg o.2.2.1) (coins (wr o.2.2.1).2) bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2)
        (Game.send Reg fun R => Game.coin (sampleDesign draw).Ω fun ω =>
          rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
            (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)) σ = 0 ∨
    ∃ h : Typed zr,
      (regFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (topFinder wr st reg bcs'
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (valueFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (valueSrcAll (fun t => coins (wr t).2) fun t => c3Of ish (innerHeld wr st reg t)) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (leafFinder (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      (saltFinder (holoAt st hp).rsch.sch.level₀ cap salt
        (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h)
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)).Finds Refine.H512 ∨
      RoundFork (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ) cmt ∨
      SessionFork (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw M
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ) ∨
      InnerForkRec (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw
        (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
          (fun t => coins (wr t).2) bind rows zr _hCust h σ)
        (regStmtτ (holoTys st hp) (CT st hp) st hp hp.τ
          (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
          (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap)
        wmsg Refine.H512
        (clashPickHL (holoTys st hp) (CT st hp) execArith Refine.H512 Refine.enc512 Refine.leafPlain
          (holoAtτ st hp hp.τ)) ∨
      prob (fun o => (AcceptsPublic obj prog st ∧ AcceptsC3 ish st (wr o.2.2.1).1 (wr o.2.2.1).2 reg) ∧
            VerifiesVStar ish bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
            ChecksVR ish (innerHeld wr st reg o.2.2.1) (coins (wr o.2.2.1).2) bind rows (zr o.1 o.2.1 o.2.2.1)
              o.2.2.2 ∧
            ¬ HoloSatSome st hp
              (regStmtτ (holoTys st hp) (CT st hp) st hp hp.τ
                (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h) dg N k Rw wmsg
                (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h σ) cap o.1 o.2.1))
          (Game.send Reg fun R => Game.coin (sampleDesign draw).Ω fun ω =>
            rounds Cm ((k : Fin (r st hp)) → CT st hp k) (r st hp + 1) fun t =>
              (outerVR (zr R ω t) dg N).map fun b => (R, ω, t, b)) σ ≤
        expAt (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
            (fun t => coins (wr t).2) bind rows zr _hCust h σ) (fun ω t s =>
            sumV (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
              (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t) dg N
            (fun R₂ j τ =>
              (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t R₂
                j).boundCR dg N (Vs j).samplingDesign.model k Rw M ρ τ) s) +
          ((r st hp + 1 : ℕ) : ℝ≥0∞) *
            expAt (checkedT (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ) (fun ω t s =>
              sumV (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t) dg N
              (fun R₂ j τ =>
                (checkedC3 (I := holoInnerAtτ (holoTys st hp) (CT st hp) st hp hp.τ) ish (innerHeld wr st reg)
                  (fun t => coins (wr t).2) bind rows zr _hCust h σ.1 ω t R₂
                  j).slackCR dg N k Rw M ρ τ) s) +
          ENNReal.ofReal (holoError (holoAt st hp))

/-- (i) states no bits: `KSLin2` at the deployment's dense statements is proved but does not reach the Holo statements (i)
is about; the dense form is `RecursiveAuditC3`. -/
def RecursiveAuditHolo : Prop :=
  RecursiveAuditC3At holoCTs holoMap ∧ RecursiveZkServerPrivate

def RecursiveAuditHoloNoA3 : Prop :=
  RecursiveAuditC3AtNoA3 holoCTs holoMap ∧ RecursiveZkServerPrivate

end Flock.Guarantees
