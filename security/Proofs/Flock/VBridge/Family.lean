module

public import Flock.RecVStar
public import Proofs.Flock.VBridge.AlgHolds
public import Proofs.Flock.VBridge.Stages
public import Proofs.Flock.VBridge.Top
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
import Proofs.Flock.VBridge.Algebra.BitsOk
import Proofs.Flock.VBridge.Full

/-! V*'s statement family (`VStarFamily`), tying the statements `vbridge_of_vstar` takes as free objects to V*'s executable Lean, and `vbridge_of_reps`, the same composition with a unit map per rep. -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecResiduals (portSizes groups)
open Flock.RecAlgebra (structureOf verifierRows el)
open FlockVBridge.OfVStar (ResZeroAt)
open FlockVBridge.AccChain (Openings OpensAt CoefsAre ConsAre ReadsEnds levelLanes)

def circuitFile (s : VStar.Stmt) (b : VStar.Bindings) : Except String ByteArray := do
  let u ← s.unit
  let l ← u.laid
  let e ← VStar.emitCircuit (m := StateM String) (fun x => modify (· ++ x)) u l s.sub b
  pure (Id.run (e.run "")).2.toUTF8

def cp (_ : ℕ) (_ : Fin 2) : ℕ := 0

def cpO (_ : ℕ) (_ : Fin 2) (_ : ℕ) : ℕ := 0

section reps

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {Cm : Type} {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Flock.Schedule)

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

/-- One unit and copy per part for both reps would force the reps' values equal, since V* computes the two reps in two
instances and a registered bit has one value. -/
theorem vbridge_of_reps
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (x : Reg → L.Ω → Hidden cls)
    (wmsg : ((j : Fin S) → (Vs j).Bits) → List (ℐ).Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (ℐ).Msg)
    (sh : Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    {GS : Flock.RecAlgebra.GfS} (hS : structureOf sh = .ok GS)
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hg : groups GS (portSizes sh) = .ok gs)
    (sess : ℕ → Fin S) (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) (cp : ℕ → Fin 2 → ℕ)
    (hvr : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∃ v info kappa,
      verifierRows sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb =
          .ok (v, info, kappa) ∧ v.size = GS.nv)
    (hsz : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw ∧
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb.size = 64)
    (hres : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
      ResZeroAt (zs R ω t R₂ (sess p)) (layers zs dg N k Rw R ω t R₂ τ (sess p)) (un p r) (cp p r) sh gs p
        (el (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).w) (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rounds
        (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).extras (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r).comb)
    (acc : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → ℕ → Array F128)
    (ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Openings)
    (sessO : ℕ → Fin S) (unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (sessO ℓ)).nv) (cpO : ℕ → Fin 2 → ℕ → ℕ)
    (hu : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < sh.sched.levels.size,
      ∀ j < (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).q ℓ,
        OpensAt (zs R ω t R₂ (sessO ℓ)) (layers zs dg N k Rw R ω t R₂ τ (sessO ℓ)) (unO ℓ r j) (cpO ℓ r j)
          (levelLanes sh ℓ) ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).rows ℓ)[j]!
          #[(acc R ω t (layers zs dg N k Rw R ω t R₂ τ)
              ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[0]!,
            (acc R ω t (layers zs dg N k Rw R ω t R₂ τ)
              ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j))[1]!]
          ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).coef ℓ j)
          (acc R ω t (layers zs dg N k Rw R ω t R₂ τ)
            ((ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r).start ℓ + j + 1)))
    (hcoef : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      CoefsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (hcons : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ConsAre sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (hrd : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ r,
      ReadsEnds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) (acc R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (ops R ω t (layers zs dg N k Rw R ω t R₂ τ) r))
    (files : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → TableFiles cls)
    (hsch : ∀ R ω t l, (files R ω t l).sch = sch)
    (hfiles : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (files R ω t (layers zs dg N k Rw R ω t R₂ τ)).Valid regs (x R ω) T CT
        ((wmsg (layers zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)))
    (hruns : ∀ R ω t R₂ τ o, Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      (∀ r, AlgebraHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r) ∧
        AccHolds sh (rep R ω t (layers zs dg N k Rw R ω t R₂ τ) r)) →
      (files R ω t (layers zs dg N k Rw R ω t R₂ τ)).Runs)
    (hcm : VCommits zs Sk cmt) (hdec : VDecodesR wmsg (fun _ _ a b => a = b) Sk msg) (hfull : VFullR (ℐ).r Sk) :
    Flock.Assumptions.VBridge (I := ℐ) zs dg N k Rw x wmsg Sk cmt msg :=
  ⟨fun R ω t R₂ hc τ o hout h₀ =>
    have h : Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o := ⟨h₀, hc, hout⟩
    (files R ω t (layers zs dg N k Rw R ω t R₂ τ)).accepts regs (x R ω) T CT sch (hsch R ω t _) _
      (hfiles R ω t R₂ τ o h) (hruns R ω t R₂ τ o h fun r => by
        obtain ⟨v, info, kappa, hv, hvs⟩ := hvr R ω t R₂ τ o h r
        obtain ⟨hw, hc⟩ := hsz R ω t R₂ τ o h r
        exact ⟨FlockVBridge.OfVStar.algebraHolds_of_decode (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
            (fun s => (h.1 s).2) hS (Flock.RecAlgebra.structureOf_bitsOk hS) hv hw hc hvs hg sess (fun p => un p r)
            (fun p => cp p r) (hres R ω t R₂ τ o h r),
          FlockVBridge.AccChain.accHolds_of_units
            (FlockVBridge.AccChain.unitSums_of_decode (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ)
              (fun s => (h.1 s).2) sh _ _ sessO (fun ℓ j => unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
            (hcoef R ω t R₂ τ o h r) (hcons R ω t R₂ τ o h r) (hrd R ω t R₂ τ o h r)⟩),
    hcm, hdec, hfull⟩

end reps

section family

variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (sch : Flock.Schedule)
variable {n : ℕ} (L : SamplingDesign n) (Reg Cm Reg' Reg₂ : Type)

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

structure VStarFamily where
  /-- The inner session's shape as V* takes it: `m`, `k_log` and claims (`rec_vstage.shape`). -/
  vsh : VStar.Algebra.Shape
  /-- The verifier's shape. -/
  sh : Shape
  hsh : RecResiduals.shape? vsh.m vsh.kLog vsh.claims = some sh
  stmts : Array VStar.Stmt
  hstmts : VStar.statements vsh = .ok stmts
  /-- Each statement's bindings: its partition's digest, its Program's SHA-512 and `program_digests`. -/
  bind : Fin stmts.size → VStar.Bindings
  Vs : Fin stmts.size → VStmt
  zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin stmts.size) → (Vs j).Outer Reg'
  hfile : ∀ R ω t R₂ j, circuitFile stmts[j] (bind j) = .ok (zs R ω t R₂ j).I.circuitFile
  /-- Stated as the check's verdict: no proof here evaluates `Flock.JsonParse.parse`, which `HmRow.parse` reads through. -/
  hparsed : ∀ j, RecVStar.checkParsed stmts[j] (Vs j).c = .ok ()
  hpart : ∀ j, RecVStar.checkPart stmts[j] (Vs j).c = .ok ()
  hnv : ∀ j, (Vs j).nv = stmts[j].instances
  /-- Each statement's circuit is over plain parts: `hfile`'s bytes carry no `holo`, but no META parse is stated here. -/
  hplain : ∀ j, RecResiduals.holoOf (Vs j).c = .ok false
  /-- The statement of part `p`. -/
  sess : ℕ → Fin stmts.size
  hsess : ∀ p (j : Fin stmts.size), (∃ nm pt, stmts[j] = .inner p nm pt) → sess p = j
  /-- Rep `r`'s instance of part `p`. -/
  un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv
  hun : ∀ p r, r.val < (Vs (sess p)).nv → (un p r).val = r.val
  /-- The statement of level `ℓ`'s openings. -/
  sessO : ℕ → Fin stmts.size
  hsessO : ∀ ℓ (j : Fin stmts.size), (∃ LANES H m, stmts[j] = .recOpen ℓ LANES H m) → sessO ℓ = j
  /-- Rep `r`'s opening `j` of level `ℓ`. -/
  unO : (ℓ : ℕ) → Fin 2 → ℕ → Fin (Vs (sessO ℓ)).nv
  hunO : ∀ ℓ r j, r.val * (sh.sched.levels[ℓ]!).q + j < (Vs (sessO ℓ)).nv →
    (unO ℓ r j).val = r.val * (sh.sched.levels[ℓ]!).q + j
  x : Reg → L.Ω → Hidden cls
  /-- Each rep's message elements in V*'s layers, its coins, extra claims and comb the verifier's at the history (its
  consistency sums are `rep`'s). -/
  rep₀ : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin stmts.size) → (Vs j).Bits) → Fin 2 → RepData
  /-- Each rep's opened rows of each level. -/
  rows : ((j : Fin stmts.size) → (Vs j).Bits) → Fin 2 → ℕ → Array (Array F128)
  /-- Each rep's `α` of each level at the history, whose `eq` weighs its openings. -/
  alpha : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin stmts.size) → (Vs j).Bits) → Fin 2 → ℕ → Array F128
  /-- The rows of `rec-acc`, the running sums. -/
  acc : ((j : Fin stmts.size) → (Vs j).Bits) → ℕ → Array F128
  /-- The inner transcript's messages V*'s layers hold. -/
  wmsg : ((j : Fin stmts.size) → (Vs j).Bits) → List (ℐ).Msg
  /-- The inner session's table files at the history (its streams' coins are the verifier's). -/
  files : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin stmts.size) → (Vs j).Bits) → TableFiles cls
  /-- Each stage's commit strings in each session. -/
  Sk : ℕ → (j : Fin stmts.size) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv)
  /-- The digests a commitment registers at each commit string. -/
  cmt : (j : Fin stmts.size) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg
  /-- The message each stage's commit strings decode to. -/
  msg : ℕ → ((j : Fin stmts.size) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → (ℐ).Msg

variable {T CT cls regs sch L Reg Cm Reg' Reg₂}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

def q (ℓ : ℕ) : ℕ := (F.sh.sched.levels[ℓ]!).q

def nOpen : ℕ := ((List.range F.sh.sched.levels.size).map F.q).sum

def coef (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2)
    (ℓ j : ℕ) : Array F128 :=
  match verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb with
  | .ok (_, _, kappa) => coefAt (kappa[ℓ]!) (levelLanes F.sh ℓ) (eqTable (F.alpha R ω t l r ℓ))[j]!
  | .error _ => #[]

def ops (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    Openings :=
  ⟨r.val * F.nOpen, F.q, F.rows l r, F.alpha R ω t l r, F.coef R ω t l r⟩

def rep (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) (r : Fin 2) :
    RepData :=
  { F.rep₀ R ω t l r with
    cons := fun ℓ i => Flock.RecAlgebra.consAt (F.rows l r ℓ) (F.q ℓ) (F.alpha R ω t l r ℓ) i }

theorem coefsAre (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits)
    (r : Fin 2) : CoefsAre F.sh (F.rep R ω t l r) (F.ops R ω t l r) := by
  intro v info kappa h ℓ _ j _
  have h' : verifierRows F.sh (F.rep₀ R ω t l r).rounds (F.rep₀ R ω t l r).extras (F.rep₀ R ω t l r).comb =
      .ok (v, info, kappa) := h
  show F.coef R ω t l r ℓ j = coefAt (kappa[ℓ]!) (levelLanes F.sh ℓ) (eqTable (F.alpha R ω t l r ℓ))[j]!
  unfold coef
  rw [h']

theorem consAre (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits)
    (r : Fin 2) : ConsAre F.sh (F.rep R ω t l r) (F.ops R ω t l r) :=
  fun _ _ _ _ => rfl

theorem parse_circuitFile (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (j : Fin F.stmts.size)
    {file : ByteArray} (h : circuitFile F.stmts[j] (F.bind j) = .ok file) :
    HmRow.parse (ZkExec.zkTags (F.zs R ω t R₂ j).I) file (F.zs R ω t R₂ j).I.tables = .ok (F.Vs j).c := by
  have e := F.hfile R ω t R₂ j
  rw [h] at e
  injection e with e
  rw [e]
  exact (F.zs R ω t R₂ j).hc

end VStarFamily

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

def HStruct (GS : Flock.RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) : Prop :=
  structureOf F.sh = .ok GS ∧ groups GS (portSizes F.sh) = .ok gs

theorem hStruct_of_check {c : Flock.Circuit} (hc : RecResiduals.check c = .ok ()) {m k cl p : ℕ}
    (hp : RecResiduals.params c.unit.name = some (m, k, cl, p)) (hsh : RecResiduals.shape? m k cl = some F.sh)
    (hh : RecResiduals.holoOf c = .ok false) : ∃ GS gs, F.HStruct GS gs := by
  obtain ⟨S, ns, nv, hpart, -⟩ := FlockVBridge.OfVStar.check_part hc hp hsh hh
  unfold RecResiduals.part at hpart
  obtain ⟨GS, hGS, hpart⟩ := FlockVBridge.PartS.bind_ok hpart
  dsimp only at hpart
  split at hpart
  · cases hpart
  · obtain ⟨gs, hgs, -⟩ := FlockVBridge.PartS.bind_ok hpart
    exact ⟨GS, gs, hGS, hgs⟩

def HShape (GS : Flock.RecAlgebra.GfS) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r,
    (∃ v info kappa, verifierRows F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
        (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb = .ok (v, info, kappa) ∧ v.size = GS.nv) ∧
    (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w.size = GS.nw ∧
      (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb.size = 64

def HReads : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r,
    ReadsEnds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r) (F.acc (layers F.zs dg N k Rw R ω t R₂ τ))
      (F.ops R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r)

def HSch : Prop := ∀ R ω t l, (F.files R ω t l).sch = sch

def HFiles : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Valid regs (F.x R ω) T CT
      ((F.wmsg (layers F.zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))

def HRuns : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    (∀ r, AlgebraHolds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r) ∧
      AccHolds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r)) →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Runs

def HCommits : Prop := VCommits F.zs F.Sk F.cmt

def HDecodes : Prop := VDecodesR F.wmsg (fun _ _ a b => a = b) F.Sk F.msg

def HFull : Prop :=
  ∀ i < (ℐ).r, ∀ j g, posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g ∈ F.Sk i j →
    ∃ u, g ∈ ((Placed.keyProg (F.Vs j).c (F.Vs j).hU (F.Vs j).nv).partition
      (Layout.outsOf (F.Vs j).c (F.Vs j).hU (F.Vs j).nv)).io u

theorem vFull_of_hFull (h : F.HFull) : VFull (ℐ).r F.Sk := FlockVBridge.Full.vFull_of_whole _ F.Sk h

end VStarFamily

end family

end FlockVBridge.Family
