module

public import Flock.RecCoins
public import Definitions.Flock.Recursive.Holo
public import Proofs.Flock.Recursive.ExecClass
public import Proofs.Flock.Recursive.Replay
public import Definitions.Flock.HoloCompiled
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
import Proofs.Flock.Soundness.Merkle

/-! Candidate 3's inner protocol as the executable runs it: its parameters, R's registration, the verifier's checks at candidate 3, its error and its clashes. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Exec StrictCR

structure HoloParams where
  ell : ℕ
  nu : ℕ
  sLog : ℕ
  tLog : ℕ
  units : Fin (2 ^ sLog) → Finset ℕ
  τ : Fin (2 ^ sLog) → Fin (2 ^ tLog)

abbrev HoloParams.EllOk (hp : HoloParams) (kLog : ℕ) : Prop := 6 ≤ hp.ell ∧ hp.ell ≤ kLog

/-- `ℓ` is 6 outside `[6, k_log]`, where `holoAt`'s registered schedule rejects; unit slots are below `2^(k_log − ℓ)`
because the sparse phase's `U` reads a unit's low `k_log − ℓ` bits (`holoPub`). -/
def HoloClass.ofExec (c : HClass) (hp : HoloParams) : HoloClass where
  m := c.m
  kLog := c.kLog
  hk := c.hk
  hk6 := c.hk6
  hm := c.hm
  pin := c.pin
  A₀pub := c.A₀pub
  B₀pub := c.B₀pub
  slotLog := if hp.EllOk c.kLog then hp.ell else 6
  hs6 := by split_ifs with h; exacts [h.1, le_rfl]
  hsk := by split_ifs with h; exacts [h.2, c.hk6]
  nu := hp.nu
  sLog := hp.sLog
  tLog := hp.tLog
  units := fun σ => (hp.units σ).filter fun u => u < 2 ^ (c.kLog - (if hp.EllOk c.kLog then hp.ell else 6))
  ptLocal := c.ptLocal
  mPts := c.mPts

structure ExecHolo where
  cls : HoloClass
  regs : List (Region cls.kLog (cls.m - cls.kLog))
  sch : Schedule
  rsch : RegSchedule cls
  τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)

noncomputable def holoAt (st : Flock.Stmt) (hp : HoloParams) : ExecHolo :=
  ⟨HoloClass.ofExec (execClassAt st).cls hp, (execClassAt st).regs, (execClassAt st).sch,
    ⟨if hp.EllOk (execClassAt st).cls.kLog then schedAt (HoloClass.ofExec (execClassAt st).cls hp).shape.mReg
      else default⟩, hp.τ⟩

/-- Every field but the typing is `holoAt st hp`'s own term, so a defeq that crosses the typing short-circuits on shared
subterms; a modified `HoloParams` instead makes the unifier unfold `holoAt`'s whole body, which times out. -/
noncomputable def holoAtτ (st : Flock.Stmt) (hp : HoloParams)
    (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) : ExecHolo :=
  { holoAt st hp with τ := τv }

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

structure HoloReg (F D : Type) where
  w : Oracle F
  cap : Cap D

def HoloReg.column (x : HoloReg F D) (lanes pos : ℕ) : List F := (List.range (2 ^ lanes)).map fun ℓ => x.w ℓ pos

def HoloReg.Opens (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (lv : Level)
    (x : HoloReg F D) : Prop :=
  ∀ pos < 2 ^ lv.logLen, ∃ salt sibs,
    Merkle.VerifiesL H E Lf x.cap (pathLen lv) pos (x.column lv.lanes pos) salt sibs

def metaRegistration (st : Flock.Stmt) : Option String :=
  match st.c.metaJ.getObjVal? "registration" with
  | .ok (.str d) => some d
  | _ => none

/-- `salt` is the salts the prover's workers hold (`registrant-holds-openings`): a guarantee that reads `TopOf` is applied
at the held salts. -/
def TopOf (lv : Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray) (bcs : Array ByteArray) : Prop :=
  bcs.size = 2 ^ capDepth lv ∧ ∀ i (h : i < bcs.size),
    bcs[i] = Flock.Hm96.Default512.commitString (Flock.ZkServer.rowDigest ((Flock.ZkServer.topRows (cap i).1).headD #[]))
      (salt i)

/-- `registrant-holds-openings`: the registrant holds the commit strings `bcs` it registered, and the guarantee is applied
at the held salts. The first two conjuncts are the verifier's checks only at a transcript's own top; at the held `bcs`
they are part of the premise, which `TopColl.cand` reads. -/
def StackedTopAt (lv : Level) (st : Flock.Stmt) (reg : Lean.Json) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (bcs : Array ByteArray) : Prop :=
  ∃ (prog : String) (e : Lean.Json),
    Flock.RecVStar.registration (some reg) (metaRegistration st) = .ok (some (prog, some e)) ∧
      Flock.RecVStar.topEntryFields prog bcs e = .ok true ∧ TopOf lv cap salt bcs

/-- It refuses a registration META does not bind, and a binding round of R's stream whose top does not make the
registered `stacked` entry. -/
def innerCommitsAt (tr : Lean.Json) (record : String) (st : Flock.Stmt) (reg : Lean.Json) :
    Except String Flock.RecVStar.InnerCommits := do
  let shape ← Flock.RecVStar.stackedShapeOf st.c.metaJ
  Flock.RecVStar.innerCommits tr record (some reg) (metaRegistration st) shape

def unitRangeOf (st : Flock.Stmt) : Except String Flock.Range :=
  Flock.RecCoins.unitRangeOf st

def SparseRangeOk (ish : Flock.RecVStar.Inner) (st : Flock.Stmt) : Prop :=
  ∀ sp, ish.sparse = some sp →
    let xs := Array.replicate ((execClassAt st).cls.kLog - 6) (0 : Flock.F128)
    sp.valid = true ∧ ¬ xs.size < sp.ell - Flock.RecSparse.PRE_ROUNDS ∧
      ∃ rng, unitRangeOf st = .ok rng ∧ (Flock.RecCoins.layoutFactors sp xs xs rng).isOk = true

/-- It is not the whole compiled acceptance: `AcceptsPublic` and `ChecksVR` carry the verifier's other checks. -/
def AcceptsC3 (ish : Flock.RecVStar.Inner) (st : Flock.Stmt) (tr : Lean.Json) (record : String) (reg : Lean.Json) :
    Prop :=
  (Flock.RecVStar.statements ish).isOk = true ∧ (innerCommitsAt tr record st reg).isOk = true ∧
    (∀ cs, innerCommitsAt tr record st reg = .ok cs → (Flock.RecVStar.checkCovers ish cs).isOk = true) ∧
    SparseRangeOk ish st ∧ (Flock.RecVStar.checkInnerStmt ish st).isOk = true

def UnitsOf (st : Flock.Stmt) (hp : HoloParams) : Prop :=
  ∃ rng, unitRangeOf st = .ok rng ∧ ∀ σ, hp.units σ = if σ.val < rng.count then {rng.pos0 + σ.val} else ∅

/-- `checkInnerStmt` refuses `T_LOG ≠ 0`, so the one possible slot type is zero; no clear private `τ` row is an input to
the verifier. -/
def holoParamsAt (ish : Flock.RecVStar.Inner) (st : Flock.Stmt) : HoloParams :=
  let sp := ish.sparse.getD { nu := 0, ell := 0, sLog := 0, tLog := 0 }
  { ell := sp.ell
    nu := sp.nu
    sLog := sp.sLog
    tLog := sp.tLog
    units := fun σ =>
      match (unitRangeOf st).toOption with
      | some rng => if σ.val < rng.count then {rng.pos0 + σ.val} else ∅
      | none => ∅
    τ := fun _ => ⟨0, Nat.two_pow_pos _⟩ }

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- The table reads only the cap: the word is what the statement is about. -/
noncomputable def holoInnerCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) : Inner :=
  schedInner T CT (HoloReg F D)
    (fun x => holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
      e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap)
    (fun _ out => out.accepted = true)
    (fun x => ∃ R : RegTable e.cls, e.rsch.Close A x.w R ∧ ∃ z, (R.toStatement e.τ e.regs).Satisfies z)

noncomputable abbrev holoInnerAt (st : Flock.Stmt) (hp : HoloParams) : Inner :=
  holoInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)

/-- Its `Stmt`, `Msg`, `Coin` and `r` are `schedInner`'s parameters, independent of the typing, so they are
`holoInnerAt st hp`'s own. -/
noncomputable abbrev holoInnerAtτ (st : Flock.Stmt) (hp : HoloParams)
    (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) : Inner :=
  holoInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAtτ st hp τv)

/-- No list: the phase's claim is on R's table, which R's opening decodes uniquely (`regOpenError`). -/
noncomputable def sparseError (sh : SparseShape) : ℝ :=
  ((18 + (2 * sh.ell + 1) * sh.nu + (2 * sh.ell + 2) * sh.sLog + sh.pLog + sh.tLog : ℕ) : ℝ) / qF

noncomputable def udQueryError (l : Level) : ℝ := ((1 + l.rate) / 2 + eta) ^ l.queries

noncomputable def regOpenError (s : Schedule) : ℝ :=
  ((7 * Flock.RecVStar.STACKED_CLAIMS + 2 : ℕ) : ℝ) / qF + udQueryError s.level₀ + ligeritoError s.levels

/-- The two reps' sparse phases have independent coins, so their term is squared. -/
noncomputable def holoError (e : ExecHolo) : ℝ :=
  linkError e.sch ⟨e.cls.kLog, e.regs.length, e.cls.mPts⟩ +
    (repError e.sch ⟨e.cls.kLog, e.regs.length, e.cls.mPts⟩ + sparseError e.cls.shape) ^ 2 +
    regOpenError e.rsch.sch

theorem holoError_holoAtτ (st : Flock.Stmt) (hp : HoloParams)
    (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) : holoError (holoAtτ st hp τv) = holoError (holoAt st hp) :=
  rfl

def _root_.FlockSoundness.Model.OutHL.circ (o : OutHL F D Sl) : OutCL F D Sl :=
  ⟨o.accepted, o.cap₀, o.pend, o.opens, o.salts⟩

def _root_.FlockSoundness.Model.OutHL.reg (o : OutHL F D Sl) (cap : Cap D) : OutCL F D Sl :=
  ⟨o.accepted, cap, (o.pendR, PendL.reject), fun _ => o.opens 2, fun _ => o.salts 2⟩

def RegClash (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (lv : Level)
    (x : HoloReg F D) (o : OutHL F D Sl) : Prop :=
  ∃ y ∈ slotOpensL (o.reg x.cap) 0 0 lv.queries, y.1 < 2 ^ lv.logLen ∧
    VerifiedL H E Lf x.cap lv.lanes (pathLen lv) y ∧ y.2.1 ≠ x.column lv.lanes y.1

def ClashHL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch rsch : Schedule)
    (x : HoloReg F D) (o o' : OutHL F D Sl) : Prop :=
  ClashCL H E Lf sch o.circ o'.circ ∨ ClashCL H E Lf rsch (o.reg x.cap) (o'.reg x.cap) ∨
    RegClash H E Lf rsch.level₀ x o ∨ RegClash H E Lf rsch.level₀ x o'

/-- Unlike `ClashHL`, no registered word is read, so no registrant's openings are needed for a collision. -/
def ClashRunHL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (cap : Cap D) (o o' : OutHL F D Sl) : Prop :=
  ClashCL H E Lf sch o.circ o'.circ ∨ ClashCL H E Lf rsch (o.reg cap) (o'.reg cap)

noncomputable def clashCandsHL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (cap : Cap D) (o o' : OutHL F D Sl) : List (Prop × (List UInt8 × List UInt8)) :=
  clashCandsCL H E Lf sch o.circ o'.circ ++ clashCandsCL H E Lf rsch (o.reg cap) (o'.reg cap)

omit [Fintype F] [DecidableEq F] in
theorem clashCandsHL_spec {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch rsch : Schedule} {cap : Cap D} {o o' : OutHL F D Sl} {c : Prop × (List UInt8 × List UInt8)}
    (hc : c ∈ clashCandsHL H E Lf sch rsch cap o o') (h : c.1) : Hash.Collision H c.2.1 c.2.2 := by
  rcases List.mem_append.1 hc with hc | hc
  · exact clashCandsCL_spec hc h
  · exact clashCandsCL_spec hc h

omit [Fintype F] [DecidableEq F] in
theorem ClashRunHL.cand {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch rsch : Schedule} {cap : Cap D} {o o' : OutHL F D Sl} (h : ClashRunHL H E Lf sch rsch cap o o') :
    ∃ c ∈ clashCandsHL H E Lf sch rsch cap o o', c.1 := by
  rcases h with h | h
  · obtain ⟨c, hc, hp⟩ := h.cand
    exact ⟨c, List.mem_append_left _ hc, hp⟩
  · obtain ⟨c, hc, hp⟩ := h.cand
    exact ⟨c, List.mem_append_right _ hc, hp⟩

/-- It is the replay of the steps (`Game.playT`): none is chosen. Two tests under it are decided classically: a step's
type against the node's, and the table's opening check (`OpensOKL`). -/
noncomputable def outHL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (t : List (List (SMsg T) × ((k : Fin r) → CT k))) : Option (OutHL F D Sl) :=
  playT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts
    e.cls.shape e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t)

noncomputable def clashPickHL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  (outHL T CT A H E Lf e x t).bind fun o => (outHL T CT A H E Lf e x t').bind fun o' =>
    collOf (clashCandsHL H E Lf e.sch e.rsch.sch x.cap o o') (fun c => c.1) fun c => c.2

omit [Fintype K] [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)] in
theorem clashPickHL_isColl {A : Arith F K} {H : List UInt8 → D} {E : Merkle.Enc (List F) D}
    {Lf : Merkle.Leaf (List F) Sl D H} {e : ExecHolo} {x : HoloReg F D}
    {t t' : List (List (SMsg T) × ((k : Fin r) → CT k))} {o o' : OutHL F D Sl}
    (ho : outHL T CT A H E Lf e x t = some o) (ho' : outHL T CT A H E Lf e x t' = some o')
    (hc : ClashRunHL H E Lf e.sch e.rsch.sch x.cap o o') : IsColl H (clashPickHL T CT A H E Lf e x t t') := by
  unfold clashPickHL
  rw [ho, ho', Option.bind_some, Option.bind_some]
  exact collOf_isColl H (fun _ hc hp => clashCandsHL_spec hc hp) hc.cand

def vstarCount (ish : Flock.RecVStar.Inner) : ℕ :=
  match Flock.RecVStar.statements ish with
  | .ok ss => ss.size
  | .error _ => 0

end FlockSoundness.Discharge.Recursion
