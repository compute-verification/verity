import Flock.RecVStar
import Flock.Firewall
import Proofs.Flock.Recursive.Holo
import Proofs.Flock.Recursive.Compile
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Soundness.Model.HoloCompiled

/-!
# Recursion: candidate 3's inner protocol as the executable runs it

The objects `Flock.Guarantees.AuditC3` reads, beside `ExecClass.lean`'s for `PublicAudit`:

* `HoloParams`: the sparse phase's public parameters, the verifier's own (the template's `ℓ`, `ν`, the slots' and types'
  bits, the slots' unit slots and types); `HoloClass.ofExec` is `HClass`'s fields with them, and `holoAt st hp` the
  class, regions and schedule of `execClassAt st` with R's registered schedule `fast100`'s at `mReg = ν + pLog + tLog`
  (the default, whose empty levels reject R, where `ℓ` is out of `[6, k_log]`).
* `HoloReg`: the registered word `w` and its level-0 cap `cap`; `Opens` is the registrant's openings of every position
  of `cap` to `w`'s columns, what defines `w` from `cap` (`RecursiveAudit`'s `_hreg` for an encoded table).
* `StackedTop lv reg d cap`: `cap` is R's registered top, the `stacked` entry of the registration record `reg` that the
  inner circuit's META binds by its digest `d` (`RecVStar.registration`): the entry `topEntry` makes from one hm96
  commit string per node, each over the node's row (`Firewall.topRows`, `rowDigest`) under a salt.
* `holoInnerCL`, `holoInnerAt`: `schedInner` at `Model.holoTableCL` with `capReg := x.cap`; the statement is true when
  some table `R` within the unique-decoding radius of `w` (`RegSchedule.Close`) has a satisfiable hidden statement at
  the slots' types.
* `holoError`: the link points, the two reps with their sparse phases (`sparseError`, each coin a degree over `qF` at
  the level-0 list bound), and R's opening at the unique-decoding radius (`regOpenError`: the two claims' ring
  switches, `γ` and the out-of-domain batching coin, level 0's queries at `(1 + ρ)/2 + η`, and Ligerito's levels).
* `ClashHL`, `clashPickHL`: two runs that clash on the reps (`ClashCL`), on R's stream (`ClashCL` at R's schedule), or
  a run opening R's level 0 to a row other than `w`'s (`RegClash`); each a collision of `H` (`ClashHL.collision`).
* `vstarCount ish`: the length of V*'s statement list as the verifier builds it (`RecVStar.statements`), zero where it
  refuses.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Exec StrictCR

/-- **The sparse phase's public parameters**, the verifier's own: the template's `ℓ`, its `2^ν` entries, the `2^sLog`
slots and `2^tLog` registered types, the unit slots each slot fills and each slot's type. -/
structure HoloParams where
  ell : ℕ
  nu : ℕ
  sLog : ℕ
  tLog : ℕ
  units : Fin (2 ^ sLog) → Finset ℕ
  τ : Fin (2 ^ sLog) → Fin (2 ^ tLog)

/-- `ℓ` is a template's size the class admits. -/
abbrev HoloParams.EllOk (hp : HoloParams) (kLog : ℕ) : Prop := 6 ≤ hp.ell ∧ hp.ell ≤ kLog

/-- **The holographic class of an executable class** at the sparse parameters `hp`: `c`'s fields, `ℓ` where it is in
`[6, k_log]` (6 elsewhere, where `holoAt`'s registered schedule rejects). -/
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
  units := hp.units
  ptLocal := c.ptLocal
  mPts := c.mPts

/-- **A holographic class, its regions, the reps' schedule, R's registered schedule and the slots' types.** -/
structure ExecHolo where
  cls : HoloClass
  regs : List (Region cls.kLog (cls.m - cls.kLog))
  sch : Schedule
  rsch : RegSchedule cls
  τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)

/-- **Candidate 3's parameters at an inner statement**: `execClassAt st`'s class at `hp`, its regions and schedule, and
R's registered schedule `fast100`'s at `mReg` (`schedAt`), the default where `ℓ` is out of range. -/
noncomputable def holoAt (st : Flock.Stmt) (hp : HoloParams) : ExecHolo :=
  ⟨HoloClass.ofExec (execClassAt st).cls hp, (execClassAt st).regs, (execClassAt st).sch,
    ⟨if hp.EllOk (execClassAt st).cls.kLog then schedAt (HoloClass.ofExec (execClassAt st).cls hp).shape.mReg
      else default⟩, hp.τ⟩

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- **R's registration**: the registered word and its level-0 cap. -/
structure HoloReg (F D : Type) where
  w : Oracle F
  cap : Cap D

/-- Position `pos`'s column of the registered word, `2^lanes` elements. -/
def HoloReg.column (x : HoloReg F D) (lanes pos : ℕ) : List F := (List.range (2 ^ lanes)).map fun ℓ => x.w ℓ pos

/-- **The registrant's openings**: every position of level `lv` opens against the registered cap to the word's
column, under some salt. -/
def HoloReg.Opens (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (lv : Level)
    (x : HoloReg F D) : Prop :=
  ∀ pos < 2 ^ lv.logLen, ∃ salt sibs,
    Merkle.VerifiesL H E Lf x.cap (pathLen lv) pos (x.column lv.lanes pos) salt sibs

/-- **`cap` is R's registered top**: the registration record `reg`, whose digest the inner circuit's META binds (`d`),
names `stacked` under its `program`, and that entry is what `topEntry` makes from `2^capDepth lv` commit strings,
node `i`'s the hm96 commit string of its row (`Firewall.topRows`) under a salt. -/
def StackedTop (lv : Level) (reg : Lean.Json) (d : String) (cap : Cap Refine.Dig) : Prop :=
  ∃ (prog : String) (e : Lean.Json) (bcs : Array ByteArray) (salt : ℕ → ByteArray),
    Flock.RecVStar.registration (some reg) (some d) = .ok (some (prog, some e)) ∧
      Flock.RecVStar.topEntry prog bcs = .ok e ∧ bcs.size = 2 ^ capDepth lv ∧
      ∀ i (h : i < bcs.size), bcs[i] = Flock.Hm96.Default512.commitString
        (Flock.Firewall.rowDigest ((Flock.Firewall.topRows (cap i).1).headD #[])) (salt i)

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- **Candidate 3's compiled inner protocol**: `schedInner` at `Model.holoTableCL` on the class's public statement,
with R's registered cap the statement's. A statement is a registration whose registrant opens every level-0 position;
it holds when some table within the unique-decoding radius of the registered word has a satisfiable hidden statement at
the slots' types. -/
noncomputable def holoInnerCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) : Inner :=
  schedInner T CT {x : HoloReg F D // x.Opens H E Lf e.rsch.sch.level₀}
    (fun x => holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
      e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.1.cap)
    (fun _ out => out.accepted = true)
    (fun x => ∃ R : RegTable e.cls, e.rsch.Close A x.1.w R ∧ ∃ z, (R.toStatement e.τ e.regs).Satisfies z)

/-- **Candidate 3's inner protocol as the executable runs it**: `holoInnerCL` at `execArith` and SHA-512 with plain
leaves, at `holoAt st hp`. -/
noncomputable abbrev holoInnerAt (st : Flock.Stmt) (hp : HoloParams) : Inner :=
  holoInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)

/-! ### The error -/

/-- **One rep's sparse phase**: H1's 6 coins and H2's 6 rounds of degree 1, H3's `ν` rounds of degree `2ℓ` and `sLog`
of degree `2ℓ + 1`, H4's `pLog` coins and the type point's `tLog`, each over `qF`, at the level-0 list bound. -/
noncomputable def sparseError (s : Schedule) (sh : SparseShape) : ℝ :=
  ((s.listBound₀ * (12 + 2 * sh.ell * sh.nu + (2 * sh.ell + 1) * sh.sLog + sh.pLog + sh.tLog) : ℕ) : ℝ) / qF

/-- Level `l`'s queries at the unique-decoding radius `(1 − ρ)/2 − η`. -/
noncomputable def udQueryError (l : Level) : ℝ := ((1 + l.rate) / 2 + eta) ^ l.queries

/-- **R's opening**, at `STACKED_CLAIMS` claims and no list: each claim's ring switch (degree 7), `γ` and the
out-of-domain claim's batching coin, level 0's queries at the unique-decoding radius, and Ligerito's levels. -/
noncomputable def regOpenError (s : Schedule) : ℝ :=
  ((7 * Flock.RecVStar.STACKED_CLAIMS + 2 : ℕ) : ℝ) / qF + udQueryError s.level₀ + ligeritoError s.levels

/-- **Candidate 3's table error**: the link points, the two reps each with its sparse phase (independent coins, so
squared), and R's opening. -/
noncomputable def holoError (e : ExecHolo) : ℝ :=
  linkError e.sch ⟨e.cls.kLog, e.regs.length, e.cls.mPts⟩ +
    (repError e.sch ⟨e.cls.kLog, e.regs.length, e.cls.mPts⟩ + sparseError e.sch e.cls.shape) ^ 2 +
    regOpenError e.rsch.sch

/-! ### Clashes -/

/-- The reps' part of a run's output, as the compiled table's. -/
def _root_.FlockSoundness.Model.OutHL.circ (o : OutHL F D Sl) : OutCL F D Sl :=
  ⟨o.accepted, o.cap₀, o.pend, o.opens, o.salts⟩

/-- R's part of a run's output, as a compiled table's whose rep 0 is R's stream, at the registered cap `cap`. -/
def _root_.FlockSoundness.Model.OutHL.reg (o : OutHL F D Sl) (cap : Cap D) : OutCL F D Sl :=
  ⟨o.accepted, cap, (o.pendR, PendL.reject), fun _ => o.opens 2, fun _ => o.salts 2⟩

/-- **A run opens R's level 0 to a row other than the registered word's**: one of R's level-0 openings verifies against
the registered cap with a row that is not `w`'s column there. -/
def RegClash (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (lv : Level)
    (x : HoloReg F D) (o : OutHL F D Sl) : Prop :=
  ∃ y ∈ slotOpensL (o.reg x.cap) 0 0 lv.queries, y.1 < 2 ^ lv.logLen ∧
    VerifiedL H E Lf x.cap lv.lanes (pathLen lv) y ∧ y.2.1 ≠ x.column lv.lanes y.1

omit [Fintype F] [DecidableEq F] in
theorem RegClash.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {lv : Level} {x : HoloReg F D} {o : OutHL F D Sl} (hx : x.Opens H E Lf lv) (h : RegClash H E Lf lv x o) :
    Merkle.Collision H := by
  obtain ⟨y, -, hlt, hv, hne⟩ := h
  obtain ⟨s, sibs, hs⟩ := hx y.1 hlt
  exact Merkle.opening_bindingL H E Lf x.cap (pathLen lv) y.1 _ _ _ _ _ _ hne hv.2 hs

/-- **Two runs of candidate 3's table clash**: on the reps (`ClashCL` at the reps' schedule), on R's stream (`ClashCL`
at R's), or either opens R's level 0 off the registered word (`RegClash`). -/
def ClashHL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch rsch : Schedule)
    (x : HoloReg F D) (o o' : OutHL F D Sl) : Prop :=
  ClashCL H E Lf sch o.circ o'.circ ∨ ClashCL H E Lf rsch (o.reg x.cap) (o'.reg x.cap) ∨
    RegClash H E Lf rsch.level₀ x o ∨ RegClash H E Lf rsch.level₀ x o'

omit [Fintype F] [DecidableEq F] in
/-- **A clash is a collision**, given the registrant's openings. -/
theorem ClashHL.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch rsch : Schedule} {x : HoloReg F D} {o o' : OutHL F D Sl} (hx : x.Opens H E Lf rsch.level₀)
    (h : ClashHL H E Lf sch rsch x o o') : Merkle.Collision H := by
  rcases h with h | h | h | h
  · exact h.collision
  · exact h.collision
  · exact h.collision hx
  · exact h.collision hx

open Classical in
/-- **Candidate 3's table's outcome on an inner transcript**: the outcome the transcript's steps play `holoTableCL` to
at the statement `x`, when they play it to one. -/
noncomputable def outHL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (t : List (List (SMsg T) × ((k : Fin r) → CT k))) : Option (OutHL F D Sl) :=
  if h : ∃ out, PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts
      e.cls.shape e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t) out
  then some h.choose else none

/-- **The inner fork's output at candidate 3's table**: on two inner transcripts whose steps play it to outcomes that
clash (`ClashHL`), a collision of `H`, else nothing. -/
noncomputable def clashPickHL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : {x : HoloReg F D // x.Opens H E Lf e.rsch.sch.level₀})
    (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  collOf H (∃ o o', outHL T CT A H E Lf e x.1 t = some o ∧ outHL T CT A H E Lf e x.1 t' = some o' ∧
      ClashHL H E Lf e.sch e.rsch.sch x.1 o o')
    fun ⟨_, _, _, _, hc⟩ => hc.collision x.2

/-! ### V*'s statement list -/

/-- **The number of V*'s statements** at the inner shape `ish`, as the verifier builds the list
(`RecVStar.statements`); zero where it refuses. -/
def vstarCount (ish : Flock.RecVStar.Inner) : ℕ :=
  match Flock.RecVStar.statements ish with
  | .ok ss => ss.size
  | .error _ => 0

end FlockSoundness.Discharge.Recursion
