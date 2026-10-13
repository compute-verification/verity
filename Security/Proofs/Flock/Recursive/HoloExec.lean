import Flock.RecVStar
import Flock.RecCoins
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
  class, regions and schedule of `execClassAt st` with R's registered schedule `fast100`'s at `mReg`, `ν + pLog + tLog`
  padded to `PAD_TO` as the verifier opens R (the default, whose empty levels reject R, where `ℓ` is out of
  `[6, k_log]`, which the verifier refuses: `SparseRangeOk`).
* `HoloReg`: a word `w` and a level-0 cap `cap`, the inner statement; `Opens` is a registrant's openings of every
  position of `cap` to `w`'s columns (`RecursiveAudit`'s `_hreg` for an encoded table), which `RecursiveAuditC3` does
  not take: its word is the one the prover's own runs open (`Recursive/RegExtract.lean`).
* `StackedTop lv st reg cap`: `cap` is R's registered top, the `stacked` entry of the registration record `reg` (the
  verifier's `--rec-inner-registration`) that the inner statement's META binds by its digest (`metaRegistration st`,
  `RecVStar.registration`): the entry `topEntry` makes from one hm96 commit string per node, each over the node's row
  (`Firewall.topRows`, `rowDigest`) under a salt. The verifier holds it of the binding round's top (`innerCommitsAt`).
* `AcceptsC3`: the verifier's checks at candidate 3 of its own inputs, which the event carries: V*'s list builds, the
  inner transcript holds against R's registration (`innerCommitsAt`), the covers check passes there (all of
  `RecVStar.checkCovers`), and check 5's sparse refusals pass (`SparseRangeOk`: `ℓ` in `[6, k_log]`, the unit range's
  slots inside the block). `UnitsOf`: the slots' units are the unit range's.
* `holoInnerCL`, `holoInnerAt`: `schedInner` at `Model.holoTableCL` with `capReg := x.cap`, at a statement `x` a word
  and a cap; it is true when some table `R` within the unique-decoding radius of `x.w` (`RegSchedule.Close`) has a
  satisfiable hidden statement at the slots' types.
* `holoError`: the link points, the two reps with their sparse phases (`sparseError`, each coin a degree over `qF`),
  and R's opening at the unique-decoding radius (`regOpenError`: the two claims' ring
  switches, `γ` and the out-of-domain batching coin, level 0's queries at `(1 + ρ)/2 + η`, and Ligerito's levels).
* `ClashRunHL`, `clashPickHL`: two runs that clash on the reps (`ClashCL`) or on R's stream (`ClashCL` at R's
  schedule), each a collision of `H` (`ClashRunHL.collision`). `ClashHL` adds a run opening R's level 0 to a row other
  than `w`'s (`RegClash`), a collision only beside a registrant's openings (`ClashHL.collision`).
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
`[6, k_log]` (6 elsewhere, where `holoAt`'s registered schedule rejects), and each slot's unit slots below
`2^(k_log − ℓ)`, the blocks a `2^k_log` table has: the sparse phase's `U` reads a unit's low `k_log − ℓ` bits
(`holoPub`), so a unit past them would count a block the table does not have. -/
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

/-- **The digest the inner statement's META binds**: META's `registration` of `st`'s circuit, the circuit registration's
digest, as `FlockVerify.innerMetaRegistration` reads it off `--rec-inner-circuit`, the file `AcceptsPublic` sets `st` up
from (`RecCoins.Inner.load`); none where META names none, or no digest, which the verifier refuses. -/
def metaRegistration (st : Flock.Stmt) : Option String :=
  match st.c.metaJ.getObjVal? "registration" with
  | .ok (.str d) => some d
  | _ => none

/-- **`bcs` are `cap`'s commit strings**: `2^capDepth lv` of them, node `i`'s the hm96 commit string of its row
(`Firewall.topRows`) under a salt. -/
def TopOf (lv : Level) (cap : Cap Refine.Dig) (bcs : Array ByteArray) : Prop :=
  bcs.size = 2 ^ capDepth lv ∧ ∃ salt : ℕ → ByteArray, ∀ i (h : i < bcs.size),
    bcs[i] = Flock.Hm96.Default512.commitString (Flock.Firewall.rowDigest ((Flock.Firewall.topRows (cap i).1).headD #[]))
      (salt i)

/-- **`cap` is R's registered top**: the registration record `reg` (`--rec-inner-registration`, the verifier's own
input) is the one the inner statement's META binds by its digest (`metaRegistration st`), it names `stacked` under its
`program`, and that entry is, canonically, what `topEntry` makes from `cap`'s commit strings (`TopOf`), as
`RecVStar.candidate3` compares them. -/
def StackedTop (lv : Level) (st : Flock.Stmt) (reg : Lean.Json) (cap : Cap Refine.Dig) : Prop :=
  ∃ (prog : String) (e e' : Lean.Json) (bcs : Array ByteArray) (s : String),
    Flock.RecVStar.registration (some reg) (metaRegistration st) = .ok (some (prog, some e)) ∧
      Flock.RecVStar.topEntry prog bcs = .ok e' ∧ Flock.canon e' = .ok s ∧ Flock.canon e = .ok s ∧ TopOf lv cap bcs

/-- **The verifier's hold of the inner transcript**: `RecVStar.innerCommits` as `RecSetup.commits` calls it, at the
coin server's record of V*'s inner session `tr` (`--rec-inner-transcript`), the inner session's record
`record` (`--rec-inner-session`), R's registration `reg` (`--rec-inner-registration`), and the digest and R's shape the
inner statement's META names (`metaRegistration`, `RecVStar.stackedShapeOf`). It refuses a registration META does not
bind, and a binding round of R's stream whose top does not make the registered `stacked` entry. -/
def innerCommitsAt (tr : Lean.Json) (record : String) (st : Flock.Stmt) (reg : Lean.Json) :
    Except String Flock.RecVStar.InnerCommits := do
  let shape ← Flock.RecVStar.stackedShapeOf st.c.metaJ
  Flock.RecVStar.innerCommits tr record (some reg) (metaRegistration st) shape

/-- **The inner circuit's unit range**, as check 5 reads it off the inner statement (`RecCoins.unitRangeOf`, which
`RecCoins.Inner.load` and `RecCoins.checkInnerRange` read): its one range of the unit net. -/
def unitRangeOf (st : Flock.Stmt) : Except String Flock.Range :=
  Flock.RecCoins.unitRangeOf st

/-- **Check 5's refusals at the sparse shape**, which read only the verifier's own inputs: at the inner shape's RecSparse
shape `sp` (`RecVStar.Inner.sparse`, the verifier's `--sparse`), RecSparse's shape check (`RecSparse.Shape.valid`,
`6 ≤ ELL`); and at inner points of the inner statement's `k_log − 6` coordinates, RecSparse's `v`'s check that they are
not short of `ELL − 6` (`RecCoins.sparseV`) and the slots' layout factors at the inner circuit's unit range
(`RecCoins.layoutFactors`: slots of `2^ELL` rows, at most `2^S_LOG` of them, inside the block). Both read only the
points' sizes, so this is their verdict at every rep's points (`RecCoins.checkSparse`). The verifier evaluates it on the
inner statement at every candidate-3 session's setup (`RecCoins.checkInnerRange`, through `RecCoins.checkAtC3`), at the
statement's own `k_log`, which is the class's where its class checks pass (`FlockC3.VStarList.sparseRangeOk_of_check`). -/
def SparseRangeOk (ish : Flock.RecVStar.Inner) (st : Flock.Stmt) : Prop :=
  ∀ sp, ish.sparse = some sp →
    let xs := Array.replicate ((execClassAt st).cls.kLog - 6) (0 : Flock.F128)
    sp.valid = true ∧ ¬ xs.size < sp.ell - Flock.RecSparse.PRE_ROUNDS ∧
      ∃ rng, unitRangeOf st = .ok rng ∧ (Flock.RecCoins.layoutFactors sp xs xs rng).isOk = true

/-- **Lean's verifier accepts its own inputs at candidate 3**: V*'s list builds (`RecVStar.statements`), it holds the
inner transcript against R's registration (`innerCommitsAt`), the covers check passes at the commits it holds
(`RecVStar.checkCovers`, all of it: the sparse-presence, stacked-presence and stacked-shape refusals, and each stream's
rounds against its schedule, `checkCommitRounds`), and check 5's sparse refusals pass (`SparseRangeOk`). It is not the
whole compiled acceptance: `AcceptsPublic` and `ChecksVR` carry the verifier's other checks. The verifier's acceptance
at both setups gives all of it (`FlockC3.VStarList.acceptsC3_of_verify`). -/
def AcceptsC3 (ish : Flock.RecVStar.Inner) (st : Flock.Stmt) (tr : Lean.Json) (record : String) (reg : Lean.Json) :
    Prop :=
  (Flock.RecVStar.statements ish).isOk = true ∧ (innerCommitsAt tr record st reg).isOk = true ∧
    (∀ cs, innerCommitsAt tr record st reg = .ok cs → (Flock.RecVStar.checkCovers ish cs).isOk = true) ∧
    SparseRangeOk ish st

/-- **The slots' unit slots are the verifier's**: slot `σ` fills unit `pos0 + σ` of the inner circuit's unit range
below its count, and none past it (`RecCoins.layoutFactors`). -/
def UnitsOf (st : Flock.Stmt) (hp : HoloParams) : Prop :=
  ∃ rng, unitRangeOf st = .ok rng ∧ ∀ σ, hp.units σ = if σ.val < rng.count then {rng.pos0 + σ.val} else ∅

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- **Candidate 3's compiled inner protocol**: `schedInner` at `Model.holoTableCL` on the class's public statement,
with R's registered cap the statement's. A statement is a registration, a word and its cap; it holds when some table
within the unique-decoding radius of the word has a satisfiable hidden statement at the slots' types. The table reads
only the cap: the word is what the statement is about. -/
noncomputable def holoInnerCL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) : Inner :=
  schedInner T CT (HoloReg F D)
    (fun x => holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
      e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap)
    (fun _ out => out.accepted = true)
    (fun x => ∃ R : RegTable e.cls, e.rsch.Close A x.w R ∧ ∃ z, (R.toStatement e.τ e.regs).Satisfies z)

/-- **Candidate 3's inner protocol as the executable runs it**: `holoInnerCL` at `execArith` and SHA-512 with plain
leaves, at `holoAt st hp`. -/
noncomputable abbrev holoInnerAt (st : Flock.Stmt) (hp : HoloParams) : Inner :=
  holoInnerCL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp)

/-! ### The error -/

/-- **One rep's sparse phase**, each coin its degree over `qF`: H1's 6 coins, H2's 6 rounds of degree 2, H3's `ν` rounds
of degree `2ℓ + 1` (`degRounds (2ℓ)`) and `sLog` of degree `2ℓ + 2`, H4's `pLog` coins and the type point's `tLog`. No
list: the phase's claim is on R's table, which R's opening decodes uniquely (`regOpenError`). -/
noncomputable def sparseError (sh : SparseShape) : ℝ :=
  ((18 + (2 * sh.ell + 1) * sh.nu + (2 * sh.ell + 2) * sh.sLog + sh.pLog + sh.tLog : ℕ) : ℝ) / qF

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
    (repError e.sch ⟨e.cls.kLog, e.regs.length, e.cls.mPts⟩ + sparseError e.cls.shape) ^ 2 +
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

/-- **Two runs of candidate 3's table clash against each other**: on the reps (`ClashCL` at the reps' schedule) or on
R's stream (`ClashCL` at R's, at the registered cap `cap`: a level-0 conflict there is two openings of one position of
R's level 0 against `cap` to different rows). Unlike `ClashHL`, no registered word is read, so no registrant's
openings are needed for a collision. -/
def ClashRunHL (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch rsch : Schedule) (cap : Cap D) (o o' : OutHL F D Sl) : Prop :=
  ClashCL H E Lf sch o.circ o'.circ ∨ ClashCL H E Lf rsch (o.reg cap) (o'.reg cap)

omit [Fintype F] [DecidableEq F] in
/-- **A clash of two runs is a collision.** -/
theorem ClashRunHL.collision {H : List UInt8 → D} {E : Merkle.Enc (List F) D} {Lf : Merkle.Leaf (List F) Sl D H}
    {sch rsch : Schedule} {cap : Cap D} {o o' : OutHL F D Sl} (h : ClashRunHL H E Lf sch rsch cap o o') :
    Merkle.Collision H := by
  rcases h with h | h
  · exact h.collision
  · exact h.collision

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
clash against each other (`ClashRunHL`, at the statement's cap), a collision of `H`, else nothing. -/
noncomputable def clashPickHL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  collOf H (∃ o o', outHL T CT A H E Lf e x t = some o ∧ outHL T CT A H E Lf e x t' = some o' ∧
      ClashRunHL H E Lf e.sch e.rsch.sch x.cap o o')
    fun ⟨_, _, _, _, hc⟩ => hc.collision

/-! ### V*'s statement list -/

/-- **The number of V*'s statements** at the inner shape `ish`, as the verifier builds the list
(`RecVStar.statements`); zero where it refuses. -/
def vstarCount (ish : Flock.RecVStar.Inner) : ℕ :=
  match Flock.RecVStar.statements ish with
  | .ok ss => ss.size
  | .error _ => 0

end FlockSoundness.Discharge.Recursion
