import FlockSoundness.Discharge.Exec.SessionL
import FlockSoundness.Discharge.Exec.StrictCRL
import FlockSoundness.Refine.Run

/-!
# Recursion: V*'s target relation

What V*, the outer circuit specialised to the inner coins, has to compute: the inner verifier's acceptance of a hidden
block statement on the inner messages and the inner coins.

* `HClass`: what the class publishes about the hidden statement: `m`, `k_log`, the pin, the public part of `A₀` and
  `B₀` (hashing and mask slots, the copies between rows and unit ports), the hidden template's slot (`2^slotLog` rows and
  columns) and the unit slots it fills, the link layout (`ptLocal`, `mPts`), and `nnz`, the number of hidden entries,
  padding included.
* `Hidden cls`: the hidden template, `nnz` entries `(side, row, column, value)` in slot coordinates. `toStatement` is the
  block R1CS statement: `A₀[i, c]` is the public part's entry plus, when `i` and `c` are in the same unit slot, the sum of
  the values of the side-`A` entries at `(i mod 2^slotLog, c mod 2^slotLog)`; `B₀` likewise; with the class's pin and the
  public regions. `Canonical`: the value-1 entries strictly increasing in (side, column, row), then the padding
  `(A, 0, 0, 0)`.
* `InnerAccepts`: C-Flock's compiled table under a leaf scheme `Lf` (`Model.tableCL`), after its level-0 cap and the
  link points, played on that statement with the inner messages and coins (`Game.Plays`), ends accepted.
* `hidden_sound_iop`, `hidden_sound_fast100`: the interactive table (oracles sent whole, live coins) on every hidden
  statement of the class: when no witness satisfies it, every prover makes it accept with probability at most
  `tableError`, at most `2^-205` for `22 ≤ m ≤ 33`. Statistical: no assumption, so it is `InnerSound`'s kind of bound.
* `hidden_sound`: the compiled theorem (`table_sound_compiled_strictL`) for every hidden statement of the class, at its
  finders' budgets.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Exec StrictCR

/-- What the class publishes about a hidden block statement. -/
structure HClass where
  m : ℕ
  kLog : ℕ
  hk : kLog ≤ m
  hk6 : 6 ≤ kLog
  hm : 13 ≤ m
  pin : Fin (2 ^ kLog)
  /-- The public part of `A₀`. -/
  A₀pub : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  /-- The public part of `B₀`. -/
  B₀pub : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  /-- The hidden template's slot has `2^slotLog` rows and columns. -/
  slotLog : ℕ
  /-- The unit slots the template fills: diagonal block `u` is rows and columns `u·2^slotLog …`. -/
  units : Finset ℕ
  ptLocal : ℕ
  mPts : ℕ
  /-- The number of hidden entries, padding included. -/
  nnz : ℕ

/-- One entry of a hidden template: its side (`false` for `A₀`, `true` for `B₀`), its row and column in a slot of
`2^slotLog`, and its value. -/
structure HEntry (slotLog : ℕ) where
  side : Bool
  row : Fin (2 ^ slotLog)
  col : Fin (2 ^ slotLog)
  val : ZMod 2
  deriving DecidableEq

namespace HEntry

variable {slotLog : ℕ}

/-- The padding entry, `(A, 0, 0, 0)`. -/
def pad : HEntry slotLog := ⟨false, ⟨0, Nat.two_pow_pos slotLog⟩, ⟨0, Nat.two_pow_pos slotLog⟩, 0⟩

/-- The canonical order's key: side, then column, then row. -/
def key (e : HEntry slotLog) : ℕ := e.side.toNat * 2 ^ (2 * slotLog) + e.col.val * 2 ^ slotLog + e.row.val

end HEntry

/-- A hidden template of the class `cls`. -/
abbrev Hidden (cls : HClass) := Fin cls.nnz → HEntry cls.slotLog

namespace Hidden

variable {cls : HClass}

/-- The template's side `s` in slot coordinates: entry `(i, c)` is the sum of the values of that side's entries at
`(i, c)`. -/
def tmpl (M : Hidden cls) (s : Bool) (i c : ℕ) : ZMod 2 :=
  ∑ e : Fin cls.nnz, if (M e).side = s ∧ (M e).row.val = i ∧ (M e).col.val = c then (M e).val else 0

/-- One side of the block's matrix: the public part, plus the template when row `i` and column `c` are in the same unit
slot. -/
def side (M : Hidden cls) (s : Bool) : Matrix (Fin (2 ^ cls.kLog)) (Fin (2 ^ cls.kLog)) (ZMod 2) :=
  fun i c => (if s then cls.B₀pub else cls.A₀pub) i c +
    if i.val / 2 ^ cls.slotLog = c.val / 2 ^ cls.slotLog ∧ i.val / 2 ^ cls.slotLog ∈ cls.units then
      M.tmpl s (i.val % 2 ^ cls.slotLog) (c.val % 2 ^ cls.slotLog)
    else 0

/-- **The hidden statement**: the block R1CS with the class's `m`, `k_log` and pin, `A₀` and `B₀` the public part plus
the template at the unit slots, and the public regions `regs`. -/
def toStatement (M : Hidden cls) (regs : List (Region cls.kLog (cls.m - cls.kLog))) : Statement :=
  ⟨cls.m, cls.kLog, cls.hk, cls.hk6, cls.pin, M.side false, M.side true, regs⟩

/-- **The canonical form**: the first `n` entries are the value-1 ones, strictly increasing in (side, column, row); the
rest are padding. -/
def Canonical (M : Hidden cls) : Prop :=
  ∃ n ≤ cls.nnz, (∀ e : Fin cls.nnz, e.val < n → (M e).val = 1) ∧ (∀ e : Fin cls.nnz, n ≤ e.val → M e = HEntry.pad) ∧
    ∀ e e' : Fin cls.nnz, e < e' → e'.val < n → (M e).key < (M e').key

end Hidden

section Target

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl W : Type}
  [Inhabited W]

/-- **V*'s target relation.** C-Flock's compiled table under the leaf scheme `Lf`, on the hidden statement
`M.toStatement regs` at the schedule `sch`, after its level-0 cap `cap₀` and the link points `pts` (`tableCL_eq`), played
with the inner messages `ms` and the inner coins `cs` (read through `ι`), ends accepted. This is the run flock-verify
refines each table it accepts to (`accepts_tables_hm96_of_leaf`). -/
def InnerAccepts (ι : W → F) (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (sch : Schedule) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (M : Hidden cls) (cap₀ : Cap D) (pts : Fin 2 → Fin cls.mPts → F) (ms : List Game.Msg) (cs : List (Array W)) :
    Prop :=
  ∃ out : OutCL F D Sl,
    Plays ι (tableAfterL A H E Lf (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts cap₀ pts) ms cs out ∧
      out.accepted = true

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
/-- A level-0 table the statement's committed values satisfy has a witness. -/
theorem satisfiable_of_committed {A : Arith F K} {S : Statement} {sch : Schedule} {C : Oracle F}
    (h : Committed A S sch C) : ∃ z, S.Satisfies z :=
  let ⟨_, _, hs⟩ := h
  ⟨_, hs⟩

/-- **The interactive table is sound on every hidden statement of the class**, with no assumption. If no witness
satisfies `M.toStatement regs`, every prover makes the table (`Model.table`: the oracles sent whole, live coins) accept
with probability at most `tableError`. Nothing reads a parsed circuit: the statement is any hidden template of the class,
so it can be a function of whatever precedes the table's coins. -/
theorem hidden_sound_iop (A : Arith F K) (hA : A.Correct) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls) (sch : Schedule)
    (hsch : fast100 cls.m = some sch) (hlay : (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts)
    (hno : ¬ ∃ z, (M.toStatement regs).Satisfies z)
    (σ : Strategy (table A (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)) :
    prob (fun out => out.1 = true) (table A (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) σ ≤
      ENNReal.ofReal (tableError sch ((M.toStatement regs).shape cls.mPts)) :=
  (prob_mono (fun _ h => ⟨h, fun hc => hno (satisfiable_of_committed hc)⟩) _ σ).trans
    (table_sound A hA (M.toStatement regs) sch hsch cls.hm cls.ptLocal cls.mPts hlay σ)

/-- **`hidden_sound_iop` with the numbers**: at most `2^-205` for `22 ≤ m ≤ 33` and a statement shape in range. -/
theorem hidden_sound_fast100 (A : Arith F K) (hA : A.Correct) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls) (sch : Schedule)
    (hsch : fast100 cls.m = some sch) (hm1 : 22 ≤ cls.m) (hm2 : cls.m ≤ 33)
    (hshape : ((M.toStatement regs).shape cls.mPts).InRange)
    (hlay : (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts) (hno : ¬ ∃ z, (M.toStatement regs).Satisfies z)
    (σ : Strategy (table A (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts)) :
    prob (fun out => out.1 = true) (table A (M.toStatement regs) sch cls.hm cls.ptLocal cls.mPts) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  (prob_mono (fun _ h => ⟨h, fun hc => hno (satisfiable_of_committed hc)⟩) _ σ).trans
    (table_sound_fast100 A hA (M.toStatement regs) sch hsch hm1 hm2 cls.ptLocal cls.mPts hshape hlay σ)

/-- **The compiled table is sound on every hidden statement of the class.** If no witness satisfies
`M.toStatement regs` and the compiled theorem's finders are collision resistant at budgets `qF` and `qS`, the table
accepts with probability at most `table_sound_compiled_strictL`'s bound. Nothing here reads a parsed circuit: the
statement is any hidden matrix of the class, so it can be a function of whatever precedes the table's coins. -/
theorem hidden_sound (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass)
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hsch : fast100 cls.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (hlay : (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts)
    (hno : ¬ ∃ z, (M.toStatement regs).Satisfies z)
    (σ : Strategy (tableCL A H E Lf (M.toStatement regs) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cls.hm cls.ptLocal cls.mPts))
    {qF qS : ℝ}
    (hCR : CompiledCRL A H E Lf (M.toStatement regs) m k0 l₀ l₁ rest ext (fast100_shape cls.m _ hsch).2.1 cls.hm
      cls.ptLocal cls.mPts σ qF qS) :
    expect (ind fun out : OutCL F D Sl => out.accepted = true)
      (tableCL A H E Lf (M.toStatement regs) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cls.hm cls.ptLocal cls.mPts) σ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ((M.toStatement regs).shape cls.mPts) +
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) * (qF ^ 2 / 2 ^ (513 : ℝ))) +
      ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
          (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
      qS ^ 2 / 2 ^ (513 : ℝ) :=
  (expect_mono (ind_mono fun _ h => ⟨h, fun hc => hno (satisfiable_of_committed hc)⟩) _ σ).trans
    (table_sound_compiled_strictL A hA H E Lf (M.toStatement regs) m k0 l₀ l₁ rest ext hsch cls.hm cls.ptLocal
      cls.mPts hlay σ hCR)

end Target

end FlockSoundness.Discharge.Recursion
