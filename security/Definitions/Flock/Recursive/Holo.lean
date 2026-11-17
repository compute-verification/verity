module

public import Definitions.Flock.Defs
public import Definitions.Flock.Holo
import Mathlib.Algebra.Order.Algebra
import Mathlib.Tactic.Positivity.Finset

/-! The recursion's hidden unit circuits: their class (`HoloClass`), registered table (`RegTable`) and registered
schedule (`RegSchedule`). -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting

structure HoloClass where
  m : ℕ
  kLog : ℕ
  hk : kLog ≤ m
  hk6 : 6 ≤ kLog
  hm : 13 ≤ m
  pin : Fin (2 ^ kLog)
  A₀pub : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  B₀pub : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  /-- A template has `2^slotLog` rows and columns (the sparse phase's `ℓ`). -/
  slotLog : ℕ
  hs6 : 6 ≤ slotLog
  hsk : slotLog ≤ kLog
  /-- A template has `2^nu` entries, padding included. -/
  nu : ℕ
  sLog : ℕ
  tLog : ℕ
  /-- The unit slots slot `σ` fills: diagonal block `u` is rows and columns `u·2^slotLog …`. -/
  units : Fin (2 ^ sLog) → Finset ℕ
  ptLocal : ℕ
  mPts : ℕ

namespace HoloClass

variable (cls : HoloClass)

def shape : SparseShape := ⟨cls.nu, cls.slotLog, cls.sLog, cls.tLog⟩

def pubStatement (regs : List (Region cls.kLog (cls.m - cls.kLog))) : Statement :=
  ⟨cls.m, cls.kLog, cls.hk, cls.hk6, cls.pin, cls.A₀pub, cls.B₀pub, regs⟩

end HoloClass

/-- The registered table of the class: bit `(t, p, k)` is plane `p` of entry `k` of type `t`'s template. -/
abbrev RegTable (cls : HoloClass) := Fin (2 ^ cls.tLog) → Fin (2 ^ cls.shape.pLog) → Fin (2 ^ cls.nu) → ZMod 2

namespace RegTable

variable {cls : HoloClass}

/-- Plane `p` of entry `k` of type `t`, zero past the planes. -/
def plane (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (p : ℕ) (k : Fin (2 ^ cls.nu)) : ZMod 2 :=
  if h : p < 2 ^ cls.shape.pLog then R t ⟨p, h⟩ k else 0

/-- The number planes `p₀ … p₀ + ℓ − 1` of entry `k` of type `t` spell, low bit first: its row at `p₀ = 0`, its column
at `p₀ = ℓ`. -/
def word (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (p₀ : ℕ) (k : Fin (2 ^ cls.nu)) : ℕ :=
  ∑ j ∈ Finset.range cls.slotLog, (R.plane t (p₀ + j) k).val * 2 ^ j

/-- Type `t`'s template, side `s` (`false` for `A₀`, `true` for `B₀`), in slot coordinates: entry `(i, c)` is the sum
of that side's values over the entries at row `i` and column `c`. -/
def tmpl (R : RegTable cls) (t : Fin (2 ^ cls.tLog)) (s : Bool) (i c : ℕ) : ZMod 2 :=
  ∑ k : Fin (2 ^ cls.nu), if R.word t 0 k = i ∧ R.word t cls.slotLog k = c then
    R.plane t (2 * cls.slotLog + s.toNat) k else 0

/-- One side of the block's matrix at the slots' types `τ`: the public part, plus, when row `i` and column `c` are in
the same unit slot `u`, the templates of the slots that fill `u`. -/
def side (R : RegTable cls) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (s : Bool) :
    Matrix (Fin (2 ^ cls.kLog)) (Fin (2 ^ cls.kLog)) (ZMod 2) :=
  fun i c => (if s then cls.B₀pub else cls.A₀pub) i c +
    if i.val / 2 ^ cls.slotLog = c.val / 2 ^ cls.slotLog then
      ∑ σ, if i.val / 2 ^ cls.slotLog ∈ cls.units σ then
        R.tmpl (τ σ) s (i.val % 2 ^ cls.slotLog) (c.val % 2 ^ cls.slotLog) else 0
    else 0

def toStatement (R : RegTable cls) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog))
    (regs : List (Region cls.kLog (cls.m - cls.kLog))) : Statement :=
  ⟨cls.m, cls.kLog, cls.hk, cls.hk6, cls.pin, R.side τ false, R.side τ true, regs⟩

theorem type_lt {a : ℕ} (h : a < 2 ^ (cls.nu + cls.shape.pLog + cls.tLog)) :
    a / 2 ^ (cls.nu + cls.shape.pLog) < 2 ^ cls.tLog := by
  rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← pow_add, Nat.add_comm]
  exact h

/-- The stacked table the registered opening reads: bit `k + 2^ν·(p + 2^pLog·t)` is `R t p k`, and the padding's bits
past the cells (`SparseShape.mReg`) are zero. -/
def bits (R : RegTable cls) : Witness cls.shape.mReg := fun a =>
  if h : a.val < 2 ^ (cls.nu + cls.shape.pLog + cls.tLog) then
    R ⟨a.val / 2 ^ (cls.nu + cls.shape.pLog), type_lt h⟩
      ⟨a.val / 2 ^ cls.nu % 2 ^ cls.shape.pLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩
      ⟨a.val % 2 ^ cls.nu, Nat.mod_lt _ (Nat.two_pow_pos _)⟩
  else 0

end RegTable

structure RegSchedule (cls : HoloClass) where
  sch : Schedule

namespace RegSchedule

variable {cls : HoloClass} (rsch : RegSchedule cls)

noncomputable def radius : ℝ := (1 - rsch.sch.level₀.rate) / 2 - eta

/-- Some level-0 message within the unique-decoding radius of `w` packs `R`'s stacked bits. The padding's bits past
them are the registrant's: no claim reads them and the verifier never sees them. -/
def Close {F K : Type} [Field F] [Field K] (A : Arith F K) (w : Oracle F) (R : RegTable cls) : Prop :=
  ∃ M : Message F rsch.sch.k0 rsch.sch.level₀.logCols,
    Model.Close rsch.radius (asTable rsch.sch w) (encode A rsch.sch.level₀.logLen M) ∧
      ∀ a : Fin (2 ^ cls.shape.mReg), a.val < 2 ^ (cls.nu + cls.shape.pLog + cls.tLog) →
        witnessOf A cls.shape.mReg rsch.sch.k0 rsch.sch.level₀.logCols M a = R.bits a

def UniqueDecoding : Prop :=
  rsch.sch.levels ≠ [] ∧ rsch.sch.m = cls.shape.mReg ∧ rsch.sch.level₀.lanes = rsch.sch.k0 ∧
    rsch.sch.k0 + rsch.sch.level₀.logCols + 7 = cls.shape.mReg ∧ 1 ≤ rsch.sch.level₀.logInvRate

end RegSchedule

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

def holoTableOf (A : Arith F K) (cls : HoloClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (rsch : RegSchedule cls) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (w : Oracle F) :
    Game (Bool × Oracle F) :=
  holoTable A (cls.pubStatement regs) sch cls.hm cls.ptLocal cls.mPts cls.shape cls.units (fun σ => (τ σ).val)
    rsch.sch w

end FlockSoundness.Discharge.Recursion
