import Proofs.Flock.Soundness.Defs
import Proofs.Flock.Soundness.Model.Holo

/-!
# Recursion: hidden unit circuits' class, registered table and registered schedule

F4 candidate 3's statement objects (note:proofs/20261010T0250Z-draft-f4-bitsumcheck §2.1, row S1 of its §4), beside
`Target.lean`'s `HClass` and `Hidden`:

* `HoloClass`: what the class publishes about a block statement with hidden unit circuits: `HClass`'s fields, with the
  template's `ℓ = slotLog` between 6 and `k_log`, `2^ν` entries a template, `2^sLog` slots, slot `σ` filling the unit
  slots `units σ`, and `2^tLog` registered types.
* `RegTable cls`: the registered table, bit `(t, p, k)` of type `t`'s template: planes `p < ℓ` are entry `k`'s row bits,
  `ℓ ≤ p < 2ℓ` its column bits, `2ℓ` its `A`-side value and `2ℓ + 1` its `B`-side value (`rec_sparse`'s `R(t, p, k)`).
  `tmpl`, `side` and `toStatement` are `Hidden`'s, with slot `σ`'s template the one of type `τ σ`; `bits` is the
  stacked table the registered opening reads, bit `k + 2^ν·(p + 2^pLog·t)`.
* `RegSchedule cls`: the registered opening's Ligerito schedule, its level 0 the registered word, read at the
  unique-decoding radius `δ = (1 − ρ)/2 − η` (`Close`); `UniqueDecoding` is the shape under which level 0 holds the
  stacked table and that radius is positive.
* `holoTableOf`: `Model.holoTable` on the class's public statement at the slots' types.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting

/-- What the class publishes about a block statement with hidden unit circuits. -/
structure HoloClass where
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
  /-- A template has `2^slotLog` rows and columns (the sparse phase's `ℓ`). -/
  slotLog : ℕ
  hs6 : 6 ≤ slotLog
  hsk : slotLog ≤ kLog
  /-- A template has `2^nu` entries, padding included. -/
  nu : ℕ
  /-- The slots' bits. -/
  sLog : ℕ
  /-- The registered types' bits. -/
  tLog : ℕ
  /-- The unit slots slot `σ` fills: diagonal block `u` is rows and columns `u·2^slotLog …`. -/
  units : Fin (2 ^ sLog) → Finset ℕ
  ptLocal : ℕ
  mPts : ℕ

namespace HoloClass

variable (cls : HoloClass)

/-- The sparse phase's shape. -/
def shape : SparseShape := ⟨cls.nu, cls.slotLog, cls.sLog, cls.tLog⟩

/-- The statement the reps' lincheck runs on: the public parts of `A₀` and `B₀`, the pin and the regions `regs`. -/
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

/-- **The hidden statement**: the block R1CS with the class's `m`, `k_log` and pin, `A₀` and `B₀` the public part plus
the slots' templates at their unit slots, and the public regions `regs`. -/
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

/-- The registered schedule: the registered opening's Ligerito schedule, whose level 0 is the registered word. -/
structure RegSchedule (cls : HoloClass) where
  sch : Schedule

namespace RegSchedule

variable {cls : HoloClass} (rsch : RegSchedule cls)

/-- Level 0's unique-decoding radius `δ = (1 − ρ)/2 − η` (§1.4). -/
noncomputable def radius : ℝ := (1 - rsch.sch.level₀.rate) / 2 - eta

/-- The registered word `w` is close to the registered table `R`: some level-0 message whose encoding is within the
unique-decoding radius of `w` packs `R`'s stacked bits. -/
def Close {F K : Type} [Field F] [Field K] (A : Arith F K) (w : Oracle F) (R : RegTable cls) : Prop :=
  ∃ M : Message F rsch.sch.k0 rsch.sch.level₀.logCols,
    Model.Close rsch.radius (asTable rsch.sch w) (encode A rsch.sch.level₀.logLen M) ∧
      witnessOf A cls.shape.mReg rsch.sch.k0 rsch.sch.level₀.logCols M = R.bits

/-- **Level 0 decodes uniquely**: the schedule is for the stacked table's `ν + pLog + tLog` bits, its level 0 holds
them exactly (`2^k₀` lanes of `2^logCols` packed elements of 128 bits), and level 0's rate is at most `1/2`, so that
the radius is positive. -/
def UniqueDecoding : Prop :=
  rsch.sch.levels ≠ [] ∧ rsch.sch.m = cls.shape.mReg ∧ rsch.sch.level₀.lanes = rsch.sch.k0 ∧
    rsch.sch.k0 + rsch.sch.level₀.logCols + 7 = cls.shape.mReg ∧ 1 ≤ rsch.sch.level₀.logInvRate

end RegSchedule

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **The holographic table of the class** at the slots' types `τ`: `Model.holoTable` on the class's public statement,
its shape and units, at the schedule `sch`, the registered schedule `rsch` and the registered word `w`. -/
def holoTableOf (A : Arith F K) (cls : HoloClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (rsch : RegSchedule cls) (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (w : Oracle F) :
    Game (Bool × Oracle F) :=
  holoTable A (cls.pubStatement regs) sch cls.hm cls.ptLocal cls.mPts cls.shape cls.units (fun σ => (τ σ).val)
    rsch.sch w

end FlockSoundness.Discharge.Recursion
