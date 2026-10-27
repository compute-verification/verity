import Mathlib

/-!
# The model M3 for Dither's words (trusted)

M3: a straight-line program over 32-bit words, each instruction any fixed function of at most three earlier words.

* At the zero instance the inputs are the dither words (`R i ℓ p`, bit `p` of `R_ℓ[i]`) and the columns of `U`
  (`U j p`, 32 entries in `±1..±64`); at a general instance the int8 entries of `Ã`'s rows and `B̃`'s columns, per
  32-entry segment.
* Words that depend on the row side alone or the column side alone are free (re-packing, tables of `U`, bit planes):
  they are shared by every column or every row and amortize to zero, so the model gives them away. A program's inputs
  are such free words (`fr`, `fu`), any functions of one side.
* The outputs are the prefix sums `T_ℓ[i,j] = Σ_{ℓ' ≤ ℓ} Σ_p R[i, 32ℓ'+p]·U[j,p]` (zero instance) or
  `Σ_{ℓ' ≤ ℓ} Σ_p Ã[i, 32ℓ'+p]·B̃[32ℓ'+p, j]` (general), as s32 words, one designated instruction each (`out`).
* `WordLocal`: the instructions split into blocks `P_τ`, one per output, holding its writer, every input of a `P_τ`
  instruction in `P_τ`, the previous segment's output `T_{ℓ−1}[i,j]`, or free (Theorem B's class).
* `ClassP`: every instruction is an output writer, a free word, or a mixed-radix tuple of fields
  `φ = β·T_{ℓ−1}[i,j] + Σ_{p∈E} R[i, 32ℓ+p]·U[j,p]` with `M_{k+1} ≥ M_k·|range_k|` and `Π_k |range_k| ≤ 2^32`, a field
  with a prefix taking it from an input that holds it (Theorem C's class).
* `Creates`: an instruction creates an incidence `(τ, p)` it holds and none of its instruction arguments holds
  (Theorem C's count).
-/

namespace Pouw.Eps1

open Finset

/-- A 32-bit word. -/
abbrev Word := BitVec 32

/-- **An M3 program** on inputs `ι` with `G` instructions: instruction `t` applies `op t` to three words, each an input
or an earlier instruction's value. -/
structure M3 (ι : Type) (G : ℕ) where
  arg : (t : Fin G) → Fin 3 → ι ⊕ Fin t.val
  op : Fin G → Word → Word → Word → Word

namespace M3

variable {ι : Type} {G : ℕ}

/-- The earlier instruction an argument names, as an instruction of the program. -/
def earlier (t : Fin G) (s : Fin t.val) : Fin G := ⟨s.val, s.isLt.trans t.isLt⟩

/-- The value of argument `j` of instruction `t`, given the inputs `x` and the instructions' values `v`. -/
def argVal (P : M3 ι G) (x : ι → Word) (v : Fin G → Word) (t : Fin G) (j : Fin 3) : Word :=
  match P.arg t j with
  | .inl i => x i
  | .inr s => v (earlier t s)

/-- `v` is the program's run on inputs `x`: each instruction's value is its function of its arguments' values. -/
def IsRun (P : M3 ι G) (x : ι → Word) (v : Fin G → Word) : Prop :=
  ∀ t, v t = P.op t (P.argVal x v t 0) (P.argVal x v t 1) (P.argVal x v t 2)

end M3

/-- The outputs' indices `τ = (ℓ, i, j)`: segment, row, column. -/
abbrev Tau (m n Lseg : ℕ) : Type := Fin Lseg × Fin m × Fin n

/-- The previous segment's output index `(ℓ − 1, i, j)`, for `ℓ > 0`. -/
def Tau.prev {m n Lseg : ℕ} (τ : Tau m n Lseg) (h : 0 < τ.1.val) : Tau m n Lseg :=
  (⟨τ.1.val - 1, by omega⟩, τ.2)

/-- The dither bits at the zero instance: `R i ℓ p` is bit `p` of `R_ℓ[i]`. -/
abbrev Dither (m Lseg : ℕ) : Type := Fin m → Fin Lseg → Fin 32 → Bool

/-- The columns of `U`: `U j p` is entry `p` of column `j`. -/
abbrev Cols (n : ℕ) : Type := Fin n → Fin 32 → ℤ

/-- `U`'s entries are in `±1..±64` (7 bits each, 224 bits a column). -/
def ColsValid {n : ℕ} (U : Cols n) : Prop := ∀ j p, 1 ≤ |U j p| ∧ |U j p| ≤ 64

/-- Segment `ℓ`'s product at the zero instance: `Σ_p R[i, 32ℓ+p]·U[j,p]`. -/
def zeroSeg {m n Lseg : ℕ} (R : Dither m Lseg) (U : Cols n) (l : Fin Lseg) (i : Fin m) (j : Fin n) : ℤ :=
  ∑ p, if R i l p then U j p else 0

/-- The prefix `Σ_{ℓ' < ℓ}` of segment products at the zero instance (`T_{ℓ−1}[i,j]` over ℤ). -/
def zeroPrefix {m n Lseg : ℕ} (R : Dither m Lseg) (U : Cols n) (τ : Tau m n Lseg) : ℤ :=
  ∑ l ∈ univ.filter (· < τ.1), zeroSeg R U l τ.2.1 τ.2.2

/-- The output word `T_ℓ[i,j] = Σ_{ℓ' ≤ ℓ} Σ_p R[i, 32ℓ'+p]·U[j,p]` at the zero instance, as an s32 word. -/
def zeroOut {m n Lseg : ℕ} (R : Dither m Lseg) (U : Cols n) (τ : Tau m n Lseg) : Word :=
  BitVec.ofInt 32 (zeroPrefix R U τ + zeroSeg R U τ.1 τ.2.1 τ.2.2)

/-- `Ã`'s rows at a general instance: `A i ℓ p` is `Ã[i, 32ℓ+p]`. -/
abbrev Rows (m Lseg : ℕ) : Type := Fin m → Fin Lseg → Fin 32 → ℤ

/-- `B̃`'s columns at a general instance: `B j ℓ p` is `B̃[32ℓ+p, j]`. -/
abbrev GCols (n Lseg : ℕ) : Type := Fin n → Fin Lseg → Fin 32 → ℤ

/-- An int8 entry. -/
def Int8Valid (z : ℤ) : Prop := -128 ≤ z ∧ z ≤ 127

/-- The output word `T_ℓ[i,j] = Σ_{ℓ' ≤ ℓ} Σ_p Ã[i, 32ℓ'+p]·B̃[32ℓ'+p, j]` at a general instance, as an s32 word. -/
def genOut {m n Lseg : ℕ} (A : Rows m Lseg) (B : GCols n Lseg) (τ : Tau m n Lseg) : Word :=
  BitVec.ofInt 32 (∑ l ∈ univ.filter (· ≤ τ.1), ∑ p, A τ.2.1 l p * B τ.2.2 l p)

/-- **An M3 program for Dither's words**, over row-side data `Rd` and column-side data `Cd`: its free words, each a
function of the row side alone (`fr`) or of the column side alone (`fu`), its instructions over them, and the writer
`out τ` of each output word. -/
structure Prog (Rd Cd : Type) (m n Lseg : ℕ) (FR FU : Type) (G : ℕ) where
  fr : FR → Rd → Word
  fu : FU → Cd → Word
  prog : M3 (FR ⊕ FU) G
  out : Tau m n Lseg → Fin G

namespace Prog

variable {Rd Cd : Type} {m n Lseg : ℕ} {FR FU : Type} {G : ℕ}

/-- The program's inputs on row data `x` and column data `y`. -/
def inputs (P : Prog Rd Cd m n Lseg FR FU G) (x : Rd) (y : Cd) : FR ⊕ FU → Word :=
  Sum.elim (fun a => P.fr a x) (fun b => P.fu b y)

/-- The program computes `T` on the instances `ok` admits: on each, every run's output writers hold `T`. -/
def Computes (P : Prog Rd Cd m n Lseg FR FU G) (ok : Rd → Cd → Prop) (T : Rd → Cd → Tau m n Lseg → Word) : Prop :=
  ∀ x y, ok x y → ∀ v, P.prog.IsRun (P.inputs x y) v → ∀ τ, v (P.out τ) = T x y τ

/-- **Word-local** (Theorem B's class) with blocks `blk`: each output's writer is in its own block, and every argument
of an instruction is a free word, an instruction of its block, or the previous segment's output writer of its block's
`(i, j)`. -/
def WordLocal (P : Prog Rd Cd m n Lseg FR FU G) (blk : Fin G → Tau m n Lseg) : Prop :=
  (∀ τ, blk (P.out τ) = τ) ∧
    ∀ t j, ∀ s, P.prog.arg t j = .inr s →
      blk (M3.earlier t s) = blk t ∨ ∃ h : 0 < (blk t).1.val, M3.earlier t s = P.out ((blk t).prev h)

end Prog

/-- A field for `τ = (ℓ, i, j)` at the zero instance: `φ = β·T_{ℓ−1}[i,j] + Σ_{p∈E} R[i, 32ℓ+p]·U[j,p]`. -/
structure Field (m n Lseg : ℕ) where
  τ : Tau m n Lseg
  β : Bool
  E : Finset (Fin 32)

namespace Field

variable {m n Lseg : ℕ}

/-- The field's value. -/
def val (φ : Field m n Lseg) (R : Dither m Lseg) (U : Cols n) : ℤ :=
  (if φ.β then zeroPrefix R U φ.τ else 0) + ∑ p ∈ φ.E, if R φ.τ.2.1 φ.τ.1 p then U φ.τ.2.2 p else 0

/-- The values the field takes over every instance. -/
def range (φ : Field m n Lseg) : Set ℤ := {z | ∃ R U, ColsValid U ∧ φ.val R U = z}

end Field

/-- **A mixed-radix tuple of fields**: fields `φ_k` with offsets `lo_k` and multipliers `M_k`, its word
`Σ_k (φ_k − lo_k)·M_k`, each `lo_k` the least value of `φ_k`'s range, `M_0 ≥ 1`, `M_{k+1} ≥ M_k·|range_k|`,
`Π_k |range_k| ≤ 2^32`, and the tuple's largest value below `2^32`, so that the word is the tuple (injective). -/
structure Packing (m n Lseg : ℕ) where
  fields : List (Field m n Lseg × ℤ × ℕ)

namespace Packing

variable {m n Lseg : ℕ}

/-- The tuple's word. -/
def word (π : Packing m n Lseg) (R : Dither m Lseg) (U : Cols n) : Word :=
  BitVec.ofInt 32 (π.fields.map fun f => (f.1.val R U - f.2.1) * (f.2.2 : ℤ)).sum

/-- The tuple is mixed-radix: each offset the least value of its field's range, multipliers growing by the ranges,
ranges multiplying to at most `2^32`, and the largest value `Σ_k (|range_k| − 1)·M_k` below `2^32`. -/
def MixedRadix (π : Packing m n Lseg) : Prop :=
  (∀ f ∈ π.fields, f.2.1 ∈ f.1.range ∧ ∀ z ∈ f.1.range, f.2.1 ≤ z) ∧
    (∀ (k : ℕ) (hk : k + 1 < π.fields.length),
      (π.fields[k].2.2 : ℝ) * (π.fields[k].1.range.ncard : ℝ) ≤ (π.fields[k + 1].2.2 : ℝ)) ∧
    (∀ h : 0 < π.fields.length, 1 ≤ π.fields[0].2.2) ∧
    (π.fields.map fun f => (f.1.range.ncard : ℝ)).prod ≤ 2 ^ 32 ∧
    (π.fields.map fun f => ((f.1.range.ncard : ℝ) - 1) * (f.2.2 : ℝ)).sum < 2 ^ 32

end Packing

/-- Instruction `t` **holds** the incidence `(τ, p)`, entry `p` of segment `ℓ = τ.1` of `(i, j) = τ.2`, under the
tuples `pk`: it writes an output `T_{ℓ'}[i,j]` with `ℓ' ≥ ℓ`, or its tuple has a field for `τ` with `p ∈ E`, or a
field with a prefix covering segment `ℓ` of `(i, j)` (a field for `(ℓ', i, j)`, `ℓ' > ℓ`, with `β = true`). Free words
hold nothing. -/
def Holds {m n Lseg : ℕ} {FR FU : Type} {G : ℕ} (P : Prog (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) (t : Fin G) (τ : Tau m n Lseg) (p : Fin 32) : Prop :=
  (∃ τ', P.out τ' = t ∧ τ'.2 = τ.2 ∧ τ.1 ≤ τ'.1) ∨
    ∃ π, pk t = some π ∧ ∃ f ∈ π.fields,
      (f.1.τ = τ ∧ p ∈ f.1.E) ∨ (f.1.β = true ∧ f.1.τ.2 = τ.2 ∧ τ.1 < f.1.τ.1)

/-- **Class P** (Theorem C's class) with tuples `pk`: every instruction is an output writer, a free word (on every
instance a function of the dither alone or of `U` alone), or a mixed-radix tuple of fields whose word it holds on
every instance (and only a tuple has `pk t = some _`); and a field with a prefix (`β = true`) for `τ = (ℓ, i, j)` with
`ℓ > 0` takes it from an argument that holds it: one argument holding every incidence of the segments before `ℓ` of
`(i, j)`. -/
def ClassP {m n Lseg : ℕ} {FR FU : Type} {G : ℕ} (P : Prog (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) : Prop :=
  ∀ t, ((∃ τ, P.out τ = t) ∧ pk t = none) ∨
    (pk t = none ∧
      ((∃ g : Dither m Lseg → Word, ∀ R U, ColsValid U → ∀ v, P.prog.IsRun (P.inputs R U) v → v t = g R) ∨
        ∃ g : Cols n → Word, ∀ R U, ColsValid U → ∀ v, P.prog.IsRun (P.inputs R U) v → v t = g U)) ∨
    ∃ π, pk t = some π ∧ π.MixedRadix ∧
      (∀ R U, ColsValid U → ∀ v, P.prog.IsRun (P.inputs R U) v → v t = π.word R U) ∧
      ∀ f ∈ π.fields, f.1.β = true → 0 < f.1.τ.1.val → ∃ j s, P.prog.arg t j = .inr s ∧
        ∀ l, l < f.1.τ.1 → ∀ p, Holds P pk (M3.earlier t s) (l, f.1.τ.2) p

/-- Instruction `t` **creates** the incidence `(τ, p)`: it holds it and no argument that is an instruction holds it.
Walking back from `τ`'s writer through arguments that hold `(τ, p)` ends at an instruction that creates it. -/
def Creates {m n Lseg : ℕ} {FR FU : Type} {G : ℕ} (P : Prog (Dither m Lseg) (Cols n) m n Lseg FR FU G)
    (pk : Fin G → Option (Packing m n Lseg)) (t : Fin G) (τ : Tau m n Lseg) (p : Fin 32) : Prop :=
  Holds P pk t τ p ∧ ∀ j s, P.prog.arg t j = .inr s → ¬ Holds P pk (M3.earlier t s) τ p

/-- The number of incidences instruction `t` creates. -/
noncomputable def created {m n Lseg : ℕ} {FR FU : Type} {G : ℕ}
    (P : Prog (Dither m Lseg) (Cols n) m n Lseg FR FU G) (pk : Fin G → Option (Packing m n Lseg)) (t : Fin G) : ℕ :=
  {x : Tau m n Lseg × Fin 32 | Creates P pk t x.1 x.2}.ncard

end Pouw.Eps1
