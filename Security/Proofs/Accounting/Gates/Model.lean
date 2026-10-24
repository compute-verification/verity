import Definitions.Accounting.Gates.Model

/-! The proof-only part of `Definitions.Accounting.Gates.Model`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Gates

namespace Op

variable {ι : Type}

/-- `ρ_b(x) = ⌊x/2^b⌋`, the rounding. -/
def round (b : ℕ) : Op ι := budgeted .rounding fun v => v.headD 0 / 2 ^ b

/-- `[a ≤ b]`, a comparison. -/
def le : Op ι := budgeted .compareSelect fun v => if v.headD 0 ≤ v.tail.headD 0 then 1 else 0

/-- `c ≠ 0 ? a : b`, a selection. -/
def select : Op ι := budgeted .compareSelect fun v => if v.headD 0 ≠ 0 then v.tail.headD 0 else v.tail.tail.headD 0

end Op

/-- **The committed words one checkpoint is split into**: `2` for an H1 + fx1 tile's receipt word (a checkpoint's two
halves, both of role `served t` at one `t`), `1` at any other site. A floor priced per checkpoint, as a dot kind's is,
counts once across a checkpoint's parts, `1/parts` a word. -/
def Site.parts : Site → ℕ
  | .receipt .. => 2
  | _ => 1

namespace Gate

variable {ι : Type} (g : Gate ι)

/-- Its charged bits: its declared bits if it is charged, else 0. -/
def chargedBits : ℕ := if g.op.charged then g.bits else 0

end Gate

variable {ι : Type}

/-- The committed outputs: the committed gates, in canonical order. -/
def committed (C : Dag ι) : List ℕ := (List.range C.length).filter fun i => C[i]?.any Gate.commit

/-- Canonical order: every gate's operands are earlier gates. -/
def Topological (C : Dag ι) : Prop := ∀ i (h : i < C.length), ∀ j ∈ C[i].args, j < i

/-- The charged bits: the declared bits of the run inputs and budgeted gates. -/
def chargedBits (C : Dag ι) : ℕ := (C.map Gate.chargedBits).sum

/-- Every budgeted gate but a gather reads at most `f` operands (a gather reads its index and the one word it
selects). -/
def FanIn (f : ℕ) (C : Dag ι) : Prop :=
  ∀ g ∈ C, g.op.charged → g.op.budget ≠ some .gather → g.args.length ≤ f

/-- Every value's declared interval lies within `[−2^w, 2^w)`: `w` bits and a sign, so s32 is `Width 31`. -/
def Width (w : ℕ) (C : Dag ι) : Prop := ∀ g ∈ C, -2 ^ w ≤ g.lo ∧ g.hi < 2 ^ w

end Accounting.Gates
