import Definitions.Accounting.Gates.Registry

/-! The proof-only part of `Definitions.Accounting.Gates.Registry`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Accounting.Gates

/-- The class of a gate's operation. -/
def Op.cls {ι : Type} : Op ι → Class
  | .input b _ => .input b
  | .const _ => .const
  | .affine .. => .affine
  | .mma a b k sat => .family a b k sat
  | .gather low d => .gather low d
  | .budgeted cls _ => .budgeted cls

/-- A class as its primitive's id lists it: a coin input's is a run input's (`Input{n}_v1`), its binding its port's. -/
def Class.listed : Class → Class
  | .input .coin => .input .run
  | c => c

/-- **Every gate is its primitive's** (what the census checks, refusing the circuit otherwise): its primitive is
listed, its operation is in the primitive's class (`Class.listed`), and a gate at a dot step (`Site.dot`) is an
s8-family step. Its role is then its primitive's and its statics' (`Gate.role`). -/
def Classed {ι : Type} (C : Dag ι) : Prop :=
  ∀ g ∈ C, classOf g.prim = some g.op.cls.listed ∧ ∀ K L tail s, g.site = .dot K L tail s → g.op.IsMma

end Accounting.Gates
