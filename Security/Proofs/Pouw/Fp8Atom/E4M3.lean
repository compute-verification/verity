import Definitions.Pouw.Fp8Atom.E4M3

/-! The proof-only part of `Definitions.Pouw.Fp8Atom.E4M3`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Fp8Atom

/-- A zero: `0x00` or `0x80`. -/
def Code.isZero (c : Code) : Bool := c.val % 128 == 0

end Pouw.Fp8Atom
