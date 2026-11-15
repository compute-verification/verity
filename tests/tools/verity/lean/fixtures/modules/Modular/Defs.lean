module

/-! What the module controls' guarantee reads: `double`, whose body `@[expose]` exports to importers, and `triple`, whose
body only the private part of this module's `.olean` holds, as `triple_eq`'s proof is. The lock's records hash both
bodies. -/

namespace Modular

@[expose] public def double (n : Nat) : Nat := 2 * n

public def triple (n : Nat) : Nat := 3 * n

public theorem triple_eq (n : Nat) : triple n = 3 * n := by
  unfold triple
  omega

end Modular
