namespace Notation

syntax "dbl! " term : term

macro_rules
  | `(dbl! $x) => `($x + $x)

end Notation
