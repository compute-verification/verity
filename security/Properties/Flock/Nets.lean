module

public import Flock.HmNets
public import VStar.Unit

/-! The verifier's two fixed nets build. -/

@[expose] public section

open Flock Flock.HmNets

namespace Properties.Flock.Nets

/-- The net `Net.ofRows` builds at these header fields, with A `a` and B `b`: one input and one output group, the
constant at the last useful column, a slot of `2 ^ 18` columns and no lookup. -/
def built (name : String) (useful : Nat) (ig og : Group) (inWords : Nat) (a b : Sparse) : Net :=
  { name, useful, constPos := useful - 1, inGroups := #[ig], outGroups := #[og], inWords, aT := .pure a, bT := .pure b,
    unitLog := 18 }

/-- The sparse matrix `Net.ofRows` makes of rows `r` at a slot of `2 ^ 18` columns, when it makes one. -/
def sparseOf (r : Array (Array Nat)) : Sparse := (Sparse.ofRows r (2 ^ 18)).toOption.getD default

/-- Each line of net `name`'s text from layout `l` is under `2 ^ 20` bytes: its header and every useful row's. -/
def LinesFit (name : String) (l : Laid) : Prop :=
  (VStar.header name l).utf8ByteSize < 2 ^ 20 ∧
    ∀ i < l.useful, (VStar.rowLine (l.ra.getD i #[]) (l.rb.getD i #[])).utf8ByteSize < 2 ^ 20

/-- **Net `name` builds from layout `l`** at these header fields, with A and B made of `l`'s rows, `l` has an A row
and a B row per useful column, and each line of its text is under `2 ^ 20` bytes. -/
def Builds (name : String) (l : Laid) (useful : Nat) (ig og : Group) (inWords : Nat) : Prop :=
  l.ra.size = useful ∧ l.rb.size = useful ∧
    netOf name l = .ok (built name useful ig og inWords (sparseOf l.ra) (sparseOf l.rb)) ∧ LinesFit name l

/-- **The verifier's `sha512x3` and `hm96` nets build**, at the header fields every statement's circuit file carries:
`sha512x3` with 256001 useful columns, its six input ports in 64 words from column 0 and its three outputs in 16 words
from word 1984; `hm96` with 246785, its three inputs in 32 words and its output in 8 words from word 1920. -/
def FixedNetsBuild : Prop :=
  Builds "sha512x3" (shaLaid ()) 256001 ⟨0, 64, SHA_IN⟩ ⟨1984, 16, SHA_OUT⟩ 64 ∧
    Builds "hm96" (hmLaid ()) 246785 ⟨0, 32, HM_IN⟩ ⟨1920, 8, HM_OUT⟩ 32

end Properties.Flock.Nets
