module

public import Flock.HmNets
public import VStar.Unit

/-!
# The verifier's two fixed nets build

Every statement with hm96 rows carries the verifier's own `sha512x3` and `hm96` nets in its circuit file, and
`HmNets.check` holds the file's to the ones the verifier builds from its own builder: `HmNets.netOf "sha512x3"
(shaLaid ())`, three SHA-512 compressions, and `HmNets.netOf "hm96" (hmLaid ())`, the `hm96-sha512/row/v1` row commitment
under the default key. `FixedNetsBuild` says that both builds succeed, with every header field a numeral and A and B the
sparse matrices of the layout's rows (`sparseOf`, a closed term), that each layout has one A row and one B row per
useful column, and that each line of the text a circuit file carries of the net (`VStar.header`, `VStar.rowLine`) is
under `2 ^ 20` bytes, so that a statement carrying both fits the lengths its encoding writes.

It is a fact about the verifier's code alone, the same for every statement. The compiled verifier checks the builds on
every run: `HmNets.check` builds both nets, and a build that failed would refuse every statement. The kernel can't check
it yet: `HmNets.symm`, the support merge every XOR of the builder runs, is defined by well-founded recursion, which the
kernel doesn't reduce, so neither layout evaluates. A structurally recursive `symm`, proved equal to the deployed one,
lets the kernel build both layouts, their nets and their lines, and discharges it. `Proofs.Properties.Flock.Nets` holds
its proof.
-/

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
