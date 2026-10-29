import Definitions.Pouw.Assumptions.Dimension.H100W
import Definitions.Hash.Tree
import Definitions.Hash.Hm96

/-!
# The lead's commitment premises the code can't be checked against yet (trusted; `Prop`s taken as hypotheses)

Three of the commitment premises the lead's guarantee reads are about what a drawn unit's proof checks, or what the
card's programs write, which no Lean function computes yet. Each is a named `Prop` whose docstring opens with its claim
id. The other two commitment premises are properties of the code, stated in `Properties.Pouw.Lead.Commit`: row-bound
(`LeafFirstBlock`, `RowBound`) and g-private's order (`RoundsOpenAfterRoots`). That `G` is uniform and hidden until the
certifier streams it is `trusted-host` at the network certifier.

* `SubtreeLayout`: every row a drawn unit opens climbs, under its instance's own frame-v3 domain, to an inner digest
  that the instance's registered root hides as an hm96 leaf.
* `SaltDerivation`: every salt a drawn unit's proof uses derives from a key whose hm96 leaf is committed before the
  window's salt, and from the window's salt.
* `CarriedWords`: every admissible program that outputs the committed words' digests writes a carrier of each word.

A drawn unit's proof is the protocol's (`Opens`): what it establishes is a parameter, so a statement reads these at its
own acceptance relation.
-/

namespace Pouw.Dimension.H100

variable {G : Type}

/-- **`r` carries `t` on `S`**: it equals `t` plus a constant there (a `wgmma` with a constant accumulator). -/
def Carries (S : Set G) (t r : G → ℚ) : Prop := ∃ q : ℚ, ∀ g ∈ S, r g = t g + q

/-- The program writes, for every target, a register in `Rel` to it. -/
def WritesBy (Rel : (G → ℚ) → (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (P : List (Op100 G))
    (L : List (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∃ r ∈ regs100 fpSem P L, Rel t r

end Pouw.Dimension.H100

namespace Pouw.Assumptions

open Hash Pouw.Dimension.H100

/-- `pouw-subtree-layout`. **Each hm96 instance hides one frame-v3 subtree under its own domain**: whenever
a drawn unit's accepted proof opens the value `v` at `rank` of instance `pos` through the inner digest `x` and the salt
`y` (`Opens pos rank v x y`), the instance's registered root `root pos` is the hm96 leaf over `x` and `y` under the
verifier's key, and `v`'s frame-v3 leaf at `rank` climbs to `x` under the instance's domain `dom pos`. An instance is a
call's `A` rows (its root the one the call's coins are drawn after) or a forward's words.

Rationale: the layout L1. A call's `A` rows are the leaves of their own subtree, in the call's domain, whose root one
hm96 instance hides, and a drawn unit's proof climbs its rows to that root and checks the instance's commit string
against the registered leaf. The commitment code has no per-instance domain, no hm96 instance over a subtree root and
no seed or record reading one, so this is a premise of the protocol until it does. With row-bound, an opened row is
bound to its instance and rank, or the checks hashed a collision (`Hash.Guarantees.TreeBinding`). -/
def SubtreeLayout (H : Msg → Digest) (key : Msg) (dom : ℕ → Tree.Domain) (schema : String) (root : ℕ → Digest)
    (Opens : ℕ → ℕ → Msg → Digest → Msg → Prop) : Prop :=
  ∀ pos rank v x y, Opens pos rank v x y →
    root pos = Hm96.leaf H key x y ∧
      ∃ sibs, Tree.accepts H (dom pos) x rank (H (Tree.leafMsg (dom pos) rank schema v)) sibs

/-- `pouw-salt-derivation`. **Every salt derives from a key committed before the window's salt**: whenever a
drawn unit's accepted proof opens instance `pos` through a salt `y` (`Opens pos rank v x y`), it opens a key `ks`
against the key's committed hm96 leaf `keyLeaf` (under some salt `y₀`), and `y` is `F₂ (F₁ ks salt) pos`: the window's
key `k_w = F₁ ks salt` and the instance's salt from `k_w` and `pos`. `keyLeaf` is fixed before the window's salt `salt`
is drawn, so no salt is chosen after a coin.

Rationale: the derivation (a). The deployed maps are `k_w = SHA-512(ks ‖ salt)` and `y = SHA-512(k_w ‖ pos ‖ r)` for
`r = 0, 1, 2`, or `verity.ml.randomness`'s `derive` and `stream` under the same key. A drawn unit's proof opens `ks`
against `keyLeaf` and recomputes `k_w` and `y`, so ShaWrites credits those compressions as forced work. The hiding
hybrid (`Pouw.SecurityProofs.Lead.Commit.hides_derived_os`) reads the derivation as the honest committer's; soundness
and the credit read it as the proof's check. The commitment code's hm96 salts are its own stream under the committer's
key, which no proof checks, so this is a premise of the protocol until it does. -/
def SaltDerivation (H : Msg → Digest) (key : Msg) (keyLeaf : Digest) (salt : Msg) (F₁ : Digest → Msg → Msg)
    (F₂ : Msg → ℕ → Msg) (Opens : ℕ → ℕ → Msg → Digest → Msg → Prop) : Prop :=
  ∀ pos rank v x y, Opens pos rank v x y → ∃ ks y₀, keyLeaf = Hm96.leaf H key ks y₀ ∧ y = F₂ (F₁ ks salt) pos

/-- `pouw-carried-words`. **Every admissible program that outputs the digests writes a carrier of each
committed word**: on the grid `S`, every program in `Adm` on inputs `L` whose registers include the digest words `Dig`
writes, for each committed word `t ∈ T`, a register equal to `t` plus a constant on `S` (`Carries`).

Rationale: SHA-512 reads the committed rows as message words, and on the card a program reaches a committed word's
value only through the register its tensor-core or ALU instruction writes, before any byte swap or conversion the
hash's schedule then reads. Over every program of `MH100w` it is false, since an instruction there may write any
function (an encoding of a word in place of the word), so it is a premise about the card's programs `Adm`, as
`ShaWrites` is. With `ShaWrites`, only the words' carriers leave the hash's count. -/
def CarriedWords {G : Type} (Adm : List (Op100 G) → List (G → ℚ) → Prop) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G)
    (T Dig : Finset (G → ℚ)) : Prop :=
  ∀ (L : List (G → ℚ)) (P : List (Op100 G)), Adm P L → OutputsOn fpSem S P L Dig → WritesBy (Carries S) fpSem P L T

end Pouw.Assumptions
