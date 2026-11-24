module

public import Definitions.Core.Structure

/-!
# A hardware model: the computations a machine can run (trusted)

A machine (`Machine`) is a set of nodes and the links between them. A node holds bits and may compute: a GPU's die, its
device memory, a host's processor, the world outside. A computation over `T` cycles (`Computation`) is a circuit with,
for every computed gate, the cycle and the node it is computed in, and for every cycle, the gates whose values each
node holds at its end and the gates whose values each link carries during the next. A value can be held at several
nodes. A cycle is one period of the machine's fastest clock, and a value of `w` bits counts `w` (`Circuit.width`).

* **Order** (`Computation.Order`). A gate reads only gates computed in its own cycle at its own node and gates its node
  held at the end of the cycle before. A node holds only input gates at the start; later it holds a gate only if it
  computed it in that cycle, held it at the end of the cycle before, or a link carried it there, and a link carries
  only what its source held.
* **Limits** (`Computation.Within`). Each node has functional units, circuits `unit i` with `count i` of each: the gates
  a node computes in a cycle are tiled by copies of its units (`Tiling`), at most `count i` of unit `i`, so they number
  at most `Node.rate`. A node holds at most `state` bits at the end of a cycle, and a link carries at most `link p q`
  bits a cycle.

`feasible M T` is the computations a machine `M` can run in `T` cycles. Three kinds of number bound three resources:
work (`rate`), memory (`state`) and bandwidth (`link`).
-/

@[expose] public section

namespace Verity.Hardware

open Finset

/-- **A copy of `K` in `C`** on the gates `G`: `φ` takes `K`'s computed gates one-to-one onto `G` and its input gates
outside `G`; each copied gate reads the images of what it reads in `K` and applies the same operation; and a copied gate
that a gate outside `G` reads is read by no gate of `K`, so only `K`'s outputs are used elsewhere. -/
structure Copy (K C : Circuit) (G : Finset (Fin C.N)) where
  φ : Fin K.N → Fin C.N
  inj : ∀ g h, K.isInput g = false → K.isInput h = false → φ g = φ h → g = h
  onto : K.computed.image φ = G
  computed : ∀ g, K.isInput g = false → C.isInput (φ g) = false
  inputs : ∀ a, K.isInput a = true → φ a ∉ G
  w : ∀ a, C.w (φ a) = K.w a
  reads : ∀ g, K.isInput g = false → C.reads (φ g) = (K.reads g).image φ
  op : ∀ g, K.isInput g = false → ∀ v : C.Trace,
    (C.op (φ g) v).cast (w g) = K.op g fun a => (v (φ a)).cast (w a)
  outputs : ∀ g h, K.isInput g = false → h ∉ G → φ g ∈ C.reads h → ∀ g', g ∉ K.reads g'

/-- **A node's limits**: its functional units, `count i` of the circuit `unit i`, and the bits it holds. -/
structure Node where
  r : ℕ
  unit : Fin r → Circuit
  count : Fin r → ℕ
  state : ℕ

/-- The gates a node computes in a cycle, at most: every unit's gates. -/
def Node.rate (n : Node) : ℕ := ∑ i, n.count i * (n.unit i).size (n.unit i).computed

/-- **A machine**: its nodes' limits and the bits each link carries in a cycle (`0` where there is no link). -/
structure Machine (ι : Type) where
  node : ι → Node
  link : ι → ι → ℕ

/-- **A tiling** of the gates `G` of `C` by copies of a node's units: disjoint copies that together are `G`, at most
`count i` of unit `i`. -/
structure Tiling (C : Circuit) (G : Finset (Fin C.N)) (n : Node) where
  tiles : Finset (Fin n.r × Finset (Fin C.N))
  copy : ∀ x ∈ tiles, Nonempty (Copy (n.unit x.1) C x.2)
  disjoint : ∀ x ∈ tiles, ∀ y ∈ tiles, x ≠ y → Disjoint x.2 y.2
  cover : tiles.biUnion Prod.snd = G
  count : ∀ i, (tiles.filter fun x => x.1 = i).card ≤ n.count i

/-- **A computation** over `T` cycles on nodes `ι`. -/
structure Computation (ι : Type) (T : ℕ) where
  C : Circuit
  /-- the cycle a computed gate is computed in -/
  cycle : Fin C.N → ℕ
  cycle_pos : ∀ g, C.isInput g = false → 1 ≤ cycle g
  cycle_le : ∀ g, C.isInput g = false → cycle g ≤ T
  /-- the node a computed gate is computed at -/
  place : Fin C.N → ι
  /-- the gates whose values a node holds at the end of a cycle -/
  hold : ℕ → ι → Finset (Fin C.N)
  /-- `sent c p q`: the gates whose values the link from `p` to `q` carries in cycle `c + 1` -/
  sent : ℕ → ι → ι → Finset (Fin C.N)

namespace Computation

variable {ι : Type} {T : ℕ} (X : Computation ι T)

open Classical in
/-- The gates node `p` computes in cycle `c`. -/
noncomputable def gatesAt (c : ℕ) (p : ι) : Finset (Fin X.C.N) := X.C.computed.filter fun g => X.cycle g = c ∧ X.place g = p

/-- **Order**: what a gate may read, and where a value may be. -/
structure Order : Prop where
  reads : ∀ g, X.C.isInput g = false → ∀ a ∈ X.C.reads g,
    (X.C.isInput a = false ∧ X.cycle a = X.cycle g ∧ X.place a = X.place g) ∨ a ∈ X.hold (X.cycle g - 1) (X.place g)
  start : ∀ p, ∀ a ∈ X.hold 0 p, X.C.isInput a = true
  kept : ∀ c p, ∀ a ∈ X.hold (c + 1) p,
    (X.C.isInput a = false ∧ X.cycle a = c + 1 ∧ X.place a = p) ∨ a ∈ X.hold c p ∨ ∃ q, a ∈ X.sent c q p
  sent : ∀ c p q, X.sent c p q ⊆ X.hold c p

/-- **Within the machine `M`**, at every cycle. -/
structure Within (M : Machine ι) : Prop where
  work : ∀ c p, Nonempty (Tiling X.C (X.gatesAt c p) (M.node p))
  state : ∀ c p, X.C.width (X.hold c p) ≤ (M.node p).state
  link : ∀ c p q, X.C.width (X.sent c p q) ≤ M.link p q

end Computation

/-- **The computations a machine can run in `T` cycles**: the ordered ones within its limits. -/
def feasible {ι : Type} (M : Machine ι) (T : ℕ) : Set (Computation ι T) := {X | X.Order ∧ X.Within M}

/-! ### What a proof of resource use establishes, and what is prohibited

`P C S` says that the gates `S` of `C` are a declared computation the policy permits, an inference circuit say. A proof
of resource use establishes that such an `S` uses at least `x` of a resource; a prohibited computation uses more than
`r` of that resource outside every such `S`. -/

/-- A permitted part of a circuit: a property of a set of its gates. -/
abbrev Permitted := (C : Circuit) → Finset (Fin C.N) → Prop

variable {ι : Type} {T : ℕ}

/-- **Work used**: a permitted part has at least `x` computed gates. -/
def usesWork (P : Permitted) (x : ℕ) : Set (Computation ι T) :=
  {X | ∃ S, P X.C S ∧ x ≤ X.C.size (X.C.computed ∩ S)}

/-- **Work prohibited**: more than `r` computed gates lie outside every permitted part. -/
def needsWork (P : Permitted) (r : ℕ) : Set (Computation ι T) :=
  {X | ∀ S, P X.C S → r < X.C.size (X.C.computed \ S)}

/-- **Memory used** at the nodes `A`: at the end of every cycle they hold at least `x` bits of a permitted part. -/
def usesMemory (P : Permitted) (A : Finset ι) (x : ℕ) : Set (Computation ι T) :=
  {X | ∃ S, P X.C S ∧ ∀ c ≤ T, x ≤ ∑ p ∈ A, X.C.width (X.hold c p ∩ S)}

/-- **Memory prohibited** at the nodes `A`: at the end of some cycle they hold more than `r` bits outside any permitted
part. -/
def needsMemory (P : Permitted) (A : Finset ι) (r : ℕ) : Set (Computation ι T) :=
  {X | ∀ S, P X.C S → ∃ c ≤ T, r < ∑ p ∈ A, X.C.width (X.hold c p \ S)}

/-- **Bandwidth used** on the link from `p` to `q`: over the `T` cycles it carries at least `x` bits of a permitted
part. -/
def usesBandwidth (P : Permitted) (p q : ι) (x : ℕ) : Set (Computation ι T) :=
  {X | ∃ S, P X.C S ∧ x ≤ ∑ c ∈ range T, X.C.width (X.sent c p q ∩ S)}

/-- **Bandwidth prohibited** on the link from `p` to `q`: over the `T` cycles it carries more than `r` bits outside any
permitted part. -/
def needsBandwidth (P : Permitted) (p q : ι) (r : ℕ) : Set (Computation ι T) :=
  {X | ∀ S, P X.C S → r < ∑ c ∈ range T, X.C.width (X.sent c p q \ S)}

end Verity.Hardware
