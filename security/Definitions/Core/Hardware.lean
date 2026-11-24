module

public import Definitions.Core.Structure
public import Mathlib.Tactic.DeriveFintype
public import Mathlib.Data.Fintype.Sum

/-!
# A hardware model: the computations a machine can run (trusted)

A machine (`Machine`) is a set of nodes and the links between them. A node holds bits and may compute: a GPU's die, its
device memory, a host's processor, the world outside. A computation over `T` cycles (`Computation`) is a circuit with,
for every computed gate, the cycle and the node it is computed in, and for every cycle, the gates whose values each
node holds at its end, the gates whose values each link carries during the next, and the copies of its units each node
runs. A value can be held at several
nodes. A cycle is one period of the machine's fastest clock, and a value of `w` bits counts `w` (`Circuit.width`).

* **Order** (`Computation.Order`). A gate reads only gates computed in its own cycle at its own node and gates its node
  held at the end of the cycle before. A node holds only input gates at the start; later it holds a gate only if it
  computed it in that cycle, held it at the end of the cycle before, or a link carried it there, and a link carries
  only what its source held.
* **Limits** (`Computation.Within`). Each node has functional units, circuits `unit i`, and resources: the gates a node
  computes in a cycle are tiled by disjoint copies of its units, and the copies use at most `cap j` of each resource
  `j`, a copy of unit `i` using `use j i`. A node holds at most `state` bits at the end of a cycle, and a link carries
  at most `link p q` bits a cycle.

`feasible M T` is the computations a machine `M` can run in `T` cycles. Three kinds of number bound three resources:
work (`cap`), memory (`state`) and bandwidth (`link`).
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

/-- **A node's limits**: its functional units, the circuits `unit i`; its resources, of which one copy of unit `i` uses
`use j i` of resource `j` and a cycle has `cap j`; and the bits it holds. With one resource per unit, used by that unit
alone, `cap` counts the copies of each unit; a resource several units use is one they share. -/
structure Node where
  r : ℕ
  unit : Fin r → Circuit
  m : ℕ
  use : Fin m → Fin r → ℕ
  cap : Fin m → ℕ
  state : ℕ

/-- A node computes at most `W` gates in a cycle: whatever numbers of copies of its units fit its resources have at
most `W` gates. -/
def Node.GatesWithin (n : Node) (W : ℕ) : Prop :=
  ∀ count : Fin n.r → ℕ, (∀ j, ∑ i, n.use j i * count i ≤ n.cap j) →
    ∑ i, count i * (n.unit i).size (n.unit i).computed ≤ W

/-- **A machine**: its nodes' limits and the bits each link carries in a cycle (`0` where there is no link). -/
structure Machine (ι : Type) where
  node : ι → Node
  link : ι → ι → ℕ

/-- **A computation** over `T` cycles on the machine `M`. -/
structure Computation {ι : Type} (M : Machine ι) (T : ℕ) where
  C : Circuit
  /-- the cycle a computed gate is computed in -/
  cycle : Fin C.N → ℕ
  cycle_pos : ∀ g, C.isInput g = false → 1 ≤ cycle g
  cycle_le : ∀ g, C.isInput g = false → cycle g ≤ T
  /-- the node a computed gate is computed at -/
  place : Fin C.N → ι
  /-- `tiles c p`: the copies of its units node `p` runs in cycle `c`, each a unit and the gates of its copy -/
  tiles : ℕ → (p : ι) → Finset (Fin (M.node p).r × Finset (Fin C.N))
  /-- the gates whose values a node holds at the end of a cycle -/
  hold : ℕ → ι → Finset (Fin C.N)
  /-- `sent c p q`: the gates whose values the link from `p` to `q` carries in cycle `c + 1` -/
  sent : ℕ → ι → ι → Finset (Fin C.N)

namespace Computation

variable {ι : Type} {M : Machine ι} {T : ℕ} (X : Computation M T)

open Classical in
/-- The gates node `p` computes in cycle `c`. -/
noncomputable def gatesAt (c : ℕ) (p : ι) : Finset (Fin X.C.N) :=
  X.C.computed.filter fun g => X.cycle g = c ∧ X.place g = p

/-- **Order**: what a gate may read, and where a value may be. -/
structure Order : Prop where
  reads : ∀ g, X.C.isInput g = false → ∀ a ∈ X.C.reads g,
    (X.C.isInput a = false ∧ X.cycle a = X.cycle g ∧ X.place a = X.place g) ∨ a ∈ X.hold (X.cycle g - 1) (X.place g)
  start : ∀ p, ∀ a ∈ X.hold 0 p, X.C.isInput a = true
  kept : ∀ c p, ∀ a ∈ X.hold (c + 1) p,
    (X.C.isInput a = false ∧ X.cycle a = c + 1 ∧ X.place a = p) ∨ a ∈ X.hold c p ∨ ∃ q, a ∈ X.sent c q p
  sent : ∀ c p q, X.sent c p q ⊆ X.hold c p

/-- **Within the machine's limits**, at every cycle and node: the gates computed are tiled by disjoint copies of the
node's units, which use at most each resource's `cap`; the node holds at most `state` bits; and each link carries at
most its capacity. -/
structure Within : Prop where
  copy : ∀ c p, ∀ x ∈ X.tiles c p, Nonempty (Copy ((M.node p).unit x.1) X.C x.2)
  disjoint : ∀ c p, ∀ x ∈ X.tiles c p, ∀ y ∈ X.tiles c p, x ≠ y → Disjoint x.2 y.2
  cover : ∀ c p, (X.tiles c p).biUnion Prod.snd = X.gatesAt c p
  work : ∀ c p j, ∑ x ∈ X.tiles c p, (M.node p).use j x.1 ≤ (M.node p).cap j
  state : ∀ c p, X.C.width (X.hold c p) ≤ (M.node p).state
  link : ∀ c p q, X.C.width (X.sent c p q) ≤ M.link p q

open Classical in
/-- What the copies inside the gates `S` use, over the `T` cycles, of the resources `R`, each a node and a resource of
it. -/
noncomputable def usedIn (R : Finset ((p : ι) × Fin (M.node p).m)) (S : Finset (Fin X.C.N)) : ℕ :=
  ∑ e ∈ R, ∑ c ∈ range T, ∑ x ∈ (X.tiles (c + 1) e.1).filter fun x => x.2 ⊆ S, (M.node e.1).use e.2 x.1

open Classical in
/-- What the copies not inside the gates `S` use, over the `T` cycles, of the resources `R`. -/
noncomputable def usedOut (R : Finset ((p : ι) × Fin (M.node p).m)) (S : Finset (Fin X.C.N)) : ℕ :=
  ∑ e ∈ R, ∑ c ∈ range T, ∑ x ∈ (X.tiles (c + 1) e.1).filter fun x => ¬ x.2 ⊆ S, (M.node e.1).use e.2 x.1

/-- The bits of the gates `S` that the links `E` carry over the `T` cycles. -/
def carried (E : Finset (ι × ι)) (S : Finset (Fin X.C.N)) : ℕ :=
  ∑ e ∈ E, ∑ c ∈ range T, X.C.width (X.sent c e.1 e.2 ∩ S)

/-- The bits the nodes `A` hold of the gates `S` at the end of cycle `c`. -/
def held (A : Finset ι) (c : ℕ) (S : Finset (Fin X.C.N)) : ℕ := ∑ p ∈ A, X.C.width (X.hold c p ∩ S)

end Computation

/-- **The computations a machine can run in `T` cycles**: the ordered ones within its limits. -/
def feasible {ι : Type} (M : Machine ι) (T : ℕ) : Set (Computation M T) := {X | X.Order ∧ X.Within}

/-! ### A server of GPUs -/

/-- **The nodes of a server of `n` GPUs**: each GPU's die and its device memory, the host, and the world outside. -/
inductive Loc (n : ℕ)
  | die (i : Fin n)
  | mem (i : Fin n)
  | host
  | outside
  deriving DecidableEq, Fintype

/-- **A server of `n` GPUs**: each die reads its device memory at `read` bits a cycle and writes it at `write`, exchanges
`bus` bits a cycle with the host each way and `peer` with each other die, and the host exchanges `net` with the world
outside. Device memory computes nothing, and the host is the side compute. -/
def server (n : ℕ) (die mem host outside : Node) (read write bus peer net : ℕ) : Machine (Loc n) where
  node
    | .die _ => die
    | .mem _ => mem
    | .host => host
    | .outside => outside
  link
    | .mem i, .die j => if i = j then read else 0
    | .die i, .mem j => if i = j then write else 0
    | .host, .die _ => bus
    | .die _, .host => bus
    | .die i, .die j => if i = j then 0 else peer
    | .outside, .host => net
    | .host, .outside => net
    | _, _ => 0

/-- The device memories of a server. -/
def mems (n : ℕ) : Finset (Loc n) := Finset.univ.image Loc.mem

/-- The links on which the dies read their device memories. -/
def reads (n : ℕ) : Finset (Loc n × Loc n) := Finset.univ.image fun i => (Loc.mem i, Loc.die i)

/-! ### What a proof of resource use establishes, and what is prohibited

`P C S` says that the gates `S` of `C` are a declared computation the policy permits, an inference circuit say. A proof
of resource use establishes that such an `S` uses at least `x` of a resource; a prohibited computation uses more than
`r` of that resource outside every such `S`. -/

/-- A permitted part of a circuit: a property of a set of its gates. -/
abbrev Permitted := (C : Circuit) → Finset (Fin C.N) → Prop

variable {ι : Type} {M : Machine ι} {T : ℕ}

/-- **Work used**: a permitted part has at least `x` computed gates. -/
def usesWork (P : Permitted) (x : ℕ) : Set (Computation M T) :=
  {X | ∃ S, P X.C S ∧ x ≤ X.C.size (X.C.computed ∩ S)}

/-- **Work prohibited**: more than `r` computed gates lie outside every permitted part. -/
def needsWork (P : Permitted) (r : ℕ) : Set (Computation M T) :=
  {X | ∀ S, P X.C S → r < X.C.size (X.C.computed \ S)}

/-- **Resources used**: copies inside a permitted part use at least `x` of the resources `R`. -/
def usesUnits (P : Permitted) (R : Finset ((p : ι) × Fin (M.node p).m)) (x : ℕ) : Set (Computation M T) :=
  {X | ∃ S, P X.C S ∧ x ≤ X.usedIn R S}

/-- **Resources prohibited**: copies not inside a permitted part use more than `r` of the resources `R`, whatever the
part. -/
def needsUnits (P : Permitted) (R : Finset ((p : ι) × Fin (M.node p).m)) (r : ℕ) : Set (Computation M T) :=
  {X | ∀ S, P X.C S → r < X.usedOut R S}

/-- **Memory used** at the nodes `A`: at the end of every cycle they hold at least `x` bits of a permitted part. -/
def usesMemory (P : Permitted) (A : Finset ι) (x : ℕ) : Set (Computation M T) :=
  {X | ∃ S, P X.C S ∧ ∀ c ≤ T, x ≤ X.held A c S}

/-- **Memory prohibited** at the nodes `A`: at the end of some cycle they hold more than `r` bits outside any permitted
part. -/
def needsMemory (P : Permitted) (A : Finset ι) (r : ℕ) : Set (Computation M T) :=
  {X | ∀ S, P X.C S → ∃ c ≤ T, r < X.held A c Sᶜ}

/-- **Bandwidth used** on the links `E`: over the `T` cycles they carry at least `x` bits of a permitted part. -/
def usesBandwidth (P : Permitted) (E : Finset (ι × ι)) (x : ℕ) : Set (Computation M T) :=
  {X | ∃ S, P X.C S ∧ x ≤ X.carried E S}

/-- **Bandwidth prohibited** on the links `E`: over the `T` cycles they carry more than `r` bits outside any permitted
part. -/
def needsBandwidth (P : Permitted) (E : Finset (ι × ι)) (r : ℕ) : Set (Computation M T) :=
  {X | ∀ S, P X.C S → r < X.carried E Sᶜ}

end Verity.Hardware
