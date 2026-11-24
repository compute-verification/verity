module

public import Definitions.Core.Structure
public import Mathlib.Tactic.DeriveFintype
public import Mathlib.Data.Fintype.Sum
public import Mathlib.Data.NNRat.Defs
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# A hardware model: the computations a machine can run (trusted)

A machine (`Machine`) is a set of nodes and the links between them. A node holds bits and may compute: a GPU's die, its
device memory, a host's processor, the world outside. A computation of a circuit over `T` cycles (`Computation`) is,
for every computed gate, the cycle and the node it is computed in and its kind (the functional unit that computes it),
and for every cycle, the gates whose values each node holds at its end and the gates whose values each link carries
during the next. A value can be held at several nodes. A cycle is one period of the machine's fastest clock, and a
value of `w` bits counts `w` (`Circuit.width`).

* **Order** (`Computation.Order`). A gate reads only gates computed in its own cycle at its own node and gates its node
  held at the end of the cycle before. A node holds only input gates at the start; later it holds a gate only if it
  computed it in that cycle, held it at the end of the cycle before, or a link carried it there, and a link carries
  only what its source held.
* **Limits** (`Computation.Within`). A node has resources, of which a gate of kind `i` uses `use i j` of resource `j`:
  the gates a node computes in a cycle use at most `cap j` of each. A node holds at most `state` bits at the end of a
  cycle, and a link carries at most `link p q` bits a cycle.

`feasible M C T` is the computations of `C` a machine `M` can run in `T` cycles. Three kinds of number bound three resources:
work (`cap`), memory (`state`) and bandwidth (`link`). A resource that one kind alone uses is that kind's rate; one
that several use is a rate they share, an issue slot say. The model does not say what a gate of a kind computes: a
statement that needs it, that a prohibited computation needs so many gates of a kind, says so itself.
-/

@[expose] public section

namespace Verity.Hardware

open Finset
open scoped NNRat

/-- **A node's limits**: its kinds of gate and its resources, a gate of kind `i` using `use i j` of resource `j`, of
which a cycle has `cap j`; and the bits it holds. -/
structure Node where
  Kind : Type
  Res : Type
  use : Kind → Res → ℚ≥0
  cap : Res → ℚ≥0
  state : ℕ

/-- **A machine**: its nodes' limits and the bits each link carries in a cycle (`0` where there is no link). -/
structure Machine (ι : Type) where
  node : ι → Node
  link : ι → ι → ℕ

/-- A resource of a machine: a node and a resource of it. -/
abbrev Machine.Res {ι : Type} (M : Machine ι) : Type := (p : ι) × (M.node p).Res

/-- **A computation** of the circuit `C` over `T` cycles on the machine `M`. -/
structure Computation {ι : Type} (M : Machine ι) (C : Circuit) (T : ℕ) where
  /-- the cycle a computed gate is computed in -/
  cycle : Fin C.N → ℕ
  cycle_pos : ∀ g, C.isInput g = false → 1 ≤ cycle g
  cycle_le : ∀ g, C.isInput g = false → cycle g ≤ T
  /-- the node a computed gate is computed at -/
  place : Fin C.N → ι
  /-- a computed gate's kind, among its node's -/
  kind : (g : Fin C.N) → (M.node (place g)).Kind
  /-- the gates whose values a node holds at the end of a cycle -/
  hold : ℕ → ι → Finset (Fin C.N)
  /-- `sent c p q`: the gates whose values the link from `p` to `q` carries in cycle `c + 1` -/
  sent : ℕ → ι → ι → Finset (Fin C.N)

namespace Computation

variable {ι : Type} {M : Machine ι} {C : Circuit} {T : ℕ} (X : Computation M C T)

open Classical in
/-- What the gate `g` uses of the resource `e`: its kind's use, at the resource's node, and nothing elsewhere. -/
noncomputable def cost (e : M.Res) (g : Fin C.N) : ℚ≥0 :=
  if h : X.place g = e.1 then (M.node e.1).use (h ▸ X.kind g) e.2 else 0

/-- The gates computed in cycle `c`. -/
def gatesAt (c : ℕ) : Finset (Fin C.N) := C.computed.filter fun g => X.cycle g = c

/-- What the gates `S` use of the resources `R`. -/
noncomputable def used (R : Finset M.Res) (S : Finset (Fin C.N)) : ℚ≥0 :=
  ∑ e ∈ R, ∑ g ∈ C.computed ∩ S, X.cost e g

/-- The bits of the gates `S` that the nodes `A` hold at the end of cycle `c`. -/
def held (A : Finset ι) (c : ℕ) (S : Finset (Fin C.N)) : ℕ := ∑ p ∈ A, C.width (X.hold c p ∩ S)

/-- The bits of the gates `S` that the links `E` carry over the `T` cycles. -/
def carried (E : Finset (ι × ι)) (S : Finset (Fin C.N)) : ℕ :=
  ∑ e ∈ E, ∑ c ∈ range T, C.width (X.sent c e.1 e.2 ∩ S)

/-- **The live values** at the end of cycle `c`: the inputs and the gates computed by then that a gate computed later
reads. -/
def live (c : ℕ) : Finset (Fin C.N) :=
  univ.filter fun a => (C.isInput a = false → X.cycle a ≤ c) ∧
    ∃ g, C.isInput g = false ∧ a ∈ C.reads g ∧ c < X.cycle g

open Classical in
/-- **The values that leave the nodes `U`**: the gates computed at a node of `U`, and the inputs no node outside `U`
holds at the start, that a gate computed at a node outside `U` reads. -/
noncomputable def leaving (U : Finset ι) : Finset (Fin C.N) :=
  univ.filter fun a => (C.isInput a = false → X.place a ∈ U) ∧ (C.isInput a = true → ∀ p ∉ U, a ∉ X.hold 0 p) ∧
    ∃ g, C.isInput g = false ∧ X.place g ∉ U ∧ a ∈ C.reads g

/-- **Order**: what a gate may read, and where a value may be. -/
structure Order : Prop where
  reads : ∀ g, C.isInput g = false → ∀ a ∈ C.reads g,
    (C.isInput a = false ∧ X.cycle a = X.cycle g ∧ X.place a = X.place g) ∨ a ∈ X.hold (X.cycle g - 1) (X.place g)
  start : ∀ p, ∀ a ∈ X.hold 0 p, C.isInput a = true
  kept : ∀ c p, ∀ a ∈ X.hold (c + 1) p,
    (C.isInput a = false ∧ X.cycle a = c + 1 ∧ X.place a = p) ∨ a ∈ X.hold c p ∨ ∃ q, a ∈ X.sent c q p
  sent : ∀ c p q, X.sent c p q ⊆ X.hold c p

/-- **Within the machine's limits**, at every cycle: the gates computed use at most each resource's `cap`, each node
holds at most `state` bits, and each link carries at most its capacity. -/
structure Within : Prop where
  work : ∀ c (e : M.Res), ∑ g ∈ X.gatesAt c, X.cost e g ≤ (M.node e.1).cap e.2
  state : ∀ c p, C.width (X.hold c p) ≤ (M.node p).state
  link : ∀ c p q, C.width (X.sent c p q) ≤ M.link p q

end Computation

/-- **The computations of `C` a machine can run in `T` cycles**: the ordered ones within its limits. -/
def feasible {ι : Type} (M : Machine ι) (C : Circuit) (T : ℕ) : Set (Computation M C T) := {X | X.Order ∧ X.Within}

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

variable {ι : Type} {M : Machine ι} {C : Circuit} {T : ℕ}

/-- **Work used**: a permitted part uses at least `x` of the resources `R`. -/
def usesWork (P : Permitted) (R : Finset M.Res) (x : ℚ≥0) : Set (Computation M C T) :=
  {X | ∃ S, P C S ∧ x ≤ X.used R S}

/-- **Work prohibited**: the gates outside a permitted part use more than `r` of the resources `R`, whatever the part. -/
def needsWork (P : Permitted) (R : Finset M.Res) (r : ℚ≥0) : Set (Computation M C T) :=
  {X | ∀ S, P C S → r < X.used R Sᶜ}

/-- **Memory used** at the nodes `A`: at the end of every cycle they hold at least `x` bits of a permitted part. -/
def usesMemory (P : Permitted) (A : Finset ι) (x : ℕ) : Set (Computation M C T) :=
  {X | ∃ S, P C S ∧ ∀ c ≤ T, x ≤ X.held A c S}

/-- **Memory prohibited** at the nodes `A`: at the end of some cycle they hold more than `r` bits outside any permitted
part. -/
def needsMemory (P : Permitted) (A : Finset ι) (r : ℕ) : Set (Computation M C T) :=
  {X | ∀ S, P C S → ∃ c ≤ T, r < X.held A c Sᶜ}

/-- **Bandwidth used** on the links `E`: over the `T` cycles they carry at least `x` bits of a permitted part. -/
def usesBandwidth (P : Permitted) (E : Finset (ι × ι)) (x : ℕ) : Set (Computation M C T) :=
  {X | ∃ S, P C S ∧ x ≤ X.carried E S}

/-- **Bandwidth prohibited** on the links `E`: over the `T` cycles they carry more than `r` bits outside any permitted
part. -/
def needsBandwidth (P : Permitted) (E : Finset (ι × ι)) (r : ℕ) : Set (Computation M C T) :=
  {X | ∀ S, P C S → r < X.carried E Sᶜ}

end Verity.Hardware
