/-!
# Protocol 2, the private circuit: the universal unit's semantics

The public universal unit of a size class is `verity.ml.boolean.universal.UniversalUnit_v1{G, N_IN, N_OUT}` (#926), a
Boolean Definition that runs a hidden netlist given as a value. This file is its meaning, the padding that turns a
developer's circuit into a program of the class, and the netlists that circuit is made of.

* `Prog G nIn nOut`, the Definition's program arguments: per gate `op` (2 bits) and two `K`-bit source indices, per
  output a `K`-bit wire index, `K = idxBits G nIn`. `interp p x` is the Definition on `p` and the data bits `x`, gate for
  gate: wire `j < nIn` is `x j`, wire `nIn + g` is gate `g`'s output, each source and output a multiplexer over the wires
  before it that reads `0` past them (`read`, `gather` with fill 0), and gate `g`'s function `build`'s own
  `s ⊕ t·(a·b ⊕ s) ⊕ n` with `s = a ⊕ b` and op's bits `t` (bit 0), `n` (bit 1) (`gateFn`).
* `Netlist`, `interpret`: a developer's unit, a netlist of any size in SSA form, run by `universal.interpret`'s rule
  (op 0..3: XOR, AND, XNOR, NAND). A circuit (`Circ`) is a list of them, one per unit.
* `Cls`: the size class `(N, G, nIn, nOut)`, public; `Cls.Prog`, the program of a class, `N` universal units.
* `padUnit`: `universal.pad`, a netlist fitted to `(G, nIn)` with unread inputs after its own (its gate wires shifted past
  them) and no-op gates `(XOR, 0, 0)` after its gates; `dummyUnit`: `G` no-op gates and every output wire `nIn`, a no-op
  gate's, so it outputs `0` on every input; `pad C c`: `C`'s units padded, then dummies up to `N`.
* `fits C c`: `C` has at most `N` units, `G ≥ 1`, and each unit has at most `G` gates and `nIn` inputs, exactly `nOut`
  outputs, and every index in range (a source below its gate's own wire, an output below its wires), `pad`'s
  precondition.
* `run c p i x`: unit `i` of program `p` run by `U c` on unit inputs `x` (bits, the first `nIn` read); `eval C i x`: unit
  `i` of `C`; `dummy c`: the dummy's outputs, `nOut` zeros.
-/

namespace FlockSoundness.Discharge.PrivateCircuit.Universal

/-- **The bits of a wire index**, `K = max(1, bit_length(W - 1))` over the `W = nIn + G` wires (`index_bits`). -/
def idxBits (G nIn : Nat) : Nat := Nat.log2 (nIn + G - 1) + 1

/-- **A gate's function**, as `build` computes it: `s ⊕ t·(a·b ⊕ s) ⊕ n` with `s = a ⊕ b`, where `t` is op's bit 0 (AND
over XOR) and `n` its bit 1 (negate). -/
def gateFn (t n a b : Bool) : Bool := xor (xor (xor a b) (t && xor (a && b) (xor a b))) n

/-- **A multiplexer over the wires so far**: wire `i`, or `0` past them (`gather` with fill 0). -/
def read (w : List Bool) (i : Nat) : Bool := w.getD i false

/-- **A universal gate**: `op`'s two bits and the two `K`-bit source indices, LSB first, as the Definition's leaves. -/
structure Gate (K : Nat) where
  op : BitVec 2
  a : BitVec K
  b : BitVec K

/-- A gate's output, read from the wires before it. -/
def Gate.eval {K : Nat} (g : Gate K) (w : List Bool) : Bool :=
  gateFn (g.op.getLsbD 0) (g.op.getLsbD 1) (read w g.a.toNat) (read w g.b.toNat)

/-- **The wires**: the inputs `x`, then each gate's output, computed from the wires before it. -/
def wires {K : Nat} (gs : List (Gate K)) (x : List Bool) : List Bool := gs.foldl (fun w g => w ++ [g.eval w]) x

/-- **A program of `UniversalUnit_v1{G, nIn, nOut}`**, its arguments `op`, `src` (as `gates`) and `out`. -/
structure Prog (G nIn nOut : Nat) where
  gates : Fin G → Gate (idxBits G nIn)
  out : Fin nOut → BitVec (idxBits G nIn)

/-- **The universal unit**, `UniversalUnit_v1{G, nIn, nOut}` on program `p` and data `x`: output `i` is wire
`out i`. -/
def interp {G nIn nOut : Nat} (p : Prog G nIn nOut) (x : Fin nIn → Bool) : Fin nOut → Bool :=
  fun i => read (wires (List.ofFn p.gates) (List.ofFn x)) (p.out i).toNat

/-- **A netlist gate** (`universal.Netlist`'s `(op, a, b)`): op 0 XOR, 1 AND, 2 XNOR, 3 NAND, of wires `a` and `b`. -/
structure NGate where
  op : Fin 4
  a : Nat
  b : Nat

/-- `interpret`'s gate rule: `(a & b if op & 1 else a ^ b) ^ (op >> 1)`. -/
def NGate.eval (g : NGate) (w : List Bool) : Bool :=
  xor (if g.op.val % 2 = 1 then read w g.a && read w g.b else xor (read w g.a) (read w g.b)) (g.op.val / 2 = 1)

/-- **A developer's unit**: a netlist over `nIn` inputs, wire `nIn + g` gate `g`'s output, output `i` wire
`outputs[i]`. -/
structure Netlist where
  nIn : Nat
  gates : List NGate
  outputs : List Nat

/-- The netlist's wires on the inputs `x`. -/
def Netlist.wires (gs : List NGate) (x : List Bool) : List Bool := gs.foldl (fun w g => w ++ [g.eval w]) x

/-- **`universal.interpret`**: the netlist run gate by gate on the first `nIn` bits of `x`. -/
def interpret (u : Netlist) (x : Nat → Bool) : List Bool :=
  u.outputs.map (read (Netlist.wires u.gates (List.ofFn fun j : Fin u.nIn => x j)))

/-- **A size class**: `N` universal units of `G` gates over `nIn` inputs with `nOut` outputs. It is public. -/
structure Cls where
  N : Nat
  G : Nat
  nIn : Nat
  nOut : Nat

/-- The class's unit, `UniversalUnit_v1{G, nIn, nOut}`'s program. -/
abbrev Cls.Unit (c : Cls) : Type := Prog c.G c.nIn c.nOut

/-- The class's program: one universal unit's program per unit. -/
abbrev Cls.Prog (c : Cls) : Type := Fin c.N → c.Unit

/-- **A hidden circuit**: its units, each a netlist. -/
abbrev Circ : Type := List Netlist

/-- **The no-op gate** `(XOR, 0, 0)`: the constant 0, which nothing reads. -/
def noop {K : Nat} : Gate K := ⟨0, 0, 0⟩

/-- `pad`'s wire map: a netlist's input `i < nIn` stays, its gate wire `i ≥ nIn` moves past the class's unread
inputs. -/
def shiftWire (u : Netlist) (c : Cls) (i : Nat) : Nat := if i < u.nIn then i else i + (c.nIn - u.nIn)

/-- A netlist gate at the class, its wires shifted. -/
def encGate (u : Netlist) (c : Cls) (g : NGate) : Gate (idxBits c.G c.nIn) :=
  ⟨BitVec.ofNat 2 g.op.val, BitVec.ofNat _ (shiftWire u c g.a), BitVec.ofNat _ (shiftWire u c g.b)⟩

/-- **`universal.pad`**: the netlist `u` in the class's `G` gates over `nIn` inputs, its gate wires shifted past the
unread inputs and no-op gates after its own. -/
def padUnit (c : Cls) (u : Netlist) : c.Unit where
  gates g := if h : g.val < u.gates.length then encGate u c u.gates[g.val] else noop
  out i := BitVec.ofNat _ (shiftWire u c (u.outputs.getD i.val 0))

/-- **The dummy unit**: `G` no-op gates, every output wire `nIn` (the first no-op gate's), so `0` on every input. -/
def dummyUnit (c : Cls) : c.Unit where
  gates _ := noop
  out _ := BitVec.ofNat _ c.nIn

/-- **The padded program**: `C`'s units padded to the class, then dummy units up to `N`. -/
def pad (C : Circ) (c : Cls) : c.Prog := fun i => if h : i.val < C.length then padUnit c C[i.val] else dummyUnit c

/-- Every index of `u` in range: a source below its gate's own wire, an output below the netlist's wires. -/
def Netlist.InRange (u : Netlist) : Prop :=
  (∀ g (h : g < u.gates.length), u.gates[g].a < u.nIn + g ∧ u.gates[g].b < u.nIn + g) ∧
    ∀ o ∈ u.outputs, o < u.nIn + u.gates.length

/-- **A unit fits the class**: at most `G` gates and `nIn` inputs, exactly `nOut` outputs, every index in range. -/
def Netlist.Fits (u : Netlist) (c : Cls) : Prop :=
  u.gates.length ≤ c.G ∧ u.nIn ≤ c.nIn ∧ u.outputs.length = c.nOut ∧ u.InRange

/-- **A circuit fits the class**: at most `N` units, each fitting, and `G ≥ 1` (the dummy reads a no-op gate). -/
def fits (C : Circ) (c : Cls) : Prop := 1 ≤ c.G ∧ C.length ≤ c.N ∧ ∀ u ∈ C, u.Fits c

/-- The class's unit count. -/
def units (c : Cls) : Nat := c.N

/-- The circuit's unit count. -/
def size (C : Circ) : Nat := C.length

/-- **Unit `i` of program `p`, run by `U c`** on the unit's input bits `x` (the first `nIn` read); nothing past `N`. -/
def run (c : Cls) (p : c.Prog) (i : Nat) (x : Nat → Bool) : List Bool :=
  if h : i < c.N then List.ofFn (interp (p ⟨i, h⟩) fun j => x j.val) else []

/-- **Unit `i` of the circuit `C`** on its input bits `x`; nothing past its units. -/
def eval (C : Circ) (i : Nat) (x : Nat → Bool) : List Bool :=
  match C[i]? with
  | some u => interpret u x
  | none => []

/-- **The dummy's outputs**: `nOut` zeros. -/
def dummy (c : Cls) : List Bool := List.replicate c.nOut false

end FlockSoundness.Discharge.PrivateCircuit.Universal
