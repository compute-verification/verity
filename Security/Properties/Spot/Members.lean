module

public import Properties.Spot.Bits

/-!
# An audit's members side by side, as the claims' program (`Spot.Bits.Members`)

`Bits.Members` lists an audit's members, each a circuit file over some of the partition's units, their programs side by
side, member `m`'s positions from `Bits.base ms m`. This is that program as the claims read it: the program of a
statement over several templates' circuit files, each template's units lowered from its own file.

* `shiftGate b x`: the claims' gate `x`, every gate it reads `b` later.
* `sideBySide ps`: programs side by side, each program's gates after the earlier programs', every gate it reads and
  every output it returns `offsetOf ps m` later, the earlier programs' gates.
* `memberProg c n`: a member's program, the circuit file `c`'s key program for `n` units lowered at its constants
  (`bitsProg` at `Bits.keyProg c n` and `Bits.consts c n`).
* `bitsProgs ms`: the members' programs side by side.
* `unitBase`, `memberBlocks`, `membersPart ms`: the members' proof blocks, one class whose stratum `m` is member `m`'s
  units' blocks (`Bits.blockOf`) at its positions, member `m`'s unit `u` slot `unitBase ms m + u`.
* `membersConsts ms`: each member's constants (`Bits.consts`) at its positions.

`Proofs.Flock.Bridge.Members` proves that member `m` sits at `Bits.base ms m`, that no member's gate reads another's,
that each unit of a member is lowered there as Bridge's lowering of the member's circuit lowers it, and where each
member's proof blocks and constants sit.
-/

@[expose] public section

namespace Properties.Spot

open _root_.Spot

/-- **The claims' gate `x`, every gate it reads `b` later.** -/
def shiftGate (b : ℕ) : Claims.Sampling.BGate → Claims.Sampling.BGate
  | .input => .input
  | .const c => .const c
  | .not i => .not (b + i)
  | .and ws => .and (ws.map (b + ·))
  | .xor ws => .xor (ws.map (b + ·))

/-- **Programs side by side**: the first program's gates and outputs, then the others' side by side, every gate they read
and every output they return the first program's gates later. -/
def sideBySide : List Claims.Sampling.Prog → Claims.Sampling.Prog
  | [] => ⟨[], []⟩
  | p :: ps =>
    ⟨p.gates ++ (sideBySide ps).gates.map (shiftGate p.gates.length),
      p.outs ++ (sideBySide ps).outs.map (p.gates.length + ·)⟩

/-- **Program `m`'s first position** side by side: the gates of the programs before it. -/
def offsetOf (ps : List Claims.Sampling.Prog) (m : ℕ) : ℕ := ((ps.take m).map (·.gates.length)).sum

/-- **A member's program**: the circuit file `c`'s key program for `n` units, lowered at its constants. -/
def memberProg (c : _root_.Flock.Circuit) (n : ℕ) : Claims.Sampling.Prog :=
  bitsProg (Bits.keyProg c n) (Bits.consts c n)

/-- **The members' programs side by side**, member `m` from `Bits.base ms m`. -/
def bitsProgs (ms : Bits.Members) : Claims.Sampling.Prog :=
  sideBySide (ms.map fun ci => memberProg ci.1 ci.2.size)

/-- **Member `m`'s first slot**: the units of the members before it. -/
def unitBase (ms : Bits.Members) (m : ℕ) : ℕ := ((ms.take m).map (·.2.size)).sum

/-- **Member `m`'s proof blocks** in `bitsProgs ms`, in unit order: each unit's (`Bits.blockOf`), `Bits.base ms m`
later; none past the members. -/
def memberBlocks (ms : Bits.Members) (m : ℕ) : List (List ℕ) :=
  match ms[m]? with
  | some ci => (List.range ci.2.size).map fun u =>
      (Bits.blockOf (Bits.keyProg ci.1 ci.2.size) (Bits.consts ci.1 ci.2.size) u).map (Bits.base ms m + ·)
  | none => []

/-- **The members' proof-block partition**: one class, whose stratum `m` is member `m`'s proof blocks, so member `m`'s
unit `u` is slot `unitBase ms m + u`. -/
def membersPart (ms : Bits.Members) : Claims.Sampling.Partition :=
  ⟨[(List.range ms.length).map (memberBlocks ms)]⟩

/-- **The members' constants**: member `m`'s (`Bits.consts`) at its positions, `Bits.base ms m` later; none past the
members. A key program's inputs come first, one lowered gate each, so `Bits.consts` reads a lowered position as the
input gate there. -/
def membersConsts (ms : Bits.Members) (j : ℕ) : Option Bool :=
  match (List.range ms.length).find? (fun m => decide (j < Bits.base ms (m + 1))) with
  | some m => (ms[m]?).bind fun ci => Bits.consts ci.1 ci.2.size (j - Bits.base ms m)
  | none => none

end Properties.Spot
