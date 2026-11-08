module

public import Properties.SampledProofs.Bits

/-!
# An audit's members side by side, as the claims' program (`SampledProofs.Bits.Members`)

`Bits.Members` lists an audit's members, each a circuit file over some of the partition's units, their programs side by
side, member `m`'s positions from `Bits.base ms m`. This is that program as the claims read it: the program of a
statement over several templates' circuit files, each template's units lowered from its own file.

* `shiftGate b x`: the claims' gate `x`, every gate it reads `b` later.
* `sideBySide ps`: programs side by side, each program's gates after the earlier programs', every gate it reads and
  every output it returns `offsetOf ps m` later, the earlier programs' gates.
* `memberProg c n`: a member's program, the circuit file `c`'s key program for `n` units lowered at its constants
  (`bitsProg` at `Bits.keyProg c n` and `Bits.consts c n`).
* `bitsProgs ms`: the members' programs side by side.

`Proofs.Flock.Bridge.Members` proves that member `m` sits at `Bits.base ms m`, that no member's gate reads another's, and
that each unit of a member is lowered there as Bridge's lowering of the member's circuit lowers it.
-/

@[expose] public section

namespace Properties.SampledProofs

open _root_.SampledProofs

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

end Properties.SampledProofs
