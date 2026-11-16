module

public import NetworkCheck.Period
public import Definitions.Certifier.SampledAudit
public import Properties.Spot.Members

/-!
# The scan's layout at the members SPOT[Flock]'s statement puts side by side

SPOT[Flock]'s statement is its members' programs side by side (`Properties.Spot.bitsProgs`, `SpotFlock.parse`'s
`prog`): member `m` is template `m`'s circuit file lowered over its units, from `Spot.Bits.base ms m`. The scan's
templates are its root's template nodes in order (`NetworkCheck.Period.templateNodes`): the start, the ingress and the
egress, members 0, 1 and 2.

**The layout** (`membersLayout`): at step `t`, slot `σ`'s presence gate is bit 0 of the egress member's unit `t`'s
returned leaf 0, and its frame's bits are its leaves `16 + 8σB … 16 + 8(σ + 1)B − 1` (`outAt`, each returned field on
16-bit words of its own, leaf `ℓ` bit `ℓ % 16` of word `ℓ / 16`, `Spot.Bits.outBit`); step `t`'s ingress is the
ingress member's unit `t`'s first parameter, the bucket's row, in bit order (`inAt`, on the parameter's 16-bit words,
`Spot.Bits.inBit`). A position the file doesn't have would read the member's first gate, its constant; the restated
scope names none, as it takes that every position it names is defined, and every ingress position one a unit reads
(`NamesDefined`), which the auditor decides of its held members and the schedule.

The fit is at the parsed statement's gates, `s.prog`, the members side by side and unconnected, where a member's gates
read only its own. It is not a fit at the connected program (`Prog.wired`), where a unit reads a source bit at its
writer's output bit, a gate of another member.

**The parse's members** (`StmtMembers`): each statement the compiled sampled verifier parses has a descriptor Flock
decodes, its gates are the members' programs side by side, and the members are the descriptor's templates (`MembersOf`):
three of them, each laid out as its template (`Spot.Bits.check`), over at least as many units as its template node has
members.

* `ScanScopeAccMembers`: given an accepted scan, at every compiled sampled verifier whose statements are their members
  side by side, a statement in its sampling scope, whose descriptor is the scan's canonical bytes and whose members
  name every position the layout reads, is in the sampled audit's scope at the members' layout and the window `T`
  (`NetTiming.ScopeAcc`).
-/

@[expose] public section

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- **Member `m`'s unit `t`'s returned leaf `leaf`**: bit `leaf % 16` of output word `leaf / 16` of its one template
instance (`Spot.Bits.outBit`), at the member's base; the member's first gate where the file has no such bit. -/
def outAt (ms : Spot.Bits.Members) (m t leaf : ℕ) : ℕ :=
  let ci := ms.getD m default
  Spot.Bits.base ms m + (Spot.Bits.outBit ci.1 ci.2.size t 0 (leaf / 16) (leaf % 16)).getD 0

/-- **Member `m`'s unit `t`'s first parameter's bit `j`**: bit `j % 16` of its input word `j / 16`
(`Spot.Bits.inBit`), at the member's base; the member's first gate where the file has no such bit. -/
def inAt (ms : Spot.Bits.Members) (m t j : ℕ) : ℕ :=
  let ci := ms.getD m default
  Spot.Bits.base ms m + (Spot.Bits.inBit ci.1 ci.2.size t 0 (j / 16) (j % 16)).getD 0

/-- **The scan's layout at the members** `ms` at the schedule `s`, over one egress link: step `t`'s slot `σ` is the
egress member's (member 2's) unit `t`'s leaf 0 and its frame's `8B` leaves, and step `t`'s ingress is the ingress
member's (member 1's) unit `t`'s row. -/
def membersLayout (ms : Spot.Bits.Members) (s : Schedule) : NetTiming.BoundaryGates 1 s.allocation :=
  let bits := 8 * s.cellBytes
  { egress := fun t _ σ => (outAt ms 2 t 0, (List.range bits).map fun b => outAt ms 2 t (outBits + σ.val * bits + b))
    ingress := fun t => (List.range (s.allocation * bits)).map fun j => inAt ms 1 t j }

/-- **The members name every position the layout reads** at the schedule `s`: at each step `t` of the window, the
egress member's unit `t` returns each egress leaf (`Spot.Bits.outBit` is `some`), and the ingress member's unit `t` has
each bit of the bucket's row as an input bit (`Spot.Bits.inBit` is `some`) that a unit reads (`Spot.Bits.owner`). -/
def NamesDefined (ms : Spot.Bits.Members) (s : Schedule) : Prop :=
  ∀ t < s.window,
    (∀ leaf < outBits + s.allocation * (8 * s.cellBytes), ∃ x,
      Spot.Bits.outBit (ms.getD 2 default).1 (ms.getD 2 default).2.size t 0 (leaf / 16) (leaf % 16) = some x) ∧
    ∀ j < s.allocation * (8 * s.cellBytes), ∃ x,
      Spot.Bits.inBit (ms.getD 1 default).1 (ms.getD 1 default).2.size t 0 (j / 16) (j % 16) = some x ∧
        Spot.Bits.owner (Spot.Bits.keyProg (ms.getD 1 default).1 (ms.getD 1 default).2.size) x ≠ none

/-- The root node of the Program `P`'s `m`-th template (`NetworkCheck.Period.templateNodes`). -/
def templateNode (P : Flock.Program.Prog) (m : ℕ) : Flock.Program.Node :=
  (P.defs[P.root]!).body.nodes[(templateNodes P).getD m 0]!

/-- **The members are the Program's templates**: three, member `m` laid out as the definition template `m`'s root node
calls (`Spot.Bits.check`), over at least as many units as that node has members. -/
def MembersOf (P : Flock.Program.Prog) (ms : Spot.Bits.Members) : Prop :=
  ms.length = 3 ∧ ∀ m < 3,
    Spot.Bits.check (ms.getD m default).1 P (P.defs[(templateNode P m).fn]!).key = none ∧
      (templateNode P m).members ≤ (ms.getD m default).2.size

/-- **The compiled verifier's statements are their members side by side**: every statement `D` parses has a descriptor
Flock decodes, its gates are the members `msOf` of it side by side (`Properties.Spot.bitsProgs`), and those members are
the descriptor's templates (`MembersOf`). -/
def StmtMembers {Coin : Type} {A : Claims.SamplingGeneric.Arg Coin} (D : Claims.SamplingGeneric.Compiled Coin A)
    (msOf : Claims.Sampling.AuditStatement → Spot.Bits.Members) : Prop :=
  ∀ x s, D.parse x = some s → ∃ P, Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧
    s.prog = Properties.Spot.bitsProgs (msOf s) ∧ MembersOf P (msOf s)

/-- **An accepted scan's statements are in the sampled audit's scope, at their members.** If `scanOf` accepts the scan
`j` at the schedule `s` and its canonical text is `c`, then at every compiled sampled verifier `D` whose statements are
their members `msOf` side by side (`StmtMembers`), a statement `x` in its sampling scope, every parse of which has `c`'s
bytes as its descriptor and members that name every position the layout reads (`NamesDefined`), is in `ScopeAcc` at the
members' layout and the horizon `T`. The fit is at the parsed statement's gates, the members unconnected. -/
def ScanScopeAccMembers : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String),
    scanOf j program s = .ok sc → Flock.canon j = .ok c →
    ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Claims.SamplingGeneric.Arg Coin}
      (D : Claims.SamplingGeneric.Compiled Coin A) (msOf : Claims.Sampling.AuditStatement → Spot.Bits.Members)
      (x : Claims.Msg),
      StmtMembers D msOf → D.Scope x →
      (∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8) →
      (∀ st, D.parse x = some st → NamesDefined (msOf st) s) →
      NetTiming.ScopeAcc D (fun st => membersLayout (msOf st) s) (fun _ => s.window) x

end Properties.NetworkScan
