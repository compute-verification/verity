import NetworkCheck.Period
import SampledProofs.Bits
import Definitions.NetworkCertifier.SampledAudit

/-!
# A sampling statement's gates are its audit's members' Boolean programs, side by side

A sampling statement's gates (`Claims.Sampling.AuditStatement.prog`, "the descriptor flattened") are the Boolean-profile
program its registered circuit files describe (`SampledProofs.Bits`): each member of the audit, a circuit file over some
of the partition's VUs, is its key program lowered at the deployed constants (`SampledProofs.Bits.prog`), the members
side by side, member `m`'s positions from `SampledProofs.Bits.base`. It returns every unit's output copy bits, the key
program's outputs.

* `bgateAt b`, `bgateOf`: a Boolean-profile gate as a sampling gate, its reads shifted by `b`.
* `consts`: the deployed constants, gate 0 set and gate 1 clear.
* `jointOf ms`, `Flattens ms C`: the members' joint program, and `C` being it.
* `FilesOf P fs`: one circuit file a template of the Program `P`, in order, each one Bits lays out as its template
  (`SampledProofs.Bits.check`).
* `vusOf`, `vbase`, `MembersOf P fs ms`: template `i`'s instances fill VUs of its own, `upv` a VU, from `vbase i`, and
  the members hold every VU once, each member one template's circuit file over VUs of that template's.
* `instPos`: the claims position of a bit of template `i`'s instance `t`, at its member, unit and copy, and past the
  joint program where nothing places it.
* `wordBit`: the port word and bit of a field's bit, each field whole u16 words, packed 16 bits a word.
-/

namespace Properties.NetworkScan

open SampledProofs.Bits (LGate Members keyProg prog base)
open NetworkCheck.Period (templateNodes bitsOf)
open NetworkCheck.Circuit (templateDef)

/-- **A Boolean-profile gate as a sampling gate** (`Claims.Sampling.BGate`), constructor for constructor, its reads
shifted by `b`. -/
def bgateAt (b : ℕ) : LGate → Claims.Sampling.BGate
  | .input => .input
  | .const v => .const v
  | .not i => .not (b + i)
  | .and ws => .and (ws.map (b + ·))
  | .xor ws => .xor (ws.map (b + ·))

/-- **A Boolean-profile gate as a sampling gate**, unshifted. -/
def bgateOf : LGate → Claims.Sampling.BGate := bgateAt 0

/-- **The deployed constants**: gate 0 set (the key program's constant) and gate 1 clear (its zero); every other input
gate is free. -/
def consts (g : ℕ) : Option Bool := if g = 0 then some true else if g = 1 then some false else none

/-- **A member's Boolean program**: its circuit file's key program on its units (`SampledProofs.Bits.keyProg`), lowered
at the deployed constants (`SampledProofs.Bits.prog`). -/
def memberProg (ci : Flock.Circuit × Array ℕ) : List LGate × List ℕ := prog (keyProg ci.1 ci.2.size) consts

/-- **The members' joint program**: each member's (`memberProg`) side by side, member `m`'s gates and outputs from
`base ms m`, as sampling gates. It returns every unit's output copy bits. -/
def jointOf (ms : Members) : Claims.Sampling.Prog :=
  ⟨(List.range ms.length).flatMap fun m => (memberProg (ms.getD m default)).1.map (bgateAt (base ms m)),
    (List.range ms.length).flatMap fun m => (memberProg (ms.getD m default)).2.map (base ms m + ·)⟩

/-- **`C` is the members `ms` flattened**: their joint program (`jointOf`). -/
def Flattens (ms : Members) (C : Claims.Sampling.Prog) : Prop := C = jointOf ms

/-- **The circuit files of `P`'s templates**: one a template node (`NetworkCheck.Period.templateNodes`), in order, each
one Bits lays out as its template (`SampledProofs.Bits.check`, at the template's Definition's key). -/
def FilesOf (P : Flock.Program.Prog) (fs : List Flock.Circuit) : Prop :=
  fs.length = (templateNodes P).length ∧ ∀ i < fs.length,
    SampledProofs.Bits.check (fs.getD i default) P (templateDef P ((templateNodes P).getD i 0)).key = none

/-- **Template `i`'s VUs** at its circuit file `fs[i]`: its template node's members, `upv` a VU, rounded up. -/
def vusOf (P : Flock.Program.Prog) (fs : List Flock.Circuit) (i : ℕ) : ℕ :=
  let upv := (fs.getD i default).upv
  (((P.defs[P.root]!).body.nodes[(templateNodes P).getD i 0]!).members + upv - 1) / upv

/-- **The VUs before template `i`'s**: template `i`'s are `vbase i`, …, `vbase (i + 1) − 1`. -/
def vbase (P : Flock.Program.Prog) (fs : List Flock.Circuit) (i : ℕ) : ℕ := ((List.range i).map (vusOf P fs)).sum

/-- **The members of `P`'s audit** at its templates' circuit files `fs`: together they hold each of the templates' VUs
once, and each is one template's circuit file over VUs of that template's. -/
def MembersOf (P : Flock.Program.Prog) (fs : List Flock.Circuit) (ms : Members) : Prop :=
  List.Perm (ms.flatMap (·.2.toList)) (List.range (vbase P fs fs.length)) ∧
    ∀ m < ms.length, ∃ i < fs.length, (ms.getD m default).1 = fs.getD i default ∧
      ∀ v ∈ (ms.getD m default).2.toList, vbase P fs i ≤ v ∧ v < vbase P fs (i + 1)

/-- **The claims position of a bit of template `i`'s instance `t`** in the members' joint program, at `pos`, its
position in a member's program (of the member's circuit file and units, and the instance's unit and copy): instance `t`
is copy `t % upv` of VU `vbase i + t / upv`, unit `u` of the first member whose VUs hold it at `u`. Where no member holds
the VU or `pos` places nothing, it is the joint program's length, a position past its gates: no gate, so neither an
input nor an output, and a layout that reaches it fits nothing. -/
def instPos (P : Flock.Program.Prog) (fs : List Flock.Circuit) (ms : Members) (i t : ℕ)
    (pos : Flock.Circuit → ℕ → ℕ → ℕ → Option ℕ) : ℕ :=
  let upv := (fs.getD i default).upv
  let v := vbase P fs i + t / upv
  let inMember := fun (m : ℕ) =>
    let ci := ms.getD m default
    ((List.range ci.2.size).find? (ci.2.getD · 0 == v)).bind fun u =>
      (pos ci.1 ci.2.size u (t % upv)).map (base ms m + ·)
  ((List.range ms.length).findSome? inMember).getD (jointOf ms).gates.length

/-- **Bit `β` of field `π`** among the fields `ts`, each whole u16 words (`NetworkCheck.Circuit.templateOf`: an input
row whole words, an output field of fewer than 16 bits one word), packed 16 bits a word least significant first, a
unit port being one word (`SampledProofs.Bits.check`): its word and its bit in the word. -/
def wordBit (ts : List Flock.Program.Ty) (π β : ℕ) : ℕ × ℕ :=
  let x := 16 * ((ts.take π).map fun t => (bitsOf t + 15) / 16).sum + β
  (x / 16, x % 16)

end Properties.NetworkScan
