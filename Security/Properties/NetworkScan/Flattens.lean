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
* `MembersOf P fs rs ms`: the members hold each unit of the template query's population once (its ranges `rs`, as
  `Flock.TemplateQuery.evaluate` gives them, a unit an instance), each member one template's circuit file over units
  the code locates at that template's node (`Flock.TemplateQuery.locate`).
* `instPos`: the claims position of a bit of template `i`'s instance `t`, at the unit the code locates there and its
  member, and past the joint program where nothing places it.
* `wordBit`: the port word and bit of a field's bit, each field whole u16 words, packed 16 bits a word.
-/

namespace Properties.NetworkScan

open SampledProofs.Bits (LGate Members keyProg prog base)
open NetworkCheck.Period (templateNodes bitsOf)
open NetworkCheck.Circuit (templateDef)
open Flock.TemplateQuery (TRange locate population)

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

/-- **The members of `P`'s audit** at its templates' circuit files `fs` and the ranges `rs` of the template query over
`P` (`Flock.TemplateQuery.evaluate`, the population as the sampled verifier reads it: each call and each batch member
of a template one unit, in root node order): together they hold each of the population's units once
(`Flock.TemplateQuery.population`), and each is one template's circuit file over units the code locates at that
template's node (`Flock.TemplateQuery.locate`). -/
def MembersOf (P : Flock.Program.Prog) (fs : List Flock.Circuit) (rs : Array TRange) (ms : Members) : Prop :=
  List.Perm (ms.flatMap (·.2.toList)) (List.range (population rs)) ∧
    ∀ m < ms.length, ∃ i < fs.length, (ms.getD m default).1 = fs.getD i default ∧
      ∀ v ∈ (ms.getD m default).2.toList, (locate rs v).map (·.1) = some ((templateNodes P).getD i 0)

/-- **The claims position of a bit of template `i`'s instance `t`** in the members' joint program, at `pos`, its
position in a member's program (of the member's circuit file and units, and the instance's unit): instance `t` is the
unit the code locates at template `i`'s node and member `t` (`Flock.TemplateQuery.locate` over the template query's
ranges `rs`), unit `u` of the first member whose units hold it at `u`. Where no member holds it or `pos` places nothing,
it is the joint program's length, a position past its gates: no gate, so neither an input nor an output, and a layout
that reaches it fits nothing. -/
def instPos (P : Flock.Program.Prog) (rs : Array TRange) (ms : Members) (i t : ℕ)
    (pos : Flock.Circuit → ℕ → ℕ → Option ℕ) : ℕ :=
  let k := (templateNodes P).getD i 0
  let inMember := fun (m : ℕ) =>
    let ci := ms.getD m default
    ((List.range ci.2.size).find? fun u => (locate rs (ci.2.getD u 0)).any fun l => l.1 == k && l.2.1 == t).bind
      fun u => (pos ci.1 ci.2.size u).map (base ms m + ·)
  ((List.range ms.length).findSome? inMember).getD (jointOf ms).gates.length

/-- **Bit `β` of field `π`** among the fields `ts`, each whole u16 words (`NetworkCheck.Circuit.templateOf`: an input
row whole words, an output field of fewer than 16 bits one word), packed 16 bits a word least significant first, a
unit port being one word (`SampledProofs.Bits.check`): its word and its bit in the word. -/
def wordBit (ts : List Flock.Program.Ty) (π β : ℕ) : ℕ × ℕ :=
  let x := 16 * ((ts.take π).map fun t => (bitsOf t + 15) / 16).sum + β
  (x / 16, x % 16)

end Properties.NetworkScan
