module

public import Properties.Accounting.Remote
public import Definitions.Core.InProofs

/-!
# PoUB's read session inside proofs: accepted proofs give openings `checkWindow` accepts

In PoUB's read session in zero knowledge, window `t`'s openings never reach the verifier. Each draw `k` of the window
(`drawn[k]`, pass and position, from its openings coin) is one proof block of the store's opening Definition, its port
values the draw's opening, staged in SPOT one-stage statements with every proof block selected and proved behind the ZK
server. The verifier accepts the window when every statement is accepted, each proof block's registered output is `1`, and
each proof block's public rows open to the verifier's own derivation.

`SpanGives` states what that acceptance gives, in `Verity.InProofs`'s shape: the openings the extraction opens at the
proof blocks' ports, decoded by the unit's port layout (`decode`), are openings `checkWindow` accepts, except with the proof
system's probability `ε` that an accepted window has a proof block whose values its circuit does not accept. It takes two
premises, both named and both open:

* the proof system's half (`Verity.InProofs.Gives` at `K₀ = 1`, at the relation `right` each proof block's circuit
  defines): `IntegrityPeriod` at `K₀ = 1` over the window's statements, whose sampling term is `0` under `All`, so `ε` is
  the statements' `ε_ks + δ_link` summed, once proofs bridge it to the deployed statement (the period form of V5);
* the unit refines today's check (`OpeningUnitRefines`): every port value its circuit accepts decodes to an opening
  `checkOpening` passes, at its draw's pass, position and receipter.

PoUB's per-window and pooled theorems read `checkWindow`'s acceptance and stand on this unchanged.
-/

@[expose] public section

open RemoteAudit

namespace Properties.RemoteAudit.InProofs

open Properties.RemoteAudit Verity.InProofs Verity.Game
open scoped ENNReal

/-- **Today's check of draw `k`'s opening**: `checkOpening` passes it at the draw's pass and position, under the
receipter of its pass's seed, as `checkWindow` checks it. -/
def OpeningOk (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (drawn : Array (Nat × Nat)) (k : Fin drawn.size) (o : Opening) : Prop :=
  checkOpening reg digest appends w (receipterOf reg w drawn[k].1) drawn[k].1 drawn[k].2 o = none

/-- **The opening unit refines today's check**: every port value the proof block of draw `k` accepts (`right k`) decodes
(`decode`, the unit's port layout) to an opening that `checkOpening` passes at draw `k`. -/
def OpeningUnitRefines (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (drawn : Array (Nat × Nat)) {V : Type} (right : Fin drawn.size → V → Prop) (decode : V → Opening) : Prop :=
  Refines (V := fun _ => V) right fun k v => OpeningOk reg digest appends w drawn k (decode v)

/-- **An accepted window inside proofs gives openings `checkWindow` accepts, except `ε`.** For window `w` of the
registration `reg` with the appends `appends`, its draws `drawn`, and a run of its proof blocks `r` (one a draw, its port
values in `V`): if the proof system gives that an accepted run leaves no proof block whose values its circuit (`right`)
rejects, except with probability `ε`, and the unit refines today's check, then the verifier accepts the run while
`checkWindow` refuses the openings the extraction opens, decoded by `decode`, with probability at most `ε`. -/
def SpanGives : Prop :=
  ∀ (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord) (drawn : Array (Nat × Nat))
    (Ω : Type) (μ : PMF Ω) (V : Type) (r : Run Ω drawn.size fun _ => V) (right : Fin drawn.size → V → Prop)
    (decode : V → Opening) (ε : ℝ≥0∞),
    Gives μ r right 1 ε →
    OpeningUnitRefines reg digest appends w drawn right decode →
    prPMF μ (fun ω => r.accepts ω ∧
        checkWindow reg digest appends w drawn (Array.ofFn fun k => decode (r.ports k ω)) ≠ Verdict.accept) ≤ ε

end Properties.RemoteAudit.InProofs
