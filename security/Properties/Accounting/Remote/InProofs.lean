module

public import Properties.Accounting.Remote
public import Definitions.Core.InProofs

/-!
# PoUB's read session inside proofs: an accepted run gives openings `checkWindow` accepts

`SpanGives` holds for any run of a window's proof blocks that meets its two hypotheses, `Gives` at the relation `right`
and `right ≤ OpeningOk`; it says nothing of the deployed run until proofs' period form of V5 gives `Gives` there.

In PoUB's read session in zero knowledge, window `t`'s openings never reach the verifier. Each draw `k` of the window
(`drawn[k]`, pass and position, from its openings coin) is one proof block of the store's opening Definition, staged in
SPOT one-stage statements with every proof block selected and proved behind the ZK server. Every value `checkOpening`
reads outside the opening (the registration, its digest, the appends, the window record and the receipter of the pass's
seed) is the verifier's own and is in the proof block's public rows, which the verifier refuses unless they are its own
derivation; so each is an argument of the lemma here, fixed before the run.

`SpanGives`, in `Verity.InProofs`'s shape at `K₀ = 1`: if an accepted run leaves a proof block whose value `right`
rejects with probability at most `ε`, and every value `right` accepts decodes (`decode`) to an opening `checkOpening`
passes at its draw, then the verifier accepts while `checkWindow` refuses the decoded openings with probability at most
`ε`. The guarantee that instantiates it builds `r.ports` from the counted extractor `IntegrityPeriod` reads, sets `right`
to the opening unit's circuit, and takes `Gives` from `IntegrityPeriod` at `K₀ = 1` over the window's statements, whose
sampling term is `0` under `All`; `ε` is then the statements' `ε_ks + δ_link` summed.
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

/-- **An accepted run gives openings `checkWindow` accepts, except `ε`, for any run meeting its hypotheses**; it says
nothing of the deployed run until proofs' period form of V5 gives `Gives` there. For window `w` of the registration `reg`
with the appends `appends`, its draws `drawn`, and a run `r` of its proof blocks (one a draw, values in `V`): if an
accepted run leaves a proof block whose value `right` rejects with probability at most `ε`, and every value `right`
accepts decodes by `decode` to an opening `checkOpening` passes at its draw, then the verifier accepts the run while
`checkWindow` refuses the decoded openings with probability at most `ε`. -/
def SpanGives : Prop :=
  ∀ (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord) (drawn : Array (Nat × Nat))
    (Ω : Type) (μ : PMF Ω) (V : Type) (r : Run Ω drawn.size fun _ => V) (right : Fin drawn.size → V → Prop)
    (decode : V → Opening) (ε : ℝ≥0∞),
    Gives μ r right 1 ε →
    right ≤ (fun k v => OpeningOk reg digest appends w drawn k (decode v)) →
    prPMF μ (fun ω => r.accepts ω ∧
        checkWindow reg digest appends w drawn (Array.ofFn fun k => decode (r.ports k ω)) ≠ Verdict.accept) ≤ ε

end Properties.RemoteAudit.InProofs
