module

public import Properties.Accounting.Remote
public import Definitions.Core.InProofs
public import Mathlib.Probability.ProbabilityMassFunction.Monad

/-!
# PoUB's read session inside proofs: an accepted run gives openings `checkWindow` accepts

`SpanGives` holds for any run of a window's proof blocks that meets its two hypotheses, `Gives` at the relation `right`
and `right ≤ OpeningOk`; it says nothing of the deployed run until proofs' period form of V5 gives `Gives` there.

In PoUB's read session in zero knowledge, window `t`'s openings never reach the verifier. Each sampled position `k` of the window
(`sampled[k]`, pass and position, from its openings coin) is one proof block of the store's opening Definition, staged in
SPOT one-stage statements with every proof block selected and proved behind the ZK server. Every value `checkOpening`
reads outside the opening (the registration, its digest, the appends, the window record and the receipter of the pass's
seed) is the verifier's own and is in the proof block's public rows, which the verifier refuses unless they are its own
derivation; so each is an argument of the lemma here, fixed before the run.

`SpanGives`, in `Verity.InProofs`'s shape at `K₀ = 1`: if an accepted run leaves a proof block whose value `right`
rejects with probability at most `ε`, and every value `right` accepts decodes (`decode`) to an opening `checkOpening`
passes at its sampled position, then the verifier accepts while `checkWindow` refuses the decoded openings with probability at most
`ε`. The guarantee that instantiates it builds `r.ports` from the counted extractor `IntegrityPeriod` reads, sets `right`
to the opening unit's circuit, and takes `Gives` from `IntegrityPeriod` at `K₀ = 1` over the window's statements, whose
sampling term is `0` under `All`; `ε` is then the statements' `ε_ks + ε_bind` summed.
-/

@[expose] public section

open RemoteAudit

namespace Properties.RemoteAudit.InProofs

open Properties.RemoteAudit Verity.InProofs Verity.Game
open scoped ENNReal

/-- **Today's check of sampled position `k`'s opening**: `checkOpening` passes it at the sampled pass and position, under the
receipter of its pass's seed, as `checkWindow` checks it. -/
def OpeningOk (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord)
    (sampled : Array (Nat × Nat)) (k : Fin sampled.size) (o : Opening) : Prop :=
  checkOpening reg digest appends w (receipterOf reg w sampled[k].1) sampled[k].1 sampled[k].2 o = none

/-- **An accepted run gives openings `checkWindow` accepts, except `ε`, for any run meeting its hypotheses**; it says
nothing of the deployed run until proofs' period form of V5 gives `Gives` there. For window `w` of the registration `reg`
with the appends `appends`, its sampled positions `sampled`, and a run `r` of its proof blocks (one per sampled position, values in `V`): if an
accepted run leaves a proof block whose value `right` rejects with probability at most `ε`, and every value `right`
accepts decodes by `decode` to an opening `checkOpening` passes at its sampled position, then the verifier accepts the run while
`checkWindow` refuses the decoded openings with probability at most `ε`. -/
def SpanGives : Prop :=
  ∀ (reg : Registration) (digest : ByteArray) (appends : Array Append) (w : WindowRecord) (sampled : Array (Nat × Nat))
    (Ω : Type) (μ : PMF Ω) (V : Type) (r : Run Ω sampled.size fun _ => V) (right : Fin sampled.size → V → Prop)
    (decode : V → Opening) (ε : ℝ≥0∞),
    Gives μ r right 1 ε →
    right ≤ (fun k v => OpeningOk reg digest appends w sampled k (decode v)) →
    prPMF μ (fun ω => r.accepts ω ∧
        checkWindow reg digest appends w sampled (Array.ofFn fun k => decode (r.ports k ω)) ≠ Verdict.accept) ≤ ε

/-- The run of one sampled position, `(0, 0)`, that always accepts and whose proof block holds the opening `o`. -/
def oneSample (o : Opening) : Run Unit (#[((0 : Nat), (0 : Nat))].size) fun _ => Opening :=
  ⟨fun _ => True, fun _ _ => o, fun _ => []⟩

/-- **`SpanGives`' hypotheses hold where the check accepts**: a registration `Registration.make?` accepts at the
deployed shape (130023424 blocks, p1024/k3/masked, sketch-sha512 receipts of 16 rows), a window of one pass over the
whole store and an opening of position 0, where the run of that one sampled position, with `OpeningOk` its relation and `decode` the
identity, meets `Gives` at `ε = 0`, and `checkWindow` accepts its opening. -/
def SpanGivesHonest : Prop :=
  ∃ (reg : Registration) (w : WindowRecord) (o : Opening),
    Registration.make? (zeros 1) 130023424 reg.payloadRoot (zeros Codec.SEED_BYTES) ⟨Codec.P1024, 3, true⟩ .sketch
        false LAYOUT 16 = .ok reg ∧
      Gives (PMF.pure ()) (oneSample o) (fun k v => OpeningOk reg reg.digest #[] w #[(0, 0)] k v) 1 0 ∧
      checkWindow reg reg.digest #[] w #[(0, 0)] (Array.ofFn fun k => (oneSample o).ports k ()) = .accept

/-- **`SpanGives`' conclusion fails without its hypotheses**: at the same registration and window, a run of the one
sampled position that always accepts and holds an opening `checkWindow` refuses gives the event probability `1`, so the bound fails
at `ε = 0`. -/
def SpanGivesNontrivial : Prop :=
  ∃ (reg : Registration) (w : WindowRecord) (o : Opening),
    ¬ prPMF (PMF.pure ()) (fun ω => (oneSample o).accepts ω ∧
        checkWindow reg reg.digest #[] w #[(0, 0)] (Array.ofFn fun k => (oneSample o).ports k ω) ≠ Verdict.accept) ≤ 0

end Properties.RemoteAudit.InProofs
