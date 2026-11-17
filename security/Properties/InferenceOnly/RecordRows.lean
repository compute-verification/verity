module

public import Properties.InferenceOnly.Record
public import Properties.InferenceOnly.RecordForm
public import InferenceOnlyVerifier.RowsProtocol

/-!
# What the rows check's acceptance means

`InferenceOnly.RowsProtocol.checkBlobs`, the core protocol `verity/inference-only/rows/v1`, reads an execution phase's
circuit record with the record check (`Record.read`) and holds one pair to the rate rule in rows: a block's rows, the
summary's units of each template times the rows of one unit of it (`RecordRows.ofSummary`), and the summary's fresh
bits.

**A block's rows** (`rowsAt`): on the record's text over its Program, the templates the text gives
(`Spot.RecordText.templatesOf`), each with its units whose slot is the block (`Spot.RecordText.slot`), times the rows of
one unit of it, by the template's id (`RecordRows.blockRows`). It reads the text's own templates and slots, the ones
`Served` lays a circuit's blocks out by, and no summary.

**The rate rule in rows** (`RowsComplies`, stated beside its proof in
`Proofs.Properties.InferenceOnly.RecordRowsComplies`): where the check accepts, the record check reads the case's instance and
trusted input, the rows table is the one its circuit files give, the record lays out a circuit, and in every circuit it
lays out every block with a gate has, on the Program and the record the case sends, rows at most `Γ` and fresh bits at
most `α = num / den` times its rows.

What it does not say: that a file `circuits` names is the file SPOT's proofs of that template's units are verified
against (a tie between the two protocols' params); that a wired row holds its writer's committed bits
(`Vocab.WiresSound`); anything of the rule in the Program's gates, which this check does not decide.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record

open Lean (Json)
open Flock.Program (Prog)
open InferenceOnlyVerifier.Record.SecurityProofs (form)

/-- **Block `u`'s rows** on the record text `doc` over the Program `P`: over the templates the text gives, each
template's units whose slot is `u` times the rows of one unit of it (`m`, by template id); none where the text gives no
templates or `m` has no entry for one of them. -/
def rowsAt (P : Prog) (doc : Json) (m : String → Option ℕ) (u : ℕ) : Option ℕ :=
  match Spot.RecordText.templatesOf P doc with
  | .ok ts => InferenceOnlyVerifier.RecordRows.blockRows m (ts.map fun t =>
      (t.tid, ((List.range t.units).filter fun v => Spot.RecordText.slot P doc t.tid v == some u).length))
  | .error _ => none

end InferenceOnlyVerifier.Properties.Record
