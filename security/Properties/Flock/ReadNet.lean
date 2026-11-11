module

public import Flock.ReadNet.Template

/-!
# The read's template check at the served expert-row read (`ExpertReadAccepted`)

What `Flock.ReadNet.check` does at the served MoE model's expert-row read (`EmbeddingRead_v1{V=256,H=3072}`), stated
against the template's code: the template `Tmpl.parse` reads back from the text of `readTmpl 256 3072` passes
`check 256 3072`. `TableReadBinding`'s honest lemma takes it, and nothing in Lean proves it yet: the kernel doesn't
evaluate the template's generators or its text, and a proof at every `V` and `H` needs each part's run (`runOk`) and a
bound on every generator's rows, which the root Definition's width sums.

It is what `flock-read-tmpl 256 3072` checks before it exits 0 (`ReadTmplCmd.template`): it renders `readTmpl 256 3072`,
parses the text back, and runs `ReadNet.check 256 3072` on the template it parsed.
-/

@[expose] public section

namespace Accounting.ReadUnit

/-- **`ReadNet.check` accepts the read's template at 256 rows of 3072 words, as its text reads back** (witnessed outside
Lean: `flock-read-tmpl 256 3072` exits 0). -/
def ExpertReadAccepted : Prop :=
  ∃ t, Flock.Tmpl.parse
      (((Flock.ReadNet.render (Flock.ReadNet.readTmpl 256 3072)).splitOn "\n").filter (· ≠ "")).toArray = .ok t ∧
    Flock.ReadNet.check 256 3072 t = true

end Accounting.ReadUnit
