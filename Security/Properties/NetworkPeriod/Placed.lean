import NetworkCheck.Period
import SampledProofs.Bits
import SampledProofs.Bound
import Definitions.Claims.Sampling.Strata
import Definitions.NetworkCertifier.PeriodConditions
import Properties.NetworkScan.Flattens

/-!
# A scan statement's layout and its audit's bound (`StmtBits`, `WrongBounds`)

`NetTiming.WindowsPlaced D δ sw Pw β` is a premise of `PeriodLeakRegistered`: each window's scan statement has its
deployed partition, and each of its possible incorrect sets at `δ` commits at most `β i` bits (`faultyBits`, every
committed gate of each incorrect proof block: its inputs, its returned gates and the gates another proof block reads).
The rows form's check charges a window's scan `faulty_bits = b · max(b_ing, b_eg)` (`NetworkCheck.Period.Scan.terms`),
`b` its audit's bound on wrong units and `b_ing`, `b_eg` the bits one ingress and one egress unit commits (`scanOf`).
Whether that number is such a `β` turns on two facts about the deployment: that its statements are laid out as
`SampledProofs.Bits` lays out a sampled statement's gates (`StmtBits`), and that the audit's bound holds (`WrongBounds`).

* `StmtBits D`: every statement `D` parses is its audit's members' joint program (`Properties.NetworkScan.Flattens`),
  each member a template of its descriptor's Program that Bits lays out (`Bits.check`, at a template node of the root,
  `templateNodes`), and each of its partition's proof blocks is a member's unit's (`Bits.blockOf`, at the member's
  constants, `Bits.consts`), at the member's first position (`Bits.base`).
* `WrongBounds s P δ b`: each possible incorrect set of `s` at `δ` holds at most `b` proof blocks. It is the sampled
  leg's: the audit's answer publishes `b` (`SampledProofs.Bound.boundOf`), and `WrongBoundsDrawn` gives it at a strata
  draw. The period check reads the answer's `bound` and not the draw.
* `WrongBoundsDrawn`: where a draw reads as strata (`SampledProofs.Bound.strataOf`) whose sizes and draws, then strata
  drawn whole (an input class), are the design's strata draw (`StrataDrawn`, beside `OneStage.strata`), the bound the
  audit publishes at `δ` (`SampledProofs.Bound.boundOf`) is a `WrongBounds` bound at `δ`.

`Proofs.Properties.NetworkPeriod` proves `WrongBoundsDrawn` (`wrongBoundsDrawn`).
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period (templateNodes)
open NetworkCheck.Circuit (templateDef)
open SampledProofs.Bits (Members keyProg blockOf consts base)
open Properties.NetworkScan (Flattens)
open Claims.Sampling

/-- **The deployment lays its statements out as `SampledProofs.Bits` does**: every statement `D` parses has a
descriptor C-Flock decodes to a Program `P` and a partition `Q`, its gates are some members' joint program (`Flattens`),
each member a template of `P`'s root (a template node's Definition, `templateNodes`) that Bits lays out (`Bits.check`),
and each of `Q`'s proof blocks is some member's unit's (`Bits.blockOf`, at the member's constants `Bits.consts`), at the
member's first position (`Bits.base`). The deployed parser is the sampled verifier's, so it is that code's premise. -/
def StmtBits {Coin Col T : Type} (D : Deploy Coin Col T) : Prop :=
  ∀ x s, D.parse x = some s → ∃ (P : Flock.Program.Prog) (Q : Partition) (ms : Members),
    Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧ D.partition s = some Q ∧ Flattens ms s.prog ∧
      (∀ v < Q.blocks.length, ∃ m < ms.length, ∃ u < (ms.getD m default).2.size,
        Q.blocks.getD v [] =
          (blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
            (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ∧
      ∀ m < ms.length, ∃ k ∈ templateNodes P,
        SampledProofs.Bits.check (ms.getD m default).1 P (templateDef P k).key = none

/-- **The audit's bound holds at `δ`**: each possible incorrect set of the statement `s` at its partition `P` and `δ`
(`NetTiming.faultySets`) holds at most `b` proof blocks. -/
def WrongBounds (s : AuditStatement) (P : Partition) (δ : ℝ) (b : ℕ) : Prop :=
  ∀ F ∈ NetTiming.faultySets s.design P δ, F.card ≤ b

/-- **The code's bound is the audit's bound at a strata draw.** If a draw `draw` over `population` units reads as the
strata `strata` (`SampledProofs.Bound.strataOf`, every draw but a Bernoulli one), then at every statement `s`, partition
`P` and sizes `ns` such that `strata`'s sizes and draws, in order, then strata of sizes `ns` drawn whole (an input class,
drawn `all`), are `s`'s design's strata draw on `P` (`StrataDrawn`), and every `δ = num/den` with `0 < num` and
`0 < den`, each possible incorrect set at `δ` holds at most the bound the audit publishes,
`SampledProofs.Bound.boundOf population draw δ`, proof blocks (`WrongBounds`). -/
def WrongBoundsDrawn : Prop :=
  ∀ (population : ℕ) (draw : Core.SamplingDesign) (strata : Array Core.Stratum),
    SampledProofs.Bound.strataOf population draw = some strata →
    ∀ (s : AuditStatement) (P : Partition) (ns : List ℕ) (δ : Core.Ratio),
      StrataDrawn s.design P ((strata.toList.map fun x => (x.size, x.k)) ++ ns.map fun m => (m, m)) →
      0 < δ.num → 0 < δ.den →
        WrongBounds s P ((δ.num : ℝ) / δ.den) (SampledProofs.Bound.boundOf population draw δ)

end Properties.NetworkPeriod
