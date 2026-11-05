import NetworkCheck.Period
import SampledProofs.Bits
import SampledProofs.Bound
import Definitions.Claims.Sampling.Strata
import Definitions.NetworkCertifier.PeriodConditions
import Properties.NetworkScan.Flattens

/-!
# An accepted scan's incorrect sets commit at most the check's `faulty_bits` (`NetTiming.WindowsPlaced`)

`NetTiming.WindowsPlaced D δ sw Pw β` is a premise of `PeriodLeakRegistered` and `PeriodTiedCount`: each window's scan
statement has its deployed partition, and each of its possible incorrect sets at `δ` commits at most `β i` bits
(`faultyBits`, every committed gate of each incorrect proof block: its inputs, its returned gates and the gates another
proof block reads). The rows form's check charges a window's scan `faulty_bits = b · max(b_ing, b_eg)`
(`NetworkCheck.Period.Scan.terms`), `b` its audit's bound on wrong units and `b_ing`, `b_eg` the bits one ingress and
one egress unit commits (`scanProg`). `ScanWindowsPlaced` says that this number is such a `β` at the circuit files `fs`
of the scan's templates that the period check accepts, each laid out by Bits as its template
(`Properties.NetworkScan.FilesOf`) and reading no input padding (`Properties.NetworkScan.PaddingUnread`), for windows
whose statements are laid out as `SampledProofs.Bits` lays out a sampled statement's gates over those files
(`StmtBits`), wherever the audit's bound holds (`WrongBounds`).

* `StmtBits D P fs s`: the statement `s` is its audit's members' joint program (`Properties.NetworkScan.Flattens`),
  each member's circuit file the one `fs` lists at a template index of the Program `P` (`templateNodes`), and each of
  its partition's proof blocks is a member's unit's (`Bits.blockOf`, at the member's constants, `Bits.consts`), at the
  member's first position (`Bits.base`).
* `WrongBounds s P δ b`: each possible incorrect set of `s` at `δ` holds at most `b` proof blocks. It is the sampled
  leg's: the audit's answer publishes `b` (`SampledProofs.Bound.boundOf`), and `WrongBoundsDrawn` gives it at a strata
  draw. The period check reads the answer's `bound` and not the draw, so `ScanWindowsPlaced` takes it as it stands.
* `WrongBoundsDrawn`: where a draw reads as strata (`SampledProofs.Bound.strataOf`) whose sizes and draws, then strata
  drawn whole (an input class), are the design's strata draw (`StrataDrawn`, beside `OneStage.strata`), the bound the
  audit publishes at `δ` (`SampledProofs.Bound.boundOf`) is a `WrongBounds` bound at `δ`.

`Proofs.Properties.NetworkPeriod` proves `ScanWindowsPlaced` (`scanWindowsPlaced`) and `WrongBoundsDrawn`
(`wrongBoundsDrawn`).
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period (Schedule Scan scanProg templateNodes)
open SampledProofs.Bits (Members keyProg blockOf consts base)
open Properties.NetworkScan (Flattens FilesOf PaddingUnread)
open Claims.Sampling

/-- **The statement `s` is laid out as `SampledProofs.Bits` lays it out over the circuit files `fs` of the Program
`P`'s templates**: `D` partitions it as some `Q`, its gates are some members' joint program (`Flattens`), each member's
circuit file the one `fs` lists at a template index of `P` (`templateNodes`; `FilesOf` holds the file there to Bits'
layout of that template), and each of `Q`'s proof blocks is some member's unit's (`Bits.blockOf`, at the member's
constants `Bits.consts`), at the member's first position (`Bits.base`). The deployed partition is the sampled
verifier's, and `fs` the files the period check lists, so it is that code's premise at those files. -/
def StmtBits {Coin Col T : Type} (D : Deploy Coin Col T) (P : Flock.Program.Prog) (fs : List Flock.Circuit)
    (s : AuditStatement) : Prop :=
  ∃ (Q : Partition) (ms : Members), D.partition s = some Q ∧ Flattens ms s.prog ∧
    (∀ v < Q.blocks.length, ∃ m < ms.length, ∃ u < (ms.getD m default).2.size,
      Q.blocks.getD v [] =
        (blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ∧
    ∀ m < ms.length, ∃ i < (templateNodes P).length, (ms.getD m default).1 = fs.getD i default

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

/-- **An accepted scan's incorrect sets commit at most the check's `faulty_bits`.** If `scanProg` accepts the Program
`P` as the registered Program `program` at the schedule `s` with terms `sc` (`scanOf` runs it on the Program a scan's
descriptor decodes to), and `fs` are `P`'s templates' circuit files, each laid out by Bits as its template (`FilesOf`)
and reading no input padding (`PaddingUnread`), then at every deployment `D`, for `n` windows whose scan statements
`sw i`, each one `D` partitions as `Pw i` and laid out by Bits over `fs` (`StmtBits`), have at `δ` possible incorrect
sets of at most `b i` proof blocks (`WrongBounds`), each such set commits at most `b i · max(b_ing, b_eg)` bits:
`WindowsPlaced` at `β i = b i · max sc.ingress sc.egress`, the check's `faulty_bits` (`Scan.terms`). -/
def ScanWindowsPlaced : Prop :=
  ∀ (P : Flock.Program.Prog) (program : String) (s : Schedule) (sc : Scan) (fs : List Flock.Circuit),
    scanProg P program s = .ok sc → FilesOf P fs → PaddingUnread P fs →
    ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      (D : Deploy Coin Col T) {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (δ : ℝ)
      (b : Fin n → ℕ),
      (∀ i, StmtBits D P fs (sw i)) → (∀ i, D.partition (sw i) = some (Pw i)) →
      (∀ i, WrongBounds (sw i) (Pw i) δ (b i)) →
        NetTiming.WindowsPlaced D δ sw Pw fun i => b i * max sc.ingress sc.egress

end Properties.NetworkPeriod
