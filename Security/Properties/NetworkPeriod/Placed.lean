import NetworkCheck.Period
import SampledProofs.Bits
import SampledProofs.Bound
import Definitions.Claims.Sampling.Strata
import Definitions.NetworkCertifier.PeriodConditions
import Definitions.NetworkCertifier.Tied

/-!
# An accepted scan's incorrect sets commit at most the check's `faulty_bits` (`NetTiming.WindowsPlaced`)

`NetTiming.WindowsPlaced D δ sw Pw β` is a premise of `PeriodLeakRegistered` and `PeriodTiedCount`: each window's scan
statement has its deployed partition, and each of its possible incorrect sets at `δ` commits at most `β i` bits
(`faultyBits`, every committed gate of each incorrect proof block: its inputs, its returned gates and the gates another
proof block reads). The rows form's check charges a window's scan `faulty_bits = b · max(b_ing, b_eg)`
(`NetworkCheck.Period.Scan.terms`), `b` its audit's bound on wrong units and `b_ing`, `b_eg` the bits one ingress and
one egress unit commits (`scanOf`). `ScanWindowsPlaced` says that this number is such a `β`, at a deployment whose
statements are laid out as `SampledProofs.Bits` lays out a sampled statement's gates (`StmtBits`), wherever the audit's
bound holds (`WrongBounds`).

* `gateOf`, `memberProg`: Bits' Boolean-profile program of a member, as a sampling statement's.
* `LaidOut s P ms κ`: the statement `s` and its partition `P` are the members `ms` laid out by Bits at the constants
  `κ`: `s`'s gates are the members' programs side by side, and each of `P`'s proof blocks is a member's unit's
  (`Bits.blockOf`), at the member's first position (`Bits.base`).
* `StmtBits D`: every statement `D` parses, with its partition, is laid out by Bits from members each a template of its
  descriptor's Program (`Bits.check`, at a template node of the root, `templateNodes`).
* `WrongBounds s P δ b`: each possible incorrect set of `s` at `δ` holds at most `b` proof blocks. It is the sampled
  leg's: the audit's answer publishes `b` (`SampledProofs.Bound.boundOf`), and `WrongBoundsDrawn` gives it at a strata
  draw. The period check reads the answer's `bound` and not the draw, so `ScanWindowsPlaced` takes it as it stands.
* `WrongBoundsDrawn`: at strata whose sizes and draws, then strata drawn whole (an input class), are the design's
  strata draw (`StrataDrawn`, beside `OneStage.strata`), the code's count bound at `δ`
  (`SampledProofs.Bound.countBound`) is a `WrongBounds` bound at `δ`.

`Proofs.Properties.NetworkPeriod` proves `ScanWindowsPlaced` (`scanWindowsPlaced`) and `WrongBoundsDrawn`
(`wrongBoundsDrawn`).
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period (Schedule Scan scanOf templateNodes)
open SampledProofs.Bits (LGate Members keyProg blockOf base)
open Claims.Sampling

/-- Bits' Boolean-profile gate as a sampling statement's, constructor for constructor. -/
def gateOf : LGate → BGate
  | .input => .input
  | .const b => .const b
  | .not i => .not i
  | .and ws => .and ws
  | .xor ws => .xor ws

/-- **A member's program** at the constants `κ`, as a sampling statement's: its circuit file's key program over its
units, lowered to the Boolean profile (`Bits.prog`). -/
def memberProg (ci : Flock.Circuit × Array ℕ) (κ : ℕ → Option Bool) : Prog :=
  let p := SampledProofs.Bits.prog (keyProg ci.1 ci.2.size) κ
  ⟨p.1.map gateOf, p.2⟩

/-- **The statement `s` and its partition `P` are the members `ms` laid out by Bits**, member `m` at the constants
`κ m`: `s`'s gates are the members' programs side by side (`NetTiming.sideBySide`), and each of `P`'s proof blocks is
some member's unit's (`Bits.blockOf`), at the member's first position (`Bits.base`). -/
def LaidOut (s : AuditStatement) (P : Partition) (ms : Members) (κ : ℕ → ℕ → Option Bool) : Prop :=
  s.prog = NetTiming.sideBySide ((List.range ms.length).map fun m => memberProg (ms.getD m default) (κ m)) ∧
    ∀ v < P.blocks.length, ∃ m < ms.length, ∃ u < (ms.getD m default).2.size,
      P.blocks.getD v [] =
        (blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size) (κ m) u).map (base ms m + ·)

/-- **The deployment lays its statements out as `SampledProofs.Bits` does**: every statement `D` parses has a
descriptor C-Flock decodes to a Program `P` and a partition, and is laid out by Bits (`LaidOut`) from members each a
template of `P`'s root (a template node's Definition, `templateNodes`) that Bits lays out (`Bits.check`). The deployed
parser is the sampled verifier's, so it is that code's premise. -/
def StmtBits {Coin Col T : Type} (D : Deploy Coin Col T) : Prop :=
  ∀ x s, D.parse x = some s → ∃ (P : Flock.Program.Prog) (Q : Partition) (ms : Members) (κ : ℕ → ℕ → Option Bool),
    Flock.Program.decode ⟨s.desc.toArray⟩ = .ok P ∧ D.partition s = some Q ∧ LaidOut s Q ms κ ∧
      ∀ m < ms.length, ∃ k ∈ templateNodes P,
        SampledProofs.Bits.check (ms.getD m default).1 P (P.defs[((P.defs[P.root]!).body.nodes[k]!).fn]!).id = none

/-- **The audit's bound holds at `δ`**: each possible incorrect set of the statement `s` at its partition `P` and `δ`
(`NetTiming.faultySets`) holds at most `b` proof blocks. -/
def WrongBounds (s : AuditStatement) (P : Partition) (δ : ℝ) (b : ℕ) : Prop :=
  ∀ F ∈ NetTiming.faultySets s.design P δ, F.card ≤ b

/-- **The code's count bound is the audit's bound at a strata draw.** At every statement `s`, partition `P`, strata
`strata` and sizes `ns` such that `strata`'s sizes and draws, in order, then strata of sizes `ns` drawn whole (an input
class, drawn `all`), are `s`'s design's strata draw on `P` (`StrataDrawn`), and every `δ = num/den` with `0 < num` and
`0 < den`, each possible incorrect set at `δ` holds at most `SampledProofs.Bound.countBound strata δ` proof blocks
(`WrongBounds`). -/
def WrongBoundsDrawn : Prop :=
  ∀ (s : AuditStatement) (P : Partition) (strata : Array Core.Stratum) (ns : List ℕ) (δ : Core.Ratio),
    StrataDrawn s.design P ((strata.toList.map fun x => (x.size, x.k)) ++ ns.map fun m => (m, m)) →
    0 < δ.num → 0 < δ.den →
      WrongBounds s P ((δ.num : ℝ) / δ.den) (SampledProofs.Bound.countBound strata δ)

/-- **An accepted scan's incorrect sets commit at most the check's `faulty_bits`.** If `scanOf` accepts the scan `j` at
the schedule `s` with terms `sc` and its canonical text is `c`, then at every deployment `D` that lays its statements out
as Bits does (`StmtBits`), for `n` windows whose scan statements `sw i`, each one `D` parses with `c`'s bytes as its
descriptor and `Pw i` as its partition, have at `δ` possible incorrect sets of at most `b i` proof blocks
(`WrongBounds`), each such set commits at most `b i · max(b_ing, b_eg)` bits: `WindowsPlaced` at
`β i = b i · max sc.ingress sc.egress`, the check's `faulty_bits` (`Scan.terms`). -/
def ScanWindowsPlaced : Prop :=
  ∀ (j : Lean.Json) (program : String) (s : Schedule) (sc : Scan) (c : String),
    scanOf j program s = .ok sc → Flock.canon j = .ok c →
    ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
      (D : Deploy Coin Col T), StmtBits D →
    ∀ {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (δ : ℝ) (b : Fin n → ℕ),
      (∀ i, ∃ x, D.parse x = some (sw i)) → (∀ i, (⟨(sw i).desc.toArray⟩ : ByteArray) = c.toUTF8) →
      (∀ i, D.partition (sw i) = some (Pw i)) → (∀ i, WrongBounds (sw i) (Pw i) δ (b i)) →
        NetTiming.WindowsPlaced D δ sw Pw fun i => b i * max sc.ingress sc.egress

end Properties.NetworkPeriod
