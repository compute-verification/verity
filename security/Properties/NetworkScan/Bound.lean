module

public import Properties.NetworkScan.Chain
public import Definitions.NetworkCertifier.PeriodConditions

/-!
# The scan's count: the period's leak bound's right-hand side at the scans, under the scan's own terms

`NetTiming.Guarantees.PeriodLeakChecked` counts the secrets a monitoring period's windows let through by
`#Θ · windowsCount (reg ∪ chk)`: the registered parameters the record reads back under, times, a window, what its audit
leaves free with the inputs `reg ∪ chk` pinned, its possible incorrect sets, their committed bits and the timing
(`NetTiming.windowsCount`). `ScanCountBounded` bounds that right-hand side at the rows form's scans, of a period the
check accepts (`NetworkCheck.Period.check`), by the scan's own terms.

**The instance.** One egress link and the schedule's allocation of slots; each window's layout its statement's scan
layout (`layOf s`) over the window's `T` steps; the jitter bound the parameters'; each window's `β` the most bits one of
its possible incorrect sets commits (`faultyMost`).

**`#Θ = 1`.** The rows' record reads back from the padding frames' count alone: a connection's frames in bucket order,
less its `r · D` padding frames, are its lane's cells, each in its bucket. So the registered parameters are the
schedule's one pair of cells a bucket and latency (`paddings`), and no factor counts the schedules or a connection's
cells.

**The checked inputs**, two instances, each with the checks `PeriodLeakChecked` takes at it:

* both ties (`chk := tiesOf s`, the checks `stateTiedAt s` and `viewTiedAt s`, each with its premise and its miss in
  the bound's error): what is left free is the slot entries, at most `Scan.entries` bits a window;
* neither tie (`chk := cutOf s`, the checks that hold of every run, with no premise and no miss): what is left free is
  the egress views, at most `viewCount` gates a window.

**Every scan unit proved** is read from the public the check accepts with, never from a bound on the incorrect proof
blocks: each link-window's published facts name as many units sampled as its level holds. What that says of the
window's statement is a premise, `AnswersVerified`: the period check reads each link-window's answer from the evidence
and does not run the sampled verifier, so that the published facts are those of the statement's own design at its
partition is taken, not proved. With it the design misses no incorrect unit, the one possible incorrect set is the
empty one, and the window's two incorrect-set factors are 1.

The bound is in the scan's terms and not in the public's `K`: the public's term for a link-window's free inputs is its
tied audits' declared free inputs (`Scan.terms`), which the check compares with neither the slot entries' bits nor the
views'.
-/

@[expose] public section

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- **Each answer the period check reads is the sampled verifier's, for its window's registration and design**, a
premise: link-window `i`'s published facts are those of its statement's design at its partition, in what the count
reads of them. If they name as many units sampled as the level holds, the design `(sw i).design` on the partition
`Pw i` accepts with no unit incorrect only (`Claims.Sampling.Design.miss` at one unit is 0). -/
def AnswersVerified (pub : Public) {n : ℕ} (sw : Fin n → Claims.Sampling.AuditStatement)
    (Pw : Fin n → Claims.Sampling.Partition) : Prop :=
  ∀ (i : Fin n) (t : AuditTerms), pub.units[i.val]? = some t → t.facts.drawn = t.facts.units →
    (sw i).design.miss (Pw i) 1 ≤ 0

/-- **The registered parameters the rows' record reads back under** at the schedule `s`: its cells a bucket and its
latency, one pair. -/
def paddings (s : Schedule) : Set (ℕ × ℕ) := {(s.slots, s.delay)}

/-- The most bits one of window `i`'s possible incorrect sets at `δ` commits: the least `β i`
`NetTiming.WindowsPlaced` takes. -/
noncomputable def faultyMost (δ : ℝ) {n : ℕ} (sw : Fin n → Claims.Sampling.AuditStatement)
    (Pw : Fin n → Claims.Sampling.Partition) (i : Fin n) : ℕ :=
  (NetTiming.faultySets (sw i).design (Pw i) δ).sup (NetTiming.faultyBits (sw i).prog (Pw i))

/-- **The period's count at the scans is the slot entries' under both ties and the views' under neither.** If the
check accepts a period in the rows form with the public `pub`, its scan `j` is the one the parameters' form holds
(`scheduleOf`, `scanOf`), with canonical text `cs` decoding to `P`, then for link-windows' statements `sw i` whose
descriptor is `cs`'s bytes and whose gates are `P` flattened, at any partitions `Pw i`, registered inputs `reg` and
`δ ≥ 0`: where every link-window's published facts name every unit sampled and each answer read is verified
(`AnswersVerified`), the right-hand side of `PeriodLeakChecked` at the scans' instance, `#Θ · windowsCount (reg ∪ chk)`
with `Θ` the rows' one registered pair (`paddings`), is

* at most `2 ^ (windows · (entries + ⌈log2 jitter^T⌉))` with both ties' gates checked (`tiesOf`), and
* at most `2 ^ (windows · (viewCount + ⌈log2 jitter^T⌉))` with every free input off the views checked (`cutOf`). -/
def ScanCountBounded : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (sj j : Lean.Json) (cs : String) (P : Flock.Program.Prog),
      scheduleOf sj = .ok s → scanOf j p.program s = .ok scan → Flock.canon j = .ok cs →
      Flock.Program.decode cs.toUTF8 = .ok P →
    ∀ (sw : Fin p.windows → Claims.Sampling.AuditStatement) (Pw : Fin p.windows → Claims.Sampling.Partition)
      (reg : Claims.Sampling.AuditStatement → Finset ℕ) (δ : ℝ), 0 ≤ δ →
      (∀ i, (⟨(sw i).desc.toArray⟩ : ByteArray) = cs.toUTF8 ∧ Flattens P (sw i).prog) →
      (∀ t ∈ pub.units, t.facts.drawn = t.facts.units) → AnswersVerified pub sw Pw →
        (paddings s).encard *
            ((NetTiming.windowsCount (layOf s) (fun _ => s.window) δ (fun x => reg x ∪ tiesOf s x) p.jitter sw Pw
              (faultyMost δ sw Pw) : ℕ) : ℕ∞) ≤
          ((2 ^ (p.windows * (scan.entries + clog2 (p.jitter ^ s.window))) : ℕ) : ℕ∞) ∧
        (paddings s).encard *
            ((NetTiming.windowsCount (layOf s) (fun _ => s.window) δ (fun x => reg x ∪ cutOf s x) p.jitter sw Pw
              (faultyMost δ sw Pw) : ℕ) : ℕ∞) ≤
          ((2 ^ (p.windows * (viewCount P s + clog2 (p.jitter ^ s.window))) : ℕ) : ℕ∞)

end Properties.NetworkScan
