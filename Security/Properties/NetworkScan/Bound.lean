module

public import Properties.NetworkScan.Chain
public import Mathlib.Algebra.BigOperators.Fin

/-!
# The scans' count under the period check's `K`

`NetTiming.Guarantees.SampledLeakChecked` counts a statement's accepted sequences at
`2^#(freeInputs \ (reg ∪ chk)) · #faultySets(δ) · 2^β · Dj^k`, `β` the most committed bits of a possible incorrect
set. `ScanCountBounded` reads that count at a rows-form period's link-windows, each one's statement the scan's Program
flattened, with the scan's state gates as the checked inputs (`stateGates`, which `ScanChainPinned` says the state
chain pins), and bounds the period's product of them by `2 ^ K` at the `K` the period check publishes.

It carries three conditions. Every unit of each scan is in a class sampled `all`, so no incorrect set of its units is
possible but the empty one and the count has no faulty term. The scan's row, state and egress arguments read input
nodes of their own (`argsApart`), so its free inputs off the state gates are its slot entries (`ScanCountsEntries`).
And each link-window's published term charges at least the slot entries' bits (`entryBits`) as its free inputs: the
check charges a link-window its tied audits' free inputs (`NetworkCheck.Period.rowsTerms`), which it does not compare
with the scan's entries, so the statement takes the comparison as a condition on the check's public output.

The schedules' and cells' factors of a period's count enter at the check's terms for them, `2 ^ (schedules + cells)`,
which `Properties.NetworkPeriod.RowsKBoundsCount` shows they are at most, and the jitter classes at the period's.
-/

@[expose] public section

namespace Properties.NetworkScan

open NetworkCheck.Period

/-- **The scans' count is at most `2 ^ K` on the period check's acceptance.** If the check accepts a period in the
rows form at a schedule `s` it reads (`scheduleOf`) and the scan `j` that `scanOf` accepts as the period's registered
Program, whose canonical text `cs` decodes to `P` with its row, state and egress arguments reading input nodes of
their own (`argsApart`); each link-window's statement `sw i` has `cs` as its descriptor and `P` flattened as its
gates; every proof block of each is in a class sampled `all`; and each link-window's published term charges at least
the slot entries' bits as its free inputs: then `2 ^ (schedules + cells)` times the product over the link-windows of
`2^#(freeInputs \ (reg ∪ stateGates)) · #faultySets(δ) · 2^β · jitter^T`, `β` the most committed bits of a possible
incorrect set, is at most `2 ^ K`, at every registered inputs `reg` and every `δ`. -/
def ScanCountBounded : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (pub : Public) (prof : Profile),
    check p c e = .accept pub prof → ∀ (s : Schedule) (sd : ByteArray) (scan : Scan), p.form = .rows s sd scan →
    ∀ (sj j : Lean.Json) (cs : String) (P : Flock.Program.Prog),
      scheduleOf sj = .ok s → scanOf j p.program s = .ok scan → Flock.canon j = .ok cs →
      Flock.Program.decode cs.toUTF8 = .ok P → argsApart P s = true →
    ∀ (sw : Fin p.windows → Claims.Sampling.AuditStatement) (Pw : Fin p.windows → Claims.Sampling.Partition)
      (reg : Claims.Sampling.AuditStatement → Finset ℕ) (δ : ℝ),
      (∀ i, (⟨(sw i).desc.toArray⟩ : ByteArray) = cs.toUTF8 ∧ Flattens P (sw i).prog) →
      (∀ i, Finset.range (Pw i).blocks.length ⊆ NetTiming.allSlots (sw i).design (Pw i)) →
      (∀ t ∈ pub.units, entryBits P s ≤ t.free) →
        2 ^ (pub.schedules + pub.cells) *
          ∏ i, (2 ^ (NetTiming.freeInputs (sw i).prog (layOf s (sw i)) s.window \
                (reg (sw i) ∪ stateGates P s)).card *
              (NetTiming.faultySets (sw i).design (Pw i) δ).card *
              2 ^ ((NetTiming.faultySets (sw i).design (Pw i) δ).sup (NetTiming.faultyBits (sw i).prog (Pw i))) *
              p.jitter ^ s.window) ≤ 2 ^ pub.K

end Properties.NetworkScan
