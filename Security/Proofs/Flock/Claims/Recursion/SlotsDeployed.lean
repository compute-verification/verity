import Proofs.Flock.Claims.Recursion.HpEarlyTrace
import Proofs.Flock.Claims.Recursion.VStarTraceOk

/-!
# `CircOkAt` at the staged parts (`circOkAt_staged`)

`HpEarly.CircOkAt x` is what `slotsOk_assign` reads of each part's circuit besides `TraceOk`. It holds at every staged
part (`StagedPart`), from the verifier's own checks of the staged table, which is part `s`'s circuit
(`partCircuit_staged`):

* `HmNets.check c = .ok ()`, and every v1 port unsegmented with `2 · words = 128 · chunks`: `HmRow.parse` runs the
  first and refuses the others (`ParseFacts`: `rowNets`, `v1Segs`, `portWords`; `parseFacts_of_parse`);
* `c.kLog ≤ 32`: the setup refuses `k_log > 27` (`table0`);
* every port v1: V\*'s parse check refuses a `sha512/row/v2` port (`checkParsed_v1`). V\*'s META writes `row_schema` and
  never `row_schemas` (`VStar/Compose.lean`, `metaHead`), so every deployed V\* port is v1 and the honest table passes.
-/

namespace RecDeploy.SlotsDeployed

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild

/-- **`slotsOk_assign`'s three premises at a staged part**: the verifier's hm96 nets, `kLog ≤ 27`, and every port a
`sha512/row/v1` row of whole 128-byte chunks. -/
theorem slotFacts_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : _root_.VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) :
    _root_.Flock.HmNets.check c = .ok () ∧ c.kLog ≤ 32 ∧ ∀ q (hq : q < c.ports.size), c.ports[q].v2 = false ∧
      c.ports[q].segs = none ∧ 2 * c.ports[q].words = 128 * c.ports[q].chunks := by
  obtain ⟨r, hr, -, hc⟩ := VStarTraceOk.partAt_some hp
  obtain ⟨p, r', sv', hx, ht, hpi, hr', -, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hpc := partCircuit_staged hS hx ht hpi hr hc h0
  obtain ⟨tags, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0).2.2)
  have hv := FlockSoundness.Discharge.Recursion.checkParsed_v1 (htab 0 h0).2.2
  have hk : p.sts[0].c.kLog ≤ 27 := by
    have hx' := hx
    unfold Flock.Readable.parseBytes at hx'
    obtain ⟨rd, -, hrd⟩ := Option.bind_eq_some_iff.1 hx'
    obtain ⟨st0, h0', -, -, -, -, hk, -⟩ :=
      Flock.Readable.table0 (Flock.Readable.tablesOf_of_parses (Flock.Readable.parses_of_parse hrd)) ht hpi
    rw [Array.getElem?_eq_getElem h0] at h0'
    cases h0'
    exact hk
  rw [hpc] at P hv hk
  refine ⟨P.rowNets, by omega, fun q hq => ?_⟩
  have hm : c.ports[q] ∈ c.ports.toList := Array.mem_toList_iff.mpr (Array.getElem_mem hq)
  have hv2 := hv _ hm
  exact ⟨hv2, P.v1Segs _ hm hv2, P.portWords _ hm hv2⟩

/-- **`CircOkAt` at the deployed V\***: when every part of `x` is staged, each part's circuit meets
`slotFacts_staged`. -/
theorem circOkAt_staged {st : Stage} {x : Msg}
    (hS : ∀ s sv c, partAt x s = some (sv, c) → ∃ xO, StagedPart st s x xO) : HpEarly.CircOkAt x := by
  intro s sv c hp
  obtain ⟨xO, hSt⟩ := hS s sv c hp
  exact slotFacts_staged hSt hp

end RecDeploy.SlotsDeployed
