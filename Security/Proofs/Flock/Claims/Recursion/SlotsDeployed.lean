import Proofs.Flock.Claims.Recursion.SlotsOk
import Proofs.Flock.Claims.Recursion.VStarNets

/-!
# `TraceAt` at the staged parts (`traceAt_staged`)

`slotsOk_assign` takes three facts of a part's circuit beyond `TraceOk`. Each holds at every staged part
(`StagedPart`), from the verifier's own checks of the staged table, which is part `s`'s circuit (`partCircuit_staged`):

* `HmNets.check c = .ok ()`, and every v1 port unsegmented with `2 · words = 128 · chunks`: `HmRow.parse` runs the
  first and refuses the others (`ParseFacts`: `rowNets`, `v1Segs`, `portWords`; `parseFacts_of_parse`);
* `c.kLog ≤ 32`: the setup refuses `k_log > 27` (`table0`);
* every port v1: V\*'s parse check refuses a `sha512/row/v2` port (`checkParsed_v1`). V\*'s META writes `row_schema` and
  never `row_schemas` (`VStar/Compose.lean`, `metaHead`), so every deployed V\* port is v1 and the honest table passes.

With CAPLOWER's `traceOk_staged`, `TraceAt` holds once every part is staged and its port inputs' rows are the honest
prover's (`RowsOk`).
-/

namespace RecDeploy.SlotsDeployed

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild
open RecDeploy.HpLast

/-- **`slotsOk_assign`'s three premises at a staged part**: the verifier's hm96 nets, `kLog ≤ 27`, and every port a
`sha512/row/v1` row of whole 128-byte chunks. -/
theorem slotFacts_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : _root_.VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) :
    _root_.Flock.HmNets.check c = .ok () ∧ c.kLog ≤ 32 ∧ ∀ q (hq : q < c.ports.size), c.ports[q].v2 = false ∧
      c.ports[q].segs = none ∧ 2 * c.ports[q].words = 128 * c.ports[q].chunks := by
  obtain ⟨r, hr, -, hc⟩ := VStarTraceOk.partAt_some hp
  obtain ⟨p, r', -, hx, ht, hpi, hr', -, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hpc := partCircuit_staged hS hx ht hpi hr hc h0
  obtain ⟨tags, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
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

/-- **`SlotsOk` at a staged part**, at any mask, from `TraceOk`. -/
theorem slotsOk_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : _root_.VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) {pin : _root_.VStar.In.PartIn} (h : VStarTrace.TraceOk c pin)
    (mask : ℕ → ℕ → Bool) : HpEarly.SlotsOk mask c pin :=
  let ⟨hck, hk, hv1⟩ := slotFacts_staged hS hp
  HpEarly.slotsOk_assign h hck hk hv1 mask

/-- **`TraceAt` at the deployed statements**: when every part of `x` is staged and its port inputs' rows are the
honest prover's (`RowsOk`), `TraceOk` (`traceOk_staged`) and `SlotsOk` (`slotsOk_staged`) hold at each. -/
theorem traceAt_staged {mask : ℕ → ℕ → Bool} {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape}
    {h : List (Msg × Msg)} {st : Stage}
    (hS : ∀ s sv c, partAt x s = some (sv, c) → ∃ xO, StagedPart st s x xO)
    (hR : ∀ s sv c src pin, partAt x s = some (sv, c) → srcOf I x w ρ h = some src →
      _root_.VStar.In.partIn src (progOf c) sv = .ok pin → VStarTraceOk.RowsOk c pin) :
    HpEarly.TraceAt mask I x w ρ h := by
  intro s sv c src pin hp hsrc hi
  obtain ⟨xO, hSt⟩ := hS s sv c hp
  have hT := VStarNets.traceOk_staged hSt hp hi (hR s sv c src pin hp hsrc hi)
  exact ⟨hT, slotsOk_staged hSt hp hT mask⟩

end RecDeploy.SlotsDeployed
