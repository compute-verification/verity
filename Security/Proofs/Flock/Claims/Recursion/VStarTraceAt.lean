import Proofs.Flock.Claims.Recursion.VStarNets
import Proofs.Flock.Claims.Recursion.SlotsDeployed
import Proofs.Flock.C3.PubRows.Shape

/-!
# `TraceOkAt` and `TraceAt` at the deployed V\*'s staged parts

`HpEarly.TraceOkAt` asks for `TraceOk` at each part's circuit and its fitted port inputs (`fitPin`). At a staged part
the circuit's fields are the verifier's checks (`VStarTraceOk.traceOk_refs`), its three nets are V\*'s file
(`VStarNets.nets_three`), `fitPin` keeps the stager's refs (`refs_fitPin`), and it fits every row to its port's words
and a 192-byte salt (`rows_fitPin`), which reads each port's words by its name: the setup's `checkStmt` refuses a circuit
naming a port twice (`names_of_checkStmt`). So the only hypothesis left is that every part is staged, the one
`SlotsDeployed.circOkAt_staged` takes, and the two give `TraceAt` (`traceAt_of`).
-/

namespace RecDeploy.VStarTraceAt

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild
open RecDeploy.VStarTrace (TraceOk)

/-- **A staged part's circuit names each port once**: the setup's `checkStmt` on it passed (`names_of_checkStmt`). -/
theorem names_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : VStar.Stmt} {c : Circuit}
    (hp : partAt x s = some (sv, c)) : ∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q' := by
  obtain ⟨r, hr, -, hc⟩ := VStarTraceOk.partAt_some hp
  obtain ⟨p, r', _, hx, ht, hpi, hr', _, h0, _⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hpc := partCircuit_staged hS hx ht hpi hr hc h0
  subst hpc
  obtain ⟨ms, i, spec, sts, rfl, hsu, hck⟩ := hS
  have hpx := parse_of_setupO hsu
  rw [hx] at hpx
  cases hpx
  have h0' : 0 < sts.size := h0
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r0, -, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hC0 := checks_stmt hC 0 h0'
  obtain ⟨j, f, sv0, hj, hf, -⟩ := checkStmt_circuit hC0
  rw [hj, hf] at hC0
  exact FlockC3.PubRows.names_of_checkStmt hC0 rfl

/-- Where each port is named once, a port's name gives its words. -/
theorem portWords_named {c : Circuit} (hn : ∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q') :
    ∀ q (hq : q < c.ports.size), portWords c c.ports[q].name = c.ports[q].words := by
  intro q hq
  cases hf : c.ports.findIdx? (·.name == c.ports[q].name) with
  | none =>
    rw [Array.findIdx?_eq_none_iff] at hf
    have := hf c.ports[q] (Array.getElem_mem hq)
    simp at this
  | some q' =>
    obtain ⟨hq', hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hf
    have he : q' = q := congrArg Fin.val (hn ⟨q', hq'⟩ ⟨q, hq⟩ (by simpa using hp))
    subst he
    exact portWords_of hf hq'

/-- **`TraceOk` at a staged part's fitted port inputs**, with no hypothesis on the stager's rows. -/
theorem traceOk_fitPin {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) {src : VStar.In.Src} {pin : VStar.In.PartIn}
    (hi : VStar.In.partIn src (progOf c) sv = .ok pin) : TraceOk c (fitPin c pin) :=
  VStarTraceOk.traceOk_refs hS hp (VStarNets.nets_three hS hp) (refs_fitPin (VStarTraceOk.partIn_refs hi))
    (rows_fitPin c pin (portWords_named (names_staged hS hp)))

/-- **`TraceOkAt` at the deployed V\***: every part of `x` staged. -/
theorem traceOkAt_staged {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {h : List (Msg × Msg)}
    {st : Stage} (hS : ∀ s sv c, partAt x s = some (sv, c) → ∃ xO, StagedPart st s x xO) :
    HpEarly.TraceOkAt I x w ρ h := by
  intro s sv c src pin hp _ hi
  obtain ⟨xO, hSt⟩ := hS s sv c hp
  exact traceOk_fitPin hSt hp hi

/-- **`TraceAt` at the deployed V\***: every part of `x` staged (`traceOkAt_staged`, `circOkAt_staged`). -/
theorem traceAt_staged {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape} {h : List (Msg × Msg)}
    {st : Stage} (hS : ∀ s sv c, partAt x s = some (sv, c) → ∃ xO, StagedPart st s x xO) (mask : ℕ → ℕ → Bool) :
    HpEarly.TraceAt mask I x w ρ h :=
  HpEarly.traceAt_of (traceOkAt_staged hS) (SlotsDeployed.circOkAt_staged hS) mask

end RecDeploy.VStarTraceAt
