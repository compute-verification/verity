module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressFacts.RowFirst

/-!
# The two-stage premises at any registration, from its decided facts

`Honest.Instance` and `Honest.Sits` state what `SpotKSTwo` and `SpotExfilTwo` take at one registration built for
them (`held R B desc q c₀`: one template, no wires). Neither conclusion reads that shape. At any auditor's copies
`h`:

* `scope_of_two`: a statement that parses is in the two-stage scope at the compiled code where the design evaluates
  (`two`). Each wire's writer is then first (`wiresFirst_of_two`), so the code's wiring map rewires the members'
  programs (`rewires_of_two`), and a rewiring keeps the two-stage scope at the wired program, the partition and the
  design's replay partition cut (`scope_rewired`);
* `inScope_of_two`, `inScope_of_run`: so some statement is in scope where the design evaluates and a statement
  decodes, and where a run is accepted, whose design evaluated (`CheckedRun.Ok`);
* `accepts_of`: what an accepted run takes, the fields `CheckedRun.Ok` derives left out: the statement check
  passing, one unit the two stages check, and a witness in Flock's relation at that unit's session statement;
* `premises_of_checks`: the premises, where the frame is SPOT's own at the wired program (`BuiltWired`), from five
  facts of the auditor's copies (`addressesConnected_of_firstK`): the reference reads are well formed, no proof
  block is always checked, the boundary registration passes the run's population check, its first stage samples a
  uniform `K`-subset, and the condition decided on the registered wires holds;
* `honest_of_run`: `HonestExists` from those five facts and one accepted run.

So a registration with wires, several templates and several replay blocks is an instance of `HonestExists` where a
run of it is accepted: the design's own decided conditions (`scopeOk`, `filesOk`, `templateOk`, `membersScoped`,
`wiringOk`) are not asked again, since the accepted run's design evaluated.

## What this does not show

* An accepted run (`Accepts`). `CheckedRun.Ok` holds a witness that Flock's relation holds of a checked unit's
  session statement, which no check of the verifier's gives: it is the prover's.
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot

/-- **A statement that parses is in the two-stage scope at the compiled code, where the design evaluates**, at any
auditor's copies: the statement at the members' programs wired by the code's wiring map, its partition and its
design's replay partition cut (`scope_rewired`), the map rewiring the members' programs (`rewires_of_two`). -/
theorem scope_of_two {h : Held} {x : Msg} {s : AuditStatement} {d : Design.Two} (hp : parse h x = some s)
    (h2 : two h s = some d) : (compiled h).Scope x := by
  obtain ⟨l, -, -, -, hsc, -⟩ := two_eq h h2
  have hP : partition h s = some (membersPart h.ms) := rfl
  have hW := rewires_of_two h2 (wiresFirst_of_two h h2)
  have hs := scope_rewired h hp hP hW (scopeOk_sound hsc s (parse_prog h hp))
  rw [parse_prog h hp] at hs
  refine ⟨_, _, _, ?_, ?_, ?_, hs⟩
  · rw [parse_compiled]
    unfold parseWired
    rw [hp]
    rfl
  · rw [partition_compiled]
    rfl
  · rw [two_compiled]
    show (two h s).map (fun d => ({ d with part := d.part.cut (wireMap h) } : Design.Two)) = _
    rw [h2]
    rfl

/-- **Some statement is in the two-stage scope where the design evaluates**: any statement that decodes. -/
theorem inScope_of_two {h : Held} {x : Msg} {st : Roots} (hx : decode x = some st) {s : AuditStatement}
    {d : Design.Two} (h2 : two h s = some d) : InScope h := by
  obtain ⟨s₁, hs₁⟩ : ∃ s₁, parse h x = some s₁ := by
    unfold parse
    rw [hx]
    exact ⟨_, rfl⟩
  have h2' : two h s₁ = some d := h2
  exact ⟨x, scope_of_two hs₁ h2'⟩

/-- **An accepted run's statement is in the two-stage scope at the compiled code**, at any auditor's copies: the
run's design evaluated. -/
theorem inScope_of_run {h : Held} (hA : Accepts h) : InScope h := by
  obtain ⟨r, Pt, d, spec, hp, -, h2, -⟩ := hA
  exact ⟨r.x, scope_of_two hp h2⟩

/-- **What an accepted run takes**: a statement that parses where the design evaluates, the statement check passing
its records at some coins, one unit the two stages check there, and a witness in Flock's relation at that unit's
session statement. The unit's proof block, its replay block and its session's setup follow, as `stmtOpensWired`
reads them. -/
theorem accepts_of {h : Held} {x m₁ m₂ : Msg} {s : AuditStatement} {d : Design.Two} {c₀ : FlockDeploy.Coin}
    {c₁ : Stage h} {wF : Claims.Flock.Wit} {u : ℕ} (hp : parse h x = some s) (h2 : two h s = some d)
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true) :
    Accepts h := by
  obtain ⟨l, -, -, -, hsc, -⟩ := two_eq h h2
  have hP : partition h s = some (membersPart h.ms) := rfl
  have hin : InScopeTwo (membersPart h.ms) d s := scopeOk_sound hsc s (parse_prog h hp)
  have hlt : u < (membersPart h.ms).blocks.length := by
    have := mem_checked_lt hu
    rwa [units_base h hp hP] at this
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u hlt)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  obtain ⟨ins, spec, sts, hset⟩ :
      ∃ ins spec sts, FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) := by
    have hs := sessionOk_of_stmtOkAt h (stmtOkAt_unit h hp hok hu)
    simp only [sessionOk, Bool.and_eq_true] at hs
    obtain ⟨ins, spec, sts, hset, -⟩ := zkOk_sound hs.1.1.1
    exact ⟨ins, spec, sts, hset⟩
  have hr : CheckedRun.Ok h
      ⟨x, s, [], m₁, m₂, c₀, c₁, [], wF, u, (membersPart h.ms).blocks[u], ρ, rb, ins, sts⟩ :=
    ⟨membersPart h.ms, d, spec, hp, hP, h2, hin, hok, hu, hck, List.getElem?_eq_getElem hlt, hρ, hrb, hset⟩
  exact ⟨_, hr⟩

/-- **The premises `SpotKSTwo` takes, at any auditor's copies whose frame is SPOT's own at the wired program**, from
five facts of the copies: the reference reads are well formed, no proof block is always checked, the boundary
registration passes the run's population check at `n` units, its first stage samples a uniform `K`-subset, and the
condition decided on the registered wires holds (`addressesConnected_of_firstK`). -/
theorem premises_of_checks {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) : Premises h :=
  ⟨hB, addressesConnected_of_firstK hWf hna hp hK hwc⟩

/-- **`HonestExists`, from five facts of the auditor's copies and one accepted run**: the copies meet the premises
(`premises_of_checks`), and the accepted run's statement is in scope (`inScope_of_run`). -/
theorem honest_of_run {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) (hA : Accepts h) :
    HonestExists :=
  ⟨h, premises_of_checks hB hWf hna hp hK hwc, inScope_of_run hA, hA⟩

end SpotFlock.Honest
