module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressFacts.RowFirst

/-! The two-stage premises at any registration from its decided facts (`honest_of_run`); a play the verifier accepts
and a witness in Flock's relation are not shown, because only an honest session and the prover give them. -/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy Properties.Spot

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

theorem inScope_of_two {h : Held} {x : Msg} {st : Roots} (hx : decode x = some st) {s : AuditStatement}
    {d : Design.Two} (h2 : two h s = some d) : InScope h := by
  obtain ⟨s₁, hs₁⟩ : ∃ s₁, parse h x = some s₁ := by
    unfold parse
    rw [hx]
    exact ⟨_, rfl⟩
  have h2' : two h s₁ = some d := h2
  exact ⟨x, scope_of_two hs₁ h2'⟩

theorem scope_of_play {h : Held} (H : Msg → Msg) {Ω : Type} {P : Claims.Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin ((V h).k x) → FlockDeploy.Coin} (hacc : (Claims.run H P (V h) x ω r).out = .acc) :
    (compiled h).Scope x := by
  obtain ⟨s, Pt, d, hp, -, h2, -⟩ := refusesTwo h H x Ω P ω r hacc
  have hp' : parse h x = some s := hp
  exact scope_of_two hp' h2

theorem inScope_of_play {h : Held} (hA : AcceptsPlay h) : InScope h := by
  obtain ⟨Ω, P, x, ω, r, hacc⟩ := hA
  exact ⟨x, scope_of_play FlockDeploy.sha512 hacc⟩

theorem honestExists_iff : HonestExists ↔ ∃ h : Held, Premises h ∧ AcceptsPlay h := by
  constructor
  · rintro ⟨h, hP, -, hA⟩
    exact ⟨h, hP, hA⟩
  · rintro ⟨h, hP, hA⟩
    exact ⟨h, hP, inScope_of_play hA, hA⟩

theorem inScope_of_run {h : Held} (hA : Accepts h) : InScope h := by
  obtain ⟨r, Pt, d, spec, hp, -, h2, -⟩ := hA
  exact ⟨r.x, scope_of_two hp h2⟩

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

theorem premises_of_checks {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) : Premises h :=
  ⟨hB, addressesConnected_of_firstK hWf hna hp hK hwc⟩

theorem honest_of_run {h : Held} (hB : BuiltWired h (wireMap h) (returned h)) (hWf : RefReadsWf h)
    (hna : ∀ v, h.reg.always v = false) {n K : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hK : firstK h.reg.first = some K) (hwc : AddressRow.wiresChecked h = true) (hA : AcceptsPlay h) :
    HonestExists :=
  ⟨h, premises_of_checks hB hWf hna hp hK hwc, inScope_of_play hA, hA⟩

end SpotFlock.Honest
