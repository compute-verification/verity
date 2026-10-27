import Properties.Pouw
import Properties.Pouw.Check
import Proofs.Properties.Pouw.Grid

/-!
# Proofs of PoUW's check properties (`Properties.Pouw.Check`)

By unfolding the check: `registered_ok` and `outcome_accept` say what each of its two rules accepts, and `acceptMeans`
composes them; the rest follow. `acceptedInRange_of_grid` proves `AcceptedInRange` from the window's grid properties
`PointEnumerates` and `floorPoint_greatest`, which `Proofs.Properties.Pouw.Grid` proves. `acceptMeans_complete` is the
converse of `AcceptMeans`, and `serviceReject_rejects` what `AcceptedIsTheService's` and `UndecidedOnlyFromService`
leave: the service's reject is the check's.
-/

namespace Properties.Pouw.Check

open PouwCheck

theorem publishesPublic : PublishesPublic := by
  intro pub; rfl

theorem neverNull : NeverNull := fun _ _ => ⟨nofun, nofun⟩

theorem digestIsOneStage's : DigestIsOneStage's := fun _ => rfl

theorem check_eq (p : Params) (c : Registration) (e : Evidence) :
    protocol.check p c e = PouwCheck.check p c e := rfl

theorem serviceReject_rejects (p : Params) (c : Registration) (e : Evidence) (r : String) (h : e.said = .reject r) :
    ∃ r', protocol.check p c e = .reject r' := by
  rw [check_eq]
  unfold PouwCheck.check
  split
  · exact ⟨_, rfl⟩
  · unfold outcome
    simp only [h]
    split
    · exact ⟨_, rfl⟩
    · split
      · exact ⟨_, rfl⟩
      · exact ⟨_, rfl⟩

theorem wrongItems_nil (p : Params) (items : Lean.Json) (f : Fixed) :
    wrongItems p items f = [] ↔ ∀ kv ∈ expected p f, cs (get items kv.1) = cs kv.2 := by
  unfold wrongItems
  rw [List.map_eq_nil_iff, List.filter_eq_nil_iff]
  constructor
  · intro h kv hkv
    have := h kv hkv
    simpa using this
  · intro h kv hkv
    simpa using h kv hkv

theorem registered_ok (p : Params) (c : Registration) (f : Fixed) :
    registered p c = .ok f ↔ RegistrationKeeps p c f := by
  constructor
  · intro h
    unfold registered at h
    dsimp only at h
    by_cases h1 : c.window.kind = "served" ∧ c.window.context = p.context
    case neg => rw [ite_eq_left h1] at h; cases h
    rw [ite_eq_right (Classical.not_not.mpr h1)] at h
    by_cases h2 : (keys c.window.items).getD [] = registeredSorted
    case neg => rw [ite_eq_left h2] at h; split at h <;> cases h
    rw [ite_eq_right (Classical.not_not.mpr h2)] at h
    cases hL : PouwWindow.lengthProblem (get c.window.items "length") p.maxTokens
    case some r => simp only [hL] at h; cases h
    simp only [hL] at h
    cases hl : levelOf c.window.items with
    | none => simp only [hl] at h; cases h
    | some ell =>
      simp only [hl] at h
      by_cases h3 : PouwWindow.level p.gridBits (PouwWindow.floorPoint p.gridBits p.hi) < ell + 1 ∨
          PouwWindow.point p.gridBits ell < p.lo
      · rw [ite_eq_left h3] at h; cases h
      rw [ite_eq_right h3] at h
      cases hs : PouwWindow.sampling (toRat p.eps) (toRat p.delta) (PouwWindow.Work.at p.gridBits ell)
          p.check.tileWork p.gridBits with
      | error e => simp only [hs] at h; cases h
      | ok s =>
        simp only [hs] at h
        by_cases h4 : wrongItems p c.window.items
            ⟨PouwWindow.Work.at p.gridBits ell, s, (c.roots.find? (·.port = "W")).map (·.root)⟩ = []
        case neg => rw [ite_eq_left h4] at h; cases h
        rw [ite_eq_right (Classical.not_not.mpr h4)] at h
        by_cases h5 : c.population = (PouwWindow.Work.at p.gridBits ell).n
        case neg => rw [ite_eq_left h5] at h; cases h
        rw [ite_eq_right (Classical.not_not.mpr h5)] at h
        by_cases h6 : cs c.law = cs (subsetLaw s.k)
        case neg => rw [ite_eq_left h6] at h; cases h
        rw [ite_eq_right (Classical.not_not.mpr h6)] at h
        by_cases h7 : c.query = p.query
        case neg => rw [ite_eq_left h7] at h; cases h
        rw [ite_eq_right (Classical.not_not.mpr h7)] at h
        by_cases h8 : c.scheme = p.check.scheme
        case neg => rw [ite_eq_left h8] at h; cases h
        rw [ite_eq_right (Classical.not_not.mpr h8)] at h
        cases h
        exact ⟨h1, h2, hL, hl, by simp only [PouwWindow.Work.at]; omega, by simp only [PouwWindow.Work.at]; omega, rfl,
          hs, rfl, (wrongItems_nil _ _ _).mp h4, h5, h6, h7, h8⟩
  · rintro ⟨h1, h2, hL, hl, h3, h4, hw, hs, hr, hi, hp, hlaw, hq, hsch⟩
    obtain ⟨work, samp, rootW⟩ := f
    simp only at hl h3 h4 hw hs hr hi hp hlaw
    generalize hell : work.level = ell at hl h3 h4 hw
    subst hw hr
    have hwi := (wrongItems_nil p c.window.items ⟨_, samp, _⟩).mpr hi
    unfold registered
    dsimp only
    rw [ite_eq_right (Classical.not_not.mpr h1), ite_eq_right (Classical.not_not.mpr h2)]
    simp only [hL]
    simp only [hl]
    rw [ite_eq_right (by omega)]
    simp only [hs]
    rw [ite_eq_right (Classical.not_not.mpr hwi), ite_eq_right (Classical.not_not.mpr hp),
      ite_eq_right (Classical.not_not.mpr hlaw),
      ite_eq_right (Classical.not_not.mpr hq), ite_eq_right (Classical.not_not.mpr hsch)]

theorem outcome_accept (p : Params) (c : Registration) (f : Fixed) (e : Evidence) (pub : Public)
    (prof : Core.Integrity) :
    outcome p c f e = .accept pub prof ↔
      ∃ draw, ServiceAccepted p c f e prof draw ∧ SaltKept p c e ∧
        pub = ⟨p.check, f.work, f.sampling, p.maxTokens, f.rootW, draw.units⟩ := by
  unfold outcome ServiceAccepted SaltKept
  dsimp only
  by_cases h1 : e.protocol = oneStage
  case neg => simp only [ite_eq_left h1, reduceCtorEq, h1, false_and, exists_false]
  rw [ite_eq_right (Classical.not_not.mpr h1)]
  by_cases h2 : e.commitment = some (Flock.hexOf c.digest)
  case neg => rw [ite_eq_left h2]; simp [h2]
  rw [ite_eq_right (Classical.not_not.mpr h2)]
  rcases hs : e.said with ⟨prof', draw⟩ | r | r
  case reject => simp
  case undecided => simp
  simp only [Said.accept.injEq]
  by_cases h3 : prof'.programs = #[c.program]
  case neg =>
    rw [ite_eq_left h3]; simp only [reduceCtorEq, false_iff]
    rintro ⟨d, ⟨-, -, ⟨rfl, -⟩, h, -⟩, -⟩; exact h3 h
  rw [ite_eq_right (Classical.not_not.mpr h3)]
  by_cases h4 : prof'.levels =
      #[{ name := unitLevel, query := some c.partition, size := f.work.n, draw := some (.subset f.sampling.k) }]
  case neg =>
    rw [ite_eq_left h4]; simp only [reduceCtorEq, false_iff]
    rintro ⟨d, ⟨-, -, ⟨rfl, -⟩, -, h, -⟩, -⟩; exact h4 h
  rw [ite_eq_right (Classical.not_not.mpr h4)]
  by_cases h5 : prof'.delta = p.delta
  case neg =>
    rw [ite_eq_left h5]; simp only [reduceCtorEq, false_iff]
    rintro ⟨d, ⟨-, -, ⟨rfl, -⟩, -, -, h, -⟩, -⟩; exact h5 h
  rw [ite_eq_right (Classical.not_not.mpr h5)]
  by_cases h6 : prof'.drawn = #[(unitLevel, f.sampling.k)]
  case neg =>
    rw [ite_eq_left h6]; simp only [reduceCtorEq, false_iff]
    rintro ⟨d, ⟨-, -, ⟨rfl, -⟩, -, -, -, h, -⟩, -⟩; exact h6 h
  rw [ite_eq_right (Classical.not_not.mpr h6)]
  by_cases h7 : draw.law ≠ .subset f.sampling.k ∨ draw.population ≠ f.work.n
  case pos =>
    rw [ite_eq_left h7]; simp only [reduceCtorEq, false_iff]
    rintro ⟨d, ⟨-, -, ⟨-, rfl⟩, -, -, -, -, hk, hn, -⟩, -⟩; exact h7.elim (· hk) (· hn)
  rw [ite_eq_right h7]
  by_cases h8 : UnitsOk draw.units f.sampling.k f.work.n
  case neg =>
    rw [ite_eq_left h8]; simp only [reduceCtorEq, false_iff]
    rintro ⟨d, ⟨-, -, ⟨-, rfl⟩, -, -, -, -, -, -, hu⟩, -⟩; exact h8 hu
  rw [ite_eq_right (Classical.not_not.mpr h8)]
  by_cases h9 : (windowSaltReason p c e.salt).isSome = true
  case pos =>
    rw [ite_eq_left h9]; simp only [reduceCtorEq, false_iff]
    rintro ⟨d, -, hw, -⟩; simp [hw] at h9
  rw [ite_eq_right h9]
  have hw : windowSaltReason p c e.salt = none := by
    cases h : windowSaltReason p c e.salt
    · rfl
    · simp [h] at h9
  simp only [Core.Outcome.accept.injEq]
  have hk : draw.law = .subset f.sampling.k := Classical.byContradiction fun h => h7 (.inl h)
  have hn : draw.population = f.work.n := Classical.byContradiction fun h => h7 (.inr h)
  constructor
  · rintro ⟨rfl, rfl⟩
    exact ⟨draw, ⟨h1, h2, ⟨rfl, rfl⟩, h3, h4, h5, h6, hk, hn, h8⟩, hw, rfl⟩
  · rintro ⟨d, ⟨_, _, ⟨rfl, rfl⟩, _⟩, _, rfl⟩
    exact ⟨rfl, rfl⟩

/-- The check accepts exactly when the registration keeps PoUW's rule and the service accepted it as the rule fixes, with
the window's salt. -/
theorem acceptMeans_iff (p : Params) (c : Registration) (e : Evidence) (pub : Public) (prof : Core.Integrity) :
    protocol.check p c e = .accept pub prof ↔
      ∃ f draw, RegistrationKeeps p c f ∧ ServiceAccepted p c f e prof draw ∧ SaltKept p c e ∧
        pub = ⟨p.check, f.work, f.sampling, p.maxTokens, f.rootW, draw.units⟩ := by
  rw [check_eq]
  unfold PouwCheck.check
  constructor
  · intro h
    split at h
    · cases h
    rename_i f hf
    obtain ⟨draw, hs, hw, hp⟩ := (outcome_accept _ _ _ _ _ _).mp h
    exact ⟨f, draw, (registered_ok _ _ _).mp hf, hs, hw, hp⟩
  · rintro ⟨f, draw, hk, hs, hw, hp⟩
    rw [(registered_ok _ _ _).mpr hk]
    exact (outcome_accept _ _ _ _ _ _).mpr ⟨draw, hs, hw, hp⟩

theorem acceptMeans : AcceptMeans := fun p c e pub prof => (acceptMeans_iff p c e pub prof).1

/-- Completeness of `AcceptMeans`: a registration that keeps PoUW's rule, which the service accepted as the rule fixes
with the window's salt, is accepted, publishing the rule's items and the service's draw. -/
theorem acceptMeans_complete (p : Params) (c : Registration) (e : Evidence) (f : Fixed) (prof : Core.Integrity)
    (draw : Flock.Draw.UnitDraw) (hk : RegistrationKeeps p c f) (hs : ServiceAccepted p c f e prof draw) (hw : SaltKept p c e) :
    protocol.check p c e = .accept ⟨p.check, f.work, f.sampling, p.maxTokens, f.rootW, draw.units⟩ prof :=
  (acceptMeans_iff p c e _ prof).2 ⟨f, draw, hk, hs, hw, rfl⟩

theorem acceptedIsTheService's : AcceptedIsTheService's := by
  intro p c e pub prof h
  obtain ⟨f, draw, -, ⟨hp, hc, hs, -⟩, -, rfl⟩ := acceptMeans p c e pub prof h
  exact ⟨hp, hc, draw, hs, rfl⟩

theorem acceptedDraw : AcceptedDraw := by
  intro p c e pub prof h
  obtain ⟨f, draw, ⟨-, -, -, -, -, -, hw, hs, -, -, hpop, hlaw, -⟩,
    ⟨-, -, -, -, -, -, -, -, -, hsz, hasc, hall⟩, -, rfl⟩ := acceptMeans p c e pub prof h
  refine ⟨hw, ?_, hpop, hlaw, hsz, hasc, hall⟩
  unfold PouwWindow.sampling at hs
  revert hs
  cases PouwWindow.kPrime (toRat p.eps) (toRat p.delta) (↑f.work.credited * ↑p.check.tileWork) ↑f.work.n
    ↑p.check.tileWork with
  | error e => intro hs; cases hs
  | ok k =>
    intro hs
    simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hs
    rw [← hs]

theorem ite_reject_eq_undecided {c : Prop} [Decidable c] {a r : String}
    {x : Core.Outcome Public Core.Integrity}
    (h : (if c then .reject a else x) = .undecided r) : x = .undecided r := by
  by_cases hc : c
  · rw [ite_eq_left hc] at h; cases h
  · rwa [ite_eq_right hc] at h

theorem undecidedOnlyFromService : UndecidedOnlyFromService := by
  intro p c e r h
  rw [check_eq] at h
  unfold PouwCheck.check at h
  split at h
  · cases h
  rename_i f hf
  unfold outcome at h
  dsimp only at h
  by_cases h1 : e.protocol = oneStage
  case neg => rw [ite_eq_left h1] at h; cases h
  rw [ite_eq_right (Classical.not_not.mpr h1)] at h
  by_cases h2 : e.commitment = some (Flock.hexOf c.digest)
  case neg => rw [ite_eq_left h2] at h; cases h
  rw [ite_eq_right (Classical.not_not.mpr h2)] at h
  refine ⟨⟨f, (registered_ok _ _ _).mp hf⟩, h1, h2, ?_⟩
  cases hs : e.said with
  | accept prof draw =>
    simp only [hs] at h
    repeat have h := ite_reject_eq_undecided h
    cases h
  | reject r₀ => simp only [hs] at h; cases h
  | undecided r₀ => simp only [hs] at h; cases h; exact ⟨r₀, rfl, rfl⟩

/-- `AcceptedInRange` from the window's grid: the credited tiles are at least `lo` by the level's rule, and the padded
count `point (level + 1)` is at most the floor point of `hi`, since `point` increases and inverts `level` on the grid. -/
theorem acceptedInRange_of_grid (hp : Window.PointEnumerates) : AcceptedInRange := by
  intro p c e pub prof h
  obtain ⟨f, draw, ⟨-, -, -, -, h3, h4, hw, -⟩, -, -, rfl⟩ := acceptMeans p c e pub prof h
  dsimp only
  rw [hw]
  refine ⟨h4, ?_⟩
  simp only [PouwWindow.Work.at]
  obtain ⟨-, hlt, -, -, hinv⟩ := hp p.gridBits
  have mono : ∀ a b, a ≤ b → PouwWindow.point p.gridBits a ≤ PouwWindow.point p.gridBits b := by
    intro a b hab
    induction b with
    | zero => rw [Nat.le_zero.mp hab]; exact Nat.le_refl _
    | succ b ih =>
      rcases Nat.lt_or_eq_of_le hab with hab | hab
      · exact Nat.le_trans (ih (Nat.le_of_lt_succ hab)) (Nat.le_of_lt (hlt b))
      · rw [hab]; exact Nat.le_refl _
  by_cases hhi : 1 ≤ p.hi
  · obtain ⟨hpt, hle, -⟩ := Window.floorPoint_greatest p.gridBits p.hi hhi
    have := mono _ _ h3
    rw [hinv _ hpt] at this
    omega
  · have h0 : p.hi = 0 := by omega
    rw [h0] at h3
    simp only [PouwWindow.floorPoint, PouwWindow.level, Nat.zero_div, Nat.zero_mul] at h3
    rw [ite_eq_left (Nat.two_pow_pos _)] at h3
    omega

theorem acceptedInRange : AcceptedInRange :=
  acceptedInRange_of_grid Window.pointEnumerates

theorem commitmentKeepsBytes : CommitmentKeepsBytes := by
  intro b c h
  change registrationOf b = .ok c at h
  unfold registrationOf at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' (first | (split at h) | (cases h; rfl) | cases h)

end Properties.Pouw.Check
