module

public import Proofs.Properties.Verity.SampledTwo.OfRun
public import Proofs.Flock.SpotFlock.RunsAs.Released
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# A stage's sample reads no byte past its source

`readsWithin`: `ReadsWithin` (`OfRun.lean`) at every held copy whose first-stage registration has no closure map. A
sample takes its bytes in order (`Draw.take`), which throws where the source runs out, so a sample that ends at position
`R` ends within the source: `drawUnits_within`, and through the stages' samples, `select_within` (with no closure map)
and `selectSecond_within`. The walk is `RunsAs.Released`'s, step for step, with "the position stays within the source"
in place of "the same on any source with the same prefix".

The form's run of record has no closure map (`closure_of_formAccepts`: `inputOf` sets none), and the held copies an
accepted play's run reads carry the run's (`closure_of_held`), so `playAccepts_of_run'` is `playAccepts_of_run` without
the `ReadsWithin` premise: where the held registration had a closure map, `HeldAt` would leave no play accepted.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open Lean (Json)
open Properties.Spot.SecurityProofs (bind_error bind_eq_ok throw_bind_ne onWith_take StopsAt)
open SpotFlock.RunsAs (run_pure_eq HeldMatches RoundTwo)
open SpotFlock.RunsAs.Round2 (NamesDistinct TwoHolds)

/-! ## The walk -/

theorem pure_within {α : Type} {a : α} {s B : ℕ} {x : α × ℕ}
    (h : (pure a : StateT ℕ (Except String) α).run s = .ok x) (hs : s ≤ B) : x.2 ≤ B := by
  obtain rfl := run_pure_eq h
  exact hs

/-- A bind whose first step ends within `B`, and whose second, from within `B`, ends within `B`, ends within `B`. -/
theorem bind_within {α β : Type} {x : StateT ℕ (Except String) α} {f : α → StateT ℕ (Except String) β}
    {s B : ℕ} {out : β × ℕ} (h : (x >>= f).run s = .ok out)
    (hx : ∀ r, x.run s = .ok r → r.2 ≤ B)
    (hf : ∀ r, x.run s = .ok r → r.2 ≤ B → (f r.1).run r.2 = .ok out → out.2 ≤ B) : out.2 ≤ B := by
  rw [StateT.run_bind] at h
  cases hxr : x.run s with
  | error e => rw [hxr] at h; exact absurd h (by rw [bind_error]; simp)
  | ok r =>
    rw [hxr, Properties.Spot.SecurityProofs.bind_ok] at h
    exact hf r hxr (hx r hxr) h

/-- A `for` loop over a list whose body, from within `B`, ends within `B`, does so too. -/
theorem forIn_within {α β : Type} (f : α → β → StateT ℕ (Except String) (ForInStep β)) (B : ℕ)
    (hf : ∀ a b s x, s ≤ B → (f a b).run s = .ok x → x.2 ≤ B) :
    ∀ (l : List α) (b : β) (s : ℕ) (x : β × ℕ), s ≤ B → (forIn l b f).run s = .ok x → x.2 ≤ B := by
  intro l
  induction l with
  | nil =>
    intro b s x hs h
    rw [List.forIn_nil] at h
    exact pure_within h hs
  | cons a as ih =>
    intro b s x hs h
    rw [List.forIn_cons] at h
    refine bind_within h (fun r hr => hf a b s r hs hr) (fun r _ hr hk => ?_)
    obtain ⟨step, s₁⟩ := r
    cases step with
    | done b' => exact pure_within hk hr
    | yield b' => exact ih b' s₁ x hr hk

/-- **A sampler's reads end within the source**: `onWith` on `take` ends at its last read's end, which `take` gave. -/
theorem onWith_within {ξ : ByteArray} {N : ℕ} (hN : 0 < N) {run : ByteArray → Option (Array ℕ)} {p : ℕ}
    {x : Option (Array ℕ) × ℕ}
    (h : (Flock.Draw.onWith (fun need => Spot.Draw.take ξ need) N run).run p = .ok x) : x.2 ≤ ξ.size := by
  obtain ⟨u, p'⟩ := x
  obtain ⟨j, D, -, rfl, hs⟩ := onWith_take ξ N hN run p p' u h
  unfold StopsAt at hs
  exact hs.2.1

theorem stratifiedWith_within {ξ : ByteArray} {strata : Array Flock.Draw.Stratum} {p : ℕ}
    {x : Option (Array ℕ) × ℕ} (hp : p ≤ ξ.size)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Spot.Draw.take ξ need) strata).run p = .ok x) :
    x.2 ≤ ξ.size := by
  unfold Flock.Draw.stratifiedWith at h
  try dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  refine bind_within h (fun r hr => forIn_within _ ξ.size (fun i b s z _ hz => ?_) _ _ _ _ hp hr)
    (fun r _ hs hk => by
      first
        | exact pure_within hk hs
        | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))
  exact bind_within hz (fun r hr => onWith_within (by unfold Flock.Draw.Stratum.need; omega) hr)
    (fun r _ hs hk => by
      first
        | exact pure_within hk hs
        | (revert hk; (repeat' split) <;> intro hk <;> exact pure_within hk hs))

theorem drawWith_within {ξ : ByteArray} (law : Flock.Draw.Law) (n : ℕ) {p : ℕ}
    {x : Option Flock.Draw.UnitDraw × ℕ} (hp : p ≤ ξ.size)
    (h : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take ξ need) law n).run p = .ok x) : x.2 ≤ ξ.size := by
  cases law with
  | subset k =>
    unfold Flock.Draw.drawWith at h
    try dsimp only at h
    exact bind_within h (fun r hr => onWith_within (by omega) hr)
      (fun r _ hs hk => by exact pure_within hk hs)
  | bernoulli a b =>
    unfold Flock.Draw.drawWith at h
    try dsimp only at h
    exact bind_within h (fun r hr => onWith_within (by omega) hr)
      (fun r _ hs hk => by exact pure_within hk hs)
  | stratified k strata =>
    unfold Flock.Draw.drawWith at h
    try dsimp only at h
    exact bind_within h (fun r hr => stratifiedWith_within hp hr)
      (fun r _ hs hk => by exact pure_within hk hs)
  | work k strata =>
    unfold Flock.Draw.drawWith at h
    try dsimp only at h
    exact bind_within h (fun r hr => stratifiedWith_within hp hr)
      (fun r _ hs hk => by exact pure_within hk hs)

/-- **A sample reads no byte past its source**: the units a sampling design samples from `ξ`, having read `r.2` bytes, read them within
`ξ`. -/
theorem drawUnits_within {ξ : ByteArray} {law : Flock.Draw.Law} {n : ℕ} {r : Array ℕ × ℕ}
    (h : Spot.Draw.drawUnits law n ξ = .ok r) : r.2 ≤ ξ.size := by
  unfold Spot.Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take ξ need) law n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨o, rd⟩ := y
    rw [hw] at h
    cases o with
    | none => cases h
    | some d =>
      obtain rfl := (Except.ok.inj h).symm
      have hb : rd ≤ ξ.size := drawWith_within law n (Nat.zero_le _) hw
      exact hb

theorem drawFirstRead_within {law : Json} {n : ℕ} {ξ : ByteArray} {r : Json × ℕ}
    (h : Spot.Draw.drawFirstRead law n ξ = .ok r) : r.2 ≤ ξ.size := by
  unfold Spot.Draw.drawFirstRead at h
  obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
  obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
  obtain ⟨u, rd⟩ := ur
  have hr : r.2 = rd := congrArg Prod.snd (Except.ok.inj h).symm
  have hb : rd ≤ ξ.size := drawUnits_within hu
  rw [hr]
  exact hb

/-- **The second stage reads no byte past its source.** -/
theorem selectSecond_within {sample ns ks : Array ℕ} {ξ : ByteArray} {r : Json × ℕ}
    (h : Spot.Select.selectSecond sample ns ks ξ = .ok r) : r.2 ≤ ξ.size := by
  unfold Spot.Select.selectSecond at h
  cases hq : Spot.Select.secondProblem sample ns ks with
  | some e =>
    try rw [hq] at h
    try dsimp only at h
    exact absurd h (throw_bind_ne _ _ _)
  | none =>
    try rw [hq] at h
    try dsimp only at h
    obtain ⟨dr, hd, h⟩ := bind_eq_ok.1 h
    obtain ⟨d, rd⟩ := dr
    have hr : r.2 = rd := by
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact congrArg Prod.snd (Except.ok.inj h).symm
    have hb : rd ≤ ξ.size := drawFirstRead_within hd
    rw [hr]
    exact hb

/-- **The first stage reads no byte past its source**, with no closure map. Sampling every unit reads nothing. -/
theorem select_within {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {ξ : ByteArray} {r : Json × Array ℕ × ℕ}
    (h : Spot.Select.select rule p query table none ξ = .ok r) : r.2.2 ≤ ξ.size := by
  unfold Spot.Select.select at h
  obtain ⟨n, hn, h⟩ := bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := bind_eq_ok.1 h
  obtain ⟨derived, hderived, h⟩ := bind_eq_ok.1 h
  try dsimp only at h
  cases rule with
  | all =>
    try dsimp only at h
    have hr : r.2.2 = 0 := by
      obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
      obtain ⟨u, rd⟩ := ur
      obtain rfl : rd = 0 := (congrArg Prod.snd (Except.ok.inj hu)).symm
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact congrArg (fun q => q.2.2) (Except.ok.inj h).symm
    rw [hr]
    exact Nat.zero_le _
  | law text =>
    try dsimp only at h
    obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
    obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
    obtain ⟨u, rd⟩ := ur
    have hr : r.2.2 = rd := by
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact congrArg (fun q => q.2.2) (Except.ok.inj h).symm
    have hb : rd ≤ ξ.size := drawUnits_within hu
    rw [hr]
    exact hb
  | twoStage grain replay k =>
    try dsimp only at h
    obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
    obtain ⟨ur, hu, h⟩ := bind_eq_ok.1 h
    obtain ⟨u, rd⟩ := ur
    have hr : r.2.2 = rd := by
      revert h
      try dsimp only
      repeat' split
      all_goals intro h
      all_goals first
        | exact absurd h (throw_bind_ne _ _ _)
        | (cases h; done)
        | exact congrArg (fun q => q.2.2) (Except.ok.inj h).symm
    have hb : rd ≤ ξ.size := drawUnits_within hu
    rw [hr]
    exact hb

/-! ## At the held copies -/

/-- **A stage's sample reads no byte past its source** (`ReadsWithin`), at a held copy whose first-stage registration
has no closure map. -/
theorem readsWithin (h : SpotFlock.Held) (hc : h.reg.first.closure = none) : ReadsWithin h := by
  unfold ReadsWithin
  refine And.intro (fun ξ draw proved read hs => ?_) (fun sample ξ draw read hs => ?_)
  · rw [hc] at hs
    have hb : (draw, proved, read).2.2 ≤ ξ.size := select_within hs
    exact hb
  · have hb : (draw, read).2 ≤ ξ.size := selectSecond_within hs
    exact hb

/-- **The held copies carry the run's closure map**: where they agree with what the run reads (`RunsAs.HeldMatches`),
the held first-stage registration's closure map is the run's input's. -/
theorem closure_of_held {i : Spot.Run.Input} {it : Spot.Run.Interior} {h : SpotFlock.Held}
    (hm : HeldMatches i it h) : h.reg.first.closure = i.closure := by
  obtain ⟨n, rule, law, wt, res, -, -, -, -, -, hreg⟩ := hm.first
  exact congrArg SamplingDeploy.Reg.closure hreg

/-- **An accepted play's run of record has no closure map**: the form hands `Spot.Run.run` none (`inputOf`). -/
theorem closure_of_formAccepts {C : CaseOf} {o : Verity.Exchange} {i : Spot.Run.Input} (hA : FormAccepts C o)
    (hi : inputOn C o = some i) : i.closure = none := by
  obtain ⟨k, hk, p, cm, e, pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := hA
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have hik : inputOfCase k = some i' := by
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  have hi' : inputOn C o = some i' := by
    unfold inputOn
    rw [hk]
    exact hik
  have hii : i = i' := Option.some.inj (hi.symm.trans hi')
  rw [hii, hieq]
  rfl

/-! ## The accept, without the premise -/

/-- **The form's accept is a play SPOT[Flock] accepts** (`PlayAccepts`): `playAccepts_of_run` without `ReadsWithin`.
Where the held first-stage registration has no closure map, `readsWithin` gives the premise. Where it has one, no play
is accepted: an accepted play's run has none (`closure_of_formAccepts`) and the held copies carry the run's
(`closure_of_held`). -/
theorem playAccepts_of_run' (A : Live₂) (C : CaseOf) (S : SessionReading)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit) (hrule : RuleIsTwo C) (hheld : HeldAt A C)
    (htwo : TwoHolds A.h) (hnd : NamesDistinct A.h) (hrec : RecordsAt A C) (hread : RecordsRead C)
    (hr2 : RoundTwo A.h) (hst : StmtsRead A C S) (hrounds : RoundsHeldAt A C S) (hscope : ScopeHeldAt A C S)
    (hpin : PinHeldAt A C S) (hsess : SessionsRun A C S P') : PlayAccepts A C S P' := by
  by_cases hc : A.h.reg.first.closure = none
  · exact playAccepts_of_run A C S P' hrule hheld htwo hnd hrec hread hr2 hst hrounds hscope hpin hsess
      (readsWithin A.h hc)
  · intro o i it x m₁ hA hi hit hx hm₁
    exact absurd ((closure_of_held (hheld o i it hA hi hit)).trans (closure_of_formAccepts hA hi)) hc

end Verity.SampledTwo
