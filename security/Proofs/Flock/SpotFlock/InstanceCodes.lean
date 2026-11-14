module

public import Proofs.Flock.SpotFlock.InstanceRun
public import Proofs.Flock.SpotFlock.InstanceReads
public import Proofs.Flock.Claims.Complete.Json.Canon

/-!
# Round 2's checks at the two-stage instance, one at a time

`SpotFlock.Instance.readsOk_held`: at `held R bnd0 desc q c₀` and the zero coin `coin0` in rounds 0 and 1, round 2's
check of a session statement that is the reference statement itself (`xS = R.x`, `readsOk`) passes wherever the checks
that read its bytes do. What the registration and the zero coins decide holds for every `R`:
* what the verifier holds of the run (`ownOf`): the replay sample `#[0]` (`ownOf_sample`), the one part
  `⟨R.name, 0, 2⟩` (`ownOf_parts`, `partAt_held`) at position `0` (`posOf_held`), and the sampled units `#[0]`
  (`ownOf_drawn`, at the sample `secondDraw_coin0` gives);
* the interior record's order, where it lists the replay sample `[0]` (`interiorOrderOk_held`);
* the statement is the reference one (`pinOk_self`), and in scope where Flock's scope check and hidden outputs hold of
  it (`sessionOk_self`);
* the verdict of record round 2 holds a statement to is the statement's own (`codesOk`), so it names the statement's
  header, registered reads and sample (`verifiedOk_self`) as one table (`tablesOk_self`), wherever the sample prints
  (`Claims.Json.Good`).

`stmtCodes_eq_nil` is `SecondStage.stmtCodes_nil`'s converse, and `readsOk_of` and `codesOk_of` take `readsOk` and
`codesOk` apart at any registration.

## What this does not show

The checks left as hypotheses of `readsOk_held` read the statement's bytes, and nothing here states any: its header is
the reference header placed in the run (`second-header`, whose bindings are SHA-512 digests of the interior record's
names), its population and roots (`second-population`, `second-roots`), its sample (`second-sample`), its registered
reads (`second-read`, `second-refs`, `second-values`), and the statement's own fields (`stmtJsonOf`, the records `x` and
`m₁` decoding). A concrete `Ref` and the run's records give them; until then `readsOk` holds at no statement.
-/

@[expose] public section

namespace SpotFlock.Instance

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open Lean (Json)

/-! ## At any registration -/

/-- **A value that prints is canonically equal to itself.** -/
theorem jeq_self_of_good {a : Json} (h : Claims.Json.Good a) : Spot.jeq a a = true := by
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok_some a h
  simp [Spot.jeq, hs]

theorem all_of_forall {xs : Array (String × Json)} {f : String × Json → Bool}
    (h : ∀ port e, (port, e) ∈ xs → f (port, e) = true) : xs.all f = true :=
  Array.all_eq_true'.2 fun ⟨port, e⟩ hm => h port e hm

/-- **Round 2's statement checks pass where each passes**: `SecondStage.stmtCodes_nil`'s converse. -/
theorem stmtCodes_eq_nil {own : Spot.SecondStage.Own} {named : Bool} {k : ℕ} {p : Spot.SecondStage.Part}
    {s : Spot.SecondStage.Stmt}
    (h₁ : Spot.SecondStage.verifiedOk own s = true) (h₂ : Spot.SecondStage.tablesOk own s = true)
    (h₃ : Spot.SecondStage.headerOk own named k s.template s.header = true)
    (h₄ : Spot.SecondStage.populationOk own.sample.size p s.header = true)
    (h₅ : Spot.SecondStage.rootsOk named own.interior s.template s.header
      ((Spot.entries (Spot.getD s.registered "ports")).map (·.1)) = true)
    (h₆ : Spot.jeq s.draw (Spot.SecondStage.ownDraw own.sample.size p own.drawn) = true)
    (h₇ : ∀ port e, (port, e) ∈ Spot.entries (Spot.getD s.registered "ports") →
      Spot.SecondStage.readOk own named p port e = true)
    (h₈ : Spot.SecondStage.linkedRead own p.template (Spot.entries (Spot.getD s.registered "ports")) = true)
    (h₉ : Spot.SecondStage.wiredRead own p.template (Spot.entries (Spot.getD s.registered "ports")) = true)
    (h₁₀ : Spot.SecondStage.refsOk s = true) (h₁₁ : Spot.SecondStage.valuesOk own s = true) :
    Spot.SecondStage.stmtCodes own named k p s = [] := by
  unfold Spot.SecondStage.stmtCodes
  dsimp only
  rw [all_of_forall (xs := Spot.entries (Spot.getD s.registered "ports")) (fun port e hm => ?_)]
  · simp [h₁, h₂, h₃, h₄, h₅, h₆, h₈, h₉, h₁₀, h₁₁]
  · exact h₇ port e hm

/-- **`codesOk` passes** where round 2's statement checks find nothing, at the statement's own verdict of record. -/
theorem codesOk_of {h : Held} {own : Spot.SecondStage.Own} {t : ℕ} {S : Spot.SecondStage.Stmt}
    (hc : Spot.SecondStage.stmtCodes { own with verified := S.checked.toArray }
      (Spot.SecondStage.byTemplate own.interior)
      (Spot.SecondStage.posOf (partsAt h own.sample.size) (h.names.getD t "")) (partAt h own.sample.size t) S = []) :
    codesOk h own t S = true := by
  unfold codesOk
  rw [hc]
  rfl

/-- **`readsOk` passes** where its parts do: the run's records decode, the second stage samples, the session statement
reads as round 2's, the interior record's order, the statement checks, and the pin to the reference statement. -/
theorem readsOk_of {h : Held} {t : ℕ} {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} {st : Roots} {rec draw : Json}
    {w : FlockDeploy.Raw} {S : Spot.SecondStage.Stmt}
    (hx : decode x = some st) (hm : jsonMsg m₁ = some rec) (hd : secondDraw h c₀ c₁ = some draw)
    (hw : FlockDeploy.decode xS = some w) (hS : stmtJsonOf h t w = some S)
    (ho : Spot.SecondStage.interiorOrderOk (ownOf h st rec draw c₀) = true)
    (hc : codesOk h (ownOf h st rec draw c₀) t S = true) (hp : pinOk h t w = true) :
    readsOk h t x m₁ xS c₀ c₁ = true := by
  unfold readsOk
  rw [hx, hm, hd, hw]
  simp [ho, hS, hc, hp]

/-- **What a verdict of record names of a statement**: registered reads that print, the statement's sample, and one
table. -/
theorem checked_fields {S : Spot.SecondStage.Stmt} {c : Spot.SecondStage.Checked} (hc : S.checked = some c) :
    Claims.Json.Good c.registered ∧ c.draw = S.draw ∧ c.tables = Spot.SecondStage.ONE_TABLE := by
  unfold Spot.SecondStage.Stmt.checked at hc
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hc
  obtain ⟨hd, -, hc⟩ := hc
  split at hc
  · simp only [pure, Option.bind, Option.some.injEq] at hc
    subst hc
    exact ⟨.null, rfl, rfl⟩
  · simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff, pure, Option.some.injEq] at hc
    obtain ⟨_, ⟨s, -, rfl⟩, rfl⟩ := hc
    exact ⟨.str s, rfl, rfl⟩

theorem jeq_one : Spot.jeq Spot.SecondStage.ONE_TABLE Spot.SecondStage.ONE_TABLE = true :=
  jeq_self_of_good (.num 1)

/-- **`second-verified` passes at the statement's own verdict of record**, wherever its sample prints. -/
theorem verifiedOk_self {own : Spot.SecondStage.Own} {S : Spot.SecondStage.Stmt} {c : Spot.SecondStage.Checked}
    (hc : S.checked = some c) (hd : Claims.Json.Good S.draw) :
    Spot.SecondStage.verifiedOk { own with verified := S.checked.toArray } S = true := by
  obtain ⟨hr, hdc, -⟩ := checked_fields hc
  have hr' := jeq_self_of_good hr
  have hd' : Spot.jeq c.draw c.draw = true := by rw [hdc]; exact jeq_self_of_good hd
  unfold Spot.SecondStage.verifiedOk
  rw [hc]
  simp [hr', hd']

/-- **`second-tables` passes at the statement's own verdict of record**: it names one table. -/
theorem tablesOk_self {own : Spot.SecondStage.Own} {S : Spot.SecondStage.Stmt} {c : Spot.SecondStage.Checked}
    (hc : S.checked = some c) : Spot.SecondStage.tablesOk { own with verified := S.checked.toArray } S = true := by
  obtain ⟨-, -, ht⟩ := checked_fields hc
  unfold Spot.SecondStage.tablesOk
  rw [hc]
  simp [ht, jeq_one]

/-! ## At the instance and the zero coins -/

variable (R : Ref) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The second stage's sample at the zero coins** proves unit `0`. -/
theorem secondDraw_coin0 :
    ∃ d, secondDraw (held R bnd0 desc q c₀) coin0 coin0 = some d ∧ Spot.Draw.proved d = #[0] := by
  obtain ⟨d, rd, hs, hp⟩ := selectSecond_coin0
  have hs' : Spot.Select.selectSecond ((held R bnd0 desc q c₀).reg.sample (SpotFlock.src coin0))
      (held R bnd0 desc q c₀).reg.ns (held R bnd0 desc q c₀).reg.ks (SpotFlock.src coin0) = .ok (d, rd) := by
    rw [show (held R bnd0 desc q c₀).reg.sample (SpotFlock.src coin0) = #[0] from sample_coin0]
    exact hs
  exact ⟨d, by simp only [secondDraw, hs'], hp⟩

theorem ownOf_sample (st : Roots) (rec draw : Json) :
    (ownOf (held R bnd0 desc q c₀) st rec draw coin0).sample = #[0] :=
  sample_coin0

theorem ownOf_parts (st : Roots) (rec draw : Json) :
    (ownOf (held R bnd0 desc q c₀) st rec draw coin0).parts = some #[⟨R.name, 0, 2⟩] :=
  congrArg some (partsAt_held R bnd0 desc q c₀ _)

theorem ownOf_drawn (st : Roots) (rec : Json) {draw : Json} (hp : Spot.Draw.proved draw = #[0]) :
    (ownOf (held R bnd0 desc q c₀) st rec draw coin0).drawn = #[0] := by
  show (Spot.Draw.proved draw).map Int.toNat = #[0]
  rw [hp]
  simp

theorem partAt_held : partAt (held R bnd0 desc q c₀) 1 0 = ⟨R.name, 0, 2⟩ := rfl

/-- **The template's position among the parts** is `0`. -/
theorem posOf_held :
    Spot.SecondStage.posOf (partsAt (held R bnd0 desc q c₀) 1) ((held R bnd0 desc q c₀).names.getD 0 "") = 0 := by
  rw [partsAt_held]
  show Spot.SecondStage.posOf #[⟨R.name, 0, 2⟩] R.name = 0
  simp [Spot.SecondStage.posOf]

/-- **The interior record's order** at the zero coins: it lists the replay sample `[0]`. -/
theorem interiorOrderOk_held {st : Roots} {rec draw : Json}
    (hw : Spot.getD (Spot.getD rec "window") "replay_sample" = .arr #[Lean.toJson (0 : ℕ)]) :
    Spot.SecondStage.interiorOrderOk (ownOf (held R bnd0 desc q c₀) st rec draw coin0) = true := by
  have hs := ownOf_sample R desc q c₀ st rec draw
  have hi : (ownOf (held R bnd0 desc q c₀) st rec draw coin0).interior = rec := rfl
  unfold Spot.SecondStage.interiorOrderOk
  rw [hs, hi, hw]
  decide

/-- **The reference statement pins to itself.** -/
theorem pinOk_self {w : FlockDeploy.Raw} (hw : FlockDeploy.decode R.x = some w) :
    pinOk (held R bnd0 desc q c₀) 0 w = true := by
  unfold pinOk
  rw [show (held R bnd0 desc q c₀).xS₀ 0 = R.x from rfl, hw]
  simp

/-- **The reference statement is in the session scope** where Flock's scope check and hidden outputs hold of it. -/
theorem sessionOk_self (hs : FlockDeploy.scopeCheck true R.x = true) (hh : hiddenOf R.x = true) :
    sessionOk (held R bnd0 desc q c₀) 0 R.x = true := by
  unfold sessionOk
  rw [show (held R bnd0 desc q c₀).xS₀ 0 = R.x from rfl, R.zk, hs, hh]
  simp

/-- **Round 2's check passes the reference statement at the zero coins** wherever the checks that read its bytes do:
the run's records `x` and `m₁` decode, the interior record lists the replay sample `[0]`, the statement reads as round
2's (`stmtJsonOf`) with a sample that prints and a verdict of record, and at every sample that proves unit `0` its
header, population, roots, sample and registered reads pass against the template's part `⟨R.name, 0, 2⟩` at position
`0`. -/
theorem readsOk_held {x m₁ : Msg} {st : Roots} {rec : Json} {w : FlockDeploy.Raw} {S : Spot.SecondStage.Stmt}
    {c : Spot.SecondStage.Checked}
    (hx : decode x = some st) (hm : jsonMsg m₁ = some rec)
    (hw : FlockDeploy.decode R.x = some w) (hS : stmtJsonOf (held R bnd0 desc q c₀) 0 w = some S)
    (horder : Spot.getD (Spot.getD rec "window") "replay_sample" = .arr #[Lean.toJson (0 : ℕ)])
    (hdraw : Claims.Json.Good S.draw) (hc : S.checked = some c)
    (hrest : ∀ draw, Spot.Draw.proved draw = #[0] →
      let own := { ownOf (held R bnd0 desc q c₀) st rec draw coin0 with verified := S.checked.toArray }
      let reads := Spot.entries (Spot.getD S.registered "ports")
      let p : Spot.SecondStage.Part := ⟨R.name, 0, 2⟩
      Spot.SecondStage.headerOk own (Spot.SecondStage.byTemplate rec) 0 S.template S.header = true ∧
      Spot.SecondStage.populationOk 1 p S.header = true ∧
      Spot.SecondStage.rootsOk (Spot.SecondStage.byTemplate rec) rec S.template S.header (reads.map (·.1)) = true ∧
      Spot.jeq S.draw (Spot.SecondStage.ownDraw 1 p #[0]) = true ∧
      (∀ port e, (port, e) ∈ reads →
        Spot.SecondStage.readOk own (Spot.SecondStage.byTemplate rec) p port e = true) ∧
      Spot.SecondStage.linkedRead own R.name reads = true ∧ Spot.SecondStage.wiredRead own R.name reads = true ∧
      Spot.SecondStage.refsOk S = true ∧ Spot.SecondStage.valuesOk own S = true) :
    readsOk (held R bnd0 desc q c₀) 0 x m₁ R.x coin0 coin0 = true := by
  obtain ⟨draw, hd, hp⟩ := secondDraw_coin0 R desc q c₀
  obtain ⟨h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₁₀, h₁₁⟩ := hrest draw hp
  have hsz : (ownOf (held R bnd0 desc q c₀) st rec draw coin0).sample.size = 1 := by
    rw [ownOf_sample]; rfl
  refine readsOk_of hx hm hd hw hS (interiorOrderOk_held R desc q c₀ horder) (codesOk_of ?_)
    (pinOk_self R desc q c₀ hw)
  rw [hsz, posOf_held, partAt_held]
  refine stmtCodes_eq_nil (verifiedOk_self hc hdraw) (tablesOk_self hc) h₃ ?_ h₅ ?_ h₇ h₈ h₉ h₁₀ h₁₁
  · show Spot.SecondStage.populationOk (ownOf (held R bnd0 desc q c₀) st rec draw coin0).sample.size _ _ = true
    rw [hsz]
    exact h₄
  · show Spot.jeq S.draw (Spot.SecondStage.ownDraw (ownOf (held R bnd0 desc q c₀) st rec draw coin0).sample.size _
      (ownOf (held R bnd0 desc q c₀) st rec draw coin0).drawn) = true
    rw [hsz, ownOf_drawn R desc q c₀ st rec hp]
    exact h₆

end SpotFlock.Instance
