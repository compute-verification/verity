module

public import Spot.Select

/-! The count a parsed selection rule names is at least its rate, whatever population the prover registers
(`RuleAtRate`), with no premise. -/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot

/-- **A rate design's count over `n` units**: the design text `t` is a rate design at `NUM/DEN` over `T` tables with
`0 < NUM ≤ DEN`, and `t'` is `subset:k` at `k = Rate.count NUM DEN T n`, with `1 ≤ k ≤ n` and `NUM·n ≤ DEN·k`: a
uniform `k`-subset samples each unit with probability `k/n ≥ NUM/DEN`. -/
def SubsetAtRate (t t' : String) (n : Nat) : Prop :=
  ∃ num den tables, Select.parseRate t = some (num, den, tables) ∧ 0 < num ∧ num ≤ den ∧
    t' = s!"subset:{Rate.count num den tables n}" ∧ 1 ≤ Rate.count num den tables n ∧
    Rate.count num den tables n ≤ n ∧ num * n ≤ den * Rate.count num den tables n

/-- **A stratified rate design's `K` over `n` units in strata of `sizes` units**: the design text `t` is the stratified
rate design at `NUM/DEN` with `0 < NUM ≤ DEN`, and `t'` is `stratified:K` at `K = Rate.stratK NUM DEN sizes n`, with
`1 ≤ K ≤ n`, at which Flock's stratified rule gives each stratum of `s` of the `n` units a count `k_s` with
`NUM·s ≤ DEN·k_s`. -/
def StrataAtRate (t t' : String) (n : Nat) (sizes : List Nat) : Prop :=
  ∃ num den, Select.parseStrataRate t = some (false, num, den) ∧ 0 < num ∧ num ≤ den ∧
    t' = s!"stratified:{Rate.stratK num den sizes n}" ∧ 1 ≤ Rate.stratK num den sizes n ∧
    Rate.stratK num den sizes n ≤ n ∧
    ∀ s ∈ sizes, 0 < s → s ≤ n → num * s ≤ den * Flock.Draw.stratumK (Rate.stratK num den sizes n) s n

/-- **A work rate design's `K` over `n` units**: the design text `t` is the work rate design at `NUM/DEN` with
`0 < NUM ≤ DEN`, and `t'` is `work:K` at `K = Rate.count NUM DEN 1 n`, with `1 ≤ K ≤ n`. A work design sizes each
stratum from `K` and never below the stratum's floor, so what holds each stratum at the rate is its floor, which is
`TableAtRate`'s. -/
def WorkAtRate (t t' : String) (n : Nat) : Prop :=
  ∃ num den, Select.parseStrataRate t = some (true, num, den) ∧ 0 < num ∧ num ≤ den ∧
    t' = s!"work:{Rate.count num den 1 n}" ∧ 1 ≤ Rate.count num den 1 n ∧ Rate.count num den 1 n ≤ n

/-- **What a design text names over `n` units in strata of `sizes` units**, as the text `t'`: a rate design's count
(`SubsetAtRate`), a stratified rate design's `K` (`StrataAtRate`), a work rate design's `K` (`WorkAtRate`), or a
Bernoulli design, itself, which `Draw.bernoulli` marks. -/
def TextAtRate (t t' : String) (n : Nat) (sizes : List Nat) : Prop :=
  SubsetAtRate t t' n ∨ StrataAtRate t t' n sizes ∨ WorkAtRate t t' n ∨
    ∃ l, t' = t ∧ Select.parseDesign t = .ok l ∧ Draw.bernoulli l = true

/-- **What a rule is over `n` units in strata of `sizes` units**, as the rule `r`: `all` stays `all`; a design's text is
`TextAtRate`'s; a two-stage rule keeps its grain and its counts, and its replay design is a rate design's count
(`SubsetAtRate`) over the `n` replay blocks. -/
def AtRate (rule r : Select.Rule) (n : Nat) (sizes : List Nat) : Prop :=
  match rule with
  | .all => r = .all
  | .samplingDesign t => ∃ t', r = .samplingDesign t' ∧ TextAtRate t t' n sizes
  | .twoStage g t k => ∃ t', r = .twoStage g t' k ∧ SubsetAtRate t t' n

/-- The units of a stratum's ranges `[lo, hi)`, as `Select.Rule.tableOver` and `Select.strataSizes` count them. -/
def rangeUnits (rs : Array (Nat × Nat)) : Nat := rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0

/-- A work table's entry for a stratum of `n` units at the rate `NUM/DEN`: the work `w` of one of its units and the
floor `Rate.floor NUM DEN n f`, as `Select.Rule.tableOver` writes it. -/
def floorEntry (num den n w f : Nat) : Json :=
  Json.mkObj [("work", Lean.toJson w), ("floor", Lean.toJson (Rate.floor num den n f))]

/-- **A work rate design's table has, under each stratum's template, a floor at the rate**: where the design text `t`
is the work rate design at `NUM/DEN`, the table `out` is the object with an entry under each stratum's template, over
the strata of the program under the query (`Partition.strata`): some work and the floor `Rate.floor NUM DEN n_s f` at
the stratum's `n_s` units and some `f` (`floorEntry`), at which Flock's work rule gives the stratum `k_s` with
`NUM·n_s ≤ DEN·k_s`, under any `K` and any population work. The work and the `f` are only said to exist, each a
function of the template: they are tied to no table the verifier was handed. -/
def TableAtRate (t : String) (p : Program) (query : Json) (out : Option Json) : Prop :=
  ∀ num den, Select.parseStrataRate t = some (true, num, den) →
    ∃ (strata : Array (String × Array (Nat × Nat))) (work floor : String → Nat),
      Partition.strata p query = .ok strata ∧
      out = some (Json.mkObj (strata.toList.map fun s =>
        (s.1, floorEntry num den (rangeUnits s.2) (work s.1) (floor s.1)))) ∧
      ∀ s ∈ strata.toList, ∀ K total, num * rangeUnits s.2 ≤
        den * Flock.Draw.workK K (work s.1) (rangeUnits s.2) total (Rate.floor num den (rangeUnits s.2) (floor s.1))

/-- A rule object as a Spec writes it under the design text `t`: `{rule: samplingDesign, samplingDesign: t}`. -/
def ruleOfText (t : String) : Json := Json.mkObj [("rule", "law"), ("law", Json.str t)]

/-- **The count a parsed rule names is at least its rate, over whatever the prover registers.** For every rule object
`j` that `Rule.ofJson` parses as `rule`, every `n` and every `sizes`, where `Rule.over` gives the rule `r` over `n`
units in strata of `sizes` units, `r` is `AtRate`'s. Under a rate design, and under a two-stage rule's replay design,
the count it names is at least the rate's share of `n`. Under a stratified rate design each stratum's count is at
least the rate's share of the stratum, and of its `K` only `1 ≤ K ≤ n` is said. Under a work rate design only
`1 ≤ K ≤ n` is said: the rate is within the table's floors. Of `all` and of a Bernoulli text only that each stays
itself. For every design text, program, query and work table, where `Rule.tableOver` gives a table under the rule
`samplingDesign t`, that table is `TableAtRate`'s: the floor under each stratum's template is at the rate. And a fixed count is
refused: at every `K`, the rule object whose design text is `subset:K`, and likewise `stratified:K` and `work:K`, is
refused with `FIXED`. -/
def RuleAtRate : Prop :=
  (∀ (j : Json) (rule : Select.Rule) (n : Nat) (sizes : List Nat) (r : Select.Rule),
    Select.Rule.ofJson j = .ok rule → rule.over n sizes = some r → AtRate rule r n sizes) ∧
  (∀ (t : String) (p : Program) (query : Json) (table out : Option Json),
    (Select.Rule.samplingDesign t).tableOver p query table = .ok out → TableAtRate t p query out) ∧
  (∀ K : Nat, Select.Rule.ofJson (ruleOfText s!"subset:{K}") = .error (Select.FIXED s!"subset:{K}")) ∧
  (∀ K : Nat, Select.Rule.ofJson (ruleOfText s!"stratified:{K}") = .error (Select.FIXED s!"stratified:{K}")) ∧
  ∀ K : Nat, Select.Rule.ofJson (ruleOfText s!"work:{K}") = .error (Select.FIXED s!"work:{K}")

end Properties.Spot
