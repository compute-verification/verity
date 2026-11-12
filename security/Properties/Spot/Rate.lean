module

public import Spot.Select

/-!
# A parsed selection rule samples at least its rate, whatever population the prover registers

The population a selection rule samples is the registration's, which the prover sizes. A fixed count `K` of `N` units
samples each unit with `K/N`, which falls as the prover pads `N`. So `Select.Rule.ofJson` parses a rule only when its
design text is a rate (`Select.rate?`, `Select.replayLaw?`) and refuses a fixed count (`Select.FIXED`), and
`Select.Rule.over` turns the rate into a count once the population is known. `RuleAtRate` says what that gives, against
those functions, and takes no premise. It has three parts:

* a rule `Rule.ofJson` parses, over any `n` units in strata of any `sizes` where `Rule.over` gives a rule, is one of
  `AtRate`'s forms. Under `all` it stays `all`. Under a rate design `rate:NUM/DEN[:T]`, and under a two-stage rule's
  replay design, it is `subset:k` with `1 ≤ k ≤ n` and `NUM·n ≤ DEN·k`. Under `stratified-rate:NUM/DEN` it is
  `stratified:K` with `1 ≤ K ≤ n`, at which Flock's stratified rule (`Flock.Draw.stratumK`) gives every stratum of `s`
  of the `n` units a count `k_s` with `NUM·s ≤ DEN·k_s`. Under `work-rate:NUM/DEN` it is `work:K` with `1 ≤ K ≤ n`. The
  one other text `Rule.ofJson` parses is `bernoulli:NUM/DEN`, which stays itself, a sampling design `Draw.bernoulli`
  marks;
* the work table `Select.Rule.tableOver` writes for a work rate design names each stratum of the program under the query
  once, in order, with the verifier's work and the floor `Rate.floor NUM DEN n_s f`, at which Flock's work rule
  (`Flock.Draw.workK`) gives the stratum a count `k_s` with `NUM·n_s ≤ DEN·k_s` under any `K` and any population work
  (`TableAtRate`);
* a rule object whose design text is a fixed count, `subset:K`, `stratified:K` or `work:K`, is refused with `FIXED`.

Each bound is on the count a rule names, for every population and every strata, so the prover's choice of them moves
none: it holds at a population the prover picks after seeing the coins of earlier windows too.

Not proved here, each the code's or another statement's:

* that the stream's run calls these functions at the population and the strata it then samples, reads the table
  `tableOver` wrote, and refuses a registration under a Bernoulli design;
* that the sample `Select.select` makes holds the rule's count of units of each stratum, which is the sampled audit's
  own statement;
* anything of a two-stage rule's second stage, whose counts are fixed per cell;
* a bound pooled over windows whose populations the prover picks one after another: each pooled statement fixes every
  window's population before the period.
-/

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
  | .law t => ∃ t', r = .law t' ∧ TextAtRate t t' n sizes
  | .twoStage g t k => ∃ t', r = .twoStage g t' k ∧ SubsetAtRate t t' n

/-- The units of a stratum's ranges `[lo, hi)`, as `Select.Rule.tableOver` and `Select.strataSizes` count them. -/
def rangeUnits (rs : Array (Nat × Nat)) : Nat := rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0

/-- A work table's entry for a stratum of `n` units at the rate `NUM/DEN`: the work `w` of one of its units and the
floor `Rate.floor NUM DEN n f`, as `Select.Rule.tableOver` writes it. -/
def floorEntry (num den n w f : Nat) : Json :=
  Json.mkObj [("work", Lean.toJson w), ("floor", Lean.toJson (Rate.floor num den n f))]

/-- **A work rate design's table holds every stratum's floor at the rate**: where the design text `t` is the work rate
design at `NUM/DEN`, the table `out` is an object with one entry for each stratum of the program under the query
(`Partition.strata`), in their order, under the stratum's template: a work and the floor `Rate.floor NUM DEN n_s f` at
the stratum's `n_s` units (`floorEntry`), at which Flock's work rule gives the stratum `k_s` with `NUM·n_s ≤ DEN·k_s`,
under any `K` and any population work. -/
def TableAtRate (t : String) (p : Program) (query : Json) (out : Option Json) : Prop :=
  ∀ num den, Select.parseStrataRate t = some (true, num, den) →
    ∃ (strata : Array (String × Array (Nat × Nat))) (work floor : String → Nat),
      Partition.strata p query = .ok strata ∧
      out = some (Json.mkObj (strata.toList.map fun s =>
        (s.1, floorEntry num den (rangeUnits s.2) (work s.1) (floor s.1)))) ∧
      ∀ s ∈ strata.toList, ∀ K total, num * rangeUnits s.2 ≤
        den * Flock.Draw.workK K (work s.1) (rangeUnits s.2) total (Rate.floor num den (rangeUnits s.2) (floor s.1))

/-- A rule object as a Spec writes it under the design text `t`: `{rule: law, law: t}`. -/
def ruleOfText (t : String) : Json := Json.mkObj [("rule", "law"), ("law", Json.str t)]

/-- The fixed counts' design texts at `K`: `subset:K`, `stratified:K` and `work:K`. -/
def fixedTexts (K : Nat) : List String := [s!"subset:{K}", s!"stratified:{K}", s!"work:{K}"]

/-- **A parsed rule samples at least its rate over whatever the prover registers.** For every rule object `j` that
`Rule.ofJson` parses as `rule`, every `n` and every `sizes`, where `Rule.over` gives the rule `r` over `n` units in
strata of `sizes` units, `r` is `AtRate`'s: the count it names is at least the rate's share of `n`, and of each
stratum. For every design text, program, query and work table, where `Rule.tableOver` gives a table under the rule
`law t`, that table is `TableAtRate`'s: each stratum's floor is at the rate. And a fixed count is refused: at every
`K`, the rule object whose design text is `subset:K`, `stratified:K` or `work:K` (`fixedTexts`) is refused with
`FIXED`. -/
def RuleAtRate : Prop :=
  (∀ (j : Json) (rule : Select.Rule) (n : Nat) (sizes : List Nat) (r : Select.Rule),
    Select.Rule.ofJson j = .ok rule → rule.over n sizes = some r → AtRate rule r n sizes) ∧
  (∀ (t : String) (p : Program) (query : Json) (table out : Option Json),
    (Select.Rule.law t).tableOver p query table = .ok out → TableAtRate t p query out) ∧
  ∀ K, ∀ t ∈ fixedTexts K, Select.Rule.ofJson (ruleOfText t) = .error (Select.FIXED t)

end Properties.Spot
