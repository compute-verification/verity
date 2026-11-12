module

public import Properties.Spot.Rate
public import Proofs.Properties.Verity.Sampled.Record.Reg
public import Proofs.Flock.SpotFlock.Jeq
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Repr

/-!
# `RuleAtRate`: a parsed selection rule samples at least its rate

* **What the parsers return** (`parseRate_bounds`, `parseStrataRate_bounds`): a rate design's `(NUM, DEN, T)` has
  `0 < NUM ≤ DEN` and `1 ≤ T`, and a stratified or work rate design's `(NUM, DEN)` has `0 < NUM ≤ DEN`, read off their
  guards: no text is evaluated.
* **`Rule.ofJson` by cases** (`ofJson_cases`): a rule it parses is `all`, a design whose text is a rate (`rate?`), or a
  two-stage rule whose replay design is a rate design (`replayLaw?`).
* **`textOver` by cases** (`subset_of_textOver`, `textOver_cases`): over `n` units, a rate text's count is
  `Rate.count`, `Rate.stratK` or `Rate.count` at one table, between 1 and `n`, and at least the rate by
  `Rate.count_ge` and `Rate.stratK_ge`; any other text `rate?` admits is a Bernoulli design.
* **The work table** (`tableOver_cases`): each entry is its stratum's, at `Rate.floor`, at least the rate by
  `Rate.workK_ge`.
* **A rule object at a design text** (`ofJson_ruleOfText`): `{rule: law, law: t}` parses as the design `t` where `t` is
  a rate, and is refused with `FIXED t` where it is not; a fixed count is no rate (`parseRate_fixed`,
  `parseStrataRate_fixed`, `rate?_fixed`), and `rate:NUM/DEN` is read back (`parseRate_repr`).
* `ruleAtRate`, with its witnesses: `ruleAtRate_honest`, the rule object at `rate:1/2` parses and is `subset:2` over
  four units, and `ruleAtRate_nontrivial`, the fixed count `subset:1` kept as it is over three units is none of
  `AtRate`'s forms, so a parser that takes every rule object as it is written breaks the property.
-/

@[expose] public section

open Lean (Json)
open Spot
open Verity.Sampled.Record (split1_colon split1_slash digits_repr toNat!_repr not_mem_repr parseDesign_count)

namespace Properties.Spot.SecurityProofs

/-! ## What the parsers return -/

/-- **A rate design's numbers**: `0 < NUM ≤ DEN` and `1 ≤ T`. -/
theorem parseRate_bounds {t : String} {num den tables : Nat}
    (h : Select.parseRate t = some (num, den, tables)) : 0 < num ∧ num ≤ den ∧ 1 ≤ tables := by
  unfold Select.parseRate at h
  repeat' split at h
  all_goals first
    | contradiction
    | (rename_i hc
       simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
       cases h
       exact ⟨hc.1.1.2, hc.1.2, hc.2⟩)

/-- **A stratified or work rate design's numbers**: `0 < NUM ≤ DEN`. -/
theorem parseStrataRate_bounds {t : String} {work : Bool} {num den : Nat}
    (h : Select.parseStrataRate t = some (work, num, den)) : 0 < num ∧ num ≤ den := by
  unfold Select.parseStrataRate at h
  repeat' split at h
  all_goals first
    | contradiction
    | (rename_i hc
       simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
       cases h
       exact ⟨hc.1.2, hc.2⟩)

/-! ## `Rule.ofJson` by cases -/

/-- **The rules `Rule.ofJson` parses**: `all`, a design whose text is a rate, or a two-stage rule whose replay design is
a rate design. -/
theorem ofJson_cases {j : Json} {rule : Select.Rule} (h : Select.Rule.ofJson j = .ok rule) :
    rule = .all ∨ (∃ t, rule = .law t ∧ Select.rate? t = true) ∨
      ∃ g t k, rule = .twoStage g t k ∧ Select.replayLaw? t = true := by
  unfold Select.Rule.ofJson at h
  dsimp only at h
  repeat' split at h
  all_goals first
    | exact absurd h (throw_ne _ _)
    | exact absurd h (throw_bind_ne _ _ _)
    | (obtain rfl := (Except.ok.inj h).symm
       first
         | exact Or.inl rfl
         | exact Or.inr (Or.inl ⟨_, rfl, by simp_all⟩)
         | exact Or.inr (Or.inr ⟨_, _, _, rfl, by simp_all⟩))

/-! ## `textOver` by cases -/

/-- **A rate design's text over `n` units** is `SubsetAtRate`'s. -/
theorem subset_of_textOver {t t' : String} {n : Nat} {sizes : List Nat} {num den tables : Nat}
    (hp : Select.parseRate t = some (num, den, tables)) (h : Select.textOver t n sizes = some t') :
    SubsetAtRate t t' n := by
  obtain ⟨hn, hd, ht⟩ := parseRate_bounds hp
  unfold Select.textOver at h
  rw [hp] at h
  dsimp only at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    obtain rfl := Option.some.inj h
    exact ⟨num, den, tables, hp, hn, hd, rfl, hc.1, hc.2,
      Rate.count_ge num den tables n (Nat.lt_of_lt_of_le hn hd) ht⟩
  · exact absurd h (by simp)

/-- **A rate's text over `n` units in strata of `sizes` units** is `TextAtRate`'s: a text `rate?` admits is a rate
design, a stratified or work rate design, or a Bernoulli design. -/
theorem textOver_cases {t t' : String} {n : Nat} {sizes : List Nat} (hr : Select.rate? t = true)
    (h : Select.textOver t n sizes = some t') : TextAtRate t t' n sizes := by
  cases hp : Select.parseRate t with
  | some x =>
    obtain ⟨num, den, tables⟩ := x
    exact Or.inl (subset_of_textOver hp h)
  | none =>
    cases hs : Select.parseStrataRate t with
    | some x =>
      obtain ⟨work, num, den⟩ := x
      obtain ⟨hn, hd⟩ := parseStrataRate_bounds hs
      unfold Select.textOver at h
      rw [hp, hs] at h
      cases work
      · dsimp only at h
        split at h
        · rename_i hc
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
          obtain rfl := Option.some.inj h
          exact Or.inr (Or.inl ⟨num, den, hs, hn, hd, rfl, hc.1.2, hc.2, fun s hm h0 hsn =>
            Rate.stratK_ge num den s n sizes hm h0 hsn (Nat.lt_of_lt_of_le hn hd) hd⟩)
        · exact absurd h (by simp)
      · dsimp only at h
        split at h
        · rename_i hc
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
          obtain rfl := Option.some.inj h
          exact Or.inr (Or.inr (Or.inl ⟨num, den, hs, hn, hd, rfl, hc.1, hc.2⟩))
        · exact absurd h (by simp)
    | none =>
      unfold Select.textOver at h
      rw [hp, hs] at h
      obtain rfl := Option.some.inj h
      unfold Select.rate? at hr
      rw [hp, hs] at hr
      simp only [Option.isSome_none, Bool.false_or] at hr
      split at hr
      · rename_i l hl
        have hlaw : getD l "law" = .str "bernoulli" := SpotFlock.str_of_jeq (SpotFlock.jeq_symm hr)
        refine Or.inr (Or.inr (Or.inr ⟨l, rfl, hl, ?_⟩))
        unfold Draw.bernoulli
        rw [hlaw]
        rfl
      · exact absurd hr (by simp)

/-! ## The work table -/

theorem mapM_ok_map {α β ε : Type} {f : α → Except ε β} {g : α → β} (hfg : ∀ a b, f a = .ok b → b = g a) :
    ∀ {xs : List α} {ys : List β}, xs.mapM f = .ok ys → ys = xs.map g
  | [], ys, h => by
    rw [List.mapM_nil] at h
    exact (Except.ok.inj h).symm
  | x :: xs, ys, h => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := bind_eq_ok.1 h
    obtain ⟨bs, hbs, h⟩ := bind_eq_ok.1 h
    obtain rfl := (Except.ok.inj h).symm
    rw [List.map_cons, hfg x b hb, mapM_ok_map hfg hbs]

/-- The work of one unit of the template `tmpl` in a work table as read, and none where the table names none. -/
def tableWork (given : Array (String × Nat × Nat)) (tmpl : String) : Nat :=
  match given.find? (·.1 == tmpl) with
  | some (_, w, _) => w
  | none => 0

/-- The floor of the template `tmpl` in a work table as read, and none where the table names none. -/
def tableFloor (given : Array (String × Nat × Nat)) (tmpl : String) : Nat :=
  match given.find? (·.1 == tmpl) with
  | some (_, _, f) => f
  | none => 0

/-- **A work rate design's table** is `TableAtRate`'s. -/
theorem tableOver_cases {t : String} {p : Program} {query : Json} {table out : Option Json}
    (h : (Select.Rule.law t).tableOver p query table = .ok out) : TableAtRate t p query out := by
  intro num den hs
  obtain ⟨hn, hd⟩ := parseStrataRate_bounds hs
  unfold Select.Rule.tableOver at h
  dsimp only at h
  rw [hs] at h
  dsimp only at h
  obtain ⟨strata, hstrata, h⟩ := bind_eq_ok.1 h
  split at h
  · obtain ⟨given, -, h⟩ := bind_eq_ok.1 h
    obtain ⟨entries, he, h⟩ := bind_eq_ok.1 h
    obtain rfl := (Except.ok.inj h).symm
    refine ⟨strata, tableWork given, tableFloor given, hstrata, ?_, fun s _ K total =>
      Rate.workK_ge num den _ _ K _ total (Nat.lt_of_lt_of_le hn hd) hd⟩
    rw [Array.mapM_eq_mapM_toList] at he
    obtain ⟨es, hes, he⟩ := bind_eq_ok.1 he
    obtain rfl := (Except.ok.inj he).symm
    congr 2
    refine mapM_ok_map (g := fun s => (s.1, floorEntry num den (rangeUnits s.2) (tableWork given s.1) (tableFloor given s.1)))
      (fun s b hb => ?_) hes
    obtain ⟨tmpl, rs⟩ := s
    dsimp only at hb
    unfold tableWork tableFloor
    split at hb
    · rename_i hf
      rw [hf]
      exact (Except.ok.inj hb).symm
    · exact absurd hb (throw_ne _ _)
  · exact absurd h (throw_bind_ne _ _ _)

/-! ## A rule object at a design text -/

theorem keys_ruleOfText (t : String) : keys? (ruleOfText t) = some #["law", "rule"] := by
  have hs : sortStrings #["law", "rule"] = #["law", "rule"] :=
    Array.toList_inj.1 (Properties.Pouw.Period.SecurityProofs.Sort.qsort_of_sorted (as := #["law", "rule"]) (by decide))
  rw [← hs]
  unfold keys? ruleOfText
  with_unfolding_all rfl

/-- **A rule object at a design text**: `{rule: law, law: t}` parses as the design `t` where `t` is a rate, and is
refused with `FIXED t` where it is not. -/
theorem ofJson_ruleOfText (t : String) :
    Select.Rule.ofJson (ruleOfText t) = if Select.rate? t then .ok (.law t) else .error (Select.FIXED t) := by
  unfold Select.Rule.ofJson
  rw [show getD (ruleOfText t) "rule" = .str "law" from rfl, keys_ruleOfText,
    show getD (ruleOfText t) "law" = .str t from rfl]
  have hk : (some #["law", "rule"] == some #["law", "rule"]) = true := by decide
  simp only [hk]
  cases Select.rate? t <;> rfl

/-- `s.splitOn ":"` of a text with no colon is the text. -/
theorem splitOn_colon_none (s : String) (hs : ':' ∉ s.toList) : s.splitOn ":" = [s] := by
  have hne : ∀ x ∈ s.toList, (x == ':') = false := fun x hx => by
    simp only [beq_eq_false_iff_ne, ne_eq]
    rintro rfl
    exact hs hx
  rw [FlockVBridge.Names.splitOn_sep ":" ':' rfl rfl rfl rfl (by decide), List.splitOn_eq_splitOnP,
    List.splitOnP_eq_singleton hne]
  simp only [List.map_cons, List.map_nil, String.ofList_toList]

/-- **`parseRate` reads `rate:NUM/DEN` back**: at `0 < NUM ≤ DEN`, each in `Nat.repr`'s digits, it is `(NUM, DEN, 1)`.
-/
theorem parseRate_repr (a b : Nat) (ha : 0 < a) (hab : a ≤ b) :
    Select.parseRate ("rate" ++ ":" ++ (Nat.repr a ++ "/" ++ Nat.repr b)) = some (a, b, 1) := by
  have hc : ':' ∉ (Nat.repr a ++ "/" ++ Nat.repr b).toList := by
    rw [String.toList_append, String.toList_append]
    simp only [List.mem_append, not_or]
    exact ⟨⟨not_mem_repr (by decide) a, by decide⟩, not_mem_repr (by decide) b⟩
  unfold Select.parseRate
  rw [split1_colon _ _ (by decide) hc]
  dsimp only
  rw [splitOn_colon_none _ hc]
  dsimp only
  rw [split1_slash _ _ (not_mem_repr (by decide) a) (not_mem_repr (by decide) b)]
  simp only [digits_repr, toNat!_repr, decide_eq_true ha, decide_eq_true hab]
  rfl

/-- **A fixed count is no rate design**: `kind:K`, for a `subset`, `stratified` or `work` design. -/
theorem parseRate_fixed (kind : String) (hk : kind = "subset" ∨ kind = "stratified" ∨ kind = "work") (K : Nat) :
    Select.parseRate (kind ++ ":" ++ Nat.repr K) = none := by
  have hka : ':' ∉ kind.toList := by rcases hk with rfl | rfl | rfl <;> decide
  have hr : (kind == "rate") = false := by rcases hk with rfl | rfl | rfl <;> decide
  unfold Select.parseRate
  rw [split1_colon kind (Nat.repr K) hka (not_mem_repr (by decide) K)]
  simp only [hr, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  repeat' split
  all_goals rfl

/-- **A fixed count is no stratified or work rate design.** -/
theorem parseStrataRate_fixed (kind : String) (hk : kind = "subset" ∨ kind = "stratified" ∨ kind = "work") (K : Nat) :
    Select.parseStrataRate (kind ++ ":" ++ Nat.repr K) = none := by
  have hka : ':' ∉ kind.toList := by rcases hk with rfl | rfl | rfl <;> decide
  have hr : (kind == "stratified-rate" || kind == "work-rate") = false := by
    rcases hk with rfl | rfl | rfl <;> decide
  unfold Select.parseStrataRate
  rw [split1_colon kind (Nat.repr K) hka (not_mem_repr (by decide) K)]
  simp only [hr, Bool.false_and, Bool.false_eq_true, ↓reduceIte]

/-- **A fixed count is no rate**: `kind:K`, for a `subset`, `stratified` or `work` design, at any `K`: it is no rate
design, no stratified or work rate design, and where it is a sampling design (`1 ≤ K`) its kind is not `bernoulli`. -/
theorem rate?_fixed (kind : String) (hk : kind = "subset" ∨ kind = "stratified" ∨ kind = "work") (K : Nat) :
    Select.rate? (kind ++ ":" ++ Nat.repr K) = false := by
  unfold Select.rate?
  rw [parseRate_fixed kind hk K, parseStrataRate_fixed kind hk K]
  simp only [Option.isSome_none, Bool.false_or]
  split
  · rename_i l hl
    cases hK : decide (1 ≤ K) with
    | true =>
      rw [parseDesign_count kind hk K (of_decide_eq_true hK)] at hl
      obtain rfl := (Except.ok.inj hl).symm
      rw [show getD (Json.mkObj [("law", Json.str kind), ("k", Lean.toJson K)]) "law" = .str kind from rfl]
      cases hj : jeq (Json.str kind) (Json.str "bernoulli") with
      | false => rfl
      | true =>
        have hb := Json.str.inj (SpotFlock.str_of_jeq hj)
        rcases hk with rfl | rfl | rfl <;> exact absurd hb (by decide)
    | false =>
      have hK' : ¬ 1 ≤ K := of_decide_eq_false hK
      have hka : ':' ∉ kind.toList := by rcases hk with rfl | rfl | rfl <;> decide
      have hb : (kind == "bernoulli") = false := by rcases hk with rfl | rfl | rfl <;> decide
      unfold Select.parseDesign at hl
      rw [split1_colon kind (Nat.repr K) hka (not_mem_repr (by decide) K)] at hl
      simp only [digits_repr, toNat!_repr, decide_eq_false hK', Bool.and_false, Bool.false_eq_true, ↓reduceIte, hb] at hl
      exact absurd hl (throw_ne _ _)
  · rfl

/-! ## The property and its witnesses -/

/-- **`RuleAtRate`**: a parsed rule samples at least its rate over whatever the prover registers, a work rate design's
table holds each stratum's floor at the rate, and a fixed count is refused. -/
theorem ruleAtRate : RuleAtRate := by
  refine ⟨fun j rule n sizes r hj ho => ?_, fun t p query table out h => tableOver_cases h, fun K t ht => ?_⟩
  · rcases ofJson_cases hj with rfl | ⟨t, rfl, hr⟩ | ⟨g, t, k, rfl, hr⟩
    · exact (Option.some.inj ho).symm
    · have ho' : (Select.textOver t n sizes).map Select.Rule.law = some r := ho
      cases hx : Select.textOver t n sizes with
      | none => rw [hx] at ho'; exact absurd ho' (by simp)
      | some t' =>
        rw [hx] at ho'
        obtain rfl := Option.some.inj ho'
        exact ⟨t', rfl, textOver_cases hr hx⟩
    · have ho' : (Select.textOver t n sizes).map (Select.Rule.twoStage g · k) = some r := ho
      cases hx : Select.textOver t n sizes with
      | none => rw [hx] at ho'; exact absurd ho' (by simp)
      | some t' =>
        rw [hx] at ho'
        obtain rfl := Option.some.inj ho'
        obtain ⟨x, hp⟩ := Option.isSome_iff_exists.1 hr
        obtain ⟨num, den, tables⟩ := x
        exact ⟨t', rfl, subset_of_textOver hp hx⟩
  · have hf : Select.rate? t = false := by
      simp only [fixedTexts, List.mem_cons, List.mem_nil_iff, or_false] at ht
      rcases ht with rfl | rfl | rfl
      · exact rate?_fixed "subset" (Or.inl rfl) K
      · exact rate?_fixed "stratified" (Or.inr (Or.inl rfl)) K
      · exact rate?_fixed "work" (Or.inr (Or.inr rfl)) K
    rw [ofJson_ruleOfText, hf]
    rfl

/-- **`RuleAtRate`'s honest lemma**: the rule object at the rate design `rate:1/2` parses, and over four units it is
`subset:2`. -/
theorem ruleAtRate_honest :
    Select.Rule.ofJson (ruleOfText "rate:1/2") = .ok (.law "rate:1/2") ∧
      (Select.Rule.law "rate:1/2").over 4 [] = some (.law "subset:2") := by
  have hp : Select.parseRate "rate:1/2" = some (1, 2, 1) := parseRate_repr 1 2 (by decide) (by decide)
  have hr : Select.rate? "rate:1/2" = true := by
    unfold Select.rate?
    rw [hp]
    rfl
  refine ⟨by rw [ofJson_ruleOfText, hr]; rfl, ?_⟩
  have ht : Select.textOver "rate:1/2" 4 [] = some "subset:2" := by
    unfold Select.textOver
    rw [hp]
    rfl
  show (Select.textOver "rate:1/2" 4 []).map Select.Rule.law = _
  rw [ht]
  rfl

/-- **`RuleAtRate`'s nontrivial lemma**: the fixed count `subset:1` kept as it is over three units is none of `AtRate`'s
forms: it is no rate design, no stratified or work rate design and no Bernoulli design. A parser that takes every rule
object as it is written, as one that accepts every input does, gives it, so it does not have `RuleAtRate`'s conclusion.
-/
theorem ruleAtRate_nontrivial : ¬ AtRate (.law "subset:1") (.law "subset:1") 3 [] := by
  have h1 : Select.parseRate "subset:1" = none := parseRate_fixed "subset" (Or.inl rfl) 1
  have h2 : Select.parseStrataRate "subset:1" = none := parseStrataRate_fixed "subset" (Or.inl rfl) 1
  have h3 : Select.parseDesign "subset:1" = .ok (Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson 1)]) :=
    parseDesign_count "subset" (Or.inl rfl) 1 (Nat.le_refl 1)
  rintro ⟨t', -, ⟨num, den, tables, hp, -⟩ | ⟨num, den, hp, -⟩ | ⟨num, den, hp, -⟩ | ⟨l, -, hl, hb⟩⟩
  · rw [h1] at hp; contradiction
  · rw [h2] at hp; contradiction
  · rw [h2] at hp; contradiction
  · rw [h3] at hl
    obtain rfl := (Except.ok.inj hl).symm
    exact absurd hb (by decide)

end Properties.Spot.SecurityProofs
