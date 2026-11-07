module

public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.Record
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.String.Search
import all Init.Data.String.Slice
public import Proofs.Flock.VBridge.Family.Names
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Pouw.Period.Sort
public import Std.Data.String.ToNat

/-!
# The registration v2's parameters name, at the record

* **v2's slots are two classes** (`partSlots_classes`): the population, in the registration's strata, and the input
  class, at most one stratum. The population's strata have the sizes of the layout's first class. The input class is
  the lowered program's, read on its own: nothing ties its size to the layout's, the descriptor's input gates.
* **The registration is v2's parameters'** (`regIs_of_setsUp`): at a session v2 sets up, the statement's descriptor
  parses as the parameters' verifier's program and its query is their verifier's (`SetsUp.params`).
* **`parseLaw` reads a law's text back** (`parseLaw_count`, `parseLaw_bernoulli`): `kind:K` at `K ≥ 1`, for a `subset`,
  `stratified` or `work` law, is `{law: kind, k: K}`, and `bernoulli:NUM/DEN` at `0 < DEN` and `NUM ≤ DEN` is
  `{law: bernoulli, p: "NUM/DEN"}`, each number in `Nat.repr`'s digits.
* **The population is `PopMatches`' count** (`units_of_popMatches`): where the descriptor parses and its layout under
  the query is laid out, `Partition.units` of the program and query is the `n` of `PopMatches`.
* **The design at the law's text is the law the draw reads** (`slotDesign_lawText`, `slotDesign_designSlots`): over the
  verifier's program and query of `n` units, where `lawAt` reads the law `L`, `slotDesign` under the rule
  `law (lawText L)` is `L`'s one-stage design and then `all`, at a subset law of at least one unit. `parseLaw` names a
  law that `deriveLaw` and `flockLaw` read as they read the verifier's (`deriveLaw_congr`, `deriveLaw_none`,
  `deriveLaw_some`, `flockLaw_subset_text`, `flockLaw_bernoulli_text`).
* **`RegMatches`' facts at a session** (`regMatches_of_setsUp`): at a session v2 sets up, the registration of the
  parameters' verifier under the rule `law (lawText A.law)` (`lawReg`) parses as the statement's descriptor, lays out
  under its query, has v2's two classes over its layout's population and has the design `designSlots A`, at a subset
  law of at least one unit.
* **`RegMatches` at the record** (`regMatches_record`): the record's code (`code`) names that registration at its one
  statement (`regOf`), and `RegMatches` holds there. Its partition is the population at the layout's strata sizes, then
  the statement's program's input class (`InputSlot.withInputs`), which is `partSlots A` by definition.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Verity.Sampled.Record

open Lean (Json)
open Claims.Sampling
open Verity.Sampled.Trees (bind_ok)
open Properties.SampledProofs.SecurityProofs (layout_ok units_eq popUnits strataOf_perm partition_evaluate_tiles
  laysOut_units length_popUnits lawObj kindOf flockLaw_cases deriveLaw_cases slotDesign_law bind_eq_ok throw_ne
  throw_bind_ne nat_toJson derive_bounds derive_ok deriveWork_ok)
open private split1 digits from SampledProofs.Select

theorem strataSizes_slotPartition (l : _root_.SampledProofs.Slots.Layout) :
    strataSizes ((_root_.Properties.SampledProofs.slotPartition l).classes.headD []) =
      (l.classes.toList.headD #[]).toList.map Array.size := by
  unfold _root_.Properties.SampledProofs.slotPartition strataSizes
  generalize l.classes.toList = cs
  cases cs <;> simp [Function.comp_def]

/-- **v2's slots are two classes**: the population in the registration's strata, whose sizes are those of the first
class of the registration's layout `l`, and the input class, at most one stratum. -/
theorem partSlots_classes (A : Live) {p : _root_.SampledProofs.Program} {l : _root_.SampledProofs.Slots.Layout}
    (hp : _root_.SampledProofs.Program.parse ⟨A.desc.toArray⟩ = .ok p)
    (hl : _root_.SampledProofs.Slots.layout p A.qr = .ok l) :
    ∃ pop ins, (partSlots A).classes = [pop, ins] ∧ ins.length ≤ 1 ∧
      strataSizes pop = strataSizes ((_root_.Properties.SampledProofs.slotPartition l).classes.headD []) := by
  refine ⟨_, InputSlot.inputClass (Flock.Bridge.progOf A.Cc A.κ), rfl, ?_, ?_⟩
  · unfold InputSlot.inputClass
    split <;> simp
  · rw [strataSizes_slotPartition, ← Flock.Bridge.nsOf_eq hp hl]
    unfold strataSizes
    apply List.map_splitLengths_length
    exact (Flock.Bridge.nsOf_sum A.hpop).le.trans (by simp)

/-- **At a session v2 sets up, the statement's registration is v2's parameters'**: the statement's descriptor parses as
their verifier's program and its query is their verifier's, at the parameters whose verifier's law the draw reads over
the population. -/
theorem regIs_of_setsUp (A : Live) {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]
    {V : Vocab dv α} {θ : ByteArray} {F : Array ByteArray} (hsu : SetsUp V θ F A) :
    ∃ p, ParamsAt θ p ∧ p.base.verifier.query = A.qr ∧ lawAt p.base.verifier A.n = .ok A.law ∧
      _root_.SampledProofs.Program.parse ⟨A.desc.toArray⟩ = .ok p.base.verifier.program := by
  obtain ⟨p, hp, -, -, hl, hd, hq, -⟩ := hsu.params
  exact ⟨p, hp, hq.symm, hl, hd⟩

/-! ## The law's text, read back -/

theorem isDigit_of_mem_repr {c : Char} {K : ℕ} (h : c ∈ (Nat.repr K).toList) : c.isDigit = true := by
  rw [Nat.toList_repr] at h
  exact Nat.isDigit_of_mem_toDigits (by decide) (by decide) h

theorem not_mem_repr {c : Char} (hc : c.isDigit = false) (K : ℕ) : c ∉ (Nat.repr K).toList := fun h => by
  rw [isDigit_of_mem_repr h] at hc
  exact absurd hc (by decide)

/-- `Select`'s `split1` at the separator `sep` of one character `c`, where `String.splitOn sep` is `List.splitOn c` on
the characters: `a ++ sep ++ b` is `a` and `b` where neither holds `c`. -/
theorem split1_sep (sep : String) (c : Char) (hsp : sep.toList = [c])
    (hsplit : ∀ s : String, s.splitOn sep = (s.toList.splitOn c).map String.ofList)
    (a b : String) (ha : c ∉ a.toList) (hb : c ∉ b.toList) : split1 (a ++ sep ++ b) sep = (a, b) := by
  have hne : ∀ l : List Char, c ∉ l → ∀ x ∈ l, (x == c) = false := fun l hl x hx => by
    simp only [beq_eq_false_iff_ne, ne_eq]
    rintro rfl
    exact hl hx
  unfold split1
  rw [hsplit, String.toList_append, String.toList_append, hsp, List.append_assoc, List.singleton_append,
    List.splitOn_eq_splitOnP, List.splitOnP_append_cons_of_forall_mem (hne _ ha) c (by simp),
    List.splitOnP_eq_singleton (hne _ hb)]
  simp

theorem split1_colon (a b : String) (ha : ':' ∉ a.toList) (hb : ':' ∉ b.toList) : split1 (a ++ ":" ++ b) ":" = (a, b) :=
  split1_sep ":" ':' (by decide) (FlockVBridge.Names.splitOn_sep ":" ':' rfl rfl rfl rfl (by decide)) a b ha hb

theorem split1_slash (a b : String) (ha : '/' ∉ a.toList) (hb : '/' ∉ b.toList) : split1 (a ++ "/" ++ b) "/" = (a, b) :=
  split1_sep "/" '/' (by decide) (FlockVBridge.Names.splitOn_sep "/" '/' rfl rfl rfl rfl (by decide)) a b ha hb

theorem digits_repr (K : ℕ) : digits (Nat.repr K) = true := by
  unfold digits
  rw [String.all_bool_eq, Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true]
  exact ⟨String.isEmpty_eq_false_iff.2 Nat.repr_ne_empty, fun c hc => isDigit_of_mem_repr hc⟩

theorem toNat!_repr (K : ℕ) : (Nat.repr K).toNat! = K := by
  have h := Nat.toNat?_repr K
  unfold String.toNat? String.Slice.toNat? at h
  unfold String.toNat! String.Slice.toNat!
  split at h
  · rename_i hi
    rw [if_pos hi]
    exact Option.some.inj h
  · exact absurd h (by simp)

/-- **`parseLaw` reads `kind:K` back**: a `subset`, `stratified` or `work` law of `K ≥ 1`, its count in `Nat.repr`'s
digits, is `{law: kind, k: K}`. -/
theorem parseLaw_count (kind : String) (hk : kind = "subset" ∨ kind = "stratified" ∨ kind = "work") (K : ℕ)
    (h1 : 1 ≤ K) :
    _root_.SampledProofs.Select.parseLaw (kind ++ ":" ++ Nat.repr K) =
      .ok (Json.mkObj [("law", Json.str kind), ("k", Lean.toJson K)]) := by
  have hka : ':' ∉ kind.toList := by rcases hk with rfl | rfl | rfl <;> decide
  unfold _root_.SampledProofs.Select.parseLaw
  rw [split1_colon kind (Nat.repr K) hka (not_mem_repr (by decide) K)]
  simp only [digits_repr, toNat!_repr, Bool.and_true, decide_eq_true h1]
  rcases hk with rfl | rfl | rfl <;> rfl

/-- **`parseLaw` reads `bernoulli:NUM/DEN` back**: at `0 < DEN` and `NUM ≤ DEN`, each in `Nat.repr`'s digits, it is
`{law: bernoulli, p: "NUM/DEN"}`. -/
theorem parseLaw_bernoulli (a b : ℕ) (hb : 0 < b) (hab : a ≤ b) :
    _root_.SampledProofs.Select.parseLaw ("bernoulli:" ++ (Nat.repr a ++ "/" ++ Nat.repr b)) =
      .ok (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]) := by
  unfold _root_.SampledProofs.Select.parseLaw
  rw [show "bernoulli:" ++ (Nat.repr a ++ "/" ++ Nat.repr b) =
    "bernoulli" ++ ":" ++ (Nat.repr a ++ "/" ++ Nat.repr b) from rfl, split1_colon _ _ (by decide) ?_]
  · dsimp only
    rw [split1_slash _ _ (not_mem_repr (by decide) a) (not_mem_repr (by decide) b)]
    simp only [digits_repr, toNat!_repr, decide_eq_true hb, decide_eq_true hab]
    rfl
  · rw [String.toList_append, String.toList_append]
    simp only [List.mem_append, not_or]
    exact ⟨⟨not_mem_repr (by decide) a, by decide⟩, not_mem_repr (by decide) b⟩

/-! ## The population -/

/-- **The population is `PopMatches`' count**: where the descriptor parses as `p` and `p`'s layout under the query is
`l`, `PopMatches desc q n` makes `Partition.units p q` the count `n`. -/
theorem units_of_popMatches {desc : Claims.Msg} {q : Json} {n : ℕ} {p : _root_.SampledProofs.Program}
    {l : _root_.SampledProofs.Slots.Layout} (h : Flock.Bridge.PopMatches desc q n)
    (hp : _root_.SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p)
    (hl : _root_.SampledProofs.Slots.layout p q = .ok l) : _root_.SampledProofs.Partition.units p q = .ok n := by
  obtain ⟨rs, he, pop, rfl, hlays⟩ := layout_ok p q _ hl
  rw [units_eq p q rs he]
  have hperm : (popUnits (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count)))).Perm
      (List.range' 0 (Flock.TemplateQuery.population rs)) :=
    strataOf_perm _ _ (partition_evaluate_tiles p q rs he)
  have hsum := Flock.Bridge.nsOf_sum h
  rw [Flock.Bridge.nsOf_eq hp hl] at hsum
  change (pop.toList.map Array.size).sum = n at hsum
  rw [(laysOut_units _ _ hlays).2, ← length_popUnits, hperm.length_eq, List.length_range'] at hsum
  rw [hsum]

/-! ## The design at the law's text -/

theorem sort_law_k :
    _root_.SampledProofs.sortStrings #["k", "law"] = _root_.SampledProofs.sortStrings #["law", "k"] :=
  Array.toList_inj.1 (Properties.Pouw.Period.SecurityProofs.Sort.qsort_eq_of_perm (List.Perm.swap "law" "k" []))

theorem keys_count (s : String) (K : ℕ) :
    _root_.SampledProofs.keys? (Json.mkObj [("law", Json.str s), ("k", Lean.toJson K)]) =
      some (_root_.SampledProofs.sortStrings #["law", "k"]) := by
  rw [← sort_law_k]
  unfold _root_.SampledProofs.keys?
  with_unfolding_all rfl

theorem keys_bernoulli (p : String) :
    _root_.SampledProofs.keys? (Json.mkObj [("law", "bernoulli"), ("p", Json.str p)]) =
      some (_root_.SampledProofs.sortStrings #["law", "p"]) := by
  unfold _root_.SampledProofs.keys?
  with_unfolding_all rfl

theorem kindOf_count (s : String) (K : ℕ) : kindOf (Json.mkObj [("law", Json.str s), ("k", Lean.toJson K)]) = s :=
  (rfl)

theorem kindOf_bernoulli (p : String) : kindOf (Json.mkObj [("law", "bernoulli"), ("p", Json.str p)]) = "bernoulli" :=
  (rfl)

theorem getD_law_of_kindOf {law : Json} {s : String} (h : kindOf law = s) (hs : s ≠ "") :
    _root_.SampledProofs.getD law "law" = .str s := by
  unfold kindOf at h
  split at h
  · rename_i s' hs'
    rw [hs', h]
  · exact absurd h.symm hs

/-- **`deriveLaw` reads only a law's kind and `k`.** -/
theorem deriveLaw_congr (p : _root_.SampledProofs.Program) (q : Json) (tbl : Option Json) {a b : Json}
    (hl : _root_.SampledProofs.getD a "law" = _root_.SampledProofs.getD b "law")
    (hn : _root_.SampledProofs.nat? (_root_.SampledProofs.getD a "k") =
      _root_.SampledProofs.nat? (_root_.SampledProofs.getD b "k")) :
    _root_.SampledProofs.Draw.deriveLaw p q a tbl = _root_.SampledProofs.Draw.deriveLaw p q b tbl := by
  unfold _root_.SampledProofs.Draw.deriveLaw
  rw [hl, hn]

/-- **`deriveLaw` expands neither a subset nor a Bernoulli law.** -/
theorem deriveLaw_none (p : _root_.SampledProofs.Program) (q : Json) (tbl : Option Json) {law : Json}
    (h1 : kindOf law ≠ "stratified") (h2 : kindOf law ≠ "work") :
    _root_.SampledProofs.Draw.deriveLaw p q law tbl = .ok none := by
  unfold _root_.SampledProofs.Draw.deriveLaw
  split
  next s hs =>
    have hk : kindOf law = s := by simp [kindOf, hs]
    rw [hk] at h1 h2
    simp only [Bool.not_or, Bool.and_eq_true, Bool.not_eq_eq_eq_not, Bool.not_true, beq_eq_false_iff_ne, ne_eq, h1, h2,
      not_false_eq_true, and_self, ↓reduceIte]
    rfl
  next => rfl

/-- `Flock.Draw.deriveWork` refuses `k = 0`. -/
theorem deriveWork_k {k : ℕ} {S : Array (String × Array (ℕ × ℕ))} {tab : Array (String × ℕ × ℕ)}
    {L : Flock.Draw.Law} (h : Flock.Draw.deriveWork k S tab = .ok L) : 1 ≤ k := by
  unfold Flock.Draw.deriveWork at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Nat.not_lt] at hc
    exact hc.1

/-- **`deriveLaw`'s expansion is the law's kind at its `k`**, of at least one unit. -/
theorem deriveLaw_some {p : _root_.SampledProofs.Program} {q law : Json} {tbl : Option Json} {L : Flock.Draw.Law}
    (h : _root_.SampledProofs.Draw.deriveLaw p q law tbl = .ok (some L)) :
    ∃ k, _root_.SampledProofs.nat? (_root_.SampledProofs.getD law "k") = some k ∧ 1 ≤ k ∧
      ((kindOf law = "stratified" ∧ ∃ S, L = .stratified k S) ∨ (kindOf law = "work" ∧ ∃ S, L = .work k S)) := by
  unfold _root_.SampledProofs.Draw.deriveLaw at h
  split at h
  next s hs =>
    have hk : kindOf law = s := by simp [kindOf, hs]
    dsimp only at h
    split at h
    · cases h
    · rename_i hc
      obtain ⟨S, -, h⟩ := bind_eq_ok.1 h
      split at h
      · rename_i k hnat
        split at h
        · rename_i hs'
          obtain ⟨L', hd, h⟩ := bind_eq_ok.1 h
          cases h
          refine ⟨k, hnat, (derive_bounds k S _ hd).1, Or.inl ⟨?_, _, derive_ok k S _ hd⟩⟩
          rw [hk]
          simpa using hs'
        · rename_i hs'
          split at h
          · obtain ⟨tab, -, h⟩ := bind_eq_ok.1 h
            obtain ⟨L', hd, h⟩ := bind_eq_ok.1 h
            cases h
            obtain ⟨_, -, hL⟩ := deriveWork_ok k S tab _ hd
            refine ⟨k, hnat, deriveWork_k hd, Or.inr ⟨?_, _, hL⟩⟩
            rw [hk]
            by_cases hw : s = "work"
            · exact hw
            · exfalso
              apply hc
              simp only [beq_iff_eq] at hs'
              simp [hs', hw]
          · exact absurd h (throw_ne _ _)
      · exact absurd h (throw_ne _ _)
  next =>
    dsimp only at h
    split at h
    · cases h
    · rename_i hc
      exact absurd (by decide) hc

theorem flockLaw_subset_text (K n : ℕ) (hn : K ≤ n) :
    _root_.SampledProofs.Draw.flockLaw (Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson K)]) n =
      .ok (.subset K) := by
  unfold _root_.SampledProofs.Draw.flockLaw
  rw [show _root_.SampledProofs.getD (Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson K)]) "law" =
    .str "subset" from rfl]
  dsimp only
  rw [keys_count, show _root_.SampledProofs.getD (Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson K)]) "k" =
    Lean.toJson K from rfl, nat_toJson]
  simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, Nat.not_lt.2 hn]
  rfl

theorem splitOn_slash_repr (a b : ℕ) :
    (Nat.repr a ++ "/" ++ Nat.repr b).splitOn "/" = [Nat.repr a, Nat.repr b] := by
  have hne : ∀ l : List Char, '/' ∉ l → ∀ x ∈ l, (x == '/') = false := fun l hl x hx => by
    simp only [beq_eq_false_iff_ne, ne_eq]
    rintro rfl
    exact hl hx
  rw [FlockVBridge.Names.splitOn_sep "/" '/' rfl rfl rfl rfl (by decide), String.toList_append, String.toList_append,
    show "/".toList = ['/'] from rfl, List.append_assoc, List.singleton_append, List.splitOn_eq_splitOnP,
    List.splitOnP_append_cons_of_forall_mem (hne _ (not_mem_repr (by decide) a)) '/' (by simp),
    List.splitOnP_eq_singleton (hne _ (not_mem_repr (by decide) b))]
  simp only [List.map_cons, List.map_nil, String.ofList_toList]

theorem probability_repr (a b : ℕ) (hb : 0 < b) (hab : a ≤ b) :
    _root_.SampledProofs.Draw.probability? s!"{a}/{b}" = some (a, b) := by
  unfold _root_.SampledProofs.Draw.probability?
  rw [show s!"{a}/{b}" = Nat.repr a ++ "/" ++ Nat.repr b from rfl, splitOn_slash_repr]
  dsimp only
  simp only [toNat!_repr, decide_eq_true hb, decide_eq_true hab, Bool.and_true]
  split
  · rfl
  · rename_i hc
    exact absurd (Bool.and_eq_true_iff.2 ⟨digits_repr a, digits_repr b⟩) hc

theorem flockLaw_bernoulli_text (a b n : ℕ) (hb : 0 < b) (hab : a ≤ b) :
    _root_.SampledProofs.Draw.flockLaw (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]) n =
      .ok (.bernoulli a b) := by
  unfold _root_.SampledProofs.Draw.flockLaw
  rw [show _root_.SampledProofs.getD (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]) "law" =
    .str "bernoulli" from rfl]
  dsimp only
  rw [keys_bernoulli, show _root_.SampledProofs.getD (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")])
    "p" = .str s!"{a}/{b}" from rfl]
  dsimp only
  rw [probability_repr a b hb hab]
  simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  rfl

/-- **`lawAt`'s steps**: the verifier's law's expansion, the law `flockLaw` reads from its law object, and the
expansion where there is one. -/
theorem lawAt_ok {v : _root_.SampledProofs.Registration.Verifier} {n : ℕ} {L : Flock.Draw.Law}
    (h : lawAt v n = .ok L) :
    ∃ derived parsed, _root_.SampledProofs.Draw.deriveLaw v.program v.query v.law v.workTable = .ok derived ∧
      _root_.SampledProofs.Draw.flockLaw (lawObj v.law derived) n = .ok parsed ∧ L = derived.getD parsed := by
  unfold lawAt _root_.SampledProofs.Draw.derive at h
  obtain ⟨lawJ, hJ, h⟩ := bind_eq_ok.1 h
  obtain ⟨d, hd, hJ⟩ := bind_eq_ok.1 hJ
  obtain ⟨d', hd', h⟩ := bind_eq_ok.1 h
  rw [hd] at hd'
  cases hd'
  cases hJ
  obtain ⟨fl, hf, h⟩ := bind_eq_ok.1 h
  exact ⟨d, fl, hd, hf, (Except.ok.inj h).symm⟩

/-- **A law's text**, as a selection rule names it (`Select.Rule.law`): `kind:K` for a `subset`, `stratified` or
`work` law of `K` units, and `bernoulli:NUM/DEN`, each number in `Nat.repr`'s digits. -/
def lawText : Flock.Draw.Law → String
  | .subset k => "subset" ++ ":" ++ Nat.repr k
  | .bernoulli a b => "bernoulli:" ++ (Nat.repr a ++ "/" ++ Nat.repr b)
  | .stratified k _ => "stratified" ++ ":" ++ Nat.repr k
  | .work k _ => "work" ++ ":" ++ Nat.repr k

/-- **The design at the law's text is the law `lawAt` reads**: over the verifier's program and query of `n` units,
where `lawAt` reads the law `L`, `slotDesign` under the rule `law (lawText L)` is `L`'s one-stage design, then `all`,
at a subset law of at least one unit. -/
theorem slotDesign_lawText {v : _root_.SampledProofs.Registration.Verifier} {n : ℕ} {L : Flock.Draw.Law}
    (hu : _root_.SampledProofs.Partition.units v.program v.query = .ok n) (hl : lawAt v n = .ok L)
    (hk : ∀ k, L = .subset k → 1 ≤ k) :
    _root_.Properties.SampledProofs.slotDesign (.law (lawText L)) v.program v.query v.workTable =
      .ok ⟨[_root_.Properties.SampledProofs.oneStageOf L, .all]⟩ := by
  obtain ⟨derived, parsed, hd, hf, rfl⟩ := lawAt_ok hl
  have hne : _root_.SampledProofs.Select.Rule.law (lawText (derived.getD parsed)) ≠ .all := nofun
  cases derived with
  | none =>
    rcases flockLaw_cases _ _ _ hf with ⟨k, rfl, hkn, -⟩ | ⟨a, b, rfl, hb, hab⟩ | hs | hw
    · exact slotDesign_law _ _ _ _ n _ none _ hne hu (parseLaw_count "subset" (Or.inl rfl) k (hk k rfl))
        (deriveLaw_none _ _ _ (by rw [kindOf_count]; decide) (by rw [kindOf_count]; decide))
        (flockLaw_subset_text k n hkn)
    · exact slotDesign_law _ _ _ _ n _ none _ hne hu (parseLaw_bernoulli a b hb hab)
        (deriveLaw_none _ _ _ (by rw [kindOf_bernoulli]; decide) (by rw [kindOf_bernoulli]; decide))
        (flockLaw_bernoulli_text a b n hb hab)
    · rcases deriveLaw_cases _ _ _ _ _ hd with ⟨-, h, -⟩ | ⟨_, _, _, -, h, -⟩
      · exact absurd hs h
      · cases h
    · rcases deriveLaw_cases _ _ _ _ _ hd with ⟨-, -, h⟩ | ⟨_, _, _, -, h, -⟩
      · exact absurd hw h
      · cases h
  | some L' =>
    obtain ⟨k, hnat, h1, ⟨hs, S, rfl⟩ | ⟨hw, S, rfl⟩⟩ := deriveLaw_some hd
    · refine slotDesign_law _ _ _ _ n _ _ _ hne hu (parseLaw_count "stratified" (Or.inr (Or.inl rfl)) k h1) ?_ hf
      rw [← hd]
      exact deriveLaw_congr _ _ _ (getD_law_of_kindOf hs (by decide)).symm (by rw [hnat]; exact nat_toJson k)
    · refine slotDesign_law _ _ _ _ n _ _ _ hne hu (parseLaw_count "work" (Or.inr (Or.inr rfl)) k h1) ?_ hf
      rw [← hd]
      exact deriveLaw_congr _ _ _ (getD_law_of_kindOf hw (by decide)).symm (by rw [hnat]; exact nat_toJson k)

/-- **v2's design at the law's text**: over the verifier's program and query of `A`'s population, where `lawAt` reads
`A`'s law, the design under `law (lawText A.law)` is `designSlots A`. -/
theorem slotDesign_designSlots (A : Live) {v : _root_.SampledProofs.Registration.Verifier}
    (hu : _root_.SampledProofs.Partition.units v.program v.query = .ok A.n) (hl : lawAt v A.n = .ok A.law)
    (hk : ∀ k, A.law = .subset k → 1 ≤ k) :
    _root_.Properties.SampledProofs.slotDesign (.law (lawText A.law)) v.program v.query v.workTable =
      .ok (designSlots A) :=
  slotDesign_lawText hu hl hk

/-! ## `RegMatches` at a session -/

/-- **v2's registration at a session**: the verifier `v` under the rule of `A`'s law's text, with the work closure
`cl`. -/
def lawReg (A : Live) (v : _root_.SampledProofs.Registration.Verifier) (cl : Option (Array (Int × Array Int))) :
    _root_.SamplingDeploy.Reg :=
  ⟨v, .law (lawText A.law), cl⟩

/-- **`RegMatches`' facts at a session v2 sets up**: at the parameters `SetsUp.params` names, whose verifier's query is the
statement's, the registration of their verifier (`lawReg`) parses as the statement's descriptor, lays out under the statement's query as `l`, has v2's two
classes with the population's strata of `slotPartition l`'s sizes, and has the design `designSlots A`, at a subset law
of at least one unit. -/
theorem regMatches_of_setsUp (A : Live) {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]
    {V : Vocab dv α} {θ : ByteArray} {F : Array ByteArray} (hsu : SetsUp V θ F A)
    (hk : ∀ k, A.law = .subset k → 1 ≤ k) (cl : Option (Array (Int × Array Int))) :
    ∃ p l, ParamsAt θ p ∧ p.base.verifier.query = A.qr ∧
      _root_.SampledProofs.Program.parse ⟨A.desc.toArray⟩ = .ok (lawReg A p.base.verifier cl).program ∧
      _root_.SampledProofs.Slots.layout (lawReg A p.base.verifier cl).program (lawReg A p.base.verifier cl).query =
        .ok l ∧
      (∃ pop ins, (partSlots A).classes = [pop, ins] ∧ ins.length ≤ 1 ∧
        strataSizes pop = strataSizes ((_root_.Properties.SampledProofs.slotPartition l).classes.headD [])) ∧
      _root_.Properties.SampledProofs.slotDesign (lawReg A p.base.verifier cl).rule
        (lawReg A p.base.verifier cl).program (lawReg A p.base.verifier cl).query (lawReg A p.base.verifier cl).table =
          .ok (designSlots A) := by
  obtain ⟨p, hp, hq, hlaw, hparse⟩ := regIs_of_setsUp A hsu
  obtain ⟨p', l, hp', hl, -⟩ := Flock.Bridge.popMatches_iff.1 A.hpop
  obtain rfl := Except.ok.inj (hparse.symm.trans hp')
  refine ⟨p, l, hp, hq, hparse, ?_, partSlots_classes A hparse hl, ?_⟩
  · show _root_.SampledProofs.Slots.layout p.base.verifier.program p.base.verifier.query = .ok l
    rw [hq]
    exact hl
  · exact slotDesign_designSlots A (v := p.base.verifier) (by rw [hq]; exact units_of_popMatches A.hpop hparse hl)
      hlaw hk

open Classical in
/-- **The registration the record's statement names**: `lawReg` at the verifier `v` at the record's one statement,
none elsewhere. -/
noncomputable def regOf (A : Live) (o : Own) (v : _root_.SampledProofs.Registration.Verifier)
    (cl : Option (Array (Int × Array Int))) (x : Claims.Msg) : Option _root_.SamplingDeploy.Reg :=
  if x = xOf A o then some (lawReg A v cl) else none

/-- **`RegMatches` at the record**: at a session v2 sets up, at the parameters `SetsUp.params` names, the record's code
and `regOf` at their verifier meet `RegMatches`. The record's one statement names that registration, whose program is
the statement's descriptor's parse and lays out under its query as `l`. Its partition `partSlots A` is the population, in
strata of `slotPartition l`'s population's sizes, then the statement's program's input class, and its design is
`slotDesign`'s. -/
theorem regMatches_record (A : Live) (o : Own) (e : Ext) {ι : Type} {dv : Accounting.Device ι} {α : Type}
    [Fintype α] {V : Vocab dv α} {θ : ByteArray} {F : Array ByteArray} (hsu : SetsUp V θ F A)
    (hk : ∀ k, A.law = .subset k → 1 ≤ k) (cl : Option (Array (Int × Array Int))) :
    ∃ p, ParamsAt θ p ∧ _root_.SamplingDeploy.RegMatches (code A o e) (regOf A o p.base.verifier cl) := by
  obtain ⟨p, l, hp, -, hparse, hl, ⟨pop, ins, hc, -, hss⟩, hds⟩ := regMatches_of_setsUp A hsu hk cl
  have h : [pop, ins] = [(Flock.Bridge.nsOf A.desc A.qr).splitLengths
      ((List.finRange A.n).map (Flock.Bridge.unitGates A.P)), InputSlot.inputClass (Flock.Bridge.progOf A.Cc A.κ)] :=
    hc.symm
  obtain rfl := (List.cons.inj h).1
  refine ⟨p, hp, ?_⟩
  intro x s Pt hx hP
  have hpe : (code A o e).parse x = if x = xOf A o then some (stmtSlots A) else none := rfl
  have hre : regOf A o p.base.verifier cl x = if x = xOf A o then some (lawReg A p.base.verifier cl) else none := rfl
  have hpp : (code A o e).partition s = some (partSlots A) := rfl
  by_cases hxo : x = xOf A o
  · obtain rfl : stmtSlots A = s := Option.some.inj ((hpe.trans (ite_eq_left hxo)).symm.trans hx)
    obtain rfl : partSlots A = Pt := Option.some.inj (hpp.symm.trans hP)
    exact ⟨lawReg A p.base.verifier cl, l, (Flock.Bridge.nsOf A.desc A.qr).splitLengths
      ((List.finRange A.n).map (Flock.Bridge.unitGates A.P)), hre.trans (ite_eq_left hxo), hparse, hl, rfl, hss, hds⟩
  · cases (hpe.trans (ite_eq_right hxo)).symm.trans hx

end Verity.Sampled.Record
