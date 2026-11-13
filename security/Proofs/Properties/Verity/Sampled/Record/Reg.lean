module

public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.Sampled.Record
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.String.Search
import all Init.Data.String.Slice
public import Proofs.Flock.VBridge.Family.UnitName
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Pouw.Period.Sort
public import Std.Data.String.ToNat

/-!
# The registration v2's parameters name, at the record

* **v2's slots are two classes** (`partSlots_classes`): the population, in the registration's strata, and the input
  class, at most one stratum. The population's strata have the sizes of the layout's first class.
* **The registration is v2's parameters'** (`regIs_of_setsUp`): at a session v2 sets up, the statement's descriptor
  parses as the parameters' verifier's program and its query is their verifier's (`SetsUp.params`).
* **`parseDesign` reads a law's text back** (`parseDesign_count`, `parseDesign_bernoulli`): `kind:K` at `K ≥ 1`, for a `subset`,
  `stratified` or `work` law, is `{law: kind, k: K}`, and `bernoulli:NUM/DEN` at `0 < DEN` and `NUM ≤ DEN` is
  `{law: bernoulli, p: "NUM/DEN"}`, each number in `Nat.repr`'s digits.
* **The population is `PopMatches`' count** (`units_of_popMatches`): where the descriptor parses and its layout under
  the query is laid out, `Partition.units` of the program and query is the `n` of `PopMatches`.
* **The design at the law's text is the law the draw reads** (`slotDesign_lawText`, `slotDesign_designSlots`): over the
  verifier's program and query of `n` units, where `lawAt` reads the law `L`, `slotDesign` under the rule
  `law (lawText L)` is `L`'s one-stage design and then `all`, at a subset law of at least one unit. `parseDesign` names a
  law that `deriveLaw` and `flockLaw` read as they read the verifier's (`deriveLaw_congr`, `deriveLaw_none`,
  `deriveLaw_some`, `flockLaw_subset_text`, `flockLaw_bernoulli_text`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Verity.Sampled.Record

open Lean (Json)
open Claims.Sampling
open Verity.Sampled.Trees (bind_ok)
open Properties.Spot.SecurityProofs (layout_ok units_eq popUnits strataOf_perm partition_evaluate_tiles
  laysOut_units length_popUnits lawObj kindOf flockLaw_cases deriveLaw_cases slotDesign_law bind_eq_ok throw_ne
  throw_bind_ne nat_toJson derive_bounds derive_ok deriveWork_ok)
open private split1 digits from Spot.Select

theorem strataSizes_slotPartition (l : _root_.Spot.Slots.Layout) :
    strataSizes ((_root_.Properties.Spot.slotPartition l).classes.headD []) =
      (l.classes.toList.headD #[]).toList.map Array.size := by
  unfold _root_.Properties.Spot.slotPartition strataSizes
  generalize l.classes.toList = cs
  cases cs <;> simp [Function.comp_def]

/-- **v2's slots are two classes**: the population in the registration's strata, whose sizes are those of the first
class of the registration's layout `l`, and the input class, at most one stratum. -/
theorem partSlots_classes (A : Live) {p : _root_.Spot.Program} {l : _root_.Spot.Slots.Layout}
    (hp : _root_.Spot.Program.parse ⟨A.desc.toArray⟩ = .ok p)
    (hl : _root_.Spot.Slots.layout p A.qr = .ok l) :
    ∃ pop ins, (partSlots A).classes = [pop, ins] ∧ ins.length ≤ 1 ∧
      strataSizes pop = strataSizes ((_root_.Properties.Spot.slotPartition l).classes.headD []) := by
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
    ∃ p, ParamsAt θ p ∧ p.base.verifier.query = A.qr ∧ lawAt p.base.verifier A.n = .ok A.design ∧
      _root_.Spot.Program.parse ⟨A.desc.toArray⟩ = .ok p.base.verifier.program := by
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

/-- **`parseDesign` reads `kind:K` back**: a `subset`, `stratified` or `work` law of `K ≥ 1`, its count in `Nat.repr`'s
digits, is `{law: kind, k: K}`. -/
theorem parseDesign_count (kind : String) (hk : kind = "subset" ∨ kind = "stratified" ∨ kind = "work") (K : ℕ)
    (h1 : 1 ≤ K) :
    _root_.Spot.Select.parseDesign (kind ++ ":" ++ Nat.repr K) =
      .ok (Json.mkObj [("law", Json.str kind), ("k", Lean.toJson K)]) := by
  have hka : ':' ∉ kind.toList := by rcases hk with rfl | rfl | rfl <;> decide
  unfold _root_.Spot.Select.parseDesign
  rw [split1_colon kind (Nat.repr K) hka (not_mem_repr (by decide) K)]
  simp only [digits_repr, toNat!_repr, Bool.and_true, decide_eq_true h1]
  rcases hk with rfl | rfl | rfl <;> rfl

/-- **`parseDesign` reads `bernoulli:NUM/DEN` back**: at `0 < DEN` and `NUM ≤ DEN`, each in `Nat.repr`'s digits, it is
`{law: bernoulli, p: "NUM/DEN"}`. -/
theorem parseDesign_bernoulli (a b : ℕ) (hb : 0 < b) (hab : a ≤ b) :
    _root_.Spot.Select.parseDesign ("bernoulli:" ++ (Nat.repr a ++ "/" ++ Nat.repr b)) =
      .ok (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]) := by
  unfold _root_.Spot.Select.parseDesign
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
theorem units_of_popMatches {desc : Claims.Msg} {q : Json} {n : ℕ} {p : _root_.Spot.Program}
    {l : _root_.Spot.Slots.Layout} (h : Flock.Bridge.PopMatches desc q n)
    (hp : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p)
    (hl : _root_.Spot.Slots.layout p q = .ok l) : _root_.Spot.Partition.units p q = .ok n := by
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
    _root_.Spot.sortStrings #["k", "law"] = _root_.Spot.sortStrings #["law", "k"] :=
  Array.toList_inj.1 (Properties.Pouw.Period.SecurityProofs.Sort.qsort_eq_of_perm (List.Perm.swap "law" "k" []))

theorem keys_count (s : String) (K : ℕ) :
    _root_.Spot.keys? (Json.mkObj [("law", Json.str s), ("k", Lean.toJson K)]) =
      some (_root_.Spot.sortStrings #["law", "k"]) := by
  rw [← sort_law_k]
  unfold _root_.Spot.keys?
  with_unfolding_all rfl

theorem keys_bernoulli (p : String) :
    _root_.Spot.keys? (Json.mkObj [("law", "bernoulli"), ("p", Json.str p)]) =
      some (_root_.Spot.sortStrings #["law", "p"]) := by
  unfold _root_.Spot.keys?
  with_unfolding_all rfl

theorem kindOf_count (s : String) (K : ℕ) : kindOf (Json.mkObj [("law", Json.str s), ("k", Lean.toJson K)]) = s :=
  (rfl)

theorem kindOf_bernoulli (p : String) : kindOf (Json.mkObj [("law", "bernoulli"), ("p", Json.str p)]) = "bernoulli" :=
  (rfl)

theorem getD_law_of_kindOf {law : Json} {s : String} (h : kindOf law = s) (hs : s ≠ "") :
    _root_.Spot.getD law "law" = .str s := by
  unfold kindOf at h
  split at h
  · rename_i s' hs'
    rw [hs', h]
  · exact absurd h.symm hs

/-- **`deriveLaw` reads only a law's kind and `k`.** -/
theorem deriveLaw_congr (p : _root_.Spot.Program) (q : Json) (tbl : Option Json) {a b : Json}
    (hl : _root_.Spot.getD a "law" = _root_.Spot.getD b "law")
    (hn : _root_.Spot.nat? (_root_.Spot.getD a "k") =
      _root_.Spot.nat? (_root_.Spot.getD b "k")) :
    _root_.Spot.Draw.deriveLaw p q a tbl = _root_.Spot.Draw.deriveLaw p q b tbl := by
  unfold _root_.Spot.Draw.deriveLaw
  rw [hl, hn]

/-- **`deriveLaw` expands neither a subset nor a Bernoulli law.** -/
theorem deriveLaw_none (p : _root_.Spot.Program) (q : Json) (tbl : Option Json) {law : Json}
    (h1 : kindOf law ≠ "stratified") (h2 : kindOf law ≠ "work") :
    _root_.Spot.Draw.deriveLaw p q law tbl = .ok none := by
  unfold _root_.Spot.Draw.deriveLaw
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
theorem deriveLaw_some {p : _root_.Spot.Program} {q law : Json} {tbl : Option Json} {L : Flock.Draw.Law}
    (h : _root_.Spot.Draw.deriveLaw p q law tbl = .ok (some L)) :
    ∃ k, _root_.Spot.nat? (_root_.Spot.getD law "k") = some k ∧ 1 ≤ k ∧
      ((kindOf law = "stratified" ∧ ∃ S, L = .stratified k S) ∨ (kindOf law = "work" ∧ ∃ S, L = .work k S)) := by
  unfold _root_.Spot.Draw.deriveLaw at h
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

theorem flockLaw_subset_text (K n : ℕ) (hk : 1 ≤ K) (hn : K ≤ n) :
    _root_.Spot.Draw.flockLaw (Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson K)]) n =
      .ok (.subset K) := by
  unfold _root_.Spot.Draw.flockLaw
  rw [show _root_.Spot.getD (Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson K)]) "law" =
    .str "subset" from rfl]
  dsimp only
  rw [keys_count, show _root_.Spot.getD (Json.mkObj [("law", Json.str "subset"), ("k", Lean.toJson K)]) "k" =
    Lean.toJson K from rfl, nat_toJson]
  simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, Nat.not_lt.2 hk, Nat.not_lt.2 hn]
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
    _root_.Spot.Draw.probability? s!"{a}/{b}" = some (a, b) := by
  unfold _root_.Spot.Draw.probability?
  rw [show s!"{a}/{b}" = Nat.repr a ++ "/" ++ Nat.repr b from rfl, splitOn_slash_repr]
  dsimp only
  simp only [toNat!_repr, decide_eq_true hb, decide_eq_true hab, Bool.and_true]
  split
  · rfl
  · rename_i hc
    exact absurd (Bool.and_eq_true_iff.2 ⟨digits_repr a, digits_repr b⟩) hc

theorem flockLaw_bernoulli_text (a b n : ℕ) (hb : 0 < b) (hab : a ≤ b) :
    _root_.Spot.Draw.flockLaw (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]) n =
      .ok (.bernoulli a b) := by
  unfold _root_.Spot.Draw.flockLaw
  rw [show _root_.Spot.getD (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]) "law" =
    .str "bernoulli" from rfl]
  dsimp only
  rw [keys_bernoulli, show _root_.Spot.getD (Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")])
    "p" = .str s!"{a}/{b}" from rfl]
  dsimp only
  rw [probability_repr a b hb hab]
  simp only [bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  rfl

/-- **`lawAt`'s steps**: the verifier's law's expansion, the law `flockLaw` reads from its law object, and the
expansion where there is one. -/
theorem lawAt_ok {v : _root_.Spot.Registration.Verifier} {n : ℕ} {L : Flock.Draw.Law}
    (h : lawAt v n = .ok L) :
    ∃ derived parsed, _root_.Spot.Draw.deriveLaw v.program v.query v.law v.workTable = .ok derived ∧
      _root_.Spot.Draw.flockLaw (lawObj v.law derived) n = .ok parsed ∧ L = derived.getD parsed := by
  unfold lawAt _root_.Spot.Draw.derive at h
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
theorem slotDesign_lawText {v : _root_.Spot.Registration.Verifier} {n : ℕ} {L : Flock.Draw.Law}
    (hu : _root_.Spot.Partition.units v.program v.query = .ok n) (hl : lawAt v n = .ok L)
    (hk : ∀ k, L = .subset k → 1 ≤ k) :
    _root_.Properties.Spot.slotDesign (.law (lawText L)) v.program v.query v.workTable =
      .ok ⟨[_root_.Properties.Spot.oneStageOf L, .all]⟩ := by
  obtain ⟨derived, parsed, hd, hf, rfl⟩ := lawAt_ok hl
  have hne : _root_.Spot.Select.Rule.law (lawText (derived.getD parsed)) ≠ .all := nofun
  cases derived with
  | none =>
    rcases flockLaw_cases _ _ _ hf with ⟨k, rfl, hkn, -⟩ | ⟨a, b, rfl, hb, hab⟩ | hs | hw
    · exact slotDesign_law _ _ _ _ n _ none _ hne hu (parseDesign_count "subset" (Or.inl rfl) k (hk k rfl))
        (deriveLaw_none _ _ _ (by rw [kindOf_count]; decide) (by rw [kindOf_count]; decide))
        (flockLaw_subset_text k n (hk k rfl) hkn)
    · exact slotDesign_law _ _ _ _ n _ none _ hne hu (parseDesign_bernoulli a b hb hab)
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
    · refine slotDesign_law _ _ _ _ n _ _ _ hne hu (parseDesign_count "stratified" (Or.inr (Or.inl rfl)) k h1) ?_ hf
      rw [← hd]
      exact deriveLaw_congr _ _ _ (getD_law_of_kindOf hs (by decide)).symm (by rw [hnat]; exact nat_toJson k)
    · refine slotDesign_law _ _ _ _ n _ _ _ hne hu (parseDesign_count "work" (Or.inr (Or.inr rfl)) k h1) ?_ hf
      rw [← hd]
      exact deriveLaw_congr _ _ _ (getD_law_of_kindOf hw (by decide)).symm (by rw [hnat]; exact nat_toJson k)

/-- **v2's design at the law's text**: over the verifier's program and query of `A`'s population, where `lawAt` reads
`A`'s law, the design under `law (lawText A.design)` is `designSlots A`. -/
theorem slotDesign_designSlots (A : Live) {v : _root_.Spot.Registration.Verifier}
    (hu : _root_.Spot.Partition.units v.program v.query = .ok A.n) (hl : lawAt v A.n = .ok A.design)
    (hk : ∀ k, A.design = .subset k → 1 ≤ k) :
    _root_.Properties.Spot.slotDesign (.law (lawText A.design)) v.program v.query v.workTable =
      .ok (designSlots A) :=
  slotDesign_lawText hu hl hk

end Verity.Sampled.Record
