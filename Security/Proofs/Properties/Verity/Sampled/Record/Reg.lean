import Proofs.Properties.Verity.Sampled.Binding
import Proofs.Properties.Verity.Sampled.Record
import Proofs.Flock.Soundness.Refine.WalkZeta
import Proofs.Flock.VBridge.Family.Names
import Proofs.Flock.Claims.Sampling.Select.Misses
import Std.Data.String.ToNat

/-!
# The registration v2's parameters name, at the record

* **The parameters' program is the one their verifier parses** (`program_parse_of_paramsAt`): one-stage v2's parameters
  (`SampledProofs.OneStageZk.paramsOfJson`) take the verifier's own program bytes, `own.program`, as the canonical form
  of the parameters' `"program"` value, and `SampledProofs.OneStage.paramsParsed` parses the verifier's program from the
  canonical form of the same value, so `Program.parse own.program` is the verifier's program.
* **v2's slots are two classes** (`partSlots_classes`): the population, in the registration's strata, and the input
  class, at most one stratum. The population's strata have the sizes of the layout's first class.
* **The registration is v2's parameters'** (`regIs_of_setsUp`): at a session v2 sets up, the statement's descriptor and
  query are the parameters' program bytes and verifier query (`RegIs`, from `SetsUp.params`), so the descriptor parses
  as the parameters' verifier's program.
* **`parseLaw` reads a law's text back** (`parseLaw_count`, `parseLaw_bernoulli`): `kind:K` at `K ≥ 1`, for a `subset`,
  `stratified` or `work` law, is `{law: kind, k: K}`, and `bernoulli:NUM/DEN` at `0 < DEN` and `NUM ≤ DEN` is
  `{law: bernoulli, p: "NUM/DEN"}`, each number in `Nat.repr`'s digits.
* **The population is `PopMatches`' count** (`units_of_popMatches`): where the descriptor parses and its layout under
  the query is laid out, `Partition.units` of the program and query is the `n` of `PopMatches`.
-/

namespace Verity.Sampled.Record

open Lean (Json)
open Claims.Sampling
open FlockSoundness.Refine (Ret throw_ne_ok)
open Verity.Sampled.Trees (bind_ok)
open Properties.SampledProofs.SecurityProofs (layout_ok units_eq popUnits strataOf_perm partition_evaluate_tiles
  laysOut_units length_popUnits)
open private split1 digits from SampledProofs.Select

/-- `OneStage.paramsParsed` parses its verifier's program from the canonical form of the parameters' `"program"`. -/
theorem paramsParsed_program {j : Json} {b : _root_.SampledProofs.OneStage.Params}
    (h : _root_.SampledProofs.OneStage.paramsParsed j = .ok b) :
    ∃ s, Flock.canon (_root_.SampledProofs.getD j "program") = .ok s ∧
      _root_.SampledProofs.Program.parse s.toUTF8 = .ok b.verifier.program := by
  unfold _root_.SampledProofs.OneStage.paramsParsed at h
  extract_lets at h
  replace h := bind_ok h
  obtain ⟨_, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨pj, hpj, h⟩ := h
  replace h := bind_ok h
  obtain ⟨s, hs, h⟩ := h
  replace h := bind_ok h
  obtain ⟨prog, hprog, h⟩ := h
  have hpj' : pj = _root_.SampledProofs.getD j "program" := by
    generalize _root_.SampledProofs.getD j "program" = x at hpj ⊢
    cases x <;> cases hpj
    rfl
  subst hpj'
  refine ⟨s, hs, ?_⟩
  suffices hb : b.verifier.program = prog by rw [hb]; exact hprog
  clear hprog hs
  revert h
  revert b
  change Ret _ (fun b : _root_.SampledProofs.OneStage.Params => b.verifier.program = prog)
  ret_walk_zeta

theorem oneStage_program {j : Json} {b : _root_.SampledProofs.OneStage.Params}
    (h : _root_.SampledProofs.OneStage.paramsOfJson j = .ok b) :
    ∃ s, Flock.canon (_root_.SampledProofs.getD j "program") = .ok s ∧
      _root_.SampledProofs.Program.parse s.toUTF8 = .ok b.verifier.program := by
  unfold _root_.SampledProofs.OneStage.paramsOfJson at h
  obtain ⟨b', hb', h⟩ := bind_ok h
  have hb : b = b' := by
    split at h
    · exact absurd h (throw_ne_ok _ _)
    · cases h
      rfl
  subst hb
  exact paramsParsed_program hb'

/-- One-stage v2's own program bytes parse as its verifier's program. -/
theorem zk_program {j : Json} {p : _root_.SampledProofs.OneStageZk.Params}
    (h : _root_.SampledProofs.OneStageZk.paramsOfJson j = .ok p) :
    _root_.SampledProofs.Program.parse p.own.program = .ok p.base.verifier.program := by
  unfold _root_.SampledProofs.OneStageZk.paramsOfJson at h
  repeat' first
    | exact absurd h (throw_ne_ok _ _)
    | (replace h := bind_ok h; obtain ⟨_, _, h⟩ := h)
    | (split at h)
  all_goals
    cases h
    have hc := ‹Flock.canon _ = Except.ok _›
    obtain ⟨s, hs, hp⟩ := oneStage_program ‹_root_.SampledProofs.OneStage.paramsOfJson _ = Except.ok _›
    rw [hc] at hs
    cases hs
    exact hp

/-- **The parameters' program is the one their verifier parses**: at parameters `ParamsAt θ p`, the verifier's own
program bytes parse as its verifier's program. -/
theorem program_parse_of_paramsAt {θ : ByteArray} {p : _root_.SampledProofs.OneStageZk.Params} (h : ParamsAt θ p) :
    _root_.SampledProofs.Program.parse p.own.program = .ok p.base.verifier.program := by
  obtain ⟨_, -, x, -, -, hx⟩ := h
  obtain ⟨_, -, hx⟩ := bind_ok hx
  obtain ⟨_, hr⟩ := Binding.dispatch_params hx
  exact zk_program hr

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

/-- **The statement's registration is v2's parameters'**: the statement's descriptor is the parameters' program bytes
and its query their verifier's. -/
def RegIs (A : Live) (p : _root_.SampledProofs.OneStageZk.Params) : Prop :=
  p.own.program = ⟨A.desc.toArray⟩ ∧ p.base.verifier.query = A.qr

/-- **At a session v2 sets up, the registration is v2's parameters'** (`RegIs`), and the statement's descriptor parses
as their verifier's program, at the parameters whose verifier's law the draw reads over the population. -/
theorem regIs_of_setsUp (A : Live) {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]
    {V : Vocab dv α} {θ : ByteArray} {F : Array ByteArray} (hsu : SetsUp V θ F A) :
    ∃ p, ParamsAt θ p ∧ RegIs A p ∧ lawAt p.base.verifier A.n = .ok A.law ∧
      _root_.SampledProofs.Program.parse ⟨A.desc.toArray⟩ = .ok p.base.verifier.program := by
  obtain ⟨p, hp, -, -, hl, hd, hq⟩ := hsu.params
  refine ⟨p, hp, ⟨hd, hq⟩, hl, ?_⟩
  rw [← hd]
  exact program_parse_of_paramsAt hp

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

end Verity.Sampled.Record
