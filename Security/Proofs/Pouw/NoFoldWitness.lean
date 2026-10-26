import Proofs.Pouw.Dither
import Proofs.Pouw.ShaRealize
import Proofs.Pouw.DitherWitness

/-!
# `WorkSoundNoFold`'s premises hold together, on a machine that meets `OracleCalls` (CMP-136)

The one-bound headline isn't vacuous: on the witness machine `nw`, round `nRound` and period `nPd` below, every premise
of `Definitions.Pouw.Guarantees.Dither.WorkSoundNoFold` holds at once, at `k = 1` round. Among them are the device
premises with `Accel.OracleCalls` at a uniform table `NX → ND` (`|ND| = 2^16`), SAK(matmul), SHA-512's speed, and
`ShaRealize` at the segmented schedule `q = (k + 1)(⌈Q⌉ + V) = 4`, `τ_0 = 2`. The conclusion's bounds say something:
`ε = ln 2 / 2 < 1` and `κ = 1/2 + 308/2^16 < 1` for every prover. The honest prover is accepted on every oracle and
coins (`witness`; `witness_freed` is the headline's conclusion there).

* `nw`: the witness machine over two resources (`0` MAC units, `1` hash units). A program is any function with its use
  and oracle calls, each call a step that uses a hash unit (G3). Its two laws (`NAlg`):
  - `words`: a program that outputs a round's words uses a MAC unit if its output varies over its inputs;
  - `table`: a program on an input `Orc × J` whose output varies over its inputs makes `|X|` calls: it reads the
    whole table.
  A program constant over its inputs is free.
* `nNF`, `nRound`: one word (`n = n' = nb = L = s = 1`), the noise the word itself (one of 16 values), the root the
  word's leaf hash `o (leaf w)` under the table, and the opening its check (a depth-0 Merkle tree).
* The period `nPd`: two resources at rate 1 for `T = 1`, prices `μ = (1, 1)`, credited use `(1, 1)`, `H_mm = H_sha = 1`
  and `H_o = 0` (`C = 2`), `η = 1` (so `η_mm = η_sha = 2`), `δ = 1/2`, the cap `Q = 1` and `V = 1`.

**Which machines it covers.** Machines on which a program that reads the oracle at all pays for the whole table, so
`OracleCalls` holds with the set of points read either everything or nothing (`nw_oracleCalls`). On such a machine no
prover within the cap reads the oracle: one whose rounds make at most `Q` calls has constant work and answers
(`allConst_of_cap`). So its words, its root and its answers read no answer of the round, and the walk's oracle reach
(`Definitions.Pouw.Assumptions.ShaRealize`'s Open) doesn't arise. `ShaRealize` holds for every prover with an empty
walk: the log is the one leaf its constant answer opens, logged after the root, among dummy queries (`realize_at`).

**Why so degenerate.** `Machine.Alg` is a function of types, and Lean can't prove `Orc × J = Orc × J' → J = J'`. So no
law can say which component of an input of type `Orc × J` is the oracle; a law that charges a program for the points it
reads must hold at every such decomposition. For programs built from arbitrary functions, the laws that can be proved
are the two above: the whole table or nothing. `Enc.node_inj` makes `|X| ≥ |D|²`, and `κ < 1` needs `q² < |D|` with
`q > Q`, so `|X| > Q`: a program within the cap that reads the oracle at all is over it. The honest prover reads the
oracle to commit, so it makes more than `Q` calls (`honest_calls_gt`), doesn't fit the period's devices, and the bound
says nothing about it. This holds for every witness on the current interface, not only this one. A machine whose oracle
is an argument of evaluation rather than a component of the input (`eval : Alg I O → (X → D) → I → O`, programs as
query trees) would let a law name the points a program queries, and give an honest prover within the cap.

**The premises' content here.**
* SAK(matmul): a varying words program costs at least `pub_mm = 1`. A constant one guesses the word, right on one
  dither in 16, so its moment is at most `e^{η_mm pub_mm}/16 = e²/16 < 1`.
* SHA-512's speed: a varying commitment program reads the table, `|X|` hash units, past `pub_sha = 1`. A constant one
  passes where the oracle sends the dither's leaf to its constant root, on at most one dither where the oracle is
  injective on the 16 leaves. `κ_sha = (16² + 16)/2^16` bounds a collision among them (`prob_bad_le`).
* No dither is degenerate (`κ_dmm = κ_dsha = 0`), advice is 0 bits, `ξw = ξl = 0`, nothing is exchanged
  (`β = K = 0`), and the verdict is the round's check (`κ_S = 0`).
-/

namespace Pouw.SecurityProofs.NoFoldWitness

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions
open Pouw.MerkleExtract (Enc Opens)
open Pouw.SecurityProofs.DitherWitness (prob_le_zero word_inj)

section Round

/-- The table's values: `2^16` digests. -/
abbrev ND : Type := Fin 65536

/-- The table's points: leaves (a word), nodes (two digests), and four points no tree hashes (the log's dummies). -/
abbrev NX : Type := Word ⊕ (ND × ND) ⊕ Fin 4

/-- The tree's encodings: leaves left, nodes in the middle. -/
def NE : Enc NX ND Word where
  leaf := Sum.inl
  node l r := Sum.inr (Sum.inl (l, r))
  leaf_inj := Sum.inl_injective
  node_inj := by
    intro l r l' r' h
    simpa using h
  disj := by
    intro w l r h
    simp at h

@[simp] theorem NE_leaf (w : Word) : NE.leaf w = Sum.inl w := rfl

/-- The oracle: a table. -/
abbrev NOrc : Type := NX → ND

/-- The witness round's words: one. -/
abbrev NTr : Type := Fin 1 → Fin 1 → Fin 1 → Word

/-- The witness round's dither: one of 16 values. -/
abbrev NDraw : Type := Fin 16

/-- **The witness round's knobs**: one word, the noise the word itself, a check of that word, and the root its leaf's
hash under the table, opened by that hash (a depth-0 tree). -/
def nNF : NoFold where
  n := 1
  n' := 1
  nb := 1
  L := 1
  Draw := NDraw
  noise _ d := (fun _ _ => ((d : ℕ) : Word), fun _ _ => 1)
  s := 1
  Chk := Unit
  cells _ _ := (0, 0, 0)
  Orc := NOrc
  Root := ND
  merkle o T := o (NE.leaf (T 0 0 0))
  Path := Unit
  opens o r _ v _ := o (NE.leaf v) = r

/-- The witness round. -/
abbrev nRound : RoundProto := noFoldRound nNF

/-- The oracle as the table. -/
def neo : nNF.Orc ≃ (NX → ND) := Equiv.refl _

/-- A path, as digests: none. -/
def nPl (_ : nNF.Path) : List (Bool × ND × ND) := []

/-- A cell's position in the tree: the root. -/
def nDirs (_ : Fin nNF.n × Fin nNF.n' × Fin nNF.nb) : List Bool := []

theorem nWord (a : nRound.Inst) (d : NDraw) (i j b : Fin 1) :
    noFoldWord nNF (nNF.noise a d) i j b = ((d : ℕ) : Word) := by
  show (∑ t : Fin 1, if t.val < (b.val + 1) * 1 then ((d : ℕ) : Word) * 1 else 0) = _
  simp

theorem nOk_iff (a : nRound.Inst) (d : NDraw) (c : Unit) (T : NTr) :
    nRound.ok a d c T ↔ T 0 0 0 = ((d : ℕ) : Word) := by
  show (∀ _ : Fin 1, T 0 0 0 = noFoldWord nNF (nNF.noise a d) (0 : Fin 1) (0 : Fin 1) (0 : Fin 1)) ↔ _
  rw [nWord]
  exact ⟨fun h => h 0, fun h _ => h⟩

theorem nOpens_iff (o : NOrc) (r : ND) (c : Unit) (T : NTr) :
    nRound.opens o r c T ↔ o (NE.leaf (T 0 0 0)) = r := by
  show (∀ _ : Fin 1, ∃ _ : Unit, o (NE.leaf (T 0 0 0)) = r) ↔ _
  exact ⟨fun h => (h 0).elim fun _ h => h, fun h _ => ⟨(), h⟩⟩

theorem nAccept_iff (o : NOrc) (a : nRound.Inst) (d : NDraw) (c : Unit) (r : ND) (y : Fin 1 → Word × Unit) :
    nRound.accept o a d c r y ↔ o (NE.leaf (y 0).1) = r ∧ (y 0).1 = ((d : ℕ) : Word) := by
  show (∀ q : Fin 1, o (NE.leaf (y q).1) = r ∧
    (y q).1 = noFoldWord nNF (nNF.noise a d) (0 : Fin 1) (0 : Fin 1) (0 : Fin 1)) ↔ _
  rw [nWord]
  exact ⟨fun h => h 0, fun h q => by rw [Subsingleton.elim q 0]; exact h⟩

/-- The verifier's check is the Merkle path check under the table. -/
theorem nHopens : ∀ o r c v p, nNF.opens o r c v p → Opens (neo o) NE (id r) (nPl p) v ∧
    (nPl p).map Prod.fst = nDirs c :=
  fun _ _ _ _ _ h => ⟨h, rfl⟩

end Round

section Machine

/-- A function varies over its inputs. -/
def Varies {I O : Type} (f : I → O) : Prop := ∃ y y', f y ≠ f y'

theorem eq_of_not_varies {I O : Type} {f : I → O} (h : ¬ Varies f) (y y' : I) : f y = f y' := by
  by_contra hne
  exact h ⟨y, y', hne⟩

/-- **A witness program**: what it computes, its use of the two resources (nonnegative), its oracle calls (each uses
a hash unit), and the machine's two laws, each read where the program's type has the shape it is about:
* `table`: a program on an input `Orc × J` whose output varies makes `|X|` calls, so it pays for the whole table;
* `words`: a program that outputs a round's words and varies uses a MAC unit. -/
structure NAlg (I O : Type) where
  run : I → O
  use : I → Fin 2 → ℝ
  use_nonneg : ∀ x r, 0 ≤ use x r
  calls : I → ℝ
  calls_nonneg : ∀ x, 0 ≤ calls x
  calls_le : ∀ x, calls x ≤ use x 1
  table : (∃ J : Type, I = (NOrc × J)) → Varies run → ∀ x, (Fintype.card NX : ℝ) ≤ calls x
  words : O = NTr → Varies run → ∀ x, 1 ≤ use x 0

open Classical in
/-- The calls the `table` law asks of a function: the whole table where it reads one, else none. -/
noncomputable def tableCalls {I O : Type} (f : I → O) : ℝ :=
  if (∃ J : Type, I = (NOrc × J)) ∧ Varies f then (Fintype.card NX : ℝ) else 0

open Classical in
/-- The MAC units the `words` law asks of a function. -/
noncomputable def wordCost {I O : Type} (f : I → O) : ℝ := if O = NTr ∧ Varies f then 1 else 0

theorem tableCalls_nonneg {I O : Type} (f : I → O) : 0 ≤ tableCalls f := by
  unfold tableCalls
  split_ifs
  · exact Nat.cast_nonneg _
  · exact le_rfl

theorem wordCost_nonneg {I O : Type} (f : I → O) : 0 ≤ wordCost f := by
  unfold wordCost
  split_ifs
  · exact zero_le_one
  · exact le_rfl

theorem tableCalls_eq {I O : Type} {f : I → O} (hJ : ∃ J : Type, I = (NOrc × J)) (hv : Varies f) :
    tableCalls f = Fintype.card NX := by
  unfold tableCalls
  rw [if_pos ⟨hJ, hv⟩]

theorem wordCost_eq {I O : Type} {f : I → O} (hO : O = NTr) (hv : Varies f) : wordCost f = 1 := by
  unfold wordCost
  rw [if_pos ⟨hO, hv⟩]

/-- Any function, at the use and calls its laws ask. -/
noncomputable def NAlg.ofFun {I O : Type} (f : I → O) : NAlg I O where
  run := f
  use _ r := if r = 1 then tableCalls f else wordCost f
  use_nonneg _ r := by
    split_ifs
    · exact tableCalls_nonneg f
    · exact wordCost_nonneg f
  calls _ := tableCalls f
  calls_nonneg _ := tableCalls_nonneg f
  calls_le _ := by
    refine le_of_eq ?_
    show tableCalls f = if (1 : Fin 2) = 1 then tableCalls f else wordCost f
    rw [if_pos rfl]
  table hJ hv _ := (tableCalls_eq hJ hv).ge
  words hO hv _ := by
    refine le_of_eq ?_
    show (1 : ℝ) = if (0 : Fin 2) = 1 then tableCalls f else wordCost f
    rw [if_neg (show ¬((0 : Fin 2) = 1) by decide), wordCost_eq hO hv]

theorem use_ofFun_const {I O : Type} {f : I → O} (hf : ¬ Varies f) (x : I) (r : Fin 2) :
    NAlg.use (NAlg.ofFun f) x r = 0 := by
  show (if r = 1 then tableCalls f else wordCost f) = 0
  simp [tableCalls, wordCost, hf]

/-- The witness machine. Its `time` is zero (the headline reads `Accel.use`, not `time`), and composing or routing
programs is any function at the use its laws ask. -/
noncomputable abbrev nm : Machine where
  Alg I O := NAlg I O
  eval f := f.run
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := NAlg.ofFun (g.run ∘ f.run)
  pair f g := NAlg.ofFun fun x => (f.run x, g.run x)
  fst := NAlg.ofFun Prod.fst
  snd := NAlg.ofFun Prod.snd
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

/-- The witness accelerator: two resources at rate 1, a run's wall time its total use, nothing exchanged, and an
oracle call a step that uses one hash unit. -/
noncomputable abbrev nw : Accel where
  Mc := nm
  Res := Fin 2
  rate _ := 1
  mem := 0
  use f x r := NAlg.use f x r
  held _ _ := 0
  wall f x := ∑ r, NAlg.use f x r
  ext _ _ := 0
  calls f x := NAlg.calls f x
  callUse r := if r = 1 then 1 else 0
  P1 := True
  P2 := True

theorem nw_G1 : nw.G1 := fun _ _ => by
  show (0 : ℝ) + 0 ≤ 0
  norm_num

theorem nw_G2 : nw.G2 := by
  refine ⟨fun _ => one_pos, fun {I O} f x => ⟨le_rfl, fun r => ⟨NAlg.use_nonneg f x r, ?_⟩⟩⟩
  show NAlg.use f x r ≤ 1 * ∑ r, NAlg.use f x r
  rw [one_mul]
  exact Finset.single_le_sum (f := fun r => NAlg.use f x r) (fun r _ => NAlg.use_nonneg f x r)
    (Finset.mem_univ r)

theorem nw_G3 : nw.G3 := by
  intro I O f x
  refine ⟨NAlg.calls_nonneg f x, fun (r : Fin 2) => ?_⟩
  show NAlg.calls f x * (if r = 1 then 1 else 0) ≤ NAlg.use f x r
  split_ifs with h
  · rw [mul_one, h]
    exact NAlg.calls_le f x
  · rw [mul_zero]
    exact NAlg.use_nonneg f x r

/-- Every point of the table. Irreducible: `whnf` of `y ∈ Finset.univ` at this size unfolds the list of points and
exceeds `maxRecDepth`. -/
@[irreducible] def univX : Finset NX := Finset.univ

theorem mem_univX (y : NX) : y ∈ univX := by
  unfold univX
  simp

theorem card_univX : univX.card = Fintype.card NX := by
  unfold univX
  exact Finset.card_univ

/-- **Programs reach the table only through their calls**: a program that varies reads the whole table and pays for
it, and one that doesn't reads nothing. -/
theorem nw_oracleCalls : nw.OracleCalls neo := by
  intro J O f o j
  by_cases hv : Varies (NAlg.run f)
  · refine ⟨univX, ?_, fun o' ho' => ?_⟩
    · rw [card_univX]
      exact NAlg.table f ⟨J, rfl⟩ hv (o, j)
    · have h : o' = o := funext fun y => ho' y (mem_univX y)
      rw [h]
  · refine ⟨∅, ?_, fun o' _ => eq_of_not_varies hv _ _⟩
    rw [Finset.card_empty, Nat.cast_zero]
    exact NAlg.calls_nonneg f (o, j)

theorem card_nx_gt : (1 : ℝ) < Fintype.card NX := by
  have h := Fintype.card_le_of_injective (fun i : Fin 4 => (Sum.inr (Sum.inr i) : NX))
    (fun a b h => by simpa using h)
  rw [Fintype.card_fin] at h
  have h4 : (4 : ℝ) ≤ Fintype.card NX := by exact_mod_cast h
  linarith

end Machine

section Sampling

/-- The prices: a unit of each resource at 1. -/
noncomputable def nμ (_ : Fin 2) : ℝ := 1

theorem nμ_nonneg (r : Fin 2) : 0 ≤ nμ r := by
  show (0 : ℝ) ≤ 1
  norm_num

theorem use_le_cost {I O : Type} (f : NAlg I O) (x : I) (r : Fin 2) : NAlg.use f x r ≤ nw.cost nμ f x := by
  show NAlg.use f x r ≤ ∑ r', nμ r' * NAlg.use f x r'
  calc NAlg.use f x r = nμ r * NAlg.use f x r := by rw [show nμ r = 1 from rfl, one_mul]
    _ ≤ ∑ r', nμ r' * NAlg.use f x r' :=
      Finset.single_le_sum (f := fun r => nμ r * NAlg.use f x r)
        (fun r _ => mul_nonneg (nμ_nonneg r) (NAlg.use_nonneg f x r)) (Finset.mem_univ r)

theorem cost_nonneg {I O : Type} (f : NAlg I O) (x : I) : 0 ≤ nw.cost nμ f x :=
  Finset.sum_nonneg fun r _ => mul_nonneg (nμ_nonneg r) (NAlg.use_nonneg f x r)

/-- A commitment program that reads the table costs at least 1. -/
theorem table_cost_ge {J O : Type} (f : NAlg (NOrc × J) O) (hv : Varies (NAlg.run f)) (z : NOrc × J) :
    (1 : ℝ) ≤ nw.cost nμ f z := by
  have h1 := NAlg.table f ⟨J, rfl⟩ hv z
  have h2 := NAlg.calls_le f z
  have h3 := use_le_cost f z 1
  have h4 := card_nx_gt
  linarith

theorem exp_two_lt : Real.exp 2 < 8 := by
  have h : Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
  rw [h]
  nlinarith [Real.exp_one_lt_d9, Real.exp_pos 1]

theorem exp_sf_le (η pub c : ℝ) (hη : η = 2) (hpub : pub = 1) (hc : 0 ≤ c) :
    Real.exp (η * roundShortfall pub c) ≤ Real.exp 2 := by
  subst hη hpub
  rw [Real.exp_le_exp, roundShortfall]
  have : max 0 (1 - c) ≤ 1 := max_le (by norm_num) (by linarith)
  linarith

theorem exp_sf_le_one (η pub c : ℝ) (h : pub ≤ c) : Real.exp (η * roundShortfall pub c) ≤ 1 := by
  rw [roundShortfall, max_eq_left (by linarith), mul_zero, Real.exp_zero]

theorem avg_le_one_of_le (g : NDraw × Unit → ℝ) (hg : ∀ x, g x ≤ 1) : avg g ≤ 1 := by
  unfold avg
  rw [div_le_one (Nat.cast_pos.mpr Fintype.card_pos)]
  calc ∑ x, g x ≤ ∑ _x : NDraw × Unit, (1 : ℝ) := Finset.sum_le_sum fun x _ => hg x
    _ = Fintype.card (NDraw × Unit) := by simp

open Classical in
/-- **The moment bound both public-best premises reduce to for a constant program**: a test on the round's coins
that passes on at most one dither and weighs a pass at most `e²` averages at most `e²/16 < 1`. -/
theorem avg_le_of_unique (P : NDraw → Prop) (hP : ∀ d d', P d → P d' → d = d') (g : NDraw × Unit → ℝ)
    (hg : ∀ x, g x ≤ if P x.1 then Real.exp 2 else 0) : avg g ≤ 1 := by
  have hcard : (Finset.univ.filter fun x : NDraw × Unit => P x.1).card ≤ 1 := by
    refine Finset.card_le_one.2 fun a ha b hb => ?_
    rw [Finset.mem_filter] at ha hb
    exact Prod.ext (hP _ _ ha.2 hb.2) (Subsingleton.elim _ _)
  have hc : ((Finset.univ.filter fun x : NDraw × Unit => P x.1).card : ℝ) ≤ 1 := by exact_mod_cast hcard
  unfold avg
  rw [div_le_one (Nat.cast_pos.mpr Fintype.card_pos)]
  calc ∑ x, g x ≤ ∑ x : NDraw × Unit, (if P x.1 then Real.exp 2 else 0) := Finset.sum_le_sum fun x _ => hg x
    _ = ((Finset.univ.filter fun x : NDraw × Unit => P x.1).card : ℝ) * Real.exp 2 := by
        rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
    _ ≤ 1 * 8 := mul_le_mul hc exp_two_lt.le (Real.exp_pos 2).le zero_le_one
    _ ≤ Fintype.card (NDraw × Unit) := by
        simp only [Fintype.card_prod, Fintype.card_fin, Fintype.card_unit]
        norm_num

/-- Round dither `i`'s leaf. -/
def lp (i : NDraw) : NX := NE.leaf ((i : ℕ) : Word)

theorem lp_inj : Function.Injective lp := fun _ _ h => word_inj (NE.leaf_inj h)

/-- The 16 leaves as a query log: the collision event's witness for `prob_bad_le`. -/
def collideLog : QueryLog nRound 1 NX ND 16 (fun _ => 0) where
  qry _ i _ := lp i
  qry_dep _ _ _ _ _ := rfl
  fresh _ _ _ _ hji h := hji.ne' (lp_inj h)
  root _ _ _ := 0
  root_dep _ _ _ _ _ := rfl

/-- SHA-512's speed's error: a collision among the 16 leaves, `(16² + 16)/2^16`. -/
noncomputable def nκsha : ℝ := 272 / 65536

/-- **The table collides on the 16 leaves** with probability at most `κ_sha`. -/
theorem prob_collide_le : prob (fun ox : nRound.Orc × (Fin 1 → nRound.Draw × nRound.Chk) =>
    ¬ Function.Injective fun i : NDraw => (ox.1 : NOrc) (lp i)) ≤ nκsha := by
  refine (Pouw.SecurityProofs.ShaRealize.prob_bad_le NE neo collideLog
    (fun o _ => ¬ Function.Injective fun i : NDraw => (o : NOrc) (lp i))
    fun o x u hu hb => Or.inr fun hinj => hb ?_).trans (le_of_eq ?_)
  · have h : (fun i : NDraw => (o : NOrc) (lp i)) = u := funext fun i => (hu i).symm
    rw [h]
    exact hinj
  · norm_num [nκsha, Fin.sum_univ_one]

end Sampling

section Period

/-- The period's credited use: one honest round of `(1, 1)`. -/
noncomputable def nHon (_ : Fin 2) : ℝ := 1

/-- **The witness period**: `T = 1` second of the two resources at rate 1, one other job per resource worth its use at
`μ`, and the credited use `hon` worth `C = μ · hon = 2`. -/
noncomputable def nPd : Period nw where
  T := 1
  J := Fin 2
  val := nμ
  needs j r := if j = r then 1 else 0
  hon := nHon
  rv := 2

/-- `μ` is the witness period's dual price at one credited unit: the honest round fills the devices. -/
theorem nDual : nPd.DualOpt nμ 1 := by
  refine ⟨nμ_nonneg, fun (j r : Fin 2) => ?_, fun (j : Fin 2) => ?_, fun _ => 0,
    ⟨fun _ => le_rfl, fun r => ?_⟩, ?_⟩
  · show (0 : ℝ) ≤ if j = r then 1 else 0
    split_ifs <;> norm_num
  · show nμ j ≤ ∑ r : Fin 2, nμ r * if j = r then 1 else 0
    simp [nμ]
  · show 1 * nHon r + ∑ j : Fin 2, (0 : ℝ) * (if j = r then 1 else 0) ≤ 1 * 1
    simp [nHon]
  · show ∑ j : Fin 2, nμ j * (0 : ℝ) = ∑ r : Fin 2, nμ r * (1 * 1 - 1 * nHon r)
    simp [nHon]

/-- A round's honest matmul cost at `μ`, its public best's too. -/
noncomputable def nHmm (_ : nRound.Inst) : ℝ := 1

/-- A round's honest hashing cost at `μ`, its public best's too. -/
noncomputable def nHsha (_ : nRound.Inst) : ℝ := 1

/-- The rest of a round's honest cost at `μ`. -/
noncomputable def nHo (_ : nRound.Inst) : ℝ := 0

/-- The rate a unit of priced cost: `η_mm = η_sha = 2`. -/
noncomputable def nη : ℝ := 1

/-- The confidence. -/
noncomputable def nδ : ℝ := 1 / 2

/-- The cap on a run's oracle calls: the period's hash units over a call's use of them. -/
noncomputable def nQ : ℝ := 1

theorem nCap : ∃ r, 0 < nw.callUse r ∧ nPd.cap r ≤ nQ * nw.callUse r := by
  refine ⟨1, ?_, ?_⟩
  · show (0 : ℝ) < if (1 : Fin 2) = 1 then 1 else 0
    norm_num
  · show (1 : ℝ) * 1 ≤ nQ * if (1 : Fin 2) = 1 then 1 else 0
    norm_num [nQ]

/-- The registered instance: the witness round's noise reads none, so any will do. -/
def nReg (_ : Fin 1) : nRound.Inst := (fun _ _ => 0, fun _ _ => 0)

theorem nCost :
    ∑ w : Fin 1, (nHmm (nReg w) + nHsha (nReg w) + nHo (nReg w)) = ∑ r, nμ r * nPd.hon r := by
  show ∑ _w : Fin 1, ((1 : ℝ) + 1 + 0) = ∑ r : Fin 2, nμ r * nHon r
  norm_num [Fin.sum_univ_two, nμ, nHon]

end Period

section Premises

/-- **The witness verdict**: the round's check on the transcript. The devices read nothing. -/
def nVerdict : Verdict nRound 1 Unit := fun o x t _ =>
  ∀ w, nRound.accept o (nReg w) (x w).1 (x w).2 (t w).1 (t w).2

/-- The devices' readings: nothing. -/
def nDev : Readings nw nRound 1 Unit := fun _ _ _ => ()

theorem nSAK : SAKMatmul nw nRound nμ Set.univ nHmm (fun a => nη * (nHmm a + nHsha a + nHo a) / nHmm a)
    (fun _ => 0) := by
  intro _ _ _ _ a _ B St _ s Mw _
  refine ⟨fun _ => False, prob_le_zero fun _ h => h, ?_⟩
  unfold wordsMoment
  by_cases hv : Varies (NAlg.run Mw)
  · refine avg_le_one_of_le _ fun x => ?_
    dsimp only
    split_ifs
    · exact exp_sf_le_one _ _ _ ((NAlg.words Mw rfl hv (s, x.1)).trans (use_le_cost Mw (s, x.1) 0))
    · exact zero_le_one
  · refine avg_le_of_unique (fun d => (NAlg.run Mw (s, (0 : NDraw)) : NTr) 0 0 0 = ((d : ℕ) : Word))
      (fun d d' h h' => word_inj (h.symm.trans h')) _ fun x => ?_
    dsimp only
    split_ifs with h1 h2
    · exact exp_sf_le _ _ _ (by norm_num [nη, nHmm, nHsha, nHo]) rfl (cost_nonneg _ _)
    · exfalso
      apply h2
      have hok := (nOk_iff _ _ _ _).1 h1.2
      rw [← hok]
      exact congrArg (fun T : NTr => T 0 0 0) (eq_of_not_varies hv _ _)
    · exact (Real.exp_pos 2).le
    · exact le_rfl

theorem nShaSpeed : ShaSpeed nw nRound 1 nReg nμ nHsha (fun a => nη * (nHmm a + nHsha a + nHo a) / nHsha a)
    (fun _ => 0) (fun _ => nκsha) := by
  intro _ _ _ _ _ A B Adv _ adv _ Mw Mh _ _ _
  refine le_trans (prob_mono fun ox hb => ?_) prob_collide_le
  intro hinj
  obtain ⟨w, hw⟩ := hb
  refine hw ⟨fun _ => False, prob_le_zero fun _ h => h, ?_⟩
  unfold commitMoment
  by_cases hv : Varies (NAlg.run (Mh w))
  · refine avg_le_one_of_le _ fun x => ?_
    dsimp only
    split_ifs
    · exact exp_sf_le_one _ _ _ (table_cost_ge (Mh w) hv _)
    · exact zero_le_one
  · refine avg_le_of_unique (fun d => ∃ z, (ox.1 : NOrc) (lp d) = NAlg.run (Mh w) z)
      (fun d d' ⟨z, hz⟩ ⟨z', hz'⟩ => hinj (hz.trans ((eq_of_not_varies hv z z').trans hz'.symm))) _
      fun x => ?_
    dsimp only
    split_ifs with h1 h2
    · exact exp_sf_le _ _ _ (by norm_num [nη, nHmm, nHsha, nHo]) rfl (cost_nonneg _ _)
    · exfalso
      apply h2
      have hok := (nOk_iff _ _ _ _).1 h1.2.1
      have hop := (nOpens_iff _ _ _ _).1 h1.2.2
      refine ⟨(ox.1, (adv ox.1 ox.2 w, x.1), NAlg.run (Mw w) (adv ox.1 ox.2 w, x.1)), ?_⟩
      show (ox.1 : NOrc) (NE.leaf ((x.1 : ℕ) : Word)) = _
      rw [← hok]
      exact hop
    · exact (Real.exp_pos 2).le
    · exact le_rfl

theorem nOutsource : OutsourceWithinEgress nw nμ 0 := by
  intro _ _ _ _ I O f
  exact ⟨f, fun i => ⟨rfl, rfl, by simp⟩⟩

theorem nService : ServiceSound nw nRound 1 nReg nVerdict nDev 0 := fun _ =>
  prob_le_zero fun _ h => h.2 h.1

theorem nImport : ImportWithinCertifier nw nRound 1 nVerdict nDev 0 := fun _ _ _ _ => by
  show ∑ _w : Fin 1, ((0 : ℝ) + 0) ≤ 0
  simp

end Premises

section Realize

/-- A fixed oracle and coins, where a constant program's outputs are read. -/
def o0 : NOrc := default

def x0 : Fin 1 → NDraw × Unit := default

/-- A prover whose round's work and answer are constant over their inputs. -/
def AllConst (A : Prover nw nRound 1) : Prop :=
  ∀ w, ¬ Varies (NAlg.run (A.work w)) ∧ ¬ Varies (NAlg.run (A.answer w))

/-- **Within the cap nothing reads the table**: a prover whose round makes at most `Q = 1` calls has constant work and
answers, since a varying one makes `|X| > 1`. -/
theorem allConst_of_cap (A : Prover nw nRound 1) (o : nRound.Orc) (x : Fin 1 → nRound.Draw × nRound.Chk)
    (h : A.calls o x ≤ nQ) : AllConst A := by
  have h' : ∑ w : Fin 1, (NAlg.calls (A.work w) (A.workIn o x w) + NAlg.calls (A.answer w) (A.answerIn o x w)) ≤
      1 := h
  rw [Fin.sum_univ_one] at h'
  have h1 := NAlg.calls_nonneg (A.work 0) (A.workIn o x 0)
  have h2 := NAlg.calls_nonneg (A.answer 0) (A.answerIn o x 0)
  have hX := card_nx_gt
  intro w
  obtain rfl : w = 0 := Subsingleton.elim _ _
  constructor
  · intro hv
    have := NAlg.table (A.work 0) ⟨_, rfl⟩ hv (A.workIn o x 0)
    linarith
  · intro hv
    have := NAlg.table (A.answer 0) ⟨_, rfl⟩ hv (A.answerIn o x 0)
    linarith

theorem root_const (A : Prover nw nRound 1) (hA : AllConst A) (o : nRound.Orc)
    (x : Fin 1 → nRound.Draw × nRound.Chk) (w : Fin 1) : A.root o x w = A.root o0 x0 w :=
  congrArg Prod.snd (eq_of_not_varies (hA w).1 _ _)

theorem ans_const (A : Prover nw nRound 1) (hA : AllConst A) (o : nRound.Orc)
    (x : Fin 1 → nRound.Draw × nRound.Chk) (w : Fin 1) : A.ans o x w = A.ans o0 x0 w :=
  congrArg Prod.snd (eq_of_not_varies (hA w).2 _ _)

theorem timedUse_nonneg (A : Prover nw nRound 1) (o : nRound.Orc) (x : Fin 1 → nRound.Draw × nRound.Chk)
    (r : Fin 2) : 0 ≤ A.timedUse o x r :=
  Finset.sum_nonneg fun _ _ => add_nonneg (NAlg.use_nonneg _ _ _) (NAlg.use_nonneg _ _ _)

/-- The word a constant prover's answer opens. -/
noncomputable def aw (A : Prover nw nRound 1) : Word := ((A.ans o0 x0 0) (0 : Fin 1)).1

/-- The log's points: the opened leaf third, dummies elsewhere. -/
def logPt (a : Word) (i : Fin 4) : NX := if (i : ℕ) = 2 then NE.leaf a else Sum.inr (Sum.inr i)

theorem logPt_inj (a : Word) : Function.Injective (logPt a) := by
  intro i j h
  by_cases hi : (i : ℕ) = 2 <;> by_cases hj : (j : ℕ) = 2
  · exact Fin.ext (hi.trans hj.symm)
  · simp [logPt, hi, hj] at h
  · simp [logPt, hi, hj] at h
  · simpa [logPt, hi, hj] using h

/-- **A prover's log** at a budget of 4 queries: the leaf its answer opens, after the root (`τ ≤ 2`), among dummies.
Its root is the prover's at a fixed oracle and coins. -/
noncomputable def nLog (A : Prover nw nRound 1) (q : ℕ) (hq : q = 4) (τ : Fin 1 → ℕ) :
    QueryLog nRound 1 NX ND q τ where
  qry _ i _ := logPt (aw A) (Fin.cast hq i)
  qry_dep _ _ _ _ _ := rfl
  fresh _ _ i j hji h := hji.ne' (Fin.ext (by
    have h' := congrArg Fin.val (logPt_inj _ h)
    exact h'))
  root _ w _ := A.root o0 x0 w
  root_dep _ _ _ _ _ := rfl

theorem pre_no_leaf (A : Prover nw nRound 1) (q : ℕ) (hq : q = 4) (τ : Fin 1 → ℕ) (hτ : ∀ w, τ w ≤ 2)
    (x : Fin 1 → nRound.Draw × nRound.Chk) (u : Fin q → ND) (w : Fin 1) (w' : Word) :
    NE.leaf w' ∉ ((nLog A q hq τ).prover x w).pre u := by
  rintro ⟨i, hi, h⟩
  have h' : logPt (aw A) (Fin.cast hq i) = NE.leaf w' := h
  have hi2 : (i : ℕ) ≠ 2 := by
    have := hτ w
    omega
  simp [logPt, hi2] at h'

theorem ext_nil (h : NX → ND) (S : Set NX) (d : ND) (hS : ∀ w', NE.leaf w' ∉ S) :
    Pouw.MerkleExtract.ext h NE S d [] = none := by
  have hn : ¬ ∃ w, NE.leaf w ∈ S ∧ h (NE.leaf w) = d := fun ⟨w', hw', _⟩ => hS w' hw'
  simp only [Pouw.MerkleExtract.ext]
  exact dif_neg hn

/-- The extraction's words program: the constant word 0 (the walk finds no leaf before the root). -/
noncomputable def mwZ : NAlg (Unit × NDraw) NTr := NAlg.ofFun fun _ => fun _ _ _ => 0

/-- The extraction's commitment program: the prover's own work, from the oracle and the dither. -/
noncomputable def mhA (A : Prover nw nRound 1) (w : Fin 1) : NAlg (NOrc × (Unit × NDraw) × NTr) ND :=
  NAlg.ofFun fun z => (NAlg.run (A.work w) (z.1, A.start, z.2.1.2)).2

theorem mwZ_use (z : Unit × NDraw) (r : Fin 2) : NAlg.use mwZ z r = 0 := by
  apply use_ofFun_const
  rintro ⟨y, y', h⟩
  exact h rfl

theorem mhA_use (A : Prover nw nRound 1) (hA : AllConst A) (w : Fin 1) (z : NOrc × (Unit × NDraw) × NTr)
    (r : Fin 2) : NAlg.use (mhA A w) z r = 0 := by
  apply use_ofFun_const
  rintro ⟨y, y', h⟩
  exact h (congrArg Prod.snd (eq_of_not_varies (hA w).1 _ _))

theorem mhA_root (A : Prover nw nRound 1) (o : nRound.Orc) (x : Fin 1 → nRound.Draw × nRound.Chk) (w : Fin 1) :
    nw.Mc.eval (mhA A w) (o, ((), (x w).1), nw.Mc.eval mwZ ((), (x w).1)) = A.root o x w := by
  obtain rfl : w = 0 := Subsingleton.elim _ _
  rfl

/-- **Every prover has a realization** at a budget of 4 queries, a schedule at most 2, advice of 0 bits and no walk
or log upkeep. -/
theorem realize_at (A : Prover nw nRound 1) (q : ℕ) (hq : q = 4) (τ : Fin 1 → ℕ) (hτ : ∀ w, τ w ≤ 2) :
    Realization nw nNF 1 nReg nQ NE neo id nPl nDirs (fun _ => 0) (fun _ => 0) A q τ 0 := by
  refine ⟨nLog A q hq τ, Unit, fun _ _ _ => (), fun _ => mwZ, fun w => mhA A w,
    ⟨fun _ => ⟨0, by norm_num⟩, fun a b _ => Subsingleton.elim a b⟩, fun _ _ _ _ _ => rfl,
    fun o x w => mhA_root A o x w, ?_, fun _ _ _ => ?_, ?_, ?_, ?_⟩
  · intro o x hcap r
    have hA := allConst_of_cap A o x hcap
    have h0 : ∀ w : Fin 1, NAlg.use mwZ ((), (x w).1) r +
        NAlg.use (mhA A w) (o, ((), (x w).1), NAlg.run mwZ ((), (x w).1)) r = 0 := fun w => by
      rw [mwZ_use, mhA_use A hA, add_zero]
    refine le_trans (le_of_eq (Finset.sum_eq_zero fun w _ => h0 w)) ?_
    show (0 : ℝ) ≤ A.timedUse o x r + (((1 : ℕ) : ℝ) * 0 + (q : ℝ) * 0)
    have := timedUse_nonneg A o x r
    simpa using this
  · show (0 : ℝ) + 0 ≤ 0
    norm_num
  · intro o x hcap u _ w
    exact root_const A (allConst_of_cap A o x hcap) o x w
  · intro o x hcap u _ _ w j y hy
    have hA := allConst_of_cap A o x hcap
    obtain rfl : w = 0 := Subsingleton.elim _ _
    obtain rfl : j = (0 : Fin 1) := @Subsingleton.elim (Fin 1) _ _ _
    have hy' : y ∈ [NE.leaf ((A.ans o x 0) (0 : Fin 1)).1] := hy
    rw [List.mem_singleton, ans_const A hA] at hy'
    refine ⟨Fin.cast hq.symm ⟨2, by norm_num⟩, ?_⟩
    rw [hy']
    rfl
  · intro o x _ u _ _ w c
    show (0 : Word) = (Pouw.MerkleExtract.ext _ NE _ _ []).getD 0
    rw [ext_nil _ _ _ (pre_no_leaf A q hq τ hτ x u w)]
    rfl

theorem hq4 : (1 + 1) * (⌈nQ⌉₊ + 1) = 4 := by
  simp [nQ]

theorem hτ2 : ∀ w : Fin 1, ((w : ℕ) + 1) * (⌈nQ⌉₊ + 1) ≤ 2 := by
  intro w
  rw [Subsingleton.elim w 0]
  simp [nQ]

/-- **`ShaRealize` on the witness machine**, at `WorkSoundNoFold`'s segmented schedule: `q = (k + 1)(⌈Q⌉ + V)`,
`τ_w = (w + 1)(⌈Q⌉ + V)` at `k = 1`, `Q = 1`, `V = 1`. -/
theorem nRealize : ShaRealize nw nNF 1 nReg nQ NE neo id nPl nDirs (fun _ => 0) (fun _ => 0)
    (fun _ => (1 + 1) * (⌈nQ⌉₊ + 1)) (fun _ w => ((w : ℕ) + 1) * (⌈nQ⌉₊ + 1)) (fun _ => 0) :=
  fun _ _ _ _ _ _ A => realize_at A _ hq4 _ hτ2

end Realize

section Honest

/-- The honest round's work: the word is the dither, the root its leaf's hash under the table. Its state is the word.
It reads the table, so it makes `|X|` calls. -/
noncomputable def hWork : NAlg (NOrc × Word × NDraw) (Word × ND) :=
  NAlg.ofFun fun p => (((p.2.2 : ℕ) : Word), p.1 (NE.leaf ((p.2.2 : ℕ) : Word)))

/-- The honest answer: the word, from the state. -/
noncomputable def hAnswer : NAlg (NOrc × Word × Unit) (Word × (Fin 1 → Word × Unit)) :=
  NAlg.ofFun fun p => (p.2.1, fun _ => (p.2.1, ()))

/-- **The honest prover**: the round computes the word, commits it and opens it. -/
noncomputable def honest : Prover nw nRound 1 where
  St := Word
  start := 0
  work _ := hWork
  answer _ := hAnswer

theorem honest_passes (o : nRound.Orc) (x : Fin 1 → nRound.Draw × nRound.Chk) :
    honest.Passes nVerdict nDev o x := fun _ =>
  (nAccept_iff _ _ _ _ _ _).2 ⟨rfl, rfl⟩

theorem hWork_varies : Varies (NAlg.run hWork) := by
  refine ⟨(default, 0, 0), (default, 0, 1), fun h => ?_⟩
  have h1 : (((0 : Fin 16) : ℕ) : Word) = (((1 : Fin 16) : ℕ) : Word) := congrArg Prod.fst h
  exact absurd (word_inj h1) (by decide)

/-- **The honest prover is past the cap**: its work reads the table. -/
theorem honest_calls_gt (o : nRound.Orc) (x : Fin 1 → nRound.Draw × nRound.Chk) : nQ < honest.calls o x := by
  have hW : tableCalls (NAlg.run hWork) = (Fintype.card NX : ℝ) := tableCalls_eq ⟨_, rfl⟩ hWork_varies
  have hA := tableCalls_nonneg (NAlg.run hAnswer)
  have hX := card_nx_gt
  have hQ : nQ = 1 := rfl
  show nQ < ∑ _w : Fin 1, (tableCalls (NAlg.run hWork) + tableCalls (NAlg.run hAnswer))
  rw [Fin.sum_univ_one, hW, hQ]
  linarith

end Honest

section Witness

/-- `WorkSoundNoFold`'s `ε` at the witness: `ln 2 / 2`. -/
noncomputable def nε : ℝ :=
  epsWorkSound (fun w : Fin 1 => nHmm (nReg w) + nHsha (nReg w) + nHo (nReg w)) (fun w => nHmm (nReg w))
      (fun w => nHsha (nReg w)) nη nδ 0 0 +
    (∑ r : Fin 2, nμ r * (((1 : ℕ) : ℝ) * (0 : ℝ) + (((1 + 1) * (⌈nQ⌉₊ + 1) : ℕ) : ℝ) * (0 : ℝ))) /
      ∑ w : Fin 1, (nHmm (nReg w) + nHsha (nReg w) + nHo (nReg w))

/-- `WorkSoundNoFold`'s `κ` at the witness: `1/2 + (q² + k q (q + 1))/2^16 + κ_sha = 1/2 + 308/2^16`. -/
noncomputable def nκ : ℝ :=
  nδ + 0 + (((1 + 1) * (⌈nQ⌉₊ + 1) * ((1 + 1) * (⌈nQ⌉₊ + 1)) +
      1 * ((1 + 1) * (⌈nQ⌉₊ + 1) * ((1 + 1) * (⌈nQ⌉₊ + 1) + 1)) : ℕ) : ℝ) / Fintype.card ND + nκsha +
    ((1 : ℕ) : ℝ) * ((0 : ℝ) + 0)

theorem nε_eq : nε = Real.log 2 / 2 := by
  simp only [nε, epsWorkSound, nHmm, nHsha, nHo, nη, nδ, nμ]
  norm_num

theorem nε_lt : nε < 1 := by
  rw [nε_eq]
  have := Real.log_two_lt_d9
  linarith

theorem nκ_lt : nκ < 1 := by
  have hc : ⌈nQ⌉₊ = 1 := by simp [nQ]
  simp only [nκ, hc, nδ, nκsha]
  norm_num

/-- **The witness statement** (CMP-136): every premise of `WorkSoundNoFold`, in its order, at the witness machine
`nw`, round `nRound` and period `nPd`, with `k = 1`, `Reg` every instance, `μ = (1, 1)`, `H_mm = pub_mm = 1`,
`H_sha = pub_sha = 1`, `H_o = 0`, `η = 1`, `δ = 1/2`, `β = K = κ_S = 0`, the cap `Q = 1`, `κ_sha = 272/2^16`,
`κ_dmm = κ_dsha = 0`, `B_adv = 0`, `V = 1`, the table `NX → ND` with `|ND| = 2^16`, and `ξw = ξl = 0`; then the
conclusion's bounds are nontrivial, `ε < 1` and `κ < 1`; the honest prover passes on every oracle and coins; and
its oracle calls are past the cap, which every honest prover is on a machine meeting `OracleCalls` this way. -/
def WitnessHolds : Prop :=
  (∀ w, nReg w ∈ (Set.univ : Set nRound.Inst)) ∧ nPd.DualOpt nμ 1 ∧
    (∀ a ∈ (Set.univ : Set nRound.Inst),
      0 < nHmm a ∧ 0 < nHsha a ∧ 0 ≤ nHo a ∧ nHmm a ≤ nHmm a ∧ nHsha a ≤ nHsha a) ∧
    ∑ w, (nHmm (nReg w) + nHsha (nReg w) + nHo (nReg w)) = ∑ r, nμ r * nPd.hon r ∧
    0 < nη ∧ 0 < nδ ∧ nδ ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧ (∀ _r : Fin 2, (0 : ℝ) ≤ 0) ∧
    (∀ _r : Fin 2, (0 : ℝ) ≤ 0) ∧ (∀ _B : ℕ, (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0) ∧
    (∃ r, 0 < nw.callUse r ∧ nPd.cap r ≤ nQ * nw.callUse r) ∧
    (∀ o r c v p, nNF.opens o r c v p → Opens (neo o) NE (id r) (nPl p) v ∧ (nPl p).map Prod.fst = nDirs c) ∧
    nw.G1 ∧ nw.G2 ∧ nw.G3 ∧ nw.OracleCalls neo ∧ nw.P1 ∧ nw.P2 ∧
    SAKMatmul nw nRound nμ Set.univ nHmm (fun a => nη * (nHmm a + nHsha a + nHo a) / nHmm a) (fun _ => 0) ∧
    ShaSpeed nw nRound 1 nReg nμ nHsha (fun a => nη * (nHmm a + nHsha a + nHo a) / nHsha a) (fun _ => 0)
      (fun _ => nκsha) ∧
    OutsourceWithinEgress nw nμ 0 ∧ ServiceSound nw nRound 1 nReg nVerdict nDev 0 ∧
    ImportWithinCertifier nw nRound 1 nVerdict nDev 0 ∧
    ShaRealize nw nNF 1 nReg nQ NE neo id nPl nDirs (fun _ => 0) (fun _ => 0)
      (fun _ => (1 + 1) * (⌈nQ⌉₊ + 1)) (fun _ w => ((w : ℕ) + 1) * (⌈nQ⌉₊ + 1)) (fun _ => 0) ∧
    nε < 1 ∧ nκ < 1 ∧ (∀ o x, honest.Passes nVerdict nDev o x) ∧ ∀ o x, nQ < honest.calls o x

theorem witness : WitnessHolds :=
  ⟨fun _ => Set.mem_univ _, nDual,
    fun _ _ => ⟨by norm_num [nHmm], by norm_num [nHsha], by norm_num [nHo], le_rfl, le_rfl⟩, nCost,
    by norm_num [nη], by norm_num [nδ], by norm_num [nδ], le_rfl, le_rfl, fun _ => le_rfl, fun _ => le_rfl,
    fun _ => ⟨le_rfl, le_rfl⟩, nCap, nHopens, nw_G1, nw_G2, nw_G3, nw_oracleCalls, trivial, trivial, nSAK,
    nShaSpeed, nOutsource, nService, nImport, nRealize, nε_lt, nκ_lt, honest_passes, honest_calls_gt⟩

/-- **The one-bound headline at the witness**: `WorkSoundNoFold`'s conclusion from `witness`. -/
theorem witness_freed :
    FreedSound nw nRound 1 nVerdict nDev nPd 1 (fun _ => nε) (∑ r, nμ r * (1 * nPd.hon r)) (fun _ => nκ) := by
  obtain ⟨hreg, hdual, hinst, hC, hη, hδ, hδ1, hβ, hK, hξw, hξl, hdeg, hcap, hopens, hG1, hG2, hG3, horc, hP1,
    hP2, hsak, hsha, hout, hsvc, himp, hR, -, -, -, -⟩ := witness
  exact Pouw.SecurityProofs.Dither.WorkSoundNoFold nw nNF 1 Set.univ nReg Unit nVerdict nDev nPd nμ nHmm nHsha nHo
    nHmm nHsha nη nδ 0 0 0 nQ nκsha (fun _ => 0) (fun _ => 0) 0 1 NX ND NE neo id nPl nDirs (fun _ => 0)
    (fun _ => 0) hreg hdual hinst hC hη hδ hδ1 hβ hK hξw hξl hdeg hcap hopens hG1 hG2 hG3 horc hP1 hP2 hsak hsha
    hout hsvc himp hR

end Witness

end Pouw.SecurityProofs.NoFoldWitness
