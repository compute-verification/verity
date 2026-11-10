module

public import Proofs.Properties.Verity.Sampled.Leaves
public import Proofs.Properties.Verity.Sampled.OpenErr
public import Proofs.Properties.Verity.Sampled.TreeGap
import Proofs.Flock.Soundness.CROnly.Hiding

/-!
# Sampled proofs' coupling: the leaves' class errors, paid once each

An opening's node costs the coupling its gap and its empty class, and both are within the node's class error
(`openGap_add_openEmpty_le_round`, in `Proofs.Properties.Verity.Sampled.OpenErr`). This file reads that class error off
the leaf the node opens, and pays for every leaf once:

* **A node's class error** (`openErr`, `openErr_eq`, `openGap_add_openEmpty_le_openErr`,
  `openGap_add_openEmpty_le_openErr_link`): the class error of the node's fiber partitioned by a round's release at
  the count of the frame the prover sent, read off any description of the node. It bounds the node's gap and empty
  class at a round's release and at a link's, of every count the node's frame has; where the frame asks for more coins
  than a block holds, no round is released and the class error is 1.
* **A leaf's class error** (`leafErr`, `leafErrMax`, `avg_leafErrMax_le`, `avg_leafErrMax_key_le`): the class error of
  the leaf's fiber at the worst count, within `2·δ₁` over the device's sample of the round, and within `2·2^-257`
  (`leafTerm`) over the tree's key as well (`avg_hidingGap_tree_le`).
* **At a tree's rounds** (`roundErr`, `treeErr`, `openErr_le_roundErr`, `avg_treeErr_hello_le`): a node's class error
  is within its round's, and a tree's unopened rounds' class errors are within `maxTreeCoins` leaves' terms over the
  sample its `Hello` samples.
* **The potential** (`treePotential`, `avg_treePotential_hello_le`, `roundErr_add_treePotential_le`,
  `treePotential_nil`): the open trees' unopened rounds' class errors, and `maxTreeCoins` leaves' terms for each
  coin-tree line the log has left. A `Hello` keeps it on average and an opening pays its round's class error out of
  it, and from no tree open it is the hiding term.
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

/-- **A class error read through a bijection** of the releases is the class error. -/
theorem classErr_equiv {Q Q' : Type} [Fintype Q] [Fintype Q'] (φ : Q ≃ Q') (N : ℕ) (k : Q' → ℕ) :
    classErr N (fun u => k (φ u)) = classErr N k := by
  have hJ : (Finset.univ.filter fun v => 0 < k (φ v)).card = (Finset.univ.filter fun v => 0 < k v).card :=
    Finset.card_equiv φ fun v => by simp
  simp only [classErr, classNum]
  rw [hJ, Fintype.card_congr φ,
    Fintype.sum_equiv φ _ (fun u => (Finset.univ.filter fun v => 0 < k v).card * k u - N) fun _ => rfl]

/-- **No class nonempty**: the class error of a partition whose classes are all empty is 1. -/
theorem classErr_zero {Q : Type} [Fintype Q] [Nonempty Q] {N : ℕ} (hN : N ≠ 0) :
    classErr N (fun _ : Q => 0) = 1 := by
  have h : classNum N (fun _ : Q => 0) = Fintype.card Q * N := by simp [classNum, Nat.mul_comm]
  rw [classErr, h, Nat.cast_mul]
  exact ENNReal.div_self (mul_ne_zero (by exact_mod_cast Fintype.card_ne_zero) (by exact_mod_cast hN))
    (ENNReal.mul_ne_top (ENNReal.natCast_ne_top _) (ENNReal.natCast_ne_top _))

/-- A subtype's size, under any of its `Fintype` instances, is its predicate's count. -/
theorem card_subtype_eq {α : Type} [Fintype α] {P : α → Prop} [DecidablePred P] {iF : Fintype {x // P x}} :
    Fintype.card {x // P x} = (Finset.univ.filter P).card :=
  Finset.card_bij (fun d _ => d.1) (fun d _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, d.2⟩)
    (fun _ _ _ _ h => Subtype.ext h) fun x hx => ⟨⟨x, (Finset.mem_filter.1 hx).2⟩, Finset.mem_univ _, rfl⟩

/-- An event of a subtype's points, counted, is the event and the predicate counted in the whole type. -/
theorem card_filter_subtype {α : Type} [Fintype α] {P : α → Prop} [DecidablePred P] {iF : Fintype {x // P x}}
    (Q : α → Prop) [DecidablePred Q] :
    (Finset.univ.filter fun d : {x // P x} => Q d.1).card = (Finset.univ.filter fun x => P x ∧ Q x).card :=
  Finset.card_bij (fun d _ => d.1)
    (fun d hd => Finset.mem_filter.2 ⟨Finset.mem_univ _, d.2, (Finset.mem_filter.1 hd).2⟩)
    (fun _ _ _ _ h => Subtype.ext h) fun x hx =>
      ⟨⟨x, (Finset.mem_filter.1 hx).2.1⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, (Finset.mem_filter.1 hx).2.2⟩, rfl⟩

open Classical in
/-- **A coin left as it is falls short of nothing.** -/
theorem shortfall_id {P C : Type} [Fintype P] [Nonempty P] [Fintype C] :
    shortfall (fun (_ : P) (d : C) => d) = 0 := by
  have h : ∀ e : C, (avg fun d : C => if d = e then (1 : ℝ≥0∞) else 0) = (Fintype.card C : ℝ≥0∞)⁻¹ := fun e => by
    unfold avg
    simp
  simp only [shortfall, h, avg_const, tsub_self, Finset.sum_const_zero]

/-- **A mean over a dependent function of what reads one of its values** is the mean over that value. -/
theorem avg_eval {C : Type} [DecidableEq C] [Fintype C] {T : C → Type} [∀ c, Fintype (T c)] [∀ c, Nonempty (T c)]
    (c : C) (F : T c → ℝ≥0∞) : avg (fun f : (c : C) → T c => F (f c)) = avg F :=
  (avg_pi_at c fun a _ => F a).trans (avg_const _)

/-- **The rounds from `o` on, split at `o`**: round `o`'s term, and the rounds from `o + 1` on. -/
theorem sum_ite_le {N : ℕ} (o : ℕ) (h : o < N) (f : Fin N → ℝ≥0∞) :
    (∑ r : Fin N, if o ≤ r.1 then f r else 0) = (∑ r : Fin N, if o + 1 ≤ r.1 then f r else 0) + f ⟨o, h⟩ := by
  have hr : ∀ r : Fin N, (if o ≤ r.1 then f r else 0) =
      (if o + 1 ≤ r.1 then f r else 0) + if (⟨o, h⟩ : Fin N) = r then f r else 0 := fun r => by
    by_cases h1 : o + 1 ≤ r.1
    · have h2 : (⟨o, h⟩ : Fin N) ≠ r := fun h3 => by
        have h4 : o = r.1 := congrArg Fin.val h3
        omega
      rw [if_pos (show o ≤ r.1 by omega), if_pos h1, if_neg h2, add_zero]
    · by_cases h2 : o = r.1
      · rw [if_pos h2.le, if_neg h1, if_pos (Fin.ext h2 : (⟨o, h⟩ : Fin N) = r), zero_add]
      · rw [if_neg (show ¬ o ≤ r.1 by omega), if_neg h1,
          if_neg fun h3 : (⟨o, h⟩ : Fin N) = r => h2 (congrArg Fin.val h3), add_zero]
  rw [Fintype.sum_congr _ _ hr, Finset.sum_add_distrib, Finset.sum_ite_eq Finset.univ (⟨o, h⟩ : Fin N) f,
    if_pos (Finset.mem_univ _)]

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## No round released -/

/-- A sample's first `n` coins are 128 bits each, for each coin the block holds. -/
theorem length_opened {block : ℕ} (d : TreeDraw block) (n : ℕ) :
    (TreeDraw.opened d n).length = 128 * min n block := by
  have h : (List.ofFn fun i => List.ofFn (d.1 i)).map List.length = List.replicate block 128 := by
    rw [List.map_ofFn, ← List.ofFn_const]
    exact congrArg List.ofFn (funext fun i => List.length_ofFn)
  rw [TreeDraw.opened, List.length_flatten, List.map_take, h, List.take_replicate, List.sum_replicate_nat,
    Nat.mul_comm]

/-- **More coins than a block holds are never released**: a sample's first `n` coins, for `n` past its block, are no
string of `128·n` bits. -/
theorem opened_ne_of_length {block n : ℕ} (hb : block < n) (d : TreeDraw block) {bits : List Bool}
    (hl : bits.length = 128 * n) : TreeDraw.opened d n ≠ bits := fun h => by
  have h1 := congrArg List.length h
  rw [length_opened, hl, min_eq_right hb.le] at h1
  omega

variable {β : Type} {E : List Release → Game β} {g' : Game β}

theorem openGap_none (h : openingOf E g' = none) {P : Type} [Fintype P] (bits : P → List Bool) :
    openGap E g' bits = 0 := by
  unfold openGap
  rw [h]
  rfl

theorem openEmpty_none (h : openingOf E g' = none) {P : Type} [Fintype P] (bits : P → List Bool) :
    openEmpty E g' bits = 0 := by
  unfold openEmpty
  refine le_antisymm ((prCoin_mono fun u hu => ?_).trans_eq prCoin_false) zero_le
  obtain ⟨o, ho, -⟩ := hu
  rw [h] at ho
  cases ho

/-- **No class nonempty at an opening's node**: where no round of the node is released as any `bits u`, the written
round is the tape's own, so the gap is none, and the empty class is at most 1. -/
theorem openGap_add_openEmpty_le_one {P : Type} [Fintype P] [Nonempty P] {o : OpeningAt E g'}
    (ho : openingOf E g' = some o) (bits : P → List Bool) (hb : ∀ u, o.classOf (bits u) = ∅) :
    openGap E g' bits + openEmpty E g' bits ≤ 1 := by
  have hp : (fun u => pickIn (o.classOf (bits u))) = fun (_ : P) (d : o.Openable) => d := by
    funext u d
    unfold pickIn
    exact dif_neg fun hne => by
      rw [hb u] at hne
      exact Finset.not_nonempty_empty hne
  have hG : openGap E g' bits = 0 := by
    unfold openGap
    rw [ho]
    show shortfall (fun u => pickIn (o.classOf (bits u))) = 0
    rw [hp]
    exact shortfall_id
  have hE : openEmpty E g' bits ≤ 1 := by
    unfold openEmpty prCoin
    exact (avg_mono fun u => by split_ifs <;> simp).trans_eq (avg_const 1)
  rw [hG, zero_add]
  exact hE

/-! ## A link's release as a round's -/

variable (e : 𝔽 ≃ (Fin 128 → Bool))

theorem classErr_roundBits_congr {m m' : ℕ} (h : m = m') (N : ℕ) (K : List Bool → ℕ) :
    classErr N (fun c : Fin m → 𝔽 => K (roundBits e c)) = classErr N fun c : Fin m' → 𝔽 => K (roundBits e c) := by
  subst h
  rfl

/-- Two rounds' challenges, appended, are released as the first's bits, then the second's. -/
theorem roundBits_append {m m' : ℕ} (a : Fin m → 𝔽) (b : Fin m' → 𝔽) :
    roundBits e (Fin.append a b) = roundBits e a ++ roundBits e b := by
  have ha : ∀ i : Fin m, Fin.append a b (i.castLE (Nat.le_add_right m m')) = a i := fun i => Fin.append_left a b i
  have hb : ∀ i : Fin m', Fin.append a b (i.natAdd m) = b i := fun i => Fin.append_right a b i
  unfold roundBits
  rw [List.ofFn_add, List.flatten_append]
  simp only [ha, hb]

/-- **A link's two rounds as one round of twice the count**: the first's challenges, then the second's. -/
def linkEquiv (m : ℕ) : (Fin 2 → Fin m → 𝔽) ≃ (Fin (m + m) → 𝔽) where
  toFun p := Fin.append (p 0) (p 1)
  invFun c := ![fun i => c (Fin.castAdd m i), fun i => c (Fin.natAdd m i)]
  left_inv p := funext (Fin.forall_fin_two.2 ⟨funext fun i => Fin.append_left (p 0) (p 1) i,
    funext fun i => Fin.append_right (p 0) (p 1) i⟩)
  right_inv c := by
    funext j
    dsimp only
    refine Fin.addCases (fun i => ?_) (fun i => ?_) j
    · exact Fin.append_left _ _ i
    · exact Fin.append_right _ _ i

/-- **A link's release is a round's**: its bits are those of its two rounds' challenges, appended. -/
theorem linkBits_eq {m : ℕ} (p : Fin 2 → Fin m → 𝔽) : linkBits e p = roundBits e (linkEquiv m p) :=
  (roundBits_append e (p 0) (p 1)).symm

/-! ## A leaf's class error -/

/-- **A leaf's class error at a count**: under the nonce `ν` and the key `key`, the class error of the fiber of the
sample `d₀`'s leaf as stream `s`'s round `r` (the samples with its leaf), partitioned by the bits of `n` challenges its
first `n` coins release. -/
noncomputable def leafErr (ν key : ByteArray) (s r : ℕ) {block : ℕ} (d₀ : TreeDraw block) (n : ℕ) : ℝ≥0∞ :=
  classErr (Finset.univ.filter fun d : TreeDraw block =>
      (TreeDraw.leaf ν key s r d).data = (TreeDraw.leaf ν key s r d₀).data).card
    fun c : Fin n → 𝔽 => (Finset.univ.filter fun d : TreeDraw block =>
      (TreeDraw.leaf ν key s r d).data = (TreeDraw.leaf ν key s r d₀).data ∧ TreeDraw.opened d n = roundBits e c).card

/-- **A leaf's class error at the worst count** of coins its block holds: the prover picks the count after seeing the
leaf. -/
noncomputable def leafErrMax (ν key : ByteArray) (s r : ℕ) {block : ℕ} (d₀ : TreeDraw block) : ℝ≥0∞ :=
  ⨆ n : Fin (block + 1), leafErr e ν key s r d₀ n

/-- **A leaf's class error is within `2·δ₁` over the device's sample** (`avg_iSup_classErr_le_hm96` at
`leaf_eq_treeH`): over the sample `d₀` of stream `s`'s round `r`, where the leaves under `key` hide within `δ₁`, the
class error at the worst count is at most `2·δ₁`. -/
theorem avg_leafErrMax_le (ν key : ByteArray) (s r block : ℕ) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding (treeMy key) (treeC ν) δ₁) :
    avg (fun d₀ : TreeDraw block => leafErrMax e ν key s r d₀) ≤ 2 * δ₁ := by
  have hl : ∀ d : TreeDraw block,
      TreeDraw.leaf ν key s r d = treeH ν key (treeX ν s r d.1 + treeMy key d.2, treeC ν d.2) :=
    fun d => leaf_eq_treeH ν key s r d.1 d.2
  refine Eq.trans_le ?_ (avg_iSup_classErr_le_hm96 (treeMy key) (treeC ν) hT1 (fun p => (treeH ν key p).data)
    (treeX ν s r (block := block)) (R := fun _ : Fin (block + 1) => List Bool)
    (fun n ω => TreeDraw.opened (ω, fun _ => false) n.1) (Q := fun n => Fin n.1 → 𝔽) (fun _ => roundBits e)
    (fun _ => injective_roundBits e) fun n g => avg_opened_roundBits e (Nat.le_of_lt_succ n.2) g)
  refine congrArg avg (funext fun d₀ => ?_)
  unfold leafErrMax
  refine iSup_congr fun n => ?_
  simp only [leafErr, hl] <;> rfl

/-- **The term a leaf costs**: twice the leftover hash lemma's `2^-257`. -/
noncomputable def leafTerm : ℝ≥0∞ := 2 * ENNReal.ofReal (1 / 2 ^ 257)

/-- **A leaf's class error is within `leafTerm` over its tree's key and its sample** (`avg_leafErrMax_le` at
`CROnly.hm96Hiding_gap`, then `avg_hidingGap_tree_le`). -/
theorem avg_leafErrMax_key_le (ν : ByteArray) (s r block : ℕ) :
    avg (fun κ : Fin 2047 → Bool => avg fun d₀ : TreeDraw block => leafErrMax e ν (treeKey κ) s r d₀) ≤
      leafTerm :=
  calc avg (fun κ : Fin 2047 → Bool => avg fun d₀ : TreeDraw block => leafErrMax e ν (treeKey κ) s r d₀)
      ≤ avg (fun κ : Fin 2047 → Bool =>
          ENNReal.ofReal (FlockSoundness.CROnly.hidingGap (treeMy (treeKey κ)) (treeC ν)) * 2) :=
        avg_mono fun κ => (avg_leafErrMax_le e ν (treeKey κ) s r block
          (FlockSoundness.CROnly.hm96Hiding_gap _ _)).trans_eq (mul_comm _ _)
    _ = avg (fun κ : Fin 2047 → Bool =>
          ENNReal.ofReal (FlockSoundness.CROnly.hidingGap (treeMy (treeKey κ)) (treeC ν))) * 2 :=
        avg_mul_const _ _
    _ ≤ ENNReal.ofReal (1 / 2 ^ 257) * 2 := by
        gcongr
        exact avg_hidingGap_tree_le ν
    _ = leafTerm := mul_comm _ _

/-! ## A tree's rounds -/

/-- **A round's class error**: the class error at the worst count of the leaf the tree `T`'s `Hello` committed to as
stream `s`'s round `r`, at the sample it sampled there. -/
noncomputable def roundErr (T : OpenTree) (s : Fin T.spec.streams.size) (r : Fin (2 ^ T.spec.roundsLog)) : ℝ≥0∞ :=
  leafErrMax e T.nonce (treeKey T.sample.1) s r (T.sample.2 s r)

/-- **A tree's unopened rounds' class errors**: each stream's rounds from the next unopened one on. -/
noncomputable def treeErr (T : OpenTree) : ℝ≥0∞ :=
  ∑ s, ∑ r : Fin (2 ^ T.spec.roundsLog), if T.opened s ≤ r.1 then roundErr e T s r else 0

/-- **A fresh tree's rounds cost at most `maxTreeCoins` leaves' terms** over the sample its `Hello` samples: a tree the
device samples holds at most `maxTreeCoins` leaves (`Leaves.treeLeaves_le`), each within `leafTerm`
(`avg_leafErrMax_key_le`). -/
theorem avg_treeErr_hello_le (ν : ByteArray) (spec : Flock.CoinTree.Spec) (hok : TreeSpecOk spec) :
    avg (fun t : TreeSample spec => treeErr e ⟨ν, spec, t, fun _ => 0⟩) ≤ (maxTreeCoins : ℝ≥0∞) * leafTerm := by
  have hr : ∀ (s : Fin spec.streams.size) (r : Fin (2 ^ spec.roundsLog)),
      avg (fun t : TreeSample spec => roundErr e ⟨ν, spec, t, fun _ => 0⟩ s r) ≤ leafTerm := fun s r =>
    calc avg (fun t : TreeSample spec => roundErr e ⟨ν, spec, t, fun _ => 0⟩ s r)
        = avg (fun κ : Fin 2047 → Bool => avg fun w : Fin spec.streams.size → Fin (2 ^ spec.roundsLog) →
            TreeDraw spec.block => leafErrMax e ν (treeKey κ) s r (w s r)) :=
          avg_prod (C := Fin 2047 → Bool)
            (D := Fin spec.streams.size → Fin (2 ^ spec.roundsLog) → TreeDraw spec.block) _
      _ = avg (fun κ : Fin 2047 → Bool => avg fun d₀ : TreeDraw spec.block => leafErrMax e ν (treeKey κ) s r d₀) :=
          congrArg avg (funext fun κ =>
            (avg_eval (T := fun _ => Fin (2 ^ spec.roundsLog) → TreeDraw spec.block) s
              fun w' => leafErrMax e ν (treeKey κ) s r (w' r)).trans
            (avg_eval (T := fun _ => TreeDraw spec.block) r fun d₀ => leafErrMax e ν (treeKey κ) s r d₀))
      _ ≤ leafTerm := avg_leafErrMax_key_le e ν s r spec.block
  have hsum : ∀ t : TreeSample spec, treeErr e ⟨ν, spec, t, fun _ => 0⟩ =
      ∑ s : Fin spec.streams.size, ∑ r : Fin (2 ^ spec.roundsLog), roundErr e ⟨ν, spec, t, fun _ => 0⟩ s r :=
    fun t => Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun r _ => ite_eq_left (Nat.zero_le _)
  calc avg (fun t : TreeSample spec => treeErr e ⟨ν, spec, t, fun _ => 0⟩)
      = ∑ s : Fin spec.streams.size, ∑ r : Fin (2 ^ spec.roundsLog),
          avg (fun t : TreeSample spec => roundErr e ⟨ν, spec, t, fun _ => 0⟩ s r) := by
        simp only [hsum, avg_sum_comm]
    _ ≤ ∑ _s : Fin spec.streams.size, ∑ _r : Fin (2 ^ spec.roundsLog), leafTerm :=
        Finset.sum_le_sum fun s _ => Finset.sum_le_sum fun r _ => hr s r
    _ = (Leaves.treeLeaves spec : ℝ≥0∞) * leafTerm := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Leaves.treeLeaves]
        push_cast
        ring
    _ ≤ (maxTreeCoins : ℝ≥0∞) * leafTerm := by
        gcongr
        exact_mod_cast Leaves.treeLeaves_le hok

/-! ## A node's class error -/

/-- **An opening's class error**: at an opening's node in a play that ends in `E`, the class error of the fiber its
canonical description flips among, partitioned by a round's release at the count of the frame the prover sent; none
at every other node. -/
noncomputable def openErr (E : List Release → Game β) (g' : Game β) : ℝ≥0∞ :=
  (openingOf E g').elim 0 fun o => classErr (Fintype.card o.Openable) fun c : Fin o.n → 𝔽 =>
    (o.classOf (roundBits e c)).card

theorem openErr_none (h : openingOf E g' = none) : openErr e E g' = 0 := by
  unfold openErr
  rw [h]
  rfl

theorem openErr_some {o : OpeningAt E g'} (ho : openingOf E g' = some o) :
    openErr e E g' = classErr (Fintype.card o.Openable) fun c : Fin o.n → 𝔽 => (o.classOf (roundBits e c)).card := by
  unfold openErr
  rw [ho]
  rfl

/-- **An opening's class error at any description** of its node, at a close that records its log: descriptions flip
among one type, release each round alike and share the count (`OpeningAt.mem_classOf_iff`, `OpeningAt.n_eq`). -/
theorem openErr_eq {lg : β → List Release} (hE : RecordsLog lg E) (o₁ : OpeningAt E g') :
    openErr e E g' = classErr (Fintype.card o₁.Openable) fun c : Fin o₁.n → 𝔽 => (o₁.classOf (roundBits e c)).card := by
  obtain ⟨o, ho⟩ := openingOf_some o₁
  rw [openErr_some e ho]
  have hc : Fintype.card o.Openable = Fintype.card o₁.Openable :=
    Fintype.card_congr (Equiv.cast (OpeningAt.openable_eq o o₁))
  have hk : ∀ bits, (o.classOf bits).card = (o₁.classOf bits).card := fun bits =>
    Finset.card_equiv (Equiv.cast (OpeningAt.openable_eq o o₁)) fun d => by
      rw [o.mem_classOf_iff hE o₁ bits d, OpeningAt.classOf, Finset.mem_filter]
      exact ⟨fun h => ⟨Finset.mem_univ _, h⟩, fun h => h.2⟩
  simp only [hc, hk]
  exact classErr_roundBits_congr e (OpeningAt.n_eq hE o o₁) _ fun bits => (o₁.classOf bits).card

/-- **An opening's class error is within its round's**: at a close that records its log, the node's class error is
the class error of the leaf of the round it opens at the frame's count, a count its block holds. -/
theorem openErr_le_roundErr {lg : β → List Release} (hE : RecordsLog lg E) (o₁ : OpeningAt E g')
    (hn : o₁.n ≤ o₁.T.spec.block) : openErr e E g' ≤ roundErr e o₁.T o₁.p.1 o₁.p.2 := by
  rw [openErr_eq e hE o₁]
  unfold roundErr leafErrMax
  refine le_trans (le_of_eq ?_) (le_iSup _ (⟨o₁.n, Nat.lt_succ_of_le hn⟩ : Fin (o₁.T.spec.block + 1)))
  simp only [leafErr]
  rw [card_subtype_eq]
  exact congrArg (classErr _) (funext fun c => card_filter_subtype fun d => TreeDraw.opened d o₁.n = roundBits e c)

/-- **A frame past the block opens no round's class**: where the frame asks for more coins than a block holds, every
class is empty and the class error is 1. -/
theorem openErr_eq_one {o : OpeningAt E g'} (ho : openingOf E g' = some o) (hb : o.T.spec.block < o.n) :
    openErr e E g' = 1 := by
  rw [openErr_some e ho]
  have h0 : (fun c : Fin o.n → 𝔽 => (o.classOf (roundBits e c)).card) = fun _ => 0 := funext fun c =>
    Finset.card_eq_zero.2 ((o.classOf_eq_empty_iff _).2 fun d =>
      opened_ne_of_length hb d.1 (length_roundBits_enc e c))
  rw [h0]
  exact classErr_zero Fintype.card_ne_zero

/-- **A round's gap and empty class within the node's class error**, at the frame's count `m`
(`openGap_add_openEmpty_le_round`; `openErr_eq_one` past the block). -/
theorem openGap_add_openEmpty_le_openErr {m : ℕ} (hn : ∀ o, openingOf E g' = some o → o.n = m) :
    openGap E g' (roundBits e : (Fin m → 𝔽) → List Bool) +
      openEmpty E g' (roundBits e : (Fin m → 𝔽) → List Bool) ≤ openErr e E g' := by
  cases ho : openingOf E g' with
  | none =>
    rw [openGap_none ho, openEmpty_none ho, add_zero]
    exact zero_le
  | some o =>
    have hm := hn o ho
    by_cases hb : m ≤ o.T.spec.block
    · refine (openGap_add_openEmpty_le_round e o ho hm hb).trans (le_of_eq ?_)
      rw [openErr_some e ho]
      exact classErr_roundBits_congr e hm.symm _ fun bits => (o.classOf bits).card
    · rw [openErr_eq_one e ho (by omega)]
      exact openGap_add_openEmpty_le_one ho _ fun c => (o.classOf_eq_empty_iff _).2 fun d =>
        opened_ne_of_length (by omega) d.1 (by rw [length_roundBits_enc, hm])

/-- **A link's gap and empty class within the node's class error**, at the frame's count `m + m`
(`openGap_add_openEmpty_le` at `injective_linkBits`, read as a round's release through `linkEquiv`). -/
theorem openGap_add_openEmpty_le_openErr_link {m : ℕ} (hn : ∀ o, openingOf E g' = some o → o.n = m + m) :
    openGap E g' (linkBits e : (Fin 2 → Fin m → 𝔽) → List Bool) +
      openEmpty E g' (linkBits e : (Fin 2 → Fin m → 𝔽) → List Bool) ≤ openErr e E g' := by
  cases ho : openingOf E g' with
  | none =>
    rw [openGap_none ho, openEmpty_none ho, add_zero]
    exact zero_le
  | some o =>
    have hm := hn o ho
    by_cases hb : o.n ≤ o.T.spec.block
    · have hb' : m + m ≤ o.T.spec.block := hm ▸ hb
      have hcov : ∀ d : o.Openable, ∃ u, TreeDraw.opened d.1 o.n = linkBits e u := fun d => by
        refine ⟨(linkEquiv m).symm fun i => e.symm (Fin.take (m + m) hb' d.1.1 i), ?_⟩
        rw [linkBits_eq, Equiv.apply_symm_apply, hm]
        exact opened_eq_roundBits e hb' d.1
      refine (openGap_add_openEmpty_le o ho (linkBits e) (injective_linkBits e) hcov).trans (le_of_eq ?_)
      rw [openErr_some e ho]
      simp only [linkBits_eq e]
      exact (classErr_equiv (linkEquiv m) _ fun c => (o.classOf (roundBits e c)).card).trans
        (classErr_roundBits_congr e hm.symm _ fun bits => (o.classOf bits).card)
    · rw [openErr_eq_one e ho (by omega)]
      refine openGap_add_openEmpty_le_one ho _ fun u => (o.classOf_eq_empty_iff _).2 fun d =>
        opened_ne_of_length (by omega) d.1 ?_
      rw [linkBits, List.length_append, length_roundBits_enc, length_roundBits_enc, hm]
      ring

/-! ## The potential -/

/-- **The open trees' unopened rounds' class errors.** -/
noncomputable def treesErr (ts : List OpenTree) : ℝ≥0∞ := (ts.map (treeErr e)).sum

/-- **The potential** with the coin trees `ts` open: their unopened rounds' class errors, and `maxTreeCoins` leaves'
terms for each coin-tree line the device's log has left. -/
noncomputable def treePotential (ts : List OpenTree) : ℝ≥0∞ :=
  treesErr e ts + ((maxTreeLines - treeLines ts : ℕ) : ℝ≥0∞) * ((maxTreeCoins : ℝ≥0∞) * leafTerm)

/-- **Opening a stream's next round takes its class error out of the tree's.** -/
theorem treeErr_bump (T : OpenTree) (s' : Fin T.spec.streams.size) (h : T.opened s' < 2 ^ T.spec.roundsLog) :
    treeErr e T = treeErr e (T.bump s') + roundErr e T s' ⟨T.opened s', h⟩ := by
  have hs : ∀ s : Fin T.spec.streams.size,
      (∑ r : Fin (2 ^ T.spec.roundsLog), if T.opened s ≤ r.1 then roundErr e T s r else 0) =
        (∑ r : Fin (2 ^ T.spec.roundsLog), if (if s = s' then T.opened s' + 1 else T.opened s) ≤ r.1 then
          roundErr e T s r else 0) + if s = s' then roundErr e T s ⟨T.opened s', h⟩ else 0 := fun s => by
    by_cases hss : s = s'
    · subst hss
      rw [if_pos rfl, if_pos rfl]
      exact sum_ite_le _ h _
    · rw [if_neg hss, if_neg hss, add_zero]
  show _ = (∑ s : Fin T.spec.streams.size, ∑ r : Fin (2 ^ T.spec.roundsLog),
      if (if s = s' then T.opened s' + 1 else T.opened s) ≤ r.1 then roundErr e T s r else 0) + _
  rw [treeErr, Fintype.sum_congr _ _ hs, Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ s',
    if_pos (Finset.mem_univ _)]

theorem treesErr_set {T T' : OpenTree} {x : ℝ≥0∞} (hT : treeErr e T = treeErr e T' + x) (ts : List OpenTree) :
    ∀ {i : ℕ}, ts[i]? = some T → treesErr e ts = treesErr e (ts.set i T') + x := by
  induction ts with
  | nil =>
    intro i h
    simp at h
  | cons T₀ ts ih =>
    intro i h
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h
      simp only [treesErr, List.set_cons_zero, List.map_cons, List.sum_cons]
      rw [hT]
      ring
    | succ i =>
      simp only [List.getElem?_cons_succ] at h
      have ih' := ih h
      simp only [treesErr, List.set_cons_succ, List.map_cons, List.sum_cons] at ih' ⊢
      rw [ih']
      ring

theorem treesErr_append (ts : List OpenTree) (T : OpenTree) :
    treesErr e (ts ++ [T]) = treesErr e ts + treeErr e T := by
  simp [treesErr]

/-- **A `Hello` keeps the potential on average**: the fresh tree's rounds cost at most the line it writes. -/
theorem avg_treePotential_hello_le (ts : List OpenTree) (ν : ByteArray) (spec : Flock.CoinTree.Spec)
    (hok : TreeSpecOk spec) (hL : treeLines ts < maxTreeLines) :
    avg (fun t : TreeSample spec => treePotential e (ts ++ [⟨ν, spec, t, fun _ => 0⟩])) ≤ treePotential e ts := by
  have hsub : maxTreeLines - treeLines ts = maxTreeLines - (treeLines ts + 1) + 1 := by omega
  have h : ∀ t : TreeSample spec, treePotential e (ts ++ [⟨ν, spec, t, fun _ => 0⟩]) =
      treeErr e ⟨ν, spec, t, fun _ => 0⟩ +
        (treesErr e ts + ((maxTreeLines - (treeLines ts + 1) : ℕ) : ℝ≥0∞) * ((maxTreeCoins : ℝ≥0∞) * leafTerm)) :=
    fun t => by
      rw [treePotential, treesErr_append, Leaves.treeLines_append_hello]
      ring
  calc avg (fun t : TreeSample spec => treePotential e (ts ++ [⟨ν, spec, t, fun _ => 0⟩]))
      = avg (fun t : TreeSample spec => treeErr e ⟨ν, spec, t, fun _ => 0⟩) +
          (treesErr e ts + ((maxTreeLines - (treeLines ts + 1) : ℕ) : ℝ≥0∞) *
            ((maxTreeCoins : ℝ≥0∞) * leafTerm)) := by
        simp only [h]
        rw [avg_add, avg_const]
    _ ≤ (maxTreeCoins : ℝ≥0∞) * leafTerm +
          (treesErr e ts + ((maxTreeLines - (treeLines ts + 1) : ℕ) : ℝ≥0∞) *
            ((maxTreeCoins : ℝ≥0∞) * leafTerm)) :=
        add_le_add (avg_treeErr_hello_le e ν spec hok) le_rfl
    _ = treePotential e ts := by
        rw [treePotential, hsub, Nat.cast_add, Nat.cast_one]
        ring

/-- **An opening pays its round's class error out of the potential**: opening stream `s'`'s next round of the `i`-th
tree takes the round's class error out of the tree's and writes a line. -/
theorem roundErr_add_treePotential_le {ts : List OpenTree} {i : ℕ} {T : OpenTree} (hT : ts[i]? = some T)
    (hL : treeLines ts < maxTreeLines) (s' : Fin T.spec.streams.size) (h : T.opened s' < 2 ^ T.spec.roundsLog) :
    roundErr e T s' ⟨T.opened s', h⟩ + treePotential e (ts.set i (T.bump s')) ≤ treePotential e ts := by
  have hs := treesErr_set e (treeErr_bump e T s' h) ts hT
  have hsub : maxTreeLines - treeLines ts = maxTreeLines - (treeLines ts + 1) + 1 := by omega
  rw [treePotential, treePotential, Leaves.treeLines_set_bump hT, hs, hsub, Nat.cast_add, Nat.cast_one]
  calc _ ≤ (roundErr e T s' ⟨T.opened s', h⟩ + (treesErr e (ts.set i (T.bump s')) +
        ((maxTreeLines - (treeLines ts + 1) : ℕ) : ℝ≥0∞) * ((maxTreeCoins : ℝ≥0∞) * leafTerm))) +
        (maxTreeCoins : ℝ≥0∞) * leafTerm := le_self_add
    _ = _ := by ring

/-- **From no tree open, the potential is the hiding term**: `maxTreeLines` lines of `maxTreeCoins` leaves, twice at
`2^-257` a leaf. -/
theorem treePotential_nil (A : Live) : treePotential e [] = A.hidingTerm := by
  have h : ENNReal.ofReal (1 / 2 ^ 257) = ((2 : ℝ≥0∞) ^ 257)⁻¹ := by
    rw [one_div, ENNReal.ofReal_inv_of_pos (by positivity), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_ofNat]
  rw [treePotential, treesErr, leafTerm, h, Live.hidingTerm, maxLeaves]
  simp only [List.map_nil, List.sum_nil, zero_add, treeLines, Nat.sub_zero, Nat.cast_mul, div_eq_mul_inv]
  ring

end Verity.Sampled
