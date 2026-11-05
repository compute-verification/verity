module

public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Properties.Verity.Sampled.CoinMap

/-!
# An opening's empty class at one hm96 leaf

An opening of a coin tree's leaf releases the leaf's first coins, which the verifier compares with live coins drawn
after the leaf was committed. The opening's class at those coins (`OpeningAt.classOf`) holds each draw with the leaf's
hash that releases them, and `openEmpty` is the probability that it is empty. In the abstract: a leaf
`H(x(ω) + M·y, c(y))` of uniform coins `ω` and salt `y`, a release `P ω`, and live coins `u` whose bits `bits u` have
the release's law. `fiberMisses_le` bounds by `2·δ₁` the probability that no coins and salt with the leaf's hash
release `bits u`. That is T1's bound twice. T1 makes the leaf ideal, so the event no longer reads the coins, and
`bits u` may be swapped for the release `P ω`. T1 on the complement then makes the leaf real again, where the leaf's
own coins and salt release `P ω`.

* **The empty class below the block** (`OpeningAt.oneMore`, `OpeningAt.classOf_eq_empty`, `emptyAt_of_lt_block`,
  `openEmpty_eq_one`): `classOf` holds the draws released as the bits at every description of the node, and a node's
  descriptions are not one. A description that opens fewer coins than a block has a twin that opens one coin more,
  whose continuation is the releases' at no frame (`relK` at `0`) reading only the release's first coins. The two
  release bits of two lengths, so the class is empty at every release, and the empty class has probability one.
-/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

/-- No coins `ω` and salt `y` whose leaf `H(x(ω) + M·y, c(y))` is `ℓ` release `r`: at a leaf whose hash is `ℓ`, the
class of an opening that releases `P ω` is empty at `r`. -/
def FiberMisses {Ω Salt B C D R : Type} [Add B] (My : Salt → B) (c : Salt → C) (H : B × C → D) (x : Ω → B) (P : Ω → R)
    (ℓ : D) (r : R) : Prop :=
  ∀ ω y, H (x ω + My y, c y) = ℓ → P ω ≠ r

/-- **An opening's class is empty with probability at most `2·δ₁`** at one hm96 leaf. Over the leaf's uniform coins `ω`
and salt `y`, and live coins `u` drawn independently whose bits `bits u` have the law of the release `P ω` (`hlaw`),
this is the probability that no coins and salt with the leaf's hash release `bits u`. -/
theorem fiberMisses_le {Ω Q Salt B C D R : Type} [Fintype Ω] [Nonempty Ω] [Fintype Q] [Nonempty Q] [Fintype Salt]
    [Nonempty Salt] [AddCommGroup B] [Fintype B] (My : Salt → B) (c : Salt → C) {δ₁ : ℝ≥0∞}
    (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) (x : Ω → B) (P : Ω → R) (bits : Q → R)
    (hlaw : ∀ g : R → ℝ≥0∞, avg (fun ω => g (P ω)) = avg fun u => g (bits u)) :
    prCoin (fun p : (Ω × Q) × Salt => FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2))
      ≤ 2 * δ₁ := by
  have : Nonempty B := ⟨0⟩
  have h1 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1)
    (fun w ℓ => FiberMisses My c H x P ℓ (bits w.2))
  have h2 := FlockSoundness.ZK.ideal_leaf_swap My c hT1 H (fun w : Ω × Q => x w.1)
    (fun w ℓ => ¬ FiberMisses My c H x P ℓ (P w.1))
  have hno : ∀ p : (Ω × Q) × Salt, ¬ FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (P p.1.1) :=
    fun p h => h p.1.1 p.2 rfl rfl
  have hown : prCoin (fun p : (Ω × Q) × Salt => ¬ FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (P p.1.1))
      = 1 := by
    unfold prCoin
    calc avg _ = avg (fun _ : (Ω × Q) × Salt => (1 : ℝ≥0∞)) := by
          congr 1
          funext p
          simp [hno p]
      _ = 1 := FlockSoundness.Game.avg_const _
  have hl : avg (fun w : Ω × Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (bits w.2))
      = avg fun q : Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (bits q) :=
    (FlockSoundness.ZK.avg_prod_split _).trans (FlockSoundness.Game.avg_const
      (avg fun q : Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (bits q)))
  have hr : avg (fun w : Ω × Q => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (P w.1))
      = avg fun ω : Ω => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (P ω) :=
    (FlockSoundness.ZK.avg_prod_split _).trans (congrArg avg (funext fun ω => FlockSoundness.Game.avg_const
      (prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) (P ω))))
  have hswap : prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (bits p.1.2))
      = prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1)) := by
    rw [FlockSoundness.ZK.prCoin_prod_split
        (fun (w : Ω × Q) (u : B × Salt) => FiberMisses My c H x P (H (u.1, c u.2)) (bits w.2)),
      FlockSoundness.ZK.prCoin_prod_split
        (fun (w : Ω × Q) (u : B × Salt) => FiberMisses My c H x P (H (u.1, c u.2)) (P w.1))]
    exact hl.trans ((hlaw fun r => prCoin fun u : B × Salt => FiberMisses My c H x P (H (u.1, c u.2)) r).symm.trans
      hr.symm)
  have hideal : prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
      ≤ δ₁ := by
    have a := FlockSoundness.ZK.prCoin_add_not
      (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
    have hfin := FlockSoundness.ZK.prCoin_ne_top
      (fun p : (Ω × Q) × (B × Salt) => ¬ FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1))
    refine (ENNReal.add_le_add_iff_right hfin).1 ?_
    rw [a, add_comm δ₁]
    exact hown.symm.le.trans h2
  calc prCoin (fun p : (Ω × Q) × Salt => FiberMisses My c H x P (H (x p.1.1 + My p.2, c p.2)) (bits p.1.2))
      ≤ prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (bits p.1.2)) + δ₁ := h1
    _ = prCoin (fun p : (Ω × Q) × (B × Salt) => FiberMisses My c H x P (H (p.2.1, c p.2.2)) (P p.1.1)) + δ₁ := by
        rw [hswap]
    _ ≤ δ₁ + δ₁ := add_le_add hideal le_rfl
    _ = 2 * δ₁ := (two_mul δ₁).symm

/-! ## The empty class below the block -/

/-- Lists of `w` bits each: the first `n + 1` lists' bits, cut to `w · n`, are the first `n` lists' bits. -/
theorem flatten_take_succ {w : ℕ} : ∀ (L : List (List Bool)) (n : ℕ), (∀ x ∈ L, x.length = w) →
    ((L.take (n + 1)).flatten).take (w * n) = (L.take n).flatten
  | [], _, _ => by simp
  | x :: L, 0, _ => by simp
  | x :: L, n + 1, h => by
    have hx : x.length = w := h x List.mem_cons_self
    rw [List.take_succ_cons, List.take_succ_cons, List.flatten_cons, List.flatten_cons, List.take_append,
      List.take_of_length_le (by rw [hx, Nat.mul_add_one]; exact Nat.le_add_left w (w * n)), hx,
      show w * (n + 1) - w = w * n by rw [Nat.mul_add_one, Nat.add_sub_cancel],
      flatten_take_succ L n fun y hy => h y (List.mem_cons_of_mem _ hy)]

/-- Lists of `w` bits each: the first `n` lists hold `w` bits for each list taken. -/
theorem flatten_take_length {w : ℕ} : ∀ (L : List (List Bool)) (n : ℕ), (∀ x ∈ L, x.length = w) →
    ((L.take n).flatten).length = w * min n L.length
  | [], _, _ => by simp
  | x :: L, 0, _ => by simp
  | x :: L, n + 1, h => by
    rw [List.take_succ_cons, List.flatten_cons, List.length_append, h x List.mem_cons_self,
      flatten_take_length L n fun y hy => h y (List.mem_cons_of_mem _ hy), List.length_cons, min_add_add_right]
    ring

/-- A draw's coins are 128 bits each. -/
theorem ofFn_lengths {block : ℕ} (d : TreeDraw block) :
    ∀ x ∈ List.ofFn (fun i => List.ofFn (d.1 i)), x.length = 128 := by
  intro x hx
  obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
  exact List.length_ofFn

/-- A draw's first `n + 1` coins' bits, cut to `128 · n`, are its first `n` coins' bits. -/
theorem opened_take_succ {block : ℕ} (d : TreeDraw block) (n : ℕ) :
    (TreeDraw.opened d (n + 1)).take (128 * n) = TreeDraw.opened d n :=
  flatten_take_succ _ n (ofFn_lengths d)

/-- A draw's first `n` coins of a block are `128 · n` bits. -/
theorem opened_length_eq {block : ℕ} (d : TreeDraw block) {n : ℕ} (h : n ≤ block) :
    (TreeDraw.opened d n).length = 128 * n := by
  unfold TreeDraw.opened
  rw [flatten_take_length _ n (ofFn_lengths d), List.length_ofFn, min_eq_left h]

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

/-- **The node's description that opens one coin more**: `o`'s node, as the releases' continuation at no frame (`relK`
at `0`) that reads the release's first `o.n` coins as `o`'s continuation does. It describes the node wherever `o`
does, since a draw's first `o.n + 1` coins hold its first `o.n` (`opened_take_succ`). -/
noncomputable def OpeningAt.oneMore {β : Type} {g' : Game β} (o : OpeningAt g') : OpeningAt g' where
  T := o.T
  p := o.p
  i := o.i
  s := o.s
  n := o.n + 1
  b := o.b
  k := 0
  fs := o.fs
  ts := o.ts
  f log := relK o.k o.fs (o.ts.set o.i (o.T.bump o.p.1)) fun log' =>
    o.f (⟨.opening o.i o.s o.n, o.b, (log.headD ⟨.opening o.i o.s o.n, o.b, []⟩).coins.take (128 * o.n)⟩ :: log')
  iF := o.iF
  iN := o.iN
  eq := o.eq.trans (congrArg (@Game.coin β _ o.iF o.iN) (funext fun d => by
    show _ = relK o.k o.fs (o.ts.set o.i (o.T.bump o.p.1)) fun log' =>
      o.f (⟨.opening o.i o.s o.n, o.b, (TreeDraw.opened d.1 (o.n + 1)).take (128 * o.n)⟩ :: log')
    rw [opened_take_succ]))

open Classical in
/-- **An opening's class is empty below the block**: where a description `o'` of the node opens fewer coins than a
block, no round is released as `bits` both at `o'` and at the description that opens one coin more
(`OpeningAt.oneMore`), whose releases are 128 bits longer. -/
theorem OpeningAt.classOf_eq_empty {β : Type} {g' : Game β} (o o' : OpeningAt g') (h : o'.n < o'.T.spec.block)
    (bits : List Bool) : o.classOf bits = ∅ := by
  refine Finset.eq_empty_of_forall_notMem fun d hd => ?_
  have h1 := congrArg List.length ((Finset.mem_filter.1 hd).2 o')
  have h2 := congrArg List.length ((Finset.mem_filter.1 hd).2 o'.oneMore)
  have e1 : 128 * o'.n = bits.length := (opened_length_eq _ h.le).symm.trans h1
  have e2 : 128 * (o'.n + 1) = bits.length :=
    (opened_length_eq (n := o'.n + 1) _ (Nat.succ_le_of_lt h)).symm.trans h2
  omega

/-- **Below the block, every release is in the empty class**: where a description of an opening's node opens fewer
coins than a block, `EmptyAt` holds at every release (`OpeningAt.classOf_eq_empty`). -/
theorem emptyAt_of_lt_block {β : Type} {g' : Game β} (o' : OpeningAt g') (h : o'.n < o'.T.spec.block)
    (bits : List Bool) : EmptyAt g' bits := by
  have hn : Nonempty (OpeningAt g') := ⟨o'⟩
  refine ⟨Classical.choice hn, by unfold openingOf; rw [dite_eq_left hn], ?_⟩
  rw [OpeningAt.classOf_eq_empty _ o' h bits]
  exact Finset.not_nonempty_empty

/-- **Below the block, the opening's empty class has probability one** (`emptyAt_of_lt_block`). -/
theorem openEmpty_eq_one {β P : Type} [Fintype P] [Nonempty P] {g' : Game β} (o' : OpeningAt g')
    (h : o'.n < o'.T.spec.block) (bits : P → List Bool) : openEmpty g' bits = 1 := by
  unfold openEmpty prCoin
  calc avg _ = avg (fun _ : P => (1 : ℝ≥0∞)) := by
        congr 1
        funext c
        simp [emptyAt_of_lt_block o' h (bits c)]
    _ = 1 := FlockSoundness.Game.avg_const _

end Verity.Sampled
