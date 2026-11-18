module

public import Proofs.Core.LevelsCircuit
public import Proofs.Core.Game.InterleaveBridge

/-!
# Committing late does not help, at any number of levels

The coins drawn level by level (`Levels.Coins`): at level `i`, one draw of `D_i^X` for every block `X`, and `Q_k`
(`Levels.drawTo`) the blocks drawn through level `k`. A commitment made after `Q_k` is drawn is a function of the coins
of the levels below `k` (`Levels.ReadsBelow`), of any of them, the other blocks' included.

`Levels.LateAtLevels`: at the levels of a circuit's partitions (`Levels.ofPartitions`), the prover commits a trace
`τ k` after `Q_k`, by any rule, each agreeing with the one before on that level's committed gates. Every sample then misses the
last trace's incorrect last-level blocks with probability at most `missProb` of one trace's, fixed in advance and
agreeing with the first on the first level's committed gates.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Coins drawn level by level**: at level `i`, a draw of `D_i^X` for every block `X`. -/
def Coins (L : Levels G) : Type :=
  (i : Fin L.levels.length) → (X : Finset G) → ((L.levels.get i).design X).Ω

instance (L : Levels G) : Fintype L.Coins := by unfold Coins; infer_instance

instance (L : Levels G) : Nonempty L.Coins := by unfold Coins; infer_instance

/-- `Q_k`: the blocks drawn through level `k`, `Q₀ = {univ}`. -/
def drawTo (L : Levels G) (ω : L.Coins) : ℕ → Finset (Finset G)
  | 0 => {Finset.univ}
  | k + 1 => if h : k < L.levels.length then
      (L.drawTo ω k).biUnion fun X => ((L.levels.get ⟨k, h⟩).design X).draw (ω ⟨k, h⟩ X)
    else L.drawTo ω k

/-- `f` reads only the coins of the levels below `k`. -/
def ReadsBelow {α : Type} (L : Levels G) (k : ℕ) (f : L.Coins → α) : Prop :=
  ∀ ω ω' : L.Coins, (∀ i : Fin L.levels.length, i.val < k → ω i = ω' i) → f ω = f ω'

/-- A product of per-coordinate functions over the coordinates in `S`, averaged over independent coordinates, is the
product of their averages. -/
theorem _root_.Verity.Game.avg_pi_prod_finset {ι : Type} [Fintype ι] [DecidableEq ι] {Ω : ι → Type}
    [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (S : Finset ι) (f : ∀ i, Ω i → ℝ≥0∞) :
    avg (fun ω : ∀ i, Ω i => ∏ i ∈ S, f i (ω i)) = ∏ i ∈ S, avg (f i) := by
  classical
  have key : (fun ω : ∀ i, Ω i => ∏ i ∈ S, f i (ω i)) =
      fun ω => ∏ i, (if i ∈ S then f i (ω i) else 1) := by
    funext ω
    rw [← Finset.prod_filter, Finset.filter_mem_eq_inter, Finset.univ_inter]
  rw [key, avg_pi_prod (fun i c => if i ∈ S then f i c else 1)]
  rw [← Finset.prod_filter_mul_prod_filter_not Finset.univ (· ∈ S), Finset.filter_mem_eq_inter,
    Finset.univ_inter]
  rw [Finset.prod_eq_one (s := Finset.univ.filter (· ∉ S)) fun i hi => by
    rw [Finset.mem_filter] at hi
    simp only [hi.2, ite_false]
    exact avg_const 1, mul_one]
  exact Finset.prod_congr rfl fun i hi => by simp only [hi, ite_true]

/-- The blocks level `k`'s designs draw inside: `P` at the first, then each level's blocks. -/
def _root_.Verity.Level.above (P : Finset (Finset G)) : List (Level G) → ℕ → Finset (Finset G)
  | _, 0 => P
  | [], _ + 1 => P
  | l :: ls, k + 1 => Level.above l.blocks.parts ls k

theorem _root_.Verity.Level.above_succ {P : Finset (Finset G)} :
    ∀ {ls : List (Level G)} {k : ℕ} (h : k < ls.length),
      Level.above P ls (k + 1) = (ls.get ⟨k, h⟩).blocks.parts
  | [], _, h => absurd h (Nat.not_lt_zero _)
  | l :: ls, 0, _ => by cases ls <;> rfl
  | l :: ls, k + 1, h => by
    rw [show Level.above P (l :: ls) (k + 1 + 1) = Level.above l.blocks.parts ls (k + 1) from rfl,
      Level.above_succ (P := l.blocks.parts) (Nat.lt_of_succ_lt_succ h)]
    rfl

theorem _root_.Verity.Level.inside_get :
    ∀ {P : Finset (Finset G)} {ls : List (Level G)}, Level.InsideBelow P ls →
      ∀ (k : ℕ) (h : k < ls.length), ∀ X ∈ Level.above P ls k, ∀ ω, ∀ b ∈ ((ls.get ⟨k, h⟩).design X).draw ω,
        b ∈ (ls.get ⟨k, h⟩).blocks.parts ∧ b ⊆ X
  | _, [], _, _, h => absurd h (Nat.not_lt_zero _)
  | _, _ :: _, hi, 0, _ => fun X hX ω b hb => hi.1 X hX ω b hb
  | _, _ :: _, hi, k + 1, h => fun X hX ω b hb =>
    Level.inside_get hi.2 k (Nat.lt_of_succ_lt_succ h) X hX ω b hb

/-- Blocks of one level meet only when equal. -/
def _root_.Verity.Level.Separate (P : Finset (Finset G)) : Prop :=
  ∀ X ∈ P, ∀ Y ∈ P, (X ∩ Y).Nonempty → X = Y

theorem _root_.Verity.Level.separate_parts (l : Level G) : Level.Separate l.blocks.parts := by
  intro X hX Y hY ⟨g, hg⟩
  exact l.blocks.eq_of_mem_parts hX hY (Finset.mem_inter.1 hg).1 (Finset.mem_inter.1 hg).2

theorem _root_.Verity.Level.separate_above {P : Finset (Finset G)} (hP : Level.Separate P) :
    ∀ (ls : List (Level G)) (k : ℕ), Level.Separate (Level.above P ls k)
  | ls, 0 => by cases ls <;> exact hP
  | [], _ + 1 => hP
  | l :: ls, k + 1 => Level.separate_above (Level.separate_parts l) ls k

theorem separate_univ : Level.Separate ({Finset.univ} : Finset (Finset G)) := by
  intro X hX Y hY _
  rw [Finset.mem_singleton.1 hX, Finset.mem_singleton.1 hY]

theorem drawTo_succ (L : Levels G) (ω : L.Coins) {k : ℕ} (h : k < L.levels.length) :
    L.drawTo ω (k + 1) =
      (L.drawTo ω k).biUnion fun X => ((L.levels.get ⟨k, h⟩).design X).draw (ω ⟨k, h⟩ X) := by
  rw [drawTo]
  split
  · rfl
  · contradiction

/-- `Q_k`'s blocks are blocks of the level above level `k`. -/
theorem drawTo_above (L : Levels G) (ω : L.Coins) :
    ∀ k, k ≤ L.levels.length → L.drawTo ω k ⊆ Level.above {Finset.univ} L.levels k
  | 0, _ => fun X hX => by cases hl : L.levels <;> simpa [drawTo, Level.above, hl] using hX
  | k + 1, hk => by
    have h : k < L.levels.length := hk
    rw [L.drawTo_succ ω h, Level.above_succ h]
    intro X hX
    obtain ⟨Y, hY, hX⟩ := Finset.mem_biUnion.1 hX
    exact (Level.inside_get L.inside k h Y (L.drawTo_above ω k h.le hY) _ X hX).1

/-- `Q_k` reads only the coins of the levels below `k`. -/
theorem drawTo_congr (L : Levels G) {ω ω' : L.Coins} :
    ∀ k, (∀ i : Fin L.levels.length, i.val < k → ω i = ω' i) → L.drawTo ω k = L.drawTo ω' k
  | 0, _ => rfl
  | k + 1, h => by
    by_cases hk : k < L.levels.length
    · rw [L.drawTo_succ ω hk, L.drawTo_succ ω' hk, L.drawTo_congr k fun i hi => h i (Nat.lt_succ_of_lt hi),
        h ⟨k, hk⟩ (Nat.lt_succ_self k)]
    · simp only [drawTo, hk, dite_false]
      exact L.drawTo_congr k fun i hi => h i (Nat.lt_succ_of_lt hi)

/-- **One level's coins, split off**: averaged over the coins, a product over `Q_{k+1}` of a reading of the coins below
`k` is the product over `Q_k` of each block's average over its own draw at level `k`. -/
theorem avg_prod_drawTo_succ (L : Levels G) {k : ℕ} (hk : k < L.levels.length)
    (F : L.Coins → Finset G → ℝ≥0∞) (hF : L.ReadsBelow k F) :
    avg (fun ω => ∏ X ∈ L.drawTo ω (k + 1), F ω X) =
      avg (fun ω => ∏ Y ∈ L.drawTo ω k,
        avg fun c : ((L.levels.get ⟨k, hk⟩).design Y).Ω =>
          ∏ X ∈ ((L.levels.get ⟨k, hk⟩).design Y).draw c, F ω X) := by
  classical
  let i : Fin L.levels.length := ⟨k, hk⟩
  have hsep := Level.separate_above separate_univ L.levels k
  refine Eq.trans (avg_update_pi (T := fun j : Fin L.levels.length => (X : Finset G) → ((L.levels.get j).design X).Ω)
    i (fun ω => ∏ X ∈ L.drawTo ω (k + 1), F ω X)).symm (congrArg avg (funext fun ω => ?_))
  have hbelow : ∀ c, ∀ j : Fin L.levels.length, j.val < k → Function.update ω i c j = ω j := fun c j hj =>
    Function.update_of_ne (fun h => by rw [h] at hj; exact lt_irrefl _ hj) _ _
  have hstep : ∀ c : (X : Finset G) → ((L.levels.get i).design X).Ω,
      ∏ X ∈ L.drawTo (Function.update ω i c) (k + 1), F (Function.update ω i c) X =
        ∏ Y ∈ L.drawTo ω k, ∏ X ∈ ((L.levels.get i).design Y).draw (c Y), F ω X := by
    intro c
    erw [L.drawTo_succ (Function.update ω i c) hk, L.drawTo_congr k (hbelow c), hF _ _ (hbelow c)]
    simp only [show Function.update ω i c ⟨k, hk⟩ = c from Function.update_self _ _ _]
    refine Finset.prod_biUnion fun Y hY Y' hY' hne => Finset.disjoint_left.2 fun X hX hX' => hne ?_
    have hab := L.drawTo_above ω k hk.le
    obtain ⟨hXp, hXY⟩ := Level.inside_get L.inside k hk Y (hab hY) _ X hX
    obtain ⟨-, hXY'⟩ := Level.inside_get L.inside k hk Y' (hab hY') _ X hX'
    obtain ⟨g, hg⟩ := (L.levels.get i).blocks.nonempty_of_mem_parts hXp
    exact hsep Y (hab hY) Y' (hab hY') ⟨g, Finset.mem_inter.2 ⟨hXY hg, hXY' hg⟩⟩
  simp only [hstep]
  exact avg_pi_prod_finset (L.drawTo ω k) fun Y c => ∏ X ∈ ((L.levels.get i).design Y).draw c, F ω X

end Levels

namespace Level

variable {G : Type} [Fintype G] [DecidableEq G]

theorem above_zero (P : Finset (Finset G)) (ls : List (Level G)) : Level.above P ls 0 = P := by
  cases ls <;> rfl

/-- A disjointness, averaged, is the product over one side of each element's miss below no level. -/
theorem prCoin_disjoint_eq {Ω : Type} [Fintype Ω] (S T : Ω → Finset (Finset G)) :
    prCoin (fun ω => Disjoint (S ω) (T ω)) = avg fun ω => ∏ X ∈ S ω, (sampleBelow [] X).missProb (T ω) := by
  unfold prCoin
  refine congrArg avg (funext fun ω => ?_)
  have leaf : ∀ X, (sampleBelow [] X).missProb (T ω) = if X ∈ T ω then 0 else 1 := fun X => by
    show prCoin (fun _ : Unit => Disjoint {X} (T ω)) = _
    unfold prCoin
    rw [avg_const]
    by_cases h : X ∈ T ω
    · rw [if_neg (by rwa [Finset.disjoint_singleton_left, not_not]), if_pos h]
    · rw [if_pos (Finset.disjoint_singleton_left.2 h), if_neg h]
  rw [Finset.prod_congr rfl fun X _ => leaf X]
  by_cases h : Disjoint (S ω) (T ω)
  · rw [if_pos h]
    exact (Finset.prod_eq_one fun X hX => if_neg (Finset.disjoint_left.1 h hX)).symm
  · obtain ⟨X, hX, hXT⟩ := Finset.not_disjoint_iff.1 h
    rw [if_neg h]
    exact (Finset.prod_eq_zero (f := fun X => if X ∈ T ω then (0 : ℝ≥0∞) else 1) hX (if_pos hXT)).symm

theorem missProb_sampleBelow_cons (l : Level G) (ls : List (Level G)) (Y : Finset G) (T : Finset (Finset G)) :
    (sampleBelow (l :: ls) Y).missProb T =
      avg fun c : (l.design Y).Ω => ∏ X ∈ (l.design Y).draw c, (sampleBelow ls X).missProb T := by
  cases ls with
  | nil => exact prCoin_disjoint_eq (fun c => (l.design Y).draw c) fun _ => T
  | cons l' ls => exact SamplingDesign.missProb_bind (l.design Y) (fun X => sampleBelow (l' :: ls) X) T

variable {Θ : Type} (bad : Θ → Finset (Finset G)) (agree : ℕ → Θ → Θ → Prop)

/-- **The late prover's best miss** below the block `Y` of the levels `ls`, the `k`-th level first, from the trace
`θ`: after each level but the last is drawn, the prover moves to any trace that `agree`s with its current one at that
level, at each drawn block separately; the bad blocks are the last trace's `bad` ones. -/
noncomputable def lateVal : List (Level G) → ℕ → Finset G → Θ → ℝ≥0∞
  | [], _, Y, θ => (sampleBelow [] Y).missProb (bad θ)
  | [l], _, Y, θ => avg fun c : (l.design Y).Ω => ∏ X ∈ (l.design Y).draw c, (sampleBelow [] X).missProb (bad θ)
  | l :: l' :: ls, k, Y, θ => avg fun c : (l.design Y).Ω => ∏ X ∈ (l.design Y).draw c,
      ⨆ (θ' : Θ) (_ : agree k θ θ'), lateVal (l' :: ls) (k + 1) X θ'

theorem lateVal_single (l : Level G) (k : ℕ) (Y : Finset G) (θ : Θ) :
    lateVal bad agree [l] k Y θ =
      avg fun c : (l.design Y).Ω => ∏ X ∈ (l.design Y).draw c, (sampleBelow [] X).missProb (bad θ) := by
  rw [lateVal]

theorem lateVal_cons_cons (l l' : Level G) (ls : List (Level G)) (k : ℕ) (Y : Finset G) (θ : Θ) :
    lateVal bad agree (l :: l' :: ls) k Y θ = avg fun c : (l.design Y).Ω => ∏ X ∈ (l.design Y).draw c,
      ⨆ (θ' : Θ) (_ : agree k θ θ'), lateVal bad agree (l' :: ls) (k + 1) X θ' := by
  rw [lateVal]

variable {bad agree} {near : Finset G → Θ → Θ → Prop}

/-- The late prover's best miss below a block reads the trace only `near` it. -/
theorem lateVal_near (hsymm : ∀ X a b, near X a b → near X b a)
    (hmono : ∀ X Y a b, X ⊆ Y → near Y a b → near X a b) (hbad : ∀ X a b, near X a b → (X ∈ bad a ↔ X ∈ bad b))
    (hreroot : ∀ k X a b a', near X a b → agree k a a' → ∃ b', agree k b b' ∧ near X a' b') :
    ∀ (ls : List (Level G)) (P : Finset (Finset G)), InsideBelow P ls → ∀ k, ∀ X ∈ P, ∀ a b, near X a b →
      lateVal bad agree ls k X a = lateVal bad agree ls k X b := by
  have hleaf : ∀ X a b, near X a b →
      (sampleBelow [] X).missProb (bad a) = (sampleBelow [] X).missProb (bad b) := fun X a b h => by
    show prCoin (fun _ : Unit => Disjoint {X} (bad a)) = prCoin (fun _ : Unit => Disjoint {X} (bad b))
    simp only [Finset.disjoint_singleton_left, hbad X a b h]
  intro ls
  induction ls with
  | nil => intro P _ k X _ a b h; exact hleaf X a b h
  | cons l ls ih =>
    intro P hin k X hX a b h
    cases ls with
    | nil =>
      rw [lateVal_single, lateVal_single]
      exact congrArg avg (funext fun c => Finset.prod_congr rfl fun X' hX' =>
        hleaf X' a b (hmono X' X a b (hin.1 X hX c X' hX').2 h))
    | cons l' ls =>
      have key : ∀ a b, near X a b → ∀ c, ∀ X' ∈ (l.design X).draw c,
          (⨆ (θ : Θ) (_ : agree k a θ), lateVal bad agree (l' :: ls) (k + 1) X' θ) ≤
            ⨆ (θ : Θ) (_ : agree k b θ), lateVal bad agree (l' :: ls) (k + 1) X' θ := by
        intro a b h c X' hX'
        obtain ⟨hpart, hsub⟩ := hin.1 X hX c X' hX'
        refine iSup₂_le fun θ hθ => ?_
        obtain ⟨θ', hθ', hn⟩ := hreroot k X' a b θ (hmono X' X a b hsub h) hθ
        rw [ih l.blocks.parts hin.2 (k + 1) X' hpart θ θ' hn]
        exact le_iSup₂_of_le (f := fun (θ : Θ) (_ : agree k b θ) => lateVal bad agree (l' :: ls) (k + 1) X' θ)
          θ' hθ' le_rfl
      rw [lateVal_cons_cons, lateVal_cons_cons]
      exact congrArg avg (funext fun c => Finset.prod_congr rfl fun X' hX' =>
        le_antisymm (key a b h c X' hX') (key b a (hsymm X a b h) c X' hX'))

/-- **The late prover's best is a trace fixed in advance**: the traces that maximize each block's best miss, glued
at each level, give one trace whose miss is at least the best. -/
theorem exists_lateVal_le [Fintype Θ] (hrefl : ∀ k θ, agree k θ θ)
    (htrans : ∀ k a b c, agree k a b → agree (k + 1) b c → agree k a c) (hsymm : ∀ X a b, near X a b → near X b a)
    (hmono : ∀ X Y a b, X ⊆ Y → near Y a b → near X a b) (hbad : ∀ X a b, near X a b → (X ∈ bad a ↔ X ∈ bad b))
    (hreroot : ∀ k X a b a', near X a b → agree k a a' → ∃ b', agree k b b' ∧ near X a' b') :
    ∀ (ls : List (Level G)) (P : Finset (Finset G)) (k : ℕ), InsideBelow P ls →
      (∀ j, j + 1 < ls.length → ∀ θ (θX : Finset G → Θ), (∀ X, agree (k + j) θ (θX X)) →
        ∃ θs, agree (k + j) θ θs ∧ ∀ X ∈ Level.above P ls (j + 1), near X θs (θX X)) →
      ∀ θ, ∃ θs, agree k θ θs ∧ ∀ Y ∈ P, lateVal bad agree ls k Y θ ≤ (sampleBelow ls Y).missProb (bad θs) := by
  classical
  intro ls
  induction ls with
  | nil => intro P k _ _ θ; exact ⟨θ, hrefl k θ, fun Y _ => le_rfl⟩
  | cons l ls ih =>
    intro P k hin hglue θ
    cases ls with
    | nil =>
      refine ⟨θ, hrefl k θ, fun Y _ => le_of_eq ?_⟩
      rw [missProb_sampleBelow_cons, lateVal_single]
    | cons l' ls =>
      have hmax : ∀ X, ∃ θ', agree k θ θ' ∧ ∀ θ'', agree k θ θ'' →
          lateVal bad agree (l' :: ls) (k + 1) X θ'' ≤ lateVal bad agree (l' :: ls) (k + 1) X θ' := fun X => by
        obtain ⟨θ', h1, h2⟩ := Finset.exists_max_image (Finset.univ.filter (agree k θ))
          (lateVal bad agree (l' :: ls) (k + 1) X) ⟨θ, Finset.mem_filter.2 ⟨Finset.mem_univ _, hrefl k θ⟩⟩
        exact ⟨θ', (Finset.mem_filter.1 h1).2, fun θ'' h => h2 θ'' (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩)⟩
      choose θX hθX hmaxX using hmax
      obtain ⟨θg, hg, hnear⟩ := hglue 0 (by simp) θ θX hθX
      obtain ⟨θs, hs, hle⟩ := ih l.blocks.parts (k + 1) hin.2 (fun j hj θ' θX' h => by
        have := hglue (j + 1) (by simp only [List.length_cons] at hj ⊢; omega) θ' θX'
          (by rwa [show k + (j + 1) = k + 1 + j by omega])
        rw [show k + (j + 1) = k + 1 + j by omega] at this
        exact this) θg
      refine ⟨θs, htrans k θ θg θs hg hs, fun Y hY => ?_⟩
      rw [missProb_sampleBelow_cons, lateVal_cons_cons]
      refine avg_mono fun c => Finset.prod_le_prod fun X hX => ?_
      have hpart := (hin.1 Y hY c X hX).1
      refine (iSup₂_le fun θ'' h => hmaxX X θ'' h).trans ?_
      rw [← lateVal_near hsymm hmono hbad hreroot (l' :: ls) l.blocks.parts hin.2 (k + 1) X hpart θg (θX X)
        (hnear X hpart)]
      exact hle X hpart

end Level

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G] {Θ : Type} (bad : Θ → Finset (Finset G))
  (agree : ℕ → Θ → Θ → Prop)

theorem lateVal_drop (L : Levels G) {k : ℕ} (hk : k + 1 < L.levels.length) (Y : Finset G) (θ : Θ) :
    Level.lateVal bad agree (L.levels.drop k) k Y θ =
      avg fun c : ((L.levels.get ⟨k, by omega⟩).design Y).Ω =>
        ∏ X ∈ ((L.levels.get ⟨k, by omega⟩).design Y).draw c,
          ⨆ (θ' : Θ) (_ : agree k θ θ'), Level.lateVal bad agree (L.levels.drop (k + 1)) (k + 1) X θ' := by
  rw [List.drop_eq_getElem_cons (by omega : k < L.levels.length), List.drop_eq_getElem_cons hk,
    Level.lateVal_cons_cons]
  rfl

theorem lateVal_drop_last (L : Levels G) {k : ℕ} (hk : k + 1 = L.levels.length) (Y : Finset G) (θ : Θ) :
    Level.lateVal bad agree (L.levels.drop k) k Y θ =
      avg fun c : ((L.levels.get ⟨k, by omega⟩).design Y).Ω =>
        ∏ X ∈ ((L.levels.get ⟨k, by omega⟩).design Y).draw c, (Level.sampleBelow [] X).missProb (bad θ) := by
  rw [List.drop_eq_getElem_cons (by omega : k < L.levels.length), List.drop_eq_nil_of_le (by omega),
    Level.lateVal_single]
  rfl

/-- **The late prover gets at most its best miss**: a prover whose trace after the `k`-th level's draw reads only the
coins drawn so far and agrees with the trace before it at that level misses no more often than `lateVal`. -/
theorem prCoin_le_lateVal (L : Levels G) (hn : L.levels ≠ []) (τ : ℕ → L.Coins → Θ)
    (hτ : ∀ k, L.ReadsBelow k (τ k)) (hstep : ∀ k ω, agree k (τ k ω) (τ (k + 1) ω)) (ω₀ : L.Coins) :
    prCoin (fun ω => Disjoint (L.drawTo ω L.levels.length) (bad (τ (L.levels.length - 1) ω))) ≤
      Level.lateVal bad agree L.levels 0 Finset.univ (τ 0 ω₀) := by
  obtain ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero (fun h => hn (List.length_eq_zero_iff.1 h))
  let P : ℕ → ℝ≥0∞ := fun j =>
    avg fun ω => ∏ Y ∈ L.drawTo ω j, Level.lateVal bad agree (L.levels.drop j) j Y (τ j ω)
  have hP : ∀ j, j + 1 < L.levels.length → P (j + 1) ≤ P j := by
    intro j hj
    calc P (j + 1) ≤ avg fun ω => ∏ X ∈ L.drawTo ω (j + 1),
          ⨆ (θ' : Θ) (_ : agree j (τ j ω) θ'), Level.lateVal bad agree (L.levels.drop (j + 1)) (j + 1) X θ' :=
          avg_mono fun ω => Finset.prod_le_prod fun X _ =>
            le_iSup₂_of_le (f := fun (θ' : Θ) (_ : agree j (τ j ω) θ') =>
              Level.lateVal bad agree (L.levels.drop (j + 1)) (j + 1) X θ') (τ (j + 1) ω) (hstep j ω) le_rfl
      _ = P j := by
          rw [L.avg_prod_drawTo_succ (by omega) _ fun ω ω' h => by rw [hτ j ω ω' h]]
          exact congrArg avg (funext fun ω => Finset.prod_congr rfl fun Y _ =>
            (L.lateVal_drop bad agree hj Y (τ j ω)).symm)
  have hP0 : ∀ j, j < L.levels.length → P j ≤ P 0 := by
    intro j
    induction j with
    | zero => exact fun _ => le_rfl
    | succ j ih => exact fun hj => (hP j hj).trans (ih (by omega))
  have hlast : prCoin (fun ω => Disjoint (L.drawTo ω L.levels.length) (bad (τ (L.levels.length - 1) ω))) = P m := by
    rw [Level.prCoin_disjoint_eq, show L.levels.length - 1 = m by omega, hm,
      L.avg_prod_drawTo_succ (by omega) _ fun ω ω' h => by rw [hτ m ω ω' h]]
    exact congrArg avg (funext fun ω => Finset.prod_congr rfl fun Y _ =>
      (L.lateVal_drop_last bad agree hm.symm Y (τ m ω)).symm)
  have hfirst : P 0 = Level.lateVal bad agree L.levels 0 Finset.univ (τ 0 ω₀) := by
    have h0 : ∀ ω, τ 0 ω = τ 0 ω₀ := fun ω => hτ 0 ω ω₀ fun i hi => absurd hi (Nat.not_lt_zero _)
    show (avg fun ω => ∏ Y ∈ ({Finset.univ} : Finset (Finset G)),
      Level.lateVal bad agree (L.levels.drop 0) 0 Y (τ 0 ω)) = _
    simp only [Finset.prod_singleton, List.drop_zero, h0, avg_const]
  rw [hlast, ← hfirst]
  exact hP0 m (by omega)

/-- **Committing late does not help, at any number of levels.** The prover commits a trace before the first draw and,
after each level's draw, moves to a trace reading only the coins drawn so far that `agree`s with the one before at
that level. Its last trace's `bad` blocks escape the sample no more often than those of one trace fixed in advance
that agrees with its first. The traces must glue: at each level but the last, traces agreeing with one at that level
are one trace `near` each of its blocks; whether a block is bad, and what agrees at a level, read a trace only near
the block. -/
theorem late_le_missProb_levels (L : Levels G) [Fintype Θ] (near : Finset G → Θ → Θ → Prop)
    (hrefl : ∀ k θ, agree k θ θ) (htrans : ∀ k a b c, agree k a b → agree (k + 1) b c → agree k a c)
    (hsymm : ∀ X a b, near X a b → near X b a) (hmono : ∀ X Y a b, X ⊆ Y → near Y a b → near X a b)
    (hbad : ∀ X a b, near X a b → (X ∈ bad a ↔ X ∈ bad b))
    (hreroot : ∀ k X a b a', near X a b → agree k a a' → ∃ b', agree k b b' ∧ near X a' b')
    (hglue : ∀ k (hk : k + 1 < L.levels.length) θ (θX : Finset G → Θ), (∀ X, agree k θ (θX X)) →
      ∃ θs, agree k θ θs ∧ ∀ X ∈ (L.levels.get ⟨k, by omega⟩).blocks.parts, near X θs (θX X))
    (τ : ℕ → L.Coins → Θ) (hτ : ∀ k, L.ReadsBelow k (τ k)) (hstep : ∀ k ω, agree k (τ k ω) (τ (k + 1) ω)) :
    ∃ θ, agree 0 (τ 0 (Classical.arbitrary _)) θ ∧
      prCoin (fun ω => Disjoint (L.drawTo ω L.levels.length) (bad (τ (L.levels.length - 1) ω))) ≤
        L.design.missProb (bad θ) := by
  set ω₀ : L.Coins := Classical.arbitrary _
  have h0 : ∀ ω, τ 0 ω = τ 0 ω₀ := fun ω => hτ 0 ω ω₀ fun i hi => absurd hi (Nat.not_lt_zero _)
  by_cases hn : L.levels = []
  · refine ⟨τ 0 ω₀, hrefl 0 _, le_of_eq ?_⟩
    have hl : L.levels.length = 0 := by rw [hn]; rfl
    show _ = (Level.sampleBelow L.levels Finset.univ).missProb _
    rw [hl, hn]
    show prCoin (fun ω => Disjoint ({Finset.univ} : Finset (Finset G)) (bad (τ 0 ω))) =
      prCoin (fun _ : Unit => Disjoint ({Finset.univ} : Finset (Finset G)) (bad (τ 0 ω₀)))
    unfold prCoin
    simp only [h0, avg_const]
    congr
  obtain ⟨θs, hs, hle⟩ := Level.exists_lateVal_le hrefl htrans hsymm hmono hbad hreroot L.levels {Finset.univ} 0
    L.inside (fun j hj θ θX h => by
      rw [Nat.zero_add] at h ⊢
      rw [Level.above_succ (by omega)]
      exact hglue j hj θ θX h) (τ 0 ω₀)
  exact ⟨θs, hs, (L.prCoin_le_lateVal bad agree hn τ hτ hstep ω₀).trans (hle _ (Finset.mem_singleton_self _))⟩

end Levels

/-- The chain's partition at level `i`, the first below the root at `0`, and the last past its end. -/
def Partition.Chain.part {C : Circuit} : {nc : ℕ} → {Pc : Partition C nc} → Partition.Chain C Pc → ℕ →
    Σ n, Partition C n
  | nc, Pc, .nil, _ => ⟨nc, Pc⟩
  | _, _, .cons Pf _ _ .nil, _ => ⟨_, Pf⟩
  | _, _, .cons Pf _ _ (.cons _ _ _ _), 0 => ⟨_, Pf⟩
  | _, _, .cons _ _ _ (.cons Pg Rg rg rest), i + 1 => (Partition.Chain.cons Pg Rg rg rest).part i

/-- **Committing late does not help, at any number of levels.** At the levels of a circuit's partitions, the trace
`τ k` is committed after `Q_k` is drawn, by any rule, and agrees with `τ (k - 1)` on the committed gates of level
`k - 1`. Every sample misses the incorrect last-level blocks of the last trace with probability at most the miss of
those of one trace fixed in advance that agrees with `τ 0` on the first level's committed gates. -/
def LateAtLevels : Prop :=
  ∀ (C : Circuit) (ch : Partition.Chain C (Partition.whole C)) (τ : ℕ → (Levels.ofPartitions ch).Coins → C.Trace),
    (∀ k, (Levels.ofPartitions ch).ReadsBelow k (τ k)) →
    (∀ k ω, ∀ g ∈ (ch.part k).2.committed, τ (k + 1) ω g = τ k ω g) →
    ∃ τ' : C.Trace, (∀ g ∈ (ch.part 0).2.committed, τ' g = τ 0 (Classical.arbitrary _) g) ∧
      prCoin (fun ω => Disjoint ((Levels.ofPartitions ch).drawTo ω ch.levels.length)
          (((ch.part (ch.levels.length - 1)).2.incorrect (τ (ch.levels.length - 1) ω)).image
            (ch.part (ch.levels.length - 1)).2.block)) ≤
        (Levels.ofPartitions ch).design.missProb
          (((ch.part (ch.levels.length - 1)).2.incorrect τ').image (ch.part (ch.levels.length - 1)).2.block)

end Verity
