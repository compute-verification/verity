import Proofs.Flock.Soundness.Discharge.ZkCoins.SessionDefs

/-!
# The session on committed coins, drawn live, is the live session

`Er g G`: the committed-coin interaction `g` is `G` with some of `G`'s coins read off committed rounds' blocks through
balanced decoders. Then `g` drawn live (`erase`) has `G`'s value for every event (`value_erase`), and `Er` is preserved
by `bind`, `interleave`, `batch`, `tagRep` and `coinAt`. So `sessionZKC`, drawn live, has `sessionZK`'s value
(`er_sessionZKC`).
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Model ZkSession Finset
open scoped ENNReal

variable {I Blk α β : Type} [Fintype Blk]

/-- **`g` is `G` with some coins read off committed blocks**: a committed round `coin i` stands for a coin `C` of `G`
through a balanced decoder `d`, a live coin for the same coin. -/
inductive Er : TGame I Blk α → Game α → Prop
  | ret (a : α) : Er (.ret a) (.ret a)
  | send (M : Type) {k : M → TGame I Blk α} {k' : M → Game α} : (∀ m, Er (k m) (k' m)) → Er (.send M k) (.send M k')
  | coin (i : I) {C : Type} [Fintype C] [Nonempty C] (d : Blk → C) (hd : Balanced d) {k : Blk → TGame I Blk α}
      {k' : C → Game α} : (∀ b, Er (k b) (k' (d b))) → Er (.coin i k) (.coin C k')
  | fresh (C : Type) [Fintype C] [Nonempty C] {k : C → TGame I Blk α} {k' : C → Game α} :
      (∀ c, Er (k c) (k' c)) → Er (.fresh C k) (.coin C k')

/-- A balanced decoder turns a uniform block into a uniform coin. -/
theorem balanced_sum [Nonempty Blk] {C : Type} [Fintype C] {d : Blk → C} (hd : Balanced d) (f : C → ℝ≥0∞) :
    ∑ b, (Fintype.card Blk : ℝ≥0∞)⁻¹ * f (d b) = ∑ c, (Fintype.card C : ℝ≥0∞)⁻¹ * f c := by
  classical
  rw [← Finset.sum_fiberwise' univ d (fun c => (Fintype.card Blk : ℝ≥0∞)⁻¹ * f c)]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.sum_const, nsmul_eq_mul, ← mul_assoc]
  congr 1
  have h : Fintype.card C * (univ.filter fun b => d b = c).card = Fintype.card Blk := by convert hd c
  apply ENNReal.eq_inv_of_mul_eq_one_left
  calc ((univ.filter fun b => d b = c).card : ℝ≥0∞) * (Fintype.card Blk : ℝ≥0∞)⁻¹ * Fintype.card C
      = ((Fintype.card C * (univ.filter fun b => d b = c).card : ℕ) : ℝ≥0∞) * (Fintype.card Blk : ℝ≥0∞)⁻¹ := by
        push_cast
        ring
    _ = 1 := by
        rw [h]
        exact ENNReal.mul_inv_cancel (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- **Drawn live, `g` has `G`'s value.** -/
theorem value_erase [Nonempty Blk] {g : TGame I Blk α} {G : Game α} (h : Er g G) (p : α → Prop) :
    value p g.erase = value p G := by
  induction h with
  | ret a => rfl
  | send M _ ih =>
    simp only [TGame.erase, value_send]
    exact iSup_congr ih
  | @coin i C _ _ d hd k k' _ ih =>
    simp only [TGame.erase, value_coin]
    rw [Finset.sum_congr rfl fun b _ => by rw [ih b]]
    exact balanced_sum hd fun c => value p (k' c)
  | fresh C _ ih =>
    simp only [TGame.erase, value_coin]
    exact Finset.sum_congr rfl fun c _ => by rw [ih c]

theorem Er.bind {g : TGame I Blk α} {G : Game α} (h : Er g G) {f : α → TGame I Blk β} {f' : α → Game β}
    (hf : ∀ a, Er (f a) (f' a)) : Er (g.bind f) (G.bind f') := by
  induction h with
  | ret a => exact hf a
  | send M _ ih => exact .send M ih
  | coin i d hd _ ih => exact .coin i d hd ih
  | fresh C _ ih => exact .fresh C ih

theorem Er.map {g : TGame I Blk α} {G : Game α} (h : Er g G) (f : α → β) : Er (g.map f) (G.map f) :=
  h.bind fun a => .ret (f a)

theorem Er.interleaveWith {γ : Type} {adv : TGame I Blk β → TGame I Blk (γ × β)} {fin : β → TGame I Blk (γ × β)}
    {adv' : Game β → Game (γ × β)} {fin' : β → Game (γ × β)} (hadv : ∀ g G, Er g G → Er (adv g) (adv' G))
    (hfin : ∀ b, Er (fin b) (fin' b)) {g : TGame I Blk β} {G : Game β} (h : Er g G) :
    Er (TGame.interleaveWith adv fin g) (Game.interleaveWith adv' fin' G) := by
  induction h with
  | ret b => exact hfin b
  | send M h ih =>
    simp only [TGame.interleaveWith, Game.interleaveWith]
    refine .send Bool fun first => ?_
    cases first
    · simp only [Bool.false_eq_true, ite_false]
      exact .send M ih
    · simp only [ite_true]
      exact hadv _ _ (.send M h)
  | coin i d hd h ih =>
    simp only [TGame.interleaveWith, Game.interleaveWith]
    refine .send Bool fun first => ?_
    cases first
    · simp only [Bool.false_eq_true, ite_false]
      exact .coin i d hd ih
    · simp only [ite_true]
      exact hadv _ _ (.coin i d hd h)
  | fresh C h ih =>
    simp only [TGame.interleaveWith, Game.interleaveWith]
    refine .send Bool fun first => ?_
    cases first
    · simp only [Bool.false_eq_true, ite_false]
      exact .fresh C ih
    · simp only [ite_true]
      exact hadv _ _ (.fresh C h)

theorem Er.interleave {g₁ : TGame I Blk α} {G₁ : Game α} (h₁ : Er g₁ G₁) :
    ∀ {g₂ : TGame I Blk β} {G₂ : Game β}, Er g₂ G₂ → Er (g₁.interleave g₂) (G₁.interleave G₂) := by
  induction h₁ with
  | ret a =>
    intro g₂ G₂ h₂
    simp only [TGame.interleave, Game.interleave]
    exact h₂.map _
  | send M h ih =>
    intro g₂ G₂ h₂
    simp only [TGame.interleave, Game.interleave]
    exact Er.interleaveWith (fun g G hgG => .send M fun m => ih m hgG) (fun b => (Er.send M h).map _) h₂
  | coin i d hd h ih =>
    intro g₂ G₂ h₂
    simp only [TGame.interleave, Game.interleave]
    exact Er.interleaveWith (fun g G hgG => .coin i d hd fun b => ih b hgG) (fun b => (Er.coin i d hd h).map _) h₂
  | fresh C h ih =>
    intro g₂ G₂ h₂
    simp only [TGame.interleave, Game.interleave]
    exact Er.interleaveWith (fun g G hgG => .fresh C fun c => ih c hgG) (fun b => (Er.fresh C h).map _) h₂

theorem Er.batch {gs : List (TGame I Blk α)} {Gs : List (Game α)} (h : List.Forall₂ Er gs Gs) :
    Er (TGame.batch gs) (Game.batch Gs) := by
  induction h with
  | nil => exact .ret []
  | cons h _ ih => exact (h.interleave ih).map _

theorem er_tagRep (dc : Decoders Blk) {R : ℕ} (ix : Fin R → I) : ∀ (t : ℕ) (g : Game α), Er (tagRep dc ix t g) g
  | _, .ret a => .ret a
  | t, .send M k => .send M fun m => er_tagRep dc ix t (k m)
  | t, @Game.coin _ C _ _ k => by
    unfold tagRep
    split_ifs with h
    · split
      · rename_i d hd
        exact .coin _ d (dc.balanced C d hd) fun b => er_tagRep dc ix (t + 1) (k (d b))
      · exact .fresh C fun c => er_tagRep dc ix t (k c)
    · exact .fresh C fun c => er_tagRep dc ix t (k c)

theorem er_coinAt (dc : Decoders Blk) (i : I) {C : Type} [Fintype C] [Nonempty C] {k : C → TGame I Blk α}
    {k' : C → Game α} (h : ∀ c, Er (k c) (k' c)) : Er (coinAt dc i k) (.coin C k') := by
  unfold coinAt
  split
  · rename_i d hd
    exact .coin i d (dc.balanced C d hd) fun b => h (d b)
  · exact .fresh C h

theorem forall₂_ofFn {T₁ T₂ : Type*} {R : T₁ → T₂ → Prop} {n : ℕ} {f : Fin n → T₁} {f' : Fin n → T₂} (h : ∀ i, R (f i) (f' i)) :
    List.Forall₂ R (List.ofFn f) (List.ofFn f') :=
  List.forall₂_iff_get.2 ⟨by simp, fun i h₁ h₂ => by simpa using h ⟨i, by simpa using h₁⟩⟩

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

omit [Fintype K] in
theorem after_eq (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) :
    T.after A C₀ pts = (Game.interleave (repOf A T C₀ pts 0) (repOf A T C₀ pts 1)).map fun a => (a.1 && a.2, C₀) :=
  rfl

omit [Fintype K] in
theorem er_afterC (dc : Decoders Blk) (R : ℕ) {J : ℕ} (j : Fin J) (A : Arith F K) (T : TabZK F mPts) (C₀ : Oracle F)
    (pts : Fin 2 → Fin mPts → F) : Er (afterC dc R j A T C₀ pts) (T.after A C₀ pts) := by
  rw [after_eq]
  exact ((er_tagRep dc _ 0 _).interleave (er_tagRep dc _ 0 _)).map _

omit [Fintype K] in
/-- **The session on committed coins is the live session with its coins read off C-Flock ZK's rounds.** -/
theorem er_sessionZKC (dc : Decoders Blk) (R : ℕ) (A : Arith F K) (ts : List (TabZK F mPts)) :
    Er (sessionZKC dc R A ts) (sessionZK A ts) :=
  .send _ fun Cs => er_coinAt dc none fun pts =>
    Er.batch (forall₂_ofFn fun i => er_afterC dc R i A (ts.get i) (Cs i) pts)

end FlockSoundness.Discharge.ZkCoins
