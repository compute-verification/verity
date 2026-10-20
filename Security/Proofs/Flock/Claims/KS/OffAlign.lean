import Proofs.Flock.Claims.KS.ForkAlign
import Proofs.Flock.Soundness.Game.LockOff

/-!
# A slot's off mass at its node, aligned with the off mass after a number of coins

`Lock.forkE_le_forkC` moves a fork at a slot's table node to the fork after `j` coins where the strategy reaches the
node after `j` coins (`SlotAt`). Here the same alignment moves the slot's per-position off mass at a plurality chooser
(`Lock.nodeOff`), the raw term of `session_sound_of_table_accL_nodeOff`, to the game's own plurality off mass after `j`
coins (`offC`), whose prefixes are explicit, so the mass can be compared there with the mass off any table that reads
the prefix:

* `Rewinding.plurOf`, `Rewinding.plurOff`: a plurality table of a run distribution and the per-position mass off it,
  at most the mass off any table (`plurOff_le`) and the same at every plurality (`plurOff_eq_of_pluralityE`);
* `offC op j`: one run to its `j`-th coin, then `plurOff` of the continuation; a run that ends before contributes `0`;
* `Rewinding.sum_off_eq_zero_of_conflict`, `Lock.nodeE_nodeOff_eq_zero`: where two runs never conflict, nothing is off a
  plurality table, so a subtree whose fork is `0` (`SlotAt.nul`) has no off mass either;
* `Lock.nodeE_nodeOff_le_offC`: the alignment, at the fork event `SlotAt` is stated for, so every `SlotAt` proof of the
  √ chain applies unchanged; `Lock.nodeE_congr`: aligned locksteps have the same `nodeE`, as `forkE_congr`.
-/

namespace FlockSoundness

namespace Rewinding

open Game

variable {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] (op : α → P → Option V)

/-- **A plurality table of the run distribution.** -/
noncomputable def plurOf [Nonempty V] (g : Game α) (s : Strategy g) : P → V :=
  Classical.choose (exists_pluralityE op g s)

theorem plurOf_spec [Nonempty V] (g : Game α) (s : Strategy g) : IsPluralityE op g s (plurOf op g s) :=
  Classical.choose_spec (exists_pluralityE op g s)

/-- **The plurality off mass**: the sum over the positions of the mass a run opens off a plurality table. -/
noncomputable def plurOff [Nonempty V] (g : Game α) (s : Strategy g) : ℝ :=
  ∑ p, expect (ind fun a => Off op (plurOf op g s) a p) g s

theorem plurOff_nonneg [Nonempty V] (g : Game α) (s : Strategy g) : 0 ≤ plurOff op g s :=
  Finset.sum_nonneg fun _ _ => expect_nonneg (ind_nonneg _) g s

/-- **The plurality off mass is at most the mass off any table.** -/
theorem plurOff_le [Nonempty V] (g : Game α) (s : Strategy g) (w : P → V) :
    plurOff op g s ≤ ∑ p, expect (ind fun a => Off op w a p) g s :=
  Finset.sum_le_sum fun p _ => expect_off_le_of_pluralityE op g s _ (plurOf_spec op g s) w p

/-- **Every plurality table has the plurality off mass.** -/
theorem plurOff_eq_of_pluralityE [Nonempty V] (g : Game α) (s : Strategy g) (vstar : P → V)
    (hv : IsPluralityE op g s vstar) : ∑ p, expect (ind fun a => Off op vstar a p) g s = plurOff op g s :=
  le_antisymm (Finset.sum_le_sum fun p _ => expect_off_le_of_pluralityE op g s vstar hv _ p) (plurOff_le op g s vstar)

/-- A message the strategy sends changes no run, so no plurality off mass. -/
theorem plurOff_send [Nonempty V] {M : Type} {k : M → Game α} (s : Strategy (.send M k)) :
    plurOff op (.send M k) s = plurOff op (k s.1) s.2 := by
  have hv : IsPluralityE op (k s.1) s.2 (plurOf op (.send M k) s) := plurOf_spec op (.send M k) s
  rw [← plurOff_eq_of_pluralityE op (k s.1) s.2 _ hv]
  rfl

/-- **Where two runs never conflict, no run opens a position off a plurality table**: the rewinding lemma's root is
`0`. -/
theorem sum_off_eq_zero_of_conflict (g : Game α) (s : Strategy g) (vstar : P → V) (hv : IsPluralityE op g s vstar)
    (h0 : expect₂ (fun a a' => ind (fun x : α × α => ∃ p, ConflictAt op x p) (a, a')) g s = 0) :
    ∑ p, expect (ind fun a => Off op vstar a p) g s = 0 := by
  have hb := expect_bad_le op g s vstar hv (Fintype.card P)
    (fun _ => (Finset.card_filter_le _ _).trans_eq Finset.card_univ)
  rw [h0, mul_zero, Real.sqrt_zero] at hb
  refine le_antisymm ?_ (Finset.sum_nonneg fun _ _ => expect_nonneg (ind_nonneg _) g s)
  calc ∑ p, expect (ind fun a => Off op vstar a p) g s ≤ ∑ _p : P, (0 : ℝ) :=
        Finset.sum_le_sum fun p _ => (expect_mono (ind_mono fun _ h => ⟨p, h⟩) g s).trans hb
    _ = 0 := Finset.sum_const_zero

end Rewinding

namespace Game

variable {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] [Nonempty V]

/-- One step of `offC`: follow the strategy to the next coin, then `next` at each of its values. -/
noncomputable def offStep (next : (g : Game α) → Strategy g → ℝ) : (g : Game α) → Strategy g → ℝ
  | .ret _, _ => 0
  | .send _ k, s => offStep next (k s.1) s.2
  | @Game.coin _ C iF _ k, s => (∑ c : C, next (k c) (s c)) / (@Fintype.card C iF)

/-- **The plurality off mass after `j` coins**: one run to its `j`-th coin, then the continuation's plurality off mass,
in expectation over the prefix. A run that ends before contributes `0`. -/
noncomputable def offC (op : α → P → Option V) : ℕ → (g : Game α) → Strategy g → ℝ
  | 0 => Rewinding.plurOff op
  | j + 1 => offStep (offC op j)

theorem offC_zero (op : α → P → Option V) (g : Game α) (s : Strategy g) :
    offC op 0 g s = Rewinding.plurOff op g s := rfl

theorem offC_send (op : α → P → Option V) (j : ℕ) {M : Type} {k : M → Game α} (s : Strategy (.send M k)) :
    offC op j (.send M k) s = offC op j (k s.1) s.2 := by
  cases j
  · exact Rewinding.plurOff_send op s
  · rfl

theorem offC_coin (op : α → P → Option V) (j : ℕ) {C : Type} [Fintype C] [Nonempty C] {k : C → Game α}
    (s : Strategy (.coin C k)) :
    offC op (j + 1) (.coin C k) s = (∑ c : C, offC op j (k c) (s c)) / Fintype.card C :=
  rfl

theorem offC_nonneg (op : α → P → Option V) : ∀ (j : ℕ) (g : Game α) (s : Strategy g), 0 ≤ offC op j g s
  | 0, g, s => Rewinding.plurOff_nonneg op g s
  | _ + 1, .ret _, _ => le_rfl
  | j + 1, .send _ k, s => by rw [offC_send]; exact offC_nonneg op (j + 1) (k s.1) s.2
  | j + 1, @Game.coin _ C _ _ k, s =>
      div_nonneg (Finset.sum_nonneg fun c _ => offC_nonneg op j (k c) (s c)) (Nat.cast_nonneg _)

namespace Lock

variable {T D I β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}

/-- **Aligned locksteps have the same node functional**: `nodeE` reads only the compiled side and the slots. -/
theorem nodeE_congr (plur : Chooser T D β) (slot : ℕ) (Ψ : D → (g : Game β) → Strategy g → ℝ)
    {R₁ R₂ : I → α → β → Prop} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : Aligned L₁ L₂) :
    ∀ σ' : Strategy g', nodeE plur slot Ψ L₁ σ' = nodeE plur slot Ψ L₂ σ' := by
  induction h with
  | ret => intro σ'; rfl
  | coin C _ ih =>
      intro σ'
      simp only [nodeE]
      exact congrArg (· / _) (Finset.sum_congr rfl fun c _ => ih c (σ' c))
  | send M _ ih => intro σ'; exact ih σ'.1 σ'.2
  | table s _ ih =>
      intro σ'
      simp only [nodeE]
      split_ifs
      · rfl
      · exact ih _ σ'.1 σ'.2

omit [Nonempty V] in
/-- **No fork, no off mass**: at a plurality chooser, a subtree whose fork conflict at `slot` is `0` has no off mass
at the slot's node. -/
theorem nodeE_nodeOff_eq_zero (op : D → β → P → Option V) (row : T → P → V) (plur : Chooser T D β) (slot : ℕ)
    (hplur : ∀ (d : D) (g'' : Game β) (σ'' : Strategy g''),
      Rewinding.IsPluralityE (op d) g'' σ'' (row (plur slot d g'' σ''))) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      forkE plur slot (fun d b b' => ind (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b')) L σ' = 0 →
        nodeE plur slot (nodeOff op row plur slot) L σ' = 0
  | _, _, _, .ret _, _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', h0 => by
      simp only [forkE] at h0
      simp only [nodeE]
      have hC : (Fintype.card C : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
      have hs := (div_eq_zero_iff.1 h0).resolve_right hC
      have hc := (Finset.sum_eq_zero_iff_of_nonneg fun c _ =>
        forkE_nonneg plur slot (fun _ _ _ => ind_nonneg _ _) (L c) (σ' c)).1 hs
      rw [Finset.sum_eq_zero fun c _ =>
        nodeE_nodeOff_eq_zero op row plur slot hplur (L c) (σ' c) (hc c (Finset.mem_univ c)), zero_div]
  | _, _, _, .send _ L, σ', h0 => nodeE_nodeOff_eq_zero op row plur slot hplur (L σ'.1) σ'.2 h0
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ', h0 => by
      simp only [forkE] at h0
      simp only [nodeE]
      split_ifs at h0 ⊢ with hs
      · exact Rewinding.sum_off_eq_zero_of_conflict (op σ'.1) (k' σ'.1) σ'.2 _ (hplur _ _ _) h0
      · exact nodeE_nodeOff_eq_zero op row plur slot hplur _ σ'.2 h0

/-- **A slot's off mass at its node is at most the plurality off mass after `j` coins** where the strategy reaches the
node after `j` coins (`SlotAt`, at the slot's fork conflict) and, on the outcomes the leaves below relate, every
opening of the slot's reading at the node's cap is an opening of `op'`. -/
theorem nodeE_nodeOff_le_offC (op : D → β → P → Option V) (row : T → P → V) (plur : Chooser T D β) (slot : ℕ)
    (hplur : ∀ (d : D) (g'' : Game β) (σ'' : Strategy g''),
      Rewinding.IsPluralityE (op d) g'' σ'' (row (plur slot d g'' σ'')))
    (op' : β → P → Option V)
    (hop : ∀ t d i a b, R i a b → get i slot = some (t, d) → ∀ p v, op d b p = some v → op' b p = some v)
    {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'}
    (h : SlotAt plur slot (fun d b b' => ind (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b'))
      j L σ') :
    nodeE plur slot (nodeOff op row plur slot) L σ' ≤ offC op' j g' σ' := by
  have hoff : ∀ t d i a b, R i a b → get i slot = some (t, d) → ∀ (w : P → V) (p : P),
      ind (fun b => Rewinding.Off (op d) w b p) b ≤ ind (fun b => Rewinding.Off op' w b p) b := by
    intro t d i a b hr hi w p
    unfold ind
    split_ifs with h1 h2
    · exact le_rfl
    · exfalso
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h1.1
      have hv' := hop t d i a b hr hi p v hv
      exact h2 ⟨by rw [hv']; rfl, by rw [hv']; exact fun e => h1.2 (hv.trans e)⟩
    · exact zero_le_one
    · exact le_rfl
  induction h with
  | ret _ _ => exact offC_nonneg op' _ _ _
  | coin C _ ih =>
      rw [offC_coin]
      simp only [nodeE]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => ih c) (Nat.cast_nonneg _)
  | send M _ ih => rw [offC_send]; exact ih
  | other s hne _ ih =>
      simp only [nodeE, hne, ↓reduceIte]
      rw [offC_send]
      exact ih
  | @here i upd k k' hf hs hk L σ' =>
      simp only [nodeE, ↓reduceIte]
      rw [offC_send, offC_zero]
      calc nodeOff op row plur slot σ'.1 (k' σ'.1) σ'.2
          ≤ ∑ p, expect (ind fun b => Rewinding.Off (op σ'.1) (Rewinding.plurOf op' (k' σ'.1) σ'.2) b p)
              (k' σ'.1) σ'.2 :=
            nodeOff_le_of_plurality op row plur slot σ'.1 (k' σ'.1) σ'.2 (hplur _ _ _) _
        _ ≤ ∑ p, expect (ind fun b => Rewinding.Off op' (Rewinding.plurOf op' (k' σ'.1) σ'.2) b p)
              (k' σ'.1) σ'.2 :=
            Finset.sum_le_sum fun p _ => expect_le_of_get (get := get) (R := R) plur slot
              (plur slot σ'.1 (k' σ'.1) σ'.2, σ'.1)
              (fun i' a b hr hi' => hoff _ _ i' a b hr hi' _ p) (L (plur slot σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2
              (hs _ _)
        _ = Rewinding.plurOff op' (k' σ'.1) σ'.2 := rfl
  | nul h0 =>
      rw [nodeE_nodeOff_eq_zero op row plur slot hplur _ _ h0]
      exact offC_nonneg op' _ _ _

end Lock

end Game

end FlockSoundness
