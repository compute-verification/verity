import Properties.Verity.Overhead
import Proofs.Properties.Verity.Soundness
import Proofs.Core.Game.Seq
import Mathlib.Algebra.Order.Floor.Ring

/-!
# The end-to-end overhead guarantee, from the one theorem

`overheadFor`: in the experiment, the event `OverheadFor` bounds lies inside three events. The first is that some window
was accepted on a registered file while it is not clean, within its error budget. The second is that some window's error
exceeds its budget. The third is coverage's exception. On none of them the counting below rules it out:

* every window is clean, so each window materializes at most `q` counted products (`ServingCap`). A window opening after
  its server's last computed gate materializes none;
* at most `q_D` are materialized outside every window (`DarkShare`), and success outside coverage's exception
  materializes at least `N`;
* so the active windows number `K ≥ (N − q_D)/q`, and a server busy for `b` ticks has at most `b/ω + 1` of them. Hence
  `ω K ≤ server-time + n ω`, and server-time is at least `ω (⌈(N − q_D)/q⌉ − n)` (`minServerTime_le`).

The first event's probability is at most the budgets summed: at window `i`, whatever the prover committed to before it,
the window's game is `Vocab.game` at the strategy it commits to (`prob_follow`), where `Vocab.SoundnessFor` and the
wipe's target bound the window's error (`prob_unclean_le`), and that bound is within the budget or the event is empty
(`prob_windows_le`).

`overhead_nontrivial`: at a window where a claim fails above the summed errors (the hypotheses of
`soundness_of_subprotocols_nontrivial_of`), one server and one window whose run succeeds exactly when the window is not
clean make `OverheadFor`'s conclusion fail for the verifier that accepts everything.
-/

namespace Verity

open Accounting Overhead
open scoped ENNReal

namespace Overhead

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- Following a strategy keeps its probabilities. -/
theorem prob_follow {β : Type} (p : β → Prop) :
    ∀ (g : Game β) (s : Game.Strategy g) (t : Game.Strategy (follow g s)),
      Game.prob p (follow g s) t = Game.prob p g s
  | .ret _, _, _ => rfl
  | .send _ k, s, t => prob_follow p (k s.1) s.2 t
  | @Game.coin _ C _ _ k, s, t => by
      show Game.avg (fun c => Game.prob p (follow (k c) (s c)) (t c)) = Game.avg fun c => Game.prob p (k c) (s c)
      exact congrArg Game.avg (funext fun c => prob_follow p (k c) (s c) (t c))

open Classical in
/-- An event that holds on every outcome or on none. -/
theorem prob_const {β : Type} (Q : Prop) (g : Game β) (s : Game.Strategy g) :
    Game.prob (fun _ => Q) g s = if Q then 1 else 0 := by
  classical
  by_cases hQ : Q
  · simpa [hQ] using Game.prob_true g s
  · simpa [hQ] using Game.prob_false g s

/-- **Window `i`'s event is bounded by its game's.** If at every window the prover could commit to, its game's
probability of `P` at that window is at most `ε`, the probability that the `i`-th window played satisfies `P` is at
most `ε`, whatever the earlier windows were. -/
theorem prob_windows_le (V : Vocab d α) (P : Window V × Outcome → Prop) (ε : ℝ≥0∞)
    (h : ∀ w : Window V, Game.prob (fun o => P (w, o)) (V.game w.θ w.F w.c) w.σ ≤ ε) :
    ∀ (n i : ℕ) (s : Game.Strategy (windows V n)),
      Game.prob (fun ws => ∃ e, ws[i]? = some e ∧ P e) (windows V n) s ≤ ε
  | 0, i, s => by
      show Game.prob _ (.ret []) s ≤ ε
      simp [Game.prob]
  | n + 1, i, s => by
      show Game.prob _ ((follow (V.game s.1.θ s.1.F s.1.c) s.1.σ).bind
        fun o => (windows V n).map ((s.1, o) :: ·)) s.2 ≤ ε
      cases i with
      | zero =>
          rw [Game.prob_bind_of_const _ _ (fun o => P (s.1, o)) (fun o t => ?_), prob_follow]
          · exact h s.1
          · rw [Game.prob_map]
            simp only [List.getElem?_cons_zero, Option.some.injEq, exists_eq_left']
            by_cases hP : P (s.1, o) <;> simp [hP, Game.prob, Game.prob_true, Game.prob_false]
      | succ i =>
          refine (Game.prob_bind_le _ _ (fun _ => True) ε (fun o _ t => ?_) _ _).trans ?_
          · rw [Game.prob_map]
            exact (Game.prob_mono (q := fun ws => ∃ e, ws[i]? = some e ∧ P e) (fun l hl => by simpa using hl) _ _).trans
              (prob_windows_le V P ε h n i _)
          · simp [Game.prob_false]

/-- A window that the acceptance `acc` accepts on a registered file while it is not clean: at most its error. -/
theorem prob_unclean_le (V : Vocab d α) {In L Srv ιP Act : Type} [Fintype Srv] {B : ℕ}
    (D : Deployment V In L Srv ιP Act B) (acc : Sent → Prop) (f : ℝ) (εWipe : Window V → ℝ≥0∞)
    (hV : V.SoundnessFor acc) (hW : D.WipeSound acc f εWipe) (w : Window V) :
    Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧ ¬ Clean V D.wipe f (w, o))
      (V.game w.θ w.F w.c) w.σ ≤ windowError V εWipe w := by
  calc Game.prob (fun o => acc o.sent ∧ V.Registered o.params o.file ∧ ¬ Clean V D.wipe f (w, o))
        (V.game w.θ w.F w.c) w.σ
      ≤ Game.prob (fun o => (acc o.sent ∧ V.Registered o.params o.file ∧ ∃ a, ¬ V.Holds w.θ w.F w.c w.σ a o) ∨
          (acc o.sent ∧ V.Registered o.params o.file ∧ ¬ (D.wipe w).Survives f o)) (V.game w.θ w.F w.c) w.σ :=
        Game.prob_mono (fun o ⟨ha, hr, hc⟩ => by
          by_cases hh : ∀ a, V.Holds w.θ w.F w.c w.σ a o
          · exact Or.inr ⟨ha, hr, fun hs => hc ⟨hh, hs⟩⟩
          · push_neg at hh
            exact Or.inl ⟨ha, hr, hh⟩) _ _
    _ ≤ _ := Game.prob_or_le _ _ _ _
    _ ≤ ∑ a, V.error a w.θ w.F w.c w.σ + εWipe w := add_le_add (hV w.θ w.F w.c w.σ) (hW w)

section Experiment

variable {V : Vocab d α} {In Y : Type} {G : Task In Y} {B : ℕ} {τ : In → Game.Strategy (windows V B)}

theorem prob_mono' {E E' : In → Set Y → List (Window V × Outcome) → Prop}
    (h : ∀ x 𝒴 ws, E x 𝒴 ws → E' x 𝒴 ws) : prob V G τ E ≤ prob V G τ E' :=
  Finset.sum_le_sum fun _ _ => mul_le_mul_right (Game.prob_mono (h _ _) _ _) _

theorem prob_or_le' (E E' : In → Set Y → List (Window V × Outcome) → Prop) :
    prob V G τ (fun x 𝒴 ws => E x 𝒴 ws ∨ E' x 𝒴 ws) ≤ prob V G τ E + prob V G τ E' := by
  unfold prob
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun _ _ => (mul_le_mul_right (Game.prob_or_le _ _ _ _) _).trans (mul_add _ _ _).le

theorem prob_exists_le' {κ : Type} [Fintype κ] (E : κ → In → Set Y → List (Window V × Outcome) → Prop) :
    prob V G τ (fun x 𝒴 ws => ∃ i, E i x 𝒴 ws) ≤ ∑ i, prob V G τ (E i) := by
  unfold prob
  rw [Finset.sum_comm]
  refine Finset.sum_le_sum fun ω _ => ?_
  rw [← Finset.mul_sum]
  exact mul_le_mul_right ((Game.prob_mono (fun ws ⟨i, hi⟩ => ⟨i, Finset.mem_univ i, hi⟩) _ _).trans
    (prob_exists_finset_le Finset.univ (fun i => E i (G.draw ω).1 (G.draw ω).2) _ _)) _

theorem prob_le_of_le {E : In → Set Y → List (Window V × Outcome) → Prop} {c : ℝ≥0∞}
    (h : ∀ x 𝒴 (s : Game.Strategy (windows V B)), Game.prob (E x 𝒴) (windows V B) s ≤ c) : prob V G τ E ≤ c :=
  calc prob V G τ E ≤ Game.avg (fun _ : G.Ω => c) := Game.avg_mono fun _ => h _ _ _
    _ = c := Game.avg_const c

/-- An event that no outcome meets has probability zero. -/
theorem prob_eq_zero_of_not {β : Type} {p : β → Prop} (h : ∀ b, ¬ p b) (g : Game β) (s : Game.Strategy g) :
    Game.prob p g s = 0 :=
  le_antisymm ((Game.prob_mono (q := fun _ => False) (fun b hb => h b hb) g s).trans_eq (Game.prob_false g s))
    zero_le

theorem prob_eq_zero' {E : In → Set Y → List (Window V × Outcome) → Prop}
    (h : ∀ ω ws, ¬ E (G.draw ω).1 (G.draw ω).2 ws) : prob V G τ E = 0 :=
  Finset.sum_eq_zero fun ω _ => by rw [prob_eq_zero_of_not (h ω), mul_zero]

end Experiment

/-- At files with the registered digest, an event that holds wherever `p` does on a registered file is at least as
likely as `p`: every play is on the files committed to. -/
theorem prob_le_of_registered (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (hreg : V.Registered θ F) {p q : Outcome → Prop} (h : ∀ o, V.Registered o.params o.file → p o → q o) :
    Game.prob p (V.game θ F c) σ ≤ Game.prob q (V.game θ F c) σ := by
  have hz : Game.prob (fun o => ¬ V.Registered o.params o.file) (V.game θ F c) σ = 0 :=
    prob_play_eq_zero (V.rounds θ) θ F c _ (fun _ _ hn => hn hreg) σ
  calc Game.prob p (V.game θ F c) σ
      ≤ Game.prob (fun o => q o ∨ ¬ V.Registered o.params o.file) (V.game θ F c) σ :=
        Game.prob_mono (fun o hp => by
          by_cases hr : V.Registered o.params o.file
          · exact Or.inl (h o hr hp)
          · exact Or.inr hr) _ σ
    _ ≤ Game.prob q (V.game θ F c) σ + Game.prob (fun o => ¬ V.Registered o.params o.file) (V.game θ F c) σ :=
        Game.prob_or_le _ _ _ σ
    _ = Game.prob q (V.game θ F c) σ := by rw [hz, add_zero]

/-- A server busy for `b` ticks opens at most `b / ω + 1` of its windows before its busy time ends. -/
theorem card_active_le (H ω b : ℕ) (hω : 0 < ω) :
    ω * (Finset.univ.filter fun j : Fin H => ω * j < b).card ≤ b + ω := by
  have hsub : ((Finset.univ.filter fun j : Fin H => ω * j < b).map Fin.valEmbedding) ⊆ Finset.range (b / ω + 1) := by
    intro j hj
    simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply] at hj
    obtain ⟨j, hj, rfl⟩ := hj
    simp only [Finset.mem_range, Nat.lt_succ_iff]
    exact (Nat.le_div_iff_mul_le hω).2 (by rw [mul_comm]; exact hj.le)
  have hc := (Finset.card_le_card hsub).trans_eq (Finset.card_range _)
  rw [Finset.card_map] at hc
  calc ω * (Finset.univ.filter fun j : Fin H => ω * j < b).card ≤ ω * (b / ω + 1) := Nat.mul_le_mul_left _ hc
    _ = ω * (b / ω) + ω := by rw [mul_add, mul_one]
    _ ≤ b + ω := Nat.add_le_add_right (Nat.mul_div_le b ω) _

namespace Deployment

variable {V : Vocab d α} {In L Srv ιP Act : Type} [Fintype Srv] {B : ℕ} (D : Deployment V In L Srv ιP Act B)

theorem inWindow_subset (x : In) (ws : List (Window V × Outcome)) (i : Fin B) :
    D.inWindow x ws i ⊆ (D.prods x ws).J := by
  classical
  exact fun _ hp => (Finset.mem_filter.1 hp).1

/-- A window that opens at or after its server's busy time ends materializes nothing. -/
theorem inWindow_eq_empty (x : In) (ws : List (Window V × Outcome)) (i : Fin B)
    (hi : busy (D.run x ws).A (D.X (D.win i).1) ≤ D.ω * (D.win i).2) : D.inWindow x ws i = ∅ := by
  classical
  refine Finset.eq_empty_of_forall_notMem fun p hp => ?_
  obtain ⟨hpJ, hx, hlo, -⟩ := Finset.mem_filter.1 hp
  have hc : (D.prods x ws).rep p ∈ (D.run x ws).A.computed := by
    simp [Circuit.computed, (D.prods x ws).rep_op p hpJ]
  have hle : (D.run x ws).A.t ((D.prods x ws).rep p) + 1 ≤ busy (D.run x ws).A (D.X (D.win i).1) := by
    unfold busy
    exact Finset.le_sup (f := fun g => (D.run x ws).A.t g + 1) (Finset.mem_filter.2 ⟨hc, hx⟩)
  exact absurd (lt_of_le_of_lt hlo (Nat.lt_of_succ_le hle)) (not_lt.2 hi)

/-- Every counted product is materialized in some window or outside them all. -/
theorem card_le_dark_add (x : In) (ws : List (Window V × Outcome)) :
    (D.prods x ws).J.card ≤ (D.dark x ws).card + ∑ i, (D.inWindow x ws i).card := by
  classical
  calc (D.prods x ws).J.card ≤ (D.dark x ws ∪ Finset.univ.biUnion fun i => D.inWindow x ws i).card := by
        refine Finset.card_le_card fun p hp => ?_
        by_cases h : ∀ i, ¬ InWindow D.X D.win D.ω (D.run x ws).A i ((D.prods x ws).rep p)
        · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hp, h⟩)
        · push_neg at h
          obtain ⟨i, hi⟩ := h
          exact Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨i, Finset.mem_univ _, Finset.mem_filter.2 ⟨hp, hi⟩⟩)
    _ ≤ (D.dark x ws).card + (Finset.univ.biUnion fun i => D.inWindow x ws i).card := Finset.card_union_le _ _
    _ ≤ _ := Nat.add_le_add_left Finset.card_biUnion_le _

/-- The windows that open before their server's busy time ends, `ω` times their number, are at most the server-time
plus one window per server. -/
theorem active_le (x : In) (ws : List (Window V × Outcome)) (hω : 0 < D.ω) :
    D.ω * (Finset.univ.filter fun i : Fin B => D.ω * (D.win i).2 < busy (D.run x ws).A (D.X (D.win i).1)).card ≤
      serverTime D.X (D.run x ws).A + Fintype.card Srv * D.ω := by
  classical
  have hcard : (Finset.univ.filter fun i : Fin B => D.ω * (D.win i).2 < busy (D.run x ws).A (D.X (D.win i).1)).card =
      ∑ s, (Finset.univ.filter fun j : Fin D.H => D.ω * j < busy (D.run x ws).A (D.X s)).card := by
    rw [Finset.card_filter, ← Equiv.sum_comp D.win.symm, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [Finset.card_filter]
    simp
  rw [hcard, Finset.mul_sum, serverTime]
  calc ∑ s, D.ω * (Finset.univ.filter fun j : Fin D.H => D.ω * j < busy (D.run x ws).A (D.X s)).card
      ≤ ∑ s, (busy (D.run x ws).A (D.X s) + D.ω) := Finset.sum_le_sum fun s _ => card_active_le D.H D.ω _ hω
    _ = ∑ s, busy (D.run x ws).A (D.X s) + Fintype.card Srv * D.ω := by
        rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, smul_eq_mul]

end Deployment

/-- **The counting step.** At a run whose `B` windows each materialize at most `q` counted products, with at most `q_D`
outside them and at least `N` in all, the server-time is at least `ω (⌈(N − q_D)/q⌉ − n)`. -/
theorem minServerTime_le {V : Vocab d α} {In L Srv ιP Act : Type} [Fintype Srv] {B : ℕ}
    (D : Deployment V In L Srv ιP Act B) (x : In) (ws : List (Window V × Outcome)) (q qD : ℝ) (N : ℕ)
    (hω : 0 < D.ω) (hq : 0 < q) (hcap : ∀ i, ((D.inWindow x ws i).card : ℝ) ≤ q)
    (hdark : ((D.dark x ws).card : ℝ) ≤ qD) (hN : N ≤ (D.prods x ws).J.card) :
    minServerTime D.ω N q qD (Fintype.card Srv) ≤ serverTime D.X (D.run x ws).A := by
  classical
  set A := Finset.univ.filter fun i : Fin B => D.ω * (D.win i).2 < busy (D.run x ws).A (D.X (D.win i).1)
  set K := A.card
  have hsum : (∑ i, ((D.inWindow x ws i).card : ℝ)) ≤ q * K := by
    calc (∑ i, ((D.inWindow x ws i).card : ℝ)) = ∑ i ∈ A, ((D.inWindow x ws i).card : ℝ) := by
          refine (Finset.sum_subset (Finset.subset_univ _) fun i _ hi => ?_).symm
          have : busy (D.run x ws).A (D.X (D.win i).1) ≤ D.ω * (D.win i).2 := by
            simpa [A, not_lt] using hi
          simp [D.inWindow_eq_empty x ws i this]
      _ ≤ ∑ _i ∈ A, q := Finset.sum_le_sum fun i _ => hcap i
      _ = q * K := by rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
  have hcount : (N : ℝ) ≤ qD + q * K := by
    have := D.card_le_dark_add x ws
    have h1 : ((D.prods x ws).J.card : ℝ) ≤ (D.dark x ws).card + ∑ i, ((D.inWindow x ws i).card : ℝ) := by
      exact_mod_cast this
    have h2 : (N : ℝ) ≤ (D.prods x ws).J.card := by exact_mod_cast hN
    linarith
  have hact : (D.ω : ℝ) * K ≤ serverTime D.X (D.run x ws).A + Fintype.card Srv * D.ω := by
    exact_mod_cast D.active_le x ws hω
  by_contra hlt
  push_neg at hlt
  unfold minServerTime at hlt
  have hωr : (0 : ℝ) < D.ω := by exact_mod_cast hω
  have hK : ((K : ℤ) : ℝ) < ((⌈((N : ℝ) - qD) / q⌉ : ℤ) : ℝ) := by
    have : (D.ω : ℝ) * K < D.ω * ((⌈((N : ℝ) - qD) / q⌉ : ℤ) : ℝ) := by linarith
    push_cast
    exact lt_of_mul_lt_mul_left this hωr.le
  have hK' : ((K : ℤ) : ℝ) < ((N : ℝ) - qD) / q := Int.lt_ceil.1 (Int.cast_lt.1 hK)
  rw [lt_div_iff₀ hq] at hK'
  push_cast at hK'
  linarith

/-! ### The one-window deployment of `overhead_nontrivial` -/

/-- The strategy of a game whose messages `follow` has fixed: there are none left to send. -/
def followStrategy {β : Type} : (g : Game β) → (s : Game.Strategy g) → Game.Strategy (follow g s)
  | .ret _, _ => PUnit.unit
  | .send _ k, s => followStrategy (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => fun c => followStrategy (k c) (s c)

/-- Commit to the window `w` and play its game. -/
def oneStrategy (V : Vocab d α) (w : Window V) : Game.Strategy (windows V 1) :=
  ⟨w, Game.Strategy.bind _ _ (followStrategy _ w.σ) fun _ => PUnit.unit⟩

/-- One input, whose acceptable outputs are those with `[true]` at the one position. -/
def oneTask : Task Unit (Fin 1 → List Bool) := ⟨Unit, fun _ => ((), {y | y 0 = [true]})⟩

/-- One window played is its window's game. -/
theorem prob_windows_one (V : Vocab d α) (P : List (Window V × Outcome) → Prop) (s : Game.Strategy (windows V 1)) :
    Game.prob P (windows V 1) s = Game.prob (fun o => P [(s.1, o)]) (V.game s.1.θ s.1.F s.1.c) s.1.σ := by
  show Game.prob P ((follow (V.game s.1.θ s.1.F s.1.c) s.1.σ).bind
    fun o => (windows V 0).map ((s.1, o) :: ·)) s.2 = _
  rw [Game.prob_bind_of_const (fun o => (windows V 0).map ((s.1, o) :: ·)) P (fun o => P [(s.1, o)])
    (fun _ _ => rfl), prob_follow]

/-- At `oneTask` and one window committed to, the experiment is that window's game. -/
theorem prob_one (V : Vocab d α) (w : Window V)
    (E : Unit → Set (Fin 1 → List Bool) → List (Window V × Outcome) → Prop) :
    prob V oneTask (fun _ => oneStrategy V w) E =
      Game.prob (fun o => E () {y | y 0 = [true]} [(w, o)]) (V.game w.θ w.F w.c) w.σ := by
  show Game.avg (fun _ : Unit => Game.prob (E () {y | y 0 = [true]}) (windows V 1) (oneStrategy V w)) = _
  rw [Game.avg_const, prob_windows_one]
  rfl

/-- One computed gate of width one, at leaf `()`, cell `0` and tick `0`, whose value is `b`. -/
def oneCircuit (b : Bool) : Placed Unit where
  N := 1
  w _ := 1
  isInput _ := false
  reads _ := ∅
  reads_lt _ _ h := by simp at h
  input_reads _ _ := rfl
  op _ _ := BitVec.ofBool b
  op_congr _ _ _ _ := rfl
  outputs := ∅
  x _ := ()
  t _ := 0
  cell _ := 0
  cell_inj g h _ := Fin.ext (by have := g.isLt; have := h.isLt; omega)

/-- `oneCircuit b`'s run: its gate's value is `b`, and nothing is held. -/
def oneRun (b : Bool) : Actual.Run Unit where
  A := oneCircuit b
  val _ := BitVec.ofBool b
  faithful _ _ := rfl
  held _ _ := ∅
  holding := ⟨fun _ _ _ h => by simp at h, fun _ h => absurd h Bool.false_ne_true,
    fun _ _ h => absurd h (Finset.notMem_empty _)⟩

/-- One counted product, at the gate, when `b`. -/
def oneProds (b : Bool) : Nci.Ldl.Products (oneRun b).A.toCircuit Unit Unit where
  J := if b then {()} else ∅
  rep _ := ⟨0, Nat.one_pos⟩
  rep_inj := by intro p _ q _ _; exact Subsingleton.elim p q
  rep_op _ _ := rfl
  act _ := ()

/-- A wipe that leaves nothing. -/
def oneWipe (d : Device ι) : WipeLive d Outcome where
  fill := 0
  fill_le := Nat.zero_le _
  accepts _ := True
  kept _ := 0

theorem oneWipe_survives (f : ℝ) (o : Outcome) : (oneWipe d).Survives f o := by
  unfold WipeLive.Survives oneWipe
  simp only [Nat.cast_zero, mul_zero, zero_add, sub_zero]
  positivity

/-- The first window's claim `a` fails on its play. -/
def oneBad (V : Vocab d α) (a : Audit) (ws : List (Window V × Outcome)) : Prop :=
  ∃ e, ws[0]? = some e ∧ ¬ V.Holds e.1.θ e.1.F e.1.c e.1.σ a e.2

open Classical in
/-- One server and one window of two ticks; the run's gate holds `1`, and is a counted product, exactly when the
window's claim `a` fails. -/
noncomputable def oneDeployment (V : Vocab d α) (a : Audit) : Deployment V Unit Unit Unit Unit Unit 1 where
  run _ ws := oneRun (decide (oneBad V a ws))
  prods _ ws := oneProds (decide (oneBad V a ws))
  X _ := Finset.univ
  H := 1
  win := (Equiv.punitProd (Fin 1)).symm
  ω := 2
  wipe _ := oneWipe d

open Classical in
theorem oneDeployment_run_of_bad (V : Vocab d α) (a : Audit) {x : Unit} {ws : List (Window V × Outcome)}
    (hb : oneBad V a ws) : (oneDeployment V a).run x ws = oneRun true := by
  show oneRun (decide (oneBad V a ws)) = oneRun true
  rw [decide_eq_true hb]

open Classical in
theorem oneDeployment_J_of_bad (V : Vocab d α) (a : Audit) {x : Unit} {ws : List (Window V × Outcome)}
    (hb : oneBad V a ws) : ((oneDeployment V a).prods x ws).J = {()} := by
  show (oneProds (decide (oneBad V a ws))).J = {()}
  rw [decide_eq_true hb]
  rfl

open Classical in
theorem oneDeployment_J_of_not_bad (V : Vocab d α) (a : Audit) {x : Unit} {ws : List (Window V × Outcome)}
    (hb : ¬ oneBad V a ws) : ((oneDeployment V a).prods x ws).J = ∅ := by
  show (oneProds (decide (oneBad V a ws))).J = ∅
  rw [decide_eq_false hb]
  rfl

/-- The gate is in the one window. -/
theorem inWindow_one (V : Vocab d α) (a : Audit) (x : Unit) (ws : List (Window V × Outcome))
    (g : Fin ((oneDeployment V a).run x ws).A.N) :
    InWindow (oneDeployment V a).X (oneDeployment V a).win (oneDeployment V a).ω ((oneDeployment V a).run x ws).A 0
      g := by
  refine ⟨Finset.mem_univ _, ?_, ?_⟩ <;> simp [oneDeployment, oneRun, oneCircuit, Fin.val_eq_zero]

theorem oneServerTime_le (V : Vocab d α) (a : Audit) (x : Unit) (ws : List (Window V × Outcome)) :
    serverTime (oneDeployment V a).X ((oneDeployment V a).run x ws).A ≤ 1 :=
  calc serverTime (oneDeployment V a).X ((oneDeployment V a).run x ws).A ≤ ∑ _s : Unit, 1 :=
        Finset.sum_le_sum fun _ _ => Finset.sup_le fun _ _ => by simp [oneDeployment, oneRun, oneCircuit]
    _ = 1 := by simp

theorem oneMin (V : Vocab d α) (a : Audit) :
    minServerTime (oneDeployment V a).ω 1 (1 / 2) 0 (Fintype.card Unit) = 2 := by
  have h : ⌈(((1 : ℕ) : ℝ) - 0) / (1 / 2)⌉ = 2 := by
    rw [Int.ceil_eq_iff]
    norm_num
  have hω : (oneDeployment V a).ω = 2 := rfl
  simp only [minServerTime, h, Fintype.card_unit, hω]
  norm_num

/-- The run succeeds when its gate holds `1`. -/
theorem oneSucceeds : Succeeds (oneRun true) (fun _ => ((), 0, 0)) {y : Fin 1 → List Bool | y 0 = [true]} :=
  ⟨fun _ => [true], by show (fun _ : Fin 1 => [true]) 0 = [true]; rfl, fun _ => ⟨⟨0, Nat.one_pos⟩, rfl, by show Actual.bits 1 1 = [true]; decide⟩⟩

/-- And only then. -/
theorem eq_true_of_succeeds {b : Bool} {pos : Fin 1 → Unit × ℕ × ℕ}
    (h : Succeeds (oneRun b) pos {y : Fin 1 → List Bool | y 0 = [true]}) : b = true := by
  obtain ⟨y, hy, hr⟩ := h
  obtain ⟨g, -, hg⟩ := hr 0
  cases b
  · have : Actual.bits 1 0 = [true] := hg.trans hy
    exact absurd this (by decide)
  · rfl

end Overhead

/-- **The overhead guarantee at every acceptance.** -/
theorem overheadFor (acc : Sent → Prop) : OverheadFor acc := by
  classical
  intro ι d α _ V In L Srv ιP Act _ m B G D f εWipe Phys q qD N η hω hq hV hW hPhys hCap hDark hCov pos τ ε
  have hin : ∀ x 𝒴 ws, (AcceptedBy V acc B ws ∧ Succeeds (D.run x ws) pos 𝒴 ∧
      (serverTime D.X (D.run x ws).A : ℝ) < minServerTime D.ω N q qD (Fintype.card Srv)) →
      (∃ i : Fin B, ∃ e, ws[(i : ℕ)]? = some e ∧
        ((acc e.2.sent ∧ V.Registered e.2.params e.2.file ∧ ¬ Clean V D.wipe f e) ∧ windowError V εWipe e.1 ≤ ε i)) ∨
      (∃ i : Fin B, ∃ e, ws[(i : ℕ)]? = some e ∧ ε i < windowError V εWipe e.1) ∨
      (Succeeds (D.run x ws) pos 𝒴 ∧ (D.prods x ws).J.card < N) := by
    intro x 𝒴 ws ⟨⟨hlen, hacc⟩, hsucc, hlt⟩
    by_contra hno
    simp only [not_or, not_exists] at hno
    obtain ⟨hbad, hhigh, hcov⟩ := hno
    have hi : ∀ i : Fin B, (i : ℕ) < ws.length := fun i => hlen ▸ i.2
    have hentry : ∀ i : Fin B, ws[(i : ℕ)]? = some (ws[(i : ℕ)]'(hi i)) := fun i => List.getElem?_eq_getElem (hi i)
    have hmem : ∀ i : Fin B, ws[(i : ℕ)]'(hi i) ∈ ws := fun i => List.getElem_mem (hi i)
    have hclean : ∀ i : Fin B, Clean V D.wipe f (ws[(i : ℕ)]'(hi i)) := by
      intro i
      obtain ⟨ha, hr⟩ := hacc _ (hmem i)
      have hle : windowError V εWipe (ws[(i : ℕ)]'(hi i)).1 ≤ ε i :=
        not_lt.1 fun h => hhigh i _ ⟨hentry i, h⟩
      by_contra hc
      exact hbad i _ ⟨hentry i, ⟨ha, hr, hc⟩, hle⟩
    have hcap : ∀ i, ((D.inWindow x ws i).card : ℝ) ≤ q := fun i =>
      hCap x ws i _ (hentry i) (hacc _ (hmem i)).2 (hclean i) (hPhys x ws)
    have hN : N ≤ (D.prods x ws).J.card := not_lt.1 fun h => hcov ⟨hsucc, h⟩
    exact absurd (minServerTime_le D x ws q qD N hω hq hcap (hDark x ws) hN) (not_le.2 hlt)
  refine (prob_mono' (V := V) (G := G) (B := B) (τ := τ) hin).trans
    ((prob_or_le' (V := V) (G := G) (B := B) (τ := τ) _ _).trans ?_)
  refine (add_le_add le_rfl (prob_or_le' (V := V) (G := G) (τ := τ) _ _)).trans ?_
  rw [← add_assoc]
  refine add_le_add (add_le_add ((prob_exists_le' (V := V) (G := G) (τ := τ) (κ := Fin B) _).trans
    (Finset.sum_le_sum fun i _ => ?_)) le_rfl) (hCov pos τ)
  refine prob_le_of_le fun x 𝒴 s => prob_windows_le V
    (fun e => (acc e.2.sent ∧ V.Registered e.2.params e.2.file ∧ ¬ Clean V D.wipe f e) ∧
      windowError V εWipe e.1 ≤ ε i) (ε i) (fun w => ?_) B i s
  show Game.prob (fun o => (acc o.sent ∧ V.Registered o.params o.file ∧ ¬ Clean V D.wipe f (w, o)) ∧
    windowError V εWipe w ≤ ε i) (V.game w.θ w.F w.c) w.σ ≤ ε i
  rw [Game.prob_and_const]
  split_ifs with hle
  · exact (prob_unclean_le V D acc f εWipe hV hW w).trans hle
  · exact zero_le

/-- **The overhead guarantee at the verifier of record.** -/
theorem overhead : Overhead := overheadFor Accepts

/-- **The overhead guarantee needs the verifier.** At a window where claim `a` fails above the summed errors, on
`oneDeployment` (one server, one window, a run that succeeds exactly when that claim fails) every premise holds and
the conclusion fails for the verifier that accepts everything. -/
theorem overhead_nontrivial : Overhead_nontrivial := by
  classical
  intro ι d α _ V θ F c σ hreg a hlt
  refine ⟨soundness_of_subprotocols_nontrivial_of V θ F c σ hreg a hlt, oneTask, oneDeployment V a, 0, fun _ => 0,
    1 / 2, 0, 1, 0, fun _ => ((), 0, 0), fun _ => oneStrategy V ⟨θ, F, c, σ⟩, fun _ => ∑ b, V.error b θ F c σ,
    Nat.succ_pos 1, by norm_num, fun w => ?_, ?_, fun x ws => ?_, fun pos τ => ?_, ?_⟩
  · apply le_of_eq
    apply prob_eq_zero_of_not
    intro o h
    exact h.2.2 (oneWipe_survives 0 o)
  · intro x ws i e he _ hc _
    have hb : ¬ oneBad V a ws := by
      rintro ⟨e', he', hn⟩
      rw [Fin.val_eq_zero, he'] at he
      cases he
      exact hn (hc.1 a)
    have hJ := Deployment.inWindow_subset (oneDeployment V a) x ws i
    rw [oneDeployment_J_of_not_bad V a hb] at hJ
    rw [Finset.subset_empty.1 hJ]
    norm_num
  · have : (oneDeployment V a).dark x ws = ∅ := by
      apply Finset.filter_false_of_mem
      intro p _ h
      exact h 0 (inWindow_one V a x ws _)
    rw [this]
    simp
  · refine le_of_eq (prob_eq_zero' fun ω ws hw => ?_)
    obtain ⟨hs, hc⟩ := hw
    have hb : oneBad V a ws := of_decide_eq_true (eq_true_of_succeeds hs)
    rw [oneDeployment_J_of_bad V a hb] at hc
    simp at hc
  · rw [prob_one, prob_one]
    intro h
    have hle : Game.prob (fun o => ¬ V.Holds θ F c σ a o) (V.game θ F c) σ ≤ ∑ b, V.error b θ F c σ := by
      refine le_trans (prob_le_of_registered V hreg fun o hr ho => ?_) (h.trans ?_)
      · have hb : oneBad V a [((⟨θ, F, c, σ⟩ : Window V), o)] := ⟨((⟨θ, F, c, σ⟩ : Window V), o), rfl, ho⟩
        refine ⟨⟨rfl, fun e he => ?_⟩, ?_, ?_⟩
        · rw [List.mem_singleton] at he
          subst he
          exact ⟨trivial, hr⟩
        · rw [oneDeployment_run_of_bad V a hb]
          exact oneSucceeds
        · rw [oneMin V a]
          exact lt_of_le_of_lt (Nat.cast_le_one.2 (oneServerTime_le V a _ _)) one_lt_two
      · have hmid : ∀ o : Outcome, ¬ ∃ i : Fin 1, ∃ e, [((⟨θ, F, c, σ⟩ : Window V), o)][(i : ℕ)]? = some e ∧
            ∑ b, V.error b θ F c σ < windowError V (fun _ => 0) e.1 := by
          rintro o ⟨i, e, he, hlt'⟩
          rw [Fin.val_eq_zero, List.getElem?_cons_zero, Option.some.injEq] at he
          subst he
          simp [windowError] at hlt'
        rw [prob_eq_zero_of_not hmid]
        simp
    exact lt_irrefl _ (hle.trans_lt hlt)

end Verity
