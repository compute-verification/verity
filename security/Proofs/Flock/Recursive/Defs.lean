module

public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-! The recursion's model: the inner protocol and its soundness, in fork form too, the recursive audit's game and the simulated inner prover. -/

@[expose] public section

namespace FlockSoundness.Game

variable {α : Type}

/-- **An outcome the strategy reaches**: `s` plays `g` to `a` at some coins. -/
def Outcome : (g : Game α) → Strategy g → α → Prop
  | .ret b, _, a => b = a
  | .send _ k, s, a => Outcome (k s.1) s.2 a
  | @Game.coin _ _ _ _ k, s, a => ∃ c, Outcome (k c) (s c) a

theorem Outcome.of_bind {β : Type} : ∀ (g : Game α) (f : α → Game β) (s : Strategy (g.bind f)) {b : β},
    Outcome (g.bind f) s b → ∃ a, Outcome g (Strategy.ofBind f g s) a ∧ ∃ u : Strategy (f a), Outcome (f a) u b
  | .ret a, _, s, _, h => ⟨a, rfl, s, h⟩
  | .send _ k, f, s, _, h => Outcome.of_bind (k s.1) f s.2 h
  | @Game.coin _ _ _ _ k, f, s, _, h => by
      obtain ⟨c, hc⟩ := h
      obtain ⟨a, ha, u, hu⟩ := Outcome.of_bind (k c) f (s c) hc
      exact ⟨a, ⟨c, ha⟩, u, hu⟩

theorem Outcome.of_map {β : Type} (f : α → β) : ∀ (g : Game α) (s : Strategy (g.map f)) {b : β},
    Outcome (g.map f) s b → ∃ a, Outcome g (Strategy.ofMap f g s) a ∧ f a = b
  | .ret a, _, _, h => ⟨a, rfl, h⟩
  | .send _ k, s, _, h => Outcome.of_map f (k s.1) s.2 h
  | @Game.coin _ _ _ _ k, s, _, h => by
      obtain ⟨c, hc⟩ := h
      obtain ⟨a, ha, hb⟩ := Outcome.of_map f (k c) (s c) hc
      exact ⟨a, ⟨c, ha⟩, hb⟩

theorem prob_mono_outcome {p q : α → Prop} :
    ∀ (g : Game α) (s : Strategy g), (∀ a, Outcome g s a → p a → q a) → prob p g s ≤ prob q g s
  | .ret a, _, h => by
      classical
      simp only [prob]
      by_cases hp : p a
      · simp [hp, h a rfl hp]
      · simp [hp]
  | .send _ k, s, h => prob_mono_outcome (k s.1) s.2 h
  | @Game.coin _ C _ _ k, s, h =>
      Finset.sum_le_sum fun c _ => mul_le_mul_right (prob_mono_outcome (k c) (s c) fun a ha => h a ⟨c, ha⟩) _

end FlockSoundness.Game

namespace FlockSoundness.StrictCR

open Game

variable {D β : Type} (H : List UInt8 → D)

/-- **A run of `g` against `s` finds a collision of `H`**: some outcome `s` reaches outputs, by `out`, two different
strings that `H` maps to one digest. -/
def FindsColl (g : Game β) (s : Strategy g) (out : β → Pair) : Prop :=
  ∃ b, Game.Outcome g s b ∧ IsColl H (out b)

/-- **A finder finds a collision of `H`** at an outcome its strategy reaches. -/
def Finder.Finds (Fd : Finder Pair) : Prop := FindsColl H Fd.g Fd.s id

end FlockSoundness.StrictCR

namespace FlockSoundness.Discharge.Recursion

open Game Audit
open scoped ENNReal

/-- **`r` committed rounds**: in each, the prover sends a message of type `M`, and then the verifier flips a live coin from
`C`, independently of everything before. The continuation reads the transcript, first round first. -/
def rounds (M C : Type) [Fintype C] [Nonempty C] {β : Type} : ℕ → (List (M × C) → Game β) → Game β
  | 0, f => f []
  | r + 1, f => .send M fun m => .coin C fun ρ => rounds M C r fun t => f ((m, ρ) :: t)

/-- **An inner protocol**: an interactive verifier with `r` rounds of public coins, messages committed before coins. In
each round the prover sends a message of type `Msg` and the verifier flips a uniform coin from `Coin`; after the last
round it decides `accepts x t` from the statement `x` and the transcript `t`, first round first. `holds x` is the
statement's truth. -/
structure Inner : Type 1 where
  Stmt : Type
  Msg : Type
  Coin : Type
  [fintype : Fintype Coin]
  [nonempty : Nonempty Coin]
  r : ℕ
  accepts : Stmt → List (Msg × Coin) → Prop
  holds : Stmt → Prop

attribute [instance] Inner.fintype Inner.nonempty

/-- The inner verifier's side of the interaction, its output the transcript. -/
def Inner.game (I : Inner) : Game (List (I.Msg × I.Coin)) := rounds I.Msg I.Coin I.r .ret

section Fork

variable {M C β γ : Type} [Fintype C] [Nonempty C]

/-- **The rest of the run from a node**: the remaining round coins, the prover's messages read from its strategy, and then
the finder `T` at the leaf reached. -/
noncomputable def contF : (r : ℕ) → (f : List (M × C) → Game β) →
    ((t : List (M × C)) → Strategy (f t) → StrictCR.Finder γ) → Strategy (rounds M C r f) → StrictCR.Finder γ
  | 0, _, T, s => T [] s
  | r + 1, f, T, s =>
      ⟨.coin C fun ρ => (contF r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).g,
        fun ρ => (contF r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).s⟩

/-- **The fork at round `k`**: the round coins to the node where the prover sends its round-`k` message, then the rest of
the run twice, independently (`contF`, the trial `T` at each leaf), the two trials' outputs read by `pick`. With no round
`k` it finds nothing. -/
noncomputable def forkG (pick : γ → γ → StrictCR.Pair) : (k r : ℕ) → (f : List (M × C) → Game β) →
    ((t : List (M × C)) → Strategy (f t) → StrictCR.Finder γ) → Strategy (rounds M C r f) →
    StrictCR.Finder StrictCR.Pair
  | _, 0, _, _, _ => ⟨.ret none, PUnit.unit⟩
  | 0, r + 1, f, T, s => StrictCR.twoRuns (contF (r + 1) f T s).g (contF (r + 1) f T s).s pick
  | k + 1, r + 1, f, T, s =>
      ⟨.coin C fun ρ => (forkG pick k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).g,
        fun ρ => (forkG pick k r (fun t => f ((s.1, ρ) :: t)) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)).s⟩

end Fork

/-- **`InnerSound I ε`**: on a false statement, every prover strategy makes the inner verifier accept with probability at
most `ε`. A strategy of `I.game` fixes each round's message before the round's coin is flipped, as a function of the
statement and the transcript before it; a randomized prover is a mixture of these. -/
def InnerSound (I : Inner) (ε : ℝ≥0∞) : Prop :=
  ∀ x, ¬ I.holds x → ∀ s : Strategy I.game, prob (I.accepts x) I.game s ≤ ε

def Inner.Fork {D : Type} (H : List UInt8 → D) (I : Inner)
    (pick : List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) (s : Strategy I.game) : Prop :=
  ∃ k < I.r, (forkG pick k I.r .ret (fun t _ => ⟨.ret t, PUnit.unit⟩) s).Finds H

/-- The bound is exact for a prover none of whose forks `pick` reads as a collision; for one that has such a fork it says
nothing. -/
def InnerSoundFork {D : Type} (H : List UInt8 → D) (I : Inner)
    (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) (ε : ℝ≥0∞) : Prop :=
  ∀ x, ¬ I.holds x → ∀ s : Strategy I.game, I.Fork H (pick x) s ∨ prob (I.accepts x) I.game s ≤ ε

open Classical in
noncomputable def pickOn {τ : Type} (G : τ → Prop) (pick : τ → τ → StrictCR.Pair) (t t' : τ) : StrictCR.Pair :=
  if G t ∧ G t' then pick t t' else none

def InnerSoundForkOn {D : Type} (H : List UInt8 → D) (I : Inner)
    (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) (ε : ℝ≥0∞) : Prop :=
  ∀ x, ¬ I.holds x → ∀ (G : List (I.Msg × I.Coin) → Prop) (s : Strategy I.game),
    I.Fork H (pickOn G (pick x)) s ∨ prob (fun t => I.accepts x t ∧ G t) I.game s ≤ ε

theorem InnerSoundForkOn.plain {D : Type} {H : List UInt8 → D} {I : Inner}
    {pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair} {ε : ℝ≥0∞}
    (h : InnerSoundForkOn H I pick ε) : InnerSoundFork H I pick ε := by
  intro x hx s
  rcases h x hx (fun _ => True) s with hf | hp
  · refine Or.inl ?_
    have hpick : pickOn (fun _ => True) (pick x) = pick x := funext fun t => funext fun t' => by simp [pickOn]
    rwa [hpick] at hf
  · exact Or.inr ((prob_mono (fun _ h => ⟨h, trivial⟩) _ s).trans hp)

/-- **The recursive audit**: the prover registers `R` (the hiding commitment `c_P` among it), the verifier samples `ω` (the
units of `P` to prove), then `I.r` rounds in which the prover sends a commitment `c_k : Cm` to its inner message and the
verifier flips the inner coin `ρ_k` live, and then the outer session `outer R ω t`, `t` the transcript of commitments and
coins. The output records all four. -/
def recGame {n : ℕ} (L : SamplingDesign n) (Reg : Type) (I : Inner) (Cm : Type) {β : Type}
    (outer : Reg → L.Ω → List (Cm × I.Coin) → Game β) : Game (Reg × L.Ω × List (Cm × I.Coin) × β) :=
  .send Reg fun R => .coin L.Ω fun ω => rounds Cm I.Coin I.r fun t => (outer R ω t).map fun b => (R, ω, t, b)

section Leaves

variable {M C β γ : Type} [Fintype C] [Nonempty C]

/-- **The average over the round coins of a quantity at the leaves**, the prover's messages read from its strategy: at the
leaf the transcript `t` reaches, `F t` reads the strategy there. -/
noncomputable def leafAvg : (r : ℕ) → (f : List (M × C) → Game β) →
    ((t : List (M × C)) → Strategy (f t) → ℝ≥0∞) → Strategy (rounds M C r f) → ℝ≥0∞
  | 0, _, F, s => F [] s
  | r + 1, f, F, s => avg fun ρ => leafAvg r (fun t => f ((s.1, ρ) :: t)) (fun t => F ((s.1, ρ) :: t)) (s.2 ρ)

theorem prob_rounds (p : β → Prop) : ∀ (r : ℕ) (f : List (M × C) → Game β) (s : Strategy (rounds M C r f)),
    prob p (rounds M C r f) s = leafAvg r f (fun t s' => prob p (f t) s') s
  | 0, _, _ => rfl
  | r + 1, f, s => congrArg avg (funext fun ρ => prob_rounds p r (fun t => f ((s.1, ρ) :: t)) (s.2 ρ))

theorem leafAvg_congr_len {r : ℕ} {f : List (M × C) → Game β} {F G : (t : List (M × C)) → Strategy (f t) → ℝ≥0∞}
    (h : ∀ t s', t.length = r → F t s' = G t s') (s : Strategy (rounds M C r f)) :
    leafAvg r f F s = leafAvg r f G s := by
  induction r generalizing f with
  | zero => exact h [] s rfl
  | succ r ih =>
      exact congrArg avg (funext fun ρ => ih (fun t s' ht => h _ s' (by simp [ht])) (s.2 ρ))

theorem leafAvg_mono_len {r : ℕ} {f : List (M × C) → Game β} {F G : (t : List (M × C)) → Strategy (f t) → ℝ≥0∞}
    (h : ∀ t s', t.length = r → F t s' ≤ G t s') (s : Strategy (rounds M C r f)) :
    leafAvg r f F s ≤ leafAvg r f G s := by
  induction r generalizing f with
  | zero => exact h [] s rfl
  | succ r ih => exact avg_mono fun ρ => ih (fun t s' ht => h _ s' (by simp [ht])) (s.2 ρ)

theorem leafAvg_mono {r : ℕ} {f : List (M × C) → Game β} {F G : (t : List (M × C)) → Strategy (f t) → ℝ≥0∞}
    (h : ∀ t s', F t s' ≤ G t s') (s : Strategy (rounds M C r f)) : leafAvg r f F s ≤ leafAvg r f G s :=
  leafAvg_mono_len (fun t s' _ => h t s') s

theorem leafAvg_add {r : ℕ} {f : List (M × C) → Game β} (F G : (t : List (M × C)) → Strategy (f t) → ℝ≥0∞)
    (s : Strategy (rounds M C r f)) :
    leafAvg r f (fun t s' => F t s' + G t s') s = leafAvg r f F s + leafAvg r f G s := by
  induction r generalizing f with
  | zero => rfl
  | succ r ih =>
      show avg _ = avg _ + avg _
      rw [← avg_add]
      exact congrArg avg (funext fun ρ => ih _ _ (s.2 ρ))

theorem leafAvg_const {r : ℕ} {f : List (M × C) → Game β} (x : ℝ≥0∞) (s : Strategy (rounds M C r f)) :
    leafAvg r f (fun _ _ => x) s = x := by
  induction r generalizing f with
  | zero => rfl
  | succ r ih =>
      show avg _ = x
      rw [show (fun ρ => leafAvg r (fun t => f ((s.1, ρ) :: t)) (fun _ _ => x) (s.2 ρ)) = fun _ => x from
        funext fun ρ => ih (s.2 ρ)]
      exact avg_const x

theorem leafAvg_const_mul {r : ℕ} {f : List (M × C) → Game β} (x : ℝ≥0∞)
    (F : (t : List (M × C)) → Strategy (f t) → ℝ≥0∞) (s : Strategy (rounds M C r f)) :
    leafAvg r f (fun t s' => x * F t s') s = x * leafAvg r f F s := by
  induction r generalizing f with
  | zero => rfl
  | succ r ih =>
      show avg _ = x * avg _
      rw [show (fun ρ => leafAvg r (fun t => f ((s.1, ρ) :: t)) (fun t s' => x * F ((s.1, ρ) :: t) s') (s.2 ρ)) =
        fun ρ => leafAvg r (fun t => f ((s.1, ρ) :: t)) (fun t => F ((s.1, ρ) :: t)) (s.2 ρ) * x from
        funext fun ρ => (ih _ (s.2 ρ)).trans (mul_comm _ _), avg_mul_const, mul_comm]

theorem avg_finset_sum {D ι : Type} [Fintype D] (S : Finset ι) (f : ι → D → ℝ≥0∞) :
    avg (fun d => ∑ i ∈ S, f i d) = ∑ i ∈ S, avg (f i) := by
  unfold avg
  simp only [Finset.mul_sum]
  exact Finset.sum_comm

theorem leafAvg_sum {ι : Type} (S : Finset ι) {r : ℕ} {f : List (M × C) → Game β}
    (F : ι → (t : List (M × C)) → Strategy (f t) → ℝ≥0∞) (s : Strategy (rounds M C r f)) :
    leafAvg r f (fun t s' => ∑ i ∈ S, F i t s') s = ∑ i ∈ S, leafAvg r f (F i) s := by
  induction r generalizing f with
  | zero => rfl
  | succ r ih =>
      show avg _ = ∑ i ∈ S, avg _
      rw [← avg_finset_sum]
      exact congrArg avg (funext fun ρ => ih _ (s.2 ρ))

theorem leafAvg_le_one {r : ℕ} {f : List (M × C) → Game β} {F : (t : List (M × C)) → Strategy (f t) → ℝ≥0∞}
    (h : ∀ t s', F t s' ≤ 1) (s : Strategy (rounds M C r f)) : leafAvg r f F s ≤ 1 :=
  (leafAvg_mono h s).trans (leafAvg_const 1 s).le

/-- A hypothesis that holds only of the prover's own strategies (`cr/sha-512` for finders built from them) is stated
this way: a strategy no transcript reaches may hard-code anything. -/
def AllLeaves : (r : ℕ) → (f : List (M × C) → Game β) → ((t : List (M × C)) → Strategy (f t) → Prop) →
    Strategy (rounds M C r f) → Prop
  | 0, _, P, s => P [] s
  | r + 1, f, P, s => ∀ ρ, AllLeaves r (fun t => f ((s.1, ρ) :: t)) (fun t => P ((s.1, ρ) :: t)) (s.2 ρ)

theorem AllLeaves.mono {r : ℕ} {f : List (M × C) → Game β} {P Q : (t : List (M × C)) → Strategy (f t) → Prop}
    {s : Strategy (rounds M C r f)} (h : AllLeaves r f P s) (hPQ : ∀ t s', t.length = r → P t s' → Q t s') :
    AllLeaves r f Q s := by
  induction r generalizing f with
  | zero => exact hPQ [] s rfl h
  | succ r ih => exact fun ρ => ih (h ρ) fun t s' ht => hPQ _ s' (by simp [ht])

theorem leafAvg_mono_all {r : ℕ} {f : List (M × C) → Game β} {F G : (t : List (M × C)) → Strategy (f t) → ℝ≥0∞}
    {s : Strategy (rounds M C r f)} (h : AllLeaves r f (fun t s' => F t s' ≤ G t s') s) :
    leafAvg r f F s ≤ leafAvg r f G s := by
  induction r generalizing f with
  | zero => exact h
  | succ r ih => exact avg_mono fun ρ => ih (h ρ)

def relabel {N : Type} : (List (M × C) → M → N) → List (M × C) → List (N × C)
  | _, [] => []
  | ref, (c, ρ) :: t => (ref [] c, ρ) :: relabel (fun p c' => ref ((c, ρ) :: p) c') t

/-- It fixes each message before the coin that follows, since `s` does. -/
def refStrat {N δ : Type} : (r : ℕ) → (f : List (M × C) → Game β) → (List (M × C) → M → N) →
    (h : List (N × C) → δ) → Strategy (rounds M C r f) → Strategy (rounds N C r fun t => .ret (h t))
  | 0, _, _, _, _ => PUnit.unit
  | r + 1, f, ref, h, s => ⟨ref [] s.1, fun ρ =>
      refStrat r (fun t => f ((s.1, ρ) :: t)) (fun p c => ref ((s.1, ρ) :: p) c) (fun t => h ((ref [] s.1, ρ) :: t))
        (s.2 ρ)⟩

open Classical in
theorem leafAvg_relabel {N δ : Type} (Q : δ → Prop) : ∀ (r : ℕ) (f : List (M × C) → Game β)
    (ref : List (M × C) → M → N) (h : List (N × C) → δ) (s : Strategy (rounds M C r f)),
    leafAvg r f (fun t _ => if Q (h (relabel ref t)) then 1 else 0) s =
      prob Q (rounds N C r fun t => .ret (h t)) (refStrat r f ref h s)
  | 0, _, _, _, _ => rfl
  | r + 1, f, ref, h, s =>
      congrArg avg (funext fun ρ => leafAvg_relabel Q r (fun t => f ((s.1, ρ) :: t))
        (fun p c => ref ((s.1, ρ) :: p) c) (fun t => h ((ref [] s.1, ρ) :: t)) (s.2 ρ))

omit [Fintype C] [Nonempty C] in
theorem zip_eq_relabel {N : Type} : ∀ (ref : List (M × C) → M → N) (t : List (M × C)) (w : List N),
    (∀ k < t.length, w[k]? = (t[k]?).map fun e => ref (t.take k) e.1) → w.zip (t.map Prod.snd) = relabel ref t
  | _, [], _, _ => by simp [relabel]
  | ref, (c, ρ) :: t, w, h => by
      obtain ⟨a, w', rfl⟩ : ∃ a w', w = a :: w' := by
        cases w with
        | nil => simpa using h 0 (by simp)
        | cons a w' => exact ⟨a, w', rfl⟩
      have h0 : a = ref [] c := by simpa using h 0 (by simp)
      have ht : ∀ k < t.length, w'[k]? = (t[k]?).map fun e => (fun p c' => ref ((c, ρ) :: p) c') (t.take k) e.1 :=
        fun k hk => by simpa using h (k + 1) (by simp [hk])
      simp only [List.map_cons, List.zip_cons_cons, relabel, h0,
        zip_eq_relabel (fun p c' => ref ((c, ρ) :: p) c') t w' ht]

end Leaves

section Outer

variable {n : ℕ} {L : SamplingDesign n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

/-- **The average of a quantity at the outer session's start** over the sample and the round coins: `F ω t s` reads the
prover's strategy `s` for the outer session after the sample `ω` and the transcript `t`. -/
noncomputable def expAt (σ : Strategy (recGame L Reg I Cm outer))
    (F : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → ℝ≥0∞) : ℝ≥0∞ :=
  avg fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => F ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)

def AllAt (σ : Strategy (recGame L Reg I Cm outer))
    (P : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → Prop) : Prop :=
  ∀ ω, AllLeaves I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => P ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)

noncomputable def probAt (σ : Strategy (recGame L Reg I Cm outer))
    (E : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → β → Prop) : ℝ≥0∞ :=
  expAt σ fun ω t s => prob (E ω t s) (outer σ.1 ω t) s

theorem prob_recGame (p : Reg × L.Ω × List (Cm × I.Coin) × β → Prop) (σ : Strategy (recGame L Reg I Cm outer)) :
    prob p (recGame L Reg I Cm outer) σ = probAt σ fun ω t _ b => p (σ.1, ω, t, b) := by
  show avg (fun ω => prob p (rounds Cm I.Coin I.r _) (σ.2 ω)) = avg _
  refine congrArg avg (funext fun ω => ?_)
  rw [prob_rounds]
  exact congrArg (fun F => leafAvg I.r _ F (σ.2 ω)) (funext fun t => funext fun s => prob_map p _ _ s)

end Outer

end FlockSoundness.Discharge.Recursion
