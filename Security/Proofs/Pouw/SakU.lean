/-!
# SAK on literally uniform operands (SAK-U): the exact reduction, and why it credits no useful work

`SAKMatmul` (`Definitions.Pouw.Assumptions.Dither`) is symmetric algorithmic knowledge on the protocol's own noised
operands. SAK-U is its mildest form: the same statement on a product of literally uniform matrices. This file proves,
over an abstract priced machine, what a design that rests on SAK-U alone gets:

* `reduction`: a design whose committed words are a function of one instance `pad d c` (a bijection of the coins for
  each data `d`, e.g. one-time pads `A + R`, `B + S` over `ℤ/256`) reduces exactly to SAK-U. An adversary on data `d₀`
  gives a SAK-U solver that succeeds on exactly the image of the adversary's successes under `pad d₀`, at the
  adversary's cost plus the cost of `unpad d₀`. The advice is the adversary's, plus `d₀`, and both are fixed before
  the coins.
* `honest_pays_both`: if SAK-U holds for the direct sum of the words' instance and an independent uniform instance of
  the useful product (`SakUSum`), the honest program pays for both, `pubW + pubO`, less the cost of `unpad`.
* `gap_of_sakUSum`: so the public best for the words alone, which outputs every committed word, undercuts the honest
  program by at least `pubO` less the two maps' costs. `pubO` is the public best for a product of uniform operands,
  stock's priced cost: the certified T1 gap of a SAK-U-only design is about 100% of stock.
* `denoise_is_useful`: premise-free. If the honest program computes the words and then de-noises with `E`, then `E`
  run at the coins that make the instance a fixed `u₀` computes the useful output for every data. So `E` costs at
  least the cheapest useful-product program, less the cost of the coin map.

Costs are data-oblivious, one number a program, as a fixed kernel's priced work is. The machine is abstract: `pre`
composes a program with a map, at `mapCost` more. The probability reading of `reduction` is the bijection: uniform
coins map to a uniform instance, so the two success probabilities are equal.
-/

namespace Pouw.SakU

/-- An abstract priced machine: programs, their evaluation, their data-oblivious priced cost, and precomposition with
a map at most `mapCost` of that map more. -/
structure Machine where
  Prog : Type → Type → Type
  eval : {α β : Type} → Prog α β → α → β
  cost : {α β : Type} → Prog α β → Nat
  mapCost : {α α' : Type} → (α' → α) → Nat
  pre : {α α' β : Type} → Prog α β → (α' → α) → Prog α' β
  eval_pre : ∀ {α α' β : Type} (p : Prog α β) (f : α' → α) (x : α'), eval (pre p f) x = eval p (f x)
  cost_pre : ∀ {α α' β : Type} (p : Prog α β) (f : α' → α), cost (pre p f) ≤ cost p + mapCost f

/-- A design whose committed words are `words (pad d c)`: a function of one instance, a bijection of the coins `c` for
each data `d` (`unpad d` its inverse), and whose served output is the useful `useful d`. -/
structure Design where
  Data : Type
  Coin : Type
  Inst : Type
  Words : Type
  Out : Type
  pad : Data → Coin → Inst
  unpad : Data → Inst → Coin
  unpad_pad : ∀ d c, unpad d (pad d c) = c
  pad_unpad : ∀ d u, pad d (unpad d u) = u
  words : Inst → Words
  useful : Data → Out

variable {M : Machine} {D : Design}

/-- The SAK-U solver an adversary on data `d₀` gives: undo the pad, then run the adversary. -/
def solver (M : Machine) (D : Design) (adv : M.Prog (D.Data × D.Coin) D.Words) (d₀ : D.Data) :
    M.Prog D.Inst D.Words :=
  M.pre adv (fun u => (d₀, D.unpad d₀ u))

/-- **The exact reduction.** The solver is right on the instance `pad d₀ c` exactly when the adversary is right on the
coins `c`, and it costs at most the adversary's cost plus the cost of `unpad d₀`. -/
theorem reduction (adv : M.Prog (D.Data × D.Coin) D.Words) (d₀ : D.Data) :
    (∀ c, M.eval (solver M D adv d₀) (D.pad d₀ c) = D.words (D.pad d₀ c) ↔
      M.eval adv (d₀, c) = D.words (D.pad d₀ c)) ∧
    M.cost (solver M D adv d₀) ≤ M.cost adv + M.mapCost (fun u : D.Inst => (d₀, D.unpad d₀ u)) := by
  refine ⟨fun c => ?_, M.cost_pre _ _⟩
  simp only [solver, M.eval_pre, D.unpad_pad]

/-- The solver succeeds on every instance exactly where the adversary succeeds on its preimage. -/
theorem reduction_inst (adv : M.Prog (D.Data × D.Coin) D.Words) (d₀ : D.Data) (u : D.Inst) :
    M.eval (solver M D adv d₀) u = D.words u ↔ M.eval adv (d₀, D.unpad d₀ u) = D.words (D.pad d₀ (D.unpad d₀ u)) := by
  simp only [solver, M.eval_pre, D.pad_unpad]

/-- **SAK-U for a direct sum**, in its always-right reading: every program that outputs, on every pair of an instance
`u` and data `d`, the words of `u` and the useful output of `d`, costs at least `pubW + pubO`. Read with `u` and `d`
independent and uniform, it is SAK-U for two independent products of literally uniform operands. Quantifying only over
always-right programs makes it weaker than the average-case premise. -/
def SakUSum (M : Machine) (D : Design) (pubW pubO : Nat) : Prop :=
  ∀ Q : M.Prog (D.Inst × D.Data) (D.Words × D.Out),
    (∀ u d, M.eval Q (u, d) = (D.words u, D.useful d)) → pubW + pubO ≤ M.cost Q

/-- **The honest program pays for both products.** An honest program that outputs the words and the useful output on
every data and coins is, after undoing the pad, a direct-sum solver. -/
theorem honest_pays_both (H : M.Prog (D.Data × D.Coin) (D.Words × D.Out))
    (hH : ∀ d c, M.eval H (d, c) = (D.words (D.pad d c), D.useful d)) {pubW pubO : Nat} (hS : SakUSum M D pubW pubO) :
    pubW + pubO ≤ M.cost H + M.mapCost (fun x : D.Inst × D.Data => (x.2, D.unpad x.2 x.1)) := by
  have hQ : ∀ u d, M.eval (M.pre H (fun x : D.Inst × D.Data => (x.2, D.unpad x.2 x.1))) (u, d) =
      (D.words u, D.useful d) := by
    intro u d
    rw [M.eval_pre, hH, D.pad_unpad]
  exact Nat.le_trans (hS _ hQ) (M.cost_pre _ _)

/-- **The certified gap of a SAK-U-only design.** The public best for the words (`pubProg`, cost `pubW`), run after
the pad, outputs every committed word. Under `SakUSum` the honest program costs at least `pubO` more than it, less the
two maps' costs. -/
theorem gap_of_sakUSum (H : M.Prog (D.Data × D.Coin) (D.Words × D.Out))
    (hH : ∀ d c, M.eval H (d, c) = (D.words (D.pad d c), D.useful d)) {pubW pubO : Nat} (hS : SakUSum M D pubW pubO)
    (pubProg : M.Prog D.Inst D.Words) (hpub : ∀ u, M.eval pubProg u = D.words u) (hcost : M.cost pubProg ≤ pubW) :
    (∀ d c, M.eval (M.pre pubProg (fun x : D.Data × D.Coin => D.pad x.1 x.2)) (d, c) = D.words (D.pad d c)) ∧
    M.cost (M.pre pubProg (fun x : D.Data × D.Coin => D.pad x.1 x.2)) + pubO ≤
      M.cost H + M.mapCost (fun x : D.Inst × D.Data => (x.2, D.unpad x.2 x.1)) +
        M.mapCost (fun x : D.Data × D.Coin => D.pad x.1 x.2) := by
  refine ⟨fun d c => by rw [M.eval_pre, hpub], ?_⟩
  have h1 := honest_pays_both H hH hS
  have h2 := M.cost_pre pubProg (fun x : D.Data × D.Coin => D.pad x.1 x.2)
  omega

/-- **The de-noise alone computes the useful output** (premise-free). If `E` turns the data, the coins and the words
into the useful output on every data and coins, then `E` at the coins `unpad d u₀`, where the words are the fixed
`words u₀`, computes the useful output on every data, at `E`'s cost plus the coin map's. Choosing `u₀` with words that
cost nothing to know (the zero instance, whose prefix sums are all zero) makes `E` a program for the useful product. -/
theorem denoise_is_useful (E : M.Prog (D.Data × D.Coin × D.Words) D.Out)
    (hE : ∀ d c, M.eval E (d, c, D.words (D.pad d c)) = D.useful d) (u₀ : D.Inst) :
    (∀ d, M.eval (M.pre E (fun d => (d, D.unpad d u₀, D.words u₀))) d = D.useful d) ∧
    M.cost (M.pre E (fun d => (d, D.unpad d u₀, D.words u₀))) ≤
      M.cost E + M.mapCost (fun d : D.Data => (d, D.unpad d u₀, D.words u₀)) := by
  refine ⟨fun d => ?_, M.cost_pre _ _⟩
  have h := hE d (D.unpad d u₀)
  rw [D.pad_unpad] at h
  rw [M.eval_pre]
  exact h

/-- The de-noise's cost bound: at least the cheapest program for the useful output, `pubUseful`, less the coin map's
cost. -/
theorem denoise_cost (E : M.Prog (D.Data × D.Coin × D.Words) D.Out)
    (hE : ∀ d c, M.eval E (d, c, D.words (D.pad d c)) = D.useful d) (u₀ : D.Inst) {pubUseful : Nat}
    (hmin : ∀ P : M.Prog D.Data D.Out, (∀ d, M.eval P d = D.useful d) → pubUseful ≤ M.cost P) :
    pubUseful ≤ M.cost E + M.mapCost (fun d : D.Data => (d, D.unpad d u₀, D.words u₀)) := by
  obtain ⟨h1, h2⟩ := denoise_is_useful E hE u₀
  exact Nat.le_trans (hmin _ h1) h2

end Pouw.SakU
