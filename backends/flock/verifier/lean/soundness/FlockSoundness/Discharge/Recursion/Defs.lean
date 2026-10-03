import FlockSoundness.Audit.Law

/-!
# Recursion: the model

A developer's circuit `P` stays hidden behind a registered hiding commitment `c_P`. The verifier draws `P`'s units to
prove, and the prover proves them with an **inner protocol** that the verifier never runs itself: in round `k` the prover
sends a hiding commitment `c_k` to its inner message `m_k`, and the verifier answers with a live coin `ρ_k`. No
Fiat–Shamir: each coin is drawn after its round's commitment is fixed. Then an ordinary `--zk` session of a fixed public
verifier circuit `V*` proves that openings of `c_P, c_1, …, c_r` exist on which the inner verifier, run on `P`'s statement
for the drawn units and the public coins, accepts.

* `rounds M C r f`: `r` rounds, each a message `m : M` and then a live coin `ρ : C`, continued by `f` on the transcript,
  first round first.
* `Inner`: the inner protocol, abstract: statements, messages, coins, `r` rounds, the verdict `accepts` on the statement
  and the transcript, and the statement's truth `holds`. Rounds of different message or coin types embed by taking sums
  of the messages and products of the coins.
* `InnerSound I ε`: on a false statement no prover strategy of `I.game` makes the inner verifier accept with probability
  above `ε`. A strategy of `I.game` fixes its round-`k` message before the coin `ρ_k` is drawn.
* `recGame`: the recursive audit: the registration, the draw, the `r` committed rounds, and the outer session, whose game
  is a parameter (`V*`'s live session in the Flock instance).
* `leafAvg`, `expAt`, `probAt`: averages over the round coins of quantities at the outer session's start, which may read
  the prover's strategy there (its state when the outer session begins).
* `relabel`, `refStrat`: the inner prover simulated from the recursive prover and a **reference opening** `ref` of each
  commitment: at the node where the recursive prover sends `c`, the inner prover sends `ref p c`, `p` the transcript
  before. `leafAvg_relabel` is the simulation: the probability that the inner verifier accepts the relabelled transcript
  is the probability that it accepts against `refStrat`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit
open scoped ENNReal

/-- **`r` committed rounds**: in each, the prover sends a message of type `M`, and then the verifier draws a live coin from
`C`, independently of everything before. The continuation reads the transcript, first round first. -/
def rounds (M C : Type) [Fintype C] [Nonempty C] {β : Type} : ℕ → (List (M × C) → Game β) → Game β
  | 0, f => f []
  | r + 1, f => .send M fun m => .coin C fun ρ => rounds M C r fun t => f ((m, ρ) :: t)

/-- **An inner protocol**: an interactive verifier with `r` rounds of public coins, messages committed before coins. In
each round the prover sends a message of type `Msg` and the verifier draws a uniform coin from `Coin`; after the last
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

/-- **`InnerSound I ε`**: on a false statement, every prover strategy makes the inner verifier accept with probability at
most `ε`. A strategy of `I.game` fixes each round's message before the round's coin is drawn, as a function of the
statement and the transcript before it; a randomized prover is a mixture of these. -/
def InnerSound (I : Inner) (ε : ℝ≥0∞) : Prop :=
  ∀ x, ¬ I.holds x → ∀ s : Strategy I.game, prob (I.accepts x) I.game s ≤ ε

/-- **The recursive audit**: the prover registers `R` (the hiding commitment `c_P` among it), the verifier draws `ω` (the
units of `P` to prove), then `I.r` rounds in which the prover sends a commitment `c_k : Cm` to its inner message and the
verifier draws the inner coin `ρ_k` live, and then the outer session `outer R ω t`, `t` the transcript of commitments and
coins. The output records all four. -/
def recGame {n : ℕ} (L : Law n) (Reg : Type) (I : Inner) (Cm : Type) {β : Type}
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

/-- **The transcript the simulated inner prover produces**: each commitment `c` replaced by its reference opening
`ref p c`, `p` the transcript before it. -/
def relabel {N : Type} : (List (M × C) → M → N) → List (M × C) → List (N × C)
  | _, [] => []
  | ref, (c, ρ) :: t => (ref [] c, ρ) :: relabel (fun p c' => ref ((c, ρ) :: p) c') t

/-- **The simulated inner prover**: it plays the recursive prover's strategy `s` on the rounds, and at the node where `s`
sends the commitment `c` it sends `ref p c`, `p` the transcript before. It fixes each message before the coin that
follows, since `s` does. -/
def refStrat {N δ : Type} : (r : ℕ) → (f : List (M × C) → Game β) → (List (M × C) → M → N) →
    (h : List (N × C) → δ) → Strategy (rounds M C r f) → Strategy (rounds N C r fun t => .ret (h t))
  | 0, _, _, _, _ => PUnit.unit
  | r + 1, f, ref, h, s => ⟨ref [] s.1, fun ρ =>
      refStrat r (fun t => f ((s.1, ρ) :: t)) (fun p c => ref ((s.1, ρ) :: p) c) (fun t => h ((ref [] s.1, ρ) :: t))
        (s.2 ρ)⟩

open Classical in
/-- **The simulation**: averaged over the round coins, the indicator that the relabelled transcript satisfies `Q` is the
probability that `Q` holds against the simulated inner prover. -/
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

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

/-- **The average of a quantity at the outer session's start** over the draw and the round coins: `F ω t s` reads the
prover's strategy `s` for the outer session after the draw `ω` and the transcript `t`. -/
noncomputable def expAt (σ : Strategy (recGame L Reg I Cm outer))
    (F : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → ℝ≥0∞) : ℝ≥0∞ :=
  avg fun ω => leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => F ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)) (σ.2 ω)

/-- **The probability of an event of the outer session** that may read the prover's strategy at its start (a decoded
witness, say): `expAt` of its probability. -/
noncomputable def probAt (σ : Strategy (recGame L Reg I Cm outer))
    (E : (ω : L.Ω) → (t : List (Cm × I.Coin)) → Strategy (outer σ.1 ω t) → β → Prop) : ℝ≥0∞ :=
  expAt σ fun ω t s => prob (E ω t s) (outer σ.1 ω t) s

/-- An event of the recursive audit's output is an event of the outer session. -/
theorem prob_recGame (p : Reg × L.Ω × List (Cm × I.Coin) × β → Prop) (σ : Strategy (recGame L Reg I Cm outer)) :
    prob p (recGame L Reg I Cm outer) σ = probAt σ fun ω t _ b => p (σ.1, ω, t, b) := by
  show avg (fun ω => prob p (rounds Cm I.Coin I.r _) (σ.2 ω)) = avg _
  refine congrArg avg (funext fun ω => ?_)
  rw [prob_rounds]
  exact congrArg (fun F => leafAvg I.r _ F (σ.2 ω)) (funext fun t => funext fun s => prob_map p _ _ s)

end Outer

end FlockSoundness.Discharge.Recursion
