import Definitions.Pous.P2.Cost

/-!
# A program cut to the ring budget, a state packed into a fixed width, and a sequential strategy's rounds

What the fixed extractor of a period's PoUS audit (`Pous.Period.extract`, `Properties.Pous.Period`) is built from:

* `Prog.clip g d D Q L P`: `P` with its output mapped by `g`, cut to the budget `(D, Q, L)`: a round that would take
  it past the budget ends it with `d` instead. It is `(D, Q, L)`-bounded whatever `P` is (`Prog.clip_ringBounded`),
  and wherever `P` stays within the budget it runs as `P` does, its output mapped by `g`.
* `StateBits.pack` and `StateBits.unpack`: a state of fewer than `n` bits as `n` bits (its bits, a `true`, then
  `false`s), and back.
* `seqRounds`: the state each round of a sequential strategy starts from and the program it runs, round after round.
-/

namespace Pous

variable {B w p : ℕ} [NeZero p]

namespace Prog

variable {α β : Type}

/-- `P` with its output mapped by `g`, cut to the budget `(D, Q, L)`: a round that would take it past `D` rounds with a
multiplication, `Q` key queries or `L` ring queries ends it with `d` instead. -/
def clip (g : α → β) (d : β) : ℕ → ℕ → ℕ → Prog (ringRO B w p) α → Prog (ringRO B w p) β
  | _, _, _, ret a => ret (g a)
  | D, Q, L, round qs k =>
    if (if qs.any RingQry.isMul then 1 else 0) ≤ D ∧ qs.countP RingQry.isKey ≤ Q ∧
        qs.countP (fun q => !q.isKey) ≤ L then
      round qs fun as => clip g d (D - if qs.any RingQry.isMul then 1 else 0) (Q - qs.countP RingQry.isKey)
        (L - qs.countP (fun q => !q.isKey)) (k as)
    else ret d

theorem clip_costs (g : α → β) (d : β) (f : (ringRO B w p).Qry → (ringRO B w p).Ans) (P : Prog (ringRO B w p) α) :
    ∀ D Q L, (clip g d D Q L P).mulDepth f ≤ D ∧ (clip g d D Q L P).keyWork f ≤ Q ∧
      (clip g d D Q L P).ringWork f ≤ L := by
  induction P with
  | ret a => intro D Q L; simp [clip, mulDepth, keyWork, ringWork]
  | round qs k ih =>
    intro D Q L
    by_cases h : (if qs.any RingQry.isMul then 1 else 0) ≤ D ∧ qs.countP RingQry.isKey ≤ Q ∧
        qs.countP (fun q => !q.isKey) ≤ L
    · rw [clip, ite_eq_left h]
      obtain ⟨h1, h2, h3⟩ := ih (qs.map f) (D - if qs.any RingQry.isMul then 1 else 0)
        (Q - qs.countP RingQry.isKey) (L - qs.countP (fun q => !q.isKey))
      simp only [mulDepth, keyWork, ringWork]
      omega
    · rw [clip, ite_eq_right h]
      simp [mulDepth, keyWork, ringWork]

/-- **A clipped program is `(D, Q, L)`-bounded**, whatever it clips. -/
theorem clip_ringBounded (g : α → β) (d : β) (D Q L : ℕ) (P : Prog (ringRO B w p) α) :
    (clip g d D Q L P).RingBounded D Q L :=
  fun _ => clip_costs g d _ P D Q L

end Prog

namespace StateBits

/-- A state of fewer than `n` bits as `n` bits: its bits, a `true`, then `false`s. -/
def pack {n : ℕ} (l : List Bool) : Bits n :=
  fun k => if h : k.val < l.length then l[k.val] else decide (k.val = l.length)

/-- The state `pack` holds: the bits before the last `true`. -/
def unpack {n : ℕ} (b : Bits n) : List Bool :=
  (List.ofFn b).take (Nat.findGreatest (fun k => ∃ h : k < n, b ⟨k, h⟩ = true) n)

end StateBits

/-- **The rounds of a sequential strategy** `step`, from round `j` in state `s`, for the indices `is`: the state each
round starts from and the program it runs, the next round starting from the state the program returns when its
queries are answered by `f`. -/
def seqRounds {M : IdealModel} {σ α ι : Type} (f : M.Qry → M.Ans) (step : ℕ → σ → ι → Prog M (α × σ)) :
    ℕ → List ι → σ → List (σ × Prog M (α × σ))
  | _, [], _ => []
  | j, i :: is, s => (s, step j s i) :: seqRounds f step (j + 1) is ((step j s i).run f).2

end Pous
