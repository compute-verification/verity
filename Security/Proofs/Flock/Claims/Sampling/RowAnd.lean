import Definitions.Claims.Sampling
import Definitions.Flock.Circuit

/-!
# A lowered row gate's AND

The game circuit's row gate `row a b` is `(⊕ a) ∧ (⊕ b)` (`FlockSoundness.Audit.Op.apply`). Lowered to the claims
program it is three gates at `o`, `o + 1` and `o + 2`: an `Xor` of `a`'s wires, an `Xor` of `b`'s, and `And [o, o + 1]`.
Given the two `Xor` gates' values, `xorSum a v` and `xorSum b v`, the `And` gate's value is the row's.

* `BGate.val_and_row`: the `And` of two wires carrying the row's two sums is the row's value, for any reader.
* `Prog.runIn_and_row'`: the same in a set of gates `u` (`Prog.runIn`, what `Prog.Correct` reads), with each of the two
  wires either a gate of `u` before the `And` gate, at its computed value, or outside `u`, at its read value.
* `Prog.runIn_and_row`: the lowering's layout, the two `Xor` gates at `o` and `o + 1` in `u` and the `And` at `o + 2`.
* `Prog.runIn_and_row_seen`: the two sums as what `u` sees at any earlier wires `p` and `q` (computed in `u`, or read
  from outside it), the form an XOR chain's last wire gives (`XorChain.seen_out`).
* `foldl_xor_map`: the claims `Xor` gate's fold, `(l.map v).foldl Bool.xor false`, is `xorSum l v`; the `_foldl`
  variants take the two sums in that form.
-/

namespace Claims.Sampling

open FlockSoundness.Audit (Op xorSum)

/-- The claims `Xor` gate's fold is the game's `xorSum`. -/
theorem foldl_xor_map {ι : Type} (l : List ι) (v : ι → Bool) :
    (l.map v).foldl Bool.xor false = xorSum l v := by
  have h : ∀ acc, (l.map v).foldl Bool.xor acc = Bool.xor acc ((l.map v).foldr Bool.xor false) := by
    induction l with
    | nil => intro acc; simp
    | cons x t ih =>
        intro acc
        simp only [List.map_cons, List.foldl_cons, List.foldr_cons, ih, Bool.xor_assoc]
  rw [h, Bool.false_xor]
  rfl

/-- **The `And` of a row's two sums is the row**: if the wires `p` and `q` read `xorSum a v` and `xorSum b v`, the
gate `And [p, q]` has the value of `row a b` on `v`. -/
theorem BGate.val_and_row {N : ℕ} (a b : List (Fin N)) (v : Fin N → Bool) {rd : ℕ → Option Bool}
    (own : Option Bool) {p q : ℕ} (hp : rd p = some (xorSum a v)) (hq : rd q = some (xorSum b v)) :
    (BGate.and [p, q]).val rd own = some ((Op.row a b).apply v) := by
  simp [BGate.val, hp, hq, Op.apply]

/-- **A lowered row's `And` in a set of gates**: gate `g` of `Cp` is `And [p, q]`, and each of `p` and `q` is either a
gate of `u` before `g` whose value in `u` is its sum, or outside `u` with its sum as its read value; then `g`'s value
in `u` is the row's. -/
theorem Prog.runIn_and_row' {N : ℕ} (Cp : Prog) (ref : ℕ → Option Bool) (u : List ℕ) (a b : List (Fin N))
    (v : Fin N → Bool) {p q g : ℕ} (hg : Cp.gates[g]? = some (.and [p, q]))
    (hp : (p ∈ u ∧ p < g ∧ Cp.runIn ref u p = some (xorSum a v)) ∨ (p ∉ u ∧ Cp.read ref p = some (xorSum a v)))
    (hq : (q ∈ u ∧ q < g ∧ Cp.runIn ref u q = some (xorSum b v)) ∨ (q ∉ u ∧ Cp.read ref q = some (xorSum b v))) :
    Cp.runIn ref u g = some ((Op.row a b).apply v) := by
  rw [Prog.runIn, hg, Option.bind_some]
  refine BGate.val_and_row a b v _ ?_ ?_
  · rcases hp with ⟨h1, h2, h3⟩ | ⟨h1, h3⟩
    · simp only [h1, h2, ↓reduceIte, ↓reduceDIte, h3]
    · simp only [h1, ↓reduceIte, h3]
  · rcases hq with ⟨h1, h2, h3⟩ | ⟨h1, h3⟩
    · simp only [h1, h2, ↓reduceIte, ↓reduceDIte, h3]
    · simp only [h1, ↓reduceIte, h3]

/-- **A lowered row's `And`, at the lowering's layout**: the row's `Xor` gates at `o` and `o + 1` are gates of `u` with
values `xorSum a v` and `xorSum b v`, and gate `o + 2` is `And [o, o + 1]`; then gate `o + 2`'s value in `u` is the
row's. At `o = off C g` the `And` is at `posOf C g`. -/
theorem Prog.runIn_and_row {N : ℕ} (Cp : Prog) (ref : ℕ → Option Bool) (u : List ℕ) (a b : List (Fin N))
    (v : Fin N → Bool) {o : ℕ} (hg : Cp.gates[o + 2]? = some (.and [o, o + 1])) (h0 : o ∈ u) (h1 : o + 1 ∈ u)
    (hp : Cp.runIn ref u o = some (xorSum a v)) (hq : Cp.runIn ref u (o + 1) = some (xorSum b v)) :
    Cp.runIn ref u (o + 2) = some ((Op.row a b).apply v) :=
  Prog.runIn_and_row' Cp ref u a b v hg (.inl ⟨h0, by omega, hp⟩) (.inl ⟨h1, by omega, hq⟩)

/-- **A lowered row's `And`, reading what `u` sees**: gate `g` is `And [p, q]` with `p, q < g`, and on each of `p` and
`q` the set `u` sees its sum (its value computed in `u` for a gate of `u`, its read value otherwise; the XOR chain's
`XorChain.seen`); then `g`'s value in `u` is the row's. -/
theorem Prog.runIn_and_row_seen {N : ℕ} (Cp : Prog) (ref : ℕ → Option Bool) (u : List ℕ) (a b : List (Fin N))
    (v : Fin N → Bool) {p q g : ℕ} (hg : Cp.gates[g]? = some (.and [p, q])) (hpg : p < g) (hqg : q < g)
    (hp : (if p ∈ u then Cp.runIn ref u p else Cp.read ref p) = some (xorSum a v))
    (hq : (if q ∈ u then Cp.runIn ref u q else Cp.read ref q) = some (xorSum b v)) :
    Cp.runIn ref u g = some ((Op.row a b).apply v) := by
  refine Prog.runIn_and_row' Cp ref u a b v hg ?_ ?_
  · by_cases h : p ∈ u
    · exact .inl ⟨h, hpg, by simpa only [h, ↓reduceIte] using hp⟩
    · exact .inr ⟨h, by simpa only [h, ↓reduceIte] using hp⟩
  · by_cases h : q ∈ u
    · exact .inl ⟨h, hqg, by simpa only [h, ↓reduceIte] using hq⟩
    · exact .inr ⟨h, by simpa only [h, ↓reduceIte] using hq⟩

/-- `BGate.val_and_row`, the two sums as the claims `Xor` gate's fold. -/
theorem BGate.val_and_row_foldl {N : ℕ} (a b : List (Fin N)) (v : Fin N → Bool) {rd : ℕ → Option Bool}
    (own : Option Bool) {p q : ℕ} (hp : rd p = some ((a.map v).foldl Bool.xor false))
    (hq : rd q = some ((b.map v).foldl Bool.xor false)) :
    (BGate.and [p, q]).val rd own = some ((Op.row a b).apply v) :=
  BGate.val_and_row a b v own (hp.trans (by rw [foldl_xor_map])) (hq.trans (by rw [foldl_xor_map]))

/-- `Prog.runIn_and_row`, the two sums as the claims `Xor` gate's fold. -/
theorem Prog.runIn_and_row_foldl {N : ℕ} (Cp : Prog) (ref : ℕ → Option Bool) (u : List ℕ) (a b : List (Fin N))
    (v : Fin N → Bool) {o : ℕ} (hg : Cp.gates[o + 2]? = some (.and [o, o + 1])) (h0 : o ∈ u) (h1 : o + 1 ∈ u)
    (hp : Cp.runIn ref u o = some ((a.map v).foldl Bool.xor false))
    (hq : Cp.runIn ref u (o + 1) = some ((b.map v).foldl Bool.xor false)) :
    Cp.runIn ref u (o + 2) = some ((Op.row a b).apply v) :=
  Prog.runIn_and_row Cp ref u a b v hg h0 h1 (hp.trans (by rw [foldl_xor_map]))
    (hq.trans (by rw [foldl_xor_map]))

end Claims.Sampling
