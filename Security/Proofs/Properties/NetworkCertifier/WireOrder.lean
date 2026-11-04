import SampledProofs.WiresSpan

/-!
# The wire order's arithmetic: no carry

`SampledProofs.Wires.accum ps i` weights the digits of `i` in the radices of `ps`, fastest digit first: a wire's writer
unit is `base + accum` over the wire's radices and steps (`Wire.writerAt`), and a segment of the wire-order certificate
ranks a unit by `accum` over the segment's radices and weights. A number written in the radices with each digit below
its radix has those digits, so its `accum` is their weighted sum: when the certificate's check keeps each digit position
a wire moves within its radix, no digit carries, and each side's rank is affine in the wire's digits. The box's greatest
value of an affine form is `SampledProofs.Wires.accum_bounds`'.

* `horner`, `weighted`: the number with given digits in given radices, and the digits' weighted sum;
* `accum_horner`: `accum` of a number written with each digit below its radix is its digits weighted;
* `digits`, `accum_digits`, `horner_digits`, `digits_lt`: a number below the radices' product is written by its own
  digits, each below its radix, and `accum` weights them.
-/

namespace Properties.NetworkCertifier.WireOrderProof

open SampledProofs.Wires

/-- The number whose digits in the radices of `ps`, fastest first, are `es`. -/
def horner : List (ℕ × ℤ) → List ℕ → ℕ
  | [], _ => 0
  | _ :: _, [] => 0
  | (r, _) :: ps, e :: es => e + r * horner ps es

/-- The digits `es` weighted by the weights of `ps`. -/
def weighted : List (ℕ × ℤ) → List ℕ → ℤ
  | [], _ => 0
  | _ :: _, [] => 0
  | (_, w) :: ps, e :: es => (e : ℤ) * w + weighted ps es

/-- The digits of `i` in the radices of `ps`, fastest first. -/
def digits : List (ℕ × ℤ) → ℕ → List ℕ
  | [], _ => []
  | (r, _) :: ps, i => i % r :: digits ps (i / r)

/-- **No carry**: a number written in the radices of `ps` with each digit below its radix has those digits, so its
`accum` is its digits weighted. -/
theorem accum_horner {ps : List (ℕ × ℤ)} {es : List ℕ} (h : List.Forall₂ (fun p e => e < p.1) ps es) :
    accum ps (horner ps es) = weighted ps es := by
  induction h with
  | nil => rfl
  | @cons p e ps es he _ ih =>
    obtain ⟨r, _⟩ := p
    have he' : e < r := he
    have hr : 0 < r := Nat.lt_of_le_of_lt (Nat.zero_le e) he'
    simp only [accum, horner, weighted]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he', Nat.add_mul_div_left _ _ hr, Nat.div_eq_of_lt he',
      Nat.zero_add, ih]

/-- `accum` weights a number's digits. -/
theorem accum_digits : ∀ (ps : List (ℕ × ℤ)) (i : ℕ), accum ps i = weighted ps (digits ps i)
  | [], _ => rfl
  | (r, _) :: ps, i => by simp only [accum, digits, weighted, accum_digits ps (i / r)]

/-- A number below the radices' product is written by its digits. -/
theorem horner_digits : ∀ (ps : List (ℕ × ℤ)) (i : ℕ), i < prodR ps → horner ps (digits ps i) = i
  | [], i, hi => by
    simp only [prodR, List.foldr] at hi
    simp only [digits, horner]
    omega
  | (r, _) :: ps, i, hi => by
    have hi' : i < r * prodR ps := hi
    have hr : 0 < r := Nat.pos_of_ne_zero fun h => by simp [h] at hi'
    have hq : i / r < prodR ps := (Nat.div_lt_iff_lt_mul hr).2 (by rw [Nat.mul_comm]; exact hi')
    simp only [digits, horner, horner_digits ps (i / r) hq]
    exact Nat.mod_add_div i r

/-- A number below the radices' product has each digit below its radix. -/
theorem digits_lt : ∀ (ps : List (ℕ × ℤ)) (i : ℕ), i < prodR ps →
    List.Forall₂ (fun p e => e < p.1) ps (digits ps i)
  | [], _, _ => .nil
  | (r, _) :: ps, i, hi => by
    have hi' : i < r * prodR ps := hi
    have hr : 0 < r := Nat.pos_of_ne_zero fun h => by simp [h] at hi'
    have hq : i / r < prodR ps := (Nat.div_lt_iff_lt_mul hr).2 (by rw [Nat.mul_comm]; exact hi')
    simp only [digits]
    exact .cons (Nat.mod_lt i hr) (digits_lt ps (i / r) hq)

end Properties.NetworkCertifier.WireOrderProof
