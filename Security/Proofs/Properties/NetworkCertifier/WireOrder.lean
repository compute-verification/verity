import SampledProofs.WiresSpan

/-!
# The wire order's arithmetic: no carry

`SampledProofs.Wires.accum ps i` weights the digits of `i` in the radices of `ps`, fastest digit first: a wire's writer
unit is `base + accum` over the wire's radices and steps (`Wire.writerAt`), and a segment of the wire-order certificate
ranks a unit by `accum` over the segment's radices and weights. A number written in the radices with each digit below
its radix has those digits, so its `accum` is their weighted sum: when the certificate's check keeps each digit position
a wire moves within its radix, no digit carries, and each side's rank is affine in the wire's digits. The box's greatest
value of an affine form is `SampledProofs.Wires.accum_bounds`'.

* `horner`, `weighted`, `Below`: the number with given digits in given radices, the digits' weighted sum, and each digit
  below its radix;
* `accum_horner`: `accum` of a number written with each digit below its radix is its digits weighted;
* `digits`, `accum_digits`, `horner_digits`, `digits_lt`: a number below the radices' product is written by its own
  digits, each below its radix, and `accum` weights them.
-/

namespace Properties.NetworkCertifier.WireOrderProof

open SampledProofs.Wires

/-- The number whose digits in the radices of `ps`, fastest first, are `es`. -/
def horner : List (Nat × Int) → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | (r, _) :: ps, e :: es => e + r * horner ps es

/-- The digits `es` weighted by the weights of `ps`. -/
def weighted : List (Nat × Int) → List Nat → Int
  | [], _ => 0
  | _ :: _, [] => 0
  | (_, w) :: ps, e :: es => (e : Int) * w + weighted ps es

/-- One digit per radix of `ps`, each below its radix. -/
inductive Below : List (Nat × Int) → List Nat → Prop
  | nil : Below [] []
  | cons {r : Nat} {w : Int} {ps : List (Nat × Int)} {e : Nat} {es : List Nat} :
      e < r → Below ps es → Below ((r, w) :: ps) (e :: es)

/-- The digits of `i` in the radices of `ps`, fastest first. -/
def digits : List (Nat × Int) → Nat → List Nat
  | [], _ => []
  | (r, _) :: ps, i => i % r :: digits ps (i / r)

/-- **No carry**: a number written in the radices of `ps` with each digit below its radix has those digits, so its
`accum` is its digits weighted. -/
theorem accum_horner {ps : List (Nat × Int)} {es : List Nat} (h : Below ps es) :
    accum ps (horner ps es) = weighted ps es := by
  induction h with
  | nil => rfl
  | @cons r w ps e es he _ ih =>
    have hr : 0 < r := Nat.lt_of_le_of_lt (Nat.zero_le e) he
    simp only [accum, horner, weighted]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he, Nat.add_mul_div_left _ _ hr, Nat.div_eq_of_lt he,
      Nat.zero_add, ih]

/-- `accum` weights a number's digits. -/
theorem accum_digits : ∀ (ps : List (Nat × Int)) (i : Nat), accum ps i = weighted ps (digits ps i)
  | [], _ => rfl
  | (r, _) :: ps, i => by simp only [accum, digits, weighted, accum_digits ps (i / r)]

/-- A number below the radices' product is written by its digits. -/
theorem horner_digits : ∀ (ps : List (Nat × Int)) (i : Nat), i < prodR ps → horner ps (digits ps i) = i
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
theorem digits_lt : ∀ (ps : List (Nat × Int)) (i : Nat), i < prodR ps → Below ps (digits ps i)
  | [], _, _ => .nil
  | (r, _) :: ps, i, hi => by
    have hi' : i < r * prodR ps := hi
    have hr : 0 < r := Nat.pos_of_ne_zero fun h => by simp [h] at hi'
    have hq : i / r < prodR ps := (Nat.div_lt_iff_lt_mul hr).2 (by rw [Nat.mul_comm]; exact hi')
    exact .cons (Nat.mod_lt i hr) (digits_lt ps (i / r) hq)

end Properties.NetworkCertifier.WireOrderProof
