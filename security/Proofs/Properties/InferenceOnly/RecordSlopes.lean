module

public import InferenceOnlyVerifier.Record
public import Mathlib.Tactic

/-!
# A registered read's sources in closed form

`Record.nodesHeld` holds each registered read that fills a declared node's rows at one corner of the units and rows it
fills, and reads the rest off the read's slopes: one slope a unit and one a row (`Record.slopesOf`), or two slopes about
place `q` (`Record.slopesAbout`). This module gives the source the read names at every unit and row it covers from
those slopes.

* `accum_outer`, `accum_small`, `accum_inner`: a wire's unit digits (`SlopedFrom`), each stepping its place times one
  slope, or its place over `q` times another, sum at index `v` to `si · (v mod q) + so · ⌊v / q⌋`.
* `sloped_of_slopesOf`, `sloped_of_slopesAbout`: the digits `slopesOf` and `slopesAbout` accept are so sloped.
* `writerAt_rows`: a read whose last digit is its cut has, at unit `v` and row `c` of its cut, the source
  `base + c · sr + accum us v`, `sr` its row slope and `us` its unit digits.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Spot.Wires (accum prodR)

/-- A wire's digits from place `P` on, fastest first: each of radix one or none, or stepping its place times `si`
where its place times its radix is at most `q`, or its place over `q` times `so` where `q` divides its place. -/
def SlopedFrom (q : Nat) (si so : Int) : List (Nat × Int) → Nat → Prop
  | [], _ => True
  | (r, s) :: rest, P => (r ≤ 1 ∨ (P * r ≤ q ∧ s = (P : Int) * si) ∨ (q ∣ P ∧ s = ((P / q : Nat) : Int) * so)) ∧
      SlopedFrom q si so rest (P * r)

private theorem prodR_cons (r : Nat) (s : Int) (rest : List (Nat × Int)) :
    prodR ((r, s) :: rest) = r * prodR rest := rfl

private theorem accum_cons (r : Nat) (s : Int) (rest : List (Nat × Int)) (v : Nat) :
    accum ((r, s) :: rest) v = ((v % r : Nat) : Int) * s + accum rest (v / r) := rfl

private theorem mod_div_cast (v r : Nat) : (v : Int) = ((v % r : Nat) : Int) + (r : Int) * ((v / r : Nat) : Int) := by
  exact_mod_cast (Nat.mod_add_div v r).symm

/-- **Digits from a place `q` divides on step their place over `q` times `so`.** -/
theorem accum_outer (q : Nat) (si so : Int) (hq : 0 < q) :
    ∀ (us : List (Nat × Int)) (P : Nat), SlopedFrom q si so us P → q ∣ P → 0 < P →
      ∀ v, v < prodR us → accum us v = ((P / q : Nat) : Int) * so * v
  | [], P, _, _, _, v, hv => by
    have hv0 : v = 0 := by
      have : prodR ([] : List (Nat × Int)) = 1 := rfl
      omega
    subst hv0
    simp [accum]
  | (r, s) :: rest, P, ⟨hd, htl⟩, hqP, hP, v, hv => by
    rw [prodR_cons] at hv
    have hr : 0 < r := by
      rcases Nat.eq_zero_or_pos r with h | h
      · subst h
        simp at hv
      · exact h
    have hvr : v / r < prodR rest := Nat.div_lt_of_lt_mul hv
    have ih := accum_outer q si so hq rest (P * r) htl (Dvd.dvd.mul_right hqP r) (Nat.mul_pos hP hr) (v / r) hvr
    rw [accum_cons, ih, Nat.mul_div_right_comm hqP]
    by_cases h1 : r = 1
    · subst h1
      simp
    · have hr2 : 2 ≤ r := by omega
      have hPq : q ≤ P := Nat.le_of_dvd hP hqP
      rcases hd with h | ⟨h, -⟩ | ⟨-, hs⟩
      · omega
      · have : P * 2 ≤ P * r := Nat.mul_le_mul_left P hr2
        omega
      · rw [hs, mod_div_cast v r]
        push_cast
        ring

/-- **Digits below a place that divides no `q` step their place times `si`**, and stay within `q`. -/
theorem accum_small (q : Nat) (si so : Int) :
    ∀ (us : List (Nat × Int)) (P : Nat), SlopedFrom q si so us P → 0 < P → P ≤ q → ¬ P ∣ q →
      P * prodR us ≤ q ∧ ∀ v, v < prodR us → accum us v = (P : Int) * si * v
  | [], P, _, _, hPq, _ => by
    refine ⟨by rw [show prodR ([] : List (Nat × Int)) = 1 from rfl, Nat.mul_one]; exact hPq, fun v hv => ?_⟩
    have hv0 : v = 0 := by
      have : prodR ([] : List (Nat × Int)) = 1 := rfl
      omega
    subst hv0
    simp [accum]
  | (r, s) :: rest, P, ⟨hd, htl⟩, hP, hPq, hnd => by
    rw [prodR_cons]
    rcases Nat.eq_zero_or_pos r with h0 | hr
    · subst h0
      refine ⟨by simp, fun v hv => by simp at hv⟩
    by_cases h1 : r = 1
    · subst h1
      have ih := accum_small q si so rest (P * 1) htl (by omega) (by omega) (by rw [Nat.mul_one]; exact hnd)
      refine ⟨by rw [Nat.one_mul]; simpa using ih.1, fun v hv => ?_⟩
      rw [accum_cons, ih.2 (v / 1) (by simpa using hv)]
      simp
    · have hr2 : 2 ≤ r := by omega
      rcases hd with h | ⟨hin, hs⟩ | ⟨hdv, -⟩
      · omega
      · have hnd' : ¬ P * r ∣ q := fun h => hnd (Nat.dvd_trans (Dvd.intro r rfl) h)
        have ih := accum_small q si so rest (P * r) htl (Nat.mul_pos hP hr) hin hnd'
        refine ⟨by rw [← Nat.mul_assoc]; exact ih.1, fun v hv => ?_⟩
        rw [accum_cons, ih.2 (v / r) (Nat.div_lt_of_lt_mul hv), hs, mod_div_cast v r]
        push_cast
        ring
      · exact absurd (Nat.le_antisymm hPq (Nat.le_of_dvd hP hdv) ▸ Nat.dvd_refl P) hnd

/-- `m · a` and a remainder `b` below `m`, `m` dividing `q`: adding `b` moves `m · a` within its block of `q`. -/
private theorem add_within {m q a b : Nat} (hm : m ∣ q) (hq : 0 < q) (hb : b < m) :
    (b + m * a) % q = b + m * a % q ∧ (b + m * a) / q = m * a / q := by
  have hdm : m ∣ m * a % q := (Nat.dvd_mod_iff hm).2 (Dvd.intro a rfl)
  have hlt : m * a % q < q := Nat.mod_lt _ hq
  obtain ⟨c, hc⟩ := hdm
  obtain ⟨d, hd⟩ := hm
  have hcd : c < d := by
    rcases Nat.eq_zero_or_pos m with h0 | hm0
    · subst h0
      omega
    · rw [hc, hd] at hlt
      exact Nat.lt_of_mul_lt_mul_left hlt
  have hle : m * a % q + m ≤ q := by
    rw [hc, hd]
    have : m * c + m = m * (c + 1) := by ring
    rw [this]
    exact Nat.mul_le_mul_left m hcd
  have hsum : b + m * a % q < q := by omega
  have hsplit : b + m * a = (b + m * a % q) + q * (m * a / q) := by
    have := Nat.mod_add_div (m * a) q
    omega
  refine ⟨?_, ?_⟩
  · rw [hsplit, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hsum]
  · rw [hsplit, Nat.add_mul_div_left _ _ hq, Nat.div_eq_of_lt hsum, Nat.zero_add]

/-- **Sloped digits from a place dividing `q` sum to `si · (P v mod q) + so · ⌊P v / q⌋`.** -/
theorem accum_inner (q : Nat) (si so : Int) (hq : 0 < q) :
    ∀ (us : List (Nat × Int)) (P : Nat), SlopedFrom q si so us P → 0 < P → P ∣ q →
      ∀ v, v < prodR us → accum us v = si * ((P * v % q : Nat) : Int) + so * ((P * v / q : Nat) : Int)
  | [], P, _, _, _, v, hv => by
    have hv0 : v = 0 := by
      have : prodR ([] : List (Nat × Int)) = 1 := rfl
      omega
    subst hv0
    simp [accum]
  | (r, s) :: rest, P, hgood, hP, hPq, v, hv => by
    obtain ⟨hd, htl⟩ := hgood
    have hv' := hv
    rw [prodR_cons] at hv'
    rcases Nat.eq_zero_or_pos r with h0 | hr
    · subst h0
      simp at hv'
    have hvr : v / r < prodR rest := Nat.div_lt_of_lt_mul hv'
    by_cases h1 : r = 1
    · subst h1
      have ih := accum_inner q si so hq rest (P * 1) htl (by omega) (by rw [Nat.mul_one]; exact hPq) (v / 1) hvr
      rw [accum_cons, ih]
      simp
    · have hr2 : 2 ≤ r := by omega
      rcases hd with h | ⟨hin, hs⟩ | ⟨hdv, hs'⟩
      · omega
      · have hb : P * (v % r) < P * r := Nat.mul_lt_mul_of_pos_left (Nat.mod_lt v hr) hP
        have hPv : P * v = P * (v % r) + P * r * (v / r) := by
          have := Nat.mod_add_div v r
          calc P * v = P * (v % r + r * (v / r)) := by rw [this]
            _ = P * (v % r) + P * r * (v / r) := by ring
        by_cases hdr : P * r ∣ q
        · have ih := accum_inner q si so hq rest (P * r) htl (Nat.mul_pos hP hr) hdr (v / r) hvr
          obtain ⟨hmod, hdiv⟩ := add_within (a := v / r) hdr hq hb
          rw [accum_cons, ih, hPv, hmod, hdiv, hs]
          push_cast
          ring
        · obtain ⟨hbound, hsm⟩ := accum_small q si so rest (P * r) htl (Nat.mul_pos hP hr) hin hdr
          have hlt : P * v < q := by
            have : P * v < P * (r * prodR rest) := Nat.mul_lt_mul_of_pos_left hv' hP
            rw [← Nat.mul_assoc] at this
            omega
          rw [accum_cons, hsm (v / r) hvr, hs, Nat.mod_eq_of_lt hlt, Nat.div_eq_of_lt hlt, hPv]
          push_cast
          ring
      · have hPeq : P = q := Nat.dvd_antisymm hPq hdv
        subst hPeq
        rw [accum_outer P si so hq ((r, s) :: rest) P ⟨Or.inr (Or.inr ⟨hdv, hs'⟩), htl⟩ hdv hP v hv, Nat.div_self hP,
          Nat.mul_mod_right, Nat.mul_div_cancel_left _ hP]
        push_cast
        ring

/-! ## The slopes `nodesHeld` accepts -/

/-- A wire's unit digits as `slopesOf` and `slopesAbout` read them: its digits fastest first, past its cut's. -/
def unitDigits (w : Spot.Wires.Wire) : List (Nat × Int) := if w.cuts > 1 then w.pairs.drop 1 else w.pairs

/-- A wire's row slope as `slopesOf` reads it: its cut digit's step, 0 for one row a unit. -/
def rowSlope (w : Spot.Wires.Wire) : Int := if w.cuts > 1 then (w.pairs.head?.map (·.2)).getD 0 else 0

/-- `slopesOf`'s digits: each unit digit of radix above one, with its place. -/
def placesOf (us : List (Nat × Int)) : List (Nat × Int) :=
  (us.foldl (fun (acc : Nat × List (Nat × Int)) (r, s) =>
    (acc.1 * r, if r > 1 then acc.2 ++ [(acc.1, s)] else acc.2)) (1, [])).2

/-- `slopesAbout`'s digits: each unit digit of radix above one, with its place and radix. -/
def placesAbout (us : List (Nat × Int)) : List (Nat × Nat × Int) :=
  (us.foldl (fun (acc : Nat × List (Nat × Nat × Int)) (r, s) =>
    (acc.1 * r, if r > 1 then acc.2 ++ [(acc.1, r, s)] else acc.2)) (1, [])).2

/-- `slopesAbout`'s slope of the digits `ds`, places over `div`. -/
def slopeOf (ds : List (Nat × Nat × Int)) (div : Nat) : Option (Option Int) := match ds with
  | [] => some none
  | (p, _, s) :: _ =>
    let u := (p / div : Nat)
    if u ≥ 1 && s % (u : Int) == 0 && ds.all (fun (p', _, t) => t == ((p' / div : Nat) : Int) * (s / (u : Int)))
    then some (some (s / (u : Int))) else none

theorem slopesOf_eq (w : Spot.Wires.Wire) : slopesOf w =
    match placesOf (unitDigits w) with
    | [] => some (0, rowSlope w)
    | (p, s) :: _ =>
      if s % (p : Int) == 0 && (placesOf (unitDigits w)).all (fun (q, t) => t == (q : Int) * (s / (p : Int)))
      then some (s / (p : Int), rowSlope w) else none := by
  unfold slopesOf placesOf unitDigits rowSlope
  by_cases hc : w.cuts > 1
  · simp only [hc, ↓reduceIte]; rfl
  · simp only [hc, ↓reduceIte]; rfl

theorem slopesAbout_eq (w : Spot.Wires.Wire) (q : Nat) : slopesAbout w q =
    if q ≥ 1 && ((placesAbout (unitDigits w)).filter fun (p, r, _) => ¬ p * r ≤ q).all (fun (p, _, _) => p % q == 0)
    then match slopeOf ((placesAbout (unitDigits w)).filter fun (p, r, _) => p * r ≤ q) 1,
        slopeOf ((placesAbout (unitDigits w)).filter fun (p, r, _) => ¬ p * r ≤ q) q with
      | some si, some so => some (si, so)
      | _, _ => none
    else none := rfl

/-- What a fold of places keeps of each digit of radix above one: `Q` of what it makes of it. -/
def PlacesQ {β : Type} (mk : Nat → Nat → Int → β) (Q : β → Prop) : List (Nat × Int) → Nat → Prop
  | [], _ => True
  | (r, s) :: rest, P => (1 < r → Q (mk P r s)) ∧ PlacesQ mk Q rest (P * r)

theorem fold_places {β : Type} (F : Nat × List β → Nat × Int → Nat × List β) (mk : Nat → Nat → Int → β)
    (hF : ∀ P L r s, F (P, L) (r, s) = (P * r, if r > 1 then L ++ [mk P r s] else L)) (Q : β → Prop) :
    ∀ (us : List (Nat × Int)) (P : Nat) (L : List β), (∀ x ∈ (us.foldl F (P, L)).2, Q x) →
      (∀ x ∈ L, Q x) ∧ PlacesQ mk Q us P
  | [], _, _, h => ⟨h, trivial⟩
  | (r, s) :: rest, P, L, h => by
    rw [List.foldl_cons, hF] at h
    obtain ⟨h1, h2⟩ := fold_places F mk hF Q rest (P * r) _ h
    by_cases hr : r > 1
    · rw [if_pos hr] at h1
      exact ⟨fun x hx => h1 x (List.mem_append_left _ hx),
        fun _ => h1 _ (List.mem_append_right _ (List.mem_singleton_self _)), h2⟩
    · rw [if_neg hr] at h1
      exact ⟨h1, fun h' => absurd h' hr, h2⟩

/-- **The unit digits `slopesOf` accepts step their place times its unit slope.** -/
theorem sloped_of_slopesOf {w : Spot.Wires.Wire} {su sr : Int} (h : slopesOf w = some (su, sr)) :
    sr = rowSlope w ∧ SlopedFrom 1 0 su (unitDigits w) 1 := by
  have key : ∀ (us : List (Nat × Int)) (P : Nat),
      PlacesQ (fun P _ s => (P, s)) (fun x : Nat × Int => x.2 = (x.1 : Int) * su) us P → SlopedFrom 1 0 su us P := by
    intro us
    induction us with
    | nil => intro _ _; trivial
    | cons x rest ih =>
      intro P hP
      obtain ⟨r, s⟩ := x
      obtain ⟨h1, h2⟩ := hP
      refine ⟨?_, ih _ h2⟩
      by_cases hr : 1 < r
      · exact Or.inr (Or.inr ⟨Nat.one_dvd P, by rw [Nat.div_one]; exact h1 hr⟩)
      · exact Or.inl (by omega)
  have hpl : ∀ Q : Nat × Int → Prop, (∀ x ∈ placesOf (unitDigits w), Q x) →
      PlacesQ (fun P _ s => (P, s)) Q (unitDigits w) 1 := fun Q hQ =>
    (fold_places _ (fun P _ s => (P, s)) (fun _ _ _ _ => rfl) Q (unitDigits w) 1 [] hQ).2
  rw [slopesOf_eq] at h
  split at h
  · rename_i heq
    cases h
    exact ⟨rfl, key _ _ (hpl _ (by rw [heq]; simp))⟩
  · rename_i p s tl heq
    split at h
    · rename_i hcond
      cases h
      refine ⟨rfl, key _ _ (hpl _ fun x hx => ?_)⟩
      simp only [Bool.and_eq_true, List.all_eq_true, beq_iff_eq] at hcond
      exact hcond.2 x hx
    · cases h

/-- `slopeOf` of digits `ds` names one slope every digit steps its place over `div` times. -/
theorem slopeOf_spec {ds : List (Nat × Nat × Int)} {div : Nat} {o : Option Int} (h : slopeOf ds div = some o) :
    ∀ x ∈ ds, ∃ c, o = some c ∧ x.2.2 = ((x.1 / div : Nat) : Int) * c := by
  unfold slopeOf at h
  dsimp only at h
  split at h
  · intro x hx
    cases hx
  · rename_i p r0 s tl
    split at h
    · rename_i hcond
      cases h
      intro x hx
      simp only [Bool.and_eq_true, List.all_eq_true] at hcond
      have hx' := hcond.2 x hx
      obtain ⟨a, b, c⟩ := x
      exact ⟨_, rfl, by simpa using hx'⟩
    · cases h

/-- **The unit digits `slopesAbout` accepts about place `q` are sloped about it.** -/
theorem sloped_of_slopesAbout {w : Spot.Wires.Wire} {q : Nat} {si' so' : Option Int} {si so : Int}
    (h : slopesAbout w q = some (si', so')) (hsi : si'.all (· == si) = true) (hso : so'.all (· == so) = true) :
    1 ≤ q ∧ SlopedFrom q si so (unitDigits w) 1 := by
  rw [slopesAbout_eq] at h
  split at h
  · rename_i hcond
    simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, beq_iff_eq] at hcond
    split at h
    · rename_i a b hin hout
      cases h
      have hQ : ∀ x ∈ placesAbout (unitDigits w),
          (x.1 * x.2.1 ≤ q → x.2.2 = (x.1 : Int) * si) ∧
            (¬ x.1 * x.2.1 ≤ q → q ∣ x.1 ∧ x.2.2 = ((x.1 / q : Nat) : Int) * so) := by
        intro x hx
        refine ⟨fun hle => ?_, fun hgt => ?_⟩
        · obtain ⟨c, hc, hx2⟩ := slopeOf_spec hin x (List.mem_filter.2 ⟨hx, by simpa using hle⟩)
          subst hc
          simp only [Option.all_some, beq_iff_eq] at hsi
          rw [hx2, hsi, Nat.div_one]
        · have hmem : x ∈ (placesAbout (unitDigits w)).filter fun (p, r, _) => ¬ p * r ≤ q :=
            List.mem_filter.2 ⟨hx, by simpa using hgt⟩
          obtain ⟨c, hc, hx2⟩ := slopeOf_spec hout x hmem
          subst hc
          simp only [Option.all_some, beq_iff_eq] at hso
          exact ⟨Nat.dvd_of_mod_eq_zero (hcond.2 x hmem), by rw [hx2, hso]⟩
      have hpl := (fold_places _ (fun P r s => (P, r, s)) (fun _ _ _ _ => rfl) _ (unitDigits w) 1 [] hQ).2
      refine ⟨hcond.1, ?_⟩
      have key : ∀ (us : List (Nat × Int)) (P : Nat),
          PlacesQ (fun P r s => (P, r, s)) (fun x : Nat × Nat × Int => (x.1 * x.2.1 ≤ q → x.2.2 = (x.1 : Int) * si) ∧
            (¬ x.1 * x.2.1 ≤ q → q ∣ x.1 ∧ x.2.2 = ((x.1 / q : Nat) : Int) * so)) us P → SlopedFrom q si so us P := by
        intro us
        induction us with
        | nil => intro _ _; trivial
        | cons x rest ih =>
          intro P hP
          obtain ⟨r, s⟩ := x
          obtain ⟨h1, h2⟩ := hP
          refine ⟨?_, ih _ h2⟩
          by_cases hr : 1 < r
          · obtain ⟨hi, ho⟩ := h1 hr
            by_cases hle : P * r ≤ q
            · exact Or.inr (Or.inl ⟨hle, hi hle⟩)
            · exact Or.inr (Or.inr (ho hle))
          · exact Or.inl (by omega)
      exact key _ _ hpl
    · cases h
  · cases h

/-! ## A read's source at a unit and a row -/

/-- **A read's source at unit `v` and row `c` of its cut**, where it is one row a unit or its last digit is its cut:
`base + c · sr + accum us v`, `sr` its row slope and `us` its unit digits, which are its reader units' radices. -/
theorem writerAt_rows (w : Spot.Wires.Wire) (hlen : w.step.length = w.radix.length)
    (hcut : (w.cuts == 1 || w.radix.getLast? == some w.cuts) = true) {v c : Nat} (hc : c < w.cuts) :
    w.writerAt (v * w.cuts + c) = w.base + (c : Int) * rowSlope w + accum (unitDigits w) v ∧
      w.readerUnits = prodR (unitDigits w) := by
  by_cases h1 : w.cuts > 1
  · have hl : w.radix.getLast? = some w.cuts := by
      simp only [Bool.or_eq_true, beq_iff_eq] at hcut
      rcases hcut with h | h
      · omega
      · exact h
    obtain ⟨init, hinit⟩ := List.getLast?_eq_some_iff.1 hl
    have hsne : w.step ≠ [] := by
      intro h0
      rw [h0, hinit] at hlen
      simp at hlen
    have hstep := (List.dropLast_append_getLast hsne).symm
    have hilen : init.length = w.step.dropLast.length := by
      rw [List.length_dropLast, hlen, hinit]
      simp
    have hp : w.pairs = (w.cuts, w.step.getLast hsne) :: (init.zip w.step.dropLast).reverse := by
      show (w.radix.zip w.step).reverse = _
      rw [hinit]
      conv => lhs; rw [hstep]
      rw [List.zip_append hilen]
      simp
    have hu : unitDigits w = (init.zip w.step.dropLast).reverse := by
      simp only [unitDigits, h1, ↓reduceIte, hp, List.drop_one, List.tail_cons]
    have hr : rowSlope w = w.step.getLast hsne := by
      simp only [rowSlope, h1, ↓reduceIte, hp, List.head?_cons, Option.map_some, Option.getD_some]
    have hc0 : 0 < w.cuts := by omega
    refine ⟨?_, ?_⟩
    · show w.base + accum w.pairs (v * w.cuts + c) = _
      rw [hp, accum_cons, hu, hr, Nat.add_comm (v * w.cuts) c, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc,
        Nat.add_mul_div_right _ _ hc0, Nat.div_eq_of_lt hc, Nat.zero_add]
      ring
    · show prodR w.pairs / w.cuts = _
      rw [hp, prodR_cons, hu, Nat.mul_div_cancel_left _ hc0]
  · have h1' : w.cuts = 1 := by omega
    have hc0 : c = 0 := by omega
    subst hc0
    have hu : unitDigits w = w.pairs := by simp only [unitDigits, h1, ↓reduceIte]
    have hr : rowSlope w = 0 := by simp only [rowSlope, h1, ↓reduceIte]
    refine ⟨?_, ?_⟩
    · show w.base + accum w.pairs (v * w.cuts + 0) = _
      rw [hu, hr, h1', Nat.mul_one, Nat.add_zero]
      ring
    · show prodR w.pairs / w.cuts = _
      rw [hu, h1', Nat.div_one]

end InferenceOnlyVerifier.Record.SecurityProofs
