module

public import Proofs.Properties.Nci.LeafWidthsType
public import Proofs.Properties.Nci.BodiesOk.Basic
public import Mathlib.Tactic

/-!
# Two ranges of leaves the check found of the same widths are

`eqRange_sound`: where `Widths.eqRange` returns `true` for `n` leaves of `t` from leaf `a` and of `S` from leaf `b`,
each of the `n` leaves of `t` is as wide as the leaf of `S` at its place.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program

/-! ## The check's monad -/

/-- A bind that returned ran its first action, then its second on what the first returned. -/
theorem m_bind {α β : Type} {x : Widths.M α} {f : α → Widths.M β} {s : ℕ} {y : β} {s' : ℕ}
    (h : (x >>= f).run s = .ok y s') : ∃ a s₁, x.run s = .ok a s₁ ∧ (f a).run s₁ = .ok y s' := by
  rcases hx : x.run s with ⟨a, s₁⟩ | ⟨e, s₁⟩
  · refine ⟨a, s₁, rfl, ?_⟩
    have e1 : (x >>= f).run s = (f a).run s₁ := by
      simp only [EStateM.run] at hx
      simp only [bind, EStateM.bind, EStateM.run, hx]
    rw [← e1]
    exact h
  · have e1 : (x >>= f).run s = .error e s₁ := by
      simp only [EStateM.run] at hx
      simp only [bind, EStateM.bind, EStateM.run, hx]
    rw [e1] at h
    exact absurd h (by simp)

/-- A `pure` returns its value and leaves the count. -/
theorem m_pure {α : Type} {a y : α} {s s' : ℕ} (h : (pure a : Widths.M α).run s = .ok y s') :
    a = y ∧ s = s' := by
  have h' : EStateM.Result.ok a s = EStateM.Result.ok y s' := h
  injection h' with h1 h2
  exact ⟨h1, h2⟩

/-- A `throw` returns nothing. -/
theorem m_throw {α : Type} {e : String} {y : α} {s s' : ℕ} (h : (throw e : Widths.M α).run s = .ok y s') :
    False := by
  have h' : EStateM.Result.error e s = EStateM.Result.ok y s' := h
  exact absurd h' (by simp)

/-- A `for` loop that returned: `G` holds of every element and `J` of the result, unless a step stopped it, when
`D` holds of the result. -/
theorem forIn_m_inv {α β : Type} (f : α → β → Widths.M (ForInStep β)) (G : α → Prop) (J D : β → Prop) :
    ∀ (l : List α) (b r : β) (s s' : ℕ), J b →
      (∀ x ∈ l, ∀ b s t s', J b → (f x b).run s = .ok t s' → match t with
        | .yield b' => G x ∧ J b'
        | .done b' => D b') →
      (forIn l b f).run s = .ok r s' → ((∀ x ∈ l, G x) ∧ J r) ∨ D r
  | [], b, r, s, s', hb, _, h => by
    rw [List.forIn_nil] at h
    obtain ⟨rfl, -⟩ := m_pure h
    exact Or.inl ⟨fun x hx => absurd hx (by simp), hb⟩
  | x :: xs, b, r, s, s', hb, hs, h => by
    rw [List.forIn_cons] at h
    obtain ⟨t, s₁, hf, h⟩ := m_bind h
    have h1 := hs x (by simp) b s t s₁ hb hf
    rcases t with b' | b'
    · obtain ⟨rfl, -⟩ := m_pure h
      exact Or.inr h1
    · rcases forIn_m_inv f G J D xs b' r s₁ s' h1.2 (fun y hy => hs y (List.mem_cons_of_mem _ hy)) h with
        ⟨h2, h3⟩ | h2
      · refine Or.inl ⟨fun y hy => ?_, h3⟩
        rcases List.mem_cons.1 hy with rfl | hy
        · exact h1.1
        · exact h2 y hy
      · exact Or.inr h2

/-! ## The pieces a range is split into -/

/-- Pieces laid end to end from leaf `o`, of the lengths `ms`: each a first leaf and a length. -/
def chainFrom : ℕ → List ℕ → List (ℕ × ℕ)
  | _, [] => []
  | o, m :: ms => (o, m) :: chainFrom (o + m) ms

/-- A leaf from the first piece on and before the end of the last lies in a piece. -/
theorem chain_mem (ms : List ℕ) (o x : ℕ) (h1 : o ≤ x) (h2 : x < o + ms.sum) :
    ∃ p ∈ chainFrom o ms, p.1 ≤ x ∧ x < p.1 + p.2 := by
  induction ms generalizing o with
  | nil =>
    simp only [List.sum_nil] at h2
    omega
  | cons m ms ih =>
    by_cases hx : x < o + m
    · exact ⟨(o, m), by simp [chainFrom], h1, hx⟩
    · simp only [List.sum_cons] at h2
      obtain ⟨p, hp, h3⟩ := ih (o + m) (by omega) (by omega)
      exact ⟨p, by simp [chainFrom, hp], h3⟩

/-- The last piece ends where the lengths end. -/
theorem chain_last (ms : List ℕ) (o : ℕ) (p : ℕ × ℕ) (h : (chainFrom o ms).getLast? = some p) :
    p.1 + p.2 = o + ms.sum := by
  induction ms generalizing o with
  | nil => simp [chainFrom] at h
  | cons m ms ih =>
    cases ms with
    | nil =>
      simp only [chainFrom, List.getLast?_singleton, Option.some.injEq] at h
      subst h
      simp
    | cons m' ms =>
      rw [chainFrom, chainFrom, List.getLast?_cons_cons] at h
      have := ih (o + m) (by rw [chainFrom]; exact h)
      simp only [List.sum_cons] at this ⊢
      omega

/-- The pieces `Widths.eqRange` splits `n` leaves from leaf `base` into, by the part `c` that starts at leaf `oc`:
an Array's elements that meet them, or a Tuple's or a Record's fields. -/
def piecesOf (c : Ty) (oc base n : ℕ) : Array (ℕ × ℕ) :=
  match c with
  | .array _ e =>
    let L := e.leaves
    if L == 0 then #[] else
      let first := (base - oc) / L
      let last := (base + n - 1 - oc) / L
      (Array.range (min (last + 1 - first) (Widths.WALK + 1))).map fun k => (oc + (first + k) * L, L)
  | _ => (Widths.childLeaves c).foldl (fun (acc : Array (ℕ × ℕ) × ℕ) m => (acc.1.push (acc.2, m), acc.2 + m))
      (#[], oc) |>.1

/-- An Array's elements from element `first + s` on are laid end to end. -/
theorem range_chain (oc first L : ℕ) : ∀ (N s : ℕ),
    (List.range' s N).map (fun k => (oc + (first + k) * L, L)) =
      chainFrom (oc + (first + s) * L) (List.replicate N L) := by
  intro N
  induction N with
  | zero => intro s; rfl
  | succ N ih =>
    intro s
    have e : oc + (first + s) * L + L = oc + (first + (s + 1)) * L := by ring
    rw [List.range'_succ, List.map_cons, List.replicate_succ, chainFrom, ih (s + 1), e]

/-- Fields pushed one after another are laid end to end. -/
theorem push_chain : ∀ (ms : List ℕ) (A : Array (ℕ × ℕ)) (o : ℕ),
    (ms.foldl (fun (acc : Array (ℕ × ℕ) × ℕ) m => (acc.1.push (acc.2, m), acc.2 + m)) (A, o)).1.toList =
      A.toList ++ chainFrom o ms := by
  intro ms
  induction ms with
  | nil => intro A o; simp [chainFrom]
  | cons m ms ih =>
    intro A o
    rw [List.foldl_cons, ih, chainFrom, Array.toList_push, List.append_assoc]
    rfl

/-- The pieces are laid end to end, from a leaf at or before `base`. -/
theorem pieces_chain (c : Ty) (oc base n : ℕ) (h : oc ≤ base) :
    ∃ o ms, (piecesOf c oc base n).toList = chainFrom o ms ∧ o ≤ base := by
  have other : ∃ o ms, ((Widths.childLeaves c).foldl
      (fun (acc : Array (ℕ × ℕ) × ℕ) m => (acc.1.push (acc.2, m), acc.2 + m)) (#[], oc)).1.toList =
        chainFrom o ms ∧ o ≤ base :=
    ⟨oc, (Widths.childLeaves c).toList, by rw [← Array.foldl_toList, push_chain]; simp, h⟩
  cases c with
  | array k e =>
    show ∃ o ms, (if (e.leaves == 0) = true then #[] else
      (Array.range (min ((base + n - 1 - oc) / e.leaves + 1 - (base - oc) / e.leaves) (Widths.WALK + 1))).map
        fun k => (oc + ((base - oc) / e.leaves + k) * e.leaves, e.leaves)).toList = chainFrom o ms ∧ o ≤ base
    by_cases hL : (e.leaves == 0) = true
    · rw [ite_eq_left hL]
      exact ⟨oc, [], rfl, h⟩
    · rw [ite_eq_right hL]
      refine ⟨oc + ((base - oc) / e.leaves + 0) * e.leaves, List.replicate
        (min ((base + n - 1 - oc) / e.leaves + 1 - (base - oc) / e.leaves) (Widths.WALK + 1)) e.leaves, ?_, ?_⟩
      · rw [Array.toList_map, Array.toList_range, List.range_eq_range', range_chain]
      · have := Nat.div_mul_le_self (base - oc) e.leaves
        rw [Nat.add_zero]
        omega
  | value w => exact other
  | tuple fs => exact other
  | record fs => exact other

/-! ## Arrays of one period, and of two -/

/-- A leaf of an Array of elements with leaves is its element's leaf. -/
theorem wAt_array (n : ℕ) (e : Ty) (j : ℕ) (hL : 0 < e.leaves) :
    Widths.wAt (.array n e) j = Widths.wAt e (j % e.leaves) := by
  rw [Widths.wAt, ite_eq_right (by simp; omega)]

/-- Leaves `u + k` and `v + k` of two Arrays of period `L` are `d` apart in the period. -/
theorem rot_mod (u v k L : ℕ) (hL : 0 < L) : ((u + k) % L + (v + L - u % L) % L) % L = (v + k) % L := by
  have h1 := Nat.div_add_mod' u L
  have h2 := Nat.mod_lt u hL
  have e1 : u + k + (v + L - u % L) = (v + k) + (u / L + 1) * L := by
    rw [Nat.add_mul, Nat.one_mul]
    omega
  rw [← Nat.add_mod, e1, Nat.add_mul_mod_self_right]

/-- An element read `d` leaves on, in two parts, is read `d` leaves on at every leaf. -/
theorem rot_all (e f : Ty) (L d : ℕ) (hd : d < L) (h1 : ∀ k < L - d, Widths.wAt e (0 + k) = Widths.wAt f (d + k))
    (h2 : ∀ k < d, Widths.wAt e (L - d + k) = Widths.wAt f (0 + k)) (r : ℕ) (hr : r < L) :
    Widths.wAt e r = Widths.wAt f ((r + d) % L) := by
  by_cases hrd : r < L - d
  · have := h1 r hrd
    rw [Nat.zero_add, Nat.add_comm d r] at this
    rw [Nat.mod_eq_of_lt (by omega)]
    exact this
  · have := h2 (r - (L - d)) (by omega)
    rw [Nat.zero_add] at this
    rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]
    have e1 : L - d + (r - (L - d)) = r := by omega
    have e2 : r - (L - d) = r + d - L := by omega
    rw [e1, e2] at this
    exact this

/-- A leaf's place in a period `L` is its place after whole periods of a multiple of `L` are dropped. -/
theorem mod_period (u k L P : ℕ) (hd : L ∣ P) : (u + k) % L = (u + k % P) % L := by
  rw [Nat.add_mod, Nat.add_mod u (k % P), Nat.mod_mod_of_dvd k hd]

/-! ## The check is sound -/

/-- **Two ranges the check found of the same widths are**: where `Widths.eqRange` returns `true`, each of the `n`
leaves of `t` from leaf `a` is as wide as the leaf of `S` at its place from leaf `b`. -/
theorem eqRange_sound (fuel : ℕ) (t : Ty) (a : ℕ) (S : Ty) (b n w w' : ℕ) (ha : a + n ≤ t.leaves)
    (hb : b + n ≤ S.leaves) (h : (Widths.eqRange fuel t a S b n).run w = .ok true w') :
    ∀ k < n, Widths.wAt t (a + k) = Widths.wAt S (b + k) := by
  induction fuel generalizing t a S b n w w' with
  | zero =>
    rw [Widths.eqRange] at h
    exact (m_throw h).elim
  | succ fuel ih =>
    rw [Widths.eqRange] at h
    by_cases hn : n = 0
    · subst hn
      intro k hk
      omega
    rw [ite_eq_right (by simpa using hn)] at h
    split at h
    · rename_i x y hx hy
      have hxy : x = y := by simpa using (m_pure h).1
      intro k hk
      have h1 := wAt_mem_rangeWidths t a (a + n) (a + k) (by omega) (by omega) (by omega)
      have h2 := wAt_mem_rangeWidths S b (b + n) (b + k) (by omega) (by omega) (by omega)
      rw [hx] at h1
      rw [hy] at h2
      rw [List.mem_singleton.1 h1, List.mem_singleton.1 h2, hxy]
    · exact absurd (m_pure h).1 (by simp)
    · exact absurd (m_pure h).1 (by simp)
    · have Ct := cover_spec t a (a + n) (by omega) ha
      have Cs := cover_spec S b (b + n) (by omega) hb
      split at h
      rename_i ct ot hct
      split at h
      rename_i cs os hcs
      rw [hct] at Ct
      rw [hcs] at Cs
      obtain ⟨t1, t2, t3, t4⟩ := Ct
      obtain ⟨s1, s2, s3, s4⟩ := Cs
      dsimp only at t1 t2 t3 t4 s1 s2 s3 s4
      by_cases hsame : (a - ot == b - os && ct == cs) = true
      · rw [Bool.and_eq_true] at hsame
        obtain ⟨e1, e2⟩ := hsame
        have e1' : a - ot = b - os := by simpa using e1
        have e3 := BodiesOk.ty_beq e2
        subst e3
        intro k hk
        rw [t4 (a + k) (by omega) (by omega), s4 (b + k) (by omega) (by omega)]
        congr 1
        omega
      rw [ite_eq_right hsame] at h
      extract_lets byS jpT jp at h
      have hjp : ∀ w₁ w₁', (jp ()).run w₁ = .ok true w₁' → ∀ k < n, Widths.wAt t (a + k) = Widths.wAt S (b + k) := by
        intro w₁ w₁' hj
        unfold jp at hj
        split at hj
        rename_i c oc base hsel
        have hoc : oc ≤ base := by
          by_cases hbS : byS = true
          · rw [ite_eq_left hbS] at hsel
            injection hsel with _ h2
            injection h2 with h3 h4
            omega
          · rw [ite_eq_right hbS] at hsel
            injection hsel with _ h2
            injection h2 with h3 h4
            omega
        extract_lets pieces at hj
        have hp : pieces = piecesOf c oc base n := rfl
        clear_value pieces
        by_cases hemp : pieces.isEmpty = true
        · rw [ite_eq_left hemp] at hj
          exact absurd (m_pure hj).1 (by simp)
        rw [ite_eq_right hemp] at hj
        obtain ⟨r, w₂, hloop, hj⟩ := m_bind hj
        rw [← Array.forIn_toList] at hloop
        have hL := forIn_m_inv _
          (fun p : ℕ × ℕ => ∀ k < n, p.1 ≤ base + k → base + k < p.1 + p.2 →
            Widths.wAt t (a + k) = Widths.wAt S (b + k))
          (fun r : Option Bool × Unit => r.1 = none) (fun r => r.1 = some false) pieces.toList (none, ()) r w₁ w₂
          rfl ?step hloop
        case step =>
          rintro ⟨o, m⟩ - bb s₀ st s₀' - hf
          dsimp only at hf
          by_cases hlt : max base o < min (base + n) (o + m)
          · rw [ite_eq_left hlt] at hf
            obtain ⟨u, s₁, -, hf⟩ := m_bind hf
            obtain ⟨res, s₂, hres, hf⟩ := m_bind hf
            cases res with
            | false =>
              rw [ite_eq_left (by decide)] at hf
              obtain ⟨rfl, -⟩ := m_pure hf
              rfl
            | true =>
              rw [ite_eq_right (by decide)] at hf
              obtain ⟨rfl, -⟩ := m_pure hf
              refine ⟨fun k hk h1 h2 => ?_, rfl⟩
              have key := ih t (a + (max base o - base)) S (b + (max base o - base))
                (min (base + n) (o + m) - max base o) s₁ s₂ (by omega) (by omega) hres
                (base + k - max base o) (by omega)
              have e1 : a + (max base o - base) + (base + k - max base o) = a + k := by omega
              have e2 : b + (max base o - base) + (base + k - max base o) = b + k := by omega
              rw [e1, e2] at key
              exact key
          · rw [ite_eq_right hlt] at hf
            obtain ⟨rfl, -⟩ := m_pure hf
            exact ⟨fun k hk h1 h2 => absurd (show max base o < min (base + n) (o + m) by omega) hlt, rfl⟩
        obtain ⟨r1, r2⟩ := r
        dsimp only at hL
        rcases hL with ⟨hall, hr⟩ | hr
        · subst hr
          rcases hbk : pieces.back? with _ | ⟨o', m'⟩
          · exfalso
            have h0 : pieces.toList.getLast? = none := by rw [Array.getLast?_toList, hbk]
            rw [List.getLast?_eq_none_iff] at h0
            exact hemp (by rw [← Array.isEmpty_toList, h0]; rfl)
          · rw [hbk] at hj
            dsimp only at hj
            by_cases hend : o' + m' < base + n
            · rw [ite_eq_left hend] at hj
              obtain ⟨u, s₃, hthrow, -⟩ := m_bind hj
              exact (m_throw hthrow).elim
            · obtain ⟨o0, ms, hch, ho0⟩ := pieces_chain c oc base n hoc
              rw [← hp] at hch
              have hlast := chain_last ms o0 (o', m') (by rw [← hch, Array.getLast?_toList, hbk])
              dsimp only at hlast
              intro k hk
              obtain ⟨p, hpm, hp1, hp2⟩ := chain_mem ms o0 (base + k) (by omega) (by omega)
              rw [← hch] at hpm
              exact hall p hpm k hk hp1 hp2
        · subst hr
          dsimp only at hj
          exact absurd (m_pure hj).1 (by simp)
      clear_value jp
      split at h
      · rename_i n1 e n2 f heq
        injection heq with hce hcf
        subst hce
        subst hcf
        dsimp only at h
        by_cases hpos : (decide (e.leaves > 0) && decide (f.leaves > 0)) = true
        · rw [ite_eq_left hpos] at h
          have hpos' : 0 < e.leaves ∧ 0 < f.leaves := by simpa using hpos
          obtain ⟨hLe, hLf⟩ := hpos'
          by_cases hper : (e.leaves == f.leaves && decide (n ≥ e.leaves)) = true
          · rw [ite_eq_left hper] at h
            have hper' : e.leaves = f.leaves ∧ e.leaves ≤ n := by simpa using hper
            obtain ⟨hLL, hnL⟩ := hper'
            obtain ⟨r1, s₁, hr1, h⟩ := m_bind h
            obtain ⟨r2, s₂, hr2, h⟩ := m_bind h
            have hb1 : r1 = true ∧ r2 = true := by simpa using (m_pure h).1
            obtain ⟨rfl, rfl⟩ := hb1
            have hdL : (b - os + e.leaves - (a - ot) % e.leaves) % e.leaves < e.leaves := Nat.mod_lt _ hLe
            have k1 := ih e 0 f _ _ w s₁ (by omega) (by omega) hr1
            have k2 := ih e _ f 0 _ s₁ s₂ (by omega) (by omega) hr2
            have hrot := rot_all e f e.leaves _ hdL k1 k2
            intro k hk
            rw [t4 (a + k) (by omega) (by omega), s4 (b + k) (by omega) (by omega),
              wAt_array n1 e _ hLe, wAt_array n2 f _ hLf, ← hLL,
              hrot ((a + k - ot) % e.leaves) (Nat.mod_lt _ hLe)]
            congr 1
            have h1 : a + k - ot = a - ot + k := by omega
            have h2 : b + k - os = b - os + k := by omega
            rw [h1, h2]
            exact rot_mod (a - ot) (b - os) k e.leaves hLe
          · rw [ite_eq_right hper] at h
            by_cases hP : n > e.leaves.lcm f.leaves
            · rw [ite_eq_left hP] at h
              have hPpos : 0 < e.leaves.lcm f.leaves := Nat.lcm_pos hLe hLf
              have kP := ih t a S b (e.leaves.lcm f.leaves) w w' (by omega) (by omega) h
              intro k hk
              have hkP : k % e.leaves.lcm f.leaves < e.leaves.lcm f.leaves := Nat.mod_lt _ hPpos
              have hk' := kP _ hkP
              rw [t4 _ (by omega) (by omega), s4 _ (by omega) (by omega),
                wAt_array n1 e _ hLe, wAt_array n2 f _ hLf] at hk'
              rw [t4 (a + k) (by omega) (by omega), s4 (b + k) (by omega) (by omega),
                wAt_array n1 e _ hLe, wAt_array n2 f _ hLf]
              have h1 : a + k - ot = a - ot + k := by omega
              have h2 : b + k - os = b - os + k := by omega
              have h3 : a + k % e.leaves.lcm f.leaves - ot = a - ot + k % e.leaves.lcm f.leaves := by omega
              have h4 : b + k % e.leaves.lcm f.leaves - os = b - os + k % e.leaves.lcm f.leaves := by omega
              rw [h1, h2, mod_period (a - ot) k _ _ (Nat.dvd_lcm_left e.leaves f.leaves),
                mod_period (b - os) k _ _ (Nat.dvd_lcm_right e.leaves f.leaves), ← h3, ← h4]
              exact hk'
            · rw [ite_eq_right hP] at h
              exact hjp _ _ h
        · rw [ite_eq_right hpos] at h
          exact hjp _ _ h
      · exact hjp _ _ h

end NciCheck.Properties.SecurityProofs.LeafWidths
