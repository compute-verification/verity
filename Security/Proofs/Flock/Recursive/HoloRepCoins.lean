import Proofs.Flock.Recursive.HoloSched

/-!
# The reps' coin rounds at the hidden-unit schedule

`coinRounds_holo`: at a claim, the rounds of `holoRounds` that draw coins are all of its rounds but z_partial's
(`zpRound`); `coinRounds_holoRep` states it at the verifier's `RecVStar.holoRepRounds`, which is `holoRounds`.
`repRounds` draws in every round (`repRounds_coins`): `fast100` (`Schedule.lean:47`) puts `m` in `[22, 35]`, the
table's variables above 0 and every level's queries at 2 or more (`fast100_facts`, by evaluating it at each `m`), and
each of `repRounds`' rounds draws a positive constant, `m - 13`, the claims, the table's variables, a split
(`n - k0 + 1`, `ext + 1`), a level's queries or `bitLength (q - 1)`. `holoRounds` puts a round drawing nothing at
`zpRound` and moves the rest one round on.
-/

namespace Flock.Guarantees.HoloRepCoins

open Flock Flock.RecVStar

theorem obind_some {α β : Type} {x : Option α} {f : α → Option β} {b : β} (h : (x >>= f) = some b) :
    ∃ a, x = some a ∧ f a = some b := by
  cases x with
  | none => cases h
  | some a => exact ⟨a, rfl, h⟩

theorem oforIn_inv' {α σ : Type} (P : α → Prop) (Q : σ → Prop) (f : α → σ → Option (ForInStep σ))
    (hf : ∀ a, P a → ∀ r y, Q r → f a r = some y → ∃ r', y = .yield r' ∧ Q r') :
    ∀ (L : List α), (∀ a ∈ L, P a) → ∀ (init v : σ), Q init → forIn L init f = some v → Q v
  | [], _, init, v, hq, h => by
    rw [List.forIn_nil] at h
    cases h
    exact hq
  | a :: L, hL, init, v, hq, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := obind_some h
    obtain ⟨r', rfl, hq'⟩ := hf a (hL a List.mem_cons_self) init y hq hy
    exact oforIn_inv' P Q f hf L (fun b hb => hL b (List.mem_cons_of_mem _ hb)) r' v hq' h

/-- An invariant of a loop in `Option` whose steps all yield: it holds at the loop's result. -/
theorem oforIn_inv {α σ : Type} {f : α → σ → Option (ForInStep σ)} {L : List α} {init v : σ}
    (h : forIn L init f = some v) (P : α → Prop) (Q : σ → Prop) (hL : ∀ a ∈ L, P a)
    (hf : ∀ a, P a → ∀ r y, Q r → f a r = some y → ∃ r', y = .yield r' ∧ Q r') (hq : Q init) : Q v :=
  oforIn_inv' P Q f hf L hL init v hq h

theorem pos_append {a b : Array SchedRound} (ha : ∀ x ∈ a, 0 < x.1) (hb : ∀ x ∈ b, 0 < x.1) :
    ∀ x ∈ a ++ b, 0 < x.1 := by
  intro x hx
  rcases Array.mem_append.1 hx with h | h
  · exact ha x h
  · exact hb x h

theorem pos_rep {k : Nat} {a : SchedRound} (h : 0 < a.1) : ∀ x ∈ Array.replicate k a, 0 < x.1 := by
  intro x hx
  rw [(Array.mem_replicate.1 hx).2]
  exact h

theorem pos_nil : ∀ x ∈ ([] : List SchedRound), 0 < x.1 := by
  intro x hx
  cases hx

theorem pos_cons {a : SchedRound} {l : List SchedRound} (ha : 0 < a.1) (hl : ∀ x ∈ l, 0 < x.1) :
    ∀ x ∈ a :: l, 0 < x.1 := by
  intro x hx
  rcases List.mem_cons.1 hx with rfl | h
  · exact ha
  · exact hl x h

theorem pos_arr {l : List SchedRound} (h : ∀ x ∈ l, 0 < x.1) : ∀ x ∈ l.toArray, 0 < x.1 := by
  intro x hx
  exact h x (List.mem_toArray.1 hx)

/-- **`fast100`'s schedules** (`Schedule.lean:47-61`), by evaluating it at each `m`: `m ≥ 22`, the table has variables,
and every level draws at least two queries. -/
theorem fast100_facts {m : Nat} {s : Schedule} (h : fast100 m = some s) :
    22 ≤ m ∧ 0 < s.n ∧ 0 < s.levels.size ∧ ∀ l < s.levels.size, 2 ≤ (s.levels.getD l default).q := by
  have hm : m < 40 := by
    refine Decidable.by_contra fun hc => ?_
    have hb : (m < 22 || m > 35) = true := by
      rw [Bool.or_eq_true, decide_eq_true_iff, decide_eq_true_iff]
      omega
    have : fast100 m = none := by
      unfold fast100
      first
      | (rw [ite_eq_left_of_eq_true _ _ (eq_true hb)]; (try rfl); done)
      | (simp [hb]; done)
    rw [this] at h
    cases h
  have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide (22 ≤ m.val ∧ 0 < s.n ∧ 0 < s.levels.size ∧
      ∀ l < s.levels.size, 2 ≤ (s.levels.getD l default).q) = true := by
    decide +kernel
  simpa using key ⟨m, hm⟩ s h

theorem bitLength_pos {q : Nat} (h : 2 ≤ q) : 0 < bitLength (q - 1) := by
  unfold bitLength
  split
  · rename_i he
    first
    | (simp only [beq_iff_eq] at he; omega)
    | (simp at he; omega)
  · omega

/-- **Every round of `repRounds` draws coins** at a claim. -/
theorem repRounds_coins {m kLog claims : Nat} {out : Array SchedRound} (hc : 0 < claims)
    (h : repRounds m kLog claims = some out) : ∀ x ∈ out, 0 < x.1 := by
  unfold repRounds at h
  obtain ⟨sch, hf, h⟩ := obind_some h
  obtain ⟨hm, hn, hl, hq⟩ := fast100_facts hf
  have hq0 := hq 0 hl
  have hb0 := bitLength_pos hq0
  have hqr := hq (sch.levels.size - 1) (by omega)
  have hbr := bitLength_pos hqr
  try dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  obtain ⟨st, hst, h⟩ := obind_some h
  first
  | cases h
  | (simp only [pure, Option.some.injEq] at h; subst h)
  | (simp at h; subst h)
  refine oforIn_inv hst (fun i => i < sch.levels.size - 1) (fun st => ∀ x ∈ st.1, 0 < x.1) ?_ ?_ ?_
  · intro a ha
    show a < sch.levels.size - 1
    first
    | (simp only [List.mem_range'_1, Std.Legacy.Range.size] at ha; omega)
    | (simp [List.mem_range'] at ha; omega)
    | (simp at ha; omega)
  · intro a ha r y hr hy
    have ha' : a < sch.levels.size - 1 := ha
    have hr' : ∀ x ∈ r.1, 0 < x.1 := hr
    have hq1 := hq (a + 1) (by omega)
    have hb1 := bitLength_pos hq1
    try dsimp only at hy
    repeat' (first | (obtain ⟨_, _, hy⟩ := obind_some hy) | (split at hy))
    all_goals first
      | exact absurd hy (by simp)
      | (cases hy
         refine ⟨_, rfl, ?_⟩
         try dsimp only
         repeat' apply pos_append
         all_goals first
           | exact hr'
           | (apply pos_rep; first | decide | (dsimp only; omega) | omega)
           | (apply pos_arr
              repeat' apply pos_cons
              all_goals first | exact pos_nil | decide | (dsimp only; omega) | omega | (dsimp only; assumption)))
  · try dsimp only
    repeat' apply pos_append
    all_goals first
      | (apply pos_rep; first | decide | (dsimp only; omega) | omega)
      | (apply pos_arr
         repeat' apply pos_cons
         all_goals first | exact pos_nil | decide | (dsimp only; omega) | omega | (dsimp only; assumption))

/-- Round `q` of `xs` with a round `z` put at `K` and round `K` replaced by `o` after it. -/
theorem getElem?_insert {xs : Array SchedRound} {K : Nat} {z o : SchedRound} (hK : K < xs.size) (q : Nat) :
    (xs.extract 0 K ++ #[z, o] ++ xs.extract (K + 1) xs.size)[q]? =
      if q < K then xs[q]? else if q = K then some z else if q = K + 1 then some o else xs[q - 1]? := by
  have hA : (xs.extract 0 K).size = K := by
    rw [Array.size_extract]
    omega
  have hAB : (xs.extract 0 K ++ #[z, o]).size = K + 2 := by
    rw [Array.size_append, hA]
    rfl
  by_cases h1 : q < K
  · rw [if_pos h1, Array.getElem?_append_left (by omega), Array.getElem?_append_left (by omega),
      Array.getElem?_extract, if_pos (by omega), Nat.zero_add]
  · rw [if_neg h1]
    by_cases h2 : q = K
    · subst h2
      rw [if_pos rfl, Array.getElem?_append_left (by omega), Array.getElem?_append_right (by omega), hA, Nat.sub_self]
      rfl
    · rw [if_neg h2]
      by_cases h3 : q = K + 1
      · subst h3
        rw [if_pos rfl, Array.getElem?_append_left (by omega), Array.getElem?_append_right (by omega), hA]
        first
        | (rw [show K + 1 - K = 1 by omega]; rfl)
        | simp
      · rw [if_neg h3, Array.getElem?_append_right (by omega), hAB, Array.getElem?_extract]
        by_cases h4 : q - (K + 2) < min xs.size xs.size - (K + 1)
        · rw [if_pos h4, show K + 1 + (q - (K + 2)) = q - 1 by omega]
        · rw [if_neg h4, Array.getElem?_eq_none (by omega)]

/-- **The hidden-unit schedule's coin rounds** (`holoRounds`), at a claim: z_partial's round (`zpRound`) is a round of
it and draws nothing, and every other round draws coins. -/
theorem holoRounds_coins {m kLog claims : Nat} {rs : Array SchedRound} (hc : 0 < claims)
    (h : holoRounds m kLog claims = some rs) :
    zpRound m kLog < rs.size ∧ ∀ q < rs.size, (0 < (rs.getD q default).1 ↔ q ≠ zpRound m kLog) := by
  unfold holoRounds at h
  obtain ⟨out, hout, h⟩ := obind_some h
  have hpos := repRounds_coins hc hout
  try dsimp only at h
  split at h
  · first
    | cases h
    | (obtain ⟨_, h1, _⟩ := obind_some h; cases h1)
    | simp at h
  · rename_i hcond
    have hK : zpRound m kLog < out.size := by
      refine Decidable.by_contra fun hge => ?_
      rw [Array.getElem?_eq_none (by omega)] at hcond
      first
      | exact hcond rfl
      | exact hcond (by decide)
      | simp at hcond
    first
    | cases h
    | (obtain ⟨_, _, h⟩ := obind_some h; cases h)
    | (simp only [pure, Option.some.injEq] at h; subst h)
    have hsz : (out.extract 0 (zpRound m kLog) ++ #[((0, #[], 0, true) : SchedRound), (1, #[], 0, true)] ++
        out.extract (zpRound m kLog + 1) out.size).size = out.size + 1 := by
      rw [Array.size_append, Array.size_append, Array.size_extract, Array.size_extract]
      first
      | (show min (zpRound m kLog) out.size - 0 + 2 + (min out.size out.size - (zpRound m kLog + 1)) = out.size + 1
         omega)
      | (simp; omega)
    refine ⟨by omega, fun q hq => ?_⟩
    rw [hsz] at hq
    rw [Array.getD_eq_getD_getElem?, getElem?_insert hK]
    by_cases h1 : q < zpRound m kLog
    · rw [if_pos h1, Array.getElem?_eq_getElem (by omega), Option.getD_some]
      exact ⟨fun _ => by omega, fun _ => hpos _ (Array.getElem_mem (by omega))⟩
    · rw [if_neg h1]
      by_cases h2 : q = zpRound m kLog
      · rw [if_pos h2, Option.getD_some]
        exact ⟨fun h0 => absurd h0 (Nat.lt_irrefl 0), fun hne => absurd h2 hne⟩
      · rw [if_neg h2]
        by_cases h3 : q = zpRound m kLog + 1
        · rw [if_pos h3, Option.getD_some]
          exact ⟨fun _ => h2, fun _ => Nat.one_pos⟩
        · rw [if_neg h3, Array.getElem?_eq_getElem (by omega), Option.getD_some]
          exact ⟨fun _ => h2, fun _ => hpos _ (Array.getElem_mem (by omega))⟩

/-- **The hidden-unit schedule's coin rounds are its rounds but z_partial's**, at a claim. -/
theorem coinRounds_holo {m kLog claims : Nat} {rs : Array SchedRound} (hc : 0 < claims)
    (h : holoRounds m kLog claims = some rs) :
    coinRounds rs = (List.range rs.size).filter fun q => q != zpRound m kLog := by
  unfold coinRounds
  apply List.filter_congr
  intro q hq
  have hq' : q < rs.size := List.mem_range.1 hq
  have e := (holoRounds_coins hc h).2 q hq'
  show decide (0 < (rs.getD q default).1) = (q != zpRound m kLog)
  by_cases hk : q = zpRound m kLog
  · have h0 : ¬ 0 < (rs.getD q default).1 := fun hp => e.1 hp hk
    rw [decide_eq_false h0, hk, bne_self_eq_false]
  · have h0 : 0 < (rs.getD q default).1 := e.2 hk
    rw [decide_eq_true h0]
    exact (bne_iff_ne.2 hk).symm

/-- `coinRounds_holo` at the verifier's `holoRepRounds` (`RecVStar.lean:640`), which is `holoRounds`. -/
theorem coinRounds_holoRep {m kLog claims : Nat} {rs : Array SchedRound} (hc : 0 < claims)
    (h : holoRepRounds m kLog claims = some rs) :
    coinRounds rs = (List.range rs.size).filter fun q => q != zpRound m kLog :=
  coinRounds_holo hc h

#print axioms fast100_facts
#print axioms repRounds_coins
#print axioms getElem?_insert
#print axioms holoRounds_coins
#print axioms coinRounds_holo
#print axioms coinRounds_holoRep

end Flock.Guarantees.HoloRepCoins
