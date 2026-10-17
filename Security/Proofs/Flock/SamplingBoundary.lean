import Definitions.Claims.Sampling

/-!
# The sampling block's faulty set is the boundary form's

`Claims.Sampling.faultyBoundary : Claims.Sampling.FaultyBoundary`. On a topologically ordered program, a replay unit
whose proof units are consecutive gate intervals has an interior making every proof unit correct exactly when the gate
interval it spans is correct on the committed values. One way, the units' checks carry the interval's computation unit
by unit: every value a later unit reads is committed, and the one the interval computes (`fwd_chain`). The other way,
the interval's own computed values are an interior (`ι g = runAt val lo g`).
-/

namespace Claims.Sampling.Boundary

open Claims.Sampling

/-- Consecutive non-empty gate intervals, from `a` to `e`. -/
def ChainFrom : ℕ → List (ℕ × ℕ) → ℕ → Prop
  | a, [], e => a = e
  | a, u :: us, e => u.1 = a ∧ u.1 < u.2 ∧ ChainFrom u.2 us e

theorem chainFrom_le : ∀ {a e : ℕ} {us : List (ℕ × ℕ)}, ChainFrom a us e → a ≤ e
  | _, _, [], h => le_of_eq h
  | _, _, _ :: _, ⟨hua, huv, hch⟩ => by have := chainFrom_le hch; omega

theorem chainFrom_bounds : ∀ {a e : ℕ} {us : List (ℕ × ℕ)}, ChainFrom a us e → ∀ u ∈ us, a ≤ u.1 ∧ u.2 ≤ e
  | _, _, [], _, _, hu => absurd hu List.not_mem_nil
  | _, _, u :: _, ⟨hua, huv, hch⟩, u', hu' => by
    rcases List.mem_cons.1 hu' with rfl | hu'
    · exact ⟨le_of_eq hua.symm, chainFrom_le hch⟩
    · have := chainFrom_bounds hch u' hu'
      omega

theorem chainFrom_cover : ∀ {a e : ℕ} {us : List (ℕ × ℕ)}, ChainFrom a us e →
    ∀ g, a ≤ g → g < e → ∃ u ∈ us, u.1 ≤ g ∧ g < u.2
  | _, _, [], h, _, h1, h2 => by simp only [ChainFrom] at h; omega
  | _, _, u :: _, ⟨hua, _, hch⟩, g, h1, h2 => by
    by_cases hg : g < u.2
    · exact ⟨u, List.mem_cons_self .., by omega, hg⟩
    · obtain ⟨u', hu', h3, h4⟩ := chainFrom_cover hch g (by omega) h2
      exact ⟨u', List.mem_cons_of_mem _ hu', h3, h4⟩

theorem consecutive_chain : ∀ (r : RU), RU.Consecutive r → ChainFrom (RU.lo r) r (RU.hi r)
  | [], h => absurd rfl h.1
  | u :: rest, h => by
    induction rest generalizing u with
    | nil => exact ⟨rfl, h.2.1 u (by simp), rfl⟩
    | cons v rest ih =>
      have huv : u.2 = v.1 := h.2.2 0 (by simp)
      have h' : RU.Consecutive (v :: rest) :=
        ⟨by simp, fun w hw => h.2.1 w (List.mem_cons_of_mem _ hw), fun i hi => by
          have := h.2.2 (i + 1) (by simp only [List.length_cons] at hi ⊢; omega)
          simp only [List.getElem_cons_succ] at this ⊢
          exact this⟩
      have hhi : RU.hi (u :: v :: rest) = RU.hi (v :: rest) := by simp [RU.hi]
      refine ⟨rfl, h.2.1 u (by simp), ?_⟩
      rw [hhi, huv]
      exact ih v h'

variable (C : Prog)

theorem runAt_eq (ref : ℕ → Option Bool) (lo g : ℕ) :
    C.runAt ref lo g = C.gates[g]?.bind fun x =>
      x.val (fun i => if i < lo then ref i else if i < g then C.runAt ref lo i else none) (ref g) := by
  rw [Prog.runAt]
  simp only [dite_eq_ite]

theorem mapM_congr' {f f' : ℕ → Option Bool} : ∀ (ws : List ℕ), (∀ i ∈ ws, f i = f' i) → ws.mapM f = ws.mapM f'
  | [], _ => rfl
  | w :: ws, h => by
    simp only [List.mapM_cons, h w List.mem_cons_self, mapM_congr' ws fun i hi => h i (List.mem_cons_of_mem _ hi)]

theorem mapM_isSome {f : ℕ → Option Bool} : ∀ (ws : List ℕ), (∀ i ∈ ws, (f i).isSome) → (ws.mapM f).isSome
  | [], _ => rfl
  | w :: ws, h => by
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (h w List.mem_cons_self)
    obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.1 (mapM_isSome ws fun i hi => h i (List.mem_cons_of_mem _ hi))
    simp [List.mapM_cons, hb, hbs]

theorem val_congr (x : BGate) {rd rd' : ℕ → Option Bool} (own : Option Bool) (h : ∀ i ∈ x.reads, rd i = rd' i) :
    x.val rd own = x.val rd' own := by
  cases x with
  | input => rfl
  | const b => rfl
  | not i => simp only [BGate.val, h i (by simp [BGate.reads])]
  | and ws => simp only [BGate.val, mapM_congr' ws h]
  | xor ws => simp only [BGate.val, mapM_congr' ws h]

theorem val_isSome (x : BGate) (hx : x ≠ .input) {rd : ℕ → Option Bool} (own : Option Bool)
    (h : ∀ i ∈ x.reads, (rd i).isSome) : (x.val rd own).isSome := by
  cases x with
  | input => exact absurd rfl hx
  | const b => rfl
  | not i => simpa [BGate.val] using h i (by simp [BGate.reads])
  | and ws => simpa [BGate.val] using mapM_isSome ws h
  | xor ws => simpa [BGate.val] using mapM_isSome ws h

theorem boundary_input {lo hi g : ℕ} (h : C.gates[g]? = some .input) : C.Boundary lo hi g = true := by
  simp [Prog.Boundary, h]

theorem boundary_mono {lo hi a b g : ℕ} (ha : lo ≤ a) (hb : b ≤ hi) (h : C.Boundary lo hi g = true) :
    C.Boundary a b g = true := by
  simp only [Prog.Boundary, Bool.or_eq_true, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h ⊢
  rcases h with (h | h) | ⟨g', hg', h1, h2⟩
  · exact Or.inl (Or.inl h)
  · exact Or.inl (Or.inr h)
  · exact Or.inr ⟨g', hg', by omega, h2⟩

theorem boundary_of_read {a b g i : ℕ} {x : BGate} (hg : C.gates[g]? = some x) (hi : i ∈ x.reads)
    (hout : g < a ∨ b ≤ g) : C.Boundary a b i = true := by
  have hlen : g < C.gates.length := by
    rcases Nat.lt_or_ge g C.gates.length with h | h
    · exact h
    · simp [List.getElem?_eq_none h] at hg
  simp only [Prog.Boundary, Bool.or_eq_true, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq]
  exact Or.inr ⟨g, List.mem_range.2 hlen, hout, by simp [hg, hi]⟩

theorem runAt_congr {ρ ρ' : ℕ → Option Bool} {lo : ℕ} (h1 : ∀ i, i < lo → ρ i = ρ' i)
    (h2 : ∀ g, C.gates[g]? = some .input → ρ g = ρ' g) : ∀ g, C.runAt ρ lo g = C.runAt ρ' lo g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    rw [runAt_eq, runAt_eq]
    cases hx : C.gates[g]? with
    | none => rfl
    | some x =>
      simp only [Option.bind_some]
      have hrd : (fun i => if i < lo then ρ i else if i < g then C.runAt ρ lo i else none) =
          (fun i => if i < lo then ρ' i else if i < g then C.runAt ρ' lo i else none) := by
        funext i
        by_cases hl : i < lo
        · simp [hl, h1 i hl]
        · by_cases hg : i < g
          · simp [hl, hg, ih i hg]
          · simp [hl, hg]
      rw [hrd]
      cases x
      · simp [BGate.val, h2 g hx]
      all_goals simp [BGate.val]

/-- Every value from `lo` and before `a` that a gate at or past `a` reads is the one the interval from `lo`
computes. -/
def Inv (ρ : ℕ → Option Bool) (lo a : ℕ) : Prop :=
  ∀ g (x : BGate) i, a ≤ g → C.gates[g]? = some x → i ∈ x.reads → lo ≤ i → i < a → ρ i = C.runAt ρ lo i

theorem runAt_shift {ρ : ℕ → Option Bool} {lo a : ℕ} (hla : lo ≤ a) (hinv : Inv C ρ lo a) :
    ∀ g, a ≤ g → C.runAt ρ a g = C.runAt ρ lo g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    intro hag
    rw [runAt_eq, runAt_eq]
    cases hx : C.gates[g]? with
    | none => rfl
    | some x =>
      simp only [Option.bind_some]
      apply val_congr
      intro i hi
      by_cases h1 : i < lo
      · simp [h1, show i < a by omega]
      · by_cases h2 : i < a
        · simp only [h1, h2, show i < g by omega, ite_true, ite_false]
          exact hinv g x i hag hx hi (by omega) h2
        · by_cases h3 : i < g
          · simp only [h1, h2, h3, ite_true, ite_false]
            exact ih i h3 (by omega)
          · simp [h1, h2, h3]

theorem fwd_chain {ρ : ℕ → Option Bool} {lo : ℕ} :
    ∀ (us : List (ℕ × ℕ)) (a e : ℕ), ChainFrom a us e → lo ≤ a → Inv C ρ lo a →
      (∀ u ∈ us, C.Correct ρ u.1 u.2) → ∀ u ∈ us, Inv C ρ lo u.1
  | [], _, _, _, _, _, _, u, hu => absurd hu List.not_mem_nil
  | u :: us, a, e, ⟨hua, huv, hch⟩, hla, hinv, hcor, u', hu' => by
    have hinv1 : Inv C ρ lo u.1 := by rw [hua]; exact hinv
    have hinv2 : Inv C ρ lo u.2 := by
      intro g x i hg hx hi hli hiu
      by_cases hia : i < u.1
      · exact hinv1 g x i (by omega) hx hi hli hia
      · have hb : C.Boundary u.1 u.2 i = true := boundary_of_read C hx hi (Or.inr hg)
        obtain ⟨b, hρ, hrun⟩ := ((hcor u List.mem_cons_self) i (by omega) hiu).2 hb
        rw [hρ, ← runAt_shift C (by omega) hinv1 i (by omega), hrun]
    rcases List.mem_cons.1 hu' with rfl | hu'
    · exact hinv1
    · exact fwd_chain us u.2 e hch (by omega) hinv2 (fun v hv => hcor v (List.mem_cons_of_mem _ hv)) u' hu'

/-- The values a replay unit spanning `[lo, hi)` reads and writes, its interior `ι` (`RU.ref`). -/
def refI (lo hi : ℕ) (val : ℕ → Option Bool) (ι : ℕ → Bool) (g : ℕ) : Option Bool :=
  if lo ≤ g ∧ g < hi ∧ C.Boundary lo hi g = false then some (ι g) else val g

theorem refI_lt {lo hi g : ℕ} (val : ℕ → Option Bool) (ι : ℕ → Bool) (h : g < lo) : refI C lo hi val ι g = val g := by
  rw [refI, ite_eq_right (by omega)]

theorem refI_boundary {lo hi g : ℕ} (val : ℕ → Option Bool) (ι : ℕ → Bool) (h : C.Boundary lo hi g = true) :
    refI C lo hi val ι g = val g := by
  rw [refI, ite_eq_right (by simp [h])]

theorem runAt_refI (lo hi : ℕ) (val : ℕ → Option Bool) (ι : ℕ → Bool) (g : ℕ) :
    C.runAt (refI C lo hi val ι) lo g = C.runAt val lo g :=
  runAt_congr C (fun _ h => refI_lt C val ι h) (fun _ h => refI_boundary C val ι (boundary_input C h)) g

/-- **The core**: proof units chained over `[lo, hi)` have an interior making each correct exactly when the gate
interval is correct on the committed values. -/
theorem core (htop : C.Topological) {lo hi : ℕ} {us : List (ℕ × ℕ)} (hch : ChainFrom lo us hi)
    (hlen : hi ≤ C.gates.length) (val : ℕ → Option Bool) :
    (∃ ι : ℕ → Bool, ∀ u ∈ us, C.Correct (refI C lo hi val ι) u.1 u.2) ↔ C.Correct val lo hi := by
  constructor
  · rintro ⟨ι, hcor⟩
    have hinv0 : Inv C (refI C lo hi val ι) lo lo := fun _ _ _ _ _ _ h1 h2 => absurd h2 (by omega)
    have hunits := fwd_chain C us lo hi hch le_rfl hinv0 hcor
    intro g hlg hgh
    obtain ⟨u, hu, hug, hgu⟩ := chainFrom_cover hch g hlg hgh
    obtain ⟨hlu, huh⟩ := chainFrom_bounds hch u hu
    have hc := hcor u hu g hug hgu
    refine ⟨fun x hx i hi hil => ?_, fun hb => ?_⟩
    · have := hc.1 x hx i hi (by omega)
      rwa [refI_lt C val ι hil] at this
    · obtain ⟨b, hρ, hrun⟩ := hc.2 (boundary_mono C hlu huh hb)
      refine ⟨b, ?_, ?_⟩
      · rwa [refI_boundary C val ι hb] at hρ
      · rw [← runAt_refI C lo hi val ι, ← runAt_shift C hlu (hunits u hu) g hug, hrun]
  · intro hc
    have hD : ∀ g, lo ≤ g → g < hi → ∃ b, C.runAt val lo g = some b := by
      intro g
      induction g using Nat.strong_induction_on with
      | _ g ih =>
        intro hlg hgh
        have hgl : g < C.gates.length := by omega
        have hx : C.gates[g]? = some C.gates[g] := List.getElem?_eq_getElem hgl
        rw [runAt_eq, hx, Option.bind_some]
        by_cases hin : C.gates[g] = .input
        · obtain ⟨b, hb, -⟩ := (hc g hlg hgh).2 (boundary_input C (hin ▸ hx))
          exact ⟨b, by rw [hin]; simp [BGate.val, hb]⟩
        · apply Option.isSome_iff_exists.1
          apply val_isSome _ hin
          intro i hi
          have hig : i < g := htop.1 g _ hx i hi
          by_cases hil : i < lo
          · simp only [hil, ite_true]
            exact Option.ne_none_iff_isSome.1 ((hc g hlg hgh).1 _ hx i hi hil)
          · obtain ⟨b, hb⟩ := ih i hig (by omega) (by omega)
            simp [hil, hig, hb]
    let ι : ℕ → Bool := fun g => (C.runAt val lo g).getD false
    have hE : ∀ i, lo ≤ i → i < hi → refI C lo hi val ι i = C.runAt val lo i := by
      intro i hli hih
      by_cases hb : C.Boundary lo hi i = true
      · obtain ⟨b, hv, hrun⟩ := (hc i hli hih).2 hb
        rw [refI_boundary C val ι hb, hv, hrun]
      · obtain ⟨b, hrun⟩ := hD i hli hih
        rw [refI, ite_eq_left ⟨hli, hih, by simpa using hb⟩, hrun]
        simp [ι, hrun]
    have hinv : ∀ a, a ≤ hi → Inv C (refI C lo hi val ι) lo a := fun a hah _ _ i _ _ _ hli hia => by
      rw [hE i hli (by omega), runAt_refI]
    refine ⟨ι, fun u hu g hug hgu => ?_⟩
    obtain ⟨hlu, huh⟩ := chainFrom_bounds hch u hu
    refine ⟨fun x hx i hi hiu => ?_, fun _ => ?_⟩
    · by_cases hil : i < lo
      · rw [refI_lt C val ι hil]
        exact (hc g (by omega) (by omega)).1 x hx i hi hil
      · obtain ⟨b, hb⟩ := hD i (by omega) (by omega)
        rw [hE i (by omega) (by omega), hb]
        simp
    · obtain ⟨b, hb⟩ := hD g (by omega) (by omega)
      refine ⟨b, by rw [hE g (by omega) (by omega), hb], ?_⟩
      rw [runAt_shift C hlu (hinv u.1 (by omega)) g hug, runAt_refI, hb]

end Claims.Sampling.Boundary

open Claims.Sampling.Boundary in
/-- **The faulty set is the boundary form's** (`Claims.Sampling.FaultyBoundary`). -/
theorem Claims.Sampling.faultyBoundary : Claims.Sampling.FaultyBoundary := by
  intro H Cm s P w v r htop hr hcons hlen
  simp only [Claims.Sampling.Faulty, Claims.Sampling.BoundaryFaulty, hr, Option.some.injEq, exists_eq_left']
  exact not_congr (core s.prog htop (consecutive_chain r hcons) hlen _)
