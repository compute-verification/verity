import Proofs.Flock.C3.Reps.StepsTie

/-!
# `RepsMsgsRead` at `repsOfVals`

The reps' part of a run's steps is the cap, the points' coin, then rep 0's steps in the reps' interleaving and rep 1's
(`repPart_eq`). So its round 0 is the cap, its round `j + 1` is rep 0's round `j` for `j` up to rep 0's coins `c₀` (at
`j = c₀` followed by rep 1's round 0), and its round `c₀ + 1 + j` is rep 1's round `j` (`repsTape_round`).
`repsMsgsRead_of_read`: `RepsMsgsRead` at a tied source from the cap's read on stage 0's commit strings (`CapRead`) and
each rep's rounds' read on their stages' (`TsRead`). At `repsOfVals` each rep's steps are the values' weave, so there
`TsRead` is a read fact of `vals` on the layers (`repsMsgsRead_repsOfVals`).
-/

namespace FlockC3.Reps

noncomputable section

open FlockSoundness.Game (Msg Step)
open FlockSoundness.Discharge.PlaysCoin (groups consHead groups_ne_nil cut_kinds)
open FlockVBridge.InnerAccepts (firstSteps)

/-! ## A tape's rounds -/

/-- A tape's coins. -/
abbrev nCoins (l : List Step) : ℕ := (l.filterMap Sum.getRight?).length

theorem cutOf_eq_groups (l : List Step) : cutOf l = groups l := cut_kinds l

theorem consHead_succ (m : Msg) {gs : List (List Msg)} (h : gs ≠ []) (j : ℕ) :
    (consHead m gs)[j + 1]? = gs[j + 1]? := by
  rcases gs with _ | ⟨g, gs⟩
  · exact absurd rfl h
  · rfl

theorem consHead_zero (m : Msg) {gs : List (List Msg)} (h : gs ≠ []) :
    (consHead m gs)[0]? = gs[0]?.map (m :: ·) := by
  rcases gs with _ | ⟨g, gs⟩
  · exact absurd rfl h
  · rfl

theorem groups_append_lt (b a : List Step) (i : ℕ) (hi : i < nCoins a) : (groups (a ++ b))[i]? = (groups a)[i]? := by
  rw [← cutOf_eq_groups, ← cutOf_eq_groups]
  exact cutOf_append b a i hi

/-- A round past a tape's last coin, in the tape after it. -/
theorem groups_append_succ (b : List Step) : ∀ (a : List Step) (j : ℕ),
    (groups (a ++ b))[nCoins a + j + 1]? = (groups b)[j + 1]?
  | [], j => by
      show (groups b)[0 + j + 1]? = _
      rw [Nat.zero_add]
  | .inl m :: a, j => by
      show (consHead m (groups (a ++ b)))[nCoins a + j + 1]? = _
      rw [consHead_succ m (groups_ne_nil _)]
      exact groups_append_succ b a j
  | .inr c :: a, j => by
      show ([] :: groups (a ++ b))[nCoins a + 1 + j + 1]? = _
      rw [List.getElem?_cons_succ, Nat.add_right_comm]
      exact groups_append_succ b a j

/-- The round at a tape's last coin: its last group, then the first of the tape after it. -/
theorem groups_append_coins (b : List Step) : ∀ a : List Step,
    (groups (a ++ b))[nCoins a]? = ((groups a)[nCoins a]?).bind fun g => ((groups b)[0]?).map (g ++ ·)
  | [] => by
      show (groups b)[0]? = _
      cases (groups b)[0]? <;> rfl
  | .inl m :: a => by
      show (consHead m (groups (a ++ b)))[nCoins a]? = ((consHead m (groups a))[nCoins a]?).bind _
      have ih := groups_append_coins b a
      generalize nCoins a = n at ih ⊢
      rcases n with _ | n
      · rw [consHead_zero m (groups_ne_nil _), consHead_zero m (groups_ne_nil _), ih]
        cases (groups a)[0]? <;> cases (groups b)[0]? <;> rfl
      · rw [consHead_succ m (groups_ne_nil _), consHead_succ m (groups_ne_nil _), ih]
  | .inr c :: a => by
      show ([] :: groups (a ++ b))[nCoins a + 1]? = (([] :: groups a)[nCoins a + 1]?).bind _
      rw [List.getElem?_cons_succ, List.getElem?_cons_succ]
      exact groups_append_coins b a

/-- A round's messages in the reps' interleaving: each after the step that picks rep 0. -/
def mark (g : List Msg) : List Msg := g.flatMap fun m => [⟨Bool, true⟩, m]

theorem nCoins_firstSteps : ∀ w : List Step, nCoins (firstSteps w) = nCoins w
  | [] => rfl
  | .inl _ :: w => nCoins_firstSteps w
  | .inr _ :: w => congrArg (· + 1) (nCoins_firstSteps w)

/-- **Rep 0's rounds in the reps' interleaving**: each marked, and one closing at a coin then picks rep 0 for it. -/
theorem groups_firstSteps : ∀ (w : List Step) (j : ℕ), (groups (firstSteps w))[j]? =
    ((groups w)[j]?).map fun g => if j < nCoins w then mark g ++ [⟨Bool, true⟩] else mark g
  | [], j => by cases j <;> rfl
  | .inl m :: w, j => by
      show (consHead _ (consHead m (groups (firstSteps w))))[j]? = ((consHead m (groups w))[j]?).map
        fun g => if j < nCoins w then mark g ++ [⟨Bool, true⟩] else mark g
      rcases j with _ | j
      · rw [consHead_zero _ (by cases groups (firstSteps w) <;> exact List.cons_ne_nil _ _),
          consHead_zero m (groups_ne_nil _), consHead_zero m (groups_ne_nil _), groups_firstSteps w 0]
        cases (groups w)[0]? with
        | none => rfl
        | some g => by_cases h : 0 < nCoins w <;> simp only [Option.map_some, h, ↓reduceIte] <;> rfl
      · rw [consHead_succ _ (by cases groups (firstSteps w) <;> exact List.cons_ne_nil _ _),
          consHead_succ m (groups_ne_nil _), consHead_succ m (groups_ne_nil _)]
        exact groups_firstSteps w (j + 1)
  | .inr c :: w, j => by
      show ([⟨Bool, true⟩] :: groups (firstSteps w))[j]? = (([] :: groups w)[j]?).map _
      rcases j with _ | j
      · show some _ = some (if 0 < nCoins w + 1 then _ else _)
        simp only [Nat.zero_lt_succ, ↓reduceIte]
        rfl
      · rw [List.getElem?_cons_succ, List.getElem?_cons_succ, groups_firstSteps w j]
        show _ = Option.map (fun g => if j + 1 < nCoins w + 1 then _ else _) _
        simp only [Nat.add_lt_add_iff_right]

/-! ## The reps' part's rounds -/

/-- The reps' part from its pieces: the cap, the points' coin, then rep 0's steps in the reps' interleaving and
rep 1's. -/
def repsTape (c p : Msg) (w₀ w₁ : List Step) : List Step := .inl c :: .inr p :: (firstSteps w₀ ++ w₁)

/-- **Round `i` of the reps' part, from the reps' steps `w` and `w'`**: rep 0's have as many coins `c₀`, rep 0's round
`j` agrees where `i = j + 1` (for `j ≤ c₀`), and rep 1's round `j` where `i = c₀ + 1 + j`. -/
def RoundsAgree (i : ℕ) (w w' : Fin 2 → List Step) : Prop :=
  nCoins (w 0) = nCoins (w' 0) ∧ (∀ j, i = j + 1 → j ≤ nCoins (w 0) → (groups (w 0))[j]? = (groups (w' 0))[j]?) ∧
    ∀ j, i = nCoins (w 0) + 1 + j → (groups (w 1))[j]? = (groups (w' 1))[j]?

theorem repsTape_round {c c' p p' : Msg} {w w' : Fin 2 → List Step} {i : ℕ} (hc : i = 0 → c = c')
    (h : RoundsAgree i w w') :
    (groups (repsTape c p (w 0) (w 1)))[i]? = (groups (repsTape c' p' (w' 0) (w' 1)))[i]? := by
  obtain ⟨hn, h₀, h₁⟩ := h
  show ([c] :: groups (firstSteps (w 0) ++ w 1))[i]? = ([c'] :: groups (firstSteps (w' 0) ++ w' 1))[i]?
  rcases i with _ | i
  · rw [List.getElem?_cons_zero, List.getElem?_cons_zero, hc rfl]
  rw [List.getElem?_cons_succ, List.getElem?_cons_succ]
  have e₀ := nCoins_firstSteps (w 0)
  have e₀' := nCoins_firstSteps (w' 0)
  rcases Nat.lt_or_ge i (nCoins (w 0)) with hi | hi
  · rw [groups_append_lt _ _ _ (by omega), groups_append_lt _ _ _ (by omega), groups_firstSteps,
      groups_firstSteps, h₀ i rfl (Nat.le_of_lt hi), hn]
  rcases Nat.eq_or_lt_of_le hi with hi | hi
  · subst hi
    have b := groups_append_coins (w 1) (firstSteps (w 0))
    have b' := groups_append_coins (w' 1) (firstSteps (w' 0))
    rw [e₀] at b
    rw [e₀', ← hn] at b'
    rw [b, b', groups_firstSteps, groups_firstSteps, h₀ _ rfl (Nat.le_refl _), hn, h₁ 0 (by omega)]
  · obtain ⟨j, rfl⟩ : ∃ j, i = nCoins (w 0) + j + 1 := ⟨i - nCoins (w 0) - 1, by omega⟩
    have s := groups_append_succ (w 1) (firstSteps (w 0)) j
    have s' := groups_append_succ (w' 1) (firstSteps (w' 0)) j
    rw [e₀] at s
    rw [e₀', ← hn] at s'
    rw [s, s', h₁ (j + 1) (by omega)]

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf maskOf)
open FlockC3.Compose

theorem repPart_eq {e : ExecHolo} (u : C3Run e) : repPart u = repsTape ⟨_, u.cap₀⟩ ⟨_, u.pts⟩ (u.ts 0) (u.ts 1) :=
  rfl

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}
  {zs : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
  {dg : ByteArray → ByteArray} {N k Rw : ℕ}
  {stp : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
  {sh : Flock.Shape} {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
  {vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
  {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}

local notation "ℐ" => holoInnerAt T CT st hp

/-! ## The read facts -/

/-- `P` at two accepted outer proofs' histories and layers, where the layers agree on the read wires of stage `i`'s
commit strings. -/
def ReadAt (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (own : Fin S → Array Flock.Registered.Port)
    (sel : ℕ → String → ℕ → Prop) (i : ℕ)
    (P : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) →
      Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Prop) : Prop :=
  ∀ R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) i (layers zs dg N k Rw R' ω' t' R₂' τ') →
    P R ω t (layers zs dg N k Rw R ω t R₂ τ) R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ')

/-- **Hypothesis: the level-0 cap is read on stage 0's commit strings.** -/
def CapRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ReadAt zs dg N k Rw own sel 0 fun R ω t Lb R' ω' t' Lb' => (src R ω t Lb).cap₀ = (src R' ω' t' Lb').cap₀

/-- **Hypothesis: each rep's rounds are read on their stages' commit strings**: rep 0's round `j` on stage `j + 1`'s,
and rep 1's round `j` on stage `c₀ + 1 + j`'s, where rep 0's coins `c₀` agree. -/
def TsRead (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ i, ReadAt zs dg N k Rw own sel i fun R ω t Lb R' ω' t' Lb' => RoundsAgree i (src R ω t Lb).ts (src R' ω' t' Lb').ts

/-- **`RepsMsgsRead` at a tied source**, from the cap's read and the reps' rounds'. -/
theorem repsMsgsRead_of_read
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hc : CapRead zs dg N k Rw src own sel) (h : TsRead zs dg N k Rw src own sel) :
    RepsMsgsRead zs dg N k Rw (tiedSrc src) own sel := by
  intro i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX _ _
  rw [cutOf_eq_groups, cutOf_eq_groups, repPart_eq, repPart_eq]
  exact repsTape_round (fun hi => by
    subst hi; exact congrArg (fun a => (⟨_, a⟩ : Msg)) (hc R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX))
    (h i R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX)

/-- **`RepsMsgsRead` at the source that reads its reps off the values**, from `stp`'s cap read and the values' weave's
read on the layers: `TsRead` at `repsOfVals`, where rep `r`'s steps are `unitWeave` of `rep Lb r` and `vals Lb r`. -/
theorem repsMsgsRead_repsOfVals (hc : CapRead zs dg N k Rw stp own sel)
    (h : TsRead zs dg N k Rw (repsOfVals stp sh rep vals) own sel) :
    RepsMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel :=
  repsMsgsRead_of_read hc h

end

end FlockC3.Reps
