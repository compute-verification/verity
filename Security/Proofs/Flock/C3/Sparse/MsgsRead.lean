import Proofs.Flock.C3.Sparse.SrcLayer
import Proofs.Flock.C3.Reps.RepsRead
import Proofs.Flock.C3.Compose.LastRead

/-!
# The sparse phases' committed rounds, from each message's stage

A sparse phase's steps weave its messages along `sparseKinds` (`stepsOf`), so its round `j` is its messages of round
`j` (`roundOf`: the coins before a message), whatever its witness and coins (`groups_weave`, `groups_stepsOf`); rep 0's
phase has `nS` coins whatever they are (`nCoins_stepsOf`). The sparse stage is rep 0's phase then rep 1's, so its round
`j` is rep 0's round `j` and rep 1's round `j − nS` (`groups_stage`).

`sparseStageRead_of_layers`: the stage's committed rounds past `c₀` coins (H's `StageMsgsRead` at `sp 0 ++ sp 1`,
`SparseStageRead` here) where each phase's steps are `stepsOf` (`SparseTieL`'s) and its message `m` is read off V*'s
layers on stage `c₀ + i · nS + roundOf m`'s commit strings (`SparseMsgsOfLayers`, named). `sparseMsgsRead_of_layers`:
`SparseMsgsRead`, through H's `sparseMsgsRead_of_stage` (`PartRead.lean`), taken in its form.
-/

namespace FlockC3.Sparse

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (groups consHead groups_ne_nil)
open FlockC3.Reps (nCoins cutOf cutOf_eq_groups consHead_succ groups_append_lt groups_append_succ groups_append_coins)
open FlockVBridge.Family (skOf maskOf)
open FlockVBridge.Sparse (phaseMsgs coinTape vtTape msgTape nc)

local notation "𝔽" => FlockLevel3.GF128

/-! ## A weave's rounds -/

/-- The round of message `m` along the kinds `σ`: the coins before it. -/
def roundOf : List Kind → ℕ → ℕ
  | [], _ => 0
  | .msg _ :: _, 0 => 0
  | .msg _ :: σ, m + 1 => roundOf σ m
  | .coin _ :: σ, m => roundOf σ m + 1

/-- **A weave's coins are fixed by its lengths.** -/
theorem nCoins_weave : ∀ (σ : List Kind) {ms ms' cs cs' : List Msg}, ms.length = ms'.length →
    cs.length = cs'.length → nCoins (weave σ ms cs) = nCoins (weave σ ms' cs')
  | [], _, _, _, _, _, _ => rfl
  | .msg _ :: _, [], [], _, _, _, _ => rfl
  | .msg _ :: σ, _ :: _, _ :: _, _, _, hm, hc => nCoins_weave σ (Nat.succ.inj hm) hc
  | .coin _ :: _, _, _, [], [], _, _ => rfl
  | .coin _ :: σ, _, _, _ :: _, _ :: _, hm, hc => congrArg (· + 1) (nCoins_weave σ hm (Nat.succ.inj hc))
  | .msg _ :: _, [], _ :: _, _, _, hm, _ => absurd hm (by simp)
  | .msg _ :: _, _ :: _, [], _, _, hm, _ => absurd hm (by simp)
  | .coin _ :: _, _, _, [], _ :: _, _, hc => absurd hc (by simp)
  | .coin _ :: _, _, _, _ :: _, [], _, hc => absurd hc (by simp)

/-- **A weave's round is its messages of that round**: two weaves along `σ` of messages and coins of the same lengths
have the same round `j` where their messages of round `j` agree. -/
theorem groups_weave : ∀ (σ : List Kind) {ms ms' cs cs' : List Msg} (j : ℕ), ms.length = ms'.length →
    cs.length = cs'.length → (∀ m, roundOf σ m = j → ms[m]? = ms'[m]?) →
    (groups (weave σ ms cs))[j]? = (groups (weave σ ms' cs'))[j]?
  | [], _, _, _, _, _, _, _, _ => rfl
  | .msg _ :: _, [], [], _, _, _, _, _, _ => rfl
  | .msg _ :: σ, m :: ms, m' :: ms', _, _, j, hm, hc, h => by
      have ih := groups_weave σ (ms := ms) (ms' := ms') j (Nat.succ.inj hm) hc fun k hk => h (k + 1) hk
      show (consHead m _)[j]? = (consHead m' _)[j]?
      rcases j with _ | j
      · obtain rfl : m = m' := Option.some.inj (h 0 rfl)
        revert ih
        cases groups (weave σ ms _) <;> cases groups (weave σ ms' _) <;> simp [consHead]
      · rw [consHead_succ m (groups_ne_nil _), consHead_succ m' (groups_ne_nil _), ih]
  | .coin _ :: _, _, _, [], [], _, _, _, _ => rfl
  | .coin _ :: σ, ms, ms', _ :: cs, _ :: cs', j, hm, hc, h => by
      show ([] :: _)[j]? = ([] :: _)[j]?
      rcases j with _ | j
      · rfl
      · exact groups_weave σ j hm (Nat.succ.inj hc) fun k hk => h k (congrArg (· + 1) hk)
  | .msg _ :: _, [], _ :: _, _, _, _, hm, _, _ => absurd hm (by simp)
  | .msg _ :: _, _ :: _, [], _, _, _, hm, _, _ => absurd hm (by simp)
  | .coin _ :: _, _, _, [], _ :: _, _, _, hc, _ => absurd hc (by simp)
  | .coin _ :: _, _, _, _ :: _, [], _, _, hc, _ => absurd hc (by simp)

/-! ## A sparse phase's rounds -/

theorem length_phaseMsgs (sh : SparseShape) (w w' : Array Flock.F128) :
    (phaseMsgs sh w).length = (phaseMsgs sh w').length := by
  have hv : (vtTape sh w).length = (vtTape sh w').length := by unfold vtTape; split_ifs <;> rfl
  simp only [phaseMsgs, msgTape, List.length_append, List.length_map, List.length_cons, List.length_nil, hv]

theorem length_coins (sh : SparseShape) (cs cs' : Array Flock.F128) :
    ((coinTape cs 0 (nc sh)).map (PlaysCoin.coinArr fOf)).length =
      ((coinTape cs' 0 (nc sh)).map (PlaysCoin.coinArr fOf)).length := by
  simp only [coinTape, List.length_map]

/-- A sparse phase's coins, whatever its witness and coins. -/
def nS (sh : SparseShape) : ℕ := nCoins (stepsOf sh #[] #[])

theorem nCoins_stepsOf (sh : SparseShape) (w cs : Array Flock.F128) : nCoins (stepsOf sh w cs) = nS sh :=
  nCoins_weave _ (length_phaseMsgs sh w #[]) (length_coins sh cs #[])

theorem groups_stepsOf {sh : SparseShape} {w cs w' cs' : Array Flock.F128} {j : ℕ}
    (h : ∀ m, roundOf (sparseKinds 𝔽 sh) m = j → (phaseMsgs sh w)[m]? = (phaseMsgs sh w')[m]?) :
    (groups (stepsOf sh w cs))[j]? = (groups (stepsOf sh w' cs'))[j]? :=
  groups_weave _ j (length_phaseMsgs sh w w') (length_coins sh cs cs') h

/-- **The sparse stage's round `j`** is rep 0's phase's round `j` and rep 1's round `j − nS`: two stages agree there
where rep 0's messages of round `j` agree and rep 1's of round `j − nS` (for `nS ≤ j`). -/
theorem groups_stage {sh : SparseShape} {w cs w' cs' : Fin 2 → Array Flock.F128} {j : ℕ}
    (h₀ : ∀ m, roundOf (sparseKinds 𝔽 sh) m = j → (phaseMsgs sh (w 0))[m]? = (phaseMsgs sh (w' 0))[m]?)
    (h₁ : ∀ m, nS sh + roundOf (sparseKinds 𝔽 sh) m = j →
      (phaseMsgs sh (w 1))[m]? = (phaseMsgs sh (w' 1))[m]?) :
    (groups (stepsOf sh (w 0) (cs 0) ++ stepsOf sh (w 1) (cs 1)))[j]? =
      (groups (stepsOf sh (w' 0) (cs' 0) ++ stepsOf sh (w' 1) (cs' 1)))[j]? := by
  have e := nCoins_stepsOf sh (w 0) (cs 0)
  have e' := nCoins_stepsOf sh (w' 0) (cs' 0)
  rcases Nat.lt_trichotomy j (nS sh) with hj | rfl | hj
  · rw [groups_append_lt _ (stepsOf sh (w 0) (cs 0)) j (by omega),
      groups_append_lt _ (stepsOf sh (w' 0) (cs' 0)) j (by omega)]
    exact groups_stepsOf h₀
  · rw [← e, groups_append_coins, e, ← e', groups_append_coins, e', groups_stepsOf (cs := cs 0) (cs' := cs' 0) h₀,
      groups_stepsOf (j := 0) (cs := cs 1) (cs' := cs' 1) fun m hm => h₁ m (by omega)]
  · obtain ⟨i, rfl⟩ : ∃ i, j = nS sh + i + 1 := ⟨j - nS sh - 1, by omega⟩
    rw [← e, groups_append_succ, e, ← e', groups_append_succ]
    exact groups_stepsOf fun m hm => h₁ m (by omega)

/-! ## The read fact -/

section src

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Hypothesis: each sparse phase's messages are read off the layers** (S's `OpensOfLayers` form): at every accepted
outer proof, rep `i`'s sparse steps are `stepsOf` at a witness `w` and coins (`SparseTieL`'s), and `w`'s message `m` is
`rd i m` of the layers' bits on the read wires of stage `c₀ + i · nS + roundOf m`'s commit strings. -/
def SparseMsgsOfLayers (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (c₀ : ℕ)
    (rd : Fin 2 → ℕ → ((j : Fin S) → (Vs j).Bits) → Option Msg) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
    ∃ w cs, (src R ω t (layers zs dg N k Rw R ω t R₂ τ)).sp i = stepsOf (holoAt st hp).cls.shape w cs ∧
      ∀ m, (phaseMsgs (holoAt st hp).cls.shape w)[m]? = rd i m (maskOf Vs (skOf Vs own sel)
        (c₀ + i.val * nS (holoAt st hp).cls.shape + roundOf (sparseKinds 𝔽 (holoAt st hp).cls.shape) m)
        (layers zs dg N k Rw R ω t R₂ τ))

/-- **The sparse stage's committed rounds past `c₀` coins** (H's `StageMsgsRead c₀ (fun u => u.sp 0 ++ u.sp 1)`). -/
def SparseStageRead (c₀ : ℕ) (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ j R ω t R₂ τ o R' ω' t' R₂' τ' o', FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
    FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf Vs (skOf Vs own sel) (c₀ + j) (layers zs dg N k Rw R ω t R₂ τ) =
      maskOf Vs (skOf Vs own sel) (c₀ + j) (layers zs dg N k Rw R' ω' t' R₂' τ') →
    j < (((fun u : C3Run (holoAt st hp) => u.sp 0 ++ u.sp 1)
      (src R ω t (layers zs dg N k Rw R ω t R₂ τ))).filterMap Sum.getRight?).length →
    j < (((fun u : C3Run (holoAt st hp) => u.sp 0 ++ u.sp 1)
      (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))).filterMap Sum.getRight?).length →
    (cutOf ((fun u : C3Run (holoAt st hp) => u.sp 0 ++ u.sp 1) (src R ω t (layers zs dg N k Rw R ω t R₂ τ))))[j]? =
      (cutOf ((fun u : C3Run (holoAt st hp) => u.sp 0 ++ u.sp 1)
        (src R' ω' t' (layers zs dg N k Rw R' ω' t' R₂' τ'))))[j]?

/-- **The sparse stage's committed rounds from V*'s layers**: each round is rep 0's and rep 1's messages of it, and
each message is read on its stage's commit strings (`SparseMsgsOfLayers`). -/
theorem sparseStageRead_of_layers
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    {rd : Fin 2 → ℕ → ((j : Fin S) → (Vs j).Bits) → Option Msg}
    (h : SparseMsgsOfLayers zs dg N k Rw src own sel c₀ rd) : SparseStageRead c₀ zs dg N k Rw src own sel := by
  intro j R ω t R₂ τ o R' ω' t' R₂' τ' o' hA hA' hX _ _
  choose w cs hs hm using h R ω t R₂ τ o hA
  choose w' cs' hs' hm' using h R' ω' t' R₂' τ' o' hA'
  rw [cutOf_eq_groups, cutOf_eq_groups]
  simp only [hs, hs']
  refine groups_stage (w := w) (cs := cs) (w' := w') (cs' := cs') (fun m hj => ?_) (fun m hj => ?_)
  · rw [hm 0 m, hm' 0 m]
    have e : c₀ + (0 : Fin 2).val * nS (holoAt st hp).cls.shape + roundOf (sparseKinds 𝔽 (holoAt st hp).cls.shape) m =
      c₀ + j := by rw [show (0 : Fin 2).val = 0 from rfl, Nat.zero_mul, Nat.add_zero, hj]
    rw [e, hX]
  · rw [hm 1 m, hm' 1 m]
    have e : c₀ + (1 : Fin 2).val * nS (holoAt st hp).cls.shape + roundOf (sparseKinds 𝔽 (holoAt st hp).cls.shape) m =
      c₀ + j := by rw [show (1 : Fin 2).val = 1 from rfl, Nat.one_mul]; omega
    rw [e, hX]

/-- **`SparseMsgsRead` from V*'s layers**, through H's lemma for the parts' committed rounds (`sparseMsgsRead_of_stage`
at the reps' part's `c₀` coins ending in one, `PartRead.lean`), taken here in its form. -/
theorem sparseMsgsRead_of_layers
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    {rd : Fin 2 → ℕ → ((j : Fin S) → (Vs j).Bits) → Option Msg}
    (hH : SparseStageRead c₀ zs dg N k Rw src own sel → SparseMsgsRead zs dg N k Rw src own sel)
    (h : SparseMsgsOfLayers zs dg N k Rw src own sel c₀ rd) : SparseMsgsRead zs dg N k Rw src own sel :=
  hH (sparseStageRead_of_layers h)

end src

end

end FlockC3.Sparse
