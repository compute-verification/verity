import FlockSoundness.Discharge.ZkTree.Hybrid
import FlockSoundness.Discharge.Exec.Live

/-!
# The tree-coin run of a one-stage audit

`treeAudit_le_model`: in a one-stage audit whose session is a committed interaction `g` of the model session `G`
(`ErD g G`, no round opened twice), the tree-coin run (`prTree`: the prover registers, the units are drawn, the prover
sends its nonce, then the server's key, blocks and salts) is, on every event of the audit's outcome, within
`|I|·2^-256` of a mixture of strategies of the model audit `Exec.auditLive` that register what the tree prover
registers. The mixture does not depend on the event: at each draw `ω` it is `treeCoins_le_model`'s mixture at the
nonce the prover sends at `ω`, independently across draws (`Mix.piOver`).

`erD_liveTableC`, `distinct_liveTableC`: the live table session on the coin tree is `Refine.liveTable` with coins read
off committed rounds, and opens each round at most once when its three streams are distinct.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins Discharge.Exec Refine Finset Function
open scoped ENNReal

section Pi

variable {ι : Type} [Fintype ι] [DecidableEq ι] {P : ι → Type} [∀ i, Fintype (P i)]

theorem sum_pi_prod (w : ∀ i, P i → ℝ≥0∞) (hw : ∀ i, ∑ p, w i p = 1) : ∑ ρ : (∀ i, P i), ∏ i, w i (ρ i) = 1 := by
  rw [← Fintype.prod_sum w]
  exact Finset.prod_eq_one fun i _ => hw i

/-- **The marginal of a product law** on a dependent product, at coordinate `i₀`. -/
theorem sum_pi_marg (w : ∀ i, P i → ℝ≥0∞) (hw : ∀ i, ∑ p, w i p = 1) (i₀ : ι) (F : P i₀ → ℝ≥0∞) :
    ∑ ρ : (∀ i, P i), (∏ i, w i (ρ i)) * F (ρ i₀) = ∑ p, w i₀ p * F p := by
  have key : ∀ ρ : (∀ i, P i), (∏ i, w i (ρ i)) * F (ρ i₀) =
      ∏ i, (w i (ρ i) * if h : i = i₀ then F (h ▸ ρ i) else 1) := by
    intro ρ
    rw [Finset.prod_mul_distrib, Finset.prod_dite_eq' Finset.univ i₀ (fun i h => F (h ▸ ρ i)),
      if_pos (Finset.mem_univ _)]
  simp only [key]
  rw [← Fintype.prod_sum (fun i p => w i p * if h : i = i₀ then F (h ▸ p) else 1)]
  rw [Fintype.prod_eq_single i₀ fun i hi => by simp only [dif_neg hi, mul_one, hw]]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [dif_pos rfl]

end Pi

section Audit

variable {n : ℕ} {Reg β : Type} {G : Game β}

open Classical in
/-- **Independent mixtures across the draw**: the prover registers `R`, and at draw `ω` plays `M ω`'s mixture, its
randomness drawn independently for each `ω`. -/
noncomputable def Mix.piOver (L : Audit.Law n) (out : Finset (Fin n) → Reg → β → Transcript) (R : Reg) {Hl : Type}
    (M : L.Ω → Mix G Hl) : Mix (auditLive L Reg fun S R => G.map (out S R)) PUnit where
  Ρ := (ω : L.Ω) → (M ω).Ρ
  w ρ := ∏ ω, (M ω).w (ρ ω)
  sum_w := sum_pi_prod _ fun ω => (M ω).sum_w
  σ ρ := (⟨R, fun ω => Strategy.map _ _ (Strategy.map (out (L.draw ω) R) G ((M ω).σ (ρ ω)))⟩ :
    (R' : Reg) × Strategy (Game.coin L.Ω fun ω => (G.map (out (L.draw ω) R')).map fun t => (R', L.draw ω, t)))
  hello _ := PUnit.unit

open Classical in
theorem Mix.piOver_pr (L : Audit.Law n) (out : Finset (Fin n) → Reg → β → Transcript) (R : Reg) {Hl : Type}
    (M : L.Ω → Mix G Hl) (E : Reg × Finset (Fin n) × Transcript → Prop) :
    (Mix.piOver L out R M).pr (fun _ => E) =
      avg fun ω => (M ω).pr fun _ o => E (R, L.draw ω, out (L.draw ω) R o) := by
  have hp : ∀ ρ : (ω : L.Ω) → (M ω).Ρ,
      prob E (auditLive L Reg fun S R => G.map (out S R)) ((Mix.piOver L out R M).σ ρ) =
        avg fun ω => prob (fun o => E (R, L.draw ω, out (L.draw ω) R o)) G ((M ω).σ (ρ ω)) := by
    intro ρ
    show avg (fun ω => prob E ((G.map (out (L.draw ω) R)).map fun t => (R, L.draw ω, t))
      (Strategy.map _ _ (Strategy.map (out (L.draw ω) R) G ((M ω).σ (ρ ω))))) = _
    congr 1
    funext ω
    rw [prob_strategy_map, prob_strategy_map]
  show ∑ ρ : ((ω : L.Ω) → (M ω).Ρ), (∏ ω, (M ω).w (ρ ω)) *
      prob E (auditLive L Reg fun S R => G.map (out S R)) ((Mix.piOver L out R M).σ ρ) = _
  simp only [hp]
  rw [sum_mul_avg]
  congr 1
  funext ω
  exact sum_pi_marg (fun ω => (M ω).w) (fun ω => (M ω).sum_w) ω
    (fun ρ' => prob (fun o => E (R, L.draw ω, out (L.draw ω) R o)) G ((M ω).σ ρ'))

end Audit

/-- **The tree-coin run of a one-stage audit against the model audit.** -/
theorem treeAudit_le_model {n : ℕ} {E₂ I Blk Cs D Nonce Reg β : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂]
    (hE : Fintype.card E₂ = 2) [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk]
    [DecidableEq Cs] (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G) (hg : g.Distinct)
    (out : Finset (Fin n) → Reg → β → Transcript)
    (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (hcs : ∀ ν, (Finset.univ.image (cs ν)).card ≤ 2 ^ 512) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    (τ : TreeProver L Reg Nonce E₂ D g) :
    ∃ M : Mix (auditLive L Reg fun S R => G.map (out S R)) PUnit, (∀ ρ, (M.σ ρ).1 = τ.1) ∧
      ∀ E, prTree L g out dig cs H τ E ≤ M.pr (fun _ => E) + Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256) := by
  choose Mω hMω using fun ω : L.Ω => treeCoins_le_model hE hgG hg (dig (τ.2 ω).1) (cs (τ.2 ω).1) (hcs _)
    (H (τ.2 ω).1) (τ.2 ω).2
  refine ⟨Mix.piOver L out τ.1 Mω, fun _ => rfl, fun E => ?_⟩
  rw [Mix.piOver_pr]
  calc prTree L g out dig cs H τ E
      ≤ avg (fun ω => (Mω ω).pr (fun _ o => E (τ.1, L.draw ω, out (L.draw ω) τ.1 o)) +
          Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256)) :=
        avg_mono fun ω => hMω ω fun _ o => E (τ.1, L.draw ω, out (L.draw ω) τ.1 o)
    _ = _ := by rw [avg_add, avg_const]

/-! ## The live table session on the coin tree -/

section Live

open Flock

local notation "𝔽" => FlockLevel3.GF128

variable {Blk : Type}

/-- Sequencing of `ErD`. -/
def ErD.bind {I α β : Type} {f : α → TGame I Blk β} {F : α → Game β} (hf : (a : α) → ErD (f a) (F a)) :
    {g : TGame I Blk α} → {G : Game α} → ErD g G → ErD (g.bind f) (G.bind F)
  | _, _, .ret a => hf a
  | _, _, .send M k => .send M fun m => ErD.bind hf (k m)
  | _, _, @ErD.coin _ _ _ i C iF iN d _ _ r => @ErD.coin _ _ _ i C iF iN d _ _ fun c rest => ErD.bind hf (r c rest)
  | _, _, @ErD.fresh _ _ _ C iF iN _ _ r => @ErD.fresh _ _ _ C iF iN _ _ fun c => ErD.bind hf (r c)

/-- `coinD` is a coin of the model. -/
def erD_coinD {I α : Type} (i : I) {C : Type} [Fintype C] [Nonempty C] (d : Option (Dec Blk C))
    {k : C → TGame I Blk α} {k' : C → Game α} (h : (c : C) → ErD (k c) (k' c)) : ErD (coinD i d k) (.coin C k') :=
  match d with
  | some d => .coin i d fun c r =>
      cast (congrArg (fun t => ErD t (k' c))
        (congrArg (fun x : C × d.Rest => k x.1) (d.e.symm_apply_apply (c, r))).symm) (h c)
  | none => .fresh C h

theorem readsIn_coinD {I α : Type} {S : Set I} {i : I} (hi : i ∈ S) {C : Type} [Fintype C] [Nonempty C]
    (d : Option (Dec Blk C)) {k : C → TGame I Blk α} (hk : ∀ c, TGame.ReadsIn (S \ {i}) (k c)) :
    TGame.ReadsIn S (coinD i d k) := by
  cases d with
  | some d => exact ⟨hi, fun b => hk _⟩
  | none => exact fun c => (hk c).mono Set.sdiff_subset

variable (dg : ByteArray → ByteArray) {nS : ℕ} (rl : ℕ) (i0 i1 : Fin nS)
  (dr : (n : ℕ) → Option (Dec Blk (Fin n → 𝔽))) (iS : Fin nS) {mPts : ℕ}
  (dl : Option (Dec Blk (Fin 2 → Fin mPts → 𝔽))) (N : ℕ)

/-- The coin server's rounds on the coin tree are its rounds, with coins read off committed rounds. -/
def erD_liveRoundsC : (N t₀ t₁ : ℕ) → (acc : List Round × List Round) →
    ErD (liveRoundsC dg rl i0 i1 dr N t₀ t₁ acc) (liveRounds dg N acc)
  | 0, _, _, acc => .ret acc
  | N + 1, t₀, t₁, acc => .send (Option Submit) fun
    | none => .ret acc
    | some s => by
        show ErD (if h : (if s.rep = 0 then t₀ else t₁) < 2 ^ rl then _ else _)
          (coin (Fin s.n → 𝔽) fun c => liveRounds dg N (addRound acc s.rep (roundOf dg s.bytes s.n c)))
        by_cases h : (if s.rep = 0 then t₀ else t₁) < 2 ^ rl
        · rw [dif_pos h]
          exact erD_coinD (C := Fin s.n → 𝔽) _ _ fun c => erD_liveRoundsC N _ _ _
        · rw [dif_neg h]
          exact .fresh (Fin s.n → 𝔽) fun c => erD_liveRoundsC N _ _ _

/-- **The live table session on the coin tree is the live table session**, with its coins read off committed
rounds. -/
noncomputable def erD_liveTableC : ErD (liveTableC dg rl i0 i1 dr iS dl N) (liveTable dg mPts N) :=
  .send ByteArray fun rootB => erD_coinD _ dl fun pts =>
    (erD_liveRoundsC dg rl i0 i1 dr N 0 0 ([], [])).bind fun rs =>
      .send (Array ByteArray) fun proofs => .ret (⟨rootB, pts, rs.1, rs.2, proofs⟩ : LiveOut mPts)

/-- The rounds of rep 0's stream from round `t₀` on and of rep 1's from round `t₁` on. -/
def repRounds (t₀ t₁ : ℕ) : Set (Fin nS × Fin (2 ^ rl)) :=
  {x | (x.1 = i0 ∧ t₀ ≤ x.2.val) ∨ (x.1 = i1 ∧ t₁ ≤ x.2.val)}

theorem readsIn_liveRoundsC (h01 : i0 ≠ i1) : ∀ (N t₀ t₁ : ℕ) (acc : List Round × List Round),
    TGame.ReadsIn (repRounds rl i0 i1 t₀ t₁) (liveRoundsC dg rl i0 i1 dr N t₀ t₁ acc)
  | 0, _, _, _ => trivial
  | N + 1, t₀, t₁, acc => by
    intro m
    rcases m with _ | s
    · trivial
    · show TGame.ReadsIn _ (if h : (if s.rep = 0 then t₀ else t₁) < 2 ^ rl then _ else _)
      by_cases hr : s.rep = 0
      · simp only [hr, ↓reduceIte]
        split
        · rename_i h
          refine readsIn_coinD ?_ (dr s.n) fun c => (readsIn_liveRoundsC h01 N _ _ _).mono ?_
          · exact Or.inl ⟨rfl, le_rfl⟩
          · rintro x (⟨hx, hle⟩ | ⟨hx, hle⟩)
            · refine ⟨Or.inl ⟨hx, by omega⟩, fun hm => ?_⟩
              rw [Set.mem_singleton_iff] at hm
              subst hm
              simp at hle
            · refine ⟨Or.inr ⟨hx, hle⟩, fun hm => ?_⟩
              rw [Set.mem_singleton_iff] at hm
              subst hm
              exact h01 hx
        · intro c
          refine (readsIn_liveRoundsC h01 N _ _ _).mono ?_
          rintro x (⟨hx, hle⟩ | hx)
          · exact Or.inl ⟨hx, by omega⟩
          · exact Or.inr hx
      · simp only [hr, ↓reduceIte]
        split
        · rename_i h
          refine readsIn_coinD ?_ (dr s.n) fun c => (readsIn_liveRoundsC h01 N _ _ _).mono ?_
          · exact Or.inr ⟨rfl, le_rfl⟩
          · rintro x (⟨hx, hle⟩ | ⟨hx, hle⟩)
            · refine ⟨Or.inl ⟨hx, hle⟩, fun hm => ?_⟩
              rw [Set.mem_singleton_iff] at hm
              subst hm
              exact h01 hx.symm
            · refine ⟨Or.inr ⟨hx, by omega⟩, fun hm => ?_⟩
              rw [Set.mem_singleton_iff] at hm
              subst hm
              simp at hle
        · intro c
          refine (readsIn_liveRoundsC h01 N _ _ _).mono ?_
          rintro x (hx | ⟨hx, hle⟩)
          · exact Or.inl hx
          · exact Or.inr ⟨hx, by omega⟩

/-- **The live table session on the coin tree opens each round at most once.** -/
theorem distinct_liveTableC (hS0 : iS ≠ i0) (hS1 : iS ≠ i1) (h01 : i0 ≠ i1) :
    (liveTableC dg rl i0 i1 dr iS dl N).Distinct := by
  refine TGame.distinct_of_readsIn (S := Set.univ) fun rootB => ?_
  refine readsIn_coinD (Set.mem_univ _) dl fun pts => ?_
  have h := TGame.readsIn_bind (S' := ∅)
    (f := fun rs => TGame.send (Array ByteArray) fun proofs => .ret (⟨rootB, pts, rs.1, rs.2, proofs⟩ : LiveOut mPts))
    (fun _ _ => trivial) (readsIn_liveRoundsC dg rl i0 i1 dr h01 N 0 0 ([], [])) (Set.disjoint_empty _)
  refine h.mono ?_
  rintro x ((⟨hx, -⟩ | ⟨hx, -⟩) | hx)
  · refine ⟨trivial, fun hm => ?_⟩
    rw [Set.mem_singleton_iff] at hm
    subst hm
    exact hS0 hx
  · refine ⟨trivial, fun hm => ?_⟩
    rw [Set.mem_singleton_iff] at hm
    subst hm
    exact hS1 hx
  · exact absurd hx (Set.notMem_empty x)

end Live

end FlockSoundness.Discharge.ZkTree
