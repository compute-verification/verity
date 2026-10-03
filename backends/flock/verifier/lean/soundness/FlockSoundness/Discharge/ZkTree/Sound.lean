import FlockSoundness.Discharge.ZkTree.Audit
import FlockSoundness.Discharge.Composed.Sound

/-!
# One `--zk` sampled session on the coin tree is sound

`zk_session_sound_tree`: `Composed.zk_session_sound` for the tree-coin run of the live audit, with `N·2^-256` added,
`N = nS·2^rl` the hm96 leaves under the session's key (`|streams|·2^rounds_log`).

The live audit's session is `liveTableC`, the coin server's live table session on the coin tree, and its prover a
`TreeProver`: it registers, the units are drawn, it sends its nonce `ν`, and the server draws the session key, every
round's block and every round's salt, and shows it the key and every leaf. For every such prover `τ` there is a
mixture of live-audit provers (`Mx`), each registering what `τ` registers, such that
* on every event of the audit's outcome, the tree-coin run is within `N·2^-256` of the mixture
  (`ZkTree.treeAudit_le_model`, from `treeCoins_le_model`: statistical, no hash property);
* when every prover of the mixture meets `zk_session_sound`'s `cr/sha-512` hypothesis, flock-verify accepts the
  tree-coin run with probability at most the mixture's probability of acceptance with fewer than `K₀` wrong units, plus
  `miss(K₀)`, the mixture's average of `ε_ks + δ_link`, and `N·2^-256` (`zk_session_sound` at each of the mixture's
  provers).

The wrong units are those of each mixture prover's committed values (`zk_session_sound`'s), so the second item says:
except with probability `miss(K₀) + ε_ks + δ_link + N·2^-256`, an accepted tree-coin run is one of the coupled model
prover's with fewer than `K₀` wrong units. Its assumptions are `zk_session_sound`'s; the tree's term reads none.
-/

namespace FlockSoundness.Discharge.ZkTree

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- Acceptance splits at `K₀` wrong units. -/
theorem prob_le_split {α : Type} {G : Game α} (s : Strategy G) (A : α → Prop) (W : α → ℕ) (K₀ : ℕ) {b : ℝ≥0∞}
    (h : prob (fun o => A o ∧ K₀ ≤ W o) G s ≤ b) : prob A G s ≤ prob (fun o => A o ∧ W o < K₀) G s + b :=
  (prob_mono (q := fun o => (A o ∧ W o < K₀) ∨ (A o ∧ K₀ ≤ W o))
    (fun o hA => (lt_or_ge (W o) K₀).imp (⟨hA, ·⟩) (⟨hA, ·⟩)) G s).trans
    ((prob_or_le _ _ G s).trans (add_le_add_right h _))

/-- A bound at each prover of a mixture is a bound on the mixture, the constant term counted once. -/
theorem Mix.pr_le_split {α Hl : Type} {G : Game α} (M : Mix G Hl) (A : α → Prop) (p b : M.Ρ → ℝ≥0∞) (c : ℝ≥0∞)
    (h : ∀ x, prob A G (M.σ x) ≤ p x + (c + b x)) :
    M.pr (fun _ => A) ≤ ∑ x, M.w x * p x + (c + ∑ x, M.w x * b x) := by
  calc M.pr (fun _ => A) = ∑ x, M.w x * prob A G (M.σ x) := rfl
    _ ≤ ∑ x, M.w x * (p x + (c + b x)) := Finset.sum_le_sum fun x _ => mul_le_mul_right (h x) _
    _ = _ := by
      simp only [mul_add, Finset.sum_add_distrib]
      rw [← Finset.sum_mul, M.sum_w, one_mul]

variable {mPts n : ℕ} {Reg : Type}

/-- **One `--zk` sampled session on the coin tree is sound.** -/
theorem zk_session_sound_tree
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the coin tree: `nS` streams of `2^rl` rounds, the session stream `iS` and the reps' streams `i0`, `i1`
    {E₂ Blk Cs D Nonce : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂] (hE : Fintype.card E₂ = 2)
    [Fintype Blk] [Nonempty Blk] [DecidableEq Blk] [DecidableEq Cs]
    {nS : ℕ} (rl : ℕ) (iS i0 i1 : Fin nS) (hS0 : iS ≠ i0) (hS1 : iS ≠ i1) (h01 : i0 ≠ i1)
    (dl : Option (Dec Blk (Fin 2 → Fin mPts → 𝔽))) (dr : (n : ℕ) → Option (Dec Blk (Fin n → 𝔽)))
    -- the leaves: the round digests, the salt hash (at most `2^512` values) and the keyed leaf hash, at each nonce
    (dig : Nonce → Fin nS × Fin (2 ^ rl) → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (hcs : ∀ ν, (Finset.univ.image (cs ν)).card ≤ 2 ^ 512) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    -- the tree-coin prover
    (τ : TreeProver (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg Nonce E₂ D
      (liveTableC dg rl i0 i1 dr iS dl N))
    {qF qS : ℝ} :
    ∃ Mx : Mix (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))
        PUnit, (∀ x, (Mx.σ x).1 = τ.1) ∧
      (∀ E, prTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (liveTableC dg rl i0 i1 dr iS dl N) wr
          dig cs H τ E ≤ Mx.pr (fun _ => E) + ((nS * 2 ^ rl : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 256)) ∧
      ((∀ x ω j, TableCRZ execArith hs512 (planZAt dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (Mx.σ x)))) j
        (zcStrat execArith hs512 (planZAt dj y₀)
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (Mx.σ x)) ω)) qF qS) →
        prTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (liveTableC dg rl i0 i1 dr iS dl N) wr
            dig cs H τ (LiveAcceptsZK I) ≤
          ∑ x, Mx.w x * prob (fun o => LiveAcceptsZK I o ∧
              (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
                (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                  ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                  (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (Mx.σ x)))
                  (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (Mx.σ x))) k Rw)).card < K₀)
              (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) (Mx.σ x) +
          ((Law.stratified σs ks hks).miss K₀ +
            ∑ x, Mx.w x * (ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
                (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (Mx.σ x))) +
              linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (Mx.σ x)))
                (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (Mx.σ x))))) +
          ((nS * 2 ^ rl : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 256)) := by
  obtain ⟨Mx, hreg, hcl⟩ := treeAudit_le_model hE (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
    (erD_liveTableC dg rl i0 i1 dr iS dl N) (distinct_liveTableC dg rl i0 i1 dr iS dl N hS0 hS1 h01) wr dig cs hcs H τ
  have hcl' : ∀ E, prTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
      (liveTableC dg rl i0 i1 dr iS dl N) wr dig cs H τ E ≤
        Mx.pr (fun _ => E) + ((nS * 2 ^ rl : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 256) := fun E => by
    have h := hcl E
    rwa [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at h
  refine ⟨Mx, hreg, hcl', fun hCR => (hcl' _).trans (add_le_add_left (Mix.pr_le_split Mx _ _ _ _ fun x => ?_) _)⟩
  exact prob_le_split (Mx.σ x) _ _ K₀ ((zk_session_sound I ht dj y₀ wr dg N hc hU hpub hpt hscope σs ks hks St hS Kd
    pop Lb src hA3 hDraw hTags hRec hk hRw hM hρ hr ht' K₀ (Mx.σ x) (hCR x)).trans_eq (add_assoc _ _ _))

end FlockSoundness.Discharge.ZkTree
