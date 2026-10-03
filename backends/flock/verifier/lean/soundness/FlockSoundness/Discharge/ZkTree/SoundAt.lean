import FlockSoundness.Discharge.ZkTree.Play
import FlockSoundness.Discharge.ZkTree.Real
import FlockSoundness.Discharge.ZkTree.Sound

/-!
# One `--zk` sampled session on the coin tree is sound, at the prover's own views

`zk_session_sound_tree` with its mixture's provers named, and for every prover of the live coin-tree session.

* `zk_session_sound_tree_at`: every prover of `zk_session_sound_tree`'s mixture is the tree-coin prover `τ` played on
  the model audit at a view `vw x` (`TreeProver.play`: `τ` with the view as fixed advice), so the `cr/sha-512`
  hypothesis is one on `τ`'s finders at each view, and the wrong units are those of `τ`'s committed values at the
  view: `XplurZC` of `simZ` at `τ.play (vw x)`.
* `HeaderBlind`, `zk_session_sound_real`: a prover of the live session (`RealProver`: it sees the key and the root at
  `Hello`, and each round's opening only after its round message) against the coin server's record, which carries the
  session's header `(ν, K, root)` (`coin_tree = {key, nonce, root, spec}`, and `Hello` with `ν`). flock-verify reads the
  header's format and nothing else of it (`Flock.CoinTree.checkRecord`: it checks the record, not the openings), so
  acceptance of the record the writer `wrH` writes at any header is acceptance of the reference writer `wr`'s
  (`HeaderBlind I wrH wr`, a property of the writer, like `RecordsLiveZK`); the bound is `zk_session_sound_tree_at`'s at
  `ρ.toTree` (`realProver_le_tree`).

Neither theorem averages the `cr/sha-512` hypothesis: it is asked of `τ`'s finders at every view `v`. A `τ` whose
finders find collisions with advice of a few views (one that reads a SHA-512 collision off a view's salts, say) falls
outside, and the theorem says nothing of it. The views' law (`Mx.w`, a salt's posterior given its leaf) is not
efficiently samplable, so an averaged hypothesis is not a hypothesis on an efficient finder.
-/

namespace FlockSoundness.Discharge.ZkTree

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {mPts n : ℕ} {Reg : Type}

/-- **One `--zk` sampled session on the coin tree is sound, at `τ`'s views.** -/
theorem zk_session_sound_tree_at
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
        PUnit, ∃ vw : Mx.Ρ → TreeView (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) E₂
          (Fin nS × Fin (2 ^ rl)) Blk D,
      (∀ x, Mx.σ x = τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)) ∧
      (∀ E, prTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (liveTableC dg rl i0 i1 dr iS dl N) wr
          dig cs H τ E ≤ Mx.pr (fun _ => E) + ((nS * 2 ^ rl : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 256)) ∧
      ((∀ v ω j, TableCRZ execArith hs512 (planZAt dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr v)))) j
        (zcStrat execArith hs512 (planZAt dj y₀)
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr v)) ω)) qF qS) →
        prTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (liveTableC dg rl i0 i1 dr iS dl N) wr
            dig cs H τ (LiveAcceptsZK I) ≤
          ∑ x, Mx.w x * prob (fun o => LiveAcceptsZK I o ∧
              (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
                (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                  ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                  (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x))))
                  (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))) k Rw)).card < K₀)
              (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))
              (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)) +
          ((Law.stratified σs ks hks).miss K₀ +
            ∑ x, Mx.w x * (ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
                (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))) +
              linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x))))
                (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) (τ.play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))))) +
          ((nS * 2 ^ rl : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 256)) := by
  sorry

/-- **The record's header is invisible to flock-verify**: acceptance of what the writer `wrH` writes at any header
`h` (the session's nonce, key and root) is acceptance of what the reference writer `wr` writes. -/
def HeaderBlind {Hd : Type} (I : Inputs) (wrH : Hd → Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ h S R o, AcceptsZK I (wrH h S R o) ↔ AcceptsZK I (wr S R o)

/-- **One `--zk` sampled session of the live coin tree is sound.** -/
theorem zk_session_sound_real
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
    {E₂ Blk Cs D Nonce Root Op : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂] (hE : Fintype.card E₂ = 2)
    [Fintype Blk] [Nonempty Blk] [DecidableEq Blk] [DecidableEq Cs] [Inhabited Op]
    {nS : ℕ} (rl : ℕ) (iS i0 i1 : Fin nS) (hS0 : iS ≠ i0) (hS1 : iS ≠ i1) (h01 : i0 ≠ i1)
    (dl : Option (Dec Blk (Fin 2 → Fin mPts → 𝔽))) (dr : (n : ℕ) → Option (Dec Blk (Fin n → 𝔽)))
    -- the leaves: the round digests, the salt hash (at most `2^512` values) and the keyed leaf hash, at each nonce
    (dig : Nonce → Fin nS × Fin (2 ^ rl) → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (hcs : ∀ ν, (Finset.univ.image (cs ν)).card ≤ 2 ^ 512) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    -- the root and the openings the prover is shown, and the coin server's record with the session's header
    (rt : Nonce → Key E₂ → (Fin nS × Fin (2 ^ rl) → D) → Root)
    (op : Nonce → Key E₂ → (Fin nS × Fin (2 ^ rl) → D) → OpenView (Fin nS × Fin (2 ^ rl)) Blk (Salt E₂) Op)
    (wrH : Nonce × Key E₂ × Root → Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (hBlind : HeaderBlind I wrH wr)
    -- the prover of the live session
    (ρ' : RealProver (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg Nonce E₂ Root Op
      (liveTable dg mPts N))
    {qF qS : ℝ} :
    ∃ Mx : Mix (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))
        PUnit, ∃ vw : Mx.Ρ → TreeView (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) E₂
          (Fin nS × Fin (2 ^ rl)) Blk D,
      (∀ x, Mx.σ x = (ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)) ∧
      ((∀ v ω j, TableCRZ execArith hs512 (planZAt dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr v)))) j
        (zcStrat execArith hs512 (planZAt dj y₀)
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr v)) ω)) qF qS) →
        prRealH (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wrH
            dig cs H rt op ρ' (LiveAcceptsZK I) ≤
          ∑ x, Mx.w x * prob (fun o => LiveAcceptsZK I o ∧
              (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
                (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                  ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                  (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x))))
                  (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))) k Rw)).card < K₀)
              (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))
              ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)) +
          ((Law.stratified σs ks hks).miss K₀ +
            ∑ x, Mx.w x * (ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
                (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))) +
              linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x))))
                (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))))) +
          ((nS * 2 ^ rl : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 256)) := by
  sorry

end FlockSoundness.Discharge.ZkTree
