import FlockSoundness.Discharge.Composed.Defs

/-!
# The composed headline of one `--zk` sampled session

`zk_session_composed`: for any live prover `τ` of a public-output one-table `--zk` session,

* **soundness**: `Pr[flock-verify --zk accepts ∧ at least K units wrong] ≤ miss(K) + ε_ks + δ_link + δ_tree`, the
  compiled count curve `zk_flock_countC_execOS_cr` at the simulated prover `simZ … τ`, carried to flock-verify's acceptance
  by `zk_headline_exec_one`;
* **zero knowledge**: the verifier's view of the session's tables at the executable's leaves (`ZkView`) is within
  `∑ⱼ 2·|Hidⱼ|·δ₁` of `S_shvzk`'s, which reads only the public outputs, both ways (`zk_wire_fixed`, with each table's
  row (ii) from `PadNonvanishing` by `padOnto_M1`, and the gap at most `δ₁` by `hash-derived-key`).

What it rests on: `cr/sha-512` (`hCR`, `hT`), `ecr/sha-512` (inside `δ_link`, `⊤` at a prover whose link finders break
it), A3 (`hA3`), A6 and the coin server (`RecordsLiveZK`, `DrawOkZK`), `hash-derived-key` (`HashDerivedKey`) and
`PadNonvanishing`; and the open pieces `TablesAtZKLive` (item 3), `RunsZCAssembly` (item 4), `ValueBindingZKLive`
(item 8) and `PlacementZK` (item 9). The composition itself is proved.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec ZK StrictCR
open FlockSoundness.Game (prCoin)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}

/-- **One `--zk` sampled session, sound and zero knowledge.** -/
theorem zk_session_composed (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) {C : Audit.Circuit} (P : Partition C n)
    (I : Inputs) (ht : I.tags.typed = false) (hh : I.tags.hiddenOutputs = false) (h1 : I.count ≤ 1)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ)
    -- A6 and the coin server
    (hRec : RecordsLiveZK I dj wr) (hDraw : ∀ S, DrawOkZK I dj mPts S)
    -- the open pieces
    (hTab : TablesAtZKLive (n := n) (mPts := mPts) (Reg := Reg))
    (hRuns : RunsZCAssembly A Hs I dj y₀ wr opn dec one fin capOf)
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOfZ (planOf (tablesAtZK (Reg := Reg) dj y₀))
      (tabOne (tablesAtZK (Reg := Reg) dj y₀)) S R u).S u)
    (_hPlace : PlacementZK P (tablesAtZK (Reg := Reg) dj y₀) place)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P (planOf (tablesAtZK (Reg := Reg) dj y₀)) (tabOne (tablesAtZK (Reg := Reg) dj y₀))
      (decodeOfPlace A P (planOf (tablesAtZK (Reg := Reg) dj y₀)) (tabOne (tablesAtZK (Reg := Reg) dj y₀)) place) Hc)
    (_hBind : ValueBindingZKLive A P (tablesAtZK (Reg := Reg) dj y₀) vb)
    -- the arithmetic, the law and A3
    (hA : A.Correct)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planOf (tablesAtZK (Reg := Reg) dj y₀) S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs (planOf (tablesAtZK (Reg := Reg) dj y₀)
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ))) j
      (zcStrat A Hs _ (cont (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ) ω)) qF qS)
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs (planOf (tablesAtZK (Reg := Reg) dj y₀))))
      (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ)
      fun o => o.1 = true ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT)
    -- hash-derived-key and PadNonvanishing
    {δ₁ : ℝ} (hKey : HashDerivedKey δ₁) (κ : 𝔽) (hPad : ∀ cols ≤ 40, Assumptions.PadNonvanishing A cols κ) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (P.wrong (XplurZC A Hs P (planOf (tablesAtZK (Reg := Reg) dj y₀)) (tabOne (tablesAtZK (Reg := Reg) dj y₀))
          vb (reg (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ))
          (cont (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planOf (tablesAtZK (Reg := Reg) dj y₀))
          (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ)) +
        linkBoundZC A Hs P (planOf (tablesAtZK (Reg := Reg) dj y₀)) (tabOne (tablesAtZK (Reg := Reg) dj y₀)) vb k Rw M ρ
          t' (reg (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ))
          (cont (simZ A Hs _ I wr opn dg dec one fin capOf N (hRuns ht hh h1 hTab hRec hDraw) τ)) +
        ENNReal.ofReal (tr.bound qT) ∧
    ∀ {W Hi VL : Type} [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] {J : ℕ} (zs : Fin J → ZkView.Sizes)
      (T : ∀ j, ZkView.Coined K W Hi VL (zs j)) {Out : Type} (enc : (∀ j, (T j).tab.View) → Out)
      (_hrank : ∀ j, (T j).tab.Rank)
      (cols : Fin J → ℕ) (_hL : ∀ j, cols j ≤ 40) (_ht : ∀ j, (T j).tab.tpad ≤ 2 ^ cols j)
      (_hS : ∀ j, (T j).tab.nS ≤ (T j).tab.tpad) (pos : ∀ j, Fin (T j).tab.nS → ℕ) (_hpos : ∀ j, Function.Injective (pos j))
      (_hlt : ∀ j q, pos j q < 2 ^ (cols j + 1))
      (_hGp : ∀ j q i, (T j).tab.Gp q i = κ * A.Xhat i.val (cols j + 1) (A.omega (pos j q))
        + A.Xhat (2 ^ cols j + i.val) (cols j + 1) (A.omega (pos j q)))
      (_hW : ∀ j r, Function.Bijective fun p : 𝔽 × 𝔽 => (T j).tab.Wr r p.1 p.2) (_hpads : ∀ j, (T j).tab.PadsOnto)
      (_hβ : ∀ j r, (T j).tab.β r ≠ 0) (w : Fin J → W) (_hreg : ∀ j, (T j).tab.sReg (w j) = (T j).tab.pubReg)
      (_hinner : ∀ j, (T j).tab.InnerHolds (w j)) (w0 : Fin J → W) (E : Out → Prop),
      prCoin (fun p : ∀ j, (T j).tab.Rand × ((T j).tab.Hid → ZkView.Salt) =>
          E (enc fun j => (T j).viewExe (w j) (p j).1 (p j).2))
          ≤ prCoin (fun p : ∀ j, (T j).tab.SimRand × ((T j).tab.Hid → ZkView.Salt) =>
              E (enc fun j => (T j).simExe (w0 j) (p j).1 (p j).2))
            + ∑ j, 2 * Fintype.card (T j).tab.Hid * ENNReal.ofReal δ₁
        ∧ prCoin (fun p : ∀ j, (T j).tab.SimRand × ((T j).tab.Hid → ZkView.Salt) =>
            E (enc fun j => (T j).simExe (w0 j) (p j).1 (p j).2))
          ≤ prCoin (fun p : ∀ j, (T j).tab.Rand × ((T j).tab.Hid → ZkView.Salt) =>
              E (enc fun j => (T j).viewExe (w j) (p j).1 (p j).2))
            + ∑ j, 2 * Fintype.card (T j).tab.Hid * ENNReal.ofReal δ₁ := by
  refine ⟨?_, ?_⟩
  · have hdec := hRuns ht hh h1 hTab hRec hDraw
    have hc := zk_flock_countC_execOS_cr A Hs P (planOf (tablesAtZK (Reg := Reg) dj y₀))
      (tabOne (tablesAtZK (Reg := Reg) dj y₀)) hA σs ks hks St hS Kd pop Lb src hA3 place vb hk hRw hM hρ hr ht' K₀
      (simZ A Hs _ I wr opn dg dec one fin capOf N hdec τ) hCR tr hT
    unfold simZ at hc
    rw [prob_audit_cast] at hc
    exact zk_headline_exec_one (Gz A Hs _) (outz Hs _) (accz _) I wr opn dg dec one fin capOf N _
      (fun S R pts r => allInh_repZCAt A Hs _ pts r) hdec τ _ _ hc
  · intro W Hi VL _ _ _ J zs T Out enc hrank cols hL ht hS pos hpos hlt hGp hW hpads hβ w hreg hinner w0 E
    have hpad : ∀ j, (T j).tab.PadOnto := fun j =>
      padOnto_M1 (T j).tab A hA (cols j) (hL j) (ht j) (hS j) κ (pos j) (hpos j) (hlt j) (hGp j) (hPad _ (hL j))
    have hgap : ∀ j, 2 * (Fintype.card (T j).tab.Hid : ℝ≥0∞) *
        ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash) ≤
        2 * Fintype.card (T j).tab.Hid * ENNReal.ofReal δ₁ := fun j =>
      mul_le_mul_left' (ENNReal.ofReal_le_ofReal hKey) _
    have h := ZkView.zk_wire_fixed zs T enc hrank hpad hW hpads hβ w hreg hinner w0 E
    exact ⟨h.1.trans (add_le_add le_rfl (Finset.sum_le_sum fun j _ => hgap j)),
      h.2.trans (add_le_add le_rfl (Finset.sum_le_sum fun j _ => hgap j))⟩

end FlockSoundness.Discharge.Composed
