import FlockSoundness.Discharge.Composed.Headline
import FlockSoundness.Discharge.ZkHidden.SoundJ

/-!
# Zero knowledge of production's `--zk` session: hidden outputs, `J` tables

`zk_session_view` at the session production runs. Its tables are `setupHidden`'s (hidden outputs: the output rows, their
Δ pairs `HmOut.delta`, `HmIn`'s public-input filter, and every row's `b ‖ c` region) at `--session-tables J`, each table
with its own coin vector.

* `zk_session_viewH`: at one table, the view of `tablesAtZKH`'s table at every draw `verify --zk` sets up.
* `ZkViewAtJ`, `zk_session_viewHJ`: at `J` tables, the joint view of `planZHJ`'s tables (`zk_session_soundHJ`'s plan),
  at any coin vector per table, within `Σ_j 2·|Hid_j|·2^-193` of the per-table simulators' joint view, both ways.
* `zk_session_composedHJ`: `zk_session_soundHJ` and `zk_session_viewHJ` on the same plan.

`avoidsMask` is a check on the statement (`st.da`, `st.db`, `st.regions`), so at a `setupHidden` statement it covers the
output rows' Δ pairs and regions as at a `setupH` one. Registered reads (`checkRegistered`) add no prover message: they
hold the statement to the verifier's own files and only shrink the accepting event.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-- **Zero knowledge at `J` `CoinedSetup`s, at `δ₁` per hidden leaf** (`ZkViewAt` at `J` tables): at any covered coin
vector of each table, for any encoding `enc` of the joint view and any event `E`, the real joint view at valid witnesses
`w j` whose registered values are the public ones and the simulators' at any `w0 j`, each table's prover randomness
independent, are within `Σ_j 2·|Hid_j|·δ₁`, both ways. -/
def ZkViewAtJ {J : ℕ} (C : Fin J → CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : ∀ j, CoinsZK (C j).Rest), (∀ j, (C j).Covered (c j)) → ∀ {Out : Type}
    (enc : (∀ j, ((C j).T (c j)).tab.View) → Out) (w w0 : ∀ j, Witness (C j).S.m), (∀ j, (C j).Valid (w j)) →
    ∀ (_hreg : ∀ j, ((C j).T (c j)).tab.sReg (w j) = ((C j).T (c j)).tab.pubReg) (E : Out → Prop),
    Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
        E (enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.SimRand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
          E (enc fun j => ((C j).T (c j)).simExe (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ j, ((C j).T (c j)).tab.Rand × (((C j).T (c j)).tab.Hid → ZkView.Salt) =>
            E (enc fun j => ((C j).T (c j)).viewExe (w j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card ((C j).T (c j)).tab.Hid * δ₁

/-- **One hidden-output `--zk` sampled session is zero knowledge, at the covered coin vectors** (`zk_session_view` under
hidden outputs). -/
theorem zk_session_viewH (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (y : DrawSetupZKH I mPts S (dj S)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (hPad : Assumptions.Zk.PadNonvanishing40) :
    ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZKH I mPts S (dj S)),
      ZkViewAt (coinedOf P (tablesAtZKH (Reg := Reg) dj y₀ S R) y.x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  sorry

/-- **One hidden-output `--zk` sampled session of `J` tables is zero knowledge, at the covered coin vectors**: at every
draw at which `verify --zk` sets every table up (`ys`), the joint view of `zk_session_soundHJ`'s plan `planZHJ`, table
`j` at its executable statement, at any covered coin vector per table. -/
theorem zk_session_viewHJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false)
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (hPad : Assumptions.Zk.PadNonvanishing40) :
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  sorry

/-- **One hidden-output `--zk` sampled session of `J` tables, sound and zero knowledge**: `zk_session_soundHJ` and
`zk_session_viewHJ` on the same plan `planZHJ dj y₀`. -/
theorem zk_session_composedHJ
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    -- every table's statement avoids the mask slot, and hash-derived-key and PadNonvanishing
    (hdm : ∀ (S : Finset (Fin n)) (j : Fin (nTab I))
      (y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)), avoidsMask y.x.st = true)
    (hKey : Assumptions.Zk.HashDerivedKeyHm96) (hPad : Assumptions.Zk.PadNonvanishing40) :
    (let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Layout.scopeOk_spec hU hscope).2.1 dj y₀ wr _ hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ)) ∧
    ∀ (S : Finset (Fin n)) (R : Reg)
      (ys : ∀ j : Fin (nTab I), DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)),
      ZkViewAtJ (fun j => coinedOf P ((planZHJ (Reg := Reg) dj y₀ S R).get
        (Fin.cast (planZHJ_length dj y₀ S R).symm j)) (ys j).x.st) ((2 : ℝ≥0∞) ^ (-193 : ℤ)) := by
  sorry

end FlockSoundness.Discharge.ZkHidden
