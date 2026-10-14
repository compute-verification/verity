import Proofs.Flock.C3.Reps.Tie
import Proofs.Flock.C3.HoloValues.Shift

/-!
# The record's frames at the holo shape (`FramesOfVals`)

`unitFrames_holo`: a rep unit's `UnitFrames` at values `v`, from E's preconditions, `LigShift` and `HoloFramed`, the
facts no check here establishes. `framesOfVals_holo` is `FramesOfVals` from them at every rep, and
`framesOfVals_holo'` the same from `HoloFramed` alone (`ligShift`).
-/

namespace FlockC3.Reps

noncomputable section

section unit

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockC3.HoloValues
open FlockVBridge.Compiled (RepData AlgebraHolds AccHolds)
open FlockVBridge.Family (frameAt streamRounds streamRounds_coins streamRounds_retains streamRounds_n frameAt_rs
  frameAt_gamma frameAt_lig answers_of_retains' OpenedAll)
open FlockSoundness.Discharge.Exec (chunks ligFrames)

/-- **A rep unit's record and proof at holo**: the record is G's stream at `holoFrameAt`'s bytes from `dom`, with
`HoloCoinsShaped`'s coins (`TableSrc.Coins` at holo); the setup's and shape's dimensions (`SetupRealizes`,
`ShapeIs`); the proof file's decode and the messages w holds (`run_of_frames`' proof-file hypotheses); the openings
(`OpenedAll`); and `v`'s claims, ring-switch coins and end are `verifierRows`' and the Ligerito's. -/
structure HoloFramed (sh : Flock.Shape) (d : RepData) (v : UnitVals) (dom : ByteArray) : Prop where
  record : v.t.rounds = streamRounds d.rounds
    (holoFrameAt v.st v.sch dom v.extras.size v.p (valOf sh d.w d.comb d.cons .T) (hOfW sh d.w))
  k0 : v.t.k = 0
  pending : v.t.pending = Transcript.header 1 0 dom.size ++ Transcript.padded dom
  coins : HoloCoinsShaped v.st.m v.st.fold.kLog v.extras.size
    (chunks (ligFrames v.sch (valOf sh d.w d.comb d.cons .T) v.p.cap v.p.ligerito)).1.length d.rounds
  m14 : 14 ≤ v.st.m
  m : v.st.m = sh.m
  kL : v.st.fold.kLog = sh.kLog
  kS : v.st.kLog = sh.kLog
  km : v.st.kLog ≤ v.st.m
  sched : sh.sched = v.sch
  fast : Flock.fast100 v.st.m = some v.sch
  extras : d.extras = v.extras
  extShape : ∀ j < v.extras.size, v.extras[j]!.skip.size = 64 ∧ v.extras[j]!.x.size = v.st.m - 6
  enc : ∃ b, decodeProof (shapeOf v.st v.sch v.extras.size) b = .ok v.p
  zp : ∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
    ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) v.p.zerocheck
  lp : LcProofOf (decoded sh d.w d.comb d.cons) v.p.lincheck
  rsw : v.p.ringSwitches = (decoded sh d.w d.comb d.cons).rs
  messages : v.p.ligerito.messages = (decoded sh d.w d.comb d.cons).msgs.map fun mm => (kv mm.1, kv mm.2)
  ood : v.p.ligerito.oodValues = (decoded sh d.w d.comb d.cons).ood
  yr : v.p.ligerito.finalProof.yr = (decoded sh d.w d.comb d.cons).yr
  initialCap : (v.p.ligerito.initialCap == v.p.cap) = true
  opened : OpenedAll v.st v.sch sh v.p d
  vals : ∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
    v.claims = claimsF sh (valOf sh d.w d.comb d.cons) info v.extras ∧
      v.rpp = (fun k => (info.rs.sws[k]!).rpp) ∧ v.gamma = info.rs.gamma
  t₄ : ∀ t', Flock.ligerito v.st.merkle v.sch (rsT v.claims v.p.ringSwitches v.rpp v.gamma) v.p.cap v.p.ligerito
      (Flock.basis (rsSws v.claims v.rpp v.gamma))
      ⟨v.t.rounds, v.t.k + 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2 + v.claims.size + 1, ByteArray.empty⟩ =
        .ok ((), t') → v.t₄ = t'

/-- **A rep unit's frames at holo** (`UnitFrames`): the counts and frames up to `r′` (`holoCounts_of_shaped`), the
claims at the record's points (`holoRs_of_residuals`, `holoRows_coins`), ring switching's rounds, and Ligerito to `v`'s
end (`holoHlg_of_frames`), with ring switching's and Ligerito's frames G's one round later (`holoFrameAt_gt`). -/
theorem unitFrames_holo (hsh : LigShift) {sh : Flock.Shape} {d : RepData} {v : UnitVals} {dom : ByteArray}
    (halg : AlgebraHolds sh d) (hacc : AccHolds sh d) (H : HoloFramed sh d v dom) : UnitFrames sh d v := by
  have hH := holoCounts_of_shaped (h := hOfW sh d.w) H.coins H.record H.k0 H.pending
  obtain ⟨hrec, hk0, -, hco, hm14, hm, hkL, hkS, hkm, hsch, hf, hxs, hext, ⟨b, hb⟩, hzp, hlp, hrsw, hmsg, hood, hyr,
    hic, hopen, hvals, ht₄⟩ := H
  obtain ⟨w, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩ := halg
  obtain ⟨hvc, hvr, hvg⟩ := hvals w info kappa hrows
  obtain ⟨hcs, -, -, -, -, -, hzp0, -, hrs7, hg, hall⟩ := hco
  have hsz : v.t.rounds.size = d.rounds.size := by rw [hrec]; simp [streamRounds]
  have hcoins : d.rounds = v.t.rounds.map (·.coins) := by rw [hrec, streamRounds_coins]
  have hct : ∀ j < v.t.rounds.size, d.rounds[j]! = (v.t.rounds[j]!).coins := fun j hj => by
    rw [hcoins, getElem!_pos _ j (by simpa using hj), Array.getElem_map, getElem!_pos v.t.rounds j hj]
  have hZ : 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) < v.t.rounds.size := by omega
  have hz0 : (v.t.rounds[3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6)]!).coins.size = 0 := by
    rw [← hct _ hZ]; exact hzp0
  have hne : ∀ j < v.t.rounds.size, j ≠ 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) →
      (v.t.rounds[j]!).coins.size ≠ 0 := fun j hj hjz => by rw [← hct j hj]; exact hall j (by omega) hjz
  have hN : ∀ j c, j < d.rounds.size → (d.rounds[j]!).size = c → (v.t.rounds[j]!).n = c :=
    fun j c hj hc => by rw [hrec]; exact streamRounds_n hj hc
  have hRg : ∀ j, 3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 1 < j → j < d.rounds.size →
      Retains v.t.rounds[j]! (frameAt v.st v.sch dom v.extras.size v.p (valOf sh d.w d.comb d.cons .T) (j - 1)) :=
    fun j h₁ h₂ => by
      have e : Retains v.t.rounds[j]! (holoFrameAt v.st v.sch dom v.extras.size v.p (valOf sh d.w d.comb d.cons .T)
          (hOfW sh d.w) j) := by rw [hrec]; exact streamRounds_retains h₂ rfl
      rwa [holoFrameAt_gt _ _ _ _ _ _ _ h₁] at e
  obtain ⟨hcl, -, -, -, hRR, hT, hSws⟩ := holoRs_of_residuals hrows hS hw hc (hacc w info kappa hrows) hres hzero
    hm14 hm hkS hkL hkm hxs hext hcoins hZ hz0 hne (hzp w info kappa hrows).fc hlp.zp hrsw rfl
    (fun k hk => hN _ _ (by omega) (hrs7 k hk))
    (fun k hk => by
      have e := hRg (3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2 + k) (by omega) (by omega)
      rw [frameAt_rs _ _ _ _ _ _ (by omega) hk] at e
      exact answers_of_retains' e)
    (hN _ _ (by omega) hg)
    (by
      have e := hRg (3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2 + (2 + v.extras.size)) (by omega) (by omega)
      rw [frameAt_gamma _ _ _ _ _ _ (by omega)] at e
      exact answers_of_retains' e)
  obtain ⟨hr, hz, hrho, hts, -⟩ := holoRows_coins hrows (by omega) hk0 hcoins fun j hj =>
    ⟨by omega, decide_eq_true (hne j (by omega) (by omega))⟩
  have hsz' := claimsF_size sh (valOf sh d.w d.comb d.cons) info v.extras
  obtain ⟨P', hl⟩ := holoHlg_of_frames hsh hrows hS hw hc (hacc w info kappa hrows) hres hzero hm14 hm.symm hkL.symm
    hsch hf hb hmsg hood hyr hic (hopen w info kappa hrows) (claims := v.claims) (rpp := v.rpp) (gamma := v.gamma)
    (by rw [hvc, hsz', hxs]) (by rw [hvc, hvr, hvg]; exact hT) (by rw [hvc, hvr, hvg]; exact hSws) hcoins hH.wf hZ
    hz0 hne (fun i hi => by
      rw [hvc, hsz']
      have e := hRg (3 + (v.st.m - 6) + 2 + (v.st.fold.kLog - 6) + 2 + (2 + v.extras.size) + 1 + i) (by omega)
        (by omega)
      rwa [frameAt_lig (i := i) _ _ _ _ _ _ (by omega)] at e)
  have e3 : v.t.k + 3 = 3 := by omega
  refine ⟨hH.k0, hH.wf, hH.n₀, hH.n₁, hH.n₂, hH.nr, hH.a₀, hH.a₁, hH.a₂, hH.ar, hH.lc, ?_, ?_, ?_⟩
  · rw [hvc, hcl, hr, hz, hrho, hts]
    simp only [hk0, Nat.zero_add, hm, hkL]
  · rw [e3, hvc, hvr, hvg]
    exact hRR
  · rw [e3] at ht₄ ⊢
    rw [ht₄ _ hl]
    exact hl

end unit

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`FramesOfVals` at holo**, from `LigShift`, when every rep unit's values meet `HoloFramed` at some domain. -/
theorem framesOfVals_holo (hsh : FlockC3.HoloValues.LigShift)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (h : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      ∃ dom, HoloFramed sh (rep R ω t Lb i) (vals R ω t Lb i) dom) :
    FramesOfVals zs dg N k Rw sh rep vals := fun R ω t R₂ τ o hA i halg hacc =>
  (h R ω t R₂ τ o hA i halg hacc).elim fun _ H => unitFrames_holo hsh halg hacc H

/-- **`FramesOfVals` at holo** when every rep unit's values meet `HoloFramed` at some domain (`framesOfVals_holo` at
`ligShift`). -/
theorem framesOfVals_holo' {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (h : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      ∃ dom, HoloFramed sh (rep R ω t Lb i) (vals R ω t Lb i) dom) :
    FramesOfVals zs dg N k Rw sh rep vals :=
  framesOfVals_holo FlockC3.HoloValues.ligShift h

end

end FlockC3.Reps
