import Proofs.Flock.Recursive.ZkReg.DrawnRefines
import Proofs.Flock.Recursive.ZkReg.RegLink
import Proofs.Flock.Soundness.Registered.ReadsX

/-!
# The committed values of a `--zk --registered` session at its registered reads

`zk_session_regValsR`: in `flock-verify verify --zk --registered F` sessions of `J` hidden-output tables, at the committed
values the simulation `simZHJ` decodes (`XplurZC` at the rows `rsZHJS`, whose value binding is `vbHJ`), no drawn unit
reads a registered read whose session leaf is the registrant's and whose committed value is not the registrant's, except
with probability `ε_ks + δ_link + δ_reg` (`RegValsSound`'s event, at `layShHJ`'s registered reads `rd`). The registrant's
opening of each commit string is its row and salt (`regOpenZ`), whose `hm96X` leaf at its encoding is `rd.leaf`; `δ_reg = regTermZC` is
`cr/sha-512` for the finders that collide an extracted opening with the registrant's (`RegCRZC`, a hypothesis as `hCR`
is).
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open Composed
open ZkHidden (planZHJ tabZHJ DrawSetupZHJ TagsOkZHJ DrawOkZKHJ layShHJ placeZHJ)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **The value binding of a hidden-output `--zk` session of `J` tables** (`rsZHJS`'s). -/
noncomputable abbrev vbHJ {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
    (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) :=
  (rsZHJS (Reg := Reg) dj y₀ ht hc hscope hU hpub).bindingZ (rsZHJS_computes dj y₀ ht hc hscope hU hpub)

/-- **The registrant's opening** of each commit string of `vbHJ`: at a registered row, the row and salt `rd` reads
there; at a single gate, the value pinned by the registration (whose plain leaf ignores the salt). -/
def regOpenZ {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub)) (R : Reg) :
    (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Pos → (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Opening :=
  fun p => match p with
    | .inl uq => (.inl uq, rd.wv (.inl uq), rd.ws (.inl uq))
    | .inr g => (.inr g, (layShHJ dj y₀ ht hc hscope hU (pub := pub)).regVal R (.inr g), [])

/-- At a registered read, `regOpenZ` is the registrant's row opening. -/
theorem regOpenZ_of_read {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub)) (R : Reg)
    (p : (vbHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub).Pos) (hp : rd.isRead p) :
    regOpenZ ht dj y₀ hc hU hpub hscope rd R p = (p, rd.wv p, rd.ws p) := by
  have hrow := rd.read_row p hp
  rcases p with uq | g
  · rfl
  · simp at hrow

/-- **The committed values of a `--zk --registered` session are the registered values at its registered reads**, except
with probability `ε_ks + δ_link + δ_reg`: for every registered-reads record `rd` of `layShHJ`, with `cr/sha-512` at
budgets `qF, qS` for the knowledge finders of `τ`'s simulation (`hCR`) and at budget `qW` for its registered-value finders
(`hW`), an accepted session whose draw has a unit reading a wire `g` at a registered read, whose session leaf there is
the registrant's, and whose committed bit on `g` is not the registrant's value's, has probability at most
`ksAvgStrictZ + linkBoundZC + regTermZC`. -/
theorem zk_session_regValsR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    -- the law
    (Lv : Law n)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (Lv.draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the registered reads
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    -- the live prover, and cr/sha-512 for the knowledge and registered-value finders of its simulation
    (τ : Strategy (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZHJ dj y₀ (Lv.draw ω) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd
        (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)))
      (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
      (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
          rd.isRead ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
          (layShHJ dj y₀ ht hc hscope hU (pub := pub)).regLeaf o.1
              ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) =
            rd.leaf ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
          XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
              (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
              (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw g ≠
            (layShHJ dj y₀ ht hc hscope hU (pub := pub)).bit
              (rd.wv ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g)) g)
        (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) Lv k qF qS (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) k Rw M ρ t'
          (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        regTermZC execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
          (tabZHJ dj y₀) (vbHJ dj y₀ ht hc hscope hU hpub) Lv k Rw ρ qW := by
  have hdec := decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
    (ZkHidden.runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr Lv hTags hRec hDraw)
  have hc' := zk_flock_regValsC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
    (tabZHJ dj y₀) execArith_correct (placeZHJ dj y₀ ht hc hscope hU) (vbHJ dj y₀ ht hc hscope hU hpub)
    (regOpenZ ht dj y₀ hc hU hpub hscope rd
      (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))) hk hRw hM hρ hr ht'
    (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) hCR hW
  refine le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_drawnJ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun R S => ∃ u ∈ S,
      ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
        rd.isRead ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
        (layShHJ dj y₀ ht hc hscope hU (pub := pub)).regLeaf R ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) =
          rd.leaf ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) ∧
        XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw g ≠
          (layShHJ dj y₀ ht hc hscope hU (pub := pub)).bit
            (rd.wv ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g)) g) _
      (le_trans (prob_mono ?_ _ _) hc'))
  rintro o ⟨hacc, u, hu, g, hg, hread, hleaf, hne⟩
  refine ⟨hacc, u, hu, g, hg, ?_, hne⟩
  rw [regOpenZ_of_read ht dj y₀ hc hU hpub hscope rd _ _ hread]
  refine Prod.ext rfl ?_
  show ((rsZHJS dj y₀ ht hc hscope hU hpub).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZX.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

/-- **A registered read whose session leaf is the registrant's, carrying another bit, is a registered-value
disagreement** (`RegBadZ`) of the registrant's opening `regOpenZ`. -/
theorem regBadZ_of_leaf {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    (R : Reg) (Xt : Fin (Layout.keyN c hU n) → Bool) (g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N)
    (hread : rd.isRead ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g))
    (hleaf : (layShHJ dj y₀ ht hc hscope hU (pub := pub)).regLeaf R ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g) =
      rd.leaf ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g))
    (hne : Xt g ≠ (layShHJ dj y₀ ht hc hscope hU (pub := pub)).bit
      (rd.wv ((layShHJ dj y₀ ht hc hscope hU (pub := pub)).pos g)) g) :
    RegBadZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd R) R Xt g := by
  rw [regOpenZ_of_read ht dj y₀ hc hU hpub hscope rd R _ hread]
  refine ⟨Prod.ext rfl ?_, hne⟩
  show ((rsZHJS dj y₀ ht hc hscope hU hpub).scheme _).leaf (rd.wv _) (rd.ws _) = _
  unfold HmRowsZX.scheme
  split_ifs with hrow
  · exact hleaf.symm
  · exact absurd (rd.read_row _ hread) hrow

/-- **A layer carrying 0 at the constant has a registered-value disagreement there**: the constant is a single-gate
commit string, and its registration-pinned value is 1. -/
theorem regBadZ_oneG {I : Inputs} (ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hscope : Layout.scopeOkH c = true)
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    (R : Reg) (Xt : Fin (Layout.keyN c hU n) → Bool)
    (hne : Xt (Layout.oneG (c := c) (hU := hU) (n := n)) ≠ true) :
    RegBadZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd R) R Xt
        (Layout.oneG (c := c) (hU := hU) (n := n)) := by
  refine ⟨?_, ?_⟩
  · have hkey : Layout.keyRowOf c hU n (Layout.portRowW c)
        (Layout.oneG (c := c) (hU := hU) (n := n)) = none := by
      unfold Layout.keyRowOf Layout.oneG
      simp [Placed.oneGate]
    have hunit : (keyProg c hU n).unit (Layout.oneG (c := c) (hU := hU) (n := n)) = none :=
      Layout.unit_uniform_input (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
        (one := Placed.oneGate n (Layout.keyB c hU))
        (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) n _
        (by simp [Layout.oneG, Placed.oneGate])
    have hrow : HiddenExec.rowOfH c hU n (Layout.outsOf c hU n)
        (Layout.oneG (c := c) (hU := hU) (n := n)) = none := by
      rw [HiddenExec.rowOfH_of_none _ hkey]
      unfold HiddenExec.outRowOf
      rw [hunit]
    have hpos : (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub)).pos
        (Layout.oneG (c := c) (hU := hU) (n := n)) =
        .inr (Layout.oneG (c := c) (hU := hU) (n := n)) := by
      dsimp only [layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout, Layout.ofBits,
        HiddenExec.keyReadersH]
      exact Layout.ofRows_pos_none _ hrow
    rw [hpos]
    rfl
  · simpa [regOpenZ, ZkHidden.layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout,
      Layout.Readers.ofRows, Layout.keyGate] using hne

/-- **`zk_session_regValsR` with the wrong units counted too**: on its assumptions, an accepted session whose draw meets
the units that compute wrongly or read a registered-value disagreement (`regWrongZ`, at the registrant's opening
`regOpenZ` and the simulation's registration), both at the committed values, has probability at most
`ksAvgStrictZ + linkBoundZC + regTermZC` (`zk_flock_regDrawnC_cr`). -/
theorem zk_session_regDrawnR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    -- the law
    (Lv : Law n)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (Lv.draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    -- the registered reads
    (rd : FlockSoundness.Registered.ReadsX (layShHJ (Reg := Reg) dj y₀ ht hc hscope hU (pub := pub))
      (rsShHJS dj y₀ ht hc hscope hU hpub))
    -- the live prover, and cr/sha-512 for the knowledge and registered-value finders of its simulation
    (τ : Strategy (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512
      (planZHJ dj y₀ (Lv.draw ω) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) ω)) qF qS)
    {qW : ℝ}
    (hW : RegCRZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd
        (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)))
      (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
      (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw qW) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ¬ Disjoint (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
            (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
              (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
              (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw) ∪
          regWrongZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd
              (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)))
            (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
              (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
              (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw))
          o.2.1)
        (auditLive Lv Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) Lv k qF qS (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) k Rw M ρ t'
          (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) +
        regTermZC execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
          (tabZHJ dj y₀) (vbHJ dj y₀ ht hc hscope hU hpub) Lv k Rw ρ qW := by
  have hdec := decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
    (ZkHidden.runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr Lv hTags hRec hDraw)
  have hc' := zk_flock_regDrawnC_cr execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
    (tabZHJ dj y₀) execArith_correct (placeZHJ dj y₀ ht hc hscope hU) (vbHJ dj y₀ ht hc hscope hU hpub)
    (regOpenZ ht dj y₀ hc hU hpub hscope rd
      (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))) hk hRw hM hρ hr ht'
    (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ) hCR hW
  exact le_trans (prob_mono (fun o h => ⟨acceptsZK_of_R h.1, h.2⟩) _ _)
    (zk_headline_exec_drawnJ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ (fun _ S => ¬ Disjoint
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw) ∪
        regWrongZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          (vbHJ dj y₀ ht hc hscope hU hpub) (regOpenZ ht dj y₀ hc hU hpub hscope rd
            (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)))
          (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            (vbHJ dj y₀ ht hc hscope hU hpub) (reg (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ))
            (cont (simZHJ dj y₀ ht hc hscope hU wr dg N hTags hRec hDraw τ)) k Rw)) S) _ hc')

end FlockSoundness.Discharge.ZkReg
