import Proofs.Flock.Claims.ZK.FinishSwFrame
import Proofs.Flock.Claims.ZK.SimLig

/-!
# Level 0's opening at the switched source

`FinishSrc`'s `repCut_fw` asks of its source, at the firewall's PIOP, level 0's opening (`hlig`, `LigLock.LigHyp`):
masked values from `zp 64` to `h_ab` (the out-of-domain value, each lane message and the pair's sums) that are what the
session's opening computes at the coins plus the pads, and the restart's `y₁'`, `ȳ` and levels ≥ 1's salts. The firewall
receives only the PIOP's `zp 64` words; it computes level 0's opening itself. So the switched prover's clear values
are the firewall's (`valsFw`), with level 0's opening's clear values `l0Clear` from `zp 64` on.

* `LigLock.l0Clear`: level 0's opening's clear value at each pad index, as the session computes it at the coins.
* **`LigLock.ligHyp_of_l0`**: `LigHyp` from a source whose masked values from `zp 64` to `h_ab` are `l0Clear` plus the
  pads, and its `y₁'`, `ȳ` and salts.
* **`Masked.swSrcE_ligHyp_frame`**: so at the switched source at the firewall's clear values, with `B = l0Clear` and
  the firewall's request's pads (`swSrcE_l0_frame`), `LigHyp` holds at its `y₁'`, `ȳ` and salts.
-/

namespace FlockComplete

namespace LigLock

open Flock Transcript Replay

/-- Word `c` of an element of `K`: `c₀` at `0`, else `c₁`. -/
def l0Word (m : F256) (c : ℕ) : F128 := if c = 0 then m.c0 else m.c1

/-- **Level 0's opening's clear value at pad index `k`**, as the session computes it at the coins from round `p₀`: at
`lgOod 0` the packed table's multilinear extension at the reply to round `p₀`; at `lgMsg j c`, word `c` of lane round
`j`'s polynomial; at `lgE i c`, word `c` of the folded weight's product with the rep's extra lane `2·rep + i`. -/
def l0Clear (coin : ℕ → ℕ → F128) (p0 : ℕ) (sch : Schedule) (t b : Array F128) (lay : Zk.PadLayout)
    (extra : ℕ → ℕ → F128) (rep k : ℕ) : F128 :=
  if k = lay.lgOod 0 then mleF t (zS coin p0 sch.n)
  else if lay.lgMsg 0 0 ≤ k ∧ k < lay.lgE 0 0 then
    l0Word (if (k - lay.lgMsg 0 0) % 4 < 2 then
        (roundPoly (tabS coin p0 t ((k - lay.lgMsg 0 0) / 4)) (wS coin p0 b sch.n ((k - lay.lgMsg 0 0) / 4))).1
      else (roundPoly (tabS coin p0 t ((k - lay.lgMsg 0 0) / 4)) (wS coin p0 b sch.n ((k - lay.lgMsg 0 0) / 4))).2)
      ((k - lay.lgMsg 0 0) % 2)
  else if lay.lgE 0 0 ≤ k then
    l0Word (dotK (wS coin p0 b sch.n sch.k0)
      (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + (k - lay.lgE 0 0) / 2)))))
      ((k - lay.lgE 0 0) % 2)
  else 0

theorem l0Clear_ood (coin : ℕ → ℕ → F128) (p0 : ℕ) (sch : Schedule) (t b : Array F128) (lay : Zk.PadLayout)
    (extra : ℕ → ℕ → F128) (rep : ℕ) :
    l0Clear coin p0 sch t b lay extra rep (lay.lgOod 0) = mleF t (zS coin p0 sch.n) := by
  unfold l0Clear
  rw [ite_eq_left rfl]

theorem l0Clear_msg (coin : ℕ → ℕ → F128) (p0 : ℕ) (sch : Schedule) (t b : Array F128) (lay : Zk.PadLayout)
    (extra : ℕ → ℕ → F128) (rep : ℕ) (hood : lay.nOod0 = 1) {j c : ℕ} (hj : j < lay.k0) (hc : c < 4) :
    l0Clear coin p0 sch t b lay extra rep (lay.lgMsg j c) =
      l0Word (if c < 2 then (roundPoly (tabS coin p0 t j) (wS coin p0 b sch.n j)).1
        else (roundPoly (tabS coin p0 t j) (wS coin p0 b sch.n j)).2) (c % 2) := by
  have hd : lay.lgMsg j c - lay.lgMsg 0 0 = 4 * j + c := by
    simp only [Zk.PadLayout.lgMsg]
    omega
  have h1 : ¬ lay.lgMsg j c = lay.lgOod 0 := by
    simp only [Zk.PadLayout.lgMsg, Zk.PadLayout.lgOod]
    omega
  have h2 : lay.lgMsg 0 0 ≤ lay.lgMsg j c ∧ lay.lgMsg j c < lay.lgE 0 0 := by
    simp only [Zk.PadLayout.lgMsg, Zk.PadLayout.lgE]
    omega
  unfold l0Clear
  rw [ite_eq_right h1, ite_eq_left h2, hd, show (4 * j + c) / 4 = j by omega, show (4 * j + c) % 4 = c by omega,
    show (4 * j + c) % 2 = c % 2 by omega]

theorem l0Clear_e (coin : ℕ → ℕ → F128) (p0 : ℕ) (sch : Schedule) (t b : Array F128) (lay : Zk.PadLayout)
    (extra : ℕ → ℕ → F128) (rep : ℕ) (hood : lay.nOod0 = 1) {i c : ℕ} (hc : c < 2) :
    l0Clear coin p0 sch t b lay extra rep (lay.lgE i c) =
      l0Word (dotK (wS coin p0 b sch.n sch.k0)
        (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + i))))) c := by
  have hd : lay.lgE i c - lay.lgE 0 0 = 2 * i + c := by
    simp only [Zk.PadLayout.lgE]
    omega
  have h1 : ¬ lay.lgE i c = lay.lgOod 0 := by
    simp only [Zk.PadLayout.lgE, Zk.PadLayout.lgOod]
    omega
  have h2 : ¬ (lay.lgMsg 0 0 ≤ lay.lgE i c ∧ lay.lgE i c < lay.lgE 0 0) := fun ⟨_, h⟩ => by
    simp only [Zk.PadLayout.lgE] at h
    omega
  have h3 : lay.lgE 0 0 ≤ lay.lgE i c := by
    simp only [Zk.PadLayout.lgE]
    omega
  unfold l0Clear
  rw [ite_eq_right h1, ite_eq_right h2, ite_eq_left h3, hd, show (2 * i + c) / 2 = i by omega, show (2 * i + c) % 2 = c by omega]

/-- **`LigHyp` from level 0's opening's clear values**: where the source's masked values from `zp 64` to `h_ab` are
`l0Clear` plus the pads `h`, at a layout with one out-of-domain value and the schedule's `k₀` lane rounds, the source holds
level 0's opening at the coins from round `p₀`, given the schedule's facts and the restart's `y₁'`, `ȳ` and salts. -/
theorem ligHyp_of_l0 (coin : ℕ → ℕ → F128) (n p0 : ℕ) (ms : MerkleScheme) (sch : Schedule) (t b : Array F128)
    (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (src : RepSrc) (hk0 : 1 ≤ sch.k0) (hn0 : sch.n ≠ 0) (hlv : 2 ≤ sch.levels.size)
    (hood : lay.nOod0 = 1) (hlk : lay.k0 = sch.k0)
    (hl0 : ∀ k, lay.zp 64 ≤ k → k < lay.hab → src.masked k = l0Clear coin p0 sch t b lay extra rep k + h[k]!)
    (hy1 : p0 + 3 + sch.k0 < n → Zk.rhoIndependent (rhoS coin p0 sch.k0) = true →
      src.y1 = (Array.range (tabS coin p0 t sch.k0).size).map fun c =>
        (tabS coin p0 t sch.k0)[c]! +
          (rhoS coin p0 sch.k0).1.smul ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep)))[c]! +
          (rhoS coin p0 sch.k0).2.smul ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + 1)))[c]!)
    (hybar : p0 + 3 + sch.k0 < n → Zk.rhoIndependent (rhoS coin p0 sch.k0) = true →
      src.ybar = (Array.range tPad).map fun i =>
        (List.range (2 ^ (sch.levels.getD 0 default).lanes)).foldl
          (fun acc l => acc + ((eqTableK ((Array.range sch.k0).map (chS coin p0)))[l]!).smul (lanePad l i))
          ((rhoS coin p0 sch.k0).1.smul (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rep) i) +
            (rhoS coin p0 sch.k0).2.smul (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rep + 1) i)))
    (hsalt : p0 + 3 + sch.k0 < n → Zk.rhoIndependent (rhoS coin p0 sch.k0) = true →
      ∀ i, 1 ≤ i → i < sch.levels.size → ∀ q < 2 ^ (sch.levels.getD i default).d,
        fitBytes ms.saltBytes (src.hiSalt i q) = fitBytes ms.saltBytes (salt i q)) :
    LigHyp coin n p0 ms sch t b lay h rep tPad extra lanePad salt src := by
  have hmsg : ∀ j < lay.k0, ∀ c < 4, src.masked (lay.lgMsg j c) =
      l0Word (if c < 2 then (roundPoly (tabS coin p0 t j) (wS coin p0 b sch.n j)).1
        else (roundPoly (tabS coin p0 t j) (wS coin p0 b sch.n j)).2) (c % 2) + h[lay.lgMsg j c]! := by
    intro j hj c hc
    have hlo : lay.zp 64 ≤ lay.lgMsg j c := by simp only [Zk.PadLayout.lgMsg]; omega
    have hhi : lay.lgMsg j c < lay.hab := by simp only [Zk.PadLayout.lgMsg, Zk.PadLayout.hab]; omega
    rw [hl0 (lay.lgMsg j c) hlo hhi, l0Clear_msg coin p0 sch t b lay extra rep hood hj hc]
  have he : ∀ i < 2, ∀ c < 2, src.masked (lay.lgE i c) =
      l0Word (dotK (wS coin p0 b sch.n sch.k0)
        (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + i))))) c + h[lay.lgE i c]! := by
    intro i hi c hc
    have hlo : lay.zp 64 ≤ lay.lgE i c := by simp only [Zk.PadLayout.lgE]; omega
    have hhi : lay.lgE i c < lay.hab := by simp only [Zk.PadLayout.lgE, Zk.PadLayout.hab]; omega
    rw [hl0 (lay.lgE i c) hlo hhi, l0Clear_e coin p0 sch t b lay extra rep hood hc]
  refine
    { k0 := hk0
      n0 := hn0
      levels := hlv
      ood := ?_
      msg := ?_
      e0 := ?_
      e1 := ?_
      y1 := hy1
      ybar := hybar
      salt := hsalt }
  · have hlo : lay.zp 64 ≤ lay.lgOod 0 := by simp only [Zk.PadLayout.lgOod]; omega
    have hhi : lay.lgOod 0 < lay.hab := by simp only [Zk.PadLayout.lgOod, Zk.PadLayout.hab]; omega
    rw [hl0 (lay.lgOod 0) hlo hhi, l0Clear_ood]
  · intro j hj
    have hj' : j < lay.k0 := hlk ▸ hj
    simp only [mmOf, maskMsg]
    rw [hmsg j hj' 0 (by omega), hmsg j hj' 1 (by omega), hmsg j hj' 2 (by omega), hmsg j hj' 3 (by omega)]
    generalize roundPoly (tabS coin p0 t j) (wS coin p0 b sch.n j) = M
    rfl
  · rw [he 0 (by omega) 0 (by omega), he 0 (by omega) 1 (by omega), Nat.add_zero]
    generalize dotK (wS coin p0 b sch.n sch.k0)
      (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep)))) = E
    rfl
  · rw [he 1 (by omega) 0 (by omega), he 1 (by omega) 1 (by omega)]
    generalize dotK (wS coin p0 b sch.n sch.k0)
      (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + 1)))) = E
    rfl

end LigLock

end FlockComplete

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **`repCut_fw`'s `hlig` at the switched source, at the firewall's request `q`**, but for the restart: at the
firewall's clear values `valsFw x c A B` whose level 0 part `B j r` is `l0Clear` at the coins from round `p₀`, the
packed table `t` and the basis `b`, at `pads_frame`'s premises, where table `j`'s rows hold at its base, rep `r`'s source
holds level 0's opening at the request's pads (`Finish.padsOf q.draws`), given the schedule's facts and the source's
`y₁'`, `ȳ` and salts. -/
theorem swSrcE_ligHyp_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.Firewall.Finish.RepIn)
    (coin : ℕ → ℕ → _root_.Flock.F128) (n p0 : ℕ) (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule)
    (t b : Array _root_.Flock.F128) (rp tPad : ℕ) (extra lanePad : ℕ → ℕ → _root_.Flock.F128)
    (salt : ℕ → ℕ → ByteArray)
    (hB : B j r = FlockComplete.LigLock.l0Clear coin p0 sch t b (layE x j) extra rp)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (hk0 : 1 ≤ sch.k0) (hn0 : sch.n ≠ 0) (hlv : 2 ≤ sch.levels.size) (hlk : (zkTab x j).sch.k0 = sch.k0)
    (hy1 : p0 + 3 + sch.k0 < n → _root_.Flock.Zk.rhoIndependent (FlockComplete.LigLock.rhoS coin p0 sch.k0) = true →
      ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).y1 =
        (Array.range (FlockComplete.LigLock.tabS coin p0 t sch.k0).size).map fun i =>
          (FlockComplete.LigLock.tabS coin p0 t sch.k0)[i]! +
            (FlockComplete.LigLock.rhoS coin p0 sch.k0).1.smul
              ((Array.range (FlockComplete.LigLock.tabS coin p0 t sch.k0).size).map (extra (2 * rp)))[i]! +
            (FlockComplete.LigLock.rhoS coin p0 sch.k0).2.smul
              ((Array.range (FlockComplete.LigLock.tabS coin p0 t sch.k0).size).map (extra (2 * rp + 1)))[i]!)
    (hybar : p0 + 3 + sch.k0 < n →
      _root_.Flock.Zk.rhoIndependent (FlockComplete.LigLock.rhoS coin p0 sch.k0) = true →
      ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).ybar = (Array.range tPad).map fun i =>
        (List.range (2 ^ (sch.levels.getD 0 default).lanes)).foldl
          (fun acc l => acc + ((_root_.Flock.eqTableK
            ((Array.range sch.k0).map (FlockComplete.LigLock.chS coin p0)))[l]!).smul (lanePad l i))
          ((FlockComplete.LigLock.rhoS coin p0 sch.k0).1.smul
              (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rp) i) +
            (FlockComplete.LigLock.rhoS coin p0 sch.k0).2.smul
              (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rp + 1) i)))
    (hsalt : p0 + 3 + sch.k0 < n →
      _root_.Flock.Zk.rhoIndependent (FlockComplete.LigLock.rhoS coin p0 sch.k0) = true →
      ∀ i, 1 ≤ i → i < sch.levels.size → ∀ p < 2 ^ (sch.levels.getD i default).d,
        FlockComplete.fitBytes ms.saltBytes (((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).hiSalt i p) =
          FlockComplete.fitBytes ms.saltBytes (salt i p)) :
    FlockComplete.LigLock.LigHyp coin n p0 ms sch t b (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).h rp
      tPad extra lanePad salt ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val) :=
  FlockComplete.LigLock.ligHyp_of_l0 coin n p0 ms sch t b (layE x j)
    (_root_.Flock.Firewall.Finish.padsOf q.draws).h rp tPad extra lanePad salt
    ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val) hk0 hn0 hlv rfl hlk
    (fun i hi0 hi => by rw [swSrcE_l0_frame L ρ t₀ rep x c w A B τ j r q hq hlay hz htape hrows i hi0 hi, hB])
    hy1 hybar hsalt

end Masked
end Claims.ZKMal
