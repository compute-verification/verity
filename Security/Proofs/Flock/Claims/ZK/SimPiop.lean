import Proofs.Flock.Claims.ZK.SimPiopGo
import Proofs.Flock.Claims.ZK.Split

/-!
# `SimPiop` from the deployed source's values

`SimPiop` (`HybridStep 0`): `repMix 0` and `repMix 1` differ only in the PIOP, the cut's (`piopCut`) against the
session's (`piopPZk`), so they send the same wherever those two do (`FlockComplete.repMix_zero_one`), and those two do
wherever the cut's source is at the cut's coins (`FlockComplete.piopCut_alike`). What is left is `PiopSrcE`: at every
live statement, witness in the deployed relation, tape, opened coins and stop with no refusal seen, table `j` has at
least `14` variables and its rep `r` source from the cut (`srcE`) is at the opened coins from the rep's first round on
(`FlockComplete.PiopSrc`), at the session's pads and trace and the statement's extra claims (`simPiop_of`).
-/

namespace FlockComplete

open Flock Transcript

theorem go_bind_congr {α β : Type} {x y : TMW PS α} {f : α → TMW PS β} {t : Transcript} {l : Array Claims.Msg}
    (h : (x.run t).go l = (y.run t).go l) : ((x >>= f).run t).go l = ((y >>= f).run t).go l := by
  rw [StateT.run_bind, StateT.run_bind, go_bind, go_bind, h]

/-- **`repMix 0` sends what `repMix 1` sends** wherever the cut's PIOP does what the session's does: the two differ
only there. -/
theorem repMix_zero_one (io : Asks PS) (hs : Hashes PS) (st : Setup) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) (src : RepSrc) (t : Transcript) (l : Array Claims.Msg)
    (h : ((piopCut io st (Zk.PadLayout.new st.m st.kLog sch.k0) src (flat cm0.cap) ic.root extras prior words).run
        t).go l =
      ((piopPZk io st (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr (flat cm0.cap) ic.root extras prior words).run
        t).go l) :
    ((repMix 0 io hs st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words src).run t).go l =
      ((repMix 1 io hs st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce prior words src).run
        t).go l := by
  unfold repMix
  simp only [ite_eq_left (show (0 : ℕ) = 0 from rfl), ite_eq_right (show ¬ (1 : ℕ) = 0 by decide),
    ite_eq_left (show (0 : ℕ) ≤ 1 by decide), ite_eq_left (show (1 : ℕ) ≤ 1 by decide),
    ite_eq_left (show (0 : ℕ) ≤ 2 by decide), ite_eq_left (show (1 : ℕ) ≤ 2 by decide)]
  exact go_bind_congr h

end FlockComplete

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **The deployed source at the opened coins**: at a live statement, a witness in the deployed relation, any tape `τ`,
opened coins `c` and stop `n` at which no refusal is seen, table `j` has at least `14` variables, and its rep `r` source
from the cut is at the opened coins from the rep's first round on (`FlockComplete.PiopSrc`), at the session's pads,
trace and the statement's extra claims. -/
def PiopSrcE : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras →
      14 ≤ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m ∧
      FlockComplete.PiopSrc (fun k i => slotAt c k i) (FlockDeploy.tabsOf (stsOf x))[j.val]!
        (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
          (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
        (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)).h
        (trsE x w (UE.tapeZk τ))[j.val]! ((srcE x c w τ j.val).rep r.val) extras (repBase x (2 * j.val + r.val))

/-- **`SimPiop` from the deployed source at the opened coins.** -/
theorem simPiop_of (h : PiopSrcE) : SimPiop := by
  intro x hx w hw τ c n hn s spec hsx j r dom cm0 ic extras he l hl
  obtain ⟨hm, hv⟩ := h x hx w hw τ c n hn s spec hsx j r extras he
  unfold mixE
  apply FlockComplete.repMix_zero_one
  exact (FlockComplete.piopCut_alike _ _ _ _ _ _ _ _ _ _ _ _ hm _ l (by rw [hl]; exact hv)).eq

end Masked

end Claims.ZKMal
