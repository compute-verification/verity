module

public import Proofs.Flock.Soundness.Refine.RepConverse
public import Proofs.Flock.VBridge.Algebra.Ligerito
public import Proofs.Flock.VBridge.Algebra.Lincheck
public import Proofs.Flock.VBridge.Compiled
public import Proofs.Flock.VBridge.Runs.Rs
public import Proofs.Flock.VBridge.Runs.ZcLc
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! `repProg` at a rep from E's preconditions (`prog_of_algebra`). -/

@[expose] public section

namespace FlockVBridge.Runs

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Compiled

theorem prog_of_algebra (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d) (hacc : AccHolds sh d)
    (st : Setup) (sch : Flock.Schedule) (p : Proof) (extras : Array Claim) (t : Transcript) (ht : t.k = 0)
    (hd : d.rounds = t.rounds.map (·.coins)) (hne : NoEmpty d.rounds) (hm : st.m = sh.m) (hm14 : 14 ≤ st.m)
    (hkL : st.fold.kLog = sh.kLog) (hkS : st.kLog = sh.kLog) (hkm : st.kLog ≤ st.m) (hxs : d.extras = extras)
    (hext : ∀ j < extras.size, extras[j]!.skip.size = 64 ∧ extras[j]!.x.size = st.m - 6)
    (hzp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) p.zerocheck)
    (hlp : LcProofOf (decoded sh d.w d.comb d.cons) p.lincheck) (hpl : sh.holo = false)
    (hrsw : p.ringSwitches = (decoded sh d.w d.comb d.cons).rs)
    (hfoldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
        .ok d.comb)
    (hn₀ : (t.rounds[t.k]!).n = 6) (hn₁ : (t.rounds[t.k + 1]!).n = st.m - 13) (hn₂ : (t.rounds[t.k + 2]!).n = 1)
    (hnr : ∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1)
    (ha₀ : AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
      .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6]))
    (ha₁ : AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)]))
    (ha₂ : AnswersL (t.rounds[t.k + 2]!)
      (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1]))
    (har : ∀ i < st.m - 6, AnswersL (t.rounds[t.k + 3 + i]!)
      (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1]))
    (hlc : LcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck)
    (hrn : ∀ k < 2 + extras.size, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).n = 7)
    (hra : ∀ k < 2 + extras.size, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).answers
      (rsFramed (rsPending ByteArray.empty k) p.ringSwitches[k]!))
    (hgn : (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).n = 2 + extras.size)
    (hga : (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size)]!).answers
      (ByteArray.empty ++ Transcript.header 4 2 (2 + extras.size)))
    {t₄ : Transcript}
    (hlg : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      Flock.ligerito st.merkle sch (valOf sh d.w d.comb d.cons .T) p.cap p.ligerito (Flock.basis info.rs.sws)
        ⟨t.rounds, t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (2 + extras.size) + 1, ByteArray.empty⟩ =
          .ok ((), t₄)) :
    repProg st sch p extras t = .ok ((), t₄) := by
  obtain ⟨v, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩ := halg
  obtain ⟨hcl, hrs, hshape, hval, hR, hT, hSws⟩ :=
    rs_of_residuals hrows hS hw hc (hacc v info kappa hrows) hres hzero hm14 hm hkS hkL hkm hxs hext hd hne ht
      (hzp v info kappa hrows).fc hlp.zp hrsw rfl rfl hrn hra hgn hga
  obtain ⟨hr, hz, hrho, -, -, -, -, hts, -⟩ := rows_coins hrows (hm ▸ hm14) ht hd hne
  rw [hr, hz, hrho, hts, ← hm, ← hkL] at hcl
  have hsz := claimsF_size sh (valOf sh d.w d.comb d.cons) info extras
  refine prog_of_zclc sh d ⟨v, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩ hacc st sch p extras t ht hd hne hm
    hm14 hkL hpl hzp hlp hfoldPts hn₀ hn₁ hn₂ hnr ha₀ ha₁ ha₂ har hlc hcl hrs hshape hval hR ?_
  rw [hT, hSws, hsz]
  exact hlg v info kappa hrows

end FlockVBridge.Runs
