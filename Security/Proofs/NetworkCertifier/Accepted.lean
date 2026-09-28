import Properties.Proofs.NetworkCertifier.Run
import Proofs.NetworkCertifier.Code
import Proofs.NetworkCertifier.Schedule

/-!
# An accepted run is a run of the certifier program (5b)

`Properties.NetworkCertifier.AcceptedRunsAreProgramRuns`, proved from what an accepted run's rows are
(`Properties.NetworkCertifier.acceptedRowsAreProgramRows`, which holds unless the run yields a SHA-512 collision or an
LMS forgery, as this does) and the program's `expectedRow` being the reference's (`CodeExpectedRow`). The declared syncs
are one value of Σ_sync per window, the same on every link (the check's `advice` and one-clock tests, through
`acceptsIff`), so they are advice of the reference's constant-rate Program, and the device's rows and statuses are what
its `Accepted` asks.

`EncardAcceptedObservationsLe` follows: accepted runs' observation sequences are among Theorem 3's `AccSeqs` of that
Program, so they number at most `#Ac · |Σ_sync|^W · D^(W·T)` (`NetTiming.encard_accSeqs_le_constantRate`), unless some
accepted run yields a collision or a forgery.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkCertifier (ClockSync)
open Properties.NetworkCertifier

namespace Proofs.NetworkCertifier

theorem acceptedRunsAreProgramRuns : AcceptedRunsAreProgramRuns := by
  intro p c e dev eg ig K In Ac ss frames a op o linkOf hacc hpd hhon hdel hrho hlen hrows hstat hlink hfr
  rcases acceptedRowsAreProgramRows p c e dev eg ig hacc hpd hhon with ⟨hrow, hst⟩ | hb
  case inr => exact Or.inr hb
  refine Or.inl ?_
  obtain ⟨hfit, eg', ig', -, he, -⟩ := (acceptsIff p c e).mp hacc
  have adm : ∀ (ℓ : Fin p.egress.size) link, e.links[ℓ.1]? = some link → ∀ w < p.windows,
      SyncAdmitted p e link w := by
    intro ℓ link hl w hw
    obtain ⟨-, -, -, hadm, -⟩ := he _ _ (zip_index hfit ℓ hl) w hw
    exact hadm
  have hσ : ∀ w : Fin p.windows, ∃ v : ss.Elem, ∀ (ℓ : Fin p.egress.size) link, e.links[ℓ.1]? = some link →
      ∃ s, link.syncs[w.1]? = some s ∧ v.1.1 = s.delta ∧ v.2.1 = s.rho := by
    intro w
    by_cases h0 : 0 < p.egress.size
    · have h0' : 0 < e.links.size := by rw [hfit.1]; exact h0
      have hl0 : e.links[0]? = some e.links[0] := Array.getElem?_eq_getElem h0'
      obtain ⟨s, hs, hd, hr, -⟩ := adm ⟨0, h0⟩ _ hl0 w.1 w.2
      refine ⟨(⟨s.delta, (hdel _).mp hd⟩, ⟨s.rho, (hrho _).mp hr⟩), fun ℓ link hl => ?_⟩
      obtain ⟨s', hs', -, -, first, s₀, hf, hf0, hd0, hr0⟩ := adm ℓ link hl w.1 w.2
      have e1 : e.links[0] = first := Option.some.inj (hl0.symm.trans hf)
      subst e1
      have e0 : s₀ = s := Option.some.inj (hf0.symm.trans hs)
      subst e0
      exact ⟨s', hs', hd0, hr0⟩
    · obtain ⟨d, hd⟩ := ss.deltas_nonempty
      obtain ⟨ρ, hρ⟩ := ss.rhos_nonempty
      exact ⟨(⟨d, hd⟩, ⟨ρ, hρ⟩), fun ℓ _ _ => (h0 (Nat.zero_lt_of_lt ℓ.2)).elim⟩
  choose σ hσ using hσ
  refine ⟨σ, hσ, fun t ht => ?_⟩
  have hto : t < o.length := hlen ▸ ht
  refine ⟨hto, ?_, fun κ => (hstat t hto κ).mpr (hst _ _ (hlink κ) _ ?_)⟩
  · rw [hrows t hto]
    funext ℓ
    have hℓ : ℓ.1 < e.links.size := by rw [hfit.1]; exact ℓ.2
    have hl : e.links[ℓ.1]? = some e.links[ℓ.1] := Array.getElem?_eq_getElem hℓ
    have hdecl := CheckProofs.declares_declared (W := p.windows) (hfit.2.1 _ (Array.mem_def.mp (Array.getElem_mem hℓ)))
    rw [hrow ℓ _ hl _ hdecl t ht]
    refine NetTiming.SecurityProofs.CodeExpectedRow _ _ _ _ (fun w => ss.sync (σ w)) (fun w => ?_) _
      (frames (NetTiming.ingressUpTo op o t) a ℓ) (hfr t ht ℓ _ hl).symm t
    obtain ⟨s, hs, hd, hr⟩ := hσ w ℓ _ hl
    have es : declared p.windows e.links[ℓ.1] w = s := Option.some.inj ((hdecl w).symm.trans hs)
    rw [es]
    exact ⟨hd.symm, hr.symm⟩
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact ht)

theorem encardAcceptedObservationsLe : EncardAcceptedObservationsLe := by
  intro p K In Ac _ ss frames op linkOf D hdel hrho hlink
  by_cases hb : ∃ o c e dev eg ig a, AcceptedRun p frames op linkOf o c e dev eg ig a ∧
      ((∃ x ∈ runStrings p c e dev eg, ∃ y ∈ runStrings p c e dev eg, x ≠ y ∧
          Flock.Sha512.hash x = Flock.Sha512.hash y) ∨
        ∃ r ∈ c.records.toList, ∃ m, certMessage r = some m ∧ LmsVerifies p m r.certificate.signature ∧ ¬ Signed dev m)
  · exact Or.inr hb
  refine Or.inl (le_trans (Set.encard_le_encard ?_)
    (NetTiming.encard_accSeqs_le_constantRate (S := p.grid.allocation) p.grid.buckets p.windows ss frames op _ D))
  rintro o ⟨hlen, hjit, c, e, dev, eg, ig, a, hrun⟩
  obtain ⟨hacc, hpd, hhon, hrows, hstat, hfr⟩ := hrun
  rcases acceptedRunsAreProgramRuns p c e dev eg ig ss frames a op o linkOf hacc hpd hhon hdel hrho hlen hrows hstat
    hlink hfr with ⟨σ, -, hA⟩ | hbr
  · exact ⟨hlen, hjit, (a, σ), hA⟩
  · exact (hb ⟨o, c, e, dev, eg, ig, a, ⟨hacc, hpd, hhon, hrows, hstat, hfr⟩, hbr⟩).elim

end Proofs.NetworkCertifier
