import Proofs.Properties.CertifierNetwork.Runs
import Proofs.NetworkCertifier.Code
import Proofs.NetworkCertifier.Schedule

/-!
# An accepted run is a run of the certifier program (5b)

`Properties.NetworkCertifier.AcceptedRunsAreProgramRuns`, proved from what an accepted run's records are
(`Properties.NetworkCertifier.acceptedRowsAreProgramRows`, which holds unless the run yields a SHA-512 collision or an
LMS forgery, as this does) and the program's `expectedRow` being the reference's (`CodeExpectedRow`). The syncs the
evidence declares are each in Σ_sync (the check's `advice` test, through `acceptsLinkWindows`), so they are advice of
the reference's constant-rate Program; the device's egress records are the Program's grids, whose rows, read back as
bits, are `expectedRow`'s (`bytesGrid_slot`); and its records of every link-window are complete.

`EncardAcceptedObservationsLe` follows: accepted runs' observation sequences are among Theorem 3's `AccSeqs` of that
Program, so they number at most `#Ac · |Σ_sync|^W · D^(W·T)` (`NetTiming.encard_accSeqs_le_constantRate`), unless some
accepted run yields a collision or a forgery.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkCertifier (ClockSync expectedRow)
open Properties.NetworkCertifier
open Properties.NetworkCertifier.GridProofs (expectedRow_slot payloads_length)
open Properties.NetworkCertifier.CheckProofs (declares_declared)

namespace Proofs.NetworkCertifier

/-- A grid of rows of bits, read back at bucket `t` of window `t / T`, is the row at `t`, where each slot's bits are
whole bytes. -/
theorem bytesGrid_slot {r T : Nat} (row : Nat → Fin r → Option (List Bool)) {t : Nat} (hT : 0 < T) (s : Fin r)
    (hb : ∀ f, row t s = some f → f.length % 8 = 0) :
    (((bytesGrid T row (t / T))[t % T]?.bind fun x => x[s.1]?).join).map bitsOfBytes = row t s := by
  have ht : t / T * T + t % T = t := by rw [Nat.mul_comm]; exact Nat.div_add_mod t T
  simp only [bytesGrid, Array.getElem?_map, List.getElem?_toArray, List.getElem?_range (Nat.mod_lt t hT),
    Option.map_some, Option.bind_some, List.getElem?_ofFn, s.2, ht]
  cases h : row t s with
  | none => simp [h]
  | some f => simp [h, Properties.NetworkCertifier.Bytes.bitsOfBytes_bytesOfBits (hb f h)]

theorem acceptedRunsAreProgramRuns : AcceptedRunsAreProgramRuns := by
  intro device x c e hacc hint
  rcases acceptedRowsAreProgramRows device x c e hacc hint with hpr | hb
  case inr => exact Or.inr hb
  refine Or.inl ?_
  intro K In Ac frames a op o linkOf hlink hlen hobs hopen
  obtain ⟨hcov, hcomp, hgrid⟩ := hpr
  obtain ⟨hfit, eg, ig, -, he, -⟩ := acceptsLinkWindows _ c e hacc
  have h0 : 0 < e.links.size := by rw [hfit.1]; exact Nat.one_pos
  have hl0 : e.links[0]? = some e.links[0] := Array.getElem?_eq_getElem h0
  have hmem : e.links[0] ∈ e.links.toList := Array.mem_toList_iff.mpr (Array.getElem_mem h0)
  have hm := zip_index hfit ⟨0, Nat.one_pos⟩ hl0
  have hσ : ∀ w : Fin x.config.windows, ∃ v : (refSyncSet x.config.syncSet).Elem,
      ∃ s, e.links[0].syncs[w.1]? = some s ∧ v.1.1 = s.delta ∧ v.2.1 = s.rho := by
    intro w
    obtain ⟨-, -, -, ⟨s, hs, hd, hr, -⟩, -⟩ := he _ _ hm w.1 w.2
    exact ⟨(⟨s.delta, List.mem_toFinset.mpr hd⟩, ⟨s.rho, List.mem_toFinset.mpr hr⟩), s, hs, rfl, rfl⟩
  choose σ hσ using hσ
  refine ⟨σ, fun w link hl => ?_, fun t ht => ?_⟩
  · obtain rfl : e.links[0] = link := Option.some.inj (hl0.symm.trans hl)
    exact hσ w
  have hto : t < o.length := hlen ▸ ht
  have hT : 0 < x.config.egress.buckets := Nat.pos_of_ne_zero fun h0 => by simp [h0] at ht
  have hw : t / x.config.egress.buckets < x.config.windows := (Nat.div_lt_iff_lt_mul hT).mpr ht
  refine ⟨hto, ?_, fun κ => ?_⟩
  · obtain ⟨r, hr, hrl, hrw⟩ := hcov _ (List.mem_cons_self) _ hw
    have hdecl := declares_declared (W := x.config.windows) (hfit.2.1 _ hmem)
    rw [hobs.1 t hto r hr hrl hrw, hgrid _ hl0 _ hdecl r hr hrl (hrw ▸ hw), hrw]
    funext ℓ s
    refine Eq.trans (bytesGrid_slot (expectedRow x.config.egress.allocation x.config.egress.buckets x.config.windows
      (declared x.config.windows e.links[0]) e.links[0].frames) hT s (fun f hf => by
        obtain ⟨k, rfl⟩ := expectedRow_slot hf
        rcases payloads_length (hfit.2.2 _ hmem) k with h8 | h8
        · rw [h8]; omega
        · rw [h8]; rfl)) ?_
    have hℓ : ℓ = 0 := Subsingleton.elim _ _
    subst hℓ
    refine congrFun (NetTiming.SecurityProofs.CodeExpectedRow _ _ _ _ (fun w => (refSyncSet x.config.syncSet).sync (σ w))
      (fun w => ?_) _ (frames (NetTiming.ingressUpTo op o t) a 0) (hopen t ht _ hl0).symm t) s
    obtain ⟨s', hs, hd, hr⟩ := hσ w
    have es : declared x.config.windows e.links[0] w = s' := by
      rw [declared, Array.getD_eq_getD_getElem?, hs]; rfl
    rw [es]
    exact ⟨hd.symm, hr.symm⟩
  · obtain ⟨r, hr, hrl, hrw⟩ := hcov _ (hlink κ) _ hw
    exact (hobs.2 t hto κ r hr hrl hrw).mpr (hcomp r hr (hrl ▸ hlink κ) (hrw ▸ hw))

theorem encardAcceptedObservationsLe : EncardAcceptedObservationsLe := by
  intro device cfg key K In Ac _ frames op linkOf D hint
  by_cases hb : ∃ o x c e a, AcceptedRun cfg key device frames op linkOf o x c e a ∧
      ((∃ y ∈ runStrings x (device x) c e, ∃ z ∈ runStrings x (device x) c e, y ≠ z ∧
          Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
        ∃ r ∈ c.records.toList, ∃ m, certMessage x.issuer.session r = some m ∧
          LmsVerifies (deviceParams x) m r.certificate.signature ∧ ¬ Signed x.issuer.session (device x) m)
  · exact Or.inr hb
  refine Or.inl (le_trans (Set.encard_le_encard ?_)
    (NetTiming.encard_accSeqs_le_constantRate (S := cfg.egress.allocation) cfg.egress.buckets cfg.windows
      (refSyncSet cfg.syncSet) frames op _ D))
  rintro o ⟨hlen, hjit, x, c, e, a, hrun⟩
  obtain ⟨hx, hk, hacc, hlink, hobs, hopen⟩ := hrun
  subst hx hk
  rcases acceptedRunsAreProgramRuns device x c e hacc hint with hr | hbr
  · obtain ⟨σ, -, hA⟩ := hr frames a op o linkOf hlink hlen hobs hopen
    exact ⟨hlen, hjit, (a, σ), hA⟩
  · exact (hb ⟨o, x, c, e, a, ⟨rfl, rfl, hacc, hlink, hobs, hopen⟩, hbr⟩).elim

end Proofs.NetworkCertifier
