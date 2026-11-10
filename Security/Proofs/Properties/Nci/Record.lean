module

public import Properties.Nci.Record
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Real.Basic

/-!
# The circuit record's properties, from their parts

`complies_of`: the rate rule on a record's circuit (`Complies`) from four statements about the check, that a record it
reads carries its instance's fields (`Owns`) and lays out a circuit (`LaidOut`), that its summary counts every block
(`Counts`), and that its rate rule is the policy's on a summary (`Rates`).

`pinnedOpens_of`: what a pinned reader opens (`PinnedOpens`) from the closure digest's pin (`DigestPins`), the
check's provenance (`Provenance`) and a layout that reads a Definition's signature alone (`LayoutReads`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)
open NciCheck.DefinitionDigest (closureDigest)

theorem complies_of {F : Form} (hown : Owns F) (hlaid : LaidOut F) (hcount : Counts F) (hrate : Rates F) :
    Complies F := by
  intro j tr c hacc
  obtain ⟨hread, hnone⟩ := hacc
  obtain ⟨hΓ, hα⟩ := hrate _ _ _ _ hnone
  have hcar := hown j tr c hread
  have hden : (0 : ℝ) < (F.inst c).den := by
    obtain ⟨-, -, -, -, h⟩ := hcar
    exact_mod_cast h
  refine ⟨hcar, hlaid j tr c hread, fun d hd => ?_⟩
  have hmeets : ∀ u, Meets d d.𝓘 (F.inst c).gamma (((F.inst c).num : ℝ) / (F.inst c).den) u := by
    intro u
    obtain ⟨hfresh, hsize | hempty⟩ := hcount j tr c hread d hd u
    · unfold Meets
      rw [hsize]
      refine ⟨hΓ, ?_⟩
      have h1 : (Nci.novelReads d (d.𝓘.gates u) : ℝ) ≤ (F.summarize c).fresh := by exact_mod_cast hfresh
      have h2 : ((F.summarize c).fresh : ℝ) * (F.inst c).den ≤ (F.inst c).num * (F.summarize c).gates := by
        exact_mod_cast hα
      rw [div_mul_eq_mul_div, le_div_iff₀ hden]
      exact (mul_le_mul_of_nonneg_right h1 hden.le).trans h2
    · unfold Meets
      rw [hempty]
      simp [Nci.novelReads, Verity.Circuit.size, Verity.Circuit.into, Verity.Circuit.width]
  exact ⟨hmeets, by unfold Nci.CompliesRate; exact ⟨d.n, d.𝓘, hmeets⟩⟩

/-- One Definition has one signature. -/
theorem Same.signature {P P' : Prog} {k k' : ℕ} (h : Same P P' k k') :
    ∃ d d', P.defs[k]? = some d ∧ P'.defs[k']? = some d' ∧ d.params = d'.params ∧ d.ret = d'.ret :=
  match h with
  | .intro _ _ d d' hd hd' _ _ _ hparams hret .. => ⟨d, d', hd, hd', hparams, hret⟩

theorem pinnedOpens_of {F : Form} (hpins : DigestPins) (hprov : Provenance F) (hlay : LayoutReads F) :
    PinnedOpens F := by
  intro Opens text₀ j₀ P₀ k₀ d₀ key pin hof hdec₀ hparse₀ hdig₀ hk₀ hd₀ hopens
  intro j tr c P pj r name p hread hdec hparse htid hheld
  obtain ⟨log, run, -, -, hgood⟩ := hprov j tr c P pj hread hdec hparse
  obtain ⟨-, -, hdig, -⟩ :
      IsTheta log run p ∧ pin ∈ (F.inst c).pins ∧ closureDigest pj r.tid = .ok pin ∧ _ := hgood r _ hheld
  rw [htid] at hdig
  by_cases hex : ∃ k, P.index[key]? = some k
  · obtain ⟨k, hk⟩ := hex
    rcases hpins text₀ (F.inst c).program j₀ pj P₀ P key pin k₀ k hdec₀ hdec hparse₀ hparse hdig₀ hdig hk₀ hk
      with hsame | hcol
    · refine Or.inl fun k' d hk' hd => ?_
      have hkk : k' = k := Option.some.inj (hk'.symm.trans hk)
      obtain ⟨e₀, e, he₀, he, hparams, hret⟩ := Same.signature hsame
      have h1 : d₀ = e₀ := Option.some.inj (hd₀.symm.trans he₀)
      have h2 : d = e := by rw [hkk] at hd; exact Option.some.inj (hd.symm.trans he)
      rw [hkk, h1, h2]
      exact ⟨hof P₀ P k₀ k hsame hopens, hlay e e₀ hparams.symm hret.symm⟩
    · exact Or.inr hcol
  · exact Or.inl fun k d hk' _ => absurd ⟨k, hk'⟩ hex

end NciCheck.Properties.Record.SecurityProofs
