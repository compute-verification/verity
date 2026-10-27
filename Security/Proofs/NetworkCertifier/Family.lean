import Proofs.NetworkCertifier.Accepted
import Proofs.Properties.CertifierNetwork.Instance

/-!
# A spatial unit of several link pairs: each pair's run is the Program's, under one clock

`AcceptedRunsAreProgramRuns` is the bridge for one link pair, the device's. A spatial unit with `L` link pairs is `L`
device inputs, each accepted by the check under its own parameters (`deviceParams`), and the reference reads one
clock per spatial unit (`NetTiming.Obs`' one σ for every egress link). `UnitRunsAreProgramRuns` is the single-pair
theorem on every pair, together with the shared-sync premise (`SharedSync`, the pairs' egress links declare one sync per
window): every pair's run is the Program's (`RunIsProgramRun`), and one sequence of syncs `s` is the one every pair's
egress link declares (`UnitClock`), so the advice σ of each pair's Program is `s` window by window; or some pair's run
carries a break, as in the single-pair theorem.

What discharges `SharedSync` is the check of the unit's links: `NetworkCheck.Check`'s `advice` test on a unit of several
egress links (each declared sync is the first egress link's, `SyncAdmitted`), or, on the period's clock form, a check
that a window's clock certificates of the unit's links sign one digest.

Its two lemmas: `unitRunsAreProgramRuns_honest`, that the premises hold for `L` pairs each the toy run of
`Properties.CertifierNetwork.Instance` (`Digested` is its one hypothesis, as there; `L = 1` is blocks/v1's unit, one
server on one uplink), and `unitRunsAreProgramRuns_nontrivial`, that a check accepting everything doesn't have the
property: it accepts a pair whose evidence opens no egress link, so no clock is declared.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkCertifier (ClockSync)

namespace Properties.NetworkCertifier

/-- What an accepted run of one pair carries in place of the Program's run (`AcceptedRunsAreProgramRuns`): two distinct
strings it hashes with one SHA-512 digest, or a certificate of the commitment that verifies over its message in the
run's session though the device never signed that message in the run. -/
def Breaks (x : DeviceInput) (out : DeviceOutput) (c : Commitment) (e : Evidence) : Prop :=
  (∃ y ∈ runStrings x out c e, ∃ z ∈ runStrings x out c e, y ≠ z ∧ Flock.Sha512.hash y = Flock.Sha512.hash z) ∨
    ∃ r ∈ c.records.toList, ∃ m, certMessage x.issuer.session r = some m ∧
      LmsVerifies (deviceParams x) m r.certificate.signature ∧ ¬ Signed x.issuer.session out m

/-- **One sync per window across the unit's pairs**: wherever two pairs' egress links declare a sync for a window
`w < W`, it is one sync. -/
def SharedSync {L : Nat} (W : Nat) (e : Fin L → Evidence) : Prop :=
  ∀ i j li lj, (e i).links[0]? = some li → (e j).links[0]? = some lj → ∀ w < W, li.syncs[w]? = lj.syncs[w]?

/-- The unit's clock is `s`: every pair's egress link declares the syncs `s` over the `W` windows. -/
def UnitClock {L W : Nat} (e : Fin L → Evidence) (s : Fin W → ClockSync) : Prop :=
  ∀ i, ∃ link, (e i).links[0]? = some link ∧ Declares link s

/-- `UnitRunsAreProgramRuns` for a check that accepts what `accepts` does. -/
def UnitRunsAreProgramRunsBy (accepts : NetworkCheck.Check.Params → Commitment → Evidence → Prop) : Prop :=
  ∀ (device : DeviceInput → DeviceOutput) {L : Nat} (W : Nat) (x : Fin L → DeviceInput) (c : Fin L → Commitment)
    (e : Fin L → Evidence),
    (∀ i, (x i).config.windows = W) → (∀ i, accepts (deviceParams (x i)) (c i) (e i)) →
    NetTiming.DeviceIntact device deviceCode → SharedSync W e →
    (∃ s : Fin W → ClockSync, UnitClock e s ∧ ∀ i, RunIsProgramRun (x i) (device (x i)) (e i)) ∨
      ∃ i, Breaks (x i) (device (x i)) (c i) (e i)

/-- **A unit's accepted runs are runs of the certifier program under one clock.** For `L` link pairs of `W` windows
each, each pair's commitment and evidence accepted by the check under its device's parameters, a device that runs as
built and pairs whose egress links declare one sync per window (`SharedSync`): every pair's run is the Program's
(`RunIsProgramRun`) and one clock `s` is every pair's declared syncs (`UnitClock`), or some pair's run carries a break
(`Breaks`). -/
def UnitRunsAreProgramRuns : Prop := UnitRunsAreProgramRunsBy protocol.Accepts

end Properties.NetworkCertifier

namespace Proofs.NetworkCertifier

open Properties.NetworkCertifier
open Properties.NetworkCertifier.CheckProofs (declares_declared)

/-- An accepted pair's evidence opens its one egress link, which declares a sync for each of the device's windows. -/
theorem first_link {x : DeviceInput} {c : Commitment} {e : Evidence} (hacc : protocol.Accepts (deviceParams x) c e) :
    ∃ link, e.links[0]? = some link ∧ link.syncs.size = x.config.windows := by
  obtain ⟨hfit, -⟩ := acceptsLinkWindows _ c e hacc
  have h0 : 0 < e.links.size := by rw [hfit.1]; exact Nat.one_pos
  exact ⟨e.links[0], Array.getElem?_eq_getElem h0, hfit.2.1 _ (Array.mem_toList_iff.mpr (Array.getElem_mem h0))⟩

theorem unitRunsAreProgramRuns : UnitRunsAreProgramRuns := by
  intro device L W x c e hW hacc hint hsh
  by_cases hb : ∃ i, Breaks (x i) (device (x i)) (c i) (e i)
  · exact Or.inr hb
  have hrun : ∀ i, RunIsProgramRun (x i) (device (x i)) (e i) := fun i =>
    (acceptedRunsAreProgramRuns device (x i) (c i) (e i) (hacc i) hint).resolve_right fun h => hb ⟨i, h⟩
  refine Or.inl ?_
  rcases Nat.eq_zero_or_pos L with hL | hL
  · subst hL
    exact ⟨fun _ => noSync, fun i => i.elim0, hrun⟩
  obtain ⟨l₀, hl₀, hs₀⟩ := first_link (hacc ⟨0, hL⟩)
  have hd₀ : Declares l₀ (declared W l₀) := declares_declared (hs₀.trans (hW ⟨0, hL⟩))
  refine ⟨declared W l₀, fun i => ?_, hrun⟩
  obtain ⟨li, hli, -⟩ := first_link (hacc i)
  exact ⟨li, hli, fun v => (hsh i ⟨0, hL⟩ li l₀ hli hl₀ v.1 v.2).trans (hd₀ v)⟩

/-! ## Its honest and nontrivial lemmas -/

/-- The toy run of `Properties.CertifierNetwork.Instance` as a device input: its configuration, issuer, coins and
trace, at leaf 0 with no session started. -/
def toy : DeviceInput :=
  ⟨Properties.CertifierNetwork.Instance.config, Properties.CertifierNetwork.Instance.issuer, 0, [],
    Properties.CertifierNetwork.Instance.salts, Properties.CertifierNetwork.Instance.trace⟩

/-- **The premises hold on a unit of toy pairs** (the honest lemma): `L` pairs, each the toy run, its certificates'
commitment and its shaper's evidence, accepted under its device's parameters, the device as built, and one sync per
window across the pairs, given the toy configuration's parameters digests (`Digested`, which no proof evaluates). -/
theorem unitRunsAreProgramRuns_honest
    (hdg : Properties.CertifierNetwork.Digested Properties.CertifierNetwork.Instance.config) (L : Nat) :
    ∃ (x : Fin L → DeviceInput) (c : Fin L → Commitment) (e : Fin L → Evidence),
      (∀ i, (x i).config.windows = 1) ∧ (∀ i, protocol.Accepts (deviceParams (x i)) (c i) (e i)) ∧
      NetTiming.DeviceIntact deviceCode deviceCode ∧ SharedSync 1 e := by
  obtain ⟨_, crs, -, hacc⟩ := Properties.CertifierNetwork.Instance.deviceOutputAccepted_instance hdg
  refine ⟨fun _ => toy, fun _ => Properties.CertifierNetwork.commitment crs .empty,
    fun _ => Properties.CertifierNetwork.evidence Properties.CertifierNetwork.Instance.config
      Properties.CertifierNetwork.Instance.trace,
    fun _ => rfl, fun _ => hacc _, fun _ => rfl, fun i j li lj hi hj w _ => ?_⟩
  obtain rfl : li = lj := Option.some.inj (hi.symm.trans hj)
  rfl

/-- The toy configuration with no events: the device publishes no record and signs nothing. -/
def idle : DeviceInput := { toy with trace := [] }

def noRecords : Commitment := ⟨#[], .empty⟩

def noLinks : Evidence := ⟨#[]⟩

theorem deviceCode_idle : deviceCode idle = ⟨[], []⟩ := rfl

theorem runStrings_idle : runStrings idle (deviceCode idle) noRecords noLinks = [] := rfl

/-- **No check that accepts everything has `UnitRunsAreProgramRuns`' property** (its nontrivial lemma): it accepts one
idle pair whose evidence opens no egress link, so no clock is the pair's declared syncs, and the run hashes nothing
and its commitment holds no certificate, so it carries no break. -/
theorem unitRunsAreProgramRuns_nontrivial :
    ∀ accepts : NetworkCheck.Check.Params → Commitment → Evidence → Prop, (∀ p c e, accepts p c e) →
      ¬ UnitRunsAreProgramRunsBy accepts := by
  intro accepts hall h
  rcases h deviceCode (L := 1) 1 (fun _ => idle) (fun _ => noRecords) (fun _ => noLinks) (fun _ => rfl)
      (fun _ => hall _ _ _) (fun _ => rfl) (fun _ _ _ _ hi => by simp [noLinks] at hi) with
    ⟨s, hs, -⟩ | ⟨i, hb⟩
  · obtain ⟨link, hl, -⟩ := hs 0
    simp [noLinks] at hl
  · rcases hb with ⟨y, hy, -⟩ | ⟨r, hr, -⟩
    · simp [runStrings_idle] at hy
    · simp [noRecords] at hr

end Proofs.NetworkCertifier
