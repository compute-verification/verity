import Properties.CertifierNetwork.Runs
import Proofs.Properties.CertifierNetwork.Records
import Proofs.Properties.CertifierNetwork.Certify
import Proofs.Properties.NetworkCertifier.Run

/-!
# An accepted run's records are the Program's (5a)

Every certificate the device's code issues is of a record it published (`issued_spec`, `deviceCode_certificates`: it
issues none where its signing state has started its session): the head `record_head` makes of the record under its
grid's parameters digest, its digest, and the record's link-window. Unless a certificate of the commitment is an LMS
forgery, the message of each one the check accepts, in the run's session, is one the device signed in it; unless two
distinct strings of the run have one SHA-512 digest, that message names the same link-window and signed digest
(`messageInput_inj`), the signed digest the same record digest whatever the two salts (`certificateDigest_inj`), and
that the same head (`head_inj`) (`device_record`). The device publishes one record per link-window
(`unique`, from `Inv`), so each record it published on its links is the one the check accepted: complete, since the
check's is settled, and on egress of the Program's grid (`recordHead_grid_inj`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open Flock
open _root_.CertifierDevice
open NetworkCertifier (ClockSync)
open Properties.CertifierNetwork (published publishes unitParams directionOf gridOf syncSetOf reserve advance certified
  Distinct)
open Properties.CertifierDevice (outputs after)

set_option maxRecDepth 20000

namespace Properties.NetworkCertifier

open Bytes _root_.Properties.NetworkCertifier.Tree GridProofs CheckProofs
open Properties.CertifierNetwork.Proofs (Inv inv_after init_inv windowsOn)
open Properties.CertifierDevice.Proofs.Issuer (except_bind_ok prepare_spec prepare_head)

/-! ## What the device's code issues -/

theorem certify_some {i i' : Issuer} {salt : ByteArray} {leaf : Except String Nat} {r : Record}
    {d : NetworkCheck.Direction} {g : NetworkCheck.Params} {pd : ByteArray} {cr : CertifiedRecord}
    (h : i.certify salt leaf r d g pd = .ok (i', some cr)) :
    NetworkCheck.recordHead r.link d r.window r.grid ((Status.reason? r.status).map reasonValue) g pd true =
        .ok cr.head ∧
      (∃ dg, cr.head.digest = .ok dg ∧ cr.certificate.digest = certificateDigest cr.salt dg) ∧
      cr.certificate.link = r.link ∧ cr.certificate.direction = d ∧ cr.certificate.window = r.window := by
  unfold Issuer.certify at h
  obtain ⟨⟨head, digest, message⟩, hprep, h⟩ := except_bind_ok h
  obtain ⟨-, -, -, -, ⟨R, hR, hdg⟩, -⟩ := prepare_spec hprep
  have hh := prepare_head hprep
  unfold Issuer.finish at h
  cases leaf with
  | error _ => simp [pure, Except.pure] at h
  | ok q =>
    obtain ⟨sig, -, h⟩ := except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq, Option.some.injEq] at h
    obtain ⟨-, rfl⟩ := h
    exact ⟨hh, ⟨R, hR, hdg⟩, rfl, rfl, rfl⟩

/-- Each certificate the code issues is of a record it was given: the head `record_head` makes of it under its grid's
parameters digest, signing that head's digest under the certificate's salt and naming the record's link-window. -/
theorem issued_spec {c : Config} : ∀ {i : Issuer} {next : Nat} {salts : Nat → ByteArray} {rs : List Record}
    {cr : CertifiedRecord},
    cr ∈ issued c i next salts rs → ∃ rec ∈ rs, ∃ pd, parameters (gridOf c rec) (syncSetOf c rec) = .ok pd ∧
      NetworkCheck.recordHead rec.link (directionOf c rec) rec.window rec.grid
        ((Status.reason? rec.status).map reasonValue) (gridOf c rec) pd true = .ok cr.head ∧
      (∃ dg, cr.head.digest = .ok dg ∧ cr.certificate.digest = certificateDigest cr.salt dg) ∧
      cr.certificate.link = rec.link ∧ cr.certificate.direction = directionOf c rec ∧ cr.certificate.window = rec.window
  | _, _, _, [], _, h => nomatch h
  | i, next, salts, r :: rs, cr, h => by
    unfold issued at h
    split at h
    · rename_i i' cr? hc
      rcases List.mem_append.mp h with h | h
      · cases cr? with
        | none => cases h
        | some cr' =>
          rw [Option.toList_some, List.mem_singleton] at h
          subst h
          cases hpd : parameters (gridOf c r) (syncSetOf c r) with
          | error _ => rw [hpd] at hc; cases hc
          | ok pd =>
            rw [hpd] at hc
            exact ⟨r, List.mem_cons_self, pd, hpd, certify_some hc⟩
      · obtain ⟨rec, hrec, rest⟩ := issued_spec h
        exact ⟨rec, List.mem_cons_of_mem _ hrec, rest⟩
    · cases h

/-- The code's certificates are ones its issuer issues: none where the signer's state has started the session. -/
theorem deviceCode_certificates {x : DeviceInput} {cr : CertifiedRecord} (h : cr ∈ (deviceCode x).certificates) :
    cr ∈ issued x.config x.issuer x.next x.salts (published x.config x.trace) := by
  unfold deviceCode at h
  dsimp only at h
  split at h
  · cases h
  · exact h

/-- **A second run under a started session signs nothing**: where the signer's state has started the issuer's session
(it is in `x.sessions`), the device's code issues no certificate. -/
theorem deviceCode_started {x : DeviceInput} (h : x.issuer.session ∈ x.sessions) : (deviceCode x).certificates = [] := by
  unfold deviceCode
  dsimp only
  exact ite_eq_left (List.any_eq_true.2 ⟨_, h, decide_eq_true rfl⟩)

/-! ## One record per link-window -/

theorem nodup_windows {l : String} {P : List Record} {n : Nat} (h : windowsOn l P = List.range n) {r r' : Record}
    (hr : r ∈ P) (hr' : r' ∈ P) (hl : r.link = l) (hl' : r'.link = l) (hw : r.window = r'.window) : r = r' := by
  have hnd : ((P.filter fun r => decide (r.link = l)).map Record.window).Nodup := by
    unfold windowsOn at h
    rw [h]
    exact List.nodup_range
  exact List.inj_on_of_nodup_map hnd (List.mem_filter.mpr ⟨hr, by simpa using hl⟩)
    (List.mem_filter.mpr ⟨hr', by simpa using hl'⟩) hw

/-- With two links, the device publishes at most one record of each link-window on each. -/
theorem unique {c : Config} {tr : List Event} (hd : Distinct c) {r r' : Record} (hr : r ∈ published c tr)
    (hr' : r' ∈ published c tr) (hl : r.link = r'.link) (hlk : r.link ∈ [c.egressLink, c.ingressLink])
    (hw : r.window = r'.window) : r = r' := by
  have h := inv_after hd tr (init c) [] (init_inv c)
  rw [List.nil_append] at h
  change Inv c (after (init c) tr) (published c tr) at h
  rcases List.mem_pair.mp hlk with hlk | hlk
  · exact nodup_windows h.eg hr hr' hlk (hl ▸ hlk) hw
  · exact nodup_windows h.ig hr hr' hlk (hl ▸ hlk) hw

/-! ## The strings of the run -/

theorem mem_run_check {x : DeviceInput} {out : DeviceOutput} {c : Commitment} {e : Evidence} {y : ByteArray}
    (h : CheckString (deviceParams x) c e (egressDigest (deviceParams x)) y) : y ∈ runStrings x out c e :=
  List.mem_append_left _ (mem_checkStrings h)

theorem mem_run_head {x : DeviceInput} {out : DeviceOutput} {c : Commitment} {e : Evidence} {cr : CertifiedRecord}
    (hcr : cr ∈ out.certificates) {y : ByteArray} (h : HeadString cr.head y) : y ∈ runStrings x out c e :=
  List.mem_append_right _ (List.mem_append_left _
    (List.mem_flatMap.mpr ⟨cr, hcr, List.mem_append_left _ (List.mem_append_left _ (mem_headStrings h))⟩))

theorem mem_run_salted {x : DeviceInput} {out : DeviceOutput} {c : Commitment} {e : Evidence}
    {cr : CertifiedRecord} (hcr : cr ∈ out.certificates) {y : ByteArray} (h : SaltedString cr.salt cr.head y) :
    y ∈ runStrings x out c e :=
  List.mem_append_right _ (List.mem_append_left _
    (List.mem_flatMap.mpr ⟨cr, hcr, List.mem_append_left _ (List.mem_append_right _ (mem_saltedStrings h))⟩))

theorem mem_run_message {x : DeviceInput} {out : DeviceOutput} {c : Commitment} {e : Evidence}
    {cr : CertifiedRecord} (hcr : cr ∈ out.certificates) {y : ByteArray}
    (h : messageInput cr.certificate.keyId x.issuer.session cr.certificate.link cr.certificate.direction
      cr.certificate.window cr.certificate.digest = .ok y) : y ∈ runStrings x out c e := by
  refine List.mem_append_right _ (List.mem_append_left _ (List.mem_flatMap.mpr ⟨cr, hcr, List.mem_append_right _ ?_⟩))
  rw [h]
  exact List.mem_singleton_self _

theorem mem_run_tree {x : DeviceInput} {out : DeviceOutput} {c : Commitment} {e : Evidence} {r : Record}
    (hr : r ∈ out.records) {y : ByteArray} (h : TreeString (gridOf x.config r) r.grid y) :
    y ∈ runStrings x out c e :=
  List.mem_append_right _ (List.mem_append_right _ (List.mem_flatMap.mpr ⟨r, hr, mem_treeStrings h⟩))

/-! ## Each accepted record is the device's -/

/-- Where the run's strings have no SHA-512 collision and every certificate of the commitment that verifies is of a
message the device signed, each record the check finds signed is the head the device made of a record it published of
that link-window, in that direction. -/
theorem device_record {x : DeviceInput} {c : Commitment} {e : Evidence}
    (hcr : CF fun y => y ∈ runStrings x (deviceCode x) c e)
    (hsig : ∀ r ∈ c.records.toList, ∀ m, certMessage x.issuer.session r = some m →
      LmsVerifies (deviceParams x) m r.certificate.signature → Signed x.issuer.session (deviceCode x) m)
    {d : NetworkCheck.Direction} {l : String} {w : Nat} {r : Certified} {dg : ByteArray}
    (hs : Signs (deviceParams x) c d l w r dg) :
    ∃ rec ∈ published x.config x.trace, rec.link = l ∧ rec.window = w ∧ directionOf x.config rec = d ∧
      ∃ pd, parameters (gridOf x.config rec) (syncSetOf x.config rec) = .ok pd ∧
        NetworkCheck.recordHead rec.link (directionOf x.config rec) rec.window rec.grid
          ((Status.reason? rec.status).map reasonValue) (gridOf x.config rec) pd true = .ok r.head := by
  obtain ⟨hrec, hdig, hcdg, hk, hcl, hcd, hcw, m, hm, hσ⟩ := hs
  have hmem : r ∈ recordsOf c d l w := by rw [hrec]; exact List.mem_singleton_self r
  have hr : r ∈ c.records.toList := (List.mem_filter.mp hmem).1
  obtain ⟨cr, hcrm, hcm⟩ := hsig r hr m
    (by rw [certMessage, hk, hcl, hcd, hcw, hcdg, show x.issuer.session = (deviceParams x).session from rfl, hm]; rfl) hσ
  obtain ⟨rec, hrecm, pd, hpd, hhead, ⟨R, hcrR, hcrd⟩, hcrl, hcrdir, hcrw⟩ :=
    issued_spec (deviceCode_certificates hcrm)
  obtain ⟨y, hy, rfl⟩ := certificateMessage_ok hm
  obtain ⟨y', hy', hyy⟩ := certificateMessage_ok (toOption_eq_some.mp hcm)
  change messageInput cr.certificate.keyId x.issuer.session cr.certificate.link cr.certificate.direction
    cr.certificate.window cr.certificate.digest = .ok y' at hy'
  have hY : y ∈ runStrings x (deviceCode x) c e :=
    mem_run_check (Or.inl ⟨r, hr, Or.inr (Or.inr (by rw [hk, hcl, hcd, hcw, hcdg]; exact hy))⟩)
  have hY' : y' ∈ runStrings x (deviceCode x) c e := mem_run_message hcrm hy'
  have := hcr _ _ hY hY' hyy
  subst this
  have hrl : r.head.link = l := (recordsOf_mem hmem).2.1
  have hcrhl : cr.head.link = rec.link := (recordHead_head hhead).1
  obtain ⟨-, -, hl, hd, hw, hg⟩ := messageInput_inj hy hy' (by rw [← hrl]; exact link_short hdig)
    (by rw [hcrl, ← hcrhl]; exact link_short hcrR)
  have hdR : dg = R := certificateDigest_inj hcr (fun z hz => mem_run_check (Or.inl ⟨r, hr, Or.inr (Or.inl hz)⟩))
    (fun z hz => mem_run_salted hcrm hz) hdig hcrR (hg.trans hcrd)
  have hh : r.head = cr.head := head_inj hcr (fun z hz => mem_run_check (Or.inl ⟨r, hr, Or.inl hz⟩))
    (fun z hz => mem_run_head hcrm hz) hdig (by rw [hcrR, hdR])
  refine ⟨rec, hrecm, by rw [hl, hcrl], ?_, by rw [hd, hcrdir], pd, hpd, by rw [hh]; exact hhead⟩
  have : (w : Int) = rec.window := by rw [hw, hcrw]
  exact_mod_cast this.symm

theorem status_complete {s : NetworkCertifier.Status} (h : (Status.reason? s).map reasonValue = none) :
    s = .complete := by
  cases s with
  | complete => rfl
  | notEstablished _ => simp [Status.reason?] at h

/-! ## (5a) -/

theorem acceptedRowsAreProgramRows : AcceptedRowsAreProgramRows := by
  intro device x c e hacc hint
  rw [show device x = deviceCode x from hint x]
  by_cases hc : ∃ y ∈ runStrings x (deviceCode x) c e, ∃ z ∈ runStrings x (deviceCode x) c e, y ≠ z ∧
      Sha512.hash y = Sha512.hash z
  · exact Or.inr (Or.inl hc)
  by_cases hf : ∃ r ∈ c.records.toList, ∃ m, certMessage x.issuer.session r = some m ∧
      LmsVerifies (deviceParams x) m r.certificate.signature ∧ ¬ Signed x.issuer.session (deviceCode x) m
  · exact Or.inr (Or.inr hf)
  refine Or.inl ?_
  have hcr : CF fun y => y ∈ runStrings x (deviceCode x) c e := fun y z hy hz hh =>
    Classical.byContradiction fun hne => hc ⟨y, hy, z, hz, hne, hh⟩
  have hsig : ∀ r ∈ c.records.toList, ∀ m, certMessage x.issuer.session r = some m →
      LmsVerifies (deviceParams x) m r.certificate.signature → Signed x.issuer.session (deviceCode x) m :=
    fun r hr m hm hv => Classical.byContradiction fun hn => hf ⟨r, hr, m, hm, hv, hn⟩
  have dev := fun {d l w r dg} (hs : Signs (deviceParams x) c d l w r dg) => device_record hcr hsig hs
  obtain ⟨hfit, eg, ig, hpd, he, hi⟩ := acceptsLinkWindows _ c e hacc
  have heg : egressDigest (deviceParams x) = eg := by
    simp [egressDigest, hpd.1, Except.toOption]
  have hpe : (deviceParams x).egress = #[x.config.egressLink] := rfl
  have hpi : (deviceParams x).ingress = #[x.config.ingressLink] := rfl
  have hpw : (deviceParams x).windows = x.config.windows := rfl
  have hpg : (deviceParams x).grid = x.config.egress := rfl
  -- every link-window has a settled record, the device's
  have settled : ∀ l ∈ [x.config.egressLink, x.config.ingressLink], ∀ w < x.config.windows,
      ∃ d r dg, Settled (deviceParams x) c d l w r dg ∧ (l = x.config.egressLink → d = .egress) ∧
        (l ≠ x.config.egressLink → d = .ingress) := by
    intro l hl w hw
    by_cases hle : l = x.config.egressLink
    · subst hle
      obtain ⟨link, hm⟩ := zip_left hfit (l := x.config.egressLink) (by rw [hpe]; simp)
      obtain ⟨r, dg, hset, -⟩ := he _ link hm w hw
      exact ⟨.egress, r, dg, hset, fun _ => rfl, fun h => (h rfl).elim⟩
    · have hli : l = x.config.ingressLink := by simpa [hle] using hl
      subst hli
      obtain ⟨r, dg, hset, -⟩ := hi x.config.ingressLink (by rw [hpi]; simp) w hw
      exact ⟨.ingress, r, dg, hset, fun h => (hle h).elim, fun _ => rfl⟩
  rcases Nat.eq_zero_or_pos x.config.windows with hW | hW
  · exact ⟨fun _ _ w hw => by omega, fun _ _ _ hw => by omega, fun _ _ _ _ _ _ _ hw => by omega⟩
  -- the two links are two: the device certifies every record on its egress link as egress
  have hd : Distinct x.config := by
    intro hdl
    obtain ⟨r, dg, hset, -⟩ := hi x.config.ingressLink (by rw [hpi]; simp) 0 hW
    obtain ⟨rec, -, hrl, -, hdir, -⟩ := dev hset.1
    simp only [directionOf, hrl.trans hdl.symm, ↓reduceIte] at hdir
    cases hdir
  -- the device's record of a link-window is the settled one
  have mine : ∀ rec ∈ published x.config x.trace, rec.link ∈ [x.config.egressLink, x.config.ingressLink] →
      rec.window < x.config.windows → ∃ d r dg, Settled (deviceParams x) c d rec.link rec.window r dg ∧
        (rec.link = x.config.egressLink → d = .egress) ∧ ∃ pd, parameters (gridOf x.config rec) (syncSetOf x.config rec) = .ok pd ∧
        NetworkCheck.recordHead rec.link (directionOf x.config rec) rec.window rec.grid
          ((Status.reason? rec.status).map reasonValue) (gridOf x.config rec) pd true = .ok r.head := by
    intro rec hrec hl hw
    obtain ⟨d, r, dg, hset, hde, -⟩ := settled _ hl _ hw
    obtain ⟨rec', hrec', hl', hw', -, pd, hpd', hh⟩ := dev hset.1
    have := unique hd hrec' hrec hl' (hl' ▸ hl) hw'
    subst this
    exact ⟨d, r, dg, hset, hde, pd, hpd', hh⟩
  refine ⟨fun l hl w hw => ?_, fun rec hrec hl hw => ?_, fun link hlink syncs hdecl rec hrec hl hw => ?_⟩
  · obtain ⟨d, r, dg, hset, -⟩ := settled l hl w hw
    obtain ⟨rec, hrec, hrl, hrw, -⟩ := dev hset.1
    exact ⟨rec, hrec, hrl, hrw⟩
  · obtain ⟨d, r, dg, hset, -, pd, -, hh⟩ := mine rec hrec hl hw
    exact status_complete ((recordHead_head hh).2.2.2.1.symm.trans hset.2.1)
  · obtain ⟨d, r, dg, hset, hde, pd, -, hh⟩ := mine rec hrec (by rw [hl]; simp) hw
    have hde := hde hl
    subst hde
    have hm := zip_index hfit ⟨0, by rw [hpe]; exact Nat.one_pos⟩ hlink
    rw [show (deviceParams x).egress[(⟨0, _⟩ : Fin _)] = x.config.egressLink from rfl, ← hl] at hm
    obtain ⟨r', dg', hset', -, hprog⟩ := he _ link hm rec.window hw
    have hrr : r' = r := by
      have := hset'.1.1.symm.trans hset.1.1
      simpa using this
    subst hrr
    obtain ⟨h, hph, hhdg⟩ := hprog syncs hdecl
    have hr'm : r' ∈ c.records.toList := (List.mem_filter.mp (show r' ∈ recordsOf c .egress rec.link rec.window by
      rw [hset.1.1]; exact List.mem_singleton_self r')).1
    have hcheck : ∀ z, CheckString (deviceParams x) c e (egressDigest (deviceParams x)) z →
        z ∈ runStrings x (deviceCode x) c e := fun z hz => mem_run_check hz
    rw [heg] at hcheck
    have hhr : h = r'.head := head_inj hcr
      (fun z hz => hcheck z (Or.inr ⟨_, link, hm, _, hw, syncs, hdecl, Or.inr ⟨h, hph, hz⟩⟩))
      (fun z hz => hcheck z (Or.inl ⟨r', hr'm, Or.inl hz⟩)) hhdg hset'.1.2.1
    rw [← hhr] at hh
    have hg : gridOf x.config rec = (deviceParams x).grid := by simp only [gridOf, hl, ↓reduceIte]; rfl
    rw [hg] at hh
    exact (recordHead_grid_inj hcr hph hh
      (fun z hz => hcheck z (Or.inr ⟨_, link, hm, _, hw, syncs, hdecl, Or.inl hz⟩))
      (fun z hz => mem_run_tree (x := x) hrec (by rw [hg]; exact hz))).1.symm

end Properties.NetworkCertifier
