import Proofs.Properties.CertifierNetwork.Records
import Proofs.Properties.CertifierNetwork.Frames
import Proofs.Properties.CertifierNetwork.Certify
import Proofs.Properties.CertifierDevice.Complete
import Proofs.Properties.CertifierDevice.Lms
import Proofs.Properties.NetworkCertifier.Outcomes

set_option maxRecDepth 20000

/-!
# The device's output is accepted

Every egress record a closed run publishes is an egress tick's (`egress_at`), so `completeRows_honest` makes its
grid the Program's grid of the frames the shaper handed under the syncs it declared (`grid_eq`). The assembly certifies
each record (`certifyAll_ok`), and since a closed run publishes one record of each link-window, the commitment has one
certificate of each, the honest one (`recordsOf_one`). `HonestAccepted` then accepts the unit.
-/

namespace Properties.CertifierNetwork.Proofs

open _root_.CertifierDevice
open _root_.NetworkCertifier (LogicalFrame ClockSync)
open Properties.CertifierDevice (outputs after EgressAt ValidKey egressActions)
open Properties.CertifierDevice.Proofs (records_proof completeRows_honest pre prefix_good egressAt_of_lt egress_acts
  statusOf rowOf)

theorem mem_published {c : Config} {tr : List Event} {r : BlockGrid} (h : r ∈ published c tr) :
    ∃ (n : Nat) (acts : List Action) (p : Params), (outputs (init c) tr)[n]? = some acts ∧ Action.publish r p ∈ acts := by
  obtain ⟨acts, hacts, hr⟩ := List.mem_flatMap.1 h
  obtain ⟨n, hn⟩ := List.mem_iff_getElem?.1 hacts
  obtain ⟨a, ha, he⟩ := List.mem_filterMap.1 hr
  cases a with
  | publish r' p => cases he; exact ⟨n, acts, p, hn, ha⟩
  | _ => cases he

/-- A record on the egress link is published by an egress tick: any other event publishes only ingress records. -/
theorem egress_at {c : Config} {tr : List Event} (hd : Distinct c) {n : Nat} {acts : List Action} {r : BlockGrid}
    {p : Params} (hn : (outputs (init c) tr)[n]? = some acts) (hm : Action.publish r p ∈ acts)
    (hl : r.link = c.egressLink) : ∃ k b, EgressAt c tr n k b := by
  by_contra hne
  obtain ⟨-, -, -, -, h3⟩ := records_proof c tr
  exact hd (hl.symm.trans ((h3 n acts hn hne _ hm).2 r p rfl).1)

/-- The declared syncs as a function of the `W` windows. -/
def syncsFn (S : Array ClockSync) (W : Nat) (h : S.size = W) : Fin W → ClockSync := fun v => S[v.1]'(h ▸ v.2)

theorem expectedRow_cast {a T W : Nat} (S : Array ClockSync) (h : S.size = W) (fr : List LogicalFrame) :
    _root_.NetworkCertifier.expectedRow a T S.size (fun w => S[w]) fr =
      _root_.NetworkCertifier.expectedRow a T W (syncsFn S W h) fr := by
  subst h; rfl

theorem window_lt {P : List BlockGrid} {l : String} {W : Nat} (h : windowsOn l P = List.range W) {r : BlockGrid}
    (hr : r ∈ P) (hl : r.link = l) : r.window < W := by
  have : r.window ∈ windowsOn l P := List.mem_map_of_mem (List.mem_filter.2 ⟨hr, by simp [hl]⟩)
  rw [h] at this
  exact List.mem_range.1 this

/-- **An egress record's grid is the Program's**: row by row, slot by slot, the bytes of `expectedRow` of the handed
frames under the declared syncs. -/
theorem grid_eq {c : Config} {tr : List Event} (key : Lms.PrivateKey) (session : ByteArray) (hT : Ticking c) (hd : Distinct c)
    (hc : Closed c tr) (ht : HonestTiming c tr) (hnf : NoFault c tr) (hsd : (declaredSyncs c tr).size = c.windows)
    {r : BlockGrid}
    (hr : r ∈ published c tr) (hl : r.link = c.egressLink) (hw : r.window < c.windows) :
    r.grid = Properties.NetworkCertifier.programGrid (unitParams c key session) (syncsFn (declaredSyncs c tr) c.windows hsd)
      (openedFrames c tr) r.window := by
  obtain ⟨n, acts, p, hn, hm⟩ := mem_published hr
  obtain ⟨k, b, hE⟩ := egress_at hd hn hm hl
  have hcr := completeRows_honest c tr ht n k b acts r p hE hn hm (hnf r hr)
    (by show r.window < (declaredSyncs c tr).size; rw [hsd]; exact hw)
  have hcr' : ∀ j < c.egress.buckets, ∀ (s : Nat) (hs : s < c.egress.allocation),
      _root_.NetworkCertifier.expectedRow c.egress.allocation c.egress.buckets c.windows
        (syncsFn (declaredSyncs c tr) c.windows hsd) (openedFrames c tr) (r.window * c.egress.buckets + j) ⟨s, hs⟩ =
        (r.grid[j]!)[s]!.map bitsOf := by
    intro j hj s hs
    rw [← expectedRow_cast]
    exact hcr j hj s hs
  have hsize : r.grid.size = c.egress.buckets ∧ ∀ row ∈ r.grid, Properties.CertifierDevice.Packed c.egress row := by
    rcases (closed_records hT hd hc).2.2 r hr with ⟨-, hs, hp⟩ | ⟨hl', -, -⟩
    · exact ⟨hs, hp⟩
    · exact absurd (hl.symm.trans hl') hd
  obtain ⟨hsz, hpk⟩ := hsize
  apply Array.ext
  · simp [Properties.NetworkCertifier.programGrid, Properties.NetworkCertifier.bytesGrid, unitParams, hsz]
  · intro j h1 h2
    have hj : j < c.egress.buckets := hsz ▸ h1
    obtain ⟨fs, hfs, -, hrow⟩ := hpk _ (Array.getElem_mem h1)
    simp only [Properties.NetworkCertifier.programGrid, Properties.NetworkCertifier.bytesGrid, unitParams,
      Array.getElem_map, List.getElem_toArray, List.getElem_range]
    apply Array.ext
    · rw [hrow]; simp; omega
    · intro s h3 h4
      have hs : s < c.egress.allocation := by rw [hrow] at h3; simp at h3; omega
      have e := hcr' j hj s hs
      rw [getElem!_pos r.grid j h1, getElem!_pos r.grid[j] s h3] at e
      simp only [Array.getElem_map, List.getElem_toArray, List.getElem_ofFn, e, Option.map_map]
      simp [Function.comp_def, bytesOfBits_bitsOf]

/-- **One certificate of each link-window**: if the records `rs` hold one record of each window `w < W` on link `l`,
each certified in direction `d`, the commitment of what the assembly issued for them has one record of `(d, l, w)`,
the certificate of that record. -/
theorem recordsOf_one {c : Config} {i : Issuer} {rs : List BlockGrid} {crs : List CertifiedRecord}
    (hall : List.Forall₂ (Issued c i) rs crs) {l : String} {d : NetworkCheck.Direction} {W w : Nat}
    (hW : windowsOn l rs = List.range W) (hw : w < W) (hld : ∀ r : BlockGrid, r.link = l → directionOf c r = d)
    (bytes : ByteArray) :
    ∃ r cr, r ∈ rs ∧ r.link = l ∧ r.window = w ∧ Issued c i r cr ∧
      Properties.NetworkCertifier.recordsOf (commitment crs bytes) d l w = [certified cr] := by
  obtain ⟨r, hr⟩ := filter_one hW hw
  have hmem : r ∈ rs.filter (fun r => decide (r.link = l ∧ r.window = w)) := by rw [hr]; exact List.mem_singleton_self r
  obtain ⟨hrs, hrd⟩ := List.mem_filter.1 hmem
  simp only [decide_eq_true_eq] at hrd
  have hf := forall₂_filter (p := fun r : BlockGrid => decide (r.link = l ∧ r.window = w))
    (q := fun cr : CertifiedRecord => decide (cr.head.direction = d ∧ cr.head.link = l ∧ cr.head.window = (w : Int)))
    (fun r cr hi => by
      obtain ⟨pd, -, hh⟩ := hi.head
      obtain ⟨hl', hd', hw', -⟩ := recordHead_fields hh
      rw [hl', hd', hw']
      apply decide_eq_decide.2
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨hld r h1, h1, congrArg _ h2⟩
      · rintro ⟨-, h1, h2⟩; exact ⟨h1, Int.ofNat_inj.1 h2⟩) hall
  rw [hr] at hf
  obtain ⟨cr, tl, hi, htl, hcrs⟩ := List.forall₂_cons_left_iff.1 hf
  rw [List.forall₂_nil_left_iff.1 htl] at hcrs
  refine ⟨r, cr, hrs, hrd.1, hrd.2, hi, ?_⟩
  unfold Properties.NetworkCertifier.recordsOf commitment
  rw [List.toList_toArray, List.filter_map]
  exact congrArg (List.map certified) hcrs

/-- **Honest timing on a closed run with no fault**: every window closed complete, so no frame was handed late. -/
theorem honestTiming_of {c : Config} {tr : List Event} (hT : Ticking c) (hc : Closed c tr) (hnf : NoFault c tr) :
    HonestTiming c tr := by
  have hlen : (pre c tr tr.length).shaper.t ≥ c.windows * c.egress.buckets := by
    have hg := (prefix_good c tr tr.length).1
    unfold Closed State.done State.ticks at hc
    simp only [pre, List.take_length] at hg ⊢
    rw [hg.config] at hc
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
    exact hc.1
  have hT0 := hT.1
  have hall : ∀ m t b', EgressAt c tr m t b' → ∃ n k b, EgressAt c tr n k b ∧ (k + 1) % c.egress.buckets = 0 ∧
      t / c.egress.buckets = k / c.egress.buckets ∧ statusOf c tr (k / c.egress.buckets) = .complete := by
    intro m t b' hEm
    have ht : t < c.windows * c.egress.buckets := hEm.2.2
    have hw : t / c.egress.buckets < c.windows := (Nat.div_lt_iff_lt_mul hT0).2 ht
    have hk1 : t / c.egress.buckets * c.egress.buckets + (c.egress.buckets - 1) + 1 =
        (t / c.egress.buckets + 1) * c.egress.buckets := by rw [Nat.succ_mul]; omega
    have hkw : t / c.egress.buckets * c.egress.buckets + (c.egress.buckets - 1) < c.windows * c.egress.buckets := by
      have := Nat.mul_le_mul_right c.egress.buckets (show t / c.egress.buckets + 1 ≤ c.windows from hw)
      omega
    obtain ⟨n, b, -, hE⟩ := egressAt_of_lt (N := tr.length) le_rfl (lt_of_lt_of_le hkw hlen)
    have hc' : (t / c.egress.buckets * c.egress.buckets + (c.egress.buckets - 1) + 1) % c.egress.buckets = 0 := by
      rw [hk1, Nat.mul_mod_left]
    have hdiv : (t / c.egress.buckets * c.egress.buckets + (c.egress.buckets - 1)) / c.egress.buckets =
        t / c.egress.buckets := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hT0, Nat.div_eq_of_lt (by omega), Nat.zero_add]
    refine ⟨n, _, b, hE, hc', hdiv.symm, ?_⟩
    have hacts := egress_acts hE
    have hmem : Action.publish ⟨c.egressLink, t / c.egress.buckets,
        Array.ofFn fun j : Fin c.egress.buckets => rowOf c tr (t / c.egress.buckets * c.egress.buckets + j),
        statusOf c tr (t / c.egress.buckets)⟩ c.egress ∈
        egressActions c (rowOf c tr) (statusOf c tr) (t / c.egress.buckets * c.egress.buckets + (c.egress.buckets - 1))
          b := by
      unfold egressActions
      simp only [hc', hdiv, ↓reduceIte]
      simp
    rw [hdiv]
    have hr := hnf _ (List.mem_flatMap.2 ⟨_, List.mem_of_getElem? hacts,
      List.mem_filterMap.2 ⟨_, hmem, rfl⟩⟩)
    exact hr
  have h := Properties.CertifierDevice.Proofs.hands_honest hall tr.length le_rfl
  simp only [pre, List.take_length] at h
  exact (List.forall₂_eq_eq_eq ▸ h).symm

/-- **`DeviceOutputAccepted`.** -/
theorem deviceOutputAccepted_proof : DeviceOutputAccepted := by
  intro c tr i next salts hT he hd hdg hk hsf hsalt hc hnf hsd hsa hleft
  have ht := honestTiming_of hT hc hnf
  replace hsd : (declaredSyncs c tr).size = c.windows := hsd
  obtain ⟨heW, hiW, hof⟩ := closed_records hT hd hc
  obtain ⟨⟨eg, heg⟩, ig, hig⟩ := hdg
  obtain ⟨hg, hW64, hel, hil⟩ := he
  have hge := hg c.egress (by simp)
  have hgi := hg c.ingress (by simp)
  have hI : i.key.I.size = 16 := by
    obtain ⟨_, _, _, _, I, seed, -, hkey⟩ := hk
    obtain ⟨-, hkI, -, hIs, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hkey
    rw [hkI]; exact hIs
  have hdi : c.ingressLink ≠ c.egressLink := Ne.symm hd
  have hprep : ∀ n < (published c tr).length, ∀ r, (published c tr)[n]? = some r →
      ∃ pd, parameters (gridOf c r) (syncSetOf c r) = .ok pd ∧ ∃ head dg m,
      NetworkCheck.recordHead r.link (directionOf c r) r.window r.grid none (gridOf c r) pd true = .ok head ∧
      i.prepare (salts n) r (directionOf c r) (gridOf c r) pd true = .ok (head, dg, m) := by
    intro n _ r hrn
    have hr := List.mem_of_getElem? hrn
    rcases hof r hr with ⟨hl, hs, hp⟩ | ⟨hl, hs, hp⟩
    · have hw := window_lt heW hr hl
      have hgr : gridOf c r = c.egress := by simp [gridOf, hl]
      have hss : syncSetOf c r = some c.syncSet := by simp [syncSetOf, hl]
      rw [hgr, hss]
      exact ⟨eg, heg, prepare_of hI hsf (hsalt n) hs hp (parameters_size heg) (hl ▸ hel) (lt_of_lt_of_le hw hW64) hge.1 hge.2.1
        hge.2.2.1 hge.2.2.2 (hnf r hr)⟩
    · have hw := window_lt hiW hr hl
      have hgr : gridOf c r = c.ingress := by simp [gridOf, hl, hdi]
      have hss : syncSetOf c r = none := by simp [syncSetOf, hl, hdi]
      rw [hgr, hss]
      exact ⟨ig, hig, prepare_of hI hsf (hsalt n) hs hp (parameters_size hig) (hl ▸ hil) (lt_of_lt_of_le hw hW64) hgi.1 hgi.2.1
        hgi.2.2.1 hgi.2.2.2 (hnf r hr)⟩
  obtain ⟨crs, hcrs, hall⟩ := certifyAll_ok hk (published c tr) next salts hleft hprep
  refine ⟨i, crs, hcrs, fun bytes => ?_⟩
  have hws : ∀ w < c.windows, w < (declaredSyncs c tr).size := fun w hw => hsd ▸ hw
  apply Properties.NetworkCertifier.honestAccepted (unitParams c i.key i.session) (commitment crs bytes) (evidence c tr)
    eg ig
  · refine ⟨rfl, ?_, ?_⟩
    · intro link hl
      simp only [evidence, List.mem_singleton, List.toList_toArray] at hl
      subst hl
      exact hsd
    · intro link hl f hf
      simp only [evidence, List.mem_singleton, List.toList_toArray] at hl
      subst hl
      exact openedFrames_length f hf
  · refine ⟨?_, hig⟩
    have e : checkSyncSet c.syncSet = NetworkCheck.Check.manifestSet c.syncSet := by
      simp [checkSyncSet, NetworkCheck.Check.manifestSet]
    show NetworkCheck.parametersDigest c.egress (some (NetworkCheck.Check.manifestSet c.syncSet)) = .ok eg
    rw [← e]
    exact heg
  · intro link hl w hw
    simp only [evidence, List.mem_singleton, List.toList_toArray] at hl
    subst hl
    have hw' := hws w hw
    have hs := hsa _ (Array.getElem_mem_toList hw')
    exact ⟨_, Array.getElem?_eq_getElem hw', hs.1, hs.2, _, _, rfl, Array.getElem?_eq_getElem hw', rfl, rfl⟩
  · intro l link hmem w hw
    simp only [unitParams, evidence, List.zip_cons_cons, List.zip_nil_right, List.mem_singleton, Prod.mk.injEq,
      List.toList_toArray] at hmem
    obtain ⟨rfl, rfl⟩ := hmem
    obtain ⟨r, cr, hr, hl, hrw, hi, hrec⟩ :=
      recordsOf_one (d := .egress) hall heW hw (fun r h => by simp [directionOf, h]) bytes
    have hdir : directionOf c r = .egress := by simp [directionOf, hl]
    obtain ⟨dg, hdg, hcd⟩ := hi.digest
    refine ⟨syncsFn (declaredSyncs c tr) c.windows hsd, fun v => Array.getElem?_eq_getElem _, cr.head, cr.salt, dg,
      cr.certificate.signature, ?_, hdg, ?_, ?_⟩
    · obtain ⟨pd, hpd, hh⟩ := hi.head
      have hgr : gridOf c r = c.egress := by simp [gridOf, hl]
      have hss : syncSetOf c r = some c.syncSet := by simp [syncSetOf, hl]
      rw [hgr, hss, heg] at hpd
      cases hpd
      rw [hgr, hdir, grid_eq i.key i.session hT hd hc ht hnf hsd hr hl (hrw ▸ hw), hl, hrw] at hh
      exact hh
    · rw [hrec]
      unfold certified Properties.NetworkCertifier.honestRecord
      rw [hi.keyId, hi.link, hi.dir, hi.window, hcd, hl, hrw, hdir]
      rfl
    · obtain ⟨m, hm, s, hs, hv⟩ := hi.signs
      rw [hdir, hl, hrw, hcd] at hm
      exact ⟨m, hm, s, hs, hv⟩
  · intro l hl w hw
    simp only [unitParams, List.mem_singleton, List.toList_toArray] at hl
    subst hl
    obtain ⟨r, cr, hr, hl, hrw, hi, hrec⟩ :=
      recordsOf_one (d := .ingress) hall hiW hw (fun r h => by simp [directionOf, h, hdi]) bytes
    have hdir : directionOf c r = .ingress := by simp [directionOf, hl, hdi]
    obtain ⟨pd, hpd, hh⟩ := hi.head
    have hgr : gridOf c r = c.ingress := by simp [gridOf, hl, hdi]
    have hss : syncSetOf c r = none := by simp [syncSetOf, hl, hdi]
    rw [hgr, hss, hig] at hpd
    cases hpd
    rw [hgr] at hh
    obtain ⟨-, -, -, hb, ha, hB, hτ, hst, hdl, hpar, -⟩ := recordHead_fields hh
    obtain ⟨dg, hdg, hcd⟩ := hi.digest
    refine ⟨cr.head, cr.salt, dg, cr.certificate.signature, hb, ha, hB, hτ, hpar, hst, hdl, hdg, ?_, ?_⟩
    · rw [hrec]
      unfold certified Properties.NetworkCertifier.honestRecord
      rw [hi.keyId, hi.link, hi.dir, hi.window, hcd, hl, hrw, hdir]
      rfl
    · obtain ⟨m, hm, s, hs, hv⟩ := hi.signs
      rw [hdir, hl, hrw, hcd] at hm
      exact ⟨m, hm, s, hs, hv⟩

end Properties.CertifierNetwork.Proofs
