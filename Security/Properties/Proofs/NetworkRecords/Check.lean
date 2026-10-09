import Properties.Proofs.NetworkRecords.Commit
import Properties.Proofs.NetworkCertifier

/-!
# What the records check decides, link-window by link-window

A link-window's records are `recordsAt`'s (`found_toList`). An egress link-window with no violation has one complete
record, an admitted sync, and a record that is the Program's (`egress_none`); an ingress one has one complete record,
and a certified one's head is of the ingress grid (`ingress_none`). Conversely each such link-window has no violation
(`egress_none_of`, `ingress_none_of`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check (Params Certified Link Evidence Grid Profile LinkWindow screen verifies expectedGrid noSync
  unitSync inSyncSet)
open NetworkCertifier (ClockSync)
open Properties.NetworkCertifier (EvidenceFits ParamsDigests Declares SyncAdmitted programGrid recordsOf Signs Settled
  IngressAccepted HeadString headStrings mem_headStrings head_inj)
open Properties.NetworkCertifier.CheckProofs (ba_beq bind_ok syncTest_iff not_not_true not_true declares_declared
  expectedGrid_eq verifies_iff screen_ok declares_unique)
open Properties.NetworkCertifier.Bytes (error_ne_ok)

set_option maxRecDepth 20000

namespace Properties.NetworkRecords.CheckProofs

open Commit Program

instance lawfulBEq : LawfulBEq ByteArray where
  eq_of_beq h := ba_beq.mp h
  rfl := ba_beq.mpr rfl

theorem found_toList (c : Records.Commitment) (d : Direction) (l : String) (w : Nat) :
    (Records.found c d l w).toList = recordsAt c d l w := by
  unfold Records.found recordsAt
  rw [Array.toList_filter]
  apply List.filter_congr
  intro r _
  rw [Bool.eq_iff_iff]
  cases r <;> simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, and_assoc]

theorem certified_names {c : Records.Commitment} {d : Direction} {l : String} {w : Nat} {r : Certified}
    (h : recordsAt c d l w = [.certified r]) : r.head.direction = d ∧ r.head.link = l ∧ r.head.window = w := by
  have hm : Records.Published.certified r ∈ recordsAt c d l w := h ▸ List.mem_singleton_self _
  unfold recordsAt at hm
  simpa using (List.mem_filter.mp hm).2

theorem single_records {d : Direction} {l : String} {w : Nat} {r : Certified}
    (h : r.head.direction = d ∧ r.head.link = l ∧ r.head.window = w) : recordsOf (Records.single r) d l w = [r] := by
  unfold recordsOf Records.single
  simp [h.1, h.2.1, h.2.2]

theorem screen_single {p : Params} {r r' : Certified} (h : screen p #[r] = .ok r') :
    r' = r ∧ verifies p r = true ∧ r.head.deadline = true ∧ r.head.status = none := by
  unfold screen at h
  rw [show #[r].toList = [r] from rfl] at h
  dsimp only at h
  split at h
  · cases h
  rename_i hv
  split at h
  · cases h
  rename_i hdl
  split at h
  · cases h
  · rename_i hst
    exact ⟨(Except.ok.inj h).symm, not_not_true hv, not_not_true hdl, hst⟩

theorem screen_single_of {p : Params} {r : Certified} (hv : verifies p r = true) (hdl : r.head.deadline = true)
    (hst : r.head.status = none) : screen p #[r] = .ok r := by
  unfold screen
  rw [show #[r].toList = [r] from rfl]
  dsimp only
  rw [ite_eq_right (by simp [hv]), ite_eq_right (by simp [hdl]), hst]

theorem verifies_cert {p : Params} {d : Direction} {l : String} {w : Nat} {r : Certified}
    (hn : r.head.direction = d ∧ r.head.link = l ∧ r.head.window = w) :
    verifies p r = true ↔ ∃ dg, CertificateVerifies p d l w r dg := by
  rw [verifies_iff (single_records hn)]
  exact ⟨fun ⟨dg, hs⟩ => ⟨dg, hs.2⟩, fun ⟨dg, hs⟩ => ⟨dg, single_records hn, hs⟩⟩

theorem statusViolation_none {s : Option String} : Records.statusViolation s = none ↔ s = none := by
  unfold Records.statusViolation
  exact Option.map_eq_none_iff

/-- The certified binding: a head with the digest of the Program's grid's head is that head, or two distinct strings
the two heads hash have one SHA-512 digest. -/
theorem head_or {h H : RecordHead} {dg : ByteArray} (e : h.digest = .ok dg) (e' : H.digest = .ok dg) :
    h = H ∨ ∃ x ∈ headStrings h ++ headStrings H, ∃ y ∈ headStrings h ++ headStrings H,
      x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y := by
  by_cases hc : ∃ x ∈ headStrings h ++ headStrings H, ∃ y ∈ headStrings h ++ headStrings H,
      x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y
  · exact Or.inr hc
  · refine Or.inl (head_inj (X := (· ∈ headStrings h ++ headStrings H)) (fun x y hx hy he => ?_)
      (fun x hx => List.mem_append_left _ (mem_headStrings hx))
      (fun x hx => List.mem_append_right _ (mem_headStrings hx)) e e')
    by_contra hne
    exact hc ⟨x, hx, y, hy, hne, he⟩

/-- The Program's grid of an egress link-window, as `verdicts` reads it, under the syncs the link declares. -/
theorem expected_eq {p : Params} {link : Link} {syncs : Fin p.windows → ClockSync} (h : Declares link syncs) {w : Nat}
    (hw : w < p.windows) :
    (Records.expectedGrids p.grid p.windows link).getD w #[] = programGrid p syncs link.frames w := by
  rw [Array.getD_eq_getD_getElem?, RecordsGrid.expectedGrids_eq _ _ _ _ hw, Option.getD_some, expectedGrid_eq h]

/-- **An egress link-window with no violation has its one complete record, the Program's.** -/
theorem egress_none {p : Params} {c : Records.Commitment} {e : Evidence} {eg : ByteArray} {l : String} {link : Link}
    {w : Nat} (hfit : EvidenceFits p e) (hl : link ∈ e.links.toList) (hw : w < p.windows)
    (hv : Records.egressVerdict p c e eg l link ((Records.expectedGrids p.grid p.windows link).getD w #[]) w = none) :
    EgressRecord p c e eg l link w := by
  have hd := declares_declared (hfit.2.1 link hl)
  have hsz := programGrid_sizes (p := p) (syncs := Properties.NetworkCertifier.declared p.windows link) (w := w)
    (hfit.2.2 link hl)
  rw [expected_eq hd hw] at hv
  unfold Records.egressVerdict at hv
  rw [found_toList] at hv
  generalize hR : recordsAt c .egress l w = R at hv
  match R, hR, hv with
  | [], _, hv => simp at hv
  | [.certified r], hR, hv =>
    dsimp only at hv
    cases hs : screen p #[r] with
    | error _ => rw [hs] at hv; cases hv
    | ok r' =>
      obtain ⟨rfl, hver, hdl, hst⟩ := screen_single hs
      rw [hs] at hv
      dsimp only [Except.map] at hv
      split at hv
      · cases hv
      rename_i hsync
      have hadm := (syncTest_iff hfit hl hw).mp (by rw [← Bool.and_assoc]; exact not_not_true hsync)
      cases hrd : (recordHead l .egress w (programGrid p (Properties.NetworkCertifier.declared p.windows link)
          link.frames w) none p.grid eg true >>= RecordHead.digest) with
      | error _ => simp [hrd] at hv
      | ok d =>
        simp only [hrd] at hv
        split at hv
        · rename_i heq
          obtain ⟨dg, hcv⟩ := (verifies_cert (certified_names hR)).mp hver
          obtain ⟨H, hH, hHd⟩ := bind_ok hrd
          have hd' : d = dg := (ba_beq.mp heq).symm.trans hcv.2.1
          subst hd'
          refine ⟨.certified r', hR, ⟨hst, hdl⟩, hadm, fun syncs hs => ?_⟩
          obtain rfl := declares_unique hd hs
          exact ⟨_, H, hcv, hH, hHd, head_or hcv.1 hHd⟩
        · cases hv
  | [.raw l' w' st grid], hR, hv =>
    dsimp only at hv
    cases hst : Records.statusViolation st with
    | some _ => rw [hst] at hv; cases hv
    | none =>
      rw [hst] at hv
      dsimp only at hv
      split at hv
      · cases hv
      rename_i hsync
      have hadm := (syncTest_iff hfit hl hw).mp (by rw [← Bool.and_assoc]; exact not_not_true hsync)
      split at hv
      · rename_i heq
        refine ⟨.raw l' w' st grid, hR, statusViolation_none.mp hst, hadm, fun syncs hs => ?_⟩
        obtain rfl := declares_unique hd hs
        exact (beq_iff_eq.mp heq).symm
      · cases hv
  | [.committed l' w' st dg], hR, hv =>
    dsimp only at hv
    cases hst : Records.statusViolation st with
    | some _ => rw [hst] at hv; cases hv
    | none =>
      rw [hst] at hv
      dsimp only at hv
      split at hv
      · cases hv
      rename_i hsync
      have hadm := (syncTest_iff hfit hl hw).mp (by rw [← Bool.and_assoc]; exact not_not_true hsync)
      split at hv
      · rename_i heq
        refine ⟨.committed l' w' st dg, hR, statusViolation_none.mp hst, hadm, fun syncs hs => ?_⟩
        obtain rfl := declares_unique hd hs
        exact commitsTo (Properties.NetworkCertifier.toOption_eq_some.mp (beq_iff_eq.mp heq)) hsz
      · cases hv
  | _ :: _ :: _, _, hv => simp at hv

/-- **An ingress link-window with no violation has its one complete record**, and a certified one's head is of the
ingress grid. -/
theorem ingress_none {p : Params} {c : Records.Commitment} {ig : ByteArray} {l : String} {w : Nat}
    (hv : Records.ingressVerdict p c ig l w = none) : IngressRecord p c ig l w := by
  unfold Records.ingressVerdict at hv
  rw [found_toList] at hv
  generalize hR : recordsAt c .ingress l w = R at hv
  match R, hR, hv with
  | [], _, hv => simp at hv
  | [.certified r], hR, hv =>
    obtain ⟨r', dg, hset, h1, h2, h3, h4, h5⟩ := Properties.NetworkCertifier.CheckProofs.ingress_none.mp hv
    have hr : r' = r := by
      have := hset.1.1.symm.trans (single_records (certified_names hR))
      exact List.singleton_inj.mp this
    subst hr
    exact ⟨.certified r', hR, ⟨hset.2.1, hset.2.2⟩, dg, hset.1.2, h1, h2, h3, h4, h5⟩
  | [.raw l' w' st grid], hR, hv => exact ⟨_, hR, statusViolation_none.mp hv, trivial⟩
  | [.committed l' w' st dg], hR, hv => exact ⟨_, hR, statusViolation_none.mp hv, trivial⟩
  | _ :: _ :: _, _, hv => simp at hv

end Properties.NetworkRecords.CheckProofs
