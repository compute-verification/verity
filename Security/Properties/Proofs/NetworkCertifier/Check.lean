import Properties.Proofs.NetworkCertifier.Grid

/-!
# What the check decides

`check` read link-window by link-window: its shape checks are `EvidenceFits`, its parameters digests
`ParamsDigests`, and with both a unit is accepted when no link-window has a violation, undecided when every violation
is `no-verdict` and there is one, and rejected otherwise (`check_accept`, `check_undecided`). A link-window has no
violation exactly when it is accepted (`egress_none`, `ingress_none`), and `no-verdict` exactly when it is `Pending`
(`egress_noVerdict`, `ingress_noVerdict`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open Flock
open NetworkCertifier (ClockSync LogicalFrame expectedRow)

set_option maxRecDepth 20000

namespace Properties.NetworkCertifier.CheckProofs

open Properties.NetworkCertifier Properties.NetworkCertifier.Bytes

theorem ba_beq {a b : ByteArray} : (a == b) = true ↔ a = b := by
  show (a.data == b.data) = true ↔ a = b
  rw [beq_iff_eq]
  exact ⟨fun h => ByteArray.ext h, fun h => h ▸ rfl⟩

/-! ## A link-window's records -/

theorem found_toList (c : Commitment) (d : Direction) (l : String) (w : Nat) :
    (found c d l w).toList = recordsOf c d l w := by
  unfold found recordsOf
  rw [Array.toList_filter]
  apply List.filter_congr
  intro r _
  rw [Bool.eq_iff_iff]
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, and_assoc]

theorem recordsOf_mem {c : Commitment} {d : Direction} {l : String} {w : Nat} {r : Certified}
    (h : r ∈ recordsOf c d l w) : r.head.direction = d ∧ r.head.link = l ∧ r.head.window = w := by
  unfold recordsOf at h
  simpa using (List.mem_filter.mp h).2

theorem toOption_true {ε : Type} {x : Except ε Bool} : (x.toOption == some true) = true ↔ x = .ok true := by
  cases x <;> simp [Except.toOption]

/-- `verifies` of a link-window's one record is `Signs`. -/
theorem verifies_iff {p : Params} {c : Commitment} {d : Direction} {l : String} {w : Nat} {r : Certified}
    (hr : recordsOf c d l w = [r]) : verifies p r = true ↔ ∃ dg, Signs p c d l w r dg := by
  obtain ⟨hd, hl, hw⟩ := recordsOf_mem (hr ▸ List.mem_singleton_self r)
  unfold verifies Signs LmsVerifies
  simp only [Bool.and_eq_true, beq_iff_eq, ba_beq]
  constructor
  · rintro ⟨⟨⟨⟨⟨hk, hcl⟩, hcd⟩, hcw⟩, hdg⟩, hsig⟩
    cases hdig : r.head.digest with
    | error _ => rw [hdig] at hdg; cases hdg
    | ok dg =>
      rw [hdig] at hdg
      have hcdg : r.certificate.digest = dg := ba_beq.mp hdg
      refine ⟨dg, hr, rfl, hcdg, hk, hcl.trans hl, hcd.trans hd, ?_, ?_⟩
      · have := hcw.trans hw
        exact_mod_cast this
      · have hcw' : r.certificate.window = w := by have := hcw.trans hw; exact_mod_cast this
        cases hm : certificateMessage r.certificate.keyId r.certificate.link r.certificate.direction
            r.certificate.window r.certificate.digest with
        | error _ => rw [hm] at hsig; cases hsig
        | ok m =>
          cases hs : Lms.Signature.decode p.lms r.certificate.signature with
          | error _ => rw [hm, hs] at hsig; cases hsig
          | ok s =>
            rw [hm, hs] at hsig
            refine ⟨m, ?_, s, rfl, toOption_true.mp hsig⟩
            rw [← hk, ← hcl.trans hl, ← hcd.trans hd, ← hcw', ← hcdg]
            exact hm
  · rintro ⟨dg, -, hdig, hcdg, hk, hcl, hcd, hcw, m, hm, s, hs, hv⟩
    refine ⟨⟨⟨⟨⟨hk, hcl.trans hl.symm⟩, hcd.trans hd.symm⟩, ?_⟩, ?_⟩, ?_⟩
    · rw [hw, hcw]
    · rw [hdig]; exact ba_beq.mpr hcdg
    · rw [hk, hcl, hcd, hcw, hcdg, hm, hs]
      exact toOption_true.mpr hv

theorem screen_def (p : Params) (c : Commitment) (d : Direction) (l : String) (w : Nat) :
    screen p (found c d l w) = match recordsOf c d l w with
      | [] => .error "no-verdict"
      | [r] =>
        if !verifies p r then .error "signature"
        else if !r.head.deadline then .error "no-verdict"
        else match r.head.status with
          | some why => .error s!"status:{why}"
          | none => .ok r
      | _ => .error "duplicate" := by
  unfold screen
  rw [found_toList]
  rfl

theorem status_ne (why : String) : s!"status:{why}" ≠ "no-verdict" := by
  intro h
  change "status:" ++ why = "no-verdict" at h
  have := congrArg (fun s : String => s.toList.head?) h
  simp at this

/-- The check screens out a link-window's one record exactly when it is `Settled`. -/
theorem screen_ok {p : Params} {c : Commitment} {d : Direction} {l : String} {w : Nat} {r : Certified} :
    screen p (found c d l w) = .ok r ↔ ∃ dg, Settled p c d l w r dg := by
  rw [screen_def]
  unfold Settled
  constructor
  · intro h
    split at h
    · cases h
    · rename_i r₀ hr
      split at h
      · cases h
      rename_i hv
      split at h
      · cases h
      rename_i hdl
      split at h
      · cases h
      · rename_i hst
        cases h
        obtain ⟨dg, hs⟩ := (verifies_iff hr).mp (by simpa using hv)
        exact ⟨dg, hs, hst, by simpa using hdl⟩
    · cases h
  · rintro ⟨dg, hs, hst, hdl⟩
    rw [hs.1]
    simp only
    rw [ite_eq_right (by simpa using (verifies_iff hs.1).mpr ⟨dg, hs⟩), ite_eq_right (by simp [hdl]), hst]

/-- The check's `no-verdict` for a link-window's records is `Pending`. -/
theorem screen_noVerdict {p : Params} {c : Commitment} {d : Direction} {l : String} {w : Nat} :
    screen p (found c d l w) = .error "no-verdict" ↔ Pending p c d l w := by
  rw [screen_def]
  unfold Pending
  constructor
  · intro h
    split at h
    · exact Or.inl ‹_›
    · rename_i r hr
      split at h
      · exact absurd (Except.error.inj h) (by decide)
      rename_i hv
      split at h
      · rename_i hdl
        obtain ⟨dg, hs⟩ := (verifies_iff hr).mp (by simpa using hv)
        exact Or.inr ⟨r, dg, hs, by simpa using hdl⟩
      split at h
      · exact absurd (Except.error.inj h) (status_ne _)
      · cases h
    · exact absurd (Except.error.inj h) (by decide)
  · rintro (h | ⟨r, dg, hs, hdl⟩)
    · rw [h]
    · rw [hs.1]
      simp only
      rw [ite_eq_right (by simpa using (verifies_iff hs.1).mpr ⟨dg, hs⟩), ite_eq_left (by simp [hdl])]

/-! ## A link-window's verdict -/

theorem declares_declared {W : Nat} {link : Link} (h : link.syncs.size = W) : Declares link (declared W link) := by
  intro v
  simp only [declared, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (h ▸ v.2 : v.1 < link.syncs.size),
    Option.getD_some]

theorem declares_unique {W : Nat} {link : Link} {s s' : Fin W → ClockSync} (h : Declares link s)
    (h' : Declares link s') : s = s' :=
  funext fun v => Option.some.inj ((h v).symm.trans (h' v))

theorem expectedGrid_eq {p : Params} {link : Link} {syncs : Fin p.windows → ClockSync} (h : Declares link syncs)
    (w : Nat) : expectedGrid p.grid p.windows link w = programGrid p syncs link.frames w := by
  have e : (fun v : Fin p.windows => link.syncs.getD v.1 noSync) = syncs :=
    funext fun v => by rw [Array.getD_eq_getD_getElem?, h v]; rfl
  simp only [expectedGrid, programGrid, bytesGrid]
  rw [e]

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error _ => exact (error_ne_ok h).elim
  | ok a => exact ⟨a, rfl, h⟩

theorem first_link {e : Evidence} {link : Link} (hl : link ∈ e.links.toList) :
    ∃ first, e.links[0]? = some first ∧ first ∈ e.links.toList := by
  have h0 : 0 < e.links.size := by
    rw [← Array.length_toList]
    exact List.length_pos_of_mem hl
  exact ⟨e.links[0], Array.getElem?_eq_getElem h0, Array.mem_def.mp (Array.getElem_mem h0)⟩

/-- The check's sync test, `sync in sync_set and sync == unit_sync`, is `SyncAdmitted`. -/
theorem syncTest_iff {p : Params} {e : Evidence} {link : Link} {w : Nat} (hfit : EvidenceFits p e)
    (hl : link ∈ e.links.toList) (hw : w < p.windows) :
    (inSyncSet p.syncSet (link.syncs.getD w noSync) &&
      ((link.syncs.getD w noSync).delta == (unitSync e w).delta &&
        (link.syncs.getD w noSync).rho == (unitSync e w).rho)) = true ↔ SyncAdmitted p e link w := by
  have hs : w < link.syncs.size := (hfit.2.1 link hl) ▸ hw
  obtain ⟨first, hf, hfm⟩ := first_link hl
  have hs₀ : w < first.syncs.size := (hfit.2.1 first hfm) ▸ hw
  have hu : unitSync e w = first.syncs[w] := by
    simp only [unitSync, hf, Option.map_some, Option.getD_some, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem hs₀]
  rw [hu, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hs, Option.getD_some]
  simp only [inSyncSet, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  unfold SyncAdmitted
  constructor
  · rintro ⟨⟨h1, h2⟩, h3, h4⟩
    exact ⟨_, Array.getElem?_eq_getElem hs, h1, h2, first, _, hf, Array.getElem?_eq_getElem hs₀, h3.symm, h4.symm⟩
  · rintro ⟨s, hs', h1, h2, first', s₀, hf', hs₀', h3, h4⟩
    rw [Array.getElem?_eq_getElem hs, Option.some.injEq] at hs'
    rw [hf, Option.some.injEq] at hf'
    subst hs' hf'
    rw [Array.getElem?_eq_getElem hs₀, Option.some.injEq] at hs₀'
    subst hs₀'
    exact ⟨⟨h1, h2⟩, h3.symm, h4.symm⟩

theorem not_not_true {b : Bool} (h : ¬ (!b) = true) : b = true := by simpa using h

theorem not_true {b : Bool} (h : (!b) = true) : b = false := by simpa using h

/-- **An egress link-window has no violation exactly when it is `EgressAccepted`.** -/
theorem egress_none {p : Params} {c : Commitment} {e : Evidence} {eg : ByteArray} {l : String} {link : Link}
    {w : Nat} (hfit : EvidenceFits p e) (hl : link ∈ e.links.toList) (hw : w < p.windows) :
    egressVerdict p c e eg l link w = none ↔ EgressAccepted p c e eg l link w := by
  constructor
  · intro hv
    unfold egressVerdict at hv
    cases hs : screen p (found c .egress l w) with
    | error _ => rw [hs] at hv; cases hv
    | ok r =>
      rw [hs] at hv
      obtain ⟨dg, hset⟩ := screen_ok.mp hs
      dsimp only at hv
      split at hv
      · cases hv
      rename_i hadm
      have hadm' := (syncTest_iff hfit hl hw).mp (not_not_true hadm)
      cases hrd : (recordHead l .egress w (expectedGrid p.grid p.windows link w) none p.grid eg true >>=
          RecordHead.digest) with
      | error _ => rw [hrd] at hv; cases hv
      | ok d =>
        rw [hrd] at hv
        dsimp only at hv
        split at hv
        · rename_i heq
          refine ⟨r, dg, hset, hadm', fun syncs hdecl => ?_⟩
          obtain ⟨h, hh, hd⟩ := bind_ok hrd
          rw [expectedGrid_eq hdecl] at hh
          refine ⟨h, hh, ?_⟩
          rw [hd, ← ba_beq.mp heq, hset.1.2.2.1]
        · cases hv
  · rintro ⟨r, dg, hset, hadm, hrec⟩
    unfold egressVerdict
    rw [screen_ok.mpr ⟨dg, hset⟩]
    dsimp only
    split
    · rename_i hc
      have h3 := (syncTest_iff hfit hl hw).mpr hadm
      have h5 : (inSyncSet p.syncSet (link.syncs.getD w noSync) &&
          ((link.syncs.getD w noSync).delta == (unitSync e w).delta &&
            (link.syncs.getD w noSync).rho == (unitSync e w).rho)) = false := not_true hc
      rw [h5] at h3
      cases h3
    · have hd := declares_declared (hfit.2.1 link hl)
      obtain ⟨h, hh, hdg⟩ := hrec _ hd
      rw [← expectedGrid_eq hd] at hh
      have : (recordHead l .egress w (expectedGrid p.grid p.windows link w) none p.grid eg true >>=
          RecordHead.digest) = .ok dg := by rw [hh]; exact hdg
      rw [this]
      dsimp only
      rw [ite_eq_left (ba_beq.mpr hset.1.2.2.1)]

/-- **An egress link-window's violation is `no-verdict` exactly when it is `Pending`.** -/
theorem egress_noVerdict {p : Params} {c : Commitment} {e : Evidence} {eg : ByteArray} {l : String} {link : Link}
    {w : Nat} : egressVerdict p c e eg l link w = some "no-verdict" ↔ Pending p c .egress l w := by
  constructor
  · intro hv
    unfold egressVerdict at hv
    cases hs : screen p (found c .egress l w) with
    | error v =>
      rw [hs] at hv
      cases Option.some.inj hv
      exact screen_noVerdict.mp hs
    | ok r =>
      rw [hs] at hv
      dsimp only at hv
      split at hv
      · exact absurd (Option.some.inj hv) (by decide)
      split at hv
      · split at hv
        · cases hv
        · exact absurd (Option.some.inj hv) (by decide)
      · exact absurd (Option.some.inj hv) (by decide)
  · intro h
    unfold egressVerdict
    rw [screen_noVerdict.mpr h]

/-- **An ingress link-window has no violation exactly when it is `IngressAccepted`.** -/
theorem ingress_none {p : Params} {c : Commitment} {ig : ByteArray} {l : String} {w : Nat} :
    ingressVerdict p c ig l w = none ↔ IngressAccepted p c ig l w := by
  constructor
  · intro hv
    unfold ingressVerdict at hv
    cases hs : screen p (found c .ingress l w) with
    | error _ => rw [hs] at hv; cases hv
    | ok r =>
      rw [hs] at hv
      obtain ⟨dg, hset⟩ := screen_ok.mp hs
      dsimp only at hv
      split at hv
      · rename_i heq
        simp only [Bool.and_eq_true, beq_iff_eq, ba_beq] at heq
        obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := heq
        exact ⟨r, dg, hset, h1, h2, h3, h4, h5⟩
      · cases hv
  · rintro ⟨r, dg, hset, h1, h2, h3, h4, h5⟩
    unfold ingressVerdict
    rw [screen_ok.mpr ⟨dg, hset⟩]
    dsimp only
    rw [ite_eq_left (by simp only [Bool.and_eq_true, beq_iff_eq, ba_beq]; exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩)]

/-- **An ingress link-window's violation is `no-verdict` exactly when it is `Pending`.** -/
theorem ingress_noVerdict {p : Params} {c : Commitment} {ig : ByteArray} {l : String} {w : Nat} :
    ingressVerdict p c ig l w = some "no-verdict" ↔ Pending p c .ingress l w := by
  constructor
  · intro hv
    unfold ingressVerdict at hv
    cases hs : screen p (found c .ingress l w) with
    | error v =>
      rw [hs] at hv
      cases Option.some.inj hv
      exact screen_noVerdict.mp hs
    | ok r =>
      rw [hs] at hv
      dsimp only at hv
      split at hv
      · cases hv
      · exact absurd (Option.some.inj hv) (by decide)
  · intro h
    unfold ingressVerdict
    rw [screen_noVerdict.mpr h]

/-! ## The verdicts -/

theorem verdicts_all {p : Params} {c : Commitment} {e : Evidence} {eg ig : ByteArray} (P : LinkWindow → Prop) :
    (∀ v ∈ (verdicts p c e eg ig).toList, P v) ↔
      (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows,
        P ⟨.egress, l, w, egressVerdict p c e eg l link w⟩) ∧
      ∀ l ∈ p.ingress.toList, ∀ w < p.windows, P ⟨.ingress, l, w, ingressVerdict p c ig l w⟩ := by
  simp only [verdicts, List.toList_toArray, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range]
  constructor
  · intro h
    exact ⟨fun l link hm w hw => h _ (Or.inl ⟨(l, link), hm, w, hw, rfl⟩),
      fun l hl w hw => h _ (Or.inr ⟨l, hl, w, hw, rfl⟩)⟩
  · rintro ⟨he, hi⟩ v (⟨⟨l, link⟩, hm, w, hw, rfl⟩ | ⟨l, hl, w, hw, rfl⟩)
    · exact he l link hm w hw
    · exact hi l hl w hw

theorem verdicts_exists {p : Params} {c : Commitment} {e : Evidence} {eg ig : ByteArray} (P : LinkWindow → Prop) :
    (∃ v ∈ (verdicts p c e eg ig).toList, P v) ↔
      (∃ l link, (l, link) ∈ p.egress.toList.zip e.links.toList ∧ ∃ w < p.windows,
        P ⟨.egress, l, w, egressVerdict p c e eg l link w⟩) ∨
      ∃ l ∈ p.ingress.toList, ∃ w < p.windows, P ⟨.ingress, l, w, ingressVerdict p c ig l w⟩ := by
  simp only [verdicts, List.toList_toArray, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨v, ⟨⟨l, link⟩, hm, w, hw, rfl⟩ | ⟨l, hl, w, hw, rfl⟩, hp⟩
    · exact Or.inl ⟨l, link, hm, w, hw, hp⟩
    · exact Or.inr ⟨l, hl, w, hw, hp⟩
  · rintro (⟨l, link, hm, w, hw, hp⟩ | ⟨l, hl, w, hw, hp⟩)
    · exact ⟨_, Or.inl ⟨(l, link), hm, w, hw, rfl⟩, hp⟩
    · exact ⟨_, Or.inr ⟨l, hl, w, hw, rfl⟩, hp⟩

/-- Under evidence that fits, every egress link is paired with its opening. -/
theorem zip_left {p : Params} {e : Evidence} (hfit : EvidenceFits p e) {l : String} (hl : l ∈ p.egress.toList) :
    ∃ link, (l, link) ∈ p.egress.toList.zip e.links.toList := by
  obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.mp hl
  have hi' : i < e.links.toList.length := by rw [Array.length_toList, hfit.1]; simpa using hi
  refine ⟨e.links.toList[i], ?_⟩
  have hz : i < (p.egress.toList.zip e.links.toList).length := by rw [List.length_zip]; omega
  have := List.getElem_mem hz
  rwa [List.getElem_zip] at this

/-! ## The check -/

theorem fits_iff {p : Params} {e : Evidence} : EvidenceFits p e ↔
    ¬ e.links.size ≠ p.egress.size ∧ ¬ (e.links.any fun x => decide (x.syncs.size ≠ p.windows)) = true ∧
      ¬ (e.links.any fun x => x.frames.any fun x => decide (x.payload.length ≠ 8 * p.grid.frameBytes)) = true := by
  unfold EvidenceFits
  simp only [← Array.any_toList, List.any_eq_true, decide_eq_true_eq, ne_eq, not_not, not_exists, not_and]

theorem check_unfit {p : Params} {c : Commitment} {e : Evidence} (h : ¬ EvidenceFits p e) :
    ∃ why, check p c e = .reject why := by
  rw [fits_iff] at h
  unfold check
  by_cases h1 : e.links.size ≠ p.egress.size
  · rw [ite_eq_left h1]; exact ⟨_, rfl⟩
  rw [ite_eq_right h1]
  by_cases h2 : (e.links.any fun x => decide (x.syncs.size ≠ p.windows)) = true
  · rw [ite_eq_left h2]; exact ⟨_, rfl⟩
  rw [ite_eq_right h2]
  by_cases h3 : (e.links.any fun x => x.frames.any fun x => decide (x.payload.length ≠ 8 * p.grid.frameBytes)) = true
  · rw [ite_eq_left h3]; exact ⟨_, rfl⟩
  exact absurd ⟨h1, h2, h3⟩ h

theorem check_undigested {p : Params} {c : Commitment} {e : Evidence} (hfit : EvidenceFits p e)
    (h : ∀ eg ig, ¬ ParamsDigests p eg ig) : ∃ why, check p c e = .reject why := by
  obtain ⟨h1, h2, h3⟩ := fits_iff.mp hfit
  unfold check
  rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3]
  cases hd1 : parametersDigest p.grid (some (manifestSet p.syncSet)) with
  | error _ => exact ⟨_, rfl⟩
  | ok eg =>
    cases hd2 : parametersDigest p.ingressGrid none with
    | error _ => exact ⟨_, rfl⟩
    | ok ig => exact absurd ⟨hd1, hd2⟩ (h eg ig)

/-- With evidence that fits and the parameters' digests, the check is the verdicts' cascade. -/
theorem check_digested {p : Params} {c : Commitment} {e : Evidence} {eg ig : ByteArray} (hfit : EvidenceFits p e)
    (hpd : ParamsDigests p eg ig) :
    ((verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) = none →
      (verdicts p c e eg ig).find? (·.violation.isSome) = none → ∃ pub prof, check p c e = .accept pub prof) ∧
    (∀ v, (verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) = some v →
      ∃ why, check p c e = .reject why) ∧
    ((verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) = none →
      ∀ v, (verdicts p c e eg ig).find? (·.violation.isSome) = some v → ∃ why, check p c e = .undecided why) := by
  obtain ⟨h1, h2, h3⟩ := fits_iff.mp hfit
  refine ⟨fun hf1 hf2 => ?_, fun v hf1 => ?_, fun hf1 v hf2 => ?_⟩ <;>
  · unfold check
    rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, hpd.1, hpd.2]
    dsimp only
    first
    | (rw [hf1, hf2]; exact ⟨_, _, rfl⟩)
    | (rw [hf1]; exact ⟨_, rfl⟩)
    | (rw [hf1, hf2]; exact ⟨_, rfl⟩)

theorem any_ne {o : Option String} : ¬ (o.any (· != "no-verdict")) = true ↔ o = none ∨ o = some "no-verdict" := by
  cases o <;> simp

theorem check_accept {p : Params} {c : Commitment} {e : Evidence} {eg ig : ByteArray} (hfit : EvidenceFits p e)
    (hpd : ParamsDigests p eg ig) :
    (∃ pub prof, check p c e = .accept pub prof) ↔ ∀ v ∈ (verdicts p c e eg ig).toList, v.violation = none := by
  obtain ⟨hA, hR, hU⟩ := check_digested (c := c) hfit hpd
  constructor
  · rintro ⟨pub, prof, h⟩
    cases hf1 : (verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) with
    | some v => obtain ⟨_, h'⟩ := hR v hf1; rw [h] at h'; cases h'
    | none =>
      cases hf2 : (verdicts p c e eg ig).find? (·.violation.isSome) with
      | some v => obtain ⟨_, h'⟩ := hU hf1 v hf2; rw [h] at h'; cases h'
      | none =>
        intro v hv
        have := Array.find?_eq_none.mp hf2 v (Array.mem_def.mpr hv)
        simpa using this
  · intro h
    refine hA (Array.find?_eq_none.mpr fun v hv => ?_) (Array.find?_eq_none.mpr fun v hv => ?_)
    · rw [h v (Array.mem_def.mp hv)]; simp
    · rw [h v (Array.mem_def.mp hv)]; simp

theorem check_undecided {p : Params} {c : Commitment} {e : Evidence} {eg ig : ByteArray} (hfit : EvidenceFits p e)
    (hpd : ParamsDigests p eg ig) :
    (∃ why, check p c e = .undecided why) ↔
      (∀ v ∈ (verdicts p c e eg ig).toList, v.violation = none ∨ v.violation = some "no-verdict") ∧
        ∃ v ∈ (verdicts p c e eg ig).toList, v.violation = some "no-verdict" := by
  obtain ⟨hA, hR, hU⟩ := check_digested (c := c) hfit hpd
  constructor
  · rintro ⟨why, h⟩
    cases hf1 : (verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) with
    | some v => obtain ⟨_, h'⟩ := hR v hf1; rw [h] at h'; cases h'
    | none =>
      have hall : ∀ v ∈ (verdicts p c e eg ig).toList, v.violation = none ∨ v.violation = some "no-verdict" :=
        fun v hv => any_ne.mp (Array.find?_eq_none.mp hf1 v (Array.mem_def.mpr hv))
      cases hf2 : (verdicts p c e eg ig).find? (·.violation.isSome) with
      | none => obtain ⟨_, _, h'⟩ := hA hf1 hf2; rw [h] at h'; cases h'
      | some v =>
        refine ⟨hall, v, Array.mem_def.mp (Array.mem_of_find?_eq_some hf2), ?_⟩
        have hs := Array.find?_some hf2
        rcases hall v (Array.mem_def.mp (Array.mem_of_find?_eq_some hf2)) with h0 | h0
        · rw [h0] at hs; cases hs
        · exact h0
  · rintro ⟨hall, v, hv, hnv⟩
    have hf1 : (verdicts p c e eg ig).find? (·.violation.any (· != "no-verdict")) = none :=
      Array.find?_eq_none.mpr fun v hv => any_ne.mpr (hall v (Array.mem_def.mp hv))
    cases hf2 : (verdicts p c e eg ig).find? (·.violation.isSome) with
    | none =>
      have := Array.find?_eq_none.mp hf2 v (Array.mem_def.mpr hv)
      rw [hnv] at this
      exact absurd rfl this
    | some v' => exact hU hf1 v' hf2

end Properties.NetworkCertifier.CheckProofs
