import Properties.NetworkCertifier.Period
import Proofs.Properties.CertifierDevice.Lms

/-!
# Proof that an accepted period accepted every link-window (`Properties.NetworkCertifier.PeriodAcceptMeans`)

`NetworkCheck.Period.check` accepts only when no finding is left (`firstFinding … = none`), so each link-window's
check of the parameters' form returned `ok`: `windowCheck` in the clock form, `rowsWindowCheck` in the rows form. Each
returns `ok` only past every refusal, which is `PeriodWindowAccepted` or `PeriodRowsWindowAccepted`.
-/

open NetworkCheck.Period
open NetworkCheck.Check (Certificate)
open Lean (Json)

namespace Properties.NetworkCertifier

namespace Period

theorem firstFinding_none {fs : List (Option (Core.Verdict × String))} (h : firstFinding fs = none) :
    ∀ f ∈ fs, f = none := by
  unfold firstFinding at h
  split at h
  · cases h
  · exact List.findSome?_eq_none_iff.mp h

theorem findingOf_none {α : Type} {r : Found α} (h : findingOf r = none) : ∃ a, r = .ok a := by
  cases r with
  | error x => simp [findingOf] at h
  | ok u => exact ⟨u, rfl⟩

theorem answered_ok {what : String} {reg : Registration} {a : Core.Answer} {v : Core.Integrity × Lean.Json}
    (h : answered what reg a = .ok v) :
    a.protocol = oneStage ∧ (a.digest == some reg.digest) = true ∧ a.verdict = .accept ∧
      ∃ prof, a.published = some v.2 ∧ a.profile = some prof ∧ Core.Integrity.ofJson prof = .ok v.1 := by
  simp only [answered, refuse, pure, Except.pure] at h
  split at h
  · cases h
  rename_i d1
  split at h
  · cases h
  rename_i d2
  split at h
  · rename_i pub prof hv hp hpr
    split at h
    · rename_i i hi
      cases h
      refine ⟨?_, ?_, hv, prof, hp, hpr, hi⟩
      · simpa [bne_iff_ne] using d1
      · simpa [bne] using d2
    · cases h
  · cases h
  · cases h

theorem windowCheck_ok {p : Params} {ss : NetworkCertifier.SyncSet} {earlier : Array ByteArray} {w : Nat}
    {reg : Registration} {x : Window} (h : windowCheck p (clearClocks ss) earlier w reg x = .ok ()) :
    PeriodWindowAccepted p ss earlier w reg x := by
  simp only [windowCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  case h_2 => cases h
  rename_i clock hx
  split at h
  · rename_i c1
    split at h
    · rename_i c2
      split at h
      · rename_i c3
        split at h
        · cases h
        · rename_i c4
          split at h
          · cases h
          · rename_i c5
            split at h
            · cases h
            · rename_i v ha
              split at h
              · rename_i c6
                obtain ⟨a1, a2, a3, prof, a4, a5, a6⟩ := answered_ok ha
                have r := (Bool.and_eq_true _ _).mp c1
                have q := (Bool.and_eq_true _ _).mp c6
                refine ⟨⟨r.1, by simpa using r.2⟩, ?_, ⟨clock, hx, c3, by simpa using c4, by simpa using c5⟩,
                  a1, a2, a3, v.2, prof, v.1, a4, a5, a6, q.1, q.2⟩
                obtain ⟨hs, hall⟩ := (Bool.and_eq_true _ _).mp c2
                unfold PeriodWindowCertified
                generalize x.certificates = cs at hs hall
                rcases cs with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨d, l⟩⟩⟩⟩
                · simp at hs
                · simp at hs
                · refine ⟨a, b, rfl, ?_⟩
                  simp [Array.zip, Array.all] at hall
                  revert hall
                  cases certifies p .egress w a <;> cases certifies p .ingress w b <;> simp [Id.run, pure]
                · simp at hs
              · cases h
      · cases h
    · cases h
  · cases h

theorem rowsCheck_ok {p : Params} {s : Schedule} {sd : ByteArray} {w : Nat} {d : NetworkCheck.Direction} {r : Rows}
    (h : rowsCheck p s sd w d r = .ok ()) : PeriodRowsChecked p s sd w d r := by
  simp only [rowsCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  · rename_i c1
    split at h
    · rename_i c2
      split at h
      · rename_i c3
        split at h
        · rename_i dom hdom
          split at h
          · rename_i c4
            split at h
            · rename_i head hhead
              split at h
              · rename_i c5
                split at h
                · rename_i c6
                  exact ⟨c1, c2, c3, dom, head, hdom, c4, hhead, c5, c6⟩
                · cases h
              · cases h
            · cases h
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

theorem factsOf_ok {what : String} {prof : Core.Integrity} {pub : Lean.Json} {f : Facts}
    (h : factsOf what prof pub = .ok f) :
    ∃ l b, prof.levels = #[l] ∧ f.units = l.size ∧
      ((∃ k, l.draw = some (.subset k)) ∨ ∃ ss, l.draw = some (.stratified ss)) ∧
      Flock.J.nat pub "bound" "public" = .ok b := by
  by_cases hl : ∃ l, prof.levels = #[l]
  · obtain ⟨l, hl⟩ := hl
    have e : factsOf what prof pub = (do
        let drawn ← match l.draw with
          | some (.subset k) => pure k
          | some (.stratified ss) => pure (ss.foldl (fun n st => n + st.k) 0)
          | _ => refuse s!"{what}: profile: not a subset or stratified draw"
        match Flock.J.nat pub "bound" "public" with
        | .ok b => pure { units := l.size, drawn, delta := prof.delta, wrong := b }
        | .error e => refuse s!"{what}: {e}" : Found Facts) := by
      unfold factsOf
      rw [hl]
      rfl
    rw [e] at h
    simp only [refuse, bind, Except.bind, pure, Except.pure] at h
    split at h
    · rename_i k hk
      split at h
      · rename_i b hb
        cases h
        exact ⟨l, b, hl, rfl, Or.inl ⟨k, hk⟩, hb⟩
      · cases h
    · rename_i ss hss
      split at h
      · rename_i b hb
        cases h
        exact ⟨l, b, hl, rfl, Or.inr ⟨ss, hss⟩, hb⟩
      · cases h
    · cases h
  · exfalso
    have : ∃ e, factsOf what prof pub = .error e := by
      unfold factsOf
      rcases h' : prof.levels with ⟨_ | ⟨l, _ | ⟨l', ls⟩⟩⟩
      · exact ⟨_, rfl⟩
      · exact absurd ⟨l, h'⟩ hl
      · exact ⟨_, rfl⟩
    obtain ⟨e, he⟩ := this
    rw [he] at h
    cases h

theorem rowsWindowCheck_ok {p : Params} {s : Schedule} {sd : ByteArray} {w : Nat} {reg : Registration} {x : Window}
    {f : Facts} (h : rowsWindowCheck p s sd w reg x = .ok f) : PeriodRowsWindowAccepted p s sd w reg x := by
  simp only [rowsWindowCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  case h_2 => cases h
  rename_i rs hx
  split at h
  · rename_i c1
    split at h
    · rename_i c2
      split at h
      · rename_i c3
        rcases rs with ⟨_ | ⟨ri, _ | ⟨re, _ | ⟨d, l⟩⟩⟩⟩
        · simp at c3
        · simp at c3
        · split at h
          · cases h
          · rename_i v hloop
            rw [← Array.forIn_toList] at hloop
            simp only [Array.toList_zip, List.zip_cons_cons, List.zip_nil_left, List.forIn_cons,
              List.forIn_nil] at hloop
            split at hloop
            · cases hloop
            rename_i u hi
            split at hloop
            · cases hloop
            rename_i u' he
            cases u
            cases u'
            split at h
            · rename_i c4
              split at h
              · cases h
              · rename_i v ha
                obtain ⟨a1, a2, a3, prof, a4, a5, a6⟩ := answered_ok ha
                split at h
                · cases h
                rename_i f' hf
                obtain ⟨l, b, hl, hu, hd, hb⟩ := factsOf_ok hf
                have hun : l.size = 2 * s.window := by
                  rw [← hu]
                  cases hc : (f'.units == 2 * s.window) with
                  | true => exact beq_iff_eq.mp hc
                  | false => simp [hc] at h
                have hh : holds reg ri = true ∧ holds reg re = true :=
                  ⟨Array.all_eq_true.mp c4 0 (by simp), Array.all_eq_true.mp c4 1 (by simp)⟩
                refine ⟨c1, ?_, ⟨ri, re, hx, rowsCheck_ok hi, rowsCheck_ok he, hh.1, hh.2⟩,
                  a1, a2, a3, v.2, prof, v.1, l, b, a4, a5, a6, hl, hun, hd, hb⟩
                obtain ⟨hs, hall⟩ := (Bool.and_eq_true _ _).mp c2
                unfold PeriodRowsCertified
                generalize x.certificates = cs at hs hall
                rcases cs with ⟨_ | ⟨q1, _ | ⟨q2, _ | ⟨q3, _ | ⟨q4, qs⟩⟩⟩⟩⟩
                · simp at hs
                · simp at hs
                · simp at hs
                · refine ⟨q1, q2, q3, rfl, ?_⟩
                  simp [records, Array.zip, Array.all] at hall
                  revert hall
                  cases certifiesOn p p.link .ingress w q1 <;> cases certifiesOn p p.link .egress w q2 <;>
                    cases certifiesOn p (p.link ++ "-fails") .egress w q3 <;> simp [Id.run, pure]
                · simp at hs
            · cases h
        · simp at c3
      · cases h
    · cases h
  · cases h

theorem map_ok {α β : Type} {g : α → β} {r : Found α} {b : β} (h : g <$> r = .ok b) : ∃ a, r = .ok a := by
  cases r with
  | error x => cases h
  | ok a => exact ⟨a, rfl⟩

end Period

theorem periodAcceptMeans : PeriodAcceptMeans := by
  intro p c e pub prof h
  change check p c e = _ at h
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i h1 h2 h3 h4
  dsimp only at h
  split at h
  · cases h
  · cases h
  rename_i hf
  have hall := Period.firstFinding_none hf
  refine ⟨by omega, by omega, fun i reg x hc he => ?_⟩
  have hmem : ((reg, x), i) ∈ (c.windows.zip e.windows).zipIdx := by
    rw [Array.mem_zipIdx_iff_getElem?, Array.getElem?_zip_eq_some]
    exact ⟨hc, he⟩
  rcases hform : p.form with ss | ⟨s, sd, scan⟩
  · simp only [hform] at hall
    have h0 := hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨_, Array.mem_toList_iff.2
      (Array.mem_map.2 ⟨_, hmem, rfl⟩), rfl⟩)))
    obtain ⟨_, ha⟩ := Period.findingOf_none h0
    obtain ⟨u, hu⟩ := Period.map_ok ha
    exact Period.windowCheck_ok hu
  · simp only [hform] at hall
    have h0 := hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨_, Array.mem_toList_iff.2
      (Array.mem_map.2 ⟨_, hmem, rfl⟩), rfl⟩)))
    obtain ⟨_, ha⟩ := Period.findingOf_none h0
    obtain ⟨f, hu⟩ := Period.map_ok ha
    exact Period.rowsWindowCheck_ok hu

theorem periodAcceptMeans_iff : PeriodAcceptMeans ↔ PeriodAcceptMeansOf protocol.check := Iff.rfl

/-! ## The honest and nontrivial lemmas

The honest period's acceptance goes through lemmas over arbitrary parameters, windows and certificates, each instantiated
by rewriting: the kernel never compares a term holding `honestKey` with anything but itself, since reducing one would
build the key's 2^20-leaf tree. -/

namespace Period

local instance instLawfulBEqByteArrayPeriod : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem except_bind_ok {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a : Except ε α).bind f = f a := rfl

theorem certifies_of {p : Params} {d : NetworkCheck.Direction} {w : Nat} {c : Certificate} {m : ByteArray}
    {sig : Lms.Signature} (hd : c.direction = d) (hl : c.link = p.link) (hw : c.window = w)
    (hk : c.keyId = p.key.extract 0 16)
    (hm : NetworkCheck.certificateMessage c.keyId p.session c.link c.direction c.window c.digest = .ok m)
    (hs : Lms.Signature.decode p.lms c.signature = .ok sig) (hv : Lms.verify p.lms p.key m sig = .ok true) :
    certifies p d w c = true := by
  unfold certifies certifiesOn signs
  rw [hm, hs]
  simp only [hv, hd, hl, hw, hk, beq_self_eq_true, Bool.and_true, Except.toOption]

theorem answered_of {what : String} {reg : Registration} {a : Core.Answer} {pub prof : Json} {i : Core.Integrity}
    (h1 : a.protocol = oneStage) (h2 : a.digest = some reg.digest) (h3 : a.verdict = .accept)
    (h4 : a.published = some pub) (h5 : a.profile = some prof) (h6 : Core.Integrity.ofJson prof = .ok i) :
    answered what reg a = .ok (i, pub) := by
  simp only [answered, h1, h2, h3, h4, h5, h6, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte, pure,
    Except.pure]

theorem windowCheck_of {p : Params} {clear : List ByteArray} {earlier : Array ByteArray} {w : Nat}
    {reg : Registration} {x : Window} {eg ig clock : Certificate} {prof : Core.Integrity} {pub : Json}
    (hp : (reg.record.getObjValD "program" == .str p.program) = true)
    (hw : canonText (reg.record.getObjValD "window") = canonText (served w))
    (hc : x.certificates = #[eg, ig]) (heg : certifies p .egress w eg = true)
    (hig : certifies p .ingress w ig = true) (hx : x.shown = .clock clock)
    (hcl : certifies p .egress w clock = true) (hclear : clear.contains clock.digest = false)
    (hearlier : earlier.contains clock.digest = false)
    (ha : ∀ what, answered what reg x.answer = .ok (prof, pub))
    (hprof : (prof.levels.map (fun l => (l.name, l.size, l.draw)) ==
        #[(unitLevel, 1, some (Core.Draw.subset 1))] && prof.drawn == #[(unitLevel, 1)]) = true) :
    windowCheck p clear earlier w reg x = .ok () := by
  have hclear' : clock.digest ∉ clear := by simpa using hclear
  have hearlier' : clock.digest ∉ earlier := by simpa using hearlier
  have hprof' := hprof
  simp only [Bool.and_eq_true, beq_iff_eq] at hprof'
  simp only [windowCheck, hx, hp, hw, hc, beq_self_eq_true, Bool.and_true, Bool.true_and, Array.size,
    List.length_cons, List.length_nil]
  simp [Array.zip, Array.all, heg, hig, hcl, hclear', hearlier', ha, hprof', bind, Except.bind, pure, Except.pure,
    Id.run]

theorem check_one {p : Params} {ss : NetworkCertifier.SyncSet} {reg : Registration} {x : Window} {r : ByteArray}
    (hf : p.form = .clock ss) (hw : p.windows = 1) (ha : p.audits = #[])
    (h : windowCheck p (clearClocks ss) #[] p.first reg x = .ok ()) :
    check p ⟨#[], #[reg], r⟩ ⟨#[], #[x]⟩ =
      .accept (terms #[] #[] p.cells p.jitter (ss.deltas.length * ss.rhos.length) 1)
        { link := p.link, windows := 1 } := by
  unfold check
  simp [hf, hw, ha, h, Functor.map, Except.map, findingOf, firstFinding, Except.toOption]


theorem honestKey_eq {k : Lms.PrivateKey} (hk : Lms.key Lms.deployed honestI ⟨Array.replicate 64 2⟩ = .ok k) :
    honestKey = k := by
  show (Lms.key Lms.deployed honestI ⟨Array.replicate 64 2⟩).toOption.getD ⟨Lms.deployed, honestI, .empty, #[]⟩ = k
  rw [hk]
  try rfl

theorem honestKey_ok : Lms.key Lms.deployed honestI ⟨Array.replicate 64 2⟩ = .ok honestKey := by
  obtain ⟨k, hk⟩ := Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed (I := honestI) (by decide)
    ⟨Array.replicate 64 2⟩
  rw [honestKey_eq hk]
  exact hk

theorem honest_make : Lms.Params.make 64 1 20 false = .ok Lms.deployed := by
  simp [Lms.Params.make, Lms.deployed, pure, Except.pure]

theorem honestKey_I : honestKey.I = honestI :=
  (Properties.CertifierDevice.Proofs.Lms.key_spec honestKey_ok).2.1

theorem honestParams_key : honestParams.key = honestKey.I ++ honestKey.tree[1]! := rfl

theorem honest_keyId : honestParams.key.extract 0 16 = honestI := by
  rw [honestParams_key, honestKey_I]
  exact ByteArray.extract_append_eq_left (by decide)

theorem honest_message (d : NetworkCheck.Direction) (x : ByteArray) (hx : x.size = 64) :
    ∃ m, NetworkCheck.certificateMessage honestI honestSession "link" d ((0 : Nat) : Int) x = .ok m := by
  obtain ⟨b, hb⟩ : ∃ b, NetworkCheck.uBE 8 ((0 : Nat) : Int) "window" = .ok b := ⟨_, rfl⟩
  have hs : honestSession.size = 32 := by decide
  have hi : honestI.size = 16 := by decide
  simp only [NetworkCheck.certificateMessage, NetworkCheck.messageInput, hx, hb, hs, hi, NetworkCheck.digestBytes,
    NetworkCheck.sessionBytes, ne_eq, not_true_eq_false, ↓reduceIte, bind, Except.bind, pure, Except.pure,
    Functor.map, Except.map]
  exact ⟨_, rfl⟩

theorem certifies_honest (d : NetworkCheck.Direction) (q : Nat) (x : ByteArray) (hq : q < 2 ^ 20) (hx : x.size = 64) :
    certifies honestParams d 0 (honestCert d 0 q x) = true := by
  obtain ⟨m, hm⟩ := honest_message d x hx
  obtain ⟨sig, hs, hv, hsq, hy, hpath, hC, hys, hps⟩ :=
    Properties.CertifierDevice.Proofs.Lms.verify_signAt (q := q) m honest_make honestKey_ok hq
  have hdec := Properties.CertifierDevice.Proofs.Lms.decode_encode Lms.deployed sig (by rw [hsq]; omega) hC hy hpath
    hys hps
  have hsig : (honestCert d 0 q x).signature = sig.encode := by
    show (((NetworkCheck.certificateMessage honestI honestSession "link" d ((0 : Nat) : Int) x).bind
      (Lms.signAt honestKey q)).toOption.map Lms.Signature.encode).getD .empty = sig.encode
    rw [hm, except_bind_ok, hs]
    try rfl
  refine certifies_of (m := m) (sig := sig) rfl rfl rfl honest_keyId.symm hm ?_ hv
  rw [hsig]
  exact hdec

theorem windowCheck_honest (hj : JsonStrBeq) (hr : HonestProfileReads) :
    windowCheck honestParams (clearClocks honestSyncs) #[] honestParams.first honestRegistration honestWindow =
      .ok () := by
  have hprog : honestRegistration.record.getObjValD "program" = .str "program" := rfl
  have hwin : honestRegistration.record.getObjValD "window" = served 0 := rfl
  refine windowCheck_of (eg := honestCert .egress 0 0 ⟨Array.replicate 64 5⟩)
    (ig := honestCert .ingress 0 1 ⟨Array.replicate 64 6⟩) (clock := honestCert .egress 0 2 ⟨Array.replicate 64 7⟩)
    (prof := honestIntegrity) (pub := Json.mkObj []) ?_ ?_ rfl ?_ ?_ rfl ?_ ?_ (by simp) ?_ ?_
  · rw [hprog]
    exact hj _
  · rw [hwin]
    rfl
  · exact certifies_honest .egress 0 _ (by decide) (by decide)
  · exact certifies_honest .ingress 1 _ (by decide) (by decide)
  · exact certifies_honest .egress 2 _ (by decide) (by decide)
  · decide +kernel
  · intro what
    exact answered_of rfl rfl rfl rfl rfl hr
  · decide +kernel

end Period

/-- **An honest period is accepted** (`PeriodAcceptMeans`' honest lemma): the check accepts one clock-form link-window,
registered as the link's Program at window 0 and proved by one unit, both ends and the clock certified with the
certifier's key at the deployed LMS parameters, under the two named assumptions about code the kernel can't evaluate. -/
theorem periodAcceptMeans_honest (hj : JsonStrBeq) (hr : HonestProfileReads) :
    protocol.check honestParams honestCommitment honestEvidence =
      .accept (terms #[] #[] #[] 0 1 1) { link := "link", windows := 1 } :=
  Period.check_one (ss := honestSyncs) rfl rfl rfl (Period.windowCheck_honest hj hr)

/-- **No period check that accepts every period has `PeriodAcceptMeans`' property** (its nontrivial lemma): such a check
accepts the honest parameters' period with a commitment that registers no link-window, and none of it was accepted. -/
theorem periodAcceptMeans_nontrivial : ¬ PeriodAcceptMeansOf acceptAllPeriod := by
  intro h
  exact absurd (h honestParams ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ _ _ rfl).1 (by decide)

end Properties.NetworkCertifier
