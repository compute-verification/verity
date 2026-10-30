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

local instance instLawfulBEqByteArrayPeriod : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

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
      · cases hb : (a.digest == some reg.digest) <;> simp_all [bne]
    · cases h
  · cases h
  · cases h

theorem opening_ok {p : Params} {l : ClockLink} {w : Nat} {reg : Registration} {certs : Array Certificate} {s : Salts}
    {e : SampledProofs.OneStageZk.Evidence} {bs : Core.Blobs} (h : opening p l w reg certs s e bs = .ok ()) :
    ∃ file row, bs.get e.pub "public file" = .ok file ∧
      NetworkCheck.LinkWindow.pubRow (linkCase p l w (rootOf reg.record) certs e.base.provedUnits (some file) s) =
        .ok (some row) ∧
      NetworkCheck.LinkWindow.opened (linkCase p l w (rootOf reg.record) certs e.base.provedUnits (some file) s)
        (some row) = some 3 := by
  simp only [opening, refuse, pure, Except.pure] at h
  split at h
  case h_2 => cases h
  rename_i file hfile
  split at h
  · cases h
  · cases h
  · rename_i row hrow
    split at h
    · rename_i c
      exact ⟨file, row, hfile, hrow, by simpa using c⟩
    · cases h

theorem windowCheck_ok {p : Params} {l : ClockLink} {earlier : Array ByteArray} {log : Array Json} {w : Nat}
    {reg : Registration} {x : Window} {bs : Core.Blobs}
    (h : windowCheck p l (clearClocks l.syncSet) earlier log w reg x bs = .ok ()) :
    PeriodWindowAccepted p l earlier log w reg x bs := by
  simp only [windowCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  case h_2 => cases h
  rename_i clock salts ev hx
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
            case h_2 => cases h
            rename_i dom hdom
            obtain ⟨ps, hps⟩ : ∃ ps, windowParams l dom log = .ok ps := by
              rcases hps : windowParams l dom log with err | ps
              · simp only [hps] at h; cases h
              · exact ⟨ps, rfl⟩
            simp only [hps] at h
            obtain ⟨pub, prof, hv⟩ : ∃ pub prof,
                SampledProofs.OneStageZk.checkBlobs ps ⟨reg.record, reg.digest⟩ ev bs = .accept pub prof := by
              rcases hv : SampledProofs.OneStageZk.checkBlobs ps ⟨reg.record, reg.digest⟩ ev bs with
                ⟨pub, prof⟩ | r | r
              · exact ⟨pub, prof, rfl⟩
              all_goals (simp only [hv] at h; cases h)
            simp only [hv] at h
            split at h
            · rename_i c6
              have ho : opening p l w reg x.certificates salts ev bs = .ok () := h
              obtain ⟨file, row, hfile, hrow, hop⟩ := opening_ok ho
              have r1 := (Bool.and_eq_true _ _).mp c1
              have q := (Bool.and_eq_true _ _).mp c6
              refine ⟨⟨r1.1, by simpa using r1.2⟩, ?_,
                ⟨clock, salts, ev, hx, c3, by simpa using c4, by simpa using c5⟩,
                ⟨clock, salts, ev, hx, dom, ps, pub, prof, hdom, hps, hv, q.1, q.2⟩,
                ⟨clock, salts, ev, hx, file, row, hfile, hrow, hop⟩⟩
              obtain ⟨hs, hall⟩ := (Bool.and_eq_true _ _).mp c2
              unfold PeriodWindowCertified
              generalize x.certificates = cs at hs hall
              rcases cs with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨y, ys⟩⟩⟩⟩
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

theorem rowsWindowCheck_ok {p : Params} {s : Schedule} {sd : ByteArray} {audits : Array Registration} {w : Nat}
    {reg : Registration} {x : Window} {f : Facts × Nat} (h : rowsWindowCheck p s sd audits w reg x = .ok f) :
    PeriodRowsWindowAccepted p s sd w reg x := by
  simp only [rowsWindowCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  case h_2 => cases h
  rename_i rs answer hx
  split at h
  · rename_i c1
    split at h
    · rename_i c2
      split at h
      · rename_i c3
        rcases rs with ⟨_ | ⟨ri, _ | ⟨re, _ | ⟨y, ys⟩⟩⟩⟩
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
              · rename_i c5
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
                  refine ⟨c1, ?_, ⟨ri, re, answer, hx, rowsCheck_ok hi, rowsCheck_ok he, hh.1, hh.2⟩,
                    ⟨_, answer, hx, a1, a2, a3, v.2, prof, v.1, l, b, a4, a5, a6, hl, hun, hd, hb⟩⟩
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
  intro p c e bs pub prof h
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
  rcases hform : p.form with l | ⟨s, sd, scan⟩
  · simp only [hform] at hall
    have h0 := hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_right _ (List.mem_map.2 ⟨_, Array.mem_toList_iff.2
        (Array.mem_map.2 ⟨_, hmem, rfl⟩), rfl⟩))))))
    obtain ⟨_, ha⟩ := Period.findingOf_none h0
    obtain ⟨u, hu⟩ := Period.map_ok ha
    exact Period.windowCheck_ok hu
  · simp only [hform] at hall
    have h0 := hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _
      (List.mem_append_left _ (List.mem_append_right _ (List.mem_map.2 ⟨_, Array.mem_toList_iff.2
        (Array.mem_map.2 ⟨_, hmem, rfl⟩), rfl⟩))))))
    obtain ⟨_, ha⟩ := Period.findingOf_none h0
    obtain ⟨f, hu⟩ := Period.map_ok ha
    exact Period.rowsWindowCheck_ok hu

theorem periodAcceptMeans_iff : PeriodAcceptMeans ↔ PeriodAcceptMeansOf check := Iff.rfl

/-! ## The honest and nontrivial lemmas

The honest period's acceptance goes through lemmas over arbitrary parameters, windows and certificates, each instantiated
by rewriting: the kernel never compares a term holding `honestKey` with anything but itself, since reducing one would
build the key's 2^20-leaf tree. -/

namespace Period

attribute [local instance] instLawfulBEqByteArrayPeriod

theorem except_bind_ok {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a : Except ε α).bind f = f a := rfl

theorem certifies_of {p : Params} {d : NetworkCheck.Direction} {w : Nat} {c : Certificate} {m : ByteArray}
    {sig : Lms.Signature} (hd : c.direction = d) (hl : c.link = p.link) (hw : c.window = w)
    (hk : c.keyId = p.key.extract 0 16)
    (hm : NetworkCheck.certificateMessage c.keyId p.session c.link c.direction c.window c.digest = .ok m)
    (hs : Lms.Signature.decode p.lms c.signature = .ok sig) (hv : Lms.verify p.lms p.key m sig = .ok true) :
    certifies p d w c = true := by
  unfold certifies certifiesOn NetworkCheck.Check.Certificate.signs
  rw [hm, hs]
  simp only [hv, hd, hl, hw, hk, beq_self_eq_true, Bool.and_true, Except.toOption]

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

theorem honestParams_key (l : ClockLink) (prog : String) : (honestParams l prog).key = honestKey.I ++ honestKey.tree[1]! :=
  rfl

theorem honest_keyId (l : ClockLink) (prog : String) : (honestParams l prog).key.extract 0 16 = honestI := by
  rw [honestParams_key l prog, honestKey_I]
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

theorem certifies_honest (l : ClockLink) (prog : String) (d : NetworkCheck.Direction) (q : Nat) (x : ByteArray)
    (hq : q < 2 ^ 20) (hx : x.size = 64) : certifies (honestParams l prog) d 0 (honestCert d 0 q x) = true := by
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
  refine certifies_of (m := m) (sig := sig) rfl rfl rfl (honest_keyId l prog).symm hm ?_ hv
  rw [hsig]
  exact hdec

theorem opening_of {p : Params} {l : ClockLink} {w : Nat} {reg : Registration} {certs : Array Certificate} {s : Salts}
    {e : SampledProofs.OneStageZk.Evidence} {bs : Core.Blobs} {file row : ByteArray}
    (hfile : bs.get e.pub "public file" = .ok file)
    (hrow : NetworkCheck.LinkWindow.pubRow (linkCase p l w (rootOf reg.record) certs e.base.provedUnits (some file) s) =
      .ok (some row))
    (hop : NetworkCheck.LinkWindow.opened (linkCase p l w (rootOf reg.record) certs e.base.provedUnits (some file) s)
      (some row) = some 3) :
    opening p l w reg certs s e bs = .ok () := by
  simp [opening, hfile, hrow, hop, pure, Except.pure]

theorem windowCheck_of_accepted {p : Params} {l : ClockLink} {earlier : Array ByteArray} {log : Array Json} {w : Nat}
    {reg : Registration} {x : Window} {bs : Core.Blobs} (ha : PeriodWindowAccepted p l earlier log w reg x bs) :
    windowCheck p l (clearClocks l.syncSet) earlier log w reg x bs = .ok () := by
  obtain ⟨⟨hp, hw⟩, ⟨eg, ig, hc, heg, hig⟩, ⟨clock, salts, ev, hx, hcl, hclear, hearlier⟩,
    ⟨clock₂, salts₂, ev₂, hx₂, dom, ps, pub, prof, hdom, hps, hv, hl, hdr⟩,
    ⟨clock₃, salts₃, ev₃, hx₃, file, row, hfile, hrow, hop⟩⟩ := ha
  have hclear' : clock.digest ∉ clearClocks l.syncSet := by simpa using hclear
  have hearlier' : clock.digest ∉ earlier := by simpa using hearlier
  have hp' : (reg.record.getObjValD "program" == .str p.program) = true := hp
  rw [hx] at hx₂ hx₃
  cases hx₂
  cases hx₃
  have hop' := opening_of hfile hrow hop
  rw [hc] at hop'
  have hdr' := hdr
  try simp only [beq_iff_eq] at hdr'
  simp only [windowCheck, hx, hp', hw, hc, beq_self_eq_true, Bool.and_true, Bool.true_and, Array.size,
    List.length_cons, List.length_nil]
  simp [Array.zip, Array.all, heg, hig, hcl, hclear', hearlier', hdom, hps, hv, hl, hdr', hop', bind,
    Except.bind, pure, Except.pure, Id.run]

theorem check_one {p : Params} {l : ClockLink} {reg : Registration} {x : Window} {r : ByteArray} {bs : Core.Blobs}
    (hf : p.form = .clock l) (hw : p.windows = 1) (ha : p.audits = #[])
    (h : windowCheck p l (clearClocks l.syncSet) #[] #[] p.first reg x bs = .ok ()) :
    check p ⟨#[], #[reg], r⟩ ⟨#[], #[x]⟩ bs =
      .accept (terms #[] #[] p.cells p.jitter (l.syncSet.deltas.length * l.syncSet.rhos.length) 1)
        { link := p.link, windows := 1 } := by
  unfold check
  simp [hf, hw, ha, h, auditChecks, wiredChecks, stateChecks, Functor.map, Except.map, findingOf, firstFinding,
    Except.toOption]

end Period

/-- **An honest period is accepted** (`PeriodAcceptMeans`' honest lemma): under `HonestWindowProved`, the check
accepts one clock-form link-window, registered as the link's Program at window 0 and proved by one unit that one-stage
v2 accepts and that opens, both ends and the clock certified with the certifier's key at the deployed LMS parameters. -/
theorem periodAcceptMeans_honest (hv : HonestWindowProved) :
    ∃ (l : ClockLink) (prog : String) (reg : Registration) (de di : ByteArray) (s : Salts)
      (ev : SampledProofs.OneStageZk.Evidence) (bs : Core.Blobs),
      check (honestParams l prog) (honestCommitment reg) (honestEvidence de di s ev) bs =
        .accept (terms #[] #[] #[] 0 1 1) { link := "link", windows := 1 } := by
  obtain ⟨l, prog, reg, de, di, s, ev, bs, hsync, hde, hdi, hreg, hv2, hop⟩ := hv
  refine ⟨l, prog, reg, de, di, s, ev, bs, ?_⟩
  have hw := Period.windowCheck_of_accepted (p := honestParams l prog) (l := l) (earlier := #[]) (log := #[]) (w := 0)
    (reg := reg) (x := honestWindow de di s ev) (bs := bs) ⟨hreg,
    ⟨honestCert .egress 0 0 de, honestCert .ingress 0 1 di, rfl, Period.certifies_honest l prog .egress 0 de
      (by decide) hde, Period.certifies_honest l prog .ingress 1 di (by decide) hdi⟩,
    ⟨honestCert .egress 0 2 ⟨Array.replicate 64 7⟩, s, ev, rfl,
      Period.certifies_honest l prog .egress 2 _ (by decide) (by decide), by rw [hsync]; decide +kernel,
      by first | rfl | simp⟩,
    hv2, hop⟩
  have := Period.check_one (p := honestParams l prog) (l := l) (reg := reg) (x := honestWindow de di s ev)
    (r := .empty) (bs := bs) rfl rfl rfl hw
  rw [hsync] at this
  exact this

/-- **No period check that accepts every period has `PeriodAcceptMeans`' property** (its nontrivial lemma): such a check
accepts the honest parameters' period with a commitment that registers no link-window, and none of it was accepted. -/
theorem periodAcceptMeans_nontrivial : ¬ PeriodAcceptMeansOf acceptAllPeriod := by
  intro h
  exact absurd (h (honestParams ⟨honestSyncs, "", ⟨⟨0, 0, 0, 0, 0⟩, none⟩, ⟨⟨0, 0, 0, 0, 0⟩, none⟩, .null, ""⟩ "")
    ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ .empty _ _ rfl).1 (by decide)

end Properties.NetworkCertifier
