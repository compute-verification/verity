import Properties.NetworkCertifier.Period
import Proofs.Properties.CertifierDevice.Lms

/-!
# Proof that an accepted period accepted every link-window (`Properties.NetworkCertifier.PeriodAcceptMeans`)

`NetworkCheck.Period.check` accepts only when no finding is left (`firstFinding … = none`), so each link-window's
check of the parameters' form returned `ok`: `windowCheck` in the clock form, `rowsWindowCheck` in the rows form. Each
returns `ok` only past every refusal, which is `PeriodWindowAccepted` or `PeriodRowsWindowAccepted`.

Stubbed (`sorry`) where the check's clock form came to run one-stage v2 on the blobs: `answered_ok`, `windowCheck_ok`,
`rowsWindowCheck_ok`, `periodAcceptMeans` and the honest lemma, until the restated statements are agreed.
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

theorem answered_ok {what : String} {reg : Json} {a : Core.Answer} {v : Core.Integrity × Lean.Json}
    (h : answered what reg a = .ok v) :
    a.protocol = oneStage ∧ (∃ d, Flock.WindowRecord.digest reg = .ok d ∧ (a.digest == some d) = true) ∧
      a.verdict = .accept ∧
      ∃ prof, a.published = some v.2 ∧ a.profile = some prof ∧ Core.Integrity.ofJson prof = .ok v.1 := by
  sorry

theorem windowCheck_ok {p : Params} {l : ClockLink} {earlier : Array ByteArray} {log : Array Json} {w : Nat}
    {reg : Json} {x : Window} {bs : Core.Blobs} (h : windowCheck p l (clearClocks l.syncSet) earlier log w reg x bs = .ok ()) :
    PeriodWindowAccepted p l earlier log w reg x bs := by
  sorry

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

theorem rowsWindowCheck_ok {p : Params} {s : Schedule} {sd : ByteArray} {audits : Array Json} {w : Nat}
    {reg : Json} {x : Window} {f : Facts × Nat} (h : rowsWindowCheck p s sd audits w reg x = .ok f) :
    PeriodRowsWindowAccepted p s sd w reg x := by
  sorry

theorem map_ok {α β : Type} {g : α → β} {r : Found α} {b : β} (h : g <$> r = .ok b) : ∃ a, r = .ok a := by
  cases r with
  | error x => cases h
  | ok a => exact ⟨a, rfl⟩

end Period

theorem periodAcceptMeans : PeriodAcceptMeans := by
  sorry

theorem periodAcceptMeans_iff : PeriodAcceptMeans ↔ PeriodAcceptMeansOf check := Iff.rfl

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

end Period

/-- **An honest period is accepted** (`PeriodAcceptMeans`' honest lemma): under `HonestWindowProved`, the check
accepts one clock-form link-window, registered as the link's Program at window 0 and proved by one unit that one-stage
v2 accepts and that opens, both ends and the clock certified with the certifier's key at the deployed LMS parameters. -/
theorem periodAcceptMeans_honest (hv : HonestWindowProved) :
    ∃ (l : ClockLink) (prog : String) (reg : Json) (de di : ByteArray) (s : Salts)
      (ev : SampledProofs.OneStageZk.Evidence) (bs : Core.Blobs),
      check (honestParams l prog) (honestCommitment reg) (honestEvidence de di s ev) bs =
        .accept (terms #[] #[] #[] 0 1 1) { link := "link", windows := 1 } := by
  sorry

/-- **No period check that accepts every period has `PeriodAcceptMeans`' property** (its nontrivial lemma): such a check
accepts the honest parameters' period with a commitment that registers no link-window, and none of it was accepted. -/
theorem periodAcceptMeans_nontrivial : ¬ PeriodAcceptMeansOf acceptAllPeriod := by
  intro h
  exact absurd (h (honestParams ⟨honestSyncs, "", ⟨⟨0, 0, 0, 0, 0⟩, none⟩, ⟨⟨0, 0, 0, 0, 0⟩, none⟩, .null, ""⟩ "")
    ⟨#[], #[], .empty⟩ ⟨#[], #[]⟩ .empty _ _ rfl).1 (by decide)

end Properties.NetworkCertifier
