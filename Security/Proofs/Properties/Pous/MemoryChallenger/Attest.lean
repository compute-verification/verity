import Properties.Attest
import Proofs.Properties.Pous.MemoryChallenger.Tree
import Proofs.Flock.Claims.Finder
import Proofs.Properties.CertifierDevice.Lms

/-!
# Proofs of a party's signed periods (`Properties.Attest`)

**Soundness.** An accepted period (`accepts_ok`) has the public certificate verify under the identity over the lower
key's party-key message, the root's signature under the lower key over the root's message, and each read open to the
root.
A role digest is SHA-512 of `[tag] ‖ I ‖ [context] ‖ [record]`, each `[·]` a u64 length then the bytes, so its input
determines its tag, identifier, context and record (`roleInput_inj`), and a context its fields, every field under `2^64`
bytes since the context is (`tail_inj`). Under no SHA-512 collision among the run's strings (`CF`), a message the
devices signed under a key and the check's message under it, with one digest, have one input: the tags tell the
certificate from the root and the liveness records, and the contexts give the party key the identity issued
(`signed_party`) and the period the lower key closed (`signed_root`), with its record count and root. A read's climb
(`climbRec`, `climb_eq`) that reaches that root starts at the tree's leaf at its position, level by level down from the
root, each node's input fixing its two children (`climb_bind`), so the read's record is the period's
(`opening_bind`). That is `PeriodSigned`; otherwise a string pair collides or a signature verifies over a message its key
never signed. `PeriodSound` is the same at each coin, with the collision `findColl` finds (`finds`) and the forger's
output (`bad_cases`), and the union bound.

**Zero knowledge.** The party's code keeps, on inputs that differ only in the records' bytes, one key, one LMS counter
and one number of open records (`trace_clear`): a block sends the same liveness record, and a close, refused for no root
or for every one since each root is 64 bytes, sends nothing in the clear. That is `LivenessContentFree`, at an input
list and its skeleton, whose skeleton is the same (`skeleton_skeleton`). Under `Confined` the party emits its code's
messages, under `Faithful` on its true inputs, and under `WitnessPathPrivate` nobody overhears its path to the prover,
so the auditor sees what the code sends in the clear on its coins and the skeleton (`auditorSeesSkeleton`); under
`Secret` the prover gets its messages and no access to its state (`proverSeesMessages`).

**Honest and nontrivial lemmas.** Each guarantee's hypotheses hold at the honest deployed instance: keys `Lms.key`
derives at `Lms.deployed`, the identity certifying the memory challenger's lower key at leaf 0 and the lower key
signing the root of the one-record period of `HONEST_RECORD` at leaf 0 (`honest_accepts`, by
`Properties.CertifierDevice.Proofs.Lms.lmsCorrect`; `periodSigned_honest`, `periodSound_honest`), and the party's code
run as it is meant to (`meantRun`). Each conclusion fails somewhere: at the empty run, which no device signed and which
carries no break (`periodSigned_nontrivial`), so a verifier that accepts everything fails `PeriodSigned` and
`PeriodSound` (`periodSigned_acceptAll`, `periodSound_acceptAll`; `periodSound_nontrivial` at `acceptAll`); at the
party's whole output, which carries the records (`livenessContentFree_nontrivial`); at an overheard witness path
(`auditorSeesSkeleton_nontrivial`); and at a prover that reads the party's state (`proverSeesMessages_nontrivial`).
-/

namespace MemoryChallenger.Attest.Properties.SecurityProofs

open Flock Flock.FV3.Sha512
open Accounting.Assumptions
open MemoryChallenger.Properties.SecurityProofs.Tree (CF next base lev le_pow_log2ceil lev_size get!_mem hash_size
  lev_width levels_size levels_get levels_eq root_eq frame_last frame_two node_hash leaf_hash next_get isEmpty_of_size)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes be64_size be64_size_inj)

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
local instance instLawfulBEqByteArray_attest : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-! ## The role digests' inputs -/

theorem lenField_bs (b : ByteArray) : bs (Lms.lenField b) = bs (be64 b.size.toUInt64) ++ bs b := bs_append _ _

/-- A length-prefixed field of fewer than `2^64` bytes and what follows it determine the field and what follows. -/
theorem field_split {a a' : ByteArray} {r r' : List UInt8} (ha : a.size < 2 ^ 64) (ha' : a'.size < 2 ^ 64)
    (h : bs (Lms.lenField a) ++ r = bs (Lms.lenField a') ++ r') : a = a' ∧ r = r' := by
  rw [lenField_bs, lenField_bs, List.append_assoc, List.append_assoc] at h
  obtain ⟨e, h⟩ := split_bytes h (by rw [be64_size, be64_size])
  exact split_bytes h (be64_size_inj ha ha' e)

theorem roleDigest_ok {tag : String} {I c r m : ByteArray} (h : Lms.roleDigest tag I c r = .ok m) :
    I.size = 16 ∧ c.size < 2 ^ 64 ∧ r.size < 2 ^ 64 ∧ m = Sha512.hash (roleInput tag I c r) := by
  unfold Lms.roleDigest at h
  by_cases hI : I.size = 16
  · simp only [hI, ne_eq, not_true_eq_false, ↓reduceIte] at h
    split at h
    · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    rename_i hs
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact ⟨hI, by omega, by omega, h.symm⟩
  · simp [hI, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

theorem roleStrings_ok {tag : String} {I c r m : ByteArray} (h : Lms.roleDigest tag I c r = .ok m) :
    roleStrings tag I (.ok c) r = [roleInput tag I c r] := by
  simp [roleStrings, h]

/-- A role digest's input determines its tag, identifier, context and record. -/
theorem roleInput_inj {t t' : String} {I I' c c' r r' : ByteArray} (ht : t.toUTF8.size < 2 ^ 64)
    (ht' : t'.toUTF8.size < 2 ^ 64) (hI : I.size = 16) (hI' : I'.size = 16) (hc : c.size < 2 ^ 64)
    (hc' : c'.size < 2 ^ 64) (h : roleInput t I c r = roleInput t' I' c' r') : t = t' ∧ I = I' ∧ c = c' ∧ r = r' := by
  have e := congrArg bs h
  simp only [roleInput, bs_append, List.append_assoc] at e
  obtain ⟨et, e⟩ := field_split ht ht' e
  obtain ⟨eI, e⟩ := split_bytes e (by rw [hI, hI'])
  obtain ⟨ec, e⟩ := field_split hc hc' e
  rw [lenField_bs, lenField_bs] at e
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact ⟨String.toByteArray_inj.mp et, eI, ec, bs_inj e⟩

theorem tag_party : Lms.PARTY_KEY_TAG.toUTF8.size < 2 ^ 64 := by decide
theorem tag_root : Lms.PERIOD_ROOT_TAG.toUTF8.size < 2 ^ 64 := by decide
theorem tag_live : Lms.LIVENESS_TAG.toUTF8.size < 2 ^ 64 := by decide

/-! ## The contexts -/

theorem u64_ok {x : Nat} {w : String} {b : ByteArray} (h : Lms.u64 x w = .ok b) : x < 2 ^ 64 ∧ b = be64 x.toUInt64 := by
  unfold Lms.u64 at h
  split at h
  · simp only [pure, Except.pure, Except.ok.injEq] at h
    exact ⟨by assumption, h.symm⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem partyContext_ok {d c : ByteArray} {p : String} (h : Lms.partyContext d p = .ok c) :
    d.size = 64 ∧ c = Lms.lenField d ++ Lms.lenField p.toUTF8 := by
  unfold Lms.partyContext at h
  by_cases hd : d.size = 64
  · simp only [hd, Lms.DEPLOYMENT_BYTES, ne_eq, not_true_eq_false, ↓reduceIte, pure, Except.pure,
      Except.ok.injEq] at h
    exact ⟨hd, h.symm⟩
  · simp [hd, Lms.DEPLOYMENT_BYTES, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

theorem periodContext_ok {d a c : ByteArray} {p l : String} {per : Nat} (h : Lms.periodContext d p a l per = .ok c) :
    ∃ pc, Lms.partyContext d p = .ok pc ∧ per < 2 ^ 64 ∧
      c = pc ++ Lms.lenField a ++ Lms.lenField l.toUTF8 ++ be64 per.toUInt64 := by
  unfold Lms.periodContext at h
  cases hp : Lms.partyContext d p with
  | error e => simp [hp, bind, Except.bind] at h
  | ok pc =>
    cases hu : Lms.u64 per "a certification period's index" with
    | error e => simp [hp, hu, bind, Except.bind] at h
    | ok u =>
      simp only [hp, hu, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      obtain ⟨hper, rfl⟩ := u64_ok hu
      exact ⟨pc, rfl, hper, h.symm⟩

/-- A period root's or a liveness record's context: the party's, the attachment, the link, the period's index and one
more u64. -/
theorem tail_ok {d a c : ByteArray} {p l : String} {per x : Nat} {w : String}
    (h : (do pure ((← Lms.periodContext d p a l per) ++ (← Lms.u64 x w)) : Lms.E ByteArray) = .ok c) :
    d.size = 64 ∧ per < 2 ^ 64 ∧ x < 2 ^ 64 ∧
      c = Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++ Lms.lenField l.toUTF8 ++ be64 per.toUInt64 ++
        be64 x.toUInt64 := by
  cases hp : Lms.periodContext d p a l per with
  | error e => simp [hp, bind, Except.bind] at h
  | ok pc =>
    cases hu : Lms.u64 x w with
    | error e => simp [hp, hu, bind, Except.bind] at h
    | ok u =>
      simp only [hp, hu, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      obtain ⟨hx, rfl⟩ := u64_ok hu
      obtain ⟨q, hq, hper, rfl⟩ := periodContext_ok hp
      obtain ⟨hd, rfl⟩ := partyContext_ok hq
      exact ⟨hd, hper, hx, h.symm⟩

theorem periodRootContext_ok {d a c : ByteArray} {p l : String} {per n : Nat}
    (h : Lms.periodRootContext d p a l per n = .ok c) :
    d.size = 64 ∧ per < 2 ^ 64 ∧ n < 2 ^ 64 ∧
      c = Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++ Lms.lenField l.toUTF8 ++ be64 per.toUInt64 ++
        be64 n.toUInt64 := tail_ok h

theorem livenessContext_ok {d a c : ByteArray} {p l : String} {per b : Nat}
    (h : Lms.livenessContext d p a l per b = .ok c) :
    d.size = 64 ∧ per < 2 ^ 64 ∧ b < 2 ^ 64 ∧
      c = Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++ Lms.lenField l.toUTF8 ++ be64 per.toUInt64 ++
        be64 b.toUInt64 := tail_ok h

theorem lenField_size (b : ByteArray) : (Lms.lenField b).size = 8 + b.size := by
  simp [Lms.lenField, ByteArray.size_append, be64_size]

theorem partyContext_inj {d d' c : ByteArray} {p p' : String} (h : Lms.partyContext d p = .ok c)
    (h' : Lms.partyContext d' p' = .ok c) : d = d' ∧ p = p' := by
  obtain ⟨hd, rfl⟩ := partyContext_ok h
  obtain ⟨hd', e⟩ := partyContext_ok h'
  have e := congrArg bs e
  simp only [bs_append] at e
  obtain ⟨ed, e⟩ := field_split (by omega) (by omega) e
  rw [lenField_bs, lenField_bs] at e
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact ⟨ed, String.toByteArray_inj.mp (bs_inj e)⟩

/-- A context of the deployment, the party, the attachment, the link, the period's index and one more u64, of fewer
than `2^64` bytes, determines them. -/
theorem tail_inj {d d' a a' : ByteArray} {p p' l l' : String} {per per' x x' : Nat} (hd : d.size = 64)
    (hd' : d'.size = 64) (hper : per < 2 ^ 64) (hper' : per' < 2 ^ 64) (hx : x < 2 ^ 64) (hx' : x' < 2 ^ 64)
    (hc : (Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++ Lms.lenField l.toUTF8 ++ be64 per.toUInt64 ++
        be64 x.toUInt64).size < 2 ^ 64)
    (h : Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++ Lms.lenField l.toUTF8 ++ be64 per.toUInt64 ++
        be64 x.toUInt64 = Lms.lenField d' ++ Lms.lenField p'.toUTF8 ++ Lms.lenField a' ++ Lms.lenField l'.toUTF8 ++
        be64 per'.toUInt64 ++ be64 x'.toUInt64) :
    d = d' ∧ p = p' ∧ a = a' ∧ l = l' ∧ per = per' ∧ x = x' := by
  have hc' := hc
  rw [h] at hc'
  simp only [ByteArray.size_append, lenField_size, be64_size] at hc hc'
  have e := congrArg bs h
  simp only [bs_append, List.append_assoc] at e
  obtain ⟨ed, e⟩ := field_split (by omega) (by omega) e
  obtain ⟨ep, e⟩ := field_split (by omega) (by omega) e
  obtain ⟨ea, e⟩ := field_split (by omega) (by omega) e
  obtain ⟨el, e⟩ := field_split (by omega) (by omega) e
  obtain ⟨eper, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact ⟨ed, String.toByteArray_inj.mp ep, ea, String.toByteArray_inj.mp el, be64_size_inj hper hper' eper,
    be64_size_inj hx hx' (bs_inj e)⟩

/-! ## The messages -/

theorem partyKeyMessage_ok {I d lower m : ByteArray} {p : String} (h : Lms.partyKeyMessage I d p lower = .ok m) :
    ∃ c, Lms.partyContext d p = .ok c ∧ Lms.roleDigest Lms.PARTY_KEY_TAG I c lower = .ok m := by
  unfold Lms.partyKeyMessage at h
  cases hc : Lms.partyContext d p with
  | error e => simp [hc, bind, Except.bind] at h
  | ok c =>
    simp only [hc, bind, Except.bind] at h
    exact ⟨c, rfl, h⟩

theorem periodRootMessage_ok {I d a root m : ByteArray} {p l : String} {per n : Nat}
    (h : Lms.periodRootMessage I d p a l per n root = .ok m) :
    ∃ c, Lms.periodRootContext d p a l per n = .ok c ∧ Lms.roleDigest Lms.PERIOD_ROOT_TAG I c root = .ok m := by
  unfold Lms.periodRootMessage at h
  cases hc : Lms.periodRootContext d p a l per n with
  | error e => simp [hc, bind, Except.bind] at h
  | ok c =>
    simp only [hc, bind, Except.bind] at h
    exact ⟨c, rfl, h⟩

theorem livenessMessage_ok {I d a m : ByteArray} {p l : String} {per b : Nat}
    (h : Lms.livenessMessage I d p a l per b = .ok m) :
    ∃ c, Lms.livenessContext d p a l per b = .ok c ∧ Lms.roleDigest Lms.LIVENESS_TAG I c ByteArray.empty = .ok m := by
  unfold Lms.livenessMessage at h
  cases hc : Lms.livenessContext d p a l per b with
  | error e => simp [hc, bind, Except.bind] at h
  | ok c =>
    simp only [hc, bind, Except.bind] at h
    exact ⟨c, rfl, h⟩

/-- Under no collision among `S`, two role digests that agree, each with its input in `S`, have one tag, identifier,
context and record. -/
theorem role_bind {S : List ByteArray} (cf : CF S) {t t' : String} {I I' c c' r r' m : ByteArray}
    (ht : t.toUTF8.size < 2 ^ 64) (ht' : t'.toUTF8.size < 2 ^ 64)
    (h : Lms.roleDigest t I c r = .ok m) (h' : Lms.roleDigest t' I' c' r' = .ok m)
    (hm : roleInput t I c r ∈ S) (hm' : roleInput t' I' c' r' ∈ S) : t = t' ∧ I = I' ∧ c = c' ∧ r = r' := by
  obtain ⟨hI, hc, -, rfl⟩ := roleDigest_ok h
  obtain ⟨hI', hc', -, e⟩ := roleDigest_ok h'
  exact roleInput_inj ht ht' hI hI' hc hc' (cf _ hm _ hm' e)

/-- A role digest's input is among the strings its role strings list, when it digests. -/
theorem role_mem {tag : String} {I c r m : ByteArray} {ctx : Lms.E ByteArray} (hc : ctx = .ok c)
    (h : Lms.roleDigest tag I c r = .ok m) : roleInput tag I c r ∈ roleStrings tag I ctx r := by
  rw [hc, roleStrings_ok h]
  exact List.mem_singleton_self _

/-! ## What the devices hash -/

theorem issued_mem {dev : Device} {pk : ByteArray} {i : Issued} (hi : i ∈ dev.issued pk) :
    ∀ x ∈ roleStrings Lms.PARTY_KEY_TAG (pk.extract 0 16) (Lms.partyContext i.deployment i.party) i.lower,
      x ∈ keyStrings dev pk := by
  intro x hx
  simp only [keyStrings, List.mem_append, List.mem_flatMap]
  exact Or.inl (Or.inl ⟨i, hi, hx⟩)

theorem closed_mem {dev : Device} {pk : ByteArray} {c : Closed} (hc : c ∈ dev.closed pk) :
    ∀ x ∈ roleStrings Lms.PERIOD_ROOT_TAG (pk.extract 0 16)
      (Lms.periodRootContext c.deployment c.party c.attachment c.link c.period c.records.size) (Attest.root c.records) ++
      treeStrings c.records, x ∈ keyStrings dev pk := by
  intro x hx
  simp only [keyStrings, List.mem_append, List.mem_flatMap]
  exact Or.inl (Or.inr ⟨c, hc, List.mem_append.mp hx⟩)

theorem live_mem {dev : Device} {pk : ByteArray} {l : Live} (hl : l ∈ dev.live pk) :
    ∀ x ∈ roleStrings Lms.LIVENESS_TAG (pk.extract 0 16)
      (Lms.livenessContext l.deployment l.party l.attachment l.link l.period l.block) ByteArray.empty,
      x ∈ keyStrings dev pk := by
  intro x hx
  simp only [keyStrings, List.mem_append, List.mem_flatMap]
  exact Or.inr ⟨l, hl, hx⟩

/-- Under no collision among `S`, which holds what the devices hash under `pk`, a party-key message the devices signed
under `pk`, whose input is in `S`, is one the key issued. -/
theorem signed_party {S : List ByteArray} (cf : CF S) {dev : Device} {pk d lower m : ByteArray} {p : String}
    (hK : ∀ x ∈ keyStrings dev pk, x ∈ S) (hm : Lms.partyKeyMessage (pk.extract 0 16) d p lower = .ok m)
    (hin : ∀ x ∈ roleStrings Lms.PARTY_KEY_TAG (pk.extract 0 16) (Lms.partyContext d p) lower, x ∈ S)
    (hs : Signed dev pk m) : ∃ i ∈ dev.issued pk, i.deployment = d ∧ i.party = p ∧ i.lower = lower := by
  obtain ⟨c, hc, hr⟩ := partyKeyMessage_ok hm
  have hin := hin _ (role_mem hc hr)
  rcases hs with ⟨i, hi, him⟩ | ⟨cl, hcl, hcm⟩ | ⟨lv, hlv, hlm⟩
  · obtain ⟨c', hc', hr'⟩ := partyKeyMessage_ok him
    obtain ⟨-, -, ec, el⟩ := role_bind cf tag_party tag_party hr hr' hin (hK _ (issued_mem hi _ (role_mem hc' hr')))
    subst ec
    obtain ⟨ed, ep⟩ := partyContext_inj hc hc'
    exact ⟨i, hi, ed.symm, ep.symm, el.symm⟩
  · obtain ⟨c', hc', hr'⟩ := periodRootMessage_ok hcm
    have hin' := hK _ (closed_mem hcl _ (List.mem_append_left _ (role_mem hc' hr')))
    exact absurd (role_bind cf tag_party tag_root hr hr' hin hin').1 (by decide)
  · obtain ⟨c', hc', hr'⟩ := livenessMessage_ok hlm
    exact absurd (role_bind cf tag_party tag_live hr hr' hin (hK _ (live_mem hlv _ (role_mem hc' hr')))).1
      (by decide)

/-- Under no collision among `S`, which holds what the devices hash under `pk`, a period root's message the devices
signed under `pk`, whose input is in `S`, is one of a period the key closed, with its fields and its root. -/
theorem signed_root {S : List ByteArray} (cf : CF S) {dev : Device} {pk d a root m : ByteArray} {p l : String}
    {per n : Nat} (hK : ∀ x ∈ keyStrings dev pk, x ∈ S)
    (hm : Lms.periodRootMessage (pk.extract 0 16) d p a l per n root = .ok m)
    (hin : ∀ x ∈ roleStrings Lms.PERIOD_ROOT_TAG (pk.extract 0 16) (Lms.periodRootContext d p a l per n) root, x ∈ S)
    (hs : Signed dev pk m) :
    ∃ c ∈ dev.closed pk, c.deployment = d ∧ c.party = p ∧ c.attachment = a ∧ c.link = l ∧ c.period = per ∧
      c.records.size = n ∧ Attest.root c.records = root := by
  obtain ⟨c, hc, hr⟩ := periodRootMessage_ok hm
  have hin := hin _ (role_mem hc hr)
  rcases hs with ⟨i, hi, him⟩ | ⟨cl, hcl, hcm⟩ | ⟨lv, hlv, hlm⟩
  · obtain ⟨c', hc', hr'⟩ := partyKeyMessage_ok him
    exact absurd (role_bind cf tag_root tag_party hr hr' hin (hK _ (issued_mem hi _ (role_mem hc' hr')))).1
      (by decide)
  · obtain ⟨c', hc', hr'⟩ := periodRootMessage_ok hcm
    have hin' := hK _ (closed_mem hcl _ (List.mem_append_left _ (role_mem hc' hr')))
    obtain ⟨-, -, ec, er⟩ := role_bind cf tag_root tag_root hr hr' hin hin'
    subst ec
    obtain ⟨-, hcs, -, -⟩ := roleDigest_ok hr
    obtain ⟨hd, hper, hn, rfl⟩ := periodRootContext_ok hc
    obtain ⟨hd', hper', hn', e⟩ := periodRootContext_ok hc'
    obtain ⟨ed, ep, ea, el, eper, en⟩ := tail_inj hd hd' hper hper' hn hn' hcs e
    exact ⟨cl, hcl, ed.symm, ep.symm, ea.symm, el.symm, eper.symm, en.symm, er.symm⟩
  · obtain ⟨c', hc', hr'⟩ := livenessMessage_ok hlm
    exact absurd (role_bind cf tag_root tag_live hr hr' hin (hK _ (live_mem hlv _ (role_mem hc' hr')))).1
      (by decide)

/-! ## The period tree -/

theorem leaves_size (vs : Array ByteArray) : (Attest.leaves vs).size = vs.size := by simp [Attest.leaves]

theorem leaves_width (vs : Array ByteArray) : ∀ x ∈ Attest.leaves vs, x.size = 64 := by
  intro x hx
  simp only [Attest.leaves, Array.mem_map] at hx
  obtain ⟨r, -, rfl⟩ := hx
  exact hash_size _

theorem leaves_isEmpty {vs : Array ByteArray} (h : vs.size ≠ 0) : (Attest.leaves vs).isEmpty = false :=
  isEmpty_of_size (by rw [leaves_size]; exact h)

theorem leaves_get {vs : Array ByteArray} {r : Nat} (hr : r < vs.size) :
    (Attest.leaves vs)[r]! = (Attest.periodDomain vs.size).leaf r r Attest.PERIOD_SCHEMA vs[r]! := by
  rw [getElem!_pos _ r (by rw [leaves_size]; exact hr)]
  simp [Attest.leaves]

theorem leaf_mem {vs : Array ByteArray} {r : Nat} (hr : r < vs.size) :
    FV3.input "leaf" [(Attest.periodDomain vs.size).id, FV3.uint r, FV3.uint r, Attest.PERIOD_SCHEMA.toUTF8, vs[r]!] ∈
      treeStrings vs := by
  simp only [treeStrings, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inl ⟨r, hr, rfl⟩

theorem node_mem {vs : Array ByteArray} (h0 : vs.size ≠ 0) {l i : Nat} (hl : l < log2ceil vs.size)
    (hi : i < (lev (Attest.periodDomain vs.size) (Attest.leaves vs) l).size / 2) :
    FV3.input "node" [(Attest.periodDomain vs.size).id, FV3.uint l, FV3.uint i,
      (lev (Attest.periodDomain vs.size) (Attest.leaves vs) l)[2 * i]!,
      (lev (Attest.periodDomain vs.size) (Attest.leaves vs) l)[2 * i + 1]!] ∈ treeStrings vs := by
  have hn := leaves_isEmpty h0
  have hs := leaves_size vs
  simp only [treeStrings, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range]
  refine Or.inr ⟨l, ?_, i, ?_, ?_⟩
  · rw [levels_size _ _ hn, hs]; omega
  · rw [levels_get _ _ hn (by rw [hs]; omega)]; exact hi
  · rw [levels_get _ _ hn (by rw [hs]; omega)]

theorem root_width (vs : Array ByteArray) : (Attest.root vs).size = 64 := by
  unfold Attest.root
  by_cases h0 : vs.size = 0
  · have : Attest.leaves vs = #[] := Array.size_eq_zero_iff.mp (by rw [leaves_size]; exact h0)
    rw [this]
    exact hash_size _
  rw [root_eq _ _ (leaves_isEmpty h0)]
  refine lev_width _ _ (leaves_width vs) _ _ (get!_mem ?_)
  rw [lev_size _ _ _ (Nat.le_refl _), Nat.sub_self]
  decide

/-! ## An opening climbs the device's tree -/

def climbRec (d : Domain) : Nat → Nat → ByteArray → List ByteArray → ByteArray
  | _, _, digest, [] => digest
  | level, cursor, digest, sib :: rest => climbRec d (level + 1) (cursor >>> 1)
      (if cursor % 2 = 0 then d.node level (cursor >>> 1) digest sib else d.node level (cursor >>> 1) sib digest) rest

theorem climbL_climbRec (d : Domain) : ∀ (sibs : List ByteArray) (l c : Nat) (dg : ByteArray),
    scheme.climbL d.id dg c l sibs = climbRec d l c dg sibs
  | [], _, _, _ => rfl
  | sib :: rest, l, c, dg => by
    have hc : c >>> 1 = c / 2 := by rw [Nat.shiftRight_eq_div_pow, Nat.pow_one]
    simp only [FV3.Scheme.climbL, climbRec, hc]
    exact climbL_climbRec d rest (l + 1) (c / 2) _

theorem climb_eq (d : Domain) (hd : d.start = 0) (pos : Nat) (schema : String) (v : ByteArray)
    (path : Array ByteArray) : d.climb pos schema v path = climbRec d 0 pos (d.leaf pos pos schema v) path.toList := by
  simp only [FV3.Domain.climb, hd, Nat.sub_zero]
  exact climbL_climbRec d path.toList 0 pos _

theorem scheme_bytes : scheme.bytes = 64 := rfl

/-- Under no collision among `S`, which holds the tree's node inputs and the climb's, a climb from `digest` at `level`
and `cursor` that reaches the tree's root started from the tree's node there. -/
theorem climb_bind {d : Domain} {L : Array ByteArray} {S : List ByteArray} (cf : CF S) (hw : ∀ x ∈ L, x.size = 64)
    (hnode : ∀ l i, l < log2ceil L.size → i < (lev d L l).size / 2 →
      FV3.input "node" [d.id, FV3.uint l, FV3.uint i, (lev d L l)[2 * i]!, (lev d L l)[2 * i + 1]!] ∈ S) :
    ∀ (rest : List ByteArray) (l c : Nat) (dg : ByteArray), l + rest.length = log2ceil L.size →
      c < 2 ^ (log2ceil L.size - l) → dg.size = 64 → (∀ x ∈ rest, x.size = 64) →
      (∀ x ∈ climbInputs d l c dg rest, x ∈ S) →
      climbRec d l c dg rest = (lev d L (log2ceil L.size))[0]! → dg = (lev d L l)[c]!
  | [], l, c, dg, hl, hc, _, _, _, hr => by
    simp only [List.length_nil, Nat.add_zero] at hl
    subst hl
    simp only [Nat.sub_self, Nat.pow_zero] at hc
    obtain rfl : c = 0 := by omega
    exact hr
  | sib :: rest, l, c, dg, hl, hc, hdg, hws, hin, hr => by
    simp only [List.length_cons] at hl
    have hlD : l < log2ceil L.size := by omega
    have hsz : (lev d L l).size = 2 ^ (log2ceil L.size - l) := lev_size d L l (by omega)
    have hk : log2ceil L.size - l = (log2ceil L.size - (l + 1)) + 1 := by omega
    have hp : 2 ^ (log2ceil L.size - l) = 2 ^ (log2ceil L.size - (l + 1)) * 2 := by rw [hk, Nat.pow_succ]
    have hcur : c >>> 1 = c / 2 := by rw [Nat.shiftRight_eq_div_pow, Nat.pow_one]
    have hc2 : c >>> 1 < 2 ^ (log2ceil L.size - (l + 1)) := by
      rw [hcur]
      omega
    have hi : c >>> 1 < (lev d L l).size / 2 := by
      rw [hsz, hp, Nat.mul_div_cancel _ (by decide)]
      exact hc2
    have hsib : sib.size = 64 := hws sib (List.mem_cons_self ..)
    have w := lev_width d L hw l
    have hl0 : ((lev d L l)[2 * (c >>> 1)]!).size = 64 := w _ (get!_mem (by omega))
    have hup := climb_bind cf hw hnode rest (l + 1) (c >>> 1) _ (by omega) hc2 (by split <;> exact hash_size _)
      (fun x hx => hws x (List.mem_cons_of_mem _ hx)) ?_ hr
    · simp only [lev] at hup
      rw [next_get _ _ _ hi, node_hash] at hup
      have hmem := hnode l (c >>> 1) hlD hi
      by_cases he : c % 2 = 0
      · simp only [climbInputs, he, ↓reduceIte, node_hash] at hup hin
        have e := cf _ (hin _ (List.mem_cons_self ..)) _ hmem hup
        obtain ⟨ea, -⟩ := frame_two (ps := [d.id, FV3.uint l, FV3.uint (c >>> 1)]) (by rw [hdg, hl0]) e
        rw [ea, hcur, show 2 * (c / 2) = c by omega]
      · simp only [climbInputs, he, ↓reduceIte, node_hash] at hup hin
        have e := cf _ (hin _ (List.mem_cons_self ..)) _ hmem hup
        obtain ⟨-, eb⟩ := frame_two (ps := [d.id, FV3.uint l, FV3.uint (c >>> 1)]) (by rw [hsib, hl0]) e
        rw [eb, hcur, show 2 * (c / 2) + 1 = c by omega]
    · intro x hx
      apply hin
      simp only [climbInputs]
      by_cases he : c % 2 = 0
      · simp only [he, ↓reduceIte, node_hash] at hx ⊢
        exact List.mem_cons_of_mem _ hx
      · simp only [he, ↓reduceIte, node_hash] at hx ⊢
        exact List.mem_cons_of_mem _ hx

/-- **An opening reads the device's record**: under no collision among `S`, which holds what the tree over `vs` hashes
and what the opening's climb hashes, an opening `verifyOpening` accepts at the tree's root reads the record at its
position. -/
theorem opening_bind {vs : Array ByteArray} {S : List ByteArray} {pos : Nat} {v : ByteArray} {path : Array ByteArray}
    (cf : CF S) (hT : ∀ x ∈ treeStrings vs, x ∈ S)
    (hC : ∀ x ∈ climbStrings (Attest.periodDomain vs.size) pos Attest.PERIOD_SCHEMA v path, x ∈ S)
    (hv : (Attest.periodDomain vs.size).verifyOpening (Attest.root vs) vs.size pos v Attest.PERIOD_SCHEMA path =
      true) :
    vs[pos]? = some v := by
  simp only [FV3.Domain.verifyOpening, scheme_bytes, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
    Array.all_eq_true] at hv
  obtain ⟨⟨⟨⟨⟨⟨-, -⟩, -⟩, hpos⟩, hlen⟩, hw⟩, hcl⟩ := hv
  have hdc : (Attest.periodDomain vs.size).count = vs.size := rfl
  have hds : (Attest.periodDomain vs.size).start = 0 := rfl
  rw [hds, hdc, Nat.zero_add] at hpos
  rw [hdc] at hlen
  have h0 : vs.size ≠ 0 := by omega
  have hL := leaves_size vs
  have hroot : Attest.root vs = (lev (Attest.periodDomain vs.size) (Attest.leaves vs) (log2ceil vs.size))[0]! := by
    unfold Attest.root
    rw [root_eq _ _ (leaves_isEmpty h0), hL]
  rw [climb_eq _ hds, hroot] at hcl
  have hleaf := climb_bind (d := Attest.periodDomain vs.size) (L := Attest.leaves vs) cf (leaves_width vs)
    (fun l i hl hi => hT _ (node_mem h0 (by rw [← hL]; exact hl) hi)) path.toList 0 pos _
    (by rw [hL, Array.length_toList, hlen]; omega)
    (by rw [hL, Nat.sub_zero]; exact Nat.lt_of_lt_of_le hpos (le_pow_log2ceil _))
    (hash_size _) (fun x hx => by
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
      simpa using hw i (by simpa using hi))
    (fun x hx => hC x (List.mem_cons_of_mem _ hx)) (by rw [hL]; exact hcl)
  simp only [lev, base] at hleaf
  rw [getElem!_pos _ pos (by rw [Array.size_append, hL]; omega), Array.getElem_append_left (by rw [hL]; exact hpos),
    ← getElem!_pos, leaves_get hpos, leaf_hash] at hleaf
  have e := frame_last (ps := [(Attest.periodDomain vs.size).id, FV3.uint pos, FV3.uint pos,
    Attest.PERIOD_SCHEMA.toUTF8]) (cf _ (hC _ (List.mem_cons_self ..)) _ (hT _ (leaf_mem hpos)) hleaf)
  rw [e, getElem?_pos vs pos hpos, getElem!_pos vs pos hpos]

/-! ## The period check -/

theorem verifyCertificate_ok {p : Lms.Params} {identity d lower : ByteArray} {party : String} {cert : Lms.Signature}
    (h : Lms.verifyCertificate p identity d party lower cert = .ok true) :
    identity.size = 16 + p.N ∧ ∃ cm, Lms.partyKeyMessage (identity.extract 0 16) d party lower = .ok cm ∧
      Lms.verify p identity cm cert = .ok true := by
  unfold Lms.verifyCertificate at h
  by_cases hi : identity.size = 16 + p.N
  · simp only [hi, ne_eq, not_true_eq_false, ↓reduceIte] at h
    split at h
    · simp [pure, Except.pure] at h
    cases hm : Lms.partyKeyMessage (identity.extract 0 16) d party lower with
    | error e => simp [hm, bind, Except.bind] at h
    | ok cm =>
      simp only [hm, bind, Except.bind] at h
      exact ⟨hi, cm, rfl, h⟩
  · simp [hi, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- What the verifier's acceptance of a period establishes: the lower key's length, the identity's, the certificate's
verification under the identity over the lower key's party-key message, the root signature's under the lower key over
the root's message, and each read's opening. -/
theorem accepts_ok {p : Lms.Params} {identity deployment : ByteArray} {party : String} {k : Attest.PartyKey}
    {s : Attest.SignedRoot} {reads : Array Attest.Read}
    (h : Attest.accepts p identity deployment party k s reads = .ok true) :
    k.lower.size = 16 + p.N ∧ identity.size = 16 + p.N ∧
      (∃ cm, Lms.partyKeyMessage (identity.extract 0 16) deployment party k.lower = .ok cm ∧
        Lms.verify p identity cm k.certificate = .ok true) ∧
      (∃ rm, s.message k.lower deployment party = .ok rm ∧ Lms.verify p k.lower rm s.signature = .ok true) ∧
      ∀ r ∈ reads, (Attest.periodDomain s.count).verifyOpening s.root s.count r.position r.record Attest.PERIOD_SCHEMA
        r.path = true := by
  unfold Attest.accepts Attest.checkKey at h
  cases hc : Lms.verifyCertificate p identity deployment party k.lower k.certificate with
  | error e => simp [hc, bind, Except.bind] at h
  | ok b =>
    cases b
    · simp [hc, bind, Except.bind, pure, Except.pure] at h
    simp only [hc, bind, Except.bind, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨hi, hcm⟩ := verifyCertificate_ok hc
    unfold Attest.check at h
    by_cases hl : k.lower.size = 16 + p.N
    · simp only [hl, ne_eq, not_true_eq_false, ↓reduceIte] at h
      cases hm : s.message k.lower deployment party with
      | error e => simp [hm, bind, Except.bind] at h
      | ok rm =>
        cases hv : Lms.verify p k.lower rm s.signature with
        | error e => simp [hm, hv, bind, Except.bind] at h
        | ok b =>
          cases b
          · simp [hm, hv, bind, Except.bind, pure, Except.pure] at h
          · simp only [hm, hv, bind, Except.bind, Bool.not_true, Bool.false_eq_true, ↓reduceIte, pure, Except.pure,
              Except.ok.injEq, Array.all_eq_true] at h
            exact ⟨hl, hi, hcm, ⟨rm, rfl, hv⟩, fun r hr => by
              obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hr
              exact h i hi⟩
    · simp [hl, pure, Except.pure] at h


/-! ## `PeriodSigned` -/

theorem periodSigned : PeriodSigned := by
  intro p identity deployment party k s reads dev h
  obtain ⟨-, -, ⟨cm, hcm, hcv⟩, ⟨rm, hrm, hrv⟩, hreads⟩ := accepts_ok h
  by_cases cf : CF (runStrings dev identity deployment party k s reads)
  · by_cases hsc : Signed dev identity cm
    · by_cases hsr : Signed dev k.lower rm
      · left
        have hK1 : ∀ x ∈ keyStrings dev identity, x ∈ runStrings dev identity deployment party k s reads :=
          fun x hx => by
            simp only [runStrings, List.mem_append]
            exact Or.inl (Or.inr hx)
        have hK2 : ∀ x ∈ keyStrings dev k.lower, x ∈ runStrings dev identity deployment party k s reads :=
          fun x hx => by
            simp only [runStrings, List.mem_append]
            exact Or.inr hx
        obtain ⟨i, hi, hid, hip, hil⟩ := signed_party cf hK1 hcm (fun x hx => by
          simp only [runStrings, List.mem_append]
          exact Or.inl (Or.inl (Or.inl (Or.inl hx)))) hsc
        obtain ⟨c, hc, hcd, hcp, hca, hcl, hcper, hcn, hcr⟩ := signed_root cf hK2 hrm (fun x hx => by
          simp only [runStrings, List.mem_append]
          exact Or.inl (Or.inl (Or.inl (Or.inr hx)))) hsr
        refine ⟨⟨i, hi, hid, hip, hil⟩, c, hc, hcd, hcp, hca, hcl, hcper, hcn, fun r hr => ?_⟩
        have hv := hreads r hr
        rw [← hcn, ← hcr] at hv
        refine opening_bind cf (fun x hx => hK2 x (closed_mem hc x (List.mem_append_right _ hx))) (fun x hx => ?_) hv
        simp only [runStrings, List.mem_append, List.mem_flatMap]
        exact Or.inl (Or.inl (Or.inr ⟨r, Array.mem_toList_iff.mpr hr, by rw [← hcn]; exact hx⟩))
      · exact Or.inr (Or.inr (Or.inr ⟨rm, hrm, hrv, hsr⟩))
    · exact Or.inr (Or.inr (Or.inl ⟨cm, hcm, hcv, hsc⟩))
  · refine Or.inr (Or.inl ?_)
    simp only [CF] at cf
    push Not at cf
    obtain ⟨x, hx, y, hy, hxy, hne⟩ := cf
    exact ⟨x, hx, y, hy, hne, hxy⟩

/-! ## `PeriodSound` -/

/-- `findColl` finds a collision whenever the list holds two different inputs with one output. -/
theorem findColl_isSome :
    ∀ l : List (Claims.Msg × Claims.Msg), (∃ q ∈ l, ∃ q' ∈ l, q.1 ≠ q'.1 ∧ q.2 = q'.2) → (Claims.findColl l).isSome := by
  intro l
  induction l with
  | nil => rintro ⟨q, hq, _⟩; simp at hq
  | cons hd tl ih =>
    rintro ⟨q, hq, q', hq', hne, heq⟩
    obtain ⟨u, a⟩ := hd
    simp only [Claims.findColl]
    cases hf : tl.find? (fun q => decide (q.1 ≠ u) && decide (q.2 = a)) with
    | some _ => rfl
    | none =>
      apply ih
      have hno : ∀ y ∈ tl, ¬ (y.1 ≠ u ∧ y.2 = a) := by
        intro y hy hy'
        have := List.find?_eq_none.1 hf y hy
        simp [hy'.1, hy'.2] at this
      rcases List.mem_cons.1 hq with rfl | hq <;> rcases List.mem_cons.1 hq' with rfl | hq'
      · exact absurd rfl hne
      · exact absurd ⟨fun h => hne h.symm, heq.symm⟩ (hno _ hq')
      · exact absurd ⟨hne, heq⟩ (hno _ hq)
      · exact ⟨q, hq, q', hq', hne, heq⟩

theorem sha512_bs (x : ByteArray) : sha512 x.data.toList = (Sha512.hash x).data.toList := by
  unfold sha512
  congr 3

/-- Where two distinct strings of `L` share a SHA-512 digest, `findColl` over their digests outputs a SHA-512
collision. -/
theorem finds {L : List ByteArray} (h : ∃ x ∈ L, ∃ y ∈ L, x ≠ y ∧ Sha512.hash x = Sha512.hash y) :
    ∃ u v, Claims.findColl (L.map fun x => (x.data.toList, (Sha512.hash x).data.toList)) = some (u, v) ∧
      Claims.Collision sha512 u v := by
  obtain ⟨x, hx, y, hy, hne, heq⟩ := h
  have hl : ∀ q ∈ L.map (fun x => (x.data.toList, (Sha512.hash x).data.toList)), sha512 q.1 = q.2 := by
    intro q hq
    obtain ⟨z, -, rfl⟩ := List.mem_map.1 hq
    dsimp only
    exact sha512_bs z
  have hs := findColl_isSome (L.map fun x => (x.data.toList, (Sha512.hash x).data.toList))
    ⟨(x.data.toList, (Sha512.hash x).data.toList), List.mem_map.2 ⟨x, hx, rfl⟩,
      (y.data.toList, (Sha512.hash y).data.toList), List.mem_map.2 ⟨y, hy, rfl⟩,
      fun e => hne (bs_inj e), by dsimp only; rw [heq]⟩
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hs
  exact ⟨u, v, huv, Claims.findColl_spec sha512 _ hl u v huv⟩

/-- On coins where the check accepts values that are not the device's, the collision finder outputs a collision or
the forger a forgery. -/
theorem bad_cases {p : Lms.Params} {Ω : Type} [Fintype Ω] [Nonempty Ω] (τ : Ω → PeriodRun) (ω : Ω)
    (ha : Attest.accepts p (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root (τ ω).reads = .ok true)
    (hn : ¬ OfDevice (τ ω).dev (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root (τ ω).reads) :
    (∃ u v, (collisionFinder Ω τ).out ω = some (u, v) ∧ Claims.Collision sha512 u v) ∨
      ∃ pk m σ, forger τ ω = some (pk, m, σ) ∧ Forges p (τ ω).dev pk m σ := by
  obtain ⟨-, -, ⟨cm, hcm, hcv⟩, -, -⟩ := accepts_ok ha
  rcases (periodSigned p _ _ _ _ _ _ (τ ω).dev ha).resolve_left hn with hc | ⟨m, hm, hf⟩ | ⟨m, hm, hf⟩
  · exact Or.inl (finds hc)
  · exact Or.inr ⟨_, _, _, by simp [forger, hm, hf.2], hf⟩
  · refine Or.inr ?_
    by_cases hs : Signed (τ ω).dev (τ ω).identity cm
    · exact ⟨_, _, _, by simp [forger, hcm, hs, hm, hf.2], hf⟩
    · exact ⟨_, _, _, by simp [forger, hcm, hs], hcv, hs⟩

theorem Pr_union_le {α : Type} [Fintype α] (p q : α → Prop) :
    Claims.Pr (fun a => p a ∨ q a) ≤ Claims.Pr p + Claims.Pr q := by
  classical
  unfold Claims.Pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  exact_mod_cast (Finset.card_le_card fun a ha => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at ha ⊢; exact ha).trans
    (Finset.card_union_le _ _)

theorem periodSound : PeriodSound := by
  intro p Ω _ _ τ
  refine (Claims.Pr_mono fun ω h => bad_cases τ ω h.1 h.2).trans ?_
  exact Pr_union_le _ _

/-! ## What the auditor and the prover see -/

theorem roleDigest_record {tag : String} {I c r r' m : ByteArray} (h : Lms.roleDigest tag I c r = .ok m)
    (hr : r'.size = r.size) : ∃ m', Lms.roleDigest tag I c r' = .ok m' := by
  obtain ⟨hI, hc, hrs, -⟩ := roleDigest_ok h
  refine ⟨Sha512.hash (roleInput tag I c r'), ?_⟩
  unfold Lms.roleDigest
  have hn : ¬(2 ^ 64 ≤ c.size ∨ 2 ^ 64 ≤ r'.size) := by omega
  simp only [hI, hn, ne_eq, not_true_eq_false, ↓reduceIte]
  rfl

/-- A period root's message is refused for one root exactly when it is for another of the same length. -/
theorem periodRootMessage_root {I d a root root' m : ByteArray} {p l : String} {per n : Nat}
    (h : Lms.periodRootMessage I d p a l per n root = .ok m) (hr : root'.size = root.size) :
    ∃ m', Lms.periodRootMessage I d p a l per n root' = .ok m' := by
  obtain ⟨c, hc, hd⟩ := periodRootMessage_ok h
  obtain ⟨m', hm'⟩ := roleDigest_record hd hr
  exact ⟨m', by simp only [Lms.periodRootMessage, hc, bind, Except.bind, hm']⟩

theorem inClear_close (x : ByteArray) (records : List ByteArray) :
    inClear ((Ch.root, x) :: records.map fun v => (Ch.record, v)) = [] := by
  simp [inClear, dest, List.filter_map, Function.comp_def]

/-- Two states that differ only in the open period's records' bytes send the same in the clear on inputs that differ
only in the records' bytes, and stay so. -/
theorem trace_clear {Ω : Type} (deployment : ByteArray) (party : String) (I : Ω → ByteArray)
    (hss : Ω → Nat → ByteArray → ByteArray) :
    ∀ (xs xs' : List Step) (st st' : Ω × Nat × List ByteArray), xs.map Step.skeleton = xs'.map Step.skeleton →
      st.1 = st'.1 → st.2.1 = st'.2.1 → st.2.2.length = st'.2.2.length →
      inClear ((code deployment party I hss).trace st xs).2 = inClear ((code deployment party I hss).trace st' xs').2
  | [], [], _, _, _, _, _, _ => rfl
  | [], _ :: _, _, _, h, _, _, _ => by simp at h
  | _ :: _, [], _, _, h, _, _, _ => by simp at h
  | x :: xs, x' :: xs', ⟨ω, q, rs⟩, ⟨ω', q', rs'⟩, h, h1, h2, h3 => by
    simp only [List.map_cons, List.cons.injEq] at h
    obtain ⟨hx, hxs⟩ := h
    simp only at h1 h2 h3
    subst h1 h2
    simp only [Parties.Code.trace, inClear, List.filter_append]
    cases x with
    | block a l per b recs =>
      cases x' with
      | close => simp [Step.skeleton] at hx
      | block a' l' per' b' recs' =>
        simp only [Step.skeleton, Step.block.injEq] at hx
        obtain ⟨rfl, rfl, rfl, rfl, hrecs⟩ := hx
        have hlen : recs.length = recs'.length := by simpa using congrArg List.length hrecs
        simp only [code]
        cases Lms.livenessMessage (I ω) deployment party a l per b with
        | ok m =>
          congr 1
          exact trace_clear deployment party I hss xs xs' _ _ hxs rfl rfl (by simp [h3, hlen])
        | error e =>
          congr 1
          exact trace_clear deployment party I hss xs xs' _ _ hxs rfl rfl (by simp [h3, hlen])
    | close a l per =>
      cases x' with
      | block => simp [Step.skeleton] at hx
      | close a' l' per' =>
        simp only [Step.skeleton, Step.close.injEq] at hx
        obtain ⟨rfl, rfl, rfl⟩ := hx
        simp only [code]
        have hw : (Attest.root rs'.toArray).size = (Attest.root rs.toArray).size := by rw [root_width, root_width]
        cases hm : Lms.periodRootMessage (I ω) deployment party a l per rs.length (Attest.root rs.toArray) with
        | ok m =>
          obtain ⟨m', hm'⟩ := periodRootMessage_root hm hw
          rw [← h3, hm']
          simp only
          rw [← inClear, ← inClear, ← inClear, ← inClear, inClear_close, inClear_close]
          exact congrArg _ (trace_clear deployment party I hss xs xs' _ _ hxs rfl rfl rfl)
        | error e =>
          cases hm' : Lms.periodRootMessage (I ω) deployment party a l per rs'.length (Attest.root rs'.toArray) with
          | ok m' =>
            have hm'' : Lms.periodRootMessage (I ω) deployment party a l per rs.length (Attest.root rs'.toArray) =
                .ok m' := by rw [h3]; exact hm'
            obtain ⟨m, hm2⟩ := periodRootMessage_root hm'' hw.symm
            simp [hm] at hm2
          | error e' =>
            simp only
            exact congrArg _ (trace_clear deployment party I hss xs xs' _ _ hxs rfl rfl h3)

theorem skeleton_skeleton (x : Step) : x.skeleton.skeleton = x.skeleton := by
  cases x <;> simp [Step.skeleton]

theorem livenessContentFree : LivenessContentFree := by
  intro Ω deployment party I hss ω xs
  refine trace_clear deployment party I hss xs _ _ _ ?_ rfl rfl rfl
  simp only [List.map_map, Function.comp_def, skeleton_skeleton]

theorem clear_map {Wi : Type} (wire : Ch × ByteArray → Wi) :
    ∀ L : List (Ch × ByteArray),
      ((L.map fun m => (dest m.1, wire m)).filter (·.1 = .clear)).map (·.2) = (inClear L).map wire
  | [] => rfl
  | m :: L => by
    have ih := clear_map wire L
    by_cases hm : dest m.1 = .clear
    · simp [inClear, hm] at ih ⊢
      exact ih
    · simp [inClear, hm] at ih ⊢
      exact ih

theorem auditorSeesSkeleton : AuditorSeesSkeleton := by
  intro Ω Wi A deployment party I hss wire r hf hc hw
  obtain ⟨ht, hs⟩ := hf
  have e : r.sent = (code deployment party I hss).messages r.coins r.truth := by
    rw [Parties.Code.messages, ← ht]; exact (congrArg Prod.snd hs)
  unfold Parties.auditorView
  rw [hc, hw, List.map_nil, List.append_nil, clear_map, e, livenessContentFree deployment party I hss]

theorem proverSeesMessages : ProverSeesMessages := by
  intro Ω Wi A deployment party I hss wire r hf hs hc
  obtain ⟨ht, hst⟩ := hf
  unfold Parties.proverView
  rw [hc, hs, List.map_map]
  have e : r.sent = (code deployment party I hss).messages r.coins r.truth := by
    rw [Parties.Code.messages, ← ht]; exact (congrArg Prod.snd hst)
  rw [e]
  rfl


/-! ## The honest period and the empty run -/

theorem roleDigest_some {tag : String} {I c r : ByteArray} (hI : I.size = 16) (hc : c.size < 2 ^ 64)
    (hr : r.size < 2 ^ 64) : Lms.roleDigest tag I c r = .ok (Sha512.hash (roleInput tag I c r)) := by
  unfold Lms.roleDigest
  have h : ¬ (2 ^ 64 ≤ c.size ∨ 2 ^ 64 ≤ r.size) := by omega
  simp only [hI, ne_eq, not_true_eq_false, ↓reduceIte, h]
  rfl

theorem partyContext_some {d : ByteArray} {p : String} (hd : d.size = 64) :
    Lms.partyContext d p = .ok (Lms.lenField d ++ Lms.lenField p.toUTF8) := by
  unfold Lms.partyContext
  simp only [hd, Lms.DEPLOYMENT_BYTES, ne_eq, not_true_eq_false, ↓reduceIte]
  rfl

theorem u64_some {x : Nat} {w : String} (h : x < 2 ^ 64) : Lms.u64 x w = .ok (be64 x.toUInt64) := by
  unfold Lms.u64
  simp only [h, ↓reduceIte]
  rfl

theorem periodContext_some {d a : ByteArray} {p l : String} {per : Nat} (hd : d.size = 64) (hper : per < 2 ^ 64) :
    Lms.periodContext d p a l per =
      .ok (Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++ Lms.lenField l.toUTF8 ++ be64 per.toUInt64) := by
  unfold Lms.periodContext
  rw [partyContext_some hd, u64_some hper]
  rfl

theorem periodRootContext_some {d a : ByteArray} {p l : String} {per n : Nat} (hd : d.size = 64)
    (hper : per < 2 ^ 64) (hn : n < 2 ^ 64) :
    Lms.periodRootContext d p a l per n = .ok (Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++
      Lms.lenField l.toUTF8 ++ be64 per.toUInt64 ++ be64 n.toUInt64) := by
  unfold Lms.periodRootContext
  rw [periodContext_some hd hper, u64_some hn]
  rfl

theorem livenessContext_some {d a : ByteArray} {p l : String} {per b : Nat} (hd : d.size = 64)
    (hper : per < 2 ^ 64) (hb : b < 2 ^ 64) :
    Lms.livenessContext d p a l per b = .ok (Lms.lenField d ++ Lms.lenField p.toUTF8 ++ Lms.lenField a ++
      Lms.lenField l.toUTF8 ++ be64 per.toUInt64 ++ be64 b.toUInt64) := by
  unfold Lms.livenessContext
  rw [periodContext_some hd hper, u64_some hb]
  rfl

theorem deployment_size : HONEST_DEPLOYMENT.size = 64 := rfl

theorem honest_small : (Lms.lenField HONEST_DEPLOYMENT ++ Lms.lenField HONEST_PARTY.toUTF8 ++
    Lms.lenField HONEST_ATTACHMENT ++ Lms.lenField HONEST_LINK.toUTF8 ++ be64 (0 : Nat).toUInt64 ++
      be64 (1 : Nat).toUInt64).size < 2 ^ 64 ∧
    (Lms.lenField HONEST_DEPLOYMENT ++ Lms.lenField HONEST_PARTY.toUTF8 ++ Lms.lenField HONEST_ATTACHMENT ++
      Lms.lenField HONEST_LINK.toUTF8 ++ be64 (0 : Nat).toUInt64 ++ be64 (0 : Nat).toUInt64).size < 2 ^ 64 ∧
    (Lms.lenField HONEST_DEPLOYMENT ++ Lms.lenField HONEST_PARTY.toUTF8).size < 2 ^ 64 := by
  simp only [ByteArray.size_append, lenField_size, be64_size]
  have hp : HONEST_PARTY.toUTF8.size = 17 := by decide
  have hl : HONEST_LINK.toUTF8.size = 8 := by decide
  have ha : HONEST_ATTACHMENT.size = 32 := rfl
  rw [hp, hl, ha, deployment_size]
  decide

/-- The honest party-key message: what an identity with identifier `I` signs to certify the lower key `lower` for the
memory challenger in the honest deployment. -/
theorem honest_partyKeyMessage {I lower : ByteArray} (hI : I.size = 16) (hl : lower.size = 80) :
    ∃ cm, Lms.partyKeyMessage I HONEST_DEPLOYMENT HONEST_PARTY lower = .ok cm := by
  unfold Lms.partyKeyMessage
  rw [partyContext_some deployment_size]
  exact ⟨_, roleDigest_some hI honest_small.2.2 (by omega)⟩

/-- The honest period-root message: what a lower key with identifier `I` signs to close the honest one-record period
whose record is `v`. -/
theorem honest_periodRootMessage {I : ByteArray} (hI : I.size = 16) (v : ByteArray) :
    ∃ rm, Lms.periodRootMessage I HONEST_DEPLOYMENT HONEST_PARTY HONEST_ATTACHMENT HONEST_LINK 0 1
      (Attest.root #[v]) = .ok rm := by
  unfold Lms.periodRootMessage
  rw [periodRootContext_some deployment_size (by decide) (by decide)]
  exact ⟨_, roleDigest_some hI honest_small.1 (by rw [root_width]; decide)⟩

/-- The honest liveness message: what a lower key with identifier `I` signs for sampling block 0 of the honest period. -/
theorem honest_livenessMessage {I : ByteArray} (hI : I.size = 16) :
    ∃ m, Lms.livenessMessage I HONEST_DEPLOYMENT HONEST_PARTY HONEST_ATTACHMENT HONEST_LINK 0 0 = .ok m := by
  unfold Lms.livenessMessage
  rw [livenessContext_some deployment_size (by decide) (by decide)]
  exact ⟨_, roleDigest_some hI honest_small.2.1 (by decide)⟩

/-- A one-record period's root is its record's leaf. -/
theorem root_one (v : ByteArray) :
    Attest.root #[v] = (Attest.periodDomain 1).leaf 0 0 Attest.PERIOD_SCHEMA v := by
  unfold Attest.root
  rw [root_eq _ _ (leaves_isEmpty (by simp))]
  simp [lev, base, Attest.leaves, nextPow2, log2ceil]

/-- A one-record period's record opens at position 0 with no siblings. -/
theorem opening_one (v : ByteArray) :
    (Attest.periodDomain 1).verifyOpening (Attest.root #[v]) 1 0 v Attest.PERIOD_SCHEMA #[] = true := by
  have hc : (Attest.periodDomain 1).climb 0 Attest.PERIOD_SCHEMA v #[] =
      (Attest.periodDomain 1).leaf 0 0 Attest.PERIOD_SCHEMA v := by
    rw [climb_eq (Attest.periodDomain 1) rfl]
    rfl
  have hw := root_width #[v]
  rw [root_one] at hw
  simp only [FV3.Domain.verifyOpening, hc, root_one, hw, scheme_bytes]
  simp [Attest.periodDomain, Attest.domainU64, log2ceil]

/-- What `Lms.key` gives at the deployed parameters: a 16-byte identifier, and a public key of `16 + N` bytes that
starts with it. -/
theorem pub_spec {I seed : ByteArray} {k : Lms.PrivateKey} (hk : Lms.key Lms.deployed I seed = .ok k) :
    I.size = 16 ∧ k.publicKey.size = 16 + Lms.deployed.N ∧ k.publicKey.extract 0 16 = I := by
  obtain ⟨-, hIk, -, hIs, -, -, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hk
  have hroot : (k.tree[1]!).size = Lms.deployed.N :=
    Properties.CertifierDevice.Proofs.Lms.node_size hk (by decide) 1 le_rfl (by decide)
  subst hIk
  exact ⟨hIs, by simp only [Lms.PrivateKey.publicKey, ByteArray.size_append, hIs, hroot],
    ByteArray.extract_append_eq_left hIs.symm⟩

/-- A key `Lms.key` derives at the deployed parameters signs every message at leaf 0 (`Lms.signAt`), and the signature
verifies under its public key (`Properties.CertifierDevice.Proofs.Lms.lmsCorrect`). -/
theorem signs {I seed : ByteArray} {k : Lms.PrivateKey} (hk : Lms.key Lms.deployed I seed = .ok k) (m : ByteArray) :
    ∃ σ, Lms.signAt k 0 m = .ok σ ∧ Lms.verify Lms.deployed k.publicKey m σ = .ok true := by
  obtain ⟨σ, hσ⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsSigns 64 1 20 false Lms.deployed I seed k 0 m rfl hk (by decide)
  exact ⟨σ, hσ, (Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed I seed k 0 m σ rfl hk hσ).1⟩

/-- **The verifier accepts the honest period**: for keys `Lms.key` derives at the deployed parameters, the identity `k`
certifying the lower key `k'` for the memory challenger at leaf 0 and `k'` signing the one-record period's root at
leaf 0, the verifier accepts the period with its one read, which is the device's. -/
theorem honest_accepts {I seed I' seed' : ByteArray} {k k' : Lms.PrivateKey}
    (hk : Lms.key Lms.deployed I seed = .ok k) (hk' : Lms.key Lms.deployed I' seed' = .ok k') (v : ByteArray) :
    ∃ cm rm cert σ, Lms.partyKeyMessage I HONEST_DEPLOYMENT HONEST_PARTY k'.publicKey = .ok cm ∧
      Lms.signAt k 0 cm = .ok cert ∧ (honestRoot σ v).message k'.publicKey HONEST_DEPLOYMENT HONEST_PARTY = .ok rm ∧
      Lms.signAt k' 0 rm = .ok σ ∧
      Attest.accepts Lms.deployed k.publicKey HONEST_DEPLOYMENT HONEST_PARTY (honestKey k'.publicKey cert)
        (honestRoot σ v) #[honestRead v] = .ok true ∧
      OfDevice (honestDevice k.publicKey k'.publicKey v) k.publicKey HONEST_DEPLOYMENT HONEST_PARTY
        (honestKey k'.publicKey cert) (honestRoot σ v) #[honestRead v] := by
  obtain ⟨hIs, hps, hpI⟩ := pub_spec hk
  obtain ⟨hIs', hps', hpI'⟩ := pub_spec hk'
  obtain ⟨cm, hcm⟩ := honest_partyKeyMessage hIs hps'
  obtain ⟨rm, hrm⟩ := honest_periodRootMessage hIs' v
  obtain ⟨cert, hcs, hcv⟩ := signs hk cm
  obtain ⟨σ, hσs, hσv⟩ := signs hk' rm
  have hmsg : (honestRoot σ v).message k'.publicKey HONEST_DEPLOYMENT HONEST_PARTY = .ok rm := by
    unfold Attest.SignedRoot.message
    simp only [honestRoot, hpI']
    exact hrm
  refine ⟨cm, rm, cert, σ, hcm, hcs, hmsg, hσs, ?_, ?_⟩
  · have hcert : Lms.verifyCertificate Lms.deployed k.publicKey HONEST_DEPLOYMENT HONEST_PARTY k'.publicKey cert =
        .ok true := by
      unfold Lms.verifyCertificate
      simp only [hps, hps', ne_eq, not_true_eq_false, ↓reduceIte, hpI, hcm]
      exact hcv
    unfold Attest.accepts Attest.checkKey Attest.check
    simp only [honestKey, hcert, hmsg, hps', ne_eq, not_true_eq_false, ↓reduceIte, bind, Except.bind,
      Bool.not_true, Bool.false_eq_true]
    simp only [honestRoot, hσv, Bool.not_true, Bool.false_eq_true, ↓reduceIte, honestRead]
    simp [opening_one]
    rfl
  · refine ⟨⟨⟨HONEST_DEPLOYMENT, HONEST_PARTY, k'.publicKey⟩, by simp [honestDevice], rfl, rfl, rfl⟩,
      ⟨HONEST_DEPLOYMENT, HONEST_PARTY, HONEST_ATTACHMENT, HONEST_LINK, 0, #[v]⟩, by simp [honestDevice, honestKey], rfl,
      rfl, rfl, rfl, rfl, rfl, ?_⟩
    intro r hr
    simp [honestRead] at hr
    subst hr
    rfl

/-- **`PeriodSigned`'s honest lemma**: its hypothesis holds at the honest deployed instance, and with it its
conclusion's first disjunct. For keys `Lms.key` derives at the deployed parameters (`Lms.deployed`, SHA-512 with
`n = 64`, `w = 1`, `h = 20`), the identity certifying the lower key for the memory challenger in a deployment at leaf 0
and the lower key signing the one-record period of `HONEST_RECORD`'s root at leaf 0 (`Lms.signAt`), the verifier
accepts the period with its one read, and the period and the read are the device's. -/
theorem periodSigned_honest :
    ∃ (k k' : Lms.PrivateKey) (cert σ : Lms.Signature),
      (∃ I seed, Lms.key Lms.deployed I seed = .ok k) ∧ (∃ I seed, Lms.key Lms.deployed I seed = .ok k') ∧
      Attest.accepts Lms.deployed k.publicKey HONEST_DEPLOYMENT HONEST_PARTY (honestKey k'.publicKey cert)
        (honestRoot σ HONEST_RECORD) #[honestRead HONEST_RECORD] = .ok true ∧
      OfDevice (honestDevice k.publicKey k'.publicKey HONEST_RECORD) k.publicKey HONEST_DEPLOYMENT HONEST_PARTY
        (honestKey k'.publicKey cert) (honestRoot σ HONEST_RECORD) #[honestRead HONEST_RECORD] := by
  obtain ⟨k, hk⟩ := Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed
    (I := ⟨Array.replicate 16 0⟩) rfl HONEST_DEPLOYMENT
  obtain ⟨k', hk'⟩ := Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed
    (I := ⟨Array.replicate 16 1⟩) rfl HONEST_DEPLOYMENT
  obtain ⟨-, -, cert, σ, -, -, -, -, hc, ho⟩ := honest_accepts hk hk' HONEST_RECORD
  exact ⟨k, k', cert, σ, ⟨_, _, hk⟩, ⟨_, _, hk'⟩, hc, ho⟩

/-- The empty run hashes nothing: its deployment hash has no bytes, so neither of its messages is formed. -/
theorem empty_strings : runStrings emptyDevice .empty .empty "" emptyKey emptyRoot #[] = [] := by
  have h1 : Lms.partyContext ByteArray.empty "" = .error
      s!"a deployment's hash is {Lms.DEPLOYMENT_BYTES} bytes, not {ByteArray.empty.size}" := rfl
  have h2 : ∃ e, Lms.periodRootContext ByteArray.empty "" ByteArray.empty "" 0 0 = .error e := ⟨_, rfl⟩
  obtain ⟨e, h2⟩ := h2
  simp only [runStrings, roleStrings, h1, h2, keyStrings, emptyDevice, emptyKey, emptyRoot]
  rfl

theorem empty_partyKeyMessage (I : ByteArray) :
    ∀ m, Lms.partyKeyMessage I .empty "" .empty ≠ .ok m := by
  intro m h
  cases h

theorem empty_message : ∀ m, emptyRoot.message .empty .empty "" ≠ .ok m := by
  intro m h
  cases h

/-- `PeriodSigned`'s conclusion fails at the empty run, for any parameters: the device signed nothing, so the run is
not its, and the run hashes no string and forms no message, so it carries no break. -/
theorem empty_not_signed (p : Lms.Params) :
    ¬ (OfDevice emptyDevice .empty .empty "" emptyKey emptyRoot #[] ∨
      Break p emptyDevice .empty .empty "" emptyKey emptyRoot #[]) := by
  rintro (⟨⟨i, hi, -⟩, -⟩ | ⟨x, hx, -⟩ | ⟨m, hm, -⟩ | ⟨m, hm, -⟩)
  · simp [emptyDevice] at hi
  · rw [empty_strings] at hx
    cases hx
  · exact empty_partyKeyMessage _ m hm
  · exact empty_message m hm

/-- **`PeriodSigned`'s nontrivial lemma**: its conclusion fails at the empty run, at the deployed LMS parameters
(`empty_not_signed`), so a verifier that accepts the empty run, as one that accepts everything does
(`periodSigned_acceptAll`), does not have `PeriodSigned`'s conclusion. -/
theorem periodSigned_nontrivial :
    ¬ (OfDevice emptyDevice .empty .empty "" emptyKey emptyRoot #[] ∨
      Break Lms.deployed emptyDevice .empty .empty "" emptyKey emptyRoot #[]) :=
  empty_not_signed Lms.deployed

/-- **`PeriodSigned` fails at a verifier that accepts everything**: such a verifier accepts the empty run, where the
conclusion fails (`empty_not_signed`). -/
theorem periodSigned_acceptAll (p : Lms.Params)
    (accepts : ByteArray → ByteArray → String → Attest.PartyKey → Attest.SignedRoot → Array Attest.Read → Lms.E Bool)
    (hall : ∀ identity deployment party k s reads, accepts identity deployment party k s reads = .ok true) :
    ¬ ∀ (identity deployment : ByteArray) (party : String) (k : Attest.PartyKey) (s : Attest.SignedRoot)
      (reads : Array Attest.Read) (dev : Device), accepts identity deployment party k s reads = .ok true →
        OfDevice dev identity deployment party k s reads ∨ Break p dev identity deployment party k s reads :=
  fun h => empty_not_signed p (h _ _ _ _ _ _ _ (hall _ _ _ _ _ _))

/-! ### `PeriodSound`'s lemmas -/

theorem Pr_unit_true (q : Unit → Prop) (h : q ()) : Claims.Pr q = 1 := by
  classical
  unfold Claims.Pr
  have : Finset.univ.filter q = Finset.univ := Finset.filter_true_of_mem fun u _ => by cases u; exact h
  rw [this]
  simp

theorem Pr_unit_false (q : Unit → Prop) (h : ¬ q ()) : Claims.Pr q = 0 := by
  classical
  unfold Claims.Pr
  have : Finset.univ.filter q = ∅ := Finset.filter_false_of_mem fun u _ => by cases u; exact h
  rw [this]
  simp

/-- The honest run: the honest period's check on the honest device's period, read through an extractor that reads
the honest values. -/
def honestRun (k k' : Lms.PrivateKey) (cert σ : Lms.Signature) (v : ByteArray) : PeriodRun :=
  ⟨honestDevice k.publicKey k'.publicKey v, k.publicKey, HONEST_DEPLOYMENT, HONEST_PARTY, honestKey k'.publicKey cert,
    honestRoot σ v, #[honestRead v]⟩

/-- **`PeriodSound`'s honest lemma**: at the honest deployed run (`periodSigned_honest`), the verifier accepts with
probability 1 and the bound's event, an accepted run that is not the device's, has probability 0. -/
theorem periodSound_honest :
    ∃ τ : Unit → PeriodRun,
      Claims.Pr (fun ω => Attest.accepts Lms.deployed (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root
        (τ ω).reads = .ok true) = 1 ∧
      Claims.Pr (fun ω => Attest.accepts Lms.deployed (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root
          (τ ω).reads = .ok true ∧
        ¬ OfDevice (τ ω).dev (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root (τ ω).reads) = 0 := by
  obtain ⟨k, k', cert, σ, -, -, hc, ho⟩ := periodSigned_honest
  exact ⟨fun _ => honestRun k k' cert σ HONEST_RECORD, Pr_unit_true _ hc, Pr_unit_false _ fun h => h.2 ho⟩

theorem empty_finder : (collisionFinder Unit emptyRun).Pr sha512 = 0 := by
  unfold Claims.Finder.Pr
  refine Pr_unit_false _ ?_
  rintro ⟨u, w, h, -⟩
  simp only [collisionFinder, emptyRun, empty_strings, List.map_nil, Claims.findColl] at h
  cases h

theorem empty_forger (p : Lms.Params) : forgeryPr p emptyRun = 0 := by
  unfold forgeryPr
  refine Pr_unit_false _ ?_
  rintro ⟨pk, m, σ, h, -⟩
  have h1 : Lms.partyKeyMessage (ByteArray.empty.extract 0 16) .empty "" .empty = .error
      s!"a deployment's hash is {Lms.DEPLOYMENT_BYTES} bytes, not {ByteArray.empty.size}" := rfl
  obtain ⟨e, h2⟩ : ∃ e, emptyRoot.message .empty .empty "" = .error e := ⟨_, rfl⟩
  simp only [forger, emptyRun, emptyKey, emptyRoot] at h
  simp only [emptyRoot] at h2
  rw [h1, h2] at h
  cases h

/-- At the empty run, for any parameters, `PeriodSound`'s right side is 0 (the collision finder outputs nothing, the
forger nothing) and the run is not the device's: a verifier that accepted it would put probability 1 on the left. -/
theorem empty_bound (p : Lms.Params) :
    (collisionFinder Unit emptyRun).Pr sha512 + forgeryPr p emptyRun = 0 ∧
      ¬ OfDevice emptyDevice .empty .empty "" emptyKey emptyRoot #[] := by
  refine ⟨by rw [empty_finder, empty_forger, add_zero], fun h => empty_not_signed p (Or.inl h)⟩

/-- **`PeriodSound`'s nontrivial lemma**: its bound fails at the empty run, at the deployed LMS parameters, for the
verifier that accepts everything (`acceptAll`): its event, an accepted run that is not the device's, has probability
1, and the bound's right side is 0 (`empty_bound`). -/
theorem periodSound_nontrivial :
    ¬ Claims.Pr (fun ω =>
        acceptAll (emptyRun ω).identity (emptyRun ω).deployment (emptyRun ω).party (emptyRun ω).key
          (emptyRun ω).root (emptyRun ω).reads = .ok true ∧
        ¬ OfDevice (emptyRun ω).dev (emptyRun ω).identity (emptyRun ω).deployment (emptyRun ω).party
          (emptyRun ω).key (emptyRun ω).root (emptyRun ω).reads) ≤
      (collisionFinder Unit emptyRun).Pr sha512 + forgeryPr Lms.deployed emptyRun := by
  obtain ⟨h0, hn⟩ := empty_bound Lms.deployed
  rw [h0, not_le, Pr_unit_true]
  · norm_num
  · exact ⟨rfl, hn⟩

/-- **`PeriodSound` fails at a verifier that accepts everything**: on the empty run, such a verifier's bad event has
probability 1 and the bound's right side is 0 (`empty_bound`). -/
theorem periodSound_acceptAll (p : Lms.Params)
    (accepts : ByteArray → ByteArray → String → Attest.PartyKey → Attest.SignedRoot → Array Attest.Read → Lms.E Bool)
    (hall : ∀ identity deployment party k s reads, accepts identity deployment party k s reads = .ok true) :
    ¬ ∀ (Ω : Type) [Fintype Ω] [Inhabited Ω] (τ : Ω → PeriodRun),
      Claims.Pr (fun ω => accepts (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root (τ ω).reads = .ok true ∧
          ¬ OfDevice (τ ω).dev (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root (τ ω).reads) ≤
        (collisionFinder Ω τ).Pr sha512 + forgeryPr p τ := by
  intro h
  have hb := h Unit emptyRun
  obtain ⟨h0, hn⟩ := empty_bound p
  rw [h0] at hb
  refine absurd hb (not_le.2 ?_)
  rw [Pr_unit_true]
  · norm_num
  · exact ⟨hall _ _ _ _ _ _, hn⟩

/-! ### What the auditor and the prover see: the lemmas -/

/-- The honest identifier of a lower key, 16 bytes. -/
def HONEST_I : ByteArray := ⟨Array.replicate 16 1⟩

/-- The party's code at the honest deployment, under the honest lower key. -/
def honestCode (hss : Unit → Nat → ByteArray → ByteArray) :=
  code HONEST_DEPLOYMENT HONEST_PARTY (fun _ => HONEST_I) hss

/-- A sampling block of the honest period with the one record `v`, then the period's close. -/
def honestSteps (v : ByteArray) : List Step :=
  [.block HONEST_ATTACHMENT HONEST_LINK 0 0 [v], .close HONEST_ATTACHMENT HONEST_LINK 0]

theorem honestSteps_skeleton (v : ByteArray) : (honestSteps v).map Step.skeleton = honestSteps .empty := rfl

/-- What the honest code sends on `honestSteps v`: the block's liveness record in the clear, then the root's signature
and the record to the prover. -/
theorem honest_messages (hss : Unit → Nat → ByteArray → ByteArray) (v : ByteArray) {m rm : ByteArray}
    (hm : Lms.livenessMessage HONEST_I HONEST_DEPLOYMENT HONEST_PARTY HONEST_ATTACHMENT HONEST_LINK 0 0 = .ok m)
    (hrm : Lms.periodRootMessage HONEST_I HONEST_DEPLOYMENT HONEST_PARTY HONEST_ATTACHMENT HONEST_LINK 0 1
      (Attest.root #[v]) = .ok rm) :
    (honestCode hss).messages () (honestSteps v) =
      [(Ch.liveness, hss () 0 m), (Ch.root, hss () 1 rm), (Ch.record, v)] := by
  simp only [honestCode, Parties.Code.messages, honestSteps, Parties.Code.trace, code, hm, List.nil_append,
    List.length_singleton]
  simp only [hrm]
  rfl

theorem honest_messages_ex (hss : Unit → Nat → ByteArray → ByteArray) :
    ∃ m, ∀ v, ∃ rm, (honestCode hss).messages () (honestSteps v) =
      [(Ch.liveness, hss () 0 m), (Ch.root, hss () 1 rm), (Ch.record, v)] := by
  obtain ⟨m, hm⟩ := honest_livenessMessage (I := HONEST_I) rfl
  refine ⟨m, fun v => ?_⟩
  obtain ⟨rm, hrm⟩ := honest_periodRootMessage (I := HONEST_I) rfl v
  exact ⟨rm, honest_messages hss v hm hrm⟩

theorem empty_ne_zero : ByteArray.empty ≠ ⟨#[0]⟩ := fun e => absurd (congrArg ByteArray.size e) (by decide)

/-- **`LivenessContentFree`'s honest lemma**: the party does publish: on a sampling block of the honest period, its
code sends the block's liveness record in the clear. -/
theorem livenessContentFree_honest (hss : Unit → Nat → ByteArray → ByteArray) (v : ByteArray) :
    ∃ m, inClear ((honestCode hss).messages () (honestSteps v)) = [(Ch.liveness, hss () 0 m)] := by
  obtain ⟨m, h⟩ := honest_messages_ex hss
  obtain ⟨rm, h⟩ := h v
  refine ⟨m, ?_⟩
  rw [h]
  rfl

/-- **`LivenessContentFree`'s nontrivial lemma**: the party's whole output carries the records: on the honest period
whose one record is a byte, what its code sends differs from what it sends on the skeleton, so a code that sent its
records in the clear would fail the conclusion. -/
theorem livenessContentFree_nontrivial (hss : Unit → Nat → ByteArray → ByteArray) :
    ∃ xs : List Step, (honestCode hss).messages () xs ≠ (honestCode hss).messages () (xs.map Step.skeleton) := by
  refine ⟨honestSteps ⟨#[0]⟩, ?_⟩
  rw [honestSteps_skeleton]
  obtain ⟨m, h⟩ := honest_messages_ex hss
  obtain ⟨rm, h1⟩ := h ⟨#[0]⟩
  obtain ⟨rm', h2⟩ := h .empty
  rw [h1, h2]
  intro e
  simp only [List.cons.injEq, Prod.mk.injEq, and_true, true_and] at e
  exact empty_ne_zero e.2.symm

/-- The run of a party's code `c` as it is meant to run: on its true inputs `xs` from its coins `ω`, emitting each
message as `wire` puts it on the wire to where its channel goes, with no access to its secret state and nothing
overheard. -/
def meantRun {Ω S I C M Wi A : Type} (c : Parties.Code Ω S I C M) (dst : C → Parties.Dest) (wire : C × M → Wi)
    (ω : Ω) (xs : List I) : Parties.Run Ω S I C M Wi A :=
  let t := c.trace (c.init ω) xs
  ⟨ω, xs, xs, t.1, t.2, t.2.map fun m => (dst m.1, wire m), [], []⟩

theorem meantRun_ok {Ω S I C M Wi A : Type} (c : Parties.Code Ω S I C M) (dst : C → Parties.Dest)
    (wire : C × M → Wi) (ω : Ω) (xs : List I) :
    Parties.Faithful c (meantRun (A := A) c dst wire ω xs) ∧ Parties.Secret (meantRun (A := A) c dst wire ω xs) ∧
      Parties.Confined dst wire (meantRun (A := A) c dst wire ω xs) ∧
      Parties.WitnessPathPrivate (meantRun (A := A) c dst wire ω xs) :=
  ⟨⟨rfl, rfl⟩, rfl, rfl, rfl⟩

/-- **`AuditorSeesSkeleton`'s honest lemma**: its hypotheses hold at the honest party as it is meant to run, on the
honest period whose one record is a byte, whose true inputs are not their skeleton. -/
theorem auditorSeesSkeleton_honest {Wi A : Type} (hss : Unit → Nat → ByteArray → ByteArray)
    (wire : Ch × ByteArray → Wi) :
    ∃ r : Parties.Run Unit (Unit × Nat × List ByteArray) Step Ch ByteArray Wi A,
      Parties.Faithful (honestCode hss) r ∧ Parties.Confined dest wire r ∧ Parties.WitnessPathPrivate r ∧
      r.truth.map Step.skeleton ≠ r.truth := by
  obtain ⟨h1, -, h2, h3⟩ := meantRun_ok (A := A) (honestCode hss) dest wire () (honestSteps ⟨#[0]⟩)
  refine ⟨_, h1, h2, h3, ?_⟩
  show (honestSteps ⟨#[0]⟩).map Step.skeleton ≠ honestSteps ⟨#[0]⟩
  rw [honestSteps_skeleton]
  intro e
  simp only [honestSteps, List.cons.injEq, Step.block.injEq, and_true, true_and] at e
  exact empty_ne_zero e

/-- A run of the honest party as it is meant to run on the honest period of `v`, but for an observer on its path to
the prover who overhears the record. -/
def overheardRun (hss : Unit → Nat → ByteArray → ByteArray) (v : ByteArray) :
    Parties.Run Unit (Unit × Nat × List ByteArray) Step Ch ByteArray (Ch × ByteArray) Unit :=
  { meantRun (honestCode hss) dest id () (honestSteps v) with overheard := [(Ch.record, v)] }

/-- **`AuditorSeesSkeleton`'s nontrivial lemma**: without `WitnessPathPrivate` its conclusion fails: the honest party,
`Faithful` and `Confined`, on the honest period whose one record is a byte, where an observer on the path to the prover
overhears the record, shows the auditor the record beside what its code sends in the clear on the skeleton. -/
theorem auditorSeesSkeleton_nontrivial (hss : Unit → Nat → ByteArray → ByteArray) :
    ¬ ∀ r : Parties.Run Unit (Unit × Nat × List ByteArray) Step Ch ByteArray (Ch × ByteArray) Unit,
      Parties.Faithful (honestCode hss) r → Parties.Confined dest id r →
        Parties.auditorView id r =
          (inClear ((honestCode hss).messages r.coins (r.truth.map Step.skeleton))).map id := by
  intro h
  have hok := meantRun_ok (A := Unit) (honestCode hss) dest id () (honestSteps ⟨#[0]⟩)
  have e := h (overheardRun hss ⟨#[0]⟩) hok.1 hok.2.2.1
  obtain ⟨m, hm⟩ := honest_messages_ex hss
  obtain ⟨rm, h1⟩ := hm ⟨#[0]⟩
  obtain ⟨rm', h2⟩ := hm .empty
  have hl : Parties.auditorView id (overheardRun hss ⟨#[0]⟩) = [(Ch.liveness, hss () 0 m), (Ch.record, ⟨#[0]⟩)] := by
    show ((((honestCode hss).messages () (honestSteps ⟨#[0]⟩)).map fun x => (dest x.1, id x)).filter (·.1 = .clear)).map
      (·.2) ++ [(Ch.record, ⟨#[0]⟩)].map id = _
    rw [h1]
    rfl
  have e' : [(Ch.liveness, hss () 0 m), (Ch.record, (⟨#[0]⟩ : ByteArray))] =
      (inClear ((honestCode hss).messages () (honestSteps .empty))).map id := by
    rw [← hl]
    exact e
  rw [h2] at e'
  exact absurd (congrArg List.length e') (by simp [inClear, dest])

/-- **`ProverSeesMessages`'s honest lemma**: its hypotheses hold at the honest party as it is meant to run, and what
the prover then gets is the period's root signature and its record. -/
theorem proverSeesMessages_honest {Wi A : Type} (hss : Unit → Nat → ByteArray → ByteArray)
    (wire : Ch × ByteArray → Wi) (v : ByteArray) :
    ∃ r : Parties.Run Unit (Unit × Nat × List ByteArray) Step Ch ByteArray Wi A,
      Parties.Faithful (honestCode hss) r ∧ Parties.Secret r ∧ Parties.Confined dest wire r ∧
      ∃ m rm, Parties.proverView r = ([wire (Ch.liveness, hss () 0 m), wire (Ch.root, hss () 1 rm),
        wire (Ch.record, v)], []) := by
  obtain ⟨h1, h2, h3, -⟩ := meantRun_ok (A := A) (honestCode hss) dest wire () (honestSteps v)
  obtain ⟨m, hm⟩ := honest_messages_ex hss
  obtain ⟨rm, h⟩ := hm v
  refine ⟨_, h1, h2, h3, m, rm, ?_⟩
  show ((((honestCode hss).messages () (honestSteps v)).map fun x => (dest x.1, wire x)).map (·.2), []) = _
  rw [h]
  rfl

/-- **`ProverSeesMessages`'s nontrivial lemma**: without `Secret` its conclusion fails: the honest party, `Faithful`
and `Confined`, whose signing state the prover reads once, gives the prover that access beside its messages. -/
theorem proverSeesMessages_nontrivial (hss : Unit → Nat → ByteArray → ByteArray) :
    ¬ ∀ r : Parties.Run Unit (Unit × Nat × List ByteArray) Step Ch ByteArray (Ch × ByteArray) Unit,
      Parties.Faithful (honestCode hss) r → Parties.Confined dest id r →
        Parties.proverView r = (((honestCode hss).messages r.coins r.truth).map id, []) := by
  intro h
  have hok := meantRun_ok (A := Unit) (honestCode hss) dest id () (honestSteps .empty)
  have e := h { meantRun (A := Unit) (honestCode hss) dest id () (honestSteps .empty) with access := [()] } hok.1
    hok.2.2.1
  simp [Parties.proverView] at e

end MemoryChallenger.Attest.Properties.SecurityProofs
