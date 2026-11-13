module

public import Proofs.Properties.Challenger.Signer
public import Proofs.Properties.Challenger.Files
public import Proofs.Properties.Challenger.Logs
public import Proofs.Properties.Challenger.Messages
public import Proofs.Properties.Challenger.Ledger
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Init.Data.String.Defs
import all Init.Data.Repr
import all Init.Data.Int.Repr

/-!
# An honest session withheld, read by the challenger's reader

The honest halves of `FamilyOneCommitment` and `FamilyOneCommitmentRead`: the reader accepts what honest parties write.
The parties are the auditor's device, which commits to one family of one challenge, records the commitment, logs the
root it answers and the release, and closes its log; and the auditor, whose ledger opens the session on a window
withheld and ends it closed. Everything they write is a function of variables (`Play`): the key's public bytes, the
challenge and its salt, the root answered, the window's root, the commit pinned, and the four signatures, all under
the one key the reader holds. Nothing is evaluated: SHA-512 stays opaque, a signature is what `Signer.signs` gives at
its leaf, and each text is the canonical text of the value it prints.

* `Play.Honest`: what makes a play honest under a key's identifier: the sizes, each text the canonical text of its
  value, and each signature one that verifies over the message its party signs.
* `draws`: `Challenger.drawCoins` returns the challenge of an honest play, and `reads`: `Challenger.coinsReason` gives
  no reason against it.
* `exists_honest`: a key `Lms.key` derives at the deployed parameters has an honest play.
* `familyOneCommitment_honest`, `familyOneCommitmentRead_honest`: the two at a key `Lms.keygen` derives.
-/

@[expose] public section

set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private verifies isDigest strOf jsonTrue isBodyOf from Spot.ChallengerSession

namespace Properties.Challenger.Withheld

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger
open Claims.Json (Same Good)
open Plain (PlainStr PlainAt)
open Built (isDigest_hex plain_hexOf hexOf_len keys?_mkObj canon_num)
open Bodies (openBody endBody read_str read_num read_bool nat?_num)
open Ledger (filesOf)
open Properties.NetworkCertifier.Bytes (hash_size)

/-! ## The names a play is written under -/

/-- The scheme of the device's commitment. -/
def SCHEME : String := "challenges/v1"

/-- The one family the device commits to. -/
def LABEL : String := "sample"

/-- The auditor's device, as its record and the ledger's open name it. -/
def DEVICE : String := "challenger"

/-- The session the ledger opens. -/
def SESSION : String := "session"

/-- Who opens the session. -/
def AUDITOR : String := "auditor"

/-- When the device's code was pinned. -/
def PINNED_AT : String := "2000-01-01T00:00:00Z"

theorem scheme_name : isName SCHEME = true := by
  rw [isName, String.all_bool_eq]
  decide

theorem label_name : isName LABEL = true := by
  rw [isName, String.all_bool_eq]
  decide

theorem device_named : isHost DEVICE = true := by
  rw [isHost, String.all_bool_eq]
  decide

theorem session_id : isChallengePhaseId SESSION = true := by
  rw [isChallengePhaseId, isHost, String.all_bool_eq]
  decide

set_option maxRecDepth 100000 in
theorem pinned_later : later PINNED_AT 0 = some PINNED_AT := by decide +kernel

theorem label_plain : PlainStr LABEL := by decide

theorem device_plain : PlainStr DEVICE := by decide

theorem session_plain : PlainStr SESSION := by decide

theorem auditor_plain : PlainStr AUDITOR := by decide

theorem open_kind : PlainStr "open" := by decide

theorem end_kind : PlainStr "end" := by decide

theorem schema_plain : PlainStr LEDGER_SCHEMA := by decide

/-- An LMS signature at the deployed parameters is 34,756 bytes. -/
theorem sig_bytes : Lms.deployed.signatureBytes = 34756 := by decide +kernel

/-! ## Small facts of lists, arrays and bytes -/

theorem all_one (f : Json → Bool) (x : Json) (h : f x = true) : (#[x] : Array Json).all f = true := by
  rw [show (#[x] : Array Json) = [x].toArray from rfl, List.all_toArray, List.all_cons, h]
  rfl

theorem any_one (f : Json → Bool) (x : Json) (h : f x = true) : (#[x] : Array Json).any f = true := by
  rw [show (#[x] : Array Json) = [x].toArray from rfl, List.any_toArray, List.any_cons, h]
  rfl

theorem find_window (w : Win) (root : String) (h : w.root = root) :
    (#[w] : Array Win).find? (·.root == root) = some w := by
  rw [show (#[w] : Array Win) = [w].toArray from rfl, List.find?_toArray, List.find?_cons_of_pos (by simp [h])]

theorem family_roots (label : String) (after : ByteArray) :
    ((([(label, [after])] : Roots).find? (·.1 == label)).map (·.2)).getD [] = [after] := by
  rw [List.find?_cons_of_pos (by simp)]
  rfl

/-- A field that reads as a string is there. -/
theorem field_some {j : Json} {k s : String} (h : getD j k = .str s) : (field? j k).isSome = true := by
  unfold getD at h
  cases hf : field? j k with
  | none => simp [hf] at h
  | some v => rfl

/-- A copy of main's history is its own. -/
theorem sameMain_one (s : String) : sameMain (.arr #[.str s]) (.arr #[.str s]) = true := by
  have h : mainLines? (.arr #[.str s]) = some [s] := rfl
  unfold sameMain
  rw [h]
  exact beq_self_eq_true _

/-- One challenge, read whole, is the start of itself with every challenge read. -/
theorem startReason_whole {coin : ByteArray} (h : coin.size = 64) : startReason coin coin = none := by
  have h1 : (coin.extract 0 coin.size != coin) = false := by
    rw [ByteArray.extract_zero_size]
    show (!(coin == coin)) = false
    rw [Steps.bytes_beq]
    rfl
  have h2 : ((coin.size + COIN_BYTES - 1) / COIN_BYTES != coin.size / COIN_BYTES) = false := by
    rw [h]
    rfl
  unfold startReason
  simp only [h1, h2, Bool.false_eq_true, ↓reduceIte]

/-! ## What the parties write -/

/-- The device's commitment: one family of one challenge, its root the challenge's leaf. -/
def commitmentOf (epoch : Nat) (dom coin salt : ByteArray) : LiveCommitment :=
  Files.one SCHEME LABEL epoch (Files.coinRoot dom SCHEME coin salt)

/-- The device's release log: its head, the root it answers and the release. -/
def linesOf (c : LiveCommitment) (rootText releaseText : String) : List String :=
  ["{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}", rootText, releaseText]

/-- The line of the pins log: main's head, pinned. -/
def pinValue (pin : ByteArray) : Json :=
  Json.mkObj [("commit", .str (Flock.hexOf pin)), ("main", .str (Flock.hexOf pin)), ("pinned_utc", .str PINNED_AT)]

/-- The files the ledger's end seals: the record's and the close's as the device wrote them, no hello refused, and
the release log. -/
def filesFor (closeFile recordFile : ByteArray) (lines : List String) : Json :=
  filesOf (Flock.Sha512.hash closeFile) (Flock.Sha512.hash (String.join (([] : List String).map (· ++ "\n"))).toUTF8)
    (Flock.Sha512.hash recordFile) (Flock.Sha512.hash (String.join (lines.map (· ++ "\n"))).toUTF8)

/-- The ledger's open of the session, on a window withheld. -/
def openOf (pub pin window : ByteArray) (pinText : String) (sig : ByteArray) : Core.Record :=
  ⟨openBody AUDITOR DEVICE (logDigest [pinText]) (Flock.Sha512.hash pub) pin window, "open", ZERO, LEDGER_SCHEMA, 0,
    some SESSION, sig⟩

/-- The ledger's end of the session, closed, after the open whose text is `openText`. -/
def endOf (root : ByteArray) (files : Json) (openText : String) (sig : ByteArray) : Core.Record :=
  ⟨endBody root files 1, "end", Flock.hexOf (Flock.Sha512.hash openText.toUTF8), LEDGER_SCHEMA, 1, some SESSION, sig⟩

theorem root_size (epoch : Nat) (dom coin salt : ByteArray) : (commitmentOf epoch dom coin salt).root.size = 64 := by
  show (Flock.Sha512.hash _).size = 64
  exact hash_size _

theorem chain_size (c : LiveCommitment) (a b : String) : (logChain (linesOf c a b)).size = 64 := by
  show (Flock.Sha512.hash _).size = 64
  exact hash_size _

theorem pin_plain (pin : ByteArray) : PlainAt PyJson.DEPTH (pinValue pin) :=
  Ledger.plainAt_mono (Ledger.plainAt_built (d := 1) (ks := ["commit", "main", "pinned_utc"]) rfl (by decide)
    (by decide)
    (by
      intro q hq
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl | rfl
      · exact .str 0 (plain_hexOf _)
      · exact .str 0 (plain_hexOf _)
      · exact .str 0 (by decide))) (by decide)

theorem pins_plain (pinText : String) : PlainStr (logDigest [pinText]) := plain_hexOf _

theorem pins_len (pinText : String) : (logDigest [pinText]).length = 128 := by
  show (Flock.hexOf (Flock.Sha512.hash _)).length = 128
  rw [hexOf_len, hash_size]

theorem files_plain (closeFile recordFile : ByteArray) (lines : List String) :
    PlainAt 2 (filesFor closeFile recordFile lines) := Ledger.plainAt_files _ _ _ _

theorem files_text (closeFile recordFile : ByteArray) (lines : List String) :
    ∃ t, Flock.canon (filesFor closeFile recordFile lines) = .ok t ∧ t.length ≤ 590 := by
  obtain ⟨t, ht, hl⟩ := Ledger.canon_files (Flock.Sha512.hash closeFile)
    (Flock.Sha512.hash (String.join (([] : List String).map (· ++ "\n"))).toUTF8) (Flock.Sha512.hash recordFile)
    (Flock.Sha512.hash (String.join (lines.map (· ++ "\n"))).toUTF8)
  simp only [hexOf_len, hash_size] at hl
  exact ⟨t, ht, by omega⟩

theorem files_digests {a b c d : ByteArray} (ha : a.size = 64) (hb : b.size = 64) (hc : c.size = 64)
    (hd : d.size = 64) :
    ∀ q ∈ [("challenger.json", Json.str (Flock.hexOf a)), ("hellos.jsonl", Json.str (Flock.hexOf b)),
      ("record.json", Json.str (Flock.hexOf c)), ("releases.jsonl", Json.str (Flock.hexOf d))],
      ∃ x : ByteArray, 2 * x.size = 128 ∧ q.2 = .str (Flock.hexOf x) := by
  intro q hq
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
  rcases hq with rfl | rfl | rfl | rfl
  · exact ⟨a, by omega, rfl⟩
  · exact ⟨b, by omega, rfl⟩
  · exact ⟨c, by omega, rfl⟩
  · exact ⟨d, by omega, rfl⟩

theorem open_plain (pub pin window : ByteArray) (pinText : String) :
    PlainAt 3 (openBody AUDITOR DEVICE (logDigest [pinText]) (Flock.Sha512.hash pub) pin window) :=
  Ledger.plainAt_openBody _ _ _ auditor_plain device_plain (pins_plain pinText)

theorem open_text (pub : ByteArray) {pin window : ByteArray} (pinText : String) (hp : pin.size = 20)
    (hw : window.size = 64) :
    ∃ t, Flock.canon (openBody AUDITOR DEVICE (logDigest [pinText]) (Flock.Sha512.hash pub) pin window) = .ok t ∧
      t.length ≤ 600 := by
  obtain ⟨t, ht, hl⟩ := Ledger.canon_openBody (Flock.Sha512.hash pub) pin window auditor_plain device_plain
    (pins_plain pinText)
  have h1 : AUDITOR.length = 7 := by decide
  have h2 : DEVICE.length = 10 := by decide
  have h3 := pins_len pinText
  have h4 : (Flock.hexOf (Flock.Sha512.hash pub)).length = 128 := by rw [hexOf_len, hash_size]
  have h5 : (Flock.hexOf pin).length = 40 := by rw [hexOf_len, hp]
  have h6 : (Flock.hexOf window).length = 128 := by rw [hexOf_len, hw]
  exact ⟨t, ht, by omega⟩

theorem end_plain (root : ByteArray) (closeFile recordFile : ByteArray) (lines : List String) :
    PlainAt 3 (endBody root (filesFor closeFile recordFile lines) 1) :=
  Ledger.plainAt_endBody root 1 (files_plain closeFile recordFile lines)

theorem end_text {root : ByteArray} (closeFile recordFile : ByteArray) (lines : List String) (hr : root.size = 64) :
    ∃ t, Flock.canon (endBody root (filesFor closeFile recordFile lines) 1) = .ok t ∧ t.length ≤ 800 := by
  obtain ⟨t, ht, hl⟩ := Ledger.canon_endBody (F := 590) root (recordLeaf := 1)
    (files_text closeFile recordFile lines) ⟨_, canon_num 1, by decide⟩
  have h1 : (Flock.hexOf root).length = 128 := by rw [hexOf_len, hr]
  exact ⟨t, ht, by omega⟩

/-! ## A ledger's entry, read back -/

/-- **An entry an honest auditor writes for the session reads back from its text**, with its body alike the one
written and its message the one the auditor signed. -/
theorem entry_read {r : Core.Record} {text : String} {mb : Std.TreeMap.Raw String Json} {pb : ByteArray} {B : Nat}
    (hb : r.body = .obj mb) (hpb : PlainAt 3 r.body) (hcb : ∃ t, Flock.canon r.body = .ok t ∧ t.length ≤ B)
    (hB : B ≤ 1024) (hk : r.kind = "open" ∨ r.kind = "end") (hprev : r.prev.length = 128) (hpp : PlainStr r.prev)
    (hhex : Flock.unhex r.prev = .ok pb) (hsch : r.schema = LEDGER_SCHEMA)
    (hq : ∃ t, Flock.canon (Lean.toJson r.seq) = .ok t ∧ t.length ≤ 1) (hse : r.session = some SESSION)
    (hsig : r.sig.size = Lms.deployed.signatureBytes) (hc : Flock.canon r.toJson = .ok text)
    (hbody : ∀ b', Same r.body b' → isBodyOf r.kind true b' = true) :
    ∃ e nb, Same r.body (.obj nb) ∧ (∀ I m, ledgerMessage I r.toJson = .ok m → ledgerMessage I e = .ok m) ∧
      ledgerEntry? (.str text) = some ⟨e, text, LEDGER_SCHEMA, r.seq, r.prev, r.kind, SESSION, r.sig, .obj nb⟩ := by
  have hkp : PlainStr r.kind := by
    rcases hk with e | e <;> rw [e]
    · exact open_kind
    · exact end_kind
  have hsp : PlainStr r.schema := by
    rw [hsch]
    exact schema_plain
  have hpl : PlainAt PyJson.DEPTH r.toJson := Ledger.plainAt_record hpb hkp hpp hsp hse session_plain
  obtain ⟨t, ht, hlen⟩ := Ledger.canon_record hcb hkp hpp hsp hq hse session_plain
  have htx : t = text := Except.ok.inj (ht.symm.trans hc)
  rw [htx] at hlen
  have hkl : r.kind.length ≤ 4 := by
    rcases hk with e | e <;> rw [e] <;> decide
  have hsl : r.schema.length = 28 := by
    rw [hsch]
    decide
  have hxl : (Flock.hexOf r.sig).length = 69512 := by rw [hexOf_len, hsig, sig_bytes]
  have hss : SESSION.length = 7 := by decide
  have h4 : 4 ≤ r.sig.size := by
    rw [hsig, sig_bytes]
    decide
  have hlt : text.length ≤ MAX_LEDGER_LINE := by
    unfold MAX_LEDGER_LINE
    omega
  have htag : ledgerTag? r.schema = some "verity/challenger-auditor/entry/v2" := by
    rw [hsch]
    rfl
  obtain ⟨e, nb, hse', hsb, hle⟩ := Entry.ledgerEntry_of_record hpl hc hlt hb hprev hhex htag h4 hk hse session_id
    (fun b' hs => by
      rw [hsch, beq_self_eq_true]
      exact hbody b' hs)
  refine ⟨e, nb, hsb, ?_, ?_⟩
  · intro I m hm
    have hg := Entry.good_of_plain hpl
    rw [Entry.ledgerMessage_same (I := I) hse' hg (Verity.Sampled.sortedKeys_of_wf _ (Built.wf_of_good hg))
      (rfl : getD r.toJson "schema" = .str r.schema)]
    exact hm
  · rw [hle, hsch]

/-! ## A play, and what makes it honest -/

/-- What an honest session withheld is written from: the key's public bytes, the commitment's epoch and domain, the
challenge and its salt, the root the challenge answers, the window's root, the commit pinned, the device's two files
as it wrote them, the four signatures, and the text of each line. -/
structure Play where
  pub : ByteArray
  epoch : Nat
  dom : ByteArray
  coin : ByteArray
  salt : ByteArray
  after : ByteArray
  window : ByteArray
  pin : ByteArray
  recordFile : ByteArray
  closeFile : ByteArray
  sigRecord : ByteArray
  sigClose : ByteArray
  sigOpen : ByteArray
  sigEnd : ByteArray
  rootText : String
  releaseText : String
  pinText : String
  openText : String
  endText : String

namespace Play

variable (p : Play)

/-- The key the verifier holds. -/
def key : ChallengerKey := ⟨Lms.deployed, p.pub⟩

def commitment : LiveCommitment := commitmentOf p.epoch p.dom p.coin p.salt

/-- The device's record of its commitment. -/
def record : ChallengerRecord := ⟨p.commitment, DEVICE, p.sigRecord⟩

def lines : List String := linesOf p.commitment p.rootText p.releaseText

/-- The device's close of its log. -/
def close : ChallengerClose := ⟨SESSION, 3, logChain p.lines, none, p.sigClose⟩

/-- The session's files, as the reader takes them. -/
def session : Json :=
  Json.mkObj [("close", p.close.toJson), ("hellos", .arr #[]), ("log", .arr (p.lines.map Json.str).toArray),
    ("record", p.record.toJson)]

def files : Json := filesFor p.closeFile p.recordFile p.lines

def openRecord : Core.Record := openOf p.pub p.pin p.window p.pinText p.sigOpen

def endRecord : Core.Record := endOf p.commitment.root p.files p.openText p.sigEnd

/-- The challenge, opened after the root it answers. -/
def opening : LiveOpening := ⟨p.epoch, LABEL, 0, p.after, p.coin, p.salt, []⟩

/-- The sample's segment: the family, from its first challenge, and the one opening. -/
def segment : Json :=
  Json.mkObj [("after", .str (Flock.hexOf p.after)), ("label", .str LABEL), ("openings", .arr #[p.opening.toJson]),
    ("start", Lean.toJson (0 : Nat))]

/-- The evidence `drawCoins` reads. -/
def evidence : Json :=
  Json.mkObj [("ledger", .arr #[.str p.openText, .str p.endText]), ("main", .arr #[.str (Flock.hexOf p.pin)]),
    ("pins", .arr #[.str p.pinText]), ("segment", p.segment), ("session", p.session)]

/-- The play is honest under the key identifier `I`: the sizes are the protocol's, each text is the canonical text of
the value it prints, and each signature verifies over the message its party signs, at the leaves 0 (the open), 1 (the
record) and 2 (the end). -/
structure Honest (I : ByteArray) : Prop where
  identifier : I.size = 16
  pubId : p.pub.extract 0 16 = I
  coinSize : p.coin.size = 64
  saltSize : p.salt.size = 192
  afterSize : p.after.size = 64
  windowSize : p.window.size = 64
  pinSize : p.pin.size = 20
  domain : liveDomain SCHEME p.epoch [(LABEL, 1)] = .ok p.dom
  recordSigned : ∃ m s, recordMessage I DEVICE p.commitment = .ok m ∧
    Lms.Signature.decode Lms.deployed p.sigRecord = .ok s ∧ Lms.verify Lms.deployed p.pub m s = .ok true
  recordLeaf : leafOf p.sigRecord = 1
  rootLine : Flock.canon (Logs.rootLine LABEL p.after) = .ok p.rootText
  releaseLine : Flock.canon (Logs.releaseLine LABEL p.after) = .ok p.releaseText
  closeSigned : ∃ m s, closeMessage I DEVICE p.commitment SESSION 3 (logChain p.lines) none = .ok m ∧
    Lms.Signature.decode Lms.deployed p.sigClose = .ok s ∧ Lms.verify Lms.deployed p.pub m s = .ok true
  pinLine : Flock.canon (pinValue p.pin) = .ok p.pinText
  openLine : Flock.canon p.openRecord.toJson = .ok p.openText
  openSigned : ∃ m, ledgerMessage I p.openRecord.toJson = .ok m ∧ verifies p.key m p.sigOpen = true
  openLeaf : leafOf p.sigOpen = 0
  openSize : p.sigOpen.size = Lms.deployed.signatureBytes
  endLine : Flock.canon p.endRecord.toJson = .ok p.endText
  endSigned : ∃ m, ledgerMessage I p.endRecord.toJson = .ok m ∧ verifies p.key m p.sigEnd = true
  endLeaf : leafOf p.sigEnd = 2
  endSize : p.sigEnd.size = Lms.deployed.signatureBytes

end Play

variable {p : Play} {I : ByteArray}

/-! ## The session's files -/

theorem commitment_root (p : Play) : p.commitment.root.size = 64 := root_size _ _ _ _

theorem record_read (p : Play) : ChallengerRecord.ofJson p.record.toJson = .ok p.record :=
  Files.record_ofJson p.sigRecord
    (Files.commitment_ofJson p.epoch scheme_name label_name (root_size p.epoch p.dom p.coin p.salt)) device_named

theorem close_read (p : Play) : ChallengerClose.ofJson p.close.toJson = .ok p.close :=
  Files.close_ofJson p.sigClose (chain_size p.commitment p.rootText p.releaseText) (by decide) (by decide) session_id

theorem released (h : p.Honest I) : releasedRootsOf p.commitment p.lines = .ok [(LABEL, [p.after])] := by
  obtain ⟨a, b, ha, hb, hr⟩ := Logs.released_one (c := p.commitment) (label := LABEL) (after := p.after) (n := 1) rfl
    label_plain (by decide) h.afterSize
  obtain rfl : a = p.rootText := Except.ok.inj (ha.symm.trans h.rootLine)
  obtain rfl : b = p.releaseText := Except.ok.inj (hb.symm.trans h.releaseLine)
  exact hr

theorem record_signed (h : p.Honest I) : recordReason p.key p.record.toJson = .ok none := by
  obtain ⟨m, s, hm, hd, hv⟩ := h.recordSigned
  have hI : p.key.pub.extract 0 16 = I := h.pubId
  exact Steps.recordReason_none_of_steps (r := p.record) rfl (record_read p)
    (by
      rw [hI]
      exact hm) hd hv

/-- **`challengePhaseOf` reads an honest session**: its commitment, the root its one challenge was released after, and the
session's name. -/
theorem session_read (h : p.Honest I) :
    challengePhaseOf p.key p.record.toJson (.arr (p.lines.map Json.str).toArray) p.close.toJson =
      .ok (p.commitment, [(LABEL, [p.after])], SESSION) := by
  obtain ⟨m, s, hm, hd, hv⟩ := h.closeSigned
  have hI : p.key.pub.extract 0 16 = I := h.pubId
  exact Steps.challengePhaseOf_ok_of_steps (r := p.record) (c := p.close) (record_signed h) (record_read p) rfl (close_read p)
    (by
      rw [hI]
      exact hm) hd hv rfl rfl rfl (released h)

/-! ## The ledger's two entries -/

theorem open_read (h : p.Honest I) :
    ∃ e nb, Same p.openRecord.body (.obj nb) ∧
      (∀ I m, ledgerMessage I p.openRecord.toJson = .ok m → ledgerMessage I e = .ok m) ∧
      ledgerEntry? (.str p.openText) =
        some ⟨e, p.openText, LEDGER_SCHEMA, 0, ZERO, "open", SESSION, p.sigOpen, .obj nb⟩ :=
  entry_read (r := p.openRecord) (B := 600) rfl (open_plain p.pub p.pin p.window p.pinText)
    (open_text p.pub p.pinText h.pinSize h.windowSize) (by decide) (.inl rfl) Ledger.zero_len Ledger.zero_plain
    Ledger.zero_unhex rfl Logs.canon_zero rfl h.openSize h.openLine
    (fun _ hs => Bodies.openBody_alike hs device_named (hash_size _) h.pinSize h.windowSize)

theorem end_read (h : p.Honest I) :
    ∃ e nb, Same p.endRecord.body (.obj nb) ∧
      (∀ I m, ledgerMessage I p.endRecord.toJson = .ok m → ledgerMessage I e = .ok m) ∧
      ledgerEntry? (.str p.endText) =
        some ⟨e, p.endText, LEDGER_SCHEMA, 1, Flock.hexOf (Flock.Sha512.hash p.openText.toUTF8), "end", SESSION,
          p.sigEnd, .obj nb⟩ :=
  entry_read (r := p.endRecord) (B := 800) rfl (end_plain _ p.closeFile p.recordFile p.lines)
    (end_text p.closeFile p.recordFile p.lines (root_size p.epoch p.dom p.coin p.salt)) (by decide) (.inr rfl)
    (by
      show (Flock.hexOf (Flock.Sha512.hash _)).length = 128
      rw [hexOf_len, hash_size]) (plain_hexOf _) (Flock.Hex.unhex_hexOf _) rfl ⟨_, canon_num 1, by decide⟩ rfl
    h.endSize h.endLine
    (fun _ hs => Bodies.endBody_alike hs (by simp only [List.map_cons, List.map_nil]; decide)
      (files_digests (hash_size _) (hash_size _) (hash_size _) (hash_size _)))

theorem pin_read (h : p.Honest I) : pinOf? (.str p.pinText) = some (Flock.hexOf p.pin) :=
  Logs.pinOf_of_canon (pin_plain p.pin) h.pinLine (keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide))
    rfl rfl h.pinSize (t := PINNED_AT) rfl pinned_later

/-! ## The reader accepts -/

/-- **`trustedChallengePhaseOf` reads an honest play**: the ledger opens the session on the window, the pins log pins main's
head, and the session's files are the ones the ledger's end seals. -/
theorem trusted (h : p.Honest I) :
    trustedChallengePhaseOf (.arr #[.str p.openText, .str p.endText]) (Json.mkObj [("root", Json.str (Flock.hexOf p.window))])
      p.key (.arr #[.str p.pinText]) (.arr #[.str (Flock.hexOf p.pin)]) p.session =
        .ok (p.commitment, [(LABEL, [p.after])]) ∧
      withheldOn (.arr #[.str p.openText, .str p.endText]) (Flock.hexOf p.window) = true := by
  obtain ⟨eo, nbo, hbo, hto, heo⟩ := open_read h
  obtain ⟨ee, nbe, hbe, hte, hee⟩ := end_read h
  obtain ⟨mo, hmo, hvo⟩ := h.openSigned
  obtain ⟨me, hme, hve⟩ := h.endSigned
  have hI : p.key.pub.extract 0 16 = I := h.pubId
  -- what the open's body reads
  have r1 : getD (.obj nbo) "host" = .str DEVICE := read_str hbo rfl
  have r2 : getD (.obj nbo) "key_sha512" = .str (Flock.hexOf (Flock.Sha512.hash p.pub)) := read_str hbo rfl
  have r3 : getD (.obj nbo) "pinned" = .str (Flock.hexOf p.pin) := read_str hbo rfl
  have r4 : getD (.obj nbo) "withheld" = .bool true := read_bool hbo rfl
  have hwn := Entry.same_getD hbo "window"
  have r5 : getD (getD (.obj nbo) "window") "root" = .str (Flock.hexOf p.window) := read_str hwn rfl
  have hpn := Entry.same_getD hbo "pins"
  have r6 : getD (getD (.obj nbo) "pins") "lines" = .num ⟨((1 : Nat) : Int), 0⟩ := read_num hpn rfl
  have r7 : getD (getD (.obj nbo) "pins") "sha512" = .str (logDigest [p.pinText]) := read_str hpn rfl
  have r8 : keys? (getD (.obj nbo) "pins") = some #["lines", "sha512"] :=
    (Entry.same_keys? hpn).symm.trans
      (keys?_mkObj (l := [("lines", Lean.toJson (1 : Nat)), ("sha512", .str (logDigest [p.pinText]))])
        (by simp only [List.map_cons, List.map_nil]; decide))
  have r9 : field? (.obj nbo) "reserve" = none := Entry.same_field?_none hbo rfl
  -- what the end's body reads
  have e1 : getD (.obj nbe) "record_leaf" = .num ⟨((1 : Nat) : Int), 0⟩ := read_num hbe rfl
  have e2 : getD (.obj nbe) "commitment" = .str (Flock.hexOf p.commitment.root) := read_str hbe rfl
  have e5 : getD (.obj nbe) "closed" = .bool true := read_bool hbe rfl
  have e6 : getD (.obj nbe) "refused_hellos" = .num ⟨((0 : Nat) : Int), 0⟩ := read_num hbe rfl
  have hfs := Entry.same_getD hbe "files"
  have e3 : getD (getD (.obj nbe) "files") LOG_FILE = .str (logDigest p.lines) := read_str hfs rfl
  have e4 : getD (getD (.obj nbe) "files") HELLOS_FILE = .str (logDigest []) := read_str hfs rfl
  have e7 : getD (getD (.obj nbe) "files") RECORD_FILE = .str (Flock.hexOf (Flock.Sha512.hash p.recordFile)) :=
    read_str hfs rfl
  have e8 : getD (getD (.obj nbe) "files") CLOSE_FILE = .str (Flock.hexOf (Flock.Sha512.hash p.closeFile)) :=
    read_str hfs rfl
  -- the ledger checks
  have hor : strOf (getD (getD (.obj nbo) "window") "root") = Flock.hexOf p.window := by
    rw [r5]
    rfl
  have hrl : nat? (getD (.obj nbe) "record_leaf") = some 1 := by
    rw [e1]
    exact nat?_num 1
  have hrn : (getD (.obj nbe) "record_leaf").isNull = false := by
    rw [e1]
    rfl
  have hl1 : leafOf p.sigOpen < 1 := by
    rw [h.openLeaf]
    decide
  have hl2 : 1 < leafOf p.sigEnd := by
    rw [h.endLeaf]
    decide
  have hmo' : ledgerMessage (p.key.pub.extract 0 16) eo = .ok mo := by
    rw [hI]
    exact hto I mo hmo
  have hme' : ledgerMessage (p.key.pub.extract 0 16) ee = .ok me := by
    rw [hI]
    exact hte I me hme
  have hv := Steps.verifyLedger_two (key := p.key)
    (o := ⟨eo, p.openText, LEDGER_SCHEMA, 0, ZERO, "open", SESSION, p.sigOpen, .obj nbo⟩)
    (e := ⟨ee, p.endText, LEDGER_SCHEMA, 1, Flock.hexOf (Flock.Sha512.hash p.openText.toUTF8), "end", SESSION,
      p.sigEnd, .obj nbe⟩) (root := Flock.hexOf p.window) (r := 1)
    rfl rfl rfl rfl hmo' hvo hor r9 rfl rfl rfl rfl hme' hve rfl hrl hrn hl1 hl2
  have hkey : strOf (getD (.obj nbo) "key_sha512") = Flock.hexOf (Flock.Sha512.hash p.key.pub) := by
    rw [r2]
    rfl
  have hck := Steps.checked_two (key := p.key) heo hee rfl rfl hkey hv
  -- the session the ledger opens on the window
  have hwin : isRun (Json.mkObj [("root", Json.str (Flock.hexOf p.window))]) = true := by
    have hkw : keys? (Json.mkObj [("root", Json.str (Flock.hexOf p.window))]) = some #["root"] :=
      keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide)
    have hrw : getD (Json.mkObj [("root", Json.str (Flock.hexOf p.window))]) "root" = .str (Flock.hexOf p.window) :=
      rfl
    unfold isRun
    rw [hkw, hrw, isDigest_hex (b := p.window) (n := 128) (by have := h.windowSize; omega), beq_self_eq_true]
    rfl
  have hcl : jsonTrue (getD (.obj nbe) "closed") = true := by
    rw [e5]
    rfl
  have hrh : nat? (getD (.obj nbe) "refused_hellos") = some 0 := by
    rw [e6]
    exact nat?_num 0
  have hcm : isDigest (getD (.obj nbe) "commitment") = true := by
    rw [e2]
    exact isDigest_hex (by have := commitment_root p; omega)
  have hfl : SEALED_FILES.all (fun f => (field? (getD (.obj nbe) "files") f).isSome) = true := by
    simp only [SEALED_FILES, List.all_cons, List.all_nil, field_some e7, field_some e8, field_some e3, field_some e4,
      Bool.and_self]
  have hao := Steps.auditedOpen_ok_of_steps (recordLeaf := 1) (refusedHellos := 0) hwin hck
    (find_window _ _ (by
      show Flock.hexOf p.window = strOf (getD (Json.mkObj [("root", Json.str (Flock.hexOf p.window))]) "root")
      rfl)) rfl rfl hcl hrl hrh hcm hfl
  -- the pins log and main
  have hpr := Steps.pinsReason_none_of (session := SESSION) (opened := .obj nbo) (pin_read h)
    (by
      rw [r6]
      exact nat?_num 1) r8
    (by
      rw [r7]
      exact isDigest_hex (b := Flock.Sha512.hash _) (by rw [hash_size]))
    (by
      rw [r7]
      exact Logs.jsonStrIs_str _)
    (by
      rw [r3]
      rfl)
  have hfp := Steps.firstParentReason_none_of (session := SESSION) (pinned := strOf (getD (.obj nbo) "pinned"))
    (cs := #[.str (Flock.hexOf p.pin)]) (all_one _ _ (isDigest_hex (by have := h.pinSize; omega)))
    (any_one _ _ (by
      rw [r3]
      exact Logs.jsonStrIs_str _))
  have hps := Steps.pinnedChallengePhase_ok_of_steps hao hpr hfp
  -- the session's files
  have g1 : getD p.session "record" = p.record.toJson := rfl
  have g2 : getD p.session "log" = .arr (p.lines.map Json.str).toArray := rfl
  have g3 : getD p.session "close" = p.close.toJson := rfl
  have g4 : getD p.session "hellos" = .arr (([] : List String).map Json.str).toArray := rfl
  have hsn := Steps.challengePhaseNamed_ok_of_steps (key := p.key) (session := p.session) (c := p.commitment)
    (roots := [(LABEL, [p.after])]) (r := p.record) (lines := p.lines) (hellos := [])
    (a :=
      { session := SESSION, host := strOf (getD (.obj nbo) "host"), prover := laneOf (.obj nbo),
        files := getD (.obj nbe) "files", recordLeaf := 1, commitment := strOf (getD (.obj nbe) "commitment"),
        refusedHellos := 0, pinned := strOf (getD (.obj nbo) "pinned"), period := none, cap := Json.null,
        circuit := Json.null })
    (keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide))
    (by
      rw [g1, g2, g3]
      exact session_read h)
    (by
      rw [g1]
      exact record_read p)
    (by
      show DEVICE = strOf (getD (.obj nbo) "host")
      rw [r1]
      rfl) h.recordLeaf
    (by
      show Flock.hexOf p.commitment.root = strOf (getD (.obj nbe) "commitment")
      rw [e2]
      rfl) g2
    (by
      show jsonStrIs (getD (getD (.obj nbe) "files") LOG_FILE) (logDigest p.lines) = true
      rw [e3]
      exact Logs.jsonStrIs_str _) g4
    (by
      show jsonStrIs (getD (getD (.obj nbe) "files") HELLOS_FILE) (logDigest []) = true
      rw [e4]
      exact Logs.jsonStrIs_str _) rfl
  exact ⟨Steps.trustedChallengePhaseOf_ok_of_steps hps hsn, Steps.withheldOn_two heo hee rfl rfl r5 r4⟩

/-- **`drawCoins` returns an honest play's challenge.** -/
theorem draws (h : p.Honest I) :
    drawCoins p.key (Flock.hexOf p.window) LABEL p.after p.evidence = .ok p.coin := by
  obtain ⟨ht, hw⟩ := trusted h
  have c1 : getD p.evidence "ledger" = .arr #[.str p.openText, .str p.endText] := rfl
  have c2 : getD p.evidence "pins" = .arr #[.str p.pinText] := rfl
  have c3 : getD p.evidence "main" = .arr #[.str (Flock.hexOf p.pin)] := rfl
  have c4 : getD p.evidence "session" = p.session := rfl
  have c5 : getD p.evidence "segment" = p.segment := rfl
  have hst := Steps.challengerStream_one (c := p.commitment) (segment := p.segment) (doc := p.opening.toJson)
    (after := p.after) (label := LABEL) (o := p.opening)
    (keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide)) rfl (Files.nat?_nat 0) rfl
    (Logs.jsonStrIs_str _) (Files.opening_ofJson _ _ _ _ _ _)
    (Files.liveReason_one p.after h.domain h.coinSize h.saltSize)
  exact Steps.drawCoins_ok_of_steps (c := p.commitment) (roots := [(LABEL, [p.after])])
    (keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide))
    (by
      rw [c1, c2, c3, c4]
      exact ht)
    (by
      rw [c1]
      exact hw)
    (by
      rw [c5]
      exact Logs.jsonStrIs_str _)
    (by
      rw [c5, family_roots]
      exact hst)

/-- **`coinsReason` gives no reason against an honest play** read whole, under the verifier's own copy of main's
history. -/
theorem reads (h : p.Honest I) :
    coinsReason p.key (.arr #[.str (Flock.hexOf p.pin)]) p.evidence (Flock.hexOf p.window) LABEL p.after p.coin =
      none := by
  have c3 : getD p.evidence "main" = .arr #[.str (Flock.hexOf p.pin)] := rfl
  exact Steps.coinsReason_none_of_steps (draws h)
    (by
      rw [c3]
      exact sameMain_one _) (startReason_whole h.coinSize)

/-! ## A key's honest play -/

/-- **A key derived at the deployed parameters has an honest play**, for every epoch, challenge and salt, root
answered, window's root and commit pinned of the protocol's sizes. -/
theorem exists_honest {I seed : ByteArray} {k : Lms.PrivateKey} (hk : Lms.key Lms.deployed I seed = .ok k)
    (epoch : Nat) {coin salt after window pin : ByteArray} (hc : coin.size = 64) (hs : salt.size = 192)
    (ha : after.size = 64) (hw : window.size = 64) (hp : pin.size = 20) (recordFile closeFile : ByteArray) :
    ∃ p : Play, p.pub = k.publicKey ∧ p.coin = coin ∧ p.after = after ∧ p.window = window ∧ p.pin = pin ∧
      p.Honest I := by
  obtain ⟨-, hIk, -, hIs, -, -, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hk
  subst hIk
  have hpub : k.publicKey.extract 0 16 = k.I := ByteArray.extract_append_eq_left hIs.symm
  obtain ⟨dom, hdom⟩ := Messages.liveDomain_ok SCHEME epoch [(LABEL, 1)]
  -- the device's record
  obtain ⟨mR, hmR⟩ := Messages.recordMessage_ok hIs DEVICE (commitmentOf epoch dom coin salt)
  obtain ⟨sR, hdR, hvR, -, hlR, -⟩ := Signer.signs hk mR (q := 1) (by decide)
  -- its log and its close
  obtain ⟨a, b, hca, hcb, -⟩ := Logs.released_one (c := commitmentOf epoch dom coin salt) (label := LABEL)
    (after := after) (n := 1) rfl label_plain (by decide) ha
  obtain ⟨mC, hmC⟩ := Messages.closeMessage_ok hIs DEVICE (commitmentOf epoch dom coin salt) SESSION 3
    (logChain (linesOf (commitmentOf epoch dom coin salt) a b)) none
  obtain ⟨sC, hdC, hvC, -, -, -⟩ := Signer.signs hk mC (q := 3) (by decide)
  -- the pins log's line
  obtain ⟨tPin, htPin⟩ := FlockComplete.JsonRT.canon_ok_some _ (Entry.good_of_plain (pin_plain pin))
  -- the ledger's open
  obtain ⟨mO, hmO⟩ := Messages.ledgerMessage_ok hIs (r := openOf k.publicKey pin window tPin .empty) rfl
    (Entry.good_of_plain (open_plain k.publicKey pin window tPin))
  obtain ⟨sO, -, -, hvO, hlO, hzO⟩ := Signer.signs hk mO (q := 0) (by decide)
  have hmO' : ledgerMessage k.I (openOf k.publicKey pin window tPin sO.encode).toJson = .ok mO :=
    (Entry.ledgerMessage_sig k.I (openOf k.publicKey pin window tPin .empty) sO.encode).trans hmO
  obtain ⟨tOpen, htOpen⟩ := FlockComplete.JsonRT.canon_ok_some _ (Entry.good_of_plain
    (Ledger.plainAt_record (r := openOf k.publicKey pin window tPin sO.encode) (open_plain k.publicKey pin window tPin)
      open_kind Ledger.zero_plain schema_plain rfl session_plain))
  -- the ledger's end
  have hfiles : PlainAt 3 (endBody (commitmentOf epoch dom coin salt).root
      (filesFor closeFile recordFile (linesOf (commitmentOf epoch dom coin salt) a b)) 1) :=
    end_plain _ closeFile recordFile _
  obtain ⟨mE, hmE⟩ := Messages.ledgerMessage_ok hIs
    (r := endOf (commitmentOf epoch dom coin salt).root
      (filesFor closeFile recordFile (linesOf (commitmentOf epoch dom coin salt) a b)) tOpen .empty) rfl
    (Entry.good_of_plain hfiles)
  obtain ⟨sE, -, -, hvE, hlE, hzE⟩ := Signer.signs hk mE (q := 2) (by decide)
  have hmE' : ledgerMessage k.I (endOf (commitmentOf epoch dom coin salt).root
      (filesFor closeFile recordFile (linesOf (commitmentOf epoch dom coin salt) a b)) tOpen
      sE.encode).toJson = .ok mE :=
    (Entry.ledgerMessage_sig k.I (endOf (commitmentOf epoch dom coin salt).root
      (filesFor closeFile recordFile (linesOf (commitmentOf epoch dom coin salt) a b)) tOpen .empty)
      sE.encode).trans hmE
  obtain ⟨tEnd, htEnd⟩ := FlockComplete.JsonRT.canon_ok_some _ (Entry.good_of_plain
    (Ledger.plainAt_record (r := endOf (commitmentOf epoch dom coin salt).root
      (filesFor closeFile recordFile (linesOf (commitmentOf epoch dom coin salt) a b)) tOpen sE.encode)
      hfiles end_kind (plain_hexOf _) schema_plain rfl session_plain))
  exact ⟨⟨k.publicKey, epoch, dom, coin, salt, after, window, pin, recordFile, closeFile, sR.encode, sC.encode,
    sO.encode, sE.encode, a, b, tPin, tOpen, tEnd⟩, rfl, rfl, rfl, rfl, rfl,
    { identifier := hIs, pubId := hpub, coinSize := hc, saltSize := hs, afterSize := ha, windowSize := hw,
      pinSize := hp, domain := hdom, recordSigned := ⟨mR, sR, hmR, hdR, hvR⟩, recordLeaf := hlR, rootLine := hca,
      releaseLine := hcb, closeSigned := ⟨mC, sC, hmC, hdC, hvC⟩, pinLine := htPin, openLine := htOpen,
      openSigned := ⟨mO, hmO', hvO⟩, openLeaf := hlO, openSize := hzO, endLine := htEnd,
      endSigned := ⟨mE, hmE', hvE⟩, endLeaf := hlE, endSize := hzE }⟩

/-- `Lms.keygen` derives a key from a seed of the deployed parameters' 64 bytes: the key `Lms.key` derives from its
identifier. -/
theorem keygen_ok {seed : ByteArray} (hs : seed.size = Lms.deployed.N) :
    ∃ k, Lms.keygen Lms.deployed seed = .ok k ∧ ∃ I, Lms.key Lms.deployed I seed = .ok k := by
  have hI : ((Flock.Sha512.hash ("verity/lms/v0/I".toUTF8 ++ seed)).extract 0 16).size = 16 := by
    simp [ByteArray.size_extract, hash_size]
  obtain ⟨k, hk⟩ := Properties.CertifierDevice.Proofs.Lms.key_ok Lms.deployed hI seed
  refine ⟨k, ?_, _, hk⟩
  unfold Lms.keygen
  simpa only [hs, ne_eq, not_true_eq_false, ↓reduceIte, pure, Except.pure, bind, Except.bind] using hk

theorem zeros_size (n : Nat) : (⟨Array.replicate n 0⟩ : ByteArray).size = n := by
  show (Array.replicate n 0).size = n
  exact Array.size_replicate ..

end Properties.Challenger.Withheld

namespace Properties.Challenger

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger

/-- **`FamilyOneCommitment`'s honest lemma**: `Challenger.drawCoins` accepts at the honest deployed instance. A key
`Lms.keygen` derives at the deployed LMS parameters signs an honest session withheld (`Withheld.Play.Honest`): the
auditor's device commits to one family of one challenge, records the commitment, logs the root answered and the
release and closes its log, and the auditor's ledger opens the session on the window and ends it closed. `drawCoins`
under that key returns the session's challenge. -/
theorem familyOneCommitment_honest :
    ∃ (seed : ByteArray) (k : Lms.PrivateKey) (w label : String) (after : ByteArray) (coins : Json) (ξ : ByteArray),
      Lms.keygen Lms.deployed seed = .ok k ∧ drawCoins ⟨Lms.deployed, k.publicKey⟩ w label after coins = .ok ξ := by
  obtain ⟨k, hkg, I, hk⟩ := Withheld.keygen_ok (seed := ⟨Array.replicate 64 0⟩) (Withheld.zeros_size 64)
  obtain ⟨p, hpub, -, -, -, -, h⟩ := Withheld.exists_honest hk 0 (Withheld.zeros_size 64) (Withheld.zeros_size 192)
    (Withheld.zeros_size 64) (Withheld.zeros_size 64) (Withheld.zeros_size 20) .empty .empty
  have hd := Withheld.draws h
  have hkey : p.key = ⟨Lms.deployed, k.publicKey⟩ := congrArg (ChallengerKey.mk Lms.deployed) hpub
  rw [hkey] at hd
  exact ⟨_, k, _, _, _, _, _, hkg, hd⟩

/-- **`FamilyOneCommitmentRead`'s honest lemma**: `Challenger.coinsReason` gives no reason at the honest deployed
instance: the honest session of `familyOneCommitment_honest`, its one challenge read whole, under the verifier's own
copy of main's history. -/
theorem familyOneCommitmentRead_honest :
    ∃ (seed : ByteArray) (k : Lms.PrivateKey) (main : Json) (w label : String) (after read : ByteArray) (coins : Json),
      Lms.keygen Lms.deployed seed = .ok k ∧
        coinsReason ⟨Lms.deployed, k.publicKey⟩ main coins w label after read = none := by
  obtain ⟨k, hkg, I, hk⟩ := Withheld.keygen_ok (seed := ⟨Array.replicate 64 0⟩) (Withheld.zeros_size 64)
  obtain ⟨p, hpub, -, -, -, -, h⟩ := Withheld.exists_honest hk 0 (Withheld.zeros_size 64) (Withheld.zeros_size 192)
    (Withheld.zeros_size 64) (Withheld.zeros_size 64) (Withheld.zeros_size 20) .empty .empty
  have hd := Withheld.reads h
  have hkey : p.key = ⟨Lms.deployed, k.publicKey⟩ := congrArg (ChallengerKey.mk Lms.deployed) hpub
  rw [hkey] at hd
  exact ⟨_, k, _, _, _, _, _, _, hkg, hd⟩

end Properties.Challenger
