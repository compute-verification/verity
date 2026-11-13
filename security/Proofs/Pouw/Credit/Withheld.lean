module

public import Proofs.Properties.Challenger.Withheld
public import Proofs.Pouw.Credit
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# An honest window credited: `credited` accepts under a key `Lms.keygen` derives

`DrawWithheld` says what a credited window's sample rests on: the auditor's ledger opens a session withheld on the
window's sample root. This file builds a window `Internal.credited` accepts, on variables, so the property is about
something: the smallest window there is, of no call and no round, whose Commit record registers no value, with the
honest session withheld of `Properties.Challenger.Withheld` opened on its sample root.

* `pyFromHex_hex`, `digest_hex`, `isHexBytes_hex`: Python's `bytes.fromhex` reads lowercase hex back, a 64-byte
  digest's hex is a digest, and the hex of at least one byte is `credit._HEX`'s.
* `commit_read`: `commitOf?` reads the canonical text of a plain record of the registration format with no
  registered value as its digest under the format.
* `key_read`: `ChallengerKey.ofJson` reads a key of the deployed parameters back from its JSON.
* `root_ok`, `draw_ok`: the two halves of `credited`, each from what its readers return.
* `credited_play`: `credited` accepts an honest play of the family `pouw/draw` whose challenge answers the window's
  sample root and whose session the ledger opens on it.
* `drawWithheld_honest`: the same at a key `Lms.keygen` derives at the deployed parameters.

SHA-512 stays opaque and no signature is evaluated: the play's signatures verify by LMS's correctness
(`Properties.Challenger.Withheld.exists_honest`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private hexVal pySpace from PouwCommit.Py
open private hexOf? from Spot.ChallengerRecord

namespace Properties.Pouw.Credit.Withheld

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger
open Claims.Json (Same Good)
open Properties.Challenger
open Properties.Challenger.Plain (PlainStr PlainAt canonical_canon)
open Properties.Challenger.Bodies (read_str)
open Properties.Challenger.Withheld (Play)
open Properties.NetworkCertifier.Bytes (hash_size)
open PouwCheck.Credit (digest? commitOf? isHexBytes callsRoot COMMIT_FORMAT DRAW_TAG Internal.creditedRoot
  Internal.creditedDraw Internal.credited)

/-! ## Hex, as Python reads it -/

theorem nibble : ∀ d < 16, pySpace (Flock.Hex.hd d) = false ∧ hexVal (Flock.Hex.hd d) = some d ∧
    ((Flock.Hex.hd d).isDigit || ('a' ≤ Flock.Hex.hd d && Flock.Hex.hd d ≤ 'f')) = true := by decide

theorem byte_back (x : UInt8) : (16 * (x.toNat / 16) + x.toNat % 16).toUInt8 = x := by
  rw [Nat.div_add_mod]
  exact UInt8.ofNat_toNat

/-- `bytes.fromhex`'s loop reads each byte's two digits as the byte. -/
theorem go_hex (s : String) : ∀ (l : List UInt8) (acc : ByteArray),
    PouwCommit.pyFromHex.go s (l.flatMap fun x => [Flock.Hex.hd (x.toNat / 16), Flock.Hex.hd (x.toNat % 16)]) acc =
      .ok ⟨acc.data ++ l.toArray⟩
  | [], acc => by simp [PouwCommit.pyFromHex.go, pure, Except.pure]
  | x :: l, acc => by
    have h1 := nibble (x.toNat / 16) (by have := x.toNat_lt; omega)
    have h2 := nibble (x.toNat % 16) (Nat.mod_lt _ (by decide))
    rw [List.flatMap_cons]
    show PouwCommit.pyFromHex.go s (Flock.Hex.hd (x.toNat / 16) :: Flock.Hex.hd (x.toNat % 16) :: _) acc = _
    rw [PouwCommit.pyFromHex.go, h1.1]
    simp only [Bool.false_eq_true, ↓reduceIte, h1.2.1, h2.2.1]
    show PouwCommit.pyFromHex.go s (List.flatMap _ l) _ = _
    rw [go_hex s l, byte_back]
    simp [ByteArray.push]

/-- **`bytes.fromhex` reads lowercase hex back.** -/
theorem pyFromHex_hex (b : ByteArray) : PouwCommit.pyFromHex (Flock.hexOf b) = .ok b := by
  unfold PouwCommit.pyFromHex
  rw [Flock.Hex.hexOf_toList]
  show PouwCommit.pyFromHex.go _ (List.flatMap _ b.data.toList) _ = _
  rw [go_hex]
  cases b
  simp [ByteArray.empty, ByteArray.emptyWithCapacity]
  rfl

theorem hex_chars (b : ByteArray) :
    (Flock.hexOf b).toList.all (fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')) = true := by
  rw [Flock.Hex.hexOf_toList, List.all_flatMap]
  refine List.all_eq_true.2 fun x _ => ?_
  have h1 := nibble (x.toNat / 16) (by have := x.toNat_lt; omega)
  have h2 := nibble (x.toNat % 16) (Nat.mod_lt _ (by decide))
  simp only [List.all_cons, List.all_nil, Bool.and_true, h1.2.2, h2.2.2]

/-- **The hex of at least one byte is `credit._HEX`'s.** -/
theorem isHexBytes_hex {b : ByteArray} (hb : b.size ≠ 0) : isHexBytes (Flock.hexOf b) = true := by
  have hl : (Flock.hexOf b).length = 2 * b.size := Built.hexOf_len b
  have he : (Flock.hexOf b).isEmpty = false := by
    cases h : (Flock.hexOf b).isEmpty with
    | false => rfl
    | true =>
      rw [String.isEmpty_iff] at h
      rw [h] at hl
      have : (2 * b.size : Nat) = 0 := by simpa using hl.symm
      omega
  unfold isHexBytes
  rw [he, hl, String.all_bool_eq, hex_chars]
  simp

/-- **A 64-byte digest's hex reads as the digest.** -/
theorem digest_hex {b : ByteArray} (h : b.size = 64) : digest? (.str (Flock.hexOf b)) = some b := by
  have hd : PouwCheck.isHexDigest (Flock.hexOf b) = true := by
    unfold PouwCheck.isHexDigest
    rw [Built.hexOf_len, h, String.all_bool_eq, hex_chars]
    rfl
  unfold digest?
  simp only [PouwCheck.str?, PouwCheck.hex?, Option.any_some, hd, pyFromHex_hex, ite_true, Except.toOption]

/-! ## Reads -/

/-- PoUW's field read is the challenger's. -/
theorem get_getD (j : Json) (k : String) : PouwCheck.get j k = getD j k := by
  cases j <;> rfl

theorem bind_ok {α β : Type} {x : Flock.V α} {f : α → Flock.V β} {v : β} (h : (x >>= f) = .ok v) :
    ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- An object's canonical text is not empty: it opens with a brace. -/
theorem canon_obj_ne {m : Std.TreeMap.Raw String Json} {text : String} (hc : Flock.canon (.obj m) = .ok text) :
    text ≠ "" := by
  rw [CanonLine.canon_obj] at hc
  obtain ⟨vals, -, h⟩ := bind_ok hc
  intro he
  rw [he] at h
  have := congrArg String.length (Except.ok.inj h)
  simp [String.length_append] at this

theorem fromUTF8_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  rw [String.toUTF8_eq_toByteArray]
  have h : s.toByteArray.IsValidUTF8 := s.isValidUTF8
  unfold String.fromUTF8?
  rw [dite_eq_left_of_eq_true (eq_true h)]
  rfl

/-- A text that is not empty has bytes. -/
theorem utf8_ne {s : String} (h : s ≠ "") : s.toUTF8.size ≠ 0 := by
  rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]
  exact fun hz => h (String.utf8ByteSize_eq_zero_iff.mp hz)

/-! ## The Commit record -/

/-- **`commitOf?` reads its writer's record**: the canonical text `text` of a plain object `d` of the registration
format that registers no value, as the digest of its bytes under the format. -/
theorem commit_read {d : Json} {m : Std.TreeMap.Raw String Json} {text : String} (hp : PlainAt PyJson.DEPTH d)
    (hc : Flock.canon d = .ok text) (ho : d = .obj m) (hf : getD d "format" = .str COMMIT_FORMAT)
    (hr : getD d "registered" = .arr #[]) :
    commitOf? (.str (Flock.hexOf text.toUTF8)) =
      some (Flock.Sha512.hash ((COMMIT_FORMAT.toUTF8.push 0) ++ text.toUTF8), #[]) := by
  obtain ⟨e, hpe, hs⟩ := FlockComplete.JsonRT.parse_canon d text (Entry.good_of_plain hp) hc
  have hpy : PyJson.canonical e = .ok text := by
    rw [canonical_canon (Entry.plainAt_same hp hs), ← FlockComplete.JsonRT.canon_same d e hs, hc]
  have h1 : PouwCheck.get e "format" = .str COMMIT_FORMAT := (get_getD e _).trans (read_str hs hf)
  have h2 : PouwCheck.get e "registered" = .arr #[] := by
    have hg := Entry.same_getD hs "registered"
    rw [hr] at hg
    obtain ⟨b, hb, hsz, -⟩ := FlockComplete.JsonRT.same_arr hg
    have hb0 : b = #[] := Array.eq_empty_of_size_eq_zero (by simpa using hsz.symm)
    rw [get_getD, hb, hb0]
  have h3 : PouwCheck.Credit.isObj e = true := by
    subst ho
    obtain ⟨n, rfl⟩ := Entry.same_obj_eq hs
    rfl
  have hne : text.toUTF8.size ≠ 0 := utf8_ne (canon_obj_ne (ho ▸ hc))
  have ha : (#[] : Array Json).all PouwCheck.Credit.isObj = true := Array.all_empty
  unfold commitOf?
  simp only [PouwCheck.str?, PouwCheck.hex?, isHexBytes_hex hne, pyFromHex_hex, fromUTF8_toUTF8, hpe, hpy, h1, h2, h3,
    ha, Option.bind, Except.toOption, Json.getArr?, jsonStrIs, beq_self_eq_true, Bool.and_self, ite_true, pure,
    Except.pure]

/-- The Commit record of a window that registers no value. -/
def commitValue : Json := Json.mkObj [("format", .str COMMIT_FORMAT), ("registered", .arr #[])]

theorem commit_plain : PlainAt PyJson.DEPTH commitValue :=
  Ledger.plainAt_mono (Ledger.plainAt_built (d := 1) (ks := ["format", "registered"]) rfl (by decide) (by decide)
    (by
      intro q hq
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl
      · exact .str 0 (by decide)
      · exact .arr 0 fun x hx => by simp at hx)) (by decide)

/-- **`commitOf?` reads the empty registration's text.** -/
theorem commit_empty {text : String} (hc : Flock.canon commitValue = .ok text) :
    commitOf? (.str (Flock.hexOf text.toUTF8)) =
      some (Flock.Sha512.hash ((COMMIT_FORMAT.toUTF8.push 0) ++ text.toUTF8), #[]) :=
  commit_read commit_plain hc rfl rfl rfl

/-! ## The verifier's key -/

/-- **A key of the deployed parameters reads back from its JSON.** -/
theorem key_read {pub : ByteArray} (hs : pub.size = 80) :
    ChallengerKey.ofJson (ChallengerKey.toJson ⟨Lms.deployed, pub⟩) = .ok ⟨Lms.deployed, pub⟩ := by
  have hk : keys? (ChallengerKey.toJson ⟨Lms.deployed, pub⟩) = some #["H", "W", "public"] :=
    Built.keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide)
  have hH : nat? (getD (ChallengerKey.toJson ⟨Lms.deployed, pub⟩) "H") = some Lms.deployed.H := Files.nat?_nat _
  have hW : nat? (getD (ChallengerKey.toJson ⟨Lms.deployed, pub⟩) "W") = some Lms.deployed.W := Files.nat?_nat _
  have hp : hexOf? (getD (ChallengerKey.toJson ⟨Lms.deployed, pub⟩) "public") = some pub := by
    show (Flock.unhex (Flock.hexOf pub)).toOption = some pub
    rw [Flock.Hex.unhex_hexOf]
    rfl
  have hm : Lms.Params.make 64 Lms.deployed.W Lms.deployed.H false = .ok Lms.deployed := rfl
  unfold ChallengerKey.ofJson
  rw [hk, hH, hW, hp, beq_self_eq_true]
  simp only [hm, hs, PUBLIC_BYTES, beq_self_eq_true, bind, Except.bind, pure, Except.pure]

/-- A key `Lms.key` derives at the deployed parameters has a public key of 80 bytes. -/
theorem pub_size {I seed : ByteArray} {k : Lms.PrivateKey} (hk : Lms.key Lms.deployed I seed = .ok k) :
    k.publicKey.size = 80 := by
  obtain ⟨-, hIk, -, hIs, -, -, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hk
  have hroot : (k.tree[1]!).size = Lms.deployed.N :=
    Properties.CertifierDevice.Proofs.Lms.node_size hk (by decide) 1 le_rfl (by decide)
  subst hIk
  show k.publicKey.size = 16 + Lms.deployed.N
  simp only [Lms.PrivateKey.publicKey, ByteArray.size_append, hIs, hroot]

/-! ## The two halves of `credited` -/

/-- **`creditedRoot` on a window of no call and no round**: the verifier's key and the sample root of the salt, the
Commit record's digest and the receipts' root, with no call credited. -/
theorem root_ok {key salt commit draw : Json} {k : ChallengerKey} {s rc : ByteArray} {cd : ByteArray × Array Json}
    (hk : ChallengerKey.ofJson key = .ok k) (hs : digest? salt = some s) (hc : commitOf? commit = some cd)
    (ho : PouwCheck.Credit.isObj draw = true) (hr : digest? (PouwCheck.get draw "receipts") = some rc) :
    Internal.creditedRoot key salt (.arr #[]) .null commit (.arr #[]) draw =
      .ok (k, [], PouwCommit.tagged DRAW_TAG [.bytes (callsRoot s [] none (some cd.1)), .bytes rc]) := by
  unfold Internal.creditedRoot
  simp only [hk, hs, hc, ho, hr, Except.mapError, bind, Except.bind, pure, Except.pure, List.zipIdx_nil,
    List.forIn_nil, Option.elim, ite_true, List.append_nil]
  rfl

/-- **`creditedDraw` on a sample whose session released its one challenge after the sample root.** -/
theorem draw_ok {ledger pins main draw : Json} {k : ChallengerKey} {root : ByteArray} {c : LiveCommitment}
    {o : LiveOpening} (hroot : PouwCheck.str? (PouwCheck.get draw "root") = some (Flock.hexOf root))
    (hsess : PouwCheck.Credit.drawSession ledger k pins main (PouwCheck.get draw "session") (Flock.hexOf root) =
      .ok (c, [(PouwWindow.DRAW, [root])]))
    (hopen : LiveOpening.ofJson (PouwCheck.get draw "coin") = .ok o)
    (hlive : liveReason c o PouwWindow.DRAW 0 root = .ok none) :
    Internal.creditedDraw ledger k pins main draw root = .ok () := by
  have hl : (([(PouwWindow.DRAW, [root])] : Roots).lookup PouwWindow.DRAW).getD [] = [root] := by
    rw [List.lookup_cons, beq_self_eq_true]
    rfl
  unfold Internal.creditedDraw
  simp only [hroot, hsess, hopen, hlive, hl, beq_self_eq_true, bind, Except.bind, pure, Except.pure,
    List.isEmpty_cons, ite_false, Bool.false_eq_true]
  rfl

/-! ## An honest play credited -/

theorem draw_name : isName PouwWindow.DRAW = true := by
  rw [isName, String.all_bool_eq]
  decide

theorem draw_plain : PlainStr PouwWindow.DRAW := by decide

theorem draw_short : PouwWindow.DRAW.length ≤ 128 := by decide

/-- The sample root of a window of no call and no round: its salt, its Commit record's text and its receipts' root. -/
def rootOf (salt receipts : ByteArray) (text : String) : ByteArray :=
  PouwCommit.tagged DRAW_TAG [.bytes (callsRoot salt [] none
    (some (Flock.Sha512.hash ((COMMIT_FORMAT.toUTF8.push 0) ++ text.toUTF8)))), .bytes receipts]

theorem rootOf_size (salt receipts : ByteArray) (text : String) : (rootOf salt receipts text).size = 64 := by
  show (Flock.Sha512.hash _).size = 64
  exact hash_size _

/-- The window's sample: the play's opened challenge, the receipts' root, the sample root, and the session's files. -/
def drawOf (p : Play) (receipts : ByteArray) : Json :=
  Json.mkObj [("coin", p.opening.toJson), ("receipts", .str (Flock.hexOf receipts)),
    ("root", .str (Flock.hexOf p.window)), ("session", p.session)]

/-- **`credited` accepts an honest play of the sample's family**: a window of no call and no round, its Commit record
the text `text` `commitOf?` reads, under the play's key, when the play's one challenge answers the window's sample
root and its session is opened on that root. No call is credited. -/
theorem credited_play {I : ByteArray} {p : Play} (h : p.Honest I) {salt receipts : ByteArray} {text : String}
    (hl : p.label = PouwWindow.DRAW) (hs : salt.size = 64) (hr : receipts.size = 64) (hps : p.pub.size = 80)
    (hc : commitOf? (.str (Flock.hexOf text.toUTF8)) =
      some (Flock.Sha512.hash ((COMMIT_FORMAT.toUTF8.push 0) ++ text.toUTF8), #[]))
    (ha : p.after = rootOf salt receipts text) (hw : p.window = rootOf salt receipts text) :
    Internal.credited p.key.toJson (.arr #[.str p.openText, .str p.endText]) (.arr #[.str p.pinText])
      (.arr #[.str (Flock.hexOf p.pin)]) (.str (Flock.hexOf salt)) (.arr #[]) .null
      (.str (Flock.hexOf text.toUTF8)) (.arr #[]) (drawOf p receipts) = .ok [] := by
  obtain ⟨ht, hwo⟩ := Properties.Challenger.Withheld.trusted h
  have g1 : PouwCheck.get (drawOf p receipts) "receipts" = .str (Flock.hexOf receipts) := (get_getD _ _).trans rfl
  have g2 : PouwCheck.get (drawOf p receipts) "root" = .str (Flock.hexOf p.window) := (get_getD _ _).trans rfl
  have g3 : PouwCheck.get (drawOf p receipts) "session" = p.session := (get_getD _ _).trans rfl
  have g4 : PouwCheck.get (drawOf p receipts) "coin" = p.opening.toJson := (get_getD _ _).trans rfl
  have hroot := root_ok (draw := drawOf p receipts) (key_read hps) (digest_hex hs) hc rfl
    (by
      rw [g1]
      exact digest_hex hr)
  have hsess : PouwCheck.Credit.drawSession (.arr #[.str p.openText, .str p.endText]) p.key (.arr #[.str p.pinText])
      (.arr #[.str (Flock.hexOf p.pin)]) (PouwCheck.get (drawOf p receipts) "session")
      (Flock.hexOf (rootOf salt receipts text)) =
        .ok (p.commitment, [(PouwWindow.DRAW, [rootOf salt receipts text])]) := by
    have hd := Properties.Pouw.Credit.SecurityProofs.drawSession_honest ht hwo
    rw [hl, ha, hw] at hd
    rw [g3]
    exact hd
  have hlive : liveReason p.commitment p.opening p.label 0 p.after = .ok none :=
    Files.liveReason_one p.after h.domain h.coinSize h.saltSize
  rw [hl, ha] at hlive
  have hdraw := draw_ok (draw := drawOf p receipts) (root := rootOf salt receipts text)
    (by
      rw [g2, hw]
      rfl)
    hsess
    (by
      rw [g4]
      exact Files.opening_ofJson _ _ _ _ _ _)
    hlive
  unfold Internal.credited
  rw [show Internal.creditedRoot p.key.toJson (.str (Flock.hexOf salt)) (.arr #[]) .null
      (.str (Flock.hexOf text.toUTF8)) (.arr #[]) (drawOf p receipts) =
        .ok (p.key, [], rootOf salt receipts text) from hroot]
  simp only [bind, Except.bind, hdraw]
  rfl

end Properties.Pouw.Credit.Withheld

namespace Properties.Pouw.Credit.SecurityProofs

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger

/-- **`DrawWithheld`'s honest lemma**: `credited` accepts at the honest deployed instance. A key `Lms.keygen` derives
at the deployed LMS parameters signs an honest session withheld (`Properties.Challenger.Withheld.Play.Honest`) of
the family `pouw/draw`: the auditor's device commits to one challenge, records the commitment, logs the window's
sample root and the release after it and closes its log, and the auditor's ledger opens the session on that root and
ends it closed. The window is the smallest there is: no call, no round, and a Commit record that registers no
value. `credited` under that key accepts it, crediting no call. -/
theorem drawWithheld_honest :
    ∃ (seed : ByteArray) (k : Lms.PrivateKey) (ledger pins main salt calls served commit rounds draw : Json)
      (got : List Nat),
      Lms.keygen Lms.deployed seed = .ok k ∧
        PouwCheck.Credit.Internal.credited (ChallengerKey.toJson ⟨Lms.deployed, k.publicKey⟩) ledger pins main salt
          calls served commit rounds draw = .ok got := by
  obtain ⟨k, hkg, I, hk⟩ := Properties.Challenger.Withheld.keygen_ok (seed := ⟨Array.replicate 64 0⟩)
    (Properties.Challenger.Withheld.zeros_size 64)
  obtain ⟨text, htext⟩ := FlockComplete.JsonRT.canon_ok_some _
    (Properties.Challenger.Entry.good_of_plain Withheld.commit_plain)
  obtain ⟨p, hpub, hlab, -, haft, hwin, -, h⟩ := Properties.Challenger.Withheld.exists_honest hk Withheld.draw_name
    Withheld.draw_plain Withheld.draw_short 0 (Properties.Challenger.Withheld.zeros_size 64)
    (Properties.Challenger.Withheld.zeros_size 192)
    (Withheld.rootOf_size ⟨Array.replicate 64 0⟩ ⟨Array.replicate 64 0⟩ text)
    (Withheld.rootOf_size ⟨Array.replicate 64 0⟩ ⟨Array.replicate 64 0⟩ text)
    (Properties.Challenger.Withheld.zeros_size 20) .empty .empty
  have hps : p.pub.size = 80 := by
    rw [hpub]
    exact Withheld.pub_size hk
  have hcr := Withheld.credited_play h hlab (Properties.Challenger.Withheld.zeros_size 64)
    (Properties.Challenger.Withheld.zeros_size 64) hps (Withheld.commit_empty htext) haft hwin
  have hkey : p.key = ⟨Lms.deployed, k.publicKey⟩ := congrArg (ChallengerKey.mk Lms.deployed) hpub
  rw [hkey] at hcr
  exact ⟨_, k, _, _, _, _, _, _, _, _, _, _, hkg, hcr⟩

end Properties.Pouw.Credit.SecurityProofs
