module

public import Proofs.Properties.Challenger.Withheld
public import Proofs.Pouw.Credit
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# An honest window credited: `credited` accepts under a key `Lms.keygen` derives

`DrawWithheld` says what a credited window's sample rests on: the auditor's ledger opens a session withheld on the
window's sample root. This file builds a window `Internal.credited` accepts, on variables, so the property is about
something: a window of one call and one round, its manifest row the fourteen fields the serving process writes with
the credit bit set, whose Commit record registers no value, with the honest session withheld of
`Properties.Challenger.Withheld` opened on its sample root. `credited` returns call 0 of it.

* `pyFromHex_hex`, `digest_hex`, `isHexBytes_hex`: Python's `bytes.fromhex` reads lowercase hex back, a 64-byte
  digest's hex is a digest, and the hex of at least one byte is `credit._HEX`'s.
* `commit_read`: `commitOf?` reads the canonical text of a plain record of the registration format with no
  registered value as its digest under the format.
* `key_read`: `ChallengerKey.ofJson` reads a key of the deployed parameters back from its JSON.
* `root_one`, `sample_ok`: the two halves of `credited`, each from what its readers return: the sample root of a
  window of one credited call and one round, and the sample's session.
* `credited_play`: `credited` accepts an honest play of the family `pouw/draw` whose challenge answers the window's
  sample root and whose session the ledger opens on it, and returns the calls `creditedRoot` names.
* `CallRow`, `roundOf`: a credited manifest row and a round's entry as their writers build them, with what
  `creditedRoot` reads of each (`row_plain`, `row_index`, `row_bit`, `row_z`, `round_root`, `round_coin`).
* `credited_one`: `credited_play` on the window of one call.
* `sampleWithheld_honest`: the same at a key `Lms.keygen` derives at the deployed parameters.

`credited` is one step of the window's check: it reads a round's opened challenge and checks it against nothing,
which is `roundsCheck`'s to do, and what it returns is the credit bits whatever the proofs' outcome. So
`sampleWithheld_honest` shows that this step returns call 0, and not that the window's check credits it.

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
open Properties.Certifier.Bytes (hash_size)
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
  simp only [hm, hs, PUBLIC_BYTES, beq_self_eq_true, ite_true, bind, Except.bind, pure, Except.pure]

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

/-- **`creditedRoot` on a window of one credited call and one round**: for any manifest row that is a plain object,
is call 0, has its credit bit set and commits its product's table, and any round whose root is a digest and whose
opened challenge reads, the verifier's key, call 0, and the sample root of the salt, the row, the Commit record's
digest, the receipts' root, and the round's root and challenge. -/
theorem root_one {key salt commit sample row x : Json} {k : ChallengerKey} {s rc z rr : ByteArray}
    {cd : ByteArray × Array Json} {o : LiveOpening} (hk : ChallengerKey.ofJson key = .ok k)
    (hs : digest? salt = some s) (hc : commitOf? commit = some cd) (ho : PouwCheck.Credit.isObj sample = true)
    (hr : digest? (PouwCheck.get sample "receipts") = some rc) (h1 : PouwCheck.Credit.isObj row = true)
    (h2 : PouwCheck.Credit.plain PouwCheck.Credit.DEPTH row = true)
    (h3 : PouwWindow.intOf? (PouwCheck.get row "index") = some 0)
    (h4 : PouwCheck.get row PouwCheck.Credit.CREDIT = .bool true)
    (h5 : digest? (PouwCheck.get (PouwCheck.get row "tables") "z") = some z)
    (h6 : PouwCheck.Credit.isObj x = true) (h7 : digest? (PouwCheck.get x "root") = some rr)
    (h8 : LiveOpening.ofJson (PouwCheck.get x "coin") = .ok o) :
    Internal.creditedRoot key salt (.arr #[row]) .null commit (.arr #[x]) sample =
      .ok (k, [0], PouwCommit.tagged DRAW_TAG
        [.bytes (callsRoot s [row] none (some cd.1)), .bytes rc, .bytes rr, .bytes o.coin]) := by
  unfold Internal.creditedRoot
  simp only [hk, hs, hc, ho, hr, h1, h2, h3, h4, h5, h6, h7, h8, Except.mapError, bind, Except.bind, pure,
    Except.pure, List.zipIdx_cons, List.zipIdx_nil, List.forIn_cons, List.forIn_nil, Option.elim, ite_true,
    Bool.and_self, Option.isNone_some, List.nil_append, List.cons_append]
  rfl

/-- **`creditedDraw` on a sample whose session released its one challenge after the sample root.** -/
theorem sample_ok {ledger pins main sample : Json} {k : ChallengerKey} {root : ByteArray} {c : LiveCommitment}
    {o : LiveOpening} (hroot : PouwCheck.str? (PouwCheck.get sample "root") = some (Flock.hexOf root))
    (hsess : PouwCheck.Credit.drawSession ledger k pins main (PouwCheck.get sample "session") (Flock.hexOf root) =
      .ok (c, [(PouwWindow.DRAW, [root])]))
    (hopen : LiveOpening.ofJson (PouwCheck.get sample "coin") = .ok o)
    (hlive : liveReason c o PouwWindow.DRAW 0 root = .ok none) :
    Internal.creditedDraw ledger k pins main sample root = .ok () := by
  have hl : (([(PouwWindow.DRAW, [root])] : Roots).lookup PouwWindow.DRAW).getD [] = [root] := by
    rw [List.lookup_cons, beq_self_eq_true]
    rfl
  unfold Internal.creditedDraw
  simp only [hroot, hsess, hopen, hlive, hl, beq_self_eq_true, bind, Except.bind, pure, Except.pure,
    List.isEmpty_cons, ite_false, Bool.false_eq_true]
  rfl

/-! ## An honest play credited -/

theorem family_name : isName PouwWindow.DRAW = true := by
  rw [isName, String.all_bool_eq]
  decide

theorem family_plain : PlainStr PouwWindow.DRAW := by decide

theorem family_short : PouwWindow.DRAW.length ≤ 128 := by decide

/-- The window's sample: the play's opened challenge, the receipts' root, the sample root, and the session's files. -/
def sampleOf (p : Play) (receipts : ByteArray) : Json :=
  Json.mkObj [("coin", p.opening.toJson), ("receipts", .str (Flock.hexOf receipts)),
    ("root", .str (Flock.hexOf p.window)), ("session", p.session)]

theorem sample_receipts (p : Play) (receipts : ByteArray) :
    PouwCheck.get (sampleOf p receipts) "receipts" = .str (Flock.hexOf receipts) :=
  (get_getD _ _).trans rfl

/-- **`credited` accepts an honest play of the sample's family**: under the play's key, a window whose sample root
`creditedRoot` computes as `root`, when the play's one challenge answers `root` and its session is opened on it. It
returns the calls `creditedRoot` names. -/
theorem credited_play {I : ByteArray} {p : Play} (h : p.Honest I) {salt calls served commit rounds : Json}
    {receipts root : ByteArray} {out : List Nat} (hl : p.label = PouwWindow.DRAW)
    (hroot : Internal.creditedRoot p.key.toJson salt calls served commit rounds (sampleOf p receipts) =
      .ok (p.key, out, root))
    (ha : p.after = root) (hw : p.window = root) :
    Internal.credited p.key.toJson (.arr #[.str p.openText, .str p.endText]) (.arr #[.str p.pinText])
      (.arr #[.str (Flock.hexOf p.pin)]) salt calls served commit rounds (sampleOf p receipts) = .ok out := by
  obtain ⟨ht, hwo⟩ := Properties.Challenger.Withheld.trusted h
  have g2 : PouwCheck.get (sampleOf p receipts) "root" = .str (Flock.hexOf p.window) := (get_getD _ _).trans rfl
  have g3 : PouwCheck.get (sampleOf p receipts) "session" = p.session := (get_getD _ _).trans rfl
  have g4 : PouwCheck.get (sampleOf p receipts) "coin" = p.opening.toJson := (get_getD _ _).trans rfl
  have hsess : PouwCheck.Credit.drawSession (.arr #[.str p.openText, .str p.endText]) p.key (.arr #[.str p.pinText])
      (.arr #[.str (Flock.hexOf p.pin)]) (PouwCheck.get (sampleOf p receipts) "session") (Flock.hexOf root) =
        .ok (p.commitment, [(PouwWindow.DRAW, [root])]) := by
    have hd := Properties.Pouw.Credit.SecurityProofs.drawSession_honest ht hwo
    rw [hl, ha, hw] at hd
    rw [g3]
    exact hd
  have hlive : liveReason p.commitment p.opening p.label 0 p.after = .ok none :=
    Files.liveReason_one p.after h.domain h.coinSize h.saltSize
  rw [hl, ha] at hlive
  have hsample := sample_ok (sample := sampleOf p receipts) (root := root)
    (by
      rw [g2, hw]
      rfl)
    hsess
    (by
      rw [g4]
      exact Files.opening_ofJson _ _ _ _ _ _)
    hlive
  unfold Internal.credited
  rw [hroot]
  simp only [bind, Except.bind, hsample]
  rfl

/-! ## A window of one call -/

/-- An object built from increasing keys is plain one level deeper than its values. -/
theorem plain_mkObj {l : List (String × Json)} {d : Nat} (hs : (l.map (·.1)).Pairwise (· < ·))
    (h : l.all (fun q => PouwCheck.Credit.plain d q.2) = true) :
    PouwCheck.Credit.plain (d + 1) (Json.mkObj l) = true := by
  rw [Json.mkObj, PouwCheck.Credit.plain, ← Array.all_toList, Std.TreeMap.Raw.toList_toArray, Built.toList_mkObj hs]
  exact h

/-- **A credited manifest row as the serving process writes it**, the first call of its window: the call's step,
layer and module, its sizes, its leaves, the digests of its operand's and its product's tables, and its bound. -/
structure CallRow where
  (step layer served rows k n base units leaf m : Int)
  module : String
  (a z : ByteArray)

/-- The row's tables: its operand's and its product's, each a digest in hex. -/
def CallRow.tables (r : CallRow) : Json :=
  Json.mkObj [("a", .str (Flock.hexOf r.a)), ("z", .str (Flock.hexOf r.z))]

/-- The row's fourteen fields in key order: call 0, its credit bit set. -/
def CallRow.items (r : CallRow) : List (String × Json) :=
  [("base", .num ⟨r.base, 0⟩), ("bound", Json.mkObj [("m", .num ⟨r.m, 0⟩), ("ok", .bool true)]),
    ("credit", .bool true), ("index", .num ⟨0, 0⟩), ("k", .num ⟨r.k, 0⟩), ("layer", .num ⟨r.layer, 0⟩),
    ("leaves", .arr #[.num ⟨r.leaf, 0⟩, .null]), ("module", .str r.module), ("n", .num ⟨r.n, 0⟩),
    ("rows", .num ⟨r.rows, 0⟩), ("served", .num ⟨r.served, 0⟩), ("step", .num ⟨r.step, 0⟩), ("tables", r.tables),
    ("units", .num ⟨r.units, 0⟩)]

def CallRow.toJson (r : CallRow) : Json := Json.mkObj r.items

theorem CallRow.sorted (r : CallRow) : (r.items.map (·.1)).Pairwise (· < ·) := by
  simp only [CallRow.items, List.map_cons, List.map_nil]
  decide

/-- A read of the row is its list's. -/
theorem row_get (r : CallRow) (k : String) : PouwCheck.get r.toJson k = (r.items.lookup k).getD .null :=
  (get_getD _ _).trans (Built.getD_mkObj r.sorted k)

theorem row_plain (r : CallRow) : PouwCheck.Credit.plain PouwCheck.Credit.DEPTH r.toJson = true := by
  have hb : PouwCheck.Credit.plain 63 (Json.mkObj [("m", .num ⟨r.m, 0⟩), ("ok", .bool true)]) = true :=
    plain_mkObj (d := 62) (by simp only [List.map_cons, List.map_nil]; decide)
      (by simp only [List.all_cons, PouwCheck.Credit.plain, BEq.rfl, List.all_nil, Bool.and_self])
  have ht : PouwCheck.Credit.plain 63 r.tables = true :=
    plain_mkObj (d := 62) (by simp only [List.map_cons, List.map_nil]; decide)
      (by simp only [List.all_cons, PouwCheck.Credit.plain, List.all_nil, Bool.and_self])
  show PouwCheck.Credit.plain (63 + 1) (Json.mkObj r.items) = true
  refine plain_mkObj r.sorted ?_
  simp only [CallRow.items, List.all_cons, PouwCheck.Credit.plain, BEq.rfl, hb, List.size_toArray, List.length_cons,
    List.length_nil, Nat.zero_add, Nat.reduceAdd, List.all_toArray', List.all_nil, Bool.and_self, ht]

theorem row_index (r : CallRow) : PouwWindow.intOf? (PouwCheck.get r.toJson "index") = some 0 := by
  rw [row_get]
  rfl

theorem row_bit (r : CallRow) : PouwCheck.get r.toJson PouwCheck.Credit.CREDIT = .bool true := by
  rw [row_get]
  rfl

/-- **The row commits its product's table**: the digest `creditedRoot` asks of a credited call. -/
theorem row_z (r : CallRow) (hz : r.z.size = 64) :
    digest? (PouwCheck.get (PouwCheck.get r.toJson "tables") "z") = some r.z := by
  have ht : PouwCheck.get r.toJson "tables" = r.tables := (row_get r _).trans rfl
  have hg : PouwCheck.get r.tables "z" = .str (Flock.hexOf r.z) := (get_getD _ _).trans rfl
  exact (congrArg (fun j => digest? (PouwCheck.get j "z")) ht).trans ((congrArg digest? hg).trans (digest_hex hz))

/-- A round's entry in the round log: its root, and its challenge as the challenger opens it, with no path. -/
def roundOf (root coin csalt : ByteArray) : Json :=
  Json.mkObj [("coin", LiveOpening.toJson ⟨0, PouwWindow.ROUND, 0, root, coin, csalt, []⟩),
    ("root", .str (Flock.hexOf root))]

theorem round_root {root : ByteArray} (coin csalt : ByteArray) (h : root.size = 64) :
    digest? (PouwCheck.get (roundOf root coin csalt) "root") = some root := by
  have hg : PouwCheck.get (roundOf root coin csalt) "root" = .str (Flock.hexOf root) := (get_getD _ _).trans rfl
  exact (congrArg digest? hg).trans (digest_hex h)

theorem round_coin (root coin csalt : ByteArray) :
    LiveOpening.ofJson (PouwCheck.get (roundOf root coin csalt) "coin") =
      .ok ⟨0, PouwWindow.ROUND, 0, root, coin, csalt, []⟩ := by
  have hg : PouwCheck.get (roundOf root coin csalt) "coin" =
      LiveOpening.toJson ⟨0, PouwWindow.ROUND, 0, root, coin, csalt, []⟩ := (get_getD _ _).trans rfl
  exact (congrArg LiveOpening.ofJson hg).trans (Files.opening_ofJson _ _ _ _ _ _)

/-- The sample root of a window of one call and one round: its salt, its row, its Commit record's text, its
receipts' root, and the round's root and challenge. -/
def rootOne (salt receipts : ByteArray) (text : String) (r : CallRow) (round coin : ByteArray) : ByteArray :=
  PouwCommit.tagged DRAW_TAG [.bytes (callsRoot salt [r.toJson] none
    (some (Flock.Sha512.hash ((COMMIT_FORMAT.toUTF8.push 0) ++ text.toUTF8)))), .bytes receipts, .bytes round,
    .bytes coin]

theorem rootOne_size (salt receipts : ByteArray) (text : String) (r : CallRow) (round coin : ByteArray) :
    (rootOne salt receipts text r round coin).size = 64 := by
  show (Flock.Sha512.hash _).size = 64
  exact hash_size _

/-- **`credited` returns call 0 of an honest play's window of one call and one round**: the row `r`, credited, and a
round of root `round` and challenge `coin`, the Commit record the text `text` `commitOf?` reads, when the play's one
challenge answers the window's sample root and its session is opened on that root. -/
theorem credited_one {I : ByteArray} {p : Play} (h : p.Honest I) {salt receipts round : ByteArray} {text : String}
    (r : CallRow) (coin csalt : ByteArray) (hl : p.label = PouwWindow.DRAW) (hs : salt.size = 64)
    (hr : receipts.size = 64) (hz : r.z.size = 64) (hrd : round.size = 64) (hps : p.pub.size = 80)
    (hc : commitOf? (.str (Flock.hexOf text.toUTF8)) =
      some (Flock.Sha512.hash ((COMMIT_FORMAT.toUTF8.push 0) ++ text.toUTF8), #[]))
    (ha : p.after = rootOne salt receipts text r round coin)
    (hw : p.window = rootOne salt receipts text r round coin) :
    Internal.credited p.key.toJson (.arr #[.str p.openText, .str p.endText]) (.arr #[.str p.pinText])
      (.arr #[.str (Flock.hexOf p.pin)]) (.str (Flock.hexOf salt)) (.arr #[r.toJson]) .null
      (.str (Flock.hexOf text.toUTF8)) (.arr #[roundOf round coin csalt]) (sampleOf p receipts) = .ok [0] := by
  have hroot := root_one (sample := sampleOf p receipts) (key_read hps) (digest_hex hs) hc rfl
    (by
      rw [sample_receipts]
      exact digest_hex hr)
    rfl (row_plain r) (row_index r) (row_bit r) (row_z r hz) rfl (round_root coin csalt hrd)
    (round_coin round coin csalt)
  exact credited_play h hl (root := rootOne salt receipts text r round coin) hroot ha hw

/-- A call of one row of 64 by 4096 in one unit, its tables' digests zero bytes. -/
def call : CallRow where
  step := 0
  layer := 0
  served := 0
  rows := 1
  k := 64
  n := 4096
  base := 0
  units := 1
  leaf := 0
  m := 1
  module := "q_proj"
  a := ⟨Array.replicate 64 0⟩
  z := ⟨Array.replicate 64 0⟩

end Properties.Pouw.Credit.Withheld

namespace Properties.Pouw.Credit.SecurityProofs

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger

/-- **`DrawWithheld`'s honest lemma, on a window of one call**: `credited` returns call 0 at the honest deployed
instance. A key `Lms.keygen` derives at the deployed LMS parameters signs an honest session withheld
(`Properties.Challenger.Withheld.Play.Honest`) of the family `pouw/draw`: the auditor's device commits to one
challenge, records the commitment, logs the window's sample root and the release after it and closes its log, and
the auditor's ledger opens the session on that root and ends it closed. The window holds one call and one round:
the manifest's one row is a call's fourteen fields as the serving process writes them, call 0 with its credit bit
set and its product's table committed, the round log's one entry is a round's root and its opened challenge, and
the Commit record registers no value. `credited` under that key accepts the window and returns call 0. It is one
step of the window's check, which reads the round's challenge and does not check it: that the window's check
credits the call is not shown here. -/
theorem sampleWithheld_honest :
    ∃ (seed : ByteArray) (k : Lms.PrivateKey) (ledger pins main salt calls served commit rounds sample : Json),
      Lms.keygen Lms.deployed seed = .ok k ∧
        PouwCheck.Credit.Internal.credited (ChallengerKey.toJson ⟨Lms.deployed, k.publicKey⟩) ledger pins main salt
          calls served commit rounds sample = .ok [0] := by
  obtain ⟨k, hkg, I, hk⟩ := Properties.Challenger.Withheld.keygen_ok (seed := ⟨Array.replicate 64 0⟩)
    (Properties.Challenger.Withheld.zeros_size 64)
  obtain ⟨text, htext⟩ := FlockComplete.JsonRT.canon_ok_some _
    (Properties.Challenger.Entry.good_of_plain Withheld.commit_plain)
  obtain ⟨p, hpub, hlab, -, haft, hwin, -, h⟩ := Properties.Challenger.Withheld.exists_honest hk Withheld.family_name
    Withheld.family_plain Withheld.family_short 0 (Properties.Challenger.Withheld.zeros_size 64)
    (Properties.Challenger.Withheld.zeros_size 192)
    (Withheld.rootOne_size ⟨Array.replicate 64 0⟩ ⟨Array.replicate 64 0⟩ text Withheld.call
      ⟨Array.replicate 64 0⟩ ⟨Array.replicate 64 0⟩)
    (Withheld.rootOne_size ⟨Array.replicate 64 0⟩ ⟨Array.replicate 64 0⟩ text Withheld.call
      ⟨Array.replicate 64 0⟩ ⟨Array.replicate 64 0⟩)
    (Properties.Challenger.Withheld.zeros_size 20) .empty .empty
  have hps : p.pub.size = 80 := by
    rw [hpub]
    exact Withheld.pub_size hk
  have hcr := Withheld.credited_one h Withheld.call ⟨Array.replicate 64 0⟩ ⟨Array.replicate 192 0⟩ hlab
    (Properties.Challenger.Withheld.zeros_size 64) (Properties.Challenger.Withheld.zeros_size 64)
    (Properties.Challenger.Withheld.zeros_size 64) (Properties.Challenger.Withheld.zeros_size 64) hps
    (Withheld.commit_empty htext) haft hwin
  have hkey : p.key = ⟨Lms.deployed, k.publicKey⟩ := congrArg (ChallengerKey.mk Lms.deployed) hpub
  rw [hkey] at hcr
  exact ⟨_, k, _, _, _, _, _, _, _, _, _, hkg, hcr⟩

end Properties.Pouw.Credit.SecurityProofs
