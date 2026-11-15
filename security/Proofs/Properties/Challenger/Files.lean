module

public import Proofs.Properties.Challenger.Built
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# A session's files read back

The files of the smallest session a challenger closes, as the challenger writes them, read back as what it wrote:

* the coin commitment of one family of one coin (`one`) from its JSON (`commitment_ofJson`, over `layout_one`);
* a record of a commitment that reads back (`record_ofJson`), a close with no end (`close_ofJson`), and an opening
  with no path (`opening_ofJson`);
* the one coin opens against the commitment whose root is its leaf (`coinRoot`, `liveReason_one`): a tree of one
  leaf has no sibling, so the leaf is the root.

Each proof rewrites the reader's checks one at a time, by the read its JSON gives.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private hexAt from Spot.Challenger
open private hexOf? from Spot.ChallengerRecord
open private hexAt? from Spot.ChallengerSession

namespace Properties.Challenger.Files

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger

theorem nat?_nat (n : Nat) : nat? (Lean.toJson n) = some n := by
  simp [nat?, int?, Lean.toJson, Lean.JsonNumber.fromNat]

theorem int?_int (i : Int) : int? (Lean.toJson i) = some i := by
  simp [int?, Lean.toJson, Lean.JsonNumber.fromInt]

theorem ite_ff {α : Type} (a b : α) : (if false = true then a else b) = b := rfl

theorem mapM_one {α β : Type} {f : α → Flock.V β} {a : α} {b : β} (h : f a = .ok b) : [a].mapM f = .ok [b] := by
  rw [List.mapM_cons, List.mapM_nil, h]
  rfl

theorem layout_one {scheme label : String} (epoch : Nat) (hs : isName scheme = true) (hl : isName label = true) :
    layoutOf (.str scheme) (Lean.toJson epoch) (familiesJson [(label, 1)]) = .ok (scheme, epoch, [(label, 1)]) := by
  unfold layoutOf
  have h1 : (match Json.str scheme with | .str s => some s | _ => none).filter isName = some scheme := by
    show (some scheme).filter isName = some scheme
    simp only [Option.filter, hs, ↓reduceIte]
  have h2 : nat? (Lean.toJson epoch) = some epoch := nat?_nat epoch
  rw [h1, h2]
  dsimp only [familiesJson]
  rw [List.map_toArray, List.map_cons, List.map_nil, List.toList_toArray]
  have hE : (#[Json.arr #[Json.str label, Lean.toJson (1 : Nat)]] : Array Json).isEmpty = false := rfl
  rw [hE, ite_ff, mapM_one (b := (label, 1)) ?_]
  · rfl
  · show (if (isName label && decide (1 ≤ 1)) = true then pure (label, 1) else throw NOT_LAYOUT) = _
    rw [hl]
    rfl

/-- The coin commitment of one family of one coin. -/
def one (scheme label : String) (epoch : Nat) (root : ByteArray) : LiveCommitment := ⟨scheme, epoch, [(label, 1)], root⟩

theorem one_keys (scheme label : String) (epoch : Nat) (root : ByteArray) :
    keys? (one scheme label epoch root).toJson = some #["epoch", "families", "root", "scheme"] :=
  Built.keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide)

/-- **The coin commitment of one family of one coin reads back from its JSON.** -/
theorem commitment_ofJson {scheme label : String} (epoch : Nat) {root : ByteArray} (hs : isName scheme = true)
    (hl : isName label = true) (hr : root.size = 64) :
    LiveCommitment.ofJson (one scheme label epoch root).toJson = .ok (one scheme label epoch root) := by
  have hroot : hexAt (one scheme label epoch root).toJson "root" = some root := by
    show (Flock.unhex (Flock.hexOf root)).toOption = some root
    rw [Flock.Hex.unhex_hexOf]
    rfl
  have hlay : layoutOf (getD (one scheme label epoch root).toJson "scheme")
      (getD (one scheme label epoch root).toJson "epoch") (getD (one scheme label epoch root).toJson "families") =
        .ok (scheme, epoch, [(label, 1)]) := layout_one epoch hs hl
  have h64 : (root.size == 64) = true := by rw [hr]; rfl
  have hfam : getD (one scheme label epoch root).toJson "families" =
      Json.arr #[Json.arr #[Json.str label, Lean.toJson (1 : Nat)]] := by
    show familiesJson [(label, 1)] = _
    rw [familiesJson, List.map_toArray]
    rfl
  unfold LiveCommitment.ofJson
  rw [one_keys, hroot]
  dsimp only
  rw [hlay]
  dsimp only [bind, Except.bind]
  rw [h64, beq_self_eq_true, hfam]
  simp only [List.all_toArray]
  rfl

/-- **A record of a commitment that reads back reads back.** -/
theorem record_ofJson {c : LiveCommitment} {host : String} (sig : ByteArray)
    (hc : LiveCommitment.ofJson c.toJson = .ok c) (hh : isHost host = true) :
    ChallengerRecord.ofJson (ChallengerRecord.toJson ⟨c, host, sig⟩) = .ok ⟨c, host, sig⟩ := by
  have hk : keys? (ChallengerRecord.toJson ⟨c, host, sig⟩) = some #["commitment", "host", "signature"] :=
    Built.keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide)
  have hsig : hexOf? (getD (ChallengerRecord.toJson ⟨c, host, sig⟩) "signature") = some sig := by
    show (Flock.unhex (Flock.hexOf sig)).toOption = some sig
    rw [Flock.Hex.unhex_hexOf]
    rfl
  have hcm : LiveCommitment.ofJson (getD (ChallengerRecord.toJson ⟨c, host, sig⟩) "commitment") = .ok c := hc
  have hhost : (match getD (ChallengerRecord.toJson ⟨c, host, sig⟩) "host" with | .str s => s | _ => "") = host := rfl
  unfold ChallengerRecord.ofJson
  rw [hk, hsig, beq_self_eq_true]
  dsimp only [bind, Except.bind]
  rw [hcm]
  show (if isHost host = true then _ else _) = _
  rw [hh]
  rfl

/-- **A close with no end reads back from its JSON.** -/
theorem close_ofJson {session : String} {lines : Nat} {chain : ByteArray} (sig : ByteArray) (hc : chain.size = 64)
    (h1 : 1 ≤ lines) (h2 : lines ≤ MAX_LOG_LINES) (hs : isChallengePhaseId session = true) :
    ChallengerClose.ofJson (ChallengerClose.toJson ⟨session, lines, chain, none, sig⟩) =
      .ok ⟨session, lines, chain, none, sig⟩ := by
  have hk : keys? (ChallengerClose.toJson ⟨session, lines, chain, none, sig⟩) =
      some #["chain", "ended", "lines", "session", "signature"] :=
    Built.keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide)
  have hch : hexAt? (getD (ChallengerClose.toJson ⟨session, lines, chain, none, sig⟩) "chain") = some chain := by
    show (Flock.unhex (Flock.hexOf chain)).toOption = some chain
    rw [Flock.Hex.unhex_hexOf]
    rfl
  have hsig : hexAt? (getD (ChallengerClose.toJson ⟨session, lines, chain, none, sig⟩) "signature") = some sig := by
    show (Flock.unhex (Flock.hexOf sig)).toOption = some sig
    rw [Flock.Hex.unhex_hexOf]
    rfl
  have hl : nat? (getD (ChallengerClose.toJson ⟨session, lines, chain, none, sig⟩) "lines") = some lines :=
    nat?_nat lines
  have hok :
      (chain.size == 64 && decide (1 ≤ lines) && decide (lines ≤ MAX_LOG_LINES) && isChallengePhaseId session) = true := by
    rw [hc, hs, decide_eq_true h1, decide_eq_true h2]
    rfl
  unfold ChallengerClose.ofJson
  rw [hk, hch, hsig, hl, beq_self_eq_true]
  show (if (chain.size == 64 && decide (1 ≤ lines) && decide (lines ≤ MAX_LOG_LINES) && isChallengePhaseId session) = true
    then _ else _) = _
  rw [hok]
  rfl

/-- **An opening with no path reads back from its JSON.** -/
theorem opening_ofJson (epoch index : Int) (label : String) (after coin salt : ByteArray) :
    LiveOpening.ofJson (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) =
      .ok ⟨epoch, label, index, after, coin, salt, []⟩ := by
  have hk : keys? (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) =
      some #["after", "coin", "epoch", "index", "label", "path", "salt"] :=
    Built.keys?_mkObj (by simp only [List.map_cons, List.map_nil]; decide)
  have hx : ∀ b : ByteArray, (Flock.unhex (Flock.hexOf b)).toOption = some b := fun b => by
    rw [Flock.Hex.unhex_hexOf]
    rfl
  have ha : hexAt (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) "after" = some after := hx after
  have hc : hexAt (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) "coin" = some coin := hx coin
  have hs : hexAt (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) "salt" = some salt := hx salt
  have he : int? (getD (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) "epoch") = some epoch :=
    int?_int epoch
  have hi : int? (getD (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) "index") = some index :=
    int?_int index
  have hp : getD (LiveOpening.toJson ⟨epoch, label, index, after, coin, salt, []⟩) "path" = Json.arr #[] := by
    show Json.arr (([] : List ByteArray).toArray.map fun p => Json.str (Flock.hexOf p)) = _
    rw [List.map_toArray]
    rfl
  unfold LiveOpening.ofJson
  rw [hk, ha, hc, hs, he, hi, hp, beq_self_eq_true]
  rfl

/-- The one coin's leaf under the domain `dom`: the root of the commitment of one family of one coin. -/
def coinRoot (dom : ByteArray) (scheme : String) (coin salt : ByteArray) : ByteArray :=
  Flock.FV3.leaf dom 0 scheme (liveLeaf scheme coin salt) Flock.Sha512.hash

/-- **The one coin opens against the commitment whose root is its leaf.** -/
theorem liveReason_one {scheme label : String} {epoch : Nat} {coin salt dom : ByteArray} (after : ByteArray)
    (hd : liveDomain scheme epoch [(label, 1)] = .ok dom) (hc : coin.size = 64) (hs : salt.size = 192) :
    liveReason (one scheme label epoch (coinRoot dom scheme coin salt)) ⟨epoch, label, 0, after, coin, salt, []⟩ label 0
      after = .ok none := by
  have hpos : (one scheme label epoch (coinRoot dom scheme coin salt)).position label 0 = some 0 := by
    show (if (label == label) = true then (if (decide ((0 : Int) ≤ 0) && decide ((0 : Int) < (1 : Nat))) = true
      then some (0 + (0 : Int).toNat) else none) else _) = some 0
    rw [beq_self_eq_true]
    rfl
  have he : (((epoch : Nat) : Int) != ((epoch : Nat) : Int)) = false := bne_self_eq_false _
  have hl : (label != label || (0 : Int) != 0) = false := by
    rw [bne_self_eq_false]
    rfl
  have hsz : (coin.size != COIN_BYTES || salt.size != SALT_BYTES) = false := by
    rw [hc, hs]
    rfl
  have ha : (after != after) = false := by
    show (!(after == after)) = false
    rw [Steps.bytes_beq]
    rfl
  have hcl : (Flock.Registered.climbL dom
      (Flock.FV3.leaf dom 0 scheme (liveLeaf scheme coin salt) Flock.Sha512.hash) 0 0 [] ==
      coinRoot dom scheme coin salt) = true := Steps.bytes_beq _
  unfold liveReason
  rw [hpos]
  show (if (((epoch : Nat) : Int) != ((epoch : Nat) : Int)) = true then _ else _) = _
  rw [he, ite_ff]
  show (if (label != label || (0 : Int) != 0) = true then _ else _) = _
  rw [hl, ite_ff]
  show (if (coin.size != COIN_BYTES || salt.size != SALT_BYTES) = true then _ else _) = _
  rw [hsz, ite_ff]
  show (liveDomain scheme epoch [(label, 1)] >>= _) = _
  rw [hd]
  show (if (true && true && Flock.Registered.climbL dom
      (Flock.FV3.leaf dom 0 scheme (liveLeaf scheme coin salt) Flock.Sha512.hash) 0 0 [] ==
      coinRoot dom scheme coin salt) = true then _ else _) = _
  rw [hcl]
  show (if (after != after) = true then _ else _) = _
  rw [ha]
  rfl

end Properties.Challenger.Files
