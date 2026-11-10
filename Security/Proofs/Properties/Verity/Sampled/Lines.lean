module

public import Properties.Challenger
public import Proofs.Properties.Verity.Sampled.Trees
public import Proofs.Properties.Challenger.CanonLine
public import Proofs.Flock.Claims.Complete.Json.Hex
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# The log v2 reads is the one the device printed, but for openings' coins of one SHA-512

v2's check reads the device's log from the lines of the challenger session's release log (`Verity.Sampled.readLog`),
whose head, the coin commitment's line, is no coin tree's line (`treeLine_head`), so it reads the lines after the head
(`Verity.Sampled.readTail`). Where the device printed each release as the challenger does, the read is the device's
log release for release, but for a round opened's coins, which the read takes from a session's record: those hash as the
coins the device opened, so they are the device's or bytes of one SHA-512 with them.

* **What the device printed** (`Prints`, `PrintsLog`): each release as lines that are their values' `Flock.canon`
  texts (`Canon`), with the fields v2 reads: a family's coin as a root's line (`RootValue`) and its release's line
  (`PlainValue`), a tree's hello as its key, nonce, root and spec (`HelloValue`, `SpecValue`), and a round opened as its
  tree's root, stream, round, count, the bytes it answers and its coins' SHA-512 (`OpenValue`); and, where the run's
  release log reads as its coin commitment's (`releasedRootsOf`), the coins it released of a family after a root,
  `COIN_BYTES` bytes each in order, the family's coins in its log (`Holds`, `freshOf`). These are the properties the
  device's history at a play is to have.
* **A line reads back** (`CanonLine.event_getD`): where v2's line reader (`Spot.Challenger.event`) accepts a
  line that is a sorted value's `canon` text, it reads the value's fields (`str_of`, `nat_of`, `spec_of`). The reader's
  parse can't be seen into, so that it accepts a printed line is what `readTail`'s success shows: a line it refuses is
  no tree's (`treeLine_error`), and a root's line it refuses is pending with no root (`rootOf_error`), which no read
  survives (`pending_none`). The head is the commitment's value's text (`head_canon`), so where `releasedRootsOf`
  accepts a log, `readLog` reads it as `readTail` reads the lines after its head (`readLog_head`).
* **The trees v2 accepts** (`TreesOf`, `trees_log`): `coinTreesOf`'s acceptance gives each tree a root no other tree
  has, a spec of distinct streams, and each round opened once, so where a session's record passes `sessionTreeReason`,
  each round it gives coins of has the coins whose SHA-512 the round's line shows (`TreeLines`, `tree_lines`).
* **The induction** (`read_prints`, `read_run`): over what the device printed, `readTail` reads each release
  (`ReadOf`): its frame and the bytes it answered, and its coins where it read any, the device's, or, at a round opened,
  other bytes than the device's with their SHA-512 (`coinBytes`, `coinBytes_bits`). The trees it reads are the ones the
  device sampled, each its root's hex and its spec's value (`TreesRead`).
* **The openings without the sample's coins** (`opensRead`, `opens_read`): v2's read with none of the sample's coins
  placed reads the same frames, bytes and openings' coins (`readTail_agree`, `readLog_agree`), and verifies no
  signature. The first opening whose coins it reads other than the device's (`openCollision`) is a SHA-512 collision
  wherever v2's read of the device's log isn't the device's (`open_collision`).
-/

@[expose] public section

namespace Verity.Sampled.Lines

open Lean (Json)
open Spot Spot.Challenger
open FlockDeploy (JsonEquiv)
open Properties.Challenger (History OpensOn)
open Properties.Challenger.CanonLine (event_ok event_getD nat?_equiv)
open Properties.Challenger.CanonRead (equiv_str sorted_getD)

/-! ## What the device printed -/

/-- **The line `l` is the value `d`'s text**, as the challenger writes a line of its release log: `d` holds its keys in
order (`sortedKeys`) and `Flock.canon` prints it as `l`. -/
def Canon (d : Json) (l : String) : Prop := sortedKeys d = true ∧ Flock.canon d = .ok l

/-- A value whose `event` is a string other than a coin tree's (`TREE_EVENT`, `OPEN_EVENT`). -/
def PlainValue (d : Json) : Prop := ∃ e, getD d "event" = .str e ∧ e ≠ TREE_EVENT ∧ e ≠ OPEN_EVENT

/-- **A root's line**, of family `label` and the root `b`, as `Verity.Sampled.rootOf` reads it. -/
structure RootValue (d : Json) (label : String) (b : ByteArray) : Prop where
  plain : PlainValue d
  label : getD d "label" = .str label
  root : getD d "root" = .str (Flock.hexOf b)

/-- **A coin tree's spec** `spec`, as `Verity.Sampled.specOf` reads it: its streams in order, its rounds' log and its
block. -/
structure SpecValue (d : Json) (spec : Flock.CoinTree.Spec) : Prop where
  streams : getD d "streams" = .arr (spec.streams.map .str)
  roundsLog : nat? (getD d "rounds_log") = some spec.roundsLog
  block : nat? (getD d "block") = some spec.block

/-- **A coin tree's hello**, of spec `spec` under the nonce `ν`, with its root and key. -/
structure HelloValue (d : Json) (spec : Flock.CoinTree.Spec) (ν root key : ByteArray) : Prop where
  event : getD d "event" = .str TREE_EVENT
  key : getD d "key" = .str (Flock.hexOf key)
  nonce : getD d "nonce" = .str (Flock.hexOf ν)
  root : getD d "root" = .str (Flock.hexOf root)
  spec : SpecValue (getD d "spec") spec

/-- **A round opened** of the tree of root `root`: stream `stream`'s round `k`, its first `n` coins, after the bytes
`b`, the coins `cb` by their SHA-512. -/
structure OpenValue (d : Json) (root : ByteArray) (stream : String) (k n : ℕ) (b cb : ByteArray) : Prop where
  event : getD d "event" = .str OPEN_EVENT
  tree : getD d "tree" = .str (Flock.hexOf root)
  stream : getD d "stream" = .str stream
  round : nat? (getD d "round") = some k
  n : nat? (getD d "n") = some n
  after : getD d "after" = .str (Flock.hexOf b)
  coins : getD d "coins" = .str (sha512Hex cb)

/-- **The lines `lines` print the releases `log`**, after the trees `ts` sampled so far, each its root and spec: a
family's coin as a root's line and its release's line, a tree's hello as its line, which samples the tree, and a round
opened of the `i`-th tree sampled and its spec's `s`-th stream as its line, each line its value's text (`Canon`). -/
inductive Prints : List (ByteArray × Flock.CoinTree.Spec) → List String → List Release → Prop
  | nil (ts : List (ByteArray × Flock.CoinTree.Spec)) : Prints ts [] []
  | fresh {ts : List (ByteArray × Flock.CoinTree.Spec)} {lines : List String} {log : List Release} {d d' : Json}
      {l l' label : String} {b : ByteArray} {cs : List Bool} :
      Canon d l → RootValue d label b → Canon d' l' → PlainValue d' → Prints ts lines log →
      Prints ts (l :: l' :: lines) (⟨.fresh label, b, cs⟩ :: log)
  | hello {ts : List (ByteArray × Flock.CoinTree.Spec)} {lines : List String} {log : List Release} {d : Json}
      {l : String} {spec : Flock.CoinTree.Spec} {ν root key : ByteArray} :
      Canon d l → HelloValue d spec ν root key → Prints (ts ++ [(root, spec)]) lines log →
      Prints ts (l :: lines) (⟨.hello spec, ν, bitsOfBytes (root ++ key)⟩ :: log)
  | opening {ts : List (ByteArray × Flock.CoinTree.Spec)} {lines : List String} {log : List Release} {d : Json}
      {l : String} {i s n k : ℕ} {root b cb : ByteArray} {spec : Flock.CoinTree.Spec} {stream : String} :
      ts[i]? = some (root, spec) → spec.streams[s]? = some stream → Canon d l → OpenValue d root stream k n b cb →
      Prints ts lines log → Prints ts (l :: lines) (⟨.opening i s n, b, bitsOfBytes cb⟩ :: log)

/-- **The coins of family `label` after `after` in the log `log`**, in order: each fresh release's of that family whose
bytes answered are `after`. -/
def freshOf (label : String) (after : ByteArray) : List Release → List (List Bool)
  | [] => []
  | x :: xs => match x.frame with
    | .fresh l => if l = label ∧ x.answers = after then x.coins :: freshOf label after xs else freshOf label after xs
    | _ => freshOf label after xs

/-- **The bytes `ξ` hold the coins `css`** in order, each a coin's `COIN_BYTES` bytes, and whatever follows. -/
inductive Holds : ByteArray → List (List Bool) → Prop
  | nil (ξ : ByteArray) : Holds ξ []
  | cons {cb ξ : ByteArray} {css : List (List Bool)} : cb.size = COIN_BYTES → Holds ξ css →
      Holds (cb ++ ξ) (bitsOfBytes cb :: css)

/-- **The device printed its log `log`**, the properties the device's history at a play is to have: each run of the
device its ledger opens withheld on a window prints `log` after its release log's head (`Prints`), and, where its
release log reads as its coin commitment's (`releasedRootsOf`), released of each family after each root the coins
`log` shows released so (`Holds`). -/
def PrintsLog (dev : History) (log : List Release) : Prop :=
  ∀ pk w r, r ∈ dev.runs pk → OpensOn dev pk w r.id →
    Prints [] r.log.tail log ∧ ((∃ c roots, r.commitment = some c ∧ releasedRootsOf c r.log = .ok roots) →
      ∀ label after, Holds (r.released label after) (freshOf label after log))

/-! ## A release as v2 reads it -/

/-- The bytes of the bits `bs`, eight a byte, bit `t` bit `t % 8` of byte `t / 8` (`Verity.bitsOfBytes`'s inverse,
`coinBytes_bits`). -/
def coinBytes (bs : List Bool) : ByteArray := bytesOfBits (bs.length / 8) fun t => bs.getD t false

/-- Whether a frame is a round opened. -/
def opening? : Frame → Bool
  | .opening .. => true
  | _ => false

/-- **The read `y` is of the release `x`, but for a round opened's coins of one SHA-512**: the same frame and bytes
answered, and its coins, where it read any, `x`'s, or, at a round opened, other bytes than `x`'s with their SHA-512. -/
def ReadOf (y x : Release) : Prop :=
  y.frame = x.frame ∧ y.answers = x.answers ∧
    (y.coins ≠ [] → y.coins = x.coins ∨ opening? y.frame = true ∧ coinBytes y.coins ≠ coinBytes x.coins ∧
      Flock.Sha512.hash (coinBytes y.coins) = Flock.Sha512.hash (coinBytes x.coins))

theorem byte_lsb : ∀ n < 256, (List.range 8).foldr (fun b acc => 2 * acc + ((n >>> b) % 2 == 1).toNat) 0 = n := by
  decide +kernel

theorem foldr_congr {f g : ℕ → ℕ → ℕ} {x : ℕ} : ∀ {l : List ℕ}, (∀ b ∈ l, ∀ acc, f b acc = g b acc) →
    l.foldr f x = l.foldr g x
  | [], _ => rfl
  | b :: l, h => by
    rw [List.foldr_cons, List.foldr_cons, foldr_congr fun b' hb' => h b' (List.mem_cons_of_mem _ hb'),
      h b List.mem_cons_self]

theorem bitsOfBytes_length (z : ByteArray) : (bitsOfBytes z).length = 8 * z.size := by
  simp [bitsOfBytes]

/-- **The bytes of a byte string's bits are the string.** -/
theorem coinBytes_bits (z : ByteArray) : coinBytes (bitsOfBytes z) = z := by
  unfold coinBytes bytesOfBits
  rw [bitsOfBytes_length, Nat.mul_div_cancel_left _ (by decide : 0 < 8)]
  apply ByteArray.ext
  refine Array.ext (by exact Array.size_ofFn) fun i h1 h2 => ?_
  rw [Array.getElem_ofFn]
  rw [foldr_congr (g := fun b acc => 2 * acc + ((z.data[i].toNat >>> b) % 2 == 1).toNat) ?_,
    byte_lsb _ z.data[i].toNat_lt]
  · simp
  · intro b hb acc
    have hb8 : b < 8 := List.mem_range.1 hb
    have hi : i < z.size := h2
    show 2 * acc + ((bitsOfBytes z).getD (8 * i + b) false).toNat = _
    rw [List.getD_eq_getElem?_getD, bitsOfBytes, List.getElem?_map,
      List.getElem?_range (show 8 * i + b < 8 * z.size by omega)]
    simp only [Option.map_some, Option.getD_some, Flock.Hm96.bit]
    have e1 : (8 * i + b) / 8 = i := by omega
    have e2 : (8 * i + b) % 8 = b := by omega
    rw [e1, e2]
    simp only [ByteArray.get!, getElem!_pos z.data i h2]

theorem lsb_bits : ∀ b0 b1 b2 b3 b4 b5 b6 b7 : Bool, ∀ j < 8,
    (List.range 8).foldr (fun b acc => 2 * acc + ([b0, b1, b2, b3, b4, b5, b6, b7].getD b false).toNat) 0 < 256 ∧
      (((List.range 8).foldr (fun b acc => 2 * acc + ([b0, b1, b2, b3, b4, b5, b6, b7].getD b false).toNat) 0
        >>> j) % 2 == 1) = [b0, b1, b2, b3, b4, b5, b6, b7].getD j false := by
  decide +kernel

/-- A byte of eight bits, least significant first, is below 256 and reads them back. -/
theorem byte_bits (f : ℕ → Bool) (k : ℕ) {j : ℕ} (hj : j < 8) :
    (List.range 8).foldr (fun b acc => 2 * acc + (f (k + b)).toNat) 0 < 256 ∧
      (((List.range 8).foldr (fun b acc => 2 * acc + (f (k + b)).toNat) 0 >>> j) % 2 == 1) = f (k + j) := by
  have hl : ∀ b < 8, [f k, f (k + 1), f (k + 2), f (k + 3), f (k + 4), f (k + 5), f (k + 6), f (k + 7)].getD b false =
      f (k + b) := by
    intro b hb
    match b, hb with
    | 0, _ => rfl
    | 1, _ => rfl
    | 2, _ => rfl
    | 3, _ => rfl
    | 4, _ => rfl
    | 5, _ => rfl
    | 6, _ => rfl
    | 7, _ => rfl
    | _ + 8, h => exact absurd h (by omega)
  have h := lsb_bits (f k) (f (k + 1)) (f (k + 2)) (f (k + 3)) (f (k + 4)) (f (k + 5)) (f (k + 6)) (f (k + 7)) j hj
  rw [foldr_congr (g := fun b acc => 2 * acc + (f (k + b)).toNat) ?_, hl j hj] at h
  · exact h
  · intro b hb acc
    rw [hl b (List.mem_range.1 hb)]

theorem toNat_toUInt8 {n : ℕ} (h : n < 256) : n.toUInt8.toNat = n := by
  simp [Nat.toUInt8, Nat.mod_eq_of_lt h]

theorem coinBytes_size (bs : List Bool) : (coinBytes bs).size = bs.length / 8 := by
  unfold coinBytes bytesOfBits
  exact Array.size_ofFn

/-- **The bits of whole bytes' worth of bits' bytes are the bits** (`coinBytes_bits`'s converse). -/
theorem bits_coinBytes {bs : List Bool} (h : bs.length % 8 = 0) : bitsOfBytes (coinBytes bs) = bs := by
  have hn : 8 * (coinBytes bs).size = bs.length := by
    rw [coinBytes_size]
    omega
  apply List.ext_getElem?
  intro t
  rw [bitsOfBytes, List.getElem?_map]
  by_cases ht : t < bs.length
  · rw [List.getElem?_range (show t < 8 * (coinBytes bs).size by omega), Option.map_some,
      List.getElem?_eq_getElem ht, Flock.Hm96.bit]
    have hb : t / 8 < bs.length / 8 := by omega
    obtain ⟨hlt, hbit⟩ := byte_bits (fun t => bs.getD t false) (8 * (t / 8)) (Nat.mod_lt t (by decide : 0 < 8))
    rw [Nat.div_add_mod t 8] at hbit
    have hget : ((coinBytes bs).get! (t / 8)).toNat =
        (List.range 8).foldr (fun b acc => 2 * acc + (bs.getD (8 * (t / 8) + b) false).toNat) 0 := by
      simp only [ByteArray.get!]
      unfold coinBytes bytesOfBits
      rw [getElem!_pos _ (t / 8) ?_, Array.getElem_ofFn]
      · exact toNat_toUInt8 hlt
      · simpa only [Array.size_ofFn] using hb
    rw [hget, hbit, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem ht, Option.getD_some]
  · have e1 : (List.range (8 * (coinBytes bs).size))[t]? = none :=
      List.getElem?_eq_none (by rw [List.length_range]; omega)
    have e2 : bs[t]? = none := List.getElem?_eq_none (by omega)
    simp [e1, e2]

/-- **A fresh coin of the game is `COIN_BYTES` bytes' bits**: the 512 bits it flips, as the device releases them. -/
theorem fresh_bytes (x : Fin 512 → Bool) :
    (coinBytes (List.ofFn x)).size = COIN_BYTES ∧ bitsOfBytes (coinBytes (List.ofFn x)) = List.ofFn x :=
  ⟨by rw [coinBytes_size, List.length_ofFn]; rfl, bits_coinBytes (by rw [List.length_ofFn])⟩

theorem flatten_length : ∀ {L : List (List Bool)}, (∀ l ∈ L, l.length = 128) → L.flatten.length % 8 = 0
  | [], _ => rfl
  | l :: L, h => by
    have := flatten_length fun l' hl' => h l' (List.mem_cons_of_mem _ hl')
    rw [List.flatten_cons, List.length_append, h l List.mem_cons_self]
    omega

/-- **A round opened's coins are bytes' bits**: its first `n` coins, 128 bits each, as the device releases them. -/
theorem opened_bytes {block : ℕ} (d : TreeDraw block) (n : ℕ) :
    bitsOfBytes (coinBytes (TreeDraw.opened d n)) = TreeDraw.opened d n := by
  refine bits_coinBytes (flatten_length fun l hl => ?_)
  have hl' := List.mem_of_mem_take hl
  simp only [List.mem_ofFn] at hl'
  obtain ⟨i, rfl⟩ := hl'
  simp only [List.length_ofFn]

theorem hash_of_hex {a b : ByteArray} (h : sha512Hex a = sha512Hex b) :
    Flock.Sha512.hash a = Flock.Sha512.hash b := by
  have e := FlockComplete.JsonRT.unhex_hexOf (Flock.Sha512.hash b)
  rw [← show Flock.hexOf (Flock.Sha512.hash a) = Flock.hexOf (Flock.Sha512.hash b) from h,
    FlockComplete.JsonRT.unhex_hexOf] at e
  exact Except.ok.inj e

/-! ## A printed line reads back -/

theorem str_of {l : String} {j d : Json} (hev : event l = .ok j) (hd : Canon d l) {k s : String}
    (h : getD d k = .str s) : getD j k = .str s := by
  have h0 := event_getD hev hd.1 hd.2 k
  rw [h] at h0
  exact equiv_str h0

theorem nat_of {l : String} {j d : Json} (hev : event l = .ok j) (hd : Canon d l) {k : String} {n : ℕ}
    (h : nat? (getD d k) = some n) : nat? (getD j k) = some n :=
  (nat?_equiv (event_getD hev hd.1 hd.2 k)).trans h

theorem equiv_strs {x : Json} {ss : Array String} (h : JsonEquiv x (.arr (ss.map .str))) :
    x = .arr (ss.map .str) := by
  cases x with
  | arr a =>
    obtain ⟨hsz, hel⟩ := (FlockDeploy.jsonEquiv_arr_iff a _).1 h
    congr 1
    refine Array.ext hsz fun i h1 h2 => ?_
    have := hel i h1 h2
    rw [Array.getElem_map] at this ⊢
    exact equiv_str this
  | obj m => rw [JsonEquiv, FlockDeploy.jsonForm_obj, Properties.Challenger.CanonRead.jsonForm_arr] at h; cases h
  | _ => rw [JsonEquiv, Properties.Challenger.CanonRead.jsonForm_arr] at h; cases h

theorem spec_of {l : String} {j d : Json} (hev : event l = .ok j) (hd : Canon d l) {spec : Flock.CoinTree.Spec}
    (h : SpecValue (getD d "spec") spec) : SpecValue (getD j "spec") spec := by
  have hs := event_getD hev hd.1 hd.2 "spec"
  have ha := sorted_getD (event_ok hev).2 "spec"
  have hb := sorted_getD hd.1 "spec"
  have h1 := Properties.Challenger.CanonRead.getD_equiv ha hb hs "streams"
  rw [h.streams] at h1
  exact ⟨equiv_strs h1, (nat?_equiv (Properties.Challenger.CanonRead.getD_equiv ha hb hs "rounds_log")).trans
    h.roundsLog, (nat?_equiv (Properties.Challenger.CanonRead.getD_equiv ha hb hs "block")).trans h.block⟩

theorem mapM_strs {f : Json → Option String} (hf : ∀ x, f (.str x) = some x) :
    ∀ l : List String, (l.map Json.str).mapM f = some l
  | [] => rfl
  | x :: l => by rw [List.map_cons, List.mapM_cons, hf, mapM_strs hf l]; rfl

theorem specOf_value {x : Json} {spec : Flock.CoinTree.Spec} (h : SpecValue x spec) :
    Verity.Sampled.specOf x = some spec := by
  unfold Verity.Sampled.specOf
  rw [h.streams, h.roundsLog, h.block]
  dsimp only
  rw [Array.mapM_eq_mapM_toList, Array.toList_map, mapM_strs]
  · rfl
  · intro x; rfl

theorem distinct_nodup : ∀ {l : List String}, distinct l = true → l.Nodup
  | [], _ => List.nodup_nil
  | a :: l, h => by
    simp only [distinct, Bool.and_eq_true] at h
    exact List.nodup_cons.2 ⟨by simpa using h.1, distinct_nodup h.2⟩

/-- **A spec `isTreeSpec` accepts names distinct streams.** -/
theorem streams_nodup {x : Json} {spec : Flock.CoinTree.Spec}
    (hs : getD x "streams" = .arr (spec.streams.map .str)) (ht : isTreeSpec x = true) : spec.streams.toList.Nodup := by
  unfold isTreeSpec at ht
  rw [hs] at ht
  simp only [Bool.and_eq_true] at ht
  have hd := distinct_nodup ht.2.2
  simp only [Array.toList_map] at hd
  exact (hd.of_map _).of_map _

theorem treeLine_error {l e : String} (h : event l = .error e) : treeLine l = false := by
  unfold treeLine
  rw [h]

theorem treeLine_ok {l : String} {j : Json} (h : event l = .ok j) :
    treeLine l = (jsonStrIs (getD j "event") TREE_EVENT || jsonStrIs (getD j "event") OPEN_EVENT) := by
  unfold treeLine
  rw [h]

theorem rootOf_error {l e : String} (h : event l = .error e) : Verity.Sampled.rootOf l = none := by
  unfold Verity.Sampled.rootOf
  rw [h]
  rfl

theorem open_tree : (OPEN_EVENT == TREE_EVENT) = false := by decide

theorem treeLine_plain {d : Json} {l : String} (hd : Canon d l) (hp : PlainValue d) : treeLine l = false := by
  obtain ⟨e, he, h1, h2⟩ := hp
  cases hev : event l with
  | error _ => exact treeLine_error hev
  | ok j =>
    rw [treeLine_ok hev, str_of hev hd he]
    simp [jsonStrIs, h1, h2]

theorem rootOf_value {l : String} {j d : Json} {label : String} {b : ByteArray} (hev : event l = .ok j)
    (hd : Canon d l) (hr : RootValue d label b) : Verity.Sampled.rootOf l = some (label, b) := by
  unfold Verity.Sampled.rootOf
  rw [hev]
  simp only [Except.toOption, Option.bind_eq_bind, Option.bind_some, str_of hev hd hr.label, str_of hev hd hr.root,
    FlockComplete.JsonRT.unhex_hexOf, Option.pure_def]

/-! ## The release log's head -/

theorem esc_hd : ∀ x < 16, Properties.Composition.Esc.esc (FlockComplete.JsonRT.hd x) = [FlockComplete.JsonRT.hd x] := by
  decide +kernel

theorem esc_hexs : ∀ l : List UInt8,
    (l.flatMap fun x => [FlockComplete.JsonRT.hd (x.toNat / 16), FlockComplete.JsonRT.hd (x.toNat % 16)]).flatMap
        Properties.Composition.Esc.esc =
      l.flatMap fun x => [FlockComplete.JsonRT.hd (x.toNat / 16), FlockComplete.JsonRT.hd (x.toNat % 16)]
  | [] => rfl
  | x :: l => by
    have hx : x.toNat < 256 := x.toNat_lt
    simp only [List.flatMap_cons, esc_hexs l, esc_hd (x.toNat / 16) (by omega),
      esc_hd (x.toNat % 16) (Nat.mod_lt _ (by decide)), List.cons_append, List.nil_append]

/-- **A hex string is its own `canonStr` text, quoted**: each digit prints as itself. -/
theorem canonStr_hex (b : ByteArray) : Flock.canonStr (Flock.hexOf b) = "\"" ++ Flock.hexOf b ++ "\"" := by
  apply String.toList_injective
  rw [Properties.Composition.Esc.canonStr_toList, String.toList_append, String.toList_append,
    FlockComplete.JsonRT.hexOf_toList, esc_hexs]
  rfl

theorem head_sort {β : Type} (a b : β) :
    #[("event", a), ("root", b)].qsort (fun x y => decide (x.1 < y.1)) = #[("event", a), ("root", b)] := by
  refine Verity.QSort.qsort_sorted _ _ fun i hi j hj => ?_
  simp only [List.size_toArray, List.length_cons, List.length_nil] at hi hj
  obtain _ | _ | i := i <;> obtain _ | _ | j := j <;> first | omega |
    (simp only [List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ]
     exact decide_eq_decide.2 (by decide))

theorem canon_str (s : String) : Flock.canon (.str s) = pure (Flock.canonStr s) := by
  simp only [Flock.canon]

/-- **The release log's head is the coin commitment's value's text**: `{"event":"commitment","root":…}` with its
root's hex, as the challenger writes it and `releasedRootsOf` reads it. -/
theorem head_canon (b : ByteArray) :
    Canon (Json.mkObj [("event", .str "commitment"), ("root", .str (Flock.hexOf b))])
      ("{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf b ++ "\"}") := by
  have ht : (Std.TreeMap.Raw.ofList [("event", Json.str "commitment"), ("root", Json.str (Flock.hexOf b))] :
      Std.TreeMap.Raw String Json).toArray = #[("event", .str "commitment"), ("root", .str (Flock.hexOf b))] := rfl
  refine ⟨Properties.Challenger.CanonRead.sortedKeys_obj.2 ⟨?_, fun p hp => ?_⟩, ?_⟩
  · rw [ht]
    exact (by decide : increasing ["event", "root"] = true)
  · rw [ht] at hp
    simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> (unfold sortedKeys; rfl)
  · rw [Json.mkObj, Properties.Challenger.CanonLine.canon_obj, ht]
    simp only [List.mapM_toArray, List.mapM_cons, List.mapM_nil, canon_str, pure_bind, map_pure]
    rw [head_sort, canonStr_hex]
    show Except.ok _ = Except.ok _
    congr 1
    apply String.toList_injective
    simp only [List.map_cons, List.map_nil, String.intercalate_cons_cons, String.intercalate_singleton,
      String.toList_append, List.append_assoc]
    rfl

theorem head_plain (b : ByteArray) :
    PlainValue (Json.mkObj [("event", .str "commitment"), ("root", .str (Flock.hexOf b))]) := by
  refine ⟨"commitment", ?_, by decide, by decide⟩
  rw [Json.mkObj, Properties.Challenger.CanonRead.getD_obj,
    Properties.Challenger.CanonRead.get?_ofList (l := [("event", Json.str "commitment"), ("root", Json.str (Flock.hexOf b))])
      (by simp only [List.map_cons, List.map_nil]; decide)]
  rfl

/-- **The release log's head is no coin tree's line.** -/
theorem treeLine_head (b : ByteArray) :
    treeLine ("{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf b ++ "\"}") = false :=
  treeLine_plain (head_canon b) (head_plain b)

/-- **A log `releasedRootsOf` accepts at the coin commitment `c` opens with `c`'s line.** -/
theorem head_of_roots {c : LiveCommitment} {log : List String} {roots : Roots} (h : releasedRootsOf c log = .ok roots) :
    log.head? = some ("{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}") := by
  by_cases hc : (log.head? == some ("{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}")) = true
  · exact beq_iff_eq.1 hc
  · rw [Bool.not_eq_true] at hc
    unfold releasedRootsOf at h
    simp only [hc, Bool.false_eq_true, ↓reduceIte] at h
    cases h

/-- The root a coin tree's hello line names, as `coinTreesOf` reads it; `none` on every other line. -/
def helloRoot (l : String) : Option String :=
  match event l with
  | .ok j => if jsonStrIs (getD j "event") TREE_EVENT then
      (match getD j "root" with | .str s => some s | _ => none) else none
  | .error _ => none

theorem helloRoot_plain {d : Json} {l : String} (hd : Canon d l) (hp : PlainValue d) : helloRoot l = none := by
  obtain ⟨e, he, h1, -⟩ := hp
  unfold helloRoot
  cases hev : event l with
  | error _ => rfl
  | ok j =>
    dsimp only
    rw [str_of hev hd he]
    simp [jsonStrIs, h1]

theorem helloRoot_tree {l s : String} (h : helloRoot l = some s) : treeLine l = true := by
  unfold helloRoot at h
  cases hev : event l with
  | error _ => rw [hev] at h; cases h
  | ok j =>
    rw [hev] at h
    dsimp only at h
    rw [treeLine_ok hev]
    split at h
    · rename_i ht
      rw [ht]
      rfl
    · cases h

/-! ## `readTail` one line at a time -/

variable {label : String} {after : ByteArray} {recs : List Json}

theorem readTail_none {l : String} {ls : List String} {ts : List (String × Json)} {ξ : ByteArray}
    (ht : treeLine l = false) :
    readTail label after recs (l :: ls) ts none ξ = readTail label after recs ls ts (some l) ξ := by
  rw [Verity.Sampled.readTail, ite_eq_right (by simp [ht])]

/-- **A pending root's line with no root survives no read.** -/
theorem pending_none {a : String} (ha : Verity.Sampled.rootOf a = none) {ls : List String} :
    ∀ {ts : List (String × Json)} {ξ : ByteArray}, readTail label after recs ls ts (some a) ξ = none := by
  induction ls with
  | nil => intro ts ξ; rfl
  | cons l ls ih =>
    intro ts ξ
    by_cases ht : treeLine l = true
    · rw [Verity.Sampled.readTail, ite_eq_left ht]
      cases (event l).toOption.bind (treeStep recs ts) with
      | none => rfl
      | some p =>
        obtain ⟨r, ts'⟩ := p
        simp only [Option.bind_eq_bind, Option.bind_some, ih, Option.bind_none]
    · rw [Verity.Sampled.readTail, ite_eq_right ht]
      simp only [ha, Option.bind_eq_bind, Option.bind_none]

/-- **v2's read of a log `releasedRootsOf` accepts is `readTail`'s of the lines after its head**: the head is the coin
commitment's line (`head_of_roots`), no coin tree's (`treeLine_head`). -/
theorem readLog_head {c : LiveCommitment} {log : List String} {roots : Roots} (h : releasedRootsOf c log = .ok roots)
    {ξ : ByteArray} : readLog label after recs ξ log = readTail label after recs log.tail [] none ξ := by
  have hh := head_of_roots h
  cases log with
  | nil => cases hh
  | cons l ls =>
    obtain rfl : l = _ := Option.some.inj hh
    simp only [Verity.Sampled.readLog, treeLine_head, Bool.false_eq_true, ↓reduceIte, List.tail_cons]

theorem treeStep_hello {ts : List (String × Json)} {j : Json} {spec : Flock.CoinTree.Spec}
    {ν root key : ByteArray} (he : getD j "event" = .str TREE_EVENT) (hk : getD j "key" = .str (Flock.hexOf key))
    (hn : getD j "nonce" = .str (Flock.hexOf ν)) (hr : getD j "root" = .str (Flock.hexOf root))
    (hs : Verity.Sampled.specOf (getD j "spec") = some spec) :
    treeStep recs ts j =
      some (⟨.hello spec, ν, bitsOfBytes (root ++ key)⟩, ts ++ [(Flock.hexOf root, getD j "spec")]) := by
  unfold Verity.Sampled.treeStep
  simp only [he, jsonStrIs, beq_self_eq_true, ↓reduceIte, hk, hn, hr, hs, FlockComplete.JsonRT.unhex_hexOf,
    Except.toOption, Option.bind_eq_bind, Option.bind_some, Option.pure_def]

/-- The coins v2 reads of round `k` of stream `stream` of the tree of root `root`: a record's of that tree, else none
(`Verity.Sampled.treeStep`'s). -/
def openCoins (recs : List Json) (root stream : String) (k : ℕ) : List Bool :=
  match recs.find? (Verity.Sampled.recordTree · == root) with
  | some rec => ((Verity.Sampled.recordRound rec stream k).map bitsOfBytes).getD []
  | none => []

theorem treeStep_open {ts : List (String × Json)} {j x : Json} {R S R' : String} {k n i s : ℕ}
    {b : ByteArray} (he : jsonStrIs (getD j "event") TREE_EVENT = false)
    (ht : getD j "tree" = .str R) (hs : getD j "stream" = .str S) (ha : getD j "after" = .str (Flock.hexOf b))
    (hk : nat? (getD j "round") = some k) (hn : nat? (getD j "n") = some n)
    (hi : ts.findIdx? (·.1 == R) = some i) (hti : ts[i]? = some (R', x))
    (hsi : (Verity.Sampled.specStreams x).findIdx? (jsonStrIs · S) = some s) :
    treeStep recs ts j = some (⟨.opening i s n, b, openCoins recs R S k⟩, ts) := by
  unfold Verity.Sampled.treeStep
  simp only [he, Bool.false_eq_true, ↓reduceIte, ht, hs, ha, hk, hn, hi, hti, hsi,
    FlockComplete.JsonRT.unhex_hexOf, Except.toOption, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  rfl

theorem specStreams_value {x : Json} {spec : Flock.CoinTree.Spec} (h : SpecValue x spec) :
    Verity.Sampled.specStreams x = spec.streams.map .str := by
  unfold Verity.Sampled.specStreams
  rw [h.streams]

theorem findIdx_nodup {β : Type} [BEq β] [LawfulBEq β] {l : List β} {i : ℕ} {x : β} (hn : l.Nodup)
    (hi : l[i]? = some x) : l.findIdx? (· == x) = some i := by
  obtain ⟨h, rfl⟩ := List.getElem?_eq_some_iff.1 hi
  refine List.findIdx?_eq_some_iff_getElem.2 ⟨h, by simp, fun j hj e => ?_⟩
  have := (hn.getElem_inj (hi := Nat.lt_trans hj h) (hj := h)).1 (beq_iff_eq.1 e)
  omega

/-- **The trees v2's read holds are the ones the device sampled**: each the hex of its root and a value of its spec
(`SpecValue`), in order. -/
abbrev TreesRead (tsj : List (String × Json)) (ts : List (ByteArray × Flock.CoinTree.Spec)) : Prop :=
  List.Forall₂ (fun q p => q.1 = Flock.hexOf p.1 ∧ SpecValue q.2 p.2) tsj ts

theorem TreesRead.push {tsj : List (String × Json)} {ts : List (ByteArray × Flock.CoinTree.Spec)}
    {q : String × Json} {p : ByteArray × Flock.CoinTree.Spec} (h : TreesRead tsj ts)
    (hq : q.1 = Flock.hexOf p.1 ∧ SpecValue q.2 p.2) : TreesRead (tsj ++ [q]) (ts ++ [p]) := by
  induction h with
  | nil => exact .cons hq .nil
  | cons h _ ih => exact .cons h ih

theorem TreesRead.roots {tsj : List (String × Json)} {ts : List (ByteArray × Flock.CoinTree.Spec)}
    (h : TreesRead tsj ts) : tsj.map Prod.fst = ts.map fun p => Flock.hexOf p.1 := by
  induction h with
  | nil => rfl
  | cons h _ ih => rw [List.map_cons, List.map_cons, ih, h.1]

theorem TreesRead.get {tsj : List (String × Json)} {ts : List (ByteArray × Flock.CoinTree.Spec)}
    (h : TreesRead tsj ts) : ∀ {i : ℕ} {p : ByteArray × Flock.CoinTree.Spec}, ts[i]? = some p →
      ∃ x, tsj[i]? = some (Flock.hexOf p.1, x) ∧ SpecValue x p.2 := by
  induction h with
  | nil => intro i p hp; simp at hp
  | @cons q p' tsj' ts' hqp _ ih =>
    intro i p hp
    cases i with
    | zero =>
      rw [List.getElem?_cons_zero, Option.some.injEq] at hp
      subst hp
      exact ⟨q.2, by rw [List.getElem?_cons_zero, ← hqp.1], hqp.2⟩
    | succ i =>
      rw [List.getElem?_cons_succ] at hp
      obtain ⟨x, hx, hs⟩ := ih hp
      exact ⟨x, by rw [List.getElem?_cons_succ]; exact hx, hs⟩

theorem findIdx_root {tsj : List (String × Json)} {ts : List (ByteArray × Flock.CoinTree.Spec)} {i : ℕ}
    {root : ByteArray} {spec : Flock.CoinTree.Spec} (hR : TreesRead tsj ts)
    (hn : (ts.map fun p => Flock.hexOf p.1).Nodup) (hi : ts[i]? = some (root, spec)) :
    tsj.findIdx? (·.1 == Flock.hexOf root) = some i := by
  have e : tsj.findIdx? (·.1 == Flock.hexOf root) = (tsj.map Prod.fst).findIdx? (· == Flock.hexOf root) := by
    simp only [List.findIdx?_map]
    rfl
  rw [e, hR.roots]
  exact findIdx_nodup hn (by simp [hi])

theorem findIdx_stream {xs : Array String} {i : ℕ} {x : String} (hn : xs.toList.Nodup) (hi : xs[i]? = some x) :
    xs.findIdx? (· == x) = some i := by
  obtain ⟨h, rfl⟩ := Array.getElem?_eq_some_iff.1 hi
  refine Array.findIdx?_eq_some_iff_getElem.2 ⟨h, by simp, fun j hj e => ?_⟩
  have e' : xs.toList[j]'(by simpa using Nat.lt_trans hj h) = xs.toList[i]'(by simpa using h) := by
    simpa using beq_iff_eq.1 e
  have := (hn.getElem_inj).1 e'
  omega

/-- **A spec's stream is the one v2 finds in the spec's value**, where its streams are distinct. -/
theorem findIdx_spec {x : Json} {spec : Flock.CoinTree.Spec} {i : ℕ} {S : String} (hx : SpecValue x spec)
    (hn : spec.streams.toList.Nodup) (hi : spec.streams[i]? = some S) :
    (Verity.Sampled.specStreams x).findIdx? (jsonStrIs · S) = some i := by
  rw [specStreams_value hx, Array.findIdx?_map]
  exact findIdx_stream hn hi

theorem holds_cons {ξ : ByteArray} {cs : List Bool} {css : List (List Bool)} (h : Holds ξ (cs :: css)) :
    ∃ cb ξ', ξ = cb ++ ξ' ∧ cb.size = COIN_BYTES ∧ cs = bitsOfBytes cb ∧ Holds ξ' css := by
  cases h with
  | cons hsz h => exact ⟨_, _, rfl, hsz, rfl, h⟩

/-! ## The trees v2 accepts -/

/-- **What `coinTreesOf` holds of the trees `trees` it read from the lines `pre`**: their roots are the hello lines'
(`helloRoot`), each once; each hello line's spec is one `isTreeSpec` accepts; each round opened's line names a tree
whose round it opens, with that line's coins' SHA-512; and each round a tree opened is below the count of its stream's
rounds opened. -/
structure TreesOf (pre : List String) (trees : Array LoggedTree) : Prop where
  roots : trees.toList.map (·.root) = pre.filterMap helloRoot
  nodup : (trees.toList.map (·.root)).Nodup
  spec : ∀ l ∈ pre, ∀ j, event l = .ok j → jsonStrIs (getD j "event") TREE_EVENT = true →
    isTreeSpec (getD j "spec") = true
  opened : ∀ l ∈ pre, ∀ j, event l = .ok j → jsonStrIs (getD j "event") TREE_EVENT = false →
    jsonStrIs (getD j "event") OPEN_EVENT = true → ∀ R S k C, getD j "tree" = .str R → getD j "stream" = .str S →
    nat? (getD j "round") = some k → getD j "coins" = .str C →
    ∃ t ∈ trees, t.root = R ∧ ∃ o, t.opening S k = some o ∧ o.coins = C
  rounds : ∀ t ∈ trees, ∀ o ∈ t.opened, o.round < (t.opened.filter (·.stream == o.stream)).size

theorem map_modify {β γ : Type} {xs : Array β} {i : ℕ} {g : β → β} {f : β → γ} (hf : ∀ a, f (g a) = f a) :
    (xs.modify i g).toList.map f = xs.toList.map f := by
  apply List.ext_getElem (by simp)
  intro j h1 h2
  simp only [List.getElem_map, Array.getElem_toList]
  rw [Array.getElem_modify]
  split
  · exact hf _
  · rfl

theorem roots_modify {xs : Array LoggedTree} {i : ℕ} {x : TreeOpening} :
    (xs.modify i fun t => { t with opened := t.opened.push x }).toList.map (·.root) = xs.toList.map (·.root) :=
  map_modify (xs := xs) (i := i) (g := fun (t : LoggedTree) => { t with opened := t.opened.push x })
    (f := fun (t : LoggedTree) => t.root) fun _ => rfl

theorem mem_modify_of_mem {β : Type} {xs : Array β} {i : ℕ} {g : β → β} {b : β} (h : b ∈ xs) :
    b ∈ xs.modify i g ∨ g b ∈ xs.modify i g := by
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 h
  have hj' : j < (xs.modify i g).size := by simpa using hj
  by_cases hij : i = j
  · subst hij
    exact .inr (Array.mem_iff_getElem.2 ⟨i, hj', Array.getElem_modify_self g hj'⟩)
  · exact .inl (Array.mem_iff_getElem.2 ⟨j, hj', Array.getElem_modify_of_ne hij g hj'⟩)

theorem mem_modify_self {β : Type} {xs : Array β} {i : ℕ} {g : β → β} {a : β} (h : xs[i]? = some a) :
    g a ∈ xs.modify i g := by
  obtain ⟨hi, rfl⟩ := Array.getElem?_eq_some_iff.1 h
  have hi' : i < (xs.modify i g).size := by simpa using hi
  exact Array.mem_iff_getElem.2 ⟨i, hi', Array.getElem_modify_self g hi'⟩

theorem opening_push {t : LoggedTree} {o : TreeOpening} {S : String} {k : ℕ} (h : t.opening S k = some o)
    (x : TreeOpening) : ({ t with opened := t.opened.push x } : LoggedTree).opening S k = some o := by
  unfold LoggedTree.opening at h ⊢
  show (t.opened.push x).find? _ = some o
  rw [Array.find?_push, h]
  rfl

/-- A tree's round opened stays opened when another round of the tree is. -/
theorem opened_modify {ts : Array LoggedTree} {i : ℕ} {x : TreeOpening} {t : LoggedTree} {R S C : String} {k : ℕ}
    (ht : t ∈ ts) (hr : t.root = R) {o : TreeOpening} (ho : t.opening S k = some o) (hc : o.coins = C) :
    ∃ t' ∈ ts.modify i (fun t => { t with opened := t.opened.push x }), t'.root = R ∧
      ∃ o, t'.opening S k = some o ∧ o.coins = C := by
  rcases mem_modify_of_mem (i := i) (g := fun (t : LoggedTree) => { t with opened := t.opened.push x }) ht with
    hm | hm
  · exact ⟨t, hm, hr, o, ho, hc⟩
  · exact ⟨_, hm, hr, o, opening_push ho x, hc⟩

/-- **A round opened next of its stream is the one `opening` finds**, where each round the tree opened is below the
count of its stream's. -/
theorem opened_new {ts : Array LoggedTree} {i : ℕ} {t : LoggedTree} {x : TreeOpening} {R S C : String} {k : ℕ}
    (hget : ts[i]? = some t) (hr : ∀ o ∈ t.opened, o.round < (t.opened.filter (·.stream == o.stream)).size)
    (hk : x.round = (t.opened.filter (·.stream == x.stream)).size) (hR : t.root = R) (hS : x.stream = S)
    (hK : x.round = k) (hC : x.coins = C) :
    ∃ t' ∈ ts.modify i (fun t => { t with opened := t.opened.push x }), t'.root = R ∧
      ∃ o, t'.opening S k = some o ∧ o.coins = C := by
  subst hR hS hK hC
  refine ⟨_, mem_modify_self hget, rfl, x, ?_, rfl⟩
  show (t.opened.push x).find? (fun o => o.stream == x.stream && o.round == x.round) = some x
  have hfind : t.opened.find? (fun o => o.stream == x.stream && o.round == x.round) = none := by
    rw [Array.find?_eq_none]
    intro o ho e
    simp only [Bool.and_eq_true, beq_iff_eq] at e
    have hlt := hr o ho
    rw [e.1, ← hk, e.2] at hlt
    exact Nat.lt_irrefl _ hlt
  rw [Array.find?_push, hfind]
  simp

theorem rounds_push {t : LoggedTree} {x : TreeOpening}
    (hr : ∀ o ∈ t.opened, o.round < (t.opened.filter (·.stream == o.stream)).size)
    (hk : x.round = (t.opened.filter (·.stream == x.stream)).size) :
    ∀ o ∈ t.opened.push x, o.round < ((t.opened.push x).filter (·.stream == o.stream)).size := by
  intro o ho
  rw [Array.filter_push]
  rcases Array.mem_push.1 ho with ho | rfl
  · have hlt := hr o ho
    split
    · rw [Array.size_push]
      omega
    · exact hlt
  · simp only [beq_self_eq_true, ↓reduceIte, Array.size_push]
    omega

/-- **Each line `coinTreesOf` reads keeps `TreesOf`.** -/
theorem trees_log {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    TreesOf log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Trees.forIn_inv_run (I := TreesOf) ?_
    (by exact ⟨rfl, List.nodup_nil, fun _ h => absurd h (by simp), fun _ h => absurd h (by simp),
      fun _ h => absurd h (by simp)⟩) h
  intro pre line ts r hI hr
  have mem : ∀ {l : String}, l ∈ pre ++ [line] → l ∈ pre ∨ l = line := fun h => by simpa using h
  split at hr
  case isFalse htl =>
    cases hr
    have hroot : helloRoot line = none := by
      cases hh : helloRoot line with
      | none => rfl
      | some _ => exact absurd (helloRoot_tree hh) htl
    refine ⟨ts, rfl, ⟨?_, hI.nodup, ?_, ?_, hI.rounds⟩⟩
    · rw [List.filterMap_append, ← hI.roots]
      simp [hroot]
    · intro l hl j hev hT
      rcases mem hl with hl | rfl
      · exact hI.spec l hl j hev hT
      · exact absurd (by simp [treeLine_ok hev, hT]) htl
    · intro l hl j hev hT hO R S k C hR hS hk hC
      rcases mem hl with hl | rfl
      · exact hI.opened l hl j hev hT hO R S k C hR hS hk hC
      · exact absurd (by simp [treeLine_ok hev, hO]) htl
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, Trees.ok_bind] at hr
    split at hr
    · rename_i htree
      split at hr
      case isFalse => cases hr
      rename_i hcond
      split at hr
      case isTrue => cases hr
      rename_i hany
      cases hr
      simp only [Bool.and_eq_true] at hcond
      have hhex := hcond.1.2
      obtain ⟨R, hR⟩ : ∃ R, getD d "root" = .str R := by
        cases hc : getD d "root" <;> rw [hc] at hhex <;> first | exact ⟨_, rfl⟩ | cases hhex
      have hroot : helloRoot line = some R := by
        unfold helloRoot
        rw [hev]
        dsimp only
        rw [ite_eq_left htree, hR]
      have hnew : ∀ t ∈ ts, t.root ≠ R := by
        intro t ht e
        apply hany
        rw [Array.any_eq_true']
        refine ⟨t, ht, ?_⟩
        rw [hR]
        show (t.root == R || _) = true
        rw [e]
        simp
      refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_, ?_⟩⟩
      · rw [Array.toList_push, List.map_append, List.filterMap_append, hI.roots, List.filterMap_cons_some hroot,
          List.filterMap_nil, List.map_cons, List.map_nil, hR]
        rfl
      · rw [Array.toList_push, List.map_append, List.map_cons, List.map_nil]
        refine List.nodup_append.2 ⟨hI.nodup, by simp, fun a ha b hb e => ?_⟩
        obtain ⟨t, ht, rfl⟩ := List.mem_map.1 ha
        rw [List.mem_singleton.1 hb, hR] at e
        exact hnew t (Array.mem_toList_iff.1 ht) e
      · intro l hl j hev' hT
        rcases mem hl with hl | rfl
        · exact hI.spec l hl j hev' hT
        · obtain rfl : j = d := Except.ok.inj (hev'.symm.trans hev)
          exact hcond.2
      · intro l hl j hev' hT hO R' S k C hR' hS hk hC
        rcases mem hl with hl | rfl
        · obtain ⟨t, ht, h⟩ := hI.opened l hl j hev' hT hO R' S k C hR' hS hk hC
          exact ⟨t, Array.mem_push_of_mem _ ht, h⟩
        · obtain rfl : j = d := Except.ok.inj (hev'.symm.trans hev)
          rw [htree] at hT
          cases hT
      · intro t ht o ho
        rcases Array.mem_push.1 ht with ht | rfl
        · exact hI.rounds t ht o ho
        · simp at ho
    · rename_i hopen
      split at hr
      case isFalse => cases hr
      split at hr
      case h_2 => cases hr
      rename_i root htree
      split at hr
      case h_2 => cases hr
      rename_i i hidx
      split at hr
      case h_2 => cases hr
      rename_i t hget
      split at hr
      case h_2 => cases hr
      rename_i s hstream
      split at hr
      case h_2 => cases hr
      rename_i k n hkn
      obtain ⟨hi, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
      have hti : ts[i] = t := by
        rw [Array.getElem?_eq_getElem hi] at hget
        exact Option.some.inj hget
      rw [hti] at hp
      have hroot : t.root = root := beq_iff_eq.1 hp
      have htmem : t ∈ ts := hti ▸ Array.getElem_mem hi
      simp only [Prod.mk.injEq] at hkn
      have hnone : helloRoot line = none := by
        unfold helloRoot
        rw [hev]
        dsimp only
        rw [ite_eq_right hopen]
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          have hcount : k = (t.opened.filter (·.stream == s)).size := beq_iff_eq.1 hcond.1.1.1.1.2
          cases hr
          refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_, ?_⟩⟩
          · rw [roots_modify, hI.roots, List.filterMap_append, List.filterMap_cons_none hnone, List.filterMap_nil,
              List.append_nil]
          · rw [roots_modify]
            exact hI.nodup
          · intro l hl j hev' hT
            rcases mem hl with hl | rfl
            · exact hI.spec l hl j hev' hT
            · obtain rfl : j = d := Except.ok.inj (hev'.symm.trans hev)
              exact absurd hT hopen
          · intro l hl j hev' hT hO R' S' k' C' hR' hS' hk' hC'
            rcases mem hl with hl | rfl
            · obtain ⟨t', ht', hr', o, ho, hoc⟩ := hI.opened l hl j hev' hT hO R' S' k' C' hR' hS' hk' hC'
              exact opened_modify ht' hr' ho hoc
            · obtain rfl : j = d := Except.ok.inj (hev'.symm.trans hev)
              exact opened_new hget (hI.rounds t htmem) (by exact hcount)
                (by rw [htree] at hR'; cases hR'; exact hroot) (by rw [hstream] at hS'; cases hS'; rfl)
                (by exact Option.some.inj (hkn.1.symm.trans hk')) (by rw [hC']; rfl)
          · intro t' ht' o ho
            rcases Trees.mem_modify ht' with ht' | ⟨a, ha, rfl⟩
            · exact hI.rounds t' ht' o ho
            · have hat : a = t := Option.some.inj (ha.symm.trans hget)
              rw [hat] at ho ⊢
              exact rounds_push (hI.rounds t htmem) (by exact hcount) o ho
        · cases hr

/-- **The lines a round's coins are read against**: each hello line's spec `isTreeSpec` accepts, and each round opened's
line names coins whose SHA-512 is that of every record's coins for that tree's round. -/
structure TreeLines (recs : List Json) (lines : List String) : Prop where
  spec : ∀ l ∈ lines, ∀ j, event l = .ok j → jsonStrIs (getD j "event") TREE_EVENT = true →
    isTreeSpec (getD j "spec") = true
  coins : ∀ l ∈ lines, ∀ j, event l = .ok j → jsonStrIs (getD j "event") TREE_EVENT = false →
    jsonStrIs (getD j "event") OPEN_EVENT = true → ∀ R S k C, getD j "tree" = .str R → getD j "stream" = .str S →
    nat? (getD j "round") = some k → getD j "coins" = .str C →
    ∀ rec ∈ recs, Verity.Sampled.recordTree rec = R → ∀ cs, Verity.Sampled.recordRound rec S k = some cs →
      sha512Hex cs = C

theorem TreeLines.sub {lines lines' : List String} (h : TreeLines recs lines) (hs : ∀ l ∈ lines', l ∈ lines) :
    TreeLines recs lines' :=
  ⟨fun l hl => h.spec l (hs l hl), fun l hl => h.coins l (hs l hl)⟩

/-- **The records v2 holds to the trees it read give each round opened the coins its line shows**, by their SHA-512:
a tree's root is no other tree's, and its round is opened once (`trees_log`), so a record's round is that line's
(`Trees.round_held`). -/
theorem tree_lines {log : List String} {ts : Array LoggedTree} (h : coinTreesOf log = .ok ts)
    (hrecs : ∀ rec ∈ recs, ∃ tabs, sessionTreeReason ts tabs rec = none) : TreeLines recs log := by
  have hT := trees_log h
  refine ⟨hT.spec, fun l hl j hev hnt hop R S k C hR hS hk hC rec hrec htree cs hcs => ?_⟩
  obtain ⟨t, ht, htr, o, ho, hoc⟩ := hT.opened l hl j hev hnt hop R S k C hR hS hk hC
  obtain ⟨tabs, htabs⟩ := hrecs rec hrec
  obtain ⟨t', ht', htr', o', ho', hoc'⟩ := Trees.round_held htabs hcs
  have e : t' = t := List.inj_on_of_nodup_map hT.nodup (Array.mem_toList_iff.2 ht') (Array.mem_toList_iff.2 ht)
    (htr'.symm.trans (htree.trans htr.symm))
  subst e
  rw [ho] at ho'
  cases ho'
  rw [← hoc, hoc']

/-! ## The induction -/

/-- **`readTail` reads what the device printed** (`Prints`) as its releases (`ReadOf`): after the trees `ts` the device
sampled, which the read holds (`TreesRead`), whose roots are distinct from each other and from the hello lines' to come
and whose streams are distinct, on lines a round's coins are read against (`TreeLines`), with the coins the device
released of the sample's family after its digest held in `ξ` (`Holds`). -/
theorem read_prints : ∀ {ts : List (ByteArray × Flock.CoinTree.Spec)} {lines : List String} {log : List Release},
    Prints ts lines log → ∀ {tsj : List (String × Json)} {ξ : ByteArray} {out : List Release}, TreesRead tsj ts →
    TreeLines recs lines → ((ts.map fun p => Flock.hexOf p.1) ++ lines.filterMap helloRoot).Nodup →
    (∀ p ∈ ts, p.2.streams.toList.Nodup) → Holds ξ (freshOf label after log) →
    readTail label after recs lines tsj none ξ = some out → List.Forall₂ ReadOf out log := by
  intro ts lines log hp
  induction hp with
  | nil ts =>
    intro tsj ξ out _ _ _ _ _ h
    rw [Verity.Sampled.readTail] at h
    cases h
    exact .nil
  | @fresh ts lines log d d' l l' lab b cs hd hr hd' hp' _ ih =>
    intro tsj ξ out hR htl hn hst hξ h
    have ht1 := treeLine_plain hd hr.plain
    have ht2 := treeLine_plain hd' hp'
    rw [readTail_none ht1, Verity.Sampled.readTail, ite_eq_right (by simp [ht2])] at h
    dsimp only at h
    have hn' : ((ts.map fun p => Flock.hexOf p.1) ++ lines.filterMap helloRoot).Nodup := by
      rwa [List.filterMap_cons_none (helloRoot_plain hd hr.plain),
        List.filterMap_cons_none (helloRoot_plain hd' hp')] at hn
    have htl' : TreeLines recs lines := htl.sub fun l hl => List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hl)
    cases hev : event l with
    | error e => rw [rootOf_error hev] at h; cases h
    | ok j =>
      rw [rootOf_value hev hd hr] at h
      simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def] at h
      split at h
      · rename_i hc
        obtain ⟨rfl, rfl⟩ := hc
        simp only [freshOf, and_self, ↓reduceIte] at hξ
        obtain ⟨cb, ξ', rfl, hsz, rfl, hξ'⟩ := holds_cons hξ
        obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
        cases h
        rw [ByteArray.extract_append_eq_right hsz.symm ByteArray.size_append] at hrest
        rw [ByteArray.extract_append_eq_left hsz.symm]
        exact .cons ⟨rfl, rfl, fun _ => .inl rfl⟩ (ih hR htl' hn' hst hξ' hrest)
      · rename_i hc
        obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
        cases h
        have hξ' : Holds ξ (freshOf label after log) := by simpa [freshOf, hc] using hξ
        exact .cons ⟨rfl, rfl, fun h => absurd rfl h⟩ (ih hR htl' hn' hst hξ' hrest)
  | @hello ts lines log d l spec ν root key hd hh _ ih =>
    intro tsj ξ out hR htl hn hst hξ h
    cases hev : event l with
    | error e =>
      rw [readTail_none (treeLine_error hev), pending_none (rootOf_error hev)] at h
      cases h
    | ok j =>
      have hE := str_of hev hd hh.event
      have hT : treeLine l = true := by rw [treeLine_ok hev, hE]; simp [jsonStrIs]
      have hS := spec_of hev hd hh.spec
      have hrel := treeStep_hello (recs := recs) (ts := tsj) hE (str_of hev hd hh.key) (str_of hev hd hh.nonce)
        (str_of hev hd hh.root) (specOf_value hS)
      rw [Verity.Sampled.readTail, ite_eq_left hT, hev] at h
      simp only [Except.toOption, Option.bind_eq_bind, Option.bind_some, hrel, Option.pure_def] at h
      obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      have hroot : helloRoot l = some (Flock.hexOf root) := by
        unfold helloRoot
        rw [hev]
        dsimp only
        rw [hE, str_of hev hd hh.root]
        simp [jsonStrIs]
      refine .cons ⟨rfl, rfl, fun _ => .inl rfl⟩
        (ih (tsj := tsj ++ [(Flock.hexOf root, getD j "spec")]) (ξ := ξ) (out := rest) (hR.push ⟨rfl, hS⟩)
          (htl.sub fun l hl => List.mem_cons_of_mem _ hl) ?_ ?_ ?_ hrest)
      · rw [List.filterMap_cons_some hroot] at hn
        simpa [List.map_append, List.append_assoc] using hn
      · intro p hp
        rcases List.mem_append.1 hp with hp | hp
        · exact hst p hp
        · rw [List.mem_singleton.1 hp]
          exact streams_nodup hS.streams (htl.spec l List.mem_cons_self j hev (by rw [hE]; simp [jsonStrIs]))
      · simpa [freshOf] using hξ
  | @opening ts lines log d l i s n k root b cb spec stream hi hs hd ho _ ih =>
    intro tsj ξ out hR htl hn hst hξ h
    cases hev : event l with
    | error e =>
      rw [readTail_none (treeLine_error hev), pending_none (rootOf_error hev)] at h
      cases h
    | ok j =>
      have hE := str_of hev hd ho.event
      have hnt : jsonStrIs (getD j "event") TREE_EVENT = false := by rw [hE]; exact open_tree
      have hop : jsonStrIs (getD j "event") OPEN_EVENT = true := by rw [hE]; simp [jsonStrIs]
      have hT : treeLine l = true := by rw [treeLine_ok hev, hop, Bool.or_true]
      have hpre : (ts.map fun p => Flock.hexOf p.1).Nodup := hn.sublist (List.sublist_append_left _ _)
      obtain ⟨x, hget, hx⟩ := hR.get hi
      have hsn : spec.streams.toList.Nodup := hst (root, spec) (List.mem_of_getElem? hi)
      have hrel := treeStep_open (recs := recs) hnt (str_of hev hd ho.tree) (str_of hev hd ho.stream)
        (str_of hev hd ho.after) (nat_of hev hd ho.round) (nat_of hev hd ho.n) (findIdx_root hR hpre hi) hget
        (findIdx_spec hx hsn hs)
      rw [Verity.Sampled.readTail, ite_eq_left hT, hev] at h
      simp only [Except.toOption, Option.bind_eq_bind, Option.bind_some, hrel, Option.pure_def] at h
      obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
      cases h
      have hnone : helloRoot l = none := by
        unfold helloRoot
        rw [hev]
        dsimp only
        rw [ite_eq_right (by rw [hnt]; decide)]
      refine .cons ⟨rfl, rfl, fun hne => ?_⟩ (ih hR (htl.sub fun l hl => List.mem_cons_of_mem _ hl)
        (by rwa [List.filterMap_cons_none hnone] at hn) hst (by simpa [freshOf] using hξ) hrest)
      unfold openCoins at hne ⊢
      cases hf : recs.find? (Verity.Sampled.recordTree · == Flock.hexOf root) with
      | none => rw [hf] at hne; exact absurd rfl hne
      | some rec =>
        cases hrr : Verity.Sampled.recordRound rec stream k with
        | none => simp [hf, hrr] at hne
        | some cs =>
          simp only [hrr, Option.map_some, Option.getD_some]
          have hrt := List.find?_some hf
          have hC := htl.coins l List.mem_cons_self j hev hnt hop _ _ _ _ (str_of hev hd ho.tree)
            (str_of hev hd ho.stream) (nat_of hev hd ho.round) (str_of hev hd ho.coins) rec
            (List.mem_of_find?_eq_some hf) (beq_iff_eq.1 hrt) cs hrr
          by_cases hcb : cs = cb
          · exact .inl (hcb ▸ rfl)
          · refine .inr ⟨rfl, ?_, ?_⟩ <;> rw [coinBytes_bits, coinBytes_bits]
            · exact hcb
            · exact hash_of_hex hC

/-- **A run the device printed reads as its log** (`ReadOf`): where the device printed its log `log` (`PrintsLog`), on
a run its ledger opens withheld on a window, whose release log reads as its coin commitment's (`releasedRootsOf`) and
whose log `coinTreesOf` reads trees from that each record passes, `readLog` reads the run's log, at the run's released
coins, as `log`: its head is the commitment's line (`readLog_head`), and the lines after it are what the device
printed. -/
theorem read_run {dev : History} {log : List Release} (hP : PrintsLog dev log) {pk : ByteArray} {w : String}
    {r : Properties.Challenger.Run} (hr : r ∈ dev.runs pk) (ho : OpensOn dev pk w r.id)
    (hcr : ∃ c roots, r.commitment = some c ∧ releasedRootsOf c r.log = .ok roots) {ts : Array LoggedTree}
    (hts : coinTreesOf r.log = .ok ts) (hrecs : ∀ rec ∈ recs, ∃ tabs, sessionTreeReason ts tabs rec = none)
    {out : List Release} (h : readLog label after recs (r.released label after) r.log = some out) :
    List.Forall₂ ReadOf out log := by
  obtain ⟨hp, hξ⟩ := hP pk w r hr ho
  obtain ⟨c, roots, -, hroots⟩ := id hcr
  rw [readLog_head hroots] at h
  have hT := trees_log hts
  have hn : (r.log.filterMap helloRoot).Nodup := hT.roots ▸ hT.nodup
  refine read_prints hp .nil ((tree_lines hts hrecs).sub fun _ hl => List.mem_of_mem_tail hl) ?_
    (fun _ h => absurd h List.not_mem_nil) (hξ hcr label after) h
  simpa using hn.sublist ((List.tail_sublist r.log).filterMap _)

/-! ## The openings without the sample's coins -/

/-- **Two reads of one release agree but for the sample's coins**: the same frame and bytes answered, and at a round
opened the same coins. -/
def Agree (y y' : Release) : Prop :=
  y'.frame = y.frame ∧ y'.answers = y.answers ∧ (opening? y.frame = true → y'.coins = y.coins)

/-- **`readTail` reads the same frames, bytes and openings' coins at any of the sample's coins.** -/
theorem readTail_agree {lines : List String} : ∀ {ts : List (String × Json)}
    {pending : Option String} {ξ : ByteArray} {out : List Release} (ξ' : ByteArray),
    readTail label after recs lines ts pending ξ = some out →
    ∃ out', readTail label after recs lines ts pending ξ' = some out' ∧ List.Forall₂ Agree out out' := by
  induction lines with
  | nil =>
    intro ts pending ξ out ξ' h
    cases pending with
    | none => rw [Verity.Sampled.readTail] at h; cases h; exact ⟨[], rfl, .nil⟩
    | some _ => rw [Verity.Sampled.readTail] at h; cases h
  | cons l ls ih =>
    intro ts pending ξ out ξ' h
    by_cases ht : treeLine l = true
    · cases pending
      all_goals
        rw [Verity.Sampled.readTail, ite_eq_left ht] at h
        rw [Verity.Sampled.readTail, ite_eq_left ht]
        cases hX : (event l).toOption.bind (treeStep recs ts) with
        | none => rw [hX] at h; cases h
        | some p =>
          obtain ⟨r, ts'⟩ := p
          rw [hX] at h
          simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def] at h ⊢
          obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
          cases h
          obtain ⟨rest', hrest', hag⟩ := ih ξ' hrest
          exact ⟨r :: rest', by rw [hrest']; rfl, .cons ⟨rfl, rfl, fun _ => rfl⟩ hag⟩
    · cases pending with
      | none =>
        rw [Verity.Sampled.readTail, ite_eq_right ht] at h
        rw [Verity.Sampled.readTail, ite_eq_right ht]
        exact ih ξ' h
      | some a =>
        rw [Verity.Sampled.readTail, ite_eq_right ht] at h
        rw [Verity.Sampled.readTail, ite_eq_right ht]
        dsimp only at h ⊢
        cases hR : Verity.Sampled.rootOf a with
        | none => rw [hR] at h; cases h
        | some p =>
          obtain ⟨lab, root⟩ := p
          simp only [hR, Option.bind_eq_bind, Option.bind_some, Option.pure_def] at h ⊢
          split at h
          · rename_i hc
            rw [ite_eq_left hc]
            obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
            cases h
            obtain ⟨rest', hrest', hag⟩ := ih (ξ'.extract COIN_BYTES ξ'.size) hrest
            refine ⟨⟨.fresh label, after, bitsOfBytes (ξ'.extract 0 COIN_BYTES)⟩ :: rest', ?_,
              .cons ⟨rfl, rfl, fun h => absurd h (by simp [opening?])⟩ hag⟩
            rw [hrest']
            rfl
          · rename_i hc
            rw [ite_eq_right hc]
            obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
            cases h
            obtain ⟨rest', hrest', hag⟩ := ih ξ' hrest
            refine ⟨⟨.fresh lab, root, []⟩ :: rest', ?_, .cons ⟨rfl, rfl, fun h => absurd h (by simp [opening?])⟩ hag⟩
            rw [hrest']
            rfl

/-- **`readLog` reads the same frames, bytes and openings' coins at any of the sample's coins**, its head's tree's line
included. -/
theorem readLog_agree {lines : List String} {ξ : ByteArray} {out : List Release} (ξ' : ByteArray)
    (h : readLog label after recs ξ lines = some out) :
    ∃ out', readLog label after recs ξ' lines = some out' ∧ List.Forall₂ Agree out out' := by
  cases lines with
  | nil => rw [Verity.Sampled.readLog] at h; cases h; exact ⟨[], rfl, .nil⟩
  | cons l ls =>
    by_cases ht : treeLine l = true
    · rw [Verity.Sampled.readLog, ite_eq_left ht] at h
      rw [Verity.Sampled.readLog, ite_eq_left ht]
      cases hX : (event l).toOption.bind (treeStep recs []) with
      | none => rw [hX] at h; cases h
      | some p =>
        obtain ⟨r, ts⟩ := p
        rw [hX] at h
        simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def] at h ⊢
        obtain ⟨rest, hrest, h⟩ := Option.bind_eq_some_iff.1 h
        cases h
        obtain ⟨rest', hrest', hag⟩ := readTail_agree ξ' hrest
        exact ⟨r :: rest', by rw [hrest']; rfl, .cons ⟨rfl, rfl, fun _ => rfl⟩ hag⟩
    · rw [Verity.Sampled.readLog, ite_eq_right ht] at h
      rw [Verity.Sampled.readLog, ite_eq_right ht]
      exact readTail_agree ξ' h

/-- **The openings as v2's check reads them, without the sample's coins**: `Verity.Sampled.coinsRead`'s read with none
of the sample's coins placed (`.empty`), so it verifies no signature of the sample; it hashes only the blobs it reads the
records from. -/
def opensRead (s : Sent) : Option (List Release) := do
  let (p, c, e) ← (caseOf s).toOption
  let (_, label) ← p.base.challenger
  let lines ← (linesOf e).toOption
  readLog label c.digest (← recordsOn (_root_.Core.Blobs.ofArray s.blobs) e) .empty lines

/-- **Where v2's check reads a log, the read without the sample's coins agrees with it** (`Agree`). -/
theorem opens_read {s : Sent} {out : List Release} (h : coinsRead s = some out) :
    ∃ out', opensRead s = some out' ∧ List.Forall₂ Agree out out' := by
  unfold Verity.Sampled.coinsRead at h
  unfold opensRead
  simp only [Option.bind_eq_bind] at h ⊢
  obtain ⟨⟨p, c, e⟩, h1, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨⟨kk, lab⟩, h2, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨ξ, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨lines, h4, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨recs, h5, h⟩ := Option.bind_eq_some_iff.1 h
  have h2' : p.base.challenger = some (kk, lab) := h2
  have h4' : (linesOf e).toOption = some lines := h4
  have h5' : recordsOn (_root_.Core.Blobs.ofArray s.blobs) e = some recs := h5
  have h' : readLog lab c.digest recs ξ lines = some out := h
  simp only [h1, h2', h4', h5', Option.bind_some]
  exact readLog_agree .empty h'

/-- **The first round opened whose coins the read shows other than the device released**: the read's bytes and the
device's, at a round opened whose coins the read shows. -/
def openCollision : List Release → List Release → Option (ByteArray × ByteArray)
  | y :: ys, x :: xs =>
    if opening? y.frame = true ∧ y.coins ≠ [] ∧ coinBytes y.coins ≠ coinBytes x.coins then
      some (coinBytes y.coins, coinBytes x.coins)
    else openCollision ys xs
  | _, _ => none

/-- **Where v2 reads a log of the device's releases but for openings' coins of one SHA-512, and it isn't the device's,
the first opening read without the sample's coins other than the device's is a SHA-512 collision.** -/
theorem open_collision : ∀ {out log out' : List Release}, List.Forall₂ ReadOf out log →
    List.Forall₂ Agree out out' → ¬ List.Forall₂ (fun y x => y.coins ≠ [] → y.coins = x.coins) out log →
    ∃ u v, openCollision out' log = some (u, v) ∧ u ≠ v ∧ Flock.Sha512.hash u = Flock.Sha512.hash v := by
  intro out log out' hr
  induction hr generalizing out' with
  | nil => intro _ hn; exact absurd .nil hn
  | @cons y x ys xs hyx _ ih =>
    intro ha hn
    cases ha with
    | @cons _ y' _ ys' hag ha =>
      rw [openCollision]
      by_cases hc : opening? y'.frame = true ∧ y'.coins ≠ [] ∧ coinBytes y'.coins ≠ coinBytes x.coins
      · rw [ite_eq_left hc]
        refine ⟨_, _, rfl, hc.2.2, ?_⟩
        have hop : opening? y.frame = true := hag.1 ▸ hc.1
        have hco : y'.coins = y.coins := hag.2.2 hop
        rw [hco] at hc ⊢
        rcases hyx.2.2 hc.2.1 with e | ⟨-, -, hh⟩
        · exact (hc.2.2 (by rw [e])).elim
        · exact hh
      · rw [ite_eq_right hc]
        refine ih ha fun hrest => hn (.cons (fun hne => ?_) hrest)
        rcases hyx.2.2 hne with e | ⟨hop, hne', -⟩
        · exact e
        · have hco : y'.coins = y.coins := hag.2.2 hop
          exact (hc ⟨hag.1 ▸ hop, hco ▸ hne, hco ▸ hne'⟩).elim

end Verity.Sampled.Lines
