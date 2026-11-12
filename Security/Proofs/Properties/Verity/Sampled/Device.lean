module

public import Proofs.Properties.Verity.Sampled.Lines
public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.NetworkCertifier.Check.LinkWindow
public import Proofs.Core.Game.Tape
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Init.Data.String.Defs

/-!
# The device's history at a play: the run that printed the play's log

`Lines.PrintsLog` is what the device's history at a play is to have for v2's read of a session to be the play's log
(`Binding.binds_of_prints`). This file builds that history from the play (`dev`) and proves it has them
(`printsLog_dev`), so `Binding.Binds` holds at it at every setup (`binds_dev`). At a deployment's setup it also signed
what the play's signatures are on (`signsPlay_dev`), so `Binding.logged_le` holds at it (`logged_le_dev`).

* **The lines** (`rootValue`, `releaseValue`, `helloValue`, `openValue`): the values the deployed challenger writes as
  its release log's lines (`LiveRecord._root` and `LiveRecord._release`, `LiveChallenger.sample_tree` and
  `LiveChallenger.open_tree` in `verity/core/protocols/spot/challenger.py`), each line its value's
  `Flock.canon` text (`lineOf`). Each value holds its keys in order and prints, so its line is its text
  (`canon_lineOf`), and it has the fields v2 reads (`root_value`, `release_plain`, `hello_value`, `open_value`).
* **The printer** (`printLog`, `prints_printLog`): a log's lines, from what the device holds between releases (its
  trees, each family's count of coins, each stream's count of rounds and the next line's place: `PrintState`); the lines
  it prints print the log (`Lines.Prints`).
* **Every play prints** (`printable_releases`, `printable_play`): the device releases only what the printer prints: a
  family's coin of `8 * COIN_BYTES` bits, a tree's root and key of whole bytes, and a round's coins of whole bytes, of a
  tree it sampled and a stream of that tree's spec.
* **The history** (`RunSetup`, `runOf`, `dev`): under the device's key, the run that released the play's log, its
  release log its head and the printed lines, its coins the play's where the play released them. It holds no run where
  the play's log releases more of a family's coins than the setup's layout holds (`Fits`), which no play under that
  layout does (`coins_fit`). At every play it is the history's one run (`dev_play`), and every run in it prints the
  play's log (`prints_dev`).
* **The roots v2 reads** (`view`, `RootsAfter`, `rootsOf_ok`, `prints_pairs`): where `releasedRootsOf` accepts a run's
  release log, each family's roots are the ones the first lines of its pairs of lines after the head, but the coin
  trees', show, in order; of lines the device printed, the bytes each fresh release of that family answered.
* **The coins it released** (`fold_released`, `position_place`, `holds_dev`): `Run.released` concatenates the run's
  coins at the positions of the family's coins whose root is the bytes after. Each has a position its layout places as
  that coin (`position_place`), where the run holds the play's coin, `COIN_BYTES` bytes of it (`coinAt`,
  `printLog_fresh`), so the run released of each family after each root the play's coins (`Lines.Holds`).
* **What the device signs** (`Deployment`, `Deployment.signer`, `signsPlay_dev`): a deployment fixes the run's setup
  from what the play fixed, under the parameters' key and at their layout, with the auditor's ledger lines the device
  signs before any coin and at the close. Its signer signs before any coin the messages of the lines the request fixes,
  and at the close the run's record and close and the messages of the lines at the close, each what
  `Properties.Challenger.signedBy` lists, so at a vocabulary whose signer is the deployment's, the device's history at
  each play signed what the play's signatures are on (`Binding.SignsPlay`). It signs at most `Q` messages in a play,
  one per ledger line and the run's record and close (`Deployment.signatures`, `Deployment.signs_le`), the vocabulary's
  `signs_le` at that signer.

What a run holds beside its releases (its id and host, its layout, the coins and salts the play didn't release, how it
ended, its lines' times and the ledger lines it signed) is the history's parameter (`RunSetup`), which a deployment
fixes from what the play fixed (`Deployment.setupOf`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private pairs from Spot.ChallengerSession

namespace Verity.Sampled.Device

open Lean (Json)
open Spot Spot.Challenger
open Properties.Challenger (History Run OpensOn)
open Verity.Sampled.Lines (Canon PlainValue RootValue SpecValue HelloValue OpenValue Prints)

/-! ## The lines the device writes -/

/-- **A root's line**, as the deployed challenger writes a family's root on its receipt (`LiveRecord._root`): coin `i`
of family `label`, received at `t`, after the bytes `b`. -/
def rootValue (label : String) (i t : ℕ) (b : ByteArray) : Json :=
  Json.mkObj [("event", .str "root"), ("index", Lean.toJson i), ("label", .str label), ("received_at", Lean.toJson t),
    ("root", .str (Flock.hexOf b))]

/-- **A release's line** (`LiveRecord._release`): coin `i` of family `label`, released at `t` after the bytes `b`. -/
def releaseValue (label : String) (i t : ℕ) (b : ByteArray) : Json :=
  Json.mkObj [("after", .str (Flock.hexOf b)), ("event", .str "release"), ("index", Lean.toJson i),
    ("label", .str label), ("released_at", Lean.toJson t)]

/-- **A coin tree's hello line** (`LiveChallenger.sample_tree`): the tree's key, its nonce, its root and its spec. -/
def helloValue (spec : Flock.CoinTree.Spec) (ν root key : ByteArray) : Json :=
  Json.mkObj [("event", .str TREE_EVENT), ("key", .str (Flock.hexOf key)), ("nonce", .str (Flock.hexOf ν)),
    ("root", .str (Flock.hexOf root)), ("spec", spec.toJson)]

/-- **A round opened's line** (`LiveChallenger.open_tree`): round `k` of stream `stream` of the tree of root `root`, its
first `n` coins, after the bytes `b`, the coins `cb` by their SHA-512. -/
def openValue (root : ByteArray) (stream : String) (k n : ℕ) (b cb : ByteArray) : Json :=
  Json.mkObj [("after", .str (Flock.hexOf b)), ("coins", .str (sha512Hex cb)), ("event", .str OPEN_EVENT),
    ("n", Lean.toJson n), ("round", Lean.toJson k), ("stream", .str stream), ("tree", .str (Flock.hexOf root))]

/-- A value's line: its `Flock.canon` text, as `canonical_json_text` writes it. -/
def lineOf (d : Json) : String :=
  match Flock.canon d with
  | .ok l => l
  | .error _ => ""

/-! ## Each line is its value's text -/

theorem sortedKeys_mkObj {l : List (String × Json)} (hl : (l.map (·.1)).Pairwise (· < ·))
    (hv : ∀ p ∈ l, sortedKeys p.2 = true) : sortedKeys (Json.mkObj l) = true := by
  have hwf : (Std.TreeMap.Raw.ofList l compare).WF := Std.TreeMap.Raw.WF.ofList
  rw [Json.mkObj, Properties.Challenger.CanonRead.sortedKeys_obj, Properties.Challenger.CanonRead.increasing_iff,
    Std.TreeMap.Raw.toList_toArray]
  refine ⟨?_, fun p hp => ?_⟩
  · rw [List.pairwise_map]
    exact (Std.TreeMap.Raw.ordered_keys_toList hwf).imp fun hab => compareOfLessAndEq_eq_lt.1 hab
  · have hp' := Array.mem_toList_iff.2 hp
    rw [Std.TreeMap.Raw.toList_toArray] at hp'
    have hg := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hwf).1 hp'
    rw [← Std.TreeMap.Raw.get?_eq_getElem?, Properties.Challenger.CanonRead.get?_ofList hl] at hg
    exact hv _ (Properties.Challenger.CanonRead.lookup_toList hg)

/-- A value that holds its keys in order and that `Flock.canon` prints. -/
def Printed (d : Json) : Prop := sortedKeys d = true ∧ ∃ t, Flock.canon d = .ok t

theorem printed_str (s : String) : Printed (.str s) :=
  ⟨by unfold sortedKeys; rfl, Properties.NetworkCertifier.canon_str s⟩

theorem printed_nat (n : ℕ) : Printed (Lean.toJson n) :=
  ⟨by unfold sortedKeys; rfl, Properties.NetworkCertifier.canon_nat n⟩

theorem printed_mkObj {l : List (String × Json)} (hl : (l.map (·.1)).Pairwise (· < ·))
    (hv : ∀ p ∈ l, Printed p.2) : Printed (Json.mkObj l) :=
  ⟨sortedKeys_mkObj hl fun p hp => (hv p hp).1, Properties.NetworkCertifier.canon_mkObj fun p hp => (hv p hp).2⟩

theorem printed_spec (spec : Flock.CoinTree.Spec) : Printed spec.toJson := by
  refine printed_mkObj (by simp only [List.map_cons, List.map_nil]; decide) fun p hp => ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl
  · exact printed_nat _
  · exact printed_nat _
  · exact printed_str _
  · refine ⟨Properties.Challenger.CanonRead.sortedKeys_arr.2 fun x hx => ?_,
      Properties.NetworkCertifier.canon_arr fun x hx => ?_⟩ <;>
    · obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
      first | exact (printed_str s).1 | exact (printed_str s).2

/-- **A printed value's line is its text.** -/
theorem canon_lineOf {d : Json} (h : Printed d) : Canon d (lineOf d) := by
  obtain ⟨hs, t, ht⟩ := h
  refine ⟨hs, ?_⟩
  unfold lineOf
  rw [ht]

theorem canon_root (label : String) (i t : ℕ) (b : ByteArray) :
    Canon (rootValue label i t b) (lineOf (rootValue label i t b)) := by
  refine canon_lineOf (printed_mkObj (by simp only [List.map_cons, List.map_nil]; decide) fun p hp => ?_)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl <;> first | exact printed_str _ | exact printed_nat _

theorem canon_release (label : String) (i t : ℕ) (b : ByteArray) :
    Canon (releaseValue label i t b) (lineOf (releaseValue label i t b)) := by
  refine canon_lineOf (printed_mkObj (by simp only [List.map_cons, List.map_nil]; decide) fun p hp => ?_)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl <;> first | exact printed_str _ | exact printed_nat _

theorem canon_hello (spec : Flock.CoinTree.Spec) (ν root key : ByteArray) :
    Canon (helloValue spec ν root key) (lineOf (helloValue spec ν root key)) := by
  refine canon_lineOf (printed_mkObj (by simp only [List.map_cons, List.map_nil]; decide) fun p hp => ?_)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl <;> first | exact printed_str _ | exact printed_spec _

theorem canon_open (root : ByteArray) (stream : String) (k n : ℕ) (b cb : ByteArray) :
    Canon (openValue root stream k n b cb) (lineOf (openValue root stream k n b cb)) := by
  refine canon_lineOf (printed_mkObj (by simp only [List.map_cons, List.map_nil]; decide) fun p hp => ?_)
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> first | exact printed_str _ | exact printed_nat _

/-! ## Each line has the fields v2 reads -/

private theorem nat_toJson (m : ℕ) : nat? (Lean.toJson m) = some m := by
  simp [nat?, int?, Lean.toJson, Lean.JsonNumber.fromNat]

theorem root_value (label : String) (i t : ℕ) (b : ByteArray) : RootValue (rootValue label i t b) label b :=
  ⟨⟨"root", rfl, by decide, by decide⟩, rfl, rfl⟩

theorem release_plain (label : String) (i t : ℕ) (b : ByteArray) : PlainValue (releaseValue label i t b) :=
  ⟨"release", rfl, by decide, by decide⟩

theorem hello_value (spec : Flock.CoinTree.Spec) (ν root key : ByteArray) :
    HelloValue (helloValue spec ν root key) spec ν root key :=
  ⟨rfl, rfl, rfl, rfl, ⟨rfl, by show nat? (Lean.toJson spec.roundsLog) = _; exact nat_toJson _,
    by show nat? (Lean.toJson spec.block) = _; exact nat_toJson _⟩⟩

theorem open_value (root : ByteArray) (stream : String) (k n : ℕ) (b cb : ByteArray) :
    OpenValue (openValue root stream k n b cb) root stream k n b cb :=
  ⟨rfl, rfl, rfl, by show nat? (Lean.toJson k) = _; exact nat_toJson _,
    by show nat? (Lean.toJson n) = _; exact nat_toJson _, rfl, rfl⟩

/-! ## The printer -/

/-- **What the device holds between releases, for its lines**: the coin trees it sampled, each its root and spec; each
family's count of coins released; each tree's streams' counts of rounds opened; and its release log's next line. -/
structure PrintState where
  trees : List (ByteArray × Flock.CoinTree.Spec)
  fresh : String → ℕ
  opened : ℕ → ℕ → ℕ
  line : ℕ

/-- Before any release: no tree, no coin, no round, and the line after the release log's head. -/
def PrintState.init : PrintState := ⟨[], fun _ => 0, fun _ _ => 0, 1⟩

/-- After a coin of family `label`: the family's count one more, and two lines on. -/
def PrintState.coin (st : PrintState) (label : String) : PrintState :=
  { st with
    fresh := fun l => if l = label then st.fresh label + 1 else st.fresh l
    line := st.line + 2 }

/-- After a tree's hello: the tree sampled, of root `root` and spec `spec`, and a line on. -/
def PrintState.tree (st : PrintState) (root : ByteArray) (spec : Flock.CoinTree.Spec) : PrintState :=
  { st with trees := st.trees ++ [(root, spec)], line := st.line + 1 }

/-- After a round of stream `s` of tree `i` opened: the stream's count one more, and a line on. -/
def PrintState.round (st : PrintState) (i s : ℕ) : PrintState :=
  { st with
    opened := fun i' s' => if i' = i ∧ s' = s then st.opened i s + 1 else st.opened i' s'
    line := st.line + 1 }

/-- **The release log's lines that print the releases `log`**, after the state `st`, at the times `time` gives each line
by its place in the log; `none` where the device prints no such release.

* A family's coin of `8 * COIN_BYTES` bits is its root's line and its release's line, its index the family's count of
  coins so far.
* A tree's hello whose bits are whole bytes is its line, its root the first 64 of the bytes released and its key the
  rest: the tree's root and key as the device releases them.
* A round opened whose bits are whole bytes, of a tree sampled and a stream of its spec, is its line, its round the
  stream's count of rounds opened so far and its coins the SHA-512 of the bytes released. -/
def printLog (time : ℕ → ℕ) : PrintState → List Release → Option (List String)
  | _, [] => some []
  | st, ⟨.fresh label, b, cs⟩ :: log =>
    if cs.length = 8 * COIN_BYTES then
      (printLog time (st.coin label) log).map fun ls =>
        lineOf (rootValue label (st.fresh label) (time st.line) b) ::
          lineOf (releaseValue label (st.fresh label) (time (st.line + 1)) b) :: ls
    else none
  | st, ⟨.hello spec, ν, cs⟩ :: log =>
    if cs.length % 8 = 0 then
      (printLog time (st.tree ((Lines.coinBytes cs).extract 0 64) spec) log).map
        (lineOf (helloValue spec ν ((Lines.coinBytes cs).extract 0 64)
          ((Lines.coinBytes cs).extract 64 (Lines.coinBytes cs).size)) :: ·)
    else none
  | st, ⟨.opening i s n, b, cs⟩ :: log =>
    match st.trees[i]? with
    | none => none
    | some (root, spec) =>
      match spec.streams[s]? with
      | none => none
      | some stream =>
        if cs.length % 8 = 0 then
          (printLog time (st.round i s) log).map
            (lineOf (openValue root stream (st.opened i s) n b (Lines.coinBytes cs)) :: ·)
        else none

/-- **What the printer prints, the device printed** (`Lines.Prints`): each line its value's text, with the fields v2
reads. -/
theorem prints_printLog (time : ℕ → ℕ) :
    ∀ (st : PrintState) (log : List Release) {lines : List String}, printLog time st log = some lines →
      Prints st.trees lines log
  | st, [], lines, h => by
    simp only [printLog, Option.some.injEq] at h
    subst h
    exact .nil _
  | st, ⟨.fresh label, b, cs⟩ :: log, lines, h => by
    simp only [printLog] at h
    split at h
    · obtain ⟨ls, hls, rfl⟩ := Option.map_eq_some_iff.1 h
      exact .fresh (canon_root ..) (root_value ..) (canon_release ..) (release_plain ..)
        (prints_printLog time (st.coin label) log hls)
    · cases h
  | st, ⟨.hello spec, ν, cs⟩ :: log, lines, h => by
    simp only [printLog] at h
    split at h
    · rename_i hcs
      obtain ⟨ls, hls, rfl⟩ := Option.map_eq_some_iff.1 h
      have hz : bitsOfBytes ((Lines.coinBytes cs).extract 0 64 ++
          (Lines.coinBytes cs).extract 64 (Lines.coinBytes cs).size) = cs := by
        rw [ByteArray.extract_append_extract, Nat.zero_min, ByteArray.extract_zero_max_size,
          Lines.bits_coinBytes hcs]
      have hp := Prints.hello
        (canon_hello spec ν ((Lines.coinBytes cs).extract 0 64)
          ((Lines.coinBytes cs).extract 64 (Lines.coinBytes cs).size))
        (hello_value ..) (prints_printLog time (st.tree ((Lines.coinBytes cs).extract 0 64) spec) log hls)
      rwa [hz] at hp
    · cases h
  | st, ⟨.opening i s n, b, cs⟩ :: log, lines, h => by
    simp only [printLog] at h
    split at h
    · cases h
    · rename_i root spec hts
      split at h
      · cases h
      · rename_i stream hst
        split at h
        · rename_i hcs
          obtain ⟨ls, hls, rfl⟩ := Option.map_eq_some_iff.1 h
          have hp := Prints.opening hts hst (canon_open root stream (st.opened i s) n b (Lines.coinBytes cs))
            (open_value ..) (prints_printLog time (st.round i s) log hls)
          rwa [Lines.bits_coinBytes hcs] at hp
        · cases h

/-! ## Every play prints -/

theorem set_self {α : Type} {l : List α} {i : ℕ} {a : α} (h : l[i]? = some a) : l.set i a = l := by
  obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.1 h
  exact List.set_getElem_self hi

theorem opened_length {block : ℕ} (d : TreeDraw block) (n : ℕ) : (TreeDraw.opened d n).length % 8 = 0 := by
  have h := congrArg List.length (Lines.opened_bytes d n)
  rw [Lines.bitsOfBytes_length] at h
  omega

/-- **The device prints every log its releases can end with**: at most `k` frames more under the families `fs`, with
the trees `ts` open, from a state whose trees' specs are theirs. -/
theorem printable_releases (time : ℕ → ℕ) :
    ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (st : PrintState) (log : List Release),
      st.trees.map (·.2) = ts.map (·.spec) → Game.Reaches (releases k fs ts) log →
      ∃ lines, printLog time st log = some lines
  | 0, _, _, st, log, _, h => by
    cases h
    exact ⟨[], rfl⟩
  | k + 1, fs, ts, st, log, hst, h => by
    obtain ⟨m, hm⟩ := h
    rcases m with _ | ⟨_ | _ | _, b⟩
    · cases hm
      exact ⟨[], rfl⟩
    · rename_i l
      cases hc : takeCoin l fs with
      | none =>
        simp only [hc] at hm
        cases hm
        exact ⟨[], rfl⟩
      | some fs' =>
        simp only [hc] at hm
        obtain ⟨x, hx⟩ := hm
        obtain ⟨log', h', hret⟩ := Game.reaches_bind hx
        cases hret
        obtain ⟨ls, hls⟩ := printable_releases time k fs' ts (st.coin l) log' hst h'
        have hlen : (List.ofFn x).length = 8 * COIN_BYTES := List.length_ofFn
        exact ⟨_, by simp only [printLog, ite_eq_left hlen, hls]; rfl⟩
    · rename_i spec
      simp only at hm
      split at hm
      · obtain ⟨t, ht⟩ := hm
        obtain ⟨log', h', hret⟩ := Game.reaches_bind ht
        cases hret
        have hcs : (bitsOfBytes (treeRoot b spec t ++ treeKey t.1)).length % 8 = 0 := by
          rw [Lines.bitsOfBytes_length]
          omega
        obtain ⟨ls, hls⟩ := printable_releases time k fs (ts ++ [⟨b, spec, t, fun _ => 0⟩])
          (st.tree ((Lines.coinBytes (bitsOfBytes (treeRoot b spec t ++ treeKey t.1))).extract 0 64) spec) log'
          (by
            show (st.trees ++ _).map (·.2) = _
            simp only [List.map_append, List.map_cons, List.map_nil, hst]) h'
        exact ⟨_, by simp only [printLog, ite_eq_left hcs, hls]; rfl⟩
      · cases hm
        exact ⟨[], rfl⟩
    · rename_i i s n
      simp only at hm
      split at hm
      · split at hm
        · cases hm
          exact ⟨[], rfl⟩
        · rename_i T hT
          split at hm
          · cases hm
            exact ⟨[], rfl⟩
          · rename_i s' r hn
            obtain ⟨d, hd⟩ := hm
            obtain ⟨log', h', hret⟩ := Game.reaches_bind hd
            cases hret
            have hi : (st.trees.map (·.2))[i]? = some T.spec := by
              rw [hst, List.getElem?_map, hT]
              rfl
            rw [List.getElem?_map] at hi
            obtain ⟨⟨root, spec⟩, hts, hsp⟩ := Option.map_eq_some_iff.1 hi
            simp only at hsp
            subst hsp
            have hs : s < T.spec.streams.size := by
              unfold OpenTree.next at hn
              split at hn
              · assumption
              · cases hn
            obtain ⟨ls, hls⟩ := printable_releases time k fs (ts.set i (T.bump s')) (st.round i s) log'
              (by
                show st.trees.map (·.2) = (ts.set i (T.bump s')).map (·.spec)
                rw [List.map_set, hst]
                exact (set_self (by rw [List.getElem?_map, hT]; rfl)).symm) h'
            exact ⟨_, by
              simp only [printLog, hts, Array.getElem?_eq_getElem hs, ite_eq_left (opened_length d.1 n), hls]; rfl⟩
      · cases hm
        exact ⟨[], rfl⟩

/-- **The device prints every play's log**, at any times. -/
theorem printable_play {fs : List (String × ℕ)} {D : Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {o : Exchange} (h : Game.Reaches (play fs D θ F c) o) (time : ℕ → ℕ) :
    ∃ lines, printLog time PrintState.init o.log = some lines := by
  obtain ⟨a, ha⟩ := h
  obtain ⟨log, hlog, hf⟩ := Game.reaches_bind ha
  obtain ⟨b, ρ, e, he⟩ := hf
  subst he
  exact printable_releases time _ fs [] PrintState.init log rfl hlog

/-! ## The device's history -/

/-- **What a run of the device holds beside its releases**: the key it runs under, its session's id and host, its coin
commitment's layout, its coins and salts by position where the play released none, how it ended, each line's time by
its place in its release log, and the auditor's ledger lines it signed. -/
structure RunSetup where
  key : ByteArray
  id : String
  host : String
  scheme : String
  epoch : ℕ
  families : Families
  coins : ℕ → ByteArray
  salts : ℕ → ByteArray
  ended : Option String
  time : ℕ → ℕ
  ledger : List Json

/-- The family and index of the coin at position `p` under the layout `fs`, as `LiveCommitment.position` places it. -/
def placeOf : Families → ℕ → Option (String × ℕ)
  | [], _ => none
  | (l, n) :: fs, p => if p < n then some (l, p) else placeOf fs (p - n)

/-- The coins of family `label` the releases `log` released, in order. -/
def coinsOf (label : String) : List Release → List (List Bool)
  | [] => []
  | x :: xs => match x.frame with
    | .fresh l => if l = label then x.coins :: coinsOf label xs else coinsOf label xs
    | _ => coinsOf label xs

/-- **The run's coin at position `p`**: the bytes of the coin the releases `log` released there, or the setup's. -/
def coinAt (S : RunSetup) (log : List Release) (p : ℕ) : ByteArray :=
  match placeOf S.families p with
  | some (l, i) =>
    match (coinsOf l log)[i]? with
    | some cs => Lines.coinBytes cs
    | none => S.coins p
  | none => S.coins p

/-- The release log's head, naming the coin commitment's root, as `releasedRootsOf` reads it. -/
def headLine (c : Option LiveCommitment) : String :=
  (c.map fun c => "{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}").getD ""

/-- **The run of the setup `S` that released `log`**, its release log its head and the lines `lines`. -/
def runOf (S : RunSetup) (log : List Release) (lines : List String) : Run :=
  let r : Run := ⟨S.id, S.host, S.scheme, S.epoch, S.families, coinAt S log, S.salts, [], S.ended⟩
  { r with log := headLine r.commitment :: lines }

/-- **The coins of family `label` the layout `fs` holds**: its first family of that name's, the ones
`LiveCommitment.position` places; none where no family has that name. -/
def firstCount (label : String) : Families → ℕ
  | [] => 0
  | (l, n) :: fs => if l = label then n else firstCount label fs

/-- **The releases `log` release at most each family's coins the layout `fs` holds**, as the device releases them
(`takeCoin`, `coins_fit`). -/
def Fits (fs : Families) (log : List Release) : Prop := ∀ label, (coinsOf label log).length ≤ firstCount label fs

open Classical in
/-- **The device's history at the play that ended with `o`**, of the setup `S o`: under its key, the run that released
the play's log, its release log the lines `printLog` prints of it after its head, and the ledger lines it signed. It
holds no run where the play's log releases more of a family's coins than the setup's layout holds, which no play under
that layout does (`coins_fit`). -/
noncomputable def dev (S : Exchange → RunSetup) (o : Exchange) : History where
  runs pk :=
    if pk = (S o).key ∧ Fits (S o).families o.log then
      match printLog (S o).time PrintState.init o.log with
      | some lines => [runOf (S o) o.log lines]
      | none => []
    else []
  ledger pk := if pk = (S o).key then (S o).ledger else []

theorem takeCoin_count {l : String} : ∀ {fs fs' : List (String × ℕ)}, takeCoin l fs = some fs' →
    ∀ label, firstCount label fs = firstCount label fs' + if l = label then 1 else 0
  | [], _, h, _ => by cases h
  | (l', n) :: fs, fs', h, label => by
    unfold takeCoin at h
    by_cases hl : l' = l
    · rw [ite_eq_left hl] at h
      cases n with
      | zero => cases h
      | succ m =>
        cases h
        subst hl
        simp only [firstCount]
        by_cases hL : l' = label
        · rw [ite_eq_left hL, ite_eq_left hL, ite_eq_left hL]
        · rw [ite_eq_right hL, ite_eq_right hL, ite_eq_right hL, Nat.add_zero]
    · rw [ite_eq_right hl] at h
      obtain ⟨fs'', h', rfl⟩ := Option.map_eq_some_iff.1 h
      simp only [firstCount]
      by_cases hL : l' = label
      · subst hL
        rw [ite_eq_left rfl, ite_eq_left rfl, ite_eq_right (Ne.symm hl), Nat.add_zero]
      · rw [ite_eq_right hL, ite_eq_right hL, takeCoin_count h' label]

/-- **The device releases at most each family's coins**: at most `k` frames more under the families `fs`, with the trees
`ts` open, a log releases at most the coins `fs` holds of each family. -/
theorem coins_fit : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree) (log : List Release),
    Game.Reaches (releases k fs ts) log → Fits fs log
  | 0, _, _, log, h => by
    cases h
    exact fun _ => Nat.zero_le _
  | k + 1, fs, ts, log, h => by
    obtain ⟨m, hm⟩ := h
    rcases m with _ | ⟨_ | _ | _, b⟩
    · cases hm
      exact fun _ => Nat.zero_le _
    · rename_i l
      cases hc : takeCoin l fs with
      | none =>
        simp only [hc] at hm
        cases hm
        exact fun _ => Nat.zero_le _
      | some fs' =>
        simp only [hc] at hm
        obtain ⟨x, hx⟩ := hm
        obtain ⟨log', h', hret⟩ := Game.reaches_bind hx
        cases hret
        intro label
        have ih := coins_fit k fs' ts log' h' label
        rw [takeCoin_count hc label]
        simp only [coinsOf]
        by_cases hl : l = label
        · rw [ite_eq_left hl, ite_eq_left hl, List.length_cons]
          omega
        · rw [ite_eq_right hl, ite_eq_right hl]
          omega
    · rename_i spec
      simp only at hm
      split at hm
      · obtain ⟨t, ht⟩ := hm
        obtain ⟨log', h', hret⟩ := Game.reaches_bind ht
        cases hret
        exact coins_fit k fs _ log' h'
      · cases hm
        exact fun _ => Nat.zero_le _
    · rename_i i s n
      simp only at hm
      split at hm
      · split at hm
        · cases hm
          exact fun _ => Nat.zero_le _
        · split at hm
          · cases hm
            exact fun _ => Nat.zero_le _
          · obtain ⟨d, hd⟩ := hm
            obtain ⟨log', h', hret⟩ := Game.reaches_bind hd
            cases hret
            exact coins_fit k fs _ log' h'
      · cases hm
        exact fun _ => Nat.zero_le _

/-- **At every play the device's history holds the play's run under its key, and no other run**, at a setup whose
layout is the play's families. -/
theorem dev_play (S : Exchange → RunSetup) {fs : List (String × ℕ)} {D : Signer} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {o : Exchange} (h : Game.Reaches (play fs D θ F c) o)
    (hS : (S o).families = fs) :
    ∃ lines, printLog (S o).time PrintState.init o.log = some lines ∧
      (dev S o).runs (S o).key = [runOf (S o) o.log lines] ∧ ∀ pk, pk ≠ (S o).key → (dev S o).runs pk = [] := by
  obtain ⟨lines, hl⟩ := printable_play h (S o).time
  have hfit : Fits (S o).families o.log := by
    obtain ⟨a, ha⟩ := h
    obtain ⟨log, hlog, hf⟩ := Game.reaches_bind ha
    obtain ⟨b, ρ, e, he⟩ := hf
    subst he
    rw [hS]
    exact coins_fit _ fs [] log hlog
  refine ⟨lines, hl, ?_, fun pk hpk => ?_⟩
  · simp only [dev]
    rw [ite_eq_left ⟨rfl, hfit⟩, hl]
  · simp only [dev]
    rw [ite_eq_right fun h => hpk h.1]

/-- **Each run of the device's history prints the play's log after its head** (`Lines.PrintsLog`'s first part). -/
theorem prints_dev (S : Exchange → RunSetup) (o : Exchange) {pk : ByteArray} {r : Run}
    (hr : r ∈ (dev S o).runs pk) : Prints [] r.log.tail o.log := by
  simp only [dev] at hr
  split at hr
  · split at hr
    · rename_i lines hl
      rw [List.mem_singleton] at hr
      subst hr
      exact prints_printLog (S o).time PrintState.init o.log hl
    · cases hr
  · cases hr

/-! ## The roots v2 reads of the run's release log -/

/-- **The roots `releasedRootsOf` holds of family `L` in the state `s`**: its first entry's of that family, as
`Run.released` reads them; none where no entry is of that family. -/
def view (s : Array (String × Array ByteArray)) (L : String) : List ByteArray :=
  ((s.toList.find? (·.1 == L)).map (·.2.toList)).getD []

theorem find?_modify {label L : String} {b : ByteArray} :
    ∀ {xs : List (String × Array ByteArray)} {k : ℕ}, xs.findIdx? (·.1 == label) = some k →
      (xs.modify k fun x => (x.fst, x.snd.push b)).find? (·.1 == L) =
        (xs.find? (·.1 == L)).map fun x => if x.1 = label then (x.fst, x.snd.push b) else x
  | [], _, h => by simp at h
  | x :: xs, k, h => by
    simp only [List.findIdx?_cons] at h
    by_cases hx : x.1 = label
    · rw [ite_eq_left (by simpa using hx)] at h
      cases h
      rw [List.modify_zero_cons, List.find?_cons, List.find?_cons]
      by_cases hL : x.1 = L
      · subst hL
        simp [hx]
      · simp only [beq_eq_false_iff_ne.2 hL]
        cases hf : xs.find? (·.1 == L) with
        | none => rfl
        | some y =>
          have hy : y.1 = L := by simpa using List.find?_some hf
          rw [Option.map_some, ite_eq_right fun e => hL (hx.trans (hy.symm.trans e).symm)]
    · rw [ite_eq_right (by simpa using hx)] at h
      obtain ⟨k', hk', rfl⟩ := Option.map_eq_some_iff.1 h
      rw [List.modify_succ_cons, List.find?_cons, List.find?_cons]
      by_cases hL : x.1 = L
      · simp only [hL, beq_self_eq_true, Option.map_some]
        rw [ite_eq_right (by rw [← hL] at *; exact hx)]
      · simp only [beq_eq_false_iff_ne.2 hL]
        exact find?_modify hk'

theorem view_modify {s : Array (String × Array ByteArray)} {label : String} {k : ℕ} {b : ByteArray}
    (h : s.findIdx? (·.1 == label) = some k) (L : String) :
    view (s.modify k fun x => (x.fst, x.snd.push b)) L = view s L ++ if label = L then [b] else [] := by
  rcases s with ⟨xs⟩
  have h' : xs.findIdx? (·.1 == label) = some k := by simpa using h
  unfold view
  rw [Array.toList_modify, find?_modify h']
  cases hf : xs.find? (·.1 == L) with
  | none =>
    simp only [Option.map_none, Option.getD_none, List.nil_append]
    split
    · rename_i hl
      subst hl
      obtain ⟨hk, hp, -⟩ := List.findIdx?_eq_some_iff_getElem.1 h'
      exact absurd hp (by simpa using List.find?_eq_none.1 hf _ (List.getElem_mem hk))
    · rfl
  | some y =>
    have hy : y.1 = L := by simpa using List.find?_some hf
    simp only [Option.map_some, Option.getD_some]
    by_cases hl : label = L
    · rw [ite_eq_left (hy.trans hl.symm), ite_eq_left hl, Array.toList_push]
    · rw [ite_eq_right (fun e => hl (e.symm.trans hy)), ite_eq_right hl, List.append_nil]

theorem view_init (fs : Families) (L : String) :
    view (fs.map fun x => (x.fst, (#[] : Array ByteArray))).toArray L = [] := by
  unfold view
  cases hf : (fs.map fun x => (x.fst, (#[] : Array ByteArray))).toArray.toList.find? (·.1 == L) with
  | none => rfl
  | some y =>
    have hy := List.mem_of_find?_eq_some hf
    simp only [List.mem_map] at hy
    obtain ⟨a, -, rfl⟩ := hy
    rfl

/-- **What `releasedRootsOf`'s loop holds after the pairs of lines `pre`**: each pair's lines parse (`event`), and each
family's roots in the state `s` are the ones `Verity.Sampled.rootOf` reads of those pairs' first lines, in order. -/
def RootsAfter (pre : List (String × String)) (s : Array (String × Array ByteArray)) : Prop :=
  (∀ p ∈ pre, (∃ j, event p.1 = .ok j) ∧ ∃ j, event p.2 = .ok j) ∧
    ∀ L, view s L = ((pre.filterMap fun p => Verity.Sampled.rootOf p.1).filter (·.1 == L)).map (·.2)

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **The roots `releasedRootsOf` accepts are its loop's** (`RootsAfter`) over the pairs of its log's lines after the
head that are no coin tree's, an even number of them. -/
theorem rootsOf_ok {c : LiveCommitment} {log : List String} {roots : Roots} (h : releasedRootsOf c log = .ok roots) :
    ∃ res, RootsAfter (pairs (log.tail.filter fun l => !treeLine l)) res ∧
      (log.tail.filter fun l => !treeLine l).length % 2 = 0 ∧
      roots = res.toList.map fun x => (x.fst, x.snd.toList) := by
  unfold releasedRootsOf at h
  dsimp only at h
  split at h
  · split at h
    · cases h
    · rename_i hodd
      obtain ⟨res, hres, hpure⟩ := bind_ok h
      refine ⟨res, ?_, by simp only [beq_iff_eq] at hodd; omega, (Except.ok.inj hpure).symm⟩
      have hI := Verity.Sampled.Trees.forIn_inv (I := RootsAfter) ?_ _ [] _ res
        ⟨fun _ h => absurd h List.not_mem_nil, fun L => by rw [view_init]; rfl⟩ hres
      · simpa using hI
      intro pre x s r hI hr
      cases hev : event x.fst with
      | error e => rw [hev] at hr; cases hr
      | ok ja =>
        rw [hev, Verity.Sampled.Trees.ok_bind] at hr
        cases hev' : event x.snd with
        | error e => rw [hev'] at hr; cases hr
        | ok jb =>
          rw [hev', Verity.Sampled.Trees.ok_bind] at hr
          split at hr
          · rename_i label hlab
            split at hr
            · rename_i k hk
              split at hr
              · rename_i root hroot
                split at hr
                · rename_i bytes hbytes
                  split at hr
                  · cases hr
                    have hro : Verity.Sampled.rootOf x.fst = some (label, bytes) := by
                      have h1 : (event x.fst).toOption = some ja := by rw [hev]; rfl
                      unfold Verity.Sampled.rootOf
                      rw [h1]
                      simp only [Option.bind_eq_bind, Option.bind_some, hlab, hroot, hbytes]
                      rfl
                    refine ⟨_, rfl, fun p hp => ?_, fun L => ?_⟩
                    · rcases List.mem_append.1 hp with hp | hp
                      · exact hI.1 p hp
                      · rw [List.mem_singleton.1 hp]
                        exact ⟨⟨ja, hev⟩, ⟨jb, hev'⟩⟩
                    · rw [view_modify hk L, hI.2 L, List.filterMap_append, List.filter_append, List.map_append]
                      congr 1
                      simp only [List.filterMap_cons, hro, List.filterMap_nil]
                      by_cases hL : label = L
                      · simp [hL]
                      · simp [hL]
                  · cases hr
                · cases hr
              · cases hr
            · cases hr
          · cases hr
  · cases h

theorem mem_pairs :
    ∀ {rest : List String}, rest.length % 2 = 0 → ∀ l ∈ rest, ∃ p ∈ pairs rest, l = p.1 ∨ l = p.2
  | [], _, _, hl => absurd hl List.not_mem_nil
  | [_], h, _, _ => by simp at h
  | a :: b :: rest, h, l, hl => by
    rcases List.mem_cons.1 hl with rfl | hl
    · exact ⟨(l, b), List.mem_cons_self, .inl rfl⟩
    · rcases List.mem_cons.1 hl with rfl | hl
      · exact ⟨(a, l), List.mem_cons_self, .inr rfl⟩
      · obtain ⟨p, hp, he⟩ := mem_pairs (rest := rest) (by simp only [List.length_cons] at h; omega) l hl
        exact ⟨p, List.mem_cons_of_mem _ hp, he⟩

/-- The family and the bytes each fresh release of `log` answered, in order. -/
def freshRoots : List Release → List (String × ByteArray)
  | [] => []
  | x :: xs => match x.frame with
    | .fresh l => (l, x.answers) :: freshRoots xs
    | _ => freshRoots xs

/-- **The roots v2 reads of lines the device printed are its fresh releases'**: where each line `releasedRootsOf` keeps
(no coin tree's) parses, its pairs' first lines read (`Verity.Sampled.rootOf`) as each fresh release's family and the
bytes it answered. -/
theorem prints_pairs : ∀ {ts : List (ByteArray × Flock.CoinTree.Spec)} {lines : List String} {log : List Release},
    Prints ts lines log → (∀ l ∈ lines.filter (fun l => !treeLine l), ∃ j, event l = .ok j) →
      (pairs (lines.filter fun l => !treeLine l)).filterMap (fun p => Verity.Sampled.rootOf p.1) = freshRoots log := by
  intro ts lines log hp
  induction hp with
  | nil ts => intro _; rfl
  | @fresh ts lines log d d' l l' lab b cs hd hr hd' hp' _ ih =>
    intro hev
    have hf : (l :: l' :: lines).filter (fun l => !treeLine l) = l :: l' :: lines.filter (fun l => !treeLine l) := by
      simp [Lines.treeLine_plain hd hr.plain, Lines.treeLine_plain hd' hp']
    rw [hf] at hev ⊢
    obtain ⟨j, hj⟩ := hev l List.mem_cons_self
    show List.filterMap _ ((l, l') :: pairs _) = (lab, b) :: freshRoots log
    rw [List.filterMap_cons_some (f := fun p : String × String => Verity.Sampled.rootOf p.1) (a := (l, l'))
      (Lines.rootOf_value hj hd hr), ih fun x hx => hev x (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hx))]
  | @hello ts lines log d l spec ν root key hd hh _ ih =>
    intro hev
    cases hevl : event l with
    | error e =>
      obtain ⟨j, hj⟩ := hev l (by simp [Lines.treeLine_error hevl])
      rw [hevl] at hj
      cases hj
    | ok j =>
      have hT : treeLine l = true := by
        rw [Lines.treeLine_ok hevl, Lines.str_of hevl hd hh.event]
        simp [jsonStrIs]
      have hf : (l :: lines).filter (fun l => !treeLine l) = lines.filter (fun l => !treeLine l) := by
        simp [hT]
      rw [hf] at hev ⊢
      exact ih hev
  | @opening ts lines log d l i s n k root b cb spec stream hi hs hd ho _ ih =>
    intro hev
    cases hevl : event l with
    | error e =>
      obtain ⟨j, hj⟩ := hev l (by simp [Lines.treeLine_error hevl])
      rw [hevl] at hj
      cases hj
    | ok j =>
      have hT : treeLine l = true := by
        rw [Lines.treeLine_ok hevl, Lines.str_of hevl hd ho.event]
        simp [jsonStrIs]
      have hf : (l :: lines).filter (fun l => !treeLine l) = lines.filter (fun l => !treeLine l) := by
        simp [hT]
      rw [hf] at hev ⊢
      exact ih hev

/-! ## The coins the run released -/

local instance lawfulBEqByteArrayDevice : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-- The coins of family `label` the releases `log` released, each with the bytes it answered, in order. -/
def freshList (label : String) : List Release → List (ByteArray × List Bool)
  | [] => []
  | x :: xs => match x.frame with
    | .fresh l => if l = label then (x.answers, x.coins) :: freshList label xs else freshList label xs
    | _ => freshList label xs

theorem coinsOf_eq (L : String) : ∀ log : List Release, coinsOf L log = (freshList L log).map (·.2)
  | [] => rfl
  | x :: xs => by
    rcases x with ⟨_ | _ | _, b, cs⟩
    · rename_i l
      simp only [coinsOf, freshList]
      by_cases hl : l = L
      · simp [hl, coinsOf_eq L xs]
      · simp [hl, coinsOf_eq L xs]
    · exact coinsOf_eq L xs
    · exact coinsOf_eq L xs

theorem freshOf_eq (L : String) (after : ByteArray) : ∀ log : List Release,
    Lines.freshOf L after log = ((freshList L log).filter (·.1 == after)).map (·.2)
  | [] => rfl
  | x :: xs => by
    rcases x with ⟨_ | _ | _, b, cs⟩
    · rename_i l
      simp only [Lines.freshOf, freshList]
      by_cases hl : l = L
      · by_cases hb : b = after
        · simp [hl, hb, freshOf_eq L after xs]
        · simp [hl, hb, freshOf_eq L after xs]
      · simp [hl, freshOf_eq L after xs]
    · exact freshOf_eq L after xs
    · exact freshOf_eq L after xs

theorem freshRoots_filter (L : String) : ∀ log : List Release,
    ((freshRoots log).filter (·.1 == L)).map (·.2) = (freshList L log).map (·.1)
  | [] => rfl
  | x :: xs => by
    rcases x with ⟨_ | _ | _, b, cs⟩
    · rename_i l
      simp only [freshRoots, freshList]
      by_cases hl : l = L
      · simp [hl, freshRoots_filter L xs]
      · simp [hl, freshRoots_filter L xs]
    · exact freshRoots_filter L xs
    · exact freshRoots_filter L xs

/-- The bytes of the coins `css`, each coin's (`Lines.coinBytes`) in order. -/
def catBytes : List (List Bool) → ByteArray
  | [] => .empty
  | cs :: css => Lines.coinBytes cs ++ catBytes css

theorem fold_released (after : ByteArray) (step : ByteArray → ByteArray × ℕ → ByteArray) :
    ∀ (F : List (ByteArray × List Bool)) (i : ℕ) (acc : ByteArray),
      (∀ j (hj : j < F.length) a, step a (F[j].1, i + j) = a ++ Lines.coinBytes F[j].2) →
      (((F.map (·.1)).zipIdx i).filter (·.1 == after)).foldl step acc =
        acc ++ catBytes ((F.filter (·.1 == after)).map (·.2))
  | [], _, acc, _ => by simp [catBytes, ByteArray.append_empty]
  | (a, cs) :: F, i, acc, h => by
    have h0 : ∀ x, step x (a, i) = x ++ Lines.coinBytes cs := fun x => h 0 (by simp) x
    have ih := fold_released after step F (i + 1)
    simp only [List.map_cons, List.zipIdx_cons, List.filter_cons]
    by_cases ha : (a == after) = true
    · rw [ite_eq_left ha, ite_eq_left ha, List.foldl_cons, h0, List.map_cons, catBytes, ← ByteArray.append_assoc]
      refine ih _ fun j hj x => ?_
      have := h (j + 1) (by simp; omega) x
      simpa [Nat.add_assoc, Nat.add_comm 1 j] using this
    · rw [ite_eq_right ha, ite_eq_right ha]
      refine ih _ fun j hj x => ?_
      have := h (j + 1) (by simp; omega) x
      simpa [Nat.add_assoc, Nat.add_comm 1 j] using this

theorem holds_catBytes : ∀ {css : List (List Bool)}, (∀ cs ∈ css, cs.length = 8 * COIN_BYTES) →
    Lines.Holds (catBytes css) css
  | [], _ => .nil _
  | cs :: css, h => by
    have hl := h cs List.mem_cons_self
    have hs : (Lines.coinBytes cs).size = COIN_BYTES := by rw [Lines.coinBytes_size, hl]; omega
    have hh := Lines.Holds.cons hs (holds_catBytes fun c hc => h c (List.mem_cons_of_mem _ hc))
    rwa [Lines.bits_coinBytes (by rw [hl]; omega)] at hh

theorem go_place (label : String) (i : ℕ) : ∀ (fs : Families) (offset : ℕ), i < firstCount label fs →
    ∃ p, LiveCommitment.position.go label (i : Int) fs offset = some (offset + p) ∧ placeOf fs p = some (label, i)
  | [], _, h => absurd h (Nat.not_lt_zero _)
  | (l, n) :: fs, offset, h => by
    simp only [firstCount] at h
    by_cases hl : l = label
    · rw [ite_eq_left hl] at h
      refine ⟨i, ?_, ?_⟩
      · simp [LiveCommitment.position.go, hl, h]
      · simp [placeOf, h, hl]
    · rw [ite_eq_right hl] at h
      obtain ⟨p, hp, hpl⟩ := go_place label i fs (offset + n) h
      refine ⟨n + p, ?_, ?_⟩
      · rw [LiveCommitment.position.go, ite_eq_right (by simpa using hl), hp, Nat.add_assoc]
      · rw [placeOf, ite_eq_right (show ¬n + p < n by omega), Nat.add_sub_cancel_left, hpl]

/-- **A coin of family `label` below its count has a position**, which the layout places as that coin. -/
theorem position_place {c : LiveCommitment} {label : String} {i : ℕ} (h : i < firstCount label c.families) :
    ∃ p, c.position label i = some p ∧ placeOf c.families p = some (label, i) := by
  obtain ⟨p, hp, hpl⟩ := go_place label i c.families 0 h
  exact ⟨p, by rw [LiveCommitment.position, hp, Nat.zero_add], hpl⟩

/-- **The printer prints only fresh coins of `8 * COIN_BYTES` bits**. -/
theorem printLog_fresh (time : ℕ → ℕ) (label : String) :
    ∀ (st : PrintState) (log : List Release) {lines : List String}, printLog time st log = some lines →
      ∀ p ∈ freshList label log, p.2.length = 8 * COIN_BYTES
  | _, [], _, _ => fun _ h => absurd h List.not_mem_nil
  | st, ⟨.fresh l, b, cs⟩ :: log, lines, h => by
    simp only [printLog] at h
    split at h
    · rename_i hcs
      obtain ⟨ls, hls, -⟩ := Option.map_eq_some_iff.1 h
      intro p hp
      simp only [freshList] at hp
      split at hp
      · rcases List.mem_cons.1 hp with rfl | hp
        · exact hcs
        · exact printLog_fresh time label _ log hls p hp
      · exact printLog_fresh time label _ log hls p hp
    · cases h
  | st, ⟨.hello spec, ν, cs⟩ :: log, lines, h => by
    simp only [printLog] at h
    split at h
    · obtain ⟨ls, hls, -⟩ := Option.map_eq_some_iff.1 h
      exact printLog_fresh time label _ log hls
    · cases h
  | st, ⟨.opening i s n, b, cs⟩ :: log, lines, h => by
    simp only [printLog] at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · obtain ⟨ls, hls, -⟩ := Option.map_eq_some_iff.1 h
          exact printLog_fresh time label _ log hls
        · cases h

theorem commitment_families {s : Run} {c : LiveCommitment} (h : s.commitment = some c) : c.families = s.families := by
  unfold Properties.Challenger.Run.commitment at h
  split at h
  · cases h
    rfl
  · cases h

/-- **Where v2 reads a run of the device's history, it released of each family after each root the play's coins**
(`Lines.PrintsLog`'s second part): at the run's commitment, where `releasedRootsOf` accepts its release log, the roots
it reads are the play's fresh releases' (`rootsOf_ok`, `prints_pairs`), each coin has a position its layout places as
that coin (`position_place`, where the play's log fits the layout), and the run's coin there is the play's
(`coinAt`). -/
theorem holds_dev (S : Exchange → RunSetup) (o : Exchange) {pk : ByteArray} {r : Run} (hr : r ∈ (dev S o).runs pk)
    {c : LiveCommitment} {roots : Roots} (hc : r.commitment = some c) (hroots : releasedRootsOf c r.log = .ok roots)
    (label : String) (after : ByteArray) :
    Lines.Holds (r.released label after) (Lines.freshOf label after o.log) := by
  simp only [dev] at hr
  split at hr
  · rename_i hk
    split at hr
    · rename_i lines hl
      rw [List.mem_singleton] at hr
      subst hr
      have hfam : c.families = (S o).families := commitment_families hc
      have hp := prints_printLog (S o).time PrintState.init o.log hl
      obtain ⟨res, hI, hev2, hres⟩ := rootsOf_ok hroots
      have e : (runOf (S o) o.log lines).log.tail = lines := rfl
      rw [e] at hI hev2
      have hevs : ∀ l ∈ lines.filter (fun l => !treeLine l), ∃ j, event l = .ok j := by
        intro l hl
        obtain ⟨p, hp, he⟩ := mem_pairs hev2 l hl
        rcases he with rfl | rfl
        · exact (hI.1 p hp).1
        · exact (hI.1 p hp).2
      have hrs : ((roots.find? (·.1 == label)).map (·.2)).getD [] = (freshList label o.log).map (·.1) := by
        rw [hres, List.find?_map, Option.map_map]
        rw [← freshRoots_filter, ← prints_pairs hp hevs, ← hI.2 label]
        rfl
      simp only [Properties.Challenger.Run.released, hc, hroots]
      rw [hrs, fold_released after _ _ 0 .empty, ByteArray.empty_append, freshOf_eq]
      · exact holds_catBytes fun cs hcs => by
          obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hcs
          exact printLog_fresh _ label _ _ hl q (List.mem_filter.1 hq).1
      · intro j hj a
        have hfit : j < firstCount label c.families := by
          rw [hfam]
          have := hk.2 label
          rw [coinsOf_eq, List.length_map] at this
          omega
        obtain ⟨p, hpos, hpl⟩ := position_place hfit
        simp only [Nat.zero_add, hpos]
        show a ++ coinAt (S o) o.log p = _
        rw [hfam] at hpl
        have hget : (coinsOf label o.log)[j]? = some (freshList label o.log)[j].2 := by
          rw [coinsOf_eq, List.getElem?_map, List.getElem?_eq_getElem hj]
          rfl
        simp only [coinAt, hpl, hget]
    · cases hr
  · cases hr

/-- **The device printed the play's log** (`Lines.PrintsLog`), at every setup: each run of its history prints the
play's log after its head (`prints_dev`), and where v2 reads it, released the play's coins (`holds_dev`). -/
theorem printsLog_dev (S : Exchange → RunSetup) (o : Exchange) : Lines.PrintsLog (dev S o) o.log :=
  fun _ _ _ hr _ => ⟨prints_dev S o hr, fun ⟨_, _, hc, hroots⟩ => holds_dev S o hr hc hroots⟩

/-- **`Binds` at v2's read, at the device's history at each play** (`Binding.binds_of_prints`): at every setup, each
play's history printed its log (`printsLog_dev`). -/
theorem binds_dev {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    (k : Verity.Registration.Composite) (S : Exchange → RunSetup) (θ : ByteArray) :
    Binding.Binds V k Binding.readsV2 (dev S) θ :=
  Binding.binds_of_prints V k (dev S) θ fun o _ => printsLog_dev S o

/-! ## What the device signs at a play -/

/-- The messages of a play's signatures are the messages its signer signs, in order. -/
theorem signatures_fst (D : Signer) (a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    (D.signatures a log b ρ).map Prod.fst = D.signs a log b ρ := by
  unfold Signer.signatures
  exact List.ext_getElem (by simp) fun i _ _ => by simp

/-- **A history signs under `pk` what another signs there**, where it holds the other's runs and ledger lines under
`pk` (`Properties.Challenger.signedBy`). -/
theorem signedBy_mono {h h' : History} {pk m : ByteArray} (hm : m ∈ Properties.Challenger.signedBy h pk)
    (hr : ∀ s ∈ h.runs pk, s ∈ h'.runs pk) (hl : ∀ l ∈ h.ledger pk, l ∈ h'.ledger pk) :
    m ∈ Properties.Challenger.signedBy h' pk := by
  unfold Properties.Challenger.signedBy at hm ⊢
  rcases List.mem_append.1 hm with hm | hm
  · obtain ⟨s, hs, hm⟩ := List.mem_flatMap.1 hm
    exact List.mem_append.2 (.inl (List.mem_flatMap.2 ⟨s, hr s hs, hm⟩))
  · obtain ⟨l, hl', hm⟩ := List.mem_filterMap.1 hm
    exact List.mem_append.2 (.inr (List.mem_filterMap.2 ⟨l, hl l hl', hm⟩))

theorem length_flatMap_le {β γ : Type} (f : β → List γ) {n : ℕ} (hf : ∀ b, (f b).length ≤ n) :
    ∀ l : List β, (l.flatMap f).length ≤ n * l.length
  | [] => Nat.zero_le _
  | b :: l => by
    rw [List.flatMap_cons, List.length_append, List.length_cons, Nat.mul_succ, Nat.add_comm (n * _)]
    exact Nat.add_le_add (hf b) (length_flatMap_le f hf l)

/-- **A history signs under `pk` at most two messages per run and one per ledger line**
(`Properties.Challenger.signedBy`): a run's record and close, and a line's entry. -/
theorem signedBy_length (h : History) (pk : ByteArray) :
    (Properties.Challenger.signedBy h pk).length ≤ 2 * (h.runs pk).length + (h.ledger pk).length := by
  unfold Properties.Challenger.signedBy
  rw [List.length_append]
  refine Nat.add_le_add (length_flatMap_le _ (fun s => ?_) _) (List.length_filterMap_le _ _)
  split
  · exact Nat.zero_le _
  · have ho : ∀ {γ : Type} (o : Option γ), o.toList.length ≤ 1 := fun o => by cases o <;> simp
    rw [List.length_append]
    exact Nat.add_le_add (ho _) (ho _)

/-- **The run of the setup `s` that released `log`**, its release log its head and the lines the printer prints of
`log` (`printLog`); none where it prints none. -/
def printedRun (s : RunSetup) (log : List Release) : List Run :=
  match printLog s.time PrintState.init log with
  | some lines => [runOf s log lines]
  | none => []

theorem printedRun_eq {s : RunSetup} {log : List Release} {lines : List String}
    (h : printLog s.time PrintState.init log = some lines) : printedRun s log = [runOf s log lines] := by
  simp only [printedRun, h]

theorem printedRun_length (s : RunSetup) (log : List Release) : (printedRun s log).length ≤ 1 := by
  unfold printedRun
  split <;> simp

/-- The public key of the parameters' challenger key (`Binding.keyOf`), the empty string where they name none. -/
def pubOf (θ : ByteArray) : ByteArray := ((Binding.keyOf θ).map (·.pub)).getD .empty

/-- **A deployment of the device**, at each of the parameters `θ`, from what a play fixed: the prover's request `a`,
the device's log, the bytes `b` the prover's close left to the device and its sample `ρ` at the close. `run` is the
run's setup but for its key, its layout and its ledger lines, which `setupAt` sets: its session's id and host, its
coins and salts by position where the play released none, how it ended and each line's time. `opens` are the auditor's
ledger lines it signs before any coin, which the request fixes (the ledger's reserve and open, and the period's open),
and `closes` the ones it signs at the close (the ledger's end and the period's close), at most `ledgerLines θ` of
them in a play (`ledger_le`). `draws` are the bits it samples at the close, and `sign θ i m` its signature on `m` as its
`i`-th in the play.

A deployment opens only the auditor's live window: the window its ledger's open names (`OpensOn`'s `w`) is the one the
auditor runs, never one the prover's side chooses, since nothing the verifier keys on comes from the prover's side.
Here `opens θ a` reads the prover's request `a`, and nothing in this structure ties the window it opens to the
auditor's. `logged_le_dev` reads only the play's log (`Binding.Logged`), so it holds at every `opens`; the leg's
assembly takes the live-window rule as a hypothesis or a field. -/
structure Deployment where
  run : ByteArray → ByteArray → List Release → ByteArray → List Bool → RunSetup
  opens : ByteArray → ByteArray → List Json
  closes : ByteArray → ByteArray → List Release → ByteArray → List Bool → List Json
  ledgerLines : ByteArray → ℕ
  ledger_le : ∀ θ a log b ρ, (opens θ a).length + (closes θ a log b ρ).length ≤ ledgerLines θ
  draws : ByteArray → ℕ
  sign : ByteArray → ℕ → Hash.Msg → Hash.Msg

namespace Deployment

variable (P : Deployment)

/-- **The run's setup at a play**, at the layouts `fs` the parameters fix: `run`'s, under the parameters' key
(`pubOf`), at their layout, with the ledger lines it signs before any coin and at the close. -/
def setupAt (fs : ByteArray → Families) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    RunSetup :=
  { P.run θ a log b ρ with key := pubOf θ, families := fs θ, ledger := P.opens θ a ++ P.closes θ a log b ρ }

/-- The run's setup at the play that ended with `o`. -/
def setupOf (fs : ByteArray → Families) (o : Exchange) : RunSetup :=
  P.setupAt fs o.params o.request o.log o.close o.drawn

/-- **The deployed device's signer at the parameters `θ`**, at the layouts `fs`: before any coin, the messages of
its ledger lines the request fixes (`opens`); at the close, its record and its close of the run of the setup that
released the log (`printedRun`), and the messages of its ledger lines at the close (`closes`); each what
`Properties.Challenger.signedBy` lists of a history of those runs and lines under the parameters' key. -/
def signer (fs : ByteArray → Families) (θ : ByteArray) : Signer where
  first a := (Properties.Challenger.signedBy ⟨fun _ => [], fun _ => P.opens θ a⟩ (pubOf θ)).map (·.data.toList)
  draws := P.draws θ
  last a log b ρ :=
    (Properties.Challenger.signedBy ⟨fun _ => printedRun (P.setupAt fs θ a log b ρ) log,
      fun _ => P.closes θ a log b ρ⟩ (pubOf θ)).map (·.data.toList)
  sign := P.sign θ

/-- **`Q` at the deployment**: the most messages its device signs in a play, one per ledger line it signs
(`ledgerLines`) and the run's record and close. -/
def signatures (θ : ByteArray) : ℕ := P.ledgerLines θ + 2

/-- **The deployed device signs at most `Q` messages in a play** (`Vocab.signs_le`): before any coin at most one per
line the request fixes, and at the close at most the printed run's record and close and one per line at the close
(`signedBy_length`, `printedRun_length`). A vocabulary whose device is the deployment's has `signer := P.signer
families`, `signatures := P.signatures` and `signs_le := P.signs_le families`, at which `logged_le_dev`'s tie is
`rfl`. -/
theorem signs_le (fs : ByteArray → Families) (θ a : ByteArray) (log : List Release) (b : ByteArray)
    (ρ : List Bool) : ((P.signer fs θ).signs a log b ρ).length ≤ P.signatures θ := by
  have h1 : (Properties.Challenger.signedBy ⟨fun _ => [], fun _ => P.opens θ a⟩ (pubOf θ)).length ≤
      2 * ([] : List Run).length + (P.opens θ a).length :=
    signedBy_length _ _
  have h2 : (Properties.Challenger.signedBy ⟨fun _ => printedRun (P.setupAt fs θ a log b ρ) log,
      fun _ => P.closes θ a log b ρ⟩ (pubOf θ)).length ≤
      2 * (printedRun (P.setupAt fs θ a log b ρ) log).length + (P.closes θ a log b ρ).length :=
    signedBy_length _ _
  have h3 := printedRun_length (P.setupAt fs θ a log b ρ) log
  have h4 := P.ledger_le θ a log b ρ
  rw [List.length_nil] at h1
  show ((Properties.Challenger.signedBy ⟨fun _ => [], fun _ => P.opens θ a⟩ (pubOf θ)).map (·.data.toList) ++
      (Properties.Challenger.signedBy ⟨fun _ => printedRun (P.setupAt fs θ a log b ρ) log,
        fun _ => P.closes θ a log b ρ⟩ (pubOf θ)).map (·.data.toList)).length ≤ P.ledgerLines θ + 2
  rw [List.length_append, List.length_map, List.length_map]
  omega

end Deployment

/-- **The deployed device's history at each play signed what the play's signatures are on** (`Binding.SignsPlay`),
at a vocabulary whose signer at the parameters is the deployment's at the vocabulary's layouts. Each message the play's
signer signs is one a history of the setup's ledger lines, or of the run that released the play's log, signs
(`Deployment.signer`), and the device's history at the play holds that run (`dev_play`) and those lines under the
parameters' key. -/
theorem signsPlay_dev {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α) (P : Deployment)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (hD : V.signer θ = P.signer V.families θ) :
    Binding.SignsPlay V (dev (P.setupOf V.families)) θ F c := by
  intro o ho key hk m hm
  obtain hθ := (Binding.play_reaches ho).1
  subst hθ
  have hkey : (P.setupOf V.families o).key = key.pub :=
    congrArg (fun k => (k.map (·.pub)).getD ByteArray.empty) hk
  obtain ⟨lines, hl, hruns, -⟩ := dev_play (P.setupOf V.families) ho rfl
  have hled : (dev (P.setupOf V.families) o).ledger (P.setupOf V.families o).key =
      (P.setupOf V.families o).ledger :=
    ite_eq_left rfl
  rw [hkey] at hruns hled
  have hm' : m.1 ∈ ((P.signer V.families o.params).signatures o.request o.log o.close o.drawn).map Prod.fst := by
    rw [← hD, ← Binding.play_signed ho]
    exact List.mem_map.2 ⟨m, hm, rfl⟩
  rw [signatures_fst] at hm'
  change m.1 ∈ (Properties.Challenger.signedBy ⟨fun _ => [], fun _ => P.opens o.params o.request⟩
      (pubOf o.params)).map (·.data.toList) ++ (Properties.Challenger.signedBy
      ⟨fun _ => printedRun (P.setupOf V.families o) o.log,
        fun _ => P.closes o.params o.request o.log o.close o.drawn⟩ (pubOf o.params)).map (·.data.toList) at hm'
  rw [← List.map_append] at hm'
  obtain ⟨x, hx, hxm⟩ := List.mem_map.1 hm'
  have hmx : ({ data := m.1.toArray } : ByteArray) = x :=
    (congrArg (fun l : List UInt8 => ({ data := l.toArray } : ByteArray)) hxm).symm.trans (Binding.bytes_ofList x)
  have hx' : x ∈ Properties.Challenger.signedBy ⟨fun _ => [], fun _ => P.opens o.params o.request⟩ key.pub ++
      Properties.Challenger.signedBy ⟨fun _ => printedRun (P.setupOf V.families o) o.log,
        fun _ => P.closes o.params o.request o.log o.close o.drawn⟩ key.pub := hkey ▸ hx
  show Properties.Challenger.Signed _ _ _
  rw [hmx]
  rcases List.mem_append.1 hx' with hx | hx
  · apply signedBy_mono hx
    · intro s hs
      exact absurd hs List.not_mem_nil
    · intro l hl'
      rw [hled]
      exact List.mem_append.2 (.inl hl')
  · apply signedBy_mono hx
    · intro s hs
      rw [hruns]
      change s ∈ printedRun (P.setupOf V.families o) o.log at hs
      rwa [printedRun_eq hl] at hs
    · intro l hl'
      rw [hled]
      exact List.mem_append.2 (.inr hl')

/-- **(a) at the deployed device's history**: at a vocabulary whose signer at the parameters is the deployment's, an
accepted play whose log v2 read isn't the device's is at most as likely as the read finder's collision plus the
forger's forgery (`Binding.logged_le`), the device's history at each play having printed the play's log (`binds_dev`)
and signed what the play's signatures are on (`signsPlay_dev`). -/
theorem logged_le_dev {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    (k : Verity.Registration.Composite) (P : Deployment) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (hD : V.signer θ = P.signer V.families θ) :
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Binding.Logged Binding.readsV2 o) (V.game θ F c) σ ≤
      ENNReal.ofReal ((Binding.readFinder V (dev (P.setupOf V.families)) θ F c σ).Pr Verity.sha512Msg) +
        ENNReal.ofReal ((Binding.forger V (dev (P.setupOf V.families)) θ F c σ).Pr Verity.lmsVerify) :=
  Binding.logged_le V k Binding.readsV2 (dev (P.setupOf V.families)) θ F c σ
    (binds_dev V k (P.setupOf V.families) θ) (signsPlay_dev V P θ F c hD)

end Verity.Sampled.Device
