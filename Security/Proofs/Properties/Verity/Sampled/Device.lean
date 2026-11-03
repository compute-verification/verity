import Proofs.Properties.Verity.Sampled.Lines
import Proofs.Properties.NetworkCertifier.LinkWindow
import Proofs.Core.Game.Tape

/-!
# The device's history at a play: the run that printed the play's log

`Lines.PrintsLog` is what the device's history at a play is to have for v2's read of a session to be the play's log
(`Binding.binds_of_prints`). This file builds that history from the play (`dev`) and proves its first part: each run in
it prints the play's log after its release log's head (`prints_dev`).

* **The lines** (`rootValue`, `releaseValue`, `helloValue`, `openValue`): the values the deployed challenger writes as
  its release log's lines (`LiveRecord._root` and `LiveRecord._release`, `LiveChallenger.sample_tree` and
  `LiveChallenger.open_tree` in `verity/core/protocols/sampled_proofs/challenger.py`), each line its value's
  `Flock.canon` text (`lineOf`). Each value holds its keys in order and prints, so its line is its text
  (`canon_lineOf`), and it has the fields v2 reads (`root_value`, `release_plain`, `hello_value`, `open_value`).
* **The printer** (`printLog`, `prints_printLog`): a log's lines, from what the device holds between releases (its
  trees, each family's count of coins, each stream's count of rounds and the next line's place: `PrintState`); the lines
  it prints print the log (`Lines.Prints`).
* **Every play prints** (`printable_releases`, `printable_play`): the device releases only what the printer prints: a
  family's coin of `8 * COIN_BYTES` bits, a tree's root and key of whole bytes, and a round's coins of whole bytes, of a
  tree it sampled and a stream of that tree's spec.
* **The history** (`RunSetup`, `runOf`, `dev`): under the device's key, the run that released the play's log, its
  release log its head and the printed lines, its coins the play's where the play released them. At every play it is
  the history's one run (`dev_play`), and every run in it prints the play's log (`prints_dev`).

What a run holds beside its releases (its id and host, its layout, the coins and salts the play didn't release, how it
ended, its lines' times and the ledger lines it signed) is the history's parameter (`RunSetup`).
-/

namespace Verity.Sampled.Device

open Lean (Json)
open SampledProofs SampledProofs.Challenger
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

/-- **The device's history at the play that ended with `o`**, of the setup `S o`: under its key, the run that released
the play's log, its release log the lines `printLog` prints of it after its head, and the ledger lines it signed. -/
def dev (S : Exchange → RunSetup) (o : Exchange) : History where
  runs pk :=
    if pk = (S o).key then
      match printLog (S o).time PrintState.init o.log with
      | some lines => [runOf (S o) o.log lines]
      | none => []
    else []
  ledger pk := if pk = (S o).key then (S o).ledger else []

/-- **At every play the device's history holds the play's run under its key, and no other run.** -/
theorem dev_play (S : Exchange → RunSetup) {fs : List (String × ℕ)} {D : Signer} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {o : Exchange} (h : Game.Reaches (play fs D θ F c) o) :
    ∃ lines, printLog (S o).time PrintState.init o.log = some lines ∧
      (dev S o).runs (S o).key = [runOf (S o) o.log lines] ∧ ∀ pk, pk ≠ (S o).key → (dev S o).runs pk = [] := by
  obtain ⟨lines, hl⟩ := printable_play h (S o).time
  refine ⟨lines, hl, ?_, fun pk hpk => ?_⟩
  · simp only [dev, hl, ite_true]
  · simp only [dev, hpk, ite_false]

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

end Verity.Sampled.Device
