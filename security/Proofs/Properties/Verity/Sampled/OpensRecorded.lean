module

public import Proofs.Properties.Verity.Sampled.Counts
public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Verity.Sampled.Lines
public import Proofs.Properties.Verity.Sampled.MovesReach
public import Proofs.Properties.Verity.Sampled.RecordRounds
public import Proofs.Properties.Verity.Sampled.RoundsRead

/-!
# The device's log opens only the record's rounds, from the record check

v2's record check (`Spot.Challenger.sessionTreeReason`) refuses a session whose coin tree shows a round opened that
its record doesn't give its stream (`EXTRA_OPENING`). Read through v2's read of the release log, that is
`OpenedRecorded` of the device's log, at every play the sampled verdict accepts:

* `coinTreesOf_coins`, `coinTreesOf_opened`: **each line opening a round is a round its tree shows opened**: where
  `coinTreesOf` accepts the release log, a line of it opening round `k` of stream `name` of the tree whose root is `ρ`
  (`OpensRound`) is an opening of that stream and round among those the tree of root `ρ` shows.
* `openedRecorded_of_reason`: **the record check holds each opened round to the record** (`openedRecorded`), at the
  tree whose root the record names (`Trees.session_counted_opened`).
* `openedRecorded_read`: **v2's read opens only the record's rounds**: where the release log shows one tree, each
  opening `readLog` reads is a line opening the round its stream's openings before it count (`readLog_round`), a round
  of the record's tree, which the record gives its stream.
* `Instance.logOpensRecorded`: **`LogOpensRecorded` at every instance and composite**: at a play the sampled verdict
  accepts whose log v2's check read is the device's, v2 read one tree's release log and held the record it parsed to it
  (`verdict_read`), and the device's log has the read's frames (`openedRecorded_congr`).
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Spot Spot.Challenger

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- Each line of `pre` opening a coin tree's round gives its coins as a string. -/
def CoinsStr (pre : List String) (_ : Array LoggedTree) : Prop :=
  ∀ l ∈ pre, treeLine l = true → ∀ d, event l = .ok d → jsonStrIs (getD d "event") TREE_EVENT = false →
    ∃ C, getD d "coins" = .str C

theorem CoinsStr.snoc {pre : List String} {ts ts' : Array LoggedTree} {line : String} (hI : CoinsStr pre ts)
    (h : treeLine line = true → ∀ d, event line = .ok d → jsonStrIs (getD d "event") TREE_EVENT = false →
      ∃ C, getD d "coins" = .str C) : CoinsStr (pre ++ [line]) ts' := by
  intro l hl
  rcases List.mem_append.1 hl with hl | hl
  · exact hI l hl
  · rw [List.mem_singleton.1 hl]
    exact h

/-- **Each line opening a round gives its coins as a string**: `coinTreesOf` accepts such a line only where its
`coins` are 128 hex digits. -/
theorem coinTreesOf_coins {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    CoinsStr log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Trees.forIn_inv_run (I := CoinsStr) ?_ (by intro l hl; cases hl) h
  intro pre line ts r hI hr
  split at hr
  case isFalse htl => cases hr; exact ⟨ts, rfl, hI.snoc fun h => absurd h htl⟩
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, Trees.ok_bind] at hr
    split at hr
    · rename_i hT
      split at hr
      case isFalse => cases hr
      split at hr
      case isTrue => cases hr
      cases hr
      exact ⟨_, rfl, hI.snoc fun _ d' hd' hT' => by
        have hdd : d' = d := Except.ok.inj (hd'.symm.trans hev)
        rw [hdd, hT] at hT'
        cases hT'⟩
    · rename_i hT
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
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          cases hr
          exact ⟨_, rfl, hI.snoc fun _ d' hd' _ => by
            have hdd : d' = d := Except.ok.inj (hd'.symm.trans hev)
            rw [hdd]
            have hhex := hcond.2
            cases hc : getD d "coins" <;> rw [hc] at hhex <;> first | exact ⟨_, rfl⟩ | cases hhex⟩
        · cases hr

/-- **Each line opening a round is a round its tree shows opened**: where `coinTreesOf` accepts the release log
`lines`, a line of it opening round `k` of stream `name` of the tree whose root is `ρ` (`OpensRound`) is an opening of
stream `name` and round `k` that the tree of root `ρ` shows (`Lines.TreesOf.opened`). -/
theorem coinTreesOf_opened {lines : List String} {trees : Array LoggedTree} (h : coinTreesOf lines = .ok trees)
    {l ρ name : String} {k : ℕ} (hl : l ∈ lines) (ho : OpensRound l ρ name k) :
    ∃ t ∈ trees, t.root = ρ ∧ ∃ o ∈ t.opened, o.stream = name ∧ o.round = k := by
  obtain ⟨htl, d, hev, hT, hroot, hs, hk⟩ := ho
  obtain ⟨C, hC⟩ := coinTreesOf_coins h l hl htl d hev hT
  have hO : jsonStrIs (getD d "event") OPEN_EVENT = true := by
    have e := Lines.treeLine_ok hev
    rw [htl, hT, Bool.false_or] at e
    exact e.symm
  obtain ⟨t, ht, hr, o, hop, -⟩ := (Lines.trees_log h).opened l hl d hev hT hO ρ name k C hroot hs hk hC
  unfold LoggedTree.opening at hop
  have hp := Array.find?_some hop
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  exact ⟨t, ht, hr, o, Array.mem_of_find?_eq_some hop, hp.1, hp.2⟩

/-- **The record check holds each opened round to the record**: a record `sessionTreeReason` accepts, at any tables,
names the root of a tree of the log each of whose opened rounds is below the rounds the record gives its stream
(`openedRecorded`). -/
theorem openedRecorded_of_reason {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, recordTree record = t.root ∧ openedRecorded t record = true := by
  obtain ⟨t, ht, hroot, -, -, hop⟩ := Trees.session_counted_opened h
  exact ⟨t, ht, hroot, hop⟩

/-- Two members of an array of one member are one. -/
theorem mem_one {β : Type} {xs : Array β} (h1 : xs.size = 1) {a b : β} (ha : a ∈ xs) (hb : b ∈ xs) : a = b := by
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 ha
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hb
  have hij : i = j := by omega
  subst hij
  rfl

/-- **v2's read opens only the record's rounds**: where `coinTreesOf` accepts the release log `lines` as one tree,
`readLog` reads it as `lr` and `sessionTreeReason` accepts `record` against that tree, each opening of `lr` is a line
opening the round its stream's openings before it count (`readLog_round`), a round the tree shows opened
(`coinTreesOf_opened`), which the record gives its stream (`openedRecorded_of_reason`). -/
theorem openedRecorded_read {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {lr : List Verity.Release} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    (h1 : trees.size = 1) (hR : readLog label after records ξ lines = some lr)
    {tables : Array (String × Option ℕ)} {record : Json} (h : sessionTreeReason trees tables record = none) :
    OpenedRecorded record lr := by
  intro pre r post he i s n hr name hname
  obtain ⟨l, ρ, name', k, hl, hro, -, hname', hk⟩ := readLog_round hT hR he hr
  have hnn : name' = name := Option.some.inj (hname'.symm.trans hname)
  obtain ⟨t, ht, -, hop⟩ := openedRecorded_of_reason h
  obtain ⟨t', ht', -, o, ho, hs, hrd⟩ := coinTreesOf_opened hT hl hro.1
  have htt : t' = t := mem_one h1 ht' ht
  rw [htt] at ho
  have hlt : o.round < recordRounds record o.stream := by
    simpa [openedRecorded] using Array.all_eq_true'.1 hop o ho
  rw [← hk, ← hrd, ← hnn, ← hs]
  exact hlt

namespace Instance

/-- **The device's log opens only the record's rounds**, at every instance and composite: at a play on files v2
sets one of the deployment's sessions up on, which the sampled verdict accepts and whose log v2's check read is the
device's, v2 read one tree's release log and its record check accepted the record it parsed (`verdict_read`), so its
read opens only the record's rounds (`openedRecorded_read`), and the device's log has the read's frames. -/
theorem logOpensRecorded (I : Instance d α) (k : Verity.Registration.Composite) : I.LogOpensRecorded k := by
  intro θ F A hA c σ x hv hl text Jr hrec hJ
  obtain ⟨_, _, _, _, lr, _, Jr', _, hV, hread, hct, h1, hJt, hJ'⟩ :=
    verdict_read (I.sessions_some hA).2.1 (play_ends I.base θ F c σ x).1 hv hl hrec
  have hJJ : Jr' = Jr := Except.ok.inj (hJ'.symm.trans hJ)
  rw [hJJ] at hJt
  exact openedRecorded_congr ((List.forall₂_map_left_iff.1 hV).imp fun _ _ h => ⟨h.1, h.2.1⟩)
    (openedRecorded_read hct h1 hread hJt)

end Instance

end Verity.Sampled
