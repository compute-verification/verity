module

public import Proofs.Properties.Verity.Sampled.Binding
public import Proofs.Properties.Verity.SampledRead

/-!
# Every round a session's coin tree opens is a round of its record

v2 holds each round a `--zk` session's record names to an opening of the session's coin tree
(`SampledProofs.Challenger.sessionTreeReason`), but not each opening to a round of the record: `coinTreesOf` accepts a
stream's next round while it is below `2^rounds_log`. The refusal that closes it, as a Bool over v2's own inputs and as
a Prop over the device's log:

* **The rounds a record gives a stream** (`recordRounds`): one of `SampledProofs.Challenger.TREE_SESSION`, its round 0
  the link points; of another stream, the `rounds` of the record's first stream so named, as `recordRound` finds it;
  none of a stream the record doesn't name.
* **v2's test** (`openedRecorded`): every round the session's tree shows opened (`coinTreesOf`'s `opened`) is below the
  rounds the record gives its stream. `coinTreesOf` numbers a stream's openings 0, 1, … in order, and
  `sessionTreeReason` holds each record round to an opening, so where both pass the tree opens each stream the record
  names exactly its `rounds.size` times, `TREE_SESSION` once (round 0, the link points), and a stream the record doesn't
  name never. Honest sessions pass it: the challenger opens a round only as the session records it.
* **On the device's log** (`OpenedRecorded`): before each opening of a coin tree's stream, the log opens that stream
  fewer times than the rounds the record gives it. The device opens a stream's rounds in order (`Verity.Frame.opening`),
  so that count is the round opened. It reads only frames, so it holds of the device's log wherever it holds of v2's
  read and the read is the device's log (`openedRecorded_congr`, `openedRecorded_logged`: `Binding.Logged
  Binding.readsV2`, `MovesReach`'s premise). On a play v2 accepts the log shows one coin tree, the record's: v2 takes
  one session (`SampledProofs.OneStageZk.oneSession`) and every tree a session's (`sessionsTreeReason`).
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)

/-- **The rounds a session's record gives stream `stream` of its coin tree**: one of
`SampledProofs.Challenger.TREE_SESSION`, its round 0 the link points; of another stream, the size of `rounds` of the
record's first stream so named (`recordRound`'s); none of a stream the record doesn't name. -/
def recordRounds (record : Json) (stream : String) : ℕ :=
  if stream == _root_.SampledProofs.Challenger.TREE_SESSION then 1
  else
    match _root_.SampledProofs.getD record "streams" with
    | .arr ss =>
      match ss.find? (fun x => match _root_.SampledProofs.getD x "stream" with | .str n => n == stream | _ => false) with
      | some x =>
        match _root_.SampledProofs.getD x "rounds" with
        | .arr rs => rs.size
        | _ => 0
      | none => 0
    | _ => 0

/-- **Every round the session's coin tree `t` shows opened is a round of its record `record`**: each round
`SampledProofs.Challenger.coinTreesOf` shows opened is below the rounds `record` gives its stream (`recordRounds`).
`sessionTreeReason` refuses a record whose tree fails it. -/
def openedRecorded (t : _root_.SampledProofs.Challenger.LoggedTree) (record : Json) : Bool :=
  t.opened.all fun o => o.round < recordRounds record o.stream

/-- The openings of stream `s` of the `i`-th coin tree among the releases `l`. -/
def streamOpenings (i s : ℕ) (l : List Verity.Release) : ℕ :=
  l.countP fun q => match q.frame with
    | .opening i' s' _ => i' == i && s' == s
    | _ => false

/-- **Every round the device's log `log` opens is a round of the record `record`**: before each opening of stream `s`
of the `i`-th coin tree, named `name` in that tree's spec (`helloSpecs`), the log opens that stream fewer times than the
rounds `record` gives `name` (`recordRounds`). The device opens a stream's rounds in order, each once, so the count is
the round opened. -/
def OpenedRecorded (record : Json) (log : List Verity.Release) : Prop :=
  ∀ pre r post, log = pre ++ r :: post → ∀ i s n, r.frame = .opening i s n →
    ∀ name, ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name →
      streamOpenings i s pre < recordRounds record name

theorem streamOpenings_congr {i s : ℕ} :
    ∀ {l l' : List Verity.Release}, List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l' →
      streamOpenings i s l = streamOpenings i s l'
  | _, _, .nil => rfl
  | _ :: _, _ :: _, .cons h t => by
    have e := streamOpenings_congr (i := i) (s := s) t
    simp only [streamOpenings, List.countP_cons, h.1] at e ⊢
    rw [e]

/-- **Releases of the same frames, answering the same bytes, open only the record's rounds together**:
`OpenedRecorded` reads only each release's frame. -/
theorem openedRecorded_congr {record : Json} {l l' : List Verity.Release}
    (hl : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l')
    (hP : OpenedRecorded record l) : OpenedRecorded record l' := by
  intro pre' r' post' heq i s n hr' name hname
  subst heq
  obtain ⟨pre, r, post, rfl, hpre, hr, -⟩ := forall₂_split hl
  rw [← helloSpecs_congr hpre] at hname
  rw [← streamOpenings_congr hpre]
  exact hP pre r post rfl i s n (hr.1.trans hr') name hname

/-- **Through the binding**: where the log v2's check read is the device's (`Binding.Logged Binding.readsV2`), a read
log `l` (`coinsRead`) that opens only rounds of `record` is one where the device's log does. -/
theorem openedRecorded_logged {record : Json} {o : Verity.Exchange} (hL : Binding.Logged Binding.readsV2 o)
    {l : List Verity.Release} (hl : coinsRead o.sent = some l) (hP : OpenedRecorded record l) :
    OpenedRecorded record o.log := by
  have h := hL (l.map Binding.Read.ofV2) (by simp [Binding.readsV2, hl])
  refine openedRecorded_congr ?_ hP
  exact (List.forall₂_map_left_iff.1 h).imp fun _ _ hab => ⟨hab.1, hab.2.1⟩

end Verity.Sampled
