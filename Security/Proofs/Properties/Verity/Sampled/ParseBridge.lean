module

public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.Sampled.Trees

/-!
# The record the log readers read is the record v2 verified

v2 reads a session's record twice: `Flock.Zk.verify` decodes it (`Flock.Record.decode`), and
`Spot.OneStageZk.treesHeld` holds it to the challenger's trees. Both parse it through `Flock.JsonParse.parse`,
which is total, so a proof reads one parsed record out of both. The log reader reads the sessions' records as
`treesHeld` does (`Verity.Sampled.recordsOn`, which `coinsRead` reads): where v2 verified a session's record, it reads
exactly that record's parse (`sent_records`), the value the decoders' lemmas read off the record
(`Verity.Sampled.Decoders`).
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)

theorem mapM_single_some {α β : Type} {f : α → Option β} {a : α} {b : β} (h : f a = some b) :
    [a].mapM f = some [b] := by
  rw [List.mapM_cons, List.mapM_nil, h]
  rfl

/-- The session's record v2 reads (`sessionRecord`), of evidence naming one session, is that session's blob as text. -/
theorem sessionRecord_blob {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {rs : _root_.Spot.OneStageZk.RunFiles} {record : String} (he : e.sessions = #[rs])
    (hr : sessionRecord e bs = some record) :
    ∃ b, bs.get rs.record s!"session {rs.name}'s record" = .ok b ∧ String.fromUTF8? b = some record := by
  unfold sessionRecord at hr
  rw [he] at hr
  change (bs.get rs.record s!"session {rs.name}'s record").toOption.bind String.fromUTF8? = some record at hr
  cases hb : bs.get rs.record s!"session {rs.name}'s record" with
  | error _ => rw [hb] at hr; cases hr
  | ok b => rw [hb] at hr; exact ⟨b, rfl, hr⟩

/-- **`recordsOn` reads the session's record's parse**: of evidence naming one session whose record parses to `J`
through `Flock.JsonParse.parse`, the records `recordsOn` reads are `[J]`. -/
theorem recordsOn_single {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {rs : _root_.Spot.OneStageZk.RunFiles} {record : String} {J : Json} (he : e.sessions = #[rs])
    (hr : sessionRecord e bs = some record) (hJ : Flock.JsonParse.parse record = .ok J) :
    recordsOn bs e = some [J] := by
  obtain ⟨b, hb, hu⟩ := sessionRecord_blob he hr
  unfold recordsOn
  rw [he]
  refine mapM_single_some ?_
  unfold recordOn
  rw [hb]
  change (String.fromUTF8? b >>= fun t => pure ((Flock.JsonParse.parse t).toOption.getD Json.null)) = some J
  rw [hu]
  change some ((Flock.JsonParse.parse record).toOption.getD Json.null) = some J
  rw [hJ]
  rfl

/-- **A record v2 verified parses** through `Flock.JsonParse.parse`: `Flock.Zk.verify` accepts only a record that
`Flock.Record.decode` decodes, and that parses it first. -/
theorem verified_parse {p : _root_.Spot.OneStageZk.Params} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} (h : V2Verified p e bs record spec tabs rest proofs) :
    ∃ J, Flock.JsonParse.parse record = .ok J := by
  obtain ⟨_, _, _, _, _, -, -, -, -, -, -, -, -, hv⟩ := h
  obtain ⟨_, hd⟩ := zkVerify_decode hv
  obtain ⟨J, _, _, hJ, -⟩ := Claims.Bridge.Custody.decode_ok hd
  exact ⟨J, hJ⟩

/-- **The log reader reads the record v2 verified**: where v2 verified a session's record on the blobs sent
(`V2Verified`, which `sent_verified` gives where v2's case accepts), the record parses through `Flock.JsonParse.parse`,
the parser `Flock.Record.decode` reads it through, to some `J`, and the sessions' records read exactly `[J]`:
`recordsOn`, as `treesHeld` and `coinsRead` read them. -/
theorem sent_records {s : Verity.Sent} {p : _root_.Spot.OneStageZk.Params}
    {e : _root_.Spot.OneStageZk.Evidence} {record : String} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {rest : Json × Json} {proofs : Array ByteArray}
    (h : V2Verified p e (_root_.Core.Blobs.ofArray s.blobs) record spec tabs rest proofs) :
    ∃ J, Flock.JsonParse.parse record = .ok J ∧ recordsOn (_root_.Core.Blobs.ofArray s.blobs) e = some [J] := by
  obtain ⟨J, hJ⟩ := verified_parse h
  obtain ⟨_, _, _, _, _, he, -, -, -, -, hr, -⟩ := h
  exact ⟨J, hJ, recordsOn_single he hr hJ⟩

end Verity.Sampled
