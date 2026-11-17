module

public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Properties.Verity.Sampled.Trees

/-! The record the log readers read is the record v2 verified. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)

theorem mapM_single_some {α β : Type} {f : α → Option β} {a : α} {b : β} (h : f a = some b) :
    [a].mapM f = some [b] := by
  rw [List.mapM_cons, List.mapM_nil, h]
  rfl

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

theorem verified_parse {p : _root_.Spot.OneStageZk.Params} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} (h : V2Verified p e bs record spec tabs rest proofs) :
    ∃ J, Flock.JsonParse.parse record = .ok J := by
  obtain ⟨_, _, _, _, _, -, -, -, -, -, -, -, -, hv⟩ := h
  obtain ⟨_, hd⟩ := zkVerify_decode hv
  obtain ⟨J, _, _, hJ, -⟩ := Claims.Bridge.Custody.decode_ok hd
  exact ⟨J, hJ⟩

theorem sent_records {s : Verity.Sent} {p : _root_.Spot.OneStageZk.Params}
    {e : _root_.Spot.OneStageZk.Evidence} {record : String} {spec : Flock.SessionSpec}
    {tabs : Array Flock.Setup} {rest : Json × Json} {proofs : Array ByteArray}
    (h : V2Verified p e (_root_.Core.Blobs.ofArray s.blobs) record spec tabs rest proofs) :
    ∃ J, Flock.JsonParse.parse record = .ok J ∧ recordsOn (_root_.Core.Blobs.ofArray s.blobs) e = some [J] := by
  obtain ⟨J, hJ⟩ := verified_parse h
  obtain ⟨_, _, _, _, _, he, -, -, -, -, hr, -⟩ := h
  exact ⟨J, hJ, recordsOn_single he hr hJ⟩

end Verity.Sampled
