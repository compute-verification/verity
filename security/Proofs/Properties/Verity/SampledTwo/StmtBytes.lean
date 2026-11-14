module

public import Proofs.Properties.Verity.SampledTwo.SessionMsgs
public import Proofs.Properties.Verity.SampledTwo.Steps

/-!
# A session's statement on an accepted case is under `2^64` bytes

The statements' message holds each template's session statement after its length in an 8-byte word
(`SessionMsgs.encStmts`), so it reads back only where each statement is under `2^64` bytes
(`SessionMsgs.stmtsOf_encStmts`). The two-stage form refuses a play one of whose second-stage sessions' statements
would be that long (`VStar.Recursive.SampledTwo.stmtsBytesFit`, its refusal `STMT_BYTES`): for each statement's
sessions on the blobs (`stmtSizeAt`) it sums the lengths the session's statement holds (`stmtSize`).

* `stmtRaw_le`: a member's session statement (`SessionMsgs.stmtRaw`) encodes in no more bytes than the form's sum.
* `pair_mem`: the `t`-th of the statements' sessions and its statement are a pair the form's step ran on.
* `stmtAt_lt`: where the step passed, template `t`'s member's statement at its session's record's sample is under
  `2^64` bytes.
* `stmtAtCase_lt`: on a case the form's check accepts, template `t`'s session statement (`SessionMsgs.stmtAtCase`)
  is under `2^64` bytes.
-/

@[expose] public section

namespace Verity.SampledTwo.StmtBytes

open Claims
open Lean (Json)
open FlockDeploy (encNat encBytes encStr encBool encOpt encList)

private theorem size_toUTF8 (s : String) : s.toUTF8.size = s.utf8ByteSize := by
  rw [String.toUTF8_eq_toByteArray, String.size_toByteArray]

private theorem len_encBytes (b : ByteArray) : (encBytes b).length = 8 + b.size := by
  show (encNat b.size ++ b.data.toList).length = _
  rw [List.length_append, FlockComplete.encNat_length, Array.length_toList]
  rfl

private theorem len_encStr (t : String) : (encStr t).length = 8 + t.utf8ByteSize := by
  rw [encStr, len_encBytes, size_toUTF8]

private theorem len_none {α : Type} (e : α → Msg) : (encOpt e none).length = 1 := rfl

private theorem len_some (b : ByteArray) : (encOpt encBytes (some b)).length = 9 + b.size := by
  show (1 :: encBytes b).length = _
  rw [List.length_cons, len_encBytes]
  omega

/-- An optional printed value's field is its tag, its length word and the value as printed. -/
private theorem len_printed (o : Option Json) :
    (encOpt encBytes (o.map fun j => (_root_.Flock.RecordText.printT j).toUTF8)).length ≤
      9 + (o.map Spot.OneStageZk.printedSize).getD 0 := by
  cases o with
  | none => exact Nat.le_add_right 1 8
  | some j =>
    show (encOpt encBytes (some (_root_.Flock.RecordText.printT j).toUTF8)).length ≤
      9 + Spot.OneStageZk.printedSize j
    rw [len_some, size_toUTF8]
    exact Nat.le_refl _

/-- The lookup tables' field is its length word and the form's `tablesSize`. -/
private theorem len_tables (tables : List (String × ByteArray)) :
    (encList (fun p : String × ByteArray => encStr p.1 ++ encBytes p.2) tables).length =
      8 + VStar.Recursive.SampledTwo.tablesSize tables := by
  have h : (fun p : String × ByteArray => (encStr p.1 ++ encBytes p.2).length) =
      fun t => 16 + t.1.utf8ByteSize + t.2.size := by
    funext t
    rw [List.length_append, len_encStr, len_encBytes]
    omega
  rw [encList, List.length_append, FlockComplete.encNat_length, List.length_flatMap, h,
    VStar.Recursive.SampledTwo.tablesSize]

/-- **A member's session statement encodes in no more bytes than the form's sum** (`stmtSize`), at the sample
`draw` as printed and the member's registered reads as printed. -/
theorem stmtRaw_le (m : Spot.OneStageZk.Member) (circuit pub : ByteArray) (tables : List (String × ByteArray))
    (draw : Option Json) :
    (SessionMsgs.stmtRaw m circuit pub tables (draw.map fun j => (_root_.Flock.RecordText.printT j).toUTF8)
      (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8)).encode.length ≤
      VStar.Recursive.SampledTwo.stmtSize m.own circuit pub (VStar.Recursive.SampledTwo.tablesSize tables) draw := by
  have hF : FlockDeploy.FORMAT.utf8ByteSize = 25 := by decide +kernel
  have hS : Spot.OneStageZk.STATEMENT.utf8ByteSize = 20 := by decide +kernel
  have hd := len_printed draw
  have hr := len_printed m.own.registered
  have hp : (_root_.Flock.RecordText.printT m.own.partition).toUTF8.size =
      Spot.OneStageZk.printedSize m.own.partition := size_toUTF8 _
  simp only [SessionMsgs.stmtRaw, FlockDeploy.Raw.encode, List.length_append, len_encStr, len_encBytes, len_tables,
    FlockComplete.encNat_length, encBool, List.length_cons, List.length_nil, hF, hS, len_none, len_some, hp]
  unfold VStar.Recursive.SampledTwo.stmtSize VStar.Recursive.SampledTwo.STMT_HEAD
  omega

/-- **The `t`-th of the statements' sessions and its statement are a pair the form's step ran on.** -/
theorem pair_mem {l : List Spot.OneStageZk.StatementFiles} {t : ℕ} {st : Spot.OneStageZk.StatementFiles}
    {s : Spot.OneStageZk.RunFiles}
    (hst : (l.flatMap fun st => st.sessions.toList.map fun _ => st)[t]? = some st)
    (hs : (l.flatMap (·.sessions.toList))[t]? = some s) :
    (st, s) ∈ l.flatMap fun st => st.sessions.toList.map fun s => (st, s) := by
  have h1 : (l.flatMap fun st => st.sessions.toList.map fun s => (st, s)).map Prod.fst =
      l.flatMap fun st => st.sessions.toList.map fun _ => st := by
    simp only [List.map_flatMap, List.map_map, Function.comp_def]
  have h2 : (l.flatMap fun st => st.sessions.toList.map fun s => (st, s)).map Prod.snd =
      l.flatMap (·.sessions.toList) := by
    simp only [List.map_flatMap, List.map_map, Function.comp_def, List.map_id']
  rw [← h1, List.getElem?_map] at hst
  rw [← h2, List.getElem?_map] at hs
  cases hq : (l.flatMap fun st => st.sessions.toList.map fun s => (st, s))[t]? with
  | none =>
    rw [hq] at hs
    cases hs
  | some q =>
    rw [hq] at hst hs
    obtain ⟨a, b⟩ := q
    cases hst
    cases hs
    exact List.mem_of_getElem? hq

/-- **Where the form's step passed, template `t`'s member's statement is under `2^64` bytes**, at its session's
record's sample, the member's circuit and public files and the blobs' lookup tables. -/
theorem stmtAt_lt {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence} {bs}
    (hfit : VStar.Recursive.SampledTwo.stmtsBytesFit p e bs = .ok ()) {t : ℕ} {m : Spot.OneStageZk.Member}
    {s : Spot.OneStageZk.RunFiles} {record : String} {circuit pub : ByteArray}
    (hm : SessionMsgs.memberAt p e t = some m) (hs : SessionMsgs.filesAt e t = some s)
    (hrec : SessionMsgs.recordAt bs s = some record)
    (hc : (bs.get m.own.circuit "circuit").toOption = some circuit)
    (hpb : (bs.get m.own.pub "public file").toOption = some pub) :
    (SessionMsgs.stmtRaw m circuit pub (Spot.OneStageZk.tablesOf bs) (SessionMsgs.drawBytes record)
      (m.own.registered.map fun j => (_root_.Flock.RecordText.printT j).toUTF8)).encode.length < 2 ^ 64 := by
  unfold SessionMsgs.memberAt at hm
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hm
  obtain ⟨st, hst, name, hname, hfind⟩ := hm
  have hmem := pair_mem (l := e.statements.toList) hst hs
  have hsz : VStar.Recursive.SampledTwo.stmtSizeAt p bs
      (VStar.Recursive.SampledTwo.tablesSize (Spot.OneStageZk.tablesOf bs)) st s =
      some (VStar.Recursive.SampledTwo.stmtSize m.own circuit pub
        (VStar.Recursive.SampledTwo.tablesSize (Spot.OneStageZk.tablesOf bs))
        (_root_.Flock.Witness.recordDraw record)) := by
    unfold SessionMsgs.recordAt at hrec
    unfold VStar.Recursive.SampledTwo.stmtSizeAt
    simp only [hname, hfind, hrec, hc, hpb, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  exact Nat.lt_of_le_of_lt (stmtRaw_le m circuit pub _ (_root_.Flock.Witness.recordDraw record))
    (VStar.Recursive.SampledTwo.stmtsBytesFit_at hfit hmem hsz)

/-- **On a case the form's check accepts, template `t`'s session statement is under `2^64` bytes**: the check ran
its step on the statement's member and session (`VStar.Recursive.SampledTwo.accepted_stmtsBytesFit`), and the
statement is that member's at that session's record's sample (`SessionMsgs.stmtAtCase_some`). -/
theorem stmtAtCase_lt {k : Case} {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    {c ξ pub prof} (hf : SessionMsgs.formOf k = some (p, e))
    (hck : VStar.Recursive.SampledTwo.checkBlobsAtWith true p c ξ e k.blobs = .accept pub prof) {t : ℕ} {x : Msg}
    (hx : SessionMsgs.stmtAtCase k t = some x) : x.length < 2 ^ 64 := by
  obtain ⟨p', e', m, s, record, circuit, pubB, hf', hm, hs, hrec, hc, hpb, rfl⟩ := SessionMsgs.stmtAtCase_some hx
  rw [hf] at hf'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf')
  exact stmtAt_lt (VStar.Recursive.SampledTwo.accepted_stmtsBytesFit hck) hm hs hrec hc hpb

end Verity.SampledTwo.StmtBytes
