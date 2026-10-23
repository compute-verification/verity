import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Claims.Recursion.DeployHonest
import Proofs.Flock.C3.VStarList.Accepts

/-!
# What a staged part's transcript shows is the record the verifier read in its own format

`PartOpensB` at `cmtTr recRead` asks the commit strings a part holds to be those its V\* statement's transcript shows,
read by `recRead`: the coin server's record in its own format (`Flock.RecRecord.readFull`). Every staged part carries
the coin server's transcript (`insO_staged_link`), and its recursive setup holds it at the verifier's inner shape
(`RecSetup.commits`), which reads it in that same format and holds what it reads there to what its JSON readers read
(`RecSetup.checkReads`, `FlockC3.VStarList.checkReads_ok`). So at a staged part (`staged_reads`) the rounds
`cmtTr recRead` reads are the record's (`Record.msgs`), and the record's link, rounds, tops and `c_k` are the JSON's the
setup pins: Hello's table, `root_B`, the link points, the schedule's rounds, and each rep's salted tops and `c_k`.

This is the transcript's half of `PartOpensB`. The other half, that a part's hm96 slot outputs are the public rows its
statement pins and those rows are the JSON's (`RecVStar.checkPubRows`, `RecVStar.checkTopRows`), is to come.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open Lean (Json)

theorem byteArray_toList_loop (b : ByteArray) (i : ℕ) (r : List UInt8) :
    ByteArray.toList.loop b i r = r.reverse ++ b.data.toList.drop i := by
  fun_induction ByteArray.toList.loop b i r with
  | case1 i r h ih =>
    rw [ih, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 1
    conv_rhs => rw [List.drop_eq_getElem_cons (by simpa using h)]
    simp [ByteArray.get!, h]
  | case2 i r h =>
    rw [List.drop_eq_nil_of_le (by simp; simpa using h), List.append_nil]

theorem byteArray_toList_eq (b : ByteArray) : b.toList = b.data.toList := by
  unfold ByteArray.toList
  rw [byteArray_toList_loop]
  simp

theorem toUTF8_of_fromUTF8? {b : ByteArray} {s : String} (h : String.fromUTF8? b = some s) : s.toUTF8 = b := by
  unfold String.fromUTF8? at h
  split at h
  · cases h; rfl
  · cases h

/-- **The commits the reader reads** are `commitsOf` at the statement's source, binding and inner shape. -/
theorem RecRaw.ins_commits {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) :
    commitsOf w s.src s.bound s.inner = .ok s.commits := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem commits_parse_error {sh : RecVStar.Inner} {tt : String} {e : String} {r : Except String (Option String)}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {record : String} {cs : RecVStar.InnerCommits}
    (hp : Json.parse tt = .error e) : RecSetup.commits (some sh) (.ok tt) r bound record ≠ .ok cs := by
  intro h
  unfold RecSetup.commits at h
  simp only [bind, Except.bind, pure, Except.pure, hp] at h
  repeat' split at h
  all_goals cases h

theorem commits_text_error {sh : RecVStar.Inner} {e : String} {r : Except String (Option String)}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {record : String} {cs : RecVStar.InnerCommits} :
    RecSetup.commits (some sh) (.error e) r bound record ≠ .ok cs := by
  intro h
  unfold RecSetup.commits at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals cases h

/-- **A transcript's commits at the verifier's inner shape read it in the coin server's own format**: where
`commitsOf` holds the transcript `t`, it is UTF-8 and JSON, the source has a record, and `checkReads` passed. -/
theorem commitsOf_reads {w : RecRaw} {src : Option RecCoins.Source}
    {bound : Flock.V (Option String × Option RecVStar.StackedShape)} {sh : RecVStar.Inner}
    {c : Option RecVStar.InnerCommits} {t : ByteArray} (ht : w.transcript = some t)
    (h : commitsOf w src bound (some sh) = .ok c) :
    ∃ cs tt tr record, c = some cs ∧ String.fromUTF8? t = some tt ∧ Json.parse tt = .ok tr ∧
      src.map (·.record) = some record ∧ RecSetup.checkReads sh tt tr record cs = .ok () := by
  unfold commitsOf at h
  rw [ht] at h
  cases hsrc : src.map (·.record) with
  | none => rw [hsrc] at h; cases h
  | some record =>
    rw [hsrc] at h
    dsimp only at h
    split at h
    · cases h
    cases hX : RecSetup.commits (some sh) (textOf "rec inner transcript" t)
        (w.registration.mapM (textOf "rec inner registration")) bound record with
    | error e => rw [hX] at h; cases h
    | ok cs =>
      rw [hX] at h
      cases h
      cases htx : String.fromUTF8? t with
      | none =>
        rw [textOf, htx] at hX
        exact absurd hX commits_text_error
      | some tt =>
        rw [textOf, htx] at hX
        cases hp : Json.parse tt with
        | error e => exact absurd hX (commits_parse_error hp)
        | ok tr => exact ⟨cs, tt, tr, record, rfl, rfl, hp, rfl, FlockC3.VStarList.reads_of_commits hp hX⟩

theorem ins_of_insO {x : Msg} {i : RecIns} (h : insO x = some i) : ∃ w, decodeRec x = some w ∧ w.ins = .ok i := by
  unfold insO at h
  obtain ⟨w, hw, h⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at h
  · cases h
  exact ⟨w, hw, toOption_eq_some h⟩

/-- **A staged part's transcript, as the verifier read it in its own format**: at a part staged at `x`, the recursive
setup holds the coin server's transcript at the verifier's inner shape `sh` (`i.commits = some cs`), `checkReads`
passed on it, and the rounds `recRead` reads off the V\* statement are the record's (`Record.msgs`), the record
`readFull` reads. So `checkReads_ok` gives that record's link, rounds, tops and `c_k` as the JSON's. -/
theorem staged_reads {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) :
    ∃ (i : RecIns) (sh : RecVStar.Inner) (cs : RecVStar.InnerCommits) (tt : String) (tr : Json) (record : String)
      (r : Flock.RecRecord.Record),
      insO xO = some i ∧ i.inner = some sh ∧ i.commits = some cs ∧
      RecSetup.checkReads sh tt tr record cs = .ok () ∧ Flock.RecRecord.readFull tt.toUTF8.toList = some r ∧
      shownOf recRead x xO = some (r.msgs.map fun mc => (mc.1, mc.2.toArray)) := by
  obtain ⟨ms, i, _spec, _sts, rfl, hsu, -⟩ := hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨xr, -, hinner, -⟩ := insO_staged hins
  obtain ⟨wr, hd, htr, -, -⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hd] at hd'
  cases hd'
  have hc := RecRaw.ins_commits hw
  rw [hinner] at hc
  obtain ⟨cs, tt, tr, record, hcs, htx, -, -, hR⟩ := commitsOf_reads htr hc
  obtain ⟨_lk, r, _want, -, hr, -⟩ := FlockC3.VStarList.checkReads_ok hR
  refine ⟨i, xr.shape, cs, tt, tr, record, r, hins, hinner, hcs, hR, hr, ?_⟩
  have hb : (st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  simp only [shownOf, hd, htr, Option.bind_some, recRead, hb, hr, Option.map_some]
