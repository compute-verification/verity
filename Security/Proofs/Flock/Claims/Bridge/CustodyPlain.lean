import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Soundness.Discharge.DecodeZLJ.Statements

/-!
# `Deploy.V false`'s custody checks, as the old theorems' `RecordsLiveJ`

`Custody` without `--zk`. The old plain theorems bound flock-verify's acceptance of the transcript a writer `wr` makes
of the live session's outcome, under `RecordsLiveJ I dj Jf wr` (`DecodeZLJ`): the record `verify` accepts carries the
draw, the outcome's proof files, link points, roots and rounds. `Deploy.V false` takes the record in its last message
and checks its custody with the same checks as `--zk` (`checks`, `custody`).

* `setupOf_plainInputs`: `verify`'s setup of a record whose draw is the statement's is `Deploy.V`'s, where the statement
  carries no `--public-inputs`.
* `RecordsLiveAtP`, `recordsLiveAtP_of_checks`: `RecordsLiveJ` at one outcome, and where the checks pass, the last
  message's record and proof files record the run's live outcome (`viewOf`).
* `wrP`, `recordsLiveJ_wrP`, `accepts_wrP`: the writer that, at an outcome, takes a transcript `verify` accepts that
  records it, where one exists. It satisfies `RecordsLiveJ`, and on every run whose decision accepts at SHA-512,
  `verify` accepts what it writes for the run's outcome (`Accepts`), the event the old theorems bound.
-/

namespace Claims.Bridge.CustodyPlain

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody
open FlockSoundness.Refine (ebind_ok ptsOf)
open Lean (Json)

/-- **`verify`'s setup of a record whose draw is the statement's** is `Deploy.V`'s, where the statement is plain and
carries no `--public-inputs` (`Exec.setupOf` reads none). -/
theorem setupOf_plainInputs {r : Readable.Read} {spec : SessionSpec} {sts : Array Stmt} {record : String}
    (hs : r.setup = .ok (spec, sts)) (hzk : r.zk = false) (hpi : r.publicInputs = none)
    (hdraw : Exec.recordDraw record = r.draw) :
    Exec.setupOf r.I record = .ok (spec, sts) := by
  rw [← hs]
  simp only [Exec.setupOf, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, hzk, hpi, hdraw,
    Bool.false_eq_true, ↓reduceIte]

/-- **`verify` refuses a session without proof files**: it expects one per rep stream, two per table, and at least one
table. -/
theorem verify_nil_plain (spec : SessionSpec) (tabs : Array Setup) (record : String) :
    verify spec tabs record #[] ≠ .ok () := by
  intro h
  unfold verify verifyFiles at h
  simp only [] at h
  by_cases c1 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c1] at h
    obtain ⟨_, h', _⟩ := ebind_ok h
    cases h'
  rw [ite_eq_right c1] at h
  obtain ⟨sess, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨schs, -, h⟩ := ebind_ok h
  have c3 : ((#[] : Array ByteArray).size != spec.repStreams.size) = true := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, Decidable.not_not] at c1
    simp only [List.size_toArray, List.length_nil, repStreams_size, bne_iff_ne, ne_eq]
    omega
  rw [ite_eq_left c3] at h
  obtain ⟨_, h', _⟩ := ebind_ok h
  cases h'

/-- **`RecordsLiveJ` at one outcome**: where `verify` accepts the transcript `t`, its record carries the draw `d`, its
proof files are the outcome `o`'s, and the session it decodes to has `o`'s link points and, at each of `o`'s tables,
`o`'s `root_B` and rep streams' rounds. -/
def RecordsLiveAtP (I : Exec.Inputs) (d : Json) {J mPts : ℕ} (o : Tables.LiveOutJ J mPts) (t : Exec.Transcript) :
    Prop :=
  Exec.Accepts I t →
    Exec.recordDraw t.record = some d ∧ t.proofs = o.proofs ∧
    ∀ spec sts sess, Exec.setupOf I t.record = .ok (spec, sts) →
      Record.decode spec t.record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

theorem recordsLiveJ_iff {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → Tables.LiveOutJ (Jf S R) mPts → Exec.Transcript) :
    DecodeZLJ.RecordsLiveJ (mPts := mPts) I dj Jf wr ↔ ∀ S R o, RecordsLiveAtP I (dj S) o (wr S R o) := Iff.rfl

/-- **Custody gives `RecordsLiveAtP`**: where `Deploy.V false`'s checks pass on a plain statement that sets up at its
draw `d` with no `--public-inputs`, and the record's draw is the statement's, the last message's record and proof
files are a record of the run's live outcome (`viewOf`), at any `J` of the session's tables. -/
theorem recordsLiveAtP_of_checks {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {J mPts : ℕ} {d : Json} (hs : s.setup = .ok (spec, sts)) (hzk : s.read.zk = false)
    (hpi : s.read.publicInputs = none) (hdraw : Exec.recordDraw f.record = s.read.draw)
    (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) (hc : checks false s spec x link c₁ mids f = .ok ()) :
    RecordsLiveAtP s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  intro _
  refine ⟨hdraw.trans hd, rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [setupOf_plainInputs (Ins.setup_read hs) hzk hpi hdraw] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decode_live hdec hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  refine ⟨by simp only [viewOf]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList = ((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)
    rw [hD]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine forall₂_map_eq hyr fun rd rd' hmem ⟨hn, hcn, hb, hsha⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hsha hbytes
    subst hn hcn hb hbytes
    simp [withSha, hsha _ rfl]

/-- **An accepting decision of `Deploy.V false` at SHA-512 is an accepting `verify` on a record of its live outcome**:
on a statement with no `--public-inputs` of draw `d`, the last message's record and proof files are accepted by `verify`
(`Accepts`) at the statement's inputs, and record the run's live outcome `viewOf` (`RecordsLiveAtP`): the record's draw
is the statement's as a value (`recordDraw_of_decide`). -/
theorem accepts_of_decide_plain {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} {J mPts : ℕ} {d : Json}
    (ha : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next false x as cs).eval sha512).1 = .acc) (hpi : s.read.publicInputs = none)
    (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    Exec.Accepts s.read.I ⟨f.record, f.proofs⟩ ∧
      RecordsLiveAtP s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  have hdraw := recordDraw_of_decide ha hacc
  obtain ⟨hv, hc⟩ := (next_decide_sha512 ha).1 hacc
  simp only [Bool.false_eq_true, ite_false] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide ha)
  have hzk : s.read.zk = false := readOf_zk (x := x) (by simp [readOf, hins])
  exact ⟨⟨spec, sts, setupOf_plainInputs (Ins.setup_read hs) hzk hpi hdraw, hv⟩,
    recordsLiveAtP_of_checks hs hzk hpi hdraw hd hJ hc⟩

/-! ## The writer that reads the record out of the last message -/

open Classical in
/-- **The record writer of `Deploy.V false`'s runs**: at the live outcome `o` of draw `S`, a transcript `verify` accepts
that records `o` (`RecordsLiveAtP`), where one exists, else a session without proof files, which `verify` refuses
(`verify_nil_plain`). A decided run whose last message's record and proof files `verify` accepts is such a transcript
for its outcome (`accepts_wrP`). -/
noncomputable def wrP {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) (S : Finset (Fin n)) (R : Reg) (o : Tables.LiveOutJ (Jf S R) mPts) :
    Exec.Transcript :=
  if h : ∃ t, Exec.Accepts I t ∧ RecordsLiveAtP I (dj S) o t then h.choose else ⟨"", #[]⟩

theorem not_accepts_nil (I : Exec.Inputs) (record : String) : ¬ Exec.Accepts I ⟨record, #[]⟩ := by
  rintro ⟨spec, sts, -, h⟩
  exact verify_nil_plain _ _ _ h

/-- **`wrP` records the live session** (`RecordsLiveJ`, the record half of the old plain theorems' custody). -/
theorem recordsLiveJ_wrP {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (dj : Finset (Fin n) → Json)
    (Jf : Finset (Fin n) → Reg → ℕ) : DecodeZLJ.RecordsLiveJ (mPts := mPts) I dj Jf (wrP I dj Jf) := by
  refine (recordsLiveJ_iff I dj Jf _).2 fun S R o => ?_
  unfold wrP
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_accepts_nil _ _)

/-- **The old event holds on `Deploy.V false`'s accepting runs**: where its decision accepts at SHA-512, on a statement
with no `--public-inputs` of draw `dj S`, `verify` accepts what `wrP` writes for the run's live outcome at `Jf S R ≤`
the session's tables. -/
theorem accepts_wrP {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ)
    {S : Finset (Fin n)} {R : Reg} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (ha : act false x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next false x as cs).eval sha512).1 = .acc) (hpi : s.read.publicInputs = none)
    (hd : s.read.draw = some (dj S)) (hJ : Jf S R ≤ spec.tables.size) :
    Exec.Accepts s.read.I (wrP s.read.I dj Jf S R (viewOf (Jf S R) mPts spec link c₁ f)) := by
  have hex : ∃ t, Exec.Accepts s.read.I t ∧
      RecordsLiveAtP s.read.I (dj S) (viewOf (Jf S R) mPts spec link c₁ f) t :=
    ⟨_, accepts_of_decide_plain ha hacc hpi hd hJ⟩
  unfold wrP
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

end Claims.Bridge.CustodyPlain
