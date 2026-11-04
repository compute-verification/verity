import Proofs.Flock.Soundness.Discharge.Hidden.Commit
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.Composed.TagsJ
import Proofs.Flock.Soundness.Discharge.ZkExec.Event

/-!
# The hidden-output `--zk` chain's statement circuit

The hidden-output `--zk` chain states its circuit hypotheses at the statement circuit, `HmOut.parseStmt … = .ok c`: a
templated file's is its template unrolled (`HmOut.unrolled`), any other file's `HmOut.parse`'s. What the chain reads of a
session it reads of the session's committed circuit (`HmOut.parseCommit`), and the two meet at `c = HmOut.unrolled st.c`
(`stmt_of_setupHidden`), which keeps `g` (`unrolled_g`):

* `parseStmt_of_flat`: a flat corollary's `HmOut.parse … = .ok c` is `HmOut.parseStmt … = .ok c` (`HmOut.parse` refuses
  templated tags, `HmOut.parse_isTmpl`).
* `setupHidden_commit`, `stmt_of_setupHidden`, `g_pos_of_setupHidden`: a session `Stmt.setupHidden` accepts commits
  `HmOut.parseCommit`'s circuit, the statement circuit is it unrolled, and its `g` is positive.
* `drawSetupH_of_setupHiddenU`, `drawSetupH_of_acceptsU`, `tableAtH_of_acceptsU`: `drawSetupH_of_setupHiddenI`,
  `drawSetupH_of_accepts` and `tableAtH_of_acceptsI` from the session alone.
* `parseStmtH_identity`, `parseStmtH_tagsAt`, `zkTags_isTmpl`: an untyped `HmOut.parseStmt` reads no identity, so it
  agrees at every table's tags.
-/

namespace FlockSoundness

open Flock

variable {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}

/-- **A flat file's statement circuit**: `HmOut.parse` refuses templated tags, so a circuit it reads is the statement
circuit. -/
theorem parseStmt_of_flat (h : HmOut.parse tags cf tables = .ok c) : HmOut.parseStmt tags cf tables = .ok c :=
  parseStmt_of_parse (HmOut.parse_isTmpl h) h

/-- `HmOut.unrolled` keeps the circuit's `g`. -/
theorem unrolled_g (c : Circuit) : (HmOut.unrolled c).g = c.g := by
  unfold HmOut.unrolled
  split
  · split <;> rfl
  · rfl

/-- `HmRow.parse` reads no identity, whatever nets it is given first. -/
theorem parsePre_identity (tags : Tags) (x : Lean.Json) (b : ByteArray) (t : List (String × ByteArray))
    (p : Array (String × Net)) :
    HmRow.parse { tags with identity := x } b t (pre := p) = HmRow.parse tags b t (pre := p) := by
  kernel_rfl

/-- `HmOut.parseTmpl` reads no identity. -/
theorem parseTmpl_identity (tags : Tags) (x : Lean.Json) (b : ByteArray) (t : List (String × ByteArray)) :
    HmOut.parseTmpl { tags with identity := x } b t = HmOut.parseTmpl tags b t := by
  unfold HmOut.parseTmpl
  simp only [parsePre_identity]

end FlockSoundness

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs Accepts recordDraw)
open FlockSoundness.Discharge.Integrate (model_sched fits_blocks)
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt} {c : Circuit}

/-- **A session commits `HmOut.parseCommit`'s circuit**: the circuit `Stmt.setupHidden` reads. -/
theorem setupHidden_commit (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    HmOut.parseCommit tags cf tables = .ok st.c :=
  Hidden.setupHidden_parseCommit h

/-- **The statement circuit is the session's committed circuit unrolled**, for untyped tags. -/
theorem stmt_of_setupHidden (ht : tags.typed = false) (hc : HmOut.parseStmt tags cf tables = .ok c)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : c = HmOut.unrolled st.c := by
  obtain ⟨cs, hcs, rfl⟩ := commit_of_parseStmt ht hc
  rw [Except.ok.inj ((setupHidden_commit h).symm.trans hcs)]

/-- The statement circuit's `g` is the session's. -/
theorem stmt_g_of_setupHidden (ht : tags.typed = false) (hc : HmOut.parseStmt tags cf tables = .ok c)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : c.g = st.c.g := by
  rw [stmt_of_setupHidden ht hc h, unrolled_g]

/-- **A session's `g` is positive**: its committed circuit is `HmRow.parse`'s then `HmOut.checkRows`
(`Hidden.parseCommit_untyped`), and `HmRow.check` holds a power of two of VUs per block (`CheckFacts.g_pow2`). -/
theorem g_pos_of_setupHidden (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : 0 < st.c.g := by
  obtain ⟨pre, c₀, -, P, -, he, -⟩ := Hidden.parseCommit_untyped ht (setupHidden_commit h)
  obtain ⟨ls, hls⟩ := P.check
  have hp := (check_facts P.typed hls).g_pow2
  simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
  rw [he]
  exact hp.1

variable {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json}

/-- **A `DrawSetupH` from setupHidden's acceptance at a draw**, from the session alone (`drawSetupH_of_setupHiddenI`
without its circuit hypothesis): a `fast100` schedule of its `m`, the draw proving `S`, and its link points `mPts`. -/
theorem drawSetupH_of_setupHiddenU (ht : I.tags.typed = false) {st : Flock.Stmt}
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : DrawSetupH I mPts S d, x.st = st := by
  have hg : 0 < st.c.g := g_pos_of_setupHidden ht h
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := setupHidden_n h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts, setupHidden_wf h⟩, rfl⟩

/-- `drawSetupH_of_accepts` from the session alone. -/
theorem drawSetupH_of_acceptsU (ht : I.tags.typed = false) (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1)
    {t : Exec.Transcript} (h : Accepts I t) (hd : recordDraw t.record = some d) :
    ∃ st D, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program =
        .ok st ∧ Draw.UnitDraw.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : DrawSetupH I mPts S d, x.st = st) := by
  obtain ⟨st, hst, hv⟩ := accepts_oneAny hc1 h
  rw [setupOne_hidden hh, hd] at hst
  obtain ⟨-, sch, -, -, hf, -, -⟩ := verify_one_ok hv
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  exact ⟨st, D, hst, hD, fun hS hm => drawSetupH_of_setupHiddenU ht hst hf hD hS hm⟩

/-- `tableAtH_of_acceptsI` from the session alone. -/
theorem tableAtH_of_acceptsU (ht : I.tags.typed = false) (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {t : Exec.Transcript}
    (h : Accepts I t) (hd : recordDraw t.record = some (dj S))
    (hS : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition
      I.program = .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
        .ok st ∧ ∃ hwf : StmtWF st, (tableAtH dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  obtain ⟨st, D, hst, hD, hx⟩ := drawSetupH_of_acceptsU (S := S) (mPts := mPts) ht hh hc1 h hd
  obtain ⟨x, rfl⟩ := hx (hS D hD) (hmPts _ hst)
  obtain ⟨y, hy, e⟩ := tableAtH_stmt dj x₀ x
  exact ⟨x.st, hst, hy ▸ ⟨y.wf.1, congrArg (fun T : TableH I mPts S => T.T.S) e⟩⟩

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.ZkHidden

open Flock Exec ZkExec

/-- The `--zk` tags are templated exactly when `--statement`'s are. -/
theorem zkTags_isTmpl (I : Inputs) : (zkTags I).isTmpl = I.tags.isTmpl := by
  rw [zkTags_eq]
  rfl

/-- An untyped `HmOut.parseStmt` reads no identity. -/
theorem parseStmtH_identity {tags : Tags} (ht : tags.typed = false) (x : Lean.Json) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parseStmt { tags with identity := x } b t = HmOut.parseStmt tags b t := by
  have hi : ({ tags with identity := x } : Tags).isTmpl = tags.isTmpl := rfl
  have hp : HmOut.parse { tags with identity := x } b t = HmOut.parse tags b t := by
    simp only [HmOut.parse, Composed.parse_identity, Tags.isTmpl]
    simp only [ht, Bool.not_false, ite_true]
  unfold HmOut.parseStmt
  rw [hi, parseTmpl_identity tags x b t, hp]

/-- **An untyped `HmOut.parseStmt` reads no identity, so it agrees at every table's tags.** -/
theorem parseStmtH_tagsAt {tags : Tags} (ht : tags.typed = false) (count j : ℕ) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parseStmt (tagsAt tags count j) b t = HmOut.parseStmt tags b t := by
  unfold tagsAt
  split_ifs
  · rfl
  · exact parseStmtH_identity ht _ _ _

end FlockSoundness.Discharge.ZkHidden
