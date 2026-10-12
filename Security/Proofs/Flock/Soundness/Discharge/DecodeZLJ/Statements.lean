import Proofs.Flock.Soundness.Discharge.DecodeZL
import Proofs.Flock.Soundness.Discharge.Tables

/-!
# The decoder at `J` tables: statements

`headline_tables_subset`'s open guarantee is `hdec : DecodesOnJ …`: at every draw, registration and table `j` of the
plan, on every live outcome whose session directory flock-verify accepts, table `j`'s reps run on the tapes decoded from
its streams `(j, 0)`, `(j, 1)`, and the salted table's verdict accepts. This file states it at the tables flock-verify
sets up at `count > 1` (`accepts_tables`), with dec-zl's one-table decoder run on each table's two streams:

* `partJson d count j`: part `j` of the draw `d`'s units, the subset draw `partDraw` that `Stmt.setupTables` sets table
  `j` up at.
* `SetupJ I mPts j d`: under table `j`'s tags (`tagsAt`), setupH accepts the draw `d`, at a `fast100` schedule of at
  least two levels and `mPts` link points. `SetupJ.spec` is its table, built as #828's `DrawSetup.specAt` builds one.
* `tableJ`, `planJ`: table `j` at the draw's JSON is `SetupJ.spec` at part `j` where flock-verify sets that part up, and
  the fallback `T₀` elsewhere; the plan has `I.count` tables.
* `decJ`, `opnJ`: table `j`'s decoder (`wZL ++ (wOP ++ wLIG)` at its dimensions, claims and schedule) and openings
  (`opnOf` at setupH's statement for part `j`, on proof files `2j` and `2j + 1`).
* `TagsOkJ`, `RecordsLiveJ`, `DrawOkJ`: `TagsOk`, `RecordsLive` and `DrawOk` at `J` tables.
* `DecodesOnJExec`: `DecodesOnJ` at `planJ`, for every fallback `T₀`, with the decoder `decJ`, single coins as field
  elements, no final messages, table `j`'s cap from `root_B[j]` and its openings `opnJ`.
-/

namespace FlockSoundness.Discharge.DecodeZLJ

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.Tables FlockSoundness.Discharge.DecodeZL
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-- **Part `j` of the draw `d`**: `partDraw` at `d`'s population and units, the subset draw `Stmt.setupTables` sets
table `j` of `count` up at, when `d` decodes. -/
def partJson (d : Json) (count j : ℕ) : Json :=
  match Draw.UnitDraw.ofJson d with
  | .ok u => partDraw u.population u.units count j
  | .error _ => d

theorem tagsAt_typed (tags : Tags) (count j : ℕ) : (tagsAt tags count j).typed = tags.typed := by
  unfold tagsAt
  split <;> rfl

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

/-- **A table flock-verify sets up**: under table `j`'s tags, `Stmt.setupH` accepts the draw `d`; the statement's
`fast100` schedule has at least two levels (as `TabSpec` asks) and its link points are `mPts`. -/
structure SetupJ (j : ℕ) (d : Json) where
  st : Flock.Stmt
  h : Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program =
    .ok st
  m : ℕ
  k0 : ℕ
  l₀ : Accounting.Level
  l₁ : Accounting.Level
  rest : List Accounting.Level
  ext : ℕ
  hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
  hm : 13 ≤ st.m
  hmPts : st.mPts = mPts

variable {I mPts} (ht : I.tags.typed = false)

/-- **A set-up table's table**: setupH's level-0 matrices and pin at the statement's regions (`Layout.model`), its
schedule, and the link layout at `PT_LOCAL`, as #828's `DrawSetup.specAt` at `regionsOf`. -/
noncomputable def SetupJ.spec {j : ℕ} {d : Json} (x : SetupJ I mPts j d) : TabSpec mPts where
  S := (setupH_layout ((tagsAt_typed I.tags I.count j).trans ht) x.h).model (regionsOf x.st)
  m := x.m
  k0 := x.k0
  l₀ := x.l₀
  l₁ := x.l₁
  rest := x.rest
  ext := x.ext
  hsch := x.hsch
  hm := x.hm
  ptLocal := PT_LOCAL
  hlay := ⟨(stmtOf_linkLayout x.st (setupH_wf x.h).1 (setupH_wf x.h).2).1, by
    obtain ⟨-, hr⟩ := setupH_wf x.h
    show PT_LOCAL + (x.st.m - x.st.c.kLog) ≤ mPts
    have h1 := hr.mPts
    have h2 := hr.hm
    have h3 := x.hmPts
    omega⟩

open Classical in
/-- **Table `j` at the draw `d`**: the table flock-verify sets up at part `j` of `d` (`SetupJ.spec`) when it sets that
part up, and the fallback `T₀` otherwise. -/
noncomputable def tableJ (T₀ : TabSpec mPts) (d : Json) (j : ℕ) : TabSpec mPts :=
  if h : Nonempty (SetupJ I mPts j (partJson d I.count j)) then (Classical.choice h).spec ht else T₀

/-- **The plan of `J` tables** at the challenger's draws `dj`: at draw `S`, `I.count` tables, table `j` the one
flock-verify sets up at part `j` of `dj S` (`tableJ`), for every registration. -/
noncomputable def planJ (T₀ : TabSpec mPts) (dj : Finset (Fin n) → Json) {Reg : Type} :
    Finset (Fin n) → Reg → List (TabSpec mPts) :=
  fun S _ => List.ofFn fun j : Fin I.count => tableJ ht T₀ (dj S) j.val

variable {Reg : Type}

/-- The number of table `T`'s extra claims (two per region; the link points don't change it). -/
noncomputable def nXT (T : TabSpec mPts) : ℕ :=
  (Model.extraClaims T.S T.ptLocal mPts (fun _ _ => (0 : 𝔽))).length

/-- **The decoder at table `j`**: dec-zl's one-table decoder, the zerocheck's and lincheck's wires, the opening's, then
Ligerito's (`wZL ++ (wOP ++ wLIG)`), at table `j`'s dimensions, number of claims and schedule. -/
noncomputable def decJ (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (S : Finset (Fin n)) (R : Reg)
    (j : Fin (plan S R).length) : ℕ → ByteArray → List Msg :=
  decOf (wZL ((plan S R).get j).S.m ((plan S R).get j).S.kLog ++
    (wOP (nXT ((plan S R).get j) + 2) ++ wLIG ((plan S R).get j).sch))

variable (I) in
/-- **Table `j`'s openings**: `opnOf` at the statement setupH sets up under table `j`'s tags at part `j` of the draw
`dj S`, and its `fast100` schedule, on proof files `2j` and `2j + 1`. -/
noncomputable def opnJ (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (dj : Finset (Fin n) → Json)
    (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) (j : Fin (plan S R).length) :
    Opens 𝔽 Dig × Salts (List UInt8) :=
  match Stmt.setupH (tagsAt I.tags I.count j.val) I.circuitFile I.publicFile I.coins I.tables
      (some (partJson (dj S) I.count j.val)) I.partition I.program with
  | .ok st => match Flock.fast100 st.m with
    | some sch => opnOf (Setup.ofCircuit st) sch (nXT ((plan S R).get j)) #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | none => default
  | .error _ => default

variable (I) in
/-- **The verify command's tags at `J` tables**: public outputs, retained rounds, `hm96-sha512/v1` leaves, more than one
table. -/
def TagsOkJ : Prop :=
  I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1" ∧ 1 < I.count

variable (I) in
/-- **`wr` is the challenger's record of the live session of `J` tables**: whenever flock-verify accepts what `wr`
writes for the outcome `o` at draw `S`, the record carries the draw `dj S`, the proof files are `o`'s, and the record
decodes to a session whose link points are `o`'s and whose table `j`'s `root_B` and rep streams' rounds are `o`'s. -/
def RecordsLiveJ (dj : Finset (Fin n) → Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript) : Prop :=
  ∀ S R o, Accepts I (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOf I (wr S R o).record = .ok (spec, sts) → Record.decode spec (wr S R o).record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin (Jf S R), sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

variable (I mPts) in
/-- **The challenger's draw at `S`, at `J` tables**: wherever setupH sets table `j` up at its part of `dj S`, it does so
with `mPts` link points. -/
def DrawOkJ (dj : Finset (Fin n) → Json) (S : Finset (Fin n)) : Prop :=
  ∀ j < I.count, ∀ st, Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables
    (some (partJson (dj S) I.count j)) I.partition I.program = .ok st → st.mPts = mPts

/-- **The decoder is right at `J` tables** (R11b at `--session-tables J`): given the tags, `wr` the challenger's record
and the challenger's draws setting each table up at `mPts` link points, `DecodesOnJ` holds at the tables flock-verify
sets up (`planJ`, at any fallback `T₀`) with the decoder `decJ` on each table's two streams, single coins as field
elements, no final messages, table `j`'s cap from its root `root_B[j]` (`capOfRoot`) and its openings `opnJ`. -/
def DecodesOnJExec : Prop :=
  ∀ {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (T₀ : TabSpec mPts) (dj : Finset (Fin n) → Json)
    {Reg : Type} (L : Law n)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planJ ht T₀ dj S R).length mPts → Exec.Transcript),
    TagsOkJ I → RecordsLiveJ I dj (fun S R => (planJ ht T₀ dj S R).length) wr →
    (∀ ω, DrawOkJ I mPts dj (L.draw ω)) →
    DecodesOnJ H512 enc512 leaf512 (planJ ht T₀ dj) I wr (opnJ I (planJ ht T₀ dj) dj) (decJ (planJ ht T₀ dj))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L

end FlockSoundness.Discharge.DecodeZLJ
