module

public import Properties.Spot.Tables
import all Init.Data.Range.Basic
import all Init.Data.ByteArray.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Decode
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.Pred
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Pattern.String
import all Init.Data.Char.Basic
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def

/-! # Proofs of v2's holding the blobs' lookup tables to the registered circuit's (`Properties.Spot.Tables`)

`tablesHeld` reads the circuit file, its lookups and the blobs' tables, and throws where the two lists of names differ,
so its acceptance is their equality (`tablesHeld_ok`). `run` binds it first (`run_tables`), and on parameters that
name no members `checkBlobsAt` accepts only where its chain of refusals ends in `run`'s acceptance
(`checkBlobsAt_run`). On parameters that name members it accepts only where the per-member run returned, which holds
every member's statement (`Spot.OneStageZk.membersRun_files`), and a member's statement binds `tablesHeld` first at
the member's own files (`statement_tables`, `checkBlobsAt_members_tables`).

`tablesRegistered_honest` is its honest lemma: blobs holding a circuit file that looks up `rcp` and `rcp`'s table, as
the served path hands v2 (`honestBlobs`), pass the refusal. `tablesRegistered_nontrivial` is its nontrivial lemma: blobs
holding a circuit file that looks up no table and `rcp`'s table (`tableBlobs`) hold a table the circuit doesn't look
up, so a check that accepts every input, as one without the refusal does there, does not have `TablesRegistered`'s
conclusion. The kernel evaluates both on the witnesses (`decide +kernel`), reading the toolchain's string, byte array
and JSON definitions through `import all`, all but `String.trimAsciiEnd`, whose recursion is well-founded: `metaOf_of`
takes the META line's trim as a hypothesis, unfolded once on each witness's line.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem tablesHeld_ok {p : OneStageZk.Own} {bs : Core.Blobs} (h : OneStageZk.tablesHeld p bs = .ok ()) :
    HoldsRegisteredTables p bs := by
  unfold OneStageZk.tablesHeld at h
  obtain ⟨c, hc, h1⟩ := bind_ok h
  obtain ⟨w, hw, h2⟩ := bind_ok h1
  refine ⟨c, hc, ?_⟩
  rw [hw]
  dsimp only at h2
  split at h2
  · cases h2
  · rename_i hne
    simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hne
    rw [hne]

theorem run_tables {p : OneStageZk.Own} {ss : Array OneStageZk.RunFiles} {pu : Option (Array Nat)} {bs : Core.Blobs}
    {r : OneStageZk.Run} (h : OneStageZk.run p ss pu bs = .ok r) : HoldsRegisteredTables p bs := by
  unfold OneStageZk.run at h
  obtain ⟨u, hu, -⟩ := bind_ok h
  exact tablesHeld_ok hu

theorem checkBlobsAt_run {p : OneStageZk.Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : OneStageZk.Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity} (hm : p.members = none)
    (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ r, OneStageZk.run p.own e.sessions (some e.base.provedUnits) bs = .ok r := by
  replace h := (OneStageZk.checkBlobsAt_accept_one hm h).1
  unfold OneStageZk.checkOneAt at h
  dsimp only at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨-, -, h1⟩ := bind_ok hr
    obtain ⟨t, -, h2⟩ := bind_ok h1
    obtain ⟨-, -, h3⟩ := bind_ok h2
    obtain ⟨-, -, h4⟩ := bind_ok h3
    obtain ⟨-, -, h5⟩ := bind_ok h4
    obtain ⟨f, -, h6⟩ := bind_ok h5
    obtain ⟨-, -, h7⟩ := bind_ok h6
    exact ⟨r, h7⟩

/-- A member's statement binds `tablesHeld` first, at the member's own files. -/
theorem statement_tables {m : OneStageZk.Member} {slice : Lean.Json} {bs : Core.Blobs} {a : OneStageZk.StepAt}
    (h : m.statement slice bs = .ok a) : HoldsRegisteredTables m.own bs := by
  unfold OneStageZk.Member.statement at h
  obtain ⟨a', hs, -⟩ := bind_ok h
  unfold OneStageZk.statementOf at hs
  obtain ⟨u, hu, -⟩ := bind_ok hs
  exact tablesHeld_ok hu

/-- **On parameters that name members, an accept holds every member's registered tables**, sampled or not: the
per-member run returned (`Spot.OneStageZk.checkMembersAt_accept`), so every member's statement was read
(`Spot.OneStageZk.membersRun_files`). -/
theorem checkBlobsAt_members_tables {p : OneStageZk.Params} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : OneStageZk.Evidence} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    {ms : Array OneStageZk.Member} (hm : p.members = some ms)
    (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) : ∀ m ∈ ms, HoldsRegisteredTables m.own bs := by
  obtain ⟨sts, -, h⟩ := OneStageZk.checkBlobsAt_accept_members hm h
  obtain ⟨r, hr, -, -⟩ := OneStageZk.checkMembersAt_accept h
  obtain ⟨roots, -, hf⟩ := OneStageZk.membersRun_files hr
  intro m hmem
  obtain ⟨a, ha, -⟩ := hf m hmem
  exact statement_tables ha

theorem tablesRegistered : TablesRegistered := by
  refine ⟨fun _ _ => tablesHeld_ok, fun p c ξ e bs pub prof h => ?_⟩
  unfold AcceptHoldsTables
  split
  · rename_i hm
    exact (checkBlobsAt_run hm h).elim fun _ hr => run_tables hr
  · rename_i ms hm
    exact checkBlobsAt_members_tables hm h

private theorem ok_of_toOption {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

private theorem ok_bind {ε α β : Type} {a : α} {f : α → Except ε β} : (Except.ok a >>= f) = f a := rfl

/-- `tablesHeld` accepts where the circuit file is among the blobs and its lookups are the blobs' tables. -/
private theorem tablesHeld_of {p : OneStageZk.Own} {bs : Core.Blobs} {c : ByteArray}
    (hc : bs.get p.circuit "circuit" = .ok c)
    (hl : OneStageZk.lookupsOf c = .ok ((OneStageZk.tablesOf bs).map (·.1))) : OneStageZk.tablesHeld p bs = .ok () := by
  unfold OneStageZk.tablesHeld
  rw [hc, ok_bind, hl, ok_bind]
  simp
  rfl

/-- `Flock.HmOut.metaOf` at a circuit file whose second line is `line`, with no trailing whitespace. -/
private theorem metaOf_of {bytes : ByteArray} {n1 : Nat} {line : String} (h1 : Flock.findByte bytes 10 = some n1)
    (h2 : String.fromUTF8? (bytes.extract (n1 + 1) ((Flock.findByte bytes 10 (n1 + 1)).getD bytes.size)) = some line)
    (h3 : line.trimAsciiEnd.toString = line) (h4 : line.startsWith "META " = true) :
    Flock.HmOut.metaOf bytes = match Flock.JsonParse.parse (line.drop 5).toString with
      | .ok j => .ok j
      | .error e => .error s!"META: {e}" := by
  unfold Flock.HmOut.metaOf
  simp only [h1, h2, h3, h4]
  rfl

private theorem honest_trim :
    ("META {\"lookups\":{\"rcp\":{}}}".trimAsciiEnd).toString = "META {\"lookups\":{\"rcp\":{}}}" := by
  unfold String.trimAsciiEnd String.Slice.trimAsciiEnd String.Slice.dropEndWhile String.Slice.skipSuffixWhile
  rw [String.Slice.Pos.revSkipWhile]
  decide +kernel

private theorem table_trim : ("META {}".trimAsciiEnd).toString = "META {}" := by
  unfold String.trimAsciiEnd String.Slice.trimAsciiEnd String.Slice.dropEndWhile String.Slice.skipSuffixWhile
  rw [String.Slice.Pos.revSkipWhile]
  decide +kernel

private theorem honest_lookups : OneStageZk.lookupsOf honestCircuit = .ok ["rcp"] := by
  apply ok_of_toOption
  unfold OneStageZk.lookupsOf
  rw [metaOf_of (n1 := 7) (by decide +kernel) (by decide +kernel) honest_trim (by decide +kernel)]
  decide +kernel

private theorem table_lookups : OneStageZk.lookupsOf tableCircuit = .ok [] := by
  apply ok_of_toOption
  unfold OneStageZk.lookupsOf
  rw [metaOf_of (n1 := 7) (by decide +kernel) (by decide +kernel) table_trim (by decide +kernel)]
  decide +kernel

/-- **`TablesRegistered`'s honest lemma**: blobs holding a circuit file that looks up `rcp` and `rcp`'s table pass v2's
refusal. -/
theorem tablesRegistered_honest : OneStageZk.tablesHeld witnessOwn honestBlobs = .ok () := by
  refine tablesHeld_of (c := honestCircuit) (ok_of_toOption (by decide +kernel)) ?_
  rw [honest_lookups]
  have h : (OneStageZk.tablesOf honestBlobs).map (·.1) = ["rcp"] := by decide +kernel
  rw [h]

/-- **`TablesRegistered`'s nontrivial lemma**: blobs holding a circuit file that looks up no table and `rcp`'s table
hold a table the circuit doesn't look up, so a check that accepts them, as one that accepts every input does, does not
have `TablesRegistered`'s conclusion. -/
theorem tablesRegistered_nontrivial : ¬ HoldsRegisteredTables witnessOwn tableBlobs := by
  rintro ⟨c, hc, hl⟩
  have hg : tableBlobs.get witnessOwn.circuit "circuit" = .ok tableCircuit := ok_of_toOption (by decide +kernel)
  rw [hg] at hc
  cases hc
  have h : (OneStageZk.tablesOf tableBlobs).map (·.1) = ["rcp"] := by decide +kernel
  rw [table_lookups, h] at hl
  cases hl

end Properties.Spot.SecurityProofs
