module

public import Proofs.Properties.Spot
import Proofs.Flock.Claims.CanonOk
import Proofs.Pouw.Period.Sort
import all Init.Data.Array.DecidableEq
import all Init.Data.Array.QSort.Basic
import all Init.Data.Range.Basic
import all Init.Data.Ord.String
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Init.Data.String.Defs
import all Init.Data.String.Slice
import all Init.Data.String.TakeDrop
import all Init.Data.String.Pattern.String
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic

/-! # `AcceptMeans`' witnesses at an honest one-stage audit (`Properties.Spot.Honest`)

`acceptMeans_honest`: the check accepts the honest audit, by completeness (`acceptMeans_complete`), each of whose
premises is proved at the instance. Its partition is evaluated to its one unit (`units_program`), R1–R6 are kept
against the verifier's copies and R8 by its one Call, which reads no run Input, its sample is drawn and its unit
proved, and its verdict accepts. The record's partition and
query digests and the commitment's digest stay the SHA-512s the check computes: they are compared only with themselves.
The parameters name no challenger, so the window names no seed salt.

`acceptMeans_nontrivial`: the conclusion fails for the same window under parameters naming a challenger and the
window's open root (`Honest.saltParams`), whose evidence carries no seed salt, which a check that accepts everything
accepts: `OneStage.seedSaltOf` refuses the missing evidence, so no seed salt is named (`SeedSaltNamed`).

The lemmas are private to this module: they unfold the code's definitions, and only the two witnesses are exported.
-/

open Lean (Json)
open Spot

namespace Properties.Spot.Honest

open Properties.Pouw.Period.SecurityProofs.Sort (qsort_of_sorted qsort_eq_of_perm)
open FlockDeploy (canon_str canon_nat canon_mkObj)

instance instDecidableEqTRange_proofs : DecidableEq Flock.TemplateQuery.TRange := fun a b =>
  decidable_of_iff ((a.node, a.template, a.first, a.count) = (b.node, b.template, b.first, b.count)) (by
    cases a; cases b; simp)

/-! ## Evaluating a check by parts -/

theorem eq_ok {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x <;> simp_all [Except.toOption]

theorem jeq_self {x : Json} (h : ∃ t, Flock.canon x = .ok t) : jeq x x = true := by
  obtain ⟨t, ht⟩ := h
  simp [jeq, ht]

theorem sorted {as bs : Array String} (hp : as.toList.Perm bs.toList) (hs : bs.toList.Pairwise (· < ·)) :
    as.qsort (· < ·) = bs :=
  Array.toList_inj.1 ((qsort_eq_of_perm hp).trans (qsort_of_sorted hs))

/-- An object's keys, sorted, are its map's keys, which it holds sorted. -/
theorem keys_ofList {l : List (String × Json)} {ks : Array String} (hs : ks.toList.Pairwise (· < ·))
    (h : (Std.TreeMap.Raw.ofList l compare).toArray.map (·.1) = ks) :
    ((Std.TreeMap.Raw.ofList l compare).toArray.map (·.1)).qsort (· < ·) = ks := by
  rw [h]
  exact Array.toList_inj.1 (qsort_of_sorted hs)

theorem keys?_mkObj {l : List (String × Json)} {ks : Array String} (hs : ks.toList.Pairwise (· < ·))
    (h : (Std.TreeMap.Raw.ofList l compare).toArray.map (·.1) = ks) : keys? (Json.mkObj l) = some ks := by
  show some ((((Std.TreeMap.Raw.ofList l compare).toArray.map (·.1))).qsort (· < ·)) = _
  rw [keys_ofList hs h]

theorem qsort_single (x : Int) : #[x].qsort (· < ·) = #[x] :=
  Array.toList_inj.1 (List.perm_singleton.1 (Array.perm_iff_toList_perm.1 (Properties.RemoteAudit.qsort_perm #[x] _)))

/-! ## The partition: one unit -/

theorem keys_query : keys? query = some #["name", "params", "version"] :=
  keys?_mkObj (by decide) (by decide +kernel)

theorem build_query : Partition.build hex query = Json.mkObj partitionItems := by
  simp only [Partition.build, keys_query]
  rfl

theorem isSha512Hex_hex : Flock.HmRow.isSha512Hex hex = true := by
  unfold Flock.HmRow.isSha512Hex
  rw [String.all_bool_eq]
  decide +kernel

theorem templatesOf_query (v : Nat) :
    Flock.TemplateQuery.templatesOf "Q_template_instance" v (Json.mkObj templateParams) =
      if v = 0 then some (.ok #["T"]) else none := by
  have k : ((Std.TreeMap.Raw.ofList templateParams compare).toArray.map (·.1)).qsort (· < ·) = #["template"] :=
    keys_ofList (by decide) (by decide +kernel)
  rw [show Json.mkObj templateParams = Json.obj (Std.TreeMap.Raw.ofList templateParams compare) from rfl]
  unfold Flock.TemplateQuery.templatesOf
  dsimp only
  rw [k]
  by_cases hv : v = 0
  · subst hv
    rfl
  · simp [hv]

theorem checkPartitionObject_partition :
    Flock.HmRow.checkPartitionObject (Json.mkObj partitionItems) = .ok ("Q_template_instance", 0) := by
  have k1 : ((Std.TreeMap.Raw.ofList partitionItems compare).toArray.map (·.1)).qsort (· < ·) =
      #["format", "program", "query"] := keys_ofList (by decide) (by decide +kernel)
  have k2 : ((Std.TreeMap.Raw.ofList queryItems compare).toArray.map (·.1)).qsort (· < ·) =
      #["name", "params", "version"] := keys_ofList (by decide) (by decide +kernel)
  have hp : ((Json.obj (Std.TreeMap.Raw.ofList partitionItems compare)).getObjValAs? String "program").toOption.getD
      "" = hex := rfl
  have hq : ((Json.obj (Std.TreeMap.Raw.ofList partitionItems compare)).getObjVal? "query").toOption.getD Json.null =
      Json.obj (Std.TreeMap.Raw.ofList queryItems compare) := rfl
  have hps : ((Json.obj (Std.TreeMap.Raw.ofList queryItems compare)).getObjVal? "params").toOption.getD Json.null =
      Json.mkObj templateParams := rfl
  have hn : ((Json.obj (Std.TreeMap.Raw.ofList queryItems compare)).getObjValAs? String "name").toOption.getD "" =
      "Q_template_instance" := rfl
  rw [show Json.mkObj partitionItems = Json.obj (Std.TreeMap.Raw.ofList partitionItems compare) from rfl]
  unfold Flock.HmRow.checkPartitionObject
  dsimp only
  simp only [k1, hp, isSha512Hex_hex, hq, k2, hps, hn, templatesOf_query]
  exact eq_ok (by decide +kernel)

theorem valid_partition : Partition.valid (Json.mkObj partitionItems) = true := by
  unfold Partition.valid
  rw [checkPartitionObject_partition]
  rfl

theorem range'_three (s n : Nat) :
    List.range' s (n + 3) 1 = s :: (s + 1) :: (s + 2) :: List.range' (s + 3) n 1 := by
  simp only [List.range'_succ]

/-- The walk's first three steps: it stops at the third, past the two Definitions. -/
theorem range'_fuel : List.range' 0 Flock.Steps.FUEL 1 = 0 :: 1 :: 2 :: List.range' 3 (Flock.Steps.FUEL - 3) 1 := by
  have h : Flock.Steps.FUEL = (Flock.Steps.FUEL - 3) + 3 := by unfold Flock.Steps.FUEL; omega
  conv => lhs; rw [h]
  exact range'_three 0 _

theorem orderViolation_prog : prog.orderViolation = none := by
  unfold Flock.Program.Prog.orderViolation
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel,
    Nat.div_one, range'_fuel]
  simp [prog, rootDefn, templateDefn, List.forIn_cons]

theorem evaluate_template : Flock.TemplateQuery.evaluate program.desc #["T"] = .ok #[⟨0, "T", 0, 1⟩] :=
  eq_ok (by decide +kernel)

theorem templateBits_template : Partition.templateBits prog 1 = .ok 16 := by
  have h : prog.defs[1]! = templateDefn := rfl
  unfold Partition.templateBits
  rw [h]
  simp only [templateDefn, Flock.Program.Ty.widths]
  rfl

theorem checkTemplates_template : Partition.checkTemplates prog #[⟨0, "T", 0, 1⟩] = .ok () := by
  have h1 : (prog.defs[prog.root]!).body.nodes[0]!.fn = 1 := rfl
  have h2 : (prog.defs[1]!).gates = 1 := rfl
  have hu : Partition.unitClass 16 = some 16 := by decide +kernel
  unfold Partition.checkTemplates
  simp [h1, h2, templateBits_template]
  simp only [bind, Except.bind, hu]
  rfl

theorem evaluate_program : Partition.evaluate program query = .ok #[⟨0, "T", 0, 1⟩] := by
  have hb : Partition.build (Flock.hexOf program.sha512) query = Json.mkObj partitionItems := build_query
  have hq : getD (Json.mkObj partitionItems) "query" = query := rfl
  have hn : (query.getObjValAs? String "name").toOption.getD "" = "Q_template_instance" := rfl
  have hv : (nat? (getD query "version")).getD 0 = 0 := rfl
  have hp : getD query "params" = Json.mkObj templateParams := rfl
  have ho : program.prog.orderViolation = none := orderViolation_prog
  have hc : Partition.checkTemplates program.prog #[⟨0, "T", 0, 1⟩] = .ok () := checkTemplates_template
  unfold Partition.evaluate
  simp only [hb, valid_partition, ho, hq, hn, hv, hp, templatesOf_query, Bool.not_true, Bool.false_eq_true,
    ↓reduceIte]
  simp only [bind, Except.bind, evaluate_template, hc]
  rfl

theorem units_program : Partition.units program query = .ok 1 := by
  unfold Partition.units
  rw [evaluate_program]
  rfl

/-! ## The registration: R1–R6 and R8 -/

theorem canon_query : ∃ t, Flock.canon query = .ok t := by
  apply canon_mkObj
  simp only [queryItems, List.mem_cons, List.not_mem_nil, or_false]
  rintro p (rfl | rfl | rfl)
  · exact canon_str _
  · apply canon_mkObj
    simp only [templateParams, List.mem_cons, List.not_mem_nil, or_false]
    rintro p rfl
    exact canon_str _
  · exact canon_nat _

theorem canon_partition : ∃ t, Flock.canon (Json.mkObj partitionItems) = .ok t := by
  apply canon_mkObj
  simp only [partitionItems, List.mem_cons, List.not_mem_nil, or_false]
  rintro p (rfl | rfl | rfl)
  · exact canon_str _
  · exact canon_str _
  · exact canon_query

theorem canon_law : ∃ t, Flock.canon law = .ok t := by
  apply canon_mkObj
  simp only [List.mem_cons, List.not_mem_nil, or_false]
  rintro p (rfl | rfl)
  · exact canon_str _
  · exact canon_nat _

theorem digest_partition : ∃ d, Partition.digest (Json.mkObj partitionItems) = some d := by
  obtain ⟨t, ht⟩ := canon_partition
  unfold Partition.digest Flock.HmRow.partitionDigest
  simp only [valid_partition, Bool.or_true, ↓reduceIte, ht, Option.map_eq_some_iff]
  exact ⟨_, _, rfl, rfl⟩

theorem keys_KEYS : Registration.KEYS = #["anchors", "format", "law", "leaf_layer", "partition", "population",
    "previous", "program", "protocol", "query", "roots", "scheme", "window"] :=
  sorted (as := #["format", "protocol", "program", "query", "partition", "population", "law", "scheme", "roots",
    "leaf_layer", "anchors", "window", "previous"]) (by decide) (by decide)

theorem keys_ROOT_KEYS : Registration.ROOT_KEYS = #["domain", "leaves", "port", "root"] :=
  sorted (as := #["port", "root", "leaves", "domain"]) (by decide) (by decide)

theorem partitionOf_verifier : partitionOf verifier = Json.mkObj partitionItems := build_query

theorem r1 : R1 record verifier law := by
  refine ⟨?_, jeq_self (canon_str _), jeq_self (canon_str _), rfl, jeq_self canon_law, rfl,
    ⟨"served", by decide, jeq_self (canon_str _)⟩⟩
  show keys? record = some Registration.KEYS
  rw [keys_KEYS]
  exact keys?_mkObj (by decide) (by decide +kernel)

theorem r2 : R2 record verifier := by
  rw [R2, partitionOf_verifier]
  refine ⟨jeq_self (canon_str _), jeq_self (canon_str _), ?_, ?_, ⟨1, units_program, jeq_self (canon_nat _)⟩, trivial⟩
  · intro q hq
    change Registration.queryDigest query = .ok q at hq
    show jeq (.str queryHex) (.str q) = true
    rw [show queryHex = q by simp [queryHex, hq]]
    exact jeq_self (canon_str _)
  · obtain ⟨d, hd⟩ := digest_partition
    refine ⟨d, hd, ?_⟩
    show jeq (.str partitionHex) (.str d) = true
    rw [show partitionHex = d by simp [partitionHex, hd]]
    exact jeq_self (canon_str _)

theorem rootsOf_record : rootsOf record = #[rootEntry] := rfl

theorem portsOf_record : portsOf record = #[.str "x"] := by
  have h : getD rootEntry "port" = .str "x" := rfl
  unfold portsOf
  rw [rootsOf_record]
  simp [h]

theorem rootFor_record : rootFor record "x" = some rootEntry := by
  have h : jeq (getD rootEntry "port") (.str "x") = true := by decide +kernel
  unfold rootFor
  rw [rootsOf_record]
  simp [h]

theorem isHex_root : isHex "00" = true := by
  unfold isHex
  rw [String.all_bool_eq]
  decide +kernel

theorem r3 : R3 record verifier := by
  refine ⟨jeq_self (canon_str _), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [rootsOf_record]
    simp
  · rw [rootsOf_record]
    intro r hr
    simp only [Array.mem_singleton] at hr
    subst hr
    rw [keys_ROOT_KEYS]
    exact keys?_mkObj (by decide) (by decide +kernel)
  · rw [portsOf_record, rootsOf_record]
    rfl
  · rw [portsOf_record]
    rfl
  · intro pw hpw
    simp only [verifier, Array.mem_singleton] at hpw
    subst hpw
    exact ⟨.str "x", by simp [portsOf_record], jeq_self (canon_str _)⟩
  · intro pw hpw r hr
    simp only [verifier, Array.mem_singleton] at hpw
    subst hpw
    rw [rootFor_record] at hr
    cases hr
    refine ⟨⟨"00", rfl, isHex_root⟩, ?_, ?_⟩
    · show jeq (Lean.toJson (1 : Nat)) (Lean.toJson (1 : Nat)) = true
      exact jeq_self (canon_nat _)
    · show jeq (.str "d") (.str "d") = true
      exact jeq_self (canon_str _)

theorem r4 : R4 record verifier := by
  unfold R4
  decide +kernel

theorem r5 : R5 record verifier := by
  unfold R5
  decide +kernel

theorem r6 : R6 record verifier :=
  ⟨by decide +kernel, by simp [verifier], by simp [verifier], by unfold WindowPoolReset; decide +kernel,
    by unfold WindowPoolWeek; decide +kernel, by unfold WindowPoolSpent; decide +kernel,
    by simp [PoolResetOnce, verifier]⟩

theorem templateRanges_program : Partition.templateRanges? program query = some #[⟨0, "T", 0, 1⟩] := by
  have hn : (query.getObjValAs? String "name").toOption.getD "" = "Q_template_instance" := rfl
  have hv : nat? (getD query "version") = some 0 := by decide +kernel
  unfold Partition.templateRanges?
  simp only [hn, hv, evaluate_program]
  rfl

/-- R8: the verifier ties no value, and the one unit's Call takes no argument, so it reads no run Input. Every array
`Partition.sharedInputs` sorts is empty, which the kernel evaluates. -/
theorem r8 : R8 verifier := by
  have ht : Partition.templateRanges? verifier.program verifier.query = some #[⟨0, "T", 0, 1⟩] :=
    templateRanges_program
  refine ⟨by decide +kernel, ?_⟩
  unfold Partition.readAlone
  rw [ht]
  decide +kernel

theorem registered : Registered verifier record law :=
  ⟨r1, r2, r3, r4, r5, r6, fun h => by simp [ExpectsV2, verifier] at h, r8⟩

/-! ## The sample, the units and the verdicts -/

theorem drawKeys_subset : drawKeys "subset" false = #["k", "law", "population", "units"] :=
  sorted (as := #["law", "population", "units"] ++ #["k"]) (by decide) (by decide)

theorem lawOf_draw : Draw.lawOf draw = law := by
  have h1 : jeq (getD draw "law") (.str "stratified") = false := by decide +kernel
  have h2 : jeq (getD draw "law") (.str "work") = false := by decide +kernel
  have h3 : jeq (getD draw "law") (.str "subset") = true := by decide +kernel
  unfold Draw.lawOf
  simp only [h1, h2, h3, Bool.or_false, Bool.false_eq_true, ↓reduceIte]
  rfl

theorem drawn : Drawn draw (getD record "population") law := by
  have hk : kindOf draw = "subset" := by decide +kernel
  have hc : ("subset" == "work" && (field? draw "closure").isSome) = false := by decide +kernel
  unfold Drawn
  simp only [hk, hc]
  refine ⟨rfl, by decide +kernel, ?_, ?_, by decide +kernel, #[Lean.toJson (0 : Nat)], #[0], rfl, by decide +kernel, ?_, ?_,
    fun _ => by decide +kernel, fun h => absurd h (by decide), fun h => absurd h (by decide),
    fun h => absurd h (by decide)⟩
  · rw [drawKeys_subset]
    exact keys?_mkObj (by decide) (by decide +kernel)
  · rw [lawOf_draw]
    exact jeq_self canon_law
  · intro i h
    simp at h
  · intro _
    exact ⟨by simp, 1, by decide +kernel, by simp⟩

theorem proved_draw : Draw.proved draw = #[0] := by
  have hu : getD draw "units" = .arr #[Lean.toJson (0 : Nat)] := rfl
  have hc : getD draw "closure" = .null := rfl
  have hf : #[Lean.toJson (0 : Nat)].filterMap int? = #[(0 : Int)] := by decide +kernel
  unfold Draw.proved
  simp only [hu, hc, hf, Array.append_empty]
  exact qsort_single 0

theorem unitsProved : UnitsProved commitment evidence := by
  have h1 : (field? record "registered").isSome = false := by decide +kernel
  have h2 : (openedRoots record evidence.verdicts).isSome = false := by decide +kernel
  refine ⟨?_, ?_, jeq_self (canon_str _), ?_, ?_, ?_⟩
  · show 1 = (Draw.proved draw).size
    rw [proved_draw]
    rfl
  · show #[(0 : Nat)].map (fun n : Nat => (n : Int)) = Draw.proved draw
    rw [proved_draw]
    decide +kernel
  · intro s h
    simp [evidence] at h
  · intro s h
    simp [evidence] at h
  · intro h
    exact absurd h (by simp only [commitment, h1, h2]; decide)

theorem verdictsAccept : VerdictsAccept evidence.verdicts := by
  refine ⟨?_, ⟨("v", Json.mkObj [("accepted", .bool true), ("of_record", .bool true)]), by simp [evidence],
    by decide +kernel⟩, ?_⟩
  · intro x hx _
    simp only [evidence, Array.mem_singleton] at hx
    subst hx
    decide +kernel
  · intro x hx _
    simp only [evidence, Array.mem_singleton] at hx
    subst hx
    show Json.bool true ≠ Json.null
    nofun

theorem profileDraw_law : Record.profileDraw law = some (.subset 1) := by
  have h : getD law "law" = .str "subset" := rfl
  have hk : nat? (getD law "k") = some 1 := by decide +kernel
  unfold Record.profileDraw
  simp only [h, hk]
  rfl

theorem population_record : nat? (getD record "population") = some 1 := by decide +kernel

theorem seedSaltNamed : SeedSaltNamed params commitment evidence := by
  refine ⟨none, rfl, ?_⟩
  unfold WindowSeedSalt
  decide +kernel

end Properties.Spot.Honest

public section

namespace Properties.Spot

/-- **`AcceptMeans`' honest lemma**: the check accepts the honest one-stage audit (`Honest`), a program of one Call of a
one-gate template whose one unit is sampled and proved, under a record keeping R1–R6 and R8, and publishes `publicOf`
and `profileOf` for its `subset` sample of one unit; by completeness, each of its premises proved at the instance. -/
theorem acceptMeans_honest :
    OneStage.check Honest.params Honest.commitment Honest.evidence =
      .accept (publicOf Honest.params Honest.evidence Honest.law (.subset 1) 1)
        (profileOf Honest.params Honest.commitment Honest.evidence (.subset 1) 1) :=
  acceptMeans_complete _ _ _ _ _ _ Honest.registered Honest.drawn Honest.unitsProved Honest.verdictsAccept
    Honest.profileDraw_law Honest.population_record Honest.seedSaltNamed

/-- **`AcceptMeans`' nontrivial lemma**: its conclusion fails at the honest audit's window under parameters naming a
challenger and the window's open root (`Honest.saltParams`), whose evidence carries no seed salt, which a check that
accepts everything accepts, whatever the law, its profile's sample and the population: the seed salt is refused as
missing (`OneStage.seedSaltOf`), so the window names none read from the challenger's log (`SeedSaltNamed`). -/
theorem acceptMeans_nontrivial :
    ∀ (law : Json) (d : Core.Draw) (n : Nat),
      ¬ (Registered Honest.saltParams.verifier Honest.commitment.record law ∧
        Drawn Honest.evidence.draw (getD Honest.commitment.record "population") law ∧
        UnitsProved Honest.commitment Honest.evidence ∧ VerdictsAccept Honest.evidence.verdicts ∧
        Record.profileDraw law = some d ∧ nat? (getD Honest.commitment.record "population") = some n ∧
        barePublic = publicOf Honest.saltParams Honest.evidence law d n ∧
        bareProfile = profileOf Honest.saltParams Honest.commitment Honest.evidence d n ∧
        SeedSaltNamed Honest.saltParams Honest.commitment Honest.evidence) := by
  rintro _ _ _ ⟨-, -, -, -, -, -, -, -, salt, hs, -⟩
  simp [OneStage.seedSaltOf, Honest.saltParams, Honest.evidence] at hs

end Properties.Spot

end
