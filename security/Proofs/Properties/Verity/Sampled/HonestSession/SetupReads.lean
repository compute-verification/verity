module

public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFileProof
public import Proofs.Properties.Verity.Sampled.HonestSession.SetupProof
public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile
public import Proofs.Flock.SessionLoop.Concrete.Delta
public import Proofs.Flock.Soundness.Discharge.Hidden.Delta
import Flock.HmRow
import Flock.Canon
import all Init.Data.Array.QSort.Basic
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.Array.Attach
import all Init.WF
meta import Lean

/-!
# The setup's reads at the honest circuit

`Stmt.setupHidden` at the honest session's inputs reads the circuit `cR` (`CircuitFileProof`), the public file as
`PublicFileFacts` gives it, and the program side through three calls. The circuit's reads are equations on `cR` by the
kernel: META's flags, the pin, the regions (`regions_Y`, whose bytes the drawn public file supplies) and Δ, which
`HmOut.delta` builds as `HmRow.delta`'s, then `outPairs` and `pinList` (`Hidden.hidden_delta'`), so `maskFree` reads
`HmRow.deltaL`'s lists (`Concrete.all_dA`) and never `delta`'s arrays. `setup_stH` walks the setup to `stH` with the
public-file and program reads as hypotheses, and `shapeOk_stH` checks its shape.
-/

@[expose] public section

open Flock
open Lean (Json)
open FlockSoundness.Discharge
open Verity.Sampled.Instance.HonestSession.CircuitFileProof

set_option maxHeartbeats 100000000
set_option maxRecDepth 200000

namespace Verity.Sampled.Instance.HonestSession.SetupReads

/-! ## META's flags -/

theorem cR_links : J.has cR.metaJ "links" = false := by kernel_rfl'
theorem cR_outRowPortsFlag : J.has cR.metaJ "out_row_ports" = false := by kernel_rfl'
theorem cR_rowSchemas : J.has cR.metaJ "row_schemas" = true := by kernel_rfl'
theorem cR_pinned : HmRow.pinnedCount cR.metaJ = 0 := by kernel_rfl'

theorem cH_links : J.has cH.metaJ "links" = false := by rw [cH_eq]; exact cR_links
theorem cH_outRowPortsFlag : J.has cH.metaJ "out_row_ports" = false := by rw [cH_eq]; exact cR_outRowPortsFlag
theorem cH_rowSchemas : J.has cH.metaJ "row_schemas" = true := by rw [cH_eq]; exact cR_rowSchemas
theorem cH_pinned : HmRow.pinnedCount cH.metaJ = 0 := by rw [cH_eq]; exact cR_pinned

/-! ## The pin and the regions -/

theorem cR_g : cR.g = 1 := by kernel_rfl'
theorem cR_ports : cR.ports = #[yPort] := by kernel_rfl'

theorem cR_pin : HmRow.pin cR = .ok 246784 := by kernel_rfl'
theorem cH_pin : HmRow.pin cH = .ok 246784 := by rw [cH_eq]; exact cR_pin

/-- The one region's bytes at block `b`: instance `b`'s row digest, or past the population the pinned dummy's. -/
noncomputable def bytesY (pub : Public) (b : Nat) : ByteArray :=
  (Array.range cR.g).foldl (fun acc g => acc ++ match cR.instOf pub.n b g with
    | some i => pub.digest i 0 cR.ports.size
    | none => cR.dummyDigests.getD 0 ByteArray.empty) ByteArray.empty

/-- The output row's region: its hm96 slot's `b ‖ c`, ten free bits. -/
noncomputable def regY (pub : Public) : Region :=
  ⟨"digest 0", (Array.range 10).map (0 + ·) ++ (Array.range (Nat.log2 cR.g)).map ((cR.rangeOf cR.hmNet).slotLog + Nat.log2 (cR.perVu.getD cR.hmNet 1) + ·),
    (2 ^ cR.kLog - 1) &&& ((2 ^ cR.kLog - 1) ^^^ (((Array.range 10).map (0 + ·) ++ (Array.range (Nat.log2 cR.g)).map
      ((cR.rangeOf cR.hmNet).slotLog + Nat.log2 (cR.perVu.getD cR.hmNet 1) + ·)).foldl (fun acc b => acc ||| 2 ^ b) 0)),
    (cR.rangeOf cR.hmNet).start + 0 * 2 ^ (cR.rangeOf cR.hmNet).slotLog +
      ((portCols (cR.nets.getD cR.hmNet default).2.outGroups).getD 0 default).1,
    bytesY pub⟩

theorem regions_Y (pub : Public) : HmOut.regions cR pub = .ok #[regY pub] := by
  unfold HmOut.regions
  dsimp only
  rw [mapM_one _ (Array.range cR.ports.size) (by simp [cR_ports])]
  kernel_rfl'

theorem regions_H (pub : Public) : HmOut.regions cH pub = .ok #[regY pub] := by rw [cH_eq]; exact regions_Y pub

theorem inRegions_Y (pub : Public) : HmIn.regions cR {} pub = .ok #[] := rfl

theorem bytesY_eq (pub : Public) (b : Nat) : bytesY pub b = ByteArray.empty ++ (match cR.instOf pub.n b 0 with
    | some i => pub.digest i 0 1
    | none => dummyBc) := by
  unfold bytesY
  rw [cR_g, cR_ports]
  all_goals kernel_rfl'

theorem bytesY_size (pub : Public) (hn : pub.n = 1) (hdig : (pub.digest 0 0 1).size = 128) (b : Nat) :
    (bytesY pub b).size = 128 := by
  have he : ByteArray.empty.size = 0 := rfl
  rw [bytesY_eq, ByteArray.size_append, he, Nat.zero_add]
  rcases Nat.eq_zero_or_pos b with h0 | hpos
  · subst h0
    have hi : cR.instOf pub.n 0 0 = some 0 := by simp [Circuit.instOf, cR_g, hn]
    rw [hi]
    exact hdig
  · have hi : cR.instOf pub.n b 0 = none := by
      simp only [Circuit.instOf, cR_g, hn]
      split <;> first | rfl | omega
    rw [hi]
    exact dummyBc_size

theorem checkRegionBytes_Y (pub : Public) (hn : pub.n = 1) (hdig : (pub.digest 0 0 1).size = 128) :
    checkRegionBytes 3 (#[regY pub] ++ (#[] : Array Region)) = .ok () := by
  have hfree : (regY pub).free.size = 10 := by kernel_rfl'
  have hall : ((#[regY pub] ++ (#[] : Array Region)).all fun reg =>
      Nat.all (2 ^ 3) fun b _ => (reg.bytes b).size * 8 == 2 ^ reg.free.size) = true := by
    have h1 : ∀ q : Region → Bool, (#[regY pub] ++ (#[] : Array Region)).all q = q (regY pub) := by
      intro q; rw [← Array.all_toList]; simp
    rw [h1]
    show (Nat.all (2 ^ 3) fun b _ => ((regY pub).bytes b).size * 8 == 2 ^ (regY pub).free.size) = true
    rw [nat_all_iff]
    intro b _
    rw [hfree, show (regY pub).bytes = bytesY pub from rfl, bytesY_size pub hn hdig b]
    all_goals rfl
  unfold checkRegionBytes
  rw [if_pos hall]
  all_goals rfl

/-! ## Δ -/

/-- Δ_A and Δ_B as the setup builds them at `cR` and the pin `p`. -/
noncomputable def DA (p : Nat) : Array (Nat × Nat) := (HmIn.delta cR {} p).1
noncomputable def DB (p : Nat) : Array (Nat × Nat) := (HmIn.delta cR {} p).2

theorem delta_DA (p : Nat) : HmIn.delta cR {} p = (DA p, DB p) := rfl

theorem DA_eq (p : Nat) :
    DA p = FlockDeploy.Concrete.dA cR p ++ (Hidden.outPairs cR).toArray ++ (Hidden.pinList cR).toArray := by
  unfold DA HmIn.delta
  rw [Hidden.hidden_delta' cR p]
  all_goals rfl

theorem DB_eq (p : Nat) :
    DB p = FlockDeploy.Concrete.dB cR p ++ (Hidden.outPairs cR).toArray ++ (Hidden.pinList cR).toArray := by
  unfold DB HmIn.delta
  rw [Hidden.hidden_delta' cR p]
  all_goals rfl

theorem all_listToArray (l : List (Nat × Nat)) (q : Nat × Nat → Bool) : l.toArray.all q = l.all q := by
  rw [← Array.all_toList, List.toList_toArray]

/-- `maskFree`'s reads of Δ, as lists the kernel evaluates without copying an array. -/
theorem all_DA (p : Nat) (q : Nat × Nat → Bool) : (DA p).all q =
    ((HmRow.deltaL cR p).1.all q && (Hidden.outPairs cR).all q && (Hidden.pinList cR).all q) := by
  rw [DA_eq, Array.all_append, Array.all_append, FlockDeploy.Concrete.all_dA, all_listToArray, all_listToArray]

theorem all_DB (p : Nat) (q : Nat × Nat → Bool) : (DB p).all q =
    ((HmRow.deltaL cR p).2.all q && (Hidden.outPairs cR).all q && (Hidden.pinList cR).all q) := by
  rw [DB_eq, Array.all_append, Array.all_append, FlockDeploy.Concrete.all_dB, all_listToArray, all_listToArray]

/-! ## The setup -/

/-- `verify --zk`'s coins at the session's one table. -/
def coinsZ : Flock.Coins := .tree (Zk.coinSpec (Flock.Tags.circuit.tableNames 1))

theorem parseCommit_tZ : HmOut.parseCommit tZ circuitFile [] = .ok cR := by rw [← tagsZ_eq]; exact parseCommit_eq
theorem hmIn_parse_R : HmIn.parse cR = .ok {} := by rw [← cH_eq]; exact hmIn_parse
theorem checkOwn_any (pop : Public) : HmIn.checkOwn {} pop none = .ok () := rfl
theorem law_draw : (Draw.UnitDraw.ofJson draw₀).map (·.law) = .ok (Draw.Law.subset 1) := by
  rw [draw_ofJson]; all_goals rfl

open Lean Elab Term Meta in
/-- `Stmt.setupHidden`'s auxiliary match on `(law, partition, program)`, the one whose discriminant's first component is
an `Except _ Draw.Law`: the walk's `reduceMatcher?` cannot evaluate a draw it reads in a tuple, so `law_match` steps it. -/
elab "lawMatch%" : term => do
  let some v := (← getConstInfo ``Stmt.setupHidden).value? | throwError "lawMatch%: Stmt.setupHidden has no body"
  let mut seen : MessageData := m!""
  for n in v.getUsedConstants do
    let some info ← getMatcherInfo? n | continue
    let ds ← forallTelescopeReducing (← getConstInfo n).type fun xs _ =>
      (List.range info.numDiscrs).mapM fun i => do whnfR (← inferType xs[info.numParams + 1 + i]!)
    seen := seen ++ m!"\n{n}: {ds}"
    if let [t] := ds then
      if t.isAppOfArity ``Prod 2 && ((t.getArg! 0).find? (·.isConstOf ``Draw.Law)).isSome then
        return ← mkConstWithFreshMVarLevels n
  throwError "lawMatch%: Stmt.setupHidden uses no match on an (Except _ Draw.Law × _); its matches:{seen}"

/-- The setup's stratified-law match at `draw₀` takes its fall-through: `draw₀`'s law is `subset`. -/
theorem law_match {β : Type} (h1 : _) (h2 : _) :
    (lawMatch%) (fun _ => β) (Except.map (fun x => x.law) (Draw.UnitDraw.ofJson draw₀), some partitionObj,
      some programFile) h1 h2 =
    h2 (Except.map (fun x => x.law) (Draw.UnitDraw.ofJson draw₀), some partitionObj, some programFile) := by
  rw [law_draw]

theorem size_qsort {α : Type} (as : Array α) (lt : α → α → Bool) : (as.qsort lt).size = as.size := by
  unfold Array.qsort
  split
  · rfl
  · exact Vector.size_toArray _

/-- `query`'s keys, unsorted: `Spot.keys?` sorts them with `qsort`, which the kernel does not evaluate. -/
def queryKeys : Array String := match query with
  | .obj kvs => kvs.toArray.map (·.1)
  | _ => #[]

theorem keys_query : Spot.keys? query = some (Spot.sortStrings queryKeys) := by kernel_rfl'
theorem queryKeys_size : queryKeys.size = 3 := by kernel_rfl'

theorem query_keys : (Spot.keys? query == some #[]) = false := by
  rw [beq_eq_false_iff_ne, keys_query]
  intro h
  have h3 := congrArg Array.size (Option.some.inj h)
  rw [Spot.sortStrings, size_qsort, queryKeys_size] at h3
  exact absurd h3 (by decide)

/-- The partition object's query, read without the program's hash beside it. -/
theorem partitionObj_query : partitionObj.getObjVal? "query" = .ok query := by
  have h : partitionObj.getObjVal? "query" =
      .ok (if Spot.keys? query == some #[] then Spot.Partition.defaultQuery else query) := by kernel_rfl'
  rw [h, query_keys]
  all_goals rfl

theorem intJ_idZ : FlockDeploy.Concrete.IntJ (Tags.rowV2Identity tZ.identity) = true := by decide +kernel
def sIdZ : String := (canonUtf8 (Tags.rowV2Identity tZ.identity)).toOption.getD ""
theorem identity_Z : identityBytes (Tags.rowV2Identity tZ.identity) = .ok sIdZ.toUTF8 := by
  unfold identityBytes
  rw [FlockDeploy.Concrete.canonUtf8_eq intJ_idZ]
  all_goals rfl

def helloZ (σ : ByteArray) (m : Nat) : String := (helloOf tZ σ m coinsZ).toOption.getD ""
theorem hello_Z (σ : ByteArray) (m : Nat) : helloOf tZ σ m coinsZ = .ok (helloZ σ m) :=
  FlockDeploy.Concrete.ok_of_isOk _ (FlockDeploy.Concrete.canon_ok _ _ (Nat.lt_succ_self _) (by kernel_rfl'))

/-! `setup_stH`: the setup at the honest inputs, over the public file's values (`pop`, `pub`) and the program side's
results (`f`, `note`, `w`) as variables, so that the walk evaluates none of them. -/

fwd_def stH setup_stH (f : HmRow.PartitionFinding) (note : String) (w : Nat) (pop pub : Public) (U : Json)
    (hpart : HmRow.checkPartition cR (some partitionObj) = .ok f)
    (hload : HmIn.loadPublic tZ cR {} publicFile = .ok pop)
    (hdrawn : HmIn.drawn cR {} pop draw₀ = .ok pub)
    (hpub : HmOut.checkPublic cR pub = .ok ())
    (hn : pub.n = 1)
    (hunits : J.get pub.header "units" "public file" = .ok U)
    (hidx : J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)])
    (hdig : (pub.digest 0 0 1).size = 128)
    (hderive : HmRow.deriveTemplateUnits cR partitionObj programFile #[0] = .ok note)
    (hlinks : HmLinks.check cR pop (HmLinks.readersOf (some draw₀)) programFile = .ok w) :
    Stmt.setupHidden tZ circuitFile publicFile coinsZ [] (some draw₀) (some partitionObj) (some programFile) none by
  [parseCommit_tZ, hmIn_parse_R, hpart, hload, checkOwn_any, hdrawn, law_match, law_draw, draw_ofJson, hpub,
    partitionObj_query, hunits, hidx,
    mapM_one, hderive, hlinks, hn, cR_pin, delta_DA, regions_Y, inRegions_Y, checkRegionBytes_Y pub hn hdig,
    identity_Z, hello_Z]
  block [HmIn.delta, DA, DB, HmRow.checkPartition, HmRow.deriveTemplateUnits, HmLinks.check, HmIn.loadPublic,
    HmIn.drawn, HmOut.checkPublic, HmOut.parseCommit, Draw.UnitDraw.ofJson]
  shallow []

theorem shapeOk_stH (f : HmRow.PartitionFinding) (note : String) (w : Nat) (pop pub : Public) (U : Json) :
    Zk.shapeOk (stH f note w pop pub U) = .ok () := by
  have hda : (stH f note w pop pub U).da = DA 246784 := rfl
  have hdb : (stH f note w pop pub U).db = DB 246784 := rfl
  unfold Zk.shapeOk Zk.maskFree
  simp only [hda, hdb, all_DA, all_DB]
  kernel_rfl'

/-! ## `SetupFacts`' two fields -/

section
variable (f : HmRow.PartitionFinding) (note : String) (w : Nat) (U : Json) (hP : PublicFileFacts)
  (hn : pubD.n = 1)
  (hunits : J.get pubD.header "units" "public file" = .ok U)
  (hidx : J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)])
  (hdig : (pubD.digest 0 0 1).size = 128)
  (hpart : HmRow.checkPartition cH (some partitionObj) = .ok f)
  (hderive : HmRow.deriveTemplateUnits cH partitionObj programFile #[0] = .ok note)
  (hlinks : HmLinks.check cH pubI (HmLinks.readersOf (some draw₀)) programFile = .ok w)
include hP hn hunits hidx hdig hpart hderive hlinks

theorem setupHidden_stH : Stmt.setupHidden (Composed.inputsJ inputs 0).tags (Composed.inputsJ inputs 0).circuitFile
    (Composed.inputsJ inputs 0).publicFile (Composed.inputsJ inputs 0).coins (Composed.inputsJ inputs 0).tables
    (some draw₀) (Composed.inputsJ inputs 0).partition (Composed.inputsJ inputs 0).program none =
    .ok (stH f note w pubI pubD U) := by
  have hload := hP.hmIn_loadPublic
  have hdrawn := hP.drawn_ok
  have hpub := hP.checkPublic_ok
  rw [tagsZ_eq, cH_eq] at hload
  rw [cH_eq] at hdrawn hpub hpart hderive hlinks
  rw [tagsJ_eq, tagsZ_eq]
  exact setup_stH f note w pubI pubD U hpart hload hdrawn hpub hn hunits hidx hdig hderive hlinks

theorem st₀_eq : st₀ = stH f note w pubI pubD U := by
  unfold st₀
  exact (congrArg (fun r => r.toOption.getD default)
    (setupHidden_stH f note w U hP hn hunits hidx hdig hpart hderive hlinks)).trans rfl

theorem setupHidden_ok_of : Stmt.setupHidden (Composed.inputsJ inputs 0).tags (Composed.inputsJ inputs 0).circuitFile
    (Composed.inputsJ inputs 0).publicFile (Composed.inputsJ inputs 0).coins (Composed.inputsJ inputs 0).tables
    (some draw₀) (Composed.inputsJ inputs 0).partition (Composed.inputsJ inputs 0).program none = .ok st₀ :=
  (setupHidden_stH f note w U hP hn hunits hidx hdig hpart hderive hlinks).trans
    (congrArg Except.ok (st₀_eq f note w U hP hn hunits hidx hdig hpart hderive hlinks).symm)

theorem shapeOk_ok_of : Zk.shapeOk st₀ = .ok () := by
  rw [st₀_eq f note w U hP hn hunits hidx hdig hpart hderive hlinks]
  exact shapeOk_stH f note w pubI pubD U

end

/-- **`SetupFacts`**, given `PublicFileFacts`, the drawn public file's reads `setup_stH` takes (its population, its units'
indices, its one row digest's size), and the three program-side results. -/
theorem setupFacts_of (hP : PublicFileFacts) (hn : pubD.n = 1)
    (hidx : ∃ U, J.get pubD.header "units" "public file" = .ok U ∧
      J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)])
    (hdig : (pubD.digest 0 0 1).size = 128)
    (hpart : ∃ f, HmRow.checkPartition cH (some partitionObj) = .ok f)
    (hderive : ∃ note, HmRow.deriveTemplateUnits cH partitionObj programFile #[0] = .ok note)
    (hlinks : ∃ w, HmLinks.check cH pubI (HmLinks.readersOf (some draw₀)) programFile = .ok w) : SetupFacts := by
  obtain ⟨U, hu, hi⟩ := hidx
  obtain ⟨f, hf⟩ := hpart
  obtain ⟨note, hd⟩ := hderive
  obtain ⟨w, hw⟩ := hlinks
  exact ⟨draw_ofJson, setupHidden_ok_of f note w U hP hn hu hi hdig hf hd hw,
    shapeOk_ok_of f note w U hP hn hu hi hdig hf hd hw⟩

end Verity.Sampled.Instance.HonestSession.SetupReads
