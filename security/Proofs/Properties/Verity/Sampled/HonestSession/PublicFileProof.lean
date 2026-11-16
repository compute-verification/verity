module

public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFile
public import Proofs.Flock.SessionLoop.Concrete.Public
public import Proofs.Flock.SessionLoop.Concrete.Inputs
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
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Pattern.Pred
import all Init.Data.String.Subslice
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.PosRaw
import all Init.Data.String.FindPos
import all Init.Data.String.Termination
import all Init.Data.String.ToSlice
import all Init.Data.Array.Attach
import all Init.WF
import all Init.Data.Nat.Fold

/-!
# Proof of `PublicFileFacts`

Each field of `PublicFileFacts` (`PublicFile.lean`), given `CircuitFileFacts`. The core is `load_pf`, a forward walk
(`Proofs.Flock.SessionLoop.Concrete.Fwd`) of `HmRow.loadPublic` over `pubF sSha sCls`: the two header strings that are
tainted by the blocked `canon`/`canonUtf8` (the circuit sha, via META's program/template digests, and the class hex) are
abstracted so the walk's constructor recognition survives, then instantiated to the concrete file by `pubF_eq`. The
load-derived fields reduce `HmOut`/`HmIn.loadPublic` to that `HmRow` result; `drawn_ok` and `checkPublic_ok` walk the
remaining readers over the known public value.
-/

@[expose] public section

open Flock Flock.Firewall RecDeploy.VStarNets FlockVBridge.Names
open Lean (Json)
open Std.DTreeMap.Internal (Impl)
open FlockSoundness.Discharge.Recursion.FirewallJson (parse_printJson)
open FlockDeploy.Concrete (objT plain_obj_of plainStr_hexOf noNl_hexOf noNl utf8_noTen findByte_eq
  canon_eq canonUtf8_eq canon_obj_congr ok_of_isOk objM IntJ)

set_option maxHeartbeats 4000000
set_option maxRecDepth 200000

namespace Verity.Sampled.Instance.HonestSession

local instance lawfulBEqByteArray : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## Tags: `Zk.tags` touches only `identity`, so the public-file fields project through to `Tags.circuit`. -/

theorem tagsZ_fmt : tagsZ.publicFormat = Flock.Tags.circuit.publicFormat := rfl
theorem tagsZ_stmt : tagsZ.statement = Flock.Tags.circuit.statement := rfl
theorem tagsZ_pck : tagsZ.publicCircuitKey = "circuit_sha512" := by decide +kernel

/-! ## The class SHA-512 is 64 bytes, so not empty. -/

theorem classSha_ne : classSha.isEmpty = false := by
  unfold classSha
  simp only [ByteArray.isEmpty, Sha512.hash_size]
  decide

/-! ## Parameterized header: the one tainted field (`classes = hexOf classSha`) is abstracted to `sCls`.
The walk runs over `pubF sCls` with `sCls` symbolic (so `printJson (hdrF sCls)` is untainted and the fwd tactic's
constructor recognition survives); the concrete file is `pubF (hexOf classSha)`, recovered by `pubF_eq`. -/

def unitsF (sCls : String) : Json := Json.mkObj
  [("blocks", .arr #[.arr #[jn 0]]), ("classes", .arr #[.str sCls]), ("indices", .arr #[jn 0]),
   ("units_per_vu", jn 1), ("vus_per_block", jn 1)]
def hdrF (sSha sCls : String) : Json := Json.mkObj
  [("circuit_sha512", .str sSha), ("format", .str Flock.Tags.circuit.publicFormat),
   ("frame_v3", fv3J), ("instances", jn nUnits), ("rows", .bool false),
   ("statement", .str Flock.Tags.circuit.statement), ("units", unitsF sCls)]
def pubF (sSha sCls : String) : ByteArray := (printJson (hdrF sSha sCls)).toUTF8 ++ "\n".toUTF8 ++ rowBc

theorem unitsF_eq : unitsF (hexOf classSha) = unitsJ := rfl
theorem hdrF_eq : hdrF (hexOf circuitSha) (hexOf classSha) = hdrJ := rfl
theorem pubF_eq : pubF (hexOf circuitSha) (hexOf classSha) = publicFile := rfl

/-! ## Plainness of the header JSON. -/

theorem objY_plain {a : String} (ha : plainStr a = true) : plain (Json.mkObj [("y", .str a)]) = true := by
  rw [show Json.mkObj [("y", .str a)] = .obj ⟨⟨objT (Json.mkObj [("y", .str a)])⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT (Json.mkObj [("y", .str a)])).toListModel = [⟨"y", .str a⟩] from by kernel_rfl')
    (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq]
  exact ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], ha⟩

theorem fmt_plain : plain (.str Flock.Tags.circuit.publicFormat) = true := by
  rw [show Flock.Tags.circuit.publicFormat = "flock-circuit-inputs" from by kernel_rfl']
  simp [plain, plainStr, String.all_bool_eq]
theorem stmt_plain : plain (.str Flock.Tags.circuit.statement) = true := by
  rw [show Flock.Tags.circuit.statement = "verity/flock-circuit" from by kernel_rfl']
  simp [plain, plainStr, String.all_bool_eq]

theorem fv3_plain : plain fv3J = true := by
  rw [show fv3J = .obj ⟨⟨objT fv3J⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT fv3J).toListModel =
    [⟨"domain_ids", Json.mkObj [("y", .str (hexOf domY))]⟩, ⟨"frame", .str "frame-v3-sha512"⟩,
     ⟨"roots", Json.mkObj [("y", .str (hexOf rootY))]⟩,
     ⟨"row_key_sha512", .str (hexOf (Sha512.hash Hm96.Default512.key))⟩,
     ⟨"schemas", Json.mkObj [("y", .str HmRow.ROW_SCHEMA_V2)]⟩] from by kernel_rfl') (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], objY_plain (plainStr_hexOf _)⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], objY_plain (plainStr_hexOf _)⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], plainStr_hexOf _⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn],
      objY_plain (by simp [HmRow.ROW_SCHEMA_V2, plainStr, String.all_bool_eq])⟩⟩

theorem units_plainF (sCls : String) (hs : plainStr sCls = true) : plain (unitsF sCls) = true := by
  rw [show unitsF sCls = .obj ⟨⟨objT (unitsF sCls)⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT (unitsF sCls)).toListModel =
    [⟨"blocks", .arr #[.arr #[jn 0]]⟩, ⟨"classes", .arr #[.str sCls]⟩, ⟨"indices", .arr #[jn 0]⟩,
     ⟨"units_per_vu", jn 1⟩, ⟨"vus_per_block", jn 1⟩] from by kernel_rfl') (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp only [plain, plainList, Bool.and_true]; exact hs⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainStr, String.all_bool_eq, jn]⟩⟩

theorem hdr_plainF (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    plain (hdrF sSha sCls) = true := by
  rw [show hdrF sSha sCls = .obj ⟨⟨objT (hdrF sSha sCls)⟩⟩ from by kernel_rfl']
  refine plain_obj_of (show (objT (hdrF sSha sCls)).toListModel =
    [⟨"circuit_sha512", .str sSha⟩, ⟨"format", .str Flock.Tags.circuit.publicFormat⟩,
     ⟨"frame_v3", fv3J⟩, ⟨"instances", jn nUnits⟩, ⟨"rows", .bool false⟩,
     ⟨"statement", .str Flock.Tags.circuit.statement⟩, ⟨"units", unitsF sCls⟩] from by kernel_rfl') (by kernel_rfl') ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], hpsSha⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], fmt_plain⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], fv3_plain⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], by simp [plain, plainList, plainStr, String.all_bool_eq, jn]⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], stmt_plain⟩,
    ⟨by simp [plain, plainList, plainStr, String.all_bool_eq, jn], units_plainF sCls hs⟩⟩

/-! ## The header text reads back (parameterized over the abstract class string). -/

theorem fj_normF (sSha sCls : String) :
    FlockSoundness.Discharge.Recursion.FirewallJson.norm (hdrF sSha sCls) = hdrF sSha sCls := by kernel_rfl'
theorem RNF (sSha sCls : String) : Flock.norm (hdrF sSha sCls) = hdrF sSha sCls := by kernel_rfl'

theorem hdr_noTenF (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    (10 : UInt8) ∉ (printJson (hdrF sSha sCls)).toUTF8.data.toList :=
  utf8_noTen _ (noNl _ (hdr_plainF sSha sCls hpsSha hs))

/-- `pubF` re-associated to match the reference's `A ++ (nl ++ body)`. -/
def tailB : ByteArray := "\n".toUTF8 ++ rowBc
theorem pfF_eq (sSha sCls : String) : pubF sSha sCls = (printJson (hdrF sSha sCls)).toUTF8 ++ tailB := by
  unfold pubF tailB; rw [ByteArray.append_assoc]

theorem R1F (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    findByte (pubF sSha sCls) 10 = some (printJson (hdrF sSha sCls)).toUTF8.size := by
  have hn : List.findIdx? (· == (10 : UInt8)) (printJson (hdrF sSha sCls)).toUTF8.data.toList = none :=
    List.findIdx?_eq_none_iff.2 fun x hx => by
      cases hx10 : (x == 10)
      · rfl
      · exact absurd ((beq_iff_eq.1 hx10) ▸ hx) (hdr_noTenF sSha sCls hpsSha hs)
  have h10 : "\n".toUTF8.data.toList = [10] := by decide +kernel
  rw [findByte_eq, pfF_eq, tailB, ByteArray.data_append, ByteArray.data_append, Array.toList_append,
    Array.toList_append, h10, List.drop_zero, List.findIdx?_append, hn, Option.none_or, List.cons_append,
    List.findIdx?_cons, beq_self_eq_true, if_pos rfl, Option.map_some, Option.map_some, Nat.zero_add, Nat.add_zero,
    Array.length_toList]
  rfl

theorem R2F (sSha sCls : String) :
    (pubF sSha sCls).extract 0 (printJson (hdrF sSha sCls)).toUTF8.size = (printJson (hdrF sSha sCls)).toUTF8 := by
  rw [pfF_eq]; exact ByteArray.extract_append_eq_left rfl

theorem R4F (sSha sCls : String) (hpsSha : plainStr sSha = true) (hs : plainStr sCls = true) :
    JsonParse.parse (printJson (hdrF sSha sCls)) = .ok (Flock.norm (hdrF sSha sCls)) := by
  rw [parse_printJson (hdr_plainF sSha sCls hpsSha hs), fj_normF, RNF]

theorem R5F (sSha sCls : String) :
    (pubF sSha sCls).extract ((printJson (hdrF sSha sCls)).toUTF8.size + 1) (pubF sSha sCls).size = rowBc := by
  rw [pfF_eq, tailB, ← ByteArray.append_assoc]
  exact ByteArray.extract_append_eq_right (by rw [ByteArray.size_append]; rfl) ByteArray.size_append

/-! ## The classes/sha checks and the units-entry canonicalisation (handled through the blocked `canon`). -/

theorem hnunits : nUnits = 1 := rfl
/-- String `!=` reflexivity (the generic `bne_self_eq_false` needs a `ReflBEq` the rewriter cannot discharge here). -/
theorem bneStr (s : String) : (s != s) = false := by simp
/-- The parsed `circuit_sha512` string `sSha` equals `hexOf c.sha`, so the circuit-key check passes -- a targeted
rewrite on the full check term so that `sSha` is *not* rewritten inside the (tainted) header text. -/
theorem hShaChkF (c : Flock.Circuit) (sSha : String) (h : sSha = hexOf c.sha) :
    (sSha != hexOf c.sha) = false := by subst h; simp
/-- The classes array as `mapM asStr` builds it (an `emptyWithCapacity … |>.push`). -/
theorem hclsSizeF (sCls : String) :
    ((Array.emptyWithCapacity #[Json.str sCls].size).push sCls).size = 1 := rfl
/-- The indices array as `mapM asNat` builds it, normalised to `#[0]` so the (concrete) ascending check reduces. -/
theorem idxArr : (Array.emptyWithCapacity #[jn 0].size).push (0 : Nat) = #[0] := rfl
/-- The one parsed class string `sCls` equals `hexOf c.classSha`, so the class check passes -- stated as a targeted
rewrite on the full check term so that `sCls` is *not* rewritten inside the (tainted) header text. -/
theorem hclsAnyF (c : Flock.Circuit) (sCls : String) (h : sCls = hexOf c.classSha) :
    Array.any ((Array.emptyWithCapacity #[Json.str sCls].size).push sCls) (· != hexOf c.classSha) 0 1 = false := by
  subst h; simp

/-! ## The concrete canonicalisations, to discharge `load_pf`'s hypotheses at the real file. -/

def s3 : String := (canon unitsJ).toOption.getD ""
def s4 : String := (canonUtf8 hdrJ).toOption.getD ""

theorem intJ_units : IntJ unitsJ = true := by decide +kernel
theorem intJ_hdr : IntJ hdrJ = true := by kernel_rfl

theorem hu1 : canon unitsJ = .ok s3 := canon_eq intJ_units
theorem hd : canonUtf8 hdrJ = .ok s4 := canonUtf8_eq intJ_hdr

/-- `unitsEntry` at a circuit with `g = upv = 1`, class `classSha`, over the one index `#[0]`. -/
def UEJmine : Json := Json.mkObj [("vus_per_block", Lean.toJson (1 : Nat)), ("units_per_vu", Lean.toJson (1 : Nat)),
  ("indices", Json.arr #[Lean.toJson (0 : Nat)]), ("blocks", Json.arr #[Json.arr #[Lean.toJson (0 : Nat)]]),
  ("classes", Json.arr #[Json.str (hexOf classSha)])]

theorem ue_eq (c : Flock.Circuit) (hg : c.g = 1) (hupv : c.upv = 1) (hcls : c.classSha = classSha) :
    HmRow.unitsEntry c #[0] = UEJmine := by
  unfold HmRow.unitsEntry
  simp only [hg, hupv, hcls, classSha_ne]
  kernel_rfl'

theorem ue_arr : (objM UEJmine).toArray = (objM unitsJ).toArray := by kernel_rfl'

theorem hu2 (c : Flock.Circuit) (hg : c.g = 1) (hupv : c.upv = 1) (hcls : c.classSha = classSha) :
    canon (HmRow.unitsEntry c #[0]) = .ok s3 := by
  rw [ue_eq c hg hupv hcls,
    show UEJmine = .obj (objM UEJmine) from by kernel_rfl',
    canon_obj_congr ue_arr,
    show (.obj (objM unitsJ) : Json) = unitsJ from by kernel_rfl', hu1]

/-! ## The walk. -/

fwd_def pubPF load_pf (c : Flock.Circuit) (sSha sCls s3' s4' : String)
    (hpsSha : plainStr sSha = true) (hps : plainStr sCls = true)
    (hsha : sSha = hexOf c.sha) (hsCls : sCls = hexOf c.classSha)
    (hports : c.ports = #[yPort]) (hg : c.g = 1) (hupv : c.upv = 1) (hout : c.outWords = 0)
    (hce : c.classSha.isEmpty = false)
    (hu1' : canon (unitsF sCls) = .ok s3') (hu2' : canon (HmRow.unitsEntry c #[0]) = .ok s3')
    (hd' : canonUtf8 (hdrF sSha sCls) = .ok s4') :
    HmRow.loadPublic tagsZ c (pubF sSha sCls) by
  [R1F sSha sCls hpsSha hps, R2F sSha sCls, fromUTF8?_toUTF8, R4F sSha sCls hpsSha hps, R5F sSha sCls, RNF sSha sCls,
    bneStr, tagsZ_fmt, tagsZ_stmt, tagsZ_pck, hShaChkF c sSha hsha, hports, hg, hupv, hout, hce,
    hnunits, hclsSizeF sCls, hclsAnyF c sCls hsCls, idxArr, hu1', hu2', hd']
  block [canon, canonUtf8]
  shallow []

/-! ## Instantiating the walk at the concrete file. -/

/-- The walk at any circuit carrying the file's fields, over the concrete `publicFile`. -/
theorem loadRowGen (c : Flock.Circuit) (hsha : c.sha = circuitSha) (hcls : c.classSha = classSha)
    (hports : c.ports = #[yPort]) (hg : c.g = 1) (hupv : c.upv = 1) (hout : c.outWords = 0) :
    HmRow.loadPublic tagsZ c publicFile = .ok (pubPF c (hexOf circuitSha) (hexOf classSha) s3 s4) := by
  have h := load_pf c (hexOf circuitSha) (hexOf classSha) s3 s4
    (plainStr_hexOf _) (plainStr_hexOf _) (by rw [hsha]) (by rw [hcls]) hports hg hupv hout
    (by rw [hcls]; exact classSha_ne)
    (by rw [unitsF_eq]; exact hu1) (hu2 c hg hupv hcls) (by rw [hdrF_eq]; exact hd)
  rwa [pubF_eq] at h

/-- The walk at the hidden-output circuit `{cH with outPorts := #[]}` that both `HmOut`/`HmIn.loadPublic` pass down. -/
theorem loadRowOut (hC : CircuitFileFacts) :
    HmRow.loadPublic tagsZ { cH with outPorts := #[] } publicFile
      = .ok (pubPF { cH with outPorts := #[] } (hexOf circuitSha) (hexOf classSha) s3 s4) :=
  loadRowGen { cH with outPorts := #[] } hC.cH_sha hC.cH_classSha hC.cH_ports hC.cH_g hC.cH_upv rfl

/-- The concrete loaded public value. -/
noncomputable def Vout : Flock.Public := pubPF { cH with outPorts := #[] } (hexOf circuitSha) (hexOf classSha) s3 s4

/-! ## The fields. -/

/-- `HmOut.loadPublic` reduces to the `HmRow` walk (its own `outPorts := #[]`): the output-row table check is skipped. -/
theorem hmOutRaw (hC : CircuitFileFacts) : HmOut.loadPublic tagsZ cH publicFile = .ok Vout := by
  unfold HmOut.loadPublic
  rw [loadRowOut hC]
  kernel_rfl'

theorem pubO_eq (hC : CircuitFileFacts) : pubO = Vout := by
  unfold pubO; rw [hmOutRaw hC]; rfl

/-- Field `hmOut_loadPublic`, proved as stated. -/
theorem hmOut_loadPublic_pf (hC : CircuitFileFacts) : HmOut.loadPublic tagsZ cH publicFile = .ok pubO := by
  rw [pubO_eq hC]; exact hmOutRaw hC

/-- `HmIn.loadPublic` over empty ports reduces to the same `HmRow` walk: the public-inputs match takes the
`(.error _, true)` branch (the header carries no `public_inputs`), and `checkReads` returns at once. -/
theorem hmInRaw (hC : CircuitFileFacts) : HmIn.loadPublic tagsZ cH {} publicFile = .ok Vout := by
  unfold HmIn.loadPublic
  rw [loadRowGen { cH with outPorts := if (({} : HmIn.Ports).ports.isEmpty) then #[]
        else #[("public inputs", ({} : HmIn.Ports).words)] }
      hC.cH_sha hC.cH_classSha hC.cH_ports hC.cH_g hC.cH_upv rfl]
  kernel_rfl'

theorem pubI_eq (hC : CircuitFileFacts) : pubI = Vout := by
  unfold pubI; rw [hmInRaw hC]; rfl

/-- Field `hmIn_loadPublic`, proved as stated. -/
theorem hmIn_loadPublic_pf (hC : CircuitFileFacts) : HmIn.loadPublic tagsZ cH {} publicFile = .ok pubI := by
  rw [pubI_eq hC]; exact hmInRaw hC

/-- Field `pubO_n`, proved as stated. -/
theorem pubO_n_pf (hC : CircuitFileFacts) : pubO.n = nUnits := by
  rw [pubO_eq hC]; kernel_rfl'

/-- Field `pubI_n`, proved as stated. -/
theorem pubI_n_pf (hC : CircuitFileFacts) : pubI.n = nUnits := by
  rw [pubI_eq hC]; kernel_rfl'

/-- Field `pubO_domain`, proved as stated: the loaded header's `frame_v3.domain_ids.y` unhexes to the zero domain. -/
theorem pubO_domain_pf (hC : CircuitFileFacts) : pubO.fv3Bytes "domain_ids" "y" = .ok domY := by
  rw [pubO_eq hC]; kernel_rfl'

/-! ## The drawn walk.

The sample draws the one unit of the population of one. The only tainted term the walk builds is the new header's
`publicDigest` (`canonUtf8`); abstract its value to `s5` and block `canonUtf8`. The new header is the loaded header with
`unit_draw`, `instances`, `population`, `units` reset (`setObjVal!`); the `instances` count and the drawn `indices`
reduce to `1` and `#[0]` by the kernel, so `hcu` states them that way. -/

fwd_def drawnPF drawn_pf (c : Flock.Circuit) (sSha sCls s3' s4' s5 : String)
    (hports : c.ports = #[yPort]) (hg : c.g = 1) (hupv : c.upv = 1) (hout : c.outWords = 0)
    (hce : c.classSha.isEmpty = false)
    (hcu : canonUtf8 (((((hdrF sSha sCls).setObjVal! "unit_draw" draw₀).setObjVal! "instances"
        (Lean.toJson (1 : Nat))).setObjVal! "population" (Lean.toJson (1 : Nat))).setObjVal!
        "units" (HmRow.unitsEntry c #[0])) = .ok s5) :
    HmRow.drawn c (pubPF c sSha sCls s3' s4') draw₀ by
  [hports, hg, hupv, hout, hce, hcu]
  block [canonUtf8]
  shallow []

/-! ## Instantiating the drawn walk at the concrete file. -/

/-- The new header the draw builds over the concrete file: the loaded `hdrJ` with the sample's `unit_draw`, one instance,
population one, and the drawn unit's entry `UEJmine`. -/
def drawHdrJ : Json :=
  (((((hdrF (hexOf circuitSha) (hexOf classSha)).setObjVal! "unit_draw" draw₀).setObjVal! "instances"
      (Lean.toJson (1 : Nat))).setObjVal! "population" (Lean.toJson (1 : Nat))).setObjVal! "units" UEJmine)

theorem intJ_drawHdr : IntJ drawHdrJ = true := by kernel_rfl
def s5d : String := (canonUtf8 drawHdrJ).toOption.getD ""
theorem hcud : canonUtf8 drawHdrJ = .ok s5d := canonUtf8_eq intJ_drawHdr

/-- The drawn walk's `canonUtf8` hypothesis at any circuit with the file's fields: its new header's `units` entry is the
concrete `UEJmine` (`ue_eq`), so its canonicalisation is the concrete `hcud`. -/
theorem hcuGen (c : Flock.Circuit) (hg : c.g = 1) (hupv : c.upv = 1) (hcls : c.classSha = classSha) :
    canonUtf8 (((((hdrF (hexOf circuitSha) (hexOf classSha)).setObjVal! "unit_draw" draw₀).setObjVal! "instances"
        (Lean.toJson (1 : Nat))).setObjVal! "population" (Lean.toJson (1 : Nat))).setObjVal!
        "units" (HmRow.unitsEntry c #[0])) = .ok s5d := by
  rw [ue_eq c hg hupv hcls]; exact hcud

/-- The drawn walk at any circuit carrying the file's fields, over the concrete loaded value. -/
theorem drawnRowGen (c : Flock.Circuit) (hsha : c.sha = circuitSha) (hcls : c.classSha = classSha)
    (hports : c.ports = #[yPort]) (hg : c.g = 1) (hupv : c.upv = 1) (hout : c.outWords = 0) :
    HmRow.drawn c (pubPF c (hexOf circuitSha) (hexOf classSha) s3 s4) draw₀
      = .ok (drawnPF c (hexOf circuitSha) (hexOf classSha) s3 s4 s5d) :=
  drawn_pf c (hexOf circuitSha) (hexOf classSha) s3 s4 s5d hports hg hupv hout
    (by rw [hcls]; exact classSha_ne) (hcuGen c hg hupv hcls)

/-- The drawn walk at the hidden-output circuit `{cH with outPorts := #[]}` over `Vout`. -/
theorem drawnRowOut (hC : CircuitFileFacts) :
    HmRow.drawn { cH with outPorts := #[] } Vout draw₀
      = .ok (drawnPF { cH with outPorts := #[] } (hexOf circuitSha) (hexOf classSha) s3 s4 s5d) :=
  drawnRowGen { cH with outPorts := #[] } hC.cH_sha hC.cH_classSha hC.cH_ports hC.cH_g hC.cH_upv rfl

/-- The sampled statement's public value: the drawn value over `Vout`. Explicit: one instance (`n = 1`), the one drawn
unit's indices `#[0]`, and the sha folded as `Sha512.hash (s5d.toUTF8 ++ …)` (the new header's `publicDigest`). -/
noncomputable def Wdrawn : Flock.Public :=
  drawnPF { cH with outPorts := #[] } (hexOf circuitSha) (hexOf classSha) s3 s4 s5d

/-- The drawn load as an equation with an explicit value, beside the field (the setup piece reads `pubD`'s indices, `n`
and sha from here). -/
theorem drawnRaw (hC : CircuitFileFacts) : HmIn.drawn cH {} pubI draw₀ = .ok Wdrawn := by
  rw [pubI_eq hC]; exact drawnRowOut hC

theorem pubD_eq (hC : CircuitFileFacts) : pubD = Wdrawn := by
  unfold pubD; rw [drawnRaw hC]; rfl

/-- Field `drawn_ok`, proved as stated. -/
theorem drawn_ok_pf (hC : CircuitFileFacts) : HmIn.drawn cH {} pubI draw₀ = .ok pubD := by
  rw [pubD_eq hC]; exact drawnRaw hC

/-- `HmIn.checkOwn` over no public input port, no own recomputation: the match `(none, isEmpty = true)` returns, whatever
the population value. -/
theorem checkOwn_ok_pf : HmIn.checkOwn {} pubI none = .ok () := rfl

/-! ## The dummy row and `checkPublic`.

`HmOut.checkPublic` recomputes each row's frame-v3 root from the population's `b ‖ c` against the header's, and checks
each row's pinned dummy `b ‖ c` against the padding instance's row under a zero salt. Everything but three circuit reads
(`c.ports`, `c.dummyDigests` and the dummy row `HmOut.dummyRow c p`) is read off the (concrete) sampled public value, so
the walk takes `hC`'s field equations for those three and reduces the rest. The dummy row at the one hidden-output port
is the zero row: its output words all come from `c.dummyOutputs = #[0]`, so every byte is zero whatever the output
group's `(at_, w)`. -/

theorem getD01 (j : Nat) : (#[(0:Nat)]).getD j 0 = 0 := by
  rcases j with _ | j
  · rfl
  · rfl

/-- The hidden-output dummy row at the one port is the zero row. -/
theorem dummyRow_zero (c : Flock.Circuit) (hp : c.ports = #[yPort]) (hd : c.dummyOutputs = #[0]) :
    HmOut.dummyRow c 0 = zeroRow := by
  simp only [HmOut.dummyRow]
  obtain ⟨a, b, at_, w⟩ :=
    (if 0 < HmOut.nIn c then ((0:Nat), (0:Nat), (0:Nat), (0:Nat))
      else HmOut.outGroup c (0 - HmOut.nIn c))
  simp only [hp, hd, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
    List.forIn_pure_yield_eq_foldl, getD01, ite_self]
  kernel_rfl'

/-- The one port's pinned-dummy check is false (no throw): the recomputed `rowBcZeroSalt yPort (dummy row)` is
`dummyBc` itself, so `a != a`. Both `#[yPort][0]!` and `#[dummyBc].getD 0 _` reduce (no SHA forced). -/
theorem dummyCheck_false :
    (HmOut.rowBcZeroSalt (#[yPort][0]!) zeroRow != #[dummyBc].getD 0 ByteArray.empty) = false := by
  rw [show HmOut.rowBcZeroSalt (#[yPort][0]!) zeroRow = #[dummyBc].getD 0 ByteArray.empty from rfl]
  exact bne_self_eq_false _

/-- Field `checkPublic_ok`, proved as stated. The one row's recomputed root matches the header's, and its pinned dummy
`b ‖ c` is the zero row's under a zero salt (`dummyBc`). -/
theorem checkPublic_ok_pf (hC : CircuitFileFacts) : HmOut.checkPublic cH pubD = .ok () := by
  rw [pubD_eq hC]
  unfold HmOut.checkPublic
  simp only [hC.cH_ports, hC.cH_dummyDigests, Std.Legacy.Range.forIn_eq_forIn_range',
    Std.Legacy.Range.size, List.size_toArray, List.length_cons, List.length_nil,
    Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, Nat.zero_add, List.range'_one,
    List.forIn_cons, List.forIn_nil, dummyRow_zero cH hC.cH_ports hC.cH_dummyOutputs]
  fwd_walk [dummyCheck_false] block [] shallow []

end Verity.Sampled.Instance.HonestSession
