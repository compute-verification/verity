module

public import Proofs.Flock.SpotFlock.InstanceStmt.Bytes
public import Proofs.Flock.SessionLoop.Concrete.Public
public import Proofs.Flock.SessionLoop.Concrete.Inputs
public import Proofs.Flock.Recursive.FirewallJson
public import Proofs.Flock.Claims.Complete.Json.Record
public import Spot.SecondStage
import Flock.Comp
import Flock.JsonParse
import Flock.Firewall
import Flock.Norm
import Flock.Canon
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

/-! # The two-instance reference statement's JSON, read back

Round 2 reads a session statement as JSON (`SpotFlock`'s `stmtJsonOf`): its public file's first line (`headerOf`), its
sample and its registered reads (`FlockDeploy.jsonOf`). For a run `r` whose strings are plain JSON strings
(`Run.Plain`), at any circuit and class digests `circ`, `cls` that are:

* `headerLine`: the public file's first line is the header's text (`findByte_pub`), which parses back to
  `hdrJ r circ cls` (`parse_hdr`);
* `jsonOf_draw`, `jsonOf_reg`: the sample and the registered reads parse back to `drawJ` and `regJ`.

`InstanceStmt.Round2` makes round 2's statement of them. Each object's members are its list in key order (`*_model`,
by `kernel_rfl`, the run's strings left as variables), so `plain` reads it member by member (`Concrete.plain_obj_of`).
-/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock Flock.Firewall
open FlockDeploy.Concrete (objT plain_obj_of noNl utf8_noTen findByte_eq plainStr_hexOf fromUTF8?_toByteArray)
open FlockSoundness.Discharge.Recursion.FirewallJson (parse_printJson)
open FlockSoundness.Refine
open Lean (Json)
open Std.DTreeMap.Internal (Impl)

/-- **A run's strings are plain**: each prints in the header as it parses back (`plainStr`), and together they are
under 2^20 characters, so the statement's every length fits its eight bytes (`FlockDeploy.Raw.Fits`). -/
structure Run.Plain (r : Run) : Prop where
  set : plainStr r.set = true
  bindI : plainStr r.bindI = true
  bindO : plainStr r.bindO = true
  size : r.set.length + r.bindI.length + r.bindO.length < 2 ^ 20

/-- A literal leaf, or a literal key, is plain. -/
macro "plain_lit" : tactic =>
  `(tactic| simp [plain, plainList, plainStr, String.all_bool_eq, jn, js, ja, v2, NAME, HmRow.SCHEME,
    HmRow.ROW_SCHEMA_V2, valueI, valueO])

/-! ## Objects of two ports -/

theorem io_obj (a b : Json) :
    Json.mkObj [("i0", a), ("o0", b)] = .obj ⟨⟨objT (Json.mkObj [("i0", a), ("o0", b)])⟩⟩ := by kernel_rfl
theorem io_model (a b : Json) : (objT (Json.mkObj [("i0", a), ("o0", b)])).toListModel = [⟨"i0", a⟩, ⟨"o0", b⟩] := by
  kernel_rfl
theorem io_asc (a b : Json) : ascending (objT (Json.mkObj [("i0", a), ("o0", b)])) = true := by kernel_rfl

theorem io_plain {a b : Json} (ha : plain a = true) (hb : plain b = true) :
    plain (Json.mkObj [("i0", a), ("o0", b)]) = true := by
  rw [io_obj]
  refine plain_obj_of (io_model a b) (io_asc a b) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, ha⟩, ⟨by plain_lit, hb⟩⟩

theorem io_str_plain {a b : String} (ha : plainStr a = true) (hb : plainStr b = true) :
    plain (Json.mkObj [("i0", .str a), ("o0", .str b)]) = true := io_plain ha hb

/-! ## The header -/

theorem fv3_obj (r : Run) : fv3J r = .obj ⟨⟨objT (fv3J r)⟩⟩ := by kernel_rfl
theorem fv3_model (r : Run) : (objT (fv3J r)).toListModel =
    [⟨"bindings", Json.mkObj [("i0", .str r.bindI), ("o0", .str r.bindO)]⟩,
     ⟨"domain_ids", Json.mkObj [("i0", .str (hexOf fdomI)), ("o0", .str (hexOf fdomO))]⟩,
     ⟨"frame", .str "frame-v3-sha512"⟩,
     ⟨"roots", Json.mkObj [("i0", .str (hexOf rootI)), ("o0", .str (hexOf rootO))]⟩,
     ⟨"row_key_sha512", .str keyHex⟩,
     ⟨"schemas", Json.mkObj [("i0", .str v2), ("o0", .str v2)]⟩] := by kernel_rfl
theorem fv3_asc (r : Run) : ascending (objT (fv3J r)) = true := by kernel_rfl

theorem fv3_plain {r : Run} (hr : r.Plain) : plain (fv3J r) = true := by
  rw [fv3_obj]
  refine plain_obj_of (fv3_model r) (fv3_asc r) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, io_str_plain hr.bindI hr.bindO⟩,
    ⟨by plain_lit, io_str_plain (plainStr_hexOf _) (plainStr_hexOf _)⟩, ⟨by plain_lit, by plain_lit⟩,
    ⟨by plain_lit, io_str_plain (plainStr_hexOf _) (plainStr_hexOf _)⟩, ⟨by plain_lit, plainStr_hexOf _⟩,
    ⟨by plain_lit, io_str_plain (by plain_lit) (by plain_lit)⟩⟩

theorem regHdr_obj : regHdrJ = .obj ⟨⟨objT regHdrJ⟩⟩ := by kernel_rfl
theorem regHdr_model : (objT regHdrJ).toListModel =
    [⟨"i0", Json.mkObj [("paths", .arr #[.str ""]), ("positions", .arr #[jn 0]), ("value", .str "i0")]⟩,
     ⟨"o0", Json.mkObj [("paths", .arr #[.str "", .str ""]), ("positions", .arr #[jn 0, jn 0]),
       ("value", .str "o0")]⟩] := by kernel_rfl
theorem regHdr_asc : ascending (objT regHdrJ) = true := by kernel_rfl

theorem pv_obj (p q : Json) (v : String) :
    Json.mkObj [("paths", p), ("positions", q), ("value", .str v)] =
      .obj ⟨⟨objT (Json.mkObj [("paths", p), ("positions", q), ("value", .str v)])⟩⟩ := by kernel_rfl
theorem pv_model (p q : Json) (v : String) :
    (objT (Json.mkObj [("paths", p), ("positions", q), ("value", .str v)])).toListModel =
      [⟨"paths", p⟩, ⟨"positions", q⟩, ⟨"value", .str v⟩] := by kernel_rfl
theorem pv_asc (p q : Json) (v : String) :
    ascending (objT (Json.mkObj [("paths", p), ("positions", q), ("value", .str v)])) = true := by kernel_rfl

theorem pv_plain {p q : Json} {v : String} (hp : plain p = true) (hq : plain q = true) (hv : plainStr v = true) :
    plain (Json.mkObj [("paths", p), ("positions", q), ("value", .str v)]) = true := by
  rw [pv_obj]
  refine plain_obj_of (pv_model p q v) (pv_asc p q v) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, hp⟩, ⟨by plain_lit, hq⟩, ⟨by plain_lit, hv⟩⟩

theorem regHdr_plain : plain regHdrJ = true := by
  rw [regHdr_obj]
  refine plain_obj_of regHdr_model regHdr_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, pv_plain (by plain_lit) (by plain_lit) (by plain_lit)⟩,
    ⟨by plain_lit, pv_plain (by plain_lit) (by plain_lit) (by plain_lit)⟩⟩

theorem units_obj (cls : String) : unitsJ cls = .obj ⟨⟨objT (unitsJ cls)⟩⟩ := by kernel_rfl
theorem units_model (cls : String) : (objT (unitsJ cls)).toListModel =
    [⟨"blocks", .arr #[.arr #[jn 0], .arr #[jn 1]]⟩, ⟨"classes", .arr #[.str cls, .str cls]⟩,
     ⟨"indices", .arr #[jn 0, jn 1]⟩, ⟨"units_per_vu", jn 1⟩, ⟨"vus_per_block", jn 1⟩] := by kernel_rfl
theorem units_asc (cls : String) : ascending (objT (unitsJ cls)) = true := by kernel_rfl

theorem units_plain {cls : String} (hk : plainStr cls = true) : plain (unitsJ cls) = true := by
  rw [units_obj]
  refine plain_obj_of (units_model cls) (units_asc cls) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, ?_⟩, ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, by plain_lit⟩,
    ⟨by plain_lit, by plain_lit⟩⟩
  simp [plain, plainList, hk]

theorem hdr_obj (r : Run) (circ cls : String) : hdrJ r circ cls = .obj ⟨⟨objT (hdrJ r circ cls)⟩⟩ := by kernel_rfl
theorem hdr_model (r : Run) (circ cls : String) : (objT (hdrJ r circ cls)).toListModel =
    [⟨"circuit_sha512", .str circ⟩, ⟨"format", .str "flock-circuit-inputs"⟩, ⟨"frame_v3", fv3J r⟩,
     ⟨"instances", jn 2⟩, ⟨"registered", regHdrJ⟩, ⟨"rows", .bool false⟩, ⟨"scheme", .str HmRow.SCHEME⟩,
     ⟨"set", .str r.set⟩, ⟨"shared_rows", Json.mkObj [("i0", jn 1), ("o0", jn 2)]⟩, ⟨"statement", .str NAME⟩,
     ⟨"units", unitsJ cls⟩] := by kernel_rfl
theorem hdr_asc (r : Run) (circ cls : String) : ascending (objT (hdrJ r circ cls)) = true := by kernel_rfl

/-- **The header is plain**, at plain run strings and digests. -/
theorem hdr_plain {r : Run} (hr : r.Plain) {circ cls : String} (hc : plainStr circ = true)
    (hk : plainStr cls = true) : plain (hdrJ r circ cls) = true := by
  rw [hdr_obj]
  refine plain_obj_of (hdr_model r circ cls) (hdr_asc r circ cls) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, hc⟩, ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, fv3_plain hr⟩,
    ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, regHdr_plain⟩, ⟨by plain_lit, by plain_lit⟩,
    ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, hr.set⟩, ⟨by plain_lit, io_plain (by plain_lit) (by plain_lit)⟩,
    ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, units_plain hk⟩⟩

theorem fjnorm_hdr (r : Run) (circ cls : String) :
    FlockSoundness.Discharge.Recursion.FirewallJson.norm (hdrJ r circ cls) = hdrJ r circ cls := by kernel_rfl
theorem norm_hdr (r : Run) (circ cls : String) : Flock.norm (hdrJ r circ cls) = hdrJ r circ cls := by kernel_rfl

theorem hdr_noTen {r : Run} (hr : r.Plain) {circ cls : String} (hc : plainStr circ = true)
    (hk : plainStr cls = true) : (10 : UInt8) ∉ (printJson (hdrJ r circ cls)).toUTF8.data.toList := by
  rw [printJson]
  exact utf8_noTen _ (noNl _ (hdr_plain hr hc hk))

/-- A file's first newline, after bytes holding none, ends them. -/
theorem findByte_nl (A B : ByteArray) (hA : (10 : UInt8) ∉ A.data.toList) :
    findByte (A ++ ("\n".toUTF8 ++ B)) 10 = some A.size := by
  have hn : List.findIdx? (· == (10 : UInt8)) A.data.toList = none :=
    List.findIdx?_eq_none_iff.2 fun x hx => by
      cases hx10 : (x == 10)
      · rfl
      · exact absurd ((beq_iff_eq.1 hx10) ▸ hx) hA
  have h10 : "\n".toUTF8.data.toList = [10] := by decide +kernel
  rw [findByte_eq, ByteArray.data_append, ByteArray.data_append, Array.toList_append, Array.toList_append, h10,
    List.drop_zero, List.findIdx?_append, hn, Option.none_or, List.cons_append, List.findIdx?_cons, beq_self_eq_true,
    ite_eq_left rfl, Option.map_some, Option.map_some, Nat.zero_add, Nat.add_zero, Array.length_toList]
  rfl

/-- The public file's first newline ends the header's text. -/
theorem findByte_pub {r : Run} (hr : r.Plain) {circ cls : String} (hc : plainStr circ = true)
    (hk : plainStr cls = true) : findByte (pubOf r circ cls) 10 = some (printJson (hdrJ r circ cls)).toUTF8.size := by
  rw [pubOf, tailB]
  exact findByte_nl _ _ (hdr_noTen hr hc hk)

theorem parse_hdr {r : Run} (hr : r.Plain) {circ cls : String} (hc : plainStr circ = true)
    (hk : plainStr cls = true) : JsonParse.parse (printJson (hdrJ r circ cls)) = .ok (hdrJ r circ cls) := by
  rw [parse_printJson (hdr_plain hr hc hk), fjnorm_hdr]

theorem fromUTF8?_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  rw [String.toUTF8_eq_toByteArray, fromUTF8?_toByteArray]

/-- A file whose first line is the text `s`, which holds no newline, reads back as `s` parses. -/
theorem headerOf_nl (s : String) (B : ByteArray) (j : Json) (hs : (10 : UInt8) ∉ s.toUTF8.data.toList)
    (hp : JsonParse.parse s = .ok j) :
    (do
      let nl ← findByte (s.toUTF8 ++ ("\n".toUTF8 ++ B)) 10
      (JsonParse.parse (← String.fromUTF8? ((s.toUTF8 ++ ("\n".toUTF8 ++ B)).extract 0 nl))).toOption) =
      some j := by
  rw [findByte_nl _ _ hs]
  show (String.fromUTF8? ((s.toUTF8 ++ ("\n".toUTF8 ++ B)).extract 0 s.toUTF8.size)).bind
    (fun t => (JsonParse.parse t).toOption) = some j
  rw [ByteArray.extract_append_eq_left rfl, fromUTF8?_toUTF8]
  show (JsonParse.parse s).toOption = some j
  rw [hp]
  rfl

/-- **The public file's header reads back**: its first line, as UTF-8, parses to `hdrJ r circ cls`, as SPOT's
`headerOf` and Flock's `Public.load` read a public file. -/
theorem headerLine {r : Run} (hr : r.Plain) {circ cls : String} (hc : plainStr circ = true)
    (hk : plainStr cls = true) :
    (do
      let nl ← findByte (pubOf r circ cls) 10
      (JsonParse.parse (← String.fromUTF8? ((pubOf r circ cls).extract 0 nl))).toOption) =
      some (hdrJ r circ cls) := by
  rw [pubOf, tailB]
  exact headerOf_nl _ _ _ (hdr_noTen hr hc hk) (parse_hdr hr hc hk)

/-! ## The sample and the registered reads -/

theorem draw_obj : drawJ = .obj ⟨⟨objT drawJ⟩⟩ := by kernel_rfl
theorem draw_model : (objT drawJ).toListModel =
    [⟨"k", jn 1⟩, ⟨"law", .str "subset"⟩, ⟨"population", jn 2⟩, ⟨"units", .arr #[jn 0]⟩] := by kernel_rfl
theorem draw_asc : ascending (objT drawJ) = true := by kernel_rfl

theorem draw_plain : plain drawJ = true := by
  rw [draw_obj]
  refine plain_obj_of draw_model draw_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, by plain_lit⟩,
    ⟨by plain_lit, by plain_lit⟩⟩

theorem fjnorm_draw : FlockSoundness.Discharge.Recursion.FirewallJson.norm drawJ = drawJ := by kernel_rfl
theorem norm_draw : Flock.norm drawJ = drawJ := by kernel_rfl

theorem port_obj (value : String) (dom bc : ByteArray) (words : Nat) :
    regPortJ value dom bc words = .obj ⟨⟨objT (regPortJ value dom bc words)⟩⟩ := by kernel_rfl
theorem port_model (value : String) (dom bc : ByteArray) (words : Nat) :
    (objT (regPortJ value dom bc words)).toListModel =
      [⟨"domain", .str (hexOf dom)⟩, ⟨"leaves", jn 1⟩, ⟨"reads", .arr #[jn 0, jn 0]⟩,
       ⟨"root", .str (hexOf (regRoot dom bc))⟩, ⟨"row_schema", .str v2⟩, ⟨"value", .str value⟩,
       ⟨"words", jn words⟩] := by kernel_rfl
theorem port_asc (value : String) (dom bc : ByteArray) (words : Nat) :
    ascending (objT (regPortJ value dom bc words)) = true := by kernel_rfl

theorem port_plain {value : String} (hv : plainStr value = true) (dom bc : ByteArray) (words : Nat) :
    plain (regPortJ value dom bc words) = true := by
  rw [port_obj]
  refine plain_obj_of (port_model value dom bc words) (port_asc value dom bc words) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, plainStr_hexOf _⟩, ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, by plain_lit⟩,
    ⟨by plain_lit, plainStr_hexOf _⟩, ⟨by plain_lit, by plain_lit⟩, ⟨by plain_lit, hv⟩, ⟨by plain_lit, by plain_lit⟩⟩

theorem reg_obj : regJ = .obj ⟨⟨objT regJ⟩⟩ := by kernel_rfl
theorem reg_model : (objT regJ).toListModel =
    [⟨"format", .str "verity/registered-reads/v1"⟩,
     ⟨"ports", Json.mkObj [("i0", regPortJ valueI domI bcI 64), ("o0", regPortJ valueO domO bcO 1)]⟩] := by kernel_rfl
theorem reg_asc : ascending (objT regJ) = true := by kernel_rfl

theorem reg_plain : plain regJ = true := by
  rw [reg_obj]
  refine plain_obj_of reg_model reg_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨by plain_lit, by plain_lit⟩,
    ⟨by plain_lit, io_plain (port_plain (by plain_lit) _ _ _) (port_plain (by plain_lit) _ _ _)⟩⟩

theorem fjnorm_reg : FlockSoundness.Discharge.Recursion.FirewallJson.norm regJ = regJ := by kernel_rfl
theorem norm_reg : Flock.norm regJ = regJ := by kernel_rfl

theorem jsonOf_print {j : Json} (h : plain j = true) (hn : FlockSoundness.Discharge.Recursion.FirewallJson.norm j = j)
    (what : String) : FlockDeploy.jsonOf what (printJson j).toUTF8 = .ok j := by
  unfold FlockDeploy.jsonOf
  rw [fromUTF8?_toUTF8]
  dsimp only
  rw [parse_printJson h, hn]
  rfl

/-- **The sample file reads back** to `drawJ`. -/
theorem jsonOf_draw (what : String) : FlockDeploy.jsonOf what (printJson drawJ).toUTF8 = .ok drawJ :=
  jsonOf_print draw_plain fjnorm_draw what

/-- **The registered-reads file reads back** to `regJ`. -/
theorem jsonOf_reg (what : String) : FlockDeploy.jsonOf what (printJson regJ).toUTF8 = .ok regJ :=
  jsonOf_print reg_plain fjnorm_reg what

end SpotFlock.InstanceStmt
