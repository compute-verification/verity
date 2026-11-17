module

public import Proofs.Flock.SessionLoop.Concrete.File
public import Proofs.Flock.Claims.DeployInputs

/-! The two-instance reference statement's bytes as Flock's `--zk` verifier reads them (`InstanceStmt.x`); the fixed
nets are parameters, which the kernel never builds. -/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock Flock.ZkServer RecDeploy.VStarNets
open Lean (Json)

structure Run where
  set : String
  bindI : String
  bindO : String

def jn (n : Nat) : Json := .num ⟨n, 0⟩
def js (s : String) : Json := .str s
def ja (l : List Json) : Json := .arr l.toArray

def t0 : Tags := Tags.circuit
def tz : Tags := Zk.tags t0 14 1

abbrev coinsX : Coins := .tree (Zk.coinSpec (t0.tableNames 1))
def NAME : String := "verity/flock-circuit"

def i0 : Port := { name := "i0", words := 64, chunks := 1, nb := 0, bits := 1024, v2 := true }
def o0 : Port := { name := "o0", words := 1, chunks := 0, nb := 0, bits := 16, v2 := true }

def rowJ (p : Port) : Json := Json.mkObj [("midstate", .arr ((HmRow.midstateOf p).map fun w => js (HmRow.hex16 w))),
  ("pad", js (hexOf (HmRow.tailOf p))), ("prefix", js (hexOf (HmRow.prefixOf p)))]

def keyHex : String := hexOf (Sha512.hash Hm96.Default512.key)

def bcI : ByteArray := HmRow.zeroRowBc i0
def bcO : ByteArray := HmRow.zeroRowBc o0

def progJ : Json := Json.mkObj [("annotations", Json.mkObj []), ("callees", ja [js "T_v1"]),
  ("definitions", Json.mkObj [
    ("Root[Two_v1]_v1", Json.mkObj [("bindings", Json.mkObj []),
      ("body", Json.mkObj [
        ("nodes", ja [Json.mkObj [("args", ja []), ("axes", ja []), ("fn", jn 0), ("form", js "batch"), ("n", jn 2),
          ("out", jn 1)]]),
        ("params", ja []), ("ret", Json.mkObj [("affine", ja [js "n", jn 0, jn 0, jn 2])]),
        ("ret_type", ja [js "Array", jn 2, ja [js "Value", jn 16]])]),
      ("form", js "composite"), ("id", js "Root[Two_v1]_v1")]),
    ("T_v1", Json.mkObj [("form", js "primitive"), ("id", js "T_v1"), ("params", ja []),
      ("ret", ja [js "Value", jn 16])])]),
  ("program_fn", js "Two_v1"), ("root", js "Root[Two_v1]_v1"), ("schema", js "verity-ir/descriptor/v1"),
  ("types", ja [ja [js "Value", jn 16], ja [js "Array", jn 2, ja [js "Value", jn 16]]])]
def progText : String := printJson progJ
def progBytes : ByteArray := progText.toUTF8

/-- A literal, because `canon` writes the canonical text by well-founded recursion that the kernel does not unfold
(`progHex_eq`). -/
def progHex : String :=
  "b1b2443c99170e8c52e7011e9041ea4dccc964c5135ccc6be432941d5427066efe02d5810606ecd3ef7a9241c39bc3236b7b077d809a3a42e9bb68aa67fe11c8"

def partObj : Json := Json.mkObj [("format", js HmRow.PARTITION_V1), ("program", js progHex),
  ("query", Json.mkObj [("name", js "Q_template_instance"), ("params", Json.mkObj [("template", js "T_v1")]),
    ("version", jn 0)])]

/-- A literal, as `progHex` is (`partHex_eq`). -/
def partHex : String :=
  "1e5de9f6dc7650c0f0a67091651cacf652fa9c56491a4f71852b5f71629454f76412e63e5a988af4233ca983ec2bc0cc14b70f79b42d026b536e29b57badbb85"

def rng (net : String) (sl p0 n : Nat) : Json := Json.mkObj [("count", jn n), ("net", js net), ("pos0", jn p0),
  ("slot_log", jn sl), ("type", js "unit")]
def mrng (sl p0 n : Nat) : Json :=
  Json.mkObj [("count", jn n), ("pos0", jn p0), ("slot_log", jn sl), ("type", js "mask")]

def lsM : Json := Json.mkObj [("id", js "hm96-sha512/v1"), ("inner", js "sha512 column digest"),
  ("key", js "default_key, a statement constant for every tree"), ("key_sha512", js keyHex), ("salt_bytes", jn 192),
  ("salt_source", js "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored")]

/-- Every object lists its members in key order, the order `JsonParse` inserts them in, so that its line parses back
to this tree (`parse_meta`). -/
def metaJ : Json := Json.mkObj [
  ("dummy", Json.mkObj [("bc", ja [js (hexOf bcI), js (hexOf bcO)]), ("outputs", ja [jn 0])]),
  ("format", js "flock-circuit"), ("input_ports", jn 1), ("k_log", jn 22), ("leaf_scheme", lsM),
  ("leaves_in", ja [ja [jn 0]]), ("leaves_out", ja [ja [jn 0]]), ("mask_log", jn 14),
  ("out_ports", ja [ja [js "o0", jn 1]]),
  ("partition", Json.mkObj [("digest", js partHex), ("rule", js HmRow.PARTITION_V1)]),
  ("per_vu", Json.mkObj [("hm96", jn 2), ("sha512x3", jn 1), ("unit", jn 1)]),
  ("ports", ja [ja [js "i0", jn 64, jn 1], ja [js "o0", jn 1, jn 0]]),
  ("program_digests", Json.mkObj [("T_v1", js progHex)]), ("program_sha512", js progHex),
  ("ranges", ja [rng "hm96" 18 0 2, rng "sha512x3" 18 2 1, rng "unit" 9 1536 1, mrng 14 64 1]),
  ("row_key_sha512", js keyHex),
  ("row_schemas", ja [ja [js HmRow.ROW_SCHEMA_V2, jn 1024], ja [js HmRow.ROW_SCHEMA_V2, jn 16]]),
  ("rows", ja [rowJ i0, rowJ o0]), ("scheme", js HmRow.SCHEME), ("statement", js NAME),
  ("units_per_vu", jn 1), ("vus_per_block", jn 1),
  ("wires", ja [ja [js "hm96", jn 0, jn 0, js "sha512x3", jn 0, jn 0]])]
def metaText : String := printJson metaJ

def lines (ls lh : HmNets.Laid) : List String := fileLs "copy16" metaText FlockDeploy.Concrete.unitLaid ls lh
def bytesOf (ls lh : HmNets.Laid) : ByteArray := (joinNl (lines ls lh)).toUTF8

def classOfB (ls lh : HmNets.Laid) : ByteArray :=
  (HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ).toOption.getD default

def circuitHash : ByteArray := Sha512.hash (bytesOf (HmNets.shaLaid ()) (HmNets.hmLaid ()))
def classHash : ByteArray := classOfB (HmNets.shaLaid ()) (HmNets.hmLaid ())

def v2 : String := HmRow.ROW_SCHEMA_V2

def fLeaf (dom : ByteArray) (r : Nat) (bc : ByteArray) : ByteArray := FV3.leaf dom r v2 (Hm96.Default512.treeLeaf bc)

def fdomI : ByteArray := ⟨#[0]⟩
def fdomO : ByteArray := ⟨#[1]⟩
def rootI : ByteArray := FV3.root fdomI #[fLeaf fdomI 0 bcI]
def rootO : ByteArray := FV3.root fdomO #[fLeaf fdomO 0 bcO, fLeaf fdomO 1 bcO]

def fv3J (r : Run) : Json := Json.mkObj [("bindings", Json.mkObj [("i0", js r.bindI), ("o0", js r.bindO)]),
  ("domain_ids", Json.mkObj [("i0", js (hexOf fdomI)), ("o0", js (hexOf fdomO))]), ("frame", js "frame-v3-sha512"),
  ("roots", Json.mkObj [("i0", js (hexOf rootI)), ("o0", js (hexOf rootO))]), ("row_key_sha512", js keyHex),
  ("schemas", Json.mkObj [("i0", js v2), ("o0", js v2)])]

def valueI : String := "i0"
def valueO : String := "o0"

def domI : ByteArray := ⟨Array.replicate 64 2⟩
def domO : ByteArray := ⟨Array.replicate 64 3⟩

def regHdrJ : Json := Json.mkObj [
  ("i0", Json.mkObj [("paths", ja [js ""]), ("positions", ja [jn 0]), ("value", js valueI)]),
  ("o0", Json.mkObj [("paths", ja [js "", js ""]), ("positions", ja [jn 0, jn 0]), ("value", js valueO)])]

def unitsJ (cls : String) : Json := Json.mkObj [("blocks", ja [ja [jn 0], ja [jn 1]]), ("classes", ja [js cls, js cls]),
  ("indices", ja [jn 0, jn 1]), ("units_per_vu", jn 1), ("vus_per_block", jn 1)]

def hdrJ (r : Run) (circ cls : String) : Json := Json.mkObj [("circuit_sha512", js circ),
  ("format", js "flock-circuit-inputs"), ("frame_v3", fv3J r), ("instances", jn 2), ("registered", regHdrJ),
  ("rows", .bool false), ("scheme", js HmRow.SCHEME), ("set", js r.set),
  ("shared_rows", Json.mkObj [("i0", jn 1), ("o0", jn 2)]), ("statement", js NAME), ("units", unitsJ cls)]

def u32 (n : Nat) : ByteArray := ⟨#[n.toUInt8, 0, 0, 0]⟩

def bodyB : ByteArray := bcI ++ bcO ++ bcO ++ u32 0 ++ u32 0 ++ u32 0 ++ u32 1
def tailB : ByteArray := "\n".toUTF8 ++ bodyB
def pubOf (r : Run) (circ cls : String) : ByteArray := (printJson (hdrJ r circ cls)).toUTF8 ++ tailB

def sampleJ : Json := Json.mkObj [("k", jn 1), ("law", js "subset"), ("population", jn 2), ("units", ja [jn 0])]

def hdrD (r : Run) (c : Circuit) (circ cls : String) : Json :=
  ((((hdrJ r circ cls).setObjVal! "unit_draw" sampleJ).setObjVal! "instances" (Lean.toJson (1 : Nat))).setObjVal!
    "population" (Lean.toJson (2 : Nat))).setObjVal! "units" (HmRow.unitsEntry { c with outPorts := #[] } #[0])

def regRoot (dom bc : ByteArray) : ByteArray := fLeaf dom 0 bc

def regPortJ (value : String) (dom bc : ByteArray) (words : Nat) : Json := Json.mkObj [("domain", js (hexOf dom)),
  ("leaves", jn 1), ("reads", ja [jn 0, jn 0]), ("root", js (hexOf (regRoot dom bc))), ("row_schema", js v2),
  ("value", js value), ("words", jn words)]

def regJ : Json := Json.mkObj [("format", js "verity/registered-reads/v1"), ("ports", Json.mkObj [
  ("i0", regPortJ valueI domI bcI 64), ("o0", regPortJ valueO domO bcO 1)])]

def rawOf (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) : FlockDeploy.Raw :=
  { statement := NAME, circuit := bytesOf ls lh, publicFile := pubOf r (hexOf H) (hexOf C), tables := [], zk := true,
    osCoins := false, count := 1, draw := some (printJson sampleJ).toUTF8, partition := some (printJson partObj).toUTF8,
    program := some progBytes, work := none, workTable := none, stratified := none, closure := none,
    registered := some (printJson regJ).toUTF8, publicInputs := none }

def raw (r : Run) : FlockDeploy.Raw := rawOf r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash

def x (r : Run) : Claims.Msg := (raw r).encode

end SpotFlock.InstanceStmt
