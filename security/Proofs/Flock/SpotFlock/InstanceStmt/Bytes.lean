module

public import Proofs.Flock.SessionLoop.Concrete.File
public import Proofs.Flock.Claims.DeployInputs

/-! # The two-instance reference statement's bytes

`InstanceStmt.x r` is the session statement of a two-instance run `r` (`Run`) as Flock's `--zk` verifier reads one
(`FlockDeploy.Raw`): the statement `verity/flock-circuit` (`Tags.circuit`: hidden outputs, bindings, shared rows) on one
table, of

* the circuit file (`bytesOf`): the unit `copy16` and the verifier's two fixed nets, which are parameters
  (`FlockDeploy.Concrete.unitLaid`, `fileLs`), under META (`metaJ`): the input row `i0` of 1,024 bits and the hidden
  output row `o0` of 16, each VU's two hm96 slots and one `sha512x3` slot, bound to the program and partition below;
* the public file (`pubOf`): two instances reading `i0`'s one shared row, each its own `o0` row, every value and salt
  zero, with the run's input set and its ports' bindings to the run;
* the program (`progText`): its root a batch of two members of the primitive `T_v1`, with `T_v1`'s Boolean Definition,
  partitioned by `Q_template_instance` v0 at `T_v1` (`partObj`);
* the sample (`sampleJ`): unit 0 of the two, and the header the verifier writes for the sampled statement (`hdrD`).

The statement reads no registered value. The circuit file's SHA-512 (`circuitHash`) and its class (`classHash`, the file with META's bindings removed, as `HmRow.classOf` hashes it)
are the only hashes of the file; nothing here asks the kernel to build the fixed nets.
-/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock Flock.ZkServer RecDeploy.VStarNets
open Lean (Json)

/-- **What a run fixes in its reference statement**: the input set its session reads (`set`) and its ports' bindings to
the run (`bindI`, `bindO`), each written as the run's record writes it. -/
structure Run where
  set : String
  bindI : String
  bindO : String

def jn (n : Nat) : Json := .num ⟨n, 0⟩
def js (s : String) : Json := .str s
def ja (l : List Json) : Json := .arr l.toArray

/-! ## The circuit file -/

def t0 : Tags := Tags.circuit
def tz : Tags := Zk.tags t0 14 1

/-- The coins `Read.setup` hands the statement: the keyed coin tree of its one table. -/
abbrev coinsX : Coins := .tree (Zk.coinSpec (t0.tableNames 1))
def NAME : String := "verity/flock-circuit"

/-- The input row: 1,024 bits, one compression before its last block. -/
def i0 : Port := { name := "i0", words := 64, chunks := 1, nb := 0, bits := 1024, v2 := true }
/-- The hidden output row: 16 bits, its first block its last. -/
def o0 : Port := { name := "o0", words := 1, chunks := 0, nb := 0, bits := 16, v2 := true }

/-- A row's constants as META lists them, which `HmRow.rowPorts` recomputes. -/
def rowJ (p : Port) : Json := Json.mkObj [("midstate", .arr ((HmRow.midstateOf p).map fun w => js (HmRow.hex16 w))),
  ("pad", js (hexOf (HmRow.tailOf p))), ("prefix", js (hexOf (HmRow.prefixOf p)))]

def keyHex : String := hexOf (Sha512.hash Hm96.Default512.key)

/-- The zero row's `b ‖ c` under a zero salt: the padding instance's row and every instance's. -/
def bcI : ByteArray := HmRow.zeroRowBc i0
def bcO : ByteArray := HmRow.zeroRowBc o0

/-- The program: its root a batch of two members of the primitive `T_v1`, whose Boolean Definition is `StandIn[T]_v1`,
sixteen constant bits. -/
def progJ : Json := Json.mkObj [("annotations", Json.mkObj []), ("boolean", Json.mkObj [("T_v1", js "StandIn[T]_v1")]),
  ("callees", ja [js "T_v1", js "Const1[0x0]_v1"]),
  ("definitions", Json.mkObj [
    ("Const1[0x0]_v1", Json.mkObj [("form", js "primitive"), ("id", js "Const1[0x0]_v1"), ("params", ja []),
      ("ret", ja [js "Value", jn 1])]),
    ("Root[Two_v1]_v1", Json.mkObj [("bindings", Json.mkObj []),
      ("body", Json.mkObj [
        ("nodes", ja [Json.mkObj [("args", ja []), ("axes", ja []), ("fn", jn 0), ("form", js "batch"), ("n", jn 2),
          ("out", jn 1)]]),
        ("params", ja []), ("ret", Json.mkObj [("affine", ja [js "n", jn 0, jn 0, jn 2])]),
        ("ret_type", ja [js "Array", jn 2, ja [js "Value", jn 16]])]),
      ("form", js "composite"), ("id", js "Root[Two_v1]_v1")]),
    ("StandIn[T]_v1", Json.mkObj [("bindings", Json.mkObj []),
      ("body", Json.mkObj [
        ("nodes", ja [Json.mkObj [("args", ja []), ("axes", ja []), ("fn", jn 1), ("form", js "batch"), ("n", jn 16),
          ("out", jn 2)]]),
        ("params", ja []), ("ret", Json.mkObj [("affine", ja [js "n", jn 0, jn 0, jn 16])]),
        ("ret_type", ja [js "Array", jn 16, ja [js "Value", jn 1]])]),
      ("form", js "composite"), ("id", js "StandIn[T]_v1")]),
    ("T_v1", Json.mkObj [("form", js "primitive"), ("id", js "T_v1"), ("params", ja []),
      ("ret", ja [js "Value", jn 16])])]),
  ("program_fn", js "Two_v1"), ("root", js "Root[Two_v1]_v1"), ("schema", js "verity-ir/descriptor/v2"),
  ("types", ja [ja [js "Value", jn 16], ja [js "Array", jn 2, ja [js "Value", jn 16]],
    ja [js "Array", jn 16, ja [js "Value", jn 1]]])]
def progText : String := printJson progJ
def progBytes : ByteArray := progText.toUTF8

/-- The program's SHA-512 in hex, as `TemplateQuery.Desc.parse` computes it (`progHex_eq`). A literal: the digest is of
the program's canonical text, which `canon` writes by well-founded recursion that the kernel does not unfold. -/
def progHex : String :=
  "2cc6b665ee6d848f77f66d8572ffc7907377f2d6085e4e019d605a0a913eab90d1347b9bdc575ae280590f46c68c28d8e0d32cd28febbb6c40c7827a1faca1c8"

/-- The partition: `Q_template_instance` v0 at `T_v1`, over the program. -/
def partObj : Json := Json.mkObj [("format", js HmRow.PARTITION_V1), ("program", js progHex),
  ("query", Json.mkObj [("name", js "Q_template_instance"), ("params", Json.mkObj [("template", js "T_v1")]),
    ("version", jn 0)])]

/-- The partition's digest in hex, as `HmRow.checkPartition` computes it (`partHex_eq`); a literal, as `progHex` is. -/
def partHex : String :=
  "431509a8a173e2c98eb5a8fc938365ccc826af926f325c686614fe5e2cdee3367fdcf633706d0a1a4891ab054b129d7aa1d9dadfe6b0af825a30ed4e48b2e410"

def rng (net : String) (sl p0 n : Nat) : Json := Json.mkObj [("count", jn n), ("net", js net), ("pos0", jn p0),
  ("slot_log", jn sl), ("type", js "unit")]
def mrng (sl p0 n : Nat) : Json :=
  Json.mkObj [("count", jn n), ("pos0", jn p0), ("slot_log", jn sl), ("type", js "mask")]

/-- The leaf scheme `Tags.circuit` pins (`Tags.leafSchemeHm96Os`), its members in key order. -/
def lsM : Json := Json.mkObj [("id", js "hm96-sha512/v1"), ("inner", js "sha512 column digest"),
  ("key", js "default_key, a statement constant for every tree"), ("key_sha512", js keyHex), ("salt_bytes", jn 192),
  ("salt_source", js "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored")]

/-- META. Every object in it lists its members in key order, the order `JsonParse` inserts them in, so that its line
parses back to this tree (`parse_meta`): a tree's shape depends on the order its keys went in. Its bindings
(`HmRow.BINDING_KEYS`) are the partition, the program's digest and its one template's; the circuit's class hashes the
rest. -/
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

/-- The circuit's class as `HmRow.parse` computes it (`HmRow.classOf`): SHA-512 of the file with META's bindings
removed. -/
def classOfB (ls lh : HmNets.Laid) : ByteArray :=
  (HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ).toOption.getD default

def circuitHash : ByteArray := Sha512.hash (bytesOf (HmNets.shaLaid ()) (HmNets.hmLaid ()))
def classHash : ByteArray := classOfB (HmNets.shaLaid ()) (HmNets.hmLaid ())

/-! ## The public file -/

def v2 : String := HmRow.ROW_SCHEMA_V2

/-- A row's frame-v3-sha512 leaf at position `r` under domain `dom`. -/
def fLeaf (dom : ByteArray) (r : Nat) (bc : ByteArray) : ByteArray := FV3.leaf dom r v2 (Hm96.Default512.treeLeaf bc)

def fdomI : ByteArray := ⟨#[0]⟩
def fdomO : ByteArray := ⟨#[1]⟩
def rootI : ByteArray := FV3.root fdomI #[fLeaf fdomI 0 bcI]
def rootO : ByteArray := FV3.root fdomO #[fLeaf fdomO 0 bcO, fLeaf fdomO 1 bcO]

def fv3J (r : Run) : Json := Json.mkObj [("bindings", Json.mkObj [("i0", js r.bindI), ("o0", js r.bindO)]),
  ("domain_ids", Json.mkObj [("i0", js (hexOf fdomI)), ("o0", js (hexOf fdomO))]), ("frame", js "frame-v3-sha512"),
  ("roots", Json.mkObj [("i0", js (hexOf rootI)), ("o0", js (hexOf rootO))]), ("row_key_sha512", js keyHex),
  ("schemas", Json.mkObj [("i0", js v2), ("o0", js v2)])]

/-- The header's `units` (`HmRow.unitsEntry` at instances `0` and `1`), each instance of class `cls`. -/
def unitsJ (cls : String) : Json := Json.mkObj [("blocks", ja [ja [jn 0], ja [jn 1]]), ("classes", ja [js cls, js cls]),
  ("indices", ja [jn 0, jn 1]), ("units_per_vu", jn 1), ("vus_per_block", jn 1)]

def hdrJ (r : Run) (circ cls : String) : Json := Json.mkObj [("circuit_sha512", js circ),
  ("format", js "flock-circuit-inputs"), ("frame_v3", fv3J r), ("instances", jn 2),
  ("rows", .bool false), ("scheme", js HmRow.SCHEME), ("set", js r.set),
  ("shared_rows", Json.mkObj [("i0", jn 1), ("o0", jn 2)]), ("statement", js NAME), ("units", unitsJ cls)]

/-- A row reference, u32 little-endian. -/
def u32 (n : Nat) : ByteArray := ⟨#[n.toUInt8, 0, 0, 0]⟩

/-- The rows' `b ‖ c` (`i0`'s one row, then `o0`'s two), then each instance's references to them. -/
def bodyB : ByteArray := bcI ++ bcO ++ bcO ++ u32 0 ++ u32 0 ++ u32 0 ++ u32 1
def tailB : ByteArray := "\n".toUTF8 ++ bodyB
def pubOf (r : Run) (circ cls : String) : ByteArray := (printJson (hdrJ r circ cls)).toUTF8 ++ tailB

/-! ## The sample -/

def sampleJ : Json := Json.mkObj [("k", jn 1), ("law", js "subset"), ("population", jn 2), ("units", ja [jn 0])]

/-- The sampled statement's header, as `HmRow.drawn` writes it at `sampleJ`: the population's, with the sample, one instance
of the two, and the sampled instance's units. -/
def hdrD (r : Run) (c : Circuit) (circ cls : String) : Json :=
  ((((hdrJ r circ cls).setObjVal! "unit_draw" sampleJ).setObjVal! "instances" (Lean.toJson (1 : Nat))).setObjVal!
    "population" (Lean.toJson (2 : Nat))).setObjVal! "units" (HmRow.unitsEntry { c with outPorts := #[] } #[0])

/-! ## The statement -/

/-- The statement at the fixed nets' layouts `ls`, `lh`, the circuit file's SHA-512 `H` and its class `C`. -/
def rawOf (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) : FlockDeploy.Raw :=
  { statement := NAME, circuit := bytesOf ls lh, publicFile := pubOf r (hexOf H) (hexOf C), tables := [], zk := true,
    osCoins := false, count := 1, draw := some (printJson sampleJ).toUTF8, partition := some (printJson partObj).toUTF8,
    program := some progBytes, work := none, workTable := none, stratified := none, closure := none,
    registered := none, publicInputs := none }

def raw (r : Run) : FlockDeploy.Raw := rawOf r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash

/-- **The two-instance reference statement** of run `r`. -/
def x (r : Run) : Hash.Msg := (raw r).encode

end SpotFlock.InstanceStmt
