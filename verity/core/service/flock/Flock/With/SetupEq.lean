/-
Setup's oracle copies under the deployed hashes: run in `V` with `Hashes.ofSha512 Sha512.hash`, each `…With` of
`Flock.With.Setup` is the compiled verifier's function it copies, so `Stmt.setupTablesWith` at SHA-512 is `Stmt.setupTables`.
-/
import Flock.With.Setup
import Flock.With.Eq

namespace Flock

open Lean (Json)

namespace With

set_option maxRecDepth 20000 in
theorem hmClassOfWith_std (bytes : ByteArray) (head : String) (mj : Json) :
    HmRow.classOfWith std bytes head mj = HmRow.classOf bytes head mj := rfl

theorem typeDigestWith_std (t : CircuitType) : t.digestWith std = t.digest := rfl

theorem layoutDigestWith_std (l : Layout) : l.digestWith std = l.digest := rfl

set_option maxRecDepth 20000 in
theorem classOfWith_std (mj : Json) (tables : List (String × ByteArray)) :
    Typed.classOfWith std mj tables = Typed.classOf mj tables := rfl

set_option maxRecDepth 20000 in
theorem templateOfWith_std (name : String) (done : Array DeriveAll.D) (ls : Array Layout) (unit : Nat)
    (t : CircuitType) (held : Array Typed.Held) :
    Typed.templateOfWith std name done ls unit t held = Typed.templateOf name done ls unit t held := by
  unfold Typed.templateOfWith Typed.templateOf
  simp only [layoutDigestWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem readWith_std (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    Typed.readWith std tags bytes tables = Typed.read tags bytes tables := by
  unfold Typed.readWith Typed.read
  simp only [classOfWith_std, templateOfWith_std]
  try rfl

theorem descParseWith_std (bytes : ByteArray) :
    TemplateQuery.Desc.parseWith std.sha512 bytes = TemplateQuery.Desc.parse bytes := rfl

set_option maxRecDepth 20000 in
theorem programDecodeWith_std (bytes : ByteArray) : Program.decodeWith std.sha512 bytes = Program.decode bytes := rfl

theorem partitionDigestWith_std (obj : Json) : HmRow.partitionDigestWith std obj = HmRow.partitionDigest obj := rfl

theorem publicDigestWith_std (header : Json) (pubs : ByteArray) :
    HmRow.publicDigestWith std header pubs = HmRow.publicDigest header pubs := rfl

set_option maxRecDepth 20000 in
-- `rfl` compares the two whole bodies of `parse`, a few hundred binds each
set_option maxHeartbeats 1000000 in
theorem parseWith_std (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) (tmpl : Option Typed.Template)
    (pre : Array (String × Net)) : HmRow.parseWith std tags bytes tables tmpl pre = HmRow.parse tags bytes tables tmpl pre := by
  unfold HmRow.parseWith HmRow.parse HmRow.parseRest
  simp -zeta only [std_sha512, pure_bind, hmClassOfWith_std, monadLift_self]
  rfl

set_option maxRecDepth 20000 in
theorem parseTypedWith_std (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    HmRow.parseTypedWith std tags bytes tables = HmRow.parseTyped tags bytes tables := by
  unfold HmRow.parseTypedWith HmRow.parseTyped
  simp only [readWith_std, parseWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem deriveTemplateUnitsWith_std (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    HmRow.deriveTemplateUnitsWith std c obj program indices = HmRow.deriveTemplateUnits c obj program indices := by
  unfold HmRow.deriveTemplateUnitsWith HmRow.deriveTemplateUnits
  simp only [descParseWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem deriveQwordUnitsWith_std (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    HmRow.deriveQwordUnitsWith std c obj program indices = HmRow.deriveQwordUnits c obj program indices := by
  unfold HmRow.deriveQwordUnitsWith HmRow.deriveQwordUnits
  simp only [programDecodeWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem deriveQcallUnitsWith_std (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    HmRow.deriveQcallUnitsWith std c obj program indices = HmRow.deriveQcallUnits c obj program indices := by
  unfold HmRow.deriveQcallUnitsWith HmRow.deriveQcallUnits
  simp only [programDecodeWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem qwordFindingWith_std (c : Circuit) (pub : Public) (obj : Json) (program : ByteArray) (f : HmRow.PartitionFinding) :
    HmRow.qwordFindingWith std c pub obj program f = HmRow.qwordFinding c pub obj program f := by
  unfold HmRow.qwordFindingWith HmRow.qwordFinding
  simp only [deriveQwordUnitsWith_std, deriveQcallUnitsWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem checkPartitionWith_std (c : Circuit) (obj : Option Json) :
    HmRow.checkPartitionWith std c obj = HmRow.checkPartition c obj := by
  unfold HmRow.checkPartitionWith HmRow.checkPartition
  simp only [partitionDigestWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem loadPublicWith_std (tags : Tags) (c : Circuit) (file : ByteArray) :
    HmRow.loadPublicWith std tags c file = HmRow.loadPublic tags c file := by
  unfold HmRow.loadPublicWith HmRow.loadPublic
  simp only [publicDigestWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem drawnWith_std (c : Circuit) (pub : Public) (draw : Json) :
    HmRow.drawnWith std c pub draw = HmRow.drawn c pub draw := by
  unfold HmRow.drawnWith HmRow.drawn
  simp only [publicDigestWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem stratifiedLawWith_std (obj : Json) (program : ByteArray) (k : Nat) :
    HmRow.stratifiedLawWith std obj program k = HmRow.stratifiedLaw obj program k := by
  unfold HmRow.stratifiedLawWith HmRow.templateStrataWith HmRow.stratifiedLaw
  simp only [descParseWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem workLawWith_std (obj : Json) (program : ByteArray) (k : Nat) (table : Array (String × Nat × Nat)) :
    HmRow.workLawWith std obj program k table = HmRow.workLaw obj program k table := by
  unfold HmRow.workLawWith HmRow.templateStrataWith HmRow.workLaw
  simp only [descParseWith_std]
  try rfl

theorem treeLeafWith_std (bc : ByteArray) : Hm96.Default512.treeLeafWith std bc = pure (Hm96.Default512.treeLeaf bc) := rfl

theorem hashWith_std (tag : String) (parts : List ByteArray) :
    FV3.hashWith std tag parts = pure (FV3.hash tag parts Sha512.hash) := rfl

theorem fv3LeafWith_std (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) :
    FV3.leafWith std dom i schema payload = pure (FV3.leaf dom i schema payload Sha512.hash) := rfl

theorem rootWith_std (dom : ByteArray) (leaves : Array ByteArray) :
    FV3.rootWith std dom leaves = pure (FV3.root dom leaves Sha512.hash) := by
  unfold FV3.rootWith FV3.root
  simp only [hashWith_std, Array.mapM_pure, pure_bind, Std.Legacy.Range.forIn_eq_forIn_range',
    List.forIn_pure_yield_eq_foldl]
  rfl

set_option maxRecDepth 20000 in
theorem checkPublicWith_std (c : Circuit) (pub : Public) : HmRow.checkPublicWith std c pub = HmRow.checkPublic c pub := by
  unfold HmRow.checkPublicWith HmRow.checkPublic
  simp -zeta only [treeLeafWith_std, fv3LeafWith_std, pure_bind, Array.mapM_pure, rootWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem outCheckPublicWith_std (c : Circuit) (pub : Public) : HmOut.checkPublicWith std c pub = HmOut.checkPublic c pub := by
  unfold HmOut.checkPublicWith HmOut.checkPublic
  simp -zeta only [treeLeafWith_std, fv3LeafWith_std, pure_bind, Array.mapM_pure, rootWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem setupHWith_std (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw : Option Json) (partition : Option Json) (program : Option ByteArray) :
    Stmt.setupHWith std tags circuitFile publicFile coins tables draw partition program
      = Stmt.setupH tags circuitFile publicFile coins tables draw partition program := by
  unfold Stmt.setupHWith Stmt.setupH
  simp -zeta only [parseTypedWith_std, parseWith_std, checkPartitionWith_std, loadPublicWith_std, drawnWith_std,
    stratifiedLawWith_std, checkPublicWith_std, deriveTemplateUnitsWith_std, qwordFindingWith_std, std_sha512, pure_bind]
  try rfl

set_option maxRecDepth 20000 in
theorem outParseWith_std (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    HmOut.parseWith std tags bytes tables = HmOut.parse tags bytes tables := by
  unfold HmOut.parseWith HmOut.parse
  simp -zeta only [parseWith_std, readWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem outParseTmplWith_std (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    HmOut.parseTmplWith std tags bytes tables = HmOut.parseTmpl tags bytes tables := by
  unfold HmOut.parseTmplWith HmOut.parseTmpl
  simp -zeta only [parseWith_std, monadLift_self]
  try rfl

theorem outParseCommitWith_std (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    HmOut.parseCommitWith std tags bytes tables = HmOut.parseCommit tags bytes tables := by
  unfold HmOut.parseCommitWith HmOut.parseCommit
  rw [outParseTmplWith_std, outParseWith_std]

set_option maxRecDepth 20000 in
theorem inLoadPublicWith_std (tags : Tags) (c : Circuit) (pi : HmIn.Ports) (file : ByteArray) :
    HmIn.loadPublicWith std tags c pi file = HmIn.loadPublic tags c pi file := by
  unfold HmIn.loadPublicWith HmIn.loadPublic
  simp -zeta only [loadPublicWith_std]
  try rfl

theorem inDrawnWith_std (c : Circuit) (pi : HmIn.Ports) (pub : Public) (draw : Json) :
    HmIn.drawnWith std c pi pub draw = HmIn.drawn c pi pub draw := by
  unfold HmIn.drawnWith HmIn.drawn
  exact drawnWith_std _ _ _

set_option maxRecDepth 20000 in
theorem linksCheckWith_std (c : Circuit) (pop : Public) (program : ByteArray) :
    HmLinks.checkWith std c pop program = HmLinks.check c pop program := by
  unfold HmLinks.checkWith HmLinks.check
  simp -zeta only [programDecodeWith_std]
  try rfl

set_option maxRecDepth 20000 in
theorem setupHiddenWith_std (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json) (program : Option ByteArray)
    (publicInputs : Option ByteArray) :
    Stmt.setupHiddenWith std tags circuitFile publicFile coins tables draw partition program publicInputs
      = Stmt.setupHidden tags circuitFile publicFile coins tables draw partition program publicInputs := by
  unfold Stmt.setupHiddenWith Stmt.setupHidden
  simp -zeta only [outParseCommitWith_std, checkPartitionWith_std, inLoadPublicWith_std, inDrawnWith_std,
    stratifiedLawWith_std, outCheckPublicWith_std, deriveTemplateUnitsWith_std, linksCheckWith_std, qwordFindingWith_std,
    std_sha512, pure_bind]
  try rfl

set_option maxRecDepth 20000 in
theorem setupTablesWith_std (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Stmt.setupTablesWith std tags circuitFile publicFile coins count tables draw partition program work stratified closure
        publicInputs
      = Stmt.setupTables tags circuitFile publicFile coins count tables draw partition program work stratified closure
        publicInputs := by
  unfold Stmt.setupTablesWith Stmt.setupTables
  simp -zeta only [stratifiedLawWith_std, workLawWith_std, setupHiddenWith_std, setupHWith_std, outParseWith_std,
    inLoadPublicWith_std, loadPublicWith_std, parseWith_std, std_sha512, pure_bind]
  try rfl

end With

/-- `Stmt.setupTables` is `Stmt.setupTablesWith` at SHA-512, run in `V`. -/
theorem Stmt.setupTables_eq_setupTablesWith (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Stmt.setupTables tags circuitFile publicFile coins count tables draw partition program work stratified closure publicInputs
      = Stmt.setupTablesWith (M := V) (Hashes.ofSha512 Sha512.hash) tags circuitFile publicFile coins count tables draw partition
          program work stratified closure publicInputs :=
  (With.setupTablesWith_std tags circuitFile publicFile coins count tables draw partition program work stratified closure
    publicInputs).symm

end Flock
