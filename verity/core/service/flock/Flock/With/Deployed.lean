/-
Every table the deployed verifier sets up has the registry's Merkle scheme of its id (`MerkleScheme.Named`): each builder
(`Stmt.setup`, `Stmt.setupH`, `Stmt.setupHidden`, and `Stmt.setupTables` over the last two) takes its scheme from
`MerkleScheme.find?` and nowhere else, and `Setup.ofCircuit` keeps it. So on every session `Stmt.setupTables` sets up,
`verify` and `Zk.verify` are `verifyWith Sha512.hash` and `Zk.verifyWith Sha512.hash`, with no hypothesis left
(`verify_eq_verifyWith_of_setupTables`, `Zk.verify_eq_verifyWith_of_setupTables`, and `flock-witness`'s
`Witness.acceptsZKRAt_setupOf`). `flock-verify`'s own builders, `buildStmt` and `buildSession`, get the same in
`FlockVerify`.

`Returns x P` (`Flock.With.Returns`): every value `x` returns satisfies `P`. `returns_walk` (`Flock.With.ReturnsWalk`)
proves it by walking `x`'s do block with reducible steps only, so a setup is never unfolded past its binds.
-/
import Flock.With.ReturnsWalk
import Flock.HmRow
import Flock.Witness
import Flock.With.Eq
import Flock.With.ZkEq

namespace Flock.With

theorem setup_named (tags : Tags) (leaf : RowLeaf) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray)) :
    Returns (Stmt.setup tags leaf circuitFile publicFile coins tables) fun st => st.merkle.Named := by
  unfold Stmt.setup
  returns_walk

theorem setupH_named (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw partition : Option Lean.Json) (program : Option ByteArray) :
    Returns (Stmt.setupH tags circuitFile publicFile coins tables draw partition program) fun st => st.merkle.Named := by
  unfold Stmt.setupH
  returns_walk

theorem setupHidden_named (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (tables : List (String × ByteArray)) (draw partition : Option Lean.Json) (program publicInputs : Option ByteArray) :
    Returns (Stmt.setupHidden tags circuitFile publicFile coins tables draw partition program publicInputs)
      fun st => st.merkle.Named := by
  unfold Stmt.setupHidden
  returns_walk

theorem setupTables_named (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins) (count : Nat)
    (tables : List (String × ByteArray)) (draw partition : Option Lean.Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Returns (Stmt.setupTables tags circuitFile publicFile coins count tables draw partition program work stratified closure
      publicInputs) fun r => ∀ st ∈ r.2, st.merkle.Named := by
  unfold Stmt.setupTables
  -- the draw checks' continuations, each proved once and closed at every call
  extract_lets setup splits names count1 pubIn work1 strat1
  have hsetup : ∀ t d, Returns (setup t d) fun st => st.merkle.Named := by
    intro t d
    returns_walk [setupHidden_named _ _ _ _ _ _ _ _ _, setupH_named _ _ _ _ _ _ _ _]
  clear_value setup
  have hcount1 : ∀ u, Returns (count1 u) fun r : SessionSpec × Array Stmt => ∀ st ∈ r.2, st.merkle.Named := by
    intro _
    returns_walk [hsetup _ _]
  clear_value count1
  have hpubIn : ∀ u, Returns (pubIn u) fun r : SessionSpec × Array Stmt => ∀ st ∈ r.2, st.merkle.Named := by
    intro _
    returns_walk [hcount1 _]
  clear_value pubIn
  have hwork1 : ∀ u, Returns (work1 u) fun r : SessionSpec × Array Stmt => ∀ st ∈ r.2, st.merkle.Named := by
    intro _
    returns_walk [hpubIn _]
  clear_value work1
  have hstrat1 : ∀ u, Returns (strat1 u) fun r : SessionSpec × Array Stmt => ∀ st ∈ r.2, st.merkle.Named := by
    intro _
    returns_walk [hwork1 _]
  clear_value strat1
  returns_walk [hstrat1 _]

theorem named_ofCircuit {sts : Array Stmt} (h : ∀ st ∈ sts, st.merkle.Named) :
    ∀ s ∈ sts.map Setup.ofCircuit, s.merkle.Named := by
  intro s hs
  obtain ⟨st, hst, rfl⟩ := Array.mem_map.mp hs
  exact h st hst

end Flock.With

namespace Flock

/-- `verify` is `verifyWith Sha512.hash` on a session whose builder `s` returns only statements of registry schemes. -/
theorem verify_eq_verifyWith_of_returns {s : V (SessionSpec × Array Stmt)}
    (hs : With.Returns s fun r => ∀ st ∈ r.2, st.merkle.Named) {spec : SessionSpec} {sts : Array Stmt}
    (h : s = .ok (spec, sts)) (record : String) (proofs : Array ByteArray) :
    verify spec (sts.map Setup.ofCircuit) record proofs
      = verifyWith Sha512.hash spec (sts.map Setup.ofCircuit) record proofs :=
  verify_eq_verifyWith _ _ _ _ (With.named_ofCircuit (hs _ h))

/-- `Zk.verify` is `Zk.verifyWith Sha512.hash` on a session whose builder `s` returns only statements of registry
schemes. -/
theorem Zk.verify_eq_verifyWith_of_returns {s : V (SessionSpec × Array Stmt)}
    (hs : With.Returns s fun r => ∀ st ∈ r.2, st.merkle.Named) {spec : SessionSpec} {sts : Array Stmt}
    (h : s = .ok (spec, sts)) (record : String) (proofs : Array ByteArray) :
    Zk.verify spec (sts.map Setup.ofCircuit) record proofs
      = Zk.verifyWith Sha512.hash spec (sts.map Setup.ofCircuit) record proofs :=
  Zk.verify_eq_verifyWith _ _ _ _ (With.named_ofCircuit (hs _ h))

/-- On every session `Stmt.setupTables` sets up, `verify` is `verifyWith Sha512.hash`. -/
theorem verify_eq_verifyWith_of_setupTables (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (count : Nat) (tables : List (String × ByteArray)) (draw partition : Option Lean.Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) {spec : SessionSpec}
    {sts : Array Stmt}
    (h : Stmt.setupTables tags circuitFile publicFile coins count tables draw partition program work stratified closure
      publicInputs = .ok (spec, sts)) (record : String) (proofs : Array ByteArray) :
    verify spec (sts.map Setup.ofCircuit) record proofs
      = verifyWith Sha512.hash spec (sts.map Setup.ofCircuit) record proofs :=
  verify_eq_verifyWith_of_returns
    (With.setupTables_named tags circuitFile publicFile coins count tables draw partition program work stratified closure publicInputs) h record proofs

/-- On every session `Stmt.setupTables` sets up, `Zk.verify` is `Zk.verifyWith Sha512.hash`. -/
theorem Zk.verify_eq_verifyWith_of_setupTables (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (count : Nat) (tables : List (String × ByteArray)) (draw partition : Option Lean.Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) {spec : SessionSpec}
    {sts : Array Stmt}
    (h : Stmt.setupTables tags circuitFile publicFile coins count tables draw partition program work stratified closure
      publicInputs = .ok (spec, sts)) (record : String) (proofs : Array ByteArray) :
    Zk.verify spec (sts.map Setup.ofCircuit) record proofs
      = Zk.verifyWith Sha512.hash spec (sts.map Setup.ofCircuit) record proofs :=
  Zk.verify_eq_verifyWith_of_returns
    (With.setupTables_named tags circuitFile publicFile coins count tables draw partition program work stratified closure publicInputs) h record proofs

/-- `flock-witness`'s `acceptsZKRAt` runs `Zk.verifyWith Sha512.hash` on a session whose builder `s` returns only
statements of registry schemes. -/
theorem Witness.acceptsZKRAt_eq {s : V (SessionSpec × Array Stmt)}
    (hs : With.Returns s fun r => ∀ st ∈ r.2, st.merkle.Named) (own : Array Registered.Port) (record : String)
    (proofs : Array ByteArray) :
    Witness.acceptsZKRAt s own record proofs = (do
      let r ← s
      r.2.forM (checkRegistered · (some own))
      r.2.forM Zk.shapeOk
      Zk.verifyWith Sha512.hash r.1 (r.2.map Setup.ofCircuit) record proofs) := by
  unfold Witness.acceptsZKRAt
  cases hse : s with
  | error e => show Except.error e = Except.error e; rfl
  | ok r =>
    obtain ⟨spec, sts⟩ := r
    show (do
        sts.forM (checkRegistered · (some own))
        sts.forM Zk.shapeOk
        Zk.verify spec (sts.map Setup.ofCircuit) record proofs) = (do
        sts.forM (checkRegistered · (some own))
        sts.forM Zk.shapeOk
        Zk.verifyWith Sha512.hash spec (sts.map Setup.ofCircuit) record proofs)
    rw [Zk.verify_eq_verifyWith_of_returns hs hse]

/-- `flock-witness`'s `acceptsZKRAt` on the session `setupOf` sets up runs `Zk.verifyWith Sha512.hash`. -/
theorem Witness.acceptsZKRAt_setupOf (I : Witness.Inputs) (own : Array Registered.Port) (record : String)
    (proofs : Array ByteArray) :
    Witness.acceptsZKRAt (Witness.setupOf I record) own record proofs = (do
      let r ← Witness.setupOf I record
      r.2.forM (checkRegistered · (some own))
      r.2.forM Zk.shapeOk
      Zk.verifyWith Sha512.hash r.1 (r.2.map Setup.ofCircuit) record proofs) :=
  Witness.acceptsZKRAt_eq (With.setupTables_named _ _ _ _ _ _ _ _ _ none none none none) own record proofs

end Flock
