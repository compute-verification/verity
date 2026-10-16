/-
Every table the deployed verifier sets up has the registry's Merkle scheme of its id (`MerkleScheme.Named`): each builder
(`Stmt.setup`, `Stmt.setupH`, `Stmt.setupHidden`, and `Stmt.setupTables` over the last two) takes its scheme from
`MerkleScheme.find?` and nowhere else, and `Setup.ofCircuit` keeps it. So on every session `Stmt.setupTables` sets up,
`verify` and `Zk.verify` are `verifyWith Sha512.hash` and `Zk.verifyWith Sha512.hash`, with no hypothesis left
(`verify_eq_verifyWith_of_setupTables`, `Zk.verify_eq_verifyWith_of_setupTables`, and `flock-witness`'s
`Witness.acceptsZKRAt_setupOf`). `flock-verify`'s own builders, `buildStmt` and `buildSession`, get the same in
`FlockVerify`.

`Returns x P`: every value `x` returns satisfies `P`. `returns_walk` proves it by walking `x`'s do block with reducible
steps only, so a setup is never unfolded past its binds.
-/
import Flock.HmRow
import Flock.Witness
import Flock.With.Eq
import Flock.With.ZkEq

namespace Flock.With

/-- Every value `x` returns satisfies `P`. -/
def Returns {α : Type} (x : V α) (P : α → Prop) : Prop := ∀ a, x = .ok a → P a

namespace Returns

theorem pure' {α : Type} {a : α} {P : α → Prop} (h : P a) : Returns (pure a) P := by
  intro b hb; cases hb; exact h

theorem throw' {α : Type} {e : String} {P : α → Prop} : Returns (throw e : V α) P := by
  intro b hb; cases hb

theorem bind' {α β : Type} {x : V α} {f : α → V β} {P : β → Prop} (h : ∀ a, Returns (f a) P) : Returns (x >>= f) P := by
  intro b hb
  cases x with
  | error e => cases hb
  | ok a => exact h a b hb

theorem bindQ {α β : Type} {x : V α} {f : α → V β} {Q : α → Prop} {P : β → Prop} (hx : Returns x Q)
    (h : ∀ a, Q a → Returns (f a) P) : Returns (x >>= f) P := by
  intro b hb
  cases hxe : x with
  | error e => rw [hxe] at hb; cases hb
  | ok a => rw [hxe] at hb; exact h a (hx a hxe) b hb

theorem mapM_list {α β : Type} {f : α → V β} {Q : β → Prop} (h : ∀ a, Returns (f a) Q) :
    ∀ l : List α, Returns (l.mapM f) fun ys => ∀ y ∈ ys, Q y
  | [] => pure' fun _ hy => nomatch hy
  | a :: l => by
    rw [List.mapM_cons]
    refine bindQ (h a) fun b hb => bindQ (mapM_list h l) fun bs hbs => pure' fun y hy => ?_
    rcases List.mem_cons.mp hy with rfl | hy
    · exact hb
    · exact hbs y hy

theorem mapM {α β : Type} {f : α → V β} {Q : β → Prop} (h : ∀ a, Returns (f a) Q) (xs : Array α) :
    Returns (xs.mapM f) fun ys => ∀ y ∈ ys, Q y := by
  rw [Array.mapM_eq_mapM_toList]
  intro ys hys
  cases hl : List.mapM f xs.toList with
  | error e => rw [hl] at hys; cases hys
  | ok l =>
    rw [hl] at hys; cases hys
    intro y hy
    exact mapM_list h _ l hl y (List.mem_toArray.mp hy)

theorem ite {α : Type} {c : Prop} [Decidable c] {x y : V α} {P : α → Prop} (h₁ : c → Returns x P)
    (h₂ : ¬c → Returns y P) : Returns (if c then x else y) P := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact h₁ hc
  · rw [ite_eq_right hc]; exact h₂ hc

theorem dite {α : Type} {c : Prop} [Decidable c] {x : c → V α} {y : ¬c → V α} {P : α → Prop}
    (h₁ : ∀ h, Returns (x h) P) (h₂ : ∀ h, Returns (y h) P) : Returns (if h : c then x h else y h) P := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact h₁ hc
  · rw [dite_eq_right hc]; exact h₂ hc

/-- A bind on a statement that has a registry scheme. -/
theorem bindNamed {β : Type} {x : V Stmt} {f : Stmt → V β} {P : β → Prop} (hx : Returns x fun st => st.merkle.Named)
    (h : ∀ st : Stmt, st.merkle.Named → Returns (f st) P) : Returns (x >>= f) P :=
  bindQ hx h

/-- A bind on statements that each have a registry scheme. -/
theorem bindNamedMapM {α β : Type} {xs : Array α} {f : α → V Stmt} {g : Array Stmt → V β} {P : β → Prop}
    (hf : ∀ a, Returns (f a) fun st => st.merkle.Named) (hg : ∀ sts : Array Stmt, (∀ st ∈ sts, st.merkle.Named) →
      Returns (g sts) P) : Returns (xs.mapM f >>= g) P :=
  bindQ (mapM hf xs) hg

end Returns

theorem named_of_find? {id : String} {ms : MerkleScheme} (h : MerkleScheme.find? id = some ms) : ms.Named := by
  intro s hs
  have hid : ms.id = id := by simpa using List.find?_some h
  rw [hid, h] at hs
  exact (Option.some.inj hs).symm

theorem named_singleton {spec : SessionSpec} {st : Stmt} (h : st.merkle.Named) :
    ∀ s ∈ (spec, #[st]).2, s.merkle.Named := by
  intro s hs
  rw [List.mem_singleton.mp (Array.mem_toList_iff.mpr hs)]
  exact h

/-- Walks a do block of `V` to a `Returns` goal at each `pure`. A bind keeps what its action returns when one of `ls`
covers the action or the action returns statements (each then shown to have a registry scheme), and forgets it
otherwise; a `throw` returns nothing; a branch is split, and a join point's call unfolded where it branches. A goal one of
`ls` states is closed by it, and a `pure` by a scheme the registry found, an assumption, or a one-table session of a
statement with one. -/
syntax "returns_walk" (" [" term,* "]")? : tactic
macro_rules
  | `(tactic| returns_walk) => `(tactic| returns_walk [])
  | `(tactic| returns_walk [$ls,*]) => `(tactic| repeat' (first
    | extract_lets
    $[| (with_reducible exact $ls)]*
    $[| (with_reducible apply Returns.bindQ $ls); intro _ _]*
    | (with_reducible refine Returns.bindNamedMapM (fun _ => ?_) fun _ _ => ?_)
    | (with_reducible refine Returns.bindNamed ?_ fun _ _ => ?_)
    | (with_reducible apply Returns.bind'); intro _
    | (with_reducible apply Returns.throw')
    | (with_reducible apply Returns.pure');
      first | exact named_of_find? (by assumption) | assumption | exact named_singleton (by assumption)
    | (with_reducible apply Returns.ite) <;> intro _
    | (with_reducible apply Returns.dite) <;> intro _
    | split
    | (with_reducible conv => arg 1; whnf); split))

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
