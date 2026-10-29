/-
The `…With` functions of `Flock.With.Setup` under a simulation (`Flock.With.Sim`): run with two oracles that `S` relates
hash by hash, each function's two runs are related by `S`, `Stmt.setupTablesWith`'s among them (`setupTablesWith_sim`).
The monads are lawful, and in each a `throw` ends the run (`Throws`), so that the walk meets a check's join point only at
its calls that do not throw.

With the oracles of `Flock.With.Log`: forgetting the log gives back setup at `h` in `V` (`Stmt.setupLogged_fst`), and
the log's length is `Stmt.setupCalls h`, the number of SHA-512 calls a run with the counting oracle makes
(`Stmt.setupLogged_size`).
-/
import Flock.With.SimWalk
import Flock.With.Setup
import Flock.With.Log

namespace Flock.With

open Lean (Json)

variable {M N : Type → Type} [Monad M] [Monad N] [MonadExceptOf String M] [MonadExceptOf String N] [MonadLiftT V M]
  [MonadLiftT V N]

/-- `throw` ends a run of `M`: nothing bound after it runs. -/
structure Throws (M : Type → Type) [Monad M] [MonadExceptOf String M] : Prop where
  bind : ∀ {α β : Type} (e : String) (f : α → M β), (throw e : M α) >>= f = throw e

theorem Throws.v : Throws V := ⟨fun _ _ => rfl⟩

theorem Throws.exceptT {m : Type → Type} [Monad m] [LawfulMonad m] : Throws (ExceptT String m) :=
  ⟨fun e f => by
    apply ExceptT.ext
    simp only [ExceptT.run_bind, ExceptT.run_throw, pure_bind]⟩

namespace Sim

theorem map (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {α β : Type} (f : α → β) {x : M α} {y : N α} (h : S.R x y) :
    S.R (f <$> x) (f <$> y) := by
  rw [map_eq_pure_bind, map_eq_pure_bind]
  exact S.bind h fun a => S.pure (f a)

end Sim

theorem treeLeafWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (bc : ByteArray) : S.R (Hm96.Default512.treeLeafWith hs bc) (Hm96.Default512.treeLeafWith hs' bc) :=
  hh.sha512 _

theorem fv3HashWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (tag : String) (parts : List ByteArray) : S.R (FV3.hashWith hs tag parts) (FV3.hashWith hs' tag parts) :=
  hh.sha512 _

theorem fv3LeafWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (dom : ByteArray) (i : Nat) (schema : String) (payload : ByteArray) :
    S.R (FV3.leafWith hs dom i schema payload) (FV3.leafWith hs' dom i schema payload) :=
  hh.sha512 _

theorem fv3RootWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] {hs : Flock.Hashes M} {hs' : Flock.Hashes N}
    (hh : S.Hashes hs hs') (dom : ByteArray) (leaves : Array ByteArray) :
    S.R (FV3.rootWith hs dom leaves) (FV3.rootWith hs' dom leaves) := by
  unfold FV3.rootWith FV3.rootM
  sim_walk [fv3HashWith_sim S hh]

theorem typeDigestWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (t : CircuitType) : S.R (t.digestWith hs) (t.digestWith hs') := by
  unfold CircuitType.digestWith
  sim_walk

theorem layoutDigestWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (l : Layout) : S.R (l.digestWith hs) (l.digestWith hs') := by
  unfold Layout.digestWith
  sim_walk

theorem typedClassOfWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (mj : Json) (tables : List (String × ByteArray)) :
    S.R (Typed.classOfWith hs mj tables) (Typed.classOfWith hs' mj tables) := by
  unfold Typed.classOfWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [typeDigestWith_sim S hh, layoutDigestWith_sim S hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [typeDigestWith_sim S hh, layoutDigestWith_sim S hh])

theorem templateOfWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (name : String) (done : Array DeriveAll.D)
    (ls : Array Layout) (unit : Nat) (t : CircuitType) (held : Array Typed.Held) :
    S.R (Typed.templateOfWith hs name done ls unit t held) (Typed.templateOfWith hs' name done ls unit t held) := by
  unfold Typed.templateOfWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [Sim.mapM_list S _ (layoutDigestWith_sim S hh)]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [Sim.mapM_list S _ (layoutDigestWith_sim S hh)])

theorem readWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) :
    S.R (Typed.readWith hs tags bytes tables) (Typed.readWith hs' tags bytes tables) := by
  unfold Typed.readWith
  sim_walk [typedClassOfWith_sim S tM tN hh, templateOfWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [typedClassOfWith_sim S tM tN hh, templateOfWith_sim S tM tN hh])

theorem descParseWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (bytes : ByteArray) :
    S.R (TemplateQuery.Desc.parseWith hs bytes) (TemplateQuery.Desc.parseWith hs' bytes) := by
  unfold TemplateQuery.Desc.parseWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk)

theorem programDecodeWith_sim (S : Sim M N) (tM : Throws M) (tN : Throws N) {hs : Flock.Hashes M}
    {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (bytes : ByteArray) :
    S.R (Program.decodeWith hs.sha512 bytes) (Program.decodeWith hs'.sha512 bytes) := by
  unfold Program.decodeWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk)

theorem hmClassOfWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (bytes : ByteArray) (head : String) (mj : Json) :
    S.R (HmRow.classOfWith hs bytes head mj) (HmRow.classOfWith hs' bytes head mj) := by
  unfold HmRow.classOfWith
  sim_walk
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk)

theorem parseWith_sim (S : Sim M N) (tM : Throws M) (tN : Throws N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N}
    (hh : S.Hashes hs hs') (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray))
    (tmpl : Option Typed.Template) (pre : Array (String × Net)) :
    S.R (HmRow.parseWith hs tags bytes tables tmpl pre) (HmRow.parseWith hs' tags bytes tables tmpl pre) := by
  unfold HmRow.parseWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [hmClassOfWith_sim S hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [hmClassOfWith_sim S hh])


theorem parseTypedWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) :
    S.R (HmRow.parseTypedWith hs tags bytes tables) (HmRow.parseTypedWith hs' tags bytes tables) := by
  unfold HmRow.parseTypedWith
  sim_walk [readWith_sim S tM tN hh, parseWith_sim S tM tN hh]

theorem partitionDigestWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (obj : Json) : S.R (HmRow.partitionDigestWith hs obj) (HmRow.partitionDigestWith hs' obj) := by
  unfold HmRow.partitionDigestWith
  sim_walk

theorem publicDigestWith_sim (S : Sim M N) {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs')
    (header : Json) (pubs : ByteArray) :
    S.R (HmRow.publicDigestWith hs header pubs) (HmRow.publicDigestWith hs' header pubs) := by
  unfold HmRow.publicDigestWith
  sim_walk

theorem deriveTemplateUnitsWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (obj : Json) (program : ByteArray)
    (indices : Array Nat) :
    S.R (HmRow.deriveTemplateUnitsWith hs c obj program indices) (HmRow.deriveTemplateUnitsWith hs' c obj program indices) := by
  unfold HmRow.deriveTemplateUnitsWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [descParseWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [descParseWith_sim S tM tN hh])

theorem deriveQwordUnitsWith_sim (S : Sim M N) (tM : Throws M) (tN : Throws N) {hs : Flock.Hashes M}
    {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    S.R (HmRow.deriveQwordUnitsWith hs c obj program indices) (HmRow.deriveQwordUnitsWith hs' c obj program indices) := by
  unfold HmRow.deriveQwordUnitsWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [programDecodeWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [programDecodeWith_sim S tM tN hh])

theorem deriveQcallUnitsWith_sim (S : Sim M N) (tM : Throws M) (tN : Throws N) {hs : Flock.Hashes M}
    {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    S.R (HmRow.deriveQcallUnitsWith hs c obj program indices) (HmRow.deriveQcallUnitsWith hs' c obj program indices) := by
  unfold HmRow.deriveQcallUnitsWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [programDecodeWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [programDecodeWith_sim S tM tN hh])

theorem qwordFindingWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (pub : Public) (obj : Json)
    (program : ByteArray) (f : HmRow.PartitionFinding) :
    S.R (HmRow.qwordFindingWith hs c pub obj program f) (HmRow.qwordFindingWith hs' c pub obj program f) := by
  unfold HmRow.qwordFindingWith
  sim_walk [deriveQwordUnitsWith_sim S tM tN hh, deriveQcallUnitsWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [deriveQwordUnitsWith_sim S tM tN hh, deriveQcallUnitsWith_sim S tM tN hh])

theorem checkPartitionWith_sim (S : Sim M N) (tM : Throws M) (tN : Throws N) {hs : Flock.Hashes M}
    {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (obj : Option Json) :
    S.R (HmRow.checkPartitionWith hs c obj) (HmRow.checkPartitionWith hs' c obj) := by
  unfold HmRow.checkPartitionWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [partitionDigestWith_sim S hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [partitionDigestWith_sim S hh])

theorem loadPublicWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (c : Circuit) (file : ByteArray) :
    S.R (HmRow.loadPublicWith hs tags c file) (HmRow.loadPublicWith hs' tags c file) := by
  unfold HmRow.loadPublicWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [publicDigestWith_sim S hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [publicDigestWith_sim S hh])

theorem drawnWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (pub : Public) (draw : Json) :
    S.R (HmRow.drawnWith hs c pub draw) (HmRow.drawnWith hs' c pub draw) := by
  unfold HmRow.drawnWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [publicDigestWith_sim S hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [publicDigestWith_sim S hh])

theorem templateStrataWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (law : String) (obj : Json) (program : ByteArray) :
    S.R (HmRow.templateStrataWith hs law obj program) (HmRow.templateStrataWith hs' law obj program) := by
  unfold HmRow.templateStrataWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [descParseWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [descParseWith_sim S tM tN hh])

theorem stratifiedLawWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (obj : Json) (program : ByteArray) (k : Nat) :
    S.R (HmRow.stratifiedLawWith hs obj program k) (HmRow.stratifiedLawWith hs' obj program k) := by
  unfold HmRow.stratifiedLawWith
  sim_walk [templateStrataWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [templateStrataWith_sim S tM tN hh])

theorem workLawWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (obj : Json) (program : ByteArray) (k : Nat)
    (table : Array (String × Nat × Nat)) :
    S.R (HmRow.workLawWith hs obj program k table) (HmRow.workLawWith hs' obj program k table) := by
  unfold HmRow.workLawWith
  sim_walk [templateStrataWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [templateStrataWith_sim S tM tN hh])

theorem checkPublicWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (pub : Public) :
    S.R (HmRow.checkPublicWith hs c pub) (HmRow.checkPublicWith hs' c pub) := by
  unfold HmRow.checkPublicWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [fv3LeafWith_sim S hh, treeLeafWith_sim S hh, fv3RootWith_sim S hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [fv3LeafWith_sim S hh, treeLeafWith_sim S hh, fv3RootWith_sim S hh])


theorem setupHWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (circuitFile publicFile : ByteArray)
    (coins : Coins) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) :
    S.R (Stmt.setupHWith hs tags circuitFile publicFile coins tables draw partition program)
      (Stmt.setupHWith hs' tags circuitFile publicFile coins tables draw partition program) := by
  unfold Stmt.setupHWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [parseTypedWith_sim S tM tN hh, parseWith_sim S tM tN hh, Sim.map S, checkPartitionWith_sim S tM tN hh,
    loadPublicWith_sim S tM tN hh, drawnWith_sim S tM tN hh, stratifiedLawWith_sim S tM tN hh, checkPublicWith_sim S tM tN hh,
    deriveTemplateUnitsWith_sim S tM tN hh, qwordFindingWith_sim S tM tN hh]
  iterate 40
    all_goals try (dsimp +zetaDelta -zeta only)
    all_goals sim_walk [parseTypedWith_sim S tM tN hh, parseWith_sim S tM tN hh, Sim.map S,
      checkPartitionWith_sim S tM tN hh, loadPublicWith_sim S tM tN hh, drawnWith_sim S tM tN hh,
      stratifiedLawWith_sim S tM tN hh, checkPublicWith_sim S tM tN hh, deriveTemplateUnitsWith_sim S tM tN hh,
      qwordFindingWith_sim S tM tN hh]

theorem hmOutParseWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) :
    S.R (HmOut.parseWith hs tags bytes tables) (HmOut.parseWith hs' tags bytes tables) := by
  unfold HmOut.parseWith
  sim_walk [parseWith_sim S tM tN hh, readWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [parseWith_sim S tM tN hh, readWith_sim S tM tN hh])

theorem hmOutCheckPublicWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (pub : Public) :
    S.R (HmOut.checkPublicWith hs c pub) (HmOut.checkPublicWith hs' c pub) := by
  unfold HmOut.checkPublicWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [fv3LeafWith_sim S hh, treeLeafWith_sim S hh, fv3RootWith_sim S hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [fv3LeafWith_sim S hh, treeLeafWith_sim S hh, fv3RootWith_sim S hh])

theorem hmInLoadPublicWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (c : Circuit) (pi : HmIn.Ports)
    (file : ByteArray) :
    S.R (HmIn.loadPublicWith hs tags c pi file) (HmIn.loadPublicWith hs' tags c pi file) := by
  unfold HmIn.loadPublicWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [loadPublicWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [loadPublicWith_sim S tM tN hh])

theorem hmInDrawnWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (pi : HmIn.Ports) (pub : Public)
    (draw : Json) : S.R (HmIn.drawnWith hs c pi pub draw) (HmIn.drawnWith hs' c pi pub draw) :=
  drawnWith_sim S tM tN hh _ pub draw

theorem hmLinksCheckWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (c : Circuit) (pop : Public) (program : ByteArray) :
    S.R (HmLinks.checkWith hs c pop program) (HmLinks.checkWith hs' c pop program) := by
  unfold HmLinks.checkWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [programDecodeWith_sim S tM tN hh]
  iterate 40 (all_goals (try (dsimp +zetaDelta -zeta only)); all_goals sim_walk [programDecodeWith_sim S tM tN hh])


theorem setupHiddenWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (circuitFile publicFile : ByteArray)
    (coins : Coins) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) (publicInputs : Option ByteArray) :
    S.R (Stmt.setupHiddenWith hs tags circuitFile publicFile coins tables draw partition program publicInputs)
      (Stmt.setupHiddenWith hs' tags circuitFile publicFile coins tables draw partition program publicInputs) := by
  unfold Stmt.setupHiddenWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [hmOutParseWith_sim S tM tN hh, Sim.map S, checkPartitionWith_sim S tM tN hh,
    hmInLoadPublicWith_sim S tM tN hh, hmInDrawnWith_sim S tM tN hh, stratifiedLawWith_sim S tM tN hh,
    hmOutCheckPublicWith_sim S tM tN hh, deriveTemplateUnitsWith_sim S tM tN hh, hmLinksCheckWith_sim S tM tN hh,
    qwordFindingWith_sim S tM tN hh]
  iterate 40
    all_goals try (dsimp +zetaDelta -zeta only)
    all_goals sim_walk [hmOutParseWith_sim S tM tN hh, Sim.map S, checkPartitionWith_sim S tM tN hh,
      hmInLoadPublicWith_sim S tM tN hh, hmInDrawnWith_sim S tM tN hh, stratifiedLawWith_sim S tM tN hh,
      hmOutCheckPublicWith_sim S tM tN hh, deriveTemplateUnitsWith_sim S tM tN hh, hmLinksCheckWith_sim S tM tN hh,
      qwordFindingWith_sim S tM tN hh]

-- the walk meets each of the three draw checks' join points at every call that does not throw
set_option maxHeartbeats 800000 in
theorem setupTablesWith_sim (S : Sim M N) [LawfulMonad M] [LawfulMonad N] (tM : Throws M) (tN : Throws N)
    {hs : Flock.Hashes M} {hs' : Flock.Hashes N} (hh : S.Hashes hs hs') (tags : Tags) (circuitFile publicFile : ByteArray)
    (coins : Coins) (count : Nat) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    S.R (Stmt.setupTablesWith hs tags circuitFile publicFile coins count tables draw partition program work stratified
        closure publicInputs)
      (Stmt.setupTablesWith hs' tags circuitFile publicFile coins count tables draw partition program work stratified
        closure publicInputs) := by
  unfold Stmt.setupTablesWith
  simp -zeta only [tM.bind, tN.bind]
  sim_walk [stratifiedLawWith_sim S tM tN hh, workLawWith_sim S tM tN hh, setupHiddenWith_sim S tM tN hh,
    setupHWith_sim S tM tN hh, hmOutParseWith_sim S tM tN hh, hmInLoadPublicWith_sim S tM tN hh,
    loadPublicWith_sim S tM tN hh, parseWith_sim S tM tN hh]
  iterate 40
    all_goals try (dsimp +zetaDelta -zeta only)
    all_goals sim_walk [stratifiedLawWith_sim S tM tN hh, workLawWith_sim S tM tN hh, setupHiddenWith_sim S tM tN hh,
      setupHWith_sim S tM tN hh, hmOutParseWith_sim S tM tN hh, hmInLoadPublicWith_sim S tM tN hh,
      loadPublicWith_sim S tM tN hh, parseWith_sim S tM tN hh]


end Flock.With

namespace Flock

open Lean (Json)

/-- `Stmt.setupTablesWith` with the logging oracle from the log `log`: its verdict and the log with every SHA-512 input
appended. -/
def Stmt.setupLogged (h : ByteArray → ByteArray) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (count : Nat) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) (log : Array ByteArray := #[]) :
    Except String (SessionSpec × Array Stmt) × Array ByteArray :=
  (Stmt.setupTablesWith (Hashes.logged h) tags circuitFile publicFile coins count tables draw partition program work
    stratified closure publicInputs).go log

/-- The number of SHA-512 calls of `Stmt.setupTablesWith` with `h` as the oracle. -/
def Stmt.setupCalls (h : ByteArray → ByteArray) (tags : Tags) (circuitFile publicFile : ByteArray) (coins : Coins)
    (count : Nat) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) : Nat :=
  ((Stmt.setupTablesWith (Hashes.counted h) tags circuitFile publicFile coins count tables draw partition program work
    stratified closure publicInputs).go 0).2

theorem Stmt.setupLogged_fst (h : ByteArray → ByteArray) (tags : Tags) (circuitFile publicFile : ByteArray)
    (coins : Coins) (count : Nat) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) (log : Array ByteArray) :
    (Stmt.setupLogged h tags circuitFile publicFile coins count tables draw partition program work stratified closure
        publicInputs log).1
      = Stmt.setupTablesWith (M := V) (Hashes.ofSha512 h) tags circuitFile publicFile coins count tables draw partition
          program work stratified closure publicInputs :=
  With.setupTablesWith_sim (With.Sim.forget _) With.Throws.exceptT With.Throws.v (With.Sim.forget_logged h) tags
    circuitFile publicFile coins count tables draw partition program work stratified closure publicInputs log

theorem Stmt.setupCounted_fst (h : ByteArray → ByteArray) (tags : Tags) (circuitFile publicFile : ByteArray)
    (coins : Coins) (count : Nat) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) (n : Nat) :
    ((Stmt.setupTablesWith (Hashes.counted h) tags circuitFile publicFile coins count tables draw partition program work
        stratified closure publicInputs).go n).1
      = Stmt.setupTablesWith (M := V) (Hashes.ofSha512 h) tags circuitFile publicFile coins count tables draw partition
          program work stratified closure publicInputs :=
  With.setupTablesWith_sim (With.Sim.forget _) With.Throws.exceptT With.Throws.v (With.Sim.forget_counted h) tags
    circuitFile publicFile coins count tables draw partition program work stratified closure publicInputs n

theorem Stmt.setupLogged_size (h : ByteArray → ByteArray) (tags : Tags) (circuitFile publicFile : ByteArray)
    (coins : Coins) (count : Nat) (tables : List (String × ByteArray)) (draw : Option Json) (partition : Option Json)
    (program : Option ByteArray) (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    (Stmt.setupLogged h tags circuitFile publicFile coins count tables draw partition program work stratified closure
        publicInputs).2.size
      = Stmt.setupCalls h tags circuitFile publicFile coins count tables draw partition program work stratified closure
          publicInputs :=
  (With.setupTablesWith_sim With.Sim.logCount With.Throws.exceptT With.Throws.exceptT (With.Sim.logCount_logged_counted h)
    tags circuitFile publicFile coins count tables draw partition program work stratified closure publicInputs #[]).2

end Flock
