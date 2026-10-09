import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Statements
import Proofs.Flock.Soundness.Registered.Binding

/-!
# Protocol 2 at the universal unit: the instance's objects

The reduction layer (`Defs`, `Statements`) is generic in the scheme, the program commitment and protocol 1's bad event.
These are the objects it is instantiated at.

* **The program commitment** is a registered value's root (`Flock.Registered.Port`, PROTOCOL.md §16.10): the program is
  `L` rows of a registered value, each an `hm96-sha512/row/v1` leaf at its position. `progCommit L` commits to the
  program's leaves (`Opens`: each leaf opens the root at its position, `Flock.Registered.opensLeaf`), and its extractor is
  the registered reads' (`TwoOpenings.pair`) at the first position where two openings' leaves differ. `rowCommit L enc`
  commits to the rows themselves, through hm96's leaf (`ExplicitLeaf.hm96`): its extractor is hm96's (`hm96Pair`) where two
  rows have one leaf, else the registered reads'.
* **The program's sites** (`ProgSites`): the `L` commit strings of the universal unit's layout at which the session reads
  the program, each a registered read (`Reads`) of the root `rt` at the program's own position. `opening R path` is the
  session's opening of the program under the registration `R` (its leaves `lay.regLeaf R`, its paths), and `regOpening`
  the registrant's (`Reads.leaf`, `Reads.wpath`).
* **The registration's view** (`hm96Reg`): the commit strings `b‖c` of the program's rows, `b = x ⊕ M·y` and
  `c = SHA-512(sp ‖ y)` for each row's digest `x` (`rowBits`, `SHA-512` of the row's bytes) under a uniform salt `y`.
  The root, its paths and every registered read's `b‖c` that a session reveals are functions of them. `hm96Scheme` is
  `Scheme` at that registration.
* **Protocol 1's headlines at the universal unit, as Props.** `ZkSessionSound s` is `zk_session_sound`'s conclusion for
  the session `s` of the universal unit's partition (`UnitSession`), and `ZkSessionView T w δ₁` is `zk_session_view`'s for the
  tables `T` at the honest witness `w`. They are hypotheses of the instance's theorems until those headlines land at the
  universal unit.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

/-! ## The program commitment -/

/-- **An opening of a program's leaves** under a registered root: the root (`port`), and at each of the program's `L`
positions its leaf and path. -/
structure ProgOpening (L : ℕ) where
  port : Flock.Registered.Port
  leaf : Fin L → Dig
  path : Fin L → ByteArray

open Classical in
/-- The first position where two openings' leaves differ, if any. -/
noncomputable def ProgOpening.diff {L : ℕ} (op op' : ProgOpening L) : Option (Fin L) :=
  if h : ∃ i, op.leaf i ≠ op'.leaf i then some (Classical.choose h) else none

/-- The two openings at position `i`, as the registered reads' `TwoOpenings`. -/
def ProgOpening.at {L : ℕ} (op op' : ProgOpening L) (i : Fin L) : TwoOpenings :=
  ⟨op.port, i, op.leaf i, op.path i, op'.leaf i, op'.path i⟩

/-- **The program commitment of a registered root**: `Opens rt p op` when `op` is an opening under `rt` whose leaves are
`p`, each opening `rt` at its position (`Flock.Registered.opensLeaf`); `ext` is the registered reads' extractor
(`TwoOpenings.pair`) at the first position where the two openings' leaves differ. -/
noncomputable def progCommit (L : ℕ) : Commit Flock.Registered.Port (Fin L → Dig) (ProgOpening L) where
  Opens rt p op := op.port = rt ∧ op.leaf = p ∧ ∀ i : Fin L, Flock.Registered.opensLeaf rt (p i).1 i (op.path i) = true
  ext op op' := match op.diff op' with
    | some i => (op.at op' i).pair
    | none => ([], [])

/-- **An opening of a program's rows** under a registered root: the root, and at each position the row, its salt and
its path. -/
structure RowOpening (L : ℕ) (Val : Type) where
  port : Flock.Registered.Port
  row : Fin L → Val
  salt : Fin L → List UInt8
  path : Fin L → ByteArray

/-- The opening of the rows' hm96 leaves. -/
def RowOpening.toProg {L : ℕ} {Val : Type} (lf : ExplicitLeaf Val) (op : RowOpening L Val) : ProgOpening L :=
  ⟨op.port, fun i => lf.leaf (op.row i) (op.salt i), op.path⟩

open Classical in
/-- The first position where two openings' rows differ, if any. -/
noncomputable def RowOpening.diff {L : ℕ} {Val : Type} (op op' : RowOpening L Val) : Option (Fin L) :=
  if h : ∃ i, op.row i ≠ op'.row i then some (Classical.choose h) else none

open Classical in
/-- **The row commitment of a registered root under hm96**: `Opens rt p op` when `op` is an opening under `rt` of the rows
`p`, each row's hm96 leaf under its salt opening `rt` at its position. `ext`, at the first position where the rows
differ: hm96's extractor (`ExplicitLeaf.pair`, `hm96Pair`) when the two leaves agree, else the registered reads'. -/
noncomputable def rowCommit (L : ℕ) {Val : Type} (enc : Val → List UInt8) (hinj : Function.Injective enc) :
    Commit Flock.Registered.Port (Fin L → Val) (RowOpening L Val) where
  Opens rt p op := op.port = rt ∧ op.row = p ∧
    ∀ i : Fin L, Flock.Registered.opensLeaf rt ((ExplicitLeaf.hm96 enc hinj).leaf (p i) (op.salt i)).1 i (op.path i) = true
  ext op op' := match op.diff op' with
    | some i =>
      if (ExplicitLeaf.hm96 enc hinj).leaf (op.row i) (op.salt i) = (ExplicitLeaf.hm96 enc hinj).leaf (op'.row i) (op'.salt i)
      then (ExplicitLeaf.hm96 enc hinj).pair (op.row i) (op.salt i) (op'.row i) (op'.salt i)
      else ((op.toProg (ExplicitLeaf.hm96 enc hinj)).at (op'.toProg (ExplicitLeaf.hm96 enc hinj)) i).pair
    | none => ([], [])

/-! ## The program's sites among the registered reads -/

section Sites

variable {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay}

/-- **The program's sites**: the `L` commit strings at which the session reads the program, each a registered read
(`Reads.isRead`) of the root `rt` at the program's own position. -/
structure ProgSites (rd : Registered.Reads lay rs) (rt : Flock.Registered.Port) (L : ℕ) where
  site : Fin L → lay.Pos
  isRead : ∀ i, rd.isRead (site i)
  port : ∀ i, rd.port (site i) = rt
  readAt : ∀ i, rd.readAt (site i) = i

namespace ProgSites

variable {rd : Registered.Reads lay rs} {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)

/-- The session's program leaves under the registration `R`: the layout's leaves at the program's sites. -/
def leaves (R : Reg) : Fin L → Dig := fun i => lay.regLeaf R (ps.site i)

/-- The session's opening of the program under the registration `R`, through `path`. -/
def opening (R : Reg) (path : lay.Pos → ByteArray) : ProgOpening L :=
  ⟨rt, ps.leaves R, fun i => path (ps.site i)⟩

/-- The registrant's program leaves: hm96's leaves of the rows and salts it committed. -/
def regLeaves : Fin L → Dig := fun i => rd.leaf (ps.site i)

/-- The registrant's opening of the program. -/
def regOpening : ProgOpening L := ⟨rt, ps.regLeaves, fun i => rd.wpath (ps.site i)⟩

end ProgSites

end Sites

/-! ## The registration's view -/

/-- A row's digest as the 512 bits of an hm96 commit string's `b` half: `SHA-512` of the row's bytes. -/
def rowBits (bytes : List UInt8) : Bits := bitsOf (Sha512.hash ⟨bytes.toArray⟩)

/-- **The hm96 registration of an `L`-row program**, as the verifier sees it: the commit strings `b‖c` of its rows'
digests `x p i` under the registrant's uniform salts `r i`, `b = x ⊕ M·y` (`My` at the pinned key) and `c = SHA-512(sp ‖ y)`
(`saltHash`). -/
def hm96Reg {Prg : Type} {L : ℕ} (x : Prg → Fin L → Bits) (p : Prg) (r : Fin L → Salt) : Fin L → Bits × ByteArray :=
  fun i => (x p i + My (r i), saltHash (r i))

/-- **A private-circuit scheme at the universal unit, registered under hm96**: the program of class `c` is `L c` rows,
row `i` of program `p` having the bytes `enc c p i`; the registrant's salts are uniform; the registration's view is
`hm96Reg` of the rows' digests; and the universal unit's `--zk` tables `U c o v` read the class, the owed outputs and that
view only. -/
def hm96Scheme {Cls Circ Out : Type} {Prog : Cls → Type} {K W Hi VL : Type} [Field K] (fits : Circ → Cls → Prop)
    (pad : Circ → (c : Cls) → Prog c) (L : Cls → ℕ) (enc : (c : Cls) → Prog c → Fin (L c) → List UInt8)
    (J : Cls → ℕ) (zs : (c : Cls) → Fin (J c) → Sizes)
    (U : (c : Cls) → Out → (Fin (L c) → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs c j)) :
    Scheme Cls Circ Out Prog (fun c => Fin (L c) → Salt) (fun c => Fin (L c) → Bits × ByteArray) K W Hi VL where
  fits := fits
  pad := pad
  reg c := hm96Reg fun p i => rowBits (enc c p i)
  J := J
  zs := zs
  U := U

/-! ## Protocol 1's headlines at the universal unit, as Props -/

/-- **Protocol 1's sampled `--zk` session of a partition `P`**, as `zk_session_sound` reads it. The live game `G`
(`auditLive` at the law's draws) and flock-verify's acceptance (`LiveAcceptsZK`); the registration each outcome records,
as the model's leaves read it (`regOf`); the law `Lm` whose `miss` bounds the sampling term (`Law.stratified`), over `P`'s
`n` units; the committed values the binding extracts from the prover (`vals τ`: `XplurZC` at `simZ … τ`), whose wrong units
(`P.wrong`) the headline counts; its knowledge-soundness and link terms (`εks`: `ksAvgStrictZ`, `δlink`: `linkBoundZC`,
`ecr/sha-512` inside); `CR τ`, `cr/sha-512` for the knowledge finders of `τ`'s simulation (`TableCRZ` at every draw and
table); A3's byte source (`Lb`, `src`, `Assumptions.UniformRandomBytes`); and A6, the coin server's record of the live
session and its draw at each of the law's draws (`RecordsLiveZK`, `DrawOkZK`). -/
structure UnitSession {C : Audit.Circuit} {n : ℕ} (P : Audit.Partition C n) (Reg : Type) where
  Out : Type
  G : Game Out
  Accepts : Out → Prop
  regOf : Out → Reg
  Lm : Audit.Law n
  vals : Strategy G → Out → Fin C.N → Bool
  εks : Strategy G → ℝ≥0∞
  δlink : Strategy G → ℝ≥0∞
  CR : Strategy G → Prop
  m : ℕ
  Ω : Type
  [finΩ : Fintype Ω]
  Lb : Fin m → ℕ
  src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256
  A6 : Prop

attribute [instance] UnitSession.finΩ

/-- **`zk_session_sound` at a session `s`**, as a Prop: under A3 and A6, for every live prover `τ` whose knowledge finders
meet `cr/sha-512`, accepting with at least `K₀` wrong units of `s`'s partition has probability at most
`miss(K₀) + ε_ks + δ_link`. -/
def ZkSessionSound {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} (s : UnitSession P Reg) : Prop :=
  Assumptions.UniformRandomBytes s.Lb s.src → s.A6 → ∀ τ : Strategy s.G, s.CR τ → ∀ K₀ : ℕ,
    prob (fun o => s.Accepts o ∧ K₀ ≤ (P.wrong (s.vals τ o)).card) s.G τ ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ

/-- **`zk_session_view` at the tables `T` and the honest witness `w`**, as a Prop: for any dummy witness `w₀`, encoding
`enc` of the tables' views and event `E`, the verifier's view at `w` (every hidden leaf the executable's,
`Coined.viewExe`) and `S_shvzk`'s at `w₀` (`Coined.simExe`) are within `∑ⱼ 2·|Hidⱼ|·δ₁`, both ways. -/
def ZkSessionView {K W Hi VL : Type} [Field K] [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] {J : ℕ}
    {zs : Fin J → Sizes} (T : ∀ j, Coined K W Hi VL (zs j)) (w : Fin J → W) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (w₀ : Fin J → W) {Obs : Type} (enc : (∀ j, (T j).tab.View) → Obs) (E : Obs → Prop),
    prCoin (fun p : ∀ j, (T j).tab.Rand × ((T j).tab.Hid → Salt) => E (enc fun j => (T j).viewExe (w j) (p j).1 (p j).2))
        ≤ prCoin (fun p : ∀ j, (T j).tab.SimRand × ((T j).tab.Hid → Salt) =>
            E (enc fun j => (T j).simExe (w₀ j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card (T j).tab.Hid * δ₁
      ∧ prCoin (fun p : ∀ j, (T j).tab.SimRand × ((T j).tab.Hid → Salt) =>
          E (enc fun j => (T j).simExe (w₀ j) (p j).1 (p j).2))
        ≤ prCoin (fun p : ∀ j, (T j).tab.Rand × ((T j).tab.Hid → Salt) =>
            E (enc fun j => (T j).viewExe (w j) (p j).1 (p j).2))
          + ∑ j, 2 * Fintype.card (T j).tab.Hid * δ₁

/-- `ZkSessionView`'s bound at the sizes `zs`: `∑ⱼ 2·(nU + 2·nPU)·δ₁`. -/
noncomputable def viewGap {J : ℕ} (zs : Fin J → Sizes) (δ₁ : ℝ≥0∞) : ℝ≥0∞ :=
  ∑ j, 2 * (((zs j).nU + 2 * (zs j).nPU : ℕ) : ℝ≥0∞) * δ₁

end FlockSoundness.Discharge.PrivateCircuit
