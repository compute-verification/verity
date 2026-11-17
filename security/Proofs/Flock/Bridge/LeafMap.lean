module

public import Proofs.Flock.Bridge.Canon
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! The deployed commitment at a leaf map, where a public file's `shared_rows` tables register a port over its table's
rows. -/

@[expose] public section

namespace Flock.Bridge

open Claims Claims.Sampling
open Properties.Certifier.Bytes (bs)

/-- **A registration's leaf map**: per port `q` its leaf count `cnt q`, unit `u`'s leaf `ref q u` there, leaf `r`'s
`b ‖ c` digest `dig q r`, the path levels `d` an opening's data reserves before the row's bits, and port `q`'s frame-v3
domain `dom q`. -/
structure LeafMap where
  cnt : ℕ → ℕ
  ref : ℕ → ℕ → ℕ
  dig : ℕ → ℕ → ByteArray
  d : ℕ
  dom : ℕ → ByteArray

/-- **One leaf per unit**: `nl` leaves at every port, unit `u` at leaf `u`, the public file's digest there, under the
published domains. -/
def LeafMap.ident (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : LeafMap :=
  ⟨fun _ => nl, fun _ u => u, fun q u => pub.digest u q c.ports.size, Flock.log2ceil nl, domOf c pub⟩

/-- **A `shared_rows` file's leaf map**: port `q` has its table's `counts[q]` rows, at the digests from `starts[q]`;
unit `u` reads row `refs[u · np + q]`; the data reserves the deepest table's levels; the domains are the published
ones. -/
def LeafMap.ofTables (c : Flock.Circuit) (pub : Flock.Public) (t : Flock.Tables) : LeafMap :=
  ⟨fun q => t.counts.getD q 0, fun q u => t.refs.getD (u * c.ports.size + q) 0,
    fun q r => pub.digests.getD (t.starts.getD q 0 + r) ByteArray.empty,
    (List.range c.ports.size).foldr (fun q m => max (Flock.log2ceil (t.counts.getD q 0)) m) 0, domOf c pub⟩

/-- **The public file's leaf map**: its tables', else one leaf per unit. -/
def LeafMap.ofPublic (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) : LeafMap :=
  match pub.tables with
  | some t => LeafMap.ofTables c pub t
  | none => LeafMap.ident c pub nl

theorem LeafMap.ofPublic_dom (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) :
    (LeafMap.ofPublic c pub nl).dom = domOf c pub := by
  unfold LeafMap.ofPublic
  cases pub.tables <;> rfl

theorem LeafMap.ofPublic_none {c : Flock.Circuit} {pub : Flock.Public} (nl : ℕ) (h : pub.tables = none) :
    LeafMap.ofPublic c pub nl = LeafMap.ident c pub nl := by
  unfold LeafMap.ofPublic; rw [h]

theorem LeafMap.ofPublic_some {c : Flock.Circuit} {pub : Flock.Public} {t : Flock.Tables} (nl : ℕ)
    (h : pub.tables = some t) : LeafMap.ofPublic c pub nl = LeafMap.ofTables c pub t := by
  unfold LeafMap.ofPublic; rw [h]

theorem LeafMap.dig_ref (c : Flock.Circuit) (pub : Flock.Public) (nl q u : ℕ) :
    (LeafMap.ofPublic c pub nl).dig q ((LeafMap.ofPublic c pub nl).ref q u) = pub.digest u q c.ports.size := by
  unfold LeafMap.ofPublic
  cases h : pub.tables with
  | none => rfl
  | some t =>
    have e : pub.digest u q c.ports.size =
        pub.digests.getD (t.starts.getD q 0 + t.refs.getD (u * c.ports.size + q) 0) ByteArray.empty := by
      simp only [Flock.Public.digest, h]
    exact e.symm

/-- Port `q`'s leaves at `L`: leaf `r` is the frame-v3 leaf, at the port's schema (`schemaOf`) and domain (`L.dom q`),
of the tree leaf of `L.dig q r`. -/
def leavesL (c : Flock.Circuit) (L : LeafMap) (q : ℕ) : Array ByteArray :=
  (Array.range (L.cnt q)).map fun r =>
    Flock.Registered.leafAs (schemaOf c q) (L.dom q) r (Flock.Hm96.Default512.treeLeaf (L.dig q r))

/-- **The registration's frame-v3 roots at a leaf map**: per port, the root over its `L.cnt q` leaves. -/
def rootsOfL (c : Flock.Circuit) (L : LeafMap) : List Msg :=
  (List.range c.ports.size).map fun q => bs (Flock.FV3.root (L.dom q) (leavesL c L q) Sha512.hash)

theorem rootsOfL_ident (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) :
    rootsOfL c (LeafMap.ident c pub nl) = rootsOf c pub nl := rfl

/-- **The registered ports at a leaf map**: `portsOf` with port `q`'s `L.cnt q` leaves, each at its own schema and
`L`'s domain. -/
def portsOfL (c : Flock.Circuit) (L : LeafMap) (roots : List Msg) : Array Flock.Registered.Port :=
  (Array.range c.ports.size).map fun q =>
    { name := (c.ports[q]?.map (·.name)).getD "", value := "", root := bytes (roots.getD q []), leaves := L.cnt q,
      words := none, wordBits := 0, domain := L.dom q, readsRel := #[], schema := schemaOf c q }

theorem portsOfL_ident (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) (roots : List Msg) :
    portsOfL c (LeafMap.ident c pub nl) roots = portsOf c pub nl roots := rfl

/-- The port index of the row at claims position `i` (`0` where none). -/
def portIdxOf (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (i : ℕ) : ℕ :=
  ((rowOfPos C rows i).map (·.port)).getD 0

/-- **An opening's path at a leaf map**: its position's port's levels, after the salt. -/
def pathOfL (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (L : LeafMap) (o : Opening Bool) :
    Msg :=
  pathOf (Flock.log2ceil (L.cnt (portIdxOf C rows o.pos))) o

theorem pathOfL_ident (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (pub : Flock.Public) (nl : ℕ) : pathOfL C rows (LeafMap.ident c pub nl) = pathOf (Flock.log2ceil nl) := rfl

/-- **The deployed trace commitment at a leaf map**: `commitDep` at `L`'s ports, each position's path at its port's
levels, the row's bits after `L.d` levels, and each row in its port's format (`rowDep`). -/
noncomputable def commitDepL (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (L : LeafMap) (roots : List Msg) : CommitmentScheme Bool :=
  pathCommit (atOf C rows) (hm96Leaf (rowDep C rows c (placeOf C rows) (bitsOf L.d)) saltOf)
    (pathOfL C rows L) (regPath (portsOfL c L roots)) fun _ _ _ => []

theorem commitDepL_ident (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (pub : Flock.Public) (nl : ℕ) (roots : List Msg) :
    commitDepL C rows c (LeafMap.ident c pub nl) roots = commitDep C rows c pub nl roots := rfl

theorem commitDepL_binds (H : Msg → Msg) (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt)
    (c : Flock.Circuit) (hseg : c.ports.all (fun p => !p.segs.isSome) = true) (L : LeafMap) (roots : List Msg) :
    (commitDepL C rows c L roots).Binds H :=
  pathCommit_binds (hm96Leaf_binds H (rowDep_places C rows c hseg _ _) _) (regPath_binds H _)

theorem schemaOf_v2 {c : Flock.Circuit} (h : RowsV2 c) {q : ℕ} (hq : q < c.ports.size) :
    schemaOf c q = Flock.HmRow.ROW_SCHEMA_V2 := by
  have hpq := (Array.all_eq_true.mp h) q hq
  simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at hpq
  simp [schemaOf, hq, Flock.Registered.portSchema, hpq.1, hpq.2]

theorem rowDep_v2 (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) {c : Flock.Circuit}
    (h : RowsV2 c) (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) :
    rowDep C rows c place bits = SamplingDeploy.row place bits := by
  funext o
  unfold rowDep
  split
  · rename_i p hp
    obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.mp hp
    obtain ⟨hq, rfl⟩ := Array.getElem?_eq_some_iff.mp hr
    have hpq := (Array.all_eq_true.mp h) r.port hq
    simp only [Bool.and_eq_true, Option.isNone_iff_eq_none] at hpq
    simp [rowMsg, hpq.1, hpq.2, SamplingDeploy.row]
  · rfl

theorem leavesL_v2 {c : Flock.Circuit} (h : RowsV2 c) (L : LeafMap) {q : ℕ} (hq : q < c.ports.size) :
    leavesL c L q = (Array.range (L.cnt q)).map fun r =>
      Flock.Registered.leafAs Flock.HmRow.ROW_SCHEMA_V2 (L.dom q) r (Flock.Hm96.Default512.treeLeaf (L.dig q r)) := by
  simp only [leavesL, schemaOf_v2 h hq]

theorem commitDepL_v2 (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) {c : Flock.Circuit}
    (h : RowsV2 c) (L : LeafMap) (roots : List Msg) :
    commitDepL C rows c L roots =
      pathCommit (atOf C rows) (hm96Leaf (SamplingDeploy.row (placeOf C rows) (bitsOf L.d)) saltOf)
        (pathOfL C rows L) (regPath (portsOfL c L roots)) fun _ _ _ => [] := by
  unfold commitDepL
  rw [rowDep_v2 C rows h]

def LayoutIsL (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (L : LeafMap)
    (roots : List Msg) (cd : SamplingDeploy.Code) : Prop :=
  RowsV2 c ∧ cd.at_ = atOf C rows ∧ cd.place = placeOf C rows ∧ cd.bits = bitsOf L.d ∧ cd.salt = saltOf ∧
    cd.π = pathOfL C rows L ∧ cd.own = portsOfL c L roots ∧ cd.ownAt = fun _ _ => cd.own

theorem layoutIsL_ident (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (pub : Flock.Public) (nl : ℕ) (roots : List Msg) (cd : SamplingDeploy.Code) :
    LayoutIsL C rows c (LeafMap.ident c pub nl) roots cd ↔ LayoutIs C rows c pub nl roots cd := Iff.rfl

def RootFixesDomainL (c : Flock.Circuit) (L : LeafMap) : Prop :=
  ∀ q q', q' < q → q < c.ports.size → (rootsOfL c L).getD q' [] = (rootsOfL c L).getD q [] → L.dom q' = L.dom q

theorem rootFixesDomainL_ident (c : Flock.Circuit) (pub : Flock.Public) (nl : ℕ) :
    RootFixesDomainL c (LeafMap.ident c pub nl) ↔ PortRoots.RootFixesDomain c pub nl := Iff.rfl

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.ZkReg (rsZHJS rsZHJS_computes)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
def _root_.Flock.Bridge.TransportsLiveCR : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.PortRoots.dupId D.unroll pub = false)
    (_hroots : Bridge.RootFixesDomainL D.unroll (Bridge.LeafMap.ofPublic D.unroll pub n))
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))),
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub).bindingZ
      (rsZHJS_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => sampleOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let L := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, L.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    let roots := Bridge.rootsOfL D.unroll L
    let Cd := Bridge.commitDepL Cc rows D.unroll L roots
    ∀ u, ∀ g ∈ P.io u, κ g = none → ∀ o : vb.Opening, vb.com o = vb.cs (reg σ) (vb.pos g) →
      Bridge.canonOpen rows g (vb.val o) o.2.2 →
      ∃ d : Claims.Msg, Cd.Opens Claims.Sampling.sha512Msg roots ⟨Bridge.posOf Cc g, vb.bit (vb.val o) g, d⟩ = true

end Flock.Guarantees
