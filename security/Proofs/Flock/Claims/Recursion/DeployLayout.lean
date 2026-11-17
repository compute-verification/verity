module

public import Proofs.Flock.Claims.Recursion.DeployReg
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
import Proofs.Flock.Claims.Recursion.KSStated

/-! The V\* layout read off the registered ports (`layoutOf`) and the recursion's KS at it; it reads no part's SHA-512
calls, and leaves `ReadsPub` and `FinalOpens` open. -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- One rep stream's rounds at the verifier's inner shape: `RecVStar.repRounds`, or `holoRepRounds` over a holo inner
session (`RecVStar.innerMsgRounds`' schedule). -/
def repSched (sh : RecVStar.Inner) : Option (Array RecVStar.SchedRound) :=
  (if sh.holo then RecVStar.holoRepRounds else RecVStar.repRounds) sh.m sh.kLog sh.claims

/-- Rep `r`'s `k`-th message round as a round of the rep streams that follow the link message: rep `r`'s stream after
the earlier reps', at its `k`-th round that absorbs a message (`RecVStar.commitRounds`). -/
def msgRound (rs : Array RecVStar.SchedRound) (r k : ℕ) : Option ℕ :=
  ((RecVStar.commitRounds rs)[k]?).map (r * rs.size + ·)

/-- The input column of row bit `i` of the circuit's port `q` in unit `u`: bit `i % 16` of the unit input leaf `k` that
reads the port's word `i / 16` (`leavesIn`), the ports' words in order, leaf `k` from column `col · WORD + 16 k` of the
unit's first input group (`Flat.leafCol`). -/
def portCol (c : Circuit) (u q i : ℕ) : Option ℕ := do
  let w : ℕ := (c.ports.extract 0 q).foldl (· + ·.words) 0 + i / 16
  let k ← (c.leavesIn.getD u #[]).findIdx? (· == (w : Int))
  pure ((c.unit.inGroups.getD 0 default).col * WORD + 16 * k + i % 16)

/-- **The hidden values of port `nm`'s row at instance `n`** of a part's one table: each row bit's input column
(`portCol`) in the instance's copy (block `n / g`, VU `n % g`, its first unit: `Circuit.instOf`), at
`Readable.copyCol`'s index for one table. -/
def rowHidden (c : Circuit) (nm : String) (n : ℕ) : Option (List ℕ) := do
  let q ← c.ports.findIdx? (·.name == nm)
  let p ← c.ports[q]?
  let base := c.unitRange.slot (n % c.g * c.upv)
  (List.range p.bits).mapM fun i => (portCol c 0 q i).map fun col => n / c.g * 2 ^ c.kLog + (base + col)

/-- **What part `s` holds**, off its registered message-round ports (`RecVStar.ownPorts`): for each port under
`rec-c<k>` and each row `r` it reads, rep `r`'s `k`-th message round as a round of the rep streams (`msgRound`), with
the port's row's hidden values at instance `r` (`rowHidden`). -/
def partHolds (r : XR) (s : ℕ) : List (ℕ × List ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (fun vs => vs[s]?), r.w.parts[s]?.bind partCircuit,
      repSched r.shape with
  | some st, some c, some rs =>
    (RecVStar.ownPorts st).flatMap fun q => match RecVStar.cIndex q.2.1 with
      | some k => q.2.2.toList.filterMap fun row => do
        let rep ← row
        let j ← msgRound rs rep k
        let ps ← rowHidden c q.1 rep
        pure (j, ps)
      | none => []
  | _, _, _ => []

/-- **The rep streams' rounds whose message is constant**, at the verifier's inner shape: round `j` of the two reps'
streams (rep `j / R`'s round `j % R`, of `R`) where that round absorbs no message (`RecVStar.SchedRound`'s flag), whose
framed message is the frame alone, `[0]` (`FlockDeploy.act`'s tag `0`, no bytes). -/
def streamFixed (x : Msg) (j : ℕ) : Option Msg :=
  (readX x).bind fun r => (repSched r.shape).bind fun rs =>
    if j < 2 * rs.size ∧ (rs.getD (j % rs.size) default).2.2.2 = false then some [0] else none

/-- **The V\* layout off the registered ports**: the deployed plain verifier's inner rounds; at round `0`, the link
message, public (the prover sends it in the clear) and no part's; at round `j + 1`, the rep streams' round `j`: the
frame `[0]` where it absorbs no message (`streamFixed`), else the frame `0` then the row of part `s`'s first port
holding it (`partHolds`); no SHA-512 evaluation. -/
def layoutOf : Claims.Recursion.VStar.Layout where
  rounds x := (FlockDeploy.V false).k (xIOf x)
  msgAt x s i := match i with
    | 0 => none
    | j + 1 => (readX x).bind fun r => ((partHolds r s).find? (·.1 == j)).map fun p => [([0], p.2)]
  shaAt _ _ := []
  fixedAt x i := match i with
    | 0 => none
    | j + 1 => streamFixed x j
  pubAt _ i := i == 0

theorem layoutOf_msgAt_zero (x : Msg) (s : ℕ) : layoutOf.msgAt x s 0 = none := rfl

theorem layoutOf_pubAt_zero (x : Msg) : layoutOf.pubAt x 0 = true := rfl

theorem layoutOf_msgAt_succ (x : Msg) (s j : ℕ) : layoutOf.msgAt x s (j + 1) =
    (readX x).bind fun r => ((partHolds r s).find? (·.1 == j)).map fun p => [(([0] : Msg), p.2)] := rfl

theorem layoutOf_fixedAt_succ (x : Msg) (j : ℕ) : layoutOf.fixedAt x (j + 1) = streamFixed x j := rfl

theorem layoutOf_rounds_pos (x : Msg) : 0 < layoutOf.rounds x := by
  show 0 < (FlockDeploy.V false).k (xIOf x)
  simp only [FlockDeploy.V]
  split <;> omega

def SchedMatches (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∃ r rs, readX x = some r ∧ repSched r.shape = some rs ∧
    (FlockDeploy.V false).k (xIOf x) = 2 + 2 * rs.size

def PortsCover (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ r rs, readX x = some r → repSched r.shape = some rs →
    ∀ rep < 2, ∀ t ∈ RecVStar.commitRounds rs, ∃ s < nS x, ∃ ps, (rep * rs.size + t, ps) ∈ partHolds r s

theorem heldPub_layoutOf {W : Type} (I : Claims.Recursion.Inner Coin W) (hI : I.V = FlockDeploy.V false)
    (ivScope : Msg → Bool) (b : Brings) (shape : Claims.Recursion.Form → Prop) (hM : SchedMatches shape)
    (hC : PortsCover shape) : Claims.Recursion.VStar.HeldPub (deployRec I.V ivScope b) I layoutOf shape := by
  intro x st hst hf
  obtain ⟨r, rs, hr, hrs, hk⟩ := hM x st hst hf
  refine ⟨?_, fun i hi => ?_⟩
  · show I.V.k (xIOf x) ≤ (FlockDeploy.V false).k (xIOf x)
    rw [hI]
  change i + 1 < I.V.k (xIOf x) at hi
  rw [hI, hk] at hi
  cases i with
  | zero => exact Or.inl rfl
  | succ j =>
    right
    have hj : j < 2 * rs.size := by omega
    have hpos : 0 < rs.size := by omega
    by_cases hab : (rs.getD (j % rs.size) default).2.2.2 = false
    · left
      rw [layoutOf_fixedAt_succ, streamFixed, hr, Option.bind_some, hrs, Option.bind_some]
      simp only [hj, hab, and_self, ↓reduceIte, Option.isSome_some]
    · right
      have ht : j % rs.size ∈ RecVStar.commitRounds rs := by
        simp only [RecVStar.commitRounds, List.mem_filter, List.mem_range]
        exact ⟨Nat.mod_lt _ hpos, by simpa using hab⟩
      obtain ⟨s, hs, ps, hmem⟩ :=
        hC x st hst hf r rs hr hrs (j / rs.size) ((Nat.div_lt_iff_lt_mul hpos).2 (by omega)) _ ht
      rw [Nat.div_add_mod' j rs.size] at hmem
      refine ⟨s, hs, ?_⟩
      rw [layoutOf_msgAt_succ, hr, Option.bind_some]
      cases hfind : (partHolds r s).find? (·.1 == j) with
      | none => exact absurd (List.find?_eq_none.1 hfind _ hmem) (by simp)
      | some p => rfl

def Brings.atPorts (b : Brings) : Brings := { b with rd := readersOf layoutOf }

theorem readsThrough_atPorts (H : Msg → Msg) (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings) :
    Claims.Recursion.VStar.ReadsThrough H (deployRec IV ivScope b.atPorts) IV layoutOf.msgs layoutOf.calls :=
  readsThrough_deployRec H IV ivScope b.atPorts layoutOf rfl

theorem vstarLen_atPorts (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings) :
    Claims.Recursion.VStar.VstarLen (deployRec IV ivScope b.atPorts) :=
  vstarLen_deployRec IV ivScope b.atPorts

theorem partCallsTrue_layoutOf {Col T : Type} (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin)
    (DO : Claims.Flock.Deploy Coin Col T) (s : ℕ) : Claims.Recursion.VStar.PartCallsTrue H D DO layoutOf s := by
  intro x st _ xO w _ e he
  simp [Claims.Recursion.VStar.Layout.calls, layoutOf] at he

/-- `ReadsPub` stays open, as the readers off the witnesses alone don't read the link message; so does `FinalOpens`, an
opening of the last message. -/
theorem ksA_deployed_concrete (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI)
    (hPub : Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf)
    (hFin : Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) opening)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) :=
  ksA_deployed_parts rel relO ivScope b.atPorts layoutOf rfl opening hF htab decI hdec
    (partCallsTrue_layoutOf _ _ _) hPub hFin hD hH hS

/-- No theorem evaluates its level (`Claims.Recursion.lam`) at the deployment. -/
theorem ksFlock_deployed_concrete (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation)
    (ivScope : Msg → Bool) (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hPub : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf)
    (hFin : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) opening)
    (hD : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Dense)
    (hH : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Holo)
    (hS : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec relO) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec =>
    ksA_deployed_concrete rel relO ivScope b opening hF htab decI hdec (hPub decI hdec) (hFin decI hdec)
      (hD decI hdec) (hH decI hdec) (hS decI hdec)

end RecDeploy
