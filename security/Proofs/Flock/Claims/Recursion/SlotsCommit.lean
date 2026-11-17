module

public import Proofs.Flock.Claims.Recursion.FinOpen
public import Proofs.Flock.Claims.Recursion.RecOpen
public import Proofs.Flock.Claims.Recursion.RecRecord
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Soundness.Discharge.Hm.Commit
import Proofs.Flock.Soundness.Discharge.Hm.Mask
import Std.Tactic.BVDecide.Normalize.BitVec

/-! `ReadersTrue` at `layoutBTops`, from the hm96 slots' commit strings (`readersTrue_layoutBTops`). -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB (readForms formBit)
open FlockSoundness.Discharge.Hm (bitL)

theorem testBit_bitsNat : ∀ (bs : List Bool) (i : ℕ), (Claims.Recursion.VStar.bitsNat bs).testBit i = bs.getD i false
  | [], i => by simp [Claims.Recursion.VStar.bitsNat]
  | b :: bs, 0 => by
      show ((if b then 1 else 0) + 2 * Claims.Recursion.VStar.bitsNat bs).testBit 0 = b
      cases b <;> simp [Nat.testBit_zero]
  | b :: bs, i + 1 => by
      show ((if b then 1 else 0) + 2 * Claims.Recursion.VStar.bitsNat bs).testBit (i + 1) = bs.getD i false
      rw [Nat.testBit_succ, show ((if b then 1 else 0) + 2 * Claims.Recursion.VStar.bitsNat bs) / 2 =
        Claims.Recursion.VStar.bitsNat bs by rw [Nat.add_mul_div_left _ _ (by omega)]; cases b <;> simp,
        testBit_bitsNat bs i]

theorem bitL_bytesOf {bs : List Bool} {q : ℕ} (hq : q < bs.length) :
    bitL (Claims.Recursion.VStar.bytesOf bs) q = bs[q] := by
  unfold bitL Claims.Recursion.VStar.bytesOf
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some,
    Option.getD_some, Claims.Recursion.VStar.toNat_byte, show (256 : ℕ) = 2 ^ 8 from rfl, Nat.testBit_mod_two_pow,
    testBit_bitsNat, List.getD_eq_getElem?_getD, List.getElem?_take, List.getElem?_drop,
    show 8 * (q / 8) + q % 8 = q by omega, List.getElem?_eq_getElem hq]
  simp [Nat.mod_lt _ (show 8 > 0 by omega)]

theorem bitL_readForms {w : Claims.Flock.Wit} {fs : List HmNets.Lin} {q : ℕ} (hq : q < fs.length) :
    bitL (readForms w fs) q = formBit w fs[q] := by
  unfold readForms
  rw [bitL_bytesOf (by simpa using hq), List.getElem_map]

theorem readForms_eq_of_bits {w : Claims.Flock.Wit} {fs : List HmNets.Lin} {l : List UInt8}
    (hlen : fs.length = 8 * l.length) (h : ∀ q (hq : q < fs.length), formBit w fs[q] = bitL l q) :
    readForms w fs = l := by
  unfold readForms
  have e : fs.map (formBit w) = (List.range (8 * l.length)).map (bitL l) :=
    List.ext_getElem (by simp [hlen]) fun q h1 _ => by
      simp only [List.getElem_map, List.getElem_range]
      exact h q (by simpa using h1)
  rw [e, Claims.Recursion.VStar.bytesOf_bitL]

theorem formBit_var (w : Claims.Flock.Wit) (v : ℕ) : formBit w (HmNets.Lin.var v) = w.hid.getD v false := by
  simp [formBit, HmNets.Lin.var]

theorem foldl_xor_filter (p u : ℕ → Bool) : ∀ (l : List ℕ) (c : Bool),
    l.foldl (fun acc j => acc ^^ (p j && u j)) c = (c ^^ ((l.filter p).map u).foldr (fun b acc => b ^^ acc) false)
  | [], c => by simp
  | j :: l, c => by
      rw [List.foldl_cons, foldl_xor_filter p u l]
      cases hp : p j <;> simp [hp]

theorem maskV_filter (key : ByteArray) (y : ℕ → Bool) (i : ℕ) :
    FlockSoundness.Discharge.Hm.maskV key y i =
      (((List.range 1536).filter fun j => Hm96.bit key (i + j)).map y).foldr (fun b acc => b ^^ acc) false := by
  rw [FlockSoundness.Discharge.Hm.maskV, foldl_xor_filter, Bool.false_xor]

def rowPfx (c : Circuit) (q : ℕ) : Msg := (c.ports[q]?.map fun p => (HmRow.rowPrefix p.words).data.toList).getD []

/-- **Port `q`'s hm96 slot at instance `n` outputs its commit string**, off the witness `w`: its `b ‖ c` (`hmCommit`)
is the commit string of the row prefix and the row at port `nm`'s unit input columns (`rowHidden`), under the salt at
the slot's salt inputs (`hmSalt`). -/
def SlotCommits (c : Circuit) (nm : String) (q n : ℕ) (w : Claims.Flock.Wit) : Prop :=
  readForms w (hmCommit c q n) = FlockSoundness.Binding.commitString
    (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)) (readForms w (hmSalt c q n))

theorem slotCommits_of_bits {c : Circuit} {nm : String} {q n : ℕ} {w : Claims.Flock.Wit}
    (h : ∀ t < 1024, w.hid.getD (slotHid c n (hmBase c q n) (hmOutCol c + t)) false =
      bitL (FlockSoundness.Binding.commitString
        (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))
        (readForms w (hmSalt c q n))) t) :
    SlotCommits c nm q n w :=
  readForms_eq_of_bits (by simp [hmCommit, FlockSoundness.Discharge.Hm.length_commitString]) fun t ht => by
    simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var]
    exact h t (by simpa [hmCommit] using ht)

theorem slotCommits_of_hm96 {c : Circuit} {nm : String} {q n : ℕ} {w : Claims.Flock.Wit} (z : HmNets.Var → Bool)
    (hz : FlockSoundness.Discharge.Hm.stHolds z ((HmNets.hm96Row Hm96.Default512.key).run HmNets.St.empty).2)
    (hcv : ∀ i < 512, z i = FlockSoundness.Discharge.Hm.wbit (FlockSoundness.Discharge.Hm.lastCv
      ⟨(rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)).toArray⟩) i)
    (hpad : ∀ i < 1024, z (512 + i) = FlockSoundness.Discharge.Hm.wbit (FlockSoundness.Discharge.Hm.lastBlock
      ⟨(rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)).toArray⟩) i)
    (hy : ∀ j < 1536, z (1536 + j) = bitL (readForms w (hmSalt c q n)) j)
    (hout : ∀ t < 1024, w.hid.getD (slotHid c n (hmBase c q n) (hmOutCol c + t)) false =
      FlockSoundness.Discharge.Hm.bit z ((HmNets.hm96Row Hm96.Default512.key).run HmNets.St.empty).1.2.2.2 t) :
    SlotCommits c nm q n w :=
  slotCommits_of_bits fun t ht => (hout t ht).trans
    ((FlockSoundness.Discharge.Hm.hm96_commit z hz _ (by simp [readForms, Claims.Recursion.VStar.bytesOf, hmSalt])
      hcv hpad hy).2 t ht)

theorem length_xorMask (r y : List UInt8) :
    (FlockSoundness.Merkle.xorBytes (FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 r))
      (FlockSoundness.Refine.K512.mask y)).length = 64 := by
  rw [FlockSoundness.Merkle.xorBytes, List.length_zipWith, FlockSoundness.Discharge.Hm.length_digBytes,
    FlockSoundness.Refine.K512.mask_len]
  rfl

theorem sha512_eq (m : Msg) :
    FlockDeploy.sha512 m = FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 m) := by
  rw [FlockSoundness.Binding.digBytes, FlockSoundness.Refine.dig_eq, FlockSoundness.Refine.H512_val]
  rfl

theorem hmCalls_true {c : Circuit} {nm : String} {q n : ℕ} {w : Claims.Flock.Wit} (h : SlotCommits c nm q n w) :
    ∀ p ∈ hmCalls c nm q n, FlockDeploy.sha512 (p.1 ++ readForms w p.2.1) = readForms w p.2.2 := by
  have hb : ∀ t (ht : t < 1024), w.hid.getD (slotHid c n (hmBase c q n) (hmOutCol c + t)) false =
      bitL (FlockSoundness.Binding.commitString
        (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))
        (readForms w (hmSalt c q n))) t := fun t ht => by
    have e := bitL_readForms (w := w) (fs := hmCommit c q n) (q := t) (by simpa [hmCommit] using ht)
    rw [h] at e
    rw [e]
    simp [hmCommit, formBit_var]
  have hy : ∀ j (hj : j < 1536), w.hid.getD (slotHid c n (hmBase c q n) (hmIn c 2 + j)) false =
      bitL (readForms w (hmSalt c q n)) j := fun j hj => by
    rw [bitL_readForms (fs := hmSalt c q n) (by simpa [hmSalt] using hj)]
    simp [hmSalt, formBit_var]
  intro p hp
  simp only [hmCalls, List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases hp with rfl | rfl
  · rw [sha512_eq]
    refine (readForms_eq_of_bits (by simp [hmCommit, FlockSoundness.Discharge.Hm.length_digBytes])
      fun t ht => ?_).symm
    have ht' : t < 512 := by simpa [hmCommit] using ht
    rw [List.getElem_drop]
    simp only [hmCommit, List.getElem_map, List.getElem_range, formBit_var]
    rw [hb _ (by omega), FlockSoundness.Binding.commitString,
      FlockSoundness.Discharge.Hm.bitL_append_hi (length_xorMask _ _)]
    rfl
  · rw [sha512_eq]
    refine (readForms_eq_of_bits (by simp [hmDigest, FlockSoundness.Discharge.Hm.length_digBytes])
      fun i hi => ?_).symm
    have hi' : i < 512 := by simpa [hmDigest] using hi
    have hm := FlockSoundness.Discharge.Hm.bitL_mask (readForms w (hmSalt c q n)) hi'
    rw [← FlockSoundness.Refine.K512_mask] at hm
    have hx := FlockSoundness.Discharge.Hm.bitL_xorBytes (FlockSoundness.Discharge.Hm.length_digBytes
      (FlockSoundness.Refine.H512 (rowPfx c q ++ readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var))))
      (FlockSoundness.Refine.K512.mask_len (readForms w (hmSalt c q n))) hi'
    have hci := hb i (by omega)
    rw [FlockSoundness.Binding.commitString, FlockSoundness.Discharge.Hm.bitL_append_lo (length_xorMask _ _) hi', hx,
      hm, maskV_filter] at hci
    have hmap : ((((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
        fun j => slotHid c n (hmBase c q n) (hmIn c 2 + j)).map fun v => w.hid.getD v false) =
        ((List.range 1536).filter fun j => Hm96.bit Hm96.Default512.key (i + j)).map
          (bitL (readForms w (hmSalt c q n))) := by
      rw [List.map_map]
      exact List.map_congr_left fun j hj => hy j (List.mem_range.1 (List.mem_filter.1 hj).1)
    simp only [hmDigest, List.getElem_map, List.getElem_range, formBit, List.foldr_cons, List.foldr_map]
    simp only [← hmap, List.foldr_map] at hci
    rw [hci, Bool.xor_assoc, Bool.xor_self, Bool.xor_false]
    rfl

def SlotsCommit (r : XR) (s : ℕ) (c : Circuit) (w : Claims.Flock.Wit) : Prop :=
  (∀ h ∈ heldRows r s, SlotCommits c h.2.1 h.2.2.1 h.2.2.2 w) ∧
    ∀ l N, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N) → ∀ q, capPort c = some q →
      ∀ n < N, SlotCommits c "top" q n w

def HmSlotsCommit {Col T : Type} (DO : Claims.Flock.Deploy Coin Col T) (st : Stage) : Prop :=
  ∀ (x : Msg) (ms : List (Msg × Out)) (s : ℕ) (w : Claims.Flock.Wit) (r : XR) (c : Circuit),
    StagedPart st s x (staged st x ms s) → Claims.Flock.check DO.parse (staged st x ms s) w = true → readX x = some r →
      r.w.parts[s]?.bind partCircuit = some c → SlotsCommit r s c w

theorem mem_capsCalls {r : XR} {s : ℕ} {p : Msg × List HmNets.Lin × List HmNets.Lin} (hp : p ∈ capsCalls r s) :
    ∃ l N c q n, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N) ∧
      r.w.parts[s]?.bind partCircuit = some c ∧ capPort c = some q ∧ n < N ∧ p ∈ hmCalls c "top" q n := by
  unfold capsCalls at hp
  generalize (RecVStar.statements r.shape).toOption.bind (·[s]?) = o at hp
  rcases o with _ | st
  · simp at hp
  · simp only [Option.bind_some] at hp
    split at hp
    · rename_i l N
      generalize r.w.parts[s]?.bind partCircuit = oc at hp
      rcases oc with _ | c
      · simp at hp
      · simp only [Option.bind_some] at hp
        cases hq : capPort c with
        | none => simp [hq] at hp
        | some q =>
          rw [hq] at hp
          simp only [Option.map_some, Option.getD_some, List.mem_flatMap, List.mem_range] at hp
          obtain ⟨n, hn, hp⟩ := hp
          exact ⟨l, N, c, q, n, rfl, rfl, hq, hn, hp⟩
    · simp at hp

theorem readersTrue_layoutBTops {Col T : Type} (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin)))
    (IV : Claims.Verifier Coin) (b : Brings) (DO : Claims.Flock.Deploy Coin Col T) (h : HmSlotsCommit DO b.st) :
    ReadersTrue FlockDeploy.sha512 IV { b with rd := (layoutBTops fin).readers } DO := by
  intro x as cs ws _ _ hsp hO e he
  obtain ⟨s, hs, he⟩ := Claims.Recursion.VStar.part_of_mem_calls he
  have hsn : s < nS x := hO.1 ▸ hs
  have hck : Claims.Flock.check DO.parse (staged b.st x (roundsOf as cs) s) ws[s] = true := by
    have h2 := hO.2 s hs
    rwa [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hsn, Option.map_some,
      Option.getD_some] at h2
  have hsg : StagedPart b.st s x (staged b.st x (roundsOf as cs) s) := by
    have h1 := hsp s hsn
    rwa [stagedAll, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hsn, Option.map_some,
      Option.getD_some] at h1
  change e ∈ (layoutBTops fin).calls x s ws[s] at he
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 he
  cases hr : readX x with
  | none => simp [layoutBTops, layoutBDeployed, hr] at hp
  | some r =>
    have hp' : p ∈ layoutBDeployed.shaAt x s ∨ p ∈ capsCalls r s := by
      simpa [layoutBTops, hr] using hp
    rcases hp' with hp | hp
    · cases hpt : r.w.parts[s]? with
      | none => simp [layoutBDeployed, hr, hpt] at hp
      | some pt =>
        cases hc : partCircuit pt with
        | none => simp [layoutBDeployed, hr, hpt, hc] at hp
        | some c =>
          simp only [layoutBDeployed, hr, hpt, hc, Option.bind_some, Option.map_some, Option.getD_some,
            List.mem_flatMap] at hp
          obtain ⟨hh, hmem, hp⟩ := hp
          have hc' : r.w.parts[s]?.bind partCircuit = some c := by rw [hpt]; exact hc
          exact hmCalls_true ((h x _ s ws[s] r c hsg hck hr hc').1 hh hmem) p hp
    · obtain ⟨l, N, c, q, n, hst, hc, hq, hn, hp⟩ := mem_capsCalls hp
      exact hmCalls_true ((h x _ s ws[s] r c hsg hck hr hc).2 l N hst q hq n hn) p hp

/-- What stays open at the dense form is `PartChecksBV`, `PartOpensB` and `DecompB`; `ksLin2_deployed_full` proves
`HmSlotsCommit`. -/
theorem ksLin2_deployed_slots (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    HmSlotsCommit (deployZkRec _root_.Flock.Readable.relation) bR.st →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hS
  exact ksLin2_deployed_rec b aux nOf hd rest fin wE join final w hP
    (readersTrue_layoutBTops (finOpen fin) Bd.IVStar bR (deployZkRec _root_.Flock.Readable.relation) hS)

/-- It reads `StrictBirthday`, `ExpectedCR` and `ksLin2_deployed_slots`' open premises: a bound on a number under
named premises, not a proved security level. -/
theorem bitsLin2_deployed_slots (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (rest : Msg → List (Msg × Out) → List UInt8)
    (fin : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd rest }
    let L := layoutBTops (finOpen fin)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    let bL : Brings := { bR with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart recRead bR) (cmtTr recRead) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    HmSlotsCommit (deployZkRec _root_.Flock.Readable.relation) bR.st →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D bL hP hS
  exact bitsLin2_deployed_rec b aux nOf hd rest fin wE join final w hP
    (readersTrue_layoutBTops (finOpen fin) Bd.IVStar bR (deployZkRec _root_.Flock.Readable.relation) hS)

end RecDeploy
