module

public import Proofs.Flock.Claims.Complete.Replay.Ligerito
public import Proofs.Flock.Claims.Complete.Replay.LigeritoRun
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
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
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.VBridge.Algebra.Sizes

/-! The goals `Ligerito` and `Consumed` at every honest run, given each level's Merkle opening (`LgMerkle`) and the
final identity (`LgFinal`). -/

@[expose] public section

namespace FlockComplete.Replay

open Flock Transcript

theorem fast100_ligerito {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) :
    sch.n ≠ 0 ∧ 2 ≤ sch.levels.size ∧ sch.k0 + 3 * (sch.levels.size - 1) ≤ sch.n ∧
      (∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) ∧
      sch.ext = sch.n - sch.k0 - 3 * (sch.levels.size - 1) := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ fast100 m.val, decide (s.n ≠ 0 ∧ 2 ≤ s.levels.size ∧
        s.k0 + 3 * (s.levels.size - 1) ≤ s.n ∧ (∀ j < s.levels.size, (s.levels.getD j default).q ≠ 0) ∧
        s.ext = s.n - s.k0 - 3 * (s.levels.size - 1)) = true := by
      decide +kernel
    exact of_decide_eq_true (key ⟨m, hm⟩ sch h)
  · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at h
    cases h

theorem fast100_isSome_of {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 35) : (fast100 m).isSome = true := by
  have key : ∀ m : Fin 40, 22 ≤ m.val → m.val ≤ 35 → (fast100 m.val).isSome = true := by
    decide +kernel
  exact key ⟨m, by omega⟩ h1 h2

theorem _root_.FlockComplete.Run.sch_fast100 {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) : fast100 o.tabs[i]!.m = some (o.sch i) := by
  obtain ⟨-, h22, h35⟩ := o.tab_m hi
  obtain ⟨s, hs⟩ := Option.isSome_iff_exists.1 (fast100_isSome_of h22 h35)
  have e : o.sch i = (fast100 o.tabs[i]!.m).getD default := by
    unfold Run.sch schedsOf
    rw [getElem!_pos (o.tabs.map _) i (by simpa using hi), getElem!_pos o.tabs i hi, Array.getElem_map]
  rw [e, hs]
  rfl

def _root_.FlockComplete.Run.lgCf {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : ℕ → ℕ → F128 :=
  fun k => coinOf (o.resp (repStart o.spec o.sts (2 * i + r) + k))

def _root_.FlockComplete.Run.lgPre {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : Array F256 × LState × Transcript :=
  Lg.pre (o.proof i r).ligerito (o.lgCf i r) (o.sch i) (Rs.T (o.trace i) (o.claims i r) (o.rsCoins i r))
    (o.proof i r).cap (o.rsEnd i r)

def _root_.FlockComplete.Run.lgOut {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) (i r : ℕ) : Lg.Vars × LState × Transcript :=
  Lg.out (o.proof i r).ligerito (o.lgCf i r) (o.sch i) (Rs.T (o.trace i) (o.claims i r) (o.rsCoins i r))
    (o.proof i r).cap (o.rsEnd i r)

def LgAt {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel) (i r : ℕ) : Prop :=
  (o.proof i r).ligerito.initialCap = (o.proof i r).cap ∧ Agrees (o.stream i r).rounds (o.lgOut i r).2.2.rounds 0

def LgCount {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel) (i r : ℕ) :
    Prop :=
  (o.stream i r).rounds.size = (o.lgOut i r).2.2.rounds.size

def LgMerkle (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    Lg.MerkleOk o.hs o.tabs[i]!.merkle (o.sch i) (o.proof i r).cap (o.proof i r).ligerito (o.lgCf i r)
      (o.rsEnd i r).k

def LgFinal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    FlockSoundness.Refine.lhsE (o.proof i r).ligerito.finalProof.yr (o.lgOut i r).1.2.2.2
      (FlockSoundness.Refine.resE (basis (Rs.sws (o.claims i r) (o.rsCoins i r))) (o.sch i).n (o.lgPre i r).1
        (o.lgOut i r).1.2.2.1 (o.lgOut i r).1.2.2.2 (o.lgOut i r).2.1) = (o.lgOut i r).2.1.s

local instance lawfulBEqByteArrayLg : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

theorem _root_.FlockComplete.Run.lg_run {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (hR1 : TraceR1CS H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) (hzc : ZcAt o i r)
    (hlc : LcAt o i r) (hal : LcAlgebra H rel) (hrs : RsAt o i r) (hra : RsAlgebra H rel) (hD : RepDecodes H rel)
    (hlg : LgAt o i r) (hmk : LgMerkle H rel) (hfin : LgFinal H rel) :
    (ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.proof i r).cap (o.proof i r).ligerito
      (basis (o.sw i r).1.2)).run (o.sw i r).2 =
      .ok ((), ⟨(o.stream i r).rounds, (o.lgOut i r).2.2.k, (o.lgOut i r).2.2.pending⟩) ∧
    (o.lgOut i r).2.2.k = (o.lgOut i r).2.2.rounds.size := by
  rw [o.sw_eq hR1 hi hr hzc hlc hal hrs hra]
  have hsz := FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ (hD o i hi r hr)
  obtain ⟨hn, hL, hsd, hq, hext⟩ := fast100_ligerito (o.sch_fast100 hi)
  have hmsg : (o.proof i r).ligerito.messages.size = (o.sch i).k0 + 8 + 7 * ((o.sch i).levels.size - 1 - 1) :=
    hsz.messages
  have hood : (o.proof i r).ligerito.oodValues.size = 1 + 2 * ((o.sch i).levels.size - 1) := hsz.oodValues
  have hqn : (o.proof i r).ligerito.queryNonces.size = (o.sch i).levels.size - 1 + 1 := hsz.queryNonces
  have hcn : (o.proof i r).ligerito.consistencyNonces.size = (o.sch i).levels.size - 1 + 1 := hsz.consistencyNonces
  have hcbn : (o.proof i r).ligerito.claimNonces.size = 3 * ((o.sch i).levels.size - 1) + 2 := hsz.claimNonces
  have hyr : (o.proof i r).ligerito.finalProof.yr.size = 2 ^ ((o.sch i).ext + 1) := hsz.yr
  exact Lg.ligeritoWith_of o.hs o.tabs[i]!.merkle (o.sch i) _ (o.proof i r).cap (o.proof i r).ligerito _ (o.lgCf i r)
    (o.rsEnd i r) (o.rsEnd_k i r) (hlg.2.mono (Nat.zero_le _)) (by rw [hlg.1]; exact beq_self_eq_true _) hn hL hsd hq
    (by omega) (by omega) (by omega) (by omega) hcbn (by rw [hyr, hext]) (fun q e => Flock.RecAlgebra.basis_size _ q e)
    (hmk o i hi r hr) (hfin o i hi r hr)

theorem _root_.FlockComplete.Run.lg_eq {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (hR1 : TraceR1CS H rel) (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) (hzc : ZcAt o i r)
    (hlc : LcAt o i r) (hal : LcAlgebra H rel) (hrs : RsAt o i r) (hra : RsAlgebra H rel) (hD : RepDecodes H rel)
    (hlg : LgAt o i r) (hmk : LgMerkle H rel) (hfin : LgFinal H rel) :
    o.lg i r = ((), ⟨(o.stream i r).rounds, (o.lgOut i r).2.2.k, (o.lgOut i r).2.2.pending⟩) := by
  unfold Run.lg
  rw [(o.lg_run hR1 hi hr hzc hlc hal hrs hra hD hlg hmk hfin).1]
  rfl

theorem ligerito_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r) (hal : LcAlgebra H rel)
    (hrs : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) (hra : RsAlgebra H rel) (hD : RepDecodes H rel)
    (hlg : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LgAt o i r) (hmk : LgMerkle H rel) (hfin : LgFinal H rel) :
    Ligerito H rel := by
  intro o i hi r hr
  refine ok_val ⟨_, (o.lg_run hR1 hi hr (hzc o i hi r hr) (hlc o i hi r hr) hal (hrs o i hi r hr) hra hD
    (hlg o i hi r hr) hmk hfin).1⟩

theorem consumed_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r)
    (hlc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LcAt o i r) (hal : LcAlgebra H rel)
    (hrs : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RsAt o i r) (hra : RsAlgebra H rel) (hD : RepDecodes H rel)
    (hlg : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LgAt o i r) (hmk : LgMerkle H rel) (hfin : LgFinal H rel)
    (hcount : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, LgCount o i r) : Consumed H rel := by
  intro o i hi r hr
  have h := o.lg_run hR1 hi hr (hzc o i hi r hr) (hlc o i hi r hr) hal (hrs o i hi r hr) hra hD (hlg o i hi r hr) hmk
    hfin
  rw [o.lg_eq hR1 hi hr (hzc o i hi r hr) (hlc o i hi r hr) hal (hrs o i hi r hr) hra hD (hlg o i hi r hr) hmk hfin]
  show ((o.lgOut i r).2.2.k != (o.stream i r).rounds.size) = false
  rw [h.2, hcount o i hi r hr]
  exact bne_self_eq_false _

end FlockComplete.Replay
