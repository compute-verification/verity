module

public import Proofs.Flock.Claims.Recursion.OpenedDecode
public import Proofs.Flock.Claims.Recursion.OpenedPlain
public import Proofs.Flock.Claims.Recursion.VerdictRows
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
import Proofs.Flock.Soundness.Refine.Seam

/-! Row D of the verdict, the Ligerito openings, from the rec-open parts one query at a time. -/

@[expose] public section

namespace RecDeploy.Rows

open Flock
open FlockVBridge.Family (CapsAt rows_conss decoded_cons)
open Flock.RecAlgebra (verifierRows decoded Opened consAt Info Lin)
open FlockSoundness.Refine (openAt shapeOf)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.OpenedPlain (QueryOpens LevelShaped opened_plain_of levelShaped_of_decode saltLen_plain)

theorem consOf_at {sh : Shape} {sch : Flock.Schedule} {p : Proof} {rounds : Array (Array F128)}
    {extras : Array Claim} {comb v : Array F128} {info : Info Lin} {kappa : Array (Array (Array F128))}
    (hv : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (h0 : 0 < sh.sched.levels.size)
    (hq : ∀ i < sh.sched.levels.size, ∀ c,
      (Flock.positions (sh.sched.levels[i]!) c).size = (sh.sched.levels[i]!).q)
    {cs : Flock.ConsCtx} (hcs : cs ∈ info.lg.conss.toList) (l : ℕ) :
    consOf sh sch p rounds extras comb cs.start l =
      consAt (openAt p.ligerito (sch.levels.size - 1) cs.start).1 (sch.levels[cs.start]!).q cs.alpha l := by
  have E := rows_conss hv h0 hq
  obtain ⟨-, he, -⟩ := FlockVBridge.Family.Conss.conss_mem E hcs
  have hfind : info.lg.conss.find? (·.start == cs.start) = some cs := by
    rcases hf : info.lg.conss.find? (·.start == cs.start) with _ | y
    · exact absurd (by simp : (cs.start == cs.start) = true)
        (Array.find?_eq_none.1 hf cs (by simpa using hcs))
    · have hy : y ∈ info.lg.conss.toList := by simpa using Array.mem_of_find?_eq_some hf
      have hp : y.start = cs.start := by simpa using Array.find?_some hf
      obtain ⟨-, hey, -⟩ := FlockVBridge.Family.Conss.conss_mem E hy
      have hyc : y = cs := by rw [← hey, hp, he]
      rw [hf, hyc]
  unfold consOf
  rw [hv]
  simp only [hfind]
  rw [FlockVBridge.Family.Conss.getD_default]

theorem capsAt_of_decode {st : Setup} {sch : Flock.Schedule} {n : ℕ} {bytes : ByteArray} {p : Proof}
    (hd : st.merkle.digestLen = 64) (h : decodeProof (shapeOf st sch n) bytes = .ok p) :
    CapsAt sch p.cap p.ligerito := by
  have hs := FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _ h
  have hcap : p.cap.size = 2 ^ (sch.levels.getD 0 default).cap := hs.cap
  have hcapE : ∀ e ∈ p.cap, e.size = st.merkle.digestLen := hs.capEach
  have hrc : p.ligerito.recursiveCaps.size = sch.levels.size - 1 := hs.rcaps
  have hrcE : ∀ (i : ℕ) (hi : i < p.ligerito.recursiveCaps.size),
      p.ligerito.recursiveCaps[i].size = 2 ^ (sch.levels.getD (i + 1) default).cap ∧
        ∀ e ∈ p.ligerito.recursiveCaps[i], e.size = st.merkle.digestLen := hs.rcapsEach
  intro ℓ hℓ
  unfold FlockSoundness.Refine.capAt
  split
  · rename_i h0
    subst h0
    rw [FlockVBridge.Family.Conss.getD_default] at hcap
    exact ⟨hcap, fun x hx => (hcapE x hx).trans hd⟩
  · rename_i h0
    have hi : ℓ - 1 < p.ligerito.recursiveCaps.size := by omega
    obtain ⟨h1, h2⟩ := hrcE (ℓ - 1) hi
    rw [FlockVBridge.Family.Conss.getD_default, show ℓ - 1 + 1 = ℓ by omega] at h1
    rw [getElem!_pos p.ligerito.recursiveCaps (ℓ - 1) hi]
    exact ⟨h1, fun x hx => (h2 x hx).trans hd⟩

namespace DecAt

variable {rd : Claims.Msg → Option FlockDeploy.FinalP} {x : Claims.Msg} {ms : List Claims.Msg}
  {cs : List FlockDeploy.Coin} (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

theorem rowD_of_levels
    (hmerkle : (D.files cls i).st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hfast : Flock.fast100 (D.files cls i).st.m = some (D.files cls i).sch)
    (hcaps : CapsAt (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito)
    (hshape : ∀ ℓ < (D.files cls i).sch.levels.size,
      LevelShaped (D.files cls i).sch ((D.files cls i).proof r).ligerito ℓ ((D.files cls i).sch.levels[ℓ]!).q)
    (hq : ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
        QueryOpens (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ j
          (info.lg.conss[ℓ]!).pos[j]!) :
    D.RowD cls acc i r := by
  intro v info kappa hv c hc
  have h0 : 0 < (D.sh cls i).sched.levels.size := (FlockVBridge.Family.fast100_q0 hfast).1
  have hseam := FlockSoundness.Refine.fast100_level _ _ hfast
  have hpos : ∀ ℓ < (D.sh cls i).sched.levels.size, ∀ c,
      (Flock.positions ((D.sh cls i).sched.levels[ℓ]!) c).size = ((D.sh cls i).sched.levels[ℓ]!).q :=
    fun ℓ hℓ c => FlockSoundness.Refine.positions_size _
      (hseam _ (by rw [getElem!_pos _ ℓ hℓ]; exact Array.getElem_mem hℓ)) c
  obtain ⟨hℓ, hc0, hpsz⟩ := FlockVBridge.Family.Conss.conss_mem (rows_conss hv h0 hpos) hc
  have hℓ' : c.start < (D.files cls i).sch.levels.size := hℓ
  have hpsz' : c.pos.size = ((D.files cls i).sch.levels[c.start]!).q := hpsz
  rw [hmerkle]
  refine opened_plain_of hℓ' hcaps (by rw [hpsz']; exact hshape _ hℓ') (fun j hj => ?_) ?_
  · have h := hq v info kappa hv c.start hℓ' j (by rw [← hpsz']; exact hj)
    rwa [hc0] at h
  · rw [decoded_cons]
    exact congrArg (fun f => Array.map f (Array.range (2 ^ ((D.files cls i).sch.levels[c.start]!).lanes)))
      (funext fun l => consOf_at hv h0 hpos hc l)

theorem rowD_of_queries
    (hmerkle : (D.files cls i).st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hfast : Flock.fast100 (D.files cls i).st.m = some (D.files cls i).sch)
    (hdec : decodeProof (shapeOf (D.files cls i).st (D.files cls i).sch (D.files cls i).extras.size)
      ((D.files cls i).bytes r) = .ok ((D.files cls i).proof r))
    (hq : ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
        QueryOpens (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ j
          (info.lg.conss[ℓ]!).pos[j]!) :
    D.RowD cls acc i r := by
  have hd : (D.files cls i).st.merkle.digestLen = 64 := (congrArg MerkleScheme.digestLen hmerkle).trans rfl
  have hs : (D.files cls i).st.merkle.saltBytes = 0 := (congrArg MerkleScheme.saltBytes hmerkle).trans rfl
  exact D.rowD_of_levels cls acc i r hmerkle hfast (capsAt_of_decode hd hdec)
    (fun ℓ hℓ => levelShaped_of_decode (saltLen_plain hs) hd hdec ℓ hℓ) hq

end DecAt

end RecDeploy.Rows
