module

public import Proofs.Flock.SamplingGeneric.Split
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
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Sampling.Cost

/-!
# The sampled compiler's extraction: assembling and binding

`Compiled.E` at an accepted run assembles the openings its sampled proof blocks' extractions keep. This file proves
`KSOf`'s deterministic half: an extraction that fails gives the failure event the probabilistic half bounds
(`Compiled.Fails` at `Compiled.Found`, `SamplingGeneric/Fails.lean`), or a `Bext` collision (`split`,
`Claims.Sampling`'s `split` with Flock's `E^rep` replaced by `Arg`'s extractor at each sampled proof block's session).
With `KS_of_sessionKS` it gives `KSOf` (`ksOf`).

* `innerBody`, `outerBody`, `outerList`, `found`: **the extractor's parts**, and what it keeps (`E_eval`); the kept
  openings pass their checks, with the checks' calls among the extractor's own (`passes_runExt`, `passes_found`,
  `ownLog_E`), and the revealed openings that pass, of the extracted witness of the block at a slot of the sample, at
  the slot's coins, are kept (`unitExt_mem`, `mem_runExt`, `mem_found`).
* `split`: **the deterministic half**. At an accepted run whose extraction fails, `Fails` holds or the extractor's own
  calls collide. Two kept openings under one roots at one position that disagree are a collision of the calls that
  checked them (`CommitmentScheme.Binds`), so without one the kept openings agree and assemble a witness; its incorrect
  set is refused at `δ`, and a block of it found at a kept inner run would be correct at the kept openings
  (`Sessions.opens`, with the interior roots fixed by the first stage, `Sessions.interior`).
* `ksOf`: **`KSOf`**, `KS_of_sessionKS` at `split`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Opening waitAccC forkOn flatMapRP assemble ownLog ownLog_bind accList OwnColl
  flatMapRP_eval waitAccC_eval_nil accList_map mem_zip_finRange getElem?_map_pair mem_checkRevealed agree_of_passes
  val_assemble ιOf ιOf_eq faulty_wrong acc_of_accList accList_length_eq)
open Claims.Recursion (serve viewing)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The extractor's parts -/

/-- The extractor at one accepted inner run `li`, under the outer run `j`: every slot's extraction (`runExt`). -/
noncomputable def innerBody (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (c : L.ECoins x) (j : Fin (L.kOut x))
    (li : Fin (L.kIn x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  L.runExt x s li.2.1 li.2.2 fun p => c.2.2 (j, li.1, p)

/-- The extractor at one accepted outer run `jo`: the inner wait on the second stage, and `innerBody` at each. -/
noncomputable def outerBody (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (c : L.ECoins x)
    (jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  (waitAccC (L.kIn x) ((List.finRange (L.capIn x)).map fun l => forkOn (L.stage2 x) jo.2.1 (c.2.1 jo.1 l)) []).bind
    fun inner => flatMapRP (L.innerBody x s c jo.1) ((List.finRange (L.kIn x)).zip inner)

/-- The outer wait's kept runs, indexed. -/
noncomputable def outerList (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (c : L.ECoins x) : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen)) :=
  (List.finRange (L.kOut x)).zip ((waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).eval H run').1

/-- **What the extractor keeps** at the outer runs `Z`: at an accepted judged run, `Z = outerList`. The list is a
parameter so that unfolding `found` never asks the kernel to evaluate the outer wait over `List.finRange (L.kOut x)`. -/
noncomputable def found (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (c : L.ECoins x)
    (Z : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen))) : List (List Msg × Opening Bool) :=
  ((flatMapRP (L.outerBody x s c) Z).eval H run').1

theorem found_eq (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (c : L.ECoins x)
    (Z : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen))) :
    L.found H x s run' c Z = ((flatMapRP (L.outerBody x s c) Z).eval H run').1 :=
  rfl

/-- The extractor's program, with its outer body named. -/
theorem E_ext_eq (L : Compiled Coin A) (x : Msg) (r₀ : Fin (L.V.k x) → Coin) (c : L.ECoins x) :
    L.E.ext x r₀ c = .rerun r₀ fun v₀ =>
      match L.parse x with
      | none => .ret none
      | some s =>
        if v₀.out = .acc then
          (waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).bind fun outer =>
            (flatMapRP (L.outerBody x s c) ((List.finRange (L.kOut x)).zip outer)).bind fun found =>
              .ret (some (assemble s found))
        else .ret none :=
  rfl

/-- **The extractor at an accepted run** assembles what it keeps. -/
theorem E_eval (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.ECoins x) {s : AuditStatement} (hs : L.parse x = some s)
    (hacc : (run H P L.V x ω r₀).out = .acc) :
    (L.E.eval H P x ω r₀ c).1 =
      some (assemble s (L.found H x s (run H P L.V x ω) c (L.outerList H x (run H P L.V x ω) c))) := by
  have hacc' : (run H P L.V x ω r₀).seen.out = .acc := hacc
  rw [found_eq]
  simp only [Extractor.eval, E, RP.eval, hs]
  rw [ite_eq_left hacc', (RP.eval_bind _ _ _ _).1, (RP.eval_bind _ _ _ _).1]
  rfl

/-- The kept openings' checks are the extractor's own calls. -/
theorem ownLog_E (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.ECoins x) {s : AuditStatement} (hs : L.parse x = some s)
    (hacc : (run H P L.V x ω r₀).out = .acc) :
    ∀ t ∈ ownLog H (run H P L.V x ω) (flatMapRP (L.outerBody x s c) (L.outerList H x (run H P L.V x ω) c)),
      t ∈ ownLog H (run H P L.V x ω) (L.E.ext x r₀ c) := by
  intro t ht
  have hacc' : (run H P L.V x ω r₀).seen.out = .acc := hacc
  simp only [E, ownLog, hs]
  rw [ite_eq_left hacc', ownLog_bind, ownLog_bind]
  exact List.mem_append_right _ (List.mem_append_left _ ht)

theorem passes_unitExt (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (s : AuditStatement) (u : ℕ) (r : Fin (L.V.k x) → Coin) (v : Seen) (c : A.EC (L.sh x u)) :
    Sampling.Passes H run' L.commit (L.unitExt x s u r v c) := by
  unfold unitExt
  split
  · exact Sampling.Passes.ret_nil H run' L.commit
  · split
    · refine Sampling.Passes.bind _ fun o => ?_
      cases o with
      | none => exact Sampling.Passes.ret_nil H run' L.commit
      | some w => exact Sampling.Passes.checkRevealed H run' L.commit _
    · exact Sampling.Passes.ret_nil H run' L.commit

/-- What a run's extractions keep passes its opening check, with the check's calls their own. -/
theorem passes_runExt (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (s : AuditStatement) (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) :
    Sampling.Passes H run' L.commit (L.runExt x s r v c) := by
  unfold runExt
  exact Sampling.Passes.flatMapRP fun _ _ => by
    split
    · exact L.passes_unitExt H x run' s _ _ _ _
    · exact Sampling.Passes.ret_nil H run' L.commit

theorem passes_found (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (s : AuditStatement) (c : L.ECoins x) (Z : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen))) :
    Sampling.Passes H run' L.commit (flatMapRP (L.outerBody x s c) Z) :=
  Sampling.Passes.flatMapRP fun _ _ => Sampling.Passes.bind _ fun _ => Sampling.Passes.flatMapRP fun li _ =>
    L.passes_runExt H x run' s li.2.1 li.2.2 _

/-- A revealed opening that passes, of the witness a block's session's extraction returns, is among the block's
extraction's. -/
theorem unitExt_mem (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (s : AuditStatement) (u : ℕ) (r : Fin (L.V.k x) → Coin) (c : A.EC (L.sh x u)) {xS : Msg} {vS : Seen}
    (hsess : L.session x u (run' r).seen = some (xS, vS)) (h : A.shape xS = L.sh x u) {w : A.W}
    (hw : ((serve (fun r' => viewing (L.view x u) (L.place x u xS r r'))
      (A.ext xS (L.coinsOf x u xS r) (cast (congrArg A.EC h.symm) c))).eval H run').1 = some w)
    {q : List Msg × Opening Bool} (hq : q ∈ L.reveal s u (run' r).seen w) (hok : L.commit.Opens H q.1 q.2 = true) :
    q ∈ ((L.unitExt x s u r (run' r).seen c).eval H run').1 := by
  unfold unitExt
  simp only [hsess]
  rw [dite_eq_left h, (RP.eval_bind _ _ _ _).1, hw]
  exact mem_checkRevealed H run' L.commit hq hok

/-- The extraction of the proof block at a slot of the sample, at the slot's coins, is among the run's extractions'. -/
theorem mem_runExt (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (s : AuditStatement) (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) {i : Fin (L.nPU x)} {u : ℕ}
    (hi : (L.checked x r)[i.val]? = some u) (hlt : u < L.nPU x) {q : List Msg × Opening Bool}
    (hq : q ∈ ((L.unitExt x s u r v (c i ⟨u, hlt⟩)).eval H run').1) :
    q ∈ ((L.runExt x s r v c).eval H run').1 := by
  unfold runExt
  rw [flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨(i, u), mem_zip_finRange hi, ?_⟩
  dsimp only
  rw [dite_eq_left hlt]
  exact hq

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem mem_outerList (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (c : L.ECoins x) {j : Fin (L.kOut x)} {r : Fin (L.V.k x) → Coin}
    (hj : (accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1))[j.val]? =
      some r) :
    (j, (r, (run' r).seen)) ∈ L.outerList H x run' c := by
  unfold outerList
  rw [waitAccC_eval_nil]
  exact mem_zip_finRange (getElem?_map_pair run' hj)

/-- **A block's extraction at a slot of a kept inner run, under a kept outer run, is kept**, at the slot's coins. -/
theorem mem_found (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (c : L.ECoins x)
    {Z : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen))} {j : Fin (L.kOut x)} {r : Fin (L.V.k x) → Coin}
    (hj : (j, (r, (run' r).seen)) ∈ Z) {l : Fin (L.kIn x)} {d : Fin (L.V.k x) → Coin}
    (hl : (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.kIn x)
      ((List.finRange (L.capIn x)).map (c.2.1 j)))[l.val]? = some d)
    {i : Fin (L.nPU x)} {v : ℕ} (hv : (L.checked x (forkOn (L.stage2 x) r d))[i.val]? = some v) (hvlt : v < L.nPU x)
    {q : List Msg × Opening Bool}
    (hq : q ∈ ((L.unitExt x s v (forkOn (L.stage2 x) r d) (run' (forkOn (L.stage2 x) r d)).seen
      (c.2.2 (j, l, i) ⟨v, hvlt⟩)).eval H run').1) :
    q ∈ L.found H x s run' c Z := by
  unfold found
  rw [flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨(j, (r, (run' r).seen)), hj, ?_⟩
  unfold outerBody
  rw [(RP.eval_bind _ _ _ _).1, flatMapRP_eval, waitAccC_eval_nil]
  have hm : ((List.finRange (L.capIn x)).map fun l => forkOn (L.stage2 x) r (c.2.1 j l)) =
      ((List.finRange (L.capIn x)).map (c.2.1 j)).map (forkOn (L.stage2 x) r) := by
    rw [List.map_map]; rfl
  dsimp only
  rw [hm, accList_map, List.map_map]
  refine List.mem_flatMap.2 ⟨(l, (forkOn (L.stage2 x) r d, (run' (forkOn (L.stage2 x) r d)).seen)),
    mem_zip_finRange ?_, ?_⟩
  · rw [List.getElem?_map, hl]; rfl
  · exact L.mem_runExt H x run' s _ _ (fun p => c.2.2 (j, l, p)) hv hvlt hq

/-! ## The split -/

/-- **The split**: at an accepted run whose extraction fails, the failure event holds or the extractor's own calls
collide. Otherwise the kept openings agree (`agree_of_passes`); take `B` the assembled witness's incorrect set, refused
at `δ`, and a kept outer run `r`. At the interior values the kept openings under `r`'s interior roots give, the proof
block at each `v ∈ B` is wrong (`faulty_wrong`); a kept inner run at which `v` is found would make it correct
(`Sessions.opens`, its revealed openings kept by `mem_found`, the interior roots `r`'s by `Sessions.interior`), so `r`
is bad for `B`. -/
theorem split {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H) (δ : ℝ)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (L.V.k x) → Coin) (c : L.ECoins x)
    (hacc : (run H P L.V x ω r₀).out = .acc)
    (hfail : ¬ (Sampling.R H L.commit L.parse L.partition δ).ext x (L.E.eval H P x ω r₀ c).1) :
    L.Fails δ x (run H P L.V x ω) (L.Found H x (run H P L.V x ω)) c ∨ OwnColl H L.E P x ω r₀ c := by
  classical
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hS.refuses x Ω P ω r₀ hacc
  by_cases hcoll : OwnColl H L.E P x ω r₀ c
  · exact Or.inr hcoll
  refine Or.inl ⟨s, Pt, hs, hPt, hin, ?_⟩
  set run' := run H P L.V x ω with hrun'
  set fd := L.found H x s run' c (L.outerList H x run' c) with hfd
  have hpass : ∀ q ∈ fd, L.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((L.commit.opens q.1 q.2).eval H).2, t ∈ ownLog H run' (L.E.ext x r₀ c) := by
    intro q hq
    rw [hfd, found_eq] at hq
    obtain ⟨h1, h2⟩ := L.passes_found H x run' s c _ q hq
    exact ⟨h1, fun t ht => L.ownLog_E H P x ω r₀ c hs hacc t (h2 t ht)⟩
  have hagree : ∀ q ∈ fd, ∀ q' ∈ fd, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => agree_of_passes hbind hpass hcoll hq hq' h1 h2
  rw [L.E_eval H P x ω r₀ c hs hacc] at hfail
  set w := assemble s fd with hw
  have hB : Sampling.faulty H L.commit s Pt w ∈ Sampling.S₁ s.design Pt δ := by
    unfold Sampling.S₁
    refine mem_filter.2 ⟨mem_powerset.2 (filter_subset _ _), ?_⟩
    by_contra hlt
    exact hfail ⟨s, Pt, hs, hPt, lt_of_not_ge hlt⟩
  by_cases hshort : (accList (fun r => (run' r).out = .acc) (L.kOut x)
    ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x
  · exact Or.inl hshort
  refine Or.inr ⟨_, hB, accList_length_eq hshort, fun j r hj => ?_⟩
  have hjZ := L.mem_outerList H x run' c hj
  by_cases hshort2 : (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.kIn x)
    ((List.finRange (L.capIn x)).map (c.2.1 j))).length < L.kIn x
  · exact Or.inl hshort2
  refine Or.inr ⟨accList_length_eq hshort2, fun l d hl v hv hgood => ?_⟩
  obtain ⟨b, hb, hnc⟩ := faulty_wrong hv (ιOf fd (L.iroots x (run' r).seen v))
  obtain ⟨i, hvlt, hvi, xS, vS, hsess, hsh, wS, hwS, hRw⟩ := hgood
  have hvc := List.mem_of_getElem? hvi
  have hacc' : (run' (forkOn (L.stage2 x) r d)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  obtain ⟨b', hb', hcor⟩ :=
    hS.opens x s Pt Ω P ω (forkOn (L.stage2 x) r d) v xS vS wS hs hPt hin hacc' hvc hsess hRw
  have hcorr := hcor (w.val H L.commit s) (ιOf fd (L.iroots x (run' r).seen v)) fun q hq hok => by
    have hqf : q ∈ fd := L.mem_found H x s run' c hjZ hl hvi hvlt
      (L.unitExt_mem H x run' s v _ _ hsess hsh hwS hq hok)
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · rw [hr] at hok
      exact val_assemble hagree hqf hr hok
    · rw [hS.interior x Ω P ω r d v] at hi
      rw [← hi]
      exact ιOf_eq hagree hqf
  rw [hb, Option.some.injEq] at hb'
  subst hb'
  exact hnc hcorr

end Compiled

/-- **`KSOf`**: the sessions' bound at `Found` (`SessionKS`) and the split give `Compiled.KS`. -/
theorem ksOf : KSOf := by
  intro Coin _ _ _ A L H hbind hS hSK
  exact Compiled.KS_of_sessionKS hS hSK fun δ _ x _ P s hacc hfail =>
    Compiled.split hbind hS δ x P s.1 s.2.1 s.2.2 hacc hfail

end Claims.SamplingGeneric
