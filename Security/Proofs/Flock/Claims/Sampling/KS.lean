import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Claims.Aggregate

/-!
# The sampling block's KS from Flock's KS

The claims block (`Definitions.Claims.Sampling`) states `KS` at its nested extractor `E`: replay; `k` accepted reruns
among `K`; at each, `k` accepted reruns of the second stage only; at each, Flock's `E^rep` on every checked unit's
session; keep the openings that pass. This file proves it from Flock's KS at the units' sessions and named facts about
the deployed sampled verifier, each a hypothesis to discharge there.

* `SessionKS`, `DrawLaw`, `Refuses`, `InteriorFixed`, `SessionOpens`: **the hypotheses**. Flock's KS at the session a
  selection picks among the checked units; the draw law over `Design.Pa` (`drawLaw_of_slots` proves it of the deployed
  draw, `Sampling.Select.DrawLaw`); the scope refusal; the interior commitments fixed by the first stage; and that a
  checked unit's Flock witness reveals openings making its proof unit correct.
* `SlotsCover`, `SessionOpensSlots`: **sessions that cover several checked units**. Every checked unit has the rounds
  and session of the unit at a slot below `S x` (one session per unit at `S = nPU D`, `slotsCover_units`; one session
  over every checked unit at `S = 1`, `slotsCover_one`), and each slot's session has one Flock witness revealing what
  makes every unit it covers correct; that is `SessionOpens` (`sessionOpensSlots_iff`).
* `interiorFixed_of_prefix`: **`InteriorFixed` from the interiors' place in the run**: a run's first `m` messages read
  only its first `m` rounds' coins (`run_msgs_take`), so interior roots read off the messages before the second stage
  are kept by fresh second-stage coins.
* `split`: **the deterministic half**. At an accepted run whose extraction fails, the failure event `Fails` holds or
  the extractor's own calls collide (`Commitment.Binds`).
* `inner_bad_le`, `outer_bad_le`, `Pr_fails_le`: **the probabilistic half**. Each wait's truncation and union bound
  (`wait_bad_le`), its union's size against the slack `c` (`k = 2^40·(n_RU + n_PU + 1)`), the draw law and Flock's
  error at the inner wait, averaged over the first stage and the tapes: `Pr[Fails] ≤ κ(δ) + √(γ·Pr[B])`.
* `KS_of`, `Bits_of`, `KSCount_of_KS`: **the claims**: `KS` through `KSLin_of_ownColl`; `Bits` with the extractor's
  costs; and the count form, at the miss clamped at `0`.
-/

namespace Claims.Sampling

open Claims Finset

/-! ## Rerun programs, evaluated -/

theorem flatMapRP_eval {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) :
    ∀ l : List β, ((flatMapRP f l).eval H run).1 = l.flatMap fun b => ((f b).eval H run).1
  | [] => rfl
  | b :: bs => by
    simp only [flatMapRP, List.flatMap_cons]
    rw [(RP.eval_bind H run _ _).1, (RP.eval_bind H run _ _).1, flatMapRP_eval H run f bs]
    rfl

theorem mem_zip_finRange {S : Type} {k : ℕ} {l : List S} {j : Fin k} {a : S} (h : l[j.val]? = some a) :
    (j, a) ∈ (List.finRange k).zip l := by
  rw [List.mem_iff_getElem?]
  exact ⟨j.val, List.getElem?_zip_eq_some.2 ⟨by simp, h⟩⟩

theorem mem_checkRevealed {R : Type} (H : Msg → Msg) (run : R → Trace) (Cm : Commitment Bool)
    {rs : List (List Msg × Opening Bool)} {p : List Msg × Opening Bool} (hp : p ∈ rs)
    (hok : Cm.Opens H p.1 p.2 = true) : p ∈ ((checkRevealed (R := R) Cm rs).eval H run).1 := by
  unfold checkRevealed
  rw [flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨p, hp, ?_⟩
  rw [(RP.eval_bind H run _ _).1, eval_ofOC]
  unfold Commitment.Opens at hok
  rw [hok]
  simp [RP.eval]

/-! ## Kept openings pass their checks -/

/-- Everything `p` returns passes its opening check, with that check's calls among `p`'s own. -/
def Passes {R : Type} (H : Msg → Msg) (run : R → Trace) (Cm : Commitment Bool)
    (p : RP R (List (List Msg × Opening Bool))) : Prop :=
  ∀ q ∈ (p.eval H run).1, Cm.Opens H q.1 q.2 = true ∧ ∀ t ∈ ((Cm.opens q.1 q.2).eval H).2, t ∈ ownLog H run p

theorem Passes.ret_nil {R : Type} (H : Msg → Msg) (run : R → Trace) (Cm : Commitment Bool) :
    Passes H run Cm (.ret []) := by
  intro q hq
  simp [RP.eval] at hq

theorem Passes.bind {R α : Type} {H : Msg → Msg} {run : R → Trace} {Cm : Commitment Bool} (p : RP R α)
    {f : α → RP R (List (List Msg × Opening Bool))} (hf : ∀ a, Passes H run Cm (f a)) :
    Passes H run Cm (p.bind f) := by
  intro q hq
  rw [(RP.eval_bind H run _ _).1] at hq
  obtain ⟨h1, h2⟩ := hf _ q hq
  refine ⟨h1, fun t ht => ?_⟩
  rw [ownLog_bind]
  exact List.mem_append_right _ (h2 t ht)

theorem Passes.flatMapRP {R β : Type} {H : Msg → Msg} {run : R → Trace} {Cm : Commitment Bool}
    {f : β → RP R (List (List Msg × Opening Bool))} {l : List β} (hf : ∀ b ∈ l, Passes H run Cm (f b)) :
    Passes H run Cm (Sampling.flatMapRP f l) := by
  intro q hq
  obtain ⟨b, hb, hq', hlog⟩ := flatMapRP_mem H run f l q hq
  obtain ⟨h1, h2⟩ := hf b hb q hq'
  exact ⟨h1, fun t ht => hlog t (h2 t ht)⟩

theorem Passes.checkRevealed {R : Type} (H : Msg → Msg) (run : R → Trace) (Cm : Commitment Bool)
    (rs : List (List Msg × Opening Bool)) : Passes H run Cm (Sampling.checkRevealed (R := R) Cm rs) := by
  intro q hq
  obtain ⟨-, h1, h2⟩ := checkRevealed_mem H run Cm rs q hq
  exact ⟨h1, h2⟩

/-- Two kept openings under one roots at one position agree, unless the calls that checked them collide. -/
theorem agree_of_passes {H : Msg → Msg} {Cm : Commitment Bool} (hb : Cm.Binds H) {l : List (List Msg × Opening Bool)}
    {L : List (Msg × Msg)}
    (hp : ∀ q ∈ l, Cm.Opens H q.1 q.2 = true ∧ ∀ t ∈ ((Cm.opens q.1 q.2).eval H).2, t ∈ L) (hL : ¬ LogColl L)
    {q q' : List Msg × Opening Bool} (hq : q ∈ l) (hq' : q' ∈ l) (h1 : q.1 = q'.1) (h2 : q.2.pos = q'.2.pos) :
    q.2.val = q'.2.val := by
  obtain ⟨rt, o⟩ := q
  obtain ⟨rt', o'⟩ := q'
  dsimp only at h1 h2 ⊢
  subst h1
  by_contra hne
  exact hL (disagree_collision hb h2 hne (hp _ hq).1 (hp _ hq').1 (hp _ hq).2 (hp _ hq').2)

/-! ## The assembled witness and the interior values -/

/-- A kept opening under the statement's roots is what the assembled witness opens there, when the kept openings
agree. -/
theorem val_assemble {H : Msg → Msg} {Cm : Commitment Bool} {s : AuditStatement} {found : List (List Msg × Opening Bool)}
    (hagree : ∀ q ∈ found, ∀ q' ∈ found, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val)
    {q : List Msg × Opening Bool} (hq : q ∈ found) (hr : q.1 = s.roots) (hok : Cm.Opens H s.roots q.2 = true) :
    (assemble s found).val H Cm s q.2.pos = some q.2.val := by
  have hmem : q.2 ∈ (assemble s found).vals :=
    List.mem_filterMap.2 ⟨q, hq, by simp [hr]⟩
  unfold Wit.val
  obtain ⟨o, ho⟩ : ∃ o, (assemble s found).vals.find? (fun o => o.pos == q.2.pos && Cm.Opens H s.roots o) = some o :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨q.2, hmem, by simp [hok]⟩)
  rw [ho]
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨a, ha, hao⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases har : a.1 = s.roots
  · simp only [har, ite_true, Option.some.injEq] at hao
    subst hao
    simp only [Option.map_some, Option.some.injEq]
    exact hagree a ha q hq (har.trans hr.symm) hpo.1
  · simp [har] at hao

/-- The interior values the kept openings under `roots` give: at each position, the first kept one's value there. -/
def ιOf (found : List (List Msg × Opening Bool)) (roots : List Msg) (g : ℕ) : Bool :=
  ((found.find? fun p => decide (p.1 = roots) && p.2.pos == g).map (·.2.val)).getD false

theorem ιOf_eq {found : List (List Msg × Opening Bool)}
    (hagree : ∀ q ∈ found, ∀ q' ∈ found, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val)
    {q : List Msg × Opening Bool} (hq : q ∈ found) : ιOf found q.1 q.2.pos = q.2.val := by
  unfold ιOf
  obtain ⟨p, hp⟩ : ∃ p, found.find? (fun p => decide (p.1 = q.1) && p.2.pos == q.2.pos) = some p :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨q, hq, by simp⟩)
  rw [hp]
  have hpp := List.find?_some hp
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hpp
  exact hagree p (List.mem_of_find?_eq_some hp) q hq hpp.1 hpp.2

/-! ## Waits, as kept lists -/

theorem accList_map {R R' : Type} (acc : R → Prop) [DecidablePred acc] (k : ℕ) (g : R' → R) (l : List R') :
    accList acc k (l.map g) = (accList (fun d => acc (g d)) k l).map g := by
  unfold accList
  rw [List.filter_map, List.map_take]
  rfl

theorem acc_of_accList {R : Type} {acc : R → Prop} [DecidablePred acc] {k : ℕ} {l : List R} {j : ℕ} {r : R}
    (h : (accList acc k l)[j]? = some r) : acc r := by
  have hm := List.mem_of_mem_take (List.mem_of_getElem? h)
  simpa using (List.mem_filter.1 hm).2

theorem accList_length_eq {R : Type} {acc : R → Prop} [DecidablePred acc] {k : ℕ} {l : List R}
    (h : ¬ (accList acc k l).length < k) : (accList acc k l).length = k :=
  le_antisymm (accList_length_le acc k l) (not_lt.1 h)

theorem getElem?_map_pair {R : Type} (run : R → Trace) {l : List R} {j : ℕ} {r : R} (h : l[j]? = some r) :
    (l.map fun r => (r, (run r).seen))[j]? = some (r, (run r).seen) := by
  rw [List.getElem?_map, h]
  rfl

/-! ## The sampling extractor's parts -/

section Extractor

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **A checked proof unit's `E^rep`** at the run `r`, on the prover's runs `run'`: Flock's waits on the unit's session
(`unitExt`'s), `none` when `r` shows no session for `u` or the waits find no witness. -/
noncomputable def unitWit (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (u : ℕ × ℕ) (r : Fin (D.V.k x) → Coin) (ρ : D.UnitCoins x) :
    Option Flock.Wit :=
  match D.session x u (run' r).seen with
  | none => none
  | some (xS, _) =>
    ((firstWait (D.innerK x) (D.sessionCols x u) (Flock.assemble D.flock.parse D.flock.table decode D.flock.read xS)
        (Flock.check D.flock.parse xS)
        ((List.finRange (D.innerWaits x)).map fun i =>
          (List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) r (ρ i j))).eval H run').1

/-- The extractor at one accepted inner run `li`, under the outer run `j`: every checked unit's extraction. -/
noncomputable def innerBody (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (s : AuditStatement)
    (c : ECoins D x) (j : Fin (D.k x)) (li : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen)) :
    RP (Fin (D.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  flatMapRP (fun iu : Fin (nPU D x) × (ℕ × ℕ) => unitExt D decode x s iu.2 li.2.1 li.2.2 (c.2.2 (j, li.1, iu.1)))
    ((List.finRange (nPU D x)).zip (D.checked x li.2.1))

/-- The extractor at one accepted outer run `jo`: the inner wait on the second stage, and `innerBody` at each. -/
noncomputable def outerBody (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (s : AuditStatement)
    (c : ECoins D x) (jo : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen)) :
    RP (Fin (D.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  (waitAccC (D.k x) ((List.finRange (D.cap x)).map fun l => forkOn (D.stage2 x) jo.2.1 (c.2.1 jo.1 l)) []).bind
    fun inner => flatMapRP (innerBody D decode x s c jo.1) ((List.finRange (D.k x)).zip inner)

/-- The outer wait's kept runs, indexed. -/
noncomputable def outerList (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (c : ECoins D x) : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen)) :=
  (List.finRange (D.k x)).zip ((waitAccC (D.k x) ((List.finRange (D.cap x)).map c.1) []).eval H run').1

/-- **What the extractor keeps** at the outer runs `Z`: at an accepted judged run, `Z = outerList`. The list is a
parameter so that unfolding `found` never asks the kernel to evaluate the outer wait over `List.finRange (D.k x)`. -/
noncomputable def found (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (run' : (Fin (D.V.k x) → Coin) → Trace) (c : ECoins D x)
    (Z : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))) : List (List Msg × Opening Bool) :=
  ((flatMapRP (outerBody D decode x s c) Z).eval H run').1

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem found_eq (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : AuditStatement) (run' : (Fin (D.V.k x) → Coin) → Trace) (c : ECoins D x)
    (Z : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))) :
    found H D decode x s run' c Z = ((flatMapRP (outerBody D decode x s c) Z).eval H run').1 :=
  rfl

/-- **The extractor at an accepted run** assembles what it keeps. -/
theorem E_eval (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : ECoins D x) {s : AuditStatement} (hs : D.parse x = some s)
    (hacc : (run H P D.V x ω r₀).out = .acc) :
    ((E D decode).eval H P x ω r₀ c).1 =
      some (assemble s (found H D decode x s (run H P D.V x ω) c (outerList H D x (run H P D.V x ω) c))) := by
  have hacc' : (run H P D.V x ω r₀).seen.out = .acc := hacc
  rw [found_eq]
  simp only [Extractor.eval, E, RP.eval, hs]
  rw [ite_eq_left hacc', (RP.eval_bind _ _ _ _).1, (RP.eval_bind _ _ _ _).1]
  rfl

/-- The kept openings' checks are the extractor's own calls. -/
theorem ownLog_E (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : ECoins D x) {s : AuditStatement}
    (hs : D.parse x = some s) (hacc : (run H P D.V x ω r₀).out = .acc) :
    ∀ t ∈ ownLog H (run H P D.V x ω) (flatMapRP (outerBody D decode x s c) (outerList H D x (run H P D.V x ω) c)),
      t ∈ ownLog H (run H P D.V x ω) ((E D decode).ext x r₀ c) := by
  intro t ht
  have hacc' : (run H P D.V x ω r₀).seen.out = .acc := hacc
  simp only [E, ownLog, hs]
  rw [ite_eq_left hacc', ownLog_bind, ownLog_bind]
  exact List.mem_append_right _ (List.mem_append_left _ ht)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Passes.unitExt (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (s : AuditStatement) (u : ℕ × ℕ) (r : Fin (D.V.k x) → Coin) (v : Seen)
    (ρ : D.UnitCoins x) : Passes H run' D.commit (Sampling.unitExt D decode x s u r v ρ) := by
  unfold Sampling.unitExt
  split
  · exact Passes.ret_nil H run' D.commit
  · refine Passes.bind _ fun o => ?_
    cases o with
    | none => exact Passes.ret_nil H run' D.commit
    | some wF => exact Passes.checkRevealed H run' D.commit _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem Passes.found (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (s : AuditStatement) (c : ECoins D x)
    (Z : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))) :
    Passes H run' D.commit (Sampling.flatMapRP (outerBody D decode x s c) Z) :=
  Passes.flatMapRP fun _ _ => Passes.bind _ fun _ => Passes.flatMapRP fun _ _ =>
    Passes.flatMapRP fun _ _ => Passes.unitExt H D decode x run' s _ _ _ _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- A revealed opening that passes, from a unit whose `E^rep` found a witness, is among the unit's extraction's. -/
theorem unitExt_mem (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (s : AuditStatement) (u : ℕ × ℕ) (r : Fin (D.V.k x) → Coin) (ρ : D.UnitCoins x)
    {xS : Msg} {vS : Seen} (hs : D.session x u (run' r).seen = some (xS, vS)) {wF : Flock.Wit}
    (hw : unitWit H D decode x run' u r ρ = some wF) {q : List Msg × Opening Bool}
    (hq : q ∈ D.reveal s u (run' r).seen wF) (hok : D.commit.Opens H q.1 q.2 = true) :
    q ∈ ((Sampling.unitExt D decode x s u r (run' r).seen ρ).eval H run').1 := by
  unfold unitWit at hw
  unfold Sampling.unitExt
  simp only [hs] at hw ⊢
  rw [(RP.eval_bind _ _ _ _).1, hw]
  exact mem_checkRevealed H run' D.commit hq hok

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem mem_outerList (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace)
    (c : ECoins D x) {j : Fin (D.k x)} {r : Fin (D.V.k x) → Coin}
    (hj : (accList (fun r => (run' r).out = .acc) (D.k x) ((List.finRange (D.cap x)).map c.1))[j.val]? = some r) :
    (j, (r, (run' r).seen)) ∈ outerList H D x run' c := by
  unfold outerList
  rw [waitAccC_eval_nil]
  exact mem_zip_finRange (getElem?_map_pair run' hj)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **A unit's extraction at a kept inner run, under a kept outer run, is kept.** -/
theorem mem_found (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (s : AuditStatement)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (c : ECoins D x) {Z : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))}
    {j : Fin (D.k x)} {r : Fin (D.V.k x) → Coin} (hj : (j, (r, (run' r).seen)) ∈ Z)
    {l : Fin (D.k x)} {d : Fin (D.V.k x) → Coin}
    (hl : (accList (fun d => (run' (forkOn (D.stage2 x) r d)).out = .acc) (D.k x)
      ((List.finRange (D.cap x)).map (c.2.1 j)))[l.val]? = some d)
    {p : Fin (nPU D x)} {u : ℕ × ℕ} (hp : (D.checked x (forkOn (D.stage2 x) r d))[p.val]? = some u)
    {q : List Msg × Opening Bool}
    (hq : q ∈ ((Sampling.unitExt D decode x s u (forkOn (D.stage2 x) r d) (run' (forkOn (D.stage2 x) r d)).seen
      (c.2.2 (j, l, p))).eval H run').1) :
    q ∈ found H D decode x s run' c Z := by
  unfold found
  rw [flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨(j, (r, (run' r).seen)), hj, ?_⟩
  unfold outerBody
  rw [(RP.eval_bind _ _ _ _).1, flatMapRP_eval, waitAccC_eval_nil]
  have hm : ((List.finRange (D.cap x)).map fun l => forkOn (D.stage2 x) r (c.2.1 j l)) =
      ((List.finRange (D.cap x)).map (c.2.1 j)).map (forkOn (D.stage2 x) r) := by
    rw [List.map_map]; rfl
  dsimp only
  rw [hm, accList_map, List.map_map]
  refine List.mem_flatMap.2 ⟨(l, (forkOn (D.stage2 x) r d, (run' (forkOn (D.stage2 x) r d)).seen)),
    mem_zip_finRange ?_, ?_⟩
  · rw [List.getElem?_map, hl]; rfl
  · unfold innerBody
    rw [flatMapRP_eval]
    exact List.mem_flatMap.2 ⟨(p, u), mem_zip_finRange hp, hq⟩

end Extractor

/-! ## The faulty set's proof units -/

/-- **A faulty unit has a wrong proof unit at any interior values**: at slot `v` of the faulty set, whatever interior
values `ι`, some proof unit `ab` of its replay unit, at index `i`, is not correct on `ref_ι`. -/
theorem faulty_choice {H : Msg → Msg} {Cm : Commitment Bool} {s : AuditStatement} {P : Partition} {w : Wit} {v : ℕ}
    (hv : v ∈ faulty H Cm s P w) (ι : ℕ → Bool) :
    ∃ rv, P.rus[v]? = some rv ∧ ∃ i : Fin (P.sizes.getD v 0), ∃ ab, rv[i.val]? = some ab ∧
      ¬ s.prog.Correct (RU.ref s.prog rv (w.val H Cm s) ι) ab := by
  classical
  unfold faulty at hv
  obtain ⟨hlt, hf⟩ := mem_filter.1 hv
  have hlt' := mem_range.1 hlt
  have hget : P.rus[v]? = some P.rus[v] := List.getElem?_eq_getElem hlt'
  refine ⟨P.rus[v], hget, ?_⟩
  unfold Faulty at hf
  push Not at hf
  obtain ⟨u, hu, hnc⟩ := hf P.rus[v] hget ι
  obtain ⟨i, hi, hiu⟩ := List.getElem_of_mem hu
  have hsz : P.sizes.getD v 0 = P.rus[v].length := by simp [Partition.sizes, hlt']
  exact ⟨⟨i, hsz ▸ hi⟩, u, by rw [List.getElem?_eq_getElem hi, hiu], hnc⟩

/-! ## What the deployed sampled verifier is taken to do -/

section Hyps

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- Flock's `γ` at the sampled verifier's rounds, `16·(√(2·k(x)·C) + k(x))²`; the block's `γ` is `c⁴·γF`. -/
noncomputable def γF (D : Deploy Coin Col T) (x : Msg) : ℝ :=
  16 * (Real.sqrt (2 * D.V.k x * D.weight x) + D.V.k x) ^ 2

/-- **Flock's KS at the checked units' sessions**, at every decoder within Flock's radius: on every prover, a
nonnegative weight `β` of each tape and first-stage coins `(ω, r)`, averaging at most `Pr[B]`, such that at `(ω, r)`,
over fresh second-stage coins and `E^rep`'s coins, the run accepts and the unit a selection rule `sel` picks among the
checked ones finds no witness with probability at most `κ_F + √(γ_F·β(ω, r))`. It is what `Claims.Flock.KS` gives at
the session a selection picks, the sampled run up to it as Flock's prover. -/
def SessionKS (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ decode, Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      ∃ β : Ω → (Fin (D.V.k x) → Coin) → ℝ, (∀ ω r, 0 ≤ β ω r) ∧
        Ex (fun p : Ω × (Fin (D.V.k x) → Coin) => β p.1 p.2) ≤ PrB H D.V P x ∧
        ∀ ω r (sel : List (ℕ × ℕ) → Option (Fin (nPU D x))),
          Pr (fun p : (Fin (D.V.k x) → Coin) × (Fin (nPU D x) → D.UnitCoins x) =>
            (run H P D.V x ω (forkOn (D.stage2 x) r p.1)).out = .acc ∧
              ∃ i, sel (D.checked x (forkOn (D.stage2 x) r p.1)) = some i ∧
                ∃ u, (D.checked x (forkOn (D.stage2 x) r p.1))[i.val]? = some u ∧
                  unitWit H D decode x (run H P D.V x ω) u (forkOn (D.stage2 x) r p.1) (p.2 i) = none) ≤
            κF D x + Real.sqrt (γF D x * β ω r)

/-- **The draw law**: on a statement in scope, for each set `B` of replay units, a weight `esc` of the first-stage coins
bounds, at each, the probability over the second stage that the checked units (the first `n_PU`) miss a fixed proof unit
`f(v)` of every replay unit `v ∈ B`, and averages at most `P_a(B)`. -/
def DrawLaw (D : Deploy Coin Col T) : Prop :=
  ∀ x s Pt, D.parse x = some s → D.partition s = some Pt → InScope Pt s →
    ∀ B ⊆ range Pt.rus.length, ∃ esc : (Fin (D.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r (f : ℕ → ℕ), (∀ v ∈ B, f v < Pt.sizes.getD v 0) →
        Pr (fun d : Fin (D.V.k x) → Coin => ∀ v ∈ B, ∀ p : Fin (nPU D x),
          (D.checked x (forkOn (D.stage2 x) r d))[p.val]? ≠ some (v, f v)) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa Pt B

/-- **The verifier refuses what is out of scope**: an accepted run's statement parses, its query evaluates, and the
pair is in scope. -/
def Refuses (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P D.V x ω r).out = .acc →
    ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt ∧ InScope Pt s

/-- **The interior commitments precede the second stage**: the roots `iroots` a run shows for replay unit `v`'s
interior values are the same at fresh second-stage coins. -/
def InteriorFixed (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r d v,
    iroots x (run H P D.V x ω (forkOn (D.stage2 x) r d)).seen v = iroots x (run H P D.V x ω r).seen v

/-- **A checked unit's Flock witness makes it correct**: on a statement in scope, at an accepted run that checks
`u = (v, i)` and shows its session, a Flock witness in the session's relation reveals openings such that any values
`val` and interior values `ι` agreeing with those that pass (under the statement's roots and the interior roots the run
showed for `v`) make proof unit `i` of replay unit `v` correct on `ref_ι`. -/
def SessionOpens (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) : Prop :=
  ∀ x s Pt (Ω : Type) (P : Prover Ω) ω r u xS vS wF,
    D.parse x = some s → D.partition s = some Pt → InScope Pt s →
    (run H P D.V x ω r).out = .acc → (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    D.session x u (run H P D.V x ω r).seen = some (xS, vS) → Flock.check D.flock.parse xS wF = true →
    ∃ rv pu, Pt.rus[u.1]? = some rv ∧ rv[u.2]? = some pu ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ D.reveal s u (run H P D.V x ω r).seen wF, D.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (run H P D.V x ω r).seen u.1 → ι q.2.pos = q.2.val)) →
        s.prog.Correct (RU.ref s.prog rv val ι) pu

/-! ### Sessions that cover several checked units -/

/-- **The slots' sessions are every checked unit's**: at coins `r`, every checked unit has the rounds and the session of
the unit checked at some slot below `S x`. One session per unit is `S = nPU D` (`slotsCover_units`); one session over
every checked unit, as the one-stage audit runs, is `S = 1` (`slotsCover_one`). -/
def SlotsCover (D : Deploy Coin Col T) (S : Msg → ℕ) : Prop :=
  ∀ x r u, (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    ∃ p < S x, ∃ u', (D.checked x r)[p]? = some u' ∧ D.rounds x u = D.rounds x u' ∧ D.session x u = D.session x u'

/-- **`SessionOpens` a session at a time**: on a statement in scope, at an accepted run, the session of the unit checked
at a slot below `S x`, shown with statement `xS`, and one Flock witness in its relation reveal, for every checked unit
that session covers, openings that make that unit correct, as `SessionOpens` says of each unit. -/
def SessionOpensSlots (H : Msg → Msg) (D : Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg)
    (S : Msg → ℕ) : Prop :=
  ∀ x s Pt (Ω : Type) (P : Prover Ω) ω r (p : ℕ) u' xS vS wF,
    D.parse x = some s → D.partition s = some Pt → InScope Pt s →
    (run H P D.V x ω r).out = .acc → p < S x → (D.checked x r)[p]? = some u' →
    D.session x u' (run H P D.V x ω r).seen = some (xS, vS) → Flock.check D.flock.parse xS wF = true →
    ∀ u, (∃ q : Fin (nPU D x), (D.checked x r)[q.val]? = some u) → D.session x u = D.session x u' →
    ∃ rv pu, Pt.rus[u.1]? = some rv ∧ rv[u.2]? = some pu ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ D.reveal s u (run H P D.V x ω r).seen wF, D.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (run H P D.V x ω r).seen u.1 → ι q.2.pos = q.2.val)) →
        s.prog.Correct (RU.ref s.prog rv val ι) pu

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **One session per unit covers every record.** -/
theorem slotsCover_units (D : Deploy Coin Col T) : SlotsCover D (nPU D) := by
  rintro x r u ⟨p, hp⟩
  exact ⟨p.val, p.isLt, u, hp, rfl, rfl⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **One session over every checked unit**: where every unit has the same rounds and session, the first checked unit's
covers them all. -/
theorem slotsCover_one (D : Deploy Coin Col T) (hr : ∀ x u u', D.rounds x u = D.rounds x u')
    (hs : ∀ x u u', D.session x u = D.session x u') : SlotsCover D (fun _ => 1) := by
  rintro x r u ⟨p, hp⟩
  cases hc : D.checked x r with
  | nil => rw [hc] at hp; simp at hp
  | cons u' l => exact ⟨0, Nat.one_pos, u', by simp, hr x u u', hs x u u'⟩

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **`SessionOpens` from its slot form**: each checked unit's session is a slot's, whose witness covers it. -/
theorem sessionOpens_of_slots {H : Msg → Msg} {D : Deploy Coin Col T} {iroots : Msg → Seen → ℕ → List Msg}
    {S : Msg → ℕ} (hc : SlotsCover D S) (h : SessionOpensSlots H D iroots S) : SessionOpens H D iroots := by
  intro x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck
  obtain ⟨p, hp, u', hu', -, hss⟩ := hc x r u hu
  exact h x s Pt Ω P ω r p u' xS vS wF hs hP hin hacc hp hu' (hss ▸ hsess) hck u hu hss

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **The slot form is `SessionOpens`** wherever the slots cover the checked units, one session per unit among them. -/
theorem sessionOpensSlots_iff {H : Msg → Msg} {D : Deploy Coin Col T} {iroots : Msg → Seen → ℕ → List Msg}
    {S : Msg → ℕ} (hc : SlotsCover D S) : SessionOpensSlots H D iroots S ↔ SessionOpens H D iroots :=
  ⟨sessionOpens_of_slots hc, fun h x s Pt Ω P ω r _ _ xS vS wF hs hP hin hacc _ _ hsess hck u hu hss =>
    h x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu (hss ▸ hsess) hck⟩

end Hyps

/-! ## The interior roots from the run's first rounds -/

section Interior

theorem stepWith_msgs_prefix {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      ∃ ext, (stepWith H P x ω vnext n hist as t).msgs = t.msgs ++ ext := by
  intro n
  induction n with
  | zero => intro hist as t; exact ⟨[], by simp [stepWith]⟩
  | succ n ih =>
    intro hist as t
    simp only [stepWith]
    split
    · obtain ⟨ext, he⟩ := ih (hist ++ [(((P.next x hist ω).eval H).1, _)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++
                   [(((P.next x hist ω).eval H).1, ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).1)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
                 vLog := t.vLog ++ ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).2 }
      rw [he]
      exact ⟨_, List.append_assoc _ _ _⟩
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩

/-- **The first `m` rounds read only the verifier's first `m` answers**: two verifiers that answer alike on at most
`m` prover messages give runs whose first `m` messages agree. -/
theorem stepWith_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext vnext' : List Msg → OC Out)
    (m : ℕ) (hv : ∀ as : List Msg, as.length ≤ m → vnext as = vnext' as) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), as.length = t.msgs.length →
      t.msgs.length ≤ m →
      (stepWith H P x ω vnext n hist as t).msgs.take m = (stepWith H P x ω vnext' n hist as t).msgs.take m := by
  intro n
  induction n with
  | zero => intro hist as t _ _; rfl
  | succ n ih =>
    intro hist as t hlen hm
    rcases Nat.lt_or_ge t.msgs.length m with hlt | hge
    · have hv' := hv (as ++ [((P.next x hist ω).eval H).1]) (by simp; omega)
      simp only [stepWith]
      rw [hv']
      split
      · exact ih _ _ _ (by simp [hlen]) (by simp; omega)
      · rfl
      · rfl
    · obtain ⟨e, he⟩ := stepWith_msgs_prefix H P x ω vnext (n + 1) hist as t
      obtain ⟨e', he'⟩ := stepWith_msgs_prefix H P x ω vnext' (n + 1) hist as t
      have hm' : t.msgs.length = m := le_antisymm hm hge
      rw [he, he', List.take_append_of_le_length hm'.ge, List.take_append_of_le_length hm'.ge]

/-- **Acceptance is the verifier's answer**: rounds that end accepting had the verifier answer `acc` to some prover
messages. -/
theorem stepWith_acc_next {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω vnext n hist as t).out = .acc → ∃ as', ((vnext as').eval H).1 = .acc := by
  intro n
  induction n with
  | zero => intro hist as t h; simp [stepWith] at h
  | succ n ih =>
    intro hist as t h
    simp only [stepWith] at h
    split at h
    · exact ih _ _ _ h
    · exact ⟨_, by assumption⟩
    · simp at h

variable {Coin : Type} [Inhabited Coin]

/-- **An accepted run's verifier answered `acc`** to some prover messages and coins. -/
theorem run_acc_next {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (ω : Ω)
    (r : Fin (V.k x) → Coin) (h : (run H P V x ω r).out = .acc) : ∃ as cs, ((V.next x as cs).eval H).1 = .acc := by
  obtain ⟨as, has⟩ := stepWith_acc_next H P x ω _ _ _ _ _ h
  exact ⟨as, _, has⟩

/-- **A run's first `m` messages read only its first `m` rounds' coins.** -/
theorem run_msgs_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg) (ω : Ω) (m : ℕ)
    (r r' : Fin (V.k x) → Coin) (h : ∀ t : Fin (V.k x), t.val < m → r t = r' t) :
    (run H P V x ω r).msgs.take m = (run H P V x ω r').msgs.take m := by
  unfold run
  refine stepWith_take H P x ω _ _ m (fun as has => ?_) _ _ _ _ rfl (Nat.zero_le _)
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi' : i < m := lt_of_lt_of_le (List.mem_range.1 hi) has
  unfold coinSeq
  split
  · exact h _ hi'
  · rfl

/-- **`InteriorFixed` from the interiors' place in the run**: where the second stage starts at round `t₀ x` and the
interior roots a run shows are read off its first `t₀ x` messages (the interiors are registered after the draw and
before any proof), fresh second-stage coins keep them. -/
theorem interiorFixed_of_prefix {Col T : Type} (H : Msg → Msg) (D : Deploy Coin Col T)
    (iroots : Msg → Seen → ℕ → List Msg) (t₀ : Msg → ℕ) (hst : ∀ x, ∀ t ∈ D.stage2 x, t₀ x ≤ t.val)
    (hread : ∀ x (s s' : Seen) v, s.msgs.take (t₀ x) = s'.msgs.take (t₀ x) → iroots x s v = iroots x s' v) :
    InteriorFixed H D iroots := by
  intro x Ω P ω r d v
  refine hread x _ _ v (run_msgs_take H P D.V x ω (t₀ x) _ _ fun t ht => ?_)
  have hn : t ∉ D.stage2 x := fun hmem => absurd (hst x t hmem) (not_le.2 ht)
  simp [forkOn, hn]

end Interior

/-! ## The failure event -/

section Split

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The extractor's coins for one outer run: its inner wait's, and at each inner run each checked unit's `E^rep`'s. -/
abbrev E1 (D : Deploy Coin Col T) (x : Msg) :=
  (Fin (D.cap x) → Fin (D.V.k x) → Coin) × (Fin (D.k x) → Fin (nPU D x) → D.UnitCoins x)

def toE1 (D : Deploy Coin Col T) (x : Msg) (c : ECoins D x) (j : Fin (D.k x)) : E1 D x :=
  (c.2.1 j, fun l p => c.2.2 (j, l, p))

/-- The extractor's coins as the outer wait's and one `E1` per outer run. -/
def ecoinsEquiv (D : Deploy Coin Col T) (x : Msg) :
    ECoins D x ≃ (Fin (D.cap x) → Fin (D.V.k x) → Coin) × (Fin (D.k x) → E1 D x) where
  toFun c := (c.1, toE1 D x c)
  invFun p := (p.1, fun j => (p.2 j).1, fun t => (p.2 t.1).2 t.2.1 t.2.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- The checked unit `(v, i)` is among the first `n_PU` the run `r'` checks, and its `E^rep` finds a witness. -/
def Good (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (r' : Fin (D.V.k x) → Coin) (v i : ℕ)
    (u : Fin (nPU D x) → D.UnitCoins x) : Prop :=
  ∃ p : Fin (nPU D x), (D.checked x r')[p.val]? = some (v, i) ∧ (unitWit H D decode x run' (v, i) r' (u p)).isSome

/-- **An outer run bad for `B`**, its second stage `g`: the inner wait comes up short, or, for some choice `f` of one
proof unit in each replay unit of `B`, no kept inner run checks a chosen one and finds its witness. -/
def BadOuter (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (Pt : Partition) (B : Finset ℕ)
    (g : (Fin (D.V.k x) → Coin) → Fin (D.V.k x) → Coin) (e : E1 D x) : Prop :=
  (accList (fun d => (run' (g d)).out = .acc) (D.k x) ((List.finRange (D.cap x)).map e.1)).length < D.k x ∨
    ∃ f ∈ (univ : Finset ((v : B) → Fin (Pt.sizes.getD v 0))),
      AllBad (fun d => (run' (g d)).out = .acc) (D.k x)
        (fun d u => ∀ v : B, ¬ Good H D decode x run' (g d) v.1 (f v).1 u) ((List.finRange (D.cap x)).map e.1) e.2

/-- The faulty sets the relation at `δ` refuses: those of acceptance probability at most `δ`. -/
noncomputable def S₁ (d : Design) (Pt : Partition) (δ : ℝ) : Finset (Finset ℕ) :=
  (range Pt.rus.length).powerset.filter fun B => d.Pa Pt B ≤ δ

/-- **The extraction's failure event**: the statement is in scope, and the outer wait comes up short or, for some
refused faulty set `B`, every kept outer run is bad for `B`. -/
def Fails (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (δ : ℝ) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (c : ECoins D x) : Prop :=
  ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt ∧ InScope Pt s ∧
    ((accList (fun r => (run' r).out = .acc) (D.k x) ((List.finRange (D.cap x)).map c.1)).length < D.k x ∨
      ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (D.k x)
        (fun r e => BadOuter H D decode x run' Pt B (forkOn (D.stage2 x) r) e) ((List.finRange (D.cap x)).map c.1)
        (toE1 D x c))

/-- **The split**: at an accepted run whose extraction fails, the failure event holds or the extractor's own calls
collide. Otherwise take `B` the assembled witness's faulty set, refused at `δ`, and a kept outer run `r`; at the
interior values the kept openings under `r`'s interior roots give, each `v ∈ B` has a wrong proof unit `f(v)`
(`faulty_choice`); a kept inner run that checks `f(v)` and finds its witness would make it correct (`SessionOpens`, the
kept openings agreeing), so `r` is bad for `B`. -/
theorem split {H : Msg → Msg} {D : Deploy Coin Col T} [DecidableEq Col] [Zero Col] (decode : List (ℕ × Col) → List T)
    (δ : ℝ) (iroots : Msg → Seen → ℕ → List Msg) (hbind : D.commit.Binds H) (hopens : SessionOpens H D iroots)
    (hfix : InteriorFixed H D iroots) (hrefuse : Refuses H D) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r₀ : Fin (D.V.k x) → Coin) (c : ECoins D x) (hacc : (run H P D.V x ω r₀).out = .acc)
    (hfail : ¬ (R H D.commit D.parse D.partition δ).ext x ((E D decode).eval H P x ω r₀ c).1) :
    Fails H D decode δ x (run H P D.V x ω) c ∨ OwnColl H (E D decode) P x ω r₀ c := by
  classical
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hrefuse x Ω P ω r₀ hacc
  by_cases hcoll : OwnColl H (E D decode) P x ω r₀ c
  · exact Or.inr hcoll
  refine Or.inl ⟨s, Pt, hs, hPt, hin, ?_⟩
  set run' := run H P D.V x ω with hrun'
  set fd := found H D decode x s run' c (outerList H D x run' c) with hfd
  have hpass : ∀ q ∈ fd, D.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((D.commit.opens q.1 q.2).eval H).2, t ∈ ownLog H run' ((E D decode).ext x r₀ c) := by
    intro q hq
    rw [hfd, found_eq] at hq
    obtain ⟨h1, h2⟩ := Passes.found H D decode x run' s c _ q hq
    exact ⟨h1, fun t ht => ownLog_E H D decode P x ω r₀ c hs hacc t (h2 t ht)⟩
  have hagree : ∀ q ∈ fd, ∀ q' ∈ fd, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => agree_of_passes hbind hpass hcoll hq hq' h1 h2
  rw [E_eval H D decode P x ω r₀ c hs hacc] at hfail
  set w := assemble s fd with hw
  have hB : faulty H D.commit s Pt w ∈ S₁ s.design Pt δ := by
    unfold S₁
    refine mem_filter.2 ⟨mem_powerset.2 (filter_subset _ _), ?_⟩
    by_contra hlt
    exact hfail ⟨s, Pt, hs, hPt, lt_of_not_ge hlt⟩
  by_cases hshort : (accList (fun r => (run' r).out = .acc) (D.k x) ((List.finRange (D.cap x)).map c.1)).length <
    D.k x
  · exact Or.inl hshort
  refine Or.inr ⟨_, hB, accList_length_eq hshort, fun j r hj => ?_⟩
  have hjZ := mem_outerList H D x run' c hj
  by_cases hshort2 : (accList (fun d => (run' (forkOn (D.stage2 x) r d)).out = .acc) (D.k x)
    ((List.finRange (D.cap x)).map (c.2.1 j))).length < D.k x
  · exact Or.inl hshort2
  have hch : ∀ v : faulty H D.commit s Pt w, ∃ rv, Pt.rus[v.1]? = some rv ∧
      ∃ i : Fin (Pt.sizes.getD v.1 0), ∃ ab, rv[i.val]? = some ab ∧
        ¬ s.prog.Correct (RU.ref s.prog rv (w.val H D.commit s) (ιOf fd (iroots x (run' r).seen v.1))) ab :=
    fun v => faulty_choice v.2 _
  choose rv hrv f ab hab hnc using hch
  refine Or.inr ⟨f, mem_univ _, accList_length_eq hshort2, fun l d hl v hgood => ?_⟩
  obtain ⟨p, hp, hsome⟩ := hgood
  obtain ⟨wF, hwF⟩ := Option.isSome_iff_exists.1 hsome
  have hacc' : (run' (forkOn (D.stage2 x) r d)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  have hwF' := hwF
  unfold unitWit at hwF'
  cases hsess : D.session x (v.1, (f v).1) (run' (forkOn (D.stage2 x) r d)).seen with
  | none =>
    rw [hsess] at hwF'
    exact absurd hwF' (by simp)
  | some xv =>
    obtain ⟨xS, vS⟩ := xv
    rw [hsess] at hwF'
    have hchk := firstWait_some H run' _ _ _ _ _ wF hwF'
    obtain ⟨rv', pu, hrv', hab', hcor⟩ :=
      hopens x s Pt Ω P ω (forkOn (D.stage2 x) r d) (v.1, (f v).1) xS vS wF hs hPt hin hacc' ⟨p, hp⟩ hsess hchk
    have hcorr := hcor (w.val H D.commit s) (ιOf fd (iroots x (run' r).seen v.1)) fun q hq hok => by
      have hqf : q ∈ fd := mem_found H D decode x s run' c hjZ hl hp
        (unitExt_mem H D decode x run' s _ _ _ hsess hwF hq hok)
      refine ⟨fun hr => ?_, fun hi => ?_⟩
      · rw [hr] at hok
        exact val_assemble hagree hqf hr hok
      · rw [hfix x Ω P ω r d v.1] at hi
        rw [← hi]
        exact ιOf_eq hagree hqf
    rw [hrv v, Option.some.injEq] at hrv'
    subst hrv'
    rw [hab v, Option.some.injEq] at hab'
    have hnc' := hnc v
    rw [hab'] at hnc'
    exact hnc' hcorr

end Split

/-! ## The unions' sizes, against the waits' slack -/

theorem sum_range_getD : ∀ l : List ℕ, ∑ v ∈ range l.length, l.getD v 0 = l.sum
  | [] => by simp
  | a :: l => by
    rw [List.length_cons, sum_range_succ', List.sum_cons]
    simp only [List.getD_cons_succ, List.getD_cons_zero, sum_range_getD l]
    omega

/-- The choices of one proof unit in each replay unit of `B` number at most `2^{n_PU}`. -/
theorem card_choices_le {Pt : Partition} {B : Finset ℕ} (hB : B ⊆ range Pt.rus.length) :
    Fintype.card ((v : B) → Fin (Pt.sizes.getD v 0)) ≤ 2 ^ Pt.sizes.sum := by
  rw [Fintype.card_pi]
  simp only [Fintype.card_fin]
  have hlen : Pt.rus.length = Pt.sizes.length := by simp [Partition.sizes]
  calc ∏ v : B, Pt.sizes.getD v 0 ≤ ∏ v : B, 2 ^ Pt.sizes.getD v 0 :=
        Finset.prod_le_prod fun v _ => Nat.lt_two_pow_self.le
    _ = 2 ^ ∑ v : B, Pt.sizes.getD v 0 := Finset.prod_pow_eq_pow_sum _ _ _
    _ ≤ 2 ^ Pt.sizes.sum := by
      refine Nat.pow_le_pow_right (by norm_num) ?_
      rw [Finset.sum_coe_sort B (fun v => Pt.sizes.getD v 0), ← sum_range_getD, ← hlen]
      exact sum_le_sum_of_subset hB

theorem card_S₁_le (d : Design) (Pt : Partition) (δ : ℝ) : (S₁ d Pt δ).card ≤ 2 ^ Pt.rus.length :=
  (card_filter_le _ _).trans (by rw [card_powerset, card_range])

/-- At most `2^m` sets, `m ≤ M + 1`, spread over `k = 2^40·(M + 1)` accepted runs cost at most the slack. -/
theorem card_rpow_le_slack {N m M : ℕ} (hN : N ≤ 2 ^ m) (hm : m ≤ M + 1) :
    (N : ℝ) ^ (1 / ((2 ^ 40 * (M + 1) : ℕ) : ℝ)) ≤ slack := by
  have hk : (0 : ℝ) < ((2 ^ 40 * (M + 1) : ℕ) : ℝ) := by positivity
  calc (N : ℝ) ^ (1 / ((2 ^ 40 * (M + 1) : ℕ) : ℝ)) ≤ ((2 : ℝ) ^ m) ^ (1 / ((2 ^ 40 * (M + 1) : ℕ) : ℝ)) :=
        Real.rpow_le_rpow (Nat.cast_nonneg _) (by exact_mod_cast hN) (by positivity)
    _ = (2 : ℝ) ^ ((m : ℝ) * (1 / ((2 ^ 40 * (M + 1) : ℕ) : ℝ))) := by
        rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num)]
    _ ≤ slack := by
        unfold slack
        refine Real.rpow_le_rpow_of_exponent_le (by norm_num) ?_
        rw [Real.rpow_neg (by norm_num), mul_one_div, div_le_iff₀ hk]
        have hm' : (m : ℝ) ≤ M + 1 := by exact_mod_cast hm
        push_cast
        norm_num
        nlinarith

/-! ## The failure event's probability -/

section Bound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- The slack bound at the waits' `k`. -/
theorem rpow_k_le_slack (D : Deploy Coin Col T) (x : Msg) {N m : ℕ} (hN : N ≤ 2 ^ m)
    (hm : m ≤ nRU D x + nPU D x + 1) : (N : ℝ) ^ (1 / (D.k x : ℝ)) ≤ slack := by
  unfold Deploy.k
  exact card_rpow_le_slack hN hm

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem nRU_eq {D : Deploy Coin Col T} {x : Msg} {s : AuditStatement} {Pt : Partition} (hs : D.parse x = some s)
    (hPt : D.partition s = some Pt) : nRU D x = Pt.rus.length := by
  simp only [nRU, hs, Deploy.part, hPt, Option.getD_some]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem nPU_eq {D : Deploy Coin Col T} {x : Msg} {s : AuditStatement} {Pt : Partition} (hs : D.parse x = some s)
    (hPt : D.partition s = some Pt) : nPU D x = Pt.sizes.sum := by
  simp only [nPU, hs, Deploy.part, hPt, Option.getD_some]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem γF_nonneg (D : Deploy Coin Col T) (x : Msg) : 0 ≤ γF D x := by
  unfold γF; positivity

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem k_pos (D : Deploy Coin Col T) (x : Msg) : 0 < D.k x := by
  unfold Deploy.k; positivity

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem cap_pos (D : Deploy Coin Col T) (x : Msg) : 0 < D.cap x := by
  unfold Deploy.cap; exact Nat.mul_pos (by positivity) (k_pos D x)

theorem Pr_or_le {α : Type} [Fintype α] {p q r : α → Prop} {a b : ℝ} (h : ∀ s, p s → q s ∨ r s)
    (hq : Pr q ≤ a) (hr : Pr r ≤ b) : Pr p ≤ a + b :=
  (Pr_mono h).trans ((Pr_union_le q r).trans (add_le_add hq hr))

omit [Inhabited Coin] in
/-- **One outer run's inner wait**: at first-stage coins `r` where the second stage misses a fixed proof unit of each
replay unit of `B` with probability at most `esc`, and the checked unit any selection picks finds no witness with
probability at most `κs`, the inner wait (times its acceptance) comes up short or keeps only runs that miss or fail,
for some choice, with probability at most `2k/K + e^{−k/8} + c·(esc + κs)`. -/
theorem inner_bad_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (Pt : Partition) (hPU : nPU D x = Pt.sizes.sum) {B : Finset ℕ}
    (hB : B ⊆ range Pt.rus.length) (r : Fin (D.V.k x) → Coin) {esc κs : ℝ} (hesc0 : 0 ≤ esc) (hκs0 : 0 ≤ κs)
    (hesc : ∀ f : ℕ → ℕ, (∀ v ∈ B, f v < Pt.sizes.getD v 0) →
      Pr (fun d : Fin (D.V.k x) → Coin => ∀ v ∈ B, ∀ p : Fin (nPU D x),
        (D.checked x (forkOn (D.stage2 x) r d))[p.val]? ≠ some (v, f v)) ≤ esc)
    (hsess : ∀ sel : List (ℕ × ℕ) → Option (Fin (nPU D x)),
      Pr (fun p : (Fin (D.V.k x) → Coin) × (Fin (nPU D x) → D.UnitCoins x) =>
        (run' (forkOn (D.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (D.checked x (forkOn (D.stage2 x) r p.1)) = some i ∧
            ∃ u, (D.checked x (forkOn (D.stage2 x) r p.1))[i.val]? = some u ∧
              unitWit H D decode x run' u (forkOn (D.stage2 x) r p.1) (p.2 i) = none) ≤ κs) :
    Pr (fun d : Fin (D.V.k x) → Coin => (run' (forkOn (D.stage2 x) r d)).out = .acc) *
        Pr (fun e : E1 D x => BadOuter H D decode x run' Pt B (forkOn (D.stage2 x) r) e) ≤
      2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + slack * (esc + κs) := by
  have hy : ∀ f ∈ (univ : Finset ((v : B) → Fin (Pt.sizes.getD v 0))),
      Pr (fun p : (Fin (D.V.k x) → Coin) × (Fin (nPU D x) → D.UnitCoins x) =>
        (run' (forkOn (D.stage2 x) r p.1)).out = .acc ∧
          ∀ v : B, ¬ Good H D decode x run' (forkOn (D.stage2 x) r p.1) v.1 (f v).1 p.2) ≤ esc + κs := by
    intro f _
    obtain ⟨f', hf'⟩ : ∃ f' : ℕ → ℕ, ∀ v (h : v ∈ B), f' v = (f ⟨v, h⟩).1 :=
      ⟨fun v => if h : v ∈ B then (f ⟨v, h⟩).1 else 0, fun v h => by simp [h]⟩
    obtain ⟨sel, hsel⟩ : ∃ sel : List (ℕ × ℕ) → Option (Fin (nPU D x)), ∀ l,
        (∃ q : Fin (nPU D x), ∃ v ∈ B, l[q.val]? = some (v, f' v)) →
          ∃ i, sel l = some i ∧ ∃ v ∈ B, l[i.val]? = some (v, f' v) := by
      classical
      refine ⟨fun l => (List.finRange (nPU D x)).find? fun q => decide (∃ v ∈ B, l[q.val]? = some (v, f' v)),
        fun l ⟨q, hq⟩ => ?_⟩
      obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1
        ((List.find?_isSome (p := fun q : Fin (nPU D x) => decide (∃ v ∈ B, l[q.val]? = some (v, f' v)))).2
          ⟨q, List.mem_finRange q, decide_eq_true hq⟩)
      have hpi := List.find?_some hi
      exact ⟨i, hi, of_decide_eq_true hpi⟩
    have hmiss := (Pr_fst (β := Fin (nPU D x) → D.UnitCoins x) (fun d : Fin (D.V.k x) → Coin =>
      ∀ v ∈ B, ∀ q : Fin (nPU D x), (D.checked x (forkOn (D.stage2 x) r d))[q.val]? ≠ some (v, f' v))).trans_le
      (hesc f' fun v hv => by rw [hf' v hv]; exact (f ⟨v, hv⟩).2)
    refine Pr_or_le (fun p hp => ?_) hmiss (hsess sel)
    by_cases hm : ∀ v ∈ B, ∀ q : Fin (nPU D x), (D.checked x (forkOn (D.stage2 x) r p.1))[q.val]? ≠ some (v, f' v)
    · exact Or.inl hm
    right
    push Not at hm
    obtain ⟨q, v, hv, hq⟩ : ∃ q : Fin (nPU D x), ∃ v ∈ B,
        (D.checked x (forkOn (D.stage2 x) r p.1))[q.val]? = some (v, f' v) := by
      obtain ⟨v, hv, q, hq⟩ := hm
      exact ⟨q, v, hv, hq⟩
    obtain ⟨i, hi, v', hv', hi'⟩ := hsel _ ⟨q, v, hv, hq⟩
    rw [hf' v' hv'] at hi'
    refine ⟨hp.1, i, hi, _, hi', ?_⟩
    have hng := hp.2 ⟨v', hv'⟩
    unfold Good at hng
    push Not at hng
    simpa using hng i hi'
  have hw := wait_bad_le (fun d => (run' (forkOn (D.stage2 x) r d)).out = .acc) (D.k x) (D.cap x) (cap_pos D x)
    (k_pos D x) (univ : Finset ((v : B) → Fin (Pt.sizes.getD v 0)))
    (fun f d u => ∀ v : B, ¬ Good H D decode x run' (forkOn (D.stage2 x) r d) v.1 (f v).1 u)
    (add_nonneg hesc0 hκs0) hy
  have hcard : ((univ : Finset ((v : B) → Fin (Pt.sizes.getD v 0))).card : ℝ) ^ (1 / (D.k x : ℝ)) ≤ slack :=
    rpow_k_le_slack D x (by rw [card_univ]; exact card_choices_le hB) (by rw [← hPU]; omega)
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard (add_nonneg hesc0 hκs0)))

set_option synthInstance.maxSize 1024 in
omit [Inhabited Coin] in
/-- **One tape's outer wait**: on a statement in scope at `(s, P)`, where the draw law holds and the checked unit any
selection picks at first-stage coins `r` finds no witness with probability at most `κs(r)`, the outer wait (times its
acceptance) comes up short or, for some refused `B`, keeps only runs bad for `B`, with probability at most
`T + c·(T + c·(δ + E[κs]))`, `T = 2k/K + e^{−k/8}`. -/
theorem outer_bad_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {δ : ℝ} (hδ : 0 ≤ δ)
    (x : Msg) (run' : (Fin (D.V.k x) → Coin) → Trace) (s : AuditStatement) (Pt : Partition) (hPU : nPU D x = Pt.sizes.sum)
    (hRU : nRU D x = Pt.rus.length) (κs : (Fin (D.V.k x) → Coin) → ℝ) (hκs0 : ∀ r, 0 ≤ κs r)
    (hdraw : ∀ B ⊆ range Pt.rus.length, ∃ esc : (Fin (D.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r (f : ℕ → ℕ), (∀ v ∈ B, f v < Pt.sizes.getD v 0) →
        Pr (fun d : Fin (D.V.k x) → Coin => ∀ v ∈ B, ∀ p : Fin (nPU D x),
          (D.checked x (forkOn (D.stage2 x) r d))[p.val]? ≠ some (v, f v)) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa Pt B)
    (hsess : ∀ r (sel : List (ℕ × ℕ) → Option (Fin (nPU D x))),
      Pr (fun p : (Fin (D.V.k x) → Coin) × (Fin (nPU D x) → D.UnitCoins x) =>
        (run' (forkOn (D.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (D.checked x (forkOn (D.stage2 x) r p.1)) = some i ∧
            ∃ u, (D.checked x (forkOn (D.stage2 x) r p.1))[i.val]? = some u ∧
              unitWit H D decode x run' u (forkOn (D.stage2 x) r p.1) (p.2 i) = none) ≤ κs r) :
    Pr (fun r : Fin (D.V.k x) → Coin => (run' r).out = .acc) *
        Pr (fun c : (Fin (D.cap x) → Fin (D.V.k x) → Coin) × (Fin (D.k x) → E1 D x) =>
          (accList (fun r => (run' r).out = .acc) (D.k x) ((List.finRange (D.cap x)).map c.1)).length < D.k x ∨
            ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (D.k x)
              (fun r e => BadOuter H D decode x run' Pt B (forkOn (D.stage2 x) r) e)
              ((List.finRange (D.cap x)).map c.1) c.2) ≤
      2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) +
        slack * (2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + slack * (δ + Ex κs)) := by
  have hc := slack_pos
  have hEκ : 0 ≤ Ex κs := Ex_nonneg hκs0
  have hy : ∀ B ∈ S₁ s.design Pt δ,
      Pr (fun p : (Fin (D.V.k x) → Coin) × E1 D x =>
        (run' p.1).out = .acc ∧ BadOuter H D decode x run' Pt B (forkOn (D.stage2 x) p.1) p.2) ≤
      2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + slack * (δ + Ex κs) := by
    intro B hB
    obtain ⟨hBsub, hBδ⟩ := mem_filter.1 hB
    obtain ⟨esc, hesc0, hesc, hEesc⟩ := hdraw B (mem_powerset.1 hBsub)
    refine (Pr_acc_forkOn (D.stage2 x) (fun r => (run' r).out = .acc)
      (fun r e => BadOuter H D decode x run' Pt B (forkOn (D.stage2 x) r) e)
      fun r d e => by rw [forkOn_forkOn']).trans_le ?_
    calc Ex (fun r => Pr (fun d => (run' (forkOn (D.stage2 x) r d)).out = .acc) *
          Pr (fun e : E1 D x => BadOuter H D decode x run' Pt B (forkOn (D.stage2 x) r) e))
        ≤ Ex (fun r => 2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + slack * (esc r + κs r)) :=
          Ex_mono fun r => inner_bad_le H D decode x run' Pt hPU (mem_powerset.1 hBsub) r (hesc0 r) (hκs0 r)
            (hesc r) (hsess r)
      _ = 2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + slack * (Ex esc + Ex κs) := by
          rw [Ex_add, Ex_const, Ex_mul_left, Ex_add]
      _ ≤ 2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + slack * (δ + Ex κs) := by
          have := hEesc.trans hBδ
          gcongr
  have hy0 : 0 ≤ 2 * (D.k x : ℝ) / D.cap x + Real.exp (-(D.k x : ℝ) / 8) + slack * (δ + Ex κs) := by
    positivity
  have hw := wait_bad_le (fun r => (run' r).out = .acc) (D.k x) (D.cap x) (cap_pos D x) (k_pos D x)
    (S₁ s.design Pt δ) (fun B r e => BadOuter H D decode x run' Pt B (forkOn (D.stage2 x) r) e) hy0 hy
  have hcard : ((S₁ s.design Pt δ).card : ℝ) ^ (1 / (D.k x : ℝ)) ≤ slack :=
    rpow_k_le_slack D x (card_S₁_le _ _ _) (by rw [← hRU]; omega)
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard hy0))

end Bound

/-! ## KS -/

section KS

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

set_option synthInstance.maxSize 1024 in
/-- **The failure event's probability**: over tapes, judged coins and the extractor's coins, the judged run accepts
and the extraction's failure event holds with probability at most `κ(δ) + √(γ·Pr[B])`. Out of scope the event is
empty. In scope, it is independent of the judged coins; at each tape, the outer wait's bound (`outer_bad_le`) at
Flock's error `κ_F + √(γ_F·β(ω, r))` averages, by Jensen, to `κ(δ) + c²·√(γ_F·E[β]) ≤ κ(δ) + √(γ·Pr[B])`. -/
theorem Pr_fails_le (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (hdec : Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) {δ : ℝ} (hδ : 0 ≤ δ) (hsess : SessionKS H D)
    (hdraw : DrawLaw D) (x : Msg) (hε : 0 ≤ D.εtable x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × ECoins D x =>
      (run H P D.V x s.1 s.2.1).out = .acc ∧ Fails H D decode δ x (run H P D.V x s.1) s.2.2) ≤
      κ D δ x + Real.sqrt (γ D x * PrB H D.V P x) := by
  refine (Pr_indep (fun ω r => (run H P D.V x ω r).out = .acc)
    (fun ω c => Fails H D decode δ x (run H P D.V x ω) c)).trans_le ?_
  have hc := slack_pos
  have hκ0 := κ_pos D hδ x hε
  by_cases hsc : ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt ∧ InScope Pt s
  swap
  · have h0 : ∀ ω, Pr (fun c : ECoins D x => Fails H D decode δ x (run H P D.V x ω) c) = 0 := fun ω =>
      (Pr_congr fun c => ⟨fun ⟨s, Pt, h1, h2, h3, _⟩ => hsc ⟨s, Pt, h1, h2, h3⟩, False.elim⟩).trans Pr_false
    simp only [h0, mul_zero, Ex_const]
    exact add_nonneg hκ0.le (Real.sqrt_nonneg _)
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  obtain ⟨β, hβ0, hβ, hκs⟩ := hsess decode hdec x Ω P
  have hper : ∀ ω, Pr (fun r => (run H P D.V x ω r).out = .acc) *
      Pr (fun c : ECoins D x => Fails H D decode δ x (run H P D.V x ω) c) ≤
      κ D δ x + slack ^ 2 * Ex (fun r => Real.sqrt (γF D x * β ω r)) := by
    intro ω
    have hF : Pr (fun c : ECoins D x => Fails H D decode δ x (run H P D.V x ω) c) =
        Pr (fun c : (Fin (D.cap x) → Fin (D.V.k x) → Coin) × (Fin (D.k x) → E1 D x) =>
          (accList (fun r => (run H P D.V x ω r).out = .acc) (D.k x)
            ((List.finRange (D.cap x)).map c.1)).length < D.k x ∨
            ∃ B ∈ S₁ s.design Pt δ, AllBad (fun r => (run H P D.V x ω r).out = .acc) (D.k x)
              (fun r e => BadOuter H D decode x (run H P D.V x ω) Pt B (forkOn (D.stage2 x) r) e)
              ((List.finRange (D.cap x)).map c.1) c.2) := by
      refine Eq.trans ?_ (Pr_equiv (ecoinsEquiv D x) _)
      refine Pr_congr fun c => ⟨fun ⟨s', Pt', hs', hPt', _, h⟩ => ?_, fun h => ⟨s, Pt, hs, hPt, hin, h⟩⟩
      rw [hs, Option.some.injEq] at hs'
      subst hs'
      rw [hPt, Option.some.injEq] at hPt'
      subst hPt'
      exact h
    rw [hF]
    refine (outer_bad_le H D decode hδ x (run H P D.V x ω) s Pt (nPU_eq hs hPt) (nRU_eq hs hPt)
      (fun r => κF D x + Real.sqrt (γF D x * β ω r)) (fun r => add_nonneg (κF_nonneg D x hε) (Real.sqrt_nonneg _))
      (hdraw x s Pt hs hPt hin) (hκs ω)).trans (le_of_eq ?_)
    rw [Ex_add, Ex_const]
    unfold κ
    ring
  calc Ex (fun ω => Pr (fun r => (run H P D.V x ω r).out = .acc) *
        Pr (fun c : ECoins D x => Fails H D decode δ x (run H P D.V x ω) c))
      ≤ Ex (fun ω => κ D δ x + slack ^ 2 * Ex (fun r => Real.sqrt (γF D x * β ω r))) := Ex_mono hper
    _ = κ D δ x + slack ^ 2 * Ex (fun p : Ω × (Fin (D.V.k x) → Coin) => Real.sqrt (γF D x * β p.1 p.2)) := by
        rw [Ex_add, Ex_const, Ex_mul_left,
          Ex_prod (fun p : Ω × (Fin (D.V.k x) → Coin) => Real.sqrt (γF D x * β p.1 p.2))]
    _ ≤ κ D δ x + slack ^ 2 * Real.sqrt (γF D x * PrB H D.V P x) := by
        refine add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ (by positivity))
        refine (Ex_sqrt_le _ fun p => mul_nonneg (γF_nonneg D x) (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_)
        rw [Ex_mul_left]
        exact mul_le_mul_of_nonneg_left hβ (γF_nonneg D x)
    _ = κ D δ x + Real.sqrt (γ D x * PrB H D.V P x) := by
        rw [show γ D x = (slack ^ 2) ^ 2 * γF D x by unfold γ γF; ring, mul_assoc,
          Real.sqrt_mul (pow_nonneg (pow_nonneg hc.le 2) 2) (γF D x * PrB H D.V P x),
          Real.sqrt_sq (pow_nonneg hc.le 2)]

set_option synthInstance.maxSize 1024 in
/-- **Sampling: KS from Flock's KS.** Flock's KS at the checked units' sessions (`SessionKS`), the trace commitment's
binding at `H`, and the deployed sampled verifier's named facts (the draw law, what a unit's Flock witness reveals,
the interior commitments fixed by the first stage, the scope refusal) give the block's `KS`: the split
(`split`) puts a failed extraction at an accepted run in the failure event or an own-call collision, and the failure
event has probability at most `κ(δ) + √(γ·Pr[B])` (`Pr_fails_le`). -/
theorem KS_of {H : Msg → Msg} {D : Deploy Coin Col T} (iroots : Msg → Seen → ℕ → List Msg)
    (hbind : D.commit.Binds H) (hsess : SessionKS H D) (hopens : SessionOpens H D iroots)
    (hfix : InteriorFixed H D iroots) (hdraw : DrawLaw D) (hrefuse : Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x) :
    KS H D := by
  intro δ decode hδ hdec x _
  exact KSLin_of_ownColl H D.V (R H D.commit D.parse D.partition δ) (E D decode) (κ D δ) (γ D)
    (fun x Ω P s => (run H P D.V x s.1 s.2.1).out = .acc ∧ Fails H D decode δ x (run H P D.V x s.1) s.2.2)
    (fun x Ω P s hacc hfail =>
      (split decode δ iroots hbind hopens hfix hrefuse x P s.1 s.2.1 s.2.2 hacc hfail).imp_left fun h => ⟨hacc, h⟩)
    (fun x Ω _ _ P => Pr_fails_le H D decode hdec hδ hsess hdraw x (hε x) Ω P) x

/-- **Sampling: bits from Flock's KS**: `KS_of`, with the extractor's expected reruns and own SHA-512 calls. -/
theorem Bits_of {H : Msg → Msg} {D : Deploy Coin Col T} (iroots : Msg → Seen → ℕ → List Msg)
    (hbind : D.commit.Binds H) (hsess : SessionKS H D) (hopens : SessionOpens H D iroots)
    (hfix : InteriorFixed H D iroots) (hdraw : DrawLaw D) (hrefuse : Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x)
    (hruns : ∀ decode, ExtractRunsExpLe H (E D decode)
      (fun x => 1 + D.k x + (D.k x : ℝ) ^ 2 * (1 + nPU D x * D.innerWaits x * D.innerK x)))
    (hown : ∀ decode, ExtractOwnLe H (E D decode) (fun x => D.k x ^ 2 * nPU D x * D.openCalls x)) : Bits H D :=
  Bits_of_KS (KS_of iroots hbind hsess hopens hfix hdraw hrefuse hε) hruns hown hε

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- A set of at least `t` replay units has acceptance probability at most the design's miss at `t`. -/
theorem Pa_le_miss (d : Design) (P : Partition) {t : ℕ} {B : Finset ℕ} (hB : B ⊆ range P.rus.length)
    (ht : t ≤ B.card) : d.Pa P B ≤ d.miss P t := by
  have : Finite {B : Finset ℕ // B ⊆ range P.rus.length ∧ t ≤ B.card} :=
    Finite.of_injective (fun B => (⟨B.1, mem_powerset.2 B.2.1⟩ : (range P.rus.length).powerset))
      fun a b h => Subtype.ext (congrArg (fun z : (range P.rus.length).powerset => z.1) h)
  exact le_ciSup (f := fun B : {B : Finset ℕ // B ⊆ range P.rus.length ∧ t ≤ B.card} => d.Pa P B.1)
    (Set.finite_range _).bddAbove ⟨B, hB, ht⟩

/-- **Sampling: the count form from KS**: `KS` at `δ = max(0, miss(t))` (`missAt`), a faulty set in `M_δ` having
fewer than `t` units (`Pa_le_miss`). -/
theorem KSCount_of_KS {H : Msg → Msg} {D : Deploy Coin Col T} (h : KS H D) : KSCount H D := by
  intro t decode hdec x hx Ω _ _ P
  have h0 : 0 ≤ missAt D t x := by
    unfold missAt
    split
    · exact le_max_left _ _
    · exact le_rfl
  have hk := h (missAt D t x) decode h0 hdec x hx Ω P
  unfold PrBad at hk ⊢
  refine le_trans (Pr_mono fun s hs => ⟨hs.1, fun hR => hs.2 ?_⟩) hk
  revert hR
  cases ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 with
  | none => exact id
  | some w =>
    rintro ⟨s', Pt, hs, hPt, hM⟩
    refine ⟨s', Pt, hs, hPt, lt_of_not_ge fun ht => ?_⟩
    have hmx : missAt D t x = max 0 (s'.design.miss Pt t) := by
      simp only [missAt, hs, Deploy.part, hPt, Option.getD_some]
    have hM' : missAt D t x < s'.design.Pa Pt (faulty H D.commit s' Pt w) := hM
    rw [hmx] at hM'
    exact absurd ((Pa_le_miss s'.design Pt (by classical exact filter_subset _ _) ht).trans (le_max_right _ _))
      (not_le.2 hM')

end KS

end Claims.Sampling
