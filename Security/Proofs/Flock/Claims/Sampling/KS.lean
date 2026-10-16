import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Escape
import Proofs.Flock.Claims.Aggregate

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
theorem val_assemble {H : Msg → Msg} {Cm : Commitment Bool} {s : Stmt} {found : List (List Msg × Opening Bool)}
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
    ((firstWait (D.innerK x) (D.sessionCols x u) (Flock.assemble D.flock.parse decode D.flock.read xS)
        (Flock.check D.flock.parse xS)
        ((List.finRange (D.innerWaits x)).map fun i =>
          (List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) r (ρ i j))).eval H run').1

/-- The extractor at one accepted inner run `li`, under the outer run `j`: every checked unit's extraction. -/
noncomputable def innerBody (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (s : Stmt)
    (c : ECoins D x) (j : Fin (D.k x)) (li : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen)) :
    RP (Fin (D.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  flatMapRP (fun iu : Fin (nPU D x) × (ℕ × ℕ) => unitExt D decode x s iu.2 li.2.1 li.2.2 (c.2.2 (j, li.1, iu.1)))
    ((List.finRange (nPU D x)).zip (D.checked x li.2.1))

/-- The extractor at one accepted outer run `jo`: the inner wait on the second stage, and `innerBody` at each. -/
noncomputable def outerBody (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (s : Stmt)
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
    (s : Stmt) (run' : (Fin (D.V.k x) → Coin) → Trace) (c : ECoins D x)
    (Z : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))) : List (List Msg × Opening Bool) :=
  ((flatMapRP (outerBody D decode x s c) Z).eval H run').1

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem found_eq (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Stmt) (run' : (Fin (D.V.k x) → Coin) → Trace) (c : ECoins D x)
    (Z : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))) :
    found H D decode x s run' c Z = ((flatMapRP (outerBody D decode x s c) Z).eval H run').1 :=
  rfl

/-- **The extractor at an accepted run** assembles what it keeps. -/
theorem E_eval (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : ECoins D x) {s : Stmt} (hs : D.parse x = some s)
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
    (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : ECoins D x) {s : Stmt}
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
    (run' : (Fin (D.V.k x) → Coin) → Trace) (s : Stmt) (u : ℕ × ℕ) (r : Fin (D.V.k x) → Coin) (v : Seen)
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
    (run' : (Fin (D.V.k x) → Coin) → Trace) (s : Stmt) (c : ECoins D x)
    (Z : List (Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen))) :
    Passes H run' D.commit (Sampling.flatMapRP (outerBody D decode x s c) Z) :=
  Passes.flatMapRP fun _ _ => Passes.bind _ fun _ => Passes.flatMapRP fun _ _ =>
    Passes.flatMapRP fun _ _ => Passes.unitExt H D decode x run' s _ _ _ _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- A revealed opening that passes, from a unit whose `E^rep` found a witness, is among the unit's extraction's. -/
theorem unitExt_mem (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (s : Stmt) (u : ℕ × ℕ) (r : Fin (D.V.k x) → Coin) (ρ : D.UnitCoins x)
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
theorem mem_found (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (s : Stmt)
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
theorem faulty_choice {H : Msg → Msg} {Cm : Commitment Bool} {s : Stmt} {P : Partition} {w : Wit} {v : ℕ}
    (hv : v ∈ faulty H Cm s P w) (ι : ℕ → Bool) :
    ∃ rv, P.rus[v]? = some rv ∧ ∃ i : Fin (P.sizes.getD v 0), ∃ ab, rv[i.val]? = some ab ∧
      ¬ s.prog.Correct (RU.ref s.prog rv (w.val H Cm s) ι) ab.1 ab.2 := by
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
  ∀ x s Pt, D.parse x = some s → D.partition s.prog s.query = some Pt → InScope Pt s →
    ∀ B ⊆ range Pt.rus.length, ∃ esc : (Fin (D.V.k x) → Coin) → ℝ, (∀ r, 0 ≤ esc r) ∧
      (∀ r (f : ℕ → ℕ), (∀ v ∈ B, f v < Pt.sizes.getD v 0) →
        Pr (fun d : Fin (D.V.k x) → Coin => ∀ v ∈ B, ∀ p : Fin (nPU D x),
          (D.checked x (forkOn (D.stage2 x) r d))[p.val]? ≠ some (v, f v)) ≤ esc r) ∧
      Ex esc ≤ s.design.Pa Pt B

/-- **The verifier refuses what is out of scope**: an accepted run's statement parses, its query evaluates, and the
pair is in scope. -/
def Refuses (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P D.V x ω r).out = .acc →
    ∃ s Pt, D.parse x = some s ∧ D.partition s.prog s.query = some Pt ∧ InScope Pt s

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
    D.parse x = some s → D.partition s.prog s.query = some Pt → InScope Pt s →
    (run H P D.V x ω r).out = .acc → (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    D.session x u (run H P D.V x ω r).seen = some (xS, vS) → Flock.check D.flock.parse xS wF = true →
    ∃ rv a b, Pt.rus[u.1]? = some rv ∧ rv[u.2]? = some (a, b) ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ D.reveal s u (run H P D.V x ω r).seen wF, D.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (run H P D.V x ω r).seen u.1 → ι q.2.pos = q.2.val)) →
        s.prog.Correct (RU.ref s.prog rv val ι) a b

end Hyps

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
  ∃ s Pt, D.parse x = some s ∧ D.partition s.prog s.query = some Pt ∧ InScope Pt s ∧
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
        ¬ s.prog.Correct (RU.ref s.prog rv (w.val H D.commit s) (ιOf fd (iroots x (run' r).seen v.1))) ab.1 ab.2 :=
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
    obtain ⟨rv', a, b, hrv', hab', hcor⟩ :=
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

end Claims.Sampling
