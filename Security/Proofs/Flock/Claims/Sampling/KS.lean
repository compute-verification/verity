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

end Claims.Sampling
