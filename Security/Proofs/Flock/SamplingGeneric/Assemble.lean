module

public import Proofs.Flock.SamplingGeneric.Split
public import Proofs.Flock.Claims.Sampling.KS

/-!
# The sampled compiler's extraction: assembling and binding

`Compiled.E` at an accepted run assembles the openings its sampled proof blocks' extractions keep. This file proves
`KSOf`'s deterministic half and its bound at one outer run: an extraction that fails gives the failure event the
probabilistic half bounds, or a `Bext` collision (`split`, `Claims.Sampling`'s `split` with Flock's `E^rep` replaced by
`Arg`'s extractor at each sampled proof block's session).

* `Good`: **a good proof block** at a run: the run samples it, and its session is extracted (`UnitOk`).
* `innerBody`, `outerBody`, `outerList`, `found`: **the extractor's parts**, and what it keeps (`E_eval`); the kept
  openings pass their checks, with the checks' calls among the extractor's own (`passes_found`, `ownLog_E`), and the
  revealed openings that pass, of a sampled block's extracted witness, are kept (`unitExt_mem`, `mem_found`).
* `Fails`, `BadOuter`: **the failure event**: the outer wait comes up short or, for some faulty set refused at `δ`,
  every kept outer run comes up short on its inner wait or keeps only inner runs at which no block of the set is good.
* `split`: **the deterministic half**. At an accepted run whose extraction fails, `Fails` holds or the extractor's own
  calls collide. Two kept openings under one roots at one position that disagree are a collision of the calls that
  checked them (`CommitmentScheme.Binds`), so without one the kept openings agree and assemble a witness; its faulty
  set is refused at `δ`, and a good block of it would be correct at the kept openings (`Sessions.opens`, with the
  interior roots fixed by the first stage, `Sessions.interior`).
* `inner_bad_le`: **one outer run's inner wait**: it comes up short or keeps only runs bad for `B`, at most
  `2k/K + e^{−k/8} + c·(esc + κs)`, at a weight `esc` of the sample missing `B` and `κs` of the session at the block
  any selection picks not extracted (`SessionKS`'s form).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition Opening InScope waitAccC forkOn flatMapRP checkRevealed assemble
  ownLog ownLog_bind accList AllBad OwnColl flatMapRP_eval waitAccC_eval_nil accList_map mem_zip_finRange
  getElem?_map_pair mem_checkRevealed agree_of_passes val_assemble ιOf ιOf_eq faulty_wrong acc_of_accList
  accList_length_eq wait_bad_le card_rpow_le_slack Pr_or_le slack)
open Claims.Recursion (serve viewing)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## A good proof block -/

/-- **Block `v` is good at the run `r'`** at the coins `u` of every block: `r'` samples `v`, and `v`'s session is
extracted. -/
def Good (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (r' : Fin (L.V.k x) → Coin) (v : ℕ) (u : (p : Fin (L.nPU x)) → A.EC (L.sh x p)) : Prop :=
  v ∈ L.checked x r' ∧ ∃ hv : v < L.nPU x, L.UnitOk H x run' v r' (u ⟨v, hv⟩)

theorem _root_.Claims.Rel.ext_some {W : Type} {R : Rel W} {x : Msg} {o : Option W} (h : R.ext x o) :
    ∃ w, o = some w ∧ R x w := by
  cases o with
  | none => exact False.elim h
  | some w => exact ⟨w, rfl, h⟩

/-! ## The extractor's parts -/

/-- The extractor at one accepted inner run `li`, under the outer run `j`: every sampled proof block's extraction. -/
noncomputable def innerBody (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (c : L.ECoins x) (j : Fin (L.k x))
    (li : Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  flatMapRP (fun u : ℕ =>
      if hu : u < L.nPU x then L.unitExt x s u li.2.1 li.2.2 (c.2.2 (j, li.1, ⟨u, hu⟩)) else .ret [])
    (L.checked x li.2.1)

/-- The extractor at one accepted outer run `jo`: the inner wait on the second stage, and `innerBody` at each. -/
noncomputable def outerBody (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (c : L.ECoins x)
    (jo : Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen)) :
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Opening Bool)) :=
  (waitAccC (L.k x) ((List.finRange (L.cap x)).map fun l => forkOn (L.stage2 x) jo.2.1 (c.2.1 jo.1 l)) []).bind
    fun inner => flatMapRP (L.innerBody x s c jo.1) ((List.finRange (L.k x)).zip inner)

/-- The outer wait's kept runs, indexed. -/
noncomputable def outerList (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (c : L.ECoins x) : List (Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen)) :=
  (List.finRange (L.k x)).zip ((waitAccC (L.k x) ((List.finRange (L.cap x)).map c.1) []).eval H run').1

/-- **What the extractor keeps** at the outer runs `Z`: at an accepted judged run, `Z = outerList`. The list is a
parameter so that unfolding `found` never asks the kernel to evaluate the outer wait over `List.finRange (L.k x)`. -/
noncomputable def found (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (c : L.ECoins x)
    (Z : List (Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen))) : List (List Msg × Opening Bool) :=
  ((flatMapRP (L.outerBody x s c) Z).eval H run').1

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem found_eq (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (c : L.ECoins x)
    (Z : List (Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen))) :
    L.found H x s run' c Z = ((flatMapRP (L.outerBody x s c) Z).eval H run').1 :=
  rfl

omit [Inhabited Coin] in
/-- The extractor's program, with its outer body named. -/
theorem E_ext_eq (L : Compiled Coin A) (x : Msg) (r₀ : Fin (L.V.k x) → Coin) (c : L.ECoins x) :
    L.E.ext x r₀ c = .rerun r₀ fun v₀ =>
      match L.parse x with
      | none => .ret none
      | some s =>
        if v₀.out = .acc then
          (waitAccC (L.k x) ((List.finRange (L.cap x)).map c.1) []).bind fun outer =>
            (flatMapRP (L.outerBody x s c) ((List.finRange (L.k x)).zip outer)).bind fun found =>
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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem passes_found (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (s : AuditStatement) (c : L.ECoins x) (Z : List (Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen))) :
    Sampling.Passes H run' L.commit (flatMapRP (L.outerBody x s c) Z) :=
  Sampling.Passes.flatMapRP fun _ _ => Sampling.Passes.bind _ fun _ => Sampling.Passes.flatMapRP fun _ _ =>
    Sampling.Passes.flatMapRP fun u _ => by
      split
      · exact L.passes_unitExt H x run' s _ _ _ _
      · exact Sampling.Passes.ret_nil H run' L.commit

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
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

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem mem_outerList (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (c : L.ECoins x) {j : Fin (L.k x)} {r : Fin (L.V.k x) → Coin}
    (hj : (accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1))[j.val]? = some r) :
    (j, (r, (run' r).seen)) ∈ L.outerList H x run' c := by
  unfold outerList
  rw [waitAccC_eval_nil]
  exact mem_zip_finRange (getElem?_map_pair run' hj)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **A block's extraction at a kept inner run, under a kept outer run, is kept**, when the inner run samples it. -/
theorem mem_found (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Trace) (c : L.ECoins x)
    {Z : List (Fin (L.k x) × ((Fin (L.V.k x) → Coin) × Seen))} {j : Fin (L.k x)} {r : Fin (L.V.k x) → Coin}
    (hj : (j, (r, (run' r).seen)) ∈ Z) {l : Fin (L.k x)} {d : Fin (L.V.k x) → Coin}
    (hl : (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x)
      ((List.finRange (L.cap x)).map (c.2.1 j)))[l.val]? = some d)
    {v : ℕ} (hv : v ∈ L.checked x (forkOn (L.stage2 x) r d)) (hvlt : v < L.nPU x) {q : List Msg × Opening Bool}
    (hq : q ∈ ((L.unitExt x s v (forkOn (L.stage2 x) r d) (run' (forkOn (L.stage2 x) r d)).seen
      (c.2.2 (j, l, ⟨v, hvlt⟩))).eval H run').1) :
    q ∈ L.found H x s run' c Z := by
  unfold found
  rw [flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨(j, (r, (run' r).seen)), hj, ?_⟩
  unfold outerBody
  rw [(RP.eval_bind _ _ _ _).1, flatMapRP_eval, waitAccC_eval_nil]
  have hm : ((List.finRange (L.cap x)).map fun l => forkOn (L.stage2 x) r (c.2.1 j l)) =
      ((List.finRange (L.cap x)).map (c.2.1 j)).map (forkOn (L.stage2 x) r) := by
    rw [List.map_map]; rfl
  dsimp only
  rw [hm, accList_map, List.map_map]
  refine List.mem_flatMap.2 ⟨(l, (forkOn (L.stage2 x) r d, (run' (forkOn (L.stage2 x) r d)).seen)),
    mem_zip_finRange ?_, ?_⟩
  · rw [List.getElem?_map, hl]; rfl
  · unfold innerBody
    rw [flatMapRP_eval]
    refine List.mem_flatMap.2 ⟨v, hv, ?_⟩
    dsimp only
    rw [dite_eq_left hvlt]
    exact hq

/-! ## The failure event -/

/-- The extractor's coins for one outer run: its inner wait's, and at each inner run each proof block's. -/
abbrev E1 (L : Compiled Coin A) (x : Msg) :=
  (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → (p : Fin (L.nPU x)) → A.EC (L.sh x p))

def toE1 (L : Compiled Coin A) (x : Msg) (c : L.ECoins x) (j : Fin (L.k x)) : L.E1 x :=
  (c.2.1 j, fun l p => c.2.2 (j, l, p))

/-- The extractor's coins as the outer wait's and one `E1` per outer run. -/
def ecoinsEquiv (L : Compiled Coin A) (x : Msg) :
    L.ECoins x ≃ (Fin (L.cap x) → Fin (L.V.k x) → Coin) × (Fin (L.k x) → L.E1 x) where
  toFun c := (c.1, L.toE1 x c)
  invFun p := (p.1, fun j => (p.2 j).1, fun t => (p.2 t.1).2 t.2.1 t.2.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- **An outer run bad for `B`**, its second stage `g`: the inner wait comes up short, or no kept inner run has a good
block of `B`. -/
def BadOuter (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (B : Finset ℕ) (g : (Fin (L.V.k x) → Coin) → Fin (L.V.k x) → Coin) (e : L.E1 x) : Prop :=
  (accList (fun d => (run' (g d)).out = .acc) (L.k x) ((List.finRange (L.cap x)).map e.1)).length < L.k x ∨
    AllBad (fun d => (run' (g d)).out = .acc) (L.k x)
      (fun d u => ∀ v ∈ B, ¬ L.Good H x run' (g d) v u) ((List.finRange (L.cap x)).map e.1) e.2

/-- **The extraction's failure event**: the statement is in scope, and the outer wait comes up short or, for some
faulty set `B` refused at `δ`, every kept outer run is bad for `B`. -/
def Fails (H : Msg → Msg) (L : Compiled Coin A) (δ : ℝ) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    (c : L.ECoins x) : Prop :=
  ∃ s Pt, L.parse x = some s ∧ L.partition s = some Pt ∧ InScope Pt s ∧
    ((accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length < L.k x ∨
      ∃ B ∈ Sampling.S₁ s.design Pt δ, AllBad (fun r => (run' r).out = .acc) (L.k x)
        (fun r e => L.BadOuter H x run' B (forkOn (L.stage2 x) r) e) ((List.finRange (L.cap x)).map c.1)
        (L.toE1 x c))

/-- **The split**: at an accepted run whose extraction fails, the failure event holds or the extractor's own calls
collide. Otherwise the kept openings agree (`agree_of_passes`); take `B` the assembled witness's faulty set, refused
at `δ`, and a kept outer run `r`. At the interior values the kept openings under `r`'s interior roots give, the proof
block at each `v ∈ B` is wrong (`faulty_wrong`); a kept inner run at which `v` is good would make it correct
(`Sessions.opens`, its revealed openings kept by `mem_found`, the interior roots `r`'s by `Sessions.interior`), so `r`
is bad for `B`. -/
theorem split {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H) (δ : ℝ)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (L.V.k x) → Coin) (c : L.ECoins x)
    (hacc : (run H P L.V x ω r₀).out = .acc)
    (hfail : ¬ (Sampling.R H L.commit L.parse L.partition δ).ext x (L.E.eval H P x ω r₀ c).1) :
    L.Fails H δ x (run H P L.V x ω) c ∨ OwnColl H L.E P x ω r₀ c := by
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
  by_cases hshort : (accList (fun r => (run' r).out = .acc) (L.k x) ((List.finRange (L.cap x)).map c.1)).length <
    L.k x
  · exact Or.inl hshort
  refine Or.inr ⟨_, hB, accList_length_eq hshort, fun j r hj => ?_⟩
  have hjZ := L.mem_outerList H x run' c hj
  by_cases hshort2 : (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x)
    ((List.finRange (L.cap x)).map (c.2.1 j))).length < L.k x
  · exact Or.inl hshort2
  refine Or.inr ⟨accList_length_eq hshort2, fun l d hl v hv hgood => ?_⟩
  obtain ⟨b, hb, hnc⟩ := faulty_wrong hv (ιOf fd (L.iroots x (run' r).seen v))
  obtain ⟨hvc, hvlt, xS, vS, hsh, hsess, hR⟩ := hgood
  obtain ⟨wS, hwS, hRw⟩ := Rel.ext_some hR
  have hacc' : (run' (forkOn (L.stage2 x) r d)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  obtain ⟨b', hb', hcor⟩ :=
    hS.opens x s Pt Ω P ω (forkOn (L.stage2 x) r d) v xS vS wS hs hPt hin hacc' hvc hsess hRw
  have hcorr := hcor (w.val H L.commit s) (ιOf fd (L.iroots x (run' r).seen v)) fun q hq hok => by
    have hqf : q ∈ fd := L.mem_found H x s run' c hjZ hl hvc hvlt
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

/-! ## One outer run's inner wait -/

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem k_pos (L : Compiled Coin A) (x : Msg) : 0 < L.k x := by
  unfold Compiled.k; positivity

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem cap_pos (L : Compiled Coin A) (x : Msg) : 0 < L.cap x := by
  unfold Compiled.cap; exact Nat.mul_pos (by positivity) (L.k_pos x)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem k_le_cap (L : Compiled Coin A) (x : Msg) : L.k x ≤ L.cap x := by
  unfold Compiled.cap; exact Nat.le_mul_of_pos_left _ (by positivity)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- The slack bound at the waits' `k`. -/
theorem rpow_k_le_slack (L : Compiled Coin A) (x : Msg) {N m : ℕ} (hN : N ≤ 2 ^ m) (hm : m ≤ 2 * L.nPU x + 1) :
    (N : ℝ) ^ (1 / (L.k x : ℝ)) ≤ slack := by
  unfold Compiled.k
  exact card_rpow_le_slack hN hm

omit [Inhabited Coin] in
/-- **One outer run's inner wait**: at first-stage coins `r` where the sample misses every block of `B` with
probability at most `esc`, and the session at the sampled block any selection picks is not extracted with probability
at most `κs`, the inner wait (times its acceptance) comes up short or keeps only runs bad for `B` with probability at
most `2k/K + e^{−k/8} + c·(esc + κs)`. -/
theorem inner_bad_le (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Trace)
    {B : Finset ℕ} (hB : ∀ v ∈ B, v < L.nPU x) (r : Fin (L.V.k x) → Coin) {esc κs : ℝ} (hesc0 : 0 ≤ esc)
    (hκs0 : 0 ≤ κs)
    (hesc : Pr (fun d : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B) ≤ esc)
    (hsess : ∀ sel : List ℕ → Option (Fin (L.nPU x)),
      Pr (fun p : (Fin (L.V.k x) → Coin) × ((u : Fin (L.nPU x)) → A.EC (L.sh x u)) =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
            i.val ∈ L.checked x (forkOn (L.stage2 x) r p.1) ∧
            ¬ L.UnitOk H x run' i (forkOn (L.stage2 x) r p.1) (p.2 i)) ≤ κs) :
    Pr (fun d : Fin (L.V.k x) → Coin => (run' (forkOn (L.stage2 x) r d)).out = .acc) *
        Pr (fun e : L.E1 x => L.BadOuter H x run' B (forkOn (L.stage2 x) r) e) ≤
      2 * (L.k x : ℝ) / L.cap x + Real.exp (-(L.k x : ℝ) / 8) + slack * (esc + κs) := by
  have hy : ∀ _u ∈ ({()} : Finset Unit),
      Pr (fun p : (Fin (L.V.k x) → Coin) × ((u : Fin (L.nPU x)) → A.EC (L.sh x u)) =>
        (run' (forkOn (L.stage2 x) r p.1)).out = .acc ∧
          ∀ v ∈ B, ¬ L.Good H x run' (forkOn (L.stage2 x) r p.1) v p.2) ≤ esc + κs := by
    intro _ _
    classical
    let sel : List ℕ → Option (Fin (L.nPU x)) := fun l =>
      (List.finRange (L.nPU x)).find? fun q => decide (q.val ∈ B ∧ q.val ∈ l)
    have hsel : ∀ (l : List ℕ) (q : Fin (L.nPU x)), q.val ∈ B → q.val ∈ l →
        ∃ i, sel l = some i ∧ i.val ∈ B ∧ i.val ∈ l := by
      intro l q hqB hql
      obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1
        ((List.find?_isSome (p := fun q : Fin (L.nPU x) => decide (q.val ∈ B ∧ q.val ∈ l))).2
          ⟨q, List.mem_finRange q, decide_eq_true ⟨hqB, hql⟩⟩)
      have hpi := List.find?_some hi
      exact ⟨i, hi, of_decide_eq_true hpi⟩
    have hmiss := (Pr_fst (β := (u : Fin (L.nPU x)) → A.EC (L.sh x u)) (fun d : Fin (L.V.k x) → Coin =>
      ∀ u ∈ L.checked x (forkOn (L.stage2 x) r d), u ∉ B)).trans_le hesc
    refine Pr_or_le (fun p hp => ?_) hmiss (hsess sel)
    by_cases hm : ∀ u ∈ L.checked x (forkOn (L.stage2 x) r p.1), u ∉ B
    · exact Or.inl hm
    right
    push Not at hm
    obtain ⟨v, hvc, hvB⟩ := hm
    obtain ⟨i, hi, hiB, hic⟩ := hsel _ ⟨v, hB v hvB⟩ hvB hvc
    exact ⟨hp.1, i, hi, hic, fun hf => hp.2 i hiB ⟨hic, i.isLt, hf⟩⟩
  have hw := wait_bad_le (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x) (L.cap x) (L.cap_pos x)
    (L.k_pos x) ({()} : Finset Unit)
    (fun _ d u => ∀ v ∈ B, ¬ L.Good H x run' (forkOn (L.stage2 x) r d) v u)
    (add_nonneg hesc0 hκs0) hy
  have hcard : (({()} : Finset Unit).card : ℝ) ^ (1 / (L.k x : ℝ)) ≤ slack :=
    L.rpow_k_le_slack x (m := 0) (by simp) (by omega)
  have hE : Pr (fun e : L.E1 x => L.BadOuter H x run' B (forkOn (L.stage2 x) r) e) =
      Pr (fun e : L.E1 x =>
        (accList (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x)
          ((List.finRange (L.cap x)).map e.1)).length < L.k x ∨
        ∃ _u ∈ ({()} : Finset Unit), AllBad (fun d => (run' (forkOn (L.stage2 x) r d)).out = .acc) (L.k x)
          (fun d u => ∀ v ∈ B, ¬ L.Good H x run' (forkOn (L.stage2 x) r d) v u)
          ((List.finRange (L.cap x)).map e.1) e.2) :=
    Pr_congr fun e => by simp [BadOuter]
  rw [hE]
  exact hw.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_right hcard (add_nonneg hesc0 hκs0)))

end Compiled

end Claims.SamplingGeneric
