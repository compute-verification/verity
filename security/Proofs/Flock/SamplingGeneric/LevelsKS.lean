module

public import Definitions.Claims.SamplingLevels
public import Proofs.Flock.SamplingGeneric.Assemble
public import Proofs.Flock.SamplingGeneric.Session

/-!
# The compiler at levels is knowledge sound: one level

`CompilerKSLev` on the statements whose levels are one (`CompiledLev.OneLevel`: no top partition), on `CompilerKS`'s
machinery: the extractor, waits, errors and numbers are `Compiled`'s, the sessions' bound is `Compiled.sessionKS_of`,
and the outer wait unions over the sets of proof blocks the levels' sample misses with probability at most `δ`
(`Compiled.outer_bad_le_pa` at `missLev`).

* `missLev`: the levels' miss of a set of proof blocks, as a real.
* `FailsOne`, `pr_failsOne_le`: **the failure event at one level** and its probability, `κ(δ) + √(γ·Pr[B])`, from
  `Samples`' one-level clause and `Unions none`.
* `splitOne`: **the deterministic half**. At an accepted run whose extraction fails, without an own-call collision the
  kept openings assemble a witness whose incorrect set at each proof block's best interior (`bestIota`) the levels miss
  with probability at most `δ`; a block of it found at a kept inner run would be correct (`Sessions.opens` at the block
  as its own top block, the interior roots fixed by `Sessions.interior`).
* `ks_one`: **`CompiledLev.KS` at one level**: `KSLinOn` on the statements in scope with no top partition.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeLev RLev slot topOf incorrectLev levM forkOn accList AllBad
  OwnColl agree_of_passes val_assemble ιOf ιOf_eq acc_of_accList accList_length_eq assemble KSLinOn Bext)

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The levels' miss** of a set `B` of proof blocks: the level-by-level sample misses `B`'s slots, as a real. -/
noncomputable def missLev (Pt : Partition) (Lv : Verity.Levels (Fin Pt.blocks.length)) (B : Finset ℕ) : ℝ :=
  (Lv.design.missProb (B.image (slot Pt))).toReal

/-- **One level at `x`**: no statement `x` parses to has a top partition. -/
def OneLevel (L : CompiledLev Coin A) (x : Msg) : Prop := ∀ s, L.parse x = some s → L.top s = none

/-- **The extraction's failure event at one level**: the statement is in scope with no top partition, and the outer
wait comes up short or, for some set `B` the levels miss with probability at most `δ`, every kept outer run is bad for
`B` (`Compiled.BadOuter`). -/
def FailsOne (L : CompiledLev Coin A) (δ : ℝ) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (c : L.ECoins x) : Prop :=
  ∃ s Pt Lv, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s Pt = some Lv ∧ InScopeLev Pt Lv (L.top s) s ∧
    L.top s = none ∧
    ((accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
      ∃ B ∈ (range Pt.blocks.length).powerset.filter (fun B => missLev Pt Lv B ≤ δ),
        AllBad (fun r => (run' r).out = .acc) (L.kOut x)
          (fun r e => L.toCompiled.BadOuter x run' found B (forkOn (L.secondStage x) r) e)
          ((List.finRange (L.capOut x)).map c.1) (L.toCompiled.toE1 x c))

set_option synthInstance.maxSize 1024 in
/-- **The failure event's probability at one level**: `Compiled.pr_le_of_tape` with, at each tape, the outer wait's
bound at the levels' miss (`Compiled.outer_bad_le_pa`), its draws from `Samples`' one-level clause and its union from
`Unions none`. -/
theorem pr_failsOne_le (H : Msg → Msg) (L : CompiledLev Coin A) {δ : ℝ} (hδ : 0 ≤ δ) (hS : L.Sessions H) (x : Msg)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (found : Ω → (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop)
    (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (hsess : ∀ ω r (sel : List ℕ → Option (Fin (L.nPU x))),
      Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω (forkOn (L.secondStage x) r p.1)).out = .acc ∧
          ∃ i, sel (L.checked x (forkOn (L.secondStage x) r p.1)) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.secondStage x) r p.1))[i.val]? = some u.val ∧
              ¬ found ω u (forkOn (L.secondStage x) r p.1) (p.2 i u)) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) :
    Pr (fun s : Ω × (Fin (L.V.k x) → Coin) × L.ECoins x =>
      (run H P L.V x s.1 s.2.1).out = .acc ∧ L.FailsOne δ x (run H P L.V x s.1) (found s.1) s.2.2) ≤
      L.κ δ x + Real.sqrt (L.γ x * PrB H L.V P x) := by
  refine L.toCompiled.pr_le_of_tape H x Ω P (fun ω c => L.FailsOne δ x (run H P L.V x ω) (found ω) c) β hβ0 hβ
    fun ω => ?_
  have hκ0 := L.toCompiled.κ_nonneg hδ x
  by_cases hsc : ∃ s Pt Lv, L.parse x = some s ∧ L.partition s = some Pt ∧ L.levels s Pt = some Lv ∧
      InScopeLev Pt Lv (L.top s) s ∧ L.top s = none
  swap
  · have h0 : Pr (fun c : L.ECoins x => L.FailsOne δ x (run H P L.V x ω) (found ω) c) = 0 :=
      (Pr_congr fun c => ⟨fun ⟨s, Pt, Lv, h1, h2, h3, h4, h5, _⟩ => hsc ⟨s, Pt, Lv, h1, h2, h3, h4, h5⟩,
        False.elim⟩).trans Pr_false
    rw [h0, mul_zero]
    exact add_nonneg hκ0 (mul_nonneg (by positivity) (Ex_nonneg fun _ => Real.sqrt_nonneg _))
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, hin, htop⟩ := hsc
  have hF : Pr (fun c : L.ECoins x => L.FailsOne δ x (run H P L.V x ω) (found ω) c) =
      Pr (fun c : (Fin (L.capOut x) → Fin (L.V.k x) → Coin) × (Fin (L.kOut x) → L.toCompiled.E1 x) =>
        (accList (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
          ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x ∨
          ∃ B ∈ (range Pt.blocks.length).powerset.filter (fun B => missLev Pt Lv B ≤ δ),
            AllBad (fun r => (run H P L.V x ω r).out = .acc) (L.kOut x)
              (fun r e => L.toCompiled.BadOuter x (run H P L.V x ω) (found ω) B (forkOn (L.secondStage x) r) e)
              ((List.finRange (L.capOut x)).map c.1) c.2) := by
    refine Eq.trans ?_ (Claims.Sampling.Pr_equiv (L.toCompiled.ecoinsEquiv x) _)
    refine Pr_congr fun c => ⟨fun ⟨s', Pt', Lv', hs', hPt', hLv', _, _, h⟩ => ?_,
      fun h => ⟨s, Pt, Lv, hs, hPt, hLv, hin, htop, h⟩⟩
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    rw [hPt, Option.some.injEq] at hPt'
    subst hPt'
    rw [hLv, Option.some.injEq] at hLv'
    subst hLv'
    exact h
  rw [hF]
  have hlen : Lv.levels.length = 1 := by
    have h := hin.2.2.2
    rw [htop] at h
    exact h
  have hun : L.nPU x ≤ L.nOut x := by
    have h := hS.unions x s Pt Lv hs hPt hLv hin
    rw [htop] at h
    exact h
  refine (L.toCompiled.outer_bad_le_pa hδ x (run H P L.V x ω) (found ω) Pt (Compiled.nPU_eq hs hPt) hun
    (missLev Pt Lv) (fun r => L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r))
    (fun r => add_nonneg (L.toCompiled.κs_nonneg x) (Real.sqrt_nonneg _))
    ((hS.samples x s Pt Lv hs hPt hLv hin).1 hlen) (hS.slots x) (hsess ω)).trans (le_of_eq ?_)
  rw [Ex_add, Ex_const]
  unfold Compiled.κ
  ring

open Classical in
/-- **A proof block's best interior** on the values `val`: one at which proof block `v` is correct, where one is. -/
noncomputable def bestIota (s : AuditStatement) (Pt : Partition) (val : ℕ → Option Bool) (v : ℕ) : ℕ → Bool :=
  if h : ∃ ι : ℕ → Bool, ∃ u, Pt.blocks[v]? = some u ∧ s.prog.Correct (s.prog.refIn u val ι) u then h.choose
  else fun _ => false

/-- **A proof block incorrect at its best interior is incorrect at every interior.** -/
theorem incorrect_bestIota {s : AuditStatement} {Pt : Partition} {val : ℕ → Option Bool} {v : ℕ}
    (hv : v ∈ incorrectLev s Pt none val (bestIota s Pt val)) (ι : ℕ → Bool) :
    ∃ b, Pt.blocks[v]? = some b ∧ ¬ s.prog.Correct (s.prog.refIn b val ι) b := by
  classical
  unfold incorrectLev at hv
  obtain ⟨hlt, hnot⟩ := mem_filter.1 hv
  have hlt' : v < Pt.blocks.length := mem_range.1 hlt
  have hget : Pt.blocks[v]? = some (Pt.blocks[v]'hlt') := List.getElem?_eq_getElem hlt'
  refine ⟨Pt.blocks[v]'hlt', hget, fun hc => hnot ?_⟩
  have hex : ∃ ι : ℕ → Bool, ∃ u, Pt.blocks[v]? = some u ∧ s.prog.Correct (s.prog.refIn u val ι) u :=
    ⟨ι, _, hget, hc⟩
  obtain ⟨u, hu, hcu⟩ := hex.choose_spec
  refine ⟨u, v, u, hu, rfl, hu, ?_⟩
  show s.prog.Correct (s.prog.refIn u val (bestIota s Pt val v)) u
  rw [bestIota, dif_pos hex]
  exact hcu

/-- **The split at one level**: at an accepted run whose extraction fails, the failure event holds or the extractor's
own calls collide. Otherwise the kept openings agree and assemble a witness; take `B` its proof blocks incorrect at
their best interiors (`bestIota`), which the levels miss with probability at most `δ`, and a kept outer run `r`. Each
`v ∈ B` is wrong at the interior values the kept openings under `r`'s interior roots give (`incorrect_bestIota`); a
kept inner run at which `v` is good would make it correct (`Sessions.opens`, `v` its own top block), so `r` is bad
for `B`. -/
theorem splitOne {H : Msg → Msg} {L : CompiledLev Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H) (δ : ℝ)
    (x : Msg) (hone : L.OneLevel x) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (L.V.k x) → Coin)
    (c : L.ECoins x) (hacc : (run H P L.V x ω r₀).out = .acc)
    (hfail : ¬ (RLev H L.commit L.parse L.partition L.levels L.top δ).ext x (L.E.eval H P x ω r₀ c).1) :
    L.FailsOne δ x (run H P L.V x ω) (L.toCompiled.Found H x (run H P L.V x ω)) c ∨ OwnColl H L.E P x ω r₀ c := by
  classical
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, hin⟩ := hS.refuses x Ω P ω r₀ hacc
  have htop := hone s hs
  by_cases hcoll : OwnColl H L.E P x ω r₀ c
  · exact Or.inr hcoll
  refine Or.inl ⟨s, Pt, Lv, hs, hPt, hLv, hin, htop, ?_⟩
  set run' := run H P L.V x ω with hrun'
  set fd := L.toCompiled.found H x s run' c (L.toCompiled.outerList H x run' c) with hfd
  have hpass : ∀ q ∈ fd, L.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((L.commit.opens q.1 q.2).eval H).2, t ∈ Claims.Sampling.ownLog H run' (L.E.ext x r₀ c) := by
    intro q hq
    rw [hfd, Compiled.found_eq] at hq
    obtain ⟨h1, h2⟩ := L.toCompiled.passes_found H x run' s c _ q hq
    exact ⟨h1, fun t ht => L.toCompiled.ownLog_E H P x ω r₀ c hs hacc t (h2 t ht)⟩
  have hagree : ∀ q ∈ fd, ∀ q' ∈ fd, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => agree_of_passes hbind hpass hcoll hq hq' h1 h2
  rw [L.toCompiled.E_eval H P x ω r₀ c hs hacc] at hfail
  set w := assemble s fd with hw
  set B := incorrectLev s Pt none (w.val H L.commit s) (bestIota s Pt (w.val H L.commit s)) with hB
  have hBδ : B ∈ (range Pt.blocks.length).powerset.filter (fun B => missLev Pt Lv B ≤ δ) := by
    refine mem_filter.2 ⟨mem_powerset.2 (filter_subset _ _), ?_⟩
    by_contra hlt
    refine hfail ⟨s, Pt, Lv, hs, hPt, hLv, bestIota s Pt (w.val H L.commit s), ?_⟩
    rw [htop]
    exact lt_of_not_ge hlt
  by_cases hshort : (accList (fun r => (run' r).out = .acc) (L.kOut x)
    ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x
  · exact Or.inl hshort
  refine Or.inr ⟨_, hBδ, accList_length_eq hshort, fun j r hj => ?_⟩
  have hjZ := L.toCompiled.mem_outerList H x run' c hj
  by_cases hshort2 : (accList (fun d => (run' (forkOn (L.secondStage x) r d)).out = .acc) (L.kIn x)
    ((List.finRange (L.capIn x)).map (c.2.1 j))).length < L.kIn x
  · exact Or.inl hshort2
  refine Or.inr ⟨accList_length_eq hshort2, fun l d hl v hv hgood => ?_⟩
  obtain ⟨b, hb, hnc⟩ := incorrect_bestIota hv (ιOf fd (L.iroots x (run' r).seen v))
  obtain ⟨hvc, xS, vS, wS, hsess, hRw, hkept⟩ :=
    L.toCompiled.good_found H P ω x s c (hS.shared x Ω P ω) hjZ hl hgood
  have hacc' : (run' (forkOn (L.secondStage x) r d)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  obtain ⟨b', ρ, rb, hb', hρ, hrb, hcor⟩ :=
    hS.opens x s Pt Lv Ω P ω (forkOn (L.secondStage x) r d) v xS vS wS hs hPt hLv hin hacc' hvc hsess hRw
  rw [htop] at hρ hrb
  have hρv : v = ρ := Option.some.inj hρ
  subst hρv
  have hcorr := hcor (w.val H L.commit s) (ιOf fd (L.iroots x (run' r).seen v)) fun q hq hok => by
    have hqf : q ∈ fd := hkept q hq hok
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · rw [hr] at hok
      exact val_assemble hagree hqf hr hok
    · rw [(hS.interior x Ω P ω r d v).1] at hi
      rw [← hi]
      exact ιOf_eq hagree hqf
  have hrb' : Pt.blocks[v]? = some rb := hrb
  rw [hb, Option.some.injEq] at hb' hrb'
  subst hb'
  subst hrb'
  exact hnc hcorr

theorem Pr_le_one' {α : Type} [Fintype α] (p : α → Prop) : Pr p ≤ 1 := by
  rcases isEmpty_or_nonempty α with h | h
  · unfold Pr Claims.prUniform
    simp [Fintype.card_eq_zero]
  · exact Pr_le_one p

set_option synthInstance.maxSize 1024 in
open Classical in
/-- **`CompiledLev.KS` at one level**: at every `δ ≥ 0`, `KSLinOn` on the statements in scope with no top partition, at
`Compiled`'s extractor and errors (`Sampling.KSLin_of_ownColl` at `FailsOne`, `pr_failsOne_le` and `splitOne`, with the
bound `1` off them). -/
theorem ks_one {H : Msg → Msg} {L : CompiledLev Coin A} (hA : A.KS H) (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (δ : ℝ) (hδ : 0 ≤ δ) :
    KSLinOn H L.V (fun x => L.Scope x ∧ L.OneLevel x) (RLev H L.commit L.parse L.partition L.levels L.top δ) L.E
      (L.κ δ) L.γ (fun _ => 1) (Bext H L.V L.E) := by
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  have h := Sampling.KSLin_of_ownColl H L.V (RLev H L.commit L.parse L.partition L.levels L.top δ) L.E
    (fun x => if L.OneLevel x then L.κ δ x else 1) L.γ
    (fun x Ω P => if L.OneLevel x then
        fun s => (run H P L.V x s.1 s.2.1).out = .acc ∧
          L.FailsOne δ x (run H P L.V x s.1) (L.toCompiled.Found H x (run H P L.V x s.1)) s.2.2
      else fun s => (run H P L.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLev H L.commit L.parse L.partition L.levels L.top δ).ext x (L.E.eval H P x s.1 s.2.1 s.2.2).1)
    (fun x Ω P s hacc hfail => by
      by_cases h1 : L.OneLevel x
      · rw [if_pos h1]
        exact (splitOne hbind hS δ x h1 P s.1 s.2.1 s.2.2 hacc hfail).imp_left fun h => ⟨hacc, h⟩
      · rw [if_neg h1]
        exact Or.inl ⟨hacc, hfail⟩)
    (fun x Ω _ _ P => by
      by_cases h1 : L.OneLevel x
      · rw [if_pos h1, if_pos h1]
        obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
        exact L.pr_failsOne_le H hδ hS x Ω P (fun ω => L.toCompiled.Found H x (run H P L.V x ω)) β hβ0 hβ hκ
      · rw [if_neg h1, if_neg h1]
        exact (Pr_le_one' _).trans (le_add_of_nonneg_right (Real.sqrt_nonneg _)))
  intro x hx Ω _ _ P
  have hx' := h x Ω P
  dsimp only at hx'
  rwa [if_pos hx.2] at hx'

end CompiledLev

end Claims.SamplingGeneric
