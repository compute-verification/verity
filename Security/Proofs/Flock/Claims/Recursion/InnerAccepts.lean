import Proofs.Flock.Claims.Recursion.InnerAbort

/-!
# `InnerAccepts` at the inner shapes

`InnerAccepts` (`Recursion/Inner.lean`) is the code fact that wherever `E_O` gives witnesses on an accepted run, the
inner view read off them is accepted. This file reduces it to what V*'s circuits compute, at each inner shape the
recursion's claims cover, read off the compiled verifier's own inner shape (`Form`): dense (`Form.Dense`: public inner
units, no sparse phase; `PublicAudit`'s), holo (`Form.Holo`: private inner circuits; `RecursiveAuditC3`'s) and sparse
(`Form.Sparse`: public inner units with candidate 3's RecSparse phase). The three cover every form
(`Form.shape_cases`), so no statement falls outside them.

* `E_O` gives only witnesses in the outer Flock's relation at the run's V* statements (`eoGives_outerHolds`): each
  session's `E^rep` returns only a candidate `Flock.check` accepts at its statement, served or not.
* `VStarAcceptsExact shape`: V*'s circuits compute the inner verifier at that shape: on an accepted run of a statement
  of the shape, witnesses in the outer relation at its V* statements give an accepted inner view. With `Refuses`, the
  three shapes' give `InnerAccepts` (`innerAccepts_of_vstar`).
* `VStarAccepts shape`: the same, or two of the witnesses' recorded calls collide. V*'s parts, and the units within a
  part, each open the inner session's commit strings themselves, so two openings of one commit string, a SHA-512
  collision, can pass every part while no one inner transcript is accepted; at SHA-512 this is the form V* can meet.
  With `Refuses`, the three shapes' give `InnerAcceptsColl` (`innerAcceptsColl_of_vstar`), and where `InnerAcceptsColl`
  holds, an accepted run whose `E_O` witnesses give a rejected inner view is a collision `Be` finds at every fork
  (`Pr_innerRejects_le_fork`), so it costs at most `Pr[Be]` (`Pr_innerRejects_le`). `InnerAcceptsKS.lean` carries
  `InnerKS` over it at the same `γ_eA`.
-/

namespace Claims.Recursion

open Claims

/-! ## The inner shapes -/

/-- **Dense**: public inner units and no sparse phase (`PublicAudit`'s inner protocol). -/
def Form.Dense (f : Form) : Prop := f.holo = false ∧ f.sparse = none

/-- **Holo**: private inner circuits (`RecursiveAuditC3`'s inner protocol). -/
def Form.Holo (f : Form) : Prop := f.holo = true

/-- **Sparse**: public inner units with candidate 3's RecSparse phase. -/
def Form.Sparse (f : Form) : Prop := f.holo = false ∧ f.sparse.isSome = true

/-- **The three shapes cover every form.** -/
theorem Form.shape_cases (f : Form) : f.Dense ∨ f.Holo ∨ f.Sparse := by
  rcases f with ⟨m, kLog, regions, claims, holo, sparse, stacked, u⟩
  cases holo <;> cases sparse <;> simp [Form.Dense, Form.Holo, Form.Sparse]

/-! ## `E_O`'s witnesses are in the outer relation -/

/-- `collect` returns, in order, what each of its programs returned. -/
theorem collect_some {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ (ps : List (RP R (Option α))) (as : List α), ((collect ps).eval H run).1 = some as →
      List.Forall₂ (fun p a => (p.eval H run).1 = some a) ps as
  | [], as, h => by
    simp only [collect, RP.eval, Option.some.injEq] at h
    subst h
    exact .nil
  | p :: ps, as, h => by
    simp only [collect] at h
    rw [(RP.eval_bind H run p _).1] at h
    rcases hp : (p.eval H run).1 with _ | a
    · rw [hp] at h
      simp [RP.eval] at h
    · rw [hp] at h
      simp only at h
      rw [(RP.eval_bind H run _ _).1] at h
      simp only [RP.eval] at h
      obtain ⟨l, hl, rfl⟩ := Option.map_eq_some_iff.1 h
      exact .cons hp (collect_some H run ps l hl)

theorem stepWith_acc_pos {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω vn n hist as t).out = .acc → 0 < n
  | 0, _, _, _, h => by simp [stepWith] at h
  | _ + 1, _, _, _, _ => Nat.succ_pos _

/-- **A verifier that takes no rounds accepts no run.** -/
theorem run_acc_pos {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (r : Fin (V.k x) → Coin) (h : (run H P V x ω r).out = .acc) : 0 < V.k x :=
  stepWith_acc_pos H P x ω _ _ _ _ _ h

section Facts

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The run's V\* statements**: `vstar` at the run's first `k_I` messages, the statements `E_O` extracts at. -/
def runVstar (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (D.V.k x) → Coin) : List Msg :=
  D.vstar x ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x))

/-- **The outer witnesses are in the outer Flock's relation at the V\* statements `xs`**: `n` of them, each accepted
by `Flock.check` at its session's statement. -/
def OuterHolds (DO : Flock.Deploy Coin Col T) (xs : List Msg) (n : ℕ) (ws : List Flock.Wit) : Prop :=
  ws.length = n ∧ ∀ (s : ℕ) (h : s < ws.length), Flock.check DO.parse (xs.getD s []) ws[s] = true

/-- **`E_O` gives only witnesses in the outer relation at the run's V\* statements**: one per session, each the
candidate its session's `E^rep` returned, which `Flock.check` accepted (`firstWait_some`, through the serving,
`RP.serve_eval`). -/
theorem eoGives_outerHolds {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {x : Msg} {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin (D.V.k x) → Coin}
    {c : (EO D I DO decO).C x} {ws : List Flock.Wit} (h : EOGives H D I DO decO P x ω r c ws) :
    OuterHolds DO (runVstar H D I P x ω r) (D.nS x) ws := by
  obtain ⟨hacc, h⟩ := h
  simp only [Extractor.eval, EO, RP.eval, Trace.seen, hacc, ↓reduceIte] at h
  have hf := collect_some H _ _ ws h
  have hlen := hf.length_eq
  rw [List.length_map, List.length_finRange] at hlen
  refine ⟨hlen.symm, fun s hs => ?_⟩
  have h1 := (List.forall₂_iff_get.1 hf).2 s (by rw [List.length_map, List.length_finRange, hlen]; exact hs) hs
  simp only [List.get_eq_getElem, List.getElem_map, List.getElem_finRange] at h1
  rw [serve_eq, (RP.serve_eval H _ _ _).1] at h1
  exact firstWait_some H _ _ _ _ _ _ _ h1

/-! ## V*'s circuits compute the inner verifier -/

/-- **V\*'s circuits compute the inner verifier at the inner shape `shape`, exactly** (a code fact about V*'s
circuits, the compiled verifier's V* statements and the readers its inner view reads): on an accepted run of a statement
whose form has the shape, outer witnesses in the outer Flock's relation at the run's V* statements give an accepted
inner view. -/
def VStarAcceptsExact (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (s : CompiledStatement), D.parse x = some s → shape s.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ ws, OuterHolds DO (runVstar H D I P x ω r) (D.nS x) ws → (D.innerView H x r ws).out = .acc

/-- **V\*'s circuits compute the inner verifier at the inner shape `shape`, off a collision** (a code fact about V*'s
circuits, the compiled verifier's V* statements and the readers its inner view reads): on an accepted run of a statement
whose form has the shape, outer witnesses in the outer Flock's relation at the run's V* statements, whose recorded
calls are true evaluations, give an accepted inner view, or two of their recorded calls are a collision, which
`findColl` finds. -/
def VStarAccepts (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (s : CompiledStatement), D.parse x = some s → shape s.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ ws, OuterHolds DO (runVstar H D I P x ω r) (D.nS x) ws → (∀ e ∈ D.calls x ws, H e.1 = e.2) →
        (D.innerView H x r ws).out = .acc ∨ ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v

omit [Fintype Coin] [Nonempty Coin] in
theorem VStarAcceptsExact.toColl {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {shape : Form → Prop} (h : VStarAcceptsExact H D I DO shape) : VStarAccepts H D I DO shape :=
  fun x s hs hf Ω P ω r hacc ws ho _ => Or.inl (h x s hs hf Ω P ω r hacc ws ho)

/-- **`InnerAccepts` off a collision** (a code fact): wherever `E_O` gives witnesses on an accepted run, the inner view
read off them is accepted, or two of the witnesses' recorded calls are a collision. -/
def InnerAcceptsColl (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (D.innerView H x r ws).out = .acc ∨ ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v

theorem InnerAccepts.toColl {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (h : InnerAccepts H D I DO) : InnerAcceptsColl H D I DO :=
  fun decO x Ω P ω r c ws hg => Or.inl (h decO x Ω P ω r c ws hg)

/-- **`InnerAcceptsColl` at the statement `x`**: the bound at one statement reads `InnerAcceptsColl` only there, so a
bound scoped to statements of one shape needs V*'s circuits only at that shape. -/
def InnerAcceptsCollAt (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (x : Msg) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (D.innerView H x r ws).out = .acc ∨ ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v

theorem InnerAcceptsColl.toAt {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (h : InnerAcceptsColl H D I DO) (x : Msg) : InnerAcceptsCollAt H D I DO x :=
  fun decO => h decO x

/-- **`InnerAccepts` from V\*'s circuits at the three inner shapes**: an accepted run's statement parses (`Refuses`),
its form is dense, holo or sparse (`Form.shape_cases`), and `E_O`'s witnesses are in the outer relation at the run's V*
statements (`eoGives_outerHolds`). -/
theorem innerAccepts_of_vstar {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hRef : Refuses H D) (hD : VStarAcceptsExact H D I DO Form.Dense) (hH : VStarAcceptsExact H D I DO Form.Holo)
    (hS : VStarAcceptsExact H D I DO Form.Sparse) : InnerAccepts H D I DO := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨s, hs, -⟩ := hRef x Ω P ω r hg.1
  have ho := eoGives_outerHolds hg
  rcases Form.shape_cases s.form with hf | hf | hf
  · exact hD x s hs hf Ω P ω r hg.1 ws ho
  · exact hH x s hs hf Ω P ω r hg.1 ws ho
  · exact hS x s hs hf Ω P ω r hg.1 ws ho

/-- **`InnerAcceptsColl` from V\*'s circuits at the three inner shapes, off a collision**: as `innerAccepts_of_vstar`,
the witnesses' recorded calls true evaluations by `CallsTrue`. -/
theorem innerAcceptsColl_of_vstar {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hRef : Refuses H D) (hT : CallsTrue H D I DO) (hD : VStarAccepts H D I DO Form.Dense)
    (hH : VStarAccepts H D I DO Form.Holo) (hS : VStarAccepts H D I DO Form.Sparse) : InnerAcceptsColl H D I DO := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨s, hs, -⟩ := hRef x Ω P ω r hg.1
  have ho := eoGives_outerHolds hg
  have ht := hT decO x Ω P ω r c ws hg
  rcases Form.shape_cases s.form with hf | hf | hf
  · exact hD x s hs hf Ω P ω r hg.1 ws ho ht
  · exact hH x s hs hf Ω P ω r hg.1 ws ho ht
  · exact hS x s hs hf Ω P ω r hg.1 ws ho ht

/-- **`InnerAcceptsColl` at a dense statement from V\*'s circuits at the dense shape**: an accepted run's statement
parses (`Refuses`), and at `x` every parse is dense. -/
theorem innerAcceptsCollAt_of_dense {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hD : VStarAccepts H D I DO Form.Dense) {x : Msg} (hx : ∀ s, D.parse x = some s → s.form.Dense) :
    InnerAcceptsCollAt H D I DO x := by
  intro decO Ω P ω r c ws hg
  obtain ⟨s, hs, -⟩ := hRef x Ω P ω r hg.1
  exact hD x s hs (hx s hs) Ω P ω r hg.1 ws (eoGives_outerHolds hg) (hT decO x Ω P ω r c ws hg)

/-! ## The collision's cost -/

/-- **An accepted run whose inner view rejects**, at the tape, the compiled verifier's coins and `E_O`'s: `E_O` gives
witnesses there, and the inner view read off them is not accepted. -/
def InnerRejects (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg)
    (t : Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : Prop :=
  ∃ ws, EOGives H D I DO decO P x t.1 t.2.1 t.2.2 ws ∧ (D.innerView H x t.2.1 ws).out ≠ .acc

theorem unifMap_judged {Ω R C : Type} [Fintype Ω] [Fintype R] [Fintype C] [Nonempty R] [Nonempty C] :
    UnifMap (fun t : Ω × R × R × C × C => (t.1, t.2.1, t.2.2.2.1)) := by
  let e : Ω × R × R × C × C ≃ (Ω × R × C) × (R × C) :=
    { toFun := fun t => ((t.1, t.2.1, t.2.2.2.1), (t.2.2.1, t.2.2.2.2))
      invFun := fun s => (s.1.1, s.1.2.1, s.2.1, s.1.2.2, s.2.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hfst : UnifMap (fun p : (Ω × R × C) × (R × C) => p.1) := fun G => by
    rw [Ex_prod]
    simp only [Ex_const]
  exact (UnifMap.equiv e).comp hfst

/-- **A rejected inner view is a collision at every fork**: where `InnerAcceptsColl` holds, it is a collision among the
judged run's recorded calls, which `Be`'s fork at any round `j` records as well (`findColl_isSome_of_sub`; every call
`Be` reads is a true evaluation, by `CallsTrue` and `CallsNil`). -/
theorem Pr_innerRejects_le_fork_at {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {x : Msg} (hA : InnerAcceptsCollAt H D I DO x) (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] (P : Prover Ω) (j : ℕ) :
    Pr (InnerRejects H D I DO decO P x) ≤ Pr (forkColl H D I DO decO P x j) := by
  rw [← (unifMap_judged (Ω := Ω) (R := Fin (D.V.k x) → Coin) (C := (EO D I DO decO).C x)).Pr]
  refine Pr_mono fun t ht => ?_
  obtain ⟨ws, hg, hrej⟩ := ht
  rcases hA decO Ω P t.1 t.2.1 t.2.2.2.1 ws hg with hacc | ⟨u, v, huv, -⟩
  · exact absurd hacc hrej
  · have hws : ((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD [] = ws := by
      rw [hg.2]; rfl
    have htrue : ∀ p ∈ D.calls x (((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD []) ++
        D.calls x (((EO D I DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2).1.getD []),
        H p.1 = p.2 :=
      fun p hp => (List.mem_append.1 hp).elim (calls_true hT hN0 t.1 (t.2.1, t.2.2.2.1) p)
        (calls_true hT hN0 t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2) p)
    have hsub : ∀ p ∈ D.calls x ws, p ∈ D.calls x (((EO D I DO decO).eval H P x t.1 t.2.1 t.2.2.2.1).1.getD []) ++
        D.calls x (((EO D I DO decO).eval H P x t.1 (fork j t.2.1 t.2.2.1) t.2.2.2.2).1.getD []) :=
      fun p hp => List.mem_append_left _ (by rw [hws]; exact hp)
    obtain ⟨⟨u', v'⟩, huv'⟩ := Option.isSome_iff_exists.1
      (findColl_isSome_of_sub H htrue hsub (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩))
    exact ⟨u', v', huv', findColl_spec H _ htrue u' v' huv'⟩

theorem Pr_innerRejects_le_fork {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hA : InnerAcceptsColl H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (j : ℕ) :
    Pr (InnerRejects H D I DO decO P x) ≤ Pr (forkColl H D I DO decO P x j) :=
  Pr_innerRejects_le_fork_at (hA.toAt x) hT hN0 decO P j

/-- **A rejected inner view costs at most `Pr[Be]`**: it is a collision at each of `Be`'s forks
(`Pr_innerRejects_le_fork`), and `Be` averages its forks. -/
theorem Pr_innerRejects_le {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hA : InnerAcceptsColl H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (InnerRejects H D I DO decO P x) ≤ PrBe H D I DO decO P x := by
  by_cases hk : 0 < D.V.k x
  · have hj : ∀ j : Fin (D.V.k x), Pr (InnerRejects H D I DO decO P x) ≤ Pr (forkColl H D I DO decO P x j) :=
      fun j => Pr_innerRejects_le_fork hA hT hN0 decO x P j
    rw [PrBe_eq_avg H D I DO decO P x hk, le_div_iff₀ (by exact_mod_cast hk)]
    calc Pr (InnerRejects H D I DO decO P x) * D.V.k x
        = ∑ _j : Fin (D.V.k x), Pr (InnerRejects H D I DO decO P x) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_comm]
      _ ≤ ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := Finset.sum_le_sum fun j _ => hj j
  · have h0 : Pr (InnerRejects H D I DO decO P x) = 0 := by
      rw [← Pr_false (α := Ω × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x)]
      refine Pr_congr fun t => ⟨fun ⟨_, hg, _⟩ => hk (run_acc_pos H P D.V x t.1 t.2.1 hg.1), False.elim⟩
    rw [h0]
    exact PrBe_nonneg H D I DO decO P x

end Facts

end Claims.Recursion
