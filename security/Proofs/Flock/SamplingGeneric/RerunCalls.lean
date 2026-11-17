module

public import Proofs.Flock.Claims.Recursion.Serve
public import Proofs.Flock.SamplingGeneric.Efficient
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Recursion.Cost
import Proofs.Flock.Claims.Recursion.Wald
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Cost

/-! Which runs a rerun program's SHA-512 calls are, so that a bound may read each rerun's own calls and not a budget on every run. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling
open Claims.Recursion (serve viewing)

theorem calls_eq_own {R α : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R α) :
    (p.eval H run).2.1 = (ownLog H run p).length +
      ((p.rerunsAt H run).map fun r => (run r).pLog.length + (run r).vLog.length).sum := by
  induction p with
  | ret a => simp [RP.eval, ownLog, RP.rerunsAt]
  | hash q k ih =>
    simp only [RP.eval, ownLog, RP.rerunsAt, List.length_cons]
    have := ih (H q)
    omega
  | rerun r k ih =>
    simp only [RP.eval, ownLog, RP.rerunsAt, List.map_cons, List.sum_cons]
    have := ih (run r).seen
    omega

theorem rerunsAt_congr {R α : Type} (H : Msg → Msg) {run₁ run₂ : R → Transcript}
    (h : ∀ r, (run₁ r).seen = (run₂ r).seen) (p : RP R α) : p.rerunsAt H run₁ = p.rerunsAt H run₂ := by
  induction p with
  | ret a => rfl
  | hash q k ih => exact ih (H q)
  | rerun r k ih =>
    simp only [RP.rerunsAt]
    rw [h r, ih]

theorem flatMapRP_rerunsAt_sum {R β γ : Type} (H : Msg → Msg) (run : R → Transcript) (w : R → ℕ)
    (f : β → RP R (List γ)) :
    ∀ l : List β, (((flatMapRP f l).rerunsAt H run).map w).sum =
      (l.map fun b => (((f b).rerunsAt H run).map w).sum).sum
  | [] => rfl
  | b :: bs => by
    simp only [flatMapRP, RP.rerunsAt_bind, RP.rerunsAt, List.append_nil, List.map_append, List.sum_append,
      List.map_cons, List.sum_cons]
    rw [flatMapRP_rerunsAt_sum H run w f bs]

theorem serve_place_rerunsAt {R₁ R₂ α : Type} (H : Msg → Msg) (f : Seen → Seen) (g : R₁ → R₂)
    (run : R₂ → Transcript) (p : RP R₁ α) :
    (RP.serve (fun r => viewing f (g r)) p).rerunsAt H run =
      (p.rerunsAt H (servedRun H (fun r => viewing f (g r)) run)).map g := by
  induction p with
  | ret a => rfl
  | hash m k ih => exact ih (H m)
  | rerun r k ih =>
    simp only [RP.serve, RP.rerunsAt, List.map_cons]
    rw [RP.rerunsAt_bind]
    exact congrArg (List.cons (g r)) (ih _)

theorem flatMapRP_checkRevealed_rerunsAt {R β : Type} (H : Msg → Msg) (run : R → Transcript)
    (Cm : CommitmentScheme Bool) (f : β → List (List Msg × Opening Bool)) (l : List β) :
    (flatMapRP (fun b => checkRevealed (R := R) Cm (f b)) l).rerunsAt H run = [] :=
  List.eq_nil_of_length_eq_zero
    ((RP.eval_runs_eq H run _).symm.trans (flatMapRP_checkRevealed_runs H run Cm f l))

namespace Arg

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (H : Msg → Msg)

omit [Inhabited Coin] in
theorem rerunsCostExp_le (x : Msg) (run : (Fin (A.V.k x) → Coin) → Transcript)
    {w : (Fin (A.V.k x) → Coin) → ℝ} (hw : ∀ d, 0 ≤ w d) :
    Ex (fun p : (Fin (A.V.k x) → Coin) × A.EC (A.shape x) =>
        (((A.ext x p.1 p.2).rerunsAt H run).map w).sum) ≤
      (1 + A.waits (A.shape x) * A.k (A.shape x)) * Ex w :=
  le_of_eq_of_le (Ex_equiv ((Equiv.refl (Fin (A.V.k x) → Coin)).prodCongr (A.ecEquiv x)) fun s =>
    (((erepProg (A.waits (A.shape x)) (A.k (A.shape x)) (A.cap (A.shape x)) A.opens
      (fun os => (A.assemble x os).1) (A.check x) s.1 s.2).rerunsAt H run).map w).sum)
    (erepProg_cost_exp H run hw (A.waits (A.shape x)) (A.k (A.shape x)) (A.cap (A.shape x)) A.opens
      (fun os => (A.assemble x os).1) (A.check x))

end Arg

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem place_congr (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg} (h : A.V.k xS = (L.rounds x u).card)
    {r r' : Fin (L.V.k x) → Coin} (hr : ∀ t ∉ L.rounds x u, r t = r' t) (d : Fin (A.V.k xS) → Coin) :
    L.place x u xS r d = L.place x u xS r' d := by
  funext t
  by_cases ht : t ∈ L.rounds x u
  · simp only [Compiled.place, dite_eq_left (And.intro h ht)]
  · rw [L.place_of_not_mem r d ht, L.place_of_not_mem r' d ht, hr t ht]

omit [Fintype Coin] [Nonempty Coin] in
theorem place_emb (L : Compiled Coin A) {x : Msg} {u : ℕ} {xS : Msg} (h : A.V.k xS = (L.rounds x u).card)
    (r σ : Fin (L.V.k x) → Coin) :
    L.place x u xS r (fun i => σ (L.emb x u xS h i)) = forkOn (L.rounds x u) r σ := by
  have h1 := L.place_coinsOf (u := u) (xS := xS) (forkOn (L.rounds x u) r σ)
  rw [L.coinsOf_forkOn h] at h1
  exact (L.place_congr h (fun t ht => by simp [forkOn, ht]) _).trans h1

theorem sessExt_rerunsAt (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} (P : Prover Ω) (ω : Ω) {u : ℕ}
    {xS : Msg} (P' : Prover Unit) (r0 : Fin (L.V.k x) → Coin) (hshape : A.shape xS = L.sh x u)
    (hcard : A.V.k xS = (L.rounds x u).card)
    (hview : ∀ r' : Fin (L.V.k x) → Coin, (∀ t ∉ L.rounds x u, (∀ t' ∈ L.rounds x u, t < t') → r' t = r0 t) →
      L.session x u (run H P L.V x ω r').seen = some (xS, (run H P' A.V xS () (L.coinsOf x u xS r')).seen))
    {r : Fin (L.V.k x) → Coin} (hr : ∀ t ∉ L.rounds x u, r t = r0 t) (s : AuditStatement) (us : List ℕ)
    (c : A.EC (L.sh x u)) :
    (L.sessExt x s us u r (run H P L.V x ω r).seen c).rerunsAt H (run H P L.V x ω) =
      ((A.ext xS (L.coinsOf x u xS r) (cast (congrArg A.EC hshape.symm) c)).rerunsAt H
        (run H P' A.V xS ())).map (L.place x u xS r) := by
  have hv := hview r fun t ht _ => hr t ht
  unfold sessExt sessWit
  simp only [hv]
  rw [dite_eq_left hshape, RP.rerunsAt_bind, Claims.Recursion.serve_eq, serve_place_rerunsAt,
    rerunsAt_congr H (L.served_seen H P ω P' r0 r hcard hview hr)]
  split
  · exact List.append_nil _
  · rw [flatMapRP_checkRevealed_rerunsAt, List.append_nil]

end Compiled

end Claims.SamplingGeneric
