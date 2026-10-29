import Proofs.Flock.Claims.Recursion.KSLin
import Proofs.Flock.Claims.Recursion.VStarBound

/-!
# Route 2C's escapes with the compiled verifier's own evaluations

A V\* part's check is accepted, or its recorded calls collide with each other or with the compiled verifier's own
SHA-512 evaluations at its V\* statement (`VStarB.PartChecksBV`; check 5's row commitments are among them). No witness
records those evaluations, so the facts route 2C reads of V*'s parts escape to a collision among a draw's evaluations
(`drawCalls`: its witnesses' recorded calls, the inner verifier's own at the draw's coins, `Deploy.callsAt`, and
`setupCalls` at the run's V\* statements):

* `CallsLogCollAt`: `CallsLogColl` with the inner run's calls among those `callsAt` lists at the draw's coins (or a
  collision among them); `CallsLogColl.toAt` gives it, and at the deployed record it holds by construction
  (`RecDeploy.callsLogCollAt_deployRec`).

* `VStarAcceptsV`, `InnerAcceptsCollAtV` and `ForksBindAtV` (with `forkCollV`): `VStarAccepts`, `InnerAcceptsCollAt`
  and `ForksBindAt` with that escape. Each old fact gives its twin (`VStarAccepts.toV`, `InnerAcceptsCollAt.toV`,
  `ForksBindAt.toV`): more calls hold a collision wherever fewer do.
* `Pr_innerRejects_le_PrBinnerV_at`: a rejecting judged draw costs at most `Pr[B_I]` over the draws' evaluations
  (`PrBinnerV`), which is at least `Pr[B_I]` (`PrBI_le_PrBinnerV`).
-/

namespace Claims.Recursion

open Claims

theorem coll_append {H : Msg → Msg} {l l' : List (Msg × Msg)} (hl : ∀ e ∈ l, H e.1 = e.2)
    (hl' : ∀ e ∈ l', H e.1 = e.2) (h : ∃ u v, findColl l = some (u, v) ∧ Collision H u v) :
    ∃ u v, findColl (l ++ l') = some (u, v) ∧ Collision H u v := by
  obtain ⟨u, v, huv, -⟩ := h
  exact VStar.coll_of_sub (fun e he => (List.mem_append.1 he).elim (hl e) (hl' e))
    (fun p hp => List.mem_append_left _ hp) (Option.isSome_iff_exists.2 ⟨_, huv⟩)

theorem setup_true (H : Msg → Msg) (us : List Msg) : ∀ e ∈ us.map (fun u => (u, H u)), H e.1 = e.2 := by
  intro e he
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 he
  rfl

section Facts

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The inner verifier's calls are among those a finder searches at the draw, off a collision** (`CallsLogCollBy` at
`Deploy.callsAt`): the recorded calls, the inner setup's and the inner verifier's own at the draw's coins. -/
abbrev CallsLogCollAt (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  CallsLogCollBy H D I DO fun x r ws => D.callsAt H x r ws

theorem mem_callsAt {H : Msg → Msg} {D : Deploy Coin} {x : Msg} {r : Fin (D.V.k x) → Coin} {ws : List Flock.Wit}
    {p : Msg × Msg} (hp : p ∈ D.calls x ws) : p ∈ D.callsAt H x r ws :=
  List.mem_append_left _ hp

theorem mem_callsAt_append {H : Msg → Msg} {D : Deploy Coin} {x : Msg} {r : Fin (D.V.k x) → Coin}
    {ws : List Flock.Wit} {L : List (Msg × Msg)} {p : Msg × Msg} (hp : p ∈ D.calls x ws ++ L) :
    p ∈ D.callsAt H x r ws ++ L :=
  (List.mem_append.1 hp).elim (fun h => List.mem_append_left _ (mem_callsAt h)) (List.mem_append_right _)

/-- The calls a finder searches at a draw are true where the recorded ones are: the rest are `H`'s answers. -/
theorem callsAt_true_of {H : Msg → Msg} {D : Deploy Coin} {x : Msg} {r : Fin (D.V.k x) → Coin}
    {ws : List Flock.Wit} (h : ∀ e ∈ D.calls x ws, H e.1 = e.2) : ∀ e ∈ D.callsAt H x r ws, H e.1 = e.2 :=
  fun e he => (List.mem_append.1 he).elim (h e) (setup_true H _ e)

/-- The calls a finder searches at a draw are true evaluations (`calls_true`, and `H`'s answers). -/
theorem callsAt_true {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    ∀ e ∈ D.callsAt H x d.1 ((eoOut H D I DO decO P x ω d).getD []), H e.1 = e.2 :=
  callsAt_true_of (calls_true hT hN0 ω d)

/-- **`CallsLogColl` gives `CallsLogCollAt`**: more calls hold the inner verifier's, and a collision among the recorded
ones, which are true, is one among them all. -/
theorem CallsLogColl.toAt {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (hL : CallsLogColl H D I DO) (hT : CallsTrue H D I DO) : CallsLogCollAt H D I DO := by
  intro decO x Ω P ω r c ws hg
  rcases hL decO x Ω P ω r c ws hg with h | ⟨u, v, huv, -⟩
  · exact Or.inl fun e he => mem_callsAt (h e he)
  · exact Or.inr (VStar.coll_of_sub (callsAt_true_of (hT decO x Ω P ω r c ws hg)) (fun p hp => mem_callsAt hp)
      (Option.isSome_iff_exists.2 ⟨_, huv⟩))

/-- **V\*'s circuits compute the inner verifier at the inner shape `shape`, off a collision with the verifier's
evaluations**: `VStarAccepts`, its escape a collision among the witnesses' recorded calls and the compiled verifier's
own evaluations at the run's V\* statements (`setupCalls`). -/
def VStarAcceptsV (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (s : CompiledStatement), D.parse x = some s → shape s.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ ws, OuterHolds DO (runVstar H D I P x ω r) (D.nS x) ws → (∀ e ∈ D.calls x ws, H e.1 = e.2) →
        (D.innerView H x r ws).out = .acc ∨
          ∃ u v, findColl (D.calls x ws ++ setupCalls H D (runVstar H D I P x ω r)) = some (u, v) ∧ Collision H u v

/-- **`InnerAcceptsCollAt` off a collision with the verifier's evaluations**: wherever `E_O` gives witnesses on an
accepted run of `x`, the inner view read off them is accepted, or the witnesses' recorded calls and the compiled
verifier's own evaluations at the run's V\* statements collide. -/
def InnerAcceptsCollAtV (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (x : Msg) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (D.innerView H x r ws).out = .acc ∨
      ∃ u v, findColl (D.calls x ws ++ setupCalls H D (runVstar H D I P x ω r)) = some (u, v) ∧ Collision H u v

/-- **A collision among two draws' evaluations at `Be`'s fork at round `j`**: `forkColl` over `drawCalls`, the calls
the draws' witnesses record and the compiled verifier's own at their V\* statements. -/
def forkCollV [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x) :
    Prop :=
  ∃ u v, findColl (drawCalls H D I DO decO P x t.1 (t.2.1, t.2.2.2.1) ++
      drawCalls H D I DO decO P x t.1 (fork j t.2.1 t.2.2.1, t.2.2.2.2)) = some (u, v) ∧ Collision H u v

/-- **`ForksBindAt` over the draws' evaluations**: at every draw of `Be`'s fork at round `j` where the inner views
differ at message `j`, the two draws' evaluations collide (`forkCollV`). -/
def ForksBindAtV [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (x : Msg) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (Ω : Type) (P : Prover Ω) (j : ℕ)
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x),
    OpensDiffer H D I DO decO P x j t → forkCollV H D I DO decO P x j t

theorem drawCalls_eq [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {ω : Ω}
    {r : Fin (D.V.k x) → Coin} {c : (EO D I DO decO).C x} {ws : List Flock.Wit}
    (hg : EOGives H D I DO decO P x ω r c ws) :
    drawCalls H D I DO decO P x ω (r, c) = D.callsAt H x r ws ++ setupCalls H D (runVstar H D I P x ω r) := by
  have hw : drawWits H D I DO decO P x ω (r, c) = ws := by
    show ((EO D I DO decO).eval H P x ω r c).1.getD [] = ws
    rw [hg.2]; rfl
  show D.callsAt H x r (drawWits H D I DO decO P x ω (r, c)) ++ setupCalls H D (runStmts H D I P x ω r) = _
  rw [hw]; rfl

theorem drawCalls_true [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg}
    (hT : CallsTrue H D I DO) (hN0 : CallsNil H D) (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    ∀ e ∈ drawCalls H D I DO decO P x ω d, H e.1 = e.2 := fun e he =>
  (List.mem_append.1 he).elim (callsAt_true (decO := decO) (P := P) hT hN0 ω d e) (setupCalls_true H D _ e)

theorem drawCalls_pair_true [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg}
    (hT : CallsTrue H D I DO) (hN0 : CallsNil H D) (ω : Ω) (d d' : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    ∀ e ∈ drawCalls H D I DO decO P x ω d ++ drawCalls H D I DO decO P x ω d', H e.1 = e.2 := fun e he =>
  (List.mem_append.1 he).elim (drawCalls_true hT hN0 ω d e) (drawCalls_true hT hN0 ω d' e)

/-- A collision `forkColl` finds among the draws' recorded calls is one `forkCollV` finds among their evaluations. -/
theorem forkCollV_of [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg}
    (hT : CallsTrue H D I DO) (hN0 : CallsNil H D) {j : ℕ}
    {t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x}
    (h : forkColl H D I DO decO P x j t) : forkCollV H D I DO decO P x j t := by
  obtain ⟨u, v, huv, -⟩ := h
  have hs := Option.isSome_iff_exists.2 ⟨_, huv⟩
  unfold forkCollV
  refine VStar.coll_of_sub (drawCalls_pair_true hT hN0 _ _ _) (fun p hp => ?_) hs
  rcases List.mem_append.1 hp with hp | hp
  · exact List.mem_append_left _ (List.mem_append_left _ (mem_callsAt hp))
  · exact List.mem_append_right _ (List.mem_append_left _ (mem_callsAt hp))

theorem ForksBindAt.toV [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {x : Msg} (h : ForksBindAt H D I DO x) (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) : ForksBindAtV H D I DO x :=
  fun decO Ω P j t hod => forkCollV_of hT hN0 (h decO Ω P j t hod)

theorem InnerAcceptsCollAt.toV {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {x : Msg} (h : InnerAcceptsCollAt H D I DO x) (hT : CallsTrue H D I DO) : InnerAcceptsCollAtV H D I DO x := by
  intro decO Ω P ω r c ws hg
  rcases h decO Ω P ω r c ws hg with h | h
  · exact Or.inl h
  · exact Or.inr (coll_append (hT decO x Ω P ω r c ws hg) (setupCalls_true H D _) h)

omit [Fintype Coin] [Nonempty Coin] in
theorem VStarAccepts.toV {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {shape : Form → Prop} (h : VStarAccepts H D I DO shape) : VStarAcceptsV H D I DO shape :=
  fun x s hs hf Ω P ω r hacc ws ho ht =>
    (h x s hs hf Ω P ω r hacc ws ho ht).imp id (coll_append ht (setupCalls_true H D _))

/-- **`InnerAcceptsCollAtV` at a dense statement from V\*'s circuits at the dense shape**: `innerAcceptsCollAt_of_dense`
with the verifier's evaluations in the escape. -/
theorem innerAcceptsCollAtV_of_dense {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hD : VStarAcceptsV H D I DO Form.Dense) {x : Msg} (hx : ∀ s, D.parse x = some s → s.form.Dense) :
    InnerAcceptsCollAtV H D I DO x := by
  intro decO Ω P ω r c ws hg
  obtain ⟨s, hs, -⟩ := hRef x Ω P ω r hg.1
  exact hD x s hs (hx s hs) Ω P ω r hg.1 ws (eoGives_outerHolds hg) (hT decO x Ω P ω r c ws hg)

/-- **A rejected inner view is a collision among a fork's evaluations**, at every round `j`: where
`InnerAcceptsCollAtV` holds, it is one among the judged draw's evaluations, which `B_I`'s fork at `j` searches too. -/
theorem Pr_innerRejects_le_forkV_at [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {x : Msg} (hA : InnerAcceptsCollAtV H D I DO x) (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] (P : Prover Ω) (j : ℕ) :
    Pr (InnerRejects H D I DO decO P x) ≤ Pr (forkCollV H D I DO decO P x j) := by
  rw [← (unifMap_judged (Ω := Ω) (R := Fin (D.V.k x) → Coin) (C := (EO D I DO decO).C x)).Pr]
  refine Pr_mono fun t ht => ?_
  obtain ⟨ws, hg, hrej⟩ := ht
  rcases hA decO Ω P t.1 t.2.1 t.2.2.2.1 ws hg with hacc | ⟨u, v, huv, -⟩
  · exact absurd hacc hrej
  · unfold forkCollV
    refine VStar.coll_of_sub (drawCalls_pair_true hT hN0 _ _ _) (fun p hp => List.mem_append_left _ ?_)
      (Option.isSome_iff_exists.2 ⟨_, huv⟩)
    rw [drawCalls_eq (ω := t.1) (r := t.2.1) (c := t.2.2.2.1) hg]
    exact mem_callsAt_append hp

theorem PrBinnerV_nonneg [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) : 0 ≤ PrBinnerV H D I DO decO P x := by
  unfold PrBinnerV; split
  · exact Pr_nonneg _
  · exact le_rfl

theorem PrFdevV_nonneg [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (M : ℕ) : 0 ≤ PrFdevV H D I DO decO P x M := by
  unfold PrFdevV; split
  · exact Pr_nonneg _
  · exact le_rfl

/-- **`B_I` over the draws' evaluations finds what `B_I` finds**: its fork searches more calls, all true. -/
theorem PrBI_le_PrBinnerV [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hT : CallsTrue H D I DO) (hN0 : CallsNil H D) (decO : List (ℕ × Col) → List T)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    PrBI H D I DO decO P x ≤ PrBinnerV H D I DO decO P x := by
  unfold PrBI PrBinnerV
  split_ifs with hk
  · exact Pr_mono fun s hs => forkCollV_of (j := s.1) (t := s.2) hT hN0 hs
  · exact le_rfl

/-- **A rejecting judged draw costs at most `Pr[B_I]` over the draws' evaluations**: it is a collision among every
fork's evaluations (`Pr_innerRejects_le_forkV_at`), so at most their average over the inner rounds, of which an
accepted run shows one (`RoundsMargin`). -/
theorem Pr_innerRejects_le_PrBinnerV_at [DecidableEq Col] {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {x : Msg} (hA : InnerAcceptsCollAtV H D I DO x) (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (hM : RoundsMargin H D I) (decO : List (ℕ × Col) → List T) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    Pr (InnerRejects H D I DO decO P x) ≤ PrBinnerV H D I DO decO P x := by
  by_cases hE : Pr (InnerRejects H D I DO decO P x) = 0
  · rw [hE]
    exact PrBinnerV_nonneg H D I DO decO P x
  obtain ⟨t, ws, hg, -⟩ : ∃ t, InnerRejects H D I DO decO P x t := by
    by_contra hn
    exact hE ((Pr_congr fun t => ⟨fun h => hn ⟨t, h⟩, False.elim⟩).trans Pr_false)
  have hkI : 0 < I.V.k (xI D x) := (hM x Ω P t.1 t.2.1 hg.1).1
  have hBI : PrBinnerV H D I DO decO P x =
      (∑ j : Fin (I.V.k (xI D x)), Pr (forkCollV H D I DO decO P x j)) / I.V.k (xI D x) := by
    simp only [PrBinnerV, hkI, ↓reduceDIte]
    unfold Finder.Pr
    exact Pr_guess (fun s : (BinnerV H D I DO decO P x hkI).S =>
      ∃ u v, (BinnerV H D I DO decO P x hkI).out s = some (u, v) ∧ Collision H u v)
  rw [hBI, le_div_iff₀ (by exact_mod_cast hkI)]
  calc Pr (InnerRejects H D I DO decO P x) * I.V.k (xI D x)
      = ∑ _j : Fin (I.V.k (xI D x)), Pr (InnerRejects H D I DO decO P x) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ j : Fin (I.V.k (xI D x)), Pr (forkCollV H D I DO decO P x j) :=
        Finset.sum_le_sum fun j _ => Pr_innerRejects_le_forkV_at hA hT hN0 decO P j

end Facts

end Claims.Recursion
