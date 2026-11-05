module

public import Definitions.Core.Guarantees.Rec
public import Definitions.Claims.Recursion
public import Proofs.Core.Leak
public import Proofs.Flock.Claims.ZK.Stop
public import Proofs.Flock.Claims.Recursion.SeqPrefix

/-!
# `RecZK`: zero knowledge carries through a recursion

`Verity.SecurityProofs.RecZK : Verity.Guarantees.RecZK`, by the two hybrids of the recursion's zero knowledge:

* the sessions' runs are replaced one at a time by `O`'s simulator at their statements (`zkmalAt_layout_stop`, with
  `O`'s zero knowledge at each session's statement on the slot's coins: `ZKmalAt.of_coins`, `ZKmalAt.of_equiv`), which
  needs each session's witness in `O`'s relation unless the stop fired (`WitnessIn`);
* then the prefix's commitment strings are swapped for zero commitments at `2 δ₁` each (`zkmalAt_prefix`, through
  `prefix_close` and `Hm96Hiding`), which is the prefix's simulator `S₀` (`Rec.Compiled.S₀_eval`).

Its corollaries are the leak at public stop positions, the multiplicative form: over any recursion (`recZK_leak`) and
at the deployed recursion's stop (`zkStop_leak`, from `Claims.Recursion.ZKStop`), through `Claims.zkmalRLeak_of_cuts`.
-/

@[expose] public section

namespace Verity

open Claims Claims.ZKMal
open scoped NNReal ENNReal

namespace Rec.Compiled

variable {W WS Sa Gp : Type} (G : Rec.Compiled W WS Sa Gp)

theorem lay_start [Fintype Sa] [Nonempty Sa] (x : Msg) : (G.lay x).start = G.start := by
  funext x' j
  induction j with
  | zero => rfl
  | succ j ih =>
    show (G.lay x).start x' j + G.n x' j = G.start x' j + G.n x' j
    rw [ih]

end Rec.Compiled

theorem Rec.queries_eval {α : Type} (H : Msg → Msg) :
    ∀ (n : ℕ) (q : Fin n → Msg) (k : (Fin n → Msg) → OC α),
      ((Rec.queries n q k).eval H).1 = ((k fun i => H (q i)).eval H).1
  | 0, q, k => by
      have e : (fun i : Fin 0 => H (q i)) = Fin.elim0 := funext fun i => i.elim0
      simp only [Rec.queries, e]
  | n + 1, q, k => by
      simp only [Rec.queries, OC.eval]
      rw [Rec.queries_eval H n]
      have e : (Fin.cons (H (q 0)) fun i : Fin n => H (q i.succ) : Fin (n + 1) → Msg) = fun i => H (q i) :=
        funext fun i => Fin.cases (by simp) (fun j => by simp) i
      rw [e]

/-- The prefix's run from its next-message function: `kI` leaves under `dI`, then the stage message, as `prefixRun`. -/
theorem interact_prefixRun {Sa Gp Cm : Type} [AddCommGroup Gp] (Hl : Gp × Cm → Msg) (My : Sa → Gp) (c : Sa → Cm)
    (ask : List Msg → Out) (kI nC : ℕ)
    (dI : List (Msg × Msg) → Gp) (dS : List (Msg × Msg) → Fin nC → Gp)
    (stg : List (Msg × Msg) → (Fin nC → Msg) → Msg) (yI : Fin kI → Sa) (yS : Fin nC → Sa)
    (pm : List (Msg × Msg) → Msg) (hI : ∀ h, h.length < kI → pm h = leafPm Hl kI dI My c yI h)
    (hS : ∀ h, h.length = kI → pm h = stg h fun j => Hl (dS h j + My (yS j), c (yS j))) :
    interact ask pm (kI + 1) [] = prefixRun Hl My c ask kI nC dI dS stg yI yS := by
  have hin : interact ask pm kI [] = interact ask (leafPm Hl kI dI My c yI) kI [] :=
    interact_congr_lt ask _ _ kI (fun h hh => hI h hh) kI [] (by simp)
  rw [interact_add ask _ kI 1 [], hin, List.nil_append]
  unfold prefixRun cont
  split
  · rename_i hg
    congr 1
    rw [interact_one]
    conv_rhs => rw [interact_one]
    congr 2
    funext _
    exact hS _ (by rw [histOf_length, goesOn_length hg])
  · rfl

section Prefix

variable {Sa Gp : Type} [Fintype Sa] [Nonempty Sa] [AddCommGroup Gp] [Fintype Gp]

/-- **The prefix is zero knowledge at `2 (kI + nC) δ₁`**: a prover whose `kI` rounds are hm96 leaves of its messages and
whose stage round is the stage message of `nC` leaves (`hC`) is simulated by the zero messages under fresh salts
(`hS`), by `Hm96Hiding` at each leaf (`prefix_close`). -/
theorem zkmalAt_prefix {W R : Type} [Fintype R] [Nonempty R] (H : Msg → Msg) (My : Sa → Gp) (cS : Sa → Msg)
    {δ₁ : ℝ≥0} (hT1 : Hm96Hiding My cS δ₁) (Hl : Gp × Msg → Msg) (kI nC : ℕ)
    (stg : List (Msg × Msg) → (Fin nC → Msg) → Msg) (P₀ : W → Prover ((R × (Fin kI → Sa)) × (Fin nC → Sa)))
    (RA : W → Prop) (x : Msg)
    (hC : ∀ w, RA w → ∃ (dI : R → List (Msg × Msg) → Gp)
      (dS : R × (Fin kI → Sa) → List (Msg × Msg) → Fin nC → Gp),
      ∀ (t : (R × (Fin kI → Sa)) × (Fin nC → Sa)) (h : List (Msg × Msg)),
        (∀ hh : h.length < kI, (((P₀ w).next x h t).eval H).1 =
          Hl (dI t.1.1 h + My (t.1.2 ⟨h.length, hh⟩), cS (t.1.2 ⟨h.length, hh⟩))) ∧
        (h.length = kI → (((P₀ w).next x h t).eval H).1 =
          stg h fun j => Hl (dS t.1 h j + My (t.2 j), cS (t.2 j))))
    (prog : (Fin kI → Sa) × (Fin nC → Sa) → SP (List (Msg × Out)))
    (hS : ∀ (ask : List Msg → Out) c, (SP.eval H ask (prog c)).1 =
      prefixRun Hl My cS ask kI nC (fun _ => 0) (fun _ _ => 0) stg c.1 c.2) :
    ZKmalAt H (kI + 1) P₀ RA x ((Fin kI → Sa) × (Fin nC → Sa)) prog (2 * (kI + nC) * δ₁) 0 := by
  intro w hw Θ Vs θ hV Ev
  obtain ⟨dI, dS, hP⟩ := hC w hw
  have hreal : Pr (fun ω : (R × (Fin kI → Sa)) × (Fin nC → Sa) => Ev (runStar H (P₀ w) Vs x ω θ).msgs) =
      Pr (fun p : (R × (Fin kI → Sa)) × (Fin nC → Sa) => Ev (prefixRun Hl My cS
        (fun as => ((Vs.next x as θ).eval H).1) kI nC (dI p.1.1) (dS p.1) stg p.1.2 p.2)) := by
    congr 1
    funext ω
    rw [runStar_msgs, hV]
    refine congrArg Ev (interact_prefixRun Hl My cS _ kI nC (dI ω.1.1) (dS ω.1) stg ω.1.2 ω.2 _
      (fun h hh => ?_) (fun h hh => (hP ω h).2 hh))
    rw [(hP ω h).1 hh, leafPm, dif_pos hh]
  have hsim : Pr (fun c : (Fin kI → Sa) × (Fin nC → Sa) => Ev (spOut H Vs θ x (prog c))) =
      Pr (fun q : (Fin kI → Sa) × (Fin nC → Sa) => Ev (prefixRun Hl My cS
        (fun as => ((Vs.next x as θ).eval H).1) kI nC (fun _ => 0) (fun _ _ => 0) stg q.1 q.2)) := by
    congr 1
    funext c
    exact congrArg Ev (hS _ c)
  rw [hreal, hsim]
  refine (prefix_close My cS hT1 Hl (fun as => ((Vs.next x as θ).eval H).1) kI nC dI dS stg Ev).trans
    (le_of_eq ?_)
  simp only [zero_mul, add_zero]

end Prefix

/-- A simulator on coins in bijection with another's is zero knowledge at the same bound. -/
theorem _root_.Claims.ZKMal.ZKmalAt.of_coins {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω]
    {H : Msg → Msg} {k : ℕ}
    {P : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} {C C' : Type} [Fintype C] [Fintype C']
    {prog : C → SP (List (Msg × Out))} {ε γ : ℝ} (h : ZKmalAt H k P R x C prog ε γ) (e : C' ≃ C) :
    ZKmalAt H k P R x C' (fun c => prog (e c)) ε γ := by
  intro w hw Θ Vs θ hV Ev
  rw [Claims.ZKMal.Pr_equiv e (fun c => Ev (spOut H Vs θ x (prog c)))]
  exact h w hw Θ Vs θ hV Ev

namespace Rec.Compiled

variable {W WS Sa Gp : Type} [Fintype Sa] [Nonempty Sa] (G : Rec.Compiled W WS Sa Gp)

/-- The prefix's simulator runs `prefixRun` at zero messages. -/
theorem S₀_eval (H : Msg → Msg) [AddCommGroup Gp] (x : Msg) (ask : List Msg → Out)
    (c : (Fin (G.kI x) → Sa) × (Fin (G.nC x) → Sa)) :
    (SP.eval H ask (G.S₀.prog x c)).1 = prefixRun G.str G.mask (fun y => H (G.saltMsg y)) ask (G.kI x) (G.nC x)
      (fun _ => 0) (fun _ _ => 0) (G.stg x) c.1 c.2 := by
  show (SP.eval H ask (spRun (G.zeroMsg x c) (G.kI x + 1) [])).1 = _
  rw [spRun_eval]
  refine interact_prefixRun _ _ _ ask _ _ _ _ _ c.1 c.2 _ (fun h hh => ?_) (fun h hh => ?_)
  · rw [zeroMsg, dif_pos hh, leafPm, dif_pos hh, zero_add]
    rfl
  · rw [zeroMsg, dif_neg (show ¬ h.length < G.kI x by omega), Rec.queries_eval]
    simp only [OC.eval, zero_add]


end Rec.Compiled

/-- **`RecZK`**: at each statement, `zkmalAt_layout_stop` over the prefix (`zkmalAt_prefix`) and the sessions, each
`O`'s zero knowledge at its statement on the slot's coins, with its witness from `WitnessIn`. -/
theorem SecurityProofs.RecZK : Guarantees.RecZK := by
  intro W WS Sa Gp NO ΩO _ _ _ _ _ _ _ _ H O G δ₁ go hgo hO hHid hC hS hW
  refine (zkmalRx_iff_at _ _ _ _ _ _ _ _ _).2 fun x => ?_
  obtain ⟨hg, hB⟩ := hS
  have h := zkmalAt_layout_stop (G.T₀ x) (G.lay x) G.stop H x (G.nS x) (fun w => G.P₀ w x)
    (Rec.slotP O (G.NB x) (G.ΩB x)) G.R O.R G.S₀ (fun j x' => G.CS x' j)
    (fun j x' u => progOf O.S (G.CS x' j) (G.xs x' j u)) go
    (2 * ((G.kI x : ℝ) + G.nC x) * δ₁) (G.γS x) (G.εS δ₁ x)
    (zkmalAt_prefix H G.mask (fun y => H (G.saltMsg y)) hHid G.str (G.kI x) (G.nC x) (G.stg x)
      (fun w => G.P₀ w x) (G.R x) x (hC x) (G.S₀.prog x) (G.S₀_eval H x))
    (fun j hj u hu hf => by
      rw [G.lay_start] at hu
      obtain ⟨hk, hε, hγ, hCe, hTe⟩ := (hB x j hj).2 u hu hf
      have h1 := (((zkmalRx_iff_at _ _ _ _ _ _ _ _ _).1 hO (G.xs x j u)).of_coins hCe.some).of_equiv
        (Rec.slotP O (G.NB x) (G.ΩB x)) hTe.some.1 hTe.some.2 fun w hist ω => by
          simp only [Rec.slotP, dif_pos hTe]
      show ZKmalAt H (G.n x j) (Rec.slotP O (G.NB x) (G.ΩB x)) (O.R (G.xs x j u)) (G.xs x j u) (G.CS x j)
        (progOf O.S (G.CS x j) (G.xs x j u)) (G.εS δ₁ x j) (G.γS x)
      rw [progOf_of hCe, ← hk]
      exact h1.mono hε hγ)
    (fun j hj w hw t₀ u hu hp hf => hW x j hj w hw t₀ u (by rwa [G.lay_start] at hu) hp hf)
    (fun j hj ask pm => by rw [G.lay_start]; exact hgo x j hj ask pm)
    (fun j hj => (hB x j hj).1) (hg x)
  rw [G.lay_start] at h
  exact h

namespace Rec.Compiled

variable {W WS Sa Gp : Type} (G : Rec.Compiled W WS Sa Gp)

theorem start_pos (x : Msg) : ∀ j, 0 < G.start x j
  | 0 => Nat.succ_pos _
  | j + 1 => by
    show 0 < G.start x j + G.n x j
    exact Nat.lt_of_lt_of_le (start_pos x j) (Nat.le_add_right _ _)

/-- `goesOn` itself is a continuation test `RecZK` takes. -/
theorem goesOnAt_goesOn : G.GoesOnAt fun j x => goesOn (G.start x j) :=
  fun _ _ _ _ _ => rfl

/-- **That the last answer is a message** (`Claims.Recursion.goesOn`, `Claims.Recursion.SimStop`'s test) is a
continuation test `RecZK` takes: every session starts after the prefix's rounds, so a real run there is cut short only
by a decision, its last answer. -/
theorem goesOnAt_recursion : G.GoesOnAt fun _ _ => Claims.Recursion.goesOn :=
  fun x j _ ask pm => recGoesOn_interact ask pm _ [] (G.start_pos x j)

end Rec.Compiled

/-- **The leak at public stop positions, over any recursion**: a prover that releases `Rec.system O G go`'s run cut at
one of `|pos x|` public positions leaks at most `log₂ |pos x|` bits beyond `RecZK`'s bound. -/
theorem recZK_leak {W WS Sa Gp : Type} {NO ΩO : Msg → Type} [Fintype Sa] [Nonempty Sa] [AddCommGroup Gp]
    [Fintype Gp] [∀ x, Fintype (NO x)] [∀ x, Nonempty (NO x)] [∀ x, Fintype (ΩO x)] [∀ x, Nonempty (ΩO x)]
    {K : Msg → Type} [∀ x, Fintype (K x)] [∀ x, Nonempty (K x)]
    {H : Msg → Msg} {O : ZkSystem WS NO ΩO} {G : Rec.Compiled W WS Sa Gp} {δ₁ : ℝ≥0}
    {go : ℕ → Msg → List (Msg × Out) → Bool} (hgo : G.GoesOnAt go)
    (hO : O.ZK H δ₁) (hHid : Hm96Hiding G.mask (fun y => H (G.saltMsg y)) δ₁) (hC : G.Commits H)
    (hS : G.ByShape O δ₁) (hW : G.WitnessIn H O)
    (P' : W → (x : Msg) → Prover ((((G.T₀ x).iter 0 (G.NB x) (G.ΩB x) (G.nS x)).N ×
      ((G.T₀ x).iter 0 (G.NB x) (G.ΩB x) (G.nS x)).Ω) × K x))
    (stop : Msg) (pos : Msg → Finset ℕ) (hcut : CutsOf H (Rec.system O G go).k P' (Rec.system O G go).P stop pos)
    (hne : ∀ x, (pos x).Nonempty) :
    ZKmalRLeak (fun x => ((G.T₀ x).iter 0 (G.NB x) (G.ΩB x) (G.nS x)).N)
      (fun x => ((G.T₀ x).iter 0 (G.NB x) (G.ΩB x) (G.nS x)).Ω) K H (Rec.system O G go).k P'
      (Rec.system O G go).P G.R (leakyAt (Rec.system O G go).S (Rec.system O G go).k stop pos hne)
      (fun x => (pos x).card) ((Rec.system O G go).εzk δ₁) (Rec.system O G go).γzk :=
  zkmalRLeak_of_cuts (SecurityProofs.RecZK H O G δ₁ go hgo hO hHid hC hS hW) hcut hne

/-- **The deployed recursion's leak at its stop positions**: from its zero knowledge with the stop
(`Claims.Recursion.ZKStop`), a release of its run cut at one of `|pos x|` public positions leaks at most
`log₂ |pos x|` bits. -/
theorem zkStop_leak {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    {K : Msg → Type} [∀ x, Fintype (K x)] [∀ x, Nonempty (K x)] {H : Msg → Msg}
    {D : Claims.Recursion.StopDeploy Coin} {I : Claims.Recursion.Inner Coin Claims.Flock.Wit}
    {DO : Claims.Flock.Deploy Coin Col T} (hZK : Claims.Recursion.ZKStop H D I DO) {t : ℕ} (ht : 0 < t)
    {δ₁ : ℝ≥0} (hδ : Hm96Hiding D.mask (D.toDeploy.saltHash H) δ₁)
    (P' : Claims.Flock.Wit → (x : Msg) → Prover ((D.Nonce x × D.Rest x) × K x)) (stop : Msg)
    (pos : Msg → Finset ℕ) (hcut : CutsOf H D.V.k P' D.prover stop pos) (hne : ∀ x, (pos x).Nonempty) :
    ZKmalRLeak D.Nonce D.Rest K H D.V.k P' D.prover (Claims.Recursion.R D.toDeploy I)
      (leakyAt (Claims.Recursion.SimStop D I (D.SO t) H) D.V.k stop pos hne) (fun x => (pos x).card)
      (Claims.Recursion.εzkStop D I t δ₁) (Claims.Recursion.γzk D.toDeploy t) :=
  zkmalRLeak_of_cuts (hZK.2 t ht δ₁ hδ) hcut hne

end Verity
