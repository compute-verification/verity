module

public import Proofs.Flock.Claims.ZK.Rewind
public import Proofs.Flock.Claims.ZK.Interact
public import Proofs.Flock.Claims.Recursion.Hybrid

/-!
# The leak of a run released cut at a public position

A prover `P'` that releases `P`'s run cut at one of `|pos x|` public positions (`CutsOf`) leaks at most
`log₂ |pos x|` bits beyond `P`'s zero knowledge (`ZKmalRLeak`, `zkmalRLeak_of_cuts`). Its simulator samples the position
uniformly, runs `P`'s and cuts its transcript there (`leakyAt`): an event of `P'`'s transcript is one at some cut of
`P`'s, so the union bound over the positions gives `|pos x|` times the leaky simulator's probability.
-/

@[expose] public section

namespace Claims

open Claims.ZKMal

/-- **The leaky simulator**: a uniform position `t ∈ pos x` and `S`'s coins; `S`'s transcript, cut at round `t`. -/
noncomputable def leakyAt (S : Simulator) (k : Msg → ℕ) (stop : Msg) (pos : Msg → Finset ℕ)
    (hne : ∀ x, (pos x).Nonempty) : Simulator where
  C x := {t // t ∈ pos x} × S.C x
  ne x := ⟨(⟨(hne x).choose, (hne x).choose_spec⟩, Classical.arbitrary _)⟩
  prog x tc := spBind (S.prog x tc.2) fun τ => spRun (fun h => .ret (cutMsg stop tc.1.1 τ h)) (k x) []

/-- The leaky simulator's transcript is `S`'s cut at its position. -/
theorem leakyAt_out {Θ : Type} (H : Msg → Msg) (S : Simulator) (k : Msg → ℕ) (stop : Msg) (pos : Msg → Finset ℕ)
    (hne : ∀ x, (pos x).Nonempty) (Vs : VStar Θ) (θ : Θ) (x : Msg) (tc : {t // t ∈ pos x} × S.C x) :
    (leakyAt S k stop pos hne).out H Vs θ x tc =
      cutT (fun as => ((Vs.next x as θ).eval H).1) stop tc.1.1 (k x) (S.out H Vs θ x tc.2) := by
  simp only [Simulator.out, leakyAt, spBind_eval, spRun_eval]
  rfl

/-- **A cut release leaks at most `log₂ |pos x|` bits**: from `P`'s zero knowledge and `P'`'s cuts of `P`'s run at
public positions, `P'` is `ZKmalRLeak` at the leaky simulator with `L x = |pos x|`. -/
theorem zkmalRLeak_of_cuts {W : Type} {N Ω K : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] [∀ x, Fintype (K x)] [∀ x, Nonempty (K x)] {H : Msg → Msg}
    {k : Msg → ℕ} {P' : W → (x : Msg) → Prover ((N x × Ω x) × K x)} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} {stop : Msg} {pos : Msg → Finset ℕ}
    (hZK : ZKmalRx N Ω H k P R S εzk γ) (hcut : CutsOf H k P' P stop pos) (hne : ∀ x, (pos x).Nonempty) :
    ZKmalRLeak N Ω K H k P' P R (leakyAt S k stop pos hne) (fun x => (pos x).card) εzk γ := by
  intro x w hw Θ Vs θ hV Ev
  have h1 : Pr (fun ω : (N x × Ω x) × K x => Ev (runStar H (P' w x) Vs x ω θ).msgs) ≤
      Pr (fun ω : N x × Ω x => ∃ t ∈ pos x,
        Ev (cutT (fun as => ((Vs.next x as θ).eval H).1) stop t (k x) (runStar H (P w x) Vs x ω θ).msgs)) := by
    rw [← Pr_fst (β := K x) (fun ω : N x × Ω x => ∃ t ∈ pos x,
      Ev (cutT (fun as => ((Vs.next x as θ).eval H).1) stop t (k x) (runStar H (P w x) Vs x ω θ).msgs))]
    refine Pr_mono fun ωκ hE => ?_
    obtain ⟨t, ht, he⟩ := hcut x w Θ Vs θ hV ωκ.1 ωκ.2
    refine ⟨t, ht, ?_⟩
    rw [← he]
    exact hE
  have h2 := hZK x w hw Θ Vs θ hV fun τ => ∃ t ∈ pos x,
    Ev (cutT (fun as => ((Vs.next x as θ).eval H).1) stop t (k x) τ)
  beta_reduce at h2
  have hsum : Pr (fun c : S.C x => ∃ t ∈ pos x,
        Ev (cutT (fun as => ((Vs.next x as θ).eval H).1) stop t (k x) (S.out H Vs θ x c))) ≤
      ∑ t : {t // t ∈ pos x}, Pr (fun c : S.C x =>
        Ev (cutT (fun as => ((Vs.next x as θ).eval H).1) stop t.1 (k x) (S.out H Vs θ x c))) := by
    refine le_trans (le_of_eq (Pr_congr fun c => ?_))
      (Claims.Pr_exists_le (fun (t : {t // t ∈ pos x}) (c : S.C x) =>
      Ev (cutT (fun as => ((Vs.next x as θ).eval H).1) stop t.1 (k x) (S.out H Vs θ x c))))
    exact ⟨fun ⟨t, ht, h⟩ => ⟨⟨t, ht⟩, h⟩, fun ⟨t, h⟩ => ⟨t.1, t.2, h⟩⟩
  have hleak : Pr (fun c : (leakyAt S k stop pos hne).C x => Ev ((leakyAt S k stop pos hne).out H Vs θ x c)) =
      Ex (fun t : {t // t ∈ pos x} => Pr (fun c : S.C x =>
        Ev (cutT (fun as => ((Vs.next x as θ).eval H).1) stop t.1 (k x) (S.out H Vs θ x c)))) := by
    refine (Claims.ZKMal.Pr_prod_left (α := {t // t ∈ pos x}) (β := S.C x) _).trans ?_
    congr 1
    funext t
    congr 1
    funext c
    exact congrArg Ev (leakyAt_out H S k stop pos hne Vs θ x (t, c))
  have hcard : ∀ f : {t // t ∈ pos x} → ℝ, ∑ t, f t = (pos x).card * Ex f := by
    intro f
    unfold Ex
    rw [Fintype.card_coe]
    have : ((pos x).card : ℝ) ≠ 0 := by exact_mod_cast (hne x).card_pos.ne'
    field_simp
  rw [hcard] at hsum
  show _ ≤ ((pos x).card : ℝ) * _ + _ + _
  rw [hleak]
  linarith [(abs_sub_le_iff.1 h2).1]

end Claims
