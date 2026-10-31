import Proofs.Flock.Claims.Recursion.SeqSim
import Proofs.Flock.Claims.ZK.Capacity
import Proofs.Flock.Claims.ZK.Prefix

/-!
# The masked recursion on one capped tape

`ZKBoundM` is `ZKmalRx`: at each statement the masked prover runs on that statement's nested tape (`SeqTape.iter` at
its `n_S(x)` V\* sessions), so the tape grows with the statement. On one tape for every statement (`ZKmalR`), the nested
tape of `M` sessions, read at a statement of `n_S(x) ≤ M` sessions down to its own, the same bound holds there:

* `ZKmalAt.of_fibers`: `ZKmalAt` carries over to a prover that reads its tape through a map of the nonces and one of
  the rests, each with fibers of one size: the real transcript's law and `Fork`'s collision probability are the same.
* `zkmalR_of_rx`: so `ZKmalRx` gives `ZKmalR` on one tape that each statement of the relation reads so.
* `SeqTape.capN`, `SeqTape.capΩ`: the nested tape of `m ≤ M` sessions in that of `M`: at each level above `m`, the
  nonce's part for the sessions before it, and the rest's (`fibersEq_capN`, `fibersEq_capΩ`). Reading through the
  guess instead (`forkTape`) would make `Fork`'s two runs share the levels above `m`, and lose `Fork`'s bound.
* `RecDeploy.zkBoundMCap_of`: `ZKBoundM` gives the masked recursion's zero knowledge on the nested tape of `M`
  sessions (`ZKBoundMCap`), at the same `Sim`, `εzk` and `γzk`, wherever the relation's statements have at most `M`
  V\* sessions.

The reading down to `n_S(x)` sessions depends on the statement, so a record on the capped tape reads it in its
prover (`Deploy.P`), and `ZkDeploy.tapeOf`, which reads no statement, is the identity there. The first part is on one
tape `T₀` too, so where it commits under fresh salts (`PreCommits`), `T₀` splits evenly into each statement's inner and
staging salts: their numbers are bounded as well.
-/

namespace Claims.ZKMal

open Claims

/-! ## Maps with fibers of one size -/

theorem FibersEq.prod {α β α' β' : Type} {f : α → β} {g : α' → β'} (hf : FibersEq f) (hg : FibersEq g) :
    FibersEq fun a : α × α' => (f a.1, g a.2) := by
  have k : ∀ b : β × β', {a : α × α' // (f a.1, g a.2) = b} ≃ {a // f a = b.1} × {a // g a = b.2} := fun b =>
    { toFun := fun a => (⟨a.1.1, congrArg Prod.fst a.2⟩, ⟨a.1.2, congrArg Prod.snd a.2⟩)
      invFun := fun p => ⟨(p.1.1, p.2.1), Prod.ext p.1.2 p.2.2⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  intro b b'
  rw [Nat.card_congr (k b), Nat.card_congr (k b'), Nat.card_prod, Nat.card_prod, hf b.1 b'.1, hg b.2 b'.2]

/-- Through a map with fibers of one size onto a nonempty type, an event's probability is its probability on the
target. -/
theorem Pr_of_fibers {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] {f : α → β} (hf : FibersEq f)
    (p : β → Prop) : Pr (fun a => p (f a)) = Pr p :=
  Pr_comp_fiber f (fun b => hf b (Classical.arbitrary β)) p

/-- **`ZKmalAt` for a prover that runs `P` at `x` on its tape through a map of the nonces and one of the rests, each
with fibers of one size**: the real transcript has `P`'s law, and `Fork`'s two runs, on one nonce and two rests, are
`P`'s on one nonce and two rests. -/
theorem ZKmalAt.of_fibers {W N Ω N' Ω' : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] [Fintype N']
    [Nonempty N'] [Fintype Ω'] [Nonempty Ω'] {H : Msg → Msg} {k : ℕ} {P : W → Prover (N × Ω)} {R : W → Prop}
    {x : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))} {ε γ : ℝ}
    (h : ZKmalAt H k P R x C prog ε γ) (P' : W → Prover (N' × Ω')) (fN : N' → N) (fΩ : Ω' → Ω)
    (hN : FibersEq fN) (hΩ : FibersEq fΩ)
    (hP : ∀ w hist ω, (P' w).next x hist ω = (P w).next x hist (fN ω.1, fΩ ω.2)) :
    ZKmalAt H k P' R x C prog ε γ := by
  intro w hw Θ Vs θ hV Ev
  have hrun : ∀ ω : N' × Ω', runStar H (P' w) Vs x ω θ = runStar H (P w) Vs x (fN ω.1, fΩ ω.2) θ :=
    fun ω => runStar_congr H (P w) (P' w) Vs x _ ω θ fun hist => hP w hist ω
  have hr : Pr (fun ω : N' × Ω' => Ev (runStar H (P' w) Vs x ω θ).msgs) =
      Pr (fun ω : N × Ω => Ev (runStar H (P w) Vs x ω θ).msgs) := by
    simp only [hrun]
    exact Pr_of_fibers (f := fun ω : N' × Ω' => (fN ω.1, fΩ ω.2)) (hN.prod hΩ)
      fun ω => Ev (runStar H (P w) Vs x ω θ).msgs
  have hf : (Fork H (P' w) Vs θ x).Pr H = (Fork H (P w) Vs θ x).Pr H := by
    simp only [Finder.Pr, Fork, hrun]
    exact Pr_of_fibers (f := fun s : N' × Ω' × Ω' => (fN s.1, fΩ s.2.1, fΩ s.2.2)) (hN.prod (hΩ.prod hΩ))
      fun s : N × Ω × Ω => ∃ u v,
        findColl ((runStar H (P w) Vs x (s.1, s.2.1) θ).pLog ++ (runStar H (P w) Vs x (s.1, s.2.2) θ).pLog) =
          some (u, v) ∧ Collision H u v
  rw [hr, hf]
  exact h w hw Θ Vs θ hV Ev

/-- **`ZKmalRx` on one tape**: where each statement the relation holds at reads the one tape `N' × Ω'` through a map
of the nonces and one of the rests with fibers of one size, the prover that reads it so is `ZKmalR` at the same
simulator and errors. -/
theorem zkmalR_of_rx {W : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {N' Ω' : Type} [Fintype N'] [Nonempty N'] [Fintype Ω'] [Nonempty Ω']
    {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)} {R : Rel W} {S : Simulator}
    {εzk γ : Msg → ℝ} (h : ZKmalRx N Ω H k P R S εzk γ) (fN : (x : Msg) → N' → N x) (fΩ : (x : Msg) → Ω' → Ω x)
    (hf : ∀ x w, R x w → FibersEq (fN x) ∧ FibersEq (fΩ x)) :
    ZKmalR H k (fun w => ⟨fun x hist ω => (P w x).next x hist (fN x ω.1, fΩ x ω.2)⟩) R S εzk γ := by
  refine (zkmalR_iff_at _ _ _ _ _ _ _).2 fun x => ?_
  intro w hw
  exact ZKmalAt.of_fibers (h x) (fun w => ⟨fun x hist ω => (P w x).next x hist (fN x ω.1, fΩ x ω.2)⟩) (fN x) (fΩ x)
    (hf x w hw).1 (hf x w hw).2 (fun _ _ _ => rfl) w hw

/-! ## The nested tape of fewer sessions -/

theorem fibersEq_fst {A B : Type} : FibersEq fun a : A × B => a.1 :=
  fibersEq_of_equiv (Equiv.refl _) fun _ => rfl

theorem fibersEq_snd_fst {A B C D : Type} : FibersEq fun a : A × B × C × D => a.2.1 :=
  fibersEq_of_equiv (D := A × C × D)
    ⟨fun a => (a.2.1, a.1, a.2.2.1, a.2.2.2), fun b => (b.2.1, b.1, b.2.2.1, b.2.2.2), fun _ => rfl, fun _ => rfl⟩
    fun _ => rfl

namespace SeqTape

variable (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]

/-- **The first `m` sessions' nonce in the first `m + k`'s**: at each level above `m`, the nonce's part for the
sessions before it. -/
def downN (m : ℕ) : (k : ℕ) → (T₀.iter c NB ΩB (m + k)).N → (T₀.iter c NB ΩB m).N
  | 0 => id
  | k + 1 => fun s => downN m k s.2.1

/-- **The first `m` sessions' rest in the first `m + k`'s**: at each level above `m`, the rest's part for the sessions
before it. -/
def downΩ (m : ℕ) : (k : ℕ) → (T₀.iter c NB ΩB (m + k)).Ω → (T₀.iter c NB ΩB m).Ω
  | 0 => id
  | k + 1 => fun s => downΩ m k s.1

theorem fibersEq_downN (m : ℕ) : ∀ k, FibersEq (T₀.downN c NB ΩB m k)
  | 0 => fibersEq_of_bijective Function.bijective_id
  | k + 1 => FibersEq.comp (f := fun s : Fin (c + (m + k) + 1) × (T₀.iter c NB ΩB (m + k)).N × NB ×
      (T₀.iter c NB ΩB (m + k)).Ω => s.2.1) fibersEq_snd_fst (fibersEq_downN m k)

theorem fibersEq_downΩ (m : ℕ) : ∀ k, FibersEq (T₀.downΩ c NB ΩB m k)
  | 0 => fibersEq_of_bijective Function.bijective_id
  | k + 1 => FibersEq.comp (f := fun s : (T₀.iter c NB ΩB (m + k)).Ω × ΩB => s.1) fibersEq_fst (fibersEq_downΩ m k)

open Classical in
/-- **The first `m` sessions' nonce in the first `M`'s**, for `m ≤ M` (`downN`); arbitrary for `m > M`. -/
noncomputable def capN (M m : ℕ) : (T₀.iter c NB ΩB M).N → (T₀.iter c NB ΩB m).N :=
  if h : m ≤ M then fun s =>
    T₀.downN c NB ΩB m (M - m) (cast (congrArg (fun n => (T₀.iter c NB ΩB n).N) (Nat.add_sub_cancel' h).symm) s)
  else fun _ => Classical.arbitrary _

open Classical in
/-- **The first `m` sessions' rest in the first `M`'s**, for `m ≤ M` (`downΩ`); arbitrary for `m > M`. -/
noncomputable def capΩ (M m : ℕ) : (T₀.iter c NB ΩB M).Ω → (T₀.iter c NB ΩB m).Ω :=
  if h : m ≤ M then fun s =>
    T₀.downΩ c NB ΩB m (M - m) (cast (congrArg (fun n => (T₀.iter c NB ΩB n).Ω) (Nat.add_sub_cancel' h).symm) s)
  else fun _ => Classical.arbitrary _

theorem fibersEq_capN {M m : ℕ} (h : m ≤ M) : FibersEq (T₀.capN c NB ΩB M m) := by
  unfold capN
  split
  · exact FibersEq.comp (fibersEq_of_bijective (cast_bijective _)) (T₀.fibersEq_downN c NB ΩB m (M - m))
  · contradiction

theorem fibersEq_capΩ {M m : ℕ} (h : m ≤ M) : FibersEq (T₀.capΩ c NB ΩB M m) := by
  unfold capΩ
  split
  · exact FibersEq.comp (fibersEq_of_bijective (cast_bijective _)) (T₀.fibersEq_downΩ c NB ΩB m (M - m))
  · contradiction

end SeqTape

end Claims.ZKMal

namespace RecDeploy

open _root_.Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts ZKmalAt SeqTape)
open scoped NNReal

variable (IV : Claims.Verifier Coin) (b : Brings)

/-- **The masked prover on the nested tape of `M` V\* sessions**: at a statement of `n_S(x) ≤ M` sessions, `ZKBoundM`'s
prover there, on the nested tape of `n_S(x)` sessions read down from `M`'s (`SeqTape.capN`, `SeqTape.capΩ`). -/
noncomputable def refPMCap (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (M : ℕ) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (w : Claims.Flock.Wit) :
    Claims.Prover ((T₀.iter 0 NB ΩB M).N × (T₀.iter 0 NB ΩB M).Ω) :=
  ⟨fun x hist ω => ((refPM IV b pre wit PS w).comap (RunParts.flatten T₀ 0 NB ΩB (nS x))).next x hist
    (T₀.capN 0 NB ΩB M (nS x) ω.1, T₀.capΩ 0 NB ΩB M (nS x) ω.2)⟩

/-- **`ZKBoundM` on one tape**: the masked recursion's zero knowledge at `ZKmalR`, on the nested tape of `M` V\*
sessions (`refPMCap`), at the claim's `Sim`, `εzk` and `γzk`. -/
def ZKBoundMCap (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB]
    [Fintype ΩB] [Nonempty ΩB] (M : ℕ) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) : Prop :=
  ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
    Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
      Claims.ZKmalR H (kM IV b) (refPMCap IV b T₀ NB ΩB M pre wit PS)
        (Claims.Recursion.R (deployRec IV ivScope b) I) (Claims.Recursion.Sim (deployRec IV ivScope b) I (SO t))
        (Claims.Recursion.εzk (deployRec IV ivScope b) I t δ₁) (Claims.Recursion.γzk (deployRec IV ivScope b) t)

/-- **The masked recursion on the nested tape of `M` sessions, from `ZKBoundM` with one first-part tape `T₀`**, wherever
the relation's statements have at most `M` V\* sessions (`hcap`): at each, the nested tape of `M` sessions reads its own
with fibers of one size,
nonce and rest apart (`SeqTape.fibersEq_capN`, `SeqTape.fibersEq_capΩ`), so `ZKmalAt` carries over at the same errors
(`zkmalR_of_rx`). -/
theorem zkBoundMCap_of (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB]
    [Fintype ΩB] [Nonempty ΩB] (M : ℕ) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (h : ZKBoundM IV b H ivScope I (fun _ => T₀) NB ΩB (fun _ => pre) wit PS SO)
    (hcap : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → nS x ≤ M) :
    ZKBoundMCap IV b H ivScope I T₀ NB ΩB M pre wit PS SO := fun t ht δ₁ hδ =>
  Claims.ZKMal.zkmalR_of_rx (h t ht δ₁ hδ) (fun x => T₀.capN 0 NB ΩB M (nS x)) (fun x => T₀.capΩ 0 NB ΩB M (nS x))
    fun x w hw => ⟨T₀.fibersEq_capN 0 NB ΩB (hcap x w hw), T₀.fibersEq_capΩ 0 NB ΩB (hcap x w hw)⟩

end RecDeploy
