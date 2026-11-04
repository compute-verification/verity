import Proofs.Flock.Claims.Recursion.Seq

/-!
# The composed simulator is the claim's `Sim`

`simM` (`Recursion.Seq`) runs the first part's simulator, then each V\* session's outer simulator at the statement its
simulated history stages, nested as sequential composition nests them. At the claim's first-part simulator `simPre`
(the inner session's and the staging round's commit strings to the zero digest under fresh salts, `Sim`'s first two
parts), it is `Claims.Recursion.Sim` on coins in bijection with `Sim`'s (`simEquiv`): the two programs give one
transcript against every verifier (`sim_eval`), so `ZKmalAt` at `simM` is `ZKmalAt` at `Sim` (`zkAt_sim`), where the
inner protocol's rounds are the inner verifier's (`I.V.k = IV.k` at the inner statement).

* `Claims.ZKMal.simRounds_eval`: `simRounds`' transcript extends the one it starts from, and where the run goes on it
  has all its rounds, each answered with a message. So where `Sim`'s first part goes on, its transcript is `kPre x`
  rounds of messages (`simPre_goesOn`), and the V\* statements `Sim` reads off it are the ones `simM` reads off each
  session's simulated history (`xsOf_of_goesOn`).
* `Claims.ZKMal.simOuter_eval`, `iterSimG_parts_eval`: both simulators' transcripts as a fold of one step per session.
* `Claims.ZKMal.ZKmalAt.of_sim`: `ZKmalAt` carries over to a simulator with the same transcripts on coins in
  bijection.
-/

namespace Claims.ZKMal

open Claims

/-! ## The claim's simulator combinators are `ZKMal`'s -/

theorem recSpBind_eq {α β : Type} : ∀ (p : SP α) (f : α → SP β), Recursion.spBind p f = spBind p f
  | .ret _, _ => rfl
  | .hash q k, f => congrArg (SP.hash q) (funext fun a => recSpBind_eq (k a) f)
  | .ask as k, f => congrArg (SP.ask as) (funext fun a => recSpBind_eq (k a) f)

theorem recSpUnder_eq {α : Type} (pre : List Msg) : ∀ p : SP α, Recursion.spUnder pre p = spUnder pre p
  | .ret _ => rfl
  | .hash q k => congrArg (SP.hash q) (funext fun a => recSpUnder_eq pre (k a))
  | .ask as k => congrArg (SP.ask (pre ++ as)) (funext fun a => recSpUnder_eq pre (k a))

theorem recSpOf_eq {α : Type} : ∀ o : OC α, Recursion.spOf o = ocSP o
  | .ret _ => rfl
  | .query q k => congrArg (SP.hash q) (funext fun a => recSpOf_eq (k a))

theorem recGoesOn_snoc (ms : List (Msg × Out)) (a : Msg) (o : Out) :
    Recursion.goesOn (ms ++ [(a, o)]) = isMsg o := by
  cases o <;> simp [Recursion.goesOn, isMsg]

/-- **`simRounds`' transcript** extends the one it starts from by rounds `r`: where the run goes on after them, they are
all `n` rounds, each answered with a message, and there are rounds only where the run went on before them. -/
theorem simRounds_eval (H : Msg → Msg) (ask : List Msg → Out) (f : List (Msg × Out) → SP Msg) :
    ∀ (n : ℕ) (ms : List (Msg × Out)), ∃ r, (SP.eval H ask (Recursion.simRounds f n ms)).1 = ms ++ r ∧
      (Recursion.goesOn (ms ++ r) = true → r.length = n ∧ ∀ p ∈ r, isMsg p.2 = true) ∧
      (r ≠ [] → Recursion.goesOn ms = true)
  | 0, ms => ⟨[], by simp [Recursion.simRounds, SP.eval]⟩
  | n + 1, ms => by
      by_cases hg : Recursion.goesOn ms = true
      · obtain ⟨r, hr, h1, h2⟩ := simRounds_eval H ask f n
          (ms ++ [((SP.eval H ask (f ms)).1, ask (ms.map Prod.fst ++ [(SP.eval H ask (f ms)).1]))])
        refine ⟨((SP.eval H ask (f ms)).1, ask (ms.map Prod.fst ++ [(SP.eval H ask (f ms)).1])) :: r, ?_,
          fun hgo => ?_, fun _ => hg⟩
        · simp only [Recursion.simRounds, hg, ite_true, recSpBind_eq, spBind_eval, SP.eval]
          rw [hr]
          simp
        · have hgo' := hgo
          rw [show ∀ (p : Msg × Out), ms ++ p :: r = (ms ++ [p]) ++ r by simp] at hgo'
          obtain ⟨hl, hm⟩ := h1 hgo'
          have hb : isMsg (ask (ms.map Prod.fst ++ [(SP.eval H ask (f ms)).1])) = true := by
            by_cases hr0 : r = []
            · subst hr0
              simpa [recGoesOn_snoc] using hgo'
            · simpa [recGoesOn_snoc] using h2 hr0
          refine ⟨by simp [hl], fun p hp => ?_⟩
          rcases List.mem_cons.1 hp with rfl | hp
          · exact hb
          · exact hm p hp
      · refine ⟨[], ?_, fun h => absurd (by simpa using h) hg, fun h => absurd rfl h⟩
        simp [Recursion.simRounds, hg, SP.eval]

/-- A fold of steps that leave a stopped run as it is, from a stopped run, is that run. -/
theorem foldl_stuck {ι : Type} (step : List (Msg × Out) → ι → List (Msg × Out))
    (hstep : ∀ τ i, Recursion.goesOn τ = false → step τ i = τ) :
    ∀ (l : List ι) (τ : List (Msg × Out)), Recursion.goesOn τ = false → l.foldl step τ = τ
  | [], _, _ => rfl
  | i :: l, τ, h => by
      rw [List.foldl_cons, hstep τ i h]
      exact foldl_stuck step hstep l τ h

/-- **`simOuter`'s transcript** against `ask`: one step per session in order, each running the session's simulator
under the transcript so far while the run goes on. -/
theorem simOuter_eval (H : Msg → Msg) (ask : List Msg → Out) {n : ℕ} (SO : Msg → ℕ → Recursion.OSim) (x : Msg)
    (xs : List Msg) (c : (s : Fin n) → (SO x s).C) :
    ∀ (l : List (Fin n)) (ms : List (Msg × Out)),
      (SP.eval H ask (Recursion.simOuter SO x xs c l ms)).1 = l.foldl (fun τ (s : Fin n) => if Recursion.goesOn τ then
        τ ++ (SP.eval H ask (spUnder (τ.map Prod.fst) ((SO x s).prog (xs.getD s []) (c s)))).1 else τ) ms
  | [], _ => by simp [Recursion.simOuter, SP.eval]
  | s :: l, ms => by
      by_cases hg : Recursion.goesOn ms = true
      · simp only [Recursion.simOuter, hg, ite_true, recSpBind_eq, spBind_eval, recSpUnder_eq, List.foldl_cons]
        exact simOuter_eval H ask SO x xs c l _
      · have hg' : Recursion.goesOn ms = false := by simpa using hg
        simp only [Recursion.simOuter, hg', List.foldl_cons, Bool.false_eq_true, ite_false, SP.eval]
        exact (foldl_stuck _ (fun τ i h => by simp [h]) l ms hg').symm

/-- A fold that keeps an invariant under which two steps agree is the other's fold. -/
theorem foldl_congr_inv {α ι : Type} (P : α → Prop) (f g : α → ι → α) (h : ∀ a i, P a → f a i = g a i ∧ P (f a i)) :
    ∀ (l : List ι) (a : α), P a → l.foldl f a = l.foldl g a
  | [], _, _ => rfl
  | i :: l, a, ha => by
      rw [List.foldl_cons, List.foldl_cons, ← (h a i ha).1]
      exact foldl_congr_inv P f g h l _ (h a i ha).2

/-! ## The nested coins -/

/-- **`iterSimG`'s nested coins** are the first part's and one per session. -/
noncomputable def iterCoins {WA WB : Type} (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB]
    [Fintype ΩB] [Nonempty ΩB] (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) (x : Msg) :
    (m : ℕ) → (iterSimG T₀ c NB ΩB Q S₀ CB progB go m).C x ≃ S₀.C x × ((s : Fin m) → CB s x)
  | 0 => (Equiv.prodUnique (S₀.C x) ((s : Fin 0) → CB s x)).symm
  | m + 1 => ((iterCoins T₀ c NB ΩB Q S₀ CB progB go x m).prodCongr (Equiv.refl (CB m x))).trans
      ((Equiv.prodAssoc _ _ _).trans
        ((Equiv.refl _).prodCongr ((Equiv.prodComm _ _).trans (Fin.snocEquiv fun s : Fin (m + 1) => CB s x))))

theorem iterCoins_zero {WA WB : Type} (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB]
    [Fintype ΩB] [Nonempty ΩB] (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) (x : Msg) (a : (iterSimG T₀ c NB ΩB Q S₀ CB progB go 0).C x) :
    (iterCoins T₀ c NB ΩB Q S₀ CB progB go x 0 a).1 = a := rfl

theorem iterCoins_succ {WA WB : Type} (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB]
    [Fintype ΩB] [Nonempty ΩB] (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) (x : Msg) (m : ℕ)
    (a : (iterSimG T₀ c NB ΩB Q S₀ CB progB go m).C x) (b' : CB m x) :
    iterCoins T₀ c NB ΩB Q S₀ CB progB go x (m + 1) (a, b') =
      ((iterCoins T₀ c NB ΩB Q S₀ CB progB go x m a).1,
        Fin.snoc (α := fun s : Fin (m + 1) => CB s x) (iterCoins T₀ c NB ΩB Q S₀ CB progB go x m a).2 b') := rfl

/-- **`ZKmalAt` at a simulator with the same transcripts** on coins in bijection. -/
theorem ZKmalAt.of_sim {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : ℕ}
    {P : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))}
    {ε γ : ℝ} (h : ZKmalAt H k P R x C prog ε γ) {C' : Type} [Fintype C'] (prog' : C' → SP (List (Msg × Out)))
    (e : C' ≃ C)
    (he : ∀ (ask : List Msg → Out) (c' : C'), (SP.eval H ask (prog' c')).1 = (SP.eval H ask (prog (e c'))).1) :
    ZKmalAt H k P R x C' prog' ε γ := by
  intro w hw Θ Vs θ hV Ev
  have hp : Pr (fun c' : C' => Ev (spOut H Vs θ x (prog' c'))) = Pr (fun c : C => Ev (spOut H Vs θ x (prog c))) := by
    simp only [spOut, he]
    exact Pr_equiv e fun c => Ev ((SP.eval H (fun as => ((Vs.next x as θ).eval H).1) (prog c)).1)
  rw [hp]
  exact h w hw Θ Vs θ hV Ev

end Claims.ZKMal

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open Claims.ZKMal (RunParts ZKmalAt SeqTape iterSimG iterProver)

variable (IV : Claims.Verifier Coin) (b : Brings)

/-- **The claim's first-part simulator**: `Sim`'s first two parts, the inner session's commit strings to the zero
digest under fresh salts, then the staging rounds with theirs. -/
def simPre {W : Type} (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W) : Claims.Simulator where
  C x := (Fin (I.V.k (Claims.Recursion.xI D x)) → Fin D.saltLen → UInt8) ×
    (Fin (D.kS x) → Fin (D.nC x) → Fin D.saltLen → UInt8)
  prog x c := Claims.Recursion.spBind (Claims.Recursion.simInner D (I.V.k (Claims.Recursion.xI D x)) c.1) fun ms =>
    Claims.Recursion.simStage D x (I.V.k (Claims.Recursion.xI D x)) c.2 ms

/-- **Where `Sim`'s first part goes on, it is `kPre x` rounds of messages**, at an inner protocol whose rounds at the
inner statement are the inner verifier's. -/
theorem simPre_goesOn {W : Type} (ivScope : Claims.Msg → Bool) (I : Claims.Recursion.Inner Coin W) (x : Claims.Msg)
    (hI : I.V.k (xIOf x) = IV.k (xIOf x)) (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (c : (simPre (deployRec IV ivScope b) I).C x)
    (hg : Claims.Recursion.goesOn (Claims.SP.eval H ask ((simPre (deployRec IV ivScope b) I).prog x c)).1 = true) :
    (Claims.SP.eval H ask ((simPre (deployRec IV ivScope b) I).prog x c)).1.length = kPre IV x ∧
      ∀ p ∈ (Claims.SP.eval H ask ((simPre (deployRec IV ivScope b) I).prog x c)).1, Claims.ZKMal.isMsg p.2 = true := by
  simp only [simPre, Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval, Claims.Recursion.simInner,
    Claims.Recursion.simStage] at hg ⊢
  obtain ⟨r₁, h₁, h₁g, -⟩ :=
    Claims.ZKMal.simRounds_eval H ask _ (I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x)) []
  obtain ⟨r₂, h₂, h₂g, h₂n⟩ := Claims.ZKMal.simRounds_eval H ask _ ((deployRec IV ivScope b).kS x)
    (Claims.SP.eval H ask (Claims.Recursion.simRounds _ (I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x)) [])).1
  rw [h₂] at hg ⊢
  rw [h₁] at hg h₂g h₂n ⊢
  simp only [List.nil_append] at hg h₂g h₂n ⊢
  obtain ⟨hl₂, hm₂⟩ := h₂g hg
  have hg₁ : Claims.Recursion.goesOn r₁ = true := by
    by_cases hr : r₂ = []
    · subst hr
      simpa using hg
    · exact h₂n hr
  obtain ⟨hl₁, hm₁⟩ := h₁g hg₁
  refine ⟨?_, fun p hp => ?_⟩
  · rw [List.length_append, hl₁, hl₂, xI_deployRec, hI]
    rfl
  · rcases List.mem_append.1 hp with hp | hp
    · exact hm₁ p hp
    · exact hm₂ p hp

/-- A transcript of messages read back from the prover's history is itself. -/
theorem asRounds_histOf {τ : List (Claims.Msg × Claims.Out)} (h : ∀ p ∈ τ, Claims.ZKMal.isMsg p.2 = true) :
    asRounds (Claims.ZKMal.histOf τ) = τ := by
  simp only [asRounds, Claims.ZKMal.histOf, List.map_map]
  refine (List.map_congr_left fun p hp => ?_).trans (List.map_id τ)
  obtain ⟨a, o⟩ := p
  cases o with
  | msg m => rfl
  | acc => exact absurd (h _ hp) (by simp [Claims.ZKMal.isMsg])
  | rej => exact absurd (h _ hp) (by simp [Claims.ZKMal.isMsg])

/-- **Past a first part of `kPre x` rounds of messages, the V\* statements `simM` reads are `Sim`'s.** -/
theorem xsOf_of_goesOn (x : Claims.Msg) (s : ℕ) (τ₀ r : List (Claims.Msg × Claims.Out))
    (hl : τ₀.length = kPre IV x) (hm : ∀ p ∈ τ₀, Claims.ZKMal.isMsg p.2 = true) :
    xsOf IV b x s (Claims.ZKMal.histOf (τ₀ ++ r)) = (vstar b.st x τ₀).getD s [] := by
  unfold xsOf
  rw [Claims.ZKMal.histOf_append, List.take_append_of_le_length (by rw [Claims.ZKMal.histOf_length, hl]),
    List.take_of_length_le (by rw [Claims.ZKMal.histOf_length, hl]), asRounds_histOf hm]

/-- One session's step of `simM`: while the run goes on, the session's outer simulator at the V\* statement the history
so far stages, under the transcript so far. -/
noncomputable def stepM (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg) (τ : List (Claims.Msg × Claims.Out)) (s : ℕ)
    (c : (SO x s).C) : List (Claims.Msg × Claims.Out) :=
  if Claims.Recursion.goesOn τ then τ ++ (Claims.SP.eval H ask (Claims.ZKMal.spUnder (τ.map Prod.fst)
    ((SO x s).prog (xsOf IV b x s (Claims.ZKMal.histOf τ)) c))).1 else τ

/-- **`simM`'s nested simulator's transcript**: the first part's, then one `stepM` per session, on the coins
`iterCoins` reads off the nested ones. -/
theorem iterSimG_parts_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (T₀ : SeqTape)
    (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (S₀ : Claims.Simulator)
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg) :
    ∀ (m : ℕ) (c : (iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω))) S₀ (fun j x => (SO x j).C)
        (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) m).C x),
      (Claims.SP.eval H ask ((iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω))) S₀
        (fun j x => (SO x j).C) (fun j x u c => (SO x j).prog (xsOf IV b x j u) c)
        (fun _ _ => Claims.Recursion.goesOn) m).prog x c)).1 =
      (List.finRange m).foldl (fun τ (s : Fin m) => stepM IV b H ask SO x τ s
          ((Claims.ZKMal.iterCoins T₀ 0 NB ΩB _ S₀ _ _ _ x m c).2 s))
        (Claims.SP.eval H ask (S₀.prog x (Claims.ZKMal.iterCoins T₀ 0 NB ΩB _ S₀ _ _ _ x m c).1)).1
  | 0, c => by simp only [List.finRange_zero, List.foldl_nil]; rfl
  | m + 1, (a, b') => by
      rw [Claims.ZKMal.iterCoins_succ, List.finRange_succ_last, List.foldl_append, List.foldl_map]
      simp only [Fin.snoc_castSucc, Fin.snoc_last, List.foldl_cons, List.foldl_nil]
      dsimp only [Fin.val_castSucc, Fin.val_last]
      rw [← iterSimG_parts_eval H ask T₀ NB ΩB S₀ SO x m a]
      simp only [iterSimG, Claims.ZKMal.Seq.simG, Claims.ZKMal.spBind_eval]
      unfold stepM
      split
      · simp only [Claims.ZKMal.spBind_eval, Claims.SP.eval]
        rfl
      · rfl

/-- **`Sim`'s coins are `simM`'s** at the claim's first-part simulator: the first part's, and one per V\* session. -/
noncomputable def simEquiv {W : Type} (ivScope : Claims.Msg → Bool) (I : Claims.Recursion.Inner Coin W)
    (T₀ : SeqTape) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (SO : Claims.Msg → ℕ → Claims.Recursion.OSim) (x : Claims.Msg) :
    (Claims.Recursion.Sim (deployRec IV ivScope b) I SO).C x ≃
      (simM IV b T₀ NB ΩB (simPre (deployRec IV ivScope b) I) SO).C x :=
  (Equiv.prodAssoc _ _ _).symm.trans (Claims.ZKMal.iterCoins T₀ 0 NB ΩB
    (RunParts.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω))) (simPre (deployRec IV ivScope b) I) (fun j x => (SO x j).C)
    (fun j x u c => (SO x j).prog (xsOf IV b x j u) c) (fun _ _ => Claims.Recursion.goesOn) x (nS x)).symm

/-- **`simM` at the claim's first-part simulator is `Sim`**: against every verifier, one transcript on the coins
`simEquiv` matches, where the inner protocol's rounds at the inner statement are the inner verifier's. -/
theorem sim_eval {W : Type} (ivScope : Claims.Msg → Bool) (I : Claims.Recursion.Inner Coin W) (T₀ : SeqTape)
    (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (SO : Claims.Msg → ℕ → Claims.Recursion.OSim)
    (x : Claims.Msg) (hI : I.V.k (xIOf x) = IV.k (xIOf x)) (H : Claims.Msg → Claims.Msg)
    (ask : List Claims.Msg → Claims.Out) (c : (Claims.Recursion.Sim (deployRec IV ivScope b) I SO).C x) :
    (Claims.SP.eval H ask ((Claims.Recursion.Sim (deployRec IV ivScope b) I SO).prog x c)).1 =
      (Claims.SP.eval H ask ((simM IV b T₀ NB ΩB (simPre (deployRec IV ivScope b) I) SO).prog x
        (simEquiv IV b ivScope I T₀ NB ΩB SO x c))).1 := by
  obtain ⟨c₁, c₂, cs⟩ := c
  have hrhs := iterSimG_parts_eval IV b H ask T₀ NB ΩB (simPre (deployRec IV ivScope b) I) SO x (nS x)
    (simEquiv IV b ivScope I T₀ NB ΩB SO x (c₁, c₂, cs))
  have hc : Claims.ZKMal.iterCoins T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (parts IV b (T₀.N × T₀.Ω)))
      (simPre (deployRec IV ivScope b) I) (fun j x => (SO x j).C) (fun j x u c => (SO x j).prog (xsOf IV b x j u) c)
      (fun _ _ => Claims.Recursion.goesOn) x (nS x) (simEquiv IV b ivScope I T₀ NB ΩB SO x (c₁, c₂, cs)) =
        ((c₁, c₂), cs) :=
    Equiv.apply_symm_apply _ ((c₁, c₂), cs)
  rw [hc] at hrhs
  refine Eq.trans ?_ hrhs.symm
  set τ₀ := (Claims.SP.eval H ask ((simPre (deployRec IV ivScope b) I).prog x (c₁, c₂))).1 with hτ₀
  have hlhs : (Claims.SP.eval H ask ((Claims.Recursion.Sim (deployRec IV ivScope b) I SO).prog x (c₁, c₂, cs))).1 =
      (Claims.SP.eval H ask (Claims.Recursion.simOuter SO x ((deployRec IV ivScope b).vstar x τ₀) cs
        (List.finRange ((deployRec IV ivScope b).nS x)) τ₀)).1 := by
    simp only [Claims.Recursion.Sim, simPre, Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval] at hτ₀ ⊢
    rw [hτ₀]
  rw [hlhs, Claims.ZKMal.simOuter_eval]
  refine Claims.ZKMal.foldl_congr_inv (fun τ => ∃ r, τ = τ₀ ++ r ∧ (r ≠ [] → Claims.Recursion.goesOn τ₀ = true))
    _ _ (fun τ s hP => ?_) _ τ₀ ⟨[], by simp, fun h => absurd rfl h⟩
  obtain ⟨r, rfl, hr⟩ := hP
  unfold stepM
  by_cases hg : Claims.Recursion.goesOn (τ₀ ++ r) = true
  · have hg₀ : Claims.Recursion.goesOn τ₀ = true := by
      by_cases hr0 : r = []
      · subst hr0
        simpa using hg
      · exact hr hr0
    obtain ⟨hl, hm⟩ := simPre_goesOn IV b ivScope I x hI H ask (c₁, c₂) hg₀
    rw [← hτ₀] at hl hm
    rw [ite_eq_left hg, ite_eq_left hg, xsOf_of_goesOn IV b x s τ₀ r hl hm]
    exact ⟨rfl, _, List.append_assoc _ _ _, fun _ => hg₀⟩
  · rw [ite_eq_right hg, ite_eq_right hg]
    exact ⟨rfl, r, rfl, hr⟩

/-- **`ZKmalAt` at `simM` is `ZKmalAt` at `Sim`**, at the claim's first-part simulator, where the inner protocol's
rounds at the inner statement are the inner verifier's. -/
theorem zkAt_sim {W : Type} (ivScope : Claims.Msg → Bool) (I : Claims.Recursion.Inner Coin W) (T₀ : SeqTape)
    (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (SO : Claims.Msg → ℕ → Claims.Recursion.OSim)
    (x : Claims.Msg) (hI : I.V.k (xIOf x) = IV.k (xIOf x)) {H : Claims.Msg → Claims.Msg} {N Ω WP : Type} [Fintype N]
    [Nonempty N] [Fintype Ω] [Nonempty Ω] {k : ℕ} {P : WP → Claims.Prover (N × Ω)} {R : WP → Prop} {ε γ : ℝ}
    (h : ZKmalAt H k P R x ((simM IV b T₀ NB ΩB (simPre (deployRec IV ivScope b) I) SO).C x)
      ((simM IV b T₀ NB ΩB (simPre (deployRec IV ivScope b) I) SO).prog x) ε γ) :
    ZKmalAt H k P R x ((Claims.Recursion.Sim (deployRec IV ivScope b) I SO).C x)
      ((Claims.Recursion.Sim (deployRec IV ivScope b) I SO).prog x) ε γ :=
  h.of_sim _ (simEquiv IV b ivScope I T₀ NB ΩB SO x) (sim_eval IV b ivScope I T₀ NB ΩB SO x hI H)

open scoped NNReal

/-- **`ZKBound` at the masked record**: `ZKBound`'s statement with the masked reference prover (`refPM` on the nested
tape read flat, `nS x` sessions at `x`, on statement-indexed tapes: `ZKmalRx`) at the masked run's rounds `kM` (the
compiled verifier's and one `Hello` per V\* session), the first part on `x`'s own tape `T₀ x`, which `pre x` reads as
the inner prover's tape and the staging salts. `Sim`, `εzk` and `γzk` are the claim's. -/
def ZKBoundM (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB]
    [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) : Prop :=
  ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
    Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
      Claims.ZKmalRx (fun x => ((T₀ x).iter 0 NB ΩB (nS x)).N) (fun x => ((T₀ x).iter 0 NB ΩB (nS x)).Ω) H
        (kM IV b) (fun w x => (refPM IV b (pre x) wit PS w).comap (RunParts.flatten (T₀ x) 0 NB ΩB (nS x)))
        (Claims.Recursion.R (deployRec IV ivScope b) I) (Claims.Recursion.Sim (deployRec IV ivScope b) I (SO t))
        (Claims.Recursion.εzk (deployRec IV ivScope b) I t δ₁) (Claims.Recursion.γzk (deployRec IV ivScope b) t)

theorem nums_εzk_nonneg (n : Claims.Recursion.Nums) (t : ℕ) (δ₁ : ℝ≥0) : 0 ≤ n.εzk t δ₁ := by
  unfold Claims.Recursion.Nums.εzk
  positivity

/-- **The recursion's zero knowledge at the masked record, by sequential composition**: `ZKBoundM` where the inner
protocol's rounds are the inner verifier's (`hI`), and at each attempt budget `t ≥ 1` and `δ₁` at which hm96 leaves
hide: the first part is `ZKmalAt` at `Sim`'s first two parts within `2δ₁` per commit string (`h₀`); each V\* session,
at every history that starts it, is `ZKmalAt` at `SO t` within its proved `--zk` error and `4t·3^t` on the sessions'
relation `RF` (`hS`); and there the witness `wit` gives is in `RF` (`hW`). -/
theorem zkBoundM_of (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB]
    [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (RF : Claims.Rel Claims.Flock.Wit) (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (h₀ : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        ZKmalAt H (kPre IV x) (preP IV b (pre x)) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
          ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
          (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0)
    (hS : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
        ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
          u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
          ZKmalAt H ((nums b.st x j).rounds + 1) PS (RF (xsOf IV b x j u)) (xsOf IV b x j u) (SO t x j).C
            ((SO t x j).prog (xsOf IV b x j u)) (((deployRec IV ivScope b).nums x j).εzk t δ₁) (4 * t * 3 ^ t))
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
      (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        RF (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)) :
    ZKBoundM IV b H ivScope I T₀ NB ΩB pre wit PS SO := by
  intro t ht δ₁ hδ x
  have h := zkAt IV b H (T₀ x) (pre x) wit PS (Claims.Recursion.R (deployRec IV ivScope b) I) RF
    (simPre (deployRec IV ivScope b) I) (SO t) x (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) (4 * t * 3 ^ t)
    (fun j => ((deployRec IV ivScope b).nums x j).εzk t δ₁) (h₀ t ht δ₁ hδ x) (hS t ht δ₁ hδ x) (hW x)
    (fun j _ => nums_εzk_nonneg _ t δ₁) (by positivity)
  refine (zkAt_sim IV b ivScope I (T₀ x) NB ΩB (SO t) x (hI x) h).mono (le_of_eq ?_) (le_of_eq ?_)
  · simp only [Claims.Recursion.εzk]
    rw [Fin.sum_univ_eq_sum_range (fun j => ((deployRec IV ivScope b).nums x j).εzk t δ₁)]
    simp only [xI_deployRec]
    simp only [deployRec]
    ring
  · simp only [Claims.Recursion.γzk, deployRec]

end RecDeploy
