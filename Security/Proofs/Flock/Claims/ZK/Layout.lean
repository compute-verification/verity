import Proofs.Flock.Claims.ZK.Compose
import Proofs.Flock.Claims.ZK.Congr

/-!
# A run in parts, as sessions in turn

`zkmalAt_iterG` composes sessions on nested tapes (`SeqTape.iter`, `forkTape`). A protocol's prover reads its tape as a
first part's and one session tape per slot. `RunParts` fixes the parts at each statement: a first part of `k₀ x` rounds,
then sessions back to back, session `j` of `n x j` rounds at the statement `xs x j h` with the witness `ws j w t₀ x h`,
read from the history `h` before it and the first part's tape `t₀`. `RunParts.prover` runs them on a flat tape (the
first part's and a session tape per slot), and `RunParts.flatten` reads the nested tape as the flat one, so the nested
prover `iterProver` at `RunParts.seq` is `RunParts.prover` on `flatten`'s tape (`RunParts.iterProver_flat`).
`zkmalAt_layout` is then `zkmalAt_iterG` at the flat prover: `m` sessions after the first part cost `Σ_{j < m} eB j`
and `(c + m)·g`.

* `ZKmalAt.congr`: `ZKmalAt` carries over to a prover with the same messages and more calls, at `γ ≥ 0`, and
  `ZKmalAt.mono` to larger errors.
* `zkmalAt_public`: a first part whose messages are a public function of the history is simulated by running that
  function, at `ε = γ = 0`.
* `RunParts.prover_pre`, `RunParts.prover_session`, `RunParts.exists_slot`: the flat prover in the first part and in each
  session, and the slot each round of the sessions is in.
* `RunParts.reach_length`, `RunParts.relOnAt_of`: a unit an honest run selects is a history of the session's start, so a
  session's zero knowledge and relation are needed only at those.
* `RunParts.iterSimG_seq_congr`, `RunParts.start_congr`: the simulator and the sessions' starts read no witness, so a
  simulator built at the parts alone is the composition's at every witness map.
* `progOf`: a session simulator's program on a slot's own coin type, through a bijection with its coins at the
  slot's statement (`progOf_of`).
-/

namespace Claims.ZKMal

open Claims

/-! ## One statement's `ZKmalAt` at another prover, relation or error -/

/-- **`ZKmalAt` carries over to a prover with the same messages and more calls** where the relation holds, at `γ ≥ 0`:
the transcripts are the same, and `Fork` finds at least as much. -/
theorem ZKmalAt.congr {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : ℕ}
    {P P' : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))}
    {ε γ : ℝ} (h : ZKmalAt H k P R x C prog ε γ) (hγ : 0 ≤ γ)
    (hv : ∀ w, R w → ∀ (hist : List (Msg × Msg)) ω, hist.length < k →
      (((P w).next x hist ω).eval H).1 = (((P' w).next x hist ω).eval H).1)
    (hl : ∀ w, R w → ∀ (hist : List (Msg × Msg)) ω, hist.length < k →
      (((P w).next x hist ω).eval H).2 ⊆ (((P' w).next x hist ω).eval H).2) :
    ZKmalAt H k P' R x C prog ε γ := by
  intro w hR Θ Vs θ hk Ev
  have e : ∀ ω, (runStar H (P' w) Vs x ω θ).msgs = (runStar H (P w) Vs x ω θ).msgs := fun ω =>
    (runStar_msgs_congr H (P w) (P' w) Vs x ω θ fun hist hh => hv w hR hist ω (hk ▸ hh)).symm
  simp only [e]
  exact (h w hR Θ Vs θ hk Ev).trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (fork_pr_mono H (P w) (P' w) Vs θ x (fun hist ω hh => hv w hR hist ω (hk ▸ hh))
      (fun hist ω hh => hl w hR hist ω (hk ▸ hh))) hγ))

/-- `ZKmalAt` at larger errors. -/
theorem ZKmalAt.mono {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : ℕ}
    {P : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))}
    {ε γ ε' γ' : ℝ} (h : ZKmalAt H k P R x C prog ε γ) (hε : ε ≤ ε') (hγ : γ ≤ γ') :
    ZKmalAt H k P R x C prog ε' γ' := fun w hR Θ Vs θ hk Ev =>
  (h w hR Θ Vs θ hk Ev).trans (add_le_add hε (mul_le_mul_of_nonneg_right hγ (Finder.pr_nonneg H _)))

/-- `ZKmalAt` on a smaller relation. -/
theorem ZKmalAt.of_rel {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : ℕ}
    {P : W → Prover (N × Ω)} {R R' : W → Prop} {x : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))}
    {ε γ : ℝ} (h : ZKmalAt H k P R x C prog ε γ) (hR : ∀ w, R' w → R w) : ZKmalAt H k P R' x C prog ε γ :=
  fun w hw => h w (hR w hw)

/-- **A part whose messages are a public function of the history is simulated exactly**: at `x`, wherever the
relation holds, the prover's message at each history within `k` rounds is `f`'s there, so the simulator that runs `f`
(`spRun`), on no coins, has the real transcript at every `V*`, at `ε = γ = 0`. -/
theorem zkmalAt_public {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (k : ℕ)
    (P : W → Prover (N × Ω)) (R : W → Prop) (x : Msg) (f : List (Msg × Msg) → OC Msg)
    (hP : ∀ w, R w → ∀ (hist : List (Msg × Msg)) ω, hist.length < k →
      (((P w).next x hist ω).eval H).1 = ((f hist).eval H).1) :
    ZKmalAt H k P R x Unit (fun _ => spRun f k []) 0 0 := by
  classical
  intro w hw Θ Vs θ hk Ev
  have e : ∀ ω, (runStar H (P w) Vs x ω θ).msgs = spOut H Vs θ x (spRun f k []) := fun ω => by
    rw [runStar_msgs, spOut, spRun_eval, hk]
    exact interact_congr_lt _ _ _ k (fun hist hh => hP w hw hist ω hh) k [] (by simp)
  simp only [e, Pr_const, sub_self, abs_zero, zero_mul, add_zero, le_refl]

/-! ## The layout -/

/-- **A run in parts** at each statement: a first part of `k₀ x` rounds, then sessions back to back, session `j` of
`n x j` rounds at the statement `xs x j h`, with the witness `ws j w t₀ x h`, from the history `h` before it and the
first part's tape `t₀`. -/
structure RunParts (W WS T : Type) where
  k₀ : Msg → ℕ
  n : Msg → ℕ → ℕ
  xs : Msg → ℕ → List (Msg × Msg) → Msg
  ws : ℕ → W → T → Msg → List (Msg × Msg) → WS

namespace RunParts

variable {W WS T TS : Type}

/-- Where session `j` starts: after the first part and the sessions before `j`. -/
def start (L : RunParts W WS T) (x : Msg) : ℕ → ℕ
  | 0 => L.k₀ x
  | j + 1 => L.start x j + L.n x j

/-- Where the sessions start reads only the rounds. -/
theorem start_congr {L L' : RunParts W WS T} (hk : L.k₀ = L'.k₀) (hn : L.n = L'.n) : L.start = L'.start := by
  funext x j
  induction j with
  | zero => exact congrFun hk x
  | succ j ih => simp only [start, ih, hn]

theorem start_mono (L : RunParts W WS T) (x : Msg) {i j : ℕ} (h : i ≤ j) : L.start x i ≤ L.start x j := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (Nat.le_add_right _ _)

/-- **Each round of the first `m` sessions is in one slot**: at a history from the first part's end to the `m`-th
session's, some session `j < m` starts at or before it and ends after it. -/
theorem exists_slot (L : RunParts W WS T) (x : Msg) (len : ℕ) (h0 : L.k₀ x ≤ len) :
    ∀ m, len < L.start x m → ∃ j < m, L.start x j ≤ len ∧ len < L.start x (j + 1)
  | 0, h => absurd h0 (Nat.not_le.2 h)
  | m + 1, h => by
      by_cases hm : len < L.start x m
      · obtain ⟨j, hj, h1, h2⟩ := L.exists_slot x len h0 m hm
        exact ⟨j, by omega, h1, h2⟩
      · exact ⟨m, by omega, by omega, h⟩

/-- **The parts on a flat tape**: the first part's prover on its tape, then session `j` on slot `j`'s tape, at the
statement and witness the history before it gives, for the first `m` sessions. -/
def prover (L : RunParts W WS T) (P₀ : W → Prover T) (PS : WS → Prover TS) : ℕ → W → Prover (T × (ℕ → TS))
  | 0 => fun w => (P₀ w).comap Prod.fst
  | j + 1 => fun w => ⟨fun x hist ω => if hist.length < L.start x j then (L.prover P₀ PS j w).next x hist ω else
      (PS (L.ws j w ω.1 x (hist.take (L.start x j)))).next (L.xs x j (hist.take (L.start x j)))
        (hist.drop (L.start x j)) (ω.2 j)⟩

/-- The first `m` sessions read no slot after the `m`-th. -/
theorem prover_congr (L : RunParts W WS T) (P₀ : W → Prover T) (PS : WS → Prover TS) :
    ∀ (m : ℕ) (w : W) (x : Msg) (hist : List (Msg × Msg)) (ω ω' : T × (ℕ → TS)), ω.1 = ω'.1 →
      (∀ i < m, ω.2 i = ω'.2 i) → (L.prover P₀ PS m w).next x hist ω = (L.prover P₀ PS m w).next x hist ω'
  | 0, w, x, hist, ω, ω', h1, _ => by
      show (P₀ w).next x hist ω.1 = (P₀ w).next x hist ω'.1
      rw [h1]
  | m + 1, w, x, hist, ω, ω', h1, h2 => by
      simp only [prover]
      split
      · exact L.prover_congr P₀ PS m w x hist ω ω' h1 fun i hi => h2 i (by omega)
      · rw [h1, h2 m (by omega)]

/-- **In the first part, the flat prover is the first part's.** -/
theorem prover_pre (L : RunParts W WS T) (P₀ : W → Prover T) (PS : WS → Prover TS) :
    ∀ (m : ℕ) (w : W) (x : Msg) (hist : List (Msg × Msg)) (ω : T × (ℕ → TS)), hist.length < L.k₀ x →
      (L.prover P₀ PS m w).next x hist ω = (P₀ w).next x hist ω.1
  | 0, _, _, _, _, _ => rfl
  | m + 1, w, x, hist, ω, h => by
      have hm : hist.length < L.start x m := lt_of_lt_of_le h (L.start_mono x (Nat.zero_le m))
      simp only [prover, hm, ite_true]
      exact L.prover_pre P₀ PS m w x hist ω h

/-- **In session `j < m`, the flat prover is the session's** at the statement and witness the history before it gives,
on the session's own history and slot `j`'s tape. -/
theorem prover_session (L : RunParts W WS T) (P₀ : W → Prover T) (PS : WS → Prover TS) :
    ∀ (m j : ℕ), j < m → ∀ (w : W) (x : Msg) (hist : List (Msg × Msg)) (ω : T × (ℕ → TS)),
      L.start x j ≤ hist.length → hist.length < L.start x (j + 1) →
      (L.prover P₀ PS m w).next x hist ω =
        (PS (L.ws j w ω.1 x (hist.take (L.start x j)))).next (L.xs x j (hist.take (L.start x j)))
          (hist.drop (L.start x j)) (ω.2 j)
  | 0, _, hj, _, _, _, _, _, _ => absurd hj (Nat.not_lt_zero _)
  | m + 1, j, hj, w, x, hist, ω, h1, h2 => by
      by_cases hjm : j = m
      · subst hjm
        have hn : ¬ hist.length < L.start x j := by omega
        simp only [prover, hn, ite_false]
      · have hm : hist.length < L.start x m := lt_of_lt_of_le h2 (L.start_mono x (by omega))
        simp only [prover, hm, ite_true]
        exact L.prover_session P₀ PS m j (by omega) w x hist ω h1 h2

/-- **The nested tape read flat**: the first part's tape, and session `j`'s at slot `j` for `j < m`; the other slots
are arbitrary. -/
noncomputable def flatten (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] : (m : ℕ) → T₀.at c NB ΩB m → (T₀.N × T₀.Ω) × (ℕ → NB × ΩB)
  | 0 => fun s => (s, fun _ => Classical.arbitrary _)
  | m + 1 => fun s => ((flatten T₀ c NB ΩB m (forkTape (c + m) s).1).1, fun i =>
      if i < m then (flatten T₀ c NB ΩB m (forkTape (c + m) s).1).2 i else (forkTape (c + m) s).2)

/-- **Session `j` after the parts before it**, as `Seq`: the first part and the sessions before `j` as `A`, on the
nested tape, then session `j` at the history before it, with the witness from the first part's tape `flatten` reads. -/
noncomputable def seq (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (L : RunParts W WS (T₀.N × T₀.Ω)) (j : ℕ) : Seq W WS (T₀.at c NB ΩB j) where
  kA x := L.start x j
  U _ := List (Msg × Msg)
  sel _ h := h
  xB x u := L.xs x j u
  wB w t x u := L.ws j w (flatten T₀ c NB ΩB j t).1 x u

/-- **The simulator reads no witness**: `iterSimG` at two layouts with the same rounds and statements is one
simulator. -/
theorem iterSimG_seq_congr (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] (L L' : RunParts W WS (T₀.N × T₀.Ω)) (hk : L.k₀ = L'.k₀) (hn : L.n = L'.n) (hx : L.xs = L'.xs)
    (S₀ : Simulator) (CB : ℕ → Msg → Type) [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → List (Msg × Msg) → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) (m : ℕ) :
    iterSimG T₀ c NB ΩB (seq T₀ c NB ΩB L) S₀ CB progB go m =
      iterSimG T₀ c NB ΩB (seq T₀ c NB ΩB L') S₀ CB progB go m := by
  obtain ⟨k₀, n, xs, ws⟩ := L
  obtain ⟨k₀', n', xs', ws'⟩ := L'
  simp only at hk hn hx
  subst hk hn hx
  induction m with
  | zero => rfl
  | succ m ih =>
      simp only [iterSimG]
      rw [ih]
      rfl

/-- **The nested prover is the flat one on `flatten`'s tape.** -/
theorem iterProver_flat (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] (L : RunParts W WS (T₀.N × T₀.Ω)) (P₀ : W → Prover (T₀.N × T₀.Ω)) (PS : WS → Prover (NB × ΩB)) :
    ∀ (m : ℕ) (w : W) (x : Msg) (hist : List (Msg × Msg)) (s : T₀.at c NB ΩB m),
      (iterProver T₀ c NB ΩB (seq T₀ c NB ΩB L) P₀ (fun _ => PS) m w).next x hist s =
        (L.prover P₀ PS m w).next x hist (flatten T₀ c NB ΩB m s)
  | 0, _, _, _, _ => rfl
  | m + 1, w, x, hist, s => by
      have ih := iterProver_flat T₀ c NB ΩB L P₀ PS m w x hist (forkTape (c + m) s).1
      by_cases h : hist.length < L.start x m
      · have e1 : (iterProver T₀ c NB ΩB (seq T₀ c NB ΩB L) P₀ (fun _ => PS) (m + 1) w).next x hist s =
            (iterProver T₀ c NB ΩB (seq T₀ c NB ΩB L) P₀ (fun _ => PS) m w).next x hist (forkTape (c + m) s).1 := by
          simp only [iterProver, Prover.comap, Seq.prover, seq, h, ite_true]
        have e2 : (L.prover P₀ PS (m + 1) w).next x hist (flatten T₀ c NB ΩB (m + 1) s) =
            (L.prover P₀ PS m w).next x hist (flatten T₀ c NB ΩB (m + 1) s) := by
          simp only [prover, h, ite_true]
        rw [e1, e2, ih]
        refine L.prover_congr P₀ PS m w x hist _ _ rfl fun i hi => ?_
        simp only [flatten, hi, ite_true]
      · simp only [iterProver, Prover.comap, Seq.prover, seq, prover, h, ite_false, flatten, lt_irrefl]

/-- **A unit an honest run selects is a history of the session's start.** -/
theorem reach_length (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (L : RunParts W WS (T₀.N × T₀.Ω)) (H : Msg → Msg) {j : ℕ} (P : W → Prover (T₀.at c NB ΩB j)) (RA : Rel W) (x : Msg)
    {u : List (Msg × Msg)} (h : (seq T₀ c NB ΩB L j).Reach H P RA x u) : u.length = L.start x j := by
  obtain ⟨w, -, Θ, Vs, θ, ωA, hgo, rfl⟩ := h
  show (histOf _).length = _
  rw [histOf_length]
  exact goesOn_length hgo

/-- **Session `j`'s relation where it holds at every history of its start**, at every first part's tape. -/
theorem relOnAt_of (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (L : RunParts W WS (T₀.N × T₀.Ω)) (H : Msg → Msg) {j : ℕ} (P : W → Prover (T₀.at c NB ΩB j)) (RA : Rel W)
    (RB : Rel WS) (x : Msg)
    (h : ∀ w, RA x w → ∀ t₀ (u : List (Msg × Msg)), u.length = L.start x j → RB (L.xs x j u) (L.ws j w t₀ x u)) :
    (seq T₀ c NB ΩB L j).RelOnAt H P RA RB x := by
  intro w hw Θ Vs θ ωA hgo
  refine h w hw _ _ ?_
  show (histOf _).length = _
  rw [histOf_length]
  exact goesOn_length hgo

end RunParts

/-- **`m` sessions after a first part, on the flat tape, at one statement**: `zkmalAt_iterG` at `RunParts.seq`, read on
the flat prover through `flatten`. The first part `ZKmalAt` with `ε₀` and `γ₀ ≤ c·g`, and session `j < m` `ZKmalAt`
with `eB j` and `g` on coins `CB j x` at every history of its start an honest run reaches (`Reach`), where its
witness is in its relation (`RelOnAt`, from `relOnAt_of`) and `go j` agrees with `goesOn` at the honest transcripts:
the parts in turn are `ZKmalAt` with `ε₀ + Σ_{j < m} eB j` and `(c + m)·g`. -/
theorem zkmalAt_layout {W WS NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (T₀ : SeqTape)
    (c : ℕ) (L : RunParts W WS (T₀.N × T₀.Ω)) (H : Msg → Msg) (x : Msg) (m : ℕ) (P₀ : W → Prover (T₀.N × T₀.Ω))
    (PS : WS → Prover (NB × ΩB)) (RA : Rel W) (RB : Rel WS) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → List (Msg × Msg) → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) (ε₀ γ₀ g : ℝ) (eB : ℕ → ℝ)
    (h₀ : ZKmalAt H (L.k₀ x) P₀ (RA x) x (S₀.C x) (S₀.prog x) ε₀ γ₀)
    (hB : ∀ j < m, ∀ u, (RunParts.seq T₀ c NB ΩB L j).Reach H
        (iterProver T₀ c NB ΩB (RunParts.seq T₀ c NB ΩB L) P₀ (fun _ => PS) j) RA x u →
      ZKmalAt H (L.n x j) PS (RB (L.xs x j u)) (L.xs x j u) (CB j x) (progB j x u) (eB j) g)
    (hR : ∀ j < m, (RunParts.seq T₀ c NB ΩB L j).RelOnAt H
      (iterProver T₀ c NB ΩB (RunParts.seq T₀ c NB ΩB L) P₀ (fun _ => PS) j) RA RB x)
    (hgo : ∀ j < m, ∀ w, RA x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (ωA : T₀.at c NB ΩB j),
      go j x (runStar H (iterProver T₀ c NB ΩB (RunParts.seq T₀ c NB ΩB L) P₀ (fun _ => PS) j w)
        (Vs.head fun x => L.start x j) x ωA θ).msgs =
      goesOn (L.start x j) (runStar H (iterProver T₀ c NB ΩB (RunParts.seq T₀ c NB ΩB L) P₀ (fun _ => PS) j w)
        (Vs.head fun x => L.start x j) x ωA θ).msgs)
    (hγ₀ : γ₀ ≤ c * g) (he0 : ∀ j < m, 0 ≤ eB j) (hg : 0 ≤ g) :
    ZKmalAt H (L.start x m) (fun w => (L.prover P₀ PS m w).comap (RunParts.flatten T₀ c NB ΩB m)) (RA x) x
      ((iterSimG T₀ c NB ΩB (RunParts.seq T₀ c NB ΩB L) S₀ CB progB go m).C x)
      ((iterSimG T₀ c NB ΩB (RunParts.seq T₀ c NB ΩB L) S₀ CB progB go m).prog x)
      (ε₀ + ∑ j ∈ Finset.range m, eB j) ((c + m) * g) := by
  have h := zkmalAt_iterG T₀ c (RunParts.seq T₀ c NB ΩB L) H x (fun j x => L.n x j) P₀ (fun _ => PS) RA (fun _ => RB)
    S₀ CB progB go ε₀ γ₀ g eB m h₀ hB hR hgo (fun _ _ => rfl) hγ₀ he0 hg
  exact h.of_equiv _ (Equiv.refl _) (Equiv.refl _) fun w hist ω =>
    (RunParts.iterProver_flat T₀ c NB ΩB L P₀ PS m w x hist ω).symm

open Classical in
/-- **A simulator's program on coins of another type**: `S`'s at `xS` through a bijection, where there is one. -/
noncomputable def progOf (S : Simulator) (C : Type) (xS : Msg) (c : C) : SP (List (Msg × Out)) :=
  if h : Nonempty (C ≃ S.C xS) then S.prog xS (h.some c) else .ret []

theorem progOf_of {S : Simulator} {C : Type} {xS : Msg} (h : Nonempty (C ≃ S.C xS)) :
    progOf S C xS = fun c => S.prog xS (h.some c) := by
  funext c
  unfold progOf
  split
  · rfl
  · contradiction

end Claims.ZKMal
