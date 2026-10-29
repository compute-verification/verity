import Proofs.Flock.Claims.ZK.Layout
import Proofs.Flock.Claims.ZK.Cut

/-!
# A run in parts with a public stop

A prover with a public stop reads a test `s` of the history at the start of each of the first part's rounds and
after its last (`fired`); once it holds, the prover sends the empty message every round left (`stopP`). The stop reads
only what the verifier has seen, so it leaks nothing: the run with the stop is the run without it cut where the stop
fires (`interact_censor`), a function of the transcript and the verifier alone, and a simulator cuts its own
transcript there (`censorSim`).

* `ZKmalAt.post`: `ZKmalAt` at `γ = 0` carries over to a prover and a program whose transcripts are one function of
  the first ones' at each verifier and tape. So a first part that rewinds nothing is `ZKmalAt` with the stop at its
  simulator cut, at the same error (`ZKmalAt.censor`).
* `RunParts.stopped`: the parts with the stop. A session at a history where the stop has fired is at the empty
  statement with no witness, and its prover is silent (`stopPS`), simulated exactly (`zkmalAt_silent`); elsewhere it
  is at its statement tagged `1 ::`, with its witness, where its zero knowledge is the untagged statement's
  (`ZKmalAt.tag`). The layout's prover is the prover with the stop (`RunParts.prover_stopped`).
* **`zkmalAt_layout_stop`**: the parts in turn with the stop are `ZKmalAt` at the stopped composition, within the
  first part's error and the sessions', where the sessions are zero knowledge at the histories where the stop hasn't
  fired and their relation holds at the honest first parts the stop leaves live. The collision term is `Fork`'s of the
  prover with the stop.
-/

namespace Claims.ZKMal

open Claims Claims.ZKServer

/-! ## A public stop -/

/-- **The stop has fired** at the history `h`: `s` holds at one of `h`'s prefixes of at most `k` rounds. -/
def fired (s : List (Msg × Msg) → Bool) (k : ℕ) (h : List (Msg × Msg)) : Bool :=
  (List.range (min h.length k + 1)).any fun i => s (h.take i)

/-- Past the first `k` rounds, the stop reads only them. -/
theorem fired_take {s : List (Msg × Msg) → Bool} {k j : ℕ} (h : List (Msg × Msg)) (hj : k ≤ j) :
    fired s k (h.take j) = fired s k h := by
  unfold fired
  have hl : min (h.take j).length k = min h.length k := by rw [List.length_take]; omega
  rw [hl]
  apply Bool.eq_iff_iff.2
  simp only [List.any_eq_true, List.mem_range]
  constructor
  · rintro ⟨i, hi, h'⟩
    refine ⟨i, hi, ?_⟩
    rwa [List.take_take, Nat.min_eq_left (by omega)] at h'
  · rintro ⟨i, hi, h'⟩
    refine ⟨i, hi, ?_⟩
    rwa [List.take_take, Nat.min_eq_left (by omega)]

/-- A stop fired at a prefix has fired. -/
theorem fired_of_take {s : List (Msg × Msg) → Bool} {k : ℕ} (h : List (Msg × Msg)) (j : ℕ)
    (hf : fired s k (h.take j) = true) : fired s k h = true := by
  unfold fired at hf ⊢
  simp only [List.any_eq_true, List.mem_range, List.length_take] at hf ⊢
  obtain ⟨i, hi, h'⟩ := hf
  refine ⟨i, by omega, ?_⟩
  rwa [List.take_take, Nat.min_eq_left (by omega)] at h'

/-- **The prover with the stop**: `P`, except that it sends the empty message once `s x` has fired within the first
`k x` rounds. -/
def stopP {Ω : Type} (P : Prover Ω) (s : Msg → List (Msg × Msg) → Bool) (k : Msg → ℕ) : Prover Ω :=
  ⟨fun x h ω => if fired (s x) (k x) h then .ret [] else P.next x h ω⟩

/-- The message function `pm` with the stop. -/
def censor (s : List (Msg × Msg) → Bool) (k : ℕ) (pm : List (Msg × Msg) → Msg) (h : List (Msg × Msg)) : Msg :=
  if fired s k h then [] else pm h

open Classical in
/-- **Where the stop fires on the run `τ`**: the first round `i ≤ k` at whose history `s` holds, else `n`. -/
noncomputable def stopIdx (s : List (Msg × Msg) → Bool) (k n : ℕ) (τ : List (Msg × Out)) : ℕ :=
  if h : ∃ i, i ≤ k ∧ s (histOf (τ.take i)) = true then Nat.find h else n

theorem stopIdx_min {s : List (Msg × Msg) → Bool} {k n : ℕ} {τ : List (Msg × Out)} {i : ℕ}
    (hi : i < stopIdx s k n τ) (hk : i ≤ k) : s (histOf (τ.take i)) = false := by
  classical
  unfold stopIdx at hi
  split at hi
  · rename_i h
    by_contra hne
    exact Nat.find_min h hi ⟨hk, by simpa using hne⟩
  · rename_i h
    by_contra hne
    exact h ⟨i, hk, by simpa using hne⟩

theorem stopIdx_spec {s : List (Msg × Msg) → Bool} {k n : ℕ} {τ : List (Msg × Out)}
    (hn : stopIdx s k n τ < n) : s (histOf (τ.take (stopIdx s k n τ))) = true := by
  classical
  unfold stopIdx at hn ⊢
  split at hn
  · rename_i h
    rw [dif_pos h]
    exact (Nat.find_spec h).2
  · omega

theorem stopIdx_eq {s : List (Msg × Msg) → Bool} {k n : ℕ} {τ : List (Msg × Out)} (hk : k < stopIdx s k n τ) :
    stopIdx s k n τ = n := by
  classical
  unfold stopIdx at hk ⊢
  split at hk
  · rename_i h
    exact absurd (Nat.find_spec h).1 (by omega)
  · rename_i h
    rw [dif_neg h]

/-- **The run with the stop is the run cut where the stop fires**: against `ask`, the prover with the stop at most
`k` rounds in sends the run's messages before the first round at whose history the stop holds, then the empty
message. -/
theorem interact_censor (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (s : List (Msg × Msg) → Bool)
    (k n : ℕ) :
    interact ask (censor s k pm) n [] =
      cutT ask [] (stopIdx s k n (interact ask pm n [])) n (interact ask pm n []) := by
  rw [← interact_stopAt]
  set T := stopIdx s k n (interact ask pm n [])
  refine interact_congr ask (stopAt pm T []) (censor s k pm) n [] fun i hi => ?_
  simp only [List.nil_append]
  have hl : (histOf ((interact ask (stopAt pm T []) n []).take i)).length = i := by
    rw [length_histOf_take]; omega
  have htk := stopAt_take ask pm [] T n
  have hpre : ∀ j ≤ i, j ≤ T →
      (histOf ((interact ask (stopAt pm T []) n []).take i)).take j = histOf ((interact ask pm n []).take j) := by
    intro j hj hjT
    have e1 : (interact ask (stopAt pm T []) n []).take j = ((interact ask (stopAt pm T []) n []).take T).take j := by
      rw [List.take_take, Nat.min_eq_left hjT]
    rw [histOf_take_take _ hj, e1, htk, List.take_take, Nat.min_eq_left hjT]
  by_cases hit : i < T
  · rw [stopAt_of_lt (by rw [hl]; exact hit)]
    unfold censor
    rw [if_neg]
    unfold fired
    rw [Bool.not_eq_true, List.any_eq_false]
    intro j hj
    have hj' := List.mem_range.1 hj
    rw [hl] at hj'
    have hji : j ≤ i := by omega
    have hjk : j ≤ k := by omega
    rw [hpre j hji (by omega), stopIdx_min (s := s) (n := n) (τ := interact ask pm n []) (by omega) hjk]
    exact Bool.false_ne_true
  · rw [stopAt_of_ge (by rw [hl]; exact hit)]
    have hle := interact_length_le ask (stopAt pm T []) n []
    have hTk : T ≤ k := by
      by_contra hTk
      have := stopIdx_eq (Nat.lt_of_not_le hTk)
      omega
    unfold censor
    rw [if_pos]
    unfold fired
    rw [List.any_eq_true]
    have hTi : T ≤ i := Nat.le_of_not_lt hit
    have hTm : T < min (histOf ((interact ask (stopAt pm T []) n []).take i)).length k + 1 := by
      rw [hl]; omega
    refine ⟨T, List.mem_range.2 hTm, ?_⟩
    rw [hpre T (by omega) le_rfl]
    exact stopIdx_spec (lt_of_le_of_lt hTi (lt_of_lt_of_le hi hle))

/-! ## The first part with the stop -/

/-- **`ZKmalAt` through a public post-processing**, at `γ = 0`: where at every verifier and tape `f` takes the
transcripts of `P` and `prog` to those of `P'` and `prog'`. -/
theorem ZKmalAt.post {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : ℕ}
    {P P' : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} {C : Type} [Fintype C]
    {prog prog' : C → SP (List (Msg × Out))} {ε : ℝ} (h : ZKmalAt H k P R x C prog ε 0)
    (f : (Θ : Type) → VStar Θ → Θ → List (Msg × Out) → List (Msg × Out))
    (hP : ∀ w, R w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k → ∀ ω,
      (runStar H (P' w) Vs x ω θ).msgs = f Θ Vs θ (runStar H (P w) Vs x ω θ).msgs)
    (hS : ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) c, spOut H Vs θ x (prog' c) = f Θ Vs θ (spOut H Vs θ x (prog c))) :
    ZKmalAt H k P' R x C prog' ε 0 := by
  intro w hw Θ Vs θ hk Ev
  have e1 : (fun ω => Ev (runStar H (P' w) Vs x ω θ).msgs) =
      fun ω => (fun τ => Ev (f Θ Vs θ τ)) (runStar H (P w) Vs x ω θ).msgs := by
    funext ω; rw [hP w hw Θ Vs θ hk ω]
  have e2 : (fun c => Ev (spOut H Vs θ x (prog' c))) = fun c => (fun τ => Ev (f Θ Vs θ τ)) (spOut H Vs θ x (prog c)) := by
    funext c; rw [hS Θ Vs θ c]
  rw [e1, e2]
  have := h w hw Θ Vs θ hk fun τ => Ev (f Θ Vs θ τ)
  simpa using this

/-- **A simulator with the stop**: `S`'s transcript at `x`, cut where the stop fires within `k x` rounds, the rounds
from there asked of the verifier with the empty message. -/
noncomputable def censorSim (S : Simulator) (s : Msg → List (Msg × Msg) → Bool) (k : Msg → ℕ) : Simulator where
  C := S.C
  prog x c := spBind (S.prog x c) fun τ =>
    spRun (fun h => .ret (cutMsg [] (stopIdx (s x) (k x) (k x) τ) τ h)) (k x) []

theorem censorSim_out {Θ : Type} (H : Msg → Msg) (S : Simulator) (s : Msg → List (Msg × Msg) → Bool)
    (k : Msg → ℕ) (Vs : VStar Θ) (θ : Θ) (x : Msg) (c : S.C x) :
    spOut H Vs θ x ((censorSim S s k).prog x c) =
      cutT (fun as => ((Vs.next x as θ).eval H).1) [] (stopIdx (s x) (k x) (k x) (spOut H Vs θ x (S.prog x c)))
        (k x) (spOut H Vs θ x (S.prog x c)) := by
  simp only [spOut, censorSim, spBind_eval, spRun_eval]
  rfl

/-- **A first part that rewinds nothing is zero knowledge with the stop** at its simulator cut where the stop fires,
at the same error. -/
theorem ZKmalAt.censor {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {P : W → Prover (N × Ω)} {R : W → Prop} {x : Msg} (S : Simulator) {ε : ℝ} (s : Msg → List (Msg × Msg) → Bool)
    (k : Msg → ℕ) (h : ZKmalAt H (k x) P R x (S.C x) (S.prog x) ε 0) :
    ZKmalAt H (k x) (fun w => stopP (P w) s k) R x (S.C x) ((censorSim S s k).prog x) ε 0 := by
  refine h.post (fun Θ Vs θ τ => cutT (fun as => ((Vs.next x as θ).eval H).1) [] (stopIdx (s x) (k x) (k x) τ)
    (k x) τ) (fun w _ Θ Vs θ hk ω => ?_) (fun Θ Vs θ c => censorSim_out H S s k Vs θ x c)
  rw [runStar_msgs, runStar_msgs, hk]
  rw [← interact_censor]
  congr 1
  funext hist
  show (((if fired (s x) (k x) hist = true then OC.ret [] else (P w).next x hist ω).eval H).1) =
    (if fired (s x) (k x) hist = true then [] else ((P w).next x hist ω).eval H |>.1)
  by_cases hf : fired (s x) (k x) hist = true
  · rw [if_pos hf, if_pos hf]
    rfl
  · rw [if_neg hf, if_neg hf]

/-! ## The sessions with the stop -/

/-- **The session prover with the stop**: silent at no witness; at a witness, `PS`'s at the statement untagged. -/
def stopPS {WS TS : Type} (PS : WS → Prover TS) : Option WS → Prover TS
  | none => ⟨fun _ _ _ => .ret []⟩
  | some w => ⟨fun x h ω => (PS w).next x.tail h ω⟩

/-- **The sessions' relation with the stop**: no witness at the empty statement; a witness of `RB` at a statement
tagged `1 ::`. -/
def stopRel {WS : Type} (RB : Rel WS) : Rel (Option WS)
  | xs, none => xs = []
  | xs, some w => ∃ xs₀, xs = 1 :: xs₀ ∧ RB xs₀ w

/-- The run reads the prover only at the statement it is at and its tape. -/
theorem stepWith_retag {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω') (x x' : Msg) (ω : Ω)
    (ω' : Ω') (vnext : List Msg → OC Out) (h : ∀ hist, P'.next x' hist ω' = P.next x hist ω) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      stepWith H P' x' ω' vnext n hist as t = stepWith H P x ω vnext n hist as t
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
      simp only [stepWith, h hist]
      split
      · exact stepWith_retag H P P' x x' ω ω' vnext h n _ _ _
      · rfl
      · rfl

/-- **A session at a tagged statement is the session at the untagged one**: its zero knowledge carries over, against
the verifier read at the tagged statement. -/
theorem ZKmalAt.tag {WS N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : ℕ}
    {PS : WS → Prover (N × Ω)} {RB : Rel WS} {xs : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))}
    {ε γ : ℝ} (h : ZKmalAt H k PS (RB xs) xs C prog ε γ) :
    ZKmalAt H k (stopPS PS) (stopRel RB (1 :: xs)) (1 :: xs) C prog ε γ := by
  intro w hw Θ Vs θ hk Ev
  obtain ⟨w₀, rfl, hw₀⟩ : ∃ w₀, w = some w₀ ∧ RB xs w₀ := by
    cases w with
    | none => exact absurd hw (by simp [stopRel])
    | some w₀ =>
      obtain ⟨xs₀, hx, hr⟩ := hw
      cases hx
      exact ⟨w₀, rfl, hr⟩
  let Vs₀ : VStar Θ := ⟨fun _ => Vs.k (1 :: xs), fun _ as θ => Vs.next (1 :: xs) as θ⟩
  have hrun : ∀ ω, runStar H (stopPS PS (some w₀)) Vs (1 :: xs) ω θ = runStar H (PS w₀) Vs₀ xs ω θ := fun ω =>
    stepWith_retag H (PS w₀) (stopPS PS (some w₀)) xs (1 :: xs) ω ω _ (fun _ => rfl) _ _ _ _
  have hf : (Fork H (stopPS PS (some w₀)) Vs θ (1 :: xs)).Pr H = (Fork H (PS w₀) Vs₀ θ xs).Pr H := by
    simp only [Finder.Pr, Fork, hrun]
    rfl
  have hs : ∀ c, spOut H Vs θ (1 :: xs) (prog c) = spOut H Vs₀ θ xs (prog c) := fun _ => rfl
  simp only [hrun, hf, hs]
  exact h w₀ hw₀ Θ Vs₀ θ hk Ev

/-- The silent simulator of `n` rounds: the empty message every round. -/
def silentSP (n : ℕ) : SP (List (Msg × Out)) := spRun (fun _ => .ret []) n []

/-- **A session with no witness is silent, and simulated exactly** by the empty message every round, on any coins. -/
theorem zkmalAt_silent {WS N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (k : ℕ)
    (PS : WS → Prover (N × Ω)) (RB : Rel WS) (C : Type) [Fintype C] [Nonempty C] {ε γ : ℝ} (hε : 0 ≤ ε)
    (hγ : 0 ≤ γ) : ZKmalAt H k (stopPS PS) (stopRel RB []) [] C (fun _ => silentSP k) ε γ := by
  classical
  intro w hw Θ Vs θ hk Ev
  obtain rfl : w = none := by
    cases w with
    | none => rfl
    | some w₀ =>
      obtain ⟨xs₀, hx, -⟩ := hw
      exact absurd hx (List.cons_ne_nil _ _).symm
  have e : ∀ ω, (runStar H (stopPS PS none) Vs [] ω θ).msgs = spOut H Vs θ [] (silentSP k) := fun ω => by
    rw [runStar_msgs, spOut, silentSP, spRun_eval, hk]
    rfl
  simp only [e, Pr_const, sub_self, abs_zero]
  exact add_nonneg hε (mul_nonneg hγ (Finder.pr_nonneg H _))

/-- **A run's message is its prover's at the history before it.** -/
theorem histOf_runStar_getElem {Ω Θ : Type} (H : Msg → Msg) (P : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ)
    (i : ℕ) (hi : i < (histOf (runStar H P Vs x ω θ).msgs).length) :
    (histOf (runStar H P Vs x ω θ).msgs)[i].1 =
      ((P.next x ((histOf (runStar H P Vs x ω θ).msgs).take i) ω).eval H).1 := by
  generalize hτ : (runStar H P Vs x ω θ).msgs = τ at hi ⊢
  rw [runStar_msgs] at hτ
  have hiτ : i < τ.length := by rwa [histOf_length] at hi
  have hget := interact_getElem_fst _ _ _ [] i (hτ ▸ hiτ)
  rw [hτ, List.getElem?_eq_getElem hiτ] at hget
  simp only [List.nil_append, Option.map_some, Option.some.injEq] at hget
  rw [← histOf_take, ← hget]
  simp [histOf]

/-! ## The parts with the stop -/

namespace RunParts

variable {W WS T TS : Type}

/-- **The parts with the stop `s`**: session `j` at a history where the stop has fired within the first part is at
the empty statement with no witness; elsewhere at its statement tagged `1 ::`, with its witness. -/
def stopped (L : RunParts W WS T) (s : Msg → List (Msg × Msg) → Bool) : RunParts W (Option WS) T where
  k₀ := L.k₀
  n := L.n
  xs x j u := if fired (s x) (L.k₀ x) u then [] else 1 :: L.xs x j u
  ws j w t x u := if fired (s x) (L.k₀ x) u then none else some (L.ws j w t x u)

theorem stopped_start (L : RunParts W WS T) (s : Msg → List (Msg × Msg) → Bool) : (L.stopped s).start = L.start := by
  funext x j
  induction j with
  | zero => rfl
  | succ j ih =>
    show (L.stopped s).start x j + L.n x j = L.start x j + L.n x j
    rw [ih]

/-- **The parts with the stop run the prover with the stop.** -/
theorem prover_stopped (L : RunParts W WS T) (s : Msg → List (Msg × Msg) → Bool) (P₀ : W → Prover T)
    (PS : WS → Prover TS) :
    ∀ (m : ℕ) (w : W) (x : Msg) (hist : List (Msg × Msg)) (ω : T × (ℕ → TS)),
      ((L.stopped s).prover (fun w => stopP (P₀ w) s L.k₀) (stopPS PS) m w).next x hist ω =
        (stopP (L.prover P₀ PS m w) s L.k₀).next x hist ω
  | 0, _, _, _, _ => rfl
  | m + 1, w, x, hist, ω => by
      have hst := congrFun (congrFun (stopped_start L s) x) m
      by_cases hm : hist.length < L.start x m
      · have e1 : ((L.stopped s).prover (fun w => stopP (P₀ w) s L.k₀) (stopPS PS) (m + 1) w).next x hist ω =
            ((L.stopped s).prover (fun w => stopP (P₀ w) s L.k₀) (stopPS PS) m w).next x hist ω := by
          simp only [prover, hst, hm, ite_true]
        have e2 : (L.prover P₀ PS (m + 1) w).next x hist ω = (L.prover P₀ PS m w).next x hist ω := by
          simp only [prover, hm, ite_true]
        rw [e1, prover_stopped L s P₀ PS m w x hist ω]
        simp only [stopP, e2]
      · have hk : L.k₀ x ≤ L.start x m := L.start_mono x (Nat.zero_le m)
        have hf : fired (s x) (L.k₀ x) (hist.take (L.start x m)) = fired (s x) (L.k₀ x) hist := fired_take hist hk
        have e1 : ((L.stopped s).prover (fun w => stopP (P₀ w) s L.k₀) (stopPS PS) (m + 1) w).next x hist ω =
            (stopPS PS ((L.stopped s).ws m w ω.1 x (hist.take (L.start x m)))).next
              ((L.stopped s).xs x m (hist.take (L.start x m))) (hist.drop (L.start x m)) (ω.2 m) := by
          show (if hist.length < (L.stopped s).start x m then _ else _) = _
          rw [if_neg (by rw [hst]; exact hm), hst]
        have e2 : (L.prover P₀ PS (m + 1) w).next x hist ω = (PS (L.ws m w ω.1 x (hist.take (L.start x m)))).next
            (L.xs x m (hist.take (L.start x m))) (hist.drop (L.start x m)) (ω.2 m) := by
          show (if hist.length < L.start x m then _ else _) = _
          rw [if_neg hm]
        rw [e1]
        show _ = (if fired (s x) (L.k₀ x) hist = true then OC.ret [] else (L.prover P₀ PS (m + 1) w).next x hist ω)
        rw [e2]
        show (stopPS PS (if fired (s x) (L.k₀ x) (hist.take (L.start x m)) = true then none
            else some (L.ws m w ω.1 x (hist.take (L.start x m))))).next
            (if fired (s x) (L.k₀ x) (hist.take (L.start x m)) = true then []
              else 1 :: L.xs x m (hist.take (L.start x m))) (hist.drop (L.start x m)) (ω.2 m) = _
        rw [hf]
        by_cases hfi : fired (s x) (L.k₀ x) hist = true
        · rw [if_pos hfi, if_pos hfi, if_pos hfi]
          rfl
        · rw [if_neg hfi, if_neg hfi, if_neg hfi]
          rfl

end RunParts

/-- The sessions' programs with the stop: silent where the stop has fired, else `progB`. -/
def stopProgB (s : Msg → List (Msg × Msg) → Bool) (k : Msg → ℕ) (n : Msg → ℕ → ℕ) {CB : ℕ → Msg → Type}
    (progB : (j : ℕ) → (x : Msg) → List (Msg × Msg) → CB j x → SP (List (Msg × Out))) :
    (j : ℕ) → (x : Msg) → List (Msg × Msg) → CB j x → SP (List (Msg × Out)) :=
  fun j x u c => if fired (s x) (k x) u then silentSP (n x j) else progB j x u c

/-- **`m` sessions after a first part, with a public stop, at one statement**: the prover with the stop `s` (read at
the first part's rounds) is `ZKmalAt` at the stopped composition (the first part's simulator cut where the stop fires,
then each session's program where the stop hasn't fired and the silent one where it has), within `ε₀ + Σ_{j < m} eB j`
and `m·g`, where the first part is `ZKmalAt` with `ε₀` and rewinds nothing; session `j < m` is `ZKmalAt` with `eB j`
and `g` at every history of its start where the stop hasn't fired; its relation holds there at every honest first
part (its messages the first part's prover's on the tape `t₀`); and `go j` is `goesOn` at every run of its start. -/
theorem zkmalAt_layout_stop {W WS NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (T₀ : SeqTape) (L : RunParts W WS (T₀.N × T₀.Ω)) (s : Msg → List (Msg × Msg) → Bool) (H : Msg → Msg) (x : Msg)
    (m : ℕ) (P₀ : W → Prover (T₀.N × T₀.Ω)) (PS : WS → Prover (NB × ΩB)) (RA : Rel W) (RB : Rel WS)
    (S₀ : Simulator) (CB : ℕ → Msg → Type) [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → List (Msg × Msg) → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) (ε₀ g : ℝ) (eB : ℕ → ℝ)
    (h₀ : ZKmalAt H (L.k₀ x) P₀ (RA x) x (S₀.C x) (S₀.prog x) ε₀ 0)
    (hB : ∀ j < m, ∀ u : List (Msg × Msg), u.length = L.start x j → fired (s x) (L.k₀ x) u = false →
      ZKmalAt H (L.n x j) PS (RB (L.xs x j u)) (L.xs x j u) (CB j x) (progB j x u) (eB j) g)
    (hR : ∀ j < m, ∀ w, RA x w → ∀ (t₀ : T₀.N × T₀.Ω) (u : List (Msg × Msg)), u.length = L.start x j →
      (∀ i (hi : i < u.length), i < L.k₀ x → u[i].1 = (((P₀ w).next x (u.take i) t₀).eval H).1) →
      fired (s x) (L.k₀ x) u = false → RB (L.xs x j u) (L.ws j w t₀ x u))
    (hgo : ∀ j < m, ∀ (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg),
      go j x (interact ask pm (L.start x j) []) = goesOn (L.start x j) (interact ask pm (L.start x j) []))
    (he0 : ∀ j < m, 0 ≤ eB j) (hg : 0 ≤ g) :
    ZKmalAt H (L.start x m) (fun w => (stopP (L.prover P₀ PS m w) s L.k₀).comap (RunParts.flatten T₀ 0 NB ΩB m))
      (RA x) x
      ((iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (L.stopped s)) (censorSim S₀ s L.k₀) CB
        (stopProgB s L.k₀ L.n progB) go m).C x)
      ((iterSimG T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (L.stopped s)) (censorSim S₀ s L.k₀) CB
        (stopProgB s L.k₀ L.n progB) go m).prog x)
      (ε₀ + ∑ j ∈ Finset.range m, eB j) (m * g) := by
  have hst := L.stopped_start s
  have h₀' : ZKmalAt H ((L.stopped s).k₀ x) (fun w => stopP (P₀ w) s L.k₀) (RA x) x ((censorSim S₀ s L.k₀).C x)
      ((censorSim S₀ s L.k₀).prog x) ε₀ 0 := ZKmalAt.censor S₀ s L.k₀ h₀
  have h := zkmalAt_layout T₀ 0 (L.stopped s) H x m (fun w => stopP (P₀ w) s L.k₀) (stopPS PS) RA (stopRel RB)
    (censorSim S₀ s L.k₀) CB (stopProgB s L.k₀ L.n progB) go ε₀ 0 g eB h₀'
    (fun j hj u hu => by
      have hl := RunParts.reach_length T₀ 0 NB ΩB (L.stopped s) H _ RA x hu
      rw [hst] at hl
      show ZKmalAt H (L.n x j) (stopPS PS) (stopRel RB (if fired (s x) (L.k₀ x) u then [] else 1 :: L.xs x j u))
        (if fired (s x) (L.k₀ x) u then [] else 1 :: L.xs x j u) (CB j x)
        (fun c => if fired (s x) (L.k₀ x) u then silentSP (L.n x j) else progB j x u c) (eB j) g
      by_cases hf : fired (s x) (L.k₀ x) u = true
      · simp only [hf, ite_true]
        exact zkmalAt_silent H _ PS RB (CB j x) (he0 j hj) hg
      · rw [Bool.not_eq_true] at hf
        simp only [hf, Bool.false_eq_true, ite_false]
        exact (hB j hj u hl hf).tag)
    (fun j hj w hw Θ Vs θ ωA hgo' => by
      set P := iterProver T₀ 0 NB ΩB (RunParts.seq T₀ 0 NB ΩB (L.stopped s)) (fun w => stopP (P₀ w) s L.k₀)
        (fun _ => stopPS PS) j w with hP
      set Vh := Vs.head (RunParts.seq T₀ 0 NB ΩB (L.stopped s) j).kA
      set u := histOf (runStar H P Vh x ωA θ).msgs with hu
      have hl : u.length = L.start x j := by
        rw [hu, histOf_length]
        exact (goesOn_length hgo').trans (congrFun (congrFun hst x) j)
      show stopRel RB (if fired (s x) (L.k₀ x) u = true then [] else 1 :: L.xs x j u)
        (if fired (s x) (L.k₀ x) u = true then none else some (L.ws j w (RunParts.flatten T₀ 0 NB ΩB j ωA).1 x u))
      by_cases hf : fired (s x) (L.k₀ x) u = true
      · rw [if_pos hf, if_pos hf]
        rfl
      · rw [if_neg hf, if_neg hf]
        rw [Bool.not_eq_true] at hf
        refine ⟨L.xs x j u, rfl, hR j hj w hw _ u hl (fun i hi hk => ?_) hf⟩
        refine (histOf_runStar_getElem H P Vh x ωA θ i hi).trans ?_
        have e1 : P.next x (u.take i) ωA =
            (stopP (L.prover P₀ PS j w) s L.k₀).next x (u.take i) (RunParts.flatten T₀ 0 NB ΩB j ωA) :=
          (RunParts.iterProver_flat T₀ 0 NB ΩB (L.stopped s) (fun w => stopP (P₀ w) s L.k₀) (stopPS PS) j w x _
            ωA).trans (L.prover_stopped s P₀ PS j w x _ _)
        have hnf : fired (s x) (L.k₀ x) (u.take i) = false := by
          by_contra hne
          rw [Bool.not_eq_false] at hne
          rw [fired_of_take _ i hne] at hf
          exact Bool.noConfusion hf
        have hlt : (u.take i).length < L.k₀ x := by
          rw [List.length_take]; omega
        show ((P.next x (u.take i) ωA).eval H).1 = _
        rw [e1]
        show ((if fired (s x) (L.k₀ x) (u.take i) = true then OC.ret []
          else (L.prover P₀ PS j w).next x (u.take i) (RunParts.flatten T₀ 0 NB ΩB j ωA)).eval H).1 = _
        rw [hnf, if_neg Bool.false_ne_true, L.prover_pre P₀ PS j w x _ _ hlt])
    (fun j hj w _ Θ Vs θ ωA => by
      rw [runStar_msgs]
      have key : ∀ K, K = L.start x j → ∀ (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg),
          go j x (interact ask pm K []) = goesOn K (interact ask pm K []) := by
        rintro K rfl
        exact hgo j hj
      exact key _ (congrFun (congrFun hst x) j) _ _)
    (by simp) he0 hg
  rw [hst] at h
  simp only [Nat.cast_zero, zero_add] at h
  exact h.of_equiv _ (Equiv.refl _) (Equiv.refl _) fun w hist ω =>
    (L.prover_stopped s P₀ PS m w x hist (RunParts.flatten T₀ 0 NB ΩB m ω)).symm

end Claims.ZKMal
