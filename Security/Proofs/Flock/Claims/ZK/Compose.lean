import Proofs.Flock.Claims.ZK.Interact
import Proofs.Flock.Claims.ZK.Rewind

/-!
# Sequential composition of `ZKmalR`

Two protocols run in turn: `A` at the statement `x`, then `B` at a statement `Seq.xB x u` that a selection `u`, read
from `A`'s history, picks among finitely many. `zkmalR_seq` proves the composed protocol `ZKmalR` from `A`'s and `B`'s
(Goldreich–Oren's sequential composition, with the hybrids in Wee's order): the hybrid that changes `B` keeps `A` real,
so `B`'s `Fork` is charged against a verifier whose tape is `A`'s real run, and the one that changes `A` runs `B`'s
simulator after it, a function of `A`'s transcript and fresh coins.

The composed tape is `forkTape a`: a guess `j < a + 1`, both nonces, and two rests for `A`, of which `A` reads the
second when `j < a` and the first when `j = a`. `Fork` then shares `A`'s run at `j = a`, so its collision is `B`'s at
`A`'s real run, and at `j < a` it is `A`'s: the composed `Fork` is at least `(a·Fork_A + Fork_B)/(a + 1)`, and `γ` adds
up to `(a + 1)·g` where `γ_A ≤ a·g` and `γ_B ≤ g`. From one session (`a = 1`), `n` sessions in turn cost `n·g`.
-/

namespace Claims.ZKMal

open Claims

/-! ## Transcripts in two parts -/

/-- The answer is a message, not a decision. -/
def isMsg : Out → Bool
  | .msg _ => true
  | _ => false

/-- **The run goes on after `k` rounds**: `k` rounds, each answered with a message. -/
def goesOn (k : ℕ) (τ : List (Msg × Out)) : Bool := τ.length == k && τ.all fun p => isMsg p.2

/-- `τB` after `τA` where `τA` goes on, else `τA`. -/
def cont (k : ℕ) (τA τB : List (Msg × Out)) : List (Msg × Out) := if goesOn k τA then τA ++ τB else τA

theorem goesOn_cons (k : ℕ) (a : Msg) (m : Msg) (τ : List (Msg × Out)) :
    goesOn (k + 1) ((a, .msg m) :: τ) = goesOn k τ := by
  simp [goesOn, isMsg]

theorem goesOn_length {k : ℕ} {τ : List (Msg × Out)} (h : goesOn k τ = true) : τ.length = k := by
  simp only [goesOn, Bool.and_eq_true, beq_iff_eq] at h
  exact h.1

/-- **`m + n` rounds are `m` rounds, then `n` more where the first `m` go on.** -/
theorem interact_add (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (m n : ℕ) (hist : List (Msg × Msg)),
      interact ask pm (m + n) hist =
        cont m (interact ask pm m hist) (interact ask pm n (hist ++ histOf (interact ask pm m hist)))
  | 0, n, hist => by simp [interact, cont, goesOn, histOf]
  | m + 1, n, hist => by
      rw [show m + 1 + n = (m + n) + 1 by omega]
      simp only [interact]
      cases hb : ask (hist.map Prod.fst ++ [pm hist]) with
      | msg b =>
          simp only
          rw [interact_add ask pm m n (hist ++ [(pm hist, b)])]
          simp only [cont, goesOn_cons]
          split <;> simp [histOf, msgOf]
      | acc => simp [cont, goesOn, isMsg]
      | rej => simp [cont, goesOn, isMsg]

/-- **Rounds after a history are rounds against the verifier and prover that read it first.** -/
theorem interact_shift (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (h0 : List (Msg × Msg)) :
    ∀ (n : ℕ) (h : List (Msg × Msg)),
      interact ask pm n (h0 ++ h) =
        interact (fun as => ask (h0.map Prod.fst ++ as)) (fun h' => pm (h0 ++ h')) n h
  | 0, _ => by simp [interact]
  | n + 1, h => by
      simp only [interact, List.map_append, List.append_assoc]
      cases ask (h0.map Prod.fst ++ (h.map Prod.fst ++ [pm (h0 ++ h)])) with
      | msg b =>
          simp only
          rw [← interact_shift ask pm h0 n (h ++ [(pm (h0 ++ h), b)])]
      | acc => rfl
      | rej => rfl

theorem histOf_map_fst (τ : List (Msg × Out)) : (histOf τ).map Prod.fst = τ.map Prod.fst := by
  simp [histOf]

theorem histOf_length (τ : List (Msg × Out)) : (histOf τ).length = τ.length := by
  simp [histOf]

theorem histOf_take (τ : List (Msg × Out)) (i : ℕ) : histOf (τ.take i) = (histOf τ).take i := by
  simp [histOf, List.map_take]

theorem histOf_append (τ τ' : List (Msg × Out)) : histOf (τ ++ τ') = histOf τ ++ histOf τ' := by
  simp [histOf]

theorem stepWith_comap {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (f : Ω' → Ω) (x : Msg) (ω : Ω')
    (vnext : List Msg → OC Out) : ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      stepWith H (P.comap f) x ω vnext n hist as t = stepWith H P x (f ω) vnext n hist as t
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
      have e : (P.comap f).next x hist ω = P.next x hist (f ω) := rfl
      simp only [stepWith, e]
      split
      · exact stepWith_comap H P f x ω vnext n _ _ _
      · rfl
      · rfl

theorem runStar_comap {Ω Ω' Θ : Type} (H : Msg → Msg) (P : Prover Ω) (f : Ω' → Ω) (Vs : VStar Θ) (x : Msg)
    (ω : Ω') (θ : Θ) : runStar H (P.comap f) Vs x ω θ = runStar H P Vs x (f ω) θ :=
  stepWith_comap H P f x ω _ _ _ _ _

/-! ## Two protocols in turn -/

/-- **Two protocols in turn**: `A` for `kA x` rounds at `x`, then `B` at the statement `xB x u` with witness
`wB w ωA x u`, where `u`, one of finitely many, is read from `A`'s history (`sel`), and `B`'s witness may read `A`'s
tape `ωA` (the salts of what `A` committed, which `B` opens). -/
structure Seq (WA WB TA : Type) where
  kA : Msg → ℕ
  U : Msg → Type
  [finU : ∀ x, Fintype (U x)]
  [decU : ∀ x, DecidableEq (U x)]
  sel : (x : Msg) → List (Msg × Msg) → U x
  xB : (x : Msg) → U x → Msg
  wB : WA → TA → (x : Msg) → U x → WB

attribute [instance] Seq.finU Seq.decU

/-- The composed prover: `A`'s on its tape for `A`'s rounds, then `B`'s on its own, after `A`'s history. -/
def Seq.prover {WA WB ΩA ΩB : Type} (Q : Seq WA WB ΩA) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB) (w : WA) :
    Prover (ΩA × ΩB) :=
  ⟨fun x hist ω => if hist.length < Q.kA x then (PA w).next x hist ω.1 else
    (PB (Q.wB w ω.1 x (Q.sel x (hist.take (Q.kA x))))).next (Q.xB x (Q.sel x (hist.take (Q.kA x))))
      (hist.drop (Q.kA x)) ω.2⟩

/-- `V*` for the first `k` rounds. -/
def _root_.Claims.VStar.head {Θ : Type} (Vs : VStar Θ) (k : Msg → ℕ) : VStar Θ := ⟨k, Vs.next⟩

/-- `V*` after the prover's messages `pre` at `x`, for `k` rounds: its tape is `V*`'s and `pre`. -/
def _root_.Claims.VStar.after {Θ : Type} (Vs : VStar Θ) (x : Msg) (k : Msg → ℕ) : VStar (Θ × List Msg) :=
  ⟨k, fun _ as θ' => Vs.next x (θ'.2 ++ as) θ'.1⟩

/-- A simulator program whose questions to the verifier come after `pre`. -/
def spUnder {α : Type} (pre : List Msg) : SP α → SP α
  | .ret a => .ret a
  | .hash q k => .hash q fun r => spUnder pre (k r)
  | .ask as k => .ask (pre ++ as) fun b => spUnder pre (k b)

theorem spUnder_eval {α : Type} (H : Msg → Msg) (ask : List Msg → Out) (pre : List Msg) :
    ∀ p : SP α, SP.eval H ask (spUnder pre p) = SP.eval H (fun as => ask (pre ++ as)) p
  | .ret _ => rfl
  | .hash q k => by simp only [spUnder, SP.eval, spUnder_eval H ask pre (k (H q))]
  | .ask as k => by simp only [spUnder, SP.eval, spUnder_eval H ask pre (k (ask (pre ++ as)))]

/-- **The composed simulator**: `A`'s, then, where its transcript goes on, `B`'s at the unit that transcript selects,
asking `V*` after `A`'s messages. Its coins are `A`'s and `B`'s at every unit. -/
def Seq.sim {WA WB TA : Type} (Q : Seq WA WB TA) (SA SB : Simulator) : Simulator where
  C x := SA.C x × ((u : Q.U x) → SB.C (Q.xB x u))
  prog x c := spBind (SA.prog x c.1) fun τA =>
    if goesOn (Q.kA x) τA then
      spBind (spUnder (τA.map Prod.fst) (SB.prog (Q.xB x (Q.sel x (histOf τA))) (c.2 (Q.sel x (histOf τA)))))
        fun τB => .ret (τA ++ τB)
    else .ret τA

theorem Seq.sim_out {WA WB TA Θ : Type} (Q : Seq WA WB TA) (SA SB : Simulator) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ)
    (x : Msg) (kB : Msg → ℕ) (c : (Q.sim SA SB).C x) :
    (Q.sim SA SB).out H Vs θ x c =
      cont (Q.kA x) (SA.out H (Vs.head Q.kA) θ x c.1)
        (SB.out H (Vs.after x kB) (θ, (SA.out H (Vs.head Q.kA) θ x c.1).map Prod.fst)
          (Q.xB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x c.1))))
          (c.2 (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x c.1))))) := by
  simp only [Simulator.out, Seq.sim, spBind_eval, cont, VStar.head, VStar.after]
  by_cases hg : goesOn (Q.kA x) (SP.eval H (fun as => ((Vs.next x as θ).eval H).1) (SA.prog x c.1)).1 = true
  · simp only [hg, ite_true, spBind_eval, spUnder_eval, SP.eval]
  · simp only [hg, ite_false, Bool.false_eq_true, SP.eval]

/-- **The composed run is `A`'s run, then where it goes on `B`'s run** against `V*` after `A`'s messages. -/
theorem Seq.msgs_eq {WA WB ΩA ΩB Θ : Type} (Q : Seq WA WB ΩA) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB)
    (w : WA) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB nB : Msg → ℕ)
    (hk : ∀ u, kB (Q.xB x u) = nB x) (hV : Vs.k x = Q.kA x + nB x) (ωA : ΩA) (ωB : ΩB) :
    (runStar H (Q.prover PA PB w) Vs x (ωA, ωB) θ).msgs =
      cont (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs
        (runStar H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))) ωB
          (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)).msgs := by
  rw [ZKMal.runStar_msgs, ZKMal.runStar_msgs, ZKMal.runStar_msgs, hV, interact_add]
  have hA : interact (fun as => ((Vs.next x as θ).eval H).1)
      (fun h => (((Q.prover PA PB w).next x h (ωA, ωB)).eval H).1) (Q.kA x) [] =
      interact (fun as => (((Vs.head Q.kA).next x as θ).eval H).1) (fun h => (((PA w).next x h ωA).eval H).1)
        ((Vs.head Q.kA).k x) [] := by
    simp only [VStar.head]
    refine interact_congr _ _ _ _ _ fun i hi => ?_
    have hl := interact_length_le (fun as => ((Vs.next x as θ).eval H).1)
      (fun h => (((PA w).next x h ωA).eval H).1) (Q.kA x) []
    have : ([] ++ histOf ((interact (fun as => ((Vs.next x as θ).eval H).1)
        (fun h => (((PA w).next x h ωA).eval H).1) (Q.kA x) []).take i)).length < Q.kA x := by
      rw [List.nil_append, histOf_length, List.length_take]
      omega
    simp only [Seq.prover, this, ite_true]
  rw [hA]
  simp only [cont, List.nil_append]
  split
  · rename_i hgo
    congr 1
    rw [← List.append_nil (histOf _), interact_shift]
    have hlen := goesOn_length hgo
    simp only [VStar.after, hk, histOf_map_fst]
    congr 1
    funext h'
    have hl : (histOf (interact (fun as => (((Vs.head Q.kA).next x as θ).eval H).1)
        (fun h => (((PA w).next x h ωA).eval H).1) ((Vs.head Q.kA).k x) [])).length = Q.kA x := by
      rw [histOf_length]; exact hlen
    have hn : ¬ (histOf (interact (fun as => (((Vs.head Q.kA).next x as θ).eval H).1)
        (fun h => (((PA w).next x h ωA).eval H).1) ((Vs.head Q.kA).k x) []) ++ h').length < Q.kA x := by
      rw [List.length_append, hl]; omega
    simp only [Seq.prover, hn, ite_false]
    rw [← hl, List.take_left, List.drop_left, List.append_nil]
  · rfl

/-! ## The prover's calls -/

/-- **Every call in a run's `pLog` is one round's**: the trace's own, or the prover's on the history before some
round. -/
theorem stepWith_pLog_mem {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), as = hist.map Prod.fst →
      ∀ e ∈ (stepWith H P x ω vnext n hist as t).pLog, e ∈ t.pLog ∨
        ∃ i < (interact (fun as => ((vnext as).eval H).1) (fun h => ((P.next x h ω).eval H).1) n hist).length,
          e ∈ ((P.next x (hist ++ histOf ((interact (fun as => ((vnext as).eval H).1)
            (fun h => ((P.next x h ω).eval H).1) n hist).take i)) ω).eval H).2
  | 0, _, _, _, _, e, he => by simp only [stepWith] at he; exact Or.inl he
  | n + 1, hist, as, t, has, e, he => by
      subst has
      simp only [stepWith, interact] at he ⊢
      cases hb : ((vnext (List.map Prod.fst hist ++ [((P.next x hist ω).eval H).1])).eval H).1 with
      | msg m =>
          rw [hb] at he
          simp only at he ⊢
          rcases stepWith_pLog_mem H P x ω vnext n _ _ _ (by simp) e he with h | ⟨i, hi, h⟩
          · rcases List.mem_append.mp h with h | h
            · exact Or.inl h
            · exact Or.inr ⟨0, by simp, by simpa [histOf] using h⟩
          · refine Or.inr ⟨i + 1, by simpa using hi, ?_⟩
            simpa [histOf, msgOf, List.take_succ_cons] using h
      | acc =>
          rw [hb] at he
          simp only at he
          rcases List.mem_append.mp he with h | h
          · exact Or.inl h
          · exact Or.inr ⟨0, by simp, by simpa [histOf] using h⟩
      | rej =>
          rw [hb] at he
          simp only at he
          rcases List.mem_append.mp he with h | h
          · exact Or.inl h
          · exact Or.inr ⟨0, by simp, by simpa [histOf] using h⟩

theorem runStar_pLog_mem {Ω Θ : Type} (H : Msg → Msg) (P : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ) :
    ∀ e ∈ (runStar H P Vs x ω θ).pLog, ∃ i < (runStar H P Vs x ω θ).msgs.length,
      e ∈ ((P.next x (histOf ((runStar H P Vs x ω θ).msgs.take i)) ω).eval H).2 := by
  intro e he
  rcases stepWith_pLog_mem H P x ω (fun as => Vs.next x as θ) (Vs.k x) [] [] ⟨[], [], [], .rej⟩ rfl e he with h | h
  · simp at h
  · rw [ZKMal.runStar_msgs]
    simpa using h

theorem runStar_msgs_length_le {Ω Θ : Type} (H : Msg → Msg) (P : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω)
    (θ : Θ) : (runStar H P Vs x ω θ).msgs.length ≤ Vs.k x := by
  rw [ZKMal.runStar_msgs]
  exact interact_length_le _ _ _ _

/-- **`A`'s calls are the composed run's.** -/
theorem Seq.pLog_A {WA WB ΩA ΩB Θ : Type} (Q : Seq WA WB ΩA) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB)
    (w : WA) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB nB : Msg → ℕ)
    (hk : ∀ u, kB (Q.xB x u) = nB x) (hV : Vs.k x = Q.kA x + nB x) (ωA : ΩA) (ωB : ΩB) :
    ∀ e ∈ (runStar H (PA w) (Vs.head Q.kA) x ωA θ).pLog, e ∈ (runStar H (Q.prover PA PB w) Vs x (ωA, ωB) θ).pLog := by
  intro e he
  obtain ⟨i, hi, he⟩ := runStar_pLog_mem H (PA w) (Vs.head Q.kA) x ωA θ e he
  have hle : (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.length ≤ Q.kA x :=
    runStar_msgs_length_le H (PA w) (Vs.head Q.kA) x ωA θ
  have hm := Q.msgs_eq PA PB w H Vs θ x kB nB hk hV ωA ωB
  have htake : (runStar H (Q.prover PA PB w) Vs x (ωA, ωB) θ).msgs.take i =
      (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.take i := by
    rw [hm]; unfold cont; split
    · exact List.take_append_of_le_length hi.le
    · rfl
  have hlt : i < (runStar H (Q.prover PA PB w) Vs x (ωA, ωB) θ).msgs.length := by
    rw [hm]; unfold cont; split
    · rw [List.length_append]; omega
    · exact hi
  refine runStar_round_log H (Q.prover PA PB w) Vs x (ωA, ωB) θ i hlt ?_
  rw [htake]
  have hs : (histOf ((runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.take i)).length < Q.kA x := by
    rw [histOf_length, List.length_take]; omega
  simpa only [Seq.prover, hs, ite_true] using he

/-- **Where `A`'s run goes on, `B`'s calls are the composed run's.** -/
theorem Seq.pLog_B {WA WB ΩA ΩB Θ : Type} (Q : Seq WA WB ΩA) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB)
    (w : WA) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB nB : Msg → ℕ)
    (hk : ∀ u, kB (Q.xB x u) = nB x) (hV : Vs.k x = Q.kA x + nB x) (ωA : ΩA) (ωB : ΩB)
    (hgo : goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs = true) :
    ∀ e ∈ (runStar H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))) ωB
          (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)).pLog,
      e ∈ (runStar H (Q.prover PA PB w) Vs x (ωA, ωB) θ).pLog := by
  intro e he
  obtain ⟨i, hi, he⟩ := runStar_pLog_mem H _ _ _ ωB _ e he
  have hm := Q.msgs_eq PA PB w H Vs θ x kB nB hk hV ωA ωB
  simp only [cont, hgo, ite_true] at hm
  have hlen := goesOn_length hgo
  refine runStar_round_log H (Q.prover PA PB w) Vs x (ωA, ωB) θ (Q.kA x + i) ?_ ?_
  · rw [hm, List.length_append, hlen]; omega
  · rw [hm, ← hlen, List.take_length_add_append, histOf_append]
    have hl : (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs).length = Q.kA x := by
      rw [histOf_length, hlen]
    have hn : ¬ (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs ++
        histOf ((runStar H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))) ωB
          (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)).msgs.take i)).length < Q.kA x := by
      rw [List.length_append, hl]; omega
    simp only [Seq.prover, hn, ite_false]
    rw [← hl, List.take_left, List.drop_left]
    exact he

/-! ## The composed tape -/

/-- **The composed tape**: the guess `j`, `A`'s and `B`'s nonces, and `A`'s first rest, with `A`'s second rest and
`B`'s rest as the rest. `A` reads its second rest when `j < a`, so `Fork`'s two runs differ in `A`, and its first when
`j = a`, so they share `A`'s run and differ in `B`. -/
def forkTape (a : ℕ) {NA ΩA NB ΩB : Type} (s : (Fin (a + 1) × NA × NB × ΩA) × (ΩA × ΩB)) :
    (NA × ΩA) × (NB × ΩB) :=
  ((s.1.2.1, if (s.1.1 : ℕ) < a then s.2.1 else s.1.2.2.2), (s.1.2.2.1, s.2.2))

/-- `forkTape` with the guess and the rest it leaves unread. -/
def forkEquiv (a : ℕ) (NA ΩA NB ΩB : Type) :
    (Fin (a + 1) × NA × NB × ΩA) × (ΩA × ΩB) ≃ Fin (a + 1) × ((NA × ΩA) × (NB × ΩB)) × ΩA where
  toFun s := (s.1.1, forkTape a s, if (s.1.1 : ℕ) < a then s.1.2.2.2 else s.2.1)
  invFun q := ((q.1, q.2.1.1.1, q.2.1.2.1, if (q.1 : ℕ) < a then q.2.2 else q.2.1.1.2),
    (if (q.1 : ℕ) < a then q.2.1.1.2 else q.2.2, q.2.1.2.2))
  left_inv s := by
    obtain ⟨⟨j, νA, νB, r0⟩, r1, rB⟩ := s
    by_cases h : (j : ℕ) < a <;> simp [forkTape, h]
  right_inv q := by
    obtain ⟨j, ⟨⟨νA, r⟩, νB, rB⟩, o⟩ := q
    by_cases h : (j : ℕ) < a <;> simp [forkTape, h]

/-- **`forkTape` reads a uniform tape**: an event of the two sessions' tapes has its probability. -/
theorem Pr_forkTape (a : ℕ) {NA ΩA NB ΩB : Type} [Fintype NA] [Fintype ΩA] [Nonempty ΩA] [Fintype NB] [Fintype ΩB]
    (p : (NA × ΩA) × (NB × ΩB) → Prop) : Pr (fun s => p (forkTape a s)) = Pr p := by
  have e := Pr_equiv (forkEquiv a NA ΩA NB ΩB) fun q => p q.2.1
  simp only [forkEquiv, Equiv.coe_fn_mk] at e
  rw [e, Pr_prod_left]
  simp only [Pr_fst p]
  exact Ex_const _

/-- **One unit's coins among every unit's** have the law of that unit's. -/
theorem Pr_pi_at {ι : Type} [Fintype ι] [DecidableEq ι] {β : ι → Type} [∀ i, Fintype (β i)]
    [∀ i, Nonempty (β i)] (i : ι) (p : β i → Prop) : Pr (fun c : (j : ι) → β j => p (c i)) = Pr p := by
  have e := Pr_equiv (Equiv.piSplitAt i β) fun q => p q.1
  rw [← Pr_fst (β := (j : {j // j ≠ i}) → β j) p, ← e]
  rfl

/-! ## The composed `Fork` -/

theorem findColl_ne_none_mono {l₁ l₂ m₁ m₂ : List (Msg × Msg)} (h₁ : ∀ e ∈ l₁, e ∈ m₁) (h₂ : ∀ e ∈ l₂, e ∈ m₂)
    (h : findColl (l₁ ++ l₂) ≠ none) : findColl (m₁ ++ m₂) ≠ none := by
  rw [findColl_ne_none_iff] at h ⊢
  obtain ⟨p, hp, q, hq, hne, heq⟩ := h
  have sub : ∀ e ∈ l₁ ++ l₂, e ∈ m₁ ++ m₂ := fun e he => by
    rcases List.mem_append.mp he with h | h
    exacts [List.mem_append_left _ (h₁ e h), List.mem_append_right _ (h₂ e h)]
  exact ⟨p, sub p hp, q, sub q hq, hne, heq⟩

theorem Fork_pr_le {N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (P : Prover (N × Ω)) (Vs : VStar Θ) (θ : Θ) (x : Msg) :
    (Fork H P Vs θ x).Pr H ≤ Pr (fun s : N × Ω × Ω => findColl ((runStar H P Vs x (s.1, s.2.1) θ).pLog ++
      (runStar H P Vs x (s.1, s.2.2) θ).pLog) ≠ none) :=
  Pr_mono fun s ⟨u, v, h, _⟩ => by
    change findColl _ = some (u, v) at h
    rw [h]; simp

/-- **The composed `Fork` is at least `A`'s at `a` of its `a + 1` guesses, and `B`'s at the last**, where `B`'s is
charged at `A`'s real run, and only where that run goes on. -/
theorem Seq.fork_ge {WA WB NA ΩA NB ΩB Θ : Type} [Fintype NA] [Nonempty NA] [Fintype ΩA] [Nonempty ΩA]
    [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (Q : Seq WA WB (NA × ΩA)) (PA : WA → Prover (NA × ΩA)) (PB : WB → Prover (NB × ΩB)) (w : WA) (H : Msg → Msg)
    (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB nB : Msg → ℕ) (hk : ∀ u, kB (Q.xB x u) = nB x)
    (hV : Vs.k x = Q.kA x + nB x) (a : ℕ) :
    a * (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H +
      Ex (fun ωA : NA × ΩA => if goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs then
        (Fork H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x kB) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)
          (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))).Pr H else 0) ≤
      (a + 1) * (Fork H ((Q.prover PA PB w).comap (forkTape a)) Vs θ x).Pr H := by
  classical
  refine le_trans ?_ (mul_le_mul_of_nonneg_left (Fork_pr_ge H _ Vs θ x) (by positivity))
  simp only [runStar_comap]
  let e : (Fin (a + 1) × NA × NB × ΩA) × (ΩA × ΩB) × (ΩA × ΩB) ≃
      Fin (a + 1) × (NA × NB × ΩA × ΩA × ΩB × ΩA × ΩB) :=
    { toFun := fun s => (s.1.1, s.1.2.1, s.1.2.2.1, s.1.2.2.2, s.2.1.1, s.2.1.2, s.2.2.1, s.2.2.2)
      invFun := fun q => ((q.1, q.2.1, q.2.2.1, q.2.2.2.1), (q.2.2.2.2.1, q.2.2.2.2.2.1),
        (q.2.2.2.2.2.2.1, q.2.2.2.2.2.2.2))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [← Pr_equiv e.symm, Pr_guess, Nat.cast_add_one, mul_div_cancel₀ _ (by positivity), Fin.sum_univ_castSucc]
  refine add_le_add ?_ ?_
  · -- `j < a`: `A`'s collision
    have hA : (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H ≤ Pr fun b : NA × NB × ΩA × ΩA × ΩB × ΩA × ΩB =>
        findColl ((runStar H (PA w) (Vs.head Q.kA) x (b.1, b.2.2.2.1) θ).pLog ++
          (runStar H (PA w) (Vs.head Q.kA) x (b.1, b.2.2.2.2.2.1) θ).pLog) ≠ none := by
      refine (Fork_pr_le H (PA w) (Vs.head Q.kA) θ x).trans (le_of_eq ?_)
      let e' : NA × NB × ΩA × ΩA × ΩB × ΩA × ΩB ≃ (NA × ΩA × ΩA) × (NB × ΩA × ΩB × ΩB) :=
        { toFun := fun b => ((b.1, b.2.2.2.1, b.2.2.2.2.2.1), (b.2.1, b.2.2.1, b.2.2.2.2.1, b.2.2.2.2.2.2))
          invFun := fun q => (q.1.1, q.2.1, q.2.2.1, q.1.2.1, q.2.2.2.1, q.1.2.2, q.2.2.2.2)
          left_inv := fun _ => rfl
          right_inv := fun _ => rfl }
      rw [← Pr_fst (β := NB × ΩA × ΩB × ΩB), ← Pr_equiv e']
      rfl
    have hle : ∀ i : Fin a, (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H ≤
        Pr fun b => findColl ((runStar H (Q.prover PA PB w) Vs x (forkTape a ((e.symm (i.castSucc, b)).1,
          (e.symm (i.castSucc, b)).2.1)) θ).pLog ++ (runStar H (Q.prover PA PB w) Vs x (forkTape a
            ((e.symm (i.castSucc, b)).1, (e.symm (i.castSucc, b)).2.2)) θ).pLog) ≠ none := by
      intro i
      refine hA.trans (Pr_mono fun b hb => ?_)
      have hi : ((i.castSucc : Fin (a + 1)) : ℕ) < a := by simp
      simp only [e, Equiv.coe_fn_symm_mk, forkTape, hi, ite_true] at hb ⊢
      exact findColl_ne_none_mono (Q.pLog_A PA PB w H Vs θ x kB nB hk hV _ _)
        (Q.pLog_A PA PB w H Vs θ x kB nB hk hV _ _) hb
    refine le_trans (le_of_eq ?_) (Finset.sum_le_sum fun i _ => hle i)
    simp
  · -- `j = a`: `B`'s collision at `A`'s real run
    have hlast : ¬ ((Fin.last a : Fin (a + 1)) : ℕ) < a := by simp
    let pB : (NA × ΩA) × (NB × ΩB × ΩB) → Prop := fun q =>
      goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x q.1 θ).msgs = true ∧
        findColl ((runStar H (PB (Q.wB w q.1 x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x q.1 θ).msgs))))
            (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x q.1 θ).msgs)))
            (q.2.1, q.2.2.1) (θ, (runStar H (PA w) (Vs.head Q.kA) x q.1 θ).msgs.map Prod.fst)).pLog ++
          (runStar H (PB (Q.wB w q.1 x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x q.1 θ).msgs))))
            (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x q.1 θ).msgs)))
            (q.2.1, q.2.2.2) (θ, (runStar H (PA w) (Vs.head Q.kA) x q.1 θ).msgs.map Prod.fst)).pLog) ≠ none
    have h1 : Ex (fun ωA : NA × ΩA => if goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs then
        (Fork H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x kB) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)
          (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))).Pr H else 0) ≤ Pr pB := by
      rw [Pr_prod_left]
      refine Ex_mono fun ωA => ?_
      split
      · rename_i hgo
        refine (Fork_pr_le H _ _ _ _).trans (le_of_eq (Pr_congr fun s => ?_))
        simp only [pB, hgo, true_and]
      · exact Pr_nonneg _
    let e'' : NA × NB × ΩA × ΩA × ΩB × ΩA × ΩB ≃ ((NA × ΩA) × (NB × ΩB × ΩB)) × (ΩA × ΩA) :=
      { toFun := fun b => (((b.1, b.2.2.1), (b.2.1, b.2.2.2.2.1, b.2.2.2.2.2.2)), (b.2.2.2.1, b.2.2.2.2.2.1))
        invFun := fun q => (q.1.1.1, q.1.2.1, q.1.1.2, q.2.1, q.1.2.2.1, q.2.2, q.1.2.2.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    have h2 : Pr (fun b => pB (e'' b).1) = Pr pB := (Pr_equiv e'' fun q => pB q.1).trans (Pr_fst pB)
    refine h1.trans (h2.symm.le.trans (Pr_mono fun b hb => ?_))
    obtain ⟨hgo, hc⟩ := hb
    simp only [e, Equiv.coe_fn_symm_mk, forkTape, hlast, ite_false]
    exact findColl_ne_none_mono (Q.pLog_B PA PB w H Vs θ x kB nB hk hV _ _ hgo)
      (Q.pLog_B PA PB w H Vs θ x kB nB hk hV _ _ hgo) hc

/-! ## The hybrids -/

/-- **`B`'s relation holds at every unit an honest run of `A` selects**, against every `V*`, at that run's tape:
what `B`'s statement reads of `A`'s messages (a commitment) is what `A` committed. -/
def Seq.RelOn {WA WB TA : Type} (Q : Seq WA WB TA) (H : Msg → Msg) (PA : WA → Prover TA) (RA : Rel WA)
    (RB : Rel WB) : Prop :=
  ∀ x w, RA x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (ωA : TA),
    goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs = true →
      RB (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))
        (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))

/-- `RelOn` where `B`'s relation holds at every unit and tape. -/
theorem Seq.relOn_of_forall {WA WB TA : Type} (Q : Seq WA WB TA) (H : Msg → Msg) (PA : WA → Prover TA)
    (RA : Rel WA) (RB : Rel WB) (h : ∀ x w, RA x w → ∀ ωA u, RB (Q.xB x u) (Q.wB w ωA x u)) :
    Q.RelOn H PA RA RB :=
  fun x w hw _ _ _ ωA _ => h x w hw ωA _

theorem Finder.pr_nonneg (H : Msg → Msg) (F : Finder) : 0 ≤ F.Pr H := Pr_nonneg _

/-- **Two hybrids**: change the second coordinate at each value of the first, then the first at each value of the
second. -/
theorem abs_sub_le_hybrid {TA TB CA CB : Type} [Fintype TA] [Fintype TB] [Fintype CA] [Fintype CB] [Nonempty CB]
    (pR : TA → TB → Prop) (pH : TA → CB → Prop) (pS : CA → CB → Prop) (b : TA → ℝ) (d : ℝ)
    (hB : ∀ t, |Pr (pR t) - Pr (pH t)| ≤ b t) (hA : ∀ c, |Pr (fun t => pH t c) - Pr (fun c' => pS c' c)| ≤ d) :
    |Pr (fun s : TA × TB => pR s.1 s.2) - Pr (fun s : CA × CB => pS s.1 s.2)| ≤ Ex b + d := by
  have h1 : |Pr (fun s : TA × TB => pR s.1 s.2) - Pr (fun s : TA × CB => pH s.1 s.2)| ≤ Ex b := by
    rw [Pr_prod_left (fun s : TA × TB => pR s.1 s.2), Pr_prod_left (fun s : TA × CB => pH s.1 s.2)]
    exact Ex_abs_sub_le hB
  have h2 : |Pr (fun s : TA × CB => pH s.1 s.2) - Pr (fun s : CA × CB => pS s.1 s.2)| ≤ d := by
    rw [Pr_prod_right (fun s : TA × CB => pH s.1 s.2), Pr_prod_right (fun s : CA × CB => pS s.1 s.2)]
    exact (Ex_abs_sub_le (b := fun _ => d) hA).trans (Ex_const d).le
  exact (abs_sub_le _ _ _).trans (add_le_add h1 h2)

/-- **Sequential composition of `ZKmalR`.** `A` then `B` at the unit `A`'s history selects is `ZKmalR` with the
simulator that runs `SA`, then `SB` at that unit with fresh coins: `ε` adds up, with `B`'s at its worst unit (`eB`),
and so does `γ` (`γ_A ≤ a·g`, `γ_B ≤ g` at every unit), to `(a + 1)·g` against the composed `Fork`. The composed
tape is `forkTape a`. `B`'s hybrid keeps `A` real and gives `V*` `A`'s messages as its tape (`Vs.after`): `B`'s
`ZKmalR` is against every tape, chosen after `H`, at the witness `A`'s tape gives, where `B`'s relation holds
(`RelOn`). `A`'s runs `SB` after `A`'s transcript, a function of that transcript and coins independent of `A`'s. -/
theorem zkmalR_seq {WA WB NA ΩA NB ΩB : Type} [Fintype NA] [Nonempty NA] [Fintype ΩA] [Nonempty ΩA]
    [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (Q : Seq WA WB (NA × ΩA)) (H : Msg → Msg) (kB nB : Msg → ℕ)
    (PA : WA → Prover (NA × ΩA)) (PB : WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : Rel WB) (SA SB : Simulator)
    (εA γA εB γB eB g : Msg → ℝ) (a : ℕ)
    (hA : ZKmalR H Q.kA PA RA SA εA γA) (hB : ZKmalR H kB PB RB SB εB γB)
    (hR : Q.RelOn H PA RA RB) (hk : ∀ x u, kB (Q.xB x u) = nB x)
    (hε : ∀ x u, εB (Q.xB x u) ≤ eB x) (he0 : ∀ x, 0 ≤ eB x)
    (hγ : ∀ x u, γB (Q.xB x u) ≤ g x) (hγA : ∀ x, γA x ≤ a * g x) (hg : ∀ x, 0 ≤ g x) :
    ZKmalR H (fun x => Q.kA x + nB x) (fun w => (Q.prover PA PB w).comap (forkTape a)) RA (Q.sim SA SB)
      (fun x => εA x + eB x) (fun x => (a + 1) * g x) := by
  classical
  intro x w hw Θ Vs θ hV Ev
  have hreal : Pr (fun ω => Ev (runStar H ((Q.prover PA PB w).comap (forkTape a)) Vs x ω θ).msgs) =
      Pr (fun s : (NA × ΩA) × (NB × ΩB) => Ev (cont (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs
        (runStar H (PB (Q.wB w s.1 x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs))))
          (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs))) s.2
          (θ, (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs.map Prod.fst)).msgs)) := by
    simp only [runStar_comap]
    refine (Pr_forkTape a fun t => Ev (runStar H (Q.prover PA PB w) Vs x t θ).msgs).trans (Pr_congr fun t => ?_)
    obtain ⟨ωA, ωB⟩ := t
    rw [Q.msgs_eq PA PB w H Vs θ x kB nB (hk x) hV ωA ωB]
  have hsim : Pr (fun c : (Q.sim SA SB).C x => Ev ((Q.sim SA SB).out H Vs θ x c)) =
      Pr (fun s : SA.C x × ((u : Q.U x) → SB.C (Q.xB x u)) => Ev (cont (Q.kA x) (SA.out H (Vs.head Q.kA) θ x s.1)
        (SB.out H (Vs.after x kB) (θ, (SA.out H (Vs.head Q.kA) θ x s.1).map Prod.fst)
          (Q.xB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x s.1))))
          (s.2 (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x s.1))))))) := by
    refine Pr_congr fun c => ?_
    rw [Q.sim_out SA SB H Vs θ x kB c]
  rw [hreal, hsim]
  have hF := Q.fork_ge PA PB w H Vs θ x kB nB (hk x) hV a
  refine (abs_sub_le_hybrid (TA := NA × ΩA) (TB := NB × ΩB) (CA := SA.C x)
    (CB := (u : Q.U x) → SB.C (Q.xB x u))
    (fun ωA ωB => Ev (cont (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs
      (runStar H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
        (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))) ωB
        (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)).msgs))
    (fun ωA cB => Ev (cont (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs
      (SB.out H (Vs.after x kB) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)
        (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))
        (cB (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))))
    (fun cA cB => Ev (cont (Q.kA x) (SA.out H (Vs.head Q.kA) θ x cA)
      (SB.out H (Vs.after x kB) (θ, (SA.out H (Vs.head Q.kA) θ x cA).map Prod.fst)
        (Q.xB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x cA))))
        (cB (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x cA)))))))
    (fun ωA => eB x + (if goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs then
        (Fork H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x kB) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)
          (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))).Pr H else 0) * g x)
    (εA x + γA x * (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H) ?_ ?_).trans ?_
  · -- `B`'s hybrid, at `A`'s real run
    intro ωA
    by_cases hgo : goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs = true
    · simp only [cont, hgo, ite_true]
      rw [Pr_pi_at (β := fun u => SB.C (Q.xB x u))
        (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))
        (fun c => Ev ((runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs ++
          SB.out H (Vs.after x kB) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)
            (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))) c))]
      refine (hB _ _ (hR x w hw Θ Vs θ ωA hgo) (Θ × List Msg)
        (Vs.after x kB) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst) rfl
        (fun τ => Ev ((runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs ++ τ))).trans ?_
      refine add_le_add (hε x _) ?_
      rw [mul_comm]
      exact mul_le_mul_of_nonneg_left (hγ x _) (Finder.pr_nonneg _ _)
    · simp only [cont, hgo, ite_false, Bool.false_eq_true, Pr_const, sub_self, abs_zero, zero_mul, add_zero]
      exact he0 x
  · -- `A`'s hybrid, with `SB` after it
    intro cB
    exact hA x w hw Θ (Vs.head Q.kA) θ rfl fun τ => Ev (cont (Q.kA x) τ
      (SB.out H (Vs.after x kB) (θ, τ.map Prod.fst) (Q.xB x (Q.sel x (histOf τ))) (cB (Q.sel x (histOf τ)))))
  · rw [Ex_const_add, Ex_mul_const]
    have hFA := Finder.pr_nonneg H (Fork H (PA w) (Vs.head Q.kA) θ x)
    have h1 : γA x * (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H ≤ a * g x * (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H :=
      mul_le_mul_of_nonneg_right (hγA x) hFA
    have h2 := mul_le_mul_of_nonneg_left hF (hg x)
    nlinarith

end Claims.ZKMal
