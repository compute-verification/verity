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

/-- **Two protocols in turn**: `A` for `kA x` rounds at `x`, then `B` at the statement `xB x u` with witness `wB w x u`,
where `u`, one of finitely many, is read from `A`'s history (`sel`). -/
structure Seq (WA WB : Type) where
  kA : Msg → ℕ
  U : Msg → Type
  [finU : ∀ x, Fintype (U x)]
  [decU : ∀ x, DecidableEq (U x)]
  sel : (x : Msg) → List (Msg × Msg) → U x
  xB : (x : Msg) → U x → Msg
  wB : WA → (x : Msg) → U x → WB

attribute [instance] Seq.finU Seq.decU

/-- The composed prover: `A`'s on its tape for `A`'s rounds, then `B`'s on its own, after `A`'s history. -/
def Seq.prover {WA WB ΩA ΩB : Type} (Q : Seq WA WB) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB) (w : WA) :
    Prover (ΩA × ΩB) :=
  ⟨fun x hist ω => if hist.length < Q.kA x then (PA w).next x hist ω.1 else
    (PB (Q.wB w x (Q.sel x (hist.take (Q.kA x))))).next (Q.xB x (Q.sel x (hist.take (Q.kA x))))
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
def Seq.sim {WA WB : Type} (Q : Seq WA WB) (SA SB : Simulator) : Simulator where
  C x := SA.C x × ((u : Q.U x) → SB.C (Q.xB x u))
  prog x c := spBind (SA.prog x c.1) fun τA =>
    if goesOn (Q.kA x) τA then
      spBind (spUnder (τA.map Prod.fst) (SB.prog (Q.xB x (Q.sel x (histOf τA))) (c.2 (Q.sel x (histOf τA)))))
        fun τB => .ret (τA ++ τB)
    else .ret τA

theorem Seq.sim_out {WA WB Θ : Type} (Q : Seq WA WB) (SA SB : Simulator) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ)
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
theorem Seq.msgs_eq {WA WB ΩA ΩB Θ : Type} (Q : Seq WA WB) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB)
    (w : WA) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB nB : Msg → ℕ)
    (hk : ∀ u, kB (Q.xB x u) = nB x) (hV : Vs.k x = Q.kA x + nB x) (ωA : ΩA) (ωB : ΩB) :
    (runStar H (Q.prover PA PB w) Vs x (ωA, ωB) θ).msgs =
      cont (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs
        (runStar H (PB (Q.wB w x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
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
theorem Seq.pLog_A {WA WB ΩA ΩB Θ : Type} (Q : Seq WA WB) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB)
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
theorem Seq.pLog_B {WA WB ΩA ΩB Θ : Type} (Q : Seq WA WB) (PA : WA → Prover ΩA) (PB : WB → Prover ΩB)
    (w : WA) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB nB : Msg → ℕ)
    (hk : ∀ u, kB (Q.xB x u) = nB x) (hV : Vs.k x = Q.kA x + nB x) (ωA : ΩA) (ωB : ΩB)
    (hgo : goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs = true) :
    ∀ e ∈ (runStar H (PB (Q.wB w x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
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
        histOf ((runStar H (PB (Q.wB w x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x kB) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))) ωB
          (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)).msgs.take i)).length < Q.kA x := by
      rw [List.length_append, hl]; omega
    simp only [Seq.prover, hn, ite_false]
    rw [← hl, List.take_left, List.drop_left]
    exact he

end Claims.ZKMal
