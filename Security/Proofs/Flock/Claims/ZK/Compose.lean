import Proofs.Flock.Claims.ZK.Interact
import Proofs.Flock.Claims.ZK.Rewind
import Proofs.Flock.Claims.ZK.Indexed

/-!
# Sequential composition of `ZKmalR`

Two protocols run in turn: `A` at the statement `x`, then `B` at a statement `Seq.xB x u` that a selection `u`, read
from `A`'s history, picks. `zkmalR_seqU` proves the composed protocol `ZKmalR` from `A`'s and from `B`'s at each unit
(`ZKmalAt`), with one coin type for `B`'s simulator at every unit, so the units need not be finitely many
(Goldreich–Oren's sequential composition, with the hybrids in Wee's order): the hybrid that changes `B` keeps `A` real,
so `B`'s `Fork` is charged against a verifier whose tape is `A`'s real run, and the one that changes `A` runs `B`'s
simulator after it, a function of `A`'s transcript and fresh coins. `zkmalR_seq` is its corollary for finitely many
units and `B` one `ZKmalR` protocol, whose simulator draws coins for every unit.

The composed tape is `forkTape a`: a guess `j < a + 1`, both nonces, and two rests for `A`, of which `A` reads the
second when `j < a` and the first when `j = a`. `Fork` then shares `A`'s run at `j = a`, so its collision is `B`'s at
`A`'s real run, and at `j < a` it is `A`'s: the composed `Fork` is at least `(a·Fork_A + Fork_B)/(a + 1)`, and `γ` adds
up to `(a + 1)·g` where `γ_A ≤ a·g` and `γ_B ≤ g`. `zkmalR_iter` composes `n + 1` sessions in turn, the
`j + 1`-th at the unit the first `j + 1`'s history selects, at `a = c + j` where the first has `γ₀ ≤ c·g`:
`ε₀ + Σ_{j < n} eB j` and `(c + n)·g`, so `n·g` after a first protocol that rewinds nothing (`zkmalR_iter_const`:
`ε₀ + n·eB` where the later sessions are one protocol). A session's `ZKmalR` gives its `ZKmalAt` at each unit on
coins for every unit (`ZKmalR.at_pi`) or on coins in bijection with its simulator's there (`ZKmalR.at_equiv`).

Both are proved at one statement (`zkmalAt_seqU`, `zkmalAt_iter`, from `A`'s and `B`'s `ZKmalAt` there) and read at
every statement: nothing at `x` reads another statement, so the tapes and the number of sessions may be chosen per
statement. A session whose tape is its own type at each unit's statement is one prover on one tape at `x` through
bijections of the nonces and of the rests (`ZKmalAt.of_equiv`). `ZKmalRx` is the spec's statement-indexed form
(`Definitions/Claims/IndexedTapes.lean`; `ZKmalR` its constant case, `zkmalRx_const`), and `zkmalRx_iter` meets it with
`m x` sessions.

`zkmalAt_seqG` and `zkmalAt_iterG` are the general forms, of which `zkmalAt_seqU` and `zkmalAt_iter` are the case
`go = goesOn`: the simulator goes on where a test `go` holds, which need agree with `goesOn` only at `A`'s honest
transcripts (as a claim's own test may), and `B`'s zero knowledge is read only at the units `A`'s honest runs select
(`Seq.Reach`), so a session's rounds need be fixed only there.
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

/-- The run reads the prover only at `x` and its tape. -/
theorem stepWith_congr {Ω Ω' : Type} (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω') (x : Msg) (ω : Ω) (ω' : Ω')
    (vnext : List Msg → OC Out) (h : ∀ hist, P'.next x hist ω' = P.next x hist ω) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      stepWith H P' x ω' vnext n hist as t = stepWith H P x ω vnext n hist as t
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
      simp only [stepWith, h hist]
      split
      · exact stepWith_congr H P P' x ω ω' vnext h n _ _ _
      · rfl
      · rfl

theorem runStar_congr {Ω Ω' Θ : Type} (H : Msg → Msg) (P : Prover Ω) (P' : Prover Ω') (Vs : VStar Θ) (x : Msg)
    (ω : Ω) (ω' : Ω') (θ : Θ) (h : ∀ hist, P'.next x hist ω' = P.next x hist ω) :
    runStar H P' Vs x ω' θ = runStar H P Vs x ω θ :=
  stepWith_congr H P P' x ω ω' _ h _ _ _ _

/-! ## Two protocols in turn -/

/-- **Two protocols in turn**: `A` for `kA x` rounds at `x`, then `B` at the statement `xB x u` with witness
`wB w ωA x u`, where the unit `u` is read from `A`'s history (`sel`), and `B`'s witness may read `A`'s tape `ωA` (the
salts of what `A` committed, which `B` opens). -/
structure Seq (WA WB TA : Type) where
  kA : Msg → ℕ
  U : Msg → Type
  sel : (x : Msg) → List (Msg × Msg) → U x
  xB : (x : Msg) → U x → Msg
  wB : WA → TA → (x : Msg) → U x → WB

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

/-- A simulator program's transcript against `V*` at `θ`. -/
def spOut {Θ : Type} (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (p : SP (List (Msg × Out))) :
    List (Msg × Out) :=
  (p.eval H fun as => ((Vs.next x as θ).eval H).1).1

/-- **The composed simulator, continuing where `go` holds**: `A`'s, then, where `go` holds at its transcript, `B`'s
program at the unit that transcript selects (`progB`), asking `V*` after `A`'s messages. `B`'s coins are one type `CB x`
for every unit. `go` need agree with `goesOn` only at `A`'s real transcripts (`zkmalAt_seqG`). -/
def Seq.simG {WA WB TA : Type} (Q : Seq WA WB TA) (SA : Simulator) (CB : Msg → Type) [∀ x, Fintype (CB x)]
    [∀ x, Nonempty (CB x)] (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out)))
    (go : Msg → List (Msg × Out) → Bool) : Simulator where
  C x := SA.C x × CB x
  prog x c := spBind (SA.prog x c.1) fun τA =>
    if go x τA then
      spBind (spUnder (τA.map Prod.fst) (progB x (Q.sel x (histOf τA)) c.2)) fun τB => .ret (τA ++ τB)
    else .ret τA

/-- **The composed simulator**: `simG` where `A`'s transcript goes on. -/
def Seq.simU {WA WB TA : Type} (Q : Seq WA WB TA) (SA : Simulator) (CB : Msg → Type) [∀ x, Fintype (CB x)]
    [∀ x, Nonempty (CB x)] (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out))) : Simulator :=
  Q.simG SA CB progB fun x => goesOn (Q.kA x)

/-- **The composed simulator at `B`'s simulator**, where the units are finitely many: `B`'s coins are `SB`'s at every
unit. -/
def Seq.sim {WA WB TA : Type} (Q : Seq WA WB TA) [∀ x, Fintype (Q.U x)] [∀ x, DecidableEq (Q.U x)]
    (SA SB : Simulator) : Simulator :=
  Q.simU SA (fun x => (u : Q.U x) → SB.C (Q.xB x u)) fun x u c => SB.prog (Q.xB x u) (c u)

theorem Seq.simG_out {WA WB TA Θ : Type} (Q : Seq WA WB TA) (SA : Simulator) (CB : Msg → Type)
    [∀ x, Fintype (CB x)] [∀ x, Nonempty (CB x)] (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out)))
    (go : Msg → List (Msg × Out) → Bool) (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB : Msg → ℕ)
    (c : (Q.simG SA CB progB go).C x) :
    (Q.simG SA CB progB go).out H Vs θ x c =
      if go x (SA.out H (Vs.head Q.kA) θ x c.1) then SA.out H (Vs.head Q.kA) θ x c.1 ++
        spOut H (Vs.after x kB) (θ, (SA.out H (Vs.head Q.kA) θ x c.1).map Prod.fst)
          (Q.xB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x c.1))))
          (progB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x c.1))) c.2)
      else SA.out H (Vs.head Q.kA) θ x c.1 := by
  simp only [Simulator.out, Seq.simG, spBind_eval, VStar.head, VStar.after, spOut]
  by_cases hg : go x (SP.eval H (fun as => ((Vs.next x as θ).eval H).1) (SA.prog x c.1)).1 = true
  · simp only [hg, ite_true, spBind_eval, spUnder_eval, SP.eval]
  · simp only [hg, ite_false, Bool.false_eq_true, SP.eval]

theorem Seq.simU_out {WA WB TA Θ : Type} (Q : Seq WA WB TA) (SA : Simulator) (CB : Msg → Type)
    [∀ x, Fintype (CB x)] [∀ x, Nonempty (CB x)] (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out)))
    (H : Msg → Msg) (Vs : VStar Θ) (θ : Θ) (x : Msg) (kB : Msg → ℕ) (c : (Q.simU SA CB progB).C x) :
    (Q.simU SA CB progB).out H Vs θ x c =
      cont (Q.kA x) (SA.out H (Vs.head Q.kA) θ x c.1)
        (spOut H (Vs.after x kB) (θ, (SA.out H (Vs.head Q.kA) θ x c.1).map Prod.fst)
          (Q.xB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x c.1))))
          (progB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x c.1))) c.2)) :=
  Q.simG_out SA CB progB _ H Vs θ x kB c

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

/-- **`B`'s relation holds at every unit an honest run of `A` at `x` selects**, against every `V*`, at that run's
tape: what `B`'s statement reads of `A`'s messages (a commitment) is what `A` committed. -/
def Seq.RelOnAt {WA WB TA : Type} (Q : Seq WA WB TA) (H : Msg → Msg) (PA : WA → Prover TA) (RA : Rel WA)
    (RB : Rel WB) (x : Msg) : Prop :=
  ∀ w, RA x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (ωA : TA),
    goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs = true →
      RB (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))
        (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))

/-- `RelOnAt` at every statement. -/
def Seq.RelOn {WA WB TA : Type} (Q : Seq WA WB TA) (H : Msg → Msg) (PA : WA → Prover TA) (RA : Rel WA)
    (RB : Rel WB) : Prop :=
  ∀ x, Q.RelOnAt H PA RA RB x

/-- **A unit an honest run of `A` at `x` selects** against some `V*`, where that run goes on: the units at which
`B`'s zero knowledge is read. -/
def Seq.Reach {WA WB TA : Type} (Q : Seq WA WB TA) (H : Msg → Msg) (PA : WA → Prover TA) (RA : Rel WA) (x : Msg)
    (u : Q.U x) : Prop :=
  ∃ w, RA x w ∧ ∃ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (ωA : TA),
    goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs = true ∧
      Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs) = u

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

/-- **`ZKmalR` at one statement `x`**, at a simulator given there by its coins `C` and its program `prog`. -/
def ZKmalAt {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (k : ℕ)
    (P : W → Prover (N × Ω)) (R : W → Prop) (x : Msg) (C : Type) [Fintype C] (prog : C → SP (List (Msg × Out)))
    (ε γ : ℝ) : Prop :=
  ∀ w, R w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k → ∀ Ev : List (Msg × Out) → Prop,
    |Pr (fun ω : N × Ω => Ev (runStar H (P w) Vs x ω θ).msgs) - Pr (fun c : C => Ev (spOut H Vs θ x (prog c)))| ≤
      ε + γ * (Fork H (P w) Vs θ x).Pr H

theorem zkmalR_iff_at {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (P : W → Prover (N × Ω)) (R : Rel W) (S : Simulator) (εzk γ : Msg → ℝ) :
    ZKmalR H k P R S εzk γ ↔ ∀ x, ZKmalAt H (k x) P (R x) x (S.C x) (S.prog x) (εzk x) (γ x) :=
  Iff.rfl

/-- **`ZKmalR` at a unit, on coins for every unit**: of finitely many units with statements `xs`, `S` at `xs u` reads
its coordinate `u`. -/
theorem _root_.Claims.ZKmalR.at_pi {W N Ω U : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] [Fintype U]
    [DecidableEq U] {H : Msg → Msg} {k : Msg → ℕ} {P : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ}
    (h : ZKmalR H k P R S εzk γ) (xs : U → Msg) (u : U) :
    ZKmalAt H (k (xs u)) P (R (xs u)) (xs u) ((u' : U) → S.C (xs u')) (fun c => S.prog (xs u) (c u)) (εzk (xs u))
      (γ (xs u)) := by
  intro w hw Θ Vs θ hV Ev
  rw [Pr_pi_at (β := fun u' => S.C (xs u')) u (fun c => Ev (spOut H Vs θ (xs u) (S.prog (xs u) c)))]
  exact h (xs u) w hw Θ Vs θ hV Ev

/-- **`ZKmalR` at a statement, on coins in bijection with its simulator's there.** -/
theorem _root_.Claims.ZKmalR.at_equiv {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {k : Msg → ℕ} {P : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ}
    (h : ZKmalR H k P R S εzk γ) (xs : Msg) {C : Type} [Fintype C] (e : C ≃ S.C xs) :
    ZKmalAt H (k xs) P (R xs) xs C (fun c => S.prog xs (e c)) (εzk xs) (γ xs) := by
  intro w hw Θ Vs θ hV Ev
  rw [Pr_equiv e (fun c => Ev (spOut H Vs θ xs (S.prog xs c)))]
  exact h xs w hw Θ Vs θ hV Ev

/-- **`ZKmalAt` for a prover that runs `P` at `x` on its tape through a bijection of nonces and one of rests**: a
session whose tape at each unit's statement is its own type is, at `x`, one prover on one tape for every unit. -/
theorem ZKmalAt.of_equiv {W N Ω N' Ω' : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] [Fintype N']
    [Nonempty N'] [Fintype Ω'] [Nonempty Ω'] {H : Msg → Msg} {k : ℕ} {P : W → Prover (N × Ω)} {R : W → Prop}
    {x : Msg} {C : Type} [Fintype C] {prog : C → SP (List (Msg × Out))} {ε γ : ℝ}
    (h : ZKmalAt H k P R x C prog ε γ) (P' : W → Prover (N' × Ω')) (eN : N' ≃ N) (eΩ : Ω' ≃ Ω)
    (hP : ∀ w hist ω, (P' w).next x hist ω = (P w).next x hist (eN ω.1, eΩ ω.2)) :
    ZKmalAt H k P' R x C prog ε γ := by
  intro w hw Θ Vs θ hV Ev
  have hrun : ∀ ω : N' × Ω', runStar H (P' w) Vs x ω θ = runStar H (P w) Vs x (eN ω.1, eΩ ω.2) θ :=
    fun ω => runStar_congr H (P w) (P' w) Vs x _ ω θ fun hist => hP w hist ω
  have hr : Pr (fun ω : N' × Ω' => Ev (runStar H (P' w) Vs x ω θ).msgs) =
      Pr (fun ω : N × Ω => Ev (runStar H (P w) Vs x ω θ).msgs) := by
    simp only [hrun]
    exact Pr_equiv (Equiv.prodCongr eN eΩ) fun ω => Ev (runStar H (P w) Vs x ω θ).msgs
  have hf : (Fork H (P' w) Vs θ x).Pr H = (Fork H (P w) Vs θ x).Pr H := by
    simp only [Finder.Pr, Fork, hrun]
    exact Pr_equiv (Equiv.prodCongr eN (Equiv.prodCongr eΩ eΩ)) fun s : N × Ω × Ω => ∃ u v,
      findColl ((runStar H (P w) Vs x (s.1, s.2.1) θ).pLog ++ (runStar H (P w) Vs x (s.1, s.2.2) θ).pLog) =
        some (u, v) ∧ Collision H u v
  rw [hr, hf]
  exact h w hw Θ Vs θ hV Ev

/-- **Sequential composition at one statement, continuing where `go` holds.** At `x`, `A` then `B` at the unit `A`'s
history selects is `ZKmalAt` with the simulator that runs `SA`, then, where `go` holds at its transcript, `B`'s program
at that unit on coins of their own (`simG`): `ε` adds up, with `B`'s at every unit `A`'s honest runs select at most
`eB` (`Reach`), and so does `γ` (`γ_A ≤ a·g`, `γ_B ≤ g` there), to `(a + 1)·g` against the composed `Fork`. `go` need
agree with `goesOn` only at `A`'s honest transcripts (`hgo`), so the simulator may test whether to go on as the claim's
does. The composed tape is `forkTape a`. `B`'s hybrid keeps `A` real and gives `V*` `A`'s messages as its tape
(`Vs.after`): `B`'s zero knowledge is against every tape, chosen after `H`, at the witness `A`'s tape gives, where `B`'s
relation holds (`RelOnAt`). `A`'s runs `B`'s program after `A`'s transcript, a function of that transcript and coins
independent of `A`'s. The units need not be finitely many. Nothing is read at another statement, so the tapes may be
chosen per statement. -/
theorem zkmalAt_seqG {WA WB NA ΩA NB ΩB : Type} [Fintype NA] [Nonempty NA] [Fintype ΩA] [Nonempty ΩA]
    [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (Q : Seq WA WB (NA × ΩA)) (H : Msg → Msg) (x : Msg) (nB : Msg → ℕ)
    (PA : WA → Prover (NA × ΩA)) (PB : WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : Rel WB) (SA : Simulator)
    (CB : Msg → Type) [∀ x, Fintype (CB x)] [∀ x, Nonempty (CB x)]
    (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out))) (go : Msg → List (Msg × Out) → Bool)
    (εA γA eB g : ℝ) (a : ℕ)
    (hA : ZKmalAt H (Q.kA x) PA (RA x) x (SA.C x) (SA.prog x) εA γA)
    (hB : ∀ u, Q.Reach H PA RA x u → ZKmalAt H (nB x) PB (RB (Q.xB x u)) (Q.xB x u) (CB x) (progB x u) eB g)
    (hR : Q.RelOnAt H PA RA RB x)
    (hgo : ∀ w, RA x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (ωA : NA × ΩA),
      go x (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs =
        goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)
    (he0 : 0 ≤ eB) (hγA : γA ≤ a * g) (hg : 0 ≤ g) :
    ZKmalAt H (Q.kA x + nB x) (fun w => (Q.prover PA PB w).comap (forkTape a)) (RA x) x
      ((Q.simG SA CB progB go).C x) ((Q.simG SA CB progB go).prog x) (εA + eB) ((a + 1) * g) := by
  classical
  intro w hw Θ Vs θ hV Ev
  have hreal : Pr (fun ω => Ev (runStar H ((Q.prover PA PB w).comap (forkTape a)) Vs x ω θ).msgs) =
      Pr (fun s : (NA × ΩA) × (NB × ΩB) => Ev (cont (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs
        (runStar H (PB (Q.wB w s.1 x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs))))
          (Vs.after x fun _ => nB x) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs))) s.2
          (θ, (runStar H (PA w) (Vs.head Q.kA) x s.1 θ).msgs.map Prod.fst)).msgs)) := by
    simp only [runStar_comap]
    refine (Pr_forkTape a fun t => Ev (runStar H (Q.prover PA PB w) Vs x t θ).msgs).trans (Pr_congr fun t => ?_)
    obtain ⟨ωA, ωB⟩ := t
    rw [Q.msgs_eq PA PB w H Vs θ x (fun _ => nB x) nB (fun _ => rfl) hV ωA ωB]
  have hsim : Pr (fun c : (Q.simG SA CB progB go).C x => Ev (spOut H Vs θ x ((Q.simG SA CB progB go).prog x c))) =
      Pr (fun s : SA.C x × CB x => Ev (if go x (SA.out H (Vs.head Q.kA) θ x s.1) then
        SA.out H (Vs.head Q.kA) θ x s.1 ++
          spOut H (Vs.after x fun _ => nB x) (θ, (SA.out H (Vs.head Q.kA) θ x s.1).map Prod.fst)
            (Q.xB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x s.1))))
            (progB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x s.1))) s.2)
        else SA.out H (Vs.head Q.kA) θ x s.1)) := by
    refine Pr_congr fun c => ?_
    change Ev ((Q.simG SA CB progB go).out H Vs θ x c) ↔ _
    rw [Q.simG_out SA CB progB go H Vs θ x (fun _ => nB x) c]
  rw [hreal, hsim]
  have hF := Q.fork_ge PA PB w H Vs θ x (fun _ => nB x) nB (fun _ => rfl) hV a
  refine (abs_sub_le_hybrid (TA := NA × ΩA) (TB := NB × ΩB) (CA := SA.C x)
    (CB := CB x)
    (fun ωA ωB => Ev (cont (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs
      (runStar H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
        (Vs.after x fun _ => nB x) (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))) ωB
        (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)).msgs))
    (fun ωA cB => Ev (if go x (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs then
      (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs ++
        spOut H (Vs.after x fun _ => nB x) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)
          (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))
          (progB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)) cB)
      else (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))
    (fun cA cB => Ev (if go x (SA.out H (Vs.head Q.kA) θ x cA) then
      SA.out H (Vs.head Q.kA) θ x cA ++
        spOut H (Vs.after x fun _ => nB x) (θ, (SA.out H (Vs.head Q.kA) θ x cA).map Prod.fst)
          (Q.xB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x cA))))
          (progB x (Q.sel x (histOf (SA.out H (Vs.head Q.kA) θ x cA))) cB)
      else SA.out H (Vs.head Q.kA) θ x cA))
    (fun ωA => eB + (if goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs then
        (Fork H (PB (Q.wB w ωA x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs))))
          (Vs.after x fun _ => nB x) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst)
          (Q.xB x (Q.sel x (histOf (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs)))).Pr H else 0) * g)
    (εA + γA * (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H) ?_ ?_).trans ?_
  · -- `B`'s hybrid, at `A`'s real run
    intro ωA
    rw [hgo w hw Θ Vs θ ωA]
    by_cases hgo' : goesOn (Q.kA x) (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs = true
    · simp only [cont, hgo', ite_true]
      refine (hB _ ⟨w, hw, Θ, Vs, θ, ωA, hgo', rfl⟩ _ (hR w hw Θ Vs θ ωA hgo') (Θ × List Msg)
        (Vs.after x fun _ => nB x) (θ, (runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs.map Prod.fst) rfl
        (fun τ => Ev ((runStar H (PA w) (Vs.head Q.kA) x ωA θ).msgs ++ τ))).trans (le_of_eq ?_)
      ring
    · simp only [cont, hgo', ite_false, Bool.false_eq_true, Pr_const, sub_self, abs_zero, zero_mul, add_zero]
      exact he0
  · -- `A`'s hybrid, with `B`'s program after it
    intro cB
    exact hA w hw Θ (Vs.head Q.kA) θ rfl fun τ => Ev (if go x τ then τ ++
      spOut H (Vs.after x fun _ => nB x) (θ, τ.map Prod.fst) (Q.xB x (Q.sel x (histOf τ)))
        (progB x (Q.sel x (histOf τ)) cB) else τ)
  · rw [Ex_const_add, Ex_mul_const]
    have hFA := Finder.pr_nonneg H (Fork H (PA w) (Vs.head Q.kA) θ x)
    have h1 : γA * (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H ≤ a * g * (Fork H (PA w) (Vs.head Q.kA) θ x).Pr H :=
      mul_le_mul_of_nonneg_right hγA hFA
    have h2 := mul_le_mul_of_nonneg_left hF hg
    nlinarith

/-- **Sequential composition at one statement**: `zkmalAt_seqG` where the simulator goes on as `goesOn` does
(`simU`), with `B`'s zero knowledge at every unit. -/
theorem zkmalAt_seqU {WA WB NA ΩA NB ΩB : Type} [Fintype NA] [Nonempty NA] [Fintype ΩA] [Nonempty ΩA]
    [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (Q : Seq WA WB (NA × ΩA)) (H : Msg → Msg) (x : Msg) (nB : Msg → ℕ)
    (PA : WA → Prover (NA × ΩA)) (PB : WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : Rel WB) (SA : Simulator)
    (CB : Msg → Type) [∀ x, Fintype (CB x)] [∀ x, Nonempty (CB x)]
    (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out))) (εA γA eB g : ℝ) (a : ℕ)
    (hA : ZKmalAt H (Q.kA x) PA (RA x) x (SA.C x) (SA.prog x) εA γA)
    (hB : ∀ u, ZKmalAt H (nB x) PB (RB (Q.xB x u)) (Q.xB x u) (CB x) (progB x u) eB g)
    (hR : Q.RelOnAt H PA RA RB x) (he0 : 0 ≤ eB) (hγA : γA ≤ a * g) (hg : 0 ≤ g) :
    ZKmalAt H (Q.kA x + nB x) (fun w => (Q.prover PA PB w).comap (forkTape a)) (RA x) x
      ((Q.simU SA CB progB).C x) ((Q.simU SA CB progB).prog x) (εA + eB) ((a + 1) * g) :=
  zkmalAt_seqG Q H x nB PA PB RA RB SA CB progB _ εA γA eB g a hA (fun u _ => hB u) hR
    (fun _ _ _ _ _ _ => rfl) he0 hγA hg

/-- **Sequential composition of `ZKmalR`**: `zkmalAt_seqU` at every statement. -/
theorem zkmalR_seqU {WA WB NA ΩA NB ΩB : Type} [Fintype NA] [Nonempty NA] [Fintype ΩA] [Nonempty ΩA]
    [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (Q : Seq WA WB (NA × ΩA)) (H : Msg → Msg) (nB : Msg → ℕ)
    (PA : WA → Prover (NA × ΩA)) (PB : WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : Rel WB) (SA : Simulator)
    (CB : Msg → Type) [∀ x, Fintype (CB x)] [∀ x, Nonempty (CB x)]
    (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out))) (εA γA eB g : Msg → ℝ) (a : ℕ)
    (hA : ZKmalR H Q.kA PA RA SA εA γA)
    (hB : ∀ x u, ZKmalAt H (nB x) PB (RB (Q.xB x u)) (Q.xB x u) (CB x) (progB x u) (eB x) (g x))
    (hR : Q.RelOn H PA RA RB) (he0 : ∀ x, 0 ≤ eB x) (hγA : ∀ x, γA x ≤ a * g x) (hg : ∀ x, 0 ≤ g x) :
    ZKmalR H (fun x => Q.kA x + nB x) (fun w => (Q.prover PA PB w).comap (forkTape a)) RA (Q.simU SA CB progB)
      (fun x => εA x + eB x) (fun x => (a + 1) * g x) :=
  (zkmalR_iff_at _ _ _ _ _ _ _).2 fun x =>
    zkmalAt_seqU Q H x nB PA PB RA RB SA CB progB (εA x) (γA x) (eB x) (g x) a (hA x) (hB x) (hR x) (he0 x)
      (hγA x) (hg x)

/-- **Sequential composition at `B`'s simulator**, where the units are finitely many: `zkmalR_seqU` at `Seq.sim`,
whose coins for `B` are `SB`'s at every unit, from `B`'s `ZKmalR` at every unit's statement. -/
theorem zkmalR_seq {WA WB NA ΩA NB ΩB : Type} [Fintype NA] [Nonempty NA] [Fintype ΩA] [Nonempty ΩA]
    [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (Q : Seq WA WB (NA × ΩA)) [∀ x, Fintype (Q.U x)] [∀ x, DecidableEq (Q.U x)] (H : Msg → Msg) (kB nB : Msg → ℕ)
    (PA : WA → Prover (NA × ΩA)) (PB : WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : Rel WB) (SA SB : Simulator)
    (εA γA εB γB eB g : Msg → ℝ) (a : ℕ)
    (hA : ZKmalR H Q.kA PA RA SA εA γA) (hB : ZKmalR H kB PB RB SB εB γB)
    (hR : Q.RelOn H PA RA RB) (hk : ∀ x u, kB (Q.xB x u) = nB x)
    (hε : ∀ x u, εB (Q.xB x u) ≤ eB x) (he0 : ∀ x, 0 ≤ eB x)
    (hγ : ∀ x u, γB (Q.xB x u) ≤ g x) (hγA : ∀ x, γA x ≤ a * g x) (hg : ∀ x, 0 ≤ g x) :
    ZKmalR H (fun x => Q.kA x + nB x) (fun w => (Q.prover PA PB w).comap (forkTape a)) RA (Q.sim SA SB)
      (fun x => εA x + eB x) (fun x => (a + 1) * g x) := by
  refine zkmalR_seqU Q H nB PA PB RA RB SA (fun x => (u : Q.U x) → SB.C (Q.xB x u))
    (fun x u c => SB.prog (Q.xB x u) (c u)) εA γA eB g a hA (fun x u w' hw' Θ Vs θ hVk Ev => ?_) hR he0 hγA hg
  have h := hB.at_pi (Q.xB x) u w' hw' Θ Vs θ (hVk.trans (hk x u).symm) Ev
  exact h.trans (add_le_add (hε x u) (mul_le_mul_of_nonneg_right (hγ x u) (Finder.pr_nonneg _ _)))

/-! ## `n` sessions in turn -/

/-- A tape: a nonce and the rest, each finite and inhabited. -/
structure SeqTape where
  N : Type
  Ω : Type
  [fN : Fintype N]
  [nN : Nonempty N]
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]

attribute [instance] SeqTape.fN SeqTape.nN SeqTape.fΩ SeqTape.nΩ

/-- The tape of `n + 1` sessions in turn: the first's is `T₀`, and session `n + 1` adds `forkTape (c + n)`'s guess, its
own `NB × ΩB`, and a second rest for the sessions before it. -/
def SeqTape.iter (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] :
    ℕ → SeqTape
  | 0 => T₀
  | n + 1 => ⟨Fin (c + n + 1) × (SeqTape.iter T₀ c NB ΩB n).N × NB × (SeqTape.iter T₀ c NB ΩB n).Ω,
      (SeqTape.iter T₀ c NB ΩB n).Ω × ΩB⟩

/-- The first `n + 1` sessions' tape as a type. -/
abbrev SeqTape.at (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (n : ℕ) : Type :=
  (T₀.iter c NB ΩB n).N × (T₀.iter c NB ΩB n).Ω

/-- **`n + 1` sessions in turn**: `P₀`, then session `j + 1` (`PB j`) after the first `j + 1`, at the unit their
history selects (`Q j`), on `forkTape (c + j)`. -/
def iterProver {WA WB : Type} (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (P₀ : WA → Prover (T₀.at c NB ΩB 0))
    (PB : ℕ → WB → Prover (NB × ΩB)) : (n : ℕ) → WA → Prover (T₀.at c NB ΩB n)
  | 0 => P₀
  | n + 1 => fun w => ((Q n).prover (iterProver T₀ c NB ΩB Q P₀ PB n) (PB n) w).comap (forkTape (c + n))

/-- The simulator of `n + 1` sessions in turn, continuing where `go` holds: `S₀`, then, where `go j` holds at the
simulated history, session `j + 1`'s program at the unit that history selects (`Seq.simG`), on coins `CB j`. -/
def iterSimG {WA WB : Type} (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) : ℕ → Simulator
  | 0 => S₀
  | n + 1 => (Q n).simG (iterSimG T₀ c NB ΩB Q S₀ CB progB go n) (CB n) (progB n) (go n)

/-- The simulator of `n + 1` sessions in turn: `iterSimG` going on where the first `j + 1`'s simulated transcript does
(`goesOn`, `Seq.simU`). -/
def iterSim {WA WB : Type} (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out))) : ℕ → Simulator :=
  iterSimG T₀ c NB ΩB Q S₀ CB progB fun n x => goesOn ((Q n).kA x)

/-- **Sequential composition of `N + 1` sessions at one statement, continuing where `go` holds.** At `x`, the first
`ZKmalAt` with `ε₀` and `γ₀ ≤ c·g`, and session `j + 1 ≤ N` `ZKmalAt` with `eB j` and `g` on coins `CB j x` at every
unit the first `j + 1`'s honest runs select (`Reach`), where `(Q n).kA x` counts the first `n + 1` sessions' rounds
(`hk`) and `go j` agrees with `goesOn` at their honest transcripts (`hgo`): the `N + 1` in turn are `ZKmalAt` with
`ε₀ + Σ_{j < N} eB j` and `(c + N)·g`. Nothing is read of a session after the `N + 1`-th. -/
theorem zkmalAt_iterG {WA WB NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (T₀ : SeqTape)
    (c : ℕ) (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (H : Msg → Msg) (x : Msg) (nB : ℕ → Msg → ℕ)
    (P₀ : WA → Prover (T₀.at c NB ΩB 0)) (PB : ℕ → WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : ℕ → Rel WB)
    (S₀ : Simulator) (CB : ℕ → Msg → Type) [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) (ε₀ γ₀ g : ℝ) (eB : ℕ → ℝ) (N : ℕ)
    (h₀ : ZKmalAt H ((Q 0).kA x) P₀ (RA x) x (S₀.C x) (S₀.prog x) ε₀ γ₀)
    (hB : ∀ n < N, ∀ u, (Q n).Reach H (iterProver T₀ c NB ΩB Q P₀ PB n) RA x u →
      ZKmalAt H (nB n x) (PB n) (RB n ((Q n).xB x u)) ((Q n).xB x u) (CB n x) (progB n x u) (eB n) g)
    (hR : ∀ n < N, (Q n).RelOnAt H (iterProver T₀ c NB ΩB Q P₀ PB n) RA (RB n) x)
    (hgo : ∀ n < N, ∀ w, RA x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ) (ωA : T₀.at c NB ΩB n),
      go n x (runStar H (iterProver T₀ c NB ΩB Q P₀ PB n w) (Vs.head (Q n).kA) x ωA θ).msgs =
        goesOn ((Q n).kA x) (runStar H (iterProver T₀ c NB ΩB Q P₀ PB n w) (Vs.head (Q n).kA) x ωA θ).msgs)
    (hk : ∀ n < N, (Q (n + 1)).kA x = (Q n).kA x + nB n x) (hγ₀ : γ₀ ≤ c * g) (he0 : ∀ n < N, 0 ≤ eB n)
    (hg : 0 ≤ g) :
    ZKmalAt H ((Q N).kA x) (iterProver T₀ c NB ΩB Q P₀ PB N) (RA x) x
      ((iterSimG T₀ c NB ΩB Q S₀ CB progB go N).C x) ((iterSimG T₀ c NB ΩB Q S₀ CB progB go N).prog x)
      (ε₀ + ∑ j ∈ Finset.range N, eB j) ((c + N) * g) := by
  suffices h : ∀ n ≤ N, ZKmalAt H ((Q n).kA x) (iterProver T₀ c NB ΩB Q P₀ PB n) (RA x) x
      ((iterSimG T₀ c NB ΩB Q S₀ CB progB go n).C x) ((iterSimG T₀ c NB ΩB Q S₀ CB progB go n).prog x)
      (ε₀ + ∑ j ∈ Finset.range n, eB j) ((c + n) * g) from h N le_rfl
  intro n
  induction n with
  | zero =>
      intro _ w hw Θ Vs θ hV Ev
      refine (h₀ w hw Θ Vs θ hV Ev).trans ?_
      simp only [Finset.range_zero, Finset.sum_empty, Nat.cast_zero, add_zero]
      exact add_le_add le_rfl (mul_le_mul_of_nonneg_right hγ₀ (Finder.pr_nonneg _ _))
  | succ n ih =>
      intro hn
      have hnN : n < N := by omega
      have h := zkmalAt_seqG (Q n) H x (nB n) (iterProver T₀ c NB ΩB Q P₀ PB n) (PB n) RA (RB n)
        (iterSimG T₀ c NB ΩB Q S₀ CB progB go n) (CB n) (progB n) (go n) (ε₀ + ∑ j ∈ Finset.range n, eB j)
        ((c + n) * g) (eB n) g (c + n) (ih hnN.le) (hB n hnN) (hR n hnN) (hgo n hnN) (he0 n hnN)
        (by push_cast; exact le_rfl) hg
      have eγ : ((c + n : ℕ) : ℝ) + 1 = (c : ℝ) + ((n + 1 : ℕ) : ℝ) := by push_cast; ring
      rw [eγ, ← hk n hnN] at h
      rw [Finset.sum_range_succ, ← add_assoc]
      exact h

/-- **Sequential composition of `n + 1` sessions at one statement.** At `x`, the first `ZKmalAt` with `ε₀` and
`γ₀ ≤ c·g`, and session `j + 1` `ZKmalAt` every unit with `eB j` and `g` on coins `CB j x`, where `(Q n).kA x` counts
the first `n + 1` sessions' rounds (`hk`): the `n + 1` in turn are `ZKmalAt` with `ε₀ + Σ_{j < n} eB j` and
`(c + n)·g` (`zkmalAt_iterG` at `goesOn`). A first protocol that rewinds nothing (`γ₀ = 0`) takes `c = 0`, and `n`
sessions after it cost `n·g`. Nothing is read at another statement, so `T₀`, `NB`, `ΩB` and `n` may be chosen per
statement. -/
theorem zkmalAt_iter {WA WB NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (T₀ : SeqTape)
    (c : ℕ) (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (H : Msg → Msg) (x : Msg) (nB : ℕ → Msg → ℕ)
    (P₀ : WA → Prover (T₀.at c NB ΩB 0)) (PB : ℕ → WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : ℕ → Rel WB)
    (S₀ : Simulator) (CB : ℕ → Msg → Type) [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out))) (ε₀ γ₀ g : ℝ) (eB : ℕ → ℝ)
    (h₀ : ZKmalAt H ((Q 0).kA x) P₀ (RA x) x (S₀.C x) (S₀.prog x) ε₀ γ₀)
    (hB : ∀ n u, ZKmalAt H (nB n x) (PB n) (RB n ((Q n).xB x u)) ((Q n).xB x u) (CB n x) (progB n x u) (eB n) g)
    (hR : ∀ n, (Q n).RelOnAt H (iterProver T₀ c NB ΩB Q P₀ PB n) RA (RB n) x)
    (hk : ∀ n, (Q (n + 1)).kA x = (Q n).kA x + nB n x) (hγ₀ : γ₀ ≤ c * g) (he0 : ∀ n, 0 ≤ eB n) (hg : 0 ≤ g)
    (n : ℕ) :
    ZKmalAt H ((Q n).kA x) (iterProver T₀ c NB ΩB Q P₀ PB n) (RA x) x ((iterSim T₀ c NB ΩB Q S₀ CB progB n).C x)
      ((iterSim T₀ c NB ΩB Q S₀ CB progB n).prog x) (ε₀ + ∑ j ∈ Finset.range n, eB j) ((c + n) * g) :=
  zkmalAt_iterG T₀ c Q H x nB P₀ PB RA RB S₀ CB progB _ ε₀ γ₀ g eB n h₀ (fun n _ u _ => hB n u)
    (fun n _ => hR n) (fun _ _ _ _ _ _ _ _ => rfl) (fun n _ => hk n) hγ₀ (fun n _ => he0 n) hg

/-- **Sequential composition of `n + 1` sessions**: `zkmalAt_iter` at every statement. -/
theorem zkmalR_iter {WA WB NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (T₀ : SeqTape)
    (c : ℕ) (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (H : Msg → Msg) (nB : ℕ → Msg → ℕ)
    (P₀ : WA → Prover (T₀.at c NB ΩB 0)) (PB : ℕ → WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : ℕ → Rel WB)
    (S₀ : Simulator) (CB : ℕ → Msg → Type) [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out))) (ε₀ γ₀ g : Msg → ℝ)
    (eB : ℕ → Msg → ℝ) (h₀ : ZKmalR H (Q 0).kA P₀ RA S₀ ε₀ γ₀)
    (hB : ∀ n x u, ZKmalAt H (nB n x) (PB n) (RB n ((Q n).xB x u)) ((Q n).xB x u) (CB n x) (progB n x u) (eB n x)
      (g x))
    (hR : ∀ n, (Q n).RelOn H (iterProver T₀ c NB ΩB Q P₀ PB n) RA (RB n))
    (hk : ∀ n x, (Q (n + 1)).kA x = (Q n).kA x + nB n x) (hγ₀ : ∀ x, γ₀ x ≤ c * g x) (he0 : ∀ n x, 0 ≤ eB n x)
    (hg : ∀ x, 0 ≤ g x) (n : ℕ) :
    ZKmalR H (Q n).kA (iterProver T₀ c NB ΩB Q P₀ PB n) RA (iterSim T₀ c NB ΩB Q S₀ CB progB n)
      (fun x => ε₀ x + ∑ j ∈ Finset.range n, eB j x) (fun x => (c + n) * g x) :=
  (zkmalR_iff_at _ _ _ _ _ _ _).2 fun x =>
    zkmalAt_iter T₀ c Q H x nB P₀ PB RA RB S₀ CB progB (ε₀ x) (γ₀ x) (g x) (fun j => eB j x) (h₀ x)
      (fun n u => hB n x u) (fun n => hR n x) (fun n => hk n x) (hγ₀ x) (fun n => he0 n x) (hg x) n

/-- `zkmalR_iter` where every later session is one protocol with `eB` at every unit: `ε₀ + n·eB`. -/
theorem zkmalR_iter_const {WA WB NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (T₀ : SeqTape) (c : ℕ) (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (H : Msg → Msg) (nB : Msg → ℕ)
    (P₀ : WA → Prover (T₀.at c NB ΩB 0)) (PB : WB → Prover (NB × ΩB)) (RA : Rel WA) (RB : Rel WB)
    (S₀ : Simulator) (CB : Msg → Type) [∀ x, Fintype (CB x)] [∀ x, Nonempty (CB x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB x → SP (List (Msg × Out))) (ε₀ γ₀ eB g : Msg → ℝ)
    (h₀ : ZKmalR H (Q 0).kA P₀ RA S₀ ε₀ γ₀)
    (hB : ∀ n x u, ZKmalAt H (nB x) PB (RB ((Q n).xB x u)) ((Q n).xB x u) (CB x) (progB n x u) (eB x) (g x))
    (hR : ∀ n, (Q n).RelOn H (iterProver T₀ c NB ΩB Q P₀ (fun _ => PB) n) RA RB)
    (hk : ∀ n x, (Q (n + 1)).kA x = (Q n).kA x + nB x) (hγ₀ : ∀ x, γ₀ x ≤ c * g x) (he0 : ∀ x, 0 ≤ eB x)
    (hg : ∀ x, 0 ≤ g x) (n : ℕ) :
    ZKmalR H (Q n).kA (iterProver T₀ c NB ΩB Q P₀ (fun _ => PB) n) RA
      (iterSim T₀ c NB ΩB Q S₀ (fun _ => CB) progB n) (fun x => ε₀ x + n * eB x) (fun x => (c + n) * g x) := by
  have h := zkmalR_iter T₀ c Q H (fun _ => nB) P₀ (fun _ => PB) RA (fun _ => RB) S₀ (fun _ => CB) progB ε₀ γ₀ g
    (fun _ => eB) h₀ hB hR hk hγ₀ (fun _ => he0) hg n
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at h
  exact h

/-! ## Statement-indexed tapes -/

theorem zkmalRx_iff_at {W : Type} (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x))
    (R : Rel W) (S : Simulator) (εzk γ : Msg → ℝ) :
    ZKmalRx N Ω H k P R S εzk γ ↔ ∀ x, ZKmalAt H (k x) (fun w => P w x) (R x) x (S.C x) (S.prog x) (εzk x) (γ x) :=
  Iff.rfl

/-- The simulator of `m x + 1` sessions at each `x`, on `x`'s tapes. -/
def iterSimX {WA WB : Type} (T₀ : Msg → SeqTape) (c : ℕ) (NB ΩB : Msg → Type) [∀ x, Fintype (NB x)]
    [∀ x, Nonempty (NB x)] [∀ x, Fintype (ΩB x)] [∀ x, Nonempty (ΩB x)]
    (Q : (x : Msg) → (n : ℕ) → Seq WA WB ((T₀ x).at c (NB x) (ΩB x) n)) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (x : Msg) → (n : ℕ) → (x' : Msg) → (Q x n).U x' → CB n x' → SP (List (Msg × Out))) (m : Msg → ℕ) :
    Simulator where
  C x := (iterSim (T₀ x) c (NB x) (ΩB x) (Q x) S₀ CB (progB x) (m x)).C x
  prog x := (iterSim (T₀ x) c (NB x) (ΩB x) (Q x) S₀ CB (progB x) (m x)).prog x

/-- **`m x + 1` sessions in turn on statement-indexed tapes**: at each `x`, `zkmalAt_iter` on `x`'s tapes with
`m x` sessions after the first, so the number of sessions and every tape may grow with the statement. -/
theorem zkmalRx_iter {WA WB : Type} (T₀ : Msg → SeqTape) (c : ℕ) (NB ΩB : Msg → Type) [∀ x, Fintype (NB x)]
    [∀ x, Nonempty (NB x)] [∀ x, Fintype (ΩB x)] [∀ x, Nonempty (ΩB x)]
    (Q : (x : Msg) → (n : ℕ) → Seq WA WB ((T₀ x).at c (NB x) (ΩB x) n)) (H : Msg → Msg) (nB : ℕ → Msg → ℕ)
    (m : Msg → ℕ) (P₀ : WA → (x : Msg) → Prover ((T₀ x).N × (T₀ x).Ω))
    (PB : (x : Msg) → ℕ → WB → Prover (NB x × ΩB x)) (RA : Rel WA) (RB : ℕ → Rel WB) (S₀ : Simulator)
    (CB : ℕ → Msg → Type) [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (x : Msg) → (n : ℕ) → (x' : Msg) → (Q x n).U x' → CB n x' → SP (List (Msg × Out)))
    (ε₀ γ₀ g : Msg → ℝ) (eB : ℕ → Msg → ℝ)
    (h₀ : ZKmalRx (fun x => (T₀ x).N) (fun x => (T₀ x).Ω) H (fun x => (Q x 0).kA x) P₀ RA S₀ ε₀ γ₀)
    (hB : ∀ x n u, ZKmalAt H (nB n x) (PB x n) (RB n ((Q x n).xB x u)) ((Q x n).xB x u) (CB n x) (progB x n x u)
      (eB n x) (g x))
    (hR : ∀ x n, (Q x n).RelOnAt H (iterProver (T₀ x) c (NB x) (ΩB x) (Q x) (fun w => P₀ w x) (PB x) n) RA (RB n) x)
    (hk : ∀ x n, (Q x (n + 1)).kA x = (Q x n).kA x + nB n x) (hγ₀ : ∀ x, γ₀ x ≤ c * g x)
    (he0 : ∀ n x, 0 ≤ eB n x) (hg : ∀ x, 0 ≤ g x) :
    ZKmalRx (fun x => ((T₀ x).iter c (NB x) (ΩB x) (m x)).N) (fun x => ((T₀ x).iter c (NB x) (ΩB x) (m x)).Ω) H
      (fun x => (Q x (m x)).kA x)
      (fun w x => iterProver (T₀ x) c (NB x) (ΩB x) (Q x) (fun w => P₀ w x) (PB x) (m x) w) RA
      (iterSimX T₀ c NB ΩB Q S₀ CB progB m) (fun x => ε₀ x + ∑ j ∈ Finset.range (m x), eB j x)
      (fun x => (c + m x) * g x) :=
  fun x => zkmalAt_iter (T₀ x) c (Q x) H x nB (fun w => P₀ w x) (PB x) RA RB S₀ CB (progB x) (ε₀ x) (γ₀ x) (g x)
    (fun j => eB j x) (h₀ x) (hB x) (hR x) (hk x) (hγ₀ x) (fun n => he0 n x) (hg x) (m x)

end Claims.ZKMal
