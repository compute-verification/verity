module

public import Proofs.Flock.RecGeneric.ZKSessions
public import Proofs.Flock.RecGeneric.ZKCost

/-!
# Z1-S's proof

The composed run of the `m` sessions from session `i`, against `V*` after the prover messages `pre` (`Sessions.lrun`), is
session `i`'s run at its statement against `V*` after `pre` (`Sessions.sRun`), then, where that goes on, the rest's after
its messages (`lrun_succ`). The composed outcome reads session `i`'s outcome off its run and the rest's off theirs
(`outs_cont`), and the composed simulator is session `i`'s simulator given its outcome, then the rest's where its run goes
on (`csim_succ`).

The hybrid (`hybrid`) is an induction on the number of sessions left. At each tape of session `i` whose real run shows
its outcome and goes on, the rest's real runs are bounded by the rest's simulator plus the rest's error (the induction).
Averaged over session `i`'s tape at each of the rest's simulator coins, session `i`'s zero knowledge, against `V*` after
the messages so far, swaps its real run for its simulator's at its error. The errors add exactly (`Sessions.errAt`).

`Flock.SecurityProofs.GenericSessionsZK` is Z1-S and `Flock.SecurityProofs.GenericSessionsZKCost` Z1-S-cost; Z1 and
Z1-cost at a release whose outer phase is the sessions follow (`Flock.SecurityProofs.GenericRecursiveSessionsZK`,
`Flock.SecurityProofs.GenericRecursiveSessionsZKCost`). Z1-L (`Flock.SecurityProofs.GenericSessionsOfZKOut`) is exact:
uniform tapes for every statement in `Xs x`, read at one statement, are a uniform tape of that statement (`Pr_pi_eval`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

theorem Pr_const_eq {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] (P : Prop) :
    Pr (fun _ : α => P) = Pr (fun _ : β => P) := by
  classical
  by_cases h : P
  · simp [Pr, Claims.prUniform, h, Fintype.card_ne_zero]
  · simp [Pr, Claims.prUniform, h]

theorem Ex_Pr_comm {α β : Type} [Fintype α] [Fintype β] (q : α → β → Prop) :
    Ex (fun a => Pr fun b => q a b) = Ex (fun b => Pr fun a => q a b) :=
  calc Ex (fun a => Pr fun b => q a b) = Pr (fun p : α × β => q p.1 p.2) :=
        (ZKMal.Pr_prod_left (fun p : α × β => q p.1 p.2)).symm
    _ = Pr (fun p : β × α => q p.2 p.1) := (ZKMal.Pr_equiv (Equiv.prodComm β α) (fun p : α × β => q p.1 p.2)).symm
    _ = Ex (fun b => Pr fun a => q a b) := ZKMal.Pr_prod_left (fun p : β × α => q p.2 p.1)

theorem Pr_pi_eval {ι : Type} [Fintype ι] [DecidableEq ι] {β : ι → Type} [∀ i, Fintype (β i)]
    [∀ i, Nonempty (β i)] (i : ι) (p : β i → Prop) : Pr (fun c : (j : ι) → β j => p (c i)) = Pr p :=
  (ZKMal.Pr_equiv (Equiv.piSplitAt i β) fun s => p s.1).trans (Pr_fst p)

/-- `V*`'s answers at `x_O` after the prover messages `pre`. -/
def askAfter (H : Msg → Msg) {Θ : Type} (Vs : VStar Θ) (xO : Msg) (θ : Θ) (pre : List Msg) : List Msg → Out :=
  fun as => ((Vs.next xO (pre ++ as) θ).eval H).1

namespace Sessions

variable {S : ZkSess} (L : Sessions S)

/-! ## Runs -/

/-- Session `i`'s run at its statement, its witness and its tape `t`, against `V*` after the prover messages `pre`. -/
def sRun (H : Msg → Msg) (x xO : Msg) (wo : ℕ → S.WS) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (pre : List Msg) (i : ℕ)
    (t : S.NS x × S.ΩS x) : List (Msg × Out) :=
  (runStar H (S.PS (wo i) x) (vUnder Vs xO pre (S.kS (L.xs x xO i))) (L.xs x xO i) t θ).msgs

/-- The composed run of the `m` sessions from session `i` on their tapes `t`, against `V*` after the prover messages
`pre`. -/
def lrun (H : Msg → Msg) (x xO : Msg) (wo : ℕ → S.WS) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (pre : List Msg) (m i : ℕ)
    (t : S.tapes x m) : List (Msg × Out) :=
  Claims.interact (askAfter H Vs xO θ pre) (fun h => ((L.cnext x xO wo m i h t).eval H).1) (L.rounds x xO m i) []

theorem sRun_length (H : Msg → Msg) (x xO : Msg) (wo : ℕ → S.WS) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (pre : List Msg)
    (i : ℕ) (t : S.NS x × S.ΩS x) : (L.sRun H x xO wo Vs θ pre i t).length ≤ S.kS (L.xs x xO i) := by
  unfold sRun
  rw [ZKMal.runStar_msgs]
  exact ZKMal.interact_length_le _ _ _ _

theorem lrun_zero (H : Msg → Msg) (x xO : Msg) (wo : ℕ → S.WS) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (pre : List Msg)
    (i : ℕ) (t : S.tapes x 0) : L.lrun H x xO wo Vs θ pre 0 i t = [] := rfl

/-- **The composed run is session `i`'s, then the rest's** after its messages where it goes on. -/
theorem lrun_succ (H : Msg → Msg) (x xO : Msg) (wo : ℕ → S.WS) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (pre : List Msg)
    (m i : ℕ) (tA : S.NS x × S.ΩS x) (tB : S.tapes x m) :
    L.lrun H x xO wo Vs θ pre (m + 1) i (tA, tB) =
      ZKMal.cont (S.kS (L.xs x xO i)) (L.sRun H x xO wo Vs θ pre i tA)
        (L.lrun H x xO wo Vs θ (pre ++ (L.sRun H x xO wo Vs θ pre i tA).map Prod.fst) m (i + 1) tB) := by
  have hA : Claims.interact (askAfter H Vs xO θ pre)
      (fun h => ((L.cnext x xO wo (m + 1) i h (tA, tB)).eval H).1) (S.kS (L.xs x xO i)) [] =
      L.sRun H x xO wo Vs θ pre i tA := by
    unfold sRun
    rw [ZKMal.runStar_msgs]
    refine interact_len_congr _ _ _ (S.kS (L.xs x xO i)) [] fun h hl => ?_
    simp only [List.length_nil, Nat.zero_add] at hl
    simp only [cnext, ite_eq_left hl]
  unfold lrun
  rw [show L.rounds x xO (m + 1) i = S.kS (L.xs x xO i) + L.rounds x xO m (i + 1) from rfl, ZKMal.interact_add, hA,
    List.nil_append]
  unfold ZKMal.cont
  by_cases hg : ZKMal.goesOn (S.kS (L.xs x xO i)) (L.sRun H x xO wo Vs θ pre i tA) = true
  · rw [ite_eq_left hg, ite_eq_left hg]
    have hl : (Claims.histOf (L.sRun H x xO wo Vs θ pre i tA)).length = S.kS (L.xs x xO i) := by
      rw [ZKMal.histOf_length]; exact ZKMal.goesOn_length hg
    rw [← List.append_nil (Claims.histOf (L.sRun H x xO wo Vs θ pre i tA)), ZKMal.interact_shift,
      ZKMal.histOf_map_fst]
    congr 2
    · funext as
      simp only [askAfter, List.append_assoc]
    · funext h'
      have hn : ¬ (Claims.histOf (L.sRun H x xO wo Vs θ pre i tA) ++ h').length < S.kS (L.xs x xO i) := by
        rw [List.length_append]; omega
      simp only [cnext, ite_eq_right hn, List.drop_left' hl]
  · rw [ite_eq_right hg, ite_eq_right hg]

/-- **The composed outcome after session `i`'s run**: session `i`'s outcome, then the rest's where its run goes on. -/
theorem outs_cont (x xO : Msg) (m i : ℕ) (τA τB : List (Msg × Out)) (hA : τA.length ≤ S.kS (L.xs x xO i)) :
    L.outs x xO (m + 1) i (ZKMal.cont (S.kS (L.xs x xO i)) τA τB) =
      some (S.outS (L.xs x xO i) τA) ::
        if ZKMal.goesOn (S.kS (L.xs x xO i)) τA then L.outs x xO m (i + 1) τB else List.replicate m none := by
  unfold ZKMal.cont
  by_cases hg : ZKMal.goesOn (S.kS (L.xs x xO i)) τA = true
  · have hl := ZKMal.goesOn_length hg
    rw [ite_eq_left hg, outs, List.take_left' hl, List.drop_left' hl, ite_eq_left hg]
  · rw [ite_eq_right hg, outs, List.take_of_length_le hA, ite_eq_right hg, ite_eq_right hg]

/-- **The composed simulator from session `i`**: session `i`'s simulator given its outcome, against `V*` after the
messages so far, then the rest's where its run goes on. -/
theorem csim_succ (H : Msg → Msg) (x xO : Msg) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (m i : ℕ) (a : S.StS)
    (os : List (Option S.StS)) (ms : List (Msg × Out)) (c : S.coins x (m + 1)) :
    (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) (L.csim x xO (m + 1) i (some a :: os) ms c)).1 =
      if ZKMal.goesOn (S.kS (L.xs x xO i)) (((S.simAt x).given a).out H
          (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) θ (L.xs x xO i) c.1) then
        (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) (L.csim x xO m (i + 1) os
          (ms ++ ((S.simAt x).given a).out H (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) θ
            (L.xs x xO i) c.1) c.2)).1
      else ms ++ ((S.simAt x).given a).out H (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) θ
        (L.xs x xO i) c.1 := by
  rw [csim, spBind_eval_fst, ZKMal.recSpUnder_eq, ZKMal.spUnder_eval]
  by_cases hg : ZKMal.goesOn (S.kS (L.xs x xO i)) (((S.simAt x).given a).out H
      (vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))) θ (L.xs x xO i) c.1) = true
  · rw [ite_eq_left hg]
    exact congrArg (fun p => (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) p).1) (ite_eq_left hg)
  · rw [ite_eq_right hg]
    exact congrArg (fun p => (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) p).1) (ite_eq_right hg)

/-! ## The hybrid -/

/-- **The real event after session `i`'s run**: session `i`'s outcome is `a`, and where its run goes on, the rest's
outcomes and the event after it, and otherwise none of the rest and the event at it. -/
theorem ev_succ (H : Msg → Msg) (x xO : Msg) (wo : ℕ → S.WS) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (m i : ℕ)
    (ms : List (Msg × Out)) (a : S.StS) (os : List (Option S.StS)) (Ev : List (Msg × Out) → Prop)
    (tA : S.NS x × S.ΩS x) (tB : S.tapes x m) :
    (L.outs x xO (m + 1) i (L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i (tA, tB)) = some a :: os ∧
        Ev (ms ++ L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i (tA, tB))) ↔
      S.outS (L.xs x xO i) (L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA) = a ∧
        if ZKMal.goesOn (S.kS (L.xs x xO i)) (L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA) = true then
          L.outs x xO m (i + 1) (L.lrun H x xO wo Vs θ ((ms ++ L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA).map
            Prod.fst) m (i + 1) tB) = os ∧
          Ev ((ms ++ L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA) ++ L.lrun H x xO wo Vs θ
            ((ms ++ L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA).map Prod.fst) m (i + 1) tB)
        else List.replicate m none = os ∧ Ev (ms ++ L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA) := by
  rw [L.lrun_succ H x xO wo Vs θ _ m i tA tB, L.outs_cont x xO m i _ _ (L.sRun_length H x xO wo Vs θ _ i tA),
    List.cons.injEq, Option.some.injEq, List.map_append]
  unfold ZKMal.cont
  by_cases hg : ZKMal.goesOn (S.kS (L.xs x xO i)) (L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA) = true
  · rw [ite_eq_left hg, ite_eq_left hg, ite_eq_left hg, List.append_assoc]
    exact and_assoc
  · rw [ite_eq_right hg, ite_eq_right hg, ite_eq_right hg]
    exact and_assoc

open Classical in
/-- **The hybrid over the sessions in turn**: the `m` sessions from session `i`, after the transcript `ms`, against `V*`
after its messages, are zero knowledge per outcome within `errAt`, on witnesses in `RS x` at those sessions. -/
theorem hybrid (H : Msg → Msg) (εS : (x : Msg) → Msg → S.WS → (Θ : Type) → VStar Θ → Θ → S.StS → ℝ) (hS : S.ZK H εS)
    (x xO : Msg) (wo : ℕ → S.WS) {Θ : Type} (Vs : VStar Θ) (θ : Θ) :
    ∀ (m i : ℕ) (ms : List (Msg × Out)) (os : List (Option S.StS)) (Ev : List (Msg × Out) → Prop),
      (∀ j, i ≤ j → j < i + m → S.RS x (L.xs x xO j) (wo j)) →
      Pr (fun t : S.tapes x m => L.outs x xO m i (L.lrun H x xO wo Vs θ (ms.map Prod.fst) m i t) = os ∧
          Ev (ms ++ L.lrun H x xO wo Vs θ (ms.map Prod.fst) m i t)) ≤
        Pr (fun c : S.coins x m =>
            Ev (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) (L.csim x xO m i os ms c)).1) +
          L.errAt H εS x xO wo Θ Vs θ m i ms os := by
  intro m
  induction m with
  | zero =>
    intro i ms os Ev _
    have he : L.errAt H εS x xO wo Θ Vs θ 0 i ms os = 0 := rfl
    rw [he, add_zero]
    refine le_trans (Pr_mono fun t h => ?_) (le_of_eq (Pr_const_eq (Ev ms)))
    rw [L.lrun_zero H x xO wo Vs θ _ i t, List.append_nil] at h
    exact h.2
  | succ m IH =>
    intro i ms os Ev hR
    rcases os with _ | ⟨_ | a, os⟩
    · have he : L.errAt H εS x xO wo Θ Vs θ (m + 1) i ms [] = 0 := rfl
      rw [he, add_zero]
      refine le_trans (le_of_eq ((Pr_congr fun t => iff_false_intro fun h => ?_).trans Pr_false)) (Pr_nonneg _)
      have h1 := h.1
      rw [outs] at h1
      exact List.cons_ne_nil _ _ h1
    · have he : L.errAt H εS x xO wo Θ Vs θ (m + 1) i ms (none :: os) = 0 := rfl
      rw [he, add_zero]
      refine le_trans (le_of_eq ((Pr_congr fun t => iff_false_intro fun h => ?_).trans Pr_false)) (Pr_nonneg _)
      have h1 := h.1
      rw [outs, List.cons.injEq] at h1
      exact Option.some_ne_none _ h1.1
    · let τA : S.NS x × S.ΩS x → List (Msg × Out) := fun tA => L.sRun H x xO wo Vs θ (ms.map Prod.fst) i tA
      let Vi : VStar Θ := vUnder Vs xO (ms.map Prod.fst) (S.kS (L.xs x xO i))
      let simS : S.CS x → List (Msg × Out) := fun cS => ((S.simAt x).given a).out H Vi θ (L.xs x xO i) cS
      let rest : List (Msg × Out) → S.coins x m → List (Msg × Out) := fun τ cB =>
        (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) (L.csim x xO m (i + 1) os (ms ++ τ) cB)).1
      let G : List (Msg × Out) → S.coins x m → Prop := fun τ cB =>
        if ZKMal.goesOn (S.kS (L.xs x xO i)) τ = true then Ev (rest τ cB)
        else List.replicate m none = os ∧ Ev (ms ++ τ)
      let errT : S.NS x × S.ΩS x → ℝ := fun tA =>
        if S.outS (L.xs x xO i) (τA tA) = a ∧ ZKMal.goesOn (S.kS (L.xs x xO i)) (τA tA) = true then
          L.errAt H εS x xO wo Θ Vs θ m (i + 1) (ms ++ τA tA) os
        else 0
      have hR' : ∀ j, i + 1 ≤ j → j < i + 1 + m → S.RS x (L.xs x xO j) (wo j) := fun j h1 h2 =>
        hR j (by omega) (by omega)
      have h1 : ∀ tA, Pr (fun tB : S.tapes x m =>
            L.outs x xO (m + 1) i (L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i (tA, tB)) = some a :: os ∧
              Ev (ms ++ L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i (tA, tB))) ≤
          Pr (fun cB : S.coins x m => S.outS (L.xs x xO i) (τA tA) = a ∧ G (τA tA) cB) + errT tA := by
        intro tA
        rw [Pr_congr fun tB => L.ev_succ H x xO wo Vs θ m i ms a os Ev tA tB]
        by_cases hg : ZKMal.goesOn (S.kS (L.xs x xO i)) (τA tA) = true
        · by_cases ha : S.outS (L.xs x xO i) (τA tA) = a
          · have he : errT tA = L.errAt H εS x xO wo Θ Vs θ m (i + 1) (ms ++ τA tA) os := ite_eq_left ⟨ha, hg⟩
            rw [he]
            refine le_trans (le_of_eq (Pr_congr fun tB => ?_))
              ((IH (i + 1) (ms ++ τA tA) os Ev hR').trans (add_le_add (le_of_eq (Pr_congr fun cB => ?_)) le_rfl))
            · show (S.outS (L.xs x xO i) (τA tA) = a ∧ if ZKMal.goesOn (S.kS (L.xs x xO i)) (τA tA) = true then _ else _)
                ↔ _
              rw [ite_eq_left hg]
              exact ⟨fun h => h.2, fun h => ⟨ha, h⟩⟩
            · show Ev (rest (τA tA) cB) ↔ S.outS (L.xs x xO i) (τA tA) = a ∧
                (if ZKMal.goesOn (S.kS (L.xs x xO i)) (τA tA) = true then Ev (rest (τA tA) cB)
                  else List.replicate m none = os ∧ Ev (ms ++ τA tA))
              rw [ite_eq_left hg]
              exact ⟨fun h => ⟨ha, h⟩, fun h => h.2⟩
          · have he : errT tA = 0 := ite_eq_right fun h => ha h.1
            rw [he, add_zero]
            refine le_trans (le_of_eq ((Pr_congr fun tB => iff_false_intro fun h => ha h.1).trans Pr_false))
              (Pr_nonneg _)
        · have he : errT tA = 0 := ite_eq_right fun h => hg h.2
          rw [he, add_zero]
          refine le_of_eq ((Pr_congr fun tB => ?_).trans ((Pr_const_eq (S.outS (L.xs x xO i) (τA tA) = a ∧
            (List.replicate m none = os ∧ Ev (ms ++ τA tA)))).trans (Pr_congr fun cB => ?_)))
          · show (S.outS (L.xs x xO i) (τA tA) = a ∧ if ZKMal.goesOn (S.kS (L.xs x xO i)) (τA tA) = true then _ else _)
              ↔ _
            rw [ite_eq_right hg]
          · show _ ↔ S.outS (L.xs x xO i) (τA tA) = a ∧
              (if ZKMal.goesOn (S.kS (L.xs x xO i)) (τA tA) = true then Ev (rest (τA tA) cB)
                else List.replicate m none = os ∧ Ev (ms ++ τA tA))
            rw [ite_eq_right hg]
      have h2 : ∀ cB : S.coins x m, Pr (fun tA : S.NS x × S.ΩS x =>
            S.outS (L.xs x xO i) (τA tA) = a ∧ G (τA tA) cB) ≤
          Pr (fun cS : S.CS x => G (simS cS) cB) + εS x (L.xs x xO i) (wo i) Θ Vi θ a := fun cB =>
        hS x (L.xs x xO i) (wo i) (hR i le_rfl (by omega)) Θ Vi θ rfl a fun τ => G τ cB
      have h3 : Pr (fun cS : S.CS x × S.coins x m => G (simS cS.1) cS.2) ≤
          Pr (fun c : S.coins x (m + 1) =>
            Ev (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) (L.csim x xO (m + 1) i (some a :: os) ms c)).1) := by
        refine Pr_mono fun c hc => ?_
        show Ev (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) (L.csim x xO (m + 1) i (some a :: os) ms c)).1
        rw [L.csim_succ H x xO Vs θ m i a os ms c]
        by_cases hg : ZKMal.goesOn (S.kS (L.xs x xO i)) (simS c.1) = true
        · have hc' : G (simS c.1) c.2 = Ev (rest (simS c.1) c.2) := ite_eq_left hg
          rw [ite_eq_left hg]
          rw [hc'] at hc
          exact hc
        · have hc' : G (simS c.1) c.2 = (List.replicate m none = os ∧ Ev (ms ++ simS c.1)) := ite_eq_right hg
          rw [ite_eq_right hg]
          rw [hc'] at hc
          exact hc.2
      have herr : L.errAt H εS x xO wo Θ Vs θ (m + 1) i ms (some a :: os) =
          εS x (L.xs x xO i) (wo i) Θ Vi θ a + Ex errT := rfl
      calc Pr (fun t : S.tapes x (m + 1) =>
            L.outs x xO (m + 1) i (L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i t) = some a :: os ∧
              Ev (ms ++ L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i t))
          = Ex (fun tA : S.NS x × S.ΩS x => Pr (fun tB : S.tapes x m =>
            L.outs x xO (m + 1) i (L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i (tA, tB)) = some a :: os ∧
              Ev (ms ++ L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i (tA, tB)))) :=
            ZKMal.Pr_prod_left (α := S.NS x × S.ΩS x) (β := S.tapes x m) fun t =>
              L.outs x xO (m + 1) i (L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i t) = some a :: os ∧
                Ev (ms ++ L.lrun H x xO wo Vs θ (ms.map Prod.fst) (m + 1) i t)
        _ ≤ Ex (fun tA => Pr (fun cB : S.coins x m => S.outS (L.xs x xO i) (τA tA) = a ∧ G (τA tA) cB) + errT tA) :=
            Ex_mono h1
        _ = Ex (fun tA => Pr (fun cB : S.coins x m => S.outS (L.xs x xO i) (τA tA) = a ∧ G (τA tA) cB)) +
            Ex errT := Ex_add _ _
        _ = Ex (fun cB : S.coins x m => Pr (fun tA : S.NS x × S.ΩS x =>
              S.outS (L.xs x xO i) (τA tA) = a ∧ G (τA tA) cB)) + Ex errT := by
            rw [Ex_Pr_comm (fun tA cB => S.outS (L.xs x xO i) (τA tA) = a ∧ G (τA tA) cB)]
        _ ≤ Ex (fun cB : S.coins x m => Pr (fun cS : S.CS x => G (simS cS) cB) +
              εS x (L.xs x xO i) (wo i) Θ Vi θ a) + Ex errT := add_le_add (Ex_mono h2) le_rfl
        _ = Pr (fun cS : S.CS x × S.coins x m => G (simS cS.1) cS.2) + εS x (L.xs x xO i) (wo i) Θ Vi θ a +
              Ex errT := by
            rw [Ex_add, Ex_const, ← Ex_Pr_comm (fun cS cB => G (simS cS) cB),
              ← ZKMal.Pr_prod_left (fun cS : S.CS x × S.coins x m => G (simS cS.1) cS.2)]
        _ ≤ Pr (fun c : S.coins x (m + 1) =>
              Ev (SP.eval H (fun as => ((Vs.next xO as θ).eval H).1) (L.csim x xO (m + 1) i (some a :: os) ms c)).1) +
              εS x (L.xs x xO i) (wo i) Θ Vi θ a + Ex errT := add_le_add (add_le_add h3 le_rfl) le_rfl
        _ = _ := by rw [herr, add_assoc]

/-- **Z1-S-cost's induction**: the `m` sessions' simulator makes at most `m` times the session simulator's calls. -/
theorem csim_cost (H : Msg → Msg) (cS : Msg → ℕ)
    (hc : ∀ x xs o (c : S.CS x) ask, ((S.progS x xs o c).eval H ask).2 ≤ cS x) (x xO : Msg)
    (ask : List Msg → Out) :
    ∀ (m i : ℕ) (os : List (Option S.StS)) (ms : List (Msg × Out)) (c : S.coins x m),
      (SP.eval H ask (L.csim x xO m i os ms c)).2 ≤ m * cS x := by
  intro m
  induction m with
  | zero => intro i os ms c; simp [csim, SP.eval]
  | succ m IH =>
    intro i os ms c
    rcases os with _ | ⟨_ | a, os⟩
    · simp [csim, SP.eval]
    · simp [csim, SP.eval]
    · rw [csim, ZKMal.recSpBind_eq, ZKMal.spBind_eval, ZKMal.recSpUnder_eq, ZKMal.spUnder_eval]
      have h1 := hc x (L.xs x xO i) a c.1 fun as => ask (ms.map Prod.fst ++ as)
      have h2 : ∀ τ : List (Msg × Out), (SP.eval H ask
          (if ZKMal.goesOn (S.kS (L.xs x xO i)) τ = true then L.csim x xO m (i + 1) os (ms ++ τ) c.2
            else .ret (ms ++ τ))).2 ≤ m * cS x := by
        intro τ
        by_cases hg : ZKMal.goesOn (S.kS (L.xs x xO i)) τ = true
        · rw [ite_eq_left hg]
          exact IH _ _ _ _
        · rw [ite_eq_right hg]
          exact Nat.zero_le _
      exact (Nat.add_le_add h1 (h2 _)).trans (le_of_eq (by ring))

end Sessions

end Claims.RecGeneric

open Claims Claims.RecGeneric in
/-- **Z1-S** (`Flock.Guarantees.GenericSessionsZK`): the hybrid over all the sessions, from the first, after no
messages. -/
theorem Flock.SecurityProofs.GenericSessionsZK : Flock.Guarantees.GenericSessionsZK := by
  intro H S L εS hS hSh x xO wo hR Θ Vs θ hk o Ev
  have hmsgs : ∀ ω : S.tapes x (L.n x) × Unit,
      (runStar H (L.PO wo x) Vs xO ω θ).msgs = L.lrun H x xO wo Vs θ [] (L.n x) 0 ω.1 := by
    intro ω
    rw [ZKMal.runStar_msgs, hk]
    unfold Sessions.lrun
    rw [hSh x xO]
    rfl
  refine le_trans (le_of_eq ((Pr_congr fun ω => ?_).trans (Pr_fst (fun t : S.tapes x (L.n x) =>
    L.outs x xO (L.n x) 0 (L.lrun H x xO wo Vs θ (([] : List (Msg × Out)).map Prod.fst) (L.n x) 0 t) = o ∧
      Ev ([] ++ L.lrun H x xO wo Vs θ (([] : List (Msg × Out)).map Prod.fst) (L.n x) 0 t)))))
    (L.hybrid H εS hS x xO wo Vs θ (L.n x) 0 [] o Ev fun j _ hj => hR j (by omega))
  rw [hmsgs ω]
  exact Iff.rfl

open Claims Claims.RecGeneric in
/-- **Z1-S-cost** (`Flock.Guarantees.GenericSessionsZKCost`). -/
theorem Flock.SecurityProofs.GenericSessionsZKCost : Flock.Guarantees.GenericSessionsZKCost :=
  fun H _ L cS hc x xO o c ask => L.csim_cost H cS hc x xO ask (L.n x) 0 o [] c

open Claims Claims.RecGeneric in
/-- **Z1 at an outer phase of sessions** (`Flock.Guarantees.GenericRecursiveSessionsZK`): Z1 at the release, its outer
phase's zero knowledge from Z1-S. -/
theorem Flock.SecurityProofs.GenericRecursiveSessionsZK : Flock.Guarantees.GenericRecursiveSessionsZK :=
  fun H _ Lv S L F εS δ hHide hS hSh =>
    Flock.SecurityProofs.GenericRecursiveZK H Lv (ZkRelease.ofSessions F L) (L.err H εS) δ hHide
      (Flock.SecurityProofs.GenericSessionsZK H S L εS hS hSh)

open Claims Claims.RecGeneric in
/-- **Z1-cost at an outer phase of sessions** (`Flock.Guarantees.GenericRecursiveSessionsZKCost`): Z1-cost at the
release, its outer simulator's calls from Z1-S-cost. -/
theorem Flock.SecurityProofs.GenericRecursiveSessionsZKCost : Flock.Guarantees.GenericRecursiveSessionsZKCost :=
  fun H _ Lv S L F cF cS hF hc =>
    Flock.SecurityProofs.GenericRecursiveZKCost H Lv (ZkRelease.ofSessions F L) cF (fun x => L.n x * cS x) hF
      (Flock.SecurityProofs.GenericSessionsZKCost H S L cS hc)

open Claims Claims.RecGeneric in
/-- **Z1-L** (`Flock.Guarantees.GenericSessionsOfZKOut`): at a session statement in `Xs x`, the session simulator's
outcome on the tapes of every statement in `Xs x` is the system's on that statement's tape (`Pr_pi_eval`). -/
theorem Flock.SecurityProofs.GenericSessionsOfZKOut : Flock.Guarantees.GenericSessionsOfZKOut := by
  intro H W R k N Ω _ _ _ _ P St out S Xs ε hZ x xs w hw Θ Vs θ hk o Ev
  obtain ⟨hx, hR⟩ := hw
  refine le_trans (hZ xs w hR Θ Vs θ hk o Ev) (le_of_eq ?_)
  congr 1
  refine (Pr_pi_eval (β := fun v : Xs x => S.C v) ⟨xs, hx⟩ fun c => Ev ((S.given o).out H Vs θ xs c)).symm.trans
    (Pr_congr fun c => ?_)
  simp only [Simulator.out, SimOut.given, ZkSess.simAt, ZkSess.ofZKOut, dite_eq_left hx]

open Claims Claims.RecGeneric in
/-- **Z1-L-cost** (`Flock.Guarantees.GenericSessionsOfZKOutCost`). -/
theorem Flock.SecurityProofs.GenericSessionsOfZKOutCost : Flock.Guarantees.GenericSessionsOfZKOutCost := by
  intro H W R k N Ω _ _ _ _ P St out S Xs cS hc x xs o c ask
  by_cases hx : xs ∈ Xs x
  · simp only [ZkSess.ofZKOut, dite_eq_left hx]
    exact hc x xs hx o _ ask
  · simp only [ZkSess.ofZKOut, dite_eq_right hx]
    exact Nat.zero_le _
