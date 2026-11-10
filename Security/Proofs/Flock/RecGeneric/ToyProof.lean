module

public import Proofs.Flock.RecGeneric.Toy
public import Proofs.Flock.RecGeneric.Plurality

/-!
# The toy meets L2's hypotheses (`EchoHolds`)

`echoHolds` proves `EchoHolds`: at every oracle, inner protocol, server and statement, the echo argument and the echo
compiler at `Open(Π, S)` meet L2's hypotheses on them.

* `echoArg` is knowledge sound at `κ = 1/|Coin|`, `γ = 0` (`echoArg_KS`): the prover's one message reads no coin, so
  where it is the statement the fresh rerun accepts it and `E^rep` gives it (`echoArg_ext_eval`); an accepted run whose
  message is not the statement was accepted on its coin, which is `default` with probability `1/|Coin|`
  (`Pr_coinAt`).
* The compiler's contract for `E_O` (`echo_sessions`): each session's bounds are `echoArg`'s errors; session `s` is
  round `k+3+s` of `echoCV`; a fork there shows
  the prover's message after the run's earlier ones, answered as `echoV` answers it at the staged statement
  (`echo_fork`, from `run_fork_get`), and a refused session round ends the fork's run refused (`echo_runs`).
* The composition's contract (`echo_contract`): `E_O`'s witnesses are paired with the staged statements
  (`Compiled.EO_fst`), so under the outer relation the inner messages they give are the run's first `k` messages
  (`echo_wits`). Two accepted reruns forked at an inner round `j` show one round-`j` message, so `ForksBind` holds
  with no collision (`echo_forks`), and an accepted run's last round is the inner view's replay (`echo_accepts`, by
  `replayOC_congr`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-! ## Lists and coins -/

theorem getD_take_of_lt {β : Type} (l : List β) (d : β) {i n : ℕ} (hi : i < n) :
    (l.take n).getD i d = l.getD i d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take, ite_eq_left hi]

theorem getD_finRange_map {β : Type} {k : ℕ} (f : Fin k → β) (d : β) {i : ℕ} (hi : i < k) :
    ((List.finRange k).map f).getD i d = f ⟨i, hi⟩ := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_eq_getElem (by rw [List.length_finRange]; exact hi)]
  simp

theorem getLastD_range_map {β : Type} (f : ℕ → β) (d : β) (n : ℕ) :
    ((List.range (n + 1)).map f).getLastD d = f n := by
  simp only [List.range_succ, List.map_append, List.map_cons, List.map_nil, List.getLastD_concat]

theorem forall₂_map_fst {α β γ : Type} {R : α → β × γ → Prop} {g : α → β} (hR : ∀ a b, R a b → b.1 = g a) :
    ∀ {l : List α} {w : List (β × γ)}, List.Forall₂ R l w → w.map Prod.fst = l.map g := by
  intro l w h
  induction h with
  | nil => rfl
  | cons h _ ih => simp [hR _ _ h, ih]

/-- `Pr[the coin at i is c] = 1/|Coin|`, beside a tape and an extractor's coins. -/
theorem Pr_coinAt {Ω Coin E : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [DecidableEq Coin]
    [Fintype E] [Nonempty E] {n : ℕ} (i : Fin n) (c : Coin) :
    Pr (fun s : Ω × (Fin n → Coin) × E => s.2.1 i = c) = 1 / Fintype.card Coin := by
  rw [Pr_snd (fun t : (Fin n → Coin) × E => t.1 i = c), Pr_fst (fun f : Fin n → Coin => f i = c),
    Claims.Bridge.Pr_coord i (fun y : Coin => y = c), Pr_eq_sum (fun y : Coin => y = c)]
  simp

/-! ## Runs -/

/-- **Two replays that agree up to round `j`, on coins that agree before it, show one message at `j`.** -/
theorem run_replay_fst_eq {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (ms ms' : List Msg) (c c' : Fin (V.k x) → Coin) (j : ℕ) (hm : ∀ i ≤ j, ms.getD i [] = ms'.getD i [])
    (hc : ∀ i < j, coinSeq c i = coinSeq c' i) :
    ((run H (replay ms) V x () c).msgs.map Prod.fst).getD j [] =
      ((run H (replay ms') V x () c').msgs.map Prod.fst).getD j [] := by
  by_cases hj : j < (run H (replay ms) V x () c).msgs.length
  · have hj' := Claims.Recursion.run_replay_plays H V x ms ms' c c' j (fun i hi => hm i hi.le) hc hj
    rw [Claims.Recursion.VStar.run_replay_fst H V x ms c hj, Claims.Recursion.VStar.run_replay_fst H V x ms' c' hj',
      hm j le_rfl]
  · have hj' : ¬ j < (run H (replay ms') V x () c').msgs.length := fun h' =>
      hj (Claims.Recursion.run_replay_plays H V x ms' ms c' c j (fun i hi => (hm i hi.le).symm)
        (fun i hi => (hc i hi).symm) h')
    rw [List.getD_eq_default _ _ (by simpa using hj), List.getD_eq_default _ _ (by simpa using hj')]

/-- **A fork at round `j` of a run that reaches it**: its message at `j` is the prover's after the run's first `j`
messages, answered by the verifier at the fork's coins. -/
theorem run_fork_get {Coin : Type} [Inhabited Coin] (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (V : Verifier Coin)
    (x : Msg) (ω : Ω) (j : ℕ) (r d : Fin (V.k x) → Coin) (hj : j < (run H P V x ω r).msgs.length) :
    (run H P V x ω (fork j r d)).msgs[j]? = some
      (((P.next x (((run H P V x ω r).msgs.take j).map RecDeploy.histOf) ω).eval H).1,
        ((V.next x (((run H P V x ω r).msgs.take j).map Prod.fst ++
            [((P.next x (((run H P V x ω r).msgs.take j).map RecDeploy.histOf) ω).eval H).1])
          ((List.range (j + 1)).map (coinSeq (fork j r d)))).eval H).1) := by
  have hag := RecDeploy.stepWith_agree H P x ω
    (fun as => V.next x as ((List.range as.length).map (coinSeq r)))
    (fun as => V.next x as ((List.range as.length).map (coinSeq (fork j r d)))) (V.k x) j [] []
    ⟨[], [], [], .rej⟩ ⟨[], [], [], .rej⟩ rfl fun bs hb => by
      have e : (List.range bs.length).map (coinSeq r) = (List.range bs.length).map (coinSeq (fork j r d)) :=
        List.map_congr_left fun i hi => (RecDeploy.coinSeq_fork_lt j r d (by simp at hi hb; omega)).symm
      simp only [e]
  simp only [List.length_nil, Nat.zero_add] at hag
  obtain ⟨new, -, hnew, hget⟩ := RecDeploy.stepWith_get H P x ω
    (fun as => V.next x as ((List.range as.length).map (coinSeq (fork j r d)))) j (V.k x) [] []
    ⟨[], [], [], .rej⟩ (by simpa using hag.2.1 hj)
  have hn : new = (run H P V x ω r).msgs.take j := by
    rw [hnew]
    simp only [List.length_nil, List.drop_zero]
    exact hag.1.symm
  subst hn
  change (stepWith H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq (fork j r d)))) (V.k x) [] []
    ⟨[], [], [], .rej⟩).msgs[j]? = _
  simpa [Nat.min_eq_left hj.le] using hget

/-! ## The outer extraction -/

theorem Compiled.sessionProg_fst {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (L : Compiled Coin A) (H : Msg → Msg) (x : Msg) (r₀ : Fin (L.V.k x) → Coin) (s : Fin (L.nS x)) (xO : Msg)
    (c : A.EC (L.sh x s)) (run₂ : (Fin (L.V.k x) → Coin) → Transcript) {a : Msg × A.W}
    (h : ((L.sessionProg x r₀ s xO c).eval H run₂).1 = some a) : a.1 = xO := by
  unfold Compiled.sessionProg at h
  split_ifs at h
  · rw [RP.eval_bind_ret] at h
    obtain ⟨w, -, rfl⟩ := Option.map_eq_some_iff.1 h
    rfl
  · simp [RP.eval] at h

/-- **`E_O` gives witnesses only on accepted runs, paired with the staged statements**, in order. -/
theorem Compiled.EO_fst {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (L : Compiled Coin A) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (L.V.k x) → Coin)
    (c : L.EO.C x) {w : List (Msg × A.W)} (h : (L.EO.eval H P x ω r c).1 = some w) :
    (run H P L.V x ω r).out = .acc ∧
      w.map Prod.fst =
        (List.finRange (L.nS x)).map fun s : Fin (L.nS x) => (L.stmts x (run H P L.V x ω r).msgs).getD s.val [] := by
  have hacc : (run H P L.V x ω r).out = .acc := by
    by_contra hn
    simp [Extractor.eval, Compiled.EO, RP.eval, Transcript.seen, hn] at h
  refine ⟨hacc, ?_⟩
  simp only [Extractor.eval, Compiled.EO, RP.eval, Transcript.seen, hacc, ↓reduceIte] at h
  exact forall₂_map_fst (fun s _ ha => L.sessionProg_fst H x r s _ (c s) _ ha)
    (List.forall₂_map_left_iff.1 (collect_some H _ _ w h))

/-! ## The echo argument -/

/-- **`echoV`'s run**: one round, the prover's first message, accepted where it is the statement or the coin is
`default`. -/
theorem run_echoV {Coin : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Coin] (H : Msg → Msg) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (c : Fin ((echoArg Coin).V.k x) → Coin) :
    (run H P (echoArg Coin).V x ω c).seen =
      ⟨[(((P.next x [] ω).eval H).1,
          if ((P.next x [] ω).eval H).1 = x ∨ coinSeq c 0 = default then .acc else .rej)],
        if ((P.next x [] ω).eval H).1 = x ∨ coinSeq c 0 = default then .acc else .rej⟩ ∧
    (run H P (echoArg Coin).V x ω c).vLog = [] := by
  simp [run, stepWith, echoArg, echoV, OC.eval, Transcript.seen]
  split_ifs with h
  · simp
    exact (ite_eq_left h).symm
  · simp
    exact (ite_eq_right h).symm

/-- **`echoArg`'s extractor where every run shows the statement accepted**: the judged run and the fresh rerun accept,
and the candidate is the rerun's message, the statement. -/
theorem echoArg_ext_eval {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (x : Msg) (r : Fin ((echoArg Coin).V.k x) → Coin) (c : (echoArg Coin).EC ((echoArg Coin).shape x))
    (rn : (Fin ((echoArg Coin).V.k x) → Coin) → Transcript) (hrn : ∀ r', (rn r').seen = ⟨[(x, .acc)], .acc⟩) :
    (((echoArg Coin).ext x r c).eval H rn).1 = some x := by
  simp [Claims.SamplingGeneric.Arg.ext, ERep, RP.eval, firstWait, waitAcc, RP.bind, hrn, echoArg,
    List.finRange_succ]
  erw [hrn]
  rfl

/-- **`echoArg` is knowledge sound** at `κ = 1/|Coin|`, `γ = 0`: the prover's message reads no coin, so where it is
the statement the rerun accepts it and the extractor gives it (`echoArg_ext_eval`); where it is not, the judged run
accepted on its coin. -/
theorem echoArg_KS {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) : (echoArg Coin).KS H := by
  intro x Ω _ _ P
  have h0 : 0 < (echoArg Coin).V.k x := Nat.one_pos
  refine le_trans ?_ (le_add_of_nonneg_right (Real.sqrt_nonneg _))
  refine (Pr_mono (q := fun s : Ω × (Fin ((echoArg Coin).V.k x) → Coin) × (echoArg Coin).E.C x =>
    s.2.1 ⟨0, h0⟩ = default) fun s hs => ?_).trans
      (Pr_coinAt (Ω := Ω) (E := (echoArg Coin).E.C x) ⟨0, h0⟩ default).le
  obtain ⟨hacc, hbad⟩ := hs
  have hr := run_echoV H P x s.1 s.2.1
  have hout : (run H P (echoArg Coin).V x s.1 s.2.1).out =
      if ((P.next x [] s.1).eval H).1 = x ∨ coinSeq s.2.1 0 = default then .acc else .rej :=
    congrArg Seen.out hr.1
  have hne : ¬ ((P.next x [] s.1).eval H).1 = x := fun h => hbad (by
    have hE : ((echoArg Coin).E.eval H P x s.1 s.2.1 s.2.2).1 = some x :=
      echoArg_ext_eval H x s.2.1 s.2.2 (run H P (echoArg Coin).V x s.1) fun r' => by
        simp [(run_echoV H P x s.1 r').1, h]
    rw [hE]
    exact rfl)
  have hc : coinSeq s.2.1 0 = default := by
    by_contra hc
    simp [hout, hne, hc] at hacc
  rwa [coinSeq, dite_eq_left h0] at hc

/-! ## The echo compiler's verifier -/

/-- At a session round, `echoCV` answers `.msg []` where the message is the inner round's or the coin is
`default`, and refuses otherwise. -/
theorem echoCV_next_session {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) (x : Msg) (bs : List Msg) (cs : List Coin)
    (h0 : IV.k x ≠ 0) (h1 : IV.k x + 3 < bs.length) (h2 : bs.length ≤ 2 * IV.k x + 3) :
    (((Compiled.echo IV).V.next x bs cs).eval H).1 =
      if bs.getLastD [] = bs.getD (bs.length - (IV.k x + 4)) [] ∨ cs.getLastD default = default then .msg []
      else .rej := by
  have e1 : ¬ bs.length ≤ IV.k x := by omega
  have e2 : ¬ bs.length ≤ IV.k x + 3 := by omega
  simp only [Compiled.echo, echoCV, h0, e1, e2, h2, ↓reduceIte]
  rfl

/-- `echoCV` accepts only at its last round, where the inner verifier's replay accepts. -/
theorem echoCV_next_acc {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) (x : Msg) (as : List Msg) (cs : List Coin)
    (h : (((Compiled.echo IV).V.next x as cs).eval H).1 = .acc) :
    IV.k x ≠ 0 ∧ 2 * IV.k x + 3 < as.length ∧
      ((Comp.replayOC IV x (as.take (IV.k x)) (fun i => cs.getD i default) (IV.k x) [] []).eval H).1.out =
        .acc := by
  by_cases h0 : IV.k x = 0
  · simp [Compiled.echo, echoCV, h0, OC.eval] at h
  by_cases h1 : as.length ≤ IV.k x
  · simp only [Compiled.echo, echoCV, h0, h1, ↓reduceIte, OC.eval_bind] at h
    rcases hb : ((IV.next x as cs).eval H).1 with m | _ | _ <;> simp [hb, OC.eval] at h
  by_cases h2 : as.length ≤ IV.k x + 3
  · simp [Compiled.echo, echoCV, h0, h1, h2, OC.eval] at h
  by_cases h3 : as.length ≤ 2 * IV.k x + 3
  · simp only [Compiled.echo, echoCV, h0, h1, h2, h3, ↓reduceIte, OC.eval] at h
    split_ifs at h
  simp only [Compiled.echo, echoCV, h0, h1, h2, h3, ↓reduceIte, OC.eval_bind, OC.eval] at h
  refine ⟨h0, by omega, ?_⟩
  by_contra hv
  rw [ite_eq_right hv] at h
  cases h

/-- **An accepted run of `echoCV`** takes all `2k + 4` rounds, `k > 0`, and the inner verifier's replay of its first
`k` messages at its first `k` coins accepts. -/
theorem echoCV_acc {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((Compiled.echo IV).V.k x) → Coin) (h : (run H P (Compiled.echo IV).V x ω r).out = .acc) :
    0 < IV.k x ∧ (run H P (Compiled.echo IV).V x ω r).msgs.length = 2 * IV.k x + 4 ∧
      ((Comp.replayOC IV x (((run H P (Compiled.echo IV).V x ω r).msgs.map Prod.fst).take (IV.k x))
        (fun i => ((List.range (2 * IV.k x + 4)).map (coinSeq r)).getD i default) (IV.k x) [] []).eval H).1.out =
        .acc := by
  obtain ⟨h0, hl, hv⟩ :=
    echoCV_next_acc H IV x _ _ (FlockDeploy.KS.run_acc_next H P (Compiled.echo IV).V x ω r h)
  have hle : (run H P (Compiled.echo IV).V x ω r).msgs.length ≤ 2 * IV.k x + 4 :=
    Claims.Recursion.run_length_le H P (Compiled.echo IV).V x ω r
  rw [List.length_map] at hl hv
  have e : (run H P (Compiled.echo IV).V x ω r).msgs.length = 2 * IV.k x + 4 := by omega
  rw [e] at hv
  exact ⟨Nat.pos_of_ne_zero h0, e, hv⟩

/-- `replayOC` reads its messages and coins only below its rounds. -/
theorem replayOC_congr {Coin : Type} [Inhabited Coin] (IV : Verifier Coin) (xI : Msg) (as as' : List Msg)
    (cs cs' : ℕ → Coin) :
    ∀ (n : ℕ) (sent : List Msg) (ms : List (Msg × Out)),
      (∀ i < sent.length + n, as.getD i [] = as'.getD i [] ∧ cs i = cs' i) →
      Comp.replayOC IV xI as cs n sent ms = Comp.replayOC IV xI as' cs' n sent ms
  | 0, _, _, _ => rfl
  | n + 1, sent, ms, h => by
    have ha := (h sent.length (by omega)).1
    have hc : (List.range (sent.length + 1)).map cs = (List.range (sent.length + 1)).map cs' :=
      List.map_congr_left fun i hi => (h i (by rw [List.mem_range] at hi; omega)).2
    simp only [Comp.replayOC, ha, hc]
    congr 1
    funext b
    cases b with
    | msg m =>
      exact replayOC_congr IV xI as as' cs cs' n _ _ fun i hi => h i (by simp at hi; omega)
    | acc => rfl
    | rej => rfl

theorem off_echo {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (IV : Verifier Coin) (x : Msg) (s : ℕ) : (Compiled.echo IV).off x s = IV.k x + 3 + s := by
  simp [Compiled.off, Compiled.echo, echoArg]

/-- **A fork of an accepted run at session `s`'s round** `o = k + 3 + s` shows there the prover's message after the
run's first `o` messages, answered `.msg []` where it is the run's round-`s` message or the fork's coin at `o` is
`default` and refused otherwise; a refusal there ends the fork's run refused. -/
theorem echo_fork {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r d : Fin ((Compiled.echo IV).V.k x) → Coin) (hacc : (run H P (Compiled.echo IV).V x ω r).out = .acc)
    {s o : ℕ} (hs : s < IV.k x) (ho : o = IV.k x + 3 + s) :
    (run H P (Compiled.echo IV).V x ω (fork o r d)).msgs.getD o ([], .rej) =
      (((P.next x (((run H P (Compiled.echo IV).V x ω r).msgs.take o).map RecDeploy.histOf) ω).eval H).1,
        if ((P.next x (((run H P (Compiled.echo IV).V x ω r).msgs.take o).map RecDeploy.histOf) ω).eval H).1 =
            ((run H P (Compiled.echo IV).V x ω r).msgs.map Prod.fst).getD s [] ∨ coinSeq d o = default
        then .msg [] else .rej) ∧
    (((run H P (Compiled.echo IV).V x ω (fork o r d)).msgs.getD o ([], .rej)).2 = .rej →
      (run H P (Compiled.echo IV).V x ω (fork o r d)).out = .rej) := by
  obtain ⟨-, hlen, -⟩ := echoCV_acc H IV P x ω r hacc
  have hj : o < (run H P (Compiled.echo IV).V x ω r).msgs.length := by omega
  have hget := run_fork_get H P (Compiled.echo IV).V x ω o r d hj
  generalize (run H P (Compiled.echo IV).V x ω r).msgs = M at hget hlen ⊢
  generalize ((P.next x ((M.take o).map RecDeploy.histOf) ω).eval H).1 = a at hget ⊢
  have hbl : ((M.take o).map Prod.fst ++ [a]).length = o + 1 := by simp; omega
  have hsl : ((M.take o).map Prod.fst).length = o := by simp; omega
  rw [echoCV_next_session H IV x ((M.take o).map Prod.fst ++ [a]) ((List.range (o + 1)).map (coinSeq (fork o r d)))
      (by omega) (by rw [hbl]; omega) (by rw [hbl]; omega),
    List.getLastD_concat, hbl, show o + 1 - (IV.k x + 4) = s by omega,
    List.getD_append ((M.take o).map Prod.fst) [a] [] s (by rw [hsl]; omega), List.map_take,
    getD_take_of_lt _ _ (show s < o by omega), getLastD_range_map, RecDeploy.coinSeq_fork_ge o r d le_rfl] at hget
  rw [List.getD_eq_getElem?_getD (l := (run H P (Compiled.echo IV).V x ω (fork o r d)).msgs), hget,
    Option.getD_some]
  refine ⟨rfl, fun hr => ?_⟩
  have hr' : (run H P (Compiled.echo IV).V x ω (fork o r d)).msgs[o]? = some (a, .rej) := by
    rw [hget]
    exact congrArg (fun b => some (a, b)) hr
  exact RecDeploy.stepWith_rej_out H P x ω _ ((Compiled.echo IV).V.k x) o [] [] ⟨[], [], [], .rej⟩ a
    (by simp only [List.length_nil, Nat.zero_add]; exact hr')

/-- **The echo compiler runs session `s` as `echoV`** at its staged statement. -/
theorem echo_runs {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) : (Compiled.echo IV).Runs H := by
  intro x s _ P ω r d hacc
  rw [off_echo]
  have hrun := run_echoV H
    (sessProver P x ω ((run H P (Compiled.echo IV).V x ω r).msgs.take (IV.k x + 3 + ↑s)))
    (((Compiled.echo IV).stmts x (run H P (Compiled.echo IV).V x ω r).msgs).getD ↑s []) ()
    (sessCoins (IV.k x + 3 + ↑s) d _)
  refine ⟨?_, fun e he => by rw [hrun.2] at he; cases he⟩
  have ha : (((sessProver P x ω ((run H P (Compiled.echo IV).V x ω r).msgs.take (IV.k x + 3 + ↑s))).next
      (((Compiled.echo IV).stmts x (run H P (Compiled.echo IV).V x ω r).msgs).getD ↑s []) [] ()).eval H).1 =
      ((P.next x (((run H P (Compiled.echo IV).V x ω r).msgs.take (IV.k x + 3 + ↑s)).map RecDeploy.histOf)
        ω).eval H).1 := by
    simp only [sessProver, List.append_nil]
    rfl
  have hx : ((Compiled.echo IV).stmts x (run H P (Compiled.echo IV).V x ω r).msgs).getD ↑s [] =
      ((run H P (Compiled.echo IV).V x ω r).msgs.map Prod.fst).getD ↑s [] := by
    show (((run H P (Compiled.echo IV).V x ω r).msgs.take (IV.k x)).map Prod.fst).getD ↑s [] = _
    rw [List.map_take]
    exact getD_take_of_lt _ _ s.isLt
  have hc0 : coinSeq (sessCoins (IV.k x + 3 + ↑s) d ((echoArg Coin).V.k
      (((Compiled.echo IV).stmts x (run H P (Compiled.echo IV).V x ω r).msgs).getD ↑s []))) 0 =
      coinSeq d (IV.k x + 3 + ↑s) :=
    RecDeploy.coinSeq_sessCoins _ d _ Nat.one_pos
  rw [hrun.1, ha, hc0, hx]
  obtain ⟨hF, hrej⟩ := echo_fork H IV P x ω r d hacc s.isLt rfl
  generalize run H P (Compiled.echo IV).V x ω (fork (IV.k x + 3 + ↑s) r d) = F at hF hrej ⊢
  generalize ((P.next x (((run H P (Compiled.echo IV).V x ω r).msgs.take (IV.k x + 3 + ↑s)).map RecDeploy.histOf)
    ω).eval H).1 = a at hF ⊢
  generalize ((run H P (Compiled.echo IV).V x ω r).msgs.map Prod.fst).getD ↑s [] = xS at hF ⊢
  generalize coinSeq d (IV.k x + 3 + ↑s) = c at hF ⊢
  rw [List.getD_eq_getElem?_getD] at hF hrej
  by_cases hcond : a = xS ∨ c = default
  · rw [ite_eq_left hcond] at hF ⊢
    simp [Compiled.echo, Transcript.seen, hF]
  · rw [ite_eq_right hcond] at hF ⊢
    simp [Compiled.echo, Transcript.seen, hF, hrej (by rw [hF])]

/-- **The echo compiler meets the outer sessions' contract.** -/
theorem echo_sessions {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (IV : Verifier Coin) (H : Msg → Msg) : (Compiled.echo IV).Sessions H where
  stages x ms := by
    show (ms.take (IV.k x)).map Prod.fst = ((ms.take (IV.k x + 3)).take (IV.k x)).map Prod.fst
    rw [List.take_take, Nat.min_eq_left (by omega)]
  shapes _ _ _ _ _ _ _ := rfl
  bounded _ _ := ⟨show (0 : ℝ) ≤ 1 / (Fintype.card Coin : ℝ) by positivity, le_rfl,
    fun _ _ _ _ _ => ⟨le_rfl, le_rfl⟩⟩
  fits x := by
    rw [off_echo]
    show IV.k x + 3 + IV.k x ≤ 2 * IV.k x + 4
    omega
  accepts _ _ _ _ hv := by simp [Compiled.echo, hv]
  runs := echo_runs H IV

/-- **Under the outer relation, the inner messages `E_O`'s witnesses give are the run's first `k` messages.** -/
theorem echo_wits {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((Compiled.echo IV).V.k x) → Coin) (c : (Compiled.echo IV).EO.C x)
    {w : List (Msg × (echoArg Coin).W)} (h : ((Compiled.echo IV).EO.eval H P x ω r c).1 = some w)
    (hR : ∀ p ∈ w, (echoArg Coin).R p.1 p.2) {i : ℕ} (hi : i < IV.k x) :
    ((Compiled.echo IV).msgs x w).getD i [] =
      ((run H P (Compiled.echo IV).V x ω r).msgs.map Prod.fst).getD i [] := by
  have key : ∀ v : List (Msg × Msg), (∀ p ∈ v, p.2 = p.1) →
      v.map Prod.fst = (List.finRange (IV.k x)).map
        (fun s : Fin (IV.k x) => (((run H P (Compiled.echo IV).V x ω r).msgs.take (IV.k x)).map Prod.fst).getD s []) →
      (v.map Prod.snd).getD i [] = ((run H P (Compiled.echo IV).V x ω r).msgs.map Prod.fst).getD i [] := by
    intro v hv hvf
    rw [List.map_congr_left (f := Prod.snd) (g := Prod.fst) hv, hvf, getD_finRange_map _ _ hi, List.map_take,
      getD_take_of_lt _ _ hi]
  exact key w hR (Compiled.EO_fst (Compiled.echo IV) H P x ω r c h).2

/-- **Forks at an inner round show one message there**: so `ForksBind` holds with no collision. -/
theorem echo_forks {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) {β : Type} (rd : Seen → β) (x : Msg) :
    (Compiled.echo IV).toComp.ForksBind H IV rd x := by
  intro _ P j ω r r' c c' w w' hj hG hG' hR hR' hne _
  exfalso
  apply hne
  have hj' : j < IV.k x := hj
  have hm : ∀ i ≤ j, ((Compiled.echo IV).msgs x w).getD i [] = ((Compiled.echo IV).msgs x w').getD i [] := by
    intro i hi
    rw [echo_wits H IV P x ω r c hG.2 hR (by omega), echo_wits H IV P x ω (fork j r r') c' hG'.2 hR' (by omega)]
    exact Claims.Recursion.run_fst_getD H P _ x ω i fun i' hi' => by
      show r i' = if i'.val < j then r i' else r' i'
      rw [ite_eq_left (show i'.val < j by omega)]
  have hc : ∀ i < j, coinSeq (fun i : Fin (IV.k x) => coinSeq r i.val) i =
      coinSeq (fun i : Fin (IV.k x) => coinSeq (fork j r r') i.val) i := by
    intro i hi
    have h1 : coinSeq (fun i : Fin (IV.k x) => coinSeq r i.val) i = coinSeq r i :=
      coinSeq_firstCoins r (by omega)
    have h2 : coinSeq (fun i : Fin (IV.k x) => coinSeq (fork j r r') i.val) i = coinSeq (fork j r r') i :=
      coinSeq_firstCoins (fork j r r') (by omega)
    rw [h1, h2, RecDeploy.coinSeq_fork_lt j r r' hi]
  exact run_replay_fst_eq H IV x _ _ _ _ j hm hc

/-- **The inner view of witnesses in the outer relation accepts**: it is the replay `echoCV`'s last round ran. -/
theorem echo_accepts {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) (x : Msg) :
    (Compiled.echo IV).toComp.InnerAcceptsColl H IV x := by
  intro _ P ω r c w hG hR
  left
  obtain ⟨-, -, hv⟩ := echoCV_acc H IV P x ω r hG.1
  have e := Comp.replayOC_eval H IV x ((Compiled.echo IV).msgs x w) (fun i : Fin (IV.k x) => coinSeq r i.val)
  have hc := replayOC_congr IV x ((Compiled.echo IV).msgs x w)
    (((run H P (Compiled.echo IV).V x ω r).msgs.map Prod.fst).take (IV.k x))
    (coinSeq fun i : Fin (IV.k x) => coinSeq r i.val)
    (fun i => ((List.range (2 * IV.k x + 4)).map (coinSeq r)).getD i default) (IV.k x) [] [] fun i hi => by
      have hi' : i < IV.k x := by simpa using hi
      refine ⟨(echo_wits H IV P x ω r c hG.2 hR hi').trans (getD_take_of_lt _ _ hi').symm, ?_⟩
      rw [Claims.Recursion.getD_range_map _ _ (show i < 2 * IV.k x + 4 by omega)]
      exact coinSeq_firstCoins r hi'
  rw [hc] at e
  exact (congrArg Seen.out e).symm.trans hv

/-- **The echo compiler meets the composition's contract** at every statement. -/
theorem echo_contract {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Coin]
    (H : Msg → Msg) (IV : Verifier Coin) {β : Type} (rd : Seen → β) (x : Msg) :
    (Compiled.echo IV).toComp.Contract H IV rd x where
  replays _ P ω r c _ h := (Compiled.EO_fst (Compiled.echo IV) H P x ω r c h).1
  refuses _ _ _ _ _ := trivial
  rounds := show IV.k x ≤ 2 * IV.k x + 4 by omega
  margin _ P ω r h := ⟨(echoCV_acc H IV P x ω r h).1, show IV.k x + 3 ≤ 2 * IV.k x + 4 by omega⟩
  callsTrue _ _ _ r _ w _ e he :=
    (Claims.run_logs H (replay ((Compiled.echo IV).msgs x w)) IV x () fun i => coinSeq r i.val).2 e he
  ownTrue _ _ _ _ _ he := nomatch he
  callsLog _ _ _ _ _ _ _ _ := Or.inl fun _ he => he
  forks _ := echo_forks H IV rd x
  accepts _ := echo_accepts H IV x

/-- **T1: the toy meets L2's hypotheses together.** -/
theorem echoHolds : EchoHolds := by
  intro Coin W _ _ _ _ H I S
  exact ⟨echoArg_KS H, echo_sessions (S.IVStar I.V I.ans) H, fun x => echo_contract H (S.IVStar I.V I.ans) _ x⟩

end Claims.RecGeneric
