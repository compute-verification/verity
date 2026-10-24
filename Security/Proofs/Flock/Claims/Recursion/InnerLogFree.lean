import Proofs.Flock.Claims.Recursion.LastBindsEC
import Proofs.Flock.Claims.Recursion.CallsAgree
import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Recursion.SetupCalls

/-!
# The inner verifier's calls read no coin

`IVStar`'s last step reads the run's coins only in `agrees`, where each of `DE`'s steps reads the coins of its own
prefix. Where each of `DE`'s steps logs the same calls at any coins of its length, and answers with its own last coin
at both or at neither (`Bound.StepsLogFree`), `IVStar`'s run on replayed messages logs the same calls at all coins
(`Bound.run_vLog_free`), each call with `H`'s answer (`Bound.run_vLog_ans`). The deployed plain verifier meets it
(`RecDeploy.stepsLogFree_of_V`, `RecDeploy.stepsLogFree_deployed`): a coin request answers its own coin at any coins,
and a decision's calls are its setup's and its verdict's, which read the messages alone (`RecDeploy.next_log_free`).

So the readers' facts `ReadersLogCollS` asks hold at `IVStar` (`RecDeploy.readersLogCollS_ivstar`): the inner
verifier's calls at the run's coins are its calls at the coins `default` (`RecDeploy.innerCallsOf`), each with `H`'s
answer, which the record searches beside the recorded ones.
-/

namespace Claims.Recursion

open Claims

variable {Coin Col T : Type} [Inhabited Coin]

/-- Programs in order log the same calls where each logs the same calls, and each before the last returns the same. -/
theorem allOC_log_congr (H : Msg → Msg) {ι : Type} (f g : ι → OC Bool) :
    ∀ l : List ι, (∀ j ∈ l, ((f j).eval H).2 = ((g j).eval H).2) →
      (∀ j ∈ l.dropLast, ((f j).eval H).1 = ((g j).eval H).1) →
      ((allOC (l.map f)).eval H).2 = ((allOC (l.map g)).eval H).2
  | [], _, _ => rfl
  | j :: l, hl, hr => by
    simp only [List.map_cons, allOC, eval_bindOC_full]
    rw [hl j List.mem_cons_self]
    congr 1
    rcases l with _ | ⟨i, l⟩
    · generalize ((f j).eval H).1 = b
      generalize ((g j).eval H).1 = b'
      cases b <;> cases b' <;> rfl
    · rw [List.dropLast_cons_cons] at hr
      rw [hr j List.mem_cons_self]
      generalize ((g j).eval H).1 = b
      cases b
      · rfl
      · exact allOC_log_congr H f g (i :: l) (fun i hi => hl i (List.mem_cons_of_mem _ hi))
          (fun i hi => hr i (List.mem_cons_of_mem _ hi))

/-- **`DE`'s steps read no coin in their calls**: at any two coins of its length a step logs the same calls, and it
answers with its own last coin at both or at neither. -/
def Bound.StepsLogFree (H : Msg → Msg) (Bd : Bound Coin Col T) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs cs' : List Coin), as ≠ [] → cs.length = as.length → cs'.length = as.length →
    ((Bd.DE.V.next x as cs).eval H).2 = ((Bd.DE.V.next x as cs').eval H).2 ∧
      (((Bd.DE.V.next x as cs).eval H).1 = .msg (Bd.ans (cs.getLastD default)) ↔
        ((Bd.DE.V.next x as cs').eval H).1 = .msg (Bd.ans (cs'.getLastD default)))

theorem take_succ_getLastD {α : Type} (l : List α) (d : α) {j : ℕ} (hj : j < l.length) :
    (l.take (j + 1)).getLastD d = l.getD j d := by
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, List.getLastD_concat,
    List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, Option.getD_some]

/-- **`agrees` logs the same calls at any coins of its length** where `DE`'s steps read no coin in their calls. -/
theorem Bound.agrees_log_free {H : Msg → Msg} {Bd : Bound Coin Col T} (h : Bd.StepsLogFree H) (x : Msg)
    (ms : List Msg) {cs cs' : List Coin} (hc : cs.length = ms.length) (hc' : cs'.length = ms.length) :
    ((Bd.agrees x ms cs).eval H).2 = ((Bd.agrees x ms cs').eval H).2 := by
  have hstep : ∀ j, j < ms.length → ms.take (j + 1) ≠ [] ∧ (cs.take (j + 1)).length = (ms.take (j + 1)).length ∧
      (cs'.take (j + 1)).length = (ms.take (j + 1)).length := fun j hj =>
    ⟨List.length_pos_iff.mp (by rw [List.length_take]; omega), by simp only [List.length_take, hc],
      by simp only [List.length_take, hc']⟩
  unfold Bound.agrees
  refine allOC_log_congr H _ _ _ (fun j hj => ?_) fun j hj => ?_
  · rw [List.mem_range] at hj
    obtain ⟨hne, hl, hl'⟩ := hstep j hj
    simp only [eval_bindOC_full, OC.eval, List.append_nil]
    exact (h x _ _ _ hne hl hl').1
  · simp only [List.dropLast_eq_take, List.length_range, List.take_range, List.mem_range] at hj
    have hj1 : j + 1 < ms.length := by omega
    obtain ⟨hne, hl, hl'⟩ := hstep j (by omega)
    simp only [eval_bindOC_full, OC.eval, hj1, ↓reduceIte]
    rw [← take_succ_getLastD cs default (by omega), ← take_succ_getLastD cs' default (by omega)]
    exact decide_eq_decide.mpr (h x _ _ _ hne hl hl').2

/-- **`IVStar`'s last step logs the same calls at any coins of its length** where `DE`'s steps read no coin in their
calls: it reads the coins only in `agrees`. -/
theorem Bound.ivstar_log_free {H : Msg → Msg} {Bd : Bound Coin Col T} (h : Bd.StepsLogFree H) (x : Msg)
    {as : List Msg} {cs cs' : List Coin} (hk : ¬ as.length < Bd.DE.V.k x) (hc : cs.length = as.length)
    (hc' : cs'.length = as.length) :
    ((Bd.IVStar.next x as cs).eval H).2 = ((Bd.IVStar.next x as cs').eval H).2 := by
  simp only [Bound.IVStar, hk, ↓reduceIte]
  rcases Bd.final (as.getLastD []) with _ | ⟨os, last⟩
  · rfl
  · dsimp only
    by_cases hcnd : os.length + 1 = as.length ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j
    · rw [ite_eq_left hcnd, ite_eq_left hcnd]
      simp only [eval_bindOC_full]
      congr 1
      generalize ((allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2)).eval H).1 = ok
      cases ok
      · rfl
      · have hlen : (Bd.opened os last).length = as.length := by
          simp only [Bound.opened, List.length_append, List.length_map, List.length_singleton]
          exact hcnd.1
        simp only [↓reduceIte, eval_bindOC_full, OC.eval, List.append_nil]
        exact Bd.agrees_log_free h x _ (hc.trans hlen.symm) (hc'.trans hlen.symm)
    · rw [ite_eq_right hcnd, ite_eq_right hcnd]

variable [Fintype Coin] [Nonempty Coin] [DecidableEq Col]

/-- **`IVStar`'s run on replayed messages logs the same calls at all coins** where `DE`'s steps read no coin in their
calls. -/
theorem Bound.run_vLog_free {H : Msg → Msg} {Bd : Bound Coin Col T} (h : Bd.StepsLogFree H) (x : Msg) (ms : List Msg)
    (r r' : Fin (Bd.IVStar.k x) → Coin) :
    (run H (replay ms) Bd.IVStar x () r).vLog = (run H (replay ms) Bd.IVStar x () r').vLog := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · rw [Bd.run_ivstar_zero H x ms r hk, Bd.run_ivstar_zero H x ms r' hk]
  · rw [Bd.run_ivstar H x ms r hk, Bd.run_ivstar H x ms r' hk]
    exact Bd.ivstar_log_free h x (by simp) (by simp) (by simp)

/-- Each call of `IVStar`'s run on replayed messages has `H`'s answer. -/
theorem Bound.run_vLog_ans (H : Msg → Msg) (Bd : Bound Coin Col T) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) : ∀ e ∈ (run H (replay ms) Bd.IVStar x () r).vLog, e.2 = H e.1 := by
  rcases Nat.eq_zero_or_pos (Bd.DE.V.k x) with hk | hk
  · rw [Bd.run_ivstar_zero H x ms r hk]
    simp
  · rw [Bd.run_ivstar H x ms r hk]
    exact FlockDeploy.KS.eval_log_ans H _

end Claims.Recursion

namespace RecDeploy

open Claims (Msg Out OC)
open Claims.Recursion (Bound)
open FlockDeploy (Coin)

/-- **A decision's calls read neither its link, its first coin nor its round messages' coins**: they are its setup's
and then its verdict's. -/
theorem decision_log_free (H : Msg → Msg) (zk : Bool) (s : FlockDeploy.Ins) (x : Msg)
    (link link' : List ByteArray × List ByteArray) (c₁ c₁' : Coin) (mids mids' : List (Msg × Coin))
    (f : FlockDeploy.Final) :
    ((FlockDeploy.decision zk s x link c₁ mids f).run.eval H).2 =
      ((FlockDeploy.decision zk s x link' c₁' mids' f).run.eval H).2 := by
  simp only [FlockDeploy.decision, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (FlockDeploy.setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rcases r with e | ⟨spec, sts⟩
  · rfl
  · show l ++ ((FlockDeploy.decisionAt zk s spec sts x link c₁ mids f).run.eval H).2 =
      l ++ ((FlockDeploy.decisionAt zk s spec sts x link' c₁' mids' f).run.eval H).2
    rw [FlockDeploy.KS.decisionAt_log, FlockDeploy.KS.decisionAt_log]

/-- A coin request at some coins is one at any nonempty coins, of their own last coin. -/
theorem act_coin_transfer {x : Msg} {as : List Msg} {cs cs' : List Coin} {c : Coin} (hc' : cs' ≠ [])
    (h : FlockDeploy.act false x as cs = .coin c) : FlockDeploy.act false x as cs' = .coin (cs'.getLast hc') := by
  rcases hs : FlockDeploy.setupOf false x with _ | ⟨s, spec, sts⟩
  · rw [FlockDeploy.KS.act_none hs] at h
    cases h
  · obtain ⟨t, ht⟩ := act_coin_tag h
    exact FlockComplete.act_coin_any hs ht (List.getLast?_eq_some_getLast hc')

/-- A decision at some coins is one at any nonempty coins, at the same session and last message. -/
theorem act_decide_transfer {x : Msg} {as : List Msg} {cs cs' : List Coin} {s : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.Final} (hc' : cs' ≠ [])
    (h : FlockDeploy.act false x as cs = .decide s spec sts link c₁ mids f) :
    ∃ link' c₁' mids', FlockDeploy.act false x as cs' = .decide s spec sts link' c₁' mids' f := by
  have hs := FlockDeploy.act_decide h
  have hf := FlockDeploy.KS.act_decide_final h
  obtain ⟨a₁, rest, crest, rfl, rfl, hlink, -⟩ := FlockDeploy.KS.act_decide_parts h
  obtain ⟨c₁', crest', rfl⟩ := List.exists_cons_of_ne_nil hc'
  have hl := List.getLast?_eq_some_getLast (List.cons_ne_nil a₁ rest)
  rw [hl, Option.bind_some] at hf
  exact ⟨link, c₁', _, FlockComplete.act_final_any hs hl hf hlink⟩

/-- **The deployed plain verifier's step reads no coin in its calls**: at any two nonempty coins it logs the same calls,
and answers with its own last coin at both or at neither. -/
theorem next_log_free (H : Msg → Msg) (x : Msg) (as : List Msg) {cs cs' : List Coin} (hc : cs ≠ []) (hc' : cs' ≠ []) :
    ((FlockDeploy.next false x as cs).eval H).2 = ((FlockDeploy.next false x as cs').eval H).2 ∧
      (((FlockDeploy.next false x as cs).eval H).1 = .msg (cs.getLastD default).bytes ↔
        ((FlockDeploy.next false x as cs').eval H).1 = .msg (cs'.getLastD default).bytes) := by
  have hlast : ∀ {l : List Coin} (hl : l ≠ []), l.getLastD default = l.getLast hl := fun hl => by
    rw [List.getLastD_eq_getLast?, List.getLast?_eq_some_getLast hl, Option.getD_some]
  have hdec : ∀ {cs₀ : List Coin} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
      {sts : Array _root_.Flock.Stmt} {link : List ByteArray × List ByteArray} {c₀ : Coin} {mids : List (Msg × Coin)}
      {f : FlockDeploy.Final}, FlockDeploy.act false x as cs₀ = .decide s spec sts link c₀ mids f →
      FlockDeploy.next false x as cs₀ = FlockDeploy.outOf (FlockDeploy.decision false s x link c₀ mids f) :=
    fun h₀ => by simp only [FlockDeploy.next, h₀]
  rcases ha : FlockDeploy.act false x as cs with _ | c | ⟨s, spec, sts, link, c₁, mids, f⟩
  · rcases ha' : FlockDeploy.act false x as cs' with _ | c' | ⟨s', spec', sts', link', c₁', mids', f'⟩
    · rw [FlockDeploy.next_rej H ha, FlockDeploy.next_rej H ha']
      exact ⟨rfl, iff_of_false (fun h => nomatch h) fun h => nomatch h⟩
    · rw [act_coin_transfer hc ha'] at ha
      cases ha
    · obtain ⟨_, _, _, h⟩ := act_decide_transfer hc ha'
      rw [h] at ha
      cases ha
  · have hcs := Claims.Bridge.act_coin ha
    rw [List.getLast?_eq_some_getLast hc, Option.some.injEq] at hcs
    subst hcs
    rw [FlockDeploy.next_coin H ha, FlockDeploy.next_coin H (act_coin_transfer hc' ha), hlast hc, hlast hc']
    exact ⟨rfl, iff_of_true rfl rfl⟩
  · obtain ⟨link', c₁', mids', ha'⟩ := act_decide_transfer hc' ha
    rw [hdec ha, hdec ha']
    refine ⟨?_, iff_of_false (Claims.Bridge.outOf_ne_msg H _ _) (Claims.Bridge.outOf_ne_msg H _ _)⟩
    rw [FlockDeploy.KS.outOf_log, FlockDeploy.KS.outOf_log]
    exact decision_log_free H false s x link link' c₁ c₁' mids mids' f

/-- **`StepsLogFree` at every bound record whose `DE` steps as the deployed plain verifier** and answers a coin with
its bytes. -/
theorem stepsLogFree_of_V {Col T : Type} (H : Msg → Msg) (Bd : Bound Coin Col T) (hV : Bd.DE.V = FlockDeploy.V false)
    (hans : ∀ c, Bd.ans c = c.bytes) : Bd.StepsLogFree H := by
  intro x as cs cs' hne hc hc'
  have hpos : 0 < as.length := List.length_pos_iff.mpr hne
  rw [hV, hans, hans]
  exact next_log_free H x as (List.length_pos_iff.mp (by omega)) (List.length_pos_iff.mp (by omega))

/-- **`StepsLogFree` at the deployed bound record**, at the deployed plain verifier's own steps, whatever the code
side's other choices. -/
theorem stepsLogFree_deployed (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (H : Msg → Msg) :
    (bdDeployed rel (FlockDeploy.V false).next colsE wE join rowIn final w).StepsLogFree H :=
  stepsLogFree_of_V H _ rfl (bdDeployed_ans rel _ colsE wE join rowIn final w)

/-- **`ReadersLogCollS` at `IVStar`** wherever `DE`'s steps read no coin in their calls: the inner verifier's calls at
the run's coins are its calls at the coins `default` (`Bound.run_vLog_free`), each with `H`'s answer
(`Bound.run_vLog_ans`), so among the inner verifier's own the record searches (`innerCallsOf`). -/
theorem readersLogCollS_ivstar {Col T Col' T' : Type} [DecidableEq Col] {H : Msg → Msg} {Bd : Bound Coin Col T}
    (h : Bd.StepsLogFree H) (b : Brings) (DO : Claims.Flock.Deploy Coin Col' T') :
    ReadersLogCollS H Bd.IVStar b DO := by
  intro x as cs ws _ _ _ _ _
  left
  intro e he
  have he' : e ∈ (Claims.run H (Claims.replay (b.rd.msgs x ws)) Bd.IVStar (xIOf x) () fun _ => default).vLog := by
    unfold innerRunAt at he
    rwa [Bd.run_vLog_free h (xIOf x) (b.rd.msgs x ws) (fun i => cs.getD i.val default) fun _ => default] at he
  refine List.mem_append_right _ (List.mem_map.2 ⟨e.1, List.mem_append_right _ (List.mem_map.2 ⟨e, he', rfl⟩), ?_⟩)
  exact Prod.ext rfl (Bd.run_vLog_ans H (xIOf x) (b.rd.msgs x ws) _ e he').symm

end RecDeploy
