import Proofs.Flock.Claims.Recursion.DeploySessions

/-!
# The scope refusals on an accepted run

`DeployScope` proves the compiled verifier's scope refusals at one session decision (`decide_scope`). This file puts
them on a run: every V* session of an accepted run ends on a decision the compiled verifier doesn't refuse
(`run_decides`), at the session's last round. Before the last session, the next session's first round refuses a
session that ended on a coin (`decided`, which `run_round` reads); the last session's last round is the run's, which
accepts (`stepWith_acc_last`). So the compiled verifier refuses an accepted run whose inner shape the setup's checks
refuse or one of whose V* records doesn't draw every unit (`run_acc_scope`; `Draw.Full`, `run_acc_full`), and what it
reads of the statement with the sessions' records is in `InScope` (`run_acc_inScope_records`).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out Prover run coinSeq)
open FlockDeploy (Coin Final Ins)

/-- **A run that accepts ends on its acceptance**: an accepted run's last round is answered `.acc` (a run out of rounds
is refused). -/
theorem stepWith_acc_last {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vn : List Msg → Claims.OC Out) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Claims.Trace),
      (Claims.stepWith H P x ω vn f hist as t).out = .acc →
      ∃ a, (Claims.stepWith H P x ω vn f hist as t).msgs.getLast? = some (a, .acc)
  | 0, _, _, _, h => by simp [Claims.stepWith] at h
  | f + 1, hist, as, t, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg m =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      exact stepWith_acc_last H P x ω vn f _ _ _ h
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb]
      exact ⟨((P.next x hist ω).eval H).1, by simp⟩
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      cases h

section Runs

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

/-- **Round `i` of a run is session `s`'s decision** on the final message `f`: at the run's messages `ms` through
round `i` and its coins `r`, the compiled verifier takes session `s`'s round, where the outer verifier decides on
`f`. -/
def DecidesAt (ms : List (Msg × Out)) (i s : ℕ) (f : Final) : Prop :=
  ∃ (j : ℕ) (xO : Msg) (as' : List Msg) (cs' : List Coin) (ins : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Msg × Coin)),
    ract IV st x ((ms.take (i + 1)).map Prod.fst) ((List.range (i + 1)).map (coinSeq r)) = .sess s j xO as' cs' ∧
      actO xO as' cs' = .decide ins spec sts link c₁ mids f

/-- **Every session of an accepted run ends on a decision the compiled verifier doesn't refuse**: session `s`'s last
round is its decision. -/
theorem run_decides (hacc : (run H P (V IV st) x ω r).out = .acc) {s : ℕ} (hs : s < nS x) :
    ∃ i f, i < (run H P (V IV st) x ω r).msgs.length ∧ DecidesAt IV st x r (run H P (V IV st) x ω r).msgs i s f ∧
      ((next IV st x (((run H P (V IV st) x ω r).msgs.take (i + 1)).map Prod.fst)
        ((List.range (i + 1)).map (coinSeq r))).eval H).1 ≠ .rej := by
  obtain ⟨xr, hx⟩ := run_acc_readX H IV st x P ω r hacc
  have hK := run_acc_length H IV st x P ω r hacc
  set R := run H P (V IV st) x ω r with hR
  set xs := vstar st x (R.msgs.take (IV.k (xIOf x) + kS x)) with hxs
  have hxl : xs.length = nS x := by rw [hxs, vstar_length]
  have hpos := VO_k_pos (xs.getD s [])
  have hthrough := rounds_through xs (s := s) (by omega)
  have hKeq : (V IV st).k x = (IV.k (xIOf x) + kS x) + (xs.map VO.k).sum := by
    show _ + _ = _
    rw [hxs, vstar_map_k]
  have hsum := take_sum_le (xs.map VO.k) (s + 1)
  rw [← List.map_take] at hsum
  set off := (IV.k (xIOf x) + kS x) + ((xs.take s).map VO.k).sum with hoff
  set i := off + (VO.k (xs.getD s []) - 1) with hi
  have hlt : i < (V IV st).k x := by omega
  obtain ⟨new, hnew, hget⟩ := run_get H P (V IV st) x ω r i (by show i < R.msgs.length; omega)
  subst hnew
  set pa := ((P.next x ((R.msgs.take i).map histOf) ω).eval H).1 with hpa
  set ans := ((next IV st x ((R.msgs.take i).map Prod.fst ++ [pa]) ((List.range (i + 1)).map (coinSeq r))).eval H).1
    with hans
  have hget' : R.msgs[i]? = some (pa, ans) := hget
  have has : (R.msgs.take i).map Prod.fst ++ [pa] = (R.msgs.take (i + 1)).map Prod.fst := by
    rw [List.take_add_one, hget']
    simp
  have hml : ((R.msgs.take i).map Prod.fst).length = i := by
    simp only [List.length_map, List.length_take]
    omega
  have hlen : ((R.msgs.take i).map Prod.fst ++ [pa]).length = i + 1 := by
    simp only [List.length_append, hml, List.length_singleton]
  have hnrej : ans ≠ .rej := by
    intro h0
    have h2 : R.out = .rej :=
      stepWith_rej_out H P x ω (vnOf IV st x r) ((V IV st).k x) i [] [] ⟨[], [], [], .rej⟩ pa
        (by show R.msgs[0 + i]? = _; rw [Nat.zero_add, hget', h0])
    rw [h2] at hacc
    cases hacc
  have hip : innerPairs (IV.k (xIOf x) + kS x) ((R.msgs.take i).map Prod.fst ++ [pa])
      ((List.range (i + 1)).map (coinSeq r)) = R.msgs.take (IV.k (xIOf x) + kS x) :=
    innerPairs_run H IV st x P ω r hacc (by omega) hlt.le [pa] (range_map_take _ (by omega))
  have hl : locate ((vstar st x (innerPairs (IV.k (xIOf x) + kS x) ((R.msgs.take i).map Prod.fst ++ [pa])
      ((List.range (i + 1)).map (coinSeq r)))).map VO.k)
      (((R.msgs.take i).map Prod.fst ++ [pa]).length - 1 - (IV.k (xIOf x) + kS x)) =
        some (s, VO.k (xs.getD s []) - 1) := by
    rw [hip, ← hxs, hlen, show i + 1 - 1 - (IV.k (xIOf x) + kS x) =
      ((xs.map VO.k).take s).sum + (VO.k (xs.getD s []) - 1) by rw [← List.map_take]; omega]
    exact locate_eq _ (by simp only [List.length_map]; omega)
      (by rw [getD_map_k xs (s := s) (by omega)]; omega)
  have hr := ract_eq (st := st) hx (by omega) hl
  rw [hip, ← hxs, hlen, show i + 1 - 1 - (VO.k (xs.getD s []) - 1) = off by omega] at hr
  have hnr : ract IV st x ((R.msgs.take i).map Prod.fst ++ [pa]) ((List.range (i + 1)).map (coinSeq r)) ≠ .rej :=
    fun h0 => hnrej (by rw [hans, next_rej H h0])
  split_ifs at hr with h1 h2
  · exact absurd hr hnr
  · exact absurd hr hnr
  have hstep := next_sess H hr
  cases ha : actO (xs.getD s []) (((R.msgs.take i).map Prod.fst ++ [pa]).drop off)
      (((List.range (i + 1)).map (coinSeq r)).drop off) with
  | rej =>
    rw [nextO_rej H ha] at hstep
    exact absurd (by rw [hans, hstep]; rfl) hnrej
  | coin c =>
    exfalso
    rw [nextO_coin H ha] at hstep
    have hmsg : ans = .msg c.bytes := by rw [hans, hstep]; rfl
    by_cases hlast : s + 1 < nS x
    · obtain ⟨-, hd⟩ := run_round H IV st x P ω r hacc (s := s + 1) (off := i + 1) (xs := xs) hlast hxs
        (by rw [← rounds_through xs (s := s) (by omega)]; omega)
      have hd' := hd (by omega)
      rw [show s + 1 - 1 = s by omega, getD_map_k xs (s := s) (by omega),
        show i + 1 - VO.k (xs.getD s []) = off by omega, ← has] at hd'
      unfold decided at hd'
      rw [ha] at hd'
      cases hd'
    · have hiK : i + 1 = R.msgs.length := by
        rw [hK]
        have : ((xs.take (s + 1)).map VO.k).sum = (xs.map VO.k).sum := by
          rw [List.take_of_length_le (by omega)]
        omega
      obtain ⟨a, hla⟩ := stepWith_acc_last H P x ω (vnOf IV st x r) ((V IV st).k x) [] [] ⟨[], [], [], .rej⟩
        hacc
      have hla' : R.msgs[i]? = some (a, .acc) := by
        have hla2 : R.msgs.getLast? = some (a, .acc) := hla
        rwa [List.getLast?_eq_getElem?, show R.msgs.length - 1 = i by omega] at hla2
      rw [hget', hmsg] at hla'
      simp at hla'
  | decide ins spec sts link c₁ mids f =>
    refine ⟨i, f, by omega, ⟨VO.k (xs.getD s []) - 1, xs.getD s [], _, _, ins, spec, sts, link, c₁, mids, ?_, ha⟩, ?_⟩
    · rw [← has]
      exact hr
    · rw [← has]
      exact hnrej

/-- **The compiled verifier refuses an accepted run out of scope**: on an accepted run, the setup's checks accept the
inner shape `x` carries, and every V* session's decision is on a record that draws every unit of its population
(`RecScope.fullDrawOk`, hence `Draw.Full` by `RecScope.fullDrawOk_iff`). -/
theorem run_acc_scope (hacc : (run H P (V IV st) x ω r).out = .acc) :
    ∃ xr, readX x = some xr ∧ RecScope.accepts (readOf xr) = true ∧
      ∀ s < nS x, ∃ i f, i < (run H P (V IV st) x ω r).msgs.length ∧
        DecidesAt IV st x r (run H P (V IV st) x ω r).msgs i s f ∧ RecScope.fullDrawOk xr.shape f.record = true := by
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ hacc
  obtain ⟨xr, hx, -, -⟩ := acc_parse (IV := IV) (st := st) H ha
  have hd : ∀ s < nS x, ∃ i f, i < (run H P (V IV st) x ω r).msgs.length ∧
      DecidesAt IV st x r (run H P (V IV st) x ω r).msgs i s f ∧ RecScope.accepts (readOf xr) = true ∧
        RecScope.fullDrawOk xr.shape f.record = true := by
    intro s hs
    obtain ⟨i, f, hi, ⟨j, xO, as'', cs'', ins, spec, sts, link, c₁, mids, hr, hO⟩, hn⟩ :=
      run_decides H IV st x P ω r hacc hs
    obtain ⟨xr', hx', hacc', hf⟩ := decide_scope H hr hO hn
    rw [hx] at hx'
    cases hx'
    exact ⟨i, f, hi, ⟨j, xO, as'', cs'', ins, spec, sts, link, c₁, mids, hr, hO⟩, hacc', hf⟩
  refine ⟨xr, hx, ?_, fun s hs => ?_⟩
  · have hpos : 0 < nS x ∨ nS x = 0 := by omega
    rcases hpos with hpos | hz
    · obtain ⟨-, -, -, -, h, -⟩ := hd 0 hpos
      exact h
    · obtain ⟨_, -, hp, -⟩ := acc_parse (IV := IV) (st := st) H ha
      simp only [parse, RecScope.parseOf, RecScope.parseBytes, decodeR, hx, Option.map_some, Option.filter] at hp
      split_ifs at hp with hh
      · exact hh
      · cases hp
  · obtain ⟨i, f, hi, hD, -, hf⟩ := hd s hs
    exact ⟨i, f, hi, hD, hf⟩

/-- **Every V\* session of an accepted run draws every unit**: its decision's record holds a draw, and it is the full
draw (`Draw.Full`). -/
theorem run_acc_full (hacc : (run H P (V IV st) x ω r).out = .acc) :
    ∀ s < nS x, ∃ i f, i < (run H P (V IV st) x ω r).msgs.length ∧
      DecidesAt IV st x r (run H P (V IV st) x ω r).msgs i s f ∧
        ∃ e, RecScope.recordDrawOf f.record = some e ∧ e.Full := by
  obtain ⟨xr, -, -, hall⟩ := run_acc_scope H IV st x P ω r hacc
  intro s hs
  obtain ⟨i, f, hi, hD, hf⟩ := hall s hs
  exact ⟨i, f, hi, hD, (RecScope.fullDrawOk_iff xr.shape f.record).1 hf⟩

/-- **An accepted run is in scope with its records**: what the compiled verifier reads of `x` with the records each
V* session decided on (`RecScope.toStmt`, at `readOf` with those records) is in `InScope`. -/
theorem run_acc_inScope_records (hacc : (run H P (V IV st) x ω r).out = .acc) :
    ∃ xr, readX x = some xr ∧ ∃ fs : Fin (nS x) → Final,
      (∀ s : Fin (nS x), ∃ i, i < (run H P (V IV st) x ω r).msgs.length ∧
        DecidesAt IV st x r (run H P (V IV st) x ω r).msgs i s (fs s)) ∧
      Claims.Recursion.InScope
        (RecScope.toStmt { readOf xr with records := (List.finRange (nS x)).map fun s => (fs s).record }) := by
  obtain ⟨xr, hx, hacc', hall⟩ := run_acc_scope H IV st x P ω r hacc
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ hacc
  obtain ⟨xr', hx', -, hin⟩ := acc_parse (IV := IV) (st := st) H ha
  rw [hx] at hx'
  cases hx'
  have hdn : RecScope.dense xr.shape = true := ((RecScope.inScope_iff hacc').1 hin).2
  have hch : ∀ s : Fin (nS x), ∃ f : Final, ∃ i, i < (run H P (V IV st) x ω r).msgs.length ∧
      DecidesAt IV st x r (run H P (V IV st) x ω r).msgs i s f ∧ RecScope.fullDrawOk xr.shape f.record = true := by
    intro s
    obtain ⟨i, f, hi, hD, hf⟩ := hall s s.isLt
    exact ⟨f, i, hi, hD, hf⟩
  choose fs hfs using hch
  refine ⟨xr, hx, fs, fun s => ?_, ?_⟩
  · obtain ⟨i, hi, hD, -⟩ := hfs s
    exact ⟨i, hi, hD⟩
  · refine (RecScope.inScope_iff
      (r := { readOf xr with records := (List.finRange (nS x)).map fun s => (fs s).record }) hacc').2
      ⟨List.all_eq_true.2 fun rec hm => ?_, hdn⟩
    obtain ⟨s, -, rfl⟩ := List.mem_map.1 hm
    obtain ⟨-, -, -, hf⟩ := hfs s
    exact hf

end Runs

end RecDeploy
