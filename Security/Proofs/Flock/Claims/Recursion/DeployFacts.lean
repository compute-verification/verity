import Proofs.Flock.Claims.Recursion.Deploy

/-!
# The compiled verifier's structural code facts

The code facts the recursion's analysis reads of a deployment that follow from `deployRec`'s definition alone, each
stated as the analysis states it (`Refuses`, `InnerRounds`, `SessionNums`, `SessionRounds`, `TableErrNonneg`,
`SessionAccepts`, `InnerViewRun`), at `deployRec H I.V …` and the outer record `deployZkRec rel`:

* `refuses`: a run accepts only at a step where the compiled verifier's does (`stepWith_acc`), which is a session's
  (`next_acc`), so the statement parses (`readX`) and `parse` reads no circuit commitment.
* `innerRounds`, `sessionRounds`: `k(x)` is the inner rounds and every session's rounds at `x`'s numbers, and every V*
  statement `vstar` gives has those numbers (`vstar_map_k`).
* `sessionNums`, `tableErrNonneg`: `vstar`'s statements have `nums`' numbers (`numsOf_vstarAt`), which are
  `deployZkRec`'s.
* `sessionAccepts`: `session` reads an accepted run's sessions as accepted.
* `innerViewRun`: the inner view is `I.V`'s run of the readers' messages at the oracle the record runs it against, and a
  run of replayed messages is the run of the messages it shows (`stepWith_replay`), so with no hypothesis on the
  witnesses.
-/

namespace RecDeploy

open FlockDeploy (Coin)

/-! ## Runs -/

/-- **A run accepts only where its verifier's step does.** -/
theorem stepWith_acc {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      (Claims.stepWith H P x ω vnext n hist as t).out = .acc → ∃ as', ((vnext as').eval H).1 = .acc
  | 0, _, _, _, h => by simp [Claims.stepWith] at h
  | n + 1, hist, as, t, h => by
    simp only [Claims.stepWith] at h
    split at h
    · exact stepWith_acc H P x ω vnext n _ _ _ h
    · exact ⟨_, by assumption⟩
    · simp at h

/-- **A replayed run plays the replayed messages in order, and only those it plays decide it**: from history `hist`,
the run's `m` new messages are the replayed ones from `hist.length` on, and any replay that agrees on those has the same
run. -/
theorem stepWith_replay (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (vnext : List Claims.Msg → Claims.OC Claims.Out)
    (ms : List Claims.Msg) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      ∃ m, (Claims.stepWith H (Claims.replay ms) x () vnext n hist as t).msgs.map Prod.fst =
          t.msgs.map Prod.fst ++ (List.range m).map (fun i => ms.getD (hist.length + i) []) ∧
        ∀ ms' : List Claims.Msg, (∀ i < m, ms'.getD (hist.length + i) [] = ms.getD (hist.length + i) []) →
          Claims.stepWith H (Claims.replay ms') x () vnext n hist as t =
            Claims.stepWith H (Claims.replay ms) x () vnext n hist as t
  | 0, hist, as, t => ⟨0, by simp [Claims.stepWith], fun _ _ => rfl⟩
  | n + 1, hist, as, t => by
    rcases hvb : (vnext (as ++ [ms.getD hist.length []])).eval H with ⟨o, log⟩
    cases o with
    | msg m =>
      obtain ⟨k, hk, hag⟩ := stepWith_replay H x vnext ms n (hist ++ [(ms.getD hist.length [], m)])
        (as ++ [ms.getD hist.length []])
        ⟨t.msgs ++ [(ms.getD hist.length [], .msg m)], t.pLog ++ [], t.vLog ++ log, t.out⟩
      refine ⟨k + 1, ?_, fun ms' hms => ?_⟩
      · simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, hvb]
        refine hk.trans ?_
        simp only [List.map_append, List.map_cons, List.map_nil, List.length_append, List.length_cons,
          List.length_nil, List.append_assoc, List.range_succ_eq_map, List.map_cons, List.map_map]
        simp only [List.map_inj_left, Function.comp, List.cons_append, List.nil_append, List.append_cancel_left_eq,
          List.cons.injEq, Nat.add_zero, true_and]
        intro a _
        rw [Nat.add_assoc, Nat.add_comm 1 a]
      · have h0 : ms'.getD hist.length [] = ms.getD hist.length [] := by simpa using hms 0 (by omega)
        simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, h0, hvb]
        exact hag ms' fun i hi => by
          simpa [Nat.add_assoc, Nat.add_comm 1] using hms (i + 1) (by omega)
    | acc =>
      refine ⟨1, ?_, fun ms' hms => ?_⟩
      · simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, hvb]
        simp
      · have h0 : ms'.getD hist.length [] = ms.getD hist.length [] := by simpa using hms 0 (by omega)
        simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, h0, hvb]
    | rej =>
      refine ⟨1, ?_, fun ms' hms => ?_⟩
      · simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, hvb]
        simp
      · have h0 : ms'.getD hist.length [] = ms.getD hist.length [] := by simpa using hms 0 (by omega)
        simp only [Claims.stepWith, Claims.replay, Claims.OC.eval, h0, hvb]

/-- **A replayed run is the run of the messages it shows.** -/
theorem run_replay_seen {Coin' : Type} [Inhabited Coin'] (H : Claims.Msg → Claims.Msg) (V' : Claims.Verifier Coin')
    (x : Claims.Msg) (ms : List Claims.Msg) (c : Fin (V'.k x) → Coin') :
    Claims.run H (Claims.replay ((Claims.run H (Claims.replay ms) V' x () c).seen.msgs.map Prod.fst)) V' x () c =
      Claims.run H (Claims.replay ms) V' x () c := by
  obtain ⟨m, hm, hag⟩ := stepWith_replay H x (fun as => V'.next x as ((List.range as.length).map (Claims.coinSeq c)))
    ms (V'.k x) [] [] ⟨[], [], [], .rej⟩
  refine hag _ fun i hi => ?_
  show (((Claims.run H (Claims.replay ms) V' x () c).msgs.map Prod.fst).getD ([].length + i) []) = _
  unfold Claims.run
  rw [hm]
  simp [List.getD_eq_getElem?_getD, hi]

/-! ## The code facts -/

variable {W : Type}

theorem ract_readX {IV : Claims.Verifier Coin} {st : Stage} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s j : ℕ} {xO : Claims.Msg} {as' : List Claims.Msg} {cs' : List Coin}
    (h : ract IV st x as cs = .sess s j xO as' cs') : ∃ r, readX x = some r := by
  unfold ract at h
  split at h
  · cases h
  · exact ⟨_, by assumption⟩

/-- **`Refuses`**: every accepted run is of a statement that parses, with no circuit commitment. -/
theorem refuses (Hr : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (st : Stage)
    (rd : Readers) (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((deployRec Hr IV ivScope st rd).V.k x) → Coin)
    (h : (Claims.run H P (deployRec Hr IV ivScope st rd).V x ω r).out = .acc) :
    ∃ s, (deployRec Hr IV ivScope st rd).parse x = some s ∧ s.circ = none := by
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ h
  obtain ⟨s, j, xO, as'', cs', hr, -⟩ := next_acc (IV := IV) (st := st) H ha
  obtain ⟨xr, hx⟩ := ract_readX hr
  refine ⟨⟨xr.w.inner, (xr.dims.m, xr.dims.kLog, xr.dims.regions), none, formOf xr⟩, ?_, rfl⟩
  simp [deployRec, parse, hx]

/-- **`InnerRounds`**: the compiled verifier takes at least the inner verifier's rounds. -/
theorem innerRounds (H : Claims.Msg → Claims.Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) (x : Claims.Msg) :
    I.V.k (Claims.Recursion.xI (deployRec H I.V ivScope st rd) x) ≤ (deployRec H I.V ivScope st rd).V.k x :=
  Nat.le_add_right _ _

/-- **`SessionNums`**: whatever the inner rounds, part `s`'s V* statement has `nums`' rounds, waits, truncation, `N₀`,
table error and weight under the outer record. -/
theorem sessionNums (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (x : Claims.Msg)
    (s : Fin ((deployRec H IV ivScope st rd).nS x)) (ms : List (Claims.Msg × Claims.Out)) :
    let D := deployRec H IV ivScope st rd
    let DO := deployZkRec rel
    DO.V.k ((D.vstar x ms).getD s []) = (D.nums x s).rounds ∧
      DO.waits ((D.vstar x ms).getD s []) = (D.nums x s).waits ∧
      DO.cap ((D.vstar x ms).getD s []) = (D.nums x s).cap ∧
      DO.N₀ ((D.vstar x ms).getD s []) = (D.nums x s).N₀ ∧
      DO.εtable ((D.vstar x ms).getD s []) = (D.nums x s).εtable ∧
      DO.weight ((D.vstar x ms).getD s []) = (D.nums x s).weight := by
  obtain ⟨s, hs⟩ := s
  dsimp only [deployRec]
  rw [vstar_getD st x ms hs, ← numsOf_vstarAt st x ms s]
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem sum_range_le (f : ℕ → ℕ) {a b : ℕ} (h : a ≤ b) :
    ((List.range a).map f).sum ≤ ((List.range b).map f).sum := by
  induction b, h using Nat.le_induction with
  | base => exact le_rfl
  | succ n _ ih =>
    rw [List.range_succ, List.map_append, List.sum_append]
    omega

/-- **`SessionRounds`**: session `s`'s rounds, after the inner rounds and the earlier sessions', end by `k(x)`. -/
theorem sessionRounds (H : Claims.Msg → Claims.Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) (rel : Flock.Readable.Parsed → Flock.Readable.Relation) (x : Claims.Msg)
    (s : Fin ((deployRec H I.V ivScope st rd).nS x)) (ms : List (Claims.Msg × Claims.Out)) :
    let D := deployRec H I.V ivScope st rd
    let DO := deployZkRec rel
    I.V.k (Claims.Recursion.xI D x) + (((D.vstar x ms).take s).map DO.V.k).sum + DO.V.k ((D.vstar x ms).getD s []) ≤
      D.V.k x := by
  obtain ⟨s, hs⟩ := s
  replace hs : s < nS x := hs
  show I.V.k (xIOf x) + (((vstar st x ms).take s).map VO.k).sum + VO.k ((vstar st x ms).getD s []) ≤
    I.V.k (xIOf x) + ((List.range (nS x)).map fun s => (nums st x s).rounds).sum
  have hk : VO.k ((vstar st x ms).getD s []) = (nums st x s).rounds := by
    rw [vstar_getD st x ms hs]
    exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt st x ms s)
  rw [List.map_take, vstar_map_k, ← List.map_take, List.take_range, Nat.min_eq_left hs.le, hk]
  have := sum_range_le (fun s => (nums st x s).rounds) (Nat.succ_le_of_lt hs)
  rw [List.range_succ, List.map_append, List.sum_append] at this
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero] at this
  omega

/-- **`TableErrNonneg`**: every session's table error is nonnegative. -/
theorem tableErrNonneg (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) (x : Claims.Msg) (s : Fin ((deployRec H IV ivScope st rd).nS x)) :
    0 ≤ ((deployRec H IV ivScope st rd).nums x s).εtable :=
  εtableO_nonneg _

/-- **`SessionAccepts`**: an accepted run shows each V* session accepting. -/
theorem sessionAccepts (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) (x : Claims.Msg) (s : ℕ) (v : Claims.Seen)
    (_ : s < (deployRec H IV ivScope st rd).nS x) (hv : v.out = .acc) :
    ((deployRec H IV ivScope st rd).session x s v).out = .acc := by
  simp [deployRec, session, hv]

/-- **`InnerViewRun`, at the record's oracle and with no hypothesis on the witnesses**: the inner view is `I.V`'s run,
at the compiled verifier's first `k_I` coins, of the messages it shows. -/
theorem innerViewRun (H : Claims.Msg → Claims.Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Claims.Msg → Bool)
    (st : Stage) (rd : Readers) (x : Claims.Msg) (r : Fin ((deployRec H I.V ivScope st rd).V.k x) → Coin)
    (ws : List Claims.Flock.Wit) :
    let D := deployRec H I.V ivScope st rd
    D.innerView x r ws =
      (Claims.run H (Claims.replay ((D.innerView x r ws).msgs.map Prod.fst)) I.V (Claims.Recursion.xI D x) ()
        fun i => Claims.coinSeq r i.val).seen := by
  show innerView H I.V rd x r ws =
    (Claims.run H (Claims.replay ((innerView H I.V rd x r ws).msgs.map Prod.fst)) I.V (xIOf x) ()
      fun i => Claims.coinSeq r i.val).seen
  unfold innerView
  rw [run_replay_seen]

end RecDeploy
