import Proofs.Flock.Claims.Recursion.BoundRoute

/-!
# The bridge at the bound record (option (d))

`Claims.Recursion.Bridge`: `Inner.KSAbort` at `IVStar` (`Bound.deploy`) from `Inner.KSAbort` at the element-level Flock
`DE`, where the commit strings bind (`Bound.Binds`) and `γ`'s weight covers the loss (`Bound.WeightCovers`).

From a prover `P` against `IVStar` and its aborts `keep`, the plurality prover (`Bound.plurP`) plays `DE`: at each round,
on the coins so far (`IVStar` is public-coin), it sends the element-level message that `IVStar`'s accepted continuations
of those coins open most often there (`Bound.plurAt`). Its run is kept where `P`'s run against `IVStar` accepts and opens
the plurality at every round (`Bound.plurKeep`); there it is `P`'s run opened (`Bound.plurRun`), so `E^rep` at `DE`
sees what `E^rep` at `IVStar` sees, off the runs that open otherwise (`Bound.BadA`). So:

* `E^rep` at `IVStar` fails at most as `E^rep` at `DE` fails against the plurality prover's aborting runs, plus `N_I`
  times one run's chance to be bad (`Bound.PrBadAbort_le`, Wald's identity at `E^rep`'s reruns);
* `B` at the plurality prover's aborting runs finds only collisions `B` at `P`'s finds (`Bound.PrBAbort_le`): a kept
  run's calls are among `P`'s run's;
* a run is bad at most as `k_I · √Pr[B]` (`Bound.Pr_badA_le`): off the plurality at round `j` at most as
  `√(Pr[two runs forked at j open round j differently])` (`Pr_plur_fork`), and two accepted runs that share their
  coins before `j` and open round `j` differently open one commit string two ways (`Bound.fork_tied`), a collision
  `B`'s fork at `j` finds; the rounds' roots sum to at most `k_I · √Pr[B]` (Cauchy–Schwarz);
* `WeightCovers` turns `N_I · k_I · √Pr[B]` into the difference of `√γ` at the two records (`Bound.bridge_of_dec`).

At the deployed bound record `ans` is a coin's bytes, which a decoder inverts, so the bridge holds there at every
choice the code side makes (`RecDeploy.bridge_bdDeployed`), and `KSBoundRoute` holds (`RecDeploy.ksBoundRoute`).
-/

namespace Claims.Recursion

open Claims

/-! ## Oracle programs -/

theorem eval_bindOC_full (H : Msg → Msg) {α β : Type} (p : OC α) (f : α → OC β) :
    OC.eval H (bindOC p f) =
      ((OC.eval H (f (OC.eval H p).1)).1, (OC.eval H p).2 ++ (OC.eval H (f (OC.eval H p).1)).2) := by
  induction p with
  | ret a => simp [bindOC, OC.eval]
  | query q k ih => simp only [bindOC, OC.eval, ih, List.cons_append]

/-- **Programs in order all return `true`**: each does, its calls among theirs. -/
theorem allOC_true (H : Msg → Msg) : ∀ (ps : List (OC Bool)), ((allOC ps).eval H).1 = true →
    ∀ p ∈ ps, (p.eval H).1 = true ∧ ∀ e ∈ (p.eval H).2, e ∈ ((allOC ps).eval H).2
  | [], _, _, hq => by simp at hq
  | p :: ps, h, q, hq => by
    have he := eval_bindOC_full H p fun b => if b then allOC ps else .ret false
    simp only [allOC] at h ⊢
    rw [he] at h ⊢
    by_cases hb : (p.eval H).1 = true
    · rw [ite_eq_left hb] at h ⊢
      rcases List.mem_cons.1 hq with rfl | hq
      · exact ⟨hb, fun e he => List.mem_append_left _ he⟩
      · obtain ⟨h1, h2⟩ := allOC_true H ps h q hq
        exact ⟨h1, fun e he => List.mem_append_right _ (h2 e he)⟩
    · rw [ite_eq_right hb] at h
      simp [OC.eval] at h

/-! ## Lists -/

theorem getD_range_map {α : Type} (f : ℕ → α) (d : α) {j k : ℕ} (h : j < k) :
    ((List.range k).map f).getD j d = f j := by
  rw [List.getD_eq_getElem _ _ (by simpa using h)]
  simp

theorem take_eq_range_getD (l : List Msg) {n : ℕ} (h : n ≤ l.length) :
    l.take n = (List.range n).map fun i => l.getD i [] := by
  refine List.ext_getElem (by rw [List.length_take, List.length_map, List.length_range]; omega) fun i h1 h2 => ?_
  rw [List.length_take] at h1
  simp only [List.getElem_take, List.getElem_map, List.getElem_range]
  rw [List.getD_eq_getElem _ _ (by omega)]

theorem take_range_map {α : Type} (f : ℕ → α) {n k : ℕ} (h : n ≤ k) :
    ((List.range k).map f).take n = (List.range n).map f := by
  refine List.ext_getElem (by simp; omega) fun i h1 h2 => ?_
  simp

theorem zip_range_map {β : Type} (l : List Msg) (g : ℕ → β) {k : ℕ} (h : l.length = k) :
    l.zip ((List.range k).map g) = (List.range k).map fun j => (l.getD j [], g j) := by
  refine List.ext_getElem (by simp [h]) fun i h1 h2 => ?_
  simp only [List.getElem_zip, List.getElem_map, List.getElem_range]
  rw [List.getD_eq_getElem _ _ (by simp at h2; omega)]

/-! ## Replayed runs -/

/-- **Replaying a prover's own messages gives its messages, its verifier's calls and its decision**: at each round the
replayed message is the prover's, so the verifier answers alike. -/
theorem stepWith_replay_same {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vn : List Msg → OC Out) (ms : List Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Trace), hist.length = t.msgs.length →
      t'.msgs = t.msgs → t'.vLog = t.vLog →
      (∀ j, t.msgs.length ≤ j → j < (stepWith H P x ω vn n hist as t).msgs.length →
        ms.getD j [] = ((stepWith H P x ω vn n hist as t).msgs.map Prod.fst).getD j []) →
      (stepWith H (replay ms) x () vn n hist as t').msgs = (stepWith H P x ω vn n hist as t).msgs ∧
        (stepWith H (replay ms) x () vn n hist as t').vLog = (stepWith H P x ω vn n hist as t).vLog ∧
        (stepWith H (replay ms) x () vn n hist as t').out = (stepWith H P x ω vn n hist as t).out
  | 0, _, _, _, _, _, hm, hv, _ => ⟨hm, hv, rfl⟩
  | n + 1, hist, as, t, t', hl, hm, hvl, h => by
    have hlt := stepWith_length_gt H P x ω vn (Nat.succ_pos n) hist as t
    have h2 : ((stepWith H P x ω vn (n + 1) hist as t).msgs.map Prod.fst)[t.msgs.length]? =
        some ((P.next x hist ω).eval H).1 := by
      have := congrArg (·[t.msgs.length]?) (stepWith_succ_fst_take H P x ω vn n hist as t)
      simpa [List.getElem?_take] using this
    have hrep : ms.getD hist.length [] = ((P.next x hist ω).eval H).1 := by
      rw [hl, h _ le_rfl hlt, List.getD_eq_getElem?_getD, h2]
      rfl
    have e1 : ((replay ms).next x hist ()).eval H = (((P.next x hist ω).eval H).1, []) := by
      simp only [replay, OC.eval, hrep]
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · have hv' : ((vn (as ++ [(((replay ms).next x hist ()).eval H).1])).eval H).1 = .msg m := by
        rw [e1]
        exact hv
      rw [stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      rw [stepWith_msg H (replay ms) x () vn n hist as t' hv']
      simp only [e1]
      exact stepWith_replay_same H P x ω vn ms n _ _ _ _ (by simp [hl]) (by simp [hm]) (by simp [hvl])
        fun j hj1 hj2 => h j (by simp only [List.length_append, List.length_singleton] at hj1; omega) hj2
    · simp [stepWith, e1, hv, hm, hvl]
    · simp [stepWith, e1, hv, hm, hvl]

/-- **Replaying a run's own messages gives its messages, its verifier's calls and its decision.** -/
theorem run_replay_same {C : Type} [Inhabited C] {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier C)
    (x : Msg) (ω : Ω) (r : Fin (V.k x) → C) :
    (run H (replay ((run H P V x ω r).msgs.map Prod.fst)) V x () r).msgs = (run H P V x ω r).msgs ∧
      (run H (replay ((run H P V x ω r).msgs.map Prod.fst)) V x () r).vLog = (run H P V x ω r).vLog ∧
      (run H (replay ((run H P V x ω r).msgs.map Prod.fst)) V x () r).out = (run H P V x ω r).out :=
  stepWith_replay_same H P x ω _ _ (V.k x) [] [] _ _ rfl rfl rfl fun _ _ _ => rfl

/-- **Replaying messages to a verifier whose answers are known**: a message at each round before the last and
acceptance at the last, so the run plays every round and accepts, its messages the replayed ones with those answers and
its calls the verifier's at each prefix. -/
theorem stepWith_replay_known (H : Msg → Msg) (x : Msg) (vn : List Msg → OC Out) (ms : List Msg) (k : ℕ)
    (b : ℕ → Out) (hv : ∀ j < k, ((vn ((List.range (j + 1)).map fun i => ms.getD i [])).eval H).1 = b j)
    (hb : ∀ j, j + 1 < k → ∃ m, b j = .msg m) (hlast : b (k - 1) = .acc) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), hist.length = t.msgs.length →
      as = (List.range t.msgs.length).map (fun i => ms.getD i []) → t.msgs.length + (n + 1) = k →
      stepWith H (replay ms) x () vn (n + 1) hist as t =
        { msgs := t.msgs ++ (List.range' t.msgs.length (n + 1)).map (fun i => (ms.getD i [], b i)),
          pLog := t.pLog,
          vLog := t.vLog ++ ((List.range' t.msgs.length (n + 1)).map fun i =>
            ((vn ((List.range (i + 1)).map fun i => ms.getD i [])).eval H).2).flatten,
          out := .acc } := by
  intro n
  induction n with
  | zero =>
    intro hist as t hh ha hk
    have hp : ((replay ms).next x hist ()).eval H = (ms.getD t.msgs.length [], []) := by
      show (OC.ret (ms.getD hist.length [])).eval H = _
      rw [hh]; rfl
    have has : as ++ [ms.getD t.msgs.length []] = (List.range (t.msgs.length + 1)).map fun i => ms.getD i [] := by
      rw [ha, List.range_succ, List.map_append, List.map_singleton]
    have hbL : b t.msgs.length = .acc := by rw [show t.msgs.length = k - 1 by omega]; exact hlast
    have hvL : ((vn ((List.range (t.msgs.length + 1)).map fun i => ms.getD i [])).eval H).1 = .acc := by
      rw [hv _ (by omega), hbL]
    simp only [stepWith, hp, has, hvL, Nat.zero_add, List.range'_one, List.map_cons, List.map_nil, hbL,
      List.flatten_cons, List.flatten_nil, List.append_nil]
  | succ n ih =>
    intro hist as t hh ha hk
    have hp : ((replay ms).next x hist ()).eval H = (ms.getD t.msgs.length [], []) := by
      show (OC.ret (ms.getD hist.length [])).eval H = _
      rw [hh]; rfl
    have has : as ++ [ms.getD t.msgs.length []] = (List.range (t.msgs.length + 1)).map fun i => ms.getD i [] := by
      rw [ha, List.range_succ, List.map_append, List.map_singleton]
    obtain ⟨m, hm⟩ := hb t.msgs.length (by omega)
    have hvL : ((vn ((List.range (t.msgs.length + 1)).map fun i => ms.getD i [])).eval H).1 = b t.msgs.length :=
      hv _ (by omega)
    have hv1 : ((vn (as ++ [(((replay ms).next x hist ()).eval H).1])).eval H).1 = .msg m := by
      rw [hp, has, hvL, hm]
    rw [stepWith_msg H (replay ms) x () vn (n + 1) hist as t hv1]
    refine (ih _ _ _ ?_ ?_ ?_).trans ?_
    · simp only [List.length_append, List.length_singleton, hh]
    · simp only [List.length_append, List.length_singleton, hp, has]
    · simp only [List.length_append, List.length_singleton]
      omega
    · simp only [hp, has, hvL, List.length_append, List.length_singleton, List.append_assoc, List.range'_succ,
        List.map_cons, List.flatten_cons, List.singleton_append, List.append_nil]

/-- **A replayed run at known answers, in closed form**: it plays every round, answered as known, accepts, and its
calls are the verifier's at the prefixes. -/
theorem run_replay_known {C : Type} [Inhabited C] (H : Msg → Msg) (V : Verifier C) (x : Msg) (ms : List Msg)
    (r : Fin (V.k x) → C) (b : ℕ → Out) (hk : 0 < V.k x)
    (hv : ∀ j < V.k x, ((V.next x ((List.range (j + 1)).map fun i => ms.getD i [])
      ((List.range (j + 1)).map (coinSeq r))).eval H).1 = b j)
    (hb : ∀ j, j + 1 < V.k x → ∃ m, b j = .msg m) (hlast : b (V.k x - 1) = .acc) :
    (run H (replay ms) V x () r).msgs = (List.range (V.k x)).map (fun i => (ms.getD i [], b i)) ∧
      (run H (replay ms) V x () r).out = .acc ∧
      ∀ e ∈ (run H (replay ms) V x () r).vLog, ∃ j < V.k x,
        e ∈ ((V.next x ((List.range (j + 1)).map fun i => ms.getD i [])
          ((List.range (j + 1)).map (coinSeq r))).eval H).2 := by
  have hrun : run H (replay ms) V x () r = stepWith H (replay ms) x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (V.k x - 1 + 1) [] [] ⟨[], [], [], .rej⟩ := by
    unfold run
    rw [Nat.sub_add_cancel (by omega : 1 ≤ V.k x)]
  have h := stepWith_replay_known H x (fun as => V.next x as ((List.range as.length).map (coinSeq r))) ms (V.k x) b
    (fun j hj => by simpa using hv j hj) hb hlast (V.k x - 1) [] [] ⟨[], [], [], .rej⟩ rfl (by simp)
    (by simp only [List.length_nil]; omega)
  rw [hrun, h]
  simp only [List.length_nil, List.nil_append, ← List.range_eq_range', Nat.sub_add_cancel (by omega : 1 ≤ V.k x)]
  refine ⟨trivial, trivial, fun e he => ?_⟩
  obtain ⟨l, hl, hel⟩ := List.mem_flatten.1 he
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hl
  exact ⟨j, List.mem_range.1 hj, by simpa using hel⟩

section Bound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-! ## What `IVStar` checks when it accepts -/

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Bound.opens_eval (Bd : Bound Coin Col T) (H : Msg → Msg) (a row : Msg) (y : Fin Bd.saltLen → UInt8) :
    (Bd.opens a row y).eval H =
      (decide (H (Bd.saltPrefix ++ List.ofFn y) = a.drop 64 ∧ H (Bd.rowIn row) = xorMask (a.take 64) (Bd.mask y)),
        [(Bd.saltPrefix ++ List.ofFn y, H (Bd.saltPrefix ++ List.ofFn y)), (Bd.rowIn row, H (Bd.rowIn row))]) :=
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **Two openings of one commit string to different rows or salts collide** among their calls (`Bound.opens`), where
`rowIn` is injective: on the salts' hashes where the salts differ, on the rows' digests where they share it. -/
theorem Bound.opens_coll (Bd : Bound Coin Col T) (H : Msg → Msg) (hB : Bd.Binds) {a row row' : Msg}
    {y y' : Fin Bd.saltLen → UInt8} (h : ((Bd.opens a row y).eval H).1 = true)
    (h' : ((Bd.opens a row' y').eval H).1 = true) (hne : (row, y) ≠ (row', y')) :
    ∃ u ∈ ((Bd.opens a row y).eval H).2.map Prod.fst, ∃ v ∈ ((Bd.opens a row' y').eval H).2.map Prod.fst,
      Collision H u v := by
  rw [Bd.opens_eval H a row y] at h ⊢
  rw [Bd.opens_eval H a row' y'] at h' ⊢
  simp only [decide_eq_true_eq] at h h'
  by_cases hy : y = y'
  · subst hy
    have hr : row ≠ row' := fun hr => hne (by rw [hr])
    exact ⟨Bd.rowIn row, by simp, Bd.rowIn row', by simp, fun he => hr (hB he), by rw [h.2, h'.2]⟩
  · exact ⟨Bd.saltPrefix ++ List.ofFn y, by simp, Bd.saltPrefix ++ List.ofFn y', by simp,
      fun he => hy (List.ofFn_injective (List.append_cancel_left he)), by rw [h.1, h'.1]⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- **A round message that opens**: it is the rows' count of 128-byte commit strings, each opening to its row. -/
theorem Bound.opensRound_true (Bd : Bound Coin Col T) (H : Msg → Msg) {a : Msg}
    {os : List (Msg × (Fin Bd.saltLen → UInt8))} (h : ((Bd.opensRound a os).eval H).1 = true) :
    a.length = 128 * os.length ∧ ∀ (i : ℕ) (hi : i < os.length),
      ((Bd.opens ((a.drop (128 * i)).take 128) os[i].1 os[i].2).eval H).1 = true ∧
        ∀ e ∈ ((Bd.opens ((a.drop (128 * i)).take 128) os[i].1 os[i].2).eval H).2,
          e ∈ ((Bd.opensRound a os).eval H).2 := by
  unfold Bound.opensRound at h ⊢
  by_cases hl : a.length = 128 * os.length
  · rw [ite_eq_left hl] at h ⊢
    refine ⟨hl, fun i hi => ?_⟩
    have hm : Bd.opens ((a.drop (128 * i)).take 128) os[i].1 os[i].2 ∈
        os.mapIdx fun i o => Bd.opens ((a.drop (128 * i)).take 128) o.1 o.2 := by
      have := List.getElem_mem (l := os.mapIdx fun i o => Bd.opens ((a.drop (128 * i)).take 128) o.1 o.2)
        (by simpa using hi)
      rwa [List.getElem_mapIdx] at this
    exact allOC_true H _ h _ hm
  · rw [ite_eq_right hl] at h
    simp [OC.eval] at h

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- **The element-level verifier accepts the opened messages**: at each round it answers as `IVStar` does (the round's
coin before the last, acceptance at the last), its calls among `agrees`'. -/
theorem Bound.agrees_true (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {ms : List Msg} {cs : List Coin}
    (h : ((Bd.agrees x ms cs).eval H).1 = true) (j : ℕ) (hj : j < ms.length) :
    ((Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).1 =
        (if j + 1 < ms.length then .msg (Bd.ans (cs.getD j default)) else .acc) ∧
      ∀ e ∈ ((Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))).eval H).2,
        e ∈ ((Bd.agrees x ms cs).eval H).2 := by
  have hm : (bindOC (Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))) fun b =>
      OC.ret (decide (b = if j + 1 < ms.length then .msg (Bd.ans (cs.getD j default)) else .acc))) ∈
      (List.range ms.length).map fun j => bindOC (Bd.DE.V.next x (ms.take (j + 1)) (cs.take (j + 1))) fun b =>
        OC.ret (decide (b = if j + 1 < ms.length then .msg (Bd.ans (cs.getD j default)) else .acc)) :=
    List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩
  obtain ⟨h1, h2⟩ := allOC_true H _ h _ hm
  rw [eval_bindOC_full] at h1 h2
  simp only [OC.eval, decide_eq_true_eq, List.append_nil] at h1 h2
  exact ⟨h1, h2⟩

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- **What `IVStar` checks when it accepts at its last round**: `final` reads the last message, the schedule's counts
hold, each earlier round opens and the element-level verifier accepts the opened messages, each check's calls among the
decision's. -/
theorem Bound.ivstar_acc (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List Coin)
    (hl : ¬ as.length < Bd.DE.V.k x) (h : ((Bd.IVStar.next x as cs).eval H).1 = .acc) :
    ∃ os last, Bd.final (as.getLastD []) = some (os, last) ∧ os.length + 1 = as.length ∧
      (∀ j < os.length, (os.getD j []).length = Bd.nAt x j) ∧
      ((allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2)).eval H).1 = true ∧
      (∀ e ∈ ((allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2)).eval H).2,
        e ∈ ((Bd.IVStar.next x as cs).eval H).2) ∧
      ((Bd.agrees x (Bd.opened os last) cs).eval H).1 = true ∧
      ∀ e ∈ ((Bd.agrees x (Bd.opened os last) cs).eval H).2, e ∈ ((Bd.IVStar.next x as cs).eval H).2 := by
  rcases hf : Bd.final (as.getLastD []) with _ | ⟨os, last⟩
  · exfalso
    have : Bd.IVStar.next x as cs = .ret .rej := by simp only [Bound.IVStar, hl, ↓reduceIte, hf]
    rw [this] at h
    simp [OC.eval] at h
  · by_cases hc : os.length + 1 = as.length ∧ ∀ j < os.length, (os.getD j []).length = Bd.nAt x j
    · have e : Bd.IVStar.next x as cs =
          bindOC (allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2)) fun ok =>
            if ok then bindOC (Bd.agrees x (Bd.opened os last) cs) fun acc => .ret (if acc then .acc else .rej)
            else .ret .rej := by
        simp only [Bound.IVStar, hl, ↓reduceIte, hf, ite_eq_left hc]
      rw [e] at h ⊢
      rw [eval_bindOC_full] at h ⊢
      rcases hok : ((allOC ((as.dropLast.zip os).map fun p => Bd.opensRound p.1 p.2)).eval H).1 with _ | _
      · rw [hok] at h
        simp [OC.eval] at h
      · rw [hok] at h ⊢
        rw [ite_eq_left rfl] at h ⊢
        rw [eval_bindOC_full] at h ⊢
        rcases hacc : ((Bd.agrees x (Bd.opened os last) cs).eval H).1 with _ | _
        · rw [hacc] at h
          simp [OC.eval] at h
        · exact ⟨os, last, rfl, hc.1, hc.2, hok, fun e he => List.mem_append_left _ he, hacc,
            fun e he => List.mem_append_right _ (List.mem_append_left _ he)⟩
    · exfalso
      have : Bd.IVStar.next x as cs = .ret .rej := by simp only [Bound.IVStar, hl, ↓reduceIte, hf, ite_eq_right hc]
      rw [this] at h
      simp [OC.eval] at h

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- **What `IVStar`'s decision on replayed messages checks when it accepts**, round by round. -/
theorem Bound.decAt_acc (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) (ms : List Msg)
    (r : Fin (Bd.IVStar.k x) → Coin) (hk : 0 < Bd.DE.V.k x) (h : (Bd.decAt H x ms r).1 = .acc) :
    ∃ os last, Bd.final (ms.getD (Bd.DE.V.k x - 1) []) = some (os, last) ∧ os.length + 1 = Bd.DE.V.k x ∧
      (∀ j < os.length, (os.getD j []).length = Bd.nAt x j ∧
        ((Bd.opensRound (ms.getD j []) (os.getD j [])).eval H).1 = true ∧
        ∀ e ∈ ((Bd.opensRound (ms.getD j []) (os.getD j [])).eval H).2, e ∈ (Bd.decAt H x ms r).2) ∧
      ((Bd.agrees x (Bd.opened os last) ((List.range (Bd.DE.V.k x)).map (coinSeq r))).eval H).1 = true ∧
      ∀ e ∈ ((Bd.agrees x (Bd.opened os last) ((List.range (Bd.DE.V.k x)).map (coinSeq r))).eval H).2,
        e ∈ (Bd.decAt H x ms r).2 := by
  obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
  have hl : ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []).getLastD [] =
      ms.getD (Bd.DE.V.k x - 1) [] := by
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.getLastD_concat, Nat.add_sub_cancel]
  have hd : ((List.range (Bd.DE.V.k x)).map fun i => ms.getD i []).dropLast =
      (List.range (Bd.DE.V.k x - 1)).map fun i => ms.getD i [] := by
    rw [hn, List.range_succ, List.map_append, List.map_singleton, List.dropLast_concat, Nat.add_sub_cancel]
  obtain ⟨os, last, hf, hlen, hn', hok, hokl, hag, hagl⟩ := Bd.ivstar_acc H x _ _ (by simp) h
  rw [hl] at hf
  rw [hd] at hok hokl
  simp only [List.length_map, List.length_range] at hlen
  refine ⟨os, last, hf, hlen, fun j hj => ⟨hn' j hj, ?_⟩, hag, hagl⟩
  have hmem : Bd.opensRound (ms.getD j []) (os.getD j []) ∈
      (((List.range (Bd.DE.V.k x - 1)).map fun i => ms.getD i []).zip os).map fun p => Bd.opensRound p.1 p.2 := by
    refine List.mem_map.2 ⟨(ms.getD j [], os.getD j []), List.mem_iff_getElem.2 ⟨j, by simp; omega, ?_⟩, rfl⟩
    simp [List.getElem?_eq_getElem hj]
  obtain ⟨h1, h2⟩ := allOC_true H _ hok _ hmem
  exact ⟨h1, fun e he => hokl e (h2 e he)⟩

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem Bound.coinSeq_iv (Bd : Bound Coin Col T) (x : Msg) (r : Fin (Bd.IVStar.k x) → Coin) :
    @coinSeq Coin _ (Bd.IVStar.k x) r = @coinSeq Coin _ (Bd.DE.V.k x) r := rfl

/-- `IVStar`'s answer at round `j` of a run it accepts at the coins `r`: the round's coin before the last, then
acceptance. -/
def Bound.ansAt (Bd : Bound Coin Col T) (x : Msg) (r : Fin (Bd.IVStar.k x) → Coin) (j : ℕ) : Out :=
  if j + 1 < Bd.DE.V.k x then .msg (Bd.ans (coinSeq r j)) else .acc

/-- **A run `IVStar` accepts**, at any prover: it has rounds; `final` reads its last message; each earlier round opens
to as many rows as the schedule fixes, its calls the run's; the element-level verifier answers the opened messages as
`IVStar` does, its calls the run's; and the run, opened, is the opened messages with `IVStar`'s answers, accepted. -/
theorem Bound.accFacts (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin (Bd.IVStar.k x) → Coin) (hacc : (run H P Bd.IVStar x ω r).out = .acc) :
    0 < Bd.DE.V.k x ∧ ∃ os last,
      Bd.final (((run H P Bd.IVStar x ω r).msgs.map Prod.fst).getD (Bd.DE.V.k x - 1) []) = some (os, last) ∧
      os.length + 1 = Bd.DE.V.k x ∧
      (∀ j < os.length, (os.getD j []).length = Bd.nAt x j ∧
        ((Bd.opensRound (((run H P Bd.IVStar x ω r).msgs.map Prod.fst).getD j []) (os.getD j [])).eval H).1 =
          true ∧
        ∀ e ∈ ((Bd.opensRound (((run H P Bd.IVStar x ω r).msgs.map Prod.fst).getD j []) (os.getD j [])).eval H).2,
          e ∈ (run H P Bd.IVStar x ω r).vLog) ∧
      (∀ j < Bd.DE.V.k x,
        ((Bd.DE.V.next x ((List.range (j + 1)).map fun i => (Bd.opened os last).getD i [])
          ((List.range (j + 1)).map (coinSeq r))).eval H).1 = Bd.ansAt x r j ∧
        ∀ e ∈ ((Bd.DE.V.next x ((List.range (j + 1)).map fun i => (Bd.opened os last).getD i [])
          ((List.range (j + 1)).map (coinSeq r))).eval H).2, e ∈ (run H P Bd.IVStar x ω r).vLog) ∧
      Bd.openView (run H P Bd.IVStar x ω r).seen =
        ⟨(List.range (Bd.DE.V.k x)).map fun j => ((Bd.opened os last).getD j [], Bd.ansAt x r j), .acc⟩ := by
  obtain ⟨hm, hv, ho⟩ := run_replay_same H P Bd.IVStar x ω r
  have hk : 0 < Bd.DE.V.k x := by
    by_contra h0
    rw [Bd.run_ivstar_zero H x _ r (by omega), hacc] at ho
    simp at ho
  refine ⟨hk, ?_⟩
  rw [Bd.run_ivstar H x _ r hk] at hm hv ho
  dsimp only at hm hv ho
  have hd : (Bd.decAt H x ((run H P Bd.IVStar x ω r).msgs.map Prod.fst) r).1 = .acc := ho.trans hacc
  obtain ⟨os, last, hf, hlen, hrounds, hag, hagl⟩ := Bd.decAt_acc H x _ r hk hd
  have hol : (Bd.opened os last).length = Bd.DE.V.k x := by simp [Bound.opened, hlen]
  refine ⟨os, last, hf, hlen, fun j hj => ?_, fun j hj => ?_, ?_⟩
  · obtain ⟨h1, h2, h3⟩ := hrounds j hj
    exact ⟨h1, h2, fun e he => hv ▸ h3 e he⟩
  · obtain ⟨h1, h2⟩ := Bd.agrees_true H x hag j (by omega)
    rw [take_eq_range_getD _ (by omega), take_range_map _ (by omega)] at h1 h2
    refine ⟨?_, fun e he => hv ▸ hagl e (h2 e he)⟩
    rw [h1, hol]
    unfold Bound.ansAt
    split_ifs with hj1
    · rw [getD_range_map _ _ hj]
    · rfl
  · have hlast : (run H P Bd.IVStar x ω r).msgs.getLastD ([], .rej) =
        (((run H P Bd.IVStar x ω r).msgs.map Prod.fst).getD (Bd.DE.V.k x - 1) [], .acc) := by
      have e := congrArg (fun l : List (Msg × Out) => l.getLastD ([], .rej)) hm
      simp only [List.getLastD_concat, hd] at e
      exact e.symm
    have hsnd : (run H P Bd.IVStar x ω r).msgs.map Prod.snd = (List.range (Bd.DE.V.k x)).map (Bd.ansAt x r) := by
      rw [← hm, hd]
      obtain ⟨n, hn⟩ : ∃ n, Bd.DE.V.k x = n + 1 := ⟨Bd.DE.V.k x - 1, by omega⟩
      have hr : List.range (Bd.DE.V.k x) = List.range (Bd.DE.V.k x - 1) ++ [Bd.DE.V.k x - 1] := by
        rw [hn, Nat.add_sub_cancel, List.range_succ]
      rw [hr, List.map_append, List.map_append, List.map_map, List.map_singleton, List.map_singleton]
      congr 1
      · refine List.map_congr_left fun i hi => ?_
        rw [List.mem_range] at hi
        simp only [Function.comp, Bound.ansAt, show i + 1 < Bd.DE.V.k x by omega, ↓reduceIte]
      · simp only [Bound.ansAt, show ¬ (Bd.DE.V.k x - 1 + 1 < Bd.DE.V.k x) by omega, ↓reduceIte]
    unfold Bound.openView
    simp only [Trace.seen]
    rw [hlast]
    simp only [hf, hsnd, zip_range_map _ _ hol, hacc]

/-! ## The plurality prover -/

section Plur

variable (Bd : Bound Coin Col T) (H : Msg → Msg) (x : Msg) {Ω A : Type} (P : Prover Ω)
  (keep : Ω → (Fin (Bd.IVStar.k x) → Coin) → A → Bool)

/-- A run against `IVStar`, as it may abort, accepted. -/
def Bound.gA (ω : Ω) (d : (Fin (Bd.IVStar.k x) → Coin) × A) : Prop :=
  (abortRun H P Bd.IVStar x keep ω d).out = .acc

/-- The element-level message a run against `IVStar`, as it may abort, opens at round `j`. -/
def Bound.fA (ω : Ω) (d : (Fin (Bd.IVStar.k x) → Coin) × A) (j : ℕ) : Msg :=
  ((Bd.openView (abortRun H P Bd.IVStar x keep ω d).seen).msgs.map Prod.fst).getD j []

/-- **The plurality prover's message at round `j`**, on the coins `l` so far: the element-level message that
`IVStar`'s accepted continuations of those coins open most often there (`plur`). -/
noncomputable def Bound.plurAt [Fintype A] (ω : Ω) (j : ℕ) (l : List Coin) : Msg :=
  plur (fun u : (Fin (Bd.IVStar.k x) → Coin) × A => Bd.gA H x P keep ω (fork j (preCoins l) u.1, u.2))
    (fun u => Bd.fA H x P keep ω (fork j (preCoins l) u.1, u.2) j)

/-- **The plurality prover against `DE`**: at the tape, it reads the coins so far off the answers (`decC`) and sends
`plurAt` there. -/
noncomputable def Bound.plurP [Fintype A] (decC : Msg → Msg → Coin) : Prover Ω :=
  ⟨fun _ hist ω => .ret (Bd.plurAt H x P keep ω hist.length (hist.map fun h => decC x h.2))⟩

/-- The run opens the plurality at every round. -/
def Bound.OnPlur [Fintype A] (ω : Ω) (d : (Fin (Bd.IVStar.k x) → Coin) × A) : Prop :=
  ∀ j < Bd.DE.V.k x, Bd.fA H x P keep ω d j = Bd.plurAt H x P keep ω j ((List.range j).map (coinSeq d.1))

/-- **A run the plurality prover's runs don't show**: accepted, and off the plurality at some round. -/
def Bound.BadA [Fintype A] (ω : Ω) (d : (Fin (Bd.IVStar.k x) → Coin) × A) : Prop :=
  Bd.gA H x P keep ω d ∧ ¬ Bd.OnPlur H x P keep ω d

open Classical in
/-- **The plurality prover's aborts**: a run is kept where `IVStar`'s run at its coins, as it may abort, accepts and
opens the plurality at every round. -/
noncomputable def Bound.plurKeep [Fintype A] (ω : Ω) (r : Fin (Bd.DE.V.k x) → Coin) (a : A) : Bool :=
  decide (Bd.gA H x P keep ω (r, a) ∧ Bd.OnPlur H x P keep ω (r, a))

variable {Bd H x P keep}

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
theorem Bound.gA_run {ω : Ω} {d : (Fin (Bd.IVStar.k x) → Coin) × A} (h : Bd.gA H x P keep ω d) :
    keep ω d.1 d.2 = true ∧ abortRun H P Bd.IVStar x keep ω d = run H P Bd.IVStar x ω d.1 := by
  unfold Bound.gA abortRun at h
  by_cases hk : keep ω d.1 d.2 = true
  · exact ⟨hk, abortRun_of_keep hk⟩
  · rw [ite_eq_right hk] at h
    simp [abortTrace] at h

/-- **On the plurality, the plurality prover's run is `P`'s run opened**: it sends at each round what `P`'s run opens
there, which `DE` answers as `IVStar` does, so it accepts, sees `P`'s run opened, and makes only calls `P`'s run makes. -/
theorem Bound.plurRun [Fintype A] (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c) (ω : Ω)
    (d : (Fin (Bd.DE.V.k x) → Coin) × A) (hg : Bd.gA H x P keep ω d) (hp : Bd.OnPlur H x P keep ω d) :
    (run H (Bd.plurP H x P keep decC) Bd.DE.V x ω d.1).seen = Bd.openView (abortRun H P Bd.IVStar x keep ω d).seen ∧
      (run H (Bd.plurP H x P keep decC) Bd.DE.V x ω d.1).out = .acc ∧
      ∀ e ∈ (run H (Bd.plurP H x P keep decC) Bd.DE.V x ω d.1).vLog, e ∈ (abortRun H P Bd.IVStar x keep ω d).vLog := by
  obtain ⟨-, hrun⟩ := Bound.gA_run hg
  have hacc : (run H P Bd.IVStar x ω d.1).out = .acc := hrun ▸ hg
  obtain ⟨hk, os, last, -, -, -, hag, hov⟩ := Bd.accFacts H x P ω d.1 hacc
  rw [Bd.coinSeq_iv x d.1] at hag
  have hp' : ∀ j < Bd.DE.V.k x,
      Bd.fA H x P keep ω d j = Bd.plurAt H x P keep ω j ((List.range j).map (coinSeq d.1)) := hp
  have hfA : ∀ j < Bd.DE.V.k x, Bd.fA H x P keep ω d j = (Bd.opened os last).getD j [] := by
    intro j hj
    unfold Bound.fA
    rw [hrun, hov, List.map_map, getD_range_map _ _ hj]
    rfl
  have hcongr : run H (Bd.plurP H x P keep decC) Bd.DE.V x ω d.1 =
      run H (replay (Bd.opened os last)) Bd.DE.V x () d.1 := by
    unfold run
    refine stepWith_congr H _ _ x ω () _ (fun M hist as => hist.length + M = Bd.DE.V.k x ∧
      as = (List.range hist.length).map (fun i => (Bd.opened os last).getD i []) ∧
      hist.map (fun h => decC x h.2) = (List.range hist.length).map (coinSeq d.1)) ?_ ?_ _ _ _ _
      ⟨by simp, by simp, by simp⟩
    · intro M hist as hG
      have hj : hist.length < Bd.DE.V.k x := by omega
      show (OC.ret (Bd.plurAt H x P keep ω hist.length (hist.map fun h => decC x h.2))).eval H =
        (OC.ret ((Bd.opened os last).getD hist.length [])).eval H
      rw [hG.2.2, ← hp' _ hj, hfA _ hj]
    · intro M hist as m hG hv
      have hj : hist.length < Bd.DE.V.k x := by omega
      have hpa : ((Bd.plurP H x P keep decC).next x hist ω).eval H = ((Bd.opened os last).getD hist.length [], []) := by
        have e : Bd.plurAt H x P keep ω hist.length (hist.map fun h => decC x h.2) =
            (Bd.opened os last).getD hist.length [] := by
          rw [hG.2.2, ← hp' _ hj, hfA _ hj]
        show (OC.ret (Bd.plurAt H x P keep ω hist.length (hist.map fun h => decC x h.2))).eval H = _
        simp only [e, OC.eval]
      have has : as ++ [(Bd.opened os last).getD hist.length []] =
          (List.range (hist.length + 1)).map fun i => (Bd.opened os last).getD i [] := by
        rw [hG.2.1, List.range_succ, List.map_append, List.map_singleton]
      rw [hpa] at hv ⊢
      dsimp only at hv ⊢
      simp only [has, List.length_map, List.length_range] at hv
      rw [(hag hist.length hj).1] at hv
      unfold Bound.ansAt at hv
      split_ifs at hv with hL
      injection hv with hm
      subst hm
      refine ⟨by simp only [List.length_append, List.length_singleton]; omega, by rw [has]; simp, ?_⟩
      simp only [List.map_append, hG.2.2, List.length_append, List.length_singleton, List.range_succ,
        List.map_singleton, hdec]
      rfl
  obtain ⟨h1, h2, h3⟩ := run_replay_known H Bd.DE.V x (Bd.opened os last) d.1 (Bd.ansAt x d.1) hk
    (fun j hj => (hag j hj).1) (fun j hj => ⟨_, by unfold Bound.ansAt; exact ite_eq_left hj⟩)
    (by simp only [Bound.ansAt, show ¬ (Bd.DE.V.k x - 1 + 1 < Bd.DE.V.k x) by omega, ↓reduceIte])
  rw [hcongr, hrun, hov]
  refine ⟨?_, h2, fun e he => ?_⟩
  · simp only [Trace.seen, h1, h2]
  · obtain ⟨j, hj, hej⟩ := h3 e he
    exact (hag j hj).2 e hej

end Plur

end Bound

end Claims.Recursion
