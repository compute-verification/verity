import Proofs.Flock.Claims.Complete.Honest

/-!
# The honest run against the plain verifier

The prover's side of `Claims.run` for a prover that replays a program (`feed`), and the plain verifier's side for a run
whose sends are all of tag `0`.

* **The program's run** (`PI.sim`): against `H`, send `i` answered by `resp i`; its sends and its result. `feed_sim`:
  fed the first `j` answers, `feed` returns the program's `j`-th message (its sends, then its result).
* **An accepted run** (`stepWith_acc`): a prover whose `j`-th message is `seq j` once it has seen the answers
  `resp 0, …, resp (j − 1)`, against a verifier that answers each of the first `N` messages with `resp j` and accepts the
  `(N + 1)`-th, is accepted within `N + 1` rounds.
* **The honest run** (`honestRun`): the honest program's sends and last message, each send answered by its coin's
  bytes. `run_acc_of`: when the statement sets up, the sends are all of tag `0`, fit the verifier's rounds, the first
  decodes as the link message and the last message as `Final`, and the shipped verdict and the custody checks pass on
  that last message, `honestP` is accepted.
-/

namespace FlockComplete

open Flock FlockDeploy

/-- **The program's run** against `H`, send `i` answered by `resp i`, from send `i`: its sends and its result. -/
def PI.sim {α : Type} (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) : PI α → ℕ → List Claims.Msg × α
  | .ret a, _ => ([], a)
  | .query q k, i => PI.sim H resp (k (H q)) i
  | .ask m k, i => (m :: (PI.sim H resp (k (resp i)) (i + 1)).1, (PI.sim H resp (k (resp i)) (i + 1)).2)

/-- **Fed the first `j` answers, the program sends its `j`-th message**: its sends, then its result. -/
theorem feed_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (p : PI Claims.Msg) :
    ∀ (i j : ℕ), j ≤ (p.sim H resp i).1.length →
      ((feed p ((List.range j).map fun k => resp (i + k))).eval H).1 =
        ((p.sim H resp i).1 ++ [(p.sim H resp i).2]).getD j [] := by
  induction p with
  | ret a =>
    intro i j hj
    simp only [PI.sim, List.length_nil, Nat.le_zero] at hj
    subst hj
    rfl
  | query q k ih =>
    intro i j hj
    exact ih (H q) i j hj
  | ask m k ih =>
    intro i j hj
    cases j with
    | zero => rfl
    | succ j =>
      simp only [PI.sim, List.length_cons, Nat.add_le_add_iff_right] at hj
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      have e : ((fun k => resp (i + k)) ∘ Nat.succ) = fun k => resp (i + 1 + k) := by
        funext k
        simp only [Function.comp, Nat.succ_eq_add_one]
        congr 1
        omega
      rw [e]
      simp only [feed, PI.sim, List.cons_append, List.getD_cons_succ]
      exact ih (resp i) (i + 1) j hj

/-- **An accepted run**: the prover's `j`-th message is `seq j` once it has seen `resp 0, …, resp (j − 1)`; the verifier
answers each of the first `N` messages with `resp j` and accepts the next; with `N + 1` rounds, the run accepts. -/
theorem stepWith_acc {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) (seq resp : ℕ → Claims.Msg) (N : ℕ)
    (hP : ∀ j ≤ N, ∀ hist : List (Claims.Msg × Claims.Msg), hist.map Prod.snd = (List.range j).map resp →
      ((P.next x hist ω).eval H).1 = seq j)
    (hV : ∀ j < N, ((vnext ((List.range (j + 1)).map seq)).eval H).1 = .msg (resp j))
    (hA : ((vnext ((List.range (N + 1)).map seq)).eval H).1 = .acc) (n : ℕ) (hn : N + 1 ≤ n) (t : Claims.Trace) :
    (Claims.stepWith H P x ω vnext n [] [] t).out = .acc := by
  suffices h : ∀ d j, j + d = N → ∀ n, d + 1 ≤ n → ∀ hist t, hist.map Prod.snd = (List.range j).map resp →
      (Claims.stepWith H P x ω vnext n hist ((List.range j).map seq) t).out = .acc from
    h N 0 (by omega) n hn [] t rfl
  intro d
  induction d with
  | zero =>
    intro j hj n hn hist t hh
    obtain ⟨n, rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
    have e : (List.range j).map seq ++ [seq j] = (List.range (j + 1)).map seq := by
      simp [List.range_succ]
    have hA' := hA
    rw [← hj, Nat.add_zero] at hA'
    simp only [Claims.stepWith, hP j (by omega) hist hh, e, hA']
  | succ d ih =>
    intro j hj n hn hist t hh
    obtain ⟨n, rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
    have e : (List.range j).map seq ++ [seq j] = (List.range (j + 1)).map seq := by
      simp [List.range_succ]
    have hh' : (hist ++ [(seq j, resp j)]).map Prod.snd = (List.range (j + 1)).map resp := by
      simp [List.range_succ, hh]
    simp only [Claims.stepWith, hP j (by omega) hist hh, e, hV j (by omega)]
    exact ih (j + 1) (by omega) n (by omega) _ _ hh'

/-- A last message of tag `0` is answered by its coin. -/
theorem act_coin {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} {t : Claims.Msg} {c : Coin} (hs : setupOf false x = some (s, spec, sts))
    (ha : as.getLast? = some (0 :: t)) (hc : cs.getLast? = some c) : act false x as cs = .coin c := by
  unfold act
  rw [hs, ha, hc]
  rfl

/-- The last message is decided: after the link message, on the link, its coin, the round messages with theirs. -/
theorem act_final {x : Claims.Msg} {a₁ : Claims.Msg} {rest : List Claims.Msg} {c₁ : Coin} {crest : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt} {f : Claims.Msg} {F : Final}
    {link : List ByteArray × List ByteArray} (hs : setupOf false x = some (s, spec, sts))
    (hl : (a₁ :: rest).getLast? = some f) (hf : finalOf f = some F) (hlink : linkOf a₁ = some link) :
    act false x (a₁ :: rest) (c₁ :: crest) =
      .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) F := by
  obtain ⟨b, rfl⟩ : ∃ b, f = 1 :: b := by
    unfold finalOf at hf
    split at hf
    · exact ⟨_, rfl⟩
    · cases hf
  unfold act
  rw [hs]
  dsimp only
  rw [hl]
  split
  · rename_i h _
    simp only [Option.some.injEq, List.cons.injEq] at h
    exact absurd h.1 (by decide)
  · rename_i h _
    simp only [Option.some.injEq] at h
    subst h
    rw [hf]
    dsimp only
    rw [hlink]
  · rename_i h
    cases h

/-- **The honest run** at the coins `r`: the honest program's sends, each answered by its coin's bytes, and its last
message. -/
def honestRun (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : Tape) {k : ℕ}
    (r : Fin k → Coin) : List Claims.Msg × Claims.Msg :=
  (prog x w ω).sim H (fun j => (Claims.coinSeq r j).bytes) 0

/-- The round messages and their coins, as the verifier's decision reads them (`act`): every message after the first
but the last, its tag dropped, with its coin. -/
def midsOf : List Claims.Msg → List Coin → List (Claims.Msg × Coin)
  | _ :: rest, _ :: crest => (rest.dropLast.map List.tail).zip crest.dropLast
  | _, _ => []

theorem map_getD_range (L : List Claims.Msg) :
    (List.range L.length).map (fun j => L.getD j []) = L := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_range, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2,
    Option.getD_some]

/-- **The honest prover is accepted** at the coins `r` when the statement sets up, the honest run's sends are of tag `0`,
fit the verifier's rounds, and start with a link message, its last message decodes, and the shipped verdict and the
checks pass on it. -/
theorem run_acc_of (H : Claims.Msg → Claims.Msg) (w : Claims.Flock.Wit) (x : Claims.Msg) (ω : Tape)
    (r : Fin ((V false).k x) → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    {link : List ByteArray × List ByteArray} {F : Final} (hs : setupOf false x = some (s, spec, sts))
    (hk : (honestRun H x w ω r).1.length + 1 ≤ (V false).k x)
    (htag : ∀ m ∈ (honestRun H x w ω r).1, ∃ t, m = 0 :: t)
    (hne : (honestRun H x w ω r).1 ≠ [])
    (hlink : linkOf ((honestRun H x w ω r).1.headD []) = some link)
    (hfin : finalOf (honestRun H x w ω r).2 = some F)
    (hv : verdictWith false (Claims.hOf H) spec (tabsOf sts) F.record F.proofs = .ok ())
    (hc : checks false s spec x link (Claims.coinSeq r 0)
      (midsOf ((honestRun H x w ω r).1 ++ [(honestRun H x w ω r).2])
        ((List.range ((honestRun H x w ω r).1.length + 1)).map (Claims.coinSeq r))) F = .ok ()) :
    (Claims.run H (honestP w) (V false) x ω r).out = .acc := by
  set ms := (honestRun H x w ω r).1 with hms
  set f := (honestRun H x w ω r).2 with hf
  let seq : ℕ → Claims.Msg := fun j => (ms ++ [f]).getD j []
  let resp : ℕ → Claims.Msg := fun j => (Claims.coinSeq r j).bytes
  have hseq : ∀ j (hj : j < ms.length), seq j = ms[j] := by
    intro j hj
    simp only [seq, List.getD_eq_getElem?_getD, List.getElem?_append_left hj, List.getElem?_eq_getElem hj,
      Option.getD_some]
  have hseqN : seq ms.length = f := by
    simp [seq, List.getD_eq_getElem?_getD]
  have hall : (List.range (ms.length + 1)).map seq = ms ++ [f] := by
    have := map_getD_range (ms ++ [f])
    rw [List.length_append, List.length_singleton] at this
    exact this
  unfold Claims.run
  refine stepWith_acc H (honestP w) x ω _ seq resp ms.length ?_ ?_ ?_ _ hk _
  · intro j hj hist hh
    show ((feed (prog x w ω) (hist.map Prod.snd)).eval H).1 = _
    have := feed_sim H resp (prog x w ω) 0 j hj
    simp only [Nat.zero_add] at this
    rw [hh]
    exact this
  · intro j hj
    obtain ⟨t, ht⟩ := htag ms[j] (List.getElem_mem hj)
    have ha : ((List.range (j + 1)).map seq).getLast? = some (0 :: t) := by
      simp only [List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel,
        Option.map_some]
      rw [hseq j hj, ht]
    have hc' : ((List.range ((List.range (j + 1)).map seq).length).map (Claims.coinSeq r)).getLast? =
        some (Claims.coinSeq r j) := by
      simp only [List.length_map, List.length_range, List.getLast?_map, List.getLast?_range, Nat.add_one_ne_zero,
        ite_false, Nat.add_sub_cancel, Option.map_some]
    exact (next_coin H (act_coin hs ha hc')).symm ▸ rfl
  · show ((next false x ((List.range (ms.length + 1)).map seq)
      ((List.range ((List.range (ms.length + 1)).map seq).length).map (Claims.coinSeq r))).eval H).1 = .acc
    rw [hall, List.length_append, List.length_singleton]
    obtain ⟨a₁, rest, hms1⟩ : ∃ a₁ rest, ms = a₁ :: rest := List.exists_cons_of_ne_nil hne
    have hl : (ms ++ [f]).getLast? = some f := by simp
    have hcs : (List.range (ms.length + 1)).map (Claims.coinSeq r) =
        Claims.coinSeq r 0 :: (List.range ms.length).map (fun j => Claims.coinSeq r (j + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      rfl
    have hlink' : linkOf a₁ = some link := by rw [hms1] at hlink; exact hlink
    rw [hcs]
    rw [hms1, List.cons_append] at hl ⊢
    have had := act_final (c₁ := Claims.coinSeq r 0) (crest := (List.range (a₁ :: rest).length).map
      (fun j => Claims.coinSeq r (j + 1))) hs hl hfin hlink'
    rw [next_decide H had]
    refine ⟨hv, ?_⟩
    rw [hcs, hms1, List.cons_append] at hc
    exact hc

end FlockComplete
