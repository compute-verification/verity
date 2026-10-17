import Proofs.Flock.Claims.Recursion.InnerAccepts

/-!
# `VStarAccepts` at the deployed readers, reduced to V*'s parts

`VStarAccepts shape` (`InnerAccepts.lean`) is a fact about the readers the compiled verifier's inner view reads its
messages through and about V*'s circuits. This file defines the deployed readers and reduces `VStarAccepts` at each
shape to named facts about one thing each: one V* part's circuit, the V* statements the compiled verifier stages, or
the inner verifier.

* **The deployed readers** (`Layout.msgs`, `Layout.calls`): the outer witnesses are V*'s parts' hidden values, one
  witness per part, in order. V*'s circuits are fixed by the compiled statement (its inner shape and the verifier's
  bindings), so the statement fixes where each part holds what the readers read (`Layout`): inner round `i`'s message
  as part `s` holds it (`Layout.held`), and each SHA-512 evaluation part `s` computes. The inner view's messages are
  each round's message off the first part that holds it (`Layout.msgs`); the recorded calls are each part's
  evaluations (`Layout.calls`). These have the types of `deployRec`'s `Readers` fields (`Recursion/Deploy.lean`), so
  `deployRec` at `⟨L.msgs, L.calls⟩` reads through them (`ReadsThrough`) by its definition.
* **The named facts**, at an inner check `chk s` per part, read at the coins a V* statement shows (`shows`) on a list
  of inner messages:
  * `PartChecks s`: part `s`'s circuit accepts only if its inner check holds on the messages it holds, or two of its
    recorded calls collide;
  * `PartOpens s`: each message part `s` holds opens the commit string its statement shows for that round, through
    calls it records (`Opening`);
  * `Staged`: on an accepted run, each V* statement shows the run's first coins and, as the commit string of each
    inner round the layout spans, the run's message at that round (only those: a V* statement is read off the run's
    first `k_I + kS` messages, so it can't show the outer sessions');
  * `Decomp`: the parts' inner checks together are the inner verifier, at the shape.
* `vstarAccepts_of_parts`: with `ReadsThrough`, these give `VStarAccepts shape`. Two parts holding different messages
  at one round open one commit string two ways, which is a collision among their calls (`Opening.bind`); otherwise
  every part holds what the readers read (`agree_or_coll`), so every part's inner check holds on the readers'
  messages, and `Decomp` gives an accepted inner view. `vstarAccepts_dense`, `vstarAccepts_holo` and
  `vstarAccepts_sparse` are it at the three shapes, and `innerAcceptsColl_of_parts` gives `InnerAcceptsColl` from
  them.
* `Decomp` is the inner verifier's checks together (`InnerConj`, a fact about the inner verifier alone) with parts
  that cover them (`Covers`, a fact about how the compiled statement cuts it into parts): `decomp_of_covers`.
* Reading through readers also gives three of the other code facts the recursion's KS reads: `InnerViewRun` at
  `I.V` (`innerViewRun_of_readsThrough`: the inner view is `I.V`'s run of the readers' messages, and replaying a
  run's own messages is the run, `run_replay_self`), `CallsNil` (`callsNil_of_readsThrough`), and `CallsRecord`'s
  second half, every recorded call a true evaluation (`CallsTrue`), from each part's SHA-512 gadgets
  (`PartCallsTrue`, `callsTrue_of_parts`; `callsRecord_of_parts` takes the first half as a hypothesis).
* **Openings at the played rounds.** `CallsOpen` asks for an opening of the inner view's message at every inner round,
  `[]` at a round the inner view doesn't play, which no part opens. The deployed Flock verifier decides at the prover's
  tag-1 message and accepts only at its last round (`AccLast`), so the inner view stops early only where it rejects,
  which `VStarAccepts` allows only at a collision among the recorded calls; `CallsOpen` must hold there too.
  `CallsOpenPlayed` asks for it only at the rounds the inner view plays, and V*'s parts give it
  (`callsOpenPlayed_of_parts`): there the inner view's message is the first holder's, which that holder opens, given
  that some part holds each inner round (`Held`). It serves the consumer as `CallsOpen` does
  (`forkColl_of_openingPlayed`, `sum_opensDiffer_le_played`): two inner views that differ at round `j` first differ
  at a round `i ≤ j` that both play, since two runs that agree on their messages and coins before a round play it
  alike (`run_replay_plays`).
* **The inner verifier's calls, off a collision.** `CallsRecord`'s first half asks that every call the inner view's run
  makes be recorded. Where two parts hold different messages at a round, the run's calls on the readers' message
  aren't the other part's, so V*'s parts give it only off a collision (`CallsLogColl`, `callsLogColl_of_parts`, from
  the inner verifier's calls split among the parts' checks, `LogDecomp`, and each part recording its check's calls,
  `PartLogs`). It serves its consumer as the first half does (`PrBAbort_le_PrBI_coll`): a collision among the recorded
  calls at either draw is one `B_I` finds.
* `codeFacts_of_parts`: all of it at once. With `ReadsThrough`, `PartCallsTrue` at every part and V*'s parts at each
  shape (`VStarParts`: the part facts above at one layout and opening), the readers give `InnerViewRun`, `CallsNil`,
  `CallsTrue`, `CallsOpenPlayed`, `CallsLogColl` and `InnerAcceptsColl`.
* **The staging, off the run.** A run's V* statements are the staging `D.vstar`'s at its first `k_I + kS` pairs
  (`runVstar`), so `Staged` follows (`staged_of_vstarShows`) from a fact about the staging alone, with no prover or run
  (`VstarShows`: a V* statement staged at pairs shows the coins their verifier messages show and their prover messages
  as the commit strings), and one about the compiled verifier's accepted runs (`RunPairs`: they send at least
  `k_I + kS` messages, each answered by a message that shows its coin). `codeFacts_of_staging` is `codeFacts_of_parts`
  so (`VStarPartsStaging`).
* `Opening.twoCalls`: an opening of a commit string `b ‖ c` through two recorded calls, its salt's hash `c` and its
  message's digest under the salt's mask, as hm96 rows open (`Deploy.zeroCommit`'s form). It binds wherever the salt's
  and the message's encodings are injective.
-/

namespace Claims.Recursion.VStar

open Claims

/-! ## The deployed readers -/

/-- A byte from up to eight bits, the low bit first. -/
def byte (bs : List Bool) : UInt8 := bs.foldr (fun b a => (if b then 1 else 0) + 2 * a) 0

/-- **Bits as bytes**: eight to a byte, the low bit first, a short last byte padded with zeros. Any other bit order is
this one at permuted positions, which a layout lists. -/
def bytesOf (bs : List Bool) : Msg := (List.range ((bs.length + 7) / 8)).map fun j => byte ((bs.drop (8 * j)).take 8)

/-- The bytes a witness's hidden values hold at the positions `ps`, in order. -/
def readAt (w : Flock.Wit) (ps : List ℕ) : Msg := bytesOf (ps.map fun p => w.hid.getD p false)

/-- **Where V\*'s parts hold what the readers read**, fixed by the compiled statement `x`: the inner rounds' count
(`rounds x`), the hidden values at which part `s` holds inner round `i`'s message, its bits in order (`msgAt x s i`,
`none` where the part holds none), and those of each SHA-512 evaluation part `s` computes, its input's bits and its
output's (`shaAt x s`). -/
structure Layout where
  rounds : Msg → ℕ
  msgAt : Msg → ℕ → ℕ → Option (List ℕ)
  shaAt : Msg → ℕ → List (List ℕ × List ℕ)

/-- Inner round `i`'s message as part `s` holds it, off its witness `w`. -/
def Layout.held (L : Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) (i : ℕ) : Option Msg := (L.msgAt x s i).map (readAt w)

/-- Inner round `i`'s message off the witnesses: as the first part that holds it holds it, empty where none does. -/
def Layout.msg (L : Layout) (x : Msg) (ws : List Flock.Wit) (i : ℕ) : Msg :=
  ((List.range ws.length).findSome? fun s => ws[s]?.bind fun w => L.held x s w i).getD []

/-- **The deployed readers' inner messages**: the inner session's `rounds x` messages, each off the first part that
holds it. -/
def Layout.msgs (L : Layout) (x : Msg) (ws : List Flock.Wit) : List Msg := (List.range (L.rounds x)).map (L.msg x ws)

/-- **The deployed readers' calls**: part `s`'s SHA-512 evaluations off its witness `w`, each its input and its output
read at their hidden values. -/
def Layout.calls (L : Layout) (x : Msg) (s : ℕ) (w : Flock.Wit) : List (Msg × Msg) :=
  (L.shaAt x s).map fun p => (readAt w p.1, readAt w p.2)

/-- A message one part holds is the readers' message at that round, when every part that holds the round holds the
same message. -/
theorem Layout.msgs_getD {L : Layout} {x : Msg} {ws : List Flock.Wit} {i : ℕ} (hi : i < L.rounds x) {s : ℕ}
    (hs : s < ws.length) {m : Msg} (hm : L.held x s ws[s] i = some m)
    (hall : ∀ s' (hs' : s' < ws.length) m', L.held x s' ws[s'] i = some m' → m' = m) :
    (L.msgs x ws).getD i [] = m := by
  have hget : (L.msgs x ws).getD i [] = L.msg x ws i := by
    simp [Layout.msgs, List.getD_eq_getElem?_getD, hi]
  rw [hget, Layout.msg]
  cases hf : (List.range ws.length).findSome? (fun s => ws[s]?.bind fun w => L.held x s w i) with
  | none =>
    have := List.findSome?_eq_none_iff.1 hf s (List.mem_range.2 hs)
    simp [List.getElem?_eq_getElem hs, hm] at this
  | some m₀ =>
    obtain ⟨s₀, hs₀, h₀⟩ := List.exists_of_findSome?_eq_some hf
    have hs₀' := List.mem_range.1 hs₀
    simp only [List.getElem?_eq_getElem hs₀', Option.bind] at h₀
    exact hall s₀ hs₀' m₀ h₀

/-- At a round some part holds, the readers' message is a message a part holds. -/
theorem Layout.msgs_getD_held {L : Layout} {x : Msg} {ws : List Flock.Wit} {i : ℕ} (hi : i < L.rounds x) {s : ℕ}
    (hs : s < ws.length) (hh : (L.msgAt x s i).isSome) :
    ∃ s₀, ∃ hs₀ : s₀ < ws.length, L.held x s₀ ws[s₀] i = some ((L.msgs x ws).getD i []) := by
  have hget : (L.msgs x ws).getD i [] = L.msg x ws i := by
    simp [Layout.msgs, List.getD_eq_getElem?_getD, hi]
  rw [hget, Layout.msg]
  cases hf : (List.range ws.length).findSome? (fun s => ws[s]?.bind fun w => L.held x s w i) with
  | none =>
    have := List.findSome?_eq_none_iff.1 hf s (List.mem_range.2 hs)
    obtain ⟨ps, hps⟩ := Option.isSome_iff_exists.1 hh
    simp [List.getElem?_eq_getElem hs, Layout.held, hps] at this
  | some m₀ =>
    obtain ⟨s₀, hs₀, h₀⟩ := List.exists_of_findSome?_eq_some hf
    have hs₀' := List.mem_range.1 hs₀
    simp only [List.getElem?_eq_getElem hs₀', Option.bind] at h₀
    exact ⟨s₀, hs₀', by simpa using h₀⟩

/-! ## Replaying a run's own messages -/

/-- A round the verifier answers with a message: the rounds go on from the history and the trace it extends. -/
theorem stepWith_msg {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) (n : ℕ)
    (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) {m : Msg}
    (hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 = .msg m) :
    stepWith H P x ω vn (n + 1) hist as t =
      stepWith H P x ω vn n (hist ++ [(((P.next x hist ω).eval H).1, m)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with
          msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1)],
          pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
          vLog := t.vLog ++ ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).2 } := by
  simp only [stepWith]
  rw [hv]

/-- Two replayed message lists that agree at every round the run plays give one run. -/
theorem stepWith_replay_congr (H : Msg → Msg) (x : Msg) (vn : List Msg → OC Out) (ms : List Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (ms' : List Msg), hist.length = t.msgs.length →
      (∀ j, t.msgs.length ≤ j → j < (stepWith H (replay ms) x () vn n hist as t).msgs.length →
        ms'.getD j [] = ms.getD j []) →
      stepWith H (replay ms') x () vn n hist as t = stepWith H (replay ms) x () vn n hist as t
  | 0, _, _, _, _, _, _ => rfl
  | n + 1, hist, as, t, ms', hl, h => by
    have hlt : t.msgs.length < (stepWith H (replay ms) x () vn (n + 1) hist as t).msgs.length := by
      have := congrArg List.length (stepWith_succ_fst_take H (replay ms) x () vn n hist as t)
      simp only [List.length_take, List.length_map, List.length_append, List.length_singleton] at this
      omega
    have hm : ms'.getD hist.length [] = ms.getD hist.length [] := h _ hl.ge (hl ▸ hlt)
    have e1 : ((replay ms').next x hist ()).eval H = ((replay ms).next x hist ()).eval H := by
      simp only [replay, OC.eval, hm]
    simp only [stepWith]
    rw [e1]
    split
    · next m hv =>
      rw [stepWith_msg H _ x () vn n hist as t hv] at h
      exact stepWith_replay_congr H x vn ms n _ _ _ ms' (by simp [hl])
        fun j hj1 hj2 => h j (by simp only [List.length_append, List.length_singleton] at hj1; omega) hj2
    · rfl
    · rfl

/-- A replayed run's message at each round it plays is the replayed list's there. -/
theorem stepWith_replay_fst (H : Msg → Msg) (x : Msg) (vn : List Msg → OC Out) (ms : List Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), hist.length = t.msgs.length →
      ∀ j, t.msgs.length ≤ j → j < (stepWith H (replay ms) x () vn n hist as t).msgs.length →
        ((stepWith H (replay ms) x () vn n hist as t).msgs.map Prod.fst).getD j [] = ms.getD j []
  | 0, _, _, t, _, j, hj1, hj => by simp only [stepWith] at hj; omega
  | n + 1, hist, as, t, hl, j, hj1, hj => by
    rcases Nat.lt_or_ge j (t.msgs.length + 1) with hlt | hge
    · obtain rfl : j = t.msgs.length := by omega
      have h2 : ((stepWith H (replay ms) x () vn (n + 1) hist as t).msgs.map Prod.fst)[t.msgs.length]? =
          some (((replay ms).next x hist ()).eval H).1 := by
        have := congrArg (·[t.msgs.length]?) (stepWith_succ_fst_take H (replay ms) x () vn n hist as t)
        simpa [List.getElem?_take] using this
      rw [List.getD_eq_getElem?_getD, h2]
      simp [replay, OC.eval, hl]
    · rcases hv : ((vn (as ++ [(((replay ms).next x hist ()).eval H).1])).eval H).1 with m | _ | _
      · rw [stepWith_msg H _ x () vn n hist as t hv] at hj ⊢
        exact stepWith_replay_fst H x vn ms n _ _ _ (by simp [hl]) j (by simpa using hge) hj
      · simp only [stepWith, hv, List.length_append, List.length_singleton] at hj
        omega
      · simp only [stepWith, hv, List.length_append, List.length_singleton] at hj
        omega

/-- **Replaying a run's own messages is the run.** -/
theorem run_replay_self {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (ms : List Msg)
    (r : Fin (V.k x) → Coin) :
    run H (replay ((run H (replay ms) V x () r).msgs.map Prod.fst)) V x () r = run H (replay ms) V x () r := by
  exact stepWith_replay_congr H x _ ms (V.k x) [] [] _ _ rfl fun j _ hj =>
    stepWith_replay_fst H x _ ms (V.k x) [] [] _ rfl j (Nat.zero_le _) hj

/-! ## Two runs play a round alike when they agree before it -/

/-- A run plays at most its rounds. -/
theorem stepWith_length_le {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω vn n hist as t).msgs.length ≤ t.msgs.length + n
  | 0, _, _, _ => le_refl _
  | n + 1, hist, as, t => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv]
      refine (stepWith_length_le H P x ω vn n _ _ _).trans ?_
      simp only [List.length_append, List.length_singleton]
      omega
    · simp [stepWith, hv]
    · simp [stepWith, hv]

/-- A run with a round left plays one. -/
theorem stepWith_length_gt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)
    {n : ℕ} (hn : 0 < n) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) :
    t.msgs.length < (stepWith H P x ω vn n hist as t).msgs.length := by
  obtain ⟨n, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have := congrArg List.length (stepWith_succ_fst_take H P x ω vn n hist as t)
  simp only [List.length_take, List.length_map, List.length_append, List.length_singleton] at this
  omega

/-- A round before a run's last is answered with a message. -/
theorem stepWith_msg_of_lt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (i : ℕ), t.msgs.length ≤ i →
      i + 1 < (stepWith H P x ω vn n hist as t).msgs.length →
        ∃ m, ((stepWith H P x ω vn n hist as t).msgs[i]?).map Prod.snd = some (.msg m)
  | 0, _, _, t, i, hi, h => by simp only [stepWith] at h; omega
  | n + 1, hist, as, t, i, hi, h => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      rcases Nat.lt_or_ge i (t.msgs.length + 1) with hlt | hge
      · obtain rfl : i = t.msgs.length := by omega
        obtain ⟨l, hl⟩ := stepWith_prefix H P x ω vn n _ _ _
        refine ⟨m, ?_⟩
        rw [← hl]
        simp [hv]
      · exact stepWith_msg_of_lt H P x ω vn n _ _ _ i (by simpa using hge) h
    · simp only [stepWith, hv, List.length_append, List.length_singleton] at h
      omega
    · simp only [stepWith, hv, List.length_append, List.length_singleton] at h
      omega

/-- A run that stops before its rounds run out is decided at its last round. -/
theorem stepWith_last_of_lt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω vn n hist as t).msgs.length < t.msgs.length + n →
        ∀ m, ((stepWith H P x ω vn n hist as t).msgs.getLast?).map Prod.snd ≠ some (.msg m)
  | 0, _, _, t, h => by simp only [stepWith] at h; omega
  | n + 1, hist, as, t, h => by
    rcases hv : ((vn (as ++ [((P.next x hist ω).eval H).1])).eval H).1 with m | _ | _
    · rw [stepWith_msg H P x ω vn n hist as t hv] at h ⊢
      exact stepWith_last_of_lt H P x ω vn n _ _ _ (by simp only [List.length_append, List.length_singleton]; omega)
    · simp [stepWith, hv]
    · simp [stepWith, hv]

/-- **Two replayed runs agree on their first `N` rounds** where their messages agree before round `N` and their
verifiers answer alike on up to `N` messages. -/
theorem stepWith_replay_take (H : Msg → Msg) (x : Msg) (vn vn' : List Msg → OC Out) (ms ms' : List Msg) (N : ℕ) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), hist.length = t.msgs.length →
      as.length = t.msgs.length → (∀ i, t.msgs.length ≤ i → i < N → ms.getD i [] = ms'.getD i []) →
      (∀ bs : List Msg, bs.length ≤ N → vn bs = vn' bs) →
      (stepWith H (replay ms) x () vn n hist as t).msgs.take N =
        (stepWith H (replay ms') x () vn' n hist as t).msgs.take N
  | 0, _, _, _, _, _, _, _ => rfl
  | n + 1, hist, as, t, hl, ha, hm, hv => by
    rcases Nat.lt_or_ge t.msgs.length N with hN | hN
    · have e1 : ((replay ms').next x hist ()).eval H = ((replay ms).next x hist ()).eval H := by
        simp only [replay, OC.eval, hm hist.length hl.ge (hl ▸ hN)]
      have e2 : vn' (as ++ [(((replay ms).next x hist ()).eval H).1]) =
          vn (as ++ [(((replay ms).next x hist ()).eval H).1]) :=
        (hv _ (by simp only [List.length_append, List.length_singleton]; omega)).symm
      simp only [stepWith]
      rw [e1, e2]
      split
      · exact stepWith_replay_take H x vn vn' ms ms' N n _ _ _ (by simp [hl]) (by simp [ha])
          (fun i hi1 hi2 => hm i (by simp only [List.length_append, List.length_singleton] at hi1; omega) hi2) hv
      · rfl
      · rfl
    · have key : ∀ (P : Prover Unit) (v : List Msg → OC Out),
          (stepWith H P x () v (n + 1) hist as t).msgs.take N = t.msgs.take N := fun P v => by
        obtain ⟨l, hl'⟩ := stepWith_prefix H P x () v (n + 1) hist as t
        rw [← hl', List.take_append_of_le_length hN]
      rw [key, key]

/-- **Two replayed runs agree on their first `N` rounds** where their messages and their coins agree before `N`. -/
theorem run_replay_take {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (ms ms' : List Msg) (c c' : Fin (V.k x) → Coin) (N : ℕ) (hm : ∀ i < N, ms.getD i [] = ms'.getD i [])
    (hc : ∀ i < N, coinSeq c i = coinSeq c' i) :
    (run H (replay ms) V x () c).msgs.take N = (run H (replay ms') V x () c').msgs.take N :=
  stepWith_replay_take H x _ _ ms ms' N (V.k x) [] [] _ rfl rfl (fun i _ hi => hm i hi) fun bs hb => by
    congr 1
    exact List.map_congr_left fun i hi => hc i (by rw [List.mem_range] at hi; omega)

/-- **Two replayed runs that agree before round `j` play round `j` alike**: a run plays round `j` unless it decided
before it or ran out of rounds, and both are read off the rounds before `j`. -/
theorem run_replay_plays {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (ms ms' : List Msg) (c c' : Fin (V.k x) → Coin) (j : ℕ) (hm : ∀ i < j, ms.getD i [] = ms'.getD i [])
    (hc : ∀ i < j, coinSeq c i = coinSeq c' i) (hj : j < (run H (replay ms) V x () c).msgs.length) :
    j < (run H (replay ms') V x () c').msgs.length := by
  by_contra hn
  push Not at hn
  have ht := run_replay_take H V x ms ms' c c' j hm hc
  rw [List.take_of_length_le hn] at ht
  have hlen : (run H (replay ms') V x () c').msgs.length = j := by
    rw [← ht, List.length_take]
    omega
  have hk : j < V.k x := by
    have := stepWith_length_le H (replay ms) x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c))) (V.k x) [] [] ⟨[], [], [], .rej⟩
    simp only [List.length_nil, Nat.zero_add] at this
    exact lt_of_lt_of_le hj this
  have hlast := stepWith_last_of_lt H (replay ms') x ()
    (fun as => V.next x as ((List.range as.length).map (coinSeq c'))) (V.k x) [] [] ⟨[], [], [], .rej⟩
    (by simp only [List.length_nil, Nat.zero_add]; exact hlen ▸ hk)
  rcases j with _ | j
  · have := stepWith_length_gt H (replay ms') x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c'))) hk [] [] ⟨[], [], [], .rej⟩
    simp only [List.length_nil] at this
    unfold run at hlen
    omega
  · obtain ⟨m, hm'⟩ := stepWith_msg_of_lt H (replay ms) x ()
      (fun as => V.next x as ((List.range as.length).map (coinSeq c))) (V.k x) [] [] ⟨[], [], [], .rej⟩ j
      (Nat.zero_le _) hj
    have hg : (run H (replay ms') V x () c').msgs.getLast? = (run H (replay ms) V x () c).msgs[j]? := by
      rw [List.getLast?_eq_getElem?, hlen, Nat.add_sub_cancel, ← ht, List.getElem?_take]
      simp
    exact hlast m (by unfold run at hg; rw [hg]; exact hm')

/-- A run plays at most its verifier's rounds. -/
theorem run_length_le {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (r : Fin (V.k x) → Coin) : (run H P V x ω r).msgs.length ≤ V.k x := by
  have := stepWith_length_le H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) (V.k x) [] []
    ⟨[], [], [], .rej⟩
  simp only [List.length_nil, Nat.zero_add] at this
  exact this

/-- A replayed run's message at each round it plays is the replayed list's there. -/
theorem run_replay_fst {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (ms : List Msg)
    (r : Fin (V.k x) → Coin) {j : ℕ} (hj : j < (run H (replay ms) V x () r).msgs.length) :
    ((run H (replay ms) V x () r).msgs.map Prod.fst).getD j [] = ms.getD j [] :=
  stepWith_replay_fst H x _ ms (V.k x) [] [] _ rfl j (Nat.zero_le _) hj

/-! ## The compiled verifier reads through the readers -/

section Facts

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`D` reads through the readers `msgs`, `calls`**: its inner view is `IV`'s run at `H` of the messages
`msgs x ws` at the run's first coins, and its recorded calls are each session's witness's, in order. `deployRec`
(`Recursion/Deploy.lean`) at the readers `⟨msgs, calls⟩` is defined so. -/
def ReadsThrough (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (msgs : Msg → List Flock.Wit → List Msg)
    (calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)) : Prop :=
  (∀ x r ws, D.innerView H x r ws = (run H (replay (msgs x ws)) IV (xI D x) () fun i => coinSeq r i.val).seen) ∧
    ∀ x ws, D.calls x ws = (ws.mapIdx fun s w => calls x s w).flatten

/-- Part `s`'s recorded calls are among the calls read through the readers. -/
theorem mem_calls_of_part {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} {x : Msg} {ws : List Flock.Wit} {s : ℕ}
    (hs : s < ws.length) {e : Msg × Msg} (he : e ∈ calls x s ws[s]) :
    e ∈ (ws.mapIdx fun s w => calls x s w).flatten :=
  List.mem_flatten.2 ⟨_, List.getElem_mem (l := ws.mapIdx fun s w => calls x s w) (by simpa using hs),
    by simpa [List.getElem_mapIdx] using he⟩

/-- A call read through the readers is some part's. -/
theorem part_of_mem_calls {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} {x : Msg} {ws : List Flock.Wit}
    {e : Msg × Msg} (he : e ∈ (ws.mapIdx fun s w => calls x s w).flatten) :
    ∃ s, ∃ hs : s < ws.length, e ∈ calls x s ws[s] := by
  obtain ⟨l, hl, he⟩ := List.mem_flatten.1 he
  obtain ⟨s, hs, rfl⟩ := List.getElem_of_mem hl
  exact ⟨s, by simpa using hs, by simpa [List.getElem_mapIdx] using he⟩

/-- **The inner view is the inner verifier's run** (`InnerViewRun`) wherever `D` reads through readers at `I.V`: it is
`I.V`'s run of the readers' messages, and replaying a run's own messages is the run (`run_replay_self`). -/
theorem innerViewRun_of_readsThrough {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {msgs : Msg → List Flock.Wit → List Msg}
    {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} (hR : ReadsThrough H D I.V msgs calls) :
    InnerViewRun H D I DO := by
  intro decO x Ω P ω r c ws _
  unfold innerRun
  rw [hR.1 x r ws]
  exact congrArg Trace.seen (run_replay_self H I.V (xI D x) (msgs x ws) _).symm

omit [Fintype Coin] [Nonempty Coin] in
/-- **No witnesses record no calls** (`CallsNil`) wherever `D` reads through readers. -/
theorem callsNil_of_readsThrough {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin}
    {msgs : Msg → List Flock.Wit → List Msg} {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)}
    (hR : ReadsThrough H D IV msgs calls) : CallsNil H D := fun x e he => by
  rw [hR.2 x []] at he
  simp at he

/-! ## V*'s parts -/

/-- **V\* part `s`'s circuit checks the inner check `chk s`** (a code fact about V*'s part `s` and the layout): wherever
the part's outer relation holds at a V* statement `xO` of a compiled statement `x` of the shape, and its recorded calls
are true evaluations, the inner check `chk s` holds at the coins `xO` shows on every list of the inner messages that
agrees with the messages the part holds; or two of its recorded calls collide. -/
def PartChecks (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) → chk s x c ms) ∨
      (findColl (L.calls x s w)).isSome

/-- **V\* part `s` opens what it holds** (a code fact about V*'s part `s` and the layout): wherever the part's outer
relation holds at a V* statement `xO` of a compiled statement `x` of the shape, each inner round's message the part
holds opens under `opening` the round's commit string that `xO` shows (`cmt x i xO`), through calls the part
records. -/
def PartOpens (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (opening : Opening) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → ∀ i < L.rounds x, ∀ m, L.held x s w i = some m →
      ∃ cs, cs ⊆ L.calls x s w ∧ opening.opens (cmt x i xO) m cs

/-- **The staged V\* statements show the run** (a code fact about the V* statements the compiled verifier stages): on
an accepted run of a compiled statement of the shape, each V* statement shows the run's coins and, as the commit string
of each inner round the layout spans, the run's message at that round. -/
def Staged (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ s < D.nS x, shows s x ((runVstar H D I P x ω r).getD s []) (coinSeq r) ∧
        ∀ i < L.rounds x,
          cmt x i ((runVstar H D I P x ω r).getD s []) = ((run H P D.V x ω r).msgs.map Prod.fst).getD i []

/-- **The parts' inner checks are the inner verifier** (a code fact about the inner verifier `IV` and V*'s parts at
the shape): at a compiled statement of the shape, inner messages on which every part's inner check holds at the coins
`c` are accepted by `IV` at those coins. -/
def Decomp (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (shape : Form → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ s < D.nS x, chk s x c ms) → (run H (replay ms) IV (xI D x) () fun i => c i.val).out = .acc

/-- **The inner verifier is its checks together** (a code fact about the inner verifier `IV` at the shape): at a
compiled statement of the shape, inner messages on which each of its `nC x` checks holds at the coins `c` are accepted
by `IV` at those coins. -/
def InnerConj (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (shape : Form → Prop) (nC : Msg → ℕ)
    (check : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    (∀ j < nC x, check j x c ms) → (run H (replay ms) IV (xI D x) () fun i => c i.val).out = .acc

/-- **V\*'s parts cover the inner checks** (a code fact about how the compiled statement cuts the inner verifier into
V*'s parts, at the shape): each of the inner verifier's checks follows from some part's inner check. -/
def Covers (D : Deploy Coin) (shape : Form → Prop) (nC : Msg → ℕ) (check : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ j < nC x, ∃ s < D.nS x,
    ∀ (c : ℕ → Coin) (ms : List Msg), chk s x c ms → check j x c ms

omit [Fintype Coin] [Nonempty Coin] in
/-- **`Decomp` from the inner verifier's checks and the parts that cover them.** -/
theorem decomp_of_covers {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {shape : Form → Prop} {nC : Msg → ℕ}
    {check chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hI : InnerConj H D IV shape nC check)
    (hC : Covers D shape nC check chk) : Decomp H D IV shape chk := fun x st hst hf c ms h =>
  hI x st hst hf c ms fun j hj => by
    obtain ⟨s, hs, hs'⟩ := hC x st hst hf j hj
    exact hs' c ms (h s hs)

/-- **V\* part `s`'s SHA-512 gadgets compute `H`** (a code fact about V*'s part `s` and the layout, at the oracle):
wherever the part's outer relation holds at a V* statement of a compiled statement that parses, each evaluation the
part records is a true evaluation of `H`. -/
def PartCallsTrue (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → ∀ (xO : Msg) (w : Flock.Wit), Flock.check DO.parse xO w = true →
    ∀ e ∈ L.calls x s w, H e.1 = e.2

/-- **Every recorded call is a true evaluation** (`CallsRecord`'s second half, a code fact): wherever `E_O` gives
witnesses on an accepted run, every call they record is a true evaluation of `H`. -/
def CallsTrue (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ e ∈ D.calls x ws, H e.1 = e.2

theorem CallsRecord.toTrue {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    (h : CallsRecord H D I DO) : CallsTrue H D I DO := fun decO x Ω P ω r c ws hg => (h decO x Ω P ω r c ws hg).2

/-- **`CallsTrue` from V\*'s parts**: wherever the compiled verifier reads through the deployed readers and every
part's SHA-512 gadgets compute `H`, at an accepted run, whose statement parses (`Refuses`) and whose witnesses are in
the outer relation at its V* statements (`eoGives_outerHolds`). -/
theorem callsTrue_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} (hRef : Refuses H D) (hR : ReadsThrough H D IV L.msgs L.calls)
    (hT : ∀ s, PartCallsTrue H D DO L s) : CallsTrue H D I DO := by
  intro decO x Ω P ω r c ws hg e he
  obtain ⟨st, hst, -⟩ := hRef x Ω P ω r hg.1
  obtain ⟨hlen, hchk⟩ := eoGives_outerHolds hg
  rw [hR.2 x ws] at he
  obtain ⟨s, hs, he⟩ := part_of_mem_calls he
  exact hT s x st hst _ ws[s] (hchk s hs) e he

/-- **`CallsRecord` from V\*'s parts**: its second half is `callsTrue_of_parts`; its first half, the inner verifier's
calls among the recorded ones, is `hlog`. -/
theorem callsRecord_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} (hRef : Refuses H D) (hR : ReadsThrough H D IV L.msgs L.calls)
    (hT : ∀ s, PartCallsTrue H D DO L s)
    (hlog : ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω)
      (r : Fin (D.V.k x) → Coin) (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
      ∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws) :
    CallsRecord H D I DO := fun decO x Ω P ω r c ws hg =>
  ⟨hlog decO x Ω P ω r c ws hg, callsTrue_of_parts (I := I) hRef hR hT decO x Ω P ω r c ws hg⟩

/-! ## `VStarAccepts` from V*'s parts -/

/-- A collision among calls that are all among the recorded calls, which are all true evaluations, is one `findColl`
finds among the recorded calls. -/
theorem coll_of_sub {H : Msg → Msg} {l l' : List (Msg × Msg)} (hl' : ∀ p ∈ l', H p.1 = p.2)
    (hsub : ∀ p ∈ l, p ∈ l') (h : (findColl l).isSome) : ∃ u v, findColl l' = some (u, v) ∧ Collision H u v := by
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome_of_sub H hl' hsub h)
  exact ⟨u, v, huv, findColl_spec H l' hl' u v huv⟩

/-- A part's own colliding calls are a collision among the calls read through the readers. -/
theorem coll_of_part {H : Msg → Msg} {calls : Msg → ℕ → Flock.Wit → List (Msg × Msg)} {x : Msg}
    {ws : List Flock.Wit} {s : ℕ} (hs : s < ws.length)
    (htrue : ∀ e ∈ (ws.mapIdx fun s w => calls x s w).flatten, H e.1 = e.2) (h : (findColl (calls x s ws[s])).isSome) :
    ∃ u v, findColl (ws.mapIdx fun s w => calls x s w).flatten = some (u, v) ∧ Collision H u v :=
  coll_of_sub htrue (fun _ he => mem_calls_of_part hs he) h

omit [Fintype Coin] [Nonempty Coin] in
/-- **The parts hold the readers' messages, or the recorded calls collide**: on an accepted run of a compiled
statement of the shape, with outer witnesses in the outer relation at its V* statements whose recorded calls are true
evaluations, every message a part holds is the readers' message at its round; or two parts hold different messages at
one round, which open one commit string (`Staged`) two ways (`PartOpens`), a collision among the recorded calls
(`Opening.bind`). -/
theorem agree_or_coll {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {opening : Opening} {cmt : Msg → ℕ → Msg → Msg} (hR : ReadsThrough H D IV L.msgs L.calls)
    (hO : ∀ s, PartOpens D DO L shape opening cmt s) (hS : Staged H D I L shape shows cmt) {x : Msg} {st : Stmt}
    (hst : D.parse x = some st) (hf : shape st.form) {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin (D.V.k x) → Coin}
    (hacc : (run H P D.V x ω r).out = .acc) {ws : List Flock.Wit}
    (ho : OuterHolds DO (runVstar H D I P x ω r) (D.nS x) ws) (htrue : ∀ e ∈ D.calls x ws, H e.1 = e.2) :
    (∀ s (hs : s < ws.length), ∀ i < L.rounds x, ∀ m, L.held x s ws[s] i = some m → (L.msgs x ws).getD i [] = m) ∨
      ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v := by
  obtain ⟨hlen, hchk⟩ := ho
  rw [hR.2 x ws] at htrue ⊢
  by_cases hdiff : ∃ i < L.rounds x, ∃ s, ∃ hs : s < ws.length, ∃ s', ∃ hs' : s' < ws.length, ∃ m m',
      L.held x s ws[s] i = some m ∧ L.held x s' ws[s'] i = some m' ∧ m ≠ m'
  · obtain ⟨i, hi, s, hs, s', hs', m, m', hm, hm', hne⟩ := hdiff
    obtain ⟨cs, hcs, hop⟩ := hO s x st hst hf _ ws[s] (hchk s hs) i hi m hm
    obtain ⟨cs', hcs', hop'⟩ := hO s' x st hst hf _ ws[s'] (hchk s' hs') i hi m' hm'
    rw [(hS x st hst hf Ω P ω r hacc s (hlen ▸ hs)).2 i hi] at hop
    rw [(hS x st hst hf Ω P ω r hacc s' (hlen ▸ hs')).2 i hi] at hop'
    have hsub : cs ++ cs' ⊆ (ws.mapIdx fun s w => L.calls x s w).flatten := fun p hp =>
      (List.mem_append.1 hp).elim (fun h => mem_calls_of_part hs (hcs h)) fun h => mem_calls_of_part hs' (hcs' h)
    obtain ⟨q, hq, q', hq', hc⟩ := opening.bind H _ m m' cs cs' hop hop' hne fun e he => htrue e (hsub he)
    exact Or.inr (findColl_of_coll H htrue (List.map_subset _ hsub hq) (List.map_subset _ hsub hq') hc)
  · push Not at hdiff
    exact Or.inl fun s hs i hi m hm =>
      Layout.msgs_getD hi hs hm fun s'' hs'' m'' hm'' => hdiff i hi s'' hs'' s hs m'' m hm'' hm

omit [Fintype Coin] [Nonempty Coin] in
/-- **`VStarAccepts` from V*'s parts**, at any shape: the compiled verifier reads through the deployed readers, every
part checks its inner check and opens what it holds, the staged V* statements show the run, and the parts' checks are
the inner verifier. Off a collision, every part holds the readers' messages (`agree_or_coll`), so every part's inner
check holds on them, and `Decomp` gives an accepted inner view. -/
theorem vstarAccepts_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hC : ∀ s, PartChecks H D DO L shape shows chk s)
    (hO : ∀ s, PartOpens D DO L shape opening cmt s) (hS : Staged H D I L shape shows cmt)
    (hD : Decomp H D IV shape chk) : VStarAccepts H D I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  rcases agree_or_coll hR hO hS hst hf hacc ho htrue with hag | hcoll
  · obtain ⟨hlen, hchk⟩ := ho
    rw [hR.2 x ws] at htrue ⊢
    by_cases hcol : ∃ s, ∃ hs : s < ws.length, (findColl (L.calls x s ws[s])).isSome
    · obtain ⟨s, hs, h⟩ := hcol
      exact Or.inr (coll_of_part hs htrue h)
    push Not at hcol
    left
    rw [hR.1 x r ws]
    refine hD x st hst hf (coinSeq r) (L.msgs x ws) fun s hs => ?_
    have hs' : s < ws.length := hlen ▸ hs
    rcases hC s x st hst hf _ ws[s] (hchk s hs') fun e he => htrue e (mem_calls_of_part hs' he) with h | h
    · exact h (coinSeq r) (hS x st hst hf Ω P ω r hacc s hs).1 (L.msgs x ws) fun i hi m hm => hag s hs' i hi m hm
    · exact absurd h (by simpa using hcol s hs')
  · exact Or.inr hcoll

/-! ## At the three inner shapes -/

omit [Fintype Coin] [Nonempty Coin] in
/-- **`VStarAccepts` at the dense shape**, from V*'s parts there (`PublicAudit`'s inner protocol: the Ligerito levels'
openings and the rep's algebra in parts). -/
theorem vstarAccepts_dense {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hC : ∀ s, PartChecks H D DO L Form.Dense shows chk s)
    (hO : ∀ s, PartOpens D DO L Form.Dense opening cmt s) (hS : Staged H D I L Form.Dense shows cmt)
    (hD : Decomp H D IV Form.Dense chk) : VStarAccepts H D I DO Form.Dense :=
  vstarAccepts_of_parts hR hC hO hS hD

omit [Fintype Coin] [Nonempty Coin] in
/-- **`VStarAccepts` at the holo shape**, from V*'s parts there (`RecursiveAuditC3`'s inner protocol, private inner
circuits). -/
theorem vstarAccepts_holo {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hC : ∀ s, PartChecks H D DO L Form.Holo shows chk s)
    (hO : ∀ s, PartOpens D DO L Form.Holo opening cmt s) (hS : Staged H D I L Form.Holo shows cmt)
    (hD : Decomp H D IV Form.Holo chk) : VStarAccepts H D I DO Form.Holo :=
  vstarAccepts_of_parts hR hC hO hS hD

omit [Fintype Coin] [Nonempty Coin] in
/-- **`VStarAccepts` at the sparse shape**, from V*'s parts there (candidate 3's RecSparse phase and R's levels and
opening parts beside the dense ones). -/
theorem vstarAccepts_sparse {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D IV L.msgs L.calls) (hC : ∀ s, PartChecks H D DO L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens D DO L Form.Sparse opening cmt s) (hS : Staged H D I L Form.Sparse shows cmt)
    (hD : Decomp H D IV Form.Sparse chk) : VStarAccepts H D I DO Form.Sparse :=
  vstarAccepts_of_parts hR hC hO hS hD

/-- **`InnerAcceptsColl` from `VStarAccepts` at the three shapes**, as `innerAcceptsColl_of_vstar`, which reads only
`CallsRecord`'s second half (`CallsTrue`). -/
theorem innerAcceptsColl_of_vstar_true {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hD : VStarAccepts H D I DO Form.Dense) (hH : VStarAccepts H D I DO Form.Holo)
    (hS : VStarAccepts H D I DO Form.Sparse) : InnerAcceptsColl H D I DO := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨s, hs, -⟩ := hRef x Ω P ω r hg.1
  have ho := eoGives_outerHolds hg
  have ht := hT decO x Ω P ω r c ws hg
  rcases Form.shape_cases s.form with hf | hf | hf
  · exact hD x s hs hf Ω P ω r hg.1 ws ho ht
  · exact hH x s hs hf Ω P ω r hg.1 ws ho ht
  · exact hS x s hs hf Ω P ω r hg.1 ws ho ht

/-- **`InnerAcceptsColl` from V*'s parts at the three shapes** (`innerAcceptsColl_of_vstar_true`), each shape with its
own inner checks and the coins and commit strings its V* statements show, at one layout and one opening. -/
theorem innerAcceptsColl_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {IV : Verifier Coin} {L : Layout} {opening : Opening}
    {showsD showsH showsS : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chkD chkH chkS : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmtD cmtH cmtS : Msg → ℕ → Msg → Msg} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hR : ReadsThrough H D IV L.msgs L.calls)
    (hCD : ∀ s, PartChecks H D DO L Form.Dense showsD chkD s) (hOD : ∀ s, PartOpens D DO L Form.Dense opening cmtD s)
    (hSD : Staged H D I L Form.Dense showsD cmtD) (hDD : Decomp H D IV Form.Dense chkD)
    (hCH : ∀ s, PartChecks H D DO L Form.Holo showsH chkH s) (hOH : ∀ s, PartOpens D DO L Form.Holo opening cmtH s)
    (hSH : Staged H D I L Form.Holo showsH cmtH) (hDH : Decomp H D IV Form.Holo chkH)
    (hCS : ∀ s, PartChecks H D DO L Form.Sparse showsS chkS s)
    (hOS : ∀ s, PartOpens D DO L Form.Sparse opening cmtS s) (hSS : Staged H D I L Form.Sparse showsS cmtS)
    (hDS : Decomp H D IV Form.Sparse chkS) : InnerAcceptsColl H D I DO :=
  innerAcceptsColl_of_vstar_true hRef hT (vstarAccepts_dense hR hCD hOD hSD hDD)
    (vstarAccepts_holo hR hCH hOH hSH hDH) (vstarAccepts_sparse hR hCS hOS hSS hDS)

/-! ## Openings at the rounds the inner view plays -/

/-- **`D.calls` records each played inner message's opening** (`CallsOpen` at the rounds the inner view plays, a code
fact): wherever `E_O` gives witnesses on an accepted run, at each round the inner view plays, some recorded calls open
the run's message there, its commit string, to the inner view's message there under `opening`. -/
def CallsOpenPlayed (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (opening : Opening) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    ∀ j < (D.innerView H x r ws).msgs.length, ∃ cs, cs ⊆ D.calls x ws ∧
      opening.opens (((run H P D.V x ω r).msgs.map Prod.fst).getD j [])
        (((D.innerView H x r ws).msgs.map Prod.fst).getD j []) cs

/-- **Some part holds each inner round** (a code fact about the layout at the shape): at a compiled statement of the
shape, the layout spans the inner verifier's rounds, and some part holds each of them. -/
def Held (D : Deploy Coin) (I : Inner Coin W) (L : Layout) (shape : Form → Prop) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form →
    I.V.k (xI D x) ≤ L.rounds x ∧ ∀ i < L.rounds x, ∃ s < D.nS x, (L.msgAt x s i).isSome

/-- **The played openings at one shape, from V*'s parts**: the inner view's message at a round it plays is the
readers' (`run_replay_fst`), which is the message the first part that holds the round holds (`Layout.msgs_getD_held`);
that part opens it to the round's commit string its V* statement shows (`PartOpens`), which is the run's message there
(`Staged`), through calls it records. -/
theorem opensPlayed_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shape : Form → Prop} {opening : Opening} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {cmt : Msg → ℕ → Msg → Msg} (hR : ReadsThrough H D I.V L.msgs L.calls) (hH : Held D I L shape)
    (hO : ∀ s, PartOpens D DO L shape opening cmt s) (hS : Staged H D I L shape shows cmt)
    {decO : List (ℕ × Col) → List T} {x : Msg} {Ω : Type} {P : Prover Ω} {ω : Ω} {r : Fin (D.V.k x) → Coin}
    {c : (EO D I DO decO).C x} {ws : List Flock.Wit} (hg : EOGives H D I DO decO P x ω r c ws) {st : Stmt}
    (hst : D.parse x = some st) (hf : shape st.form) :
    ∀ j < (D.innerView H x r ws).msgs.length, ∃ cs, cs ⊆ D.calls x ws ∧
      opening.opens (((run H P D.V x ω r).msgs.map Prod.fst).getD j [])
        (((D.innerView H x r ws).msgs.map Prod.fst).getD j []) cs := by
  intro j hj
  obtain ⟨hlen, hchk⟩ := eoGives_outerHolds hg
  rw [hR.1 x r ws] at hj ⊢
  simp only [Trace.seen] at hj ⊢
  rw [run_replay_fst H I.V (xI D x) (L.msgs x ws) _ hj]
  have hjr : j < L.rounds x := lt_of_lt_of_le (lt_of_lt_of_le hj (run_length_le H _ I.V _ () _))
    (hH x st hst hf).1
  obtain ⟨s, hs, hh⟩ := (hH x st hst hf).2 j hjr
  obtain ⟨s₀, hs₀, hm⟩ := Layout.msgs_getD_held hjr (hlen ▸ hs) hh
  obtain ⟨cs, hcs, ho⟩ := hO s₀ x st hst hf _ ws[s₀] (hchk s₀ hs₀) j hjr _ hm
  rw [(hS x st hst hf Ω P ω r hg.1 s₀ (hlen ▸ hs₀)).2 j hjr] at ho
  refine ⟨cs, fun e he => ?_, ho⟩
  rw [hR.2 x ws]
  exact mem_calls_of_part hs₀ (hcs he)

/-- **`CallsOpenPlayed` from V*'s parts at the three shapes** (`opensPlayed_of_parts`): an accepted run's statement
parses (`Refuses`), and its form is dense, holo or sparse (`Form.shape_cases`). -/
theorem callsOpenPlayed_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {showsD showsH showsS : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {cmtD cmtH cmtS : Msg → ℕ → Msg → Msg} (hRef : Refuses H D) (hR : ReadsThrough H D I.V L.msgs L.calls)
    (hHD : Held D I L Form.Dense) (hOD : ∀ s, PartOpens D DO L Form.Dense opening cmtD s)
    (hSD : Staged H D I L Form.Dense showsD cmtD)
    (hHH : Held D I L Form.Holo) (hOH : ∀ s, PartOpens D DO L Form.Holo opening cmtH s)
    (hSH : Staged H D I L Form.Holo showsH cmtH)
    (hHS : Held D I L Form.Sparse) (hOS : ∀ s, PartOpens D DO L Form.Sparse opening cmtS s)
    (hSS : Staged H D I L Form.Sparse showsS cmtS) : CallsOpenPlayed H D I DO opening := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨st, hst, -⟩ := hRef x Ω P ω r hg.1
  rcases Form.shape_cases st.form with hf | hf | hf
  · exact opensPlayed_of_parts hR hHD hOD hSD hg hst hf
  · exact opensPlayed_of_parts hR hHH hOH hSH hg hst hf
  · exact opensPlayed_of_parts hR hHS hOS hSS hg hst hf

/-- **Binding at a fork, at the played rounds**: two inner views that differ at round `j` first differ at some round
`i ≤ j`. Both play round `i` (`run_replay_plays`: each is the inner verifier's run of its own messages, `InnerViewRun`,
and the two agree on their messages before `i` and on their coins before `j`), so both open their message `i` to the
runs' message `i`, which is one commit string since the runs share their coins before `j` (`run_fst_getD`): a
collision among the calls both record, which `Be` finds at its fork at `j`. -/
theorem forkColl_of_openingPlayed {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} (hV : InnerViewRun H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hT : CallsTrue H D I DO) (decO : List (ℕ × Col) → List T) (x : Msg)
    {Ω : Type} (P : Prover Ω) {j : ℕ}
    (t : Ω × (Fin (D.V.k x) → Coin) × (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x × (EO D I DO decO).C x)
    (h : OpensDiffer H D I DO decO P x j t) : forkColl H D I DO decO P x j t := by
  classical
  obtain ⟨ws, ws', hg, hg', hd⟩ := h
  have hex : ∃ i, ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD i [] ≠
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD i [] := ⟨j, hd⟩
  have hi := Nat.find_spec hex
  have hmin : ∀ i < Nat.find hex, ((D.innerView H x t.2.1 ws).msgs.map Prod.fst).getD i [] =
      ((D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.map Prod.fst).getD i [] :=
    fun i h => not_not.1 (Nat.find_min hex h)
  have hij : Nat.find hex ≤ j := Nat.find_min' hex hd
  generalize Nat.find hex = i at hi hmin hij
  have hc : ∀ i' < i, coinSeq (innerCoins D I x t.2.1) i' =
      coinSeq (innerCoins D I x (fork j t.2.1 t.2.2.1)) i' := fun i' hi' => by
    unfold coinSeq innerCoins coinSeq
    split_ifs with h1 h2
    · exact fork_agree j t.2.1 t.2.2.1 ⟨i', h2⟩ (by simp only; omega)
    · rfl
    · rfl
  have hv := congrArg Seen.msgs (hV decO x Ω P t.1 t.2.1 _ ws hg)
  have hv' := congrArg Seen.msgs (hV decO x Ω P t.1 (fork j t.2.1 t.2.2.1) _ ws' hg')
  simp only [innerRun, Trace.seen] at hv hv'
  have hboth : i < (D.innerView H x t.2.1 ws).msgs.length ∧
      i < (D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.length := by
    by_cases h1 : i < (D.innerView H x t.2.1 ws).msgs.length
    · refine ⟨h1, ?_⟩
      rw [hv']
      exact run_replay_plays H I.V (xI D x) _ _ _ _ _ hmin hc (hv ▸ h1)
    · by_cases h2 : i < (D.innerView H x (fork j t.2.1 t.2.2.1) ws').msgs.length
      · refine absurd ?_ h1
        rw [hv]
        exact run_replay_plays H I.V (xI D x) _ _ _ _ _ (fun i h => (hmin i h).symm) (fun i h => (hc i h).symm)
          (hv' ▸ h2)
      · refine absurd ?_ hi
        push Not at h1 h2
        rw [List.getD_eq_default _ _ (by simpa using h1), List.getD_eq_default _ _ (by simpa using h2)]
  obtain ⟨cs, hcs, ho⟩ := hO decO x Ω P t.1 t.2.1 _ ws hg _ hboth.1
  obtain ⟨cs', hcs', ho'⟩ := hO decO x Ω P t.1 (fork j t.2.1 t.2.2.1) _ ws' hg' _ hboth.2
  rw [run_fst_getD H P D.V x t.1 i fun i' hi' => fork_agree j t.2.1 t.2.2.1 i' (by omega)] at ho
  have hsub : cs ++ cs' ⊆ D.calls x ws ++ D.calls x ws' := fun e he =>
    (List.mem_append.1 he).elim (fun he => List.mem_append_left _ (hcs he)) fun he => List.mem_append_right _ (hcs' he)
  have hall : ∀ e ∈ D.calls x ws ++ D.calls x ws', H e.1 = e.2 := fun e he =>
    (List.mem_append.1 he).elim (hT decO x Ω P t.1 t.2.1 _ ws hg e) (hT decO x Ω P t.1 _ _ ws' hg' e)
  obtain ⟨q, hq, q', hq', hcol⟩ := opening.bind H _ _ _ cs cs' ho ho' hi fun e he => hall e (hsub he)
  obtain ⟨u, v, huv, hcuv⟩ := findColl_of_coll H hall (List.map_subset _ hsub hq) (List.map_subset _ hsub hq') hcol
  refine ⟨u, v, ?_, hcuv⟩
  rw [hg.2, hg'.2]
  exact huv

/-- **The inner rounds' differing openings cost at most `k(x)·Pr[Be]`, from the played openings**:
`sum_opensDiffer_le` with `CallsOpenPlayed` for `CallsOpen` and `CallsTrue` for `CallsRecord`, the inner view the inner
verifier's run (`InnerViewRun`). -/
theorem sum_opensDiffer_le_played {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {opening : Opening} (hV : InnerViewRun H D I DO)
    (hO : CallsOpenPlayed H D I DO opening) (hT : CallsTrue H D I DO) (hR : InnerRounds D I)
    (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) ≤ D.V.k x * PrBe H D I DO decO P x := by
  have hq : ∀ j, 0 ≤ Pr (forkColl H D I DO decO P x j) := fun j => Pr_nonneg _
  have hle : ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) ≤
      ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := by
    calc ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j)
        ≤ ∑ j : Fin (I.V.k (xI D x)), Pr (forkColl H D I DO decO P x j) :=
          Finset.sum_le_sum fun j _ => Pr_mono fun t => forkColl_of_openingPlayed hV hO hT decO x P t
      _ ≤ ∑ j : Fin (D.V.k x), Pr (forkColl H D I DO decO P x j) := by
          rw [Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j)),
            Fin.sum_univ_eq_sum_range (fun j => Pr (forkColl H D I DO decO P x j))]
          exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (hR x)) fun j _ _ => hq j
  by_cases hk : 0 < D.V.k x
  · rw [PrBe_eq_avg H D I DO decO P x hk, mul_div_cancel₀ _ (by exact_mod_cast hk.ne')]
    exact hle
  · have h0 : D.V.k x = 0 := Nat.eq_zero_of_not_pos hk
    have hI : I.V.k (xI D x) = 0 := Nat.eq_zero_of_le_zero (h0 ▸ hR x)
    have hS : ∑ j : Fin (I.V.k (xI D x)), Pr (OpensDiffer H D I DO decO P x j) = 0 := by
      rw [Fin.sum_univ_eq_sum_range (fun j => Pr (OpensDiffer H D I DO decO P x j)), hI, Finset.sum_range_zero]
    rw [hS]
    exact mul_nonneg (Nat.cast_nonneg _) (PrBe_nonneg H D I DO decO P x)

/-! ## The inner verifier's calls, off a collision -/

/-- **`CallsRecord`'s first half off a collision** (a code fact): wherever `E_O` gives witnesses on an accepted run,
the inner verifier's calls in the inner view's run are among the calls the witnesses record, or two recorded calls are
a collision. -/
def CallsLogColl (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) : Prop :=
  ∀ (decO : List (ℕ × Col) → List T) (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin)
    (c : (EO D I DO decO).C x) (ws : List Flock.Wit), EOGives H D I DO decO P x ω r c ws →
    (∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws) ∨
      ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v

/-- **The inner verifier's calls are its parts'** (a code fact about the inner verifier `IV` at the shape): at a
compiled statement of the shape, each SHA-512 call `IV` makes on the inner messages `ms` at the coins `c` is one that
some part's inner check makes there (`logs s x c ms`). -/
def LogDecomp (H : Msg → Msg) (D : Deploy Coin) (IV : Verifier Coin) (shape : Form → Prop)
    (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (c : ℕ → Coin) (ms : List Msg),
    ∀ e ∈ (run H (replay ms) IV (xI D x) () fun i => c i.val).vLog, ∃ s < D.nS x, e ∈ logs s x c ms

/-- **V\* part `s` records its inner check's calls** (a code fact about V*'s part `s` and the layout): wherever the
part's outer relation holds at a V* statement `xO` of a compiled statement `x` of the shape, and its recorded calls
are true evaluations, the calls its inner check makes at the coins `xO` shows, on every list of the inner messages
that agrees with the messages the part holds, are among the calls it records; or two of its recorded calls collide. -/
def PartLogs (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : Layout) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Flock.Wit),
    Flock.check DO.parse xO w = true → (∀ e ∈ L.calls x s w, H e.1 = e.2) →
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg,
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) → ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) ∨
      (findColl (L.calls x s w)).isSome

/-- **The inner verifier's calls at one shape, from V*'s parts, off a collision**: off a collision every part holds
the readers' messages (`agree_or_coll`); the inner view's run is `I.V`'s run of them (`run_replay_self`), each of whose
calls some part's inner check makes (`LogDecomp`), which that part records (`PartLogs`). -/
theorem logColl_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hR : ReadsThrough H D I.V L.msgs L.calls) (hL : ∀ s, PartLogs H D DO L shape shows logs s)
    (hO : ∀ s, PartOpens D DO L shape opening cmt s) (hS : Staged H D I L shape shows cmt)
    (hLD : LogDecomp H D I.V shape logs) {decO : List (ℕ × Col) → List T} {x : Msg} {Ω : Type} {P : Prover Ω}
    {ω : Ω} {r : Fin (D.V.k x) → Coin} {c : (EO D I DO decO).C x} {ws : List Flock.Wit}
    (hg : EOGives H D I DO decO P x ω r c ws) (htrue : ∀ e ∈ D.calls x ws, H e.1 = e.2) {st : Stmt}
    (hst : D.parse x = some st) (hf : shape st.form) :
    (∀ e ∈ (innerRun H D I x r ws).vLog, e ∈ D.calls x ws) ∨
      ∃ u v, findColl (D.calls x ws) = some (u, v) ∧ Collision H u v := by
  have ho := eoGives_outerHolds hg
  rcases agree_or_coll hR hO hS hst hf hg.1 ho htrue with hag | hcoll
  · obtain ⟨hlen, hchk⟩ := ho
    have hrun : innerRun H D I x r ws =
        run H (replay (L.msgs x ws)) I.V (xI D x) () fun i => coinSeq r i.val := by
      unfold innerRun
      rw [hR.1 x r ws]
      exact run_replay_self H I.V (xI D x) (L.msgs x ws) _
    rw [hR.2 x ws] at htrue ⊢
    by_cases hcol : ∃ s, ∃ hs : s < ws.length, (findColl (L.calls x s ws[s])).isSome
    · obtain ⟨s, hs, h⟩ := hcol
      exact Or.inr (coll_of_part hs htrue h)
    push Not at hcol
    refine Or.inl fun e he => ?_
    rw [hrun] at he
    obtain ⟨s, hs, hes⟩ := hLD x st hst hf (coinSeq r) (L.msgs x ws) e he
    have hs' : s < ws.length := hlen ▸ hs
    rcases hL s x st hst hf _ ws[s] (hchk s hs') fun e he => htrue e (mem_calls_of_part hs' he) with h | h
    · exact mem_calls_of_part hs' (h (coinSeq r) (hS x st hst hf Ω P ω r hg.1 s hs).1 (L.msgs x ws)
        (fun i hi m hm => hag s hs' i hi m hm) e hes)
    · exact absurd h (by simpa using hcol s hs')
  · exact Or.inr hcoll

/-- **`CallsLogColl` from V*'s parts at the three shapes** (`logColl_of_parts`): an accepted run's statement parses
(`Refuses`), its form is dense, holo or sparse (`Form.shape_cases`), and its recorded calls are true evaluations
(`CallsTrue`). -/
theorem callsLogColl_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {showsD showsH showsS : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logsD logsH logsS : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {cmtD cmtH cmtS : Msg → ℕ → Msg → Msg} (hRef : Refuses H D) (hT : CallsTrue H D I DO)
    (hR : ReadsThrough H D I.V L.msgs L.calls)
    (hLD : ∀ s, PartLogs H D DO L Form.Dense showsD logsD s) (hOD : ∀ s, PartOpens D DO L Form.Dense opening cmtD s)
    (hSD : Staged H D I L Form.Dense showsD cmtD) (hDD : LogDecomp H D I.V Form.Dense logsD)
    (hLH : ∀ s, PartLogs H D DO L Form.Holo showsH logsH s) (hOH : ∀ s, PartOpens D DO L Form.Holo opening cmtH s)
    (hSH : Staged H D I L Form.Holo showsH cmtH) (hDH : LogDecomp H D I.V Form.Holo logsH)
    (hLS : ∀ s, PartLogs H D DO L Form.Sparse showsS logsS s)
    (hOS : ∀ s, PartOpens D DO L Form.Sparse opening cmtS s) (hSS : Staged H D I L Form.Sparse showsS cmtS)
    (hDS : LogDecomp H D I.V Form.Sparse logsS) : CallsLogColl H D I DO := by
  intro decO x Ω P ω r c ws hg
  obtain ⟨st, hst, -⟩ := hRef x Ω P ω r hg.1
  have ht := hT decO x Ω P ω r c ws hg
  rcases Form.shape_cases st.form with hf | hf | hf
  · exact logColl_of_parts hR hLD hOD hSD hDD hg ht hst hf
  · exact logColl_of_parts hR hLH hOH hSH hDH hg ht hst hf
  · exact logColl_of_parts hR hLS hOS hSS hDS hg ht hst hf

/-- The calls `Be` reads at a draw are true evaluations: `E_O`'s witnesses' where it gives them (`CallsTrue`), none's
where it gives up (`CallsNil`). -/
theorem calls_true_of {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} (hT : CallsTrue H D I DO)
    (hN0 : CallsNil H D) (ω : Ω) (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    ∀ e ∈ D.calls x ((eoOut H D I DO decO P x ω d).getD []), H e.1 = e.2 := by
  rcases hws : eoOut H D I DO decO P x ω d with _ | ws
  · exact hN0 x
  · exact hT decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws)

/-- `P_I`'s aborting run records only calls that `E_O`'s witnesses there record, or those witnesses' calls collide
(`innerF_vLog_sub` off a collision). -/
theorem innerF_vLog_sub_coll {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decO : List (ℕ × Col) → List T} {Ω : Type} {P : Prover Ω} {x : Msg} {decC : Msg → Msg → Coin}
    (hL : CallsLogColl H D I DO) (hpc : PublicCoin H I decC) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    (∀ e ∈ (innerF H D I DO decO P x decC ω d).vLog, e ∈ D.calls x ((eoOut H D I DO decO P x ω d).getD [])) ∨
      ∃ u v, findColl (D.calls x ((eoOut H D I DO decO P x ω d).getD [])) = some (u, v) ∧ Collision H u v := by
  by_cases hk : keepD H D I DO decO P x ω d = true
  · have hk' : (eoOut H D I DO decO P x ω d).isSome = true ∧ Agrees H D I DO decO P x ω d := by
      simpa only [keepD, decide_eq_true_eq] using hk
    obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hk'.1
    rw [innerF_eq_innerRun hpc hws hk'.2, hws]
    exact hL decO x Ω P ω d.1 d.2 ws (eoGives_of_some hws)
  · exact Or.inl fun e he => by simp [innerF, hk, abortTrace] at he

/-- **`B` at `P_I`'s aborting runs finds only collisions `B_I` finds, from `CallsLogColl`**: `PrBAbort_le_PrBI` with
`CallsLogColl` and `CallsTrue` for `CallsRecord`. Where a kept run's calls aren't all recorded, the recorded calls at
its draw collide, which `B_I` finds as well. -/
theorem PrBAbort_le_PrBI_coll {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {decC : Msg → Msg → Coin} (hL : CallsLogColl H D I DO) (hT : CallsTrue H D I DO) (hN0 : CallsNil H D)
    (hpc : PublicCoin H I decC) (hR : InnerRounds D I) (decO : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    PrBAbort H I.V (innerP H D I DO decO P x decC) (xI D x) (keepA H D I DO decO P x) ≤
      PrBI H D I DO decO P x := by
  have hcoll : ∀ (j : Fin (I.V.k (xI D x))) (ω : Ω) (ρ ρ' : Fin (I.V.k (xI D x)) → Coin)
      (a a' : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x),
      (findColl ((abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) ω (ρ, a)).vLog ++
        (abortRun H (innerP H D I DO decO P x decC) I.V (xI D x) (keepA H D I DO decO P x) ω
          (forkCoins j ρ ρ', a')).vLog)).isSome →
      ∃ u v, findColl (D.calls x ((eoOut H D I DO decO P x ω (extend ρ a.1, a.2)).getD []) ++
        D.calls x ((eoOut H D I DO decO P x ω (fork j.val (extend ρ a.1) (extend ρ' a'.1), a'.2)).getD [])) =
          some (u, v) ∧ Collision H u v := by
    intro j ω ρ ρ' a a' h
    rw [abortRun_eq (hR x), abortRun_eq (hR x)] at h
    rw [extend_forkCoins]
    have htrue : ∀ p ∈ D.calls x ((eoOut H D I DO decO P x ω (extend ρ a.1, a.2)).getD []) ++
        D.calls x ((eoOut H D I DO decO P x ω (extend (forkCoins j ρ ρ') a'.1, a'.2)).getD []), H p.1 = p.2 :=
      fun p hp => (List.mem_append.1 hp).elim (calls_true_of hT hN0 ω _ p) (calls_true_of hT hN0 ω _ p)
    have hsome : (findColl (D.calls x ((eoOut H D I DO decO P x ω (extend ρ a.1, a.2)).getD []) ++
        D.calls x ((eoOut H D I DO decO P x ω (extend (forkCoins j ρ ρ') a'.1, a'.2)).getD []))).isSome := by
      rcases innerF_vLog_sub_coll (decC := decC) hL hpc ω (extendDraw (ρ, a)) with h1 | ⟨u, v, huv, -⟩
      · rcases innerF_vLog_sub_coll (decC := decC) hL hpc ω (extendDraw (forkCoins j ρ ρ', a')) with
          h2 | ⟨u, v, huv, -⟩
        · exact findColl_isSome_of_sub H htrue (fun p hp => (List.mem_append.1 hp).elim
            (fun hp => List.mem_append_left _ (h1 p hp)) fun hp => List.mem_append_right _ (h2 p hp)) h
        · exact findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_right _ hp)
            (Option.isSome_iff_exists.2 ⟨_, huv⟩)
      · exact findColl_isSome_of_sub H htrue (fun p hp => List.mem_append_left _ hp)
          (Option.isSome_iff_exists.2 ⟨_, huv⟩)
    obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hsome
    exact ⟨u, v, huv, findColl_spec H _ htrue u v huv⟩
  unfold PrBAbort PrBI
  by_cases hk : 0 < I.V.k (xI D x)
  · simp only [hk, ↓reduceDIte]
    unfold Finder.Pr
    refine (Pr_mono fun s hs => ?_).trans_eq
      ((unifMap_bAbort (Ω := Ω) (C := (EO D I DO decO).C x) (hR x)).Pr _)
    obtain ⟨u, v, huv, -⟩ := hs
    exact hcoll s.1.1 s.1.2.1 s.1.2.2.1 s.1.2.2.2 s.2.1 s.2.2 (Option.isSome_iff_exists.2 ⟨(u, v), huv⟩)
  · simp only [hk, ↓reduceDIte, le_refl]

/-! ## The staged V* statements, off the run -/

/-- **An accepted run's first pairs show its coins** (a code fact about the compiled verifier): on an accepted run, the
prover sends at least the inner and staging rounds' `k_I + kS` messages, and the verifier's answer to each of them is a
message that shows that round's coin, which `decC` reads off it. -/
def RunPairs (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (decC : Msg → Msg → Coin) : Prop :=
  ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
    I.V.k (xI D x) + D.kS x ≤ (run H P D.V x ω r).msgs.length ∧ ∀ i < I.V.k (xI D x) + D.kS x,
      ∃ m, ((run H P D.V x ω r).msgs.getD i ([], .rej)).2 = .msg m ∧ decC x m = coinSeq r i

/-- **The V\* statements show what they are staged at** (a code fact about the staging `D.vstar` alone, with no run): at
a compiled statement of the shape, each V* statement staged at `k_I + kS` pairs shows every coin sequence the pairs'
verifier messages show (`shows`) and, as the commit string of each inner round the layout spans, the pairs' prover
message at that round (`cmt`). -/
def VstarShows (D : Deploy Coin) (I : Inner Coin W) (L : Layout) (shape : Form → Prop) (decC : Msg → Msg → Coin)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ ps : List (Msg × Out),
    ps.length = I.V.k (xI D x) + D.kS x → ∀ c : ℕ → Coin,
      (∀ i < ps.length, ∃ m, (ps.getD i ([], .rej)).2 = .msg m ∧ decC x m = c i) →
        ∀ s < D.nS x, shows s x ((D.vstar x ps).getD s []) c ∧
          ∀ i < L.rounds x, cmt x i ((D.vstar x ps).getD s []) = (ps.map Prod.fst).getD i []

omit [Fintype Coin] [Nonempty Coin] in
/-- **`Staged` off the run**: the staging shows what it is staged at (`VstarShows`), an accepted run's first pairs show
its coins (`RunPairs`), and the layout spans no more than those rounds. A run's V* statements are the staging's at its
first `k_I + kS` pairs (`runVstar`). -/
theorem staged_of_vstarShows {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {L : Layout}
    {shape : Form → Prop} {decC : Msg → Msg → Coin} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {cmt : Msg → ℕ → Msg → Msg} (hP : RunPairs H D I decC) (hV : VstarShows D I L shape decC shows cmt)
    (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x) : Staged H D I L shape shows cmt := by
  intro x st hst hf Ω P ω r hacc s hs
  obtain ⟨hlen, hpr⟩ := hP x Ω P ω r hacc
  have hpl : ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x)).length = I.V.k (xI D x) + D.kS x := by
    simp only [List.length_take]
    omega
  have hc : ∀ i < ((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x)).length, ∃ m,
      (((run H P D.V x ω r).msgs.take (I.V.k (xI D x) + D.kS x)).getD i ([], .rej)).2 = .msg m ∧
        decC x m = coinSeq r i := by
    intro i hi
    rw [hpl] at hi
    obtain ⟨m, hm, hd⟩ := hpr i hi
    refine ⟨m, ?_, hd⟩
    rwa [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hi, ← List.getD_eq_getElem?_getD]
  obtain ⟨h1, h2⟩ := hV x st hst hf _ hpl (coinSeq r) hc s hs
  refine ⟨h1, fun i hi => ?_⟩
  unfold runVstar
  rw [h2 i hi]
  have hi' : i < I.V.k (xI D x) + D.kS x := Nat.lt_of_lt_of_le hi (hL x)
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_take_of_lt hi']

/-! ## The code facts about the readers and V*, together -/

/-- **V\*'s parts at one shape** (the code facts about V*'s parts, the staged V* statements and the inner verifier at
the shape, at one layout and one opening): some inner check and calls per part, and the coins and commit strings the
V* statements show, at which every part checks its inner check, opens what it holds and records its check's calls
(`PartChecks`, `PartOpens`, `PartLogs`), the staged V* statements show the run (`Staged`), the parts' checks are the
inner verifier and its calls are theirs (`Decomp`, `LogDecomp`), and some part holds each inner round (`Held`). -/
def VStarParts (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (L : Layout)
    (opening : Opening) (shape : Form → Prop) : Prop :=
  ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)),
    (∀ s, PartChecks H D DO L shape shows chk s) ∧ (∀ s, PartOpens D DO L shape opening cmt s) ∧
      (∀ s, PartLogs H D DO L shape shows logs s) ∧ Staged H D I L shape shows cmt ∧ Decomp H D I.V shape chk ∧
      LogDecomp H D I.V shape logs ∧ Held D I L shape

/-- **The code facts the recursion's KS reads about the readers and V\*, from V*'s parts**: wherever the compiled
verifier reads through the deployed readers at the inner verifier (`ReadsThrough`, which `deployRec` at
`⟨L.msgs, L.calls⟩` does by its definition), every part's SHA-512 gadgets compute `H` (`PartCallsTrue`), and V*'s parts
meet `VStarParts` at each shape: the inner view is the inner verifier's run (`InnerViewRun`), no witnesses record no
calls (`CallsNil`), every recorded call is a true evaluation (`CallsTrue`), the played rounds' openings are recorded
(`CallsOpenPlayed`), and off a collision the inner verifier's calls are recorded (`CallsLogColl`) and the inner view
accepts (`InnerAcceptsColl`). -/
theorem codeFacts_of_parts {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} (hRef : Refuses H D) (hR : ReadsThrough H D I.V L.msgs L.calls)
    (hT : ∀ s, PartCallsTrue H D DO L s) (hD : VStarParts H D I DO L opening Form.Dense)
    (hH : VStarParts H D I DO L opening Form.Holo) (hS : VStarParts H D I DO L opening Form.Sparse) :
    InnerViewRun H D I DO ∧ CallsNil H D ∧ CallsTrue H D I DO ∧ CallsOpenPlayed H D I DO opening ∧
      CallsLogColl H D I DO ∧ InnerAcceptsColl H D I DO := by
  obtain ⟨showsD, chkD, cmtD, logsD, hCD, hOD, hLD, hSD, hDD, hLDD, hHD⟩ := hD
  obtain ⟨showsH, chkH, cmtH, logsH, hCH, hOH, hLH, hSH, hDH, hLDH, hHH⟩ := hH
  obtain ⟨showsS, chkS, cmtS, logsS, hCS, hOS, hLS, hSS, hDS, hLDS, hHS⟩ := hS
  have hTr : CallsTrue H D I DO := callsTrue_of_parts hRef hR hT
  exact ⟨innerViewRun_of_readsThrough hR, callsNil_of_readsThrough hR, hTr,
    callsOpenPlayed_of_parts hRef hR hHD hOD hSD hHH hOH hSH hHS hOS hSS,
    callsLogColl_of_parts hRef hTr hR hLD hOD hSD hLDD hLH hOH hSH hLDH hLS hOS hSS hLDS,
    innerAcceptsColl_of_parts hRef hTr hR hCD hOD hSD hDD hCH hOH hSH hDH hCS hOS hSS hDS⟩

/-- **V\*'s parts at one shape, with the staging off the run**: `VStarParts` with the staging's fact about `D.vstar`
alone (`VstarShows`, at the coins `decC` reads) in place of `Staged`. -/
def VStarPartsStaging (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) (H : Msg → Msg) (L : Layout)
    (opening : Opening) (decC : Msg → Msg → Coin) (shape : Form → Prop) : Prop :=
  ∃ (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop)
    (cmt : Msg → ℕ → Msg → Msg) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)),
    (∀ s, PartChecks H D DO L shape shows chk s) ∧ (∀ s, PartOpens D DO L shape opening cmt s) ∧
      (∀ s, PartLogs H D DO L shape shows logs s) ∧ VstarShows D I L shape decC shows cmt ∧
        Decomp H D I.V shape chk ∧ LogDecomp H D I.V shape logs ∧ Held D I L shape

omit [Fintype Coin] [Nonempty Coin] in
/-- **`VStarParts` from the staging off the run** (`staged_of_vstarShows`). -/
theorem vstarParts_of_staging {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin} {shape : Form → Prop} (hP : RunPairs H D I decC)
    (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x) (h : VStarPartsStaging D I DO H L opening decC shape) :
    VStarParts H D I DO L opening shape := by
  obtain ⟨shows, chk, cmt, logs, hC, hO, hLg, hV, hD, hLD, hH⟩ := h
  exact ⟨shows, chk, cmt, logs, hC, hO, hLg, staged_of_vstarShows hP hV hL, hD, hLD, hH⟩

/-- **The code facts about the readers and V\*, with the staging off the run**: `codeFacts_of_parts`, with each shape's
`Staged` from the staging's fact about `D.vstar` (`VstarShows`) and an accepted run's first pairs (`RunPairs`). -/
theorem codeFacts_of_staging {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin} (hRef : Refuses H D)
    (hR : ReadsThrough H D I.V L.msgs L.calls) (hT : ∀ s, PartCallsTrue H D DO L s) (hP : RunPairs H D I decC)
    (hL : ∀ x, L.rounds x ≤ I.V.k (xI D x) + D.kS x) (hD : VStarPartsStaging D I DO H L opening decC Form.Dense)
    (hH : VStarPartsStaging D I DO H L opening decC Form.Holo)
    (hS : VStarPartsStaging D I DO H L opening decC Form.Sparse) :
    InnerViewRun H D I DO ∧ CallsNil H D ∧ CallsTrue H D I DO ∧ CallsOpenPlayed H D I DO opening ∧
      CallsLogColl H D I DO ∧ InnerAcceptsColl H D I DO :=
  codeFacts_of_parts hRef hR hT (vstarParts_of_staging hP hL hD) (vstarParts_of_staging hP hL hH)
    (vstarParts_of_staging hP hL hS)

end Facts

/-! ## An opening through two calls -/

/-- **An opening through two recorded calls**, as hm96 rows open: the commit string `a` splits as `b ‖ c`
(`split a`); the opening's salt `y` hashes to `c` (the call `(salt y, c)`), and the message `m` digests to `dig b y`,
`b` unmasked at the salt (the call `(enc m, dig b y)`). Two openings of one commit string to different messages differ
in their salts, whose hashes then collide, or share the salt, whose messages' digests then collide; so it binds where
`salt` and `enc` are injective. -/
def Opening.twoCalls (salt enc : Msg → Msg) (hsalt : Function.Injective salt) (henc : Function.Injective enc)
    (split : Msg → Msg × Msg) (dig : Msg → Msg → Msg) : Opening where
  opens a m cs := ∃ y, (salt y, (split a).2) ∈ cs ∧ (enc m, dig (split a).1 y) ∈ cs
  bind H a m m' cs cs' := by
    rintro ⟨y, hy, hm⟩ ⟨y', hy', hm'⟩ hne hall
    have hin : ∀ {p}, p ∈ cs → p.1 ∈ (cs ++ cs').map Prod.fst := fun hp =>
      List.mem_map.2 ⟨_, List.mem_append_left _ hp, rfl⟩
    have hin' : ∀ {p}, p ∈ cs' → p.1 ∈ (cs ++ cs').map Prod.fst := fun hp =>
      List.mem_map.2 ⟨_, List.mem_append_right _ hp, rfl⟩
    by_cases h : y = y'
    · subst h
      exact ⟨enc m, hin hm, enc m', hin' hm', fun e => hne (henc e),
        (hall _ (List.mem_append_left _ hm)).trans (hall _ (List.mem_append_right _ hm')).symm⟩
    · exact ⟨salt y, hin hy, salt y', hin' hy', fun e => h (hsalt e),
        (hall _ (List.mem_append_left _ hy)).trans (hall _ (List.mem_append_right _ hy')).symm⟩

end Claims.Recursion.VStar
