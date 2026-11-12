module

public import Proofs.Flock.Claims.Sampling.SessionSlots
public import Proofs.Flock.Claims.Sampling.SessionSteps
public import Proofs.Flock.Claims.Sampling.SessionTo

/-!
# `SessionRunsTo` from a sampled verifier that runs its sessions in sequence

`SessionSeq D`: the sampled verifier `D.V` answers its first `pre x` rounds itself, with no `H` call and never
accepting; then runs `nS x` sessions back to back, session `i` on its `kS x i` rounds from `sessOff (pre x) (kS x) i`,
each round as Flock's verifier `D.flock.V` at the statement the prefix's prover messages name for it (`stmt x as i`), on
the session's messages and coins, with Flock's acceptance shown as the message `mark`, which Flock never answers; and
decides in one last round, the only one that may accept, which accepts only where each session's messages have a prefix
at which Flock does not answer with a message. Every checked unit's session is one of the sessions (`idx`), and the
session read off a run's transcript is that session's statement and its rounds' transcript cut at its first decision
(`cutSeen`) where the prefix passed and the run answered every round before the session with a message, and else a
refused run: `xR x i`, at which Flock refuses the first message `[]`.

`sessionRunsTo_of_seq`: such a verifier runs each session as Flock runs (`SessionRunsTo`). At any coins, a session's
Flock rounds are its rounds in the run, in order; Flock's prover plays the sampled prover's messages after the
transcript before them; the session the run shows is that Flock run's transcript, whose `H` calls the run makes; and the
run accepts only where that Flock run accepts: a Flock run that refuses ends the run, and one that answers all its
rounds with messages leaves the last round no decision to find in the session's messages.
-/

@[expose] public section

namespace Claims.Sampling

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The sampled verifier's sequential structure**: a prefix of `pre x` rounds the verifier answers with no `H` call
and no acceptance; then `nS x` sessions back to back, session `i` on its `kS x i` rounds from
`sessOff (pre x) (kS x) i`, each round Flock's verifier at the statement the prefix's prover messages name for the
session (`stmt`), on the session's messages and coins, with an acceptance shown as `mark`, a message Flock never
answers, wherever Flock answered every shorter prefix of the session's messages with a message; no acceptance before the
last round; and a last round that accepts only where each session's messages have a prefix at which Flock does not
answer with a message. Where the prefix's last answer is a message, session `i`'s statement has its `kS x i` rounds. A
checked unit's session is one of the sessions (`idx`) and its rounds that session's, and the session read off a
transcript is that session's statement and its rounds' transcript cut at their first decision (`cutSeen`) where the
prefix passed and every round before the session was answered with a message, and the refused run of `xR x i` (`kS x i`
rounds, refusing the message `[]` at once) where the run refused before the session. -/
structure SessionSeq (D : Deploy Coin Col T) where
  pre : Msg → ℕ
  nS : Msg → ℕ
  kS : Msg → ℕ → ℕ
  idx : Msg → ℕ → ℕ
  stmt : Msg → List Msg → ℕ → Msg
  xR : Msg → ℕ → Msg
  mark : Msg
  pre_pos : ∀ x, 0 < pre x
  kS_pos : ∀ x i, 0 < kS x i
  k_eq : ∀ x, D.V.k x = sessOff (pre x) (kS x) (nS x) + 1
  idx_lt : ∀ x r u, (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) → idx x u < nS x
  rounds_iff : ∀ x u (t : Fin (D.V.k x)), t ∈ D.rounds x u ↔
    sessOff (pre x) (kS x) (idx x u) ≤ t.val ∧ t.val < sessOff (pre x) (kS x) (idx x u) + kS x (idx x u)
  next_pre : ∀ x (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ pre x → cs.length = as.length →
    ∃ o, D.V.next x as cs = .ret o ∧ o ≠ .acc
  next_mid : ∀ (H : Msg → Msg) x (as : List Msg) (cs : List Coin), pre x < as.length → as.length < D.V.k x →
    cs.length = as.length → ((D.V.next x as cs).eval H).1 ≠ .acc
  next_sess : ∀ (H : Msg → Msg) x (as : List Msg) (cs : List Coin) i, i < nS x →
    sessOff (pre x) (kS x) i < as.length → as.length ≤ sessOff (pre x) (kS x) i + kS x i → cs.length = as.length →
    (∀ j, 0 < j → j < as.length - sessOff (pre x) (kS x) i → ∃ m,
      ((D.flock.V.next (stmt x (as.take (pre x)) i) ((as.drop (sessOff (pre x) (kS x) i)).take j)
        ((cs.drop (sessOff (pre x) (kS x) i)).take j)).eval H).1 = .msg m) →
    (D.V.next x as cs).eval H =
      (markOut mark ((D.flock.V.next (stmt x (as.take (pre x)) i) (as.drop (sessOff (pre x) (kS x) i))
          (cs.drop (sessOff (pre x) (kS x) i))).eval H).1,
        ((D.flock.V.next (stmt x (as.take (pre x)) i) (as.drop (sessOff (pre x) (kS x) i))
          (cs.drop (sessOff (pre x) (kS x) i))).eval H).2)
  next_last : ∀ (H : Msg → Msg) x (as : List Msg) (cs : List Coin), as.length = D.V.k x → cs.length = as.length →
    ((D.V.next x as cs).eval H).1 = .acc → ∀ i < nS x, ∃ j, 0 < j ∧ j ≤ kS x i ∧ ∀ m,
      ((D.flock.V.next (stmt x (as.take (pre x)) i) ((as.drop (sessOff (pre x) (kS x) i)).take j)
        ((cs.drop (sessOff (pre x) (kS x) i)).take j)).eval H).1 ≠ .msg m
  flock_ne : ∀ (H : Msg → Msg) xS (as : List Msg) (cs : List Coin), ((D.flock.V.next xS as cs).eval H).1 ≠ .msg mark
  k_stmt : ∀ x (as : List Msg) (cs : List Coin) m i, as.length = pre x → cs.length = as.length →
    D.V.next x as cs = .ret (.msg m) → i < nS x → D.flock.V.k (stmt x as i) = kS x i
  k_xR : ∀ x i, D.flock.V.k (xR x i) = kS x i
  refuses : ∀ x i (c : Coin), D.flock.V.next (xR x i) [[]] [c] = .ret .rej
  session_pass : ∀ x u (ms : List (Msg × Out)) rest out (cs : List Coin) m,
    ms.length = sessOff (pre x) (kS x) (idx x u) → idx x u < nS x → (∀ e ∈ ms, ∃ m, e.2 = .msg m) →
    cs.length = pre x → D.V.next x ((ms.take (pre x)).map Prod.fst) cs = .ret (.msg m) →
    D.session x u ⟨ms ++ rest, out⟩ =
      some (stmt x ((ms.take (pre x)).map Prod.fst) (idx x u), cutSeen mark (kS x (idx x u)) rest)
  session_refused : ∀ x u (ms : List (Msg × Out)) a, ms.length < sessOff (pre x) (kS x) (idx x u) →
    (∀ e ∈ ms, ∃ m, e.2 = .msg m) →
    D.session x u ⟨ms ++ [(a, .rej)], .rej⟩ = some (xR x (idx x u), ⟨[([], .rej)], .rej⟩)

/-! ## A run's rounds against `H` -/

section Rounds

variable {Ω Ω' : Type} (H : Msg → Msg)

/-- **A round answered with a message**: the prover's message, the answer and both parties' calls are appended, and the
run continues. -/
theorem stepWith_msg_of (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (n : ℕ) (hist : List (Msg × Msg))
    (as : List Msg) (t : Transcript) (pa : Msg × List (Msg × Msg)) (hpa : (P.next x hist ω).eval H = pa) (m : Msg)
    (l : List (Msg × Msg)) (hv : (v (as ++ [pa.1])).eval H = (.msg m, l)) :
    stepWith H P x ω v (n + 1) hist as t =
      stepWith H P x ω v n (hist ++ [(pa.1, m)]) (as ++ [pa.1])
        ⟨t.msgs ++ [(pa.1, .msg m)], t.pLog ++ pa.2, t.vLog ++ l, t.out⟩ := by
  subst hpa
  simp only [stepWith, hv]

/-- **A round accepted**: the prover's message, the acceptance and both parties' calls are appended, and the run
accepts. -/
theorem stepWith_acc_of (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (n : ℕ) (hist : List (Msg × Msg))
    (as : List Msg) (t : Transcript) (pa : Msg × List (Msg × Msg)) (hpa : (P.next x hist ω).eval H = pa)
    (l : List (Msg × Msg)) (hv : (v (as ++ [pa.1])).eval H = (.acc, l)) :
    stepWith H P x ω v (n + 1) hist as t = ⟨t.msgs ++ [(pa.1, .acc)], t.pLog ++ pa.2, t.vLog ++ l, .acc⟩ := by
  subst hpa
  simp only [stepWith, hv]

/-- **A round refused**: the prover's message, the refusal and both parties' calls are appended, and the run refuses. -/
theorem stepWith_rej_of (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (n : ℕ) (hist : List (Msg × Msg))
    (as : List Msg) (t : Transcript) (pa : Msg × List (Msg × Msg)) (hpa : (P.next x hist ω).eval H = pa)
    (l : List (Msg × Msg)) (hv : (v (as ++ [pa.1])).eval H = (.rej, l)) :
    stepWith H P x ω v (n + 1) hist as t = ⟨t.msgs ++ [(pa.1, .rej)], t.pLog ++ pa.2, t.vLog ++ l, .rej⟩ := by
  subst hpa
  simp only [stepWith, hv]

/-- **A run only extends its transcript and the verifier's calls.** -/
theorem stepWith_ext (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (n : ℕ) :
    ∀ (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), ∃ (rest : List (Msg × Out)) (lm : List (Msg × Msg)),
      (stepWith H P x ω v n hist as t).msgs = t.msgs ++ rest ∧
        (stepWith H P x ω v n hist as t).vLog = t.vLog ++ lm := by
  induction n with
  | zero => intro hist as t; exact ⟨[], [], by simp [stepWith], by simp [stepWith]⟩
  | succ n ih =>
    intro hist as t
    rcases hvb : (v (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, l⟩
    cases o with
    | msg m =>
      rw [stepWith_msg_of H P x ω v n hist as t _ rfl m l hvb]
      obtain ⟨rest, lm, h1, h2⟩ := ih (hist ++ [(((P.next x hist ω).eval H).1, m)])
        (as ++ [((P.next x hist ω).eval H).1])
        ⟨t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)], t.pLog ++ ((P.next x hist ω).eval H).2, t.vLog ++ l,
          t.out⟩
      exact ⟨(((P.next x hist ω).eval H).1, .msg m) :: rest, l ++ lm, by simp [h1], by simp [h2]⟩
    | acc =>
      rw [stepWith_acc_of H P x ω v n hist as t _ rfl l hvb]
      exact ⟨[(((P.next x hist ω).eval H).1, .acc)], l, rfl, rfl⟩
    | rej =>
      rw [stepWith_rej_of H P x ω v n hist as t _ rfl l hvb]
      exact ⟨[(((P.next x hist ω).eval H).1, .rej)], l, rfl, rfl⟩

/-- **An acceptance is an answer**: a run that accepts has the verifier accept at the messages so far followed by some
of the run's, at most as many as its rounds. -/
theorem acc_of_stepWith (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (n : ℕ) :
    ∀ (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), (stepWith H P x ω v n hist as t).out = .acc →
      ∃ bs : List Msg, bs ≠ [] ∧ bs.length ≤ n ∧ ((v (as ++ bs)).eval H).1 = .acc := by
  induction n with
  | zero => intro hist as t h; simp [stepWith] at h
  | succ n ih =>
    intro hist as t h
    rcases hvb : (v (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, l⟩
    cases o with
    | msg m =>
      rw [stepWith_msg_of H P x ω v n hist as t _ rfl m l hvb] at h
      obtain ⟨bs, hne, hlen, hacc⟩ := ih _ _ _ h
      exact ⟨((P.next x hist ω).eval H).1 :: bs, by simp, by simp; omega, by simpa using hacc⟩
    | acc => exact ⟨[((P.next x hist ω).eval H).1], by simp, by simp, by rw [hvb]⟩
    | rej =>
      rw [stepWith_rej_of H P x ω v n hist as t _ rfl l hvb] at h
      simp at h

/-- **A run's transcript, its acceptance shown as `mark`, cut back**: where the verifier never answers the message
`mark`, the entries a run of `k` rounds adds, its acceptance shown as `mark` and followed by anything, cut at `k`
entries (`cutSeen`), are those entries and the run's decision. -/
theorem cutSeen_stepWith (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (mark : Msg)
    (hne : ∀ bs, ((v bs).eval H).1 ≠ .msg mark) (k : ℕ) :
    ∀ (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), ∃ new : List (Msg × Out),
      (stepWith H P x ω v k hist as t).msgs = t.msgs ++ new ∧
        ∀ rest, cutSeen mark k (new.map (markE mark) ++ rest) = ⟨new, (stepWith H P x ω v k hist as t).out⟩ := by
  induction k with
  | zero => intro hist as t; exact ⟨[], by simp [stepWith], fun rest => by simp [stepWith, cutSeen]⟩
  | succ k ih =>
    intro hist as t
    rcases hvb : (v (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, l⟩
    cases o with
    | msg m =>
      have hm : m ≠ mark := fun h => hne (as ++ [((P.next x hist ω).eval H).1]) (by rw [hvb, h])
      rw [stepWith_msg_of H P x ω v k hist as t _ rfl m l hvb]
      obtain ⟨new, h1, h2⟩ := ih (hist ++ [(((P.next x hist ω).eval H).1, m)])
        (as ++ [((P.next x hist ω).eval H).1])
        ⟨t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)], t.pLog ++ ((P.next x hist ω).eval H).2, t.vLog ++ l,
          t.out⟩
      refine ⟨(((P.next x hist ω).eval H).1, .msg m) :: new, by simp [h1], fun rest => ?_⟩
      show cutSeen mark (k + 1) ((((P.next x hist ω).eval H).1, .msg m) :: (new.map (markE mark) ++ rest)) = _
      simp only [cutSeen, hm, ↓reduceIte, h2 rest]
    | acc =>
      rw [stepWith_acc_of H P x ω v k hist as t _ rfl l hvb]
      exact ⟨[(((P.next x hist ω).eval H).1, .acc)], rfl, fun rest => by simp [markE, markOut, cutSeen]⟩
    | rej =>
      rw [stepWith_rej_of H P x ω v k hist as t _ rfl l hvb]
      exact ⟨[(((P.next x hist ω).eval H).1, .rej)], rfl, fun rest => by simp [markE, markOut, cutSeen]⟩

/-- **A session's rounds inside a run**: rounds against `v`, which at the messages `as₀` followed by `bs` (at most `K`
of them, where `v'` answered every shorter prefix of `bs` with a message) answers what `v'` answers at `bs`, an
acceptance shown as `mark`, with `v'`'s calls, of a prover that after `hist₀` plays `P'`: the run shows `P'`'s run
against `v'`, its acceptance shown as `mark`, after the transcript `m₀`, makes its calls after `l₀`, and accepts only
where that run accepts or, that run having answered all `K` of its rounds with messages, `v` accepts at `as₀`, those `K`
messages and at most `n` more. -/
theorem stepWith_session (P : Prover Ω) (P' : Prover Ω') (x x' : Msg) (ω : Ω) (ω' : Ω') (v v' : List Msg → OC Out)
    (mark : Msg) (hist₀ : List (Msg × Msg)) (as₀ : List Msg) (m₀ : List (Msg × Out)) (p₀ l₀ : List (Msg × Msg))
    (K n : ℕ) (hP : ∀ hist, P.next x (hist₀ ++ hist) ω = P'.next x' hist ω')
    (hv : ∀ bs : List Msg, bs ≠ [] → bs.length ≤ K →
      (∀ j, 0 < j → j < bs.length → ∃ m, ((v' (bs.take j)).eval H).1 = .msg m) →
      (v (as₀ ++ bs)).eval H = (markOut mark ((v' bs).eval H).1, ((v' bs).eval H).2))
    (hne : ∀ bs, ((v' bs).eval H).1 ≠ .msg mark) (k : ℕ) :
    ∀ (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), as.length + k = K →
      (∀ j, 0 < j → j ≤ as.length → ∃ m, ((v' (as.take j)).eval H).1 = .msg m) →
      ∃ (rest : List (Msg × Out)) (lm : List (Msg × Msg)),
        (stepWith H P x ω v (k + n) (hist₀ ++ hist) (as₀ ++ as)
            ⟨m₀ ++ t.msgs.map (markE mark), p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩).msgs =
          m₀ ++ (stepWith H P' x' ω' v' k hist as t).msgs.map (markE mark) ++ rest ∧
        (stepWith H P x ω v (k + n) (hist₀ ++ hist) (as₀ ++ as)
            ⟨m₀ ++ t.msgs.map (markE mark), p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩).vLog =
          l₀ ++ (stepWith H P' x' ω' v' k hist as t).vLog ++ lm ∧
        ((stepWith H P x ω v (k + n) (hist₀ ++ hist) (as₀ ++ as)
            ⟨m₀ ++ t.msgs.map (markE mark), p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩).out = .acc →
          (stepWith H P' x' ω' v' k hist as t).out = .acc ∨
          ∃ (as' bs : List Msg), as'.length = K ∧
            (∀ j, 0 < j → j ≤ K → ∃ m, ((v' (as'.take j)).eval H).1 = .msg m) ∧
            bs ≠ [] ∧ bs.length ≤ n ∧ ((v (as₀ ++ as' ++ bs)).eval H).1 = .acc) := by
  induction k with
  | zero =>
    intro hist as t hlen hdec
    rw [Nat.zero_add]
    obtain ⟨rest, lm, h1, h2⟩ := stepWith_ext H P x ω v n (hist₀ ++ hist) (as₀ ++ as)
      ⟨m₀ ++ t.msgs.map (markE mark), p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩
    refine ⟨rest, lm, ?_, ?_, fun hacc => Or.inr ?_⟩
    · rw [h1]
      simp [stepWith]
    · rw [h2]
      simp [stepWith]
    · obtain ⟨bs, hbs, hbl, hb⟩ := acc_of_stepWith H P x ω v n _ _ _ hacc
      exact ⟨as, bs, by omega, fun j hj hjK => hdec j hj (by omega), hbs, hbl, hb⟩
  | succ k ih =>
    intro hist as t hlen hdec
    obtain ⟨pa, hpa⟩ : ∃ pa, (P'.next x' hist ω').eval H = pa := ⟨_, rfl⟩
    have hpaW : (P.next x (hist₀ ++ hist) ω).eval H = pa := by rw [hP hist, hpa]
    have hund : ∀ j, 0 < j → j < (as ++ [pa.1]).length → ∃ m, ((v' ((as ++ [pa.1]).take j)).eval H).1 = .msg m := by
      intro j hj hjl
      rw [List.length_append, List.length_singleton] at hjl
      rw [List.take_append_of_le_length (by omega)]
      exact hdec j hj (by omega)
    have hw := hv (as ++ [pa.1]) (by simp) (by simp; omega) hund
    rw [← List.append_assoc] at hw
    have hkn : k + 1 + n = (k + n) + 1 := by omega
    rw [hkn]
    rcases hvb : (v' (as ++ [pa.1])).eval H with ⟨o, l⟩
    rw [hvb] at hw
    cases o with
    | msg m =>
      have hm : m ≠ mark := fun h => hne (as ++ [pa.1]) (by rw [hvb, h])
      have hw' : (v (as₀ ++ as ++ [pa.1])).eval H = (.msg m, l) := by simpa [markOut] using hw
      rw [stepWith_msg_of H P x ω v (k + n) (hist₀ ++ hist) (as₀ ++ as)
          ⟨m₀ ++ t.msgs.map (markE mark), p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩ pa hpaW m l hw',
        stepWith_msg_of H P' x' ω' v' k hist as t pa hpa m l hvb]
      have hdec' : ∀ j, 0 < j → j ≤ (as ++ [pa.1]).length → ∃ m, ((v' ((as ++ [pa.1]).take j)).eval H).1 = .msg m := by
        intro j hj hjl
        rcases Nat.lt_or_ge j (as ++ [pa.1]).length with h | h
        · exact hund j hj h
        · rw [List.take_of_length_le h, hvb]
          exact ⟨m, rfl⟩
      obtain ⟨rest, lm, h1, h2, h3⟩ := ih (hist ++ [(pa.1, m)]) (as ++ [pa.1])
        ⟨t.msgs ++ [(pa.1, .msg m)], t.pLog ++ pa.2, t.vLog ++ l, t.out⟩ (by simp; omega) hdec'
      refine ⟨rest, lm, ?_, ?_, fun hacc => h3 ?_⟩
      · simpa only [List.append_assoc, List.map_append, List.map_cons, List.map_nil, markE, markOut] using h1
      · simpa only [List.append_assoc, List.map_append, List.map_cons, List.map_nil, markE, markOut] using h2
      · simpa only [List.append_assoc, List.map_append, List.map_cons, List.map_nil, markE, markOut] using hacc
    | acc =>
      have hw' : (v (as₀ ++ as ++ [pa.1])).eval H = (.msg mark, l) := by simpa [markOut] using hw
      rw [stepWith_msg_of H P x ω v (k + n) (hist₀ ++ hist) (as₀ ++ as)
          ⟨m₀ ++ t.msgs.map (markE mark), p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩ pa hpaW mark l hw',
        stepWith_acc_of H P' x' ω' v' k hist as t pa hpa l hvb]
      obtain ⟨rest, lm, h1, h2⟩ := stepWith_ext H P x ω v (k + n) (hist₀ ++ hist ++ [(pa.1, mark)])
        (as₀ ++ as ++ [pa.1])
        ⟨m₀ ++ t.msgs.map (markE mark) ++ [(pa.1, .msg mark)], p₀ ++ t.pLog ++ pa.2, l₀ ++ t.vLog ++ l, t.out⟩
      refine ⟨rest, lm, ?_, ?_, fun _ => Or.inl rfl⟩
      · rw [h1]
        simp [markE, markOut]
      · rw [h2]
        simp
    | rej =>
      have hw' : (v (as₀ ++ as ++ [pa.1])).eval H = (.rej, l) := by simpa [markOut] using hw
      rw [stepWith_rej_of H P x ω v (k + n) (hist₀ ++ hist) (as₀ ++ as)
          ⟨m₀ ++ t.msgs.map (markE mark), p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩ pa hpaW l hw',
        stepWith_rej_of H P' x' ω' v' k hist as t pa hpa l hvb]
      exact ⟨[], [], by simp [markE, markOut], by simp, fun h => by simp at h⟩

end Rounds

omit [Fintype Coin] [Nonempty Coin] in
/-- **A session's coins**: the run's coins from round `o` on, `l` of them, are those of the session's rounds `emb`, the
rounds from `o`. -/
theorem coins_drop {k k' : ℕ} (r : Fin k → Coin) (o l : ℕ) (emb : Fin k' → Fin k)
    (hemb : ∀ j, (emb j).val = o + j.val) (hk : o + k' ≤ k) (hl : l ≤ k') :
    ((List.range (o + l)).map (coinSeq r)).drop o = (List.range l).map (coinSeq fun j => r (emb j)) := by
  rw [List.range_add, List.map_append, List.drop_left' (by simp), List.map_map]
  apply List.map_congr_left
  intro j hj
  rw [List.mem_range] at hj
  have h1 : o + j < k := by omega
  have h2 : j < k' := by omega
  simp only [Function.comp, coinSeq, h1, h2, ↓reduceDIte]
  congr 1
  exact Fin.ext (by rw [hemb])

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **The rounds up to `n`**: where the verifier accepts at no messages up to `n`, from a transcript `ms` of rounds
answered with messages, `j` more rounds either reach a transcript `ms'` of `n` rounds answered with messages, whose last
round the verifier answers with a message at `r`, or refuse one, ending the run; both, with the transcript and both
parties' calls, are the same at every coins `r'` that agree with `r` below `n`. -/
theorem run_upto {D : Deploy Coin Col T} (H : Msg → Msg) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin (D.V.k x) → Coin) (n rest : ℕ)
    (hna : ∀ (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ n → cs.length = as.length →
      ((D.V.next x as cs).eval H).1 ≠ .acc) (j : ℕ) :
    ∀ (ms : List (Msg × Out)) (ps ls : List (Msg × Msg)), ms.length + j = n → (∀ e ∈ ms, ∃ m, e.2 = .msg m) →
      (ms ≠ [] → ∃ m, ((D.V.next x (ms.map Prod.fst) ((List.range ms.length).map (coinSeq r))).eval H).1 = .msg m) →
      (∃ (ms' : List (Msg × Out)) (ps' ls' : List (Msg × Msg)), ms'.length = n ∧ ms <+: ms' ∧
          (∀ e ∈ ms', ∃ m, e.2 = .msg m) ∧
          (ms' ≠ [] → ∃ m,
            ((D.V.next x (ms'.map Prod.fst) ((List.range ms'.length).map (coinSeq r))).eval H).1 = .msg m) ∧
          ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < n → r' t = r t) →
            stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (j + rest)
                (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, ls, .rej⟩ =
              stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) rest
                (ms'.map fun e => (e.1, msgOf e.2)) (ms'.map Prod.fst) ⟨ms', ps', ls', .rej⟩) ∨
      (∃ (ms' : List (Msg × Out)) (a : Msg) (ps' ls' : List (Msg × Msg)), ms'.length < n ∧
          (∀ e ∈ ms', ∃ m, e.2 = .msg m) ∧
          ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < n → r' t = r t) →
            stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (j + rest)
                (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, ls, .rej⟩ =
              ⟨ms' ++ [(a, .rej)], ps', ls', .rej⟩) := by
  induction j with
  | zero =>
    intro ms ps ls hlen hms hlast
    exact Or.inl ⟨ms, ps, ls, by simpa using hlen, List.prefix_refl ms, hms, hlast, fun r' _ => by rw [Nat.zero_add]⟩
  | succ j ih =>
    intro ms ps ls hlen hms _
    obtain ⟨pa, hpa⟩ : ∃ pa, (P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H = pa := ⟨_, rfl⟩
    have hle : (ms.map Prod.fst ++ [pa.1]).length ≤ n := by
      simp only [List.length_append, List.length_map, List.length_singleton]
      omega
    rcases hvb : (D.V.next x (ms.map Prod.fst ++ [pa.1])
        ((List.range (ms.map Prod.fst ++ [pa.1]).length).map (coinSeq r))).eval H with ⟨o, l⟩
    have hvb' : ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < n → r' t = r t) →
        ((fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (ms.map Prod.fst ++ [pa.1])).eval H =
          (o, l) := by
      intro r' hr'
      show (D.V.next x (ms.map Prod.fst ++ [pa.1])
        ((List.range (ms.map Prod.fst ++ [pa.1]).length).map (coinSeq r'))).eval H = (o, l)
      rw [coins_eq hr' hle]
      exact hvb
    have hsucc : j + 1 + rest = (j + rest) + 1 := by omega
    cases o with
    | msg m =>
      have hms' : ∀ e ∈ ms ++ [(pa.1, Out.msg m)], ∃ m, e.2 = .msg m := by
        intro e he
        rcases List.mem_append.1 he with he | he
        · exact hms e he
        · rw [List.mem_singleton] at he
          exact ⟨m, by rw [he]⟩
      have hlast' : ∃ m', ((D.V.next x ((ms ++ [(pa.1, Out.msg m)]).map Prod.fst)
          ((List.range (ms ++ [(pa.1, Out.msg m)]).length).map (coinSeq r))).eval H).1 = .msg m' := by
        refine ⟨m, ?_⟩
        simp only [List.map_append, List.map_cons, List.map_nil, List.length_append, List.length_map,
          List.length_singleton] at hvb ⊢
        rw [hvb]
      rcases ih (ms ++ [(pa.1, .msg m)]) (ps ++ pa.2) (ls ++ l) (by simp; omega) hms' (fun _ => hlast') with
        ⟨ms', ps', ls', h1, hp, h2, h3, h4⟩ | ⟨ms', a, ps', ls', h1, h2, h4⟩
      · refine Or.inl ⟨ms', ps', ls', h1, (List.prefix_append ms _).trans hp, h2, h3, fun r' hr' => ?_⟩
        rw [hsucc, stepWith_msg_of H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
          (j + rest) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, ls, .rej⟩ pa hpa m l
          (hvb' r' hr'), ← h4 r' hr']
        simp [msgOf]
      · refine Or.inr ⟨ms', a, ps', ls', h1, h2, fun r' hr' => ?_⟩
        rw [hsucc, stepWith_msg_of H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
          (j + rest) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, ls, .rej⟩ pa hpa m l
          (hvb' r' hr'), ← h4 r' hr']
        simp [msgOf]
    | rej =>
      refine Or.inr ⟨ms, pa.1, ps ++ pa.2, ls ++ l, by omega, hms, fun r' hr' => ?_⟩
      rw [hsucc]
      exact stepWith_rej_of H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
        (j + rest) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, ls, .rej⟩ pa hpa l (hvb' r' hr')
    | acc =>
      exact (hna (ms.map Prod.fst ++ [pa.1]) ((List.range (ms.map Prod.fst ++ [pa.1]).length).map (coinSeq r))
        (by simp) hle (by simp) (by rw [hvb])).elim

namespace SessionSeq

variable {D : Deploy Coin Col T} (S : SessionSeq D)

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **A run refused before a unit's session** shows the refused run of `xR x i` as the session, and does not accept. -/
theorem runsTo_refused (H : Msg → Msg) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (u : ℕ)
    (hi : S.idx x u < S.nS x) (ms : List (Msg × Out)) (a : Msg) (ps ls : List (Msg × Msg))
    (hlen : ms.length < sessOff (S.pre x) (S.kS x) (S.idx x u)) (hms : ∀ e ∈ ms, ∃ m, e.2 = .msg m)
    (hrun : ∀ r' : Fin (D.V.k x) → Coin,
      (∀ t : Fin (D.V.k x), t.val < sessOff (S.pre x) (S.kS x) (S.idx x u) → r' t = r t) →
        run H P D.V x ω r' = ⟨ms ++ [(a, .rej)], ps, ls, .rej⟩) :
    ∃ (xS : Msg) (emb : Fin (D.flock.V.k xS) → Fin (D.V.k x)) (P' : Prover Unit),
      StrictMono emb ∧ (∀ t, t ∈ D.rounds x u ↔ ∃ i, emb i = t) ∧
      (∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
        (run H P D.V x ω r').out = .acc → (run H P' D.flock.V xS () fun i => r' (emb i)).out = .acc) ∧
      ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, (∀ i, t < emb i) → r' t = r t) →
        D.session x u (run H P D.V x ω r').seen = some (xS, (run H P' D.flock.V xS () fun i => r' (emb i)).seen) ∧
          ∀ q ∈ (run H P' D.flock.V xS () fun i => r' (emb i)).vLog, q ∈ (run H P D.V x ω r').vLog := by
  have hk : D.flock.V.k (S.xR x (S.idx x u)) = S.kS x (S.idx x u) := S.k_xR x _
  have hoK : sessOff (S.pre x) (S.kS x) (S.idx x u) + S.kS x (S.idx x u) < D.V.k x := by
    have := sessOff_add_le (S.pre x) (S.kS x) hi
    rw [S.k_eq]
    omega
  let emb : Fin (D.flock.V.k (S.xR x (S.idx x u))) → Fin (D.V.k x) := fun j =>
    ⟨sessOff (S.pre x) (S.kS x) (S.idx x u) + j.val, by have := j.isLt; omega⟩
  let P' : Prover Unit := ⟨fun _ _ _ => .ret []⟩
  have hflock : ∀ c : Fin (D.flock.V.k (S.xR x (S.idx x u))) → Coin,
      run H P' D.flock.V (S.xR x (S.idx x u)) () c = ⟨[([], .rej)], [], [], .rej⟩ := fun c =>
    stepWith_refused H (S.xR x (S.idx x u))
      (fun as => D.flock.V.next (S.xR x (S.idx x u)) as ((List.range as.length).map (coinSeq c)))
      (S.refuses x _ (coinSeq c 0)) _ (by rw [hk]; exact S.kS_pos x _)
  have hbefore : ∀ t : Fin (D.V.k x), t.val < sessOff (S.pre x) (S.kS x) (S.idx x u) → t ∉ D.rounds x u :=
    fun t ht h => by
      rw [S.rounds_iff] at h
      omega
  refine ⟨S.xR x (S.idx x u), emb, P', fun _ _ h => Nat.add_lt_add_left h _, fun t => ?_, fun r' hr' h => ?_,
    fun r' hr' => ?_⟩
  · rw [S.rounds_iff]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨⟨t.val - sessOff (S.pre x) (S.kS x) (S.idx x u), by rw [hk]; omega⟩,
        Fin.ext (show sessOff (S.pre x) (S.kS x) (S.idx x u) + (t.val - sessOff (S.pre x) (S.kS x) (S.idx x u)) =
          t.val by omega)⟩
    · rintro ⟨j, rfl⟩
      have := lt_of_lt_of_eq j.isLt hk
      exact ⟨Nat.le_add_right _ _, by show sessOff (S.pre x) (S.kS x) (S.idx x u) + j.val < _; omega⟩
  · rw [hrun r' fun t ht => hr' t (hbefore t ht)] at h
    simp at h
  · rw [hrun r' fun t ht => hr' t (hbefore t ht) fun j => show t.val < sessOff (S.pre x) (S.kS x) (S.idx x u) + j.val by
      omega, hflock]
    exact ⟨S.session_refused x u ms a hlen hms, fun _ hq => by simp at hq⟩

end SessionSeq

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **`SessionRunsTo` from the sequential structure**: at coins that check a unit, its session is Flock's run on its
session's rounds, of the prover that plays the sampled prover's messages after the transcript before them (or, where the
run refused before them, of a prover whose first message Flock refuses at `xR x i`); the run accepts only where that run
accepts; and it is the session the run shows, whose `H` calls the run makes. -/
theorem sessionRunsTo_of_seq {D : Deploy Coin Col T} (S : SessionSeq D) (H : Msg → Msg) : SessionRunsTo H D := by
  intro x Ω P ω r u hu
  have hi : S.idx x u < S.nS x := S.idx_lt x r u hu
  have hoK : sessOff (S.pre x) (S.kS x) (S.idx x u) + S.kS x (S.idx x u) < D.V.k x := by
    have := sessOff_add_le (S.pre x) (S.kS x) hi
    rw [S.k_eq]
    omega
  have hpo : S.pre x ≤ sessOff (S.pre x) (S.kS x) (S.idx x u) := le_sessOff _ _ _
  have hrun : ∀ r' : Fin (D.V.k x) → Coin, run H P D.V x ω r' =
      stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
        (S.pre x + (D.V.k x - S.pre x)) [] [] ⟨[], [], [], .rej⟩ := fun r' => by
    rw [Nat.add_sub_of_le (by omega)]
    rfl
  have hnaA : ∀ (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ S.pre x → cs.length = as.length →
      ((D.V.next x as cs).eval H).1 ≠ .acc := fun as cs h1 h2 h3 => by
    obtain ⟨o, ho, hne⟩ := S.next_pre x as cs h1 h2 h3
    rw [ho]
    exact hne
  rcases run_upto H x P ω r (S.pre x) (D.V.k x - S.pre x) hnaA (S.pre x) [] [] [] (by simp) (by simp) (by simp) with
    ⟨msA, psA, lsA, hlenA, -, hmsA, hlastA, hA⟩ | ⟨ms, a, ps, ls, hlen, hms, hA⟩
  swap
  · exact S.runsTo_refused H x P ω r u hi ms a ps ls (by omega) hms fun r' hr' => by
      have h0 := hA r' fun t ht => hr' t (by omega)
      simp only [List.map_nil] at h0
      rw [hrun r', h0]
  have hA' : ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < S.pre x → r' t = r t) →
      stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
          (S.pre x + (D.V.k x - S.pre x)) [] [] ⟨[], [], [], .rej⟩ =
        stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (D.V.k x - S.pre x)
          (msA.map fun e => (e.1, msgOf e.2)) (msA.map Prod.fst) ⟨msA, psA, lsA, .rej⟩ := fun r' hr' => by
    have h0 := hA r' hr'
    simp only [List.map_nil] at h0
    exact h0
  have hAne : msA ≠ [] := by
    rintro rfl
    have := S.pre_pos x
    simp at hlenA
    omega
  obtain ⟨mA, hmA⟩ := hlastA hAne
  obtain ⟨oA, hoA, -⟩ := S.next_pre x (msA.map Prod.fst) ((List.range msA.length).map (coinSeq r))
    (by simpa using hAne) (by simp [hlenA]) (by simp)
  rw [hoA] at hmA
  simp only [OC.eval] at hmA
  subst hmA
  have hnaB : ∀ (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ sessOff (S.pre x) (S.kS x) (S.idx x u) →
      cs.length = as.length → ((D.V.next x as cs).eval H).1 ≠ .acc := fun as cs h1 h2 h3 => by
    by_cases hp : as.length ≤ S.pre x
    · exact hnaA as cs h1 hp h3
    · exact S.next_mid H x as cs (by omega) (by omega) h3
  have hsplit : D.V.k x - S.pre x =
      (sessOff (S.pre x) (S.kS x) (S.idx x u) - S.pre x) + (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u)) := by
    omega
  rcases run_upto H x P ω r (sessOff (S.pre x) (S.kS x) (S.idx x u)) (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u))
      hnaB (sessOff (S.pre x) (S.kS x) (S.idx x u) - S.pre x) msA psA lsA (by omega) hmsA
      (fun _ => ⟨mA, by rw [hoA]; rfl⟩) with
    ⟨ms, ps, ls, hlen, hpref, hms, -, hB⟩ | ⟨ms, a, ps, ls, hlen, hms, hB⟩
  swap
  · exact S.runsTo_refused H x P ω r u hi ms a ps ls hlen hms fun r' hr' => by
      rw [hrun r', hA' r' fun t ht => hr' t (by omega), hsplit, hB r' hr']
  have htake : (ms.take (S.pre x)).map Prod.fst = msA.map Prod.fst := by
    obtain ⟨l, rfl⟩ := hpref
    rw [List.take_left' hlenA]
  have hpass : D.V.next x ((ms.take (S.pre x)).map Prod.fst) ((List.range (S.pre x)).map (coinSeq r)) =
      .ret (.msg mA) := by
    rw [htake, ← hlenA]
    exact hoA
  have hk' := S.k_stmt x (msA.map Prod.fst) ((List.range msA.length).map (coinSeq r)) mA (S.idx x u)
    (by simp [hlenA]) (by simp) hoA hi
  generalize hxS : S.stmt x (msA.map Prod.fst) (S.idx x u) = xS at hk'
  have hlm : (ms.map Prod.fst).length = sessOff (S.pre x) (S.kS x) (S.idx x u) := by simp [hlen]
  let emb : Fin (D.flock.V.k xS) → Fin (D.V.k x) := fun j =>
    ⟨sessOff (S.pre x) (S.kS x) (S.idx x u) + j.val, by have := j.isLt; omega⟩
  let P' : Prover Unit := ⟨fun _ hist _ => P.next x ((ms.map fun e => (e.1, msgOf e.2)) ++ hist) ω⟩
  have hW : ∀ r' : Fin (D.V.k x) → Coin,
      (∀ t : Fin (D.V.k x), t.val < sessOff (S.pre x) (S.kS x) (S.idx x u) → r' t = r t) →
      run H P D.V x ω r' = stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
        (D.flock.V.k xS + (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u) - D.flock.V.k xS))
        (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, ls, .rej⟩ := by
    intro r' hr'
    have e : D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u) =
        D.flock.V.k xS + (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u) - D.flock.V.k xS) := by
      omega
    rw [hrun r', hA' r' fun t ht => hr' t (by omega), hsplit, hB r' hr', ← e]
  have hv : ∀ r' : Fin (D.V.k x) → Coin, ∀ bs : List Msg, bs ≠ [] → bs.length ≤ D.flock.V.k xS →
      (∀ j, 0 < j → j < bs.length → ∃ m, ((D.flock.V.next xS (bs.take j)
        ((List.range (bs.take j).length).map (coinSeq fun i => r' (emb i)))).eval H).1 = .msg m) →
      (D.V.next x (ms.map Prod.fst ++ bs) ((List.range (ms.map Prod.fst ++ bs).length).map (coinSeq r'))).eval H =
        (markOut S.mark ((D.flock.V.next xS bs ((List.range bs.length).map (coinSeq fun i => r' (emb i)))).eval H).1,
          ((D.flock.V.next xS bs ((List.range bs.length).map (coinSeq fun i => r' (emb i)))).eval H).2) := by
    intro r' bs hbs hbl hund
    have hcs : ((List.range (ms.map Prod.fst ++ bs).length).map (coinSeq r')).drop
        (sessOff (S.pre x) (S.kS x) (S.idx x u)) = (List.range bs.length).map (coinSeq fun i => r' (emb i)) := by
      rw [List.length_append, hlm]
      exact coins_drop r' _ _ emb (fun _ => rfl) (by omega) hbl
    have htk : (ms.map Prod.fst ++ bs).take (S.pre x) = msA.map Prod.fst := by
      rw [List.take_append_of_le_length (by omega), ← List.map_take, htake]
    have hdr : (ms.map Prod.fst ++ bs).drop (sessOff (S.pre x) (S.kS x) (S.idx x u)) = bs := List.drop_left' hlm
    have hbpos := List.length_pos_of_ne_nil hbs
    have := S.next_sess H x (ms.map Prod.fst ++ bs) ((List.range (ms.map Prod.fst ++ bs).length).map (coinSeq r'))
      (S.idx x u) hi (by rw [List.length_append, hlm]; omega) (by rw [List.length_append, hlm]; omega) (by simp)
      (by
        intro j hj hjl
        rw [List.length_append, hlm] at hjl
        rw [htk, hxS, hdr, hcs, ← List.map_take, List.take_range, min_eq_left (by omega)]
        obtain ⟨m, hm⟩ := hund j hj (by omega)
        rw [List.length_take, min_eq_left (by omega)] at hm
        exact ⟨m, hm⟩)
    rw [htk, hxS, hdr, hcs] at this
    exact this
  have hC := fun r' : Fin (D.V.k x) → Coin => stepWith_session H P P' x xS ω ()
    (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
    (fun as => D.flock.V.next xS as ((List.range as.length).map (coinSeq fun i => r' (emb i)))) S.mark
    (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ms ps ls (D.flock.V.k xS)
    (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u) - D.flock.V.k xS) (fun _ => rfl) (hv r')
    (fun _ => S.flock_ne H xS _ _) (D.flock.V.k xS) [] [] ⟨[], [], [], .rej⟩ (by simp)
    (fun j hj hjl => by simp at hjl; omega)
  have hbefore : ∀ t : Fin (D.V.k x), t.val < sessOff (S.pre x) (S.kS x) (S.idx x u) → t ∉ D.rounds x u :=
    fun t ht h => by
      rw [S.rounds_iff] at h
      omega
  refine ⟨xS, emb, P', fun _ _ h => Nat.add_lt_add_left h _, fun t => ?_, fun r' hr' hacc => ?_, fun r' hr' => ?_⟩
  · rw [S.rounds_iff]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨⟨t.val - sessOff (S.pre x) (S.kS x) (S.idx x u), by rw [hk']; omega⟩,
        Fin.ext (show sessOff (S.pre x) (S.kS x) (S.idx x u) + (t.val - sessOff (S.pre x) (S.kS x) (S.idx x u)) =
          t.val by omega)⟩
    · rintro ⟨j, rfl⟩
      have := lt_of_lt_of_eq j.isLt hk'
      exact ⟨Nat.le_add_right _ _, by show sessOff (S.pre x) (S.kS x) (S.idx x u) + j.val < _; omega⟩
  · rw [hW r' fun t ht => hr' t (hbefore t ht)] at hacc
    obtain ⟨_, _, -, -, hfwd⟩ := hC r'
    simp only [List.append_nil, List.map_nil] at hfwd
    rcases hfwd hacc with h | ⟨as', bs, hl', hdec', hbs, hbl, hb⟩
    · exact h
    · exfalso
      have hbpos := List.length_pos_of_ne_nil hbs
      have hle : (ms.map Prod.fst ++ as' ++ bs).length ≤ D.V.k x := by
        rw [List.length_append, List.length_append, hlm, hl']
        omega
      have hfl : (ms.map Prod.fst ++ as' ++ bs).length = D.V.k x := by
        by_contra hne
        exact S.next_mid H x _ _ (by rw [List.length_append, List.length_append, hlm]; omega) (by omega) (by simp) hb
      obtain ⟨j, hj0, hjK, hj⟩ := S.next_last H x _ _ hfl (by simp) hb (S.idx x u) hi
      have htk : (ms.map Prod.fst ++ as' ++ bs).take (S.pre x) = msA.map Prod.fst := by
        rw [List.append_assoc, List.take_append_of_le_length (by omega), ← List.map_take, htake]
      have hdr : ((ms.map Prod.fst ++ as' ++ bs).drop (sessOff (S.pre x) (S.kS x) (S.idx x u))).take j =
          as'.take j := by
        rw [List.append_assoc, List.drop_left' hlm, List.take_append_of_le_length (by omega)]
      have hcs : (((List.range (ms.map Prod.fst ++ as' ++ bs).length).map (coinSeq r')).drop
          (sessOff (S.pre x) (S.kS x) (S.idx x u))).take j = (List.range j).map (coinSeq fun i => r' (emb i)) := by
        have hsum : (ms.map Prod.fst ++ as' ++ bs).length =
            sessOff (S.pre x) (S.kS x) (S.idx x u) + (D.flock.V.k xS + bs.length) := by
          rw [List.length_append, List.length_append, hlm, hl']
          omega
        rw [hsum, List.range_add, List.map_append, List.drop_left' (by simp), List.map_map, ← List.map_take,
          List.take_range, min_eq_left (by omega)]
        apply List.map_congr_left
        intro i hi'
        rw [List.mem_range] at hi'
        have h1 : sessOff (S.pre x) (S.kS x) (S.idx x u) + i < D.V.k x := by omega
        have h2 : i < D.flock.V.k xS := by omega
        simp only [Function.comp, coinSeq, h1, h2, ↓reduceDIte]
        rfl
      rw [htk, hxS, hdr, hcs] at hj
      obtain ⟨m, hm⟩ := hdec' j hj0 (by omega)
      rw [List.length_take, min_eq_left (by omega)] at hm
      exact hj m hm
  · have hr'' : ∀ t : Fin (D.V.k x), t.val < sessOff (S.pre x) (S.kS x) (S.idx x u) → r' t = r t := fun t ht =>
      hr' t (hbefore t ht) fun j => show t.val < sessOff (S.pre x) (S.kS x) (S.idx x u) + j.val by omega
    obtain ⟨rest, lm, h1, h2, -⟩ := hC r'
    simp only [List.append_nil, List.map_nil] at h1 h2
    obtain ⟨new, hnew, hcut⟩ := cutSeen_stepWith H P' xS ()
      (fun as => D.flock.V.next xS as ((List.range as.length).map (coinSeq fun i => r' (emb i)))) S.mark
      (fun _ => S.flock_ne H xS _ _) (D.flock.V.k xS) [] [] ⟨[], [], [], .rej⟩
    simp only [List.nil_append] at hnew
    have hrunF : run H P' D.flock.V xS () (fun i => r' (emb i)) =
        stepWith H P' xS () (fun as => D.flock.V.next xS as ((List.range as.length).map
          (coinSeq fun i => r' (emb i)))) (D.flock.V.k xS) [] [] ⟨[], [], [], .rej⟩ := rfl
    rw [hW r' hr'', hrunF]
    refine ⟨?_, fun q hq => ?_⟩
    · simp only [Transcript.seen]
      rw [h1, List.append_assoc, S.session_pass x u ms _ _ ((List.range (S.pre x)).map (coinSeq r)) mA hlen hi hms
        (by simp) hpass, htake, hxS, ← hk', hnew, hcut rest]
    · rw [h2]
      simp only [List.mem_append]
      exact Or.inl (Or.inr hq)

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **An accepted run shows each unit's session as its statement's**: at a run that accepts, the session read off its
transcript for a unit of session `idx x u < nS x` is that session's statement at the prefix's prover messages, with the
transcript from the session's first round on cut at its rounds (`cutSeen`). -/
theorem SessionSeq.session_acc {D : Deploy Coin Col T} (S : SessionSeq D) (H : Msg → Msg) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (u : ℕ) (hi : S.idx x u < S.nS x)
    (hacc : (run H P D.V x ω r).out = .acc) :
    ∃ ms rest, (run H P D.V x ω r).msgs = ms ++ rest ∧ ms.length = sessOff (S.pre x) (S.kS x) (S.idx x u) ∧
      D.session x u (run H P D.V x ω r).seen =
        some (S.stmt x ((ms.take (S.pre x)).map Prod.fst) (S.idx x u), cutSeen S.mark (S.kS x (S.idx x u)) rest) := by
  have hoK : sessOff (S.pre x) (S.kS x) (S.idx x u) + S.kS x (S.idx x u) < D.V.k x := by
    have := sessOff_add_le (S.pre x) (S.kS x) hi
    rw [S.k_eq]
    omega
  have hpo : S.pre x ≤ sessOff (S.pre x) (S.kS x) (S.idx x u) := le_sessOff _ _ _
  have hrun : run H P D.V x ω r =
      stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r)))
        (S.pre x + (D.V.k x - S.pre x)) [] [] ⟨[], [], [], .rej⟩ := by
    rw [Nat.add_sub_of_le (by omega)]
    rfl
  have hnaA : ∀ (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ S.pre x → cs.length = as.length →
      ((D.V.next x as cs).eval H).1 ≠ .acc := fun as cs h1 h2 h3 => by
    obtain ⟨o, ho, hne⟩ := S.next_pre x as cs h1 h2 h3
    rw [ho]
    exact hne
  rcases run_upto H x P ω r (S.pre x) (D.V.k x - S.pre x) hnaA (S.pre x) [] [] [] (by simp) (by simp) (by simp) with
    ⟨msA, psA, lsA, hlenA, -, hmsA, hlastA, hA⟩ | ⟨ms, a, ps, ls, -, -, hA⟩
  swap
  · have h0 := hA r fun _ _ => rfl
    simp only [List.map_nil] at h0
    rw [hrun, h0] at hacc
    simp at hacc
  have hA0 := hA r fun _ _ => rfl
  simp only [List.map_nil] at hA0
  have hAne : msA ≠ [] := by
    rintro rfl
    have := S.pre_pos x
    simp at hlenA
    omega
  obtain ⟨mA, hmA⟩ := hlastA hAne
  obtain ⟨oA, hoA, -⟩ := S.next_pre x (msA.map Prod.fst) ((List.range msA.length).map (coinSeq r))
    (by simpa using hAne) (by simp [hlenA]) (by simp)
  rw [hoA] at hmA
  simp only [OC.eval] at hmA
  subst hmA
  have hnaB : ∀ (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ sessOff (S.pre x) (S.kS x) (S.idx x u) →
      cs.length = as.length → ((D.V.next x as cs).eval H).1 ≠ .acc := fun as cs h1 h2 h3 => by
    by_cases hp : as.length ≤ S.pre x
    · exact hnaA as cs h1 hp h3
    · exact S.next_mid H x as cs (by omega) (by omega) h3
  have hsplit : D.V.k x - S.pre x =
      (sessOff (S.pre x) (S.kS x) (S.idx x u) - S.pre x) + (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u)) := by
    omega
  rcases run_upto H x P ω r (sessOff (S.pre x) (S.kS x) (S.idx x u)) (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u))
      hnaB (sessOff (S.pre x) (S.kS x) (S.idx x u) - S.pre x) msA psA lsA (by omega) hmsA
      (fun _ => ⟨mA, by rw [hoA]; rfl⟩) with
    ⟨ms, ps, ls, hlen, hpref, hms, -, hB⟩ | ⟨ms, a, ps, ls, -, -, hB⟩
  swap
  · rw [hrun, hA0, hsplit, hB r fun _ _ => rfl] at hacc
    simp at hacc
  have htake : (ms.take (S.pre x)).map Prod.fst = msA.map Prod.fst := by
    obtain ⟨l, rfl⟩ := hpref
    rw [List.take_left' hlenA]
  have hpass : D.V.next x ((ms.take (S.pre x)).map Prod.fst) ((List.range (S.pre x)).map (coinSeq r)) =
      .ret (.msg mA) := by
    rw [htake, ← hlenA]
    exact hoA
  have hW : run H P D.V x ω r = stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r)))
      (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u)) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst)
      ⟨ms, ps, ls, .rej⟩ := by
    rw [hrun, hA0, hsplit, hB r fun _ _ => rfl]
  obtain ⟨rest, lm, h1, -⟩ := stepWith_ext H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r)))
    (D.V.k x - sessOff (S.pre x) (S.kS x) (S.idx x u)) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst)
    ⟨ms, ps, ls, .rej⟩
  refine ⟨ms, rest, by rw [hW]; exact h1, hlen, ?_⟩
  rw [hW]
  simp only [Transcript.seen]
  rw [h1]
  exact S.session_pass x u ms rest _ ((List.range (S.pre x)).map (coinSeq r)) mA hlen hi hms (by simp) hpass

end Claims.Sampling
