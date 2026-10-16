import Proofs.Flock.Claims.Recursion.DeployFacts

/-!
# The compiled verifier runs each V* session as the outer verifier (`SessionRuns`)

`SessionRuns` is the recursion analysis's code fact that the compiled verifier runs session `s` as the outer Flock's
verifier at its V* statement: on a run that accepts, a fork at the session's first round (`offO`) shows through
`D.session` the run of the prover's continuation (`sessP`) against the outer verifier at the fork's coins from that
round on, and that run's SHA-512 calls are among the compiled verifier's. Its definitions are restated here as the
analysis states them (`outMsg`, `sessProver`, `sessCoins`, `sessStmt`, `offO`, `sessP`, `SessionRuns`).

`sessionRuns` proves it at `deployRec H I.V …` and `deployZkRec rel`, at every oracle where the outer verifier
accepts only at its last round (`OuterAtEnd`), which SHA-512 is (`sessionRuns_sha512`, by `nextO_acc_length_sha512`):

* the fork's first `offO` rounds are the judged run's (`stepWith_agree`), so `offO` is the compiled verifier's session
  offset: the inner rounds' messages are the judged run's (`take_inner`) and the V* statements read off them are its;
* at each round of session `s` the compiled verifier's step is the outer verifier's (`ract_at`, `next_sess`): its
  numbers and `decided` checks read only the inner rounds and the rounds before the session, where the judged run,
  which accepts, passed them (`sessChecks`);
* the session's answers are the outer verifier's as `sessOut` gives them, which `D.session` reads back (`sim`): an
  acceptance at the session's last round, `.msg []` before the last session, as `.acc` (the outer verifier's other
  answers are coins' bytes, never empty, `coin_bytes_ne_nil`), and a session that ends on a coin refused at the next
  session's first round (`decided`);
* the log inclusion: each of the session's oracle calls is its outer verifier's, which `next_sess` logs as the
  compiled verifier's.

At an oracle where the outer verifier can accept before its last round, the compiled verifier refuses that round while
the session's own run accepts, so `OuterAtEnd` is the hypothesis `FlockDeploy.next_acc_length` takes (`hr`).
-/

namespace RecDeploy

open Claims (Msg Out Seen Trace Prover stepWith run coinSeq)

/-! ## The goal, as the recursion's analysis states it -/

/-- What a verifier's answer tells the prover: the message, or nothing once it decided. -/
def outMsg : Out → Msg
  | .msg m => m
  | _ => []

/-- **The prover of one session**: `P` at tape `ω`, after the messages `pre` of the rounds before the session. -/
def sessProver {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (pre : List (Msg × Out)) : Prover Unit :=
  ⟨fun _ hist _ => P.next x (pre.map (fun p => (p.1, outMsg p.2)) ++ hist) ω⟩

/-- The `n` coins from round `o` on. -/
def sessCoins {C : Type} [Inhabited C] {k : ℕ} (o : ℕ) (d : Fin k → C) (n : ℕ) : Fin n → C :=
  fun i => coinSeq d (o + i.val)

/-- Session `s`'s V* statement in the run at coins `r`, read off its inner rounds' messages. -/
def sessStmt {C W : Type} [Inhabited C] (H : Msg → Msg) (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (r : Fin (D.V.k x) → C) :
    Msg :=
  (D.vstar x ((run H P D.V x ω r).msgs.take (I.V.k (Claims.Recursion.xI D x)))).getD s []

/-- Session `s`'s first round in the run at coins `r`: the inner rounds, then the earlier sessions' rounds at the V*
statements read off the run's inner messages. -/
def offO {C W Col T : Type} [Inhabited C] (H : Msg → Msg) (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W) (DO : Claims.Flock.Deploy C Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (s : ℕ) (r : Fin (D.V.k x) → C) : ℕ :=
  I.V.k (Claims.Recursion.xI D x) +
    (((D.vstar x ((run H P D.V x ω r).msgs.take (I.V.k (Claims.Recursion.xI D x)))).take s).map DO.V.k).sum

/-- **Session `s`'s prover at the coins `r`**: `P` after the run's messages before the session's first round. -/
def sessP {C W Col T : Type} [Inhabited C] (H : Msg → Msg) (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W) (DO : Claims.Flock.Deploy C Col T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (s : ℕ) (r : Fin (D.V.k x) → C) : Prover Unit :=
  sessProver P x ω ((run H P D.V x ω r).msgs.take (offO H D I DO P x ω s r))

/-- **The compiled verifier runs session `s` as the outer Flock's verifier at its V* statement** (a code fact): on a run
that accepts, each fork at the session's first round shows through `D.session` the run of `P`'s continuation after the
run's earlier messages (`sessP`) against the outer verifier at the fork's coins from that round on, and the outer
verifier's SHA-512 calls in that run are among the compiled verifier's. -/
def SessionRuns {C W Col T : Type} [Inhabited C] (H : Msg → Msg) (D : Claims.Recursion.Deploy C)
    (I : Claims.Recursion.Inner C W) (DO : Claims.Flock.Deploy C Col T) : Prop :=
  ∀ (x : Msg) (s : Fin (D.nS x)) (Ω : Type) (P : Prover Ω) (ω : Ω) (r d : Fin (D.V.k x) → C),
    (run H P D.V x ω r).out = .acc →
    (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).seen =
      D.session x s (run H P D.V x ω (Claims.Recursion.fork (offO H D I DO P x ω s r) r d)).seen ∧
    ∀ e ∈ (run H (sessP H D I DO P x ω s r) DO.V (sessStmt H D I P x ω s r) ()
        (sessCoins (offO H D I DO P x ω s r) d _)).vLog,
      e ∈ (run H P D.V x ω (Claims.Recursion.fork (offO H D I DO P x ω s r) r d)).vLog

/-! ## Runs -/

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)

/-- The history entry of a round, as the prover sees it. -/
def histOf (p : Msg × Out) : Msg × Msg := (p.1, outMsg p.2)

theorem stepWith_succ_msg (vn : List Msg → Claims.OC Out) (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg)
    (t : Trace) {m : Msg} {lo : List (Msg × Msg)}
    (h : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H = (.msg m, lo)) :
    stepWith H P x ω vn (f + 1) hist as t =
      stepWith H P x ω vn f (hist ++ [(((P.next x hist ω).eval H).1, m)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo } := by
  simp only [stepWith, h]

theorem stepWith_succ_acc (vn : List Msg → Claims.OC Out) (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg)
    (t : Trace) {lo : List (Msg × Msg)} (h : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H = (.acc, lo)) :
    stepWith H P x ω vn (f + 1) hist as t =
      { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .acc)],
               pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo, out := .acc } := by
  simp only [stepWith, h]

theorem stepWith_succ_rej (vn : List Msg → Claims.OC Out) (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg)
    (t : Trace) {lo : List (Msg × Msg)} (h : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H = (.rej, lo)) :
    stepWith H P x ω vn (f + 1) hist as t =
      { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .rej)],
               pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo, out := .rej } := by
  simp only [stepWith, h]

/-- **A run only appends** to its state's messages and verifier log, at most one message per round left. -/
theorem stepWith_mono (vn : List Msg → Claims.OC Out) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      t.msgs <+: (stepWith H P x ω vn f hist as t).msgs ∧ t.vLog <+: (stepWith H P x ω vn f hist as t).vLog ∧
        (stepWith H P x ω vn f hist as t).msgs.length ≤ t.msgs.length + f
  | 0, _, _, t => by simp [stepWith]
  | f + 1, hist, as, t => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg m =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb]
      obtain ⟨h1, h2, h3⟩ := stepWith_mono vn f (hist ++ [(((P.next x hist ω).eval H).1, m)])
        (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo }
      refine ⟨(List.prefix_append _ _).trans h1, (List.prefix_append _ _).trans h2, ?_⟩
      simp only [List.length_append, List.length_singleton] at h3
      omega
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb]
      exact ⟨List.prefix_append _ _, List.prefix_append _ _, by simp⟩
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb]
      exact ⟨List.prefix_append _ _, List.prefix_append _ _, by simp⟩

/-- A round left is a message more. -/
theorem stepWith_length_succ (vn : List Msg → Claims.OC Out) (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg)
    (t : Trace) : t.msgs.length < (stepWith H P x ω vn (f + 1) hist as t).msgs.length := by
  rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
  cases o with
  | msg m =>
    rw [stepWith_succ_msg H P x ω vn f hist as t hvb]
    have := (stepWith_mono H P x ω vn f (hist ++ [(((P.next x hist ω).eval H).1, m)])
      (as ++ [((P.next x hist ω).eval H).1])
      { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)],
               pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo }).1.length_le
    simp only [List.length_append, List.length_singleton] at this
    omega
  | acc => rw [stepWith_succ_acc H P x ω vn f hist as t hvb]; simp
  | rej => rw [stepWith_succ_rej H P x ω vn f hist as t hvb]; simp

/-- **A run that goes past round `m` reaches its state after `m` rounds**: the history and messages so far, its
messages its first `m` new ones, and `m` fewer rounds left. -/
theorem stepWith_reach (vn : List Msg → Claims.OC Out) :
    ∀ (m f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      t.msgs.length + m < (stepWith H P x ω vn f hist as t).msgs.length →
      ∃ t' : Trace, m ≤ f ∧
        t'.msgs = t.msgs ++ ((stepWith H P x ω vn f hist as t).msgs.drop t.msgs.length).take m ∧
        t.vLog <+: t'.vLog ∧
        stepWith H P x ω vn f hist as t =
          stepWith H P x ω vn (f - m)
            (hist ++ (((stepWith H P x ω vn f hist as t).msgs.drop t.msgs.length).take m).map histOf)
            (as ++ (((stepWith H P x ω vn f hist as t).msgs.drop t.msgs.length).take m).map Prod.fst) t'
  | 0, f, hist, as, t, _ => ⟨t, Nat.zero_le _, by simp, List.prefix_refl _, by simp⟩
  | m + 1, 0, hist, as, t, h => by simp [stepWith] at h
  | m + 1, f + 1, hist, as, t, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      set a := ((P.next x hist ω).eval H).1
      set t₁ : Trace :=
        { t with msgs := t.msgs ++ [(a, .msg mm)], pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
                 vLog := t.vLog ++ lo }
      have hl₁ : t₁.msgs.length = t.msgs.length + 1 := by simp [t₁]
      obtain ⟨t', hle, hm, hv, hR⟩ := stepWith_reach vn m f (hist ++ [(a, mm)]) (as ++ [a]) t₁ (by omega)
      obtain ⟨rest, hrest⟩ := (stepWith_mono H P x ω vn f (hist ++ [(a, mm)]) (as ++ [a]) t₁).1
      set R := stepWith H P x ω vn f (hist ++ [(a, mm)]) (as ++ [a]) t₁ with hRdef
      have hd : R.msgs.drop t.msgs.length = (a, .msg mm) :: R.msgs.drop t₁.msgs.length := by
        rw [← hrest]
        show (t.msgs ++ [(a, Out.msg mm)] ++ rest).drop t.msgs.length =
          (a, Out.msg mm) :: (t.msgs ++ [(a, Out.msg mm)] ++ rest).drop (t.msgs ++ [(a, Out.msg mm)]).length
        simp
      refine ⟨t', by omega, ?_, (List.prefix_append _ _).trans hv, ?_⟩
      · rw [hm, hd, List.take_succ_cons]
        show t.msgs ++ [(a, Out.msg mm)] ++ _ = _
        simp
      · rw [hd, Nat.add_sub_add_right]
        simpa [histOf, outMsg] using hR
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb] at h
      simp at h
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      simp at h

/-- **A run's message at round `i`**, past its state: the prover's message after the history so far, answered by the
verifier on the messages so far and it. -/
theorem stepWith_get (vn : List Msg → Claims.OC Out) (i f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace)
    (h : t.msgs.length + i < (stepWith H P x ω vn f hist as t).msgs.length) :
    ∃ new : List (Msg × Out), new.length = i ∧
      new = ((stepWith H P x ω vn f hist as t).msgs.drop t.msgs.length).take i ∧
      (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? =
        some (((P.next x (hist ++ new.map histOf) ω).eval H).1,
          ((vn (as ++ new.map Prod.fst ++ [((P.next x (hist ++ new.map histOf) ω).eval H).1])).eval H).1) := by
  obtain ⟨t', hle, hm, -, hR⟩ := stepWith_reach H P x ω vn i f hist as t h
  have hf := (stepWith_mono H P x ω vn f hist as t).2.2
  generalize hnew : ((stepWith H P x ω vn f hist as t).msgs.drop t.msgs.length).take i = new at hm hR
  have hlen : new.length = i := by
    rw [← hnew]
    simp only [List.length_take, List.length_drop]
    omega
  refine ⟨new, hlen, rfl, ?_⟩
  rw [hR]
  obtain ⟨k, hk⟩ : ∃ k, f - i = k + 1 := ⟨f - i - 1, by omega⟩
  rw [hk]
  have ht' : t'.msgs.length = t.msgs.length + i := by rw [hm]; simp [hlen]
  rcases hvb : (vn (as ++ new.map Prod.fst ++ [((P.next x (hist ++ new.map histOf) ω).eval H).1])).eval H
    with ⟨o, lo⟩
  cases o with
  | msg mm =>
    rw [stepWith_succ_msg H P x ω vn k _ _ t' hvb]
    obtain ⟨rest, hrest⟩ := (stepWith_mono H P x ω vn k _ _ _).1
    rw [← hrest, ← ht']
    simp
  | acc =>
    rw [stepWith_succ_acc H P x ω vn k _ _ t' hvb, ← ht']
    simp
  | rej =>
    rw [stepWith_succ_rej H P x ω vn k _ _ t' hvb, ← ht']
    simp

/-- **Two verifiers that answer alike on the first `m` rounds' messages give the same first `m` rounds**, and the
runs go past round `m` together. -/
theorem stepWith_agree (vn vn' : List Msg → Claims.OC Out) :
    ∀ (f m : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Trace), t.msgs = t'.msgs →
      (∀ bs : List Msg, bs.length ≤ as.length + m → ((vn bs).eval H).1 = ((vn' bs).eval H).1) →
      (stepWith H P x ω vn f hist as t).msgs.take (t.msgs.length + m) =
          (stepWith H P x ω vn' f hist as t').msgs.take (t.msgs.length + m) ∧
        (t.msgs.length + m < (stepWith H P x ω vn f hist as t).msgs.length ↔
          t.msgs.length + m < (stepWith H P x ω vn' f hist as t').msgs.length)
  | 0, m, hist, as, t, t', ht, _ => by simp [stepWith, ht]
  | f + 1, 0, hist, as, t, t', ht, _ => by
    obtain ⟨r₁, h₁⟩ := (stepWith_mono H P x ω vn (f + 1) hist as t).1
    obtain ⟨r₂, h₂⟩ := (stepWith_mono H P x ω vn' (f + 1) hist as t').1
    have l₁ := stepWith_length_succ H P x ω vn f hist as t
    have l₂ := stepWith_length_succ H P x ω vn' f hist as t'
    rw [ht] at l₁ ⊢
    refine ⟨?_, by simp only [Nat.add_zero]; omega⟩
    rw [← h₁, ← h₂, ← ht]
    simp
  | f + 1, m + 1, hist, as, t, t', ht, hag => by
    have he := hag (as ++ [((P.next x hist ω).eval H).1]) (by simp)
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    rcases hvb' : (vn' (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o', lo'⟩
    rw [hvb, hvb'] at he
    simp only at he
    subst he
    cases o with
    | msg mm =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb, stepWith_succ_msg H P x ω vn' f hist as t' hvb']
      have := stepWith_agree vn vn' f m (hist ++ [(((P.next x hist ω).eval H).1, mm)])
        (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo }
        { t' with msgs := t'.msgs ++ [(((P.next x hist ω).eval H).1, .msg mm)],
                  pLog := t'.pLog ++ ((P.next x hist ω).eval H).2, vLog := t'.vLog ++ lo' }
        (by simp [ht]) (fun bs hb => hag bs (by simp at hb; omega))
      simp only [List.length_append, List.length_singleton] at this
      rwa [Nat.add_right_comm, Nat.add_assoc] at this
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb, stepWith_succ_acc H P x ω vn' f hist as t' hvb']
      simp [ht]
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb, stepWith_succ_rej H P x ω vn' f hist as t' hvb']
      simp [ht]

/-- **An accepted run accepts at its last message**: its verifier accepts the run's messages. -/
theorem stepWith_acc_length (vn : List Msg → Claims.OC Out) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), as.length = t.msgs.length →
      (stepWith H P x ω vn f hist as t).out = .acc →
      ∃ as' : List Msg, as'.length = (stepWith H P x ω vn f hist as t).msgs.length ∧ ((vn as').eval H).1 = .acc
  | 0, _, _, _, _, h => by simp [stepWith] at h
  | f + 1, hist, as, t, hl, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      exact stepWith_acc_length vn f _ _ _ (by simp [hl]) h
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb]
      exact ⟨as ++ [((P.next x hist ω).eval H).1], by simp [hl], by rw [hvb]⟩
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb] at h
      cases h

/-- **A run that refuses a round ends refused.** -/
theorem stepWith_rej_out (vn : List Msg → Claims.OC Out) :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (a : Msg),
      (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .rej) →
      (stepWith H P x ω vn f hist as t).out = .rej
  | 0, i, _, _, t, a, h => by simp [stepWith] at h
  | f + 1, i, hist, as, t, a, h => by
    rcases hvb : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H with ⟨o, lo⟩
    cases o with
    | msg mm =>
      rw [stepWith_succ_msg H P x ω vn f hist as t hvb] at h ⊢
      cases i with
      | zero =>
        obtain ⟨rest, hrest⟩ := (stepWith_mono H P x ω vn f _ _ _).1
        rw [← hrest] at h
        simp at h
      | succ i =>
        exact stepWith_rej_out vn f i _ _ _ a (by simpa [Nat.add_assoc, Nat.add_comm 1 i] using h)
    | acc =>
      rw [stepWith_succ_acc H P x ω vn f hist as t hvb] at h
      cases i <;> simp at h
    | rej =>
      rw [stepWith_succ_rej H P x ω vn f hist as t hvb]

theorem stepWith_zero (vn : List Msg → Claims.OC Out) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) :
    stepWith H P x ω vn 0 hist as t = { t with out := .rej } := rfl

end Steps

/-! ## A session's run beside the compiled run -/

/-- **A session's view in a run's view** at its first round `off` and its `n` rounds, as `session` reads it. -/
def sview (last : Bool) (off n : ℕ) (v : Seen) : Seen :=
  ⟨fixLast last n ((v.msgs.drop off).take n),
    if v.out = .acc then .acc else segOut (fixLast last n ((v.msgs.drop off).take n))⟩

theorem fixLast_keep {last : Bool} {k : ℕ} (seg : List (Msg × Out)) (a : Msg) {o : Out} (ho : o ≠ .msg []) :
    fixLast last k (seg ++ [(a, o)]) = seg ++ [(a, o)] := by
  unfold fixLast
  split_ifs
  · rw [List.getLast?_concat]
    cases o with
    | msg m =>
      cases m with
      | nil => exact absurd rfl ho
      | cons b m => rfl
    | acc => rfl
    | rej => rfl
  · rfl

theorem fixLast_acc (seg : List (Msg × Out)) (a : Msg) {k : ℕ} (hk : seg.length + 1 = k) :
    fixLast false k (seg ++ [(a, .msg [])]) = seg ++ [(a, .acc)] := by
  simp [fixLast, hk]

theorem fixLast_true (k : ℕ) (seg : List (Msg × Out)) : fixLast true k seg = seg := by
  simp [fixLast]

theorem segOut_concat (seg : List (Msg × Out)) (a : Msg) (o : Out) :
    segOut (seg ++ [(a, o)]) = if o = .acc then .acc else .rej := by
  unfold segOut
  rw [List.getLast?_concat]
  cases o <;> simp

theorem view_end (M₀ seg : List (Msg × Out)) (p : Msg × Out) {n : ℕ} (hn : seg.length + 1 ≤ n) :
    ((M₀ ++ seg ++ [p]).drop M₀.length).take n = seg ++ [p] := by
  rw [List.append_assoc, List.drop_left]
  exact List.take_of_length_le (by simpa using hn)

theorem view_mid (M₀ seg : List (Msg × Out)) (p : Msg × Out) (rest : List (Msg × Out)) {n : ℕ}
    (hn : seg.length + 1 = n) : ((M₀ ++ seg ++ [p] ++ rest).drop M₀.length).take n = seg ++ [p] := by
  rw [List.append_assoc, List.append_assoc, List.drop_left, ← List.append_assoc]
  exact List.take_left' (by simpa using hn)

section Sim

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (PS : Prover Unit) (xO : Msg)
  (vS vF : List Msg → Claims.OC Out) (last : Bool) (n e : ℕ) (hist₀ : List (Msg × Msg)) (as₀ : List Msg)
  (M₀ : List (Msg × Out))

/-- **A session's run is the compiled run's from the session's first round on**: where the session's prover is the
compiled run's prover after its first rounds (`hP`), the compiled verifier answers each of the session's rounds as
`sessOut` gives the session verifier's answer (`hV`), the session verifier accepts only at its last round (`hacc`) and
answers no empty message (`hmsg`), and the compiled verifier refuses the round after a session that ended on a message
(`hnext`, with `hlast`: the last session has no round after it), the session's view is `sview` of the compiled run's,
and its oracle log is in the compiled run's. -/
theorem sim (hP : ∀ hs, PS.next xO hs () = P.next x (hist₀ ++ hs) ω)
    (hV : ∀ bs : List Msg, 0 < bs.length → bs.length ≤ n →
      (vF (as₀ ++ bs)).eval H = (sessOut last (bs.length == n) ((vS bs).eval H).1, ((vS bs).eval H).2))
    (hacc : ∀ bs, ((vS bs).eval H).1 = .acc → bs.length = n)
    (hmsg : ∀ bs m, ((vS bs).eval H).1 = .msg m → m ≠ [])
    (hlast : last = true → e = 0)
    (hnext : 0 < e → ∀ (bs : List Msg) (a m : Msg), bs.length = n → ((vS bs).eval H).1 = .msg m →
      ((vF (as₀ ++ bs ++ [a])).eval H).1 = .rej) :
    ∀ (f : ℕ) (hs : List (Msg × Msg)) (bs : List Msg) (tS tF : Trace), 0 < f → bs.length + f = n →
      tF.msgs = M₀ ++ tS.msgs → bs = tS.msgs.map Prod.fst → hs = tS.msgs.map histOf →
      (∀ p ∈ tS.msgs, ∃ m, p.2 = .msg m ∧ m ≠ []) → (∀ l ∈ tS.vLog, l ∈ tF.vLog) →
      (stepWith H PS xO () vS f hs bs tS).seen =
          sview last M₀.length n (stepWith H P x ω vF (f + e) (hist₀ ++ hs) (as₀ ++ bs) tF).seen ∧
        ∀ l ∈ (stepWith H PS xO () vS f hs bs tS).vLog,
          l ∈ (stepWith H P x ω vF (f + e) (hist₀ ++ hs) (as₀ ++ bs) tF).vLog := by
  intro f
  induction f with
  | zero => intro _ _ _ _ h; omega
  | succ f ih =>
    intro hs bs tS tF _ hn hF hbs hhs hall hlog
    have hbl : bs.length = tS.msgs.length := by rw [hbs, List.length_map]
    set a := ((P.next x (hist₀ ++ hs) ω).eval H).1 with ha
    have hpa : ((PS.next xO hs ()).eval H) = (P.next x (hist₀ ++ hs) ω).eval H := by rw [hP hs]
    rcases hvS : (vS (bs ++ [a])).eval H with ⟨o, lo⟩
    have hvF : (vF (as₀ ++ bs ++ [a])).eval H = (sessOut last ((bs ++ [a]).length == n) o, lo) := by
      rw [List.append_assoc, hV (bs ++ [a]) (by simp) (by simp; omega), hvS]
    have hvS' : (vS (bs ++ [((PS.next xO hs ()).eval H).1])).eval H = (o, lo) := by rw [hpa]; exact hvS
    rw [show f + 1 + e = (f + e) + 1 by omega]
    cases o with
    | msg m =>
      have hm := hmsg _ _ (by rw [hvS])
      rw [stepWith_succ_msg H PS xO () vS f hs bs tS hvS',
        stepWith_succ_msg H P x ω vF (f + e) (hist₀ ++ hs) (as₀ ++ bs) tF (by rw [hvF]; rfl)]
      simp only [hpa]
      cases f with
      | zero =>
        have hlen : tS.msgs.length + 1 = n := by omega
        rw [stepWith_zero, Nat.zero_add]
        cases e with
        | zero =>
          rw [stepWith_zero]
          refine ⟨?_, fun l hl => ?_⟩
          · simp only [Trace.seen, sview, hF]
            rw [view_end M₀ tS.msgs _ (n := n) (by omega), fixLast_keep _ _ (by simpa using hm), segOut_concat]
            simp
          · simp only [List.mem_append] at hl ⊢
            rcases hl with hl | hl
            · exact Or.inl (hlog l hl)
            · exact Or.inr hl
        | succ e =>
          have hr := hnext (by omega) (bs ++ [a]) ((P.next x (hist₀ ++ (hs ++ [(a, m)])) ω).eval H).1 m
            (by simp; omega) (by rw [hvS])
          rcases hvr : (vF (as₀ ++ bs ++ [a] ++
              [((P.next x (hist₀ ++ hs ++ [(a, m)]) ω).eval H).1])).eval H with ⟨o', lo'⟩
          have ho' : o' = .rej := by
            simp only [List.append_assoc] at hr hvr
            rw [hvr] at hr
            exact hr
          subst ho'
          rw [stepWith_succ_rej H P x ω vF e _ _ _ hvr]
          refine ⟨?_, fun l hl => ?_⟩
          · simp only [Trace.seen, sview, hF]
            rw [view_mid M₀ tS.msgs _ _ (n := n) (by omega), fixLast_keep _ _ (by simpa using hm), segOut_concat]
            simp
          · simp only [List.mem_append] at hl ⊢
            rcases hl with hl | hl
            · exact Or.inl (Or.inl (hlog l hl))
            · exact Or.inl (Or.inr hl)
      | succ f =>
        have := ih (hs ++ [(a, m)]) (bs ++ [a])
          { tS with msgs := tS.msgs ++ [(a, .msg m)], pLog := tS.pLog ++ ((P.next x (hist₀ ++ hs) ω).eval H).2,
                    vLog := tS.vLog ++ lo }
          { tF with msgs := tF.msgs ++ [(a, .msg m)], pLog := tF.pLog ++ ((P.next x (hist₀ ++ hs) ω).eval H).2,
                    vLog := tF.vLog ++ lo }
          (by omega) (by simp; omega) (by simp [hF]) (by simp [hbs]) (by simp [hhs, histOf, outMsg])
          (fun p hp => by
            simp only [List.mem_append, List.mem_singleton] at hp
            rcases hp with hp | rfl
            · exact hall p hp
            · exact ⟨m, rfl, hm⟩)
          (fun l hl => by
            simp only [List.mem_append] at hl ⊢
            rcases hl with hl | hl
            · exact Or.inl (hlog l hl)
            · exact Or.inr hl)
        simpa only [List.append_assoc] using this
    | acc =>
      have hlen : (bs ++ [a]).length = n := hacc _ (by rw [hvS])
      have hf : f = 0 := by simp at hlen; omega
      subst hf
      have hlen' : tS.msgs.length + 1 = n := by simp at hlen; omega
      rw [stepWith_succ_acc H PS xO () vS 0 hs bs tS hvS']
      simp only [hpa]
      cases last with
      | false =>
        rw [stepWith_succ_msg H P x ω vF (0 + e) (hist₀ ++ hs) (as₀ ++ bs) tF (m := []) (lo := lo)
          (by rw [hvF]; simp [sessOut, hlen])]
        obtain ⟨rest, hrest⟩ := (stepWith_mono H P x ω vF (0 + e) (hist₀ ++ hs ++ [(a, [])]) (as₀ ++ bs ++ [a])
          { tF with msgs := tF.msgs ++ [(a, .msg [])], pLog := tF.pLog ++ ((P.next x (hist₀ ++ hs) ω).eval H).2,
                    vLog := tF.vLog ++ lo }).1
        obtain ⟨restL, hrestL⟩ := (stepWith_mono H P x ω vF (0 + e) (hist₀ ++ hs ++ [(a, [])]) (as₀ ++ bs ++ [a])
          { tF with msgs := tF.msgs ++ [(a, .msg [])], pLog := tF.pLog ++ ((P.next x (hist₀ ++ hs) ω).eval H).2,
                    vLog := tF.vLog ++ lo }).2.1
        refine ⟨?_, fun l hl => ?_⟩
        · simp only [Trace.seen, sview]
          rw [← hrest]
          simp only [hF]
          rw [view_mid M₀ tS.msgs _ rest (n := n) (by omega), fixLast_acc _ _ hlen', segOut_concat]
          simp [ha]
        · rw [← hrestL]
          simp only [List.mem_append] at hl ⊢
          rcases hl with hl | hl
          · exact Or.inl (Or.inl (hlog l hl))
          · exact Or.inl (Or.inr hl)
      | true =>
        rw [stepWith_succ_acc H P x ω vF (0 + e) (hist₀ ++ hs) (as₀ ++ bs) tF (lo := lo)
          (by rw [hvF]; simp [sessOut, hlen])]
        refine ⟨?_, fun l hl => ?_⟩
        · simp only [Trace.seen, sview, hF]
          rw [view_end M₀ tS.msgs _ (n := n) (by omega), fixLast_true]
          simp
        · simp only [List.mem_append] at hl ⊢
          rcases hl with hl | hl
          · exact Or.inl (hlog l hl)
          · exact Or.inr hl
    | rej =>
      rw [stepWith_succ_rej H PS xO () vS f hs bs tS hvS',
        stepWith_succ_rej H P x ω vF (f + e) (hist₀ ++ hs) (as₀ ++ bs) tF (by rw [hvF]; rfl)]
      simp only [hpa]
      refine ⟨?_, fun l hl => ?_⟩
      · simp only [Trace.seen, sview, hF]
        rw [view_end M₀ tS.msgs _ (n := n) (by omega), fixLast_keep _ _ (by simp), segOut_concat]
        simp
      · simp only [List.mem_append] at hl ⊢
        rcases hl with hl | hl
        · exact Or.inl (hlog l hl)
        · exact Or.inr hl

end Sim

end RecDeploy
