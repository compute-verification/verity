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
  offset: the inner rounds' messages are the judged run's (`innerPairs_run`) and the V* statements read off them are
  its;
* at each round of session `s` the compiled verifier's step is the outer verifier's (`ract_eq`, `next_sess`): its
  numbers and `decided` checks read only the inner rounds and the rounds before the session, where the judged run,
  which accepts, passed them (`run_round`);
* the session's answers are the outer verifier's as `sessOut` gives them, which `D.session` reads back (`sim`): an
  acceptance at the session's last round, `.msg []` before the last session, as `.acc` (the outer verifier's other
  answers are coins' bytes, never empty, `coin_bytes_ne_nil`), and a session that ends on a coin refused at the next
  session's first round (`decided`);
* the log inclusion: each of the session's oracle calls is its outer verifier's, which `next_sess` logs as the
  compiled verifier's.

At an oracle where the outer verifier can accept before its last round, the compiled verifier refuses that round while
the session's own run accepts, so `sessionRuns` takes `OuterAtEnd`, which `nextO_acc_length` gives wherever Flock's
setup at `H` has its recursive setup's rounds (`hr`).
-/

namespace RecDeploy

open Claims (Msg Out Seen Trace Prover stepWith run coinSeq)
open FlockDeploy (Coin)

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

/-! ## Coins -/

theorem coinSeq_fork_lt {C : Type} [Inhabited C] {k : ℕ} (off : ℕ) (r d : Fin k → C) {i : ℕ} (h : i < off) :
    coinSeq (Claims.Recursion.fork off r d) i = coinSeq r i := by
  by_cases hk : i < k <;> simp [coinSeq, Claims.Recursion.fork, hk, h]

theorem coinSeq_fork_ge {C : Type} [Inhabited C] {k : ℕ} (off : ℕ) (r d : Fin k → C) {i : ℕ} (h : off ≤ i) :
    coinSeq (Claims.Recursion.fork off r d) i = coinSeq d i := by
  by_cases hk : i < k <;> simp [coinSeq, Claims.Recursion.fork, hk, Nat.not_lt.2 h]

theorem coinSeq_sessCoins {C : Type} [Inhabited C] {k : ℕ} (o : ℕ) (d : Fin k → C) (n : ℕ) {i : ℕ} (h : i < n) :
    coinSeq (sessCoins o d n) i = coinSeq d (o + i) := by
  simp [coinSeq, sessCoins, h]

theorem range_map_take {α : Type} (f : ℕ → α) {a b : ℕ} (h : a ≤ b) :
    ((List.range b).map f).take a = (List.range a).map f := by
  rw [← List.map_take, List.take_range, Nat.min_eq_left h]

theorem range_map_drop {α : Type} (f : ℕ → α) (a b : ℕ) :
    ((List.range (a + b)).map f).drop a = (List.range b).map fun i => f (a + i) := by
  rw [List.range_add, List.map_append, List.drop_left' (by simp), List.map_map]
  rfl

/-- **The coins from round `off` on of a fork at `off`** are the session's coins (`sessCoins`) at the second draw. -/
theorem fork_drop_coins {k : ℕ} (off : ℕ) (r d : Fin k → Coin) {b n : ℕ} (hb : b ≤ n) :
    ((List.range (off + b)).map (coinSeq (Claims.Recursion.fork off r d))).drop off =
      (List.range b).map (coinSeq (sessCoins off d n)) := by
  rw [range_map_drop]
  refine List.map_congr_left fun i hi => ?_
  rw [List.mem_range] at hi
  rw [coinSeq_fork_ge off r d (Nat.le_add_right _ _), coinSeq_sessCoins off d n (by omega)]

/-- **The coins before round `off` of a fork at `off`** are the first draw's. -/
theorem fork_take_coins {k : ℕ} (off : ℕ) (r d : Fin k → Coin) {m : ℕ} (hm : m ≤ off) :
    (List.range m).map (coinSeq (Claims.Recursion.fork off r d)) = (List.range m).map (coinSeq r) :=
  List.map_congr_left fun i hi => coinSeq_fork_lt off r d (by rw [List.mem_range] at hi; omega)

theorem coin_bytes_ne_nil (c : Coin) : c.bytes ≠ [] := by
  intro h
  rw [FlockDeploy.Coin.bytes, List.flatMap_eq_nil_iff] at h
  have := congrArg List.length (h 0 (List.mem_finRange 0))
  rw [FlockSoundness.Refine.toBytes_length] at this
  cases this

/-! ## The outer verifier's answers -/

theorem VO_k_pos (xO : Msg) : 0 < VO.k xO := by
  show 0 < (match setupO xO with | some (_, spec, sts) => 2 + FlockDeploy.rounds true spec sts | none => 1)
  split
  · exact Nat.add_pos_left Nat.two_pos _
  · exact Nat.one_pos

/-- **The outer verifier answers a message only at a coin**, with the coin's bytes. -/
theorem nextO_msg (H : Msg → Msg) {xO : Msg} {as : List Msg} {cs : List Coin} {m : Msg}
    (h : ((nextO xO as cs).eval H).1 = .msg m) : ∃ c, actO xO as cs = .coin c ∧ m = c.bytes := by
  cases ha : actO xO as cs with
  | rej => rw [nextO_rej H ha] at h; cases h
  | coin c =>
    rw [nextO_coin H ha] at h
    exact ⟨c, rfl, by simpa using h.symm⟩
  | decide s spec sts link c₁ mids f =>
    rcases nextO_decide_out H ha with ho | ho <;> rw [ho] at h <;> cases h

theorem nextO_msg_ne_nil (H : Msg → Msg) {xO : Msg} {as : List Msg} {cs : List Coin} {m : Msg}
    (h : ((nextO xO as cs).eval H).1 = .msg m) : m ≠ [] := by
  obtain ⟨c, -, hm⟩ := nextO_msg H h
  rw [hm]
  exact coin_bytes_ne_nil c

/-- A session that the outer verifier answers with a message has not decided. -/
theorem decided_of_msg (H : Msg → Msg) {xO : Msg} {as : List Msg} {cs : List Coin} {m : Msg}
    (h : ((nextO xO as cs).eval H).1 = .msg m) : decided xO as cs = false := by
  obtain ⟨c, ha, -⟩ := nextO_msg H h
  unfold decided
  rw [ha]

/-! ## The compiled verifier's step -/

theorem locate_eq : ∀ (ks : List ℕ) {s j : ℕ}, s < ks.length → j < ks.getD s 0 →
    locate ks ((ks.take s).sum + j) = some (s, j)
  | [], _, _, h, _ => by simp at h
  | k :: ks, 0, j, _, hj => by
    simp only [List.getD_cons_zero] at hj
    simp [locate, hj]
  | k :: ks, s + 1, j, hs, hj => by
    simp only [List.length_cons] at hs
    simp only [List.getD_cons_succ] at hj
    have := locate_eq ks (by omega) hj
    rw [List.take_succ_cons, List.sum_cons]
    simp only [locate]
    split_ifs with h
    · omega
    · rw [show k + (ks.take s).sum + j - k = (ks.take s).sum + j by omega, this]
      rfl

theorem ract_inner {IV : Claims.Verifier Coin} {st : Stage} {x : Msg} {as : List Msg} {cs : List Coin} {xr : XR}
    {c : Coin} (hx : readX x = some xr) (hk : as.length ≤ IV.k (xIOf x)) (hc : cs.getLast? = some c) :
    ract IV st x as cs = .coin c := by
  unfold ract
  rw [hx]
  dsimp only
  rw [ite_eq_left hk, hc]

open Classical in
/-- **The compiled verifier's step past the inner rounds**, at the round `locate` places: part `s`'s numbers check,
then at a session's first round the session before it, then the session's round. -/
theorem ract_eq {IV : Claims.Verifier Coin} {st : Stage} {x : Msg} {as : List Msg} {cs : List Coin} {xr : XR}
    (hx : readX x = some xr) (hk : IV.k (xIOf x) < as.length) {s j : ℕ}
    (hl : locate ((vstar st x (innerPairs (IV.k (xIOf x)) as cs)).map VO.k) (as.length - 1 - IV.k (xIOf x)) =
      some (s, j)) :
    ract IV st x as cs =
      if numsOf (staged st x (innerPairs (IV.k (xIOf x)) as cs) s) ≠ nums st x s then RAct.rej
      else if j = 0 ∧ 0 < s ∧ decided ((vstar st x (innerPairs (IV.k (xIOf x)) as cs)).getD (s - 1) [])
          ((as.take (as.length - 1 - j)).drop (as.length - 1 - j -
            ((vstar st x (innerPairs (IV.k (xIOf x)) as cs)).map VO.k).getD (s - 1) 0))
          ((cs.take (as.length - 1 - j)).drop (as.length - 1 - j -
            ((vstar st x (innerPairs (IV.k (xIOf x)) as cs)).map VO.k).getD (s - 1) 0)) = false then RAct.rej
      else .sess s j ((vstar st x (innerPairs (IV.k (xIOf x)) as cs)).getD s []) (as.drop (as.length - 1 - j))
        (cs.drop (as.length - 1 - j)) := by
  unfold ract
  rw [hx]
  dsimp only
  rw [ite_eq_right (Nat.not_le.2 hk), hl]

theorem getD_map_k (xs : List Msg) {s : ℕ} (hs : s < xs.length) : (xs.map VO.k).getD s 0 = VO.k (xs.getD s []) := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hs]

theorem rounds_through (xs : List Msg) {s : ℕ} (hs : s < xs.length) :
    ((xs.take s).map VO.k).sum + VO.k (xs.getD s []) = ((xs.take (s + 1)).map VO.k).sum := by
  rw [List.take_add_one, List.map_append, List.sum_append]
  simp [List.getElem?_eq_getElem hs, List.getD_eq_getElem?_getD]

theorem take_sum_le (l : List ℕ) (m : ℕ) : (l.take m).sum ≤ l.sum := by
  have := List.sum_take_add_sum_drop l m
  omega

/-! ## The compiled verifier's runs -/

/-- The compiled verifier's next message at coins `r`, as a run asks it. -/
noncomputable def vnOf (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {k : ℕ} (r : Fin k → Coin)
    (as : List Msg) : Claims.OC Out :=
  next IV st x as ((List.range as.length).map (coinSeq r))

/-- **A run's message at round `i`**: the prover's after the run's first `i` messages, answered by the verifier on
the prover's messages so far and the first `i + 1` coins. -/
theorem run_get {C : Type} [Inhabited C] (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (V' : Claims.Verifier C)
    (x : Msg) (ω : Ω) (r : Fin (V'.k x) → C) (i : ℕ) (h : i < (run H P V' x ω r).msgs.length) :
    ∃ new : List (Msg × Out), new = (run H P V' x ω r).msgs.take i ∧
      (run H P V' x ω r).msgs[i]? = some (((P.next x (new.map histOf) ω).eval H).1,
        ((V'.next x (new.map Prod.fst ++ [((P.next x (new.map histOf) ω).eval H).1])
          ((List.range (i + 1)).map (coinSeq r))).eval H).1) := by
  obtain ⟨new, hnl, hnew, hget⟩ := stepWith_get H P x ω
    (fun as => V'.next x as ((List.range as.length).map (coinSeq r))) i (V'.k x) [] [] ⟨[], [], [], .rej⟩
    (by simpa [Claims.run] using h)
  refine ⟨new, by simpa [Claims.run] using hnew, ?_⟩
  simpa [hnl, Claims.run] using hget

section Runs

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

theorem run_eq :
    run H P (V IV st) x ω r = stepWith H P x ω (vnOf IV st x r) ((V IV st).k x) [] [] ⟨[], [], [], .rej⟩ := rfl

theorem run_acc_readX (h : (run H P (V IV st) x ω r).out = .acc) : ∃ xr, readX x = some xr := by
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ h
  obtain ⟨s, j, xO, as'', cs', hr, -⟩ := next_acc (IV := IV) (st := st) H ha
  exact ract_readX hr

theorem run_acc_length (h : (run H P (V IV st) x ω r).out = .acc) :
    (run H P (V IV st) x ω r).msgs.length = (V IV st).k x := by
  obtain ⟨as', hl, ha⟩ := stepWith_acc_length H P x ω (vnOf IV st x r) ((V IV st).k x) [] [] ⟨[], [], [], .rej⟩ rfl h
  rw [run_eq, ← hl]
  exact next_acc_length H ha

/-- **An inner round's answer is its coin's bytes.** -/
theorem run_inner {xr : XR} (hx : readX x = some xr) {i : ℕ} (hi : i < IV.k (xIOf x))
    (hl : i < (run H P (V IV st) x ω r).msgs.length) :
    ∃ a, (run H P (V IV st) x ω r).msgs[i]? = some (a, .msg (coinSeq r i).bytes) := by
  obtain ⟨new, hnew, hget⟩ := run_get H P (V IV st) x ω r i hl
  have hnl : new.length = i := by
    rw [hnew, List.length_take]
    omega
  have hc := ract_inner (IV := IV) (st := st) hx (as := new.map Prod.fst ++ [((P.next x (new.map histOf) ω).eval H).1])
    (cs := (List.range (i + 1)).map (coinSeq r)) (c := coinSeq r i)
    (by simp only [List.length_append, List.length_map, List.length_singleton, hnl]; omega)
    (by simp [List.range_succ])
  refine ⟨((P.next x (new.map histOf) ω).eval H).1, ?_⟩
  rw [hget]
  show some (_, ((next IV st x _ _).eval H).1) = _
  rw [next_coin H hc]

/-- **An accepted run's inner rounds are its inner messages with their coins' bytes**, so messages that extend its
first `off ≥ k_I` and coins that extend its first `k_I` show them as `innerPairs`. -/
theorem innerPairs_run (h : (run H P (V IV st) x ω r).out = .acc) {off : ℕ} (hk : IV.k (xIOf x) ≤ off)
    (ho : off ≤ (V IV st).k x) (bs : List Msg) {cs : List Coin}
    (hc : cs.take (IV.k (xIOf x)) = (List.range (IV.k (xIOf x))).map (coinSeq r)) :
    innerPairs (IV.k (xIOf x)) (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs) cs =
      (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x)) := by
  obtain ⟨xr, hx⟩ := run_acc_readX H IV st x P ω r h
  have hlen := run_acc_length H IV st x P ω r h
  unfold innerPairs
  rw [hc, List.take_append_of_le_length (by simp only [List.length_map, List.length_take]; omega),
    ← List.map_take, List.take_take, Nat.min_eq_left hk, List.map_map]
  conv_rhs => rw [← List.zip_unzip ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x)))]
  rw [List.unzip_fst, List.unzip_snd]
  congr 1
  apply List.ext_getElem (by simp only [List.length_map, List.length_range, List.length_take]; omega)
  intro i h₁ h₂
  simp only [List.length_map, List.length_range] at h₁
  obtain ⟨a, ha⟩ := run_inner H IV st x P ω r hx h₁ (by omega)
  rw [List.getElem?_eq_getElem (by omega)] at ha
  simp [Option.some.inj ha]

open Classical in
/-- **An accepted run passed session `s`'s checks at its first round**: part `s`'s staged numbers are its numbers,
and the session before it ended on a decision. -/
theorem run_round (hacc : (run H P (V IV st) x ω r).out = .acc) {s off : ℕ} {xs : List Msg} (hs : s < nS x)
    (hxs : xs = vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x))))
    (hoff : off = IV.k (xIOf x) + ((xs.take s).map VO.k).sum) :
    numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x))) s) = nums st x s ∧
      (0 < s → decided (xs.getD (s - 1) [])
        ((((run H P (V IV st) x ω r).msgs.take off).map Prod.fst).drop (off - (xs.map VO.k).getD (s - 1) 0))
        (((List.range off).map (coinSeq r)).drop (off - (xs.map VO.k).getD (s - 1) 0)) = true) := by
  obtain ⟨xr, hx⟩ := run_acc_readX H IV st x P ω r hacc
  have hK := run_acc_length H IV st x P ω r hacc
  have hxl : xs.length = nS x := by rw [hxs, vstar_length]
  have hpos := VO_k_pos (xs.getD s [])
  have hthrough := rounds_through xs (s := s) (by omega)
  have hKeq : (V IV st).k x = IV.k (xIOf x) + (xs.map VO.k).sum := by
    show _ + _ = _
    rw [hxs, vstar_map_k]
  have hsum := take_sum_le (xs.map VO.k) (s + 1)
  rw [← List.map_take] at hsum
  have hlt : off < (V IV st).k x := by omega
  obtain ⟨new, hnew, hget⟩ := run_get H P (V IV st) x ω r off (by omega)
  subst hnew
  set R := run H P (V IV st) x ω r with hR
  set pa := ((P.next x ((R.msgs.take off).map histOf) ω).eval H).1 with hpa
  have hml : ((R.msgs.take off).map Prod.fst).length = off := by
    simp only [List.length_map, List.length_take]
    omega
  have hip : innerPairs (IV.k (xIOf x)) ((R.msgs.take off).map Prod.fst ++ [pa])
      ((List.range (off + 1)).map (coinSeq r)) = R.msgs.take (IV.k (xIOf x)) :=
    innerPairs_run H IV st x P ω r hacc (by omega) hlt.le [pa] (range_map_take _ (by omega))
  have hlen : ((R.msgs.take off).map Prod.fst ++ [pa]).length = off + 1 := by
    simp only [List.length_append, hml, List.length_singleton]
  have hl : locate ((vstar st x (innerPairs (IV.k (xIOf x)) ((R.msgs.take off).map Prod.fst ++ [pa])
      ((List.range (off + 1)).map (coinSeq r)))).map VO.k)
      (((R.msgs.take off).map Prod.fst ++ [pa]).length - 1 - IV.k (xIOf x)) = some (s, 0) := by
    rw [hip, ← hxs, hlen, show off + 1 - 1 - IV.k (xIOf x) = ((xs.map VO.k).take s).sum + 0 by
      rw [← List.map_take]; omega]
    exact locate_eq _ (by simp only [List.length_map]; omega) (by rw [getD_map_k xs (s := s) (by omega)]; exact hpos)
  have hnr : ract IV st x ((R.msgs.take off).map Prod.fst ++ [pa]) ((List.range (off + 1)).map (coinSeq r)) ≠
      .rej := by
    intro h0
    have h1 : R.msgs[off]? = some (pa, .rej) := by
      rw [hget]
      show some (_, ((next IV st x _ _).eval H).1) = _
      rw [next_rej H h0]
    have h2 : R.out = .rej :=
      stepWith_rej_out H P x ω (vnOf IV st x r) ((V IV st).k x) off [] [] ⟨[], [], [], .rej⟩ pa
        (by show R.msgs[0 + off]? = _; rwa [Nat.zero_add])
    rw [h2] at hacc
    cases hacc
  rw [ract_eq (st := st) hx (by omega) hl, hip, ← hxs, hlen, show off + 1 - 1 - 0 = off by omega] at hnr
  split_ifs at hnr with h1 h2
  · exact absurd rfl hnr
  · exact absurd rfl hnr
  · refine ⟨not_not.1 h1, fun hs0 => ?_⟩
    rw [List.take_left' hml, range_map_take _ (by omega)] at h2
    by_contra hd
    exact h2 ⟨rfl, hs0, by simpa using hd⟩

end Runs

/-! ## Each session's run -/

/-- **The outer verifier accepts only at its last round** at the oracle `H`, with one coin per message. -/
def OuterAtEnd (H : Msg → Msg) : Prop :=
  ∀ (xO : Msg) (as : List Msg) (cs : List Coin), cs.length = as.length → ((nextO xO as cs).eval H).1 = .acc →
    as.length = VO.k xO

theorem outerAtEnd_sha512 : OuterAtEnd FlockDeploy.sha512 := fun _ _ _ hcs h => nextO_acc_length_sha512 hcs h

open Classical in
/-- **Session `s` of an accepted run, run alone**: at its first round `off` and its V* statement `xO`, read off the
judged run's inner rounds, the outer verifier's run against `P` after the judged run's first `off` messages, at the
second draw's coins from `off` on, is `session`'s view of the fork at `off`, and its oracle calls are the fork's. -/
theorem session_core (H : Msg → Msg) (hO : OuterAtEnd H) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {s : ℕ}
    (hs : s < nS x) {Ω : Type} (P : Prover Ω) (ω : Ω) (r d : Fin ((V IV st).k x) → Coin)
    (hacc : (run H P (V IV st) x ω r).out = .acc) {off : ℕ} {xs : List Msg} {xO : Msg}
    (hxs : xs = vstar st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x))))
    (hoff : off = IV.k (xIOf x) + ((xs.take s).map VO.k).sum) (hxO : xO = xs.getD s []) :
    (run H (sessProver P x ω ((run H P (V IV st) x ω r).msgs.take off)) VO xO ()
        (sessCoins off d (VO.k xO))).seen =
      session IV st x s (run H P (V IV st) x ω (Claims.Recursion.fork off r d)).seen ∧
    ∀ e ∈ (run H (sessProver P x ω ((run H P (V IV st) x ω r).msgs.take off)) VO xO ()
        (sessCoins off d (VO.k xO))).vLog,
      e ∈ (run H P (V IV st) x ω (Claims.Recursion.fork off r d)).vLog := by
  obtain ⟨xr, hx⟩ := run_acc_readX H IV st x P ω r hacc
  have hK := run_acc_length H IV st x P ω r hacc
  obtain ⟨hnum, hdec⟩ := run_round H IV st x P ω r hacc hs hxs hoff
  have hxl : xs.length = nS x := by rw [hxs, vstar_length]
  have hpos := VO_k_pos xO
  have hthrough := rounds_through xs (s := s) (by omega)
  rw [← hxO] at hthrough
  have hKeq : (V IV st).k x = IV.k (xIOf x) + (xs.map VO.k).sum := by
    show _ + _ = _
    rw [hxs, vstar_map_k]
  have hsum := take_sum_le (xs.map VO.k) (s + 1)
  rw [← List.map_take] at hsum
  have hend : s + 1 = nS x → off + VO.k xO = (V IV st).k x := by
    intro h
    rw [hKeq, hoff, Nat.add_assoc, hthrough, h, ← hxl, List.take_length]
  have hkoff : IV.k (xIOf x) ≤ off := by omega
  have hml : (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst).length = off := by
    simp only [List.length_map, List.length_take]
    omega
  have hM : ((run H P (V IV st) x ω r).msgs.take off).length = off := by
    rw [List.length_take]
    omega
  have hip : ∀ (bs : List Msg) (cs : List Coin),
      cs.take (IV.k (xIOf x)) = (List.range (IV.k (xIOf x))).map (coinSeq r) →
      innerPairs (IV.k (xIOf x)) (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs) cs =
        (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x)) :=
    fun bs cs hc => innerPairs_run H IV st x P ω r hacc hkoff (by omega) bs hc
  have hcI : ∀ b, ((List.range (off + b)).map (coinSeq (Claims.Recursion.fork off r d))).take (IV.k (xIOf x)) =
      (List.range (IV.k (xIOf x))).map (coinSeq r) := fun b => by
    rw [range_map_take _ (by omega), fork_take_coins off r d hkoff]
  have hag := stepWith_agree H P x ω (vnOf IV st x r) (vnOf IV st x (Claims.Recursion.fork off r d))
    ((V IV st).k x) off [] [] ⟨[], [], [], .rej⟩ ⟨[], [], [], .rej⟩ rfl (fun bs hb => by
      simp only [List.length_nil, Nat.zero_add] at hb
      simp only [vnOf]
      rw [fork_take_coins off r d hb])
  simp only [List.length_nil, Nat.zero_add] at hag
  rw [← run_eq H IV st x P ω r, ← run_eq H IV st x P ω (Claims.Recursion.fork off r d)] at hag
  obtain ⟨hpre, hoffF⟩ := hag
  have hltF : off < (run H P (V IV st) x ω (Claims.Recursion.fork off r d)).msgs.length := hoffF.1 (by omega)
  have hpreI : (run H P (V IV st) x ω (Claims.Recursion.fork off r d)).msgs.take (IV.k (xIOf x)) =
      (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x)) := by
    have := congrArg (List.take (IV.k (xIOf x))) hpre
    rw [List.take_take, List.take_take, Nat.min_eq_left hkoff] at this
    exact this.symm
  obtain ⟨t', -, ht'm, -, hRFeq⟩ := stepWith_reach H P x ω (vnOf IV st x (Claims.Recursion.fork off r d)) off
    ((V IV st).k x) [] [] ⟨[], [], [], .rej⟩ (by show 0 + off < _; rw [Nat.zero_add]; exact hltF)
  simp only [List.length_nil, List.drop_zero, List.nil_append] at ht'm hRFeq
  rw [← run_eq H IV st x P ω (Claims.Recursion.fork off r d)] at ht'm hRFeq
  rw [← hpre] at ht'm hRFeq
  obtain ⟨e, he⟩ : ∃ e, (V IV st).k x - off = VO.k xO + e := ⟨(V IV st).k x - off - VO.k xO, by omega⟩
  rw [he] at hRFeq
  have hV : ∀ bs : List Msg, 0 < bs.length → bs.length ≤ VO.k xO →
      (vnOf IV st x (Claims.Recursion.fork off r d)
          (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs)).eval H =
        (sessOut (s + 1 == nS x) (bs.length == VO.k xO)
          ((nextO xO bs ((List.range bs.length).map (coinSeq (sessCoins off d (VO.k xO))))).eval H).1,
          ((nextO xO bs ((List.range bs.length).map (coinSeq (sessCoins off d (VO.k xO))))).eval H).2) := by
    intro bs hb0 hbn
    have hlen : (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs).length = off + bs.length := by
      rw [List.length_append, hml]
    have hk' : IV.k (xIOf x) < (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs).length := by
      rw [hlen]
      omega
    have hl : locate ((vstar st x (innerPairs (IV.k (xIOf x))
        (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs)
        ((List.range (off + bs.length)).map (coinSeq (Claims.Recursion.fork off r d))))).map VO.k)
        ((((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs).length - 1 - IV.k (xIOf x)) =
        some (s, bs.length - 1) := by
      rw [hip bs _ (hcI bs.length), ← hxs, hlen,
        show off + bs.length - 1 - IV.k (xIOf x) = ((xs.map VO.k).take s).sum + (bs.length - 1) by
          rw [← List.map_take]; omega]
      exact locate_eq _ (by simp only [List.length_map]; omega)
        (by rw [getD_map_k xs (s := s) (by omega), ← hxO]; omega)
    have hract : ract IV st x (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs)
        ((List.range (off + bs.length)).map (coinSeq (Claims.Recursion.fork off r d))) =
        .sess s (bs.length - 1) xO bs ((List.range bs.length).map (coinSeq (sessCoins off d (VO.k xO)))) := by
      rw [ract_eq (st := st) hx hk' hl, hip bs _ (hcI bs.length), ← hxs, hlen,
        show off + bs.length - 1 - (bs.length - 1) = off by omega, ← hxO, List.take_left' hml,
        List.drop_left' hml, range_map_take (coinSeq (Claims.Recursion.fork off r d)) (Nat.le_add_right off bs.length),
        fork_take_coins off r d (le_refl off), fork_drop_coins off r d hbn]
      split_ifs with h1 h2
      · exact absurd hnum h1
      · have := hdec h2.2.1
        rw [h2.2.2] at this
        exact absurd this Bool.false_ne_true
      · rfl
    rw [vnOf, hlen, next_sess H hract, show bs.length - 1 + 1 = bs.length by omega]
  have hnext : 0 < e → ∀ (bs : List Msg) (a m : Msg), bs.length = VO.k xO →
      ((nextO xO bs ((List.range bs.length).map (coinSeq (sessCoins off d (VO.k xO))))).eval H).1 = .msg m →
      ((vnOf IV st x (Claims.Recursion.fork off r d)
        (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs ++ [a])).eval H).1 = .rej := by
    intro he0 bs a m hbn hm
    have hs1 : s + 1 < nS x := by
      rcases Nat.lt_or_ge (s + 1) (nS x) with h | h
      · exact h
      · have := hend (by omega)
        omega
    have hml2 : (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ bs).length = off + VO.k xO := by
      rw [List.length_append, hml, hbn]
    have hlen : (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ (bs ++ [a])).length =
        off + (VO.k xO + 1) := by
      rw [← List.append_assoc, List.length_append, hml2, List.length_singleton, Nat.add_assoc]
    have hk' : IV.k (xIOf x) < (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ (bs ++ [a])).length := by
      rw [hlen]
      omega
    have hpos1 := VO_k_pos (xs.getD (s + 1) [])
    have hl : locate ((vstar st x (innerPairs (IV.k (xIOf x))
        (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ (bs ++ [a]))
        ((List.range (off + (VO.k xO + 1))).map (coinSeq (Claims.Recursion.fork off r d))))).map VO.k)
        ((((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ (bs ++ [a])).length - 1 - IV.k (xIOf x)) =
        some (s + 1, 0) := by
      rw [hip (bs ++ [a]) _ (hcI (VO.k xO + 1)), ← hxs, hlen,
        show off + (VO.k xO + 1) - 1 - IV.k (xIOf x) = ((xs.map VO.k).take (s + 1)).sum + 0 by
          rw [← List.map_take]; omega]
      exact locate_eq _ (by simp only [List.length_map]; omega)
        (by rw [getD_map_k xs (s := s + 1) (by omega)]; exact hpos1)
    rw [hbn] at hm
    have hd := decided_of_msg H hm
    have hrej : ract IV st x (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst ++ (bs ++ [a]))
        ((List.range (off + (VO.k xO + 1))).map (coinSeq (Claims.Recursion.fork off r d))) = .rej := by
      rw [ract_eq (st := st) hx hk' hl, hip (bs ++ [a]) _ (hcI (VO.k xO + 1)), ← hxs, hlen,
        show off + (VO.k xO + 1) - 1 - 0 = off + VO.k xO by omega, show s + 1 - 1 = s by omega,
        getD_map_k xs (s := s) (by omega), ← hxO, show off + VO.k xO - VO.k xO = off by omega,
        ← List.append_assoc, List.take_left' hml2, List.drop_left' hml,
        range_map_take (coinSeq (Claims.Recursion.fork off r d)) (show off + VO.k xO ≤ off + (VO.k xO + 1) by omega),
        fork_drop_coins off r d (le_refl (VO.k xO))]
      split_ifs with h1 h2
      · rfl
      · rfl
      · exact absurd ⟨rfl, Nat.succ_pos s, hd⟩ h2
    rw [List.append_assoc, vnOf, hlen, next_rej H hrej]
  have hacc' : ∀ bs, ((nextO xO bs ((List.range bs.length).map (coinSeq (sessCoins off d (VO.k xO))))).eval H).1 =
      .acc → bs.length = VO.k xO := fun bs h => hO xO bs _ (by simp) h
  have hlast : (s + 1 == nS x) = true → e = 0 := fun h => by
    have := hend (beq_iff_eq.1 h)
    omega
  have key := sim H P x ω (sessProver P x ω ((run H P (V IV st) x ω r).msgs.take off)) xO
    (fun bs => nextO xO bs ((List.range bs.length).map (coinSeq (sessCoins off d (VO.k xO)))))
    (vnOf IV st x (Claims.Recursion.fork off r d)) (s + 1 == nS x) (VO.k xO) e
    (((run H P (V IV st) x ω r).msgs.take off).map histOf) (((run H P (V IV st) x ω r).msgs.take off).map Prod.fst)
    ((run H P (V IV st) x ω r).msgs.take off) (fun _ => rfl) hV hacc' (fun _ _ h => nextO_msg_ne_nil H h) hlast
    hnext (VO.k xO) [] [] ⟨[], [], [], .rej⟩ t' hpos (by simp) (by rw [ht'm]; simp) rfl rfl (by simp) (by simp)
  simp only [List.append_nil] at key
  rw [← hRFeq, hM] at key
  have hsess : session IV st x s (run H P (V IV st) x ω (Claims.Recursion.fork off r d)).seen =
      sview (s + 1 == nS x) off (VO.k xO) (run H P (V IV st) x ω (Claims.Recursion.fork off r d)).seen := by
    show sview (s + 1 == nS x) (IV.k (xIOf x) + (((vstar st x ((run H P (V IV st) x ω
        (Claims.Recursion.fork off r d)).msgs.take (IV.k (xIOf x)))).take s).map VO.k).sum)
      (VO.k ((vstar st x ((run H P (V IV st) x ω (Claims.Recursion.fork off r d)).msgs.take
        (IV.k (xIOf x)))).getD s [])) (run H P (V IV st) x ω (Claims.Recursion.fork off r d)).seen = _
    rw [hpreI, ← hxs, ← hoff, ← hxO]
  rw [hsess]
  exact key

/-- **`SessionRuns` at the compiled verifier** and the outer record, at every oracle where the outer verifier accepts
only at its last round. -/
theorem sessionRuns {W : Type} (Hr H : Msg → Msg) (hO : OuterAtEnd H) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (st : Stage) (rd : Readers) (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    SessionRuns H (deployRec Hr I.V ivScope st rd) I (deployZkRec rel) :=
  fun x s _ P ω r d hacc => session_core H hO I.V st x s.2 P ω r d hacc rfl rfl rfl

/-- **`SessionRuns` at SHA-512.** -/
theorem sessionRuns_sha512 {W : Type} (Hr : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (st : Stage) (rd : Readers) (rel : Flock.Readable.Parsed → Flock.Readable.Relation) :
    SessionRuns FlockDeploy.sha512 (deployRec Hr I.V ivScope st rd) I (deployZkRec rel) :=
  sessionRuns Hr FlockDeploy.sha512 outerAtEnd_sha512 I ivScope st rd rel

end RecDeploy
