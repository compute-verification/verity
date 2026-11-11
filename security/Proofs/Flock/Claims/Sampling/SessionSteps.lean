module

public import Proofs.Flock.Claims.Sampling.Session

/-!
# `SessionRuns` from the sampled verifier's step structure

`SessionSteps D`: the sampled verifier `D.V` answers its first `pre x` rounds itself, with no `H` call and never
accepting, and every later round is Flock's verifier `D.flock.V` at the session's statement, which the prefix's prover
messages name (`stmt`), on the later messages and coins. Where the prefix's last answer is a message, that statement has
the session's `kS x` rounds. Every unit's session is on the rounds after the prefix, and the session read off a run's
transcript is the statement and the transcript after the prefix where the prefix answered every round with a message,
and else a refused run: the run of `xR x`, a statement with `kS x` rounds at which Flock refuses the first message `[]`.

`sessionRuns_of_steps`: such a verifier runs its sessions as Flock runs (`SessionRuns`). At any coins, Flock's rounds are
the rounds after the prefix, in order; Flock's prover plays the sampled prover's messages after the prefix's transcript;
and the run decides what that Flock run decides, shows its transcript as the session, and makes its `H` calls.
-/

@[expose] public section

namespace Claims.Sampling

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The sampled verifier's step structure**: a prefix of `pre x` rounds the verifier answers with no `H` call and no
acceptance, then Flock's verifier at the statement the prefix's prover messages name (`stmt`), on the later messages and
coins; the prefix's last answer is a message only where that statement has `kS x` rounds; every unit's session rounds are
the rounds after the prefix; and the session read off a transcript is the statement and the transcript after the prefix
where the transcript shows the prefix answered with messages and the verifier answers its last round with a message at
some coins, and the refused run of `xR x` (`kS x` rounds, refusing the message `[]` at once) where the prefix refused. -/
structure SessionSteps (D : Deploy Coin Col T) where
  pre : Msg → ℕ
  kS : Msg → ℕ
  stmt : Msg → List Msg → Msg
  xR : Msg → Msg
  pre_pos : ∀ x, 0 < pre x
  k_eq : ∀ x, D.V.k x = pre x + kS x
  next_pre : ∀ x (as : List Msg) (cs : List Coin), as ≠ [] → as.length ≤ pre x → cs.length = as.length →
    ∃ o, D.V.next x as cs = .ret o ∧ o ≠ .acc
  next_post : ∀ x (as : List Msg) (cs : List Coin), pre x < as.length → cs.length = as.length →
    D.V.next x as cs = D.flock.V.next (stmt x (as.take (pre x))) (as.drop (pre x)) (cs.drop (pre x))
  k_stmt : ∀ x (as : List Msg) (cs : List Coin) m, as.length = pre x → cs.length = as.length →
    D.V.next x as cs = .ret (.msg m) → D.flock.V.k (stmt x as) = kS x
  rounds_iff : ∀ x u (t : Fin (D.V.k x)), t ∈ D.rounds x u ↔ pre x ≤ t.val
  k_xR : ∀ x, D.flock.V.k (xR x) = kS x
  kS_pos : ∀ x, 0 < kS x
  refuses : ∀ x (c : Coin), D.flock.V.next (xR x) [[]] [c] = .ret .rej
  session_pass : ∀ x u (ms : List (Msg × Out)) rest out (cs : List Coin) m, ms.length = pre x →
    cs.length = ms.length → (∀ e ∈ ms, ∃ m, e.2 = .msg m) → D.V.next x (ms.map Prod.fst) cs = .ret (.msg m) →
    D.session x u ⟨ms ++ rest, out⟩ = some (stmt x (ms.map Prod.fst), ⟨rest, out⟩)
  session_refused : ∀ x u (ms : List (Msg × Out)) a, ms.length < pre x → (∀ e ∈ ms, ∃ m, e.2 = .msg m) →
    D.session x u ⟨ms ++ [(a, .rej)], .rej⟩ = some (xR x, ⟨[([], .rej)], .rej⟩)

/-! ## A run's rounds -/

section Rounds

variable {Ω Ω' : Type} (H : Msg → Msg)

/-- **A round answered with a message and no `H` call** appends the prover's message and the answer, and continues. -/
theorem stepWith_msg (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (n : ℕ) (hist : List (Msg × Msg))
    (as : List Msg) (t : Transcript) (m : Msg) (hv : v (as ++ [((P.next x hist ω).eval H).1]) = .ret (.msg m)) :
    stepWith H P x ω v (n + 1) hist as t =
      stepWith H P x ω v n (hist ++ [(((P.next x hist ω).eval H).1, m)]) (as ++ [((P.next x hist ω).eval H).1])
        ⟨t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)], t.pLog ++ ((P.next x hist ω).eval H).2, t.vLog,
          t.out⟩ := by
  simp only [stepWith, hv, OC.eval, List.append_nil]

/-- **A round refused with no `H` call** appends the prover's message and the refusal, and ends the run. -/
theorem stepWith_rej (P : Prover Ω) (x : Msg) (ω : Ω) (v : List Msg → OC Out) (n : ℕ) (hist : List (Msg × Msg))
    (as : List Msg) (t : Transcript) (hv : v (as ++ [((P.next x hist ω).eval H).1]) = .ret .rej) :
    stepWith H P x ω v (n + 1) hist as t =
      ⟨t.msgs ++ [(((P.next x hist ω).eval H).1, .rej)], t.pLog ++ ((P.next x hist ω).eval H).2, t.vLog, .rej⟩ := by
  simp only [stepWith, hv, OC.eval, List.append_nil]

/-- **The rounds after a prefix**: rounds against `v`, which at the prefix's messages `as₀` followed by `bs` is `v'` at
`bs`, of a prover that after the prefix's history `hist₀` plays `P'`, are `P'`'s rounds against `v'` after the prefix's
transcript `m₀` and calls `p₀`, `l₀`. -/
theorem stepWith_shift (P : Prover Ω) (P' : Prover Ω') (x x' : Msg) (ω : Ω) (ω' : Ω') (v v' : List Msg → OC Out)
    (hist₀ : List (Msg × Msg)) (as₀ : List Msg) (m₀ : List (Msg × Out)) (p₀ l₀ : List (Msg × Msg))
    (hP : ∀ hist, P.next x (hist₀ ++ hist) ω = P'.next x' hist ω')
    (hv : ∀ bs : List Msg, bs ≠ [] → v (as₀ ++ bs) = v' bs) (n : ℕ) :
    ∀ (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript),
      stepWith H P x ω v n (hist₀ ++ hist) (as₀ ++ as) ⟨m₀ ++ t.msgs, p₀ ++ t.pLog, l₀ ++ t.vLog, t.out⟩ =
        ⟨m₀ ++ (stepWith H P' x' ω' v' n hist as t).msgs, p₀ ++ (stepWith H P' x' ω' v' n hist as t).pLog,
          l₀ ++ (stepWith H P' x' ω' v' n hist as t).vLog, (stepWith H P' x' ω' v' n hist as t).out⟩ := by
  induction n with
  | zero => intro hist as t; rfl
  | succ n ih =>
    intro hist as t
    have hv' : v (as₀ ++ as ++ [((P'.next x' hist ω').eval H).1]) = v' (as ++ [((P'.next x' hist ω').eval H).1]) := by
      rw [List.append_assoc]
      exact hv _ (by simp)
    simp only [stepWith, hP hist, hv']
    generalize OC.eval H (v' (as ++ [((P'.next x' hist ω').eval H).1])) = vb
    obtain ⟨o, l⟩ := vb
    cases o with
    | msg m =>
      have := ih (hist ++ [(((P'.next x' hist ω').eval H).1, m)]) (as ++ [((P'.next x' hist ω').eval H).1])
        ⟨t.msgs ++ [(((P'.next x' hist ω').eval H).1, .msg m)], t.pLog ++ ((P'.next x' hist ω').eval H).2,
          t.vLog ++ l, t.out⟩
      simpa only [List.append_assoc] using this
    | acc => simp only [List.append_assoc]
    | rej => simp only [List.append_assoc]

/-- **A one-round refusal**: a prover whose first message is `[]`, against a verifier that refuses it with no `H` call,
shows the refused run `[([], rej)]` in any positive number of rounds. -/
theorem stepWith_refused (x : Msg) (v : List Msg → OC Out) (hv : v [[]] = .ret .rej) (n : ℕ) (hn : 0 < n) :
    stepWith H (⟨fun _ _ _ => .ret []⟩ : Prover Unit) x () v n [] [] ⟨[], [], [], .rej⟩ =
      ⟨[([], .rej)], [], [], .rej⟩ := by
  obtain ⟨n, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  exact stepWith_rej H (⟨fun _ _ _ => .ret []⟩ : Prover Unit) x () v n [] [] ⟨[], [], [], .rej⟩ hv

end Rounds

omit [Fintype Coin] [Nonempty Coin] in
/-- **Coins that agree below `n`** are the same coins at up to `n` messages. -/
theorem coins_eq {k n m : ℕ} {r r' : Fin k → Coin} (h : ∀ t : Fin k, t.val < n → r' t = r t) (hm : m ≤ n) :
    (List.range m).map (coinSeq r') = (List.range m).map (coinSeq r) := by
  apply List.map_congr_left
  intro i hi
  rw [List.mem_range] at hi
  by_cases hk : i < k
  · simp only [coinSeq, hk, ↓reduceDIte]
    exact h ⟨i, hk⟩ (show i < n by omega)
  · simp only [coinSeq, hk, ↓reduceDIte]

namespace SessionSteps

variable {D : Deploy Coin Col T} (S : SessionSteps D)

/-- **The rounds after the prefix**, as the rounds of a Flock run with `kS x` of them. -/
def emb (x : Msg) {k : ℕ} (hk : k = S.kS x) (i : Fin k) : Fin (D.V.k x) :=
  ⟨S.pre x + i.val, by have := i.isLt; have := S.k_eq x; omega⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
theorem emb_strictMono (x : Msg) {k : ℕ} (hk : k = S.kS x) : StrictMono (S.emb x hk) :=
  fun _ _ hij => Nat.add_lt_add_left hij _

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- **A unit's session rounds are the rounds after the prefix**, each Flock round's once. -/
theorem mem_rounds_iff (x : Msg) (u : ℕ) {k : ℕ} (hk : k = S.kS x) (t : Fin (D.V.k x)) :
    t ∈ D.rounds x u ↔ ∃ i, S.emb x hk i = t := by
  rw [S.rounds_iff]
  constructor
  · intro ht
    have := t.isLt
    have := S.k_eq x
    exact ⟨⟨t.val - S.pre x, by omega⟩, Fin.ext (show S.pre x + (t.val - S.pre x) = t.val by omega)⟩
  · rintro ⟨i, rfl⟩
    exact Nat.le_add_right _ _

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **The prefix's rounds**: from a transcript `ms` of answered rounds of the prefix, `j` more rounds either answer every
prefix round, reaching a transcript `ms'` of all of it whose last round the verifier answers with a message at `r`, or
refuse one, ending the run; and both, with the transcript and the prover's calls, are the same at every coins `r'` that
agree with `r` below the prefix. -/
theorem run_prefix (H : Msg → Msg) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (j : ℕ) :
    ∀ (ms : List (Msg × Out)) (ps : List (Msg × Msg)), ms.length + j = S.pre x → (∀ e ∈ ms, ∃ m, e.2 = .msg m) →
      (ms ≠ [] → ∃ m, D.V.next x (ms.map Prod.fst) ((List.range ms.length).map (coinSeq r)) = .ret (.msg m)) →
      (∃ (ms' : List (Msg × Out)) (ps' : List (Msg × Msg)), ms'.length = S.pre x ∧
          (∀ e ∈ ms', ∃ m, e.2 = .msg m) ∧
          (∃ m, D.V.next x (ms'.map Prod.fst) ((List.range ms'.length).map (coinSeq r)) = .ret (.msg m)) ∧
          ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < S.pre x → r' t = r t) →
            stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (j + S.kS x)
                (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, [], .rej⟩ =
              stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (S.kS x)
                (ms'.map fun e => (e.1, msgOf e.2)) (ms'.map Prod.fst) ⟨ms', ps', [], .rej⟩) ∨
      (∃ (ms' : List (Msg × Out)) (a : Msg) (ps' : List (Msg × Msg)), ms'.length < S.pre x ∧
          (∀ e ∈ ms', ∃ m, e.2 = .msg m) ∧
          ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < S.pre x → r' t = r t) →
            stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (j + S.kS x)
                (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, [], .rej⟩ =
              ⟨ms' ++ [(a, .rej)], ps', [], .rej⟩) := by
  induction j with
  | zero =>
    intro ms ps hlen hms hlast
    have hne : ms ≠ [] := by
      rintro rfl
      have := S.pre_pos x
      simp at hlen
      omega
    exact Or.inl ⟨ms, ps, by simpa using hlen, hms, hlast hne, fun r' _ => by rw [Nat.zero_add]⟩
  | succ j ih =>
    intro ms ps hlen hms _
    have hle : (ms.map Prod.fst ++ [((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1]).length ≤
        S.pre x := by
      simp only [List.length_append, List.length_map, List.length_singleton]
      omega
    obtain ⟨o, ho, hacc⟩ := S.next_pre x
      (ms.map Prod.fst ++ [((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1])
      ((List.range (ms.map Prod.fst ++ [((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1]).length).map
        (coinSeq r)) (by simp) hle (by simp)
    have hco : ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < S.pre x → r' t = r t) →
        D.V.next x (ms.map Prod.fst ++ [((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1])
          ((List.range (ms.map Prod.fst ++
            [((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1]).length).map (coinSeq r')) = .ret o := by
      intro r' hr'
      rw [coins_eq hr' hle]
      exact ho
    have hsucc : j + 1 + S.kS x = (j + S.kS x) + 1 := by omega
    cases o with
    | msg m =>
      have hms' : ∀ e ∈ ms ++ [(((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1, Out.msg m)],
          ∃ m, e.2 = .msg m := by
        intro e he
        rcases List.mem_append.1 he with he | he
        · exact hms e he
        · rw [List.mem_singleton] at he
          exact ⟨m, by rw [he]⟩
      rcases ih (ms ++ [(((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1, .msg m)])
          (ps ++ ((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).2) (by simp; omega) hms'
          (fun _ => ⟨m, by simpa using ho⟩) with ⟨ms', ps', h1, h2, h3, h4⟩ | ⟨ms', a, ps', h1, h2, h4⟩
      · refine Or.inl ⟨ms', ps', h1, h2, h3, fun r' hr' => ?_⟩
        rw [hsucc, stepWith_msg H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
          (j + S.kS x) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, [], .rej⟩ m (hco r' hr'),
          ← h4 r' hr']
        simp [msgOf]
      · refine Or.inr ⟨ms', a, ps', h1, h2, fun r' hr' => ?_⟩
        rw [hsucc, stepWith_msg H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
          (j + S.kS x) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, [], .rej⟩ m (hco r' hr'),
          ← h4 r' hr']
        simp [msgOf]
    | rej =>
      refine Or.inr ⟨ms, ((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).1,
        ps ++ ((P.next x (ms.map fun e => (e.1, msgOf e.2)) ω).eval H).2, by omega, hms, fun r' hr' => ?_⟩
      rw [hsucc]
      exact stepWith_rej H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
        (j + S.kS x) (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ⟨ms, ps, [], .rej⟩ (hco r' hr')
    | acc => exact absurd rfl hacc

end SessionSteps

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **`SessionRuns` from the step structure**: at coins that check a unit, its session is Flock's run on the rounds after
the prefix, of the prover that plays the sampled prover's messages after the prefix's transcript (or, where the prefix
refused, of a prover whose first message Flock refuses at `xR x`), and it decides the run, is the session the run shows,
and makes only `H` calls the run makes. -/
theorem sessionRuns_of_steps {D : Deploy Coin Col T} (S : SessionSteps D) (H : Msg → Msg) : SessionRuns H D := by
  intro x Ω P ω r u _
  have hrun : ∀ r' : Fin (D.V.k x) → Coin, run H P D.V x ω r' =
      stepWith H P x ω (fun as => D.V.next x as ((List.range as.length).map (coinSeq r'))) (S.pre x + S.kS x) [] []
        ⟨[], [], [], .rej⟩ := fun r' => by
    rw [← S.k_eq]
    rfl
  rcases S.run_prefix H x P ω r (S.pre x) [] [] (by simp) (by simp) (by simp) with
    ⟨ms, ps, hlen, hms, ⟨m, hm⟩, hpre⟩ | ⟨ms, a, ps, hlen, hms, hpre⟩
  · have hk := S.k_stmt x (ms.map Prod.fst) _ m (by simpa using hlen) (by simp) hm
    let P' : Prover Unit := ⟨fun _ hist _ => P.next x (ms.map (fun e => (e.1, msgOf e.2)) ++ hist) ω⟩
    have hcomp : ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < S.pre x → r' t = r t) →
        run H P D.V x ω r' =
          ⟨ms ++ (run H P' D.flock.V (S.stmt x (ms.map Prod.fst)) () fun i => r' (S.emb x hk i)).msgs,
            ps ++ (run H P' D.flock.V (S.stmt x (ms.map Prod.fst)) () fun i => r' (S.emb x hk i)).pLog,
            (run H P' D.flock.V (S.stmt x (ms.map Prod.fst)) () fun i => r' (S.emb x hk i)).vLog,
            (run H P' D.flock.V (S.stmt x (ms.map Prod.fst)) () fun i => r' (S.emb x hk i)).out⟩ := by
      intro r' hr'
      have h0 := hpre r' hr'
      simp only [List.map_nil] at h0
      rw [hrun r', h0, ← hk]
      have hl : (ms.map Prod.fst).length = S.pre x := by simp [hlen]
      have := stepWith_shift H P P' x (S.stmt x (ms.map Prod.fst)) ω ()
        (fun as => D.V.next x as ((List.range as.length).map (coinSeq r')))
        (fun as => D.flock.V.next (S.stmt x (ms.map Prod.fst)) as
          ((List.range as.length).map (coinSeq fun i => r' (S.emb x hk i))))
        (ms.map fun e => (e.1, msgOf e.2)) (ms.map Prod.fst) ms ps [] (fun _ => rfl) (fun bs hbs => by
          have hlt : S.pre x < (ms.map Prod.fst ++ bs).length := by
            cases bs with
            | nil => exact absurd rfl hbs
            | cons b bs => simp [hlen]
          show D.V.next x (ms.map Prod.fst ++ bs) ((List.range (ms.map Prod.fst ++ bs).length).map (coinSeq r')) =
            D.flock.V.next (S.stmt x (ms.map Prod.fst)) bs
              ((List.range bs.length).map (coinSeq fun i => r' (S.emb x hk i)))
          have hcs : ((List.range (ms.map Prod.fst ++ bs).length).map (coinSeq r')).length =
              (ms.map Prod.fst ++ bs).length := by
            simp
          have hpre : ((List.range (S.pre x)).map (coinSeq r')).length = S.pre x := by simp
          rw [S.next_post x (ms.map Prod.fst ++ bs) _ hlt hcs, List.take_left' hl, List.drop_left' hl,
            List.length_append, hl, List.range_add, List.map_append, List.drop_left' hpre, List.map_map]
          congr 1
          apply List.map_congr_left
          intro i _
          have hiff : S.pre x + i < D.V.k x ↔ i < D.flock.V.k (S.stmt x (ms.map Prod.fst)) := by
            rw [S.k_eq, hk]
            omega
          show coinSeq r' (S.pre x + i) = coinSeq (fun i => r' (S.emb x hk i)) i
          by_cases hi : i < D.flock.V.k (S.stmt x (ms.map Prod.fst))
          · simp only [coinSeq, hiff.2 hi, hi, ↓reduceDIte]
            rfl
          · simp only [coinSeq, mt hiff.1 hi, hi, ↓reduceDIte])
        (D.flock.V.k (S.stmt x (ms.map Prod.fst))) [] [] ⟨[], [], [], .rej⟩
      simp only [List.append_nil, List.nil_append] at this
      exact this
    refine ⟨S.stmt x (ms.map Prod.fst), S.emb x hk, P', S.emb_strictMono x hk, S.mem_rounds_iff x u hk,
      fun r' hr' => ?_, fun r' hr' => ?_⟩
    · simp only [hcomp r' fun t ht => hr' t (by rw [S.rounds_iff]; omega), iff_self]
    · rw [hcomp r' fun t ht => hr' t (by rw [S.rounds_iff]; omega) fun i => show t.val < S.pre x + i.val by omega]
      exact ⟨S.session_pass x u ms _ _ _ m hlen (by simp) hms hm, fun _ hq => hq⟩
  · have hk : D.flock.V.k (S.xR x) = S.kS x := S.k_xR x
    let P' : Prover Unit := ⟨fun _ _ _ => .ret []⟩
    have hflock : ∀ c : Fin (D.flock.V.k (S.xR x)) → Coin,
        run H P' D.flock.V (S.xR x) () c = ⟨[([], .rej)], [], [], .rej⟩ := fun c =>
      stepWith_refused H (S.xR x)
        (fun as => D.flock.V.next (S.xR x) as ((List.range as.length).map (coinSeq c))) (S.refuses x (coinSeq c 0))
        (D.flock.V.k (S.xR x)) (by rw [hk]; exact S.kS_pos x)
    have hcomp : ∀ r' : Fin (D.V.k x) → Coin, (∀ t : Fin (D.V.k x), t.val < S.pre x → r' t = r t) →
        run H P D.V x ω r' = ⟨ms ++ [(a, .rej)], ps, [], .rej⟩ := by
      intro r' hr'
      have h0 := hpre r' hr'
      simp only [List.map_nil] at h0
      rw [hrun r', h0]
    refine ⟨S.xR x, S.emb x hk, P', S.emb_strictMono x hk, S.mem_rounds_iff x u hk, fun r' hr' => ?_,
      fun r' hr' => ?_⟩
    · simp only [hcomp r' fun t ht => hr' t (by rw [S.rounds_iff]; omega), hflock, iff_self]
    · rw [hcomp r' fun t ht => hr' t (by rw [S.rounds_iff]; omega) fun i => show t.val < S.pre x + i.val by omega,
        hflock]
      exact ⟨S.session_refused x u ms a hlen hms, fun _ hq => by simp at hq⟩

end Claims.Sampling
