import Proofs.Flock.Claims.ZK.GKExp
import Proofs.Flock.Soundness.ZK.GK.Link

/-!
# The extraction run hides the witness: `PrefixHides` from Lemma C

`PrefixHides` reads runs of a Claims `V*` (`runStar`, whose messages are `interact`'s); Lemma C
(`FlockSoundness.ZK.Session.adaptive_prefinal_hm96`) reads runs of the session analysis's model, a causal strategy
against a deterministic responder (`FlockSoundness.ZK.run`, the fixed point of a pass). The bridge:

* `Proto.preStrat`: the committed-coin prover's first `K + 1` messages (`Hello` and the `K` coin rounds, all but the
  last) as a strategy whose answers are `V*`'s; it reads only their messages (`strat_snd`), so message `i` reads the
  answers before `i`;
* `vResp`: `V*` at its tape, as its answer to each prefix of the messages;
* `Proto.run_prefix`: a transcript's first `K + 1` messages are that strategy's run against that responder;
* `Proto.view0_run`: so is what the extraction run passes on (`view0`), through `replay`: the coins a transcript opens
  read only its first `K + 1` rounds (`cls_take`), and a transcript of all `K + 2` rounds that opens none has `abort` as
  its last message (`pmG_last`).

`Proto.prefixHides_of_runs`: a bound on every event of the two runs, on the witness and on the dummy, against every
responder is `PrefixHides`. `Proto.runs_close_hm96`: Lemma C gives that bound at `2·|L|·δ₁` when the strategy, on the
witness and on the dummy, is the session's prover on its randomness read through a map with fibers of one size
(`hlink`, `hlink₀`); `Flock.zkR_gkE_hm96` takes that bound at each statement, witness and nonce.
-/

namespace Claims.ZKMal

open Claims

/-! ## Probabilities -/

theorem Pr_eq_prCoin {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) :
    Pr p = (FlockSoundness.Game.prCoin p).toReal := by
  rw [← Ex_ind_eq, ← avgR_eq_Ex, FlockSoundness.ZK.GK.avgR_ind_eq_prCoin]

/-- Through a map whose fibers have one size, an event's probability is its probability on the target. -/
theorem Pr_comp_fiber {α β : Type} [Fintype α] [Nonempty α] [Fintype β] (e : α → β) {k : ℕ}
    (he : ∀ b, Nat.card {a // e a = b} = k) (p : β → Prop) : Pr (fun a => p (e a)) = Pr p := by
  rw [← Ex_ind_comp p e, ← Ex_ind_eq p]
  exact FlockSoundness.ZK.GK.avgR_comp_fiber e he (FlockSoundness.Game.ind p)

/-! ## Transcripts, round by round -/

@[simp] theorem msgOf_msg (m : Msg) : msgOf (.msg m) = m := rfl

theorem histOf_fst (τ : List (Msg × Out)) : (histOf τ).map Prod.fst = τ.map Prod.fst := by
  unfold histOf
  rw [List.map_map]
  rfl

/-- **Round `i` of a transcript**: the prover's message on the history before it, and the verifier's answer to the
messages up to it. -/
theorem interact_getElem (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (i : ℕ) (hi : i < (interact ask pm n hist).length),
      (interact ask pm n hist)[i] =
        (pm (hist ++ histOf ((interact ask pm n hist).take i)),
          ask ((hist ++ histOf ((interact ask pm n hist).take i)).map Prod.fst ++
            [pm (hist ++ histOf ((interact ask pm n hist).take i))]))
  | 0, _, _, hi => by simp [interact] at hi
  | n + 1, hist, i, hi => by
      rcases hb : ask (hist.map Prod.fst ++ [pm hist]) with m | _ | _
      · cases i with
        | zero => simp [interact, hb, histOf]
        | succ i =>
            have hi' : i < (interact ask pm n (hist ++ [(pm hist, m)])).length := by
              simp only [interact, hb, List.length_cons] at hi
              omega
            have := interact_getElem ask pm n (hist ++ [(pm hist, m)]) i hi'
            simp only [interact, hb, List.getElem_cons_succ, List.take_succ_cons, histOf, List.map_cons, msgOf_msg]
            simpa [histOf, List.append_assoc] using this
      · cases i with
        | zero => simp [interact, hb, histOf]
        | succ i => simp [interact, hb] at hi
      · cases i with
        | zero => simp [interact, hb, histOf]
        | succ i => simp [interact, hb] at hi

/-- The verifier's answer in round `i` reads the messages up to round `i`. -/
theorem interact_snd (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (n i : ℕ)
    (hi : i < (interact ask pm n []).length) :
    ((interact ask pm n [])[i]).2 = ask (((interact ask pm n []).take (i + 1)).map Prod.fst) := by
  have h := interact_getElem ask pm n [] i hi
  have h1 := congrArg Prod.fst h
  have h2 := congrArg Prod.snd h
  simp only [List.nil_append] at h1 h2
  rw [h2, List.take_add_one, List.getElem?_eq_getElem hi, Option.toList_some, List.map_append, List.map_cons,
    List.map_nil, h1, histOf_fst]

/-! ## The strategy and the responder -/

/-- A committed-coin prover reads only the answers in its history. -/
theorem strat_snd {Coin : Type} (hello abort : Msg) (dec : Msg → Coin) (acc : Msg → ℕ → Msg → OC Bool)
    (gen : List Coin → OC Msg) : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd →
      strat hello abort dec acc gen h = strat hello abort dec acc gen h'
  | [], [], _ => rfl
  | [], _ :: _, e => by simp at e
  | _ :: _, [], e => by simp at e
  | (a, ρ) :: ops, (a', ρ') :: ops', e => by
      simp only [List.map_cons, List.cons.injEq] at e
      obtain ⟨rfl, e⟩ := e
      have e2 : ops.map (fun p => dec p.2) = ops'.map (fun p => dec p.2) := by
        rw [show (fun p : Msg × Msg => dec p.2) = dec ∘ Prod.snd from rfl, ← List.map_map, ← List.map_map, e]
      simp only [strat, e, e2]

/-- A history of `i` rounds whose answers are `a`'s first `i`, the prover's messages blank. -/
def ansHist {n : ℕ} (a : Fin n → Out) (i : Fin n) : List (Msg × Msg) :=
  List.ofFn fun j : Fin i.val => ([], msgOf (a ⟨j.val, j.isLt.trans i.isLt⟩))

theorem ansHist_congr {n : ℕ} (a a' : Fin n → Out) (i : Fin n) (h : ∀ j, j < i → a j = a' j) :
    ansHist a i = ansHist a' i := by
  unfold ansHist
  congr 1
  funext j
  rw [h _ j.isLt]

/-- **`V*` at its tape as a responder**: answer `i` is its answer to the first `i + 1` messages. -/
def vResp (ask : List Msg → Out) (n : ℕ) : FlockSoundness.ZK.Responder Msg Out n where
  ans τ i := ask (List.ofFn fun j : Fin (i.val + 1) => τ ⟨j.val, lt_of_le_of_lt (Nat.lt_succ_iff.mp j.isLt) i.isLt⟩)
  causal τ τ' i h := by
    congr 1
    refine congrArg List.ofFn (funext fun j => h _ (Nat.lt_succ_iff.mp j.isLt))

/-- Equal message functions run the same against every responder. -/
theorem run_congr {Ω Ω' M C : Type} {n : ℕ} [Inhabited M] (P : FlockSoundness.ZK.Strategy Ω M C n)
    (P' : FlockSoundness.ZK.Strategy Ω' M C n) (V : FlockSoundness.ZK.Responder M C n) (ω : Ω) (ω' : Ω')
    (h : P.msg ω = P'.msg ω') : FlockSoundness.ZK.run P V ω = FlockSoundness.ZK.run P' V ω' := by
  unfold FlockSoundness.ZK.run
  have : P.step V ω = P'.step V ω' := funext fun τ => by simp only [FlockSoundness.ZK.Strategy.step, h]
  rw [this]

namespace Proto

variable {W : Type}

/-- The committed-coin prover's next message, evaluated, with `gen` its message after the coins. -/
def pmG (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (gen : List G.Coin → OC Msg)
    (h : List (Msg × Msg)) : Msg :=
  ((strat (G.hello x ν) G.abort G.dec (G.chk x ν) gen h).eval H).1

theorem pmG_snd (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (gen : List G.Coin → OC Msg)
    (h h' : List (Msg × Msg)) (e : h.map Prod.snd = h'.map Prod.snd) : G.pmG H x ν gen h = G.pmG H x ν gen h' := by
  unfold pmG
  rw [strat_snd _ _ _ _ _ h h' e]

/-- **The committed-coin prover before its last message, as the session analysis's strategy**: on tape `r`, its
message `i` (of `Hello` and the `K` coin rounds) at the answers `a`, from `a`'s first `i`. -/
def preStrat (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (gen : G.Rand → List G.Coin → OC Msg) :
    FlockSoundness.ZK.Strategy G.Rand Msg Out (G.K x + 1) where
  msg r a i := G.pmG H x ν (gen r) (ansHist a i)
  causal r a a' i h := by
    show G.pmG H x ν (gen r) (ansHist a i) = G.pmG H x ν (gen r) (ansHist a' i)
    rw [ansHist_congr a a' i h]

/-- The prover that sends `σ`'s messages in order, then `abort`. -/
def replay (G : Proto W) {n : ℕ} (σ : Fin n → Msg) (h : List (Msg × Msg)) : Msg :=
  if hl : h.length < n then σ ⟨h.length, hl⟩ else G.abort

/-- What the extraction run passes on, from its first `K + 1` messages against `V*`'s answers `ask`. -/
def view0R (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (ask : List Msg → Out)
    (σ : Fin (G.K x + 1) → Msg) : List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin) :=
  G.view0 H x ν (interact ask (G.replay σ) (G.k x) [])

/-- The coins a transcript opens read only its first `K + 1` rounds. -/
theorem cls_take (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out)) :
    G.cls H x ν (τ.take (G.K x + 1)) = G.cls H x ν τ := by
  refine Option.ext fun c => ?_
  rw [cls_eq_some_iff, cls_eq_some_iff]
  unfold TCons
  rw [resp_take (show 0 < G.K x + 1 by omega)]
  constructor
  · rintro ⟨ρ, h0, hall⟩
    refine ⟨ρ, h0, fun l hl => ?_⟩
    obtain ⟨m, h1, h2, h3⟩ := hall l hl
    rw [resp_take (by omega)] at h1
    exact ⟨m, h1, h2, h3⟩
  · rintro ⟨ρ, h0, hall⟩
    refine ⟨ρ, h0, fun l hl => ?_⟩
    obtain ⟨m, h1, h2, h3⟩ := hall l hl
    exact ⟨m, by rw [resp_take (by omega)]; exact h1, h2, h3⟩

/-- **A transcript of all `K + 2` rounds that opens no coin vector has `abort` as its last message**: every round but
the last is answered, so some opening failed the check. -/
theorem pmG_last (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (gen : List G.Coin → OC Msg)
    (τ : List (Msg × Out)) (hlen : τ.length = G.K x + 2)
    (hmsg : ∀ (j : ℕ) (hj : j + 1 < τ.length), ∃ m, (τ[j]'(by omega)).2 = .msg m) (hc : G.cls H x ν τ = none) :
    G.pmG H x ν gen (histOf (τ.take (G.K x + 1))) = G.abort := by
  rcases τ with _ | ⟨⟨a, b⟩, τ'⟩
  · simp at hlen
  obtain ⟨ρ, hb⟩ := hmsg 0 (by simp only [List.length_cons] at hlen ⊢; omega)
  simp only [List.getElem_cons_zero] at hb
  subst hb
  simp only [List.length_cons] at hlen
  simp only [List.take_succ_cons, histOf, List.map_cons, show msgOf (.msg ρ) = ρ from rfl]
  set ops := List.map (fun p => (p.1, msgOf p.2)) (List.take (G.K x) τ') with hops
  have hopsLen : ops.length = G.K x := by simp only [hops, List.length_map, List.length_take]; omega
  have hopsGet : ∀ (l : ℕ) (hl : l < ops.length), resp ((a, .msg ρ) :: τ') (l + 1) = some ops[l].2 := by
    intro l hl
    obtain ⟨m, hm⟩ := hmsg (l + 1) (by simp only [List.length_cons]; omega)
    simp only [List.getElem_cons_succ] at hm
    rw [resp_cons_succ, resp_eq_some.mpr ⟨by omega, hm⟩]
    simp [hops, hm, msgOf]
  unfold pmG
  rw [strat_eval_cons]
  have hR : ((allOk (G.chk x ν ρ) 0 (ops.map Prod.snd)).eval H).1 = false := by
    refine Bool.eq_false_iff.mpr fun hok => ?_
    rw [allOk_eval] at hok
    let c : Fin (G.K x) → G.Coin := fun j => G.dec (ops[j.val]'(by omega)).2
    have hsome : G.cls H x ν ((a, .msg ρ) :: τ') = some c := by
      rw [cls_eq_some_iff]
      refine ⟨ρ, by simp [resp], fun l hl => ⟨(ops[l]'(by omega)).2, hopsGet l (by omega), ?_, ?_⟩⟩
      · have := hok l (by simp only [List.length_map]; omega)
        simpa [Proto.chkP] using this
      · rw [coinAt_eq_some]
        exact ⟨hl, rfl⟩
    rw [hsome] at hc
    cases hc
  rw [hR]
  rfl

section Runs

variable (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
  (gen : G.Rand → List G.Coin → OC Msg) (r : G.Rand)

/-- **A transcript's first `K + 1` messages are the strategy's run against `V*`'s responder.** -/
theorem run_prefix : ∀ (i : ℕ) (hi : i < (interact ask (G.pmG H x ν (gen r)) (G.k x) []).length)
    (hK : i < G.K x + 1),
    ((interact ask (G.pmG H x ν (gen r)) (G.k x) [])[i]).1 =
      FlockSoundness.ZK.run (G.preStrat H x ν gen) (vResp ask (G.K x + 1)) r ⟨i, hK⟩ := by
  set T := interact ask (G.pmG H x ν (gen r)) (G.k x) [] with hT
  set σ := FlockSoundness.ZK.run (G.preStrat H x ν gen) (vResp ask (G.K x + 1)) r with hσ
  have hfix := FlockSoundness.ZK.run_fixed (G.preStrat H x ν gen) (vResp ask (G.K x + 1)) r
  rw [← hσ] at hfix
  intro i
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    intro hi hK
    have hmsgs : ∀ l (hl : l < i), (T.take (l + 1)).map Prod.fst =
        List.ofFn fun j : Fin (l + 1) => σ ⟨j.val, by omega⟩ := by
      intro l hl
      refine List.ext_getElem (by simp; omega) fun j h1 h2 => ?_
      simp only [List.length_map, List.length_take] at h1
      simp only [List.getElem_map, List.getElem_take, List.getElem_ofFn]
      exact ih j (by omega) (by omega) (by omega)
    have h1 : T[i].1 = G.pmG H x ν (gen r) (histOf (T.take i)) := by
      simpa using congrArg Prod.fst (interact_getElem ask (G.pmG H x ν (gen r)) (G.k x) [] i hi)
    rw [h1, ← hfix]
    show _ = G.pmG H x ν (gen r) (ansHist ((vResp ask (G.K x + 1)).ans σ) ⟨i, hK⟩)
    refine G.pmG_snd H x ν (gen r) _ _ (List.ext_getElem (by simp [histOf, ansHist]; omega) fun l h2 h3 => ?_)
    simp only [histOf, List.length_map, List.length_take, lt_min_iff] at h2
    simp only [histOf, ansHist, List.getElem_map, List.getElem_take, List.getElem_ofFn]
    congr 1
    rw [interact_snd ask (G.pmG H x ν (gen r)) (G.k x) l (show l < T.length by omega), ← hT, hmsgs l h2.1]
    rfl

/-- **What the extraction run passes on is a function of its first `K + 1` messages**: `view0R` of the strategy's run
against `V*`'s responder. -/
theorem view0_run :
    G.view0 H x ν (interact ask (G.pmG H x ν (gen r)) (G.k x) []) =
      G.view0R H x ν ask (FlockSoundness.ZK.run (G.preStrat H x ν gen) (vResp ask (G.K x + 1)) r) := by
  set T := interact ask (G.pmG H x ν (gen r)) (G.k x) [] with hT
  set σ := FlockSoundness.ZK.run (G.preStrat H x ν gen) (vResp ask (G.K x + 1)) r with hσ
  have hlen : T.length ≤ G.K x + 2 := interact_length_le ask _ (G.k x) []
  have hagree : ∀ i ≤ G.K x, i < T.length →
      G.replay σ ([] ++ histOf (T.take i)) = G.pmG H x ν (gen r) ([] ++ histOf (T.take i)) := by
    intro i hi hiT
    have hl : (histOf (T.take i)).length = i := by simp [histOf]; omega
    have h1 : T[i].1 = G.pmG H x ν (gen r) (histOf (T.take i)) := by
      simpa using congrArg Prod.fst (interact_getElem ask (G.pmG H x ν (gen r)) (G.k x) [] i hiT)
    simp only [List.nil_append]
    rw [← h1, G.run_prefix H ask x ν gen r i hiT (by omega), replay, dite_eq_left (by rw [hl]; omega)]
    simp only [hl, ← hσ]
  have htake : (interact ask (G.replay σ) (G.k x) []).take (G.K x + 1) = T.take (G.K x + 1) :=
    interact_take_congr ask _ _ (G.k x) [] (G.K x) hagree
  unfold view0R
  cases hc : G.cls H x ν T with
  | some c =>
      have hc' : G.cls H x ν (interact ask (G.replay σ) (G.k x) []) = some c := by
        rw [← G.cls_take H x ν, htake, G.cls_take H x ν, hc]
      simp only [view0, hc, hc']
  | none =>
      have hT' : interact ask (G.replay σ) (G.k x) [] = T := by
        refine interact_congr ask _ _ (G.k x) [] fun i hi => ?_
        rw [← hT] at hi ⊢
        by_cases hiK : i ≤ G.K x
        · exact hagree i hiK hi
        · have hi' : i = G.K x + 1 := by omega
          subst hi'
          simp only [List.nil_append]
          have hl : (histOf (T.take (G.K x + 1))).length = G.K x + 1 := by simp [histOf]; omega
          rw [replay, dite_eq_right (by rw [hl]; omega)]
          exact (G.pmG_last H x ν (gen r) T (by omega) (interact_msg ask _ (G.k x) []) hc).symm
      rw [hT']

end Runs

/-! ## `PrefixHides` from runs of the strategy -/

/-- **`PrefixHides` from the strategy's runs**: if, at every nonce and against every responder, every event of the
strategy's run on the witness and on the dummy has probabilities within `δ_A(x)`, the dummy-witness run hides the
witness until the last coin. -/
theorem prefixHides_of_runs (G : Proto W) (H : Msg → Msg) (R : Rel W) (δA : Msg → ℝ)
    (h : ∀ x w, R x w → ∀ (ν : G.Nonce) (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1))
      (E : (Fin (G.K x + 1) → Msg) → Prop),
      |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
        Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤ δA x) :
    G.PrefixHides H R δA := by
  intro x w hR ν Θ Vs θ hk E
  have e1 : ∀ r, G.view0 H x ν (runStar H (G.prover w) Vs x (ν, r) θ).msgs =
      G.view0R H x ν (fun as => ((Vs.next x as θ).eval H).1)
        (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) (vResp (fun as => ((Vs.next x as θ).eval H).1) _) r) :=
    fun r => by
      rw [runStar_msgs, hk]
      exact G.view0_run H _ x ν (G.msg x w ν) r
  have e2 : ∀ r, G.view0 H x ν (runStar H G.dprover Vs x (ν, r) θ).msgs =
      G.view0R H x ν (fun as => ((Vs.next x as θ).eval H).1)
        (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) (vResp (fun as => ((Vs.next x as θ).eval H).1) _) r) :=
    fun r => by
      rw [runStar_msgs, hk]
      exact G.view0_run H _ x ν (G.dmsg x ν) r
  simp only [e1, e2]
  exact h x w hR ν _ fun σ => E (G.view0R H x ν (fun as => ((Vs.next x as θ).eval H).1) σ)

/-! ## The strategy's runs from Lemma C -/

/-- **Lemma C for the committed-coin prover's runs** (`Session.adaptive_prefinal_hm96`, at `2·|L|·δ₁`): at the
statement `x` and nonce `ν`, in Lemma C's setting with the verifier's answers as the session's (`T a`, the session's
tables at the coins the answers `a` open; the prover `P` given by `msgs` of its leaves, `hP`; its world-`W₀` form `hI`),
for table witnesses `wt` (the witness's) and `wt₀` (the dummy's) carrying the public region data, if the strategy on
`w` and on the dummy is `P` on the randomness `eR r`, read through a map with fibers of one size (`hlink`, `hlink₀`),
every event of the two runs against every responder has probabilities within `2·|L|·δ₁`. -/
theorem runs_close_hm96 (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (w : W)
    {F K Wt Salt B D Hi VL Cc Ω L : Type} [Field F] [Field K] [CharP F 2] [AddCommGroup Salt] [AddCommGroup B]
    [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B]
    [Fintype Hi] [DecidableEq Hi] [Fintype Ω] [Nonempty Ω] [Fintype L] [DecidableEq L] {J : ℕ}
    (T : (Fin (G.K x + 1) → Out) → Fin J → FlockSoundness.ZK.Table F K Wt Salt (B × Salt) D Hi VL)
    (rep : ∀ a j, Fin (T a j).nC → Fin 2) (My : Salt → B) (cs : Salt → Cc) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding My cs δ₁) (hδ₁ : δ₁ ≠ ⊤) (Hl : B × Cc → D)
    (msgs : (Fin J → Wt) → Ω → (Fin (G.K x + 1) → Out) → (L → D) → Fin (G.K x + 1) → Msg)
    (dig : (Fin J → Wt) → Ω → (Fin (G.K x + 1) → Out) → L → B)
    (P : (Fin J → Wt) → FlockSoundness.ZK.Strategy (Ω × (L → B × Cc)) Msg Out (G.K x + 1))
    (hP : ∀ wt ω p a, (P wt).msg (ω, p) a = msgs wt ω a fun l => Hl (dig wt ω a l + (p l).1, (p l).2))
    (e : ∀ a, Ω × (L → B × Salt) → ∀ j, (T a j).Rand) (k : (Fin (G.K x + 1) → Out) → ℕ)
    (he : ∀ a r, Nat.card {u // e a u = r} = k a) (out : ∀ a, (∀ j, (T a j).Cut) → Fin (G.K x + 1) → Msg)
    (hI : ∀ wt ω u a, msgs wt ω a (fun l => Hl ((u l).1, cs (u l).2))
      = out a fun j => (T a j).cut (FlockSoundness.ZK.Session.sent (T a) (rep a) j) ((T a j).preView (wt j) (e a (ω, u) j)))
    (wt wt₀ : Fin J → Wt) (hreg : ∀ a j, (T a j).sReg (wt j) = (T a j).pubReg)
    (hreg₀ : ∀ a j, (T a j).sReg (wt₀ j) = (T a j).pubReg)
    (eR : G.Rand → Ω × (L → Salt)) (kR : ℕ) (heR : ∀ p, Nat.card {r // eR r = p} = kR)
    (hlink : ∀ r, (G.preStrat H x ν (G.msg x w ν)).msg r =
      (P wt).msg ((eR r).1, fun l => (My ((eR r).2 l), cs ((eR r).2 l))))
    (hlink₀ : ∀ r, (G.preStrat H x ν (G.dmsg x ν)).msg r =
      (P wt₀).msg ((eR r).1, fun l => (My ((eR r).2 l), cs ((eR r).2 l))))
    (V : FlockSoundness.ZK.Responder Msg Out (G.K x + 1)) (E : (Fin (G.K x + 1) → Msg) → Prop) :
    |Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) -
      Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r))| ≤ 2 * Fintype.card L * δ₁.toReal := by
  have : Nonempty Salt := ⟨0⟩
  have eW : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.msg x w ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) => E (FlockSoundness.ZK.run (P wt) V (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink r)]
  have eD : Pr (fun r : G.Rand => E (FlockSoundness.ZK.run (G.preStrat H x ν (G.dmsg x ν)) V r)) =
      (FlockSoundness.Game.prCoin fun p : Ω × (L → Salt) => E (FlockSoundness.ZK.run (P wt₀) V (p.1, fun l => (My (p.2 l), cs (p.2 l))))).toReal := by
    rw [← Pr_eq_prCoin, ← Pr_comp_fiber eR heR]
    congr 1
    funext r
    rw [run_congr _ _ V r _ (hlink₀ r)]
  have h1 := FlockSoundness.ZK.Session.adaptive_prefinal_hm96 T rep My cs hT1 Hl msgs dig P hP e k he out hI wt wt₀ hreg hreg₀ V E
  have h2 := FlockSoundness.ZK.Session.adaptive_prefinal_hm96 T rep My cs hT1 Hl msgs dig P hP e k he out hI wt₀ wt hreg₀ hreg V E
  have hε : 2 * (Fintype.card L : ENNReal) * δ₁ ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.mul_ne_top (by simp) (ENNReal.natCast_ne_top _)) hδ₁
  have hεr : (2 * (Fintype.card L : ENNReal) * δ₁).toReal = 2 * Fintype.card L * δ₁.toReal := by
    simp [ENNReal.toReal_mul]
  have a := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h1
  have b := FlockSoundness.ZK.GK.toReal_le_add (FlockSoundness.ZK.prCoin_ne_top _) hε h2
  rw [hεr] at a b
  rw [eW, eD, abs_le]
  constructor <;> linarith

end Proto

end Claims.ZKMal
