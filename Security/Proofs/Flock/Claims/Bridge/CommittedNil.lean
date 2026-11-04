import Proofs.Flock.Claims.Bridge.Hm96Tree

/-!
# Committed coins at a verifier that may answer a round with nothing

`Committed.lean` bounds the committed-coins run at a verifier that answers every round with its coin (`AnswersCoin`).
The recursion's compiled verifier doesn't: it answers a V\* session that accepts before the last one with `[]`
(`RecDeploy.sessOut`), and there the deployment opens no coin, each V\* session's coin tree ending at its decision.

* `AnswersCoinOrNil H V x ans`: every message `V` answers round `j` with is `ans` of coin `j`, or `[]`.
* `CProver.atM`, `runCM` (**the committed-coins run with masked openings**): the prover reads round `i`'s opening with
  its answer where that answer is not `[]`, and no opening where it is.
* `OnlineSim.LastOnly`: the simulator's opening of round `j` reads only its tape, `j` and round `j`'s answer, as the hm96
  tree's simulator's does (`Hm96Tree.sim_lastOnly`).
* `simProverM S k P`: the prover with `S`'s root and openings, masked as `runCM` masks them.
* `runCM_near`: every event of the masked committed run is within `δ_hide` of the same event at `Claims.run` of
  `simProverM`, wherever the tree hides at `δ_hide` against a simulator that reads only the last answer;
  `Pr_runCM_le_hm96` is the one-sided bound at the hm96 tree, `δ_hide = 2 · k(x) · δ₁`.
* `Pr_acc_runCM_le`: knowledge soundness at those coins, from `PrBad` at live ones: the masked committed run accepts at
  most as often as the extractor, run on `simProverM`, recovers a witness, plus `PrBad` at `simProverM` and
  `2 · k(x) · δ₁`.
-/

namespace Claims.Bridge

open FlockSoundness.Game (ind)
open scoped ENNReal

/-- **The verifier answers with its coin or with nothing**: every message it answers a round with is `ans` of that
round's coin, the last of the coins it is given, or `[]`. -/
def AnswersCoinOrNil {Coin : Type} (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (ans : Coin → Msg) : Prop :=
  ∀ (as : List Msg) (cs : List Coin) (c : Coin) (m : Msg), ((V.next x as (cs ++ [c])).eval H).1 = .msg m →
    m = ans c ∨ m = []

theorem AnswersCoin.orNil {Coin : Type} {H : Msg → Msg} {V : Verifier Coin} {x : Msg} {ans : Coin → Msg}
    (h : AnswersCoin H V x ans) : AnswersCoinOrNil H V x ans :=
  fun as cs c m hm => .inl (h as cs c m hm)

/-- The openings a prover reads at the history `hist`: round `i`'s, `π i`, where round `i`'s answer is not `[]`, and
`[]` where it is. -/
def maskOpen (π : ℕ → Msg) (hist : List (Msg × Msg)) : List Msg :=
  (List.range hist.length).map fun i => if (hist.getD i ([], [])).2 = [] then [] else π i

/-- The prover at the root `h` and the openings `π`, reading round `i`'s opening once it has round `i`'s answer, where
that answer is not `[]`. -/
def CProver.atM {Ω : Type} (P : CProver Ω) (h : Msg) (π : ℕ → Msg) : Prover Ω :=
  ⟨fun x hist ω => P.next x h hist (maskOpen π hist) ω⟩

/-- **The committed-coins run with masked openings** at tape `ω`, coins `r` and the tree's randomness `τ`: the verifier
sends the root of `r` before the prover's first message and opens round `j`'s coin with its answer, except where that
answer is `[]`; `V` decides on the coins `r`. -/
def runCM {Coin Ω T : Type} [Inhabited Coin] (H : Msg → Msg) (P : CProver Ω) (V : Verifier Coin) (x : Msg)
    (C : CoinTree Coin T (V.k x)) (ω : Ω) (r : Fin (V.k x) → Coin) (τ : T) : Trace :=
  run H (P.atM (C.hello r τ) (openSeq (C.opn r τ))) V x ω r

/-- **The simulator's opening reads only its round's answer**: two answer lists of one length with one last answer get
one opening. -/
def OnlineSim.LastOnly (S : OnlineSim) : Prop :=
  ∀ (σ : S.Tape) (as bs : List Msg), as.length = bs.length → as.getLast? = bs.getLast? → S.opn σ as = S.opn σ bs

/-- **The prover with a simulated root and masked openings**: a prover of `Claims.run` with tape `Ω × S.Tape`, which
reads `S`'s root, and round `i`'s opening from `S`'s tape and the answers to rounds `0, …, i` where round `i`'s answer
is not `[]`. -/
def simProverM {Ω : Type} (S : OnlineSim) (k : Msg → ℕ) (P : CProver Ω) : Prover (Ω × S.Tape) :=
  ⟨fun x hist ω => P.next x (S.hello ω.2) hist
    (maskOpen (fun i => if i < k x then S.opn ω.2 ((hist.take (i + 1)).map Prod.snd) else []) hist) ω.1⟩

/-! ## Two provers that agree on the histories a run reaches -/

/-- The history answers each round `i` with `ans` of the coin `r i`, or with `[]`. -/
def AnsweredOr {Coin : Type} (ans : Coin → Msg) (r : ℕ → Coin) (hist : List (Msg × Msg)) : Prop :=
  ∀ i (hi : i < hist.length), hist[i].2 = ans (r i) ∨ hist[i].2 = []

theorem answeredOr_nil {Coin : Type} (ans : Coin → Msg) (r : ℕ → Coin) : AnsweredOr ans r [] :=
  fun _ hi => absurd hi (Nat.not_lt_zero _)

theorem answeredOr_snoc {Coin : Type} {ans : Coin → Msg} {r : ℕ → Coin} {hist : List (Msg × Msg)}
    (h : AnsweredOr ans r hist) (a m : Msg) (hm : m = ans (r hist.length) ∨ m = []) :
    AnsweredOr ans r (hist ++ [(a, m)]) := by
  intro i hi
  rw [List.length_append, List.length_singleton] at hi
  rcases Nat.lt_or_ge i hist.length with hl | hl
  · rw [List.getElem_append_left hl]
    exact h i hl
  · have hi' : i = hist.length := by omega
    subst hi'
    simpa using hm

/-- **Provers that agree on every history the run can reach run alike**: against a verifier that answers with its
coins or with nothing, two provers whose next messages agree on every history answering each round with its coin or
with `[]` have the same rounds. -/
theorem stepWith_congrOr {Coin : Type} (H : Msg → Msg) {Ω Ω' : Type} (P : Prover Ω) (P' : Prover Ω')
    (V : Verifier Coin) (x : Msg) (ω : Ω) (ω' : Ω') (ans : Coin → Msg) (r : ℕ → Coin)
    (hV : AnswersCoinOrNil H V x ans) (hP : ∀ hist, AnsweredOr ans r hist → P.next x hist ω = P'.next x hist ω') :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), AnsweredOr ans r hist →
      hist.length = as.length →
      stepWith H P x ω (fun as => V.next x as ((List.range as.length).map r)) n hist as t =
        stepWith H P' x ω' (fun as => V.next x as ((List.range as.length).map r)) n hist as t := by
  intro n
  induction n with
  | zero => intro _ _ _ _ _; rfl
  | succ n ih =>
    intro hist as t hA hl
    simp only [stepWith]
    rw [hP hist hA]
    generalize hvb : (V.next x (as ++ [((P'.next x hist ω').eval H).1])
      ((List.range (as ++ [((P'.next x hist ω').eval H).1]).length).map r)).eval H = vb
    obtain ⟨b, l⟩ := vb
    cases b with
    | msg m =>
      have hm : m = ans (r hist.length) ∨ m = [] := by
        rw [hl]
        refine hV (as ++ [((P'.next x hist ω').eval H).1]) ((List.range as.length).map r) (r as.length) m ?_
        have e : (List.range (as ++ [((P'.next x hist ω').eval H).1]).length).map r =
            (List.range as.length).map r ++ [r as.length] := by
          simp [List.range_succ]
        rw [← e, hvb]
      exact ih _ _ _ (answeredOr_snoc hA _ _ hm) (by simp [hl])
    | acc => rfl
    | rej => rfl

/-- **The simulating prover's run** is the committed prover's at the simulator's root and openings, both masked,
wherever the simulator's opening reads only its round's answer. -/
theorem run_simProverM {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) {Ω : Type}
    (S : OnlineSim) (hS : S.LastOnly) (P : CProver Ω) (ans : Coin → Msg) (hV : AnswersCoinOrNil H V x ans) (ω : Ω)
    (σ : S.Tape) (r : Fin (V.k x) → Coin) :
    run H (simProverM S V.k P) V x (ω, σ) r = run H (P.atM (S.hello σ) (openSeq (S.opnSeq ans σ r))) V x ω r := by
  unfold run
  refine stepWith_congrOr H _ _ V x _ _ ans (coinSeq r) hV (fun hist hA => ?_) _ _ _ _ (answeredOr_nil _ _) rfl
  simp only [simProverM, CProver.atM, maskOpen]
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi' := List.mem_range.1 hi
  have hg : hist.getD i ([], []) = hist[i] := by simp [List.getD_eq_getElem?_getD, hi']
  rw [hg]
  by_cases h0 : hist[i].2 = []
  · simp only [h0, if_true]
  rw [if_neg h0, if_neg h0]
  have ha : hist[i].2 = ans (coinSeq r i) := (hA i hi').resolve_right h0
  unfold openSeq OnlineSim.opnSeq
  by_cases hk : i < V.k x
  · rw [if_pos hk, dif_pos hk]
    refine hS σ _ _ (by simp; omega) ?_
    rw [List.getLast?_eq_getElem?, List.getLast?_eq_getElem?]
    simp only [List.length_map, List.length_take, List.length_range]
    have e1 : min (i + 1) hist.length - 1 = i := by omega
    rw [e1, List.getElem?_map, List.getElem?_take_of_lt (by omega), List.getElem?_eq_getElem hi',
      show i + 1 - 1 = i by omega, List.getElem?_map, List.getElem?_range (show i < i + 1 by omega)]
    simp [ha]
  · rw [if_neg hk, dif_neg hk]

/-! ## Masked committed coins against fresh ones -/

/-- **Masked committed coins are within `δ_hide` of fresh ones.** Against a verifier that answers each round with its
coin or with nothing (`AnswersCoinOrNil`), at a tree that hides at distance `δ_hide` against a simulator `S` whose
opening reads only its round's answer: every event of the masked committed run, with the prover's tape and the coins,
has probability within `δ_hide` of the same event at `Claims.run` of the simulating prover (`simProverM`). -/
theorem runCM_near {Coin Ω T : Type} [Fintype Coin] [Inhabited Coin] [Fintype Ω] [Nonempty Ω] [Fintype T]
    (H : Msg → Msg) (V : Verifier Coin) (x : Msg) (C : CoinTree Coin T (V.k x)) (ans : Coin → Msg) (S : OnlineSim)
    {δ_hide : ℝ} (hS : SimHides C ans S δ_hide) (hL : S.LastOnly) (hV : AnswersCoinOrNil H V x ans) (P : CProver Ω)
    (Ev : Ω → (Fin (V.k x) → Coin) → Trace → Prop) :
    |Pr (fun p : Ω × (Fin (V.k x) → Coin) × T => Ev p.1 p.2.1 (runCM H P V x C p.1 p.2.1 p.2.2)) -
      Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => Ev p.1.1 p.2 (run H (simProverM S V.k P) V x p.1 p.2))| ≤
      δ_hide := by
  have hL' : Pr (fun p : Ω × (Fin (V.k x) → Coin) × T => Ev p.1 p.2.1 (runCM H P V x C p.1 p.2.1 p.2.2)) =
      Ex fun ω => Pr fun q : (Fin (V.k x) → Coin) × T =>
        Ev ω q.1 (run H (P.atM (C.hello q.1 q.2) (openSeq (C.opn q.1 q.2))) V x ω q.1) := by
    rw [Pr_eq_Ex, Ex_prod]
    simp only [Pr_eq_Ex]
    rfl
  have hR : Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) =>
        Ev p.1.1 p.2 (run H (simProverM S V.k P) V x p.1 p.2)) =
      Ex fun ω => Pr fun q : (Fin (V.k x) → Coin) × S.Tape =>
        Ev ω q.1 (run H (P.atM (S.hello q.2) (openSeq (S.opnSeq ans q.2 q.1))) V x ω q.1) := by
    have e : (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) =>
          Ev p.1.1 p.2 (run H (simProverM S V.k P) V x p.1 p.2)) =
        fun p => Ev p.1.1 p.2 (run H (P.atM (S.hello p.1.2) (openSeq (S.opnSeq ans p.1.2 p.2))) V x p.1.1 p.2) := by
      funext p
      obtain ⟨⟨ω, σ⟩, r⟩ := p
      rw [run_simProverM H V x S hL P ans hV]
    rw [e, Pr_eq_Ex, Ex_prod, Ex_prod]
    simp only [Pr_eq_Ex, Ex_prod]
    exact congrArg Ex (funext fun ω => Ex_comm _)
  rw [hL', hR]
  exact abs_Ex_sub_le fun ω => hS fun r h π => Ev ω r (run H (P.atM h (openSeq π)) V x ω r)

/-! ## At the hm96 coin tree -/

namespace Hm96Tree

variable {Coin B Y C T₀ : Type} {k : ℕ} (T : Hm96Tree Coin B Y C T₀ k)

/-- **The hm96 tree's simulator reads only the last answer**: its round-`j` opening is the tree's at the coin the last
answer decodes to. -/
theorem sim_lastOnly [Add B] [Fintype Coin] [Nonempty Coin] [Fintype Y] [Nonempty Y] [Fintype T₀] [Nonempty T₀]
    (dec : Msg → Coin) : (T.sim dec).LastOnly := by
  intro σ as bs hl hg
  by_cases ha : as = []
  · subst ha
    have hb : bs = [] := List.eq_nil_of_length_eq_zero (by simpa using hl.symm)
    subst hb
    rfl
  have hb : bs ≠ [] := fun hb => ha (List.eq_nil_of_length_eq_zero (by simpa [hb] using hl))
  have hlast : as.getLast ha = bs.getLast hb := by
    rw [List.getLast?_eq_getLast ha, List.getLast?_eq_getLast hb] at hg
    exact Option.some.inj hg
  simp only [sim]
  by_cases hk : as.length - 1 < k
  · have hk' : bs.length - 1 < k := hl ▸ hk
    rw [dif_pos ⟨ha, hk⟩, dif_pos ⟨hb, hk'⟩]
    have e : (⟨as.length - 1, hk⟩ : Fin k) = ⟨bs.length - 1, hk'⟩ := Fin.ext (by simp [hl])
    rw [e, hlast]
  · have hk' : ¬ bs.length - 1 < k := hl ▸ hk
    rw [dif_neg (fun h => hk h.2), dif_neg (fun h => hk' h.2)]

end Hm96Tree

/-- **The bound a KS proof charges, with masked openings, from `Hm96Hiding`**: against a verifier that answers each
round with its coin, through an injective `ans`, or with nothing, and an hm96 coin tree whose leaves hide at `δ₁ ≠ ∞`,
an event of the masked committed run is at most `2 · k(x) · δ₁` likelier than at `Claims.run` of a prover that reads a
simulator's root and openings, masked alike. -/
theorem Pr_runCM_le_hm96 {Coin B Y C T₀ : Type} [Fintype Coin] [Inhabited Coin] [AddGroup B] [Fintype B] [Fintype Y]
    [Nonempty Y] [Fintype T₀] [Nonempty T₀] (H : Msg → Msg) (V : Verifier Coin) (x : Msg)
    (T : Hm96Tree Coin B Y C T₀ (V.k x)) {δ₁ : ℝ≥0∞} (hH : Hm96Hiding T.My T.cs δ₁) (hδ : δ₁ ≠ ⊤) (ans : Coin → Msg)
    (hans : Function.Injective ans) (hV : AnswersCoinOrNil H V x ans) :
    ∃ S : OnlineSim, ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : CProver Ω)
      (Ev : Ω → (Fin (V.k x) → Coin) → Trace → Prop),
      Pr (fun p : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Y) × T₀ =>
          Ev p.1 p.2.1 (runCM H P V x T.tree p.1 p.2.1 p.2.2)) ≤
        Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => Ev p.1.1 p.2 (run H (simProverM S V.k P) V x p.1 p.2)) +
          2 * V.k x * δ₁.toReal := by
  refine ⟨T.sim (Function.invFun ans), ?_⟩
  intro Ω _ _ P Ev
  have h := (abs_le.1 (runCM_near H V x T.tree ans (T.sim (Function.invFun ans))
    (T.simHides hH hδ ans _ (Function.leftInverse_invFun hans)) (T.sim_lastOnly _) hV P Ev)).2
  linarith

/-- **Knowledge soundness at masked committed coins, from `PrBad` at live ones**: against a verifier that answers each
round with its coin, through an injective `ans`, or with nothing, and an hm96 coin tree whose leaves hide at `δ₁ ≠ ∞`,
one simulator `S` of the tree serves every prover: the masked committed run accepts at most as often as the extractor
`E`, run on the simulating prover `simProverM S`, recovers a witness, plus `PrBad` at that prover and `2 · k(x) · δ₁`. -/
theorem Pr_acc_runCM_le {Coin W B Y C T₀ : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [AddGroup B]
    [Fintype B] [Fintype Y] [Nonempty Y] [Fintype T₀] [Nonempty T₀] (H : Msg → Msg) (V : Verifier Coin) (R : Rel W)
    (E : Extractor Coin W V) (x : Msg) (T : Hm96Tree Coin B Y C T₀ (V.k x)) {δ₁ : ℝ≥0∞}
    (hH : Hm96Hiding T.My T.cs δ₁) (hδ : δ₁ ≠ ⊤) (ans : Coin → Msg) (hans : Function.Injective ans)
    (hV : AnswersCoinOrNil H V x ans) :
    ∃ S : OnlineSim, ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : CProver Ω),
      Pr (fun p : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Y) × T₀ =>
          (runCM H P V x T.tree p.1 p.2.1 p.2.2).out = .acc) ≤
        Pr (fun s : (Ω × S.Tape) × (Fin (V.k x) → Coin) × E.C x =>
            R.ext x (E.eval H (simProverM S V.k P) x s.1 s.2.1 s.2.2).1) +
          PrBad H V R E x (simProverM S V.k P) + 2 * V.k x * δ₁.toReal := by
  obtain ⟨S, hS⟩ := Pr_runCM_le_hm96 H V x T hH hδ ans hans hV
  refine ⟨S, ?_⟩
  intro Ω _ _ P
  have h1 : Pr (fun p : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Y) × T₀ =>
        (runCM H P V x T.tree p.1 p.2.1 p.2.2).out = .acc) ≤
      Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => (run H (simProverM S V.k P) V x p.1 p.2).out = .acc) +
        2 * V.k x * δ₁.toReal :=
    hS Ω P fun _ _ t => t.out = .acc
  have h2 : Pr (fun p : (Ω × S.Tape) × (Fin (V.k x) → Coin) => (run H (simProverM S V.k P) V x p.1 p.2).out = .acc) =
      Pr (fun s : (Ω × S.Tape) × (Fin (V.k x) → Coin) × E.C x =>
        (run H (simProverM S V.k P) V x s.1 s.2.1).out = .acc) :=
    (Pr_drop fun a b => (run H (simProverM S V.k P) V x a b).out = .acc).symm
  have h3 : Pr (fun s : (Ω × S.Tape) × (Fin (V.k x) → Coin) × E.C x =>
        (run H (simProverM S V.k P) V x s.1 s.2.1).out = .acc) ≤
      Pr (fun s : (Ω × S.Tape) × (Fin (V.k x) → Coin) × E.C x =>
          R.ext x (E.eval H (simProverM S V.k P) x s.1 s.2.1 s.2.2).1) +
        PrBad H V R E x (simProverM S V.k P) := by
    unfold PrBad
    refine le_trans ?_ (Claims.Bridge.Pr_or_le _ _)
    refine Pr_mono fun s hs => ?_
    by_cases hR : R.ext x (E.eval H (simProverM S V.k P) x s.1 s.2.1 s.2.2).1
    · exact .inl hR
    · exact .inr ⟨hs, hR⟩
  linarith

end Claims.Bridge
