module

public import Proofs.Flock.SpotFlock.KS
public import Proofs.Flock.SamplingGeneric.OpensTwo
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.ProgReads

/-!
# SPOT[Flock]'s session opens each checked proof block at its replay block

`sessionOpensTwo h`: `Claims.SamplingGeneric.SessionOpensTwo` at SPOT[Flock]'s two-stage code (`code h`), Flock's
masked record and SHA-512, from what one session statement establishes (`StmtOpens h`).

**The run** (`run_seenAt`). An accepted run of `V h` answered rounds 0 to `stage` with their coins, the prover sending
nothing in rounds 2 to `stage`, and round `stage + 1` with nothing, so the verifier's check of the session statements
passed there (`stmtOk h x m₁ m₂ c₀ c₁`, at the round-1 message `m₁`, round `stage + 1`'s message `m₂`, round 0's coin
`c₀` and the second stage's coins `c₁`, rounds 1 to `stage`'s), and its transcript is `seenAt` of them. A checked
unit's session is its template's, at the statement `m₂` states for that template (`unitStmt`), which the run answered
every round before (`SessionSeq.session_acc`); the interior roots are those `m₁` states, and its checked units are the
second stage's on `c₀` and `c₁` (`checked_eq`).

**The statement** (`StmtOpens h`): with no run and no coin beyond `c₀` and `c₁`, session statements that the verifier's
check passes, at a Flock witness in the relation of a unit's statement, reveal for that unit the two stages check
openings under which its proof block is correct on its replay block's values. That is what the session statement's
relation, the witness's encoding of its openings and the registered reads establish; the run only selects the
statements.

**The witness's values** (`WitOpens h`, `stmtOpens`). A checked unit `u` has a proof block `b` in a replay block `ρ`
with gates `rb` (`mem_checked_lt`, `InScopeTwo`, `replay_block`). `StmtOpens` follows from one pair of values: boundary
values `W` and interior values `I` that make `b` correct on `ρ`'s values, each that `b`'s correctness needs
(`Prog.Needs`) revealed by an opening that passes, an interior one of `ρ` under the interior roots, every other under
the run's roots. Values that agree with those openings agree with `W` and `I` wherever `b`'s correctness reads them
(`Prog.correct_of_needs`).

**The witness's values, decoded** (`Decodes`, `Reveals`, `witOpens`). `WitOpens` splits in two at a decoder `dec` of
the session witness, each gate's value for each checked unit: the decoded values make each checked unit correct
(`Decodes`, what the session statement's relation says of its circuit), and the reveal opens each decoded value the
unit's correctness reads, under the roots its gate is committed under (`Reveals`, what the commitment's layout and the
reveal say of the witness). `Decodes` follows from `Evaluates`, that the decoded values evaluate each of the unit's
gates on the decoded values of the gates it reads (`decodes_of_evaluates`, by `Prog.correct_of_eval`), which is what a
session circuit that computes the unit's gates gives. Neither holds of every `h`. The check fixes template `t`'s session
statement to `xS₀ t` but for what the run sets (`pin_of_stmtOkAt`), so `Evaluates` says that each
`xS₀ t`'s circuit computes the gates of the drawn units of template `t` on the values its witness encodes, and `Reveals`
that the frame (`h.frame`) opens them.

So `ksTwo_spot_of_eval`: SPOT[Flock]'s two-stage knowledge soundness at its code (`ksTwo_spot`) from `Evaluates` and
`Reveals`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-! ## Runs -/

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)

theorem stepWith_succ_msg {f : ℕ} {hist : List (Msg × Msg)} {as : List Msg} {t : Transcript} {m : Msg}
    {lo : List (Msg × Msg)} (h : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H = (.msg m, lo)) :
    stepWith H P x ω vn (f + 1) hist as t =
      stepWith H P x ω vn f (hist ++ [(((P.next x hist ω).eval H).1, m)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo } := by
  simp only [stepWith, h]

/-- **The later rounds of a second stage**: a verifier that, past its first two rounds, answers each of `n` rounds
with that round's message `c j` where the prover sent nothing and refuses it otherwise, and then checks the next round,
accepts only past those rounds answered and a round answered with nothing, at a check that passed. -/
theorem stepWith_pad (c : ℕ → Msg) (n : ℕ) (ok : List Msg → Bool) (a₀ a₁ : Msg)
    (hpad : ∀ (as : List Msg) (a : Msg), 2 ≤ as.length → as.length ≤ n + 1 →
      (a = [] ∧ vn (as ++ [a]) = .ret (.msg (c as.length))) ∨ vn (as ++ [a]) = .ret .rej)
    (hstmt : ∀ as : List Msg, as.length = n + 3 → vn as = .ret (if ok as then .msg [] else .rej)) :
    ∀ (d k : ℕ), k + d = n → ∀ (f : ℕ) (hist : List (Msg × Msg)) (t : Transcript),
      (stepWith H P x ω vn f hist (a₀ :: a₁ :: List.replicate k []) t).out = .acc →
      t.msgs = (a₀, .msg (c 0)) :: (a₁, .msg (c 1)) ::
        (List.range k).map (fun i => (([] : Msg), Out.msg (c (i + 2)))) →
      ∃ m₂ rest, (stepWith H P x ω vn f hist (a₀ :: a₁ :: List.replicate k []) t).msgs =
          (a₀, .msg (c 0)) :: (a₁, .msg (c 1)) ::
            ((List.range n).map (fun i => (([] : Msg), Out.msg (c (i + 2)))) ++ (m₂, .msg []) :: rest) ∧
        ok ((a₀ :: a₁ :: List.replicate n []) ++ [m₂]) = true := by
  intro d
  induction d with
  | zero =>
    intro k hk f hist t hacc hmsgs
    have hkn : n = k := by omega
    subst hkn
    cases f with
    | zero => simp [stepWith] at hacc
    | succ g =>
      have hl : ((a₀ :: a₁ :: List.replicate n ([] : Msg)) ++ [((P.next x hist ω).eval H).1]).length = n + 3 := by
        simp
      cases hok : ok ((a₀ :: a₁ :: List.replicate n []) ++ [((P.next x hist ω).eval H).1]) with
      | false =>
        have e : (vn ((a₀ :: a₁ :: List.replicate n []) ++ [((P.next x hist ω).eval H).1])).eval H = (.rej, []) := by
          rw [hstmt _ hl, hok]; rfl
        simp only [stepWith, e] at hacc
        cases hacc
      | true =>
        rw [stepWith_succ_msg H P x ω vn (m := []) (lo := []) (by rw [hstmt _ hl, hok]; rfl)]
        obtain ⟨rest, hr⟩ := Claims.Sampling.stepWith_msgs_prefix H P x ω vn g _ _ _
        refine ⟨((P.next x hist ω).eval H).1, rest, ?_, hok⟩
        rw [hr, hmsgs]
        simp
  | succ d ih =>
    intro k hk f hist t hacc hmsgs
    cases f with
    | zero => simp [stepWith] at hacc
    | succ g =>
      have hl : (a₀ :: a₁ :: List.replicate k ([] : Msg)).length = k + 2 := by simp
      rcases hpad (a₀ :: a₁ :: List.replicate k []) ((P.next x hist ω).eval H).1 (by rw [hl]; omega)
        (by rw [hl]; omega) with ⟨ha, hv⟩ | hv
      · rw [hl] at hv
        rw [stepWith_succ_msg H P x ω vn (m := c (k + 2)) (lo := []) (by rw [hv]; rfl)] at hacc ⊢
        have has : (a₀ :: a₁ :: List.replicate k ([] : Msg)) ++ [((P.next x hist ω).eval H).1] =
            a₀ :: a₁ :: List.replicate (k + 1) [] := by
          rw [ha]
          exact (congrArg (fun l => a₀ :: a₁ :: l) List.replicate_succ').symm
        rw [has] at hacc ⊢
        refine ih (k + 1) (by omega) g _ _ hacc ?_
        simp [hmsgs, List.range_succ, ha]
      · have e : (vn ((a₀ :: a₁ :: List.replicate k []) ++ [((P.next x hist ω).eval H).1])).eval H = (.rej, []) := by
          rw [hv]; rfl
        simp only [stepWith, e] at hacc
        cases hacc

/-- **A run whose verifier answers its first two rounds with their messages, then `n` rounds with theirs where the
prover sent nothing, and checks the next** accepts only past that round answered with nothing, at a check that
passed. -/
theorem stepWith_stage (c : ℕ → Msg) (n : ℕ) (ok : List Msg → Bool) (hv1 : ∀ a, vn [a] = .ret (.msg (c 0)))
    (hv2 : ∀ a b, vn [a, b] = .ret (.msg (c 1)))
    (hpad : ∀ (as : List Msg) (a : Msg), 2 ≤ as.length → as.length ≤ n + 1 →
      (a = [] ∧ vn (as ++ [a]) = .ret (.msg (c as.length))) ∨ vn (as ++ [a]) = .ret .rej)
    (hstmt : ∀ as : List Msg, as.length = n + 3 → vn as = .ret (if ok as then .msg [] else .rej)) :
    ∀ (f : ℕ), (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).out = .acc →
      ∃ m₀ m₁ m₂ rest, (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).msgs =
          (m₀, .msg (c 0)) :: (m₁, .msg (c 1)) ::
            ((List.range n).map (fun i => (([] : Msg), Out.msg (c (i + 2)))) ++ (m₂, .msg []) :: rest) ∧
        ok ((m₀ :: m₁ :: List.replicate n []) ++ [m₂]) = true
  | 0, hacc => by simp [stepWith] at hacc
  | 1, hacc => by simp [stepWith, hv1, OC.eval] at hacc
  | g + 1 + 1, hacc => by
    rw [stepWith_succ_msg H P x ω vn (m := c 0) (lo := []) (by rw [List.nil_append, hv1]; rfl)] at hacc ⊢
    simp only [List.nil_append] at hacc ⊢
    rw [stepWith_succ_msg H P x ω vn (m := c 1) (lo := []) (by rw [List.singleton_append, hv2]; rfl)] at hacc ⊢
    simp only [List.cons_append, List.nil_append] at hacc ⊢
    obtain ⟨m₂, rest, hm, hok⟩ :=
      stepWith_pad H P x ω vn c n ok _ _ hpad hstmt n 0 (Nat.zero_add n) g _ _ hacc rfl
    exact ⟨_, _, m₂, rest, hm, hok⟩

end Steps

variable (h : Held)

theorem k_pos (x : Msg) : 0 < (V h).k x := by show 0 < off h h.ms.length + 1; omega

/-- **A run's transcript through its session statements**: round 0 answered with its coin `c₀`, rounds 1 to `stage`
with the second stage's coins `c₁`, the prover sending `m₁` in round 1 and nothing after it, round `stage + 1`'s message
`m₂` answered with nothing, then `rest`, accepted. -/
def seenAt {h : Held} (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) : Seen :=
  ⟨(m₀, .msg c₀.bytes) :: (m₁, .msg (coinSeq c₁ 0).bytes) ::
    (((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq c₁ (i + 1)).bytes)) ++ (m₂, .msg []) :: rest),
    .acc⟩

/-- The session statements' message of `seenAt`, round `stage + 1`'s. -/
theorem seenAt_stmts {m₀ m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {rest : List (Msg × Out)} :
    ((seenAt m₀ m₁ m₂ c₀ c₁ rest).msgs.getD (h.stage + 1) ([], .rej)).1 = m₂ := by
  have hl : ((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq c₁ (i + 1)).bytes)).length = h.more := by
    simp
  have e := List.getD_append_right
    ((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq c₁ (i + 1)).bytes)) ((m₂, Out.msg []) :: rest)
    ([], Out.rej) h.more hl.le
  rw [hl, Nat.sub_self] at e
  exact congrArg Prod.fst e

/-- The verifier's step at a run's coins in round 0: the round's coin. -/
theorem next_run_one {x : Msg} (r : Fin ((V h).k x) → FlockDeploy.Coin) (a : Msg) :
    next h x [a] ((List.range [a].length).map (coinSeq r)) = .ret (.msg (coinSeq r 0).bytes) := by
  have e := next_two h x (as := [a]) (cs := (List.range [a].length).map (coinSeq r)) (List.cons_ne_nil a [])
    (by simp) (by simp)
  rw [getD_map_range (coinSeq r) (show [a].length - 1 < [a].length by simp) default] at e
  exact e

/-- The verifier's step at a run's coins in round 1: the round's coin. -/
theorem next_run_two {x : Msg} (r : Fin ((V h).k x) → FlockDeploy.Coin) (a b : Msg) :
    next h x [a, b] ((List.range [a, b].length).map (coinSeq r)) = .ret (.msg (coinSeq r 1).bytes) := by
  have e := next_two h x (as := [a, b]) (cs := (List.range [a, b].length).map (coinSeq r)) (List.cons_ne_nil a [b])
    (by simp) (by simp)
  rw [getD_map_range (coinSeq r) (show [a, b].length - 1 < [a, b].length by simp) default] at e
  exact e

/-- **The verifier's step at a run's coins in rounds 2 to `stage`**: the round's coin where the prover sent nothing,
refused otherwise. -/
theorem next_run_pad {x : Msg} (r : Fin ((V h).k x) → FlockDeploy.Coin) (as : List Msg) (a : Msg)
    (h2 : 2 ≤ as.length) (hle : as.length ≤ h.more + 1) :
    (a = [] ∧ next h x (as ++ [a]) ((List.range (as ++ [a]).length).map (coinSeq r)) =
        .ret (.msg (coinSeq r as.length).bytes)) ∨
      next h x (as ++ [a]) ((List.range (as ++ [a]).length).map (coinSeq r)) = .ret .rej := by
  have hs : h.stage = h.more + 1 := rfl
  have hl : (as ++ [a]).length = as.length + 1 := by simp
  have hn := next_coin h x (as := as ++ [a]) (cs := (List.range (as ++ [a]).length).map (coinSeq r)) (by simp)
    (by simp) (by rw [hl]; omega)
  rw [hl, Nat.add_sub_cancel, getD_concat, getD_map_range (coinSeq r) (Nat.lt_succ_self as.length)] at hn
  rw [hl]
  by_cases ha : a = []
  · exact Or.inl ⟨ha, by rw [hn, ite_eq_left (Or.inr ha)]⟩
  · exact Or.inr (by rw [hn, ite_eq_right fun hc => hc.elim (fun h' => by omega) ha])

/-- **The verifier's step at a run's coins in round `stage + 1`**: the session statements' check at round 0's coin and
the second stage's. -/
theorem next_run_stmt {x : Msg} (r : Fin ((V h).k x) → FlockDeploy.Coin) (as : List Msg)
    (hlen : as.length = h.more + 3) :
    next h x as ((List.range as.length).map (coinSeq r)) =
      .ret (if stmtOk h x (as.getD 1 []) (as.getD (h.stage + 1) []) (r ⟨0, k_pos h x⟩) (stageCoins h x r) then
        .msg [] else .rej) := by
  have hs : h.stage = h.more + 1 := rfl
  have e0 : coinSeq r 0 = r ⟨0, k_pos h x⟩ := dite_eq_left (k_pos h x)
  have hst : stageOf h ((List.range (h.stage + 2)).map (coinSeq r)) = stageCoins h x r :=
    funext fun i => (getD_map_range (coinSeq r) (show i.val + 1 < h.stage + 2 by have := i.isLt; omega) default).trans
      (coinSeq_stage h x r i)
  have hlen' : as.length = h.stage + 2 := hlen
  rw [next_stmt h x (by simp) hlen', hlen', getD_map_range (coinSeq r) (show 0 < h.stage + 2 by omega), e0, hst]

/-- The session statements' message of a run's prover messages through round `stage + 1`. -/
theorem getD_stmts (m₀ m₁ m₂ : Msg) :
    ((m₀ :: m₁ :: List.replicate h.more ([] : Msg)) ++ [m₂]).getD (h.stage + 1) [] = m₂ := by
  have hs : h.stage = h.more + 1 := rfl
  have hl : (m₀ :: m₁ :: List.replicate h.more ([] : Msg)).length = h.stage + 1 := by simp [hs]
  rw [← hl]
  exact getD_concat _ _ _

/-- **A run's messages through round `stage + 1` are `seenAt`'s**, at the run's coins. -/
theorem msgs_seenAt {x : Msg} (r : Fin ((V h).k x) → FlockDeploy.Coin) (m₀ m₁ m₂ : Msg) (rest : List (Msg × Out)) :
    ((m₀, Out.msg (coinSeq r 0).bytes) :: (m₁, Out.msg (coinSeq r 1).bytes) ::
        (((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq r (i + 2)).bytes)) ++
          (m₂, Out.msg []) :: rest)) =
      (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs := by
  have hs : h.stage = h.more + 1 := rfl
  have e0 : coinSeq r 0 = r ⟨0, k_pos h x⟩ := dite_eq_left (k_pos h x)
  have h1 : coinSeq (stageCoins h x r) 0 = coinSeq r 1 := coinSeq_stageCoins h x r (Nat.succ_pos _)
  have hpadl : ((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq (stageCoins h x r) (i + 1)).bytes)) =
      (List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq r (i + 2)).bytes) :=
    List.map_congr_left fun i hi => by
      rw [coinSeq_stageCoins h x r (show i + 1 < h.stage by have := List.mem_range.1 hi; omega)]
  show _ = (m₀, Out.msg (r ⟨0, k_pos h x⟩).bytes) :: (m₁, Out.msg (coinSeq (stageCoins h x r) 0).bytes) ::
    (((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq (stageCoins h x r) (i + 1)).bytes)) ++
      (m₂, Out.msg []) :: rest)
  rw [h1, hpadl, ← e0]

/-- **An accepted run passed the session statements' check**: its transcript is `seenAt` of its messages of rounds 0, 1
and `stage + 1`, round 0's coin and the second stage's (`stageCoins`), and `stmtOk` held at them. -/
theorem run_seenAt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin) (hacc : (run H P (V h) x ω r).out = .acc) :
    ∃ m₀ m₁ m₂ rest, (run H P (V h) x ω r).seen = seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest ∧
      stmtOk h x m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) = true := by
  obtain ⟨m₀, m₁, m₂, rest, hm, hok⟩ := stepWith_stage H P x ω
    (fun as => (V h).next x as ((List.range as.length).map (coinSeq r))) (fun j => (coinSeq r j).bytes) h.more
    (fun as => stmtOk h x (as.getD 1 []) (as.getD (h.stage + 1) []) (r ⟨0, k_pos h x⟩) (stageCoins h x r))
    (next_run_one h r) (next_run_two h r) (next_run_pad h r) (next_run_stmt h r) ((V h).k x) hacc
  have hok' : stmtOk h x m₁ (((m₀ :: m₁ :: List.replicate h.more ([] : Msg)) ++ [m₂]).getD (h.stage + 1) [])
      (r ⟨0, k_pos h x⟩) (stageCoins h x r) = true := hok
  rw [getD_stmts h] at hok'
  refine ⟨m₀, m₁, m₂, rest, ?_, hok'⟩
  have hm' : (run H P (V h) x ω r).msgs =
      (m₀, Out.msg (coinSeq r 0).bytes) :: (m₁, Out.msg (coinSeq r 1).bytes) ::
        (((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq r (i + 2)).bytes)) ++
          (m₂, Out.msg []) :: rest) := hm
  show (⟨(run H P (V h) x ω r).msgs, (run H P (V h) x ω r).out⟩ : Seen) = _
  rw [hm', hacc]
  exact congrArg (fun l => (⟨l, .acc⟩ : Seen)) (msgs_seenAt h r m₀ m₁ m₂ rest)

/-- **Template `t`'s session statement as the verifier's check passes it is the template's reference one** (`pinOk`)
but for its sample, its registered reads and its public file. -/
theorem pin_of_stmtOkAt {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {t : ℕ}
    (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) :
    ∃ w w₀, FlockDeploy.decode xS = some w ∧ FlockDeploy.decode (h.xS₀ t) = some w₀ ∧
      { w with draw := none, registered := none, publicFile := .empty } =
        { w₀ with draw := none, registered := none, publicFile := .empty } := by
  unfold stmtOkAt at hok
  cases hs : parse h x with
  | none => simp [hs] at hok
  | some s =>
    simp only [hs, Bool.and_eq_true] at hok
    have hr := hok.2
    unfold readsOk at hr
    cases hd : decode x with
    | none => simp [hd] at hr
    | some st =>
      cases hm : jsonMsg m₁ with
      | none => simp [hd, hm] at hr
      | some rec =>
        cases hdr : secondDraw h c₀ c₁ with
        | none => simp [hd, hm, hdr] at hr
        | some draw =>
          cases hw : FlockDeploy.decode xS with
          | none => simp [hd, hm, hdr, hw] at hr
          | some w =>
            simp only [hd, hm, hdr, hw, Bool.and_eq_true] at hr
            have hpin := hr.2
            unfold pinOk at hpin
            cases hw₀ : FlockDeploy.decode (h.xS₀ t) with
            | none => simp [hw₀] at hpin
            | some w₀ =>
              simp only [hw₀, decide_eq_true_eq] at hpin
              exact ⟨w, w₀, rfl, rfl, hpin⟩

/-- **The checked units** of a run whose statement decodes: the second stage's on round 0's coin and the stage's. -/
theorem checked_eq (x : Msg) {st : Roots} (hst : decode x = some st) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) :
    (code h).checked x r =
      h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) := by
  have hr : reg h x = some h.reg := by simp [SpotFlock.reg, hst]
  show Option.getD (Option.map _ (reg h x)) [] = _
  rw [hr, Option.map_some, Option.getD_some]
  rfl

/-! ## The session statements' openings -/

/-- **Unit `u`'s session statement** as round `stage + 1`'s message `m₂` states it: its template's. -/
abbrev unitStmt (m₂ : Msg) (u : ℕ) : Msg := stmtAt h m₂ (templateOf h.ms u)

/-- **The session statements open every unit they check**: at a run's statement in the two-stage scope, session
statements `m₂` that the verifier's check passes at the round-1 message `m₁` and the coins `c₀` and `c₁`, and a Flock
witness in the relation of unit `u`'s statement, each unit `u` the two stages check on those coins has a proof block
`b` in a replay block `ρ` with gates `rb`, and any values `val` and interior values `ι` that agree with the openings the
witness reveals and that pass (under the run's roots, and the interior roots `m₁` states for `ρ`) make `b` correct on
`ρ`'s values. -/
def StmtOpens : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ replayOf Pt d.part u = some ρ ∧ d.part.blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ (code h).reveal s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF,
          (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn rb val ι) b

/-- **SPOT[Flock]'s session opens each checked proof block at its replay block**: `SessionOpensTwo` at SPOT[Flock]'s
two-stage code, from `StmtOpens`. -/
theorem sessionOpensTwo (hs : StmtOpens h) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots := by
  intro x s Pt d Ω P ω r u xS vS wF hx hP hd hin hacc hu hsess hck
  obtain ⟨m₀, m₁, m₂, rest, hseen, hok⟩ := run_seenAt h FlockDeploy.sha512 P x ω r hacc
  obtain ⟨ms, rest', hms, hlen, hses⟩ := (seq h _root_.Flock.Readable.relation).session_acc FlockDeploy.sha512 x P ω
    r u ((seq h _root_.Flock.Readable.relation).idx_lt x r u hu) hacc
  have hseen' : (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (code h)).V x ω r).seen =
      seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest := hseen
  have h3 : ((ms.take (h.stage + 2)).map Prod.fst).getD (h.stage + 1) [] = m₂ := by
    have hl : h.stage + 2 ≤ ms.length := hlen ▸ stage_le_off h _
    have hms' : (run FlockDeploy.sha512 P (V h) x ω r).msgs = ms ++ rest' := hms
    have hm : (run FlockDeploy.sha512 P (V h) x ω r).msgs =
        (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs := congrArg Seen.msgs hseen
    rw [← getD_fst_pre h ms rest' (by omega), ← hms', hm]
    exact seenAt_stmts h
  have hxS : xS = unitStmt h m₂ u := by
    have e : (deploy _root_.Flock.Readable.relation (code h)).session x u
        (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (code h)).V x ω r).seen =
          some (stmtAt h (((ms.take (h.stage + 2)).map Prod.fst).getD (h.stage + 1) []) (templateOf h.ms u),
            cutSeen [] (k₀ h (templateOf h.ms u)) rest') := hses
    rw [hsess, h3] at e
    exact (Prod.mk.inj (Option.some.inj e)).1
  rw [hxS] at hck
  rw [hseen'] at ⊢
  have hx' : parse h x = some s := hx
  obtain ⟨st, hst⟩ : ∃ st, decode x = some st := by
    cases hdx : decode x with
    | none => simp [SpotFlock.parse, hdx] at hx'
    | some st => exact ⟨st, rfl⟩
  obtain ⟨p, hp⟩ := hu
  have hum : u ∈ h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) :=
    (congrArg (u ∈ ·) (checked_eq h x hst r)).mp (List.mem_of_getElem? hp)
  exact hs x s Pt d m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest wF u hx hP hd hin hok hum hck

/-! ## The witness's values -/

theorem mem_checked_lt {g : RegTwo} {N : ℕ} {src₁ src₂ : ByteArray} {u : ℕ} (hu : u ∈ g.checked N src₁ src₂) :
    u < N := by
  unfold SamplingDeploy.RegTwo.checked at hu
  split at hu
  · exact List.mem_range.1 (List.mem_filter.1 hu).1
  · exact List.mem_range.1 hu

theorem units_base {x : Msg} {s : AuditStatement} {Pt : Partition} (hx : parse h x = some s)
    (hP : partition h s = some Pt) : units (base h) x = Pt.blocks.length := by
  have hx' : (base h).parse x = some s := hx
  have hP' : (base h).partition s = some Pt := hP
  unfold SamplingDeploy.units
  rw [hx']
  simp [hP']

/-- **A checked unit's template is a member's**: at a run's statement that parses, each unit the two stages check is
below the members' units. -/
theorem templateOf_lt_of_checked {x : Msg} {s : AuditStatement} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u : ℕ}
    (hx : parse h x = some s) (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁)) :
    templateOf h.ms u < h.ms.length := by
  have hN : units (base h) x = Properties.Spot.unitBase h.ms h.ms.length := by
    show (match parse h x with
      | some s => ((partition h s).getD ⟨[]⟩).blocks.length
      | none => 0) = _
    rw [hx]
    exact _root_.Flock.Bridge.Members.length_blocks_membersPart (ms := h.ms)
  exact templateOf_lt (hN ▸ mem_checked_lt hu)

/-- **The statement check holds at each checked unit's statement**, at its template (`stmtOkAt`). -/
theorem stmtOkAt_unit {x m₁ m₂ : Msg} {s : AuditStatement} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u : ℕ}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁)) :
    stmtOkAt h x m₁ (unitStmt h m₂ u) c₀ c₁ (templateOf h.ms u) = true :=
  stmtOkAt_of h hok (templateOf_lt_of_checked h hx hu)

theorem replay_block {Pt Pρ : Partition} {v ρ : ℕ} (hρ : replayOf Pt Pρ v = some ρ) :
    ∃ rb, Pρ.blocks[ρ]? = some rb :=
  ⟨_, List.getElem?_eq_getElem (List.mem_range.1 (List.mem_of_find?_eq_some hρ))⟩

/-- **The session statement's witness holds values that make each checked unit correct**: at a run's statement in the
two-stage scope, session statements `m₂` that the verifier's check passes and a Flock witness in the relation of unit
`u`'s statement, each unit `u` the two stages check, with proof block `b` in replay block `ρ` of gates `rb`, has
boundary values `W` and interior values `I` that make `b` correct on `ρ`'s values, and the witness reveals each value
`b`'s correctness needs by an opening that passes: one of `ρ`'s interior under the interior roots `m₁` states for `ρ`,
every other under the run's roots. -/
def WitOpens : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u b ρ rb,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    ∃ (W : ℕ → Option Bool) (I : ℕ → Bool), s.prog.Correct (s.prog.refIn rb W I) b ∧
      ∀ g, s.prog.Needs b g →
        (g ∈ rb ∧ s.prog.Boundary rb g = false →
          ∃ q ∈ (code h).reveal s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF,
            (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧
              q.1 = iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ ∧ q.2.pos = g ∧ q.2.val = I g) ∧
        (¬(g ∈ rb ∧ s.prog.Boundary rb g = false) →
          ∃ q ∈ (code h).reveal s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF,
            (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧
              q.1 = s.roots ∧ q.2.pos = g ∧ W g = some q.2.val)

/-- **`StmtOpens` from the witness's values**: values that agree with the openings that pass agree with `W` and `I`
wherever the proof block's correctness reads them. -/
theorem stmtOpens (hw : WitOpens h) : StmtOpens h := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u hx hP hd hin hok hu hck
  have hlt : u < Pt.blocks.length := by
    have := mem_checked_lt hu
    rwa [units_base h hx hP] at this
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u hlt)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  refine ⟨Pt.blocks[u], ρ, rb, List.getElem?_eq_getElem hlt, hρ, hrb, fun val ι hv => ?_⟩
  obtain ⟨W, I, hc, hcov⟩ := hw x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u _ ρ rb hx hP hd hin hok hu hck
    (List.getElem?_eq_getElem hlt) hρ hrb
  refine s.prog.correct_of_needs (fun g hg => ?_) hc
  unfold Prog.refIn
  by_cases hi : g ∈ rb ∧ s.prog.Boundary rb g = false
  · obtain ⟨q, hq, hop, hr, hpos, hval⟩ := (hcov g hg).1 hi
    rw [ite_eq_left hi, ite_eq_left hi, ← hval, ← hpos, (hv q hq hop).2 hr]
  · obtain ⟨q, hq, hop, hr, hpos, hval⟩ := (hcov g hg).2 hi
    rw [ite_eq_right hi, ite_eq_right hi, hval, ← hpos, (hv q hq hop).1 hr]

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from the witness's values.** -/
theorem sessionOpensTwo_of_wit (hw : WitOpens h) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo h (stmtOpens h hw)

/-! ## The witness's values, decoded -/

/-- **A decoder of the session witness**: `dec x m₁ xS wF u g`, the value at the statement's gate `g` that the session
statement `xS` (unit `u`'s) and its Flock witness `wF` hold for the checked unit `u`, at the run's statement `x` and
interior registration `m₁`. It is per unit: two checked units that read one gate need not read one value in a witness
Flock's relation accepts, since only the commitment's binding ties them. -/
abbrev Dec : Type := Msg → Msg → Msg → Claims.Flock.Wit → ℕ → ℕ → Bool

/-- **The decoded values make each checked unit correct**: at session statements the verifier's check passes and a
Flock witness in the relation of a unit's statement, each unit the two stages check is correct on its replay block's
values with the decoded values for both the committed and the interior ones. -/
def Decodes (dec : Dec) : Prop :=
  ∀ x s Pt d (m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) wF u b ρ rb,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    s.prog.Correct (s.prog.refIn rb (fun g => some (dec x m₁ (unitStmt h m₂ u) wF u g))
      (dec x m₁ (unitStmt h m₂ u) wF u)) b

/-- **The decoded values evaluate each checked unit**: at the same statements as `Decodes`, each gate of the unit but
its inputs holds, in the decoded values, its value on the decoded values of the gates it reads, and each constant
gate's decoded value is the constant. -/
def Evaluates (dec : Dec) : Prop :=
  ∀ x s Pt d (m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) wF u b ρ rb,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    (∀ g ∈ b, ∀ y : BGate, s.prog.gates[g]? = some y → y ≠ .input →
      y.val (fun i => some (dec x m₁ (unitStmt h m₂ u) wF u i)) none = some (dec x m₁ (unitStmt h m₂ u) wF u g)) ∧
    ∀ g c, s.prog.gates[g]? = some (.const c) → dec x m₁ (unitStmt h m₂ u) wF u g = c

/-- **`Decodes` from an evaluation**: the values a decoder gives every gate, committed or interior, make a unit correct
where they evaluate its gates (`Prog.correct_of_eval`). -/
theorem decodes_of_evaluates (dec : Dec) (he : Evaluates h dec) : Decodes h dec := by
  intro x s Pt d m₁ m₂ c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  obtain ⟨hv, hc⟩ := he x s Pt d m₁ m₂ c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  exact s.prog.correct_of_eval (v := dec x m₁ (unitStmt h m₂ u) wF u) hin.1 hv hc fun g _ => by
    simp only [Prog.refIn, ite_self]

/-- **The reveal opens each decoded value a checked unit's correctness reads**: at the same statements, for each gate
`Prog.Needs` names, the reveal holds an opening of the decoded value at that gate that passes, under the interior roots
`m₁` states for the replay block where the gate is one of its interior gates, and under the run's roots otherwise. -/
def Reveals (dec : Dec) : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u b ρ rb,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    ∀ g, s.prog.Needs b g →
      ∃ q ∈ (code h).reveal s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF,
        (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧ q.2.pos = g ∧
          q.2.val = dec x m₁ (unitStmt h m₂ u) wF u g ∧
          q.1 = if g ∈ rb ∧ s.prog.Boundary rb g = false then iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ else s.roots

/-- **`WitOpens` from a decoder**: the decoded values are the witness's values, committed and interior. -/
theorem witOpens (dec : Dec) (hd : Decodes h dec) (hr : Reveals h dec) : WitOpens h := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  refine ⟨fun g => some (dec x m₁ (unitStmt h m₂ u) wF u g), dec x m₁ (unitStmt h m₂ u) wF u,
    hd x s Pt d m₁ m₂ c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb, fun g hg => ⟨fun hi => ?_, fun hi => ?_⟩⟩
  · obtain ⟨q, hq, hop, hpos, hval, hroot⟩ :=
      hr x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb g hg
    exact ⟨q, hq, hop, hroot.trans (ite_eq_left hi), hpos, hval⟩
  · obtain ⟨q, hq, hop, hpos, hval, hroot⟩ :=
      hr x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb g hg
    exact ⟨q, hq, hop, hroot.trans (ite_eq_right hi), hpos, congrArg some hval.symm⟩

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a decoder** that `Decodes` and `Reveals`. -/
theorem sessionOpensTwo_of_dec (dec : Dec) (hd : Decodes h dec) (hr : Reveals h dec) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_wit h (witOpens h dec hd hr)

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from a decoder** whose values evaluate each checked unit
(`Evaluates`) and that the reveal opens (`Reveals`). -/
theorem sessionOpensTwo_of_eval (dec : Dec) (he : Evaluates h dec) (hr : Reveals h dec) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_dec h dec (decodes_of_evaluates h dec he) hr

/-- **SPOT[Flock]'s two-stage knowledge soundness at its code from a decoder**: `ksTwo_spot`, with the sessions'
openings from a decoder whose values evaluate each checked unit (`Evaluates`) and that the reveal opens (`Reveals`). -/
theorem ksTwo_spot_of_eval (dec : Dec) (he : Evaluates h dec) (hr : Reveals h dec) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h) (rs h)
      (cap h)).KS FlockDeploy.sha512 :=
  ksTwo_spot h (sessionOpensTwo_of_eval h dec he hr)

end SpotFlock
