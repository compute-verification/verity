module

public import Proofs.Flock.SpotFlock.Frame
public import Proofs.Flock.SpotFlock.FrameIsBuilt
public import Proofs.Flock.SpotFlock.KS
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.Flock.SamplingGeneric.TwoCount
public import Proofs.Flock.Claims.ERep

/-!
# A checked proof block's extraction opens every gate it needs, or none

At SPOT[Flock]'s two-stage code, the extraction of a checked proof block's session keeps, opening by opening, what its
witness reveals that passes the commitment's check (`Compiled.sessExt`, `checkRevealed`). This file shows that at the
code the openings of one reveal pass together: at every run, accepted or refused, a witness the session's extraction
returns reveals only openings that pass. So a slot's extraction keeps its whole reveal or keeps nothing, and the
gates one run's extractions open are a union of whole reveals.

The wiring has two halves, each read at a run with no hypothesis that the outer verifier accepted it.

* **The extractor's half** (`Arg.ext_some`, `Compiled.sessWit_some`), for any compiled verifier whose sessions run as
  the contract's `runs` says (`Compiled.RunsFixed`): a witness `sessWit` returns comes from a session view that
  accepted, and passes the proof system's `check` at the session's statement. `E^rep` replays the judged run and
  waits only where it accepted, and its waits return only a candidate `check` accepts (`firstWait_some`); the
  session's view is a run of the proof system's verifier against the session's prover at every run of the outer
  verifier (`served_eval`).
* **The code's half** (`session_acc_msg`, `run_msgs_of_answered`, `session_acc_run`): a session whose view accepted
  is not the refused one, so every round before the session's was answered with a message; round `stage + 1` is
  answered only where the session statements' check `stmtOk` passed, and the session's statement is the checked
  unit's.

Those are the premises of `Reveals` (the decoded values' openings pass at every gate the proof block needs), which the
frame's construction gives under `Built` and `Addresses`. `Reveals` is stated at an accepted transcript (`seenAt`);
the code's reveal reads only a transcript's messages under `Built` (`reveal_out`), so it applies at a refused run too.

* `reveal_passes`: at a checked unit whose session's view accepted and a witness `check` accepts at its statement,
  every opening of the code's reveal passes.
* `sessWit_reveal_passes`: the two halves composed at the compiled two-stage verifier. Over every transcript and
  witness the statement fails: the commitment's check places the opening's value in the row and never compares it
  with the row's own bit there, so only a witness the relation accepts ties the two.

So the gates one reveal opens are opened together. Which openings of it share their roots is the class:

* `gateKey`: a gate's key, the proof blocks that need it opened (the block that writes it among them), the replay
  block that holds it, and whether it is on that block's boundary. Gates with one key are needed by the same units
  (`gateKey_needs`), lie in one replay block (`gateKey_block`) and on one side of it (`gateKey_side`), so each unit's
  reveal opens them under the same roots (`Reveal.rootsAt_same_side`).
* `sessWit_reveal_key`: an opening of what an extracted witness reveals at a checked unit has, at every gate of its
  gate's key, an opening of the same reveal under the same roots that passes.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Inhabited Coin] in
/-- **The extractor returns only a witness `check` accepts, from an accepted judged run**: `E^rep` waits only where
the judged run accepted, and its waits return only a candidate in the relation's check. -/
theorem Arg.ext_some (H : Msg → Msg) (xS : Msg) (run' : (Fin (A.V.k xS) → Coin) → Transcript)
    (d : Fin (A.V.k xS) → Coin) (c : A.EC (A.shape xS)) {w : A.W}
    (hw : ((A.ext xS d c).eval H run').1 = some w) : (run' d).seen.out = .acc ∧ A.check xS w = true := by
  by_cases hacc : (run' d).seen.out = .acc
  · refine ⟨hacc, ?_⟩
    simp only [Arg.ext, ERep, RP.eval, hacc, ↓reduceIte] at hw
    exact firstWait_some H run' _ _ _ _ _ w hw
  · simp [Arg.ext, ERep, RP.eval, hacc] at hw

/-- **A session's witness comes from an accepted view and passes `check`**, at every run: where the sessions run as
the contract's `runs` says, a witness `sessWit` returns at a checked proof block `u` is one the proof system's check
accepts at the session's statement, and the session's view at the run accepted. The run itself may be refused. -/
theorem Compiled.sessWit_some {H : Msg → Msg} {L : Compiled Coin A} (hS : L.RunsFixed H) {x : Msg} {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin) {u : ℕ} (hu : u ∈ L.checked x r) (c : A.EC (L.sh x u))
    {w : A.W} (hw : ((L.sessWit x u r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).1 = some w) :
    ∃ xS vS, L.session x u (run H P L.V x ω r).seen = some (xS, vS) ∧ vS.out = .acc ∧ A.check xS w = true := by
  obtain ⟨xS, P', hshape, -, -, hcard, -, hview⟩ := hS.runs x Ω P ω r u hu
  have hsess := (hview r fun _ _ _ => rfl).1
  refine ⟨xS, _, hsess, ?_⟩
  unfold Compiled.sessWit at hw
  simp only [hsess] at hw
  rw [dite_eq_left hshape, L.served_eval H P ω P' r r hcard (fun r' hr' => (hview r' hr').1) fun _ _ => rfl] at hw
  exact Arg.ext_some H xS _ _ _ hw

end Claims.SamplingGeneric

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-! ## A run whose session statements' round was answered -/

/-- A transcript whose last round was refused has no message answered there or after. -/
theorem getElem?_concat_rej {l : List (Msg × Out)} {a m y : Msg} {i : ℕ} (hi : l.length ≤ i)
    (h2 : (l ++ [(a, Out.rej)])[i]? = some (m, Out.msg y)) : False := by
  rw [List.getElem?_append_right hi] at h2
  cases hd : i - l.length with
  | zero =>
    rw [hd] at h2
    simp at h2
  | succ j =>
    rw [hd] at h2
    simp at h2

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)

/-- **The later rounds of a second stage, at a run that need not accept**: `stepWith_pad` with the round after the
stage's answered with a message in place of the run's acceptance. -/
theorem stepWith_pad_msg (c : ℕ → Msg) (n : ℕ) (ok : List Msg → Bool) (a₀ a₁ : Msg)
    (hpad : ∀ (as : List Msg) (a : Msg), 2 ≤ as.length → as.length ≤ n + 1 →
      (a = [] ∧ vn (as ++ [a]) = .ret (.msg (c as.length))) ∨ vn (as ++ [a]) = .ret .rej)
    (hstmt : ∀ as : List Msg, as.length = n + 3 → vn as = .ret (if ok as then .msg [] else .rej)) (m y : Msg) :
    ∀ (d k : ℕ), k + d = n → ∀ (f : ℕ) (hist : List (Msg × Msg)) (t : Transcript),
      (stepWith H P x ω vn f hist (a₀ :: a₁ :: List.replicate k []) t).msgs[n + 2]? = some (m, .msg y) →
      t.msgs = (a₀, .msg (c 0)) :: (a₁, .msg (c 1)) ::
        (List.range k).map (fun i => (([] : Msg), Out.msg (c (i + 2)))) →
      ∃ m₂ rest, (stepWith H P x ω vn f hist (a₀ :: a₁ :: List.replicate k []) t).msgs =
          (a₀, .msg (c 0)) :: (a₁, .msg (c 1)) ::
            ((List.range n).map (fun i => (([] : Msg), Out.msg (c (i + 2)))) ++ (m₂, .msg []) :: rest) ∧
        ok ((a₀ :: a₁ :: List.replicate n []) ++ [m₂]) = true := by
  intro d
  induction d with
  | zero =>
    intro k hk f hist t h2 hmsgs
    have hkn : n = k := by omega
    subst hkn
    have hlen : t.msgs.length = n + 2 := by rw [hmsgs]; simp
    cases f with
    | zero =>
      have hnone : t.msgs[n + 2]? = none := List.getElem?_eq_none (by omega)
      simp only [stepWith] at h2
      rw [hnone] at h2
      cases h2
    | succ g =>
      have hl : ((a₀ :: a₁ :: List.replicate n ([] : Msg)) ++ [((P.next x hist ω).eval H).1]).length = n + 3 := by
        simp
      cases hok : ok ((a₀ :: a₁ :: List.replicate n []) ++ [((P.next x hist ω).eval H).1]) with
      | false =>
        have e : (vn ((a₀ :: a₁ :: List.replicate n []) ++ [((P.next x hist ω).eval H).1])).eval H = (.rej, []) := by
          rw [hstmt _ hl, hok]; rfl
        simp only [stepWith, e] at h2
        exact (getElem?_concat_rej (by omega) h2).elim
      | true =>
        rw [stepWith_succ_msg H P x ω vn (m := []) (lo := []) (by rw [hstmt _ hl, hok]; rfl)]
        obtain ⟨rest, hr⟩ := Claims.Sampling.stepWith_msgs_prefix H P x ω vn g _ _ _
        refine ⟨((P.next x hist ω).eval H).1, rest, ?_, hok⟩
        rw [hr, hmsgs]
        simp
  | succ d ih =>
    intro k hk f hist t h2 hmsgs
    have hlen : t.msgs.length = k + 2 := by rw [hmsgs]; simp
    cases f with
    | zero =>
      have hnone : t.msgs[n + 2]? = none := List.getElem?_eq_none (by omega)
      simp only [stepWith] at h2
      rw [hnone] at h2
      cases h2
    | succ g =>
      have hl : (a₀ :: a₁ :: List.replicate k ([] : Msg)).length = k + 2 := by simp
      rcases hpad (a₀ :: a₁ :: List.replicate k []) ((P.next x hist ω).eval H).1 (by rw [hl]; omega)
        (by rw [hl]; omega) with ⟨ha, hv⟩ | hv
      · rw [hl] at hv
        rw [stepWith_succ_msg H P x ω vn (m := c (k + 2)) (lo := []) (by rw [hv]; rfl)] at h2 ⊢
        have has : (a₀ :: a₁ :: List.replicate k ([] : Msg)) ++ [((P.next x hist ω).eval H).1] =
            a₀ :: a₁ :: List.replicate (k + 1) [] := by
          rw [ha]
          exact (congrArg (fun l => a₀ :: a₁ :: l) List.replicate_succ').symm
        rw [has] at h2 ⊢
        refine ih (k + 1) (by omega) g _ _ h2 ?_
        simp [hmsgs, List.range_succ, ha]
      · have e : (vn ((a₀ :: a₁ :: List.replicate k []) ++ [((P.next x hist ω).eval H).1])).eval H = (.rej, []) := by
          rw [hv]; rfl
        simp only [stepWith, e] at h2
        exact (getElem?_concat_rej (by omega) h2).elim

/-- **A run whose verifier answers its first two rounds, then `n` rounds where the prover sent nothing, and checks the
next**, and whose round after those was answered with a message, answered it with nothing at a check that passed.
The run need not accept. -/
theorem stepWith_stage_msg (c : ℕ → Msg) (n : ℕ) (ok : List Msg → Bool) (hv1 : ∀ a, vn [a] = .ret (.msg (c 0)))
    (hv2 : ∀ a b, vn [a, b] = .ret (.msg (c 1)))
    (hpad : ∀ (as : List Msg) (a : Msg), 2 ≤ as.length → as.length ≤ n + 1 →
      (a = [] ∧ vn (as ++ [a]) = .ret (.msg (c as.length))) ∨ vn (as ++ [a]) = .ret .rej)
    (hstmt : ∀ as : List Msg, as.length = n + 3 → vn as = .ret (if ok as then .msg [] else .rej)) :
    ∀ (f : ℕ) (m y : Msg),
      (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).msgs[n + 2]? = some (m, .msg y) →
      ∃ m₀ m₁ m₂ rest, (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).msgs =
          (m₀, .msg (c 0)) :: (m₁, .msg (c 1)) ::
            ((List.range n).map (fun i => (([] : Msg), Out.msg (c (i + 2)))) ++ (m₂, .msg []) :: rest) ∧
        ok ((m₀ :: m₁ :: List.replicate n []) ++ [m₂]) = true
  | 0, _, _, h2 => by simp [stepWith] at h2
  | 1, _, _, h2 => by
    rw [stepWith_succ_msg H P x ω vn (m := c 0) (lo := []) (by rw [List.nil_append, hv1]; rfl)] at h2
    simp [stepWith] at h2
  | g + 1 + 1, m, y, h2 => by
    rw [stepWith_succ_msg H P x ω vn (m := c 0) (lo := []) (by rw [List.nil_append, hv1]; rfl)] at h2 ⊢
    simp only [List.nil_append] at h2 ⊢
    rw [stepWith_succ_msg H P x ω vn (m := c 1) (lo := []) (by rw [List.singleton_append, hv2]; rfl)] at h2 ⊢
    simp only [List.cons_append, List.nil_append] at h2 ⊢
    obtain ⟨m₂, rest, hm, hok⟩ :=
      stepWith_pad_msg H P x ω vn c n ok _ _ hpad hstmt m y n 0 (Nat.zero_add n) g _ _ h2 rfl
    exact ⟨_, _, m₂, rest, hm, hok⟩

end Steps

/-- **Rounds answered through the `n`-th**: a transcript's messages of which at least `n` are there, the first `n`
answered with a message, have each of those answered with a message. -/
theorem answered_at {l : List (Msg × Out)} {n i : ℕ} (hi : i < n) (hlen : n ≤ l.length)
    (hall : (l.take n).all (fun p => answered p.2) = true) : ∃ m y, l[i]? = some (m, Out.msg y) := by
  have hil : i < l.length := by omega
  have hmem : l[i] ∈ l.take n := List.mem_take_iff_getElem.2 ⟨i, by omega, rfl⟩
  have ha : answered l[i].2 = true := List.all_eq_true.1 hall _ hmem
  cases ho : l[i].2 with
  | msg y =>
    have e : l[i] = (l[i].1, Out.msg y) := Prod.ext rfl ho
    exact ⟨l[i].1, y, (List.getElem?_eq_getElem hil).trans (congrArg some e)⟩
  | acc =>
    rw [ho] at ha
    simp [answered] at ha
  | rej =>
    rw [ho] at ha
    simp [answered] at ha

variable (h : Held)

/-- **A run whose round `stage + 1` was answered passed the session statements' check**: its messages are `seenAt`'s,
rounds 0 to `stage` answered with their coins and round `stage + 1` with nothing, and `stmtOk` held at them.
`run_seenAt` without the run's acceptance. -/
theorem run_msgs_of_answered {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin) {m y : Msg}
    (h2 : (run H P (V h) x ω r).msgs[h.stage + 1]? = some (m, .msg y)) :
    ∃ m₀ m₁ m₂ rest, (run H P (V h) x ω r).msgs =
        (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs ∧
      stmtOk h x m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) = true := by
  have h2' : (run H P (V h) x ω r).msgs[h.more + 2]? = some (m, .msg y) := h2
  obtain ⟨m₀, m₁, m₂, rest, hm, hok⟩ := stepWith_stage_msg H P x ω
    (fun as => (V h).next x as ((List.range as.length).map (coinSeq r))) (fun j => (coinSeq r j).bytes) h.more
    (fun as => stmtOk h x (as.getD 1 []) (as.getD (h.stage + 1) []) (r ⟨0, k_pos h x⟩) (stageCoins h x r))
    (next_run_one h r) (next_run_two h r) (next_run_pad h r) (next_run_stmt h r) ((V h).k x) m y h2'
  have hok' : stmtOk h x m₁ (((m₀ :: m₁ :: List.replicate h.more ([] : Msg)) ++ [m₂]).getD (h.stage + 1) [])
      (r ⟨0, k_pos h x⟩) (stageCoins h x r) = true := hok
  rw [getD_stmts h] at hok'
  have hm' : (run H P (V h) x ω r).msgs =
      (m₀, Out.msg (coinSeq r 0).bytes) :: (m₁, Out.msg (coinSeq r 1).bytes) ::
        (((List.range h.more).map fun i => (([] : Msg), Out.msg (coinSeq r (i + 2)).bytes)) ++
          (m₂, Out.msg []) :: rest) := hm
  exact ⟨m₀, m₁, m₂, rest, hm'.trans (msgs_seenAt h r m₀ m₁ m₂ rest), hok'⟩

/-- **A session whose view accepted is not the refused one**: the transcript's round `stage + 1` was answered with a
message, and the session's statement is the unit's template's as that round states it. -/
theorem session_acc_msg {u : ℕ} {v : Seen} {xS : Msg} {vS : Seen} (hsess : session h u v = some (xS, vS))
    (hacc : vS.out = .acc) :
    (∃ m y, v.msgs[h.stage + 1]? = some (m, Out.msg y)) ∧
      xS = stmtAt h (v.msgs.getD (h.stage + 1) ([], .rej)).1 (templateOf h.ms u) := by
  by_cases hans : off h (templateOf h.ms u) ≤ v.msgs.length ∧
      (v.msgs.take (off h (templateOf h.ms u))).all (answered ·.2) = true
  · by_cases hok : sessionOk h (templateOf h.ms u)
        (stmtAt h (v.msgs.getD (h.stage + 1) ([], .rej)).1 (templateOf h.ms u)) = true
    · rw [session, ite_eq_left hans, ite_eq_left hok] at hsess
      have e := Prod.mk.inj (Option.some.inj hsess)
      have hlt : h.stage + 1 < off h (templateOf h.ms u) := by
        have := stage_le_off h (templateOf h.ms u)
        omega
      exact ⟨answered_at hlt hans.1 hans.2, e.1.symm⟩
    · rw [session, ite_eq_left hans, ite_eq_right hok] at hsess
      have e := Prod.mk.inj (Option.some.inj hsess)
      rw [← e.2] at hacc
      simp [refusedSeen] at hacc
  · rw [session, ite_eq_right hans] at hsess
    have e := Prod.mk.inj (Option.some.inj hsess)
    rw [← e.2] at hacc
    simp [refusedSeen] at hacc

/-- **The code's half of the wiring**: at any run, a unit's session whose view accepted has the run's rounds 0 to
`stage + 1` answered, the session statements' check passed at them, and its statement the unit's. -/
theorem session_acc_run {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ} {xS : Msg} {vS : Seen}
    (hsess : session h u (run H P (V h) x ω r).seen = some (xS, vS)) (hacc : vS.out = .acc) :
    ∃ m₀ m₁ m₂ rest, (run H P (V h) x ω r).msgs =
        (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs ∧
      stmtOk h x m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) = true ∧ xS = unitStmt h m₂ u := by
  obtain ⟨⟨m, y, h2⟩, hxS⟩ := session_acc_msg h hsess hacc
  obtain ⟨m₀, m₁, m₂, rest, hm, hok⟩ := run_msgs_of_answered h H P x ω r h2
  refine ⟨m₀, m₁, m₂, rest, hm, hok, ?_⟩
  have hm' : (run H P (V h) x ω r).seen.msgs =
      (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs := hm
  have hg : ((run H P (V h) x ω r).seen.msgs.getD (h.stage + 1) ([], .rej)).1 = m₂ := by
    rw [hm']
    exact seenAt_stmts h
  exact hxS.trans (congrArg (fun a => stmtAt h a (templateOf h.ms u)) hg)

/-! ## The code's reveal reads a transcript's messages -/

/-- **The reveal reads only the transcript's messages**, at a registration whose frame is SPOT's own: the roots read
the round-1 message and the opener rounds 1 and `stage + 1`, so the reveal at a refused transcript is the one at an
accepted transcript of the same messages. -/
theorem reveal_out (hB : Built h) (s : AuditStatement) (u : ℕ) (ms : List (Msg × Out)) (o o' : Out)
    (wF : Claims.Flock.Wit) : (code h).reveal s u ⟨ms, o⟩ wF = (code h).reveal s u ⟨ms, o'⟩ wF := by
  obtain ⟨c₀, hc⟩ := hB
  have hroots : ∀ g, Reveal.rootsAt (partition h) (two h) (iroots []) s u ⟨ms, o⟩ g =
      Reveal.rootsAt (partition h) (two h) (iroots []) s u ⟨ms, o'⟩ g := fun _ => rfl
  have hop : ∀ g R, addressOp h (addressMap h) s u ⟨ms, o⟩ wF g R =
      addressOp h (addressMap h) s u ⟨ms, o'⟩ wF g R := fun _ _ => rfl
  have hinner : h.frame.reveal s u ⟨ms, o⟩ wF = h.frame.reveal s u ⟨ms, o'⟩ wF := by
    rw [hc]
    show (Reveal.needs (partition h) s u).map
        (Reveal.openAt (partition h) (two h) (iroots []) (addressOp h (addressMap h)) s u ⟨ms, o⟩ wF) =
      (Reveal.needs (partition h) s u).map
        (Reveal.openAt (partition h) (two h) (iroots []) (addressOp h (addressMap h)) s u ⟨ms, o'⟩ wF)
    refine List.map_congr_left fun g _ => ?_
    exact congrArg (fun op' : Opening Bool =>
      (Reveal.rootsAt (partition h) (two h) (iroots []) s u ⟨ms, o'⟩ g, ({ op' with pos := g } : Opening Bool)))
      (hop g (Reveal.rootsAt (partition h) (two h) (iroots []) s u ⟨ms, o'⟩ g))
  have hfr : ∀ g R, Reveal.frameAt h.frame s u ⟨ms, o⟩ wF g R = Reveal.frameAt h.frame s u ⟨ms, o'⟩ wF g R := by
    intro g R
    unfold Reveal.frameAt
    rw [hinner]
  show (Reveal.needs (partition h) s u).map
      (Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u ⟨ms, o⟩ wF) =
    (Reveal.needs (partition h) s u).map
      (Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u ⟨ms, o'⟩ wF)
  refine List.map_congr_left fun g _ => ?_
  exact congrArg (fun op' : Opening Bool =>
    (Reveal.rootsAt (partition h) (two h) (iroots []) s u ⟨ms, o'⟩ g, ({ op' with pos := g } : Opening Bool)))
    (hfr g (Reveal.rootsAt (partition h) (two h) (iroots []) s u ⟨ms, o'⟩ g))

/-! ## Every opening of a checked unit's reveal passes -/

/-- **A checked unit's reveal passes whole**, at every run: under `Built` and `Addresses`, at a statement in the
two-stage scope, a unit `u` the two stages check at the run's coins, whose session's view at the run accepted, and a
witness the proof system's check accepts at the session's statement, every opening the code reveals for `u` passes
the commitment's check. The run may be refused: the premises are `Reveals`', which `session_acc_run` gives. -/
theorem reveal_passes (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt) (hd : two h s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (code h).checked x r) {xS : Msg} {vS : Seen}
    (hsess : session h u (run FlockDeploy.sha512 P (V h) x ω r).seen = some (xS, vS)) (hacc : vS.out = .acc)
    {wF : Claims.Flock.Wit}
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true) :
    ∀ q ∈ (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen wF,
      (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true := by
  intro q hq
  obtain ⟨m₀, m₁, m₂, rest, hm, hok, hxS⟩ := session_acc_run h FlockDeploy.sha512 P x ω r hsess hacc
  rw [hxS] at hck
  obtain ⟨st, hst⟩ : ∃ st, decode x = some st := by
    cases hdx : decode x with
    | none => simp [SpotFlock.parse, hdx] at hx
    | some st => exact ⟨st, rfl⟩
  have hum : u ∈ h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (stageSrc (stageCoins h x r)) :=
    (congrArg (u ∈ ·) (checked_eq h x hst r)).mp hu
  have hlt : u < Pt.blocks.length := by
    have := mem_checked_lt hum
    rwa [units_base h hx hP] at this
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u hlt)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  have hb : Pt.blocks[u]? = some Pt.blocks[u] := List.getElem?_eq_getElem hlt
  have hseen : (run FlockDeploy.sha512 P (V h) x ω r).seen =
      ⟨(seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest).msgs,
        (run FlockDeploy.sha512 P (V h) x ω r).out⟩ :=
    congrArg (fun l => (⟨l, (run FlockDeploy.sha512 P (V h) x ω r).out⟩ : Seen)) hm
  have hrev : (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen wF =
      (code h).reveal s u (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest) wF :=
    (congrArg (fun v => (code h).reveal s u v wF) hseen).trans (reveal_out h hB s u _ _ .acc wF)
  rw [hrev] at hq
  obtain ⟨g, hg, hgq⟩ :=
    (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).1 hq
  obtain ⟨q', hq', hop, hpos, -, -⟩ :=
    reveals_of_frameRows h (layoutFor h) (frameRows_of_frameIs h (layoutFor h) (frameIs_built h hB hS))
      x s Pt d m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (stageCoins h x r) rest wF u _ ρ rb hx hP hd hin hok hum hck hb hρ
      hrb g ((Reveal.mem_needs (partition h) hP hb).1 hg)
  obtain ⟨g', -, hgq'⟩ :=
    (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).1 hq'
  have hgg : g' = g := by
    rw [← hgq'] at hpos
    exact hpos
  rw [hgg] at hgq'
  have hqq : q = q' := hgq.symm.trans hgq'
  rw [hqq]
  exact hop

/-! ## The two halves, at the compiled verifier -/

/-- **A slot's extraction keeps its whole reveal or nothing**: at SPOT[Flock]'s compiled two-stage verifier, under
`Built` and `Addresses`, at a statement in the two-stage scope and every run, a witness the session's extraction
(`sessWit`) returns at a checked unit `u` reveals only openings that pass the commitment's check. -/
theorem sessWit_reveal_passes (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt) (hd : two h s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).toCompiled.checked x r)
    (c : (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)).EC
      ((SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
        (two h) (rs h) (cap h)).toCompiled.sh x u))
    {w : Claims.Flock.Wit}
    (hw : (((SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).toCompiled.sessWit x u r
        (run FlockDeploy.sha512 P (V h) x ω r).seen c).eval FlockDeploy.sha512
          (run FlockDeploy.sha512 P (V h) x ω)).1 = some w) :
    ∀ q ∈ (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen w,
      (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true := by
  have hSess := sessionsTwo_spot h
    (sessionOpensTwo_of_frameIs h (layoutFor h) (embeds_layoutFor h) (frameIs_built h hB hS))
  have hRF : (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).toCompiled.RunsFixed FlockDeploy.sha512 := ⟨hSess.fixed, hSess.runs⟩
  obtain ⟨xS, vS, hsess, hacc, hck⟩ := Claims.SamplingGeneric.Compiled.sessWit_some hRF P ω r hu c hw
  exact reveal_passes h hB hS hx hP hd hin P ω r (List.mem_of_mem_take hu) hsess hacc hck

/-! ## A gate's key -/

/-- A unit that needs a gate opened is one of the partition's. -/
theorem Reveal.lt_of_mem_needs (part : AuditStatement → Option Partition) {s : AuditStatement} {P : Partition}
    {u g : ℕ} (hP : part s = some P) (hg : g ∈ Reveal.needs part s u) : u < P.blocks.length := by
  cases hb : P.blocks[u]? with
  | none => simp [Reveal.needs, hP, hb] at hg
  | some b => exact (List.getElem?_eq_some_iff.1 hb).1

/-- **Gates on the same side of the unit's replay block's boundary open under the same roots**, at every
transcript: both under the block's interior roots, or both under the statement's. -/
theorem Reveal.rootsAt_same_side (part : AuditStatement → Option Partition)
    (two : AuditStatement → Option Design.Two) (ir : Seen → ℕ → List Msg) {s : AuditStatement} {u ρ g g' : ℕ}
    {v : Seen} {P : Partition} {d : Design.Two} {rb : List ℕ} (hP : part s = some P) (hd : two s = some d)
    (hρ : replayOf P d.part u = some ρ) (hrb : d.part.blocks[ρ]? = some rb)
    (hside : (g ∈ rb ∧ s.prog.Boundary rb g = false) ↔ (g' ∈ rb ∧ s.prog.Boundary rb g' = false)) :
    Reveal.rootsAt part two ir s u v g = Reveal.rootsAt part two ir s u v g' := by
  rw [Reveal.rootsAt_eq part two ir hP hd hρ hrb (g := g), Reveal.rootsAt_eq part two ir hP hd hρ hrb (g := g')]
  exact if_congr hside rfl rfl

/-- **A gate's key** at the statement's partition `Pt` and two-stage design `d`: the proof blocks that need it opened
(`Reveal.needs`: its readers, and the block that writes it where it leaves that block), the replay block that holds it
(`gateBlock`), and whether it is on that replay block's boundary. -/
noncomputable def gateKey (s : AuditStatement) (Pt : Partition) (d : Design.Two) (g : ℕ) :
    Finset ℕ × Option ℕ × Bool :=
  ((Finset.range Pt.blocks.length).filter fun v => g ∈ Reveal.needs (partition h) s v, gateBlock d.part g,
    (gateBlock d.part g).any fun ρ => s.prog.Boundary (d.part.blocks.getD ρ []) g)

/-- Gates with one key are needed by the same units. -/
theorem gateKey_needs {s : AuditStatement} {Pt : Partition} (hP : partition h s = some Pt) {d : Design.Two}
    {g g' : ℕ} (hk : gateKey h s Pt d g = gateKey h s Pt d g') {u : ℕ}
    (hg : g ∈ Reveal.needs (partition h) s u) : g' ∈ Reveal.needs (partition h) s u := by
  have h1 : ((Finset.range Pt.blocks.length).filter fun v => g ∈ Reveal.needs (partition h) s v) =
      (Finset.range Pt.blocks.length).filter fun v => g' ∈ Reveal.needs (partition h) s v :=
    congrArg Prod.fst hk
  have hmem : u ∈ (Finset.range Pt.blocks.length).filter fun v => g ∈ Reveal.needs (partition h) s v :=
    Finset.mem_filter.2 ⟨Finset.mem_range.2 (Reveal.lt_of_mem_needs (partition h) hP hg), hg⟩
  rw [h1] at hmem
  exact (Finset.mem_filter.1 hmem).2

/-- Gates with one key lie in one replay block. -/
theorem gateKey_block {s : AuditStatement} {Pt : Partition} {d : Design.Two} {g g' : ℕ}
    (hk : gateKey h s Pt d g = gateKey h s Pt d g') : gateBlock d.part g = gateBlock d.part g' :=
  congrArg (fun k : Finset ℕ × Option ℕ × Bool => k.2.1) hk

/-- **Gates with one key lie on one side of each replay block's boundary**, where the replay blocks are disjoint: a
gate of replay block `rb` off its boundary has its key only with gates of `rb` off its boundary. -/
theorem gateKey_side {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hnd : d.part.gates.Nodup) {g g' : ℕ}
    (hk : gateKey h s Pt d g = gateKey h s Pt d g') {ρ : ℕ} {rb : List ℕ} (hrb : d.part.blocks[ρ]? = some rb)
    (hg : g ∈ rb ∧ s.prog.Boundary rb g = false) : g' ∈ rb ∧ s.prog.Boundary rb g' = false := by
  have hgb : gateBlock d.part g = some ρ := gateBlock_of_mem hnd hrb hg.1
  have hgb' : gateBlock d.part g' = some ρ := (gateKey_block h hk).symm.trans hgb
  have h3 : ((gateBlock d.part g).any fun ρ => s.prog.Boundary (d.part.blocks.getD ρ []) g) =
      ((gateBlock d.part g').any fun ρ => s.prog.Boundary (d.part.blocks.getD ρ []) g') :=
    congrArg (fun k : Finset ℕ × Option ℕ × Bool => k.2.2) hk
  simp only [hgb, hgb', Option.any_some] at h3
  obtain ⟨hr', hg'⟩ := gateBlock_some hgb'
  have hrbeq : d.part.blocks.getD ρ [] = rb := Option.some.inj (hr'.symm.trans hrb)
  rw [hrbeq] at h3 hg'
  exact ⟨hg', h3.symm.trans hg.2⟩

/-- **A reveal that passes whole holds the classes of the key whole**: at a statement in the two-stage scope and every
transcript, where every opening the code reveals for a unit `u` at a witness passes the commitment's check, an opening
of that reveal has, at every gate `g'` with its gate's key, an opening of the same reveal at `g'`, under the same
roots, that passes too. -/
theorem reveal_key_of_passes {s : AuditStatement} {Pt : Partition} {d : Design.Two}
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {u : ℕ} {v : Seen}
    {w : Claims.Flock.Wit}
    (hpass : ∀ q ∈ (code h).reveal s u v w, (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true) :
    ∀ p ∈ (code h).reveal s u v w, ∀ g' : ℕ, gateKey h s Pt d p.2.pos = gateKey h s Pt d g' →
      ∃ p' ∈ (code h).reveal s u v w, p'.1 = p.1 ∧ p'.2.pos = g' ∧
        (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 p'.1 p'.2 = true := by
  intro p hp g' hk
  obtain ⟨g, hg, hgp⟩ := (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).1 hp
  have hpos : p.2.pos = g := congrArg (fun q : List Msg × Opening Bool => q.2.pos) hgp.symm
  rw [hpos] at hk
  have hg' : g' ∈ Reveal.needs (partition h) s u := gateKey_needs h hP hk hg
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u (Reveal.lt_of_mem_needs (partition h) hP hg))
  obtain ⟨rb, hrb⟩ := replay_block hρ
  have hroots : Reveal.rootsAt (partition h) (two h) (iroots []) s u v g' =
      Reveal.rootsAt (partition h) (two h) (iroots []) s u v g :=
    (Reveal.rootsAt_same_side (partition h) (two h) (iroots []) (g := g) (g' := g') hP hd hρ hrb
      ⟨gateKey_side h hin.2.2.1 hk hrb, gateKey_side h hin.2.2.1 hk.symm hrb⟩).symm
  have hmem : Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u v w g' ∈
      (code h).reveal s u v w :=
    (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).2 ⟨g', hg', rfl⟩
  have hfst : (Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u v w g').1 = p.1 :=
    hroots.trans (congrArg Prod.fst hgp)
  exact ⟨_, hmem, hfst, rfl, hpass _ hmem⟩

/-- **What an extracted witness reveals holds the classes of the key whole**: at SPOT[Flock]'s compiled two-stage
verifier, under `Built` and `Addresses`, at a statement in the two-stage scope and every run, an opening of what a
witness the session's extraction returns reveals at a checked unit `u` has, at every gate `g'` with its gate's key, an
opening of the same reveal at `g'`, under the same roots, that passes the commitment's check. -/
theorem sessWit_reveal_key (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt) (hd : two h s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).toCompiled.checked x r)
    (c : (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)).EC
      ((SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
        (two h) (rs h) (cap h)).toCompiled.sh x u))
    {w : Claims.Flock.Wit}
    (hw : (((SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h)).toCompiled.sessWit x u r
        (run FlockDeploy.sha512 P (V h) x ω r).seen c).eval FlockDeploy.sha512
          (run FlockDeploy.sha512 P (V h) x ω)).1 = some w) :
    ∀ p ∈ (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen w, ∀ g' : ℕ,
      gateKey h s Pt d p.2.pos = gateKey h s Pt d g' →
      ∃ p' ∈ (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen w,
        p'.1 = p.1 ∧ p'.2.pos = g' ∧
          (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 p'.1 p'.2 = true :=
  reveal_key_of_passes h hP hd hin (sessWit_reveal_passes h hB hS hx hP hd hin P ω r hu c hw)

end SpotFlock
