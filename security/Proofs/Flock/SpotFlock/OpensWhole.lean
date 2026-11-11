module

public import Proofs.Flock.SpotFlock.Frame
public import Proofs.Flock.SpotFlock.FrameIsBuilt
public import Proofs.Flock.SpotFlock.KS
public import Proofs.Flock.SamplingGeneric.Session
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
  is not the refused one, so every round before the session's was answered with a message; round 2 is answered only
  where the session statements' check `stmtOk` passed, and the session's statement is the checked unit's.

Those are the premises of `Reveals` (the decoded values' openings pass at every gate the proof block needs), which the
frame's construction gives under `Built` and `Addresses`. `Reveals` is stated at an accepted transcript (`seenAt`);
the code's reveal reads only a transcript's messages under `Built` (`reveal_out`), so it applies at a refused run too.

* `reveal_passes`: at a checked unit whose session's view accepted and a witness `check` accepts at its statement,
  every opening of the code's reveal passes.
* `reveal_class`: so an opening of the reveal has, at every other gate the unit needs whose roots are the same, an
  opening of the same reveal under the same roots that passes. Over every transcript and witness this fails: the
  commitment's check places the opening's value in the row and never compares it with the row's own bit there, so
  only a witness the relation accepts ties the two.
* `sessWit_reveal_passes`: the two halves composed at the compiled two-stage verifier.
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
  rw [dif_pos hshape, L.served_eval H P ω P' r r hcard (fun r' hr' => (hview r' hr').1) fun _ _ => rfl] at hw
  exact Arg.ext_some H xS _ _ _ hw

end Claims.SamplingGeneric

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-! ## A run whose round 2 was answered -/

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)

/-- **A run whose verifier answers its first two rounds and checks its third**, and whose third round was answered
with a message, answered it with nothing at a check that passed. The run need not accept. -/
theorem stepWith_three_msg (c₀ c₁ : Msg) (ok : Msg → Msg → Bool) (hv1 : ∀ a, vn [a] = .ret (.msg c₀))
    (hv2 : ∀ a b, vn [a, b] = .ret (.msg c₁))
    (hv3 : ∀ a b c, vn [a, b, c] = .ret (if ok b c then .msg [] else .rej)) :
    ∀ (f : ℕ), 3 ≤ f → ∀ m y : Msg,
      (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).msgs[2]? = some (m, .msg y) →
      ∃ m₀ m₁ xS rest, (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).msgs =
        (m₀, .msg c₀) :: (m₁, .msg c₁) :: (xS, .msg []) :: rest ∧ ok m₁ xS = true
  | 0, hf, _, _, _ => absurd hf (by decide)
  | 1, hf, _, _, _ => absurd hf (by decide)
  | 2, hf, _, _, _ => absurd hf (by decide)
  | g + 1 + 1 + 1, _, m, y, h2 => by
    rw [stepWith_succ_msg H P x ω vn (m := c₀) (lo := []) (by rw [List.nil_append, hv1]; rfl)] at h2 ⊢
    simp only [List.nil_append] at h2 ⊢
    rw [stepWith_succ_msg H P x ω vn (m := c₁) (lo := []) (by rw [List.singleton_append, hv2]; rfl)] at h2 ⊢
    simp only [List.cons_append, List.nil_append] at h2 ⊢
    generalize ((P.next x [] ω).eval H).1 = a₀ at h2 ⊢
    generalize ((P.next x [(a₀, c₀)] ω).eval H).1 = a₁ at h2 ⊢
    cases hok : ok a₁ ((P.next x [(a₀, c₀), (a₁, c₁)] ω).eval H).1 with
    | false =>
      have e₂ : (vn ([a₀, a₁] ++ [((P.next x [(a₀, c₀), (a₁, c₁)] ω).eval H).1])).eval H = (.rej, []) := by
        rw [List.cons_append, List.singleton_append, hv3, hok]; rfl
      exfalso
      first
        | (simp only [stepWith, e₂] at h2; simp at h2)
        | simp [stepWith, hv3, hok, OC.eval] at h2
    | true =>
      rw [stepWith_succ_msg H P x ω vn (m := []) (lo := [])
        (by rw [List.cons_append, List.singleton_append, hv3, hok]; rfl)]
      obtain ⟨rest, hr⟩ := Claims.Sampling.stepWith_msgs_prefix H P x ω vn g _ _ _
      refine ⟨a₀, a₁, _, rest, ?_, hok⟩
      rw [hr]
      simp

end Steps

/-- **Rounds answered through the third**: a transcript's messages of which at least `n ≥ 3` are there, the first `n`
answered with a message, have a third one answered with a message. -/
theorem answered_two {l : List (Msg × Out)} {n : ℕ} (hn : 3 ≤ n) (hlen : n ≤ l.length)
    (hall : (l.take n).all (fun p => answered p.2) = true) : ∃ m y, l[2]? = some (m, Out.msg y) := by
  rcases l with _ | ⟨a, _ | ⟨b, _ | ⟨⟨m, o⟩, tl⟩⟩⟩
  · have : n ≤ 0 := hlen
    omega
  · have : n ≤ 1 := hlen
    omega
  · have : n ≤ 2 := hlen
    omega
  · obtain ⟨n', rfl⟩ : ∃ n', n = n' + 3 := ⟨n - 3, by omega⟩
    cases o with
    | msg y => exact ⟨m, y, by first | rfl | simp⟩
    | acc =>
      first
        | simp [List.take_succ_cons, answered] at hall
        | exact absurd (List.all_eq_true.1 hall (m, Out.acc) (by simp [List.take_succ_cons])) (by simp [answered])
    | rej =>
      first
        | simp [List.take_succ_cons, answered] at hall
        | exact absurd (List.all_eq_true.1 hall (m, Out.rej) (by simp [List.take_succ_cons])) (by simp [answered])

variable (h : Held)

/-- **A run whose round 2 was answered passed the session statements' check**: its messages start with rounds 0 and
1 answered with their coins and round 2 answered with nothing, and `stmtOk` held at them. `run_seenAt` without the
run's acceptance. -/
theorem run_msgs_of_answered {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin) {m y : Msg}
    (h2 : (run H P (V h) x ω r).msgs[2]? = some (m, .msg y)) :
    ∃ m₀ m₁ m₂ rest, (run H P (V h) x ω r).msgs =
        (m₀, .msg (r ⟨0, k_pos h x⟩).bytes) :: (m₁, .msg (r ⟨1, one_lt_k h x⟩).bytes) :: (m₂, .msg []) :: rest ∧
      stmtOk h x m₁ m₂ (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) = true := by
  have e0 : coinSeq r 0 = r ⟨0, k_pos h x⟩ := dite_eq_left (k_pos h x)
  have e1 : coinSeq r 1 = r ⟨1, one_lt_k h x⟩ := dite_eq_left (one_lt_k h x)
  have hk : 3 ≤ (V h).k x := by
    have := three_le_off h h.ms.length
    show 3 ≤ off h h.ms.length + 1
    omega
  obtain ⟨m₀, m₁, m₂, rest, hm, hok⟩ := stepWith_three_msg H P x ω
    (fun as => (V h).next x as ((List.range as.length).map (coinSeq r)))
    (r ⟨0, k_pos h x⟩).bytes (r ⟨1, one_lt_k h x⟩).bytes
    (fun m₁ m₂ => stmtOk h x m₁ m₂ (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩))
    (fun a => by show next h x [a] [coinSeq r 0] = _; rw [e0]; rfl)
    (fun a b => by show next h x [a, b] [coinSeq r 0, coinSeq r 1] = _; rw [e1]; rfl)
    (fun a b c => by show next h x [a, b, c] [coinSeq r 0, coinSeq r 1, coinSeq r 2] = _; rw [e0, e1]; rfl)
    ((V h).k x) hk m y h2
  exact ⟨m₀, m₁, m₂, rest, hm, hok⟩

/-- **A session whose view accepted is not the refused one**: the transcript's round 2 was answered with a message,
and the session's statement is the unit's template's as round 2 states it. -/
theorem session_acc_msg {u : ℕ} {v : Seen} {xS : Msg} {vS : Seen} (hsess : session h u v = some (xS, vS))
    (hacc : vS.out = .acc) :
    (∃ m y, v.msgs[2]? = some (m, Out.msg y)) ∧
      xS = stmtAt h (v.msgs.getD 2 ([], .rej)).1 (templateOf h.ms u) := by
  by_cases hans : off h (templateOf h.ms u) ≤ v.msgs.length ∧
      (v.msgs.take (off h (templateOf h.ms u))).all (answered ·.2) = true
  · by_cases hok : sessionOk h (templateOf h.ms u)
        (stmtAt h (v.msgs.getD 2 ([], .rej)).1 (templateOf h.ms u)) = true
    · rw [session, if_pos hans, if_pos hok] at hsess
      have e := Prod.mk.inj (Option.some.inj hsess)
      exact ⟨answered_two (three_le_off h _) hans.1 hans.2, e.1.symm⟩
    · rw [session, if_pos hans, if_neg hok] at hsess
      have e := Prod.mk.inj (Option.some.inj hsess)
      rw [← e.2] at hacc
      simp [refusedSeen] at hacc
  · rw [session, if_neg hans] at hsess
    have e := Prod.mk.inj (Option.some.inj hsess)
    rw [← e.2] at hacc
    simp [refusedSeen] at hacc

/-- **The code's half of the wiring**: at any run, a unit's session whose view accepted has the run's rounds 0 to 2
answered, the session statements' check passed at them, and its statement the unit's. -/
theorem session_acc_run {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ} {xS : Msg} {vS : Seen}
    (hsess : session h u (run H P (V h) x ω r).seen = some (xS, vS)) (hacc : vS.out = .acc) :
    ∃ m₀ m₁ m₂ rest, (run H P (V h) x ω r).msgs =
        (m₀, .msg (r ⟨0, k_pos h x⟩).bytes) :: (m₁, .msg (r ⟨1, one_lt_k h x⟩).bytes) :: (m₂, .msg []) :: rest ∧
      stmtOk h x m₁ m₂ (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) = true ∧ xS = unitStmt h m₂ u := by
  obtain ⟨⟨m, y, h2⟩, hxS⟩ := session_acc_msg h hsess hacc
  obtain ⟨m₀, m₁, m₂, rest, hm, hok⟩ := run_msgs_of_answered h H P x ω r h2
  refine ⟨m₀, m₁, m₂, rest, hm, hok, ?_⟩
  have hm' : (run H P (V h) x ω r).seen.msgs =
      (m₀, .msg (r ⟨0, k_pos h x⟩).bytes) :: (m₁, .msg (r ⟨1, one_lt_k h x⟩).bytes) :: (m₂, .msg []) :: rest := hm
  have hg : ((run H P (V h) x ω r).seen.msgs.getD 2 ([], .rej)).1 = m₂ := by
    rw [hm']
    all_goals rfl
  exact hxS.trans (congrArg (fun a => stmtAt h a (templateOf h.ms u)) hg)

/-! ## The code's reveal reads a transcript's messages -/

/-- **The reveal reads only the transcript's messages**, at a registration whose frame is SPOT's own: the roots read
the round-1 message and the opener rounds 1 and 2, so the reveal at a refused transcript is the one at an accepted
transcript of the same messages. -/
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
  have hfr : ∀ g R, Reveal.frameAt h.frame s u ⟨ms, o⟩ wF g R = Reveal.frameAt h.frame s u ⟨ms, o'⟩ wF g R :=
    fun g R => congrArg (fun l : List (List Msg × Opening Bool) =>
      ((l.find? fun q => q.2.pos == g && q.1 == R).map Prod.snd).getD ⟨g, false, []⟩) hinner
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
  have hum : u ∈ h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (src (r ⟨1, one_lt_k h x⟩)) :=
    (congrArg (u ∈ ·) (checked_eq h x hst r)).mp hu
  have hlt : u < Pt.blocks.length := by
    have := mem_checked_lt hum
    rwa [units_base h hx hP] at this
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u hlt)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  have hb : Pt.blocks[u]? = some Pt.blocks[u] := List.getElem?_eq_getElem hlt
  have hseen : (run FlockDeploy.sha512 P (V h) x ω r).seen =
      ⟨(m₀, .msg (r ⟨0, k_pos h x⟩).bytes) :: (m₁, .msg (r ⟨1, one_lt_k h x⟩).bytes) :: (m₂, .msg []) :: rest,
        (run FlockDeploy.sha512 P (V h) x ω r).out⟩ :=
    congrArg (fun l => (⟨l, (run FlockDeploy.sha512 P (V h) x ω r).out⟩ : Seen)) hm
  have hrev : (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen wF =
      (code h).reveal s u (seenAt m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) rest) wF :=
    (congrArg (fun v => (code h).reveal s u v wF) hseen).trans (reveal_out h hB s u _ _ .acc wF)
  rw [hrev] at hq
  obtain ⟨g, hg, hgq⟩ :=
    (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).1 hq
  obtain ⟨q', hq', hop, hpos, -, -⟩ :=
    reveals_of_frameRows h (layoutFor h) (frameRows_of_frameIs h (layoutFor h) (frameIs_built h hB hS))
      x s Pt d m₀ m₁ m₂ (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) rest wF u _ ρ rb hx hP hd hin hok hum hck hb hρ
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

/-- **One opening that passes brings its class**: at `reveal_passes`' premises, an opening of the reveal has, at
every other gate `g'` the unit needs whose roots are its own, an opening of the same reveal under the same roots at
`g'` that passes. -/
theorem reveal_class (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt) (hd : two h s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin) {u : ℕ}
    (hu : u ∈ (code h).checked x r) {xS : Msg} {vS : Seen}
    (hsess : session h u (run FlockDeploy.sha512 P (V h) x ω r).seen = some (xS, vS)) (hacc : vS.out = .acc)
    {wF : Claims.Flock.Wit}
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true) :
    ∀ p ∈ (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen wF,
      ∀ g' ∈ Reveal.needs (partition h) s u,
        Reveal.rootsAt (partition h) (two h) (iroots []) s u (run FlockDeploy.sha512 P (V h) x ω r).seen g' =
          p.1 →
        ∃ p' ∈ (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen wF,
          p'.1 = p.1 ∧ p'.2.pos = g' ∧
            (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 p'.1 p'.2 = true := by
  intro p _ g' hg' hroots
  have hmem : Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u
      (run FlockDeploy.sha512 P (V h) x ω r).seen wF g' ∈
        (code h).reveal s u (run FlockDeploy.sha512 P (V h) x ω r).seen wF :=
    (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).2 ⟨g', hg', rfl⟩
  exact ⟨_, hmem, hroots, rfl, reveal_passes h hB hS hx hP hd hin P ω r hu hsess hacc hck _ hmem⟩

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

end SpotFlock
