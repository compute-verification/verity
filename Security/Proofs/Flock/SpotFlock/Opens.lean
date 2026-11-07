module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Claims.Sampling.OpensTwo
public import Proofs.Flock.Claims.Sampling.Deployed
public import Proofs.Flock.Claims.Sampling.ProgReads

/-!
# SPOT-Flock's session opens each checked proof block at its replay block

`sessionOpensTwo h`: `Claims.SamplingGeneric.SessionOpensTwo` at the deployed two-stage code (`code h`), Flock's
masked record and SHA-512, from what one session statement establishes (`StmtOpens h`).

**The run** (`run_seenAt`). An accepted run of `V h` answered rounds 0 and 1 with their coins and round 2 with nothing,
so the verifier's check of the session's statement passed there (`stmtOk h x m₁ xS c₀ c₁`, at the round-1 message
`m₁`, the round-2 message `xS` and the coins `c₀`, `c₁` of rounds 0 and 1), and its transcript is `seenAt` of them.
Its session is `xS` and the transcript from round 3 on, the interior roots are those `m₁` states, and its checked units
are the second stage's on `c₀` and `c₁` (`checked_eq`).

**The statement** (`StmtOpens h`): with no run and no coin beyond `c₀` and `c₁`, a session statement that the verifier's
check passes, at a Flock witness in its relation, reveals for each unit the two stages check openings under which its
proof block is correct on its replay block's values. That is what the session statement's relation, the witness's
encoding of its openings and the registered reads establish; the run only selects the statement.

**The witness's values** (`WitOpens h`, `stmtOpens`). A checked unit `u` has a proof block `b` in a replay block `ρ`
with gates `rb` (`mem_checked_lt`, `InScopeTwo`, `replay_block`). `StmtOpens` follows from one pair of values: boundary
values `W` and interior values `I` that make `b` correct on `ρ`'s values, each that `b`'s correctness needs
(`Prog.Needs`) revealed by an opening that passes, an interior one of `ρ` under the interior roots, every other under the
run's roots. Values that agree with those openings agree with `W` and `I` wherever `b`'s correctness reads them
(`Prog.correct_of_needs`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-! ## Runs -/

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → OC Out)

theorem stepWith_succ_msg {f : ℕ} {hist : List (Msg × Msg)} {as : List Msg} {t : Trace} {m : Msg}
    {lo : List (Msg × Msg)} (h : (vn (as ++ [((P.next x hist ω).eval H).1])).eval H = (.msg m, lo)) :
    stepWith H P x ω vn (f + 1) hist as t =
      stepWith H P x ω vn f (hist ++ [(((P.next x hist ω).eval H).1, m)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++ [(((P.next x hist ω).eval H).1, .msg m)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2, vLog := t.vLog ++ lo } := by
  simp only [stepWith, h]

/-- **A run whose verifier answers its first two rounds and checks its third** accepts only past a third round it
answered with nothing, at a check that passed. -/
theorem stepWith_three (c₀ c₁ : Msg) (ok : Msg → Msg → Bool) (hv1 : ∀ a, vn [a] = .ret (.msg c₀))
    (hv2 : ∀ a b, vn [a, b] = .ret (.msg c₁))
    (hv3 : ∀ a b c, vn [a, b, c] = .ret (if ok b c then .msg [] else .rej)) :
    ∀ (f : ℕ), (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).out = .acc →
      ∃ m₀ m₁ xS rest, (stepWith H P x ω vn f [] [] ⟨[], [], [], .rej⟩).msgs =
        (m₀, .msg c₀) :: (m₁, .msg c₁) :: (xS, .msg []) :: rest ∧ ok m₁ xS = true
  | 0, hacc => by simp [stepWith] at hacc
  | 1, hacc => by simp [stepWith, hv1, OC.eval] at hacc
  | 2, hacc => by simp [stepWith, hv1, hv2, OC.eval] at hacc
  | g + 1 + 1 + 1, hacc => by
    rw [stepWith_succ_msg H P x ω vn (m := c₀) (lo := []) (by rw [List.nil_append, hv1]; rfl)] at hacc ⊢
    simp only [List.nil_append] at hacc ⊢
    rw [stepWith_succ_msg H P x ω vn (m := c₁) (lo := []) (by rw [List.singleton_append, hv2]; rfl)] at hacc ⊢
    simp only [List.cons_append, List.nil_append] at hacc ⊢
    generalize ((P.next x [] ω).eval H).1 = a₀ at hacc ⊢
    generalize ((P.next x [(a₀, c₀)] ω).eval H).1 = a₁ at hacc ⊢
    cases hok : ok a₁ ((P.next x [(a₀, c₀), (a₁, c₁)] ω).eval H).1 with
    | false =>
      have e₂ : (vn ([a₀, a₁] ++ [((P.next x [(a₀, c₀), (a₁, c₁)] ω).eval H).1])).eval H = (.rej, []) := by
        rw [List.cons_append, List.singleton_append, hv3, hok]; rfl
      simp only [stepWith, e₂] at hacc
      cases hacc
    | true =>
      rw [stepWith_succ_msg H P x ω vn (m := []) (lo := [])
        (by rw [List.cons_append, List.singleton_append, hv3, hok]; rfl)]
      obtain ⟨rest, hr⟩ := Claims.Sampling.stepWith_msgs_prefix H P x ω vn g _ _ _
      refine ⟨a₀, a₁, _, rest, ?_, hok⟩
      rw [hr]
      simp

end Steps

variable (h : Held)

theorem k_pos (x : Msg) : 0 < (V h).k x := by show 0 < 3 + k₀ h; omega

theorem one_lt_k (x : Msg) : 1 < (V h).k x := by show 1 < 3 + k₀ h; omega

/-- **A run's transcript through its session's statement**: rounds 0 and 1 answered with their coins `c₀` and `c₁`,
round 2's message `xS` answered with nothing, then `rest`, accepted. -/
def seenAt (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) : Seen :=
  ⟨(m₀, .msg c₀.bytes) :: (m₁, .msg c₁.bytes) :: (xS, .msg []) :: rest, .acc⟩

/-- **An accepted run passed the session statement's check**: its transcript is `seenAt` of its first three messages and
the coins of rounds 0 and 1, and `stmtOk` held at them. -/
theorem run_seenAt {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((V h).k x) → FlockDeploy.Coin) (hacc : (run H P (V h) x ω r).out = .acc) :
    ∃ m₀ m₁ xS rest, (run H P (V h) x ω r).seen = seenAt m₀ m₁ xS (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) rest ∧
      stmtOk h x m₁ xS (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) = true := by
  have e0 : coinSeq r 0 = r ⟨0, k_pos h x⟩ := dif_pos (k_pos h x)
  have e1 : coinSeq r 1 = r ⟨1, one_lt_k h x⟩ := dif_pos (one_lt_k h x)
  obtain ⟨m₀, m₁, xS, rest, hm, hok⟩ := stepWith_three H P x ω
    (fun as => (V h).next x as ((List.range as.length).map (coinSeq r)))
    (r ⟨0, k_pos h x⟩).bytes (r ⟨1, one_lt_k h x⟩).bytes
    (fun m₁ xS => stmtOk h x m₁ xS (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩))
    (fun a => by show next h x [a] [coinSeq r 0] = _; rw [e0]; rfl)
    (fun a b => by show next h x [a, b] [coinSeq r 0, coinSeq r 1] = _; rw [e1]; rfl)
    (fun a b c => by show next h x [a, b, c] [coinSeq r 0, coinSeq r 1, coinSeq r 2] = _; rw [e0, e1]; rfl)
    ((V h).k x) hacc
  refine ⟨m₀, m₁, xS, rest, ?_, hok⟩
  have hm' : (run H P (V h) x ω r).msgs = _ := hm
  show (⟨(run H P (V h) x ω r).msgs, (run H P (V h) x ω r).out⟩ : Seen) = _
  rw [hm', hacc]
  rfl

/-- **The checked units** of a run whose statement decodes: the second stage's on the coins of rounds 0 and 1. -/
theorem checked_eq (x : Msg) {st : Stmt} (hst : decode x = some st) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) :
    (code h).checked x r = h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (src (r ⟨1, one_lt_k h x⟩)) := by
  have hr : reg h x = some h.reg := by simp [SpotFlock.reg, hst]
  show Option.getD (Option.map _ (reg h x)) [] = _
  rw [hr, Option.map_some, Option.getD_some]
  rfl

/-! ## The session statement's openings -/

/-- **A session statement opens every unit it checks**: at a run's statement in the two-stage scope, a session
statement `xS` that the verifier's check passes at the round-1 message `m₁` and the coins `c₀` and `c₁`, and a Flock
witness in `xS`'s relation, each unit `u` the two stages check on those coins has a proof block `b` in a replay block
`ρ` with gates `rb`, and any values `val` and interior values `ι` that agree with the openings the witness reveals and
that pass (under the run's roots, and the interior roots `m₁` states for `ρ`) make `b` correct on `ρ`'s values. -/
def StmtOpens : Prop :=
  ∀ x s Pt d (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ replayOf Pt d.part u = some ρ ∧ d.part.blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ (code h).reveal s u (seenAt m₀ m₁ xS c₀ c₁ rest) wF,
          (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots h x (seenAt m₀ m₁ xS c₀ c₁ rest) ρ → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn rb val ι) b

/-- **SPOT-Flock's session opens each checked proof block at its replay block**: `SessionOpensTwo` at the deployed
two-stage code, from `StmtOpens`. -/
theorem sessionOpensTwo (hs : StmtOpens h) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) (iroots h) := by
  intro x s Pt d Ω P ω r u xS vS wF hx hP hd hin hacc hu hsess hck
  obtain ⟨m₀, m₁, xS', rest, hseen, hok⟩ := run_seenAt h FlockDeploy.sha512 P x ω r hacc
  have hseen' : (run FlockDeploy.sha512 P (deploy _root_.Flock.Readable.relation (code h)).V x ω r).seen =
      seenAt m₀ m₁ xS' (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) rest := hseen
  rw [hseen'] at hsess ⊢
  have hxS : some (xS', (⟨rest, .acc⟩ : Seen)) = some (xS, vS) := hsess
  obtain ⟨rfl, -⟩ := Prod.mk.inj_iff.1 (Option.some.inj hxS)
  have hx' : parse h x = some s := hx
  obtain ⟨st, hst⟩ : ∃ st, decode x = some st := by
    cases hdx : decode x with
    | none => simp [SpotFlock.parse, hdx] at hx'
    | some st => exact ⟨st, rfl⟩
  obtain ⟨p, hp⟩ := hu
  have hum : u ∈ h.reg.checked (units (base h) x) (src (r ⟨0, k_pos h x⟩)) (src (r ⟨1, one_lt_k h x⟩)) := by
    rw [← checked_eq h x hst r]
    exact List.mem_of_getElem? hp
  exact hs x s Pt d m₀ m₁ xS' (r ⟨0, k_pos h x⟩) (r ⟨1, one_lt_k h x⟩) rest wF u hx hP hd hin hok hum hck

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

theorem replay_block {Pt Pρ : Partition} {v ρ : ℕ} (hρ : replayOf Pt Pρ v = some ρ) :
    ∃ rb, Pρ.blocks[ρ]? = some rb :=
  ⟨_, List.getElem?_eq_getElem (List.mem_range.1 (List.mem_of_find?_eq_some hρ))⟩

/-- **The session statement's witness holds values that make each checked unit correct**: at a run's statement in the
two-stage scope, a session statement `xS` that the verifier's check passes and a Flock witness in its relation, each
unit `u` the two stages check, with proof block `b` in replay block `ρ` of gates `rb`, has boundary values `W` and
interior values `I` that make `b` correct on `ρ`'s values, and the witness reveals each value `b`'s correctness needs by
an opening that passes: one of `ρ`'s interior under the interior roots `m₁` states for `ρ`, every other under the run's
roots. -/
def WitOpens : Prop :=
  ∀ x s Pt d (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) wF u b ρ rb,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    ∃ (W : ℕ → Option Bool) (I : ℕ → Bool), s.prog.Correct (s.prog.refIn rb W I) b ∧
      ∀ g, s.prog.Needs b g →
        (g ∈ rb ∧ s.prog.Boundary rb g = false →
          ∃ q ∈ (code h).reveal s u (seenAt m₀ m₁ xS c₀ c₁ rest) wF,
            (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧
              q.1 = iroots h x (seenAt m₀ m₁ xS c₀ c₁ rest) ρ ∧ q.2.pos = g ∧ q.2.val = I g) ∧
        (¬(g ∈ rb ∧ s.prog.Boundary rb g = false) →
          ∃ q ∈ (code h).reveal s u (seenAt m₀ m₁ xS c₀ c₁ rest) wF,
            (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧
              q.1 = s.roots ∧ q.2.pos = g ∧ W g = some q.2.val)

/-- **`StmtOpens` from the witness's values**: values that agree with the openings that pass agree with `W` and `I`
wherever the proof block's correctness reads them. -/
theorem stmtOpens (hw : WitOpens h) : StmtOpens h := by
  intro x s Pt d m₀ m₁ xS c₀ c₁ rest wF u hx hP hd hin hok hu hck
  have hlt : u < Pt.blocks.length := units_base h hx hP ▸ mem_checked_lt hu
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u hlt)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  refine ⟨Pt.blocks[u], ρ, rb, List.getElem?_eq_getElem hlt, hρ, hrb, fun val ι hv => ?_⟩
  obtain ⟨W, I, hc, hcov⟩ := hw x s Pt d m₀ m₁ xS c₀ c₁ rest wF u _ ρ rb hx hP hd hin hok hu hck
    (List.getElem?_eq_getElem hlt) hρ hrb
  refine s.prog.correct_of_needs (fun g hg => ?_) hc
  unfold Prog.refIn
  by_cases hi : g ∈ rb ∧ s.prog.Boundary rb g = false
  · obtain ⟨q, hq, hop, hr, hpos, hval⟩ := (hcov g hg).1 hi
    rw [ite_eq_left hi, ite_eq_left hi, ← hval, ← hpos, (hv q hq hop).2 hr]
  · obtain ⟨q, hq, hop, hr, hpos, hval⟩ := (hcov g hg).2 hi
    rw [ite_eq_right hi, ite_eq_right hi, hval, ← hpos, (hv q hq hop).1 hr]

/-- **`SessionOpensTwo` at the deployed two-stage code from the witness's values.** -/
theorem sessionOpensTwo_of_wit (hw : WitOpens h) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) (iroots h) :=
  sessionOpensTwo h (stmtOpens h hw)

end SpotFlock
