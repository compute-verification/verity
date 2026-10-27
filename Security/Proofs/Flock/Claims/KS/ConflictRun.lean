import Proofs.Flock.Claims.KS.ConflictWithin
import Proofs.Flock.Claims.KS.PlainQueryRound
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts

/-!
# An accepted run, within its own log

What an accepted run of the deployed verifier (`Deploy.V`, plain or `--zk`) gives `ConflictTied`, at either record:

* the verifier decides on the run's messages, and the decision runs within the run's verifier log `vLog`
  (`acc_decide`, from `run_acc_log`: the deciding step's queries are the last ones `vLog` appends);
* so the shipped verdict accepts with SHA-512 answered only within that log (`acc_within`: `verdictM` at
  `hashesIn (vLog.map Prod.fst)`, through `within`), and the custody checks pass;
* the link the decision reads is the prover's first message, which depends on the tape alone (`run_head`,
  `act_decide_link`), so two runs at one tape decide on one link;
* the scope check those checks run pins every table's Merkle scheme to `hm96Sha512` or, for a statement read
  without `--zk`, plain SHA-512 leaves (`checks_leaf`); under `--zk` it is `hm96Sha512` (`checks_hm96`).
-/

namespace FlockDeploy.KS

open Flock
open Claims (OC OCV)
open FlockSoundness.Refine (ebind_ok)

/-! ## The deciding step's queries are the run's last -/

/-- **A run that accepts logs its deciding step's queries**: where the rounds accept, every query of the verifier's
step at the prover's messages is in the trace's verifier log. -/
theorem stepWith_acc_log {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      t.msgs.map Prod.fst = as → (Claims.stepWith H P x ω vnext n hist as t).out = .acc →
        ∀ p ∈ ((vnext ((Claims.stepWith H P x ω vnext n hist as t).msgs.map Prod.fst)).eval H).2,
          p ∈ (Claims.stepWith H P x ω vnext n hist as t).vLog := by
  intro n
  induction n with
  | zero => intro hist as t _ h; simp [Claims.stepWith] at h
  | succ n ih =>
    intro hist as t ht h
    simp only [Claims.stepWith] at h ⊢
    split at h
    · rename_i m heq
      simp only [heq] at h ⊢
      exact ih _ _ _ (by simp [ht]) h
    · rename_i heq
      simp only [heq, List.map_append, ht, List.map_cons, List.map_nil]
      exact fun p hp => List.mem_append_right _ hp
    · simp at h

theorem run_acc_log {Ω Coin : Type} [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω)
    (V : Claims.Verifier Coin) (x : Claims.Msg) (ω : Ω) (r : Fin (V.k x) → Coin)
    (h : (Claims.run H P V x ω r).out = .acc) :
    ∀ p ∈ ((V.next x ((Claims.run H P V x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run H P V x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r))).eval H).2,
      p ∈ (Claims.run H P V x ω r).vLog :=
  stepWith_acc_log H P x ω _ (V.k x) [] [] ⟨[], [], [], .rej⟩ rfl h

/-! ## The first message is the tape's -/

theorem stepWith_prefix {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      ∃ ext, (Claims.stepWith H P x ω vnext n hist as t).msgs = t.msgs ++ ext := by
  intro n
  induction n with
  | zero => intro hist as t; exact ⟨[], by simp [Claims.stepWith]⟩
  | succ n ih =>
    intro hist as t
    simp only [Claims.stepWith]
    split
    · obtain ⟨ext, he⟩ := ih (hist ++ [(((P.next x hist ω).eval H).1, _)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++
                   [(((P.next x hist ω).eval H).1, ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).1)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
                 vLog := t.vLog ++ ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).2 }
      rw [he]
      exact ⟨_, List.append_assoc _ _ _⟩
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩

theorem stepWith_head {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (vnext : List Claims.Msg → Claims.OC Claims.Out) (n : ℕ) (hist : List (Claims.Msg × Claims.Msg))
    (as : List Claims.Msg) (t : Claims.Trace) (h : (Claims.stepWith H P x ω vnext n hist as t).out = .acc) :
    ∃ rest, (Claims.stepWith H P x ω vnext n hist as t).msgs.map Prod.fst =
      t.msgs.map Prod.fst ++ ((P.next x hist ω).eval H).1 :: rest := by
  cases n with
  | zero => simp [Claims.stepWith] at h
  | succ n =>
    simp only [Claims.stepWith]
    split
    · obtain ⟨ext, he⟩ := stepWith_prefix H P x ω vnext n
        (hist ++ [(((P.next x hist ω).eval H).1, _)]) (as ++ [((P.next x hist ω).eval H).1])
        { t with msgs := t.msgs ++
                   [(((P.next x hist ω).eval H).1, ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).1)],
                 pLog := t.pLog ++ ((P.next x hist ω).eval H).2,
                 vLog := t.vLog ++ ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).2 }
      rw [he]
      exact ⟨ext.map Prod.fst, by simp⟩
    · exact ⟨[], by simp⟩
    · exact ⟨[], by simp⟩

/-- **An accepted run's first message is the prover's first answer at the tape**, which reads no coin. -/
theorem run_head {Ω Coin : Type} [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω)
    (V : Claims.Verifier Coin) (x : Claims.Msg) (ω : Ω) (r : Fin (V.k x) → Coin)
    (h : (Claims.run H P V x ω r).out = .acc) :
    ∃ rest, (Claims.run H P V x ω r).msgs.map Prod.fst = ((P.next x [] ω).eval H).1 :: rest :=
  stepWith_head H P x ω _ (V.k x) [] [] ⟨[], [], [], .rej⟩ h

namespace Gen

variable [S : ZkSetup]

/-- **A decision reads its link from the first message.** -/
theorem act_decide_link {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∃ rest, as = (as.headD []) :: rest ∧ linkOf (as.headD []) = some link := by
  unfold act at h
  split at h
  · cases h
  · repeat' split at h
    all_goals first | (cases h; done) | (cases h; exact ⟨_, rfl, by assumption⟩)

end Gen

/-- **A decision reads its link from the first message.** -/
theorem act_decide_link {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ∃ rest, as = (as.headD []) :: rest ∧ linkOf (as.headD []) = some link :=
  Gen.act_decide_link (S := S₀) (zk := zk) (x := x) (as := as) (cs := cs) (s := s) (spec := spec) (sts := sts)
    (link := link) (c₁ := c₁) (mids := mids) (f := f) (h := h)

/-! ## The decision, within the run's log -/

theorem outOf_log (H : Claims.Msg → Claims.Msg) (d : OCV Unit) : ((outOf d).eval H).2 = (d.run.eval H).2 := by
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]

theorem outOf_acc (H : Claims.Msg → Claims.Msg) (d : OCV Unit) (h : ((outOf d).eval H).1 = .acc) :
    (d.run.eval H).1 = .ok () := by
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval] at h
  cases hr : (d.run.eval H).1 with
  | ok u => rfl
  | error e => rw [hr] at h; cases h

namespace Gen

variable [S : ZkSetup]

/-- **An accepted run decides, within its own verifier log.** -/
theorem acc_decide {zk : Bool} {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {r : Fin ((V zk).k x) → Coin} (h : (Claims.run H P (V zk) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act zk x ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      Runs H (decision zk s x link c₁ mids f) () ((Claims.run H P (V zk) x ω r).vLog.map Prod.fst) := by
  have ha := run_acc_next H P (V zk) x ω r h
  have hl := run_acc_log H P (V zk) x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha hl
  replace ha : ((next zk x ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst) cs).eval H).1 = .acc := ha
  replace hl : ∀ p ∈ ((next zk x ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst) cs).eval H).2,
      p ∈ (Claims.run H P (V zk) x ω r).vLog := hl
  cases hact : act zk x ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst) cs with
  | rej => rw [next_rej H hact] at ha; cases ha
  | coin c => rw [next_coin H hact] at ha; cases ha
  | decide s spec sts link c₁ mids f =>
    refine ⟨s, spec, sts, link, c₁, mids, f, rfl, ?_, fun p hp => List.mem_map_of_mem (hl p ?_)⟩
    · simp only [next, hact] at ha
      exact outOf_acc H _ ha
    · simp only [next, hact, outOf_log]
      exact hp

end Gen

/-- **An accepted run decides, within its own verifier log.** -/
theorem acc_decide {zk : Bool} {H : Claims.Msg → Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {r : Fin ((V zk).k x) → Coin} (h : (Claims.run H P (V zk) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act zk x ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run H P (V zk) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      Runs H (decision zk s x link c₁ mids f) () ((Claims.run H P (V zk) x ω r).vLog.map Prod.fst) :=
  Gen.acc_decide (S := S₀) (zk := zk) (H := H) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (h := h)

namespace Gen

variable [S : ZkSetup]

/-- **A decision within `L`**, at SHA-512 on a statement that sets up at `(spec, sts)`: the setup over the oracle is the
statement's, then the verdict within `L`, and the checks. -/
theorem decision_runs {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {L : List Claims.Msg} (hs : S.Ok zk s spec sts) (h : Runs sha512 (decision zk s x link c₁ mids f) () L) :
    Runs sha512 (verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs) () L ∧
      checks zk s spec x link c₁ mids f = .ok () := by
  unfold decision at h
  obtain ⟨⟨spec', sts'⟩, h0, h⟩ := Runs.bind h
  have e : setupWith (Claims.hOf sha512) s.read = .ok (spec', sts') := by
    rw [← (setupM_eval sha512 s.read).1]
    exact h0.1
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at e
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj e)
  unfold decisionAt at h
  obtain ⟨⟨⟩, h1, h2⟩ := Runs.bind h
  exact ⟨h1, Runs.lift h2⟩

end Gen

/-- **A decision within `L`**, at SHA-512 on a statement that sets up at `(spec, sts)`: the setup over the oracle is the
statement's, then the verdict within `L`, and the checks. -/
theorem decision_runs {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {L : List Claims.Msg} (hs : s.setup = .ok (spec, sts)) (h : Runs sha512 (decision zk s x link c₁ mids f) () L) :
    Runs sha512 (verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs) () L ∧
      checks zk s spec x link c₁ mids f = .ok () :=
  Gen.decision_runs (S := S₀) (zk := zk) (s := s) (spec := spec) (sts := sts) (x := x) (link := link) (c₁ := c₁)
    (mids := mids) (f := f) (L := L) (hs := hs) (h := h)

/-- **The verdict within `L` is the verdict with SHA-512 answered only within `L`.** -/
theorem verdict_within {zk : Bool} {spec : SessionSpec} {tabs : Array Setup} {record : String}
    {proofs : Array ByteArray} {L : List Claims.Msg}
    (h : Runs sha512 (verdictM zk Claims.Hashes.oc spec tabs record proofs) () L) :
    verdictM zk (hashesIn L) spec tabs record proofs = .ok () := by
  unfold verdictM at h ⊢
  cases zk
  · exact (With.verifyWithM_sim (within L) (within_hashes L) spec tabs record proofs) () h
  · exact (With.zk_verifyWithM_sim (within L) (within_hashes L) spec tabs record proofs) () h

namespace Gen

variable [S : ZkSetup]

/-- **An accepted run at `sha512`**: the statement sets up, the run decides on its last message, read as a final
message on the statement, and on the link of the prover's first message; the custody checks pass; and the shipped
verdict accepts with SHA-512 answered only within the run's verifier log. -/
theorem acc_within {zk : Bool} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V zk).k x) → Coin} (h : (Claims.run sha512 P (V zk) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf zk x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P (V zk) x ω r).seen = some f ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧ checks zk s spec x link c₁ mids f = .ok () ∧
      verdictM zk (hashesIn ((Claims.run sha512 P (V zk) x ω r).vLog.map Prod.fst)) spec (tabsOf sts) f.record
        f.proofs = .ok () := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := acc_decide h
  obtain ⟨hv, hc⟩ := decision_runs (setupOf_some (act_decide hact)).2 hrun
  obtain ⟨rest, hhead⟩ := run_head sha512 P (V zk) x ω r h
  obtain ⟨rest', has, hlink⟩ := act_decide_link hact
  rw [hhead] at hlink
  refine ⟨s, spec, sts, link, c₁, mids, f, act_decide hact, ?_, hlink, hc, verdict_within hv⟩
  have hl := act_decide_final hact
  rw [List.getLast?_map, Option.bind_map] at hl
  exact hl

end Gen

/-- **An accepted run at `sha512`**: the statement sets up, the run decides on its last message, read as a final
message on the statement, and on the link of the prover's first message; the custody checks pass; and the shipped
verdict accepts with SHA-512 answered only within the run's verifier log. -/
theorem acc_within {zk : Bool} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V zk).k x) → Coin} (h : (Claims.run sha512 P (V zk) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f, setupOf zk x = some (s, spec, sts) ∧
      lastFinal (Claims.run sha512 P (V zk) x ω r).seen = some f ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧ checks zk s spec x link c₁ mids f = .ok () ∧
      verdictM zk (hashesIn ((Claims.run sha512 P (V zk) x ω r).vLog.map Prod.fst)) spec (tabsOf sts) f.record
        f.proofs = .ok () :=
  Gen.acc_within (S := S₀) (zk := zk) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (h := h)

/-! ## The tables' Merkle scheme -/

namespace Gen

variable [S : ZkSetup]

/-- **The checks pin the tables' Merkle scheme**: every table the statement sets up commits under `hm96Sha512`, or,
for a statement read without `--zk`, under plain SHA-512 leaves. -/
theorem checks_leaf {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {sts : Array Stmt} (hc : checks zk s spec x link c₁ mids f = .ok ()) (hs : S.Ok zk s spec sts) :
    ∀ (j : ℕ) (hj : j < sts.size), sts[j].merkle = MerkleScheme.hm96Sha512 ∨
      (s.read.zk = false ∧ sts[j].merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") := by
  replace hc := checks_base hc
  unfold FlockDeploy.checks at hc
  obtain ⟨_, -, hc⟩ := ebind_ok hc
  obtain ⟨_, -, hc⟩ := ebind_ok hc
  obtain ⟨_, -, hc⟩ := ebind_ok hc
  obtain ⟨_, -, hc⟩ := ebind_ok hc
  obtain ⟨_, hck, -⟩ := ebind_ok hc
  have hck' : ProvedScope.check s.read.tags s.read.zk s.read.I.circuitFile s.read.I.publicFile s.read.I.tables
      s.read.I.count s.read.publicInputs.isSome s.read.registered = .ok () := hck
  obtain ⟨⟨-, -, -, hl, -, hpi⟩, -⟩ := (Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck'
  have hpi' : s.read.publicInputs = none := by
    cases hp : s.read.publicInputs with
    | none => rfl
    | some _ => rw [hp] at hpi; cases hpi
  have hset := Ins.setup_read hs
  unfold Readable.Read.setup at hset
  rw [hpi'] at hset
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset
  intro j hj
  obtain ⟨d, hd, -⟩ := hall j hj
  rcases hl with hl | ⟨hz, hh, hl⟩
  · exact Or.inl (FlockSoundness.Discharge.HiddenExec.setupOne_tagsAt_hm96 hl hd)
  · rcases FlockSoundness.Discharge.HiddenExec.setupOne_tagsAt_leaf (Or.inr ⟨hh, hl⟩) hd with ⟨-, hm⟩ | ⟨-, hm⟩
    · exact Or.inl hm
    · exact Or.inr ⟨hz, hm⟩

end Gen

/-- **The checks pin the tables' Merkle scheme**: every table the statement sets up commits under `hm96Sha512`, or,
for a statement read without `--zk`, under plain SHA-512 leaves. -/
theorem checks_leaf {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {sts : Array Stmt} (hc : checks zk s spec x link c₁ mids f = .ok ()) (hs : s.setup = .ok (spec, sts)) :
    ∀ (j : ℕ) (hj : j < sts.size), sts[j].merkle = MerkleScheme.hm96Sha512 ∨
      (s.read.zk = false ∧ sts[j].merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") :=
  Gen.checks_leaf (S := S₀) (zk := zk) (s := s) (spec := spec) (x := x) (link := link) (c₁ := c₁) (mids := mids)
    (f := f) (sts := sts) (hc := hc) (hs := hs)

namespace Gen

variable [S : ZkSetup]

/-- A statement set up at `zk` is read at `zk`. -/
theorem setupOf_zk {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf zk x = some (s, spec, sts)) : s.read.zk = zk :=
  readOf_zk (x := x) (by simp [readOf, (setupOf_some h).1])

end Gen

/-- A statement set up at `zk` is read at `zk`. -/
theorem setupOf_zk {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf zk x = some (s, spec, sts)) : s.read.zk = zk :=
  Gen.setupOf_zk (S := S₀) (zk := zk) (x := x) (s := s) (spec := spec) (sts := sts) (h := h)

namespace Gen

variable [S : ZkSetup]

/-- **The checks pin `hm96-sha512/v1` under `--zk`**: every table a masked statement sets up commits under
`hm96Sha512`. -/
theorem checks_hm96 {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {sts : Array Stmt} (hc : checks zk s spec x link c₁ mids f = .ok ()) (hs : S.Ok zk s spec sts)
    (hz : s.read.zk = true) : ∀ (j : ℕ) (hj : j < sts.size), sts[j].merkle = MerkleScheme.hm96Sha512 := by
  intro j hj
  rcases checks_leaf hc hs j hj with h | ⟨h, -⟩
  · exact h
  · rw [hz] at h
    cases h

end Gen

/-- **The checks pin `hm96-sha512/v1` under `--zk`**: every table a masked statement sets up commits under
`hm96Sha512`. -/
theorem checks_hm96 {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {sts : Array Stmt} (hc : checks zk s spec x link c₁ mids f = .ok ()) (hs : s.setup = .ok (spec, sts))
    (hz : s.read.zk = true) : ∀ (j : ℕ) (hj : j < sts.size), sts[j].merkle = MerkleScheme.hm96Sha512 :=
  Gen.checks_hm96 (S := S₀) (zk := zk) (s := s) (spec := spec) (x := x) (link := link) (c₁ := c₁) (mids := mids)
    (f := f) (sts := sts) (hc := hc) (hs := hs) (hz := hz)

end FlockDeploy.KS
