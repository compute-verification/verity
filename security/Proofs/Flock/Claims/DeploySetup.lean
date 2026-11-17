module

public import Proofs.Flock.Claims.Bridge.Committed
public import Proofs.Flock.Claims.Bridge.Custody
import Flock.With.Deployed
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-! `KS.ZkSetup`, the deployed verifier's setup, which is all the masked knowledge-soundness chain reads of it. -/

@[expose] public section

namespace FlockDeploy

open Flock
open Lean (Json)

namespace KS

/-- `Ok zk s spec sts` is what the setup's success gave the chain; `wrap` is applied to the checks after the verdict. -/
class ZkSetup where
  ins : Bool → Claims.Msg → Option Ins
  setup : Bool → Claims.Msg → Option (Ins × SessionSpec × Array Stmt)
  wrap : Bool → Claims.Msg → Final → Flock.V Unit → Flock.V Unit
  Ok : Bool → Ins → SessionSpec → Array Stmt → Prop
  setup_some {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} :
    setup zk x = some (s, spec, sts) → ins zk x = some s ∧ Ok zk s spec sts
  ins_zk {zk : Bool} {x : Claims.Msg} {s : Ins} : ins zk x = some s → s.read.zk = zk
  ok_read {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} :
    Ok zk s spec sts → s.read.setup = .ok (spec, sts)
  wrap_ok {zk : Bool} {x : Claims.Msg} {f : Final} {v : Flock.V Unit} : wrap zk x f v = .ok () → v = .ok ()
  ok_shape {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} :
    Ok zk s spec sts → s.read.zk = true → sts.forM Zk.shapeOk = .ok ()

@[reducible] def S₀ : ZkSetup where
  ins := insOf
  setup := setupOf
  wrap _ _ _ v := v
  Ok _ s spec sts := s.setup = .ok (spec, sts)
  setup_some h := setupOf_some h
  ins_zk {zk} {x} {_} h := readOf_zk (zk := zk) (x := x) (by simp [readOf, h])
  ok_read h := Ins.setup_read h
  wrap_ok h := h
  ok_shape h hz := Claims.Bridge.Custody.setup_shapeOk h hz

namespace Gen

variable [S : ZkSetup]

def insOf (zk : Bool) (x : Claims.Msg) : Option Ins := S.ins zk x

def setupOf (zk : Bool) (x : Claims.Msg) : Option (Ins × SessionSpec × Array Stmt) := S.setup zk x

theorem setupOf_some {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf zk x = some (s, spec, sts)) : insOf zk x = some s ∧ S.Ok zk s spec sts :=
  S.setup_some h

theorem Ins.setup_read {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (h : S.Ok zk s spec sts) :
    s.read.setup = .ok (spec, sts) :=
  S.ok_read h

def readOf (zk : Bool) (x : Claims.Msg) : Option Readable.Read := (insOf zk x).map Ins.read

def parse (zk : Bool) (rel : Readable.Parsed → Readable.Relation) : Claims.Msg → Option Claims.Flock.Stmt :=
  Readable.parseOf rel (readOf zk)

def table (zk : Bool) (rel : Readable.Parsed → Readable.Relation) : Claims.Msg → ℕ → Option Claims.Flock.Stmt :=
  Readable.tableOf rel (readOf zk)

def scopeCheck (zk : Bool) : Claims.Msg → Bool := Readable.scopeCheckOf (readOf zk)

def checks (zk : Bool) (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) : Flock.V Unit :=
  S.wrap zk x f (FlockDeploy.checks zk s spec x link c₁ mids f)

theorem checks_base {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) : FlockDeploy.checks zk s spec x link c₁ mids f = .ok () :=
  S.wrap_ok h

def decisionAt (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    Claims.OCV Unit := do
  verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs
  monadLift (checks zk s spec x link c₁ mids f)

def decision (zk : Bool) (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAt zk s spec sts x link c₁ mids f

def act (zk : Bool) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Act :=
  match setupOf zk x with
  | none => .rej
  | some (s, spec, sts) =>
    match as.getLast?, cs.getLast? with
    | some (0 :: _), some c => .coin c
    | some a, _ =>
      match finalOf a, as, cs with
      | some f, a₁ :: rest, c₁ :: crest =>
        match linkOf a₁ with
        | some link => .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) f
        | none => .rej
      | _, _, _ => .rej
    | none, _ => .rej

def next (zk : Bool) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Claims.OC Claims.Out :=
  match act zk x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decision zk s x link c₁ mids f)

/-- `k(x)` is the link message, the session's recorded rounds and the last message; one round where the statement does
not set up. -/
def V (zk : Bool) : Claims.Verifier Coin where
  k x := match setupOf zk x with
    | some (_, spec, sts) => 2 + rounds zk spec sts
    | none => 1
  next := next zk

theorem next_rej (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : act zk x as cs = .rej) : (next zk x as cs).eval H = (.rej, []) := by
  simp only [next, h, Claims.OC.eval]

theorem next_coin (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {c : Coin} (h : act zk x as cs = .coin c) : (next zk x as cs).eval H = (.msg c.bytes, []) := by
  simp only [next, h, Claims.OC.eval]

theorem decisionAt_eval (H : Claims.Msg → Claims.Msg) (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : Final) :
    ((decisionAt zk s spec sts x link c₁ mids f).run.eval H).1 =
        (do verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
            checks zk s spec x link c₁ mids f) ∧
      ((decisionAt zk s spec sts x link c₁ mids f).run.eval H).2.length =
        shaCallsOf zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs := by
  have he := verdictM_eval H zk spec (tabsOf sts) f.record f.proofs
  have hl := verdictM_log_length H zk spec (tabsOf sts) f.record f.proofs
  simp only [decisionAt, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H with ⟨r, l⟩
  rw [hv] at he hl
  simp only at he hl
  subst he
  cases verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs with
  | error e =>
    refine ⟨rfl, ?_⟩
    show (l ++ []).length = _
    rw [List.append_nil, hl]
  | ok u =>
    refine ⟨rfl, ?_⟩
    show (l ++ []).length = _
    rw [List.append_nil, hl]

theorem decision_eval (H : Claims.Msg → Claims.Msg) (zk : Bool) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    ((decision zk s x link c₁ mids f).run.eval H).1 =
        (do let (spec, sts) ← setupWith (Claims.hOf H) s.read
            verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
            checks zk s spec x link c₁ mids f) ∧
      ((decision zk s x link c₁ mids f).run.eval H).2.length =
        setupCallsOf (Claims.hOf H) s.read +
          match setupWith (Claims.hOf H) s.read with
          | .ok (spec, sts) => shaCallsOf zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
          | .error _ => 0 := by
  obtain ⟨he, hl⟩ := setupM_eval H s.read
  simp only [decision, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at he hl
  simp only at he hl
  subst he
  cases hr : setupWith (Claims.hOf H) s.read with
  | error e =>
    refine ⟨rfl, ?_⟩
    show (l ++ []).length = _
    rw [List.append_nil, hl, Nat.add_zero]
  | ok p =>
    obtain ⟨spec, sts⟩ := p
    obtain ⟨he', hl'⟩ := decisionAt_eval H zk s spec sts x link c₁ mids f
    exact ⟨he', by rw [List.length_append, hl, hl']⟩

theorem next_decide (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval H).1 = .acc ↔
      ∃ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        verdictWith zk (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs = .ok () ∧
          checks zk s spec' x link c₁ mids f = .ok () := by
  obtain ⟨he, -⟩ := decision_eval H zk s x link c₁ mids f
  simp only [next, h, outOf, Claims.OC.eval_bind, Claims.OC.eval]
  rw [he]
  cases hs : setupWith (Claims.hOf H) s.read with
  | error e => simp [bind, Except.bind]
  | ok p =>
    obtain ⟨spec', sts'⟩ := p
    cases hv : verdictWith zk (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs with
    | error e => simp [bind, Except.bind, hv]
    | ok u =>
      cases hc : checks zk s spec' x link c₁ mids f with
      | error e => simp [bind, Except.bind, hv, hc]
      | ok u' => simp [bind, Except.bind, hv, hc]

theorem next_decide_log (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval H).2.length =
      setupCallsOf (Claims.hOf H) s.read +
        match setupWith (Claims.hOf H) s.read with
        | .ok (spec', sts') => shaCallsOf zk (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs
        | .error _ => 0 := by
  obtain ⟨-, hl⟩ := decision_eval H zk s x link c₁ mids f
  simp only [next, h, outOf, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]
  exact hl

theorem act_decide {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    setupOf zk x = some (s, spec, sts) := by
  unfold act at h
  split at h
  · cases h
  · rename_i s' spec' sts' hs
    repeat' split at h
    all_goals first | (cases h; done) | (cases h; exact hs)

theorem verdictWith_sha512 {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf zk x = some (s, spec, sts)) (record : String) (proofs : Array ByteArray) :
    verdictWith zk (Claims.hOf sha512) spec (tabsOf sts) record proofs =
      if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs := by
  have hr := Ins.setup_read (setupOf_some h).2
  rw [hOf_sha512]
  unfold verdictWith tabsOf
  cases zk
  · exact (verify_eq_verifyWith_of_setupTables _ _ _ _ _ _ _ _ _ _ _ _ _ hr record proofs).symm
  · exact (Zk.verify_eq_verifyWith_of_setupTables _ _ _ _ _ _ _ _ _ _ _ _ _ hr record proofs).symm

theorem act_decide_setupWith {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    setupWith (Claims.hOf sha512) s.read = .ok (spec, sts) := by
  rw [hOf_sha512, setupWith_sha512]
  exact Ins.setup_read (setupOf_some (act_decide h)).2

theorem next_decide_sha512 {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval sha512).1 = .acc ↔
      (if zk then Zk.verify spec (tabsOf sts) f.record f.proofs else verify spec (tabsOf sts) f.record f.proofs) =
          .ok () ∧
        checks zk s spec x link c₁ mids f = .ok () := by
  rw [next_decide sha512 h, act_decide_setupWith h, ← verdictWith_sha512 (act_decide h)]
  constructor
  · rintro ⟨spec', sts', he, hv, hc⟩
    cases he
    exact ⟨hv, hc⟩
  · rintro ⟨hv, hc⟩
    exact ⟨spec, sts, rfl, hv, hc⟩

theorem next_decide_log_sha512 {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval sha512).2.length =
      setupCallsOf Sha512.hash s.read + shaCallsOf zk Sha512.hash spec (tabsOf sts) f.record f.proofs := by
  rw [next_decide_log sha512 h, act_decide_setupWith h, hOf_sha512]

theorem readOf_zk {zk : Bool} {x : Claims.Msg} {r : Readable.Read} (h : readOf zk x = some r) : r.zk = zk := by
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  exact S.ins_zk hs


theorem Claims.Bridge.Custody.setup_shapeOk {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : S.Ok zk s spec sts) (hz : s.read.zk = true) : sts.forM Zk.shapeOk = .ok () :=
  S.ok_shape h hz

section
open _root_.Claims _root_.Claims.Bridge

open _root_.Claims _root_.Claims.Bridge



theorem Claims.Bridge.act_coin {zk : Bool} {x : Msg} {as : List Msg} {cs : List Coin} {c : Coin}
    (h : act zk x as cs = .coin c) : cs.getLast? = some c := by
  unfold act at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

theorem Claims.Bridge.answersCoin_deploy (zk : Bool) (H : Msg → Msg) (x : Msg) :
    AnswersCoin H (V zk) x Coin.bytes := by
  intro as cs c m h
  change ((next zk x as (cs ++ [c])).eval H).1 = .msg m at h
  cases ha : act zk x as (cs ++ [c]) with
  | rej =>
    rw [next_rej H ha] at h
    cases h
  | coin c' =>
    rw [next_coin H ha] at h
    have hc : c = c' := by simpa using act_coin ha
    rw [← Out.msg.inj h, hc]
  | decide s spec sts link c₁ mids f =>
    simp only [next, ha] at h
    exact absurd h (_root_.Claims.Bridge.outOf_ne_msg _ _ _)

end

section
open _root_.Flock FlockSoundness FlockSoundness.Discharge

open _root_.Flock FlockSoundness FlockSoundness.Discharge
open Lean (Json)



theorem Claims.Bridge.Custody.checks_ok {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    f.stmt = x ∧ ∃ lv, liveOf spec f.record = .ok lv ∧ custody spec lv link c₁ mids = .ok () :=
  _root_.Claims.Bridge.Custody.checks_ok (checks_base h)

theorem Claims.Bridge.Custody.publicInputs_none_of_checks {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    s.read.publicInputs = none :=
  _root_.Claims.Bridge.Custody.publicInputs_none_of_checks (checks_base h)

theorem Bridge.Custody.checks_ok {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    f.stmt = x ∧ ∃ lv, liveOf spec f.record = .ok lv ∧ custody spec lv link c₁ mids = .ok () :=
  _root_.Claims.Bridge.Custody.checks_ok (checks_base h)

theorem Bridge.Custody.publicInputs_none_of_checks {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    s.read.publicInputs = none :=
  _root_.Claims.Bridge.Custody.publicInputs_none_of_checks (checks_base h)

theorem checks_ok {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    f.stmt = x ∧ ∃ lv, liveOf spec f.record = .ok lv ∧ custody spec lv link c₁ mids = .ok () :=
  _root_.Claims.Bridge.Custody.checks_ok (checks_base h)

theorem publicInputs_none_of_checks {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    s.read.publicInputs = none :=
  _root_.Claims.Bridge.Custody.publicInputs_none_of_checks (checks_base h)

theorem Claims.Bridge.Custody.checks_scope {zk : Bool} {s : Ins} {spec : SessionSpec} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : checks zk s spec x link c₁ mids f = .ok ()) :
    s.read.check = .ok () :=
  _root_.Claims.Bridge.Custody.checks_scope (checks_base h)

theorem Claims.Bridge.Custody.publicInputs_none_of_inScope {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {st : Claims.Flock.Stmt} {s : Ins} (hp : parse zk rel x = some st)
    (hin : Claims.Flock.InScope zk st) (hs : insOf zk x = some s) : s.read.publicInputs = none := by
  obtain ⟨p, hpb, rfl⟩ := Option.map_eq_some_iff.1 hp
  obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hpb
  have hr' : r = s.read := by
    simp only [readOf, hs, Option.map_some] at hr
    exact (Option.some.inj hr).symm
  obtain ⟨-, -, -, -, -, -, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at hpi
  rw [← hr', ← Readable.toRead_of_parse hrp]
  exact Option.not_isSome_iff_eq_none.1 (by simp [hpi])

theorem publicInputs_none_of_inScope {zk : Bool} {rel : Readable.Parsed → Readable.Relation} {x : Claims.Msg}
    {st : Claims.Flock.Stmt} {s : Ins} (hp : parse zk rel x = some st)
    (hin : Claims.Flock.InScope zk st) (hs : insOf zk x = some s) : s.read.publicInputs = none :=
  Claims.Bridge.Custody.publicInputs_none_of_inScope hp hin hs

end

end Gen

example : @Gen.insOf S₀ = insOf := rfl
example : @Gen.setupOf S₀ = setupOf := rfl
example : @Gen.readOf S₀ = readOf := rfl
example : @Gen.checks S₀ = checks := rfl
example : @Gen.act S₀ = act := rfl
example : @Gen.decision S₀ = decision := rfl
example : @Gen.next S₀ = next := rfl
example : @Gen.V S₀ = V := rfl
example : @Gen.parse S₀ = parse := rfl
example : @Gen.scopeCheck S₀ = scopeCheck := rfl

end KS

end FlockDeploy
