import Proofs.Flock.Claims.Recursion.DeploySessions
import Proofs.Flock.Claims.Recursion.InnerAcceptsKS
import Proofs.Flock.Claims.ZkDecodeFlock
import Proofs.Flock.Claims.Scope.Setup

/-!
# The compiled verifier's margin over the inner rounds

On an accepted run, the inner protocol takes at least one round and the compiled verifier at least three more: the
recursion's `RoundsMargin` at `deployRec` (`roundsMargin_deployRec`). The margin counts the V* sessions' rounds alone:
the staging round (`kS`) sits between the inner rounds and them, so it only adds to it.

* A V* session that sets up runs at least three rounds (`VO_k_ge_three`): the link message, the session's recorded
  rounds and the last message, and the recorded rounds are at least one. The recursive setup makes at least one table
  (`setupTables_setupOne`) and, its statement being masked (`readOfO_zk`), passes each table's `Zk.shapeOk`
  (`RecIns.setup_shapeOk`), which puts its `m` in `[25, 35]` (`m25_of_shapeOk`, `m35_of_shapeOk`), where `Zk.schedule`
  accepts (`zkShape_table`), so each of the table's reps runs rounds (`repRounds_pos`).
* The compiled verifier accepts only at the last session's last round, where the outer verifier accepts (`next_acc`),
  so where that session's statement sets up (`actO_decide`), and that session's rounds are among `k(x)` past the inner
  and staging ones (`roundsMargin_V`).
* The inner protocol's round is its verifier's: at Flock's deployed verifier every statement takes one
  (`V_k_pos_flock`, as `FlockDeploy.Assembled.V_k_pos`).
-/

namespace RecDeploy

open Claims (Msg Prover run)
open Flock (SessionSpec Stmt)
open FlockDeploy (Coin)

/-! ## A session's rounds -/

/-- **The recursive setup passes every table's `Zk.shapeOk`** where the statement is masked. -/
theorem RecIns.setup_shapeOk {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts))
    (hz : s.base.read.zk = true) : sts.forM Flock.Zk.shapeOk = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure, hz, ↓reduceIte] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

/-- Under `--zk`, a statement `Zk.shapeOk` passes has `m ≤ 35` (`queries0` refuses `m > 35`). -/
theorem m35_of_shapeOk {st : Stmt} (h : Flock.Zk.shapeOk st = .ok ()) : st.m ≤ 35 := by
  unfold Flock.Zk.shapeOk at h
  obtain ⟨_, hq, -⟩ := FlockSoundness.except_bind_ok h
  unfold Flock.Zk.queries0 at hq
  by_contra hm
  split_ifs at hq <;> simp only [beq_iff_eq] at * <;> omega

/-- `Zk.schedule` accepts every `m` from 25 to 35. -/
theorem zkSchedule_ok {m : ℕ} (h25 : 25 ≤ m) (h35 : m ≤ 35) : ∃ p, Flock.Zk.schedule m = .ok p := by
  have h := (Claims.Flock.zkShape_table ⟨m - 25, by omega⟩).1
  have hi : Claims.Flock.ZkIdx.m ⟨m - 25, by omega⟩ = m := by
    simp only [Claims.Flock.ZkIdx.m]
    omega
  rw [hi] at h
  cases hs : Flock.Zk.schedule m with
  | ok p => exact ⟨p, rfl⟩
  | error e => rw [hs] at h; simp [Except.toBool] at h

/-- **A masked rep that `Zk.shapeOk` passes runs rounds.** -/
theorem repRounds_pos (spec : SessionSpec) {st : Stmt} (h : Flock.Zk.shapeOk st = .ok ()) :
    0 < FlockDeploy.repRounds true spec st := by
  obtain ⟨⟨sch, t⟩, hp⟩ := zkSchedule_ok (Flock.Readable.m25_of_shapeOk h) (m35_of_shapeOk h)
  simp only [FlockDeploy.repRounds, ↓reduceIte, hp]
  omega

/-- **A masked session with a table, each passing `Zk.shapeOk`, records rounds.** -/
theorem rounds_pos {spec : SessionSpec} {sts : Array Stmt} (hsz : 0 < sts.size)
    (h : sts.forM Flock.Zk.shapeOk = .ok ()) : 0 < FlockDeploy.rounds true spec sts := by
  have h0 := repRounds_pos spec (Flock.Readable.array_forM_ok h 0 hsz)
  have hle : 2 * FlockDeploy.repRounds true spec sts[0]! ≤
      ((List.range sts.size).map fun i => 2 * FlockDeploy.repRounds true spec sts[i]!).sum :=
    List.le_sum_of_mem (List.mem_map.2 ⟨0, List.mem_range.2 hsz, rfl⟩)
  rw [getElem!_pos sts 0 hsz] at hle
  rw [FlockDeploy.rounds_sum]
  omega

/-- The checks run the scope check. -/
theorem scope_of_checks {zk : Bool} {s : FlockDeploy.Ins} {spec : SessionSpec} {x : Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.Final}
    (h : FlockDeploy.checks zk s spec x link c₁ mids f = .ok ()) : s.read.check = .ok () := by
  unfold FlockDeploy.checks at h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨_, h5, -⟩ := FlockSoundness.Refine.ebind_ok h
  exact h5

/-- The scope check refuses `--public-inputs`. -/
theorem publicInputs_none_of_scope {r : Flock.Readable.Read} (h : r.check = .ok ()) : r.publicInputs = none := by
  have hR := ((Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 h).1
  exact Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])

/-- **A V* session that sets up, with no `--public-inputs`, runs at least three rounds**: the link message, at least
one recorded round, and the last message. -/
theorem VO_k_ge_three {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupO xO = some (i, spec, sts)) (hpi : i.base.read.publicInputs = none) : 3 ≤ VO.k xO := by
  obtain ⟨hins, hsu⟩ := setupO_some h
  have hr := RecIns.setup_read hsu
  unfold Flock.Readable.Read.setup at hr
  rw [hpi] at hr
  have hsz := (FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hr).1
  have hz : i.base.read.zk = true := readOfO_zk (r := i.base.read) (by rw [readOfO, hins, Option.map_some])
  have hpos := rounds_pos (spec := spec) hsz (RecIns.setup_shapeOk hsu hz)
  have hk : VO.k xO = 2 + FlockDeploy.rounds true spec sts := by
    show (match setupO xO with | some (_, spec, sts) => 2 + FlockDeploy.rounds true spec sts | none => 1) = _
    rw [h]
  omega

/-- **The outer verifier accepts only at a statement that sets up and passes the scope check**, so one whose session
runs at least three rounds. -/
theorem VO_k_of_acc (H : Msg → Msg) {xO : Msg} {as : List Msg} {cs : List Coin}
    (h : ((nextO xO as cs).eval H).1 = .acc) : 3 ≤ VO.k xO := by
  cases ha : actO xO as cs with
  | rej => rw [nextO_rej H ha] at h; cases h
  | coin c => rw [nextO_coin H ha] at h; cases h
  | decide s spec sts link c₁ mids f =>
    obtain ⟨i, hi, rfl⟩ := actO_decide ha
    obtain ⟨spec', sts', -, -, hc⟩ := (nextO_decide H ha).1 h
    exact VO_k_ge_three hi (publicInputs_none_of_scope (scope_of_checks hc))

/-! ## The margin -/

/-- **On an accepted run, the compiled verifier takes at least three rounds past the inner and staging ones**: it
accepts only at the last session's last round, where the outer verifier accepts (`next_acc`), so that session's
statement sets up and runs at least three rounds (`VO_k_of_acc`), all among `k(x)` past the inner and staging rounds. -/
theorem roundsMargin_V (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r : Fin ((V IV st).k x) → Coin) (h : (run H P (V IV st) x ω r).out = .acc) :
    IV.k (xIOf x) + kS x + 3 ≤ (V IV st).k x := by
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ h
  obtain ⟨s, j, xO, as'', cs', hr, hs, -, hO⟩ := next_acc (IV := IV) (st := st) H ha
  obtain ⟨-, -, hx⟩ := ract_sess_locate hr
  have h3 := VO_k_of_acc H hO
  have hk : VO.k xO = (nums st x s).rounds := by
    rw [hx, vstar_getD st x _ (by omega)]
    exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt st x _ s)
  have hle : (nums st x s).rounds ≤ ((List.range (nS x)).map fun s => (nums st x s).rounds).sum :=
    List.le_sum_of_mem (List.mem_map.2 ⟨s, List.mem_range.2 (by omega), rfl⟩)
  show IV.k (xIOf x) + kS x + 3 ≤ IV.k (xIOf x) + kS x + ((List.range (nS x)).map fun s => (nums st x s).rounds).sum
  omega

/-- **`RoundsMargin` at `deployRec`**: on an accepted run the inner protocol takes at least one round, wherever its
verifier takes one at every statement, and the compiled verifier at least three more. -/
theorem roundsMargin_deployRec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (hk : ∀ x, 0 < I.V.k x) :
    Claims.Recursion.RoundsMargin H (deployRec I.V ivScope b) I :=
  fun x _ P ω r h => ⟨hk _, by
    have := roundsMargin_V H I.V b.st x P ω r h
    show I.V.k (xIOf x) + 3 ≤ (V I.V b.st).k x
    omega⟩

/-- Flock's deployed verifier takes at least one round at every statement (`FlockDeploy.Assembled.V_k_pos`). -/
theorem V_k_pos_flock (zk : Bool) (x : Msg) : 0 < (FlockDeploy.V zk).k x := by
  simp only [FlockDeploy.V]
  split <;> omega

/-- **`RoundsMargin` at `deployRec` over Flock's deployed verifier as the inner protocol**, with no hypothesis. -/
theorem roundsMargin_deployRec_V {W : Type} (zk : Bool) (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (hI : I.V = FlockDeploy.V zk) (ivScope : Msg → Bool) (b : Brings) :
    Claims.Recursion.RoundsMargin H (deployRec I.V ivScope b) I :=
  roundsMargin_deployRec H I ivScope b fun x => hI ▸ V_k_pos_flock zk x

end RecDeploy
