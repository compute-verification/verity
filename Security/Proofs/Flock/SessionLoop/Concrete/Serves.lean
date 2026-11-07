module

public import Proofs.Flock.SessionLoop.Concrete.Scope
public import Proofs.Properties.Flock.Nets
public import Proofs.Flock.Claims.ZK.FrameZK
public import Proofs.Flock.Claims.Complete.Zk.Json.Stmt
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Soundness.Discharge.Exec.Setup
public import Proofs.Flock.Soundness.Discharge.Exec.Retained
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
public import Proofs.Flock.VBridge.Family.Lig

/-! # The session loop serves the concrete statement

`Claims.ZKServer.Loop.Serves` at `FlockDeploy.concreteStmt`: from a fresh session, the loop's `Hello` leaves at the
concrete statement's setup, one table set up by `Stmt.setupH` under `Concrete.tz` and the keyed coin tree.

* `Loop.sizesOk_one`: one table's session passes `Session.sizesOk` where the table's link points, regions, rep stream
  domains and frame start are within the frames'; `Loop.sizesOk_setupH`: `Stmt.setupH`'s range check
  (`checkInRange`) and the `--zk` schedule's at most 6 lanes (`Loop.schedule_k0`) put them there.
* `Loop.msg_ne_nil`: the setup's `Hello` at the zero nonce `Loop.z32` is JSON (`helloZk_of`, `canon_ok_some`), so
  `Hello.msg` frames it.
* `Loop.concreteServes`: so `State.sendHello` lets it leave.
-/

@[expose] public section

namespace Claims.ZKServer.Loop

open _root_.Flock (Stmt SessionSpec Tags Coins CoinTree PT_LOCAL)
open _root_.Flock.Firewall (Hello Drive)
open _root_.Flock.Firewall.Session (sizesOk State)

private theorem ok_bind {α β : Type} {y : _root_.Flock.V α} {f : α → _root_.Flock.V β} {b : β}
    (h : y >>= f = .ok b) : ∃ a, y = .ok a ∧ f a = .ok b := by
  cases y with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-! ## Sizes -/

/-- The `--zk` schedule has at most 6 lanes at level 0, at every `m` below 68. -/
theorem schedule_k0 : ∀ m < 68, (match _root_.Flock.Zk.schedule m with
    | .ok (s, _) => decide (s.k0 ≤ 6)
    | .error _ => true) = true := by
  decide +kernel

theorem schedule_ok_k0 {m : ℕ} (hm : m < 68) {p : _root_.Flock.Schedule × ℕ}
    (h : _root_.Flock.Zk.schedule m = .ok p) : p.1.k0 ≤ 6 := by
  obtain ⟨sch, q⟩ := p
  have hk := schedule_k0 m hm
  rw [h] at hk
  simpa using hk

/-- **A rep's frame start leaves its pads and pad row within a `u64`'s count** at a statement with `m` below 68 and
`k_log` at most 27. -/
theorem start_len {st : Stmt} (hm : st.m < 68) (hk : st.c.kLog ≤ 27) {lay : _root_.Flock.Zk.PadLayout} {mask}
    (h : Drive.start st = .ok (lay, mask)) :
    lay.len + _root_.Flock.Zk.K_PAD < 2 ^ 64 := by
  unfold Drive.start at h
  obtain ⟨⟨sch, q⟩, hs, h⟩ := ok_bind h
  obtain ⟨ms, -, h⟩ := ok_bind h
  have hlay : lay = _root_.Flock.Zk.PadLayout.new st.m st.c.kLog sch.k0 := by
    cases h
    rfl
  have hk0 : sch.k0 ≤ 6 := schedule_ok_k0 hm hs
  subst hlay
  simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw, _root_.Flock.Zk.PadLayout.hab,
    _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc, _root_.Flock.Zk.PadLayout.fc,
    _root_.Flock.Zk.PadLayout.fa, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.K_PAD]
  omega

/-- **One table's session passes `sizesOk`** where its link points, its regions, its rep streams' domains and its
frame start are within the frames'. -/
theorem sizesOk_one {st : Stmt} (hm0 : 0 < st.mPts) (hm : st.mPts ≤ 64) (hreg : st.regions.size ≤ 1024)
    (hdom : ∀ r < 2, (st.tags.domain r).toUTF8.size < 2 ^ 64)
    (hlay : ∀ lay mask, Drive.start st = .ok (lay, mask) → lay.len + _root_.Flock.Zk.K_PAD < 2 ^ 64) :
    sizesOk st.spec #[st] = true := by
  have hrs : st.spec.repStreams.size = 2 := by
    first
    | rfl
    | simp [SessionSpec.repStreams, Stmt.spec]
  simp only [sizesOk, List.all_eq_true, List.mem_range, hrs]
  intro k hk
  have hk2 : k / 2 = 0 := by omega
  rw [hk2]
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨⟨rfl, hm0, ?_⟩, hdom _ (Nat.mod_lt _ (by norm_num))⟩, ?_⟩, ?_⟩
  · show 2 * st.mPts ≤ 65536
    omega
  · show st.regions.size * 65536 + 2 < 2 ^ 64
    have h1 : st.regions.size * 65536 + 2 ≤ 1024 * 65536 + 2 := by omega
    exact lt_of_le_of_lt h1 (by norm_num)
  · split
    · rename_i lay mask hs
      exact decide_eq_true (hlay lay mask hs)
    · rfl

/-- The rep streams' domains under the concrete statement's tags. -/
theorem dom_tz : ∀ r < 2, (FlockDeploy.Concrete.tz.domain r).toUTF8.size < 2 ^ 64 := by
  decide +kernel

/-- **The concrete tags' one-table setups pass `sizesOk`**: `Stmt.setupH` checks `k_log ≤ 27`, at most 1,024 regions and
at most 64 link points (`setupH_inRange`), with `PT_LOCAL + nbl` link points and `m = k_log + nbl` (`setupH_wf`). -/
theorem sizesOk_setupH {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH FlockDeploy.Concrete.tz cf pf coins tables draw partition program = .ok st) :
    sizesOk st.spec #[st] = true := by
  have hwf := (FlockSoundness.Refine.setupH_wf h).2
  obtain ⟨hk, hreg, hpts⟩ := (_root_.Flock.checkInRange_ok _ _ _).1 (FlockVBridge.Family.setupH_inRange h)
  have htg : st.tags = FlockDeploy.Concrete.tz := FlockSoundness.Discharge.Exec.setupH_tags h
  have hmP : st.mPts = PT_LOCAL + st.nbl := hwf.mPts
  have hm : st.m = st.c.kLog + st.nbl := hwf.hm
  have hpt : PT_LOCAL = 24 := rfl
  refine sizesOk_one ?_ ?_ hreg ?_ ?_
  · omega
  · omega
  · intro r hr
    rw [htg]
    exact dom_tz r hr
  · intro lay mask hs
    exact start_len (by omega) hk hs

/-! ## The `Hello` -/

/-- The nonce of 32 zero bytes, whose hex is the 64 zeros of the honest prover's `Hello` (`helloJ`). -/
def z32 : ByteArray := ⟨Array.replicate 32 0⟩

theorem hexOf_z32 : _root_.Flock.hexOf z32 = FlockComplete.zeros 64 := by
  decide +kernel

/-- **The `Hello` at the zero nonce is JSON** at a table set up under the keyed coin tree `cs`, so `Hello.msg` frames
it. -/
theorem msg_ne_nil {tags : Tags} {cf pf : ByteArray} {cs : CoinTree.Spec} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : FlockSoundness.Discharge.HiddenExec.setupOne tags cf pf (.tree cs) tables draw partition program = .ok st) :
    Hello.msg st.spec z32 ≠ [] := by
  obtain ⟨-, -, -, hhello, hct⟩ := FlockComplete.JsonRT.setupOne_stmt h
  obtain ⟨m, kvs, hp, -, -, -, hg⟩ := FlockComplete.JsonRT.helloZk_of hhello
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok_some _ hg
  have hct' : st.spec.coinTree = some cs := hct
  have hp' : _root_.Flock.JsonParse.parse st.spec.hello = .ok (.obj m) := hp
  have hj : Hello.json st.spec (_root_.Flock.hexOf z32) = .ok s := by
    simp only [Hello.json, hct', hp', Except.toOption, hexOf_z32]
    exact hs
  have hm : Hello.msg st.spec z32 = Hello.frame s := by
    simp only [Hello.msg, hj]
  rw [hm]
  unfold Hello.frame
  exact List.cons_ne_nil _ _

/-! ## The loop serves the concrete statement -/

/-- **The loop's `Hello` leaves** from a fresh session at a setup passing `sizesOk` whose `Hello` is JSON. -/
theorem sendHello_ok {spec : SessionSpec} {sts : Array Stmt} {nonce : ByteArray} (hz : sizesOk spec sts = true)
    (hm : Hello.msg spec nonce ≠ []) : ∃ r, ({} : State).sendHello (.ok (spec, sts)) nonce = .ok r := by
  unfold State.sendHello
  try dsimp only
  simp only [hz, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  cases hmsg : Hello.msg spec nonce with
  | nil => exact absurd hmsg hm
  | cons b bs => exact ⟨_, rfl⟩

/-- **The loop serves every concrete-file statement's setup** (`Concrete.rC`): its one table is `Stmt.setupH`'s under
`Concrete.tz`, at the keyed coin tree. -/
theorem serves_rC {ls lh : _root_.Flock.HmNets.Laid} {H : ByteArray} {spec : SessionSpec} {sts : Array Stmt}
    (h : (FlockDeploy.Concrete.rC ls lh H).setup = .ok (spec, sts)) :
    ∃ nonce r, ({} : State).sendHello (.ok (spec, sts)) nonce = .ok r := by
  have ht : (FlockDeploy.Concrete.rC ls lh H).tags = FlockDeploy.Concrete.tz := FlockDeploy.Concrete.tags_rC ls lh H
  obtain ⟨cs, hcs⟩ : ∃ cs, (FlockDeploy.Concrete.rC ls lh H).coins = .tree cs := ⟨_, rfl⟩
  unfold _root_.Flock.Readable.Read.setup at h
  rw [ht, hcs] at h
  rcases FlockSoundness.Discharge.Exec.setupTables_ok FlockDeploy.Concrete.hidden_tz h with
    ⟨-, st, hst, rfl, rfl⟩ | ⟨hc, -⟩
  · exact ⟨z32, sendHello_ok (sizesOk_setupH hst)
      (msg_ne_nil ((FlockSoundness.Discharge.HiddenExec.setupOne_public FlockDeploy.Concrete.hidden_tz).trans hst))⟩
  · exact absurd hc (by show ¬ (1 < 1); omega)

/-- **The session loop serves the concrete statement** (`Serves`). -/
theorem concreteServes : Serves FlockDeploy.concreteStmt := by
  intro s0 spec sts h
  rw [FlockDeploy.concreteStmt_setupOf _root_.Properties.Flock.Nets.fixedNetsBuild] at h
  obtain ⟨-, hss⟩ := Prod.mk.inj (Option.some.inj h)
  have hR : FlockDeploy.concreteIns.read.setup = .ok (spec, sts) := by
    rw [← hss]
    exact FlockDeploy.Ins.setup_read (spec := FlockDeploy.concreteSetup.1) (sts := FlockDeploy.concreteSetup.2)
      (FlockDeploy.concreteIns_setup _root_.Properties.Flock.Nets.fixedNetsBuild)
  exact serves_rC (ls := _root_.Flock.HmNets.shaLaid ()) (lh := _root_.Flock.HmNets.hmLaid ())
    (H := FlockDeploy.concreteHash) hR

end Claims.ZKServer.Loop
