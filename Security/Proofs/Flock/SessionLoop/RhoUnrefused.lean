import Proofs.Flock.SessionLoop.PcsRank

/-!
# A rep's `ρ₀, ρ₁` at its `ρ₁` coins

Past a rep's ring switch its schedule is the opening's binding and level 0's bodies (`Bind.sched_l0`), and the frame
moves through them: the binding starts level 0's masked phase at the algebraic phase's last position, each body absorbs
its values, `ρ₀`'s coins after the last are the frame's first `ρ` words (`Masker.answer`), and at `ρ₁`'s coins the frame
refuses words dependent on them.

* **`Bind.sched_gamma`**: a rep's schedule fed past `γ`'s coins and into level 0's holds the zerocheck's head, the
  opening's label and quiet rounds up to `γ`'s.
* **`RhoKeep.label_rho`**, **`RhoKeep.roundFree_rho`**: a label and a round keep the frame's `ρ` words.
* **`RhoKeep.L0Inv`**, **`RhoKeep.bind_step`**, **`RhoKeep.l0_step`**: the frame inside level 0 after its body `b`,
  from the binding on.
* **`RhoKeep.RhoInv`**, **`RhoKeep.rho_step`**: after `ρ₀`'s coins and `ρ₁`'s round, the frame holds `ρ₀`'s words.
* **`RhoKeep.sched_l0Run`**, **`RhoKeep.repState_rho`**: along a rep's schedule, the frame after level 0's bodies, and
  at `ρ₁`'s coins the pair is independent.
* **`LeanRelease.rho_unrefused_sched_pf`**: `LeanRelease.rho_unrefused_sched`.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.Bind

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Firewall.Frame (Cfg Masker)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)
open _root_.Flock.Firewall.Finish (repOpening repPiop commitFw claimsOf ringSwitchP_eq switchVec)
open FlockSoundness.Refine (Frame encs)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (bind_go liftTM_ok)
open FlockComplete.LigLock.Fed (feedR sentQ pendC feed_eq CoinAt repOpening_feed cutAsks cutAsks_toC openStep_sent)
open Switch (rsBody)
open Loop

/-- **A rep's schedule fed into level 0**: fed the coins of the algebraic phase's rounds, the ring switch's, `γ`'s and
up to level 0's first `j - 1`, a rep's schedule holds the zerocheck's head at round `0`, the opening's label at its
first round and quiet rounds up to `γ`'s. -/
theorem sched_gamma {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hm14 : 14 ≤ t.st.m) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64)
    (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hn0 : t.sch.n ≠ 0) {j : ℕ} (hj : j ≤ t.sch.k0 + 4) (hcr : cr.size = alg.size + (xs.size + 2) + j) :
    sched.size = alg.size + (xs.size + 2) + j + 1 ∧ 1 < alg.size ∧
      sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
        .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
        .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) ∧
      Flags.PcsRound (sched.getD alg.size (.empty, 0)) ∧
      ∀ k, 1 ≤ k → k ≤ alg.size + (xs.size + 2) → k ≠ alg.size → Quiet.QRound (sched.getD k (.empty, 0)) := by
  sorry

end Claims.ZKServer.Bind

namespace Claims.ZKServer.RhoKeep

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot Draws roundFree)
open FlockSoundness.Refine (Frame encs)
open Loop

/-! ## Rounds keep the `ρ` words -/

/-- A label keeps the frame's `ρ` words. -/
theorem label_rho {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (h : s.label c l = .ok q) :
    q.1.rho = s.rho := by
  sorry

/-- A round keeps the frame's `ρ` words, whatever its bytes. -/
theorem roundFree_rho {c : Cfg} {s : Masker} {content : ByteArray} {n : ℕ} {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s content n = .ok q) : q.2.1.rho = s.rho := by
  sorry

/-! ## Level 0 -/

/-- **The frame after level 0's body `b`** (the binding at `b = 0`), owing its `n` coins: in level 0's masked phase at
the next body's first value, with no `ρ` word yet; or it owes nothing (the round refused). -/
def L0Inv (c : Cfg) (b n : ℕ) (s : Masker) : Prop :=
  s.owed = none ∨ (s.owed = some n ∧ s.done = true ∧ s.on = true ∧ s.l0 = true ∧ s.doneL0 = false ∧
    s.inner = none ∧ s.pos = c.nPiop + l0Off (b + 1) ∧ s.rho = #[])

/-- **The binding**: after `γ`'s round, past the opening's label, with no `ρ` word, `γ`'s coins and the binding with
`z`'s squeeze start level 0 at the algebraic phase's last position. -/
theorem bind_step {c : Cfg} {A k : ℕ} {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128}
    {r : ByteArray × ℕ} {T : F128} {cap : ByteArray} (hI : Flags.FInv c A k S.m) (hAk : A ≤ k)
    (hrho : S.m.rho = #[]) (hlig : c.lig = true) (hcap : cap.size < 2 ^ 64)
    (hr : r.1.data.toList = encs (openBind T cap) ++ (Frame.sq 2 r.2).enc) (hn0 : r.2 ≠ 0) (hn : r.2 < 2 ^ 64)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    L0Inv c 0 r.2 (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m := by
  sorry

/-- **Level 0's body `b + 1`**: after body `b`, its coins and body `b + 1`, frames `G` absorbing `l0N (b + 1)` values
inside the layout with its squeeze, keep the frame in level 0. -/
theorem l0_step {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) {b n : ℕ}
    {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128} {r : ByteArray × ℕ} (hI : L0Inv c b n S.m)
    (hsz : c.nPiop + l0Off (b + 1) + l0N (b + 1) ≤ c.order.size) {G : List Frame} (hm : ∀ f ∈ G, midFrame f)
    (hp : ∀ f ∈ G, plainFrame f) (habs : nAbs G = l0N (b + 1))
    (hr : r.1.data.toList = encs G ++ (Frame.sq (l0K (b + 1)) (l0Nc (b + 1))).enc) (hr2 : r.2 = l0Nc (b + 1))
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    L0Inv c (b + 1) r.2 (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m := by
  sorry

/-- **The frame holding `ρ₀`'s words** `r0, r1` at `ρ₁`'s round, owing its two coins, past level 0's last masked value;
or it owes nothing (the round refused). -/
def RhoInv (c : Cfg) (r0 r1 : F128) (s : Masker) : Prop :=
  s.owed = none ∨ (s.owed = some 2 ∧ s.l0 = true ∧ s.on = true ∧ s.inner = none ∧ s.pos = c.order.size ∧
    s.rho = #[r0, r1])

/-- **`ρ₀`'s coins and `ρ₁`'s round**: after level 0's last masked value, owing `ρ₀`'s two coins, its coins and `ρ₁`'s
squeeze leave the frame holding them as its `ρ` words. -/
theorem rho_step {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) {b : ℕ}
    {S : _root_.Flock.Firewall.Frame.State} {cs : Array F128} {r : ByteArray × ℕ} (hI : L0Inv c b 2 S.m)
    (hsz : c.nPiop + l0Off (b + 1) = c.order.size) (hr : r.1.data.toList = (Frame.sq 2 2).enc) (hr2 : r.2 = 2)
    (h2 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).2 = false) :
    RhoInv c cs[0]! cs[1]! (_root_.Flock.Firewall.Frame.run c S [.coins (.ok cs), .round r.2 r.1 (.sched r.1)]).1.m := by
  sorry

/-! ## Along a rep's schedule -/

/-- **A rep's frame through level 0**: where a rep's schedule is fed the coins of its algebraic phase's rounds, the
ring switch's, `γ`'s and level 0's first `j`, the frame after the binding and each body `b ≤ j` up to `k₀ + 2`, where the
rep runs on, is in level 0 owing the body's count (`L0Inv`). -/
theorem sched_l0Run {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hm14 : 14 ≤ t.st.m) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout} {d : Draws}
    {mk : Option MaskSlot} (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
    (hcd : c = Cfg.ofDraws lay d mk) {j : ℕ} (hj : j ≤ t.sch.k0 + 3)
    (hcr : cr.pop.size = alg.size + (xs.size + 2) + 1 + j) (P : Array F128) :
    sched.size = alg.size + (xs.size + 2) + 1 + j + 1 ∧ ∀ b, b ≤ j → b ≤ t.sch.k0 + 2 →
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr (alg.size + (xs.size + 2) + 1 + b))).2 = false →
      L0Inv c b (if b = 0 then t.sch.n else l0Nc b)
        (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr (alg.size + (xs.size + 2) + 1 + b))).1.m := by
  sorry

/-- **At a rep's `ρ₁` coins its pair is independent**: where the loop's state at a rep's round past `ρ₁`'s takes the
round's coins `cs` at a layout of the rep's size, `ρ₀`'s coins and `cs` are independent. -/
theorem repState_rho {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout} {d : Draws}
    {mk : Option MaskSlot} (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
    (hcd : c = Cfg.ofDraws lay d mk) (hord : c''.order.size = c.order.size)
    (hcr : cr.size = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1)).answer c'' cs =
      .ok m) :
    _root_.Flock.Zk.rhoIndependent
      (⟨(cr.getD (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) #[])[0]!,
        (cr.getD (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) #[])[1]!⟩, ⟨cs[0]!, cs[1]!⟩) = true := by
  sorry

end Claims.ZKServer.RhoKeep

namespace Claims.ZKServer.LeanRelease

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open Loop

/-- **The frame refuses dependent `ρ₀, ρ₁` at `ρ₁`'s coins** (`rho_unrefused_sched`): the round past `ρ₁`'s is the
rep's, whose state takes `ρ₁`'s coins, where `ρ₀`'s and `ρ₁`'s are independent (`RhoKeep.repState_rho`), and those are
the opened coins (`wordsAt_slot`). -/
theorem rho_unrefused_sched_pf (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hN : Masked.rhoSlot x j r + 1 < ops.length) :
    _root_.Flock.Zk.rhoIndependent (Masked.rhoAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r) = true := by
  sorry

end Claims.ZKServer.LeanRelease
