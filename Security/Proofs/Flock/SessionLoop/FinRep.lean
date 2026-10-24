import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.Claims.ZK.FinishBridge
import Proofs.Flock.Claims.ZK.FrameReal
import Proofs.Flock.Claims.Complete.ZkTables.Scope

/-!
# `FinishParts`' (c), the reps' proof files, in parts

`Loop.FinRepProofs` asks that rep stream `k`'s run from the views (`Loop.viewRep`) return a proof whose file is the loop's
`k`-th, `Finish.proofFile` on the loop's table (`Loop.realOf`) and request (`Loop.repInOf`). `proofFile` runs the
firewall's rep (`repFw`, answered by the request's coins from round `0`) and encodes its proof (`proofFile_ok`). So (c) is
two parts, each named here, none a premise of a guarantee:

* **`Loop.FinRepCut`**: where the firewall's rep on the loop's table and request returns a proof, the session's rep
  (`repFwC`) answered by the opened coins (`asksCut`), from a history at the rep's first round, returns the same.
* **(c.3) `Loop.FinRepFwView`**: where that session's rep returns a proof, the rep from the views returns a proof whose
  file at the session's shape is the firewall's at the table's.

**`Loop.finRepProofs_of_parts`**: they give `FinRepProofs`.

`FinRepCut` is `repFw_cut` at the opened coins, which agree with the request's only below each round's count: an opening
carries the round's `n` coins, then `T_n`, its salt and its path (`CoinLayer.openOf`), and the session's coin is all of it
(`CoinLayer.rd`), where the firewall answers `0` past `n` (`asksOf`). A squeeze reads only its first `n` coins
(`squeezeP`), and the firewall checks `n` against the request's (`asksOf`), so the two runs agree, given every squeeze's
count below `2^64` (the round header's `u64`). Its premises, stated here:

* **(c.1) `Loop.FinRepCoins`**: the request's coins are the opened coins from the rep's first round (`Masked.repBase`),
  below each round's count, and the rep ends by the last opening. It holds in the masked scope (`Loop.finRepCoins`).
* **(c.2) `Loop.FinRepPads`**: the rep's table commits under `hm96-sha512`, and the request's pads have the layout's
  length, `h_ab = h_a·h_b` and `w = x·y` (`repFw_eq`'s premises). It holds in the masked scope (`Loop.finRepPads`).
-/

namespace Flock.Firewall.Finish

open Flock With Transcript

/-- **A proof file that `proofFile` returns**: the request's pads commitment returned, the firewall's rep answered by
the request's coins returned at its last coin, and the file is its proof's `encZk` at `shapeZk`. -/
theorem proofFile_ok {t : Tab} {rep : ℕ} {dom : ByteArray} {q : RepIn} {f : ByteArray}
    (h : proofFile t rep dom q = .ok f) :
    ∃ ic p tr, (innerCommitP (Hashes.ofSha512 Sha512.hash) (padsOf q.draws) (saltAt q.draws.salts) : V InnerCommit) =
        .ok ic ∧
      ((repFw (asksOf q.coins) (Hashes.ofSha512 Sha512.hash) (fun lv g s => pure (commitFw lv g s)) t.st t.sch t.tPad
        rep dom t.g t.cm0 (claimsOf q.xs) (padsOf q.draws) ic q.vals t.extra t.lanePad
        (fun l => saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run (Transcript.init #[] dom)).run 0 =
        .ok ((p, tr), q.coins.size) ∧
      f = encZk (shapeZk t.st t.sch (claimsOf q.xs).size) p := by
  unfold proofFile at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i ic hic
    split at h
    · cases h
    · rename_i a hrun
      obtain ⟨⟨p, tr⟩, k⟩ := a
      by_cases hk : k = q.coins.size
      · subst hk
        refine ⟨ic, p, tr, hic, hrun, ?_⟩
        simp at h
        exact (Except.ok.inj h).symm
      · exfalso
        simp [hk] at h

end Flock.Firewall.Finish

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **Rep stream `k`'s proof file, as the loop's `Finish` makes it**: its table and salts for levels ≥ 1 (`realOf`), its
request (`repInOf`, at the frame's draws) and the file `proofFile` returns. -/
def RepAt {R : Type} (draws : R → ℕ → ByteArray)
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) (x : Msg) (w : Flock.Wit) (ρ : R)
    (h : List (Msg × Msg)) (spec : SessionSpec) (k : ℕ) (tb : _root_.Flock.Firewall.Finish.Tab)
    (hi : Array ByteArray) (q : _root_.Flock.Firewall.Finish.RepIn) (f : ByteArray) : Prop :=
  realOf src x w ρ k = .ok (tb, hi) ∧ repInOf x h spec (Masked.stsOf x) k (draws ρ k) hi = .ok q ∧
    _root_.Flock.Firewall.Finish.proofFile tb (k % 2) (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 q = .ok f

/-- **`proofsOf`'s `k`-th file is the loop's rep stream `k`'s** (`RepAt`). -/
theorem proofsOf_at {R : Type} {draws : R → ℕ → ByteArray}
    {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit} {ρ : R}
    {h : List (Msg × Msg)} {spec : SessionSpec} {files : Array ByteArray}
    (hp : proofsOf src draws x w ρ h spec (Masked.stsOf x) = .ok files) {k : ℕ} (hk : k < spec.repStreams.size) :
    ∃ tb hi q, RepAt draws src x w ρ h spec k tb hi q files[k]! := by
  unfold proofsOf at hp
  obtain ⟨hsz, hall⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hp
  have e := hall k (by rw [hsz]; exact hk)
  rw [← getElem!_pos files k (by rw [hsz]; exact hk)] at e
  simp only [bind, Except.bind] at e
  split at e
  · cases e
  · rename_i a ha
    obtain ⟨tb, hi⟩ := a
    try dsimp only at e
    split at e
    · cases e
    · rename_i q hq
      exact ⟨tb, hi, q, ha, hq, e⟩

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **(c.1) The request's coins**: at a `Finish` `FinishIs` reads, rep stream `k`'s request's coins are the opened coins
from the rep's first round (`Masked.repBase`), below each round's count, and the rep's rounds end by the last opening. -/
def FinRepCoins : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ k < 2 * spec.tables.size, ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
      Masked.repBase x k + q.coins.size ≤ refK x ∧
        ∀ k' < q.coins.size, ∀ i < (q.coins[k']!).size,
          Masked.slotAt (finCoins x ops) (Masked.repBase x k + k') i = (q.coins[k']!)[i]!

theorem take_map_range (f : ℕ → ℕ) {k N : ℕ} (h : k ≤ N) :
    ((List.range N).map f).take k = (List.range k).map f := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_map, List.length_range]
    omega
  · intro i h1 h2
    simp

/-- **(c.1) holds in the masked scope**: the request's coins are `seenRounds`', rep stream `k`'s round `k'` the
history's entry `repBase x k + k'` past `Hello` (`streamAt` is `repStartZk` less the link's round), whose words below the
round's count are the opened coin's (`CoinLayer.rd`), and the rep's last round is before `repStartZk` of the last
stream, `refK x` (`repStart_last`, `refK_eq`). -/
theorem finRepCoins (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) :
    FinRepCoins L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q f hR
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  have hops : ops.length = refK x := hctx.1
  obtain ⟨-, hq, -⟩ := hR
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
  have hqc : q.coins =
      ((seenRounds x ((m₀, ρ₀) :: ops) (roundsOf spec (Masked.stsOf x)) k).map fun r => r.2.2).toArray := by
    cases hq
    rfl
  have hrk : (roundsOf spec (Masked.stsOf x)).getD k 0 = FlockDeploy.repRounds true spec (Masked.stsOf x)[k / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hsz : q.coins.size = FlockDeploy.repRounds true spec (Masked.stsOf x)[k / 2]! := by
    rw [hqc, List.size_toArray, List.length_map]
    simp only [seenRounds, List.length_map, List.length_range, hrk]
  have hbase : Masked.repBase x k = FlockComplete.repStartZk spec (Masked.stsOf x) k := Masked.repBase_eq hsx k
  have hstream : streamAt (roundsOf spec (Masked.stsOf x)) k + 1 = Masked.repBase x k := by
    rw [hbase]
    unfold streamAt roundsOf FlockComplete.repStartZk
    rw [take_map_range _ (by omega)]
    first
    | omega
    | rw [Nat.add_comm]
  have hend : Masked.repBase x k + q.coins.size ≤ refK x := by
    rw [hsz, hbase, ← FlockComplete.ZkTables.repStart_succ, Masked.refK_eq hsx,
      ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  refine ⟨hend, fun k' hk' i hi => ?_⟩
  have hQ : Masked.repBase x k + k' < ops.length := by omega
  have hQK : Masked.repBase x k + k' < refK x := by omega
  have hent : q.coins[k']! =
      wordsAt x (ops.map fun p => CoinLayer.rd p.2) (Masked.repBase x k + k') := by
    rw [getElem!_pos q.coins k' hk']
    simp only [hqc, List.getElem_toArray, List.getElem_map]
    unfold seenRounds
    simp only [List.getElem_map, List.getElem_range, List.tail_cons]
    congr 1
    omega
  have hw : wordsAt x (ops.map fun p => CoinLayer.rd p.2) (Masked.repBase x k + k') =
      (CoinLayer.rd ops[Masked.repBase x k + k'].2).vals
        (min (loopNr x (Masked.repBase x k + k')) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold wordsAt
    rw [List.getElem?_map, List.getElem?_eq_getElem hQ]
    rfl
  rw [hent, hw] at hi ⊢
  have hi' : i < min (loopNr x (Masked.repBase x k + k')) _root_.Flock.Firewall.Frames.COIN_MAX := by
    simpa [FlockDeploy.Coin.vals] using hi
  rw [getElem!_pos _ i hi]
  simp only [FlockDeploy.Coin.vals, Array.getElem_map, Array.getElem_range]
  unfold Masked.slotAt
  rw [dif_pos hQK]
  show ((ops.map fun p => CoinLayer.rd p.2).getD (Masked.repBase x k + k') default).at i = _
  rw [List.getD_eq_getElem _ _ (by simpa using hQ), List.getElem_map]

/-- **(c.2) The rep's table and pads**: at a `Finish` `FinishIs` reads, rep stream `k`'s table commits under
`hm96-sha512`, and its request's pads have the layout's length, `h_ab = h_a·h_b` and `w = x·y`. -/
def FinRepPads : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ k < 2 * spec.tables.size, ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
      tb.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 ∧
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h.size =
        (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).len ∧
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).hab]! =
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fa]! *
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fb]! ∧
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).tw]! =
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).tx]! *
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).ty]!

/-- **A rep's draws are pads `repFw_cut` takes**: `Draws.ofBytes` reads `h` at the layout's length and sets `h_ab := h_a·h_b`
and `w := x·y`, no index of which the other set moves. -/
theorem drawsOfBytes_pads (lay : _root_.Flock.Zk.PadLayout) (b : ByteArray) :
    (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h.size = lay.len ∧
      (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[lay.hab]! =
        (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[lay.fa]! *
        (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[lay.fb]! ∧
      (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[lay.tw]! =
        (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[lay.tx]! *
        (_root_.Flock.Firewall.Frame.Draws.ofBytes lay b).h[lay.ty]! := by
  have gs : ∀ (B : Array F128) {i : ℕ} (j : ℕ) (v : F128), i < B.size → (B.set! i v)[j]! = if j = i then v else B[j]! := by
    intro B i j v hi
    by_cases hj : j < B.size
    · have hj' : j < (B.set! i v).size := by simpa using hj
      rw [getElem!_pos _ j hj', getElem!_pos _ j hj]
      simp only [Array.set!, Array.getElem_setIfInBounds hj]
      by_cases h : i = j
      · simp [h]
      · simp [h, Ne.symm h]
    · have hj' : ¬ j < (B.set! i v).size := by simpa using hj
      have hne : j ≠ i := by omega
      rw [getElem!_neg _ j hj', getElem!_neg _ j hj, if_neg hne]
  have hs : (_root_.Flock.Firewall.Frame.words b 0 lay.len).size = lay.len := by
    unfold _root_.Flock.Firewall.Frame.words
    rw [Array.size_map, Array.size_range]
  have hfa : lay.fa + 1 = lay.fb := rfl
  have hab : lay.fb + 1 < lay.hab := by
    simp only [_root_.Flock.Zk.PadLayout.hab, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc,
      _root_.Flock.Zk.PadLayout.fc, _root_.Flock.Zk.PadLayout.fb]
    omega
  have htx : lay.tx = lay.hab + 1 := rfl
  have hty : lay.ty = lay.hab + 2 := rfl
  have htw : lay.tw = lay.hab + 3 := rfl
  have hlen : lay.len = lay.tw + 1 := rfl
  have key : ∀ h0 : Array F128, h0.size = lay.len →
      ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)).set! lay.tw
          ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.tx]! *
            (h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.ty]!)).size = lay.len ∧
      ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)).set! lay.tw
          ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.tx]! *
            (h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.ty]!))[lay.hab]! =
        ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)).set! lay.tw
          ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.tx]! *
            (h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.ty]!))[lay.fa]! *
        ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)).set! lay.tw
          ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.tx]! *
            (h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.ty]!))[lay.fb]! ∧
      ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)).set! lay.tw
          ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.tx]! *
            (h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.ty]!))[lay.tw]! =
        ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)).set! lay.tw
          ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.tx]! *
            (h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.ty]!))[lay.tx]! *
        ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!)).set! lay.tw
          ((h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.tx]! *
            (h0.set! lay.hab (h0[lay.fa]! * h0[lay.fb]!))[lay.ty]!))[lay.ty]! := by
    intro h0 hs0
    generalize hv1 : h0[lay.fa]! * h0[lay.fb]! = v1
    generalize hh1 : h0.set! lay.hab v1 = h1
    have hs1 : h1.size = lay.len := by rw [← hh1, Array.size_set!, hs0]
    have e1 : ∀ j, h1[j]! = if j = lay.hab then v1 else h0[j]! := fun j => by
      rw [← hh1]; exact gs h0 j v1 (by omega)
    generalize hv2 : h1[lay.tx]! * h1[lay.ty]! = v2
    have e2 : ∀ j, (h1.set! lay.tw v2)[j]! = if j = lay.tw then v2 else h1[j]! := fun j =>
      gs h1 j v2 (by omega)
    have n1 : lay.hab ≠ lay.tw := by omega
    have n2 : lay.fa ≠ lay.tw := by omega
    have n3 : lay.fb ≠ lay.tw := by omega
    have n4 : lay.fa ≠ lay.hab := by omega
    have n5 : lay.fb ≠ lay.hab := by omega
    have n6 : lay.tx ≠ lay.tw := by omega
    have n7 : lay.ty ≠ lay.tw := by omega
    refine ⟨by rw [Array.size_set!, hs1], ?_, ?_⟩
    · rw [e2, e2, e2, if_neg n1, if_neg n2, if_neg n3, e1, e1, e1, if_pos rfl, if_neg n4, if_neg n5, hv1]
    · rw [e2, e2, e2, if_pos rfl, if_neg n6, if_neg n7, hv2]
  first
  | exact key _ hs
  | (unfold _root_.Flock.Firewall.Frame.Draws.ofBytes; dsimp only; exact key _ hs)

/-- **(c.2) holds in the masked scope**: there every table commits with `hm96-sha512` (`ZkTables.tables_of_setup`), rep
stream `k`'s table is `tabOfLevel0` at table `k / 2`'s statement (`Count.tabOfLevel0_sched`), whose `--zk` schedule is
the frame's (`Drive.start`), and the request's draws are `Draws.ofBytes` at that layout (`drawsOfBytes_pads`). -/
theorem finRepPads (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) :
    FinRepPads L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  intro ν t κ m₀ ρ₀ ops _ s0 spec hsx k hk tb hi q f hR
  obtain ⟨-, -, hT, hall⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨hr, hq, -⟩ := hR
  unfold realOf at hr
  rw [hsx] at hr
  try dsimp only at hr
  obtain ⟨⟨g, d, hi'⟩, -, hr⟩ := bok hr
  obtain ⟨l, -, hr⟩ := bok hr
  obtain ⟨tb', htb, hr⟩ := bok hr
  have htb' : tb' = tb := by cases hr; rfl
  rw [htb'] at htb
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htb
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, hst, hq⟩ := bok hq
  have hqd : q.draws = _root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) k) := by
    cases hq
    rfl
  unfold _root_.Flock.Firewall.Drive.start at hst
  obtain ⟨⟨sch, tPad⟩, hsch', hst⟩ := bok hst
  obtain ⟨ms', -, hst⟩ := bok hst
  have hlay : lay = _root_.Flock.Zk.PadLayout.new (Masked.stsOf x)[k / 2]!.m (Masked.stsOf x)[k / 2]!.c.kLog sch.k0 := by
    cases hst; rfl
  have hsc2 : (Except.ok (sch, tPad) : V _) = .ok (tb.sch, tb.tPad) := hsch'.symm.trans hsch
  have hss : sch = tb.sch := (Prod.mk.inj (Except.ok.inj hsc2)).1
  have hP : (_root_.Flock.Firewall.Finish.padsOf q.draws).h =
      (_root_.Flock.Firewall.Frame.Draws.ofBytes (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
        (L₀.draws (ν, t) k)).h := by
    show q.draws.h = _
    rw [hqd, hlay, hss, hts]
    try rfl
  rw [hP]
  refine ⟨?_, drawsOfBytes_pads _ _⟩
  rw [hts]
  show (Masked.stsOf x)[k / 2]!.merkle = _
  rw [getElem!_pos (Masked.stsOf x) (k / 2) hj]
  exact (hall (k / 2) hj).2.2

/-- **The session's rep at the opened coins is the firewall's**: at a `Finish` `FinishIs` reads, where the firewall's rep
on rep stream `k`'s table and request (`RepAt`), answered by the request's coins from round `0`, returns a proof and
transcript at its last coin (`proofFile_ok`), `repFwC` on the same inputs answered by the opened coins (`asksCut`), from a
history at the rep's first round, returns the same. -/
def FinRepCut : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ k < 2 * spec.tables.size, ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
    ∀ ic' p tr, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' →
      ((_root_.Flock.Firewall.Finish.repFw (_root_.Flock.Firewall.Finish.asksOf q.coins)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
          (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0 (_root_.Flock.Firewall.Finish.claimsOf q.xs)
          (_root_.Flock.Firewall.Finish.padsOf q.draws) ic' q.vals tb.extra tb.lanePad
          (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).run 0 =
        .ok ((p, tr), q.coins.size) →
    ∀ l : Array Msg, l.size = Masked.repBase x k →
      (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr)

/-- **(c.3) The firewall's rep at the views**: at a `Finish` `FinishIs` reads, at the schedules and level 0's
commitments (a) gives, rep stream `k`'s extra claims and pads commitment from the views, and a history at its first
round, where the session's rep on the loop's table and request (`RepAt`), answered by the opened coins, returns a proof
`p`, rep stream `k`'s run from the views returns a proof whose file at the session's shape is `p`'s at the table's. -/
def FinRepFwView : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs →
    ∀ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
    ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
    ∀ ic' p tr, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' →
      (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr) →
      ∃ p' tr' l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p', tr'), l') ∧
        FlockSoundness.Discharge.ZkView.encZk
          (FlockComplete.shapeZk (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 extras.size) p' =
        _root_.Flock.Firewall.Finish.encZk
          (_root_.Flock.Firewall.Finish.shapeZk tb.st tb.sch (_root_.Flock.Firewall.Finish.claimsOf q.xs).size) p

/-- **`FinRepCut` and (c.3) give (c).** Rep stream `k`'s file is `proofFile`'s on the loop's table and request
(`proofsOf_at`), the firewall's rep answered by the request's coins and its proof encoded (`proofFile_ok`); answered by
the opened coins from the rep's first round instead, it returns the same proof (`FinRepCut`), and the rep from the views
returns a proof of the same file (c.3). -/
theorem finRepProofs_of_parts (h : FinRepCut L₀ src real wk x w ∧ FinRepFwView vs L₀ src real wk x w) :
    FinRepProofs vs L₀ src real wk x w := by
  obtain ⟨hCut, hFV⟩ := h
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms files hp k hk extras hxe ic l hic hl
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  obtain ⟨tb, hi, q, hR⟩ := proofsOf_at hp (by rw [hrs]; exact hk)
  obtain ⟨ic', p, tr, hic', hrun, hfe⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hR.2.2
  have hcut := hCut ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q _ hR ic' p tr hic' hrun l hl
  obtain ⟨p', tr', l', hv, he⟩ := hFV ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl
    tb hi q _ hR ic' p tr hic' hcut
  exact ⟨p', tr', l', hv, he.trans hfe.symm⟩

end Loop

end Claims.ZKServer
