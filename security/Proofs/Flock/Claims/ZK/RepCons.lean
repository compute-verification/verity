module

public import Proofs.Flock.Claims.ZK.RepRet
public import Proofs.Flock.Claims.ZK.L0PinFrame
public import Proofs.Flock.Claims.Complete.Json.Goals
public import Proofs.Flock.Claims.ZK.L0PinSwitch
public import Proofs.Flock.Claims.ZK.FinishZc
public import Proofs.Flock.Claims.ZK.PrefixLive
public import Proofs.Flock.Claims.ZK.FinishTau
public import Proofs.Flock.Claims.ZK.FinishRank
public import Proofs.Flock.Claims.ZK.FinishSwPts
public import Proofs.Flock.Claims.ZK.RepReplay

/-!
# The ZK server's `repCons` returns where the cut rep does

`flock-firewall finish` runs a rep's PIOP and opening (`Finish.repCons`) in `StateT Nat V`, round `k` answered by the
rep's coins `C[k]` (`asksOf C`). Here it returns on any values, pads, level-0 commitment and extra claims, wherever the
coins pass the checks the rep samples them for, at the round count `consCounts` gives:

* **The opening's asks** (`Count.repCons_ot`): over `OpenM`, every round asking for its coins, `repCons` asks for the
  algebraic phase's counts (`replayCoins` without its zeros), then the ring switch's and Ligerito's (`openList`).
* **The cut's `repCons` returns** (`CutFree.repCons_cut`): answered by `asksCut` with room for its sends, at a `--zk`
  schedule of `m ∈ [25, 35]` and a fold defined at every point of the lincheck's shape, wherever its pair's coins are
  independent, with the masked `final_c`, the verifier's replay and the lane phase's constraints it batches.
* **The sends, counted** (`countS`, `countRel`): a cut run that returns sent rounds whose headers are the counts
  `repCons` asks for over `OpenM`, wherever the coins at those rounds have those sizes (`SizedAt`).
* **The sends, ported** (`portS`, `portRel`): where those sizes hold, the port at `asksOf C` returns what the cut
  returns, at round `|sent|`.
* **`repCons_asksOf`**: where `consCounts` is a prefix of `C`'s sizes, `repCons` at `asksOf C` from round `0` returns at
  round `|consCounts|`; **`repCons_asksOf_size`**: where they are `C`'s sizes, at round `C.size`.

No hypothesis reads a value: they are on the coins, the sizes and the schedule.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-! ## The opening's asks -/

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open _root_.Flock.Firewall.Finish (sampleP sampleVecP zerocheckFw lincheckFw repPiop)

variable {I : Got → Prop}

/-- The counts a rep's `repCons` asks for over `OpenM`, at `e` extra claims: the algebraic phase's without its zeros, then
the opening's. -/
def consCounts (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (e : ℕ) : List ℕ :=
  (_root_.Flock.Firewall.Session.replayCoins st.m st.fold.kLog).toList.filter (· != 0) ++ openList (2 + e) sch

theorem zerocheckFw_ot (coins : Got) (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) (hm : m < 2 ^ 64) :
    OT (Past coins) (zcCounts m) (fun _ => True) (zerocheckFw (stepAsks coins true) m digest cap root lay mk) := by
  unfold zerocheckFw
  apply ot_eq
  · apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_sampleVecP coins 6 (by decide)); intro _ _
    apply ot_bind (ot_sampleVecP coins (m - 13) (by omega)); intro _ _
    try dsimp only
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_sampleP coins); intro _ _
    apply ot_forIn_bind
    · intro i _ u
      try dsimp only
      apply ot_bind (ot_liftTM _); intro _ _
      apply ot_bind (ot_liftTM _); intro _ _
      apply ot_bind (ot_sampleP coins); intro _ _
      exact ot_pure ⟨_, rfl⟩
    · intro r
      try dsimp only
      apply ot_bind (ot_liftTM _); intro _ _
      apply ot_bind (ot_liftTM _); intro _ _
      exact ot_pure trivial
  · simp [zcCounts, sq]

theorem lincheckFw_ot (coins : Got) (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (zc : _root_.Flock.ZcOut) :
    OT (Past coins) (lcCounts c.kLog) (fun _ => True) (lincheckFw (stepAsks coins true) c lay mk zc) := by
  unfold lincheckFw
  apply ot_eq
  · apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_sampleP coins); intro _ _
    apply ot_bind (ot_sampleP coins); intro _ _
    try dsimp only
    split
    · apply ot_bind (ot_pure (Q := fun _ => True) trivial); intro _ _
      try dsimp only
      apply ot_forIn_bind
      · intro i _ u
        try dsimp only
        apply ot_bind (ot_liftTM _); intro _ _
        apply ot_bind (ot_liftTM _); intro _ _
        apply ot_bind (ot_sampleP coins); intro _ _
        exact ot_pure ⟨_, rfl⟩
      · intro r
        try dsimp only
        apply ot_bind (ot_liftTM _); intro _ _
        apply ot_bind (ot_sampleP coins); intro _ _
        exact ot_pure trivial
    · exact ot_throw_bind _ _
  · simp [lcCounts]

/-- **A rep's PIOP's asks**: the zerocheck's, then the lincheck's, leaving the extras and two claims more. -/
theorem repPiop_ot (coins : Got) (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hm : st.m < 2 ^ 64) :
    OT (Past coins) (zcCounts st.m ++ lcCounts st.fold.kLog) (fun r => r.2.2.size = 2 + extras.size)
      (repPiop (stepAsks coins true) st lay cap root extras mk) := by
  unfold repPiop
  apply ot_eq
  · apply ot_bind (zerocheckFw_ot coins st.m st.digest cap root lay mk hm); intro a _
    obtain ⟨zp, zc⟩ := a
    dsimp only
    apply ot_bind (lincheckFw_ot coins st.fold lay mk zc); intro b _
    obtain ⟨lp, lc⟩ := b
    dsimp only
    refine ot_pure ?_
    first
      | simp only [Array.size_append, FlockComplete.abcClaims_size]
      | simp [FlockComplete.abcClaims_size]
  · simp

theorem ot_get : OT I [] (fun _ => True) (get : TMW OpenM Flock.Transcript) := by
  intro t got rest hI
  exact ⟨trivial, hI, rfl⟩

theorem ot_liftV {α : Type} (v : V α) :
    OT I [] (fun _ => True) (StateT.lift (monadLift v : OpenM α) : TMW OpenM α) := by
  intro t got rest hI
  cases v with
  | error e =>
    have h : ((StateT.lift (monadLift (Except.error e : V α) : OpenM α) : TMW OpenM α).run t).run got =
        .fail e := rfl
    rw [h]
    trivial
  | ok a =>
    have h : ((StateT.lift (monadLift (Except.ok a : V α) : OpenM α) : TMW OpenM α).run t).run got =
        .done ((a, t), got) := rfl
    rw [h]
    exact ⟨trivial, hI, rfl⟩

/-- **A rep's `repCons` asks for `consCounts`** over `OpenM`, past no coins: the PIOP's counts, then the opening's at the
extras and two claims more. -/
theorem repCons_ot (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : _root_.Flock.Firewall.Finish.Commit) (extras : Array _root_.Flock.Claim)
    (pads : _root_.Flock.Firewall.Finish.PadsZk) (ic : _root_.Flock.Firewall.Finish.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (hm : st.m < 2 ^ 64)
    (hes : extras.size + 2 < 2 ^ 64) (hk : 1 ≤ sch.k0) (hn : sch.n < 2 ^ 62)
    (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 63) :
    OT (Past #[]) (consCounts st sch extras.size) (fun _ => True)
      (_root_.Flock.Firewall.Finish.repCons (stepAsks #[] true)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic
        vals extra lanePad salt) := by
  unfold _root_.Flock.Firewall.Finish.repCons
  apply ot_eq
  · try dsimp only
    apply ot_bind (repPiop_ot #[] st _ _ _ extras _ hm); intro r hr
    obtain ⟨zp, lp, claims⟩ := r
    have hc : claims.size = 2 + extras.size := hr
    have hO := repOpening_ot #[] st sch (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) tPad rep g cm0 claims
      pads.h extra lanePad salt (by rw [hc]; omega) hk hn hq
    rw [hc] at hO
    try dsimp only
    apply ot_bind hO; intro r _
    obtain ⟨ss, lg, zl, rec⟩ := r
    try dsimp only
    apply ot_bind ot_get; intro tr' _
    apply ot_bind (ot_liftV _); intro r _
    obtain ⟨rp, _⟩ := r
    try dsimp only
    apply ot_bind (ot_liftV _); intro lgc _
    exact ot_pure trivial
  · unfold consCounts openList
    rw [← piop_counts]
    simp only [List.append_nil, List.nil_append, List.append_assoc]

/-- A step program bound to `f` that asks for `ns` asks, before `f`, for a prefix of them. -/
theorem sp_bind_inv {α β : Type} {R : β → List ℕ → Prop} {f : α → Step β} :
    ∀ (s : Step α) (ns : List ℕ), SP R (s.bind f) ns → SP (fun a rest => SP R (f a) rest) s ns
  | .done _, _, h => h
  | .fail _, _, _ => trivial
  | .ask _ _ k, _, h => by
    obtain ⟨ns', e, h⟩ := h
    exact ⟨ns', e, fun cs => sp_bind_inv (k cs) ns' (h cs)⟩

/-- **A live round's ask, read back**: past the coins it was given, a round that asks for `ns` asks first for its
header's count, and goes on at any coins of that count. -/
theorem sp_round_inv {Q : (ℕ → F128) × Got → List ℕ → Prop} (coins got : Got) (c : ByteArray) (ns : List ℕ)
    (h : coins.size ≤ got.size) (hsp : SP Q (((stepAsks coins true).round c).run got) ns) :
    ∃ ns', ns = (_root_.Flock.u64OfLE c (c.size - 8)).toNat :: ns' ∧
      ∀ cs : Array F128, cs.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat →
        Q ((fun i => cs[i]!), got.push cs) ns' := by
  have hn : coins[got.size]? = none := Array.getElem?_eq_none h
  simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind, hn,
    StateT.lift, SP] at hsp
  obtain ⟨ns', e, hk⟩ := hsp
  refine ⟨ns', e, fun cs hc => ?_⟩
  have h1 := hk cs
  simp only [hc, bne_self_eq_false, Bool.false_eq_true, if_false] at h1
  exact h1

end Claims.ZKServer.Count

/-! ## The cut's `repCons` returns -/

namespace FlockComplete.CutFree

open Flock Transcript Replay

/-- Coins as the bytes a reply carries them in: each element's `le64(lo) ‖ le64(hi)`. -/
def encC (cs : Array F128) : Claims.Msg := cs.toList.flatMap fun a => a.toBytes.data.toList

theorem encL_length : ∀ xs : List F128, (xs.flatMap fun a => a.toBytes.data.toList).length = 16 * xs.length
  | [] => by simp
  | x :: xs => by
    rw [List.flatMap_cons, List.length_append, encL_length xs, JsonRT.toBytes_length', List.length_cons]
    omega

theorem encC_length (cs : Array F128) : (encC cs).length = 16 * cs.size := by
  rw [encC, encL_length, Array.length_toList]

theorem encC_split (cs : Array F128) (i : ℕ) (hi : i < cs.size) :
    ∃ pre post : List UInt8, encC cs = pre ++ cs[i].toBytes.data.toList ++ post ∧ pre.length = 16 * i := by
  refine ⟨(cs.toList.take i).flatMap fun a => a.toBytes.data.toList,
    (cs.toList.drop (i + 1)).flatMap fun a => a.toBytes.data.toList, ?_, ?_⟩
  · have hi' : i < cs.toList.length := by simpa using hi
    conv_lhs => rw [encC, ← List.take_append_drop i cs.toList, List.drop_eq_getElem_cons hi']
    simp only [List.flatMap_append, List.flatMap_cons, Array.getElem_toList, List.append_assoc]
  · rw [encL_length, List.length_take, Array.length_toList]
    omega

/-- **Coins read back from their bytes**: element `i` is `cs[i]`, and `0` past `cs`. -/
theorem coinOf_encC (cs : Array F128) : coinOf (encC cs) = fun i => cs[i]! := by
  funext i
  show coinOf (encC cs) i = cs[i]!
  by_cases hi : i < cs.size
  · rw [getElem!_pos cs i hi]
    obtain ⟨pre, post, e, hl⟩ := encC_split cs i hi
    have hb : Claims.ofMsg (encC cs) = ⟨pre.toArray⟩ ++ le64 cs[i].lo ++ (le64 cs[i].hi ++ ⟨post.toArray⟩) := by
      apply ByteArray.ext
      apply Array.toList_inj.1
      simp only [Claims.ofMsg, e, FlockSoundness.Discharge.ZkView.toBytes_split, FlockSoundness.Refine.data_append,
        List.append_assoc]
    have hp : (⟨pre.toArray⟩ : ByteArray).size = 16 * i := by simp [ByteArray.size, hl]
    have hp8 : (⟨pre.toArray⟩ ++ le64 cs[i].lo : ByteArray).size = 16 * i + 8 := by
      rw [ByteArray.size_append, hp, FlockSoundness.Discharge.ZkView.le64_size]
    unfold coinOf
    rw [hb, ← hp8, ← hp, FlockSoundness.Discharge.ZkView.u64OfLE_mid, ← ByteArray.append_assoc,
      FlockSoundness.Discharge.ZkView.u64OfLE_mid]
  · rw [getElem!_neg cs i hi]
    have hs : (Claims.ofMsg (encC cs)).size = 16 * cs.size := by
      simp [Claims.ofMsg, ByteArray.size, encC_length]
    unfold coinOf
    rw [JsonRT.u64OfLE_past _ _ (by omega), JsonRT.u64OfLE_past _ _ (by omega)]
    first | rfl | decide

theorem ok_name {α β : Type} {x : TMW PS (α × β)} {t : Transcript} {a : α} {b : β} {t' : Transcript}
    {l l' : Array Claims.Msg} (h : (x.run t).go l = (.ok ((a, b), t'), l')) :
    ∃ a', (x.run t).go l = (.ok ((a', b), t'), l') ∧ a' = a :=
  ⟨a, h, rfl⟩

/-- **The cut's level-0-masked Ligerito opening returns where its pair's coins are independent**: answered by `asksCut`
with room for its `k₀ + 7 + 11R` sends at a `--zk` schedule of `m ∈ [25, 35]`, at any table, basis, pads, extra lanes
and salts, with a record of the layout's shape. `ligeritoPZk` returns at the replies (`ligeritoPZk_ret`), so at the cut's
coins, where the coupling (`ligeritoPZk_agree`) throws only at send `n + 1`, past its send count. -/
theorem ligeritoPZk_cut_ret (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg}
    (hr : ∀ k, coinOf (resp k) = coin k) {m : ℕ} (h25 : 25 ≤ m) (h35 : m ≤ 35) {sch : Schedule} {tp : ℕ}
    (hz : Zk.schedule m = .ok (sch, tp)) (ms : MerkleScheme) (T : F128) (cm0 : Commit) (tb b : Array F128)
    (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (t : Transcript) (l : Array Claims.Msg) (hN : l.size + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) ≤ n)
    (hρ : Zk.rhoIndependent ((⟨coin (l.size + 2 + sch.k0) 0, coin (l.size + 2 + sch.k0) 1⟩ : F256),
      (⟨coin (l.size + 3 + sch.k0) 0, coin (l.size + 3 + sch.k0) 1⟩ : F256)) = true) :
    ∃ lg zl rec t' l', ((ligeritoPZk (asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 tb b lay hv rep tPad
        extra lanePad salt (fun _ _ => 0)).run t).go l = (.ok ((lg, zl, rec), t'), l') ∧
      rec.ood.size = 1 ∧ rec.challenges.size = sch.k0 ∧ sch.k0 ≤ lg.messages.size := by
  obtain ⟨hk0, hR⟩ : 1 ≤ sch.k0 ∧ 2 ≤ sch.levels.size := zkSchedule_shape hz
  have hn := L0.zkSchedule_n hz
  obtain ⟨hn', hq'⟩ : sch.n + 1 ≤ 65536 ∧ ∀ j, (sch.levels.getD j default).q + 1 ≤ 65536 :=
    JsonRT.zkSchedule_small h25 h35 hz
  obtain ⟨⟨⟨P, zl, rc⟩, t'⟩, l', hret⟩ := L0.ligeritoPZk_ret resp Sha512.hash ms sch T cm0 tb b lay hv rep tPad
    extra lanePad salt (fun _ _ => 0) hn hk0 t l (by simp only [coinKOf, hr]; exact hρ)
  have hle := sends_le (ligeritoPZk_sends resp Sha512.hash ms sch T cm0 tb b lay hv rep tPad extra lanePad salt _) hret
  have hLF := L0.ligeritoPZk_lanes resp _ ms sch T cm0 tb b lay hv rep tPad extra lanePad salt _ hn hk0 hret
  obtain ⟨-, hC⟩ : PrefixAgree.Agree n _ _ := _root_.Flock.With.Sim.run _
    (PrefixAgree.ligeritoPZk_agree resp coin (n := n) (fun k _ i _ => congrFun (hr k) i) Sha512.hash ms sch T cm0
      tb b lay hv rep tPad extra lanePad salt (fun _ _ => 0) hn' hq') t
  have hpzk : ((ligeritoPZk (asksCut coin n) (Hashes.ofSha512 Sha512.hash) ms sch T cm0 tb b lay hv rep tPad
      extra lanePad salt fun _ _ => 0).run t).go l = (.ok ((P, zl, rc), t'), l') := by
    have hc := hC l (by omega)
    rw [hret] at hc
    rcases hc with ⟨he, -⟩ | ⟨-, hs, bs, hbs⟩
    · exact he.symm
    · have := congrArg Array.size hbs
      simp only [Array.size_append] at this
      omega
  refine ⟨P, zl, rc, t', l', hpzk, ?_, ?_, ?_⟩
  · rw [hLF.record]
    try rfl
  · rw [hLF.record]
    simp only [ZkTables.Level0.laneCh, List.size_toArray, List.length_map, List.length_range]
  · have := hLF.msgs_size
    omega

set_option maxHeartbeats 4000000 in
/-- **The cut's `repCons` returns on its coins alone**: answered by `asksCut` with room for its sends up to the inner
proof, at a `--zk` schedule of `m ∈ [25, 35]` and a fold defined at every point of the lincheck's shape, on any values,
pads, table, level-0 commitment and extra claims, `Finish.repCons` returns wherever its pair's coins are independent. Its
`final_c` is the masked value at `fc`, the verifier's replay of its proof returns, and its constraints are the replay's,
the ring switch's and the lane phase's. -/
theorem repCons_cut (coin : ℕ → ℕ → F128) (n : ℕ) {resp : ℕ → Claims.Msg} (hr : ∀ k, coinOf (resp k) = coin k)
    {st : Setup} (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hkm : st.fold.kLog ≤ st.m) (hkk : st.kLog = st.fold.kLog)
    (hmk : st.merkle = MerkleScheme.hm96Sha512)
    (hfold : ∀ (a b : F128) (lw ρ ts : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ts.size = st.fold.kLog - 6 → ∃ w, st.fold.folded a b lw ρ ts = .ok w ∧ w.size = 64)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ∃ v, st.fold.fold a lw ρ = .ok v)
    {sch : Schedule} {tp : ℕ} (hz : Zk.schedule st.m = .ok (sch, tp)) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : _root_.Flock.Firewall.Finish.Commit) (extras : Array Claim)
    (pads : _root_.Flock.Firewall.Finish.PadsZk) (ic : _root_.Flock.Firewall.Finish.InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (l : Array Claims.Msg)
    (hN : CutEq.innerAt l.size st sch extras ≤ n)
    (hρ : Zk.rhoIndependent
      ((⟨coin (CutEq.ligAt l.size st extras + 2 + sch.k0) 0, coin (CutEq.ligAt l.size st extras + 2 + sch.k0) 1⟩ :
          F256),
        (⟨coin (CutEq.ligAt l.size st extras + 3 + sch.k0) 0, coin (CutEq.ligAt l.size st extras + 3 + sch.k0) 1⟩ :
          F256)) = true) :
    ∃ r tr l', ((_root_.Flock.Firewall.Finish.repCons (LigLock.Fed.cutAsks coin n)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic
        vals extra lanePad salt).run (Transcript.init #[] dom)).go l = (.ok (r, tr), l') ∧
      r.1.zerocheck.finalC = vals[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]! +
        pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]! ∧
      ∃ (rp : Zk.Replayed) (t' : Transcript) (lgc : Array (String × Zk.Lin)) (rec : Zk.L0Rec),
        (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0) r.1 ic.root).run (Transcript.init tr.rounds dom) =
          .ok (rp, t') ∧
        Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec r.1.ligerito r.2.1 = .ok lgc ∧
        r.2.2 = rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! r.1.ringSwitches[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! r.1.ringSwitches[1]!))] ++ lgc := by
  have hm : 14 ≤ st.m := by omega
  have hN' := hN
  simp only [CutEq.innerAt, CutEq.ligAt] at hN'
  obtain ⟨zc, t₁, l₁, h₁⟩ := Claims.ZKServer.zerocheckCut_proof coin n hm st.digest
    (_root_.Flock.Firewall.Finish.flat cm0.cap) ic.root (Zk.PadLayout.new st.m st.kLog sch.k0)
    (fun i => vals[i]! + pads.h[i]!) (Transcript.init #[] dom) l (by omega)
  obtain ⟨hz₁, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  obtain ⟨zp, h₁, hzp⟩ := ok_name h₁
  have hρs : (zc.rho.extract 0 (st.fold.kLog - 6)).size = st.fold.kLog - 6 := by
    rw [Array.size_extract, hz₁.2.1, Replay.pushes_size]
    omega
  obtain ⟨lc, t₂, l₂, h₂⟩ := Claims.ZKServer.lincheckCut_proof coin n st.fold st.m
    (Zk.PadLayout.new st.m st.kLog sch.k0) (fun i => vals[i]! + pads.h[i]!) zc t₁ l₁ (by omega)
    (hf _ _ _ (FlockSoundness.Refine.skipWeights_size _) hρs)
  obtain ⟨-, hl₂⟩ := lincheckCut_at coin n st.fold st.m _ _ zc h₂
  obtain ⟨lp, h₂, -⟩ := ok_name h₂
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    rw [Array.size_append, abcClaims_size]
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, -, -⟩ := Claims.ZKServer.Switch.ringSwitchCut_bodies coin n
    (fun k => _root_.Flock.Firewall.Finish.switchVec g (abcClaims st.m st.kLog zc lc ++ extras)[k]!)
    (abcClaims st.m st.kLog zc lc ++ extras) (by rw [hsz]; omega) t₂ l₂ (by rw [hsz]; omega)
  obtain ⟨-, hl₃, -⟩ := ringSwitchCut_sws coin n _ _ (by rw [hsz]; omega) h₃
  rw [hsz] at hl₃
  have hL₃ : l₃.size = CutEq.ligAt l.size st extras := by
    first
      | (simp only [CutEq.ligAt]; omega)
      | (unfold CutEq.ligAt; omega)
      | omega
  obtain ⟨lg, zl, rec, t₄, l₄, h₄, hood, hch, hmsg⟩ := ligeritoPZk_cut_ret coin n hr h25 h35 hz
    MerkleScheme.hm96Sha512 T cm0.toC g (_root_.Flock.Firewall.Finish.basisVals sws (st.m - 7))
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt t₃ l₃ (by omega)
    (by rw [hL₃]; exact hρ)
  have hR : Replay.Prefix t₂.rounds t₄.rounds :=
    (RankAt.prefix_of_grows coin 0 (RankAt.ringSwitchCut_grows coin n 0 _ _) h₃).trans
      (RankAt.prefix_of_grows coin 0 (RankAt.ligeritoPZk_grows coin n 0 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) h₄)
  obtain ⟨⟨rp, tt⟩, hq⟩ := replay_cut_ok coin n hr hm hkk (lay := Zk.PadLayout.new st.m st.kLog sch.k0)
    (Nat.le_refl _) (by show st.fold.kLog - 6 ≤ st.kLog - 6; rw [hkk]) (fun i => vals[i]! + pads.h[i]!)
    { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
      ligerito := lg } ic.root dom h₁ h₂ hR
    (by
      refine hfold _ _ _ _ _ (FlockSoundness.Refine.skipWeights_size _) ?_ (Replay.pushes_size _ _)
      rw [Array.size_extract, Replay.pushes_size]
      omega)
  obtain ⟨lgc, hlgc⟩ := ligeritoConstraints_ok (lay := Zk.PadLayout.new st.m st.kLog sch.k0) (msgs := zl) hood hch hmsg
  have hP : ((_root_.Flock.Firewall.Finish.repPiop (LigLock.Fed.cutAsks coin n) st
      (Zk.PadLayout.new st.m st.kLog sch.k0) (_root_.Flock.Firewall.Finish.flat cm0.cap) ic.root extras
      (fun i => vals[i]! + pads.h[i]!)).run (Transcript.init #[] dom)).go l =
        (.ok ((zp, lp, abcClaims st.m st.kLog zc lc ++ extras), t₂), l₂) := by
    unfold _root_.Flock.Firewall.Finish.repPiop
    rw [_root_.Flock.Firewall.Finish.zerocheckFw_eq, LigLock.Fed.cutAsks_toC, bind_go h₁]
    try dsimp only
    rw [_root_.Flock.Firewall.Finish.lincheckFw_eq _ _ st.m, LigLock.Fed.cutAsks_toC, bind_go h₂]
    try dsimp only
    try exact pure_go _ _ _
  have hO : ((_root_.Flock.Firewall.Finish.repOpening (LigLock.Fed.cutAsks coin n)
      (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch
      (Zk.PadLayout.new st.m st.kLog sch.k0) tPad rep g cm0 (abcClaims st.m st.kLog zc lc ++ extras) pads.h extra
      lanePad salt).run t₂).go l₂ = (.ok ((ss, lg, zl, rec), t₄), l₄) := by
    unfold _root_.Flock.Firewall.Finish.repOpening
    rw [_root_.Flock.Firewall.Finish.ringSwitchP_eq, LigLock.Fed.cutAsks_toC, bind_go h₃, hmk]
    try dsimp only
    rw [_root_.Flock.Firewall.Finish.ligeritoPZk_fw, LigLock.Fed.cutAsks_toC, bind_go h₄]
    try dsimp only
    try exact pure_go _ _ _
  unfold _root_.Flock.Firewall.Finish.repCons
  try dsimp only
  rw [bind_go hP]
  try dsimp only
  rw [bind_go hO]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.get_run_go _ _)]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.liftV_run_go hq _ _)]
  try dsimp only
  rw [bind_go (_root_.Flock.Firewall.Finish.liftV_run_go hlgc _ _)]
  try dsimp only
  subst hzp
  exact ⟨_, _, _, pure_go _ _ _, rfl, rp, tt, lgc, rec, hq, hlgc, rfl⟩

end FlockComplete.CutFree

/-! ## The ZK server's `repCons` at `asksOf` -/

namespace Flock.Firewall.Finish

open Flock With Transcript
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open Claims.ZKServer.Count (SP)

set_option maxHeartbeats 4000000 in
/-- `repCons` sends no link message: it reads only its sender's rounds. -/
theorem repCons_link {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (io io' : Asks M)
    (h : io.round = io'.round) (commit : Level → Array F128 → (ℕ → ByteArray) → M Commit) (st : Setup)
    (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) :
    repCons io commit st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt =
      repCons io' commit st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt := by
  obtain ⟨l1, r1⟩ := io
  obtain ⟨l2, r2⟩ := io'
  change r1 = r2 at h
  subst h
  rfl

/-- The cut's rounds with no link: a link message throws, as the port's does. -/
def cutNoLink (coin : ℕ → ℕ → F128) (n : ℕ) : Asks FlockComplete.PS :=
  ⟨fun _ _ => throw "finish: a link message inside a rep", (FlockComplete.asksCut coin n).round⟩

/-- Round `k`'s coin `i`, from the rep's coins `C`. -/
def coinC (C : Array (Array F128)) (k i : ℕ) : F128 := (C[k]!)[i]!

theorem getBang_of_some {C : Array (Array F128)} {k : ℕ} {cs : Array F128} (h : C[k]? = some cs) : C[k]! = cs := by
  obtain ⟨hk, e⟩ := Array.getElem?_eq_some_iff.1 h
  rw [getElem!_pos C k hk, e]

/-- A sent round's coin count, from its framed header. -/
def roundHdr (b : Claims.Msg) : ℕ := (u64OfLE (Claims.ofMsg b) ((Claims.ofMsg b).size - 8)).toNat

theorem roundHdr_toMsg (c : ByteArray) : roundHdr (Claims.toMsg c) = (u64OfLE c (c.size - 8)).toNat := by
  simp only [roundHdr, Claims.ofMsg_toMsg]

/-- **`C`'s rounds from `base` have the sizes `ns`**. -/
def SizedAt (C : Array (Array F128)) (base : ℕ) (ns : List ℕ) : Prop :=
  ∀ j < ns.length, ∃ cs, C[base + j]? = some cs ∧ ns[j]? = some cs.size

theorem sizedAt_shift {C : Array (Array F128)} {base : ℕ} {sfx : Array Claims.Msg} {ns : List ℕ}
    (h : SizedAt C base (sfx.toList.map roundHdr ++ ns)) : SizedAt C (base + sfx.size) ns := by
  intro j hj
  have e : (sfx.toList.map roundHdr).length = sfx.size := by rw [List.length_map, Array.length_toList]
  obtain ⟨cs, h1, h2⟩ := h (sfx.size + j) (by rw [List.length_append, e]; omega)
  refine ⟨cs, by rw [Nat.add_assoc]; exact h1, ?_⟩
  have h3 : (sfx.toList.map roundHdr).length ≤ sfx.size + j := by rw [e]; exact Nat.le_add_right _ _
  rw [List.getElem?_append_right h3, e, Nat.add_sub_cancel_left] at h2
  exact h2

theorem sizedAt_left {C : Array (Array F128)} {base : ℕ} {ns ms : List ℕ} (h : SizedAt C base (ns ++ ms)) :
    SizedAt C base ns := by
  intro j hj
  obtain ⟨cs, h1, h2⟩ := h j (by rw [List.length_append]; omega)
  exact ⟨cs, h1, by rw [List.getElem?_append_left hj] at h2; exact h2⟩

theorem sizedAt_all (C : Array (Array F128)) : SizedAt C 0 (C.toList.map Array.size) := by
  intro j hj
  have hj' : j < C.size := by simpa using hj
  refine ⟨C[j], by rw [Nat.zero_add]; exact Array.getElem?_eq_getElem hj', ?_⟩
  first
    | (rw [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hj']; rfl)
    | rw [List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hj']
    | simp [hj']

/-! ### The sends, counted -/

/-- **A cut run's sends are the asks**: where `y`, from `got`, asks for `ns` and ends in `Q`, and `C`'s rounds from the
log's have the sizes `ns`, a run of `x` from the log that returns sent rounds whose headers are a prefix of `ns`, and `y`
ends with the same value in `Q` at the counts after them. -/
def CountR (C : Array (Array F128)) {α : Type} (x : FlockComplete.PS α) (y : OpenM α) : Prop :=
  ∀ (l : Array Claims.Msg) (got : Array (Array F128)) (ns : List ℕ) (Q : α × Array (Array F128) → List ℕ → Prop)
    (a : α) (l' : Array Claims.Msg),
    SP Q (y.run got) ns → SizedAt C l.size ns → x.go l = (.ok a, l') →
      ∃ sfx : Array Claims.Msg, l' = l ++ sfx ∧ ∃ ns₂, ns = sfx.toList.map roundHdr ++ ns₂ ∧
        ∃ got' : Array (Array F128), Q (a, got') ns₂

def countS (C : Array (Array F128)) : Sim FlockComplete.PS OpenM where
  R x y := CountR C x y
  pure a := by
    intro l got ns Q a' l' hsp _ hx
    have h' : ((Except.ok a : Except String _), l) = (.ok a', l') := hx
    cases h'
    exact ⟨#[], by rw [Array.append_empty], ns, by simp, got, hsp⟩
  bind {α β x y f g} hxy hfg := by
    intro l got ns Q b l'' hsp hs hx
    obtain ⟨a, l₁, h1, h2⟩ := FlockComplete.JsonRT.go_bind_ok hx
    rw [StateT.run_bind] at hsp
    obtain ⟨sfx₁, rfl, ns₁, rfl, got₁, hsp₁⟩ := hxy l got _ (fun p rest => SP Q ((g p.1).run p.2) rest) a l₁
      (Claims.ZKServer.Count.sp_bind_inv _ _ hsp) hs h1
    obtain ⟨sfx₂, rfl, ns₂, rfl, got₂, hq⟩ := hfg a (l ++ sfx₁) got₁ ns₁ Q b l'' hsp₁
      (by rw [Array.size_append]; exact sizedAt_shift hs) h2
    refine ⟨sfx₁ ++ sfx₂, by rw [Array.append_assoc], ns₂, ?_, got₂, hq⟩
    rw [Array.toList_append, List.map_append, List.append_assoc]
  throw e := by
    intro l got ns Q a l' _ _ hx
    have h' : ((Except.error e : Except String _), l) = (.ok a, l') := hx
    cases h'
  lift v := by
    cases v with
    | error e =>
      intro l got ns Q a l' _ _ hx
      have h' : ((Except.error e : Except String _), l) = (.ok a, l') := hx
      cases h'
    | ok a0 =>
      intro l got ns Q a l' hsp _ hx
      have h' : ((Except.ok a0 : Except String _), l) = (.ok a, l') := hx
      cases h'
      exact ⟨#[], by rw [Array.append_empty], ns, by simp, got, hsp⟩

/-- **The cut's rounds are the opening's asks**, each answered by `C`'s round at the log's. -/
theorem countRel (C : Array (Array F128)) (N : ℕ) :
    AsksRel (countS C) (cutNoLink (coinC C) N).toC (stepAsks #[] true).toC where
  link _ _ := by
    intro l got ns Q x l' _ _ hx
    have h' : ((Except.error "finish: a link message inside a rep" : Except String (ℕ → F128)), l) = (.ok x, l') :=
      hx
    cases h'
  round c := by
    intro l got ns Q a l' hsp hs hx
    change ((FlockComplete.asksCut (coinC C) N).round c).go l = _ at hx
    rw [asksCut_round_fw] at hx
    by_cases hN : N ≤ l.size
    · rw [if_pos hN] at hx
      cases hx
    · rw [if_neg hN] at hx
      cases hx
      change SP Q (((stepAsks #[] true).round c).run got) ns at hsp
      obtain ⟨ns', rfl, hk⟩ := Claims.ZKServer.Count.sp_round_inv #[] got c ns (Nat.zero_le _) hsp
      obtain ⟨cs, h1, h2⟩ := hs 0 (by simp)
      have h1' : C[l.size]? = some cs := by simpa using h1
      have hcs : (u64OfLE c (c.size - 8)).toNat = cs.size := by
        first
          | exact Option.some.inj h2
          | simpa using h2
      refine ⟨#[Claims.toMsg c], Array.push_eq_append, ns', ?_, got.push cs, ?_⟩
      · first
          | simp [roundHdr_toMsg]
          | (show _ = [roundHdr (Claims.toMsg c)] ++ ns'; rw [roundHdr_toMsg]; rfl)
      · have ecs : coinC C l.size = fun i => cs[i]! := by
          funext i
          show (C[l.size]!)[i]! = cs[i]!
          rw [getBang_of_some h1']
        rw [ecs]
        exact hk cs hcs.symm

/-! ### The sends, ported -/

/-- **A cut run as the port's**: a run of `x` from the log `l` that returns sent rounds, and where `C`'s rounds from
`l`'s have their headers' sizes, `z` from round `l.size` returns the same at the log's size. -/
def PortR (C : Array (Array F128)) {α : Type} (x : FlockComplete.PS α) (z : StateT ℕ V α) : Prop :=
  ∀ (l : Array Claims.Msg) (a : α) (l' : Array Claims.Msg), x.go l = (.ok a, l') →
    ∃ sfx : Array Claims.Msg, l' = l ++ sfx ∧
      (SizedAt C l.size (sfx.toList.map roundHdr) → z.run l.size = .ok (a, l'.size))

def portS (C : Array (Array F128)) : Sim FlockComplete.PS (StateT ℕ V) where
  R x z := PortR C x z
  pure a := by
    intro l a' l' hx
    have h' : ((Except.ok a : Except String _), l) = (.ok a', l') := hx
    cases h'
    exact ⟨#[], by rw [Array.append_empty], fun _ => rfl⟩
  bind {α β x y f g} hxy hfg := by
    intro l b l'' hx
    obtain ⟨a, l₁, h1, h2⟩ := FlockComplete.JsonRT.go_bind_ok hx
    obtain ⟨sfx₁, rfl, H1⟩ := hxy l a l₁ h1
    obtain ⟨sfx₂, rfl, H2⟩ := hfg a (l ++ sfx₁) b l'' h2
    refine ⟨sfx₁ ++ sfx₂, by rw [Array.append_assoc], fun hs => ?_⟩
    rw [Array.toList_append, List.map_append] at hs
    have hs2 : SizedAt C (l ++ sfx₁).size (sfx₂.toList.map roundHdr) := by
      rw [Array.size_append]
      exact sizedAt_shift hs
    rw [StateT.run_bind, H1 (sizedAt_left hs)]
    exact H2 hs2
  throw e := by
    intro l a l' hx
    have h' : ((Except.error e : Except String _), l) = (.ok a, l') := hx
    cases h'
  lift v := by
    cases v with
    | error e =>
      intro l a l' hx
      have h' : ((Except.error e : Except String _), l) = (.ok a, l') := hx
      cases h'
    | ok a0 =>
      intro l a l' hx
      have h' : ((Except.ok a0 : Except String _), l) = (.ok a, l') := hx
      cases h'
      exact ⟨#[], by rw [Array.append_empty], fun _ => rfl⟩

/-- **The cut's rounds are the port's**: where the round at the log's has its header's size in `C`, `asksOf C` answers it
as the cut does, at the next round. -/
theorem portRel (C : Array (Array F128)) (N : ℕ) :
    AsksRel (portS C) (cutNoLink (coinC C) N).toC (asksOf C).toC where
  link _ _ := by
    intro l x l' hx
    have h' : ((Except.error "finish: a link message inside a rep" : Except String (ℕ → F128)), l) = (.ok x, l') :=
      hx
    cases h'
  round c := by
    intro l a l' hx
    change ((FlockComplete.asksCut (coinC C) N).round c).go l = _ at hx
    rw [asksCut_round_fw] at hx
    by_cases hN : N ≤ l.size
    · rw [if_pos hN] at hx
      cases hx
    · rw [if_neg hN] at hx
      cases hx
      refine ⟨#[Claims.toMsg c], Array.push_eq_append, fun hs => ?_⟩
      obtain ⟨cs, h1, h2⟩ := hs 0 (by simp)
      have h1' : C[l.size]? = some cs := by simpa using h1
      have hcs : (u64OfLE c (c.size - 8)).toNat = cs.size := by
        have h2' : roundHdr (Claims.toMsg c) = cs.size := by
          first
            | exact Option.some.inj h2
            | simpa using h2
        rwa [roundHdr_toMsg] at h2'
      have hn' : (cs.size != (u64OfLE c (c.size - 8)).toNat) = false := by
        rw [hcs]
        simp
      have ecs : coinC C l.size = fun i => cs[i]! := by
        funext i
        show (C[l.size]!)[i]! = cs[i]!
        rw [getBang_of_some h1']
      rw [ecs, Array.size_push]
      show ((asksOf C).round c).run l.size = _
      first
        | (simp only [asksOf, StateT.run_bind, StateT.run_get, StateT.run_set, StateT.run_pure, pure_bind, h1', hn',
            Bool.false_eq_true, ↓reduceIte]; done)
        | (simp only [asksOf, StateT.run_bind, StateT.run_get, StateT.run_set, StateT.run_pure, pure_bind, h1', hn',
            Bool.false_eq_true, ↓reduceIte]; rfl)
        | simp [asksOf, h1', hcs]
        | (simp [asksOf, h1', hcs]; rfl)

/-! ### `repCons` at `asksOf` -/

set_option maxHeartbeats 4000000 in
/-- **The ZK server's `repCons` returns on its coins alone**: at a `--zk` schedule of `m ∈ [25, 35]`, `hm96-sha512/v1`
and a fold defined at every point of the lincheck's shape, on any values, pads, table, level-0 commitment and extra
claims, `repCons` answered by `asksOf C` from round `0` returns at round `|consCounts|`, wherever `consCounts` is a prefix
of `C`'s sizes and the pair's coins at the opening's lane rounds are independent. Its `final_c` is the masked value at
`fc`, the verifier's replay of its proof returns, and its constraints are the replay's, the ring switch's and the lane
phase's. -/
theorem repCons_asksOf (C : Array (Array F128)) {st : Setup} (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35)
    (hkm : st.fold.kLog ≤ st.m) (hkk : st.kLog = st.fold.kLog) (hmk : st.merkle = MerkleScheme.hm96Sha512)
    (hfold : ∀ (a b : F128) (lw ρ ts : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ts.size = st.fold.kLog - 6 → ∃ w, st.fold.folded a b lw ρ ts = .ok w ∧ w.size = 64)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ∃ v, st.fold.fold a lw ρ = .ok v)
    {sch : Flock.Schedule} {tp : ℕ} (hz : Zk.schedule st.m = .ok (sch, tp)) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (hes : extras.size + 2 < 2 ^ 64)
    (hC : Claims.ZKServer.Count.consCounts st sch extras.size <+: C.toList.map Array.size)
    (hρ : Zk.rhoIndependent
      ((⟨(C[FlockComplete.CutEq.ligAt 0 st extras + 2 + sch.k0]!)[0]!,
          (C[FlockComplete.CutEq.ligAt 0 st extras + 2 + sch.k0]!)[1]!⟩ : F256),
        (⟨(C[FlockComplete.CutEq.ligAt 0 st extras + 3 + sch.k0]!)[0]!,
          (C[FlockComplete.CutEq.ligAt 0 st extras + 3 + sch.k0]!)[1]!⟩ : F256)) = true) :
    ∃ r tr, ((repCons (asksOf C) (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
        extra lanePad salt).run (Transcript.init #[] dom)).run 0 =
          .ok ((r, tr), (Claims.ZKServer.Count.consCounts st sch extras.size).length) ∧
      r.1.zerocheck.finalC = vals[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]! +
        pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]! ∧
      ∃ (rp : Zk.Replayed) (t' : Transcript) (lgc : Array (String × Zk.Lin)) (rec : Zk.L0Rec),
        (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0) r.1 ic.root).run (Transcript.init tr.rounds dom) =
          .ok (rp, t') ∧
        Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec r.1.ligerito r.2.1 = .ok lgc ∧
        r.2.2 = rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! r.1.ringSwitches[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! r.1.ringSwitches[1]!))] ++ lgc := by
  obtain ⟨-, hk0, hn62, hq63⟩ := Claims.ZKServer.Count.sched_facts hz
  have hm64 : st.m < 2 ^ 64 := by omega
  obtain ⟨rest, hrest⟩ := hC
  obtain ⟨r, tr, l', hcut, hout⟩ := FlockComplete.CutFree.repCons_cut (coinC C)
    (FlockComplete.CutEq.innerAt 0 st sch extras) (resp := fun k => FlockComplete.CutFree.encC C[k]!)
    (fun k => FlockComplete.CutFree.coinOf_encC _) h25 h35 hkm hkk hmk hfold hf hz tPad rep dom g cm0 extras pads ic
    vals extra lanePad salt #[] (Nat.le_refl _) hρ
  rw [repCons_link (FlockComplete.LigLock.Fed.cutAsks (coinC C) (FlockComplete.CutEq.innerAt 0 st sch extras))
    (cutNoLink (coinC C) (FlockComplete.CutEq.innerAt 0 st sch extras)) rfl] at hcut
  have hot := Claims.ZKServer.Count.repCons_ot st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt hm64
    hes hk0 hn62 hq63 (Transcript.init #[] dom) #[] rest (Nat.le_refl _)
  have hall : SizedAt C 0 (Claims.ZKServer.Count.consCounts st sch extras.size ++ rest) := by
    rw [hrest]
    exact sizedAt_all C
  obtain ⟨sfx, hl', ns₂, hns, got', -, -, hns₂⟩ := repCons_rel (countS C)
    (countRel C (FlockComplete.CutEq.innerAt 0 st sch extras)) st hmk sch tPad rep dom g cm0 extras pads ic vals extra
    lanePad salt (Transcript.init #[] dom) #[] #[] _ _ (r, tr) l' hot hall hcut
  subst hl'
  subst hns₂
  have hpre := (List.append_cancel_right hns).symm
  obtain ⟨sfx', e', H⟩ := repCons_rel (portS C) (portRel C (FlockComplete.CutEq.innerAt 0 st sch extras)) st hmk sch
    tPad rep dom g cm0 extras pads ic vals extra lanePad salt (Transcript.init #[] dom) #[] (r, tr) _ hcut
  simp only [Array.empty_append] at e'
  subst e'
  have hs : SizedAt C (#[] : Array Claims.Msg).size (sfx.toList.map roundHdr) := by
    rw [hpre]
    exact sizedAt_left hall
  have hlen : (#[] ++ sfx : Array Claims.Msg).size = (Claims.ZKServer.Count.consCounts st sch extras.size).length := by
    rw [← hpre, List.length_map, Array.length_toList, Array.empty_append]
  refine ⟨r, tr, ?_, hout⟩
  rw [← hlen]
  exact H hs

/-- **`repCons_asksOf` at `C`'s every round**: where `C`'s sizes are `consCounts`, `repCons` returns at round
`C.size`. -/
theorem repCons_asksOf_size (C : Array (Array F128)) {st : Setup} (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35)
    (hkm : st.fold.kLog ≤ st.m) (hkk : st.kLog = st.fold.kLog) (hmk : st.merkle = MerkleScheme.hm96Sha512)
    (hfold : ∀ (a b : F128) (lw ρ ts : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ts.size = st.fold.kLog - 6 → ∃ w, st.fold.folded a b lw ρ ts = .ok w ∧ w.size = 64)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = st.fold.kLog - 6 →
      ∃ v, st.fold.fold a lw ρ = .ok v)
    {sch : Flock.Schedule} {tp : ℕ} (hz : Zk.schedule st.m = .ok (sch, tp)) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (hes : extras.size + 2 < 2 ^ 64)
    (hC : C.toList.map Array.size = Claims.ZKServer.Count.consCounts st sch extras.size)
    (hρ : Zk.rhoIndependent
      ((⟨(C[FlockComplete.CutEq.ligAt 0 st extras + 2 + sch.k0]!)[0]!,
          (C[FlockComplete.CutEq.ligAt 0 st extras + 2 + sch.k0]!)[1]!⟩ : F256),
        (⟨(C[FlockComplete.CutEq.ligAt 0 st extras + 3 + sch.k0]!)[0]!,
          (C[FlockComplete.CutEq.ligAt 0 st extras + 3 + sch.k0]!)[1]!⟩ : F256)) = true) :
    ∃ r tr, ((repCons (asksOf C) (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals
        extra lanePad salt).run (Transcript.init #[] dom)).run 0 = .ok ((r, tr), C.size) ∧
      r.1.zerocheck.finalC = vals[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]! +
        pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fc]! ∧
      ∃ (rp : Zk.Replayed) (t' : Transcript) (lgc : Array (String × Zk.Lin)) (rec : Zk.L0Rec),
        (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0) r.1 ic.root).run (Transcript.init tr.rounds dom) =
          .ok (rp, t') ∧
        Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rec r.1.ligerito r.2.1 = .ok lgc ∧
        r.2.2 = rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! r.1.ringSwitches[0]!)),
          ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! r.1.ringSwitches[1]!))] ++ lgc := by
  have hl : (Claims.ZKServer.Count.consCounts st sch extras.size).length = C.size := by
    rw [← hC, List.length_map, Array.length_toList]
  rw [← hl]
  exact repCons_asksOf C h25 h35 hkm hkk hmk hfold hf hz tPad rep dom g cm0 extras pads ic vals extra lanePad salt hes
    ⟨[], by rw [List.append_nil, hC]⟩ hρ

end Flock.Firewall.Finish
