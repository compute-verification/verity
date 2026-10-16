import Proofs.Flock.Claims.Complete.Shape
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# The zerocheck goal

`zerocheck_goal`: the skeleton's `Zerocheck` at every honest run, given that each table's trace satisfies the block
R1CS (`TraceR1CS`) and what the session walk establishes for each rep (`ZcAt`): the decoded proof's zerocheck is the
prover's at the replies from the rep's first send (`repStart`'s) on, and the recorded stream carries the prover's
rounds.

* `coinOf_bytes`: the prover reads each coin element back from the coin's bytes.
* `Run.outer_ne_one`: at a good case no zerocheck outer coin is `1` (`¬ Bad`).
* `zerocheck_replay`: the replay returns on rounds that carry the prover's (`zerocheckWith_of` with `fold_final`).
* `repP_zerocheck`: a rep that returns carries the prover's zerocheck and the level-0 cap, at the replies from its
  first send on.
-/

namespace FlockComplete.Replay

open Flock FlockDeploy Transcript FlockSoundness.Discharge.ZkView

/-! ## The coins the prover reads back -/

theorem flatMap_getElem? {α β : Type} {f : α → List β} {n : ℕ} (hf : ∀ a, (f a).length = n) :
    ∀ (l : List α) (k j : ℕ), j < n → (l.flatMap f)[n * k + j]? = l[k]?.bind fun a => (f a)[j]?
  | [], k, j, _ => by simp
  | a :: l, 0, j, hj => by
    rw [List.flatMap_cons, Nat.mul_zero, Nat.zero_add, List.getElem?_append_left (by rw [hf]; exact hj)]
    simp
  | a :: l, k + 1, j, hj => by
    rw [List.flatMap_cons, List.getElem?_append_right (by rw [hf, Nat.mul_succ]; omega), hf,
      show n * (k + 1) + j - n = n * k + j by rw [Nat.mul_succ]; omega, flatMap_getElem? hf l k j hj]
    simp

theorem toBytes_get! (x : F128) {j : ℕ} (hj : j < 16) :
    x.toBytes.get! j =
      if j < 8 then (x.lo >>> (8 * j).toUInt64).toUInt8 else (x.hi >>> (8 * (j - 8)).toUInt64).toUInt8 := by
  rw [get!_eq, getElem!_pos _ _ (by simp [F128.toBytes]; exact hj)]
  simp [F128.toBytes]

theorem bytes_get! (c : Coin) {k : ℕ} (hk : k < 65536) {j : ℕ} (hj : j < 16) :
    (Claims.ofMsg c.bytes).get! (16 * k + j) = (c ⟨k, hk⟩).toBytes.get! j := by
  have h := flatMap_getElem? (f := fun i : Fin 65536 => (c i).toBytes.data.toList) (n := 16)
    (fun i => by simp [F128.toBytes]) (List.finRange 65536) k j hj
  have hf : (List.finRange 65536)[k]? = some ⟨k, hk⟩ := by
    rw [List.getElem?_eq_getElem (by simpa using hk)]
    simp
  rw [hf, Option.bind_some] at h
  have e : (Claims.ofMsg c.bytes).data[16 * k + j]? = (c ⟨k, hk⟩).toBytes.data[j]? := by
    show c.bytes.toArray[16 * k + j]? = _
    rw [List.getElem?_toArray, ← Array.getElem?_toList (xs := (c ⟨k, hk⟩).toBytes.data)]
    exact h
  rw [get!_eq, get!_eq, getElem!_def, getElem!_def, e]

theorem u64_of_bytes {b : ByteArray} {o : ℕ} {x : UInt64}
    (h : ∀ i < 8, b.get! (o + i) = (x >>> (8 * i).toUInt64).toUInt8) : u64OfLE b o = x := by
  rw [u64OfLE_eq, List.foldl_ext _
    (fun acc i => acc ||| (x >>> (8 * i).toUInt64).toUInt8.toUInt64 <<< (8 * i).toUInt64) _ fun acc i hi => by
      rw [h i (by have := (List.mem_range'_1.1 hi).2; omega)]]
  apply UInt64.toNat_inj.1
  rw [foldl_le64 x 8 le_rfl]
  exact Nat.mod_eq_of_lt x.toNat_lt

/-- **The prover reads each coin element back** from the coin's bytes (`coinOf` of `Coin.bytes`). -/
theorem coinOf_bytes (c : Coin) {k : ℕ} (hk : k < 65536) : coinOf c.bytes k = c.at k := by
  have hlo : u64OfLE (Claims.ofMsg c.bytes) (16 * k) = (c ⟨k, hk⟩).lo := u64_of_bytes fun i hi => by
    rw [bytes_get! c hk (by omega), toBytes_get! _ (by omega)]
    simp only [hi, ↓reduceIte]
  have hhi : u64OfLE (Claims.ofMsg c.bytes) (16 * k + 8) = (c ⟨k, hk⟩).hi := u64_of_bytes fun i hi => by
    rw [Nat.add_assoc, bytes_get! c hk (by omega), toBytes_get! _ (by omega)]
    simp only [show ¬ 8 + i < 8 by omega, ↓reduceIte, Nat.add_sub_cancel_left]
  show (⟨u64OfLE (Claims.ofMsg c.bytes) (16 * k), u64OfLE (Claims.ofMsg c.bytes) (16 * k + 8)⟩ : F128) = c.at k
  rw [hlo, hhi]
  simp only [Coin.at, hk, ↓reduceDIte]

/-! ## The run's table and outer coins -/

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- Table `i`'s trace, as the honest session proves it. -/
def _root_.FlockComplete.Run.trace (o : Run H rel) (i : ℕ) : Trace := (tracesOf o.sts o.w)[i]!

/-- Table `i`'s rep `r`'s zerocheck coins: the replies from the rep's first send on. -/
def _root_.FlockComplete.Run.zcCoins (o : Run H rel) (i r : ℕ) : Zc.Coins :=
  Zc.coinsAt o.resp (repStart o.spec o.sts (2 * i + r))

theorem _root_.FlockComplete.Run.tab_m (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) :
    o.tabs[i]!.m = o.sts[i]!.m ∧ 22 ≤ o.tabs[i]!.m ∧ o.tabs[i]!.m ≤ 35 := by
  obtain ⟨st, hsts, -, h22, h35⟩ := o.table
  unfold Run.tabs tabsOf at hi ⊢
  rw [hsts] at hi ⊢
  simp only [Array.map_singleton, List.size_toArray, List.length_singleton, Nat.lt_one_iff] at hi
  subst hi
  simp [Setup.ofCircuit, h22, h35]

/-- **No zerocheck outer coin is `1`** at a good case: table `i`'s rep `r`'s `r_outer` round is `repStart`'s second. -/
theorem _root_.FlockComplete.Run.outer_ne_one (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2)
    {k : ℕ} (hk : k < o.tabs[i]!.m - 13) :
    (Claims.coinSeq o.r (repStart o.spec o.sts (2 * i + r) + 1)).at k ≠ 1 := by
  intro h1
  apply o.good
  refine ⟨(repStart o.spec o.sts (2 * i + r) + 1, k), ?_, h1⟩
  have hsz : o.tabs.size = o.sts.size := by simp [Run.tabs, tabsOf]
  have hq : (2 * i + r) / 2 = i := by omega
  unfold outerSlots
  rw [o.setup]
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  exact ⟨2 * i + r, by omega, k, by rw [hq, ← (o.tab_m hi).1]; exact hk, rfl, rfl⟩

/-- The prover's `r_outer` coins at table `i`'s rep `r` are not `1`. -/
theorem _root_.FlockComplete.Run.zc_c1 (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) :
    ∀ k < o.tabs[i]!.m - 13, (o.zcCoins i r).c1 k ≠ 1 := by
  intro k hk
  obtain ⟨-, -, hm35⟩ := o.tab_m hi
  show coinOf (Claims.coinSeq o.r (repStart o.spec o.sts (2 * i + r) + 1)).bytes k ≠ 1
  rw [coinOf_bytes _ (by omega)]
  exact o.outer_ne_one hi hr hk

/-! ## The goal -/

/-- **The replay returns** on rounds that carry the prover's zerocheck rounds, from the rep's start, when the trace
satisfies the block R1CS and no outer coin is `1`. -/
theorem zerocheck_replay (hs : Hashes V) {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (cs : Zc.Coins) (tr : Trace)
    (R : Array Round) (dom : ByteArray)
    (hR : Agrees R (Zc.tEnd m digest cap cs tr (Transcript.init #[] dom)).rounds 0)
    (hR1CS : ∀ x < 2 ^ m, (tr.az[x]! && tr.bz[x]!) = tr.z[x]!) (hc1 : ∀ k < m - 13, cs.c1 k ≠ 1) :
    ∃ v, (bindAndZerocheckWith hs m digest cap (Zc.proof m cs tr)).run (Transcript.init R dom) = .ok v :=
  ⟨_, zerocheckWith_of hs hm digest cap cs tr (Transcript.init #[] dom) rfl hR (Zc.fold_final hm cs tr hR1CS hc1)⟩

/-- **Each table's trace satisfies the block R1CS** bitwise (`a·b = c`). -/
def TraceR1CS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ x < 2 ^ o.tabs[i]!.m,
    ((o.trace i).az[x]! && (o.trace i).bz[x]!) = (o.trace i).z[x]!

/-- **What the session walk establishes for table `i`'s rep `r`'s zerocheck**: the decoded proof's zerocheck is the
prover's on the table's trace at the replies from the rep's first send on, and the recorded stream carries the
prover's rounds through zerocheck from the rep's start. -/
def ZcAt (o : Run H rel) (i r : ℕ) : Prop :=
  (o.proof i r).zerocheck = Zc.proof o.tabs[i]!.m (o.zcCoins i r) (o.trace i) ∧
    Agrees (o.stream i r).rounds (Zc.tEnd o.tabs[i]!.m o.tabs[i]!.digest (flat (o.proof i r).cap) (o.zcCoins i r)
      (o.trace i) (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).rounds 0

theorem ok_val {α : Type} [Inhabited α] {x : Except String α} (h : ∃ v, x = .ok v) : x = .ok (val x) := by
  obtain ⟨v, rfl⟩ := h
  rfl

/-- **The zerocheck goal**: zerocheck replays at every honest run, on R1CS-valid traces, at every rep the session
walk aligns. -/
theorem zerocheck_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hR1 : TraceR1CS H rel) (hzc : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ZcAt o i r) : Zerocheck H rel := by
  intro o i hi r hr
  obtain ⟨hp, hag⟩ := hzc o i hi r hr
  obtain ⟨-, hm, -⟩ := o.tab_m hi
  refine ok_val ?_
  rw [hp]
  exact zerocheck_replay o.hs (by omega) _ _ _ _ _ _ hag (hR1 o i hi) (o.zc_c1 hi hr)

/-! ## What a returning rep carries -/

theorem Ret.run_bind {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t : Transcript} {Q : α × Transcript → Prop}
    {P : β × Transcript → Prop} (hx : Ret (x.run t) Q) (hf : ∀ a t', Q (a, t') → Ret ((f a).run t') P) :
    Ret ((x >>= f).run t) P := by
  rw [StateT.run_bind]
  exact FlockComplete.Ret.bind hx fun ⟨a, t'⟩ h => hf a t' h

/-- **A rep that returns** carries the prover's zerocheck, at the replies from its first send on, and the level-0 cap. -/
theorem repP_zerocheck (resp : ℕ → Claims.Msg) (hsP : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t : Transcript) (l : Array Claims.Msg) {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repP (asksS resp) hsP st sch tr tbl cm0 extras salt nonce).run t).go l = (.ok (p, t'), l')) :
    p.zerocheck = Zc.proof st.m (Zc.coinsAt resp l.size) tr ∧ p.cap = cm0.cap := by
  obtain ⟨l₁, -, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  unfold repP at h
  rw [bind_go hz] at h
  refine (?_ : Ret _ fun r : Proof × Transcript => r.1.zerocheck = Zc.proof st.m (Zc.coinsAt resp l.size) tr ∧
    r.1.cap = cm0.cap) l₁ (p, t') l' h
  refine Ret.run_bind (Ret.true _) fun ⟨lp, lc⟩ _ _ => ?_
  refine Ret.run_bind (Ret.true _) fun ⟨ss, T, sws⟩ _ _ => ?_
  refine Ret.run_bind (Ret.true _) fun lg _ _ => ?_
  rw [StateT.run_pure]
  exact FlockComplete.Ret.pure ⟨rfl, rfl⟩

end FlockComplete.Replay
