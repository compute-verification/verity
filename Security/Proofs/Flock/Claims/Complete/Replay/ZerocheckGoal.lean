import Proofs.Flock.Claims.Complete.Shape
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Proofs.Flock.Claims.Complete.Json.Goals

/-!
# The zerocheck goal

`zerocheck_goal`: the skeleton's `Zerocheck` at every honest run, given that each table's trace satisfies the block
R1CS (`TraceR1CS`) and what the session walk establishes for each rep (`ZcAt`): the decoded proof's zerocheck is the
prover's at the replies from the rep's first send (`repStart`'s) on, and the recorded stream carries the prover's
rounds.

* `Run.outer_ne_one`: at a good case no zerocheck outer coin is `1` (`¬ Bad`).
* `zerocheck_replay`: the replay returns on rounds that carry the prover's (`zerocheckWith_of` with `fold_final`).
* `repP_zerocheck`: a rep that returns carries the prover's zerocheck and the level-0 cap, at the replies from its
  first send on.
-/

namespace FlockComplete.Replay

open Flock FlockDeploy Transcript

/-! ## The run's table and outer coins -/

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- Table `i`'s trace, as the honest session proves it. -/
def _root_.FlockComplete.Run.trace (o : Run H rel) (i : ℕ) : Trace := (tracesOf o.sts o.w)[i]!

/-- Table `i`'s rep `r`'s zerocheck coins: the replies from the rep's first send on. -/
def _root_.FlockComplete.Run.zcCoins (o : Run H rel) (i r : ℕ) : Zc.Challenges :=
  Zc.coinsAt o.resp (repStart o.spec o.sts (2 * i + r))

theorem _root_.FlockComplete.Run.tab_m (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) :
    o.tabs[i]!.m = o.sts[i]!.m ∧ 22 ≤ o.tabs[i]!.m ∧ o.tabs[i]!.m ≤ 35 := by
  obtain ⟨st, hsts, -, h22, h35, -⟩ := o.table
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
  show coinOf (Claims.coinSeq o.r (repStart o.spec o.sts (2 * i + r) + 1)).bytes k ≠ 1
  rw [JsonRT.coinOf_bytes]
  exact o.outer_ne_one hi hr hk

/-! ## The goal -/

/-- **The replay returns** on rounds that carry the prover's zerocheck rounds, from the rep's start, when the trace
satisfies the block R1CS and no outer coin is `1`. -/
theorem zerocheck_replay (hs : Hashes V) {m : ℕ} (hm : 14 ≤ m) (digest cap : ByteArray) (cs : Zc.Challenges) (tr : Trace)
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
