module

public import Proofs.Flock.SessionLoop.FinInnerCut
public import Proofs.Flock.Claims.ZK.FinishInner
public import Proofs.Flock.Claims.ZK.FinishStop
public import Proofs.Flock.Claims.ZK.RepCons
public import Proofs.Flock.Claims.ZK.FramePlan
public import Proofs.Flock.Claims.ZK.SessionLean
public import Proofs.Flock.Claims.Complete.Zk.Json.Runs
public import Proofs.Flock.Claims.Complete.ZkTables.LaneRows
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckReplay
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.LincheckReplay
public import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRep

/-!
# The loop's finish at a rep's inner rounds, run forward

What makes the finish's live inner proof (`Loop.finInner`) return: its pads commitment returns, its `repCons` returns at
the request's rounds, `consWhy` passes, and its inner proof, answered by the request's coins and past them by ones
(`asksPad`), returns where each of its four rounds is past the coins or holds the count it squeezes, and its `β` is not
`0`. Each part is named here, none a premise of a guarantee:

* `Loop.OkAt`: a run that returns, from a transcript and a round.
* `Loop.squeezeP_pad_ok`, `Loop.sampleP_pad_ok`, `Loop.sampleVecP_pad_ok`: a squeeze answered by `asksPad` returns
  round `k`'s coins, where round `k` is past the coins or holds the squeeze's count.
* `Loop.innerP_pad_ok`: the inner proof answered by `asksPad` returns.
* `Loop.innerCommitP_fw_ok`: the pads commitment returns.
* `Loop.finInner_ok_of`: `finInner` returns where its parts do.
* `Loop.repPlan_le`: each round of a rep's plan at a `--zk` schedule of `m ≤ 35`, at most `65,534` claims, counts at
  most `COIN_MAX` coins; `Loop.repPlan_cons`: the plan is `repCons`'s rounds (`Count.consCounts`), then the inner
  proof's four.
* `Loop.prefix_of_sizes`: coins whose counts are a plan's, over its first rounds, start with those rounds' counts.
* **`Loop.finInner_ok_at`**: `finInner` returns at a request whose coins `repCons` takes (`Finish.repCons_asksOf`,
  `repCons_asksPad`), whose pair's coins are independent, whose value at `final_c` is the c_eval of its round-1c values
  at its round-2 coin, whose inner proof's rounds are past its coins or hold their counts, and whose `β` is not `0`.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 TMW Transcript liftTM Stmt SessionSpec)
open _root_.Flock.Firewall.Finish (Tab RepIn)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

/-- **A run that returns**: `x` from the transcript `t` at round `k` returns some value, transcript and round. -/
def OkAt {α : Type} (x : TMW (StateT ℕ V) α) (t : Flock.Transcript) (k : ℕ) : Prop :=
  ∃ a t' k', (x.run t).run k = .ok ((a, t'), k')

theorem okAt_bind {α β : Type} {x : TMW (StateT ℕ V) α} {f : α → TMW (StateT ℕ V) β} {t : Flock.Transcript} {k : ℕ}
    {a : α} {k' : ℕ} (hx : ∃ t', (x.run t).run k = .ok ((a, t'), k')) (hf : ∀ t', OkAt (f a) t' k') :
    OkAt (x >>= f) t k := by
  obtain ⟨t', hx⟩ := hx
  obtain ⟨b, t'', k'', hb⟩ := hf t'
  refine ⟨b, t'', k'', ?_⟩
  rw [StateT.run_bind, StateT.run_bind, hx]
  exact hb

theorem okAt_pure {α : Type} (a : α) (t : Flock.Transcript) (k : ℕ) : OkAt (pure a : TMW (StateT ℕ V) α) t k :=
  ⟨a, t, k, rfl⟩

theorem liftTM_run {α : Type} {x : _root_.Flock.Transcript.TM α} {t : Flock.Transcript} {a : α} {t' : Flock.Transcript} {k : ℕ}
    (h : x t = .ok (a, t')) : ((liftTM x : TMW (StateT ℕ V) α).run t).run k = .ok ((a, t'), k) := by
  change (monadLift (x t) : StateT ℕ V _).run k = _
  rw [h]
  rfl

/-- Where round `k` is past `C` or holds the count `c` frames, `asksPad C` answers it with `padCoin C k`. -/
theorem asksPad_round_ok (C : Array (Array F128)) (c : ByteArray) {k : ℕ}
    (hC : ∀ cs, C[k]? = some cs → cs.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat) :
    ((asksPad C).round c).run k = .ok (padCoin C k, k + 1) := by
  simp only [asksPad, StateT.run_bind, StateT.run_get, pure_bind]
  unfold padCoin
  rcases hk : C[k]? with _ | cs
  · rfl
  · dsimp only
    split
    · rename_i hne
      simp [hC cs hk] at hne
    · rfl

/-- **A squeeze answered by `asksPad C`** of `0 < n < 2^64` coins from round `k`, where round `k` is past `C` or holds
`n` coins, returns round `k`'s coins and moves to round `k + 1`. -/
theorem squeezeP_pad_ok (C : Array (Array F128)) (kind : UInt8) {n : ℕ} (hn : n ≠ 0) (hn64 : n < 2 ^ 64)
    (t : Flock.Transcript) {k : ℕ} (hC : ∀ cs, C[k]? = some cs → cs.size = n) :
    ∃ t', ((_root_.Flock.Firewall.Finish.squeezeP (asksPad C) kind n).run t).run k =
      .ok (((Array.range n).map (padCoin C k), t'), k + 1) :=
  ⟨_, by
    change ((_root_.Flock.Firewall.Finish.squeezeP (asksPad C) kind n t).run k) = _
    unfold _root_.Flock.Firewall.Finish.squeezeP
    dsimp only
    rw [ite_eq_right hn, StateT.run_bind,
      asksPad_round_ok C _ fun cs hcs => (hC cs hcs).trans (Count.framed_count t kind hn64).symm]
    rfl⟩

theorem sampleVecP_pad_ok (C : Array (Array F128)) {n : ℕ} (hn : n ≠ 0) (hn64 : n < 2 ^ 64) (t : Flock.Transcript) {k : ℕ}
    (hC : ∀ cs, C[k]? = some cs → cs.size = n) :
    ∃ t', ((_root_.Flock.Firewall.Finish.sampleVecP (asksPad C) n).run t).run k =
      .ok (((Array.range n).map (padCoin C k), t'), k + 1) :=
  squeezeP_pad_ok C 2 hn hn64 t hC

/-- A coin answered by `asksPad C` from round `k`, where round `k` is past `C` or holds one coin, is round `k`'s first. -/
theorem sampleP_pad_ok (C : Array (Array F128)) (t : Flock.Transcript) {k : ℕ} (hC : ∀ cs, C[k]? = some cs → cs.size = 1) :
    ∃ t', ((_root_.Flock.Firewall.Finish.sampleP (asksPad C)).run t).run k = .ok ((padCoin C k 0, t'), k + 1) := by
  obtain ⟨t₁, h1⟩ := squeezeP_pad_ok C 1 (by decide) (by norm_num) t hC
  have e : ((Array.range 1).map (padCoin C k)).getD 0 0 = padCoin C k 0 := by
    simp [Array.getD_eq_getD_getElem?]
  refine ⟨t₁, ?_⟩
  unfold _root_.Flock.Firewall.Finish.sampleP
  rw [StateT.run_bind, StateT.run_bind, h1, ← e]
  rfl

theorem tauLift_run {hh : ByteArray → ByteArray} {tau : F128} {salt : ByteArray} {t : Flock.Transcript} {k : ℕ} :
    ((StateT.lift (_root_.Flock.Zk.tauCommitmentWith
      (_root_.Flock.Hashes.ofSha512 hh : _root_.Flock.Hashes (StateT ℕ V)) tau salt) :
      TMW (StateT ℕ V) ByteArray).run t).run k = .ok ((FlockComplete.Replay.InZ.tcOf hh tau salt, t), k) := by
  rw [tauCommitmentWith_sv]
  rfl

/-- **The inner proof answered by `asksPad C` from round `k` returns**, where each of its rounds `k`, `k + 1`, `k + 2`
and `k + 3` is past `C` or holds the count it squeezes (one, `|cons| + 3`, one and `Zk.Q`) and its `β`, round `k + 2`'s
first coin, is not `0`. -/
theorem innerP_pad_ok (C : Array (Array F128)) (lay : _root_.Flock.Zk.PadLayout)
    (p : _root_.Flock.Firewall.Finish.PadsZk) (ic : _root_.Flock.Firewall.Finish.InnerCommit) (finalC : F128)
    (cons : Array (String × _root_.Flock.Zk.Lin)) (hcs : cons.size + 3 < 2 ^ 64) (t : Flock.Transcript) {k : ℕ}
    (h0 : ∀ cs, C[k]? = some cs → cs.size = 1) (h1 : ∀ cs, C[k + 1]? = some cs → cs.size = cons.size + 3)
    (h2 : ∀ cs, C[k + 2]? = some cs → cs.size = 1)
    (h3 : ∀ cs, C[k + 3]? = some cs → cs.size = _root_.Flock.Zk.Q)
    (hβ : (padCoin C (k + 2) 0).isZero = false) :
    OkAt (_root_.Flock.Firewall.Finish.innerP (asksPad C) (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) lay p
      ic finalC cons) t k := by
  have hsz : ∀ e r s, (cons ++ _root_.Flock.Zk.triple lay e r s).size = cons.size + 3 := fun _ _ _ => by
    simp [_root_.Flock.Zk.triple]
  unfold _root_.Flock.Firewall.Finish.innerP
  try dsimp only
  refine okAt_bind ⟨_, liftTM_run rfl⟩ fun t₁ => ?_
  try dsimp only
  refine okAt_bind ⟨_, liftTM_run rfl⟩ fun t₂ => ?_
  try dsimp only
  refine okAt_bind (sampleP_pad_ok C t₂ h0) fun t₃ => ?_
  try dsimp only
  refine okAt_bind ⟨_, liftTM_run rfl⟩ fun t₄ => ?_
  try dsimp only
  refine okAt_bind (sampleVecP_pad_ok C (by rw [hsz]; omega) (by rw [hsz]; exact hcs) t₄
    fun cs hc => (h1 cs hc).trans (hsz _ _ _).symm) fun t₅ => ?_
  try dsimp only
  refine okAt_bind ⟨_, tauLift_run⟩ fun t₆ => ?_
  try dsimp only
  refine okAt_bind ⟨_, liftTM_run rfl⟩ fun t₇ => ?_
  try dsimp only
  refine okAt_bind (sampleP_pad_ok C t₇ h2) fun t₈ => ?_
  try dsimp only
  split
  next hz => exact absurd (hβ.symm.trans hz) Bool.false_ne_true
  next =>
    try dsimp only
    try simp only [pure_bind]
    refine okAt_bind ⟨_, liftTM_run rfl⟩ fun t₉ => ?_
    try dsimp only
    refine okAt_bind (sampleVecP_pad_ok C (by decide) (by decide) t₉ h3) fun t₁₀ => ?_
    exact okAt_pure _ _ _

/-- **The pads commitment returns** through SHA-512 (`JsonRT.innerCommitP_ok`, through `Finish.innerCommitP_eq`). -/
theorem innerCommitP_fw_ok (p : _root_.Flock.Firewall.Finish.PadsZk) (salt : ℕ → ByteArray) :
    ∃ ic, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) p salt :
      V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic := by
  obtain ⟨icC, h⟩ := FlockComplete.JsonRT.innerCommitP_ok _root_.Flock.Sha512.hash p.toC salt
  have e := _root_.Flock.Firewall.Finish.innerCommitP_eq (M := V)
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) p salt
  rw [h] at e
  rcases hic : (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) p
    salt : V _root_.Flock.Firewall.Finish.InnerCommit) with err | ic
  · rw [hic] at e
    cases e
  · exact ⟨ic, rfl⟩

theorem v_bind_of {α β : Type} {x : V α} {f : α → V β} {a : α} {b : β} (hx : x = .ok a) (hf : f a = .ok b) :
    x >>= f = .ok b := by
  rw [hx]
  exact hf

theorem tmw_bind_run {α β : Type} {x : TMW (StateT ℕ V) α} {f : α → TMW (StateT ℕ V) β} {t : Flock.Transcript} {k : ℕ}
    {a : α} {t' : Flock.Transcript} {k' : ℕ} (hx : (x.run t).run k = .ok ((a, t'), k')) :
    ((x >>= f).run t).run k = ((f a).run t').run k' := by
  rw [StateT.run_bind, StateT.run_bind, hx]
  rfl

theorem getThe_bind_run {β : Type} (f : ℕ → TMW (StateT ℕ V) β) (t : Flock.Transcript) (k : ℕ) :
    ((getThe ℕ >>= f : TMW (StateT ℕ V) β).run t).run k = ((f k).run t).run k := rfl

/-- **`consWhy` passes** where the rep took the request's rounds and its `zerocheck c_eval` constraints vanish on the
pads (`consWhy_none`'s converse). -/
theorem consWhy_none_of {n : ℕ} {cons : Array (String × _root_.Flock.Zk.Lin)} {h : Array F128}
    (hz : ∀ q ∈ cons, q.1 = "zerocheck c_eval" → _root_.Flock.Firewall.Finish.linAt q.2 h = 0) :
    _root_.Flock.Firewall.Finish.consWhy n n cons h = none := by
  have he : _root_.Flock.Firewall.Finish.failingOn (cons.filter (·.1 == "zerocheck c_eval")) h = #[] :=
    _root_.Flock.Firewall.Finish.failingOn_eq_empty_iff.2 fun q hq => by
      obtain ⟨hq, hname⟩ := Array.mem_filter.1 hq
      exact hz q hq (by simpa using hname)
  unfold _root_.Flock.Firewall.Finish.consWhy
  simp [he]

/-- **`finInner` returns where its parts do**: its pads commitment returns, its `repCons`, answered by `asksPad` from
round `0` and a transcript with no rounds, returns at the request's round `n`, `consWhy` passes, and its inner proof
returns from there. -/
theorem finInner_ok_of {t : Tab} {rep : ℕ} {dom : ByteArray} {n : ℕ} {q : RepIn}
    {ic : _root_.Flock.Firewall.Finish.InnerCommit}
    (hic : (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
      (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
      V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic)
    {p : _root_.Flock.Proof} {zl : _root_.Flock.Zk.Level0Msgs} {cons : Array (String × _root_.Flock.Zk.Lin)}
    {tF : Flock.Transcript}
    (hrep : ((_root_.Flock.Firewall.Finish.repCons (asksPad q.coins)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) t.st t.sch t.tPad rep dom t.g t.cm0
        (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws) ic q.vals t.extra
        t.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
        (_root_.Flock.Transcript.init #[] dom)).run 0 = .ok (((p, zl, cons), tF), n))
    (hcw : _root_.Flock.Firewall.Finish.consWhy n n cons (_root_.Flock.Firewall.Finish.padsOf q.draws).h = none)
    (hin : OkAt (_root_.Flock.Firewall.Finish.innerP (asksPad q.coins)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
      (_root_.Flock.Firewall.Finish.padsOf q.draws) ic p.zerocheck.finalC cons) tF n) :
    ∃ (fc' : F128) (tc : ByteArray), finInner t rep dom n q = .ok (fc', tc) := by
  obtain ⟨ip, tG, kG, hip⟩ := hin
  refine ⟨p.zerocheck.finalC, ip.tauCommit, ?_⟩
  unfold finInner
  refine v_bind_of hic ?_
  try dsimp only
  refine v_bind_of (a := (((p.zerocheck.finalC, ip.tauCommit), tG), kG)) ?_ rfl
  rw [tmw_bind_run hrep]
  try dsimp only
  rw [getThe_bind_run]
  try dsimp only
  rw [hcw]
  try dsimp only
  try simp only [pure_bind]
  rw [tmw_bind_run hip]
  rfl

/-! ## A rep's plan, bounded -/

theorem all_append {a b : Array ℕ} {B : ℕ} (ha : ∀ c ∈ a, c ≤ B) (hb : ∀ c ∈ b, c ≤ B) : ∀ c ∈ a ++ b, c ≤ B :=
  fun c hc => (Array.mem_append.1 hc).elim (ha c) (hb c)

theorem all_replicate {n v B : ℕ} (h : v ≤ B) : ∀ c ∈ Array.replicate n v, c ≤ B :=
  fun _ hc => (Array.mem_replicate.1 hc).2 ▸ h

theorem all_lit {l : List ℕ} {B : ℕ} (h : ∀ c ∈ l, c ≤ B) : ∀ c ∈ l.toArray, c ≤ B :=
  fun c hc => h c (by simpa using hc)

theorem queryCoins_le {sch : _root_.Flock.Schedule} (hq : ∀ i, (sch.levels.getD i default).q + 1 ≤ 65536) (i : ℕ) :
    ∀ c ∈ _root_.Flock.Firewall.Session.queryCoins sch i, c ≤ 65536 := by
  have h1 := hq i
  have h2 := Nat.log2_le_self ((sch.levels.getD i default).q - 1)
  unfold _root_.Flock.Firewall.Session.queryCoins
  dsimp only
  refine all_lit fun c hc => ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
  omega

/-- Each round of Ligerito's plan counts at most `65,536` coins, where its `n` and each level's `q` are below. -/
theorem ligeritoCoins_le {sch : _root_.Flock.Schedule} (hn : sch.n + 1 ≤ 65536)
    (hq : ∀ i, (sch.levels.getD i default).q + 1 ≤ 65536) :
    ∀ c ∈ _root_.Flock.Firewall.Session.ligeritoCoins sch, c ≤ 65536 := by
  have hs : ∀ i, _root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) i ≤ 65536
    | 0 => show sch.n - sch.k0 + 1 ≤ 65536 by omega
    | i + 1 => by
      have := Count.splitAt_le (sch.n - sch.k0 + 1) i
      show _root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) i - 4 + 1 ≤ 65536
      omega
  unfold _root_.Flock.Firewall.Session.ligeritoCoins
  dsimp only
  refine all_append (all_append (all_append (all_append (all_lit ?_) (all_replicate (by omega))) (all_lit ?_))
    (queryCoins_le hq 0)) ?_
  · intro c hc
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
    omega
  · intro c hc
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
    have := hs 0
    omega
  · intro c hc
    obtain ⟨i, -, hc⟩ := Array.mem_flatMap.1 hc
    refine all_append (all_replicate (by omega)) ?_ c hc
    split
    · refine all_append (all_lit fun c hc => ?_) (queryCoins_le hq _)
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      have := hs (i + 1)
      omega
    · exact queryCoins_le hq _

/-- **Each round of a rep's plan counts at most `COIN_MAX` coins**, at a `--zk` schedule of `m ≤ 35` whose `n` and
levels' `q` are below `65,536`, with at most `65,536` claims. -/
theorem repPlan_le {spec : SessionSpec} {st : Stmt} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule st.m = .ok (sch, tp)) (h35 : st.m ≤ 35) (hn : sch.n + 1 ≤ 65536)
    (hq : ∀ i, (sch.levels.getD i default).q + 1 ≤ 65536) (hc : 2 + spec.points * st.regions.size ≤ 65536) (i : ℕ) :
    (_root_.Flock.Firewall.Session.repPlan spec st).getD i 0 ≤ _root_.Flock.Firewall.Frames.COIN_MAX := by
  have hall : ∀ c ∈ _root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog ++
      Array.replicate (2 + spec.points * st.regions.size) 7 ++ #[2 + spec.points * st.regions.size] ++
      _root_.Flock.Firewall.Session.ligeritoCoins sch ++
      #[1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q], c ≤ 65536 := by
    refine all_append (all_append (all_append (all_append ?_ (all_replicate (by omega))) (all_lit ?_))
      (ligeritoCoins_le hn hq)) (all_lit ?_)
    · unfold _root_.Flock.Firewall.Session.replayCoins
      refine all_append (all_append (all_append (all_append (all_lit ?_) (all_replicate (by omega))) (all_lit ?_))
        (all_replicate (by omega))) (all_lit ?_) <;>
      · intro c hc
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
        omega
    · intro c hc
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      omega
    · intro c hc
      simp only [List.mem_cons, List.not_mem_nil, or_false, _root_.Flock.Firewall.Session.INNER_CONS,
        _root_.Flock.Zk.Q] at hc
      omega
  show _ ≤ 65536
  rw [Array.getD_eq_getD_getElem?]
  rcases hget : (_root_.Flock.Firewall.Session.repPlan spec st)[i]? with _ | c
  · exact Nat.zero_le _
  · obtain ⟨hi, e⟩ := Array.getElem?_eq_some_iff.1 hget
    have hmem : c ∈ _root_.Flock.Firewall.Session.repPlan spec st := e ▸ Array.getElem_mem hi
    simp only [_root_.Flock.Firewall.Session.repPlan, repCoins_ok hz, Except.toOption, Option.getD_some] at hmem
    exact hall c (Array.mem_filter.1 hmem).1

/-- **A rep's plan is `repCons`'s rounds, then the inner proof's four**, at the statement's setup and `--zk`
schedule. -/
theorem repPlan_cons {spec : SessionSpec} {st : Stmt} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule st.m = .ok (sch, tp)) :
    (_root_.Flock.Firewall.Session.repPlan spec st).toList =
      Count.consCounts (_root_.Flock.Setup.ofCircuit st) sch (spec.points * st.regions.size) ++
        [1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q] := by
  rw [Count.repPlan_list hz]
  rfl

/-- **Coins whose counts are a plan's start with its first rounds' counts**: where the plan is `cc` then more, and
each of the coins' first `|cc|` rounds counts the plan's coins. -/
theorem prefix_of_sizes {C : Array (Array F128)} {cc I : List ℕ} {P : Array ℕ} (hP : P.toList = cc ++ I)
    (hlen : cc.length ≤ C.size) (hs : ∀ k < cc.length, (C[k]!).size = P.getD k 0) :
    cc <+: C.toList.map Array.size := by
  refine List.prefix_iff_eq_take.2 (List.ext_getElem (by simp; omega) fun i h1 h2 => ?_)
  have hi : i < C.size := by omega
  rw [List.getElem_take, List.getElem_map, Array.getElem_toList (xs := C), ← getElem!_pos C i hi, hs i h1,
    Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hP, List.getElem?_append_left h1,
    List.getElem?_eq_getElem h1, Option.getD_some]

/-! ## `finInner` returns -/

set_option maxHeartbeats 1000000 in
/-- **`finInner` returns** at a request whose coins' counts start with `repCons`'s rounds, `n` of them
(`Finish.repCons_asksOf`, `repCons_asksPad`), whose pair's coins are independent, whose value at `final_c` is the c_eval
of its round-1c values at its round-2 coin (so `consWhy`'s `zerocheck c_eval` holds on the pads: `c_eval_eval`), whose
inner proof's four rounds are past its coins or hold their counts, and whose `β` is not `0` (`finInner_ok_of`). -/
theorem finInner_ok_at {t : Tab} {rep : ℕ} {dom : ByteArray} {n : ℕ} {q : RepIn} {tp : ℕ}
    (h25 : 25 ≤ t.st.m) (h35 : t.st.m ≤ 35) (hkm : t.st.fold.kLog ≤ t.st.m) (hkk : t.st.kLog = t.st.fold.kLog)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512)
    (hfold : ∀ (a b : F128) (lw ρ ts : Array F128), lw.size = 64 → ρ.size = t.st.fold.kLog - 6 →
      ts.size = t.st.fold.kLog - 6 → ∃ w, t.st.fold.folded a b lw ρ ts = .ok w ∧ w.size = 64)
    (hf : ∀ (a : F128) (lw ρ : Array F128), lw.size = 64 → ρ.size = t.st.fold.kLog - 6 →
      ∃ v, t.st.fold.fold a lw ρ = .ok v)
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp))
    (hes : (_root_.Flock.Firewall.Finish.claimsOf q.xs).size + 2 < 2 ^ 64)
    (hC : Count.consCounts t.st t.sch (_root_.Flock.Firewall.Finish.claimsOf q.xs).size <+:
      q.coins.toList.map Array.size)
    (hn : (Count.consCounts t.st t.sch (_root_.Flock.Firewall.Finish.claimsOf q.xs).size).length = n) (hn2 : 2 < n)
    (hρ : _root_.Flock.Zk.rhoIndependent
      ((⟨(q.coins[FlockComplete.CutEq.ligAt 0 t.st (_root_.Flock.Firewall.Finish.claimsOf q.xs) + 2 + t.sch.k0]!)[0]!,
          (q.coins[FlockComplete.CutEq.ligAt 0 t.st (_root_.Flock.Firewall.Finish.claimsOf q.xs) + 2 + t.sch.k0]!)[1]!⟩ :
          _root_.Flock.F256),
        (⟨(q.coins[FlockComplete.CutEq.ligAt 0 t.st (_root_.Flock.Firewall.Finish.claimsOf q.xs) + 3 + t.sch.k0]!)[0]!,
          (q.coins[FlockComplete.CutEq.ligAt 0 t.st (_root_.Flock.Firewall.Finish.claimsOf q.xs) + 3 + t.sch.k0]!)[1]!⟩ :
          _root_.Flock.F256)) = true)
    (hval : q.vals[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]! =
      _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (padCoin q.coins 2 0))
        ((Array.range 64).map fun i => q.vals[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).r1c i]!))
    (h0 : ∀ cs, q.coins[n]? = some cs → cs.size = 1)
    (h1 : ∀ cs, q.coins[n + 1]? = some cs → cs.size = _root_.Flock.Firewall.Session.INNER_CONS)
    (h2 : ∀ cs, q.coins[n + 2]? = some cs → cs.size = 1)
    (h3 : ∀ cs, q.coins[n + 3]? = some cs → cs.size = _root_.Flock.Zk.Q)
    (hβ : (padCoin q.coins (n + 2) 0).isZero = false) :
    ∃ (fc' : F128) (tc : ByteArray), finInner t rep dom n q = .ok (fc', tc) := by
  obtain ⟨ic, hic⟩ := innerCommitP_fw_ok (_root_.Flock.Firewall.Finish.padsOf q.draws)
    (_root_.Flock.Firewall.Finish.saltAt q.draws.salts)
  obtain ⟨⟨p, zl, cons⟩, tr, hrun, -, rp, t', lgc, rec, hrp, hlgc, hcons⟩ :=
    _root_.Flock.Firewall.Finish.repCons_asksOf q.coins h25 h35 hkm hkk hmk hfold hf hz t.tPad rep dom t.g t.cm0
      (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws) ic q.vals t.extra
      t.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)) hes hC hρ
  rw [hn] at hrun
  have hle : n ≤ q.coins.size := by
    have := hC.length_le
    rw [List.length_map, Array.length_toList (xs := q.coins), hn] at this
    exact this
  have hrep := repCons_asksPad q.coins q.coins (fun _ _ => rfl) t.st hmk t.sch t.tPad rep dom t.g t.cm0 _ _ ic
    q.vals t.extra t.lanePad _ _ hrun hle
  obtain ⟨rs, -, -, -, hrs, hkrs, hrcoins, -, -, hr1c, hfc⟩ := repCons_parts hrep
  dsimp only at hcons
  have hsz3 := FlockSoundness.Discharge.ZkExec.replay_cons_size t.st _ p ic.root hrp
  have hl2 := FlockComplete.Replay.lanes_eq hlgc rfl
  have h7 : cons.size = 7 := by
    rw [hcons, hl2, Array.size_append, Array.size_append, hsz3]
    rfl
  refine finInner_ok_of hic hrep ?_ (innerP_pad_ok q.coins _ _ ic _ cons (by rw [h7]; norm_num) tr h0
    (fun cs hc => (h1 cs hc).trans (by rw [h7]; rfl)) h2 h3 hβ)
  have hzc := FlockSoundness.Discharge.Composed.ZerocheckMasked.replay_zerocheck t.st _ p ic.root (by omega) hrp
  obtain ⟨_, -, hlc, -, -⟩ := FlockSoundness.Discharge.Composed.Lincheck.replay_lincheck t.st _ p ic.root (by omega) hrp
  refine consWhy_none_of fun qq hq hname => ?_
  rw [hcons, hl2] at hq
  rcases Array.mem_append.1 hq with hq | hq
  · rcases Array.mem_append.1 hq with hq | hq
    · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hq
      have hi3 : i < 3 := hsz3 ▸ hi
      obtain rfl | rfl | rfl : i = 0 ∨ i = 1 ∨ i = 2 := by omega
      · -- the c_eval: the request's value at `final_c` is the c_eval of its round-1c values at the round-2 coin
        rw [← getElem!_pos rp.cons 0 hi, hzc.1]
        dsimp only
        have hMk : FlockSoundness.Discharge.Composed.ZerocheckMasked.Masks
            (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (_root_.Flock.Firewall.Finish.padsOf q.draws).h
            ⟨(Array.range 64).map fun i => p.zerocheck.round1ab[i]! +
                (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog
                  t.sch.k0).r1ab i]!,
              (Array.range 64).map fun i => q.vals[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).r1c i]!,
              p.zerocheck.rounds,
              p.zerocheck.finalA + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new
                t.st.m t.st.kLog t.sch.k0).fa]!,
              p.zerocheck.finalB + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new
                t.st.m t.st.kLog t.sch.k0).fb]!,
              q.vals[(_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0).fc]!⟩ p.zerocheck 0 :=
          ⟨fun i hi => by
              simp only [getElem!_map_range hi, FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self],
            fun i hi => by simp only [hr1c, getElem!_map_range hi],
            fun i hi => absurd hi (Nat.not_lt_zero i),
            (FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self _ _).symm,
            (FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self _ _).symm, hfc⟩
        have hz2 : FlockSoundness.Discharge.Composed.ZerocheckMasked.zAt
            (_root_.Flock.Transcript.init tr.rounds dom) = padCoin q.coins 2 0 := by
          have h2rs : 2 < rs.length := by omega
          obtain ⟨hpos, hco⟩ := hrcoins 2 h2rs
          have htr : tr.rounds = rs.toArray := by rw [← hrs, Array.toArray_toList]
          show (tr.rounds[2]!).coins.getD 0 0 = _
          rw [htr, getElem!_pos rs.toArray 2 (by simpa using h2rs), List.getElem_toArray, hco, getD_map_range hpos]
        rw [_root_.Flock.Firewall.Finish.linAt_eq,
          FlockSoundness.Discharge.Composed.ZerocheckMasked.c_eval_eval _ _ _ _ 0 hMk]
        dsimp only
        rw [hz2, ← hval]
        exact FlockSoundness.Discharge.Composed.ZerocheckMasked.add_self' _
      · rw [← getElem!_pos rp.cons 1 hi, hzc.2.1] at hname
        simp at hname
      · rw [← getElem!_pos rp.cons 2 hi, hlc] at hname
        simp at hname
    · simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl <;> simp at hname
  · simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl <;> simp at hname

end Loop

end Claims.ZKServer
