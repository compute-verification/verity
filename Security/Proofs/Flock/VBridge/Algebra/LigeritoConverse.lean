import Proofs.Flock.VBridge.Algebra.Ligerito
import Proofs.Flock.Soundness.Refine.LigeritoRun

/-!
# The verifier's Ligerito runs where V*'s does

The converse of `Refine.ligerito_refines`' premise: when V*'s Ligerito up to its final check (`ligeritoRun`) succeeds on
the rounds `c`, its final check passes (`FinalOK`), the proof carries V*'s messages, the Merkle check of every level
it queried passes and the levels' consistency sums are the openings' (`Opened`), then the verifier's `Flock.ligerito`
(Ligerito.lean:165–277) returns `.ok` on every record whose rounds are `c`'s and retained the bytes the verifier frames
before each coin (`Fits`, some bytes `F`).

* `RunsT`, `RunsL`: a run of a verifier step on every such record; they compose (`RunsT.bind`, `RunsL.bind`), and each
  of the transcript's and the Ligerito state's steps runs (`RunsT.squeeze`, `nextMsg_runs`, `queryLevel_runs`, …).
* `toL`: the verifier's Ligerito state at V*'s, read in K, with its nonce cursors at V*'s counts; `Good`: what the
  verifier needs of V*'s final state (nonces, and every level's queries pass), which holds of every earlier state (`Le`).
* One lemma per V* step with the verifier's run beside it: `challenge_conv`, `challenges_conv`, `oodIntro_conv`,
  `vquery`, `vqueryFold`; the level loop (`loop_runs`); `ligerito_of_run`, and `ligerito_refines_of_run`, which feeds
  it to `Refine.ligerito_refines`.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine Transcript

/-! ## Runs of the verifier on V*'s rounds -/

/-- The record's rounds `k, …, k' − 1` are V*'s rounds `c`, round `j` retaining the bytes `F j`. -/
def Fits (c : Array (Array F128)) (F : ℕ → ByteArray) (rs : Array Round) (k k' : ℕ) : Prop :=
  ∀ j, k ≤ j → j < k' → ∃ r, rs[j]? = some r ∧ r.n = (c[j]!).size ∧ r.coins = c[j]! ∧ r.bytes = some (F j)

/-- `y` runs from round `k` to round `k'`, returning `b`: from any pending bytes, on every record whose rounds in
between are V*'s rounds `c` and retained some bytes `F`, which depend on the pending bytes only. -/
def RunsT {β : Type} (c : Array (Array F128)) (y : TM β) (k k' : ℕ) (b : β) : Prop :=
  k ≤ k' ∧ ∀ P : ByteArray, ∃ (P' : ByteArray) (F : ℕ → ByteArray),
    ∀ rs, Fits c F rs k k' → y ⟨rs, k, P⟩ = .ok (b, ⟨rs, k', P'⟩)

variable {c : Array (Array F128)} {α β : Type}

theorem tbind_of {x : TM α} {f : α → TM β} {t t₁ : Transcript} {a : α} (hx : x t = .ok (a, t₁)) :
    (x >>= f) t = f a t₁ := by
  change (StateT.bind x f) t = _
  unfold StateT.bind
  rw [hx]
  rfl

theorem RunsT.bind {x : TM α} {f : α → TM β} {k k₁ k₂ : ℕ} {a : α} {b : β}
    (hx : RunsT c x k k₁ a) (hf : RunsT c (f a) k₁ k₂ b) : RunsT c (x >>= f) k k₂ b := by
  refine ⟨hx.1.trans hf.1, fun P => ?_⟩
  obtain ⟨P₁, F₁, h₁⟩ := hx.2 P
  obtain ⟨P₂, F₂, h₂⟩ := hf.2 P₁
  refine ⟨P₂, fun j => if j < k₁ then F₁ j else F₂ j, fun rs hrs => ?_⟩
  rw [tbind_of (h₁ rs fun j hj hj' => ?_)]
  · exact h₂ rs fun j hj hj' => by
      obtain ⟨r, h⟩ := hrs j (hx.1.trans hj) hj'
      simp only [show ¬ j < k₁ by omega, ite_false] at h
      exact ⟨r, h⟩
  · obtain ⟨r, h⟩ := hrs j hj (Nat.lt_of_lt_of_le hj' hf.1)
    simp only [hj', ite_true] at h
    exact ⟨r, h⟩

theorem RunsT.pure {k : ℕ} (b : β) : RunsT c (Pure.pure b : TM β) k k b :=
  ⟨le_rfl, fun P => ⟨P, fun _ => ByteArray.empty, fun _ _ => rfl⟩⟩

theorem RunsT.check {k : ℕ} {b : Bool} {why : String} (hb : b = true) : RunsT c (Flock.check b why) k k () := by
  subst hb
  exact RunsT.pure ()

theorem RunsT.label {k : ℕ} (l : String) : RunsT c (Transcript.label l) k k () :=
  ⟨le_rfl, fun P => ⟨P ++ header 2 0 l.toUTF8.size ++ padded l.toUTF8, fun _ => ByteArray.empty, fun _ _ => rfl⟩⟩

theorem RunsT.absorbF {k : ℕ} (v : F128) : RunsT c (Transcript.absorbF v) k k () :=
  ⟨le_rfl, fun P => ⟨F128.pushBytes (P ++ header 3 1 1) v, fun _ => ByteArray.empty, fun _ _ => rfl⟩⟩

theorem RunsT.absorbK {k : ℕ} (v : F256) : RunsT c (Transcript.absorbK v) k k () :=
  ⟨le_rfl, fun P => ⟨#[v.c0, v.c1].foldl F128.pushBytes (P ++ header 3 2 #[v.c0, v.c1].size),
    fun _ => ByteArray.empty, fun _ _ => rfl⟩⟩

theorem RunsT.absorbBytes {k : ℕ} (b : ByteArray) : RunsT c (Transcript.absorbBytes b) k k () :=
  ⟨le_rfl, fun P => ⟨P ++ header 5 0 b.size ++ padded b, fun _ => ByteArray.empty, fun _ _ => rfl⟩⟩

theorem RunsT.pow {k : ℕ} (n : UInt64) : RunsT c (Transcript.pow n) k k () :=
  RunsT.absorbBytes (le64 n)

/-- A squeeze of `n ≠ 0` coins at round `k` returns V*'s round `k`, when it has `n` coins: the round retained the
pending bytes and the squeeze's header. -/
theorem RunsT.squeeze {kind : UInt8} {n k : ℕ} (hn : n ≠ 0) (hc : (c[k]!).size = n) :
    RunsT c (Transcript.squeeze kind n) k (k + 1) (c[k]!) := by
  refine ⟨Nat.le_succ k, fun P => ⟨ByteArray.empty, fun _ => P ++ header 4 kind n, fun rs hrs => ?_⟩⟩
  obtain ⟨r, hr, hrn, hrc, hrb⟩ := hrs k le_rfl (Nat.lt_succ_self k)
  unfold Transcript.squeeze
  dsimp only
  rw [ite_eq_right hn]
  simp only [hr]
  rw [ite_eq_right (by rw [hrn, hc]; exact not_not.mpr rfl)]
  rw [ite_eq_right (by unfold Round.answers; rw [hrb]; exact not_not.mpr rfl)]
  rw [hrc]

theorem RunsT.sampleVec {n k : ℕ} (hn : n ≠ 0) (hc : (c[k]!).size = n) :
    RunsT c (Transcript.sampleVec n) k (k + 1) (c[k]!) :=
  RunsT.squeeze hn hc

theorem RunsT.sample {k : ℕ} (hc : (c[k]!).size = 1) :
    RunsT c Transcript.sample k (k + 1) ((c[k]!).getD 0 0) :=
  RunsT.bind (RunsT.squeeze (by decide) hc) (RunsT.pure _)

theorem RunsT.sampleK {k : ℕ} (hc : (c[k]!).size = 2) :
    RunsT c Transcript.sampleK k (k + 1) (⟨(c[k]!).getD 0 0, (c[k]!).getD 1 0⟩ : F256) :=
  RunsT.bind (RunsT.squeeze (by decide) hc) (RunsT.pure _)

/-- A run of a step of the verifier's Ligerito, from its state `st` to `st'`. -/
def RunsL {β : Type} (c : Array (Array F128)) (y : Flock.LM β) (st : LState) (k k' : ℕ) (b : β) (st' : LState) : Prop :=
  RunsT c (y st) k k' (b, st')

theorem RunsL.bind {x : Flock.LM α} {f : α → Flock.LM β} {st st₁ st₂ : LState} {k k₁ k₂ : ℕ} {a : α} {b : β}
    (hx : RunsL c x st k k₁ a st₁) (hf : RunsL c (f a) st₁ k₁ k₂ b st₂) : RunsL c (x >>= f) st k k₂ b st₂ :=
  RunsT.bind (f := fun p : α × LState => f p.1 p.2) hx hf

theorem RunsL.pure {st : LState} {k : ℕ} (b : β) : RunsL c (Pure.pure b : Flock.LM β) st k k b st :=
  RunsT.pure (b, st)

theorem RunsL.lift {x : TM α} {st : LState} {k k' : ℕ} {a : α} (hx : RunsT c x k k' a) :
    RunsL c (Flock.lift x) st k k' a st :=
  RunsT.bind (f := fun a => Pure.pure (a, st)) hx (RunsT.pure _)

theorem RunsL.get' {st : LState} {k : ℕ} : RunsL c (MonadState.get : Flock.LM LState) st k k st st :=
  RunsT.pure (st, st)

theorem RunsL.set' {st s : LState} {k : ℕ} : RunsL c (MonadStateOf.set s : Flock.LM PUnit) st k k () s :=
  RunsT.pure ((), s)

theorem RunsL.modify' {st : LState} {k : ℕ} (g : LState → LState) :
    RunsL c (modify g : Flock.LM PUnit) st k k () (g st) :=
  RunsT.pure ((), g st)

theorem RunsL.get_bind {f : LState → Flock.LM β} {st st' : LState} {k k' : ℕ} {b : β}
    (h : RunsL c (f st) st k k' b st') : RunsL c ((MonadState.get : Flock.LM LState) >>= f) st k k' b st' :=
  RunsL.bind RunsL.get' h

theorem RunsL.cast {y : Flock.LM β} {st st₁ st₂ : LState} {k k₁ k₂ : ℕ} {b b' : β}
    (h : RunsL c y st k k₁ b st₁) (hk : k₁ = k₂) (hb : b = b') (hs : st₁ = st₂) : RunsL c y st k k₂ b' st₂ := by
  subst hk hb hs
  exact h

/-! ## The verifier's Ligerito steps -/

theorem nextMsg_runs {p : LigeritoProof} {vs : LState} {k : ℕ} {m : F256 × F256}
    (hm : p.messages[vs.msg]? = some m) : RunsL c (Flock.nextMsg p) vs k k m { vs with msg := vs.msg + 1 } := by
  unfold Flock.nextMsg
  refine RunsL.get_bind ?_
  simp only [hm]
  exact RunsL.bind RunsL.set' (RunsL.bind (RunsL.lift (RunsT.absorbK _))
    (RunsL.bind (RunsL.lift (RunsT.absorbK _)) (RunsL.pure _)))

theorem nextOod_runs {p : LigeritoProof} {vs : LState} {k : ℕ} {v : F128}
    (hv : p.oodValues[vs.ood]? = some v) : RunsL c (Flock.nextOod p) vs k k v { vs with ood := vs.ood + 1 } := by
  unfold Flock.nextOod
  refine RunsL.get_bind ?_
  simp only [hv]
  exact RunsL.bind RunsL.set' (RunsL.pure _)

theorem nonceFrom_runs {xs : Array UInt64} {i : ℕ} {what : String} {vs : LState} {k : ℕ} (h : i < xs.size) :
    RunsL c (nonceFrom xs i what) vs k k xs[i] vs := by
  unfold nonceFrom
  simp only [getElem?_pos xs i h]
  exact RunsL.pure _

theorem claimPow_runs {p : LigeritoProof} {vs : LState} {k : ℕ} (h : vs.cbn < p.claimNonces.size) :
    RunsL c (claimPow p) vs k k () { vs with cbn := vs.cbn + 1 } := by
  unfold claimPow
  refine RunsL.get_bind ?_
  exact RunsL.bind (nonceFrom_runs h) (RunsL.bind RunsL.set' (RunsL.lift (RunsT.pow _)))

theorem challenge_runs {p : LigeritoProof} {vs : LState} {k : ℕ} {m : F256 × F256} (hc : (c[k]!).size = 2)
    (hm : p.messages[vs.msg]? = some m) :
    RunsL c (Flock.challenge p) vs k (k + 1) (⟨(c[k]!).getD 0 0, (c[k]!).getD 1 0⟩ : F256)
      { vs with s := vs.q.eval ⟨(c[k]!).getD 0 0, (c[k]!).getD 1 0⟩,
                q := Poly.ofMsg m (vs.q.eval ⟨(c[k]!).getD 0 0, (c[k]!).getD 1 0⟩), msg := vs.msg + 1 } := by
  unfold Flock.challenge
  refine RunsL.bind (RunsL.lift (RunsT.sampleK hc)) ?_
  refine RunsL.bind (RunsL.modify' _) ?_
  refine RunsL.bind (nextMsg_runs hm) ?_
  refine RunsL.bind (RunsL.modify' _) ?_
  exact RunsL.pure _

theorem oodIntro_runs {p : LigeritoProof} {dim split : ℕ} {vs : LState} {k : ℕ} {y : F128} {m : F256 × F256}
    (hd : dim ≠ 0) (hz : (c[k]!).size = dim) (hb : (c[k + 1]!).size = 1) (hy : p.oodValues[vs.ood]? = some y)
    (hm : p.messages[vs.msg]? = some m) (hcb : vs.cbn < p.claimNonces.size) :
    RunsL c (Flock.oodIntro p dim split) vs k (k + 1 + 1) ()
      { vs with s := vs.s + (F256.ofF y).smul ((c[k + 1]!).getD 0 0),
                q := vs.q.addScaled (Poly.ofMsg m (F256.ofF y)) ((c[k + 1]!).getD 0 0),
                msg := vs.msg + 1, ood := vs.ood + 1, cbn := vs.cbn + 1,
                oods := vs.oods.push ⟨c[k]!, (c[k + 1]!).getD 0 0, some split⟩ } := by
  unfold Flock.oodIntro
  refine RunsL.bind (RunsL.lift (RunsT.sampleVec hd hz)) ?_
  refine RunsL.bind (nextOod_runs hy) ?_
  refine RunsL.bind (RunsL.lift (RunsT.absorbF _)) ?_
  refine RunsL.bind (nextMsg_runs hm) ?_
  refine RunsL.bind (claimPow_runs hcb) ?_
  refine RunsL.bind (RunsL.lift (RunsT.sample hb)) ?_
  exact RunsL.modify' _

theorem queryLevel_runs {ms : MerkleScheme} {p : LigeritoProof} {lv : Flock.Level} {cap : Array ByteArray}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {fold : Array F256} {vs : LState} {k : ℕ}
    (hq : lv.q ≠ 0) (hcq : (c[k]!).size = lv.q) (hca : (c[k + 1]!).size = Nat.log2 (lv.q - 1) + 1)
    (hqn : vs.qn < p.queryNonces.size) (hcn : vs.cn < p.consistencyNonces.size)
    (hmc : merkleCheck ms cap lv.cap lv.d (positions lv c[k]!) rows sibs salts (2 ^ lv.lanes) = .ok ()) :
    RunsL c (queryLevel ms p lv cap rows sibs salts fold) vs k (k + 1 + 1)
      (positions lv c[k]!, c[k + 1]!, consE rows lv.q c[k + 1]! fold) { vs with qn := vs.qn + 1, cn := vs.cn + 1 } := by
  unfold queryLevel
  refine RunsL.get_bind ?_
  refine RunsL.bind (nonceFrom_runs hqn) ?_
  refine RunsL.bind (nonceFrom_runs hcn) ?_
  refine RunsL.bind RunsL.set' ?_
  refine RunsL.bind (RunsL.lift (RunsT.pow _)) ?_
  refine RunsL.bind (RunsL.lift (RunsT.sampleVec hq hcq)) ?_
  refine RunsL.bind (RunsL.lift (RunsT.pow _)) ?_
  refine RunsL.bind (RunsL.lift (RunsT.sampleVec (by omega) hca)) ?_
  simp only [hmc, List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.forIn_eq_forIn_range', pure_bind, range_size,
    ← List.range_eq_range']
  exact RunsL.pure _

theorem forIn_absorbF_runs : ∀ (ws : List F128) (vs : LState) (k : ℕ),
    RunsL c (forIn ws PUnit.unit (fun w (_ : PUnit) => Flock.lift (absorbF w) >>= fun _ =>
      pure (ForInStep.yield PUnit.unit))) vs k k PUnit.unit vs
  | [], vs, k => RunsL.pure _
  | w :: ws, vs, k => by
    rw [List.forIn_cons]
    exact RunsL.bind (RunsL.bind (RunsL.lift (RunsT.absorbF w)) (RunsL.pure _)) (forIn_absorbF_runs ws vs k)

/-! ## V*'s Ligerito steps -/

section VStar

variable {A : Type} {src : CoinSource}

theorem vbind_ok {x : RecAlgebra.LM F128 α} {f : α → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ}
    {r : (β × LSt F128) × ℕ} (h : (x >>= f) st k src = .ok r) :
    ∃ a st₁ k₁, x st k src = .ok ((a, st₁), k₁) ∧ f a st₁ k₁ src = .ok r := by
  have e : (x >>= f) st k src = (x st >>= fun p => f p.1 p.2) k src := rfl
  rw [e] at h
  obtain ⟨⟨a, st₁⟩, k₁, h₁, h₂⟩ := cm_bind h
  exact ⟨a, st₁, k₁, h₁, h₂⟩

theorem vlift_bind {x : CM α} {f : α → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ} {r : (β × LSt F128) × ℕ}
    (h : ((StateT.lift x : RecAlgebra.LM F128 α) >>= f) st k src = .ok r) :
    ∃ a k₁, x k src = .ok (a, k₁) ∧ f a st k₁ src = .ok r := by
  obtain ⟨a, st₁, k₁, h₁, h₂⟩ := vbind_ok h
  have e : (StateT.lift x : RecAlgebra.LM F128 α) st k src = (x >>= fun a => pure (a, st)) k src := rfl
  rw [e] at h₁
  obtain ⟨a', k₂, h₃, h₄⟩ := cm_bind h₁
  cases cm_pure h₄
  exact ⟨_, _, h₃, h₂⟩

theorem vget_bind {f : LSt F128 → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ} {r : (β × LSt F128) × ℕ}
    (h : ((get : RecAlgebra.LM F128 (LSt F128)) >>= f) st k src = .ok r) : f st st k src = .ok r := by
  obtain ⟨a, st₁, k₁, h₁, h₂⟩ := vbind_ok h
  cases h₁
  exact h₂

theorem vset_bind {s : LSt F128} {f : PUnit → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ} {r : (β × LSt F128) × ℕ}
    (h : ((set s : RecAlgebra.LM F128 PUnit) >>= f) st k src = .ok r) : f () s k src = .ok r := by
  obtain ⟨a, st₁, k₁, h₁, h₂⟩ := vbind_ok h
  cases h₁
  exact h₂

theorem vmodify_bind {g : LSt F128 → LSt F128} {f : PUnit → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ}
    {r : (β × LSt F128) × ℕ} (h : ((modify g : RecAlgebra.LM F128 PUnit) >>= f) st k src = .ok r) :
    f () (g st) k src = .ok r := by
  obtain ⟨a, st₁, k₁, h₁, h₂⟩ := vbind_ok h
  cases h₁
  exact h₂

theorem vmodify_ok {g : LSt F128 → LSt F128} {st st' : LSt F128} {k k' : ℕ} {u : PUnit}
    (h : (modify g : RecAlgebra.LM F128 PUnit) st k src = .ok ((u, st'), k')) : st' = g st ∧ k' = k := by
  cases h
  exact ⟨rfl, rfl⟩

theorem vpure {a a' : α} {st st' : LSt F128} {k k' : ℕ}
    (h : (pure a : RecAlgebra.LM F128 α) st k src = .ok ((a', st'), k')) : a' = a ∧ st' = st ∧ k' = k := by
  cases h
  exact ⟨rfl, rfl, rfl⟩

theorem vnextMsg_bind {M : Messages F128} {f : KV F128 × KV F128 → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ}
    {r : (β × LSt F128) × ℕ} (h : (RecAlgebra.nextMsg M >>= f) st k src = .ok r) :
    ∃ mm, M.msgs[st.msg]? = some mm ∧ f mm { st with msg := st.msg + 1 } k src = .ok r := by
  obtain ⟨a, st₁, k₁, h₁, h₂⟩ := vbind_ok h
  unfold RecAlgebra.nextMsg at h₁
  have h₁ := vget_bind h₁
  cases hm : M.msgs[st.msg]? with
  | none => simp only [hm] at h₁; cases h₁
  | some mm =>
    simp only [hm] at h₁
    have h₁ := vset_bind h₁
    cases h₁
    exact ⟨_, rfl, h₂⟩

theorem vnextOod_bind {M : Messages F128} {f : F128 → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ}
    {r : (β × LSt F128) × ℕ} (h : (RecAlgebra.nextOod M >>= f) st k src = .ok r) :
    ∃ y, M.ood[st.ood]? = some y ∧ f y { st with ood := st.ood + 1 } k src = .ok r := by
  obtain ⟨a, st₁, k₁, h₁, h₂⟩ := vbind_ok h
  unfold RecAlgebra.nextOod at h₁
  have h₁ := vget_bind h₁
  cases hy : M.ood[st.ood]? with
  | none => simp only [hy] at h₁; cases h₁
  | some y =>
    simp only [hy] at h₁
    have h₁ := vset_bind h₁
    cases h₁
    exact ⟨_, rfl, h₂⟩

end VStar

-- V*'s rounds hold no round of no coins, so `CoinSource.ofRounds` reads them as recorded.
variable [hne : Fact (NoEmpty c)]

/-- A round of the record that V* drew `n` coins from. -/
theorem ofRounds_round {k n : ℕ} {v : Array F128} (h : (CoinSource.ofRounds c).round k n = .ok v) :
    c[k]! = v ∧ v.size = n := by
  simp only [CoinSource.ofRounds, filter_noEmpty hne.out] at h
  cases hc : c[k]? with
  | none => rw [hc] at h; cases h
  | some r =>
    rw [hc] at h
    dsimp only at h
    split at h
    · rename_i hr
      cases h
      exact ⟨getElem!_of_getElem? hc, hr⟩
    · cases h

theorem vcoinVec {n k k' : ℕ} {v : Array F128} (hn : n ≠ 0) (h : coinVec n k (CoinSource.ofRounds c) = .ok (v, k')) :
    v = c[k]! ∧ (c[k]!).size = n ∧ k' = k + 1 := by
  rcases coinVec_ok h with ⟨h0, -, -⟩ | ⟨-, hr, rfl⟩
  · exact absurd h0 hn
  · obtain ⟨e, hs⟩ := ofRounds_round hr
    subst e
    exact ⟨rfl, hs, rfl⟩

theorem vcoinVec_bind {n : ℕ} {f : Array F128 → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ}
    {r : (β × LSt F128) × ℕ} (hn : n ≠ 0)
    (h : ((StateT.lift (coinVec n) : RecAlgebra.LM F128 (Array F128)) >>= f) st k (CoinSource.ofRounds c) = .ok r) :
    (c[k]!).size = n ∧ f (c[k]!) st (k + 1) (CoinSource.ofRounds c) = .ok r := by
  obtain ⟨v, k₁, h₁, h₂⟩ := vlift_bind h
  obtain ⟨rfl, hs, rfl⟩ := vcoinVec hn h₁
  exact ⟨hs, h₂⟩

theorem vcoinOne_bind {f : F128 → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ} {r : (β × LSt F128) × ℕ}
    (h : ((StateT.lift coinOne : RecAlgebra.LM F128 F128) >>= f) st k (CoinSource.ofRounds c) = .ok r) :
    (c[k]!).size = 1 ∧ f ((c[k]!).getD 0 0) st (k + 1) (CoinSource.ofRounds c) = .ok r := by
  obtain ⟨x, k₁, h₁, h₂⟩ := vlift_bind h
  obtain ⟨v, hr, rfl, rfl⟩ := coinOne_ok h₁
  obtain ⟨rfl, hs⟩ := ofRounds_round hr
  rw [getElem!_eq_getD] at h₂
  exact ⟨hs, h₂⟩

theorem vcoinK2_bind {f : F256 → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ} {r : (β × LSt F128) × ℕ}
    (h : ((StateT.lift coinK2 : RecAlgebra.LM F128 F256) >>= f) st k (CoinSource.ofRounds c) = .ok r) :
    (c[k]!).size = 2 ∧ f ⟨(c[k]!).getD 0 0, (c[k]!).getD 1 0⟩ st (k + 1) (CoinSource.ofRounds c) = .ok r := by
  obtain ⟨x, k₁, h₁, h₂⟩ := vlift_bind h
  obtain ⟨v, k₂, h₃, h₄⟩ := cm_bind h₁
  cases cm_pure h₄
  obtain ⟨rfl, hs, rfl⟩ := vcoinVec (by decide) h₃
  rw [getElem!_eq_getD, getElem!_eq_getD] at h₂
  exact ⟨hs, h₂⟩

omit hne in
theorem vthrow {e : String} {f : α → RecAlgebra.LM F128 β} {st : LSt F128} {k : ℕ} {r : (β × LSt F128) × ℕ}
    (h : ((throw e : RecAlgebra.LM F128 α) >>= f) st k (CoinSource.ofRounds c) = .ok r) : False := by
  cases h

/-! ## V*'s state as the verifier's -/

theorem getElem!_getD {γ : Type} [Inhabited γ] (a : Array γ) (i : ℕ) : a[i]! = a.getD i default := by
  rw [Array.getD_eq_getD_getElem?, getElem!_def]
  cases a[i]? <;> rfl

theorem eqTableK_size' (r : Array F256) : (eqTableK r).size = 2 ^ r.size := by
  rw [eqTableK_foldl, ← Array.foldl_toList, foldl_eqStepK_size]
  simp

/-- The verifier's Ligerito state at V*'s (Ligerito.lean:74–84): the running sum and round polynomial read in K, V*'s
cursors and contexts, a query and a consistency nonce read per consistency context (Ligerito.lean:140–142), and a claim
nonce per OOD value and per consistency context (Ligerito.lean:112–116, 180, 194, 219, 231). -/
def toL (st : LSt F128) : LState :=
  ⟨kv st.s, polyOf st.q, st.msg, st.ood, st.conss.size, st.conss.size, st.ood + st.conss.size, st.oods, st.conss⟩

theorem toL_eq {vs : LState} {st : LSt F128} (hs : vs.s = kv st.s) (hq : vs.q = polyOf st.q) (hm : vs.msg = st.msg)
    (ho : vs.ood = st.ood) (hqn : vs.qn = st.conss.size) (hcn : vs.cn = st.conss.size)
    (hcb : vs.cbn = st.ood + st.conss.size) (hoo : vs.oods = st.oods) (hc : vs.conss = st.conss) : vs = toL st := by
  obtain ⟨⟩ := vs
  simp only at hs hq hm ho hqn hcn hcb hoo hc
  subst hs hq hm ho hqn hcn hcb hoo hc
  rfl

theorem msg_of {M : Messages F128} {p : LigeritoProof} (hp : p.messages = M.msgs.map fun mm => (kv mm.1, kv mm.2))
    {i : ℕ} {mm : KV F128 × KV F128} (h : M.msgs[i]? = some mm) : p.messages[i]? = some (kv mm.1, kv mm.2) := by
  rw [hp, Array.getElem?_map, h]
  rfl

/-- V*'s state grows: OOD values are read and consistency contexts appended. -/
def Le (st st' : LSt F128) : Prop := st.ood ≤ st'.ood ∧ ∃ more : Array Flock.ConsCtx, st'.conss = st.conss ++ more

theorem Le.refl (st : LSt F128) : Le st st := ⟨le_rfl, #[], by simp⟩

theorem Le.trans {a b d : LSt F128} (h₁ : Le a b) (h₂ : Le b d) : Le a d := by
  obtain ⟨o₁, m₁, e₁⟩ := h₁
  obtain ⟨o₂, m₂, e₂⟩ := h₂
  exact ⟨o₁.trans o₂, m₁ ++ m₂, by rw [e₂, e₁, Array.append_assoc]⟩

theorem Le.of {st st' : LSt F128} (ho : st.ood ≤ st'.ood) (hc : st'.conss = st.conss) : Le st st' :=
  ⟨ho, #[], by simp [hc]⟩

theorem Le.push {st st' : LSt F128} {x : Flock.ConsCtx} (ho : st.ood ≤ st'.ood) (hc : st'.conss = st.conss.push x) :
    Le st st' :=
  ⟨ho, #[x], by rw [hc]; apply Array.ext' ; simp⟩

/-- Level `cs.start`'s queries pass (Ligerito.lean:148–158): the rows the proof opens there verify against the level's
cap at the positions `cs.pos`, and the messages' consistency sums at `cs.alpha` are the rows'. -/
def Opened (ms : MerkleScheme) (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof) (M : Messages F128)
    (cs : Flock.ConsCtx) : Prop :=
  merkleCheck ms (capAt cap p cs.start) (sch.levels.getD cs.start default).cap (sch.levels.getD cs.start default).d
      cs.pos (openAt p (sch.levels.size - 1) cs.start).1 (openAt p (sch.levels.size - 1) cs.start).2.1
      (openAt p (sch.levels.size - 1) cs.start).2.2 (2 ^ (sch.levels.getD cs.start default).lanes) = .ok () ∧
    M.cons cs.start ((eqTable cs.alpha).extract 0 (sch.levels.getD cs.start default).q) =
      (Array.range (2 ^ (sch.levels.getD cs.start default).lanes)).map
        (consAt (openAt p (sch.levels.size - 1) cs.start).1 (sch.levels.getD cs.start default).q cs.alpha)

/-- What the verifier needs of V*'s final state: nonces for every query and claim, and every level's queries pass. -/
structure Good (ms : MerkleScheme) (sch : Flock.Schedule) (cap : Array ByteArray) (p : LigeritoProof)
    (M : Messages F128) (st : LSt F128) : Prop where
  qn : st.conss.size ≤ p.queryNonces.size
  cn : st.conss.size ≤ p.consistencyNonces.size
  cbn : st.ood + st.conss.size ≤ p.claimNonces.size
  opened : ∀ cs ∈ st.conss.toList, Opened ms sch cap p M cs

theorem Good.mono {ms : MerkleScheme} {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof}
    {M : Messages F128} {st st' : LSt F128} (hle : Le st st') (h : Good ms sch cap p M st') : Good ms sch cap p M st := by
  obtain ⟨ho, more, he⟩ := hle
  have hs : st'.conss.size = st.conss.size + more.size := by rw [he, Array.size_append]
  have h1 := h.qn
  have h2 := h.cn
  have h3 := h.cbn
  exact ⟨by omega, by omega, by omega,
    fun cs hcs => h.opened cs (by rw [he, Array.toList_append]; exact List.mem_append_left _ hcs)⟩

theorem Good.opened_push {ms : MerkleScheme} {sch : Flock.Schedule} {cap : Array ByteArray} {p : LigeritoProof}
    {M : Messages F128} {st : LSt F128} (g : Good ms sch cap p M st) {xs : Array Flock.ConsCtx} {x : Flock.ConsCtx}
    (hc : st.conss = xs.push x) : Opened ms sch cap p M x :=
  g.opened x (by rw [hc]; simp)

/-- V*'s enforced sum of a level's queries (`query`, Structure.lean:394–400). -/
def qE (M : Messages F128) (i : ℕ) (lv : Flock.Level) (fold : Array F256) (al : Array F128) : KV F128 :=
  kaddAll (((eqTableK fold).zip (M.cons i ((eqTable al).extract 0 lv.q))).toList.map fun fa => kf fa.1 fa.2)

/-- V*'s enforced sum is the verifier's (Ligerito.lean:151–158) when the level's queries pass. -/
theorem query_e {M : Messages F128} {i : ℕ} {lv : Flock.Level} {fold : Array F256} {al : Array F128}
    {ms : MerkleScheme} {capq : Array ByteArray} {pos : Array ℕ} {rows : Array (Array F128)}
    {sibs salts : Array ByteArray} (hfold : fold.size = lv.lanes)
    (hmc : merkleCheck ms capq lv.cap lv.d pos rows sibs salts (2 ^ lv.lanes) = .ok ()) (hpos : pos.size = lv.q)
    (hcons : M.cons i ((eqTable al).extract 0 lv.q) = (Array.range (2 ^ lv.lanes)).map (consAt rows lv.q al)) :
    kv (qE M i lv fold al) = consE rows lv.q al fold := by
  obtain ⟨-, hrl⟩ := merkleCheck_rows hmc
  unfold qE
  rw [hcons]
  exact kv_e rows lv.q al fold (2 ^ lv.lanes) (by rw [eqTableK_size', hfold]) (fun j hj => hrl j (by omega)) _
    (fun _ _ => rfl)

/-! ## V*'s Ligerito steps, and the verifier's beside them -/

section Conv

variable {M : Messages F128} {p : LigeritoProof}

theorem challenge_conv {st st' : LSt F128} {k k' : ℕ} {rk : F256}
    (h : RecAlgebra.challenge M st k (CoinSource.ofRounds c) = .ok ((rk, st'), k')) :
    k' = k + 1 ∧ st'.msg = st.msg + 1 ∧ st'.ood = st.ood ∧ st'.oods = st.oods ∧ st'.conss = st.conss ∧
      (p.messages = M.msgs.map (fun mm => (kv mm.1, kv mm.2)) →
        RunsL c (Flock.challenge p) (toL st) k k' rk (toL st')) := by
  obtain ⟨s, ⟨a0, a1, a2⟩, msg, ood, oods, conss⟩ := st
  unfold RecAlgebra.challenge at h
  obtain ⟨hc, h⟩ := vcoinK2_bind h
  have h := vget_bind h
  dsimp only at h
  have h := vset_bind h
  obtain ⟨mm, hmm, h⟩ := vnextMsg_bind h
  have h := vmodify_bind h
  obtain ⟨rfl, rfl, rfl⟩ := vpure h
  refine ⟨rfl, rfl, rfl, rfl, rfl, fun hp => ?_⟩
  refine RunsL.cast (challenge_runs hc (msg_of hp hmm)) rfl rfl ?_
  apply toL_eq <;> dsimp only [toL] <;> simp only [kv_challenge, polyOf_ofMsg]

theorem challenges_conv : ∀ (n j : ℕ) (acc : Array F256) (st st' : LSt F128) (k k' : ℕ) (chs : Array F256),
    RecAlgebra.challenges M n st k (CoinSource.ofRounds c) = .ok ((chs, st'), k') →
    k' = k + n ∧ chs.size = n ∧ st'.msg = st.msg + n ∧ st'.ood = st.ood ∧ st'.oods = st.oods ∧
      st'.conss = st.conss ∧
      (p.messages = M.msgs.map (fun mm => (kv mm.1, kv mm.2)) →
        RunsL c (forIn (List.range' j n) acc fun _ chs =>
          Flock.challenge p >>= fun x => pure (ForInStep.yield (chs.push x))) (toL st) k k' (acc ++ chs) (toL st'))
  | 0, j, acc, st, st', k, k', chs, h => by
    simp only [RecAlgebra.challenges] at h
    obtain ⟨rfl, rfl, rfl⟩ := vpure h
    refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, fun _ => ?_⟩
    simp only [List.range'_zero, List.forIn_nil, Array.append_empty]
    exact RunsL.pure _
  | n + 1, j, acc, st, st', k, k', chs, h => by
    simp only [RecAlgebra.challenges] at h
    obtain ⟨rk, st₁, k₁, h₁, h⟩ := vbind_ok h
    obtain ⟨rest, st₂, k₂, h₂, h⟩ := vbind_ok h
    obtain ⟨rfl, rfl, rfl⟩ := vpure h
    obtain ⟨rfl, hm₁, ho₁, hoo₁, hc₁, hr₁⟩ := challenge_conv (p := p) h₁
    obtain ⟨rfl, hs₂, hm₂, ho₂, hoo₂, hc₂, hr₂⟩ := challenges_conv n (j + 1) (acc.push rk) _ _ _ _ rest h₂
    refine ⟨by omega, by simp [hs₂]; omega, by omega, ho₂.trans ho₁, hoo₂.trans hoo₁, hc₂.trans hc₁, fun hp => ?_⟩
    rw [List.range'_succ, List.forIn_cons]
    refine RunsL.bind (RunsL.bind (hr₁ hp) (RunsL.pure _)) (RunsL.cast (hr₂ hp) rfl ?_ rfl)
    simp

theorem oodIntro_conv {dim split : ℕ} {st st' : LSt F128} {k k' : ℕ} {u : PUnit} (hd : dim ≠ 0)
    (h : RecAlgebra.oodIntro M dim split st k (CoinSource.ofRounds c) = .ok ((u, st'), k')) :
    k' = k + 1 + 1 ∧ st'.msg = st.msg + 1 ∧ st'.ood = st.ood + 1 ∧ st'.conss = st.conss ∧
      (p.messages = M.msgs.map (fun mm => (kv mm.1, kv mm.2)) → p.oodValues = M.ood →
        st.ood + st.conss.size < p.claimNonces.size →
        RunsL c (Flock.oodIntro p dim split) (toL st) k k' () (toL st')) := by
  unfold RecAlgebra.oodIntro at h
  obtain ⟨hz, h⟩ := vcoinVec_bind hd h
  obtain ⟨y, hy, h⟩ := vnextOod_bind h
  obtain ⟨mm, hmm, h⟩ := vnextMsg_bind h
  obtain ⟨hb, h⟩ := vcoinOne_bind h
  unfold RecAlgebra.addScaled at h
  obtain ⟨rfl, rfl⟩ := vmodify_ok (vmodify_bind h)
  refine ⟨rfl, rfl, rfl, rfl, fun hp hpo hcb => ?_⟩
  refine RunsL.cast (oodIntro_runs (vs := toL st) hd hz hb (by rw [hpo]; exact hy) (msg_of hp hmm) hcb) rfl rfl ?_
  apply toL_eq <;> dsimp only [toL]
  · rw [kv_ood]
  · rw [polyOf_addScaled, polyOf_ofMsg]
    rfl
  · omega

theorem vquery {sch : Flock.Schedule} {i : ℕ} {fold : Array F256} {st st' : LSt F128} {k k' : ℕ}
    {r : Array ℕ × Array F128 × KV F128} (hq : (sch.levels.getD i default).q ≠ 0)
    (h : RecAlgebra.query sch M i fold st k (CoinSource.ofRounds c) = .ok ((r, st'), k')) :
    st' = st ∧ k' = k + 1 + 1 ∧ (c[k]!).size = (sch.levels.getD i default).q ∧
      (c[k + 1]!).size = Nat.log2 ((sch.levels.getD i default).q - 1) + 1 ∧
      r = (positions (sch.levels.getD i default) c[k]!, c[k + 1]!,
        qE M i (sch.levels.getD i default) fold c[k + 1]!) := by
  unfold RecAlgebra.query at h
  rw [getElem!_getD] at h
  obtain ⟨hz, h⟩ := vcoinVec_bind hq h
  obtain ⟨ha, h⟩ := vcoinVec_bind (by omega) h
  obtain ⟨rfl, rfl, rfl⟩ := vpure h
  exact ⟨rfl, rfl, hz, ha, rfl⟩

theorem vqueryFold {sch : Flock.Schedule} {i cols : ℕ} {fold : Array F256} {st st' : LSt F128} {k k' : ℕ}
    {u : PUnit} (hq : (sch.levels.getD i default).q ≠ 0)
    (h : RecAlgebra.queryFold sch M i cols fold st k (CoinSource.ofRounds c) = .ok ((u, st'), k')) :
    k' = k + 1 + 1 + 1 ∧ (c[k]!).size = (sch.levels.getD i default).q ∧
      (c[k + 1]!).size = Nat.log2 ((sch.levels.getD i default).q - 1) + 1 ∧ (c[k + 1 + 1]!).size = 1 ∧
      ∃ mm, M.msgs[st.msg]? = some mm ∧ st'.msg = st.msg + 1 ∧ st'.ood = st.ood ∧ st'.oods = st.oods ∧
        st'.conss = st.conss.push
          ⟨cols, positions (sch.levels.getD i default) c[k]!, c[k + 1]!, (c[k + 1 + 1]!).getD 0 0, i⟩ ∧
        kv st'.s = kv st.s + (kv (qE M i (sch.levels.getD i default) fold c[k + 1]!)).smul ((c[k + 1 + 1]!).getD 0 0) ∧
        polyOf st'.q = (polyOf st.q).addScaled
          (Poly.ofMsg (kv mm.1, kv mm.2) (kv (qE M i (sch.levels.getD i default) fold c[k + 1]!)))
          ((c[k + 1 + 1]!).getD 0 0) := by
  unfold RecAlgebra.queryFold at h
  obtain ⟨r, st₁, k₁, h₁, h⟩ := vbind_ok h
  obtain ⟨rfl, rfl, hz, ha, rfl⟩ := vquery hq h₁
  dsimp only at h
  obtain ⟨mm, hmm, h⟩ := vnextMsg_bind h
  obtain ⟨hb, h⟩ := vcoinOne_bind h
  unfold RecAlgebra.addScaled at h
  obtain ⟨rfl, rfl⟩ := vmodify_ok (vmodify_bind h)
  refine ⟨rfl, hz, ha, hb, mm, hmm, rfl, rfl, rfl, rfl, kv_query _ _ _, ?_⟩
  dsimp only
  rw [polyOf_addScaled, polyOf_ofMsg]

theorem levelsLoop_le {sch : Flock.Schedule} {R : ℕ} (hq : ∀ i ≤ R, (sch.levels.getD i default).q ≠ 0) :
    ∀ (fuel i split : ℕ) (levels : Array (Array F256)) (e : ℕ) (st : LSt F128) (k : ℕ)
      (out : Array (Array F256) × ℕ) (st' : LSt F128) (k' : ℕ), i + fuel ≤ R →
      RecAlgebra.levelsLoop sch M R fuel i split levels e st k (CoinSource.ofRounds c) = .ok ((out, st'), k') → Le st st'
  | 0, _, _, _, _, st, k, out, st', k', _, h => by
    simp only [RecAlgebra.levelsLoop] at h
    obtain ⟨-, rfl, -⟩ := vpure h
    exact Le.refl _
  | fuel + 1, i, split, levels, e, st, k, out, st', k', hi, h => by
    simp only [RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := vbind_ok h
    obtain ⟨-, -, -, ho₁, -, hc₁, -⟩ := challenges_conv (p := default) 4 0 #[] st st₁ k k₁ chs h₁
    have l₁ : Le st st₁ := Le.of (by omega) hc₁
    try dsimp only at h
    by_cases hi' : i + 1 < R
    · rw [ite_eq_left hi'] at h
      obtain ⟨u₂, st₂, k₂, h₂, h⟩ := vbind_ok h
      obtain ⟨u₃, st₃, k₃, h₃, h⟩ := vbind_ok h
      obtain ⟨u₄, st₄, k₄, h₄, h⟩ := vbind_ok h
      obtain ⟨-, -, ho₂, hc₂, -⟩ := oodIntro_conv (p := default) (by omega) h₂
      obtain ⟨-, -, ho₃, hc₃, -⟩ := oodIntro_conv (p := default) (by omega) h₃
      obtain ⟨-, -, -, -, mm, -, -, ho₄, -, hc₄, -⟩ := vqueryFold (hq (i + 1) (by omega)) h₄
      exact l₁.trans ((Le.of (by omega) hc₂).trans ((Le.of (by omega) hc₃).trans ((Le.push (by omega) hc₄).trans
        (levelsLoop_le hq fuel (i + 1) _ _ _ st₄ k₄ out st' k' (by omega) h))))
    · rw [ite_eq_right hi'] at h
      obtain ⟨r, st₂, k₂, h₂, h⟩ := vbind_ok h
      obtain ⟨rfl, rfl, -, -, rfl⟩ := vquery (hq R le_rfl) h₂
      try dsimp only at h
      obtain ⟨-, h⟩ := vcoinOne_bind h
      obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := vbind_ok h
      obtain ⟨e₃, rfl⟩ := vmodify_ok h₃
      exact l₁.trans ((Le.push (le_of_eq (by rw [e₃])) (by rw [e₃])).trans
        (levelsLoop_le hq fuel (i + 1) _ _ _ st₃ _ out st' k' (by omega) h))

end Conv

/-- V*'s state after the last level's queries (Structure.lean:427–429). -/
def qPush (st : LSt F128) (b : F128) (e : KV F128) (cs : Flock.ConsCtx) : LSt F128 :=
  { st with s := kadd st.s (fk b e), conss := st.conss.push cs }

omit hne in
theorem RunsL.run {y : Flock.LM β} {st st' : LState} {k k' : ℕ} {b : β} (h : RunsL c y st k k' b st') :
    RunsT c (y.run st) k k' (b, st') := h

/-- The verifier's level loop (Ligerito.lean:202–233) runs where V*'s (`levelsLoop`) does, given a run of each of its
iterations from V*'s: `hstep` for a level with a next one, `hlast` for the last; `Inv` relates the loop's variables to
V*'s split and levels, and `fOut` is the variables the last iteration yields. -/
theorem loop_runs {M : Messages F128} {p : LigeritoProof} {ms : MerkleScheme} {sch : Flock.Schedule}
    {cap : Array ByteArray} {σ : Type}
    (body : ℕ → σ → Flock.LM (ForInStep σ)) (R : ℕ) (hq : ∀ i ≤ R, (sch.levels.getD i default).q ≠ 0)
    (Inv : ℕ → σ → ℕ → Array (Array F256) → Prop) (fOut : Array (Array F256) × ℕ → σ)
    (Hf : Array (Array F256) × ℕ → LSt F128 → Prop)
    (hstep : ∀ (i : ℕ) (s : σ) (split : ℕ) (levels : Array (Array F256)) (st : LSt F128) (k : ℕ) (chs : Array F256)
      (st₁ : LSt F128) (k₁ : ℕ) (st₂ : LSt F128) (k₂ : ℕ) (st₃ : LSt F128) (k₃ : ℕ) (st₄ : LSt F128) (k₄ : ℕ),
      i + 1 < R → Inv i s split levels → ¬ split < 4 →
      RecAlgebra.challenges M 4 st k (CoinSource.ofRounds c) = .ok ((chs, st₁), k₁) →
      RecAlgebra.oodIntro M (split - 4 + 1) (i + 1) st₁ k₁ (CoinSource.ofRounds c) = .ok (((), st₂), k₂) →
      RecAlgebra.oodIntro M (split - 4 + 1) (i + 1) st₂ k₂ (CoinSource.ofRounds c) = .ok (((), st₃), k₃) →
      RecAlgebra.queryFold sch M (i + 1) (split - 4) chs st₃ k₃ (CoinSource.ofRounds c) = .ok (((), st₄), k₄) →
      Good ms sch cap p M st₄ →
      ∃ s', RunsL c (body i s) (toL st) k k₄ (.yield s') (toL st₄) ∧ Inv (i + 1) s' (split - 4 + 1) (levels.push chs))
    (hlast : ∀ (i : ℕ) (s : σ) (split : ℕ) (levels : Array (Array F256)) (st : LSt F128) (k : ℕ) (chs : Array F256)
      (st₁ : LSt F128) (k₁ : ℕ) (pos : Array ℕ) (al : Array F128) (e : KV F128) (st₂ : LSt F128) (k₂ : ℕ),
      i + 1 = R → Inv i s split levels → ¬ split < 4 →
      RecAlgebra.challenges M 4 st k (CoinSource.ofRounds c) = .ok ((chs, st₁), k₁) →
      RecAlgebra.query sch M R chs st₁ k₁ (CoinSource.ofRounds c) = .ok (((pos, al, e), st₂), k₂) → (c[k₂]!).size = 1 →
      Hf (levels.push chs, split - 4) (qPush st₂ ((c[k₂]!).getD 0 0) e ⟨split - 4, pos, al, (c[k₂]!).getD 0 0, R⟩) →
      Good ms sch cap p M (qPush st₂ ((c[k₂]!).getD 0 0) e ⟨split - 4, pos, al, (c[k₂]!).getD 0 0, R⟩) →
      RunsL c (body i s) (toL st) k (k₂ + 1) (.yield (fOut (levels.push chs, split - 4)))
        (toL (qPush st₂ ((c[k₂]!).getD 0 0) e ⟨split - 4, pos, al, (c[k₂]!).getD 0 0, R⟩))) :
    ∀ (fuel i : ℕ) (s : σ) (split : ℕ) (levels : Array (Array F256)) (e : ℕ) (st : LSt F128) (k : ℕ)
      (out : Array (Array F256) × ℕ) (st' : LSt F128) (k' : ℕ), i + fuel = R → 0 < fuel → Inv i s split levels →
      RecAlgebra.levelsLoop sch M R fuel i split levels e st k (CoinSource.ofRounds c) = .ok ((out, st'), k') →
      Hf out st' → Good ms sch cap p M st' →
      RunsL c (forIn (List.range' i fuel) s body) (toL st) k k' (fOut out) (toL st')
  | 0, _, _, _, _, _, _, _, _, _, _, _, h0, _, _, _, _ => absurd h0 (lt_irrefl 0)
  | 1, i, s, split, levels, e, st, k, out, st', k', hR, _, hinv, h, hf, hg => by
    simp only [RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := vbind_ok h
    try dsimp only at h
    rw [ite_eq_right (show ¬ i + 1 < R by omega)] at h
    obtain ⟨⟨pos, al, e'⟩, st₂, k₂, h₂, h⟩ := vbind_ok h
    try dsimp only at h
    obtain ⟨hb, h⟩ := vcoinOne_bind h
    have h := vmodify_bind h
    try simp only [RecAlgebra.levelsLoop] at h
    obtain ⟨rfl, rfl, rfl⟩ := vpure h
    rw [List.range'_succ, List.forIn_cons]
    refine RunsL.bind (hlast i s split levels st k chs st₁ k₁ pos al e' st₂ k₂ hR hinv hs h₁ h₂ hb hf hg) ?_
    simp only [List.range'_zero, List.forIn_nil]
    exact RunsL.pure _
  | fuel + 2, i, s, split, levels, e, st, k, out, st', k', hR, _, hinv, h, hf, hg => by
    rw [RecAlgebra.levelsLoop] at h
    by_cases hs : split < 4
    · rw [ite_eq_left hs] at h
      exact (vthrow h).elim
    rw [ite_eq_right hs] at h
    obtain ⟨chs, st₁, k₁, h₁, h⟩ := vbind_ok h
    try dsimp only at h
    rw [ite_eq_left (show i + 1 < R by omega)] at h
    obtain ⟨⟨⟩, st₂, k₂, h₂, h⟩ := vbind_ok h
    obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := vbind_ok h
    obtain ⟨⟨⟩, st₄, k₄, h₄, h⟩ := vbind_ok h
    have hg₄ := hg.mono (levelsLoop_le hq (fuel + 1) (i + 1) _ _ _ st₄ k₄ out st' k' (by omega) h)
    obtain ⟨s', hrun, hinv'⟩ :=
      hstep i s split levels st k chs st₁ k₁ st₂ k₂ st₃ k₃ st₄ k₄ (by omega) hinv hs h₁ h₂ h₃ h₄ hg₄
    rw [List.range'_succ, List.forIn_cons]
    exact RunsL.bind hrun (loop_runs body R hq Inv fOut Hf hstep hlast (fuel + 1) (i + 1) s' _ _ _ st₄ k₄ out st' k'
      (by omega) (by omega) hinv' h hf hg)

/-- **The verifier's Ligerito runs where V*'s does** (`Refine.ligerito_refines`' premise; Ligerito.lean:165–277). When
V*'s Ligerito up to its final check succeeds from `T` on the rounds `c` and its final check passes (what
`ligerito_of_residuals` gives), the proof carries V*'s messages, OOD values and residual, one nonce per query and claim,
and the commitment's cap, and every level's queries pass (`Opened`), then `Flock.ligerito` returns on every record whose
rounds `k₀, …, k' − 1` are `c`'s and retained the bytes it frames. -/
theorem ligerito_of_run {ms : MerkleScheme} {sch : Flock.Schedule} {T : F128} {cap : Array ByteArray}
    {p : LigeritoProof} {M : Messages F128} {sws : Array Switched} {k₀ k' : ℕ} {initCh : Array F256}
    {levels : Array (Array F256)} {ext : ℕ} {stV : LSt F128}
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hrun : (ligeritoRun sch M).run
        { s := kofF T, q := ((Val.zero, Val.zero), (Val.zero, Val.zero), (Val.zero, Val.zero)) } k₀
        (CoinSource.ofRounds c) = .ok (((initCh, levels, ext), stV), k'))
    (hfin : FinalOK sch sws M.yr initCh levels ext stV) (hmsg : stV.msg = M.msgs.size) (hood : stV.ood = M.ood.size)
    (hpm : p.messages = M.msgs.map fun mm => (kv mm.1, kv mm.2)) (hpo : p.oodValues = M.ood)
    (hyr : p.finalProof.yr = M.yr) (hcap : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = stV.conss.size) (hcn : p.consistencyNonces.size = stV.conss.size)
    (hcbn : p.claimNonces.size = stV.ood + stV.conss.size)
    (hopen : ∀ cs ∈ stV.conss.toList, Opened ms sch cap p M cs) :
    RunsT c (Flock.ligerito ms sch T cap p (basis sws)) k₀ k' () := by
  obtain ⟨hn7, h2, hl0, hc0, hl4, hcols, hext⟩ := hdim
  have hc01 := hcols 0 (by omega)
  have hq : ∀ i ≤ sch.levels.size - 1, (sch.levels.getD i default).q ≠ 0 :=
    fun i hi => (hlv i (by omega)).q_ne_zero
  have hgV : Good ms sch cap p M stV := ⟨by omega, by omega, by omega, hopen⟩
  rw [StateT.run] at hrun
  unfold ligeritoRun at hrun
  obtain ⟨hz0, h⟩ := vcoinVec_bind (by omega) hrun
  obtain ⟨y0, hy0, h⟩ := vnextOod_bind h
  obtain ⟨hb0, h⟩ := vcoinOne_bind h
  obtain ⟨⟨⟩, stZ, kZ, hZ, h⟩ := vbind_ok h
  obtain ⟨eZ, rfl⟩ := vmodify_ok hZ
  subst eZ
  obtain ⟨m0, hm0, h⟩ := vnextMsg_bind h
  obtain ⟨⟨⟩, stA, kA, hA, h⟩ := vbind_ok h
  obtain ⟨eA, rfl⟩ := vmodify_ok hA
  clear hA hZ hrun
  obtain ⟨ic, st₁, k₁, h₁, h⟩ := vbind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, st₂, k₂, h₂, h⟩ := vbind_ok h
  obtain ⟨⟨⟩, st₃, k₃, h₃, h⟩ := vbind_ok h
  obtain ⟨⟨⟩, st₄, k₄, h₄, h⟩ := vbind_ok h
  obtain ⟨⟨lv', ex'⟩, st₅, k₅, h₅, h⟩ := vbind_ok h
  try dsimp only at h
  obtain ⟨he, e5, e6⟩ := vpure h
  simp only [Prod.mk.injEq] at he
  obtain ⟨e1, e2, e3⟩ := he
  subst ic lv' ex' st₅ k₅
  obtain ⟨ek₁, hs₁, hm₁, ho₁, hoo₁, hc₁, hr₁⟩ := challenges_conv (p := p) sch.k0 0 #[] _ _ _ _ _ h₁
  obtain ⟨ek₂, hm₂, ho₂, hc₂, hr₂⟩ := oodIntro_conv (p := p) (by omega) h₂
  obtain ⟨ek₃, hm₃, ho₃, hc₃, hr₃⟩ := oodIntro_conv (p := p) (by omega) h₃
  have hlv0 := hlv 0 (by omega)
  obtain ⟨ek₄, hz₄, ha₄, hb₄, mq, hmq, hm₄, ho₄, hoo₄, hc₄, hs₄, hq₄⟩ := vqueryFold hlv0.q_ne_zero h₄
  subst ek₄
  have g₄ : Good ms sch cap p M st₄ := hgV.mono (levelsLoop_le hq _ _ _ _ _ st₄ _ _ stV k' (by omega) h₅)
  have g₃ := g₄.mono (Le.push (by omega) hc₄)
  have g₂ := g₃.mono (Le.of (by omega) hc₃)
  have g₁ := g₂.mono (Le.of (by omega) hc₂)
  have gA := g₁.mono (Le.of (by omega) hc₁)
  have hAo : stA.ood = 1 := by rw [eA]
  obtain ⟨hmc0, hcons0⟩ := g₄.opened_push hc₄
  dsimp only at hmc0 hcons0
  have e0 : capAt cap p 0 = cap := by simp [capAt]
  have o0 : openAt p (sch.levels.size - 1) 0 =
      (p.initialProof.rows, p.initialProof.siblings, p.initialProof.salts) := by simp [openAt]
  rw [e0, o0] at hmc0
  rw [o0] at hcons0
  have hE0 := query_e (by rw [hs₁, hl0]) hmc0 (positions_size _ hlv0 _) hcons0
  unfold Flock.ligerito
  refine RunsT.bind (RunsT.check hcap) ?_
  refine RunsT.bind (RunsT.label _) ?_
  refine RunsT.bind (RunsT.absorbF _) ?_
  refine RunsT.bind (RunsT.absorbBytes _) ?_
  refine RunsT.bind (a := ((), toL stV)) (RunsL.run ?_) (RunsT.pure ())
  refine RunsL.bind (RunsL.lift (RunsT.sampleVec (by omega) hz0)) ?_
  refine RunsL.bind (nextOod_runs (by rw [hpo]; exact hy0)) ?_
  refine RunsL.bind (RunsL.lift (RunsT.absorbF _)) ?_
  refine RunsL.bind (claimPow_runs (by have := gA.cbn; show 0 < _; omega)) ?_
  refine RunsL.bind (RunsL.lift (RunsT.sample hb0)) ?_
  refine RunsL.bind (RunsL.modify' _) ?_
  refine RunsL.bind (nextMsg_runs (msg_of hpm hm0)) ?_
  refine RunsL.bind (st₁ := toL stA) (RunsL.cast (RunsL.modify' _) rfl rfl ?_) ?_
  · rw [eA]
    apply toL_eq <;> dsimp only [toL]
    all_goals first | rfl | (rw [kv_ood]; rfl) | (rw [polyOf_ofMsg, kv_ood]; rfl)
  dsimp only
  rw [lforIn_range_eq sch.k0]
  refine RunsL.bind (RunsL.cast (hr₁ hpm) rfl Array.empty_append rfl) ?_
  refine RunsL.bind (RunsL.lift (RunsT.absorbBytes _)) ?_
  refine RunsL.bind (hr₂ hpm hpo (by have := g₃.cbn; rw [hc₃, hc₂] at this; omega)) ?_
  refine RunsL.bind (hr₃ hpm hpo (by have := g₃.cbn; rw [hc₃] at this; omega)) ?_
  refine RunsL.bind (queryLevel_runs hlv0.q_ne_zero hz₄ ha₄
    (by have := g₄.qn; rw [hc₄, Array.size_push] at this; exact this)
    (by have := g₄.cn; rw [hc₄, Array.size_push] at this; exact this) hmc0) ?_
  refine RunsL.bind (nextMsg_runs (msg_of hpm hmq)) ?_
  refine RunsL.bind (claimPow_runs
    (by have := g₄.cbn; rw [hc₄, Array.size_push, ho₄] at this; show st₃.ood + st₃.conss.size < _; omega)) ?_
  refine RunsL.bind (RunsL.lift (RunsT.sample hb₄)) ?_
  refine RunsL.bind (st₁ := toL st₄) (RunsL.cast (RunsL.modify' _) rfl rfl ?_) ?_
  · apply toL_eq <;> dsimp only [toL]
    all_goals (try simp only [hs₄, hq₄, hE0, hm₄, ho₄, hoo₄, hc₄, Array.size_push])
    all_goals (try omega)
  try dsimp only
  rw [lforIn_range_eq (sch.levels.size - 1)]
  refine RunsL.bind (loop_runs (p := p) (ms := ms) (cap := cap) (σ := ℕ × Array ByteArray × Array (Array F256) × ℕ)
    _ (sch.levels.size - 1) hq
    (fun i s split lvls => split = s.1 ∧ lvls = s.2.2.1 ∧ s.2.1 = p.recursiveCaps[i]! ∧
      s.1 = (sch.levels.getD (i + 1) default).cols + 4)
    (fun o => ((sch.levels.getD (sch.levels.size - 1) default).cols + 4, p.recursiveCaps[sch.levels.size - 1 - 1]!,
      o.1, o.2))
    (fun o st => st.msg = M.msgs.size ∧ M.yr.size = 2 ^ (o.2 + 1))
    ?hstep ?hlast (sch.levels.size - 1) 0 (sch.n - sch.k0 + 1, p.recursiveCaps[0]!, #[], 0) (sch.n - sch.k0 + 1)
    #[] 0 st₄ _ _ stV k' (by omega) (by omega)
    ⟨rfl, rfl, rfl, by omega⟩ h₅ ⟨hmsg, hfin.yr_size⟩ hgV) ?_
  case hstep =>
    intro i s split lvls st k chs st₁ k₁ st₂ k₂ st₃ k₃ st₄ k₄ hi hinv hs h₁ h₂ h₃ h₄ g₄
    obtain ⟨rfl, rfl, hprev, hsplit⟩ := hinv
    obtain ⟨ek₁, hs₁, hm₁, ho₁, hoo₁, hc₁, hr₁⟩ := challenges_conv (p := p) 4 0 #[] _ _ _ _ _ h₁
    obtain ⟨ek₂, hm₂, ho₂, hc₂, hr₂⟩ := oodIntro_conv (p := p) (by omega) h₂
    obtain ⟨ek₃, hm₃, ho₃, hc₃, hr₃⟩ := oodIntro_conv (p := p) (by omega) h₃
    have hlvi := hlv (i + 1) (by omega)
    obtain ⟨ek₄, hz₄, ha₄, hb₄, mq, hmq, hm₄, ho₄, hoo₄, hc₄, hs₄, hq₄⟩ := vqueryFold hlvi.q_ne_zero h₄
    subst ek₄
    obtain ⟨hmc, hcons⟩ := g₄.opened_push hc₄
    dsimp only at hmc hcons
    have ecap : capAt cap p (i + 1) = s.2.1 := by rw [hprev]; simp [capAt]
    have eopen : openAt p (sch.levels.size - 1) (i + 1) =
        ((p.recursiveProofs[i]!).rows, (p.recursiveProofs[i]!).siblings, (p.recursiveProofs[i]!).salts) := by
      simp [openAt, hi]
    rw [ecap, eopen] at hmc
    rw [eopen] at hcons
    have hE := query_e (by rw [hs₁, hl4 (i + 1) (by omega) (by omega)]) hmc (positions_size _ hlvi _) hcons
    have g₃ := g₄.mono (Le.push (by omega) hc₄)
    have hc1 := hcols (i + 1) (by omega)
    refine ⟨(s.1 - 4 + 1, p.recursiveCaps[i + 1]!, s.2.2.1.push chs, s.1 - 4), ?_, rfl, rfl, rfl, by omega⟩
    try dsimp only
    rw [ite_eq_right hs, lforIn_range_eq 4]
    refine RunsL.bind (RunsL.cast (hr₁ hpm) rfl Array.empty_append rfl) ?_
    try dsimp only
    rw [ite_eq_left hi]
    refine RunsL.bind (RunsL.lift (RunsT.absorbBytes _)) ?_
    refine RunsL.bind (hr₂ hpm hpo (by have := g₃.cbn; rw [hc₃, hc₂] at this; omega)) ?_
    refine RunsL.bind (hr₃ hpm hpo (by have := g₃.cbn; rw [hc₃] at this; omega)) ?_
    refine RunsL.bind (queryLevel_runs hlvi.q_ne_zero hz₄ ha₄
      (by have := g₄.qn; rw [hc₄, Array.size_push] at this; exact this)
      (by have := g₄.cn; rw [hc₄, Array.size_push] at this; exact this) hmc) ?_
    refine RunsL.bind (nextMsg_runs (msg_of hpm hmq)) ?_
    refine RunsL.bind (claimPow_runs
      (by have := g₄.cbn; rw [hc₄, Array.size_push, ho₄] at this; show st₃.ood + st₃.conss.size < _; omega)) ?_
    refine RunsL.bind (RunsL.lift (RunsT.sample hb₄)) ?_
    refine RunsL.bind (RunsL.modify' _) ?_
    try dsimp only
    rw [ite_eq_right (show ¬ s.1 - 4 + 1 < 4 by omega)]
    refine RunsL.cast (RunsL.pure _) rfl rfl ?_
    apply toL_eq <;> dsimp only [toL]
    all_goals (try simp only [hs₄, hq₄, hE, hm₄, ho₄, hoo₄, hc₄, Array.size_push])
    all_goals (try omega)
  case hlast =>
    intro i s split lvls st k chs st₁ k₁ pos al e st₂ k₂ hi hinv hs h₁ h₂ hb hf gL
    obtain ⟨rfl, rfl, hprev, hsplit⟩ := hinv
    obtain ⟨ek₁, hs₁, hm₁, ho₁, hoo₁, hc₁, hr₁⟩ := challenges_conv (p := p) 4 0 #[] _ _ _ _ _ h₁
    have hlvR := hlv (sch.levels.size - 1) (by omega)
    obtain ⟨e₂, rfl, hz, ha, he⟩ := vquery hlvR.q_ne_zero h₂
    subst st₂
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl⟩ := he
    obtain ⟨hmc, hcons⟩ := gL.opened_push rfl
    dsimp only at hmc hcons
    have ecap : capAt cap p (sch.levels.size - 1) = s.2.1 := by
      rw [hprev]
      simp [capAt, show sch.levels.size - 1 ≠ 0 by omega, show sch.levels.size - 1 - 1 = i by omega]
    have eopen : openAt p (sch.levels.size - 1) (sch.levels.size - 1) =
        (p.finalProof.rows, p.finalProof.siblings, p.finalProof.salts) := by
      simp [openAt, show sch.levels.size - 1 ≠ 0 by omega]
    rw [ecap, eopen] at hmc
    rw [eopen] at hcons
    have hE := query_e (by rw [hs₁, hl4 _ (by omega) (by omega)]) hmc (positions_size _ hlvR _) hcons
    obtain ⟨hmf, hyf⟩ := hf
    have gq := gL.qn
    have gc := gL.cn
    have gb := gL.cbn
    simp only [qPush, Array.size_push] at hmf gq gc gb
    try dsimp only
    rw [ite_eq_right hs, lforIn_range_eq 4]
    refine RunsL.bind (RunsL.cast (hr₁ hpm) rfl Array.empty_append rfl) ?_
    try dsimp only
    rw [ite_eq_right (show ¬ i + 1 < sch.levels.size - 1 by omega)]
    refine RunsL.get_bind ?_
    try dsimp only
    rw [ite_eq_right (by simp only [bne_iff_ne, ne_eq, not_not]; rw [hpm, Array.size_map]; exact hmf)]
    rw [ite_eq_right (by simp only [bne_iff_ne, ne_eq, not_not]; rw [hyr]; exact hyf)]
    rw [← Array.forIn_toList]
    refine RunsL.bind (forIn_absorbF_runs _ _ _) ?_
    refine RunsL.bind (queryLevel_runs hlvR.q_ne_zero hz ha gq gc hmc) ?_
    refine RunsL.bind (claimPow_runs (by show _ + _ < _; omega)) ?_
    refine RunsL.bind (RunsL.lift (RunsT.sample hb)) ?_
    refine RunsL.bind (RunsL.modify' _) ?_
    refine RunsL.cast (RunsL.pure _) rfl ?_ ?_
    · refine congrArg ForInStep.yield ?_
      have e1 : sch.levels.size - 1 - 1 = i := by omega
      have e2 : i + 1 = sch.levels.size - 1 := by omega
      try dsimp only
      rw [e1, ← hprev, ← e2, ← hsplit]
    · apply toL_eq <;> dsimp only [toL, qPush]
      all_goals (try simp only [kv_query, hE, Array.size_push])
      all_goals (try omega)
  exact ⟨le_rfl, fun P => ⟨P, fun _ => ByteArray.empty, fun rs _ =>
    finalProg_of_finalOK hfin hyr rfl rfl rfl ⟨by rw [hpo, ← hood]; rfl, hqn.symm, hcn.symm, hcbn.symm⟩⟩⟩

open FlockSoundness FlockSoundness.Model FlockSoundness.Game FlockSoundness.Accounting in
/-- **The model's plays at V*'s run** (`Refine.ligerito_refines` on `ligerito_of_run`): on every well-formed record whose
rounds `k₀, …, k' − 1` are `c`'s and retained the bytes the verifier frames, the model's compiled Ligerito, played with
the proof's messages and those rounds, ends as `ligerito_refines` concludes. -/
theorem ligerito_refines_of_run {ms : MerkleScheme} {sch : Flock.Schedule} {T : F128} {cap : Array ByteArray}
    {p : LigeritoProof} {M : Messages F128} {sws : Array Switched} {k₀ k' : ℕ} {initCh : Array F256}
    {levels : Array (Array F256)} {ext : ℕ} {stV : LSt F128}
    (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hrun : (ligeritoRun sch M).run
        { s := kofF T, q := ((Val.zero, Val.zero), (Val.zero, Val.zero), (Val.zero, Val.zero)) } k₀
        (CoinSource.ofRounds c) = .ok (((initCh, levels, ext), stV), k'))
    (hfin : FinalOK sch sws M.yr initCh levels ext stV) (hmsg : stV.msg = M.msgs.size) (hood : stV.ood = M.ood.size)
    (hpm : p.messages = M.msgs.map fun mm => (kv mm.1, kv mm.2)) (hpo : p.oodValues = M.ood)
    (hyr : p.finalProof.yr = M.yr) (hcap : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = stV.conss.size) (hcn : p.consistencyNonces.size = stV.conss.size)
    (hcbn : p.claimNonces.size = stV.ood + stV.conss.size)
    (hopen : ∀ cs ∈ stV.conss.toList, Opened ms sch cap p M cs)
    {D : Type} (r : ℕ) (b : ℕ → FlockLevel3.GF128) (capD : Array ByteArray → Cap D) (o : Opens FlockLevel3.GF128 D)
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf) (P : ByteArray) :
    ∃ (P' : ByteArray) (F : ℕ → ByteArray), ∀ rs, Fits c F rs k₀ k' → RoundsWF rs →
      ∃ (pend : PendL FlockLevel3.GF128 D) (initCh' : Array F256) (levels' : Array (Array F256))
        (stF stF' : LState) (tF : Flock.Transcript),
        tF.rounds = rs ∧ k₀ ≤ tF.k ∧
        Plays fOf (Model.ligeritoC execArith r (schedOf sch) (fOf T) b) (ligMsgs p capD sch)
          (consumed rs k₀ tF.k) pend ∧
        pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.recursiveCaps[j]!) ∧
        (∀ req ∈ pend.reqs, ReqOK ms sch cap p req) ∧
        initCh'.size = sch.k0 ∧ levels'.size = sch.levels.size - 1 ∧ (∀ l ∈ levels'.toList, l.size = 4) ∧
        p.finalProof.yr.size = 2 ^ (sch.ext + 1) ∧ (∀ cs ∈ stF.conss.toList, cs.pos.size ≤ 2 ^ cs.alpha.size) ∧
        finalProg p (basis sws) sch.n initCh' levels' sch.ext stF tF = .ok (((), stF'), ⟨rs, k', P'⟩) ∧
        pend.verdict o = Model.finalCheck execArith (sch.m - 7) b (initCh'.toList.map gOf)
          (levels'.toList.map fun l => l.toList.map gOf) (stF.oods.toList.map FlockSoundness.Refine.oodOf)
          (stF.conss.toList.map FlockSoundness.Refine.consOf)
          sch.ext (yrOf p sch.ext) (gOf stF.s) := by
  obtain ⟨P', F, hF⟩ := (ligerito_of_run hdim hlv hrun hfin hmsg hood hpm hpo hyr hcap hqn hcn hcbn hopen).2 P
  exact ⟨P', F, fun rs hfit hwf =>
    ligerito_refines ms sch T cap p (basis sws) r b capD o hdim hlv hα1 ho (t := ⟨rs, k₀, P⟩) hwf (hF rs hfit)⟩

end Flock.RecAlgebra
