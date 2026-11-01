import Proofs.Flock.Claims.Complete.Zk.RingSwitch

/-!
# The masked replay returns on the prover's rounds

`replayWith_ok`: on rounds that carry the masked prover's zerocheck and lincheck rounds (`Agrees`), from the prover's
start position and pending bytes, the verifier's masked replay (`Zk.replayWith`) returns under any hash, wherever the
circuit fold at the prover's coins returns a vector of width `64`. Each squeeze reads the round the prover's matching
send recorded, whose bytes are the ones the replay framed, since the replay absorbs the prover's messages
(`ZcZ.proof`, `LcZ.proof`) in the prover's order; its only refusals are the fold's and the width check.

* **Runs that return** (`OkV x t P`): some run of `x` from `t` returns and ends in `P`; composed by `OkV.bind`,
  through loops by `OkV.forIn_range` (`RetV`'s forward forms).
* `zc_carries`, `lc_carries`: such rounds carry each squeeze at its position. Each is its own declaration, and the
  replay names the prover's transcripts in full rather than through `set`: the kernel times out on the combined
  proof otherwise.
-/

namespace FlockComplete

open Flock Transcript FlockComplete.Replay

/-- **A run of `x` from `t` returns, and ends in `P`.** -/
def OkV {α : Type} (x : TMW V α) (t : Transcript) (P : α → Transcript → Prop) : Prop :=
  ∃ a t', x t = .ok (a, t') ∧ P a t'

namespace OkV

theorem bind {α β : Type} {x : TMW V α} {f : α → TMW V β} {t : Transcript} {P : β → Transcript → Prop}
    (h : OkV x t fun a t₁ => OkV (f a) t₁ P) : OkV (x >>= f) t P := by
  obtain ⟨a, t₁, e1, b, t', e2, hP⟩ := h
  exact ⟨b, t', (Flock.RecAlgebra.bind_of_ok e1).trans e2, hP⟩

theorem pure {α : Type} {a : α} {t : Transcript} {P : α → Transcript → Prop} (h : P a t) :
    OkV (Pure.pure a : TMW V α) t P :=
  ⟨a, t, rfl, h⟩

/-- A step that returns `a` at `t₁`, then `f a`. -/
theorem ok_then {α β : Type} {x : TMW V α} {f : α → TMW V β} {t t₁ : Transcript} {a : α}
    {P : β → Transcript → Prop} (hx : x t = .ok (a, t₁)) (h : OkV (f a) t₁ P) : OkV (x >>= f) t P :=
  bind ⟨a, t₁, hx, h⟩

theorem mono {α : Type} {x : TMW V α} {t : Transcript} {P Q : α → Transcript → Prop} (h : OkV x t P)
    (hPQ : ∀ a t', P a t' → Q a t') : OkV x t Q := by
  obtain ⟨a, t', e, hP⟩ := h
  exact ⟨a, t', e, hPQ a t' hP⟩

/-- **A loop each of whose steps returns and yields** keeps its invariant and returns. -/
theorem forIn_range {σ : Type} (n : ℕ) (f : ℕ → σ → TMW V (ForInStep σ)) (inv : ℕ → σ → Transcript → Prop)
    (hf : ∀ i b t, i < n → inv i b t → OkV (f i b) t fun r t' => ∃ b', r = .yield b' ∧ inv (i + 1) b' t')
    (b : σ) (t : Transcript) (h0 : inv 0 b t) : OkV (forIn [:n] b f) t fun b' t' => inv n b' t' := by
  rw [FlockSoundness.Refine.forIn_range_eq]
  suffices h : ∀ k a b t, a + k = n → inv a b t → OkV (forIn (List.range' a k) b f) t fun b' t' => inv n b' t' from
    h n 0 b t (Nat.zero_add n) h0
  intro k
  induction k with
  | zero =>
    intro a b t hk hi
    rw [Nat.add_zero] at hk
    subst hk
    exact pure hi
  | succ k ih =>
    intro a b t hk hi
    rw [List.range'_succ, List.forIn_cons]
    obtain ⟨r, t₁, e, b', rfl, hi'⟩ := hf a b t (by omega) hi
    exact ok_then e (ih (a + 1) b' t₁ (by omega) hi')

end OkV

namespace Replay

/-- Zerocheck's residual round `i` before its squeeze, from `t₀`: its padded message absorbed. -/
def ZcZ.pre (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (cs : Zc.Challenges) (tr : Trace) (t₀ : Transcript) (i : ℕ) :
    Transcript :=
  ⟨(ZcZ.loopT lay h m cs tr t₀ i).rounds, (ZcZ.loopT lay h m cs tr t₀ i).k,
    pF (ZcZ.msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ i).2
      (pF (ZcZ.msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ i).1 (ZcZ.loopT lay h m cs tr t₀ i).pending)⟩

/-- Lincheck's round `j` before its squeeze, from `t`: its padded message absorbed. -/
def LcZ.pre (lay : Zk.PadLayout) (h : Array F128) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut)
    (full : Array F128) (cs : Lc.Challenges) (t : Transcript) (j : ℕ) : Transcript :=
  ⟨(LcZ.loopT lay h c m tr zc full cs t j).rounds, (LcZ.loopT lay h c m tr zc full cs t j).k,
    pF (LcZ.msg lay h c m tr zc full cs j).2
      (pF (LcZ.msg lay h c m tr zc full cs j).1 (LcZ.loopT lay h c m tr zc full cs t j).pending)⟩

variable {R : Array Round} {lay : Zk.PadLayout} {h : Array F128} {m : ℕ} {tr : Trace}

/-- **Rounds carrying the masked prover's zerocheck carry each of its squeezes**, at its position. -/
theorem zc_carries {digest cap root : ByteArray} {zcs : Zc.Challenges} {t₀ : Transcript} (hk₀ : t₀.k = t₀.rounds.size)
    (hZ : Agrees R (ZcZ.tEnd lay h m digest cap root zcs tr t₀).rounds 0) :
    Carries R (ZcZ.t₁ digest cap root t₀).k (sent (ZcZ.t₁ digest cap root t₀) 2 6 zcs.c0) ∧
      Carries R (ZcZ.t₂ digest cap root zcs t₀).k (sent (ZcZ.t₂ digest cap root zcs t₀) 2 (m - 13) zcs.c1) ∧
      Carries R (ZcZ.t₄ lay h m digest cap root zcs tr t₀).k
        (sent (ZcZ.t₄ lay h m digest cap root zcs tr t₀) 1 1 zcs.c2) ∧
      ∀ i < m - 6, Carries R (ZcZ.pre lay h m zcs tr (ZcZ.t₅ lay h m digest cap root zcs tr t₀) i).k
        (sent (ZcZ.pre lay h m zcs tr (ZcZ.t₅ lay h m digest cap root zcs tr t₀) i) 1 1 (zcs.cr i)) := by
  refine ⟨?_, ?_, ?_, fun i hi => ?_⟩
  · have := hZ.at_push (((Prefix.push _ _).trans (Prefix.push _ _)).trans (ZcZ.prefix_loop (i := 0) t₀
      (Nat.zero_le _))) (Nat.zero_le _)
    rwa [show (ZcZ.t₁ digest cap root t₀).rounds.size = (ZcZ.t₁ digest cap root t₀).k from hk₀.symm] at this
  · have := hZ.at_push ((Prefix.push _ _).trans (ZcZ.prefix_loop (i := 0) t₀ (Nat.zero_le _))) (Nat.zero_le _)
    rwa [show (ZcZ.t₂ digest cap root zcs t₀).rounds.size = (ZcZ.t₂ digest cap root zcs t₀).k by
      simp [ZcZ.t₂, ZcZ.t₁, afterSq, hk₀]] at this
  · have := hZ.at_push (ZcZ.prefix_loop (i := 0) t₀ (Nat.zero_le _)) (Nat.zero_le _)
    rwa [show (ZcZ.t₄ lay h m digest cap root zcs tr t₀).rounds.size =
      (ZcZ.t₄ lay h m digest cap root zcs tr t₀).k by simp [ZcZ.t₄, ZcZ.t₃, ZcZ.t₂, ZcZ.t₁, afterSq, hk₀]] at this
  · have := hZ.at_push (ZcZ.prefix_loop (i := i + 1) t₀ (by omega)) (Nat.zero_le _)
    rwa [show (ZcZ.loopT lay h m zcs tr (ZcZ.t₅ lay h m digest cap root zcs tr t₀) i).rounds.size =
        (ZcZ.loopT lay h m zcs tr (ZcZ.t₅ lay h m digest cap root zcs tr t₀) i).k by
      rw [ZcZ.loopT_size, ZcZ.loopT_k]
      simp [ZcZ.t₅, ZcZ.t₄, ZcZ.t₃, ZcZ.t₂, ZcZ.t₁, afterSq, hk₀]] at this

/-- **Rounds carrying the masked prover's lincheck carry each of its squeezes**, at its position, from any start `T₁`
whose earlier rounds are recorded. -/
theorem lc_carries {c : CircuitFold} {zc : ZcOut} {full : Array F128} {lcs : Lc.Challenges} (T₁ : Transcript)
    (hT₁k : T₁.k = T₁.rounds.size) (hR : Agrees R (LcZ.tEnd lay h c m tr zc full lcs T₁).rounds 0) :
    Carries R (Lc.t₁ T₁).k (sent (Lc.t₁ T₁) 1 1 lcs.ca) ∧
      Carries R (Lc.t₂ lcs T₁).k (sent (Lc.t₂ lcs T₁) 1 1 lcs.cb) ∧
      (∀ j < c.kLog - 6, Carries R (LcZ.pre lay h c m tr zc full lcs T₁ j).k
        (sent (LcZ.pre lay h c m tr zc full lcs T₁ j) 1 1 (lcs.cr j))) ∧
      Carries R (LcZ.t₄ lay h c m tr zc full lcs T₁).k (sent (LcZ.t₄ lay h c m tr zc full lcs T₁) 1 1 lcs.cf) := by
  refine ⟨?_, ?_, fun j hj => ?_, ?_⟩
  · have := hR.at_push ((Prefix.push _ _).trans (LcZ.prefix_loop (i := 0) T₁ (Nat.zero_le _))) (Nat.zero_le _)
    rwa [show (Lc.t₁ T₁).rounds.size = (Lc.t₁ T₁).k from hT₁k.symm] at this
  · have := hR.at_push (LcZ.prefix_loop (i := 0) T₁ (Nat.zero_le _)) (Nat.zero_le _)
    rwa [show (Lc.t₂ lcs T₁).rounds.size = (Lc.t₂ lcs T₁).k by simp [Lc.t₂, Lc.t₁, afterSq, hT₁k]] at this
  · have := hR.at_push (LcZ.prefix_loop (i := j + 1) T₁ (by omega)) (Nat.zero_le _)
    rwa [show (LcZ.loopT lay h c m tr zc full lcs T₁ j).rounds.size = (LcZ.loopT lay h c m tr zc full lcs T₁ j).k by
      rw [LcZ.loopT_size, LcZ.loopT_k, hT₁k]] at this
  · have := hR.at_push (Prefix.refl _) (Nat.zero_le _)
    rwa [show (LcZ.t₄ lay h c m tr zc full lcs T₁).rounds.size = (LcZ.t₄ lay h c m tr zc full lcs T₁).k by
      simp only [LcZ.t₄, LcZ.loopT_size, LcZ.loopT_k, hT₁k]] at this

end Replay

/-- **The masked replay returns on the prover's rounds**: on rounds `R` that carry the masked prover's zerocheck and
lincheck rounds from `t₀` (`t₀.k` its position, every earlier round recorded), on its proof (`ZcZ.proof`,
`LcZ.proof`) under the cap `p.cap`, from `t₀`'s position and pending bytes, the verifier's masked replay returns under
any hash, wherever the circuit fold at the prover's coins returns a vector of width `64`. -/
theorem replayWith_ok (hs : Hashes V) (st : Setup) (hm : 14 ≤ st.m) (lay : Zk.PadLayout) (h : Array F128)
    (p : Proof) (root : ByteArray) (zcs : Zc.Challenges) (lcs : Lc.Challenges) (full : Array F128) (tr : Trace)
    (t₀ : Transcript) (hk₀ : t₀.k = t₀.rounds.size) {R : Array Round}
    (hR : Agrees R (LcZ.tEnd lay h st.fold st.m tr (Zc.out st.m zcs tr) full lcs
      (ZcZ.tEnd lay h st.m st.digest (flat p.cap) root zcs tr t₀)).rounds 0)
    (hpz : p.zerocheck = ZcZ.proof lay h st.m zcs tr)
    (hpl : p.lincheck = LcZ.proof lay h st.fold st.m tr (Zc.out st.m zcs tr) full lcs)
    (hkk : st.kLog = st.fold.kLog)
    (hL : ∃ comb, st.fold.folded (lcs.ca 0) (lcs.cb 0) (skipWeights (Zc.out st.m zcs tr).z)
      ((Zc.out st.m zcs tr).rho.extract 0 (st.fold.kLog - 6)) (pushes lcs.τ (st.fold.kLog - 6)) = .ok comb ∧
      comb.size = 64) :
    ∃ q, (Zk.replayWith hs st lay p root).run ⟨R, t₀.k, t₀.pending⟩ = .ok q := by
  obtain ⟨comb, hcomb, hcsz⟩ := hL
  have hT₁k : (ZcZ.tEnd lay h st.m st.digest (flat p.cap) root zcs tr t₀).k =
      (ZcZ.tEnd lay h st.m st.digest (flat p.cap) root zcs tr t₀).rounds.size := by
    rw [ZcZ.tEnd_k, ZcZ.tEnd_size, hk₀]
  obtain ⟨c0, c1, c2, cl⟩ := zc_carries hk₀ (hR.of_prefix (LcZ.prefix_start _))
  obtain ⟨ca, cb, cτ, cf⟩ := lc_carries _ hT₁k hR
  have hzr : ∀ i < st.m - 6, p.zerocheck.rounds[i]! = ZcZ.msg lay h tr (st.m - 6) (zcs.r st.m) zcs.z zcs.ρ i :=
    fun i hi => by rw [hpz]; exact pushes_get _ _ i hi
  have hlr : ∀ j < st.fold.kLog - 6, p.lincheck.rounds[j]! =
      LcZ.msg lay h st.fold st.m tr (Zc.out st.m zcs tr) full lcs j :=
    fun j hj => by rw [hpl]; exact pushes_get _ _ j hj
  suffices hO : OkV (Zk.replayWith hs st lay p root) ⟨R, t₀.k, t₀.pending⟩ fun _ _ => True by
    obtain ⟨a, t', e, -⟩ := hO
    exact ⟨(a, t'), e⟩
  unfold Zk.replayWith
  simp only [Flock.With.liftTM_V]
  refine OkV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.absorbBytes_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.absorbBytes_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.absorbBytes_eq _ _) ?_
  refine OkV.ok_then (sampleVecWith_sent_at hs (by decide) _ (by rfl) (by rfl) (by rfl) c0) ?_
  refine OkV.ok_then (sampleVecWith_sent_at hs (by omega) _ (by rfl) (by rfl) (by rfl) c1) ?_
  refine OkV.ok_then (Flock.RecAlgebra.absorbSlice_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.absorbSlice_eq _ _) ?_
  refine OkV.ok_then (sampleWith_sent_at hs _ (by rfl) (by rfl) (by rw [hpz]; rfl) c2) ?_
  refine OkV.bind (OkV.mono (OkV.forIn_range _ _
    (fun j b u => u.rounds = R ∧
      u.k = (ZcZ.loopT lay h st.m zcs tr (ZcZ.t₅ lay h st.m st.digest (flat p.cap) root zcs tr t₀) j).k ∧
      u.pending = (ZcZ.loopT lay h st.m zcs tr (ZcZ.t₅ lay h st.m st.digest (flat p.cap) root zcs tr t₀) j).pending ∧
      b.2 = pushes zcs.ρ j) ?_ _ _ ⟨rfl, rfl, rfl, rfl⟩) fun b₁ t₁ h₁ => ?_)
  · intro i b u hi hinv
    obtain ⟨hRu, hku, hpu, hb⟩ := hinv
    refine OkV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine OkV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine OkV.ok_then (sampleWith_sent_at hs _ (by exact hRu) (by exact hku)
      (by simp only [ZcZ.pre, hpu, hzr i hi, pF]) (cl i hi)) ?_
    refine OkV.pure ⟨_, rfl, rfl, rfl, rfl, ?_⟩
    simp only [hb, pushes, Zc.Challenges.ρ]
  obtain ⟨hR₁, hk₁, hp₁, hb₁⟩ := h₁
  refine OkV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine OkV.ok_then (sampleWith_sent_at hs _ (by exact hR₁) (by exact hk₁) (by simp only [hp₁, hpz]; rfl) ca) ?_
  refine OkV.ok_then (sampleWith_sent_at hs _ (by rfl) (by rfl) (by rfl) cb) ?_
  refine OkV.bind (OkV.mono (OkV.forIn_range _ _
    (fun j b u => u.rounds = R ∧
      u.k = (LcZ.loopT lay h st.fold st.m tr (Zc.out st.m zcs tr) full lcs
        (ZcZ.tEnd lay h st.m st.digest (flat p.cap) root zcs tr t₀) j).k ∧
      u.pending = (LcZ.loopT lay h st.fold st.m tr (Zc.out st.m zcs tr) full lcs
        (ZcZ.tEnd lay h st.m st.digest (flat p.cap) root zcs tr t₀) j).pending ∧
      b.2 = pushes lcs.τ j) ?_ _ _ ⟨rfl, rfl, rfl, rfl⟩) fun b₂ t₂ h₂ => ?_)
  · intro j b u hj hinv
    obtain ⟨hRu, hku, hpu, hb⟩ := hinv
    rw [hkk] at hj
    refine OkV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine OkV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine OkV.ok_then (sampleWith_sent_at hs _ (by exact hRu) (by exact hku)
      (by simp only [LcZ.pre, hpu, hlr j hj, pF]) (cτ j hj)) ?_
    refine OkV.pure ⟨_, rfl, rfl, rfl, rfl, ?_⟩
    simp only [hb, pushes, Lc.Challenges.τ]
  obtain ⟨hR₂, hk₂, hp₂, hb₂⟩ := h₂
  have hfold : st.fold.folded (lcs.ca 0) (lcs.cb 0) (skipWeights (zcs.c2 0)) (b₁.2.extract 0 (st.kLog - 6)) b₂.2 =
      .ok comb := by
    rw [hb₁, hb₂, hkk]
    exact hcomb
  rw [hfold]
  refine OkV.ok_then (x := Pure.pure comb) rfl ?_
  refine OkV.ok_then (Flock.RecAlgebra.check_eq (by simp [hcsz]) _ _) ?_
  refine OkV.ok_then (Flock.RecAlgebra.absorbSlice_eq _ _) ?_
  refine OkV.ok_then (sampleWith_sent_at hs _ (by exact hR₂) (by rw [hk₂, hkk]; rfl)
    (by simp only [hp₂, hpl, hkk]; rfl) cf) ?_
  exact OkV.pure trivial

end FlockComplete
