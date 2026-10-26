import Proofs.Flock.Claims.ZK.L0PinFlags
import Proofs.Flock.Claims.ZK.L0PinFrame
import Proofs.Flock.Claims.ZK.FrameRounds

/-!
# The ring switch's rounds, and the PIOP's last transcript

Level 0's scheduled rounds (`LigLock.Fed.openStep_l0`) start past the ring switch's cut run from the PIOP's last
transcript. This file states both:

* **The ring switch's bodies** (`ringSwitchCut_bodies`, `rsBody`): answered by `asksCut` with room for its
  `claims.size + 1` asks, the cut ring switch returns, having sent, after the pending bytes and the opening's label,
  each claim's round (the ring switch's label, the claim's switched vector, a squeeze of `7`), then `γ`'s squeeze of
  `claims.size`, and leaves nothing pending. The loop is `forIn_bind_go`, each step's state left open.
* **The PIOP leaves nothing pending** (`EndE`, `endE_repPiop`): a rep's PIOP that returns on the opening's coins ends
  on the lincheck's last squeeze.
* **The rounds' shapes** (`pcsRound_rs`, `qRound_rs`, `qRound_gamma`): from empty pending bytes the switch's first
  round is the opening's (`Flags.PcsRound`), and its later rounds and `γ`'s are quiet (`Quiet.QRound`).
-/

namespace Claims.ZKServer.Switch

open _root_.Flock (F128 Transcript TMW)
open _root_.Flock.Transcript (framed header padded)
open _root_.Flock.Firewall.Frame (LABEL_PCS)
open FlockSoundness.Refine (Frame encs label_enc absorbSlice_enc framed_enc)
open FlockComplete (PS asksCut ringSwitchCut)
open FlockComplete.Replay (afterSq bind_go liftTM_ok)

/-! ## A loop whose steps' states are left open -/

theorem forIn_go_ex {σ : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (T : ℕ → Transcript) (B : ℕ → Claims.Msg)
    (L n : ℕ)
    (hbody : ∀ i < n, ∀ (s : σ) (l : Array Claims.Msg), l.size = L + i →
      ∃ s', ((body i s).run (T i)).go l = (.ok (.yield s', T (i + 1)), l.push (B i)))
    (s₀ : σ) (l : Array Claims.Msg) (hl : l.size = L) :
    ∃ s, ((forIn [:n] s₀ body).run (T 0)).go l = (.ok (s, T n), l ++ ((List.range n).map B).toArray) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices h : ∀ k a (s : σ) (l : Array Claims.Msg), a + k ≤ n → l.size = L + a →
      ∃ s', ((forIn (List.range' a k) s body).run (T a)).go l =
        (.ok (s', T (a + k)), l ++ ((List.range' a k).map B).toArray) by
    simpa [List.range_eq_range'] using h n 0 s₀ l (by omega) (by simpa using hl)
  intro k
  induction k with
  | zero =>
    intro a s l _ _
    exact ⟨s, by show (Except.ok (s, T a), l) = _; simp⟩
  | succ k ih =>
    intro a s l hk hl
    obtain ⟨s', hs'⟩ := hbody a (by omega) s l hl
    rw [List.range'_succ, List.forIn_cons, bind_go hs']
    dsimp only
    obtain ⟨s'', hs''⟩ := ih (a + 1) s' (l.push (B a)) (by omega) (by simp [hl]; omega)
    rw [hs'', show a + 1 + k = a + (k + 1) by omega]
    exact ⟨s'', by simp⟩

/-- **A loop, then a continuation**: what holds of the continuation's run from every state the loop may end in holds
of the whole. -/
theorem forIn_bind_go {σ β : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (T : ℕ → Transcript)
    (B : ℕ → Claims.Msg) (L n : ℕ) {t₀ : Transcript} (hT : T 0 = t₀)
    (hbody : ∀ i < n, ∀ (s : σ) (l : Array Claims.Msg), l.size = L + i →
      ∃ s', ((body i s).run (T i)).go l = (.ok (.yield s', T (i + 1)), l.push (B i)))
    (s₀ : σ) (f : σ → TMW PS β) (l : Array Claims.Msg) (hl : l.size = L)
    (P : Except String (β × Transcript) × Array Claims.Msg → Prop)
    (hf : ∀ s, P (((f s).run (T n)).go (l ++ ((List.range n).map B).toArray))) :
    P (((forIn [:n] s₀ body >>= f).run t₀).go l) := by
  obtain ⟨s, hs⟩ := forIn_go_ex body T B L n hbody s₀ l hl
  rw [← hT, bind_go hs]
  exact hf s

/-! ## The ring switch, round by round -/

/-- The transcript after the opening's label. -/
def pcsT (t : Transcript) : Transcript :=
  { t with pending := t.pending ++ header 2 0 "flock-pcs-open-batch-v0".toUTF8.size ++
      padded "flock-pcs-open-batch-v0".toUTF8 }

/-- The transcript of the switch's round `i` before its squeeze: the ring switch's label and `sv i` absorbed. -/
def rsPre (sv : ℕ → Array F128) (i : ℕ) (t : Transcript) : Transcript :=
  let p := t.pending ++ header 2 0 "flock-ring-switch-v0".toUTF8.size ++ padded "flock-ring-switch-v0".toUTF8 ++
    header 3 2 (sv i).size
  { t with pending := (sv i).foldl F128.pushBytes p }

/-- The transcript of the switch's loop from `t₀`: round `i` squeezes `7` coins at body `L + i`. -/
def rsLoopT (sv : ℕ → Array F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Transcript) : ℕ → Transcript
  | 0 => t₀
  | i + 1 => afterSq (rsPre sv i (rsLoopT sv coin L t₀ i)) 2 7 (coin (L + i))

/-- The body the switch's round `i` sends. -/
def rsLoopB (sv : ℕ → Array F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Transcript) (i : ℕ) : Claims.Msg :=
  Claims.toMsg (framed (rsPre sv i (rsLoopT sv coin L t₀ i)) 2 7)

/-- **The switch's bodies** after the pending bytes `P`, of `N` claims: round `i < N` the ring switch's label, the
claim's switched vector and a squeeze of `7`, the first after `P` and the opening's label; round `N` `γ`'s squeeze. -/
def rsBody (sv : ℕ → Array F128) (N : ℕ) (P : List UInt8) (i : ℕ) : Claims.Msg :=
  if i < N then
    (if i = 0 then P ++ (Frame.label "flock-pcs-open-batch-v0".toUTF8).enc else []) ++
      encs [.label "flock-ring-switch-v0".toUTF8, .vec (sv i).toList] ++ (Frame.sq 2 7).enc
  else (Frame.sq 2 N).enc

theorem rsLoopB_eq (sv : ℕ → Array F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Transcript) (i : ℕ) :
    rsLoopB sv coin L t₀ i = (rsLoopT sv coin L t₀ i).pending.data.toList ++
      encs [.label "flock-ring-switch-v0".toUTF8, .vec (sv i).toList] ++ (Frame.sq 2 7).enc := by
  simp only [rsLoopB, Claims.toMsg, framed_enc, rsPre, absorbSlice_enc, label_enc, encs, List.flatMap_cons,
    List.flatMap_nil, List.append_nil, List.append_assoc]

theorem rsLoopB_map (sv : ℕ → Array F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t : Transcript) (N : ℕ) :
    (List.range N).map (rsLoopB sv coin L (pcsT t)) = (List.range N).map (rsBody sv N t.pending.data.toList) := by
  refine List.map_congr_left fun i hi => ?_
  rw [List.mem_range] at hi
  rw [rsLoopB_eq, rsBody, if_pos hi]
  cases i with
  | zero =>
    rw [if_pos rfl]
    simp only [rsLoopT, pcsT, label_enc, List.append_assoc]
  | succ j =>
    rw [if_neg (Nat.succ_ne_zero j)]
    simp only [rsLoopT, afterSq, empty_toList, List.nil_append]

theorem rsLast (sv : ℕ → Array F128) (coin : ℕ → ℕ → F128) (L : ℕ) (t₀ : Transcript) {N : ℕ} (hN : N ≠ 0)
    (P : List UInt8) : Claims.toMsg (framed (rsLoopT sv coin L t₀ N) 2 N) = rsBody sv N P N := by
  obtain ⟨j, rfl⟩ : ∃ j, N = j + 1 := ⟨N - 1, by omega⟩
  rw [rsBody, if_neg (lt_irrefl _)]
  simp only [Claims.toMsg, framed_enc, rsLoopT, afterSq, empty_toList, List.nil_append]

/-- **The cut ring switch, answered by `asksCut` with room for its asks**: from `l`, with `claims.size + 1` asks before
round `n`, it returns, having sent `rsBody` after `t`'s pending bytes, and leaves nothing pending. -/
theorem ringSwitchCut_bodies (coin : ℕ → ℕ → F128) (n : ℕ) (sv : ℕ → Array F128)
    (claims : Array _root_.Flock.Claim) (hN : claims.size ≠ 0) (t : Transcript) (l : Array Claims.Msg)
    (hl : l.size + claims.size + 1 ≤ n) :
    ∃ res t' L', ((ringSwitchCut (asksCut coin n) sv claims).run t).go l = (.ok (res, t'), L') ∧
      L'.toList = l.toList ++ (List.range (claims.size + 1)).map (rsBody sv claims.size t.pending.data.toList) ∧
      t'.pending = ByteArray.empty := by
  unfold ringSwitchCut
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  try dsimp only
  refine forIn_bind_go _ (rsLoopT sv coin l.size (pcsT t)) (rsLoopB sv coin l.size (pcsT t)) l.size claims.size
    rfl ?hbody _ _ l rfl
    (fun r => ∃ res t' L', r = (.ok (res, t'), L') ∧
      L'.toList = l.toList ++ (List.range (claims.size + 1)).map (rsBody sv claims.size t.pending.data.toList) ∧
      t'.pending = ByteArray.empty) ?_
  case hbody =>
    intro i hi s l' hl'
    try dsimp only
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
    rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbSlice_eq _ _) _)]
    rw [bind_go (sampleVecP_cut_ok coin (by decide) _ (by omega))]
    rw [hl']
    exact ⟨_, rfl⟩
  intro s
  try dsimp only
  rw [bind_go (sampleVecP_cut_ok coin hN _ (by simp; omega))]
  refine ⟨_, _, _, rfl, ?_, rfl⟩
  rw [Array.toList_push, Array.toList_append, List.toList_toArray, rsLoopB_map, List.range_succ, List.map_append,
    List.map_cons, List.map_nil, List.append_assoc]
  rw [rsLast sv coin l.size (pcsT t) hN t.pending.data.toList]

/-! ## The PIOP's last transcript -/

section Piop

open _root_.Flock.Firewall.Drive (Step OpenM)

/-- **A program that ends with nothing pending**: wherever it returns, its transcript holds no pending bytes. -/
def EndE {α : Type} (p : TMW OpenM α) : Prop :=
  ∀ t got r, (p.run t).run got = Step.done r → r.1.2.pending = ByteArray.empty

theorem step_done_inv {α β : Type} {s : Step α} {f : α → Step β} {b : β} (h : s >>= f = .done b) :
    ∃ a, s = .done a ∧ f a = .done b := by
  cases s with
  | done a => exact ⟨a, rfl, h⟩
  | fail e => cases h
  | ask c n k => cases h

theorem endE_bind {α β : Type} {p : TMW OpenM α} {g : α → TMW OpenM β} (hg : ∀ a, EndE (g a)) :
    EndE (p >>= g) := by
  intro t got r h
  rw [StateT.run_bind, StateT.run_bind] at h
  obtain ⟨x, -, hx⟩ := step_done_inv h
  exact hg x.1.1 x.1.2 x.2 r hx

theorem endE_ret {α β : Type} {p : TMW OpenM α} {g : α → TMW OpenM β} (hp : EndE p)
    (hg : ∀ a, ∃ b, g a = pure b) : EndE (p >>= g) := by
  intro t got r h
  rw [StateT.run_bind, StateT.run_bind] at h
  obtain ⟨x, hx, hr⟩ := step_done_inv h
  obtain ⟨b, hb⟩ := hg x.1.1
  have e : ((g x.1.1).run x.1.2).run x.2 = Step.done r := hr
  rw [hb] at e
  have e' : ((b, x.1.2), x.2) = r := Step.done.inj e
  subst e'
  exact hp t got x hx

/-- The transcript after a squeeze of `n` coins of `kind` from `t`, answered by `coin`. -/
def sqT (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) : Transcript :=
  { t with
    pending := ByteArray.empty
    k := t.k + 1
    rounds := t.rounds.push ⟨n, ByteArray.empty, (Array.range n).map coin, some (framed t kind n)⟩ }

theorem endE_squeezeP (io : _root_.Flock.Firewall.Finish.Asks OpenM) (kind : UInt8) {n : ℕ} (hn : n ≠ 0) :
    EndE (_root_.Flock.Firewall.Finish.squeezeP io kind n) := by
  intro t got r h
  have e : (_root_.Flock.Firewall.Finish.squeezeP io kind n).run t =
      io.round (framed t kind n) >>= fun coin => pure ((Array.range n).map coin, sqT t kind n coin) := by
    simp only [_root_.Flock.Firewall.Finish.squeezeP, StateT.run, hn, ite_false, sqT] <;> rfl
  rw [e, StateT.run_bind] at h
  obtain ⟨x, -, hx⟩ := step_done_inv h
  have e2 : (pure ((Array.range n).map x.1, sqT t kind n x.1) : OpenM (Array F128 × Transcript)).run x.2 =
      Step.done r := hx
  have e3 := Step.done.inj e2
  subst e3
  rfl

theorem endE_sampleP (io : _root_.Flock.Firewall.Finish.Asks OpenM) :
    EndE (_root_.Flock.Firewall.Finish.sampleP io) := by
  unfold _root_.Flock.Firewall.Finish.sampleP
  exact endE_ret (endE_squeezeP io 1 (by decide)) fun _ => ⟨_, rfl⟩

theorem endE_lincheckFw (io : _root_.Flock.Firewall.Finish.Asks OpenM) (c : _root_.Flock.CircuitFold)
    (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (zc : _root_.Flock.ZcOut) :
    EndE (_root_.Flock.Firewall.Finish.lincheckFw io c lay mk zc) := by
  unfold _root_.Flock.Firewall.Finish.lincheckFw
  dsimp only
  refine endE_bind fun _ => ?_
  refine endE_bind fun _ => ?_
  refine endE_bind fun _ => ?_
  split
  all_goals
    refine endE_bind fun _ => ?_
    refine endE_bind fun _ => ?_
    refine endE_bind fun _ => ?_
    exact endE_ret (endE_sampleP io) fun _ => ⟨_, rfl⟩

/-- **A rep's PIOP ends on the lincheck's last squeeze**, nothing pending. -/
theorem endE_repPiop (io : _root_.Flock.Firewall.Finish.Asks OpenM) (st : _root_.Flock.Setup)
    (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray) (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) :
    EndE (_root_.Flock.Firewall.Finish.repPiop io st lay cap root extras mk) := by
  unfold _root_.Flock.Firewall.Finish.repPiop
  refine endE_bind fun a => ?_
  obtain ⟨zp, zc⟩ := a
  exact endE_ret (endE_lincheckFw io _ _ _ _) fun b => by
    obtain ⟨lp, lc⟩ := b
    exact ⟨_, rfl⟩

end Piop

/-! ## The rounds' shapes -/

theorem plain_rs (sv : ℕ → Array F128) (i : ℕ) (hs : (sv i).size < 2 ^ 64) :
    ∀ f ∈ [Frame.label "flock-ring-switch-v0".toUTF8, .vec (sv i).toList], Quiet.Plain f := by
  intro f hf
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl
  · refine ⟨show "flock-ring-switch-v0".toUTF8.size < 2 ^ 64 by decide, fun b e => ?_⟩
    cases e
    decide
  · refine ⟨show (sv i).toList.length < 2 ^ 64 by simpa using hs, fun b e => ?_⟩
    cases e

/-- **The switch's first round from empty pending bytes is the opening's.** -/
theorem pcsRound_rs (sv : ℕ → Array F128) {N : ℕ} (hN : N ≠ 0) (hs : (sv 0).size < 2 ^ 64) {c : ByteArray}
    (hc : c.data.toList = rsBody sv N [] 0) : Flags.PcsRound (c, 7) := by
  refine ⟨[.label "flock-ring-switch-v0".toUTF8, .vec (sv 0).toList], 2, ?_, plain_rs sv 0 hs, .inl rfl,
    show (7 : ℕ) ≠ 0 by decide, show (7 : ℕ) < 2 ^ 64 by decide⟩
  show c.data.toList = encs (.label LABEL_PCS.toUTF8 :: _) ++ (Frame.sq 2 7).enc
  rw [hc, rsBody, if_pos (Nat.pos_of_ne_zero hN), if_pos rfl]
  simp only [LABEL_PCS, encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append,
    List.append_assoc]

/-- **The switch's later rounds are quiet.** -/
theorem qRound_rs (sv : ℕ → Array F128) {N i : ℕ} (hi : i ≠ 0) (hiN : i < N) (hs : (sv i).size < 2 ^ 64)
    {c : ByteArray} (hc : c.data.toList = rsBody sv N [] i) : Quiet.QRound (c, 7) := by
  refine ⟨[.label "flock-ring-switch-v0".toUTF8, .vec (sv i).toList], 2, ?_, plain_rs sv i hs, .inl rfl,
    show (7 : ℕ) ≠ 0 by decide, show (7 : ℕ) < 2 ^ 64 by decide⟩
  show c.data.toList = encs _ ++ (Frame.sq 2 7).enc
  rw [hc, rsBody, if_pos hiN, if_neg hi, List.nil_append]

/-- **`γ`'s squeeze is quiet.** -/
theorem qRound_gamma (sv : ℕ → Array F128) {N : ℕ} (hN : N ≠ 0) (hN' : N < 2 ^ 64) {c : ByteArray}
    (hc : c.data.toList = rsBody sv N [] N) : Quiet.QRound (c, N) := by
  refine ⟨[], 2, ?_, ?_, .inl rfl, hN, hN'⟩
  swap
  · intro f hf
    cases hf
  show c.data.toList = [] ++ (Frame.sq 2 N).enc
  rw [hc, rsBody, if_neg (lt_irrefl _), List.nil_append]

end Claims.ZKServer.Switch
