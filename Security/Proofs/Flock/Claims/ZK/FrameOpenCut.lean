import Proofs.Flock.Claims.ZK.SimLig
import Proofs.Flock.Claims.ZK.FrameOpen

/-!
# Level 0's masked rounds of the opening, cut round by round

The masked Ligerito (`FlockComplete.ligeritoPZk`), answered by `asksCut` from a log `l` and cut `j` rounds on, stops
past the openings having sent exactly level 0's first `j + 1` bodies, for every `j` up to `ρ_1`'s squeeze, at any pads
`h` and grinding nonces: the rep's (`h` its pads) and the firewall's live opening's (`openLive`, zero pads) alike.
Its bodies are the rounds `FrameOpen` releases (`oodFr`, `pairFr`), at the values SimLig's `LigHyp` reads, the coins
those of rounds `l.size`, `l.size + 1`, …:

* **The bodies** (`l0Body`, `l0Msg`, `l0BodyAt`): the opening's binding (the basis label, `T`, level 0's cap,
  `LABEL_OPEN`) after the pending bytes, with `z`'s squeeze; the out-of-domain value and the claim nonce's bytes, with
  `β_0`'s; lane round `i`'s masked message for `i < k_0`, with its challenge's; the pair's sums, with `ρ_0`'s; `ρ_1`'s
  squeeze alone.
* **A run that stops** (`CutStop`, `cutStop_bind`, `cutStop_lbind`, `sent_push`): it threw past the openings, its log
  as a property reads it; a bind after it stops too.
* **The lane loop that never finishes** (`lforIn_stop`): where each step from an invariant either stops or yields into
  it at the next step, and the invariant holds at no step past the loop's, the loop stops.
* **The cut opening** (`ligeritoPZk_cutL0`): cut `j < k_0 + 4` rounds on, the opening stops past the openings having
  sent `l0BodyAt`'s first `j + 1` bodies after `l`.
-/

namespace FlockComplete.LigLock

open Flock Transcript Replay
open FlockSoundness.Refine (Frame encs label_enc absorbBytes_enc absorbSlice_enc absorbF_enc framed_enc)

/-! ## The bodies -/

/-- **Level 0's masked bodies** from the pending bytes `P`: the opening's binding with `z`'s squeeze of `n` coins; the
out-of-domain value `y` and the nonce's bytes `nb` with `β_0`'s; lane round `i`'s message `msg i` with its
challenge's, for `i < k_0`; the pair's sums `e_0`, `e_1` with `ρ_0`'s; `ρ_1`'s squeeze alone. -/
def l0Body (P : List UInt8) (T : F128) (cap : ByteArray) (n k0 : ℕ) (y : F128) (nb : ByteArray)
    (msg : ℕ → F256 × F256) (e0 e1 : F256) (i : ℕ) : Claims.Msg :=
  if i = 0 then
    P ++ encs [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, .bytes cap, .label Zk.LABEL_OPEN.toUTF8] ++
      (Frame.sq 2 n).enc
  else if i = 1 then encs (Claims.ZKServer.oodFr y nb) ++ (Frame.sq 1 1).enc
  else if i < k0 + 2 then encs (Claims.ZKServer.pairFr (msg (i - 2)).1 (msg (i - 2)).2) ++ (Frame.sq 2 2).enc
  else if i = k0 + 2 then encs (Claims.ZKServer.pairFr e0 e1) ++ (Frame.sq 2 2).enc
  else (Frame.sq 2 2).enc

section Body

variable {P : List UInt8} {T : F128} {cap : ByteArray} {n k0 : ℕ} {y : F128} {nb : ByteArray}
  {msg : ℕ → F256 × F256} {e0 e1 : F256}

theorem l0Body_zero : l0Body P T cap n k0 y nb msg e0 e1 0 =
    P ++ encs [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, .bytes cap, .label Zk.LABEL_OPEN.toUTF8] ++
      (Frame.sq 2 n).enc := by
  unfold l0Body
  rw [if_pos (show (0 : ℕ) = 0 from rfl)]

theorem l0Body_one : l0Body P T cap n k0 y nb msg e0 e1 1 =
    encs (Claims.ZKServer.oodFr y nb) ++ (Frame.sq 1 1).enc := by
  unfold l0Body
  rw [if_neg (show ¬ (1 : ℕ) = 0 by omega), if_pos (show (1 : ℕ) = 1 from rfl)]

theorem l0Body_msg {i : ℕ} (hi : i < k0) : l0Body P T cap n k0 y nb msg e0 e1 (i + 2) =
    encs (Claims.ZKServer.pairFr (msg i).1 (msg i).2) ++ (Frame.sq 2 2).enc := by
  unfold l0Body
  rw [if_neg (show ¬ i + 2 = 0 by omega), if_neg (show ¬ i + 2 = 1 by omega),
    if_pos (show i + 2 < k0 + 2 by omega), Nat.add_sub_cancel]

theorem l0Body_e : l0Body P T cap n k0 y nb msg e0 e1 (k0 + 2) =
    encs (Claims.ZKServer.pairFr e0 e1) ++ (Frame.sq 2 2).enc := by
  unfold l0Body
  rw [if_neg (show ¬ k0 + 2 = 0 by omega), if_neg (show ¬ k0 + 2 = 1 by omega),
    if_neg (show ¬ k0 + 2 < k0 + 2 by omega), if_pos (show k0 + 2 = k0 + 2 from rfl)]

theorem l0Body_r1 : l0Body P T cap n k0 y nb msg e0 e1 (k0 + 3) = (Frame.sq 2 2).enc := by
  unfold l0Body
  rw [if_neg (show ¬ k0 + 3 = 0 by omega), if_neg (show ¬ k0 + 3 = 1 by omega),
    if_neg (show ¬ k0 + 3 < k0 + 2 by omega), if_neg (show ¬ k0 + 3 = k0 + 2 by omega)]

end Body

/-- Lane round `i`'s masked message at the coins from round `p₀`. -/
def l0Msg (coin : ℕ → ℕ → F128) (p0 : ℕ) (sch : Schedule) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128)
    (i : ℕ) : F256 × F256 :=
  maskMsg lay h i (roundPoly (tabS coin p0 t i) (wS coin p0 b sch.n i))

/-- **Level 0's masked bodies of `ligeritoPZk` at the coins from round `p₀`**, from the pending bytes `P`: `LigHyp`'s
out-of-domain value, lane messages and pair's sums at the pads `h`, the claim nonce `0` (R4). -/
def l0BodyAt (coin : ℕ → ℕ → F128) (p0 : ℕ) (P : List UInt8) (T : F128) (cap : ByteArray) (sch : Schedule)
    (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep : ℕ) (extra : ℕ → ℕ → F128)
    (_nonce : ℕ → ℕ → UInt64) : ℕ → Claims.Msg :=
  l0Body P T cap sch.n sch.k0 (mleF t (zS coin p0 sch.n) + h[lay.lgOod 0]!) (le64 0)
    (l0Msg coin p0 sch t b lay h)
    (dotK (wS coin p0 b sch.n sch.k0) (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep)))) +
      ⟨h[lay.lgE 0 0]!, h[lay.lgE 0 1]!⟩)
    (dotK (wS coin p0 b sch.n sch.k0) (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + 1)))) +
      ⟨h[lay.lgE 1 0]!, h[lay.lgE 1 1]!⟩)

section BodyAt

variable {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray} {sch : Schedule}
  {t b : Array F128} {lay : Zk.PadLayout} {h : Array F128} {rep : ℕ} {extra : ℕ → ℕ → F128}
  {nonce : ℕ → ℕ → UInt64}

theorem l0BodyAt_zero : l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce 0 =
    P ++ encs [.label "flock-ligerito-basis-f256-split-v0".toUTF8, .one T, .bytes cap, .label Zk.LABEL_OPEN.toUTF8] ++
      (Frame.sq 2 sch.n).enc :=
  l0Body_zero

theorem l0BodyAt_one : l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce 1 =
    encs (Claims.ZKServer.oodFr (mleF t (zS coin p0 sch.n) + h[lay.lgOod 0]!) (le64 0)) ++
      (Frame.sq 1 1).enc :=
  l0Body_one

theorem l0BodyAt_msg {i : ℕ} (hi : i < sch.k0) : l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce (i + 2) =
    encs (Claims.ZKServer.pairFr (l0Msg coin p0 sch t b lay h i).1 (l0Msg coin p0 sch t b lay h i).2) ++
      (Frame.sq 2 2).enc :=
  l0Body_msg hi

theorem l0BodyAt_e : l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce (sch.k0 + 2) =
    encs (Claims.ZKServer.pairFr
      (dotK (wS coin p0 b sch.n sch.k0) (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep)))) +
        ⟨h[lay.lgE 0 0]!, h[lay.lgE 0 1]!⟩)
      (dotK (wS coin p0 b sch.n sch.k0)
          (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + 1)))) +
        ⟨h[lay.lgE 1 0]!, h[lay.lgE 1 1]!⟩)) ++ (Frame.sq 2 2).enc :=
  l0Body_e

theorem l0BodyAt_r1 : l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce (sch.k0 + 3) = (Frame.sq 2 2).enc :=
  l0Body_r1

end BodyAt

/-! ## A run that stops -/

/-- **A run that stopped past the openings**, its log `L` as `Q` reads it. -/
def CutStop {α : Type} (Q : Array Claims.Msg → Prop) (r : Except String α × Array Claims.Msg) : Prop :=
  ∃ L, r = (.error "past the openings", L) ∧ Q L

theorem cutStop_err {α : Type} {Q : Array Claims.Msg → Prop} {r : Except String α × Array Claims.Msg}
    {L : Array Claims.Msg} (h : r = (.error "past the openings", L)) (hQ : Q L) : CutStop Q r :=
  ⟨L, h, hQ⟩

theorem cutStop_bind {α β : Type} {Q : Array Claims.Msg → Prop} {x : TMW PS α} {f : α → TMW PS β} {t : Transcript}
    {l : Array Claims.Msg} (h : CutStop Q ((x.run t).go l)) : CutStop Q (((x >>= f).run t).go l) := by
  obtain ⟨L, hx, hQ⟩ := h
  exact ⟨L, LigLock.bind_go_err hx, hQ⟩

theorem cutStop_lbind {α β : Type} {Q : Array Claims.Msg → Prop} {x : LP PS α} {f : α → LP PS β} {s : LigState}
    {t : Transcript} {l : Array Claims.Msg} (h : CutStop Q (((x.run s).run t).go l)) :
    CutStop Q ((((x >>= f).run s).run t).go l) := by
  rw [StateT.run_bind]
  exact cutStop_bind h

/-- A log that sent bodies `B 0, …, B (i − 1)` after `l`, with `B i` pushed. -/
theorem sent_push (i : ℕ) {k : ℕ} {l L : Array Claims.Msg} {B : ℕ → Claims.Msg} {m : Claims.Msg}
    (h : L.toList = l.toList ++ (List.range i).map B) (hm : m = B i) (hk : k = i + 1) :
    (L.push m).toList = l.toList ++ (List.range k).map B := by
  subst hm hk
  simp [h, List.range_succ]

/-! ## `asksCut`'s squeezes at a log's size -/

theorem sampleVecP_at (coin : ℕ → ℕ → F128) {n : ℕ} (k p : ℕ) (hk : k ≠ 0) (t : Transcript) {l : Array Claims.Msg}
    (hp : l.size = p) (hl : p < n) :
    ((sampleVecP (asksCut coin n) k).run t).go l =
      (.ok ((Array.range k).map (coin p), afterSq t 2 k (coin p)), l.push (Claims.toMsg (framed t 2 k))) := by
  subst hp
  exact squeezeP_cut_ok coin 2 hk t hl

theorem sampleP_at (coin : ℕ → ℕ → F128) {n : ℕ} (p : ℕ) (t : Transcript) {l : Array Claims.Msg} (hp : l.size = p)
    (hl : p < n) :
    ((sampleP (asksCut coin n)).run t).go l =
      (.ok (coin p 0, afterSq t 1 1 (coin p)), l.push (Claims.toMsg (framed t 1 1))) := by
  subst hp
  exact sampleP_cut_ok coin t hl

theorem sampleKP_at (coin : ℕ → ℕ → F128) {n : ℕ} (p : ℕ) (t : Transcript) {l : Array Claims.Msg} (hp : l.size = p)
    (hl : p < n) :
    ((sampleKP (asksCut coin n)).run t).go l =
      (.ok (⟨coin p 0, coin p 1⟩, afterSq t 2 2 (coin p)), l.push (Claims.toMsg (framed t 2 2))) := by
  subst hp
  unfold sampleKP
  rw [bind_go (squeezeP_cut_ok coin 2 two_ne_zero t hl)]
  show (Except.ok (_, _), _) = _
  simp

theorem sampleKP_stop (coin : ℕ → ℕ → F128) {n : ℕ} (t : Transcript) {l : Array Claims.Msg} (hl : n ≤ l.size) :
    ((sampleKP (asksCut coin n)).run t).go l = (.error "past the openings", l.push (Claims.toMsg (framed t 2 2))) := by
  unfold sampleKP
  exact LigLock.bind_go_err (squeezeP_cut_err coin 2 two_ne_zero t hl)

theorem absorbK_ok (v : F256) (t : Transcript) :
    absorbK v t =
      .ok ((), { t with
        pending := (#[v.c0, v.c1] : Array F128).foldl F128.pushBytes
          (t.pending ++ header 3 2 (#[v.c0, v.c1] : Array F128).size) }) := rfl

theorem pow_ok (v : UInt64) (t : Transcript) :
    pow v t = .ok ((), { t with pending := t.pending ++ header 5 0 (le64 v).size ++ padded (le64 v) }) := rfl

theorem powP2_bind_run {γ : Type} (nonce : ℕ → ℕ → UInt64) (f : Unit → LP PS γ) (s : LigState) :
    (powP nonce 2 >>= f).run s =
      (liftT (liftTM (pow 0)) >>= f).run { s with cbn := s.cbn.push 0 } := by
  simp only [powP, bind_assoc, get_bind_run, set_bind_run]

/-! ## The lane loop that never finishes -/

/-- **A loop that stops**: where step `i` from `I i` either stops or yields into `I (i + 1)`, and `I` holds at no step
past the loop's, the loop over `[0:k]` from `I 0` stops. -/
theorem lforIn_stop {σ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (k : ℕ) (Q : Array Claims.Msg → Prop)
    (I : ℕ → σ → LigState → Transcript → Array Claims.Msg → Prop) (hI : ∀ i u s t L, I i u s t L → i < k)
    (hb : ∀ i < k, ∀ u s t L, I i u s t L →
      CutStop Q ((((body i u).run s).run t).go L) ∨
        ∃ v s' t' L', (((body i u).run s).run t).go L = (.ok ((.yield v, s'), t'), L') ∧ I (i + 1) v s' t' L')
    {u : σ} {s : LigState} {t : Transcript} {L : Array Claims.Msg} (h₀ : I 0 u s t L) :
    CutStop Q ((((forIn [:k] u body).run s).run t).go L) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ m a (u : σ) (s : LigState) (t : Transcript) (L : Array Claims.Msg), a + m = k → I a u s t L →
      CutStop Q ((((forIn (List.range' a m) u body).run s).run t).go L) from
    key k 0 u s t L (by omega) h₀
  intro m
  induction m with
  | zero =>
    intro a u s t L ha hIa
    exact absurd (hI a u s t L hIa) (by omega)
  | succ m ih =>
    intro a u s t L ha hIa
    rw [List.range'_succ, List.forIn_cons, StateT.run_bind]
    rcases hb a (hI a u s t L hIa) u s t L hIa with hs | ⟨v, s', t', L', hv, hI'⟩
    · exact cutStop_bind hs
    · rw [bind_go hv]
      exact ih (a + 1) v s' t' L' (by omega) hI'

/-! ## The cut opening -/

/-- **Level 0's masked opening, cut round by round**: answered by `asksCut` from `l` and cut `j < k_0 + 4` rounds on, the
masked Ligerito at any pads `h` and nonces stops past the openings having sent level 0's first `j + 1` bodies
(`l0BodyAt`, at the coins from round `l.size`) after `l`. -/
theorem ligeritoPZk_cutL0 (coin : ℕ → ℕ → F128) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule) (T : F128)
    (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (tr : Transcript)
    (l : Array Claims.Msg) {j : ℕ} (hj : j < sch.k0 + 4) (hk : 1 ≤ sch.k0) (hn : sch.n ≠ 0) :
    CutStop (fun L => L.toList = l.toList ++ (List.range (j + 1)).map
        (l0BodyAt coin l.size tr.pending.data.toList T (flat cm0.cap) sch t b lay h rep extra nonce))
      (((ligeritoPZk (asksCut coin (l.size + j)) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce).run
        tr).go l) := by
  unfold ligeritoPZk
  dsimp only
  rw [bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _),
    bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _),
    bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbBytes_eq _ _) _),
    bind_go (liftTM_ok (_root_.Flock.RecAlgebra.label_eq _ _) _)]
  refine cutStop_bind ?_
  rw [liftT_bind_run]
  -- `z`'s squeeze, round `l.size`
  rcases Nat.eq_zero_or_pos j with rfl | hj0
  · refine cutStop_bind (cutStop_err (squeezeP_cut_err coin 2 hn _ (by simp)) ?_)
    refine sent_push 0 (by simp) ?_ rfl
    rw [l0BodyAt_zero]
    simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
      Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
      List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
    try rfl
  rw [bind_go (sampleVecP_at coin sch.n l.size hn _ rfl (by omega))]
  try dsimp only
  rw [modify_bind_run, liftT_bind_run, bind_go (liftTM_ok (_root_.Flock.RecAlgebra.absorbF_eq _ _) _),
    powP2_bind_run, liftT_bind_run, bind_go (liftTM_ok (pow_ok _ _) _), liftT_bind_run]
  -- `β_0`'s squeeze, round `l.size + 1`
  rcases Nat.lt_or_ge 1 j with hj1 | hj1
  swap
  · obtain rfl : j = 1 := by omega
    refine cutStop_bind (cutStop_err (sampleP_cut_err coin _ (by simp)) ?_)
    refine sent_push 1 (sent_push 0 (by simp) ?_ rfl) ?_ rfl
    · rw [l0BodyAt_zero]
      simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
        Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
        List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
      try rfl
    · rw [l0BodyAt_one]
      simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
        Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
        List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
      try rfl
  rw [bind_go (sampleP_at coin (l.size + 1) _ ?hp1 ?hl1)]
  case hp1 => simp
  case hl1 => omega
  try dsimp only
  rw [modify_bind_run, get_bind_run, sendMsg_bind_run, liftT_bind_run,
    bind_go (liftTM_ok (absorbK_ok _ _) _), liftT_bind_run, bind_go (liftTM_ok (absorbK_ok _ _) _)]
  -- the lane rounds, round `l.size + 2 + i` lane round `i`'s challenge
  refine cutStop_lbind (lforIn_stop _ sch.k0 _
    (fun i _ (s : LigState) (tt : Transcript) (L : Array Claims.Msg) =>
      i < sch.k0 ∧ i + 2 ≤ j ∧ s.tab = tabS coin l.size t i ∧ s.w = wS coin l.size b sch.n i ∧
        tt.pending.data.toList = encs (Claims.ZKServer.pairFr (l0Msg coin l.size sch t b lay h i).1
          (l0Msg coin l.size sch t b lay h i).2) ∧
        L.toList = l.toList ++ (List.range (i + 2)).map
          (l0BodyAt coin l.size tr.pending.data.toList T (flat cm0.cap) sch t b lay h rep extra nonce))
    (fun _ _ _ _ _ hI => hI.1) ?step ?init)
  case init =>
    refine ⟨by omega, by omega, rfl, rfl, ?_, ?_⟩
    · simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
        Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
        List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
      try rfl
    · refine sent_push 1 (sent_push 0 (by simp) ?_ rfl) ?_ rfl
      · rw [l0BodyAt_zero]
        simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
          Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
          List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
        try rfl
      · rw [l0BodyAt_one]
        simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
          Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
          List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
        try rfl
  case step =>
    intro i hi u s tt L hI
    obtain ⟨-, hji, hts, hws, hpend, hL⟩ := hI
    have hLs : L.size = l.size + 2 + i := by
      have := congrArg List.length hL
      simp only [Array.length_toList, List.length_append, List.length_map, List.length_range] at this
      omega
    have hmi : Claims.toMsg (framed tt 2 2) =
        l0BodyAt coin l.size tr.pending.data.toList T (flat cm0.cap) sch t b lay h rep extra nonce (i + 2) := by
      rw [l0BodyAt_msg hi]
      simp only [Claims.toMsg, framed_enc, hpend]
    rcases u with ⟨initCh, rho, zl⟩
    rcases s with ⟨tab, w, msgs, oods, qn, cn, cbn⟩
    dsimp only at hts hws ⊢
    subst hts hws
    rw [liftT_bind_run]
    by_cases hj2 : j = i + 2
    · left
      subst hj2
      refine cutStop_bind (cutStop_err (sampleKP_stop coin tt (by omega)) ?_)
      exact sent_push (i + 2) hL hmi rfl
    rw [bind_go (sampleKP_at coin (l.size + 2 + i) tt hLs (by omega))]
    try dsimp only
    rw [modify_bind_run]
    by_cases hlast : i + 1 = sch.k0
    · -- the last lane round: the pair's sums with `ρ_0`'s squeeze, then `ρ_1`'s
      have hb : (i + 1 == sch.k0) = true := by simpa using hlast
      rw [ite_eq_left_of_eq_true _ _ (eq_true hb), get_bind_run]
      try dsimp only
      rw [liftT_bind_run, bind_go (liftTM_ok (absorbK_ok _ _) _), liftT_bind_run,
        bind_go (liftTM_ok (absorbK_ok _ _) _), liftT_bind_run]
      left
      have hme : ∀ tt' : Transcript, tt'.pending.data.toList = encs (Claims.ZKServer.pairFr
          (dotK (foldK ⟨coin (l.size + 2 + i) 0, coin (l.size + 2 + i) 1⟩ (wS coin l.size b sch.n i))
            (embedF ((Array.range (foldK ⟨coin (l.size + 2 + i) 0, coin (l.size + 2 + i) 1⟩
              (tabS coin l.size t i)).size).map (extra (2 * rep)))) + ⟨h[lay.lgE 0 0]!, h[lay.lgE 0 1]!⟩)
          (dotK (foldK ⟨coin (l.size + 2 + i) 0, coin (l.size + 2 + i) 1⟩ (wS coin l.size b sch.n i))
            (embedF ((Array.range (foldK ⟨coin (l.size + 2 + i) 0, coin (l.size + 2 + i) 1⟩
              (tabS coin l.size t i)).size).map (extra (2 * rep + 1)))) + ⟨h[lay.lgE 1 0]!, h[lay.lgE 1 1]!⟩)) →
          Claims.toMsg (framed tt' 2 2) =
            l0BodyAt coin l.size tr.pending.data.toList T (flat cm0.cap) sch t b lay h rep extra nonce (i + 3) := by
        intro tt' he
        rw [show i + 3 = sch.k0 + 2 by omega, l0BodyAt_e, ← hlast]
        simp only [Claims.toMsg, framed_enc, he]
        try rfl
      by_cases hj3 : j = i + 3
      · subst hj3
        refine cutStop_bind (cutStop_err (sampleKP_stop coin _ (by simp only [Array.size_push]; omega)) ?_)
        refine sent_push (i + 3) (sent_push (i + 2) hL hmi (by omega)) (hme _ ?_) rfl
        simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
          Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
          List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
        try rfl
      rw [bind_go (sampleKP_at coin (l.size + 2 + i + 1) _ ?hp ?hl)]
      case hp => simp only [Array.size_push, hLs]
      case hl => omega
      try dsimp only
      rw [liftT_bind_run]
      obtain rfl : j = i + 4 := by omega
      refine cutStop_bind (cutStop_err (sampleKP_stop coin _ (by simp only [Array.size_push]; omega)) ?_)
      refine sent_push (i + 4) (sent_push (i + 3) (sent_push (i + 2) hL hmi (by omega)) (hme _ ?_)
        (by omega)) ?_ rfl
      · simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
          Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
          List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
        try rfl
      · rw [show i + 4 = sch.k0 + 3 by omega, l0BodyAt_r1]
        simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
          Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
          List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
        try rfl
    · -- a lane round before the last: the next masked message
      have hb : ¬ (i + 1 == sch.k0) = true := by simpa using hlast
      rw [ite_eq_right_of_eq_false _ _ (eq_false hb), get_bind_run, sendMsg_bind_run, liftT_bind_run,
        bind_go (liftTM_ok (absorbK_ok _ _) _), liftT_bind_run, bind_go (liftTM_ok (absorbK_ok _ _) _)]
      right
      refine ⟨_, _, _, _, rfl, by omega, by omega, rfl, rfl, ?_, sent_push (i + 2) hL hmi (by omega)⟩
      simp only [Claims.toMsg, framed_enc, label_enc, absorbF_enc, absorbBytes_enc, absorbSlice_enc, afterSq,
        Claims.ZKServer.empty_toList, List.toList_toArray, Claims.ZKServer.oodFr, Claims.ZKServer.pairFr, encs,
        List.flatMap_cons, List.flatMap_nil, List.append_nil, List.nil_append, List.append_assoc]
      try rfl

end FlockComplete.LigLock
