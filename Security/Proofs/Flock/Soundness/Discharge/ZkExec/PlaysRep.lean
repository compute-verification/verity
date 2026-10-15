import Proofs.Flock.Soundness.Refine.Opening
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysReplay
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysLig
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysInner
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsFrames

/-!
# `repProg`'s run is a run of the compiled `--zk` rep

`zk_rep_plays`: the plays half of `--zk`'s `rep_refines`, composed phase by phase as the non-ZK session's `rep_refines`
is. The pads root is the rep's first message; `zk_replay_plays` gives the masked zerocheck's and lincheck's runs; the
non-ZK session's `opening_refines` gives the opening's on the two claims at their claim checks and the extras, and
`openingZK_plays` carries it to the masked opening, which recomputes the two claims' values from the switched `s_hat_v`
(`opResult_two`); `zk_ligerito_plays` gives level 0's; `final_c'` is received; and `zk_inner_plays` gives the inner
proof's on the seven rows (the replay's three, `replay_cons_size`, the ring switch's two and the lane phase's two).

A run of a `bind` splits (`plays_bind_inv`), which is how the opening's run is taken apart.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

section Inv

variable {W F : Type} [Inhabited W] {ι : W → F} {α β : Type}

theorem plays_ret_inv {a b : α} {ms : List Msg} {cs : List (Array W)} (h : Plays ι (.ret a) ms cs b) :
    ms = [] ∧ cs = [] ∧ b = a := by
  cases h
  exact ⟨rfl, rfl, rfl⟩

/-- **A run of a `bind` splits** into a run of the first game and a run of the continuation at its result. -/
theorem plays_bind_inv {f : α → Game β} (g : Game α) :
    ∀ {ms : List Msg} {cs : List (Array W)} {b : β}, Plays ι (g.bind f) ms cs b →
      ∃ ms₁ ms₂ cs₁ cs₂ a, ms = ms₁ ++ ms₂ ∧ cs = cs₁ ++ cs₂ ∧ Plays ι g ms₁ cs₁ a ∧ Plays ι (f a) ms₂ cs₂ b := by
  induction g with
  | ret a => intro ms cs b h; exact ⟨[], ms, [], cs, a, rfl, rfl, .ret a, h⟩
  | send M k ih =>
      intro ms cs b h
      rw [bind_send] at h
      cases h with
      | send m h' =>
          obtain ⟨ms₁, ms₂, cs₁, cs₂, a, rfl, rfl, h₁, h₂⟩ := ih m h'
          exact ⟨_ :: ms₁, ms₂, cs₁, cs₂, a, rfl, rfl, .send m h₁, h₂⟩
  | coin C k ih =>
      intro ms cs b h
      rw [bind_coin] at h
      cases h with
      | coinVec c hc h1 h' =>
          obtain ⟨ms₁, ms₂, cs₁, cs₂, a, rfl, rfl, h₁, h₂⟩ := ih _ h'
          exact ⟨ms₁, ms₂, c :: cs₁, cs₂, a, rfl, rfl, .coinVec c hc h1 h₁, h₂⟩
      | coinOne c hc h' =>
          obtain ⟨ms₁, ms₂, cs₁, cs₂, a, rfl, rfl, h₁, h₂⟩ := ih _ h'
          exact ⟨ms₁, ms₂, c :: cs₁, cs₂, a, rfl, rfl, .coinOne c hc h₁, h₂⟩

end Inv

/-! ## The masked opening -/

/-- A ring switch's run returns its claim, and runs the same at any claim. -/
theorem opStep_inv {m : ℕ} {c : Claim 𝔽 m} {ms : List Msg} {cs : List (Array F128)}
    {x : Claim 𝔽 m × (Fin 128 → 𝔽) × (Fin 7 → 𝔽)} (h : Plays fOf (opStep c) ms cs x) :
    ∃ s, x = (c, s) ∧ ∀ c' : Claim 𝔽 m, Plays fOf (opStep c') ms cs (c', s) := by
  simp only [opStep, bind_eq, pure_eq] at h
  obtain ⟨ms₁, ms₂, cs₁, cs₂, s, rfl, rfl, h₁, h⟩ := plays_bind_inv _ h
  obtain ⟨ms₃, ms₄, cs₃, cs₄, r, rfl, rfl, h₃, h⟩ := plays_bind_inv _ h
  obtain ⟨rfl, rfl, rfl⟩ := plays_ret_inv h
  exact ⟨(s, r), rfl, fun c' => Plays.bind' h₁ (Plays.bind' h₃ (Plays.pure _) rfl rfl) rfl rfl⟩

/-- **The masked opening runs where the opening at the claim checks accepts.** A run of `Model.opening` on the two
claims at values `va`, `vc` and the extras that ends in an accepting `op` is a run of `openingZK` on the claims'
points and the extras with the same messages and coins, ending in the same `op`: the opening accepts only when the two
claims' checks are their values (`opResult_two`), which are the values `openingZK` gives them. -/
theorem openingZK_plays {m : ℕ} (hm : 7 ≤ m) (va vc : 𝔽) (a c : Claim 𝔽 m) (extra : List (Claim 𝔽 m))
    {ms : List Msg} {cs : List (Array F128)} {op : OpenOut 𝔽}
    (h : Plays fOf (Model.opening execArith hm (⟨va, a.skip, a.x⟩ :: ⟨vc, c.skip, c.x⟩ :: extra)) ms cs (some op)) :
    ∃ x, Plays fOf (ZkSession.openingZK execArith hm a c extra) ms cs x ∧ x.1 = some op := by
  rw [opening_eq] at h
  obtain ⟨ms₁, ms₂, cs₁, cs₂, sw, rfl, rfl, h₁, h⟩ := plays_bind_inv _ h
  obtain ⟨ms₃, ms₄, cs₃, cs₄, γ, rfl, rfl, h₃, h⟩ := plays_bind_inv _ h
  obtain ⟨rfl, rfl, hop⟩ := plays_ret_inv h
  simp only [List.mapM_cons, bind_eq, pure_eq, bind_assoc, bind_ret] at h₁
  obtain ⟨msx, ms₅, csx, cs₅, x, rfl, rfl, hx, h₁⟩ := plays_bind_inv _ h₁
  obtain ⟨msy, ms₆, csy, cs₆, y, rfl, rfl, hy, h₁⟩ := plays_bind_inv _ h₁
  obtain ⟨msE, ms₇, csE, cs₇, swE, rfl, rfl, hE, h₁⟩ := plays_bind_inv _ h₁
  obtain ⟨rfl, rfl, rfl⟩ := plays_ret_inv h₁
  obtain ⟨x2, rfl, hxa⟩ := opStep_inv hx
  obtain ⟨y2, rfl, hyc⟩ := opStep_inv hy
  rw [ZkSession.opResult_two execArith hm va vc a c extra x2 y2 swE γ] at hop
  split_ifs at hop with hck
  · exact ⟨_, by
      unfold ZkSession.openingZK
      exact Plays.bind' (hxa a) (Plays.bind' (hyc c) (Plays.bind' hE (Plays.bind' h₃ (Plays.pure _) rfl rfl) rfl rfl)
        rfl rfl) (by simp) (by simp), by exact hop.symm⟩

/-! ## The masked replay's constraints -/

/-- The masked replay hands on three constraints: the zerocheck's `c_eval`, its final `a·b`, and the lincheck's
final. -/
theorem replay_cons_size (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) {t t' : Flock.Transcript}
    {rp : Zk.Replayed} (h : Zk.replay st lay p root t = .ok (rp, t')) : rp.cons.size = 3 := by
  unfold Zk.replay at h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨zz, _, _, h⟩ := bind_ok h
  obtain ⟨s, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨_, _, _, h⟩ := bind_ok h
  obtain ⟨α₀, _, _, h⟩ := bind_ok h
  obtain ⟨β₀, _, _, h⟩ := bind_ok h
  obtain ⟨s₂, _, _, h⟩ := bind_ok h
  dsimp only at h
  cases hf : st.fold.folded α₀ β₀ (skipWeights zz) (s.2.extract 0 (st.kLog - 6)) s₂.2 with
  | ok v =>
    rw [hf] at h
    try dsimp only at h
    obtain ⟨combv, _, h₁₇, h'⟩ := bind_ok h
    clear h
    obtain ⟨rfl, rfl⟩ := pure_ok h₁₇
    obtain ⟨-, h⟩ := check_bind h'
    obtain ⟨_, _, _, h⟩ := bind_ok h
    obtain ⟨_, _, _, h⟩ := bind_ok h
    obtain ⟨rfl, rfl⟩ := pure_ok h
    simp
  | error e =>
    rw [hf] at h
    try dsimp only at h
    obtain ⟨_, _, h₁₇, _⟩ := bind_ok h
    cases h₁₇

/-! ## Runs with a property of their result -/

variable {α β : Type}

theorem pq_bind {Q : β → Prop} {g : Game α} {f : α → Game β} {ms₁ ms₂ ms : List Msg}
    {cs₁ cs₂ cs : List (Array F128)} {a : α} (h₁ : Plays fOf g ms₁ cs₁ a)
    (h₂ : ∃ b, Plays fOf (f a) ms₂ cs₂ b ∧ Q b) (hms : ms = ms₁ ++ ms₂) (hcs : cs = cs₁ ++ cs₂) :
    ∃ b, Plays fOf (g >>= f) ms cs b ∧ Q b :=
  let ⟨b, hb, hQ⟩ := h₂
  ⟨b, Plays.bind' h₁ hb hms hcs, hQ⟩

theorem pq_bindQ {Q' : α → Prop} {Q : β → Prop} {g : Game α} {f : α → Game β} {ms₁ ms₂ ms : List Msg}
    {cs₁ cs₂ cs : List (Array F128)} (h₁ : ∃ a, Plays fOf g ms₁ cs₁ a ∧ Q' a)
    (h₂ : ∀ a, Q' a → ∃ b, Plays fOf (f a) ms₂ cs₂ b ∧ Q b) (hms : ms = ms₁ ++ ms₂) (hcs : cs = cs₁ ++ cs₂) :
    ∃ b, Plays fOf (g >>= f) ms cs b ∧ Q b :=
  let ⟨a, ha, hQ'⟩ := h₁
  let ⟨b, hb, hQ⟩ := h₂ a hQ'
  ⟨b, Plays.bind' ha hb hms hcs, hQ⟩

theorem pq_map {Q : β → Prop} {g : Game α} {f : α → β} {ms : List Msg} {cs : List (Array F128)}
    (h : ∃ a, Plays fOf g ms cs a ∧ Q (f a)) : ∃ b, Plays fOf (g.map f) ms cs b ∧ Q b :=
  let ⟨a, ha, hQ⟩ := h
  ⟨f a, Plays.map f ha, hQ⟩

theorem two_append_get! {γ : Type} [Inhabited γ] (a b : γ) (xs : Array γ) :
    (#[a, b] ++ xs)[0]! = a ∧ (#[a, b] ++ xs)[1]! = b ∧ ∀ k < xs.size, (#[a, b] ++ xs)[k + 2]! = xs[k]! := by
  refine ⟨?_, ?_, fun k hk => ?_⟩
  · rw [getElem!_pos _ 0 (by simp)]; simp
  · rw [getElem!_pos _ 1 (by simp; omega)]; simp
  · rw [getElem!_pos _ (k + 2) (by simp; omega), getElem!_pos xs k hk, Array.getElem_append_right (by simp)]
    simp

/-! ## The rep -/

theorem zk_rep_plays : ZkRepPlays := by
  intro mPts Sl Hs T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf hrun hk
  obtain ⟨h25, -, -, hk0, -, hdim, hlv, hα1, -⟩ := zk_schedule_seams _ _ _ hsch
  have hTm : T.S.m = st.m := hseam.hm
  have h7 : 7 ≤ T.S.m := by omega
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨Tb, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok lg =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg', t₄, h₄, h₅⟩ := bind_ok hrun
  clear hrun
  obtain ⟨hlg, ht₄⟩ := pure_ok h₄
  rw [hlg, ht₄] at h₅
  have hwf₀ : RoundsWF (Flock.Transcript.init rounds dom).rounds := hwf
  have hcs3 := replay_cons_size _ _ _ _ h₁
  -- the masked zerocheck and lincheck
  obtain ⟨hr₁, hk₁, hRP⟩ := zk_replay_plays st _ zp.proof zp.inner.root _ t₁ rp T.S hTm.symm hseam.hk.symm
    (by omega) hwf₀ h₁
  obtain ⟨k₁, zc, row, abM, row', hk₁a, hk₁b, hpz, hpl, hc0, hc1⟩ := hRP T.padsA T.tr
  -- the opening, at the two claims' checks and the extras
  rw [← hTm] at h₂
  obtain ⟨g₀, g₁, g₂⟩ := two_append_get! ({ rp.claims[0]! with value := Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]! } : Flock.Claim)
    ({ rp.claims[1]! with value := Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]! } : Flock.Claim) extras
  obtain ⟨hr₂, hk₂, op, hop, -⟩ := opening_refines T.S.m h7 _
    (⟨fOf (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!), abM.skip, abM.x⟩ ::
      ⟨fOf (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!), execArith.lagS zc.z, zc.r⟩ ::
        Model.extraClaims T.S T.ptLocal mPts pts)
    (by rw [size_two_append]; simp; omega)
    (fun k hk => by
      rcases k with _ | _ | k
      · rw [g₀]; exact hc0 _
      · rw [g₁]; exact hc1 _
      · have hk' : k < (Model.extraClaims T.S T.ptLocal mPts pts).length := by simp at hk; omega
        rw [g₂ k (by omega)]
        exact hext k hk')
    (by rw [size_two_append]; omega) (by rw [size_two_append]; omega) _ t₁ t₂ Tb sws (hr₁ ▸ hwf₀) h₂
  rw [size_two_append] at hop
  obtain ⟨xo, hxo, hxo1⟩ := openingZK_plays h7 _ _ abM.pub ⟨0, execArith.lagS zc.z, zc.r⟩ _ hop
  obtain ⟨xo1, xo2⟩ := xo
  obtain rfl : xo1 = some op := hxo1
  -- level 0
  have hwf₂ : RoundsWF t₂.rounds := hr₂ ▸ hr₁ ▸ hwf₀
  obtain ⟨hr₃, hk₃, hLG⟩ := zk_ligerito_plays st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩
    t₂ t₃ rc hdim hlv hα1 hk0 hwf₂ h₃
  have hlig : ∀ {e : ℕ} (P : ZkSession.Pads0 T.np e), e = 2 → ∀ (Tm : 𝔽) (b : ℕ → 𝔽),
      ∃ out : ZkSession.LinK 𝕂 T.np × PendL 𝔽 Dig,
        Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P T.sch T.t T.c₀ r.val Tm b)
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0) (consumed t₂.rounds t₂.k t₃.k) out ∧
        out.2 ≠ PendL.reject := by
    intro e P he Tm b
    subst he
    rw [hseam.hsch, hseam.ht]
    exact hLG P T.c₀ r.val Tm b
  -- the inner proof, on seven rows
  have hwf₃ : RoundsWF t₃.rounds := hr₃ ▸ hwf₂
  obtain ⟨hr₅, hk₅, hIP⟩ := zk_inner_plays _ zp.inner zp.proof.zerocheck.finalC _ t₃ tr
    (by rw [Array.size_append, Array.size_append, hcs3, ligeritoConstraints_size hv]; rfl) hwf₃ h₅
  -- the masked phase
  have hM : ∃ o, Plays fOf (ZkLink.repMaskedZC (D := Dig) execArith T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r.val
        (Model.extraClaims T.S T.ptLocal mPts pts))
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC]))))
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) o ∧ o.2 ≠ PendL.reject := by
    unfold ZkLink.repMaskedZC
    refine pq_map ?_
    refine pq_bind hpz ?_ rfl rfl
    refine pq_map ?_
    refine pq_bind hpl ?_ rfl rfl
    refine pq_map ?_
    unfold ZkLink.restZC
    refine pq_bind hxo ?_ rfl rfl
    obtain ⟨out, hout, hne⟩ := hlig T.pads hseam.he op.T op.b
    refine pq_bind hout ?_ rfl (List.append_nil _).symm
    exact ⟨_, Plays.bind' (Plays.recv _ (fOf zp.proof.zerocheck.finalC)) (Plays.pure _) rfl rfl, hne⟩
  -- the messages and coins, phase by phase
  have hmsgs : restMsgsZ st sch (2 + extras.size) tPad zp =
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC])))) ++
      [⟨𝔽 × 𝔽, (fOf zp.inner.rho, fOf zp.inner.sigma)⟩, ⟨Dig, toDig zp.inner.tauCommit⟩,
        ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) zp.inner.y⟩] := by
    rw [hTm, hseam.hk, hseam.hnp, hseam.hnP]
    simp [restMsgsZ, innerMsgs]
  have hcoins : rounds.toList.map (·.coins) =
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) ++ consumed t₃.rounds t₃.k tr.k := by
    rw [hr₃, hr₂, hr₁]
    simp only [List.append_assoc]
    rw [← consumed_append _ hk₃ hk₅, ← consumed_append _ hk₂ (hk₃.trans hk₅),
      ← consumed_append _ hk₁b (hk₂.trans (hk₃.trans hk₅)),
      ← consumed_append _ hk₁a (hk₁b.trans (hk₂.trans (hk₃.trans hk₅))), hk]
    exact (consumed_all _).symm
  unfold ZkLink.repZCAt ZkLink.repZC
  rw [hseam.hq]
  refine exQ_recv (toDig zp.inner.root) ?_
  refine pq_bindQ hM (fun o ho => ?_) hmsgs hcoins
  obtain ⟨v, hv⟩ := hIP Hs T.dom T.np T.nP T.pos o.1 T.tr (toDig zp.inner.root)
  exact ⟨⟨o.2, v⟩, Plays.bind' hv (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm, ho⟩

end FlockSoundness.Discharge.ZkExec
