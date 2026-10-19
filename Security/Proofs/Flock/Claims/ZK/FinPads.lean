import Proofs.Flock.Claims.ZK.FinExe
import Proofs.Flock.Claims.ZK.Round1Root

/-!
# The pads' opened columns and salts read from the views are the session's (`repOpenRest_pads`)
-/

namespace FlockComplete

open Flock

/-- **What every returning run of `m` satisfies**: `P` at its value, its last transcript and its last history. -/
def RunPost {α : Type} (m : TMW PS α) (P : α → Transcript → Array Claims.Msg → Prop) : Prop :=
  ∀ s l a t l', (m.run s).go l = (.ok (a, t), l') → P a t l'

namespace RunPost

variable {α β : Type} {P : β → Transcript → Array Claims.Msg → Prop}

theorem bind {x : TMW PS α} {f : α → TMW PS β} (h : ∀ a, RunPost (f a) P) : RunPost (x >>= f) P := by
  intro s l b t l' e
  rw [StateT.run_bind] at e
  obtain ⟨⟨a, s'⟩, l₁, -, e₂⟩ := JsonRT.go_bind_ok e
  exact h a s' l₁ b t l' e₂

theorem bind_pure {x : TMW PS α} {g : α → β} (h : RunPost x fun a t l => P (g a) t l) :
    RunPost (x >>= fun a => pure (g a)) P := by
  intro s l b t l' e
  rw [StateT.run_bind] at e
  obtain ⟨⟨a, s'⟩, l₁, e₁, e₂⟩ := JsonRT.go_bind_ok e
  have e₃ : ((pure (g a) : TMW PS β).run s').go l₁ = (.ok (g a, s'), l₁) := rfl
  rw [e₃] at e₂
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₂
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := e₂
  exact h s l a s' l₁ e₁

theorem ite {c : Prop} [Decidable c] {x y : TMW PS β} (hx : RunPost x P) (hy : RunPost y P) :
    RunPost (if c then x else y) P := by
  split
  · exact hx
  · exact hy

theorem throw {e : String} : RunPost (MonadExcept.throw e : TMW PS β) P := by
  intro s l a t l' h
  cases h

theorem mono {x : TMW PS β} {Q : β → Transcript → Array Claims.Msg → Prop} (h : RunPost x Q)
    (hQP : ∀ a t l, Q a t l → P a t l) : RunPost x P :=
  fun s l a t l' e => hQP _ _ _ (h s l a t l' e)

end RunPost

/-- **A draw of `m ≠ 0` coins answered by `asksCut`** returns the coin of its round, the last of its history. -/
theorem sampleVecP_asksCut (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : m ≠ 0) :
    RunPost (sampleVecP (asksCut coin n) m) fun v _ l => v = (Array.range m).map (coin (l.size - 1)) := by
  intro s l v t l' e
  simp only [sampleVecP, squeezeP, StateT.run, hm, ite_false] at e
  obtain ⟨a, l₁, e₁, e₂⟩ := JsonRT.go_bind_ok e
  have e₀ : ((asksCut coin n).round (Transcript.framed s 2 m)).go l =
      (if n ≤ l.size then (MonadExcept.throw "past the openings" : PS (ℕ → F128)) else pure (coin l.size)).go
        (l.push (Claims.toMsg (Transcript.framed s 2 m))) := rfl
  rw [e₀] at e₁
  by_cases hn : n ≤ l.size
  · simp only [hn, ↓reduceIte] at e₁
    cases e₁
  · simp only [hn, ↓reduceIte] at e₁
    simp only [JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at e₁
    obtain ⟨rfl, rfl⟩ := e₁
    simp only [JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at e₂
    obtain ⟨⟨rfl, -⟩, rfl⟩ := e₂
    simp

/-- **The pads positions the inner proof queries** at its last history `l`: its last round's `Q` coins, each mod `n`. -/
def padQs (coin : ℕ → ℕ → F128) (l : Array Claims.Msg) (n : ℕ) : Array ℕ :=
  ((Array.range Zk.Q).map (coin (l.size - 1))).map fun w => w.lo.toNat % n

/-- **An inner proof opens `ic` at the queried positions**: its columns and salts are `ic`'s at `padQs`. -/
def PadsAt (coin : ℕ → ℕ → F128) (ic : InnerCommit) (ip : Zk.InnerProof) (l : Array Claims.Msg) : Prop :=
  ip.opened = (padQs coin l ic.ch.size).map (fun j => (ic.ch[j]!, ic.cm[j]!)) ∧
    ip.salts = (padQs coin l ic.ch.size).map fun j => ic.salts[j]!

theorem padsAt_last (coin : ℕ → ℕ → F128) (n : ℕ) (ic : InnerCommit) {g : Array F128 → Zk.InnerProof}
    (hg : ∀ v, (g v).opened = (v.map fun w => w.lo.toNat % ic.ch.size).map (fun j => (ic.ch[j]!, ic.cm[j]!)) ∧
      (g v).salts = (v.map fun w => w.lo.toNat % ic.ch.size).map fun j => ic.salts[j]!) :
    RunPost (sampleVecP (asksCut coin n) Zk.Q >>= fun v => pure (g v)) fun ip _ l => PadsAt coin ic ip l :=
  RunPost.bind_pure (RunPost.mono (sampleVecP_asksCut coin n (by decide)) fun v _ l hv => by
    subst hv
    exact hg _)

/-- **`innerCut` answered by `asksCut` opens its commitment at the queried positions.** -/
theorem innerCut_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (lay : Zk.PadLayout) (ic : InnerCommit) (finalC : F128)
    (src : RepSrc) (cons : Array (String × Zk.Lin)) :
    RunPost (innerCut (asksCut coin n) lay ic finalC src cons) fun ip _ l => PadsAt coin ic ip l := by
  unfold innerCut
  refine RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ =>
    RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => ?_
  dsimp only
  refine RunPost.ite (RunPost.bind fun _ => ?_) ?_ <;>
    exact RunPost.bind fun _ => padsAt_last coin n ic fun _ => ⟨rfl, rfl⟩

/-- **`innerP` answered by `asksCut` opens its commitment at the queried positions.** -/
theorem innerP_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes PS) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) :
    RunPost (innerP (asksCut coin n) hs lay p ic finalC cons) fun ip _ l => PadsAt coin ic ip l := by
  unfold innerP
  refine RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ =>
    RunPost.bind fun _ => ?_
  dsimp only
  refine RunPost.ite (RunPost.bind fun _ => ?_) ?_ <;>
  · refine RunPost.bind fun _ => RunPost.bind fun _ => RunPost.bind fun _ => ?_
    refine RunPost.ite (RunPost.bind fun _ => ?_) ?_ <;>
      exact RunPost.bind fun _ => padsAt_last coin n ic fun _ => ⟨rfl, rfl⟩

/-- **The rep from the source answered by `asksCut` opens its pads commitment at the queried positions.** -/
theorem repCut_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes PS) (st : Setup) (sch : Schedule) (rep : ℕ)
    (dom : ByteArray) (cm0 : Commit) (extras : Array Claim) (ic : InnerCommit) (src : RepSrc)
    (prior : Array (List F128)) (words : List ℕ) :
    RunPost (repCut (asksCut coin n) hs st sch rep dom cm0 extras ic src prior words)
      fun p _ l => PadsAt coin ic p.inner l := by
  unfold repCut
  exact RunPost.bind fun ⟨_, _, _, _, _⟩ => RunPost.bind fun ⟨_, _, _⟩ => RunPost.bind fun _ =>
    RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ =>
      RunPost.bind_pure (RunPost.mono (innerCut_padsAt coin n _ ic _ src _) fun _ _ _ h => h)

/-- **The session's rep answered by `asksCut` opens its pads commitment at the queried positions.** -/
theorem repPZk_padsAt (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes PS) (st : Setup) (sch : Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) :
    RunPost (repPZk (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words)
      fun p _ l => PadsAt coin ic p.inner l := by
  unfold repPZk
  exact RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ => RunPost.bind fun ⟨_, _, _⟩ =>
    RunPost.bind fun ⟨_, _, _⟩ => RunPost.bind fun _ => RunPost.bind fun ⟨_, _⟩ => RunPost.bind fun _ =>
      RunPost.bind_pure (RunPost.mono (innerP_padsAt coin n hs _ pads ic _ _) fun _ _ _ h => h)

end FlockComplete

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge CoinLayer

/-- **Table `j`'s rep `r` pads column and salt at tree position `p`** at the executable's layout: the codes of
`h ‖ pad_h` and `μ ‖ pad_μ` at `X = p + 1` and the tape's pads salt `p` fitted to 192 bytes (`exeOf_pads`' leaf's
parts), at every coin vector. -/
theorem colPAt_exe (P : Composed.LigProver) {x : Msg} (w : Claims.Flock.Wit)
    (t : Fin Claims.Flock.zkTableCap → ExeCell) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (j : Fin (jOf x))
    (r : Fin 2) (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j)
        r p).1 =
        fOf (_root_.Flock.Zk.codeAt ((FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t)
            j.val r.val (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).h ++
          (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padH) p.val) ∧
      (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j)
        r p).2 =
        fOf (_root_.Flock.Zk.codeAt ((FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t)
            j.val r.val (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).mu ++
          (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padMu) p.val) ∧
      ZkView.saltBytes (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩) =
        FlockComplete.fitBytes 192 (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t).innerSalt j.val r.val
          p.val) := by
  set T := (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t with hT
  set L := _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 with hLy
  set B := ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j with hB
  have hz := fits_zkTab P (zkTab x j)
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hE : (tabC P x j).z.nE = L.hab := by
    show L.len - 4 = L.hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlen : L.len = (tabC P x j).z.nE + 4 := by
    rw [hE]
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
  have hE64 : (tabC P x j).z.nE ≤ 2 ^ 64 := hz.nE
  have hfa : L.fa < L.fb := by simp only [_root_.Flock.Zk.PadLayout.fb]; omega
  have hfb : L.fb < L.hab := Composed.ZerocheckMasked.fb_lt_hab L
  have htx : L.tx = L.hab + 1 := rfl
  have hty : L.ty = L.hab + 2 := rfl
  have htw : L.tw = L.hab + 3 := rfl
  have hhs := FlockComplete.padsOf_size T j.val r.val L
  have hpH : (FlockComplete.padsOf T j.val r.val L).padH.size = _root_.Flock.Zk.K_PAD := by
    simp [FlockComplete.padsOf]
  have hms : (FlockComplete.padsOf T j.val r.val L).mu.size = L.len := by simp [FlockComplete.padsOf]
  have hpM : (FlockComplete.padsOf T j.val r.val L).padMu.size = _root_.Flock.Zk.K_PAD := by
    simp [FlockComplete.padsOf]
  have tH : ∀ k : Fin (2 ^ 64 + 4), fOf (T.h j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.1 r k := fun k => by
    show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.1 r) k.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tM : ∀ k : Fin (2 ^ 64 + 4), fOf (T.mu j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.2.2.1 r k := fun k => by
    show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.2.2.1 r) k.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tPH : ∀ k : Fin _root_.Flock.Zk.K_PAD, fOf (T.padH j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.1 r k :=
    fun k => by
      show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.1 r) k.val) r.val) = _
      rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tPM : ∀ k : Fin _root_.Flock.Zk.K_PAD, fOf (T.padMu j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.2.1 r k :=
    fun k => by
      show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.2.1 r) k.val) r.val) = _
      rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have hpl : Composed.padLenOf (zkTab x j).table (stAt x j) = (tabC P x j).z.nE + 4 := hlen
  set pd := FlockComplete.padsOf T j.val r.val L with hpd
  have hsz : (pd.h ++ pd.padH).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad := by
    rw [Array.size_append, hhs, hpH, hlen]; rfl
  have hszm : (pd.mu ++ pd.padMu).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad := by
    rw [Array.size_append, hms, hpM, hlen]; rfl
  have hget0 : ∀ k, k < L.len → k ≠ L.hab → k ≠ L.tw → pd.h[k]! = T.h j.val r.val k :=
    fun k hk h1 h2 => FlockComplete.padsOf_get T _ _ L k hk h1 h2
  have hget : ∀ k, k < L.len → k ≠ L.hab → k ≠ L.tw → (pd.h ++ pd.padH)[k]! = T.h j.val r.val k :=
    fun k hk h1 h2 => by rw [FlockComplete.get_append_left _ _ (by rw [hhs]; exact hk), hget0 k hk h1 h2]
  have hfaE : L.fa < (tabC P x j).z.nE := by omega
  have hfbE : L.fb < (tabC P x j).z.nE := by omega
  have hEv : ∀ b : Fin 2, (hEmb hz (.inr b)).val = L.hab + 1 + b.val := fun b => by
    show (tabC P x j).z.nE + 1 + b.val = _
    rw [hE]
  have hf : ∀ i : ZK.PadIdx (tabC P x j).z.nE, fOf (pd.h ++ pd.padH)[Composed.padIdx i]! =
      Composed.padH (nE := (tabC P x j).z.nE) ⟨L.fa, hfaE⟩ ⟨L.fb, hfbE⟩ (B.2.2.1 r) i := by
    rintro (k | i)
    · have hk := k.isLt
      show fOf (pd.h ++ pd.padH)[k.val]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl k))
      rw [hget k.val (by omega) (by omega) (by omega)]
      exact tH (hEmb hz (.inl k))
    · fin_cases i
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 1]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0))
        rw [hget ((tabC P x j).z.nE + 1) (by omega) (by omega) (by omega), ← tH, hEv, hE]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 2]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1))
        rw [hget ((tabC P x j).z.nE + 2) (by omega) (by omega) (by omega), ← tH, hEv, hE]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 0]! =
          (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨L.fa, hfaE⟩)) *
            (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨L.fb, hfbE⟩))
        rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), show (tabC P x j).z.nE + 0 = L.hab by omega,
          FlockComplete.padsOf_hab, hget0 L.fa (by omega) (by omega) (by omega),
          hget0 L.fb (by omega) (by omega) (by omega), fOf_mul, tH ⟨L.fa, by omega⟩, tH ⟨L.fb, by omega⟩]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 3]! =
          (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0)) * (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1))
        rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), show (tabC P x j).z.nE + 3 = L.tw by omega,
          FlockComplete.padsOf_tw, hget0 L.tx (by omega) (by omega) (by omega),
          hget0 L.ty (by omega) (by omega) (by omega), fOf_mul, ← tH, ← tH, hEv, hEv]
        rfl
  have hfm : ∀ i : ZK.PadIdx (tabC P x j).z.nE, fOf (pd.mu ++ pd.padMu)[Composed.padIdx i]! = B.2.2.2.2.2.1 r i := by
    intro i
    have hi := Composed.padIdx_lt i
    rw [FlockComplete.get_append_left _ _ (by rw [hms]; omega),
      show pd.mu[Composed.padIdx i]! = T.mu j.val r.val (Composed.padIdx i) from Composed.Zerocheck.get_map_range _ (by omega)]
    exact tM (padEmb' hz i)
  have hg : ∀ k : Fin (tabC P x j).z.nPad, fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.h.size + k.val by rw [hhs, hlen],
      FlockComplete.get_append_right _ _ (by rw [hpH]; exact hk),
      show pd.padH[k.val]! = T.padH j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPH ⟨k.val, hk⟩
  have hgm : ∀ k : Fin (tabC P x j).z.nPad,
      fOf (pd.mu ++ pd.padMu)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.mu.size + k.val by rw [hms, hlen],
      FlockComplete.get_append_right _ _ (by rw [hpM]; exact hk),
      show pd.padMu[k.val]! = T.padMu j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPM ⟨k.val, hk⟩
  have h1 : (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).1 =
      fOf (_root_.Flock.Zk.codeAt (pd.h ++ pd.padH) p.val) := by
    rw [colPAt_tabC, ZkExec.fOf_codeAt _ p.val _ hsz]
    exact Composed.sum_padCode hpl (pd.h ++ pd.padH) (ZkLink.innerPt p.val) _ _ hf hg
  have h2 : (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).2 =
      fOf (_root_.Flock.Zk.codeAt (pd.mu ++ pd.padMu) p.val) := by
    rw [colPAt_tabC, ZkExec.fOf_codeAt _ p.val _ hszm]
    exact Composed.sum_padCode hpl (pd.mu ++ pd.padMu) (ZkLink.innerPt p.val) _ _ hfm hgm
  have hp : p.val < _root_.Flock.Zk.Q + 2 ^ 64 := by
    have h1 := p.isLt
    have h2 : (tabC P x j).z.nQ ≤ _root_.Flock.Zk.Q := hz.nQ
    have h3 : (tabC P x j).z.nPU ≤ 2 ^ 64 := hz.nPU
    omega
  have hs : ZkView.saltBytes (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩) =
      FlockComplete.fitBytes 192 (T.innerSalt j.val r.val p.val) := by
    show ZkView.saltBytes ((t ⟨j.val, hc⟩).2.1.2.1 r (Fin.castLE (Nat.add_le_add hz.nQ hz.nPU) p)) =
      FlockComplete.fitBytes 192 (fAt ByteArray.empty (fun r => fAt ByteArray.empty
        (fun p => ZkView.saltBytes ((cellAt t j.val).2.1.2.1 r p)) p.val) r.val)
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt, fitBytes_of_size (by rw [fAt_lt _ _ hp]; exact saltBytes_size _),
      fAt_lt _ _ hp]
    rfl
  exact ⟨h1, h2, hs⟩

/-- `atP` at an opened position's tree position is its value. -/
theorem atP_sP (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (j : Fin (jOf x))
    (r : Fin 2) {α : Type} (d : α) (f : Fin (tabC P x j).z.nQ → α) (a : Fin (tabC P x j).z.nQ) :
    atP P x cz j r d f (((tabC P x j).T (cz j)).sP r (.inl a)).val = f a := by
  unfold atP
  simp only [Fin.isLt, ↓reduceDIte, Fin.eta, Equiv.symm_apply_apply]

/-- **The views' pads columns and salt at an opened position are the session's pads'**: at the tree position of opened
position `a`, table `i`'s rep `r` source from the views reads the codes of `h ‖ pad_h` and `μ ‖ pad_μ` and the tape's
pads salt fitted to 192 bytes (`colPAt_exe`). -/
theorem srcV_pads (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) {i r : ℕ}
    (hi : i < jOf x) (hr : r < 2) (a : Fin (tabC PE x ⟨i, hi⟩).z.nQ) :
    ((srcV x c w τ i).rep r).ch (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a)).val =
        _root_.Flock.Zk.codeAt ((FlockComplete.padsOf (UE.tapeZk τ) i r
            (_root_.Flock.Zk.PadLayout.new (stAt x ⟨i, hi⟩).m (stAt x ⟨i, hi⟩).c.kLog (zkTab x ⟨i, hi⟩).sch.k0)).h ++
          (FlockComplete.padsOf (UE.tapeZk τ) i r
            (_root_.Flock.Zk.PadLayout.new (stAt x ⟨i, hi⟩).m (stAt x ⟨i, hi⟩).c.kLog (zkTab x ⟨i, hi⟩).sch.k0)).padH)
          (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a)).val ∧
      ((srcV x c w τ i).rep r).cm (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a)).val =
        _root_.Flock.Zk.codeAt ((FlockComplete.padsOf (UE.tapeZk τ) i r
            (_root_.Flock.Zk.PadLayout.new (stAt x ⟨i, hi⟩).m (stAt x ⟨i, hi⟩).c.kLog (zkTab x ⟨i, hi⟩).sch.k0)).mu ++
          (FlockComplete.padsOf (UE.tapeZk τ) i r
            (_root_.Flock.Zk.PadLayout.new (stAt x ⟨i, hi⟩).m (stAt x ⟨i, hi⟩).c.kLog (zkTab x ⟨i, hi⟩).sch.k0)).padMu)
          (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a)).val ∧
      ((srcV x c w τ i).rep r).padSalt (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a)).val =
        FlockComplete.fitBytes 192 ((UE.tapeZk τ).innerSalt i r
          (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a)).val) := by
  obtain ⟨h1, h2, h3⟩ := colPAt_exe PE w τ (czExe PE x c) ⟨i, hi⟩ ⟨r, hr⟩
    (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a))
  have e : (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).colPAt (wtOf PE x w ⟨i, hi⟩) ((UE.tape x τ).1 ⟨i, hi⟩)
      ⟨r, hr⟩ (((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a))) =
      ((viewsE x c w τ ⟨i, hi⟩).ch ⟨r, hr⟩ a, (viewsE x c w τ ⟨i, hi⟩).cm ⟨r, hr⟩ a) := by
    unfold ZkView.Coined.colPAt
    simp only [Equiv.symm_apply_apply]
    rfl
  rw [e] at h1 h2
  have es : (viewsE x c w τ ⟨i, hi⟩).saltsP ⟨r, hr⟩ a = (UE.tape x τ).2.1
      ⟨⟨i, hi⟩, .inr (.inl (⟨r, hr⟩, ((tabC PE x ⟨i, hi⟩).T (czExe PE x c ⟨i, hi⟩)).sP ⟨r, hr⟩ (.inl a)))⟩ := by
    set_option maxRecDepth 8000 in rfl
  rw [← es] at h3
  show ((tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).rep r).ch _ = _ ∧
    ((tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).rep r).cm _ = _ ∧
    ((tabSrcView PE x (czExe PE x c) (viewsE x c w τ) i).rep r).padSalt _ = _
  unfold tabSrcView
  rw [dite_eq_left hi]
  dsimp only
  rw [dite_eq_left hr]
  dsimp only
  rw [atP_sP, atP_sP, atP_sP]
  refine ⟨?_, ?_, ?_⟩
  · exact congrArg FlockLevel3.GF128.val h1
  · exact congrArg FlockLevel3.GF128.val h2
  · exact h3

end Claims.ZKMal.Masked
