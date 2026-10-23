import Proofs.Flock.Claims.ZK.FinishSwFw
import Proofs.Flock.Claims.Complete.ZkTables.View

/-!
# The firewall's PIOP's points and switched claims at the coins

`repCut_fw`'s `hlig` reads the firewall's PIOP's switched claims (`basisVals sws`) and the log it leaves (`l₃.size`).
Answered by `asksCut`, those are the coins' alone, whatever the masked values: the zerocheck's `z`, `ρ` and `r` are its
coins' (`ZcOutAt`), the lincheck's `r'` and `r_in` are its coins' (`LcOutAt`), and a ring switch's switched claims are
`Rs.sws` at its coins of claims with its claims' points. So the firewall's are the session's closed form at any trace.

* `zerocheckCut_at`, `lincheckCut_at`: a returned zerocheck's and lincheck's points, and the bodies each sent.
* `ringSwitchCut_sws`: a returned ring switch's switched claims, and the bodies it sent.
* **`piopFw_at`**: the firewall's PIOP's switched claims are `Rs.sws` of the closed-form claims (`Zc.out`, `Lc.out` at
  any trace) at the coins from its base, after `3 + (m − 6) + 3 + (k_log − 6) + (2 + |extras|) + 1` bodies.
-/

namespace FlockComplete

open Flock Transcript Replay

/-- **A returned zerocheck's points are its coins'** (`ZcOutAt`), whatever the masked values, after `3 + (m − 6)`
bodies. -/
theorem zerocheckCut_at (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray)
    (lay : Zk.PadLayout) (mk : ℕ → F128) {t t' : Transcript} {l l' : Array Claims.Msg} {zp : ZerocheckProof}
    {zc : ZcOut} (h : ((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l = (.ok ((zp, zc), t'), l')) :
    ZcOutAt m (zcCoins coin l.size) zc ∧ l'.size = l.size + 3 + (m - 6) := by
  have hA : GoAlike (fun (p q : ZerocheckProof × ZcOut) L => p = q ∧ ZcOutAt m (zcCoins coin l.size) p.2 ∧
        L.size = l.size + 3 + (m - 6))
      (((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l)
      (((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l) := by
    unfold zerocheckCut
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
    obtain rfl := liftTM_inv e₂
    refine alike_bind_eq rfl fun _ t₃ l₃ e₃ => ?_
    obtain rfl := liftTM_inv e₃
    refine alike_bind_eq rfl fun _ t₄ l₄ e₄ => ?_
    obtain rfl := liftTM_inv e₄
    refine alike_bind_eq rfl fun _ t₅ l₅ e₅ => ?_
    obtain rfl := liftTM_inv e₅
    refine alike_bind_eq rfl fun _ t₆ l₆ e₆ => ?_
    obtain rfl := liftTM_inv e₆
    refine alike_bind_eq rfl fun _ t₇ l₇ e₇ => ?_
    obtain ⟨-, -, rfl, rfl⟩ := sampleVecP_cut_inv coin (by decide) e₇
    refine alike_bind_eq rfl fun ro t₈ l₈ e₈ => ?_
    obtain ⟨-, rfl, rfl, rfl⟩ := sampleVecP_cut_inv coin (by omega) e₈
    simp only [Array.size_push]
    refine alike_bind_eq rfl fun _ t₉ l₉ e₉ => ?_
    obtain rfl := liftTM_inv e₉
    refine alike_bind_eq rfl fun _ t₁₀ l₁₀ e₁₀ => ?_
    obtain rfl := liftTM_inv e₁₀
    refine alike_bind_eq rfl fun z t₁₁ l₁₁ e₁₁ => ?_
    obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₁₁
    simp only [Array.size_push]
    refine alike_bind (forIn_alike _ _
      (fun i (s s' : Array (F128 × F128) × Array F128) => s = s' ∧ s.2 = pushes (zcCoins coin l.size).ρ i)
      (l.size + 3) (m - 6) ?hb _ _ ?hl ?h0) ?tail
    case hl => simp only [Array.size_push]
    case h0 => exact ⟨rfl, rfl⟩
    case hb =>
      intro i hi s s' t' l' hl' hI
      obtain ⟨rounds, rho⟩ := s
      obtain ⟨rounds', rho'⟩ := s'
      dsimp only at hI
      obtain ⟨hs, rfl⟩ := hI
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
      try dsimp only
      refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
      obtain rfl := liftTM_inv e₁
      refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
      obtain rfl := liftTM_inv e₂
      refine alike_bind_eq rfl fun rr t₃ l₃ e₃ => ?_
      obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₃
      rw [hl']
      exact .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, rfl⟩, by simp only [Array.size_push, hl']⟩
    case tail =>
      intro s s' t' l' _ _ hR
      obtain ⟨rounds, rho⟩ := s
      obtain ⟨rounds', rho'⟩ := s'
      obtain ⟨hI, hsz⟩ := hR
      dsimp only at hI
      obtain ⟨hs, rfl⟩ := hI
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
      try dsimp only
      refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
      obtain rfl := liftTM_inv e₁
      refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
      obtain rfl := liftTM_inv e₂
      exact .inr ⟨_, _, _, _, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, hsz⟩
  rcases hA with ⟨e, L, hx, -⟩ | ⟨a, b, t₁, L, hx, -, hR⟩
  · rw [h] at hx
    simp at hx
  · rw [h] at hx
    simp only [Prod.mk.injEq, Except.ok.injEq] at hx
    obtain ⟨⟨rfl, -⟩, rfl⟩ := hx
    exact ⟨hR.2.1, hR.2.2⟩

/-- **A returned lincheck's points are its coins'** (`LcOutAt`), whatever the masked values, after `3 + (k_log − 6)`
bodies. -/
theorem lincheckCut_at (coin : ℕ → ℕ → F128) (n : ℕ) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128)
    (zc : ZcOut) {t t' : Transcript} {l l' : Array Claims.Msg} {lp : LincheckProof} {lc : LcOut}
    (h : ((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l = (.ok ((lp, lc), t'), l')) :
    LcOutAt (c.kLog - 6) (lcCoins coin l.size (c.kLog - 6)) lc ∧ l'.size = l.size + 3 + (c.kLog - 6) := by
  have hA : GoAlike (fun (p q : LincheckProof × LcOut) L => p = q ∧
        LcOutAt (c.kLog - 6) (lcCoins coin l.size (c.kLog - 6)) p.2 ∧ L.size = l.size + 3 + (c.kLog - 6))
      (((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l)
      (((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l) := by
    unfold lincheckCut
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind_eq rfl fun alpha t₂ l₂ e₂ => ?_
    obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₂
    refine alike_bind_eq rfl fun beta t₃ l₃ e₃ => ?_
    obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₃
    simp only [Array.size_push]
    rcases hf : c.fold (coin l.size 0) (skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) with err | full
    · exact .inl ⟨err, _, bind_go_err rfl, bind_go_err rfl⟩
    dsimp only
    refine alike_bind_eq rfl fun _ t₄ l₄ e₄ => ?_
    obtain ⟨rfl, rfl, rfl⟩ := pure_inv e₄
    refine alike_bind (forIn_alike _ _
      (fun j (s s' : Array (F128 × F128) × Array F128) =>
        s = s' ∧ s.2 = pushes (lcCoins coin l.size (c.kLog - 6)).τ j)
      (l.size + 2) (c.kLog - 6) ?hb _ _ ?hl ?h0) ?tail
    case hl => simp only [Array.size_push]
    case h0 => exact ⟨rfl, rfl⟩
    case hb =>
      intro j hj s s' t' l' hl' hI
      obtain ⟨rounds, ts⟩ := s
      obtain ⟨rounds', ts'⟩ := s'
      dsimp only at hI
      obtain ⟨hs, rfl⟩ := hI
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
      try dsimp only
      refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
      obtain rfl := liftTM_inv e₁
      refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
      obtain rfl := liftTM_inv e₂
      refine alike_bind_eq rfl fun tt t₃ l₃ e₃ => ?_
      obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₃
      rw [hl']
      exact .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, rfl⟩, by simp only [Array.size_push, hl']⟩
    case tail =>
      intro s s' t' l' _ _ hR
      obtain ⟨rounds, ts⟩ := s
      obtain ⟨rounds', ts'⟩ := s'
      obtain ⟨hI, hsz⟩ := hR
      dsimp only at hI
      obtain ⟨hs, rfl⟩ := hI
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
      try dsimp only
      refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
      obtain rfl := liftTM_inv e₁
      refine alike_bind_eq rfl fun rP t₂ l₂ e₂ => ?_
      obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₂
      rw [hsz]
      exact .inr ⟨_, _, _, _, rfl, rfl, rfl, ⟨rfl, rfl⟩, by simp only [Array.size_push]; omega⟩
  rcases hA with ⟨e, L, hx, -⟩ | ⟨a, b, t₁, L, hx, -, hR⟩
  · rw [h] at hx
    simp at hx
  · rw [h] at hx
    simp only [Prod.mk.injEq, Except.ok.injEq] at hx
    obtain ⟨⟨rfl, -⟩, rfl⟩ := hx
    exact ⟨hR.2.1, hR.2.2⟩

/-- **A returned ring switch's switched claims are `Rs.sws` at its coins**: claim `k`'s `r''` at `l.size + k`, the
batching `γ` at `l.size + |claims|`, whatever the switched vectors, after `|claims| + 1` bodies, the last before the
cut's stop `n`. -/
theorem ringSwitchCut_sws (coin : ℕ → ℕ → F128) (n : ℕ) (sv : ℕ → Array F128) (claims : Array Claim)
    (hc : claims.size ≠ 0) {t t' : Transcript} {l l' : Array Claims.Msg} {ss : Array (Array F128)} {T : F128}
    {sws : Array Switched}
    (h : ((ringSwitchCut (asksCut coin n) sv claims).run t).go l = (.ok ((ss, T, sws), t'), l')) :
    sws = Rs.sws claims ⟨fun k => coin (l.size + k), coin (l.size + claims.size)⟩ ∧
      l'.size = l.size + claims.size + 1 ∧ l'.size ≤ n := by
  have hA : GoAlike (fun (a b : Array (Array F128) × F128 × Array Switched) L => a = b ∧
        a.2.2 = Rs.sws claims ⟨fun k => coin (l.size + k), coin (l.size + claims.size)⟩ ∧
        L.size = l.size + claims.size + 1 ∧ L.size ≤ n)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l) := by
    unfold ringSwitchCut
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind (forIn_alike _ _
      (fun k (s s' : Array (Array F128) × Array F128 × Array (Array F128 × Array F128)) =>
        s = s' ∧ s.2.2 = pushes (fun k => ((Array.range 7).map (coin (l.size + k)),
          claims[k]!.x.extract 1 claims[k]!.x.size)) k)
      l.size claims.size ?hb _ _ ?hl ?h0) ?tail
    case hl => exact rfl
    case h0 => exact ⟨rfl, rfl⟩
    case hb =>
      intro k hk s s' t' l' hl' hI
      obtain ⟨rfl, hpc⟩ := hI
      obtain ⟨ss, betas, perClaim⟩ := s
      try dsimp only
      refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
      obtain rfl := liftTM_inv e₁
      refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
      obtain rfl := liftTM_inv e₂
      refine alike_bind_eq rfl fun rpp t₃ l₃ e₃ => ?_
      obtain ⟨-, rfl, -, rfl⟩ := sampleVecP_cut_inv coin (by decide) e₃
      refine .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, ?_⟩, by simp only [Array.size_push, hl']⟩
      dsimp only at hpc ⊢
      rw [hpc, hl']
      all_goals rfl
    case tail =>
      intro s s' t' l' _ _ hR
      obtain ⟨⟨hs, hpc⟩, hsz⟩ := hR
      obtain rfl : s = s' := hs
      obtain ⟨ss, betas, perClaim⟩ := s
      try dsimp only
      refine alike_bind_eq rfl fun gamma t₁ l₁ e₁ => ?_
      obtain ⟨hlt, rfl, -, rfl⟩ := sampleVecP_cut_inv coin hc e₁
      refine alike_pure _ _ ⟨rfl, ?_, by simp only [Array.size_push, hsz], by simp only [Array.size_push]; omega⟩
      dsimp only at hpc ⊢
      rw [hpc, hsz]
      all_goals rfl
  rcases hA with ⟨e, L, hx, -⟩ | ⟨a, b, t₁, L, hx, -, hR⟩
  · rw [h] at hx
    simp at hx
  · rw [h] at hx
    simp only [Prod.mk.injEq, Except.ok.injEq] at hx
    obtain ⟨⟨rfl, -⟩, rfl⟩ := hx
    exact hR.2

/-- **The firewall's PIOP's switched claims at the coins**: answered by `asksCut` from `l`, where the firewall's PIOP
returns, its switched claims are `Rs.sws` of the closed-form claims at any trace `tr` (the zerocheck's and lincheck's
points are their coins', and a switched claim reads its claim's point alone, `Rs.sws_congr`) at the ring switch's coins,
and it sent `3 + (m − 6) + 3 + (k_log − 6) + (2 + |extras|) + 1` bodies, all before the cut's stop `n`. -/
theorem piopFw_at {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} {lay : Zk.PadLayout} {g : Array F128} {mk : ℕ → F128}
    {cap root : ByteArray} {extras : Array Claim} {t₀ : Transcript} {l : Array Claims.Msg} {zp lp ss T sws t₃ l₃}
    (hm : 14 ≤ st.m)
    (h : ((_root_.Flock.Firewall.Finish.piopFw (asksCut coin n) st lay g mk cap root extras).run t₀).go l =
      (.ok ((zp, lp, ss, T, sws), t₃), l₃)) (tr : Trace) :
    sws = Rs.sws (abcClaims st.m st.kLog (Zc.out st.m (zcCoins coin l.size) tr)
          (Lc.out st.fold st.m tr (Zc.out st.m (zcCoins coin l.size) tr)
            (lcCoins coin (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
        ⟨fun k => coin (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + k),
          coin (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + (2 + extras.size))⟩ ∧
      l₃.size = l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + (2 + extras.size) + 1 ∧ l₃.size ≤ n := by
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, h₃⟩ := _root_.Flock.Firewall.Finish.piopFw_split h
  obtain ⟨hz, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  obtain ⟨hlc, hl₂⟩ := lincheckCut_at coin n _ _ _ _ _ h₂
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = 2 + extras.size := by
    rw [Array.size_append, abcClaims_size]
  obtain ⟨hs, hl₃, hn₃⟩ := ringSwitchCut_sws coin n _ _ (by rw [hsz]; omega) h₃
  rw [hl₂, hl₁, hsz] at hs
  rw [hl₁] at hlc
  refine ⟨hs.trans (Rs.sws_congr (by rw [Array.size_append, Array.size_append, abcClaims_size, abcClaims_size])
    (claims_x (abcClaims_size _ _ _ _) (abcClaims_size _ _ _ _)
      (abc_x hz.2.1 hz.2.2 hlc.2).1 (abc_x hz.2.1 hz.2.2 hlc.2).2) _), ?_, hn₃⟩
  rw [hsz] at hl₃
  omega

end FlockComplete
