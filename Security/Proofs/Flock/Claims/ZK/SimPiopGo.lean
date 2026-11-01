import Proofs.Flock.Claims.ZK.SimZerocheck
import Proofs.Flock.Claims.Complete.Zk.RingSwitch

/-!
# A rep's PIOP from the cut, against the session's (`SimPiop`'s lincheck, ring switch and PIOP)

`SimZerocheck`'s lockstep (`GoAlike`), past the zerocheck:

* **The lincheck** (`lincheckCut_alike`): both lincheck read only their zerocheck's `z` and `ρ` (`lincheckCut_congr`,
  `lincheckPZk_congr`); where the cut's masked values at each round's and `z_partial`'s pad indices are the plain
  prover's at the lincheck's coins (`lcCoins`) plus their pads, the two send the same `3 + (kLog − 6)` bodies, throw
  together (at a squeeze, or where the circuit fold refuses), and return the same proof, both with the `r'` and `r_in`
  of those coins (`LcOutAt`), with at most `n` bodies sent.
* **The ring switch** (`ringSwitchCut_alike`): on claims of one count and the same points, where the cut's switched
  vector `k` is the session's `switchVec` wherever claim `k`'s body is sent (the cut holds no `ŝ` the session doesn't
  send), the two run alike and return the same.
* **The PIOP** (`piopCut_alike`): where a source's masked values and switched vectors are the plain prover's at the
  cut's coins from `l.size` on plus their pads (`PiopSrc`), the cut's PIOP (`piopCut`) and the session's (`piopPZk`)
  return the same after the same bodies, or throw the same error after the same bodies.
-/

namespace FlockComplete

open Flock Transcript Replay

theorem pure_inv {α : Type} {a b : α} {t t' : Transcript} {l l' : Array Claims.Msg}
    (h : ((pure a : TMW PS α).run t).go l = (.ok (b, t'), l')) : a = b ∧ t = t' ∧ l = l' := by
  rw [pure_go] at h
  simp only [Prod.mk.injEq, Except.ok.injEq] at h
  exact ⟨h.1.1, h.1.2, h.2⟩

theorem abcClaims_size (m kLog : ℕ) (zc : ZcOut) (lc : LcOut) : (abcClaims m kLog zc lc).size = 2 := rfl

/-! ## The lincheck -/

/-- **The lincheck's coins at the cut's coins** from round `L` on, over `n` rounds: `α`'s at `L`, `β`'s at `L + 1`,
round `j`'s at `L + 2 + j`, `r'`'s at `L + 2 + n`. -/
def lcCoins (coin : ℕ → ℕ → F128) (L n : ℕ) : Lc.Challenges :=
  ⟨coin L, coin (L + 1), fun j => coin (L + 2 + j), coin (L + 2 + n)⟩

/-- **A lincheck's `r'` and `r_in` at the coins `cs`**, over `n` rounds, those `Lc.out` returns. -/
abbrev LcOutAt (n : ℕ) (cs : Lc.Challenges) (o : LcOut) : Prop :=
  o.rPrime = cs.cf 0 ∧ o.rIn = (pushes cs.τ n).reverse

theorem lincheckCut_congr (io : Asks PS) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (mk : ℕ → F128)
    {zc zc' : ZcOut} (hz : zc.z = zc'.z) (hρ : zc.rho = zc'.rho) :
    lincheckCut io c m lay mk zc = lincheckCut io c m lay mk zc' := by
  unfold lincheckCut
  rw [hz, hρ]

theorem lincheckPZk_congr (io : Asks PS) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) {zc zc' : ZcOut} (hz : zc.z = zc'.z) (hρ : zc.rho = zc'.rho) :
    lincheckPZk io c m lay h tr zc = lincheckPZk io c m lay h tr zc' := by
  unfold lincheckPZk
  rw [hz, hρ]

/-- **The cut's lincheck in lockstep with the session's**: answered by `asksCut` from `l`, after zerocheck outputs with
the `z` and `ρ` of `zo`, where the masked values `mk` at each round's pad indices (wherever the circuit fold at `α`
returns `full`) and at `z_partial`'s are the plain prover's values at the coins from `l.size` on (`lcCoins`) plus their
pads `h`, the two send the same `3 + (kLog − 6)` bodies, throw together, and return the same proof, both with the `r'`
and `r_in` of those coins. -/
theorem lincheckCut_alike (coin : ℕ → ℕ → F128) (n : ℕ) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout)
    (h : Array F128) (tr : Trace) (mk : ℕ → F128) {zc zc' : ZcOut} (zo : ZcOut) (hz : zc.z = zo.z)
    (hρ : zc.rho = zo.rho) (hz' : zc'.z = zo.z) (hρ' : zc'.rho = zo.rho) (t : Transcript) (l : Array Claims.Msg)
    (hq : ∀ full, c.fold (coin l.size 0) (skipWeights zo.z) (zo.rho.extract 0 (c.kLog - 6)) = .ok full →
      ∀ j < c.kLog - 6,
        mk (lay.lc j 0) = (LcZ.msg lay h c m tr zo full (lcCoins coin l.size (c.kLog - 6)) j).1 ∧
        mk (lay.lc j 1) = (LcZ.msg lay h c m tr zo full (lcCoins coin l.size (c.kLog - 6)) j).2)
    (hzp : ∀ i < 64, mk (lay.zp i) =
      (Lc.zrT c m tr zo (lcCoins coin l.size (c.kLog - 6)) (c.kLog - 6))[i]! + h[lay.zp i]!) :
    GoAlike (fun (p q : LincheckProof × LcOut) L => p.1 = q.1 ∧
        LcOutAt (c.kLog - 6) (lcCoins coin l.size (c.kLog - 6)) p.2 ∧
        LcOutAt (c.kLog - 6) (lcCoins coin l.size (c.kLog - 6)) q.2 ∧ L.size = l.size + 3 + (c.kLog - 6) ∧
        L.size ≤ n)
      (((lincheckCut (asksCut coin n) c m lay mk zc).run t).go l)
      (((lincheckPZk (asksCut coin n) c m lay h tr zc').run t).go l) := by
  rw [lincheckCut_congr (asksCut coin n) c m lay mk hz hρ, lincheckPZk_congr (asksCut coin n) c m lay h tr hz' hρ']
  have hZ : (Array.range 64).map (fun i => mk (lay.zp i)) = (Array.range 64).map fun i =>
      (Lc.zrT c m tr zo (lcCoins coin l.size (c.kLog - 6)) (c.kLog - 6))[i]! + h[lay.zp i]! :=
    Array.map_congr_left fun i hi => hzp i (Array.mem_range.mp hi)
  unfold lincheckCut lincheckPZk
  rw [hZ]
  refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
  obtain rfl := liftTM_inv e₁
  refine alike_bind_eq rfl fun alpha t₂ l₂ e₂ => ?_
  obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₂
  refine alike_bind_eq rfl fun beta t₃ l₃ e₃ => ?_
  obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₃
  simp only [Array.size_push]
  rcases hf : c.fold (coin l.size 0) (skipWeights zo.z) (zo.rho.extract 0 (c.kLog - 6)) with err | full
  · exact .inl ⟨err, _, bind_go_err rfl, bind_go_err rfl⟩
  dsimp only
  refine alike_bind_eq rfl fun _ t₄ l₄ e₄ => ?_
  obtain ⟨rfl, rfl, rfl⟩ := pure_inv e₄
  refine alike_bind (forIn_alike _ _
    (fun j (s : Array (F128 × F128) × Array F128)
        (s' : Array F128 × Array F128 × Array (F128 × F128) × Array F128) =>
      s = s'.2.2 ∧ s.2 = pushes (lcCoins coin l.size (c.kLog - 6)).τ j ∧
        s'.1 = Lc.combT c full (lcCoins coin l.size (c.kLog - 6)) j ∧
        s'.2.1 = Lc.zrT c m tr zo (lcCoins coin l.size (c.kLog - 6)) j)
    (l.size + 2) (c.kLog - 6) ?hb _ _ ?hl ?h0) ?tail
  case hl => simp only [Array.size_push]
  case h0 => exact ⟨rfl, rfl, rfl, rfl⟩
  case hb =>
    intro j hj s s' t' l' hl' hI
    obtain ⟨rounds, ts⟩ := s
    obtain ⟨comb, zr, rounds', ts'⟩ := s'
    dsimp only at hI
    obtain ⟨hs, rfl, rfl, rfl⟩ := hI
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
    try dsimp only
    rw [(hq full hf j hj).1, (hq full hf j hj).2]
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
    obtain rfl := liftTM_inv e₂
    refine alike_bind_eq rfl fun tt t₃ l₃ e₃ => ?_
    obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₃
    rw [hl']
    exact .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, rfl, rfl, rfl⟩, by simp only [Array.size_push, hl']⟩
  case tail =>
    intro s s' t' l' _ _ hR
    obtain ⟨rounds, ts⟩ := s
    obtain ⟨comb, zr, rounds', ts'⟩ := s'
    obtain ⟨hI, hsz⟩ := hR
    dsimp only at hI
    obtain ⟨hs, rfl, rfl, rfl⟩ := hI
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
    try dsimp only
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind_eq rfl fun rP t₂ l₂ e₂ => ?_
    obtain ⟨hlt, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₂
    rw [hsz]
    exact .inr ⟨_, _, _, _, rfl, rfl, rfl, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, by simp only [Array.size_push]; omega,
      by simp only [Array.size_push]; omega⟩

/-! ## The ring switch -/

/-- **The cut's ring switch in lockstep with the session's**: on claims of one count and the same points, from at most
`n` bodies sent, where the cut's switched vector `k` is the session's `switchVec` of claim `k` wherever claim `k`'s
body is sent (`l.size + k ≤ n`), the two run alike and return the same. -/
theorem ringSwitchCut_alike (coin : ℕ → ℕ → F128) (n : ℕ) (tr : Trace) (sv : ℕ → Array F128)
    {claims claims' : Array Claim} (hsz : claims.size = claims'.size) (hx : ∀ k : ℕ, claims[k]!.x = claims'[k]!.x)
    (t : Transcript) (l : Array Claims.Msg) (hn : l.size ≤ n)
    (hsv : ∀ k < claims'.size, l.size + k ≤ n → sv k = switchVec tr claims'[k]!) :
    GoAlike (fun a b _ => a = b)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l)
      (((ringSwitchP (asksCut coin n) tr claims').run t).go l) := by
  unfold ringSwitchCut ringSwitchP
  rw [hsz]
  refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
  obtain rfl := liftTM_inv e₁
  refine alike_bind (forIn_alike _ _
    (fun k (s s' : Array (Array F128) × Array F128 × Array (Array F128 × Array F128)) => s = s' ∧ l.size + k ≤ n)
    l.size claims'.size ?hb _ _ ?hl ?h0) ?tail
  case hl => exact rfl
  case h0 => exact ⟨rfl, by omega⟩
  case hb =>
    intro k hk s s' t' l' hl' hI
    obtain ⟨rfl, hkn⟩ := hI
    obtain ⟨ss, betas, perClaim⟩ := s
    try dsimp only
    rw [hsv k hk hkn, hx k]
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
    obtain rfl := liftTM_inv e₂
    refine alike_bind_eq rfl fun rpp t₃ l₃ e₃ => ?_
    obtain ⟨hlt, -, -, rfl⟩ := sampleVecP_cut_inv coin (by decide) e₃
    exact .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, by omega⟩, by simp only [Array.size_push, hl']⟩
  case tail =>
    intro s s' t' l' _ _ hR
    obtain ⟨⟨hs, -⟩, -⟩ := hR
    obtain rfl : s = s' := hs
    obtain ⟨ss, betas, perClaim⟩ := s
    try dsimp only
    exact GoAlike.refl _

/-! ## The PIOP -/

/-- **A source at the cut's coins** from round `L` on: its masked values at the zerocheck's pad indices are the plain
prover's messages at `zcCoins` plus their pads `h`; at the lincheck's, wherever the circuit fold at `α` returns, the plain
prover's at `lcCoins` after the zerocheck's `3 + (m − 6)` rounds plus their pads; and its switched vector `k` is the
plain prover's `switchVec` of claim `k` at those coins wherever claim `k`'s ring-switch body is sent before the stop
`n`, after the zerocheck's and the lincheck's bodies. -/
structure PiopSrc (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace)
    (src : RepSrc) (extras : Array Claim) (L : ℕ) : Prop where
  r1ab : ∀ i < 64, src.masked (lay.r1ab i) = (Zc.r1ab tr (st.m - 6) ((zcCoins coin L).r st.m))[i]! + h[lay.r1ab i]!
  r1c : ∀ i < 64, src.masked (lay.r1c i) = (Zc.r1c tr (st.m - 6) ((zcCoins coin L).r st.m))[i]! + h[lay.r1c i]!
  ml0 : ∀ i < st.m - 6, src.masked (lay.ml i 0) =
    (Zc.msg tr (st.m - 6) ((zcCoins coin L).r st.m) (zcCoins coin L).z (zcCoins coin L).ρ i).1 + h[lay.ml i 0]!
  ml1 : ∀ i < st.m - 6, src.masked (lay.ml i 1) =
    (Zc.msg tr (st.m - 6) ((zcCoins coin L).r st.m) (zcCoins coin L).z (zcCoins coin L).ρ i).2 + h[lay.ml i 1]!
  fa : src.masked lay.fa =
    (Zc.tab (st.m - 6) (zcCoins coin L).z (zcCoins coin L).ρ tr.az (st.m - 6))[0]! + h[lay.fa]!
  fb : src.masked lay.fb =
    (Zc.tab (st.m - 6) (zcCoins coin L).z (zcCoins coin L).ρ tr.bz (st.m - 6))[0]! + h[lay.fb]!
  fc : src.masked lay.fc =
    dot (lagrange nodesL den6Inv (zcCoins coin L).z) (Zc.r1c tr (st.m - 6) ((zcCoins coin L).r st.m)) + h[lay.fc]!
  lc : ∀ full, st.fold.fold (coin (L + 3 + (st.m - 6)) 0) (skipWeights (Zc.out st.m (zcCoins coin L) tr).z)
      ((Zc.out st.m (zcCoins coin L) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full →
    ∀ j < st.fold.kLog - 6,
      src.masked (lay.lc j 0) = (LcZ.msg lay h st.fold st.m tr (Zc.out st.m (zcCoins coin L) tr) full
        (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)) j).1 ∧
      src.masked (lay.lc j 1) = (LcZ.msg lay h st.fold st.m tr (Zc.out st.m (zcCoins coin L) tr) full
        (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)) j).2
  zp : ∀ i < 64, src.masked (lay.zp i) = (Lc.zrT st.fold st.m tr (Zc.out st.m (zcCoins coin L) tr)
      (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6)) (st.fold.kLog - 6))[i]! + h[lay.zp i]!
  sv : ∀ k < (abcClaims st.m st.kLog (Zc.out st.m (zcCoins coin L) tr) (Lc.out st.fold st.m tr
      (Zc.out st.m (zcCoins coin L) tr) (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size,
    L + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + k ≤ n → src.sv k = switchVec tr (abcClaims st.m st.kLog (Zc.out st.m (zcCoins coin L) tr) (Lc.out st.fold st.m tr
      (Zc.out st.m (zcCoins coin L) tr) (lcCoins coin (L + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)[k]!

/-- **The cut's PIOP is the session's**: answered by `asksCut` from `l`, from a source at the cut's coins from `l.size`
on (`PiopSrc`), the cut's zerocheck, lincheck, rank check and ring switch return what the session's do after the same
bodies, or throw the same error after the same bodies. -/
theorem piopCut_alike (coin : ℕ → ℕ → F128) (n : ℕ) (st : Setup) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) (src : RepSrc) (cap root : ByteArray) (extras : Array Claim) (prior : Array (List F128))
    (words : List Nat) (hm : 14 ≤ st.m) (t : Transcript) (l : Array Claims.Msg)
    (hv : PiopSrc coin n st lay h tr src extras l.size) :
    GoAlike (fun a b _ => a = b)
      (((piopCut (asksCut coin n) st lay src cap root extras prior words).run t).go l)
      (((piopPZk (asksCut coin n) st lay h tr cap root extras prior words).run t).go l) := by
  unfold piopCut piopPZk
  refine alike_bind (zerocheckCut_alike coin n hm st.digest cap root lay h tr src.masked t l hv.r1ab hv.r1c hv.ml0
    hv.ml1 hv.fa hv.fb hv.fc) ?_
  rintro ⟨zp, zc⟩ ⟨zp', zc'⟩ t₁ l₁ - - ⟨hp, ⟨hz, hρ, hr⟩, ⟨hz', hρ', hr'⟩, hl₁⟩
  obtain rfl : zp = zp' := hp
  try dsimp only
  refine alike_bind (lincheckCut_alike coin n st.fold st.m lay h tr src.masked (Zc.out st.m (zcCoins coin l.size) tr)
    hz hρ hz' hρ' t₁ l₁ (by rw [hl₁]; exact hv.lc) (by rw [hl₁]; exact hv.zp)) ?_
  rintro ⟨lp, lc⟩ ⟨lp', lc'⟩ t₂ l₂ - - ⟨hp, ⟨-, hi⟩, ⟨-, hi'⟩, hl₂, hn₂⟩
  obtain rfl : lp = lp' := hp
  rw [hl₁] at hi hi'
  try dsimp only
  refine alike_bind_eq rfl fun _ t₃ l₃ e₃ => ?_
  have hs₃ : l₃.size = l₂.size := by rw [(rankCheckP_go e₃).2]
  have hx : ∀ k : ℕ, (abcClaims st.m st.kLog zc lc ++ extras)[k]!.x = (abcClaims st.m st.kLog zc' lc' ++ extras)[k]!.x := by
    obtain ⟨hx₀, hx₁⟩ := abc_x (m := st.m) (kLog := st.kLog) (hρ.trans hρ'.symm) (hr.trans hr'.symm)
      (hi.trans hi'.symm)
    exact claims_x (A := abcClaims st.m st.kLog zc lc) (B := abcClaims st.m st.kLog zc' lc') rfl rfl hx₀ hx₁
  have hsv : ∀ k < (abcClaims st.m st.kLog zc' lc' ++ extras).size, l₃.size + k ≤ n →
      src.sv k = switchVec tr (abcClaims st.m st.kLog zc' lc' ++ extras)[k]! := by
    intro k hk hkn
    obtain ⟨hx₀, hx₁⟩ := abc_x (m := st.m) (kLog := st.kLog) (z₂ := Zc.out st.m (zcCoins coin l.size) tr)
      (l₂ := Lc.out st.fold st.m tr (Zc.out st.m (zcCoins coin l.size) tr)
        (lcCoins coin (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) hρ' hr' hi'
    have hxo := claims_x (A := abcClaims st.m st.kLog zc' lc') (e := extras) rfl rfl hx₀ hx₁
    exact (hv.sv k (by simpa only [Array.size_append, abcClaims_size] using hk) (by omega)).trans
      (switchVec_x (hxo k)).symm
  refine alike_bind (ringSwitchCut_alike coin n tr src.sv (by simp only [Array.size_append, abcClaims_size]) hx
    t₃ l₃ (by omega) hsv) ?_
  intro a b t₄ l₄ _ _ hR
  obtain rfl : a = b := hR
  obtain ⟨ss, T, sws⟩ := a
  try dsimp only
  exact GoAlike.refl _

end FlockComplete
