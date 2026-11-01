import Proofs.Flock.Claims.ZK.CutSession
import Proofs.Flock.Claims.Complete.Zk.PiopGo

/-!
# A rep's zerocheck from the cut, against the session's (`SimPiop`'s first phase)

`SimPiop` (`Claims.ZKMal.Masked`, `Split`) compares a rep's PIOP read from the cut (`piopCut`) with the session's
(`piopPZk`), both answered by `asksCut`, which throws right after the send at round `n`. Either may throw at any squeeze,
so the two are compared in lockstep (`GoAlike R`): both throw the same error after the same bodies, or both return, at the
same transcript and bodies, values related by `R`.

* **Lockstep** (`GoAlike`): a bind of related programs whose continuations are related from every common return
  (`alike_bind`, `alike_bind_eq` for a shared prefix), a loop whose every step yields related states (`forIn_alike`).
* **`asksCut`'s squeezes**: before round `n` a squeeze sends the framed bytes and returns the coins at `l.size`
  (`squeezeP_cut_ok`, `sampleP_cut_ok`); at or after it, it throws once the bytes are sent (`squeezeP_cut_err`,
  `sampleP_cut_err`); a squeeze that returned was before round `n` (`squeezeP_cut_inv`, `sampleP_cut_inv`).
* **The zerocheck** (`zerocheckCut_alike`): where the cut's masked values at the layout's indices are the plain
  prover's messages at the zerocheck's coins (`zcCoins`) plus their pads, the cut's zerocheck sends what the session's
  sends and returns the session's proof, both at the `z`, `ρ` and `r` of those coins (`ZcOutAt`; the cut's `v_A`,
  `v_B`, `v_C` are zeros).
-/

namespace FlockComplete

open Flock Transcript Replay

/-! ## Lockstep -/

/-- **Two runs in lockstep**: both threw the same error after the same bodies, or both returned at the same transcript
after the same bodies `L`, with values related by `R` at `L`. -/
abbrev GoAlike {α β : Type} (R : α → β → Array Claims.Msg → Prop)
    (x : Except String (α × Transcript) × Array Claims.Msg) (y : Except String (β × Transcript) × Array Claims.Msg) :
    Prop :=
  (∃ e L, x = (.error e, L) ∧ y = (.error e, L)) ∨ ∃ a b t L, x = (.ok (a, t), L) ∧ y = (.ok (b, t), L) ∧ R a b L

theorem bind_go_err {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t : Transcript} {l l' : Array Claims.Msg}
    {e : String} (h : (x.run t).go l = (.error e, l')) : ((x >>= f).run t).go l = (.error e, l') := by
  rw [StateT.run_bind, go_bind, h]

theorem GoAlike.refl {α : Type} (x : Except String (α × Transcript) × Array Claims.Msg) :
    GoAlike (fun a b _ => a = b) x x := by
  rcases x with ⟨r, L⟩
  rcases r with e | ⟨a, t⟩
  · exact .inl ⟨e, L, rfl, rfl⟩
  · exact .inr ⟨a, a, t, L, rfl, rfl, rfl⟩

theorem GoAlike.eq {α : Type} {x y : Except String (α × Transcript) × Array Claims.Msg}
    (h : GoAlike (fun a b _ => a = b) x y) : x = y := by
  rcases h with ⟨e, L, rfl, rfl⟩ | ⟨a, b, t, L, rfl, rfl, hab⟩
  · rfl
  · obtain rfl : a = b := hab
    rfl

theorem alike_pure {α β : Type} {R : α → β → Array Claims.Msg → Prop} {a : α} {b : β} (t : Transcript)
    (l : Array Claims.Msg) (h : R a b l) :
    GoAlike R (((pure a : TMW PS α).run t).go l) (((pure b : TMW PS β).run t).go l) :=
  .inr ⟨a, b, t, l, rfl, rfl, h⟩

/-- **A bind in lockstep**: related programs, and continuations related from every common return. -/
theorem alike_bind {α β γ δ : Type} {R : α → β → Array Claims.Msg → Prop} {S : γ → δ → Array Claims.Msg → Prop}
    {x : TMW PS α} {y : TMW PS β} {f : α → TMW PS γ} {g : β → TMW PS δ} {t : Transcript} {l : Array Claims.Msg}
    (hxy : GoAlike R ((x.run t).go l) ((y.run t).go l))
    (hfg : ∀ a b t' l', (x.run t).go l = (.ok (a, t'), l') → (y.run t).go l = (.ok (b, t'), l') → R a b l' →
      GoAlike S (((f a).run t').go l') (((g b).run t').go l')) :
    GoAlike S (((x >>= f).run t).go l) (((y >>= g).run t).go l) := by
  rcases hxy with ⟨e, L, hx, hy⟩ | ⟨a, b, t', L, hx, hy, hR⟩
  · exact .inl ⟨e, L, bind_go_err hx, bind_go_err hy⟩
  · rw [bind_go hx, bind_go hy]
    exact hfg a b t' L hx hy hR

/-- **A shared step**: the same program, then continuations related from each of its returns. -/
theorem alike_bind_eq {α γ δ : Type} {S : γ → δ → Array Claims.Msg → Prop} {x y : TMW PS α}
    {f : α → TMW PS γ} {g : α → TMW PS δ} {t : Transcript} {l : Array Claims.Msg} (hxy : x = y)
    (hfg : ∀ a t' l', (x.run t).go l = (.ok (a, t'), l') → GoAlike S (((f a).run t').go l') (((g a).run t').go l')) :
    GoAlike S (((x >>= f).run t).go l) (((y >>= g).run t).go l) := by
  subst hxy
  exact alike_bind (GoAlike.refl _) fun a b t' l' hx _ (hab : a = b) => by subst hab; exact hfg a t' l' hx

/-- **A loop in lockstep**: if step `i`, from states related by `I i` with `L + i` bodies sent, yields states related by
`I (i + 1)` after one more body, the loops over `[0:n]` end in states related by `I n` after `L + n` bodies. -/
theorem forIn_alike {σ τ : Type} (body : ℕ → σ → TMW PS (ForInStep σ)) (body' : ℕ → τ → TMW PS (ForInStep τ))
    (I : ℕ → σ → τ → Prop) (L n : ℕ)
    (hb : ∀ i < n, ∀ (s : σ) (s' : τ) (t : Transcript) (l : Array Claims.Msg), l.size = L + i → I i s s' →
      GoAlike (fun r r' l₁ => ∃ u u', r = .yield u ∧ r' = .yield u' ∧ I (i + 1) u u' ∧ l₁.size = L + i + 1)
        (((body i s).run t).go l) (((body' i s').run t).go l))
    {s : σ} {s' : τ} (t : Transcript) (l : Array Claims.Msg) (hl : l.size = L) (h₀ : I 0 s s') :
    GoAlike (fun u u' l₁ => I n u u' ∧ l₁.size = L + n)
      (((forIn [:n] s body).run t).go l) (((forIn [:n] s' body').run t).go l) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (s : σ) (s' : τ) (t : Transcript) (l : Array Claims.Msg), a + k = n → l.size = L + a →
      I a s s' → GoAlike (fun u u' l₁ => I n u u' ∧ l₁.size = L + n)
        (((forIn (List.range' a k) s body).run t).go l) (((forIn (List.range' a k) s' body').run t).go l) from
    key n 0 s s' t l (by omega) (by omega) h₀
  intro k
  induction k with
  | zero =>
    intro a s s' t l ha hl hI
    have e : a = n := by omega
    exact .inr ⟨s, s', t, l, rfl, rfl, e ▸ hI, by omega⟩
  | succ k ih =>
    intro a s s' t l ha hl hI
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine alike_bind (hb a (by omega) s s' t l hl hI) fun r r' t₁ l₁ _ _ hr => ?_
    obtain ⟨u, u', rfl, rfl, hI₁, hl₁⟩ := hr
    exact ih (a + 1) u u' t₁ l₁ (by omega) (by omega) hI₁

/-! ## `asksCut`'s squeezes -/

theorem asksCut_round_go (coin : ℕ → ℕ → F128) (n : ℕ) (c : ByteArray) (l : Array Claims.Msg) :
    ((asksCut coin n).round c).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : PS (ℕ → F128)).go
        (l.push (Claims.toMsg c))) := rfl

theorem squeezeP_cut_ok (coin : ℕ → ℕ → F128) {n : ℕ} (kind : UInt8) {k : ℕ} (hk : k ≠ 0) (t : Transcript)
    {l : Array Claims.Msg} (hl : l.size < n) :
    ((squeezeP (asksCut coin n) kind k).run t).go l =
      (.ok ((Array.range k).map (coin l.size), afterSq t kind k (coin l.size)),
        l.push (Claims.toMsg (framed t kind k))) := by
  simp only [squeezeP, StateT.run, hk, ite_false]
  rw [go_bind, asksCut_round_go, ite_eq_right (show ¬ n ≤ l.size by omega)]
  rfl

theorem squeezeP_cut_err (coin : ℕ → ℕ → F128) {n : ℕ} (kind : UInt8) {k : ℕ} (hk : k ≠ 0) (t : Transcript)
    {l : Array Claims.Msg} (hl : n ≤ l.size) :
    ((squeezeP (asksCut coin n) kind k).run t).go l =
      (.error "past the openings", l.push (Claims.toMsg (framed t kind k))) := by
  simp only [squeezeP, StateT.run, hk, ite_false]
  rw [go_bind, asksCut_round_go, ite_eq_left hl]
  rfl

/-- **A squeeze that returned** was before round `n`, returned the coins at `l.size` and sent the framed bytes. -/
theorem squeezeP_cut_inv (coin : ℕ → ℕ → F128) {n : ℕ} (kind : UInt8) {k : ℕ} (hk : k ≠ 0) {t t' : Transcript}
    {l l' : Array Claims.Msg} {a : Array F128} (h : ((squeezeP (asksCut coin n) kind k).run t).go l = (.ok (a, t'), l')) :
    l.size < n ∧ a = (Array.range k).map (coin l.size) ∧ t' = afterSq t kind k (coin l.size) ∧
      l' = l.push (Claims.toMsg (framed t kind k)) := by
  by_cases hn : n ≤ l.size
  · rw [squeezeP_cut_err coin kind hk t hn] at h
    simp at h
  · rw [squeezeP_cut_ok coin kind hk t (by omega)] at h
    simp only [Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    exact ⟨by omega, rfl, rfl, rfl⟩

theorem sampleVecP_cut_inv (coin : ℕ → ℕ → F128) {n k : ℕ} (hk : k ≠ 0) {t t' : Transcript}
    {l l' : Array Claims.Msg} {a : Array F128} (h : ((sampleVecP (asksCut coin n) k).run t).go l = (.ok (a, t'), l')) :
    l.size < n ∧ a = (Array.range k).map (coin l.size) ∧ t' = afterSq t 2 k (coin l.size) ∧
      l' = l.push (Claims.toMsg (framed t 2 k)) :=
  squeezeP_cut_inv coin 2 hk h

theorem sampleP_cut_ok (coin : ℕ → ℕ → F128) {n : ℕ} (t : Transcript) {l : Array Claims.Msg} (hl : l.size < n) :
    ((sampleP (asksCut coin n)).run t).go l =
      (.ok (coin l.size 0, afterSq t 1 1 (coin l.size)), l.push (Claims.toMsg (framed t 1 1))) := by
  unfold sampleP
  rw [bind_go (squeezeP_cut_ok coin 1 one_ne_zero t hl)]
  show (Except.ok (_, _), _) = _
  simp

theorem sampleP_cut_err (coin : ℕ → ℕ → F128) {n : ℕ} (t : Transcript) {l : Array Claims.Msg} (hl : n ≤ l.size) :
    ((sampleP (asksCut coin n)).run t).go l = (.error "past the openings", l.push (Claims.toMsg (framed t 1 1))) := by
  unfold sampleP
  exact bind_go_err (squeezeP_cut_err coin 1 one_ne_zero t hl)

theorem sampleP_cut_inv (coin : ℕ → ℕ → F128) {n : ℕ} {t t' : Transcript} {l l' : Array Claims.Msg} {a : F128}
    (h : ((sampleP (asksCut coin n)).run t).go l = (.ok (a, t'), l')) :
    l.size < n ∧ a = coin l.size 0 ∧ t' = afterSq t 1 1 (coin l.size) ∧ l' = l.push (Claims.toMsg (framed t 1 1)) := by
  by_cases hn : n ≤ l.size
  · rw [sampleP_cut_err coin t hn] at h
    simp at h
  · rw [sampleP_cut_ok coin t (by omega)] at h
    simp only [Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    exact ⟨by omega, rfl, rfl, rfl⟩

/-- A transcript action sends nothing. -/
theorem liftTM_inv {α : Type} {x : TM α} {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    (h : ((liftTM x : TMW PS α).run t).go l = (.ok (a, t'), l')) : l = l' :=
  congrArg Prod.snd ((liftTM_go x t l).symm.trans h)

/-! ## The zerocheck -/

/-- **The zerocheck's coins at the cut's coins** from round `L` on: `r_skip`'s at `L`, `r_outer`'s at `L + 1`, `z`'s at
`L + 2`, residual round `i`'s at `L + 3 + i`. -/
def zcCoins (coin : ℕ → ℕ → F128) (L : ℕ) : Zc.Challenges :=
  ⟨coin L, coin (L + 1), coin (L + 2), fun i => coin (L + 3 + i)⟩

/-- **A zerocheck's `z`, `ρ` and `r` at the coins `cs`**, those `Zc.out` returns. -/
abbrev ZcOutAt (m : ℕ) (cs : Zc.Challenges) (o : ZcOut) : Prop :=
  o.z = cs.z ∧ o.rho = pushes cs.ρ (m - 6) ∧ o.rRest = cs.r m

/-- **The cut's zerocheck in lockstep with the session's**: answered by `asksCut` from `l`, where the masked values `mk`
at `round1_ab`'s, `round1_c`'s, each residual round's and the finals' pad indices are the plain prover's values at the
coins from `l.size` on (`zcCoins`) plus their pads `h`, the two send the same `3 + (m − 6)` bodies, throw together, and
return the same proof, both with the `z`, `ρ` and `r` of those coins. -/
theorem zerocheckCut_alike (coin : ℕ → ℕ → F128) (n : ℕ) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray)
    (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (mk : ℕ → F128) (t : Transcript) (l : Array Claims.Msg)
    (hab : ∀ i < 64, mk (lay.r1ab i) = (Zc.r1ab tr (m - 6) ((zcCoins coin l.size).r m))[i]! + h[lay.r1ab i]!)
    (hc : ∀ i < 64, mk (lay.r1c i) = (Zc.r1c tr (m - 6) ((zcCoins coin l.size).r m))[i]! + h[lay.r1c i]!)
    (hm0 : ∀ i < m - 6, mk (lay.ml i 0) =
      (Zc.msg tr (m - 6) ((zcCoins coin l.size).r m) (zcCoins coin l.size).z (zcCoins coin l.size).ρ i).1 +
        h[lay.ml i 0]!)
    (hm1 : ∀ i < m - 6, mk (lay.ml i 1) =
      (Zc.msg tr (m - 6) ((zcCoins coin l.size).r m) (zcCoins coin l.size).z (zcCoins coin l.size).ρ i).2 +
        h[lay.ml i 1]!)
    (hfa : mk lay.fa =
      (Zc.tab (m - 6) (zcCoins coin l.size).z (zcCoins coin l.size).ρ tr.az (m - 6))[0]! + h[lay.fa]!)
    (hfb : mk lay.fb =
      (Zc.tab (m - 6) (zcCoins coin l.size).z (zcCoins coin l.size).ρ tr.bz (m - 6))[0]! + h[lay.fb]!)
    (hfc : mk lay.fc =
      dot (lagrange nodesL den6Inv (zcCoins coin l.size).z) (Zc.r1c tr (m - 6) ((zcCoins coin l.size).r m)) +
        h[lay.fc]!) :
    GoAlike (fun (p q : ZerocheckProof × ZcOut) L => p.1 = q.1 ∧ ZcOutAt m (zcCoins coin l.size) p.2 ∧
        ZcOutAt m (zcCoins coin l.size) q.2 ∧ L.size = l.size + 3 + (m - 6))
      (((zerocheckCut (asksCut coin n) m digest cap root lay mk).run t).go l)
      (((zerocheckPZk (asksCut coin n) m digest cap root lay h tr).run t).go l) := by
  have hA : (Array.range 64).map (fun i => mk (lay.r1ab i)) =
      ZcZ.r1ab lay h tr (m - 6) ((zcCoins coin l.size).r m) :=
    Array.map_congr_left fun i hi => hab i (Array.mem_range.mp hi)
  have hC : (Array.range 64).map (fun i => mk (lay.r1c i)) =
      ZcZ.r1c lay h tr (m - 6) ((zcCoins coin l.size).r m) :=
    Array.map_congr_left fun i hi => hc i (Array.mem_range.mp hi)
  unfold zerocheckCut zerocheckPZk
  rw [hA, hC, hfa, hfb, hfc]
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
    (fun i (s : Array (F128 × F128) × Array F128)
        (s' : Array F128 × Array F128 × Array (F128 × F128) × Array F128) =>
      s = s'.2.2 ∧ s.2 = pushes (zcCoins coin l.size).ρ i ∧
        s'.1 = Zc.tab (m - 6) (zcCoins coin l.size).z (zcCoins coin l.size).ρ tr.az i ∧
        s'.2.1 = Zc.tab (m - 6) (zcCoins coin l.size).z (zcCoins coin l.size).ρ tr.bz i)
    (l.size + 3) (m - 6) ?hb _ _ ?hl ?h0) ?tail
  case hl => simp only [Array.size_push]
  case h0 => exact ⟨rfl, rfl, rfl, rfl⟩
  case hb =>
    intro i hi s s' t' l' hl' hI
    obtain ⟨rounds, rho⟩ := s
    obtain ⟨a, b, rounds', rho'⟩ := s'
    dsimp only at hI
    obtain ⟨hs, rfl, rfl, rfl⟩ := hI
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
    try dsimp only
    rw [hm0 i hi, hm1 i hi]
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
    obtain rfl := liftTM_inv e₂
    refine alike_bind_eq rfl fun rr t₃ l₃ e₃ => ?_
    obtain ⟨-, rfl, rfl, rfl⟩ := sampleP_cut_inv coin e₃
    rw [hl']
    exact .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, rfl, rfl, rfl⟩, by simp only [Array.size_push, hl']⟩
  case tail =>
    intro s s' t' l' _ _ hR
    obtain ⟨rounds, rho⟩ := s
    obtain ⟨a, b, rounds', rho'⟩ := s'
    obtain ⟨hI, hsz⟩ := hR
    dsimp only at hI
    obtain ⟨hs, rfl, rfl, rfl⟩ := hI
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hs
    try dsimp only
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
    obtain rfl := liftTM_inv e₂
    exact .inr ⟨_, _, _, _, rfl, rfl, rfl, ⟨rfl, rfl, rfl⟩, ⟨rfl, rfl, rfl⟩, hsz⟩

end FlockComplete
