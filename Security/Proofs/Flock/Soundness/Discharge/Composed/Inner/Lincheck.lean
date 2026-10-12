import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictInner

/-!
# The lincheck is complete on the masked replay (live PROTOCOL.md §4, §10)

`Zk.replay`'s lincheck reads the masked `final_a`, `final_b`, the `k_log − 6` round messages and `z_partial`, each a sent
value plus its pad, as affine forms in the pads (`Zk.Lin`); it gives the "lincheck final" constraint, one of the ten the
inner proof batches, and the `ab` claim's value `w` the ring switch reads. `lcFinal` and `lcAbValue` are that code with
its coins given (`α`, `β`, the round coins `ts`, `r′`, and the zerocheck's `ζ`, `ρ`), and `linVal` reads a form at the
pads `h`.

`lincheck_complete`: for a witness `w` of the statement, when the unmasked `final_a`, `final_b` are the zerocheck's true
`a`, `b` claims at `(ζ, ρ)` and the honest prover sends `lcMsg`'s round messages and the folded columns as `z_partial`,
each under its pad, the constraint vanishes at `h` and the `ab` claim holds for `w`, at every coin: the lincheck neither
divides nor refuses.
-/

namespace FlockSoundness.Discharge.Composed.Lincheck

open Flock FlockLevel3 Model Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## Pads and affine forms -/

/-- Pad `i` of `h`, and `0` past the end, where `Zk.Lin.masked` puts no coefficient either. -/
noncomputable def pad {s : ℕ} (h : Fin s → 𝔽) (i : ℕ) : 𝔽 := if hi : i < s then h ⟨i, hi⟩ else 0

/-- **An executable constraint's value at the pads** `h`: `c + Σ_{i<s} a_i·h_i` in GF(2^128) (`ZkSession.Lin.eval` of
the row it is under `LinRel`). -/
noncomputable def linVal {s : ℕ} (h : Fin s → 𝔽) (l : Zk.Lin) : 𝔽 := fOf l.c + ∑ i : Fin s, fOf l.a[i.val]! * h i

/-! ## `Zk.replay`'s lincheck, its coins given -/

/-- Round `j` of `Zk.replay`'s lincheck loop at the coin `ts[j]`, on the masked messages `rounds[j]`. -/
def lcStep (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (ts : Array F128) (running : Zk.Lin) (j : ℕ) :
    Zk.Lin :=
  let s := lay.len
  let rr := ts[j]!
  let e1 := Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) s
  let einf := Zk.Lin.masked rounds[j]!.2 (lay.lc j 1) s
  let e0 := running.add e1
  let c1 := (e0.add e1).add einf
  ((einf.scale (rr * rr)).add (c1.scale rr)).add e0

/-- `Zk.replay`'s lincheck claim after `n` rounds, from `α·a + b + β` on the masked `final_a`, `final_b`. -/
def lcRunning (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128)) (ts : Array F128)
    (n : ℕ) : Zk.Lin :=
  let s := lay.len
  let a := Zk.Lin.masked fA lay.fa s
  let b := Zk.Lin.masked fB lay.fb s
  (List.range n).foldl (lcStep lay rounds ts) (((a.scale alpha).add b).addConst beta)

/-- `Zk.replay`'s masked `z_partial`. -/
def lcZp (lay : Zk.PadLayout) (zPartial : Array F128) : Array Zk.Lin :=
  (Array.range 64).map fun i => Zk.Lin.masked zPartial[i]! (lay.zp i) lay.len

/-- **`Zk.replay`'s "lincheck final" constraint** after `n` rounds, on the circuit's 64 folded entries `combv`. -/
def lcFinal (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128)) (zPartial ts : Array F128)
    (n : ℕ) (combv : Array F128) : Zk.Lin :=
  (lcRunning lay fA fB alpha beta rounds ts n).add (Zk.Lin.combo combv (lcZp lay zPartial) lay.len)

/-- **The `ab` claim's value** `w` that `Zk.replay` hands the ring switch (`Replayed.abValue`). -/
def lcAbValue (lay : Zk.PadLayout) (zPartial : Array F128) (rPrime : F128) : Zk.Lin :=
  Zk.Lin.combo (skipWeights rPrime) (lcZp lay zPartial) lay.len

/-- The `ab` claim `Zk.replay` hands the opening (`abcClaims`' first, its value left at the masked replay's `0`):
skip weights `L_s(r′)`, point the round coins reversed, then `ρ[k_log − 6 …]`. -/
def lcAbClaim (m kLog : ℕ) (rho ts : Array F128) (rPrime : F128) : Flock.Claim :=
  { value := 0, skip := skipWeights rPrime, x := ts.reverse ++ rho.extract (kLog - 6) (m - 6) }

/-- An executable claim's skip weights and point, read in GF(2^128), with the value `v`. -/
noncomputable def claimOf (m : ℕ) (c : Flock.Claim) (v : 𝔽) : Model.Claim 𝔽 m :=
  ⟨v, fun s => fOf c.skip[s.val]!, fun i => fOf c.x[i.val]!⟩

/-! ## The honest prover -/

/-- The circuit's column vector at the coins (§10, `Model.comb`): `Σ_i L_{i mod 64}(ζ)·eq(ρ_in, ⌊i/64⌋)·(α·A₀[i,c] +
B₀[i,c])`, plus `β` at the pin, with `ρ_in = ρ[.. k_log − 6]`. -/
noncomputable def lcU (S : Statement) (ζ α β : F128) (rho : Array F128) : ℕ → 𝔽 :=
  comb execArith S (fOf ζ) (fOf α) (fOf β) (fun j => fOf rho[j.val]!)

/-- The witness's block-folded columns at `ρ_out = ρ[k_log − 6 ..]` (`hcol`). -/
noncomputable def lcV (S : Statement) (rho : Array F128) (w : Witness S.m) : ℕ → 𝔽 :=
  hcol S (fun j => fOf rho[S.kLog - 6 + j.val]!) w

/-- One round's honest message `(e(1), e(∞))` for `Σ_{i<2H} u_i·v_i`, where
`e(X) = Σ_{i<H} (u_i + X·(u_{i+H} + u_i))·(v_i + X·(v_{i+H} + v_i))`. -/
def roundMsg {F : Type} [Field F] (H : ℕ) (u v : ℕ → F) : F × F :=
  (∑ i ∈ Finset.range H, u (i + H) * v (i + H),
    ∑ i ∈ Finset.range H, (u (i + H) + u i) * (v (i + H) + v i))

/-- **The honest prover's round-`j` message**: `roundMsg` on `u`, `v` of length `len` folded at the first `j` coins. -/
noncomputable def lcMsg (len : ℕ) (u v : ℕ → 𝔽) (ts : List 𝔽) (j : ℕ) : 𝔽 × 𝔽 :=
  roundMsg (len / 2 ^ (j + 1)) (foldAll (ts.take j) len u) (foldAll (ts.take j) len v)

/-! ## The forms' values at the pads -/

theorem masked_size (v : F128) (i s : ℕ) : (Zk.Lin.masked v i s).a.size = s := by
  simp [Zk.Lin.masked]

theorem masked_get (v : F128) (i s k : ℕ) (hk : k < s) :
    (Zk.Lin.masked v i s).a[k]! = if i = k then 1 else 0 := by
  show ((Array.replicate s (0 : F128)).set! i 1)[k]! = _
  rw [Array.set!_eq_setIfInBounds, getElem!_pos _ k (by simpa using hk),
    Array.getElem_setIfInBounds (by simpa using hk)]
  by_cases e : i = k
  · subst e; simp
  · simp [e]

theorem linVal_masked {s : ℕ} (h : Fin s → 𝔽) (v : F128) (i : ℕ) :
    linVal h (Zk.Lin.masked v i s) = fOf v + pad h i := by
  unfold linVal pad
  have hc : (Zk.Lin.masked v i s).c = v := rfl
  rw [hc]
  congr 1
  simp only [masked_get v i s _ (Fin.isLt _)]
  split
  · rename_i hi
    rw [Finset.sum_eq_single ⟨i, hi⟩]
    · simp [fOf_one]
    · intro k _ hk
      rw [ite_eq_right (fun e : i = k.val => absurd (Fin.ext e.symm) hk), fOf_zero, zero_mul]
    · simp
  · rename_i hi
    refine Finset.sum_eq_zero fun k _ => ?_
    rw [ite_eq_right (fun e : i = k.val => absurd (e ▸ k.isLt) hi), fOf_zero, zero_mul]

theorem add_size (x y : Zk.Lin) : (x.add y).a.size = min x.a.size y.a.size := by
  simp [Zk.Lin.add]

theorem linVal_add {s : ℕ} (h : Fin s → 𝔽) (x y : Zk.Lin) (hx : x.a.size = s) (hy : y.a.size = s) :
    linVal h (x.add y) = linVal h x + linVal h y := by
  unfold linVal Zk.Lin.add
  dsimp only
  rw [hx, hy, Nat.min_self, fOf_add]
  have : ∀ i : Fin s, fOf ((Array.range s).map fun i => x.a[i]! + y.a[i]!)[i.val]! * h i =
      fOf x.a[i.val]! * h i + fOf y.a[i.val]! * h i := fun i => by
    rw [getElem!_range_map _ _ _ i.isLt, fOf_add, add_mul]
  simp only [this, Finset.sum_add_distrib]
  ring

theorem scale_size (x : Zk.Lin) (k : F128) : (x.scale k).a.size = x.a.size := by
  simp [Zk.Lin.scale]

theorem linVal_scale {s : ℕ} (h : Fin s → 𝔽) (x : Zk.Lin) (k : F128) :
    linVal h (x.scale k) = linVal h x * fOf k := by
  unfold linVal Zk.Lin.scale
  dsimp only
  have : ∀ i : Fin s, fOf (x.a.map (· * k))[i.val]! * h i = fOf x.a[i.val]! * h i * fOf k := fun i => by
    by_cases hi : i.val < x.a.size
    · rw [getElem!_pos (x.a.map (· * k)) i.val (by simpa using hi), Array.getElem_map,
        getElem!_pos x.a i.val hi, fOf_mul]; ring
    · rw [getElem!_neg (x.a.map (· * k)) i.val (by simpa using hi), getElem!_neg x.a i.val hi]
      show fOf 0 * h i = fOf 0 * h i * fOf k
      rw [fOf_zero]; ring
  rw [Finset.sum_congr rfl fun i _ => this i, fOf_mul, add_mul]
  exact congrArg _ (Finset.sum_mul _ _ _).symm

theorem addConst_size (x : Zk.Lin) (k : F128) : (x.addConst k).a.size = x.a.size := rfl

theorem linVal_addConst {s : ℕ} (h : Fin s → 𝔽) (x : Zk.Lin) (k : F128) :
    linVal h (x.addConst k) = linVal h x + fOf k := by
  unfold linVal Zk.Lin.addConst
  dsimp only
  rw [fOf_add]; ring

theorem sum_combo {F : Type} [Field F] {m s : ℕ} (w c : Fin m → F) (a : Fin m → Fin s → F) (x : Fin s → F) :
    ∑ k, w k * c k + ∑ i, (∑ k, w k * a k i) * x i = ∑ k, w k * (c k + ∑ i, a k i * x i) := by
  rw [Finset.sum_congr rfl fun k _ => mul_add (w k) (c k) _, Finset.sum_add_distrib]
  congr 1
  rw [Finset.sum_congr rfl fun i _ => Finset.sum_mul .., Finset.sum_comm]
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => mul_assoc ..

/-- `Lin.combo` at the pads is the `w`-sum of the forms' values, for forms on the pads. -/
theorem linVal_combo {s m : ℕ} (h : Fin s → 𝔽) (ws : Array F128) (ls : Array Zk.Lin) (hw : ws.size = m)
    (hl : ls.size = m) (hls : ∀ k < m, (ls[k]!).a.size = s) :
    linVal h (Zk.Lin.combo ws ls s) = ∑ k : Fin m, fOf ws[k.val]! * linVal h ls[k.val]! := by
  unfold linVal
  have e : ∑ i : Fin s, fOf (Zk.Lin.combo ws ls s).a[i.val]! * h i =
      ∑ i : Fin s, (∑ k : Fin m, fOf ws[k.val]! * fOf (ls[k.val]!).a[i.val]!) * h i :=
    Finset.sum_congr rfl fun i _ => by rw [ZkExec.combo_a ws ls s m hw hl hls i.val i.isLt]
  rw [ZkExec.combo_c ws ls s m hw hl, e]
  exact sum_combo _ _ (fun k i => fOf (ls[k.val]!).a[i.val]!) h

/-! ## The rounds -/

variable {F : Type} [Field F]

/-- One honest round: from the claim `Σ_{i<2H} u_i·v_i`, the replay's next claim is `Σ_{i<H}` of the folded
vectors' products. -/
theorem round_next [CharP F 2] (H : ℕ) (u v : ℕ → F) (t : F) :
    (roundMsg H u v).2 * (t * t) +
        ((∑ i ∈ Finset.range (2 * H), u i * v i) + (roundMsg H u v).1 + (roundMsg H u v).1 + (roundMsg H u v).2) * t +
        ((∑ i ∈ Finset.range (2 * H), u i * v i) + (roundMsg H u v).1) =
      ∑ i ∈ Finset.range H, foldTop t H u i * foldTop t H v i := by
  have h2 : (2 : F) = 0 := CharTwo.two_eq_zero
  have hsplit : ∑ i ∈ Finset.range (2 * H), u i * v i =
      ∑ i ∈ Finset.range H, u i * v i + ∑ i ∈ Finset.range H, u (i + H) * v (i + H) := by
    rw [two_mul, Finset.sum_range_add]
    exact congrArg _ (Finset.sum_congr rfl fun x _ => by rw [Nat.add_comm H x])
  simp only [roundMsg]
  have hR : ∑ i ∈ Finset.range H, foldTop t H u i * foldTop t H v i =
      ∑ i ∈ Finset.range H, ((u (i + H) + u i) * (v (i + H) + v i) * (t * t)
        + (u i * v i + u (i + H) * v (i + H) + u (i + H) * v (i + H) + u (i + H) * v (i + H)
          + (u (i + H) + u i) * (v (i + H) + v i)) * t
        + (u i * v i + u (i + H) * v (i + H) + u (i + H) * v (i + H))) := by
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [foldTop]
    linear_combination (-(2 * u (i + H) * v (i + H) * t + u (i + H) * v (i + H))) * h2
  rw [hsplit, hR]
  simp only [Finset.sum_add_distrib, Finset.sum_mul, add_mul]

/-- Folding at one more coin, at the end of the list. -/
theorem foldAll_concat : ∀ (l : List F) (t : F) (len : ℕ) (v : ℕ → F),
    foldAll (l ++ [t]) len v = foldTop t (len / 2 ^ l.length / 2) (foldAll l len v)
  | [], t, len, v => by simp [foldAll]
  | a :: l, t, len, v => by
      rw [List.cons_append, foldAll, foldAll, foldAll_concat l t (len / 2)]
      congr 2
      rw [List.length_cons, pow_succ', Nat.div_div_eq_div_mul]

theorem lcStep_size (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (ts : Array F128) (x : Zk.Lin) (j : ℕ)
    (hx : x.a.size = lay.len) : (lcStep lay rounds ts x j).a.size = lay.len := by
  simp [lcStep, add_size, scale_size, masked_size, hx]

theorem foldl_lcStep_size (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (ts : Array F128) (x : Zk.Lin)
    (hx : x.a.size = lay.len) : ∀ n, ((List.range n).foldl (lcStep lay rounds ts) x).a.size = lay.len
  | 0 => by simpa using hx
  | n + 1 => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      exact lcStep_size _ _ _ _ _ (foldl_lcStep_size lay rounds ts x hx n)

/-- Round `j` at the pads: `e(∞)·r² + (e(0) + e(1) + e(∞))·r + e(0)` with `e(0) = claim + e(1)`. -/
theorem linVal_lcStep (lay : Zk.PadLayout) (h : Fin lay.len → 𝔽) (rounds : Array (F128 × F128))
    (ts : Array F128) (x : Zk.Lin) (j : ℕ) (hx : x.a.size = lay.len) :
    linVal h (lcStep lay rounds ts x j) =
      (fOf rounds[j]!.2 + pad h (lay.lc j 1)) * (fOf ts[j]! * fOf ts[j]!) +
        (linVal h x + (fOf rounds[j]!.1 + pad h (lay.lc j 0)) + (fOf rounds[j]!.1 + pad h (lay.lc j 0)) +
          (fOf rounds[j]!.2 + pad h (lay.lc j 1))) * fOf ts[j]! +
        (linVal h x + (fOf rounds[j]!.1 + pad h (lay.lc j 0))) := by
  have m1 := masked_size rounds[j]!.1 (lay.lc j 0) lay.len
  have m2 := masked_size rounds[j]!.2 (lay.lc j 1) lay.len
  have e0 : (x.add (Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) lay.len)).a.size = lay.len := by
    rw [add_size, hx, m1, Nat.min_self]
  have e01 : ((x.add (Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) lay.len)).add
      (Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) lay.len)).a.size = lay.len := by
    rw [add_size, e0, m1, Nat.min_self]
  have c1 : (((x.add (Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) lay.len)).add
      (Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) lay.len)).add
        (Zk.Lin.masked rounds[j]!.2 (lay.lc j 1) lay.len)).a.size = lay.len := by
    rw [add_size, e01, m2, Nat.min_self]
  unfold lcStep
  dsimp only
  rw [linVal_add _ _ _ (by rw [add_size, scale_size, scale_size, m2, c1, Nat.min_self]) e0,
    linVal_add _ _ _ (by rw [scale_size, m2]) (by rw [scale_size, c1]), linVal_scale, linVal_scale,
    linVal_add _ _ _ e01 m2, linVal_add _ _ _ e0 m1, linVal_add _ _ _ hx m1, linVal_masked, linVal_masked,
    fOf_mul]

/-- **The honest rounds**: after `j` rounds the replay's claim, at the pads, is `Σ u·v` of the columns folded at the
first `j` coins. -/
theorem linVal_running (lay : Zk.PadLayout) (h : Fin lay.len → 𝔽) (rounds : Array (F128 × F128))
    (ts : Array F128) (x : Zk.Lin) (hx : x.a.size = lay.len) (K n : ℕ) (hn : n ≤ K) (hts : ts.size = n)
    (u v : ℕ → 𝔽) (h0 : linVal h x = ∑ i ∈ Finset.range (2 ^ K), u i * v i)
    (hround : ∀ j < n, (fOf rounds[j]!.1 + pad h (lay.lc j 0), fOf rounds[j]!.2 + pad h (lay.lc j 1)) =
      lcMsg (2 ^ K) u v (ts.toList.map fOf) j) :
    ∀ j ≤ n, linVal h ((List.range j).foldl (lcStep lay rounds ts) x) =
      ∑ i ∈ Finset.range (2 ^ K / 2 ^ j),
        foldAll ((ts.toList.map fOf).take j) (2 ^ K) u i * foldAll ((ts.toList.map fOf).take j) (2 ^ K) v i
  | 0, _ => by simpa [foldAll] using h0
  | j + 1, hj => by
      have ih := linVal_running lay h rounds ts x hx K n hn hts u v h0 hround j (by omega)
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        linVal_lcStep _ _ _ _ _ _ (foldl_lcStep_size lay rounds ts x hx j), ih]
      have hr1 : fOf rounds[j]!.1 + pad h (lay.lc j 0) = (lcMsg (2 ^ K) u v (ts.toList.map fOf) j).1 :=
        congrArg Prod.fst (hround j (by omega))
      have hr2 : fOf rounds[j]!.2 + pad h (lay.lc j 1) = (lcMsg (2 ^ K) u v (ts.toList.map fOf) j).2 :=
        congrArg Prod.snd (hround j (by omega))
      rw [hr1, hr2]
      set H := 2 ^ K / 2 ^ (j + 1) with hH
      have h2H : 2 ^ K / 2 ^ j = 2 * H := by
        rw [hH, Nat.pow_div (by omega) (by norm_num), Nat.pow_div (by omega) (by norm_num),
          show K - j = K - (j + 1) + 1 by omega, pow_succ]
        ring
      have hlen : ((ts.toList.map fOf).take j).length = j := by simp; omega
      have htj : (ts.toList.map fOf).take (j + 1) = (ts.toList.map fOf).take j ++ [fOf ts[j]!] := by
        rw [List.take_add_one, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem (by omega),
          getElem!_pos ts j (by omega)]
        rfl
      have hH' : 2 ^ K / 2 ^ ((ts.toList.map fOf).take j).length / 2 = H := by
        rw [hlen, Nat.div_div_eq_div_mul, ← pow_succ]
      rw [h2H]
      unfold lcMsg
      rw [← hH, round_next, htj, foldAll_concat, foldAll_concat, hH']

/-! ## Completeness -/

/-- **The lincheck is complete on the masked replay** (live PROTOCOL.md §4, §10), at every coin. For a witness `w` of
`S`, a circuit fold `c` realizing `S`, and pads `h`: if the unmasked `final_a`, `final_b` are the true `a`, `b` claims at
the zerocheck's `(ζ, ρ)`, the unmasked round messages are the honest ones (`lcMsg` on the circuit's columns `lcU` and the
witness's `lcV`) and the unmasked `z_partial` is `lcV` folded at the round coins, then on the circuit's 64 folded
entries `combv` the "lincheck final" constraint vanishes at `h`, and the `ab` claim, valued at `w`'s `linVal`, holds
for `w`. -/
theorem lincheck_complete (S : Statement) (c : CircuitFold) (hc : FoldRealizes c S) (w : Witness S.m)
    (hw : S.Satisfies w) (lay : Zk.PadLayout) (h : Fin lay.len → 𝔽) (ζ α β rPrime : F128) (rho ts : Array F128)
    (hrho : rho.size = S.m - 6) (hts : ts.size = S.kLog - 6) (fA fB : F128) (rounds : Array (F128 × F128))
    (zPartial combv : Array F128)
    (ha : (⟨fOf fA + pad h lay.fa, execArith.lagS (fOf ζ), vecOf (S.m - 6) rho⟩ : Model.Claim 𝔽 S.m).Holds (aVec S w))
    (hb : (⟨fOf fB + pad h lay.fb, execArith.lagS (fOf ζ), vecOf (S.m - 6) rho⟩ : Model.Claim 𝔽 S.m).Holds (bVec S w))
    (hround : ∀ j < S.kLog - 6,
      (fOf rounds[j]!.1 + pad h (lay.lc j 0), fOf rounds[j]!.2 + pad h (lay.lc j 1)) =
        lcMsg (2 ^ S.kLog) (lcU S ζ α β rho) (lcV S rho w) (ts.toList.map fOf) j)
    (hzp : ∀ i < 64,
      fOf zPartial[i]! + pad h (lay.zp i) = foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcV S rho w) i)
    (hcomb : c.folded α β (skipWeights ζ) (rho.extract 0 (S.kLog - 6)) ts = .ok combv) :
    linVal h (lcFinal lay fA fB α β rounds zPartial ts (S.kLog - 6) combv) = 0 ∧
      (claimOf S.m (lcAbClaim S.m S.kLog rho ts rPrime) (linVal h (lcAbValue lay zPartial rPrime))).Holds w := by
  have hk6 := S.hk6
  have hkm := S.hk
  let zc : Model.ZcOut 𝔽 S.m :=
    ⟨fOf ζ, vecOf (S.m - 6) rho, vecOf (S.m - 6) rho, fOf fA + pad h lay.fa, fOf fB + pad h lay.fb, 0⟩
  have hV : lcV S rho w = hcol S (zc.ρOut S) w := by
    unfold lcV
    congr 1
    funext j
    simp only [Model.ZcOut.ρOut, zc]
    have hj : j.val < S.m - S.kLog := j.isLt
    rw [dite_eq_left (by omega)]
    rfl
  -- the start: `α·a + b + β` at the pads is the lincheck's sum `Σ_c comb[c]·h[c]`
  have hstart_sz : ((((Zk.Lin.masked fA lay.fa lay.len).scale α).add
      (Zk.Lin.masked fB lay.fb lay.len)).addConst β).a.size = lay.len := by
    simp [addConst_size, add_size, scale_size, masked_size]
  have h0 : linVal h ((((Zk.Lin.masked fA lay.fa lay.len).scale α).add
      (Zk.Lin.masked fB lay.fb lay.len)).addConst β) =
        ∑ i ∈ Finset.range (2 ^ S.kLog), lcU S ζ α β rho i * lcV S rho w i := by
    rw [linVal_addConst, linVal_add _ _ _ (by rw [scale_size, masked_size]) (masked_size _ _ _), linVal_scale,
      linVal_masked, linVal_masked]
    have hsum := combSum_eq execArith S zc w (fOf α) (fOf β)
    rw [← hV] at hsum
    have ha' : (⟨zc.va, execArith.lagS zc.z, zc.ρ⟩ : Model.Claim 𝔽 S.m).eval (aVec S w) =
        fOf fA + pad h lay.fa := ha
    have hb' : (⟨zc.vb, execArith.lagS zc.z, zc.ρ⟩ : Model.Claim 𝔽 S.m).eval (bVec S w) =
        fOf fB + pad h lay.fb := hb
    have hpin : pinMLE S w (zc.ρOut S) = 1 := by
      unfold pinMLE
      simp only [hw.2.1, ZMod.val_one, Nat.cast_one, mul_one]
      rw [Fin.sum_univ_eq_sum_range (fun i => Model.eqAt (zc.ρOut S) i)]
      exact sum_eqAt _ _
    rw [ha', hb', hpin] at hsum
    unfold lcU
    rw [show (fun j : Fin (S.kLog - 6) => fOf rho[j.val]!) = ρIn S zc from rfl, hsum]
    ring
  -- the rounds
  have hpow : 2 ^ S.kLog / 2 ^ (S.kLog - 6) = 64 := by
    rw [Nat.pow_div (by omega) (by norm_num), show S.kLog - (S.kLog - 6) = 6 by omega]
  have htake : (ts.toList.map fOf).take (S.kLog - 6) = ts.toList.map fOf :=
    List.take_of_length_le (by simp [hts])
  have hrun := linVal_running lay h rounds ts _ hstart_sz S.kLog (S.kLog - 6) (by omega) hts _ _ h0 hround
    (S.kLog - 6) le_rfl
  rw [htake, hpow] at hrun
  -- the circuit's 64 entries: `comb` folded at the round coins
  obtain ⟨v, hv, hvsz, hwv⟩ := hc.folded α β (skipWeights ζ) (rho.extract 0 (S.kLog - 6)) ts combv hcomb
  rw [hc.pin] at hwv
  have hmsz : (v.modify S.pin.val fun x => x + β).size = 2 ^ S.kLog := by simp [hvsz]
  have hcombv_sz : combv.size = 64 := by
    rw [hwv, ← Array.foldl_toList, foldl_halve_size, hmsz, Array.length_toList, hts, hpow]
  have hU : ∀ i < 64, fOf combv[i]! = foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcU S ζ α β rho) i := by
    intro i hi
    have hi' : i < (v.modify S.pin.val fun x => x + β).size / 2 ^ ts.toList.length := by
      rw [hmsz, Array.length_toList, hts, hpow]; exact hi
    rw [hwv, ← Array.foldl_toList, foldl_halve ts.toList _ i hi', hmsz]
    refine foldAll_congr (ts.toList.map fOf) (2 ^ S.kLog) _ _ (fun col hcol => ?_) i
      (by rw [List.length_map, Array.length_toList, hts, hpow]; exact hi)
    have hfr := hc.fold α (skipWeights ζ) (rho.extract 0 (S.kLog - 6)) v (skipWeights_size _) (by simp; omega) hv
      ⟨col, hcol⟩
    have hρ : vecOf (S.kLog - 6) (rho.extract 0 (S.kLog - 6)) = fun j : Fin (S.kLog - 6) => fOf rho[j.val]! := by
      funext j
      simp only [vecOf]
      rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by omega), Array.getElem_extract]
      simp
    have hsum : fOf v[col]! = ∑ i : Fin (2 ^ S.kLog), execArith.lagS (fOf ζ) ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
        Model.eqAt (fun j : Fin (S.kLog - 6) => fOf rho[j.val]!) (i.val / 64) *
          (fOf α * ((S.A₀ i ⟨col, hcol⟩).val : 𝔽) + ((S.B₀ i ⟨col, hcol⟩).val : 𝔽)) := by
      rw [hfr, hρ]
      rfl
    rw [modify_get! v S.pin.val col _ (by rw [hvsz]; exact hcol)]
    unfold lcU
    rw [comb, dite_eq_left hcol]
    by_cases hp : S.pin.val = col
    · rw [ite_eq_left hp, ite_eq_left hp.symm, fOf_add, hsum]
    · rw [ite_eq_right hp, ite_eq_right (Ne.symm hp), add_zero, hsum]
  -- `z_partial`
  have hzp_sz : (lcZp lay zPartial).size = 64 := by simp [lcZp]
  have hzp_get : ∀ k < 64, (lcZp lay zPartial)[k]! = Zk.Lin.masked zPartial[k]! (lay.zp k) lay.len := fun k hk => by
    rw [lcZp, getElem!_pos _ k (by simpa using hk), Array.getElem_map, Array.getElem_range]
  have hzp_a : ∀ k < 64, ((lcZp lay zPartial)[k]!).a.size = lay.len := fun k hk => by
    rw [hzp_get k hk, masked_size]
  have hzp_val : ∀ k : Fin 64, linVal h (lcZp lay zPartial)[k.val]! =
      foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcV S rho w) k.val := fun k => by
    rw [hzp_get k.val k.isLt, linVal_masked, hzp k.val k.isLt]
  refine ⟨?final, ?ab⟩
  case final =>
    unfold lcFinal
    have hrs : (lcRunning lay fA fB α β rounds ts (S.kLog - 6)).a.size = lay.len :=
      foldl_lcStep_size _ _ _ _ hstart_sz _
    rw [linVal_add _ _ _ hrs (ZkExec.combo_size _ _ _)]
    rw [show linVal h (lcRunning lay fA fB α β rounds ts (S.kLog - 6)) = _ from hrun,
      linVal_combo h combv _ hcombv_sz hzp_sz hzp_a,
      show ∑ k : Fin 64, fOf combv[k.val]! * linVal h (lcZp lay zPartial)[k.val]! =
          ∑ k : Fin 64, foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcU S ζ α β rho) k.val *
            foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcV S rho w) k.val from
        Finset.sum_congr rfl fun k _ => by rw [hzp_val k, hU k.val k.isLt],
      Fin.sum_univ_eq_sum_range (fun k => foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcU S ζ α β rho) k *
        foldAll (ts.toList.map fOf) (2 ^ S.kLog) (lcV S rho w) k)]
    exact CharTwo.add_self_eq_zero _
  case ab =>
    let tsFin : Fin (S.kLog - 6) → 𝔽 := fun j => fOf ts[j.val]!
    have hof : List.ofFn tsFin = ts.toList.map fOf := by
      refine List.ext_getElem (by simp [hts]) fun j h₁ h₂ => ?_
      rw [List.getElem_ofFn, List.getElem_map]
      simp only [tsFin]
      rw [getElem!_pos _ _ (by simp at h₁; omega)]
      simp
    have hxs : (ts.reverse ++ rho.extract (S.kLog - 6) (S.m - 6)).size = S.m - 6 := by
      simp [hts, hrho]; omega
    have hx : (fun i : Fin (S.m - 6) => fOf (lcAbClaim S.m S.kLog rho ts rPrime).x[i.val]!) =
        abPoint S zc tsFin := by
      funext i
      simp only [abPoint, lcAbClaim]
      rw [getElem!_pos _ i.val (by rw [hxs]; exact i.isLt), Array.getElem_append]
      by_cases hi : i.val < S.kLog - 6
      · have hi' : i.val < ts.reverse.size := by simp [hts, hi]
        rw [dite_eq_left hi, dite_eq_left hi', Array.getElem_reverse]
        simp only [tsFin]
        rw [getElem!_pos _ _ (by omega)]
        congr 2
        omega
      · have hi' : ¬ i.val < ts.reverse.size := by simp [hts, hi]
        rw [dite_eq_right hi, dite_eq_right hi', Array.getElem_extract]
        show _ = fOf rho[i.val]!
        rw [getElem!_pos _ _ (by omega)]
        congr 2
        simp [hts]
        omega
    show (⟨_, fun s => fOf (skipWeights rPrime)[s.val]!,
      fun i => fOf (lcAbClaim S.m S.kLog rho ts rPrime).x[i.val]!⟩ : Model.Claim 𝔽 S.m).eval w = _
    rw [hx]
    change (⟨linVal h (lcAbValue lay zPartial rPrime), execArith.lagS (fOf rPrime), abPoint S zc tsFin⟩ :
      Model.Claim 𝔽 S.m).eval w = _
    rw [abEval_eq, hof, ← hV]
    unfold lcAbValue
    rw [linVal_combo h _ _ (skipWeights_size _) hzp_sz hzp_a]
    show _ = ∑ k : Fin 64, fOf (skipWeights rPrime)[k.val]! * linVal h (lcZp lay zPartial)[k.val]!
    refine Finset.sum_congr rfl fun k _ => ?_
    have hl : execArith.lagS (fOf rPrime) k = fOf (skipWeights rPrime)[k.val]! := rfl
    rw [hzp_val k, hl]

/-- `lcFinal` is a form on the layout's pads (`LinRel`'s size). -/
theorem lcFinal_size (lay : Zk.PadLayout) (fA fB alpha beta : F128) (rounds : Array (F128 × F128))
    (zPartial ts : Array F128) (n : ℕ) (combv : Array F128) :
    (lcFinal lay fA fB alpha beta rounds zPartial ts n combv).a.size = lay.len := by
  unfold lcFinal lcRunning
  rw [add_size, foldl_lcStep_size _ _ _ _ (by simp [addConst_size, add_size, scale_size, masked_size]),
    ZkExec.combo_size, Nat.min_self]

/-- `Zk.replay`'s `ab` claim is `lcAbClaim` at its coins: the lincheck's `r_in` is the round coins reversed. -/
theorem abcClaims_ab (m kLog : ℕ) (zc : Flock.ZcOut) (rPrime : F128) (ts : Array F128) :
    (abcClaims m kLog zc { w := 0, rPrime, rIn := ts.reverse })[0]! = lcAbClaim m kLog zc.rho ts rPrime :=
  rfl

/-- `lcAbValue` is a form on the layout's pads. -/
theorem lcAbValue_size (lay : Zk.PadLayout) (zPartial : Array F128) (rPrime : F128) :
    (lcAbValue lay zPartial rPrime).a.size = lay.len :=
  ZkExec.combo_size _ _ _

end FlockSoundness.Discharge.Composed.Lincheck
