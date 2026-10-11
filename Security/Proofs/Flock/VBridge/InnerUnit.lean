import Proofs.Flock.VBridge.OfVStar
import VStar.InnerRepCheck

/-!
# V*'s `InnerRepCheck{S}` unit

`VStar.innerUnit S name` is one part of a rep's inner algebraic checks: one input of all its ports' words, cut into the row
ports (port `i` of `nᵢ` elements is `portWordsOf nᵢ` words) and then `v`'s; `GfResiduals{S.gf}` on the row ports' first
`128 nᵢ` bits (`w`, in port order) and `v`'s first `128 nv`; its residuals' bits, then zero bits up to 128 words.

* The unit. Its builder's state and outputs are `VStar.residualForms` on `rowForms S.ports 0` (the row ports' first bits,
  from input bit 0) and `vars (vBase S) (128 nv)` (`innerUnit_ok`), and `VStar.checkRepS` gives `w`'s elements as the row
  ports' total and at most 16 residuals.
* The theorems. `laid_innerUnit`: whenever the laid-out unit's rows hold at `v`, if the input columns hold `w`'s bits
  where the row ports put them (`rowCol`) and `vv`'s on `v`'s port (`InCols`), the output columns hold each residual's
  value at `w`, `vv` (`resVal`), 128 bits each, then zeros (`ResCols`). `unit_innerUnit`: the same of a copy of the unit in
  a unit of any partition.
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## Ports -/

/-- `InnerRepCheck{S}`'s input ports in words: the row ports', then `v`'s. -/
def repPorts (S : VStar.RepS) : List ℕ := S.ports.map (fun p => VStar.portWordsOf p.2) ++ [VStar.portWordsOf S.gf.nv]

/-- Its input bits. -/
def innerIn (S : VStar.RepS) : ℕ := 16 * (repPorts S).foldl (· + ·) 0

/-- The row ports' first `128 n` bits each, the ports from input bit `k`: `InnerRepCheck`'s `w`. -/
def rowForms : List (String × ℕ) → ℕ → List Lin
  | [], _ => []
  | p :: ps, k => vars k (128 * p.2) ++ rowForms ps (k + 16 * VStar.portWordsOf p.2)

/-- The input bit that carries bit `q` of `w`, the row ports from input bit `k`. -/
def rowCol : List (String × ℕ) → ℕ → ℕ → ℕ
  | [], k, _ => k
  | p :: ps, k, q => if q < 128 * p.2 then k + q else rowCol ps (k + 16 * VStar.portWordsOf p.2) (q - 128 * p.2)

/-- The input bit where `v`'s port starts, after the row ports. -/
def vBase (S : VStar.RepS) : ℕ := 16 * (S.ports.map fun p => VStar.portWordsOf p.2).sum

/-- The builder's run on its input bits: `VStar.residualForms` on `w` and `v`. -/
def innerRun (S : VStar.RepS) : List Lin × St :=
  (VStar.residualForms S.gf (rowForms S.ports 0).toArray (vars (vBase S) (128 * S.gf.nv)).toArray).run ⟨innerIn S, []⟩

theorem le_portWords (n : ℕ) : 128 * n ≤ 16 * VStar.portWordsOf n := by
  unfold VStar.portWordsOf
  omega

theorem foldl_add : ∀ (l : List ℕ) (a : ℕ), l.foldl (· + ·) a = a + l.sum
  | [], a => by simp
  | x :: l, a => by rw [List.foldl_cons, foldl_add l (a + x), List.sum_cons, Nat.add_assoc]

theorem take_vars (k n m : ℕ) (h : m ≤ n) : (vars k n).take m = vars k m := by
  apply List.ext_getElem
  · simp only [List.length_take, length_vars]
    omega
  · intro i _ _
    simp [vars]

theorem length_varSlices : ∀ (k : ℕ) (ws : List ℕ), (varSlices k ws).length = ws.length
  | _, [] => rfl
  | k, _ :: ws => by simp only [varSlices, List.length_cons, length_varSlices _ ws]

theorem varSlices_append : ∀ (k : ℕ) (a b : List ℕ),
    varSlices k (a ++ b) = varSlices k a ++ varSlices (k + 16 * a.sum) b
  | k, [], b => by simp [varSlices]
  | k, x :: a, b => by
    simp only [List.cons_append, varSlices, varSlices_append (k + 16 * x) a b, List.sum_cons, Nat.mul_add,
      Nat.add_assoc]

theorem slices_vars_foldl (ws : List ℕ) (k : ℕ) :
    VStar.slices (vars k (16 * ws.foldl (· + ·) 0)) ws = varSlices k ws := by
  have h := slices_vars ws k 0
  rw [Nat.add_zero] at h
  rw [foldl_add, Nat.zero_add]
  exact h

/-- `InnerRepCheck`'s `w`: each row port's slice cut to its first `128 n` bits. -/
theorem flatMap_rowForms (f : (String × ℕ) × List Lin → List Lin) (hf : ∀ p b, f (p, b) = b.take (128 * p.2)) :
    ∀ (ps : List (String × ℕ)) (k : ℕ) (rest : List (List Lin)),
      (ps.zip (varSlices k (ps.map fun p => VStar.portWordsOf p.2) ++ rest)).flatMap f = rowForms ps k
  | [], _, _ => by simp [rowForms, varSlices]
  | p :: ps, k, rest => by
    simp only [List.map_cons, varSlices, List.cons_append, List.zip_cons_cons, List.flatMap_cons, hf, rowForms,
      take_vars _ _ _ (le_portWords p.2), flatMap_rowForms f hf ps]

/-- `InnerRepCheck`'s `v`: the slice after the row ports', cut to its first `128 nv` bits. -/
theorem vPart (S : VStar.RepS) :
    (((varSlices 0 (S.ports.map fun p => VStar.portWordsOf p.2) ++
      varSlices (0 + 16 * (S.ports.map fun p => VStar.portWordsOf p.2).sum) [VStar.portWordsOf S.gf.nv]).drop
        S.ports.length).headD []).take (128 * S.gf.nv) = vars (vBase S) (128 * S.gf.nv) := by
  rw [List.drop_left' (by rw [length_varSlices, List.length_map])]
  simp only [varSlices, List.headD_cons, Nat.zero_add, vBase]
  exact take_vars _ _ _ (le_portWords _)

theorem length_rowForms : ∀ (ps : List (String × ℕ)) (k : ℕ), (rowForms ps k).length = 128 * (ps.map (·.2)).sum
  | [], _ => rfl
  | p :: ps, k => by
    simp only [rowForms, List.length_append, length_vars, length_rowForms ps, List.map_cons, List.sum_cons, Nat.mul_add]

theorem bit_rowForms (z : Var → Bool) : ∀ (ps : List (String × ℕ)) (k q : ℕ), q < 128 * (ps.map (·.2)).sum →
    bit z (rowForms ps k) q = z (rowCol ps k q)
  | [], _, _, hq => by simp at hq
  | p :: ps, k, q, hq => by
    have hl : (vars k (128 * p.2)).length = 128 * p.2 := length_vars _ _
    simp only [List.map_cons, List.sum_cons] at hq
    simp only [rowForms, rowCol]
    by_cases h : q < 128 * p.2
    · simp only [h, ↓reduceIte]
      rw [bit, List.getElem?_append_left (by rw [hl]; exact h)]
      exact bit_vars z k (128 * p.2) h
    · simp only [h, ↓reduceIte]
      rw [bit, List.getElem?_append_right (by rw [hl]; omega), hl]
      exact bit_rowForms z ps _ _ (by omega)

theorem rowCol_lt : ∀ (ps : List (String × ℕ)) (k q : ℕ), q < 128 * (ps.map (·.2)).sum →
    rowCol ps k q < k + 16 * (ps.map fun p => VStar.portWordsOf p.2).sum
  | [], _, _, hq => by simp at hq
  | p :: ps, k, q, hq => by
    simp only [List.map_cons, List.sum_cons] at hq ⊢
    have hp := le_portWords p.2
    simp only [rowCol]
    by_cases h : q < 128 * p.2
    · simp only [h, ↓reduceIte]
      omega
    · simp only [h, ↓reduceIte]
      have := rowCol_lt ps (k + 16 * VStar.portWordsOf p.2) (q - 128 * p.2) (by omega)
      omega

theorem innerIn_eq (S : VStar.RepS) : innerIn S = vBase S + 16 * VStar.portWordsOf S.gf.nv := by
  simp only [innerIn, repPorts, vBase, foldl_add, List.sum_append, List.sum_cons, List.sum_nil]
  omega

/-! ## The unit -/

theorem checkRepS_ok {S : VStar.RepS} (h : VStar.checkRepS S = .ok ()) :
    (S.ports.map (·.2)).foldl (· + ·) 0 = S.gf.nw ∧ 8 * S.gf.res.length ≤ VStar.RES_OUT_WORDS := by
  unfold VStar.checkRepS at h
  by_cases h1 : ((S.ports.map (·.1)).eraseDups.length != (S.ports.map (·.1)).length ||
      (S.ports.map (·.1)).contains "v" || S.ports.any (·.2 < 1)) = true
  · simp only [h1, ↓reduceIte] at h
    contradiction
  · by_cases h2 : ((S.ports.map (·.2)).foldl (· + ·) 0 != S.gf.nw) = true
    · simp only [h1, h2, ↓reduceIte] at h
      contradiction
    · by_cases h3 : 8 * S.gf.res.length > VStar.RES_OUT_WORDS
      · simp only [h1, h2, h3, ↓reduceIte] at h
        contradiction
      · simp only [bne_iff_ne, ne_eq, Decidable.not_not] at h2
        exact ⟨h2, by omega⟩

/-- **V*'s `InnerRepCheck` unit is `VStar.residualForms` on its ports**: its builder's state and outputs are `innerRun`'s,
then zero forms up to 128 words; its input ports `repPorts S`, its output port 128 words; and `checkRepS` gives `w`'s
elements as the row ports' total and at most 16 residuals. -/
theorem innerUnit_ok {S : VStar.RepS} {name : String} {U : VStar.UnitC} (h : VStar.innerUnit S name = .ok U) :
    U.st = (innerRun S).2 ∧
      U.outs = ((innerRun S).1 ++ List.replicate (16 * VStar.RES_OUT_WORDS - (innerRun S).1.length) Lin.zero).toArray ∧
      U.inPorts = repPorts S ∧ U.outPorts = [VStar.RES_OUT_WORDS] ∧
      (S.ports.map (·.2)).foldl (· + ·) 0 = S.gf.nw ∧ 8 * S.gf.res.length ≤ VStar.RES_OUT_WORDS := by
  unfold VStar.innerUnit at h
  cases hc : VStar.checkRepS S with
  | error e => simp [hc, bind, Except.bind] at h
  | ok u =>
    cases hk : VStar.checkS S.gf with
    | error e => simp [hc, hk, bind, Except.bind] at h
    | ok _ =>
      simp only [hc, hk, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      subst h
      simp only [run_input_bind, slices_vars_foldl, varSlices_append]
      rw [flatMap_rowForms _ _, vPart]
      · refine ⟨?_, ?_, ?_, ?_, checkRepS_ok hc⟩ <;> first | rfl | trivial
      · intro p b
        rfl

theorem fresh_innerRun (S : VStar.RepS) (hS : ∀ r ∈ S.gf.res, OpsInRange S.gf r) :
    Good (innerIn S) (innerRun S).2 := by
  unfold innerRun
  rw [residualForms_eq _ _ _ hS]
  exact fresh_residualsGo _ _ _ _ (innerIn S) ⟨innerIn S, []⟩ ⟨by simp, fun k r h => by simp at h⟩

theorem bitOf_zero (k : ℕ) : bitOf 0 k = false := by
  have h : FlockLevel3.w128 (0 : F128).lo (0 : F128).hi = 0 := rfl
  unfold bitOf
  rw [h]
  exact Nat.zero_testBit k

theorem bit_of_length_le {z : Var → Bool} {W : List Lin} {q : ℕ} (h : W.length ≤ q) : bit z W q = false := by
  simp [bit, List.getElem?_eq_none h]

/-! ## The laid-out unit -/

/-- The input columns hold `w`'s bits where the row ports put them (`rowCol`) and `vv`'s on `v`'s port. -/
def InCols (S : VStar.RepS) (w vv : ℕ → F128) (v : ℕ → Bool) : Prop :=
  (∀ q < 128 * S.gf.nw, v (rowCol S.ports 0 q) = wbit w q) ∧ ∀ q < 128 * S.gf.nv, v (vBase S + q) = wbit vv q

/-- The columns from `base` hold each residual of `S` at `w`, `vv`, 128 bits each, then zeros to 128 words. -/
def ResCols (S : VStar.GfS) (w vv : ℕ → F128) (v : ℕ → Bool) (base : ℕ) : Prop :=
  (∀ r (hr : r < S.res.length), ∀ k < 128,
    v (base + (128 * r + k)) = bitOf (resVal (gfS S) (termVal w vv) (res S.res[r])) k) ∧
    ∀ j, 128 * S.res.length ≤ j → j < 16 * VStar.RES_OUT_WORDS → v (base + j) = false

/-- **V*'s laid-out `InnerRepCheck` unit computes its residuals**: whenever its rows hold at `v`, if its input columns
hold `w`'s bits on the row ports and `vv`'s on `v`'s (`InCols`), `w` and `vv` zero past `nw` and `nv` elements, its output
columns from `outStart` hold each residual's value (`resVal`), then zeros (`ResCols`). At an S whose products' operands are
among its operands (`OpsInRange`, which `VStar.checkS` asks of every S). -/
theorem laid_innerUnit {S : VStar.RepS} {name : String} {U : VStar.UnitC} (hU : VStar.innerUnit S name = .ok U)
    (hS : ∀ r ∈ S.gf.res, OpsInRange S.gf r) {l : Laid} (hl : U.laid = .ok l) (v : ℕ → Bool)
    (h : LaidRows l (inRowsOf (Array.replicate U.inWords 16)) v) (w vv : ℕ → F128)
    (hw0 : ∀ e, S.gf.nw ≤ e → w e = 0) (hv0 : ∀ e, S.gf.nv ≤ e → vv e = 0) (hin : InCols S w vv v) :
    ResCols S.gf w vv v (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16)) := by
  obtain ⟨hst, houts, hinP, houtP, hnw, hres⟩ := innerUnit_ok hU
  have hN : (Array.replicate U.inWords 16).foldl (· + ·) 0 = innerIn S := by
    rw [foldl_replicate, VStar.UnitC.inWords, hinP, innerIn]
    exact Nat.mul_comm _ _
  have hL : LaidRows (layout U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16) U.outs)
      (inRowsOf (Array.replicate U.inWords 16)) v := by
    rw [laid_ok hl] at h
    exact h
  have hg : Good ((Array.replicate U.inWords 16).foldl (· + ·) 0) U.st := by
    rw [hN, hst]
    exact fresh_innerRun S hS
  have hrows := laid_rows U.st _ _ U.outs v hg hL
  rw [hst] at hrows
  have hP := (sound_residualForms S.gf (rowForms S.ports 0) (vars (vBase S) (128 * S.gf.nv)) hS ⟨innerIn S, []⟩).2 _
    hrows
  have hcol : ∀ x < innerIn S,
      v (col ((Array.replicate U.inWords 16).foldl (· + ·) 0) (inRowsOf (Array.replicate U.inWords 16)) x) = v x := by
    intro x hx
    rw [hN]
    simp only [col, hx, ↓reduceIte]
  have hsum : (S.ports.map (·.2)).sum = S.gf.nw := by rw [← hnw, foldl_add, Nat.zero_add]
  have hWz : ∀ q, bit (fun x => v (col ((Array.replicate U.inWords 16).foldl (· + ·) 0)
      (inRowsOf (Array.replicate U.inWords 16)) x)) (rowForms S.ports 0) q = wbit w q := by
    intro q
    by_cases hq : q < 128 * S.gf.nw
    · have hq' : q < 128 * (S.ports.map (·.2)).sum := by rw [hsum]; exact hq
      rw [bit_rowForms _ _ _ _ hq']
      have hlt := rowCol_lt S.ports 0 q hq'
      rw [hcol _ (by rw [innerIn_eq, vBase]; omega)]
      exact hin.1 q hq
    · rw [bit_of_length_le (by rw [length_rowForms, hsum]; omega), wbit, hw0 _ (by omega), bitOf_zero]
  have hVz : ∀ q, bit (fun x => v (col ((Array.replicate U.inWords 16).foldl (· + ·) 0)
      (inRowsOf (Array.replicate U.inWords 16)) x)) (vars (vBase S) (128 * S.gf.nv)) q = wbit vv q := by
    intro q
    by_cases hq : q < 128 * S.gf.nv
    · rw [bit_vars _ _ _ hq]
      have hp := le_portWords S.gf.nv
      rw [hcol _ (by rw [innerIn_eq]; omega)]
      exact hin.2 q hq
    · rw [bit_of_length_le (by rw [length_vars]; omega), wbit, hv0 _ (by omega), bitOf_zero]
  obtain ⟨hlen, hval⟩ := hP
  change (innerRun S).1.length = _ at hlen
  have hval' := hval w vv hWz hVz
  change ∀ r (hr : r < S.gf.res.length), ∀ k < 128, bit _ (innerRun S).1 (128 * r + k) = _ at hval'
  have h16 : 16 * VStar.RES_OUT_WORDS = 2048 := rfl
  have hsz : U.outs.size = 16 * VStar.RES_OUT_WORDS := by
    rw [houts, List.size_toArray, List.length_append, List.length_replicate, hlen]
    have : VStar.RES_OUT_WORDS = 128 := rfl
    omega
  have hob : (Array.replicate U.outWords 16).foldl (· + ·) 0 = 16 * VStar.RES_OUT_WORDS := by
    rw [foldl_replicate, VStar.UnitC.outWords, houtP]
    rfl
  have hout := fun j (hj : j < U.outs.size) => laid_outs U.st _ _ U.outs v (by rw [hsz, hob]) hL j hj
  have hres' : 128 * S.gf.res.length ≤ 16 * VStar.RES_OUT_WORDS := by
    have : VStar.RES_OUT_WORDS = 128 := rfl
    omega
  refine ⟨fun r hr k hk => ?_, fun j hj1 hj2 => ?_⟩
  · have hj : 128 * r + k < (innerRun S).1.length := by rw [hlen]; omega
    rw [hout _ (by rw [hsz]; omega), ← hval' r hr k hk]
    simp only [houts, List.getElem_toArray, List.getElem_append_left hj, bit, List.getElem?_eq_getElem hj,
      Option.getD_some]
  · rw [hout _ (by rw [hsz]; omega)]
    simp [houts, List.getElem_append, show ¬ j < (innerRun S).1.length by rw [hlen]; omega]

section Unit

open FlockSoundness FlockSoundness.Audit

/-- **A copy of V*'s laid-out `InnerRepCheck` unit, in a unit of any partition, computes its residuals**
(`laid_innerUnit`). Let the verifier build `net` from the layout of `VStar.innerUnit S name` (`netOf`), and let unit `u`
be an instance of `k` stacked copies of the net's rows, the transcript `X` carrying 1 at the stack's constant. Then at
copy `j`'s column values (`copyVal`), `laid_innerUnit` holds. -/
theorem unit_innerUnit {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {net : Flock.Net} (hB : Built net)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : X (I.wire ((Rows.ofNet hB).stack k).one) = true)
    {S : VStar.RepS} {name : String} {U : VStar.UnitC} (hU : VStar.innerUnit S name = .ok U)
    (hS : ∀ r ∈ S.gf.res, OpsInRange S.gf r) {l : Laid} (hl : U.laid = .ok l) {name' : String}
    (hn : Flock.HmNets.netOf name' l = .ok net) (hu : net.unitLog ≤ 32) (w vv : ℕ → F128)
    (hw0 : ∀ e, S.gf.nw ≤ e → w e = 0) (hv0 : ∀ e, S.gf.nv ≤ e → vv e = 0) (hin : InCols S w vv (copyVal I X hj)) :
    ResCols S.gf w vv (copyVal I X hj)
      (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16)) := by
  obtain ⟨hs, h1⟩ := sat_copy I X hj
  have hL := laidRows_ofNet hB hn hu hs (h1.trans hone)
  rw [inWords_laid hl hn] at hL
  exact laid_innerUnit hU hS hl _ hL w vv hw0 hv0 hin

end Unit

end FlockVBridge.OfVStar
