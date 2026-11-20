module

public import Properties.Flock.Circuit.Chain
public import Proofs.Core.Circuit
public import Proofs.Flock.Soundness.TmplValue

/-!
# The chain's statements, proved

* `portsBind`: `PortsBind`. A port is among the block's committed inputs and outputs, which `Rblk` opens
  (`into_union_exits_eq_io`) at the position `PortsOpenAt` opens it, so the two values agree unless the openings'
  messages collide (`Binds`); `enc` is injective, and C's port bits read only the port gates (`portBits_congr`).
-/

@[expose] public section

namespace Flock.Chain

open Flock.Program Flock.OfProgram

/-- C's bits at `ports` read only the ports' gates. -/
theorem portBits_congr {P : Prog} {X Y : (ofProgram P).Trace} {ports : Array Spot.BlockFront.Port}
    (h : ∀ q ∈ ports.toList, ∀ hq : q.gate < (ofProgram P).N, X ⟨q.gate, hq⟩ = Y ⟨q.gate, hq⟩) :
    portBits X ports = portBits Y ports := by
  unfold portBits
  generalize ports.toList = l at h ⊢
  induction l with
  | nil => rfl
  | cons q l ih =>
    simp only [List.flatMap_cons]
    rw [ih fun q' hq' => h q' (List.mem_cons_of_mem _ hq')]
    congr 1
    split
    · rename_i hq
      rw [h q List.mem_cons_self hq]
    · rfl

/-- **`PortsBind`**: at one commitment, a block's openings and the statement's own at the ports give `BlockCorrect`'s
port hypotheses, or two messages the openings hashed collide. -/
theorem portsBind : PortsBind := by
  classical
  intro P units fr t α Sc H enc blocks u c X π π' z _ _ _ henc hio hbind hR hO
  obtain ⟨Y, hYi, hYo, hYop⟩ := hO
  by_cases hall : ∀ q ∈ fr.ins ++ fr.outs, ∀ hq : q.gate < (ofProgram P).N, X ⟨q.gate, hq⟩ = Y ⟨q.gate, hq⟩
  · left
    have hi : portBits X fr.ins = portBits Y fr.ins :=
      portBits_congr fun q hq => hall q (Array.mem_append.2 (Or.inl (Array.mem_toList_iff.1 hq)))
    have ho : portBits X fr.outs = portBits Y fr.outs :=
      portBits_congr fun q hq => hall q (Array.mem_append.2 (Or.inr (Array.mem_toList_iff.1 hq)))
    exact ⟨fun i hi' => by rw [hi]; exact hYi i hi', fun k hk => by rw [ho]; exact hYo k hk⟩
  · right
    push Not at hall
    obtain ⟨q, hq, hlt, hne⟩ := hall
    obtain ⟨_, hmem⟩ := hio q hq
    have hg : (⟨q.gate, hlt⟩ : Fin (ofProgram P).N) ∈
        (ofProgram P).into ((partOf P blocks).gates u.castSucc) ∪
          (ofProgram P).exits ((partOf P blocks).gates u.castSucc) := by
      rw [Verity.Partition.into_union_exits_eq_io]
      exact hmem
    obtain ⟨p, hp, hop⟩ := hR.1 _ hg
    obtain ⟨p', hp', hop'⟩ := hYop q hq hlt
    rw [hp] at hp'
    cases hp'
    rcases hbind _ _ _ _ _ _ hop hop' with heq | hcol
    · exact absurd (henc _ heq) hne
    · exact ⟨_, hcol⟩

section LowerRoot

open FlockSoundness Flock.Tmpl Flock.TmplValue

theorem le_alignUp (m span : ℕ) (h : 0 < span) : m ≤ Flock.Lower.alignUp m span := by
  unfold Flock.Lower.alignUp
  have := Nat.lt_div_mul_add (a := m + span - 1) h
  omega

theorem le_pow2_words (b : ℕ) : b ≤ ((b + WORD - 1) / WORD).nextPowerOfTwo * WORD := by
  have := Nat.le_nextPowerOfTwo ((b + WORD - 1) / WORD)
  have hW : WORD = 128 := rfl
  rw [hW] at this ⊢
  omega

theorem xorAt_single (y : ℕ) (z : ℕ → Bool) : xorAt [y] z = z y := by
  simp [xorAt]

/-- **The root `lower` writes**: rows of the net its template unrolls to, satisfied with the constant 1, make the root's
one copy of the lowered Definition `e'` (at `base`) satisfy its equations on the input bits `0 … N − 1`, and hold at
output bit `k` that copy's return column `rets[k]`; `lowerDef`'s template Definitions begin the template's. -/
theorem lower_root {P : Program.Prog} {d : ℕ} {inP outP : Array ℕ} {t : Tmpl.Tmpl} {n : Net}
    (hl : Flock.Lower.lower P d inP outP = .ok t) (h : Tmpl.unroll t = .ok n) (hu : n.unitLog ≤ 32)
    (hB : Built n) {z : ℕ → Bool} (hs : (Rows.ofNet hB).Sat fun c => z c.val) (hone : z t.constPos = true) :
    ∃ (Q : Program.Prog) (e : ℕ) (defn : Program.Defn) (e' : ℕ) (rets : Array ℕ) (s : Flock.Lower.St) (base : ℕ),
      Flock.Lift.lift P d = .ok (Q, e) ∧ Q.defs[e]? = some defn ∧
      (Flock.Lower.lowerDef Q (Q.defs.size + 1) e).run {} = .ok ((e', rets), s) ∧
      s.defs.toList <+: t.defs.toList ∧
      Holds t.defs true t.defs.size e' (fun y => z (base + y)) ((List.range defn.paramLeaves).map z) ∧
      ∀ k < defn.outLeaves, rets[k]! < (s.defs[e']!).cols →
        z (Flock.Lower.outBit t k) = z (base + rets[k]!) := by
  obtain ⟨hwf, hap, hord, -, -⟩ := Flock.Lower.lower_checked hl
  have hev := fun {c j f e o inp} => FlockSoundness.TmplValue.unroll_eval (c := c) (j := j) (f := f) (e := e) (o := o)
    (inp := inp) hwf hap hord h hu hs
  have h' := h
  unfold Tmpl.unroll at h'
  obtain ⟨acc, hacc, hn⟩ := Flock.TmplOrder.bind_ok h'
  have hrow := FlockSoundness.TmplValue.rowHolds_of_rowsSat hn hu hs
  unfold Flock.Lower.lower at hl
  obtain ⟨t', hb, hsc⟩ := Flock.TmplOrder.bind_ok hl
  have htt : t' = t := by
    unfold Flock.Lower.selfCheck at hsc
    split at hsc
    · simp only [pure, Except.pure, Except.ok.injEq] at hsc
      exact hsc
    · simp [throw, throwThe, MonadExceptOf.throw] at hsc
  subst htt
  unfold Flock.Lower.build at hb
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hb
  split at hb
  · cases hb
  cases hq : Flock.Lift.lift P d with
  | error err => simp [hq] at hb
  | ok qe =>
  obtain ⟨Q, e⟩ := qe
  simp only [hq] at hb
  cases hdf : Q.defs[e]? with
  | none => simp [hdf] at hb
  | some defn =>
  simp only [hdf] at hb
  split at hb
  · cases hb
  rename_i hN
  split at hb
  · cases hb
  rename_i hM
  cases hrun : StateT.run (Flock.Lower.lowerDef Q (Q.defs.size + 1) e) {} with
  | error err => simp [hrun] at hb
  | ok v =>
  obtain ⟨⟨e', rets⟩, s⟩ := v
  simp only [hrun, Except.ok.injEq] at hb
  clear hl hsc
  generalize hIB : Array.foldl (fun x1 x2 => x1 + x2) 0 inP = inBits at hb hN
  generalize hOB : Array.foldl (fun x1 x2 => x1 + x2) 0 outP = outBits at hb hM
  generalize hC : (s.defs[e']!).cols = cols at hb
  generalize hinW : ((inBits + WORD - 1) / WORD).nextPowerOfTwo = inW at hb
  generalize houtW : ((outBits + WORD - 1) / WORD).nextPowerOfTwo = outW at hb
  generalize hoS : Flock.Lower.alignUp (inW * WORD + cols) (outW * WORD) = oStart at hb
  have hW : WORD = 128 := rfl
  have hinb : inBits ≤ inW * WORD := hinW ▸ le_pow2_words inBits
  have houtb : outBits ≤ outW * WORD := houtW ▸ le_pow2_words outBits
  have hspan : 0 < outW * WORD := by
    have := Nat.pos_of_isPowerOfTwo (Nat.isPowerOfTwo_nextPowerOfTwo ((outBits + WORD - 1) / WORD))
    rw [houtW] at this
    rw [hW]; omega
  have hge : inW * WORD + cols ≤ oStart := hoS ▸ le_alignUp _ _ hspan
  have hdiv : oStart / WORD * WORD = oStart := by
    rw [← hoS]
    unfold Flock.Lower.alignUp
    rw [← Nat.mul_assoc, Nat.mul_div_cancel _ (by rw [hW]; decide)]
  subst hb
  let sg : Array Seg := if (defn.paramLeaves == 0) = true then #[]
    else #[⟨defn.paramLeaves, 0, 0, 1, 0⟩]
  let c : Call := ⟨e', 1, inW * WORD, cols, sg⟩
  generalize hT : (⟨defn.key, s.defs.size, _, _, _, _⟩ : Tmpl.Tmpl) = T at *
  have hroot : T.root = s.defs.size := by rw [← hT]
  have hrootlt : T.root < T.defs.size := by rw [← hT]; simp
  generalize hD : T.defs.getD T.root default = D at *
  have hDcols : D.cols = oStart + outW * WORD + 1 := by rw [← hD, ← hT]; simp [Array.getD_eq_getD_getElem?]
  have hDins : D.ins = 0 := by rw [← hD, ← hT]; simp [Array.getD_eq_getD_getElem?]
  have hDcalls : D.calls = #[c] := by rw [← hD, ← hT]; simp [Array.getD_eq_getD_getElem?, c, sg]
  have hDrows : D.rows = ((List.range inBits).map (fun c => (c, #[c], #[c]))).toArray ++
      ((List.range defn.outLeaves).map fun k =>
        (oStart + k, #[inW * WORD + rets[k]!], #[oStart + outW * WORD + 1])).toArray := by
    rw [← hD, ← hT]; simp [Array.getD_eq_getD_getElem?]
  have hK : T.constPos = oStart + outW * WORD := by
    unfold Tmpl.constPos; rw [hD, hDcols]; omega
  have hinC : T.inCols = inW * WORD := by rw [← hT]; simp [Tmpl.inCols]
  have hout : ∀ k, Flock.Lower.outBit T k = oStart + k := by
    intro k; unfold Flock.Lower.outBit; rw [← hT]; simp [hdiv]
  have hrefin : ∀ x < D.cols, D.ref 0 [] T.constPos x = x := by
    intro x hx; simp [Def.ref, hx]
  have hrefone : D.ref 0 [] T.constPos (oStart + outW * WORD + 1) = T.constPos := by
    simp [Def.ref, hDcols, hDins]
  refine ⟨Q, e, defn, e', rets, s, inW * WORD, rfl, hdf, hrun,
    by rw [← hT]; simp only [Array.toList_push]; exact List.prefix_append _ _, ?_, ?_⟩
  · have hc : c ∈ D.calls.toList := by rw [hDcalls]; simp
    have hbound : c.bound (D.ref 0 [] T.constPos) 0 = List.range defn.paramLeaves := by
      by_cases h0 : defn.paramLeaves = 0
      · simp [c, sg, Call.bound, h0]
      · simp only [c, sg, Call.bound, h0, beq_iff_eq, ite_false,
          List.flatMap_cons, List.flatMap_nil, List.append_nil, List.range_one, Nat.zero_mul, Nat.zero_add,
          Nat.add_zero, Nat.mul_zero]
        exact (List.map_congr_left fun x hx => hrefin x (by
          have := List.mem_range.1 hx; omega)).trans (List.map_id _)
    have hH := Flock.TmplValue.rows_holds hwf hap hacc z hrow c (by rw [hD]; exact hc) (by rw [hinC]) 0 Nat.one_pos
    rw [hD, hbound, hone] at hH
    simpa [c] using hH
  · intro k hk hlt
    rw [hC] at hlt
    unfold Tmpl.unroll at h
    have hdfR : T.defs[T.root]? = some D := by rw [← hD]; simp [Array.getD_eq_getD_getElem?, hrootlt]
    unfold Tmpl.rows at hacc
    obtain ⟨acc1, h1, h2⟩ := Flock.TmplOrder.bind_ok hacc
    rw [hD] at h1
    simp only [place, hdfR] at h1
    obtain ⟨acc2, h3, h4⟩ := Flock.TmplOrder.bind_ok h1
    have hmem : (oStart + k, #[inW * WORD + rets[k]!], #[oStart + outW * WORD + 1]) ∈ D.rows.toList := by
      rw [hDrows]
      simp only [Array.toList_append, List.mem_append]
      exact Or.inr (List.mem_map.2 ⟨k, List.mem_range.2 hk, rfl⟩)
    obtain ⟨l1, l2, u, u', -, e1, e2, e3⟩ := Flock.TmplValue.foldlM_split h3 hmem
    have hsu : Flock.TmplRows.Sized u :=
      Flock.TmplOrder.foldlM_inv Flock.TmplRows.Sized (fun _ _ _ _ hu hw => Flock.TmplRows.write_sized hw hu)
        (Flock.TmplRows.sized_empty _) e1
    have hk1 := Flock.TmplValue.write_made hsu e2
    have hk2 := Flock.TmplOrder.foldlM_inv (Flock.TmplValue.Kept _ _ _)
      (fun _ _ _ _ hu hw => Flock.TmplValue.write_kept hw hu) hk1 e3
    have hk3 := Flock.TmplOrder.foldlM_inv (Flock.TmplValue.Kept _ _ _)
      (fun _ _ _ _ hu hw => Flock.TmplOrder.foldlM_inv (Flock.TmplValue.Kept _ _ _)
        (fun _ _ _ _ hw0 hp => Flock.TmplValue.place_inv (Flock.TmplValue.Kept _ _ _)
          (fun hw' hk' => Flock.TmplValue.write_kept hw' hk') _ hp hw0) hu hw) hk2 h4
    have hk4 := Flock.TmplValue.write_kept h2 hk3
    simp only [Nat.zero_add] at hk4
    have hx := hrow (oStart + k) (by rw [hinC]; omega) (by rw [hK]; have := houtb; omega)
    unfold Flock.TmplValue.RowHolds at hx
    rw [hk4.2.1, hk4.2.2, xorAt_canon, xorAt_canon] at hx
    simp only [Array.toList_map, List.map_cons, List.map_nil, xorAt_single,
      hrefone, hone, Bool.and_true] at hx
    rw [hrefin _ (by rw [hDcols]; omega)] at hx
    rw [hout]
    exact hx

end LowerRoot

end Flock.Chain
