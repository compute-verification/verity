import Models.Pous.Assumptions.P2
import Proofs.Pous.P2.RingTight

/-!
# P2 in the ring model: dropping low bits breaks `k = 88` at `L = 2^30`

A responder keeps `m` code blocks without their low `t` bits (`w − t` bits a block) and, when block `i` is
challenged, runs P2's decoder on all `2^t` completions at once (`truncProg`): a round asks for the block's two keys,
a round for the `2^t` squares, a round for the `2^t` squares after the mask, and it answers with a completion that is
a valid label and decodes to the block's offset label. Decoding is injective on valid labels, so that completion is
the code block (`truncProg_run`). Every challenge below `m` is answered within `2` multiplication rounds, `2` key queries
and `2^(t+1)` ring queries, and the sequential audit passes with probability at least `(m/B)^k`
(`trunc_seq_accepted`). At `2^19` blocks of P2's modulus, `t = 29` keeps `497,571` blocks within the state bound and
`(497571/2^19)^88 > 1% + 2^-128`, so `MeetsOn` fails at every `k ≤ 88`, `D ≥ 2`, `Q ≥ 2` and `L ≥ 2^30`
(`not_meetsOn_b19_k88`). The per-round form fails at every `c < m/B` (`not_roundSecureOn_trunc`): at P2's parameters
`t = 161` keeps `501,579` blocks, so `P2RingRoundSecure` fails from `L = 2^162` (`not_roundSecure_trunc`).
-/

namespace PousP2Ring

open Pous PousP2p PousThm1 Pous.P2.Assumptions
open scoped ENNReal

/-! ### The responder -/

section Programs

variable {B w p : ℕ} [NeZero p]

/-- Bits `t, …, w − 1` of a label: what the responder keeps of a block. -/
def hiBits (t : ℕ) (c : Bits w) : Bits (w - t) := fun k => c ⟨k + t, by have := k.2; omega⟩

/-- Bits `0, …, t − 1` of a label: what the responder drops. -/
def loBits {t : ℕ} (ht : t ≤ w) (c : Bits w) : Bits t := fun k => c (Fin.castLE ht k)

/-- The completion of the kept bits `h` by `e` in the low `t` bits. -/
def cand (t : ℕ) (h : Bits (w - t)) (e : ℕ) : Bits w := fun k =>
  if hk : (k : ℕ) < t then bitsOfNat t e ⟨k, hk⟩ else h ⟨k - t, by have := k.2; omega⟩

theorem cand_hiBits {t : ℕ} (ht : t ≤ w) (c : Bits w) : cand t (hiBits t c) (bitsVal (loBits ht c)) = c := by
  funext k
  unfold cand
  split_ifs with hk
  · rw [bitsOfNat_bitsVal]
    rfl
  · exact congrArg c (Fin.ext (Nat.sub_add_cancel (not_lt.1 hk)))

/-- The first multiplication round: the square of every completion. -/
def sqQs (t : ℕ) (h : Bits (w - t)) : List (RingQry B w) :=
  (List.range (2 ^ t)).map fun e => RingQry.mul (cand t h e) (cand t h e)

/-- Completion `e` after the first square and the mask, from the keys `r₀` and the squares `r₁`. -/
def uOf (p t : ℕ) (h : Bits (w - t)) (r₀ r₁ : List (Bits w)) (e : ℕ) : Bits w :=
  sigmaK p (nthAns r₀ 0) (sqFix p (cand t h e) (nthAns r₁ e))

/-- The second multiplication round: the square of every completion after the first square and the mask. -/
def sqQs₂ (p t : ℕ) (h : Bits (w - t)) (r₀ r₁ : List (Bits w)) : List (RingQry B w) :=
  (List.range (2 ^ t)).map fun e => RingQry.mul (uOf p t h r₀ r₁ e) (uOf p t h r₀ r₁ e)

open Classical in
/-- A completion that is a valid label and decodes to the offset label, if there is one. -/
noncomputable def pick (p t : ℕ) (h : Bits (w - t)) (r₀ r₁ r₂ : List (Bits w)) : Bits w :=
  if hx : ∃ e, e < 2 ^ t ∧ bitsVal (cand t h e) < p ∧ sqFix p (uOf p t h r₀ r₁ e) (nthAns r₂ e) = nthAns r₀ 1
  then cand t h (Classical.choose hx) else fun _ => false

/-- **The search**: block `i`'s two keys, then two rounds of `2^t` multiplications each. -/
noncomputable def truncProg (t : ℕ) (i : Fin B) (h : Bits (w - t)) : Prog (ringRO B w p) (Bits w) :=
  .round [RingQry.key i .mask, RingQry.key i .offset] fun r₀ =>
    .round (sqQs t h) fun r₁ =>
      .round (sqQs₂ (B := B) p t h r₀ r₁) fun r₂ =>
        .ret (pick p t h r₀ r₁ r₂)

theorem countP_key_mulMap (l : List ℕ) (g₁ g₂ : ℕ → Bits w) :
    (l.map fun e => (RingQry.mul (g₁ e) (g₂ e) : RingQry B w)).countP RingQry.isKey = 0 := by
  induction l <;> simp_all [RingQry.isKey]

theorem countP_ring_mulMap (l : List ℕ) (g₁ g₂ : ℕ → Bits w) :
    (l.map fun e => (RingQry.mul (g₁ e) (g₂ e) : RingQry B w)).countP (fun q => !q.isKey) = l.length := by
  induction l <;> simp_all [RingQry.isKey]

theorem countP_key_sqQs₂ (p t : ℕ) (h : Bits (w - t)) (r₀ r₁ : List (Bits w)) :
    (sqQs₂ (B := B) p t h r₀ r₁).countP RingQry.isKey = 0 :=
  countP_key_mulMap _ _ _

theorem countP_ring_sqQs₂ (p t : ℕ) (h : Bits (w - t)) (r₀ r₁ : List (Bits w)) :
    (sqQs₂ (B := B) p t h r₀ r₁).countP (fun q => !q.isKey) = 2 ^ t :=
  (countP_ring_mulMap _ _ _).trans List.length_range

/-- No multiplication in the key round, then two rounds with one: two key queries, `2^(t+1)` ring queries. -/
theorem truncProg_bounded (t : ℕ) (i : Fin B) (h : Bits (w - t)) :
    (truncProg (p := p) t i h).RingBounded 2 2 (2 ^ (t + 1)) := fun ω => by
  let r₀ : List (Bits w) :=
    @List.map (RingQry B w) (Bits w) ((ringRO B w p).answer ω) [RingQry.key i .mask, RingQry.key i .offset]
  let r₁ : List (Bits w) := @List.map (RingQry B w) (Bits w) ((ringRO B w p).answer ω) (sqQs t h)
  refine ⟨?_, ?_, ?_⟩
  · change 0 + ((if _ then 1 else 0) + ((if _ then 1 else 0) + 0)) ≤ 2
    split_ifs <;> omega
  · change 2 + (@List.countP (RingQry B w) RingQry.isKey (sqQs t h) +
      (@List.countP (RingQry B w) RingQry.isKey (sqQs₂ p t h r₀ r₁) + 0)) ≤ 2
    rw [countP_key_sqQs₂, sqQs, countP_key_mulMap]
  · change 0 + (@List.countP (RingQry B w) (fun q => !q.isKey) (sqQs t h) +
      (@List.countP (RingQry B w) (fun q => !q.isKey) (sqQs₂ p t h r₀ r₁) + 0)) ≤ 2 ^ (t + 1)
    rw [countP_ring_sqQs₂, sqQs, countP_ring_mulMap, List.length_range, pow_succ]
    omega

theorem nthAns_range {w : ℕ} (n : ℕ) (g : ℕ → Bits w) {e : ℕ} (he : e < n) :
    nthAns ((List.range n).map g) e = g e := by
  simp [nthAns, he]

theorem nthAns_map_range {w : ℕ} {α : Type} (f : α → Bits w) (n : ℕ) (g : ℕ → α) {e : ℕ} (he : e < n) :
    nthAns (((List.range n).map g).map f) e = f (g e) := by
  rw [List.map_map]
  exact nthAns_range _ _ he

end Programs

/-! ### The search finds the code block -/

section Run

variable {B w p : ℕ} [NeZero p]

/-- The decoder up to the offset, on a valid label `lab a`: `sqFix(u, u²)` with `u = σ_K(sqFix(a, a²))` is the label
of `sqE(σ_K(sqE a))`. -/
theorem fwd_lab (hp2 : p ≤ 2 ^ w) (hodd : p % 2 = 1) (ω : (ringRO B w p).Ω) (K : Bits w) (a : ZMod p) :
    sqFix p (sigmaK p K (sqFix p (lab w ω.1 a) ((ringRO B w p).answer ω (RingQry.mul (lab w ω.1 a) (lab w ω.1 a)))))
      ((ringRO B w p).answer ω (RingQry.mul
        (sigmaK p K (sqFix p (lab w ω.1 a) ((ringRO B w p).answer ω (RingQry.mul (lab w ω.1 a) (lab w ω.1 a)))))
        (sigmaK p K (sqFix p (lab w ω.1 a) ((ringRO B w p).answer ω (RingQry.mul (lab w ω.1 a) (lab w ω.1 a)))))))
      = lab w ω.1 (sqE ω.1 (eltD ω.1 (sigmaK p K (lab w ω.1 (sqE ω.1 a))))) := by
  rw [answer_mul ω hp2, sqFix_lab ω.1 hp2 hodd]
  have hv : bitsVal (sigmaK p K (lab w ω.1 (sqE ω.1 a))) < p := sigmaK_lt (lab_lt ω.1 hp2 _)
  generalize sigmaK p K (lab w ω.1 (sqE ω.1 a)) = v at hv ⊢
  conv_lhs => rw [← lab_eltD ω.1 hv]
  rw [answer_mul ω hp2, sqFix_lab ω.1 hp2 hodd]

/-- **`pick` returns `lab c`** when the answers are the model's and `lab c` decodes to block `i`'s offset: the true
completion passes the check, and any completion that passes is `lab c`. -/
theorem pick_eq (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) (ω : (ringRO B w p).Ω) {t : ℕ} (ht : t ≤ w)
    (i : Fin B) (c : ZMod p)
    (hc : sqE ω.1 (eltD ω.1 (sigmaK p (ω.2 i).1 (lab w ω.1 (sqE ω.1 c)))) = ω.1.1.symm (ω.2 i).2)
    (r₀ r₁ r₂ : List (Bits w)) (h0 : nthAns r₀ 0 = (ω.2 i).1)
    (h1 : nthAns r₀ 1 = lab w ω.1 (ω.1.1.symm (ω.2 i).2))
    (hr₁ : ∀ e < 2 ^ t, nthAns r₁ e = (ringRO B w p).answer ω
      (RingQry.mul (cand t (hiBits t (lab w ω.1 c)) e) (cand t (hiBits t (lab w ω.1 c)) e)))
    (hr₂ : ∀ e < 2 ^ t, nthAns r₂ e = (ringRO B w p).answer ω
      (RingQry.mul (uOf p t (hiBits t (lab w ω.1 c)) r₀ r₁ e) (uOf p t (hiBits t (lab w ω.1 c)) r₀ r₁ e))) :
    pick p t (hiBits t (lab w ω.1 c)) r₀ r₁ r₂ = lab w ω.1 c := by
  have hodd := odd_of_mod_four h4
  set h := hiBits t (lab w ω.1 c)
  set K := (ω.2 i).1
  have hdec : ∀ a : ZMod p, sqE ω.1 (eltD ω.1 (sigmaK p K (lab w ω.1 (sqE ω.1 a)))) = ω.1.1.symm (ω.2 i).2 →
      a = c := fun a ha => by
    have h1 := sqE_injective hp h4 ω.1 (ha.trans hc.symm)
    have h2 := congrArg (lab w ω.1) h1
    rw [lab_eltD ω.1 (sigmaK_lt (lab_lt ω.1 hp2 _)), lab_eltD ω.1 (sigmaK_lt (lab_lt ω.1 hp2 _))] at h2
    have h3 := congrArg (sigmaK p K) h2
    rw [sigmaK_sigmaK (lab_lt ω.1 hp2 _), sigmaK_sigmaK (lab_lt ω.1 hp2 _)] at h3
    exact sqE_injective hp h4 ω.1 (lab_injective ω.1 hp2 h3)
  have key : ∀ e < 2 ^ t, (bitsVal (cand t h e) < p ∧
      sqFix p (uOf p t h r₀ r₁ e) (nthAns r₂ e) = nthAns r₀ 1) ↔ cand t h e = lab w ω.1 c := by
    intro e he
    rw [hr₂ e he, h1, uOf, h0, hr₁ e he]
    constructor
    · rintro ⟨hv, hz⟩
      rw [← lab_eltD ω.1 hv] at hz ⊢
      rw [fwd_lab hp2 hodd] at hz
      exact congrArg (lab w ω.1) (hdec _ (lab_injective ω.1 hp2 hz))
    · intro hce
      rw [hce]
      exact ⟨lab_lt ω.1 hp2 c, (fwd_lab hp2 hodd ω K c).trans (congrArg (lab w ω.1) hc)⟩
  have he : bitsVal (loBits ht (lab w ω.1 c)) < 2 ^ t := bitsVal_lt _
  have hex : ∃ e, e < 2 ^ t ∧ bitsVal (cand t h e) < p ∧
      sqFix p (uOf p t h r₀ r₁ e) (nthAns r₂ e) = nthAns r₀ 1 :=
    ⟨_, he, (key _ he).2 (cand_hiBits ht _)⟩
  unfold pick
  rw [dite_eq_left hex]
  obtain ⟨he', hspec⟩ := Classical.choose_spec hex
  exact (key _ he').1 hspec

/-- **The search answers `lab c`** when `lab c` decodes to block `i`'s offset. -/
theorem truncProg_run (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) (ω : (ringRO B w p).Ω) {t : ℕ} (ht : t ≤ w)
    (i : Fin B) (c : ZMod p)
    (hc : sqE ω.1 (eltD ω.1 (sigmaK p (ω.2 i).1 (lab w ω.1 (sqE ω.1 c)))) = ω.1.1.symm (ω.2 i).2) :
    (truncProg t i (hiBits t (lab w ω.1 c))).run ((ringRO B w p).answer ω) = lab w ω.1 c :=
  pick_eq hp h4 hp2 ω ht i c hc _ _ _ rfl (answer_offset ω i) (fun _ he => nthAns_map_range _ _ _ he)
    (fun _ he => nthAns_map_range _ _ _ he)

end Run

/-! ### The audit passes with probability `(m/B)^k` -/

section Accept

variable {B w p b : ℕ} [NeZero B] [NeZero p]

/-- Keep the first `m` blocks without their low `t` bits, padded to `S` bits. -/
noncomputable def truncPre (t : ℕ) {m S : ℕ} (hm : m ≤ B) : Preprocessor (p2ring B w p b) S :=
  fun _ _ C _ => Pous.pad (codeEquiv m (w - t) fun j => hiBits t (C (Fin.castLE hm j)))

open Classical in
/-- Search a kept block, and answer zeros for any other index. -/
noncomputable def truncStep (t : ℕ) {m S : ℕ} (hS : m * (w - t) ≤ S) (σ : Bits S) (i : Fin B) :
    Prog (ringRO B w p) (Bits w) :=
  if hi : (i : ℕ) < m then truncProg t i ((codeEquiv m (w - t)).symm (Pous.trunc hS σ) ⟨i, hi⟩)
  else .ret fun _ => false

/-- The responder: `truncStep` on every challenge, keeping its state. -/
noncomputable def truncSeq (t : ℕ) {m S : ℕ} (hS : m * (w - t) ≤ S) :
    SeqResponderOn (p2ring B w p b) S 2 2 (2 ^ (t + 1)) :=
  ⟨fun _ _ _ σ i => progMap (fun c => (c, σ)) (truncStep (p := p) t hS σ i), fun _ _ _ σ i =>
    ringBounded_progMap _ (by
      unfold truncStep
      split
      · exact truncProg_bounded _ _ _
      · exact ringBounded_ret _ _ _ _)⟩

omit [NeZero B] in
/-- With the all-zero payload, block `i` decodes to its offset: `p2ring`'s code block passes `truncProg_run`'s
check. -/
theorem p2ring_zero_dec (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) (ω : (ringRO B w p).Ω) (i : Fin B) :
    let c := rhoE ω.1 (eltD ω.1 (sigmaK p (ω.2 i).1
      (lab w ω.1 (rhoE ω.1 (eltD ω.1 (PousP2p.pad B w b (fun _ => false) i) + ω.1.1.symm (ω.2 i).2)))))
    sqE ω.1 (eltD ω.1 (sigmaK p (ω.2 i).1 (lab w ω.1 (sqE ω.1 c)))) = ω.1.1.symm (ω.2 i).2 := by
  intro c
  have hodd := odd_of_mod_four h4
  have h0 : eltD ω.1 (PousP2p.pad B w b (fun _ => false) i) = 0 := by
    have hz : PousP2p.pad B w b (fun _ => false) i = lab w ω.1 0 := by
      have hv : bitsVal ((codeEquiv B b).symm (fun _ => false) i) = 0 := by
        simp [bitsVal, bitsEquiv, codeEquiv]
        exact fun _ => rfl
      rw [PousP2p.pad, hv, lab, lab_zero ω.1 hodd]
      rfl
    rw [hz, eltD_lab ω.1 hp2]
  simp only [c, sqE_rhoE hp h4]
  rw [lab_eltD ω.1 (sigmaK_lt (lab_lt ω.1 hp2 _)), sigmaK_sigmaK (lab_lt ω.1 hp2 _), eltD_lab ω.1 hp2,
    sqE_rhoE hp h4, h0, zero_add]

/-- **Dropping `t` low bits of `m` blocks passes with probability at least `(m/B)^k`**, for every prime `p ≡ 3 mod 4`
with `p ≤ 2^w`, every `b` and every `k`, against `truncSeq` within `(2, 2, 2^(t+1))`, at the all-zero payload. -/
theorem trunc_seq_accepted (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) {t m S : ℕ} (k : ℕ) (ht : t ≤ w)
    (hm : m ≤ B) (hS : m * (w - t) ≤ S) :
    ((m : ℝ≥0∞) / B) ^ k ≤
      pr (seqAcceptsOn (p2ring B w p b) k (fun _ => false) (truncPre t hm) (truncSeq (b := b) t hS)) := by
  rw [← pr_all_lt k B m hm, ← pr_snd (α := (p2ring B w p b).Coins),
    ← pr_snd (α := (ringRO B w p).Ω)]
  refine pr_mono fun x hx => ?_
  show seqAnswers _ (fun _ σ i => progMap (fun c => (c, σ)) (truncStep (p := p) t hS σ i)) 0 _ _ = _
  refine (seqAnswers_keep _ (fun σ i => truncStep (p := p) t hS σ i) 0 _ _).trans ?_
  refine List.map_ofFn.trans ?_
  refine congrArg List.ofFn (funext fun j => ?_)
  have hi : ((x.2.2 j : Fin B) : ℕ) < m := hx j
  show (truncStep t hS (truncPre t hm _ _ _ _) (x.2.2 j)).run _ = _
  rw [truncStep, dite_eq_left hi, truncPre, Pous.trunc_pad, Equiv.symm_apply_apply]
  exact truncProg_run hp h4 hp2 x.1 ht _ _ (p2ring_zero_dec hp h4 hp2 x.1 _)

/-! ### The per-round form -/

omit [NeZero B] in
theorem pr_lt (m : ℕ) (hm : m ≤ B) : pr (fun i : Fin B => (i : ℕ) < m) = (m : ℝ≥0∞) / B := by
  classical
  rw [pr_def, Fin.card_filter_val_lt, min_eq_right hm, Fintype.card_fin]

open Classical in
/-- Search a kept block, from a state read as `m` blocks of `w − t` bits, and answer zeros for any other index. -/
noncomputable def truncAns (t m : ℕ) (σ : List Bool) (i : Fin B) : Prog (ringRO B w p) (Bits w) :=
  if hi : (i : ℕ) < m then truncProg t i ((codeEquiv m (w - t)).symm (PousP2Round.ofList σ) ⟨i, hi⟩)
  else .ret fun _ => false

/-- **Dropping low bits breaks the per-round form.** For every prime `p ≡ 3 mod 4` with `p ≤ 2^w`: a responder that
holds `m` blocks without their low `t` bits at a round's boundary (`m·(w − t) ≤ S` bits) and searches the `2^t`
completions (`truncProg`) answers every index below `m`, so `RoundSecureOn` fails at every `D ≥ 2`, `Q ≥ 2`,
`L ≥ 2^(t+1)`, `c < m/B` and `ε < 1`, for every payload width `b`. -/
theorem not_roundSecureOn_trunc (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) {t m S : ℕ} (ht : t ≤ w)
    (hm : m ≤ B) (hS : m * (w - t) ≤ S) (D Q L : ℕ) (hD : 2 ≤ D) (hQ : 2 ≤ Q) (hL : 2 ^ (t + 1) ≤ L)
    (c ε : ℝ≥0∞) (hc : c < (m : ℝ≥0∞) / B) (hε : ε < 1) : ¬ RoundSecureOn (p2ring B w p b) S D Q L c ε := by
  intro h
  have h1 := h (fun _ => false)
    (fun a => List.ofFn (codeEquiv m (w - t) fun j =>
      hiBits t (((p2ring B w p b).enc (fun _ => false) a.1 a.2).2 (Fin.castLE hm j))))
    (fun _ σ i => truncAns t m σ i) fun _ σ i => by
      unfold truncAns
      split
      · exact ringBounded_mono (truncProg_bounded _ _ _) hD hQ hL
      · exact ringBounded_ret _ _ _ _
  refine absurd ((pr_eq_one fun a => ⟨by rw [List.length_ofFn]; exact hS, ?_⟩).symm.le.trans h1) (not_le.2 hε)
  refine hc.trans_le ((pr_lt m hm).symm.le.trans (pr_mono fun i hi => ?_))
  show (truncAns t m _ i).run _ = _
  rw [truncAns, dite_eq_left hi, PousP2Round.ofList_ofFn, Equiv.symm_apply_apply]
  exact truncProg_run hp h4 hp2 a.1 ht _ _ (p2ring_zero_dec hp h4 hp2 a.1 _)

end Accept

/-! ### At P2's parameters -/

/-- `S = ⌊(18/19)·2^19·16448⌋`, whatever `p` and `b`. -/
theorem stateBound_p2ring_b19 (p b : ℕ) [NeZero p] :
    (p2ring (2 ^ 19) 16448 p b).stateBound Params.ρ = 8169621180 := by
  rw [Scheme.stateBound, Nat.floor_eq_iff (mul_nonneg (by norm_num [Params.ρ]) (Nat.cast_nonneg _))]
  change ((8169621180 : ℕ) : ℚ) ≤ 18 / 19 * ((2 ^ 19 * 16448 : ℕ) : ℚ) ∧
    18 / 19 * ((2 ^ 19 * 16448 : ℕ) : ℚ) < (8169621180 : ℕ) + 1
  norm_num

/-- **P2 fails `MeetsOn` at `k = 88` from `L = 2^30`.** At `2^19` blocks of P2's modulus `p = 2^16448 − 21065`, if
`p` is prime: keeping `497,571` blocks without their low `29` bits (`16,419` bits a block, within the state bound
`8,169,621,180`) and searching the `2^29` completions with the decoder (`trunc_seq_accepted`) passes `k`
sequential challenges with probability at least `(497571/2^19)^k ≥ (497571/2^19)^88 > 1% + 2^-128`. So `p2ring`
fails `MeetsOn` for every payload width `b`, `k ≤ 88`, `D ≥ 2`, `Q ≥ 2`, `L ≥ 2^30` and `ε`: at `k = 88` the
requirement can hold only under a ring-operation cap below `2^30` per answer. -/
theorem not_meetsOn_b19_k88 (hP : PrimeP16448) (b D Q L k : ℕ) (hD : 2 ≤ D) (hQ : 2 ≤ Q) (hL : 2 ^ 30 ≤ L)
    (hk : k ≤ 88) (ε : ℝ≥0∞) :
    ¬ MeetsOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential D Q L k ε := by
  rintro ⟨hε, -, -, -, hsec⟩
  have hS : 497571 * (16448 - 29) ≤ (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) b).stateBound Params.ρ := by
    rw [stateBound_p2ring_b19]
    norm_num
  let A₂ : SeqResponderOn (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) b)
      ((p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) b).stateBound Params.ρ) D Q L :=
    ⟨(truncSeq (b := b) 29 hS).step, fun W pp j σ i =>
      ringBounded_mono ((truncSeq (b := b) 29 hS).bounded W pp j σ i) hD hQ (le_trans (by norm_num) hL)⟩
  have h := ((trunc_seq_accepted (b := b) hP p16448_mod_four hyps_16448.2.1 k (by norm_num) (by norm_num) hS).trans
    (hsec _ _ A₂)).trans (add_le_add le_rfl hε)
  have hL' : (((497571 : ℕ) : ℝ≥0∞) / ((2 ^ 19 : ℕ) : ℝ≥0∞)) ^ k = ENNReal.ofReal (((497571 : ℝ) / 2 ^ 19) ^ k) := by
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast (2 ^ 19), ← ENNReal.ofReal_div_of_pos (by norm_num),
      ← ENNReal.ofReal_pow (by positivity)]
    norm_num
  have hR : ENNReal.ofReal Params.δ + Params.εMax = ENNReal.ofReal (1 / 100 + (2⁻¹ : ℝ) ^ 128) := by
    rw [ENNReal.ofReal_add (by norm_num) (by positivity), ENNReal.ofReal_pow (by norm_num),
      ENNReal.ofReal_inv_of_pos (by norm_num), ENNReal.ofReal_ofNat]
    simp only [Params.δ, Params.εMax, Params.secParam]
    norm_num
  rw [hL', hR, ENNReal.ofReal_le_ofReal_iff (by positivity)] at h
  have h88 : ((497571 : ℝ) / 2 ^ 19) ^ 88 ≤ ((497571 : ℝ) / 2 ^ 19) ^ k :=
    pow_le_pow_of_le_one (by positivity) (by norm_num) hk
  have hlt : (1 / 100 + (2⁻¹ : ℝ) ^ 128) < ((497571 : ℝ) / 2 ^ 19) ^ 88 := by norm_num
  linarith

/-- **Dropping 161 low bits breaks `P2RingRoundSecure` from `L = 2^162`.** At `2^19` blocks of P2's modulus, if `p` is
prime: keeping `501,579` blocks without their low 161 bits (`16,287` bits a block, within the state bound
`8,169,621,180`) and searching the `2^161` completions answers `501,579` of the `2^19` indices, more than the
conjecture's `501,578`, within `2` multiplication rounds, `2` key queries and `2^162` ring queries. 161 is the fewest
dropped bits at which `501,579` blocks fit. -/
theorem not_roundSecure_trunc (hP : PrimeP16448) (D Q L : ℕ) (hD : 2 ≤ D) (hQ : 2 ≤ Q) (hL : 2 ^ 162 ≤ L) :
    ¬ P2RingRoundSecure D Q L := by
  have hS : 501579 * (16448 - 161) ≤ (p2ring (2 ^ 19) 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ := by
    rw [stateBound_p2ring_b19]
    norm_num
  refine not_roundSecureOn_trunc (b := 16432) hP p16448_mod_four hyps_16448.2.1 (by norm_num) (by norm_num) hS D Q L
    hD hQ hL _ _ ?_ PousP2Round.eps_lt_one
  have hm : ((501579 : ℕ) : ℝ≥0∞) / ((2 ^ 19 : ℕ) : ℝ≥0∞) = ENNReal.ofReal ((501579 : ℝ) / 2 ^ 19) := by
    rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast (2 ^ 19), ← ENNReal.ofReal_div_of_pos (by norm_num)]
    norm_num
  rw [PousP2Round.c_eq, hm, ENNReal.ofReal_lt_ofReal_iff (by positivity)]
  norm_num

end PousP2Ring
