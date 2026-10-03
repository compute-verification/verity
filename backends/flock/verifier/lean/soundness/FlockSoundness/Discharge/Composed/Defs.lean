import FlockSoundness.Discharge.ZkLink
import FlockSoundness.Discharge.ZkExec
import FlockSoundness.Discharge.ZkView
import FlockSoundness.Discharge.ZkBind
import FlockSoundness.Discharge.Composed.Base
import FlockSoundness.Discharge.Composed.Inner

/-!
# One `--zk` sampled session: the pieces the composed headline reads

* `Gz`, `outz`, `accz`: the compiled `--zk` table at each draw (one table) as `modelTable`'s pieces, so that
  `ZkExec.sessionZC` at them is `ZkLink.batchedSessionZC` at the one-table plan (`sessionZC_planOf`).
* `DecodesOnZK`, `simZ`, `execRefinesZK_on`: `DecodesZK`, `simAuditZK` and `execRefinesZK_one` at the law's draws only
  (as `Tables.DecodesOnJ` is on the public path), the simulated strategy carried into the compiled batched session.
* `CoinedSetup`, `coinedOf`, `CoinedOK`, `CoinedOfSetup` (new item, open): the `--zk` table of a draw as `ZkView`'s
  `Coined` tables at the executable's leaves, at each coin vector, and the structural facts Lemma B reads of them.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts n : ℕ} {Reg : Type}

/-- The one-table plan at the draw's tables. -/
abbrev planOf (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) :=
  fun S R => [T S R]

/-- The compiled reps at each draw. -/
noncomputable def Gz (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (PendZ 𝔽 D Sl) :=
  fun S R => repZCAt A Hs (T S R)

/-- The compiled table's output at each draw. -/
noncomputable def outz (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), (Fin 1 → Cap D) → PendZ 𝔽 D Sl × PendZ 𝔽 D Sl → OpensZ 𝔽 D Sl →
      List (OutZC 𝔽 D Sl) :=
  fun S R caps pend os => [outZC Hs (T S R) (caps 0) pend os]

variable [Inhabited D] [Inhabited Sl]

open Classical in
/-- The compiled table's verdict at each draw. -/
noncomputable def accz (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ∀ (_ : Finset (Fin n)) (_ : Reg), List (OutZC 𝔽 D Sl) → Bool :=
  fun S R outs => decide (acceptedZC [T S R] outs)

omit [Fintype K] in
theorem sessionZC_planOf (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) :
    ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = batchedSessionZC A Hs (planOf T) :=
  funext fun S => funext fun R => batchedSessionZC_one A Hs T S R

section On

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs)
  (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : ∀ (_ : Finset (Fin n)) (_ : Reg), Array ByteArray → OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : ∀ (_ : Finset (Fin n)) (_ : Reg), ByteArray → Fin 1 → Cap D) (N : ℕ) (L : Law n)

/-- **The decoder is right at the law's draws** (`DecodesZK` at `L.draw ω` only): on every live outcome flock-verify
`--zk` accepts at a draw the law makes, each rep's game runs on the tape decoded from its stream and proof file, and
the compiled table's verdict on those runs holds. -/
def DecodesOnZK : Prop :=
  ∀ ω R, Decodes (outz Hs T (L.draw ω) R) (fun o => AcceptsZK I (wr (L.draw ω) R o))
    (fun x => accz T (L.draw ω) R x = true) (opn (L.draw ω) R) (dec (L.draw ω) R) (one (L.draw ω) R)
    (fin (L.draw ω) R) (capOf (L.draw ω) R) (Gz A Hs T (L.draw ω) R)

variable {L}

/-- `simAuditZK` from the decoder at the law's draws. -/
noncomputable def simAuditZKOn (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T))) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    ((simTable (outz Hs T (L.draw ω) τ.1) (fun o => AcceptsZK I (wr (L.draw ω) τ.1 o))
      (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1)
      (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (Gz A Hs T (L.draw ω) τ.1)
      (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1)).strategy
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **The simulated audit strategy in the compiled batched session**: `simAuditZKOn`'s, carried along
`sessionZC_planOf`. -/
noncomputable def simZ (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) : Strategy (audit L Reg (batchedSessionZC A Hs (planOf T))) :=
  cast (congrArg (fun g => Strategy (audit L Reg g)) (sessionZC_planOf A Hs T))
    (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)

/-- **R11 at `--zk` from the decoder at the law's draws** (`execRefinesZK_one`'s proof, reading `hdec` only there). -/
theorem execRefinesZK_on (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAcceptsZK I) (fun _ => False)
      (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
      (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOf dg mPts N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAcceptsZK I) ((liveOf dg mPts N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true)
        ((ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) (L.draw ω) τ.1).map fun ok => (ok, L.draw ω))
        ((simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simAuditZKOn ZkExec.sessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOf
  rw [prob_map]
  exact live_le (outz Hs T (L.draw ω) τ.1) _ (fun x => accz T (L.draw ω) τ.1 x = true) (opn (L.draw ω) τ.1) dg
    (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (Gz A Hs T (L.draw ω) τ.1)
    (fun pts r => allInh_repZCAt A Hs (T (L.draw ω) τ.1) pts r) (hdec ω τ.1) _

/-- **The `--zk` headline at flock-verify's acceptance, from the decoder at the law's draws**: a bound on the compiled
batched audit's acceptance with any event `Q` at `simZ … τ` bounds flock-verify `--zk`'s acceptance with `Q`. -/
theorem zk_headline_exec_on (hdec : DecodesOnZK A Hs T I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) (Q : Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionZC A Hs (planOf T)))
      (simZ A Hs T I wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => LiveAcceptsZK I o ∧ Q) (auditLive L Reg (liveOf dg mPts N wr)) τ ≤ B := by
  have hc : ∀ (g : Finset (Fin n) → Reg → Game Bool)
      (h : ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T) = g),
      prob (fun o => o.1 = true ∧ Q) (audit L Reg g)
          (cast (congrArg (fun g => Strategy (audit L Reg g)) h)
            (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ)) =
        prob (fun o => o.1 = true ∧ Q) (audit L Reg (ZkExec.sessionZC (Gz A Hs T) (outz Hs T) (accz T)))
          (simAuditZKOn A Hs T I wr opn dg dec one fin capOf N hdec τ) := by
    intro g h
    subst h
    rfl
  have hB' := (hc _ (sessionZC_planOf A Hs T)).symm.trans_le hB
  have h := exec_le_of_refines (execRefinesZK_on A Hs T I wr opn dg dec one fin capOf N hdec τ) Q hB'
  rwa [prob_eq_zero_of_out (fun _ _ h => h) τ, add_zero] at h

omit [Fintype K] in
/-- **The decoder at the law's draws from `RunsOnZC`** (`ZkExec.decodes_of_runsOnZC`, the coin step `true`). -/
theorem decodesOnZK_of_runsOnZC (h : RunsOnZC A Hs T I wr opn dec fin capOf L) :
    DecodesOnZK A Hs T I wr opn dec (fun _ _ _ => true) fin capOf L :=
  fun ω R => decodes_of_runsOnZC A Hs T I wr opn dec fin capOf h ω R

end On

/-! ## The open pieces -/

/-! ## The session's tables at the executable's leaves (new item, open) -/

/-- `eq(p, x)` over the bits of word index `x`, least significant first (`zk_veil.rs`'s `eq_word`). -/
def eqWord (p : List F128) (x : ℕ) : F128 :=
  (List.range p.length).foldl (fun acc i => acc * if x.testBit i then p[i]! else 1 + p[i]!) 1

/-- **The mask slot's weights span** (`zk_veil.rs`'s `mask_rank_ok`, rank `= 128·k`): the vectors
`(eq(p_1, x), …, eq(p_k, x))` over the mask words `x` span `GF(2)^{128k}`, i.e. every value of the claims' masked sums is
reached by some mask bits. -/
def MaskRankOk {k M : ℕ} (pts : Fin k → List F128) (words : Fin M → ℕ) : Prop :=
  Function.Surjective fun u : Fin M → ZMod 2 => fun j : Fin k =>
    ∑ x, fOf (eqWord (pts j) (words x)) * ZK.Table.castF (F := 𝔽) (u x)

/-- **The mask slot's word indices in the packed witness** (`flock-circuit.rs`'s `mask_words`): every block's slot, block
`b`'s words at `b·2^(k_log−7) + start/128 + j` for `j < 2^(slot_log−7)`. -/
def maskWordsOf (st : Stmt) : List ℕ :=
  match st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => []
  | some r => (List.range (2 ^ st.nbl)).flatMap fun b =>
      (List.range (2 ^ (r.slotLog - 7))).map fun j => b * 2 ^ (st.c.kLog - 7) + r.start / 128 + j

/-- **The positions below `L` nobody opens**, in increasing order, the `u`-th of them. -/
def unPos {t : ℕ} (L : ℕ) (opened : Fin t → ℕ) (u : ℕ) : ℕ :=
  ((List.range L).filter (fun x => decide (x ∉ List.ofFn opened))).getD u 0

open Classical in
/-- **A tree's positions, opened then the others**: the equivalence placing each at its tree index `f`, where one exists
(the positions fill the tree: at the executable's sizes), and otherwise opened first. -/
noncomputable def treeEquiv {a b : ℕ} (f : Fin a ⊕ Fin b → ℕ) : Fin a ⊕ Fin b ≃ Fin (a + b) :=
  if h : ∃ e : Fin a ⊕ Fin b ≃ Fin (a + b), ∀ x, (e x).val = f x then h.choose else finSumFinEquiv

/-- **A `--zk` table as `ZkView`'s tables**: the statement (its witnesses `Witness S.m` are the tables' witnesses), the
hidden-value and value-list types, the type of the other coins, the sizes, the mask slot's word indices, the `Coined`
table at each coin vector (its leaves the executable's: `Coined.tab`), and M1's level-0 padding data (the level-0
message's `cols` and, at each coin vector, the opened positions `pos`) that `padOnto_M1` reads. -/
structure CoinedSetup where
  S : Statement
  Hi : Type
  VL : Type
  [addHi : AddCommGroup Hi]
  [finHi : Fintype Hi]
  [decHi : DecidableEq Hi]
  Rest : Type
  z : ZkView.Sizes
  maskWords : Fin z.nMask → ℕ
  T : CoinsZK Rest → ZkView.Coined FlockLevel3.GF256 (Witness S.m) Hi VL z
  cols : ℕ
  pos : CoinsZK Rest → Fin z.nS → ℕ
  points : CoinsZK Rest → Fin z.nC → List F128
  rOuter : CoinsZK Rest → Fin 2 → List F128

attribute [instance] CoinedSetup.addHi CoinedSetup.finHi CoinedSetup.decHi

/-- **The coin vectors the honest prover doesn't refuse** (`refused_coins`, `flock-circuit.rs`): `β ≠ 0` in both reps
(`beta.isZero`, which `Zk.verifyInner` also checks), the mask slot's weights at the claim points of rank `128·k`
(`mask_rank_ok`, the rank check in `at_open`; the verifier does not check it), and `ρ₀, ρ₁` independent over F128 in both
reps (`ZkLigerito.rhoIndependent`, which the verifier also checks); and no zerocheck outer coin `1` in either rep (the
residual round's `(1 + r_j)⁻¹`, `inv 0 = 0`: a bad event, not a refusal of the prover). -/
def CoinedSetup.Admits (C : CoinedSetup) (c : CoinsZK C.Rest) : Prop :=
  (∀ r, (c.beta r).isZero = false) ∧ MaskRankOk (C.points c) C.maskWords ∧ (∀ r, Zk.rhoIndependent (c.rho r) = true) ∧
    ∀ r, ∀ x ∈ C.rOuter c r, x ≠ 1

/-- **The valid witnesses**: those satisfying the table's statement (`Statement.Satisfies`: the R1CS in every block, the
pin, and the public regions' values). -/
def CoinedSetup.Valid (C : CoinedSetup) (w : Witness C.S.m) : Prop := C.S.Satisfies w

/-- A `--zk` table's sizes with the rank check's `RANK_POINTS` claim points over `nMask` mask words, and `t` level-0
padding columns and as many opened positions (the opened set padded to `t_pad`, `Sizes`), and each rep's `Q` pads
queries over `K_PAD` padding symbols (`Zk.verifyInner`). -/
def withMask (z : ZkView.Sizes) (nMask t : ℕ) : ZkView.Sizes :=
  { z with nC := Zk.RANK_POINTS, nMask, tpad := t, nS := t, nQ := Zk.Q, nPad := Zk.K_PAD }

/-- **The opened positions from the queried ones**: the positions below `L` that are queried, in increasing order, then
the others; the first `t` of them. Distinct and below `L` by construction (`padPos_injective`, `padPos_lt`), and the
queried positions below `L` come first, so where at most `t` distinct positions are queried they are all opened. -/
def padPos (L t : ℕ) (qs : List ℕ) (i : Fin t) : ℕ :=
  ((List.range L).filter (fun x => decide (x ∈ qs)) ++ (List.range L).filter (fun x => decide (x ∉ qs))).getD i.val 0

/-- **The refinement of a `--zk` table that its coins do not fix**: the hidden-value and value-list types, the type of
the coins other than `β`, `ρ` and the claim points (the query coins among them), one such coin, the other sizes,
the level-0 positions each such coin queries (`coinedOf` opens them, padded by `padPos` to `t_pad` distinct positions),
each rep's pads message length before its padding symbols (`lay.len`), pads code length `n` and the pads positions each
such coin queries, and each coin vector's `Coined` table, whose `embed`, `β`, `ρ`, `eqw` and
`Gp` `coinedOf` replaces by what the coins and M1's code fix. -/
structure CoinedBody (S : Statement) (nMask t : ℕ) where
  Hi : Type
  VL : Type
  [addHi : AddCommGroup Hi]
  [finHi : Fintype Hi]
  [decHi : DecidableEq Hi]
  Rest : Type
  rest0 : Rest
  sz : ZkView.Sizes
  queries : Rest → List ℕ
  padLen : ℕ
  padN : ℕ
  padQueries : Fin 2 → Rest → List ℕ
  T : CoinsZK Rest → ZkView.Coined FlockLevel3.GF256 (Witness S.m) Hi VL (withMask sz nMask t)


/-- `eq(p, l)` over K at lane `l`'s bits, least significant first. -/
noncomputable def eqLane {k : ℕ} (p : Fin k → F256) (l : ℕ) : FlockLevel3.GF256 :=
  ∏ i : Fin k, if l.testBit i.val then FlockLevel3.GF256.of (p i) else 1 + FlockLevel3.GF256.of (p i)

/-- **The draw's other sizes** (`withMask` sets the rest): a region claim per region, `2^k₀` witness lanes of `2^cols`
words, level 0's `2^logLen − t_pad` positions nobody opens, the pads but `h_ab` and the triple's `x, y, w` (`PadIdx`'s
`Fin 4`) exposed, and the pads code's `n − Q` positions a rep doesn't open. -/
def szOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : ZkView.Sizes where
  nMask := 0
  nC := 0
  nR := st.regions.size
  nL := 2 ^ Tz.sch.k0
  L := 2 ^ Tz.sch.level₀.logCols
  tpad := 0
  nS := 0
  nU := 2 ^ Tz.sch.level₀.logLen - Tz.t
  nE := padLenOf Tz st - 4
  nQ := 0
  nPU := padNOf Tz st - Zk.Q
  nPad := 0

/-- **The draw's tables but their coin-fixed fields** (open, refinement: `nu`, `sReg` from `verify --zk` at the draw; `coinedOf` and `bodyOf` replace the
others). -/
noncomputable def tableOf (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    CoinsZK (RestOf Tz st) →
      ZkView.Coined FlockLevel3.GF256 (Witness Tz.S.m) (HiOf Tz) ByteArray
        (withMask (szOf Tz st) (maskWordsOf st).length Tz.t) := sorry

/-- A pad's index in the pads layout: the exposed ones first, then `h_ab` and the triple's `x, y, w` (`PadLayout.hab ..
tw`, the layout's last four). -/
def padIdx {nE : ℕ} : ZK.PadIdx nE → ℕ
  | .inl i => i.val
  | .inr j => nE + j.val

theorem fa_lt (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb < (szOf Tz st).nE := by
  simp only [szOf, padLenOf, Zk.PadLayout.new, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab, Zk.PadLayout.zp,
    Zk.PadLayout.lc, Zk.PadLayout.fc, Zk.PadLayout.fb, Zk.PadLayout.fa]
  omega

/-- **The draw's `--zk` table's refinement**: levels ≥ 1's salts, `ByteArray` value lists, `RestOf`'s coins (both reps'
level-0 queries, opened as `S = Q⁰ ∪ Q¹`, the pads queries, `ζ`), the pads layout's sizes, and `tableOf`'s tables reading `ζ` and `r̄` (`rbar = eq(r̄^r, l)`), each claim's rep (the rank check's points: rep 0's `ab`, `c`, then rep 1's), and
the exposed `h_fa`, `h_fb` (`ia`, `ib`). -/
noncomputable def bodyOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : CoinedBody Tz.S (maskWordsOf st).length Tz.t where
  Hi := HiOf Tz
  VL := ByteArray
  Rest := RestOf Tz st
  rest0 := ⟨fun _ _ => ⟨0, Nat.two_pow_pos _⟩, fun _ _ => 0, fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0,
    fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ => 0,
    fun _ _ => 0, fun _ _ => 0⟩
  sz := szOf Tz st
  queries := fun ρ => (List.ofFn (ρ.q0s 0) ++ List.ofFn (ρ.q0s 1)).map Fin.val
  padLen := padLenOf Tz st
  padN := padNOf Tz st
  padQueries := fun r ρ => (List.ofFn (ρ.qP r)).map fun w => w.lo.toNat % padNOf Tz st
  T := fun c =>
    { tableOf Tz st c with
      ζ := fun r => fOf (c.rest.zeta r)
      pubReg := pubRegOf st
      rbar := fun r l => eqLane (c.rest.lane r) l.val
      rep := fun k => if k.val < 2 then 0 else 1
      ia := ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
      ib := ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩ }

/-- **The draw's `--zk` table as `Coined` tables**: `bodyOf`'s tables over `Tz.S`, with `RANK_POINTS` claim points over
the mask slot's words (`maskWordsOf st`) and `innerOf`'s inner-proof fields, reading their coins (`β`, `ρ`, the rank
check's weights `eq(p_k, x)`; `embed` the inclusion of F128 in F256). Level 0: `cols` from the table's level 0, the
opened positions `padPos` of the coins' queries below `2^(cols+1)`, its code at a position `x` the message columns
`X̂_j(ω^x)` (`j < 2^cols`) and M1's padding columns `κ·X̂_i + X̂_{2^cols+i}` at `κ = PAD_C`, at the opened positions and at
the others (`unPos`). Each rep's pads code (`codeAt`, at `X = p + 1`): pad `i` contributes `X^i` at its layout index
(`padIdx`), padding symbol `j` contributes `X^(len + j)`, at the opened pads positions `padPos` of the queries below `n`
and at the others; each tree's positions at their tree indices (`treeEquiv`). -/
noncomputable def coinedOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : CoinedSetup :=
  let B := bodyOf Tz st
  letI := B.addHi
  letI := B.finHi
  letI := B.decHi
  let cols := Tz.sch.level₀.logCols
  let pos := fun c : CoinsZK B.Rest => padPos (2 ^ (cols + 1)) Tz.t (B.queries c.rest)
  let pL := max (min B.padN (2 ^ 63)) Zk.Q
  let ppos := fun (r : Fin 2) (c : CoinsZK B.Rest) => padPos pL Zk.Q (B.padQueries r c.rest)
  { S := Tz.S
    Hi := B.Hi
    VL := B.VL
    Rest := B.Rest
    z := withMask B.sz (maskWordsOf st).length Tz.t
    maskWords := fun x => (maskWordsOf st).getD x.val 0
    T := fun c =>
      let I := innerOf Tz st B.Hi B.VL (withMask B.sz (maskWordsOf st).length Tz.t) (CoinsZK B.Rest) c
      { B.T c with
        lanes := I.lanes
        vals := I.vals
        lig := I.lig
        lam := I.lam
        theta := I.theta
        embed := FlockLevel3.embed
        β := fun r => fOf (c.beta r)
        ρ := fun r i => if i = 0 then FlockLevel3.GF256.of (c.rho r).1 else FlockLevel3.GF256.of (c.rho r).2
        E := fun q j => execArith.Xhat j.val (cols + 1) (execArith.omega (pos c q))
        EU := fun u j => execArith.Xhat j.val (cols + 1)
          (execArith.omega (unPos (2 ^ Tz.sch.level₀.logLen) (pos c) u.val))
        GpU := fun u i => fOf Zk.PAD_C * execArith.Xhat i.val (cols + 1)
            (execArith.omega (unPos (2 ^ Tz.sch.level₀.logLen) (pos c) u.val))
          + execArith.Xhat (2 ^ cols + i.val) (cols + 1)
            (execArith.omega (unPos (2 ^ Tz.sch.level₀.logLen) (pos c) u.val))
        ChU := fun r u i => fOf ⟨(unPos B.padN (ppos r c) u.val + 1).toUInt64, 0⟩ ^ padIdx i
        CpU := fun r u j => fOf ⟨(unPos B.padN (ppos r c) u.val + 1).toUInt64, 0⟩ ^ (B.padLen + j.val)
        s0 := treeEquiv (Sum.elim (pos c) fun u => unPos (2 ^ Tz.sch.level₀.logLen) (pos c) u.val)
        sP := fun r => treeEquiv (Sum.elim (ppos r c) fun u => unPos B.padN (ppos r c) u.val)
        Ch := fun r k i => fOf ⟨(ppos r c k + 1).toUInt64, 0⟩ ^ padIdx i
        Cp := fun r k j => fOf ⟨(ppos r c k + 1).toUInt64, 0⟩ ^ (B.padLen + j.val)
        eqw := fun k x => fOf (eqWord (pointsOf c.rest k) ((maskWordsOf st).getD x.val 0))
        Gp := fun q i => fOf Zk.PAD_C * execArith.Xhat i.val (cols + 1) (execArith.omega (pos c q))
          + execArith.Xhat (2 ^ cols + i.val) (cols + 1) (execArith.omega (pos c q)) }
    cols := cols
    pos := pos
    points := fun c => pointsOf c.rest
    rOuter := fun c r => List.ofFn (c.rest.rOuter r) }

/-- **The structural facts Lemma B reads**, of a `CoinedSetup` at the arithmetic `A` and M1's padding constant `κ`: M1's
level-0 padding geometry (`cols ≤ 40`, `t_pad ≤ 2^cols`, at most `t_pad` opened positions), and at every coin vector
the honest prover doesn't refuse: the opened positions distinct and inside `2^(cols+1)`, the table reads its coins (`β`, `ρ`, the rank check's
weights `eq(p_k, x)` and `embed` the inclusion of F128 in F256), the rank check, M1's padding columns `Gp`, each rep's
`W^r` bijective, the pads onto, `β ≠ 0`, and the inner relation at every valid witness whose registered values are the
public ones. -/
def CoinedOK (A : Arith 𝔽 FlockLevel3.GF256) (κ : 𝔽) (C : CoinedSetup) : Prop :=
  C.cols ≤ 40 ∧ C.z.tpad ≤ 2 ^ C.cols ∧ C.z.nS ≤ C.z.tpad ∧
    ∀ c : CoinsZK C.Rest, C.Admits c →
      (Function.Injective (C.pos c) ∧ ∀ q, C.pos c q < 2 ^ (C.cols + 1)) ∧
      ((∀ r, (C.T c).β r = fOf (c.beta r)) ∧
        (∀ r, (C.T c).ρ r 0 = FlockLevel3.GF256.of (c.rho r).1 ∧ (C.T c).ρ r 1 = FlockLevel3.GF256.of (c.rho r).2) ∧
        (∀ k x, (C.T c).eqw k x = fOf (eqWord (C.points c k) (C.maskWords x))) ∧
        ∀ a : F128, (C.T c).embed (fOf a) = FlockLevel3.GF256.of ⟨a, 0⟩) ∧
      (C.T c).tab.Rank ∧
      (∀ q i, (C.T c).tab.Gp q i = κ * A.Xhat i.val (C.cols + 1) (A.omega (C.pos c q))
        + A.Xhat (2 ^ C.cols + i.val) (C.cols + 1) (A.omega (C.pos c q))) ∧
      (∀ r, Function.Bijective fun p : 𝔽 × 𝔽 => (C.T c).tab.Wr r p.1 p.2) ∧ (C.T c).tab.PadsOnto ∧
      (∀ r, (C.T c).tab.β r ≠ 0) ∧
      ∀ w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg → (C.T c).tab.InnerHolds w

/-- **The session's tables satisfy Lemma B's facts** (new item, open): at every draw `verify --zk` sets up (`y`), the
`Coined` tables of the draw's `--zk` table at its executable statement (`coinedOf (tablesAtZK dj y₀ S R) y.x.st`) are over
the table's statement, check rank over the `RANK_POINTS` claim points at the statement's mask words (`maskWordsOf`), have
other coins, and are `CoinedOK` at `execArith` and `PAD_C`. -/
def CoinedOfSetup (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
    (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).S = (tablesAtZK (Reg := Reg) dj y₀ S R).S ∧
    (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).z.nC = Zk.RANK_POINTS ∧
    (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).z.nMask = (maskWordsOf y.x.st).length ∧
    (∀ x, (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).maskWords x = (maskWordsOf y.x.st).getD x 0) ∧
    Nonempty (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Rest ∧
    CoinedOK execArith (fOf Zk.PAD_C) (coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st)

/-- **Zero knowledge at a `CoinedSetup`, at `δ₁` per hidden leaf** (Lemma B, `ZkWireFixed`'s form at one table): at any
coin vector the honest prover doesn't refuse, for any encoding `enc` of the view (the proof files) and any event `E`, the real view
at any valid witness `w` whose registered values are the public ones and `S_shvzk`'s at any `w0` are within
`2·|Hid|·δ₁`, both ways. -/
def ZkViewAt (C : CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : CoinsZK C.Rest), C.Admits c → ∀ {Out : Type} (enc : (∀ _ : Fin 1, (C.T c).tab.View) → Out)
    (w w0 : Fin 1 → Witness C.S.m), (∀ j, C.Valid (w j)) → ∀ (_hreg : ∀ j, (C.T c).tab.sReg (w j) = (C.T c).tab.pubReg)
    (E : Out → Prop),
    Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) =>
        E (enc fun j => (C.T c).viewExe (w j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) =>
            E (enc fun j => (C.T c).simExe (w0 j) (p j).1 (p j).2))
          + ∑ _j : Fin 1, 2 * Fintype.card (C.T c).tab.Hid * δ₁
      ∧ Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.SimRand × ((C.T c).tab.Hid → ZkView.Salt) =>
          E (enc fun j => (C.T c).simExe (w0 j) (p j).1 (p j).2))
        ≤ Game.prCoin (fun p : ∀ _ : Fin 1, (C.T c).tab.Rand × ((C.T c).tab.Hid → ZkView.Salt) =>
            E (enc fun j => (C.T c).viewExe (w j) (p j).1 (p j).2))
          + ∑ _j : Fin 1, 2 * Fintype.card (C.T c).tab.Hid * δ₁

end FlockSoundness.Discharge.Composed
