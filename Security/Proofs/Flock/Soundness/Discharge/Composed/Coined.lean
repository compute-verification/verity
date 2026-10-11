import Proofs.Flock.Soundness.Discharge.Composed.Defs
import Proofs.Flock.Soundness.Discharge.Composed.Base
import Proofs.Flock.Soundness.Discharge.Composed.Inner

/-!
# One `--zk` sampled session: the draw's tables as `Coined` tables

* `CoinedSetup`, `coinedOf`, `CoinedOK`, `CoinedOfSetup`: the `--zk` table of a draw as `ZkView`'s `Coined` tables at the
  executable's leaves, at each coin vector, and the structural facts Lemma B reads of them.
* `ZkViewAt`: the view statement the composed headline proves at each draw; `ZkViewAtReg`, at the witnesses whose
  registered values are the public ones.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {mPts n : ℕ} {Reg : Type} (P : LigProver)

/-! ## The session's tables at the executable's leaves -/

/-- **The mask slot's weights span** (`zk_veil.rs`'s `mask_rank_ok`, rank `= 128·k`): the vectors
`(eq(p_1, x), …, eq(p_k, x))` over the mask words `x` span `GF(2)^{128k}`, i.e. every value of the claims' masked sums is
reached by some mask bits. -/
def MaskRankOk {k M : ℕ} (pts : Fin k → List F128) (words : Fin M → ℕ) : Prop :=
  Function.Surjective fun u : Fin M → ZMod 2 => fun j : Fin k =>
    ∑ x, fOf (eqWord (pts j) (words x)) * ZK.Table.castF (F := 𝔽) (u x)

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
table at each coin vector (its leaves the executable's: `Coined.tab`), and C-Flock ZK's level-0 padding data (the
level-0 message's `cols` and, at each coin vector, the opened positions `pos`) that `padOnto_M1` reads. -/
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
(`mask_rank_ok`, the rank check in `at_open`; the verifier does not check it), `ρ₀, ρ₁` independent over F128 in both
reps (`ZkLigerito.rhoIndependent`, which the verifier also checks), and no zerocheck outer coin `1` in either rep
(`zk_veil::Masking`'s `refuse_outer`, as the coin opens; the verifier does not check it).

At an outer coin `r_j = 1` the residual round's `(1 + r_j)⁻¹` is `inv 0 = 0` (`Zk.replay`), and the honest pads miss the
inner proof's batched constraint by a witness-dependent amount, which the view would show; the prover stops there before
it sends anything that depends on the coin. Each refusal reads only opened coins. Under uniform coins the outer-coin
refusal has probability at most `(m − 13)·2^-128` per rep (`ZkEveryCoin.outer_one_le`), a completeness cost. -/
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
such coin queries (`coinedOf` builds the tables). -/
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


/-- Bit `v` of witness word `x` (word `x` the bits `128·x … 128·x + 127`). -/
def wordBit {m : ℕ} (w : Fin (2 ^ m) → ZMod 2) (x v : ℕ) : ZMod 2 := if h : 128 * x + v < 2 ^ m then w ⟨_, h⟩ else 0

/-- `eq(p, l)` over K at lane `l`'s bits, least significant first. -/
noncomputable def eqLane {k : ℕ} (p : Fin k → F256) (l : ℕ) : FlockLevel3.GF256 :=
  ∏ i : Fin k, if l.testBit i.val then FlockLevel3.GF256.of (p i) else 1 + FlockLevel3.GF256.of (p i)

/-- **The draw's other sizes** (`withMask` sets the rest): a region claim per region and link point, `2^k₀` witness lanes of `2^cols`
words, level 0's `2^logLen − t_pad` positions nobody opens, the pads but `h_ab` and the triple's `x, y, w` (`PadIdx`'s
`Fin 4`) exposed, and the pads code's `n − Q` positions a rep doesn't open. -/
def szOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : ZkView.Sizes where
  nMask := 0
  nC := 0
  nR := 2 * st.regions.size
  nL := 2 ^ Tz.sch.k0
  L := 2 ^ Tz.sch.level₀.logCols
  tpad := 0
  nS := 0
  nU := 2 ^ Tz.sch.level₀.logLen - Tz.t
  nE := padLenOf Tz st - 4
  nQ := 0
  nPU := padNOf Tz st - Zk.Q
  nPad := 0

/-- A pad's index in the pads layout: the exposed ones first, then the layout's last four `h_ab, h_x, h_y, h_w`
(`PadLayout.hab .. tw`), which `PadIdx`'s `inr 0 .. 3` names `h_x, h_y, h_ab, h_w` (`Table.hFull`). -/
def padIdx {nE : ℕ} : ZK.PadIdx nE → ℕ
  | .inl i => i.val
  | .inr j => nE + (![1, 2, 0, 3] : Fin 4 → ℕ) j

theorem fa_lt (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb < (szOf Tz st).nE := by
  simp only [szOf, padLenOf, Zk.PadLayout.new, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab, Zk.PadLayout.zp,
    Zk.PadLayout.lc, Zk.PadLayout.fc, Zk.PadLayout.fb, Zk.PadLayout.fa]
  omega

/-- **The draw's `--zk` table's refinement**: levels ≥ 1's salts, `VLOf` value lists, `RestOf`'s coins (both reps'
level-0 queries, opened as `S = Q⁰ ∪ Q¹`, and the pads queries, as positions), the sizes and the pads layout's sizes. -/
noncomputable def bodyOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : CoinedBody Tz.S (maskWordsOf st).length Tz.t where
  Hi := HiOf Tz
  VL := VLOf
  Rest := RestOf Tz st
  rest0 := ⟨fun _ _ => 0, fun _ _ => ⟨0, Nat.two_pow_pos _⟩, fun _ _ => 0, fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0,
    fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ _ _ => 0, fun _ _ => 0, fun _ _ _ => 0, fun _ _ => 0, fun _ _ => 0,
    fun _ _ _ _ => 0, fun _ _ _ => 0, fun _ _ => 0, fun _ => 0, fun _ _ _ => 0, fun _ _ _ => 0, fun _ _ _ => 0, fun _ _ => 0,
    fun _ => 0, fun _ _ => 0, fun _ _ => 0⟩
  sz := szOf Tz st
  queries := fun ρ => (List.ofFn (ρ.q0s 0) ++ List.ofFn (ρ.q0s 1)).map Fin.val
  padLen := padLenOf Tz st
  padN := padNOf Tz st
  padQueries := fun r ρ => (List.ofFn (ρ.qP r)).map fun w => w.lo.toNat % padNOf Tz st

/-- **The draw's `--zk` table as `Coined` tables**: over `Tz.S` at `bodyOf`'s types and coins, with `RANK_POINTS` claim points over
the mask slot's words (`maskWordsOf st`) and `innerOf`'s inner-proof fields, reading their coins (`β`, `ρ`, the rank
check's weights `eq(p_k, x)`; `embed` the inclusion of F128 in F256). Level 0: `cols` from the table's level 0, the
opened positions `padPos` of the coins' queries below `2^(cols+1)`, its code at a position `x` the message columns
`X̂_j(ω^x)` (`j < 2^cols`) and C-Flock ZK's padding columns `κ·X̂_i + X̂_{2^cols+i}` at `κ = PAD_C`, at the opened positions and at
the others (`unPos`). Each rep's pads code (`codeAt`, at `X = p + 1`): pad `i` contributes `X^i` at its layout index
(`padIdx`), padding symbol `j` contributes `X^(len + j)`, at the opened pads positions `padPos` of the queries below `n`
and at the others; each tree's positions at their tree indices (`treeEquiv`). The claims' ring-switch messages from the
witness's words (`wordBit`): masked claim `k`'s non-mask words at its rank point (`nu`), region claim `j`'s at its point
(`sReg`), and the public ones (`pubRegOf`); `r̄`'s `eq` (`rbar`), `ζ` the triple's `ε`, each claim's rep (rep 0's `ab`,
`c`, then rep 1's), the exposed `h_fa`, `h_fb` (`ia`, `ib`). -/
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
      let I := innerOf P Tz st (withMask B.sz (maskWordsOf st).length Tz.t) c
      { embed := FlockLevel3.embed
        nu := fun w k v => ∑ x : Fin (2 ^ (Tz.S.m - 7)),
          if x.val ∈ maskWordsOf st then 0
          else fOf (eqWord (pointsOf c.rest k) x.val) * ZK.Table.castF (F := 𝔽) (wordBit w x.val v.val)
        sReg := fun w j v => ∑ x : Fin (2 ^ (Tz.S.m - 7)),
          fOf (eqWord (regionPtOf st c.rest.link j) x.val) * ZK.Table.castF (F := 𝔽) (wordBit w x.val v.val)
        pubReg := pubRegOf st c.rest.link
        rbar := fun r l => eqLane (c.rest.lane r) l.val
        ia := ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fa, Nat.lt_of_succ_lt (fa_lt Tz st)⟩
        ib := ⟨(Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).fb, fa_lt Tz st⟩
        ζ := fun r => fOf (c.rest.zeta r)
        rep := fun k => if k.val < 2 then 0 else 1
        lanes := I.lanes
        vals := I.vals
        lig := I.lig
        lam := I.lam
        theta := I.theta
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

/-- **The structural facts Lemma B reads**, of a `CoinedSetup` at the arithmetic `A` and C-Flock ZK's padding constant
`κ`: C-Flock ZK's level-0 padding geometry (`cols ≤ 40`, `t_pad ≤ 2^cols`, at most `t_pad` opened positions), and at
every coin vector the honest prover doesn't refuse (`Admits`): the opened positions distinct and inside `2^(cols+1)`,
the table reads its coins (`β`, `ρ`, the rank check's weights `eq(p_k, x)` and `embed` the inclusion of F128 in F256),
the rank check, C-Flock ZK's padding columns `Gp`, each rep's `W^r` bijective, the pads onto, `β ≠ 0`, and the inner
relation at every valid witness whose registered values are the public ones. -/
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

/-- **The session's tables satisfy Lemma B's facts** (proved: `coinedOf_setup`): at every draw `verify --zk` sets up (`y`), the
`Coined` tables of the draw's `--zk` table at its executable statement (`coinedOf P (tablesAtZK dj y₀ S R) y.x.st`) are over
the table's statement, check rank over the `RANK_POINTS` claim points at the statement's mask words (`maskWordsOf`), have
other coins, and are `CoinedOK` at `execArith` and `PAD_C`. -/
def CoinedOfSetup (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)),
    (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).S = (tablesAtZK (Reg := Reg) dj y₀ S R).S ∧
    (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).z.nC = Zk.RANK_POINTS ∧
    (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).z.nMask = (maskWordsOf y.x.st).length ∧
    (∀ x, (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).maskWords x = (maskWordsOf y.x.st).getD x 0) ∧
    Nonempty (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).Rest ∧
    CoinedOK execArith (fOf Zk.PAD_C) (coinedOf P (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st)

/-- **Zero knowledge at a `CoinedSetup`, at `δ₁` per hidden leaf** (Lemma B, `ZkWireFixed`'s form at one table): at every
coin vector the honest prover doesn't refuse (`Admits`), for any encoding `enc` of the view (the proof files) and any
event `E`, the real view at any valid witness `w` and `S_shvzk`'s at any `w0` are within `2·|Hid|·δ₁`, both ways.

This is honest-verifier zero knowledge at fixed coins. It covers a verifier that chooses its coins because the coin
server commits every coin at `Hello`, before the prover's first message (`coin_tree.rs`), and the prover checks each
opening against that commitment before it uses the coin; the binding is collision resistance of `x ↦ SHA-512(ν ‖ x)`
(`keyed-cr`). No statement here models that commitment yet. -/
def ZkViewAt (C : CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
  ∀ (c : CoinsZK C.Rest), C.Admits c → ∀ {Out : Type} (enc : (∀ _ : Fin 1, (C.T c).tab.View) → Out)
    (w w0 : Fin 1 → Witness C.S.m), (∀ j, C.Valid (w j)) → ∀ (E : Out → Prop),
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

/-- `ZkViewAt` at the valid witnesses whose registered values are the public ones at the coin vector (`H_reg`), which
Lemma B reads; `zkViewAt_of_ok` discharges `H_reg` where it holds at every valid witness (`coinedOf_reg`). -/
def ZkViewAtReg (C : CoinedSetup) (δ₁ : ℝ≥0∞) : Prop :=
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
