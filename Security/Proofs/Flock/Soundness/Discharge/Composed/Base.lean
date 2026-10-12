import Proofs.Flock.Soundness.Discharge.ZkLink
import Proofs.Flock.Soundness.Discharge.ZkExec
import Proofs.Flock.Soundness.Discharge.ZkView
import Proofs.Flock.Soundness.Discharge.ZkBind

/-!
# A draw's `--zk` table: its coins and sizes

What `coinedOf` (`Composed/Coined.lean`) and `innerOf` (`Composed/Inner.lean`) both read:

* `CoinsZK`: a table's coin vector, the coins `refused_coins` reads (`β`, `ρ`) and the others (`rest`).
* `RestZK`, `RestOf`: the other coins at a draw, every coin `verify --zk` draws for a rep, in its order: level 0's queries,
  the zerocheck's (`rSkip`, `rOuter`, `z`, the residual rounds' `ρ`), the lincheck's (`α`, `β`, its rounds', `r′`), the ring
  switch's (`rpp` per claim, `γ` per claim), level 0's out-of-domain and lane coins, Ligerito's coins after `T'`, and the inner proof's (`ε`, its
  constraints' `γ`, its queries).
* `pointsOf`: the rank check's claim points, from those coins (`abcClaims`; `point_parts`: a claim's point but its first
  coordinate), rep 0's `ab`, `c`, then rep 1's.
* `hiCount`, `padLenOf`, `padNOf`, `HiOf`, `VLOf`: the hidden salts, the pads layout's sizes and the value lists.
* `eqWord`, `maskWordsOf`: the rank check's weights and the mask slot's words.
* `regionPtOf`, `pubRegOf`: the region claims' points and public ring-switch messages at the link points.
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession Refine

local notation "𝔽" => FlockLevel3.GF128

variable {mPts : ℕ}

/-- **A table's coin vector**: each rep's inner-proof `β` (`Zk.verifyInner`) and Ligerito's `ρ` (`ZkLigerito`), the coins
`refused_coins` reads but the rank check's points, and the table's other coins. -/
structure CoinsZK (Rest : Type) where
  beta : Fin 2 → F128
  rho : Fin 2 → F256 × F256
  rest : Rest

/-- **A `--zk` table's other coins**: the session's two link points (`mP` coordinates, shared by both reps,
the region claims' points through `st.extra`), then each rep's, in the order `verify --zk` draws them: `q₀` level-0 queries over `L₀`
positions; the zerocheck's skip coins, outer coins (rounds `7 … m−7`), `z` and the residual rounds' `ρ`; the lincheck's
`α`, `β`, its `kLog − 6` rounds' coins and `r′`; the ring switch's 7 coins per claim and `γ` per claim (`nCl` claims);
level 0's out-of-domain points (`z ← sampleVec (m − 7)`) and coins (`b`), and its lane challenges; then, after the restart
`T'` (and `CoinsZK.rho`), every coin `ZkLigerito.ligerito` draws from there on (`ligZ` … `ligB`, below); the triple's
sacrifice coin `ε`; the inner proof's constraints' `γ` (ten); its `Q` queries.

Ligerito's coins after `T'`, for `R + 1` levels with `qs j` queries at level `j`, in draw order: at each split `j < R` (level
0's after the lane phase, then level `j`'s inside fold round `j − 1`), two out-of-domain samples (`oodIntro`: a point
`z ← sampleVec (m − 6 − k0 − 3j)` and its coin `b`); level 0's consistency coins `α` (`queryLevel0`; its query coins are
`q0s`, as positions) and its batching coin `b₀`; then per fold round `i < R` its four challenges (F256) and, at level `i + 1`,
the query coins (`sampleVec q`), the consistency coins `α` (`log2 (q − 1) + 1` of them) and the batching coin. Fold round `i`'s
split, for `i + 1 < R`, comes after its challenges and before its queries. -/
structure RestZK (q0 L0 k0 m kLog nCl nOod mP R : ℕ) (qs : ℕ → ℕ) where
  link : Fin 2 → Fin mP → F128
  q0s : Fin 2 → Fin q0 → Fin L0
  rSkip : Fin 2 → Fin 6 → F128
  rOuter : Fin 2 → Fin (m - 13) → F128
  zc : Fin 2 → F128
  rho : Fin 2 → Fin (m - 6) → F128
  lcAlpha : Fin 2 → F128
  lcBeta : Fin 2 → F128
  ts : Fin 2 → Fin (kLog - 6) → F128
  rPrime : Fin 2 → F128
  rpp : Fin 2 → Fin nCl → Fin 7 → F128
  rsGamma : Fin 2 → Fin nCl → F128
  oodPt : Fin 2 → Fin nOod → Fin (m - 7) → F128
  ood : Fin 2 → Fin nOod → F128
  lane : Fin 2 → Fin k0 → F256
  ligZ : Fin 2 → (j : Fin R) → Fin 2 → Fin (m - 6 - k0 - 3 * j.val) → F128
  ligZB : Fin 2 → Fin R → Fin 2 → F128
  ligAlpha0 : Fin 2 → Fin (Nat.log2 (qs 0 - 1) + 1) → F128
  ligB0 : Fin 2 → F128
  ligFold : Fin 2 → Fin R → Fin 4 → F256
  ligQ : Fin 2 → (i : Fin R) → Fin (qs (i.val + 1)) → F128
  ligAlpha : Fin 2 → (i : Fin R) → Fin (Nat.log2 (qs (i.val + 1) - 1) + 1) → F128
  ligB : Fin 2 → Fin R → F128
  zeta : Fin 2 → F128
  gammas : Fin 2 → Fin 10 → F128
  qP : Fin 2 → Fin Zk.Q → F128

/-- The salts of levels ≥ 1, one per leaf of each level's tree in each rep (`Table`'s `Hi`). -/
def hiCount (Tz : TabZK 𝔽 mPts) : ℕ := 2 * (Tz.sch.levels.tail.map fun l => 2 ^ l.logLen).sum

/-- The pads message's length before its `K_PAD` padding symbols (`verify --zk`'s `PadLayout.new st.m st.kLog k0`). -/
def padLenOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : ℕ := (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).len

/-- The pads code's length (`verifyInner`'s `n`: the message rounded up to a power of two, at rate `2^-3`). -/
def padNOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : ℕ :=
  (padLenOf Tz st + Zk.K_PAD).nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE

/-- **The draw's other coins**: `q₀ = t_pad / 2` level-0 queries (`Zk.schedule`) over `2^logLen` positions, the
executable statement's `m` and `kLog`, a claim per region and link point besides `ab` and `c` (`extraClaims`), and the layout's one out-of-domain
coin, the statement's `mPts` link coordinates, and Ligerito's `R = |levels| − 1` fold rounds with each level's query count. -/
abbrev RestOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : Type :=
  RestZK (Tz.t / 2) (2 ^ Tz.sch.level₀.logLen) Tz.sch.k0 st.m st.c.kLog (2 + 2 * st.regions.size)
    (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).nOod0 st.mPts (Tz.sch.levels.length - 1)
    (fun j => (Tz.sch.levels.getD j default).queries)

/-- **The draw's value lists** (`Table`'s `VL`): Ligerito's restart `T'` (a clear K message, `Level0Msgs.tPrime`) and levels
≥ 1's bytes. -/
abbrev VLOf : Type := FlockLevel3.GF256 × ByteArray

/-- The draw's hidden-salt type. -/
abbrev HiOf (Tz : TabZK 𝔽 mPts) : Type := Fin (hiCount Tz) → ZkView.Salt

/-- **Rep `r`'s claims** (`abcClaims`) at its coins: `ab` at the lincheck's point, `c` at the zerocheck's. -/
def claimsOf {q0 L0 k0 m kLog nCl nOod mP R : ℕ} {qs : ℕ → ℕ} (ρ : RestZK q0 L0 k0 m kLog nCl nOod mP R qs) (r : Fin 2) : Array Claim :=
  abcClaims m kLog (⟨ρ.zc r, Array.ofFn (ρ.rho r), pinned ++ Array.ofFn (ρ.rOuter r), 0, 0, 0⟩ : ZcOut)
    (⟨0, ρ.rPrime r, (Array.ofFn (ρ.ts r)).reverse⟩ : LcOut)

/-- **The rank check's claim points** (`claim_word_points`): rep 0's `ab`, `c`, then rep 1's, each its claim's point
but the first coordinate (`point_parts`). -/
def pointsOf {q0 L0 k0 m kLog nCl nOod mP R : ℕ} {qs : ℕ → ℕ} (ρ : RestZK q0 L0 k0 m kLog nCl nOod mP R qs) (k : Fin Zk.RANK_POINTS) :
    List F128 :=
  ((claimsOf ρ (if k.val < 2 then 0 else 1))[k.val % 2]!).x.toList.drop 1

/-- **Region claim `j`'s point** (`extraClaims`: region `j / 2` at link point `j % 2`), but its first seven coordinates
(the six skip coordinates and `x₀`, as `point_parts` drops). -/
def regionPtOf (st : Stmt) (link : Fin 2 → Fin st.mPts → F128) (j : Fin (2 * st.regions.size)) : List F128 :=
  (st.regionPoint st.regions[j.val / 2]! (Array.ofFn (link (if j.val % 2 = 0 then 0 else 1)))).toList.drop 7

/-- `eq(p, x)` over the bits of word index `x`, least significant first (`zk_veil.rs`'s `eq_word`). -/
def eqWord (p : List F128) (x : ℕ) : F128 :=
  (List.range p.length).foldl (fun acc i => acc * if x.testBit i then p[i]! else 1 + p[i]!) 1

/-- **The mask slot's word indices in the packed witness** (`flock-circuit.rs`'s `mask_words`): every block's slot, block
`b`'s words at `b·2^(k_log−7) + start/128 + j` for `j < 2^(slot_log−7)`. -/
def maskWordsOf (st : Stmt) : List ℕ :=
  match st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => []
  | some r => (List.range (2 ^ st.nbl)).flatMap fun b =>
      (List.range (2 ^ (r.slotLog - 7))).map fun j => b * 2 ^ (st.c.kLog - 7) + r.start / 128 + j

/-- Column `i` is in the first mask range's word-aligned span, decidably (`Inner/Holds.lean`'s `MaskCol`). -/
def maskColB (st : Stmt) (i : ℕ) : Bool :=
  match st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) with
  | none => false
  | some r => decide (r.start / 128 ≤ i / 128 ∧ i / 128 < r.start / 128 + 2 ^ (r.slotLog - 7))

/-- **Δ avoids the mask slot**: no pair of Δ (`da`, `db`) has a row or a column in the mask range's span. A condition on
the parsed statement (`avoidsMask`), which `HmRow.check` does not imply: `verify --zk` checks it (`Zk.maskFree`). -/
def deltaAvoidsMask (st : Stmt) : Bool :=
  (st.da.toList ++ st.db.toList).all fun p => !maskColB st p.1 && !maskColB st p.2

/-- Every cell of every region (`regionsOf`, the in-block address of each region bit) lies outside the mask range. -/
def regionCellsOff (st : Stmt) : Bool :=
  (regionsOf st).all fun R => (List.range (2 ^ R.free.length)).all fun i => !maskColB st (R.inBlock i)

/-- Every region differs from every mask-slot word at an in-block word bit it fixes: some bit `i + 7 < k_log`, not free,
whose fixed value is not bit `i` of the word. -/
def regionWordsOff (st : Stmt) : Bool :=
  st.regions.toList.all fun reg => (maskWordsOf st).all fun x =>
    (List.range (st.c.kLog - 7)).any fun i => !reg.free.contains (i + 7) &&
      ((reg.mask >>> (i + 7)) % 2 == 1 && (reg.fixed >>> (i + 7)) % 2 == 1) != x.testBit i

/-- **The regions avoid the mask slot**, by cell and by word: `HmRow.check` does not imply it, since `Circuit.outRegion`
and `HmRow.regions`' tail region sit at an output group's column, which `Net.ofRows` bounds by the unit's slot (D11) but
not away from the mask range; `verify --zk` checks it (`Zk.maskFree`). -/
def regionsAvoidMask (st : Stmt) : Bool := regionCellsOff st && regionWordsOff st

/-- **The statement avoids the mask slot**: its Δ (`deltaAvoidsMask`) and its regions (`regionsAvoidMask`). A condition on
the parsed statement that the honest prover doesn't check: its `Stmt::new` (`backends/flock/live/src/circuit.rs`) refuses
only a column that shares a word with a partial region's bits and isn't forced to zero (`region_word_violation`), and
checks no Δ pair or region cell against the mask span. `verify --zk` refuses a statement that fails it (`Zk.shapeOk`'s
`Zk.maskFree`, D12), so it holds at every `--zk` setup (`FailClosed.avoidsMask_of_shapeOk`). The zero-knowledge
statements here take it as a hypothesis (`hdm`); `FailClosed.ViewZ` and `ViewZHJ` take it from the setup. No proof
derives it from D11's bound on output groups and the layout checks (Δ's forced-zero column `ur.slot 0 + useful` can lie
past the unit's slot); the check makes it a fact of every `--zk` setup. -/
def avoidsMask (st : Stmt) : Bool := deltaAvoidsMask st && regionsAvoidMask st

/-- **A region's public bit at witness index `idx`** (`regionValue`): bit `i` of the region's bytes at block `b`, `i` the
index's free bits and `b` its block bits. Only its values inside the region's subcube matter: a region claim's `eq`
weight vanishes off it (`regionPoint` fixes the other coordinates). -/
def regionBit (st : Stmt) (reg : Region) (idx : ℕ) : ZMod 2 :=
  let i := (List.range reg.free.size).foldl (fun a t => a + if idx.testBit reg.free[t]! then 2 ^ t else 0) 0
  let b := (List.range st.nbl).foldl (fun a k => a + if idx.testBit (st.c.kLog + k) then 2 ^ k else 0) 0
  if ((reg.bytes b).get! (i / 8) >>> (i % 8).toUInt8) % 2 == 1 then 1 else 0

/-- **The region claims' public ring-switch messages**: region claim `j`'s message `Σ_x eq(p_j, x)·bit_v(word x)` with
the region's public bits (`regionBit`) for the witness's. -/
noncomputable def pubRegOf (st : Stmt) (link : Fin 2 → Fin st.mPts → F128) :
    Fin (2 * st.regions.size) → Fin 128 → 𝔽 := fun j v =>
  ∑ x : Fin (2 ^ (st.m - 7)), fOf (eqWord (regionPtOf st link j) x.val) *
    ZK.Table.castF (F := 𝔽) (regionBit st st.regions[j.val / 2]! (128 * x.val + v.val))

end FlockSoundness.Discharge.Composed
