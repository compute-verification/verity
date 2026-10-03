import FlockSoundness.Discharge.ZkLink
import FlockSoundness.Discharge.ZkExec
import FlockSoundness.Discharge.ZkView
import FlockSoundness.Discharge.ZkBind

/-!
# A draw's `--zk` table: its coins and sizes

What `coinedOf` (`Composed/Defs.lean`) and `innerOf` (`Composed/Inner.lean`) both read:

* `CoinsZK`: a table's coin vector, the coins `refused_coins` reads (`β`, `ρ`) and the others (`rest`).
* `RestZK`, `RestOf`: the other coins at a draw, every coin `verify --zk` draws for a rep, in its order: level 0's queries,
  the zerocheck's (`rSkip`, `rOuter`, `z`, the residual rounds' `ρ`), the lincheck's (`α`, `β`, its rounds', `r′`), the ring
  switch's (`rpp` per claim, `γ` per claim), level 0's out-of-domain and lane coins, and the inner proof's (`ε`, its
  constraints' `γ`, its queries).
* `pointsOf`: the rank check's claim points, from those coins (`abcClaims`; `point_parts`: a claim's point but its first
  coordinate), rep 0's `ab`, `c`, then rep 1's.
* `hiCount`, `padLenOf`, `padNOf`, `HiOf`: the hidden salts and the pads layout's sizes.
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
level 0's out-of-domain coins and lane challenges; the triple's sacrifice coin `ε`; the inner proof's constraints' `γ`
(ten); its `Q` queries. -/
structure RestZK (q0 L0 k0 m kLog nCl nOod mP : ℕ) where
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
  ood : Fin 2 → Fin nOod → F128
  lane : Fin 2 → Fin k0 → F256
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
coin, and the statement's `mPts` link coordinates. -/
abbrev RestOf (Tz : TabZK 𝔽 mPts) (st : Stmt) : Type :=
  RestZK (Tz.t / 2) (2 ^ Tz.sch.level₀.logLen) Tz.sch.k0 st.m st.c.kLog (2 + 2 * st.regions.size)
    (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).nOod0 st.mPts

/-- The draw's hidden-salt type. -/
abbrev HiOf (Tz : TabZK 𝔽 mPts) : Type := Fin (hiCount Tz) → ZkView.Salt

/-- **Rep `r`'s claims** (`abcClaims`) at its coins: `ab` at the lincheck's point, `c` at the zerocheck's. -/
def claimsOf {q0 L0 k0 m kLog nCl nOod mP : ℕ} (ρ : RestZK q0 L0 k0 m kLog nCl nOod mP) (r : Fin 2) : Array Claim :=
  abcClaims m kLog (⟨ρ.zc r, Array.ofFn (ρ.rho r), pinned ++ Array.ofFn (ρ.rOuter r), 0, 0, 0⟩ : ZcOut)
    (⟨0, ρ.rPrime r, (Array.ofFn (ρ.ts r)).reverse⟩ : LcOut)

/-- **The rank check's claim points** (`claim_word_points`): rep 0's `ab`, `c`, then rep 1's, each its claim's point
but the first coordinate (`point_parts`). -/
def pointsOf {q0 L0 k0 m kLog nCl nOod mP : ℕ} (ρ : RestZK q0 L0 k0 m kLog nCl nOod mP) (k : Fin Zk.RANK_POINTS) :
    List F128 :=
  ((claimsOf ρ (if k.val < 2 then 0 else 1))[k.val % 2]!).x.toList.drop 1

/-- **Region claim `j`'s point** (`extraClaims`: region `j / 2` at link point `j % 2`), but its first seven coordinates
(the six skip coordinates and `x₀`, as `point_parts` drops). -/
def regionPtOf (st : Stmt) (link : Fin 2 → Fin st.mPts → F128) (j : Fin (2 * st.regions.size)) : List F128 :=
  (st.regionPoint st.regions[j.val / 2]! (Array.ofFn (link (if j.val % 2 = 0 then 0 else 1)))).toList.drop 7

/-- `eq(p, x)` over the bits of word index `x`, least significant first (`zk_veil.rs`'s `eq_word`). -/
def eqWord (p : List F128) (x : ℕ) : F128 :=
  (List.range p.length).foldl (fun acc i => acc * if x.testBit i then p[i]! else 1 + p[i]!) 1

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
