import Proofs.Flock.Claims.Recursion.VerdictRowETarget
import Proofs.Flock.Claims.Recursion.VerdictRowDCoin
import Proofs.Flock.Claims.Recursion.VerdictRowDCover

/-!
# Row E's coin tie, and its target at row D's pieces

`Top.targetE` (`VerdictRowETarget`) takes the `rec-open` pieces, the reader of their sides and of their `coef`, and three
ties: the side (`hside`), the coin (`hcoef`) and the cover (`hcover`). Row D's pieces (`rowDPieces`) and side reader
(`sideAt`) come with their side tie (`DecAt.hside`) and cover (`DecAt.rowDPieces_cover`). This file adds the `coef`
reader and its tie, and states the target at those.

* `coefRead`: at a statement that sets up, the verifier rows at the piece's table's shape, its extra claims at the
  first coin's link points and its rep's `canonRounds` (comb empty: `kappa` and `info` don't read it,
  `verifierRows_comb`), and the `coef` they register for the piece's query: `κ[ℓ] · eq(α)[j]`, as `DecAt.coefAt`.
* `DecAt.coefRead_eq`, `DecAt.hcoef`: at a decision of an inner statement read at the dense form, `coefRead` reads
  `D.coefAt`, from `coins_of_decode` as `DecAt.posAt_eq` does.
* `Top.targetE_rowD`: `TargetE` at `accR`, row D's pieces and side reader, and `coefRead`. What it still takes:
  the pieces' running sums (`hholds`).
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP repLens cutLens tabsOf pointsOf setupOf jAt)
open FlockVBridge.AccChain (levelLanes)
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (HClass)

noncomputable section

/-- The `α` the verifier rows' consistency context for level `ℓ` reads, as `DecAt.alphaAt`. -/
def alphaOf (info : RecAlgebra.Info RecAlgebra.Lin) (ℓ : ℕ) : Array F128 :=
  match info.lg.conss.find? (·.start == ℓ) with
  | some c => c.alpha
  | none => #[]

/-- **The `coef` a `rec-open` piece reads off the statement and the coins**: at a statement that sets up, the verifier
rows at table `table`'s shape, its extra claims at the first coin's link points, rep `2 table + rep`'s `canonRounds` at
the coins after the link and no comb; the `coef` they register for query `query` of level `level`. -/
def coefRead (x : Claims.Msg) (cs : List Coin) (q : OpenPiece) : Array F128 :=
  match setupOf false x with
  | some v =>
    match verifierRows
        (shapeOf (tabsOf v.2.2)[q.table]! ((fast100 (tabsOf v.2.2)[q.table]!.m).getD default)
          (((tabsOf v.2.2)[q.table]!.extra (pointsOf v.2.1 (cs.headD default))).toOption.getD #[]).size)
        (canonRounds v.2.1 v.2.2 cs.tail.dropLast (2 * q.table + q.rep))
        (((tabsOf v.2.2)[q.table]!.extra (pointsOf v.2.1 (cs.headD default))).toOption.getD #[]) #[] with
    | .ok (_, info, kappa) =>
      FlockVBridge.coefAt (kappa[q.level]!)
        (levelLanes (shapeOf (tabsOf v.2.2)[q.table]! ((fast100 (tabsOf v.2.2)[q.table]!.m).getD default)
          (((tabsOf v.2.2)[q.table]!.extra (pointsOf v.2.1 (cs.headD default))).toOption.getD #[]).size) q.level)
        (eqTable (alphaOf info q.level))[q.query]!
    | .error _ => #[]
  | none => #[]

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128) (i : ℕ)

/-- **A piece reads its query's `coef`**: at a decision where the rep streams' names are distinct, a piece of table `i`
and rep `r` reads the `coef` the verifier rows at rep `r`'s values register for its query. -/
theorem coefRead_eq (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (r : Fin 2) (q : OpenPiece) (ht : q.table = i) (hr : q.rep = r.val) :
    coefRead x cs q = D.coefAt cls acc i r q.level q.query := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hv₀, -, hc₁, -, -, -, hpf, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hsp' : D.spec = spec := by rw [hsp, hd]
  have hsts' : D.sts = sts := by rw [hsts, hd]
  have hdec := D.decode
  rw [hd, hsp', hsts'] at hdec
  simp only [jAt] at hdec
  have hpts := (FlockDeploy.decodeJWith_sub _ hdec).1
  have hhead : cs.headD default = c₁ := by
    cases cs with
    | nil => cases hc₁
    | cons c cs => exact Option.some.inj hc₁
  have hk : 2 * i + r.val < 2 * sts.size := by rw [← hsts']; omega
  rw [hsp'] at hnd hti
  obtain ⟨hks, e⟩ := Claims.Bridge.Custody.repStreams_getElem spec hti r.isLt
  have hrounds : (D.rep cls acc i r).rounds = canonRounds spec sts cs.tail.dropLast (2 * i + r.val) := by
    show ((D.files cls i).stream r).rounds.map (·.coins) = _
    have hs : (D.files cls i).stream r =
        (D.sess.streams.find? (·.name == spec.repStreams[2 * i + r.val].2.2)).getD default := by
      rw [e]
      show (D.sess.streams.find? fun y => y.name == toString D.spec.tables[i]! ++ toString "/rep" ++
        toString (r : ℕ)).getD default = _
      rw [hsp']
    rw [hs]
    exact coins_of_decode _ hnd (readPs_blocks hpf) hdec hk hks
  have hext : (D.rep cls acc i r).extras =
      ((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[] := by
    show ((tabsOf D.sts)[i]!.extra D.sess.points).toOption.getD #[] = _
    rw [hsts', hpts, hhead]
  have hsh : D.sh cls i = shapeOf (tabsOf sts)[i]! ((fast100 (tabsOf sts)[i]!.m).getD default)
      (((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]).size := by
    show shapeOf (tabsOf D.sts)[i]! ((fast100 (tabsOf D.sts)[i]!.m).getD default)
      (((tabsOf D.sts)[i]!.extra D.sess.points).toOption.getD #[]).size = _
    rw [hsts', hpts, hhead]
  unfold coefRead DecAt.coefAt DecAt.alphaAt
  rw [hv₀]
  dsimp only
  rw [ht, hr, hsh, hrounds, hext, RecDeploy.Rows.verifierRows_comb _ _ _ (D.rep cls acc i r).comb]
  cases verifierRows (shapeOf (tabsOf sts)[i]! ((fast100 (tabsOf sts)[i]!.m).getD default)
      (((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]).size)
      (canonRounds spec sts cs.tail.dropLast (2 * i + r.val))
      (((tabsOf sts)[i]!.extra (pointsOf spec (cs.headD default))).toOption.getD #[]) #[] with
  | error e => rfl
  | ok t =>
    obtain ⟨t₁, t₂, t₃⟩ := t
    rfl

/-- **`hcoef`, the coin tie, in `targetE`'s form**: at a decision of an inner statement read at the dense form and a
table `i` of the statement, every piece of rep `r` of table `i` reads the verifier's `coef` for its query through
`coefRead`. -/
theorem hcoef {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size) (pieces : List OpenPiece) :
    ∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val →
      coefRead x cs q = D.coefAt cls acc i r q.level q.query :=
  fun r q _ ht hr => D.coefRead_eq cls acc i (D.names_nodup hst hf hx) hi hti r q ht hr

end DecAt

end

end RecDeploy.Rows

namespace RecDeploy.Top

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open RecDeploy.Rows (accHolds sideAt coefRead rowDPieces)

/-- **Row E's target at row D's pieces**: `targetE` at `accR`, the pieces `rowDPieces key`, their side reader
`sideAt rd` and `coefRead`. The side, coin and cover ties are `DecAt.hside`, `DecAt.hcoef` and
`DecAt.rowDPieces_cover`, and the dense statement's one table is `DecAt.tables_of_dense`; what's left is that the
pieces hold their running sums where `P` does (`hholds`). -/
theorem targetE_rowD (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop)
    (chainAt : Msg → List Msg → ℕ → ℕ → Array F128) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hholds : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), P x ms cs →
      ∀ q ∈ rowDPieces key x, accHolds (sideAt rd) coefRead chainAt q x ms cs) :
    TargetE rd P (accR rd chainAt) :=
  targetE rd P chainAt (rowDPieces key) (sideAt rd) coefRead hholds
    (fun x ms _ _ D cls i hi => D.hside cls i hi (rowDPieces key x))
    (fun x ms cs hA D cls i hi => by
      obtain ⟨⟨xo, sto, hst, hf, hx⟩, -⟩ := id hA
      have hT := D.tables_of_dense hst hf hx
      exact D.hcoef cls (accR rd chainAt x ms cs) i hst hf hx hi (by omega) _)
    (fun _ _ _ _ D cls _ hi r => D.rowDPieces_cover cls key hi r)

end RecDeploy.Top
