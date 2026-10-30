import Proofs.Flock.Claims.Recursion.NextP
import Proofs.Flock.VBridge.Family.Encode

/-!
# The record's proof digests written from the proofs' files (`decisionAtS`)

S17 (`verifyJWithM`) holds each proof file to the digest the record's value lists for its rep stream (`proof_{dk}`).
`decisionAtS` is `decisionAtP` at the value with `proof_{dk}` written from the files' digests, as `coinSubJ` writes the
live coins: one entry per rep stream, in `spec.repStreams`' order, its name and its file's digest (`shaSubJ`), the entry
the honest prover writes (`shaArr_honest`). The digests are the oracle's (`dgWith Claims.Hashes.oc`), the ones S17
computes.

Nothing else reads `proof_{dk}`. In the decision only the verdict's decoder does (`Record.decodeJWith`, into
`Session.proofSha`, which S17 alone reads); custody (`liveOfJ`) reads the link and the streams, the draw check
`unit_draw`. Outside it, check 5 (`RecCoins.Inner.load`) and the inner transcript's checks (`RecVStar.innerCommits`,
`RecVStar.innerLink`) read the inner record's streams, link points and draw. The rec-coins file's `record_sha512` binds
the record's whole text, but only on `--rec-inner-coins`' path, which candidate 3's session refuses
(`RecCoins.checkAtC3`), and what it binds (`Inner.load`'s values) reads no `proof_{dk}`.

* **Plain acceptance carries over** (`decisionAtS_canonical`): run against `H`, at a record value whose `proof_{dk}` is
  already the one the files' digests at `H` write (`shaSubJ` fixes it), the verdict is `decisionAtP`'s.
* **`shapeAt` is the verdict's shape** (`repFileShape_shapeAt`): at the link points `pts`, rep `r` of table `i` decodes
  its file at `shapeAt`'s shape for file `2 i + r` (`Flock.verifyRepWith_split`), and each shaped proof's file decodes
  to the proof (`decode_filesAt`), so the verdict's decode of `filesAt`'s files refuses none.
* **Oracle runs** (`EvOk`): a run against `H` that returns `a`; through binds, `mapM` and the record digest.
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-! ## Oracle runs that return -/

/-- `x` run against `H` returns `a`. -/
def EvOk {α : Type} (H : Claims.Msg → Claims.Msg) (x : Claims.OCV α) (a : α) : Prop :=
  ((x.run).eval H).1 = .ok a

theorem evOk_pure {α : Type} (H : Claims.Msg → Claims.Msg) (a : α) : EvOk H (pure a : Claims.OCV α) a := rfl

/-- A bind of a run that returns `a` runs as its continuation at `a`. -/
theorem evOk_bind_fst {α β : Type} {H : Claims.Msg → Claims.Msg} {x : Claims.OCV α} {f : α → Claims.OCV β} {a : α}
    (hx : EvOk H x a) : (((x >>= f).run).eval H).1 = (((f a).run).eval H).1 := by
  unfold EvOk at hx
  simp only [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind, hx]

theorem evOk_bind {α β : Type} {H : Claims.Msg → Claims.Msg} {x : Claims.OCV α} {f : α → Claims.OCV β} {a : α} {b : β}
    (hx : EvOk H x a) (hf : EvOk H (f a) b) : EvOk H (x >>= f) b := by
  unfold EvOk at hf ⊢
  rw [evOk_bind_fst hx]
  exact hf

theorem evOk_mapM_list {α β : Type} {H : Claims.Msg → Claims.Msg} {g : α → Claims.OCV β} {G : α → β}
    (hg : ∀ a, EvOk H (g a) (G a)) : ∀ l : List α, EvOk H (l.mapM g) (l.map G)
  | [] => by
    simp only [List.mapM_nil, List.map_nil]
    exact evOk_pure H []
  | a :: l => by
    rw [List.mapM_cons]
    exact evOk_bind (hg a) (evOk_bind (evOk_mapM_list hg l) (evOk_pure H _))

theorem evOk_mapM_array {α β : Type} {H : Claims.Msg → Claims.Msg} {g : α → Claims.OCV β} {G : α → β}
    (hg : ∀ a, EvOk H (g a) (G a)) (xs : Array α) : EvOk H (xs.mapM g) (xs.map G) := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]
  refine evOk_bind (evOk_mapM_list hg xs.toList) ?_
  have e : (xs.toList.map G).toArray = xs.map G := by rw [← Array.toList_map, Array.toArray_toList]
  rw [← e]
  exact evOk_pure H _

/-- The record digest through the oracle returns `dgOf` at `H`. -/
theorem evOk_dg (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (b : ByteArray) :
    EvOk H (spec.dgWith Claims.Hashes.oc b) (FlockComplete.dgOf spec (Claims.hOf H) b) := by
  unfold EvOk SessionSpec.dgWith FlockComplete.dgOf
  cases spec.sha512Digests <;> rfl

/-! ## The substitution -/

/-- `proof_{dk}` at the digests `ds`: rep stream `w`'s entry `{"stream": w, dk: hex d}`, in `spec.repStreams`' order. -/
def shaArr (spec : SessionSpec) (ds : Array ByteArray) : Json :=
  .arr ((spec.repStreams.zip ds).map fun q =>
    Json.mkObj [("stream", Json.str q.1.2.2), (spec.dk, FlockComplete.hexJ q.2)])

/-- **The record's value with its proof digests `ds`'s**: `proof_{dk}`'s value replaced in place by `shaArr` (`mapKey`,
so a value without the key keeps none). -/
def shaSubJ (spec : SessionSpec) (ds : Array ByteArray) (j : Json) : Json :=
  mapKey s!"proof_{spec.dk}" (fun _ => shaArr spec ds) j

/-- At the files' digests through `h`, `shaArr` is the honest record's `proof_{dk}` (`FlockComplete.JsonRT.proofObj`). -/
theorem shaArr_honest (spec : SessionSpec) (h : ByteArray → ByteArray) (bs : Array ByteArray) :
    shaArr spec (bs.map (FlockComplete.dgOf spec h)) =
      .arr ((spec.repStreams.zip bs).map (FlockComplete.JsonRT.proofObj spec h)) := by
  unfold shaArr
  rw [Array.zip_map_right, Array.map_map]
  rfl

/-- **The decision with the record's proof digests the files'**: the files' digests through the oracle, then
`decisionAtP` at the record's value with `proof_{dk}` written from them. -/
def decisionAtS (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    Claims.OCV Unit := do
  let ds ← (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).mapM (spec.dgWith Claims.Hashes.oc)
  decisionAtP s spec sts x link c₁ mids ⟨f.stmt, shaSubJ spec ds f.j, f.ps⟩

/-- The decision with the proof digests substituted: the statement's setup over `OC`, then `decisionAtS`. -/
def decisionSF (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalP) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAtS s spec sts x link c₁ mids f

/-- **Plain acceptance carries over**: run against `H`, at a record value that the substitution at the files' digests
at `H` fixes, `decisionAtS`'s verdict is `decisionAtP`'s. -/
theorem decisionAtS_canonical (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    {stmt : Claims.Msg} {j : Json} {ps : Array Flock.Proof}
    (hfix : shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) ps).map
      (FlockComplete.dgOf spec (Claims.hOf H))) j = j) :
    ((decisionAtS s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 =
      ((decisionAtP s spec sts x link c₁ mids ⟨stmt, j, ps⟩).run.eval H).1 := by
  dsimp only [decisionAtS]
  rw [evOk_bind_fst (evOk_mapM_array (evOk_dg H spec) (filesAt (tabsOf sts) (pointsOf spec c₁) ps)), hfix]

/-! ## The verdict's shape -/

/-- **`shapeAt` is the verdict's shape**: at the link points `pts`, where table `i`'s `fast100` schedule is `sch` and
its extra claims at `pts` are `extras`, rep `r < 2` of table `i` decodes its file at `shapeAt`'s shape for file
`2 i + r` (`Flock.verifyRepWith_split`). -/
theorem repFileShape_shapeAt {tabs : Array Flock.Setup} {pts : Array (Array F128)} {i r : ℕ} (hr : r < 2)
    {sch : Flock.Schedule} (hs : fast100 tabs[i]!.m = some sch) {extras : Array Flock.Claim}
    (he : tabs[i]!.extra pts = .ok extras) :
    Flock.repFileShape tabs[i]! sch extras.size = shapeAt tabs pts (2 * i + r) := by
  have hk : (2 * i + r) / 2 = i := by omega
  simp only [shapeAt, hk, hs, he, Option.getD_some]

/-- **A shaped proof's file decodes to it**: `filesAt`'s file `k` decodes at `shapeAt` to proof `k`, where that proof is
shaped for it (`FlockVBridge.Family.ProofShaped`). -/
theorem decode_filesAt {tabs : Array Flock.Setup} {pts : Array (Array F128)} {ps : Array Flock.Proof} {k : ℕ}
    (hk : k < ps.size) (hsh : FlockVBridge.Family.ProofShaped (shapeAt tabs pts k) ps[k]) :
    decodeProof (shapeAt tabs pts k) ((filesAt tabs pts ps)[k]'(by simpa only [filesAt, Array.size_mapIdx] using hk)) =
      .ok ps[k] := by
  simp only [filesAt, Array.getElem_mapIdx]
  exact FlockVBridge.Family.decodeProof_encodeProof _ _ hsh

end FlockDeploy
