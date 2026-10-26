import Proofs.Flock.Claims.Recursion.VerdictRowRLeaf
import Proofs.Flock.Claims.Recursion.PartsVStar

/-!
# The concrete `rec-residuals` pieces and their owners (`resPD`, `ownRD`)

Row R's leaf asks `resP x` to hold, at every decision `At` admits, a piece of part `p`'s `S` keyed by group `p` at a
copy `(o, g)` with `o + g = r`, for each rep `r` and group `p` (`ResPiecesAt`), and every piece to have an owner
(`PiecesOwned`).

* `resPD x`: at the inner statement `x`, one piece per group `p` of the verifier's algebra at table 0's coin-free
  shape (`Rows.shapeNoCoin`) and rep `r`, keyed `(p, r, 0, 0)` (instance `r`, `ResLower.keyCopy`), at part `p`'s
  `S`.
* `ownRD x q`: the first of V\*'s statements at `x` that is a `rec-residuals` part of group `q.key.1` (`isResOf`).
* `resPiecesAt_D`: `ResPiecesAt` at `resPD`, at every decision, where each group of that shape has its part
  (`PartsOkD`): at a dense decision there is one table, whose shape is `shapeNoCoin` (`DecAt.sh_eq`).
* `piecesOwned_D`: `PiecesOwned` at `resPD` and `ownRD`, where V\* lists a `rec-residuals` part of each group
  (`ResListedD`) and every `rec-open` query has its level's part (`OpenOwned`). An owner found is a statement, and
  there are as many parts as statements (`readXPre_spec`).
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Flock.RecAlgebra (structureOf)
open Flock.RecResiduals (portSizes groups)
open RecDeploy.Rows (DecAt shapeNoCoin decR_last)
open RecDeploy.RestRead (DenseX)
open RecDeploy.OpenLower (ownQR)

/-- The verifier's algebra's groups at shape `sh`, counted; `0` where it doesn't group. -/
def groupsN (sh : Shape) : ℕ :=
  match structureOf sh with
  | .ok S =>
    match groups S (portSizes sh) with
    | .ok gs => gs.size
    | .error _ => 0
  | .error _ => 0

/-- Group `p`'s pieces at shape `sh`: one per rep `r`, keyed `(p, r, 0, 0)`, at part `p`'s `S`; none where the part
doesn't build. -/
def resGroup (sh : Shape) (p : ℕ) : List ResPiece :=
  match RecResiduals.part sh p with
  | .ok (S, _, _) => [⟨(p, 0, 0, 0), S⟩, ⟨(p, 1, 0, 0), S⟩]
  | .error _ => []

/-- **The `rec-residuals` pieces** at the inner statement `x`: each group's (`resGroup`) at table 0's coin-free shape. -/
def resPD (x : Msg) : List ResPiece :=
  match setupOf false x with
  | some v => (List.range (groupsN (shapeNoCoin v.2.1 v.2.2))).flatMap (resGroup (shapeNoCoin v.2.1 v.2.2))
  | none => []

/-- A V\* statement is a `rec-residuals` part of group `jj`, as its name says (`RecResiduals.params`). -/
def isResOf (jj : ℕ) : VStar.Stmt → Bool
  | .inner _ nm _ => (RecResiduals.params nm).map (·.2.2.2) == some jj
  | _ => false

/-- **A piece's owner**: the first of V\*'s statements at the compiled statement `x` that is a `rec-residuals` part of
the piece's group. -/
def ownRD (x : Msg) (q : ResPiece) : Option ℕ :=
  (readX x).bind fun r => (_root_.Flock.RecVStar.statements r.shape).toOption.bind fun ss =>
    ss.findIdx? (isResOf q.key.1)

/-- **Each group of table 0's coin-free shape has its part**, at every dense compiled statement. -/
def PartsOkD : Prop :=
  ∀ x, DenseX x → ∀ v, setupOf false (xIOf x) = some v → ∀ p < groupsN (shapeNoCoin v.2.1 v.2.2),
    ∃ S ns nv, RecResiduals.part (shapeNoCoin v.2.1 v.2.2) p = .ok (S, ns, nv)

/-- **V\* lists a `rec-residuals` part of each group** of table 0's coin-free shape, at every dense compiled
statement. -/
def ResListedD : Prop :=
  ∀ x, DenseX x → ∀ v, setupOf false (xIOf x) = some v → ∀ p < groupsN (shapeNoCoin v.2.1 v.2.2),
    ∃ r ss, readX x = some r ∧ _root_.Flock.RecVStar.statements r.shape = .ok ss ∧ ∃ sv ∈ ss, isResOf p sv = true

/-- **Every `rec-open` query has an owner** (`PiecesOwned`'s second half): its level's part. -/
def OpenOwned (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) : Prop :=
  ∀ x, DenseX x → ∀ q ∈ Rows.rowDPieces key (xIOf x), ∃ s < nS x, ownQR x q = some s

theorem groupsN_eq {sh : Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf sh = .ok S) (hg : groups S (portSizes sh) = .ok gs) : groupsN sh = gs.size := by
  simp [groupsN, hS, hg]

theorem mem_resPD {x : Msg} {q : ResPiece} (hq : q ∈ resPD x) :
    ∃ v, setupOf false x = some v ∧ q.key.1 < groupsN (shapeNoCoin v.2.1 v.2.2) := by
  unfold resPD at hq
  cases hv : setupOf false x with
  | none => rw [hv] at hq; cases hq
  | some v =>
    rw [hv] at hq
    obtain ⟨p, hp, hq⟩ := List.mem_flatMap.1 hq
    refine ⟨v, rfl, ?_⟩
    unfold resGroup at hq
    split at hq
    · rcases List.mem_cons.1 hq with rfl | hq
      · exact List.mem_range.1 hp
      · rw [List.mem_singleton] at hq
        subst hq
        exact List.mem_range.1 hp
    · cases hq

/-- **`ResPiecesAt` at the concrete pieces**: at a dense decision there is one table (`tables_of_dense`), whose shape
is `shapeNoCoin` at the statement's setup (`DecAt.sh_eq`), and `resPD` holds each group's two pieces there. -/
theorem resPiecesAt_D (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) (hok : PartsOkD) :
    ResPiecesAt rd resPD key side posAt := by
  intro x ms cs hA D cls i hi S gs hS hg r p hp
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -⟩ := id hA
  obtain ⟨hT1, -⟩ := D.tables_of_dense hst hf hx
  obtain rfl : i = 0 := by omega
  have hG := D.rowG cls (fun _ => #[]) hst hf hx hi
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, -, hd⟩ := decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hds : D.d.spec = v.2.1 := by rw [hd]
  have hdt : D.d.sts = v.2.2 := by rw [hd]
  have hsh : D.sh cls 0 = shapeNoCoin v.2.1 v.2.2 := by
    rw [D.sh_eq cls (fun _ => #[]) hG hi, hsp, hsts, hds, hdt]
  rw [hsh] at hS hg ⊢
  have hN := groupsN_eq hS hg
  have hv' : setupOf false (xIOf xo) = some v := by rw [← hx]; exact hv
  have hp' : p < groupsN (shapeNoCoin v.2.1 v.2.2) := by rw [hN]; exact hp
  obtain ⟨S', ns, nv, hpart⟩ := hok xo ⟨sto, hst, hf⟩ v hv' p hp'
  refine ⟨⟨(p, r.val, 0, 0), S'⟩, ?_, ⟨ns, nv, hpart⟩, rfl, by simp⟩
  unfold resPD
  rw [hv]
  refine List.mem_flatMap.2 ⟨p, List.mem_range.2 hp', ?_⟩
  unfold resGroup
  rw [hpart]
  rcases r with ⟨_ | _ | n, hr⟩
  · exact List.mem_cons_self
  · exact List.mem_cons_of_mem _ List.mem_cons_self
  · omega

/-- **`PiecesOwned` at the concrete pieces and owners**: each piece's group has a `rec-residuals` part among V\*'s
statements (`ResListedD`), the first of which `ownRD` finds, below the statements' count, which is the parts'
(`readXPre_spec`). -/
theorem piecesOwned_D (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (hL : ResListedD) (hQ : OpenOwned key) :
    PiecesOwned resPD key ownRD := by
  intro x hX
  refine ⟨fun q hq => ?_, hQ x hX⟩
  obtain ⟨v, hv, hp⟩ := mem_resPD hq
  obtain ⟨r, ss, hr, hss, sv, hsv, hres⟩ := hL x hX v hv q.key.1 hp
  have hsome : (ss.findIdx? (isResOf q.key.1)).isSome = true := by
    rw [Array.findIdx?_isSome]
    exact Array.any_eq_true'.2 ⟨sv, hsv, hres⟩
  obtain ⟨s, hs⟩ := Option.isSome_iff_exists.1 hsome
  obtain ⟨hsz, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs
  obtain ⟨-, vs, hvs, hlen⟩ := readXPre_spec (readXPre_of_readX hr)
  rw [hss] at hvs
  cases hvs
  refine ⟨s, ?_, ?_⟩
  · simp only [nS, hr]
    omega
  · simp [ownRD, hr, hss, Except.toOption, hs]

end RecDeploy.DecompBR
