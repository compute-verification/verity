import Proofs.Flock.Claims.Recursion.RestFinal
import Proofs.Flock.Claims.Recursion.RestLive
import Proofs.Flock.Claims.Recursion.DeployChecked

/-!
# `ReadsAt` at the deployed readers `rdR`, `finalR` and `restR elem`

`hrest_read` takes `ReadsAt rd final fin`. At `rdR`, `finalR` and `restR elem` it holds wherever `elem` lays the
element-level last message (`ElemFramed elem`), every rep's opened rounds fit (`OpenedFit elem`) and the proofs read
off them are shaped for their files (`OpenedShaped elem`) (`readsAt`). Every other input of `readOkR_of` is proved
here:

* the inner statement sets up and passes its scope check (`parse_innerOk`);
* `finalR` reads `elem`'s pieces as the last message (`finalR_last`), which `rdR` reads (`ElemFramed`), so its tag is
  `1`;
* each earlier round reads back as its opened rows at every length (`rowsOf_joinR'`), round 0 as many as the link's
  nodes and the rounds after it as many as the reps' schedules (`finalReads`' counts);
* the record value reads as custody's (`decLive`).

`OpenedFit elem` and `OpenedShaped elem` are the parts that read the opened rows' contents: every rep's block of the
rounds after the link fits some proof and `T` (`FitsRep`), at round 0's nodes (`nodesR`), and the proofs `readPs` reads
there are shaped (`ShapedAt`).
-/

namespace RecDeploy.RestRead

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open RecDeploy.DeployedR
open FlockDeploy (Coin FinalP ReadOkR FitsRep repCtx repLens cutLens joinR rowsOf rowsOf_joinR')
open Lean (Json)

/-- A round's opened rows, as `rowsOf` reads them back. -/
def rowsR (o : List (Msg × Salt)) : List ByteArray := (o.map Prod.fst).map fun r => (⟨r.toArray⟩ : ByteArray)

/-- Round 0's link nodes, as `decR` reads them off the opened rows. -/
def nodesR (o : List (Msg × Salt)) : List ByteArray := (rowsR o).map FlockDeploy.nodeOf

/-- **Every rep's opened rounds fit**: at a dense compiled statement, witnesses one a part whose calls are true, and
coins `c`, wherever `finalR` reads the last message as `k - 1` rounds, each rep's block of the rounds after the link
fits some proof and `T` at the rep's context. -/
def OpenedFit (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = nS x →
    (∀ e ∈ (layAt (restR elem)).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
    ∀ os last, finalR (lastAt (restR elem) x ws) = some (os, last) → os.length + 1 = kI x →
      ∀ v, FlockDeploy.setupOf false (xIOf x) = some v → ∀ k < 2 * v.2.2.size,
        ∃ pT : Flock.Proof × F128,
          FitsRep (repCtx v.2.1 v.2.2 (c 0) (nodesR (os.headD [])) ((List.range (kI x)).map c).tail.dropLast k)
            ((cutLens (repLens v.2.1 v.2.2) (os.tail.map rowsR)).getD k []) pT.1 pT.2

/-- **The proofs read off the opened rounds are shaped for their files** (`decR`'s `ShapedAt` filter), at the same
places as `OpenedFit`, at the last message's proof. -/
def OpenedShaped (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) : Prop :=
  ∀ x, DenseX x → ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = nS x →
    (∀ e ∈ (layAt (restR elem)).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) →
    ∀ os last, finalR (lastAt (restR elem) x ws) = some (os, last) → os.length + 1 = kI x →
      ∀ v, FlockDeploy.setupOf false (xIOf x) = some v → ∀ fl, rdR last = some fl → ∀ pf,
        FlockDeploy.readPs v.2.1 v.2.2 (c 0) (nodesR (os.headD [])) (os.tail.map rowsR)
          ((List.range (kI x)).map c).tail.dropLast fl = some pf →
        FlockDeploy.ShapedAt v.2.1 v.2.2 (c 0) pf.1

theorem mapM_map_some {α β γ : Type} {f : β → Option γ} {g : α → β} {h : α → γ} (hf : ∀ a, f (g a) = some (h a)) :
    ∀ l : List α, (l.map g).mapM f = some (l.map h)
  | [] => rfl
  | a :: l => by simp [List.mapM_cons, hf, mapM_map_some hf l]

/-- **The last step reads at the deployed readers**, wherever `elem` lays the element-level last message and every
rep's opened rounds fit, with shaped proofs. -/
theorem readsAt (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (hE : ElemFramed elem) (hfit : OpenedFit elem) (hsh : OpenedShaped elem) :
    ReadsAt rdR finalR (restR elem) := by
  intro x hx c ws hl hT os last hf hos
  obtain ⟨st0, hst, -⟩ := id hx
  obtain ⟨s, spec, sts, hv, hck, -⟩ := parse_innerOk hst
  obtain ⟨r, hr, hlast⟩ := finalR_last elem hx hl hf
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  obtain ⟨ps, hps, hrd⟩ := hE x r hr ws (hl.trans hn)
  rw [← hlast] at hrd
  obtain ⟨os', last', hf', -, hcnt⟩ := finalReads elem x hx ws hl
  rw [hf] at hf'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hf')
  obtain ⟨r', rs, s', spec', st, hr', hrs, hset, hm, hsc, hk⟩ := schedBridge hx
  rw [hv] at hset
  obtain ⟨rfl, rfl, rfl⟩ : s = s' ∧ spec = spec' ∧ sts = #[st] := by
    simpa using Option.some.inj hset
  obtain ⟨o₀, ot, rfl⟩ : ∃ o₀ ot, os = o₀ :: ot := by
    cases os with
    | nil => simp at hos; omega
    | cons o₀ ot => exact ⟨o₀, ot, rfl⟩
  have hot : ot.length = 2 * rs.size := by simp at hos; omega
  obtain ⟨n, hkn⟩ : ∃ n, kI x = n + 1 := ⟨kI x - 1, by omega⟩
  have hRL : repLens spec #[st] = [rs.size, rs.size] := by
    have e0 : (#[st] : Array Stmt)[0]! = st := rfl
    unfold repLens
    rw [show 2 * (#[st] : Array Stmt).size = 2 from rfl, show List.range 2 = [0, 1] from rfl]
    simp only [List.map_cons, List.map_nil, Nat.zero_div, show (1 : ℕ) / 2 = 0 from rfl, e0, hsc,
      Array.length_toList]
  refine readOkR_of (a := last) (fl := ⟨xIOf x, Json.null, ps⟩) (c₁ := c 0) (r₀ := nodesR o₀)
    (rss := ot.map rowsR) hv (by exact List.getLast?_concat) ?_ hrd rfl ?_ ?_ ?_ ?_ (hps _ hv) ?_ ?_
    (hsh x hx c ws hl hT (o₀ :: ot) last hf hos _ hv _ hrd) ?_ hck (decLive rdR)
  · intro t ht
    rw [ht] at hrd
    simp [rdR] at hrd
  · rw [hkn, List.range_succ_eq_map]
    rfl
  · show (rowsOf (joinR (o₀.map Prod.fst))).map (List.map FlockDeploy.nodeOf) = some (nodesR o₀)
    rw [rowsOf_joinR']
    rfl
  · simp only [nodesR, rowsR, List.length_map, FlockDeploy.level0Lens]
    have h0 := hcnt 0 (by simp)
    simp only [List.getD_cons_zero] at h0
    rw [h0, nAtSched_zero hv]
    simp
  · simp only [List.map_cons, List.cons_append, List.tail_cons, List.dropLast_concat]
    exact mapM_map_some (fun o => rowsOf_joinR' _) ot
  · rw [hRL, List.length_map, hot]
    simp only [List.sum_cons, List.sum_nil]
    omega
  · intro k hk
    exact hfit x hx c ws hl hT (o₀ :: ot) last hf hos _ hv k hk
  · simp only [List.length_append, List.length_map, List.length_cons, List.length_range, List.length_singleton]
    omega

end RecDeploy.RestRead
