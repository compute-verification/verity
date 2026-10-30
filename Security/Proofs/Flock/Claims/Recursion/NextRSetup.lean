import Proofs.Flock.Claims.Recursion.NextRFits

/-!
# An honest run decides at every set-up statement

`ReadsBack` takes, besides the statement's setup, an honest proof and cap per rep. At every statement that sets up
(`setupOf`) whose tables have a `fast100` schedule and extra claims at the first coin's link points, those exist: rep
`k`'s proof is `sizedProof` at its shape (`honestP`), and table `i`'s cap is its first rep's (`honestCaps`). So on that
honest run `decR` decides (`honestRun_decides`), wherever its rows fit a message (`RowsFit`) and the last message reads
as its proofs.

No statement that sets up is exhibited here: every statement with `hm96-sha512` rows (which `Ins.setup` requires) has a
circuit file of 22 MB or more, beyond what a Lean file holds, and `setupOf` reads JSON (`Json.parse`), which the kernel
can't unfold.
-/

namespace FlockDeploy

open Flock

/-- Rep `k`'s extra claims at the first coin's link points, as `repCtx` counts them. -/
def repExtra (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (k : ℕ) : ℕ :=
  match (tabsOf sts)[k / 2]!.extra (pointsOf spec c₁) with
  | .ok e => e.size
  | .error _ => 0

theorem repCtx_n (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (r₀ : List ByteArray) (cs : List Coin) (k : ℕ) :
    (repCtx spec sts c₁ r₀ cs k).n = 2 + repExtra spec sts c₁ k := rfl

/-- **Rep `k`'s honest proof**: `sizedProof` at the rep's table, schedule and shape, at its zerocheck point in the
coins `cs` (`repCtx`'s `z`, which reads no cap). -/
def honestP (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (cs : List Coin) (k : ℕ) : Flock.Proof :=
  sizedProof (tabsOf sts)[k / 2]! ((fast100 (tabsOf sts)[k / 2]!.m).getD default) (repCtx spec sts c₁ [] cs k).z
    (FlockSoundness.Refine.shapeOf (tabsOf sts)[k / 2]! ((fast100 (tabsOf sts)[k / 2]!.m).getD default)
      (repExtra spec sts c₁ k))

/-- The honest proofs, two reps a table. -/
def honestPs (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (cs : List Coin) : Array Flock.Proof :=
  (Array.range (2 * sts.size)).map (honestP spec sts c₁ cs)

/-- Each table's level-0 cap: its first rep's, which reads no coin (`honestP_cap`). -/
def honestCaps (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) : Array (Array ByteArray) :=
  (Array.range sts.size).map fun i => (honestP spec sts c₁ [] (2 * i)).cap

theorem honestPs_size (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (cs : List Coin) :
    (honestPs spec sts c₁ cs).size = 2 * sts.size := by
  unfold honestPs
  rw [Array.size_map, Array.size_range]

theorem honestPs_get {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {cs : List Coin} {k : ℕ}
    (hk : k < 2 * sts.size) : (honestPs spec sts c₁ cs)[k]! = honestP spec sts c₁ cs k := by
  have hk' : k < (honestPs spec sts c₁ cs).size := by rw [honestPs_size]; exact hk
  rw [getElem!_pos (honestPs spec sts c₁ cs) k hk']
  simp only [honestPs, Array.getElem_map, Array.getElem_range]

theorem honestCaps_size (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) :
    (honestCaps spec sts c₁).size = sts.size := by
  unfold honestCaps
  rw [Array.size_map, Array.size_range]

theorem honestCaps_get {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {i : ℕ} (hi : i < sts.size) :
    (honestCaps spec sts c₁)[i]! = (honestP spec sts c₁ [] (2 * i)).cap := by
  have hi' : i < (honestCaps spec sts c₁).size := by rw [honestCaps_size]; exact hi
  rw [getElem!_pos (honestCaps spec sts c₁) i hi']
  simp only [honestCaps, Array.getElem_map, Array.getElem_range]

/-- A rep's honest cap is its table's: it reads only `k / 2`, and no coin. -/
theorem honestP_cap (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (cs : List Coin) (k : ℕ) :
    (honestP spec sts c₁ cs k).cap = .replicate (2 ^ (((fast100 (tabsOf sts)[k / 2]!.m).getD default).levels.getD 0
      default).cap) (zeroBytes (tabsOf sts)[k / 2]!.merkle.digestLen) := rfl

theorem tabsOf_get! {sts : Array Stmt} {i : ℕ} (hi : i < sts.size) : (tabsOf sts)[i]! = Setup.ofCircuit sts[i] := by
  have hi' : i < (tabsOf sts).size := by
    unfold tabsOf
    rw [Array.size_map]
    exact hi
  rw [getElem!_pos (tabsOf sts) i hi']
  exact Array.getElem_map ..

/-- **An honest run decides at every set-up statement** whose tables have a `fast100` schedule and extra claims at the
first coin's link points: on the honest prover's messages from `honestCaps` and `honestPs`, wherever their rows fit a
message and the last message reads as `honestPs`, shaped for their files, at the statement, `decR` decides. -/
theorem honestRun_decides {rd : Claims.Msg → Option FinalP} {x last : Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {Ts : ℕ → F128} {fl : FinalP}
    (hx : setupOf false x = some (s, spec, sts)) (hc : cs.head? = some c₁)
    (htab : ∀ (i : ℕ) (hi : i < sts.size),
      (fast100 sts[i].m).isSome ∧ ∃ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e)
    (hfit0 : RowsFit ((capRows (honestCaps spec sts c₁)).map nodeRow))
    (hfit : ∀ k < (honestPs spec sts c₁ cs.tail.dropLast).size,
      ∀ rs ∈ repRows (repCtx spec sts c₁ (capRows (honestCaps spec sts c₁)) cs.tail.dropLast k)
        (honestPs spec sts c₁ cs.tail.dropLast)[k]! (Ts k), RowsFit rs)
    (hrd : rd last = some fl) (hps : fl.ps = honestPs spec sts c₁ cs.tail.dropLast) (hst : fl.stmt = x)
    (hsh : ShapedAt spec sts c₁ (honestPs spec sts c₁ cs.tail.dropLast)) :
    ∃ d, decR rd x (honestMsgs (honestCaps spec sts c₁)
      ((List.range (honestPs spec sts c₁ cs.tail.dropLast).size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows (honestCaps spec sts c₁)) cs.tail.dropLast k)
          (honestPs spec sts c₁ cs.tail.dropLast)[k]! (Ts k)) last) cs = some d := by
  have hsch : ∀ (i : ℕ) (hi : i < sts.size),
      fast100 (tabsOf sts)[i]!.m = some ((fast100 (tabsOf sts)[i]!.m).getD default) := by
    intro i hi
    obtain ⟨sch, hs⟩ := Option.isSome_iff_exists.1 (htab i hi).1
    rw [tabsOf_get! hi]
    show fast100 sts[i].m = some ((fast100 sts[i].m).getD default)
    rw [hs, Option.getD_some]
  have hcaps : ∀ i < (honestCaps spec sts c₁).size,
      (honestCaps spec sts c₁)[i]!.size = RecDeploy.level0Nodes sts[i]! := by
    intro i hi
    rw [honestCaps_size] at hi
    obtain ⟨sch, hs⟩ := Option.isSome_iff_exists.1 (htab i hi).1
    have h2 : 2 * i / 2 = i := by omega
    rw [honestCaps_get hi, honestP_cap, Array.size_replicate, h2, tabsOf_get! hi, getElem!_pos sts i hi]
    show 2 ^ (((fast100 sts[i].m).getD default).levels.getD 0 default).cap =
      2 ^ ((fast100 sts[i].m).map fun s => (s.levels.getD 0 default).cap).getD 0
    rw [hs, Option.getD_some, Option.map_some, Option.getD_some]
  have hrep : ∀ k < (honestPs spec sts c₁ cs.tail.dropLast).size,
      SchedOf (repCtx spec sts c₁ (capRows (honestCaps spec sts c₁)) cs.tail.dropLast k) ∧
        HonestRep (repCtx spec sts c₁ (capRows (honestCaps spec sts c₁)) cs.tail.dropLast k)
          (honestPs spec sts c₁ cs.tail.dropLast)[k]! ∧
          (honestPs spec sts c₁ cs.tail.dropLast)[k]!.cap = (honestCaps spec sts c₁)[k / 2]! := by
    intro k hk
    rw [honestPs_size] at hk
    have hk2 : k / 2 < sts.size := by omega
    obtain ⟨-, e, he⟩ := htab (k / 2) hk2
    have hs := schedOf_repCtx (r₀ := capRows (honestCaps spec sts c₁)) (cs := cs.tail.dropLast) hk2
      (htab (k / 2) hk2).1 he
    obtain ⟨-, -, -, -, hlv⟩ := RecDeploy.fast100_sizes (hsch (k / 2) hk2)
    have hL : 2 ≤ ((fast100 (tabsOf sts)[k / 2]!.m).getD default).levels.size := by
      rw [hlv]
      split_ifs <;> omega
    rw [honestPs_get hk]
    refine ⟨hs, ⟨⟨repExtra spec sts c₁ k, repCtx_n .., sizedProof_sizes ..⟩, hL, sizedProof_nonces ..,
      (sizedProof_unabsorbed ..).1, (sizedProof_unabsorbed ..).2.1, (sizedProof_unabsorbed ..).2.2⟩, ?_⟩
    have h2 : 2 * (k / 2) / 2 = k / 2 := by omega
    rw [honestCaps_get hk2, honestP_cap, honestP_cap, h2]
  exact ⟨_, readsBack rd x last cs s spec sts c₁ (honestCaps spec sts c₁) (honestPs spec sts c₁ cs.tail.dropLast) Ts
    fl hx hc (honestCaps_size ..) hcaps (honestPs_size ..) hrep hfit0 hfit hrd hps hst hsh⟩

end FlockDeploy
