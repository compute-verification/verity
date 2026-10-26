import Proofs.Flock.Claims.Recursion.VStarHh
import Proofs.Flock.Claims.Recursion.VStarCap0

/-!
# Round 0 at the deployed builder: RecCap's part holds the inner prover's first message (`held0_A0`)

At the deployed assigner `A0` and the fitted records, RecCap's part holds round 0 (`capHold`) as its `N` slots' commit
strings at port `top`. Slot `n` reads round 0's `n`-th fitted row under its salt (`portRow_cap0`), so where it commits
(`SlotCommits`) and the trace places its row (`TraceOk` at the fitted port inputs) it outputs the inner prover's `n`-th
commit string (`hmCommit_A0`). So the part holds the inner prover's round-0 message, the round's commit strings end to end
(`strs`), and no other part holds round 0.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)
open Claims.Recursion.VStarB (readForms readSegs LayoutB)

/-- **A slot at the deployed assigner outputs the inner prover's commit string** (`hmCommit_vstarWit` at `A0`, where
`TraceOk` holds at the fitted port inputs). -/
theorem hmCommit_A0 {mask : ℕ → ℕ → Bool} {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape}
    {hist : List (Msg × Msg)} {s : ℕ} {st : _root_.VStar.Stmt} {c : Circuit} {src : _root_.VStar.In.Src}
    {pin : _root_.VStar.In.PartIn} (hp : partAt x s = some (st, c)) (hsrc : srcOf I x w ρ hist = some src)
    (hi : _root_.VStar.In.partIn src (progOf c) st = .ok pin) (h : VStarTrace.TraceOk c (fitPin c pin))
    {nm : String} {q n : ℕ} {ws : Array ℕ} {y : Salt} {ps : List ℕ}
    (hsc : SlotCommits c nm q n (vstarWit (A0 mask) I x w ρ hist s))
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hrw : portRow (fitPin c pin) nm n = some ⟨ws, saltBytes y⟩)
    (hps : rowHidden c nm n = some ps) :
    readForms (vstarWit (A0 mask) I x w ρ hist s) (hmCommit c q n) =
      hmLeaf (asX (digestOf ws) + maskX y, saltH FlockDeploy.sha512 y) := by
  rw [vstarWit_eq hp hsrc hi] at hsc ⊢
  simp only [A0] at hsc ⊢
  rw [hmCommit_placed (VStarTrace.placesR_assign h mask) hsc hq hrw hps (rowPfx_of_traceOk h hq hrw)]
  show FlockSoundness.Binding.commitString (rowMsg ws) (saltBytes y).data.toList = _
  rw [show (saltBytes y).data.toList = List.ofFn y by simp [saltBytes]]
  exact commitString_rowMsg ws y

theorem readSegs_cmts0 {α : Type} (w : Claims.Flock.Wit) (l : List α) (f : α → List HmNets.Lin) :
    readSegs w (l.map fun a => (([] : Msg), f a)) = (l.map fun a => readForms w (f a)).flatten := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.map_cons, List.map_cons, List.flatten_cons, ← ih]
    simp [readSegs]

theorem strs_length_rows (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (h : List (Msg × Msg)) :
    (strs I w ρ h).length = (I.rows w ρ.1 h).length := by
  simp [strs]

/-- Round 0's `n`-th commit string: hm96's leaf of the `n`-th row under the salt `ρ.2 0 n`. -/
theorem strs_getElem?0 (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (n : ℕ) :
    (strs I w ρ [])[n]? = ((I.rows w ρ.1 [])[n]?).map fun ws =>
      hmLeaf (asX (digestOf ws) + maskX (ρ.2 0 n), saltH FlockDeploy.sha512 (ρ.2 0 n)) := by
  simp only [strs, List.getElem?_map, List.getElem?_mapIdx, Option.map_map, List.length_nil]
  try rfl

theorem roundRows_length0 (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) :
    (roundRows I w ρ hist 0).length = (I.rows w ρ.1 []).length := by
  simp only [roundRows, List.take_zero, List.length_mapIdx]

theorem roundRows_getElem?0 (I : InnerRec) (w : Claims.Flock.Wit) (ρ : I.Tape) (hist : List (Msg × Msg)) (n : ℕ) :
    (roundRows I w ρ hist 0)[n]? =
      ((I.rows w ρ.1 [])[n]?).map fun ws => (⟨ws, saltBytes (ρ.2 0 n)⟩ : _root_.VStar.In.Row) := by
  simp only [roundRows, List.take_zero, List.getElem?_mapIdx]
  try rfl

/-- **RecCap's slots output round 0's commit strings**, in order, at the deployed assigner: where `TraceOk` holds at the
fitted port inputs, each slot commits and has its row's input columns. -/
theorem cap0_commits {mask : ℕ → ℕ → Bool} {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} {rs : Array RecVStar.SchedRound} {p N : ℕ} {c : Circuit}
    (hr : readX x = some r) (hrs : repSched r.shape = some rs) (hcap : capAt r = some (p, N))
    (hc : r.w.parts[p]?.bind partCircuit = some c) (hcp : RecVStar.checkParsed (.cap 0 N) c = .ok ())
    (hT : ∀ src pin, srcOf (I.fit x) x w ρ hist = some src →
      _root_.VStar.In.partIn src (progOf c) (.cap 0 N) = .ok pin → VStarTrace.TraceOk c (fitPin c pin))
    (hsc : ∀ n < N, SlotCommits c "top" 0 n (vstarWit (A0 mask) (I.fit x) x w ρ hist p))
    (hps : ∀ n < N, ∃ ps, rowHidden c "top" n = some ps) :
    (List.range N).map (fun n => readForms (vstarWit (A0 mask) (I.fit x) x w ρ hist p) (hmCommit c 0 n)) =
      strs (I.fit x) w ρ [] := by
  obtain ⟨src, pin, hsrc, hi, hlen, hrow⟩ :=
    portRow_cap0 (I := I) (w := w) (ρ := ρ) (hist := hist) hr hrs hcap hc hcp
  obtain ⟨p0, p1, hp, h0, -, -⟩ := cap_ports hcp
  have hq : c.ports.findIdx? (·.name == "top") = some 0 := capPort_cap hp h0
  have hTO := hT src pin hsrc hi
  have hpa := partAt_cap0 hr hcap hc
  have hrl : ((I.fit x).rows w ρ.1 []).length = N := by
    rw [← hlen, roundRows_length0]
  apply List.ext_getElem?
  intro n
  by_cases hn : n < N
  · have hn' : n < ((I.fit x).rows w ρ.1 []).length := by rw [hrl]; exact hn
    have h1 := hrow n hn
    rw [roundRows_getElem?0, List.getElem?_eq_getElem hn', Option.map_some] at h1
    obtain ⟨ps, hps'⟩ := hps n hn
    rw [List.getElem?_map, List.getElem?_range hn, Option.map_some, strs_getElem?0, List.getElem?_eq_getElem hn',
      Option.map_some, hmCommit_A0 hpa hsrc hi hTO (hsc n hn) hq h1 hps']
  · have hl1 : ((List.range N).map fun n =>
        readForms (vstarWit (A0 mask) (I.fit x) x w ρ hist p) (hmCommit c 0 n)).length ≤ n := by
      rw [List.length_map, List.length_range]
      omega
    have hl2 : (strs (I.fit x) w ρ []).length ≤ n := by
      rw [strs_length_rows, hrl]
      omega
    rw [List.getElem?_eq_none_iff.2 hl1, List.getElem?_eq_none_iff.2 hl2]

/-- **What a part holds of round 0, at the deployed builder**: RecCap's part holds the inner prover's round-0 message,
its commit strings end to end; no other part holds round 0. -/
theorem held0_A0 (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    {mask : ℕ → ℕ → Bool} {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} {rs : Array RecVStar.SchedRound} {p N : ℕ} {c : Circuit}
    (hr : readX x = some r) (hrs : repSched r.shape = some rs) (hcap : capAt r = some (p, N))
    (hc : r.w.parts[p]?.bind partCircuit = some c) (hcp : RecVStar.checkParsed (.cap 0 N) c = .ok ())
    (hT : ∀ src pin, srcOf (I.fit x) x w ρ hist = some src →
      _root_.VStar.In.partIn src (progOf c) (.cap 0 N) = .ok pin → VStarTrace.TraceOk c (fitPin c pin))
    (hsc : ∀ n < N, SlotCommits c "top" 0 n (vstarWit (A0 mask) (I.fit x) x w ρ hist p))
    (hps : ∀ n < N, ∃ ps, rowHidden c "top" n = some ps) {s : ℕ} {m : Msg}
    (hm : (DeployedR.layR elem).held x s (vstarWit (A0 mask) (I.fit x) x w ρ hist s) 0 = some m) :
    m = (strs (I.fit x) w ρ []).flatten := by
  have hm0 : (DeployedR.layR elem).msgAt x s 0 = capHold r s := by
    show (readX x).bind (fun r => capHold r s) = _
    rw [hr, Option.bind_some]
  have hf : (DeployedR.layR elem).fixedAt x 0 = none := rfl
  unfold LayoutB.held at hm
  rw [hf, hm0] at hm
  simp only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at hm
  obtain ⟨p0, p1, hp, h0, -, -⟩ := cap_ports hcp
  unfold capHold at hm
  simp only [hcap, Option.bind_some] at hm
  by_cases hs : s = p
  · subst hs
    simp only [eq_self_iff_true, ↓reduceIte, hc, Option.bind_some, capPort_cap hp h0, Option.map_some, Option.some.injEq] at hm
    rw [← hm, readSegs_cmts0, cap0_commits hr hrs hcap hc hcp hT hsc hps]
  · simp [hs] at hm

end RecDeploy.VStarBuild
