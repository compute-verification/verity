import Proofs.Flock.Claims.Recursion.VStarFit

/-!
# A slot of the deployed builder's witness outputs the inner prover's commit string

hh and hp's rounds before the last read, at part `s`'s outer witness, the hm96 slot of a port at an instance
(`hmCommit`). At the deployed builder's witness (`vstarWit` at the trace assigner `VStarTrace.A`) that slot outputs the
commit string the inner prover sent for the row the stager placed there:

* the slot outputs its commit string at its row's input columns and salt (`SlotCommits`, at any witness `Flock.check`
  accepts at a staged part, `hmSlotsCommit_of_full`);
* the trace places the stager's row and salt there (`PlacesR`, at `TraceOk`, `placesR_assign`), and the port's word
  count is the row's (`TraceOk.rows`), so that commit string is the row's message's (`rowMsg`) under the salt
  (`hmCommit_placed`);
* which is the hm96 leaf the inner prover sends, `b ‖ c` of the row's digest under the salt (`commitString_rowMsg`).

`rowHidden_staged` gives the row's input columns at every slot `layoutBTops` reads at a staged part.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)
open Claims.Recursion.VStarB (readForms formBit)

theorem ofFn_xor (d : Fin 64 → UInt8) (y : Salt) :
    List.ofFn (asX d + maskX y) = List.ofFn (fun i : Fin 64 => d i ^^^ maskOf y i) := rfl

theorem xorBytes_ofFn (D K : List UInt8) (d m : Fin 64 → UInt8) (hDl : D.length = 64) (hKl : K.length = 64)
    (hD : ∀ i : Fin 64, d i = D.getD i.val 0) (hK : ∀ i : Fin 64, m i = K.getD i.val 0) :
    FlockSoundness.Merkle.xorBytes D K = List.ofFn (fun i : Fin 64 => d i ^^^ m i) := by
  apply List.ext_getElem (by rw [FlockSoundness.Merkle.xorBytes, List.length_zipWith, hDl, hKl, List.length_ofFn]; rfl)
  intro i h1 h2
  have hi : i < 64 := by rw [List.length_ofFn] at h2; exact h2
  have hiD : i < D.length := by omega
  have hiK : i < K.length := by omega
  rw [List.getElem_ofFn, hD, hK]
  simp only [FlockSoundness.Merkle.xorBytes, List.getElem_zipWith, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem hiD, List.getElem?_eq_getElem hiK, Option.getD_some]

theorem digestOf_getD (ws : Array ℕ) (i : Fin 64) :
    digestOf ws i = (FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 (rowMsg ws))).getD i.val 0 := by
  unfold digestOf
  exact congrArg (fun l : List UInt8 => l.getD i.val 0) (sha512_eq (rowMsg ws))

theorem maskOf_getD (y : Salt) (i : Fin 64) :
    maskOf y i = (FlockSoundness.Refine.K512.mask (List.ofFn y)).getD i.val 0 := by
  unfold maskOf
  rw [FlockSoundness.Refine.K512_mask, List.toArray_ofFn, VStarTrace.get!_eq, List.getD_eq_getElem?_getD,
    Array.getElem?_toList] <;> rfl

/-- **hm96's commit string of a row's message is the inner prover's leaf**: `SHA-512(rowMsg) ⊕ M·y`, then `c`. -/
theorem commitString_rowMsg (ws : Array ℕ) (y : Salt) :
    FlockSoundness.Binding.commitString (rowMsg ws) (List.ofFn y) =
      hmLeaf (asX (digestOf ws) + maskX y, saltH FlockDeploy.sha512 y) := by
  have hsp : Claims.toMsg Hm96.Default512.saltPrefix = FlockSoundness.Refine.K512.saltPrefix := by
    unfold FlockSoundness.Refine.K512; rfl
  have hKl : (FlockSoundness.Refine.K512.mask (List.ofFn y)).length = 64 := by
    rw [FlockSoundness.Refine.K512.mask_len]; rfl
  unfold FlockSoundness.Binding.commitString hmLeaf saltH
  dsimp only
  refine congrArg₂ HAppend.hAppend ?_ ?_
  · rw [ofFn_xor]
    exact xorBytes_ofFn (FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 (rowMsg ws)))
      (FlockSoundness.Refine.K512.mask (List.ofFn y)) (digestOf ws) (maskOf y)
      (FlockSoundness.Discharge.Hm.length_digBytes _) hKl (digestOf_getD ws) (maskOf_getD y)
  · rw [hsp, sha512_eq]

/-- **The port's row prefix is the placed row's**, where `TraceOk` holds: its rows are of the port's words. -/
theorem rowPfx_of_traceOk {c : Circuit} {pin : _root_.VStar.In.PartIn} (h : VStarTrace.TraceOk c pin) {nm : String}
    {q n : ℕ} {rw : _root_.VStar.In.Row} (hq : c.ports.findIdx? (·.name == nm) = some q)
    (hr : portRow pin nm n = some rw) : rowPfx c q = (HmRow.rowPrefix rw.words.size).data.toList := by
  obtain ⟨hlt, hname, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq
  have hnm : c.ports[q].name = nm := by simpa using hname
  have hw := (h.rows q hlt n rw (by rw [hnm]; exact hr)).1
  simp [rowPfx, Array.getElem?_eq_getElem hlt, hw]

/-- **A placed slot outputs its row's commit string**: where the slot commits (`SlotCommits`), the trace places the
stager's row and salt at port `nm`'s instance `n` (`PlacesR`), and the port's row prefix is the row's. -/
theorem hmCommit_placed {c : Circuit} {pin : _root_.VStar.In.PartIn} {w : Claims.Flock.Wit} {nm : String} {q n : ℕ}
    {rw : _root_.VStar.In.Row} {ps : List ℕ} (hP : VStarTrace.PlacesR c pin w) (hsc : SlotCommits c nm q n w)
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hr : portRow pin nm n = some rw)
    (hps : rowHidden c nm n = some ps) (hw : rowPfx c q = (HmRow.rowPrefix rw.words.size).data.toList) :
    readForms w (hmCommit c q n) = FlockSoundness.Binding.commitString (rowMsg rw.words) rw.salt.data.toList := by
  unfold SlotCommits at hsc
  rw [hsc]
  have hrow : readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var) =
      Claims.Recursion.VStar.bytesOf (_root_.VStar.In.wordBits rw.words) := by
    rw [hps, Option.getD_some, Claims.Recursion.VStarB.readForms_pos]
    unfold Claims.Recursion.VStar.readAt
    rw [hP.places.row nm n rw ps hr hps, hP.take_full hr hps]
  have hsl : readForms w (hmSalt c q n) = rw.salt.data.toList := by
    unfold Claims.Recursion.VStarB.readForms
    rw [hP.places.salt nm q n rw hq hr, HpLast.bytesOf_saltBits]
  rw [hrow, hsl, hw]
  rfl

/-- **Every slot `layoutBTops` reads at a staged part has its row's input columns**: each held port's (`heldRows`) and,
at a RecCap, port `top`'s at every instance (`rowHidden_eq`, the ports among unit 0's leaves, as in
`hmSlotsCommit_of_full`). -/
theorem rowHidden_staged {st : Stage} {x xO : Msg} {s : ℕ} {r : XR} {c : Circuit} (hS : StagedPart st s x xO)
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) :
    (∀ h ∈ heldRows r s, ∃ ps, rowHidden c h.2.1 h.2.2.2 = some ps) ∧
      ∀ l N, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N) → ∀ q, capPort c = some q →
        ∀ n, ∃ ps, rowHidden c "top" n = some ps := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss, hss, hsv⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨u, -, -, -, hl, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  have hsv' : (Flock.RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv := by
    rw [hss]
    exact hsv
  rw [hc0] at hrows hl
  refine ⟨fun h hh => ?_, fun l N hN q hq n => ?_⟩
  · obtain ⟨q, hq, k, hk, hrep, hnm, hqi⟩ := mem_heldRows hsv' hpc hh
    obtain ⟨hout, -⟩ := ownPorts_held hq
    rw [hnm]
    have hqs := (Array.findIdx?_eq_some_iff_getElem.1 hqi).1
    have hN := in_block hrows hout hqi
    rw [getElem!_pos c.ports _ hqs] at hN
    exact ⟨_, rowHidden_eq _ hl hqi hqs hN⟩
  · rw [hsv'] at hN
    have e : sv = .cap l N := Option.some.inj hN
    subst e
    have htop : "top" ∉ (VStar.Stmt.cap l N).sub.outputs := by show "top" ∉ ["out"]; decide
    have hqs := (Array.findIdx?_eq_some_iff_getElem.1 hq).1
    have hN' := in_block hrows htop hq
    rw [getElem!_pos c.ports _ hqs] at hN'
    exact ⟨_, rowHidden_eq _ hl hq hqs hN'⟩

/-- **A slot at the deployed builder's witness outputs the inner prover's commit string** of the row the stager placed
there: at part `s`, where the stager accepts, `TraceOk` holds, the slot commits and the stager placed the row `ws`
under the salt `saltBytes y` at port `nm`'s instance `n`. -/
theorem hmCommit_vstarWit {mask : ℕ → ℕ → Bool} {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape}
    {hist : List (Msg × Msg)} {s : ℕ} {st : _root_.VStar.Stmt} {c : Circuit} {src : _root_.VStar.In.Src}
    {pin : _root_.VStar.In.PartIn} (hp : partAt x s = some (st, c)) (hsrc : srcOf I x w ρ hist = some src)
    (hi : _root_.VStar.In.partIn src (progOf c) st = .ok pin) (h : VStarTrace.TraceOk c pin)
    {nm : String} {q n : ℕ} {ws : Array ℕ} {y : Salt} {ps : List ℕ}
    (hsc : SlotCommits c nm q n (vstarWit (VStarTrace.A mask) I x w ρ hist s))
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hrw : portRow pin nm n = some ⟨ws, saltBytes y⟩)
    (hps : rowHidden c nm n = some ps) :
    readForms (vstarWit (VStarTrace.A mask) I x w ρ hist s) (hmCommit c q n) =
      hmLeaf (asX (digestOf ws) + maskX y, saltH FlockDeploy.sha512 y) := by
  rw [vstarWit_eq hp hsrc hi] at hsc ⊢
  rw [hmCommit_placed (VStarTrace.placesR_assign h mask) hsc hq hrw hps (rowPfx_of_traceOk h hq hrw)]
  show FlockSoundness.Binding.commitString (rowMsg ws) (saltBytes y).data.toList = _
  rw [show (saltBytes y).data.toList = List.ofFn y by simp [saltBytes]]
  exact commitString_rowMsg ws y

end RecDeploy.VStarBuild
