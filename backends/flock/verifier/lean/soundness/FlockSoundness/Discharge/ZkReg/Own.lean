import FlockSoundness.Discharge.ZkReg.Sound

/-!
# The verifier's own registered reads, from F

`zk_session_soundR` holds the registered reads that are the verifier's own at the outcome's draw (`OwnReads`) to the
registered check. Here are those reads, built from F (`own`) and the population alone, and the theorem that
they are the verifier's own at every live outcome the session accepts:

* `ownReads own`: at row `(u, q)`, F's port named as the circuit's port `q` (`ownPort`, `checkPort`'s `find?`), at
  the position it gives for `u`'s unit in the population's header (`popUnits`, `Registered.unitIndices`); a read
  wherever that position is some. The registrant's openings (`wv`, `ws`, `wpath`, `committed`) are parameters.
  `ownReads_shape` is its shape.
* `drawnOwn`: `ownReads` at the rows the drawn units hold (`laySh_pos_unit`: a row a unit holds is its own).
* `ownAtDraws_of_check`: at a live outcome `(R, S, wr S R lo)` that `verify --zk --registered F` accepts, `drawnOwn`
  is the verifier's own at the record's draw, from the coin server's record (`RecordsLiveZK`) and its draw proving `S`
  (`DrawOkZK`'s first conjunct): the check reads the units `Registered.tableUnits` reads, at the draw `dj S`.
* `zk_session_soundR_own`: `zk_session_soundR` at `drawnOwn`, with `OwnReads` discharged.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open Composed
open scoped ENNReal

/-- The population's units: its header's `units.indices` (`Registered.unitIndices`), empty when it has none. -/
def popUnits (pub : Public) : Array ℕ :=
  ((Registered.unitIndices pub.header).toOption).getD #[]

/-- F's port named as the circuit's port `q`: the first of `own` whose name's first port of the circuit is `q`. -/
def ownPort (own : Array Registered.Port) (c : Flock.Circuit) (q : ℕ) : Option Registered.Port :=
  own.find? fun reg => (List.range c.ports.size).find? (fun j => c.ports[j]!.name == reg.name) == some q

/-- The position F gives at row `(u, q)`: `ownPort`'s read in `u`'s unit of the population `pub`. -/
def ownPos (own : Array Registered.Port) (c : Flock.Circuit) (pub : Public) (u q : ℕ) : Option ℕ :=
  (ownPort own c q).bind fun reg => (reg.reads[(popUnits pub).getD u 0]?).join

variable {n : ℕ} {Gate : Type}

/-- F reads row `(u, q)`: it gives a position there. -/
def ownIsRead (own : Array Registered.Port) (c : Flock.Circuit) (pub : Public) :
    (Fin n × Fin c.ports.size) ⊕ Gate → Prop
  | .inl (u, q) => ∃ k, ownPos own c pub u.val q.val = some k
  | .inr _ => False

/-- F's port at row `(u, q)`. -/
def ownPortAt (own : Array Registered.Port) (c : Flock.Circuit) : (Fin n × Fin c.ports.size) ⊕ Gate → Registered.Port
  | .inl (_, q) => (ownPort own c q.val).getD default
  | .inr _ => default

/-- F's position at row `(u, q)`. -/
def ownReadAt (own : Array Registered.Port) (c : Flock.Circuit) (pub : Public) :
    (Fin n × Fin c.ports.size) ⊕ Gate → ℕ
  | .inl (u, q) => (ownPos own c pub u.val q.val).getD 0
  | .inr _ => 0

section

variable {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}
  {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hscope : Layout.scopeOk c = true)

/-- **The reads the verifier builds from F** (`own`) over the population `pub`, at the registrant's openings
`wv`, `ws`, `wpath` (`committed`: they open F's root at F's positions). -/
def ownReads (own : Array Registered.Port)
    (wv : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
    (ws : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → List UInt8)
    (wpath : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → ByteArray)
    (committed : ∀ p, ownIsRead own c pub p →
      Flock.Registered.opensLeaf (ownPortAt own c p)
        ((Binding.ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc
          (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf (wv p) (ws p)).1 (ownReadAt own c pub p) (wpath p) = true) :
    FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsSh ht dj y₀ hc hU hpub hscope)
    where
  isRead := ownIsRead own c pub
  read_row p h := by
    rcases p with _ | _
    · rfl
    · exact h.elim
  port := ownPortAt own c
  readAt := ownReadAt own c pub
  wv := wv
  ws := ws
  wpath := wpath
  committed := committed

/-- **`ownReads` at the rows the units of `S` hold** (the bridge's `restrict _ (drawnReads _ S)`). -/
def drawnOwn (own : Array Registered.Port)
    (wv : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
    (ws : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → List UInt8)
    (wpath : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → ByteArray)
    (committed : ∀ p, ownIsRead own c pub p →
      Flock.Registered.opensLeaf (ownPortAt own c p)
        ((Binding.ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc
          (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf (wv p) (ws p)).1 (ownReadAt own c pub p) (wpath p) = true)
    (S : Finset (Fin n)) :
    FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsSh ht dj y₀ hc hU hpub hscope)
    where
  isRead p := ownIsRead own c pub p ∧
    ∃ u ∈ S, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
      (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).pos g = p
  read_row p h := (ownReads ht dj y₀ hc hU hpub hscope own wv ws wpath committed).read_row p h.1
  port := ownPortAt own c
  readAt := ownReadAt own c pub
  wv := wv
  ws := ws
  wpath := wpath
  committed p h := committed p h.1

/-- **`ownReads`'s shape**: each read is a row `(u, q)`, its port one of `own` and the circuit's port `q` by name, at the
position it gives for `u`'s unit of the population. -/
theorem ownReads_shape {own : Array Registered.Port}
    {p : (Fin n × Fin c.ports.size) ⊕ Gate} (hp : ownIsRead own c pub p) :
    ∃ u q, p = .inl (u, q) ∧ ownPortAt own c p ∈ own ∧
      (List.range c.ports.size).find? (fun j => c.ports[j]!.name == (ownPortAt own c p).name) = some q.val ∧
      (ownPortAt own c p).reads[(popUnits pub).getD u.val 0]? = some (some (ownReadAt own c pub p)) := by
  rcases p with ⟨u, q⟩ | g
  · obtain ⟨k, hk⟩ : ∃ k, ownPos own c pub u.val q.val = some k := hp
    have hk0 := hk
    unfold ownPos at hk
    cases hreg : ownPort own c q.val with
    | none => simp [hreg] at hk
    | some reg =>
      simp only [hreg, Option.bind_some] at hk
      have hport : ownPortAt own c (Gate := Gate) (.inl (u, q)) = reg := by
        show (ownPort own c q.val).getD default = reg
        rw [hreg]; rfl
      refine ⟨u, q, rfl, ?_⟩
      rw [hport]
      refine ⟨Array.mem_of_find?_eq_some hreg, beq_iff_eq.mp (Array.find?_some hreg), ?_⟩
      show reg.reads[(popUnits pub).getD u.val 0]? = some (some ((ownPos own c pub u.val q.val).getD 0))
      rw [hk0]
      exact Option.join_eq_some_iff.mp hk
  · exact hp.elim

/-- **A row a unit holds is its own**: a committed wire of unit `u` at row `(u', q)` of the shadow layout has `u' = u`
(`Layout.keyRowOf_wire`). -/
theorem laySh_pos_unit {u u' : Fin n} {q : Fin c.ports.size}
    {g : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N}
    (hg : g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u)
    (hpos : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).pos g = .inl (u', q)) : u' = u := by
  obtain ⟨x, rfl⟩ := (Const.Prog.instC (keyProg c hU n) (Layout.outsOf c hU n) u).exists_wire_of_mem_io hg
  have e : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).pos
      ((Const.Prog.instC (keyProg c hU n) (Layout.outsOf c hU n) u).wire x) =
      match Layout.keyRowOf c hU n (Layout.portRowW c)
          ((Const.Prog.instC (keyProg c hU n) (Layout.outsOf c hU n) u).wire x) with
      | some (u, ⟨q, _⟩) => Sum.inl (u, q)
      | none => Sum.inr ((Const.Prog.instC (keyProg c hU n) (Layout.outsOf c hU n) u).wire x) := rfl
  rw [e] at hpos
  rcases hk : Layout.keyRowOf c hU n (Layout.portRowW c)
      ((Const.Prog.instC (keyProg c hU n) (Layout.outsOf c hU n) u).wire x) with _ | ⟨u'', q'', k⟩
  · rw [hk] at hpos; simp at hpos
  · rw [hk] at hpos
    simp only [Sum.inl.injEq, Prod.mk.injEq] at hpos
    obtain ⟨rfl, rfl⟩ := hpos
    exact (Layout.keyRowOf_wire _ x hk).1

/-- **The verifier's own reads at an accepted live outcome.** At a live outcome `(R, S, wr S R lo)` that
`verify --zk --registered F` accepts (`own` F's ports), where the coin server records the draw `dj S` (`RecordsLiveZK`)
and that draw proves `S` (`DrawOkZK`'s first conjunct), the reads F gives at the rows the drawn units hold
(`drawnOwn`) are the verifier's own at the record's draw (`OwnReads`). -/
theorem ownAtDraws_of_check (own : Array Registered.Port) (hTags : TagsOk I)
    {wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript} (hRec : RecordsLiveZK I dj wr)
    {S : Finset (Fin n)} (hS : ∀ D, Flock.Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (R : Reg) (lo : LiveOut mPts) (hacc : AcceptsZKR I own (wr S R lo))
    (wv : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
    (ws : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → List UInt8)
    (wpath : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → ByteArray)
    (committed : ∀ p, ownIsRead own c pub p →
      Flock.Registered.opensLeaf (ownPortAt own c p)
        ((Binding.ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc
          (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf (wv p) (ws p)).1 (ownReadAt own c pub p)
        (wpath p) = true) :
    OwnReads own c pub (wr S R lo)
      (drawnOwn ht dj y₀ hc hU hpub hscope own wv ws wpath committed S).isRead
      (drawnOwn ht dj y₀ hc hU hpub hscope own wv ws wpath committed S).port
      (drawnOwn ht dj y₀ hc hU hpub hscope own wv ws wpath committed S).readAt := by
  rintro p ⟨hp, u₀, hu₀, g, hg, hpos⟩
  obtain ⟨u, q, rfl, hmem, hfind, hreads⟩ := ownReads_shape hp
  have hu : u ∈ S := (laySh_pos_unit ht dj y₀ hc hU (pub := pub) hg hpos) ▸ hu₀
  obtain ⟨st, -, hck⟩ := check_of_accepts hTags hacc
  obtain ⟨-, idx, -, hidx, -⟩ := check_tables hck
  obtain ⟨hd, -⟩ := hRec S R lo (acceptsZK_of_R hacc)
  have hidx' : Registered.tableUnits I.publicFile (some (dj S)) I.count 0 = .ok idx := by
    rw [← hd]; exact hidx
  obtain ⟨header, units, D, hh, hun, hD, -⟩ := Registered.tableUnits_one hTags.2.2.2 hidx'
  rw [loadPublic_header hpub] at hh
  cases hh
  have hpop : popUnits pub = units := by
    show ((Registered.unitIndices pub.header).toOption).getD #[] = units
    rw [hun]; rfl
  rw [hpop] at hreads
  exact ⟨u, q, dj S, D, units, rfl, hmem, hfind, hd, hD, hS D hD u hu, hun, hreads⟩

end

variable {mPts : ℕ} {Reg : Type}

/-- **One `--zk --registered` sampled session is sound, and F's reads at the drawn units are checked.**
`zk_session_soundR` at the reads F gives at the rows the drawn units hold (`drawnOwn`), at every live outcome
`(R, L.draw ω, wr (L.draw ω) R lo)` the session accepts, for any registrant's openings that open F's roots at F's
positions: `OwnReads` follows from the session's own hypotheses (`ownAtDraws_of_check`, at `hRec` and `hDraw`). -/
theorem zk_session_soundR_own
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtS ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtS_computes ht dj y₀ hc hU hpub hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) ∧
    ∀ (R : Reg) (ω : Ω) (lo : LiveOut mPts),
      LiveAcceptsZKR I own (R, (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω,
        wr ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R lo) →
      ∀ (wv : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos →
          (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Val)
        (ws : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → List UInt8)
        (wpath : (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).Pos → ByteArray)
        (committed : ∀ p, ownIsRead own c pub p →
          Flock.Registered.opensLeaf (ownPortAt own c p)
            ((Binding.ExplicitLeaf.hm96 (rowsSh ht dj y₀ hc hU hpub hscope).enc
              (rowsSh ht dj y₀ hc hU hpub hscope).enc_inj).leaf (wv p) (ws p)).1 (ownReadAt own c pub p)
            (wpath p) = true),
        (drawnOwn ht dj y₀ hc hU hpub hscope own wv ws wpath committed
            ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)).Checked R
          (regPaths c pub (wr ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) R lo)
            (ownPortAt own c)) := by
  have h := zk_session_soundR I ht own dj y₀ wr dg N hc hU hpub hscope σs ks hks St hS Kd pop Lb src hA3 hDraw hTags hRec
    hk hRw hM hρ hr ht' K₀ τ hCR
  refine ⟨h.1, fun R ω lo hacc wv ws wpath committed => h.2 _ hacc _ ?_⟩
  exact ownAtDraws_of_check ht dj y₀ hc hU hpub hscope own hTags hRec (hDraw ω).1 R lo hacc wv ws wpath committed

end FlockSoundness.Discharge.ZkReg
