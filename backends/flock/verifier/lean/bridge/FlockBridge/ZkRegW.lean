import FlockSoundness.Discharge.ZkReg
import FlockSoundness.Discharge.ScopeW.Keyed
import FlockBridge.RegisteredReadsP

/-!
# Registered reads of a `--zk --registered` session, at the widened scope

`ZkReg`'s forms with `hscope : scopeOkW c = true` in place of `Layout.scopeOk c = true`, so that their circuits include
PoUW's, whose ports are `sha512/row/v2` and `sha512/row-seg/v1` rows. The shadow layout is `ZkReg.laySh`, which reads no
row's shape; its rows are `rowsShW`, `ScopeW.keyedRowsOutsW` at `laySh`'s sites, each row hashed under its own schema;
and the registrant's reads are `ScopeW.ReadsP` over them, read through `RegisteredReadsP`.

* `checked_of_acceptsP`: `ZkReg.checked_of_accepts` for reads over any rows of `laySh`. Its proof is the same: it reads
  the reads' commit strings, ports and positions and the layout's registered leaves, never a row's leaf scheme.
* `sitesSh_bcSiteSW`, `rowsShW`: `ZkReg.sitesSh_bcSiteS` and `ZkReg.rowsSh` at `scopeOkW` (`ScopeW.scopeOkW_spec`).
  `rowsShW`'s leaves take a row's extractor from `ScopeW.pairW_spec`, which C-Flock proves from two `sorry` stubs
  (`ScopeW.rowEncV2_inj`, `ScopeW.segPair_spec`), so every declaration that names `rowsShW` rests on their statements.
* `OwnAtDrawsW` (named), `checked_at_drawsW`, `regReadSound_zkW`: `ZkReg.OwnAtDraws`, the check at every reachable
  accepted outcome, and `ZkReg.regReadSound_zk`, at `rowsShW`.
* `registered_zkW`, `registered_zk_jointW`: `ZkReg.registered_zk_session`'s step into the core game, from the drawn wrong
  units' bound (`hdrawn`) and the registrant's values (`RegisteredReadsP.RegValsSound`), or from both bounded together
  (`RegisteredReadsP.RegJointSound`), each named. At `Layout.scopeOk` they are `ZkReg.zk_session_drawnR` and
  `ZkReg.zk_session_regValsR`; C-Flock has no `--zk` soundness at `scopeOkW` yet (its widened headlines are
  `verify`'s, `LiveAccepts`), so here they stay hypotheses.
-/

namespace FlockBridge

namespace ZkRegW

open scoped ENNReal

open FlockSoundness.Discharge.Exec (Inputs auditLive liveOf)
open FlockSoundness.Discharge.ZkReg (laySh LiveAcceptsZKR OwnReads regPaths)
open FlockSoundness.Discharge.ZkBind (planSh tabSh sitesSh sitesSh_parse tableSh zkTyped)
open FlockSoundness.Discharge.ZkLink (DrawSetupZK)
open FlockSoundness.Discharge.ScopeW (HmRowsP ReadsP scopeOkW scopeOkW_spec)
open FlockSoundness.Discharge.Placed (keyProg)
open FlockSoundness.Discharge.Layout (outsOf)
open RegisteredReads (drawnReads)
open RegisteredReadsP (restrict regVals regGates RegValsSound RegJointSound)

/-- **The registered reads are checked at every accepted outcome, over any rows.** `ZkReg.checked_of_accepts` for the
registrant's reads `rd` over any rows `rs` of the shadow layout (`ReadsP`): at an outcome `verify --zk --registered F`
accepts (`own` F's ports), each registered read that is the verifier's own at the outcome's draw (`OwnReads`) has its
registered leaf open the root `own` holds at the position `own` gives, through `regPaths`. -/
theorem checked_of_acceptsP {mPts n : ℕ} {Reg : Type} (I : Inputs) (ht : I.tags.typed = false)
    (own : Array Flock.Registered.Port) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {c : Flock.Circuit}
    (hc : Flock.HmRow.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
    (hU : FlockSoundness.Built c.unit) {pub : Flock.Public}
    (hpub : Flock.HmRow.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) c I.publicFile = .ok pub)
    (hTags : FlockSoundness.Discharge.DecodeZL.TagsOk I)
    (o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) (hacc : LiveAcceptsZKR I own o)
    {rs : HmRowsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))}
    (rd : ReadsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) rs)
    (hown : OwnReads own c pub o.2.2 rd.isRead rd.port rd.readAt) :
    rd.Checked o.1 (regPaths c pub o.2.2 rd.port) := by
  intro p hp
  obtain ⟨u, q, d, D, dp, idx, rfl, hmem, hfind, hd, hD, hdp, hidx, hu, hread⟩ := hown p hp
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  have hh' : (FlockSoundness.Discharge.ZkExec.zkTags I).hiddenOutputs = false :=
    (FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs I).trans hTags.1
  rcases FlockSoundness.Discharge.Exec.setupTables_ok hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
  swap
  · exact absurd hTags.2.2.2 (by change 1 < I.count at hc'; omega)
  change Flock.Stmt.setupH (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.publicFile
    (FlockSoundness.Discharge.ZkExec.zkCoins I) I.tables (FlockSoundness.Discharge.Exec.recordDraw o.2.2.record)
    I.partition I.program = .ok st at hst
  rw [hd] at hst
  have hcst : st.c = c := by
    have h1 := (FlockSoundness.setupH_spec ((FlockSoundness.Discharge.ZkExec.zkTags_typed I).trans ht) hst).1
    rw [hc] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', hdr⟩ := FlockSoundness.Discharge.Layout.setupH_drawn hst
  rw [hcst, hpub] at hpop'
  cases hpop'
  rw [hcst, hdp] at hdr
  have hpd : st.pub = dp := (Except.ok.inj hdr).symm
  have hck := FlockSoundness.Discharge.ZkReg.checkRegistered_ok
    (FlockSoundness.Discharge.ZkExec.array_forM_ok hreg 0 (by simp))
  simp only [List.getElem_toArray, List.getElem_cons_zero] at hck
  rw [hcst, hpd] at hck
  obtain ⟨tb, idx', htb, hidx', hports⟩ := FlockSoundness.Discharge.ZkReg.check_tables hck
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  rw [hfind] at hp'
  cases hp'
  obtain ⟨D', hD', -, hn, -, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdp
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, hopens⟩ := hall _ (by rw [hn]; exact hi)
  rw [hread] at hreads
  have hra : rd.readAt (.inl (u, q)) =
      positions.getD (Flock.Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hpath : regPaths c pub o.2.2 rd.port (.inl (u, q)) =
      paths.getD (Flock.Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) ByteArray.empty := by
    simp [regPaths, hd, hD, hdp, htb, hent, Except.toOption]
  have hleaf : ((laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)).regLeaf o.1 (.inl (u, q))).1 =
      Flock.Hm96.Default512.treeLeaf (pub.digest u.val q.val c.ports.size) := by
    show (FlockSoundness.Refine.H512 (FlockSoundness.Refine.K512.leafPrefix ++
      (pub.digest u.val q.val c.ports.size).data.toList)).1 = _
    rw [FlockSoundness.Refine.H512_val]
    show Flock.Sha512.hash ⟨(Flock.Hm96.Default512.leafPrefix.data.toList ++
        (pub.digest u.val q.val c.ports.size).data.toList).toArray⟩ =
      Flock.Sha512.hash (Flock.Hm96.Default512.leafPrefix ++ pub.digest u.val q.val c.ports.size)
    rw [← FlockSoundness.Refine.data_append, FlockSoundness.Refine.bytes_of_list]
  have hget : D.proved[D.proved.idxOf u.val]'hi = u.val := by
    have := Array.findIdx_getElem (xs := D.proved) (p := (· == u.val)) (w := hi)
    simp only [beq_iff_eq] at this
    exact this
  have hdig : pub.digest u.val q.val c.ports.size =
      Flock.Registered.tableRow dp tb q.val
        (Flock.Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) := by
    have h1 := FlockSoundness.Discharge.Hidden.drawn_digestW hpub hdp hD
      (FlockSoundness.Discharge.Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Flock.Public.digest, htb, Flock.Registered.tableRow, Flock.Registered.ref]
  show Flock.Registered.opensLeaf (rd.port (.inl (u, q))) _ (rd.readAt (.inl (u, q))) _ = true
  rw [hleaf, hra, hpath, hdig]
  exact hopens

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}
  {c : Flock.Circuit} (hc : Flock.HmRow.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
  (hU : FlockSoundness.Built c.unit) {pub : Flock.Public}
  (hpub : Flock.HmRow.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) c I.publicFile = .ok pub)
  (hscope : scopeOkW c = true)

include hc hU hpub hscope in
/-- **The site fact at every satisfiable site of the shadow plan, at the widened scope**: `ZkReg.sitesSh_bcSiteS` with
`ho` from `scopeOkW` (`ScopeW.scopeOkW_spec`). -/
theorem sitesSh_bcSiteSW :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : FlockSoundness.Binding.Msg (planSh ht dj y₀) (tabSh ht dj y₀) S R u),
      FlockSoundness.Binding.Sat (planSh ht dj y₀) (tabSh ht dj y₀) S R u M →
        FlockSoundness.Discharge.Layout.BcSite ((sitesSh ht dj y₀).acc S R u h) pub ((sitesSh ht dj y₀).o S R u h)
          ((sitesSh ht dj y₀).vu S R u h) u.val := by
  intro S R u h M hM
  show FlockSoundness.Discharge.Layout.BcSite (tableSh ht dj y₀ S).acc pub ((tableSh ht dj y₀ S).o u h)
    ((tableSh ht dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZK I mPts S (dj S))
  · have e : tableSh ht dj y₀ S = (Classical.choice hS).x.table (zkTyped ht) := by
      unfold tableSh; rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact FlockSoundness.Discharge.Integrate.table_bcSiteS (zkTyped ht) dj hc (scopeOkW_spec hU hscope).2.1 hpub _ h
  · have e : tableSh ht dj y₀ S = y₀.x.refused (zkTyped ht) S := by
      unfold tableSh; rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableSh ht dj y₀ S).T.S.Satisfies z := by
      rw [e]; exact y₀.x.refused_unsat (zkTyped ht) S
    exact absurd hM (hns _)

/-- **The shadow plan's rows at the widened scope, each under its own schema**: `ScopeW.keyedRowsOutsW` at `laySh`'s
sites, its circuit conditions from `scopeOkW` (`ScopeW.scopeOkW_spec`), with `hsite` from `sitesSh_bcSiteSW`: `ZkReg.rowsSh`
at `scopeOkW`. -/
noncomputable abbrev rowsShW : HmRowsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) :=
  FlockSoundness.Discharge.ScopeW.keyedRowsOutsW (sitesSh_parse ht dj y₀ hc) FlockSoundness.Discharge.Hm.saltCol hc
    (scopeOkW_spec hU hscope).1 (scopeOkW_spec hU hscope).2.2 (scopeOkW_spec hU hscope).2.1
    (sitesSh_bcSiteSW ht dj y₀ hc hU hpub hscope)

/-- **The registrant's reads are the verifier's own at each accepted live outcome's draw** (named hypothesis):
`ZkReg.OwnAtDraws` for the registrant's reads over `rowsShW`. -/
def OwnAtDrawsW (own : Array Flock.Registered.Port) (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → FlockSoundness.Refine.LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (rd : ReadsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsShW ht dj y₀ hc hU hpub hscope)) : Prop :=
  ∀ (R : Reg) (ω : L.Ω) (lo : FlockSoundness.Refine.LiveOut mPts),
    LiveAcceptsZKR I own (R, L.draw ω, wr (L.draw ω) R lo) →
    OwnReads own c pub (wr (L.draw ω) R lo)
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (L.draw ω))).isRead rd.port rd.readAt

/-- **The registered check at every accepted outcome the live audit can reach**: at `(R, L.draw ω, wr (L.draw ω) R lo)`
(`RegisteredReads.out_auditLive`), `checked_of_acceptsP` at the registrant's reads where the draw's units read, which are
the verifier's own there (`OwnAtDrawsW`). -/
theorem checked_at_drawsW (own : Array Flock.Registered.Port) (hTags : FlockSoundness.Discharge.DecodeZL.TagsOk I)
    (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → FlockSoundness.Refine.LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (rd : ReadsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsShW ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDrawsW ht dj y₀ hc hU hpub hscope own L wr rd) :
    ∀ o : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript,
      FlockSoundness.Game.Out (auditLive L Reg (liveOf dg mPts N wr)) o → LiveAcceptsZKR I own o →
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).Checked o.1
        (regPaths c pub o.2.2 rd.port) := by
  intro o hout hacc
  obtain ⟨R, ω, lo, rfl⟩ := RegisteredReads.out_auditLive hout
  exact checked_of_acceptsP I ht own dj y₀ hc hU hpub hTags _ hacc
    (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (L.draw ω))) (hown R ω lo hacc)

/-- **Registered reads of a `--zk --registered` session are sound, at the widened scope.** `ZkReg.regReadSound_zk` at
`rowsShW`: in the live audit under any law `L` and live session `wr`, played by `σ`, if the registrant's reads `rd` are
the verifier's own at each accepted outcome's draw (`OwnAtDrawsW`), the registered-path finder meets `cr/sha-512` at
`qR`, and the registrant's values hold for `Xc` up to `δ` (`RegisteredReadsP.RegValsSound`), then registered reads are
sound for `Xc` against `regVals rd` on `regGates rd`, up to `qR²/2^513 + δ`. -/
theorem regReadSound_zkW (own : Array Flock.Registered.Port) (hTags : FlockSoundness.Discharge.DecodeZL.TagsOk I)
    (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → FlockSoundness.Refine.LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : FlockSoundness.Game.Strategy (auditLive L Reg (liveOf dg mPts N wr)))
    (rd : ReadsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsShW ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDrawsW ht dj y₀ hc hU hpub hscope own L wr rd)
    (Xc : Fin ((keyProg c hU n).circuit (outsOf c hU n)).N → Bool) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun (auditLive L Reg (liveOf dg mPts N wr)) σ fun o =>
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hvals : RegValsSound (auditLive L Reg (liveOf dg mPts N wr)) σ (LiveAcceptsZKR I own)
      (fun o => o.2.1) (fun o => o.1) rd Xc δ) :
    Registered.RegReadSound ((keyProg c hU n).partition (outsOf c hU n)) (auditLive L Reg (liveOf dg mPts N wr)) σ
      (LiveAcceptsZKR I own) (fun o => o.2.1) Xc
      (regVals rd) (regGates rd) (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) :=
  RegisteredReadsP.regReadSound_of_checked (lay := laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
    (rs := rowsShW ht dj y₀ hc hU hpub hscope) (auditLive L Reg (liveOf dg mPts N wr)) σ (LiveAcceptsZKR I own)
    (fun o => o.2.1) (fun o => o.1) (fun o => regPaths c pub o.2.2 rd.port) rd Xc
    (checked_at_drawsW ht dj y₀ hc hU hpub hscope own hTags L wr dg N rd hown) hR hvals

/-- **Soundness against registered values of a `--zk --registered` session in the core game, at the widened scope.**
`ZkReg.registered_zk_session`'s step at `rowsShW`, its two session bounds named: for any strategy `σ` of the live
audit's core game, at its C-Flock strategy, if an accepting run draws a wrong unit of `Xc` with probability at most `ε`
(`hdrawn`) and `regReadSound_zkW`'s assumptions hold, some core transcript makes an accepting run draw a wrong unit or
be off the registered values `regVals rd` on the registered gates `regGates rd` with probability at most
`ε + (qR²/2^513 + δ)`. `Registered.registered_core` at `regReadSound_zkW`. -/
theorem registered_zkW (own : Array Flock.Registered.Port) (hTags : FlockSoundness.Discharge.DecodeZL.TagsOk I)
    (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → FlockSoundness.Refine.LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Verity.Game.Strategy (Game.toCore (auditLive L Reg (liveOf dg mPts N wr))))
    (rd : ReadsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsShW ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDrawsW ht dj y₀ hc hU hpub hscope own L wr rd)
    (Xc : Fin ((keyProg c hU n).circuit (outsOf c hU n)).N → Bool) {ε : ℝ≥0∞}
    (hdrawn : FlockSoundness.Game.prob (fun o => LiveAcceptsZKR I own o ∧
        ¬ Disjoint (((keyProg c hU n).partition (outsOf c hU n)).wrong Xc) o.2.1)
      (auditLive L Reg (liveOf dg mPts N wr)) (Game.Strategy.ofCore _ σ) ≤ ε) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun (auditLive L Reg (liveOf dg mPts N wr)) (Game.Strategy.ofCore _ σ) fun o =>
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hvals : RegValsSound (auditLive L Reg (liveOf dg mPts N wr)) (Game.Strategy.ofCore _ σ)
      (LiveAcceptsZKR I own) (fun o => o.2.1) (fun o => o.1) rd Xc δ) :
    ∃ X' : (Circuit.toCore ((keyProg c hU n).circuit (outsOf c hU n))).Transcript,
      Verity.Game.prob (fun o => LiveAcceptsZKR I own o ∧
          (¬ Disjoint ((Partition.toCore ((keyProg c hU n).partition (outsOf c hU n))).wrong X') o.2.1 ∨
            ∃ g ∈ regGates rd, X' g ≠ Circuit.Transcript.toCore (regVals rd) g))
        (Game.toCore (auditLive L Reg (liveOf dg mPts N wr))) σ ≤
      ε + (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) := by
  have hv := regReadSound_zkW ht dj y₀ hc hU hpub hscope own hTags L wr dg N (Game.Strategy.ofCore _ σ) rd hown Xc
    hR hvals
  rw [← Circuit.Transcript.ofCore_toCore (regVals rd)] at hv
  refine Registered.registered_core _ _ _ _ _ _ σ _ False ?_ hv
  exact le_trans (FlockSoundness.Game.prob_mono (fun o h => ⟨h.1, h.2.resolve_right id⟩) _ _) hdrawn

/-- **Soundness against registered values of a `--zk --registered` session in the core game, at the widened scope, from
one joint bound.** `RegisteredReadsP.registered_of_checked_joint` at `rowsShW`, its check from `checked_at_drawsW`: if
the registrant's reads are the verifier's own at each accepted outcome's draw (`OwnAtDrawsW`), the registered-path finder
meets `cr/sha-512` at `qR`, and the drawn wrong units of `Xc` and the registrant's values are bounded together by `δ`
(`RegisteredReadsP.RegJointSound`), some core transcript makes an accepting run draw a wrong unit or be off `regVals rd`
on `regGates rd` with probability at most `qR²/2^513 + δ`. -/
theorem registered_zk_jointW (own : Array Flock.Registered.Port) (hTags : FlockSoundness.Discharge.DecodeZL.TagsOk I)
    (L : FlockSoundness.Audit.Law n)
    (wr : Finset (Fin n) → Reg → FlockSoundness.Refine.LiveOut mPts → FlockSoundness.Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (σ : Verity.Game.Strategy (Game.toCore (auditLive L Reg (liveOf dg mPts N wr))))
    (rd : ReadsP (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) (rowsShW ht dj y₀ hc hU hpub hscope))
    (hown : OwnAtDrawsW ht dj y₀ hc hU hpub hscope own L wr rd)
    (Xc : Fin ((keyProg c hU n).circuit (outsOf c hU n)).N → Bool) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun (auditLive L Reg (liveOf dg mPts N wr)) (Game.Strategy.ofCore _ σ) fun o =>
      (restrict rd (drawnReads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hjoint : RegJointSound (auditLive L Reg (liveOf dg mPts N wr)) (Game.Strategy.ofCore _ σ)
      (LiveAcceptsZKR I own) (fun o => o.2.1) (fun o => o.1) rd Xc δ) :
    ∃ X' : (Circuit.toCore ((keyProg c hU n).circuit (outsOf c hU n))).Transcript,
      Verity.Game.prob (fun o => LiveAcceptsZKR I own o ∧
          (¬ Disjoint ((Partition.toCore ((keyProg c hU n).partition (outsOf c hU n))).wrong X') o.2.1 ∨
            ∃ g ∈ regGates rd, X' g ≠ Circuit.Transcript.toCore (regVals rd) g))
        (Game.toCore (auditLive L Reg (liveOf dg mPts N wr))) σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ :=
  RegisteredReadsP.registered_of_checked_joint (lay := laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
    (rs := rowsShW ht dj y₀ hc hU hpub hscope) (auditLive L Reg (liveOf dg mPts N wr)) σ (LiveAcceptsZKR I own)
    (fun o => o.2.1) (fun o => o.1) (fun o => regPaths c pub o.2.2 rd.port) rd Xc
    (checked_at_drawsW ht dj y₀ hc hU hpub hscope own hTags L wr dg N rd hown) hR hjoint

end ZkRegW

end FlockBridge
