import Proofs.Flock.Claims.Recursion.VerdictRowEOpens
import Proofs.Flock.Claims.Recursion.RegSetupOpens

/-!
# Two staged parts' `rec-acc` slots at one chain position carry one commit string

Row E's cross-part tie (`held_rows_eq_layR`, `out_rows_eq_layR`) takes two parts' hm96 slots to carry one commit string
(`hcm`). At two staged `.recOpen` parts, each slot is the table row its setup opens against the part's registered
`rec-acc` root at the slot's chain position (`accRow_table`), and the two ports' roots are one (`accRow_site_eq`,
`sameRoot_of_site`). So both rows are openings the setups list (`mem_regOpenings`, `mem_portOpenings`), at one
position under one tree, and `bc_eq_of_setup` gives one commit string wherever the true calls and the setups'
evaluations have no collision (`accCommit_eq`). The labels in order (`hidx`) are `labels_run`'s at an accepted run.
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **Two staged rec-open parts' `acc`/`acc_out` slots at one chain position carry one commit string**: at staged
`.recOpen` parts `s` and `s'` of `x` at one staging message (`hxO`, `hxO'`) the witnesses check against, with their tables labelled in order and the reads
files `own`, `own'`, the slots at units `n` and `n'` whose chain positions agree (`hpos`) read one commit string,
wherever the true calls `L` and the setups' evaluations at `xs ∋ xO, xO'` have no collision, at a record that lists
the registered reads' evaluations (`RegListed`). -/
theorem accCommit_eq {st : Stage} {s s' : ℕ} {x xO xO' : Msg} {w w' : Claims.Flock.Wit}
    (hS : StagedPart st s x xO) (hS' : StagedPart st s' x xO')
    {ms : List (Msg × Claims.Out)} (hxO : xO = staged st x ms s) (hxO' : xO' = staged st x ms s')
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    (hck' : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO' w' = true) {r : XR}
    (hr : readX x = some r) {c c' : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some c)
    (hpc' : r.w.parts[s']?.bind partCircuit = some c') {l LANES HH m l' LANES' HH' m' : ℕ}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen l LANES HH m))
    (hsv' : (RecVStar.statements r.shape).toOption.bind (·[s']?) = some (.recOpen l' LANES' HH' m'))
    {sch : Schedule} (hsch : fast100 r.shape.m = some sch) {i i' : RecIns} {spec spec' : SessionSpec}
    {sts sts' : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (hsu' : setupO xO' = some (i', spec', sts'))
    (h0 : 0 < sts.size) (h0' : 0 < sts'.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    (hidx' : Registered.unitIndices sts'[0].pub = .ok (Array.range sts'[0].pub.n))
    {own own' : Array Registered.Port} (hown : i.base.registered = some own)
    (hown' : i'.base.registered = some own') {nm nm' : String} {d d' : ℕ}
    (hnm : (nm, d) ∈ [("acc", 0), ("acc_out", 1)]) (hnm' : (nm', d') ∈ [("acc", 0), ("acc_out", 1)]) {q q' : ℕ}
    (hq : c.ports.findIdx? (·.name == nm) = some q) (hq' : c'.ports.findIdx? (·.name == nm') = some q') {n n' : ℕ}
    (hn : n < 2 * (sch.levels[l]!).q) (hnp : n < sts[0].pub.n) (hn' : n' < 2 * (sch.levels[l']!).q)
    (hnp' : n' < sts'[0].pub.n)
    (hpos : n / (sch.levels[l]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l +
        n % (sch.levels[l]!).q + d =
      n' / (sch.levels[l']!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l' +
        n' % (sch.levels[l']!).q + d')
    {Co : Type} {D : Claims.Recursion.Deploy Co} (hD : RegListed D) {L : List (Msg × Msg)}
    (hl : ∀ e ∈ L, FlockDeploy.sha512 e.1 = e.2) {xs : List Msg}
    (hc : (Claims.findColl (L ++ Claims.Recursion.setupCalls FlockDeploy.sha512 D xs)).isSome = false)
    (hx : xO ∈ xs) (hx' : xO' ∈ xs) :
    Claims.Recursion.VStarB.readForms w (hmCommit c q n) =
      Claims.Recursion.VStarB.readForms w' (hmCommit c' q' n') := by
  obtain ⟨reg, rws, a0, hreg, -, hrw, ha, hs, t, positions, paths, ht, hp, he, -, hps, -, hrd⟩ :=
    accRow_table hS hck hr hpc hsv hsch hsu h0 hidx hown hnm hq hn hnp
  obtain ⟨reg', rws', a0', hreg', -, hrw', ha', hs', t', positions', paths', ht', hp', he', -, hps', -, hrd'⟩ :=
    accRow_table hS' hck' hr hpc' hsv' hsch hsu' h0' hidx' hown' hnm' hq' hn' hnp'
  have ho := mem_regOpenings hsu hown (st := sts[0]) (by simp) (by simpa using hreg) (mem_portOpenings ht hp he hnp)
  have ho' := mem_regOpenings hsu' hown' (st := sts'[0]) (by simp) (by simpa using hreg')
    (mem_portOpenings ht' hp' he' hnp')
  have hpp : positions.getD (Registered.ref t sts[0].c.ports.size q n) 0 =
      positions'.getD (Registered.ref t' sts'[0].c.ports.size q' n') 0 := by
    rw [hps, hps', hpos]
  rw [hpp] at ho
  subst hxO hxO'
  have hsite := accRow_site_eq (setupO_some hsu).1 (setupO_some hsu').1 hrw ha hs hrw' ha' hs'
  have hbc := bc_eq_of_setup hD hl hc hx hx' ho ho' (sameRoot_of_site hsite)
  rw [hrd, hrd', hbc]

end RecDeploy
