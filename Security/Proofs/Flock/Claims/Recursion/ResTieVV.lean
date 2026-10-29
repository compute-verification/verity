import Proofs.Flock.Claims.Recursion.VVTieOf

/-!
# Check 5's tie at a piece's copy, at RECR's verifier values

`resTie_resVRead`: `resTie_site` at `vvAt := Rows.resVRead`, its coin tie `VVTie` discharged by `vvTie_dense` at a dense
compiled statement whose coins run to the deployed plain verifier's inner rounds.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-- **`ResTie` at a piece's copy, at `Rows.resVRead`** (the `ResTie` conjunct `resPieces_computesAt` asks of a piece at
key `(jj, o, g, 0)`, instance `o + g`): `resTie_site`, with `VVTie` from `vvTie_dense`. -/
theorem resTie_resVRead {b : Brings} {IV : Claims.Verifier Coin}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} {cs : ℕ → Coin} {Bk : Msg → ℕ} {st0 : Claims.Recursion.CompiledStatement}
    (hst : RecDeploy.parse x = some st0) (hd : Claims.Recursion.Form.Dense st0.form)
    (hB : RestRead.kI x ≤ Bk (xIOf x))
    (hT : ∀ e ∈ (DeployedR.layR elem).calls x s w, FlockDeploy.sha512 e.1 = e.2)
    (hnc : (Claims.findColl ((DeployedR.layR elem).calls x s w ++ vcallsAt FlockDeploy.sha512
      (recBound _root_.Flock.Readable.relation b (DeployedR.layR elem) IV) xO)).isSome = false)
    (hsh : showsPart recRead b s x xO cs)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm' : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm' pt)) {c : Circuit}
    (hpc : r.w.parts[s]?.bind partCircuit = some c) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (ht : 0 < p.sts.size)
    {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size)
    (hidx : Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n))
    {m k cl jj : ℕ} (hres : RecResiduals.params sts[0].c.unit.name = some (m, k, cl, jj))
    (hset : ∀ u ∈ resSetup xO, u ∈ b.setup xO)
    {sh : _root_.Flock.Shape} {holo : Bool} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hshp : RecResiduals.shape? m k cl = some sh) (hholo : RecResiduals.holoOf sts[0].c = .ok holo)
    (hpart : RecResiduals.part { sh with holo } jj = .ok (S, ns, nv))
    {o g n : ℕ} (hG : 0 < sts[0].c.g) (hn : o + g = n) (hn2 : n < 2) (hnp : n < sts[0].pub.n)
    {qv qo : ℕ} (hqv : c.ports.findIdx? (·.name == "v") = some qv)
    (hqo : c.ports.findIdx? (·.name == "out") = some qo) (hvin : qv < HmOut.nIn c) (hoout : ¬ qo < HmOut.nIn c)
    {ptv pto : Port} (hptv : c.ports[qv]? = some ptv) (hpto : c.ports[qo]? = some pto)
    (hrows : ∀ (f : RecCoins.Source), RecCoins.Source.atC3 i.src i.inner i.commits = some f →
      ∀ (inn : RecCoins.Inner) (rows : Array (Array ℕ)), f.load sts[0].c.unit.name = .ok inn →
      FlockSoundness.Discharge.Recursion.residualRows inn jj = some rows →
      ptv.words = (RecCoins.padTo rows[n]! (Registered.rowWords rows[n]!.size 16)).size ∧ n < rows.size ∧
        8 * nv ≤ rows[n]!.size)
    (hwo : pto.words = (RecCoins.padTo (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!
      (Registered.rowWords (RecCoins.zeroRows 2 RecCoins.OUT_WORDS)[n]!.size 16)).size) :
    ResTie S ns nv (Rows.resVRead (xIOf x) ((List.range (Bk (xIOf x))).map cs) (jj, o, g, 0))
      (keyCopy p ht w (jj, o, g, 0)) :=
  resTie_site hT hnc hsh hck hr hss hsv hpc hx ht hsu h0 hidx hres hset (vvTie_dense hst hd hB) hshp hholo hpart
    hG hn hn2 hnp hqv hqo hvin hoout hptv hpto hrows hwo

end RecDeploy.ResLower
