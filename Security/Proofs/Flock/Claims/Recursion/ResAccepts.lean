import Proofs.Flock.Claims.Recursion.PartAccepts
import Proofs.Flock.Claims.Recursion.VStarPartsB
import Proofs.Flock.Claims.Recursion.VStarLinks

/-!
# The circuit at a staged part, from `PartAccepts`

`decompB_lastSplitP`'s lowering fact `hlow` starts from `PartAccepts` at the deployed outer record
(`deployZkRec relation`) and the staging `showsPart`: some V\* statement `xO`, shown on the coins and staged, has an
outer witness `w` passing `Flock.check`, its recorded calls true and none colliding, and agreeing with the messages. This
file reads the circuit's semantics off that, with no premise beyond `PartAccepts`:

* `partAccepts_staged`: the staged statement parses untyped, without public input words, to tables each of part `s`'s
  V\* statement `sv` at `x`'s inner shape, and every copy of every table holds its unit's outputs at what its input
  columns read (`StmtCopies`): `stagedReads` for the staging, `allOuts_of_check` for the outputs.
* `partAccepts_inner`: at a part whose V\* statement is an algebra part (`.inner`), every copy of every table computes
  the verifier's part's residuals (`ResOuts`: `PartCols` at the elements the copy reads, `laid_part`'s conclusion), the
  part being the one `RecResiduals.check` derives from the unit's name and META (`innerCopies`).

What is left of a residuals piece's `hlow` is not the circuit's: the input tie (the elements the copy reads are the held
messages' rows and the coins' coefficients) and the output tie (the residuals there give the piece).
-/

namespace RecDeploy

open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB

/-- **What part `s`'s circuit accepting gives at its staged statement**: at the deployed outer record and the staging
`showsPart`, `PartAccepts` names a staged V\* statement `xO` and an outer witness `w` agreeing with `ms`, whose recorded
calls are true and don't collide, at which `xO` parses untyped, without public input words, to at least one table, each
a table of part `s`'s V\* statement `sv` at `x`'s inner shape, and every copy of every table holds its unit's outputs at
what its input columns read (`StmtCopies`). -/
theorem partAccepts_staged {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms) :
    ∃ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed) (r : XR) (sv : VStar.Stmt),
      _root_.Flock.Readable.parseBytes readOfO xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      readX x = some r ∧ (∃ ss, _root_.Flock.RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some sv) ∧
      0 < p.sts.size ∧ _root_.Flock.Readable.StmtCopies sv p w ∧ L.Agrees cmt x s w xO ms ∧
      (∀ e ∈ L.calls x s w, H e.1 = e.2) ∧ (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false ∧
      showsTr untr s x xO c := by
  obtain ⟨-, xO, w, hck, hT, hc, hs, hh, hp, hm, hrw⟩ := h
  obtain ⟨p, r, sv, hx, ht, hpi, hr, hss, hsz, htab⟩ := stagedReads hs
  have hck' : Claims.Flock.check (_root_.Flock.Readable.parseOf _root_.Flock.Readable.relation readOfO) xO w =
      true := hck
  exact ⟨xO, w, p, r, sv, hx, ht, hpi, hr, hss, hsz,
    _root_.Flock.Readable.stmtCopies_of_allOuts htab (_root_.Flock.Readable.allOuts_of_check hck' hx ht hpi),
    ⟨hh, hp, hm, hrw⟩, hT, hc, hs.1⟩

/-- **An algebra part's copies compute its residuals, from `PartAccepts`**: at the deployed outer record and the staging
`showsPart`, where part `s`'s V\* statement at `x`'s inner shape is an algebra part (`.inner`), `PartAccepts` names a
staged V\* statement and an outer witness `w` agreeing with `ms`, whose recorded calls are true and don't collide, at
which every copy (block `o`, VU `g`, copy `u`) of every table `t` holds, on its output columns, the residuals of the
verifier's part (`RecResiduals.part` at the shape, holo flag and part the unit's name and META give) at the elements its
input columns read (`ResOuts`, which is `laid_part`'s `PartCols` at those elements). -/
theorem partAccepts_inner {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {L : LayoutB}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {ms : List Msg}
    (h : PartAccepts H D (deployZkRec _root_.Flock.Readable.relation) L shape (showsPart untr b) cmt s x c ms)
    {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt} (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    {j : ℕ} {nm : String} {pt : VStar.Part} (hsv : ss[s]? = some (.inner j nm pt)) :
    ∃ (xO : Msg) (w : Claims.Flock.Wit) (p : _root_.Flock.Readable.Parsed),
      _root_.Flock.Readable.parseBytes readOfO xO = some p ∧ 0 < p.sts.size ∧ L.Agrees cmt x s w xO ms ∧
      (∀ e ∈ L.calls x s w, H e.1 = e.2) ∧ (Claims.findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false ∧
      showsTr untr s x xO c ∧
      ∀ t (ht : t < p.sts.size) (o g u : ℕ), o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g →
        u < p.sts[t].c.upv →
        ∃ (m k cl jj : ℕ) (sh : _root_.Flock.Shape) (holo : Bool) (S : _root_.Flock.RecOpen.GfS) (ns : List ℕ)
          (nv : ℕ),
          _root_.Flock.RecResiduals.params p.sts[t].c.unit.name = some (m, k, cl, jj) ∧
          _root_.Flock.RecResiduals.shape? m k cl = some sh ∧
          _root_.Flock.RecResiduals.holoOf p.sts[t].c = .ok holo ∧
          _root_.Flock.RecResiduals.part { sh with holo } jj = .ok (S, ns, nv) ∧
          _root_.Flock.Readable.ResOuts S ns nv (_root_.Flock.Readable.copyOf p t ht w o g u) := by
  obtain ⟨xO, w, p, r', sv, hx, -, -, hr', ⟨ss', hss', hsv'⟩, hsz, hcp, hA, hT, hc, hs⟩ := partAccepts_staged h
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  exact ⟨xO, w, p, hx, hsz, hA, hT, hc, hs, fun t ht o g u ho hg hu =>
    _root_.Flock.Readable.innerCopies ht ho hg hu hcp⟩

end RecDeploy
