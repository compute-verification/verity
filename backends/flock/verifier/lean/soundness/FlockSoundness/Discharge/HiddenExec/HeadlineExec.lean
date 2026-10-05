import FlockSoundness.Discharge.HiddenExec.KeyH
import FlockSoundness.Discharge.HiddenExec.Headline
import FlockSoundness.Discharge.HiddenExec.Sites
import FlockSoundness.Discharge.Layout.Scope
import FlockSoundness.Protocol.Discharge.HiddenExec.HeadlineExec
import FlockSoundness.Discharge.Hidden.Wide.Shared
import FlockSoundness.Discharge.HiddenExec.Ret

set_option Elab.async false

/-!
# The headline at the executable over hidden-output sessions, with every open guarantee named

#828's `flock_headline_exec` (`Integrate/Headline.lean`) for the statement main's prover writes since #757
(`hiddenOutputs = true`, `Stmt.setupHidden`). `flock_headline_exec_hidden` instantiates `flock_headline_hidden` at the
production program `Placed.keyProg c hU n`, its returned outputs' bits for the program's outputs (`Layout.outsOf`), and
the one-table plan of the session's inputs (`planAtH`), with:
* `pp := ppH`, `hConst := hConstH`, `hZero := hZeroH` (`Sites.lean`);
* `lay := keyLayoutH` (`layAtH`) and `rs := keyRowsH` (`rsAtH`) at row widths `16 · words` (`KeyH.lean`): every row of
  the circuit, input or output, is a commit string whose `b ‖ c` the public file registers, and a unit's computed
  committed gate is a bit of an output row; `hsite` from `sitesAtH_bcSite`;
* `hHm := keyRowsH_computes` (`rsAtH_computes`).

So the hypotheses left are those of #828's headline: `hscope`, `hdec`, A3, `cr/sha-512`, `ecr/sha-512` and the numeric
conditions. flock-verify refuses a statement failing `hscope` or the statement's other form hypotheses (PROTOCOL.md
§16.15), and `FailClosed.flock_headline_exec_hidden` takes none of them.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOk c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) {Reg : Type}

/-- **The headline at the executable over hidden-output sessions.** Let `I` be a verify command's inputs (one table)
under a tag with hidden outputs, whose one circuit file parses to an untyped circuit `c` (`hc`: `HmOut.parse`, which
checks each output row holds its port's words, `ht`) and whose public file loads to `pub` with no shared-row tables
(`hpub`: `HmOut.loadPublic`, every row's `b ‖ c` and no output word; `hpt`), and let the coin server write draw `S` as
the JSON `dj S` (`x₀` one draw it sets up with `Stmt.setupHidden`). Let the verifier draw the units with
`Flock.Draw.drawOS`'s stratified law on a byte source A3 makes uniform, and let each draw's session be the live coin
server's one table (`liveTable`, A6), written out by `wr`. Against any live prover `τ`, flock-verify accepts with at
least `K` wrong units of the production program `Placed.keyProg c hU n`, at the simulated prover's committed values (1
on the constant's gate, 0 on the zero's, the plurality of its openings elsewhere), with probability at most
`miss(K) + ε_ks + δ_link`.

Its hypotheses, past the session's inputs, are #828's `flock_headline_exec`'s at the hidden tables:
* the circuit is in `rs`'s scope, `hscope : Layout.scopeOk c = true` (as #828's);
* `hdec` (`DecodesOn`, R11b, at the law's draws only), open, at the hidden tables `tablesAtH`: at a draw flock-verify
  accepts under hidden outputs whose record carries the coin server's draw `dj S`, proving `S`, with link points
  `mPts`, the table is `stmtOf` of the `Stmt.setupHidden` statement it accepted (`tableAtH_of_accepts`);
* the assumptions of record: A3 (`hA3`, `uniform/io-getrandombytes`), `cr/sha-512` for the knowledge finders (`hKS`),
  `ecr/sha-512` for the link finders (`hCR`); A6 (`uniform/flock-coin-server`) is the live session `liveTable` itself;
* numeric conditions and the strata's match (`hks`, `hS`, `hk`, `hRw`, `hM`, `hρ`, `hr`, `ht'`).

What differs from #828's headline: the statement is `Stmt.setupHidden`'s, whose public file registers each unit's
output rows' `b ‖ c` in place of its output words, so a unit's computed committed gates are bits of output rows bound
through their `b ‖ c` (`rsAtH`), not single gates pinned by region "out"; the units' committed values and the bound are
the same expressions at the hidden tables. Scope otherwise as #828's: W6 is definitional on the untyped path, `keyProg`
has no edges between units, the proof coins are the live coin server's OS coins (A6), and sessions of more than one
table are outside it. -/
theorem flock_headline_exec_hidden
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planAtH (Reg := Reg) dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (K : ℕ) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (hCR : LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512
      (planAtH dj x₀) (tabAtH dj x₀) ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
        (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt))
      (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
      (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) k Rw M t')
    {qF qS : ℝ} (hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planAtH dj x₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))) j
      (batchStratL H512 enc512 leaf512 (planAtH dj x₀)
        (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ) ω)) qF qS) :
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planAtH dj x₀)
              (tabAtH dj x₀) ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
                (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt))
              (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              (cont (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ))
              k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (planAtH dj x₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditOn H512 enc512 leaf512 (tablesAtH dj x₀) I wr opn dg dec one fin capOf N hdec τ)) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (planAtH dj x₀) (tabAtH dj x₀)
          ((rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).binding
            (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt)) k Rw M ρ t' :=
  flock_headline_hidden dj x₀ (Placed.keyProg c hU n) (Layout.outsOf c hU n) σs ks hks St hS Kd pop Lb src hA3
    (ppH ht hc (v2_of_scope hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.oneG} (hConstH ht hc (v2_of_scope hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.zeroG} (hZeroH ht hc (v2_of_scope hscope) dj x₀ hU (Layout.outsOf c hU n))
    (layAtH ht hc hscope dj x₀ hU (pub := pub)) (rsAtH ht hc hscope dj x₀ hU hpub hpt)
    (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt) hk hRw hM hρ hr ht' K wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.HiddenExec
