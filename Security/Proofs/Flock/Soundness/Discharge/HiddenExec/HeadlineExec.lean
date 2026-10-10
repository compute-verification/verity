import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Soundness.Discharge.HiddenExec.Headline
import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites
import Proofs.Flock.Soundness.Discharge.Layout.Scope

set_option Elab.async false

/-!
# The headline at the executable over hidden-output sessions, with every open guarantee named

#828's `flock_headline_exec` (`Integrate/Headline.lean`) for the statement main's prover writes since #757
(`hiddenOutputs = true`, `Stmt.setupHidden`). `flock_headline_exec_hidden` instantiates `flock_headline_hidden` at the
production program `Placed.keyProg c hU n`, its returned outputs' bits for the program's outputs (`Layout.outsOf`), and
the one-table plan of the session's inputs (`planAtH`), with:
* `pp := ppH`, `hConst := hConstH`, `hZero := hZeroH` (`Sites.lean`);
* `lay := keyLayoutH` (`layAtH`) and `rs := keyRowsH` (`rsAtH`) at row widths `16 · words` (`KeyH.lean`), each row under
  its own prefix (`Hidden.rowPre`): every row of
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

/-- `rs`'s scope gives `sitesAtH`'s rows: on a circuit an untyped statement's `HmOut.parse` accepts, `scopeOkH`'s rows
are each fit or `sha512/row-seg/v1`, the output rows fit (`rowsFit_of_scope`). -/
theorem rowsFit_of_scopeH {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (ht : tags.typed = false) (hc : HmOut.parse tags cf tables = .ok c) (hscope : Layout.scopeOkH c = true) :
    Hidden.Any.RowsFitS c :=
  rowsFit_of_scope (parse_untyped ht hc).1 (parse_untyped ht hc).2 hscope

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) (hU : Built c.unit)
  {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none) {Reg : Type}

include hU hpub in
/-- **The site fact at a drawn table under hidden outputs**: at a draw flock-verify sets up with `Stmt.setupHidden`,
each drawn unit's site is `BcSiteH`: the acceptance's draw is `dj S`, which decodes to `D`, its regions are the
executable's, its public file loads (`HmOut.loadPublic`) to `pub`, with or without shared rows, and the unit's block
and template instance index its rank in `D.proved`, the template instance below `2^⌊log₂ c.g⌋` (`Layout.vu_lt_pow2`). -/
theorem tableH_bcSiteS {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    rw [hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact Layout.vu_lt_pow2 (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).2.1 (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

set_option linter.unusedSectionVars false in
include hU hpub hpt in
/-- `tableH_bcSiteS` at a public file without shared rows. -/
theorem tableH_bcSite {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.acc ht hc (rowsFit_of_scopeH ht hc hscope)) pub (x.table.o u h) (x.table.vu u h) u.val :=
  tableH_bcSiteS ht hc hscope dj hU hpub x h

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable site** of the one-table hidden plan: at a draw flock-verify sets up by
`tableH_bcSite`; at a refused draw no message satisfies the table (`tableAtH_unsat`). -/
theorem sitesAtH_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planAtH dj x₀) (tabAtH dj x₀) S R u), Binding.Sat (planAtH dj x₀) (tabAtH dj x₀) S R u M →
        BcSiteH ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).acc S R u h) pub
          ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).o S R u h)
          ((sitesAtH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableAtH dj x₀ S).acc ht hc (rowsFit_of_scopeH ht hc hscope)) pub ((tableAtH dj x₀ S).o u h)
    ((tableAtH dj x₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupH I mPts S (dj S))
  · have e : tableAtH dj x₀ S = (Classical.choice hS).table := by
      unfold tableAtH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSite ht hc hscope dj hU hpub hpt _ h
  · exact absurd hM (tableAtH_unsat dj x₀ hS _)

/-- **`lay` at the hidden tables**: `keyLayoutH` at `sitesAtH`, the program's outputs the returned outputs' bits, the
registration the population's public file. -/
noncomputable abbrev layAtH :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtH (Reg := Reg) dj x₀)
      (tabAtH dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planAtH dj x₀) (tabAtH dj x₀)
        (keyPlaceH (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the hidden tables**: `keyRowsH` at `sitesAtH`, its circuit conditions from `scopeOkH`
(`Layout.scopeOkH_spec`, `rowsFit_of_scopeH`, `Layout.keysRow_of_parse`), with `hsite` from `sitesAtH_bcSite` at a
public file without shared-row tables (`hpt`). -/
noncomputable abbrev rsAtH : Binding.HmRowsX (layAtH ht hc hscope dj x₀ hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtH_bcSite ht hc hscope dj x₀ hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`hHm` at the hidden tables** (`keyRowsH_computes`): in a satisfying witness of a drawn unit's table, each row's
`b ‖ c` at the unit's site, input or output, is the commit string of the row's bytes under the salt at its `hm96`
slot. -/
theorem rsAtH_computes : (rsAtH ht hc hscope dj x₀ hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesAtH_parse ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesAtH_bcSite ht hc hscope dj x₀ hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-- **The headline at the executable over hidden-output sessions.** Let `I` be a verify command's inputs (one table)
under a tag with hidden outputs, whose one circuit file parses to an untyped circuit `c` (`hc`: `HmOut.parse`, which
checks each output row holds its port's words, `ht`) and whose public file loads to `pub` with no shared-row tables
(`hpub`: `HmOut.loadPublic`, every row's `b ‖ c` and no output word; `hpt`), and let the challenger write draw `S` as
the JSON `dj S` (`x₀` one draw it sets up with `Stmt.setupHidden`). Let the verifier draw the units with
`Flock.Draw.drawOS`'s stratified law on a byte source A3 makes uniform, and let each draw's session be the live coin
server's one table (`liveTable`, A6), written out by `wr`. Against any live prover `τ`, flock-verify accepts with at
least `K` wrong units of the production program `Placed.keyProg c hU n`, at the simulated prover's committed values (1
on the constant's gate, 0 on the zero's, the plurality of its openings elsewhere), with probability at most
`miss(K) + ε_ks + δ_link`.

Its hypotheses, past the session's inputs, are #828's `flock_headline_exec`'s at the hidden tables:
* the circuit is in `rs`'s scope, `hscope : Layout.scopeOkH c = true`: #828's, or its rows each a `sha512/row/v2` row
  that fits or a `sha512/row-seg/v1` row, of any mix, the output rows fit;
* `hdec` (`DecodesOn`, R11b, at the law's draws only), open, at the hidden tables `tablesAtH`: at a draw flock-verify
  accepts under hidden outputs whose record carries the challenger's draw `dj S`, proving `S`, with link points
  `mPts`, the table is `stmtOf` of the `Stmt.setupHidden` statement it accepted (`tableAtH_of_accepts`);
* the assumptions of record: A3 (`hA3`, `uniform/io-getrandombytes`), `cr/sha-512` for the knowledge finders (`hKS`),
  `ecr/sha-512` for the link finders (`hCR`); A6 (`uniform/flock-coin-server`) is the live session `liveTable` itself;
* numeric conditions and the strata's match (`hks`, `hS`, `hk`, `hRw`, `hM`, `hρ`, `hr`, `ht'`).

What differs from #828's headline: the statement is `Stmt.setupHidden`'s, whose public file registers each unit's
output rows' `b ‖ c` in place of its output words, so a unit's computed committed gates are bits of output rows bound
through their `b ‖ c` (`rsAtH`), not single gates pinned by region "out"; the units' committed values and the bound are
the same expressions at the hidden tables. Scope otherwise as #828's: W6 is definitional on the untyped path, `keyProg`
has no edges between units, the proof coins are the challenger's OS coins (A6), and sessions of more than one
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
    (ppH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.oneG} (hConstH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀ hU (Layout.outsOf c hU n))
    {Layout.zeroG} (hZeroH ht hc (rowsFit_of_scopeH ht hc hscope) dj x₀ hU (Layout.outsOf c hU n))
    (layAtH ht hc hscope dj x₀ hU (pub := pub)) (rsAtH ht hc hscope dj x₀ hU hpub hpt)
    (rsAtH_computes ht hc hscope dj x₀ hU hpub hpt) hk hRw hM hρ hr ht' K wr opn dg dec one fin capOf N hdec τ hCR hKS

end FlockSoundness.Discharge.HiddenExec
