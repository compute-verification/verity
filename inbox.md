[2026-11-14T20:44Z] 9ea322386de6
ROWSMERGE to network-accounting (lead), Sat 14 Nov 12:45 PM PST, mid-pass. (1) Local head 9209e7629d3 holds collect 4163f90b8ff1 (the world deletion 2f8cd83dee7c, 22f5f749e2e, 92c1cc2d06e0) and architecture d5848ca68518, all files resolved, every changed .py compiles; architecture's maps since are crossed onto our lines (law_lean, theta, inference_only with NciCheck to InferenceOnlyCheck, docket_names), each narrowed to our files, since run over the tree they refuse (stale words_in paths) or rewrite files architecture left alone. Next: architecture cfed9c4de201 (119 commits, 9 moves, 9 conflicts, mostly security/Proofs/Certifier and NetworkPeriod), then push both branches. (2) Splitting this step costs more than it saves. Handable after my push: the collect branch's newer 267 commits (4163f90b8ff1 to 1e498a266349) onto what I push, one worker on a branch of mine; I name the push in my next line. (3) Yes: a freeze of pushes to the collect branch from now to 1:20 PM PST lets this push land before the 1:05 maps. (4) Doing that now: this push is the first step (ours up to 4163f90b8ff1); the newer collect commits come next.

[2026-11-14T20:45Z] 4cdee64060c4
POLICYROWS2 to network-accounting (lead), Sat 14 Nov 12:55 PM PST, for NET-430.

Commit 97ea2472100b8980b2249391b048936e2043155a. The collect branch's head after merge and push is 866a98dd871b67c5e7414dcce866b361ad0094c5. The merge brought in `Move by tools/move/certifier_tests_back_map.toml` (7a709988129) and restacks. None of them touches `security/` or any Nci Lean, and my two files are unchanged by the merge. No file was resolved by hand and no .py changed. Pushed unbuilt; the lock was not regenerated.

Which values the premise states over (item b): what soundness asked for, reachable as written, with no fallback. The run's record's `window.registered_roots[param]` is a Lean value: `NciCheck.Baseline.strAt (getD (getD run "window") "registered_roots") p`, over `TrustedRun tr log run`, the same reading `IsTheta` uses (`security/Properties/Nci/Record.lean:366`). The domain id is the log's first record's registered entry's `domain`, `lastWith log[0]! "registered" "value" p`, the same entry whose `root` `IsTheta` reads. A window's supplied domain id comes from that entry too (`verity/commitments/windows/interface.py:1191`, `supplied`). Each is compared as lower-case hex, `Flock.hexOf`, the idiom `RunNames` uses. The premise is over the check's own form (`NciCheck.Record.SecurityProofs.form`), not an arbitrary `Form`: over an arbitrary form, `held` means nothing and the premise would be false. The root and domain-id rows are named as `Good` names them (`Record.lean:385`).

Names: `ThetaRootRowRegistered` (the open premise); `PinnedReadRegistered` (the composed statement), proved by `pinnedReadRegistered : ThetaRootRowRegistered → PinnedReadRegistered` in `Proofs.Properties.Nci.Lemmas.Opening`, a few lines from the premise and `pinnedReadBinds`. Neither is in `Assumptions.lean`, and `PinnedReadBinds` takes neither as a hypothesis.

(a) The module docstring of `security/Properties/Nci/PinnedRead.lean`, verbatim:

/-!
# A pinned reader's unit returns its table's row: the statements

`PinnedReadAt`, a pinned read as a check holds it and its circuit file lays it out, at a verdict bit; `PinnedRead`, at
the verdict bit 1; `ReturnsRow`, that the unit returns its table's row, the fill row, or a SHA-512 collision;
`PinnedReadBinds`, that a pinned read returns its row; `PinnedReadNontrivial`, that at the verdict bit 0 some pinned
read does not; `PinnedReadHonest`, that the check's own form (`NciCheck.Record.SecurityProofs.form`) has a pinned
read; `OpenedRowReturned`, that an opening's declared row is the row its reader returns; `ThetaRootRowRegistered`,
that the unit's root and domain id gates hold the ones registered for the opening's parameter; and
`PinnedReadRegistered`, the two together: the unit returns the row of the table registered under that parameter. The
proofs are `Proofs.Properties.Nci.Lemmas.Opening`'s.

A pinned read is a read by a pinned reader: the pin is the closure digest of the reader's Definition
(`NciCheck.Record.pinOf`), and pins no table and no root.
-/

(a) The sentence added to `PinnedReadBinds`' docstring (its statement is unchanged):

By itself it fixes no table: `table` is whatever is committed under the root and domain id the unit's gates hold in
`Yc`, and no hypothesis ties those to a registered root. The open premise `ThetaRootRowRegistered` ties them to the
ones registered for the opening's parameter, and `PinnedReadRegistered` is what the two give together. -/

(b) The premise, verbatim:

/-- **A θ table read's root and domain id rows hold the registered ones** (an open premise): where the check's own form
holds a pinned read of row `r`, opened from the table `name` of the parameter `p`, and classes every row of the same
unit's root port as a table read of that table's root and every row of its domain id port as one of its domain id,
the root the unit's root gates hold in `Yc` is the one the run's own record registers `p` under
(`window.registered_roots`, the root `IsTheta` reads), and the domain id its domain gates hold is the one the log's
first record registers for `p` (its last `registered` entry of that value, whose root `IsTheta` reads), each as its
lower-case hex. The check classes those rows and holds no row to a value, so this is not the check's; what would give
it is that a θ table read's rows, the values a window supplies for `p`'s root and domain id, are opened against the
registered root at those gates. Not proved, and stated nowhere else. -/
def ThetaRootRowRegistered : Prop :=
  ∀ (c : NciCheck.Record.SecurityProofs.form.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) (ct : Flock.Circuit) (tu : Flock.TmplUnit)
    (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening) (ok : Bool)
    (log : Array Json) (run e : Json) (R D : String),
    PinnedReadAt NciCheck.Record.SecurityProofs.form c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n
      consts u q Yc table ok →
    (∀ row < (NciCheck.Record.SecurityProofs.form.wiring c).cutsOf r.tid ro.root,
      NciCheck.Record.SecurityProofs.form.held c ⟨r.tid, r.unit, ro.root, row⟩ = some (.table name p true)) →
    (∀ row < (NciCheck.Record.SecurityProofs.form.wiring c).cutsOf r.tid ro.domain,
      NciCheck.Record.SecurityProofs.form.held c ⟨r.tid, r.unit, ro.domain, row⟩ = some (.table name p false)) →
    TrustedRun tr log run →
    NciCheck.Baseline.strAt (NciCheck.Baseline.getD (NciCheck.Baseline.getD run "window") "registered_roots") p =
      some R →
    NciCheck.Baseline.lastWith log[0]! "registered" "value" p = some e → NciCheck.Baseline.strAt e "domain" = some D →
    Flock.hexOf (ByteArray.mk (Verity.Digest.bytes (q.rootOf Yc)).toArray) = R ∧
      Flock.hexOf (ByteArray.mk (Verity.Digest.bytes (q.domainOf Yc)).toArray) = D

(c) The composed statement, verbatim:

/-- **A pinned read returns the row of the table registered under its parameter**: with the hypotheses of
`PinnedReadBinds` at the check's own form and those of `ThetaRootRowRegistered`, the unit's root and domain id gates
hold the root and domain id registered for the opening's parameter `p`, so `table`, committed under them
(`PinnedRead`), is the table registered under `p`; and the unit returns its row (`ReturnsRow`). It follows from
`ThetaRootRowRegistered` and `PinnedReadBinds` (`pinnedReadRegistered`), and holds no further than the open premise. -/
def PinnedReadRegistered : Prop :=
  ∀ (c : NciCheck.Record.SecurityProofs.form.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) (ct : Flock.Circuit) (tu : Flock.TmplUnit)
    (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening)
    (log : Array Json) (run e : Json) (R D : String),
    PinnedRead NciCheck.Record.SecurityProofs.form c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n
      consts u q Yc table →
    (∀ row < (NciCheck.Record.SecurityProofs.form.wiring c).cutsOf r.tid ro.root,
      NciCheck.Record.SecurityProofs.form.held c ⟨r.tid, r.unit, ro.root, row⟩ = some (.table name p true)) →
    (∀ row < (NciCheck.Record.SecurityProofs.form.wiring c).cutsOf r.tid ro.domain,
      NciCheck.Record.SecurityProofs.form.held c ⟨r.tid, r.unit, ro.domain, row⟩ = some (.table name p false)) →
    TrustedRun tr log run →
    NciCheck.Baseline.strAt (NciCheck.Baseline.getD (NciCheck.Baseline.getD run "window") "registered_roots") p =
      some R →
    NciCheck.Baseline.lastWith log[0]! "registered" "value" p = some e → NciCheck.Baseline.strAt e "domain" = some D →
    Flock.hexOf (ByteArray.mk (Verity.Digest.bytes (q.rootOf Yc)).toArray) = R ∧
      Flock.hexOf (ByteArray.mk (Verity.Digest.bytes (q.domainOf Yc)).toArray) = D ∧
      ReturnsRow ct hB n u q Yc table

(c) Its proof, verbatim:

/-- **`PinnedReadRegistered` from the open premise**: the premise gives the registered root and domain id, and
`pinnedReadBinds` the row. -/
theorem pinnedReadRegistered (h : ThetaRootRowRegistered) : PinnedReadRegistered := by
  intro c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table log run e R D hp hroot hdom
    htr hR he hD
  obtain ⟨h1, h2⟩ := h c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table true log run
    e R D hp hroot hdom htr hR he hD
  exact ⟨h1, h2, pinnedReadBinds NciCheck.Record.SecurityProofs.form c j tr P r name p pin k dfn ro out tags bytes
    tables ct tu hB n consts u q Yc table hp⟩

(d) `PinnedReadNontrivial`, restated as proposed, with soundness's sentence, verbatim:

/-- **The verdict bit does the work**: some form, record, circuit file and transcript meet every hypothesis of
`PinnedRead` but the verdict bit, which is 0 (`PinnedReadAt` at `false`); the opening the unit's gates hold is
`table`'s own at the index, and the row the unit returns is neither that opening's row nor the fill row. It shows the
first disjunct of `ReturnsRow` failing at an opening equal to the table's, and says nothing of the collision
disjuncts, which no kernel proof can exclude: at that opening each is a collision among one opening's own hashed
inputs. Not proved: it needs a circuit file that parses and a template the read's check accepts, runs of code on a
file, which the kernel does not evaluate. -/
def PinnedReadNontrivial : Prop :=
  ∃ (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn) (ro : Roles)
    (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
    (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedReadAt F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table false ∧
      q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) =
        table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ∧
      q.hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) ≠
        Hash.EmbeddingRead.h q.count q.words (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
          (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)))

(e) The lock's `open` entry needed at the close, in the shape of the existing entries (`security/lean-audit.json` `open`: owner, digest, issue, due; `--update` writes the digest):
  "NciCheck.Properties.Record.ThetaRootRowRegistered": {"owner": "@network-accounting", "issue": "<your Linear id>", "due": "<date>"}
What discharges it: a statement on the sampled proofs' side that a θ table read's rows are opened at the unit's root and domain-id gates against the registered root and domain id, so the transcript's gates hold them. Those rows are the window's supplied `<param>/root` and `<param>/domain` values, each committed as a registered value under its public salts (`windows/interface.py:1178`).

(f) Soundness's candidate. `V.Registered` (`security/Properties/Verity/Game.lean:802`) is the commit phase's file check: the committed files hold the circuit file and public file whose digests the parameters name. It says nothing of a unit's input rows. There is no `FamiliesPinned` on the branch, and `V.families` is the coin commitment's families, not rows. The nearest is `V.Opens` (`Game.lean:806`): a gate opens to a registered constant (`(V.read θ F).consts`), or by a row-leaf opening at its registered position. Reaching the root gates would take a statement that v2's file read registers a θ table read's supplied root and domain-id rows at those gates, and that the supplied values are `window.registered_roots[param]` and the registered entry's `domain`. None states that today.

SCANLEAN2 is told the modules: `Properties.Nci.PinnedRead` and `Proofs.Properties.Nci.Lemmas.Opening`.

