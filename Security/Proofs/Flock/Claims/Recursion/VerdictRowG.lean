import Proofs.Flock.Claims.Recursion.VerdictRows
import Proofs.Flock.Recursive.ExecClass

/-!
# Row G: the verdict's setup is the reader's

Row G of the verdict's rows (`VerdictRows`), the facts about the setup:

* `DecAt.read_setup`, `DecAt.setup_eq`: at SHA-512 the verifier's setup is the shipped one (`setupWith_sha512`), so it
  is the one `decR` read (`setupOf`), `spec` and `sts` both.
* `DecAt.sts_hidden`: where the statement is read at the inner tags (`RecVStar.innerTags`), `setupTables` sets every
  table up with `setupHidden` (`setupTables_okH`), at tags naming `flock-leaf/sha512-unsalted`.
* `DecAt.tab_eq`, `DecAt.merkle_of_tags`, `DecAt.foldKLog_of`, `DecAt.kLog_le_of_tags`: so table `i`'s setup commits
  under the registry's plain SHA-512 scheme (`Hidden.setupHidden_plain`), a `Named` one, its fold's `k_log` is its
  own, and its `k_log` is at most its `m` (`Hidden.setupHidden_wf`).
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs)

/-- **At SHA-512 the verifier's setup is the shipped one**, at the statement's read. -/
theorem read_setup : D.d.s.read.setup = .ok (D.spec, D.sts) := by
  have h := D.setup
  rw [FlockDeploy.hOf_sha512, FlockDeploy.setupWith_sha512] at h
  exact h

/-- **The verifier's setup is the reader's**: the session spec and statements `decR` read (`setupOf`). -/
theorem setup_eq : D.spec = D.d.spec ∧ D.sts = D.d.sts := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  obtain ⟨-, hs⟩ := FlockDeploy.setupOf_some hv
  have h1 := FlockDeploy.Ins.setup_read hs
  have h2 := D.read_setup
  rw [hd] at h2
  have h3 : (Except.ok (spec, sts) : Flock.V (SessionSpec × Array Stmt)) = .ok (D.spec, D.sts) := h1.symm.trans h2
  have e := Prod.mk.inj (Except.ok.inj h3)
  rw [hd]
  exact ⟨e.1.symm, e.2.symm⟩

/-- **Every table is a hidden-output table at plain leaves** where the statement is read at the inner tags:
`setupTables` sets each table up with `setupHidden` (`setupTables_okH`), at tags naming
`flock-leaf/sha512-unsalted`. -/
theorem sts_hidden (hT : D.d.s.read.tags = RecVStar.innerTags) :
    ∀ t ∈ D.sts, ∃ (tags : Tags) (cf pf : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
      (draw partition : Option Lean.Json) (program : Option ByteArray),
      tags.merkleLeaf = "flock-leaf/sha512-unsalted" ∧
        Stmt.setupHidden tags cf pf coins tables draw partition program = .ok t := by
  have h := D.read_setup
  unfold Readable.Read.setup at h
  rw [hT] at h
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH
      FlockSoundness.Discharge.Recursion.innerTags_hiddenOutputs h with
    ⟨-, st, hst, -, hsts⟩ | ⟨-, -, hall, -, -⟩
  · rw [hsts]
    intro t ht
    obtain rfl : t = st := by simpa using ht
    exact ⟨_, _, _, _, _, _, _, _, rfl, hst⟩
  · intro t ht
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 ht
    obtain ⟨d, hd⟩ := hall j hj
    exact ⟨_, _, _, _, _, _, _, _, rfl, hd⟩

/-- Table `i`'s setup is its statement's. -/
theorem tab_eq {i : ℕ} (hi : i < D.sts.size) : (tabsOf D.sts)[i]! = Setup.ofCircuit D.sts[i] := by
  unfold FlockDeploy.tabsOf
  rw [getElem!_pos (D.sts.map Setup.ofCircuit) i (by simpa using hi), Array.getElem_map]

/-- **Row G's `merkle` and `named`**: table `i`'s setup commits under plain SHA-512, a `Named` scheme. -/
theorem merkle_of_tags (hT : D.d.s.read.tags = RecVStar.innerTags) {i : ℕ} (hi : i < D.sts.size) :
    (tabsOf D.sts)[i]!.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" ∧
      (tabsOf D.sts)[i]!.merkle.Named := by
  have hm : (tabsOf D.sts)[i]!.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rw [D.tab_eq hi]
    obtain ⟨_, _, _, _, _, _, _, _, hl, hs⟩ := D.sts_hidden hT _ (Array.getElem_mem hi)
    exact FlockSoundness.Discharge.Hidden.setupHidden_plain hl hs
  refine ⟨hm, ?_⟩
  rw [hm]
  exact With.named_of_find? (id := "flock-leaf/sha512-unsalted") rfl

variable (cls : FlockSoundness.Discharge.Recursion.HClass)

/-- **Row G's `foldKLog`**: table `i`'s fold is at its own `k_log`. -/
theorem foldKLog_of {i : ℕ} (hi : i < D.sts.size) : (D.files cls i).st.fold.kLog = (D.sh cls i).kLog := by
  show (tabsOf D.sts)[i]!.fold.kLog = (tabsOf D.sts)[i]!.kLog
  rw [D.tab_eq hi]
  rfl

/-- **Row G's `kLog_le`**: table `i`'s `k_log` is at most its `m` (`StmtWF`). -/
theorem kLog_le_of_tags (hT : D.d.s.read.tags = RecVStar.innerTags) {i : ℕ} (hi : i < D.sts.size) :
    (D.files cls i).st.kLog ≤ (D.files cls i).st.m := by
  show (tabsOf D.sts)[i]!.kLog ≤ (tabsOf D.sts)[i]!.m
  rw [D.tab_eq hi]
  obtain ⟨_, _, _, _, _, _, _, _, -, hs⟩ := D.sts_hidden hT _ (Array.getElem_mem hi)
  exact (FlockSoundness.Discharge.Hidden.setupHidden_wf hs).1.hk

end DecAt

end RecDeploy.Rows
