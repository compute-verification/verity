import Proofs.Flock.Claims.KS.ZkDecJ
import Proofs.Flock.Claims.Recursion.DeployPremises
import Proofs.Flock.Claims.Scope.Tables

/-!
# Recursion completeness at `deployRec`: the honest outer witnesses are in the V\* relation

On an honest first part, the reference prover's (`refP`) inner and staging rounds, each answered by a coin
(`HonestPre`), where the inner verifier accepts the inner proof's messages at those coins (`InnerAccepts`), every part's
V\* statement as the compiled verifier reads it (`vstar` at those rounds) is the one staged there, it passes the
recursive setup (`setupO`), and the outer witness the reference prover hands its masked session (`RefParts.wit`) is in
the V\* relation at it (`Claims.Flock.R` of `deployZkRec Readable.relation`): `vstar_complete`. So
`Claims.Recursion.OuterHolds` holds of the parts' honest witnesses (`outerHolds_honest`), and on the compiled
verifier's own runs of `refP` at a statement in the recursion's relation (`honestPre_run`, `vstar_complete_run`) every
V\* statement the run reads (`runVstar`) is in the relation at its honest witness.

The proof is the glue between the compiled verifier's code and three facts about what V\*'s builder lays out, each a
named `Prop`:

* `NumsFixed` (the stager's layout, `vstar stage`): a part's numbers are the same at any inner rounds where it sets up,
  so the honest staged statement passes the session's numbers check and `vstarAt` takes it (`vstarAt_staged`);
* `StagedSetsUp` (V\*'s builder): at an honest first part whose inner proof the inner verifier accepts, each staged
  statement passes the recursive setup, from which it parses (`parseBytes_of_setupO`);
* `WitSatisfies` (V\*'s builder and its lowering): there, each table of a staged statement's setup fits the first
  table's blocks and its share of the honest witness satisfies its model statement (`TablesSat`), which is the relation
  at the deployed `Readable.relation` (`relS_relation`).

`LinksHonest` is the interface to the checks that bind `rec-top-L<l>` rows and the link points to the inner transcript
(`RecVStar.checkLinkRounds`, `RecVStar.checkTopRows`, `RecCoins.checkLink` at `RecVStar.innerLink`), passed in as
`linkOk`: they pass at the coin server's transcript of the honest inner rounds and the record the honest prover staged.
A staged statement holds exactly that transcript (`transcript_honest`), so the checks pass at the statement's own
inputs. The deployed checks are in `RecIns.setup`, so at them (`linkOkSetup`) `LinksHonest` holds with no
premise (`linksHonest_setup`): `vstar_complete_setup` and `vstar_complete_run_setup` are the theorems without it, and
the link input they pass at is the one `RecVStar.innerLink` reads off the staged transcript (`insO_staged_link`).

The inner verifier is arbitrary here. At the deployed plain inner verifier, at SHA-512, `InnerAccepts` holds only at an
inner statement that verifier admits (`InnerAdmitted`, `innerAdmitted_of_accepts`): its scope check passes and it
carries a unit draw. A recursive session's inner statement has plain SHA-512 leaves, which the scope check admits
without `--zk` and with hidden outputs (`ProvedScope.leafOk`); where the scope check or the draw fails,
`InnerAccepts` never holds and the theorem says nothing about the statement.
`InnerAccepts` is a hypothesis about the coins, not a consequence of the relation: at coins in the inner completeness
error the honest inner proof is refused, and the V\* statements staged there need not have a witness.
-/

namespace RecDeploy

open Claims (Msg Out Prover run coinSeq replay)
open Flock
open FlockDeploy (Coin)

/-! ## An honest first part -/

/-- **An honest first part**: `v` holds the inner and staging rounds, each message the reference prover's against `H`
on tape `ω` after the rounds before it, each answer the bytes of the coin `cs` holds at its round. -/
structure HonestPre (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) (x : Msg)
    (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)) : Prop where
  length : v.length = IV.k (xIOf x) + kS x
  coins : cs.length = v.length
  msg : ∀ i (h : i < v.length), v[i].1 = (((refP IV b w).next x (v.take i) ω).eval H).1
  ans : ∀ i (h : i < v.length), v[i].2 = (cs.getD i default).bytes

/-- **The inner verifier accepts the honest inner proof**: `IV`'s run against `H` of the inner rounds' messages of `v`,
at the coins `cs`, accepts. -/
def InnerAccepts (H : Msg → Msg) (IV : Claims.Verifier Coin) (x : Msg) (cs : List Coin) (v : List (Msg × Msg)) :
    Prop :=
  (run H (replay ((v.take (IV.k (xIOf x))).map Prod.fst)) IV (xIOf x) () fun i => cs.getD i.val default).out = .acc

/-! ## What V\*'s builder lays out -/

/-- **A part's numbers are fixed by the statement** (the stager's layout, `vstar stage`): at any two inner rounds where
part `s`'s staged statement passes the recursive setup, its numbers are the same. The staging message fills the part's
public file at a layout the compiled statement fixes, and the numbers read only the setup's session and tables. -/
def NumsFixed (st : Stage) : Prop :=
  ∀ (x : Msg) (ms ms' : List (Msg × Out)) (s : ℕ), (setupO (staged st x ms s)).isSome →
    (setupO (staged st x ms' s)).isSome → numsOf (staged st x ms s) = numsOf (staged st x ms' s)

/-- **V\*'s builder stages statements that set up**: at an honest first part whose inner proof the inner verifier
accepts, each part's staged statement passes the recursive setup (`setupO`: `RecIns.setup`, the checks `verify` runs
on the recursive inputs). -/
def StagedSetsUp (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) : Prop :=
  ∀ (x : Msg) (w : Claims.Flock.Wit) (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)),
    HonestPre H IV b w x ω cs v → InnerAccepts H IV x cs v → ∀ s < nS x,
      (setupO (staged b.st x (asRounds v) s)).isSome

/-- **A witness satisfies tables `sts`, table by table**: each fits the first table's blocks, and its share of the
witness's hidden values (`zOf`) satisfies its model statement. -/
def TablesSat (sts : Array Stmt) (w : Claims.Flock.Wit) : Prop :=
  ∀ c < sts.size, ∃ h : (Readable.Tab.ofStmt sts[c]!).Fits ((sts[0]?.map (·.m)).getD 0)
      ((sts[0]?.map (·.c.kLog)).getD 0),
    ((Readable.Tab.ofStmt sts[c]!).toModel h.2.2).Satisfies (Readable.zOf sts.size w c sts[c]!.m)

/-- **V\*'s builder's witnesses satisfy its tables**: at an honest first part whose inner proof the inner verifier
accepts, the outer witness the reference prover hands part `s`'s session (`RefParts.wit`: the inner proof's messages,
the part's rows and the openings, as its circuit evaluates them) satisfies every table of the part's staged statement's
setup (`TablesSat`). V\*'s circuits compute the inner verifier's checks, which pass there. -/
def WitSatisfies (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) : Prop :=
  ∀ (x : Msg) (w : Claims.Flock.Wit) (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)),
    HonestPre H IV b w x ω cs v → InnerAccepts H IV x cs v → ∀ s < nS x, ∀ i spec sts,
      setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) →
        TablesSat sts (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s)

/-- **The link checks pass on an honest run** (`linkOk` at a statement's recursive inputs, its inner transcript, its
inner session's record and its tables: the checks `verify`'s setup runs after `RecIns.setup`'s,
`RecVStar.innerLink`, which holds the record's link points to the transcript's, then `RecVStar.checkLinkRounds`,
`RecVStar.checkTopRows` and `RecCoins.checkLink`, which hold the `rec-top-L<l>` rows and the openings' tops to the
transcript's). At an honest first part, at each part's staged statement that sets up, they pass at the coin server's
transcript of the honest inner rounds and the record the honest prover staged: its link roots and points are the ones
the coin server recorded. -/
def LinksHonest (linkOk : RecIns → ByteArray → ByteArray → Array Stmt → Bool) (H : Msg → Msg)
    (IV : Claims.Verifier Coin) (b : Brings) : Prop :=
  ∀ (x : Msg) (w : Claims.Flock.Wit) (ω : Tape b) (cs : List Coin) (v : List (Msg × Msg)),
    HonestPre H IV b w x ω cs v → ∀ s < nS x, ∀ i spec sts,
      setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) →
        linkOk i (b.tr x (asRounds (v.take (IV.k (xIOf x))))) (b.st.record x (asRounds v)) sts = true

/-! ## The deployed link checks -/

/-- **The link checks the recursive setup runs** (in `RecIns.setup`), as a `linkOk`:
`RecVStar.checkLinkRounds` at the inner shape and the link input, then `RecVStar.checkTopRows` and `RecCoins.checkLink`
on each table. They read the link input `RecRaw.ins` reads (`linkInputOf`: `RecVStar.innerLink` on the statement's own
inner transcript and record, `insO_staged_link`), not the transcript and record passed in. -/
def linkOkSetup (i : RecIns) (_ _ : ByteArray) (sts : Array Stmt) : Bool :=
  (match i.inner, i.link with
    | some n, some lk => Except.toBool (RecVStar.checkLinkRounds n lk)
    | _, _ => true) &&
    Except.toBool (sts.forM (RecVStar.checkTopRows · i.base.registered i.link)) &&
    Except.toBool (sts.forM (RecCoins.checkLink · i.base.registered (RecCoins.Source.atC3 i.src i.inner i.commits)
      i.link))

theorem RecIns.setup_linkRounds {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts))
    {n : RecVStar.Inner} {lk : RecVStar.InnerLink} (hn : s.inner = some n) (hl : s.link = some lk) :
    RecVStar.checkLinkRounds n lk = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure, hn, hl] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem RecIns.setup_topRows {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    sts.forM (RecVStar.checkTopRows · s.base.registered s.link) = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem RecIns.setup_checkLink {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    sts.forM (RecCoins.checkLink · s.base.registered (RecCoins.Source.atC3 s.src s.inner s.commits) s.link) =
      .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

/-- **A statement that passes the recursive setup passes its link checks**, at any transcript and record. -/
theorem linkOkSetup_of_setup {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : i.setup = .ok (spec, sts))
    (t r : ByteArray) : linkOkSetup i t r sts = true := by
  unfold linkOkSetup
  rw [RecIns.setup_topRows h, RecIns.setup_checkLink h]
  rcases hn : i.inner with _ | n <;> rcases hl : i.link with _ | lk
  · simp [Except.toBool]
  · simp [Except.toBool]
  · simp [Except.toBool]
  · simp [RecIns.setup_linkRounds h hn hl, Except.toBool]

/-- **The deployed link checks pass on an honest run**: `LinksHonest` at `linkOkSetup` holds with no premise, the
recursive setup running them (`linkOkSetup_of_setup`). -/
theorem linksHonest_setup (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) :
    LinksHonest linkOkSetup H IV b := by
  intro x w ω cs v hv s hs i spec sts hsu
  exact linkOkSetup_of_setup (setupO_some hsu).2 _ _

/-- **The link input the reader reads** is `linkInputOf` at the statement's source. -/
theorem RecRaw.ins_link {w : RecRaw} {s : RecIns} (h : w.ins = .ok s) : linkInputOf w s.src = .ok s.link := by
  unfold RecRaw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

/-- **A staged statement's link input is read off its own inner transcript and record**, the stage's: its raw inputs
carry `Stage.transcript` and `Stage.record` (`partRaw`), and `linkInputOf` reads them. -/
theorem insO_staged_link {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) :
    ∃ wr, decodeRec (staged st x ms s) = some wr ∧ wr.transcript = some (st.transcript x ms) ∧
      wr.innerSession = some (st.record x ms) ∧ linkInputOf wr i.src = .ok i.link := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, -, -, hd⟩ := staged_decode hne
  refine ⟨partRaw r st x ms s p, hd, rfl, rfl, ?_⟩
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  exact RecRaw.ins_link (toOption_eq_some h')

/-! ## The inner statement an honest inner run is accepted at -/

/-- **The deployed plain verifier admits the inner statement**: it sets up, `verify`'s scope check
(`ProvedScope.check`) passes on it, and it carries a unit draw (`ProvedScope.drawLaw` refuses a session without one).
At a recursive session's inner statement, whose leaves are plain SHA-512, the scope check passes only without `--zk`
and with hidden outputs (`ProvedScope.leafOk`, `Readable.check_ok_iff`, `Readable.Rules`; `inner_check_plain`). -/
def InnerAdmitted (x : Msg) : Prop :=
  ∃ ins spec sts, FlockDeploy.setupOf false (xIOf x) = some (ins, spec, sts) ∧ ins.read.check = .ok () ∧
    ∃ d, ins.read.draw = some d

/-- **An honest inner run the deployed plain verifier accepts is of a statement it admits** (`acc_check`,
`acc_draw_of_setup`), at SHA-512: `InnerAccepts` there holds only where `InnerAdmitted` does. -/
theorem innerAdmitted_of_accepts {x : Msg} {cs : List Coin} {v : List (Msg × Msg)}
    (h : InnerAccepts FlockDeploy.sha512 (FlockDeploy.V false) x cs v) : InnerAdmitted x := by
  have hacc : ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.V false) (xIOf x) := fun hn => hn _ _ _ _ h
  obtain ⟨ins, spec, sts, -, -, hs, -, hc⟩ := FlockDeploy.KS.acc_check hacc
  exact ⟨ins, spec, sts, hs, hc, FlockDeploy.KS.acc_draw_of_setup hacc hs⟩

/-! ## The code's side -/

/-- **A statement that passes the recursive setup parses**: its reading's setup is the recursive one's, and its
tables pass `Zk.shapeOk`, the statement being masked. -/
theorem parseBytes_of_setupO {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupO xO = some (i, spec, sts)) :
    Readable.parseBytes readOfO xO = some ({ i.base.read with sts } : Readable.Parsed) := by
  obtain ⟨hins, hsu⟩ := setupO_some h
  have hr : readOfO xO = some i.base.read := by rw [readOfO, hins, Option.map_some]
  have hz := readOfO_zk hr
  unfold Readable.parseBytes Readable.Read.parse Readable.Read.tablesOf
  rw [hr, Option.bind_some, RecIns.setup_read hsu]
  simp only [hz, ↓reduceIte, RecIns.setup_shapeOk hsu hz, Option.map_some]

/-- **The V\* relation at a statement that sets up**, from its tables: at the deployed relation, `RelS` is every
table's model statement (`relS_relation`). -/
theorem R_of_setupO {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupO xO = some (i, spec, sts)) {w : Claims.Flock.Wit} (ht : TablesSat sts w) :
    Claims.Flock.R (deployZkRec Readable.relation).parse xO w := by
  refine ⟨Readable.toReadable Readable.relation ({ i.base.read with sts } : Readable.Parsed), ?_, ?_⟩
  · show Readable.parseOf Readable.relation readOfO xO = _
    unfold Readable.parseOf
    rw [parseBytes_of_setupO h, Option.map_some]
  · exact (Readable.relS_relation _ w).2 ht

/-- `Flock.check` accepts where the relation holds. -/
theorem check_of_R {parse : Msg → Option Claims.Flock.Stmt} {x : Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.R parse x w) : Claims.Flock.check parse x w = true := by
  obtain ⟨s, hs, hr⟩ := h
  unfold Claims.Flock.check
  rw [hs]
  exact decide_eq_true hr

/-- **The compiled verifier reads a staged statement that sets up**: its numbers are `pick`'s, whose statement sets up
too, so `vstarAt` takes it. -/
theorem vstarAt_staged {st : Stage} (hN : NumsFixed st) {x : Msg} {ms : List (Msg × Out)} {s : ℕ}
    (hs : (setupO (staged st x ms s)).isSome) : vstarAt st x ms s = staged st x ms s := by
  have he : ∃ ms, (setupO (staged st x ms s)).isSome := ⟨ms, hs⟩
  have hp : (setupO (staged st x (pick st x s) s)).isSome := by
    unfold pick
    simp only [he, ↓reduceDIte]
    exact Classical.choose_spec he
  unfold vstarAt
  simp only [show numsOf (staged st x ms s) = nums st x s from hN x ms (pick st x s) s hs hp, ↓reduceIte]

/-- **A staged statement's inner transcript is the coin server's of the inner rounds**: the stage reads the transcript
of the rounds before the staging message. -/
theorem transcript_honest (b : Brings) (x : Msg) {v : List (Msg × Msg)} {n : ℕ} (hv : v.length = n + kS x) :
    b.st.transcript x (asRounds v) = b.tr x (asRounds (v.take n)) := by
  show b.tr x (asRounds v).dropLast = _
  unfold asRounds
  rw [List.dropLast_eq_take, List.length_map, hv, List.map_take]
  rfl

/-! ## Completeness of the V\* statements -/

/-- **The honest outer witness is in the V\* relation, part by part**: at an honest first part whose inner proof the
inner verifier accepts, part `s`'s V\* statement as the compiled verifier reads it is the staged one, it passes the
recursive setup, and the outer witness the reference prover hands its session is in the V\* relation at it. -/
theorem vstar_staged_relation (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v)
    {s : ℕ} (hs : s < nS x) :
    (vstar b.st x (asRounds v)).getD s [] = staged b.st x (asRounds v) s ∧
      ∃ i spec sts, setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) ∧
        Claims.Flock.R (deployZkRec Readable.relation).parse (staged b.st x (asRounds v) s)
          (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) := by
  have hsu := hS x w ω cs v hv hI s hs
  obtain ⟨⟨i, spec, sts⟩, he⟩ := Option.isSome_iff_exists.1 hsu
  refine ⟨?_, i, spec, sts, he, R_of_setupO he (hW x w ω cs v hv hI s hs i spec sts he)⟩
  rw [vstar_getD _ _ _ hs]
  exact vstarAt_staged hN hsu

/-- **Recursion completeness at `deployRec`, part by part**: at an honest first part whose inner proof the inner
verifier accepts, part `s`'s V\* statement as the compiled verifier reads it is the staged one, it passes the
recursive setup and the link checks at its own inner transcript and record, and the honest outer witness is in the V\*
relation at it. -/
theorem vstar_complete (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings)
    (linkOk : RecIns → ByteArray → ByteArray → Array Stmt → Bool) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) (hL : LinksHonest linkOk H IV b) {x : Msg}
    {w : Claims.Flock.Wit} {ω : Tape b} {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v)
    (hI : InnerAccepts H IV x cs v) {s : ℕ} (hs : s < nS x) :
    (vstar b.st x (asRounds v)).getD s [] = staged b.st x (asRounds v) s ∧
      ∃ i spec sts, setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) ∧
        Claims.Flock.R (deployZkRec Readable.relation).parse (staged b.st x (asRounds v) s)
          (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) ∧
        linkOk i (b.st.transcript x (asRounds v)) (b.st.record x (asRounds v)) sts = true := by
  obtain ⟨he, i, spec, sts, hsu, hR⟩ := vstar_staged_relation H IV b hN hS hW hv hI hs
  refine ⟨he, i, spec, sts, hsu, hR, ?_⟩
  rw [transcript_honest b x hv.length]
  exact hL x w ω cs v hv s hs i spec sts hsu

/-- **Recursion completeness at `deployRec` with the deployed link checks**: `vstar_complete` at `linkOkSetup`, whose
`LinksHonest` is `linksHonest_setup`. The checks pass at the link input read off the staged statement's inner
transcript, the coin server's of the honest inner rounds, and its record (`insO_staged_link`). -/
theorem vstar_complete_setup (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v)
    {s : ℕ} (hs : s < nS x) :
    (vstar b.st x (asRounds v)).getD s [] = staged b.st x (asRounds v) s ∧
      ∃ i spec sts, setupO (staged b.st x (asRounds v) s) = some (i, spec, sts) ∧
        Claims.Flock.R (deployZkRec Readable.relation).parse (staged b.st x (asRounds v) s)
          (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) ∧
        linkOkSetup i (b.st.transcript x (asRounds v)) (b.st.record x (asRounds v)) sts = true ∧
        ∃ wr, decodeRec (staged b.st x (asRounds v) s) = some wr ∧
          wr.transcript = some (b.tr x (asRounds (v.take (IV.k (xIOf x))))) ∧
          wr.innerSession = some (b.st.record x (asRounds v)) ∧ linkInputOf wr i.src = .ok i.link := by
  obtain ⟨he, i, spec, sts, hsu, hR, hl⟩ :=
    vstar_complete H IV b linkOkSetup hN hS hW (linksHonest_setup H IV b) hv hI hs
  obtain ⟨wr, hd, ht, hr, hk⟩ := insO_staged_link (setupO_some hsu).1
  refine ⟨he, i, spec, sts, hsu, hR, hl, wr, hd, ?_, hr, hk⟩
  rw [ht, transcript_honest b x hv.length]

/-- **The honest outer witnesses are in the V\* relation at the V\* statements the compiled verifier reads**, in the
form the recursion's ZK composition takes them (its `hW`), at an honest first part: part `s`'s statement `vstar` reads
at the first part's rounds, and the witness the reference prover hands its session. -/
theorem vstar_relation (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v) :
    ∀ s < nS x, Claims.Flock.R (deployZkRec Readable.relation).parse ((vstar b.st x (asRounds v)).getD s [])
      (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))) s) := by
  intro s hs
  obtain ⟨he, -, -, -, -, hR⟩ := vstar_staged_relation H IV b hN hS hW hv hI hs
  rw [he]
  exact hR

/-- **`OuterHolds` of the honest outer witnesses**: one per part, each accepted by `Flock.check` at its V\*
statement. -/
theorem outerHolds_honest (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H IV b) (hW : WitSatisfies H IV b) {x : Msg} {w : Claims.Flock.Wit} {ω : Tape b}
    {cs : List Coin} {v : List (Msg × Msg)} (hv : HonestPre H IV b w x ω cs v) (hI : InnerAccepts H IV x cs v) :
    Claims.Recursion.OuterHolds (deployZkRec Readable.relation) (vstar b.st x (asRounds v)) (nS x)
      ((List.range (nS x)).map (b.ref.wit w ω.1 (v.take (IV.k (xIOf x))))) := by
  refine ⟨by simp, fun s h => ?_⟩
  have hs : s < nS x := by simpa using h
  rw [List.getElem_map, List.getElem_range]
  exact check_of_R (vstar_relation H IV b hN hS hW hv hI s hs)

/-! ## On the compiled verifier's runs -/

/-- **A run whose verifier answers every round before `K` with a message gets past them**, with rounds enough. -/
theorem stepWith_length_ge {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vn : List Msg → Claims.OC Out) (K : ℕ)
    (hvn : ∀ (as : List Msg) (a : Msg), as.length < K → ∃ m lo, (vn (as ++ [a])).eval H = (.msg m, lo)) :
    ∀ (f : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Claims.Trace), as.length = t.msgs.length →
      K ≤ t.msgs.length + f → K ≤ (Claims.stepWith H P x ω vn f hist as t).msgs.length
  | 0, _, _, t, _, hK => by simpa [Claims.stepWith] using hK
  | f + 1, hist, as, t, ha, hK => by
    by_cases hk : K ≤ t.msgs.length
    · exact hk.trans (stepWith_mono H P x ω vn (f + 1) hist as t).1.length_le
    obtain ⟨m, lo, hm⟩ := hvn as ((P.next x hist ω).eval H).1 (by omega)
    rw [stepWith_succ_msg H P x ω vn f hist as t hm]
    exact stepWith_length_ge H P x ω vn K hvn f _ _ _ (by simp [ha]) (by simp; omega)

/-- **A run of a statement that reads gets past the inner and staging rounds**: the compiled verifier answers each
with its coin. -/
theorem run_length_ge (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) {x : Msg} {xr : XR}
    (hx : readX x = some xr) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V IV st).k x) → Coin) :
    IV.k (xIOf x) + kS x ≤ (run H P (V IV st) x ω r).msgs.length := by
  rw [run_eq]
  refine stepWith_length_ge H P x ω _ _ (fun as a hl => ?_) _ [] [] _ rfl
    (by simp only [List.length_nil, Nat.zero_add, V]; omega)
  refine ⟨(coinSeq r as.length).bytes, [], ?_⟩
  have hc := ract_inner (IV := IV) (st := st) hx (as := as ++ [a])
    (cs := (List.range (as ++ [a]).length).map (coinSeq r)) (c := coinSeq r as.length)
    (by simp only [List.length_append, List.length_singleton]; omega)
    (by simp [List.range_succ])
  exact next_coin H hc

/-- **An inner or staging round of a run, as a round**: its answer is a message. -/
theorem run_take_asRounds (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) {x : Msg} {xr : XR}
    (hx : readX x = some xr) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V IV b.st).k x) → Coin) :
    asRounds (((run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x)).map histOf) =
      (run H P (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x) := by
  have hlen := run_length_ge H IV b.st hx P ω r
  unfold asRounds
  rw [List.map_map]
  apply List.ext_getElem (by simp)
  intro i h₁ h₂
  simp only [List.length_map, List.length_take] at h₁
  obtain ⟨a, ha⟩ := run_inner H IV b.st x P ω r hx (i := i) (by omega) (by omega)
  rw [List.getElem?_eq_getElem (by omega)] at ha
  simp only [List.getElem_map, List.getElem_take, Function.comp, histOf, Option.some.inj ha, Claims.Recursion.outMsg]

/-- **The honest compiled run's inner and staging rounds are an honest first part**, at the run's coins, at a
statement that reads. -/
theorem honestPre_run (H : Msg → Msg) (IV : Claims.Verifier Coin) (b : Brings) (w : Claims.Flock.Wit) {x : Msg}
    {xr : XR} (hx : readX x = some xr) (ω : Tape b) (r : Fin ((V IV b.st).k x) → Coin) :
    HonestPre H IV b w x ω (runCs (IV.k (xIOf x) + kS x) r)
      (((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x)).map histOf) where
  length := by
    have := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [List.length_map, List.length_take]
    omega
  coins := by
    have := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [runCs, List.length_map, List.length_range, List.length_take]
    omega
  msg i h := by
    have hlen := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [List.length_map, List.length_take] at h
    obtain ⟨new, hnew, hget⟩ := run_get H (refP IV b w) (V IV b.st) x ω r i (by omega)
    rw [List.getElem?_eq_getElem (by omega)] at hget
    have hv : (((run H (refP IV b w) (V IV b.st) x ω r).msgs.take (IV.k (xIOf x) + kS x)).map histOf).take i =
        new.map histOf := by
      rw [hnew, ← List.map_take, List.take_take, Nat.min_eq_left (by omega)]
    rw [hv]
    simp only [List.getElem_map, List.getElem_take, histOf, Option.some.inj hget]
  ans i h := by
    have hlen := run_length_ge H IV b.st hx (refP IV b w) ω r
    simp only [List.length_map, List.length_take] at h
    obtain ⟨a, ha⟩ := run_inner H IV b.st x (refP IV b w) ω r hx (i := i) (by omega) (by omega)
    rw [List.getElem?_eq_getElem (by omega)] at ha
    have hc : (runCs (IV.k (xIOf x) + kS x) r).getD i default = coinSeq r i := by
      simp [runCs, List.getD_eq_getElem?_getD, show i < IV.k (xIOf x) + kS x by omega]
    simp only [List.getElem_map, List.getElem_take, histOf, Option.some.inj ha, Claims.Recursion.outMsg, hc]

/-- A statement that parses reads. -/
theorem readX_some_of_parse {x : Msg} {s : Claims.Recursion.CompiledStatement} (h : parse x = some s) : ∃ xr, readX x = some xr := by
  cases hx : readX x with
  | some xr => exact ⟨xr, rfl⟩
  | none =>
    simp [parse, RecScope.parseOf, RecScope.parseBytes, decodeR, hx] at h

/-- **Recursion completeness at `deployRec`, on its runs**: at a statement in the recursion's relation, on the honest
compiled run (the record's reference prover `refP` on tape `ω`, at the compiled verifier's coins `r`), where the inner
verifier accepts the inner proof's messages at the run's inner coins, every V\* statement the run reads (`runVstar`)
is in the V\* relation at the outer witness the reference prover hands its session, read off the inner rounds as the
prover saw them, and passes the recursive setup and the link checks at its own inner transcript. -/
theorem vstar_complete_run (H : Msg → Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (ivScope : Msg → Bool)
    (b : Brings) (linkOk : RecIns → ByteArray → ByteArray → Array Stmt → Bool) (hN : NumsFixed b.st)
    (hS : StagedSetsUp H I.V b) (hW : WitSatisfies H I.V b) (hL : LinksHonest linkOk H I.V b) {x : Msg}
    {w : Claims.Flock.Wit} (hR : Claims.Recursion.R (deployRec I.V ivScope b) I x w) (ω : Tape b)
    (r : Fin ((V I.V b.st).k x) → Coin)
    (hI : InnerAccepts H I.V x (runCs (I.V.k (xIOf x) + kS x) r)
      (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map histOf))
    {s : ℕ} (hs : s < nS x) :
    ∃ i spec sts, setupO ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω
        r).getD s []) = some (i, spec, sts) ∧
      Claims.Flock.R (deployZkRec Readable.relation).parse
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r).getD s [])
        (b.ref.wit w ω.1 (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s) ∧
      linkOk i (b.st.transcript x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)))
        (b.st.record x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))) sts = true := by
  obtain ⟨s', hp, -, -⟩ := hR
  obtain ⟨xr, hx⟩ := readX_some_of_parse hp
  have hv := honestPre_run H I.V b w hx ω r
  obtain ⟨he, i, spec, sts, hsu, hrel, hl⟩ := vstar_complete H I.V b linkOk hN hS hW hL hv hI hs
  have hrv : Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r =
      vstar b.st x (asRounds (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take
        (I.V.k (xIOf x) + kS x)).map histOf)) := by
    rw [run_take_asRounds H I.V b hx]
    rfl
  have hwit : (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map histOf).take
      (I.V.k (xIOf x)) = ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf := by
    rw [← List.map_take, List.take_take, Nat.min_eq_left (Nat.le_add_right _ _)]
  rw [hwit] at hrel
  rw [run_take_asRounds H I.V b hx] at he hsu hrel hl
  refine ⟨i, spec, sts, ?_, ?_, hl⟩
  · rw [hrv, run_take_asRounds H I.V b hx, he]
    exact hsu
  · rw [hrv, run_take_asRounds H I.V b hx, he]
    exact hrel

/-- **Recursion completeness at `deployRec`, on its runs, with the deployed link checks**: `vstar_complete_run` at
`linkOkSetup`, whose `LinksHonest` is `linksHonest_setup`. -/
theorem vstar_complete_run_setup (H : Msg → Msg) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit)
    (ivScope : Msg → Bool) (b : Brings) (hN : NumsFixed b.st) (hS : StagedSetsUp H I.V b) (hW : WitSatisfies H I.V b)
    {x : Msg} {w : Claims.Flock.Wit} (hR : Claims.Recursion.R (deployRec I.V ivScope b) I x w) (ω : Tape b)
    (r : Fin ((V I.V b.st).k x) → Coin)
    (hI : InnerAccepts H I.V x (runCs (I.V.k (xIOf x) + kS x) r)
      (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).map histOf))
    {s : ℕ} (hs : s < nS x) :
    ∃ i spec sts, setupO ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω
        r).getD s []) = some (i, spec, sts) ∧
      Claims.Flock.R (deployZkRec Readable.relation).parse
        ((Claims.Recursion.runVstar H (deployRec I.V ivScope b) I (refP I.V b w) x ω r).getD s [])
        (b.ref.wit w ω.1 (((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x))).map histOf) s) ∧
      linkOkSetup i (b.st.transcript x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)))
        (b.st.record x ((run H (refP I.V b w) (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))) sts = true :=
  vstar_complete_run H I ivScope b linkOkSetup hN hS hW (linksHonest_setup H I.V b) hR ω r hI hs

end RecDeploy
