module

public import Proofs.Properties.Verity.Sampled.Record
public import Proofs.Properties.Verity.Sampled.Record.Session
public import Proofs.Properties.Verity.Recursive.ScheduleInner
public import Proofs.Properties.Verity.Recursive.PartStatement
import all Init.Data.String.Defs

/-!
# v2's acceptance is the record's, past the draw file round

Where one-stage v2 (`Spot.OneStageZk.checkBlobsAt`) accepts, the record's verifier (`Record.V`) accepts a run
whose draw file round keeps the run's draw file (`drawOk`) and whose session the blobs show (`acc_of_pass`):

* **the run walks to the session** (`reach`): the draw rounds answer their coins, and the draw file round, keeping the
  draw file, answers `[]`, so the run goes past both;
* **v2's acceptance is its one session's verdict** (`verdict_of_accepts`): one session (`oneSession`), its record on the
  blobs, and `verdictOf` at `A`'s inputs (`inputsAt`) and the verifier's registered reads, `.ok`;
* **the session's statement reads as v2's inputs** (`ins_sess`, `setupOf_sess`): flock-verify's reading of the session's
  statement at the draw file holds v2's inputs (`SessionMatch`) and sets up where v2's `setupAt` does;
* **the run's last answer is flock-verify's acceptance** (`decide_of_verdictOf`), at the session's messages and coins.

What the record takes beside v2's acceptance:
* `Names`: v2 reads `A`'s files and `o`'s objects;
* `Shows`: the evidence's session record draws the run's draw file, and past the draw file round the run's session is
  the evidence's;
* `LiveCoins`: live-verifier at the draw (v2's coins are the draw rounds' coins' first bytes) and at the session, as at
  the recursive session (`Verity.Recursive.LiveCoins`);
* `drawOk` at the run's draw file: its units are the coins' draw, its session has the statement's numbers, and its
  statement fits its 8-byte length words (`Fits`). v2 refuses a file or a draw its statement can't hold (`fitsWhy`,
  `drawFitsWhy`), which bounds every length but three (`lengths_of_accepts`, `fits_of_accepts`): the run's draw file
  and `o`'s partition object and registered file, which v2 reads only as the JSON they parse to.
-/

@[expose] public section

namespace Verity.Sampled.Record

open Lean (Json)
open Claims (Msg Out Seen Prover Transcript stepWith run coinSeq)
open FlockDeploy (Coin Final Ins)
open Verity.Recursive.BaseSetup (bind_ok)

private theorem ok_bind' {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {a : α} {b : β} (hx : x = .ok a)
    (hf : f a = .ok b) : (x >>= f) = .ok b := by
  rw [hx]
  exact hf

/-- A blob's bytes don't depend on the name a refusal would give it. -/
theorem get_what {bs : Core.Blobs} {d w w' : String} {b : ByteArray} (h : bs.get d w = .ok b) :
    bs.get d w' = .ok b := by
  unfold Core.Blobs.get at h ⊢
  obtain ⟨a, hd, h⟩ := bind_ok h
  refine ok_bind' hd ?_
  try dsimp only at h ⊢
  by_cases hc : (a.size != 64 || Flock.hexOf a != d) = true
  · rw [ite_eq_left hc] at h
    cases h
  · rw [ite_eq_right hc] at h ⊢
    cases hf : bs.entries.find? (·.1 == a) with
    | none =>
      rw [hf] at h
      cases h
    | some e =>
      rw [hf] at h
      obtain ⟨e1, e2⟩ := e
      exact h

private theorem mapM_ok_of_getElem {α β ε : Type} (g : α → Except ε β) :
    ∀ (l : List α) (l' : List β), l.length = l'.length →
      (∀ i (hi : i < l.length) (hi' : i < l'.length), g l[i] = .ok l'[i]) → l.mapM g = .ok l'
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => absurd hl (by simp)
  | _ :: _, [], hl, _ => absurd hl (by simp)
  | a :: l, b :: l', hl, hg => by
    have h0 : g a = .ok b := hg 0 (by simp) (by simp)
    have ht : l.mapM g = .ok l' := mapM_ok_of_getElem g l l' (by simpa using hl)
      (fun i hi hi' => by
        have h := hg (i + 1) (by simp [hi]) (by simp [hi'])
        rwa [List.getElem_cons_succ, List.getElem_cons_succ] at h)
    simp [List.mapM_cons, h0, ht, bind, Except.bind, pure, Except.pure]

/-! ## What the record takes -/

/-- **The run's draw file**: the prover's message at the draw file round. -/
def drawOf (v : Seen) : Msg := (v.msgs.map Prod.fst).getD DRAW_ROUNDS []

/-- **v2's inputs at `A`** (`Spot.OneStageZk.inputsOf`): `A`'s statement tags and committed files, the
parameters' own partition object and program, and the lookup tables the blobs hold. -/
def inputsAt (A : Live) (p : _root_.Spot.OneStageZk.Params) (bs : Core.Blobs) : Flock.Witness.Inputs :=
  _root_.Spot.OneStageZk.inputsOf A.I.tags p.own A.I.circuitFile A.I.publicFile bs

/-- **v2 reads `A`'s and `o`'s inputs** at the parameters `p` and the blobs `bs`: the committed circuit and public
files are `A`'s, so are the lookup tables the blobs hold and the program; `o`'s partition object and registered file
read (`FlockDeploy.jsonOf`) as the parameters' own; and the statement's tags are `A`'s. -/
structure Names (A : Live) (o : Own) (p : _root_.Spot.OneStageZk.Params) (bs : Core.Blobs) : Prop where
  circuit : bs.get p.own.circuit "circuit" = .ok A.I.circuitFile
  pub : bs.get p.own.pub "public file" = .ok A.I.publicFile
  tables : _root_.Spot.OneStageZk.tablesOf bs = A.I.tables
  program : A.I.program = some p.own.program
  partition : FlockDeploy.jsonOf "partition" o.partition = .ok p.own.partition
  registered : o.registered.mapM (FlockDeploy.jsonOf "registered reads") = .ok p.own.registered
  tags : Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some A.I.tags

/-- **The blobs show the run's session.** Where the run reaches the draw file round, at each session `s` the evidence
names, `s`'s record is on the blobs and its draw is the run's draw file, as the session's statement reads it
(`FlockDeploy.jsonOf`, `Flock.norm`). Where the run goes past that round, at each such `s`: the session's first message
reads as the link message, and its last `f` as the prover's last, at the session's statement at the run's draw file;
`s`'s record is `f`'s; and `s` names each rep stream's file of the session's setup, the proof file `f` carries in its
place. -/
def Shows (A : Live) (o : Own) (ev : _root_.Spot.OneStageZk.Evidence) (bs : Core.Blobs) (v : Seen) : Prop :=
  (DRAW_ROUNDS < v.msgs.length → ∀ s ∈ ev.sessions, ∃ rec : String,
    bs.get s.record "a session's record" = .ok rec.toUTF8 ∧
    ∃ J, FlockDeploy.jsonOf "draw" ⟨(drawOf v).toArray⟩ = .ok J ∧ Flock.Witness.recordDraw rec = some (Flock.norm J)) ∧
  (DRAW_ROUNDS + 1 < v.msgs.length → ∀ s ∈ ev.sessions,
    (∃ m link, v.msgs[DRAW_ROUNDS + 1]? = some m ∧ FlockDeploy.linkOf m.1 = some link) ∧
    ∃ f, v.msgs.getLast?.bind (fun m => FlockDeploy.finalOf m.1) = some f ∧ f.stmt = sessStmt A o (drawOf v) ∧
      bs.get s.record "a session's record" = .ok f.record.toUTF8 ∧
      ∀ si spec sts, FlockDeploy.setupOf true (sessStmt A o (drawOf v)) = some (si, spec, sts) →
        f.proofs.size = spec.repStreams.size ∧ ∀ k (hk : k < spec.repStreams.size), ∃ d,
          (s.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
          ∃ hk' : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k])

/-- **live-verifier** at the draw and at the session. At the draw: the bytes `ξ` v2 draws from, the challenger's coins,
are the first bytes of the draw rounds' coins (`drawBytes`). At the session: wherever flock-verify `--zk` decides on the
run's session (its messages after the draw file round, at the coins `cs` after it), the live session the record shows is
the session's link message, first coin and round messages with their coins (`FlockDeploy.custody`), as at the recursive
session (`Verity.Recursive.LiveCoins`). -/
def LiveCoins (A : Live) (o : Own) (ξ : ByteArray) (v : Seen) (cs : List Coin) : Prop :=
  (∃ t, drawBytes (cs.take DRAW_ROUNDS) = ξ ++ t) ∧
  ∀ si spec sts link c₁ mids f,
    FlockDeploy.act true (sessStmt A o (drawOf v)) ((v.msgs.map Prod.fst).drop (DRAW_ROUNDS + 1))
        (cs.drop (DRAW_ROUNDS + 1)) = .decide si spec sts link c₁ mids f →
      (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()

/-! ## The run's walk -/

section Walk

variable {A : Live} {o : Own} {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (ω : Ω)
  (r : Fin ((V A o).k (xOf A o)) → Coin)

/-- The record's verifier's step at the record's statement, on the prover's messages so far. -/
noncomputable abbrev vnAt (A : Live) (o : Own) (r : Fin ((V A o).k (xOf A o)) → Coin) :
    List Msg → Claims.OC Out :=
  fun as => (V A o).next (xOf A o) as ((List.range as.length).map (coinSeq r))

theorem run_eq_stepWith :
    run H P (V A o) (xOf A o) ω r =
      stepWith H P (xOf A o) ω (vnAt A o r) ((V A o).k (xOf A o)) [] [] ⟨[], [], [], .rej⟩ := rfl

/-- **The run's answer at round `i`** is the record's verifier's step on the run's first `i + 1` messages and coins. -/
theorem run_answer {i : ℕ} (h : i < (run H P (V A o) (xOf A o) ω r).msgs.length) :
    ∃ a b, (run H P (V A o) (xOf A o) ω r).msgs[i]? = some (a, b) ∧
      b = ((nextAt A o (((run H P (V A o) (xOf A o) ω r).msgs.take (i + 1)).map Prod.fst)
        ((List.range (i + 1)).map (coinSeq r))).eval H).1 := by
  obtain ⟨new, hnew, hget⟩ := RecDeploy.run_get H P (V A o) (xOf A o) ω r i h
  refine ⟨_, _, hget, ?_⟩
  have htake : ((run H P (V A o) (xOf A o) ω r).msgs.take (i + 1)).map Prod.fst =
      new.map Prod.fst ++ [((P.next (xOf A o) (new.map RecDeploy.histOf) ω).eval H).1] := by
    rw [List.take_add_one, hget, ← hnew]
    simp only [Option.toList_some, List.map_append, List.map_cons, List.map_nil]
  rw [htake, V_next_xOf]

/-- The run's rounds are `DRAW_ROUNDS + 1 + kS A o`, at least one past the draw file round. -/
theorem k_gt : DRAW_ROUNDS + 1 < (V A o).k (xOf A o) := by
  rw [V_k_xOf]
  have : 1 ≤ kS A o := by
    unfold kS
    simp only [FlockDeploy.V]
    split <;> omega
  omega

theorem run_full {i : ℕ} {a m : Msg} (h : (run H P (V A o) (xOf A o) ω r).msgs[i]? = some (a, .msg m)) :
    i + 1 < (run H P (V A o) (xOf A o) ω r).msgs.length ∨
      (run H P (V A o) (xOf A o) ω r).msgs.length = (V A o).k (xOf A o) := by
  rw [run_eq_stepWith] at h ⊢
  have := Verity.Recursive.Schedule.stepWith_full H P (xOf A o) ω (vnAt A o r) ((V A o).k (xOf A o)) i [] []
    ⟨[], [], [], .rej⟩ a m (by simpa using h)
  simpa using this

theorem run_acc_out {i : ℕ} {a : Msg} (h : (run H P (V A o) (xOf A o) ω r).msgs[i]? = some (a, .acc)) :
    (run H P (V A o) (xOf A o) ω r).out = .acc := by
  rw [run_eq_stepWith] at h ⊢
  exact Verity.Recursive.Schedule.stepWith_acc_out H P (xOf A o) ω (vnAt A o r) ((V A o).k (xOf A o)) i [] []
    ⟨[], [], [], .rej⟩ a (by simpa using h)

/-- **The run reaches the draw file round**: the draw rounds answer their coins, so the prover sends a draw file. -/
theorem reach_draw : DRAW_ROUNDS < (run H P (V A o) (xOf A o) ω r).msgs.length := by
  have hk := k_gt (A := A) (o := o)
  have hpos : 0 < (run H P (V A o) (xOf A o) ω r).msgs.length := by
    rw [run_eq_stepWith]
    obtain ⟨g, hg⟩ : ∃ g, (V A o).k (xOf A o) = g + 1 := ⟨(V A o).k (xOf A o) - 1, by omega⟩
    rw [hg]
    exact RecDeploy.stepWith_length_succ H P (xOf A o) ω (vnAt A o r) g [] [] ⟨[], [], [], .rej⟩
  by_contra hle
  obtain ⟨a, b, hab, hb⟩ := run_answer H P ω r
    (i := (run H P (V A o) (xOf A o) ω r).msgs.length - 1) (by omega)
  have hlen : ((((run H P (V A o) (xOf A o) ω r).msgs.take
      ((run H P (V A o) (xOf A o) ω r).msgs.length - 1 + 1)).map Prod.fst)).length ≤ DRAW_ROUNDS := by
    simp only [List.length_map, List.length_take]
    omega
  unfold nextAt at hb
  rw [ite_eq_left hlen] at hb
  simp only [Claims.OC.eval] at hb
  rw [hb] at hab
  rcases run_full H P ω r hab with h' | h' <;> omega

/-- **The run goes past the draw file round** where that round keeps the run's draw file: the draw rounds answer their
coins and the draw file round answers `[]`. -/
theorem reach (hpass : drawOk A o (drawOf (run H P (V A o) (xOf A o) ω r).seen) (coinsAt r)) :
    DRAW_ROUNDS + 1 < (run H P (V A o) (xOf A o) ω r).msgs.length := by
  have hk := k_gt (A := A) (o := o)
  have h64 := reach_draw H P ω r
  obtain ⟨a, b, hab, hb⟩ := run_answer H P ω r (i := DRAW_ROUNDS) h64
  have hl : (((run H P (V A o) (xOf A o) ω r).msgs.take (DRAW_ROUNDS + 1)).map Prod.fst).length =
      DRAW_ROUNDS + 1 := by
    simp only [List.length_map, List.length_take]
    omega
  have hnl : ¬ (((run H P (V A o) (xOf A o) ω r).msgs.take (DRAW_ROUNDS + 1)).map Prod.fst).length ≤
      DRAW_ROUNDS := by
    rw [hl]
    omega
  have hd : (((run H P (V A o) (xOf A o) ω r).msgs.take (DRAW_ROUNDS + 1)).map Prod.fst).getD DRAW_ROUNDS [] =
      drawOf (run H P (V A o) (xOf A o) ω r).seen := by
    unfold drawOf
    simp only [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_take, Transcript.seen,
      Nat.lt_add_one, ↓reduceIte]
  have hc : ((List.range (DRAW_ROUNDS + 1)).map (coinSeq r)).take DRAW_ROUNDS = coinsAt r := by
    unfold coinsAt
    rw [← List.map_take, List.take_range, Nat.min_eq_left (Nat.le_add_right DRAW_ROUNDS 1)]
  unfold nextAt at hb
  rw [ite_eq_right hnl, ite_eq_left hl, hd, hc, ite_eq_left hpass] at hb
  simp only [Claims.OC.eval] at hb
  rw [hb] at hab
  rcases run_full H P ω r hab with h' | h' <;> omega

end Walk

/-! ## v2's acceptance, at its one session -/

section Accepts

variable {A : Live} {o : Own} {p : _root_.Spot.OneStageZk.Params}
  {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {bs : Core.Blobs}
  {ξ : ByteArray} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}

private theorem if_throw_ok {b : Bool} {e : String} (h : (if b then (throw e : Flock.V Unit) else pure ()) = .ok ()) :
    b = false := by
  cases b with
  | false => rfl
  | true => cases h

/-- **v2's acceptance is its one session's verdict**: the evidence names one session `s`; the statement's tags are
`A`'s, with hm96 leaves; the verifier's registered reads are `rd`; `s`'s record is on the blobs, as UTF-8; and
`verdictOf` at `A`'s inputs on `s` set up at its record's draw is `.ok`. -/
theorem verdict_of_accepts (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof)
    (hn : Names A o p bs) :
    ∃ s rd B record ksFiles population, ev.sessions = #[s] ∧ A.I.tags.plainLeaves = false ∧
      _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd ∧
      bs.get s.record s!"session {s.name}'s record" = .ok B ∧ String.fromUTF8? B = some record ∧
      _root_.Spot.OneStageZk.verdictOf (inputsAt A p bs) (scopeOf (inputsAt A p bs) rd) ksFiles population
        #[] record (Flock.Witness.recordDraw record)
        (_root_.Spot.OneStageZk.setupAt (inputsAt A p bs) rd (Flock.Witness.recordDraw record))
        (_root_.Spot.OneStageZk.filesOf bs s
          (_root_.Spot.OneStageZk.setupAt (inputsAt A p bs) rd (Flock.Witness.recordDraw record)))
        (_root_.Spot.OneStageZk.coinKeyOf record
          (_root_.Spot.OneStageZk.setupAt (inputsAt A p bs) rd (Flock.Witness.recordDraw record))) =
        .ok () := by
  unfold _root_.Spot.OneStageZk.checkBlobsAt at hacc
  dsimp only at hacc
  split at hacc
  case h_1 => cases hacc
  rename_i R hR
  have hall : R.all = true := by
    cases hca : _root_.Spot.OneStage.checkAt p.base c (some ξ) { ev.base with verdicts := R.out } with
    | accept pb pf =>
      simp only [hca] at hacc
      cases hb : R.all
      · simp only [hb, Bool.false_eq_true, ↓reduceIte] at hacc
        cases hacc
      · rfl
    | _ =>
      simp only [hca] at hacc
      cases hacc
  obtain ⟨_, h1, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  have hsz : ev.sessions.size = 1 := by
    unfold _root_.Spot.OneStageZk.oneSession at h1
    by_contra hne
    have hb : ¬ (ev.sessions.size == 1) = true := by simpa using hne
    rw [ite_eq_right hb] at h1
    cases h1
  obtain ⟨s, hs⟩ : ∃ s, ev.sessions = #[s] := by
    rcases hl : ev.sessions with ⟨_ | ⟨s, _ | ⟨_, _⟩⟩⟩
    all_goals rw [hl] at hsz
    all_goals first | (simp at hsz; done) | exact ⟨s, rfl⟩
  unfold _root_.Spot.OneStageZk.run at hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  simp only [hn.tags] at hR
  obtain ⟨_, hpl, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨cB, hcB, hR⟩ := bind_ok hR
  obtain ⟨pB, hpB, hR⟩ := bind_ok hR
  obtain rfl : cB = A.I.circuitFile := Except.ok.inj (hcB.symm.trans hn.circuit)
  obtain rfl : pB = A.I.publicFile := Except.ok.inj (hpB.symm.trans hn.pub)
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨rd, hrd, hR⟩ := bind_ok hR
  obtain ⟨R', hfold, hR⟩ := bind_ok hR
  have hRR : R' = R := by
    split at hR
    all_goals first
      | exact Except.ok.inj hR
      | (obtain ⟨_, -, hR⟩ := bind_ok hR; exact Except.ok.inj hR)
  subst hRR
  rw [hs, ← Array.foldlM_toList] at hfold
  simp only [List.foldlM_cons, List.foldlM_nil] at hfold
  obtain ⟨R'', hstep, hpure⟩ := bind_ok hfold
  obtain rfl : R'' = R' := Except.ok.inj hpure
  unfold _root_.Spot.OneStageZk.step at hstep
  obtain ⟨B, hB, hstep⟩ := bind_ok hstep
  cases hu : String.fromUTF8? B with
  | none => rw [hu] at hstep; cases hstep
  | some record =>
    rw [hu] at hstep
    dsimp only at hstep
    obtain ⟨_, -, hstep⟩ := bind_ok hstep
    try dsimp only at hstep
    unfold _root_.Spot.OneStageZk.push at hstep
    dsimp only at hstep
    split at hstep
    · rename_i why hvd
      obtain rfl := Except.ok.inj hstep
      cases hall
    · rename_i hvd
      exact ⟨s, rd, B, record, _, _, hs, if_throw_ok hpl, hrd, hB, hu, hvd⟩

/-- **v2's acceptance bounds the lengths its length refusal reads** (`fitsWhy` at `WORD`): `A`'s circuit and public
files, its lookup tables (their number, names and bytes) and its program, and the printed text of the verifier's
partition object and registered reads (`printedSize`), each under `2^64`. -/
theorem lengths_of_accepts (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof)
    (hn : Names A o p bs) :
    A.I.circuitFile.size < 2 ^ 64 ∧ A.I.publicFile.size < 2 ^ 64 ∧ A.I.tables.length < 2 ^ 64 ∧
      (∀ t ∈ A.I.tables, t.1.toUTF8.size < 2 ^ 64 ∧ t.2.size < 2 ^ 64) ∧ (∀ b ∈ A.I.program, b.size < 2 ^ 64) ∧
      _root_.Spot.OneStageZk.printedSize p.own.partition < 2 ^ 64 ∧
      (∀ j ∈ p.own.registered, _root_.Spot.OneStageZk.printedSize j < 2 ^ 64) := by
  unfold _root_.Spot.OneStageZk.checkBlobsAt at hacc
  dsimp only at hacc
  split at hacc
  case h_1 => cases hacc
  rename_i R hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  unfold _root_.Spot.OneStageZk.run at hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  simp only [hn.tags] at hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨cB, hcB, hR⟩ := bind_ok hR
  obtain ⟨pB, hpB, hR⟩ := bind_ok hR
  obtain rfl : cB = A.I.circuitFile := Except.ok.inj (hcB.symm.trans hn.circuit)
  obtain rfl : pB = A.I.publicFile := Except.ok.inj (hpB.symm.trans hn.pub)
  obtain ⟨_, hfw, -⟩ := bind_ok hR
  split at hfw
  · cases hfw
  rename_i hf
  have hf' : _root_.Spot.OneStageZk.fitsWhy _root_.Spot.OneStageZk.WORD A.I.circuitFile
      A.I.publicFile A.I.tables A.I.program (some p.own.partition) p.own.registered = none := by
    rw [← hn.tables, hn.program]
    exact hf
  have hW : _root_.Spot.OneStageZk.WORD = 2 ^ 64 := rfl
  unfold _root_.Spot.OneStageZk.fitsWhy at hf'
  try rw [hW] at hf'
  split_ifs at hf' with h1 h2 h3 h4 h5 h6 h7
  refine ⟨Nat.lt_of_not_le h1, Nat.lt_of_not_le h2, Nat.lt_of_not_le h3, ?_, ?_, ?_, ?_⟩
  · intro t ht
    have hno : ¬ (decide (2 ^ 64 ≤ t.1.utf8ByteSize) || decide (2 ^ 64 ≤ t.2.size)) = true :=
      fun hx => h4 (List.any_eq_true.2 ⟨t, ht, hx⟩)
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, Nat.not_le] at hno
    exact hno
  · intro b hb
    rw [Option.mem_def] at hb
    rw [hb] at h5
    simpa using h5
  · simpa using h6
  · intro j hj
    rw [Option.mem_def] at hj
    rw [hj] at h7
    simpa using h7

/-- **The session's statement fits its length words** where v2 accepts reading `A`'s and `o`'s inputs (`Names`), and
the draw file `d` and the verifier's own partition object and registered file (`o`'s bytes) are each under `2^64`
bytes: v2's length refusal gives the rest (`lengths_of_accepts`). Where `o`'s bytes are the printed text of what v2
reads of them, `lengths_of_accepts` bounds those too. -/
theorem fits_of_accepts (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof)
    (hn : Names A o p bs) {d : Msg} (hd : d.length < 2 ^ 64) (hpart : o.partition.size < 2 ^ 64)
    (hreg : ∀ b ∈ o.registered, b.size < 2 ^ 64) : Fits A o d := by
  obtain ⟨hc, hp, htl, ht, hpr, -, -⟩ := lengths_of_accepts hacc hn
  refine ⟨?_, hc, hp, htl, ht, ?_, ?_, ?_, hpr, ?_, ?_, ?_, ?_, hreg, ?_⟩
  · show _root_.Spot.OneStageZk.STATEMENT.toUTF8.size < 2 ^ 64
    decide
  · show (1 : ℕ) < 2 ^ 64
    decide
  · intro b hb
    cases hb
    show d.toArray.size < 2 ^ 64
    rw [List.size_toArray]
    exact hd
  · intro b hb
    cases hb
    exact hpart
  · intro n hn
    cases hn
  · intro n hn
    cases hn
  · intro n hn
    cases hn
  · intro n hn
    cases hn
  · intro n hn
    cases hn

end Accepts

/-! ## The session's statement, as flock-verify reads it -/

section Ins

variable {A : Live} {o : Own} {p : _root_.Spot.OneStageZk.Params} {bs : Core.Blobs}

/-- The registered file reads as the verifier's registered reads `rd` (`readsOf`'s, on what `o`'s bytes read as). -/
theorem registered_of_names (hn : Names A o p bs) {rd : _root_.Spot.OneStageZk.Reads}
    (hrd : _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd) :
    o.registered.mapM (fun b => do Flock.Registered.ofJson (← FlockDeploy.jsonOf "registered reads" b)) =
      .ok rd.registered := by
  have hreg := hn.registered
  cases ho : o.registered with
  | none =>
    rw [ho] at hreg
    obtain hp : none = p.own.registered := Except.ok.inj hreg
    rw [← hp] at hrd
    obtain rfl := Except.ok.inj hrd
    rfl
  | some b =>
    rw [ho, Option.mapM_some] at hreg
    cases hj : FlockDeploy.jsonOf "registered reads" b with
    | error e => rw [hj] at hreg; cases hreg
    | ok j =>
      rw [hj] at hreg
      obtain hp : some j = p.own.registered := Except.ok.inj hreg
      rw [← hp] at hrd
      unfold _root_.Spot.OneStageZk.readsOf at hrd
      obtain ⟨own, hown, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain ⟨_, -, hrd⟩ := bind_ok hrd
      obtain rfl := Except.ok.inj hrd
      rw [Option.mapM_some, hj]
      simp only [bind, Except.bind, hown]
      rfl

/-- **flock-verify's reading of the session's statement holds v2's inputs**: at the draw file `d`, which reads as `J`,
`FlockDeploy.Raw.ins` gives a reading `si` that matches v2's inputs at `A` and the verifier's registered reads `rd`
(`SessionMatch`), with the draw `J`. -/
theorem ins_sess (hn : Names A o p bs) (hpl : A.I.tags.plainLeaves = false)
    {rd : _root_.Spot.OneStageZk.Reads}
    (hrd : _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd) {d : Msg} {J : Json}
    (hJ : FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J) :
    ∃ si, ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok si ∧
      SessionMatch si (inputsAt A p bs) rd ∧ si.read.draw = some (Flock.norm J) := by
  have hf : Flock.Tags.byName? _root_.Spot.OneStageZk.STATEMENT = some A.I.tags :=
    Flock.Tags.byName?_of_find? hn.tags
  have hdm : (some (⟨d.toArray⟩ : ByteArray)).mapM (FlockDeploy.jsonOf "draw") = .ok (some J) := by
    rw [Option.mapM_some, hJ]
    rfl
  have hpm : (some o.partition).mapM (FlockDeploy.jsonOf "partition") = .ok (some p.own.partition) := by
    rw [Option.mapM_some, hn.partition]
    rfl
  have hreg := registered_of_names hn hrd
  apply Exists.intro
  refine ⟨?hi, ?_, ?_⟩
  case hi =>
    unfold FlockDeploy.Raw.ins
    simp only [baseRaw, hf, hpl, hdm, hpm, hreg, Option.mapM_none, Bool.true_and, Bool.false_eq_true, ↓reduceIte]
    rfl
  · exact
      { zk := rfl
        stmtTags := rfl
        tags := rfl
        circuitFile := rfl
        publicFile := rfl
        count := rfl
        tables := hn.tables.symm
        partition := rfl
        program := hn.program
        work := rfl
        stratified := rfl
        closure := rfl
        publicInputs := rfl
        registered := rfl
        readRegistered := rfl }
  · rfl

private theorem setupOf_raw {w : FlockDeploy.Raw} (hfits : Verity.Recursive.Part.RawFits w) (hzk : w.zk = true)
    {si : Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (hins : w.ins = .ok si)
    (hsu : si.setup = .ok (spec, sts)) :
    FlockDeploy.setupOf true w.encode = some (si, spec, sts) := by
  unfold FlockDeploy.setupOf FlockDeploy.insOf
  rw [Verity.Recursive.Part.decode_encode_fits hfits]
  simp [hzk, hins, hsu, Except.toOption]

/-- **The session's statement sets up** where it fits its length words and flock-verify's reading of it sets up. -/
theorem setupOf_sess {d : Msg} (hfits : Fits A o d) {si : Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hins : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok si)
    (hsu : si.setup = .ok (spec, sts)) :
    FlockDeploy.setupOf true (sessStmt A o d) = some (si, spec, sts) :=
  setupOf_raw hfits rfl hins hsu

/-- **flock-verify reads the session's statement as `baseRaw` with the draw file**: where the session's statement at the
draw file `d` fits its length words, what flock-verify `--zk` reads of it (`FlockDeploy.readOf`) is what
`FlockDeploy.Raw.ins` reads of `baseRaw A o` with the draw `d`. -/
theorem readOf_sess {d : Msg} (hfits : Fits A o d) {r : Flock.Readable.Read}
    (h : FlockDeploy.readOf true (sessStmt A o d) = some r) :
    ∃ si, ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok si ∧ si.read = r := by
  unfold FlockDeploy.readOf FlockDeploy.insOf sessStmt at h
  rw [Verity.Recursive.Part.decode_encode_fits hfits] at h
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  obtain ⟨w, hw, hs⟩ := Option.bind_eq_some_iff.1 hs
  obtain rfl := Option.some.inj hw
  split_ifs at hs with hz
  · cases hs
  change ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins.toOption = some s at hs
  cases hi : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins with
  | error e => rw [hi] at hs; cases hs
  | ok s' =>
    rw [hi] at hs
    cases hs
    exact ⟨_, rfl, rfl⟩

/-- **flock-verify reads the session's statement as v2's inputs**: where the session's statement at the draw file `d`
fits its length words and flock-verify `--zk` reads it as `r`, `r` is `ins_sess`'s reading: v2's inputs at `A`
(`SessionMatch`), its draw the draw file's JSON. -/
theorem read_sess (hn : Names A o p bs) (hpl : A.I.tags.plainLeaves = false)
    {rd : _root_.Spot.OneStageZk.Reads}
    (hrd : _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd) {d : Msg} (hfits : Fits A o d)
    {r : Flock.Readable.Read} (h : FlockDeploy.readOf true (sessStmt A o d) = some r) :
    ∃ si J, si.read = r ∧ SessionMatch si (inputsAt A p bs) rd ∧
      FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J ∧ r.draw = some (Flock.norm J) := by
  obtain ⟨si, hsi, rfl⟩ := readOf_sess hfits h
  obtain ⟨draw, hdm, -⟩ := FlockDeploy.Raw.ins_draw hsi
  change (some (⟨d.toArray⟩ : ByteArray)).mapM (FlockDeploy.jsonOf "draw") = .ok draw at hdm
  rw [Option.mapM_some] at hdm
  cases hJ : FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ with
  | error e => rw [hJ] at hdm; cases hdm
  | ok J =>
    obtain ⟨si', hsi', hm, hdr⟩ := ins_sess hn hpl hrd hJ
    rw [hsi] at hsi'
    obtain rfl := Except.ok.inj hsi'
    exact ⟨si, J, rfl, hm, rfl, hdr⟩

end Ins

/-- **flock-verify decides on a session** whose statement sets up, whose last message reads as the prover's last and
whose first reads as the link message. -/
theorem act_decide_of {x : Msg} {s : Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (a₁ : Msg) (as : List Msg) (c₁ : Coin) (cs : List Coin)
    {a : Msg} {f : Final} {link : List ByteArray × List ByteArray} (hl : (a₁ :: as).getLast? = some a)
    (hf : FlockDeploy.finalOf a = some f) (hk : FlockDeploy.linkOf a₁ = some link) :
    FlockDeploy.act true x (a₁ :: as) (c₁ :: cs) =
      .decide s spec sts link c₁ ((as.dropLast.map List.tail).zip cs.dropLast) f := by
  have ha' : ∀ t, a ≠ 0 :: t := by
    intro t ht
    subst ht
    simp [FlockDeploy.finalOf] at hf
  obtain ⟨c, hc⟩ : ∃ c, (c₁ :: cs).getLast? = some c := ⟨_, List.getLast?_eq_some_getLast (List.cons_ne_nil _ _)⟩
  unfold FlockDeploy.act
  rw [hs]
  dsimp only
  rw [hl, hc]
  split
  · rename_i heq _
    exact absurd (Option.some.inj heq) (ha' _)
  · rename_i a₂ heq _
    obtain rfl := Option.some.inj heq
    simp only [hf, hk]
  · rename_i heq
    cases heq

/-- **The session's rep files are its last message's proof files** where each stream's file is that proof file and
there are as many proof files as streams. -/
theorem filesOf_eq {bs : Core.Blobs} {s : _root_.Spot.OneStageZk.RunFiles}
    {st : _root_.Spot.OneStageZk.SetUp} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {pj oj : Json}
    (hst : st = .ok (spec, tabs, pj, oj)) {proofs : Array ByteArray}
    (hreps : ∀ k (hk : k < spec.repStreams.size), ∃ d,
      (s.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < proofs.size, bs.get d "session files" = .ok proofs[k])
    (hnp : proofs.size = spec.repStreams.size) :
    _root_.Spot.OneStageZk.filesOf bs s st = .ok proofs := by
  unfold _root_.Spot.OneStageZk.filesOf
  rw [hst]
  show spec.repStreams.mapM _ = _
  rw [Array.mapM_eq_mapM_toList]
  rw [mapM_ok_of_getElem _ spec.repStreams.toList proofs.toList (by simp [hnp]) (fun k hk hk' => ?_)]
  · simp [Functor.map, Except.map]
  · have hk2 : k < spec.repStreams.size := by simpa using hk
    obtain ⟨d, hd, hk3, hget⟩ := hreps k hk2
    simp only [Array.getElem_toList]
    rcases hx : spec.repStreams[k] with ⟨a, b, w⟩
    rw [hx] at hd
    obtain ⟨⟨n', d'⟩, hfind, hdd⟩ := Option.map_eq_some_iff.1 hd
    simp only at hdd
    subst hdd
    simp only [hfind]
    exact get_what hget

/-- `setupAt` is `.ok` where `verdictOf` on it is. -/
theorem setupAt_of_verdictOf {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit}
    {coinKeys : Array String} {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) :
    ∃ spec tabs pj oj, st = .ok (spec, tabs, pj, oj) := by
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨⟨spec, tabs, pj, oj⟩, hst, -⟩ := bind_ok h
  exact ⟨spec, tabs, pj, oj, hst⟩

/-! ## v2's acceptance is the record's -/

/-- **v2's acceptance is the record's verifier's**, at a run whose draw file round keeps its draw file: where
one-stage v2 accepts on the blobs `bs` reading `A`'s and `o`'s inputs (`Names`), the run of the record's verifier
against `sha512` accepts, given that the blobs show the run's session (`Shows`), live-verifier at it (`LiveCoins`), and
the draw file round keeps the run's draw file (`drawOk`, whose statement then fits its length words). -/
theorem acc_of_pass (A : Live) (o : Own) {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {bs : Core.Blobs}
    {ξ : ByteArray} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof) (hn : Names A o p bs)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V A o).k (xOf A o)) → Coin)
    (hshow : Shows A o ev bs (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen)
    (hlive : LiveCoins A o ξ (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen
      ((List.range (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length).map (coinSeq r)))
    (hpass : drawOk A o (drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen) (coinsAt r)) :
    (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).out = .acc := by
  have hfits := fits_of_drawOk hpass
  obtain ⟨s, rd, B, record, ksFiles, population, hs, hpl, hrd, hB, hu, hv⟩ := verdict_of_accepts hacc hn
  have hreach := reach FlockDeploy.sha512 P ω r hpass
  have hsm : (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen.msgs =
      (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs := by
    simp only [Transcript.seen]
  have hmem : s ∈ ev.sessions := by rw [hs]; exact Array.mem_singleton_self s
  obtain ⟨⟨m, link, hm1, hlink⟩, f, hfin, hstmt, hrecB, hfiles⟩ := hshow.2 (by rw [hsm]; exact hreach) s hmem
  obtain ⟨rec, hrecD, J, hJ, hrJ⟩ := hshow.1 (by rw [hsm]; omega) s hmem
  rw [hsm] at hm1 hfin
  have hrec' : rec = record := by
    have h2 := congrArg String.fromUTF8?
      (Except.ok.inj ((get_what (w' := s!"session {s.name}'s record") hrecD).symm.trans hB))
    rw [FlockComplete.fromUTF8?_toUTF8, hu] at h2
    exact Option.some.inj h2
  have hrec : f.record = record := by
    have h2 := congrArg String.fromUTF8? (Except.ok.inj (hrecB.symm.trans hrecD))
    rw [FlockComplete.fromUTF8?_toUTF8, FlockComplete.fromUTF8?_toUTF8] at h2
    exact (Option.some.inj h2).trans hrec'
  obtain ⟨si, hins, hm, hsd⟩ := ins_sess hn hpl hrd hJ
  have hdraw : si.read.draw = Flock.Witness.recordDraw record := by
    rw [hsd, ← hrec', hrJ]
  obtain ⟨spec, tabs, pj, oj, hst⟩ := setupAt_of_verdictOf hv
  obtain ⟨sts, hsu, -⟩ := setup_of_setupAt hm hdraw hst
  have hso := setupOf_sess hfits hins hsu
  obtain ⟨hnp, hreps⟩ := hfiles si spec sts hso
  have hfl := filesOf_eq (bs := bs) (s := s) hst hreps hnp
  obtain ⟨mL, hmL, hfL⟩ := Option.bind_eq_some_iff.1 hfin
  have hlen1 : DRAW_ROUNDS + 1 < ((run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.map Prod.fst).length := by
    rw [List.length_map]
    exact hreach
  have hlenc : DRAW_ROUNDS + 1 <
      ((List.range (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length).map (coinSeq r)).length := by
    rw [List.length_map, List.length_range]
    exact hreach
  obtain ⟨hm1l, hm1'⟩ := List.getElem?_eq_some_iff.1 hm1
  obtain ⟨c₁, mids, hact⟩ : ∃ c₁ mids, FlockDeploy.act true
      (sessStmt A o (drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen))
      (((run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen.msgs.map Prod.fst).drop (DRAW_ROUNDS + 1))
      (((List.range (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length).map (coinSeq r)).drop
        (DRAW_ROUNDS + 1)) = .decide si spec sts link c₁ mids f := by
    rw [hsm, List.drop_eq_getElem_cons hlen1, List.drop_eq_getElem_cons hlenc]
    refine ⟨_, _, act_decide_of hso _ _ _ _ (a := mL.1) ?_ hfL ?_⟩
    · rw [← List.drop_eq_getElem_cons hlen1, List.getLast?_drop, ite_eq_right (Nat.not_le.2 hlen1),
        List.getLast?_map, hmL]
      rfl
    · rw [List.getElem_map, hm1']
      exact hlink
  have hdec := decide_of_verdictOf hv hact hm hdraw hstmt hrec hfl
    (hlive.2 si spec sts link c₁ mids f hact)
  have hdraw0 : drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen =
      ((run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.map Prod.fst).getD DRAW_ROUNDS [] := by
    unfold drawOf
    rw [hsm]
  rw [hsm] at hdec
  have hpos : 0 < (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length := by omega
  obtain ⟨a, b, hab, hb⟩ := run_answer FlockDeploy.sha512 P ω r
    (i := (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length - 1) (by omega)
  have hL : (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length - 1 + 1 =
      (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length := by omega
  rw [hL, List.take_length] at hb
  have h1 : ¬ ((run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.map Prod.fst).length ≤ DRAW_ROUNDS := by
    rw [List.length_map]
    omega
  have h2 : ¬ ((run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.map Prod.fst).length = DRAW_ROUNDS + 1 := by
    rw [List.length_map]
    omega
  unfold nextAt at hb
  rw [ite_eq_right h1, ite_eq_right h2, ← hdraw0, hdec] at hb
  rw [hb] at hab
  exact run_acc_out FlockDeploy.sha512 P ω r hab

end Verity.Sampled.Record
