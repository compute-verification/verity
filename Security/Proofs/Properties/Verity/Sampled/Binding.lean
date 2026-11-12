module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Properties.Challenger
public import Proofs.Core.Game.Tape
public import Proofs.Properties.Challenger.Break
public import Proofs.Properties.Challenger.Drawn
public import Proofs.Properties.Verity.Sampled.Trees
public import Proofs.Properties.Challenger.CanonLine
public import Proofs.Properties.Challenger.Limits
public import Proofs.Properties.Verity.Sampled.Lines
import all Init.Data.String.Defs

/-!
# The log the sampled check read is the device's, or a break: the read finder and the forger

The read finder's and the forger's terms of `Verity.Sampled.CouplingTarget`. On a play of the period's game whose
sampled verdict accepts, the device's log as v2's check read it is the play's, or the read finder outputs a SHA-512
collision, or the forger outputs a forgery of the device's LMS signature.

* **The log a check read** (`Read`, `Logged`): each release's frame and the bytes it answered, and its coins where the
  check's evidence shows them. v2's evidence shows the sample's segment's coins and each coin tree's root and key, and of
  a tree opening only its coins' SHA-512. `Logged reads` is `Vocab.Logged` at reads that show every release's coins
  (`vocab_logged_iff`), so it binds as much wherever the evidence shows the coins, and it can hold on an honest play
  where the evidence doesn't. v2's read (`readsV2`) is `Verity.Sampled.coinsRead`'s, each release's coins where v2
  read any, and `Vocab.Logged` at it gives `Logged readsV2` (`logged_v2_of_vocab`); the reduction takes any `reads`.
* **v2's records in what the prover sent** (`records`): the components' parameters and cases the composition decodes
  that decode as sampled proofs' one-stage form, v2's, under parameters naming a challenger and a window, each with what
  its coins are read under (`Properties.Challenger.oneStageZkSource`) and the record its evidence carries. The
  parameters' challenger key (`keyOf`) is that of their first component of sampled proofs' id that names one.
* **An accepted v2 case is one of them** (`accepted_case`): a composite names each component's id once (`ids_nodup`), so
  on parameters naming the one-stage form, the case the sampled component accepts is v2's (`caseOf`), its record is one
  of `records` under `keyOf`'s key, `drawCoins` reads from it the coins v2's check read, and v2 held the sessions'
  records to the coin trees it read from it (`Verity.Sampled.Trees.accepted_trees`).
* **An accepted read is a run's log** (`accepted_run`): where v2's check reads a log, it is what `readLog` reads
  from the lines of a run of the device that the device's ledger opens withheld on the sample's window, whose release
  log reads as its coin commitment's, with the run's released coins as the sample's (`Challenger.Drawn.drawn_log`), or
  one of the records under `keyOf`'s key carries a break. The records the evidence names each pass the coin trees
  `coinTreesOf` reads from that run's log (`trees_of`, `records_of`).
* **The constructions** at a strategy, on a uniform tape of its play (`Verity.Game.Tape`), at the device's history at
  each play (`dev`). The read finder (`readFinder`) outputs the first collision `findBreak` finds in a record, else the
  first tree opening whose coins v2 read only as a digest and the play released otherwise
  (`Lines.openCollision`), and the forger (`forger`) the first forgery `findBreak` finds in a record under the
  parameters' key. Each outputs only two different strings with one digest (`findBreak_valid`, `Lines.open_collision`),
  or a signature that verifies over a message the device didn't sign (`Properties.Challenger.Signed`). The play releases
  the device's signatures at its close, so the forger had the device sign, on each tape, the messages the play's
  signatures are on (`forger`'s `signed`), at most `Q` of them (`forger_signed_le`, from `Vocab.signs_le`). Each runs
  the play once and `findBreak` once per record: its SHA-512 calls are the strategy's, one per string `findBreak`
  hashes and `lmsCalls` per signature it verifies
  (`breakCalls`), and the read finder's one more per blob the prover sent, as v2's read indexes them
  (`Core.Blobs.ofArray`).
  `withBinding` puts them in a `Constructions`.
* **The records are read at the deployed key, within `LIMITS`** (`keyOf_deployed`, `breakCalls_le`): v2 refuses a
  challenger key off `Lms.deployed` and coins past `LIMITS` (`Properties.Challenger.Limits`), so the parameters' key
  is at `lmsVerify`'s parameters and `findBreak` spends at most `recordCalls` per record beside the strings the device
  hashes.
* **The reduction** (`logged_le`). Suppose every accepted run that isn't `Logged` carries a break under the
  parameters' key in one of its records, or a tree opening whose read coins collide with the released ones (`Binds`),
  and the device's history at each play signed under that key every message the play's signatures are on
  (`SignsPlay`), so a message it didn't sign is none the forger had signed. Then an accepted play that isn't `Logged` is
  at most as likely as the read finder's collision plus the forger's forgery.
* **Where the device printed its log** (`binds_of_prints`): if at every play under the parameters the device's runs
  print the play's log as `readLog` reads it, with the play's fresh coins as the runs' released coins wherever a
  run's release log reads as its coin commitment's (`Lines.PrintsLog`), `Binds` holds: an accepted read is the log's
  (`Lines.read_run`), so where it isn't `Logged` an opening's read coins differ from the released ones under one
  digest (`Lines.open_collision`).
-/

@[expose] public section

namespace Verity.Sampled.Binding

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Trees (ok_bind)
open scoped ENNReal

local instance lawfulBEqByteArrayBinding : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## The log a check read -/

/-- **One release as a check read it**: its frame, the bytes it answered, and its coins where the check's evidence
shows them (`none` where it shows only their digest, or nothing of them). -/
structure Read where
  frame : Frame
  answers : ByteArray
  coins : Option (List Bool)

/-- The read is of the release `x`: its frame and the bytes it answered, and its coins where it shows them. -/
def Read.Of (r : Read) (x : Release) : Prop :=
  r.frame = x.frame ∧ r.answers = x.answers ∧ ∀ cs ∈ r.coins, cs = x.coins

/-- The read that shows every coin of the release `x`. -/
def Read.ofRelease (x : Release) : Read := ⟨x.frame, x.answers, some x.coins⟩

/-- **The log the check read is the device's**: each log `reads` decodes from what the prover sent reads the device's
releases, every one in order and no other. -/
def Logged (reads : Sent → Option (List Read)) (o : Exchange) : Prop :=
  ∀ log ∈ reads o.sent, List.Forall₂ Read.Of log o.log

theorem ofRelease_of_iff (x y : Release) : (Read.ofRelease x).Of y ↔ x = y := by
  cases x; cases y
  simp [Read.Of, Read.ofRelease, Option.mem_def]

theorem forall₂_ofRelease_iff : ∀ l l' : List Release, List.Forall₂ Read.Of (l.map Read.ofRelease) l' ↔ l = l'
  | [], [] => ⟨fun _ => rfl, fun _ => .nil⟩
  | [], _ :: _ => ⟨nofun, nofun⟩
  | _ :: _, [] => ⟨nofun, nofun⟩
  | x :: xs, y :: ys => by
    constructor
    · intro h
      cases h with
      | cons hxy hs => rw [(ofRelease_of_iff x y).1 hxy, (forall₂_ofRelease_iff xs ys).1 hs]
    · intro h
      cases h
      exact .cons ((ofRelease_of_iff _ _).2 rfl) ((forall₂_ofRelease_iff _ _).2 rfl)

/-- **`Vocab.Logged` is `Logged` at reads that show every coin.** -/
theorem vocab_logged_iff {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    (a : Subprotocol) (o : Exchange) :
    V.Logged a o ↔ Logged (fun s => (V.coinsRead a s).map (·.map Read.ofRelease)) o := by
  unfold Vocab.Logged Logged
  constructor
  · intro h log hlog
    obtain ⟨l, hl, rfl⟩ := Option.map_eq_some_iff.1 (Option.mem_def.1 hlog)
    exact (forall₂_ofRelease_iff l o.log).2 (h l (Option.mem_def.2 hl))
  · intro h l hl
    exact (forall₂_ofRelease_iff l o.log).1 (h _ (Option.mem_map_of_mem _ hl))

/-- **A release as v2's check reads it** (`Verity.Sampled.coinsRead`): its frame, the bytes it answered, and its coins
where it read them, a release whose coins it read none of having `[]`. -/
def Read.ofV2 (x : Release) : Read := ⟨x.frame, x.answers, if x.coins.isEmpty then none else some x.coins⟩

/-- **The log v2's check read** from what the prover sent (`Verity.Sampled.coinsRead`). -/
def readsV2 (s : Sent) : Option (List Read) := (Verity.Sampled.coinsRead s).map (·.map Read.ofV2)

theorem ofV2_of (x : Release) : (Read.ofV2 x).Of x := by
  refine ⟨rfl, rfl, fun cs hcs => ?_⟩
  change cs ∈ (if x.coins.isEmpty then none else some x.coins) at hcs
  split at hcs
  · cases Option.mem_def.1 hcs
  · exact (Option.some.inj (Option.mem_def.1 hcs)).symm

theorem forall₂_ofV2 : ∀ l : List Release, List.Forall₂ Read.Of (l.map Read.ofV2) l
  | [] => .nil
  | x :: l => .cons (ofV2_of x) (forall₂_ofV2 l)

/-- **`Vocab.Logged` at v2's read gives `Logged readsV2`**: where the vocabulary's log at `.sampled` is v2's
(`Deployed.coinsRead`), a run whose whole log the check read is the device's has the log v2 read the device's. -/
theorem logged_v2_of_vocab {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] {V : Vocab d α}
    (hV : ∀ s, V.coinsRead .sampled s = Verity.Sampled.coinsRead s) {o : Exchange} (h : V.Logged .sampled o) :
    Logged readsV2 o := by
  intro log hlog
  obtain ⟨l, hl, rfl⟩ := Option.map_eq_some_iff.1 (Option.mem_def.1 hlog)
  obtain rfl : l = o.log := h l (by rw [hV]; exact Option.mem_def.2 hl)
  exact forall₂_ofV2 _

/-! ## v2's records in what the prover sent -/

/-- **A component's parameters and case as a v2 record**: where they decode as sampled proofs' one-stage form, v2's,
under parameters naming a challenger and a window, what its coins are read under (`oneStageZkSource`) and the record its
evidence carries. -/
def record? (n : Core.Composition.Named) (x : Core.Composition.Case) : Option (Draws × Json) :=
  match Core.canonicalJson n.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson,
      Spot.OneStage.commitmentOf x.commitment, VStar.Recursive.Sampled.evidenceOf x.evidence with
  | .ok (.inl p), .ok c, .ok (.inl e) => (oneStageZkSource p c).map fun d => (d, e.base.coins)
  | _, _, _ => none

/-- **v2's records in what the prover sent**: the components' parameters and cases the composition decodes, each read
as v2's (`record?`). -/
def records (s : Sent) : List (Draws × Json) :=
  match Core.canonicalJson s.params >>= Core.Composition.decodeParams, Core.Composition.decodeEvidence s.evidence with
  | .ok ps, .ok ev => (ps.zip ev).filterMap fun x => record? x.1 x.2
  | _, _ => []

/-- A component's challenger key: where it is sampled proofs' and its parameters name the one-stage form and a
challenger, that challenger's key. -/
def keyAt (n : Core.Composition.Named) : Option ChallengerKey :=
  if n.id = VStar.Recursive.Sampled.PROTOCOL then
    match Core.canonicalJson n.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson with
    | .ok (.inl p) => p.base.challenger.map (·.1)
    | _ => none
  else none

/-- **The parameters' challenger key**: that of their first component of sampled proofs' id whose parameters name the
one-stage form and a challenger (`keyAt`). -/
def keyOf (θ : ByteArray) : Option ChallengerKey :=
  match Core.canonicalJson θ >>= Core.Composition.decodeParams with
  | .ok ps => ps.findSome? keyAt
  | .error _ => none

/-! ## An accepted v2 case is one of the records, under the parameters' key -/

/-- In a composition whose parameters and evidence name its components' ids in order, each case's parameters and
evidence name its component's id. -/
theorem triples_ids : ∀ {cs : List _root_.Core.Protocol} {ps : List Core.Composition.Named}
    {ev : List Core.Composition.Case}, ps.map (·.id) = cs.map (·.id) → ev.map (·.id) = cs.map (·.id) →
    ∀ x ∈ Properties.Composition.triples cs ps ev, x.2.1.id = x.1.id ∧ x.2.2.id = x.1.id
  | [], _, _, _, _, x, hx => by simp [Properties.Composition.triples] at hx
  | _ :: _, [], _, h, _, _, _ => by simp at h
  | _ :: _, _ :: _, [], _, h, _, _ => by simp at h
  | c :: cs, p :: ps, e :: ev, hp, he, x, hx => by
    simp only [List.map_cons, List.cons.injEq] at hp he
    simp only [Properties.Composition.triples, List.zip_cons_cons, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact ⟨hp.1, he.1⟩
    · exact triples_ids (cs := cs) (ps := ps) (ev := ev) hp.2 he.2 x hx

/-- Each composite names each of its components' ids once. -/
theorem ids_nodup (k : Verity.Registration.Composite) : ((componentsOf k).map (·.id)).Nodup := by
  cases k <;> decide

/-- `findSome?` at the one element of a list where the function is defined. -/
theorem findSome_eq {β γ : Type} {f : β → Option γ} :
    ∀ {l : List β} {a : β} {b : γ}, a ∈ l → f a = some b → (∀ a' ∈ l, (f a').isSome → a' = a) →
      l.findSome? f = some b
  | [], _, _, ha, _, _ => by cases ha
  | a' :: l, a, b, ha, hf, hu => by
    cases hfa : f a' with
    | some b' =>
      obtain rfl : a' = a := hu a' (List.mem_cons.2 (.inl rfl)) (by simp [hfa])
      rw [List.findSome?_cons_of_isSome (by simp [hfa])]
      exact hf
    | none =>
      rw [List.findSome?_cons_of_isNone (by simp [hfa])]
      refine findSome_eq ?_ hf fun a'' h => hu a'' (List.mem_cons_of_mem _ h)
      rcases List.mem_cons.1 ha with rfl | h
      · rw [hf] at hfa
        cases hfa
      · exact h

/-- `find?` at the one element of a list that satisfies the test. -/
theorem find_eq {β : Type} {q : β → Bool} : ∀ {l : List β} {a : β}, a ∈ l → q a = true →
    (∀ a' ∈ l, q a' = true → a' = a) → l.find? q = some a
  | [], _, ha, _, _ => by cases ha
  | a' :: l, a, ha, hq, hu => by
    by_cases hq' : q a' = true
    · rw [List.find?_cons_of_pos hq', hu a' (List.mem_cons.2 (.inl rfl)) hq']
    · rw [List.find?_cons_of_neg hq']
      refine find_eq ?_ hq fun a'' h => hu a'' (List.mem_cons_of_mem _ h)
      rcases List.mem_cons.1 ha with rfl | h
      · exact absurd hq hq'
      · exact h

/-- The challenger's key is the key of the samples v2's coins are read under. -/
theorem key_of_source {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment} {d : Draws}
    (h : oneStageZkSource p c = some d) : p.base.challenger.map (·.1) = some d.key := by
  unfold oneStageZkSource oneStageSource at h
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h <;> cases h
  simp

/-- **An accepted v2 case is one of the records, under the parameters' key** (P1): at a composite that reads sampled
proofs at v2 (`ReadsV2`), on parameters that name its one-stage form (`ParamsAt`), a run accepted as a `same`
composition whose sampled component accepts its own case has that case as v2's (`caseOf`), read under a challenger
and a window its parameters name (`oneStageZkSource`); its record is one of the run's v2 records (`records`) under
the parameters' challenger key (`keyOf`), v2's coins are what `drawCoins` reads from it, and v2 read the challenger's
coin trees from it and held its sessions' records to them (`treesOf`, `treesHeld`). -/
theorem accepted_case {k : Verity.Registration.Composite} (hk : ReadsV2 k) {s : Sent}
    {p : Spot.OneStageZk.Params} (hp : ParamsAt s.params p) (hs : SameAccepts k s)
    (ha : ComponentAccepts k .sampled s) :
    ∃ c e d ξ, caseOf s = .ok (p, c, e) ∧ oneStageZkSource p c = some d ∧ (d, e.base.coins) ∈ records s ∧
      keyOf s.params = some d.key ∧ Spot.OneStageZk.coinsOf p c e = .ok ξ ∧
      drawCoins d.key d.window d.label d.after e.base.coins = .ok ξ ∧
      ∃ ts, Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧
        Spot.OneStageZk.treesHeld p.own ts e (_root_.Core.Blobs.ofArray s.blobs) = .ok () := by
  obtain ⟨P, hP, ps, b, ev, ⟨hps, -, hev⟩, x, hx, hx1, hacc⟩ := ha
  obtain rfl : P = VStar.Recursive.Sampled.protocol := Option.some.inj (hP.symm.trans hk)
  obtain ⟨ps', b', ev', ⟨hps', -, hev'⟩, -, -, ⟨⟨⟨hpi, hei, -, -⟩, -, -⟩, -⟩, -⟩ := hs
  have e1 : ps' = ps := Except.ok.inj (hps'.symm.trans hps)
  have e2 : ev' = ev := Except.ok.inj (hev'.symm.trans hev)
  subst ps' ev'
  obtain ⟨ps₀, hps₀, n, hnf, hnp⟩ := hp
  have hn : n ∈ ps₀ := List.mem_of_find?_eq_some hnf
  have hnq := List.find?_some hnf
  have hnid : n.id = VStar.Recursive.Sampled.PROTOCOL := beq_iff_eq.1 hnq
  have e3 : ps₀ = ps := Except.ok.inj (hps₀.symm.trans hps)
  subst ps₀
  have hcs : (s.components k).map (·.id) = (componentsOf k).map (·.id) := by
    unfold Sent.components
    rw [List.map_map]
    rfl
  have hnd : (ps.map (·.id)).Nodup := by rw [hpi, hcs]; exact ids_nodup k
  have hnde : (ev.map (·.id)).Nodup := by rw [hei, hcs]; exact ids_nodup k
  obtain ⟨hid1, hid2⟩ := triples_ids hpi hei x hx
  rw [hx1] at hid1 hid2 hacc
  have hid1' : x.2.1.id = VStar.Recursive.Sampled.PROTOCOL := hid1
  have hid2' : x.2.2.id = VStar.Recursive.Sampled.PROTOCOL := hid2
  have hmem := List.of_mem_zip (a := x.1) (b := x.2) hx
  have hmem2 := List.of_mem_zip (a := x.2.1) (b := x.2.2) hmem.2
  obtain rfl : n = x.2.1 := List.inj_on_of_nodup_map hnd hn hmem2.1 (hnid.trans hid1'.symm)
  obtain ⟨p', c, e', hp', hc, he, pub, prof, hch⟩ := hacc
  obtain rfl : p' = Sum.inl p := Except.ok.inj (hp'.symm.trans hnp)
  rcases e' with e | e
  swap
  · cases hch
  have hc' : Spot.OneStage.commitmentOf x.2.2.commitment = .ok c := hc
  have he' : VStar.Recursive.Sampled.evidenceOf x.2.2.evidence = .ok (Sum.inl e) := he
  have hch' : Spot.OneStageZk.checkBlobs p c e (_root_.Core.Blobs.ofArray s.blobs) = .accept pub prof := hch
  cases hξ : Spot.OneStageZk.coinsOf p c e with
  | error r =>
    unfold Spot.OneStageZk.checkBlobs at hch'
    rw [hξ] at hch'
    cases hch'
  | ok ξ =>
    obtain ⟨d, hd⟩ := oneStageZk_source (p := p) (c := c) (e := e) hξ
    obtain ⟨raw, hraw, hrx⟩ := oneStageZk_reads p c d hd e ξ hξ
    subst raw
    have hrec : record? x.2.1 x.2.2 = some (d, e.base.coins) := by
      simp only [record?, hnp, hc', he', hd, Option.map_some]
    have hkeyAt : keyAt x.2.1 = some d.key := by
      unfold keyAt
      rw [ite_eq_left hid1', hnp]
      exact key_of_source hd
    have hf1 : ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x.2.1 :=
      find_eq hmem2.1 (beq_iff_eq.2 hid1') fun a' ha' h =>
        List.inj_on_of_nodup_map hnd ha' hmem2.1 ((beq_iff_eq.1 h).trans hid1'.symm)
    have hf2 : ev.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x.2.2 :=
      find_eq hmem2.2 (beq_iff_eq.2 hid2') fun a' ha' h =>
        List.inj_on_of_nodup_map hnde ha' hmem2.2 ((beq_iff_eq.1 h).trans hid2'.symm)
    refine ⟨c, e, d, ξ, ?_, hd, ?_, ?_, hξ, hraw, Verity.Sampled.Trees.accepted_trees hch'⟩
    · simp only [caseOf, hps, ok_bind, hf1, hev, hf2, hnp, hc', he']
      rfl
    · unfold records
      rw [hps, hev]
      exact List.mem_filterMap.2 ⟨x.2, hmem.2, hrec⟩
    · unfold keyOf
      rw [hps]
      refine findSome_eq hmem2.1 hkeyAt fun a' ha' hsome => ?_
      by_cases hid : a'.id = VStar.Recursive.Sampled.PROTOCOL
      · exact List.inj_on_of_nodup_map hnd ha' hmem2.1 (hid.trans hid1'.symm)
      · unfold keyAt at hsome
        rw [ite_eq_right hid] at hsome
        cases hsome

/-! ## An accepted read is a run's log -/

/-- v2's case names v2's parameters in the parameters sent. -/
theorem caseOf_params {s : Sent} {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment}
    {e : Spot.OneStageZk.Evidence} (h : caseOf s = .ok (p, c, e)) : ParamsAt s.params p := by
  unfold caseOf at h
  cases hps : (_root_.Core.canonicalJson s.params >>= Core.Composition.decodeParams) with
  | error _ => rw [hps] at h; cases h
  | ok ps =>
    rw [hps, ok_bind] at h
    cases hx : ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) with
    | none => rw [hx] at h; cases h
    | some x =>
      rw [hx] at h
      dsimp only at h
      cases hev : Core.Composition.decodeEvidence s.evidence with
      | error _ => rw [hev] at h; cases h
      | ok ev =>
        rw [hev, ok_bind] at h
        cases hy : ev.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) with
        | none => rw [hy] at h; cases h
        | some y =>
          rw [hy] at h
          dsimp only at h
          cases hnp : (_root_.Core.canonicalJson x.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) with
          | error _ => rw [hnp] at h; cases h
          | ok q =>
            rw [hnp, ok_bind] at h
            rcases q with p' | q
            · dsimp only at h
              cases hc : Spot.OneStage.commitmentOf y.commitment with
              | error _ => rw [hc] at h; cases h
              | ok c' =>
                rw [hc, ok_bind] at h
                cases he : VStar.Recursive.Sampled.evidenceOf y.evidence with
                | error _ => rw [he] at h; cases h
                | ok r =>
                  rw [he, ok_bind] at h
                  rcases r with e' | _
                  · obtain rfl : p' = p := congrArg Prod.fst (Except.ok.inj h)
                    exact ⟨ps, hps, x, hx, hnp⟩
                  · cases h
            · cases h

/-- A list of strings' JSON strings read back by a reader of strings is the list. -/
theorem mapM_map_str {f : Json → Flock.V String} (hf : ∀ x, f (.str x) = .ok x) :
    ∀ l : List String, (l.map Json.str).mapM f = .ok l
  | [] => rfl
  | x :: l => by
    rw [List.map_cons, List.mapM_cons, hf, mapM_map_str hf l]
    rfl

/-- v2's evidence's log, read as lines (`Verity.Sampled.linesOf`), is the lines the session's log holds. -/
theorem linesOf_eq {e : Spot.OneStageZk.Evidence} {ls : Array Json} {l : List String}
    (h : _root_.Spot.getD (_root_.Spot.getD e.base.coins "session") "log" = .arr ls)
    (hl : ls.toList = l.map Json.str) : Verity.Sampled.linesOf e = .ok l := by
  unfold Verity.Sampled.linesOf
  rw [h]
  show ls.toList.mapM _ = _
  rw [hl]
  exact mapM_map_str (fun _ => rfl) l

/-- A reader of strings that reads a list of JSON strings back reads the list. -/
theorem mapM_str_eq {f : Json → Flock.V String} {l l' : List String} (h : (l.map Json.str).mapM f = .ok l')
    (hf : ∀ x, f (.str x) = .ok x) : l' = l :=
  Except.ok.inj (h.symm.trans (mapM_map_str hf l))

theorem mapError_ok {ε ε' α : Type} {f : ε → ε'} {x : Except ε α} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error _ => cases h
  | ok _ => cases h; rfl

/-- **v2's coin trees are the log's**: where v2 reads the challenger's trees from its evidence (`treesOf`) and the
session's log is the lines `l`, they are what `coinTreesOf` reads from `l`. -/
theorem trees_of {p : Spot.OneStageZk.Params} {e : Spot.OneStageZk.Evidence}
    {ts : Array LoggedTree} {ls : Array Json} {l : List String}
    (h : Spot.OneStageZk.treesOf p.base e.base = .ok ts)
    (hlog : _root_.Spot.getD (_root_.Spot.getD e.base.coins "session") "log" = .arr ls)
    (hls : ls.toList = l.map Json.str) : coinTreesOf l = .ok ts := by
  unfold Spot.OneStageZk.treesOf at h
  split at h
  case h_2 => cases h
  have ht := mapError_ok h
  unfold trustedTreesOf at ht
  simp only [bind, Except.bind] at ht
  split at ht
  · split at ht
    · cases ht
    · simp only [hlog] at ht
      split at ht
      · cases ht
      · rename_i l' hl'
        rw [hls] at hl'
        obtain rfl := mapM_str_eq hl' fun _ => rfl
        exact ht
  · cases ht

theorem obind_some {α β : Type} {x : Option α} {g : α → Option β} {b : β} (h : (x >>= g) = some b) :
    ∃ a, x = some a ∧ g a = some b := by
  cases hx : x with
  | none => rw [hx] at h; cases h
  | some a => rw [hx] at h; exact ⟨a, rfl, h⟩

/-- Each value an `Option` `mapM` returns is its function's at an element of the list. -/
theorem mapM_mem {β γ : Type} {f : β → Option γ} :
    ∀ {l : List β} {l' : List γ}, l.mapM f = some l' → ∀ y ∈ l', ∃ x ∈ l, f x = some y
  | [], _, h, _, hy => by
    rw [List.mapM_nil] at h
    cases h
    cases hy
  | x :: l, _, h, y, hy => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := obind_some h
    obtain ⟨bs, hbs, h⟩ := obind_some h
    cases h
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨x, List.mem_cons.2 (.inl rfl), hb⟩
    · obtain ⟨x', hx', e⟩ := mapM_mem hbs y hy
      exact ⟨x', List.mem_cons.2 (.inr hx'), e⟩

/-- **The records v2's check read pass the trees it held them to**: each record `recordsOn` reads from the blobs sent
is one `sessionTreeReason` accepts, at some tables, against trees `treesHeld` accepts the evidence's sessions at. -/
theorem records_of {s : Sent} {p : Spot.OneStageZk.Own} {e : Spot.OneStageZk.Evidence}
    {ts : Array LoggedTree} {recs : List Json}
    (h : Spot.OneStageZk.treesHeld p ts e (_root_.Core.Blobs.ofArray s.blobs) = .ok ())
    (hr : Verity.Sampled.recordsOn (_root_.Core.Blobs.ofArray s.blobs) e = some recs) :
    ∀ rec ∈ recs, ∃ tabs, sessionTreeReason ts tabs rec = none := by
  intro rec hrec
  obtain ⟨r, hr', hf⟩ := mapM_mem hr rec hrec
  unfold Verity.Sampled.recordOn at hf
  obtain ⟨b, hb, hf⟩ := obind_some hf
  obtain ⟨str, hs, hp⟩ := obind_some hf
  exact Verity.Sampled.Trees.records_held h (Array.mem_def.2 hr') hb hs (Option.some.inj hp)

/-- **An accepted read is `readLog` over a run of the device**: at a composite that reads sampled proofs at v2, on a
run accepted as a `same` composition whose sampled component accepts its own case, where v2's check reads a log
(`Verity.Sampled.coinsRead`), that log is what `readLog` reads from the lines of a run of the device under the
challenger's key, a run the device's own ledger opens withheld on the sample's window, whose release log reads as its
coin commitment's (`releasedRootsOf`), with that run's released coins as the sample's, on the records the evidence
names, each of which passes the coin trees the run's log holds (`coinTreesOf`, `sessionTreeReason`); or one of the
run's v2 records under the parameters' key carries a break. -/
theorem accepted_run {k : Verity.Registration.Composite} (hk : ReadsV2 k) (dev : History) {s : Sent}
    (hs : SameAccepts k s) (ha : ComponentAccepts k .sampled s) {l : List Release}
    (hl : Verity.Sampled.coinsRead s = some l) :
    (∃ x ∈ records s, keyOf s.params = some x.1.key ∧ Break dev x.1.key x.1.label x.2) ∨
      ∃ (d : Draws) (e : Spot.OneStageZk.Evidence) (recs : List Json), ∃ r ∈ dev.runs d.key.pub,
        OpensOn dev d.key.pub d.window r.id ∧
        (∃ c roots, r.commitment = some c ∧ releasedRootsOf c r.log = .ok roots) ∧
        Verity.Sampled.recordsOn (_root_.Core.Blobs.ofArray s.blobs) e = some recs ∧
        Verity.Sampled.readLog d.label d.after recs (r.released d.label d.after) r.log = some l ∧
        ∃ ts, coinTreesOf r.log = .ok ts ∧ ∀ rec ∈ recs, ∃ tabs, sessionTreeReason ts tabs rec = none := by
  unfold Verity.Sampled.coinsRead at hl
  cases hcase : caseOf s with
  | error _ => simp [hcase, Except.toOption] at hl
  | ok pce =>
    obtain ⟨p, c, e⟩ := pce
    cases hch : p.base.challenger with
    | none => simp [hcase, hch, Except.toOption] at hl
    | some kl =>
      obtain ⟨kk, label⟩ := kl
      cases hξ : Spot.OneStageZk.coinsOf p c e with
      | error _ => simp [hcase, hch, hξ, Except.toOption] at hl
      | ok ξ =>
        cases hlines : Verity.Sampled.linesOf e with
        | error _ => simp [hcase, hch, hξ, hlines, Except.toOption] at hl
        | ok lines =>
          cases hrecs : Verity.Sampled.recordsOn (_root_.Core.Blobs.ofArray s.blobs) e with
          | none => simp [hcase, hch, hξ, hlines, hrecs, Except.toOption] at hl
          | some recs =>
            simp [hcase, hch, hξ, hlines, hrecs, Except.toOption] at hl
            obtain ⟨c', e', d, ξ', hcase', hd, hrec, hkey, hξ', hdraw, ts, hts, hheld⟩ :=
              accepted_case hk (caseOf_params hcase) hs ha
            rw [hcase] at hcase'
            have hpce := Except.ok.inj hcase'
            simp only [Prod.mk.injEq] at hpce
            have e1 : c' = c := hpce.2.1.symm
            have e2 : e' = e := hpce.2.2.symm
            subst c' e'
            have e3 : ξ' = ξ := Except.ok.inj (hξ'.symm.trans hξ)
            subst ξ'
            obtain ⟨hdl, hda⟩ : d.label = label ∧ d.after = c.digest := by
              unfold oneStageZkSource oneStageSource at hd
              rw [hch] at hd
              cases hw : p.base.verifier.windowRoot with
              | none => rw [hw] at hd; cases hd
              | some w => rw [hw] at hd; cases hd; exact ⟨rfl, rfl⟩
            rcases Drawn.drawn_log (dev := dev) hdraw with ⟨r, hr, ho, hξr, hcr, ls, hlog, hls⟩ | b
            · have hlr : lines = r.log := Except.ok.inj (hlines.symm.trans (linesOf_eq hlog hls))
              refine .inr ⟨d, e, recs, r, hr, ho, hcr, hrecs, ?_, ts, trees_of hts hlog hls, records_of hheld hrecs⟩
              rw [← hξr, ← hlr, hdl, hda]
              exact hl
            · exact .inl ⟨(d, e.base.coins), hrec, hkey, b⟩

/-! ## The read finder and the forger -/

/-- **One LMS verification's SHA-512 calls at the parameters `q`**, at most: the message's digest, each of the `P`
chains' `2^W − 1` steps, the one-time key, the leaf and the `H` path nodes (`Lms.rootOf`). -/
def lmsCalls (q : Lms.Params) : ℕ := q.P * (2 ^ q.W - 1) + q.H + 3

/-- **`findBreak`'s SHA-512 calls on the records of `s`** at the device's history `dev`: on each, one per string it
hashes (those the reader hashes and those the device hashes under the record's key) and `lmsCalls` per signature it
verifies. -/
def breakCalls (dev : History) (s : Sent) : ℕ :=
  ((records s).map fun x => (signatures x.1.key x.2).length * lmsCalls x.1.key.params +
    (readHashed x.1.key x.1.label x.2).length + (deviceHashed dev x.1.key.pub).length).sum

/-! ## v2's records are read at the deployed key, within `LIMITS` -/

/-- The sampled component's one-stage parameters are `OneStageZk.paramsOfJson`'s, at what it reads past the form. -/
theorem dispatch_params {j : Json} {p : Spot.OneStageZk.Params}
    (h : VStar.Recursive.Sampled.paramsOfJson j = .ok (.inl p)) :
    ∃ rest, Spot.OneStageZk.paramsOfJson rest = .ok p := by
  unfold VStar.Recursive.Sampled.paramsOfJson at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨_, by assumption⟩)

/-- The sampled component's one-stage evidence is `OneStageZk.evidenceOf`'s, at the bytes it reads past the form. -/
theorem dispatch_evidence {b : ByteArray} {e : Spot.OneStageZk.Evidence}
    (h : VStar.Recursive.Sampled.evidenceOf b = .ok (.inl e)) :
    ∃ b', Spot.OneStageZk.evidenceOf b' = .ok e := by
  unfold VStar.Recursive.Sampled.evidenceOf at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨_, by assumption⟩)

/-- **The challenger key the sampled component's one-stage parameters name is at `Lms.deployed`**
(`Limits.paramsOfJson_deployed`). -/
theorem dispatch_deployed {a : ByteArray} {p : Spot.OneStageZk.Params}
    (h : (Core.canonicalJson a >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p)) :
    ∃ key label, p.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  cases hj : Core.canonicalJson a with
  | error _ => rw [hj] at h; cases h
  | ok j =>
    rw [hj, ok_bind] at h
    obtain ⟨rest, hr⟩ := dispatch_params h
    exact Limits.paramsOfJson_deployed hr

theorem keyAt_deployed {n : Core.Composition.Named} {key : ChallengerKey} (h : keyAt n = some key) :
    key.params = Lms.deployed := by
  unfold keyAt at h
  split at h
  · split at h
    · obtain ⟨key', label, hch, hk⟩ := dispatch_deployed (by assumption)
      rw [hch] at h
      cases h
      exact hk
    · cases h
  · cases h

/-- **The parameters' challenger key is at the deployed LMS parameters** (`lmsVerify`'s): v2 refuses a key at any
other (`Limits.paramsOfJson_deployed`). -/
theorem keyOf_deployed (θ : ByteArray) : ∀ key ∈ keyOf θ, key.params = Lms.deployed := by
  intro key hkey
  have hkey : keyOf θ = some key := hkey
  unfold keyOf at hkey
  split at hkey
  · obtain ⟨n, -, hn⟩ := List.exists_of_findSome?_eq_some hkey
    exact keyAt_deployed hn
  · cases hkey

/-- **At most `findBreak`'s SHA-512 calls on one v2 record**, apart from the strings the device hashes:
`Limits.signaturesAt LIMITS` signatures at `lmsCalls Lms.deployed` each and `Limits.readsAt LIMITS` strings
(`Limits.record_calls_le`); its key is at `Lms.deployed`. -/
def recordCalls : ℕ := Limits.signaturesAt LIMITS * lmsCalls Lms.deployed + Limits.readsAt LIMITS

/-- **Each v2 record is read at the deployed key, within `recordCalls`**: v2's parameters name its key at
`Lms.deployed` and v2's evidence holds its coins within `LIMITS` (`Limits.evidenceOf_within`). -/
theorem record?_le {n : Core.Composition.Named} {x : Core.Composition.Case} {r : Draws × Json}
    (h : record? n x = some r) :
    r.1.key.params = Lms.deployed ∧
      (signatures r.1.key r.2).length * lmsCalls r.1.key.params + (readHashed r.1.key r.1.label r.2).length ≤
        recordCalls := by
  unfold record? at h
  split at h
  · obtain ⟨d, hd, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨key, label, hch, hk⟩ := dispatch_deployed (by assumption)
    have hkd := key_of_source hd
    rw [hch] at hkd
    have hkd' : key = d.key := Option.some.inj hkd
    have hdp : d.key.params = Lms.deployed := hkd' ▸ hk
    obtain ⟨b, hb⟩ := dispatch_evidence (by assumption)
    have hw := Limits.evidenceOf_within hb
    refine ⟨hdp, ?_⟩
    dsimp only
    rw [hdp]
    exact Nat.add_le_add (Nat.mul_le_mul_right _ (Limits.signatures_le _ _ _ hw))
      (Limits.readHashed_le _ _ _ _ hw)
  · cases h

theorem records_le {s : Sent} {x : Draws × Json} (hx : x ∈ records s) :
    x.1.key.params = Lms.deployed ∧
      (signatures x.1.key x.2).length * lmsCalls x.1.key.params + (readHashed x.1.key x.1.label x.2).length ≤
        recordCalls := by
  unfold records at hx
  split at hx
  · obtain ⟨y, -, hy⟩ := List.mem_filterMap.1 hx
    exact record?_le hy
  · simp at hx

/-- **`findBreak`'s SHA-512 calls on the records of `s` are at most `recordCalls` per record**, beside the strings
the device hashes under each record's key (`Properties.Challenger.deviceHashed`). -/
theorem breakCalls_le (dev : History) (s : Sent) :
    breakCalls dev s ≤ (records s).length * recordCalls +
      ((records s).map fun x => (deviceHashed dev x.1.key.pub).length).sum := by
  unfold breakCalls
  have hle : ∀ x ∈ records s, (signatures x.1.key x.2).length * lmsCalls x.1.key.params +
      (readHashed x.1.key x.1.label x.2).length ≤ recordCalls := fun x hx => (records_le hx).2
  generalize records s = l at hle ⊢
  induction l with
  | nil => simp
  | cons x l ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.succ_mul]
    have h1 := hle x (List.mem_cons.2 (.inl rfl))
    have h2 := ih fun y hy => hle y (List.mem_cons_of_mem _ hy)
    omega

/-- A finding's collision, as two messages. -/
def collisionOf : Finding → Option (Claims.Msg × Claims.Msg)
  | .collision x y => some (x.data.toList, y.data.toList)
  | .forgery _ _ => none

/-- A finding's forgery, as a message and a signature's bytes. -/
def forgeryOf : Finding → Option (Hash.Msg × Hash.Msg)
  | .forgery m sig => some (m.data.toList, sig.data.toList)
  | .collision _ _ => none

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The read finder at a strategy**, at the device's history at each play `dev`: on a uniform tape of the play, the
first collision `findBreak` finds in one of the play's records, else the first round opened whose coins v2's read,
without the sample's coins, shows other than the device released (`Lines.opensRead`, `Lines.openCollision`). The read
hashes each blob sent once (`Core.Blobs.ofArray`). -/
noncomputable def readFinder (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Claims.Finder where
  S := Verity.Game.Tape (V.game θ F c) σ
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    (((records o.sent).filterMap fun x => findBreak (dev o) x.1.key x.1.label x.2).findSome? collisionOf).or
      (((Lines.opensRead o.sent).bind fun l => Lines.openCollision l o.log).map fun uv =>
        (uv.1.data.toList, uv.2.data.toList))
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent + o.sent.blobs.size

theorem readFinder_out (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ) :
    (readFinder V dev θ F c σ).out t =
      ((((records (Verity.Game.Tape.out _ σ t).sent).filterMap fun x =>
          findBreak (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2).findSome? collisionOf).or
        (((Lines.opensRead (Verity.Game.Tape.out _ σ t).sent).bind fun l =>
          Lines.openCollision l (Verity.Game.Tape.out _ σ t).log).map fun uv =>
            (uv.1.data.toList, uv.2.data.toList))) :=
  rfl

open Classical in
/-- **The forger at a strategy**, against the parameters' challenger key (`keyOf`), at the device's history at each
play `dev`: on a uniform tape of the play, the first forgery `findBreak` finds in one of the play's records under that
key. The play releases the device's signatures at its close, so on each tape the device signed the messages the play's
signatures are on, which the forger saw. -/
noncomputable def forger (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Hash.Assumptions.LmsForger where
  S := Verity.Game.Tape (V.game θ F c) σ
  key := ((keyOf θ).map fun k => k.pub.data.toList).getD []
  signed t := (Verity.Game.Tape.out _ σ t).signed.map Prod.fst
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    ((records o.sent).filterMap fun x =>
      if keyOf θ = some x.1.key then findBreak (dev o) x.1.key x.1.label x.2 else none).findSome? forgeryOf
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent

/-- **The leg's constructions with this read finder and forger**, at the device's history at each play `dev`: `K`'s
projection and two-continuation finders, and `readFinder` and `forger`. -/
noncomputable def withBinding {V : Vocab d α} (K : Constructions V) (dev : Exchange → History) : Constructions V :=
  { K with readFinder := readFinder V dev, forger := forger V dev }

/-! ## What the finders output -/

theorem bytes_ofList (b : ByteArray) : ({ data := b.data.toList.toArray } : ByteArray) = b := by
  cases b
  simp

/-- A collision of the compiled SHA-512 is one of `sha512Msg`. -/
theorem collision_of_valid {x y : ByteArray} (h : x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) :
    Claims.Collision Verity.sha512Msg x.data.toList y.data.toList := by
  refine ⟨fun e => h.1 (ByteArray.ext (Array.ext' e)), ?_⟩
  unfold Verity.sha512Msg
  rw [bytes_ofList, bytes_ofList, h.2]

/-- **A signature that verifies under a key at the deployed LMS parameters verifies under `lmsVerify`.** -/
theorem lmsVerify_of_verifies {key : ChallengerKey} (hk : key.params = Lms.deployed) {m sig : ByteArray}
    (h : LmsVerifies key m sig) : Verity.lmsVerify key.pub.data.toList m.data.toList sig.data.toList = true := by
  unfold LmsVerifies lmsVerifies at h
  unfold Verity.lmsVerify
  rw [hk] at h
  rw [bytes_ofList, bytes_ofList, bytes_ofList]
  cases hd : Lms.Signature.decode Lms.deployed sig with
  | error e => simp [hd] at h
  | ok s =>
    cases hv : Lms.verify Lms.deployed key.pub m s with
    | error e => simp [hd, hv] at h
    | ok b =>
      simp only [hd, hv] at h
      simp [hv, bind, Except.bind, h]

/-- The first collision among findings that are each valid where they are collisions is a collision of
`sha512Msg`. -/
theorem findSome_collisionOf_valid {l : List Finding}
    (hl : ∀ x y, Finding.collision x y ∈ l → x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) {u v : Claims.Msg}
    (h : l.findSome? collisionOf = some (u, v)) : Claims.Collision Verity.sha512Msg u v := by
  obtain ⟨w, hw, hc⟩ := List.exists_of_findSome?_eq_some h
  cases w with
  | forgery _ _ => simp [collisionOf] at hc
  | collision a b =>
    simp only [collisionOf, Option.some.injEq, Prod.mk.injEq] at hc
    obtain ⟨rfl, rfl⟩ := hc
    exact collision_of_valid (hl a b hw)

/-- The first collision among findings that are each valid where they are collisions is a collision of
`sha512Msg`, and there is one where a finding is a collision. -/
theorem findSome_collisionOf {l : List Finding}
    (hl : ∀ x y, Finding.collision x y ∈ l → x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) {x y : ByteArray}
    (hxy : Finding.collision x y ∈ l) :
    ∃ u v, l.findSome? collisionOf = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  cases hf : l.findSome? collisionOf with
  | none => exact absurd (List.findSome?_eq_none_iff.1 hf _ hxy) (by simp [collisionOf])
  | some uv =>
    obtain ⟨u, v⟩ := uv
    exact ⟨u, v, rfl, findSome_collisionOf_valid hl hf⟩

/-- `findSome_collisionOf` ahead of any other output: the first collision among valid findings comes first. -/
theorem findSome_collisionOf_or {l : List Finding} {o : Option (Claims.Msg × Claims.Msg)}
    (hl : ∀ x y, Finding.collision x y ∈ l → x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) {x y : ByteArray}
    (hxy : Finding.collision x y ∈ l) :
    ∃ u v, (l.findSome? collisionOf).or o = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  obtain ⟨u, v, hf, hc⟩ := findSome_collisionOf hl hxy
  exact ⟨u, v, by rw [hf, Option.some_or], hc⟩

/-- The first forgery among findings that each verify under `key` where they are forgeries verifies under
`lmsVerify`, at a key at the deployed LMS parameters, over the message of one of those forgeries, and there is one
where a finding is a forgery. -/
theorem findSome_forgeryOf {key : ChallengerKey} (hk : key.params = Lms.deployed) {l : List Finding}
    (hl : ∀ m sig, Finding.forgery m sig ∈ l → LmsVerifies key m sig) {m sig : ByteArray}
    (hms : Finding.forgery m sig ∈ l) :
    ∃ u v, l.findSome? forgeryOf = some (u, v) ∧ Verity.lmsVerify key.pub.data.toList u v = true ∧
      ∃ m' sig', Finding.forgery m' sig' ∈ l ∧ u = m'.data.toList := by
  cases hf : l.findSome? forgeryOf with
  | none => exact absurd (List.findSome?_eq_none_iff.1 hf _ hms) (by simp [forgeryOf])
  | some uv =>
    obtain ⟨w, hw, hc⟩ := List.exists_of_findSome?_eq_some hf
    cases w with
    | collision _ _ => simp [forgeryOf] at hc
    | forgery a b =>
      simp only [forgeryOf, Option.some.injEq] at hc
      subst hc
      exact ⟨_, _, rfl, lmsVerify_of_verifies hk (hl a b hw), a, b, hw, rfl⟩

/-! ## The reduction -/

/-- A run of the play after `θ`, `F` and `c` ends with them. -/
theorem play_reaches {fs : List (String × ℕ)} {D : Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {o : Exchange} (h : Verity.Game.Reaches (play fs D θ F c) o) :
    o.params = θ ∧ o.files = F ∧ o.commitment = c := by
  obtain ⟨_, ha⟩ := h
  obtain ⟨log, -, b, ρ, e, he⟩ := Verity.Game.reaches_bind ha
  cases he
  exact ⟨rfl, rfl, rfl⟩

/-- A run of the play under the signer `D` ends with `D`'s signatures on what it fixed: its request, its log, the
bytes its close left to the device and the device's sample. -/
theorem play_signed {fs : List (String × ℕ)} {D : Signer} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {o : Exchange} (h : Verity.Game.Reaches (play fs D θ F c) o) :
    o.signed = D.signatures o.request o.log o.close o.drawn := by
  obtain ⟨_, ha⟩ := h
  obtain ⟨log, -, b, ρ, e, he⟩ := Verity.Game.reaches_bind ha
  cases he
  rfl

/-- **The forger has at most `Q` of the device's messages signed on each of its tapes** (`Vocab.signs_le`). -/
theorem forger_signed_le (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ) :
    ((forger V dev θ F c σ).signed t).length ≤ V.signatures θ := by
  show ((Verity.Game.Tape.out _ σ t).signed.map Prod.fst).length ≤ _
  rw [List.length_map, play_signed (Verity.Game.Tape.reaches_out _ σ t), Signer.signatures, List.length_mapIdx]
  exact V.signs_le θ _ _ _ _

/-- **The device's history at each play signed what the play's signatures are on**: on every run of the period's
game after `θ`, `F` and `c`, each message the play's signer signed is one the device's history at the run signed
under the parameters' key (`Properties.Challenger.Signed`). -/
def SignsPlay (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) :
    Prop :=
  ∀ o, Verity.Game.Reaches (V.game θ F c) o → ∀ key, keyOf θ = some key →
    ∀ m ∈ o.signed, Signed (dev o) key.pub ⟨m.1.toArray⟩

/-- **The binding at the parameters `θ`, pointwise**: on every run of the period's game after `θ` whose sampled
verdict at `k` accepts and whose log, as `reads` reads it, isn't the device's, one of its v2 records under the
parameters' key carries a break at the device's history at the run (`Properties.Challenger.Break`), or v2's read
without the sample's coins shows a round opened's coins other than the device's, of one SHA-512 with them
(`Lines.openCollision`). -/
def Binds (V : Vocab d α) (k : Verity.Registration.Composite) (reads : Sent → Option (List Read))
    (dev : Exchange → History) (θ : ByteArray) : Prop :=
  ∀ o : Exchange, o.params = θ → V.Verdict k .sampled o → ¬ Logged reads o →
    (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
      ∃ u v, ((Lines.opensRead o.sent).bind fun l => Lines.openCollision l o.log) = some (u, v) ∧ u ≠ v ∧
        Flock.Sha512.hash u = Flock.Sha512.hash v

/-- **A run with a break under the parameters' key in one of its records is one where the read finder or the
forger succeeds**, the parameters' key being at the deployed LMS parameters (`keyOf_deployed`), where the device's
history at each play signed what the play's signatures are on (`SignsPlay`). -/
theorem finders_of_break (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ) (hs : SignsPlay V dev θ F c) {x : Draws × Json}
    (hx : x ∈ records (Verity.Game.Tape.out _ σ t).sent) (hk : keyOf θ = some x.1.key)
    (hb : Break (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2) :
    (∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
      (forger V dev θ F c σ).Forges Verity.lmsVerify t := by
  classical
  obtain ⟨w, hw, -⟩ := hb
  cases w with
  | collision a b =>
    left
    refine findSome_collisionOf_or (fun a' b' h => ?_) (List.mem_filterMap.2 ⟨x, hx, hw⟩)
    obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 h
    exact findBreak_valid hx'
  | forgery m sig =>
    right
    have hd : x.1.key.params = Lms.deployed := keyOf_deployed θ _ hk
    let o := Verity.Game.Tape.out _ σ t
    let l := (records o.sent).filterMap fun x' =>
      if keyOf θ = some x'.1.key then findBreak (dev o) x'.1.key x'.1.label x'.2 else none
    have hfx : (if keyOf θ = some x.1.key then findBreak (dev o) x.1.key x.1.label x.2 else none) =
        some (Finding.forgery m sig) := by
      rw [ite_eq_left hk]
      exact hw
    have hmem : Finding.forgery m sig ∈ l := List.mem_filterMap.2 ⟨x, hx, hfx⟩
    have hl : ∀ m' sig', Finding.forgery m' sig' ∈ l →
        LmsVerifies x.1.key m' sig' ∧ ¬ Signed (dev o) x.1.key.pub m' := by
      intro m' sig' h
      obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 h
      have hx'' : (if keyOf θ = some x'.1.key then findBreak (dev o) x'.1.key x'.1.label x'.2 else none) =
          some (Finding.forgery m' sig') := hx'
      by_cases hk' : keyOf θ = some x'.1.key
      · rw [ite_eq_left hk'] at hx''
        have e : x'.1.key = x.1.key := Option.some.inj (hk'.symm.trans hk)
        have hv : LmsVerifies x'.1.key m' sig' ∧ ¬ Signed (dev o) x'.1.key.pub m' := findBreak_valid hx''
        exact e ▸ hv
      · rw [ite_eq_right hk'] at hx''
        cases hx''
    obtain ⟨_, v, hf, hv, m', sig', hm', rfl⟩ := findSome_forgeryOf hd (fun m' sig' h => (hl m' sig' h).1) hmem
    refine ⟨_, v, hf, fun hu => ?_, ?_⟩
    · have hu' : m'.data.toList ∈ o.signed.map Prod.fst := hu
      obtain ⟨p, hp, hpu⟩ := List.mem_map.1 hu'
      have hsg := hs o (Verity.Game.Tape.reaches_out _ σ t) x.1.key hk p hp
      rw [hpu, bytes_ofList] at hsg
      exact (hl m' sig' hm').2 hsg
    · show Verity.lmsVerify (((keyOf θ).map fun k => k.pub.data.toList).getD []) m'.data.toList v = true
      rw [hk]
      exact hv

/-- **A run whose read without the sample's coins shows a round opened's coins other than the device's, of one SHA-512
with them, is one where the read finder succeeds.** -/
theorem finder_of_open (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (t : Verity.Game.Tape (V.game θ F c) σ)
    {u v : ByteArray} (h : ((Lines.opensRead (Verity.Game.Tape.out _ σ t).sent).bind fun l =>
      Lines.openCollision l (Verity.Game.Tape.out _ σ t).log) = some (u, v))
    (huv : u ≠ v ∧ Flock.Sha512.hash u = Flock.Sha512.hash v) :
    ∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  rw [readFinder_out]
  cases hf : (((records (Verity.Game.Tape.out _ σ t).sent).filterMap fun x =>
      findBreak (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2).findSome? collisionOf) with
  | some uv =>
    obtain ⟨u', v'⟩ := uv
    refine ⟨u', v', rfl, findSome_collisionOf_valid (fun a b hm => ?_) hf⟩
    obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 hm
    exact findBreak_valid hx'
  | none =>
    refine ⟨u.data.toList, v.data.toList, ?_, collision_of_valid huv⟩
    simp only [Option.none_or, h, Option.map_some]

/-- **(a): an accepted play whose log the check read isn't the device's is at most as likely as the read finder's
collision plus the forger's forgery**, where each accepted run that isn't `Logged` carries a break under the
parameters' key in one of its records, or shows a round opened's coins other than the device's of one SHA-512 with
them (`Binds`), and the device's history at each play signed what the play's signatures are on (`SignsPlay`). That
key is at the deployed LMS parameters (`keyOf_deployed`), and each finder spends on a play the strategy's calls and at
most `recordCalls` per record beside the strings the device hashes (`breakCalls_le`), the read finder one more per blob
sent. -/
theorem logged_le (V : Vocab d α) (k : Verity.Registration.Composite) (reads : Sent → Option (List Read))
    (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (hb : Binds V k reads dev θ) (hs : SignsPlay V dev θ F c) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Logged reads o) (V.game θ F c) σ ≤
      ENNReal.ofReal ((readFinder V dev θ F c σ).Pr Verity.sha512Msg) +
        ENNReal.ofReal ((forger V dev θ F c σ).Pr Verity.lmsVerify) := by
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) ∧
          ¬ Logged reads (Verity.Game.Tape.out _ σ t))
      ≤ Verity.Game.prCoin (fun t =>
          (∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
            (forger V dev θ F c σ).Forges Verity.lmsVerify t) := by
        refine Verity.Game.prCoin_mono fun t ⟨hv, hl⟩ => ?_
        have hθ := (play_reaches (Verity.Game.Tape.reaches_out _ σ t)).1
        rcases hb _ hθ hv hl with ⟨x, hx, hk, hbr⟩ | ⟨u, v, huv, hne, hh⟩
        · exact finders_of_break V dev θ F c σ t hs hx hk hbr
        · exact .inl (finder_of_open V dev θ F c σ t huv ⟨hne, hh⟩)
    _ ≤ _ := Verity.Game.prCoin_or_le _ _
    _ = _ := congrArg₂ (· + ·) (Verity.Game.prCoin_eq_ofReal_pr _) (Verity.Game.prCoin_eq_ofReal_pr _)

/-- **(a) at the leg's constructions with this read finder and forger** (`withBinding`). -/
theorem logged_le_withBinding {V : Vocab d α} (K : Constructions V) (k : Verity.Registration.Composite)
    (reads : Sent → Option (List Read)) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hb : Binds V k reads dev θ)
    (hs : SignsPlay V dev θ F c) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Logged reads o) (V.game θ F c) σ ≤
      ENNReal.ofReal (((withBinding K dev).readFinder θ F c σ).Pr Verity.sha512Msg) +
        ENNReal.ofReal (((withBinding K dev).forger θ F c σ).Pr Verity.lmsVerify) :=
  logged_le V k reads dev θ F c σ hb hs

/-! ## The binding where the device printed its log -/

/-- A read of the device's log but for openings' coins of one SHA-512 (`Lines.ReadOf`), whose coins where it read any
are the device's, is the device's log as v2 reads it. -/
theorem logged_of_reads {l log : List Release} (h : List.Forall₂ Lines.ReadOf l log)
    (hc : List.Forall₂ (fun y x => y.coins ≠ [] → y.coins = x.coins) l log) :
    List.Forall₂ Read.Of (l.map Read.ofV2) log := by
  revert hc
  induction h with
  | nil => intro _; exact .nil
  | @cons y x ys xs hyx _ ih =>
    intro hc
    cases hc with
    | cons hc hcs =>
      refine .cons ⟨hyx.1, hyx.2.1, fun cs hcs' => ?_⟩ (ih hcs)
      change cs ∈ (if y.coins.isEmpty then none else some y.coins) at hcs'
      split at hcs'
      · cases Option.mem_def.1 hcs'
      · rename_i hne
        obtain rfl := Option.some.inj (Option.mem_def.1 hcs')
        exact hc fun e => hne (by simp [e])

/-- **`Binds` at v2's read where the device printed each play's log** (`Lines.PrintsLog`): an accepted run whose log
v2 read isn't the device's carries a break under the parameters' key in one of its records (`accepted_run`), or v2
read its log from the lines of a run that printed the play's log (`Lines.read_run`), so the first round opened whose
coins v2's read without the sample's coins shows other than the device's is a SHA-512 collision (`Lines.opens_read`,
`Lines.open_collision`). -/
theorem binds_of_prints (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History)
    (θ : ByteArray) (hP : ∀ o : Exchange, o.params = θ → Lines.PrintsLog (dev o) o.log) :
    Binds V k readsV2 dev θ := by
  intro o hθ hv hl
  obtain ⟨-, hs, ha⟩ := hv
  cases hc : Verity.Sampled.coinsRead o.sent with
  | none => exact (hl fun log hlog => by simp [readsV2, hc] at hlog).elim
  | some l =>
    rcases accepted_run (composite_readsV2 k) (dev o) hs ha hc with ⟨x, hx, hk, hb⟩ |
      ⟨dr, e, recs, r, hr, ho, hcr, -, h, ts, hts, hrecs⟩
    · exact .inl ⟨x, hx, by rw [← hθ]; exact hk, hb⟩
    · have hread := Lines.read_run (hP o hθ) hr ho hcr hts hrecs h
      obtain ⟨out', hout, hag⟩ := Lines.opens_read hc
      have hcoins : ¬ List.Forall₂ (fun y x => y.coins ≠ [] → y.coins = x.coins) l o.log := fun hc' =>
        hl fun log hlog => by
          simp only [readsV2, hc, Option.map_some, Option.mem_def, Option.some.injEq] at hlog
          subst hlog
          exact logged_of_reads hread hc'
      obtain ⟨u, v, hu, hne, hh⟩ := Lines.open_collision hread hag hcoins
      exact .inr ⟨u, v, by rw [hout]; exact hu, hne, hh⟩

end Verity.Sampled.Binding
